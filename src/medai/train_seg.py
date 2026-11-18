"""DVC stage `train_seg`: 3D U-Net spleen segmentation with MONAI (MSD Task09 recipe)."""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import (AsDiscrete, Compose, CropForegroundd, EnsureChannelFirstd, LoadImaged, Orientationd,
                              RandCropByPosNegLabeld, RandFlipd, RandRotate90d, RandShiftIntensityd,
                              ScaleIntensityRanged, Spacingd)

from .config import MODELS, PROCESSED, params, path
from .models import build_unet

log = logging.getLogger("medai.train_seg")


def base_transforms(cfg, keys=("image", "label")):
    lo, hi = cfg["hu_window"]
    has_label = "label" in keys
    return [
        LoadImaged(keys=list(keys)),
        EnsureChannelFirstd(keys=list(keys)),
        ScaleIntensityRanged(keys="image", a_min=lo, a_max=hi, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=list(keys), source_key="image", allow_smaller=True),
        Orientationd(keys=list(keys), axcodes="RAS"),
        Spacingd(keys=list(keys), pixdim=cfg["pixdim"], mode=("bilinear", "nearest") if has_label else "bilinear"),
    ]


def run() -> None:
    p = params()
    cfg = p["segmentation"]
    torch.manual_seed(p["data"]["seed"])
    split = json.load(open(PROCESSED / "ct_split.json"))
    train_tf = Compose(base_transforms(cfg) + [
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=cfg["roi"], pos=1, neg=1,
                               num_samples=2, image_key="image", image_threshold=0),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.2),
        RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3, spatial_axes=(0, 1)),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ])
    val_tf = Compose(base_transforms(cfg))
    tr = DataLoader(CacheDataset(split["train"], train_tf, cache_rate=1.0, num_workers=2), batch_size=2, shuffle=True)
    va = DataLoader(CacheDataset(split["val"], val_tf, cache_rate=1.0, num_workers=2), batch_size=1)
    model = build_unet(cfg["channels"])
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    dice = DiceMetric(include_background=False, reduction="mean")
    post_pred, post_lab = AsDiscrete(argmax=True, to_onehot=2), AsDiscrete(to_onehot=2)
    it, best, history, t0 = 0, -1.0, [], time.time()
    MODELS.mkdir(exist_ok=True)
    while it < cfg["max_iterations"]:
        for batch in tr:
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(batch["image"]), batch["label"])
            loss.backward()
            opt.step()
            it += 1
            if it % cfg["val_every"] == 0 or it == cfg["max_iterations"]:
                model.eval()
                with torch.no_grad():
                    for vb in va:
                        out = sliding_window_inference(vb["image"], cfg["roi"], 4, model, overlap=0.25)
                        dice(y_pred=[post_pred(o) for o in decollate_batch(out)],
                             y=[post_lab(y) for y in decollate_batch(vb["label"])])
                d = float(dice.aggregate().item())
                dice.reset()
                history.append({"iteration": it, "loss": float(loss.item()), "val_dice": d,
                                "minutes": round((time.time() - t0) / 60, 1)})
                log.info("iter %d/%d  loss %.4f  val Dice %.4f  (%.1f min)", it, cfg["max_iterations"], loss.item(), d,
                         (time.time() - t0) / 60)
                if d > best:
                    best = d
                    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, MODELS / "spleen_unet3d.pt")
            if it >= cfg["max_iterations"]:
                break
    json.dump({"history": history, "best_val_dice": best, "train_volumes": len(split["train"]),
               "val_volumes": len(split["val"])}, open(path("metrics/train_seg.json"), "w"), indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
