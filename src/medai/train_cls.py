"""DVC stage `train_cls`: multi-label chest X-ray classifier (transfer learning)."""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose, RandAffine, RandAdjustContrast, RandGaussianNoise
from sklearn.metrics import roc_auc_score

from .config import MODELS, PROCESSED, params, path
from .models import build_classifier, to_input

log = logging.getLogger("medai.train_cls")


def load_split(split: str):
    idx = pd.read_csv(PROCESSED / "cxr_index.csv")
    imgs = np.load(PROCESSED / "cxr_images.npy", mmap_mode="r")
    sel = np.flatnonzero(idx.split == split)
    return idx.iloc[sel].reset_index(drop=True), imgs, sel


def augment():
    # No horizontal flips: they would move the heart to the wrong side (anatomically invalid for CXR).
    return Compose([
        RandAffine(prob=0.8, rotate_range=0.12, scale_range=0.1, translate_range=(8, 8), padding_mode="border"),
        RandAdjustContrast(prob=0.5, gamma=(0.8, 1.25)),
        RandGaussianNoise(prob=0.3, std=0.02),
    ])


@torch.no_grad()
def predict(model, imgs, sel, bs=64) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(sel), bs):
        x = to_input(np.asarray(imgs[sel[i:i + bs]], dtype=np.float32) / 255)
        out.append(torch.sigmoid(model(x)).numpy())
    return np.concatenate(out)


def macro_auc(y, p) -> float:
    aucs = [roc_auc_score(y[:, k], p[:, k]) for k in range(y.shape[1]) if 0 < y[:, k].sum() < len(y)]
    return float(np.mean(aucs))


def run() -> None:
    p = params()
    cfg = p["classification"]
    torch.manual_seed(p["data"]["seed"])
    torch.set_num_threads(max(1, torch.get_num_threads()))
    findings = cfg["findings"]
    tr, imgs, tr_sel = load_split("train")
    va, _, va_sel = load_split("val")
    ytr, yva = tr[findings].to_numpy(np.float32), va[findings].to_numpy(np.float32)
    model, weights_used = build_classifier(cfg["arch"], len(findings), cfg["pretrained"])
    pos = ytr.sum(0)
    pos_weight = torch.tensor(np.clip(np.sqrt((len(ytr) - pos) / np.maximum(pos, 1)), 1, 10), dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)        # class imbalance
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * int(np.ceil(len(tr_sel) / cfg["batch_size"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.15)
    aug = augment()
    rng = np.random.default_rng(p["data"]["seed"])
    history, best, t0 = [], -1.0, time.time()
    MODELS.mkdir(exist_ok=True)
    for ep in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(len(tr_sel))
        losses = []
        for b in range(0, len(order), cfg["batch_size"]):
            ids = order[b:b + cfg["batch_size"]]
            batch = np.asarray(imgs[tr_sel[ids]], dtype=np.float32) / 255
            batch = np.stack([aug(x[None])[0] for x in batch])
            x, y = to_input(batch), torch.from_numpy(ytr[ids])
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        pva = predict(model, imgs, va_sel)
        auc = macro_auc(yva, pva)
        history.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)), "val_macro_auroc": auc,
                        "minutes": round((time.time() - t0) / 60, 1)})
        log.info("epoch %d/%d  loss %.4f  val macro-AUROC %.4f  (%.1f min)", ep + 1, cfg["epochs"],
                 np.mean(losses), auc, (time.time() - t0) / 60)
        if auc > best:
            best = auc
            torch.save({"state_dict": model.state_dict(), "arch": cfg["arch"], "findings": findings,
                        "image_size": cfg["image_size"], "weights_init": weights_used}, MODELS / "cxr_classifier.pt")
    # per-finding decision thresholds chosen on VALIDATION (Youden J), never on test
    ckpt = torch.load(MODELS / "cxr_classifier.pt", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    pva = predict(model, imgs, va_sel)
    from sklearn.metrics import roc_curve
    thr = {}
    for k, f in enumerate(findings):
        if 0 < yva[:, k].sum() < len(yva):
            fpr, tpr, t = roc_curve(yva[:, k], pva[:, k])
            thr[f] = float(t[np.argmax(tpr - fpr)])
        else:
            thr[f] = 0.5
    ckpt["thresholds"] = thr
    torch.save(ckpt, MODELS / "cxr_classifier.pt")
    path("metrics").mkdir(exist_ok=True)
    json.dump({"history": history, "best_val_macro_auroc": best, "weights_init": weights_used,
               "train_images": int(len(tr_sel)), "val_images": int(len(va_sel)),
               "pos_weight": dict(zip(findings, pos_weight.round(decimals=2).tolist()))},
              open(path("metrics/train_cls.json"), "w"), indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
