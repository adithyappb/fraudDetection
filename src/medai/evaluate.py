"""DVC stage `evaluate`: test-set metrics, Grad-CAM localisation, subgroup (bias) audit, 3D Dice.

Writes metrics/*.json (shown by `dvc metrics show`) and docs/figures/*.png.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import torch
from monai.data import Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete, Compose
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import MODELS, PROCESSED, params, path
from .explain import Explainer, localisation, overlay
from .models import build_classifier, build_unet
from .train_cls import load_split, predict
from .train_seg import base_transforms

log = logging.getLogger("medai.evaluate")


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None


def load_classifier():
    ckpt = torch.load(MODELS / "cxr_classifier.pt", weights_only=False)
    model, _ = build_classifier(ckpt["arch"], len(ckpt["findings"]), "none")
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


def eval_classifier() -> dict:
    model, ckpt = load_classifier()
    findings, thr = ckpt["findings"], ckpt["thresholds"]
    te, imgs, sel = load_split("test")
    y = te[findings].to_numpy()
    prob = predict(model, imgs, sel)
    np.save(path("metrics/test_probs.npy"), prob)
    per = {}
    for k, f in enumerate(findings):
        pred = prob[:, k] >= thr[f]
        pos, neg = y[:, k] == 1, y[:, k] == 0
        per[f] = {"auroc": safe_auc(y[:, k], prob[:, k]), "ap": float(average_precision_score(y[:, k], prob[:, k])),
                  "prevalence": float(y[:, k].mean()), "threshold": thr[f],
                  "sensitivity": float(pred[pos].mean()) if pos.any() else None,
                  "specificity": float((~pred[neg]).mean()) if neg.any() else None, "positives": int(pos.sum())}
    aucs = [v["auroc"] for v in per.values() if v["auroc"] is not None]

    # ---- subgroup audit (bias): macro-AUROC + per-finding AUROC by sex, age band and view position
    te = te.assign(age_band=pd.cut(te.age, [0, 40, 60, 200], labels=["<40", "40-60", ">60"]).astype(str))
    groups = {}
    for col in ("sex", "age_band", "view"):
        for g, part in te.groupby(col):
            i = part.index.to_numpy()
            a = [safe_auc(y[i, k], prob[i, k]) for k in range(len(findings))]
            a = [v for v in a if v is not None]
            groups[f"{col}={g}"] = {"n": int(len(i)), "macro_auroc": float(np.mean(a)) if a else None}
    cardio = findings.index("Cardiomegaly")
    fp_by_view = {v: float(((prob[part.index, cardio] >= thr["Cardiomegaly"]) & (y[part.index, cardio] == 0)).sum()
                           / max((y[part.index, cardio] == 0).sum(), 1)) for v, part in te.groupby("view")}

    # ---- Grad-CAM localisation on radiologist boxes (NIH BBox_List_2017)
    boxes = pd.read_csv(PROCESSED / "cxr_boxes.csv")
    pos_in_test = {n: i for i, n in enumerate(te.image)}
    boxes = boxes[boxes.image.isin(pos_in_test)]
    exp = Explainer(model, ckpt["arch"])
    rows, examples = [], []
    for b in boxes.itertuples():
        i = pos_in_test[b.image]
        k = findings.index(b.finding)
        img = np.asarray(imgs[sel[i]], np.float32) / 255
        cam = exp(img, k)
        loc = localisation(cam, (b.x, b.y, b.w, b.h), params()["explain"]["cam_threshold"])
        rows.append({"finding": b.finding, **loc, "prob": float(prob[i, k])})
        if len(examples) < 8 and loc["hit"] and prob[i, k] >= thr[b.finding] and b.finding not in {e[1] for e in examples}:
            examples.append((b.image, b.finding, img, cam, (b.x, b.y, b.w, b.h), float(prob[i, k])))
    loc = pd.DataFrame(rows)
    loc_summary = {"boxes": int(len(loc)), "pointing_game": float(loc.hit.mean()), "mean_iou": float(loc.iou.mean()),
                   "per_finding_pointing": loc.groupby("finding").hit.mean().round(3).to_dict()}
    _figure_cam_examples(examples)
    _figure_roc(y, prob, findings)
    return {"test_images": int(len(te)), "macro_auroc": float(np.mean(aucs)), "per_finding": per,
            "subgroups": groups, "cardiomegaly_false_positive_rate_by_view": fp_by_view,
            "gradcam_localisation": loc_summary, "weights_init": ckpt.get("weights_init")}


def eval_segmentation() -> dict:
    cfg = params()["segmentation"]
    ckpt = torch.load(MODELS / "spleen_unet3d.pt", weights_only=False)
    model = build_unet(cfg["channels"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    test = json.load(open(PROCESSED / "ct_split.json"))["test"]
    ds = Dataset(test, Compose(base_transforms(cfg)))
    dice, hd = DiceMetric(include_background=False), HausdorffDistanceMetric(include_background=False, percentile=95)
    post_pred, post_lab = AsDiscrete(argmax=True, to_onehot=2), AsDiscrete(to_onehot=2)
    per = []
    with torch.no_grad():
        for d in ds:
            out = sliding_window_inference(d["image"][None], cfg["roi"], 4, model, overlap=0.25)
            p, t = [post_pred(o) for o in decollate_batch(out)], [post_lab(d["label"])]
            dv = float(dice(y_pred=p, y=t).item())
            hv = float(hd(y_pred=p, y=t).item()) * cfg["pixdim"][0]
            per.append({"volume": d["image"].meta["filename_or_obj"].split("/")[-1], "dice": dv, "hd95_mm": hv})
    return {"test_volumes": len(per), "mean_dice": float(np.mean([r["dice"] for r in per])),
            "mean_hd95_mm": float(np.mean([r["hd95_mm"] for r in per])), "per_volume": per}


def _figure_cam_examples(examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not examples:
        return
    n = len(examples)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 7), facecolor="#fcfcfb")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, f, img, cam, (x, y, w, h), pr) in zip(axes.ravel(), examples):
        ax.imshow(overlay(img, cam))
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="white", lw=1.5, ls="--"))
        ax.set_title(f"{f}  p={pr:.2f}", fontsize=10, color="#0b0b0b")
    fig.suptitle("Grad-CAM vs radiologist box (dashed) - test set", color="#0b0b0b", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path("docs/figures/gradcam_examples.png"), dpi=120)
    plt.close(fig)


def _figure_roc(y, prob, findings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#fcfcfb")
    for k, f in enumerate(findings):
        if 0 < y[:, k].sum() < len(y):
            fpr, tpr, _ = roc_curve(y[:, k], prob[:, k])
            ax.plot(fpr, tpr, color=colors[k], lw=2, label=f"{f} ({roc_auc_score(y[:, k], prob[:, k]):.3f})")
    ax.plot([0, 1], [0, 1], color="#52514e", lw=1, ls="--")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e6e5e0")
    ax.legend(frameon=False, loc="lower right", fontsize=9, title="Finding (AUROC)")
    ax.set_title("Test-set ROC per finding", loc="left", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path("docs/figures/roc_per_finding.png"), dpi=130)
    plt.close(fig)


def run(which: str = "all") -> None:
    path("docs/figures").mkdir(parents=True, exist_ok=True)
    if which in ("all", "cls"):
        c = eval_classifier()
        json.dump(c, open(path("metrics/classification.json"), "w"), indent=2)
        log.info("classification: macro-AUROC %.4f | Grad-CAM pointing %.3f", c["macro_auroc"],
                 c["gradcam_localisation"]["pointing_game"])
    if which in ("all", "seg"):
        s = eval_segmentation()
        json.dump(s, open(path("metrics/segmentation.json"), "w"), indent=2)
        log.info("segmentation: mean Dice %.4f | HD95 %.1f mm", s["mean_dice"], s["mean_hd95_mm"])


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(sys.argv[1] if len(sys.argv) > 1 else "all")
