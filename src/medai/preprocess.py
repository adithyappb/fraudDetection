"""DVC stage `preprocess`: resize + cache images, patient-level split, scale boxes.

Outputs (data/processed/):
  cxr_images.npy   uint8 [N, S, S]  (resized with area interpolation; normalisation happens in the model transform)
  cxr_index.csv    image, patient, age, sex, view, split, one column per finding
  cxr_boxes.csv    radiologist boxes rescaled to S x S (test-set Grad-CAM evaluation)
  ct_split.json    train / val / test volume lists
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from PIL import Image

from .config import PROCESSED, params, path
from .data import msd, nih

log = logging.getLogger("medai.preprocess")


def patient_split(meta: pd.DataFrame, box_images: set, fracs, seed) -> pd.Series:
    """Every image of a patient lands in one split. Patients with radiologist boxes go to test
    (as in the official NIH test list) so Grad-CAM can be scored there."""
    rng = np.random.default_rng(seed)
    box_patients = set(meta.loc[meta.image.isin(box_images), "patient"])
    others = np.array(sorted(set(meta.patient) - box_patients))
    rng.shuffle(others)
    n = len(others)
    n_train, n_val = int(fracs[0] * n), int(fracs[1] * n)       # fractions of the non-box patients
    assign = {p: "train" for p in others[:n_train]}
    assign.update({p: "val" for p in others[n_train:n_train + n_val]})
    assign.update({p: "test" for p in others[n_train + n_val:]})
    assign.update({p: "test" for p in box_patients})
    return meta.patient.map(assign)


def run() -> None:
    p = params()
    cfg, size = p["classification"], p["classification"]["image_size"]
    raw = path(p["data"]["cxr_raw_dir"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    meta, bb = nih.read_metadata(raw)
    files = {f.name for f in (raw / "images").glob("*.png")}
    meta = meta[meta.image.isin(files)].sort_values("image").reset_index(drop=True)
    log.info("NIH CXR14: %d images, %d patients", len(meta), meta.patient.nunique())
    imgs = np.zeros((len(meta), size, size), np.uint8)
    for i, name in enumerate(meta.image):
        imgs[i] = np.asarray(Image.open(raw / "images" / name).convert("L").resize((size, size), Image.BOX))
    np.save(PROCESSED / "cxr_images.npy", imgs)

    meta["split"] = patient_split(meta, set(bb.image), p["classification"]["split"], p["data"]["seed"])
    y = nih.label_matrix(meta, cfg["findings"])
    idx = pd.concat([meta, pd.DataFrame(y.astype(int), columns=cfg["findings"])], axis=1)
    idx.to_csv(PROCESSED / "cxr_index.csv", index=False)
    boxes = bb[bb.image.isin(set(meta.image)) & bb.finding.isin(cfg["findings"])].copy()
    boxes[["x", "y", "w", "h"]] *= size / nih.NIH_SIZE      # NIH boxes are in 1024 x 1024 pixel space
    boxes.to_csv(PROCESSED / "cxr_boxes.csv", index=False)

    pairs = msd.list_pairs(path(p["data"]["ct_raw_dir"]))
    rng = np.random.default_rng(p["data"]["seed"])
    order = rng.permutation(len(pairs))
    n_test, n_val = max(2, round(0.15 * len(pairs))), max(2, round(0.1 * len(pairs)))
    split = {"test": [pairs[i] for i in order[:n_test]], "val": [pairs[i] for i in order[n_test:n_test + n_val]],
             "train": [pairs[i] for i in order[n_test + n_val:]]}
    json.dump(split, open(PROCESSED / "ct_split.json", "w"), indent=1)

    counts = idx.groupby("split")[cfg["findings"]].sum().astype(int)
    summary = {"images": int(len(idx)), "patients": int(idx.patient.nunique()),
               "split_images": idx.split.value_counts().to_dict(),
               "positives_per_split": counts.to_dict(orient="index"),
               "prevalence_pct": (100 * idx[cfg["findings"]].mean()).round(2).to_dict(),
               "no_finding_pct": round(100 * float((y.sum(1) == 0).mean()), 2),
               "view_pct": (100 * idx.view.value_counts(normalize=True)).round(1).to_dict(),
               "sex_pct": (100 * idx.sex.value_counts(normalize=True)).round(1).to_dict(),
               "age_median": float(idx.age.median()),
               "test_boxes": int(len(boxes)),
               "ct_volumes": {k: len(v) for k, v in split.items()},
               "source": "phantom" if (raw / "PHANTOM.txt").exists() else "real"}
    path("metrics").mkdir(exist_ok=True)
    json.dump(summary, open(path("metrics/data_summary.json"), "w"), indent=2)
    log.info("data summary: %s", json.dumps(summary)[:600])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
