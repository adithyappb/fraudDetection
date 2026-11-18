"""DVC stage `calibrate`: per-finding decision thresholds, chosen on the VALIDATION split only.

Operating point: the lowest threshold whose validation specificity is >= target_specificity, i.e. at most
~20% false alerts per finding (limits alert fatigue in the worklist). Test data is never used here.
"""
from __future__ import annotations

import json
import logging

import numpy as np

from .config import MODELS, params
from .evaluate import load_classifier
from .train_cls import load_split, predict

log = logging.getLogger("medai.calibrate")


def run() -> None:
    target = params()["classification"]["target_specificity"]
    model, ckpt = load_classifier()
    va, imgs, sel = load_split("val")
    prob = predict(model, imgs, sel)
    out = {}
    for k, f in enumerate(ckpt["findings"]):
        neg = prob[va[f].to_numpy() == 0, k]
        out[f] = float(np.quantile(neg, target)) if len(neg) else 0.5
    json.dump({"target_specificity": target, "split": "val", "thresholds": out},
              open(MODELS / "thresholds.json", "w"), indent=2)
    log.info("thresholds @ specificity %.2f: %s", target, {k: round(v, 3) for k, v in out.items()})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
