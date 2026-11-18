"""Evaluation metrics for heavily imbalanced fraud detection (accuracy is meaningless here:
predicting 'never fraud' is >96% accurate)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def recall_at_precision(y, p, target=0.8) -> tuple[float, float]:
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.flatnonzero(prec[:-1] >= target)
    if len(ok) == 0:
        return 0.0, 1.0
    i = ok[np.argmax(rec[:-1][ok])]
    return float(rec[i]), float(thr[i])


def precision_at_k(y, p, frac=0.01) -> float:
    k = max(1, int(len(p) * frac))
    top = np.argsort(-p)[:k]
    return float(np.asarray(y)[top].mean())


def evaluate(y, p, target_precision=0.8) -> dict:
    y = np.asarray(y)
    p = np.asarray(p)
    r, thr = recall_at_precision(y, p, target_precision)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        f"recall_at_p{int(target_precision * 100)}": r,
        "threshold_at_target_precision": thr,
        "precision_at_top1pct": precision_at_k(y, p, 0.01),
        "base_rate": float(y.mean()),
    }
