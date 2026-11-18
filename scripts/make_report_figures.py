"""Regenerate docs/figures/*.png from the latest training artifacts.

  python scripts/make_report_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ART, OUT = ROOT / "artifacts", ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SERIES = {"Blend (XGB + GNN)": ("p_blend", "#2a78d6"), "XGBoost": ("p_xgb", "#eb6834"),
          "GNN ring score only": ("p_gnn", "#1baf7a")}
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF})


def pr_curves():
    pt = pd.read_parquet(ART / "pred_test.parquet")
    summ = json.loads((ART / "gnn_summary.json").read_text())["test"]
    keys = {"p_blend": "blend", "p_xgb": "xgb", "p_gnn": "gnn"}
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for name, (col, color) in SERIES.items():
        p, r, _ = precision_recall_curve(pt.y, pt[col])
        ap = summ[keys[col]]["pr_auc"]
        ax.plot(r, p, color=color, lw=2, label=f"{name}  (PR-AUC {ap:.3f})")
    ax.axhline(pt.y.mean(), color=MUTED, lw=1, ls="--")
    ax.text(0.45, pt.y.mean() + 0.02, f"random = base rate {pt.y.mean():.3f}", color=MUTED, fontsize=9, ha="center")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 0.12), labelcolor=INK)
    ax.set_title("Precision-recall on the held-out test period", color=INK, loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "pr_curves.png", dpi=150)


def imbalance_variants():
    v = json.loads((ART / "xgb_summary.json").read_text())
    names = {"none": "No weighting\n(baseline)", "class_weight": "Class weights", "smote": "SMOTE"}
    rows = [(names[k], d["test"]["pr_auc"], k == v["selected_variant"]) for k, d in v["variants"].items()]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.bar([r[0] for r in rows], [r[1] for r in rows], width=0.5,
                  color=["#2a78d6" if r[2] else "#86b6ef" for r in rows])
    for b, r in zip(bars, rows, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, r[1] + 0.01, f"{r[1]:.3f}" + ("  selected" if r[2] else ""),
                ha="center", color=INK, fontsize=9)
    ax.set(ylabel="Test PR-AUC", ylim=(0, 1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("XGBoost imbalance strategies (tracked in MLflow)", color=INK, loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "imbalance_variants.png", dpi=150)


if __name__ == "__main__":
    pr_curves()
    imbalance_variants()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
