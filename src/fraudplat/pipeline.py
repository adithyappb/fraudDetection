"""End-to-end training pipeline.

  python -m fraudplat.pipeline                 # auto: IEEE-CIS if present, else synthetic
  python -m fraudplat.pipeline --source ieee
  python -m fraudplat.pipeline --quick         # small synthetic run (~1 min) for smoke tests
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import warnings

import pandas as pd

from . import prepare, train_gnn, train_xgb
from .config import load_config, path
from .feature_store import get_store

log = logging.getLogger("fraudplat.pipeline")


def run(source: str | None = None, retrain: bool = False, variants: list[str] | None = None) -> dict:
    t0 = time.time()
    cfg = load_config()
    table = prepare.run(source)
    xgb_summary = train_xgb.run(table, variants)
    gnn_summary = train_gnn.run(table)
    # online store reflects the world as of the start of the replay window
    test_start = pd.to_datetime(int(table.ts[table.split == "test"].min()), unit="s", utc=True)
    get_store().materialize(test_start)
    art = path(cfg["serving"]["artifacts_dir"])
    (art / "retrain_request.json").unlink(missing_ok=True)
    report = {
        "eda": json.loads((path(cfg["data"]["processed_dir"]) / "eda.json").read_text()),
        "xgboost": xgb_summary, "gnn_blend": gnn_summary, "retrain": retrain,
        "pipeline_seconds": round(time.time() - t0, 1),
    }
    (art / "training_report.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("Pipeline finished in %.0fs. Test PR-AUC: xgb %.4f, blend %.4f", report["pipeline_seconds"],
             gnn_summary["test"]["xgb"]["pr_auc"], gnn_summary["test"]["blend"]["pr_auc"])
    return report


def print_summary(report: dict) -> None:
    g = report["gnn_blend"]
    rows = [("GNN ring score only", g["test"]["gnn"]), ("XGBoost (" + report["xgboost"]["selected_variant"] + ")", g["test"]["xgb"]),
            ("Blend XGBoost + GNN", g["test"]["blend"])]
    e = report["eda"]
    print("\n" + "=" * 78)
    print(f" RESULTS - held-out test period   {e['rows']:,} transactions, fraud rate {e['fraud_rate_pct']}%")
    print("=" * 78)
    print(f" {'Model':<30}{'PR-AUC':>9}{'ROC-AUC':>9}{'Recall@P80':>12}{'Prec@top1%':>12}")
    for name, m in rows:
        print(f" {name:<30}{m['pr_auc']:>9.4f}{m['roc_auc']:>9.4f}{m['recall_at_p80']:>12.4f}{m['precision_at_top1pct']:>12.4f}")
    t = g["test_at_threshold"]
    print(f" Random baseline (base rate)   {g['test']['xgb']['base_rate']:>8.4f}")
    print("-" * 78)
    print(f" Alert threshold {t['threshold']:.4f} (set on validation for 80% precision):")
    print(f"   {t['alerts']:,} alerts ({t['alert_rate_pct']}% of traffic)  precision {t['precision']:.3f}  recall {t['recall']:.3f}")
    print("=" * 78 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["auto", "ieee", "synthetic"], default=None)
    ap.add_argument("--quick", action="store_true", help="60k synthetic rows, class_weight only")
    a = ap.parse_args()
    warnings.simplefilter("ignore")                       # third-party deprecation chatter
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("feast", "mlflow", "alembic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    variants = None
    if a.quick:
        cfg = load_config()
        cfg["data"]["synthetic_rows"] = 60_000
        cfg["xgboost"]["n_estimators"] = 200
        a.source = a.source or "synthetic"
        variants = ["class_weight"]
    print_summary(run(a.source, variants=variants))


if __name__ == "__main__":
    main()
