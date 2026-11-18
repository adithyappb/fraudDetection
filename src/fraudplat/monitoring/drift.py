"""Step 6: drift monitoring with Evidently + automatic retraining trigger.

Retraining is triggered when EITHER
  * data drift: share of drifted model features > monitoring.drift_share_threshold, OR
  * any monitoring.critical_features (amount behaviour) drifts on its own, OR
  * concept drift: PR-AUC on analyst/chargeback feedback drops more than
    monitoring.pr_auc_drop_threshold below the offline test PR-AUC
    (only evaluated once min_feedback_labels labels exist).

  python -m fraudplat.monitoring.drift                    # live traffic from the case DB
  python -m fraudplat.monitoring.drift --offline-test     # test split vs training (demo)
  python -m fraudplat.monitoring.drift --auto-retrain     # run the training pipeline if triggered
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import warnings

import pandas as pd
from sklearn.metrics import average_precision_score

from ..config import load_config, path
from ..features import CATEGORICAL, MODEL_FEATURES, CategoryEncoder, assemble

log = logging.getLogger(__name__)
# Cumulative counters and "days since" fields grow with time by construction, so they always
# look drifted against any past window; monitoring them would trigger retraining forever.
NON_STATIONARY = {"uid_prior_cnt", "acct_txn_count", "acct_active_days", "acct_n_devices", "dev_n_accounts",
                  "dev_txn_count", "dev_prior_uids", "card_prior_uids", "card_prior_addrs", "dow",
                  "D1", "D2", "D3", "D4", "D10", "D11", "D15"}
MONITORED = [f for f in MODEL_FEATURES if f not in CATEGORICAL and f not in NON_STATIONARY]


def reference_frame(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    cfg = load_config()
    t = pd.read_parquet(path(cfg["data"]["processed_dir"]) / "training_table.parquet")
    enc = CategoryEncoder(json.loads((path(cfg["serving"]["artifacts_dir"]) / "encoder.json").read_text()))
    ref = t[t.split == "valid"]          # most recent labelled window = closest to live traffic
    return assemble(ref.sample(min(n, len(ref)), random_state=seed), enc)


def evidently_drift(reference: pd.DataFrame, current: pd.DataFrame, html_out=None) -> dict:
    # skip columns that are empty in either window (e.g. IEEE columns absent from a data source)
    cols = [c for c in MONITORED if c in current and current[c].notna().any() and reference[c].notna().any()]
    ref, cur = reference[cols].astype(float), current[cols].astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from evidently import Report
        from evidently.presets import DataDriftPreset
        snap = Report([DataDriftPreset()]).run(cur, ref)
    if html_out:
        snap.save_html(str(html_out))
    per_col, share = {}, 0.0
    for m in snap.dict()["metrics"]:
        if m["metric_name"].startswith("DriftedColumnsCount"):
            share = float(m["value"]["share"])
        elif m["metric_name"].startswith("ValueDrift"):
            method, thr, val = m["config"].get("method", ""), m["config"].get("threshold", 0.05), float(m["value"])
            per_col[m["config"]["column"]] = {"method": method, "value": round(val, 5),
                                              "drifted": bool(val < thr if "p_value" in method else val > thr)}
    drifted = sorted([c for c, v in per_col.items() if v["drifted"]])
    return {"engine": "evidently", "drift_share": share, "n_features": len(cols), "drifted_features": drifted,
            "per_feature": per_col}


def check(current: pd.DataFrame, labels: pd.Series | None, scores: pd.Series | None, html_out=None) -> dict:
    cfg = load_config()
    mon = cfg["monitoring"]
    art = path(cfg["serving"]["artifacts_dir"])
    drift = evidently_drift(reference_frame(), current, html_out)
    res = {"checked_at": time.time(), "n_current": int(len(current)), "data_drift": drift, "reasons": []}
    if drift["drift_share"] > mon["drift_share_threshold"]:
        res["reasons"].append(f"data drift: {drift['drift_share']:.0%} of features drifted "
                              f"(> {mon['drift_share_threshold']:.0%})")
    crit = sorted(set(mon.get("critical_features", [])) & set(drift["drifted_features"]))
    if crit:
        res["reasons"].append(f"critical feature drift: {', '.join(crit)}")
    baseline = json.loads((art / "gnn_summary.json").read_text())["test"]["blend"]["pr_auc"]
    res["baseline_pr_auc"] = baseline
    if labels is not None and labels.notna().sum() >= mon["min_feedback_labels"] and labels.dropna().nunique() == 2:
        m = labels.notna()
        live = float(average_precision_score(labels[m].astype(int), scores[m]))
        res["live_pr_auc"] = live
        if baseline - live > mon["pr_auc_drop_threshold"]:
            res["reasons"].append(f"performance drop: live PR-AUC {live:.3f} vs baseline {baseline:.3f}")
    else:
        res["live_pr_auc"] = None
    res["retrain"] = bool(res["reasons"])
    (art / "drift_status.json").write_text(json.dumps(res, indent=2))
    if res["retrain"]:
        (art / "retrain_request.json").write_text(json.dumps({"requested_at": time.time(), "reasons": res["reasons"]}))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-test", action="store_true")
    ap.add_argument("--shift", type=float, default=0.0, help="demo: multiply amounts by this factor to simulate drift")
    ap.add_argument("--auto-retrain", action="store_true")
    a = ap.parse_args()
    import os
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("feast", "mlflow", "alembic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config()
    art = path(cfg["serving"]["artifacts_dir"])
    if a.offline_test:
        t = pd.read_parquet(path(cfg["data"]["processed_dir"]) / "training_table.parquet")
        te = t[t.split == "test"].sample(5000, random_state=1)
        if a.shift:
            te = te.assign(amount=te.amount * a.shift)
        enc = CategoryEncoder(json.loads((art / "encoder.json").read_text()))
        cur = assemble(te, enc)
        preds = pd.read_parquet(art / "pred_test.parquet").set_index("transaction_id")
        labels = te.is_fraud.reset_index(drop=True).astype(float)
        scores = preds.loc[te.transaction_id, "p_blend"].reset_index(drop=True)
    else:
        from ..serving.cases import CaseStore
        rows = CaseStore(path(cfg["serving"]["db_path"])).recent_scores(5000)
        if len(rows) < 100:
            log.info("only %d scored transactions; need >= 100 for a drift check", len(rows))
            return
        cur = pd.DataFrame([r["features"] for r in rows])
        labels = pd.Series([r["label"] for r in rows], dtype=float)
        scores = pd.Series([r["fraud_score"] for r in rows])
    res = check(cur, labels, scores, html_out=art / "drift_report.html")
    log.info("drift share %.2f | live PR-AUC %s | retrain=%s %s", res["data_drift"]["drift_share"],
             res["live_pr_auc"], res["retrain"], res["reasons"])
    if res["retrain"] and a.auto_retrain:
        from .. import pipeline
        log.info("Retraining triggered")
        pipeline.run(retrain=True)


if __name__ == "__main__":
    main()
