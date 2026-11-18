"""Step 3: XGBoost with imbalance handling, evaluated by PR-AUC, tracked in MLflow."""
from __future__ import annotations

import json
import os
import logging
import time
import warnings

import mlflow
import pandas as pd
import xgboost as xgb

from .config import load_config, path
from .features import MODEL_FEATURES, CategoryEncoder, assemble
from .metrics import evaluate

log = logging.getLogger(__name__)


def setup_mlflow():
    cfg = load_config()["mlflow"]
    uri = cfg["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = "sqlite:///" + str(path(uri.replace("sqlite:///", "")))
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    warnings.filterwarnings("ignore", module="mlflow|sqlalchemy")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["experiment"])


def _fit(Xtr, ytr, Xva, yva, params, imbalance, seed=42):
    if imbalance == "class_weight":
        # weight = (neg/pos)^power; power 0.5 (sqrt) keeps the minority boost without the
        # over-confident, poorly ranked scores that the full ratio produces on PR curves
        ratio = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        params = {**params, "scale_pos_weight": float(ratio ** params.get("class_weight_power", 0.5))}
    elif imbalance == "smote":
        from imblearn.over_sampling import SMOTE
        Xtr = Xtr.fillna(-999)
        Xva = Xva.fillna(-999)
        Xtr, ytr = SMOTE(sampling_strategy=0.25, random_state=seed, k_neighbors=5).fit_resample(Xtr, ytr)
    model = xgb.XGBClassifier(
        n_estimators=params["n_estimators"], max_depth=params["max_depth"],
        learning_rate=params["learning_rate"], subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"], min_child_weight=params["min_child_weight"],
        scale_pos_weight=params.get("scale_pos_weight", 1.0), tree_method="hist",
        eval_metric="aucpr", early_stopping_rounds=params["early_stopping_rounds"],
        random_state=seed, n_jobs=-1)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return model, imbalance == "smote"


def run(table: pd.DataFrame | None = None, variants: list[str] | None = None) -> dict:
    cfg = load_config()
    params = cfg["xgboost"]
    tp = cfg["serving"]["target_precision"]
    if table is None:
        table = pd.read_parquet(path(cfg["data"]["processed_dir"]) / "training_table.parquet")
    tr, va, te = (table[table.split == s] for s in ("train", "valid", "test"))
    enc = CategoryEncoder().fit(tr)
    Xtr, Xva, Xte = (assemble(d, enc) for d in (tr, va, te))
    ytr, yva, yte = (d.is_fraud.to_numpy() for d in (tr, va, te))

    setup_mlflow()
    variants = variants or ["none", "class_weight", "smote"]
    results = {}
    for v in variants:
        with mlflow.start_run(run_name=f"xgb_{v}") as run:
            t0 = time.time()
            model, filled = _fit(Xtr, ytr, Xva, yva, params, v)
            fit_s = time.time() - t0
            pva = model.predict_proba(Xva.fillna(-999) if filled else Xva)[:, 1]
            pte = model.predict_proba(Xte.fillna(-999) if filled else Xte)[:, 1]
            mva, mte = evaluate(yva, pva, tp), evaluate(yte, pte, tp)
            mlflow.set_tag("role", "baseline" if v == "none" else "candidate")
            mlflow.log_params({**params, "imbalance": v, "n_features": len(MODEL_FEATURES),
                               "train_rows": len(Xtr), "best_iteration": model.best_iteration})
            mlflow.log_metrics({f"valid_{k}": x for k, x in mva.items()})
            mlflow.log_metrics({f"test_{k}": x for k, x in mte.items()})
            mlflow.log_metric("fit_seconds", fit_s)
            results[v] = {"run_id": run.info.run_id, "valid": mva, "test": mte, "model": model,
                          "filled": filled, "pva": pva, "pte": pte}
            log.info("xgb[%s] valid PR-AUC %.4f | test PR-AUC %.4f (%.0fs)", v, mva["pr_auc"], mte["pr_auc"], fit_s)

    # The brief requires imbalance handling: "none" is logged only as a reference baseline and
    # the shipped model is the better of class weighting / SMOTE on validation PR-AUC.
    eligible = [k for k in results if k != "none"] or list(results)
    best = max(eligible, key=lambda k: results[k]["valid"]["pr_auc"])
    b = results[best]
    art = path(cfg["serving"]["artifacts_dir"])
    art.mkdir(parents=True, exist_ok=True)
    b["model"].save_model(art / "xgb_model.json")
    (art / "encoder.json").write_text(json.dumps(enc.mapping))
    imp = pd.Series(b["model"].feature_importances_, index=MODEL_FEATURES).sort_values(ascending=False)
    with mlflow.start_run(run_id=b["run_id"]):
        mlflow.set_tag("selected", "true")
        mlflow.log_artifact(str(art / "xgb_model.json"))
        mlflow.log_dict(imp.round(5).to_dict(), "feature_importance.json")
        try:
            mlflow.xgboost.log_model(b["model"], name="model", registered_model_name="fraud-xgb")
        except Exception as e:  # registry optional
            log.warning("Model registry skipped: %s", e)
    summary = {
        "selected_variant": best,
        "variants": {k: {"valid": r["valid"], "test": r["test"], "run_id": r["run_id"]} for k, r in results.items()},
        "fill_na": b["filled"],
        "top_features": imp.head(15).round(4).to_dict(),
    }
    (art / "xgb_summary.json").write_text(json.dumps(summary, indent=2))
    # persist split predictions for blending
    pd.DataFrame({"transaction_id": va.transaction_id, "uid": va.uid, "y": yva, "p_xgb": b["pva"]}).to_parquet(art / "pred_valid.parquet")
    pd.DataFrame({"transaction_id": te.transaction_id, "uid": te.uid, "y": yte, "p_xgb": b["pte"]}).to_parquet(art / "pred_test.parquet")
    return summary
