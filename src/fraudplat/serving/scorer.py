"""Online scoring: raw transaction -> features (stream state + Feast online) -> XGBoost ->
blend with GNN ring score -> decision + explanation. Used by both the HTTP API and the Kafka
consumer, so both paths are guaranteed identical."""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque

import numpy as np
import pandas as pd
import xgboost as xgb

from ..config import load_config, path
from ..data import canonical_record
from ..feature_store import get_store, online_profile
from ..features import MODEL_FEATURES, CategoryEncoder, OnlineFeatureEngine, assemble_record
from .cases import CaseStore

log = logging.getLogger(__name__)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Scorer:
    def __init__(self, warm_start: bool = True):
        cfg = load_config()
        art = path(cfg["serving"]["artifacts_dir"])
        self.booster = xgb.Booster()
        self.booster.load_model(art / "xgb_model.json")
        self.booster.set_param({"nthread": 1})          # lowest single-request latency
        self.encoder = CategoryEncoder(json.loads((art / "encoder.json").read_text()))
        self.fill_na = json.loads((art / "xgb_summary.json").read_text()).get("fill_na", False)
        self.blend = json.loads((art / "blend.json").read_text())
        thr = cfg["serving"]["alert_threshold"]
        self.threshold = float(self.blend["threshold"] if thr == "auto" else thr)
        self.store = get_store()
        self.engine = OnlineFeatureEngine()
        self.cases = CaseStore(path(cfg["serving"]["db_path"]))
        self.lock = threading.Lock()
        self.latencies = deque(maxlen=5000)
        self.n_scored = 0
        self.n_alerts = 0
        self.model_version = (art / "xgb_model.json").stat().st_mtime_ns
        if warm_start:
            hist = path(cfg["data"]["processed_dir"]) / "stream_history.parquet"
            if hist.exists():
                t0 = time.time()
                self.engine.warm_start(pd.read_parquet(hist))
                log.info("Stream state warm-started from history in %.1fs", time.time() - t0)

    def score_raw(self, raw: dict, record_case: bool = True, explain: bool = False) -> dict:
        t0 = time.perf_counter()
        canon = canonical_record(raw)
        with self.lock:                                   # stream state is per-process
            rt = self.engine.compute(canon, update=True)
        prof = online_profile(self.store, canon["uid"], canon["device_id"])
        dflt = float(self.blend["gnn_default"])
        graph = {k: (dflt if prof.get(k) != prof.get(k) or prof.get(k) is None else float(prof[k]))
                 for k in ("gnn_score", "gnn_dev_mean", "gnn_dev_max")}
        gnn = graph["gnn_score"]

        x = assemble_record({**canon, **rt, **prof}, self.encoder)
        if self.fill_na:
            x = np.nan_to_num(x, nan=-999.0)
        x = x.reshape(1, -1)
        p_xgb = float(self.booster.inplace_predict(x)[0])
        inputs = {"p_xgb": p_xgb, "p_gnn": gnn, **graph}
        z = self.blend["intercept"] + sum(c * _logit(inputs[f]) for f, c in self.blend["coef"].items())
        score = float(1 / (1 + np.exp(-z)))
        alert = score >= self.threshold
        reasons = []
        if alert or explain:                              # SHAP-style contributions only when needed
            contrib = self.booster.predict(xgb.DMatrix(x, feature_names=MODEL_FEATURES), pred_contribs=True)[0][:-1]
            top = np.argsort(-contrib)[:3]
            reasons = [{"feature": MODEL_FEATURES[i], "value": None if np.isnan(x[0, i]) else float(x[0, i]),
                        "contribution": round(float(contrib[i]), 4)} for i in top if contrib[i] > 0]
            if gnn > 0.5:
                reasons.append({"feature": "gnn_account_ring_score", "value": round(gnn, 4), "contribution": None})
            if graph["gnn_dev_max"] > 0.5:
                reasons.append({"feature": "gnn_device_ring_score", "value": round(graph["gnn_dev_max"], 4),
                                "contribution": None})
        latency_ms = (time.perf_counter() - t0) * 1000
        result = {
            "transaction_id": int(canon["transaction_id"]), "uid": canon["uid"], "device_id": canon["device_id"],
            "fraud_score": round(score, 6), "xgb_score": round(p_xgb, 6), "gnn_score": round(gnn, 6), "device_ring_score": round(graph["gnn_dev_max"], 6),
            "decision": "REVIEW" if alert else "APPROVE", "threshold": self.threshold,
            "reasons": reasons, "latency_ms": round(latency_ms, 2),
        }
        self.latencies.append(latency_ms)
        self.n_scored += 1
        if record_case:
            self.cases.log_score(result, canon, dict(zip(MODEL_FEATURES, map(float, x[0]), strict=False)))
            if alert:
                self.n_alerts += 1
                self.cases.open_case(result, canon)
        return result

    def stats(self) -> dict:
        lat = np.array(self.latencies) if self.latencies else np.array([0.0])
        return {"scored": self.n_scored, "alerts": self.n_alerts, "threshold": self.threshold,
                "latency_ms_p50": round(float(np.percentile(lat, 50)), 2),
                "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
                "latency_ms_p99": round(float(np.percentile(lat, 99)), 2)}
