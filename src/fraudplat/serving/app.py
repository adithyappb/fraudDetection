"""FastAPI scoring + case-management service.

Run:  uvicorn fraudplat.serving.app:app --port 8000
With KAFKA_ENABLED=1 it also runs the Kafka consumer loop (transactions -> fraud-scores).
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from .scorer import Scorer

log = logging.getLogger(__name__)
STATE: dict = {}


class Transaction(BaseModel):
    """IEEE-CIS transaction schema (only TransactionID/DT/Amt/card1 are required)."""
    model_config = ConfigDict(extra="allow", json_schema_extra={"example": {
        "TransactionID": 3500001, "TransactionDT": 15811200, "TransactionAmt": 249.99, "ProductCD": "C",
        "card1": 13926, "card2": 358.0, "card5": 226.0, "card4": "visa", "card6": "credit",
        "addr1": 315.0, "D1": 14.0, "P_emaildomain": "gmail.com", "DeviceType": "mobile",
        "DeviceInfo": "iOS Device", "id_31": "mobile safari 11.0", "id_33": "2208x1242_1a2b"}})
    TransactionID: int
    TransactionDT: int = Field(..., description="seconds from the dataset reference time")
    TransactionAmt: float = Field(..., gt=0)
    ProductCD: str = "W"
    card1: int
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    D1: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = None
    DeviceInfo: Optional[str] = None
    id_31: Optional[str] = None
    id_33: Optional[str] = None


class CaseUpdate(BaseModel):
    status: Optional[Literal["open", "investigating", "closed"]] = None
    label: Optional[Literal["fraud", "legit"]] = None
    analyst: Optional[str] = None
    note: Optional[str] = None


def _kafka_loop(scorer: Scorer, stop: threading.Event):
    from ..streaming.consumer import consume_forever
    consume_forever(scorer, stop)


def _hot_reload(stop: threading.Event, every: float = 30.0):
    """Swap in a retrained model (written by the retraining job) without downtime.
    Stream state is carried over so velocity features stay continuous."""
    from ..config import load_config, path
    f = path(load_config()["serving"]["artifacts_dir"]) / "xgb_model.json"
    while not stop.wait(every):
        try:
            cur = STATE["scorer"]
            if f.exists() and f.stat().st_mtime_ns != cur.model_version:
                new = Scorer(warm_start=False)
                new.engine = cur.engine
                STATE["scorer"] = new
                log.info("Hot-reloaded model version %s", new.model_version)
        except Exception:
            log.exception("hot reload failed; keeping current model")


@asynccontextmanager
async def lifespan(app: FastAPI):
    scorer = STATE.get("scorer") or Scorer(warm_start=os.environ.get("WARM_START", "1") == "1")
    STATE["scorer"] = scorer
    stop = threading.Event()
    threading.Thread(target=_hot_reload, args=(stop,), daemon=True).start()
    if os.environ.get("KAFKA_ENABLED") == "1":
        t = threading.Thread(target=_kafka_loop, args=(scorer, stop), daemon=True)
        t.start()
        log.info("Kafka consumer thread started")
    yield
    stop.set()


app = FastAPI(title="Real-Time Fraud Detection API", version="1.0.0", lifespan=lifespan,
              description="Scores transactions in milliseconds (XGBoost + GNN fraud-ring risk) and manages analyst cases.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def scorer() -> Scorer:
    return STATE["scorer"]


@app.get("/health")
def health():
    s = scorer()
    return {"status": "ok", "model_version": s.model_version, "threshold": s.threshold}


@app.post("/score")
def score(txn: Transaction, explain: bool = Query(False, description="always return feature contributions")):
    return scorer().score_raw(txn.model_dump(), explain=explain)


@app.post("/score/batch")
def score_batch(txns: list[Transaction]):
    if len(txns) > 1000:
        raise HTTPException(413, "max 1000 transactions per batch")
    return [scorer().score_raw(t.model_dump()) for t in txns]


@app.get("/cases")
def list_cases(status: Optional[str] = Query(None), limit: int = Query(200, le=2000)):
    return scorer().cases.list_cases(status, limit)


@app.get("/cases/{tid}")
def get_case(tid: int):
    c = scorer().cases.get_case(tid)
    if not c:
        raise HTTPException(404, "case not found")
    return c


@app.patch("/cases/{tid}")
def update_case(tid: int, upd: CaseUpdate):
    if not scorer().cases.get_case(tid):
        raise HTTPException(404, "case not found")
    return scorer().cases.update_case(tid, **upd.model_dump())


@app.post("/feedback/{tid}")
def feedback(tid: int, label: Literal["fraud", "legit"]):
    scorer().cases.add_feedback(tid, 1 if label == "fraud" else 0, "chargeback")
    return {"transaction_id": tid, "label": label}


@app.get("/stats")
def stats():
    s = scorer()
    return {**s.stats(), **s.cases.summary()}


@app.get("/model")
def model_info():
    """Offline training report + latest drift status (for the dashboard)."""
    import json
    from ..config import load_config, path
    art = path(load_config()["serving"]["artifacts_dir"])
    out = {}
    for name in ("training_report", "drift_status"):
        f = art / f"{name}.json"
        out[name] = json.loads(f.read_text()) if f.exists() else None
    out["retrain_requested"] = (art / "retrain_request.json").exists()
    return out


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus exposition format."""
    st = stats()
    lines = [f"fraud_scored_total {st['scored']}", f"fraud_alerts_total {st['alerts']}",
             f"fraud_feedback_labels_total {st['feedback_labels']}"]
    for q in ("p50", "p95", "p99"):
        lines.append(f'fraud_score_latency_ms{{quantile="{q[1:]}"}} {st["latency_ms_" + q]}')
    return "\n".join(lines) + "\n"
