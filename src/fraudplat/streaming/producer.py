"""Replays held-out (test-period) transactions as a live stream.

  python -m fraudplat.streaming.producer --sink kafka --rate 200      # to Kafka topic
  python -m fraudplat.streaming.producer --sink http --limit 2000      # straight to the API
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

from ..config import load_config, path

log = logging.getLogger(__name__)


def load_replay(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path(load_config()["data"]["processed_dir"]) / "replay.parquet")
    return df.head(limit) if limit else df


def records(df: pd.DataFrame):
    labels = df["isFraud"].to_numpy() if "isFraud" in df else None
    clean = df.drop(columns=["isFraud"], errors="ignore")
    for i, rec in enumerate(clean.to_dict("records")):
        rec = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()}
        rec = {k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}
        yield rec, (int(labels[i]) if labels is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sink", choices=["kafka", "http"], default="kafka")
    ap.add_argument("--rate", type=float, default=200.0, help="messages per second (0 = as fast as possible)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--send-labels", action="store_true",
                    help="simulate delayed chargeback feedback by posting true labels for scored txns")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = load_config()["kafka"]
    df = load_replay(a.limit)
    sent, t0 = 0, time.time()
    alerts = tp = frauds = 0
    if a.sink == "kafka":
        from kafka import KafkaProducer
        prod = KafkaProducer(bootstrap_servers=cfg["bootstrap"], linger_ms=10, acks=1,
                             value_serializer=lambda v: json.dumps(v).encode())
    else:
        import httpx
        client = httpx.Client(base_url=a.api, timeout=10)
    for rec, label in records(df):
        if a.sink == "kafka":
            key = f"{rec.get('card1')}_{rec.get('addr1')}".encode()       # account-affine partitioning
            prod.send(cfg["topic_in"], key=key, value=rec)
        else:
            r = client.post("/score", json=rec)
            r.raise_for_status()
            if label is not None:
                flagged = r.json()["decision"] == "REVIEW"
                alerts += flagged
                frauds += label
                tp += flagged and label == 1
            if a.send_labels and label is not None:
                client.post(f"/feedback/{rec['TransactionID']}", params={"label": "fraud" if label else "legit"})
        sent += 1
        if a.rate > 0:
            time.sleep(max(0.0, sent / a.rate - (time.time() - t0)))
        if sent % 500 == 0:
            if a.sink == "http" and frauds:
                log.info("scored %5d | %4.0f txn/s | alerts %4d | alert precision %.3f | fraud caught %d/%d (recall %.3f)",
                         sent, sent / (time.time() - t0), alerts, tp / max(alerts, 1), tp, frauds, tp / frauds)
            else:
                log.info("sent %d (%.0f msg/s)", sent, sent / (time.time() - t0))
    if a.sink == "kafka":
        prod.flush()
    log.info("done: %d messages in %.1fs", sent, time.time() - t0)


if __name__ == "__main__":
    main()
