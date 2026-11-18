"""Kafka consumer: transactions topic -> score -> fraud-scores topic.

Partitioning: the producer keys messages by account (card1_addr1), so all events of one
account land on the same partition/consumer -> the per-account stream state stays local
and consistent when scaling consumers horizontally (one replica per partition)."""
from __future__ import annotations

import json
import logging
import threading
import time

from ..config import load_config

log = logging.getLogger(__name__)


def process_batch(scorer, messages, producer=None, topic_out=None) -> list[dict]:
    """Pure function over an iterable of decoded transaction dicts (unit-testable without Kafka)."""
    out = []
    for raw in messages:
        try:
            res = scorer.score_raw(raw)
        except Exception as e:  # poison message -> dead-letter, never crash the loop
            log.exception("scoring failed for %s", raw.get("TransactionID"))
            res = {"transaction_id": raw.get("TransactionID"), "error": str(e)}
        if producer is not None:
            producer.send(topic_out, key=str(res.get("uid", "")).encode(), value=json.dumps(res).encode())
        out.append(res)
    return out


def consume_forever(scorer, stop: threading.Event | None = None):
    from kafka import KafkaConsumer, KafkaProducer
    cfg = load_config()["kafka"]
    stop = stop or threading.Event()
    while not stop.is_set():
        try:
            consumer = KafkaConsumer(cfg["topic_in"], bootstrap_servers=cfg["bootstrap"], group_id=cfg["group_id"],
                                     enable_auto_commit=False, auto_offset_reset="earliest",
                                     value_deserializer=lambda b: json.loads(b.decode()), max_poll_records=500)
            producer = KafkaProducer(bootstrap_servers=cfg["bootstrap"], linger_ms=5, acks=1)
            log.info("Consuming %s from %s", cfg["topic_in"], cfg["bootstrap"])
            while not stop.is_set():
                polled = consumer.poll(timeout_ms=500)
                for _tp, recs in polled.items():
                    process_batch(scorer, (r.value for r in recs), producer, cfg["topic_out"])
                if polled:
                    producer.flush()
                    consumer.commit()             # at-least-once: commit after results are published
        except Exception as e:
            log.warning("Kafka unavailable (%s); retrying in 5s", e)
            time.sleep(5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from ..serving.scorer import Scorer
    consume_forever(Scorer())
