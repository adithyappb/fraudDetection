import json

import pandas as pd


class FakeProducer:
    def __init__(self):
        self.sent = []

    def send(self, topic, key=None, value=None):
        self.sent.append((topic, key, json.loads(value)))


def test_consumer_scores_and_publishes(trained):
    from fraudplat.serving.scorer import Scorer
    from fraudplat.streaming.consumer import process_batch
    from fraudplat.streaming.producer import load_replay, records
    s = Scorer(warm_start=False)
    msgs = [r for r, _ in records(load_replay(50))] + [{"TransactionID": 1}]   # last one is a poison message
    prod = FakeProducer()
    out = process_batch(s, msgs, prod, "fraud-scores")
    assert len(out) == len(prod.sent) == 51
    assert "error" in out[-1] and all("fraud_score" in o for o in out[:-1])


def test_drift_detection_triggers_retrain_only_on_shift(trained):
    from fraudplat.config import load_config, path
    from fraudplat.features import CategoryEncoder, assemble
    from fraudplat.monitoring.drift import check
    t = pd.read_parquet(path(load_config()["data"]["processed_dir"]) / "training_table.parquet")
    enc = CategoryEncoder(json.loads((trained / "encoder.json").read_text()))
    same = assemble(t[t.split == "valid"].sample(3000, random_state=5), enc)
    assert check(same, None, None)["retrain"] is False
    shifted = t[t.split == "test"].sample(3000, random_state=5)
    shifted = shifted.assign(amount=shifted.amount * 4, C1=shifted.C1 * 3 + 5, C13=shifted.C13 * 3 + 5,
                             C14=shifted.C14 * 3 + 5, D10=shifted.D10 + 200, D15=shifted.D15 + 200,
                             acct_amt_mean=shifted.acct_amt_mean * 4, dev_amt_mean=shifted.dev_amt_mean * 4,
                             dist1=shifted.dist1 * 5, uid_amt_sum_24h=shifted.uid_amt_sum_24h * 4,
                             acct_amt_std=shifted.acct_amt_std * 4)
    res = check(assemble(shifted, enc), None, None)
    assert res["retrain"] is True and res["data_drift"]["drift_share"] > 0.3
    (trained / "retrain_request.json").unlink(missing_ok=True)


def test_kafka_consume_loop_commits_after_publish(trained, monkeypatch):
    """Runs the real consume_forever loop against an in-memory stand-in for kafka-python."""
    import sys
    import threading
    import types
    from fraudplat.serving.scorer import Scorer
    from fraudplat.streaming import consumer as cons
    from fraudplat.streaming.producer import load_replay, records

    msgs = [types.SimpleNamespace(value=r) for r, _ in records(load_replay(30))]
    stop = threading.Event()
    log = {"commits": 0, "sent": []}

    class FakeConsumer:
        def __init__(self, topic, **kw):
            assert kw["enable_auto_commit"] is False
            self.batches = [msgs[:20], msgs[20:]]

        def poll(self, timeout_ms=0):
            if self.batches:
                return {"tp0": self.batches.pop(0)}
            stop.set()
            return {}

        def commit(self):
            log["commits"] += 1

    class FakeProducer:
        def __init__(self, **kw):
            pass

        def send(self, topic, key=None, value=None):
            log["sent"].append((topic, json.loads(value)))

        def flush(self):
            pass

    monkeypatch.setitem(sys.modules, "kafka", types.SimpleNamespace(KafkaConsumer=FakeConsumer, KafkaProducer=FakeProducer))
    cons.consume_forever(Scorer(warm_start=False), stop)
    assert len(log["sent"]) == 30 and log["commits"] == 2
    assert all(t == "fraud-scores" and "fraud_score" in v for t, v in log["sent"])
