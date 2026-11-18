import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(trained, tmp_path_factory):
    import os
    os.environ["KAFKA_ENABLED"] = "0"
    from fraudplat.config import load_config
    load_config()["serving"]["db_path"] = str(tmp_path_factory.mktemp("db") / "cases.db")
    from fraudplat.serving import app as app_mod
    app_mod.STATE.clear()
    with TestClient(app_mod.app) as c:
        yield c


@pytest.fixture(scope="module")
def replay(trained):
    from fraudplat.streaming.producer import load_replay, records
    return list(records(load_replay(1500)))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_score_contract(client, replay):
    rec, _ = replay[0]
    r = client.post("/score", json=rec, params={"explain": True})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_score"] <= 1
    assert body["decision"] in {"APPROVE", "REVIEW"}
    assert {"xgb_score", "gnn_score", "reasons", "latency_ms"} <= body.keys()


def test_validation_rejects_bad_payload(client):
    assert client.post("/score", json={"TransactionID": 1, "TransactionDT": 1, "TransactionAmt": -5, "card1": 1}).status_code == 422
    assert client.post("/score", json={"TransactionAmt": 5}).status_code == 422


def test_stream_alerts_cases_and_feedback(client, replay):
    scores, labels = [], []
    for rec, y in replay[1:]:
        scores.append(client.post("/score", json=rec).json()["fraud_score"])
        labels.append(y)
    from sklearn.metrics import average_precision_score
    assert average_precision_score(labels, scores) > 3 * np.mean(labels)
    cases = client.get("/cases", params={"status": "open"}).json()
    assert cases, "fraud in the replay must raise alerts"
    tid = cases[0]["transaction_id"]
    upd = client.patch(f"/cases/{tid}", json={"status": "closed", "label": "fraud", "analyst": "test"}).json()
    assert upd["status"] == "closed" and upd["label"] == "fraud"
    assert client.patch(f"/cases/{tid}", json={"status": "bogus"}).status_code == 422
    assert client.get("/stats").json()["feedback_labels"] >= 1


def test_latency_budget(client):
    st = client.get("/stats").json()
    assert st["latency_ms_p95"] < 20, st          # in-process scoring budget (excl. HTTP)


def test_batch_and_metrics(client, replay):
    r = client.post("/score/batch", json=[rec for rec, _ in replay[:5]])
    assert r.status_code == 200 and len(r.json()) == 5
    assert "fraud_score_latency_ms" in client.get("/metrics").text
