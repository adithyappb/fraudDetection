# API reference

Base URL `http://localhost:8000`. Interactive Swagger docs are at `/docs` and the OpenAPI schema at `/openapi.json`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness/readiness: model version, alert threshold |
| POST | `/score` | score one transaction (`?explain=true` always returns reasons) |
| POST | `/score/batch` | score up to 1,000 transactions |
| GET | `/cases?status=open&limit=200` | alert queue (status: open, investigating, closed) |
| GET | `/cases/{transaction_id}` | one case |
| PATCH | `/cases/{transaction_id}` | update status / label / analyst / note |
| POST | `/feedback/{transaction_id}?label=fraud\|legit` | external label (e.g. a chargeback) |
| GET | `/stats` | counters, latency percentiles, case breakdown |
| GET | `/model` | offline training report + latest drift status |
| GET | `/metrics` | Prometheus text format |

## POST /score

Request: an IEEE-CIS transaction. Required fields are `TransactionID`, `TransactionDT`, `TransactionAmt` (> 0) and `card1`. Extra IEEE-CIS columns (C1…C14, D2…D15, dist1, …) are accepted and used.

```json
{
  "TransactionID": 9000001, "TransactionDT": 15811200, "TransactionAmt": 899.0, "ProductCD": "C",
  "card1": 13926, "card2": 358, "card5": 226, "card4": "visa", "card6": "credit",
  "addr1": 315, "D1": 0, "P_emaildomain": "anonymous.com",
  "DeviceType": "mobile", "DeviceInfo": "SM-G955U", "id_31": "chrome 64.0 for android", "id_33": "1334x750_ab12"
}
```

Response (a real alert from the replayed test period, a confirmed fraud-ring member):

```json
{
  "transaction_id": 3244573,
  "uid": "u16710_378_-355",
  "device_id": "dLG-H870|firefox 57.0|1920x1080_31216",
  "fraud_score": 0.962847,
  "xgb_score": 0.944211,
  "gnn_score": 0.936401,
  "device_ring_score": 0.999976,
  "decision": "REVIEW",
  "threshold": 0.054048,
  "reasons": [
    {"feature": "dev_prior_uids",   "value": 12.0,   "contribution": 1.8255},
    {"feature": "amt_to_acct_mean", "value": 2.8823, "contribution": 1.7773},
    {"feature": "dist1",            "value": 51.0,   "contribution": 0.5209},
    {"feature": "gnn_account_ring_score", "value": 0.9364, "contribution": null},
    {"feature": "gnn_device_ring_score",  "value": 1.0,    "contribution": null}
  ],
  "latency_ms": 2.04
}
```

* `fraud_score`: blended probability (XGBoost + account and device ring risk).
* `decision`: `REVIEW` when `fraud_score ≥ threshold`. The threshold is chosen on the validation period to hit 80% precision. `REVIEW` also opens a case.
* `reasons`: top XGBoost feature contributions (TreeSHAP, log-odds), plus GNN ring flags.
* Errors: `422` with field details for invalid payloads; `413` for batches over 1,000.

## PATCH /cases/{id}

```json
{"status": "closed", "label": "fraud", "analyst": "dp", "note": "same device as ring #12"}
```
`status` ∈ {open, investigating, closed}; `label` ∈ {fraud, legit}. Setting a label also records feedback used by drift monitoring and retraining. Returns the updated case; `404` if unknown, `422` for invalid values.

## GET /stats

```json
{"scored": 6000, "alerts": 254, "threshold": 0.054, "latency_ms_p50": 1.37, "latency_ms_p95": 2.46,
 "latency_ms_p99": 3.44, "cases_by_status": {"open": 254}, "cases_by_label": {"unlabelled": 254},
 "scored_total": 6000, "feedback_labels": 6000}
```

## Kafka interface

When run with `KAFKA_ENABLED=1`, the service consumes `transactions` (same JSON as `POST /score`) and publishes each `/score` response to `fraud-scores`, keyed by `uid`. Offsets are committed after publishing. A message that fails to score is published with an `error` field instead of crashing the consumer.
