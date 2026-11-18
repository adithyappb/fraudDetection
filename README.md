# Real-Time Financial Fraud Detection Platform

Reads live transactions from Kafka, scores each one for fraud risk in **~1–3 ms**, links suspicious
accounts into **fraud rings** with a graph neural network, and gives analysts a dashboard to work
the resulting cases. Drift is monitored with Evidently and triggers automatic retraining.

**Stack:** Kafka · XGBoost · PyTorch Geometric · Feast · MLflow · FastAPI · Evidently AI · Streamlit · Kubernetes (Minikube)
**Track:** AI / Machine Learning

![architecture](docs/figures/architecture.png)

| Milestone (from the brief) | Where |
|---|---|
| 1. Dataset + EDA, class imbalance | `src/fraudplat/data.py`, `prepare.py` → `data/processed/eda.json` |
| 2. Features (velocity, amount vs user average, device & location change) in **Feast** | `features.py`, `feature_repo/`, `feature_store.py` |
| 3. XGBoost with class weights / SMOTE, **PR-AUC**, **MLflow** | `train_xgb.py`, `metrics.py` |
| 4. Account–device–card graph, **GNN (PyG)**, blended with XGBoost | `train_gnn.py` |
| 5. Kafka producer (replay) + FastAPI consumer that fetches features and scores | `streaming/producer.py`, `streaming/consumer.py`, `serving/` |
| 6. Evidently drift reports + automatic retraining trigger | `monitoring/drift.py` |
| 7. Analyst dashboard (alerts, case status, feedback labels) + Kubernetes | `dashboard/app.py`, `k8s/`, `docker/` |

Results, design decisions and challenges: **[docs/REPORT.md](docs/REPORT.md)** · API reference: **[docs/API.md](docs/API.md)** · Architecture & data flow: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

---

## 1. Setup (once)

Requires **Python 3.11–3.13** (3.11 recommended; Feast has the widest support there), ~4 GB RAM, ~3 GB disk.

```bash
unzip fraud-platform.zip && cd fraud-platform
python3.11 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build, small download
pip install -r requirements.txt
```

All commands below are run from the project root with the virtual environment active.
`make <target>` shortcuts exist for each step. On Windows without `make`, use the plain command shown next to each one.
Set `PYTHONPATH` first if you're not using `make`:

```bash
export PYTHONPATH=src                  # Windows PowerShell: $env:PYTHONPATH="src"
```

## 2. Get the data (IEEE-CIS Fraud Detection)

IEEE-CIS is the dataset used because it is the only option with real **entity columns** (card,
address, e-mail, device fingerprint) for the fraud-ring graph, plus a real time axis for velocity
features. See [docs/REPORT.md](docs/REPORT.md#dataset-choice).

**Option A – Kaggle CLI (recommended)**

1. Sign in at kaggle.com → profile picture → **Settings** → **API** → **Create New Token**. This downloads `kaggle.json`.
2. Put it in place:
   ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   # Windows: move kaggle.json to C:\Users\<you>\.kaggle\kaggle.json
   ```
3. Open https://www.kaggle.com/competitions/ieee-fraud-detection/rules and click **"I Understand and Accept"**.
   Without this, the download returns *403 Forbidden*.
4. Download:
   ```bash
   make data                           # = bash scripts/download_data.sh
   ```
   Or manually:
   ```bash
   kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv -p data/raw
   kaggle competitions download -c ieee-fraud-detection -f train_identity.csv -p data/raw
   cd data/raw && unzip -o '*.zip' && rm -f *.zip && cd ../..
   ```

**Option B – browser (if the CLI or your network blocks Kaggle's API)**

Accept the rules (step 3 above). Then on the competition's **Data** tab, download `train_transaction.csv`
and `train_identity.csv` (or "Download All" and unzip). Place both files here:

```
data/raw/train_transaction.csv     (~650 MB)
data/raw/train_identity.csv        (~25 MB)
```

Only these two files are used (the competition's `test_*` files have no labels).

**No data? It still runs.** If `data/raw/train_transaction.csv` is missing, the pipeline uses a built-in
generator with **identical column names** (TransactionDT, TransactionAmt, card1–6, addr1, D1, C*,
DeviceInfo, id_31, id_33, …) that injects fraud rings, account takeovers and low-signal fraud.
Every number in `docs/REPORT.md` comes from that generator because Kaggle was not reachable from the
build environment. On real IEEE-CIS, expect lower absolute numbers (see the report).

## 3. Train

```bash
make train                             # = python -m fraudplat.pipeline
```

The source is chosen automatically: IEEE-CIS if `data/raw/train_transaction.csv` exists, otherwise synthetic.
Force one with `python -m fraudplat.pipeline --source ieee` or `--source synthetic`.
Takes ~2 min on the synthetic data (measured). Expect several times longer on IEEE-CIS (590k rows).

The pipeline:
1. Runs EDA, engineers features, writes Feast feature views and registers them (`feast apply`).
2. Builds a point-in-time-correct training table.
3. Trains 3 XGBoost variants (no weighting = baseline, class weights, SMOTE) and logs each to MLflow.
   It ships the better of class weights / SMOTE by validation PR-AUC.
4. Builds the graph, trains GraphSAGE, publishes ring scores to Feast, and fits the blend.
5. Materializes the Feast online store.

Outputs land in `artifacts/`: `training_report.json` (all metrics) plus the model files.
Quick 30-second smoke run: `make quick`.

## 4. Test

```bash
make test                              # = python -m pytest -q -W ignore      (18 tests, ~30 s)
```

The tests cover training/serving feature parity (batch vs. streaming), no future-data leakage,
Feast vs. reference point-in-time join, model quality gates, the API contract and latency,
the Kafka consume→score→publish→commit loop, and drift-triggered retraining.

## 5. Run the platform locally (no Docker)

Use three terminals, each with the venv active and `PYTHONPATH=src`:

```bash
# Terminal 1 – scoring API (Swagger UI at http://localhost:8000/docs)
make api                    # = uvicorn fraudplat.serving.app:app --port 8000

# Terminal 2 – analyst dashboard at http://localhost:8501
make dashboard              # = streamlit run src/fraudplat/dashboard/app.py

# Terminal 3 – stream 5,000 held-out transactions into the API, plus simulated chargeback labels
make replay                 # = python -m fraudplat.streaming.producer --sink http --rate 0 --limit 5000 --send-labels
```

Then:
```bash
make drift                  # Evidently check on live traffic; retrains automatically if triggered
make drift-demo             # same check on the test split with an injected 3x amount shift -> triggers retrain
make bench                  # latency benchmark
make mlflow                 # MLflow UI at http://localhost:5000 (compare the XGBoost runs)
```

The drift HTML report is written to `artifacts/drift_report.html`.

Try a single score request:
```bash
curl -s -X POST localhost:8000/score?explain=true -H 'Content-Type: application/json' -d '{
  "TransactionID": 9000001, "TransactionDT": 15811200, "TransactionAmt": 899.0, "ProductCD": "C",
  "card1": 13926, "card2": 358, "card5": 226, "card4": "visa", "card6": "credit", "addr1": 315, "D1": 0,
  "P_emaildomain": "anonymous.com", "DeviceType": "mobile", "DeviceInfo": "SM-G955U",
  "id_31": "chrome 64.0 for android", "id_33": "1334x750_ab12"}'
```

## 6. Run the full stack with Kafka (Docker Compose)

```bash
make train                                   # artifacts are baked into the image
docker compose up -d --build                 # Kafka (KRaft), API + Kafka consumer, dashboard, MLflow UI
docker compose --profile replay run --rm producer     # replay test transactions into Kafka at 200 msg/s
```

Then open the dashboard at http://localhost:8501, the API at http://localhost:8000/docs, and MLflow at http://localhost:5000.
Scores are also published to the `fraud-scores` topic:
```bash
docker compose exec kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic fraud-scores --max-messages 5
```

## 7. Deploy on Kubernetes (Minikube)

```bash
make train
make k8s                    # = bash scripts/deploy_minikube.sh
```

This builds the image inside Minikube, applies `k8s/` (namespace + PVC, Kafka StatefulSet, API
Deployment with the Kafka consumer, dashboard, replay Job, hourly drift CronJob), starts a replay,
and prints the API and dashboard URLs. Useful commands:

```bash
kubectl -n fraud get pods
kubectl -n fraud logs deploy/fraud-api -f
kubectl -n fraud create job replay-2 --from=cronjob/replay-producer      # replay again
kubectl -n fraud create job drift-now --from=cronjob/drift-monitor        # run a drift check now
```

When the drift job retrains, the API hot-reloads the new model from the shared volume within 30 s.

## Project layout

```
configs/config.yaml         every tunable (splits, model params, thresholds, Kafka, drift rules)
feature_repo/               Feast definitions (entities: account, device; 4 feature views)
src/fraudplat/
  data.py                   IEEE-CIS loader, synthetic generator, canonical schema + entity keys
  features.py               real-time features (batch + streaming twins), Feast profile features
  feature_store.py          Feast adapter (+ built-in point-in-time store with identical semantics)
  prepare.py                EDA + point-in-time training table
  train_xgb.py              XGBoost x {none, class weights, SMOTE} -> MLflow
  train_gnn.py              graph construction, GraphSAGE, ring scores -> Feast, blend
  pipeline.py               end-to-end training entry point
  metrics.py                PR-AUC, recall@precision, precision@top-k
  serving/                  scorer (shared by HTTP + Kafka), FastAPI app, case store
  streaming/                Kafka producer (replay) and consumer loop
  monitoring/drift.py       Evidently drift + retraining trigger
  dashboard/app.py          Streamlit analyst console
tests/                      18 tests
docker/, docker-compose.yml, k8s/, scripts/
docs/                       REPORT.md, ARCHITECTURE.md, API.md, figures/
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `403 Forbidden` from Kaggle | Accept the competition rules (Section 2, step 3) |
| `kaggle: command not found` | `pip install kaggle` |
| Feast install/import fails (e.g. on Windows) | The pipeline falls back to the built-in point-in-time store automatically. Same features, verified equal by a test |
| `FileNotFoundError: artifacts/xgb_model.json` when starting the API | Run `make train` first |
| Dashboard says "API not reachable" | Start the API first, or set `API_URL=http://host:8000` |
| Out of memory on IEEE-CIS | Close other apps. The loader already reads only ~45 of the 394 columns |
| Start over | `make clean && make train` |
