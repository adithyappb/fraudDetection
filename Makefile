PY ?= python
export PYTHONPATH := src
export PYTHONWARNINGS := ignore
export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: install data train quick test api dashboard replay drift drift-demo mlflow bench docker-build up k8s clean

install:            ## install Python dependencies (CPU torch)
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt

data:               ## download IEEE-CIS from Kaggle into data/raw
	bash scripts/download_data.sh

train:              ## full pipeline (IEEE-CIS if present in data/raw, else synthetic)
	$(PY) -m fraudplat.pipeline

quick:              ## 1-minute smoke run on small synthetic data
	$(PY) -m fraudplat.pipeline --quick

test:               ## unit + integration tests
	$(PY) -m pytest -q -W ignore

api:                ## scoring API on :8000 (docs at /docs)
	uvicorn fraudplat.serving.app:app --host 0.0.0.0 --port 8000

dashboard:          ## analyst dashboard on :8501
	streamlit run src/fraudplat/dashboard/app.py

replay:             ## stream held-out transactions into the running API (no Kafka needed)
	$(PY) -m fraudplat.streaming.producer --sink http --rate 0 --limit 5000 --send-labels

drift:              ## Evidently drift check on live traffic (+ retrain if triggered)
	$(PY) -m fraudplat.monitoring.drift --auto-retrain

drift-demo:         ## drift check on the offline test split, with an injected 3x amount shift
	$(PY) -m fraudplat.monitoring.drift --offline-test --shift 3

mlflow:             ## MLflow UI on :5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

bench:              ## latency benchmark
	$(PY) scripts/benchmark.py

docker-build:
	docker build -f docker/Dockerfile -t fraud-platform:latest .

up: docker-build    ## Kafka + API/consumer + dashboard + MLflow via docker compose
	docker compose up -d && docker compose --profile replay run --rm producer

k8s:                ## deploy to Minikube
	bash scripts/deploy_minikube.sh

clean:
	rm -rf data/processed artifacts mlflow.db mlruns
