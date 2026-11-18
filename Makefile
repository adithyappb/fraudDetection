PY ?= python
export PYTHONPATH := src
export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: install data download version train metrics test orthanc serve send web up clean

install:        ## Python deps (CPU torch) + viewer deps
	$(PY) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt
	cd web && npm ci

download:       ## real NIH ChestX-ray14 (Kaggle) + MSD Task09 Spleen into data/raw
	bash scripts/download_data.sh

data:           ## use real data in data/raw if present, else render phantoms; then version it with DVC
	$(PY) -m medai.acquire
	dvc add data/raw

train:          ## DVC pipeline: preprocess -> train_cls -> train_seg -> evaluate (only changed stages rerun)
	dvc repro

metrics:        ## test-set metrics table (tracked by DVC; raw: dvc metrics show)
	$(PY) scripts/show_metrics.py

test:
	$(PY) -m pytest -q -W ignore

orthanc:        ## local Orthanc (apt install orthanc) on :8042 / DICOM :4242
	bash scripts/run_orthanc.sh

serve:          ## API + viewer on :8000, with the Orthanc worker in-process
	RUN_WORKER=1 uvicorn medai.service.api:app --host 0.0.0.0 --port 8000

send:           ## act as a modality: C-STORE 12 chest X-rays + 2 CTs to Orthanc
	$(PY) scripts/send_studies.py --cxr 12 --ct 2

web:            ## rebuild the React viewer
	cd web && npm run build

up:             ## everything in Docker (Orthanc + worker + API/viewer)
	docker compose up -d --build

clean:
	rm -rf data/processed data/service models metrics/*.json
