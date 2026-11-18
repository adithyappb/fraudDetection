# One image, several roles (api | dashboard | producer | drift | train) chosen by command.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app/src
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt

COPY configs configs
COPY feature_repo feature_repo
COPY src src
COPY scripts scripts
# Trained artifacts + Feast data are baked in by `make docker-build` (run `make train` first).
# In Kubernetes they can instead be mounted from a PVC at /app/artifacts and /app/data.
COPY artifacts artifacts
COPY data/processed data/processed

RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000 8501
HEALTHCHECK --interval=15s --timeout=3s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "fraudplat.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
