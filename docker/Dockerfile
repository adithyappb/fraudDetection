# Stage 1: build the React viewer
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm ci
COPY web/ .
RUN npm run build

# Stage 2: Python service (API + Orthanc worker), CPU-only PyTorch
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app/src
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
COPY params.yaml ./
COPY src src
COPY scripts scripts
COPY metrics metrics
COPY models models
COPY --from=web /web/dist web/dist
RUN useradd -m app && mkdir -p data/service && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD curl -fs http://localhost:8000/api/health || exit 1
CMD ["uvicorn", "medai.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
