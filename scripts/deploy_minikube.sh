#!/usr/bin/env bash
# Build the image inside Minikube's Docker daemon and deploy everything.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f artifacts/xgb_model.json ] || { echo "Train first: make train"; exit 1; }

minikube status >/dev/null 2>&1 || minikube start --cpus=4 --memory=8g
eval "$(minikube docker-env)"
docker build -f docker/Dockerfile -t fraud-platform:latest .

kubectl apply -f k8s/
kubectl -n fraud rollout status statefulset/kafka --timeout=180s
kubectl -n fraud rollout status deployment/fraud-api --timeout=300s
kubectl -n fraud rollout status deployment/fraud-dashboard --timeout=180s

kubectl -n fraud create job "replay-$(date +%s)" --from=cronjob/replay-producer
echo
echo "API:        $(minikube service -n fraud fraud-api --url)"
echo "Dashboard:  $(minikube service -n fraud fraud-dashboard --url)"
