#!/usr/bin/env bash
set -euo pipefail

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5001}"
MODEL_URI="${MODEL_URI:-models:/cs2-skin-price-predictor@production}"
IMAGE_NAME="${IMAGE_NAME:-cs2-mlflow-serving:v1}"
ENV_MANAGER="${ENV_MANAGER:-local}"

echo "Building MLflow serving Docker image..."
echo "Tracking URI : ${MLFLOW_TRACKING_URI}"
echo "Model URI    : ${MODEL_URI}"
echo "Image Name   : ${IMAGE_NAME}"
echo "Env Manager  : ${ENV_MANAGER}"

mlflow models build-docker \
  --model-uri "${MODEL_URI}" \
  --name "${IMAGE_NAME}" \
  --env-manager "${ENV_MANAGER}"

echo "Docker image ${IMAGE_NAME} built successfully from MLflow Model Registry."