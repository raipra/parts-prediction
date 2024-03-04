#!/bin/bash

mlflow server \
  --backend-store-uri "${MLFLOW_BACKEND_STORE}" \
  --default-artifact-root "${exit}" \
  --host 0.0.0.0 \
  --port "6101"
