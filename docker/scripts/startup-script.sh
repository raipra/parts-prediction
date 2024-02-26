#!/usr/bin/env bash
sudo apt update
pip3 install mlflow psycopg2-binary
sudo apt -y install apache2
cat <<EOF > /var/www/html/index.html
<html><body><p>Linux startup script from Cloud Storage.</p></body></html>
EOF

export PROD_MLFLOW_SERVER_PORT=5000

export MLFLOW_ARTIFACT_STORE=gs://pp-mlflow-artifacts/mlflow-artifact
export POSTGRESUSER=postgres
export POSTGRESPWD=postgres
export POSTGRESDB=postgres
export POSTGRES_IP=localhost
export MLFLOW_BACKEND_STORE=postgresql://${POSTGRESUSER}:${POSTGRESPWD}@${POSTGRES_IP}:5432/${POSTGRESDB}
export MLFLOW_TRACKING_URI=http://127.0.0.1:${PROD_MLFLOW_SERVER_PORT}
mlflow server \
--backend-store-uri "${MLFLOW_BACKEND_STORE}" \
--default-artifact-root "${MLFLOW_ARTIFACT_STORE}" \
--host 0.0.0.0 \
--port "${PROD_MLFLOW_SERVER_PORT}"

tail -F anythoig
