FROM ghcr.io/mlflow/mlflow:latest

# Create directory for mlflow database and artifacts
WORKDIR /mlflow

# Run mlflow server
CMD mlflow server \
    --host 0.0.0.0 \
    --port ${PORT:-5000} \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root mlflow-artifacts:/ \
    --artifacts-destination /mlflow/artifacts \
    --allowed-hosts "*" \
    --serve-artifacts
