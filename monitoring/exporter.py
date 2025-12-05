from prometheus_client import start_http_server, Gauge
from mlflow.tracking import MlflowClient
import time
import os

# Use tracking URI from env if provided
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5002")
client = MlflowClient(tracking_uri=tracking_uri)

model_version_gauge = Gauge(
    "mlflow_model_latest_version",
    "Latest model version in registry",
    ["model_name"]
)

model_update_age = Gauge(
    "mlflow_model_update_age_seconds",
    "Seconds since model last updated",
    ["model_name"]
)

def collect():
    # MLflow 2.x API: search_registered_models()
    registered = client.search_registered_models()  # returns list of RegisteredModel
    for rm in registered:
        model_name = rm.name

        # rm.latest_versions is a list of ModelVersion objects
        versions = [int(v.version) for v in rm.latest_versions]
        latest_version = max(versions)
        model_version_gauge.labels(model_name).set(latest_version)

        # last_updated_timestamp is in milliseconds
        last_update_ts = max([v.last_updated_timestamp for v in rm.latest_versions]) / 1000.0
        age_seconds = time.time() - last_update_ts
        model_update_age.labels(model_name).set(age_seconds)

if __name__ == "__main__":
    # Expose metrics on :9009
    start_http_server(9009)
    while True:
        collect()
        time.sleep(30)
