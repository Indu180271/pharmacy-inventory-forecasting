import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # -------------------------
    # CLICKHOUSE CONFIG
    # -------------------------
    CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
    CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", 9000))
    CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "clickuser")
    CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "clickpass")
    CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "feature_store")

    # -------------------------
    # MLflow Tracking
    # -------------------------
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5002")
    MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", MLFLOW_TRACKING_URI)

    # -------------------------
    # Model Hyperparameters
    # -------------------------
    RANDOM_SEED = 42
    N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", 200))
    MAX_DEPTH = int(os.getenv("MAX_DEPTH", 6))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.1))

    raw_sku_list = os.getenv("SKU_LIST")  # no default

    if raw_sku_list:
        SKU_LIST = [
            int(s.strip())
            for s in raw_sku_list.split(",")
            if s.strip()
        ]
    else:
        SKU_LIST = None

    # -------------------------
    # Model Save Path
    # -------------------------
    MODELS_PATH = os.getenv("MODELS_PATH", "/app/models")


settings = Settings()
