"""
Train XGBoost demand forecasting models for multiple SKUs
using features stored in ClickHouse and tracking using MLflow.
"""

import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import randint, uniform

from src.config.settings import settings
from src.utils.clickhouse_client import clickhouse_client

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from prometheus_client import start_http_server
from mlflow.models import infer_signature

# === MLflow Host Header Fix (Force correct Host in requests) ===
import requests

_original_request = requests.Session.request

def wrapped_request(self, method, url, *args, **kwargs):
    headers = kwargs.pop("headers", {})
    headers["Host"] = "mlflow"  # Required for Docker DNS access
    kwargs["headers"] = headers
    return _original_request(self, method, url, *args, **kwargs)

requests.Session.request = wrapped_request

mlflow.set_tracking_uri("http://mlflow:5002")
# ==============================================================


# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s:%(message)s",
)

# ------------------------------------------------------------------------------
# MLflow setup
# ------------------------------------------------------------------------------
#mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(settings.MLFLOW_REGISTRY_URI)
experiment_name = "xgb_forecasting"
mlflow.set_experiment(experiment_name)

# ------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------
def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error with safe handling of zeros.
    """
    eps = 1e-8
    errors = []
    for yt, yp in zip(y_true, y_pred):
        if abs(yt) < eps:
            errors.append(0.0 if abs(yp) < eps else 100.0)
        else:
            errors.append(abs(yt - yp) / abs(yt) * 100.0)
    return float(np.mean(errors))


# ------------------------------------------------------------------------------
# Hyperparameter tuning (optional)
# ------------------------------------------------------------------------------
def tune_params(X_train, y_train, n_iter: int = 5) -> Dict[str, Any]:
    logger.info("Running hyperparameter tuning...")

    param_dist = {
        "n_estimators": randint(100, 200),
        "max_depth": randint(3, 6),
        "learning_rate": uniform(0.05, 0.1),
        "subsample": uniform(0.8, 0.2),
        "colsample_bytree": uniform(0.8, 0.2),
        "min_child_weight": randint(1, 5),
    }

    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=settings.RANDOM_SEED,
    )

    tscv = TimeSeriesSplit(n_splits=2)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        random_state=settings.RANDOM_SEED,
        n_jobs=1,
    )

    search.fit(X_train, y_train)
    logger.info(f"Best parameters: {search.best_params_}")

    return search.best_params_


# ------------------------------------------------------------------------------
# Train a single SKU model
# ------------------------------------------------------------------------------
def train_sku_model(
    sku_id: int, df: pd.DataFrame, tune: bool = False
) -> Optional[Dict[str, Any]]:
    logger.info(f"\n========== TRAINING SKU {sku_id} ==========")

    sku = df[df["sku_id"] == sku_id].copy()
    if len(sku) < 90:
        logger.warning(f"Skipping SKU {sku_id} - insufficient samples ({len(sku)} rows)")
        return None

    sku["ds"] = pd.to_datetime(sku["ds"])
    sku = sku.sort_values("ds")

    feature_cols = [
        "dow",
        "is_weekend",
        "week_of_year",
        "month",
        "qty_7d",
        "qty_28d",
        "avg_qty_7d",
        "avg_qty_28d",
        "lag_qty_1d",
        "lag_qty_7d",
        "rev_7d",
        "rev_28d",
        "avg_rev_7d",
        "avg_rev_28d",
    ]

    X = sku[feature_cols]
    y = sku["qty_1d"]

    split = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    with mlflow.start_run(run_name=f"sku_{sku_id}") as run:
        params: Dict[str, Any] = {
            "objective": "reg:squarederror",
            "n_estimators": settings.N_ESTIMATORS,
            "max_depth": settings.MAX_DEPTH,
            "learning_rate": settings.LEARNING_RATE,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "random_state": settings.RANDOM_SEED,
        }

        if tune and len(X_train) > 100:
            try:
                best = tune_params(X_train, y_train)
                params.update(best)
            except Exception as e:
                logger.warning(f"Tuning failed for SKU {sku_id}: {e}")

        mlflow.log_params(params)

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mape = calculate_mape(y_test, y_pred)
        mlflow.log_metric("mape", mape)
        logger.info(f"SKU {sku_id} - MAPE: {mape:.2f}%")

        # ------------------------------------------------------------------
        # Save model under MODELS_PATH and log as MLflow artifact
        # ------------------------------------------------------------------
        model_dir = Path(settings.MODELS_PATH) / f"sku_{sku_id}"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "xgb_model.pkl"
        joblib.dump(model, model_path)

        # Log local model file to MLflow artifacts
        mlflow.log_artifact(str(model_path))

        # ------------------------------------------------------------------
        # Register model in MLflow Model Registry
        # ------------------------------------------------------------------
        artifact_path = f"sku_{sku_id}_model"
        mlflow.xgboost.log_model(model, artifact_path=artifact_path)

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/{artifact_path}"
        model_name = f"sku_forecast_model_{sku_id}"

        try:
            client = MlflowClient()
            result = mlflow.register_model(model_uri, model_name)
            client.transition_model_version_stage(
                name=model_name,
                version=result.version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info(
                f"Registered model {model_name} version {result.version} → Production"
            )
        except Exception as e:
            logger.warning(f"Model registry step failed for SKU {sku_id}: {e}")

        return {
            "sku_id": sku_id,
            "mape": mape,
            "model_path": str(model_path),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        }


# ------------------------------------------------------------------------------
# Train all SKUs
# ------------------------------------------------------------------------------
def train_all(tune: bool = False) -> None:
    logger.info("📥 Loading features from ClickHouse...")

    query = """
        SELECT *
        FROM feature_store.sku_demand_daily_features_top_10
        ORDER BY ds
    """

    df = clickhouse_client.execute_query(query)
    logger.info(f"Loaded {len(df)} rows from ClickHouse")

    if df.empty:
        logger.error("Feature table is empty. Aborting training.")
        return

    df["sku_id"] = df["sku_id"].astype(int)
    available_skus = sorted(df["sku_id"].unique().tolist())

    if settings.SKU_LIST:
        configured = [int(s) for s in settings.SKU_LIST]
        sku_ids = [s for s in configured if s in available_skus]
        if not sku_ids:
            sku_ids = available_skus
    else:
        sku_ids = available_skus

    logger.info(f"Training models for SKUs: {sku_ids}")

    results = []
    for sku in sku_ids:
        try:
            r = train_sku_model(sku, df, tune=tune)
            if r:
                results.append(r)
        except Exception as e:
            logger.error(f"Error training SKU {sku}: {e}", exc_info=True)

    if results:
        summary = pd.DataFrame(results).sort_values("mape")
        summary_path = Path(settings.MODELS_PATH) / "training_summary.csv"
        summary.to_csv(summary_path, index=False)

        # Log summary CSV to MLflow artifacts
        mlflow.log_artifact(str(summary_path))

        logger.info(f"\n Training Summary Saved → {summary_path}")
        logger.info(summary.to_string(index=False))

    logger.info(" All SKU training complete!")


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Expose Prometheus metrics
    start_http_server(8000)

    # Run training once
    train_all(tune=False)

    # Keep process alive so Prometheus can continue scraping metrics
    print("Training finished. Keeping metrics endpoint alive on :8000 for Prometheus.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down training process.")
