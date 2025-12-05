import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from src.utils.clickhouse_client import clickhouse_client
from src.config.settings import settings
import mlflow

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s:%(message)s"
)


def calculate_mape(y_true, y_pred):
    eps = 1e-8
    errors = []
    for yt, yp in zip(y_true, y_pred):
        if abs(yt) < eps:
            errors.append(0.0 if abs(yp) < eps else 100.0)
        else:
            errors.append(abs(yt - yp) / abs(yt) * 100.0)
    return float(np.mean(errors))


def load_local_model(sku_id: int) -> Optional[xgb.XGBRegressor]:
    model_path = Path(settings.MODELS_PATH) / f"sku_{sku_id}/xgb_model.pkl"
    if model_path.exists():
        logger.info(f"📦 Loading local model for SKU {sku_id} from {model_path}")
        return joblib.load(model_path)
    return None


def load_registry_model(sku_id: int) -> Optional[xgb.XGBRegressor]:
    model_name = f"sku_forecast_model_{sku_id}"
    model_uri = f"models:/{model_name}/Production"
    try:
        logger.info(f"☁ Loading MLflow Registry model: {model_uri}")
        return mlflow.xgboost.load_model(model_uri)
    except Exception as e:
        logger.warning(f"⚠ Registry model not found for SKU {sku_id}: {e}")
        return None


def evaluate_sku_model(sku_id: int, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    logger.info(f"\n========== EVALUATING SKU {sku_id} ==========")

    sku = df[df["sku_id"] == sku_id].copy()
    if sku.empty:
        logger.warning(f"No data available for SKU {sku_id}")
        return None

    sku["ds"] = pd.to_datetime(sku["ds"])
    sku = sku.sort_values("ds")

    feature_cols = [
        "dow", "is_weekend", "week_of_year", "month",
        "qty_7d", "qty_28d", "avg_qty_7d", "avg_qty_28d",
        "lag_qty_1d", "lag_qty_7d",
        "rev_7d", "rev_28d", "avg_rev_7d", "avg_rev_28d"
    ]

    X = sku[feature_cols]
    y = sku["qty_1d"]

    split = int(0.8 * len(X))
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    model = load_local_model(sku_id)

    if model is None:
        model = load_registry_model(sku_id)

    if model is None:
        logger.error(f"❌ No model found for SKU {sku_id} (local & registry failed)")
        return None

    y_pred = model.predict(X_test)

    mape = calculate_mape(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    logger.info(f"📊 SKU {sku_id} → MAPE={mape:.2f}%, MAE={mae:.2f}, RMSE={rmse:.2f}")

    with mlflow.start_run(run_name=f"eval_sku_{sku_id}"):
        mlflow.log_metric("eval_mape", mape)
        mlflow.log_metric("eval_mae", mae)
        mlflow.log_metric("eval_rmse", rmse)
        mlflow.set_tag("stage", "evaluation")
        mlflow.set_tag("sku_id", sku_id)

    return {
        "sku_id": sku_id,
        "mape": mape,
        "mae": mae,
        "rmse": rmse,
        "rows_tested": len(y_test),
    }


def evaluate_all() -> None:
    logger.info("📥 Loading features from ClickHouse for evaluation...")

    query = "SELECT * FROM feature_store.sku_demand_daily_features_top_10 ORDER BY ds"
    df = clickhouse_client.execute_query(query)

    if df.empty:
        logger.error("❌ Feature table empty. Aborting evaluation.")
        return

    df["sku_id"] = df["sku_id"].astype(int)
    sku_ids = sorted(df["sku_id"].unique())
    logger.info(f"Evaluating SKUs: {sku_ids}")

    results = []
    for sku in sku_ids:
        try:
            r = evaluate_sku_model(sku, df)
            if r:
                results.append(r)
        except Exception as e:
            logger.error(f"❌ Evaluation error for SKU {sku}: {e}", exc_info=True)

    if results:
        summary = pd.DataFrame(results).sort_values("mape")
        summary_path = Path(settings.MODELS_PATH) / "evaluation_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info(f"📄 Evaluation Summary saved → {summary_path}")
        logger.info("\n" + summary.to_string(index=False))

    logger.info("🎯 Evaluation completed successfully!")


if __name__ == "__main__":
    evaluate_all()

