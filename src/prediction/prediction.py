"""
XGBoost Forecast Prediction Script
→ Generates forecast for an SKU
→ Logs prediction to MLflow
→ Saves forecast results to ClickHouse
"""

from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np
import mlflow

from src.config.settings import settings
from src.utils.clickhouse_client import clickhouse_client
from src.prediction.model_loader import load_model  

# ===================== MLflow CONFIG =====================
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
# If you have a registry URI, set it as well; otherwise this is optional
if getattr(settings, "MLFLOW_REGISTRY_URI", None):
    mlflow.set_registry_uri(settings.MLFLOW_REGISTRY_URI)


# ===================== FEATURE GENERATOR =====================
def generate_features_for_forecast(sku_id: int, horizon: int = 7) -> pd.DataFrame:
    """
    Build future feature rows for a given SKU and horizon (days ahead),
    based on recent history from ClickHouse.
    """
    query = f"""
        SELECT ds, qty_1d, rev_1d
        FROM feature_store.sku_demand_daily_features_top_10
        WHERE sku_id = '{sku_id}'
        ORDER BY ds DESC
        LIMIT 35
    """

    #  use our wrapper, returns a pandas DataFrame
    hist = clickhouse_client.execute_query(query)
    hist = hist.sort_values("ds")

    if len(hist) < 30:
        raise Exception(f"Not enough data for SKU {sku_id} (only {len(hist)} rows)")

    future_rows = []
    last_date = pd.to_datetime(hist["ds"].max())
    qty = list(hist["qty_1d"])
    rev = list(hist["rev_1d"])

    for i in range(1, horizon + 1):
        date = last_date + timedelta(days=i)

        row = {
            "ds": date,
            "dow": int(date.weekday()),
            "is_weekend": int(date.weekday() >= 5),
            "week_of_year": int(date.isocalendar().week),
            "month": int(date.month),
            "lag_qty_1d": float(qty[-1]),
            "lag_qty_7d": float(qty[-7] if len(qty) >= 7 else qty[-1]),
            "qty_7d": float(sum(qty[-7:])),
            "qty_28d": float(sum(qty[-28:])),
            "avg_qty_7d": float(sum(qty[-7:]) / 7),
            "avg_qty_28d": float(sum(qty[-28:]) / 28),
            "rev_7d": float(sum(rev[-7:])),
            "rev_28d": float(sum(rev[-28:])),
            "avg_rev_7d": float(sum(rev[-7:]) / 7),
            "avg_rev_28d": float(sum(rev[-28:]) / 28),
        }

        future_rows.append(row)

        # roll forward “history” to include this predicted day
        qty.append(row["qty_7d"] / 7)  # or row["predicted_qty"] once available
        rev.append(row["rev_7d"] / 7)

    return pd.DataFrame(future_rows)


# ===================== FORECAST FUNCTION =====================
def predict_sku(sku_id: int, horizon: int = 7) -> pd.DataFrame:
    """
    Forecast next 'horizon' days for a given SKU using the latest registered model
    from MLflow Model Registry. Also logs forecast and writes results to ClickHouse.
    """

    # ---------- Load model via shared loader (no models:/Production) ----------
    model = load_model(sku_id, stage=None)  # latest version

    # ---------- Generate future features ----------
    df_future = generate_features_for_forecast(sku_id, horizon)

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

    preds = model.predict(df_future[feature_cols])
    #df_future["predicted_qty"] = np.round(preds, 2)
    df_future["predicted_qty"] = np.round(preds).astype(int)


    # ------------------- MLflow LOGGING -------------------
    mlflow.set_experiment("sku_xgb_forecasting_predictions")

    with mlflow.start_run(run_name=f"forecast_sku_{sku_id}"):
        mlflow.log_param("sku_id", sku_id)
        mlflow.log_param("forecast_horizon", horizon)
        mlflow.log_metric("mean_prediction", float(df_future["predicted_qty"].mean()))

        out_csv = f"/tmp/forecast_sku_{sku_id}.csv"
        df_future.to_csv(out_csv, index=False)
        mlflow.log_artifact(out_csv)

        mlflow.log_text(
            df_future.to_string(index=False),
            f"forecast_sku_{sku_id}.txt",
        )

    # ------------------- Save Forecast to ClickHouse -------------------
    # use client.execute for DDL and inserts
    clickhouse_client.client.execute(
        """
        CREATE TABLE IF NOT EXISTS forecast_results (
            sku_id UInt32,
            date Date,
            predicted_qty Float32,
            created_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (sku_id, date)
        """
    )

    rows = [
        (int(sku_id), row["ds"], float(row["predicted_qty"]))
        for _, row in df_future.iterrows()
    ]

    clickhouse_client.client.execute(
        "INSERT INTO forecast_results (sku_id, date, predicted_qty) VALUES",
        rows,
    )

    print("Forecast saved to ClickHouse.")
    return df_future


# ======================== RUN TEST ========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SKU forecast prediction")
    parser.add_argument("--sku", type=int, required=True, help="SKU ID to forecast")
    parser.add_argument("--horizon", type=int, default=7, help="Forecast horizon (days)")
    args = parser.parse_args()

    print(predict_sku(args.sku, horizon=args.horizon))

