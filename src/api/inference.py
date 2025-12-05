# src/api/inference.py
from fastapi import HTTPException
import pandas as pd
from typing import Dict, List, Any
from src.prediction.model_loader import load_model
from src.prediction.intervals import compute_mape_ci
from src.utils.clickhouse_client import clickhouse_client

FEATURE_COLS = [
    "dow","is_weekend","week_of_year","month",
    "qty_7d","qty_28d","avg_qty_7d","avg_qty_28d",
    "lag_qty_1d","lag_qty_7d",
    "rev_7d","rev_28d","avg_rev_7d","avg_rev_28d"
]

INT_COLS = {"dow", "is_weekend", "week_of_year", "month"}


def validate_features_dict(features: Dict[str, Any]):
    missing = [c for c in FEATURE_COLS if c not in features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")


def predict_single_sku_with_ci(sku_id: int, features: Dict[str, Any], stage: str = None) -> Dict[str, Any]:
    """
    Predict for single SKU given features dict. Returns dict with CI and metadata.
    """
    validate_features_dict(features)

    # Cast integer fields
    for col in INT_COLS:
        try:
            features[col] = int(features[col])
        except:
            features[col] = 0

    model = load_model(sku_id, stage=None)
    df = pd.DataFrame([features], columns=FEATURE_COLS)

    try:
        prediction = float(model.predict(df)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    lower, upper, mape_used = compute_mape_ci(sku_id, prediction, stage=stage)
    return {
        "prediction": prediction,
        "lower_95": lower,
        "upper_95": upper,
        "method": "mape_based",
        "mape_used_pct": mape_used,
        "model_name": f"sku_forecast_model_{sku_id}",
        "model_stage": stage
    }


def predict_batch_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Accept df with columns: sku_id + FEATURE_COLS. Returns list of dict results.
    """
    required = ["sku_id"] + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    results = []

    for _, row in df.iterrows():
        sku_id = int(row["sku_id"])
        features = row[FEATURE_COLS].to_dict()

        # Cast integer fields
        for col in INT_COLS:
            try:
                features[col] = int(features[col])
            except:
                features[col] = 0

        try:
            res = predict_single_sku_with_ci(sku_id, features)
            results.append({"sku_id": sku_id, **res})
        except HTTPException as e:
            results.append({"sku_id": sku_id, "error": e.detail})
        except Exception as e:
            results.append({"sku_id": sku_id, "error": str(e)})

    return results


def get_latest_features_for_sku(sku_id: int) -> Dict[str, Any]:
    """
    Query ClickHouse for latest feature row for this SKU and return a features dict.
    """
    query = f"""
        SELECT *
        FROM feature_store.sku_demand_daily_features_top_10
        WHERE sku_id = '{int(sku_id)}'
        ORDER BY ds DESC
        LIMIT 1
    """

    df = clickhouse_client.execute_query(query)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No feature row found for SKU {sku_id}")

    row = df.iloc[0].to_dict()

    # Include only model feature columns
    features = {
        k: float(row[k]) if isinstance(row.get(k), (int, float)) else row.get(k)
        for k in FEATURE_COLS
    }

    return features
