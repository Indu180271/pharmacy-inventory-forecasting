import logging
from pathlib import Path
from fastapi import HTTPException
import joblib
import pandas as pd
from src.config.settings import settings

logger = logging.getLogger(__name__)

# Same feature columns used in training
FEATURE_COLS = [
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


def load_model(sku_id: int):
    """Load trained XGBoost model for given SKU"""
    model_path = Path(settings.MODELS_PATH) / f"sku_{sku_id}/xgb_model.pkl"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found for SKU {sku_id}")

    return joblib.load(model_path)


def predict_single_sku(sku_id: int, features: dict) -> float:
    """Predict single value for one SKU"""
    model = load_model(sku_id)

    # Convert input to DataFrame
    df = pd.DataFrame([features], columns=FEATURE_COLS)

    # Predict
    prediction = model.predict(df)[0]

    return float(prediction)


def predict_batch(batch: list):
    """Predict batch values for multiple SKUs"""
    results = []
    for item in batch:
        sku_id = item["sku_id"]
        features = item["features"]
        pred = predict_single_sku(sku_id, features)

        results.append({
            "sku_id": sku_id,
            "prediction": pred
        })

    return results
