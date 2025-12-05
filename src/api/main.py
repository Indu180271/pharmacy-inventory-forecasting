from fastapi import FastAPI, UploadFile, File, HTTPException, Query
import pandas as pd
from io import StringIO

# Correct imports based on your actual inference.py
from src.api.inference import (
    get_latest_features_for_sku,     # FIXED NAME
    predict_single_sku_with_ci,      # FIXED NAME
    predict_batch_from_df            # this one is correct
)

from src.prediction.prediction import predict_sku   # your forecasting logic

app = FastAPI(title="SKU Forecasting API")


@app.get("/")
def root():
    return {"status": "running"}


# ----------- Single Prediction (using latest ClickHouse features) -----------
@app.get("/predict/single/{sku_id}")
def predict_single(sku_id: int):
    try:
        features = get_latest_features_for_sku(sku_id)
    except Exception as e:
        raise HTTPException(400, f"Feature retrieval failed: {str(e)}")

    try:
        result = predict_single_sku_with_ci(sku_id, features)
        return {"sku_id": sku_id, **result, "source": "ClickHouse_latest"}
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")


# ----------- Multi-Day Forecast -----------
@app.get("/predict/forecast/{sku_id}")
def forecast(sku_id: int, horizon: int = Query(7, ge=1, le=60)):
    try:
        df = predict_sku(sku_id, horizon)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(500, f"Forecast failed: {str(e)}")


# ----------- CSV Batch Prediction -----------
@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception:
        raise HTTPException(400, "Invalid CSV format")

    try:
        return predict_batch_from_df(df)
    except Exception as e:
        raise HTTPException(500, f"Batch prediction failed: {str(e)}")

