# src/api/app.py
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
import pandas as pd
from io import StringIO
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
import time
from src.api.inference import (
    predict_single_sku_with_ci,
    predict_batch_from_df,
    get_latest_features_for_sku,
)
from src.prediction.forecasting import forecast_iterative  # your existing file

logger = logging.getLogger(__name__)
app = FastAPI(title="SKU Demand Forecasting API")


# ---- PROMETHEUS METRICS ----
# Automatic instrumentation
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Custom metrics
inference_success_total = Counter(
    "inference_success_total",
    "Total number of successful inference calls"
)

inference_failure_total = Counter(
    "inference_failure_total",
    "Total number of failed inference calls"
)

inference_latency = Histogram(
    "inference_latency_seconds",
    "Latency of inference calls",
    buckets=(0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 5)
)


@app.get("/")
def root():
    return {"message": "SKU Forecasting API Running"}

# Single SKU prediction using ClickHouse latest feature row (no JSON input)
@app.get("/predict/single/{sku_id}")
def predict_single_from_store(sku_id: int):
    try:
        features = get_latest_features_for_sku(sku_id)
        res = predict_single_sku_with_ci(sku_id, features)
        return {"sku_id": sku_id, **res, "feature_source": "ClickHouse_latest"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in single prediction")
        raise HTTPException(status_code=500, detail=str(e))

# Forecast endpoint (iterative forecasting)
@app.get("/predict/forecast/{sku_id}")
def forecast(sku_id: int, horizon: int = Query(7, ge=1, le=90)):
    try:
        df = forecast_iterative(sku_id, horizon=horizon)
        # Optionally compute CI per day using SKU mape (apply same width)
        # We'll attach mape-based CI using first day's pred
        if not df.empty:
            first_pred = float(df.iloc[0]["predicted_qty"])
            # import compute_mape_ci locally to avoid circular imports
            from src.prediction.intervals import compute_mape_ci
            lower, upper, mape_used = compute_mape_ci(sku_id, first_pred)
            df["lower_95"] = lower
            df["upper_95"] = upper
        return df.to_dict(orient="records")
    except Exception as e:
        logger.exception("Error forecasting")
        raise HTTPException(status_code=500, detail=str(e))

# Batch predictions via CSV upload
@app.post("/predict/csv")
async def predict_from_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    results = predict_batch_from_df(df)
    return results

# Optionally - CSV download endpoint
@app.post("/predict/csv/download")
async def predict_csv_download(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    results = predict_batch_from_df(df)
    out_df = pd.DataFrame(results)
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    return StreamingResponse(io.BytesIO(csv_bytes), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=predictions.csv"})
                             
                             
app.get("/")
def root():
    return {"message": "SKU Forecasting API Running"}


@app.get("/predict/single/{sku_id}")
def predict_single_from_store(sku_id: int):
    start = time.time()
    try:
        features = get_latest_features_for_sku(sku_id)
        res = predict_single_sku_with_ci(sku_id, features)
        inference_success_total.inc()
        return {"sku_id": sku_id, **res}
    except Exception as e:
        inference_failure_total.inc()
        raise
    finally:
        inference_latency.observe(time.time() - start)

