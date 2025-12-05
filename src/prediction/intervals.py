# src/prediction/intervals.py

import logging
from typing import Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient
from fastapi import HTTPException

logger = logging.getLogger(__name__)


def _get_latest_model_version(
    client: MlflowClient, model_name: str, stage: Optional[str]
):
    """
    Helper to fetch the latest model version for a given model name.
    If stage is provided, use that stage; otherwise use the highest version number.
    """
    if stage:
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            raise HTTPException(
                status_code=404,
                detail=f"No version in stage '{stage}' for model '{model_name}'",
            )
        return versions[0]

    # No stage -> use latest version by version number
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise HTTPException(
            status_code=404,
            detail=f"No versions found for model '{model_name}'",
        )
    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    return latest


def compute_mape_ci(
    sku_id: int,
    prediction: float,
    stage: Optional[str] = None,
    z: float = 1.96,  # approx 95% interval
) -> Tuple[float, float, float]:
    """
    Compute a simple MAPE-based confidence interval around `prediction`
    for a given SKU.

    - Reads the logged `mape` metric from MLflow for the latest model version.
    - Does NOT load the model artifacts at all (avoids 'models:/...' URIs).
    """
    client = MlflowClient()
    model_name = f"sku_forecast_model_{sku_id}"

    try:
        mv = _get_latest_model_version(client, model_name, stage)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error resolving model version for CI")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resolve model version for '{model_name}': {e}",
        )

    run_id = mv.run_id

    try:
        run = client.get_run(run_id)
        mape = float(run.data.metrics.get("mape", 20.0))  # default 20% if missing
    except Exception as e:
        logger.exception("Error reading MAPE metric from MLflow")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read MAPE metric for run {run_id}: {e}",
        )

    # Convert MAPE (%) into an approximate symmetric interval around prediction.
    # Very simple approximation: error_pct = mape/100, interval = prediction ± z * error_pct * prediction
    error_pct = mape / 100.0
    half_width = z * error_pct * abs(prediction)

    lower = prediction - half_width
    upper = prediction + half_width

    return lower, upper, mape

