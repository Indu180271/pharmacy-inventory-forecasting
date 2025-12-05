# src/prediction/model_loader.py

import logging
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Optional in-memory cache
_MODEL_CACHE: Dict[str, Any] = {}

def _cache_key(sku_id: int) -> str:
    return f"{sku_id}"


def load_model(sku_id: int, stage: str = None):
    """
    Load the latest model version for a SKU from MLflow.
    If stage is provided, load that stage (Production/Staging).
    If stage=None, load the latest version regardless of stage.
    """

    client = MlflowClient()
    model_name = f"sku_forecast_model_{sku_id}"
    key = _cache_key(sku_id)

    # Return from cache if exists
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    # ---------- CASE 1: stage is provided ----------
    if stage:
        try:
            versions = client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                raise HTTPException(404, f"No version in stage '{stage}' for {model_name}")
            uri = versions[0].source
            logger.info(f"Loading model from stage URI: {uri}")
            model = mlflow.pyfunc.load_model(uri)
            _MODEL_CACHE[key] = model
            return model
        except Exception as e:
            raise HTTPException(500, f"Failed to load staged model: {e}")

    # ---------- CASE 2: NO STAGE – load latest version ----------
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise HTTPException(404, f"No versions found for model: {model_name}")

        # latest version = highest version number
        latest = sorted(versions, key=lambda v: int(v.version))[-1]
        uri = latest.source
        logger.info(f"Loading latest model version from URI: {uri}")

        model = mlflow.pyfunc.load_model(uri)
        _MODEL_CACHE[key] = model
        return model

    except Exception as e:
        raise HTTPException(500, f"Failed to load model for {model_name}: {e}")

