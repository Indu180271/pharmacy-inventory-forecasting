# ML-Ops Inventory Forecasting - Starter Repo

This repository contains a starter ML-Ops pipeline for inventory forecasting (hospital consumables).
It includes sample code for ingestion, validation (Great Expectations), ETL (Airflow + DBT),
feature store (Feast), training (XGBoost + MLflow), serving (FastAPI), monitoring hooks (Evidently / Prometheus),
and CI/CD examples.

**Raw data (local):**

- `/mnt/data/inventory_transactions_raw.csv`
- `/mnt/data/skus_master_raw.csv`

## Quickstart (development)

1. Install requirements: `pip install -r requirements.txt`
2. Start local infra (optional): `docker-compose up -d` (edit docker-compose.yml)
3. Run initial ingestion scripts:
   - `python src/ingestion/ingest_skus.py`
   - `python src/ingestion/ingest_transactions.py`
4. Run Great Expectations checkpoint: `great_expectations checkpoint run transactions_checkpoint`
5. Train locally: `python src/models/train_xgb_mlflow.py --mlflow-uri http://localhost:5000`

## Local development (full stack)

Requirements: Docker, docker-compose

Start full stack:

```bash
bash scripts/run_local.sh

See docs in `/docs` (TBD).
# pharmacy-inventory-forecasting
CI Pipeline Test Trigger
