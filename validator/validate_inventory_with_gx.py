#!/usr/bin/env python3
"""
FINAL simplified GE validation pipeline (no imputations / no dropping)
- Read SKU from Postgres
- Read Inventory from TimescaleDB
- Normalize column names only
- Run one EDA (before validation) per table
- Run Great Expectations on the normalized dataframe
- PASSED -> ClickHouse (raw_zone)
- FAILED -> TimescaleDB (recreated rejected table)
- Exports valid CSVs and EDA artifacts
- NO OPENLINEAGE
"""

import os
import sys
import re
from datetime import datetime
import math
import numpy as np
import pandas as pd
import warnings
from typing import Optional

from sqlalchemy import create_engine, inspect, Table, Column, MetaData, Integer, Float, String, DateTime, Boolean, text
from sqlalchemy.exc import SQLAlchemyError
from clickhouse_driver import Client as CHClient

import uuid

import great_expectations as ge
from great_expectations.data_context import FileDataContext
from great_expectations.core.batch import RuntimeBatchRequest

# Optional EDA: sweetviz
try:
    import sweetviz as sv
except Exception:
    sv = None

# ======================================================
# CONFIG
# ======================================================
PG_HOST = os.getenv("PG_HOST", "marquez-db")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "marquez")
PG_USER = os.getenv("PG_USER", "marquez")
PG_PASSWORD = os.getenv("PG_PASSWORD", "marquez")
POSTGRES_URI = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"

TS_HOST = os.getenv("TS_HOST", "marquez-db")
TS_PORT = int(os.getenv("TS_PORT", "5432"))
TS_DB = os.getenv("TS_DB", "marquez")
TS_USER = os.getenv("TS_USER", "marquez")
TS_PASSWORD = os.getenv("TS_PASSWORD", "marquez")
TIMESCALE_URI = f"postgresql+psycopg2://{TS_USER}:{TS_PASSWORD}@{TS_HOST}:{TS_PORT}/{TS_DB}"


CH_HOST = os.getenv("CH_HOST", "clickhouse")
CH_PORT = int(os.getenv("CH_PORT", "9000"))
CH_DB = os.getenv("CH_DB", "raw_zone")
CH_USER = os.getenv("CH_USER", "clickuser")
CH_PASS = os.getenv("CH_PASS", "clickpass")

GX_CONTEXT_ROOT = os.getenv("GX_CONTEXT_ROOT", "./gx")
SKU_SUITE_NAME = os.getenv("SKU_SUITE_NAME", "sku_master")
INV_SUITE_NAME = os.getenv("INV_SUITE_NAME", "inventory_transactions")

VALID_OUTPUT_DIR = "./validated_output"
EDA_OUTPUT_DIR = "./eda_reports"
EDA_DIR = os.path.join(EDA_OUTPUT_DIR, "before_validation")

os.makedirs(VALID_OUTPUT_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

CHUNK_SIZE = 10000
TS_CHUNK_SIZE = 10000

# ======================================================
# DB CLIENTS
# ======================================================
engine_pg = create_engine(POSTGRES_URI, pool_pre_ping=True)
engine_ts = create_engine(TIMESCALE_URI, pool_pre_ping=True)
ch = CHClient(host=CH_HOST, port=CH_PORT, user=CH_USER, password=CH_PASS)

# GE context
context = FileDataContext(context_root_dir=GX_CONTEXT_ROOT)
print("[INFO] GE context loaded")

# ======================================================
# EDA
# ======================================================
def run_eda_and_save(df, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    try:
        df.describe(include="all").transpose().to_csv(os.path.join(out_dir, f"{name}_describe.csv"))
        df.isna().sum().to_csv(os.path.join(out_dir, f"{name}_nulls.csv"))
    except Exception as e:
        print(f"[WARN] EDA CSV failed: {e}")

    if sv:
        try:
            report = sv.analyze(df)
            html_path = os.path.join(out_dir, f"{name}_sweetviz.html")
            report.show_html(html_path, open_browser=False)
            print(f"[INFO] Sweetviz report saved → {html_path}")
        except Exception as e:
            print(f"[WARN] Sweetviz failed: {e}")

# ======================================================
# CLICKHOUSE HELPERS
# ======================================================
def ensure_clickhouse_database():
    ch.execute(f"CREATE DATABASE IF NOT EXISTS {CH_DB}")
    ch.execute(f"USE {CH_DB}")
    print("[INFO] Using ClickHouse DB:", CH_DB)

def recreate_clickhouse_table(df: pd.DataFrame, table: str):
    """
    Recreate table with correct ClickHouse column types (works for tz-aware dtypes).
    """

    ch.execute(f"DROP TABLE IF EXISTS {table}")

    cols = []
    for c, dt in df.dtypes.items():
        dt_str = str(dt)

        # Detect datetime64, even with timezone
        if "datetime64" in dt_str:
            coltype = "DateTime64(3)"

        elif "int" in dt_str:
            coltype = "Int64"

        elif "float" in dt_str:
            coltype = "Float64"

        elif "bool" in dt_str:
            coltype = "UInt8"

        else:
            coltype = "String"

        cols.append(f"`{c}` {coltype}")

    create_sql = f"""
        CREATE TABLE {table} (
            {', '.join(cols)}
        )
        ENGINE = MergeTree()
        ORDER BY tuple()
    """

    ch.execute(create_sql)
    print(f"[INFO] Recreated table → {table}")


def insert_df_clickhouse(df: pd.DataFrame, table: str):
    """
    Safe insert → handles Timestamp errors by converting properly
    and ensures ClickHouse receives plain python datetime objects.
    """
    if df.empty:
        print("[INFO] Nothing to insert.")
        return

    df2 = df.copy()

    for c in df2.columns:
        dt_str = str(df2[c].dtype)

        # Normalize ALL datetimes
        if "datetime64" in dt_str or "Timestamp" in dt_str:
            df2[c] = pd.to_datetime(df2[c], errors="coerce")

            # Remove timezone
            try:
                df2[c] = df2[c].dt.tz_localize(None)
            except:
                pass

            # Convert to python datetime
            df2[c] = df2[c].apply(
                lambda x: x.to_pydatetime() if pd.notnull(x) else None
            )

    rows = list(df2.itertuples(index=False, name=None))
    cols = ", ".join([f"`{c}`" for c in df2.columns])
    query = f"INSERT INTO {table} ({cols}) VALUES"

    ch.execute(query, rows)
    print(f"[INFO] Inserted {len(rows)} rows into {table}")

# ======================================================
# TIMESCALE HELPERS
# ======================================================
def recreate_timescale_rejected(df: pd.DataFrame, table: str):
    dtype_map = {
        "int64": Integer,
        "float64": Float,
        "object": String,
        "datetime64[ns]": DateTime,
        "bool": Boolean
    }

    metadata = MetaData()

    with engine_ts.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table}"))

    if df.empty:
        cols = [Column("_dummy", String)]
    else:
        cols = [Column(c, dtype_map.get(str(dt), String)) for c, dt in df.dtypes.items()]

    tbl = Table(table, metadata, *cols)
    metadata.create_all(engine_ts)

# ======================================================
# GE VALIDATION
# ======================================================
def run_ge(df, suite_name):
    if df.empty:
        return df, pd.DataFrame()

    df2 = df.copy()
    df2["_idx"] = df2.index

    suite = context.get_expectation_suite(suite_name)

    batch_request = RuntimeBatchRequest(
        datasource_name="default_pandas_datasource",
        data_connector_name="default_runtime_data_connector",
        data_asset_name=f"{suite_name}_asset",
        runtime_parameters={"batch_data": df2},
        batch_identifiers={"run_id": str(uuid.uuid4())},
    )

    validator = context.get_validator(batch_request=batch_request, expectation_suite=suite)
    result = validator.validate(result_format={"result_format": "COMPLETE"})

    failing = set()

    for r in result.get("results", []):
        idxs = r.get("result", {}).get("unexpected_index_list", [])
        failing.update(idxs)

    passed = df2.drop(list(failing), errors="ignore").drop(columns=["_idx"])
    failed = df2.loc[list(failing)].drop(columns=["_idx"])

    return passed.reset_index(drop=True), failed.reset_index(drop=True)

# ======================================================
# SKU PIPELINE
# ======================================================
def process_sku_master():
    print("[INFO] Processing SKU master...")

    try:
        df = pd.read_sql_table("sku_master", engine_pg)
    except:
        df = pd.read_sql_table("sku_master", engine_ts)

    if df.empty:
        print("[INFO] SKU master empty.")
        return

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    run_eda_and_save(df, "sku_master_before_validation", EDA_DIR)

    passed, failed = run_ge(df, SKU_SUITE_NAME)
    print(f"[INFO] SKU — Passed={len(passed)} Failed={len(failed)}")

    passed.to_csv(os.path.join(VALID_OUTPUT_DIR, "sku_master_valid.csv"), index=False)

    table = f"{CH_DB}.sku_master"
    recreate_clickhouse_table(passed, table)
    insert_df_clickhouse(passed, table)

    if not failed.empty:
        rej_table = "sku_master_rejected"
        recreate_timescale_rejected(failed, rej_table)
        failed.to_sql(rej_table, engine_ts, index=False, chunksize=TS_CHUNK_SIZE)

# ======================================================
# INVENTORY PIPELINE
# ======================================================
def process_inventory_transactions():
    print("[INFO] Processing inventory_transactions...")

    df = pd.read_sql_table("inventory_transactions", engine_ts)
    if df.empty:
        print("[INFO] Inventory empty.")
        return

    df.columns = [c.lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    run_eda_and_save(df, "inventory_before_validation", EDA_DIR)

    passed, failed = run_ge(df, INV_SUITE_NAME)
    print(f"[INFO] Inventory — Passed={len(passed)} Failed={len(failed)}")

    passed.to_csv(os.path.join(VALID_OUTPUT_DIR, "inventory_transactions_valid.csv"), index=False)

    table = f"{CH_DB}.inventory_transactions"
    recreate_clickhouse_table(passed, table)
    insert_df_clickhouse(passed, table)

    if not failed.empty:
        rej_table = "inventory_transactions_rejected"
        recreate_timescale_rejected(failed, rej_table)
        failed.to_sql(rej_table, engine_ts, index=False, chunksize=TS_CHUNK_SIZE)

# ======================================================
# MAIN
# ======================================================
def main():
    print("[INFO] Validation pipeline START")

    ensure_clickhouse_database()

    process_sku_master()
    process_inventory_transactions()

    print("[INFO] Validation pipeline COMPLETE")


if __name__ == "__main__":
    main()

