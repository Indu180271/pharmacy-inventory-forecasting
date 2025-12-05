"""
Stage-2 CLEAN ZONE PREPROCESSING
--------------------------------
Loads RAW_ZONE tables from ClickHouse,
applies preprocessing,
stores results into CLEAN_ZONE,
and exports cleaned CSVs locally.

NO OPENLINEAGE EMITS INSIDE THIS FILE.
The orchestrator handles lineage.
"""

import os
import numpy as np
import pandas as pd
from clickhouse_driver import Client as CHClient


# ----------------------------------------------------
# CLICKHOUSE INITIALIZATION
# ----------------------------------------------------
CH_HOST = os.getenv("CH_HOST", "clickhouse")
CH_PORT = int(os.getenv("CH_PORT", "9000"))
CH_USER = os.getenv("CH_USER", "clickuser")
CH_PASS = os.getenv("CH_PASS", "clickpass")

RAW_DB = os.getenv("RAW_DB", "raw_zone")
CLEAN_DB = os.getenv("CLEAN_DB", "clean_zone")

CLEAN_OUTPUT_DIR = "./clean_output"
os.makedirs(CLEAN_OUTPUT_DIR, exist_ok=True)

ch = CHClient(host=CH_HOST, port=CH_PORT, user=CH_USER, password=CH_PASS)


# ----------------------------------------------------
# CLICKHOUSE HELPERS
# ----------------------------------------------------
def get_clickhouse_schema(full_table: str):
    try:
        desc = ch.execute(f"DESCRIBE TABLE {full_table}")
        return {row[0]: row[1] for row in desc}
    except Exception:
        return {}


def load_raw(db: str, table: str) -> pd.DataFrame:
    try:
        schema = get_clickhouse_schema(f"{db}.{table}")
        rows = ch.execute(f"SELECT * FROM {db}.{table}")
        df = pd.DataFrame(rows, columns=schema.keys())
        print(f"[CLEAN] Loaded {len(df)} rows from {db}.{table}")
        return df
    except Exception as e:
        print(f"[CLEAN][ERROR] Cannot read {db}.{table}: {e}")
        return pd.DataFrame()


def recreate_clickhouse_table(df: pd.DataFrame, full_table: str):
    if df.empty:
        ch.execute(f"DROP TABLE IF EXISTS {full_table}")
        ch.execute(f"CREATE TABLE {full_table} (`_dummy` String) ENGINE=MergeTree() ORDER BY tuple()")
        return

    # Mapping for normal types
    type_map = {
        "int64": "Int64",
        "float64": "Float64",
        "int32": "Int32",
        "bool": "UInt8",
        "datetime64[ns]": "DateTime",
    }

    # Columns that must ALWAYS be Float64
    numeric_force = {
        "quantity_consumed",
        "unit_cogs",
        "unit_price",
        "margin_amount",
        "margin_percentage",
        "total_revenue",
    }

    col_defs = []

    for c, t in df.dtypes.items():
        if c in numeric_force:
            col_defs.append(f"`{c}` Float64")
        else:
            click_type = type_map.get(str(t), "String")
            col_defs.append(f"`{c}` {click_type}")

    # recreate table
    ch.execute(f"DROP TABLE IF EXISTS {full_table}")
    ch.execute(
        f"""
        CREATE TABLE {full_table} (
            {", ".join(col_defs)}
        )
        ENGINE=MergeTree()
        ORDER BY tuple()
        """
    )


def insert_df_clickhouse(df: pd.DataFrame, full_table: str):
    if df.empty:
        print(f"[CLEAN] No rows to write: {full_table}")
        return

    rows = [tuple(x) for x in df.itertuples(index=False, name=None)]
    cols = ", ".join([f"`{c}`" for c in df.columns])
    query = f"INSERT INTO {full_table} ({cols}) VALUES"

    for i in range(0, len(rows), 10000):
        ch.execute(query, rows[i:i + 10000])

    print(f"[CLEAN] Written {len(rows)} rows → {full_table}")


def write_clean(df: pd.DataFrame, table_name: str):
    csv_path = os.path.join(CLEAN_OUTPUT_DIR, f"{table_name}_clean.csv")
    df.to_csv(csv_path, index=False)
    print(f"[CLEAN] Exported cleaned CSV → {csv_path}")

    full_table = f"{CLEAN_DB}.{table_name}"
    recreate_clickhouse_table(df, full_table)
    insert_df_clickhouse(df, full_table)


# ----------------------------------------------------
# SKU MASTER PREPROCESSING
# ----------------------------------------------------
def preprocess_sku_master(df):
    df = df.copy()

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    null_cols = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=null_cols, errors="ignore")

    sentinel = ["nan", "NaN", "NULL", "None", "", "unknown", "Unknown"]
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].replace(sentinel, np.nan)

    df["therapeutic_area"] = df["therapeutic_area"].fillna("Medical_supplies")
    df["brand_generic_flag"] = df["brand_generic_flag"].fillna("Not Applicable")
    df["formulary_status"] = df["formulary_status"].fillna("Not Applicable")
    df["base_drug_name"] = df["base_drug_name"].fillna("Not a Drug")

    return df


def cleanzone_sku_master():
    raw = load_raw(RAW_DB, "sku_master")
    if raw.empty:
        print("[CLEAN] Raw SKU master empty — skipping.")
        return

    cleaned = preprocess_sku_master(raw)
    write_clean(cleaned, "sku_master")


# ----------------------------------------------------
# INVENTORY PREPROCESSING
# ----------------------------------------------------
def preprocess_inventory(df):
    df = df.copy()

    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("-", "_")
    )

    if "transaction_time" in df:
        df["transaction_time"] = pd.to_datetime(df["transaction_time"], errors="coerce")
        #df["transaction_time"] = df["transaction_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df["transaction_time"] = pd.to_datetime(df["transaction_time"], errors="coerce")

    if "last_updated" in df:
        df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
        #df["last_updated"] = df["last_updated"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")

    print("\n[CLEAN] Missing Values Inventory:\n", df.isnull().sum())

    # remove unnecessary / sensitive cols
    drop_cols = ["physician_id", "prescription_id", "prescription_detail_id", "procedure_id", "bill_id"]
    df = df.drop(columns=[c for c in drop_cols if c in df], errors="ignore")


    numeric_cols = ["quantity_consumed", "unit_cogs", "unit_price",
                    "margin_amount", "margin_percentage", "total_revenue"]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["transaction_code", "transaction_type"]:
        if col in df:
            df[col] = (
                df[col].astype(str)
                       .str.strip()
                       .str.replace("_", " ")
                       .str.replace("-", " ")
                       .str.lower()
            )

    if "quantity_consumed" in df:
        df = df[df["quantity_consumed"] >= 0]

    return df


def cleanzone_inventory():
    raw = load_raw(RAW_DB, "inventory_transactions")
    if raw.empty:
        print("[CLEAN] Raw inventory empty — skipping.")
        return

    cleaned = preprocess_inventory(raw)
    write_clean(cleaned, "inventory_transactions")


# ----------------------------------------------------
# RUN CLEAN ZONE PIPELINE (NO LINEAGE)
# ----------------------------------------------------
def run_clean_zone_pipeline():
    print("\n========== CLEAN ZONE PIPELINE ==========\n")

    cleanzone_sku_master()
    cleanzone_inventory()

    print("\n========== CLEAN ZONE COMPLETE ==========\n")


if __name__ == "__main__":
    run_clean_zone_pipeline()

