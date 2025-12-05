# data_pipeline/ingest_pipeline.py
import os
import hashlib
import re
import logging
from datetime import datetime
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest_pipeline")

### ---------- CONFIG ----------

PG_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": int(os.getenv("PG_PORT", 5432)),
    "dbname": os.getenv("PG_DB", "ml_opps"),
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", "1234"),
}

TS_CONFIG = {
    "host": os.getenv("TS_HOST", "localhost"),
    "port": int(os.getenv("TS_PORT", 5432)),
    "dbname": os.getenv("TS_DB", "ml_opps"),
    "user": os.getenv("TS_USER", "postgres"),
    "password": os.getenv("TS_PASSWORD", "1234"),
}

### ---------- Helpers ----------

def _connect_pg(cfg):
    return psycopg2.connect(
        dbname=cfg["dbname"],
        user=cfg["user"],
        password=cfg["password"],
        host=cfg["host"],
        port=str(cfg["port"]),
    )

def _sanitize_cols(cols):
    return [re.sub(r"\W+", "_", c.lower()) for c in cols]

def _gen_row_hash(df):
    """deterministic row hash"""
    return (
        df.astype(str)
        .agg("||".join, axis=1)
        .apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    )

### ---------- Ingestion functions ----------

def ingest_skus_postgres(csv_path: str, table_name: str = "sku_master"):
    job = "ingest_skus_postgres"
    
    logger.info("Starting SKU ingestion from %s → PostgreSQL %s", csv_path, table_name)

    try:
        df = pd.read_csv(csv_path)
        df.columns = _sanitize_cols(df.columns)

        # primary key logic
        if "sku_id" in df.columns:
            pk = "sku_id"
        else:
            df["sku_hash"] = _gen_row_hash(df)
            pk = "sku_hash"

        conn = _connect_pg(PG_CONFIG)

        with conn.cursor() as cur:
            cols_sql = []
            for col in df.columns:
                if col == pk:
                    cols_sql.append(f'"{col}" TEXT PRIMARY KEY')
                else:
                    cols_sql.append(f'"{col}" TEXT')

            cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
            cur.execute(f"CREATE TABLE {table_name} ({', '.join(cols_sql)});")

        conn.commit()

        cols = ",".join([f'"{c}"' for c in df.columns])
        rows = [tuple(str(x) for x in row) for row in df.to_numpy()]
        insert_sql = (
            f'INSERT INTO {table_name} ({cols}) VALUES %s ON CONFLICT ("{pk}") DO NOTHING;'
        )

        with conn.cursor() as cur:
            execute_values(cur, insert_sql, rows)
        conn.commit()
        conn.close()

        logger.info("SKU ingestion complete. Rows=%s", len(df))

    except Exception as e:
        logger.exception("SKU ingestion FAILED")
        raise


def ingest_inventory_timescale(csv_path: str, table_name: str = "inventory_transactions"):
    job = "ingest_inventory_timescale"

    logger.info("Starting inventory ingestion from %s → Timescale table %s",
                csv_path, table_name)

    try:
        df = pd.read_csv(csv_path)
        df.columns = _sanitize_cols(df.columns)

        if "transaction_time" not in df.columns:
            raise Exception("transaction_time column missing in CSV!")

        df["transaction_time"] = pd.to_datetime(df["transaction_time"], errors="coerce")
        df["row_hash"] = _gen_row_hash(df)

        conn = _connect_pg(TS_CONFIG)

        with conn.cursor() as cur:
            cols_sql = []
            for col in df.columns:
                if col == "transaction_time":
                    cols_sql.append(f'"{col}" TIMESTAMPTZ')
                else:
                    cols_sql.append(f'"{col}" TEXT')

            cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")

            create_sql = f"""
            CREATE TABLE {table_name} (
                {', '.join(cols_sql)},
                PRIMARY KEY (row_hash, transaction_time)
            );
            """
            cur.execute(create_sql)

            cur.execute(f"""
                SELECT create_hypertable(
                    '{table_name}',
                    'transaction_time',
                    if_not_exists => TRUE,
                    migrate_data => TRUE
                );
            """)

        conn.commit()

        cols = ",".join([f'"{c}"' for c in df.columns])
        rows = [tuple("" if pd.isna(x) else str(x) for x in row) for row in df.to_numpy()]
        insert_sql = f"""
            INSERT INTO {table_name} ({cols}) VALUES %s
            ON CONFLICT DO NOTHING;
        """

        with conn.cursor() as cur:
            execute_values(cur, insert_sql, rows)
        conn.commit()
        conn.close()

        logger.info("Inventory ingestion completed. Rows=%s", len(df))

    except Exception as e:
        logger.exception("Inventory ingestion FAILED")
        raise


### ---------- Orchestration ----------

def run_all(skus_csv: str, transactions_csv: str):
    ingest_skus_postgres(skus_csv, table_name="sku_master")
    ingest_inventory_timescale(transactions_csv, table_name="inventory_transactions")


if __name__ == "__main__":
    skus = "/app/data/skus_master_raw.csv"
    txns = "/app/data/inventory_transactions_raw.csv"
    run_all(skus, txns)

    flag_path = "/app/validator/intermediate/ingestion_done.flag"
    os.makedirs(os.path.dirname(flag_path), exist_ok=True)
    with open(flag_path, "w") as f:
        f.write("ingestion_completed\n")

    print(f"[INFO] DVC flag created at: {flag_path}")

