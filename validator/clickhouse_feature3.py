import os
from clickhouse_driver import Client as CHClient


# ---------------------------------------
# Config via Env Vars (works inside/outside Docker)
# ---------------------------------------
CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "9000"))
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "clickuser")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "clickpass")

# ---------------------------------------
# Get ClickHouse client
# ---------------------------------------


def get_client():
    return CHClient(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        user=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD
    )

# Raw table with transaction-level data (your screenshot)
SOURCE_DB = os.getenv("SOURCE_DB", "clean_zone")            # WHERE inventory_transactions is located
RAW_TABLE = os.getenv("RAW_TABLE", "inventory_transactions")

CLEAN_DB = os.getenv("CLEAN_DB", "clean_zone")        # WHERE daily_sku_demand will be created
CLEAN_TABLE = os.getenv("CLEAN_TABLE", "daily_sku_demand")

FEATURE_DB = os.getenv("FEATURE_DB", "feature_store")
FEATURE_TABLE = os.getenv("FEATURE_TABLE", "sku_demand_daily_features")

# ---------------------------------------
# Step 1: Create clean_zone.daily_sku_demand
# ---------------------------------------
def create_daily_table():
    client = get_client()
    client.execute(f"CREATE DATABASE IF NOT EXISTS {CLEAN_DB}")

    sql = f"""
    CREATE TABLE IF NOT EXISTS {CLEAN_DB}.{CLEAN_TABLE} (
        sku_id String,
        hospital_id String,
        tx_date Date,
        total_quantity Float64,
        total_revenue Float64
    )
    ENGINE = MergeTree()
    ORDER BY (sku_id, hospital_id, tx_date);
    """
    client.execute(sql)
    print(f" Created table {CLEAN_DB}.{CLEAN_TABLE}")


def populate_daily_table():
    client = get_client()

    # Full rebuild
    client.execute(f"TRUNCATE TABLE IF EXISTS {CLEAN_DB}.{CLEAN_TABLE}")

    # IMPORTANT: transaction_time is STRING → convert to Date
    sql = f"""
    INSERT INTO {CLEAN_DB}.{CLEAN_TABLE}
    SELECT
        sku_id,
        hospital_id,
        toDate(transaction_time) AS tx_date,
     	sum(quantity_consumed) AS total_quantity,
        sum(total_revenue) AS total_revenue
    FROM {SOURCE_DB}.{RAW_TABLE}
    GROUP BY
        sku_id,
        hospital_id,
        tx_date
    ORDER BY
        sku_id,
        hospital_id,
        tx_date;
    """
    client.execute(sql)
    print(f" Populated daily aggregated table {CLEAN_DB}.{CLEAN_TABLE}")


# ---------------------------------------
# Step 2: Create Feature Store Table
# ---------------------------------------
def create_feature_table():
    client = get_client()
    client.execute(f"CREATE DATABASE IF NOT EXISTS {FEATURE_DB}")

    sql = f"""
    CREATE TABLE IF NOT EXISTS {FEATURE_DB}.{FEATURE_TABLE} (
        ds Date,
        hospital_id String,
        sku_id String,

        dow UInt8,
        is_weekend UInt8,
        week_of_year UInt16,
        month UInt8,

        qty_1d Float64,
        qty_7d Float64,
        qty_28d Float64,
        avg_qty_7d Float64,
        avg_qty_28d Float64,
        lag_qty_1d Float64,
        lag_qty_7d Float64,

        rev_1d Float64,
        rev_7d Float64,
        rev_28d Float64,
        avg_rev_7d Float64,
        avg_rev_28d Float64
    )
    ENGINE = MergeTree()
    ORDER BY (sku_id, hospital_id, ds);
    """
    client.execute(sql)
    print(f" Created feature table {FEATURE_DB}.{FEATURE_TABLE}")


# ---------------------------------------
# Step 3: Populate Feature Store Table
# ---------------------------------------
def populate_feature_table():
    client = get_client()

    client.execute(f"TRUNCATE TABLE IF EXISTS {FEATURE_DB}.{FEATURE_TABLE}")

    sql = f"""
    INSERT INTO {FEATURE_DB}.{FEATURE_TABLE}
    SELECT
        tx_date AS ds,
        hospital_id,
        sku_id,

        toDayOfWeek(tx_date) AS dow,
        (dow IN (6, 7)) AS is_weekend,
        toWeek(tx_date) AS week_of_year,
        toMonth(tx_date) AS month,

        total_quantity AS qty_1d,

        sum(total_quantity) OVER (
            PARTITION BY hospital_id, sku_id
            ORDER BY tx_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS qty_7d,

        sum(total_quantity) OVER (
            PARTITION BY hospital_id, sku_id
            ORDER BY tx_date
            ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
        ) AS qty_28d,

        avg(total_quantity) OVER (
            PARTITION BY hospital_id, sku_id
            ORDER BY tx_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS avg_qty_7d,

        avg(total_quantity) OVER (
            PARTITION BY hospital_id, sku_id
            ORDER BY tx_date
            ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
        ) AS avg_qty_28d,

        lag(total_quantity, 1, 0) OVER (
            PARTITION BY hospital_id, sku_id
            ORDER BY tx_date
        ) AS lag_qty_1d,

        lag(total_quantity, 7, 0) OVER (
            PARTITION BY hospital_id, sku_id
            ORDER BY tx_date
        ) AS lag_qty_7d,

        total_revenue AS rev_1d,

        sum(total_revenue) OVER (
            PARTITION BY hospital_id, sku_id
            ORDER BY tx_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rev_7d,

        sum(total_revenue) OVER (
            PARTITION BY hospital_id, sku_id
            ORDER BY tx_date
            ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
        ) AS rev_28d,

        avg(total_revenue) OVER (
            PARTITION BY hospital_id, sku_id
            ORDER BY tx_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS avg_rev_7d,

        avg(total_revenue) OVER (
            PARTITION BY hospital_id, sku_id
            ORDER BY tx_date
            ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
        ) AS avg_rev_28d

    FROM {CLEAN_DB}.{CLEAN_TABLE}
    ORDER BY hospital_id, sku_id, ds;
    """

    client.execute(sql)
    print(f" Populated feature store: {FEATURE_DB}.{FEATURE_TABLE}")


# ---------------------------------------
# Entry Point
# ---------------------------------------
def main():
    print("\n  Building FULL Feature Store Pipeline...\n")

    print("STEP 1: Build daily_sku_demand...")
    create_daily_table()
    populate_daily_table()

    print("\n STEP 2: Create feature store table...")
    create_feature_table()

    print("\n STEP 3: Populate feature store table...")
    populate_feature_table()

    print("\n Feature Store build complete!\n")


if __name__ == "__main__":
    main()
