"""
Stage-3 FEATURE ENGINEERING
---------------------------
Creates feature_zone tables (if missing),
runs SQL to generate features,
inserts into ClickHouse,
and exports feature CSVs locally.

NO OPENLINEAGE EMITS INSIDE THIS FILE.
The orchestrator handles lineage.
"""

import os
from datetime import datetime
from clickhouse_driver import Client

# ----------------------------------------------------
# CONFIGURATION (paths inside the container)
# ----------------------------------------------------
BASE_DIR = "/app"                                # container-safe
SQL_DIR = f"{BASE_DIR}/sql"
OUTPUT_DIR = f"{BASE_DIR}/manual_features"       # container-safe

os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTAINER_ID = os.getenv("CH_CONTAINER", "clickhouse")

DDL_SQL = f"{SQL_DIR}/create_feature_tables.sql"
FEATURE_SQL = f"{SQL_DIR}/generate_features.sql"


# ----------------------------------------------------
# CLICKHOUSE CONNECTION
# ----------------------------------------------------
CH_HOST = os.getenv("CH_HOST", "clickhouse")
CH_PORT = int(os.getenv("CH_PORT", 9000))
CH_USER = os.getenv("CH_USER", "clickuser")
CH_PASS = os.getenv("CH_PASS", "clickpass")

ch = Client(
    host=CH_HOST,
    port=CH_PORT,
    user=CH_USER,
    password=CH_PASS
)


# ----------------------------------------------------
# CLICKHOUSE EXEC HELPERS
# ----------------------------------------------------
def run_clickhouse_query(sql_text: str):
    """Runs SQL directly in ClickHouse (no docker exec)."""
    try:
        ch.execute(sql_text)
    except Exception as e:
        print(f"[FEATURE][ERROR] Failed SQL execution: {e}")
        raise


def export_table_to_csv(query: str, output_path: str):
    """Exports a ClickHouse query result to CSV file."""
    try:
        # Fetch rows
        rows = ch.execute(query)

        # Describe schema for header
        schema = ch.execute("DESCRIBE TABLE feature_zone.inventory_features")
        header = [col[0] for col in schema]

        with open(output_path, "w") as f:
            # Write header
            f.write(",".join(header) + "\n")

            # Write data rows
            for row in rows:
                f.write(",".join([str(x) for x in row]) + "\n")

        print(f"[FEATURE] Exported CSV → {output_path}")

    except Exception as e:
        print(f"[FEATURE][ERROR] Export failed: {e}")
        raise



# ----------------------------------------------------
# FEATURE TABLE CREATION
# ----------------------------------------------------
def create_feature_tables():
    """Creates database and tables if not exists."""
    try:
        ddl_text = open(DDL_SQL).read()

        # Split queries by semicolon
        statements = ddl_text.split(";")

        for stmt in statements:
            stmt = stmt.strip()
            if stmt:   # run only non-empty statements
                ch.execute(stmt)

        print("[FEATURE] feature_zone tables created (or already exist).")

    except Exception as e:
        print("[FEATURE][ERROR] Table creation failed:", e)
        raise


# ----------------------------------------------------
# FEATURE GENERATION
# ----------------------------------------------------
def generate_features():
    """Runs feature SQL and exports final table."""
    try:
        # Read the entire SQL file
        feature_text = open(FEATURE_SQL).read()

        # Execute as ONE SINGLE QUERY (ClickHouse requires this)
        ch.execute(feature_text)

        print("[FEATURE] Feature generation SQL executed successfully.")

        # Export file
        today = datetime.now().strftime("%Y-%m-%d")
        output_file = f"{OUTPUT_DIR}/inventory_features_{today}.csv"

        export_query = "SELECT * FROM feature_zone.inventory_features"
        export_table_to_csv(export_query, output_file)

    except Exception as e:
        print("[FEATURE][ERROR] Feature generation failed:", e)
        raise

# ----------------------------------------------------
# MAIN FEATURE PIPELINE (NO LINEAGE)
# ----------------------------------------------------
def run_feature_pipeline():
    print("\n========== FEATURE ENGINEERING ==========\n")

    create_feature_tables()
    generate_features()

    print("\n========== FEATURE ENGINEERING COMPLETE ==========\n")


if __name__ == "__main__":
    run_feature_pipeline()

