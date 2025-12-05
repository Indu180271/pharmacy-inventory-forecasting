import sys
import time
import subprocess
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Add project root paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "validator"))
sys.path.append(str(BASE_DIR / "src"))

# ---------------------------------------------------------------------------
# Import pipeline modules (lightweight ones only)
# ---------------------------------------------------------------------------
from validator.validate_inventory_with_gx import main as gx_validate
from validator.clean_preprocess import run_clean_zone_pipeline
from validator.clickhouse_feature3 import main as feature_full
from validator.clickhouse_feature_top10 import main as feature_top10
from validator.openlineage_emitter import emit_event

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
from prometheus_client import (
    Counter, Gauge, Histogram, start_http_server
)

PIPELINE_RUNS = Counter("pipeline_runs_total", "Unified pipeline total runs")
PIPELINE_FAILURES = Counter("pipeline_failures_total", "Unified pipeline failures")
PIPELINE_DURATION = Histogram("pipeline_duration_seconds", "Total runtime")
STEP_DURATION = Histogram("pipeline_step_duration_seconds", "Each step duration")
PIPELINE_STATUS = Gauge("pipeline_status", "1=Running, 0=Idle")


# ---------------------------------------------------------------------------
# Helper: execute a step with timing + lineage + printing
# ---------------------------------------------------------------------------
def run_step(step_name: str, fn):
    print(f"\n========= STEP: {step_name} =========")

    with STEP_DURATION.time():
        try:
            emit_event(step_name, "START")
            fn()
            emit_event(step_name, "COMPLETE")
            print(f" {step_name} completed")
        except Exception:
            emit_event(step_name, "FAIL")
            print(f" Step failed: {step_name}")
            traceback.print_exc()
            raise



# ---------------------------------------------------------------------------
# Optional serving function
# ---------------------------------------------------------------------------
def start_fastapi():
    print("Starting FastAPI (model server)...")
    subprocess.Popen([
        "uvicorn",
        "src.api.main:app",
        "--host", "0.0.0.0",
        "--port", "8002",
    ])
    print(" FastAPI running at http://localhost:8002")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_unified_pipeline(start_api: bool = False):
    print("\n======================================================")
    print("          RUNNING UNIFIED END-TO-END PIPELINE          ")
    print("======================================================\n")

    PIPELINE_STATUS.set(1)
    PIPELINE_RUNS.inc()
    start_time = time.time()

    try:
        emit_event("unified_pipeline", "START")

        run_step("data_validation", gx_validate)
        run_step("clean_zone", run_clean_zone_pipeline)
        run_step("feature_store_full", feature_full)
        run_step("feature_store_top10", feature_top10)
    
        if start_api:
            run_step("model_serving", start_fastapi)

        emit_event("unified_pipeline", "COMPLETE")

        PIPELINE_DURATION.observe(time.time() - start_time)
        print("\n UNIFIED PIPELINE COMPLETED SUCCESSFULLY\n")

    except Exception:
        PIPELINE_FAILURES.inc()
        emit_event("unified_pipeline", "FAIL")
        print("\n UNIFIED PIPELINE FAILED\n")
        traceback.print_exc()

    finally:
        PIPELINE_STATUS.set(0)


# ---------------------------------------------------------------------------
# CLI ENTRYPOINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    # Start Prometheus endpoint
    start_http_server(8000)

    parser = argparse.ArgumentParser(description="Unified MLOps Pipeline")
    parser.add_argument(
        "--step",
        choices=[
            "all",
            "validate",
            "clean",
            "feature_full",
            "feature_top10",
            "serve",
        ],
        default="all",
        help="Which part of the pipeline to run",
    )

    args = parser.parse_args()

    if args.step == "all":
        run_unified_pipeline(start_api=False)
    elif args.step == "validate":
        run_step("data_validation", gx_validate)
    elif args.step == "clean":
        run_step("clean_zone", run_clean_zone_pipeline)
    elif args.step == "feature_full":
        run_step("feature_store_full", feature_full)
    elif args.step == "feature_top10":
        run_step("feature_store_top10", feature_top10)
    elif args.step == "serve":
        start_fastapi()

    print("Metrics server on :8000 – keeping process alive for Prometheus. Ctrl+C to exit.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down pipeline process.")

