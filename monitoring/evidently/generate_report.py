import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import os

BASE = "/app"
VALID_DIR = f"{BASE}/validated_output"
REPORT_DIR = f"{BASE}/monitoring/evidently/reports"

os.makedirs(REPORT_DIR, exist_ok=True)

def build_report():
    df_ref = pd.read_csv(f"{VALID_DIR}/sku_master_valid.csv")
    df_cur = pd.read_csv(f"{VALID_DIR}/inventory_transactions_valid.csv")

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset()
    ])

    report.run(reference_data=df_ref, current_data=df_cur)
    report.save_html(f"{REPORT_DIR}/data_drift_report.html")

    print("[EVIDENTLY] Report generated successfully.")

if __name__ == "__main__":
    build_report()

