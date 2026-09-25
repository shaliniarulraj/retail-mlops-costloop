"""
Compares the reference (training) data distribution to a current
production batch and reports data drift.

Run standalone:
    python src/monitoring/drift_check.py

Outputs an HTML report to data/processed/drift_report.html you can open
in a browser, plus a printed summary for use in scripts/CI.
"""
import sys
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.pipeline.retrain_trigger import simulate_drifted_batch  # noqa: E402
from src.training.train import FEATURE_COLS  # noqa: E402

FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"
REPORT_PATH = ROOT / "data" / "processed" / "drift_report.html"


def run_drift_check():
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])

    reference = df[df["date"] < df["date"].max() - pd.Timedelta(days=14)]
    current = simulate_drifted_batch(FEATURES_PATH, n_days=14)

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference[FEATURE_COLS], current_data=current[FEATURE_COLS])

    result.save_html(str(REPORT_PATH))
    print(f"Drift report saved to {REPORT_PATH}")
    print("Open the HTML file in a browser for the full per-feature drift breakdown.")
    return result


if __name__ == "__main__":
    run_drift_check()
