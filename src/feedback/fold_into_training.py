"""
Pulls unfolded manager overrides and merges them into the training feature
set, treating the manager's number as the corrected label for that
store/SKU/date -- so the NEXT retrain learns from human corrections.

Run standalone:
    python src/feedback/fold_into_training.py

This should run as a step in the retrain pipeline, BEFORE train.py,
whenever the cost-based retrain trigger fires.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.feedback.db import get_unfolded_overrides, mark_folded  # noqa: E402

FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"


def fold_overrides_into_features() -> int:
    overrides = get_unfolded_overrides()
    if not overrides:
        print("No pending overrides to fold in.")
        return 0

    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = df["sales"].astype(float)
    folded_ids = []
    for o in overrides:
        mask = (
            (df["store_id"] == o["store_id"])
            & (df["sku_id"] == o["sku_id"])
            & (df["date"] == pd.Timestamp(o["forecast_date"]))
        )
        if mask.any():
            # Replace the noisy/observed label with the manager's corrected number.
            # This is the actual "learning from human correction" step.
            df.loc[mask, "sales"] = o["manager_override"]
            folded_ids.append(o["id"])
        else:
            print(f"Warning: no matching feature row for override id={o['id']} "
                  f"({o['store_id']}/{o['sku_id']}/{o['forecast_date']}) -- skipped.")

    df.to_parquet(FEATURES_PATH, index=False)
    mark_folded(folded_ids)
    print(f"Folded {len(folded_ids)} override(s) into {FEATURES_PATH.name}.")
    return len(folded_ids)


if __name__ == "__main__":
    fold_overrides_into_features()
