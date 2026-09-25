"""
Builds model-ready features from validated raw sales data.

Run standalone:
    python src/features/build_features.py

Output: data/processed/features.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "sales.csv"
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"

LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 28]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "sku_id", "date"])

    # --- calendar features ---
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    # --- lag + rolling features, computed per store-sku series ---
    grp = df.groupby(["store_id", "sku_id"])["sales"]

    for lag in LAGS:
        df[f"sales_lag_{lag}"] = grp.shift(lag)

    for window in ROLLING_WINDOWS:
        df[f"sales_rolling_mean_{window}"] = (
            grp.shift(1).rolling(window).mean().reset_index(drop=True)
        )
        df[f"sales_rolling_std_{window}"] = (
            grp.shift(1).rolling(window).std().reset_index(drop=True)
        )

    # --- special-day lead/lag: an upcoming special day affects today's restocking ---
    df["special_day_in_next_3d"] = (
        df.groupby(["store_id", "sku_id"])["is_special_day"]
        .transform(lambda s: s.shift(-1).rolling(3, min_periods=1).max())
        .fillna(0)
        .astype(int)
    )

    # drop rows where lag features can't be computed (start of each series)
    feature_cols = [c for c in df.columns if "lag" in c or "rolling" in c]
    df = df.dropna(subset=feature_cols)

    return df


if __name__ == "__main__":
    df = pd.read_csv(RAW_PATH)
    feats = build_features(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(feats):,} feature rows to {OUT_PATH}")
    print("Columns:", list(feats.columns))
    print(feats.tail(3))
