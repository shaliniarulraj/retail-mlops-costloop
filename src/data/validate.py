"""
Validates the raw sales data against an explicit schema before it's
allowed into the feature engineering step.

Run standalone:
    python src/data/validate.py

Used as an importable check inside the training pipeline too --
if validation fails, training should refuse to run rather than
fail confusingly three steps later.
"""
import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "sales.csv"

sales_schema = DataFrameSchema(
    {
        "date": Column(str, Check.str_matches(r"^\d{4}-\d{2}-\d{2}$"), nullable=False),
        # Synthetic/Rossmann use STORE_n; real M5 uses codes like "CA_1" --
        # kept permissive (alphanumeric + underscore) to cover all three
        # real-world ID conventions this project has actually ingested.
        "store_id": Column(str, Check.str_matches(r"^[A-Za-z0-9_]+$"), nullable=False),
        "sku_id": Column(str, Check.str_matches(r"^[A-Za-z0-9_]+$"), nullable=False),
        "sales": Column(int, Check.ge(0), nullable=False),
        "is_special_day": Column(int, Check.isin([0, 1]), nullable=False),
        # Range widened after real Rossmann weather data showed German winter
        # lows of -13C -- the original -10..55 range was calibrated only for
        # the synthetic Indian-climate generator and was too narrow for real
        # European winter data. -25..50 covers both comfortably.
        "temp_c": Column(float, Check.in_range(-25, 50), nullable=False),
    },
    strict=True,   # no unexpected extra columns allowed
    coerce=True,
)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Raises pandera.errors.SchemaError on failure."""
    return sales_schema.validate(df)


if __name__ == "__main__":
    if not RAW_PATH.exists():
        print(f"No raw data found at {RAW_PATH}. Run generate_synthetic_data.py first.")
        sys.exit(1)

    df = pd.read_csv(RAW_PATH, dtype={"date": str, "store_id": str, "sku_id": str})
    try:
        validate(df)
        print(f"PASSED -- {len(df):,} rows validated against schema.")
    except pa.errors.SchemaError as e:
        print("VALIDATION FAILED:")
        print(e)
        sys.exit(1)
