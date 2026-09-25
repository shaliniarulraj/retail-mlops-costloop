"""
Loads the REAL M5 Forecasting (Walmart) dataset for TRUE per-SKU,
per-store demand forecasting -- the genuine granularity the synthetic
generator only approximates and Rossmann doesn't have at all.

WHY THIS SCRIPT NEEDS FILES FROM YOU, NOT A URL:
M5's files are large (sales_train_validation.csv ~120MB, sell_prices.csv
~200MB) and Kaggle-gated -- every GitHub mirror checked while building
this either excludes the large files via .gitignore (standard practice,
respecting Kaggle's redistribution terms) or doesn't exist. This
sandbox's network is also restricted to package registries and GitHub
raw/codeload, not kaggle.com.

You have normal internet access and can download the data yourself:
    1. Go to https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
       (free Kaggle account + accepting competition rules, no purchase)
    2. Download these 3 files:
         - calendar.csv            (small, ~100KB)
         - sales_train_validation.csv   (the real per-SKU sales, ~120MB)
         - sell_prices.csv         (optional but recommended, ~200MB)
    3. Upload them to this conversation (drag-and-drop into chat) --
       file uploads land in /mnt/user-data/uploads and ARE reachable by
       Claude regardless of the network sandbox.
    4. Tell Claude they're uploaded; this script will be pointed at them
       and produce TRUE per-SKU, per-store sales.csv.

WHAT M5 GIVES YOU THAT ROSSMANN DOESN'T:
  - 30,490 individual item-store series (HOBBIES/HOUSEHOLD/FOODS
    categories), not one store-level total -- this is what your proposal's
    "multi-SKU" framing actually needs.
  - calendar.csv includes real US holiday/event flags AND SNAP
    (food-assistance) days -- another real "is_special_day" signal,
    this time for a third real-world calendar system (after Indian
    festivals and German holidays).
  - sell_prices.csv enables a genuine price-elasticity feature if you
    want to extend the project further.

WHAT IT DOESN'T GIVE YOU:
  - No real weather (same situation as Rossmann -- not bundled).
  - US holidays are more about closures/observance, similar caveat as the
    Rossmann StateHoliday mapping: treat is_special_day consistently as
    "an unusual, calendar-driven demand day," not a literal festival.

Run (after uploading the files):
    python src/data/load_real_m5.py \
        --calendar-csv /mnt/user-data/uploads/calendar.csv \
        --sales-csv /mnt/user-data/uploads/sales_train_validation.csv \
        --n-items 50        # OPTIONAL: subsample for fast local iteration;
                             # full M5 is 30,490 series and will be slow to
                             # feature-engineer with this project's pandas
                             # groupby approach -- start small.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "data" / "raw" / "sales.csv"


def load_and_map(calendar_csv: Path, sales_csv: Path, n_items: int | None) -> pd.DataFrame:
    calendar = pd.read_csv(calendar_csv)
    sales_wide = pd.read_csv(sales_csv)

    if n_items is not None and n_items > 0:
        # M5's file is sorted store-by-store -- head(n) would silently grab
        # only the first store. A random sample preserves the multi-store,
        # multi-category diversity that's the whole point of using M5.
        sales_wide = sales_wide.sample(n=min(n_items, len(sales_wide)), random_state=42)

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]

    # M5 ships wide (one column per day) -- melt to the long format every
    # other script in this project expects (one row per store/SKU/day).
    long_df = sales_wide.melt(
        id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales"
    )

    long_df = long_df.merge(calendar[["d", "date", "event_name_1", "event_name_2", "snap_CA", "snap_TX", "snap_WI"]],
                             on="d", how="left")

    long_df["own_state_snap"] = np.select(
        [long_df["state_id"] == "CA", long_df["state_id"] == "TX", long_df["state_id"] == "WI"],
        [long_df["snap_CA"], long_df["snap_TX"], long_df["snap_WI"]],
        default=0,
    )
    long_df["is_special_day"] = (
        long_df["event_name_1"].notna()
        | long_df["event_name_2"].notna()
        | (long_df["own_state_snap"] == 1)
    ).astype(int)

    out = pd.DataFrame({
        "date": long_df["date"],
        "store_id": long_df["store_id"],          # already e.g. "CA_1", "TX_2" -- real M5 store codes
        "sku_id": long_df["item_id"],              # TRUE per-product granularity
        "sales": long_df["sales"].astype(int),
        "is_special_day": long_df["is_special_day"],
        # M5 has no weather either -- reuse the same honest placeholder
        # pattern as the Rossmann loader. A real Walmart deployment would
        # source actual weather per DMA; out of scope here.
        "temp_c": 18.0,
    })

    out = out.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calendar-csv", type=Path, required=True)
    parser.add_argument("--sales-csv", type=Path, required=True)
    parser.add_argument("--n-items", type=int, default=200,
                         help="Subsample to N item-store series for fast local iteration "
                              "(full M5 has 30,490 -- melting all of them to long format is "
                              "~59M rows and needs several GB of RAM; pass 0 only if your "
                              "machine has the memory for it)")
    args = parser.parse_args()

    if not args.calendar_csv.exists() or not args.sales_csv.exists():
        print(f"Missing source files. Expected:\n  {args.calendar_csv}\n  {args.sales_csv}")
        print("See this script's docstring for how to get them (you'll need to download "
              "from Kaggle yourself and upload them to this conversation).")
        sys.exit(1)

    df = load_and_map(args.calendar_csv, args.sales_csv, args.n_items)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows ({df['sku_id'].nunique()} SKUs, {df['store_id'].nunique()} stores) "
          f"to {OUTPUT_PATH}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(df.head())


if __name__ == "__main__":
    main()
