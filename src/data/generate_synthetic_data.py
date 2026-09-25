"""
Generates synthetic multi-store, multi-SKU retail sales data.

Why this exists: the real pipeline (training, cost calculation, drift
detection, retraining) needs *some* data to run against. Swap this file
out for a real loader (M5, Rossmann, or your own POS export) later --
every downstream script only cares about the output schema below, not
where the rows came from.

Output schema (data/raw/sales.csv):
    date        : YYYY-MM-DD
    store_id    : STORE_1 .. STORE_N
    sku_id      : SKU_0001 .. SKU_000M
    sales       : units sold that day (int, >= 0)
    is_special_day : 1 if a major Indian retail festival falls on this date
                     (this column means something DIFFERENT in the real
                     Rossmann loader -- German state/school holidays. Both
                     represent "an unusual calendar-driven demand day," just
                     via different real-world calendars. See
                     load_real_rossmann.py's docstring.)
    temp_c      : simulated daily temperature (affects some SKUs)
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "sales.csv"

N_STORES = 5
N_SKUS = 20
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"

# A small fixed festival calendar -- swap/extend with a real one later.
FESTIVALS = [
    "2023-01-15", "2023-04-14", "2023-10-24", "2023-11-12",  # Pongal, Tamil New Year, Dussehra, Diwali
    "2024-01-15", "2024-04-14", "2024-10-12", "2024-11-01",
]


def generate(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    festival_dates = set(pd.to_datetime(FESTIVALS))

    rows = []
    for store_idx in range(N_STORES):
        store_id = f"STORE_{store_idx + 1}"
        # each store has a baseline traffic multiplier
        store_mult = rng.uniform(0.7, 1.4)

        for sku_idx in range(N_SKUS):
            sku_id = f"SKU_{sku_idx + 1:04d}"
            base_demand = rng.uniform(5, 80)          # mean daily units
            weekly_amp = rng.uniform(0.1, 0.4)         # weekend effect
            is_weather_sensitive = sku_idx % 4 == 0    # every 4th SKU reacts to temp
            festival_boost = rng.uniform(1.5, 4.0)     # festival spike multiplier
            intermittent = sku_idx % 5 == 0            # every 5th SKU has sparse/zero days

            for d in dates:
                temp_c = 26 + 8 * np.sin(2 * np.pi * d.dayofyear / 365) + rng.normal(0, 1.5)

                seasonal = 1 + weekly_amp * (1 if d.dayofweek >= 5 else 0)
                weather_effect = 1 + (0.03 * (temp_c - 26) if is_weather_sensitive else 0)
                is_fest = d in festival_dates
                fest_effect = festival_boost if is_fest else 1.0

                lam = base_demand * store_mult * seasonal * weather_effect * fest_effect
                sales = rng.poisson(max(lam, 0.1))

                if intermittent and rng.random() < 0.4:
                    sales = 0  # simulate stockout / no-purchase days

                rows.append(
                    (d.date().isoformat(), store_id, sku_id, int(sales), int(is_fest), round(float(temp_c), 1))
                )

    df = pd.DataFrame(rows, columns=["date", "store_id", "sku_id", "sales", "is_special_day", "temp_c"])
    return df


if __name__ == "__main__":
    df = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUTPUT_PATH}")
    print(df.head())
    print("\nDate range:", df.date.min(), "to", df.date.max())
    print("Stores:", df.store_id.nunique(), "| SKUs:", df.sku_id.nunique())
