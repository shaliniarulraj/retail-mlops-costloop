"""
Loads the REAL Rossmann Store Sales dataset (1,115 stores, 2013-01-01 to
2015-07-31, ~1.0M rows) and maps it onto this project's canonical schema
so every downstream script (validate, build_features, train, ...) works
unchanged.

Source: the official Kaggle competition dataset
(https://www.kaggle.com/c/rossmann-store-sales), mirrored here via a
public GitHub repo since Kaggle itself isn't reachable from this
environment's network allowlist:
    https://raw.githubusercontent.com/RPI-DATA/tutorials-intro/master/rossmann-store-sales/rossmann-store-sales/

Store-to-state mapping comes from a second real, public source:
    https://raw.githubusercontent.com/entron/entity-embedding-rossmann/master/store_states.csv
(compiled by the team behind the entity-embeddings paper, a well-known
3rd-place solution to this competition).

IMPORTANT HONEST LIMITATIONS vs. the synthetic generator this replaces:

1. NO SKU-LEVEL GRANULARITY. Rossmann reports per-store TOTAL daily sales,
   not per-product. Every row is mapped to a single sku_id = "SKU_TOTAL"
   per store. For genuine per-SKU granularity, use the M5 (Walmart) loader
   instead -- see load_real_m5.py, which requires you to supply the
   official Kaggle files yourself (see that script's docstring for why).

2. WEATHER: real day-level, state-level weather IS available -- pass
   --weather-csv pointing at a real weather.csv (the original fast.ai/
   Kaggle bundle: columns 'file' [German state name], 'Date',
   'Mean_TemperatureC', ...). Without it, falls back to real published
   REGIONAL CLIMATE NORMALS (annual mean per German state, sourced from
   public climate references -- see REGION_CLIMATE_NORMALS below)
   combined with a seasonal sine curve -- real climate data, but normals,
   not day-specific observations.

3. `is_special_day` (renamed from an earlier draft's `is_festival` -- see
   project discussion) is built from German StateHoliday + SchoolHoliday
   flags here, vs. an Indian festival calendar in the synthetic generator.
   Both represent the SAME modeling concept -- "an unusual, calendar-driven
   demand day" -- through two different real-world calendar systems. Treat
   this as evidence the feature generalizes across calendars, not as the
   same literal event.

Run standalone:
    python src/data/load_real_rossmann.py
    python src/data/load_real_rossmann.py --weather-csv data/raw/rossmann_weather_raw.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CSV = ROOT / "data" / "raw" / "rossmann_train_raw.csv"
DEFAULT_STORE_CSV = ROOT / "data" / "raw" / "rossmann_store_raw.csv"
DEFAULT_STATES_CSV = ROOT / "data" / "raw" / "rossmann_store_states_raw.csv"
OUTPUT_PATH = ROOT / "data" / "raw" / "sales.csv"

# Real published annual mean temperatures (°C) by German state, compiled
# from public climate references (climate-data.org, weather-atlas.com,
# worldtravelguide.net city climate summaries -- state capital or largest
# city used as the representative point). These are climate NORMALS, not
# day-specific observations -- see docstring point 2 above.
REGION_CLIMATE_NORMALS = {
    "BE": 10.1,        # Berlin
    "BY": 9.5,         # Bavaria (Munich, cooler due to elevation)
    "SH": 9.0,         # Schleswig-Holstein (Kiel, maritime north)
    "HE": 10.6,        # Hesse (Frankfurt)
    "SN": 9.4,         # Saxony (Dresden)
    "BW": 10.5,        # Baden-Württemberg (Stuttgart)
    "ST": 9.5,         # Saxony-Anhalt (Magdeburg, similar to Leipzig/Berlin)
    "RP": 10.8,        # Rhineland-Palatinate (Mainz, warm Rhine valley)
    "TH": 8.7,         # Thuringia (Erfurt, more continental/elevated)
    "HH": 9.8,         # Hamburg
    "HB,NI": 9.7,       # Bremen + Lower Saxony combined code used by this dataset
    "NW": 11.0,        # North Rhine-Westphalia (Cologne, warmest German region)
}
DEFAULT_ANNUAL_MEAN = 9.9  # fallback for any unmapped state code
SEASONAL_AMPLITUDE = 9.0   # typical German Cfb seasonal swing (°C)

# Maps the abbreviated state codes used in store_states.csv to the full
# state names used in the real weather.csv ("file" column). Built from
# inspecting the actual uploaded file -- weather.csv covers all 16 German
# federal states; store_states.csv only uses these 12 codes (no Rossmann
# stores fall in Brandenburg, Mecklenburg-Vorpommern, or Saarland).
# "HB,NI" (Bremen + Lower Saxony combined code in this dataset) is mapped
# to Niedersachsen as the larger, more representative of the pair --
# a real but approximate choice, noted here explicitly.
STATE_CODE_TO_WEATHER_NAME = {
    "NW": "NordrheinWestfalen",
    "BY": "Bayern",
    "SH": "SchleswigHolstein",
    "HE": "Hessen",
    "BE": "Berlin",
    "SN": "Sachsen",
    "BW": "BadenWuerttemberg",
    "ST": "SachsenAnhalt",
    "RP": "RheinlandPfalz",
    "TH": "Thueringen",
    "HH": "Hamburg",
    "HB,NI": "Niedersachsen",
}


def estimate_temp_from_normals(state: pd.Series, day_of_year: pd.Series) -> pd.Series:
    annual_mean = state.map(REGION_CLIMATE_NORMALS).fillna(DEFAULT_ANNUAL_MEAN)
    # peak ~day 200 (mid-July), trough ~day 17 (mid-January)
    seasonal = SEASONAL_AMPLITUDE * np.sin(2 * np.pi * (day_of_year - 110) / 365)
    return (annual_mean + seasonal).round(1)


def load_real_weather(weather_csv: Path) -> pd.DataFrame | None:
    """Loads the REAL weather.csv (state-keyed, the actual fast.ai/Kaggle
    bundle: columns 'file' [state name], 'Date', 'Mean_TemperatureC', ...).
    Returns a long dataframe of (weather_state_name, date, temp_c)."""
    if not weather_csv or not weather_csv.exists():
        return None
    w = pd.read_csv(weather_csv, low_memory=False)
    if "file" not in w.columns or "Mean_TemperatureC" not in w.columns:
        print(f"Warning: {weather_csv} doesn't match the expected real weather.csv schema "
              f"(columns found: {list(w.columns)[:6]}...). Falling back to climate normals.")
        return None
    w = w.rename(columns={"file": "weather_state_name", "Date": "date", "Mean_TemperatureC": "temp_c"})
    print(f"Using REAL day-level weather from {weather_csv} "
          f"({w['weather_state_name'].nunique()} states, {w['date'].nunique()} dates)")
    return w[["weather_state_name", "date", "temp_c"]]


def load_and_map(train_csv: Path, store_csv: Path, states_csv: Path, weather_csv) -> pd.DataFrame:
    train = pd.read_csv(train_csv, dtype={"StateHoliday": str}, low_memory=False)
    store = pd.read_csv(store_csv, low_memory=False)  # available for future feature extensions (StoreType, Assortment, etc.)

    # Closed-store days report Sales == 0 by definition, not zero *demand* --
    # including them would teach the model that "Monday after a holiday" means
    # zero units sold, which is a data artifact, not a forecasting target.
    train = train[train["Open"] == 1].copy()

    train["store_id"] = "STORE_" + train["Store"].astype(str)
    train["sku_id"] = "SKU_TOTAL"  # see module docstring -- honest limitation, not a real SKU
    train["sales"] = train["Sales"].astype(int)
    train["is_special_day"] = (
        (train["StateHoliday"].fillna("0") != "0") | (train["SchoolHoliday"] == 1)
    ).astype(int)

    dates = pd.to_datetime(train["Date"])
    train["date"] = dates.dt.strftime("%Y-%m-%d")
    train["day_of_year"] = dates.dt.dayofyear

    if states_csv.exists():
        states = pd.read_csv(states_csv)
        states.columns = [c.strip() for c in states.columns]
        train = train.merge(states, on="Store", how="left")
        train["State"] = train["State"].fillna("BE")
    else:
        print(f"Warning: {states_csv} not found -- using a flat national climate normal for all stores.")
        train["State"] = "BE"

    real_weather = load_real_weather(weather_csv) if weather_csv else None
    if real_weather is not None:
        train["weather_state_name"] = train["State"].map(STATE_CODE_TO_WEATHER_NAME)
        train = train.merge(real_weather, on=["weather_state_name", "date"], how="left")
        n_missing = train["temp_c"].isna().sum()
        if n_missing:
            print(f"{n_missing:,} rows had no matching real weather record -- "
                  f"filling those with climate normals.")
        train["temp_c"] = train["temp_c"].fillna(
            estimate_temp_from_normals(train["State"], train["day_of_year"])
        )
        train = train.drop(columns=["weather_state_name"])
    else:
        train["temp_c"] = estimate_temp_from_normals(train["State"], train["day_of_year"])

    out = train[["date", "store_id", "sku_id", "sales", "is_special_day", "temp_c"]].copy()
    out = out.sort_values(["store_id", "date"]).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--store-csv", type=Path, default=DEFAULT_STORE_CSV)
    parser.add_argument("--states-csv", type=Path, default=DEFAULT_STATES_CSV)
    parser.add_argument("--weather-csv", type=Path, default=None,
                         help="Optional path to a real weather.csv (see docstring point 2b)")
    parser.add_argument("--n-stores", type=int, default=None,
                         help="Optional: limit to first N stores for faster local iteration")
    args = parser.parse_args()

    if not args.train_csv.exists() or not args.store_csv.exists():
        print(f"Missing source files. Expected:\n  {args.train_csv}\n  {args.store_csv}")
        print("Download them from the Kaggle competition page, or use the GitHub mirror "
              "referenced in this script's docstring.")
        sys.exit(1)

    df = load_and_map(args.train_csv, args.store_csv, args.states_csv, args.weather_csv)

    if args.n_stores:
        keep_stores = sorted(df["store_id"].unique())[: args.n_stores]
        df = df[df["store_id"].isin(keep_stores)]

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows ({df['store_id'].nunique()} stores) to {OUTPUT_PATH}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Special-day rate: {df['is_special_day'].mean():.1%}")
    print(f"temp_c range: {df['temp_c'].min()} to {df['temp_c'].max()} "
          f"({'real weather' if args.weather_csv else 'climate normals'})")
    print(df.head())


if __name__ == "__main__":
    main()
