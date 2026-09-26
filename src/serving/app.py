"""
Serves the trained model as an API, and converts raw forecasts into
inventory recommendations (reorder qty + safety stock).

Run standalone:
    uvicorn src.serving.app:app --reload --port 8000

Endpoints:
    POST /predict            - raw forecast, full feature vector required (original, unchanged)
    POST /recommend          - full feature vector -> forecast + reorder recommendation (original, unchanged)
    POST /recommend-by-date  - store_id + sku_id + date only; the API derives
                                calendar features and looks up lag/rolling
                                features from history itself.
    GET  /daily-forecasts    - no input. Automatically forecasts tomorrow for
                                every store/sku the model has history for
                                (or a `limit`-sized subset), using the same
                                history-derived approach as /recommend-by-date.
"""
import sys
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.train import FEATURE_COLS  # noqa: E402

MODEL_PATH = ROOT / "data" / "processed" / "model.txt"
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"

app = FastAPI(title="Retail Demand Forecasting & Inventory Recommendation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app)  # exposes GET /metrics for Prometheus to scrape

_model: lgb.Booster | None = None
_features_df: pd.DataFrame | None = None

# Safety-stock z-score for ~90% service level (tunable per category)
SAFETY_STOCK_Z = 1.28


class ForecastRequest(BaseModel):
    store_id: str
    sku_id: str
    is_special_day: int
    temp_c: float
    day_of_week: int
    is_weekend: int
    month: int
    day_of_year: int
    sales_lag_1: float
    sales_lag_7: float
    sales_lag_14: float
    sales_lag_28: float
    sales_rolling_mean_7: float
    sales_rolling_std_7: float
    sales_rolling_mean_28: float
    sales_rolling_std_28: float
    special_day_in_next_3d: int


class ForecastResponse(BaseModel):
    store_id: str
    sku_id: str
    predicted_demand: float


class RecommendationResponse(ForecastResponse):
    safety_stock: float
    recommended_order_qty: float


class RecommendByDateRequest(BaseModel):
    store_id: str
    sku_id: str
    date: str  # "YYYY-MM-DD"


class DailyForecastRow(RecommendationResponse):
    date: str
    is_weekend: int
    is_special_day: int


def get_model() -> lgb.Booster:
    global _model
    if _model is None:
        _model = lgb.Booster(model_file=str(MODEL_PATH))
    return _model


def get_features_df() -> pd.DataFrame:
    global _features_df
    if _features_df is None:
        _features_df = pd.read_parquet(FEATURES_PATH)
        _features_df["date"] = pd.to_datetime(_features_df["date"])
    return _features_df


def _predict_from_row(row: dict) -> float:
    model = get_model()
    x = [[row[col] for col in FEATURE_COLS]]
    pred = model.predict(x)[0]
    return float(max(pred, 0))


def _recommend_from_row(store_id: str, sku_id: str, row: dict) -> RecommendationResponse:
    pred = _predict_from_row(row)
    demand_std = row.get("sales_rolling_std_7") or 1.0
    safety_stock = SAFETY_STOCK_Z * demand_std * np.sqrt(1)
    recommended_qty = pred + safety_stock
    return RecommendationResponse(
        store_id=store_id,
        sku_id=sku_id,
        predicted_demand=round(pred, 2),
        safety_stock=round(float(safety_stock), 2),
        recommended_order_qty=round(float(recommended_qty), 2),
    )


def _build_row_for_date(store_id: str, sku_id: str, target_date: pd.Timestamp) -> dict:
    """
    Builds a full feature row for an arbitrary date using only store_id,
    sku_id and date:
      - calendar features (day_of_week, is_weekend, month, day_of_year) are
        recomputed directly from the date -- the caller never supplies these.
      - lag/rolling/special-day features are looked up from the most recent
        known history for that store+sku (the last date on or before
        target_date). This is the same assumption /daily-forecasts uses for
        "tomorrow" -- it's the most recent information the system actually has.
    """
    df = get_features_df()
    series = df[(df["store_id"] == store_id) & (df["sku_id"] == sku_id)]
    if series.empty:
        raise HTTPException(status_code=404, detail=f"No history found for store_id={store_id}, sku_id={sku_id}")

    history = series[series["date"] <= target_date]
    ref = history.iloc[-1] if not history.empty else series.iloc[0]

    row = {
        "is_special_day": int(ref["is_special_day"]),
        "temp_c": float(ref["temp_c"]),
        "day_of_week": int(target_date.dayofweek),
        "is_weekend": int(target_date.dayofweek >= 5),
        "month": int(target_date.month),
        "day_of_year": int(target_date.dayofyear),
        "sales_lag_1": float(ref["sales_lag_1"]),
        "sales_lag_7": float(ref["sales_lag_7"]),
        "sales_lag_14": float(ref["sales_lag_14"]),
        "sales_lag_28": float(ref["sales_lag_28"]),
        "sales_rolling_mean_7": float(ref["sales_rolling_mean_7"]),
        "sales_rolling_std_7": float(ref["sales_rolling_std_7"]),
        "sales_rolling_mean_28": float(ref["sales_rolling_mean_28"]),
        "sales_rolling_std_28": float(ref["sales_rolling_std_28"]),
        "special_day_in_next_3d": int(ref["special_day_in_next_3d"]),
    }
    return row


@app.on_event("startup")
def load_on_startup():
    get_model()
    get_features_df()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=ForecastResponse)
def predict(req: ForecastRequest):
    pred = _predict_from_row(req.model_dump())
    return ForecastResponse(store_id=req.store_id, sku_id=req.sku_id, predicted_demand=round(pred, 2))


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: ForecastRequest):
    """Original endpoint -- unchanged. Requires the full feature vector."""
    return _recommend_from_row(req.store_id, req.sku_id, req.model_dump())


@app.post("/recommend-by-date", response_model=RecommendationResponse)
def recommend_by_date(req: RecommendByDateRequest):
    """
    What the manual-entry form actually calls. Only store_id, sku_id and
    date are required -- calendar features are computed from the date, and
    lag/rolling/special-day features are looked up from the store+sku's
    most recent known history. Nothing is asked of the user beyond identity
    + date.
    """
    try:
        target_date = pd.to_datetime(req.date)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format")

    row = _build_row_for_date(req.store_id, req.sku_id, target_date)
    return _recommend_from_row(req.store_id, req.sku_id, row)


@app.get("/daily-forecasts", response_model=list[DailyForecastRow])
def daily_forecasts(limit: int = Query(default=25, ge=1, le=200)):
    """
    No input required. Forecasts tomorrow (relative to the latest date each
    store/sku has history for) for up to `limit` store+sku series, so the
    dashboard can show predictions automatically without anyone filling in
    a form. This is a synchronous batch computed on request; wire a
    scheduler (e.g. a Render Cron Job or Airflow/Prefect DAG) to hit this
    endpoint on a schedule and cache the result if you want it refreshed on
    a fixed cadence rather than computed per page load.
    """
    df = get_features_df()
    series_keys = (
        df[["store_id", "sku_id"]]
        .drop_duplicates()
        .sort_values(["store_id", "sku_id"])
        .head(limit)
        .itertuples(index=False)
    )

    results = []
    for store_id, sku_id in series_keys:
        series = df[(df["store_id"] == store_id) & (df["sku_id"] == sku_id)]
        last_date = series["date"].max()
        target_date = last_date + timedelta(days=1)

        row = _build_row_for_date(store_id, sku_id, target_date)
        rec = _recommend_from_row(store_id, sku_id, row)

        results.append(
            DailyForecastRow(
                date=target_date.strftime("%Y-%m-%d"),
                is_weekend=row["is_weekend"],
                is_special_day=row["is_special_day"],
                **rec.model_dump(),
            )
        )

    return results