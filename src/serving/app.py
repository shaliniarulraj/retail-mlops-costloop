"""
Serves the trained model as an API, and converts raw forecasts into
inventory recommendations (reorder qty + safety stock).

Run standalone:
    uvicorn src.serving.app:app --reload --port 8000

Endpoints:
    GET  /, /health          - health checks
    POST /predict            - raw forecast, full feature vector required
    POST /recommend          - full feature vector -> forecast + reorder recommendation
    POST /recommend-by-date  - store_id + sku_id + date only
    GET  /daily-forecasts    - forecasts for tomorrow (optionally scoped to one store)
    GET  /history            - a store+sku's recent actual sales, for trend charts
    POST /login              - demo login (NOT production-grade auth: plaintext
                               passwords, no tokens, no server-side enforcement
                               of store scoping)
    POST /override           - logs a manager override to SQLite feedback store

Memory notes (Render free tier = 512 MB):
    - No import from src.training.train (it pulls in the whole training stack).
      Feature order is read from the trained model itself.
    - Feature CSV is loaded with only needed columns, small dtypes, and only
      the last HISTORY_DAYS days, with a one-time per-series index.
    - /daily-forecasts results are cached for CACHE_TTL seconds.
"""
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "data" / "processed" / "model.txt"
FEATURES_CSV_PATH = ROOT / "data" / "processed" / "latest_features.csv"

# Fallback feature order, used only if the model file has no real feature names
# (e.g. "Column_0"). Normally the order is read from the model itself.
FALLBACK_FEATURE_COLS = [
    "is_special_day", "temp_c", "day_of_week", "is_weekend", "month",
    "day_of_year", "sales_lag_1", "sales_lag_7", "sales_lag_14",
    "sales_lag_28", "sales_rolling_mean_7", "sales_rolling_std_7",
    "sales_rolling_mean_28", "sales_rolling_std_28", "special_day_in_next_3d",
]

NEEDED_COLS = [
    "date", "store_id", "sku_id", "sales", "is_special_day", "temp_c",
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_rolling_mean_7", "sales_rolling_std_7",
    "sales_rolling_mean_28", "sales_rolling_std_28",
    "special_day_in_next_3d",
]
HISTORY_DAYS = 90   # only keep the recent window in memory
CACHE_TTL = 600     # seconds

app = FastAPI(title="Retail Demand Forecasting & Inventory Recommendation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app)  # GET /metrics for Prometheus

_model: Optional[lgb.Booster] = None
_feature_cols: Optional[list] = None
_features_df: Optional[pd.DataFrame] = None
_series_index: dict = {}
_forecast_cache: dict = {}

# Safety-stock z-score for ~90% service level (tunable per category)
SAFETY_STOCK_Z = 1.28

# Demo-only user directory. Replace with a real user store + hashed passwords
# + signed tokens before using outside a course project.
USERS = {
    "store1_manager": {"password": "store1pass", "store_id": "STORE_1", "role": "store_manager"},
    "store10_manager": {"password": "store10pass", "store_id": "STORE_10", "role": "store_manager"},
    "regional_manager": {"password": "regionalpass", "store_id": None, "role": "regional_manager"},
}


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    username: str
    role: str
    store_id: Optional[str] = None  # None means access to all stores


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


class OverrideRequest(BaseModel):
    store_id: str
    sku_id: str
    overridden_qty: float
    reason: str = ""


# ---------------------------------------------------------------- loaders

def get_model() -> lgb.Booster:
    global _model, _feature_cols
    if _model is None:
        _model = lgb.Booster(model_file=str(MODEL_PATH))
        names = list(_model.feature_name())
        if names and not all(n.startswith("Column_") for n in names):
            _feature_cols = names
        else:
            _feature_cols = FALLBACK_FEATURE_COLS
    return _model


def get_feature_cols() -> list:
    get_model()
    return _feature_cols


def get_features_df() -> pd.DataFrame:
    global _features_df, _series_index
    if _features_df is None:
        if not FEATURES_CSV_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Feature store not found. Run the slim feature export first.",
            )
        header = pd.read_csv(FEATURES_CSV_PATH, nrows=0).columns
        use = [c for c in NEEDED_COLS if c in header]
        df = pd.read_csv(
            FEATURES_CSV_PATH,
            usecols=use,
            parse_dates=["date"],
            dtype={"store_id": str, "sku_id": str},
        )
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
        df = df[df["date"] >= df["date"].max() - pd.Timedelta(days=HISTORY_DAYS)]
        df = df.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
        _series_index = df.groupby(["store_id", "sku_id"]).indices
        _features_df = df
    return _features_df


def get_series(store_id: str, sku_id: str) -> Optional[pd.DataFrame]:
    df = get_features_df()
    idx = _series_index.get((store_id, sku_id))
    return None if idx is None else df.iloc[idx]


# ---------------------------------------------------------------- helpers

def _predict_from_row(row: dict) -> float:
    model = get_model()
    x = [[row[col] for col in get_feature_cols()]]
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
    Builds a full feature row for a date using only store_id, sku_id and date.
    Calendar features come from the date; lag/rolling/special-day features are
    looked up from the most recent known history on or before target_date.
    """
    series = get_series(store_id, sku_id)
    if series is None or series.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No history found for store_id={store_id}, sku_id={sku_id}",
        )

    history = series[series["date"] <= target_date]
    ref = history.iloc[-1] if not history.empty else series.iloc[0]

    return {
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


# ---------------------------------------------------------------- routes

@app.on_event("startup")
def load_on_startup():
    get_model()
    get_features_df()


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "features_loaded": _features_df is not None,
    }


@app.post("/predict", response_model=ForecastResponse)
def predict(req: ForecastRequest):
    pred = _predict_from_row(req.model_dump())
    return ForecastResponse(store_id=req.store_id, sku_id=req.sku_id, predicted_demand=round(pred, 2))


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: ForecastRequest):
    """Requires the full feature vector."""
    return _recommend_from_row(req.store_id, req.sku_id, req.model_dump())


@app.post("/recommend-by-date", response_model=RecommendationResponse)
def recommend_by_date(req: RecommendByDateRequest):
    """Only store_id, sku_id and date are required."""
    try:
        target_date = pd.to_datetime(req.date)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format")

    row = _build_row_for_date(req.store_id, req.sku_id, target_date)
    return _recommend_from_row(req.store_id, req.sku_id, row)


@app.post("/override")
def submit_override(req: OverrideRequest):
    """
    Logs a manager's override to the same SQLite store that
    src/feedback/fold_into_training.py reads from.
    """
    from src.feedback.db import log_override  # lazy import keeps startup light

    today = pd.Timestamp.today().normalize()
    try:
        row = _build_row_for_date(req.store_id, req.sku_id, today)
        system_recommendation = _recommend_from_row(req.store_id, req.sku_id, row).recommended_order_qty
    except HTTPException:
        system_recommendation = 0.0

    override_id = log_override(
        store_id=req.store_id,
        sku_id=req.sku_id,
        forecast_date=today.strftime("%Y-%m-%d"),
        system_recommendation=system_recommendation,
        manager_override=req.overridden_qty,
        override_reason=req.reason,
    )
    return {"id": override_id, "status": "logged", "system_recommendation": system_recommendation}


@app.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    user = USERS.get(req.username)
    if not user or user["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return LoginResponse(username=req.username, role=user["role"], store_id=user["store_id"])


@app.get("/history")
def history(store_id: str, sku_id: str, days: int = Query(default=60, ge=7, le=365)):
    """Last `days` of actual sales (capped by HISTORY_DAYS held in memory)."""
    series = get_series(store_id, sku_id)
    if series is None or series.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No history found for store_id={store_id}, sku_id={sku_id}",
        )

    tail = series.tail(days)
    return {
        "store_id": store_id,
        "sku_id": sku_id,
        "dates": tail["date"].dt.strftime("%Y-%m-%d").tolist(),
        "sales": [round(float(v), 2) for v in tail["sales"].tolist()],
    }


@app.get("/daily-forecasts", response_model=list[DailyForecastRow])
def daily_forecasts(
    limit: int = Query(default=25, ge=1, le=200),
    store_id: Optional[str] = Query(default=None),
):
    """
    Forecasts tomorrow for store/sku series. With `store_id`, only that
    store's series; otherwise up to `limit` series. Results are cached for
    CACHE_TTL seconds. The scoping is a convenience filter, not an access
    boundary (see the /login note above).
    """
    cache_key = (store_id, limit)
    hit = _forecast_cache.get(cache_key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]

    get_features_df()
    keys = sorted(_series_index.keys())

    if store_id:
        keys = [k for k in keys if k[0] == store_id]
        if not keys:
            raise HTTPException(status_code=404, detail=f"No history found for store_id={store_id}")
    else:
        keys = keys[:limit]

    results = []
    for sid, sku in keys:
        series = get_series(sid, sku)
        target_date = series["date"].max() + timedelta(days=1)

        row = _build_row_for_date(sid, sku, target_date)
        rec = _recommend_from_row(sid, sku, row)

        results.append(
            DailyForecastRow(
                date=target_date.strftime("%Y-%m-%d"),
                is_weekend=row["is_weekend"],
                is_special_day=row["is_special_day"],
                **rec.model_dump(),
            )
        )

    _forecast_cache[cache_key] = (time.time(), results)
    return results