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
    GET  /daily-forecasts    - forecasts tomorrow for every store/sku the
                                model has history for (or a `limit`-sized
                                subset), or for a single store if `store_id`
                                is passed.
    GET  /history            - a store+sku's recent actual sales, for
                                trend charts.
    POST /login              - lightweight demo login. Returns which store
                                (or "all stores") the account is scoped to.
                                See the USERS dict below -- this is NOT
                                production-grade auth (plaintext password
                                comparison, no hashing, no token expiry, no
                                HTTPS-only cookie). It exists to demonstrate
                                store-level access scoping for the course
                                project. A real deployment should replace
                                this with a proper identity provider
                                (e.g. OAuth/SSO) and issue signed, expiring
                                tokens verified on every request -- right
                                now nothing stops a client from calling
                                /daily-forecasts directly with any store_id
                                regardless of login.
"""
import sys
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

from src.training.train import FEATURE_COLS  # noqa: E402
from src.feedback.db import log_override  # noqa: E402


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

# Demo-only user directory: username -> (password, store_id or None for
# "all stores" / regional access). Replace with a real user store + hashed
# passwords + a proper auth flow before using this outside a course project.
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


def get_model() -> lgb.Booster:
    global _model
    if _model is None:
        _model = lgb.Booster(model_file=str(MODEL_PATH))
    return _model


def get_features() -> pd.DataFrame:
    global _features_df
    if _features_df is None:
        slim_path = ROOT / "data" / "processed" / "latest_features.csv"
        if slim_path.exists():
            _features_df = pd.read_csv(slim_path)
            _features_df["date"] = pd.to_datetime(_features_df["date"])
        else:
            raise HTTPException(
                status_code=503,
                detail="Feature store not found. Run the slim feature export first."
            )
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


class OverrideRequest(BaseModel):
    store_id: str
    sku_id: str
    overridden_qty: float
    reason: str = ""


@app.post("/override")
def submit_override(req: OverrideRequest):
    """
    Logs a manager's override of a recommendation to the same SQLite store
    that src/feedback/fold_into_training.py reads from, so overrides
    submitted here actually feed back into the next training run.
    """
    try:
        row = _build_row_for_date(req.store_id, req.sku_id, pd.Timestamp.today())
        system_recommendation = _recommend_from_row(req.store_id, req.sku_id, row).recommended_order_qty
    except HTTPException:
        system_recommendation = 0.0

    override_id = log_override(
        store_id=req.store_id,
        sku_id=req.sku_id,
        forecast_date=pd.Timestamp.today().strftime("%Y-%m-%d"),
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
    """
    Returns the last `days` of actual recorded sales for a store+sku, for
    plotting a trend chart on the frontend. Read-only, no model call.
    """
    df = get_features_df()
    series = df[(df["store_id"] == store_id) & (df["sku_id"] == sku_id)].sort_values("date")
    if series.empty:
        raise HTTPException(status_code=404, detail=f"No history found for store_id={store_id}, sku_id={sku_id}")

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
    Forecasts tomorrow for store/sku series. If `store_id` is given, returns
    only that store's series (a store manager's scoped view); otherwise
    returns up to `limit` series across the whole dataset (a regional/admin
    view). The frontend decides which to call based on who's logged in --
    the API itself does not enforce that scoping (see the /login docstring
    above), so this parameter should be treated as a convenience filter,
    not an access boundary, until real auth is added.
    """
    df = get_features_df()
    keys_df = df[["store_id", "sku_id"]].drop_duplicates().sort_values(["store_id", "sku_id"])

    if store_id:
        keys_df = keys_df[keys_df["store_id"] == store_id]
        if keys_df.empty:
            raise HTTPException(status_code=404, detail=f"No history found for store_id={store_id}")
    else:
        keys_df = keys_df.head(limit)

    series_keys = keys_df.itertuples(index=False)

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