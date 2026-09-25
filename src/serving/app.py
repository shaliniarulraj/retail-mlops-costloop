"""
Serves the trained model as an API, and converts raw forecasts into
inventory recommendations (reorder qty + safety stock).

Run standalone:
    uvicorn src.serving.app:app --reload --port 8000

Then try:
    curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d '{
        "store_id": "STORE_1", "sku_id": "SKU_0002",
        "is_special_day": 0, "temp_c": 28.0, "day_of_week": 2, "is_weekend": 0,
        "month": 6, "day_of_year": 171,
        "sales_lag_1": 40, "sales_lag_7": 38, "sales_lag_14": 35, "sales_lag_28": 33,
        "sales_rolling_mean_7": 37.5, "sales_rolling_std_7": 4.2,
        "sales_rolling_mean_28": 35.0, "sales_rolling_std_28": 5.1,
        "special_day_in_next_3d": 0
    }'
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.train import FEATURE_COLS  # noqa: E402

MODEL_PATH = ROOT / "data" / "processed" / "model.txt"

app = FastAPI(title="Retail Demand Forecasting & Inventory Recommendation API")
Instrumentator().instrument(app).expose(app)  # exposes GET /metrics for Prometheus to scrape

_model: lgb.Booster | None = None

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


def get_model() -> lgb.Booster:
    global _model
    if _model is None:
        _model = lgb.Booster(model_file=str(MODEL_PATH))
    return _model


def _predict(req: ForecastRequest) -> float:
    model = get_model()
    x = [[getattr(req, col) for col in FEATURE_COLS]]
    pred = model.predict(x)[0]
    return float(max(pred, 0))


@app.on_event("startup")
def load_model_on_startup():
    get_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=ForecastResponse)
def predict(req: ForecastRequest):
    pred = _predict(req)
    return ForecastResponse(store_id=req.store_id, sku_id=req.sku_id, predicted_demand=round(pred, 2))


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: ForecastRequest):
    """Turns a raw demand forecast into an actionable reorder recommendation."""
    pred = _predict(req)

    # Use the rolling std as a proxy for demand volatility -> safety stock
    demand_std = req.sales_rolling_std_7 or 1.0
    safety_stock = SAFETY_STOCK_Z * demand_std * np.sqrt(1)  # lead time = 1 review period (simplify for course scope)

    recommended_qty = pred + safety_stock

    return RecommendationResponse(
        store_id=req.store_id,
        sku_id=req.sku_id,
        predicted_demand=round(pred, 2),
        safety_stock=round(float(safety_stock), 2),
        recommended_order_qty=round(float(recommended_qty), 2),
    )
