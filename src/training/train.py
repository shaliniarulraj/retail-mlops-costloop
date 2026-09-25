"""
Trains a LightGBM demand forecasting model and logs everything to MLflow:
params, accuracy metrics, AND the business cost metric.

Run standalone:
    python src/training/train.py

View results:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    (then open http://localhost:5000)
"""
import sys
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cost.cost_calculator import compute_cost  # noqa: E402

FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"
MODEL_OUT_PATH = ROOT / "data" / "processed" / "model.txt"

TARGET = "sales"
FEATURE_COLS = [
    "is_special_day", "temp_c", "day_of_week", "is_weekend", "month", "day_of_year",
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_rolling_mean_7", "sales_rolling_std_7",
    "sales_rolling_mean_28", "sales_rolling_std_28",
    "special_day_in_next_3d",
]
CATEGORICAL_COLS = []  # store_id/sku_id excluded from this simple baseline; see README for per-series scaling


def wmape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100)


def time_based_split(df: pd.DataFrame, val_days: int = 60):
    df = df.sort_values("date")
    cutoff = df["date"].max() - pd.Timedelta(days=val_days)
    train = df[df["date"] <= cutoff]
    val = df[df["date"] > cutoff]
    return train, val


def train_model(df: pd.DataFrame, params: dict | None = None):
    params = params or {
        "objective": "tweedie",      # handles intermittent/zero-inflated demand better than plain regression
        "tweedie_variance_power": 1.3,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_samples": 20,
        "verbose": -1,
    }

    train_df, val_df = time_based_split(df)

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET]
    X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET]

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    preds_val = np.clip(model.predict(X_val), a_min=0, a_max=None)

    mae = mean_absolute_error(y_val, preds_val)
    w = wmape(y_val.values, preds_val)
    cost = compute_cost(y_val.values, preds_val)

    metrics = {
        "mae": mae,
        "wmape": w,
        "business_cost_total": cost.total_cost,
        "business_cost_avg_per_row": cost.avg_cost_per_row,
        "stockout_cost": cost.stockout_cost,
        "holding_cost": cost.holding_cost,
    }
    return model, metrics, params


def run():
    if not FEATURES_PATH.exists():
        print("No features found -- run src/features/build_features.py first.")
        sys.exit(1)

    df = pd.read_parquet(FEATURES_PATH)

    # MLflow's file-based store is now in maintenance mode (3.x) -- SQLite is
    # the recommended lightweight backend for local/course-project use.
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("retail-demand-forecasting")

    with mlflow.start_run() as run:
        model, metrics, params = train_model(df)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(model, artifact_path="model", registered_model_name="retail-demand-model")

        model.booster_.save_model(str(MODEL_OUT_PATH))  # also save a plain copy for the FastAPI app to load fast

        print(f"Run ID: {run.info.run_id}")
        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:,.3f}")

    return metrics


if __name__ == "__main__":
    run()
