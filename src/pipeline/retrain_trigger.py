"""
Decides whether the current production model needs retraining, based on
BUSINESS COST -- not raw accuracy drift.

The idea: a model can drift statistically without actually costing the
business much (e.g. error is symmetric and cheap). Conversely a small
accuracy drop during a festival week, where stockouts are very costly,
might warrant immediate retraining. Cost is the real decision variable.

The baseline cost is pulled LIVE from whatever model currently holds the
'production' alias in MLflow -- not hardcoded -- so this script stays
correct across dataset swaps (synthetic -> real, or any future retrain)
without manual edits.

Run standalone (uses a simulated "next batch" of production data):
    python src/pipeline/retrain_trigger.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cost.cost_calculator import compute_cost, cost_by_group  # noqa: E402
from src.training.train import FEATURE_COLS, TARGET  # noqa: E402

MODEL_PATH = ROOT / "data" / "processed" / "model.txt"
MODEL_NAME = "retail-demand-model"
PRODUCTION_ALIAS = "production"
METRIC_KEY = "business_cost_avg_per_row"

# Fallback ONLY used if no production model exists yet in the registry
# (e.g. very first run before anything has been promoted).
FALLBACK_BASELINE = 150.0

# Threshold: if avg cost-per-row on the latest production batch exceeds
# this multiple of the baseline cost, trigger a retrain. Tune once you
# have a sense of acceptable cost variance for your own data.
COST_INCREASE_THRESHOLD_MULTIPLIER = 1.5


def get_production_baseline_cost() -> float:
    """Pulls business_cost_avg_per_row from whichever model version
    currently holds the 'production' alias -- this is the number new
    production batches get compared against."""
    try:
        client = MlflowClient(tracking_uri=f"sqlite:///{ROOT / 'mlflow.db'}")
        mv = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
        run = client.get_run(mv.run_id)
        return float(run.data.metrics[METRIC_KEY])
    except Exception as e:
        print(f"Could not fetch production baseline from MLflow ({e}). "
              f"Using fallback baseline = {FALLBACK_BASELINE}.")
        return FALLBACK_BASELINE


def evaluate_production_batch(df: pd.DataFrame, model: lgb.Booster) -> dict:
    X = df[FEATURE_COLS]
    y = df[TARGET]
    preds = np.clip(model.predict(X), a_min=0, a_max=None)

    cb = compute_cost(y.values, preds)
    by_store = cost_by_group(
        df.assign(_pred=preds), actual_col=TARGET, pred_col="_pred", group_cols=["store_id"]
    )
    return {"cost_breakdown": cb, "by_store": by_store, "preds": preds}


def should_retrain(avg_cost_per_row: float, baseline: float) -> bool:
    return avg_cost_per_row > baseline * COST_INCREASE_THRESHOLD_MULTIPLIER


def simulate_drifted_batch(features_path: Path, n_days: int = 14) -> pd.DataFrame:
    """Grabs the most recent n_days as a stand-in for 'new production data',
    and injects a synthetic demand shock to prove the trigger actually fires."""
    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=n_days)
    batch = df[df["date"] > cutoff].copy()

    # Inject an unseen demand shock on one store (e.g. a local event the
    # model has no feature for) -- works regardless of which store_id
    # naming the current dataset uses, as long as it exists in the batch.
    target_store = sorted(batch["store_id"].unique())[0]
    shock_mask = batch["store_id"] == target_store
    batch.loc[shock_mask, "sales"] = (batch.loc[shock_mask, "sales"] * 2.8).astype(int)
    return batch


def run():
    model = lgb.Booster(model_file=str(MODEL_PATH))
    batch = simulate_drifted_batch(ROOT / "data" / "processed" / "features.parquet")
    baseline = get_production_baseline_cost()

    result = evaluate_production_batch(batch, model)
    cb = result["cost_breakdown"]

    print(f"Production batch size : {cb.n_rows} rows")
    print(f"Avg cost per row       : {cb.avg_cost_per_row:,.2f}  (baseline: {baseline:,.2f})")
    print(f"Stockout cost          : {cb.stockout_cost:,.2f}")
    print(f"Holding cost           : {cb.holding_cost:,.2f}")
    print("\nCost by store (top offenders):")
    print(result["by_store"].head())

    trigger = should_retrain(cb.avg_cost_per_row, baseline)
    print(f"\n{'>>> RETRAIN TRIGGERED <<<' if trigger else 'No retrain needed -- cost within threshold.'}")
    return trigger


if __name__ == "__main__":
    triggered = run()
    sys.exit(0 if not triggered else 2)  # non-zero exit lets CI/CD branch on this
