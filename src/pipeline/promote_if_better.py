"""
The actual CI/CD promotion gate: compares the newest trained model's
business cost metric against whatever model currently holds the
'production' alias in the MLflow registry. Only promotes if the new
model is cheaper to run.

This is what makes the deploy step "intelligent" rather than "blind" --
without this, you'd be auto-deploying every retrain regardless of whether
it actually improved anything.

Run standalone (after train.py has produced a new run):
    python src/pipeline/promote_if_better.py
"""
import sys
from pathlib import Path

from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "retail-demand-model"
METRIC_KEY = "business_cost_avg_per_row"   # lower is better
ALIAS = "production"


def get_client() -> MlflowClient:
    return MlflowClient(tracking_uri=f"sqlite:///{ROOT / 'mlflow.db'}")


def get_metric_for_version(client: MlflowClient, version: str) -> float:
    mv = client.get_model_version(MODEL_NAME, version)
    run = client.get_run(mv.run_id)
    return run.data.metrics[METRIC_KEY]


def latest_version(client: MlflowClient) -> str:
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    return str(max(int(v.version) for v in versions))


def run():
    client = get_client()
    new_version = latest_version(client)
    new_cost = get_metric_for_version(client, new_version)

    try:
        current_prod = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        current_cost = get_metric_for_version(client, current_prod.version)
    except Exception:
        current_prod = None
        current_cost = None

    print(f"New model     : version {new_version}, cost/row = {new_cost:,.2f}")
    if current_prod is None:
        print("No model currently in production -- promoting by default.")
        client.set_registered_model_alias(MODEL_NAME, ALIAS, new_version)
        print(f"Promoted version {new_version} to '{ALIAS}'.")
        return True

    print(f"Current prod  : version {current_prod.version}, cost/row = {current_cost:,.2f}")

    if new_cost < current_cost:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, new_version)
        print(f"PROMOTED: version {new_version} beats production ({new_cost:,.2f} < {current_cost:,.2f}).")
        return True
    else:
        print(f"REJECTED: version {new_version} does not beat production "
              f"({new_cost:,.2f} >= {current_cost:,.2f}). Production unchanged.")
        return False


if __name__ == "__main__":
    promoted = run()
    sys.exit(0 if promoted else 1)
