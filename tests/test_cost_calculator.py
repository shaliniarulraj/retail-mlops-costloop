"""
Run: pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cost.cost_calculator import compute_cost, cost_by_group  # noqa: E402


def test_exact_prediction_has_zero_cost():
    cb = compute_cost([10, 20, 30], [10, 20, 30])
    assert cb.total_cost == 0
    assert cb.stockout_cost == 0
    assert cb.holding_cost == 0


def test_under_forecast_triggers_stockout_cost_only():
    cb = compute_cost(actual=[100], predicted=[80], stockout_cost_per_unit=25, holding_cost_per_unit=4)
    assert cb.stockout_cost == 20 * 25
    assert cb.holding_cost == 0
    assert cb.total_cost == 500


def test_over_forecast_triggers_holding_cost_only():
    cb = compute_cost(actual=[50], predicted=[70], stockout_cost_per_unit=25, holding_cost_per_unit=4)
    assert cb.holding_cost == 20 * 4
    assert cb.stockout_cost == 0
    assert cb.total_cost == 80


def test_mixed_batch_matches_manual_calculation():
    actual = [100, 50, 0, 30]
    predicted = [80, 70, 5, 30]
    cb = compute_cost(actual, predicted, stockout_cost_per_unit=25, holding_cost_per_unit=4)
    # under: 20 units @ 25 = 500 | over: 20 + 5 = 25 units @ 4 = 100
    assert cb.stockout_cost == 500
    assert cb.holding_cost == 100
    assert cb.total_cost == 600
    assert cb.avg_cost_per_row == 150


def test_avg_cost_per_row_handles_empty_input():
    cb = compute_cost([], [])
    assert cb.n_rows == 0
    assert cb.avg_cost_per_row == 0.0


def test_cost_by_group_identifies_worst_offender():
    df = pd.DataFrame({
        "store_id": ["A", "A", "B", "B"],
        "actual": [100, 100, 10, 10],
        "pred": [50, 50, 9, 9],   # store A is wildly under-forecast, store B is nearly exact
    })
    result = cost_by_group(df, actual_col="actual", pred_col="pred", group_cols=["store_id"])
    assert result.iloc[0]["store_id"] == "A"  # worst offender sorted first
    assert result.iloc[0]["total_cost"] > result.iloc[1]["total_cost"]


def test_negative_errors_never_produced():
    """Cost should never go negative regardless of input order."""
    actual = np.array([5, 10, 0])
    predicted = np.array([10, 5, 100])
    cb = compute_cost(actual, predicted)
    assert cb.stockout_cost >= 0
    assert cb.holding_cost >= 0
    assert cb.total_cost >= 0
