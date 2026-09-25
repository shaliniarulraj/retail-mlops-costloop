"""
Converts forecast error into simulated business cost.

This is the novel core of the project: instead of judging a model purely
on RMSE/MAPE, we ask "how much did being wrong actually cost the business?"

    under-forecast (predicted < actual)  -> stockout cost (lost sale)
    over-forecast  (predicted > actual)  -> holding cost (excess inventory)

These per-unit costs are configurable per SKU/category in a real deployment
(a loaf of bread spoiling is a different cost than a t-shirt sitting on a
shelf) -- defaults below are placeholders you should replace with real
estimates once you have them.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_STOCKOUT_COST_PER_UNIT = 25.0   # e.g. INR -- lost margin + customer goodwill
DEFAULT_HOLDING_COST_PER_UNIT = 4.0     # e.g. INR per unit per review period


@dataclass
class CostBreakdown:
    total_cost: float
    stockout_cost: float
    holding_cost: float
    n_rows: int

    @property
    def avg_cost_per_row(self) -> float:
        return self.total_cost / self.n_rows if self.n_rows else 0.0


def compute_cost(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    stockout_cost_per_unit: float = DEFAULT_STOCKOUT_COST_PER_UNIT,
    holding_cost_per_unit: float = DEFAULT_HOLDING_COST_PER_UNIT,
) -> CostBreakdown:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    under_forecast = np.clip(actual - predicted, a_min=0, a_max=None)   # stockout units
    over_forecast = np.clip(predicted - actual, a_min=0, a_max=None)    # excess units

    stockout_cost = float(np.sum(under_forecast) * stockout_cost_per_unit)
    holding_cost = float(np.sum(over_forecast) * holding_cost_per_unit)

    return CostBreakdown(
        total_cost=stockout_cost + holding_cost,
        stockout_cost=stockout_cost,
        holding_cost=holding_cost,
        n_rows=len(actual),
    )


def cost_by_group(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    group_cols: list[str],
    stockout_cost_per_unit: float = DEFAULT_STOCKOUT_COST_PER_UNIT,
    holding_cost_per_unit: float = DEFAULT_HOLDING_COST_PER_UNIT,
) -> pd.DataFrame:
    """Per-store or per-SKU cost breakdown -- useful for spotting *where*
    the model is bleeding money, not just that it is."""
    rows = []
    for keys, g in df.groupby(group_cols):
        cb = compute_cost(g[actual_col], g[pred_col], stockout_cost_per_unit, holding_cost_per_unit)
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append((*keys, cb.total_cost, cb.stockout_cost, cb.holding_cost, cb.n_rows))
    cols = group_cols + ["total_cost", "stockout_cost", "holding_cost", "n_rows"]
    return pd.DataFrame(rows, columns=cols).sort_values("total_cost", ascending=False)


if __name__ == "__main__":
    # quick sanity demo
    actual = pd.Series([100, 50, 0, 30])
    predicted = pd.Series([80, 70, 5, 30])   # under, over, over, exact
    cb = compute_cost(actual, predicted)
    print(f"Total cost   : {cb.total_cost:,.2f}")
    print(f"Stockout cost: {cb.stockout_cost:,.2f}  (from under-forecasting)")
    print(f"Holding cost : {cb.holding_cost:,.2f}  (from over-forecasting)")
    print(f"Avg cost/row : {cb.avg_cost_per_row:,.2f}")
