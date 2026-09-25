"""
A minimal interface simulating what a store manager would see: a system
recommendation, with the ability to override it and give a reason.

Run standalone:
    streamlit run src/feedback/override_app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.feedback.db import log_override, get_unfolded_overrides  # noqa: E402
from src.serving.app import get_model, FEATURE_COLS  # noqa: E402

st.set_page_config(page_title="Store Manager Console", layout="centered")
st.title("📦 Reorder Recommendation Console")
st.caption("Simulated store-manager interface for the human-in-the-loop feedback loop")

FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"


@st.cache_data
def load_latest_rows():
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    return df[df["date"] == latest_date]


df = load_latest_rows()
model = get_model()

store_id = st.selectbox("Store", sorted(df["store_id"].unique()))
sku_id = st.selectbox("SKU", sorted(df["sku_id"].unique()))

row = df[(df["store_id"] == store_id) & (df["sku_id"] == sku_id)]
if row.empty:
    st.warning("No data for this store/SKU on the latest date.")
else:
    row_df = row[FEATURE_COLS].astype(float).iloc[[0]]
    row = row.iloc[0]
    pred = float(max(model.predict(row_df)[0], 0))

    st.metric("System-recommended order quantity", f"{pred:.1f} units")
    st.write(f"Forecast date: {row['date'].date()}")

    st.subheader("Manager override")
    override_value = st.number_input(
        "Your order quantity", min_value=0.0, value=round(pred, 1), step=1.0
    )
    reason = st.text_input("Reason for override (optional, but trains the model better)")

    if st.button("Submit override"):
        if abs(override_value - pred) < 0.01:
            st.info("No change from system recommendation -- nothing logged.")
        else:
            log_override(
                store_id=store_id,
                sku_id=sku_id,
                forecast_date=str(row["date"].date()),
                system_recommendation=pred,
                manager_override=override_value,
                override_reason=reason,
            )
            st.success(f"Logged: system said {pred:.1f}, you set {override_value:.1f}.")

st.divider()
st.subheader("Pending overrides (not yet folded into training)")
pending = get_unfolded_overrides()
if pending:
    st.dataframe(pd.DataFrame(pending))
else:
    st.write("None yet.")