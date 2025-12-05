import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any
import os
import sys

# Resolve project root: /home/aispry/gx_openlineage_pipeline
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))      # .../src/frontend
SRC_DIR = os.path.dirname(CURRENT_DIR)                        # .../src
PROJECT_ROOT = os.path.dirname(SRC_DIR)                       # .../
INT_FEATURES = {"dow", "is_weekend", "week_of_year", "month"}

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.inference import (
    FEATURE_COLS,
    predict_single_sku_with_ci,
    predict_batch_from_df,
    get_latest_features_for_sku,
)
from src.prediction.prediction import predict_sku


st.set_page_config(
    page_title="Hospital SKU Demand Forecasting",
    layout="wide",
)


@st.cache_data
def load_sku_master(path: str = "skus_master_full.csv") -> pd.DataFrame:
    """Load SKU master data if the CSV is available."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_transactions(path: str = "inventory_transactions_sample.csv") -> pd.DataFrame:
    """Load sample inventory transactions if the CSV is available."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        return pd.DataFrame()


def feature_form(initial: Optional[Dict[str, Any]], key_prefix: str = "single") -> Dict[str, float]:
    """Render inputs for all model features and return a dict of values."""
    values: Dict[str, float] = {}
    cols = st.columns(3)
    initial = initial or {}

    for i, col_name in enumerate(FEATURE_COLS):
        with cols[i % 3]:
            default = initial.get(col_name, 0.0)
            if col_name == "is_weekend":
                default_bool = bool(default)
                val = st.checkbox(
                    col_name,
                    value=default_bool,
                    key=f"{key_prefix}_{col_name}",
                )
                values[col_name] = float(int(val))
            else:
                try:
                    default_float = float(default)
                except (TypeError, ValueError):
                    default_float = 0.0
                val = st.number_input(
                    col_name,
                    value=default_float,
                    key=f"{key_prefix}_{col_name}",
                )
                values[col_name] = float(val)
    return values


def main():
    st.title("Hospital SKU Demand Forecasting")

    sku_master = load_sku_master()
    transactions = load_transactions()

    # ---- Sidebar: choose SKU ----
    st.sidebar.header("SKU selection")

    selected_sku_id: int

    if not sku_master.empty:
        sku_master_sorted = sku_master.sort_values("sku_id")
        options = sku_master_sorted.to_dict("records")

        def format_sku(option: Dict[str, Any]) -> str:
            return f"{option['sku_id']} - {option.get('sku_code', '')} - {option.get('sku_name', '')}"

        selected_row = st.sidebar.selectbox(
            "Choose a SKU",
            options=options,
            format_func=format_sku,
        )
        selected_sku_id = int(selected_row["sku_id"])
        st.sidebar.write(f"Selected SKU ID: **{selected_sku_id}**")
    else:
        selected_sku_id = int(
            st.sidebar.number_input("SKU ID", min_value=1, step=1, value=1)
        )

    # ---- Tabs ----
    tab_info, tab_single, tab_batch, tab_forecast = st.tabs(
        ["SKU overview", "Single prediction", "Batch prediction", "Multi-day forecast"]
    )

    # ---- Tab 1: Overview ----
    with tab_info:
        st.subheader("SKU overview")

        if not sku_master.empty:
            sku_details = sku_master[sku_master["sku_id"] == selected_sku_id]
            if sku_details.empty:
                st.warning("SKU not found in master file.")
            else:
                st.markdown("**Master data**")
                st.dataframe(sku_details)

        if not transactions.empty:
            st.markdown("**Historical consumption (sample)**")
            sku_tx = transactions[transactions["sku_id"] == selected_sku_id]
            if sku_tx.empty:
                st.info("No transactions found for this SKU in the sample file.")
            else:
                st.dataframe(sku_tx.head(50))
                agg = sku_tx.agg(
                    total_qty=("quantity_consumed", "sum"),
                    total_revenue=("total_revenue", "sum"),
                    avg_margin_pct=("margin_percentage", "mean"),
                )
                col1, col2, col3 = st.columns(3)
                col1.metric("Total qty consumed", f"{agg['total_qty']:.0f}")
                col2.metric("Total revenue", f"{agg['total_revenue']:.2f}")
                col3.metric("Avg margin %", f"{agg['avg_margin_pct']:.2f}")

    # ---- Tab 2: Single prediction ----
    with tab_single:
        st.subheader("Single-day prediction with confidence interval")

        if "single_features" not in st.session_state:
            st.session_state.single_features = {col: 0.0 for col in FEATURE_COLS}

        col_load, col_info = st.columns([1, 3])
        with col_load:
            if st.button("Load latest features from feature store"):
                try:
                    latest_features = get_latest_features_for_sku(selected_sku_id)
                    st.session_state.single_features = latest_features
                    st.success("Loaded latest features from ClickHouse / feature store.")
                except Exception as e:
                    st.error(f"Could not load latest features: {e}")

        with col_info:
            st.markdown(
                "Use the button to pull the latest engineered features for this SKU, "
                "then adjust any fields before running the prediction."
            )

        with st.form("single_prediction_form"):
            features = feature_form(
                initial=st.session_state.single_features,
                key_prefix="single",
            )
            submitted = st.form_submit_button("Predict demand")
            if submitted:
                try:
                
                    result = predict_single_sku_with_ci(
                        sku_id=selected_sku_id,
                        features=features,
                    )
                    st.success("Prediction succeeded.")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Point forecast", f"{result['prediction']:.2f}")
                    c2.metric("Lower 95% CI", f"{result['lower_95']:.2f}")
                    c3.metric("Upper 95% CI", f"{result['upper_95']:.2f}")

                    with st.expander("Raw result"):
                        st.json(result)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # ---- Tab 3: Batch prediction ----
    with tab_batch:
        st.subheader("Batch prediction from CSV")

        st.markdown(
            "Upload a CSV with the following columns:\n\n"
            "- `sku_id`\n"
            "- all engineered feature columns used in the model (`FEATURE_COLS`)."
        )

        uploaded_file = st.file_uploader("Upload feature CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                batch_df = None

            if batch_df is not None:
                st.markdown("**Preview**")
                st.dataframe(batch_df.head())

                if st.button("Run batch prediction"):
                    try:
                        results = predict_batch_from_df(batch_df)
                        results_df = pd.DataFrame(results)
                        st.markdown("**Results**")
                        st.dataframe(results_df)

                        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download results as CSV",
                            data=csv_bytes,
                            file_name="batch_predictions.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")

    # ---- Tab 4: Multi-day forecast ----
    with tab_forecast:
        st.subheader("Multi-day forecast for a SKU")

        horizon = st.slider(
            "Forecast horizon (days)",
            min_value=1,
            max_value=90,
            value=7,
        )

        if st.button("Generate forecast"):
            try:
                forecast_df = predict_sku(selected_sku_id, horizon)
                st.markdown("**Forecast table**")
                st.dataframe(forecast_df)

                # Try to plot if there is a date-like column
                if hasattr(forecast_df, "columns"):
                    df = forecast_df.copy()
                    date_col = None
                    for candidate in ["ds", "date", "day", "forecast_date"]:
                        if candidate in df.columns:
                            date_col = candidate
                            break

                    if date_col:
                        try:
                            df[date_col] = pd.to_datetime(df[date_col])
                            df = df.set_index(date_col)
                            numeric_cols = df.select_dtypes("number")
                            if not numeric_cols.empty:
                                st.line_chart(numeric_cols)
                        except Exception:
                            pass
            except Exception as e:
                st.error(f"Forecast failed: {e}")


if __name__ == "__main__":
    main()
