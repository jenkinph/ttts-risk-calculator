]# app.py
import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st


# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="TTTS Risk Calculator (Prototype)",
    page_icon="🍼",
    layout="wide",
)


# ---------------------------------------------------------
# Load models + feature lists (cached)
# ---------------------------------------------------------
@st.cache_resource
def load_models_and_features():
    base_path = Path(__file__).parent

    # Calibrated XGBoost models
    pod1_model = joblib.load(base_path / "pod1_xgb_platt.joblib")
    live_model = joblib.load(base_path / "live_birth_xgb_iso.joblib")

    # Feature name lists used *during training*
    with open(base_path / "pod1_features.json", "r") as f:
        pod1_features = json.load(f)

    with open(base_path / "live_features_for_xgb.json", "r") as f:
        live_features = json.load(f)

    # Make sure they are lists of strings
    pod1_features = list(pod1_features)
    live_features = list(live_features)

    return pod1_model, live_model, pod1_features, live_features


pod1_model, live_model, pod1_features, live_features = load_models_and_features()


# ---------------------------------------------------------
# Small utilities
# ---------------------------------------------------------
def predict_probability(model, feature_order, feature_values_dict):
    """
    Build a 1xN NumPy array in the exact order expected by the model and
    return the class-1 probability.
    """
    x_vec = [feature_values_dict[name] for name in feature_order]
    X = np.array([x_vec], dtype=float)  # shape (1, n_features)
    prob = float(model.predict_proba(X)[0, 1])
    return prob


def binary_input(label: str, key: str):
    """0/1 input with a selectbox, but returns numeric 0 or 1."""
    choice = st.selectbox(label, options=[0, 1], key=key, format_func=lambda v: f"{v}")
    return float(choice)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("TTTS Risk Calculator (Prototype)")
st.markdown(
    """
This prototype exposes calibrated XGBoost models for predicting:
- **Donor POD1 demise** after laser therapy
- **Donor live birth** after laser therapy  

It is intended **only for exploratory and research use**.  
**Do not** use this tool for real-time clinical decision-making.
"""
)

st.success("Models loaded successfully.")


# Optional: quick sanity about feature counts
st.caption(
    f"POD1 model uses **{len(pod1_features)}** features; "
    f"Live-birth model uses **{len(live_features)}** features."
)


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_pod1, tab_live, tab_debug = st.tabs(
    ["☠️ Donor POD1 Demise", "👶 Donor Live Birth", "🧪 Smoke test / debug"]
)


# ---------------------------------------------------------
# TAB 1 — POD1 Donor Demise
# ---------------------------------------------------------
with tab_pod1:
    st.header("POD1 Donor Demise — Enter Inputs")

    st.markdown(
        "Enter the donor Doppler and measurement values below. "
        "All inputs here are treated as numeric (continuous) features, "
        "matching the model’s training set."
    )

    cols = st.columns(2)
    pod1_inputs = {}

    for i, feat in enumerate(pod1_features):
        col = cols[i % 2]
        with col:
            # No binaries in this feature set in practice, but this keeps things flexible
            if "bin" in feat.lower():
                pod1_inputs[feat] = binary_input(feat, key=f"pod1_{feat}")
            else:
                pod1_inputs[feat] = st.number_input(
                    feat,
                    value=0.0,
                    key=f"pod1_{feat}",
                    format="%.2f",
                )

    if st.button("Predict POD1 Donor Demise Risk", key="btn_pod1"):
        try:
            prob = predict_probability(pod1_model, pod1_features, pod1_inputs)
            st.success(f"Predicted probability of POD1 donor demise: **{prob:.3f}**")
        except Exception as e:
            st.error(f"Error computing POD1 risk: {e}")


# ---------------------------------------------------------
# TAB 2 — Donor Live Birth
# ---------------------------------------------------------
with tab_live:
    st.header("Donor Live Birth — Enter Inputs")

    st.markdown(
        "Enter donor Doppler findings and measurements below. "
        "Binary Doppler variables are entered as 0/1, and the rest are continuous."
    )

    # For nicer layout, split into “Doppler (binary)” and “Continuous” columns
    col_bin, col_cont = st.columns(2)

    live_inputs = {}

    for feat in live_features:
        # Heuristic: anything with "bin" in the name is binary
        if "bin" in feat.lower():
            with col_bin:
                live_inputs[feat] = binary_input(feat, key=f"live_{feat}")
        else:
            with col_cont:
                live_inputs[feat] = st.number_input(
                    feat,
                    value=0.0,
                    key=f"live_{feat}",
                    format="%.2f",
                )

    if st.button("Predict Donor Live Birth Probability", key="btn_live"):
        try:
            prob = predict_probability(live_model, live_features, live_inputs)
            st.success(f"Predicted probability of donor live birth: **{prob:.3f}**")
        except Exception as e:
            st.error(f"Error computing live-birth probability: {e}")


# ---------------------------------------------------------
# TAB 3 — Smoke test / debug
# ---------------------------------------------------------
with tab_debug:
    st.header("Smoke Test: Dummy Prediction")

    st.markdown(
        "This tab runs a simple check with all inputs set to 0. "
        "It’s just to confirm that the models execute end-to-end in this environment."
    )

    if st.button("Run smoke test", key="btn_smoke"):
        try:
            # All zeros for POD1
            pod1_zero = {f: 0.0 for f in pod1_features}
            pod1_prob_zero = predict_probability(pod1_model, pod1_features, pod1_zero)

            # All zeros for live-birth
            live_zero = {f: 0.0 for f in live_features}
            live_prob_zero = predict_probability(live_model, live_features, live_zero)

            st.write("**POD1 donor demise (all inputs 0):**", round(pod1_prob_zero, 3))
            st.write(
                "**Donor live birth (all inputs 0):**", round(live_prob_zero, 3)
            )

            st.success("Smoke test completed without model/shape errors.")
        except Exception as e:
            st.error(f"Smoke test failed: {e}")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "This prototype is based on calibrated XGBoost models trained on the TTTS dataset. "
    "It is for research/testing only and **must not** be used for direct clinical care."
)
