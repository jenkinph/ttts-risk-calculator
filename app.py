import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="TTTS Risk Calculator (Prototype)",
    layout="wide",
)

# -----------------------------
# Load models + feature lists
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_models_and_features():
    base = Path(__file__).parent

    # Models
    pod1_model = joblib.load(base / "pod1_xgb_platt.joblib")          # Calibrated XGBoost (Platt)
    live_model = joblib.load(base / "live_birth_xgb_iso.joblib")      # Calibrated XGBoost (Isotonic)

    # Feature lists (must match training)
    with open(base / "pod1_features.json", "r") as f:
        pod1_features = json.load(f)

    with open(base / "live_features_for_xgb.json", "r") as f:
        live_features = json.load(f)

    return pod1_model, live_model, pod1_features, live_features


pod1_model, live_model, pod1_features, live_features = load_models_and_features()

st.markdown(
    "## TTTS Risk Calculator (Prototype)\n\n"
    "This app exposes the calibrated XGBoost models for predicting **donor POD1 demise** "
    "and **donor live birth** after laser therapy.\n\n"
    "> ⚠️ Research tool only – not for clinical decision making."
)

st.success("Models loaded successfully.")

# -----------------------------
# Small helpers
# -----------------------------
def split_binary_continuous(feature_names):
    """Very simple heuristic: *_bin treated as binary, everything else numeric."""
    binary = [f for f in feature_names if f.lower().endswith("_bin")]
    continuous = [f for f in feature_names if f not in binary]
    return binary, continuous


def predict_prob(model, feature_order, values_dict):
    """
    Build a 1-row DataFrame in the correct column order and
    return P(y=1) from the calibrated model.
    """
    row = {name: float(values_dict.get(name, 0.0)) for name in feature_order}
    X = pd.DataFrame([row], columns=feature_order)
    proba = model.predict_proba(X)[0, 1]
    return float(proba)


# -----------------------------
# UI – two tabs
# -----------------------------
tab_pod1, tab_live = st.tabs(
    ["🩸 Donor POD1 Demise", "👶 Donor Live Birth"]
)

# --- POD1 Donor Demise Section ---
st.header("POD1 Donor Demise — Enter Inputs")

pod1_inputs = {}
st.subheader("Continuous Variables")

for f in pod1_features:
    pod1_inputs[f] = st.number_input(f, value=0.0)
    
if st.button("Predict POD1 Donor Demise Risk"):
    try:
        X = np.array([[pod1_inputs[f] for f in pod1_features]])
        prob = float(pod1_model.predict_proba(X)[0, 1])
        st.success(f"POD1 donor demise predicted probability: {prob:.3f}")
    except Exception as e:
        st.error(f"Error computing POD1 risk: {str(e)}")

# =============================
# TAB 2 – DONOR LIVE BIRTH
# =============================
with tab_live:
    st.header("Donor Live Birth — Enter Inputs")

    bin_live, cont_live = split_binary_continuous(live_features)

    col_bin_l, col_cont_l = st.columns(2)

    live_inputs = {}

    with col_bin_l:
        st.subheader("Doppler Findings (Binary)")
        st.caption("0 = No / normal, 1 = Yes / abnormal")
        for f in bin_live:
            label = f.replace("_", " ")
            live_inputs[f] = st.selectbox(
                label,
                options=[0, 1],
                format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
                key=f"live_bin_{f}",
            )

    with col_cont_l:
        st.subheader("Continuous Variables")
        for f in cont_live:
            label = f.replace("_", " ")
            live_inputs[f] = st.number_input(
                label,
                value=0.0,
                step=0.1,
                format="%.2f",
                key=f"live_cont_{f}",
            )

    if st.button("Predict Donor Live Birth Probability", type="primary", key="btn_live"):
        try:
            prob = predict_prob(live_model, live_features, live_inputs)
            st.subheader("Result")
            st.metric(
                "Estimated probability of donor live birth",
                f"{prob * 100:.1f} %",
            )

            # Simple interpretation bands around 0.7 / 0.8 that we used in diagnostics
            if prob < 0.5:
                st.warning("Model-estimated probability is **<50%**.")
            elif prob < 0.8:
                st.info("Model-estimated probability is **moderate** (50–80%).")
            else:
                st.success("Model-estimated probability is **high** (≥80%).")

        except Exception as e:
            st.error(f"Error computing live birth probability: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    "---\n"
    "This prototype is based on calibrated XGBoost models trained on the TTTS dataset. "
    "**Do not** use for real-time clinical decisions."
)
