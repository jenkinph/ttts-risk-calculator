import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="TTTS Risk Calculator",
    layout="wide",
    page_icon="🍼",
)

# ---------------------------------------------------------
# Load models + feature lists
# ---------------------------------------------------------
@st.cache_resource
def load_models_and_features():
    base_path = Path(__file__).resolve().parent

    pod1_model = joblib.load(base_path / "pod1_xgb_platt.joblib")
    live_model = joblib.load(base_path / "live_birth_xgb_iso.joblib")

    with open(base_path / "pod1_features.json", "r") as f:
        pod1_features = json.load(f)

    with open(base_path / "live_features_for_xgb.json", "r") as f:
        live_features = json.load(f)

    return pod1_model, live_model, pod1_features, live_features


pod1_model, live_model, pod1_features, live_features = load_models_and_features()

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def nice_label(feature_name: str) -> str:
    """
    Turn 'pre_DV_abnormal_bin' into 'pre DV abnormal (binary)' etc.
    Just for display – does not affect model inputs.
    """
    label = feature_name.replace("_don", "").replace("_bin", " (binary)")
    label = label.replace("_", " ")
    return label


def build_input_df(feature_list, values_dict):
    """Return a 1-row DataFrame with columns in the correct feature order."""
    row = {f: values_dict[f] for f in feature_list}
    return pd.DataFrame([row], columns=feature_list)


def format_prob(p: float) -> str:
    return f"{p*100:.1f}%"


# ---------------------------------------------------------
# Main header
# ---------------------------------------------------------
st.title("TTTS Risk Calculator (Prototype)")
st.write(
    "This app exposes calibrated XGBoost models for predicting **donor POD1 demise** "
    "and **donor live birth** after laser therapy in twin–twin transfusion syndrome. "
    "It is a research prototype and **not** a clinical decision support tool."
)

st.success("Models loaded successfully.")

# Optional: technical info in an expander
with st.expander("Show model feature lists (for reference)", expanded=False):
    st.subheader("POD1 donor demise model features")
    st.code(pod1_features, language="python")
    st.subheader("Live birth model features")
    st.code(live_features, language="python")

# Optional: smoke test expander
with st.expander("Smoke test / debug"):
    if st.button("Run smoke test"):
        # All-zero inputs, but passed as DataFrames with proper columns
        pod1_df = pd.DataFrame([ {f: 0.0 for f in pod1_features} ], columns=pod1_features)
        live_df = pd.DataFrame([ {f: 0.0 for f in live_features} ], columns=live_features)

        pod1_prob = float(pod1_model.predict_proba(pod1_df)[0, 1])
        live_prob = float(live_model.predict_proba(live_df)[0, 1])
        st.write("POD1 model dummy probability:", pod1_prob)
        st.write("Live birth model dummy probability:", live_prob)

st.markdown("---")

# ---------------------------------------------------------
# Tabs for the two outcomes
# ---------------------------------------------------------
tab_pod1, tab_live = st.tabs(
    ["🩸 Donor POD1 Demise", "👶 Donor Live Birth"]
)

# =========================================================
# TAB 1 – POD1 Donor Demise
# =========================================================
with tab_pod1:
    st.header("POD1 Donor Demise — Enter Inputs")

    # Split features into binary vs continuous for UI
    pod1_bin_feats = [f for f in pod1_features if f.endswith("_bin")]
    pod1_cont_feats = [f for f in pod1_features if not f.endswith("_bin")]

    col_bin, col_cont = st.columns(2)

    pod1_inputs = {}

    with col_bin:
        st.subheader("Ultrasound Doppler Findings (Binary)")
        for f in pod1_bin_feats:
            pod1_inputs[f] = st.selectbox(
                nice_label(f),
                options=[0, 1],
                index=0,
                key=f"pod1_bin_{f}",
                help="0 = absent/normal, 1 = present/abnormal",
            )

    with col_cont:
        st.subheader("Continuous Variables")
        for f in pod1_cont_feats:
            pod1_inputs[f] = st.number_input(
                nice_label(f),
                value=0.0,
                step=0.1,
                key=f"pod1_cont_{f}",
            )

    st.markdown("")
    if st.button("Predict POD1 Donor Demise Risk", key="pod1_predict"):
        X = build_input_df(pod1_features, pod1_inputs)
        prob = float(pod1_model.predict_proba(X)[0, 1])

        st.subheader("Prediction")
        st.metric(
            label="Estimated probability of POD1 donor demise",
            value=format_prob(prob),
        )
        st.caption(
            "Model: calibrated XGBoost with Platt (sigmoid) scaling. "
            "Probabilities are approximate and intended for research use only."
        )

# =========================================================
# TAB 2 – Donor Live Birth
# =========================================================
with tab_live:
    st.header("Donor Live Birth — Enter Inputs")

    # Split features into binary vs continuous
    live_bin_feats = [f for f in live_features if f.endswith("_bin")]
    live_cont_feats = [f for f in live_features if not f.endswith("_bin")]

    col_bin_l, col_cont_l = st.columns(2)

    live_inputs = {}

    with col_bin_l:
        st.subheader("Doppler Findings (Binary)")
        if not live_bin_feats:
            st.caption("No binary features in this model.")
        for f in live_bin_feats:
            live_inputs[f] = st.selectbox(
                nice_label(f),
                options=[0, 1],
                index=0,
                key=f"live_bin_{f}",
                help="0 = absent/normal, 1 = present/abnormal",
            )

    with col_cont_l:
        st.subheader("Continuous Variables")
        for f in live_cont_feats:
            live_inputs[f] = st.number_input(
                nice_label(f),
                value=0.0,
                step=0.1,
                key=f"live_cont_{f}",
            )

    st.markdown("")
    if st.button("Predict Donor Live Birth Probability", key="live_predict"):
        X_live = build_input_df(live_features, live_inputs)
        prob_live = float(live_model.predict_proba(X_live)[0, 1])

        st.subheader("Prediction")
        st.metric(
            label="Estimated probability of donor live birth",
            value=format_prob(prob_live),
        )
        st.caption(
            "Model: calibrated XGBoost with isotonic regression. "
            "Probabilities are approximate and intended for research use only."
        )
