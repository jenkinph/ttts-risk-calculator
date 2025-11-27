import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# App config
# ---------------------------------------------------------
st.set_page_config(
    page_title="TTTS Risk Calculator (Prototype)",
    layout="wide",
)

st.title("TTTS Risk Calculator (Prototype)")
st.write(
    "This app exposes the calibrated XGBoost models for predicting donor POD1 demise "
    "and donor live birth after laser therapy."
)


# ---------------------------------------------------------
# Load models & feature lists
# ---------------------------------------------------------
@st.cache_resource
def load_models_and_features():
    base_path = Path(__file__).parent

    pod1_model = joblib.load(base_path / "pod1_xgb_platt.joblib")
    live_model = joblib.load(base_path / "live_birth_xgb_iso.joblib")

    with open(base_path / "pod1_features.json", "r") as f:
        pod1_features = json.load(f)

    with open(base_path / "live_features_for_xgb.json", "r") as f:
        live_features = json.load(f)

    return pod1_model, live_model, pod1_features, live_features


try:
    pod1_model, live_model, pod1_features, live_features = load_models_and_features()
    st.success("Models loaded successfully.")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.markdown("---")


# ---------------------------------------------------------
# Helper: build DF and predict
# ---------------------------------------------------------
def predict_with_model(model, feature_names, value_dict):
    """
    value_dict: dict {feature_name: value}
    feature_names: list of feature names in the correct order
    """
    row = {name: float(value_dict.get(name, 0.0)) for name in feature_names}
    X = pd.DataFrame([row], columns=feature_names)
    prob = float(model.predict_proba(X)[0, 1])
    return prob


# ---------------------------------------------------------
# Feature groupings for the UI (binary vs continuous)
# ---------------------------------------------------------

# POD1 donor demise model
pod1_binary_features = [
    "pre UA aedf don_bin",
    "pre UA rvdf don_bin",
    "pre UA intermittent aedf don_bin",
    "pre UA intermittent rvdf don_bin",
    "pre_DV_abnormal_bin",
]

pod1_cont_features = [
    "EFW Percent don",
    "pre UA pi don",
    "pre DV pi don",
    "pre MCA psv don",
    "pre MCA MoM don",
    "pre MCA pi don",
]

# Live birth model
live_binary_features = [
    "pre UA aedf don_bin",
    "pre UA rvdf don_bin",
    "pre_DV_abnormal_bin",
    "post_DV_abnormal_bin",
]

live_cont_features = [
    "EFW Percent don",
    "pre UA pi don",
    "pre DV pi don",
    "pre MCA psv don",
    "pre MCA MoM don",
    "pre MCA pi don",
    "post UA pi don",
    "post DV pi don",
    "post MCA psv don",
    "post MCA MoM don",
    "post MCA pi don",
]

# Sanity check: union of our UI feature lists should match JSON lists.
# (If not, model will still work but we might be missing an input.)
# We don't stop the app here, just log to the page.
if set(pod1_binary_features + pod1_cont_features) != set(pod1_features):
    st.warning(
        "POD1 feature list mismatch between app UI and saved feature list. "
        "Model will still run, but please confirm feature names."
    )

if set(live_binary_features + live_cont_features) != set(live_features):
    st.warning(
        "Live-birth feature list mismatch between app UI and saved feature list. "
        "Model will still run, but please confirm feature names."
    )

# ---------------------------------------------------------
# Tabs for the two outcomes
# ---------------------------------------------------------
tab_pod1, tab_live = st.tabs(["🩸 Donor POD1 Demise", "👶 Donor Live Birth"])


# ---------------------------------------------------------
# TAB 1: POD1 DONOR DEMISE
# ---------------------------------------------------------
with tab_pod1:
    st.header("POD1 Donor Demise — Enter Inputs")

    col_bin, col_cont = st.columns(2)

    pod1_values = {}

    with col_bin:
        st.subheader("Ultrasound Doppler Findings (Binary)")
        st.caption("0 = No / Normal, 1 = Yes / Abnormal")

        for f in pod1_binary_features:
            pretty_label = f.replace("_", " ")
            val = st.selectbox(
                pretty_label,
                options=[0, 1],
                index=0,
                key=f"pod1_bin_{f}",
            )
            pod1_values[f] = val

    with col_cont:
        st.subheader("Continuous Variables")
        for f in pod1_cont_features:
            pretty_label = f.replace("_", " ")
            val = st.number_input(
                pretty_label,
                value=0.0,
                step=0.01,
                key=f"pod1_cont_{f}",
            )
            pod1_values[f] = val

    if st.button("Predict POD1 Donor Demise Risk", key="btn_pod1"):
        try:
            prob = predict_with_model(pod1_model, pod1_features, pod1_values)
            st.success(f"Predicted probability of POD1 donor demise: {prob:.3f}")
            st.write(f"Equivalent to **{prob*100:.1f}%** risk.")
        except Exception as e:
            st.error(f"Error during POD1 prediction: {e}")


# ---------------------------------------------------------
# TAB 2: DONOR LIVE BIRTH
# ---------------------------------------------------------
with tab_live:
    st.header("Donor Live Birth — Enter Inputs")

    col_bin2, col_cont2 = st.columns(2)

    live_values = {}

    with col_bin2:
        st.subheader("Doppler Findings (Binary)")
        st.caption("0 = No / Normal, 1 = Yes / Abnormal")

        for f in live_binary_features:
            pretty_label = f.replace("_", " ")
            val = st.selectbox(
                pretty_label,
                options=[0, 1],
                index=0,
                key=f"live_bin_{f}",
            )
            live_values[f] = val

    with col_cont2:
        st.subheader("Continuous Variables")
        for f in live_cont_features:
            pretty_label = f.replace("_", " ")
            val = st.number_input(
                pretty_label,
                value=0.0,
                step=0.01,
                key=f"live_cont_{f}",
            )
            live_values[f] = val

    if st.button("Predict Donor Live Birth Probability", key="btn_live"):
        try:
            prob = predict_with_model(live_model, live_features, live_values)
            st.success(f"Predicted probability of donor live birth: {prob:.3f}")
            st.write(f"Equivalent to **{prob*100:.1f}%** probability of live birth.")
        except Exception as e:
            st.error(f"Error during live-birth prediction: {e}")
