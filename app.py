import joblib
from pathlib import Path

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
# Load models
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    base_path = Path(__file__).parent
    pod1_model = joblib.load(base_path / "pod1_xgb_platt.joblib")
    live_model = joblib.load(base_path / "live_birth_xgb_iso.joblib")
    return pod1_model, live_model


try:
    pod1_model, live_model = load_models()
    st.success("Models loaded successfully.")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.markdown("---")

# ---------------------------------------------------------
# Pull feature names directly from the fitted models
# (CalibratedClassifierCV -> estimator_ is the Pipeline)
# ---------------------------------------------------------
try:
    pod1_features = list(pod1_model.estimator_.feature_names_in_)
    live_features = list(live_model.estimator_.feature_names_in_)
except Exception as e:
    st.error(f"Could not read feature names from models: {e}")
    st.stop()

# Group features into "binary" and "continuous" based on name
def split_binary_continuous(features):
    binary = [f for f in features if f.endswith("_bin")]
    cont = [f for f in features if f not in binary]
    return binary, cont


pod1_binary_features, pod1_cont_features = split_binary_continuous(pod1_features)
live_binary_features, live_cont_features = split_binary_continuous(live_features)

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
