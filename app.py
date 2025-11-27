# app.py
import json
from pathlib import Path

import numpy as np
import streamlit as st
import joblib

# These imports help joblib unpickle the models
from sklearn.pipeline import Pipeline  # noqa: F401
from sklearn.impute import SimpleImputer  # noqa: F401
from sklearn.calibration import CalibratedClassifierCV  # noqa: F401
from xgboost import XGBClassifier  # noqa: F401


# -------------------------------------------------------------------
# 1. Paths and cached loading of models + feature lists
# -------------------------------------------------------------------
BASE_PATH = Path(__file__).resolve().parent


@st.cache_resource
def load_models_and_features():
    """Load calibrated XGBoost models and their feature order."""
    pod1_model = joblib.load(BASE_PATH / "pod1_xgb_platt.joblib")
    live_model = joblib.load(BASE_PATH / "live_birth_xgb_iso.joblib")

    with open(BASE_PATH / "pod1_features.json", "r") as f:
        pod1_features = json.load(f)

    with open(BASE_PATH / "live_features_for_xgb.json", "r") as f:
        live_features = json.load(f)

    return pod1_model, live_model, pod1_features, live_features


pod1_model, live_model, pod1_features, live_features = load_models_and_features()

# -------------------------------------------------------------------
# 2. Display-name mapping (raw_name -> nice clinical label)
# -------------------------------------------------------------------
DISPLAY_NAMES = {
    # Shared / continuous
    "EFW Percent don": "Donor Estimated Fetal Weight Percentile",

    # Pre-laser continuous (POD1 + Live Birth)
    "pre UA pi don": "Pre-laser Umbilical Artery PI",
    "pre DV pi don": "Pre-laser Ductus Venosus PI",
    "pre MCA psv don": "Pre-laser MCA Peak Systolic Velocity (PSV)",
    "pre MCA MoM don": "Pre-laser MCA PSV MoM",
    "pre MCA pi don": "Pre-laser MCA Pulsatility Index",

    # Post-laser continuous (POD1 model)
    "post UA pi don": "Post-laser Umbilical Artery PI",
    "post DV pi don": "Post-laser Ductus Venosus PI",
    "post MCA psv don": "Post-laser MCA Peak Systolic Velocity (PSV)",
    "post MCA MoM don": "Post-laser MCA PSV MoM",
    "post MCA pi don": "Post-laser MCA Pulsatility Index",

    # Delta variables (both models)
    "delta_DV_pi_don": "Change in Ductus Venosus PI (Post − Pre)",
    "delta_MCA_psv_don": "Change in MCA PSV (Post − Pre)",
    "delta_MCA_MoM_don": "Change in MCA PSV MoM (Post − Pre)",
    "delta_MCA_pi_don": "Change in MCA Pulsatility Index (Post − Pre)",

    # Binary Doppler abnormalities (Live Birth model)
    "pre UA aedf don_bin": "Pre-laser UA AEDF (Absent End-Diastolic Flow)",
    "pre UA rvdf don_bin": "Pre-laser UA REDF (Reversed End-Diastolic Flow)",
    "pre UA intermittent aedf don_bin": "Pre-laser UA Intermittent AEDF",
    "pre UA intermittent rvdf don_bin": "Pre-laser UA Intermittent REDF",
    "pre_DV_abnormal_bin": "Pre-laser DV Abnormal (A-wave absent or reversed)",
}


# -------------------------------------------------------------------
# 3. Simple helpers
# -------------------------------------------------------------------
def predict_probability(model, feature_order, values_dict):
    """
    Build a 2D numpy array in the correct feature order and
    return the predicted probability of the positive class.
    """
    x_vec = np.array([[values_dict[name] for name in feature_order]], dtype=float)
    proba = model.predict_proba(x_vec)[0, 1]
    return float(proba)


def binary_input(label, key, default=0):
    """Binary select with nice labels; returns 0 or 1."""
    choice = st.selectbox(
        label,
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        key=key,
    )
    return int(choice)


# -------------------------------------------------------------------
# 4. App layout
# -------------------------------------------------------------------
st.set_page_config(page_title="TTTS Risk Calculator", layout="wide")

st.title("TTTS Risk Calculator (Prototype)")
st.write(
    "This app exposes calibrated XGBoost models for predicting donor **POD1 demise** "
    "and **donor live birth** after laser therapy for TTTS."
)

st.success("Models loaded successfully.")

st.markdown("---")

tab_pod1, tab_live = st.tabs(
    [
        "🩸 Donor POD1 Demise",
        "👶 Donor Live Birth",
    ]
)

# -------------------------------------------------------------------
# 5. POD1 Donor Demise Tab
# -------------------------------------------------------------------
with tab_pod1:
    st.header("POD1 Donor Demise — Enter Inputs")

    # Group variables for display only; order for the model comes from pod1_features
    pre_vars = [
        "EFW Percent don",
        "pre UA pi don",
        "pre DV pi don",
        "pre MCA psv don",
        "pre MCA MoM don",
        "pre MCA pi don",
    ]
    post_vars = [
        "post UA pi don",
        "post DV pi don",
        "post MCA psv don",
        "post MCA MoM don",
        "post MCA pi don",
    ]
    delta_vars = [
        "delta_DV_pi_don",
        "delta_MCA_psv_don",
        "delta_MCA_MoM_don",
        "delta_MCA_pi_don",
    ]

    # Initialize dictionary for all feature values (float)
    pod1_values = {name: 0.0 for name in pod1_features}

    with st.form("pod1_form"):
        col_pre, col_post, col_delta = st.columns(3)

        with col_pre:
            st.subheader("Pre-laser Doppler")
            for raw in pre_vars:
                if raw in pod1_values:
                    label = DISPLAY_NAMES.get(raw, raw)
                    pod1_values[raw] = st.number_input(
                        label,
                        key=f"pod1_{raw}",
                        value=0.0,
                        format="%.2f",
                    )

        with col_post:
            st.subheader("Post-laser Doppler")
            for raw in post_vars:
                if raw in pod1_values:
                    label = DISPLAY_NAMES.get(raw, raw)
                    pod1_values[raw] = st.number_input(
                        label,
                        key=f"pod1_{raw}",
                        value=0.0,
                        format="%.2f",
                    )

        with col_delta:
            st.subheader("Change (Post − Pre)")
            for raw in delta_vars:
                if raw in pod1_values:
                    label = DISPLAY_NAMES.get(raw, raw)
                    pod1_values[raw] = st.number_input(
                        label,
                        key=f"pod1_{raw}",
                        value=0.0,
                        format="%.2f",
                    )

        submitted_pod1 = st.form_submit_button("Predict POD1 Donor Demise Risk")

    if submitted_pod1:
        try:
            prob = predict_probability(pod1_model, pod1_features, pod1_values)
            st.success(f"Predicted probability of donor POD1 demise: **{prob:.3f}**")
        except Exception as e:
            st.error(f"Error computing POD1 risk: {e}")

# -------------------------------------------------------------------
# 6. Donor Live Birth Tab
# -------------------------------------------------------------------
with tab_live:
    st.header("Donor Live Birth — Enter Inputs")

    # Group for display:
    live_binary_vars = [
        "pre UA aedf don_bin",
        "pre UA rvdf don_bin",
        "pre UA intermittent aedf don_bin",
        "pre UA intermittent rvdf don_bin",
        "pre_DV_abnormal_bin",
    ]
    live_cont_vars = [
        "EFW Percent don",
        "pre UA pi don",
        "pre DV pi don",
        "pre MCA psv don",
        "pre MCA MoM don",
        "pre MCA pi don",
        "delta_DV_pi_don",
        "delta_MCA_psv_don",
        "delta_MCA_MoM_don",
        "delta_MCA_pi_don",
    ]

    live_values = {}

    with st.form("live_form"):
        col_bin, col_cont = st.columns(2)

        with col_bin:
            st.subheader("Doppler Abnormalities (Binary)")
            for raw in live_binary_vars:
                if raw in live_features:
                    label = DISPLAY_NAMES.get(raw, raw)
                    live_values[raw] = binary_input(label, key=f"live_{raw}", default=0)

        with col_cont:
            st.subheader("Continuous Doppler Measures")
            for raw in live_cont_vars:
                if raw in live_features:
                    label = DISPLAY_NAMES.get(raw, raw)
                    live_values[raw] = st.number_input(
                        label,
                        key=f"live_{raw}",
                        value=0.0,
                        format="%.2f",
                    )

        submitted_live = st.form_submit_button("Predict Donor Live Birth Probability")

    if submitted_live:
        # Ensure that all features in live_features have values (default 0 if missing)
        for name in live_features:
            live_values.setdefault(name, 0.0)

        try:
            prob = predict_probability(live_model, live_features, live_values)
            st.success(f"Predicted probability of donor live birth: **{prob:.3f}**")
        except Exception as e:
            st.error(f"Error computing live birth probability: {e}")

# -------------------------------------------------------------------
# 7. Footer disclaimer
# -------------------------------------------------------------------
st.markdown("---")
st.caption(
    "This prototype is based on calibrated XGBoost models trained on a TTTS cohort. "
    "It is intended for research and educational use only and must **not** be used "
    "for real-time clinical decision-making."
)
