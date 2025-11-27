import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------
# Paths & model loading
# ---------------------------------------------------------------------
BASE_PATH = Path(__file__).parent


@st.cache_resource
def load_models():
    """Load calibrated XGBoost models from .joblib files."""
    pod1_model = joblib.load(BASE_PATH / "pod1_xgb_platt.joblib")
    live_model = joblib.load(BASE_PATH / "live_birth_xgb_iso.joblib")
    return pod1_model, live_model


pod1_model, live_model = load_models()

# ---------------------------------------------------------------------
# Feature definitions – MUST MATCH TRAINED MODELS
# ---------------------------------------------------------------------

# POD1 donor demise model features (15)
POD1_FEATURES = [
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
    "delta_DV_pi_don",
    "delta_MCA_psv_don",
    "delta_MCA_MoM_don",
    "delta_MCA_pi_don",
]

# Donor live-birth model features (15)
LIVE_FEATURES = [
    "EFW Percent don",
    "pre UA aedf don_bin",
    "pre UA rvdf don_bin",
    "pre UA intermittent aedf don_bin",
    "pre UA intermittent rvdf don_bin",
    "pre_DV_abnormal_bin",
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

# Human-readable labels for UI
POD1_LABELS = {
    "EFW Percent don": "Donor estimated fetal weight percentile",
    "pre UA pi don": "Pre-laser donor umbilical artery PI",
    "pre DV pi don": "Pre-laser donor ductus venosus PI",
    "pre MCA psv don": "Pre-laser donor MCA peak systolic velocity (cm/s)",
    "pre MCA MoM don": "Pre-laser donor MCA PSV MoM",
    "pre MCA pi don": "Pre-laser donor MCA PI",
    "post UA pi don": "Post-laser donor umbilical artery PI",
    "post DV pi don": "Post-laser donor ductus venosus PI",
    "post MCA psv don": "Post-laser donor MCA peak systolic velocity (cm/s)",
    "post MCA MoM don": "Post-laser donor MCA PSV MoM",
    "post MCA pi don": "Post-laser donor MCA PI",
    "delta_DV_pi_don": "Change in donor DV PI (post – pre)",
    "delta_MCA_psv_don": "Change in donor MCA PSV (post – pre, cm/s)",
    "delta_MCA_MoM_don": "Change in donor MCA PSV MoM (post – pre)",
    "delta_MCA_pi_don": "Change in donor MCA PI (post – pre)",
}

LIVE_LABELS = {
    "EFW Percent don": "Donor estimated fetal weight percentile",
    "pre UA aedf don_bin": "Pre-laser donor UA absent/reversed EDF (0 = no, 1 = yes)",
    "pre UA rvdf don_bin": "Pre-laser donor UA reversed flow (0 = no, 1 = yes)",
    "pre UA intermittent aedf don_bin": "Pre-laser donor UA intermittent absent EDF (0/1)",
    "pre UA intermittent rvdf don_bin": "Pre-laser donor UA intermittent reversed flow (0/1)",
    "pre_DV_abnormal_bin": "Pre-laser donor DV abnormal (a-wave absent/reversed) (0/1)",
    "pre UA pi don": "Pre-laser donor umbilical artery PI",
    "pre DV pi don": "Pre-laser donor ductus venosus PI",
    "pre MCA psv don": "Pre-laser donor MCA peak systolic velocity (cm/s)",
    "pre MCA MoM don": "Pre-laser donor MCA PSV MoM",
    "pre MCA pi don": "Pre-laser donor MCA PI",
    "delta_DV_pi_don": "Change in donor DV PI (post – pre)",
    "delta_MCA_psv_don": "Change in donor MCA PSV (post – pre, cm/s)",
    "delta_MCA_MoM_don": "Change in donor MCA PSV MoM (post – pre)",
    "delta_MCA_pi_don": "Change in donor MCA PI (post – pre)",
}

# Which live-birth features are binary flags
LIVE_BINARY_FEATURES = {
    "pre UA aedf don_bin",
    "pre UA rvdf don_bin",
    "pre UA intermittent aedf don_bin",
    "pre UA intermittent rvdf don_bin",
    "pre_DV_abnormal_bin",
}


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def build_pod1_inputs():
    """Collect POD1 model inputs from the user."""
    st.subheader("Enter donor ultrasound and Doppler values (pre and post laser)")
    cols = st.columns(2)
    values = {}
    prefix = "pod1_"  # unique key prefix for this tab

    # Left column
    with cols[0]:
        values["EFW Percent don"] = st.number_input(
            POD1_LABELS["EFW Percent don"],
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
            key=prefix + "EFW_Percent_don",
        )
        values["pre UA pi don"] = st.number_input(
            POD1_LABELS["pre UA pi don"],
            min_value=0.0,
            max_value=4.0,
            value=1.2,
            step=0.01,
            key=prefix + "pre_UA_pi_don",
        )
        values["pre DV pi don"] = st.number_input(
            POD1_LABELS["pre DV pi don"],
            min_value=0.0,
            max_value=4.0,
            value=0.6,
            step=0.01,
            key=prefix + "pre_DV_pi_don",
        )
        values["pre MCA psv don"] = st.number_input(
            POD1_LABELS["pre MCA psv don"],
            min_value=0.0,
            max_value=80.0,
            value=30.0,
            step=0.5,
            key=prefix + "pre_MCA_psv_don",
        )
        values["pre MCA MoM don"] = st.number_input(
            POD1_LABELS["pre MCA MoM don"],
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.01,
            key=prefix + "pre_MCA_MoM_don",
        )
        values["pre MCA pi don"] = st.number_input(
            POD1_LABELS["pre MCA pi don"],
            min_value=0.0,
            max_value=4.0,
            value=1.6,
            step=0.01,
            key=prefix + "pre_MCA_pi_don",
        )
        values["post UA pi don"] = st.number_input(
            POD1_LABELS["post UA pi don"],
            min_value=0.0,
            max_value=4.0,
            value=1.0,
            step=0.01,
            key=prefix + "post_UA_pi_don",
        )
        values["post DV pi don"] = st.number_input(
            POD1_LABELS["post DV pi don"],
            min_value=0.0,
            max_value=4.0,
            value=0.6,
            step=0.01,
            key=prefix + "post_DV_pi_don",
        )

    # Right column
    with cols[1]:
        values["post MCA psv don"] = st.number_input(
            POD1_LABELS["post MCA psv don"],
            min_value=0.0,
            max_value=80.0,
            value=28.0,
            step=0.5,
            key=prefix + "post_MCA_psv_don",
        )
        values["post MCA MoM don"] = st.number_input(
            POD1_LABELS["post MCA MoM don"],
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.01,
            key=prefix + "post_MCA_MoM_don",
        )
        values["post MCA pi don"] = st.number_input(
            POD1_LABELS["post MCA pi don"],
            min_value=0.0,
            max_value=4.0,
            value=1.4,
            step=0.01,
            key=prefix + "post_MCA_pi_don",
        )
        values["delta_DV_pi_don"] = st.number_input(
            POD1_LABELS["delta_DV_pi_don"],
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.01,
            key=prefix + "delta_DV_pi_don",
        )
        values["delta_MCA_psv_don"] = st.number_input(
            POD1_LABELS["delta_MCA_psv_don"],
            min_value=-50.0,
            max_value=50.0,
            value=-2.0,
            step=0.5,
            key=prefix + "delta_MCA_psv_don",
        )
        values["delta_MCA_MoM_don"] = st.number_input(
            POD1_LABELS["delta_MCA_MoM_don"],
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.01,
            key=prefix + "delta_MCA_MoM_don",
        )
        values["delta_MCA_pi_don"] = st.number_input(
            POD1_LABELS["delta_MCA_pi_don"],
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.01,
            key=prefix + "delta_MCA_pi_don",
        )

    # Build DataFrame in correct feature order
    X = pd.DataFrame([[values[f] for f in POD1_FEATURES]], columns=POD1_FEATURES)
    return X


def build_live_inputs():
    """Collect live-birth model inputs from the user."""
    st.subheader("Enter donor findings before laser therapy")
    cols = st.columns(2)
    values = {}
    prefix = "live_"  # unique key prefix for this tab

    with cols[0]:
        values["EFW Percent don"] = st.number_input(
            LIVE_LABELS["EFW Percent don"],
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
            key=prefix + "EFW_Percent_don",
        )

        # Binary Doppler flags as selectboxes
        for f in [
            "pre UA aedf don_bin",
            "pre UA rvdf don_bin",
            "pre UA intermittent aedf don_bin",
            "pre UA intermittent rvdf don_bin",
            "pre_DV_abnormal_bin",
        ]:
            values[f] = st.selectbox(
                LIVE_LABELS[f],
                options=[0, 1],
                index=0,
                format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
                key=prefix + f,
            )

        values["pre UA pi don"] = st.number_input(
            LIVE_LABELS["pre UA pi don"],
            min_value=0.0,
            max_value=4.0,
            value=1.4,
            step=0.01,
            key=prefix + "pre_UA_pi_don",
        )
        values["pre DV pi don"] = st.number_input(
            LIVE_LABELS["pre DV pi don"],
            min_value=0.0,
            max_value=4.0,
            value=0.7,
            step=0.01,
            key=prefix + "pre_DV_pi_don",
        )

    with cols[1]:
        values["pre MCA psv don"] = st.number_input(
            LIVE_LABELS["pre MCA psv don"],
            min_value=0.0,
            max_value=80.0,
            value=30.0,
            step=0.5,
            key=prefix + "pre_MCA_psv_don",
        )
        values["pre MCA MoM don"] = st.number_input(
            LIVE_LABELS["pre MCA MoM don"],
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.01,
            key=prefix + "pre_MCA_MoM_don",
        )
        values["pre MCA pi don"] = st.number_input(
            LIVE_LABELS["pre MCA pi don"],
            min_value=0.0,
            max_value=4.0,
            value=1.6,
            step=0.01,
            key=prefix + "pre_MCA_pi_don",
        )
        values["delta_DV_pi_don"] = st.number_input(
            LIVE_LABELS["delta_DV_pi_don"],
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.01,
            key=prefix + "delta_DV_pi_don",
        )
        values["delta_MCA_psv_don"] = st.number_input(
            LIVE_LABELS["delta_MCA_psv_don"],
            min_value=-50.0,
            max_value=50.0,
            value=-2.0,
            step=0.5,
            key=prefix + "delta_MCA_psv_don",
        )
        values["delta_MCA_MoM_don"] = st.number_input(
            LIVE_LABELS["delta_MCA_MoM_don"],
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.01,
            key=prefix + "delta_MCA_MoM_don",
        )
        values["delta_MCA_pi_don"] = st.number_input(
            LIVE_LABELS["delta_MCA_pi_don"],
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.01,
            key=prefix + "delta_MCA_pi_don",
        )

    X = pd.DataFrame([[values[f] for f in LIVE_FEATURES]], columns=LIVE_FEATURES)
    return X


def predict_probability(model, X: pd.DataFrame) -> float:
    """Run model.predict_proba safely and return class-1 probability."""
    # Sanity check: feature alignment
    if hasattr(model, "feature_names_in_"):
        model_feats = list(model.feature_names_in_)
        if model_feats != list(X.columns):
            st.error(
                "Feature mismatch between app and model. "
                "Please contact the developer."
            )
            st.write("Model expects:", model_feats)
            st.write("App is sending:", list(X.columns))
            st.stop()

    proba = model.predict_proba(X)[0, 1]
    return float(proba)


# ---------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------
st.title("TTTS Donor Risk Calculator (Prototype)")

st.markdown(
    """
This prototype exposes calibrated XGBoost models for:

1. **Donor POD1 demise** after TTTS laser therapy  
2. **Donor live birth**  

⚠️ **Research tool only** – not for clinical decision-making.
"""
)

tab1, tab2 = st.tabs(["🩸 Donor POD1 demise", "👶 Donor live birth"])

with tab1:
    st.markdown(
        "Use this tab to estimate the probability of **donor POD1 demise** "
        "given pre- and post-laser Doppler values."
    )

    X_pod1 = build_pod1_inputs()

    if st.button("Predict POD1 donor demise risk", type="primary"):
        try:
            prob = predict_probability(pod1_model, X_pod1)
            st.success(
                f"Predicted probability of **donor POD1 demise**: **{prob:.3f}**"
            )
        except Exception as e:
            st.error(f"Error computing POD1 risk: {e}")

with tab2:
    st.markdown(
        "Use this tab to estimate the probability of **donor live birth** "
        "based on pre-laser Doppler findings."
    )

    X_live = build_live_inputs()

    if st.button("Predict donor live-birth probability", type="primary"):
        try:
            prob = predict_probability(live_model, X_live)
            st.success(
                f"Predicted probability of **donor live birth**: **{prob:.3f}**"
            )
        except Exception as e:
            st.error(f"Error computing live-birth probability: {e}")

st.markdown(
    """
---
**Disclaimer:** These models were trained on a single TTTS cohort and are
for exploratory / research use only. They are **not validated** for bedside
clinical decision-making.
"""
)
