import json
from pathlib import Path

import numpy as np
import streamlit as st
import joblib


# -----------------------------------------------------------------------------
# Load models and feature lists
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    base_path = Path(__file__).resolve().parent
    pod1_model = joblib.load(base_path / "pod1_xgb_platt.joblib")
    live_model = joblib.load(base_path / "live_birth_xgb_iso.joblib")
    return pod1_model, live_model


@st.cache_data
def load_feature_lists():
    base_path = Path(__file__).resolve().parent
    with open(base_path / "pod1_features.json", "r") as f:
        pod1_features = json.load(f)
    with open(base_path / "live_features_for_xgb.json", "r") as f:
        live_features = json.load(f)
    return pod1_features, live_features


pod1_model, live_model = load_models()
pod1_features, live_features = load_feature_lists()

# -----------------------------------------------------------------------------
# Human-friendly metadata for each feature
# -----------------------------------------------------------------------------
# Note: bounds are deliberately generous but still physiologically reasonable.
FEATURE_META = {
    # Shared / general
    "EFW Percent don": dict(
        label="Estimated fetal weight percentile (donor)",
        group="General",
        min=0.0,
        max=100.0,
        step=1.0,
        default=50.0,
        help="Estimated fetal weight percentile for the donor fetus (0–100).",
        type="continuous",
    ),

    # Binary Doppler findings (live-birth model)
    "pre UA aedf don_bin": dict(
        label="Pre-laser UA absent end-diastolic flow (AEDF)",
        group="Pre-laser Doppler (binary)",
        type="binary",
        default=0,
        help="1 if umbilical artery shows persistent absent end-diastolic flow before laser; 0 if not.",
    ),
    "pre UA rvdf don_bin": dict(
        label="Pre-laser UA reversed end-diastolic flow (REDF)",
        group="Pre-laser Doppler (binary)",
        type="binary",
        default=0,
        help="1 if umbilical artery shows persistent reversed end-diastolic flow before laser; 0 if not.",
    ),
    "pre UA intermittent aedf don_bin": dict(
        label="Pre-laser UA intermittent AEDF",
        group="Pre-laser Doppler (binary)",
        type="binary",
        default=0,
        help="1 if AEDF is intermittent before laser; 0 if not.",
    ),
    "pre UA intermittent rvdf don_bin": dict(
        label="Pre-laser UA intermittent REDF",
        group="Pre-laser Doppler (binary)",
        type="binary",
        default=0,
        help="1 if REDF is intermittent before laser; 0 if not.",
    ),
    "pre_DV_abnormal_bin": dict(
        label="Pre-laser abnormal DV waveform",
        group="Pre-laser Doppler (binary)",
        type="binary",
        default=0,
        help="1 if ductus venosus waveform is abnormal (e.g., absent/reversed a-wave) before laser; 0 if normal.",
    ),

    # Pre-laser continuous indices
    "pre UA pi don": dict(
        label="Pre-laser UA PI (donor)",
        group="Pre-laser Doppler indices",
        min=0.0,
        max=5.0,
        step=0.1,
        default=1.0,
        help="Umbilical artery pulsatility index for the donor fetus before laser.",
        type="continuous",
    ),
    "pre DV pi don": dict(
        label="Pre-laser DV PI (donor)",
        group="Pre-laser Doppler indices",
        min=0.0,
        max=3.0,
        step=0.05,
        default=0.5,
        help="Ductus venosus pulsatility index for the donor fetus before laser.",
        type="continuous",
    ),
    "pre MCA psv don": dict(
        label="Pre-laser MCA peak systolic velocity (cm/s)",
        group="Pre-laser Doppler indices",
        min=0.0,
        max=100.0,
        step=1.0,
        default=30.0,
        help="Middle cerebral artery peak systolic velocity before laser (cm/s).",
        type="continuous",
    ),
    "pre MCA MoM don": dict(
        label="Pre-laser MCA PSV MoM",
        group="Pre-laser Doppler indices",
        min=0.0,
        max=3.0,
        step=0.05,
        default=1.0,
        help="Multiple of the median (MoM) for MCA peak systolic velocity before laser.",
        type="continuous",
    ),
    "pre MCA pi don": dict(
        label="Pre-laser MCA PI (donor)",
        group="Pre-laser Doppler indices",
        min=0.0,
        max=5.0,
        step=0.1,
        default=1.5,
        help="Middle cerebral artery pulsatility index for the donor fetus before laser.",
        type="continuous",
    ),

    # Post-laser continuous indices (POD1 demise model only)
    "post UA pi don": dict(
        label="Post-laser UA PI (donor)",
        group="Post-laser Doppler indices",
        min=0.0,
        max=5.0,
        step=0.1,
        default=1.0,
        help="Umbilical artery pulsatility index for the donor fetus after laser.",
        type="continuous",
    ),
    "post DV pi don": dict(
        label="Post-laser DV PI (donor)",
        group="Post-laser Doppler indices",
        min=0.0,
        max=3.0,
        step=0.05,
        default=0.5,
        help="Ductus venosus pulsatility index for the donor fetus after laser.",
        type="continuous",
    ),
    "post MCA psv don": dict(
        label="Post-laser MCA peak systolic velocity (cm/s)",
        group="Post-laser Doppler indices",
        min=0.0,
        max=100.0,
        step=1.0,
        default=30.0,
        help="Middle cerebral artery peak systolic velocity after laser (cm/s).",
        type="continuous",
    ),
    "post MCA MoM don": dict(
        label="Post-laser MCA PSV MoM",
        group="Post-laser Doppler indices",
        min=0.0,
        max=3.0,
        step=0.05,
        default=1.0,
        help="Multiple of the median (MoM) for MCA peak systolic velocity after laser.",
        type="continuous",
    ),
    "post MCA pi don": dict(
        label="Post-laser MCA PI (donor)",
        group="Post-laser Doppler indices",
        min=0.0,
        max=5.0,
        step=0.1,
        default=1.5,
        help="Middle cerebral artery pulsatility index for the donor fetus after laser.",
        type="continuous",
    ),

    # Delta features
    "delta_DV_pi_don": dict(
        label="Δ DV PI (post − pre)",
        group="Change from pre- to post-laser",
        min=-3.0,
        max=3.0,
        step=0.05,
        default=0.0,
        help="Change in ductus venosus PI (post-laser minus pre-laser).",
        type="continuous",
    ),
    "delta_MCA_psv_don": dict(
        label="Δ MCA PSV (post − pre, cm/s)",
        group="Change from pre- to post-laser",
        min=-50.0,
        max=50.0,
        step=1.0,
        default=0.0,
        help="Change in MCA peak systolic velocity (cm/s).",
        type="continuous",
    ),
    "delta_MCA_MoM_don": dict(
        label="Δ MCA PSV MoM (post − pre)",
        group="Change from pre- to post-laser",
        min=-3.0,
        max=3.0,
        step=0.05,
        default=0.0,
        help="Change in MCA PSV MoM (post minus pre).",
        type="continuous",
    ),
    "delta_MCA_pi_don": dict(
        label="Δ MCA PI (post − pre)",
        group="Change from pre- to post-laser",
        min=-3.0,
        max=3.0,
        step=0.05,
        default=0.0,
        help="Change in MCA PI (post minus pre).",
        type="continuous",
    ),
}


# -----------------------------------------------------------------------------
# Helper: render inputs for a model and return values as a dict
# -----------------------------------------------------------------------------
def render_feature_inputs(feature_names, outcome_label: str):
    """
    Render Streamlit widgets for all features in feature_names.
    Returns a dict: {feature_name: value}.
    """
    values = {}
    current_group = None

    for feat in feature_names:
        meta = FEATURE_META.get(feat, {})
        group = meta.get("group", None)

        if group and group != current_group:
            st.subheader(group)
            current_group = group

        ftype = meta.get("type", "continuous")
        label = meta.get("label", feat)
        help_text = meta.get("help", None)

        if ftype == "binary":
            # 0/1 with human labels
            choice = st.selectbox(
                label,
                options=["No / normal (0)", "Yes / abnormal (1)"],
                index=meta.get("default", 0),
                help=help_text,
                key=f"{outcome_label}_{feat}",
            )
            value = 1 if "abnormal" in choice.lower() or "yes" in choice.lower() else 0

        else:  # continuous
            min_v = meta.get("min", 0.0)
            max_v = meta.get("max", 10.0)
            step = meta.get("step", 0.1)
            default = meta.get("default", 0.0)
            value = st.number_input(
                label,
                min_value=float(min_v),
                max_value=float(max_v),
                value=float(default),
                step=float(step),
                help=help_text,
                key=f"{outcome_label}_{feat}",
            )

        values[feat] = float(value)

    return values


def predict_probability(model, features, values_dict):
    """Builds an input row in the model's feature order and returns P(event=1)."""
    x = np.array([[values_dict[f] for f in features]], dtype=float)
    prob = model.predict_proba(x)[0, 1]
    return float(prob)


# -----------------------------------------------------------------------------
# Streamlit layout
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TTTS Donor Risk Calculator (Prototype)",
    layout="wide",
)

st.title("TTTS Donor Risk Calculator (Prototype)")

st.markdown(
    """
This app exposes **research models** built to estimate:

- The probability of **donor POD1 demise** after laser therapy, and  
- The probability of **donor live birth**.

> ⚠️ **Important:** This is a **prototype research tool**, not a clinical device.  
> Predictions are based on a single institutional dataset and should **not** be used
> for real-time clinical decision-making or patient counseling.
"""
)

st.markdown(
    """
### How to use

1. Choose the outcome tab (POD1 demise or live birth).  
2. Enter the donor Doppler measurements.  
3. Use **0 = normal / absent**, **1 = abnormal / present** for binary findings.  
4. Review the predicted probability and interpret it in the context of the full clinical picture.
"""
)

tabs = st.tabs(
    [
        "🟥 Donor POD1 Demise",
        "🟩 Donor Live Birth",
        "🛠 Developer tools / smoke test",
    ]
)

# -----------------------------------------------------------------------------
# POD1 Donor Demise Tab
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("Donor POD1 Demise — Enter Inputs")

    pod1_values = render_feature_inputs(pod1_features, outcome_label="pod1")

    if st.button("Predict POD1 donor demise risk", type="primary"):
        try:
            prob = predict_probability(pod1_model, pod1_features, pod1_values)
            st.success(f"Predicted probability of donor POD1 demise: **{prob:.3f}**")
        except Exception as e:
            st.error(
                "Error computing POD1 risk. Please double-check inputs or contact the developer."
            )
            st.exception(e)

# -----------------------------------------------------------------------------
# Live Birth Tab
# -----------------------------------------------------------------------------
with tabs[1]:
    st.header("Donor Live Birth — Enter Inputs")

    live_values = render_feature_inputs(live_features, outcome_label="live")

    if st.button("Predict donor live birth probability", type="primary"):
        try:
            prob = predict_probability(live_model, live_features, live_values)
            st.success(f"Predicted probability of donor live birth: **{prob:.3f}**")
        except Exception as e:
            st.error(
                "Error computing live birth probability. Please double-check inputs or contact the developer."
            )
            st.exception(e)

# -----------------------------------------------------------------------------
# Developer / Smoke test tab (optional)
# -----------------------------------------------------------------------------
with tabs[2]:
    st.header("Developer tools / smoke test")

    st.markdown(
        """
This section is mainly for debugging.  
It feeds a simple synthetic case into each model to confirm end-to-end wiring.
"""
    )

    if st.button("Run smoke test"):
        try:
            # "Neutral" patient: all defaults from metadata
            neutral_pod1 = {f: FEATURE_META.get(f, {}).get("default", 0.0) for f in pod1_features}
            neutral_live = {f: FEATURE_META.get(f, {}).get("default", 0.0) for f in live_features}

            pod1_prob = predict_probability(pod1_model, pod1_features, neutral_pod1)
            live_prob = predict_probability(live_model, live_features, neutral_live)

            st.write(f"POD1 demise model – neutral case: **{pod1_prob:.3f}**")
            st.write(f"Live birth model – neutral case: **{live_prob:.3f}**")
        except Exception as e:
            st.error("Smoke test failed.")
            st.exception(e)

st.markdown(
    """
---
**Prototype disclaimer:** These models are exploratory and derived from retrospective data.  
They have **not** been prospectively validated and should not replace clinical judgment.
"""
)
