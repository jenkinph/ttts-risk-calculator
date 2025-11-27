import json
import joblib
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="TTTS Risk Calculator",
    layout="wide",
)

# =========================================================
# Load models + features
# =========================================================
@st.cache_resource
def load_models_and_features():
    base = Path(__file__).resolve().parent
    pod1_model = joblib.load(base / "pod1_xgb_platt.joblib")
    live_model = joblib.load(base / "live_birth_xgb_iso.joblib")

    with open(base / "pod1_features.json") as f:
        pod1_features = json.load(f)
    with open(base / "live_features_for_xgb.json") as f:
        live_features = json.load(f)

    return pod1_model, live_model, pod1_features, live_features


pod1_model, live_model, pod1_features, live_features = load_models_and_features()

# =========================================================
# UI Styling
# =========================================================
st.markdown("""
# **TTTS Risk Calculator**
This tool estimates:
- **POD1 donor demise risk**
- **Likelihood of donor live birth**

based on calibrated XGBoost models developed from clinical TTTS laser data.
""")

st.success("Models loaded successfully.")

# Utility
def predict(model, feature_order, input_dict):
    x = np.array([[input_dict[f] for f in feature_order]])
    return float(model.predict_proba(x)[0,1])


def risk_label(p):
    if p < 0.05:
        return ("Low Risk", "🟢")
    elif p < 0.15:
        return ("Moderate Risk", "🟡")
    else:
        return ("High Risk", "🔴")


# =========================================================
# MAIN LAYOUT
# =========================================================
tab1, tab2 = st.tabs(["🔻 Donor POD1 Demise", "👶 Donor Live Birth"])


# =========================================================
# TAB 1 — POD1 DONOR DEMISE
# =========================================================
with tab1:
    st.header("POD1 Donor Demise — Enter Inputs")

    col1, col2 = st.columns(2)

    inputs_pod1 = {}
    with col1:
        st.subheader("Ultrasound Doppler Findings (Binary)")
        for f in pod1_features:
            if "bin" in f:
                inputs_pod1[f] = st.selectbox(f, [0,1])

    with col2:
        st.subheader("Continuous Variables")
        for f in pod1_features:
            if "bin" not in f:
                inputs_pod1[f] = st.number_input(f, value=0.0)

    if st.button("Predict POD1 Donor Demise Risk"):
        p = predict(pod1_model, pod1_features, inputs_pod1)
        label, icon = risk_label(p)

        st.markdown(f"""
        ## **Risk: {p:.3f} ({icon} {label})**
        """)
        st.info("Interpretation:\n- Low: <5%\n- Moderate: 5–15%\n- High: >15%")


# =========================================================
# TAB 2 — LIVE BIRTH
# =========================================================
with tab2:
    st.header("Donor Live Birth — Enter Inputs")

    # Explicit classification
    LIVE_BINARY = [f for f in live_features if f.endswith("_bin")]
    LIVE_CONT   = [f for f in live_features if f not in LIVE_BINARY]

    col1, col2 = st.columns(2)

    inputs_live = {}
    with col1:
        st.subheader("Doppler Findings (Binary)")
        for f in LIVE_BINARY:
            inputs_live[f] = st.selectbox(f, [0, 1], key=f"bin_{f}")

    with col2:
        st.subheader("Continuous Variables")
        for f in LIVE_CONT:
            inputs_live[f] = st.number_input(f, value=0.0, key=f"cont_{f}")

    if st.button("Predict Donor Live Birth Probability"):
        p = predict(live_model, live_features, inputs_live)

        if p > 0.90:
            label = "Very High Probability"
            icon = "🟢"
        elif p > 0.70:
            label = "High Probability"
            icon = "🟡"
        else:
            label = "Lower Probability"
            icon = "🔴"

        st.markdown(f"""
        ## **Probability: {p:.3f} ({icon} {label})**
        """)
