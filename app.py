import streamlit as st
import joblib
import json
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="TTTS Risk Calculator", layout="wide")

st.title("TTTS Risk Calculator (Prototype)")
st.write("This app exposes the calibrated XGBoost models for predicting "
         "donor POD1 demise and live birth after laser therapy.")

# --------------------------------------------------
# Load models + feature lists
# --------------------------------------------------
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
st.success("Models loaded successfully.")


# --------------------------------------------------
# Show feature lists
# --------------------------------------------------
st.header("Model inputs (feature names)")

st.subheader("POD1 donor demise model features:")
st.code(str(pod1_features))

st.subheader("Live birth model features:")
st.code(str(live_features))


# --------------------------------------------------
# POD1 – Input Form
# --------------------------------------------------
st.header("POD1 Donor Demise — Enter Inputs")

pod1_inputs = {}

# Create Streamlit inputs based on feature type
for feat in pod1_features:
    if feat.endswith("_bin"):
        val = st.selectbox(feat, [0, 1], help="0 = No, 1 = Yes")
        pod1_inputs[feat] = val
    else:
        val = st.number_input(feat, value=0.0, step=0.1)
        pod1_inputs[feat] = val

if st.button("Predict POD1 Demise Risk"):
    X = pd.DataFrame([pod1_inputs])[pod1_features]
    prob = pod1_model.predict_proba(X)[0, 1]
    st.success(f"Predicted POD1 donor demise probability: **{prob:.3f}**")


# --------------------------------------------------
# Smoke test at bottom
# --------------------------------------------------
st.header("Smoke Test: Dummy Prediction")
if st.button("Run smoke test"):
    X0 = pd.DataFrame([[0]*len(pod1_features)], columns=pod1_features)
    p0 = pod1_model.predict_proba(X0)[0, 1]
    st.write(f"POD1 model outputs: {p0:.4f}")
