"""
app.py
AayuSense-AI-ETongue — Real-Time Analytics Dashboard

Streamlit-based dashboard for live classification output
and sensor data visualization.

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="AayuSense | Herbal Quality Dashboard",
    page_icon="🌿",
    layout="wide"
)

# --- Title ---
st.title("🌿 AayuSense — Herbal Quality Assessment Dashboard")
st.markdown("> Real-time AI-powered electronic tongue analysis for herbal sample quality classification.")

st.divider()

# --- Sidebar ---
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("**AayuSense v1.0**")
st.sidebar.markdown("SIH 2025 Qualifier Project")

# --- Model Loading ---
model_path = Path("models/RandomForest_model.pkl")
label_path = Path("models/label_encoder.pkl")

model_loaded = model_path.exists() and label_path.exists()
if model_loaded:
    model = joblib.load(model_path)
    le = joblib.load(label_path)
    st.sidebar.success("✅ Model loaded successfully")
else:
    st.sidebar.warning("⚠️ No trained model found. Train a model first using src/train_evaluate.py")

# --- Live Input Simulation ---
st.subheader("🔬 Sensor Input")
col1, col2, col3, col4 = st.columns(4)

with col1:
    ph_val = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.8, step=0.1)
with col2:
    cond_val = st.number_input("Conductivity (mS/cm)", min_value=0.0, max_value=100.0, value=1.2, step=0.1)
with col3:
    turb_val = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=1000.0, value=45.0, step=1.0)
with col4:
    orp_val = st.number_input("ORP (mV)", min_value=-500.0, max_value=500.0, value=180.0, step=1.0)

# --- Classification ---
if st.button("🔍 Classify Sample", type="primary"):
    if model_loaded:
        # NOTE: Feature vector must match trained feature set
        input_features = np.array([[ph_val, cond_val, turb_val, orp_val]])
        try:
            prediction = model.predict(input_features)
            label = le.inverse_transform(prediction)[0]
            proba = model.predict_proba(input_features)[0]
            st.success(f"🏷️ Classification Result: **{label}**")
            st.bar_chart({cls: p for cls, p in zip(le.classes_, proba)})
        except Exception as e:
            st.error(f"Prediction error: {e}. Ensure the model was trained with matching features.")
    else:
        # Demo output when no model is trained
        st.info("📊 Demo Mode: Train a model to enable live classification.")
        st.success("🏷️ Demo Result: **Genuine Sample** (placeholder)")

st.divider()

# --- Data Upload Section ---
st.subheader("📂 Batch Analysis")
uploaded = st.file_uploader("Upload sensor CSV for batch classification", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head(20))
    st.markdown(f"**Rows loaded:** {len(df)}")

    # Visualization
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        fig, axes = plt.subplots(1, min(len(numeric_cols), 4), figsize=(14, 3))
        if len(numeric_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, numeric_cols[:4]):
            ax.hist(df[col].dropna(), bins=20, color="steelblue", edgecolor="white")
            ax.set_title(col)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
        plt.tight_layout()
        st.pyplot(fig)

st.markdown("---")
st.markdown("Built by [Umesh Pandey](https://github.com/umeshpandeysh) | AayuSense v1.0 | SIH 2025")
