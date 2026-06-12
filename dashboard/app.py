"""AayuSense Streamlit Dashboard — Herbal Quality Monitor."""
import sys
import pathlib
import random
from datetime import datetime
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    import streamlit as st
except ImportError:
    raise SystemExit("Install streamlit: pip install streamlit")

try:
    import pandas as pd
except ImportError:
    raise SystemExit("Install pandas: pip install pandas")

st.set_page_config(
    page_title="AayuSense — Herbal Quality Monitor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.1);
}
.metric-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(8px);
}
.metric-value { font-size: 2.2rem; font-weight: 700; color: #00d4aa; }
.metric-label { font-size: 0.85rem; color: #aeb8c2; margin-top: 4px; }
.badge-genuine { background: #1a7a4a; color: #fff; padding: 6px 18px; border-radius: 20px; font-size: 1.1rem; font-weight: 600; }
.badge-adulterant { background: #8b1a1a; color: #fff; padding: 6px 18px; border-radius: 20px; font-size: 1.1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

CLASS_PARAMS = {
    'Genuine':      {'ph': (6.8, 0.3),  'conductivity': (1.2, 0.15), 'turbidity': (45, 8),   'orp': (180, 25)},
    'Adulterant_A': {'ph': (5.2, 0.4),  'conductivity': (2.8, 0.3),  'turbidity': (120, 20), 'orp': (95, 30)},
    'Adulterant_B': {'ph': (7.9, 0.5),  'conductivity': (0.4, 0.1),  'turbidity': (25, 10),  'orp': (310, 40)},
    'Adulterant_C': {'ph': (6.1, 0.6),  'conductivity': (1.9, 0.25), 'turbidity': (85, 15),  'orp': (140, 35)},
}

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / 'models'

if 'history' not in st.session_state:
    st.session_state['history']: List[Dict] = []
if 'current_reading' not in st.session_state:
    st.session_state['current_reading'] = None


def simulate_reading(label: str) -> Dict:
    """Generate a synthetic sensor reading for a given quality class."""
    params = CLASS_PARAMS[label]
    rng = np.random.default_rng()
    return {
        'timestamp':    datetime.now().strftime('%H:%M:%S'),
        'ph':           round(float(rng.normal(*params['ph'])), 3),
        'conductivity': round(float(rng.normal(*params['conductivity'])), 3),
        'turbidity':    round(float(rng.normal(*params['turbidity'])), 3),
        'orp':          round(float(rng.normal(*params['orp'])), 3),
        'quality_label': label,
    }


def mock_classify(reading: Dict, model_type: str) -> Dict:
    """Rule-based mock classifier used when no model files exist."""
    label      = reading['quality_label']
    confidence = round(random.uniform(0.78, 0.97), 3)
    return {'label': label, 'confidence': confidence, 'model_used': f'{model_type} (mock)'}


def try_load_model(model_type: str):
    """Attempt to load a real trained model; return (None, None) on failure."""
    suffix = 'rf' if model_type == 'Random Forest' else 'xgb'
    model_file   = MODELS_DIR / f'{suffix}_model.joblib'
    encoder_file = MODELS_DIR / 'label_encoder.joblib'
    if model_file.exists() and encoder_file.exists():
        try:
            import joblib
            return joblib.load(str(model_file)), joblib.load(str(encoder_file))
        except Exception as exc:
            st.warning(f"Model load error: {exc}")
    return None, None


# ---- Sidebar ----
with st.sidebar:
    st.image("https://img.shields.io/badge/SIH-2025%20Qualifier-orange?style=for-the-badge", width=220)
    st.title("🌿 AayuSense")
    st.caption("AI-Powered E-Tongue for Herbal Quality Assessment")
    st.divider()
    model_choice = st.selectbox("🤖 Model", ['Random Forest', 'XGBoost'])
    sample_type  = st.selectbox("🧪 Sample Class", list(CLASS_PARAMS.keys()))
    simulate_btn = st.button("▶ Simulate New Reading", use_container_width=True, type="primary")
    st.divider()
    st.markdown("### 📊 Session Stats")
    total         = len(st.session_state['history'])
    genuine_count = sum(1 for r in st.session_state['history'] if r.get('predicted_label') == 'Genuine')
    st.metric("Total Readings", total)
    st.metric("Genuine Detected", genuine_count)
    if total:
        st.metric("Adulterant Rate", f"{round((total - genuine_count) / total * 100, 1)}%")

# ---- Main ----
st.markdown("## 🌿 AayuSense — Herbal Quality Monitor")
st.caption("Real-time E-Tongue sensor classification | SIH 2025")

if simulate_btn:
    reading = simulate_reading(sample_type)
    model, encoder = try_load_model(model_choice)
    if model is not None and encoder is not None:
        try:
            from src.predict import predict_single_reading
            result = predict_single_reading(
                reading['ph'], reading['conductivity'],
                reading['turbidity'], reading['orp'],
                model, encoder,
            )
            classification = {'label': result['label'], 'confidence': result['confidence'], 'model_used': model_choice}
        except Exception as exc:
            st.warning(f"Prediction error: {exc}")
            classification = mock_classify(reading, model_choice)
    else:
        classification = mock_classify(reading, model_choice)

    reading['predicted_label'] = classification['label']
    reading['confidence']      = classification['confidence']
    reading['model_used']      = classification['model_used']
    st.session_state['current_reading'] = reading
    st.session_state['history'].append(reading)
    if len(st.session_state['history']) > 50:
        st.session_state['history'] = st.session_state['history'][-50:]

current = st.session_state.get('current_reading')

col1, col2, col3, col4 = st.columns(4)
for col, label, key, unit in [
    (col1, "🧪 pH",          'ph',           '0-14 scale'),
    (col2, "⚡ Conductivity", 'conductivity', 'mS/cm'),
    (col3, "💧 Turbidity",   'turbidity',    'NTU'),
    (col4, "🔋 ORP",         'orp',          'mV'),
]:
    with col:
        val = current[key] if current else '—'
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{val}</div><div class="metric-label">{unit}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("")
st.markdown("### 🔍 Classification Result")
if current:
    is_genuine = current['predicted_label'] == 'Genuine'
    badge_cls  = 'badge-genuine' if is_genuine else 'badge-adulterant'
    icon       = '✅' if is_genuine else '⚠️'
    col_res, col_conf, col_model = st.columns([2, 1, 1])
    with col_res:
        st.markdown(f'<span class="{badge_cls}">{icon} {current["predicted_label"]}</span>', unsafe_allow_html=True)
    with col_conf:
        st.metric("Confidence", f"{round(current['confidence'] * 100, 1)}%")
    with col_model:
        st.metric("Model", current.get('model_used', model_choice))
else:
    st.info("Click **▶ Simulate New Reading** in the sidebar to begin.")
    if not (MODELS_DIR / 'rf_model.joblib').exists():
        st.warning("No trained models found in models/. Run python src/train.py first. Using mock classifier until then.")

if st.session_state['history']:
    st.markdown("### 📈 Sensor Readings — Last 20 Samples")
    hist_df  = pd.DataFrame(st.session_state['history']).tail(20)
    chart_df = hist_df[['ph', 'conductivity', 'turbidity', 'orp']].reset_index(drop=True)
    st.line_chart(chart_df, use_container_width=True, height=260)

if st.session_state['history']:
    with st.expander("📋 Prediction History", expanded=False):
        hist_df  = pd.DataFrame(st.session_state['history'])
        disp_cols = [c for c in ['timestamp', 'ph', 'conductivity', 'turbidity', 'orp',
                                  'predicted_label', 'confidence', 'model_used'] if c in hist_df.columns]
        st.dataframe(hist_df[disp_cols].sort_index(ascending=False), use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear History"):
            st.session_state['history'] = []
            st.session_state['current_reading'] = None
            st.rerun()
