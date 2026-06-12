# AayuSense System Integration Guide

This document describes how the three layers of AayuSense work together:
the ESP32 hardware, the Python ML backend (this repo), and the Next.js web dashboard.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    AayuSense System                      │
├──────────────┬──────────────────────┬────────────────────┤
│ Hardware     │ Python ML Backend    │ Web Dashboard      │
│ (ESP32)      │ (this repo)          │ (Swayam-jhaa repo) │
├──────────────┼──────────────────────┼────────────────────┤
│ ESP32-01     │ esp32_interface.py   │ Next.js app        │
│ ESP32-02     │ data_pipeline.py     │ aayu-sense.vercel  │
│ 8 sensors    │ feature_engineering  │ .app               │
│              │ train_evaluate.py    │ Clerk auth         │
│              │ explainability.py    │ Radix UI           │
└──────────────┴──────────────────────┴────────────────────┘
```

## ESP32 → Python Data Flow

### 1. ESP32 Firmware Output

The ESP32 firmware samples all 8 sensors and emits JSON:

```json
{
  "sample_id": "S20250916-01",
  "device_id": "ESP32-01",
  "timestamp": "2025-09-16T09:12:00Z",
  "sensors": {
    "pH": 6.1,
    "TDS": 210,
    "orp_mV": 345,
    "turbidity": 0.14,
    "Reduction_value": 0.120,
    "Ionic_value": 0.080,
    "Salt_content": 0.220,
    "temp_c": 25.2
  },
  "herb_name": "Neem (Azadirachta indica)",
  "operator": "Umesh"
}
```

### 2. Python Parsing

```python
from src.esp32_interface import ESP32DataParser
parser = ESP32DataParser(strict=True)
reading = parser.parse(json_string)  # returns SensorReading dataclass
print(reading.sensor_vector())       # {'pH': 6.1, 'TDS': 210, ...}
```

### 3. Feature Engineering

```python
from src.data_pipeline import preprocess_pipeline
from src.feature_engineering import extract_features

df = preprocess_pipeline("data/raw/sample_sensor_data.csv")
features = extract_features(df)  # adds rolling stats, Rasa, interactions
```

### 4. Prediction + SHAP

```python
from src.explainability import AayuSenseExplainer
explainer = AayuSenseExplainer(trained_model, feature_names)
explainer.fit(X_train)
shap_values = explainer.compute_shap_values(X_test)
top_features = explainer.get_top_features(shap_values, class_index=0)
```

Output matches the web dashboard SHAP format:
```json
[
  {"feature": "Salt_content", "impact": 0.38},
  {"feature": "pH", "impact": 0.31},
  {"feature": "turbidity", "impact": 0.15}
]
```

## Web Dashboard Integration

The Next.js dashboard at [aayu-sense.vercel.app](https://aayu-sense.vercel.app) displays:
- Real-time sensor readings
- Classification result (Genuine / Adulterated)
- Adulteration probability score (0.0–1.0, threshold 0.5)
- Rasa intensity radar chart
- SHAP feature impact breakdown
- Historical sample results

See the web repository: [github.com/Swayam-jhaa/AayuSense](https://github.com/Swayam-jhaa/AayuSense)

## Running the Development Stack

```bash
# 1. Start the Python Streamlit dev dashboard
streamlit run dashboard/app.py

# 2. Or use the sensor simulator directly
python src/sensor_simulator.py

# 3. Train models on your dataset
python src/train_evaluate.py
```
