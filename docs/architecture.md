# AayuSense System Architecture

## Overview

AayuSense is a three-layer system:
1. **Hardware Layer** — Sensor array + microcontroller
2. **Software Layer** — Data pipeline + ML engine
3. **Application Layer** — Real-time dashboard

## Data Flow

```
Sensor Array
    ↓ Analog signals
ADC + Microcontroller
    ↓ Digital readings (Serial/USB)
Data Acquisition Pipeline (Python)
    ↓ Raw CSV
Preprocessing Module
    ↓ Cleaned, normalized features
Feature Engineering Module
    ↓ ML-ready feature matrix
ML Classification Engine (RF + XGBoost)
    ↓ Class prediction + confidence
Dashboard (Streamlit)
    ↓
Operator / End User
```

## ML Pipeline Details

- **Algorithm 1**: Random Forest (ensemble of decision trees)
- **Algorithm 2**: XGBoost (gradient-boosted trees)
- **Validation**: Stratified k-fold cross-validation (k=5)
- **Selection criterion**: Weighted F1-score
- **Evaluation**: Precision, Recall, F1, Confusion Matrix
