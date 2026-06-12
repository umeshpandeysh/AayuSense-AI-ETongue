# AayuSense API Reference

> Complete reference for all public modules in the AayuSense-AI-ETongue project.

---

## Table of Contents
1. [data_pipeline](#data_pipeline)
2. [feature_engineering](#feature_engineering)
3. [models.random_forest_model](#modelsrandom_forest_model)
4. [models.xgboost_model](#modelsxgboost_model)
5. [predict](#predict)

---

## `data_pipeline`

Handles loading, cleaning, and preprocessing of raw sensor CSV data.

### `load_raw_data(filepath: str) -> pd.DataFrame`

Load and validate raw sensor data from a CSV file.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| `filepath` | `str` | Path to the raw CSV file. |

**Returns** — `pd.DataFrame` with validated sensor columns.

**Raises** — `FileNotFoundError` if the file does not exist; `ValueError` if required columns are absent.

```python
from src.data_pipeline import load_raw_data
df = load_raw_data('data/raw/sample_sensor_data.csv')
print(df.shape)  # (500, 6)
```

---

### `handle_missing_values(df: pd.DataFrame) -> pd.DataFrame`

Fill missing values using column medians for numeric columns.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| `df` | `pd.DataFrame` | Input DataFrame possibly containing NaN values. |

**Returns** — DataFrame with no missing values.

```python
df_clean = handle_missing_values(df)
assert df_clean.isnull().sum().sum() == 0
```

---

### `remove_outliers(df: pd.DataFrame, columns: List[str], z_threshold: float = 3.0) -> pd.DataFrame`

Remove rows where any specified column exceeds `z_threshold` standard deviations.

**Parameters**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `df` | `pd.DataFrame` | — | Input DataFrame. |
| `columns` | `List[str]` | — | Columns to check for outliers. |
| `z_threshold` | `float` | `3.0` | Z-score threshold. |

**Returns** — Filtered DataFrame.

```python
df_clean = remove_outliers(df, columns=['ph', 'conductivity', 'turbidity', 'orp'])
```

---

### `normalize_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame`

Apply min-max normalisation to specified columns, scaling to [0, 1].

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| `df` | `pd.DataFrame` | Input DataFrame. |
| `columns` | `List[str]` | Columns to normalise. |

**Returns** — DataFrame with normalised columns.

```python
df_norm = normalize_features(df, columns=['ph', 'conductivity', 'turbidity', 'orp'])
```

---

## `feature_engineering`

Transforms raw sensor readings into a richer feature set for ML models.

### `compute_rolling_statistics(df: pd.DataFrame, columns: List[str], window: int = 5) -> pd.DataFrame`

Add rolling mean and rolling std columns for each sensor.

**Parameters**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `df` | `pd.DataFrame` | — | Input DataFrame. |
| `columns` | `List[str]` | — | Sensor columns to process. |
| `window` | `int` | `5` | Rolling window size. |

**Returns** — DataFrame with added `{col}_roll_mean` and `{col}_roll_std` columns.

```python
df_fe = compute_rolling_statistics(df, columns=['ph', 'conductivity'], window=5)
```

---

### `compute_range_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame`

Compute peak-to-trough range within a rolling window.

**Returns** — DataFrame with `{col}_range` columns added.

---

### `compute_gradient_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame`

Compute first-order difference (gradient) for each sensor column.

**Returns** — DataFrame with `{col}_gradient` columns.

---

### `compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame`

Create pairwise interaction terms between the four sensor readings.

Generated features include:
- `ph_x_conductivity` — product of pH and conductivity
- `turbidity_over_orp` — turbidity divided by ORP
- `ph_over_turbidity` — pH divided by turbidity
- `conductivity_x_orp` — product of conductivity and ORP

**Returns** — DataFrame with interaction columns appended.

---

### `extract_features(df: pd.DataFrame) -> pd.DataFrame`

Full feature extraction pipeline. Applies rolling statistics, range, gradient, and interaction features in sequence, then drops NaN rows introduced by windowing.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| `df` | `pd.DataFrame` | Raw or cleaned sensor DataFrame. |

**Returns** — Feature-enriched DataFrame ready for model training.

```python
from src.feature_engineering import extract_features
df_features = extract_features(df_clean)
print(df_features.shape)  # (n_rows, n_features + 1 label)
```

---

## `models.random_forest_model`

### `class RandomForestModel`

Scikit-learn Random Forest classifier with GridSearchCV hyper-parameter tuning.

#### `__init__(self, config: ModelConfig)`

```python
from src.config import ModelConfig
from src.models.random_forest_model import RandomForestModel

model = RandomForestModel(ModelConfig())
```

#### `train(self, X_train, y_train) -> RandomForestModel`

Fit using 5-fold GridSearchCV over `n_estimators`, `max_depth`, `min_samples_split`.

```python
model.train(X_train, y_train)
print(model.best_params_)   # e.g. {'n_estimators': 200, 'max_depth': 10, ...}
print(model.cv_score_)      # e.g. 0.9732
```

#### `predict(self, X) -> np.ndarray`
Return predicted class labels.

#### `predict_proba(self, X) -> np.ndarray`
Return class probability matrix of shape `(n_samples, n_classes)`.

#### `get_feature_importance(self, feature_names) -> Dict[str, float]`
Return feature importances as a `{name: score}` dict sorted by score (descending).

#### `save(self, filepath: str)` / `load(cls, filepath: str)`
Joblib serialisation / deserialisation.

---

## `models.xgboost_model`

### `class XGBoostModel`

Identical interface to `RandomForestModel` but uses `XGBClassifier` with `eval_metric='mlogloss'`.

GridSearch over `n_estimators`, `max_depth`, `learning_rate`.

> **Note:** Requires `pip install xgboost`. Gracefully warns if not installed.

```python
from src.models.xgboost_model import XGBoostModel
model = XGBoostModel(ModelConfig())
model.train(X_train, y_train)
```

---

## `predict`

Provides single-sample and batch inference utilities.

### `load_model_and_encoder(model_path: str, encoder_path: str)`

Load serialised model and `LabelEncoder` from disk.

```python
model, encoder = load_model_and_encoder('models/rf_model.joblib', 'models/label_encoder.joblib')
```

---

### `predict_single_reading(ph, conductivity, turbidity, orp, model, encoder, ...) -> Dict`

Classify one sensor reading.

**Returns**
```python
{
    'label': 'Genuine',
    'confidence': 0.9341,
    'probabilities': {
        'Adulterant_A': 0.0213,
        'Adulterant_B': 0.0211,
        'Adulterant_C': 0.0235,
        'Genuine': 0.9341,
    }
}
```

---

### `predict_from_csv(csv_path, model_path, encoder_path, output_path=None) -> pd.DataFrame`

Batch-predict from CSV. Adds `predicted_label` and `confidence` columns.

```python
results = predict_from_csv(
    csv_path='data/raw/sample_sensor_data.csv',
    model_path='models/rf_model.joblib',
    encoder_path='models/label_encoder.joblib',
    output_path='data/processed/predictions.csv',
)
```

---

### CLI Usage

```bash
# Single reading
python src/predict.py \
    --model models/rf_model.joblib \
    --encoder models/label_encoder.joblib \
    --ph 6.8 --conductivity 1.2 --turbidity 45 --orp 180

# Batch CSV
python src/predict.py \
    --model models/rf_model.joblib \
    --encoder models/label_encoder.joblib \
    --csv data/raw/sample_sensor_data.csv
```
