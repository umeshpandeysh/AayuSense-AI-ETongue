"""
data_pipeline.py
AayuSense-AI-ETongue — Data Loading & Preprocessing

Handles loading raw sensor CSV data from the AayuSense ESP32 hardware,
validation against the 8-sensor schema, cleaning, and normalization.

Sensor columns match the real AayuSense ESP32 firmware output:
  pH, TDS, orp_mV, turbidity, Reduction_value, Ionic_value, Salt_content, temp_c
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# 8 sensor columns matching the real ESP32 firmware output
SENSOR_COLUMNS = [
    "pH", "TDS", "orp_mV", "turbidity",
    "Reduction_value", "Ionic_value", "Salt_content", "temp_c"
]
LABEL_COLUMN = "quality_label"


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw sensor data from a CSV file.

    Args:
        filepath: Path to the CSV file containing sensor readings.

    Returns:
        DataFrame with raw sensor readings.

    Raises:
        FileNotFoundError: If filepath does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    preview = pd.read_csv(filepath, nrows=0)
    parse_dates = ["timestamp"] if "timestamp" in preview.columns else []
    df = pd.read_csv(filepath, parse_dates=parse_dates)
    logger.info("Loaded %d rows from %s", len(df), filepath)
    return df


def validate_schema(df: pd.DataFrame, sensor_columns: Optional[list] = None) -> bool:
    """
    Validate that the DataFrame contains required sensor columns.

    Args:
        df: Input DataFrame.
        sensor_columns: Columns to validate (defaults to SENSOR_COLUMNS).

    Returns:
        True if schema is valid.

    Raises:
        ValueError: If required columns are missing.
    """
    cols = sensor_columns or SENSOR_COLUMNS
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {missing}")
    logger.info("Schema validation passed. Columns: %s", cols)
    return True


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using forward-fill then column-mean imputation.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with no missing values in sensor columns.
    """
    initial_nulls = df[SENSOR_COLUMNS].isnull().sum().sum()
    if initial_nulls > 0:
        df[SENSOR_COLUMNS] = (
            df[SENSOR_COLUMNS]
            .ffill()
            .fillna(df[SENSOR_COLUMNS].mean())
        )
        logger.info("Imputed %d missing sensor values", initial_nulls)
    return df


def remove_outliers(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove rows where any sensor reading deviates beyond z_threshold std devs.

    Args:
        df: Input DataFrame.
        z_threshold: Z-score threshold for outlier detection.

    Returns:
        DataFrame with outliers removed.
    """
    from scipy import stats
    z_scores = np.abs(stats.zscore(df[SENSOR_COLUMNS]))
    mask = (z_scores < z_threshold).all(axis=1)
    removed = len(df) - mask.sum()
    if removed > 0:
        logger.info("Removed %d outlier rows (z > %.1f)", removed, z_threshold)
    return df[mask].reset_index(drop=True)


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply min-max normalization to sensor columns.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with sensor columns normalized to [0, 1].
    """
    for col in SENSOR_COLUMNS:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max != col_min:
            df[col] = (df[col] - col_min) / (col_max - col_min)
    logger.info("Min-max normalization applied to %d sensor columns", len(SENSOR_COLUMNS))
    return df


def parse_esp32_payload(payload: dict) -> pd.Series:
    """
    Convert a single parsed ESP32 JSON payload into a DataFrame row.

    Args:
        payload: Dict with 'sensors' key (from ESP32DataParser).

    Returns:
        pandas Series with sensor values and metadata.
    """
    sensors = payload.get("sensors", payload)
    row = {col: sensors.get(col, np.nan) for col in SENSOR_COLUMNS}
    row["sample_id"] = payload.get("sample_id", "")
    row["timestamp"] = payload.get("timestamp", "")
    row["herb_name"] = payload.get("herb_name", "")
    return pd.Series(row)


def preprocess_pipeline(
    filepath: str,
    remove_outliers_flag: bool = True
) -> pd.DataFrame:
    """
    Full preprocessing pipeline: load -> validate -> clean -> normalize.

    Args:
        filepath: Path to raw sensor CSV.
        remove_outliers_flag: Whether to apply outlier removal.

    Returns:
        Cleaned and normalized DataFrame ready for feature engineering.
    """
    logger.info("Starting preprocessing pipeline for: %s", filepath)
    df = load_raw_data(filepath)
    validate_schema(df)
    df = handle_missing_values(df)
    if remove_outliers_flag:
        df = remove_outliers(df)
    df = normalize_features(df)
    logger.info("Preprocessing complete. Output shape: %s", df.shape)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_path = RAW_DATA_DIR / "sample_sensor_data.csv"
    if sample_path.exists():
        processed_df = preprocess_pipeline(str(sample_path))
        out_path = PROCESSED_DATA_DIR / "processed_sensor_data.csv"
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(out_path, index=False)
        logger.info("Saved processed data to %s", out_path)
    else:
        logger.warning("No data found at %s. Add sensor CSV files to data/raw/", sample_path)
