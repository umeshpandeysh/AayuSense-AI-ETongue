"""
data_pipeline.py
AayuSense-AI-ETongue — Data Loading & Preprocessing Module

Handles loading raw sensor CSV data, validation, cleaning,
and preparation for the feature engineering pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

SENSOR_COLUMNS = ["ph", "conductivity", "turbidity", "orp"]
LABEL_COLUMN = "quality_label"


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw sensor data from a CSV file.

    Args:
        filepath: Path to the CSV file containing sensor readings.

    Returns:
        DataFrame with raw sensor readings.
    """
    df = pd.read_csv(filepath, parse_dates=["timestamp"] if "timestamp" in pd.read_csv(filepath, nrows=0).columns else [])
    print(f"[INFO] Loaded {len(df)} rows from {filepath}")
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """
    Validate that the DataFrame contains required sensor columns.

    Args:
        df: Input DataFrame.

    Returns:
        True if schema is valid, raises ValueError otherwise.
    """
    missing = [col for col in SENSOR_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing sensor columns: {missing}")
    print("[INFO] Schema validation passed.")
    return True


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using forward-fill then mean imputation.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with no missing values.
    """
    initial_nulls = df[SENSOR_COLUMNS].isnull().sum().sum()
    df[SENSOR_COLUMNS] = df[SENSOR_COLUMNS].fillna(method="ffill").fillna(df[SENSOR_COLUMNS].mean())
    print(f"[INFO] Handled {initial_nulls} missing values.")
    return df


def remove_outliers(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove rows where any sensor reading deviates beyond z_threshold standard deviations.

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
    print(f"[INFO] Removed {removed} outlier rows.")
    return df[mask].reset_index(drop=True)


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply min-max normalization to sensor columns.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with normalized sensor columns.
    """
    for col in SENSOR_COLUMNS:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max != col_min:
            df[col] = (df[col] - col_min) / (col_max - col_min)
    print("[INFO] Feature normalization complete.")
    return df


def preprocess_pipeline(filepath: str, remove_outliers_flag: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline: load -> validate -> clean -> normalize.

    Args:
        filepath: Path to raw sensor CSV.
        remove_outliers_flag: Whether to apply outlier removal.

    Returns:
        Cleaned and normalized DataFrame ready for feature engineering.
    """
    df = load_raw_data(filepath)
    validate_schema(df)
    df = handle_missing_values(df)
    if remove_outliers_flag:
        df = remove_outliers(df)
    df = normalize_features(df)
    print("[INFO] Preprocessing pipeline complete.")
    return df


if __name__ == "__main__":
    # Example usage — replace with your actual data path
    sample_path = RAW_DATA_DIR / "sample_sensor_data.csv"
    if sample_path.exists():
        processed_df = preprocess_pipeline(str(sample_path))
        out_path = PROCESSED_DATA_DIR / "processed_sensor_data.csv"
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(out_path, index=False)
        print(f"[INFO] Saved processed data to {out_path}")
    else:
        print(f"[WARN] No data found at {sample_path}. Add your sensor CSV files to data/raw/")
