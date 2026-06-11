"""
feature_engineering.py
AayuSense-AI-ETongue — Feature Engineering Module

Extracts and computes ML-relevant features from preprocessed sensor data.
"""

import pandas as pd
import numpy as np


SENSOR_COLUMNS = ["ph", "conductivity", "turbidity", "orp"]


def compute_rolling_statistics(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add rolling mean and std for each sensor column.

    Args:
        df: Preprocessed sensor DataFrame.
        window: Rolling window size.

    Returns:
        DataFrame with additional rolling statistics columns.
    """
    for col in SENSOR_COLUMNS:
        df[f"{col}_mean"] = df[col].rolling(window=window, min_periods=1).mean()
        df[f"{col}_std"] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
    return df


def compute_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute range (max - min) for each sensor within a rolling window.

    Args:
        df: Preprocessed sensor DataFrame.

    Returns:
        DataFrame with range features.
    """
    for col in SENSOR_COLUMNS:
        df[f"{col}_range"] = df[col].rolling(window=5, min_periods=1).max() - \
                              df[col].rolling(window=5, min_periods=1).min()
    return df


def compute_gradient_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the first-order gradient (rate of change) for each sensor.

    Args:
        df: Preprocessed sensor DataFrame.

    Returns:
        DataFrame with gradient features.
    """
    for col in SENSOR_COLUMNS:
        df[f"{col}_gradient"] = df[col].diff().fillna(0)
    return df


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute interaction features between sensor channels.

    Args:
        df: Preprocessed sensor DataFrame.

    Returns:
        DataFrame with cross-sensor interaction terms.
    """
    df["ph_conductivity_ratio"] = df["ph"] / (df["conductivity"] + 1e-6)
    df["orp_turbidity_product"] = df["orp"] * df["turbidity"]
    df["sensor_energy"] = (df[SENSOR_COLUMNS] ** 2).sum(axis=1)
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature extraction pipeline.

    Args:
        df: Preprocessed sensor DataFrame.

    Returns:
        Feature-rich DataFrame ready for ML training.
    """
    df = compute_rolling_statistics(df)
    df = compute_range_features(df)
    df = compute_gradient_features(df)
    df = compute_interaction_features(df)
    print(f"[INFO] Feature engineering complete. Total features: {len(df.columns)}")
    return df


if __name__ == "__main__":
    print("Feature engineering module loaded. Import extract_features() to use.")
