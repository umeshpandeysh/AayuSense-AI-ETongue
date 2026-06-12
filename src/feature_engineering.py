"""
feature_engineering.py
AayuSense-AI-ETongue — Feature Engineering

Extracts ML-relevant features from preprocessed 8-sensor data.
Includes Rasa intensity mapping — an Ayurvedic taste classification
approach that maps electrochemical sensor signatures to the six
taste categories (Rasas) used in traditional herbal assessment.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SENSOR_COLUMNS = [
    "pH", "TDS", "orp_mV", "turbidity",
    "Reduction_value", "Ionic_value", "Salt_content", "temp_c"
]

# Rasa (Ayurvedic taste) intensity mapping weights
# These weights reflect empirical observations from the AayuSense project:
# different Rasas correlate with specific electrochemical signatures.
# Madhura (sweet) -> low acidity, moderate TDS, low ORP
# Amla (sour) -> low pH (acidic), high ORP
# Lavana (salty) -> high TDS, high salt content, high ionic
# Tikta (bitter) -> high ORP, low turbidity, high reduction
# Katu (pungent) -> high orp, high reduction, elevated temp
# Kashaya (astringent) -> high ionic, moderate turbidity, high reduction
RASA_WEIGHTS = {
    "Madhura":  {"pH": +0.35, "TDS": +0.20, "orp_mV": -0.25, "Salt_content": +0.10, "turbidity": +0.10},
    "Amla":     {"pH": -0.50, "orp_mV": +0.30, "Reduction_value": +0.20},
    "Lavana":   {"TDS": +0.40, "Salt_content": +0.35, "Ionic_value": +0.25},
    "Tikta":    {"orp_mV": +0.35, "Reduction_value": +0.30, "turbidity": -0.20, "temp_c": +0.15},
    "Katu":     {"orp_mV": +0.30, "Reduction_value": +0.25, "temp_c": +0.30, "pH": -0.15},
    "Kashaya":  {"Ionic_value": +0.40, "turbidity": +0.25, "Reduction_value": +0.25, "TDS": +0.10},
}


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
    logger.debug("Rolling statistics computed (window=%d)", window)
    return df


def compute_range_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Compute range (max - min) within a rolling window for each sensor.

    Args:
        df: Preprocessed sensor DataFrame.
        window: Rolling window size.

    Returns:
        DataFrame with range features.
    """
    for col in SENSOR_COLUMNS:
        df[f"{col}_range"] = (
            df[col].rolling(window=window, min_periods=1).max()
            - df[col].rolling(window=window, min_periods=1).min()
        )
    return df


def compute_gradient_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute first-order gradient (rate of change) for each sensor.

    Args:
        df: Preprocessed sensor DataFrame.

    Returns:
        DataFrame with gradient (delta) features.
    """
    for col in SENSOR_COLUMNS:
        df[f"{col}_gradient"] = df[col].diff().fillna(0)
    return df


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-sensor interaction terms relevant to herbal assessment.

    Key interactions:
    - pH × TDS: Reflects combined acidity and dissolved solids
    - ORP × Reduction_value: Redox activity composite
    - Ionic_value × Salt_content: Electrolyte loading index
    - Sensor energy: L2 norm of all 8 sensors

    Args:
        df: Preprocessed sensor DataFrame.

    Returns:
        DataFrame with interaction features.
    """
    df["pH_TDS_product"]          = df["pH"] * df["TDS"]
    df["orp_reduction_composite"] = df["orp_mV"] * df["Reduction_value"]
    df["electrolyte_index"]       = df["Ionic_value"] * df["Salt_content"]
    df["pH_conductivity_ratio"]   = df["pH"] / (df["TDS"] + 1e-6)
    df["sensor_energy"]           = (df[SENSOR_COLUMNS] ** 2).sum(axis=1)
    return df


def compute_rasa_intensities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Ayurvedic Rasa (taste) intensity scores from sensor readings.

    Maps electrochemical sensor signatures to the six Rasa categories
    used in traditional herbal quality assessment. This is an experimental
    mapping based on the AayuSense project's domain-knowledge approach.

    The six Rasas:
    - Madhura (Sweet):      Associated with low acidity, moderate TDS
    - Amla (Sour):          Associated with low pH, high ORP
    - Lavana (Salty):       Associated with high TDS, Salt_content, Ionic_value
    - Tikta (Bitter):       Associated with high ORP, high Reduction_value
    - Katu (Pungent):       Associated with high ORP, temperature, reduction
    - Kashaya (Astringent): Associated with high Ionic_value, Reduction_value

    Args:
        df: Preprocessed (normalized) sensor DataFrame.

    Returns:
        DataFrame with six Rasa intensity score columns added.
    """
    for rasa, weights in RASA_WEIGHTS.items():
        score = pd.Series(np.zeros(len(df)), index=df.index)
        for sensor, weight in weights.items():
            if sensor in df.columns:
                score += weight * df[sensor]
        # Clip to [0, 1] and scale relative to max weight sum
        max_weight = sum(abs(w) for w in weights.values())
        df[f"rasa_{rasa.lower()}"] = (score / max_weight).clip(0, 1)
    logger.debug("Rasa intensity profiles computed for %d samples", len(df))
    return df


def compute_adulteration_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a simple adulteration likelihood score from sensor deviation.

    Uses the combined deviation of Salt_content, TDS, and Reduction_value
    from typical genuine herb baseline ranges. This is a heuristic score
    (not a validated model output) useful as an additional feature.

    Args:
        df: Preprocessed sensor DataFrame.

    Returns:
        DataFrame with 'adulteration_heuristic' column added.
    """
    # Weights reflect AayuSense SHAP analysis: Salt_content and TDS most impactful
    df["adulteration_heuristic"] = (
        0.40 * df["Salt_content"]
        + 0.30 * df["TDS"]
        + 0.20 * df["Reduction_value"]
        + 0.10 * (1 - df["pH"] / 14)
    ).clip(0, 1)
    return df


def extract_features(df: pd.DataFrame, include_rasa: bool = True) -> pd.DataFrame:
    """
    Full feature extraction pipeline.

    Args:
        df: Preprocessed sensor DataFrame (8 sensor columns).
        include_rasa: Whether to include Rasa intensity features.

    Returns:
        Feature-rich DataFrame ready for ML training.
    """
    df = compute_rolling_statistics(df)
    df = compute_range_features(df)
    df = compute_gradient_features(df)
    df = compute_interaction_features(df)
    df = compute_adulteration_score(df)
    if include_rasa:
        df = compute_rasa_intensities(df)
    total_features = len([c for c in df.columns if c not in ["quality_label", "timestamp", "sample_id", "herb_name"]])
    logger.info("Feature engineering complete. Total features: %d", total_features)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Feature engineering module. Import extract_features() to use.")
    print("Rasa profiles available:", list(RASA_WEIGHTS.keys()))
