"""pytest configuration and shared fixtures for AayuSense tests."""
from __future__ import annotations

import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

# Ensure the repo root is on sys.path so all src.* imports work
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


from src.feature_engineering import SENSOR_COLUMNS


@pytest.fixture()
def sensor_df_small() -> pd.DataFrame:
    """Return a normalised 8-sensor DataFrame with 15 rows."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        col: rng.uniform(0.1, 0.9, 15) for col in SENSOR_COLUMNS
    } | {'quality_label': ['Genuine'] * 15})


@pytest.fixture()
def sensor_df_large() -> pd.DataFrame:
    """Return a normalised 8-sensor DataFrame with 120 rows (4 classes)."""
    rng = np.random.default_rng(1)
    n_per_class = 30
    classes = ['Genuine', 'Adulterated_Starch', 'Adulterated_Synthetic_Dye', 'Adulterated_Heavy_Metal']
    frames = []
    for label in classes:
        df = pd.DataFrame({col: rng.uniform(0.1, 0.9, n_per_class) for col in SENSOR_COLUMNS})
        df['quality_label'] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
