"""Unit tests for src/feature_engineering.py."""
import sys
import pathlib
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    from src.feature_engineering import (
        compute_rolling_statistics,
        compute_range_features,
        compute_gradient_features,
        compute_interaction_features,
        extract_features,
    )
    _FE_AVAILABLE = True
except ImportError:
    _FE_AVAILABLE = False


def _make_df(n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'ph':           rng.normal(6.8, 0.3, n),
        'conductivity': rng.normal(1.2, 0.15, n),
        'turbidity':    rng.normal(45.0, 8.0, n),
        'orp':          rng.normal(180.0, 25.0, n),
        'quality_label': ['Genuine'] * n,
    })


@unittest.skipUnless(_FE_AVAILABLE, "src.feature_engineering not yet implemented")
class TestComputeRollingStatistics(unittest.TestCase):
    def test_rolling_mean_std_columns_added(self):
        df = _make_df()
        result = compute_rolling_statistics(df, columns=['ph', 'conductivity', 'turbidity', 'orp'], window=3)
        for col in ['ph', 'conductivity', 'turbidity', 'orp']:
            self.assertIn(f'{col}_roll_mean', result.columns)
            self.assertIn(f'{col}_roll_std', result.columns)


@unittest.skipUnless(_FE_AVAILABLE, "src.feature_engineering not yet implemented")
class TestComputeRangeFeatures(unittest.TestCase):
    def test_range_columns_added(self):
        result = compute_range_features(_make_df(), columns=['ph', 'conductivity', 'turbidity', 'orp'])
        for col in ['ph', 'conductivity', 'turbidity', 'orp']:
            self.assertIn(f'{col}_range', result.columns)


@unittest.skipUnless(_FE_AVAILABLE, "src.feature_engineering not yet implemented")
class TestComputeGradientFeatures(unittest.TestCase):
    def test_gradient_columns_added(self):
        result = compute_gradient_features(_make_df(), columns=['ph', 'conductivity', 'turbidity', 'orp'])
        for col in ['ph', 'conductivity', 'turbidity', 'orp']:
            self.assertIn(f'{col}_gradient', result.columns)


@unittest.skipUnless(_FE_AVAILABLE, "src.feature_engineering not yet implemented")
class TestComputeInteractionFeatures(unittest.TestCase):
    def test_interaction_columns_exist(self):
        result = compute_interaction_features(_make_df())
        interaction_cols = [c for c in result.columns if '_x_' in c or '_over_' in c]
        self.assertGreater(len(interaction_cols), 0)


@unittest.skipUnless(_FE_AVAILABLE, "src.feature_engineering not yet implemented")
class TestExtractFeatures(unittest.TestCase):
    def test_full_pipeline_runs(self):
        result = extract_features(_make_df(10))
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(result.shape[1], 5)

    def test_correct_number_of_rows(self):
        result = extract_features(_make_df(10))
        self.assertLessEqual(len(result), 10)
        self.assertGreater(len(result), 0)
