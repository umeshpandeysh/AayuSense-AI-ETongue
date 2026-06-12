"""Unit tests for src/feature_engineering.py.

Run with: pytest tests/test_feature_engineering.py -v
"""
from __future__ import annotations

import sys
import pathlib
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.feature_engineering import (  # noqa: E402
    compute_rolling_statistics,
    compute_range_features,
    compute_gradient_features,
    compute_interaction_features,
    compute_rasa_intensities,
    compute_adulteration_score,
    extract_features,
    SENSOR_COLUMNS,
)


def _make_sensor_df(n: int = 15) -> pd.DataFrame:
    """Return a minimal 8-sensor DataFrame for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'pH': rng.uniform(5.5, 7.5, n),
        'TDS': rng.uniform(0.2, 0.8, n),      # normalised [0,1]
        'orp_mV': rng.uniform(0.3, 0.9, n),
        'turbidity': rng.uniform(0.05, 0.4, n),
        'Reduction_value': rng.uniform(0.05, 0.3, n),
        'Ionic_value': rng.uniform(0.04, 0.2, n),
        'Salt_content': rng.uniform(0.1, 0.35, n),
        'temp_c': rng.uniform(0.4, 0.7, n),      # normalised
        'quality_label': ['Genuine'] * n,
    })


class TestComputeRollingStatistics(unittest.TestCase):
    def test_mean_std_columns_added(self):
        df = _make_sensor_df()
        result = compute_rolling_statistics(df, window=3)
        for col in SENSOR_COLUMNS:
            self.assertIn(f'{col}_mean', result.columns, f'Missing {col}_mean')
            self.assertIn(f'{col}_std', result.columns, f'Missing {col}_std')

    def test_no_rows_dropped(self):
        df = _make_sensor_df(10)
        result = compute_rolling_statistics(df)
        self.assertEqual(len(result), 10)

    def test_mean_values_finite(self):
        df = _make_sensor_df()
        result = compute_rolling_statistics(df)
        self.assertTrue(result['pH_mean'].notna().all())


class TestComputeRangeFeatures(unittest.TestCase):
    def test_range_columns_added(self):
        df = _make_sensor_df()
        result = compute_range_features(df)
        for col in SENSOR_COLUMNS:
            self.assertIn(f'{col}_range', result.columns)

    def test_range_nonnegative(self):
        df = _make_sensor_df()
        result = compute_range_features(df)
        for col in SENSOR_COLUMNS:
            self.assertTrue((result[f'{col}_range'] >= 0).all())


class TestComputeGradientFeatures(unittest.TestCase):
    def test_gradient_columns_added(self):
        df = _make_sensor_df()
        result = compute_gradient_features(df)
        for col in SENSOR_COLUMNS:
            self.assertIn(f'{col}_gradient', result.columns)

    def test_first_gradient_is_zero(self):
        df = _make_sensor_df()
        result = compute_gradient_features(df)
        # diff().fillna(0) makes first row 0
        self.assertEqual(result['pH_gradient'].iloc[0], 0.0)


class TestComputeInteractionFeatures(unittest.TestCase):
    def test_interaction_columns_exist(self):
        df = _make_sensor_df()
        result = compute_interaction_features(df)
        expected = [
            'pH_TDS_product',
            'orp_reduction_composite',
            'electrolyte_index',
            'pH_conductivity_ratio',
            'sensor_energy',
        ]
        for col in expected:
            self.assertIn(col, result.columns, f'Missing interaction feature: {col}')

    def test_sensor_energy_positive(self):
        df = _make_sensor_df()
        result = compute_interaction_features(df)
        self.assertTrue((result['sensor_energy'] >= 0).all())


class TestComputeRasaIntensities(unittest.TestCase):
    def test_six_rasa_columns_added(self):
        df = _make_sensor_df()
        result = compute_rasa_intensities(df)
        expected = [
            'rasa_madhura', 'rasa_amla', 'rasa_lavana',
            'rasa_tikta', 'rasa_katu', 'rasa_kashaya',
        ]
        for col in expected:
            self.assertIn(col, result.columns, f'Missing Rasa column: {col}')

    def test_rasa_values_in_zero_one(self):
        df = _make_sensor_df()
        result = compute_rasa_intensities(df)
        for col in ['rasa_madhura', 'rasa_tikta', 'rasa_kashaya']:
            self.assertTrue((result[col] >= 0).all())
            self.assertTrue((result[col] <= 1).all())


class TestComputeAdulterationScore(unittest.TestCase):
    def test_score_column_exists(self):
        df = _make_sensor_df()
        result = compute_adulteration_score(df)
        self.assertIn('adulteration_heuristic', result.columns)

    def test_score_bounded(self):
        df = _make_sensor_df()
        result = compute_adulteration_score(df)
        self.assertTrue((result['adulteration_heuristic'] >= 0).all())
        self.assertTrue((result['adulteration_heuristic'] <= 1).all())


class TestExtractFeatures(unittest.TestCase):
    def test_full_pipeline_returns_dataframe(self):
        result = extract_features(_make_sensor_df(15))
        self.assertIsInstance(result, pd.DataFrame)

    def test_feature_count_increases(self):
        df = _make_sensor_df(15)
        result = extract_features(df, include_rasa=True)
        # Should have significantly more columns than original 8 sensors
        self.assertGreater(result.shape[1], 20)

    def test_no_rows_lost(self):
        df = _make_sensor_df(12)
        result = extract_features(df)
        self.assertEqual(len(result), 12)

    def test_rasa_excluded_when_flag_false(self):
        result_with = extract_features(_make_sensor_df(10), include_rasa=True)
        result_without = extract_features(_make_sensor_df(10), include_rasa=False)
        self.assertGreater(result_with.shape[1], result_without.shape[1])


if __name__ == '__main__':
    unittest.main()
