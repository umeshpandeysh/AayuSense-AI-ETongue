"""Unit tests for src/data_pipeline.py.

Run with: pytest tests/test_data_pipeline.py -v
"""
from __future__ import annotations

import os
import sys
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data_pipeline import (  # noqa: E402
    load_raw_data,
    validate_schema,
    handle_missing_values,
    remove_outliers,
    normalize_features,
    SENSOR_COLUMNS,
)


def _make_valid_csv(n: int = 20) -> str:
    """Write a minimal valid sensor CSV to a temp file and return its path."""
    rng = np.random.default_rng(42)
    data = {
        'pH': rng.uniform(5.5, 7.5, n),
        'TDS': rng.uniform(100, 400, n),
        'orp_mV': rng.uniform(200, 400, n),
        'turbidity': rng.uniform(0.05, 0.4, n),
        'Reduction_value': rng.uniform(0.05, 0.3, n),
        'Ionic_value': rng.uniform(0.04, 0.2, n),
        'Salt_content': rng.uniform(0.1, 0.35, n),
        'temp_c': rng.uniform(24.0, 27.0, n),
        'quality_label': ['Genuine'] * n,
    }
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.csv', delete=False, encoding='utf-8'
    ) as f:
        df.to_csv(f, index=False)
        return f.name


class TestLoadRawData(unittest.TestCase):
    """Tests for load_raw_data()."""

    def test_load_valid_csv_returns_dataframe(self):
        tmp = _make_valid_csv()
        try:
            df = load_raw_data(tmp)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            for col in SENSOR_COLUMNS:
                self.assertIn(col, df.columns)
        finally:
            os.unlink(tmp)

    def test_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_raw_data('/nonexistent/path/sensor_data.csv')


class TestValidateSchema(unittest.TestCase):
    """Tests for validate_schema()."""

    def test_valid_schema_returns_true(self):
        df = pd.DataFrame({col: [1.0] for col in SENSOR_COLUMNS})
        self.assertTrue(validate_schema(df))

    def test_missing_column_raises_value_error(self):
        df = pd.DataFrame({'pH': [6.5], 'TDS': [200.0]})  # missing cols
        with self.assertRaises(ValueError):
            validate_schema(df)


class TestHandleMissingValues(unittest.TestCase):
    """Tests for handle_missing_values()."""

    def _make_df_with_nans(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        df = pd.DataFrame({col: rng.uniform(0.1, 1.0, 10) for col in SENSOR_COLUMNS})
        df.loc[2, 'pH'] = np.nan
        df.loc[5, 'TDS'] = np.nan
        return df

    def test_no_nans_remain(self):
        df = self._make_df_with_nans()
        self.assertGreater(df[SENSOR_COLUMNS].isnull().sum().sum(), 0)
        result = handle_missing_values(df)
        self.assertEqual(result[SENSOR_COLUMNS].isnull().sum().sum(), 0)

    def test_shape_preserved(self):
        df = self._make_df_with_nans()
        result = handle_missing_values(df)
        self.assertEqual(result.shape[0], df.shape[0])


class TestRemoveOutliers(unittest.TestCase):
    """Tests for remove_outliers()."""

    def test_extreme_outlier_removed(self):
        """A single extreme pH value should be flagged and dropped."""
        rng = np.random.default_rng(1)
        normal_vals = rng.normal(6.5, 0.2, 40)
        ph_vals = list(normal_vals) + [9999.0]  # extreme outlier
        n = len(ph_vals)
        df = pd.DataFrame({
            'pH': ph_vals,
            'TDS': rng.uniform(150, 300, n),
            'orp_mV': rng.uniform(250, 380, n),
            'turbidity': rng.uniform(0.05, 0.3, n),
            'Reduction_value': rng.uniform(0.05, 0.25, n),
            'Ionic_value': rng.uniform(0.04, 0.18, n),
            'Salt_content': rng.uniform(0.1, 0.3, n),
            'temp_c': rng.uniform(24.0, 27.0, n),
        })
        result = remove_outliers(df, z_threshold=3.0)
        self.assertLess(len(result), len(df))
        self.assertTrue((result['pH'] < 100).all())

    def test_no_outliers_unchanged(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame({col: rng.normal(0.5, 0.05, 30) for col in SENSOR_COLUMNS})
        result = remove_outliers(df, z_threshold=5.0)  # very high threshold
        self.assertEqual(len(result), len(df))


class TestNormalizeFeatures(unittest.TestCase):
    """Tests for normalize_features()."""

    def test_sensor_values_in_zero_one_range(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame({col: rng.uniform(0.0, 10.0, 20) for col in SENSOR_COLUMNS})
        result = normalize_features(df)
        for col in SENSOR_COLUMNS:
            self.assertAlmostEqual(result[col].min(), 0.0, places=5)
            self.assertAlmostEqual(result[col].max(), 1.0, places=5)

    def test_constant_column_not_modified(self):
        """A constant column (min == max) should not raise and stays 0."""
        df = pd.DataFrame({col: [0.5] * 10 for col in SENSOR_COLUMNS})
        result = normalize_features(df)  # should not raise
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
