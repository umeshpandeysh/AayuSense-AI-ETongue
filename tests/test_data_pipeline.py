"""Unit tests for src/data_pipeline.py."""
import os
import sys
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    from src.data_pipeline import (
        load_raw_data,
        handle_missing_values,
        remove_outliers,
        normalize_features,
    )
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False


@unittest.skipUnless(_PIPELINE_AVAILABLE, "src.data_pipeline not yet implemented")
class TestLoadRawData(unittest.TestCase):
    """Tests for load_raw_data()."""

    def setUp(self):
        self.required_columns = ['ph', 'conductivity', 'turbidity', 'orp', 'quality_label']

    def test_load_valid_csv(self):
        """Loading a valid CSV returns a non-empty DataFrame with expected columns."""
        data = {
            'ph': [6.8, 5.2],
            'conductivity': [1.2, 2.8],
            'turbidity': [45, 120],
            'orp': [180, 95],
            'quality_label': ['Genuine', 'Adulterant_A'],
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame(data).to_csv(f, index=False)
            tmp = f.name
        try:
            df = load_raw_data(tmp)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            for col in self.required_columns:
                self.assertIn(col, df.columns)
        finally:
            os.unlink(tmp)

    def test_load_missing_columns_raises(self):
        """CSV missing required columns raises ValueError or KeyError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame({'ph': [6.8], 'turbidity': [45]}).to_csv(f, index=False)
            tmp = f.name
        try:
            with self.assertRaises((ValueError, KeyError)):
                load_raw_data(tmp)
        finally:
            os.unlink(tmp)


@unittest.skipUnless(_PIPELINE_AVAILABLE, "src.data_pipeline not yet implemented")
class TestHandleMissingValues(unittest.TestCase):
    def test_nan_values_filled(self):
        df = pd.DataFrame({
            'ph':           [6.8, np.nan, 7.0],
            'conductivity': [1.2, 1.5, np.nan],
            'turbidity':    [45.0, 50.0, 55.0],
            'orp':          [180.0, np.nan, 200.0],
            'quality_label': ['Genuine'] * 3,
        })
        df_clean = handle_missing_values(df)
        self.assertEqual(df_clean.isnull().sum().sum(), 0)


@unittest.skipUnless(_PIPELINE_AVAILABLE, "src.data_pipeline not yet implemented")
class TestRemoveOutliers(unittest.TestCase):
    def test_extreme_outlier_removed(self):
        normal = [6.5, 6.7, 6.8, 6.9, 7.0, 6.6, 6.75, 6.85] * 5
        df = pd.DataFrame({
            'ph':           normal + [999.0],
            'conductivity': [1.2] * 41,
            'turbidity':    [45.0] * 41,
            'orp':          [180.0] * 41,
            'quality_label': ['Genuine'] * 41,
        })
        df_clean = remove_outliers(df, columns=['ph'])
        self.assertTrue((df_clean['ph'] < 100).all())
        self.assertLess(len(df_clean), len(df))


@unittest.skipUnless(_PIPELINE_AVAILABLE, "src.data_pipeline not yet implemented")
class TestNormalizeFeatures(unittest.TestCase):
    def test_values_in_zero_one_range(self):
        df = pd.DataFrame({
            'ph':           [5.0, 6.0, 7.0, 8.0],
            'conductivity': [0.5, 1.0, 2.0, 3.0],
            'turbidity':    [20.0, 50.0, 80.0, 120.0],
            'orp':          [80.0, 150.0, 200.0, 320.0],
            'quality_label': ['Genuine'] * 4,
        })
        cols = ['ph', 'conductivity', 'turbidity', 'orp']
        df_norm = normalize_features(df, columns=cols)
        for col in cols:
            self.assertAlmostEqual(df_norm[col].min(), 0.0, places=5)
            self.assertAlmostEqual(df_norm[col].max(), 1.0, places=5)
