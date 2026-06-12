"""Unit tests for src/esp32_interface.py.

Run with: pytest tests/test_esp32_interface.py -v
"""
from __future__ import annotations

import json
import sys
import pathlib
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.esp32_interface import ESP32DataParser, SensorReading, SENSOR_RANGES  # noqa: E402


def _valid_payload(**overrides) -> str:
    """Return a valid ESP32 JSON payload string."""
    sensors = {
        'pH': 6.1, 'TDS': 210.0, 'orp_mV': 345.0,
        'turbidity': 0.14, 'Reduction_value': 0.12,
        'Ionic_value': 0.08, 'Salt_content': 0.22,
        'temp_c': 25.2,
    }
    sensors.update(overrides)
    payload = {
        'sample_id': 'TEST-001',
        'device_id': 'ESP32-01',
        'timestamp': '2025-09-16T09:12:00Z',
        'sensors': sensors,
        'herb_name': 'Neem (Azadirachta indica)',
        'operator': 'Umesh',
    }
    return json.dumps(payload)


class TestESP32DataParser(unittest.TestCase):

    def setUp(self):
        self.parser = ESP32DataParser(strict=True)

    def test_parse_valid_payload_returns_sensor_reading(self):
        reading = self.parser.parse(_valid_payload())
        self.assertIsInstance(reading, SensorReading)
        self.assertEqual(reading.device_id, 'ESP32-01')
        self.assertAlmostEqual(reading.pH, 6.1)

    def test_sensor_vector_returns_eight_fields(self):
        reading = self.parser.parse(_valid_payload())
        vec = reading.sensor_vector()
        self.assertEqual(len(vec), 8)
        self.assertIn('pH', vec)
        self.assertIn('Salt_content', vec)

    def test_missing_sensor_field_raises_value_error(self):
        payload = {
            'sample_id': 'BAD-001',
            'device_id': 'ESP32-01',
            'timestamp': '2025-09-16T09:12:00Z',
            'sensors': {'pH': 6.1},  # missing TDS, orp_mV, etc.
        }
        with self.assertRaises(ValueError):
            self.parser.parse(json.dumps(payload))

    def test_out_of_range_strict_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.parser.parse(_valid_payload(pH=999.0))  # pH > 14

    def test_non_strict_does_not_raise_for_out_of_range(self):
        lenient_parser = ESP32DataParser(strict=False)
        reading = lenient_parser.parse(_valid_payload(pH=999.0))
        self.assertAlmostEqual(reading.pH, 999.0)

    def test_parse_batch_skips_bad_payloads(self):
        good = _valid_payload()
        bad = 'not-valid-json'
        results = self.parser.parse_batch([good, bad, good])
        self.assertEqual(len(results), 2)

    def test_to_dict_has_all_fields(self):
        reading = self.parser.parse(_valid_payload())
        d = reading.to_dict()
        self.assertIn('pH', d)
        self.assertIn('herb_name', d)
        self.assertIn('sample_id', d)


class TestSensorRanges(unittest.TestCase):
    def test_eight_sensors_defined(self):
        self.assertEqual(len(SENSOR_RANGES), 8)

    def test_all_ranges_are_tuples(self):
        for k, v in SENSOR_RANGES.items():
            self.assertIsInstance(v, tuple)
            self.assertEqual(len(v), 2)

    def test_ph_range_zero_to_fourteen(self):
        lo, hi = SENSOR_RANGES['pH']
        self.assertEqual(lo, 0.0)
        self.assertEqual(hi, 14.0)


if __name__ == '__main__':
    unittest.main()
