"""
esp32_interface.py
AayuSense-AI-ETongue — ESP32 Hardware Interface

Parses and validates real-time sensor payloads from the ESP32 device.
The AayuSense ESP32 firmware sends JSON over serial/MQTT with all 8 sensor readings.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Sensor valid ranges from AayuSense hardware calibration
SENSOR_RANGES: Dict[str, tuple] = {
    "pH":              (0.0,  14.0),
    "TDS":             (0.0,  1000.0),
    "orp_mV":          (-500.0, 500.0),
    "turbidity":       (0.0,  1.0),
    "Reduction_value": (0.0,  1.0),
    "Ionic_value":     (0.0,  1.0),
    "Salt_content":    (0.0,  1.0),
    "temp_c":          (15.0, 45.0),
}

@dataclass
class SensorReading:
    """Structured representation of one ESP32 sensor reading."""
    sample_id: str
    device_id: str
    timestamp: str
    pH: float
    TDS: float
    orp_mV: float
    turbidity: float
    Reduction_value: float
    Ionic_value: float
    Salt_content: float
    temp_c: float
    herb_name:       Optional[str] = None
    operator:        Optional[str] = None
    notes:           Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def sensor_vector(self) -> Dict[str, float]:
        """Return only numeric sensor values as dict."""
        return {
            "pH": self.pH,
            "TDS": self.TDS,
            "orp_mV": self.orp_mV,
            "turbidity": self.turbidity,
            "Reduction_value": self.Reduction_value,
            "Ionic_value": self.Ionic_value,
            "Salt_content": self.Salt_content,
            "temp_c": self.temp_c,
        }


class ESP32DataParser:
    """
    Parses JSON payloads sent by the AayuSense ESP32 firmware.

    The firmware sends payloads in this format:
    {
        "sample_id": "S20250916-01",
        "device_id": "ESP32-01",
        "timestamp": "2025-09-16T09:12:00Z",
        "sensors": {
            "pH": 6.1, "TDS": 210, "orp_mV": 345,
            "turbidity": 0.14, "Reduction_value": 0.120,
            "Ionic_value": 0.080, "Salt_content": 0.220,
            "temp_c": 25.2
        },
        "herb_name": "Neem (Azadirachta indica)",
        "operator": "Umesh"
    }
    """

    REQUIRED_FIELDS = list(SENSOR_RANGES.keys())

    def __init__(self, strict: bool = True):
        """
        Args:
            strict: If True, raises on out-of-range values. If False, logs warning.
        """
        self.strict = strict
        logger.info("ESP32DataParser initialized (strict=%s)", strict)

    def parse(self, payload: str) -> SensorReading:
        """
        Parse a raw JSON string from the ESP32 into a SensorReading.

        Args:
            payload: JSON string from ESP32 firmware.

        Returns:
            Validated SensorReading dataclass instance.

        Raises:
            ValueError: If required fields are missing or values are out of range (strict mode).
            json.JSONDecodeError: If payload is not valid JSON.
        """
        data = json.loads(payload)
        sensors = data.get("sensors", {})

        missing = [f for f in self.REQUIRED_FIELDS if f not in sensors]
        if missing:
            raise ValueError(f"Missing sensor fields in payload: {missing}")

        reading = SensorReading(
            sample_id=data.get("sample_id", f"S{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
            device_id=data.get("device_id", "ESP32-UNKNOWN"),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            pH=float(sensors["pH"]),
            TDS=float(sensors["TDS"]),
            orp_mV=float(sensors["orp_mV"]),
            turbidity=float(sensors["turbidity"]),
            Reduction_value=float(sensors["Reduction_value"]),
            Ionic_value=float(sensors["Ionic_value"]),
            Salt_content=float(sensors["Salt_content"]),
            temp_c=float(sensors["temp_c"]),
            herb_name=data.get("herb_name"),
            operator=data.get("operator"),
            notes=data.get("notes"),
        )
        self._validate(reading)
        logger.debug("Parsed reading: %s from device %s", reading.sample_id, reading.device_id)
        return reading

    def _validate(self, reading: SensorReading) -> None:
        """Validate sensor readings are within calibrated hardware ranges."""
        vector = reading.sensor_vector()
        for sensor, (lo, hi) in SENSOR_RANGES.items():
            val = vector[sensor]
            if not (lo <= val <= hi):
                msg = f"Sensor {sensor}={val} out of range [{lo}, {hi}]"
                if self.strict:
                    raise ValueError(msg)
                logger.warning(msg)

    def parse_batch(self, payloads: list[str]) -> list[SensorReading]:
        """
        Parse a list of JSON payloads, skipping malformed entries.

        Args:
            payloads: List of JSON strings.

        Returns:
            List of successfully parsed SensorReading objects.
        """
        results = []
        for i, p in enumerate(payloads):
            try:
                results.append(self.parse(p))
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning("Skipping payload %d: %s", i, exc)
        logger.info("Parsed %d/%d payloads successfully", len(results), len(payloads))
        return results


if __name__ == "__main__":
    # Demo
    sample_payload = json.dumps({
        "sample_id": "S20250916-01",
        "device_id": "ESP32-01",
        "timestamp": "2025-09-16T09:12:00Z",
        "sensors": {
            "pH": 6.1, "TDS": 210, "orp_mV": 345,
            "turbidity": 0.14, "Reduction_value": 0.120,
            "Ionic_value": 0.080, "Salt_content": 0.220,
            "temp_c": 25.2
        },
        "herb_name": "Neem (Azadirachta indica)",
        "operator": "Umesh"
    })
    parser = ESP32DataParser()
    reading = parser.parse(sample_payload)
    print("Sample ID:", reading.sample_id)
    print("Sensor vector:", reading.sensor_vector())
