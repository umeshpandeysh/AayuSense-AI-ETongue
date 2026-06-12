"""
sensor_simulator.py
AayuSense-AI-ETongue — Sensor Data Simulator

Generates synthetic sensor readings mimicking the AayuSense ESP32 hardware
for development, testing, and dashboard demonstration when real hardware
is not connected.

Herb profiles are based on expected electrochemical signatures for the
five herbs tested in the AayuSense project. Statistical parameters
(mean ± std) are illustrative and for development purposes only.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 8-sensor profiles per herb — (mean, std) for each sensor channel
# Based on AayuSense project herb library
HERB_PROFILES: Dict[str, Dict[str, tuple]] = {
    "Neem (Azadirachta indica)": {
        "pH": (6.1, 0.3),
        "TDS": (210, 25),
        "orp_mV": (345, 30),
        "turbidity": (0.14, 0.04),
        "Reduction_value": (0.12, 0.02),
        "Ionic_value": (0.08, 0.015),
        "Salt_content": (0.22, 0.03),
        "temp_c": (25.2, 0.5),
    },
    "Turmeric (Curcuma longa)": {
        "pH": (5.8, 0.4),
        "TDS": (180, 20),
        "orp_mV": (290, 35),
        "turbidity": (0.22, 0.05),
        "Reduction_value": (0.18, 0.03),
        "Ionic_value": (0.15, 0.02),
        "Salt_content": (0.19, 0.025),
        "temp_c": (24.8, 0.4),
    },
    "Ashwagandha (Withania somnifera)": {
        "pH": (6.5, 0.3),
        "TDS": (155, 18),
        "orp_mV": (320, 28),
        "turbidity": (0.08, 0.03),
        "Reduction_value": (0.095, 0.015),
        "Ionic_value": (0.065, 0.012),
        "Salt_content": (0.14, 0.02),
        "temp_c": (25.5, 0.5),
    },
    "Ginger (Zingiber officinale)": {
        "pH": (5.5, 0.5),
        "TDS": (230, 30),
        "orp_mV": (370, 40),
        "turbidity": (0.30, 0.06),
        "Reduction_value": (0.22, 0.04),
        "Ionic_value": (0.18, 0.03),
        "Salt_content": (0.28, 0.04),
        "temp_c": (25.0, 0.6),
    },
    "Tulsi (Ocimum tenuiflorum)": {
        "pH": (6.3, 0.35),
        "TDS": (165, 22),
        "orp_mV": (305, 32),
        "turbidity": (0.10, 0.035),
        "Reduction_value": (0.11, 0.018),
        "Ionic_value": (0.07, 0.013),
        "Salt_content": (0.16, 0.022),
        "temp_c": (25.3, 0.45),
    },
}

# Adulteration perturbation factors applied on top of genuine profiles
ADULTERATION_PERTURBATIONS = {
    "Adulterated_Starch": {
        "TDS": 1.6,
        "Reduction_value": 1.5,
        "pH": 0.92,
    },
    "Adulterated_Synthetic_Dye": {
        "orp_mV": 1.3,
        "turbidity": 1.8,
        "Salt_content": 1.4,
    },
    "Adulterated_Heavy_Metal": {
        "TDS": 2.1,
        "Ionic_value": 1.9,
        "Salt_content": 1.7,
        "orp_mV": 1.4,
    },
}

SENSOR_COLUMNS = [
    "pH", "TDS", "orp_mV", "turbidity",
    "Reduction_value", "Ionic_value", "Salt_content", "temp_c"
]


class SensorSimulator:
    """
    Simulates AayuSense ESP32 sensor readings for development and testing.

    Generates Gaussian-distributed readings centered on herb-specific
    electrochemical profiles, with optional adulteration perturbation.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.device_id = "ESP32-SIM"
        logger.info("SensorSimulator initialized (seed=%s)", seed)

    def generate_genuine_reading(self, herb_name: str) -> Dict:
        """
        Generate a genuine (unadulterated) sensor reading for a given herb.

        Args:
            herb_name: One of the herb keys in HERB_PROFILES.

        Returns:
            Dict matching the AayuSense ESP32 JSON payload format.
        """
        if herb_name not in HERB_PROFILES:
            raise ValueError(f"Unknown herb: {herb_name}. Available: {list(HERB_PROFILES.keys())}")
        profile = HERB_PROFILES[herb_name]
        sensors = {
            col: round(float(self.rng.normal(*profile[col])), 4)
            for col in SENSOR_COLUMNS
        }
        return self._build_payload(sensors, herb_name, "Genuine")

    def generate_adulterated_reading(self, herb_name: str, adulteration_type: str) -> Dict:
        """
        Generate an adulterated sensor reading.

        Args:
            herb_name: Base herb to use as starting profile.
            adulteration_type: One of the ADULTERATION_PERTURBATIONS keys.

        Returns:
            Dict with perturbed sensor readings.
        """
        if adulteration_type not in ADULTERATION_PERTURBATIONS:
            raise ValueError(f"Unknown adulteration type: {adulteration_type}")
        base = self.generate_genuine_reading(herb_name)
        sensors = dict(base["sensors"])
        perturbs = ADULTERATION_PERTURBATIONS[adulteration_type]
        for sensor, factor in perturbs.items():
            if sensor in sensors:
                sensors[sensor] = round(sensors[sensor] * factor, 4)
        return self._build_payload(sensors, herb_name, adulteration_type)

    def generate_dataset(
        self,
        n_per_herb: int = 100,
        adulteration_fraction: float = 0.3
    ) -> pd.DataFrame:
        """
        Generate a complete dataset for ML training.

        Args:
            n_per_herb: Number of readings per herb.
            adulteration_fraction: Fraction of samples that are adulterated.

        Returns:
            DataFrame with all sensor columns, herb_name, and quality_label.
        """
        records = []
        adulterants = list(ADULTERATION_PERTURBATIONS.keys())
        for herb_name in HERB_PROFILES:
            n_genuine = int(n_per_herb * (1 - adulteration_fraction))
            n_adulterated = n_per_herb - n_genuine
            for _ in range(n_genuine):
                payload = self.generate_genuine_reading(herb_name)
                row = {**payload["sensors"], "herb_name": herb_name, "quality_label": "Genuine",
                       "timestamp": payload["timestamp"]}
                records.append(row)
            for _ in range(n_adulterated):
                atype = self.rng.choice(adulterants)
                payload = self.generate_adulterated_reading(herb_name, atype)
                row = {**payload["sensors"], "herb_name": herb_name, "quality_label": atype,
                       "timestamp": payload["timestamp"]}
                records.append(row)
        df = pd.DataFrame(records)
        self.rng.shuffle(df.values)
        logger.info("Generated dataset: %d samples, %d herbs", len(df), len(HERB_PROFILES))
        return df.reset_index(drop=True)

    def _build_payload(self, sensors: Dict, herb_name: str, quality_label: str) -> Dict:
        return {
            "sample_id": f"SIM-{datetime.utcnow().strftime('%Y%m%d-%H%M%S%f')[:19]}",
            "device_id": self.device_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sensors": sensors,
            "herb_name": herb_name,
            "quality_label": quality_label,
        }

    def to_esp32_json(self, herb_name: str) -> str:
        """Generate a JSON string matching the real ESP32 firmware output format."""
        payload = self.generate_genuine_reading(herb_name)
        return json.dumps(payload, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sim = SensorSimulator(seed=42)
    # Demo: print one reading for each herb
    for herb in list(HERB_PROFILES.keys())[:2]:
        print(f"\n--- {herb} ---")
        print(sim.to_esp32_json(herb))
    # Generate dataset
    df = sim.generate_dataset(n_per_herb=50)
    print(f"\nDataset shape: {df.shape}")
    print(df.head(3).to_string())
