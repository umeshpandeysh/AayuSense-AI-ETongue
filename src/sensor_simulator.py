"""Sensor Simulator for AayuSense E-Tongue project."""
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CLASS_PARAMS = {
    'Genuine':      {'ph': (6.8, 0.3),  'conductivity': (1.2, 0.15), 'turbidity': (45, 8),   'orp': (180, 25)},
    'Adulterant_A': {'ph': (5.2, 0.4),  'conductivity': (2.8, 0.3),  'turbidity': (120, 20), 'orp': (95, 30)},
    'Adulterant_B': {'ph': (7.9, 0.5),  'conductivity': (0.4, 0.1),  'turbidity': (25, 10),  'orp': (310, 40)},
    'Adulterant_C': {'ph': (6.1, 0.6),  'conductivity': (1.9, 0.25), 'turbidity': (85, 15),  'orp': (140, 35)},
}


class SensorSimulator:
    """Simulates E-Tongue sensor readings for herbal sample quality classes.

    Args:
        random_state: Seed for reproducibility.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        logger.info("SensorSimulator initialised with random_state=%d", random_state)

    def generate_sample(self, quality_label: str, noise_factor: float = 1.0) -> dict:
        """Generate a single sensor reading for the given quality class.

        Args:
            quality_label: One of 'Genuine', 'Adulterant_A', 'Adulterant_B', 'Adulterant_C'.
            noise_factor: Multiplier for Gaussian noise standard deviation.

        Returns:
            Dict with keys: timestamp, ph, conductivity, turbidity, orp, quality_label.

        Raises:
            ValueError: If quality_label is not recognised.
        """
        if quality_label not in CLASS_PARAMS:
            raise ValueError(
                f"Unknown quality label: '{quality_label}'. "
                f"Choose from {list(CLASS_PARAMS.keys())}"
            )
        params = CLASS_PARAMS[quality_label]
        sample = {
            'timestamp': datetime.utcnow().isoformat(timespec='seconds'),
            'ph': round(float(self.rng.normal(params['ph'][0], params['ph'][1] * noise_factor)), 4),
            'conductivity': round(
                float(self.rng.normal(params['conductivity'][0], params['conductivity'][1] * noise_factor)), 4
            ),
            'turbidity': round(
                float(self.rng.normal(params['turbidity'][0], params['turbidity'][1] * noise_factor)), 4
            ),
            'orp': round(float(self.rng.normal(params['orp'][0], params['orp'][1] * noise_factor)), 4),
            'quality_label': quality_label,
        }
        sample['ph'] = float(np.clip(sample['ph'], 0.0, 14.0))
        sample['conductivity'] = max(0.0, sample['conductivity'])
        sample['turbidity'] = max(0.0, sample['turbidity'])
        logger.debug("Generated sample: %s", sample)
        return sample

    def generate_dataset(
        self,
        n_samples_per_class: int = 125,
        noise_factor: float = 1.0,
        start_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Generate a balanced dataset with samples from all quality classes.

        Args:
            n_samples_per_class: Number of samples per quality class.
            noise_factor: Noise multiplier applied to each reading.
            start_time: Starting timestamp (default: 2025-01-15 00:00:00 UTC).

        Returns:
            Shuffled DataFrame with sensor columns and quality_label.
        """
        if start_time is None:
            start_time = datetime(2025, 1, 15, 0, 0, 0)

        records = []
        current_time = start_time
        interval = timedelta(minutes=10)

        for label in CLASS_PARAMS:
            for _ in range(n_samples_per_class):
                sample = self.generate_sample(label, noise_factor)
                sample['timestamp'] = current_time.isoformat(timespec='seconds')
                current_time += interval
                records.append(sample)

        df = pd.DataFrame(records)
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        logger.info(
            "Dataset generated: %d rows, class distribution=%s",
            len(df),
            df['quality_label'].value_counts().to_dict(),
        )
        return df

    @staticmethod
    def save_to_csv(df: pd.DataFrame, filepath: str) -> None:
        """Persist a DataFrame to CSV.

        Args:
            df: DataFrame to save.
            filepath: Destination path (parent directories created automatically).
        """
        import pathlib
        pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info("Dataset saved to '%s' (%d rows).", filepath, len(df))
