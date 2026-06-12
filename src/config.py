"""Configuration dataclasses for the AayuSense project."""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Paths and column configuration for data artefacts."""

    raw_data_dir: str = 'data/raw'
    processed_data_dir: str = 'data/processed'
    models_dir: str = 'models'
    sensor_columns: List[str] = field(
        default_factory=lambda: ['ph', 'conductivity', 'turbidity', 'orp']
    )
    label_column: str = 'quality_label'


@dataclass
class ModelConfig:
    """Hyper-parameter search spaces and cross-validation settings."""

    # Random Forest grid
    rf_n_estimators_range: List[int] = field(default_factory=lambda: [100, 200, 300])
    rf_max_depth_range: List[Optional[int]] = field(default_factory=lambda: [None, 10, 20])
    rf_min_samples_split: List[int] = field(default_factory=lambda: [2, 5])
    # XGBoost grid
    xgb_n_estimators_range: List[int] = field(default_factory=lambda: [100, 200])
    xgb_max_depth_range: List[int] = field(default_factory=lambda: [3, 6, 9])
    xgb_lr_range: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    # Validation
    n_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class DashboardConfig:
    """Streamlit dashboard runtime settings."""

    refresh_interval_sec: int = 2
    history_window: int = 50
    alert_threshold: float = 0.7


def get_default_config() -> Tuple[DataConfig, ModelConfig, DashboardConfig]:
    """Return default DataConfig, ModelConfig, and DashboardConfig instances."""
    return DataConfig(), ModelConfig(), DashboardConfig()


def load_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Parsed dictionary, or empty dict on error.
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        logger.warning("pyyaml not installed; returning empty config dict.")
        return {}

    try:
        with open(yaml_path, 'r', encoding='utf-8') as fh:
            config = yaml.safe_load(fh)
        logger.info("Config loaded from '%s'.", yaml_path)
        return config or {}
    except FileNotFoundError:
        logger.error("Config file not found: '%s'.", yaml_path)
        return {}
    except yaml.YAMLError as exc:
        logger.error("YAML parse error in '%s': %s", yaml_path, exc)
        return {}
