"""Factory function for AayuSense model instantiation."""
from typing import Union

from src.config import ModelConfig
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel


def get_model(
    model_type: str,
    config: ModelConfig,
) -> Union[RandomForestModel, XGBoostModel]:
    """Return an instantiated (unfitted) model by type name.

    Args:
        model_type: ``'random_forest'`` / ``'rf'`` or ``'xgboost'`` / ``'xgb'``
                    (case-insensitive).
        config: ModelConfig passed to the model constructor.

    Returns:
        Untrained :class:`RandomForestModel` or :class:`XGBoostModel`.

    Raises:
        ValueError: If *model_type* is not recognised.

    Example:
        >>> model = get_model('rf', ModelConfig())
    """
    registry = {
        'random_forest': RandomForestModel,
        'rf': RandomForestModel,
        'xgboost': XGBoostModel,
        'xgb': XGBoostModel,
    }
    key = model_type.strip().lower()
    if key not in registry:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {sorted(set(registry.keys()))}"
        )
    return registry[key](config)
