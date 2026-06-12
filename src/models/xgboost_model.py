"""XGBoost classifier wrapper for AayuSense."""
import logging
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    XGBClassifier = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class XGBoostModel:
    """Encapsulates an XGBoost classifier with grid-search training.

    Args:
        config: ModelConfig instance with xgb_n_estimators_range, xgb_max_depth_range,
                xgb_lr_range, n_folds, random_state attributes.

    Note:
        If xgboost is not installed, a warning is logged and .train() will raise ImportError.
    """

    def __init__(self, config: Any) -> None:
        if not _XGB_AVAILABLE:
            logger.warning("xgboost not installed. Run: pip install xgboost")
        self.config = config
        self._model: Optional[GridSearchCV] = None
        self._is_fitted: bool = False
        logger.info("XGBoostModel initialised (xgboost_available=%s).", _XGB_AVAILABLE)

    @property
    def is_fitted_(self) -> bool:
        """True if the model has been trained."""
        return self._is_fitted

    @property
    def best_params_(self) -> Dict[str, Any]:
        """Best hyper-parameters found by GridSearchCV."""
        self._check_fitted()
        return self._model.best_params_  # type: ignore[union-attr]

    @property
    def cv_score_(self) -> float:
        """Best mean cross-validation score."""
        self._check_fitted()
        return float(self._model.best_score_)  # type: ignore[union-attr]

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'XGBoostModel':
        """Fit the model using GridSearchCV.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels (string or int encoded).

        Returns:
            self (enables method chaining).

        Raises:
            ImportError: If xgboost is not installed.
        """
        if not _XGB_AVAILABLE:
            raise ImportError("Install xgboost: pip install xgboost")

        param_grid = {
            'n_estimators': self.config.xgb_n_estimators_range,
            'max_depth': self.config.xgb_max_depth_range,
            'learning_rate': self.config.xgb_lr_range,
        }
        logger.info("GridSearchCV for XGBoost — param_grid: %s", param_grid)
        base = XGBClassifier(
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=self.config.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self._model = GridSearchCV(
            estimator=base,
            param_grid=param_grid,
            cv=self.config.n_folds,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        self._model.fit(X_train, y_train)
        self._is_fitted = True
        logger.info("Training done — best_params=%s  cv_score=%.4f", self.best_params_, self.cv_score_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        self._check_fitted()
        return self._model.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates."""
        self._check_fitted()
        return self._model.predict_proba(X)  # type: ignore[union-attr]

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Return feature importances sorted descending.

        Args:
            feature_names: Names matching training columns.
        """
        self._check_fitted()
        imps = self._model.best_estimator_.feature_importances_  # type: ignore[union-attr]
        return dict(sorted(zip(feature_names, imps), key=lambda kv: kv[1], reverse=True))

    def save(self, filepath: str) -> None:
        """Serialise the fitted model using joblib."""
        self._check_fitted()
        import pathlib
        pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("XGBoostModel saved to '%s'.", filepath)

    @classmethod
    def load(cls, filepath: str) -> 'XGBoostModel':
        """Deserialise an XGBoostModel from *filepath*."""
        import pathlib
        if not pathlib.Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: '{filepath}'")
        obj = joblib.load(filepath)
        logger.info("XGBoostModel loaded from '%s'.", filepath)
        return obj

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call .train() first.")
