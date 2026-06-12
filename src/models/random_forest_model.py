"""Random Forest classifier wrapper for AayuSense."""
import logging
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


class RandomForestModel:
    """Encapsulates a Random Forest classifier with grid-search training.

    Args:
        config: A ModelConfig instance with rf_n_estimators_range, rf_max_depth_range,
                rf_min_samples_split, n_folds, random_state attributes.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._model: Optional[GridSearchCV] = None
        self._is_fitted: bool = False
        logger.info("RandomForestModel initialised.")

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
        """Best mean cross-validation score found during grid search."""
        self._check_fitted()
        return float(self._model.best_score_)  # type: ignore[union-attr]

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'RandomForestModel':
        """Fit the model using GridSearchCV.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.

        Returns:
            self (enables method chaining).
        """
        param_grid = {
            'n_estimators':    self.config.rf_n_estimators_range,
            'max_depth':       self.config.rf_max_depth_range,
            'min_samples_split': self.config.rf_min_samples_split,
        }
        logger.info("GridSearchCV for RandomForest — param_grid: %s", param_grid)
        base = RandomForestClassifier(
            random_state=self.config.random_state,
            class_weight='balanced',
            n_jobs=-1,
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
        """Return class predictions for X."""
        self._check_fitted()
        return self._model.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates for X."""
        self._check_fitted()
        return self._model.predict_proba(X)  # type: ignore[union-attr]

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Return feature importances sorted descending.

        Args:
            feature_names: Names matching training columns.

        Returns:
            Ordered dict of {feature: importance}.
        """
        self._check_fitted()
        imps = self._model.best_estimator_.feature_importances_  # type: ignore[union-attr]
        return dict(sorted(zip(feature_names, imps), key=lambda kv: kv[1], reverse=True))

    def save(self, filepath: str) -> None:
        """Serialise the fitted model to *filepath* using joblib."""
        self._check_fitted()
        import pathlib
        pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("RandomForestModel saved to '%s'.", filepath)

    @classmethod
    def load(cls, filepath: str) -> 'RandomForestModel':
        """Deserialise a RandomForestModel from *filepath*.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        import pathlib
        if not pathlib.Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: '{filepath}'")
        obj = joblib.load(filepath)
        logger.info("RandomForestModel loaded from '%s'.", filepath)
        return obj

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call .train() first.")
