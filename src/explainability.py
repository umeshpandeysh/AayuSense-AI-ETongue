"""
explainability.py
AayuSense-AI-ETongue — SHAP-based Model Explainability

Provides SHAP TreeExplainer wrappers for interpreting Random Forest
and XGBoost classification decisions on sensor readings.
SHAP values reveal which sensors most influence each prediction.
"""

import logging
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Install with: pip install shap")


class AayuSenseExplainer:
    """
    SHAP-based explainability wrapper for AayuSense classifiers.

    Computes per-feature impact scores that reveal which sensor readings
    most influenced the adulteration classification decision — aligned with
    the SHAP output displayed in the AayuSense web dashboard.
    """

    def __init__(self, model, feature_names: List[str]):
        """
        Args:
            model: Trained sklearn/XGBoost classifier.
            feature_names: List of feature column names.
        """
        self.model = model
        self.feature_names = feature_names
        self._explainer = None
        self._is_fitted = False

    def fit(self, X_background: np.ndarray) -> None:
        """
        Fit the SHAP TreeExplainer on background data.

        Args:
            X_background: Background dataset (e.g., training set or sample) for SHAP.
        """
        if not SHAP_AVAILABLE:
            logger.error("Cannot fit explainer: shap not installed.")
            return
        self._explainer = shap.TreeExplainer(self.model, data=X_background)
        self._is_fitted = True
        logger.info("SHAP TreeExplainer fitted on %d background samples", len(X_background))

    def compute_shap_values(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute SHAP values for input samples.

        Args:
            X: Feature matrix to explain (n_samples, n_features).

        Returns:
            SHAP values array or None if SHAP is unavailable.
        """
        if not SHAP_AVAILABLE or not self._is_fitted:
            logger.warning("Explainer not ready. Call fit() first.")
            return None
        shap_values = self._explainer.shap_values(X)
        logger.debug("Computed SHAP values for %d samples", len(X))
        return shap_values

    def get_top_features(
        self,
        shap_values: np.ndarray,
        class_index: int = 0,
        k: int = 3
    ) -> List[Dict]:
        """
        Return the top-k most impactful features for a given class.

        Matches the SHAP output format used in the AayuSense web dashboard:
        [{"feature": "Salt_content", "impact": 0.38}, ...]

        Args:
            shap_values: SHAP values from compute_shap_values().
            class_index: Which class to explain (for multi-class output).
            k: Number of top features to return.

        Returns:
            List of dicts with 'feature' and 'impact' keys.
        """
        if isinstance(shap_values, list):
            vals = np.abs(shap_values[class_index]).mean(axis=0)
        else:
            vals = np.abs(shap_values).mean(axis=0)

        top_indices = np.argsort(vals)[::-1][:k]
        return [
            {"feature": self.feature_names[i], "impact": round(float(vals[i]), 4)}
            for i in top_indices
        ]

    def feature_importance_fallback(self) -> List[Dict]:
        """
        Fallback when SHAP is unavailable: use model's built-in feature importances.

        Returns:
            List of dicts with 'feature' and 'importance' keys.
        """
        if not hasattr(self.model, "feature_importances_"):
            return []
        importances = self.model.feature_importances_
        pairs = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1], reverse=True
        )
        return [{"feature": f, "importance": round(float(imp), 4)} for f, imp in pairs[:10]]

    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Generate a SHAP summary beeswarm plot.

        Args:
            shap_values: SHAP values array.
            X: Feature matrix used for computation.
            save_path: If provided, save plot to this path instead of showing.
        """
        if not SHAP_AVAILABLE:
            logger.warning("Cannot plot: shap not installed.")
            return
        import matplotlib.pyplot as plt
        shap.summary_plot(
            shap_values, X,
            feature_names=self.feature_names,
            show=save_path is None
        )
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()
            logger.info("SHAP summary plot saved to %s", save_path)


if __name__ == "__main__":
    print("AayuSenseExplainer — SHAP-based model interpretability")
    print("Usage: fit(X_train) then compute_shap_values(X_test)")
    print("SHAP available:", SHAP_AVAILABLE)
