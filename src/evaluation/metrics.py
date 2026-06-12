"""Model evaluation utilities for the AayuSense project."""
import logging
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)


def evaluate_classifier(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: List[str],
) -> Dict[str, Any]:
    """Compute classification metrics on a held-out test set.

    Args:
        model: Fitted classifier with a ``.predict()`` method.
        X_test: Feature matrix for the test set.
        y_test: True labels.
        label_names: Ordered list of class names.

    Returns:
        Dict with accuracy, precision, recall, f1, classification_report, model_name.
    """
    y_pred = model.predict(X_test)
    results = {
        'model_name': type(model).__name__,
        'accuracy':   round(accuracy_score(y_test, y_pred), 4),
        'precision':  round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
        'recall':     round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
        'f1':         round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
        'classification_report': classification_report(
            y_test, y_pred, target_names=label_names, zero_division=0
        ),
    }
    logger.info(
        "Evaluation — accuracy=%.4f  f1=%.4f", results['accuracy'], results['f1']
    )
    return results


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
) -> Dict[str, float]:
    """Run stratified k-fold cross-validation.

    Args:
        model: Sklearn-compatible classifier (need not be pre-fitted).
        X: Full feature matrix.
        y: Full label array.
        n_folds: Number of CV folds.

    Returns:
        Dict with mean_accuracy, std_accuracy, mean_f1, std_f1.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy',    n_jobs=-1)
    f1  = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
    results = {
        'mean_accuracy': round(float(acc.mean()), 4),
        'std_accuracy':  round(float(acc.std()),  4),
        'mean_f1':       round(float(f1.mean()),  4),
        'std_f1':        round(float(f1.std()),   4),
    }
    logger.info(
        "CV — acc=%.4f±%.4f  f1=%.4f±%.4f",
        results['mean_accuracy'], results['std_accuracy'],
        results['mean_f1'],       results['std_f1'],
    )
    return results


def print_classification_report(results_dict: Dict[str, Any]) -> None:
    """Pretty-print evaluation metrics from evaluate_classifier().

    Args:
        results_dict: Dict returned by :func:`evaluate_classifier`.
    """
    sep = '=' * 60
    print(sep)
    print(f" Model      : {results_dict.get('model_name', 'N/A')}")
    print(f" Accuracy   : {results_dict.get('accuracy',   'N/A')}")
    print(f" Precision  : {results_dict.get('precision',  'N/A')}")
    print(f" Recall     : {results_dict.get('recall',     'N/A')}")
    print(f" F1 (wtd)   : {results_dict.get('f1',         'N/A')}")
    print(sep)
    if 'classification_report' in results_dict:
        print(results_dict['classification_report'])


def compare_models(results_list: List[Dict[str, Any]]) -> None:
    """Print a comparison table for multiple evaluate_classifier() outputs.

    Args:
        results_list: List of dicts returned by :func:`evaluate_classifier`.
    """
    header = f"{'Model':<30} {'Accuracy':>10} {'Precision':>11} {'Recall':>8} {'F1':>8}"
    sep = '=' * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results_list:
        print(
            f"{r.get('model_name', '?'):<30} "
            f"{r.get('accuracy', 0):>10.4f} "
            f"{r.get('precision', 0):>11.4f} "
            f"{r.get('recall', 0):>8.4f} "
            f"{r.get('f1', 0):>8.4f}"
        )
    print(sep)
