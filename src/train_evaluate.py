"""
train_evaluate.py
AayuSense-AI-ETongue — Model Training & Evaluation

Trains Random Forest and XGBoost classifiers on sensor feature data,
performs k-fold cross-validation, and evaluates with precision, recall,
F1-score, and confusion matrix.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] XGBoost not installed. Install with: pip install xgboost")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_features(filepath: str, label_column: str = "quality_label"):
    """
    Load feature matrix and labels from a processed CSV.

    Args:
        filepath: Path to processed feature CSV.
        label_column: Name of the target label column.

    Returns:
        X (features), y (labels), label_encoder
    """
    df = pd.read_csv(filepath)
    le = LabelEncoder()
    y = le.fit_transform(df[label_column])
    X = df.drop(columns=[label_column])
    # Drop non-numeric or timestamp columns
    X = X.select_dtypes(include=[np.number])
    print(f"[INFO] Loaded features: {X.shape[1]} features, {len(y)} samples")
    print(f"[INFO] Classes: {list(le.classes_)}")
    return X, y, le


def train_random_forest(X, y, n_folds: int = 5) -> dict:
    """
    Train and cross-validate a Random Forest classifier.

    Args:
        X: Feature matrix.
        y: Target labels.
        n_folds: Number of folds for cross-validation.

    Returns:
        Dictionary with trained model and cross-validation scores.
    """
    print("\n[INFO] Training Random Forest...")
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    gs = GridSearchCV(rf, rf_params, cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=1)
    gs.fit(X, y)
    print(f"[INFO] Best RF params: {gs.best_params_}")
    print(f"[INFO] Best RF CV F1: {gs.best_score_:.4f}")
    return {"model": gs.best_estimator_, "cv_score": gs.best_score_, "name": "RandomForest"}


def train_xgboost(X, y, n_folds: int = 5) -> dict:
    """
    Train and cross-validate an XGBoost classifier.

    Args:
        X: Feature matrix.
        y: Target labels.
        n_folds: Number of folds for cross-validation.

    Returns:
        Dictionary with trained model and cross-validation scores.
    """
    if not XGBOOST_AVAILABLE:
        print("[WARN] Skipping XGBoost — not installed.")
        return None

    print("\n[INFO] Training XGBoost...")
    xgb_params = {
        "n_estimators": [100, 200],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1]
    }
    xgb = XGBClassifier(random_state=42, eval_metric="mlogloss", verbosity=0)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    gs = GridSearchCV(xgb, xgb_params, cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=1)
    gs.fit(X, y)
    print(f"[INFO] Best XGB params: {gs.best_params_}")
    print(f"[INFO] Best XGB CV F1: {gs.best_score_:.4f}")
    return {"model": gs.best_estimator_, "cv_score": gs.best_score_, "name": "XGBoost"}


def evaluate_model(model, X_test, y_test, label_names: list):
    """
    Evaluate a trained classifier on test data.

    Args:
        model: Trained sklearn-compatible classifier.
        X_test: Test feature matrix.
        y_test: True test labels.
        label_names: List of class label names.
    """
    y_pred = model.predict(X_test)
    print("\n[INFO] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))
    print("[INFO] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def save_model(model_dict: dict, label_encoder):
    """
    Save trained model and label encoder to disk.

    Args:
        model_dict: Dictionary containing 'model' and 'name' keys.
        label_encoder: Fitted LabelEncoder instance.
    """
    name = model_dict["name"]
    joblib.dump(model_dict["model"], MODELS_DIR / f"{name}_model.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")
    print(f"[INFO] Model saved to models/{name}_model.pkl")


if __name__ == "__main__":
    print("[INFO] AayuSense Training Pipeline")
    print("[INFO] Place processed feature CSV at: data/processed/features.csv")
    print("[INFO] Ensure column 'quality_label' contains class labels.")
    print("[INFO] Then run: python src/train_evaluate.py")
    print()
    print("[NOTE] To use with your data:")
    print("  X, y, le = load_features('data/processed/features.csv')")
    print("  rf_result = train_random_forest(X, y)")
    print("  xgb_result = train_xgboost(X, y)")
    print("  save_model(rf_result, le)")
