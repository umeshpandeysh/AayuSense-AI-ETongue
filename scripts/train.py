#!/usr/bin/env python3
"""AayuSense end-to-end training script.

Usage
-----
    python scripts/train.py --data data/raw/sample_sensor_data.csv \\
                            --model rf \\
                            --output models/ \\
                            --report artifacts/

This script:
    1. Loads and preprocesses the sensor CSV.
    2. Runs feature engineering (rolling stats, Rasa profiles, interactions).
    3. Splits into train/test (80/20 stratified).
    4. Trains a Random Forest or XGBoost classifier with GridSearchCV.
    5. Evaluates on the held-out test set and prints a classification report.
    6. Saves the model and label encoder to --output.
    7. Saves a JSON metrics report to --report.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data_pipeline import preprocess_pipeline, SENSOR_COLUMNS
from src.feature_engineering import extract_features
from src.evaluation.metrics import evaluate_classifier, cross_validate_model, print_classification_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
)
logger = logging.getLogger('train')

NON_FEATURE_COLS = {'quality_label', 'timestamp', 'sample_id', 'herb_name'}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='AayuSense training pipeline')
    p.add_argument('--data',   default='data/raw/sample_sensor_data.csv',
                   help='Path to raw sensor CSV')
    p.add_argument('--model',  default='rf', choices=['rf', 'xgb'],
                   help='Classifier to train (rf=RandomForest, xgb=XGBoost)')
    p.add_argument('--output', default='models/',
                   help='Directory to save trained model and encoder')
    p.add_argument('--report', default='artifacts/',
                   help='Directory to save JSON metrics report')
    p.add_argument('--test-size', type=float, default=0.2,
                   help='Fraction of data held out for testing')
    p.add_argument('--folds',  type=int, default=5,
                   help='Number of CV folds')
    p.add_argument('--seed',   type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- 1. Load & preprocess -------------------------------------------------
    logger.info('Loading data from %s', args.data)
    df = preprocess_pipeline(args.data, remove_outliers_flag=True)

    # ---- 2. Feature engineering -----------------------------------------------
    logger.info('Running feature engineering')
    df = extract_features(df, include_rasa=True)

    # ---- 3. Split X / y -------------------------------------------------------
    label_col = 'quality_label'
    if label_col not in df.columns:
        logger.error("Column '%s' not found in data.", label_col)
        sys.exit(1)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].select_dtypes(include=[np.number]).values
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])

    logger.info('Features: %d   Samples: %d   Classes: %s', X.shape[1], len(y), list(le.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed,
    )
    logger.info('Train: %d  Test: %d', len(y_train), len(y_test))

    # ---- 4. Train model -------------------------------------------------------
    if args.model == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth':    [None, 10],
            'min_samples_split': [2, 5],
        }
        base = RandomForestClassifier(class_weight='balanced', random_state=args.seed, n_jobs=-1)
        gs = GridSearchCV(base, param_grid, cv=args.folds, scoring='f1_weighted', n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        logger.info('RF best_params=%s  cv_f1=%.4f', gs.best_params_, gs.best_score_)
    else:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.error('xgboost not installed. Run: pip install xgboost')
            sys.exit(1)
        from sklearn.model_selection import GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth':    [3, 6],
            'learning_rate': [0.05, 0.1],
        }
        base = XGBClassifier(eval_metric='mlogloss', random_state=args.seed, verbosity=0)
        gs = GridSearchCV(base, param_grid, cv=args.folds, scoring='f1_weighted', n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        logger.info('XGB best_params=%s  cv_f1=%.4f', gs.best_params_, gs.best_score_)

    # ---- 5. Evaluate ----------------------------------------------------------
    results = evaluate_classifier(model, X_test, y_test, list(le.classes_))
    print_classification_report(results)

    cv_results = cross_validate_model(model, X, y, n_folds=args.folds)
    logger.info('CV acc=%.4f±%.4f  f1=%.4f±%.4f',
                cv_results['mean_accuracy'], cv_results['std_accuracy'],
                cv_results['mean_f1'], cv_results['std_f1'])

    # ---- 6. Save model --------------------------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f'{args.model}_model.joblib'
    encoder_path = output_dir / 'label_encoder.joblib'
    joblib.dump(model, str(model_path))
    joblib.dump(le, str(encoder_path))
    logger.info('Model saved to %s', model_path)
    logger.info('Encoder saved to %s', encoder_path)

    # ---- 7. Save report -------------------------------------------------------
    report_dir = Path(args.report)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'model': args.model,
        'data': args.data,
        'n_train': int(len(y_train)),
        'n_test': int(len(y_test)),
        'n_features': int(X.shape[1]),
        'classes': list(le.classes_),
        'test_accuracy': results['accuracy'],
        'test_precision': results['precision'],
        'test_recall': results['recall'],
        'test_f1': results['f1'],
        'cv_mean_f1': cv_results['mean_f1'],
        'cv_std_f1': cv_results['std_f1'],
    }
    report_path = report_dir / f'{args.model}_metrics.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info('Metrics report saved to %s', report_path)
    logger.info('Done.')


if __name__ == '__main__':
    main()
