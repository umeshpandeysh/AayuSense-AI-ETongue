"""Inference script for AayuSense E-Tongue quality classification."""
import argparse
import logging
import pathlib
import sys
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

RAW_FEATURE_NAMES: List[str] = ['ph', 'conductivity', 'turbidity', 'orp']

ENGINEERED_FEATURE_NAMES: List[str] = [
    'ph', 'conductivity', 'turbidity', 'orp',
    'ph_x_conductivity', 'turbidity_over_orp',
    'ph_over_turbidity', 'conductivity_x_orp',
]


def load_model_and_encoder(model_path: str, encoder_path: str):
    """Load a serialised model and label encoder from disk.

    Args:
        model_path: Path to the joblib model file.
        encoder_path: Path to the joblib LabelEncoder file.

    Returns:
        Tuple ``(model, label_encoder)``.

    Raises:
        FileNotFoundError: If either path does not exist.
    """
    for p in (model_path, encoder_path):
        if not pathlib.Path(p).exists():
            raise FileNotFoundError(f"Required file not found: '{p}'")
    model   = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    logger.info("Model loaded from '%s'.", model_path)
    logger.info("Encoder loaded from '%s'.", encoder_path)
    return model, encoder


def _engineer_features(ph: float, conductivity: float, turbidity: float, orp: float) -> np.ndarray:
    """Apply feature engineering to a single raw sensor reading."""
    return np.array([
        ph,
        conductivity,
        turbidity,
        orp,
        ph * conductivity,
        turbidity / (orp + 1e-6),
        ph / (turbidity + 1e-6),
        conductivity * orp,
    ], dtype=np.float32).reshape(1, -1)


def predict_single_reading(
    ph: float,
    conductivity: float,
    turbidity: float,
    orp: float,
    model: Any,
    encoder: Any,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """Classify a single sensor reading.

    Args:
        ph: pH value (0–14).
        conductivity: Conductivity in mS/cm.
        turbidity: Turbidity in NTU.
        orp: Oxidation-Reduction Potential in mV.
        model: Fitted classifier.
        encoder: Fitted LabelEncoder.
        feature_names: Optional feature name list (unused, kept for API compatibility).

    Returns:
        Dict with keys: label (str), confidence (float 0-1), probabilities (dict).
    """
    X = _engineer_features(ph, conductivity, turbidity, orp)
    proba = model.predict_proba(X)[0]
    idx   = int(np.argmax(proba))
    label = encoder.inverse_transform([idx])[0]
    confidence = float(proba[idx])
    class_labels = encoder.inverse_transform(list(range(len(proba))))
    probabilities = {cls: round(float(p), 4) for cls, p in zip(class_labels, proba)}
    result = {'label': label, 'confidence': round(confidence, 4), 'probabilities': probabilities}
    logger.info("Prediction: %s (confidence=%.1f%%)", label, confidence * 100)
    return result


def predict_from_csv(
    csv_path: str,
    model_path: str,
    encoder_path: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Batch-predict quality labels from a CSV file.

    Args:
        csv_path: Path to input CSV with columns ph, conductivity, turbidity, orp.
        model_path: Path to serialised model.
        encoder_path: Path to serialised LabelEncoder.
        output_path: If provided, save predictions CSV here.

    Returns:
        DataFrame with original data plus 'predicted_label' and 'confidence' columns.
    """
    model, encoder = load_model_and_encoder(model_path, encoder_path)
    df = pd.read_csv(csv_path)
    missing = [c for c in RAW_FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    preds = []
    for _, row in df.iterrows():
        r = predict_single_reading(
            float(row['ph']), float(row['conductivity']),
            float(row['turbidity']), float(row['orp']),
            model, encoder,
        )
        preds.append({'predicted_label': r['label'], 'confidence': r['confidence']})

    df = pd.concat([df, pd.DataFrame(preds)], axis=1)
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info("Batch predictions written to '%s'.", output_path)
    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='AayuSense — herbal quality prediction CLI.')
    p.add_argument('--model',         required=True,  help='Path to joblib model.')
    p.add_argument('--encoder',       required=True,  help='Path to joblib LabelEncoder.')
    p.add_argument('--ph',            type=float,     help='pH reading.')
    p.add_argument('--conductivity',  type=float,     help='Conductivity (mS/cm).')
    p.add_argument('--turbidity',     type=float,     help='Turbidity (NTU).')
    p.add_argument('--orp',           type=float,     help='ORP (mV).')
    p.add_argument('--csv',                           help='CSV file for batch prediction.')
    return p.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()

    if args.csv:
        df = predict_from_csv(args.csv, args.model, args.encoder)
        print(df[['predicted_label', 'confidence']].to_string(index=False))
        return

    if any(v is None for v in [args.ph, args.conductivity, args.turbidity, args.orp]):
        print("ERROR: Provide --ph, --conductivity, --turbidity, --orp for single prediction.")
        sys.exit(1)

    model, encoder = load_model_and_encoder(args.model, args.encoder)
    result = predict_single_reading(
        args.ph, args.conductivity, args.turbidity, args.orp, model, encoder,
    )
    print(f"\nPredicted Quality : {result['label']}")
    print(f"Confidence        : {result['confidence'] * 100:.1f}%")
    print("\nClass Probabilities:")
    for cls, prob in result['probabilities'].items():
        print(f"  {cls:<15}: {prob * 100:.1f}%")


if __name__ == '__main__':
    main()
