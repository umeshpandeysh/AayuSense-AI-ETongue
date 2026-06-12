"""Visualisation utilities for AayuSense E-Tongue project."""
import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

_PALETTE = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Figure saved to '%s'.", save_path)
    else:
        plt.show()
    plt.close(fig)


def plot_feature_distributions(
    df: pd.DataFrame,
    sensor_columns: List[str],
    label_column: str,
    save_path: Optional[str] = None,
) -> None:
    """Plot per-class KDE distributions for each sensor column.

    Args:
        df: Input DataFrame.
        sensor_columns: Feature column names to plot.
        label_column: Column containing class labels.
        save_path: Path to save the figure (None = display interactively).
    """
    n = len(sensor_columns)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    labels = df[label_column].unique()
    for ax, col in zip(axes, sensor_columns):
        for lbl, colour in zip(labels, _PALETTE):
            subset = df[df[label_column] == lbl][col].dropna()
            subset.plot.kde(ax=ax, label=lbl, color=colour, linewidth=2)
        ax.set_title(f'Distribution: {col}', fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle('Sensor Feature Distributions by Quality Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plot a Pearson correlation heatmap of numeric columns.

    Args:
        df: Input DataFrame (only numeric columns are used).
        save_path: Path to save the figure (None = display).
    """
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.8), max(5, len(corr) * 0.7)))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
        linewidths=0.5, ax=ax, vmin=-1, vmax=1,
    )
    ax.set_title('Pearson Correlation Heatmap', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Display a colour-coded confusion matrix.

    Args:
        cm: Square confusion matrix array (true labels as rows).
        class_names: Ordered list of class names.
        save_path: Path to save the figure (None = display).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None,
) -> None:
    """Horizontal bar chart of feature importances.

    Args:
        feature_names: Ordered list of feature names.
        importances: Corresponding importance scores.
        model_name: Label used in the chart title.
        save_path: Path to save the figure (None = display).
    """
    sorted_idx = np.argsort(importances)
    colours = plt.cm.viridis(np.linspace(0.2, 0.9, len(feature_names)))
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.35)))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color=colours,
    )
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title(f'Feature Importances — {model_name}', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_cv_scores(
    cv_scores_dict: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing cross-validation scores across models.

    Args:
        cv_scores_dict: Model name -> list of per-fold CV accuracy scores.
        save_path: Path to save the figure (None = display).
    """
    model_names = list(cv_scores_dict.keys())
    means = [np.mean(s) for s in cv_scores_dict.values()]
    stds = [np.std(s) for s in cv_scores_dict.values()]

    x = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 2), 5))
    bars = ax.bar(
        x, means, yerr=stds, capsize=6,
        color=_PALETTE[:len(model_names)], edgecolor='black', linewidth=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Mean CV Accuracy', fontsize=11)
    ax.set_title('Cross-Validation Score Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=10,
        )
    plt.tight_layout()
    _save_or_show(fig, save_path)
