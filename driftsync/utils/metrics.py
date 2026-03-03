"""
Evaluation Metrics
==================
All classification and calibration metrics used for model evaluation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from typing import Dict, Tuple


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute a full set of binary classification metrics.

    Args:
        y_true: Ground-truth binary labels (N,).
        y_pred_proba: Predicted probabilities for positive class (N,).
        threshold: Decision threshold.

    Returns:
        Dictionary of metric name -> value.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else 0.0,
    }
    return metrics


def compute_ece(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Expected Calibration Error (ECE).

    Partitions predictions into confidence bins and measures the mean
    absolute difference between confidence and empirical accuracy.

    ECE = Σ_b (|B_b| / N) * |acc(B_b) − conf(B_b)|

    Args:
        y_true: Ground-truth binary labels (N,).
        y_pred_proba: Predicted probabilities (N,).
        n_bins: Number of equal-width bins in [0, 1].

    Returns:
        Tuple of (ece_value, bin_confidences, bin_accuracies, bin_counts).
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidences = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include upper edge in last bin
        mask = (y_pred_proba >= lo) & (y_pred_proba < hi if i < n_bins - 1 else y_pred_proba <= hi)
        if mask.sum() > 0:
            bin_confidences[i] = y_pred_proba[mask].mean()
            bin_accuracies[i] = y_true[mask].mean()
            bin_counts[i] = mask.sum()

    ece = float(np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / len(y_true))
    return ece, bin_confidences, bin_accuracies, bin_counts


def compute_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve and AUC.

    Returns:
        (fpr, tpr, auc_value)
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
    return fpr, tpr, float(auc)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return 2×2 confusion matrix."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals for a scalar metric.

    Args:
        y_true: Ground-truth labels.
        y_pred_proba: Predicted probabilities.
        metric_fn: Callable(y_true, y_pred_proba) -> float.
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level.
        seed: RNG seed.

    Returns:
        (mean, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores[i] = metric_fn(y_true[idx], y_pred_proba[idx])

    alpha = 1 - ci
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return float(scores.mean()), lower, upper
