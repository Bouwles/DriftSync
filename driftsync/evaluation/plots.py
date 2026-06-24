"""
Visualisation Library
=====================
All publication-quality plots for DriftSync experiments:

  - ROC curves (single and comparative)
  - Confusion matrices (with annotation)
  - Calibration plots (reliability diagrams)
  - Attention heatmaps (Transformer)
  - Training history curves
  - Inference speed bar chart
"""

from __future__ import annotations


from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for headless / server)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from driftsync.utils import compute_ece, compute_roc_curve, compute_confusion_matrix

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

STYLE = {
    "figure.facecolor":  "#0f0f19",
    "axes.facecolor":    "#0f0f19",
    "axes.edgecolor":    "#444",
    "axes.labelcolor":   "#ccc",
    "xtick.color":       "#999",
    "ytick.color":       "#999",
    "text.color":        "#eee",
    "grid.color":        "#333",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "axes.grid":         True,
    "font.family":       "monospace",
    "lines.linewidth":   2.0,
}

MODEL_COLORS = {
    "lstm":        "#64dcb4",   # teal-green
    "transformer": "#b464dc",   # purple
}


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


def save_fig(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# ROC Curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_path: str | Path,
    title: str = "ROC Curves — DriftSync",
) -> None:
    """
    Plot ROC curves for one or more models on the same axes.

    Args:
        results: {model_name: (y_true, y_proba)} dictionary.
        out_path: Output file path.
        title: Plot title.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, (y_true, y_proba) in results.items():
        fpr, tpr, auc = compute_roc_curve(y_true, y_proba)
        color = MODEL_COLORS.get(name, "#ffffff")
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{name.upper()}  AUC={auc:.3f}")

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "--", color="#555", lw=1.5, label="Random")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.25)

    save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    out_path: str | Path,
    threshold: float = 0.5,
) -> None:
    """
    Plot an annotated confusion matrix.
    """
    _apply_style()
    cm = compute_confusion_matrix(y_true, y_proba, threshold)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["No Error", "Error"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    # Cell annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "#222",
                fontsize=15, fontweight="bold",
            )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix — {model_name.upper()}", fontsize=12, fontweight="bold")

    save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Calibration Plot (Reliability Diagram)
# ---------------------------------------------------------------------------

def plot_calibration(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
    n_bins: int = 10,
    title: str = "Calibration — Reliability Diagram",
) -> None:
    """
    Reliability diagram comparing model confidence vs. empirical accuracy.

    Args:
        results: {model_name: (y_true, y_proba)}.
        out_path: Output path.
        n_bins: Number of confidence bins.
        title: Plot title.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Perfect calibration")

    for name, (y_true, y_proba) in results.items():
        ece, bin_conf, bin_acc, bin_cnt = compute_ece(y_true, y_proba, n_bins)
        color = MODEL_COLORS.get(name, "#ffffff")
        # Only plot bins with actual data
        mask = bin_cnt > 0
        ax.plot(
            bin_conf[mask], bin_acc[mask],
            "o-", color=color, lw=2,
            label=f"{name.upper()}  ECE={ece:.4f}",
            markersize=6,
        )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Positives (Accuracy)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.25)

    save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Training History
# ---------------------------------------------------------------------------

def plot_training_history(
    history: Dict[str, list],
    model_name: str,
    out_path: str | Path,
) -> None:
    """
    4-panel training history: loss, accuracy, F1, learning rate.
    """
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Training History — {model_name.upper()}", fontsize=14, fontweight="bold")

    color = MODEL_COLORS.get(model_name.lower(), "#64dcb4")
    epochs = range(1, len(history["train_loss"]) + 1)

    panels = [
        (axes[0, 0], "Loss",     "train_loss", "val_loss"),
        (axes[0, 1], "Accuracy", "train_acc",  "val_acc"),
        (axes[1, 0], "F1 Score", "train_f1",   "val_f1"),
    ]

    for ax, title, train_key, val_key in panels:
        if train_key in history and history[train_key]:
            ax.plot(epochs, history[train_key], label="Train", color=color, lw=2)
        if val_key in history and history[val_key]:
            ax.plot(epochs, history[val_key], "--", label="Val", color="#ffaa44", lw=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(framealpha=0.25)

    # Learning rate
    ax_lr = axes[1, 1]
    if history.get("lr"):
        ax_lr.semilogy(epochs, history["lr"], color="#ff6464", lw=2)
    ax_lr.set_title("Learning Rate")
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("LR (log scale)")

    plt.tight_layout()
    save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Inference Speed
# ---------------------------------------------------------------------------

def plot_inference_speed(
    speed_data: Dict[str, float],   # {model_name: ms_per_sample}
    out_path: str | Path,
) -> None:
    """
    Bar chart comparing inference latency in ms per sample.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    names  = list(speed_data.keys())
    values = list(speed_data.values())
    colors = [MODEL_COLORS.get(n.lower(), "#888") for n in names]

    bars = ax.bar(names, values, color=colors, edgecolor="#333", linewidth=1.2, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f} ms",
            ha="center", va="bottom", fontsize=11,
        )

    ax.set_ylabel("Inference latency (ms / sample)")
    ax.set_title("Inference Speed Comparison", fontsize=12, fontweight="bold")

    save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Attention Heatmap (Transformer)
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    attn_weights: np.ndarray,
    layer_idx: int,
    out_path: str | Path,
    title: str | None = None,
) -> None:
    """
    Visualise a (L, L) attention weight matrix as a heatmap.

    Args:
        attn_weights: (L, L) attention map from one Transformer layer.
        layer_idx: Index of encoder layer (for labelling).
        out_path: Output path.
        title: Custom title (optional).
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        attn_weights, cmap="viridis", aspect="auto",
        vmin=0.0, vmax=attn_weights.max(),
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("Key position (t)")
    ax.set_ylabel("Query position (t)")
    ax.set_title(
        title or f"Self-Attention Weights — Layer {layer_idx + 1}",
        fontsize=12, fontweight="bold",
    )

    save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Combined model comparison bar chart
# ---------------------------------------------------------------------------

def plot_metric_comparison(
    metrics: Dict[str, Dict[str, float]],   # {model: {metric: value}}
    metric_names: List[str],
    out_path: str | Path,
    title: str = "Model Comparison",
) -> None:
    """
    Grouped bar chart comparing selected metrics across models.
    """
    _apply_style()
    n_metrics = len(metric_names)
    n_models  = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(7, n_metrics * 2), 5))

    for i, (model_name, model_metrics) in enumerate(metrics.items()):
        values = [model_metrics.get(m, 0.0) for m in metric_names]
        offset = (i - (n_models - 1) / 2) * width
        color  = MODEL_COLORS.get(model_name.lower(), "#888")
        bars = ax.bar(x + offset, values, width=width * 0.9,
                      label=model_name.upper(), color=color, edgecolor="#333")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.3)

    save_fig(fig, out_path)
