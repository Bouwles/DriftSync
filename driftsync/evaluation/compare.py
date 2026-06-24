"""
Model Comparison Engine
========================
Rigorously compares LSTM vs Transformer on the held-out test set.

Outputs
-------
  - Printed comparison table.
  - ROC comparison plot.
  - Per-model confusion matrices.
  - Calibration reliability diagram.
  - Inference speed benchmark.
  - Attention heatmaps (Transformer).
  - Metric comparison bar chart.
  - JSON summary saved to results dir.

Usage
-----
    python -m driftsync.evaluation.compare
"""

from __future__ import annotations


import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from driftsync.configs import CONFIG, DataConfig, TrainingConfig
from driftsync.data import load_processed, split_data, make_dataloaders
from driftsync.models import build_model
from driftsync.utils import (
    get_logger,
    get_device,
    compute_classification_metrics,
    compute_ece,
    bootstrap_ci,
)
from driftsync.evaluation.plots import (
    plot_roc_curves,
    plot_confusion_matrix,
    plot_calibration,
    plot_inference_speed,
    plot_attention_heatmap,
    plot_metric_comparison,
    plot_training_history,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_trained_model(model_type: str, device: torch.device):
    """Load model from best checkpoint saved during training."""
    ckpt_dir = Path(CONFIG.training.checkpoint_dir)
    # Checkpoint name convention: matches Trainer._save_checkpoint()
    # model_name = classname.lower().replace("driftpredictor", "") -> "lstm" / "transformer"
    name_map = {
        "lstm":        "lstm_best.pt",
        "transformer": "transformer_best.pt",
    }
    ckpt_path = ckpt_dir / name_map[model_type]

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Train the model first: python -m driftsync.training.pipeline --model {model_type}"
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt.get("model_cfg", CONFIG.lstm if model_type == "lstm" else CONFIG.transformer)
    model = build_model(model_type, model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Loaded %s checkpoint (epoch %d)", model_type.upper(), ckpt["epoch"])
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    test_loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on test set. Returns (y_true, y_proba)."""
    all_labels, all_proba = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits  = model(X_batch)
            proba   = torch.sigmoid(logits).cpu().numpy()
            all_proba.append(proba)
            all_labels.append(y_batch.numpy())
    return np.concatenate(all_labels), np.concatenate(all_proba)


def benchmark_inference_speed(
    model,
    test_loader,
    device: torch.device,
    n_repeats: int = 5,
) -> float:
    """
    Measure mean inference latency in milliseconds per sample.

    Warms up the model for 2 batches, then times n_repeats full passes.
    """
    model.eval()
    batches = list(test_loader)
    n_samples = sum(len(y) for _, y in batches)

    # Warm-up
    with torch.no_grad():
        for X_batch, _ in batches[:2]:
            model(X_batch.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed passes
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            for X_batch, _ in batches:
                model(X_batch.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean_total_s = np.mean(times)
    ms_per_sample = (mean_total_s / n_samples) * 1000
    return ms_per_sample


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def run_comparison(
    data_cfg: DataConfig | None = None,
    results_dir: str | None = None,
) -> dict:
    """
    Compare LSTM and Transformer models and generate all plots.

    Args:
        data_cfg: DataConfig.
        results_dir: Directory to save plots and JSON.

    Returns:
        Full comparison results dict.
    """
    data_cfg    = data_cfg    or CONFIG.data
    results_dir = Path(results_dir or CONFIG.evaluation.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(CONFIG.training.device)

    # --- Load processed data ---
    X, y = load_processed(str(data_cfg.processed_data_dir))
    _, _, (X_test, y_test) = split_data(
        X, y,
        train_ratio=data_cfg.train_ratio,
        val_ratio=data_cfg.val_ratio,
    )
    _, _, test_loader = make_dataloaders(
        X[:1], y[:1], X[:1], y[:1], X_test, y_test,
        batch_size=CONFIG.training.batch_size,
    )

    model_types = ["lstm", "transformer"]
    roc_data      = {}
    calib_data    = {}
    metric_data   = {}
    speed_data    = {}
    summary       = {}

    for mtype in model_types:
        logger.info("=" * 50)
        logger.info("Evaluating: %s", mtype.upper())

        try:
            model = load_trained_model(mtype, device)
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue

        # --- Test predictions ---
        y_true, y_proba = evaluate_model(model, test_loader, device)

        # --- Metrics ---
        metrics = compute_classification_metrics(y_true, y_proba, CONFIG.evaluation.threshold)
        ece, _, _, _ = compute_ece(y_true, y_proba, CONFIG.evaluation.calibration_bins)
        metrics["ece"] = ece

        # Bootstrap AUC CI
        from sklearn.metrics import roc_auc_score
        _, auc_lo, auc_hi = bootstrap_ci(
            y_true, y_proba, roc_auc_score, n_bootstrap=CONFIG.evaluation.n_bootstrap
        )
        metrics["auc_ci_lo"] = auc_lo
        metrics["auc_ci_hi"] = auc_hi

        # --- Inference speed ---
        ms = benchmark_inference_speed(model, test_loader, device)
        metrics["inference_ms_per_sample"] = ms

        logger.info(
            "%s | acc=%.4f  f1=%.4f  auc=%.4f [%.4f, %.4f]  ece=%.4f  speed=%.4f ms/sample",
            mtype.upper(), metrics["accuracy"], metrics["f1"],
            metrics["roc_auc"], auc_lo, auc_hi, ece, ms,
        )

        roc_data[mtype]    = (y_true, y_proba)
        calib_data[mtype]  = (y_true, y_proba)
        metric_data[mtype] = metrics
        speed_data[mtype]  = ms
        summary[mtype]     = metrics

        # --- Per-model plots ---
        plot_confusion_matrix(
            y_true, y_proba, mtype,
            results_dir / f"{mtype}_confusion_matrix.png",
        )

        # Attention heatmap (Transformer only)
        if mtype == "transformer":
            attn_maps = model.get_attention_maps()
            if attn_maps:
                last_map = attn_maps[-1][0].cpu().numpy()   # first sample
                plot_attention_heatmap(
                    last_map, layer_idx=len(attn_maps) - 1,
                    out_path=results_dir / "transformer_attention_last_layer.png",
                    title=f"Attention Map — Transformer Layer {len(attn_maps)}",
                )

    if not roc_data:
        logger.error("No model results available. Train models first.")
        return {}

    # --- Comparative plots ---
    plot_roc_curves(roc_data, results_dir / "roc_comparison.png")
    plot_calibration(calib_data, results_dir / "calibration_comparison.png")
    plot_inference_speed(speed_data, results_dir / "inference_speed.png")
    plot_metric_comparison(
        metric_data,
        metric_names=["accuracy", "precision", "recall", "f1", "roc_auc"],
        out_path=results_dir / "metric_comparison.png",
        title="LSTM vs Transformer — DriftSync",
    )

    # --- Print summary table ---
    _print_table(metric_data)

    # --- Save JSON ---
    out_path = results_dir / "comparison_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Comparison summary saved -> %s", out_path)

    return summary


def _print_table(metric_data: dict) -> None:
    """Print a clean comparison table to stdout."""
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "ece", "inference_ms_per_sample"]
    col_w = 12

    header = f"{'Model':<15}" + "".join(f"{c:>{col_w}}" for c in cols)
    print("\n" + "=" * len(header))
    print("  DriftSync — Model Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for model_name, metrics in metric_data.items():
        row = f"{model_name.upper():<15}"
        for c in cols:
            v = metrics.get(c, float("nan"))
            row += f"{v:>{col_w}.4f}"
        print(row)
    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DriftSync models.")
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()
    run_comparison(results_dir=args.results_dir)


if __name__ == "__main__":
    main()
