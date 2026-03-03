"""
Training Pipeline
=================
Orchestrates the full train -> evaluate -> save workflow for a single model.

Callable via:
    python -m driftsync.training.pipeline --model lstm
    python -m driftsync.training.pipeline --model transformer
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from driftsync.configs import (
    CONFIG,
    LSTMConfig,
    TransformerConfig,
    TrainingConfig,
    DataConfig,
)
from driftsync.data import (
    load_all_sessions,
    preprocess_all_sessions,
    build_sequences_from_df,
    split_data,
    make_dataloaders,
    save_processed,
    load_processed,
)
from driftsync.models import build_model
from driftsync.training.trainer import Trainer
from driftsync.utils import get_logger, set_seed, get_device, compute_classification_metrics

logger = get_logger(__name__)


def run_pipeline(
    model_type: str = "lstm",
    data_cfg: DataConfig | None = None,
    train_cfg: TrainingConfig | None = None,
    model_cfg=None,
    force_reprocess: bool = False,
) -> dict:
    """
    End-to-end training pipeline.

    1. Load raw sessions (or synthetic data).
    2. Preprocess features and labels.
    3. Build sequence arrays.
    4. Split into train / val / test.
    5. Train model.
    6. Evaluate on test set.
    7. Return evaluation results.

    Args:
        model_type: "lstm" or "transformer".
        data_cfg: DataConfig (uses global CONFIG if None).
        train_cfg: TrainingConfig (uses global CONFIG if None).
        model_cfg: Model-specific config.
        force_reprocess: Re-run preprocessing even if processed files exist.

    Returns:
        Dictionary of test metrics.
    """
    data_cfg  = data_cfg  or CONFIG.data
    train_cfg = train_cfg or CONFIG.training
    if model_cfg is None:
        model_cfg = CONFIG.lstm if model_type == "lstm" else CONFIG.transformer

    set_seed(train_cfg.seed)
    device = get_device(train_cfg.device)
    logger.info("Device: %s", device)

    # --- Data ---
    processed_dir = Path(data_cfg.processed_data_dir)
    x_path = processed_dir / "X.npy"
    y_path = processed_dir / "y.npy"

    if not force_reprocess and x_path.exists() and y_path.exists():
        logger.info("Loading pre-processed arrays from %s", processed_dir)
        X, y = load_processed(str(processed_dir))
    else:
        logger.info("Processing raw sessions from %s", data_cfg.raw_data_dir)
        raw_df   = load_all_sessions(data_cfg.raw_data_dir)
        proc_df  = preprocess_all_sessions(raw_df, horizon=data_cfg.prediction_horizon)
        X, y     = build_sequences_from_df(proc_df, seq_len=data_cfg.sequence_length)
        save_processed(X, y, str(processed_dir))

    # Update model input_dim from actual feature count
    input_dim = X.shape[2]
    if hasattr(model_cfg, "input_dim"):
        model_cfg.input_dim = input_dim

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
        X, y,
        train_ratio=data_cfg.train_ratio,
        val_ratio=data_cfg.val_ratio,
    )

    train_loader, val_loader, test_loader = make_dataloaders(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        batch_size=train_cfg.batch_size,
    )

    # --- Model ---
    model = build_model(model_type, model_cfg)
    logger.info("Model: %s  |  params=%d", type(model).__name__, model.count_parameters())

    # --- Train ---
    trainer = Trainer(model, train_loader, val_loader, train_cfg, device)
    history = trainer.train()

    # --- Load best checkpoint for evaluation ---
    trainer.load_best_checkpoint()
    model.eval()

    # --- Test evaluation ---
    all_labels, all_proba = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits  = model(X_batch)
            proba   = torch.sigmoid(logits).cpu().numpy()
            all_proba.append(proba)
            all_labels.append(y_batch.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_proba)

    test_metrics = compute_classification_metrics(y_true, y_pred)
    logger.info(
        "Test  | acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
        test_metrics["accuracy"], test_metrics["precision"],
        test_metrics["recall"],   test_metrics["f1"],
        test_metrics["roc_auc"],
    )

    # Save results
    results_dir = Path(CONFIG.evaluation.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "model": model_type,
        "test_metrics": test_metrics,
        "n_test": int(len(y_true)),
        "n_params": model.count_parameters(),
        "history_final_epoch": {k: v[-1] for k, v in history.items() if v},
    }
    out_path = results_dir / f"{model_type}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved -> %s", out_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DriftSync model.")
    parser.add_argument(
        "--model", type=str, default="lstm",
        choices=["lstm", "transformer"],
        help="Model type to train",
    )
    parser.add_argument(
        "--force-reprocess", action="store_true",
        help="Re-run data preprocessing even if cached files exist",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    train_cfg = TrainingConfig()
    if args.epochs:     train_cfg.max_epochs   = args.epochs
    if args.lr:         train_cfg.learning_rate = args.lr
    if args.batch_size: train_cfg.batch_size    = args.batch_size

    run_pipeline(
        model_type=args.model,
        train_cfg=train_cfg,
        force_reprocess=args.force_reprocess,
    )


if __name__ == "__main__":
    main()
