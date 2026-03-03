"""
Full Experiment Runner
======================
Runs the complete DriftSync research pipeline end-to-end:

    1. Generate synthetic cognitive drift data
    2. Preprocess and build sequence arrays
    3. Train LSTM model
    4. Train Transformer model
    5. Compare models and generate all visualisations

Usage
-----
    python run_experiment.py
    python run_experiment.py --sessions 20 --epochs 50 --quick
"""

import argparse
import json
from pathlib import Path

from driftsync.configs import (
    SimulatorConfig, DataConfig, TrainingConfig, LSTMConfig, TransformerConfig
)
from driftsync.simulator.headless_generator import generate_dataset
from driftsync.data import (
    load_all_sessions, preprocess_all_sessions,
    build_sequences_from_df, split_data, make_dataloaders, save_processed
)
from driftsync.models import build_model
from driftsync.training.trainer import Trainer
from driftsync.evaluation.compare import run_comparison
from driftsync.evaluation.plots import plot_training_history
from driftsync.utils import get_logger, set_seed, get_device, compute_classification_metrics
import numpy as np
import torch

logger = get_logger(__name__, log_file="driftsync/results/experiment.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DriftSync end-to-end experiment.")
    parser.add_argument("--sessions",   type=int,   default=20,   help="Number of synthetic sessions")
    parser.add_argument("--trials",     type=int,   default=200,  help="Trials per session")
    parser.add_argument("--epochs",     type=int,   default=60,   help="Max training epochs")
    parser.add_argument("--horizon",    type=int,   default=5,    help="Prediction horizon K")
    parser.add_argument("--seq-len",    type=int,   default=20,   help="Sequence window length")
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--quick",      action="store_true",
                        help="Quick mode: fewer sessions, epochs, MC samples")
    args = parser.parse_args()

    if args.quick:
        args.sessions = max(args.sessions, 5)
        args.epochs   = min(args.epochs, 20)
        logger.info("Quick mode: sessions=%d  epochs=%d", args.sessions, args.epochs)

    set_seed(args.seed)

    # -----------------------------------------------------------------------
    # Step 1 — Generate data
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Generating synthetic sessions")
    logger.info("=" * 60)

    sim_cfg = SimulatorConfig(num_trials=args.trials)
    generate_dataset(num_sessions=args.sessions, cfg=sim_cfg, base_seed=args.seed)

    # -----------------------------------------------------------------------
    # Step 2 — Preprocess
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing")
    logger.info("=" * 60)

    data_cfg = DataConfig(
        prediction_horizon=args.horizon,
        sequence_length=args.seq_len,
    )

    raw_df  = load_all_sessions(data_cfg.raw_data_dir)
    proc_df = preprocess_all_sessions(raw_df, horizon=data_cfg.prediction_horizon)
    X, y    = build_sequences_from_df(proc_df, seq_len=data_cfg.sequence_length)
    save_processed(X, y, data_cfg.processed_data_dir)

    input_dim = X.shape[2]
    logger.info("Input features: %d  |  Sequences: %d  |  Pos rate: %.3f",
                input_dim, len(X), y.mean())

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
        X, y, train_ratio=data_cfg.train_ratio, val_ratio=data_cfg.val_ratio
    )

    train_loader, val_loader, test_loader = make_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=args.batch_size
    )

    train_cfg = TrainingConfig(
        max_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    device = get_device(train_cfg.device)

    results_dir = Path("driftsync/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # -----------------------------------------------------------------------
    # Steps 3 & 4 — Train models
    # -----------------------------------------------------------------------
    for model_type in ["lstm", "transformer"]:
        logger.info("=" * 60)
        logger.info("STEP 3/4: Training %s", model_type.upper())
        logger.info("=" * 60)

        if model_type == "lstm":
            model_cfg = LSTMConfig(input_dim=input_dim)
        else:
            model_cfg = TransformerConfig(input_dim=input_dim)

        model = build_model(model_type, model_cfg)
        trainer = Trainer(model, train_loader, val_loader, train_cfg, device)
        history = trainer.train()

        # Save training history plot
        plot_training_history(
            history, model_type,
            out_path=results_dir / f"{model_type}_training_history.png",
        )

        # Evaluate on test set (best checkpoint)
        trainer.load_best_checkpoint()
        model.eval()

        all_labels, all_proba = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                proba   = torch.sigmoid(model(X_batch)).cpu().numpy()
                all_proba.append(proba)
                all_labels.append(y_batch.numpy())

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_proba)
        metrics = compute_classification_metrics(y_true, y_pred)

        logger.info(
            "%s Test | acc=%.4f  f1=%.4f  auc=%.4f",
            model_type.upper(), metrics["accuracy"], metrics["f1"], metrics["roc_auc"],
        )
        all_results[model_type] = {
            "test_metrics": metrics,
            "n_params": model.count_parameters(),
        }

    # -----------------------------------------------------------------------
    # Step 5 — Compare
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5: Comparing models and generating plots")
    logger.info("=" * 60)

    comparison = run_comparison(data_cfg=data_cfg, results_dir=str(results_dir))
    all_results["comparison"] = comparison

    # Save final summary
    summary_path = results_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("=" * 60)
    logger.info("Experiment complete!")
    logger.info("Results saved to: %s", results_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
