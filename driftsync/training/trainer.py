"""
Training Loop
=============
Implements:
  - Full training loop with BCE loss.
  - Early stopping based on validation loss.
  - ReduceLROnPlateau learning rate scheduling.
  - Gradient clipping.
  - Best-model checkpointing.
  - Per-epoch metric logging.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from driftsync.configs import TrainingConfig
from driftsync.models.base import DriftPredictor
from driftsync.utils import get_logger, compute_classification_metrics

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Epoch helpers
# ---------------------------------------------------------------------------

def _run_epoch(
    model: DriftPredictor,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float,
    train: bool,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Run a single training or validation epoch.

    Returns:
        (mean_loss, all_labels, all_proba)
    """
    model.train(train)

    total_loss = 0.0
    all_labels: List[np.ndarray] = []
    all_proba:  List[np.ndarray] = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(X_batch)                   # (B,)
            loss   = criterion(logits, y_batch)

        if train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * len(y_batch)
        proba = torch.sigmoid(logits).detach().cpu().numpy()
        all_labels.append(y_batch.cpu().numpy())
        all_proba.append(proba)

    n = sum(len(l) for l in all_labels)
    mean_loss = total_loss / max(n, 1)
    return (
        mean_loss,
        np.concatenate(all_labels),
        np.concatenate(all_proba),
    )


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class Trainer:
    """
    Manages the training lifecycle for a DriftPredictor model.

    Args:
        model: Model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        cfg: TrainingConfig.
        device: torch.device.
    """

    def __init__(
        self,
        model: DriftPredictor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainingConfig,
        device: torch.device,
    ):
        self.model   = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg     = cfg
        self.device  = device

        # --- Compute positive class weight for imbalanced data ---
        all_labels = np.concatenate([y.numpy() for _, y in train_loader])
        n_pos = all_labels.sum()
        n_neg = len(all_labels) - n_pos
        pos_weight = torch.tensor(n_neg / max(n_pos, 1), device=device).float()
        logger.info("BCE pos_weight=%.3f  (n_pos=%d, n_neg=%d)", pos_weight.item(), n_pos, n_neg)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            min_lr=cfg.min_lr,
        )

        # --- History ---
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
            "train_f1":   [], "val_f1":   [],
            "val_auc":    [], "lr":       [],
        }

        # --- Early stopping state ---
        self._best_val_loss = float("inf")
        self._patience_counter = 0
        self._best_epoch = 0

        # --- Checkpoint dir ---
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, on_epoch_end=None) -> Dict[str, List[float]]:
        """
        Run training for up to cfg.max_epochs epochs.

        Args:
            on_epoch_end: Optional callable(epoch, metrics_dict) called after
                          each epoch. metrics_dict has keys: train_loss,
                          val_loss, val_acc, val_f1, val_auc.

        Returns:
            Training history dictionary.
        """
        logger.info(
            "Training %s  |  params=%d  |  device=%s",
            type(self.model).__name__,
            self.model.count_parameters(),
            self.device,
        )
        t0 = time.time()

        for epoch in range(1, self.cfg.max_epochs + 1):
            ep_t0 = time.time()

            # --- Train ---
            train_loss, train_labels, train_proba = _run_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.device, self.cfg.grad_clip, train=True,
            )
            train_metrics = compute_classification_metrics(train_labels, train_proba)

            # --- Validate ---
            val_loss, val_labels, val_proba = _run_epoch(
                self.model, self.val_loader, self.criterion,
                None, self.device, self.cfg.grad_clip, train=False,
            )
            val_metrics = compute_classification_metrics(val_labels, val_proba)

            # --- LR scheduler ---
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)

            # --- Record history ---
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["train_f1"].append(train_metrics["f1"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_auc"].append(val_metrics["roc_auc"])
            self.history["lr"].append(current_lr)

            ep_time = time.time() - ep_t0

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d | lr=%.2e | "
                    "train loss=%.4f acc=%.3f f1=%.3f | "
                    "val loss=%.4f acc=%.3f f1=%.3f auc=%.3f | %.1fs",
                    epoch, self.cfg.max_epochs, current_lr,
                    train_loss, train_metrics["accuracy"], train_metrics["f1"],
                    val_loss,   val_metrics["accuracy"],   val_metrics["f1"],
                    val_metrics["roc_auc"], ep_time,
                )

            # --- Epoch callback (for live UI progress) ---
            if on_epoch_end is not None:
                on_epoch_end(epoch, {
                    "train_loss": train_loss,
                    "val_loss":   val_loss,
                    "val_acc":    val_metrics["accuracy"],
                    "val_f1":     val_metrics["f1"],
                    "val_auc":    val_metrics["roc_auc"],
                    "max_epochs": self.cfg.max_epochs,
                })

            # --- Checkpointing ---
            improved = val_loss < self._best_val_loss - self.cfg.early_stop_delta
            if improved:
                self._best_val_loss = val_loss
                self._best_epoch    = epoch
                self._patience_counter = 0
                self._save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self._patience_counter += 1

            # --- Early stopping ---
            if self._patience_counter >= self.cfg.early_stop_patience:
                logger.info(
                    "Early stopping at epoch %d (best=%d, val_loss=%.4f)",
                    epoch, self._best_epoch, self._best_val_loss,
                )
                break

        total_time = time.time() - t0
        logger.info(
            "Training complete in %.1fs | best epoch=%d | best val_loss=%.4f",
            total_time, self._best_epoch, self._best_val_loss,
        )
        return self.history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> None:
        model_name = type(self.model).__name__.lower().replace("driftpredictor", "")
        ckpt = {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config":               self.cfg,
            "model_cfg":            self.model.cfg,
        }
        path = Path(self.cfg.checkpoint_dir) / f"{model_name}_best.pt"
        torch.save(ckpt, path)

    def load_best_checkpoint(self) -> int:
        """Load the best checkpoint saved during training. Returns best epoch."""
        model_name = type(self.model).__name__.lower().replace("driftpredictor", "")
        path = Path(self.cfg.checkpoint_dir) / f"{model_name}_best.pt"
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded best checkpoint (epoch %d, val_loss=%.4f)", ckpt["epoch"], ckpt["val_loss"])
        return ckpt["epoch"]
