"""
Sequence Dataset
================
Converts preprocessed trial DataFrames into fixed-length sequence
windows and wraps them in a PyTorch Dataset.

Sequence window of length L ending at trial i:
    X[i] = features[i-L+1 : i+1]   shape: (L, num_features)
    y[i] = label[i]                 binary scalar

Split strategy:
    Chronological split by session ordering (no data leakage).
    Training sessions -> train set.
    Next sessions     -> val set.
    Last sessions     -> test set.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Optional

from driftsync.utils import get_logger

logger = get_logger(__name__)

FEATURE_COLS = [
    "reaction_time_norm",
    "correctness",
    "elapsed_time_norm",
    "rolling_error_rate_5",
    "rolling_error_rate_10",
    "inter_trial_interval_norm",
    "cumulative_errors_norm",
    "streak_correct",
    "streak_incorrect",
    "target_match",
    "action_click",
    # Extended features (v2)
    "rolling_rt_variance",
    "time_since_last_error_norm",
    "rt_trend",
    "fatigue_index",
]


# ---------------------------------------------------------------------------
# Sequence extraction
# ---------------------------------------------------------------------------

def extract_sequences(
    df: pd.DataFrame,
    seq_len: int,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract overlapping windows from a session DataFrame.

    Args:
        df: Processed DataFrame (sorted by trial_idx).
        seq_len: Window length L.
        feature_cols: Feature column names.

    Returns:
        X: (N, L, F) array of float32 sequences.
        y: (N,) array of int32 binary labels.
    """
    X_list, y_list = [], []
    features = df[feature_cols].values.astype(np.float32)
    labels   = df["label"].values.astype(np.int32)
    n = len(features)

    for i in range(seq_len - 1, n):
        window = features[i - seq_len + 1: i + 1]   # (L, F)
        X_list.append(window)
        y_list.append(labels[i])

    if not X_list:
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), np.empty(0, dtype=np.int32)

    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int32)


def build_sequences_from_df(
    processed_df: pd.DataFrame,
    seq_len: int,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sequence arrays from the full multi-session DataFrame.
    Sequences do NOT cross session boundaries.

    Args:
        processed_df: Output of preprocess_all_sessions.
        seq_len: Sequence length.
        feature_cols: Feature columns (defaults to FEATURE_COLS).

    Returns:
        X: (N_total, L, F) float32.
        y: (N_total,) int32.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    X_all, y_all = [], []
    for sid in processed_df["session_id"].unique():
        sess_df = processed_df[processed_df["session_id"] == sid].reset_index(drop=True)
        X_s, y_s = extract_sequences(sess_df, seq_len, feature_cols)
        if len(X_s) > 0:
            X_all.append(X_s)
            y_all.append(y_s)

    if not X_all:
        raise RuntimeError("No sequences could be extracted. Check seq_len vs. session length.")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    logger.info("Sequences: X=%s  y=%s  pos_rate=%.3f", X.shape, y.shape, y.mean())
    return X, y


# ---------------------------------------------------------------------------
# Train / Val / Test split
# ---------------------------------------------------------------------------

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
    seed: int          = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Chronological split (no shuffling — preserves temporal order).

    Returns:
        ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    n = len(X)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    X_train, y_train = X[:n_train],            y[:n_train]
    X_val,   y_val   = X[n_train: n_train + n_val], y[n_train: n_train + n_val]
    X_test,  y_test  = X[n_train + n_val:],    y[n_train + n_val:]

    logger.info(
        "Split -> train: %d  val: %d  test: %d  (pos_rates: %.3f / %.3f / %.3f)",
        len(X_train), len(X_val), len(X_test),
        y_train.mean(), y_val.mean(), y_test.mean(),
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class CognitiveDriftDataset(Dataset):
    """
    PyTorch Dataset wrapping (X, y) sequence arrays.

    Args:
        X: (N, L, F) float32 feature sequences.
        y: (N,) int32 binary labels.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap split arrays in DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = CognitiveDriftDataset(X_train, y_train)
    val_ds   = CognitiveDriftDataset(X_val,   y_val)
    test_ds  = CognitiveDriftDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_processed(X: np.ndarray, y: np.ndarray, out_dir: str) -> None:
    """Save processed arrays to disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    logger.info("Saved processed arrays to %s", out_dir)


def load_processed(in_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load processed arrays from disk."""
    in_dir = Path(in_dir)
    X = np.load(in_dir / "X.npy")
    y = np.load(in_dir / "y.npy")
    logger.info("Loaded X=%s y=%s from %s", X.shape, y.shape, in_dir)
    return X, y
