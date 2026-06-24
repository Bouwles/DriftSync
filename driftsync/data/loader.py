"""
Session Data Loader
===================
Loads raw JSON session files, validates them, and converts to flat
pandas DataFrames for further processing.
"""

from __future__ import annotations


import json
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

from driftsync.utils import get_logger

logger = get_logger(__name__)


def load_session(path: str | Path) -> pd.DataFrame:
    """
    Load a single session JSON file into a DataFrame.

    Each row is one trial with the following columns:
        trial_idx, timestamp, reaction_time, action_taken,
        target_shape, stimulus_shape, is_correct, elapsed_session_time,
        prev_5_errors_{0..4}

    Args:
        path: Path to session JSON file.

    Returns:
        DataFrame with one row per trial, sorted by trial_idx.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Session file not found: {path}")

    with open(path) as f:
        raw = json.load(f)

    trials = raw.get("trials", [])
    if not trials:
        logger.warning("Session %s has no trials — skipping.", path.name)
        return pd.DataFrame()

    records = []
    for t in trials:
        rec = {
            "session_id":         raw["session_id"],
            "trial_idx":          t["trial_idx"],
            "timestamp":          t["timestamp"],
            "reaction_time":      t["reaction_time"],
            "action_taken":       t["action_taken"],
            "target_shape":       t["target_shape"],
            "stimulus_shape":     t["stimulus_shape"],
            "is_correct":         bool(t["is_correct"]),
            "elapsed_session_time": t["elapsed_session_time"],
        }
        # Flatten prev_5_errors
        prev5 = t.get("prev_5_errors", [False] * 5)
        for k, v in enumerate(prev5[:5]):
            rec[f"prev_error_{k}"] = int(v)
        records.append(rec)

    df = pd.DataFrame(records).sort_values("trial_idx").reset_index(drop=True)
    return df


def load_all_sessions(data_dir: str | Path) -> pd.DataFrame:
    """
    Load all session JSON files from a directory.

    Args:
        data_dir: Directory containing session_*.json files.

    Returns:
        Combined DataFrame across all sessions.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("session_*.json"))

    if not files:
        raise RuntimeError(f"No session files found in {data_dir}. "
                           "Run the simulator or headless generator first.")

    dfs: List[pd.DataFrame] = []
    for f in files:
        df = load_session(f)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise RuntimeError("All session files were empty.")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(
        "Loaded %d sessions, %d total trials from %s",
        len(dfs), len(combined), data_dir,
    )
    return combined
