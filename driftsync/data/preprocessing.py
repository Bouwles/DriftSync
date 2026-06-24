"""
Feature Engineering & Preprocessing
=====================================
Transforms raw trial DataFrames into a rich feature set suitable for
sequence modelling.

Feature set per trial
---------------------
1.  reaction_time_norm         – reaction time normalised per-session [0, 1]
2.  correctness                – 1.0 = correct, 0.0 = error
3.  elapsed_time_norm          – elapsed session time / total session time
4.  rolling_error_rate_5       – error rate over the last 5 trials
5.  rolling_error_rate_10      – error rate over the last 10 trials
6.  inter_trial_interval_norm  – time since last trial (gap), normalised
7.  cumulative_errors_norm     – cumulative errors / trial_idx
8.  streak_correct             – consecutive correct answers (normalised)
9.  streak_incorrect           – consecutive incorrect answers (normalised)
10. target_match               – 1 if stimulus == target shape else 0
11. action_click               – 1 if player clicked else 0

TARGET
------
Binary label: did any error occur within the next K trials?
"""

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import RobustScaler

from driftsync.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to a per-session DataFrame.

    Args:
        df: Raw session DataFrame (single session, sorted by trial_idx).

    Returns:
        DataFrame with additional feature columns.
    """
    df = df.copy().sort_values("trial_idx").reset_index(drop=True)
    n = len(df)

    # --- Normalise reaction time per session (robust to outliers) ---
    rt = df["reaction_time"].values
    rt_norm = (rt - np.median(rt)) / (np.percentile(rt, 75) - np.percentile(rt, 25) + 1e-6)
    # Clip to [-3, 3] and rescale to [0, 1]
    rt_norm = np.clip(rt_norm, -3, 3) / 6.0 + 0.5
    df["reaction_time_norm"] = rt_norm

    # --- Correctness ---
    df["correctness"] = df["is_correct"].astype(float)

    # --- Elapsed time normalised ---
    max_elapsed = df["elapsed_session_time"].max()
    df["elapsed_time_norm"] = df["elapsed_session_time"] / (max_elapsed + 1e-6)

    # --- Rolling error rates ---
    error_series = (~df["is_correct"]).astype(float)
    df["rolling_error_rate_5"]  = error_series.rolling(5,  min_periods=1).mean()
    df["rolling_error_rate_10"] = error_series.rolling(10, min_periods=1).mean()

    # --- Inter-trial interval (time gap between trials) ---
    ts = df["timestamp"].values
    iti = np.diff(ts, prepend=ts[0])
    iti_norm = np.clip(iti / (np.median(iti) + 1e-6), 0, 5) / 5.0
    df["inter_trial_interval_norm"] = iti_norm

    # --- Cumulative errors normalised ---
    cumulative_errors = error_series.cumsum().values
    df["cumulative_errors_norm"] = cumulative_errors / (np.arange(1, n + 1))

    # --- Streaks ---
    streak_correct   = np.zeros(n, dtype=float)
    streak_incorrect = np.zeros(n, dtype=float)
    c_streak = i_streak = 0
    for j, correct in enumerate(df["is_correct"].values):
        if correct:
            c_streak += 1
            i_streak  = 0
        else:
            i_streak += 1
            c_streak  = 0
        streak_correct[j]   = c_streak
        streak_incorrect[j] = i_streak
    # Normalise by window = 20
    df["streak_correct"]   = np.clip(streak_correct   / 20.0, 0.0, 1.0)
    df["streak_incorrect"] = np.clip(streak_incorrect / 20.0, 0.0, 1.0)

    # --- Stimulus match ---
    df["target_match"] = (df["stimulus_shape"] == df["target_shape"]).astype(float)

    # --- Action: click = 1 ---
    df["action_click"] = (df["action_taken"] == "click").astype(float)

    # --- Rolling RT variance (input inconsistency score) ---
    rt_series = pd.Series(rt)
    rt_var    = rt_series.rolling(5, min_periods=1).std().fillna(0.0)
    iqr = float(np.percentile(rt, 75) - np.percentile(rt, 25)) + 1e-6
    df["rolling_rt_variance"] = np.clip(rt_var.values / iqr, 0.0, 3.0) / 3.0

    # --- Time since last error (normalised to [0,1], cap at 20 trials) ---
    is_error  = (~df["is_correct"]).values
    since_err = np.zeros(n, dtype=float)
    count     = 20
    for j in range(n):
        if is_error[j]:
            count = 0
        else:
            count = min(count + 1, 20)
        since_err[j] = count
    df["time_since_last_error_norm"] = since_err / 20.0

    # --- RT trend: linear slope over last 5 trials (positive = slowing) ---
    rt_vals  = df["reaction_time"].values
    rt_trend = np.zeros(n, dtype=float)
    for j in range(n):
        window = rt_vals[max(0, j - 4): j + 1]
        if len(window) >= 2:
            x     = np.arange(len(window), dtype=float)
            slope = np.polyfit(x, window, 1)[0]
            rt_trend[j] = slope
    df["rt_trend"] = np.clip(rt_trend, -1.0, 1.0) / 2.0 + 0.5

    # --- Fatigue index: session time * cumulative error rate ---
    df["fatigue_index"] = np.clip(
        df["elapsed_time_norm"].values * df["cumulative_errors_norm"].values, 0.0, 1.0
    )

    return df


def add_target_label(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Add binary label: 1 if any error occurs within the next `horizon` trials.

    The label for trial i = OR(error at i+1, ..., error at i+horizon).

    Trials within `horizon` steps of the end are dropped (no future info).

    Args:
        df: Per-session engineered DataFrame (must be sorted by trial_idx).
        horizon: Number of future steps to look ahead.

    Returns:
        DataFrame with 'label' column, last `horizon` rows removed.
    """
    df = df.copy()
    errors = (~df["is_correct"]).astype(int).values
    n = len(errors)

    labels = np.zeros(n, dtype=int)
    for i in range(n - horizon):
        labels[i] = int(errors[i + 1: i + 1 + horizon].any())

    df["label"] = labels
    # Drop the last `horizon` rows (undefined label)
    df = df.iloc[: n - horizon].copy()
    return df


# ---------------------------------------------------------------------------
# Full pipeline per session
# ---------------------------------------------------------------------------

def preprocess_session(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Full preprocessing pipeline for a single session DataFrame.

    Args:
        df: Raw trial DataFrame for one session.
        horizon: Prediction horizon (K).

    Returns:
        Processed DataFrame ready for sequence extraction.
    """
    if len(df) < 20:
        logger.warning("Session has only %d trials — skipping.", len(df))
        return pd.DataFrame()

    df = engineer_features(df)
    df = add_target_label(df, horizon=horizon)
    return df


def preprocess_all_sessions(
    combined_df: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Preprocess all sessions in the combined DataFrame.

    Args:
        combined_df: DataFrame from load_all_sessions (multi-session).
        horizon: Prediction horizon.

    Returns:
        Concatenated processed DataFrame.
    """
    sessions = combined_df["session_id"].unique()
    processed = []

    for sid in sessions:
        sess_df = combined_df[combined_df["session_id"] == sid].copy()
        sess_df = sess_df.sort_values("trial_idx").reset_index(drop=True)
        p = preprocess_session(sess_df, horizon=horizon)
        if not p.empty:
            processed.append(p)

    if not processed:
        raise RuntimeError("No valid sessions after preprocessing.")

    result = pd.concat(processed, ignore_index=True)
    logger.info(
        "Preprocessing complete: %d sessions -> %d samples (horizon=%d)",
        len(processed), len(result), horizon,
    )
    return result
