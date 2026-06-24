"""
Real-Time Inference Engine
===========================
Maintains a rolling feature buffer of length `seq_len`, accepts a new
trial observation at each timestep, and outputs:

    - P(error in next K steps)         — mean probability
    - Uncertainty                      — std of MC-Dropout samples
    - Warning flag                     — whether thresholds are exceeded

Designed to be decoupled from any GUI so it can be unit-tested.
"""

from __future__ import annotations


from collections import deque
from pathlib import Path
from typing import Optional, Tuple
import json
import time

import numpy as np
import torch

from driftsync.configs import RealtimeConfig, DataConfig, CONFIG
from driftsync.models import build_model
from driftsync.data.preprocessing import engineer_features
from driftsync.utils import get_logger, get_device

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature names (must match preprocessing.FEATURE_COLS order)
# ---------------------------------------------------------------------------

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

NUM_FEATURES = len(FEATURE_COLS)


class RealtimeInferenceEngine:
    """
    Streaming inference engine for cognitive drift prediction.

    Usage
    -----
        engine = RealtimeInferenceEngine(rt_cfg, data_cfg)
        engine.load_model()

        # In each trial callback:
        proba, uncertainty, warning = engine.update(trial)

    Args:
        rt_cfg: RealtimeConfig.
        data_cfg: DataConfig (for seq_len).
    """

    def __init__(
        self,
        rt_cfg: RealtimeConfig | None = None,
        data_cfg: DataConfig | None = None,
        baseline=None,
    ):
        self.rt_cfg   = rt_cfg   or CONFIG.realtime
        self.data_cfg = data_cfg or CONFIG.data
        self.baseline = baseline  # optional BaselineStats for deviation features

        self._seq_len = self.data_cfg.sequence_length
        self._device  = get_device(CONFIG.training.device)

        self._raw_buffer: deque = deque(maxlen=max(self._seq_len + 20, 50))
        self._feat_buffer: deque = deque(maxlen=self._seq_len)

        self._session_start: float  = time.time()
        self._trial_idx: int        = 0
        self._prev_timestamp: float = time.time()
        self._error_history: deque  = deque(maxlen=20)
        self._correct_streak: int   = 0
        self._error_streak:   int   = 0

        # Extra state for new features
        self._rt_history: deque          = deque(maxlen=10)
        self._trials_since_last_error: int = 20

        self.model = None
        self._mc_samples = 30

        self._log: list = []
        self._log_file = Path(self.rt_cfg.log_file)
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, model_type: Optional[str] = None) -> None:
        """
        Load the trained model from the best checkpoint.

        Args:
            model_type: "lstm" or "transformer". Defaults to rt_cfg.model_type.
        """
        model_type = model_type or self.rt_cfg.model_type
        ckpt_dir   = Path(self.rt_cfg.checkpoint_dir)

        name_map = {
            "lstm":        "lstm_best.pt",
            "transformer": "transformer_best.pt",
        }
        ckpt_path = ckpt_dir / name_map[model_type]

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {ckpt_path}. "
                "Run training first: python -m driftsync.training.pipeline --model {model_type}"
            )

        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)
        model_cfg = CONFIG.lstm if model_type == "lstm" else CONFIG.transformer
        model_cfg.input_dim = NUM_FEATURES  # ensure dimension matches

        self.model = build_model(model_type, model_cfg)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self._device)
        self.model.eval()
        logger.info("Inference engine loaded %s (epoch %d)", model_type.upper(), ckpt.get("epoch", "?"))

    # ------------------------------------------------------------------
    # Trial update
    # ------------------------------------------------------------------

    def update(
        self,
        reaction_time: float,
        is_correct: bool,
        stimulus_shape: str,
        target_shape: str,
        action: str,
    ) -> Tuple[float, float, bool]:
        """
        Accept one new trial observation and compute live predictions.

        Args:
            reaction_time: Seconds from stimulus to response.
            is_correct: Whether player's response was correct.
            stimulus_shape: Shape displayed.
            target_shape: Shape player should have clicked.
            action: "click", "skip", or "timeout".

        Returns:
            (mean_probability, uncertainty, warning_flag)
        """
        now = time.time()
        iti = now - self._prev_timestamp
        self._prev_timestamp = now
        elapsed = now - self._session_start

        if is_correct:
            self._correct_streak += 1
            self._error_streak    = 0
            self._trials_since_last_error = min(self._trials_since_last_error + 1, 20)
        else:
            self._error_streak   += 1
            self._correct_streak  = 0
            self._trials_since_last_error = 0
        self._error_history.append(int(not is_correct))
        self._rt_history.append(reaction_time)

        # Compute streaming features
        feats = self._compute_features(
            reaction_time=reaction_time,
            is_correct=is_correct,
            elapsed=elapsed,
            iti=iti,
            stimulus_shape=stimulus_shape,
            target_shape=target_shape,
            action=action,
        )

        self._feat_buffer.append(feats)
        self._trial_idx += 1

        # Insufficient history
        if len(self._feat_buffer) < self._seq_len:
            return 0.0, 0.0, False

        # Build tensor: (1, seq_len, NUM_FEATURES)
        seq = np.stack(list(self._feat_buffer), axis=0)   # (L, F)
        x   = torch.from_numpy(seq).float().unsqueeze(0).to(self._device)

        # MC-Dropout forward passes
        mean_p, uncertainty = self.model.mc_dropout_predict(x, n_samples=self._mc_samples)
        mean_p_val   = float(mean_p.item())
        uncertainty_val = float(uncertainty.item())

        warning = (
            mean_p_val   > self.rt_cfg.warning_threshold
            or uncertainty_val > self.rt_cfg.uncertainty_threshold
        )

        # Log event
        self._log.append({
            "trial_idx":    self._trial_idx,
            "timestamp":    now,
            "probability":  mean_p_val,
            "uncertainty":  uncertainty_val,
            "warning":      warning,
            "is_correct":   is_correct,
        })

        return mean_p_val, uncertainty_val, warning

    # ------------------------------------------------------------------
    # Feature computation (streaming, no pandas dependency)
    # ------------------------------------------------------------------

    def _compute_features(
        self,
        reaction_time: float,
        is_correct: bool,
        elapsed: float,
        iti: float,
        stimulus_shape: str,
        target_shape: str,
        action: str,
    ) -> np.ndarray:
        """
        Compute the feature vector for one trial using running statistics.
        """
        # Reaction time: normalise using historical median / IQR if available
        all_rts = [e.get("rt", reaction_time) for e in self._log[-20:]]
        all_rts.append(reaction_time)
        median_rt = float(np.median(all_rts))
        iqr_rt    = float(np.percentile(all_rts, 75) - np.percentile(all_rts, 25)) + 1e-6
        rt_norm   = float(np.clip((reaction_time - median_rt) / iqr_rt / 6.0 + 0.5, 0, 1))

        correctness = float(is_correct)

        # Elapsed time: rough normalisation (assume ~5 min session)
        elapsed_norm = min(elapsed / 300.0, 1.0)

        # Rolling error rates
        err_hist = list(self._error_history)
        err_rate_5  = float(np.mean(err_hist[-5:]  if len(err_hist) >= 5  else err_hist)) if err_hist else 0.0
        err_rate_10 = float(np.mean(err_hist[-10:] if len(err_hist) >= 10 else err_hist)) if err_hist else 0.0

        # ITI normalised (clip to 5s)
        iti_norm = float(np.clip(iti / 5.0, 0.0, 1.0))

        # Cumulative error rate
        cumulative_err = float(sum(err_hist) / max(self._trial_idx, 1))

        # Streaks (normalised by 20)
        streak_c = float(min(self._correct_streak / 20.0, 1.0))
        streak_i = float(min(self._error_streak  / 20.0, 1.0))

        # Target match
        target_match = float(stimulus_shape == target_shape)
        action_click = float(action == "click")

        self._log.append({"rt": reaction_time})

        # --- Rolling RT variance (inconsistency score) ---
        rts = list(self._rt_history)
        if len(rts) >= 2:
            rt_std = float(np.std(rts))
            iqr    = float(np.percentile(rts, 75) - np.percentile(rts, 25)) + 1e-6
            rt_var_norm = float(np.clip(rt_std / iqr, 0.0, 3.0) / 3.0)
        else:
            rt_var_norm = 0.0

        # --- Time since last error (normalised, cap 20) ---
        since_err_norm = float(min(self._trials_since_last_error, 20) / 20.0)

        # --- RT trend over last 5 trials ---
        if len(rts) >= 3:
            x     = np.arange(len(rts), dtype=float)
            slope = float(np.polyfit(x, rts, 1)[0])
            rt_trend_norm = float(np.clip(slope, -1.0, 1.0) / 2.0 + 0.5)
        else:
            rt_trend_norm = 0.5

        # --- Fatigue index ---
        fatigue = float(np.clip(elapsed_norm * cumulative_err, 0.0, 1.0))

        return np.array([
            rt_norm, correctness, elapsed_norm,
            err_rate_5, err_rate_10, iti_norm,
            cumulative_err, streak_c, streak_i,
            target_match, action_click,
            rt_var_norm, since_err_norm, rt_trend_norm, fatigue,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_log(self) -> None:
        """Flush inference log to disk."""
        with open(self._log_file, "w") as f:
            json.dump(self._log, f, indent=2)
        logger.info("Inference log saved -> %s (%d events)", self._log_file, len(self._log))

    def get_live_stats(self) -> dict:
        """Return current rolling performance stats for explainability."""
        err_hist = list(self._error_history)
        rts      = list(self._rt_history)
        return {
            "mean_rt_recent":      float(np.mean(rts)) if rts else 1.0,
            "rolling_acc_10":      float(1.0 - np.mean(err_hist[-10:])) if len(err_hist) >= 10 else 0.8,
            "rolling_err_rate_5":  float(np.mean(err_hist[-5:]))  if len(err_hist) >= 5  else 0.0,
            "rolling_err_rate_10": float(np.mean(err_hist[-10:])) if len(err_hist) >= 10 else 0.0,
            "error_streak":        self._error_streak,
            "rt_variance":         float(np.std(rts) / (np.percentile(rts, 75) - np.percentile(rts, 25) + 1e-6)) if len(rts) >= 2 else 0.0,
            "rt_trend":            float(np.polyfit(np.arange(len(rts)), rts, 1)[0]) if len(rts) >= 3 else 0.0,
            "fatigue_index":       float(min(self._trial_idx / 200.0, 1.0) * (sum(err_hist) / max(self._trial_idx, 1))),
        }

    def reset(self) -> None:
        """Reset internal state for a new session."""
        self._feat_buffer.clear()
        self._raw_buffer.clear()
        self._session_start           = time.time()
        self._trial_idx               = 0
        self._prev_timestamp          = time.time()
        self._error_history.clear()
        self._correct_streak          = 0
        self._error_streak            = 0
        self._rt_history.clear()
        self._trials_since_last_error = 20
        self._log.clear()
