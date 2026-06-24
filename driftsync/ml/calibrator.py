"""
Calibration Engine
==================
Measures a user's normal performance during a short calibration phase and
stores the resulting baseline statistics. During the main session, live
features are compared against these values to detect deviation.

Calibration data is saved as JSON in the sessions/calibration/ directory.
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

CALIBRATION_DIR = Path("driftsync/sessions/calibration")
BASELINE_FILE   = CALIBRATION_DIR / "baseline_latest.json"


@dataclass
class BaselineStats:
    """Baseline performance statistics computed from calibration trials."""
    mean_rt: float
    std_rt: float
    median_rt: float
    q25_rt: float
    q75_rt: float
    accuracy: float
    error_rate: float
    mean_iti: float
    num_trials: int
    computed_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BaselineStats":
        return cls(**d)


class CalibrationEngine:
    """
    Computes and persists a baseline performance profile from calibration trials.

    Usage
    -----
        engine = CalibrationEngine()
        baseline = engine.compute_baseline(trial_list)
        engine.save(baseline)

        # Later:
        baseline = CalibrationEngine.load()
    """

    def compute_baseline(self, trials: list) -> BaselineStats:
        """
        Compute baseline stats from a list of trial dicts or Trial objects.
        Accepts both dict (from session JSON) and Trial dataclass instances.
        """
        def _get(t, key):
            if isinstance(t, dict):
                return t.get(key)
            return getattr(t, key, None)

        rts       = [_get(t, "reaction_time") for t in trials if _get(t, "reaction_time") is not None]
        correct   = [_get(t, "is_correct")    for t in trials if _get(t, "is_correct")    is not None]
        timestamps = [_get(t, "timestamp")    for t in trials if _get(t, "timestamp")     is not None]

        if not rts:
            return self._default_baseline(len(trials))

        rts_arr = np.array(rts, dtype=float)
        itis    = np.diff(timestamps) if len(timestamps) > 1 else np.array([1.0])

        return BaselineStats(
            mean_rt=float(np.mean(rts_arr)),
            std_rt=float(np.std(rts_arr) + 1e-6),
            median_rt=float(np.median(rts_arr)),
            q25_rt=float(np.percentile(rts_arr, 25)),
            q75_rt=float(np.percentile(rts_arr, 75)),
            accuracy=float(np.mean(correct)) if correct else 0.8,
            error_rate=float(1.0 - np.mean(correct)) if correct else 0.2,
            mean_iti=float(np.mean(itis)),
            num_trials=len(trials),
            computed_at=datetime.now().isoformat(),
        )

    def _default_baseline(self, n: int) -> BaselineStats:
        return BaselineStats(
            mean_rt=1.0, std_rt=0.3, median_rt=1.0,
            q25_rt=0.7, q75_rt=1.3,
            accuracy=0.8, error_rate=0.2,
            mean_iti=1.5, num_trials=n,
            computed_at=datetime.now().isoformat(),
        )

    def save(self, baseline: BaselineStats, path: Optional[Path] = None) -> None:
        path = path or BASELINE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(baseline.to_dict(), f, indent=2)

    @staticmethod
    def load(path: Optional[Path] = None) -> Optional[BaselineStats]:
        path = path or BASELINE_FILE
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return BaselineStats.from_dict(json.load(f))
        except Exception:
            return None

    @staticmethod
    def is_calibrated(path: Optional[Path] = None) -> bool:
        path = path or BASELINE_FILE
        return path.exists()
