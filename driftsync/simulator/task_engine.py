"""
Task Engine
===========
Core logic for the cognitive drift simulator task.

The task is a rapid-fire reaction + classification game:
- A stimulus appears at a random screen position.
- The stimulus is one of three shapes: CIRCLE, SQUARE, TRIANGLE.
- A rule is displayed: e.g. "Click CIRCLES only".
- The player must click the correct shape within a time window.
- Reaction time, correctness, and metadata are recorded per trial.

Fatigue is modelled by progressively tightening the time window and
adding mild spatial noise to stimulus placement after trial N.
"""

import time
import random
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from driftsync.configs import SimulatorConfig
from driftsync.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    """Single trial record."""
    trial_idx: int
    timestamp: float              # wall-clock seconds since session start
    reaction_time: float          # seconds from stimulus appearance to click
    action_taken: str             # "click" or "skip"
    target_shape: str             # expected correct shape
    stimulus_shape: str           # shape that appeared on screen
    is_correct: bool
    elapsed_session_time: float   # total session time elapsed (fatigue proxy)
    prev_5_errors: List[bool]     # error history of previous 5 trials


@dataclass
class SessionData:
    """Full session record."""
    session_id: str
    start_time: str
    config: dict
    trials: List[Trial] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["trials"] = [asdict(t) for t in self.trials]
        return d


# ---------------------------------------------------------------------------
# Task Engine (non-GUI core logic)
# ---------------------------------------------------------------------------

class TaskEngine:
    """
    Manages trial sequencing and state for the cognitive task.

    Decoupled from rendering so it can be tested headlessly.
    """

    SHAPES = ["CIRCLE", "SQUARE", "TRIANGLE"]

    def __init__(self, cfg: SimulatorConfig):
        self.cfg = cfg
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_data = SessionData(
            session_id=self.session_id,
            start_time=datetime.now().isoformat(),
            config=cfg.__dict__,
        )
        self._trial_count = 0
        self._session_start = time.time()
        self._error_history: List[bool] = []   # True = error
        self._active_rule: str = random.choice(self.SHAPES)
        self._rule_change_interval = 20         # change rule every N trials

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def trial_count(self) -> int:
        return self._trial_count

    @property
    def active_rule(self) -> str:
        """Shape the player must click."""
        return self._active_rule

    @property
    def is_finished(self) -> bool:
        return self._trial_count >= self.cfg.num_trials

    # ------------------------------------------------------------------
    # Trial generation
    # ------------------------------------------------------------------

    def next_stimulus(self) -> dict:
        """
        Generate the next trial stimulus.

        Returns a dict describing:
            shape, position, time_window (deadline to respond)
        """
        # Refresh rule periodically
        if self._trial_count > 0 and self._trial_count % self._rule_change_interval == 0:
            # Pick a different shape
            candidates = [s for s in self.SHAPES if s != self._active_rule]
            self._active_rule = random.choice(candidates)

        # Determine which shape appears (target appears ~60% of trials)
        if random.random() < 0.60:
            shape = self._active_rule
        else:
            shape = random.choice([s for s in self.SHAPES if s != self._active_rule])

        # Position with fatigue-induced spatial drift
        margin = 80
        fatigue_noise = self._fatigue_noise()
        x = random.randint(margin, self.cfg.window_width - margin)
        y = random.randint(margin + 60, self.cfg.window_height - margin)
        x = int(x + random.gauss(0, fatigue_noise))
        y = int(y + random.gauss(0, fatigue_noise))
        x = max(margin, min(self.cfg.window_width - margin, x))
        y = max(margin + 60, min(self.cfg.window_height - margin, y))

        # Time window shrinks with fatigue
        max_window = self.cfg.max_reaction_window - (
            self._trial_count / self.cfg.num_trials
        ) * (self.cfg.max_reaction_window - self.cfg.min_reaction_window) * 0.5

        return {
            "shape": shape,
            "x": x,
            "y": y,
            "time_window": max_window,
            "rule": self._active_rule,
        }

    def record_trial(
        self,
        stimulus_shape: str,
        action: str,           # "click" | "skip" | "timeout"
        reaction_time: float,
    ) -> Trial:
        """
        Record the outcome of a completed trial.

        Args:
            stimulus_shape: The shape that was shown.
            action: What the player did.
            reaction_time: Seconds from appearance to response.

        Returns:
            Completed Trial record.
        """
        # A click on the target shape is correct; a skip on a non-target is also correct.
        if action == "click":
            is_correct = (stimulus_shape == self._active_rule)
        elif action in ("skip", "timeout"):
            is_correct = (stimulus_shape != self._active_rule)
        else:
            is_correct = False

        prev_5 = list(self._error_history[-5:])
        while len(prev_5) < 5:
            prev_5.insert(0, False)

        trial = Trial(
            trial_idx=self._trial_count,
            timestamp=time.time() - self._session_start,
            reaction_time=reaction_time,
            action_taken=action,
            target_shape=self._active_rule,
            stimulus_shape=stimulus_shape,
            is_correct=is_correct,
            elapsed_session_time=time.time() - self._session_start,
            prev_5_errors=prev_5,
        )

        self.session_data.trials.append(trial)
        self._error_history.append(not is_correct)
        self._trial_count += 1
        return trial

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fatigue_noise(self) -> float:
        """
        Spatial noise that increases as a function of trial index.
        Simulates motor imprecision due to fatigue.
        """
        progress = self._trial_count / max(1, self.cfg.num_trials)
        return self.cfg.noise_base * self.cfg.window_width * progress * 0.3

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_session(self) -> Path:
        """Save session JSON to disk."""
        out_dir = Path(self.cfg.data_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"session_{self.session_id}.json"
        with open(fname, "w") as f:
            json.dump(self.session_data.to_dict(), f, indent=2)
        logger.info("Session saved -> %s  (%d trials)", fname, self._trial_count)
        return fname
