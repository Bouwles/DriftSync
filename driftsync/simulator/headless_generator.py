"""
Headless Session Generator
===========================
Generates synthetic session data without a GUI for development, CI, and
quick-start demos. Simulates a realistic cognitive drift pattern:

  - Early trials: fast, accurate.
  - Middle trials: slight fatigue, minor errors creep in.
  - Late trials: increased error rate, slower reaction time.

Usage
-----
    python -m driftsync.simulator.headless_generator --sessions 5 --trials 200
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from datetime import datetime

from driftsync.configs import SimulatorConfig
from driftsync.simulator.task_engine import TaskEngine, Trial
from driftsync.utils import get_logger, set_seed

logger = get_logger(__name__)

SHAPES = ["CIRCLE", "SQUARE", "TRIANGLE"]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def generate_synthetic_session(
    cfg: SimulatorConfig,
    session_seed: int,
) -> Path:
    """
    Generate one synthetic session and save it to disk.

    Cognitive model:
        - Reaction time ~ N(mu_rt, sigma_rt) where mu_rt increases with fatigue.
        - Error probability ~ sigmoid(a * progress + b * recent_errors).
    """
    random.seed(session_seed)

    engine = TaskEngine(cfg)

    # Override session_id for reproducibility
    engine.session_id = f"synthetic_{session_seed:04d}"
    engine.session_data.session_id = engine.session_id
    engine.session_data.start_time = datetime.fromtimestamp(
        time.time() - session_seed
    ).isoformat()

    num_trials = cfg.num_trials

    for i in range(num_trials):
        progress = i / num_trials                        # [0, 1]

        # --- Compute fatigue parameters ---
        mu_rt    = 0.45 + 0.55 * progress               # gets slower
        sigma_rt = 0.08 + 0.12 * progress               # more variable

        # Error probability increases with progress and recent errors
        recent_err_rate = (
            sum(engine._error_history[-10:]) / 10
            if len(engine._error_history) >= 10
            else sum(engine._error_history) / max(1, len(engine._error_history))
        )
        # logit of error probability
        logit = -3.0 + 5.0 * progress + 2.0 * recent_err_rate
        p_error = _sigmoid(logit)

        # Generate stimulus
        stimulus = engine.next_stimulus()
        shape     = stimulus["shape"]
        rule      = stimulus["rule"]

        # Simulate player response
        rt = max(0.05, random.gauss(mu_rt, sigma_rt))
        rt = min(rt, stimulus["time_window"])

        # Correct response = click target, skip distractor
        should_click = (shape == rule)
        if random.random() < p_error:
            # make a mistake
            action = "skip" if should_click else "click"
        else:
            action = "click" if should_click else "skip"

        engine.record_trial(shape, action, rt)

    path = engine.save_session()
    logger.info(
        "Synthetic session %s saved: %d trials, seed=%d",
        engine.session_id, num_trials, session_seed,
    )
    return path


def generate_dataset(
    num_sessions: int = 10,
    cfg: SimulatorConfig | None = None,
    base_seed: int = 42,
) -> list[Path]:
    """
    Generate multiple synthetic sessions.

    Args:
        num_sessions: Number of independent sessions to generate.
        cfg: Simulator configuration.
        base_seed: Base random seed; each session uses base_seed + i.

    Returns:
        List of paths to saved session files.
    """
    cfg = cfg or SimulatorConfig()
    paths = []
    for i in range(num_sessions):
        p = generate_synthetic_session(cfg, session_seed=base_seed + i)
        paths.append(p)
    logger.info("Generated %d synthetic sessions in %s", num_sessions, cfg.data_dir)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic DriftSync sessions.")
    parser.add_argument("--sessions", type=int, default=10, help="Number of sessions")
    parser.add_argument("--trials", type=int, default=200, help="Trials per session")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--out", type=str, default="driftsync/data/raw", help="Output directory")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = SimulatorConfig(num_trials=args.trials, data_dir=args.out)
    generate_dataset(num_sessions=args.sessions, cfg=cfg, base_seed=args.seed)


if __name__ == "__main__":
    main()
