"""
Run the GUI Simulator
=====================
Entry point for the interactive human task simulator.

Usage
-----
    python -m driftsync.simulator.run_gui
    python -m driftsync.simulator.run_gui --trials 300
"""

import argparse
from driftsync.configs import SimulatorConfig
from driftsync.simulator.gui import DriftSimulator
from driftsync.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="DriftSync — Human Task Simulator")
    parser.add_argument("--trials", type=int, default=200, help="Number of trials")
    parser.add_argument("--out", type=str, default="driftsync/data/raw", help="Output directory")
    args = parser.parse_args()

    cfg = SimulatorConfig(num_trials=args.trials, data_dir=args.out)
    sim = DriftSimulator(cfg)
    session_path = sim.run()
    logger.info("Session data saved to: %s", session_path)


if __name__ == "__main__":
    main()
