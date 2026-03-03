"""
DriftSync
=========
Real-Time Neural Cognitive Drift Prediction System.

Quick-start
-----------
    # 1. Generate synthetic training data
    python -m driftsync.simulator.headless_generator --sessions 20

    # 2. Train LSTM
    python -m driftsync.training.pipeline --model lstm

    # 3. Train Transformer
    python -m driftsync.training.pipeline --model transformer

    # 4. Compare models and generate all plots
    python -m driftsync.evaluation.compare

    # 5. Run live inference simulator
    python -m driftsync.realtime.live_simulator --model lstm

    # 6. Play the GUI simulator (collects real human data)
    python -m driftsync.simulator.run_gui
"""

__version__ = "1.0.0"
__author__  = "DriftSync Research"
