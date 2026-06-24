"""
DriftSync Configuration
=======================
Central configuration for all system components.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SimulatorConfig:
    """Configuration for the human task simulator."""
    window_width: int = 900
    window_height: int = 650
    fps: int = 60
    target_radius: int = 30
    min_reaction_window: float = 0.3   # seconds
    max_reaction_window: float = 3.0   # seconds
    num_trials: int = 200
    data_dir: str = "driftsync/data/raw"
    session_file: str = "session_{timestamp}.json"

    # Session identity
    session_name: str = ""             # user-supplied label (optional)

    # Fatigue / drift simulation parameters
    fatigue_start_trial: int = 50      # drift increases after this
    noise_base: float = 0.05           # base noise on reaction times


@dataclass
class DataConfig:
    """Configuration for data pipeline."""
    raw_data_dir: str = "driftsync/data/raw"
    processed_data_dir: str = "driftsync/data/processed"
    sequence_length: int = 20          # input window size
    prediction_horizon: int = 5        # K: steps ahead to predict
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    min_samples: int = 100             # minimum trials needed
    feature_columns: List[str] = field(default_factory=lambda: [
        "reaction_time_norm",
        "correctness",
        "elapsed_time_norm",
        "rolling_error_rate",
        "inter_trial_interval_norm",
        "cumulative_errors_norm",
        "streak_correct",
        "streak_incorrect",
    ])


@dataclass
class LSTMConfig:
    """Configuration for LSTM model."""
    input_dim: int = 8
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    bidirectional: bool = False
    output_dim: int = 1

    # Monte Carlo Dropout
    mc_samples: int = 50


@dataclass
class TransformerConfig:
    """Configuration for Transformer model."""
    input_dim: int = 8
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_seq_len: int = 512
    output_dim: int = 1

    # Monte Carlo Dropout
    mc_samples: int = 50


@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    batch_size: int = 64
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Scheduler
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    # Early stopping
    early_stop_patience: int = 15
    early_stop_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: str = "driftsync/results/checkpoints"
    save_best_only: bool = True

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "auto"   # "auto", "cpu", "cuda", "mps"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and experiment comparison."""
    results_dir: str = "driftsync/results"
    n_bootstrap: int = 1000            # bootstrap iterations for CIs
    calibration_bins: int = 10
    threshold: float = 0.5             # decision threshold


@dataclass
class CalibrationConfig:
    """Configuration for the pre-session calibration phase."""
    num_trials: int = 25
    calibration_dir: str = "driftsync/sessions/calibration"
    baseline_file: str = "driftsync/sessions/calibration/baseline_latest.json"


@dataclass
class RealtimeConfig:
    """Configuration for real-time inference."""
    model_type: str = "lstm"           # "lstm" or "transformer"
    checkpoint_dir: str = "driftsync/results/checkpoints"
    warning_threshold: float = 0.65    # P(error) threshold for warning
    uncertainty_threshold: float = 0.20
    display_history: int = 30          # timesteps shown in live plot
    log_file: str = "driftsync/data/realtime_log.json"


@dataclass
class DriftSyncConfig:
    """Master configuration aggregating all sub-configs."""
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)


# Default global config instance
CONFIG = DriftSyncConfig()
