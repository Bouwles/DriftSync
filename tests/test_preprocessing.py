import numpy as np
import pandas as pd

from driftsync.data.dataset import FEATURE_COLS
from driftsync.data.preprocessing import add_target_label, engineer_features


def sample_trials() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["s1"] * 8,
            "trial_idx": list(range(8)),
            "timestamp": [0.0, 0.7, 1.5, 2.6, 4.0, 5.8, 7.7, 9.9],
            "reaction_time": [0.50, 0.48, 0.62, 0.70, 0.91, 0.88, 1.05, 1.20],
            "action_taken": ["click", "skip", "click", "timeout", "click", "skip", "click", "skip"],
            "target_shape": ["CIRCLE", "CIRCLE", "SQUARE", "SQUARE", "TRIANGLE", "TRIANGLE", "CIRCLE", "CIRCLE"],
            "stimulus_shape": ["CIRCLE", "SQUARE", "SQUARE", "SQUARE", "TRIANGLE", "CIRCLE", "SQUARE", "TRIANGLE"],
            "is_correct": [True, True, True, False, True, True, False, True],
            "elapsed_session_time": [0.0, 0.7, 1.5, 2.6, 4.0, 5.8, 7.7, 9.9],
        }
    )


def test_engineer_features_emits_model_feature_columns():
    engineered = engineer_features(sample_trials())

    assert list(FEATURE_COLS) == [
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
        "rolling_rt_variance",
        "time_since_last_error_norm",
        "rt_trend",
        "fatigue_index",
    ]
    assert set(FEATURE_COLS).issubset(engineered.columns)
    assert engineered[FEATURE_COLS].to_numpy().dtype == np.float64


def test_engineered_features_are_bounded():
    engineered = engineer_features(sample_trials())

    for column in FEATURE_COLS:
        assert engineered[column].between(0.0, 1.0).all(), column


def test_engineered_features_capture_expected_behavior():
    engineered = engineer_features(sample_trials())

    assert engineered.loc[3, "rolling_error_rate_5"] == 0.25
    assert engineered.loc[3, "streak_incorrect"] == 0.05
    assert engineered.loc[6, "target_match"] == 0.0
    assert engineered.loc[4, "action_click"] == 1.0
