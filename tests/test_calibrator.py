import pytest

from driftsync.ml.calibrator import BaselineStats, CalibrationEngine


def test_compute_baseline_from_trial_dicts():
    trials = [
        {"reaction_time": 0.4, "is_correct": True, "timestamp": 0.0},
        {"reaction_time": 0.6, "is_correct": False, "timestamp": 1.0},
        {"reaction_time": 0.5, "is_correct": True, "timestamp": 2.5},
    ]

    baseline = CalibrationEngine().compute_baseline(trials)

    assert baseline.mean_rt == 0.5
    assert baseline.accuracy == pytest.approx(2 / 3)
    assert baseline.error_rate == pytest.approx(1 / 3)
    assert baseline.num_trials == 3


def test_compute_baseline_returns_default_without_reaction_times():
    baseline = CalibrationEngine().compute_baseline([{"is_correct": True}])

    assert baseline.mean_rt == 1.0
    assert baseline.accuracy == 0.8
    assert baseline.num_trials == 1


def test_baseline_round_trip_dict():
    baseline = CalibrationEngine().compute_baseline(
        [{"reaction_time": 0.5, "is_correct": True, "timestamp": 0.0}]
    )

    assert BaselineStats.from_dict(baseline.to_dict()) == baseline
