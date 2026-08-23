from driftsync.configs import SimulatorConfig
from driftsync.simulator.task_engine import TaskEngine


def make_engine(num_trials: int = 10) -> TaskEngine:
    return TaskEngine(SimulatorConfig(num_trials=num_trials))


def test_clicking_target_shape_is_correct():
    engine = make_engine()
    target = engine.active_rule

    trial = engine.record_trial(target, "click", 0.42)

    assert trial.is_correct is True
    assert trial.action_taken == "click"
    assert engine.trial_count == 1


def test_skipping_non_target_shape_is_correct():
    engine = make_engine()
    distractor = next(shape for shape in engine.SHAPES if shape != engine.active_rule)

    trial = engine.record_trial(distractor, "skip", 0.31)

    assert trial.is_correct is True
    assert trial.stimulus_shape == distractor


def test_timeout_on_target_shape_is_incorrect():
    engine = make_engine()
    target = engine.active_rule

    trial = engine.record_trial(target, "timeout", 1.5)

    assert trial.is_correct is False
    assert engine.session_data.trials[-1] == trial
