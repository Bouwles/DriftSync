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


def test_next_stimulus_stays_inside_playable_bounds():
    cfg = SimulatorConfig(window_width=900, window_height=650, target_radius=30, num_trials=40)
    engine = TaskEngine(cfg)

    for _ in range(25):
        stimulus = engine.next_stimulus()
        assert 80 <= stimulus["x"] <= cfg.window_width - 80
        assert 140 <= stimulus["y"] <= cfg.window_height - 80
        assert cfg.min_reaction_window <= stimulus["time_window"] <= cfg.max_reaction_window
        engine.record_trial(stimulus["shape"], "skip", 0.2)


def test_rule_changes_to_a_different_shape_at_interval():
    engine = make_engine(num_trials=45)
    first_rule = engine.active_rule

    for _ in range(20):
        engine.record_trial(first_rule, "click", 0.2)

    stimulus = engine.next_stimulus()

    assert stimulus["rule"] != first_rule
