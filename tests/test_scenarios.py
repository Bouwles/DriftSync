from driftsync.configs import SimulatorConfig
from driftsync.realtime.live_simulator import build_simulator_config
from driftsync.simulator.scenarios import SCENARIOS, apply_scenario


def test_scenario_presets_include_showcase_modes():
    assert {"fatigue_drift", "sudden_overload", "stable_expert", "recovery_after_break"} <= set(SCENARIOS)


def test_apply_scenario_returns_named_config_without_mutating_base():
    base = SimulatorConfig(num_trials=50, max_reaction_window=3.0, noise_base=0.05)

    cfg = apply_scenario("sudden_overload", base)

    assert cfg.session_name == "Sudden overload"
    assert cfg.num_trials == 50
    assert cfg.max_reaction_window < base.max_reaction_window
    assert cfg.noise_base > base.noise_base


def test_live_simulator_config_builder_applies_scenario_and_trial_count():
    cfg = build_simulator_config(num_trials=24, scenario="stable_expert")

    assert cfg.num_trials == 24
    assert cfg.session_name == "Stable expert"
    assert cfg.noise_base < SimulatorConfig().noise_base
