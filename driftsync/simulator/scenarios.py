"""Named simulator presets for reliable showcase demos."""

from __future__ import annotations

from dataclasses import replace

from driftsync.configs import SimulatorConfig


SCENARIOS = {
    "fatigue_drift": {
        "session_name": "Fatigue drift",
        "fatigue_start_trial": 20,
        "noise_base": 0.08,
        "min_reaction_window": 0.25,
    },
    "sudden_overload": {
        "session_name": "Sudden overload",
        "fatigue_start_trial": 8,
        "noise_base": 0.12,
        "max_reaction_window": 2.2,
        "min_reaction_window": 0.22,
    },
    "stable_expert": {
        "session_name": "Stable expert",
        "fatigue_start_trial": 10_000,
        "noise_base": 0.015,
        "max_reaction_window": 3.4,
        "min_reaction_window": 0.55,
    },
    "recovery_after_break": {
        "session_name": "Recovery after break",
        "fatigue_start_trial": 18,
        "noise_base": 0.07,
        "max_reaction_window": 2.8,
        "min_reaction_window": 0.35,
    },
}


def apply_scenario(name: str, base: SimulatorConfig | None = None) -> SimulatorConfig:
    """Return a SimulatorConfig with a named showcase scenario applied."""
    if name not in SCENARIOS:
        choices = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown scenario '{name}'. Choose one of: {choices}")
    cfg = base or SimulatorConfig()
    return replace(cfg, **SCENARIOS[name])
