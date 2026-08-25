# Scenario Presets

Scenario presets make live demos more repeatable. They change simulator difficulty and drift pressure without changing the model pipeline.

| Scenario | Purpose |
| --- | --- |
| `fatigue_drift` | Gradual fatigue-style decline for a classic demo arc. |
| `sudden_overload` | Fast risk escalation for short recordings. |
| `stable_expert` | Low-noise control run with fewer warnings. |
| `recovery_after_break` | Moderate drift setup for explaining recovery and thresholds. |

Run one directly:

```bash
python -m driftsync.realtime.live_simulator --scenario fatigue_drift --trials 80
python -m driftsync.realtime.live_simulator --scenario sudden_overload --trials 40
```

The same presets are available from Python:

```python
from driftsync.configs import SimulatorConfig
from driftsync.simulator.scenarios import apply_scenario

cfg = apply_scenario("stable_expert", SimulatorConfig(num_trials=60))
```
