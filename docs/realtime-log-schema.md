# Realtime Inference Log Schema

DriftSync writes live inference events to `driftsync/data/realtime_log.json` when live mode is run with a trained model.

The file contains a JSON array. Each object represents one prediction event emitted after the rolling sequence buffer is full.

| Field | Type | Meaning |
| --- | --- | --- |
| `trial_idx` | integer | One-based trial number at the time the prediction was made. |
| `timestamp` | number | Wall-clock UNIX timestamp for the event. |
| `probability` | number | Mean predicted probability of an error in the configured future horizon. |
| `uncertainty` | number | Standard deviation across Monte Carlo dropout samples. |
| `warning` | boolean | `true` when probability or uncertainty crosses the configured warning threshold. |
| `is_correct` | boolean | Whether the just-completed trial was correct. |

Example:

```json
[
  {
    "trial_idx": 20,
    "timestamp": 1787482983.91,
    "probability": 0.72,
    "uncertainty": 0.04,
    "warning": true,
    "is_correct": false
  }
]
```

Raw helper values used for feature engineering are intentionally not written to this log. The log is designed for analysis, demos, and portfolio inspection.
