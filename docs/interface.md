# Desktop interface

Launch with `python launch.py`. The workspace links to live analysis, task recording,
model training, results and the field guide. No server or browser is needed.

## Controls

| Context | Controls |
| --- | --- |
| Workspace | Mouse; Tab / Shift+Tab and Enter for actions |
| Navigation | Alt+1 workspace, Alt+2 live, Alt+3 record, Alt+4 train, Alt+5 results, Alt+6 guide |
| Window | Resize or F11 to toggle fullscreen |
| Task | Click matching shape; Space to skip; Esc ends the session |
| Live setup | L / T select LSTM or Transformer |
| Results | Scroll; click a plot to inspect; Esc closes the plot first |
| Field guide | Left / Right pages; mouse wheel scrolls long content |

The logical canvas scales with the window and maps pointer input back to task
coordinates. The desktop workspace has a 960 × 600 minimum window. Task dimensions,
timing, hit radius and model features retain their existing definitions.

## Reading live analysis

- **Current drift state** describes predicted task-error risk, or reports that a
  model/history is unavailable. It is not a clinical measurement of cognition.
- **Error risk** is the model's probability of an error in the configured prediction
  horizon (five trials by default). Warm-up trials are not plotted as zero risk.
- **Model uncertainty** is the standard deviation across Monte Carlo dropout passes,
  shown in percentage points. It is not a confidence percentage.
- **Prediction confidence** explicitly says “Not calibrated”: this model does not
  produce a calibrated confidence score.
- **Risk over time** plots actual predictions. Hover to inspect a trial. The dashed
  line shows the configured warning threshold; the band is ±1 standard deviation,
  not a statistical confidence interval. The change label compares the oldest and
  newest predictions in the visible history.
- **Behavioural signals** uses the existing rule-based explainer. These observations
  are not causal feature attributions. Personal calibration supplies a behavioural
  baseline; it does not calibrate model probabilities.

Training can run while other workspace pages are open. “New run” becomes available
after completion or failure. Results retain model plots, comparison metrics, human
session details, folders and CSV export.

Canceling setup returns without writing task data or replacing prediction logs.
Canceling calibration before completion does not save an incomplete baseline.
After actual trials, ending the session preserves the existing save workflow.

## Visual verification

`python scripts/capture_ui.py` renders the actual UI to `build/ui-review/`, including
the workspace, all guide pages, training, results, task setup, live setup, calibration,
task, missing-model/warm-up states and smaller windows. The prediction preview is
visibly labeled and uses `tests/fixtures/realtime_log_sample.json`. It does not run
inference or invent chart values. Captures reflect local checkpoint availability.

`python -m pytest` includes headless interaction and lifecycle regressions.
`python -m driftsync.smoke` checks feature/model contracts. These checks do not claim
screen-reader support: Pygame is a drawn canvas rather than an accessibility tree.

Shared UI primitives are in `driftsync/ui.py`; live presentation is isolated in
`driftsync/realtime/presentation.py`. See [DESIGN.md](../DESIGN.md) for the visual system.
