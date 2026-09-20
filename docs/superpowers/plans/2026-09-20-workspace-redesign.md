# DriftSync workspace redesign

Goal: implement the supplied desktop UI brief and push the verified result to origin.

Architecture: preserve TaskEngine, feature engineering, checkpoints, inference, training,
calibration and exports. Share presentation tokens and a coordinate-mapped viewport
between the application shell and both task windows. Keep task coordinates and timing.

## Audit

- Python/Pygame shell: splash, home, seven learning pages, threaded pipeline with
  logs and epoch metrics, model results gallery/image inspection, session table,
  CSV export, task setup, calibration toggle, live model selection.
- Simulator: calibration intro/trials/summary, task intro/trials/feedback/summary.
- Live simulator: model loading, intro, timed task, MC-dropout risk/standard deviation,
  rule-based explanations, warning/error tracking, summary, JSON persistence.
- Working backend: 15 engineered features, 20-trial windows, five-trial horizon,
  LSTM/Transformer, sklearn baselines, synthetic scenarios and reproducible reports.
- Issues: hard-coded coordinates and missing resize handling; setup NameError;
  live model hitbox mismatch; warm-up zeros presented as predictions; no calibrated
  confidence output; training navigation discards worker reference; plots/metrics
  and long text can clip; failed launches claim success; learning copy says 11 features.
- Existing documentation assets are illustrations, not current UI captures.

## Implementation sequence

- [x] Regression tests for setup, model selection, viewport input, unavailable predictions.
- [x] Shared theme, typography, text wrapping, line chart and viewport primitives.
- [x] Redesign shell, home, learning, task/live setup, pipeline and results.
- [x] Redesign task/calibration/live workspace and session lifecycle feedback.
- [x] Run pytest/smoke, capture real rendered screens at desktop and smaller sizes,
      review, document design and usage.

Validation: dummy-SDL interaction tests, entire pytest suite, package smoke check,
screenshots from the real renderers using explicitly labeled fixture data when used.

Verification: 64 tests passed; smoke feature/model checks passed; visual review
confirmed the results-table fix; code review confirmed all listed lifecycle fixes.
Delivery: commit and push the verified files to origin/main.
