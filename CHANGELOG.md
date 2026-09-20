# Changelog

## v2.2.0 — 2026-09-20 — Mathematics workspace & Windows release

- Added a dedicated Mathematics section with typeset equations, architecture diagrams,
  interactive worked examples and links to the implementing source files.
- Five topics cover prediction targets, behavioural features, LSTM/Transformer models,
  Monte Carlo dropout uncertainty, weighted loss and calibration error.
- Added Alt+7 navigation and an in-app version label; retained full keyboard controls
  and readable layouts at the minimum window size.
- Live mode now discovers checkpoints saved by named in-app training runs when no
  default checkpoint exists.
- Published a portable Windows x64 ZIP with bundled dependencies, a checksum and a
  frozen-executable smoke check. Personal sessions and local checkpoints are excluded.
- Includes the complete desktop workspace redesign described below.

## 2026-09-20 - Desktop workspace redesign

- Shared charcoal/teal design system and native desktop typography across setup,
  training, results, task, calibration and live analysis.
- Resizable canvases with mapped pointer input, keyboard focus and navigation.
- Separate live risk, uncertainty and unavailable calibrated confidence; real
  prediction history with threshold, standard deviation band and hover inspection.
- Fixed task-setup crash, model-selection hitboxes, plot dismissal, training restart,
  results access and canceled-session/calibration persistence.
- Added reproducible UI captures and headless UI regression tests.

## 2026-08-25 - Showcase v3

- Added replay timeline generation that merges task sessions with realtime prediction logs.
- Added Markdown replay reports for warnings, actual errors, uncertainty, and explanation notes.
- Added named simulator scenarios for repeatable portfolio demos.
- Added a one-command showcase bundle generator.
- Added tests for replay reports, scenario presets, and bundle generation.

## 2026-08-23 - Showcase Polish

- Added pytest coverage for task logic, preprocessing, sequence extraction, metrics, realtime inference, checkpoint resolution, launcher behavior, and smoke checks.
- Added GitHub Actions CI for tests and smoke verification.
- Added README screenshots and a live inference GIF under `docs/assets/`.
- Rebuilt the README as a portfolio-grade project page.
- Fixed stale feature configuration defaults for the v2 15-feature pipeline.
- Fixed realtime inference logs so saved JSON contains only prediction events.
- Added a checkpoint resolver with clearer missing-model errors.
- Made launcher dependency installation opt-in via `DRIFTSYNC_AUTO_INSTALL=1`.
- Added developer commands, smoke checks, and generated-asset tooling.

## v2.0

- Added calibration, baseline models, explainability, and lead-time tracking.
- Added Random Forest, Logistic Regression, and threshold fallback model paths.
- Expanded feature engineering from 11 to 15 behavioral features.
