# Changelog

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
