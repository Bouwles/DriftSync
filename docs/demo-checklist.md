# DriftSync Demo Checklist

Use this before sharing DriftSync as portfolio work.

## Code Health

- [ ] `python -m pytest`
- [ ] `python -m driftsync.smoke`
- [ ] `python scripts/generate_showcase_assets.py`
- [ ] Confirm `git status --short` only shows intentional changes.

## Demo Path

- [ ] Open `python launch.py`.
- [ ] Confirm the main menu loads.
- [ ] Run or show the Full Demo path.
- [ ] Open Results and confirm model/session evidence is visible.
- [ ] Launch Live Mode after a model has been trained, or explain the fallback path.

## README

- [ ] Hero screenshot renders on GitHub.
- [ ] Live GIF animates on GitHub.
- [ ] Quick Start commands are still accurate.
- [ ] Metrics are labeled as synthetic, not clinical validation.

## Talking Points

- DriftSync predicts future error risk from behavioral sequences.
- The most important metric is useful warning lead time, not only accuracy.
- Calibration makes explanations personal instead of absolute.
- Synthetic data is a limitation and the next serious milestone is real user data.
