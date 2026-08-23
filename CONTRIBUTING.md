# Contributing

Thanks for taking a look at DriftSync. The project is intentionally small enough to run locally, but it has several moving pieces: simulator, feature pipeline, models, realtime inference, and documentation assets.

## Local Setup

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
python -m driftsync.smoke
```

## Development Rules

- Keep generated data, checkpoints, logs, and PyInstaller output out of git.
- Add or update tests when changing simulator logic, feature engineering, metrics, checkpoint loading, or realtime inference.
- Keep README media in `docs/assets/` and regenerate it with:

```bash
python scripts/generate_showcase_assets.py
```

## Before Opening A PR

Run:

```bash
python -m pytest
python -m driftsync.smoke
```

For README or asset changes, also run:

```bash
python scripts/generate_showcase_assets.py
```
