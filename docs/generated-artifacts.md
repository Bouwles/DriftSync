# Generated Artifacts

DriftSync can generate datasets, model checkpoints, logs, result plots, screenshots, and bundled executables. Most of those outputs should not be committed.

## Ignored Runtime Artifacts

These are reproducible or environment-specific and are ignored by `.gitignore`:

- `build/`
- `dist/`
- `driftsync/data/raw/`
- `driftsync/data/processed/`
- `driftsync/data/realtime_log.json`
- `driftsync/results/checkpoints/`
- `driftsync/results/showcase_bundle/`
- `driftsync/results/*.log`
- `driftsync/results/*.png`
- `__pycache__/`

## Committed Showcase Artifacts

README media is committed intentionally:

- `docs/assets/driftsync-overview.png`
- `docs/assets/driftsync-pipeline.png`
- `docs/assets/driftsync-results.png`
- `docs/assets/driftsync-live-demo.gif`

Regenerate these with:

```bash
python scripts/generate_showcase_assets.py
```

## Why This Split Exists

Checkpoints and raw data can be large and change on every experiment run. README media is small, stable, and useful for GitHub visitors who are deciding whether to run the project.
