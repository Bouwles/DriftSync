# Replay Reports

DriftSync can turn a saved task session plus realtime prediction log into a replay timeline and Markdown report.

The replay layer merges:

- trial index, reaction time, action, target shape, stimulus shape, and correctness;
- predicted error probability;
- Monte Carlo Dropout uncertainty;
- warning state;
- simple explanation notes for high-risk or error trials.

## Build A Showcase Bundle

```bash
python scripts/build_showcase_bundle.py
# or
make showcase-bundle
```

Output is written to `driftsync/results/showcase_bundle/`:

- `index.md`
- `sample-session.json`
- `sample-realtime-log.json`
- `replay-timeline.json`
- `replay-report.md`

The generated report is meant for portfolio review: it shows the model's warning moments, actual errors, peak risk, and why the warning fired.

## Python API

```python
from driftsync.showcase import build_replay_timeline, export_replay_report

timeline = build_replay_timeline("session.json", "realtime-log.json")
export_replay_report(timeline, "replay-report.md")
```
