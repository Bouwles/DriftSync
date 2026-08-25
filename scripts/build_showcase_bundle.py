"""Build a compact DriftSync showcase bundle."""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from driftsync.showcase.replay import build_replay_timeline, export_replay_report


def build_showcase_bundle(output_dir: str | Path = "driftsync/results/showcase_bundle", generate_assets: bool = True) -> Path:
    """Create a shareable bundle with sample replay data and reports."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    session_path = out / "sample-session.json"
    realtime_path = out / "sample-realtime-log.json"
    _write_sample_session(session_path)
    _write_sample_realtime_log(realtime_path)

    timeline = build_replay_timeline(session_path, realtime_path)
    timeline_path = out / "replay-timeline.json"
    timeline_path.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    report_path = export_replay_report(timeline, out / "replay-report.md")

    if generate_assets:
        from scripts.generate_showcase_assets import main as generate_showcase_assets

        generate_showcase_assets()

    index = [
        "# DriftSync Showcase Bundle",
        "",
        "This bundle demonstrates the replay/report layer for a cognitive drift session.",
        "",
        "## Files",
        "",
        f"- [Replay Report]({report_path.name})",
        f"- [Replay Timeline]({timeline_path.name})",
        f"- [Sample Session]({session_path.name})",
        f"- [Realtime Log]({realtime_path.name})",
    ]
    (out / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return out


def _write_sample_session(path: Path) -> None:
    trials = []
    for idx in range(12):
        is_error = idx in {7, 10}
        trials.append(
            {
                "trial_idx": idx,
                "timestamp": idx * 1.15,
                "reaction_time": round(0.36 + idx * 0.035 + (0.20 if is_error else 0), 3),
                "action_taken": "timeout" if is_error else "click",
                "target_shape": "CIRCLE",
                "stimulus_shape": "CIRCLE" if idx % 3 else "SQUARE",
                "is_correct": not is_error,
                "elapsed_session_time": idx * 1.15,
                "prev_5_errors": [False, False, idx > 5, False, is_error],
            }
        )
    path.write_text(
        json.dumps(
            {
                "session_id": "showcase_sample",
                "session_name": "Fatigue drift sample",
                "trials": trials,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_sample_realtime_log(path: Path) -> None:
    rows = []
    for idx in range(12):
        probability = min(0.92, 0.18 + idx * 0.065)
        rows.append(
            {
                "trial_idx": idx,
                "timestamp": idx * 1.15 + 0.2,
                "probability": round(probability, 3),
                "uncertainty": round(0.06 + max(0, idx - 7) * 0.025, 3),
                "warning": probability >= 0.65,
                "is_correct": idx not in {7, 10},
            }
        )
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a DriftSync showcase bundle.")
    parser.add_argument("--output-dir", default="driftsync/results/showcase_bundle")
    parser.add_argument("--no-assets", action="store_true", help="Skip regenerating README image/GIF assets")
    args = parser.parse_args()

    out = build_showcase_bundle(args.output_dir, generate_assets=not args.no_assets)
    print(f"Showcase bundle written to {out}")


if __name__ == "__main__":
    main()
