import json
from pathlib import Path

from driftsync.showcase.replay import build_replay_timeline, export_replay_report


def write_session(path: Path) -> Path:
    payload = {
        "session_id": "demo",
        "session_name": "Fatigue drift demo",
        "trials": [
            {
                "trial_idx": 0,
                "timestamp": 0.0,
                "reaction_time": 0.42,
                "action_taken": "click",
                "target_shape": "CIRCLE",
                "stimulus_shape": "CIRCLE",
                "is_correct": True,
                "elapsed_session_time": 0.0,
                "prev_5_errors": [False, False, False, False, False],
            },
            {
                "trial_idx": 1,
                "timestamp": 1.2,
                "reaction_time": 0.76,
                "action_taken": "timeout",
                "target_shape": "CIRCLE",
                "stimulus_shape": "CIRCLE",
                "is_correct": False,
                "elapsed_session_time": 1.2,
                "prev_5_errors": [False, False, False, False, False],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_realtime_log(path: Path) -> Path:
    payload = [
        {"trial_idx": 0, "timestamp": 0.2, "probability": 0.44, "uncertainty": 0.08, "warning": False, "is_correct": True},
        {"trial_idx": 1, "timestamp": 0.9, "probability": 0.78, "uncertainty": 0.12, "warning": True, "is_correct": False},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_replay_timeline_merges_trials_and_predictions(tmp_path):
    timeline = build_replay_timeline(
        write_session(tmp_path / "session.json"),
        write_realtime_log(tmp_path / "realtime.json"),
    )

    assert timeline["session_id"] == "demo"
    assert timeline["summary"]["n_trials"] == 2
    assert timeline["summary"]["n_errors"] == 1
    assert timeline["summary"]["n_warnings"] == 1
    assert timeline["events"][1]["probability"] == 0.78
    assert timeline["events"][1]["risk_band"] == "high"
    assert timeline["events"][1]["explanations"]


def test_export_replay_report_writes_markdown_summary(tmp_path):
    timeline = build_replay_timeline(
        write_session(tmp_path / "session.json"),
        write_realtime_log(tmp_path / "realtime.json"),
    )

    out = export_replay_report(timeline, tmp_path / "report.md")

    text = out.read_text(encoding="utf-8")
    assert "# DriftSync Replay Report" in text
    assert "Warnings" in text
    assert "trial 1" in text
