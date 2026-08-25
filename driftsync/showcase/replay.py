"""Replay timeline and report generation for DriftSync sessions."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


def build_replay_timeline(session_path: str | Path, realtime_log_path: str | Path | None = None) -> dict[str, Any]:
    """Merge task session trials and realtime predictions into a replay timeline."""
    session = _read_json(session_path)
    trials = session.get("trials", [])
    predictions = _index_predictions(realtime_log_path)
    events = []

    for trial in trials:
        trial_idx = int(trial.get("trial_idx", len(events)))
        pred = predictions.get(trial_idx, {})
        probability = float(pred.get("probability", 0.0))
        uncertainty = float(pred.get("uncertainty", 0.0))
        warning = bool(pred.get("warning", probability >= 0.65))
        event = {
            "trial_idx": trial_idx,
            "timestamp": float(trial.get("timestamp", trial.get("elapsed_session_time", 0.0))),
            "reaction_time": float(trial.get("reaction_time", 0.0)),
            "action_taken": trial.get("action_taken", ""),
            "target_shape": trial.get("target_shape", ""),
            "stimulus_shape": trial.get("stimulus_shape", ""),
            "is_correct": bool(trial.get("is_correct", False)),
            "probability": probability,
            "uncertainty": uncertainty,
            "warning": warning,
            "risk_band": _risk_band(probability, uncertainty),
            "explanations": _explain_event(trial, probability, uncertainty),
        }
        events.append(event)

    summary = _summarize(events)
    return {
        "session_id": session.get("session_id", "unknown"),
        "session_name": session.get("session_name", ""),
        "summary": summary,
        "events": events,
    }


def export_replay_report(timeline: dict[str, Any], output_path: str | Path) -> Path:
    """Write a Markdown replay report and return its path."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = timeline["summary"]
    lines = [
        "# DriftSync Replay Report",
        "",
        f"Session: `{timeline.get('session_id', 'unknown')}`",
        "",
        "## Summary",
        "",
        f"- Trials: {summary['n_trials']}",
        f"- Errors: {summary['n_errors']}",
        f"- Warnings: {summary['n_warnings']}",
        f"- Average reaction time: {summary['avg_reaction_time']:.3f}s",
        f"- Peak risk: {summary['peak_probability']:.2f}",
        "",
        "## Timeline Highlights",
        "",
    ]
    highlights = [e for e in timeline["events"] if e["warning"] or not e["is_correct"]]
    if not highlights:
        lines.append("No warnings or errors were recorded.")
    for event in highlights[:12]:
        status = "warning" if event["warning"] else "error"
        lines.append(
            f"- trial {event['trial_idx']}: {status}, risk {event['probability']:.2f}, "
            f"uncertainty {event['uncertainty']:.2f}, RT {event['reaction_time']:.3f}s"
        )
        for explanation in event["explanations"][:2]:
            lines.append(f"  - {explanation}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _index_predictions(path: str | Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    data = _read_json(path)
    return {int(row["trial_idx"]): row for row in data if "trial_idx" in row}


def _risk_band(probability: float, uncertainty: float) -> str:
    if probability >= 0.65 or uncertainty >= 0.20:
        return "high"
    if probability >= 0.40:
        return "elevated"
    return "low"


def _explain_event(trial: dict[str, Any], probability: float, uncertainty: float) -> list[str]:
    notes = []
    rt = float(trial.get("reaction_time", 0.0))
    if probability >= 0.65:
        notes.append("Predicted error risk crossed the warning threshold.")
    if uncertainty >= 0.20:
        notes.append("Model uncertainty is high enough to warrant caution.")
    if rt >= 0.70:
        notes.append("Reaction time is slower than the showcase baseline.")
    if not bool(trial.get("is_correct", True)):
        notes.append("Actual task error occurred on this trial.")
    return notes or ["Behavior stayed inside the low-risk band."]


def _summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    probabilities = [e["probability"] for e in events]
    reaction_times = [e["reaction_time"] for e in events]
    return {
        "n_trials": len(events),
        "n_errors": sum(1 for e in events if not e["is_correct"]),
        "n_warnings": sum(1 for e in events if e["warning"]),
        "avg_reaction_time": float(mean(reaction_times)) if reaction_times else 0.0,
        "peak_probability": max(probabilities) if probabilities else 0.0,
    }
