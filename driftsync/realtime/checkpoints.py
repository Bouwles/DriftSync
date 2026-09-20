"""Checkpoint path helpers for realtime inference."""

from pathlib import Path

CHECKPOINT_FILENAMES = {
    "lstm": "lstm_best.pt",
    "transformer": "transformer_best.pt",
}


def resolve_workspace_checkpoint(model_type: str, results_dir: str | Path = "driftsync/results") -> Path:
    """Default checkpoint, or newest named training run on a fresh installation."""
    base = Path(results_dir)
    try:
        return resolve_checkpoint_path(model_type, base / "checkpoints")
    except FileNotFoundError:
        candidates = list(base.glob(f"*/checkpoints/{CHECKPOINT_FILENAMES[model_type]}"))
        if not candidates:
            raise
        return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def resolve_checkpoint_path(model_type: str, checkpoint_dir: str | Path) -> Path:
    """Return an existing checkpoint path for a supported model type."""
    if model_type not in CHECKPOINT_FILENAMES:
        supported = ", ".join(sorted(CHECKPOINT_FILENAMES))
        raise ValueError(f"Unsupported model_type '{model_type}'. Expected one of: {supported}.")

    ckpt_path = Path(checkpoint_dir) / CHECKPOINT_FILENAMES[model_type]
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No {model_type} checkpoint found at {ckpt_path}. "
            "Train a model first with `python run_experiment.py --quick`, "
            "or run the Full Demo inside the DriftSync launcher."
        )

    return ckpt_path
