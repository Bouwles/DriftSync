from pathlib import Path

import pytest

from driftsync.realtime.checkpoints import CHECKPOINT_FILENAMES, resolve_checkpoint_path


def test_resolve_checkpoint_path_returns_existing_model_checkpoint(tmp_path):
    checkpoint = tmp_path / CHECKPOINT_FILENAMES["lstm"]
    checkpoint.write_bytes(b"checkpoint")

    assert resolve_checkpoint_path("lstm", tmp_path) == checkpoint


def test_resolve_checkpoint_path_rejects_unknown_model_type(tmp_path):
    with pytest.raises(ValueError, match="Unsupported model_type"):
        resolve_checkpoint_path("gru", tmp_path)


def test_resolve_checkpoint_path_reports_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train a model first"):
        resolve_checkpoint_path("transformer", tmp_path)
def test_workspace_discovers_trained_run_on_fresh_install(tmp_path):
    from driftsync.realtime.checkpoints import resolve_workspace_checkpoint
    checkpoint = tmp_path / "example_run" / "checkpoints" / "lstm_best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"test")
    assert resolve_workspace_checkpoint("lstm", tmp_path) == checkpoint


def test_workspace_preserves_explicit_default_checkpoint(tmp_path):
    from driftsync.realtime.checkpoints import resolve_workspace_checkpoint
    root = tmp_path / "checkpoints" / "lstm_best.pt"
    root.parent.mkdir(parents=True)
    root.write_bytes(b"test")
    newer = tmp_path / "new_run" / "checkpoints" / "lstm_best.pt"
    newer.parent.mkdir(parents=True)
    newer.write_bytes(b"test")
    assert resolve_workspace_checkpoint("lstm", tmp_path) == root
