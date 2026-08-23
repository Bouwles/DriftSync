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
