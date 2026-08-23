import json

import pytest
import torch

from driftsync.configs import DataConfig, RealtimeConfig
from driftsync.realtime.inference_engine import RealtimeInferenceEngine


class ConstantModel:
    def mc_dropout_predict(self, x, n_samples=30):
        batch_size = x.shape[0]
        return torch.full((batch_size,), 0.72), torch.full((batch_size,), 0.04)


def make_engine(tmp_path, seq_len=2):
    return RealtimeInferenceEngine(
        rt_cfg=RealtimeConfig(log_file=str(tmp_path / "realtime_log.json")),
        data_cfg=DataConfig(sequence_length=seq_len),
    )


def test_realtime_log_contains_only_inference_events(tmp_path):
    engine = make_engine(tmp_path)
    engine.model = ConstantModel()

    engine.update(0.40, True, "CIRCLE", "CIRCLE", "click")
    engine.update(0.65, False, "SQUARE", "CIRCLE", "click")

    assert len(engine._log) == 1
    assert set(engine._log[0]) == {
        "trial_idx",
        "timestamp",
        "probability",
        "uncertainty",
        "warning",
        "is_correct",
    }


def test_save_log_writes_clean_event_schema(tmp_path):
    engine = make_engine(tmp_path)
    engine.model = ConstantModel()

    engine.update(0.40, True, "CIRCLE", "CIRCLE", "click")
    engine.update(0.65, False, "SQUARE", "CIRCLE", "click")
    engine.save_log()

    events = json.loads((tmp_path / "realtime_log.json").read_text())

    assert len(events) == 1
    assert events[0]["probability"] == pytest.approx(0.72)
    assert events[0]["warning"] is True
    assert "rt" not in events[0]


def test_sample_realtime_log_fixture_matches_schema():
    fixture = json.loads(open("tests/fixtures/realtime_log_sample.json", encoding="utf-8").read())
    expected_keys = {"trial_idx", "timestamp", "probability", "uncertainty", "warning", "is_correct"}

    assert fixture
    assert all(set(event) == expected_keys for event in fixture)
    assert all(0.0 <= event["probability"] <= 1.0 for event in fixture)
    assert all(isinstance(event["warning"], bool) for event in fixture)
