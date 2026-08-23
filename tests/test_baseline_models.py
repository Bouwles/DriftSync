import numpy as np
import pytest

from driftsync.ml.baseline_models import ThresholdModel, train_baseline_models


def test_threshold_model_scores_weighted_error_signals():
    model = ThresholdModel(threshold=0.35)
    features = np.zeros(15, dtype=np.float32)
    features[3] = 0.6
    features[4] = 0.4
    features[8] = 0.5

    assert model.predict_proba(features) == pytest.approx(0.52)
    assert model.is_high_risk(features) is True


def test_threshold_model_handles_short_feature_vectors():
    assert ThresholdModel().predict_proba(np.array([0.1, 0.2])) == 0.0


def test_train_baseline_models_falls_back_when_data_is_small():
    X = np.zeros((4, 3, 15), dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int32)

    mode, model = train_baseline_models(X, y)

    assert mode == ThresholdModel.MODE
    assert isinstance(model, ThresholdModel)
