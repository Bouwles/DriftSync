import torch

from driftsync.configs import LSTMConfig, TransformerConfig
from driftsync.data.dataset import FEATURE_COLS
from driftsync.models import build_model


def test_lstm_forward_shape_matches_binary_risk_output():
    model = build_model("lstm", LSTMConfig(input_dim=len(FEATURE_COLS), hidden_dim=16, num_layers=1))
    x = torch.zeros(3, 5, len(FEATURE_COLS))

    logits = model(x)

    assert logits.shape == (3,)
    assert model.count_parameters() > 0


def test_transformer_forward_shape_matches_binary_risk_output():
    model = build_model(
        "transformer",
        TransformerConfig(input_dim=len(FEATURE_COLS), d_model=16, nhead=4, num_encoder_layers=1),
    )
    x = torch.zeros(3, 5, len(FEATURE_COLS))

    logits = model(x)

    assert logits.shape == (3,)
    assert model.count_parameters() > 0
