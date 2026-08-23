from driftsync.configs import DataConfig, LSTMConfig, TransformerConfig
from driftsync.data.dataset import FEATURE_COLS


def test_data_config_feature_columns_match_dataset_contract():
    cfg = DataConfig()

    assert cfg.feature_columns == FEATURE_COLS


def test_model_default_input_dims_match_feature_contract():
    assert LSTMConfig().input_dim == len(FEATURE_COLS)
    assert TransformerConfig().input_dim == len(FEATURE_COLS)
