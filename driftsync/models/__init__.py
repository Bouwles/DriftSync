from .base import DriftPredictor
from .lstm_model import LSTMDriftPredictor
from .transformer_model import TransformerDriftPredictor, SinusoidalPositionalEncoding

__all__ = [
    "DriftPredictor",
    "LSTMDriftPredictor",
    "TransformerDriftPredictor",
    "SinusoidalPositionalEncoding",
]


def build_model(model_type: str, cfg=None):
    """
    Factory function.

    Args:
        model_type: "lstm" or "transformer".
        cfg: Model-specific config object. If None, uses defaults.

    Returns:
        Instantiated DriftPredictor model.
    """
    model_type = model_type.lower()
    if model_type == "lstm":
        from driftsync.configs import LSTMConfig
        return LSTMDriftPredictor(cfg if cfg is not None else LSTMConfig())
    elif model_type == "transformer":
        from driftsync.configs import TransformerConfig
        return TransformerDriftPredictor(cfg if cfg is not None else TransformerConfig())
    else:
        raise ValueError(f"Unknown model type: '{model_type}'. Choose 'lstm' or 'transformer'.")
