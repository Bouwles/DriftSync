"""Fast non-GUI smoke checks for DriftSync."""

from __future__ import annotations

from driftsync.configs import CONFIG
from driftsync.data.dataset import FEATURE_COLS
from driftsync.realtime.inference_engine import NUM_FEATURES


def run_smoke_checks() -> list[str]:
    """Run lightweight project checks and return human-readable results."""
    checks = []

    if CONFIG.data.feature_columns != FEATURE_COLS:
        raise RuntimeError("DataConfig.feature_columns does not match dataset.FEATURE_COLS")
    checks.append(f"feature contract: {len(FEATURE_COLS)} features")

    if NUM_FEATURES != len(FEATURE_COLS):
        raise RuntimeError("Realtime inference feature count is out of sync")
    checks.append("realtime feature count: aligned")

    if CONFIG.lstm.input_dim != len(FEATURE_COLS):
        raise RuntimeError("LSTM input_dim is out of sync")
    checks.append("lstm input_dim: aligned")

    if CONFIG.transformer.input_dim != len(FEATURE_COLS):
        raise RuntimeError("Transformer input_dim is out of sync")
    checks.append("transformer input_dim: aligned")

    return checks


def main() -> None:
    print("DriftSync smoke checks")
    for line in run_smoke_checks():
        print(f"- {line}")
    print("OK")


if __name__ == "__main__":
    main()
