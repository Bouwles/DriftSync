from .inference_engine import RealtimeInferenceEngine

# LiveDriftSimulator requires pygame — imported lazily to avoid hard dependency
__all__ = ["RealtimeInferenceEngine", "LiveDriftSimulator"]


def __getattr__(name):
    if name == "LiveDriftSimulator":
        from .live_simulator import LiveDriftSimulator
        return LiveDriftSimulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
