"""Exercise a packaged executable without opening a visible window."""
import json
import os
from pathlib import Path
import sys


def run_release_check(report_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    import pygame
    import torch
    from driftsync import __version__
    from driftsync.app.application import DriftSyncApplication, State
    from driftsync.models.lstm_model import LSTMDriftPredictor
    from driftsync.models.transformer_model import TransformerDriftPredictor
    from driftsync.smoke import run_smoke_checks

    report_path = Path(report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks = run_smoke_checks()
    app = DriftSyncApplication()
    try:
        for state in (State.MENU, State.LEARN, State.DEMO, State.RESULTS, State.PLAY_TASK, State.LIVE_MODE):
            app._enter_state(state)
            app._render()
        checks.append("all workspace screens rendered")
        app._enter_state(State.MATHEMATICS)
        for model in ("LSTM", "Transformer"):
            app.math_view.model = model
            for page in range(5):
                app.math_view.page = page
                app._render()
        app.math_view.page = 0
        app._render()
        pygame.image.save(app.screen, str(report_path.with_suffix(".png")))
        checks.append("five mathematics topics and both architectures rendered")
        for cls in (LSTMDriftPredictor, TransformerDriftPredictor):
            model = cls().eval()
            with torch.no_grad():
                output = model(torch.zeros(1, 20, 15))
            if output.shape != (1,) or not torch.isfinite(output).all():
                raise RuntimeError(f"Invalid packaged model output: {cls.__name__}")
        checks.append("LSTM and Transformer CPU forward passes")
        report = {"version": __version__, "frozen": bool(getattr(sys, "frozen", False)),
                  "status": "passed", "checks": checks}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
        return report
    finally:
        pygame.quit()
