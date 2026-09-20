"""Render the actual desktop UI headlessly for visual regression review.

Usage: python scripts/capture_ui.py [--output build/ui-review]
The live prediction capture uses the checked-in replay fixture, visibly labeled.
No trained model, human session or fabricated chart series is required.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame

from driftsync import ui
from driftsync.app.application import DriftSyncApplication, State
from driftsync.configs import SimulatorConfig, RealtimeConfig
from driftsync.realtime.live_simulator import LiveDriftSimulator, SHAPE_DRAWERS
from driftsync.realtime.presentation import render_live
from driftsync.simulator.gui import DriftSimulator


def capture(output):
    output.mkdir(parents=True, exist_ok=True)
    app = DriftSyncApplication()
    for state in (State.MENU, State.LEARN, State.DEMO, State.RESULTS, State.PLAY_TASK, State.LIVE_MODE):
        app._enter_state(state)
        app._render()
        pygame.image.save(app.screen, str(output / f"{state.name.lower()}.png"))
    for i in range(7):
        app._enter_state(State.LEARN)
        app.learn_page = i
        app._render()
        pygame.image.save(app.screen, str(output / f"guide-{i + 1}.png"))
    app._enter_state(State.MENU)
    app.viewport.event(pygame.event.Event(pygame.VIDEORESIZE, w=960, h=600))
    app._render()
    pygame.image.save(app.viewport.window, str(output / "workspace-small.png"))

    sim = LiveDriftSimulator(SimulatorConfig(), RealtimeConfig())
    sim.viewport = ui.Viewport((1300, 900), window_size=(1300, 900))
    for label, ready in (("live-unavailable", False), ("live-warmup", True)):
        sim._model_ready = ready
        render_live(sim, sim.viewport.surface, "CIRCLE", 380, 300, 30, "CIRCLE", .4, 3, SHAPE_DRAWERS)
        pygame.image.save(sim.viewport.surface, str(output / f"{label}.png"))
    events = json.loads((Path(__file__).resolve().parents[1] / "tests/fixtures/realtime_log_sample.json").read_text())
    sim.inference._feat_buffer.extend([[0] * 15 for _ in range(20)])
    sim._explainer = None
    sim._last_prob = events[-1]["probability"]
    sim._last_unc = events[-1]["uncertainty"]
    sim._prob_history.extend(event["probability"] for event in events)
    sim._unc_history.extend(event["uncertainty"] for event in events)
    render_live(sim, sim.viewport.surface, "CIRCLE", 380, 300, 30, "CIRCLE", .4, 3, SHAPE_DRAWERS)
    pygame.draw.rect(sim.viewport.surface, ui.BG, (20, 840, 870, 60))
    ui.text(sim.viewport.surface, "UI preview / checked-in replay fixture / no active session", ui.font(16), ui.YELLOW, 28, 860)
    pygame.image.save(sim.viewport.surface, str(output / "live-replay-fixture.png"))
    sim.viewport.event(pygame.event.Event(pygame.VIDEORESIZE, w=960, h=680))
    sim.viewport.present()
    pygame.image.save(sim.viewport.window, str(output / "live-small.png"))

    task = DriftSimulator(SimulatorConfig(), skip_calibration=True)
    task.viewport = ui.Viewport((900, 650))
    task._render_frame(task.viewport.surface, ui.font(28, True), ui.font(20), ui.font(15),
                       "CIRCLE", 380, 300, 30, "CIRCLE", .4, 3)
    pygame.image.save(task.viewport.surface, str(output / "task.png"))
    ui.session_message(task.viewport.surface, [
        ("Personal calibration", ui.font(28, True), ui.TEXT),
        ("You will complete 25 practice trials.", ui.font(20), ui.TEXT),
        ("This measures your normal performance level.", ui.font(20), ui.DIM),
        ("Results are saved as your personal baseline.", ui.font(20), ui.DIM),
        ("Press Enter to begin calibration", ui.font(20), ui.ACCENT)])
    pygame.image.save(task.viewport.surface, str(output / "calibration.png"))
    pygame.quit()
    print(f"UI captures saved to {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/ui-review"))
    capture(parser.parse_args().output)
