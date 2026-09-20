"""Headless UI regression coverage for real controls and state transitions."""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from driftsync.app.application import DriftSyncApplication, State


@pytest.fixture
def app():
    application = DriftSyncApplication()
    yield application
    pygame.quit()


def test_task_setup_renders(app):
    app._enter_state(State.PLAY_TASK)
    app._render()
    assert app._btn_play_start.rect.width > 0


def test_live_model_click_matches_visible_selector(app):
    app._enter_state(State.LIVE_MODE)
    app._render()
    assert hasattr(app, "_live_model_rects"), "Selectors must share render and input geometry"
    rect = app._live_model_rects["transformer"]
    app._handle_live_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rect.center))
    assert app.live_model_choice == "transformer"


def test_return_to_training_keeps_active_worker(app):
    app._enter_state(State.DEMO)
    worker = app.demo_worker
    worker.status = "running"
    app._enter_state(State.MENU)
    app._enter_state(State.DEMO)
    assert app.demo_worker is worker


def test_escape_closes_plot_before_leaving_results(app):
    app._enter_state(State.RESULTS)
    app.result_full_view = pygame.Surface((100, 100))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    app._handle_events()
    assert app.state == State.RESULTS
    assert app.result_full_view is None


def test_resized_pointer_still_activates_live_model(app):
    app._enter_state(State.LIVE_MODE)
    app._render()
    app.viewport.event(pygame.event.Event(pygame.VIDEORESIZE, w=960, h=600))
    rect = app._live_model_rects["transformer"]
    bounds = app.viewport.bounds
    point = (bounds.x + rect.centerx * bounds.width / 1280,
             bounds.y + rect.centery * bounds.height / 800)
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=point))
    app._handle_events()
    assert app.live_model_choice == "transformer"


def test_live_warmup_and_missing_model_are_not_low_risk():
    from driftsync.configs import SimulatorConfig, RealtimeConfig
    from driftsync.realtime.live_simulator import LiveDriftSimulator
    from driftsync.realtime.presentation import prediction_state

    sim = LiveDriftSimulator(SimulatorConfig(), RealtimeConfig())
    assert not sim._prob_history
    assert prediction_state(sim)[0] == "Recording only"
    sim._model_ready = True
    assert prediction_state(sim)[0] == "Collecting history"
    for _ in range(sim.inference._seq_len):
        sim.inference._feat_buffer.append([0] * 15)
    sim._last_prob = .2
    sim._last_unc = .3
    assert prediction_state(sim)[0] == "Uncertain prediction"
    sim._last_unc = .01
    assert prediction_state(sim)[0] == "Low error risk"


def test_failed_task_does_not_claim_session_saved(app, monkeypatch):
    from driftsync.simulator.gui import DriftSimulator

    def fail(_self):
        raise RuntimeError("test write error")

    monkeypatch.setattr(DriftSimulator, "run", fail)
    app._launch_play_task()
    assert not app._play_task_done
    assert "test write error" in app.notice


def test_tab_enter_operates_workspace(app):
    app.state = State.MENU
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB, mod=0))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode="\r"))
    app._handle_events()
    assert app.state == State.LIVE_MODE


def test_results_keyboard_focus_survives_render(app):
    app._enter_state(State.RESULTS)
    app.results_tab = "human"
    app._render()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB, mod=0))
    app._handle_events()
    app._render()
    assert app._btn_human_refresh.focused


@pytest.mark.parametrize("status", ["done", "error"])
def test_training_can_start_new_run_after_terminal_state(app, status):
    app._enter_state(State.DEMO)
    app.demo_worker.status = status
    app._render()
    app._handle_demo_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                                             pos=app._btn_demo_start.rect.center))
    assert app.demo_worker.status == "idle"
    assert not app._btn_demo_start.disabled


def test_checked_in_results_show_actual_metrics(app):
    app._enter_state(State.RESULTS)
    # The root experiment summary is nested under test_metrics.
    root_index = next(i for i, path in enumerate(app.result_runs) if path.name == "results")
    app._load_ml_results(root_index)
    assert app.result_metrics["lstm"]["accuracy"] == pytest.approx(.7443181818)


def test_cancelled_calibration_does_not_save_partial_baseline(app, monkeypatch):
    from driftsync.configs import SimulatorConfig
    from driftsync.simulator.gui import DriftSimulator, CalibrationEngine

    sim = DriftSimulator(SimulatorConfig())
    monkeypatch.setattr(sim, "_render_text_screen", lambda *args, **kwargs: "ok")

    def cancel(*args):
        engine = args[6]
        engine.record_trial("CIRCLE", "click", .5)
        return "quit"

    monkeypatch.setattr(sim, "_run_trial_calibration", cancel)
    saved = []
    monkeypatch.setattr(CalibrationEngine, "save", lambda *args: saved.append(args))
    result = sim._run_calibration(app.screen, app.clock, app.f_title, app.f_body, app.f_small)
    assert result is None
    assert sim._calibration_cancelled
    assert not saved


def test_live_intro_escape_returns_without_exit(app):
    from driftsync.configs import SimulatorConfig, RealtimeConfig
    from driftsync.realtime.live_simulator import LiveDriftSimulator

    sim = LiveDriftSimulator(SimulatorConfig(), RealtimeConfig())
    sim.viewport = app.viewport
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    assert sim._show_intro(app.screen, app.f_title, app.f_body, app.clock) == "quit"


def test_cancelled_live_setup_does_not_overwrite_session_or_log(app, monkeypatch):
    from driftsync.configs import SimulatorConfig, RealtimeConfig
    from driftsync.realtime.live_simulator import LiveDriftSimulator

    sim = LiveDriftSimulator(SimulatorConfig(), RealtimeConfig())
    saved = []
    monkeypatch.setattr(sim.inference, "load_model", lambda *args: None)
    monkeypatch.setattr(sim, "_show_intro", lambda *args: "quit")
    monkeypatch.setattr(sim.engine, "save_session", lambda: saved.append("session") or "unused.json")
    monkeypatch.setattr(sim.inference, "save_log", lambda: saved.append("log"))
    monkeypatch.setattr(sim, "_save_session_metrics", lambda *args: saved.append("metrics"))
    assert sim.run() == ""
    assert saved == []
