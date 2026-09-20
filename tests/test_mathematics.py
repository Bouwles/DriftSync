import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest
import numpy as np


def test_math_navigation_and_interactive_example():
    from driftsync.app.application import DriftSyncApplication, State
    app = DriftSyncApplication()
    try:
        assert hasattr(State, "MATHEMATICS"), "Mathematics needs its own discoverable page"
        app._enter_state(State.MATHEMATICS)
        app._render()
        view = app.math_view
        assert view.example_label == 1
        # Toggle the only future error; the target must become zero.
        rect = view.outcome_rects[2]
        view.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rect.center))
        assert view.example_label == 0
        view.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        assert view.page == 1
    finally:
        pygame.quit()


def test_uncertainty_example_uses_sample_standard_deviation():
    from driftsync.app.mathematics import sample_summary
    values = [.2, .3, .7, .8]
    mean, sd = sample_summary(values)
    assert mean == pytest.approx(.5)
    assert sd == pytest.approx(np.std(values, ddof=1))


def test_all_math_topics_render_and_remain_keyboard_accessible():
    from driftsync.app.application import DriftSyncApplication, State
    app = DriftSyncApplication()
    try:
        app._enter_state(State.MATHEMATICS)
        for model in ("LSTM", "Transformer"):
            app.math_view.model = model
            for page in range(5):
                app.math_view.page = page
                app._render()
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB, mod=0))
                app._handle_events()
                assert app.math_view.focus >= 0
                assert app.math_view.controls
    finally:
        pygame.quit()
