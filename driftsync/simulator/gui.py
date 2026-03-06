"""
Simulator GUI
=============
Pygame-based interactive cognitive task.

Controls
--------
- Mouse click on the stimulus shape -> "click" action
- Spacebar / right-click to SKIP (no click = the player judges it wrong shape)
- ESC to end session early

Displays
--------
- Current rule ("Click CIRCLES")
- Score, accuracy, trial count
- Countdown bar for current trial's time window
- Fatigue / drift warning overlay (high error rate indicator)
"""

import sys
import math
import time

import pygame

from driftsync.configs import SimulatorConfig
from driftsync.simulator.task_engine import TaskEngine
from driftsync.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG_COLOR      = (15,  15,  25)
TEXT_COLOR    = (220, 220, 220)
RULE_COLOR    = (100, 220, 255)
TARGET_COLOR  = (80,  200, 120)
DISTRACT_COLOR= (220,  80,  80)
TIMER_OK      = (80,  200, 120)
TIMER_WARN    = (240, 180,  50)
TIMER_CRIT    = (220,  60,  60)
SCORE_COLOR   = (180, 180, 255)
DRIFT_OVERLAY = (220, 60, 60, 60)   # semi-transparent red tint when drifting


# ---------------------------------------------------------------------------
# Shape drawing helpers
# ---------------------------------------------------------------------------

def draw_circle(surface: pygame.Surface, color, x: int, y: int, r: int) -> None:
    pygame.draw.circle(surface, color, (x, y), r, 0)
    pygame.draw.circle(surface, (255, 255, 255), (x, y), r, 2)


def draw_square(surface: pygame.Surface, color, x: int, y: int, r: int) -> None:
    rect = pygame.Rect(x - r, y - r, r * 2, r * 2)
    pygame.draw.rect(surface, color, rect, border_radius=6)
    pygame.draw.rect(surface, (255, 255, 255), rect, 2, border_radius=6)


def draw_triangle(surface: pygame.Surface, color, x: int, y: int, r: int) -> None:
    pts = [
        (x, y - r),
        (x - int(r * 0.866), y + r // 2),
        (x + int(r * 0.866), y + r // 2),
    ]
    pygame.draw.polygon(surface, color, pts)
    pygame.draw.polygon(surface, (255, 255, 255), pts, 2)


SHAPE_DRAWERS = {
    "CIRCLE":   draw_circle,
    "SQUARE":   draw_square,
    "TRIANGLE": draw_triangle,
}


# ---------------------------------------------------------------------------
# Main simulator class
# ---------------------------------------------------------------------------

class DriftSimulator:
    """
    Pygame cognitive task simulator.

    Usage
    -----
        sim = DriftSimulator(cfg)
        session_file = sim.run()
    """

    def __init__(self, cfg: SimulatorConfig | None = None):
        self.cfg = cfg or SimulatorConfig()
        self.engine = TaskEngine(self.cfg)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> str:
        """Run the simulator and return the path to the saved session file."""
        pygame.init()
        pygame.display.set_caption("DriftSync — Cognitive Task Simulator")

        screen = pygame.display.set_mode((self.cfg.window_width, self.cfg.window_height))
        clock = pygame.time.Clock()

        font_large  = pygame.font.SysFont("Consolas", 32, bold=True)
        font_med    = pygame.font.SysFont("Consolas", 22)
        font_small  = pygame.font.SysFont("Consolas", 16)

        if self._show_intro(screen, font_large, font_med, clock) == "quit":
            pygame.quit()
            path = self.engine.save_session()
            return str(path)

        # --------------- main task loop --------------------------------
        while not self.engine.is_finished:
            stimulus = self.engine.next_stimulus()
            result = self._run_trial(
                screen, clock, stimulus, font_large, font_med, font_small
            )
            if result == "quit":
                break

        # --------------- outro -----------------------------------------
        self._show_outro(screen, font_large, font_med, clock)
        pygame.quit()

        path = self.engine.save_session()
        return str(path)

    # ------------------------------------------------------------------
    # Trial execution
    # ------------------------------------------------------------------

    def _run_trial(
        self,
        screen: pygame.Surface,
        clock: pygame.time.Clock,
        stimulus: dict,
        font_large,
        font_med,
        font_small,
    ) -> str:
        """
        Render and handle one trial.

        Returns "quit" if user closes window, else "ok".
        """
        shape      = stimulus["shape"]
        sx, sy     = stimulus["x"], stimulus["y"]
        rule       = stimulus["rule"]
        time_window= stimulus["time_window"]
        radius     = self.cfg.target_radius

        trial_start = time.time()
        action      = None
        rt          = 0.0

        while True:
            elapsed = time.time() - trial_start

            # --- timeout ---
            if elapsed >= time_window:
                rt = time_window
                action = "timeout"
                break

            # --- events ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.engine.record_trial(shape, "timeout", elapsed)
                    return "quit"

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.engine.record_trial(shape, "timeout", elapsed)
                        return "quit"
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        rt = elapsed
                        action = "skip"

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    dist = math.hypot(mx - sx, my - sy)
                    # Generous hit radius (+30 %) so it's fair
                    if dist <= radius * 1.3:
                        rt = elapsed
                        action = "click"

            if action is not None:
                break

            # --- render ---
            self._render_frame(
                screen, font_large, font_med, font_small,
                shape, sx, sy, radius, rule, elapsed, time_window,
            )
            clock.tick(self.cfg.fps)

        trial = self.engine.record_trial(shape, action, rt)
        self._flash_feedback(screen, trial.is_correct, clock)
        return "ok"

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_frame(
        self,
        screen,
        font_large,
        font_med,
        font_small,
        shape: str,
        sx: int, sy: int, radius: int,
        rule: str,
        elapsed: float,
        time_window: float,
    ) -> None:
        screen.fill(BG_COLOR)

        # Drift warning overlay when recent error rate is high
        recent_errors = self.engine._error_history[-10:] if self.engine._error_history else []
        if recent_errors and sum(recent_errors) / len(recent_errors) > 0.5:
            overlay = pygame.Surface(
                (self.cfg.window_width, self.cfg.window_height), pygame.SRCALPHA
            )
            overlay.fill(DRIFT_OVERLAY)
            screen.blit(overlay, (0, 0))

        # --- HUD top bar ---
        rule_surf = font_large.render(f"CLICK  {rule}S", True, RULE_COLOR)
        screen.blit(rule_surf, (self.cfg.window_width // 2 - rule_surf.get_width() // 2, 12))

        trial_surf = font_small.render(
            f"Trial {self.engine.trial_count + 1} / {self.cfg.num_trials}", True, TEXT_COLOR
        )
        screen.blit(trial_surf, (12, 14))

        # Accuracy
        trials_done = self.engine.session_data.trials
        if trials_done:
            acc = sum(t.is_correct for t in trials_done) / len(trials_done)
            acc_surf = font_small.render(f"Acc: {acc:.1%}", True, SCORE_COLOR)
            screen.blit(acc_surf, (self.cfg.window_width - 120, 14))

        # --- Timer bar ---
        bar_w = self.cfg.window_width - 40
        bar_h = 10
        bar_x, bar_y = 20, 55
        ratio = 1.0 - elapsed / time_window
        if ratio > 0.5:
            bar_color = TIMER_OK
        elif ratio > 0.25:
            bar_color = TIMER_WARN
        else:
            bar_color = TIMER_CRIT

        pygame.draw.rect(screen, (50, 50, 60), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        pygame.draw.rect(screen, bar_color, (bar_x, bar_y, int(bar_w * ratio), bar_h), border_radius=4)

        # --- Stimulus ---
        is_target = (shape == rule)
        color = TARGET_COLOR if is_target else DISTRACT_COLOR
        SHAPE_DRAWERS[shape](screen, color, sx, sy, radius)

        # Shape label near stimulus (helpful for learning)
        lbl = font_small.render(shape, True, TEXT_COLOR)
        screen.blit(lbl, (sx - lbl.get_width() // 2, sy + radius + 8))

        # --- Footer hint ---
        hint = font_small.render("Click shape  |  SPACE = skip  |  ESC = quit", True, (80, 80, 90))
        screen.blit(hint, (self.cfg.window_width // 2 - hint.get_width() // 2,
                            self.cfg.window_height - 28))

        pygame.display.flip()

    def _flash_feedback(self, screen, is_correct: bool, clock) -> None:
        """Brief 200 ms coloured flash indicating correct/incorrect."""
        color = (40, 180, 80) if is_correct else (180, 40, 40)
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        overlay.fill((*color, 70))
        screen.blit(overlay, (0, 0))
        pygame.display.flip()
        start = time.time()
        while time.time() - start < 0.18:
            pygame.event.pump()
            clock.tick(self.cfg.fps)

    # ------------------------------------------------------------------
    # Intro / outro screens
    # ------------------------------------------------------------------

    def _show_intro(self, screen, font_large, font_med, clock) -> None:
        lines = [
            ("DriftSync — Cognitive Task", font_large, RULE_COLOR),
            ("", font_med, TEXT_COLOR),
            ("A coloured shape will appear on screen.", font_med, TEXT_COLOR),
            ("Click it if it matches the rule shown at the top.", font_med, TEXT_COLOR),
            ("Press SPACE to SKIP if it does NOT match.", font_med, TEXT_COLOR),
            ("", font_med, TEXT_COLOR),
            ("Your reaction time and accuracy are tracked.", font_med, (160, 160, 200)),
            ("Try to stay focused — cognitive drift is being measured.", font_med, (160, 160, 200)),
            ("", font_med, TEXT_COLOR),
            ("Press ENTER to begin", font_large, SCORE_COLOR),
        ]
        self._render_text_screen(screen, lines, clock, wait_key=pygame.K_RETURN)

    def _show_outro(self, screen, font_large, font_med, clock) -> None:
        trials = self.engine.session_data.trials
        acc = sum(t.is_correct for t in trials) / max(1, len(trials))
        avg_rt = sum(t.reaction_time for t in trials) / max(1, len(trials))
        lines = [
            ("Session Complete!", font_large, RULE_COLOR),
            ("", font_med, TEXT_COLOR),
            (f"Trials completed : {len(trials)}", font_med, TEXT_COLOR),
            (f"Overall accuracy : {acc:.1%}", font_med, SCORE_COLOR),
            (f"Avg reaction time: {avg_rt:.3f}s", font_med, SCORE_COLOR),
            ("", font_med, TEXT_COLOR),
            ("Data saved. Press ENTER or close window.", font_med, (160, 160, 200)),
        ]
        self._render_text_screen(screen, lines, clock, wait_key=pygame.K_RETURN, timeout=8.0)

    def _render_text_screen(
        self, screen, lines, clock, wait_key=None, timeout: float | None = None
    ) -> str:
        """Returns 'quit' if window closed, 'ok' on key press, 'timeout' on timer."""
        start = time.time()
        while True:
            if timeout and (time.time() - start) > timeout:
                return "timeout"
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return "quit"
                if wait_key and event.type == pygame.KEYDOWN and event.key == wait_key:
                    return "ok"

            screen.fill(BG_COLOR)
            total_h = sum(fnt.get_height() + 6 for _, fnt, _ in lines)
            y = (self.cfg.window_height - total_h) // 2
            for text, fnt, color in lines:
                surf = fnt.render(text, True, color)
                screen.blit(surf, (self.cfg.window_width // 2 - surf.get_width() // 2, y))
                y += fnt.get_height() + 6
            pygame.display.flip()
            clock.tick(self.cfg.fps)
