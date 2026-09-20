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

from __future__ import annotations


import sys
import math
import time

import pygame

from driftsync import ui

from driftsync.configs import SimulatorConfig
from driftsync.simulator.task_engine import TaskEngine
from driftsync.utils import get_logger

try:
    from driftsync.ml.calibrator import CalibrationEngine, BaselineStats
    _CALIBRATOR_AVAILABLE = True
except ImportError:
    _CALIBRATOR_AVAILABLE = False

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG_COLOR, TEXT_COLOR, RULE_COLOR = ui.BG, ui.TEXT, ui.ACCENT
TARGET_COLOR, DISTRACT_COLOR = ui.GREEN, ui.RED
TIMER_OK, TIMER_WARN, TIMER_CRIT = ui.ACCENT, ui.YELLOW, ui.RED
SCORE_COLOR = ui.TEXT
DRIFT_OVERLAY = (*ui.RED, 30)


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

    def __init__(self, cfg: SimulatorConfig | None = None, skip_calibration: bool = False):
        self.cfg              = cfg or SimulatorConfig()
        self.engine           = TaskEngine(self.cfg)
        self.skip_calibration = skip_calibration
        self.baseline         = None
        self._calibration_cancelled = False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> str:
        """Return the session file path, or an empty string when setup is canceled."""
        pygame.init()
        pygame.display.set_caption("DriftSync — Cognitive Task Simulator")

        self.viewport = ui.Viewport((self.cfg.window_width, self.cfg.window_height), "DriftSync | Record task")
        screen = self.viewport.surface
        clock  = pygame.time.Clock()

        font_large = ui.font(28, True)
        font_med   = ui.font(20)
        font_small = ui.font(15)

        # Calibration phase (unless skipped or already calibrated this session)
        if _CALIBRATOR_AVAILABLE and not self.skip_calibration:
            needs_calib = not CalibrationEngine.is_calibrated()
            if needs_calib:
                self.baseline = self._run_calibration(screen, clock, font_large, font_med, font_small)
            else:
                self.baseline = CalibrationEngine.load()
        else:
            self.baseline = None

        if self._calibration_cancelled:
            pygame.quit()
            return ""

        if self._show_intro(screen, font_large, font_med, clock) == "quit":
            pygame.quit()
            return ""

        while not self.engine.is_finished:
            stimulus = self.engine.next_stimulus()
            result   = self._run_trial(
                screen, clock, stimulus, font_large, font_med, font_small
            )
            if result == "quit":
                break

        self._show_outro(screen, font_large, font_med, clock)
        pygame.quit()

        path = self.engine.save_session()
        return str(path)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _run_calibration(
        self,
        screen: pygame.Surface,
        clock: pygame.time.Clock,
        font_large,
        font_med,
        font_small,
        num_trials: int = 25,
    ):
        """
        Run a short calibration phase before the main task.

        Runs `num_trials` trials with a "Personal calibration" header.
        After completion, computes and saves the user's baseline stats.

        Returns a BaselineStats object, or None on failure.
        """
        # Show calibration intro screen
        calib_lines = [
            ("Personal calibration", font_large, RULE_COLOR),
            ("", font_med, TEXT_COLOR),
            (f"You will complete {num_trials} practice trials.", font_med, TEXT_COLOR),
            ("This measures your normal performance level.", font_med, TEXT_COLOR),
            ("Results are saved as your personal baseline.", font_med, ui.DIM),
            ("", font_med, TEXT_COLOR),
            ("No risk warnings shown during calibration.", font_med, ui.DIM),
            ("", font_med, TEXT_COLOR),
            ("Press ENTER to begin calibration", font_large, SCORE_COLOR),
        ]
        result = self._render_text_screen(screen, calib_lines, clock, wait_key=pygame.K_RETURN)
        if result == "quit":
            self._calibration_cancelled = True
            return None

        # Create a temporary engine for calibration trials
        from driftsync.configs import SimulatorConfig as SC
        calib_cfg    = SC(num_trials=num_trials, session_name="calibration",
                          data_dir=self.cfg.data_dir)
        calib_engine = TaskEngine(calib_cfg)

        completed = 0
        while not calib_engine.is_finished:
            stimulus = calib_engine.next_stimulus()
            result   = self._run_trial_calibration(
                screen, clock, stimulus, font_large, font_med, font_small,
                calib_engine, completed, num_trials
            )
            completed += 1
            if result == "quit":
                self._calibration_cancelled = True
                return None

        # Compute baseline from calibration trials
        if not calib_engine.session_data.trials:
            return None

        engine = CalibrationEngine()
        baseline = engine.compute_baseline(calib_engine.session_data.trials)
        engine.save(baseline)
        logger.info(
            "Calibration complete: mean_rt=%.3f  accuracy=%.1f%%",
            baseline.mean_rt, baseline.accuracy * 100,
        )

        # Show calibration summary
        acc_val = baseline.accuracy
        summary_lines = [
            ("Calibration Complete", font_large, RULE_COLOR),
            ("", font_med, TEXT_COLOR),
            ("Your baseline has been recorded:", font_med, TEXT_COLOR),
            (f"Mean reaction time : {baseline.mean_rt:.3f}s", font_med, SCORE_COLOR),
            (f"Accuracy           : {acc_val:.1%}", font_med, SCORE_COLOR),
            (f"Error rate         : {baseline.error_rate:.1%}", font_med, SCORE_COLOR),
            ("", font_med, TEXT_COLOR),
            ("The main task will now begin.", font_med, ui.DIM),
            ("Press ENTER to continue", font_large, SCORE_COLOR),
        ]
        if self._render_text_screen(screen, summary_lines, clock, wait_key=pygame.K_RETURN, timeout=12.0) == "quit":
            self._calibration_cancelled = True
        return baseline

    def _run_trial_calibration(
        self,
        screen: pygame.Surface,
        clock: pygame.time.Clock,
        stimulus: dict,
        font_large,
        font_med,
        font_small,
        engine: "TaskEngine",
        completed: int,
        total: int,
    ) -> str:
        """Run a single calibration trial (no risk overlay)."""
        shape       = stimulus["shape"]
        sx, sy      = stimulus["x"], stimulus["y"]
        rule        = stimulus["rule"]
        time_window = stimulus["time_window"]
        radius      = self.cfg.target_radius

        trial_start = time.time()
        action      = None
        rt          = 0.0

        while True:
            elapsed = time.time() - trial_start
            if elapsed >= time_window:
                rt, action = time_window, "timeout"
                break

            for raw_event in pygame.event.get():
                event = self.viewport.event(raw_event)
                if event.type == pygame.QUIT:
                    engine.record_trial(shape, "timeout", elapsed)
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "quit"
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        rt, action = elapsed, "skip"
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if math.hypot(mx - sx, my - sy) <= radius * 1.3:
                        rt, action = elapsed, "click"

            if action is not None:
                break

            # Render calibration frame
            screen.fill(BG_COLOR)
            # Calibration header
            cal_surf = font_large.render("Calibration", True, ui.YELLOW)
            screen.blit(cal_surf, (self.cfg.window_width // 2 - cal_surf.get_width() // 2, 8))
            prog_surf = font_small.render(f"Trial {completed + 1} / {total}", True, TEXT_COLOR)
            screen.blit(prog_surf, (12, 14))
            rule_surf = font_med.render(f"Click {rule}S", True, RULE_COLOR)
            screen.blit(rule_surf, (self.cfg.window_width // 2 - rule_surf.get_width() // 2, 44))

            # Timer bar
            bar_w = self.cfg.window_width - 40
            ratio = max(0, 1.0 - elapsed / time_window)
            bar_color = TIMER_OK if ratio > 0.5 else TIMER_WARN if ratio > 0.25 else TIMER_CRIT
            pygame.draw.rect(screen, ui.PANEL2, (20, 68, bar_w, 10), border_radius=4)
            if ratio > 0:
                pygame.draw.rect(screen, bar_color, (20, 68, int(bar_w * ratio), 10), border_radius=4)

            # Progress bar for calibration
            prog_frac = completed / max(total, 1)
            pygame.draw.rect(screen, ui.PANEL2, (20, 82, bar_w, 6), border_radius=3)
            if prog_frac > 0:
                pygame.draw.rect(screen, ui.ACCENT, (20, 82, int(bar_w * prog_frac), 6), border_radius=3)

            # Stimulus
            is_target = (shape == rule)
            SHAPE_DRAWERS[shape](screen, TARGET_COLOR if is_target else DISTRACT_COLOR, sx, sy, radius)
            lbl = font_small.render(shape, True, TEXT_COLOR)
            screen.blit(lbl, (sx - lbl.get_width() // 2, sy + radius + 8))

            hint = font_small.render("Click shape  |  SPACE = skip", True, ui.DIM)
            screen.blit(hint, (self.cfg.window_width // 2 - hint.get_width() // 2, self.cfg.window_height - 28))
            self.viewport.present()
            clock.tick(self.cfg.fps)

        trial = engine.record_trial(shape, action, rt)
        self._flash_feedback(screen, trial.is_correct, clock)
        return "ok"

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
            for raw_event in pygame.event.get():
                event = self.viewport.event(raw_event)
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

        recent_errors = self.engine._error_history[-10:]
        if recent_errors and sum(recent_errors) / len(recent_errors) > 0.5:
            ui.text(screen, "Recent errors elevated. Check the rule.", ui.font(15), ui.YELLOW,
                    24, self.cfg.window_height - 62)
        ui.signal_mark(screen, 22, 21, 22)

        # --- HUD top bar ---
        rule_surf = font_large.render(f"Click {rule.lower()}s", True, RULE_COLOR)
        screen.blit(rule_surf, (self.cfg.window_width // 2 - rule_surf.get_width() // 2, 12))

        trial_surf = font_small.render(
            f"Trial {self.engine.trial_count + 1} / {self.cfg.num_trials}", True, TEXT_COLOR
        )
        screen.blit(trial_surf, (54, 24))

        # Accuracy
        trials_done = self.engine.session_data.trials
        if trials_done:
            acc = sum(t.is_correct for t in trials_done) / len(trials_done)
            acc_surf = font_small.render(f"Accuracy: {acc:.1%}", True, SCORE_COLOR)
            screen.blit(acc_surf, (self.cfg.window_width - 120, 14))

        # --- Timer bar ---
        bar_w = self.cfg.window_width - 40
        bar_h = 5
        bar_x, bar_y = 20, 68
        ratio = max(0, 1.0 - elapsed / time_window)
        if ratio > 0.5:
            bar_color = TIMER_OK
        elif ratio > 0.25:
            bar_color = TIMER_WARN
        else:
            bar_color = TIMER_CRIT

        pygame.draw.rect(screen, ui.PANEL2, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        pygame.draw.rect(screen, bar_color, (bar_x, bar_y, int(bar_w * ratio), bar_h), border_radius=4)

        # --- Stimulus ---
        is_target = (shape == rule)
        color = TARGET_COLOR if is_target else DISTRACT_COLOR
        SHAPE_DRAWERS[shape](screen, color, sx, sy, radius)

        # Shape label near stimulus (helpful for learning)
        lbl = font_small.render(shape, True, TEXT_COLOR)
        screen.blit(lbl, (sx - lbl.get_width() // 2, sy + radius + 8))

        # --- Footer hint ---
        hint = font_small.render("Click shape  |  SPACE = skip  |  ESC = quit", True, ui.DIM)
        screen.blit(hint, (self.cfg.window_width // 2 - hint.get_width() // 2,
                            self.cfg.window_height - 28))

        self.viewport.present()

    def _flash_feedback(self, screen, is_correct: bool, clock) -> None:
        """Brief 200 ms coloured flash indicating correct/incorrect."""
        color = ui.GREEN if is_correct else ui.RED
        pygame.draw.rect(screen, ui.PANEL2, (20, 98, 170, 30), border_radius=4)
        ui.text(screen, "Correct response" if is_correct else "Incorrect response", ui.font(15), color, 30, 102)
        self.viewport.present()
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
            ("Your reaction time and accuracy are tracked.", font_med, ui.DIM),
            ("Try to stay focused — cognitive drift is being measured.", font_med, ui.DIM),
            ("", font_med, TEXT_COLOR),
            ("Press ENTER to begin", font_large, SCORE_COLOR),
        ]
        return self._render_text_screen(screen, lines, clock, wait_key=pygame.K_RETURN)

    def _show_outro(self, screen, font_large, font_med, clock) -> None:
        trials = self.engine.session_data.trials
        acc = sum(t.is_correct for t in trials) / max(1, len(trials))
        avg_rt = sum(t.reaction_time for t in trials) / max(1, len(trials))
        lines = [
            ("Session complete", font_large, RULE_COLOR),
            ("", font_med, TEXT_COLOR),
            (f"Trials completed : {len(trials)}", font_med, TEXT_COLOR),
            (f"Overall accuracy : {acc:.1%}", font_med, SCORE_COLOR),
            (f"Avg reaction time: {avg_rt:.3f}s", font_med, SCORE_COLOR),
            ("", font_med, TEXT_COLOR),
            ("Session complete. Press ENTER to save and return.", font_med, ui.DIM),
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
            for raw_event in pygame.event.get():
                event = self.viewport.event(raw_event)
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return "quit"
                if wait_key and event.type == pygame.KEYDOWN and event.key == wait_key:
                    return "ok"

            ui.session_message(screen, lines)
            self.viewport.present()
            clock.tick(self.cfg.fps)
