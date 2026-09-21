"""Run the attention task with live risk estimates and session recording.

Run `python -m driftsync.realtime.live_simulator --model transformer`
to select the Transformer instead of the default LSTM.
"""

import argparse
import json
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame

from driftsync import ui
from driftsync.realtime.presentation import render_live

from driftsync.configs import SimulatorConfig, RealtimeConfig, CONFIG
from driftsync.simulator.task_engine import TaskEngine
from driftsync.simulator.scenarios import SCENARIOS, apply_scenario
from driftsync.realtime.inference_engine import RealtimeInferenceEngine
from driftsync.utils import get_logger, compute_lead_time_metrics

logger = get_logger(__name__)

BG_COLOR, TEXT_COLOR, RULE_COLOR = ui.BG, ui.TEXT, ui.ACCENT
TARGET_COLOR, DISTRACT_COLOR = ui.GREEN, ui.RED
TIMER_OK, TIMER_WARN, TIMER_CRIT = ui.ACCENT, ui.YELLOW, ui.RED
GAUGE_LOW, GAUGE_MED, GAUGE_HIGH = ui.GREEN, ui.YELLOW, ui.RED
WARNING_BG = (*ui.RED, 30)
UNCERTAINTY_COL = ui.DIM


# Shape drawing (same as simulator.gui)

def draw_circle(surface, color, x, y, r):
    pygame.draw.circle(surface, color, (x, y), r)
    pygame.draw.circle(surface, (255, 255, 255), (x, y), r, 2)

def draw_square(surface, color, x, y, r):
    rect = pygame.Rect(x - r, y - r, r * 2, r * 2)
    pygame.draw.rect(surface, color, rect, border_radius=6)
    pygame.draw.rect(surface, (255, 255, 255), rect, 2, border_radius=6)

def draw_triangle(surface, color, x, y, r):
    pts = [(x, y - r), (x - int(r * 0.866), y + r // 2), (x + int(r * 0.866), y + r // 2)]
    pygame.draw.polygon(surface, color, pts)
    pygame.draw.polygon(surface, (255, 255, 255), pts, 2)

SHAPE_DRAWERS = {"CIRCLE": draw_circle, "SQUARE": draw_square, "TRIANGLE": draw_triangle}


def draw_drift_gauge(
    screen: pygame.Surface,
    x: int, y: int, w: int, h: int,
    probability: float,
    uncertainty: float,
    font,
) -> None:
    """Render a horizontal probability gauge bar with uncertainty shading."""
    pygame.draw.rect(screen, (30, 30, 45), (x, y, w, h), border_radius=4)

    if probability < 0.4:
        bar_color = GAUGE_LOW
    elif probability < 0.65:
        bar_color = GAUGE_MED
    else:
        bar_color = GAUGE_HIGH

    fill_w = int(w * probability)
    if fill_w > 0:
        pygame.draw.rect(screen, bar_color, (x, y, fill_w, h), border_radius=4)

    # Uncertainty shading (semi-transparent bracket)
    lo = max(0, probability - uncertainty)
    hi = min(1, probability + uncertainty)
    unc_x = x + int(w * lo)
    unc_w = int(w * (hi - lo))
    if unc_w > 1:
        unc_surf = pygame.Surface((unc_w, h), pygame.SRCALPHA)
        unc_surf.fill((180, 130, 255, 60))
        screen.blit(unc_surf, (unc_x, y))

    # Threshold marker
    thresh_x = x + int(w * CONFIG.realtime.warning_threshold)
    pygame.draw.line(screen, (255, 255, 255), (thresh_x, y - 3), (thresh_x, y + h + 3), 2)

    pygame.draw.rect(screen, (70, 70, 90), (x, y, w, h), 2, border_radius=4)

    lbl = font.render(f"Drift P: {probability:.2f}  ±{uncertainty:.2f}", True, TEXT_COLOR)
    screen.blit(lbl, (x + w + 10, y))


def draw_sparkline(
    screen: pygame.Surface,
    history: deque,
    x: int, y: int, w: int, h: int,
    color=(100, 200, 150),
) -> None:
    """Render a tiny line chart of recent probability values."""
    vals = list(history)
    if len(vals) < 2:
        return

    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        px = x + int(i / (n - 1) * w)
        py = y + h - int(v * h)
        pts.append((px, py))

    pygame.draw.lines(screen, color, False, pts, 2)
    # Threshold line
    ty = y + h - int(CONFIG.realtime.warning_threshold * h)
    pygame.draw.line(screen, GAUGE_HIGH, (x, ty), (x + w, ty), 1)


class LiveDriftSimulator:
    """Pygame-based simulator with live cognitive drift prediction overlay."""

    def __init__(self, sim_cfg: SimulatorConfig, rt_cfg: RealtimeConfig, model_type: str = "lstm"):
        self.sim_cfg   = sim_cfg
        self.rt_cfg    = rt_cfg
        self.engine    = TaskEngine(sim_cfg)

        # Load calibration baseline if available
        try:
            from driftsync.ml.calibrator import CalibrationEngine
            self._baseline = CalibrationEngine.load()
        except Exception:
            self._baseline = None

        self.inference = RealtimeInferenceEngine(rt_cfg, CONFIG.data, baseline=self._baseline)

        # Try to load a baseline sklearn model for fallback display
        try:
            from driftsync.ml.baseline_models import get_best_available_model
            self._sklearn_model, self._sklearn_mode = get_best_available_model()
        except Exception:
            self._sklearn_model = None
            self._sklearn_mode  = "threshold"

        # Try to load rule-based explainer
        try:
            from driftsync.ml.explainer import RuleBasedExplainer
            self._explainer = RuleBasedExplainer(self._baseline)
        except Exception:
            self._explainer = None

        self._prob_history: deque = deque(maxlen=rt_cfg.display_history)
        self._unc_history: deque = deque(maxlen=rt_cfg.display_history)

        self._last_prob      = 0.0
        self._last_unc       = 0.0
        self._warning_flag   = False
        self._model_type     = model_type
        self._model_ready    = False
        self._last_feats     = None  # latest feature vector for sklearn model
        self._explanation    = []    # list of explanation strings

        # Lead time tracking
        self._warning_events: list = []
        self._error_events:   list = []

    def run(self) -> str:
        """Return a saved session path, or an empty string when setup is canceled."""
        pygame.init()
        W, H = self.sim_cfg.window_width, self.sim_cfg.window_height
        pygame.display.set_caption("DriftSync — Live Inference")
        self.viewport = ui.Viewport((W + 400, max(H, 900)), "DriftSync | Live session", (1280, 886))
        screen = self.viewport.surface
        clock  = pygame.time.Clock()

        font_large = ui.font(30, True)
        font_med   = ui.font(20)
        font_small = ui.font(14)

        ui.session_message(screen, [("Loading prediction model", font_large, ui.TEXT),
                                   ("Preparing the live workspace...", font_med, ui.DIM)])
        self.viewport.present()
        pygame.event.pump()
        try:
            self.inference.load_model(self._model_type)
            self._model_ready = True
            logger.info("Model loaded — live inference active.")
        except FileNotFoundError:
            logger.warning(
                "No trained model found. Running the task without live risk scoring. "
                "Train quickly with: python run_experiment.py --quick"
            )

        begin = self._show_intro(screen, font_large, font_med, clock)
        if begin == "quit":
            pygame.quit()
            return ""

        while not self.engine.is_finished:
            stimulus = self.engine.next_stimulus()
            result = self._run_trial(screen, clock, stimulus, font_large, font_med, font_small)
            if result == "quit":
                break

        lead_metrics = compute_lead_time_metrics(self._warning_events, self._error_events)
        if not self.engine.session_data.trials:
            pygame.quit()
            return ""
        self._show_outro(screen, font_large, font_med, clock, lead_metrics)
        pygame.quit()

        path = self.engine.save_session()
        if self._model_ready and self.inference._log:
            self.inference.save_log()
        self._save_session_metrics(str(path), lead_metrics)
        return str(path)


    def _run_trial(self, screen, clock, stimulus, font_large, font_med, font_small) -> str:
        shape      = stimulus["shape"]
        sx, sy     = stimulus["x"], stimulus["y"]
        rule       = stimulus["rule"]
        time_window= stimulus["time_window"]
        radius     = self.sim_cfg.target_radius

        trial_start = time.time()
        action = None
        rt = 0.0

        while True:
            elapsed = time.time() - trial_start
            if elapsed >= time_window:
                rt, action = time_window, "timeout"
                break

            for raw_event in pygame.event.get():
                event = self.viewport.event(raw_event)
                if event.type == pygame.QUIT:
                    self.engine.record_trial(shape, "timeout", elapsed)
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "quit"
                    if event.key == pygame.K_SPACE:
                        rt, action = elapsed, "skip"
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if math.hypot(mx - sx, my - sy) <= radius * 1.3:
                        rt, action = elapsed, "click"

            if action is not None:
                break

            self._render(screen, font_large, font_med, font_small,
                         shape, sx, sy, radius, rule, elapsed, time_window)
            clock.tick(self.sim_cfg.fps)

        trial = self.engine.record_trial(shape, action, rt)

        now = time.time()

        if self._model_ready:
            prob, unc, warn = self.inference.update(
                reaction_time=rt,
                is_correct=trial.is_correct,
                stimulus_shape=shape,
                target_shape=rule,
                action=action,
            )
            self._last_prob    = prob
            self._last_unc     = unc
            self._warning_flag = warn
            if len(self.inference._feat_buffer) >= self.inference._seq_len:
                self._prob_history.append(prob)
                self._unc_history.append(unc)

            # Update last feature vector for sklearn model display
            feat_buf = list(self.inference._feat_buffer)
            if feat_buf:
                self._last_feats = feat_buf[-1]
        else:
            # Threshold / sklearn fallback
            feat_buf = list(self.inference._feat_buffer)
            if feat_buf:
                self._last_feats = feat_buf[-1]
                if self._sklearn_model is not None:
                    self._last_prob    = self._sklearn_model.predict_proba(feat_buf[-1])
                    self._warning_flag = self._last_prob >= self.rt_cfg.warning_threshold
                    self._prob_history.append(self._last_prob)

        # Track warnings and errors for lead time
        trial_idx = self.engine.trial_count - 1
        if self._warning_flag:
            self._warning_events.append({"trial_idx": trial_idx, "timestamp": now, "probability": self._last_prob})
        if not trial.is_correct:
            self._error_events.append({"trial_idx": trial_idx, "timestamp": now})

        # Generate explanation when risk is elevated
        if self._last_prob >= 0.40 and self._explainer is not None:
            live_stats       = self.inference.get_live_stats()
            self._explanation = self._explainer.format_panel(live_stats, self._last_prob)
        elif self._last_prob < 0.35:
            self._explanation = []

        self._flash_feedback(screen, trial.is_correct, clock)
        return "ok"


    def _render(self, screen, font_large, font_med, font_small,
                shape, sx, sy, radius, rule, elapsed, time_window):
        render_live(self, screen, shape, sx, sy, radius, rule, elapsed, time_window, SHAPE_DRAWERS)
        self.viewport.present()

    def _flash_feedback(self, screen, is_correct, clock):
        color = ui.GREEN if is_correct else ui.RED
        pygame.draw.rect(screen, ui.PANEL2, (28, 112, 165, 30), border_radius=4)
        ui.text(screen, "Correct response" if is_correct else "Incorrect response", ui.font(14), color, 38, 117)
        self.viewport.present()
        t = time.time()
        while time.time() - t < 0.15:
            pygame.event.pump()
            clock.tick(self.sim_cfg.fps)

    def _show_intro(self, screen, font_large, font_med, clock):
        W, H = screen.get_size()
        model_status = (
            f"Model: {self._model_type.upper()}"
            if self._model_ready
            else "No checkpoint loaded - task recording only"
        )
        lines = [
            ("DriftSync — Live Inference Mode", font_large, RULE_COLOR),
            ("", font_med, TEXT_COLOR),
            (f"{model_status}", font_med, (160, 255, 160) if self._model_ready else GAUGE_HIGH),
            ("", font_med, TEXT_COLOR),
            ("The drift probability gauge updates after each trial.", font_med, TEXT_COLOR),
            ("Risk and model uncertainty are shown separately.", font_med, GAUGE_HIGH),
            ("", font_med, TEXT_COLOR),
            ("Press ENTER to begin", font_large, (100, 220, 255)),
        ]
        waiting = True
        while waiting:
            for raw_event in pygame.event.get():
                event = self.viewport.event(raw_event)
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    waiting = False
            ui.session_message(screen, lines)
            self.viewport.present()
            clock.tick(60)

    def _show_outro(self, screen, font_large, font_med, clock, lead_metrics: dict = None):
        trials = self.engine.session_data.trials
        acc    = sum(t.is_correct for t in trials) / max(1, len(trials))
        avg_rt = sum(t.reaction_time for t in trials) / max(1, len(trials))
        W, H   = screen.get_size()
        lm     = lead_metrics or {}

        predicted  = lm.get("n_predicted_before", 0)
        missed     = lm.get("n_missed", 0)
        avg_lead   = lm.get("avg_lead_time_s", 0.0)
        fp_warns   = lm.get("false_positive_warnings", 0)

        lines = [
            ("Session Complete!", font_large, RULE_COLOR),
            ("", font_med, TEXT_COLOR),
            (f"Trials: {len(trials)}   Accuracy: {acc:.1%}   Avg RT: {avg_rt:.3f}s", font_med, TEXT_COLOR),
            ("", font_med, TEXT_COLOR),
            ("Prediction Lead Time", font_med, (100, 220, 255)),
            (f"Errors predicted early: {predicted}   Missed: {missed}", font_med, TEXT_COLOR),
            (f"Avg lead time: {avg_lead:.2f}s   False warnings: {fp_warns}", font_med, TEXT_COLOR),
            ("", font_med, TEXT_COLOR),
            ("Press Enter to save session data and metrics.", font_med, (160, 200, 160)),
            ("Press ENTER or close window.", font_med, TEXT_COLOR),
        ]
        t0 = time.time()
        waiting = True
        while waiting and time.time() - t0 < 15:
            for raw_event in pygame.event.get():
                event = self.viewport.event(raw_event)
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    waiting = False
            ui.session_message(screen, lines)
            self.viewport.present()
            clock.tick(60)

    def _save_session_metrics(self, session_path: str, lead_metrics: dict) -> None:
        """Save session summary including lead time metrics to JSON."""
        trials  = self.engine.session_data.trials
        acc     = sum(t.is_correct for t in trials) / max(1, len(trials))
        avg_rt  = sum(t.reaction_time for t in trials) / max(1, len(trials))
        summary = {
            "session_id":    self.engine.session_id,
            "session_path":  session_path,
            "timestamp":     datetime.now().isoformat(),
            "n_trials":      len(trials),
            "accuracy":      round(acc, 4),
            "avg_rt":        round(avg_rt, 4),
            "model_mode":    self._model_type if self._model_ready else self._sklearn_mode,
            "calibrated":    self._baseline is not None,
            **lead_metrics,
        }
        out_dir = Path("driftsync/sessions")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"metrics_{self.engine.session_id}.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Session metrics saved -> %s", out_path)


def build_simulator_config(num_trials: int, scenario: str | None = None) -> SimulatorConfig:
    """Build the live simulator config, optionally applying a showcase scenario."""
    cfg = SimulatorConfig(num_trials=num_trials)
    if scenario:
        cfg = apply_scenario(scenario, cfg)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DriftSync live inference simulator.")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--scenario", type=str, default=None, choices=sorted(SCENARIOS))
    args = parser.parse_args()

    sim_cfg = build_simulator_config(args.trials, args.scenario)
    rt_cfg  = RealtimeConfig(model_type=args.model)
    if args.threshold is not None:
        rt_cfg.warning_threshold = args.threshold

    sim = LiveDriftSimulator(sim_cfg, rt_cfg, model_type=args.model)
    session_path = sim.run()
    logger.info("Session saved to: %s", session_path)


if __name__ == "__main__":
    main()
