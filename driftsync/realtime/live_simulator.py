"""
Live Inference Simulator
=========================
Integrates the task simulator with the real-time inference engine.
At every trial completion:
  1. The inference engine receives the new observation.
  2. P(error_next_K) and uncertainty are computed.
  3. If P > threshold, a coloured drift warning overlay is shown.

The HUD shows a live "Drift Probability" gauge that updates every trial.

Usage
-----
    python -m driftsync.realtime.live_simulator
    python -m driftsync.realtime.live_simulator --model transformer
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

from driftsync.configs import SimulatorConfig, RealtimeConfig, CONFIG
from driftsync.simulator.task_engine import TaskEngine
from driftsync.realtime.inference_engine import RealtimeInferenceEngine
from driftsync.utils import get_logger, compute_lead_time_metrics

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
BG_COLOR        = (15,  15,  25)
TEXT_COLOR      = (220, 220, 220)
RULE_COLOR      = (100, 220, 255)
TARGET_COLOR    = (80,  200, 120)
DISTRACT_COLOR  = (220,  80,  80)
TIMER_OK        = (80,  200, 120)
TIMER_WARN      = (240, 180,  50)
TIMER_CRIT      = (220,  60,  60)
GAUGE_LOW       = (80,  200, 120)
GAUGE_MED       = (240, 180,  50)
GAUGE_HIGH      = (220,  60,  60)
WARNING_BG      = (200,  40,  40, 80)
UNCERTAINTY_COL = (180, 130, 255)


# ---------------------------------------------------------------------------
# Shape drawing (same as simulator.gui)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Gauge rendering
# ---------------------------------------------------------------------------

def draw_drift_gauge(
    screen: pygame.Surface,
    x: int, y: int, w: int, h: int,
    probability: float,
    uncertainty: float,
    font,
) -> None:
    """Render a horizontal probability gauge bar with uncertainty shading."""
    # Background
    pygame.draw.rect(screen, (30, 30, 45), (x, y, w, h), border_radius=4)

    # Fill colour
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

    # Border
    pygame.draw.rect(screen, (70, 70, 90), (x, y, w, h), 2, border_radius=4)

    # Label
    lbl = font.render(f"Drift P: {probability:.2f}  ±{uncertainty:.2f}", True, TEXT_COLOR)
    screen.blit(lbl, (x + w + 10, y))


# ---------------------------------------------------------------------------
# History sparkline
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main live simulator
# ---------------------------------------------------------------------------

class LiveDriftSimulator:
    """
    Pygame-based simulator with live cognitive drift prediction overlay.
    """

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
        self._prob_history.append(0.0)

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
        """Run live simulator. Returns path to saved session."""
        pygame.init()
        W, H = self.sim_cfg.window_width, self.sim_cfg.window_height
        pygame.display.set_caption("DriftSync — Live Inference")
        screen = pygame.display.set_mode((W, H))
        clock  = pygame.time.Clock()

        font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        font_med   = pygame.font.SysFont("Consolas", 20)
        font_small = pygame.font.SysFont("Consolas", 14)

        # Load model (non-blocking attempt)
        try:
            self.inference.load_model(self._model_type)
            self._model_ready = True
            logger.info("Model loaded — live inference active.")
        except FileNotFoundError:
            logger.warning(
                "No trained model found. Running the task without live risk scoring. "
                "Train quickly with: python run_experiment.py --quick"
            )

        self._show_intro(screen, font_large, font_med, clock)

        while not self.engine.is_finished:
            stimulus = self.engine.next_stimulus()
            result = self._run_trial(screen, clock, stimulus, font_large, font_med, font_small)
            if result == "quit":
                break

        lead_metrics = compute_lead_time_metrics(self._warning_events, self._error_events)
        self._show_outro(screen, font_large, font_med, clock, lead_metrics)
        pygame.quit()

        path = self.engine.save_session()
        if self._model_ready:
            self.inference.save_log()
        self._save_session_metrics(str(path), lead_metrics)
        return str(path)

    # ------------------------------------------------------------------

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

            for event in pygame.event.get():
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

        # Feed to inference engine
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
            self._prob_history.append(prob)

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

    # ------------------------------------------------------------------

    def _render(self, screen, font_large, font_med, font_small,
                shape, sx, sy, radius, rule, elapsed, time_window):
        screen.fill(BG_COLOR)

        # Warning overlay
        if self._warning_flag:
            overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            overlay.fill(WARNING_BG)
            screen.blit(overlay, (0, 0))
            warn_surf = font_large.render("⚠  COGNITIVE DRIFT DETECTED", True, GAUGE_HIGH)
            screen.blit(warn_surf, (
                screen.get_width() // 2 - warn_surf.get_width() // 2,
                screen.get_height() - 75
            ))

        # Rule
        rule_surf = font_large.render(f"CLICK  {rule}S", True, RULE_COLOR)
        screen.blit(rule_surf, (screen.get_width() // 2 - rule_surf.get_width() // 2, 12))

        # Trial counter
        t_surf = font_small.render(
            f"Trial {self.engine.trial_count + 1}/{self.sim_cfg.num_trials}", True, TEXT_COLOR
        )
        screen.blit(t_surf, (12, 14))

        # Timer bar
        W = screen.get_width()
        bar_w = W - 40
        ratio = 1.0 - elapsed / time_window
        bar_color = TIMER_OK if ratio > 0.5 else TIMER_WARN if ratio > 0.25 else TIMER_CRIT
        pygame.draw.rect(screen, (40, 40, 55), (20, 55, bar_w, 10), border_radius=4)
        if ratio > 0:
            pygame.draw.rect(screen, bar_color, (20, 55, int(bar_w * ratio), 10), border_radius=4)

        # Drift gauge
        if self._model_ready:
            draw_drift_gauge(
                screen, 20, 75, bar_w - 200, 18,
                self._last_prob, self._last_unc, font_small,
            )
            # Sparkline
            draw_sparkline(screen, self._prob_history, W - 210, 75, 190, 50)
        else:
            no_model = font_small.render(
                "(no checkpoint - run: python run_experiment.py --quick)",
                True,
                (120, 128, 142),
            )
            screen.blit(no_model, (20, 78))

        # Stimulus
        is_target = (shape == rule)
        SHAPE_DRAWERS[shape](screen, TARGET_COLOR if is_target else DISTRACT_COLOR, sx, sy, radius)
        lbl = font_small.render(shape, True, TEXT_COLOR)
        screen.blit(lbl, (sx - lbl.get_width() // 2, sy + radius + 8))

        # Explanation panel (shown when risk >= 0.40)
        if self._explanation:
            exp_y = screen.get_height() - 22 - (len(self._explanation) * 16) - 10
            bg = pygame.Surface((W - 40, len(self._explanation) * 16 + 8), pygame.SRCALPHA)
            bg.fill((20, 20, 35, 180))
            screen.blit(bg, (20, exp_y - 4))
            for i, line in enumerate(self._explanation):
                col = GAUGE_HIGH if i == 0 else TEXT_COLOR
                s = font_small.render(line, True, col)
                screen.blit(s, (24, exp_y + i * 16))

        # Model mode label
        mode_lbl = self._sklearn_mode if not self._model_ready else self._model_type.upper()
        mode_surf = font_small.render(f"Mode: {mode_lbl}", True, (80, 80, 100))
        screen.blit(mode_surf, (W - mode_surf.get_width() - 8, screen.get_height() - 22))

        # Footer
        hint = font_small.render("Click shape  |  SPACE = skip  |  ESC = quit", True, (70, 70, 80))
        screen.blit(hint, (W // 2 - hint.get_width() // 2, screen.get_height() - 22))

        pygame.display.flip()

    def _flash_feedback(self, screen, is_correct, clock):
        color = (40, 180, 80) if is_correct else (180, 40, 40)
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        overlay.fill((*color, 70))
        screen.blit(overlay, (0, 0))
        pygame.display.flip()
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
            ("A RED overlay means high error risk predicted.", font_med, GAUGE_HIGH),
            ("", font_med, TEXT_COLOR),
            ("Press ENTER to begin", font_large, (100, 220, 255)),
        ]
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    waiting = False
            screen.fill(BG_COLOR)
            total_h = sum(fnt.get_height() + 4 for _, fnt, _ in lines)
            y = (H - total_h) // 2
            for text, fnt, color in lines:
                surf = fnt.render(text, True, color)
                screen.blit(surf, (W // 2 - surf.get_width() // 2, y))
                y += fnt.get_height() + 4
            pygame.display.flip()
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
            ("Data and session metrics saved.", font_med, (160, 200, 160)),
            ("Press ENTER or close window.", font_med, TEXT_COLOR),
        ]
        t0 = time.time()
        waiting = True
        while waiting and time.time() - t0 < 15:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    waiting = False
            screen.fill(BG_COLOR)
            total_h = sum(fnt.get_height() + 4 for _, fnt, _ in lines)
            y = (H - total_h) // 2
            for text, fnt, color in lines:
                surf = fnt.render(text, True, color)
                screen.blit(surf, (W // 2 - surf.get_width() // 2, y))
                y += fnt.get_height() + 4
            pygame.display.flip()
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run DriftSync live inference simulator.")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    sim_cfg = SimulatorConfig(num_trials=args.trials)
    rt_cfg  = RealtimeConfig(model_type=args.model)
    if args.threshold is not None:
        rt_cfg.warning_threshold = args.threshold

    sim = LiveDriftSimulator(sim_cfg, rt_cfg, model_type=args.model)
    session_path = sim.run()
    logger.info("Session saved to: %s", session_path)


if __name__ == "__main__":
    main()
