"""
DriftSync Interactive Application
===================================
Full Pygame GUI for the DriftSync cognitive drift prediction system.

Screens
-------
  SPLASH    - Animated intro (auto-advances after 2.5 s)
  MENU      - Main navigation hub
  LEARN     - 7-page educational guide explaining everything
  DEMO      - Full ML pipeline runs automatically with live feedback
  RESULTS   - Gallery of generated plots + metric comparison table
  PLAY_TASK - Launch interactive human task simulator
  LIVE_MODE - Launch live AI inference simulator

This is a single-file Pygame state machine. All UI components are
defined here for maximum portability.
"""

import io
import json
import logging
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pygame

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class State(Enum):
    SPLASH    = auto()
    MENU      = auto()
    LEARN     = auto()
    DEMO      = auto()
    RESULTS   = auto()
    PLAY_TASK = auto()
    LIVE_MODE = auto()


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

BG      = (10,  10,  20)
PANEL   = (22,  22,  40)
PANEL2  = (28,  28,  52)
BORDER  = (50,  50,  85)
ACCENT  = (100, 220, 255)
ACCENT2 = (150, 100, 255)
GREEN   = (80,  200, 120)
YELLOW  = (240, 180,  50)
RED     = (220,  60,  60)
TEXT    = (220, 220, 230)
DIM     = (120, 120, 145)
WHITE   = (255, 255, 255)
LSTM_C  = (80,  220, 170)
TF_C    = (180,  80, 255)

W, H = 1150, 740
FPS  = 60


# ---------------------------------------------------------------------------
# Logging queue handler — captures log output for the DEMO screen
# ---------------------------------------------------------------------------

class QueueLogHandler(logging.Handler):
    """Redirect log records to a thread-safe queue."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.log_queue.put_nowait(msg)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# UI component helpers
# ---------------------------------------------------------------------------

def draw_rect_border(surface: pygame.Surface, rect, color, radius: int = 8, width: int = 2) -> None:
    pygame.draw.rect(surface, color, rect, width, border_radius=radius)


def draw_filled_rect(surface: pygame.Surface, rect, color, radius: int = 8) -> None:
    pygame.draw.rect(surface, color, rect, border_radius=radius)


def draw_text(surface: pygame.Surface, text: str, font, color, x: int, y: int,
              anchor: str = "topleft") -> pygame.Rect:
    surf = font.render(text, True, color)
    rect = surf.get_rect()
    setattr(rect, anchor, (x, y))
    surface.blit(surf, rect)
    return rect


def draw_wrapped_text(
    surface: pygame.Surface,
    text: str,
    font,
    color,
    rect: pygame.Rect,
    line_spacing: int = 6,
) -> int:
    """
    Word-wrap `text` inside `rect`. Returns the Y position after the last line.
    """
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        if font.size(test)[0] <= rect.width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    y = rect.top
    for line in lines:
        if y + font.get_height() > rect.bottom:
            break
        surf = font.render(line, True, color)
        surface.blit(surf, (rect.left, y))
        y += font.get_height() + line_spacing
    return y


def draw_progress_bar(
    surface: pygame.Surface,
    rect: pygame.Rect,
    value: float,         # 0.0 – 1.0
    fg_color=GREEN,
    bg_color=PANEL2,
    radius: int = 5,
) -> None:
    draw_filled_rect(surface, rect, bg_color, radius)
    if value > 0:
        fill = rect.copy()
        fill.width = max(1, int(rect.width * min(value, 1.0)))
        draw_filled_rect(surface, fill, fg_color, radius)
    draw_rect_border(surface, rect, BORDER, radius, 1)


def draw_sparkline(
    surface: pygame.Surface,
    data: list,
    rect: pygame.Rect,
    color=ACCENT,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> None:
    """Draw a miniature line chart inside `rect`."""
    if len(data) < 2:
        return
    y_range = max(y_max - y_min, 1e-6)
    n = len(data)
    pts = []
    for i, v in enumerate(data):
        px = rect.left + int(i / (n - 1) * rect.width)
        py = rect.bottom - int((v - y_min) / y_range * rect.height)
        py = max(rect.top, min(rect.bottom, py))
        pts.append((px, py))
    pygame.draw.lines(surface, color, False, pts, 2)


def load_png_surface(path: str | Path, size: Tuple[int, int] | None = None) -> Optional[pygame.Surface]:
    """Load a PNG file into a pygame Surface, optionally scaled."""
    try:
        surf = pygame.image.load(str(path))
        if size:
            surf = pygame.transform.smoothscale(surf, size)
        return surf
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Button component
# ---------------------------------------------------------------------------

class Button:
    """Clickable button with hover and disabled states."""

    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        font,
        color=ACCENT,
        text_color=BG,
        hover_color=WHITE,
        radius: int = 10,
        sub_text: str = "",
        sub_font=None,
    ):
        self.rect        = rect
        self.text        = text
        self.sub_text    = sub_text
        self.font        = font
        self.sub_font    = sub_font
        self.color       = color
        self.text_color  = text_color
        self.hover_color = hover_color
        self.radius      = radius
        self.hovered     = False
        self.disabled    = False

    def draw(self, surface: pygame.Surface) -> None:
        if self.disabled:
            base = tuple(min(255, c + 10) for c in PANEL2)
            tc   = DIM
        elif self.hovered:
            base = self.hover_color
            tc   = BG
        else:
            base = self.color
            tc   = self.text_color

        draw_filled_rect(surface, self.rect, base, self.radius)
        draw_rect_border(surface, self.rect, BORDER, self.radius, 1)

        # Main text
        cx = self.rect.centerx
        cy = self.rect.centery if not self.sub_text else self.rect.centery - 10
        draw_text(surface, self.text, self.font, tc, cx, cy, anchor="center")

        # Sub-text
        if self.sub_text and self.sub_font:
            draw_text(surface, self.sub_text, self.sub_font, DIM if not self.hovered else (80, 80, 80),
                      cx, cy + self.font.get_height() + 2, anchor="center")

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Returns True if this button was clicked."""
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self.disabled:
            if self.rect.collidepoint(event.pos):
                return True
        return False


# ---------------------------------------------------------------------------
# Educational content (7 pages)
# ---------------------------------------------------------------------------

LEARN_PAGES = [
    {
        "title": "1. What Is Cognitive Drift?",
        "body": (
            "Cognitive drift is the gradual decline in mental performance during sustained tasks. "
            "As your brain fatigues, your reaction times slow down, errors creep in, and your attention "
            "becomes scattered. This happens to pilots, surgeons, air traffic controllers, and anyone "
            "performing repetitive cognitive work.\n\n"
            "Drift is not random — it follows measurable temporal patterns. Early in a session, "
            "performance is sharp. After 20-30 minutes, subtle errors begin. After an hour, drift "
            "becomes significant and potentially dangerous.\n\n"
            "DriftSync asks: can an AI model learn these patterns from your interaction history "
            "and predict WHEN your next mistake will happen — before it happens?"
        ),
        "visual": "drift_curve",   # rendered by the app
    },
    {
        "title": "2. How the Task Works",
        "body": (
            "You play a rapid shape-classification game:\n\n"
            "  A shape appears on screen: Circle, Square, or Triangle.\n"
            "  A rule is shown at the top: e.g. \"CLICK CIRCLES\".\n"
            "  CLICK the shape if it matches the rule.\n"
            "  Press SPACE to skip if it does NOT match.\n"
            "  You have a shrinking time window (3s early, 1.5s late).\n\n"
            "Your reaction time and correctness are recorded for every trial. "
            "The time window gets shorter as the session progresses, simulating increasing "
            "cognitive load. The rule changes every 20 trials to prevent memorisation.\n\n"
            "After playing, the AI processes your trial history to learn your drift pattern."
        ),
        "visual": "task_demo",
    },
    {
        "title": "3. The 11 AI Input Features",
        "body": (
            "For each trial, the AI receives 11 numbers describing your recent performance:\n\n"
            "  1. Reaction Time (normalised)  — how fast you responded\n"
            "  2. Correctness                 — 1.0 = right, 0.0 = error\n"
            "  3. Elapsed Time (normalised)   — how long you've been playing\n"
            "  4. Rolling Error Rate (5)      — errors in last 5 trials\n"
            "  5. Rolling Error Rate (10)     — errors in last 10 trials\n"
            "  6. Inter-Trial Interval        — time gap between trials\n"
            "  7. Cumulative Error Rate       — total errors so far\n"
            "  8. Correct Streak              — consecutive correct answers\n"
            "  9. Error Streak                — consecutive wrong answers\n"
            " 10. Target Match                — was the right shape shown?\n"
            " 11. Action                      — did you click or skip?\n\n"
            "The AI looks at the last 20 trials at once, forming a sequence of shape (20, 11)."
        ),
        "visual": "features",
    },
    {
        "title": "4. The LSTM Model",
        "body": (
            "LSTM (Long Short-Term Memory) is a type of recurrent neural network designed "
            "for sequential data. Unlike a regular network, it has memory gates that decide "
            "what to remember and what to forget across time.\n\n"
            "Architecture:\n"
            "  Input (20 trials x 11 features)\n"
            "    -> Linear projection to 128 dimensions\n"
            "    -> 3 stacked LSTM layers with residual connections\n"
            "    -> Layer normalisation between each layer\n"
            "    -> Last hidden state -> classification head\n"
            "    -> Sigmoid -> P(error in next 5 steps)\n\n"
            "LSTM processes the sequence step-by-step, left-to-right. "
            "Each trial's output depends on the current input AND the hidden state "
            "carried forward from all previous trials. This makes it naturally suited "
            "for modelling cognitive fatigue."
        ),
        "visual": "lstm_arch",
    },
    {
        "title": "5. The Transformer Model",
        "body": (
            "Transformers use self-attention: rather than processing trials one-by-one, "
            "they look at ALL 20 trials simultaneously and learn which ones matter most "
            "for predicting the next mistake.\n\n"
            "Architecture:\n"
            "  Input (20 x 11)\n"
            "    -> Linear projection to 128 dimensions\n"
            "    -> Sinusoidal positional encoding (encodes time order)\n"
            "    -> 4 encoder layers (each: multi-head attention + FFN)\n"
            "    -> Global average pool over the 20 timesteps\n"
            "    -> Classification head -> P(error in next 5 steps)\n\n"
            "Attention Heatmap: After training, you can visualise which past trials "
            "the model 'looked at' most. High attention to recent error-heavy trials "
            "indicates the model learned to track cognitive load patterns."
        ),
        "visual": "transformer_arch",
    },
    {
        "title": "6. Predictions & Uncertainty",
        "body": (
            "The model outputs P(error) — a probability between 0.0 and 1.0:\n\n"
            "  0.0 - 0.3   Low risk. You're performing well.\n"
            "  0.3 - 0.65  Moderate risk. Slight drift detected.\n"
            "  0.65 - 1.0  HIGH RISK. Warning triggered.\n\n"
            "Monte Carlo Dropout (uncertainty estimation):\n"
            "The model runs 50 forward passes with random dropout active, "
            "producing 50 slightly different predictions. The standard deviation "
            "of these predictions is the UNCERTAINTY. High uncertainty means "
            "the model is unsure — possibly because the input pattern is unusual.\n\n"
            "A warning fires if EITHER:\n"
            "  - P(error) > 0.65  (high probability)\n"
            "  - Uncertainty > 0.20  (high model confusion)\n\n"
            "Calibration (ECE): measures how well the model's confidence matches "
            "real-world accuracy. Lower ECE = better calibrated = more trustworthy."
        ),
        "visual": "probability_gauge",
    },
    {
        "title": "7. Reading the Results",
        "body": (
            "After training, these plots are generated in driftsync/results/:\n\n"
            "  ROC Curve: Shows the tradeoff between catching real errors (True "
            "Positive Rate) and false alarms (False Positive Rate). AUC closer "
            "to 1.0 = better model. Random guessing = 0.5.\n\n"
            "  Confusion Matrix: 2x2 grid — actual vs predicted errors. "
            "Top-right = missed errors, bottom-left = false alarms.\n\n"
            "  Calibration Plot (Reliability Diagram): Perfect model follows the "
            "diagonal. Points above = underconfident, below = overconfident.\n\n"
            "  Attention Heatmap (Transformer only): Shows which past trials "
            "the model attended to. Bright diagonal = local attention, "
            "bright top-right = global drift pattern detection.\n\n"
            "  Training History: Loss and accuracy curves showing learning progress. "
            "Good training shows decreasing loss and increasing accuracy."
        ),
        "visual": "results_guide",
    },
]


# ---------------------------------------------------------------------------
# Background worker for the DEMO screen
# ---------------------------------------------------------------------------

class DemoWorker:
    """
    Runs the full ML pipeline in a daemon thread.

    The main thread reads from log_queue and metric_queue each frame.
    """

    STEPS = [
        "Generating synthetic training data",
        "Preprocessing features & labels",
        "Training LSTM model",
        "Training Transformer model",
        "Comparing models & generating plots",
    ]

    def __init__(self):
        self.log_queue    = queue.Queue()
        self.metric_queue = queue.Queue()   # dict: epoch metrics
        self.status       = "idle"          # idle | running | done | error
        self.error_msg    = ""
        self.current_step = 0               # 0-4
        self.total_steps  = len(self.STEPS)
        self._thread: Optional[threading.Thread] = None
        self._handler: Optional[QueueLogHandler] = None

    def start(self) -> None:
        """Launch the background training thread."""
        self.status       = "running"
        self.current_step = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def poll_logs(self) -> List[str]:
        """Drain the log queue. Call from main thread each frame."""
        lines = []
        try:
            while True:
                lines.append(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        return lines

    def poll_metrics(self) -> Optional[dict]:
        """Get the most recent epoch metric dict (or None)."""
        latest = None
        try:
            while True:
                latest = self.metric_queue.get_nowait()
        except queue.Empty:
            pass
        return latest

    # ------------------------------------------------------------------

    def _attach_logging(self) -> None:
        """Add queue handler to root driftsync logger."""
        self._handler = QueueLogHandler(self.log_queue)
        root = logging.getLogger("driftsync")
        root.addHandler(self._handler)

    def _detach_logging(self) -> None:
        if self._handler:
            logging.getLogger("driftsync").removeHandler(self._handler)

    def _log(self, msg: str) -> None:
        self.log_queue.put(msg)

    def _run(self) -> None:
        self._attach_logging()
        try:
            self._execute_pipeline()
            self.status = "done"
            self._log("")
            self._log("=" * 55)
            self._log("  Pipeline complete!  Click 'View Results' below.")
            self._log("=" * 55)
        except Exception as e:
            import traceback
            self.status    = "error"
            self.error_msg = str(e)
            self._log(f"ERROR: {e}")
            self._log(traceback.format_exc())
        finally:
            self._detach_logging()

    def _execute_pipeline(self) -> None:
        import numpy as np
        import torch

        from driftsync.configs import (
            SimulatorConfig, DataConfig, TrainingConfig,
            LSTMConfig, TransformerConfig,
        )
        from driftsync.simulator.headless_generator import generate_dataset
        from driftsync.data import (
            load_all_sessions, preprocess_all_sessions,
            build_sequences_from_df, split_data, make_dataloaders, save_processed,
        )
        from driftsync.models import build_model
        from driftsync.training.trainer import Trainer
        from driftsync.evaluation.compare import run_comparison
        from driftsync.utils import set_seed, get_device

        set_seed(42)
        device = get_device("auto")

        # ---- Step 1: Generate data ----------------------------------------
        self.current_step = 0
        self._log("Generating 15 synthetic sessions (150 trials each)...")
        sim_cfg = SimulatorConfig(num_trials=150)
        generate_dataset(num_sessions=15, cfg=sim_cfg, base_seed=42)
        self._log("Data generation complete.")

        # ---- Step 2: Preprocess --------------------------------------------
        self.current_step = 1
        self._log("")
        self._log("Preprocessing raw sessions -> feature sequences...")
        data_cfg = DataConfig(sequence_length=20, prediction_horizon=5)
        raw_df   = load_all_sessions(data_cfg.raw_data_dir)
        proc_df  = preprocess_all_sessions(raw_df, horizon=data_cfg.prediction_horizon)
        X, y     = build_sequences_from_df(proc_df, seq_len=data_cfg.sequence_length)
        save_processed(X, y, data_cfg.processed_data_dir)
        self._log(f"Sequences: {X.shape}   Positive rate: {y.mean():.1%}")
        input_dim = X.shape[2]

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)
        train_loader, val_loader, test_loader = make_dataloaders(
            X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64
        )

        train_cfg = TrainingConfig(
            max_epochs=40,
            early_stop_patience=10,
            learning_rate=1e-3,
            batch_size=64,
        )

        # ---- Step 3: Train LSTM --------------------------------------------
        self.current_step = 2
        self._log("")
        self._log("Training LSTM model (up to 40 epochs)...")

        lstm_cfg = LSTMConfig(input_dim=input_dim)
        lstm     = build_model("lstm", lstm_cfg)
        self._log(f"LSTM parameters: {lstm.count_parameters():,}")

        def lstm_epoch_cb(epoch, metrics):
            metrics["model"]  = "LSTM"
            metrics["epoch"]  = epoch
            self.metric_queue.put(metrics)

        Trainer(lstm, train_loader, val_loader, train_cfg, device).train(
            on_epoch_end=lstm_epoch_cb
        )

        # ---- Step 4: Train Transformer -------------------------------------
        self.current_step = 3
        self._log("")
        self._log("Training Transformer model (up to 40 epochs)...")

        tf_cfg = TransformerConfig(input_dim=input_dim)
        tf     = build_model("transformer", tf_cfg)
        self._log(f"Transformer parameters: {tf.count_parameters():,}")

        def tf_epoch_cb(epoch, metrics):
            metrics["model"] = "Transformer"
            metrics["epoch"] = epoch
            self.metric_queue.put(metrics)

        Trainer(tf, train_loader, val_loader, train_cfg, device).train(
            on_epoch_end=tf_epoch_cb
        )

        # ---- Step 5: Compare -----------------------------------------------
        self.current_step = 4
        self._log("")
        self._log("Comparing models and generating all plots...")
        run_comparison(data_cfg=data_cfg)
        self._log("All plots saved to driftsync/results/")


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class DriftSyncApplication:
    """
    DriftSync interactive GUI application.

    State machine: each state has a render_ and handle_event_ method.
    """

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("DriftSync — Cognitive Drift Prediction")
        self.screen = pygame.display.set_mode((W, H))
        self.clock  = pygame.time.Clock()

        # Fonts
        self.f_title  = pygame.font.SysFont("Consolas", 42, bold=True)
        self.f_head   = pygame.font.SysFont("Consolas", 26, bold=True)
        self.f_sub    = pygame.font.SysFont("Consolas", 19, bold=True)
        self.f_body   = pygame.font.SysFont("Consolas", 15)
        self.f_small  = pygame.font.SysFont("Consolas", 13)
        self.f_mono   = pygame.font.SysFont("Consolas", 13)
        self.f_btn    = pygame.font.SysFont("Consolas", 17, bold=True)
        self.f_btn_sm = pygame.font.SysFont("Consolas", 13)

        # State
        self.state          = State.SPLASH
        self.splash_start   = time.time()
        self.learn_page     = 0
        self.running        = True

        # DEMO state
        self.demo_worker    = DemoWorker()
        self.demo_logs: List[str] = []
        self.demo_metrics: dict   = {}
        self.demo_train_loss: deque  = deque(maxlen=50)
        self.demo_val_loss: deque    = deque(maxlen=50)
        self.demo_val_auc: deque     = deque(maxlen=50)
        self.demo_view_results_btn: Optional[Button] = None

        # RESULTS state
        self.result_thumbs: List[Tuple[pygame.Surface, str, Path]] = []
        self.result_scroll     = 0
        self.result_full_view: Optional[pygame.Surface] = None
        self.result_full_title = ""
        self.result_metrics: dict = {}

        # LIVE_MODE model selector
        self.live_model_choice = "lstm"   # "lstm" or "transformer"

        # Build MENU buttons
        self._build_menu_buttons()
        # Build LEARN nav buttons
        self._build_learn_buttons()

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        while self.running:
            self._handle_events()
            self._update()
            self._render()
            self.clock.tick(FPS)
        pygame.quit()

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if self.state in (State.LEARN, State.DEMO, State.RESULTS,
                                  State.PLAY_TASK, State.LIVE_MODE):
                    self.state = State.MENU
                    return
                if self.state == State.MENU:
                    self.running = False
                    return

            # Per-state event handling
            if self.state == State.MENU:
                self._handle_menu_event(event)
            elif self.state == State.LEARN:
                self._handle_learn_event(event)
            elif self.state == State.DEMO:
                self._handle_demo_event(event)
            elif self.state == State.RESULTS:
                self._handle_results_event(event)
            elif self.state == State.LIVE_MODE:
                self._handle_live_event(event)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def _update(self) -> None:
        if self.state == State.SPLASH:
            if time.time() - self.splash_start > 2.8:
                self.state = State.MENU

        elif self.state == State.DEMO:
            # Drain log queue
            new_logs = self.demo_worker.poll_logs()
            self.demo_logs.extend(new_logs)
            # Keep last 120 lines
            if len(self.demo_logs) > 120:
                self.demo_logs = self.demo_logs[-120:]

            # Drain metric queue
            m = self.demo_worker.poll_metrics()
            if m:
                self.demo_metrics = m
                if "train_loss" in m:
                    self.demo_train_loss.append(m["train_loss"])
                    self.demo_val_loss.append(m["val_loss"])
                    self.demo_val_auc.append(m.get("val_auc", 0.0))

            # Activate "View Results" button when done
            if self.demo_view_results_btn:
                self.demo_view_results_btn.disabled = (self.demo_worker.status != "done")

    # ------------------------------------------------------------------
    # Render dispatch
    # ------------------------------------------------------------------

    def _render(self) -> None:
        self.screen.fill(BG)

        if self.state == State.SPLASH:
            self._render_splash()
        elif self.state == State.MENU:
            self._render_menu()
        elif self.state == State.LEARN:
            self._render_learn()
        elif self.state == State.DEMO:
            self._render_demo()
        elif self.state == State.RESULTS:
            self._render_results()
        elif self.state == State.PLAY_TASK:
            self._render_play_task()
        elif self.state == State.LIVE_MODE:
            self._render_live_mode()

        pygame.display.flip()

    # ==================================================================
    # SPLASH
    # ==================================================================

    def _render_splash(self) -> None:
        t = time.time() - self.splash_start
        alpha = min(255, int(t / 1.0 * 255))

        # Animated title
        title = self.f_title.render("DriftSync", True, ACCENT)
        title.set_alpha(alpha)
        self.screen.blit(title, title.get_rect(center=(W // 2, H // 2 - 55)))

        sub = self.f_sub.render("Real-Time Cognitive Drift Prediction", True, ACCENT2)
        sub.set_alpha(max(0, int((t - 0.5) / 0.8 * 255)))
        self.screen.blit(sub, sub.get_rect(center=(W // 2, H // 2)))

        dot_count = int(t * 2) % 4
        dots = self.f_body.render("Loading" + "." * dot_count, True, DIM)
        dots.set_alpha(max(0, int((t - 1.2) / 0.6 * 255)))
        self.screen.blit(dots, dots.get_rect(center=(W // 2, H // 2 + 55)))

        # Subtle animated line
        if t > 0.3:
            progress = min(1.0, (t - 0.3) / 1.5)
            lw = int(W * 0.5 * progress)
            cx = W // 2
            y  = H // 2 + 90
            pygame.draw.line(self.screen, ACCENT, (cx - lw // 2, y), (cx + lw // 2, y), 2)

    # ==================================================================
    # MENU
    # ==================================================================

    def _build_menu_buttons(self) -> None:
        """Create the 6 main menu tile buttons."""
        tile_w, tile_h = 330, 120
        gap            = 18
        cols           = 3
        rows           = 2
        total_w        = cols * tile_w + (cols - 1) * gap
        total_h        = rows * tile_h + (rows - 1) * gap
        ox = (W - total_w) // 2
        oy = 200

        tiles = [
            ("What Is This?",    "Learn what cognitive drift is",     ACCENT,  State.LEARN),
            ("Run Full Demo",    "Auto-run AI pipeline end-to-end",   GREEN,   State.DEMO),
            ("Play the Task",    "Collect your own cognitive data",   ACCENT2, State.PLAY_TASK),
            ("View Results",     "See plots & model comparisons",     YELLOW,  State.RESULTS),
            ("Live AI Mode",     "Play with real-time predictions",   LSTM_C,  State.LIVE_MODE),
            ("Quit",             "Exit the application",              DIM,     None),
        ]

        self._menu_buttons: List[Tuple[Button, Optional[State]]] = []
        for i, (label, desc, color, target_state) in enumerate(tiles):
            col = i % cols
            row = i // cols
            x = ox + col * (tile_w + gap)
            y = oy + row * (tile_h + gap)
            rect = pygame.Rect(x, y, tile_w, tile_h)
            btn = Button(
                rect=rect,
                text=label,
                sub_text=desc,
                font=self.f_btn,
                sub_font=self.f_small,
                color=color,
                text_color=BG,
                hover_color=WHITE,
                radius=12,
            )
            self._menu_buttons.append((btn, target_state))

    def _render_menu(self) -> None:
        # Title
        draw_text(self.screen, "DriftSync", self.f_title, ACCENT, W // 2, 38, anchor="midtop")
        draw_text(self.screen, "Real-Time Neural Cognitive Drift Prediction System",
                  self.f_body, DIM, W // 2, 96, anchor="midtop")

        # Separator
        pygame.draw.line(self.screen, BORDER, (80, 135), (W - 80, 135), 1)

        draw_text(self.screen, "Select an option to get started:",
                  self.f_small, DIM, W // 2, 158, anchor="midtop")

        for btn, _ in self._menu_buttons:
            btn.draw(self.screen)

        # Footer
        draw_text(self.screen, "ESC = quit   |   DriftSync v1.0",
                  self.f_small, DIM, W // 2, H - 22, anchor="midbottom")

    def _handle_menu_event(self, event: pygame.event.Event) -> None:
        for btn, target in self._menu_buttons:
            if btn.handle_event(event):
                if target is None:
                    self.running = False
                elif target == State.RESULTS:
                    self._load_results()
                    self.state = State.RESULTS
                elif target == State.DEMO:
                    # Reset demo state each time we enter
                    self.demo_logs    = []
                    self.demo_metrics = {}
                    self.demo_train_loss.clear()
                    self.demo_val_loss.clear()
                    self.demo_val_auc.clear()
                    self.demo_worker  = DemoWorker()
                    self._build_demo_buttons()
                    self.state = State.DEMO
                else:
                    self.state = target

    # ==================================================================
    # LEARN
    # ==================================================================

    def _build_learn_buttons(self) -> None:
        bw = 140
        bh = 38
        by = H - 58
        self._btn_learn_back = Button(
            pygame.Rect(80,         by, bw, bh), "< Back",
            self.f_btn, color=PANEL2, text_color=TEXT, hover_color=ACCENT2, radius=8,
        )
        self._btn_learn_next = Button(
            pygame.Rect(W - 80 - bw, by, bw, bh), "Next >",
            self.f_btn, color=ACCENT, text_color=BG, hover_color=WHITE, radius=8,
        )
        self._btn_learn_menu = Button(
            pygame.Rect(W // 2 - 70, by, 140, bh), "Main Menu",
            self.f_btn, color=PANEL2, text_color=TEXT, hover_color=ACCENT2, radius=8,
        )

    def _render_learn(self) -> None:
        page = LEARN_PAGES[self.learn_page]

        # Sidebar
        sidebar_rect = pygame.Rect(0, 0, 220, H)
        pygame.draw.rect(self.screen, PANEL, sidebar_rect)
        pygame.draw.line(self.screen, BORDER, (220, 0), (220, H), 1)

        draw_text(self.screen, "Contents", self.f_sub, ACCENT, 14, 18)
        for i, p in enumerate(LEARN_PAGES):
            short = p["title"].split(".")[0] + ". " + p["title"].split(". ", 1)[1][:24]
            y = 62 + i * 38
            if i == self.learn_page:
                pygame.draw.rect(self.screen, ACCENT2, (4, y - 4, 212, 30), border_radius=6)
            draw_text(self.screen, short, self.f_small,
                      BG if i == self.learn_page else DIM, 12, y)

        # Page dot indicators at bottom of sidebar
        for i in range(len(LEARN_PAGES)):
            col = ACCENT if i == self.learn_page else BORDER
            pygame.draw.circle(self.screen, col, (14 + i * 22, H - 22), 6)

        # Main content area
        content_x  = 240
        content_w  = W - content_x - 30
        title_rect = pygame.Rect(content_x, 20, content_w, 50)
        body_rect  = pygame.Rect(content_x, 78, content_w, H - 78 - 80)

        # Page title
        draw_text(self.screen, page["title"], self.f_head, ACCENT, content_x, 24)
        pygame.draw.line(self.screen, ACCENT2, (content_x, 58), (content_x + content_w, 58), 1)

        # Body text — split on \n and handle sections
        body_text = page["body"]
        self._render_learn_body(body_text, body_rect)

        # Visual element
        vis = page.get("visual", "")
        self._render_learn_visual(vis, content_x, content_w, body_rect.bottom + 4)

        # Navigation buttons
        self._btn_learn_back.disabled = (self.learn_page == 0)
        self._btn_learn_next.disabled = (self.learn_page == len(LEARN_PAGES) - 1)
        self._btn_learn_back.draw(self.screen)
        self._btn_learn_next.draw(self.screen)
        self._btn_learn_menu.draw(self.screen)

        draw_text(self.screen,
                  f"Page {self.learn_page + 1} of {len(LEARN_PAGES)}  |  ESC = Menu",
                  self.f_small, DIM, W // 2, H - 16, anchor="midbottom")

    def _render_learn_body(self, text: str, rect: pygame.Rect) -> None:
        """Render body text with indented lines in DIM and normal lines in TEXT."""
        lines = text.split("\n")
        y = rect.top
        for line in lines:
            if y + self.f_body.get_height() > rect.bottom:
                break
            if not line.strip():
                y += self.f_body.get_height() // 2
                continue
            stripped = line.lstrip()
            indent   = len(line) - len(stripped)
            color    = DIM if indent >= 2 else TEXT
            # Word-wrap within available width
            available_w = rect.width - indent * 8
            words  = stripped.split()
            cur    = ""
            for w in words:
                test = (cur + " " + w).strip()
                if self.f_body.size(test)[0] <= available_w:
                    cur = test
                else:
                    surf = self.f_body.render(cur, True, color)
                    self.screen.blit(surf, (rect.left + indent * 8, y))
                    y  += self.f_body.get_height() + 3
                    cur = w
                    if y + self.f_body.get_height() > rect.bottom:
                        break
            if cur and y + self.f_body.get_height() <= rect.bottom:
                surf = self.f_body.render(cur, True, color)
                self.screen.blit(surf, (rect.left + indent * 8, y))
                y += self.f_body.get_height() + 3

    def _render_learn_visual(self, vis: str, cx: int, cw: int, y_hint: int) -> None:
        """Draw an inline visual element for the current learn page."""
        avail_h = H - 80 - y_hint - 10
        if avail_h < 40:
            return
        if vis == "drift_curve":
            self._draw_drift_curve(cx, y_hint, cw, min(avail_h, 90))
        elif vis == "probability_gauge":
            self._draw_gauge_demo(cx, y_hint, cw, min(avail_h, 60))
        elif vis == "lstm_arch":
            self._draw_arch_label(cx, y_hint, "Input (20, 11) -> Projection -> LSTM x3 -> Head -> P(error)", LSTM_C)
        elif vis == "transformer_arch":
            self._draw_arch_label(cx, y_hint, "Input (20, 11) -> Projection -> Pos.Enc -> Attn x4 -> Pool -> Head -> P(error)", TF_C)

    def _draw_drift_curve(self, x: int, y: int, w: int, h: int) -> None:
        """Draw a schematic cognitive drift curve."""
        import math
        pts = []
        for i in range(w):
            t = i / w
            val = 0.85 * (1 - 0.7 * (1 - math.exp(-5 * (1 - t))))
            val += 0.04 * math.sin(i * 0.3)
            pts.append((x + i, y + h - int(val * h)))
        pygame.draw.lines(self.screen, GREEN, False, pts, 2)
        draw_text(self.screen, "Accuracy", self.f_small, GREEN, x, y)
        draw_text(self.screen, "Time ->", self.f_small, DIM, x + w - 60, y + h)
        draw_text(self.screen, "Cognitive drift curve (schematic)", self.f_small, DIM,
                  x + w // 2, y + h + 4, anchor="midtop")

    def _draw_gauge_demo(self, x: int, y: int, w: int, h: int) -> None:
        """Draw example probability gauges at low/medium/high."""
        labels = [("LOW  0.15", 0.15, GREEN), ("MED  0.45", 0.45, YELLOW), ("HIGH 0.78", 0.78, RED)]
        gw = min(w // 3 - 20, 220)
        for i, (lbl, val, col) in enumerate(labels):
            gx = x + i * (gw + 12)
            draw_text(self.screen, lbl, self.f_small, col, gx, y)
            bar = pygame.Rect(gx, y + 18, gw, 14)
            draw_progress_bar(self.screen, bar, val, fg_color=col)

    def _draw_arch_label(self, x: int, y: int, text: str, color) -> None:
        draw_text(self.screen, text, self.f_small, color, x, y)

    def _handle_learn_event(self, event: pygame.event.Event) -> None:
        if self._btn_learn_next.handle_event(event) and self.learn_page < len(LEARN_PAGES) - 1:
            self.learn_page += 1
        if self._btn_learn_back.handle_event(event) and self.learn_page > 0:
            self.learn_page -= 1
        if self._btn_learn_menu.handle_event(event):
            self.state = State.MENU

        # Keyboard navigation
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_RIGHT, pygame.K_SPACE) and self.learn_page < len(LEARN_PAGES) - 1:
                self.learn_page += 1
            if event.key == pygame.K_LEFT and self.learn_page > 0:
                self.learn_page -= 1

    # ==================================================================
    # DEMO
    # ==================================================================

    def _build_demo_buttons(self) -> None:
        self._btn_demo_start = Button(
            pygame.Rect(30, 90, 200, 40), "Start Demo",
            self.f_btn, color=GREEN, text_color=BG, hover_color=WHITE, radius=8,
        )
        self._btn_demo_menu = Button(
            pygame.Rect(W - 160, 90, 130, 40), "< Menu",
            self.f_btn, color=PANEL2, text_color=TEXT, hover_color=ACCENT2, radius=8,
        )
        self.demo_view_results_btn = Button(
            pygame.Rect(W // 2 - 120, H - 54, 240, 40), "View Results ->",
            self.f_btn, color=YELLOW, text_color=BG, hover_color=WHITE, radius=8,
        )
        self.demo_view_results_btn.disabled = True

    def _render_demo(self) -> None:
        # Header
        draw_text(self.screen, "Full ML Pipeline Demo", self.f_head, ACCENT, 30, 20)
        status_color = GREEN if self.demo_worker.status == "done" else \
                       RED   if self.demo_worker.status == "error" else YELLOW
        status_txt   = {"idle": "Ready", "running": "Running...",
                        "done": "Complete!", "error": "Error"}.get(self.demo_worker.status, "")
        draw_text(self.screen, status_txt, self.f_sub, status_color, W - 30, 24, anchor="topright")

        # Buttons
        if self.demo_worker.status == "idle":
            self._btn_demo_start.draw(self.screen)
        self._btn_demo_menu.draw(self.screen)

        # Overall progress bar
        step  = self.demo_worker.current_step
        n     = self.demo_worker.total_steps
        prog  = (step + (1.0 if self.demo_worker.status == "done" else 0.0)) / n
        pb    = pygame.Rect(30, 140, W - 60, 16)
        draw_progress_bar(self.screen, pb, prog, fg_color=ACCENT)

        # Step labels
        step_w = (W - 60) // n
        for i, lbl in enumerate(DemoWorker.STEPS):
            sx     = 30 + i * step_w
            is_done= i < step or self.demo_worker.status == "done"
            is_cur = i == step and self.demo_worker.status == "running"
            col    = GREEN if is_done else YELLOW if is_cur else DIM
            short  = lbl[:20]
            draw_text(self.screen, short, self.f_small, col, sx + step_w // 2, 162, anchor="midtop")

        # ---- Log panel (left 58%) ----
        log_rect = pygame.Rect(30, 192, int(W * 0.58) - 40, H - 192 - 70)
        draw_filled_rect(self.screen, log_rect, PANEL, 6)
        draw_rect_border(self.screen, log_rect, BORDER, 6, 1)
        draw_text(self.screen, "Pipeline Log", self.f_small, DIM, log_rect.left + 8, log_rect.top + 6)

        visible_lines = (log_rect.height - 24) // (self.f_mono.get_height() + 2)
        start_idx     = max(0, len(self.demo_logs) - visible_lines)
        y = log_rect.top + 24
        for line in self.demo_logs[start_idx:]:
            col = GREEN if "complete" in line.lower() or "done" in line.lower() else \
                  RED   if "error" in line.lower() else \
                  YELLOW if "training" in line.lower() else TEXT
            surf = self.f_mono.render(line[:90], True, col)
            self.screen.blit(surf, (log_rect.left + 8, y))
            y += self.f_mono.get_height() + 2
            if y > log_rect.bottom - 4:
                break

        # ---- Metrics panel (right 42%) ----
        mx       = int(W * 0.58) + 10
        mw       = W - mx - 20
        met_rect = pygame.Rect(mx, 192, mw, H - 192 - 70)
        draw_filled_rect(self.screen, met_rect, PANEL, 6)
        draw_rect_border(self.screen, met_rect, BORDER, 6, 1)
        draw_text(self.screen, "Live Metrics", self.f_small, DIM, met_rect.left + 8, met_rect.top + 6)

        m = self.demo_metrics
        if m:
            model_col = LSTM_C if m.get("model") == "LSTM" else TF_C
            my = met_rect.top + 28
            draw_text(self.screen, f"Model: {m.get('model', '?')}",
                      self.f_sub, model_col, met_rect.left + 10, my)
            my += 30

            epoch     = m.get("epoch", "?")
            max_ep    = m.get("max_epochs", 40)
            ep_frac   = epoch / max_ep if isinstance(epoch, int) else 0
            draw_text(self.screen, f"Epoch {epoch} / {max_ep}", self.f_body, TEXT, met_rect.left + 10, my)
            ep_bar = pygame.Rect(met_rect.left + 10, my + 20, mw - 20, 10)
            draw_progress_bar(self.screen, ep_bar, ep_frac, fg_color=model_col)
            my += 38

            def metric_row(label, value, good_thresh, col):
                nonlocal my
                col2 = GREEN if value < good_thresh else YELLOW
                draw_text(self.screen, label, self.f_body, DIM, met_rect.left + 10, my)
                draw_text(self.screen, f"{value:.4f}", self.f_sub, col2, met_rect.right - 10, my, anchor="topright")
                my += 24

            metric_row("Train Loss:", m.get("train_loss", 0), 0.3, GREEN)
            metric_row("Val   Loss:", m.get("val_loss", 0),   0.3, GREEN)
            draw_text(self.screen, "Val   AUC:", self.f_body, DIM, met_rect.left + 10, my)
            auc_val = m.get("val_auc", 0.0)
            auc_col = GREEN if auc_val > 0.75 else YELLOW if auc_val > 0.60 else RED
            draw_text(self.screen, f"{auc_val:.4f}", self.f_sub, auc_col, met_rect.right - 10, my, anchor="topright")
            my += 24

            draw_text(self.screen, "Val   F1:", self.f_body, DIM, met_rect.left + 10, my)
            f1_val = m.get("val_f1", 0.0)
            f1_col = GREEN if f1_val > 0.70 else YELLOW if f1_val > 0.55 else RED
            draw_text(self.screen, f"{f1_val:.4f}", self.f_sub, f1_col, met_rect.right - 10, my, anchor="topright")
            my += 36

            # Loss sparkline
            if self.demo_train_loss:
                draw_text(self.screen, "Loss curve:", self.f_small, DIM, met_rect.left + 10, my)
                my += 16
                spark_rect = pygame.Rect(met_rect.left + 10, my, mw - 20, 55)
                draw_filled_rect(self.screen, spark_rect, BG, 4)
                all_loss = list(self.demo_train_loss) + list(self.demo_val_loss)
                y_max = max(all_loss) + 0.02 if all_loss else 1.0
                draw_sparkline(self.screen, list(self.demo_train_loss), spark_rect, LSTM_C, 0, y_max)
                draw_sparkline(self.screen, list(self.demo_val_loss), spark_rect, RED, 0, y_max)
                draw_text(self.screen, "Train", self.f_small, LSTM_C, met_rect.left + 12, my + 58)
                draw_text(self.screen, "Val",   self.f_small, RED,    met_rect.left + 62, my + 58)
                my += 80

            # AUC sparkline
            if self.demo_val_auc:
                draw_text(self.screen, "Val AUC curve:", self.f_small, DIM, met_rect.left + 10, my)
                my += 16
                auc_rect = pygame.Rect(met_rect.left + 10, my, mw - 20, 50)
                draw_filled_rect(self.screen, auc_rect, BG, 4)
                draw_sparkline(self.screen, list(self.demo_val_auc), auc_rect, GREEN, 0, 1.0)
                # Threshold line at 0.7
                thr_y = auc_rect.bottom - int(0.7 * auc_rect.height)
                pygame.draw.line(self.screen, DIM, (auc_rect.left, thr_y), (auc_rect.right, thr_y), 1)
                draw_text(self.screen, "0.70", self.f_small, DIM, auc_rect.right + 2, thr_y)
        else:
            if self.demo_worker.status == "idle":
                msg = "Click 'Start Demo' to run the full AI pipeline."
            elif self.demo_worker.status == "running":
                msg = "Pipeline running — metrics will appear here..."
            else:
                msg = "Pipeline complete. No metrics to display."
            draw_text(self.screen, msg, self.f_body, DIM, met_rect.centerx, met_rect.centery, anchor="center")

        # View Results button
        self.demo_view_results_btn.draw(self.screen)
        draw_text(self.screen, "ESC = Menu", self.f_small, DIM, W - 20, H - 16, anchor="bottomright")

    def _handle_demo_event(self, event: pygame.event.Event) -> None:
        if hasattr(self, "_btn_demo_start") and self.demo_worker.status == "idle":
            if self._btn_demo_start.handle_event(event):
                self.demo_worker.start()

        if hasattr(self, "_btn_demo_menu") and self._btn_demo_menu.handle_event(event):
            self.state = State.MENU

        if self.demo_view_results_btn and self.demo_view_results_btn.handle_event(event):
            self._load_results()
            self.state = State.RESULTS

    # ==================================================================
    # RESULTS
    # ==================================================================

    def _load_results(self) -> None:
        """Scan results directory and load thumbnail surfaces."""
        results_dir = Path("driftsync/results")
        self.result_thumbs.clear()
        self.result_full_view = None
        self.result_scroll    = 0

        if results_dir.exists():
            pngs = sorted(results_dir.glob("*.png"))
            for p in pngs:
                surf = load_png_surface(p, (310, 210))
                if surf:
                    title = p.stem.replace("_", " ").title()
                    self.result_thumbs.append((surf, title, p))

        # Load metrics JSON
        json_path = results_dir / "comparison_summary.json"
        if json_path.exists():
            with open(json_path) as f:
                self.result_metrics = json.load(f)
        else:
            self.result_metrics = {}

        # Rebuild results scroll buttons
        self._btn_results_menu = Button(
            pygame.Rect(20, 20, 130, 36), "< Menu",
            self.f_btn, color=PANEL2, text_color=TEXT, hover_color=ACCENT2, radius=8,
        )
        self._btn_results_open = Button(
            pygame.Rect(W - 200, 20, 180, 36), "Open Folder",
            self.f_btn, color=PANEL, text_color=ACCENT, hover_color=ACCENT, radius=8,
        )

    def _render_results(self) -> None:
        # Full-screen image view
        if self.result_full_view:
            self._render_full_image()
            return

        draw_text(self.screen, "Results & Plots", self.f_head, ACCENT, W // 2, 18, anchor="midtop")
        self._btn_results_menu.draw(self.screen)
        self._btn_results_open.draw(self.screen)

        if not self.result_thumbs:
            msg1 = "No results found."
            msg2 = "Run the Full Demo first to train models and generate plots."
            draw_text(self.screen, msg1, self.f_sub, YELLOW, W // 2, H // 2 - 20, anchor="center")
            draw_text(self.screen, msg2, self.f_body, DIM,    W // 2, H // 2 + 20, anchor="center")
            return

        # Thumbnail grid (3 columns)
        tw, th  = 310, 210
        gap     = 16
        cols    = 3
        row_h   = th + 30 + gap
        ox      = (W - (cols * tw + (cols - 1) * gap)) // 2
        oy      = 70
        visible_area = H - oy - 70

        pygame.draw.line(self.screen, BORDER, (0, 65), (W, 65), 1)

        # Scroll indicators
        max_scroll = max(0, len(self.result_thumbs) * row_h // cols - visible_area)
        self.result_scroll = max(0, min(self.result_scroll, max_scroll))

        for i, (surf, title, path) in enumerate(self.result_thumbs):
            col = i % cols
            row = i // cols
            tx  = ox + col * (tw + gap)
            ty  = oy + row * row_h - self.result_scroll

            if ty + th < oy or ty > H - 70:
                continue

            # Border
            clip_rect = pygame.Rect(tx - 3, max(oy, ty - 3), tw + 6, min(th + 6, H - 70 - ty + 3))
            draw_filled_rect(self.screen, pygame.Rect(tx - 3, ty - 3, tw + 6, th + 6), PANEL, 8)
            self.screen.blit(surf, (tx, ty))
            draw_text(self.screen, title, self.f_small, TEXT, tx + tw // 2, ty + th + 4, anchor="midtop")

        # Metrics table
        if self.result_metrics:
            table_y = oy + ((len(self.result_thumbs) + 2) // cols) * row_h - self.result_scroll
            if table_y < H - 70:
                self._render_metrics_table(ox, table_y)

        # Scroll hint
        draw_text(self.screen, "Scroll: mouse wheel   |   Click image to enlarge   |   ESC = Menu",
                  self.f_small, DIM, W // 2, H - 16, anchor="midbottom")

    def _render_metrics_table(self, x: int, y: int) -> None:
        if not self.result_metrics:
            return
        draw_text(self.screen, "Model Comparison", self.f_sub, ACCENT, x, y)
        y += 30
        cols_names = ["accuracy", "f1", "roc_auc", "ece"]
        headers    = ["Model", "Accuracy", "F1", "AUC", "ECE"]
        col_w      = [130, 110, 90, 90, 90]

        # Header row
        cx = x
        for header, cw in zip(headers, col_w):
            draw_text(self.screen, header, self.f_body, DIM, cx, y)
            cx += cw
        y += 22
        pygame.draw.line(self.screen, BORDER, (x, y), (x + sum(col_w), y), 1)
        y += 4

        for model_name, metrics in self.result_metrics.items():
            cx = x
            color = LSTM_C if model_name == "lstm" else TF_C
            draw_text(self.screen, model_name.upper(), self.f_body, color, cx, y)
            cx += col_w[0]
            for metric in cols_names:
                val = metrics.get(metric, float("nan"))
                vc  = GREEN if (metric in ("accuracy", "f1", "roc_auc") and val > 0.70) else \
                      YELLOW if (metric in ("accuracy", "f1", "roc_auc") and val > 0.55) else \
                      GREEN  if (metric == "ece" and val < 0.10) else TEXT
                draw_text(self.screen, f"{val:.4f}", self.f_body, vc, cx, y)
                cx += col_w[cols_names.index(metric) + 1]
            y += 24

    def _render_full_image(self) -> None:
        """Full-screen single-image view."""
        if self.result_full_view:
            # Fit to screen
            img = self.result_full_view
            ratio = min((W - 40) / img.get_width(), (H - 80) / img.get_height())
            new_w = int(img.get_width()  * ratio)
            new_h = int(img.get_height() * ratio)
            scaled = pygame.transform.smoothscale(img, (new_w, new_h))
            x = (W - new_w) // 2
            y = 60 + (H - 80 - new_h) // 2
            self.screen.blit(scaled, (x, y))
            draw_text(self.screen, self.result_full_title, self.f_sub, ACCENT, W // 2, 12, anchor="midtop")
            draw_text(self.screen, "ESC or click to go back", self.f_small, DIM, W // 2, H - 16, anchor="midbottom")

    def _handle_results_event(self, event: pygame.event.Event) -> None:
        # Full-screen image: any click or ESC goes back
        if self.result_full_view:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.result_full_view = None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.result_full_view = None
            return

        if event.type == pygame.MOUSEWHEEL:
            self.result_scroll -= event.y * 40

        if self._btn_results_menu.handle_event(event):
            self.state = State.MENU

        if self._btn_results_open.handle_event(event):
            try:
                import subprocess
                results_path = str(Path("driftsync/results").resolve())
                if sys.platform == "win32":
                    os.startfile(results_path)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", results_path])
                else:
                    subprocess.Popen(["xdg-open", results_path])
            except Exception:
                pass

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            tw, th  = 310, 210
            gap     = 16
            cols    = 3
            ox      = (W - (cols * tw + (cols - 1) * gap)) // 2
            oy      = 70
            row_h   = th + 30 + gap
            for i, (surf, title, path) in enumerate(self.result_thumbs):
                col = i % cols
                row = i // cols
                tx  = ox + col * (tw + gap)
                ty  = oy + row * row_h - self.result_scroll
                rect = pygame.Rect(tx, ty, tw, th)
                if rect.collidepoint(event.pos):
                    full = load_png_surface(path)
                    if full:
                        self.result_full_view  = full
                        self.result_full_title = title

    # ==================================================================
    # PLAY TASK
    # ==================================================================

    def _render_play_task(self) -> None:
        self._render_info_screen(
            title="Play the Cognitive Task",
            lines=[
                "You will play a rapid shape-classification game.",
                "",
                "A shape appears on screen: Circle, Square, or Triangle.",
                "The rule at the top tells you which shape to click.",
                "  -> Click it if it matches   |   SPACE to skip if it does not",
                "",
                "Your reaction time and accuracy are recorded every trial.",
                "After 200 trials, your data is saved for AI training.",
                "",
                "Tip: Try to stay focused. The time window shrinks over time.",
                "     Notice when you start making more mistakes — that is drift!",
            ],
            btn_label="Start Task ->",
            btn_color=ACCENT2,
            note="ESC = back to menu",
        )

    def _handle_play_event(self, event: pygame.event.Event) -> None:
        pass  # handled below via the info screen button

    # ==================================================================
    # LIVE MODE
    # ==================================================================

    def _render_live_mode(self) -> None:
        self._render_info_screen(
            title="Live AI Inference Mode",
            lines=[
                "Play the task while the AI predicts your drift in real-time.",
                "",
                "A DRIFT PROBABILITY gauge shows at the top of the screen.",
                "When P(error) > 0.65, the screen turns RED as a warning.",
                "",
                "The uncertainty band shows how confident the AI is.",
                "A sparkline tracks your drift probability over the last 30 trials.",
                "",
                f"Current model: {self.live_model_choice.upper()}",
                "Press L for LSTM  |  Press T for Transformer",
                "",
                "Note: Train a model first by running the Full Demo.",
            ],
            btn_label="Start Live Mode ->",
            btn_color=LSTM_C,
            note="ESC = back to menu",
            extra_key_hint="L = LSTM   |   T = Transformer",
        )

    def _handle_live_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_l:
                self.live_model_choice = "lstm"
            if event.key == pygame.K_t:
                self.live_model_choice = "transformer"

    # ==================================================================
    # Generic info screen (for PLAY_TASK and LIVE_MODE)
    # ==================================================================

    def _render_info_screen(
        self,
        title: str,
        lines: List[str],
        btn_label: str,
        btn_color,
        note: str = "",
        extra_key_hint: str = "",
    ) -> None:
        cx = W // 2

        # Panel
        panel = pygame.Rect(W // 2 - 380, 60, 760, H - 130)
        draw_filled_rect(self.screen, panel, PANEL, 14)
        draw_rect_border(self.screen, panel, BORDER, 14, 1)

        draw_text(self.screen, title, self.f_head, ACCENT, cx, 80, anchor="midtop")
        pygame.draw.line(self.screen, ACCENT2, (panel.left + 20, 120), (panel.right - 20, 120), 1)

        y = 136
        for line in lines:
            if not line.strip():
                y += 10
                continue
            stripped = line.lstrip()
            indent   = len(line) - len(stripped)
            col      = DIM if indent >= 2 else TEXT
            surf     = self.f_body.render(line, True, col)
            self.screen.blit(surf, (panel.left + 30, y))
            y += self.f_body.get_height() + 5

        if extra_key_hint:
            draw_text(self.screen, extra_key_hint, self.f_body, YELLOW,
                      cx, y + 10, anchor="midtop")
            y += 34

        # Action button
        btn = Button(
            pygame.Rect(cx - 130, panel.bottom - 60, 260, 44),
            btn_label, self.f_btn,
            color=btn_color, text_color=BG, hover_color=WHITE, radius=10,
        )
        btn.draw(self.screen)
        if note:
            draw_text(self.screen, note, self.f_small, DIM, cx, panel.bottom - 8, anchor="midbottom")

        # Handle this button's click inline via mouse state
        mx, my = pygame.mouse.get_pos()
        btn.hovered = btn.rect.collidepoint(mx, my)
        btn.draw(self.screen)   # redraw with hover state

        # Store button so _handle_events can check it
        self._info_btn = btn

        # Handle clicks (detected in next event loop iteration)
        # We check in _handle_events using _check_info_btn

    def _check_info_btn_click(self, event: pygame.event.Event) -> bool:
        if hasattr(self, "_info_btn"):
            return self._info_btn.handle_event(event)
        return False

    # ------------------------------------------------------------------
    # Override _handle_events to catch info screen button clicks
    # ------------------------------------------------------------------

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if self.state in (State.LEARN, State.DEMO, State.RESULTS,
                                  State.PLAY_TASK, State.LIVE_MODE):
                    self.state = State.MENU
                    return
                if self.state == State.MENU:
                    self.running = False
                    return

            if self.state == State.MENU:
                self._handle_menu_event(event)
            elif self.state == State.LEARN:
                self._handle_learn_event(event)
            elif self.state == State.DEMO:
                self._handle_demo_event(event)
            elif self.state == State.RESULTS:
                self._handle_results_event(event)
            elif self.state == State.LIVE_MODE:
                self._handle_live_event(event)
                if self._check_info_btn_click(event):
                    self._launch_live_mode()
            elif self.state == State.PLAY_TASK:
                if self._check_info_btn_click(event):
                    self._launch_play_task()

    # ==================================================================
    # Launch external simulators (quit pygame, run, re-init)
    # ==================================================================

    def _launch_play_task(self) -> None:
        """Pause app, run DriftSimulator, restore app."""
        pygame.quit()
        try:
            from driftsync.configs import SimulatorConfig
            from driftsync.simulator.gui import DriftSimulator
            sim = DriftSimulator(SimulatorConfig(num_trials=150))
            sim.run()
        except Exception as e:
            print(f"Simulator error: {e}")
        finally:
            pygame.init()
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("DriftSync — Cognitive Drift Prediction")
            self.state = State.MENU

    def _launch_live_mode(self) -> None:
        """Pause app, run LiveDriftSimulator, restore app."""
        pygame.quit()
        try:
            from driftsync.configs import SimulatorConfig, RealtimeConfig
            from driftsync.realtime.live_simulator import LiveDriftSimulator
            sim_cfg = SimulatorConfig(num_trials=150)
            rt_cfg  = RealtimeConfig(model_type=self.live_model_choice)
            live = LiveDriftSimulator(sim_cfg, rt_cfg, model_type=self.live_model_choice)
            live.run()
        except Exception as e:
            print(f"Live mode error: {e}")
        finally:
            pygame.init()
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("DriftSync — Cognitive Drift Prediction")
            self.state = State.MENU
