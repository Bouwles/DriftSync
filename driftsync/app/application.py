"""
DriftSync Interactive Application
Pygame GUI with flat developer-tool aesthetic and left sidebar navigation.

Screens: SPLASH / MENU / LEARN / DEMO / RESULTS / PLAY_TASK / LIVE_MODE
"""

import json
import logging
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Tuple

import pygame


from driftsync import ui, __version__
from driftsync.app.mathematics import MathematicsView


class State(Enum):
    SPLASH    = auto()
    MENU      = auto()
    LEARN     = auto()
    DEMO      = auto()
    RESULTS   = auto()
    PLAY_TASK = auto()
    LIVE_MODE = auto()
    MATHEMATICS = auto()


BG, PANEL, PANEL2, BORDER = ui.BG, ui.PANEL, ui.PANEL2, ui.BORDER
ACCENT, GREEN, YELLOW, RED = ui.ACCENT, ui.GREEN, ui.YELLOW, ui.RED
TEXT, DIM, WHITE = ui.TEXT, ui.DIM, ui.TEXT
LSTM_C, TF_C = ACCENT, (196, 188, 163)
SIDEBAR_W = 184
SIDEBAR_BG = (20, 24, 27)
W, H = 1280, 800
FPS = 60


class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.log_queue.put_nowait(self.format(record))
        except Exception:
            pass


def draw_rect(surface, rect, color, radius=4) -> None:
    pygame.draw.rect(surface, color, rect, border_radius=radius)

def draw_border(surface, rect, color, radius=4, width=1) -> None:
    pygame.draw.rect(surface, color, rect, width, border_radius=radius)

def draw_text(surface, text, font, color, x, y, anchor="topleft") -> pygame.Rect:
    surf = font.render(text, True, color)
    r = surf.get_rect()
    setattr(r, anchor, (x, y))
    surface.blit(surf, r)
    return r

def draw_progress_bar(surface, rect, value, fg=GREEN, bg=PANEL2, radius=3) -> None:
    draw_rect(surface, rect, bg, radius)
    if value > 0:
        fill = rect.copy()
        fill.width = max(1, int(rect.width * min(value, 1.0)))
        draw_rect(surface, fill, fg, radius)
    draw_border(surface, rect, BORDER, radius, 1)

def draw_sparkline(surface, data, rect, color=ACCENT, y_min=0.0, y_max=1.0) -> None:
    if len(data) < 2:
        return
    span = max(y_max - y_min, 1e-6)
    n = len(data)
    pts = []
    for i, v in enumerate(data):
        px = rect.left + int(i / (n - 1) * rect.width)
        py = rect.bottom - int((v - y_min) / span * rect.height)
        py = max(rect.top, min(rect.bottom, py))
        pts.append((px, py))
    pygame.draw.lines(surface, color, False, pts, 2)

def load_png(path, size=None) -> Optional[pygame.Surface]:
    try:
        surf = pygame.image.load(str(path))
        if size:
            surf = pygame.transform.smoothscale(surf, size)
        return surf
    except Exception:
        return None

def hline(surface, y, x0, x1, color=BORDER) -> None:
    pygame.draw.line(surface, color, (x0, y), (x1, y), 1)

def vline(surface, x, y0, y1, color=BORDER) -> None:
    pygame.draw.line(surface, color, (x, y0), (x, y1), 1)


class Button:
    def __init__(self, rect, text, font,
                 color=PANEL2, text_color=TEXT,
                 hover_color=PANEL2, hover_text=ACCENT,
                 radius=5, sub_text="", sub_font=None,
                 accent_fill=False):
        self.rect        = rect
        self.text        = text
        self.sub_text    = sub_text
        self.font        = font
        self.sub_font    = sub_font
        self.color       = color
        self.text_color  = text_color
        self.hover_color = hover_color
        self.hover_text  = hover_text
        self.radius      = radius
        self.accent_fill = accent_fill
        self.hovered     = False
        self.focused     = False
        self.disabled    = False

    def draw(self, surface) -> None:
        if self.disabled:
            bg, tc, bc = PANEL2, DIM, BORDER
        elif self.accent_fill:
            bg = (151, 217, 210) if self.hovered else ACCENT
            tc = (10, 15, 20)
            bc = bg
        elif self.hovered or self.focused:
            bg = self.hover_color
            tc = self.hover_text
            bc = ACCENT
        else:
            bg = self.color
            tc = self.text_color
            bc = BORDER

        previous_clip = surface.get_clip()
        surface.set_clip(self.rect.clip(previous_clip))
        draw_rect(surface, self.rect, bg, self.radius)
        draw_border(surface, self.rect, bc, self.radius, 1)

        cy = self.rect.centery if not self.sub_text else self.rect.centery - 10
        draw_text(surface, self.text, self.font, tc, self.rect.centerx, cy, anchor="center")
        if self.sub_text and self.sub_font:
            sc = (160, 170, 180) if self.hovered else DIM
            draw_text(surface, self.sub_text, self.sub_font, sc,
                      self.rect.centerx, cy + self.font.get_height() + 3, anchor="center")
        surface.set_clip(previous_clip)
        if self.focused and not self.disabled:
            draw_border(surface, self.rect.inflate(6, 6), ACCENT, self.radius, 2)

    def handle_event(self, event) -> bool:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN and self.focused and not self.disabled:
            return True
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self.disabled:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class TextInput:
    def __init__(self, rect, font, placeholder="", max_len=50):
        self.rect        = rect
        self.font        = font
        self.placeholder = placeholder
        self.max_len     = max_len
        self.text        = ""
        self.active      = False
        self._cur_vis    = True
        self._cur_t      = 0.0

    def handle_event(self, event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key not in (pygame.K_RETURN, pygame.K_TAB, pygame.K_ESCAPE):
                if len(self.text) < self.max_len and event.unicode.isprintable():
                    self.text += event.unicode

    def draw(self, surface, dt=0.016) -> None:
        self._cur_t += dt
        if self._cur_t > 0.5:
            self._cur_t = 0.0
            self._cur_vis = not self._cur_vis
        bc = ACCENT if self.active else BORDER
        draw_rect(surface, self.rect, PANEL2, 5)
        draw_border(surface, self.rect, bc, 5, 1)
        previous_clip = surface.get_clip()
        surface.set_clip(self.rect.inflate(-16, -4).clip(previous_clip))
        txt = self.text if self.text else self.placeholder
        col = TEXT if self.text else DIM
        s   = self.font.render(txt, True, col)
        offset = max(0, s.get_width() - self.rect.width + 24) if self.text else 0
        surface.blit(s, (self.rect.left + 10 - offset, self.rect.centery - s.get_height() // 2))
        if self.active and self._cur_vis:
            cx = self.rect.left + 10 + self.font.size(self.text)[0] - offset
            pygame.draw.line(surface, ACCENT,
                             (cx, self.rect.top + 6), (cx, self.rect.bottom - 6), 1)

        surface.set_clip(previous_clip)


LEARN_PAGES = [
    {
        "title": "1. What Is Cognitive Drift?",
        "body": (
            "Cognitive drift is the gradual decline in mental performance during sustained tasks. "
            "As your brain fatigues, reaction times slow, errors creep in, and attention scatters. "
            "This happens to pilots, surgeons, air traffic controllers, and anyone performing "
            "repetitive cognitive work.\n\n"
            "Drift is not random — it follows measurable temporal patterns. Early in a session, "
            "performance is sharp. Changes vary with the person and task. Drift "
            "can become noticeable over time.\n\n"
            "DriftSync asks: can an AI model learn these patterns from your interaction history "
            "and predict WHEN your next mistake will happen — before it happens?"
        ),
        "visual": "drift_curve",
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
        "title": "3. The 15 AI Input Features",
        "body": (
            "For each trial, the AI receives 15 numbers describing your recent performance:\n\n"
            "  1. Reaction Time (normalised)  — how fast you responded\n"
            "  2. Correctness                 — 1.0 = right, 0.0 = error\n"
            "  3. Elapsed Time (normalised)   — how long you have been playing\n"
            "  4. Rolling Error Rate (5)      — errors in last 5 trials\n"
            "  5. Rolling Error Rate (10)     — errors in last 10 trials\n"
            "  6. Inter-Trial Interval        — time gap between trials\n"
            "  7. Cumulative Error Rate       — total errors so far\n"
            "  8. Correct Streak              — consecutive correct answers\n"
            "  9. Error Streak                — consecutive wrong answers\n"
            " 10. Target Match                — was the right shape shown?\n"
            " 11. Action                      — did you click or skip?\n\n"
            " 12. Reaction-time variance\n"
            " 13. Time since the last error\n"
            " 14. Reaction-time trend\n"
            " 15. Fatigue index\n\n"
            "The AI looks at the last 20 trials at once, forming a sequence of shape (20, 15)."
        ),
        "visual": "features",
    },
    {
        "title": "4. The LSTM Model",
        "body": (
            "LSTM (Long Short-Term Memory) is a recurrent neural network designed "
            "for sequential data. Unlike a regular network, it has memory gates that decide "
            "what to remember and what to forget across time.\n\n"
            "Architecture:\n"
            "  Input (20 trials x 15 features)\n"
            "    -> Linear projection to 128 dimensions\n"
            "    -> 3 stacked LSTM layers with residual connections\n"
            "    -> Layer normalisation between each layer\n"
            "    -> Last hidden state -> classification head\n"
            "    -> Sigmoid -> P(error in next 5 steps)\n\n"
            "LSTM processes the sequence step-by-step, left-to-right. "
            "Each trial's output depends on the current input AND the hidden state "
            "carried forward from all previous trials."
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
            "  Input (20 x 15)\n"
            "    -> Linear projection to 128 dimensions\n"
            "    -> Sinusoidal positional encoding (encodes time order)\n"
            "    -> 4 encoder layers (each: multi-head attention + FFN)\n"
            "    -> Global average pool over the 20 timesteps\n"
            "    -> Classification head -> P(error in next 5 steps)\n\n"
            "Attention Heatmap: after training, you can visualise which past trials "
            "the model attended to most."
        ),
        "visual": "transformer_arch",
    },
    {
        "title": "6. Predictions & Uncertainty",
        "body": (
            "The model outputs P(error) — a probability between 0.0 and 1.0:\n\n"
            "  0.0 - 0.3   Low risk. Performing well.\n"
            "  0.3 - 0.65  Moderate risk. Slight drift detected.\n"
            "  0.65 - 1.0  HIGH RISK. Warning triggered.\n\n"
            "Monte Carlo Dropout (uncertainty estimation):\n"
            "The model runs 50 forward passes with random dropout active, "
            "producing 50 slightly different predictions. The standard deviation "
            "of these predictions is the UNCERTAINTY.\n\n"
            "A warning fires if EITHER:\n"
            "  - P(error) > 0.65  (high probability)\n"
            "  - Uncertainty > 0.20  (high model confusion)\n\n"
            "Calibration (ECE) compares predicted probability with observed error frequency on evaluation data."
        ),
        "visual": "probability_gauge",
    },
    {
        "title": "7. Reading the Results",
        "body": (
            "After training, plots are generated in driftsync/results/:\n\n"
            "  ROC Curve: tradeoff between catching real errors (TPR) and false alarms "
            "(FPR). AUC closer to 1.0 = better. Random guessing = 0.5.\n\n"
            "  Confusion Matrix: 2x2 grid — actual vs predicted. "
            "Top-right = missed errors, bottom-left = false alarms.\n\n"
            "  Calibration Plot: perfect model follows the diagonal. "
            "Points above = underconfident, below = overconfident.\n\n"
            "  Attention Heatmap (Transformer): which past trials the model "
            "attended to. Bright diagonal = local attention.\n\n"
            "  Training History: loss and accuracy curves. "
            "Good training = decreasing loss, increasing accuracy."
        ),
        "visual": "results_guide",
    },
]


class DemoWorker:
    STEPS = [
        "Generating data",
        "Preprocessing",
        "Training LSTM",
        "Training Transformer",
        "Comparing models",
    ]

    def __init__(self, results_dir=Path("driftsync/results"),
                 num_trials=150, num_sessions=15):
        self.results_dir  = results_dir
        self.num_trials   = num_trials
        self.num_sessions = num_sessions
        self.log_queue    = queue.Queue()
        self.metric_queue = queue.Queue()
        self.status       = "idle"
        self.error_msg    = ""
        self.current_step = 0
        self.total_steps  = len(self.STEPS)
        self._thread      = None
        self._handler     = None

    def start(self) -> None:
        self.status = "running"
        self.current_step = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def poll_logs(self) -> List[str]:
        lines = []
        try:
            while True:
                lines.append(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        return lines

    def poll_metrics(self) -> Optional[dict]:
        latest = None
        try:
            while True:
                latest = self.metric_queue.get_nowait()
        except queue.Empty:
            pass
        return latest

    def _attach_logging(self) -> None:
        self._handler = QueueLogHandler(self.log_queue)
        logging.getLogger("driftsync").addHandler(self._handler)

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
            self._log("=" * 50)
            self._log("  Pipeline complete.")
            self._log("=" * 50)
        except Exception as e:
            import traceback
            self.status    = "error"
            self.error_msg = str(e)
            self._log(f"ERROR: {e}")
            self._log(traceback.format_exc())
        finally:
            self._detach_logging()

    def _execute_pipeline(self) -> None:
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

        self.current_step = 0
        self._log(f"Generating {self.num_sessions} sessions ({self.num_trials} trials each)...")
        sim_cfg = SimulatorConfig(num_trials=self.num_trials)
        generate_dataset(num_sessions=self.num_sessions, cfg=sim_cfg, base_seed=42)
        self._log("Data generation complete.")

        self.current_step = 1
        self._log("Preprocessing raw sessions -> feature sequences...")
        data_cfg = DataConfig(sequence_length=20, prediction_horizon=5)
        raw_df   = load_all_sessions(data_cfg.raw_data_dir)
        proc_df  = preprocess_all_sessions(raw_df, horizon=data_cfg.prediction_horizon)
        X, y     = build_sequences_from_df(proc_df, seq_len=data_cfg.sequence_length)
        save_processed(X, y, data_cfg.processed_data_dir)
        self._log(f"Sequences: {X.shape}   Positive rate: {y.mean():.1%}")
        input_dim = X.shape[2]

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)
        train_loader, val_loader, _ = make_dataloaders(
            X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64
        )

        ckpt_dir  = str(self.results_dir / "checkpoints")
        train_cfg = TrainingConfig(
            max_epochs=40, early_stop_patience=10,
            learning_rate=1e-3, batch_size=64, checkpoint_dir=ckpt_dir,
        )
        from driftsync.configs import CONFIG as _CFG
        _CFG.training.checkpoint_dir = ckpt_dir

        self.current_step = 2
        self._log("Training LSTM model (up to 40 epochs)...")
        lstm_cfg = LSTMConfig(input_dim=input_dim)
        lstm     = build_model("lstm", lstm_cfg)
        self._log(f"LSTM parameters: {lstm.count_parameters():,}")

        def lstm_cb(epoch, metrics):
            metrics["model"] = "LSTM"
            metrics["epoch"] = epoch
            self.metric_queue.put(metrics)

        Trainer(lstm, train_loader, val_loader, train_cfg, device).train(on_epoch_end=lstm_cb)

        self.current_step = 3
        self._log("Training Transformer model (up to 40 epochs)...")
        tf_cfg = TransformerConfig(input_dim=input_dim)
        tf     = build_model("transformer", tf_cfg)
        self._log(f"Transformer parameters: {tf.count_parameters():,}")

        def tf_cb(epoch, metrics):
            metrics["model"] = "Transformer"
            metrics["epoch"] = epoch
            self.metric_queue.put(metrics)

        Trainer(tf, train_loader, val_loader, train_cfg, device).train(on_epoch_end=tf_cb)

        self.current_step = 4
        self._log("Training baseline sklearn models (LogisticRegression + RandomForest)...")
        try:
            from driftsync.ml.baseline_models import train_baseline_models
            _mode, _mdl = train_baseline_models(X_train, y_train)
            self._log(f"Baseline models trained: {_mode}")
        except Exception as _be:
            self._log(f"Baseline model training skipped: {_be}")

        self._log("Comparing models and generating plots...")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        run_comparison(data_cfg=data_cfg, results_dir=str(self.results_dir))
        self._log(f"Plots saved to {self.results_dir}")


class DriftSyncApplication:
    """State-machine Pygame application with left sidebar navigation."""

    _NAV_ITEMS = [
        ("Learn",     State.LEARN),
        ("Run Demo",  State.DEMO),
        ("Play Task", State.PLAY_TASK),
        ("Results",   State.RESULTS),
        ("Live AI",   State.LIVE_MODE),
    ]

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("DriftSync  [F11 = fullscreen]")
        self.fullscreen = False
        self.viewport = ui.Viewport((W, H), "DriftSync | Focus analysis")
        self.screen = self.viewport.surface
        self.clock  = pygame.time.Clock()
        self._init_fonts()

        self.state        = State.MENU
        self.splash_start = time.time()
        self.learn_page   = 0
        self.math_view = MathematicsView()
        self.running      = True

        self.demo_worker       = DemoWorker()
        self.demo_logs: List[str] = []
        self.demo_metrics: dict   = {}
        self.demo_train_loss  = deque(maxlen=60)
        self.demo_val_loss    = deque(maxlen=60)
        self.demo_val_auc     = deque(maxlen=60)
        self.demo_view_results_btn = None
        self.demo_num_trials   = 150
        self.demo_num_sessions = 15
        self._demo_name_input  = None

        self.result_runs: List[Path] = []
        self.result_run_idx  = 0
        self.result_thumbs: List[Tuple[pygame.Surface, str, Path]] = []
        self.result_scroll   = 0
        self.result_full_view: Optional[pygame.Surface] = None
        self.result_full_title = ""
        self.result_metrics: dict = {}
        self.results_tab     = "ml"
        self.human_sessions: List[dict] = []
        self.human_scroll    = 0
        self.human_selected  = None

        self.play_num_trials     = 150
        self._play_session_name  = ""
        self._play_task_done     = False
        self._play_name_input    = None
        self._play_skip_calib    = False

        self.live_model_choice = "lstm"
        self.notice = ""
        self._focus_index = -1
        self.learn_scroll = 0

        # Session metrics (loaded from driftsync/sessions/*.json)
        self.session_metrics: list = []

        self._build_menu_buttons()
        self._build_learn_buttons()
        self._build_play_task_buttons()

    def _init_fonts(self) -> None:
        ui.font.cache_clear()
        self.f_title = ui.font(36, True)
        self.f_head = ui.font(25, True)
        self.f_sub = ui.font(20, True)
        self.f_body = ui.font(17)
        self.f_small = ui.font(14)
        self.f_mono = ui.font(13, mono=True)
        self.f_btn = ui.font(17, True)
        self.f_btn_sm = ui.font(14)

    def _mouse(self):
        return self.viewport.point(pygame.mouse.get_pos())

    @property
    def _cx(self) -> int:
        return SIDEBAR_W

    def run(self) -> None:
        while self.running:
            self._handle_events()
            self._update()
            self._render()
            self.clock.tick(FPS)
        pygame.quit()

    # Sidebar

    def _render_sidebar(self) -> None:
        draw_rect(self.screen, pygame.Rect(0, 0, SIDEBAR_W, H), SIDEBAR_BG, 0)
        vline(self.screen, SIDEBAR_W, 0, H)
        ui.signal_mark(self.screen, 20, 26, 25)
        draw_text(self.screen, "DriftSync", self.f_sub, TEXT, 54, 26)
        draw_text(self.screen, "Focus analysis", self.f_small, DIM, 20, 62)
        nav = [("Workspace", State.MENU), ("Live session", State.LIVE_MODE),
               ("Record task", State.PLAY_TASK), ("Train models", State.DEMO),
               ("Results", State.RESULTS), ("Field guide", State.LEARN), ("Mathematics", State.MATHEMATICS)]
        self._nav_rects = []
        for i, (label, target) in enumerate(nav):
            rect = pygame.Rect(12, 120 + i * 48, SIDEBAR_W - 24, 40)
            active = self.state == target
            if active or rect.collidepoint(self._mouse()):
                draw_rect(self.screen, rect, PANEL2 if active else PANEL)
            if active:
                pygame.draw.circle(self.screen, ACCENT, (rect.right - 14, rect.centery), 3)
            draw_text(self.screen, label, self.f_body, TEXT if active else DIM,
                      rect.x + 12, rect.centery, "midleft")
            self._nav_rects.append((rect, target))
        hline(self.screen, H - 142, 20, SIDEBAR_W - 20)
        draw_text(self.screen, f"DriftSync {__version__}", self.f_small, DIM, 20, H - 124)
        draw_text(self.screen, "Synthetic training data", self.f_small, DIM, 20, H - 102)
        draw_text(self.screen, "Paul Nercessian", self.f_small, TEXT, 20, H - 64)
        draw_text(self.screen, "Alt 1-7  Navigate", self.f_small, DIM, 20, H - 40)

    def _handle_sidebar_click(self, pos) -> Optional[State]:
        if self.result_full_view is not None:
            return None
        for rect, target in getattr(self, "_nav_rects", []):
            if rect.collidepoint(pos):
                return target
        return None

    def _focus_controls(self):
        if self.result_full_view is not None or self.state == State.MATHEMATICS:
            return []
        prefixes = {State.MENU: ("_menu_",), State.PLAY_TASK: ("_btn_play_", "_play_name"),
                    State.DEMO: ("_btn_demo_", "_btn_trials_", "_btn_sessions_", "_demo_name", "demo_view"),
                    State.LEARN: ("_btn_learn_",), State.LIVE_MODE: ("_live_btn",)}
        controls = []
        for key, value in vars(self).items():
            if key.startswith(prefixes.get(self.state, ("_btn_results_", "_btn_human_", "_btn_export_"))):
                if isinstance(value, (Button, TextInput)) and not getattr(value, "disabled", False):
                    if key == "_btn_play_human_results" and not self._play_task_done:
                        continue
                    controls.append(value)
        if self.state == State.MENU:
            controls = [btn for btn, _ in self._menu_btns] + [self._menu_quit_btn]
        elif self.state == State.RESULTS:
            names = ("_btn_results_open",) if self.results_tab == "ml" else ("_btn_human_refresh", "_btn_human_open", "_btn_export_csv")
            controls = [getattr(self, key) for key in names if hasattr(self, key)]
        elif self.state == State.DEMO and self.demo_worker.status != "idle":
            controls = [button for button in (self._btn_demo_start, self.demo_view_results_btn)
                        if button is not None and not button.disabled]
        return controls

    # Event / update / render dispatch

    def _handle_events(self) -> None:
        for raw_event in pygame.event.get():
            event = self.viewport.event(raw_event)
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN and getattr(event, "mod", 0) & pygame.KMOD_ALT:
                targets = (State.MENU, State.LIVE_MODE, State.PLAY_TASK, State.DEMO, State.RESULTS, State.LEARN, State.MATHEMATICS)
                if pygame.K_1 <= event.key <= pygame.K_7:
                    self._enter_state(targets[event.key - pygame.K_1])
                    continue

            if self.state == State.MATHEMATICS and event.type == pygame.KEYDOWN and event.key in (pygame.K_TAB, pygame.K_RETURN, pygame.K_SPACE, pygame.K_LEFT, pygame.K_RIGHT):
                self.math_view.handle_event(event)
                continue

            if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                controls = self._focus_controls()
                if controls:
                    self._focus_index = (self._focus_index + (-1 if event.mod & pygame.KMOD_SHIFT else 1)) % len(controls)
                    for i, control in enumerate(controls):
                        if isinstance(control, TextInput):
                            control.active = i == self._focus_index
                        else:
                            control.focused = i == self._focus_index
                continue

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if self.result_full_view is not None:
                    self.result_full_view = None
                    continue
                if self.state == State.MENU:
                    self.running = False
                else:
                    self.state = State.MENU
                return

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._focus_index = -1
                for control in self._focus_controls():
                    if isinstance(control, Button):
                        control.focused = False
                target = self._handle_sidebar_click(event.pos)
                if target is not None:
                    self._enter_state(target)
                    return

            if self.state == State.MENU:
                self._handle_menu_event(event)
            elif self.state == State.LEARN:
                self._handle_learn_event(event)
            elif self.state == State.DEMO:
                self._handle_demo_event(event)
            elif self.state == State.RESULTS:
                self._handle_results_event(event)
            elif self.state == State.PLAY_TASK:
                self._handle_play_event(event)
            elif self.state == State.LIVE_MODE:
                self._handle_live_event(event)
            elif self.state == State.MATHEMATICS:
                self.math_view.handle_event(event)

    def _enter_state(self, target: State) -> None:
        self._focus_index = -1
        if target == State.RESULTS:
            self._load_ml_results()
            self._load_human_sessions()
            self._load_session_metrics()
        elif target == State.DEMO:
            if not hasattr(self, "_btn_demo_start"):
                self._build_demo_buttons()
        elif target == State.PLAY_TASK:
            self._play_task_done = False
            self._build_play_task_buttons()
        self.state = target

    def _update(self) -> None:
        if self.state == State.SPLASH:
            if time.time() - self.splash_start > 2.5:
                self.state = State.MENU

        if self.demo_worker.status != "idle":
            for line in self.demo_worker.poll_logs():
                self.demo_logs.append(line)
            if len(self.demo_logs) > 150:
                self.demo_logs = self.demo_logs[-150:]
            m = self.demo_worker.poll_metrics()
            if m:
                self.demo_metrics = m
                if "train_loss" in m:
                    self.demo_train_loss.append(m["train_loss"])
                    self.demo_val_loss.append(m["val_loss"])
                    self.demo_val_auc.append(m.get("val_auc", 0.0))
            if self.demo_view_results_btn:
                self.demo_view_results_btn.disabled = (self.demo_worker.status != "done")

    def _render(self) -> None:
        self.screen.fill(BG)

        if self.state == State.SPLASH:
            self._render_splash()
            self.viewport.present()
            return

        self._render_sidebar()

        if self.state == State.MENU:
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
        elif self.state == State.MATHEMATICS:
            self.math_view.render(self.screen)

        for i, control in enumerate(self._focus_controls()):
            if isinstance(control, Button):
                control.focused = i == self._focus_index
                if control.focused and not control.disabled:
                    draw_border(self.screen, control.rect.inflate(6, 6), ACCENT, 5, 2)
        self.viewport.present()

    # SPLASH

    def _render_splash(self) -> None:
        t = time.time() - self.splash_start
        a = min(255, int(t / 0.8 * 255))

        s = self.f_title.render("DriftSync", True, TEXT)
        s.set_alpha(a)
        self.screen.blit(s, s.get_rect(center=(W // 2, H // 2 - 30)))

        s2 = self.f_body.render("Real-Time Neural Cognitive Drift Prediction System", True, DIM)
        s2.set_alpha(max(0, int((t - 0.4) / 0.5 * 255)))
        self.screen.blit(s2, s2.get_rect(center=(W // 2, H // 2 + 14)))

        s3 = self.f_small.render("by Paul Nercessian", True, DIM)
        s3.set_alpha(max(0, int((t - 1.0) / 0.5 * 255)))
        self.screen.blit(s3, s3.get_rect(midbottom=(W // 2, H - 22)))

    # MENU

    def _build_menu_buttons(self) -> None:
        x = self._cx + 40
        self._menu_btns = []
        items = [("Start live session", State.LIVE_MODE), ("Record a task", State.PLAY_TASK),
                 ("Train models", State.DEMO), ("Review results", State.RESULTS), ("Read the field guide", State.LEARN)]
        for i, (label, target) in enumerate(items):
            rect = pygame.Rect(x, 276, 218, 46) if i == 0 else pygame.Rect(x, 430 + (i - 1) * 62, 280, 42)
            self._menu_btns.append((Button(rect, label, self.f_btn, accent_fill=i == 0), target))
        self._menu_quit_btn = Button(pygame.Rect(W - 132, H - 56, 92, 34), "Quit", self.f_btn_sm)

    def _render_menu(self) -> None:
        x = self._cx + 40
        draw_text(self.screen, "Workspace", self.f_head, TEXT, x, 32)
        draw_text(self.screen, "Ready for a session", self.f_small, DIM, W - 40, 42, "topright")
        hline(self.screen, 86, x, W - 40)
        draw_text(self.screen, "See when focus starts to drift.", self.f_title, TEXT, x, 125)
        ui.paragraph(self.screen, "Follow a timed attention task. Inspect predicted error risk alongside the behaviour behind it.",
                     pygame.Rect(x, 190, 580, 64), self.f_body)
        draw_text(self.screen, "Live model analysis", self.f_sub, TEXT, x + 258, 280)
        draw_text(self.screen, "20 trials of history. A 5-trial forecast.", self.f_small, DIM, x + 258, 310)
        hline(self.screen, 378, x, W - 40)
        draw_text(self.screen, "Explore the workflow", self.f_sub, TEXT, x, 396)
        descriptions = ["Collect reaction times and accuracy, with optional calibration.",
                        "Generate synthetic sessions and compare sequence models.",
                        "Inspect experiment plots, recorded sessions and exports.",
                        "Understand the task, model inputs and uncertainty."]
        for i, description in enumerate(descriptions):
            draw_text(self.screen, description, self.f_body, DIM, x + 306, 440 + i * 62)
        for btn, _ in self._menu_btns:
            btn.hovered = btn.rect.collidepoint(self._mouse())
            btn.draw(self.screen)
        self._menu_quit_btn.draw(self.screen)
        draw_text(self.screen, "Research demonstration. Predictions are estimates of task error risk.",
                  self.f_small, DIM, x, H - 45)
        if self.notice:
            draw_text(self.screen, self.notice[:110], self.f_small, YELLOW, x, 346)

    def _handle_menu_event(self, event) -> None:
        for btn, target in self._menu_btns:
            if btn.handle_event(event):
                self._enter_state(target)
                return
        if self._menu_quit_btn.handle_event(event):
            self.running = False

    # LEARN

    def _build_learn_buttons(self) -> None:
        bw, bh = 150, 40
        by = H - 54
        cx = self._cx
        self._btn_learn_back = Button(
            pygame.Rect(cx + 12, by, bw, bh), "Previous",
            self.f_btn, color=PANEL2, text_color=DIM, hover_text=TEXT)
        self._btn_learn_next = Button(
            pygame.Rect(W - 12 - bw, by, bw, bh), "Next",
            self.f_btn, accent_fill=True)
        self._btn_learn_menu = Button(
            pygame.Rect(cx + (W - cx) // 2 - 75, by, 150, bh), "Workspace",
            self.f_btn, color=PANEL2, text_color=DIM, hover_text=TEXT)

    def _render_learn(self) -> None:
        page = LEARN_PAGES[self.learn_page]
        self._render_screen_header("Field guide")
        x = self._cx + 24
        for i, p in enumerate(LEARN_PAGES):
            rect = pygame.Rect(x, 82 + i * 48, 230, 40)
            if i == self.learn_page:
                draw_rect(self.screen, rect, PANEL2)
            title = p["title"].split(". ", 1)[-1]
            ui.paragraph(self.screen, title, rect.inflate(-16, -6), self.f_small,
                         TEXT if i == self.learn_page else DIM, 2)
        vline(self.screen, x + 254, 80, H - 90)
        content = pygame.Rect(x + 286, 144, W - x - 330, H - 244)
        draw_text(self.screen, page["title"], self.f_head, TEXT, content.x, 84)
        # Scroll long educational content without losing the footer controls.
        old = self.screen.get_clip()
        self.screen.set_clip(content)
        height = ui.paragraph(self.screen, page["body"],
                             pygame.Rect(content.x, content.y - self.learn_scroll, content.width, 2400),
                             self.f_body, TEXT, 7) - (content.y - self.learn_scroll)
        self.screen.set_clip(old)
        self._learn_max_scroll = max(0, height - content.height)
        self.learn_scroll = min(self.learn_scroll, self._learn_max_scroll)
        draw_text(self.screen, f"{self.learn_page + 1} of {len(LEARN_PAGES)}    Scroll to read", self.f_small, DIM,
                  content.x, H - 88)
        self._btn_learn_back.disabled = self.learn_page == 0
        self._btn_learn_next.disabled = self.learn_page == len(LEARN_PAGES) - 1
        for btn in (self._btn_learn_back, self._btn_learn_next, self._btn_learn_menu):
            btn.draw(self.screen)

    def _render_learn_body(self, text: str, rect: pygame.Rect) -> None:
        lines = text.split("\n")
        y = rect.top
        lh = self.f_body.get_height() + 3
        for line in lines:
            if y + lh > rect.bottom:
                break
            if not line.strip():
                y += 9
                continue
            stripped = line.lstrip()
            indent   = len(line) - len(stripped)
            color    = TEXT
            avail_w  = rect.width - indent * 9
            words, cur = stripped.split(), ""
            for w in words:
                test = (cur + " " + w).strip()
                if self.f_body.size(test)[0] <= avail_w:
                    cur = test
                else:
                    if cur:
                        s = self.f_body.render(cur, True, color)
                        self.screen.blit(s, (rect.left + indent * 9, y))
                        y += lh
                    cur = w
                    if y + lh > rect.bottom:
                        break
            if cur and y + lh <= rect.bottom:
                s = self.f_body.render(cur, True, color)
                self.screen.blit(s, (rect.left + indent * 9, y))
                y += lh

    def _render_learn_visual(self, vis: str, cx: int, cw: int, y: int) -> None:
        avail = H - 66 - y
        if avail < 30:
            return
        if vis == "drift_curve":
            self._draw_drift_curve(cx, y, cw, min(avail, 72))
        elif vis == "probability_gauge":
            self._draw_gauge_demo(cx, y, cw)
        elif vis == "lstm_arch":
            draw_text(self.screen,
                      "Input(20,15) -> Proj -> LSTM x3 -> Head -> sigmoid -> P(error)",
                      self.f_small, TEXT, cx, y)
        elif vis == "transformer_arch":
            draw_text(self.screen,
                      "Input(20,15) -> Proj -> PosEnc -> Attn x4 -> Pool -> Head -> P(error)",
                      self.f_small, TEXT, cx, y)

    def _draw_drift_curve(self, x, y, w, h) -> None:
        pts = []
        for i in range(w):
            t   = i / w
            val = 0.85 * (1 - 0.7 * (1 - math.exp(-5 * (1 - t))))
            val += 0.03 * math.sin(i * 0.3)
            pts.append((x + i, y + h - int(val * h)))
        pygame.draw.lines(self.screen, GREEN, False, pts, 2)
        draw_text(self.screen, "accuracy", self.f_small, DIM, x, y)
        draw_text(self.screen, "time ->", self.f_small, DIM, x + w - 54, y + h)

    def _draw_gauge_demo(self, x, y, w) -> None:
        labels = [("LOW 0.15", 0.15, GREEN), ("MED 0.45", 0.45, YELLOW), ("HIGH 0.78", 0.78, RED)]
        gw = min(w // 3 - 18, 200)
        for i, (lbl, val, col) in enumerate(labels):
            gx = x + i * (gw + 16)
            draw_text(self.screen, lbl, self.f_small, col, gx, y)
            draw_progress_bar(self.screen, pygame.Rect(gx, y + 18, gw, 11), val, fg=col)

    def _handle_learn_event(self, event) -> None:
        previous_page = self.learn_page
        if event.type == pygame.MOUSEWHEEL:
            self.learn_scroll = max(0, min(getattr(self, "_learn_max_scroll", 0), self.learn_scroll - event.y * 40))
        if self._btn_learn_next.handle_event(event) and self.learn_page < len(LEARN_PAGES) - 1:
            self.learn_page += 1
        if self._btn_learn_back.handle_event(event) and self.learn_page > 0:
            self.learn_page -= 1
        if self._btn_learn_menu.handle_event(event):
            self.state = State.MENU
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_RIGHT, pygame.K_SPACE) and self.learn_page < len(LEARN_PAGES) - 1:
                self.learn_page += 1
            if event.key == pygame.K_LEFT and self.learn_page > 0:
                self.learn_page -= 1
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            cx = self._cx
            for i in range(len(LEARN_PAGES)):
                if pygame.Rect(cx + 24, 82 + i * 48, 230, 40).collidepoint(event.pos):
                    self.learn_page = i

        if previous_page != self.learn_page:
            self.learn_scroll = 0

    def _render_screen_header(self, title: str) -> None:
        cx = self._cx
        draw_rect(self.screen, pygame.Rect(cx, 0, W - cx, 50), PANEL, 0)
        hline(self.screen, 50, cx, W)
        draw_text(self.screen, title, self.f_head, TEXT, cx + 14, 13)
        self._hdr_home_btn = Button(
            pygame.Rect(W - 118, 8, 106, 34), "Workspace",
            self.f_btn_sm, color=PANEL2, text_color=DIM, hover_text=ACCENT)
        mx, my = self._mouse()
        self._hdr_home_btn.hovered = self._hdr_home_btn.rect.collidepoint(mx, my)
        self._hdr_home_btn.draw(self.screen)

    # DEMO

    def _build_demo_buttons(self) -> None:
        cx = self._cx + 12
        self._btn_demo_start = Button(
            pygame.Rect(cx, 58, 140, 40), "Start training",
            self.f_btn, accent_fill=True)
        self._btn_demo_start.disabled = (self.demo_worker.status == "running")

        self.demo_view_results_btn = Button(
            pygame.Rect(W - 180, 58, 168, 40), "View results",
            self.f_btn, accent_fill=True)
        self.demo_view_results_btn.disabled = True

        self._demo_name_input = TextInput(
            pygame.Rect(cx, 108, 230, 34), self.f_body, placeholder="Run name (optional)")

        self._btn_trials_minus = Button(
            pygame.Rect(cx + 322, 108, 34, 34), "-", self.f_btn,
            color=PANEL2, text_color=TEXT, hover_text=ACCENT)
        self._btn_trials_plus = Button(
            pygame.Rect(cx + 414, 108, 34, 34), "+", self.f_btn,
            color=PANEL2, text_color=TEXT, hover_text=ACCENT)

        self._btn_sessions_minus = Button(
            pygame.Rect(cx + 550, 108, 34, 34), "-", self.f_btn,
            color=PANEL2, text_color=TEXT, hover_text=ACCENT)
        self._btn_sessions_plus = Button(
            pygame.Rect(cx + 642, 108, 34, 34), "+", self.f_btn,
            color=PANEL2, text_color=TEXT, hover_text=ACCENT)

    def _render_demo(self) -> None:
        cx    = self._cx
        idle  = self.demo_worker.status == "idle"
        run   = self.demo_worker.status == "running"
        done  = self.demo_worker.status == "done"
        error = self.demo_worker.status == "error"

        self._render_screen_header("Train models")
        sc  = GREEN if done else RED if error else YELLOW if run else DIM
        slb = {"idle": "idle", "running": "running...", "done": "done", "error": "error"}.get(
            self.demo_worker.status, "")
        draw_text(self.screen, slb, self.f_body, sc, W - 148, 16, anchor="topright")

        if hasattr(self, "_btn_demo_start"):
            self._btn_demo_start.disabled = run
            self._btn_demo_start.text = "Start training" if idle else "Training..." if run else "New run"
            self._btn_demo_start.draw(self.screen)
        if self.demo_view_results_btn:
            self.demo_view_results_btn.draw(self.screen)

        if idle and self._demo_name_input:
            dt = self.clock.get_time() / 1000.0
            self._demo_name_input.draw(self.screen, dt)

            draw_text(self.screen, "Trials:", self.f_small, DIM, cx + 262, 114)
            self._btn_trials_minus.draw(self.screen)
            draw_text(self.screen, str(self.demo_num_trials), self.f_body, TEXT,
                      cx + 397, 114, anchor="midtop")
            self._btn_trials_plus.draw(self.screen)

            draw_text(self.screen, "Sessions:", self.f_small, DIM, cx + 475, 114)
            self._btn_sessions_minus.draw(self.screen)
            draw_text(self.screen, str(self.demo_num_sessions), self.f_body, TEXT,
                      cx + 625, 114, anchor="midtop")
            self._btn_sessions_plus.draw(self.screen)

        step  = self.demo_worker.current_step
        n     = self.demo_worker.total_steps
        prog  = (step + (1.0 if done else 0.0)) / n
        pb_y  = 152
        draw_progress_bar(self.screen, pygame.Rect(cx + 12, pb_y, W - cx - 24, 10), prog, fg=ACCENT)

        step_w = (W - cx - 24) // n
        for i, lbl in enumerate(DemoWorker.STEPS):
            sx  = cx + 12 + i * step_w
            col = GREEN if (i < step or done) else YELLOW if (i == step and run) else DIM
            draw_text(self.screen, lbl, self.f_small, col,
                      sx + step_w // 2, pb_y + 14, anchor="midtop")

        panel_y = pb_y + 38

        LOG_W    = int((W - cx) * 0.58) - 10
        log_rect = pygame.Rect(cx + 12, panel_y, LOG_W, H - panel_y - 14)
        draw_rect(self.screen, log_rect, PANEL, 5)
        draw_border(self.screen, log_rect, BORDER, 5)
        draw_text(self.screen, "Training log", self.f_small, DIM, log_rect.left + 8, log_rect.top + 6)
        hline(self.screen, log_rect.top + 22, log_rect.left, log_rect.right)

        lh       = self.f_mono.get_height() + 2
        vis_lines = (log_rect.height - 26) // lh
        start_idx = max(0, len(self.demo_logs) - vis_lines)
        if not self.demo_logs:
            ui.paragraph(self.screen, "Training output will appear here.\nRuns use generated synthetic task sessions.",
                         pygame.Rect(log_rect.x + 20, log_rect.y + 48, log_rect.width - 40, 100), self.f_body)
        self.screen.set_clip(log_rect.inflate(-12, -4))
        ly = log_rect.top + 26
        for line in self.demo_logs[start_idx:]:
            col = GREEN if any(k in line.lower() for k in ("complete", "done", "saved")) else \
                  RED   if "error" in line.lower() else \
                  YELLOW if any(k in line.lower() for k in ("training", "epoch", "generating", "preprocessing")) \
                  else TEXT
            s = self.f_mono.render(line[:78], True, col)
            self.screen.blit(s, (log_rect.left + 8, ly))
            ly += lh
            if ly > log_rect.bottom - 4:
                break

        self.screen.set_clip(None)
        mx_r     = cx + 12 + LOG_W + 10
        mw       = W - mx_r - 12
        met_rect = pygame.Rect(mx_r, panel_y, mw, H - panel_y - 14)
        draw_rect(self.screen, met_rect, PANEL, 5)
        draw_border(self.screen, met_rect, BORDER, 5)
        draw_text(self.screen, "Model performance", self.f_small, DIM, met_rect.left + 8, met_rect.top + 6)
        hline(self.screen, met_rect.top + 22, met_rect.left, met_rect.right)

        m = self.demo_metrics
        if m:
            model_col = LSTM_C if m.get("model") == "LSTM" else TF_C
            my = met_rect.top + 30
            draw_text(self.screen, m.get("model", "?"), self.f_sub, model_col, met_rect.left + 10, my)
            my += 28

            epoch  = m.get("epoch", 0)
            max_ep = m.get("max_epochs", 40)
            draw_text(self.screen, f"epoch  {epoch} / {max_ep}", self.f_body, TEXT, met_rect.left + 10, my)
            ep_bar = pygame.Rect(met_rect.left + 10, my + 18, mw - 20, 7)
            draw_progress_bar(self.screen, ep_bar, epoch / max(max_ep, 1), fg=model_col)
            my += 34

            def row(label, val, good_low=True, thresh=0.3):
                nonlocal my
                if good_low:
                    col = GREEN if val < thresh else YELLOW if val < thresh * 2 else RED
                else:
                    col = GREEN if val > thresh else YELLOW if val > thresh * 0.8 else RED
                draw_text(self.screen, label, self.f_body, TEXT, met_rect.left + 10, my)
                draw_text(self.screen, f"{val:.4f}", self.f_body, col,
                          met_rect.right - 10, my, anchor="topright")
                my += 22

            row("train_loss", m.get("train_loss", 0), good_low=True, thresh=0.3)
            row("val_loss",   m.get("val_loss", 0),   good_low=True, thresh=0.3)
            row("val_auc",    m.get("val_auc", 0),    good_low=False, thresh=0.75)
            row("val_f1",     m.get("val_f1", 0),     good_low=False, thresh=0.70)
            my += 6

            if self.demo_train_loss:
                draw_text(self.screen, "loss curve", self.f_small, DIM, met_rect.left + 10, my)
                my += 16
                sp = pygame.Rect(met_rect.left + 10, my, mw - 20, 50)
                draw_rect(self.screen, sp, BG, 3)
                all_l = list(self.demo_train_loss) + list(self.demo_val_loss)
                ym = max(all_l) + 0.01 if all_l else 1.0
                draw_sparkline(self.screen, list(self.demo_train_loss), sp, LSTM_C, 0, ym)
                draw_sparkline(self.screen, list(self.demo_val_loss),   sp, RED,    0, ym)
                draw_text(self.screen, "train", self.f_small, LSTM_C, met_rect.left + 12, my + 52)
                draw_text(self.screen, "val",   self.f_small, RED,    met_rect.left + 58, my + 52)
                my += 72

            if self.demo_val_auc:
                draw_text(self.screen, "val AUC", self.f_small, DIM, met_rect.left + 10, my)
                my += 16
                ar = pygame.Rect(met_rect.left + 10, my, mw - 20, 44)
                draw_rect(self.screen, ar, BG, 3)
                draw_sparkline(self.screen, list(self.demo_val_auc), ar, GREEN, 0, 1.0)
                thr_y = ar.bottom - int(0.7 * ar.height)
                pygame.draw.line(self.screen, BORDER, (ar.left, thr_y), (ar.right, thr_y), 1)
                draw_text(self.screen, "0.70", self.f_small, DIM, ar.right + 3, thr_y)
        else:
            msg = {
                "idle":    "Configure a run above to begin training. Epoch metrics will appear here.",
                "running": "Waiting for first epoch...",
                "done":    "Complete.",
                "error":   f"Error: {self.demo_worker.error_msg[:58]}",
            }.get(self.demo_worker.status, "")
            ui.paragraph(self.screen, msg, met_rect.inflate(-40, -90), self.f_body, DIM)

    def _handle_demo_event(self, event) -> None:
        if self.demo_worker.status in ("done", "error") and self._btn_demo_start.handle_event(event):
            self.demo_worker = DemoWorker()
            self.demo_logs.clear()
            self.demo_metrics.clear()
            self.demo_train_loss.clear()
            self.demo_val_loss.clear()
            self.demo_val_auc.clear()
            self._build_demo_buttons()
            return
        if hasattr(self, "_hdr_home_btn") and self._hdr_home_btn.handle_event(event):
            self.state = State.MENU
            return
        if self.demo_view_results_btn and self.demo_view_results_btn.handle_event(event):
            self._load_ml_results()
            self._load_human_sessions()
            self.state = State.RESULTS

        if self.demo_worker.status == "idle":
            if self._demo_name_input:
                self._demo_name_input.handle_event(event)
            if hasattr(self, "_btn_trials_minus") and self._btn_trials_minus.handle_event(event):
                self.demo_num_trials = max(50, self.demo_num_trials - 25)
            if hasattr(self, "_btn_trials_plus") and self._btn_trials_plus.handle_event(event):
                self.demo_num_trials = min(500, self.demo_num_trials + 25)
            if hasattr(self, "_btn_sessions_minus") and self._btn_sessions_minus.handle_event(event):
                self.demo_num_sessions = max(5, self.demo_num_sessions - 5)
            if hasattr(self, "_btn_sessions_plus") and self._btn_sessions_plus.handle_event(event):
                self.demo_num_sessions = min(50, self.demo_num_sessions + 5)
            if hasattr(self, "_btn_demo_start") and self._btn_demo_start.handle_event(event):
                raw  = (self._demo_name_input.text.strip()
                        if self._demo_name_input and self._demo_name_input.text.strip() else "Run")
                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw)
                dstr = datetime.now().strftime("%Y-%m-%d_%H-%M")
                rdir = Path("driftsync/results") / f"{safe}_{dstr}"
                self.demo_worker = DemoWorker(
                    results_dir=rdir,
                    num_trials=self.demo_num_trials,
                    num_sessions=self.demo_num_sessions,
                )
                self._build_demo_buttons()
                self.demo_worker.start()

    # RESULTS

    def _discover_ml_runs(self) -> List[Path]:
        base = Path("driftsync/results")
        if not base.exists():
            return []
        runs = []
        for d in sorted(base.iterdir(), reverse=True):
            if d.is_dir() and d.name != "checkpoints" and (any(d.glob("*.png")) or (d / "comparison_summary.json").exists()):
                runs.append(d)
        if any(base.glob("*.png")) or (base / "comparison_summary.json").exists() or (base / "experiment_summary.json").exists():
            runs.append(base)
        return runs

    def _load_ml_results(self, run_idx: int = 0) -> None:
        self.result_runs    = self._discover_ml_runs()
        self.result_run_idx = max(0, min(run_idx, len(self.result_runs) - 1))
        self.result_thumbs.clear()
        self.result_full_view = None
        self.result_scroll    = 0

        if self.result_runs:
            sel = self.result_runs[self.result_run_idx]
            for p in sorted(sel.glob("*.png")):
                surf = load_png(p, (290, 196))
                if surf:
                    self.result_thumbs.append((surf, p.stem.replace("_", " ").title(), p))
            jp = sel / "comparison_summary.json"
            if not jp.exists():
                jp = sel / "experiment_summary.json"
            if jp.exists():
                try:
                    with open(jp, encoding="utf-8") as f:
                        raw_metrics = json.load(f)
                    self.result_metrics = {
                        key: value.get("test_metrics", value)
                        for key, value in raw_metrics.items()
                        if isinstance(value, dict) and ("test_metrics" in value or "accuracy" in value)
                    }
                except (OSError, ValueError, AttributeError) as exc:
                    self.result_metrics = {}
                    self.notice = f"Unable to read run metrics: {exc}"

            else:
                self.result_metrics = {}
        else:
            self.result_metrics = {}

    def _load_session_metrics(self) -> None:
        """Load per-session metric summaries from driftsync/sessions/."""
        metrics_dir = Path("driftsync/sessions")
        self.session_metrics = []
        if not metrics_dir.exists():
            return
        for p in sorted(metrics_dir.glob("metrics_*.json"), reverse=True):
            try:
                with open(p, encoding="utf-8") as f:
                    self.session_metrics.append(json.load(f))
            except Exception:
                continue

    def _export_sessions_csv(self) -> None:
        """Export all human session trial data as CSV."""
        raw_dir = Path("driftsync/data/raw")
        if not raw_dir.exists():
            return
        import csv
        from datetime import datetime as _dt
        out = Path("driftsync/sessions") / f"export_{_dt.now().strftime('%Y%m%d_%H%M%S')}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for p in sorted(raw_dir.glob("session_*.json")):
            try:
                with open(p) as f:
                    data = json.load(f)
                for t in data.get("trials", []):
                    t["session_id"] = data.get("session_id", "")
                    rows.append(t)
            except Exception:
                continue
        if rows:
            keys = list(rows[0].keys())
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)

    def _load_human_sessions(self) -> None:
        raw_dir = Path("driftsync/data/raw")
        self.human_sessions = []
        if not raw_dir.exists():
            return
        for p in sorted(raw_dir.glob("session_*.json"), reverse=True):
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                trials = data.get("trials", [])
                if not trials:
                    continue
                acc    = sum(t.get("is_correct", False) for t in trials) / max(1, len(trials))
                avg_rt = sum(t.get("reaction_time", 0) for t in trials) / max(1, len(trials))
                self.human_sessions.append({
                    "path":     p,
                    "id":       data.get("session_id", p.stem),
                    "name":     data.get("session_name", ""),
                    "start":    data.get("start_time", "")[:19].replace("T", " "),
                    "trials":   len(trials),
                    "accuracy": acc,
                    "avg_rt":   avg_rt,
                })
            except Exception:
                continue

    def _render_results(self) -> None:
        if self.result_full_view:
            self._render_full_image()
            return

        cx = self._cx
        self._render_screen_header("Results")

        tab_y = 54
        for i, (label, key) in enumerate([("ML Runs", "ml"), ("Human Sessions", "human")]):
            tw  = 148
            tx  = cx + 12 + i * (tw + 8)
            active = (self.results_tab == key)
            draw_rect(self.screen, pygame.Rect(tx, tab_y, tw, 30), PANEL2 if active else PANEL, 4)
            if active:
                pygame.draw.rect(self.screen, ACCENT, pygame.Rect(tx, tab_y, tw, 2))
            draw_border(self.screen, pygame.Rect(tx, tab_y, tw, 30), ACCENT if active else BORDER, 4)
            draw_text(self.screen, label, self.f_body, TEXT if active else DIM,
                      tx + tw // 2, tab_y + 8, anchor="midtop")
        hline(self.screen, 86, cx, W)

        if self.results_tab == "ml":
            self._render_ml_results(cx, 90)
        else:
            self._render_human_results(cx, 90)

        draw_text(self.screen, "ESC = menu", self.f_small, DIM, W - 12, H - 14, anchor="bottomright")

    def _render_ml_results(self, cx: int, oy: int) -> None:
        self._btn_results_open = Button(
            pygame.Rect(W - 140, oy - 28, 128, 26), "Open Folder",
            self.f_btn_sm, color=PANEL2, text_color=DIM, hover_text=ACCENT)
        self._btn_results_open.draw(self.screen)

        if not self.result_runs:
            draw_text(self.screen, "No ML results yet. Run Full Demo or `python run_experiment.py --quick`.",
                      self.f_body, DIM, cx + 20, oy + 30)
            return

        px = cx + 12
        for i, run_path in enumerate(self.result_runs):
            label = run_path.name[:28]
            lw    = self.f_small.size(label)[0] + 20
            active = (i == self.result_run_idx)
            draw_rect(self.screen, pygame.Rect(px, oy, lw, 24), PANEL2 if active else PANEL, 3)
            if active:
                pygame.draw.rect(self.screen, ACCENT, pygame.Rect(px, oy, lw, 2))
            draw_border(self.screen, pygame.Rect(px, oy, lw, 24), BORDER, 3)
            draw_text(self.screen, label, self.f_small, TEXT if active else DIM, px + 10, oy + 5)
            px += lw + 5
            if px > W - 150:
                break

        oy += 30
        hline(self.screen, oy, cx, W)
        oy += 8

        if not self.result_thumbs:
            draw_text(self.screen, "Experiment metrics", self.f_body, DIM, cx + 20, oy + 20)
            draw_text(self.screen, "Saved evaluation results. Train models to generate inspectable plots.",
                      self.f_small, DIM, cx + 20, oy + 44)
            self._render_metrics_table(cx + 24, oy + 94)
            return

        tw, th = 290, 196
        gap    = 14
        cols   = 3
        row_h  = th + 28 + gap
        ox     = cx + ((W - cx) - (cols * tw + (cols - 1) * gap)) // 2
        vis_h  = H - oy - 56

        max_scroll = max(0, math.ceil(len(self.result_thumbs) / cols) * row_h + (160 if self.result_metrics else 0) - vis_h)
        self.result_scroll = max(0, min(self.result_scroll, max_scroll))

        self.screen.set_clip(pygame.Rect(cx, oy, W - cx, vis_h))
        for i, (surf, title, _) in enumerate(self.result_thumbs):
            col = i % cols
            row = i // cols
            tx  = ox + col * (tw + gap)
            ty  = oy + row * row_h - self.result_scroll
            if ty + th < oy or ty > H - 56:
                continue
            draw_rect(self.screen, pygame.Rect(tx - 3, ty - 3, tw + 6, th + 6), PANEL, 5)
            self.screen.blit(surf, (tx, ty))
            draw_text(self.screen, title, self.f_small, DIM,
                      tx + tw // 2, ty + th + 5, anchor="midtop")

        if self.result_metrics:
            rows_used = math.ceil(len(self.result_thumbs) / cols)
            table_y   = oy + rows_used * row_h - self.result_scroll
            if table_y < H - 56:
                self._render_metrics_table(ox, table_y)

        self.screen.set_clip(None)
        draw_text(self.screen, "Scroll to explore    Click a plot to inspect",
                  self.f_small, DIM, cx + 12, H - 14, anchor="bottomleft")

    def _render_metrics_table(self, x: int, y: int) -> None:
        if not self.result_metrics:
            return
        draw_text(self.screen, "Model Comparison", self.f_sub, TEXT, x, y)
        y += 28
        headers   = ["Model", "Accuracy", "F1", "AUC", "ECE"]
        col_names = ["accuracy", "f1", "roc_auc", "ece"]
        col_w     = [174, 112, 92, 92, 92]
        hx = x
        for h, cw in zip(headers, col_w):
            draw_text(self.screen, h, self.f_small, DIM, hx, y)
            hx += cw
        y += 20
        hline(self.screen, y, x, x + sum(col_w))
        y += 5
        for mname, metrics in self.result_metrics.items():
            hx  = x
            col = LSTM_C if mname == "lstm" else TF_C
            draw_text(self.screen, mname.upper(), self.f_body, col, hx, y)
            hx += col_w[0]
            for metric in col_names:
                val = metrics.get(metric, float("nan"))
                if not math.isnan(val):
                    if metric in ("accuracy", "f1", "roc_auc"):
                        vc = GREEN if val > 0.70 else YELLOW if val > 0.55 else RED
                    else:
                        vc = GREEN if val < 0.10 else YELLOW if val < 0.15 else RED
                else:
                    vc = DIM
                draw_text(self.screen, f"{val:.4f}" if not math.isnan(val) else "n/a",
                          self.f_body, vc, hx, y)
                hx += col_w[col_names.index(metric) + 1]
            y += 24

    def _render_human_results(self, cx: int, oy: int) -> None:
        self._btn_human_refresh = Button(
            pygame.Rect(W - 110, oy - 28, 98, 26), "Refresh",
            self.f_btn_sm, color=PANEL2, text_color=DIM, hover_text=ACCENT)
        self._btn_human_refresh.draw(self.screen)

        self._btn_human_open = Button(
            pygame.Rect(W - 220, oy - 28, 98, 26), "Open Folder",
            self.f_btn_sm, color=PANEL2, text_color=DIM, hover_text=ACCENT)
        self._btn_human_open.draw(self.screen)

        self._btn_export_csv = Button(
            pygame.Rect(W - 340, oy - 28, 108, 26), "Export CSV",
            self.f_btn_sm, color=PANEL2, text_color=DIM, hover_text=GREEN)
        self._btn_export_csv.disabled = not self.human_sessions
        self._btn_export_csv.draw(self.screen)

        if not self.human_sessions:
            draw_text(self.screen, "No human sessions found.", self.f_body, TEXT, cx + 20, oy + 20)
            draw_text(self.screen, "Use Play Task to record a session, then refresh this view.",
                      self.f_small, DIM, cx + 20, oy + 44)
            return

        cols_x = [cx + 20, cx + 60, cx + 330, cx + 540, cx + 640, cx + 780]
        hdrs   = ["#", "Name", "Date", "Trials", "Accuracy", "Avg RT"]
        for hx, hdr in zip(cols_x, hdrs):
            draw_text(self.screen, hdr, self.f_small, DIM, hx, oy)
        oy += 20
        hline(self.screen, oy, cx, W - 12)
        oy += 5

        vis_h  = H - oy - 160
        row_h  = 28
        max_sc = max(0, len(self.human_sessions) * row_h - vis_h)
        self.human_scroll = max(0, min(self.human_scroll, max_sc))

        self.screen.set_clip(pygame.Rect(cx, oy, W - cx, vis_h))
        for i, sess in enumerate(self.human_sessions):
            y      = oy + i * row_h - self.human_scroll
            if y < oy - row_h or y > H - 160:
                continue
            is_sel = (self.human_selected == i)
            if is_sel:
                draw_rect(self.screen, pygame.Rect(cx + 8, y, W - cx - 20, row_h - 2), PANEL2, 3)

            acc_col  = GREEN if sess["accuracy"] > 0.75 else YELLOW if sess["accuracy"] > 0.55 else RED
            name_lbl = sess["name"] if sess["name"] else sess["id"]
            draw_text(self.screen, str(i + 1),                self.f_small, DIM,     cols_x[0], y + 7)
            draw_text(self.screen, name_lbl[:22],             self.f_body,  TEXT,    cols_x[1], y + 6)
            draw_text(self.screen, sess["start"][:16],        self.f_small, DIM,     cols_x[2], y + 7)
            draw_text(self.screen, str(sess["trials"]),       self.f_small, DIM,     cols_x[3], y + 7)
            draw_text(self.screen, f"{sess['accuracy']:.1%}", self.f_body,  acc_col, cols_x[4], y + 6)
            draw_text(self.screen, f"{sess['avg_rt']:.3f}s",  self.f_small, DIM,     cols_x[5], y + 7)

        self.screen.set_clip(None)
        detail_y = H - 165
        hline(self.screen, detail_y, cx, W)
        if self.human_selected is not None and self.human_selected < len(self.human_sessions):
            s = self.human_sessions[self.human_selected]
            draw_text(self.screen, s["name"] or s["id"], self.f_sub, TEXT, cx + 14, detail_y + 8)
            draw_text(self.screen,
                      f"Date: {s['start']}   Trials: {s['trials']}   "
                      f"Accuracy: {s['accuracy']:.1%}   Avg RT: {s['avg_rt']:.3f}s",
                      self.f_body, DIM, cx + 14, detail_y + 30)

            # Show lead time metrics if available for this session
            sid = s.get("id", "")
            matched_m = next(
                (m for m in self.session_metrics if m.get("session_id") == sid), None
            )
            if matched_m:
                n_pred  = matched_m.get("n_predicted_before", 0)
                n_miss  = matched_m.get("n_missed", 0)
                avg_lt  = matched_m.get("avg_lead_time_s", 0.0)
                mode    = matched_m.get("model_mode", "?")
                calibd  = matched_m.get("calibrated", False)
                draw_text(self.screen,
                          f"Predicted before error: {n_pred}   Missed: {n_miss}   "
                          f"Avg lead time: {avg_lt:.2f}s   Model: {mode}   "
                          f"Calibrated: {'yes' if calibd else 'no'}",
                          self.f_small, ACCENT, cx + 14, detail_y + 52)
            draw_text(self.screen, str(s["path"]), self.f_small, DIM, cx + 14, detail_y + 70)
        else:
            draw_text(self.screen, "Click a row to view session details.",
                      self.f_body, DIM, cx + 14, detail_y + 26)

    def _render_full_image(self) -> None:
        if not self.result_full_view:
            return
        img   = self.result_full_view
        ratio = min((W - 40) / img.get_width(), (H - 70) / img.get_height())
        nw    = int(img.get_width() * ratio)
        nh    = int(img.get_height() * ratio)
        scaled = pygame.transform.smoothscale(img, (nw, nh))
        self.screen.blit(scaled, ((W - nw) // 2, 50 + (H - 70 - nh) // 2))
        draw_text(self.screen, self.result_full_title, self.f_sub, TEXT, W // 2, 12, anchor="midtop")
        draw_text(self.screen, "ESC or click to close", self.f_small, DIM, W // 2, H - 14, anchor="midbottom")

    def _handle_results_event(self, event) -> None:
        if hasattr(self, "_hdr_home_btn") and self._hdr_home_btn.handle_event(event):
            self.state = State.MENU
            return
        if self.result_full_view:
            if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE) or \
               event.type == pygame.MOUSEBUTTONDOWN:
                self.result_full_view = None
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            cx   = self._cx
            tab_y = 54
            for i, (_, key) in enumerate([("ml", "ml"), ("human", "human")]):
                tw = 148
                tx = cx + 12 + i * (tw + 8)
                if pygame.Rect(tx, tab_y, tw, 30).collidepoint(event.pos):
                    self.results_tab = key
                    return

        if event.type == pygame.MOUSEWHEEL:
            if self.results_tab == "ml":
                self.result_scroll -= event.y * 44
            else:
                self.human_scroll -= event.y * 32

        if self.results_tab == "ml":
            if hasattr(self, "_btn_results_open") and self._btn_results_open.handle_event(event):
                self._open_folder(
                    self.result_runs[self.result_run_idx] if self.result_runs
                    else Path("driftsync/results")
                )
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                cx, oy = self._cx, 90
                px = cx + 12
                for i, run_path in enumerate(self.result_runs):
                    label = run_path.name[:28]
                    lw    = self.f_small.size(label)[0] + 20
                    if pygame.Rect(px, oy, lw, 24).collidepoint(event.pos):
                        self._load_ml_results(i)
                        return
                    px += lw + 5

                oy += 38
                tw, th = 290, 196
                gap    = 14
                cols   = 3
                row_h  = th + 28 + gap
                ox     = cx + ((W - cx) - (cols * tw + (cols - 1) * gap)) // 2
                for i, (_, title, path) in enumerate(self.result_thumbs):
                    col = i % cols
                    row = i // cols
                    tx  = ox + col * (tw + gap)
                    ty  = oy + row * row_h - self.result_scroll
                    if pygame.Rect(cx, oy, W - cx, H - oy - 56).collidepoint(event.pos) and pygame.Rect(tx, ty, tw, th).collidepoint(event.pos):
                        full = load_png(path)
                        if full:
                            self.result_full_view  = full
                            self.result_full_title = title
                        return

        if self.results_tab == "human":
            if hasattr(self, "_btn_human_refresh") and self._btn_human_refresh.handle_event(event):
                self._load_human_sessions()
                self._load_session_metrics()
            if hasattr(self, "_btn_human_open") and self._btn_human_open.handle_event(event):
                self._open_folder(Path("driftsync/data/raw"))
            if hasattr(self, "_btn_export_csv") and self._btn_export_csv.handle_event(event):
                self._export_sessions_csv()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                cx, oy = self._cx, 90 + 25
                row_h  = 28
                for i in range(len(self.human_sessions)):
                    y = oy + i * row_h - self.human_scroll
                    if oy <= event.pos[1] < H - 160 and pygame.Rect(cx + 8, y, W - cx - 20, row_h - 2).collidepoint(event.pos):
                        self.human_selected = i
                        return

    def _open_folder(self, path: Path) -> None:
        try:
            import subprocess
            target = str(path.resolve())
            if sys.platform == "win32":
                os.startfile(target)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", target])
            else:
                subprocess.Popen(["xdg-open", target])
        except Exception:
            pass

    # PLAY TASK

    def _build_play_task_buttons(self) -> None:
        cx   = self._cx + 24
        WY   = 236   # widget row y  (name input + trial buttons)
        BTY  = 286   # start button y
        AFTY = 348   # after-task row y
        self._play_name_input = TextInput(
            pygame.Rect(cx, WY, 300, 36), self.f_body, placeholder="Session name (optional)")
        self._btn_play_trials_minus = Button(
            pygame.Rect(cx + 326, WY, 34, 36), "-", self.f_btn,
            color=PANEL2, text_color=TEXT, hover_text=ACCENT)
        self._btn_play_trials_plus = Button(
            pygame.Rect(cx + 326 + 78, WY, 34, 36), "+", self.f_btn,
            color=PANEL2, text_color=TEXT, hover_text=ACCENT)
        self._btn_play_start = Button(
            pygame.Rect(cx, BTY, 210, 46), "Start task",
            self.f_btn, accent_fill=True)
        self._btn_play_skip_calib = Button(
            pygame.Rect(cx + 226, BTY, 180, 46), "Skip Calibration",
            self.f_btn, color=PANEL2, text_color=DIM, hover_text=ACCENT)
        self._btn_play_human_results = Button(
            pygame.Rect(cx, 450, 260, 42), "View recorded sessions",
            self.f_btn, color=PANEL2, text_color=TEXT, hover_text=ACCENT)

    def _render_play_task(self) -> None:
        cx  = self._cx
        x   = cx + 24
        WY   = 236   # widget row y
        AFTY = 408
        BTY = 286

        self._render_screen_header("Record a task")

        instrs = [
            "A shape appears: Circle, Square, or Triangle.",
            "The rule at top tells you which shape to CLICK.",
            "Press SPACE to skip if the shape does NOT match.",
            "Reaction time and accuracy are recorded per trial.",
            "The time window shrinks as the session progresses.",
        ]
        panel_y = 62
        panel   = pygame.Rect(cx + 12, panel_y, W - cx - 24, len(instrs) * 22 + 18)
        draw_rect(self.screen, panel, PANEL, 5)
        draw_border(self.screen, panel, BORDER, 5)
        iy = panel_y + 9
        for line in instrs:
            draw_text(self.screen, line, self.f_body, TEXT, x + 6, iy)
            iy += 22

        draw_text(self.screen, "Session name:", self.f_body, TEXT, x, WY - 26)
        draw_text(self.screen, "Trials:", self.f_body, TEXT, x + 328, WY - 26)
        if self._play_name_input:
            self._play_name_input.draw(self.screen, self.clock.get_time() / 1000.0)
        self._btn_play_trials_minus.draw(self.screen)
        draw_text(self.screen, str(self.play_num_trials), self.f_sub, TEXT,
                  x + 382, WY + 4, anchor="midtop")
        self._btn_play_trials_plus.draw(self.screen)

        self._btn_play_start.draw(self.screen)
        self._btn_play_skip_calib.text = "Calibration: off" if self._play_skip_calib else "Calibration: on"
        self._btn_play_skip_calib.text_color = YELLOW if self._play_skip_calib else TEXT
        self._btn_play_skip_calib.draw(self.screen)

        # Calibration status
        try:
            from driftsync.ml.calibrator import CalibrationEngine
            is_calib = CalibrationEngine.is_calibrated()
            calib_col = GREEN if is_calib else YELLOW
            calib_lbl = "Calibrated" if is_calib else "Not calibrated (will run on first task)"
        except Exception:
            is_calib  = False
            calib_col = DIM
            calib_lbl = "Calibration unavailable"
        draw_text(self.screen, f"Calibration: {calib_lbl}", self.f_small, calib_col, x, BTY + 56)

        if self._play_task_done:
            draw_text(self.screen, "Session saved in driftsync/data/raw/",
                      self.f_body, GREEN, x, AFTY)
            self._btn_play_human_results.draw(self.screen)

        if self.notice:
            ui.paragraph(self.screen, self.notice, pygame.Rect(x, 510, 780, 100), self.f_body, YELLOW)

        draw_text(self.screen, "ESC = menu   |   SPACE = skip   |   Click = respond",
                  self.f_small, DIM, cx + 14, H - 16, anchor="bottomleft")

    def _handle_play_event(self, event) -> None:
        if hasattr(self, "_hdr_home_btn") and self._hdr_home_btn.handle_event(event):
            self.state = State.MENU
            return
        if self._play_name_input:
            self._play_name_input.handle_event(event)
        if hasattr(self, "_btn_play_trials_minus") and self._btn_play_trials_minus.handle_event(event):
            self.play_num_trials = max(50, self.play_num_trials - 25)
        if hasattr(self, "_btn_play_trials_plus") and self._btn_play_trials_plus.handle_event(event):
            self.play_num_trials = min(500, self.play_num_trials + 25)
        if hasattr(self, "_btn_play_skip_calib") and self._btn_play_skip_calib.handle_event(event):
            self._play_skip_calib = not self._play_skip_calib
        if hasattr(self, "_btn_play_start") and self._btn_play_start.handle_event(event):
            name = self._play_name_input.text.strip() if self._play_name_input else ""
            self._play_session_name = name
            self._launch_play_task()
        if hasattr(self, "_btn_play_human_results") and self._play_task_done and \
                self._btn_play_human_results.handle_event(event):
            self._load_human_sessions()
            self.results_tab = "human"
            self._enter_state(State.RESULTS)

    # LIVE MODE

    def _render_live_mode(self) -> None:
        self._render_screen_header("Live session")
        x = self._cx + 40
        draw_text(self.screen, "Observe your focus in real time", self.f_title, TEXT, x, 106)
        ui.paragraph(self.screen, "Complete the shape task while a sequence model estimates the chance of an error in the next five trials.",
                     pygame.Rect(x, 168, 730, 72), self.f_body)
        draw_text(self.screen, "Prediction model", self.f_sub, TEXT, x, 264)
        self._live_model_rects = {}
        for i, (label, key) in enumerate([("LSTM", "lstm"), ("Transformer", "transformer")]):
            rect = pygame.Rect(x + i * 202, 304, 186, 48)
            self._live_model_rects[key] = rect
            active = self.live_model_choice == key
            draw_rect(self.screen, rect, PANEL2 if active else PANEL)
            draw_border(self.screen, rect, ACCENT if active else BORDER)
            draw_text(self.screen, label + ("  / selected" if active else ""), self.f_body,
                      TEXT if active else DIM, *rect.center, "center")
        from driftsync.realtime.checkpoints import resolve_workspace_checkpoint
        from driftsync.configs import RealtimeConfig
        try:
            checkpoint = resolve_workspace_checkpoint(self.live_model_choice)
            status = f"Checkpoint: {checkpoint.parent.parent.name[:35]}. Predictions begin after 20 trials."
        except FileNotFoundError:
            status = "No trained checkpoint. The session will record behaviour only."
        draw_text(self.screen, status, self.f_body, DIM, x, 378)
        self._live_btn = getattr(self, "_live_btn", Button(pygame.Rect(x, 424, 220, 48), "Start live session", self.f_btn, accent_fill=True))
        self._live_btn.hovered = self._live_btn.rect.collidepoint(self._mouse())
        self._live_btn.draw(self.screen)
        draw_text(self.screen, "L / T  Choose model     Tab / Enter  Start", self.f_small, DIM, x, 490)
        hline(self.screen, 554, x, W - 40)
        columns = [("Error risk", "Probability of an error within the prediction horizon."),
                   ("Model uncertainty", "Spread across dropout passes. Reported as standard deviation."),
                   ("Prediction confidence", "A calibrated confidence score is not available from this model.")]
        for i, (label, description) in enumerate(columns):
            xx = x + i * 330
            draw_text(self.screen, label, self.f_sub, TEXT, xx, 588)
            ui.paragraph(self.screen, description, pygame.Rect(xx, 630, 290, 90), self.f_body)

    def _handle_live_event(self, event) -> None:
        if hasattr(self, "_hdr_home_btn") and self._hdr_home_btn.handle_event(event):
            self.state = State.MENU
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_l:
                self.live_model_choice = "lstm"
            elif event.key == pygame.K_t:
                self.live_model_choice = "transformer"
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for key, rect in getattr(self, "_live_model_rects", {}).items():
                if rect.collidepoint(event.pos):
                    self.live_model_choice = key
        if hasattr(self, "_live_btn") and self._live_btn.handle_event(event):
            self._launch_live_mode()

    # Launch helpers

    def _reinit_pygame(self) -> None:
        pygame.init()
        self.viewport = ui.Viewport((W, H), "DriftSync | Focus analysis")
        self.screen = self.viewport.surface
        pygame.display.set_caption("DriftSync  [F11 = fullscreen]")
        self.clock  = pygame.time.Clock()
        self._init_fonts()
        self._build_menu_buttons()
        self._build_learn_buttons()
        self._build_play_task_buttons()
        self._build_demo_buttons()
        if hasattr(self, "_live_btn"):
            del self._live_btn

    def _launch_play_task(self) -> None:
        name       = getattr(self, "_play_session_name", "")
        trials     = getattr(self, "play_num_trials", 150)
        skip_calib = getattr(self, "_play_skip_calib", False)
        self.notice = ""
        self._play_task_done = False
        pygame.quit()
        try:
            from driftsync.configs import SimulatorConfig
            from driftsync.simulator.gui import DriftSimulator
            saved_path = DriftSimulator(
                SimulatorConfig(num_trials=trials, session_name=name),
                skip_calibration=skip_calib,
            ).run()
            self._play_task_done = bool(saved_path)
            if not saved_path:
                self.notice = "Session setup canceled. No task data saved."
        except Exception as e:
            self.notice = f"Task could not finish: {e}. Try starting a new session."
        finally:
            self._reinit_pygame()
            self.state = State.PLAY_TASK

    def _launch_live_mode(self) -> None:
        self.notice = ""
        pygame.quit()
        try:
            from driftsync.configs import SimulatorConfig, RealtimeConfig
            from driftsync.realtime.checkpoints import resolve_workspace_checkpoint
            from driftsync.realtime.live_simulator import LiveDriftSimulator
            rt_config = RealtimeConfig(model_type=self.live_model_choice)
            try:
                rt_config.checkpoint_dir = str(resolve_workspace_checkpoint(self.live_model_choice).parent)
            except FileNotFoundError:
                pass
            saved_path = LiveDriftSimulator(
                SimulatorConfig(num_trials=150),
                rt_config,
                model_type=self.live_model_choice,
            ).run()
            self.notice = "Live session saved. Open Results to review it." if saved_path else "Session canceled. Existing recordings are unchanged."
        except Exception as e:
            self.notice = f"Live session could not finish: {e}. Check the model and retry."
        finally:
            self._reinit_pygame()
            self.state = State.MENU
