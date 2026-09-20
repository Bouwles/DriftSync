"""Typeset, interactive explanations of DriftSync's implemented mathematics.

Examples are deterministic illustrations, never presented as model predictions.
MathText supplies proper fractions, indices and Greek symbols without a TeX install.
"""
from functools import lru_cache
from io import BytesIO
import math

import numpy as np
import pygame

from driftsync import ui
from driftsync.configs import CONFIG


def sample_summary(values):
    """The same sample standard deviation convention used by torch.std (N - 1)."""
    return float(np.mean(values)), float(np.std(values, ddof=1))


@lru_cache(maxsize=48)
def _equation_surface(expression, size):
    from matplotlib import rc_context
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image
    buffer = BytesIO()
    with rc_context({"savefig.transparent": True}):
        math_to_image(f"${expression}$", buffer, prop=FontProperties(size=size, math_fontfamily="stix"),
                      dpi=150, format="png", color="#ebefed")
    buffer.seek(0)
    return pygame.image.load(buffer).convert_alpha()


def equation(screen, expression, rect, size=23):
    image = _equation_surface(expression, size)
    scale = min(rect.width / image.get_width(), rect.height / image.get_height(), 1.0)
    if scale < 1:
        image = pygame.transform.smoothscale(image, (max(1, round(image.get_width() * scale)),
                                                    max(1, round(image.get_height() * scale))))
    screen.blit(image, image.get_rect(midleft=(rect.left, rect.centery)))


class MathematicsView:
    TABS = ("Prediction", "Behaviour", "Sequence models", "Uncertainty", "Learning")
    TITLES = ("Predict the next mistake, not the current one.",
              "Turn behaviour into comparable signals.",
              "Two ways to read a sequence.",
              "A probability is only part of the story.",
              "Learn from errors. Check the probabilities.")
    SOURCES = ("data/preprocessing.py / add_target_label",
               "data/preprocessing.py / engineer_features; realtime/inference_engine.py",
               "models/lstm_model.py; models/transformer_model.py",
               "models/base.py / mc_dropout_predict; realtime/inference_engine.py",
               "training/trainer.py / Trainer; utils/metrics.py / compute_ece")

    def __init__(self):
        self.page = 0
        self.outcomes = [0, 0, 1, 0, 0]
        self.outcome_rects = []
        self.model = "LSTM"
        self.spread = .10
        self.probability = .6
        self.label = 1
        self.controls = []
        self.focus = -1

    @property
    def example_label(self):
        return int(any(self.outcomes))

    def _control(self, screen, rect, label, action, selected=False):
        index = len(self.controls)
        focused = index == self.focus
        pygame.draw.rect(screen, ui.PANEL2 if selected else ui.PANEL, rect, border_radius=4)
        pygame.draw.rect(screen, ui.ACCENT if selected or focused else ui.BORDER, rect, 1, border_radius=4)
        if focused:
            pygame.draw.rect(screen, ui.ACCENT, rect.inflate(6, 6), 2, border_radius=5)
        ui.text(screen, label, ui.font(14, selected), ui.TEXT if selected else ui.DIM, *rect.center, "center")
        self.controls.append((rect, action))

    def _copy(self, screen, value, x, y, width=608, height=90, color=None):
        return ui.paragraph(screen, value, pygame.Rect(x, y, width, height), ui.font(16), color or ui.DIM, 5)

    def render(self, screen, x=224):
        self.controls = []
        ui.text(screen, "Mathematics", ui.font(25, True), ui.TEXT, x, 28)
        ui.text(screen, "From observation to prediction", ui.font(15), ui.DIM, 1240, 37, "topright")
        for i, name in enumerate(self.TABS):
            self._control(screen, pygame.Rect(x + i * 203, 84, 187, 38), name, ("page", i), self.page == i)
        ui.text(screen, self.TITLES[self.page], ui.font(27, True), ui.TEXT, x, 160)
        pygame.draw.line(screen, ui.BORDER, (x + 642, 226), (x + 642, 707))
        ui.text(screen, "Worked example", ui.font(19, True), ui.TEXT, x + 678, 230)
        ui.text(screen, "Illustrative values / not a live session", ui.font(12), ui.DIM, x + 678, 261)
        (self._prediction, self._behaviour, self._networks, self._uncertainty, self._learning)[self.page](screen, x)
        pygame.draw.line(screen, ui.BORDER, (x, 731), (1240, 731))
        ui.text(screen, "Implementation: " + self.SOURCES[self.page], ui.font(12), ui.DIM, x, 747)
        ui.text(screen, "Left / Right  Topics     Tab / Enter  Interact     Esc  Workspace", ui.font(13), ui.DIM, x, 773)

    def _prediction(self, s, x):
        length, horizon = CONFIG.data.sequence_length, CONFIG.data.prediction_horizon
        self._copy(s, "The model reads a window of past trials. Its target asks whether at least one error occurs in the following prediction horizon.", x, 229)
        equation(s, r"X_t=[x_{t-L+1},\ldots,x_t]\in\mathbb{R}^{L\times 15}", pygame.Rect(x, 320, 610, 63))
        equation(s, r"y_t=\mathbb{1}\!\left[\sum_{j=1}^{K}(1-c_{t+j})>0\right]", pygame.Rect(x, 405, 610, 70))
        ui.text(s, f"L = {length} past trials     K = {horizon} future trials     c = correctness", ui.font(15), ui.DIM, x, 499)
        ui.text(s, "Input window", ui.font(17, True), ui.TEXT, x, 552)
        # Schematic tensor dimensions, deliberately not a heatmap of invented inputs.
        for col in range(20):
            for row in range(15):
                pygame.draw.rect(s, ui.BORDER, (x + col * 14, 591 + row * 5, 10, 3))
        self._copy(s, "20 trials x 15 features\nOne probability for the next 5 trials", x + 310, 596, 300, 70)
        rx = x + 678
        self._copy(s, "Click a future outcome to change it.", rx, 304, 332, 65)
        self.outcome_rects = []
        for i, error in enumerate(self.outcomes):
            rect = pygame.Rect(rx + i * 66, 365, 57, 58)
            self.outcome_rects.append(rect)
            self._control(s, rect, "Error" if error else "OK", ("outcome", i), bool(error))
            ui.text(s, f"t+{i + 1}", ui.font(13), ui.DIM, rect.centerx, 437, "midtop")
        equation(s, rf"y_t={self.example_label}", pygame.Rect(rx, 490, 325, 58), 30)
        self._copy(s, "At least one future error makes the target 1." if self.example_label else "All five future trials are correct. The target is 0.", rx, 574, 325, 85)
        self._copy(s, "Future outcomes supply training labels; they are never model inputs.", rx, 654, 325, 66, ui.ACCENT)

    def _behaviour(self, s, x):
        self._copy(s, "Reaction times are scaled against a median and interquartile range. Rolling error rates describe how often recent responses were wrong.", x, 229)
        equation(s, r"z_t=\frac{r_t-\operatorname{median}(r)}{Q_{75}(r)-Q_{25}(r)+10^{-6}}", pygame.Rect(x, 319, 610, 78))
        equation(s, r"\widetilde r_t=\frac{\operatorname{clip}(z_t,-3,3)}{6}+\frac{1}{2}", pygame.Rect(x, 419, 610, 70))
        equation(s, r"e_t^{(w)}=\frac{1}{n}\sum_{i=t-n+1}^{t}(1-c_i),\quad n=\min(w,t+1)", pygame.Rect(x, 511, 610, 72), 21)
        self._copy(s, "15 features cover response timing, accuracy, streaks, task context and fatigue. Training uses session statistics; live inference uses available history. Several scaling rules differ between those paths.", x, 617, 610, 100)
        rx = x + 678
        self._copy(s, "Reaction times (seconds)", rx, 304, 330, 35)
        ui.text(s, "0.4    0.5    0.6    0.7    0.8", ui.font(17, mono=True), ui.TEXT, rx, 351)
        for i, (label, value) in enumerate((("Median", "0.60 s"), ("Interquartile range", "0.20 s"), ("Latest response", "0.80 s"), ("Normalized response", "0.667"))):
            yy = 405 + i * 48
            ui.text(s, label, ui.font(15), ui.DIM, rx, yy)
            ui.text(s, value, ui.font(18, True), ui.TEXT, rx + 325, yy - 2, "topright")
            pygame.draw.line(s, ui.BORDER, (rx, yy + 32), (rx + 325, yy + 32))
        self._copy(s, "A slower-than-median response sits above 0.5. Clipping limits the influence of extreme values.", rx, 625, 325, 85)

    def _networks(self, s, x):
        lstm = self.model == "LSTM"
        self._copy(s, "LSTM updates a memory state one trial at a time. The Transformer compares past trials through masked self-attention. Both produce a single logit, then a probability.", x, 229)
        if lstm:
            equation(s, r"c_t=f_t\odot c_{t-1}+i_t\odot g_t", pygame.Rect(x, 337, 610, 57), 26)
            equation(s, r"h_t=o_t\odot\tanh(c_t)", pygame.Rect(x, 415, 610, 57), 26)
            self._copy(s, "f, i and o are learned sigmoid gates: forget, write and expose. g is the candidate memory; the circled dot multiplies elements. Each gate depends on the current input and previous hidden state.", x, 507, 610, 116)
        else:
            equation(s, r"A=\operatorname{softmax}\!\left(\frac{QK^{\mathsf{T}}}{\sqrt{d_k}}+M\right)", pygame.Rect(x, 328, 610, 90), 24)
            equation(s, r"\operatorname{Attention}(Q,K,V)=AV", pygame.Rect(x, 445, 610, 54), 25)
            self._copy(s, "Q, K and V are learned query, key and value projections. The causal mask M is zero for available positions and negative infinity for future positions. Sinusoidal position encodings preserve order.", x, 541, 610, 130)
        equation(s, r"p=\sigma(a)=\frac{1}{1+e^{-a}}", pygame.Rect(x, 652, 600, 59), 22)
        rx = x + 678
        self._control(s, pygame.Rect(rx, 304, 150, 38), "LSTM", ("model", "LSTM"), lstm)
        self._control(s, pygame.Rect(rx + 166, 304, 158, 38), "Transformer", ("model", "Transformer"), not lstm)
        if lstm:
            nodes = [("20 x 15", "Input window"), (f"{CONFIG.lstm.hidden_dim} dimensions", "Projection + normalization + GELU"),
                     (f"{CONFIG.lstm.num_layers} recurrent layers", "LayerNorm, dropout, residuals"), ("Last hidden state", "Classification head + sigmoid")]
        else:
            nodes = [("20 x 15", "Input window"), (f"{CONFIG.transformer.d_model} dimensions", "Projection + positional encoding"),
                     (f"{CONFIG.transformer.num_encoder_layers} layers / {CONFIG.transformer.nhead} heads", "Pre-LN attention + feed-forward"), ("Final norm + mean pooling", "Classification head + sigmoid")]
        for i, (value, label) in enumerate(nodes):
            yy = 371 + i * 79
            pygame.draw.rect(s, ui.PANEL2, (rx, yy, 325, 58), border_radius=4)
            ui.text(s, value, ui.font(17, True), ui.TEXT, rx + 14, yy + 7)
            ui.text(s, label, ui.font(12), ui.DIM, rx + 14, yy + 34)
            if i < len(nodes) - 1:
                pygame.draw.line(s, ui.ACCENT, (rx + 162, yy + 59), (rx + 162, yy + 76))
        ui.text(s, "Architecture schematic / default configuration", ui.font(12), ui.DIM, rx, 698)

    def _uncertainty(self, s, x):
        from driftsync.realtime.inference_engine import MC_DROPOUT_SAMPLES
        count = MC_DROPOUT_SAMPLES
        self._copy(s, f"Dropout remains active for {count} stochastic passes. Their mean estimates error risk; their sample standard deviation describes variation between model outputs.", x, 229)
        equation(s, r"\bar p=\frac{1}{N}\sum_{m=1}^{N}p^{(m)}", pygame.Rect(x, 335, 610, 68), 25)
        equation(s, r"s=\sqrt{\frac{1}{N-1}\sum_{m=1}^{N}(p^{(m)}-\bar p)^2}", pygame.Rect(x, 430, 610, 86), 25)
        equation(s, rf"\mathrm{{warn}}=(\bar p>{CONFIG.realtime.warning_threshold})\ \vee\ (s>{CONFIG.realtime.uncertainty_threshold})", pygame.Rect(x, 564, 610, 51), 22)
        self._copy(s, "Uncertainty is not 1 minus confidence. The shaded live band is +/-1 standard deviation, not a calibrated confidence interval.", x, 653, 610, 66)
        rx = x + 678
        for i, (label, sd) in enumerate((("Tight", .02), ("Moderate", .10), ("Wide", .24))):
            self._control(s, pygame.Rect(rx + i * 111, 304, 103, 38), label, ("spread", sd), self.spread == sd)
        basis = np.linspace(-1, 1, count)
        values = .55 + basis / np.std(basis, ddof=1) * self.spread
        mean, sd = sample_summary(values)
        axis = pygame.Rect(rx + 8, 393, 309, 94)
        pygame.draw.line(s, ui.BORDER, (axis.left, axis.bottom), (axis.right, axis.bottom))
        for i, value in enumerate(values):
            pygame.draw.circle(s, ui.ACCENT, (axis.left + round(value * axis.width), axis.bottom - 18 - (i % 5) * 13), 3)
        ui.text(s, "0%", ui.font(12), ui.DIM, axis.left, axis.bottom + 8)
        ui.text(s, "100%", ui.font(12), ui.DIM, axis.right, axis.bottom + 8, "topright")
        ui.text(s, "Same mean. Different spread.", ui.font(16), ui.TEXT, rx, 539)
        ui.text(s, f"Mean risk        {mean:.0%}", ui.font(18, True), ui.TEXT, rx, 581)
        ui.text(s, f"Uncertainty     {sd * 100:.1f} pp", ui.font(18, True), ui.TEXT, rx, 618)
        warning = mean > CONFIG.realtime.warning_threshold or sd > CONFIG.realtime.uncertainty_threshold
        self._copy(s, "Uncertainty threshold crossed." if warning else "Neither warning threshold is crossed.", rx, 667, 325, 48, ui.YELLOW if warning else ui.DIM)

    def _learning(self, s, x):
        self._copy(s, "Training minimizes weighted binary cross-entropy on sequence labels. Positive examples are weighted by the ratio of negative to positive labels in the training split.", x, 229)
        equation(s, r"\mathcal{L}=-\frac{1}{B}\sum_{i=1}^{B}\left[w y_i\log p_i+(1-y_i)\log(1-p_i)\right]", pygame.Rect(x, 325, 610, 89), 23)
        equation(s, r"w=\frac{n_{\mathrm{negative}}}{\max(n_{\mathrm{positive}},1)}", pygame.Rect(x, 431, 610, 59), 23)
        ui.text(s, "Check calibration on held-out data", ui.font(19, True), ui.TEXT, x, 533)
        equation(s, r"\mathrm{ECE}=\sum_b\frac{|B_b|}{N}\left|\overline{y}_b-\overline{p}_b\right|", pygame.Rect(x, 571, 610, 63), 23)
        self._copy(s, "ECE compares the observed error frequency with mean predicted risk in probability bins. Training uses stable logits, AdamW, gradient clipping and early stopping.", x, 658, 610, 64)
        rx = x + 678
        self._control(s, pygame.Rect(rx, 304, 150, 38), "Error / y = 1", ("label", 1), self.label == 1)
        self._control(s, pygame.Rect(rx + 166, 304, 158, 38), "No error / y = 0", ("label", 0), self.label == 0)
        self._copy(s, "Example class weight w = 2", rx, 362, 325, 35)
        chart = pygame.Rect(rx + 20, 413, 289, 112)
        points = []
        for p in np.linspace(.01, .99, 99):
            loss = -(2 * self.label * math.log(p) + (1 - self.label) * math.log(1 - p))
            points.append((chart.left + round(p * chart.width), chart.bottom - round(min(loss, 10) / 10 * chart.height)))
        pygame.draw.line(s, ui.BORDER, chart.bottomleft, chart.bottomright)
        pygame.draw.lines(s, ui.ACCENT, False, points, 2)
        loss = -(2 * self.label * math.log(self.probability) + (1 - self.label) * math.log(1 - self.probability))
        pygame.draw.circle(s, ui.TEXT, (chart.x + round(self.probability * chart.width), chart.bottom - round(min(loss, 10) / 10 * chart.height)), 5)
        ui.text(s, "Loss 0-10", ui.font(12), ui.DIM, rx, 396)
        ui.text(s, "0%", ui.font(12), ui.DIM, chart.left, chart.bottom + 8)
        ui.text(s, "Predicted risk", ui.font(12), ui.DIM, chart.centerx, chart.bottom + 8, "midtop")
        ui.text(s, "100%", ui.font(12), ui.DIM, chart.right, chart.bottom + 8, "topright")
        self._control(s, pygame.Rect(rx, 578, 42, 38), "-", ("probability", -.1))
        self._control(s, pygame.Rect(rx + 280, 578, 42, 38), "+", ("probability", .1))
        ui.text(s, f"p = {self.probability:.0%}", ui.font(20, True), ui.TEXT, rx + 162, 583, "midtop")
        ui.text(s, f"Single-example loss   {loss:.3f}", ui.font(19, True), ui.TEXT, rx, 650)

    def handle_event(self, event):
        action = None
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                self.page = (self.page + (1 if event.key == pygame.K_RIGHT else -1)) % len(self.TABS)
                self.focus = -1
            elif event.key == pygame.K_TAB and self.controls:
                self.focus = (self.focus + (-1 if getattr(event, "mod", 0) & pygame.KMOD_SHIFT else 1)) % len(self.controls)
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE) and 0 <= self.focus < len(self.controls):
                action = self.controls[self.focus][1]
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for index, (rect, candidate) in enumerate(self.controls):
                if rect.collidepoint(event.pos):
                    self.focus = index
                    action = candidate
                    break
        if action:
            kind, value = action
            if kind == "outcome":
                self.outcomes[value] = 1 - self.outcomes[value]
            elif kind == "probability":
                self.probability = round(max(.1, min(.9, self.probability + value)), 1)
            else:
                setattr(self, kind, value)

