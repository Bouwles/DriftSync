"""Shared desktop presentation. No prediction or task logic lives here."""
from functools import lru_cache
import os
from pathlib import Path

import pygame

BG = (23, 27, 30)
PANEL = (30, 35, 39)
PANEL2 = (39, 46, 50)
BORDER = (59, 68, 73)
TEXT = (235, 239, 237)
DIM = (164, 177, 180)
ACCENT = (119, 199, 192)
GREEN = (139, 199, 159)
YELLOW = (225, 185, 112)
RED = (235, 143, 135)


@lru_cache(maxsize=32)
def font(size=16, bold=False, mono=False):
    # SDL's family lookup can select Segoe UI Light as the regular face on Windows.
    # Resolve the native regular/bold files explicitly when installed.
    filename = ("consolab.ttf" if bold else "consola.ttf") if mono else ("segoeuib.ttf" if bold else "segoeui.ttf")
    native = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / filename
    if native.is_file():
        return pygame.font.Font(str(native), size)
    return pygame.font.SysFont(
        "cascadiacode,consolas,dejavusansmono" if mono else "segoeui,inter,dejavusans",
        size, bold=bold,
    )


def text(surface, value, face, color, x, y, anchor="topleft"):
    rendered = face.render(str(value), True, color)
    rect = rendered.get_rect(**{anchor: (x, y)})
    surface.blit(rendered, rect)
    return rect


def paragraph(surface, value, rect, face=None, color=DIM, leading=6):
    face = face or font()
    y = rect.top
    old_clip = surface.get_clip()
    surface.set_clip(rect.clip(old_clip))
    for line in value.split("\n"):
        current = ""
        for word in line.split():
            candidate = (current + " " + word).strip()
            if current and face.size(candidate)[0] > rect.width:
                text(surface, current, face, color, rect.left, y)
                y += face.get_height() + leading
                current = word
            else:
                current = candidate
        text(surface, current, face, color, rect.left, y)
        y += face.get_height() + leading
    surface.set_clip(old_clip)
    return y


def signal_mark(surface, x, y, size=26):
    points = [(0, .55), (.24, .55), (.40, .18), (.60, .82), (.76, .45), (1, .45)]
    pygame.draw.lines(surface, ACCENT, False,
                      [(x + int(a * size), y + int(b * size)) for a, b in points], 2)


class Viewport:
    """Fit a stable logical canvas to a resizable window and invert pointer input.

    Letterboxing preserves target size/positions relative to the task canvas.
    No task configuration or model inputs change when the OS window is resized.
    """
    def __init__(self, size, title="DriftSync", window_size=None):
        font.cache_clear()
        self.size = size
        self.surface = pygame.Surface(size)
        self.fullscreen = False
        self.window_size = window_size or size
        self.window = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
        pygame.display.set_caption(title)
        icon = pygame.Surface((40, 40))
        icon.fill(BG)
        signal_mark(icon, 7, 7)
        pygame.display.set_icon(icon)

    @property
    def bounds(self):
        w, h = self.window.get_size()
        scale = min(w / self.size[0], h / self.size[1])
        sw, sh = max(1, round(self.size[0] * scale)), max(1, round(self.size[1] * scale))
        return pygame.Rect((w - sw) // 2, (h - sh) // 2, sw, sh)

    def point(self, pos):
        r = self.bounds
        return ((pos[0] - r.x) * self.size[0] / r.width,
                (pos[1] - r.y) * self.size[1] / r.height)

    def event(self, event):
        if event.type == pygame.VIDEORESIZE and not self.fullscreen:
            self.window_size = (max(min(960, self.size[0]), event.w),
                                max(min(600, self.size[1]), event.h))
            self.window = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
            self.fullscreen = not self.fullscreen
            self.window = pygame.display.set_mode(
                (0, 0) if self.fullscreen else self.window_size,
                pygame.FULLSCREEN if self.fullscreen else pygame.RESIZABLE)
        if hasattr(event, "pos"):
            attrs = event.dict.copy()
            attrs["pos"] = self.point(event.pos)
            return pygame.event.Event(event.type, attrs)
        return event

    def present(self):
        self.window.fill(BG)
        rect = self.bounds
        if rect.size == self.size:
            self.window.blit(self.surface, rect)
        else:
            self.window.blit(pygame.transform.smoothscale(self.surface, rect.size), rect)
        pygame.display.flip()


def history_chart(surface, rect, values, *, threshold=None, first_trial=1,
                  deviations=None, pointer=None):
    """Actual per-trial probability, with optional MC standard deviation band."""
    plot = pygame.Rect(rect.x + 36, rect.y + 10, rect.width - 48, rect.height - 42)
    for value in (0, .5, 1):
        y = plot.bottom - round(value * plot.height)
        pygame.draw.line(surface, BORDER, (plot.left, y), (plot.right, y))
        text(surface, f"{value:.0%}", font(12), DIM, plot.left - 8, y, "midright")
    if threshold is not None:
        y = plot.bottom - round(threshold * plot.height)
        for x in range(plot.left, plot.right, 8):
            pygame.draw.line(surface, YELLOW, (x, y), (min(x + 4, plot.right), y))
    values = list(values)
    if not values:
        text(surface, "Waiting for predictions", font(14), DIM, *plot.center, "center")
        return
    points = [(plot.left + round(i / max(1, len(values) - 1) * plot.width),
               plot.bottom - round(max(0, min(1, p)) * plot.height)) for i, p in enumerate(values)]
    if deviations and len(values) > 1:
        deviations = list(deviations)
        band = [(x, plot.bottom - round(min(1, p + sd) * plot.height))
                for (x, _), p, sd in zip(points, values, deviations)]
        band += [(x, plot.bottom - round(max(0, p - sd) * plot.height))
                 for (x, _), p, sd in reversed(list(zip(points, values, deviations)))]
        pygame.draw.polygon(surface, PANEL2, band)
    if len(points) > 1:
        pygame.draw.lines(surface, ACCENT, False, points, 2)
    pygame.draw.circle(surface, ACCENT, points[-1], 3)
    text(surface, f"Trial {first_trial}", font(12), DIM, plot.left, plot.bottom + 10)
    if len(values) > 1:
        text(surface, str(first_trial + len(values) - 1), font(12), DIM,
             plot.right, plot.bottom + 10, "topright")
    if pointer and plot.collidepoint(pointer):
        i = round((pointer[0] - plot.x) / plot.width * max(0, len(values) - 1))
        x, y = points[i]
        pygame.draw.line(surface, DIM, (x, plot.top), (x, plot.bottom))
        pygame.draw.circle(surface, TEXT, (x, y), 4)
        label = f"Trial {first_trial + i}   {values[i]:.1%}"
        width = font(13).size(label)[0] + 16
        tip = pygame.Rect(min(x, rect.right - width), plot.top + 4, width, 26)
        pygame.draw.rect(surface, BG, tip, border_radius=3)
        text(surface, label, font(13), TEXT, tip.x + 8, tip.y + 4)


def session_message(screen, lines):
    """Shared intro, calibration and summary typography, retaining supplied copy."""
    screen.fill(BG)
    w, h = screen.get_size()
    left = max(36, (w - 720) // 2)
    signal_mark(screen, left, 32)
    text(screen, "DriftSync", font(20, True), TEXT, left + 38, 32)
    pygame.draw.line(screen, BORDER, (left, 78), (w - left, 78))
    nonempty = [(value, color) for value, _, color in lines if value]
    y = max(108, (h - len(nonempty) * 42) // 2)
    for i, (value, color) in enumerate(nonempty):
        face = font(28, True) if i == 0 else font(18)
        y = paragraph(screen, value, pygame.Rect(left, y, w - left * 2, h - y - 36),
                      face, TEXT if i == 0 else color, leading=7) + (20 if i == 0 else 10)
    text(screen, "Enter to continue     Esc to return", font(14), DIM, left, h - 32)
