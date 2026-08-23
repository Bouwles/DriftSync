"""Generate DriftSync README screenshots and a working demo GIF."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path("docs/assets")
W, H = 1200, 760

BG = (14, 18, 26)
PANEL = (25, 32, 43)
PANEL_2 = (34, 43, 56)
BORDER = (73, 86, 108)
TEXT = (237, 242, 248)
MUTED = (143, 156, 178)
BLUE = (89, 168, 255)
GREEN = (80, 206, 136)
YELLOW = (245, 181, 72)
RED = (247, 93, 82)
PURPLE = (189, 142, 255)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = font(46, True)
F_H1 = font(30, True)
F_H2 = font(22, True)
F_BODY = font(18)
F_SMALL = font(14)
F_MONO = font(16)


def panel(draw: ImageDraw.ImageDraw, xy, fill=PANEL, outline=BORDER, radius=10) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=1)


def label(draw: ImageDraw.ImageDraw, xy, text, fill=TEXT, fnt=F_BODY) -> None:
    draw.text(xy, text, fill=fill, font=fnt)


def progress(draw: ImageDraw.ImageDraw, xy, value, fill, bg=PANEL_2) -> None:
    x0, y0, x1, y1 = xy
    panel(draw, xy, fill=bg, radius=6)
    draw.rounded_rectangle((x0, y0, x0 + int((x1 - x0) * value), y1), radius=6, fill=fill)


def draw_sidebar(draw: ImageDraw.ImageDraw, active: str) -> None:
    draw.rectangle((0, 0, 214, H), fill=(16, 22, 32))
    label(draw, (28, 30), "DriftSync", BLUE, F_H1)
    label(draw, (30, 72), "cognitive risk lab", MUTED, F_SMALL)
    items = ["Learn", "Run Demo", "Play Task", "Live Mode", "Results"]
    y = 130
    for item in items:
        selected = item == active
        if selected:
            draw.rounded_rectangle((22, y - 8, 192, y + 32), radius=8, fill=(31, 45, 63))
            draw.rectangle((22, y - 8, 27, y + 32), fill=BLUE)
        label(draw, (42, y), item, TEXT if selected else MUTED, F_BODY)
        y += 54


def draw_curve(draw: ImageDraw.ImageDraw, box, values, color=BLUE, threshold=0.65) -> None:
    x0, y0, x1, y1 = box
    panel(draw, box, fill=(18, 24, 34), radius=8)
    for i in range(1, 4):
        y = y0 + i * (y1 - y0) // 4
        draw.line((x0 + 16, y, x1 - 16, y), fill=(40, 50, 64))
    ty = y1 - int((y1 - y0 - 36) * threshold) - 18
    draw.line((x0 + 16, ty, x1 - 16, ty), fill=RED, width=2)
    pts = []
    for idx, value in enumerate(values):
        x = x0 + 20 + idx * (x1 - x0 - 40) / (len(values) - 1)
        y = y1 - 18 - value * (y1 - y0 - 36)
        pts.append((x, y))
    draw.line(pts, fill=color, width=4)
    for x, y in pts[-3:]:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)


def screenshot_overview() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw_sidebar(draw, "Run Demo")
    label(draw, (252, 40), "Real-time cognitive drift prediction", TEXT, F_TITLE)
    label(draw, (256, 100), "Synthetic sessions -> feature windows -> model risk -> warning lead time", MUTED, F_BODY)

    panel(draw, (252, 150, 1138, 356))
    values = [0.16, 0.18, 0.22, 0.21, 0.30, 0.36, 0.42, 0.50, 0.61, 0.70, 0.78, 0.73, 0.83]
    draw_curve(draw, (282, 186, 840, 326), values)
    label(draw, (880, 190), "Drift probability", MUTED, F_SMALL)
    label(draw, (880, 220), "0.78", RED, font(58, True))
    label(draw, (880, 286), "Warning active", RED, F_H2)

    for idx, (title, value, color) in enumerate([
        ("Accuracy", "74.4%", GREEN),
        ("AUC", "0.89", BLUE),
        ("Lead time", "4.3s", YELLOW),
    ]):
        x = 252 + idx * 296
        panel(draw, (x, 390, x + 270, 522))
        label(draw, (x + 24, 416), title, MUTED, F_BODY)
        label(draw, (x + 24, 452), value, color, font(44, True))

    panel(draw, (252, 560, 1138, 704))
    label(draw, (278, 586), "Why the warning fired", TEXT, F_H2)
    for i, line in enumerate([
        "Reaction time is 28% above baseline",
        "Error rate reached 40% in the last 5 trials",
        "Reaction time trending slower over recent trials",
    ]):
        label(draw, (282, 624 + i * 26), f"- {line}", MUTED, F_BODY)
    return img


def screenshot_pipeline() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw_sidebar(draw, "Learn")
    label(draw, (252, 40), "End-to-end ML pipeline", TEXT, F_TITLE)
    steps = [
        ("Task Engine", "reaction time, action, correctness", BLUE),
        ("15 Features", "rolling errors, streaks, RT trend, fatigue", GREEN),
        ("Sequence Window", "last 20 trials become one model input", YELLOW),
        ("LSTM / Transformer", "P(error in next 5 trials) + uncertainty", PURPLE),
        ("Live Warning", "risk overlay, explanations, lead time", RED),
    ]
    y = 154
    for idx, (title, detail, color) in enumerate(steps):
        panel(draw, (286, y, 1034, y + 74), fill=PANEL)
        draw.ellipse((306, y + 20, 340, y + 54), fill=color)
        label(draw, (360, y + 16), title, TEXT, F_H2)
        label(draw, (360, y + 44), detail, MUTED, F_BODY)
        if idx < len(steps) - 1:
            draw.line((660, y + 78, 660, y + 108), fill=BORDER, width=3)
        y += 110
    return img


def screenshot_results() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw_sidebar(draw, "Results")
    label(draw, (252, 40), "Model comparison and session evidence", TEXT, F_TITLE)
    panel(draw, (252, 132, 1138, 692))
    headers = ["Model", "Accuracy", "F1", "AUC", "Params"]
    rows = [
        ["LSTM", "0.744", "0.837", "0.872", "407k"],
        ["Transformer", "0.754", "0.844", "0.888", "540k"],
        ["Random Forest", "fallback", "fast", "explainable", "100 trees"],
    ]
    x_positions = [296, 506, 686, 846, 1002]
    for x, header in zip(x_positions, headers):
        label(draw, (x, 172), header, MUTED, F_BODY)
    draw.line((286, 206, 1098, 206), fill=BORDER)
    for row_i, row in enumerate(rows):
        y = 234 + row_i * 68
        if row_i == 1:
            draw.rounded_rectangle((280, y - 12, 1108, y + 44), radius=8, fill=(29, 40, 56))
        for x, cell in zip(x_positions, row):
            color = BLUE if cell == "Transformer" else GREEN if cell in {"0.754", "0.844", "0.888"} else TEXT
            label(draw, (x, y), cell, color, F_H2 if x == 296 else F_BODY)
    label(draw, (296, 492), "Generated outputs", TEXT, F_H2)
    for i, name in enumerate(["training curves", "calibration", "confusion matrix", "realtime log"]):
        x = 296 + i * 190
        progress(draw, (x, 544, x + 144, 560), 0.86 - i * 0.1, [BLUE, GREEN, YELLOW, PURPLE][i])
        label(draw, (x, 574), name, MUTED, F_SMALL)
    return img


def gif_frames() -> list[Image.Image]:
    frames = []
    for frame in range(24):
        img = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)
        draw_sidebar(draw, "Live Mode")
        label(draw, (252, 40), "Live inference: warning before the error", TEXT, F_TITLE)
        t = frame / 23
        risk = min(0.90, 0.18 + t * 0.72 + 0.05 * math.sin(t * math.pi * 4))
        unc = 0.05 + 0.09 * max(0.0, t - 0.45)
        values = [max(0.05, min(0.95, 0.18 + (i / 22) * 0.72 + 0.05 * math.sin(i))) for i in range(frame + 2)]
        panel(draw, (252, 130, 1138, 704))
        label(draw, (286, 164), "CLICK CIRCLES", BLUE, F_H1)
        progress(draw, (286, 220, 858, 242), risk, RED if risk > 0.65 else YELLOW if risk > 0.4 else GREEN)
        label(draw, (884, 214), f"Drift P: {risk:.2f} +/- {unc:.2f}", TEXT, F_BODY)
        draw_curve(draw, (286, 292, 858, 520), values, RED if risk > 0.65 else BLUE)
        shape_color = GREEN if frame % 3 != 0 else RED
        draw.ellipse((940, 312, 1060, 432), fill=shape_color, outline=TEXT, width=4)
        label(draw, (952, 456), "CIRCLE", TEXT, F_H2)
        if risk > 0.65:
            draw.rounded_rectangle((304, 582, 1088, 654), radius=10, fill=(75, 28, 32), outline=RED, width=2)
            label(draw, (332, 604), "COGNITIVE DRIFT DETECTED", RED, F_H2)
            label(draw, (332, 632), "Slow RT trend + recent error cluster", TEXT, F_BODY)
        else:
            label(draw, (316, 604), "Monitoring recent behavior...", MUTED, F_H2)
        frames.append(img)
    return frames


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    screenshot_overview().save(OUT_DIR / "driftsync-overview.png")
    screenshot_pipeline().save(OUT_DIR / "driftsync-pipeline.png")
    screenshot_results().save(OUT_DIR / "driftsync-results.png")
    frames = gif_frames()
    frames[0].save(
        OUT_DIR / "driftsync-live-demo.gif",
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=True,
    )
    print(f"Generated showcase assets in {OUT_DIR}")


if __name__ == "__main__":
    main()
