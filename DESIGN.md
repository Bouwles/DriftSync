---
name: DriftSync
description: A calm, precise native research workspace for attention and error-risk analysis.
colors:
  background: "#171b1e"
  panel: "#1e2327"
  raised: "#272e32"
  sidebar: "#14181b"
  border: "#3b4449"
  text: "#ebefed"
  secondary-text: "#a4b1b4"
  accent: "#77c7c0"
  accent-hover: "#97d9d2"
  on-accent: "#0a0f14"
  success: "#8bc79f"
  warning: "#e1b970"
  danger: "#eb8f87"
  comparison: "#c4bca3"
typography:
  display:
    fontFamily: "Segoe UI, Inter, DejaVu Sans"
    fontSize: "36px"
    fontWeight: 700
  headline:
    fontFamily: "Segoe UI, Inter, DejaVu Sans"
    fontSize: "25px"
    fontWeight: 700
  title:
    fontFamily: "Segoe UI, Inter, DejaVu Sans"
    fontSize: "20px"
    fontWeight: 700
  body:
    fontFamily: "Segoe UI, Inter, DejaVu Sans"
    fontSize: "17px"
    fontWeight: 400
  label:
    fontFamily: "Segoe UI, Inter, DejaVu Sans"
    fontSize: "14px"
    fontWeight: 400
  log:
    fontFamily: "Consolas, Cascadia Code, DejaVu Sans Mono"
    fontSize: "13px"
    fontWeight: 400
rounded:
  chart: "3px"
  panel: "4px"
  control: "5px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.on-accent}"
    rounded: "{rounded.control}"
  button-primary-hover:
    backgroundColor: "{colors.accent-hover}"
    textColor: "{colors.on-accent}"
    rounded: "{rounded.control}"
  button-secondary:
    backgroundColor: "{colors.raised}"
    textColor: "{colors.text}"
    rounded: "{rounded.control}"
  input:
    backgroundColor: "{colors.raised}"
    textColor: "{colors.text}"
    rounded: "{rounded.control}"
  panel:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text}"
    rounded: "{rounded.panel}"
---

# Design System: DriftSync

## Overview

**Creative North Star: "The Research Workspace"**

DriftSync is a native Pygame interface with calm charcoal surfaces, readable instrumentation and restrained teal emphasis. Its visual hierarchy separates the participant's task from analytical evidence while keeping both visible. Credibility comes from clear labels, real measurements and explicit unavailable states.

**Key Characteristics:**

- Quiet, flat surfaces with compact controls.
- Clear separation between task interaction and analysis.
- Recorded evidence and uncertainty remain visible.

## Colors

The palette uses graphite neutrals and a muted teal primary accent. Normative values are in the frontmatter and implemented in `driftsync/ui.py`.

### Primary

Teal identifies the main action, keyboard focus and probability series. Its lighter hover variant provides immediate feedback on filled actions.

### Neutral

Charcoal forms the canvas; panel and raised tones establish grouping. Off-white carries primary text, cool grey carries supporting labels, and graphite borders separate regions. The sidebar is slightly darker than the canvas.

Green means correct or low risk, amber means caution or uncertainty, and soft red means error or elevated risk. A muted sand comparison colour distinguishes the Transformer series from teal LSTM traces.

**The Semantic Colour Rule.** Pair state colours with words or values; colour alone must not explain risk or availability.

## Typography

Use native Segoe UI regular and bold font files on Windows (`segoeui.ttf`, `segoeuib.ttf`). `pygame.font.SysFont` supplies the listed sans-serif fallbacks elsewhere. Logs use native Consolas files with the monospace fallback list. Avoid accidental Segoe UI Light selection through family lookup.

The shell uses display for the landing statement, headline for page headings, title for section headings, body for prose and controls, label for secondary information, and log for training output. Live-session instrumentation additionally uses compact 12–16px annotations, 23–30px metrics and a 26px task rule. These are logical canvas pixels before viewport scaling.

## Layout

The shell draws into a stable logical canvas (1280 × 800), with a fixed left sidebar (184px). Navigation rows are 40px high on a 48px rhythm. Content uses purposeful whitespace and aligned sections rather than a uniform grid of cards.

The live canvas is 1300 × 900 with the default simulator configuration: the original 900 × 650 stimulus coordinate space is preserved, a 400px analytical rail sits on the right, and additional vertical space holds session statistics. The rail uses 28px horizontal insets.

Resizable windows enforce a minimum of 960 × 600. `Viewport` scales the complete logical canvas proportionally and letterboxes unused space. It inversely maps pointer coordinates before hit testing. F11 toggles fullscreen. Layout scaling must preserve stimulus positions and task behaviour; it does not trigger a web-style reflow.

## Elevation & Depth

There are no decorative shadows, gradients or floating glass surfaces. Depth comes from the canvas, panel and raised tones, with thin borders and dividers. Hover and focus are immediate state changes, not animated movement.

## Shapes

Panels and controls are restrained rectangles with small corner radii. Borders are generally one pixel; keyboard focus uses a two-pixel teal outline around the control. The small angular signal mark is the recurring identity element. Task stimulus geometry remains governed by the simulator.

## Components

- **Buttons:** Teal filled primary actions and raised secondary actions use centred labels. Secondary hover changes border and text to teal. Keyboard focus adds an external outline; disabled controls use secondary text and ignore activation. Tab navigation and Enter activation keep shell actions available without a mouse.
- **Inputs:** Raised fill, muted placeholder, teal active border and blinking caret. Long text scrolls within the clipped field instead of covering adjacent controls.
- **Navigation:** Persistent left sidebar with a clear active row and compact text labels. Alt 1–6 shortcuts supplement pointer interaction.
- **Panels and logs:** Flat tonal groups with thin borders. Training logs use monospace and real process output; empty panels explain which action produces their evidence.
- **Live analysis rail:** State and advice precede error probability, forecast horizon, MC-dropout standard deviation, unavailable calibrated confidence, risk history and behavioural signals.
- **History chart:** Plot actual per-trial probability values with straight connecting segments. Show a dashed warning threshold, optional plus/minus one standard deviation band, and a pointer-selected trial/value tooltip. Empty history displays a waiting message. Do not smooth, interpolate extra samples or invent a trajectory.
- **Outcome strip:** Small semantic green/red markers represent recorded trials; nearby session accuracy and mean response use real completed outcomes.

## Do's and Don'ts

### Do:

- **Do** preserve logical task coordinates and inverse pointer mapping when resizing.
- **Do** label probability with its forecast horizon and uncertainty as MC-dropout standard deviation.
- **Do** show unavailable prediction confidence explicitly and wait for enough real trial history before showing risk.
- **Do** retain visible keyboard focus, clipped text and meaningful empty states.

### Don't:

- **Don't** introduce decorative gradients, glow, fake scores or fabricated charts.
- **Don't** present standard deviation as calibrated confidence or behavioural signals as causal explanations.
- **Don't** interpolate extra prediction samples or move task targets to accommodate the analysis rail.
