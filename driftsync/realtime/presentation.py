"""Live session rendering, isolated from sequence inference and task recording."""
import pygame

from driftsync import ui
from driftsync.configs import CONFIG


def prediction_state(sim):
    if not sim._model_ready:
        return "Recording only", ui.DIM, "Train a model to enable predictions."
    count = len(sim.inference._feat_buffer)
    needed = sim.inference._seq_len
    if count < needed:
        return "Collecting history", ui.DIM, f"{count} of {needed} trials collected."
    if sim._last_prob > sim.rt_cfg.warning_threshold:
        return "Elevated error risk", ui.RED, "Check the rule before your next response."
    if sim._last_unc > sim.rt_cfg.uncertainty_threshold:
        return "Uncertain prediction", ui.YELLOW, "Model outputs vary. Interpret risk cautiously."
    if sim._last_prob >= .4:
        return "Moderate error risk", ui.YELLOW, "Watch your response pace and recent errors."
    return "Low error risk", ui.GREEN, "Continue following the task rule."


def render_live(sim, screen, shape, sx, sy, radius, rule, elapsed, time_window, drawers):
    screen.fill(ui.BG)
    task_w = sim.sim_cfg.window_width
    width, height = screen.get_size()
    pygame.draw.rect(screen, ui.PANEL, (task_w, 0, width - task_w, height))
    pygame.draw.line(screen, ui.BORDER, (task_w, 0), (task_w, height))
    ui.signal_mark(screen, 28, 26)
    ui.text(screen, "DriftSync", ui.font(20, True), ui.TEXT, 65, 26)
    ui.text(screen, "Live session", ui.font(14), ui.DIM, 28, 58)
    ui.text(screen, f"Click {rule.lower()}s", ui.font(26, True), ui.TEXT,
            task_w // 2, 26, "midtop")
    ui.text(screen, f"Trial {sim.engine.trial_count + 1} / {sim.sim_cfg.num_trials}",
            ui.font(14), ui.DIM, task_w - 28, 34, "topright")
    ratio = max(0, 1 - elapsed / time_window)
    color = ui.ACCENT if ratio > .5 else ui.YELLOW if ratio > .25 else ui.RED
    pygame.draw.rect(screen, ui.PANEL2, (28, 86, task_w - 56, 4))
    pygame.draw.rect(screen, color, (28, 86, round((task_w - 56) * ratio), 4))
    ui.text(screen, f"{max(0, time_window - elapsed):.1f}s remaining", ui.font(13), ui.DIM,
            task_w - 28, 101, "topright")
    color = ui.GREEN if shape == rule else ui.RED
    drawers[shape](screen, color, sx, sy, radius)
    ui.text(screen, shape.title(), ui.font(14), ui.DIM, sx, sy + radius + 12, "midtop")

    bottom = max(sim.sim_cfg.window_height - 20, height - 130)
    pygame.draw.line(screen, ui.BORDER, (28, bottom), (task_w - 28, bottom))
    trials = sim.engine.session_data.trials
    accuracy = f"{sum(t.is_correct for t in trials) / len(trials):.0%}" if trials else "--"
    mean_rt = f"{sum(t.reaction_time for t in trials) / len(trials):.2f}s" if trials else "--"
    for i, (label, value) in enumerate([("Session accuracy", accuracy), ("Mean response", mean_rt)]):
        x = 28 + i * 200
        ui.text(screen, label, ui.font(14), ui.DIM, x, bottom + 20)
        ui.text(screen, value, ui.font(23, True), ui.TEXT, x, bottom + 44)
    ui.text(screen, "Recent outcomes", ui.font(14), ui.DIM, 444, bottom + 20)
    for i, trial in enumerate(trials[-20:]):
        rect = pygame.Rect(444 + i * 17, bottom + 49, 10, 14)
        pygame.draw.rect(screen, ui.GREEN if trial.is_correct else ui.RED, rect, border_radius=2)
    ui.text(screen, "Click matching shape     Space  Skip     Esc  End & save     F11  Full screen",
            ui.font(14), ui.DIM, 28, height - 27)

    x, right = task_w + 28, width - 28
    state, color, action = prediction_state(sim)
    available = sim._model_ready and len(sim.inference._feat_buffer) >= sim.inference._seq_len
    ui.text(screen, "Current drift state", ui.font(14), ui.DIM, x, 30)
    ui.text(screen, state, ui.font(25, True), color, x, 60)
    ui.paragraph(screen, action, pygame.Rect(x, 100, right - x, 62), ui.font(15))
    ui.text(screen, "Error risk", ui.font(16), ui.TEXT, x, 174)
    ui.text(screen, f"{sim._last_prob:.0%}" if available else "--", ui.font(30, True),
            color, right, 166, "topright")
    ui.text(screen, f"Error within the next {CONFIG.data.prediction_horizon} trials", ui.font(13), ui.DIM, x, 205)
    pygame.draw.line(screen, ui.BORDER, (x, 241), (right, 241))
    ui.text(screen, "Model uncertainty", ui.font(15), ui.TEXT, x, 260)
    ui.text(screen, f"{sim._last_unc * 100:.1f} pp" if available else "--", ui.font(17, True),
            ui.YELLOW if available and sim._last_unc > sim.rt_cfg.uncertainty_threshold else ui.TEXT,
            right, 258, "topright")
    ui.text(screen, "MC-dropout standard deviation", ui.font(13), ui.DIM, x, 287)
    ui.text(screen, "Prediction confidence", ui.font(15), ui.TEXT, x, 326)
    ui.text(screen, "Not calibrated", ui.font(14), ui.DIM, right, 352, "topright")
    pygame.draw.line(screen, ui.BORDER, (x, 385), (right, 385))
    ui.text(screen, "Risk over time", ui.font(17, True), ui.TEXT, x, 405)
    if len(sim._prob_history) > 1:
        delta = (sim._prob_history[-1] - sim._prob_history[0]) * 100
        direction = "Rising" if delta > .5 else "Falling" if delta < -.5 else "Steady"
        ui.text(screen, f"{direction} / {delta:+.0f} pp", ui.font(13), ui.DIM, right, 410, "topright")
    first = max(sim.inference._seq_len, sim.engine.trial_count - len(sim._prob_history) + 1)
    pointer = sim.viewport.point(pygame.mouse.get_pos()) if hasattr(sim, "viewport") else None
    ui.history_chart(screen, pygame.Rect(x - 4, 445, right - x + 4, 137), sim._prob_history,
                     threshold=sim.rt_cfg.warning_threshold, first_trial=first,
                     deviations=sim._unc_history, pointer=pointer)
    ui.text(screen, f"Dashed: {sim.rt_cfg.warning_threshold:.0%} warning   Band: +/-1 SD",
            ui.font(12), ui.DIM, x, 591)
    pygame.draw.line(screen, ui.BORDER, (x, 624), (right, 624))
    ui.text(screen, "Behavioural signals", ui.font(17, True), ui.TEXT, x, 644)
    reasons = sim._explainer.explain(sim.inference.get_live_stats(), sim._last_prob) if available and sim._explainer else []
    message = "\n".join(reasons) if reasons else "No strong individual signal detected." if available else "Signals will appear after enough trials."
    ui.paragraph(screen, message, pygame.Rect(x, 678, right - x, height - 730), ui.font(14), leading=2)
    ui.text(screen, f"{sim._model_type.upper()} / MC dropout" if sim._model_ready else "Task recording / no model",
            ui.font(12), ui.DIM, x, height - 25)
