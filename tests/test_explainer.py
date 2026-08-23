from driftsync.ml.calibrator import BaselineStats
from driftsync.ml.explainer import RuleBasedExplainer


def baseline() -> BaselineStats:
    return BaselineStats(
        mean_rt=0.5,
        std_rt=0.1,
        median_rt=0.5,
        q25_rt=0.4,
        q75_rt=0.6,
        accuracy=0.9,
        error_rate=0.1,
        mean_iti=1.0,
        num_trials=25,
        computed_at="2026-08-23T00:00:00",
    )


def test_explainer_reports_personal_baseline_deviation():
    explainer = RuleBasedExplainer(baseline())

    reasons = explainer.explain({"mean_rt_recent": 0.65, "rolling_acc_10": 0.72}, risk_score=0.70)

    assert "Reaction time is 30% above personal baseline" in reasons
    assert "Accuracy dropped 18% from baseline in last 10 trials" in reasons


def test_explainer_formats_low_risk_panel_without_reasons():
    panel = RuleBasedExplainer().format_panel({}, risk_score=0.20)

    assert panel[0] == "Risk: 20%  [LOW]"
    assert "No strong individual signal detected." in panel
    assert "  (Run calibration for personal baseline)" in panel
