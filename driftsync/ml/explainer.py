"""
Rule-Based Explainability
==========================
When DriftSync generates a high risk score, this module produces a plain-text
explanation of the main contributing factors.

The explanations compare the current rolling performance statistics against
the user's personal baseline (from calibration) or against fixed norms if
calibration has not been done.

This is intentionally rule-based so it works without model introspection.
SHAP or attention-based explanations could be added in the future.
"""

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from driftsync.ml.calibrator import BaselineStats


class RuleBasedExplainer:
    """
    Generates human-readable reasons for a high risk score.

    Usage
    -----
        explainer = RuleBasedExplainer(baseline)
        reasons = explainer.explain(live_stats, risk_score)
    """

    RT_SLOW_THRESHOLD      = 0.20
    ACC_DROP_THRESHOLD     = 0.10
    ERR_RATE_THRESHOLD     = 0.30
    STREAK_ERROR_THRESHOLD = 2
    RT_VARIANCE_THRESHOLD  = 0.25
    TREND_THRESHOLD        = 0.15
    FATIGUE_THRESHOLD      = 0.20

    def __init__(self, baseline: Optional["BaselineStats"] = None):
        self.baseline = baseline

    def explain(self, live_stats: dict, risk_score: float) -> List[str]:
        """
        Generate explanation items for the current risk score.

        Args:
            live_stats: Dictionary containing current performance metrics.
                Expected keys (all optional):
                  mean_rt_recent, rolling_acc_10, rolling_err_rate_5,
                  rolling_err_rate_10, error_streak, rt_variance,
                  rt_trend, fatigue_index
            risk_score: float in [0, 1].
        Returns:
            List of short explanation strings (max 4).
        """
        if risk_score < 0.40:
            return []

        reasons = []
        b = self.baseline

        rt_recent = live_stats.get("mean_rt_recent")
        if rt_recent is not None:
            if b is not None and b.mean_rt > 0:
                rt_pct = (rt_recent - b.mean_rt) / b.mean_rt
                if rt_pct >= self.RT_SLOW_THRESHOLD:
                    reasons.append(f"Reaction time is {rt_pct:.0%} above personal baseline")
            elif rt_recent > 1.8:
                reasons.append("Reaction time is elevated")

        acc_recent = live_stats.get("rolling_acc_10")
        if acc_recent is not None:
            baseline_acc = b.accuracy if b else 0.80
            drop = baseline_acc - acc_recent
            if drop >= self.ACC_DROP_THRESHOLD:
                reasons.append(f"Accuracy dropped {drop:.0%} from baseline in last 10 trials")

        err5  = live_stats.get("rolling_err_rate_5",  0.0)
        err10 = live_stats.get("rolling_err_rate_10", 0.0)
        if err5 >= self.ERR_RATE_THRESHOLD:
            reasons.append(f"Error rate reached {err5:.0%} in last 5 trials")
        elif err10 >= self.ERR_RATE_THRESHOLD:
            reasons.append(f"Error rate reached {err10:.0%} in last 10 trials")

        streak = live_stats.get("error_streak", 0)
        if streak >= self.STREAK_ERROR_THRESHOLD:
            reasons.append(f"{streak} consecutive error{'s' if streak != 1 else ''} detected")

        rt_var = live_stats.get("rt_variance", 0.0)
        if rt_var >= self.RT_VARIANCE_THRESHOLD:
            reasons.append("Response timing has become inconsistent")

        rt_trend = live_stats.get("rt_trend", 0.0)
        if rt_trend >= self.TREND_THRESHOLD:
            reasons.append("Reaction time trending slower over recent trials")

        fatigue = live_stats.get("fatigue_index", 0.0)
        if fatigue >= self.FATIGUE_THRESHOLD:
            reasons.append("Fatigue index elevated (session time vs error accumulation)")

        if not reasons and risk_score >= 0.65:
            reasons.append("Multiple behavioral signals indicate elevated risk")

        return reasons[:4]

    def format_panel(self, live_stats: dict, risk_score: float) -> List[str]:
        """
        Format a multi-line explanation panel ready for display.
        Returns list of strings, one per display line.
        """
        pct   = int(risk_score * 100)
        level = "HIGH" if risk_score >= 0.65 else "MODERATE" if risk_score >= 0.40 else "LOW"
        lines = [f"Risk: {pct}%  [{level}]"]

        reasons = self.explain(live_stats, risk_score)
        if reasons:
            lines.append("Factors:")
            for r in reasons:
                lines.append(f"  - {r}")
        else:
            lines.append("No strong individual signal detected.")

        if not self.baseline:
            lines.append("  (Run calibration for personal baseline)")

        return lines
