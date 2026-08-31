"""
Drift Score: deterministic prototype fatigue-risk formula.

The modeled factors — PVT performance, sleep debt, circadian disruption,
and workload — are informed by established human-performance and fatigue
research.

The specific normalization rules, weights, and risk thresholds used here
are prototype design assumptions and are not NASA-validated operational limits.

IBM Granite is used only after the deterministic calculations to explain
the resulting mission context. It does not calculate or modify the risk score.
"""

from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Prototype weights.
#
# The modeled factors are literature-informed, but the exact weights below
# are prototype design assumptions and are not NASA-validated operational
# weights. PVT receives the highest prototype weight because vigilant-attention
# performance is widely used in sleep-loss and fatigue research.
# ---------------------------------------------------------------------------
@dataclass
class DriftWeights:
    reaction_time: float = 0.40
    sleep_debt: float = 0.30
    circadian: float = 0.15
    workload: float = 0.15

    def validate(self):
        total = (
            self.reaction_time
            + self.sleep_debt
            + self.circadian
            + self.workload
        )

        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class AstronautSignals:
    """Raw inputs for one astronaut at one point in time."""

    # PVT (Psychomotor Vigilance Test)
    pvt_lapses: int
    pvt_baseline_lapses: float

    # Sleep
    hours_slept_last_24: float

    # Circadian
    minutes_phase_shift: float

    # Workload
    current_task_load: float
    rolling_avg_task_load: float

    # Defaults / carried-over state
    recommended_sleep_hours: float = 8.0
    prior_sleep_debt: float = 0.0

    # Prototype normalization assumption:
    # a 12-hour phase shift maps to the maximum circadian sub-score.
    max_tolerable_shift_minutes: float = 720.0


@dataclass
class DriftResult:
    reaction_time_score: float
    sleep_debt_score: float
    circadian_score: float
    workload_score: float
    drift_score: float
    updated_sleep_debt_hours: float
    risk_level: Literal["nominal", "elevated", "high", "critical"]


def _clip01(x: float) -> float:
    """Keep a score within the prototype 0-1 range."""
    return max(0.0, min(1.0, x))


def reaction_time_score(pvt_lapses: int, baseline: float) -> float:
    """
    Measures change in PVT lapses relative to the astronaut's personal
    baseline.

    PVT is used in fatigue and sleep-loss research as a measure of
    vigilant-attention performance.

    Prototype normalization:
    a doubling of lapses relative to baseline maps to a score of 1.0.
    """

    baseline = max(baseline, 1.0)

    deviation = (pvt_lapses - baseline) / baseline

    # Better-than-baseline performance does not create negative fatigue risk.
    return _clip01(deviation)


def sleep_debt_score(
    hours_slept_last_24: float,
    recommended: float,
    prior_debt: float,
    recovery_rate: float = 0.7,
    cap_hours: float = 12.0,
) -> tuple[float, float]:
    """
    Prototype representation of accumulated sleep debt.

    Sleep loss and accumulated sleep pressure are informed by established
    sleep and fatigue research.

    The exact recovery rate and 12-hour normalization cap are prototype
    assumptions, not NASA operational thresholds.

    Returns:
        (normalized sleep-debt score, updated sleep debt in hours)
    """

    todays_deficit = max(
        0.0,
        recommended - hours_slept_last_24,
    )

    # When recommended sleep is reached, part of the previous debt recovers.
    carried = (
        prior_debt * recovery_rate
        if hours_slept_last_24 >= recommended
        else prior_debt
    )

    updated_debt = min(
        cap_hours,
        carried + todays_deficit,
    )

    return (
        _clip01(updated_debt / cap_hours),
        updated_debt,
    )


def circadian_score(
    minutes_phase_shift: float,
    max_tolerable_minutes: float,
) -> float:
    """
    Prototype proxy for circadian misalignment.

    Circadian disruption is a documented human-performance concern in
    spaceflight and can be influenced by factors such as shifted schedules,
    mistimed light exposure, and operational demands.

    The conversion from phase-shift minutes to a 0-1 score is a prototype
    normalization rule.
    """

    return _clip01(
        abs(minutes_phase_shift) / max_tolerable_minutes
    )


def workload_score(
    current_load: float,
    rolling_avg: float,
) -> float:
    """
    Measures current task load relative to the astronaut's own rolling
    workload baseline.

    Workload is included because operational workload is a relevant
    contributor to fatigue and performance risk.

    Prototype normalization:
    average workload maps to 0 and 2x the rolling average maps to 1.0.
    """

    rolling_avg = max(rolling_avg, 1.0)

    ratio = current_load / rolling_avg

    return _clip01(ratio - 1.0)


def classify_risk(
    score: float,
) -> Literal["nominal", "elevated", "high", "critical"]:
    """
    Convert the 0-1 Drift Score into prototype risk bands.

    These thresholds are prototype design assumptions and are not
    NASA operational thresholds.
    """

    if score < 0.35:
        return "nominal"

    if score < 0.55:
        return "elevated"

    if score < 0.75:
        return "high"

    return "critical"


def compute_drift_score(
    signals: AstronautSignals,
    weights: DriftWeights = DriftWeights(),
) -> DriftResult:
    """
    Calculate the deterministic Fatigue Drift Score.

    Each input is converted into a normalized sub-score and combined using
    the prototype weights defined in DriftWeights.

    No AI or LLM is involved in this calculation.
    """

    weights.validate()

    rt = reaction_time_score(
        signals.pvt_lapses,
        signals.pvt_baseline_lapses,
    )

    sleep, updated_debt = sleep_debt_score(
        signals.hours_slept_last_24,
        signals.recommended_sleep_hours,
        signals.prior_sleep_debt,
    )

    circ = circadian_score(
        signals.minutes_phase_shift,
        signals.max_tolerable_shift_minutes,
    )

    work = workload_score(
        signals.current_task_load,
        signals.rolling_avg_task_load,
    )

    drift = (
        weights.reaction_time * rt
        + weights.sleep_debt * sleep
        + weights.circadian * circ
        + weights.workload * work
    )

    drift = _clip01(drift)

    return DriftResult(
        reaction_time_score=round(rt, 4),
        sleep_debt_score=round(sleep, 4),
        circadian_score=round(circ, 4),
        workload_score=round(work, 4),
        drift_score=round(drift, 4),
        updated_sleep_debt_hours=round(updated_debt, 2),
        risk_level=classify_risk(drift),
    )


# Prototype threshold for the legacy AI explanation layer.
# 0.55 corresponds to "high" risk and above under the current risk bands.
EXPLANATION_TRIGGER_THRESHOLD = 0.55


if __name__ == "__main__":
    # Quick smoke test
    example = AstronautSignals(
        pvt_lapses=9,
        pvt_baseline_lapses=3,
        hours_slept_last_24=5.0,
        prior_sleep_debt=2.5,
        minutes_phase_shift=310,
        current_task_load=14,
        rolling_avg_task_load=8,
    )

    result = compute_drift_score(example)

    print(result)
    print(
        "Explain?",
        result.drift_score >= EXPLANATION_TRIGGER_THRESHOLD,
    )