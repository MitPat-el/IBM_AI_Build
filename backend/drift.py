"""
Drift Score: deterministic fatigue-risk formula.

No AI/LLM involved in this file. This is pure math, grounded in two
well-established fatigue models used in aerospace/aviation ops:

  - Two-Process Model of Sleep Regulation (Borbely) -> homeostatic sleep debt
  - Three-Process Model of Alertness (Akerstedt/Folkard) -> + circadian process
  - SAFTE/FAST (Fatigue Avoidance Scheduling Tool) -> the general pattern of
    combining homeostatic + circadian + task-load into one effectiveness score

The output is a single 0-1 "DriftScore" per astronaut per time window.
IBM Bob / watsonx only gets called AFTER this number crosses a threshold,
purely to turn the four sub-scores into plain-language text. It never
touches the math.
"""

from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Weights: literature-informed defaults, NOT AI-tuned. Reaction time (PVT) is
# weighted highest because it is the most validated real-time fatigue
# biomarker in the NASA/military sleep literature. Expose these as
# parameters so you can show sensitivity analysis in your demo ("trust
# score" hook).
# ---------------------------------------------------------------------------
@dataclass
class DriftWeights:
    reaction_time: float = 0.40
    sleep_debt: float = 0.30
    circadian: float = 0.15
    workload: float = 0.15

    def validate(self):
        total = self.reaction_time + self.sleep_debt + self.circadian + self.workload
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class AstronautSignals:
    """Raw inputs for one astronaut at one point in time."""
    # PVT (Psychomotor Vigilance Test)
    pvt_lapses: int              # # of lapses (RT > 500ms) in a standard PVT bout
    pvt_baseline_lapses: float   # this astronaut's personal pre-mission baseline

    # Sleep
    hours_slept_last_24: float

    # Circadian
    minutes_phase_shift: float   # cumulative drift from entrained 24h rhythm

    # Workload
    current_task_load: float     # e.g. scheduled task-minutes or cognitive demand index
    rolling_avg_task_load: float # that astronaut's own rolling baseline

    # Defaults / carried-over state
    recommended_sleep_hours: float = 8.0
    prior_sleep_debt: float = 0.0   # carried over from previous day, in hours
    max_tolerable_shift_minutes: float = 720.0  # 12h = fully misaligned, floor for scoring


@dataclass
class DriftResult:
    reaction_time_score: float
    sleep_debt_score: float
    circadian_score: float
    workload_score: float
    drift_score: float
    updated_sleep_debt_hours: float  # carry this forward to next day's calc
    risk_level: Literal["nominal", "elevated", "high", "critical"]


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def reaction_time_score(pvt_lapses: int, baseline: float) -> float:
    """
    Deviation from personal PVT baseline, normalized 0-1.
    A doubling of lapses vs baseline -> ~1.0 (saturates).
    Guards against baseline=0 (early mission, no history yet).
    """
    baseline = max(baseline, 1.0)
    deviation = (pvt_lapses - baseline) / baseline
    return _clip01(deviation)  # negative deviation (fewer lapses) -> 0, not negative


def sleep_debt_score(
    hours_slept_last_24: float,
    recommended: float,
    prior_debt: float,
    recovery_rate: float = 0.7,
    cap_hours: float = 12.0,
) -> tuple[float, float]:
    """
    Homeostatic sleep pressure, two-process-model style.
    Debt accumulates when sleep < recommended, partially recovers with
    good sleep (recovery_rate applied to prior debt each cycle before
    adding today's deficit). Returns (normalized_score, updated_debt_hours).
    """
    todays_deficit = max(0.0, recommended - hours_slept_last_24)
    carried = prior_debt * recovery_rate if hours_slept_last_24 >= recommended else prior_debt
    updated_debt = min(cap_hours, carried + todays_deficit)
    return _clip01(updated_debt / cap_hours), updated_debt


def circadian_score(minutes_phase_shift: float, max_tolerable_minutes: float) -> float:
    """
    How far the astronaut's rhythm has drifted from a stable 24h cycle,
    e.g. from cycling through 16 sunrise/sunsets a day plus shifted work
    schedules. Symmetric around 0 drift.
    """
    return _clip01(abs(minutes_phase_shift) / max_tolerable_minutes)


def workload_score(current_load: float, rolling_avg: float) -> float:
    """
    Task load relative to the astronaut's own baseline, not an absolute
    number (a 12-hour EVA day means something different for different
    crew roles). >2x rolling average -> saturates at 1.0.
    """
    rolling_avg = max(rolling_avg, 1.0)
    ratio = current_load / rolling_avg
    return _clip01((ratio - 1.0))  # at avg -> 0, at 2x avg -> 1.0


def classify_risk(score: float) -> Literal["nominal", "elevated", "high", "critical"]:
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
    weights.validate()

    rt = reaction_time_score(signals.pvt_lapses, signals.pvt_baseline_lapses)
    sleep, updated_debt = sleep_debt_score(
        signals.hours_slept_last_24,
        signals.recommended_sleep_hours,
        signals.prior_sleep_debt,
    )
    circ = circadian_score(signals.minutes_phase_shift, signals.max_tolerable_shift_minutes)
    work = workload_score(signals.current_task_load, signals.rolling_avg_task_load)

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


# Threshold at which the AI explanation layer (IBM Bob) gets invoked.
EXPLANATION_TRIGGER_THRESHOLD = 0.55  # "elevated" and above


if __name__ == "__main__":
    # quick smoke test
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
    print("Explain?", result.drift_score >= EXPLANATION_TRIGGER_THRESHOLD)