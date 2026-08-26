"""
engine/scoring.py — Deterministic Fatigue Drift Score calculation engine.

THIS MODULE MUST NEVER CALL AN LLM OR EXTERNAL SERVICE.
The score is computed with a transparent weighted-sum formula so it is fully
auditable and reproducible. All weights/thresholds come from config.py.

Formula:
    Fatigue Drift Score = Σ (signal_value × signal_weight)

Extending this module:
  - To add a new signal: add its weight to config.SIGNAL_WEIGHTS (keeping the
    sum at 1.0), then add its key to the signals dict in `calculate()`.
  - To change thresholds: edit config.RISK_LEVELS only.
  - DO NOT add I/O, database calls, or AI inference here.
"""

from app.config import SIGNAL_WEIGHTS, RISK_LEVELS
from app.models.fatigue import FatigueInput, FatigueResult, SignalBreakdown


def _classify_risk_level(score: float) -> str:
    """
    Map a numeric Fatigue Drift Score to a categorical risk level.

    Thresholds are read from config.RISK_LEVELS so they can be updated
    without touching this function.
    """
    for low, high, label in RISK_LEVELS:
        if low <= score <= high:
            return label
    # Clamp edge-cases (should not occur given validated 0–100 input)
    return "LOW" if score < 40 else "CRITICAL"


def _build_breakdown(signals: dict[str, float]) -> list[SignalBreakdown]:
    """
    Build per-signal breakdown objects: raw value, weight, and contribution.

    Args:
        signals: mapping of signal_name → raw_value (0–100)

    Returns:
        List of SignalBreakdown instances, one per signal.
    """
    breakdown: list[SignalBreakdown] = []
    for signal_name, raw_value in signals.items():
        weight = SIGNAL_WEIGHTS[signal_name]
        contribution = round(raw_value * weight, 4)
        breakdown.append(
            SignalBreakdown(
                signal=signal_name,
                raw_value=raw_value,
                weight=weight,
                weighted_contribution=contribution,
            )
        )
    return breakdown


def calculate_fatigue_score(payload: FatigueInput) -> FatigueResult:
    """
    Core scoring function — the only place where the Fatigue Drift Score
    is computed.

    Steps:
      1. Extract signal values from the validated input.
      2. Multiply each signal by its configured weight.
      3. Sum all weighted contributions → Fatigue Drift Score.
      4. Classify the score into a risk level.
      5. Rank signals by contribution (highest first) for explainability.
      6. Return a structured FatigueResult.

    Args:
        payload: Validated FatigueInput from the API layer.

    Returns:
        FatigueResult with score, risk level, breakdown, and top factors.
    """
    # Step 1 — collect signals in the same key order as SIGNAL_WEIGHTS
    signals: dict[str, float] = {
        "pvt_risk":       payload.pvt_risk,
        "sleep_risk":     payload.sleep_risk,
        "circadian_risk": payload.circadian_risk,
        "workload_risk":  payload.workload_risk,
    }

    # Step 2 & 3 — weighted sum
    breakdown = _build_breakdown(signals)
    fatigue_score = round(sum(b.weighted_contribution for b in breakdown), 2)

    # Step 4 — risk classification
    risk_level = _classify_risk_level(fatigue_score)

    # Step 5 — rank signals by contribution for top-factor guidance
    ranked = sorted(breakdown, key=lambda b: b.weighted_contribution, reverse=True)
    top_factors = [b.signal for b in ranked]

    # Step 6 — assemble response
    return FatigueResult(
        astronaut_id=payload.astronaut_id,
        timestamp=payload.timestamp,
        fatigue_score=fatigue_score,
        risk_level=risk_level,
        signal_breakdown=breakdown,
        top_contributing_factors=top_factors,
    )
