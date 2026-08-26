"""
services/fatigue_service.py — Fatigue assessment service layer.

Wraps the deterministic scoring engine and adds:
  - Trend calculation (STABLE / RISING / RAPIDLY_RISING / FALLING / UNKNOWN)
  - Data quality scoring
  - Top-3 contributing factors (not all four)

The scoring engine (engine/scoring.py) is never modified.
This service layer calls it and enriches its output.

RULE: This service must never call an LLM or external service.
"""

from app.models.fatigue import FatigueInput, FatigueResult
from app.engine.scoring import calculate_fatigue_score
from app.services.data_quality_service import score_data_quality
from app.core.config import TREND_THRESHOLDS


def _calculate_trend(current_score: float, previous_score: float | None) -> str:
    """
    Classify the fatigue trend by comparing current vs previous score.

    Returns:
        RAPIDLY_RISING | RISING | FALLING | STABLE | UNKNOWN
    """
    if previous_score is None:
        return "UNKNOWN"

    delta = current_score - previous_score

    if delta >= TREND_THRESHOLDS["RAPIDLY_RISING"]:
        return "RAPIDLY_RISING"
    if delta >= TREND_THRESHOLDS["RISING"]:
        return "RISING"
    if delta <= TREND_THRESHOLDS["FALLING"]:
        return "FALLING"
    return "STABLE"


def assess_fatigue(payload: FatigueInput) -> FatigueResult:
    """
    Full fatigue assessment pipeline.

    1. Call deterministic scoring engine → FatigueResult (score + breakdown)
    2. Calculate trend from previous_fatigue_score (if provided)
    3. Score data quality
    4. Limit top_contributing_factors to top 3
    5. Return enriched FatigueResult

    Args:
        payload: Validated FatigueInput from the API layer.

    Returns:
        FatigueResult with score, level, breakdown, trend, and data quality.
    """
    # Step 1 — deterministic score (engine is unchanged)
    result = calculate_fatigue_score(payload)

    # Step 2 — trend
    trend = _calculate_trend(result.fatigue_score, payload.previous_fatigue_score)

    # Step 3 — data quality
    data_quality = score_data_quality(payload)

    # Step 4 — top 3 only
    top_3 = result.top_contributing_factors[:3]

    # Step 5 — return enriched result
    return FatigueResult(
        astronaut_id=result.astronaut_id,
        timestamp=result.timestamp,
        mission_day=payload.mission_day,
        fatigue_score=result.fatigue_score,
        risk_level=result.risk_level,
        signal_breakdown=result.signal_breakdown,
        top_contributing_factors=top_3,
        trend=trend,
        data_quality=data_quality,
    )
