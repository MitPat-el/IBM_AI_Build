"""
services/data_quality_service.py — Data quality / confidence scoring.

Evaluates how complete and fresh the input data is, and returns a
DataQuality object that accompanies every FatigueResult.

The system never invents missing data. Instead it reduces the confidence
score and adds a note explaining what is missing.

RULE: If key data is missing, reduce confidence — do NOT invent values.
"""

from datetime import datetime, timezone
from app.models.fatigue import FatigueInput, DataQuality
from app.core.config import DATA_FRESHNESS_THRESHOLD_SECONDS


def score_data_quality(payload: FatigueInput) -> DataQuality:
    """
    Compute a DataQuality score for a FatigueInput.

    Factors evaluated:
      1. All four fatigue signals present (always true if Pydantic validated)
      2. Freshness of the reading (age vs DATA_FRESHNESS_THRESHOLD_SECONDS)
      3. Astronaut baseline is available
      4. Task information is available
      5. Previous score available (enables trend)

    Args:
        payload: Validated FatigueInput

    Returns:
        DataQuality with score 0–100, level, and human-readable notes.
    """
    score = 100.0
    notes: list[str] = []

    # Factor 1: all four signals present — guaranteed by Pydantic validation,
    # but we still check for None defensively.
    signals = [payload.pvt_risk, payload.sleep_risk, payload.circadian_risk, payload.workload_risk]
    missing_signals = sum(1 for s in signals if s is None)
    if missing_signals:
        deduction = missing_signals * 20.0
        score -= deduction
        notes.append(f"{missing_signals} fatigue signal(s) missing — score confidence reduced.")

    # Factor 2: freshness
    now_utc = datetime.now(timezone.utc)
    reading_time = payload.timestamp
    if reading_time.tzinfo is None:
        reading_time = reading_time.replace(tzinfo=timezone.utc)
    age_seconds = (now_utc - reading_time).total_seconds()
    if age_seconds > DATA_FRESHNESS_THRESHOLD_SECONDS:
        hours_old = round(age_seconds / 3600, 1)
        score -= 20.0
        notes.append(
            f"Reading is {hours_old}h old (threshold: "
            f"{DATA_FRESHNESS_THRESHOLD_SECONDS // 3600}h). Freshness penalty applied."
        )

    # Factor 3: baseline available
    if not payload.baseline_available:
        score -= 10.0
        notes.append("No individual astronaut baseline on record — population norms used.")

    # Factor 4: task information available
    if not payload.task_info_available:
        score -= 10.0
        notes.append("Task assignment data unavailable — workload context is limited.")

    # Factor 5: previous score available (trend)
    if payload.previous_fatigue_score is None:
        score -= 5.0
        notes.append("No prior fatigue score available — trend cannot be calculated.")

    score = max(0.0, round(score, 1))

    if score >= 80:
        level = "HIGH"
    elif score >= 50:
        level = "MEDIUM"
    else:
        level = "LOW"
        notes.append(
            "Risk score is present but confidence is limited — "
            "review missing data before acting on this assessment."
        )

    return DataQuality(score=score, level=level, notes=notes)
