"""
api/mission.py — Mission risk routes.

Routes:
  POST /mission/risk          — calculate mission risk for a given moment
  POST /mission/project-risk  — project risk across multiple mission days
"""

from datetime import datetime, timezone
from fastapi import APIRouter
from pydantic import BaseModel

from app.models.mission import (
    Mission, MissionRiskResult,
    MissionProjectionInput, MissionProjectionResult, DayRiskSummary,
)
from app.models.astronaut import FatigueReading
from app.services.mission_risk_service import calculate_mission_risk
from app.engine.scoring import _classify_risk_level

router = APIRouter(prefix="/mission", tags=["Mission Risk"])


class MissionRiskRequest(BaseModel):
    mission: Mission
    fatigue_readings: list[FatigueReading] = []


@router.post(
    "/risk",
    response_model=MissionRiskResult,
    summary="Calculate Mission Risk Score",
    description=(
        "Calculates a deterministic mission risk score from astronaut fatigue "
        "readings and task definitions. Risk is NOT simply equal to fatigue — "
        "it accounts for task criticality, demand, and dependencies. "
        "**No AI is involved in this calculation.**"
    ),
)
def mission_risk(request: MissionRiskRequest) -> MissionRiskResult:
    return calculate_mission_risk(request.mission, request.fatigue_readings)


@router.post(
    "/project-risk",
    response_model=MissionProjectionResult,
    summary="Project Mission Risk Across Multiple Days",
    description=(
        "Projects mission risk across all days of a mission using provided "
        "fatigue readings and task schedule. "
        "**MODEL-BASED ESTIMATES ONLY** — not NASA-validated forecasts."
    ),
)
def project_risk(request: MissionProjectionInput) -> MissionProjectionResult:
    """
    For each mission day that has fatigue readings, calculate the mission risk.
    Days without readings get a warning and no score.
    """
    from app.models.task import Task
    from app.models.astronaut import Astronaut

    daily_summaries: list[DayRiskSummary] = []
    all_risk_scores: list[float] = []
    highest_risk_days: list[int] = []
    all_warnings: list[str] = []

    # Group readings by mission day
    days_with_readings: set[int] = {
        r.mission_day for r in request.fatigue_readings if r.mission_day is not None
    }

    for day in range(1, request.total_days + 1):
        day_readings = [r for r in request.fatigue_readings if r.mission_day == day]
        day_tasks = [t for t in request.tasks if t.mission_day == day or t.mission_day is None]

        if not day_readings:
            all_warnings.append(f"Day {day}: no fatigue readings — skipped in projection.")
            continue

        # Build a one-day Mission snapshot
        day_mission = Mission(
            mission_id=request.mission_id,
            mission_name=request.mission_name,
            mission_day=day,
            total_days=request.total_days,
            astronauts=request.astronauts,
            tasks=day_tasks,
        )

        day_result = calculate_mission_risk(day_mission, day_readings)
        all_risk_scores.append(day_result.mission_risk_score)

        if day_result.risk_level in ("HIGH", "CRITICAL"):
            highest_risk_days.append(day)

        daily_summaries.append(DayRiskSummary(
            mission_day=day,
            mission_risk_score=day_result.mission_risk_score,
            risk_level=day_result.risk_level,
            highest_risk_astronaut=day_result.highest_risk_astronaut,
            highest_risk_task=day_result.highest_risk_task,
            warnings=day_result.warnings,
        ))
        all_warnings.extend(day_result.warnings)

    # Overall projected trend
    if len(all_risk_scores) >= 2:
        delta = all_risk_scores[-1] - all_risk_scores[0]
        if delta >= 15:
            trend = "RAPIDLY_RISING"
        elif delta >= 5:
            trend = "RISING"
        elif delta <= -5:
            trend = "FALLING"
        else:
            trend = "STABLE"
    elif all_risk_scores:
        trend = "STABLE"
    else:
        trend = "UNKNOWN"

    return MissionProjectionResult(
        mission_id=request.mission_id,
        total_days=request.total_days,
        daily_summaries=daily_summaries,
        highest_risk_days=highest_risk_days,
        projected_fatigue_trend=trend,
        warnings=list(set(all_warnings)),
    )
