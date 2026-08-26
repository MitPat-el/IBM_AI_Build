"""
models/mission.py — Mission, MissionRiskResult, and projection models.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from app.models.astronaut import Astronaut, FatigueReading
from app.models.task import Task


class Mission(BaseModel):
    """A complete mission definition used as input to risk calculations."""

    mission_id: str = Field(..., min_length=1, max_length=64, examples=["ARTEMIS-DEMO-01"])
    mission_name: str = Field(..., examples=["Artemis Demo Mission 1"])
    mission_day: int = Field(..., ge=1, description="Current mission day being evaluated.")
    total_days: int = Field(..., ge=1, description="Total planned mission duration in days.")
    astronauts: list[Astronaut] = Field(..., min_length=1)
    tasks: list[Task] = Field(default_factory=list)


class TaskRiskDetail(BaseModel):
    """Risk breakdown for a single task/astronaut combination."""
    task_id: str
    task_name: str
    astronaut_id: str
    fatigue_score: float
    task_demand_score: float
    mission_task_risk: float
    risk_level: str
    reasons: list[str]


class MissionRiskResult(BaseModel):
    """
    Result of a mission risk calculation.

    The deterministic model calculates risk. IBM AI explains the result.
    Human mission personnel make the decision.
    NOTE: Results are model-based estimates, not NASA-validated operational outputs.
    """

    mission_id: str
    timestamp: datetime
    mission_risk_score: float = Field(description="Overall mission risk score (0–100).")
    risk_level: str = Field(description="LOW | MODERATE | HIGH | CRITICAL")

    highest_risk_astronaut: Optional[str] = Field(None, description="astronaut_id with highest task risk.")
    highest_risk_task: Optional[str] = Field(None, description="task_id with highest mission risk.")

    task_risk_details: list[TaskRiskDetail] = Field(default_factory=list)
    contributing_factors: list[str] = Field(description="Human-readable reasons for the risk level.")
    warnings: list[str] = Field(default_factory=list, description="Specific operational warnings.")

    data_quality_score: float = Field(default=100.0, description="Confidence in input data (0–100).")
    data_quality_level: str = Field(default="HIGH")

    disclaimer: str = Field(
        default=(
            "This mission risk score is a model-based estimate for decision support only. "
            "It is NOT a NASA-validated operational output. "
            "All final decisions must be made by qualified human personnel."
        )
    )


class MissionProjectionInput(BaseModel):
    """Input for a multi-day mission risk projection."""
    mission_id: str
    mission_name: str
    total_days: int = Field(..., ge=1, le=365)
    astronauts: list[Astronaut]
    tasks: list[Task] = Field(default_factory=list)
    # Per-day fatigue readings keyed by "mission_day:astronaut_id"
    fatigue_readings: list[FatigueReading] = Field(default_factory=list)


class DayRiskSummary(BaseModel):
    """Risk summary for one mission day."""
    mission_day: int
    mission_risk_score: float
    risk_level: str
    highest_risk_astronaut: Optional[str] = None
    highest_risk_task: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


class MissionProjectionResult(BaseModel):
    """
    Multi-day mission risk projection.
    Clearly labelled as MODEL-BASED ESTIMATES.
    """
    mission_id: str
    total_days: int
    daily_summaries: list[DayRiskSummary]
    highest_risk_days: list[int] = Field(description="Mission days with HIGH or CRITICAL risk.")
    projected_fatigue_trend: str = Field(description="Overall crew fatigue trend across the mission.")
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = Field(
        default=(
            "PROTOTYPE MODEL-BASED ESTIMATES ONLY. "
            "These projections are not NASA-validated operational forecasts. "
            "Human mission personnel must review and decide."
        )
    )
