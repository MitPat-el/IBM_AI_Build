"""
models/fatigue.py — Pydantic schemas for the fatigue calculation endpoint.

Defines:
  - FatigueInput   : validated request body for POST /fatigue/calculate
  - SignalBreakdown: per-signal contribution in the response
  - FatigueResult  : full response body (includes trend + data quality)
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.core.config import MIN_RISK_VALUE, MAX_RISK_VALUE


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class FatigueInput(BaseModel):
    """
    Incoming payload for a fatigue assessment request.

    All risk values must be in the range [0, 100].
    These are NOT medical diagnoses — they are mission-planning risk signals.
    """

    astronaut_id: str = Field(
        ..., min_length=1, max_length=64,
        description="Unique identifier for the astronaut.",
        examples=["CDR-001"],
    )
    timestamp: datetime = Field(
        ...,
        description="ISO-8601 UTC timestamp of the measurement.",
        examples=["2025-01-15T14:30:00Z"],
    )
    pvt_risk: float = Field(..., ge=MIN_RISK_VALUE, le=MAX_RISK_VALUE,
                            description="PVT / reaction-time risk score (0–100).")
    sleep_risk: float = Field(..., ge=MIN_RISK_VALUE, le=MAX_RISK_VALUE,
                              description="Sleep quality/duration risk score (0–100).")
    circadian_risk: float = Field(..., ge=MIN_RISK_VALUE, le=MAX_RISK_VALUE,
                                  description="Circadian-alignment risk score (0–100).")
    workload_risk: float = Field(..., ge=MIN_RISK_VALUE, le=MAX_RISK_VALUE,
                                 description="Workload risk score (0–100).")

    # Optional context fields (used for trend and data quality)
    previous_fatigue_score: Optional[float] = Field(
        None, ge=0, le=100,
        description="Most recent prior Fatigue Drift Score for this astronaut (enables trend).",
    )
    mission_day: Optional[int] = Field(None, ge=1, description="Current mission day.")
    baseline_available: bool = Field(
        False, description="Whether an individual astronaut baseline is on record."
    )
    task_info_available: bool = Field(
        False, description="Whether current task assignment data is available."
    )

    @field_validator("astronaut_id")
    @classmethod
    def astronaut_id_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("astronaut_id must not be blank or whitespace only.")
        return value.strip()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class SignalBreakdown(BaseModel):
    """Shows how much a single risk signal contributed to the final score."""
    signal: str
    raw_value: float
    weight: float
    weighted_contribution: float


class DataQuality(BaseModel):
    """Confidence rating for the fatigue assessment based on data completeness."""
    score: float = Field(description="Data quality score 0–100.")
    level: str = Field(description="HIGH | MEDIUM | LOW")
    notes: list[str] = Field(default_factory=list,
                             description="Specific quality issues identified.")


class FatigueResult(BaseModel):
    """
    Full response returned by POST /fatigue/calculate.

    DISCLAIMER: This score is a mission-planning decision-support tool.
    It is NOT a medical diagnosis. Final operational decisions must remain
    with qualified human personnel.
    """

    astronaut_id: str
    timestamp: datetime
    mission_day: Optional[int] = None

    fatigue_score: float = Field(description="Fatigue Drift Score — deterministic weighted sum (0–100).")
    risk_level: str = Field(description="LOW | MODERATE | HIGH | CRITICAL")

    signal_breakdown: list[SignalBreakdown]
    top_contributing_factors: list[str] = Field(
        description="Top 3 signal names ranked by weighted contribution (highest first)."
    )

    trend: str = Field(
        default="UNKNOWN",
        description="STABLE | RISING | RAPIDLY_RISING | FALLING | UNKNOWN (if no prior reading).",
    )
    data_quality: DataQuality = Field(
        default_factory=lambda: DataQuality(score=100.0, level="HIGH"),
        description="Confidence rating for this assessment.",
    )

    disclaimer: str = Field(
        default=(
            "This score is a decision-support tool for mission planning only. "
            "It is NOT a medical diagnosis. All final decisions must be made "
            "by qualified human personnel."
        )
    )
