"""
models/astronaut.py — Astronaut and FatigueReading data models.
"""
from datetime import datetime, time
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class Astronaut(BaseModel):
    """Profile of an astronaut participating in a mission."""

    astronaut_id: str = Field(..., min_length=1, max_length=64, examples=["A01"])
    name: str = Field(..., min_length=1, max_length=128, examples=["Dr. Elena Vasquez"])
    role: str = Field(..., examples=["Commander"])
    qualifications: list[str] = Field(
        default_factory=list,
        description="List of qualification codes this astronaut holds.",
        examples=[["EVA", "ROBOTICS", "MEDICAL"]],
    )

    # Baselines — used by data-quality scoring and trend analysis
    baseline_pvt_ms: Optional[float] = Field(
        None, ge=100, le=600,
        description="Individual baseline PVT reaction time in milliseconds.",
    )
    baseline_sleep_hours: Optional[float] = Field(
        None, ge=0, le=12,
        description="Individual baseline nightly sleep duration (hours).",
    )
    baseline_workload: Optional[float] = Field(
        None, ge=0, le=100,
        description="Individual baseline workload score (0–100).",
    )
    baseline_sleep_time: Optional[str] = Field(
        None,
        description="Typical sleep time as HH:MM string, e.g. '22:00'.",
        examples=["22:00"],
    )
    baseline_wake_time: Optional[str] = Field(
        None,
        description="Typical wake time as HH:MM string, e.g. '06:00'.",
        examples=["06:00"],
    )

    @field_validator("astronaut_id", "name", "role")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be blank.")
        return v.strip()


class FatigueReading(BaseModel):
    """
    A single set of raw risk-signal readings for one astronaut at one moment.
    This is the input to the fatigue scoring engine.
    """

    astronaut_id: str = Field(..., min_length=1, max_length=64)
    timestamp: datetime = Field(..., description="ISO-8601 UTC timestamp of the reading.")
    mission_day: Optional[int] = Field(None, ge=1, description="Mission day number.")

    pvt_risk: float = Field(..., ge=0, le=100, description="PVT/reaction-time risk (0–100).")
    sleep_risk: float = Field(..., ge=0, le=100, description="Sleep quality/duration risk (0–100).")
    circadian_risk: float = Field(..., ge=0, le=100, description="Circadian-alignment risk (0–100).")
    workload_risk: float = Field(..., ge=0, le=100, description="Workload risk (0–100).")

    @field_validator("astronaut_id")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("astronaut_id must not be blank.")
        return v.strip()
