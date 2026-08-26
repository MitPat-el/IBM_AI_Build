"""
models/task.py — Task data model.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Task(BaseModel):
    """A scheduled mission task."""

    task_id: str = Field(..., min_length=1, max_length=64, examples=["T-EVA-01"])
    name: str = Field(..., min_length=1, max_length=256, examples=["EVA Solar Panel Repair"])
    assigned_astronaut_id: str = Field(..., description="ID of the astronaut currently assigned.")
    start_time: datetime = Field(..., description="Scheduled UTC start time.")
    duration_minutes: int = Field(..., gt=0, description="Expected duration in minutes.")

    # Demand dimensions — all 1–5
    criticality: int = Field(..., ge=1, le=5, description="Mission criticality (1=low, 5=critical).")
    cognitive_demand: int = Field(..., ge=1, le=5, description="Cognitive demand (1–5).")
    physical_demand: int = Field(..., ge=1, le=5, description="Physical demand (1–5).")

    required_qualifications: list[str] = Field(
        default_factory=list,
        description="Qualification codes required to perform this task.",
    )
    can_delay: bool = Field(True, description="Whether the task start time can be delayed.")
    can_reassign: bool = Field(True, description="Whether the task can be reassigned to another crew member.")
    dependencies: list[str] = Field(
        default_factory=list,
        description="task_ids that must complete before this task can start.",
    )
    mission_day: Optional[int] = Field(None, ge=1, description="Mission day this task is scheduled on.")
