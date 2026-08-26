"""
models/intervention.py — Intervention and What-If simulation models.
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field


InterventionType = Literal["REASSIGN_TASK", "DELAY_TASK", "REDUCE_WORKLOAD"]


class Intervention(BaseModel):
    """A proposed operational intervention to evaluate in the What-If simulator."""

    intervention_type: InterventionType = Field(
        ...,
        description="Type of intervention: REASSIGN_TASK | DELAY_TASK | REDUCE_WORKLOAD",
    )
    task_id: str = Field(..., description="ID of the task to intervene on.")

    # REASSIGN_TASK
    from_astronaut_id: Optional[str] = Field(None, description="Current astronaut (for REASSIGN_TASK).")
    to_astronaut_id: Optional[str] = Field(None, description="Target astronaut (for REASSIGN_TASK).")

    # DELAY_TASK
    delay_minutes: Optional[int] = Field(None, gt=0, description="Minutes to delay task start (for DELAY_TASK).")

    # REDUCE_WORKLOAD
    workload_reduction: Optional[float] = Field(
        None, gt=0, le=100,
        description="Amount to reduce workload_risk by (for REDUCE_WORKLOAD).",
    )


class WhatIfResult(BaseModel):
    """
    Result of a What-If simulation.

    The deterministic engine calculates both before/after scores.
    AI explains the result. Humans decide.
    The real mission state is NEVER modified by this endpoint.
    """

    before_fatigue_score: float
    after_fatigue_score: float
    before_mission_risk: float
    after_mission_risk: float
    risk_change: float = Field(description="after_mission_risk − before_mission_risk (negative = improvement).")

    intervention: Intervention
    feasible: bool = Field(description="Whether the intervention passes all constraint checks.")
    constraint_violations: list[str] = Field(
        default_factory=list,
        description="Human-readable list of violated constraints (empty if feasible).",
    )

    # Data fed into AI explanation layer
    explanation_data: dict = Field(
        default_factory=dict,
        description="Structured data passed to IBM AI for explanation (read-only).",
    )

    disclaimer: str = Field(
        default=(
            "This simulation result does not modify the real mission state. "
            "It is a model-based estimate for decision support. "
            "Human mission personnel must review and authorize any changes."
        )
    )
