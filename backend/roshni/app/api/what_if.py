"""
api/what_if.py — POST /what-if/simulate route.

The real mission state is NEVER modified here.
All simulation runs on a deep copy inside what_if_service.
Feasibility is determined deterministically — AI does not decide.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.models.intervention import Intervention, WhatIfResult
from app.models.mission import Mission
from app.models.astronaut import FatigueReading
from app.services.what_if_service import simulate

router = APIRouter(prefix="/what-if", tags=["What-If Simulator"])


class WhatIfRequest(BaseModel):
    intervention: Intervention
    mission: Mission
    fatigue_readings: list[FatigueReading] = []


@router.post(
    "/simulate",
    response_model=WhatIfResult,
    summary="What-If Intervention Simulation",
    description=(
        "Simulates the effect of a proposed intervention (REASSIGN_TASK, "
        "DELAY_TASK, REDUCE_WORKLOAD) on fatigue and mission risk scores. "
        "**The real mission state is never modified.** "
        "**Feasibility is determined by deterministic rules — not AI.** "
        "AI can explain the result via POST /ai/explain-intervention."
    ),
)
def what_if_simulate(request: WhatIfRequest) -> WhatIfResult:
    return simulate(request.intervention, request.mission, request.fatigue_readings)
