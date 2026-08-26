"""
api/fatigue.py — POST /fatigue/calculate route.

Thin route: validates input, delegates to fatigue_service, returns result.
No business logic lives here.
"""

from fastapi import APIRouter
from app.models.fatigue import FatigueInput, FatigueResult
from app.services.fatigue_service import assess_fatigue

router = APIRouter(prefix="/fatigue", tags=["Fatigue Assessment"])


@router.post(
    "/calculate",
    response_model=FatigueResult,
    summary="Calculate Fatigue Drift Score",
    description=(
        "Receives astronaut risk signals and returns a deterministic "
        "Fatigue Drift Score with per-signal breakdown, risk level, trend, "
        "and data quality confidence rating. "
        "**No AI or LLM is involved in the score calculation.** "
        "This is a decision-support tool — all final operational decisions "
        "must remain with qualified human personnel."
    ),
)
def calculate(payload: FatigueInput) -> FatigueResult:
    return assess_fatigue(payload)
