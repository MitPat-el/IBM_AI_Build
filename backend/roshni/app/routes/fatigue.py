"""
routes/fatigue.py — API routes for the /fatigue module.

Routes defined here:
  POST /fatigue/calculate  — submit raw risk signals, receive Fatigue Drift Score

Future routes to add in this file (or a new routes/ file):
  POST /fatigue/whatif     — What-If Simulator
  GET  /fatigue/history    — Historical Replay
  POST /fatigue/mission    — Mission Risk Projection
  POST /fatigue/explain    — IBM watsonx AI explanation of an existing score
"""

from fastapi import APIRouter
from app.models.fatigue import FatigueInput, FatigueResult
from app.engine.scoring import calculate_fatigue_score

router = APIRouter(prefix="/fatigue", tags=["Fatigue Assessment"])


@router.post(
    "/calculate",
    response_model=FatigueResult,
    summary="Calculate Fatigue Drift Score",
    description=(
        "Receives astronaut risk signals and returns a deterministic "
        "Fatigue Drift Score with per-signal breakdown and risk level. "
        "**No AI or LLM is involved in the score calculation.** "
        "This is a decision-support tool — all final operational decisions "
        "must remain with qualified human personnel."
    ),
)
def calculate(payload: FatigueInput) -> FatigueResult:
    """
    POST /fatigue/calculate

    Validates all inputs via Pydantic, then delegates to the deterministic
    scoring engine. The route layer contains no business logic.
    """
    return calculate_fatigue_score(payload)
