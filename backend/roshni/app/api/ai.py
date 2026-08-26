"""
api/ai.py — IBM watsonx AI explanation routes.

Routes:
  POST /ai/explain              — explain a FatigueResult (audience-aware)
  POST /ai/explain-intervention — explain a What-If simulation result

DESIGN CONTRACT:
  - AI receives only structured backend results (never raw sensor data).
  - AI never recalculates or overrides any score.
  - AI never decides feasibility of interventions.
  - Every response carries a human-review disclaimer.
"""

from typing import Literal
from fastapi import APIRouter
from pydantic import BaseModel

from app.models.fatigue import FatigueResult
from app.models.intervention import WhatIfResult
from app.services.ai_service import explain_fatigue_result, explain_intervention_result

router = APIRouter(prefix="/ai", tags=["AI Explanation (IBM watsonx)"])

AudienceMode = Literal["ASTRONAUT", "MISSION_TEAM", "FLIGHT_SURGEON"]


# ---------------------------------------------------------------------------
# /ai/explain
# ---------------------------------------------------------------------------

class ExplainRequest(BaseModel):
    result: FatigueResult
    audience: AudienceMode = "MISSION_TEAM"
    task_context: str = "No task context provided."


class ExplainResponse(BaseModel):
    astronaut_id: str
    fatigue_score: float
    risk_level: str
    audience: str
    explanation: str
    disclaimer: str = (
        "This AI-generated explanation is for informational purposes only. "
        "It does NOT constitute a medical diagnosis. "
        "Final decisions must be made by qualified human personnel."
    )


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="AI Explanation of Fatigue Score (IBM watsonx)",
    description=(
        "Accepts a computed FatigueResult and returns an audience-aware "
        "natural-language explanation from IBM watsonx.ai. "
        "**The AI does not recalculate or override the score.** "
        "Audience modes: ASTRONAUT | MISSION_TEAM | FLIGHT_SURGEON."
    ),
)
def explain(request: ExplainRequest) -> ExplainResponse:
    result_dict = request.result.model_dump()
    result_dict["task_context"] = request.task_context
    explanation_text = explain_fatigue_result(result_dict, request.audience)
    return ExplainResponse(
        astronaut_id=request.result.astronaut_id,
        fatigue_score=request.result.fatigue_score,
        risk_level=request.result.risk_level,
        audience=request.audience,
        explanation=explanation_text,
    )


# ---------------------------------------------------------------------------
# /ai/explain-intervention
# ---------------------------------------------------------------------------

class InterventionExplainRequest(BaseModel):
    what_if_result: WhatIfResult
    top_factors: list[str] = []


class InterventionExplainResponse(BaseModel):
    summary: str
    why_it_helps: str
    limitations: str
    human_review_required: bool = True
    disclaimer: str = (
        "This AI-generated explanation is for informational purposes only. "
        "The AI does not decide whether the intervention is authorized. "
        "Human mission personnel must review and authorize any changes."
    )


@router.post(
    "/explain-intervention",
    response_model=InterventionExplainResponse,
    summary="AI Explanation of What-If Intervention Result",
    description=(
        "Accepts a WhatIfResult and returns a natural-language explanation of "
        "why the intervention does or doesn't help. "
        "**The AI does not decide feasibility — only explains the deterministic result.**"
    ),
)
def explain_intervention(request: InterventionExplainRequest) -> InterventionExplainResponse:
    data = request.what_if_result.explanation_data.copy()
    data["top_factors"] = ", ".join(request.top_factors) if request.top_factors else "Not provided"
    result = explain_intervention_result(data)
    return InterventionExplainResponse(**result)
