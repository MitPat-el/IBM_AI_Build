"""
services/ai_service.py — IBM watsonx AI explanation service.

DESIGN CONTRACT:
  - This service READS structured backend results.
  - It NEVER recalculates or overrides any score.
  - AI is used only to EXPLAIN, never to DECIDE.
  - Three audience modes: ASTRONAUT | MISSION_TEAM | FLIGHT_SURGEON
  - The human operator always has final authority.

TO ACTIVATE IBM WATSONX:
  1. pip install ibm-watsonx-ai
  2. Set env vars: WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL
  3. Uncomment the real implementation block in _call_watsonx().

AI RULES (enforced structurally):
  - Never change calculated scores.
  - Never invent sensor values.
  - Never diagnose.
  - Never issue mandatory medical instructions.
  - Clearly say when data is missing.
  - Keep humans responsible for final decisions.
  - Use only information supplied by the backend.
"""

import os
from typing import Literal

AudienceMode = Literal["ASTRONAUT", "MISSION_TEAM", "FLIGHT_SURGEON"]

# ---------------------------------------------------------------------------
# Prompt templates — one per audience mode
# Adjust tone/length here without touching business logic.
# ---------------------------------------------------------------------------

_FATIGUE_PROMPTS: dict[str, str] = {
    "ASTRONAUT": """
You are a crew support assistant. Speak directly to the astronaut.
Use simple, calm, actionable language. Do not diagnose. Maximum 3 sentences.

Astronaut: {astronaut_id}
Fatigue Drift Score: {fatigue_score}/100  |  Risk Level: {risk_level}
Trend: {trend}
Top contributing factors: {top_factors}
Data confidence: {data_quality_level} ({data_quality_score}/100)

Briefly explain what this score means for them right now, and suggest one
practical step they can take. End with: "Your mission team has been notified."
Do NOT state a medical diagnosis. Do NOT invent any data not shown above.
""".strip(),

    "MISSION_TEAM": """
You are a mission support analyst. Address the flight control team.
Be concise and operationally focused. Include task context if provided.

Astronaut: {astronaut_id}
Fatigue Drift Score: {fatigue_score}/100  |  Risk Level: {risk_level}
Trend: {trend}
Signal breakdown: {signal_lines}
Top contributing factors: {top_factors}
Task context: {task_context}
Data confidence: {data_quality_level} ({data_quality_score}/100)

1. Explain the operational significance of this score in 2–3 sentences.
2. For each top factor, state what it indicates operationally.
3. Suggest one intervention option for the team to consider.
4. State clearly: "Final decision rests with qualified human personnel."
Do NOT diagnose. Do NOT invent data.
""".strip(),

    "FLIGHT_SURGEON": """
You are a flight surgeon support tool. Provide a clinically objective summary
for medical review. Use numerical evidence. Do not diagnose.

Astronaut: {astronaut_id}  |  Timestamp: {timestamp}
Fatigue Drift Score: {fatigue_score}/100  |  Risk Level: {risk_level}
Trend: {trend}
Signal breakdown (raw → weighted contribution):
{signal_lines}
Top contributing factors: {top_factors}
Data confidence: {data_quality_level} ({data_quality_score}/100)
{quality_notes}

Provide:
1. A 2–3 sentence objective summary of the significant signals.
2. Which signal(s) most warrant attention and why.
3. One evidence-based countermeasure per high-contribution signal.
4. Note any data quality limitations that affect confidence.
5. End: "This is a decision-support summary. Clinical judgment and final
   authority remain with the flight surgeon."
Do NOT state a diagnosis. Do NOT invent any values.
""".strip(),
}

_INTERVENTION_PROMPT = """
You are a mission operations analyst. Explain the result of a What-If simulation
to the mission team. Do NOT decide whether to implement the intervention.

Intervention type: {intervention_type}
Task: {task_name} (ID: {task_id})
Before mission risk: {before_mission_risk}/100 ({before_risk_level})
After mission risk:  {after_mission_risk}/100 ({after_risk_level})
Risk change: {risk_change}
Feasible: {feasible}
Constraint violations: {violations}
Contributing signals: {contributing_signals}

Provide:
1. summary: One sentence summarizing what changed.
2. why_it_helps: 1–2 sentences on why this reduces (or doesn't reduce) risk.
3. limitations: Any caveats or remaining risks after the intervention.
4. Conclude: "Human mission personnel must review and authorize any changes."
Do NOT approve or reject the intervention. Do NOT invent data.
""".strip()


# ---------------------------------------------------------------------------
# Internal watsonx call
# ---------------------------------------------------------------------------

def _call_watsonx(prompt: str) -> str:
    """
    Send a prompt to IBM watsonx.ai and return the generated text.

    Returns a stub message if credentials are not configured.
    """
    api_key    = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

    if not api_key or not project_id:
        return (
            "[IBM watsonx explanation — credentials not configured. "
            "Set WATSONX_API_KEY and WATSONX_PROJECT_ID to enable live explanations.]\n\n"
            f"PROMPT THAT WOULD BE SENT:\n{prompt}"
        )

    # --- REAL IMPLEMENTATION ---
    # Uncomment when ibm-watsonx-ai is installed and credentials are set.
    #
    # from ibm_watsonx_ai import Credentials
    # from ibm_watsonx_ai.foundation_models import ModelInference
    #
    # credentials = Credentials(url=url, api_key=api_key)
    # model = ModelInference(
    #     model_id="ibm/granite-13b-instruct-v2",
    #     credentials=credentials,
    #     project_id=project_id,
    #     params={"max_new_tokens": 400, "temperature": 0.2},
    # )
    # return model.generate_text(prompt=prompt)

    return "[IBM watsonx stub — implementation pending after credentials are configured.]"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_fatigue_result(result_dict: dict, audience: AudienceMode = "MISSION_TEAM") -> str:
    """
    Generate a natural-language explanation of a FatigueResult.

    Args:
        result_dict: Serialized FatigueResult (from .model_dump()).
        audience:    ASTRONAUT | MISSION_TEAM | FLIGHT_SURGEON

    Returns:
        Explanation string from IBM watsonx (or stub).

    NOTE: This function only explains an existing result.
          It does NOT recalculate or modify any score.
    """
    template = _FATIGUE_PROMPTS.get(audience, _FATIGUE_PROMPTS["MISSION_TEAM"])

    signal_breakdown = result_dict.get("signal_breakdown", [])
    signal_lines = "\n".join(
        f"  - {b['signal']}: {b['raw_value']} → {b['weighted_contribution']} (weight {b['weight']})"
        for b in signal_breakdown
    )

    data_quality = result_dict.get("data_quality", {})
    quality_notes_raw = data_quality.get("notes", [])
    quality_notes = "\n".join(f"  * {n}" for n in quality_notes_raw) if quality_notes_raw else "None."

    task_context = result_dict.get("task_context", "No task context provided.")

    prompt = template.format(
        astronaut_id=result_dict.get("astronaut_id", "UNKNOWN"),
        timestamp=str(result_dict.get("timestamp", "UNKNOWN")),
        fatigue_score=result_dict.get("fatigue_score", "N/A"),
        risk_level=result_dict.get("risk_level", "N/A"),
        trend=result_dict.get("trend", "UNKNOWN"),
        signal_lines=signal_lines,
        top_factors=", ".join(result_dict.get("top_contributing_factors", [])),
        data_quality_level=data_quality.get("level", "UNKNOWN"),
        data_quality_score=data_quality.get("score", "N/A"),
        quality_notes=quality_notes,
        task_context=task_context,
    )

    return _call_watsonx(prompt)


def explain_intervention_result(what_if_data: dict) -> dict:
    """
    Generate a structured explanation of a What-If simulation result.

    Args:
        what_if_data: Serialized WhatIfResult.explanation_data dict.

    Returns:
        dict with keys: summary, why_it_helps, limitations, human_review_required.

    NOTE: AI explains the deterministic result. It does NOT decide feasibility.
    """
    from app.engine.scoring import _classify_risk_level

    before_risk = what_if_data.get("before_mission_risk", 0)
    after_risk  = what_if_data.get("after_mission_risk", 0)

    prompt = _INTERVENTION_PROMPT.format(
        intervention_type=what_if_data.get("intervention_type", "UNKNOWN"),
        task_name=what_if_data.get("task_name", "UNKNOWN"),
        task_id=what_if_data.get("task_id", "UNKNOWN"),
        before_mission_risk=before_risk,
        before_risk_level=_classify_risk_level(before_risk),
        after_mission_risk=after_risk,
        after_risk_level=_classify_risk_level(after_risk),
        risk_change=what_if_data.get("risk_change", 0),
        feasible=what_if_data.get("feasible", False),
        violations=what_if_data.get("constraint_violations", []) or "None",
        contributing_signals=what_if_data.get("top_factors", "Not provided"),
    )

    raw_explanation = _call_watsonx(prompt)

    return {
        "summary": raw_explanation,
        "why_it_helps": "(See summary above — structured parsing available after watsonx activation.)",
        "limitations": (
            "This is a model-based simulation. Real mission constraints may differ."
        ),
        "human_review_required": True,
    }
