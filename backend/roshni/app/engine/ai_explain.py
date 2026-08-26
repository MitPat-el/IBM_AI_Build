"""
engine/ai_explain.py — IBM watsonx AI explanation layer (STUB).

PURPOSE:
  This module will provide natural-language explanations of an already-computed
  Fatigue Drift Score using IBM watsonx.ai (or watsonx Assistant).

IMPORTANT DESIGN CONTRACT:
  - This module READS a FatigueResult produced by scoring.py.
  - It NEVER recalculates or overrides the score.
  - AI is used only to EXPLAIN, never to DECIDE.
  - The human operator always has final authority.

TO ACTIVATE:
  1. Install ibm-watsonx-ai: `pip install ibm-watsonx-ai`
  2. Set env vars: WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL
  3. Replace the stub body of `explain_score()` with the real implementation.

EXTENSION POINTS:
  - Swap the prompt template to change explanation style/verbosity.
  - Add a `language` parameter for multilingual mission support.
  - Add structured output parsing if the frontend needs JSON explanation blocks.
"""

import os
from app.models.fatigue import FatigueResult


# ---------------------------------------------------------------------------
# Prompt template
# Adjust tone, length, and format here without touching business logic.
# ---------------------------------------------------------------------------
_EXPLANATION_PROMPT_TEMPLATE = """
You are a mission-support analyst assistant. You are reviewing a Fatigue Drift Score
for astronaut {astronaut_id} computed at {timestamp}.

Fatigue Drift Score: {fatigue_score} / 100
Risk Level: {risk_level}

Signal breakdown (raw value → weighted contribution):
{signal_lines}

Top contributing factors (highest first): {top_factors}

Your task:
1. In 2–3 plain sentences, explain what this score means for mission readiness.
2. For each top contributing factor, briefly explain what it represents.
3. Suggest one concrete, evidence-based countermeasure per high-contribution signal.
4. End with a clear reminder that this is a decision-support tool only and final
   decisions must be made by qualified human personnel.

Do NOT invent any data not present above. Do NOT state a medical diagnosis.
""".strip()


def _build_prompt(result: FatigueResult) -> str:
    """Format the explanation prompt from a FatigueResult."""
    signal_lines = "\n".join(
        f"  - {b.signal}: {b.raw_value} → {b.weighted_contribution} (weight {b.weight})"
        for b in result.signal_breakdown
    )
    top_factors = ", ".join(result.top_contributing_factors)
    return _EXPLANATION_PROMPT_TEMPLATE.format(
        astronaut_id=result.astronaut_id,
        timestamp=result.timestamp.isoformat(),
        fatigue_score=result.fatigue_score,
        risk_level=result.risk_level,
        signal_lines=signal_lines,
        top_factors=top_factors,
    )


def explain_score(result: FatigueResult) -> str:
    """
    Generate a natural-language explanation of a Fatigue Drift Score
    using IBM watsonx.ai.

    Args:
        result: A FatigueResult already computed by scoring.calculate_fatigue_score().

    Returns:
        A plain-text explanation string. Returns a stub message until
        watsonx credentials are configured.

    NOTE: This function only explains an existing score. It does not
    recalculate or modify the score in any way.
    """
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    watsonx_url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

    # --- STUB: return placeholder until credentials are wired up ---
    if not api_key or not project_id:
        prompt = _build_prompt(result)
        return (
            "[IBM watsonx AI explanation not yet configured. "
            "Set WATSONX_API_KEY and WATSONX_PROJECT_ID environment variables "
            "to enable natural-language explanations.]\n\n"
            f"Prompt that would be sent:\n{prompt}"
        )

    # --- REAL IMPLEMENTATION (activate when credentials are available) ---
    # Uncomment and complete the block below:
    #
    # from ibm_watsonx_ai import Credentials
    # from ibm_watsonx_ai.foundation_models import ModelInference
    #
    # credentials = Credentials(url=watsonx_url, api_key=api_key)
    # model = ModelInference(
    #     model_id="ibm/granite-13b-instruct-v2",   # swap model as needed
    #     credentials=credentials,
    #     project_id=project_id,
    #     params={
    #         "max_new_tokens": 400,
    #         "temperature": 0.2,   # low temp for consistent, factual output
    #     },
    # )
    # prompt = _build_prompt(result)
    # response = model.generate_text(prompt=prompt)
    # return response

    return "[IBM watsonx AI explanation stub — implementation pending.]"
