"""
IBM Bob / watsonx.ai explanation layer.

This is the ONLY place in the codebase that calls an LLM. It is invoked
strictly after compute_drift_score() has already produced a numeric
result and that result has crossed EXPLANATION_TRIGGER_THRESHOLD. Bob
never sees raw sensor data and never computes a score itself -- it
receives the four already-computed sub-scores plus context, and its
only job is to translate that into two pieces of plain-language text:

  1. A short, non-alarming message TO the astronaut
  2. A technical brief TO the flight surgeon, including the suggested
     intervention

Fill in WATSONX_API_KEY / WATSONX_PROJECT_ID / WATSONX_URL via
environment variables. Until those are set, `explain_drift()` falls
back to a deterministic template so the rest of the app is runnable
and demoable without live credentials.
"""

import os
import json
from dataclasses import dataclass

import httpx

from drift import DriftResult


WATSONX_URL = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_API_KEY = os.environ.get("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID")
WATSONX_MODEL_ID = os.environ.get("WATSONX_MODEL_ID", "ibm/granite-13b-instruct-v2")


@dataclass
class Explanation:
    astronaut_message: str
    flight_surgeon_brief: str
    suggested_intervention: str
    source: str  # "watsonx" or "fallback_template"


def _build_prompt(astronaut_name: str, day: int, result: DriftResult) -> str:
    return f"""You are a fatigue-monitoring assistant for spaceflight operations.
An astronaut's computed drift score has crossed the alert threshold. You are
given ONLY the already-computed sub-scores below -- do not invent numbers,
do not diagnose, do not recommend medication. Your job is to explain what
is driving the score and suggest one concrete, non-medical operational
intervention (e.g. reassign a task, add rest, delay a non-critical activity).

Astronaut: {astronaut_name}
Mission day: {day}
Overall drift score (0-1): {result.drift_score} ({result.risk_level})
Reaction-time sub-score: {result.reaction_time_score}
Sleep-debt sub-score: {result.sleep_debt_score}
Circadian-misalignment sub-score: {result.circadian_score}
Workload sub-score: {result.workload_score}

Respond ONLY in this exact JSON shape, no other text:
{{
  "astronaut_message": "<one calm, supportive sentence to the astronaut, no jargon>",
  "flight_surgeon_brief": "<2-3 sentences, technical, cites which sub-score(s) are driving the score>",
  "suggested_intervention": "<one concrete operational action>"
}}"""


def _fallback_explanation(astronaut_name: str, result: DriftResult) -> Explanation:
    """Deterministic, no-API-key-needed fallback so the app runs end-to-end in a demo."""
    drivers = sorted(
        [
            ("reaction time", result.reaction_time_score),
            ("sleep debt", result.sleep_debt_score),
            ("circadian misalignment", result.circadian_score),
            ("workload", result.workload_score),
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    top_driver, top_value = drivers[0]

    intervention_map = {
        "reaction time": "Move any high-precision or safety-critical tasks to another crew member for the next work block.",
        "sleep debt": "Schedule a protected recovery sleep period before the next demanding task.",
        "circadian misalignment": "Adjust lighting schedule and shift the next rest period to encourage re-entrainment.",
        "workload": "Redistribute or delay non-critical tasks scheduled for today.",
    }

    return Explanation(
        astronaut_message=(
            f"Your fatigue indicators are trending {result.risk_level}. "
            f"Consider taking your next rest window as scheduled."
        ),
        flight_surgeon_brief=(
            f"{astronaut_name}: drift score {result.drift_score} ({result.risk_level}). "
            f"Primary driver: {top_driver} (sub-score {top_value}). "
            f"Secondary contributors: "
            + ", ".join(f"{name} ({val})" for name, val in drivers[1:])
            + "."
        ),
        suggested_intervention=intervention_map[top_driver],
        source="fallback_template",
    )


def explain_drift(astronaut_name: str, day: int, result: DriftResult) -> Explanation:
    if not (WATSONX_API_KEY and WATSONX_PROJECT_ID):
        return _fallback_explanation(astronaut_name, result)

    try:
        # NOTE: watsonx.ai requires an IAM bearer token exchange before calling
        # the inference endpoint. Swap this for the ibm-watsonx-ai SDK if
        # preferred -- shown raw here so the HTTP contract is explicit.
        token_resp = httpx.post(
            "https://iam.cloud.ibm.com/identity/token",
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": WATSONX_API_KEY,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        token_resp.raise_for_status()
        access_token = token_resp.json()["access_token"]

        prompt = _build_prompt(astronaut_name, day, result)
        infer_resp = httpx.post(
            f"{WATSONX_URL}/ml/v1/text/generation?version=2023-05-29",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={
                "model_id": WATSONX_MODEL_ID,
                "project_id": WATSONX_PROJECT_ID,
                "input": prompt,
                "parameters": {"max_new_tokens": 300, "temperature": 0.2},
            },
            timeout=30,
        )
        infer_resp.raise_for_status()
        generated_text = infer_resp.json()["results"][0]["generated_text"]
        parsed = json.loads(generated_text.strip())

        return Explanation(
            astronaut_message=parsed["astronaut_message"],
            flight_surgeon_brief=parsed["flight_surgeon_brief"],
            suggested_intervention=parsed["suggested_intervention"],
            source="watsonx",
        )
    except Exception:
        # Never let an LLM/network hiccup take down the alert pipeline --
        # fall back to the deterministic template.
        return _fallback_explanation(astronaut_name, result)