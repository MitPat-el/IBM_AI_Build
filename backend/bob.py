"""
IBM Bob / watsonx.ai explanation layer.

This is the ONLY place in the codebase that calls an LLM.

Two AI features live here:

  1. explain_drift()  [original]
     Invoked when a drift score crosses EXPLANATION_TRIGGER_THRESHOLD.
     Receives a DriftResult and produces a 3-field Explanation:
       - astronaut_message
       - flight_surgeon_brief
       - suggested_intervention

  2. generate_mission_brief()  [new]
     Invoked from POST /mission-brief.
     Receives a BriefContext (pre-assembled deterministic facts) and
     produces a MissionDecisionBrief that synthesizes fatigue state,
     mission risk, What-If trajectory, feasibility, and dependency
     impact into a single structured advisory document.

ARCHITECTURE CONSTRAINT
------------------------
Granite never calculates, modifies, or overrides any deterministic value.
It only narrates facts that have already been computed before the prompt
is constructed.  The Python layer enforces human_review_required=True
regardless of what Granite returns.

FALLBACK GUARANTEE
------------------
Both functions fall back to deterministic templates on any of:
  - missing credentials
  - network / HTTP error
  - malformed or incomplete JSON from Granite
The fallback source strings distinguish the cause so developers can
diagnose without exposing secrets.

Fill in WATSONX_API_KEY / WATSONX_PROJECT_ID / WATSONX_URL via
environment variables.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import httpx

from drift import DriftResult


# ---------------------------------------------------------------------------
# watsonx connection config (from env; no credentials in source)
# ---------------------------------------------------------------------------
WATSONX_URL = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_API_KEY = os.environ.get("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID")
WATSONX_MODEL_ID = os.environ.get("WATSONX_MODEL_ID", "ibm/granite-13b-instruct-v2")

# ---------------------------------------------------------------------------
# Ollama / local Granite config (from env; safe defaults for local dev)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "granite4.1:8b")

# Required keys for each output schema; validated before accepting Granite's response.
_EXPLANATION_REQUIRED_KEYS = {"astronaut_message", "flight_surgeon_brief", "suggested_intervention"}
_BRIEF_REQUIRED_KEYS = {
    "astronaut_message", "executive_summary", "primary_drivers",
    "mission_implications", "intervention_assessment", "recommended_actions",
    "uncertainties", "human_review_required",
}


# ---------------------------------------------------------------------------
# ── ORIGINAL FEATURE ── Single-astronaut fatigue explanation
# ---------------------------------------------------------------------------

@dataclass
class Explanation:
    astronaut_message: str
    flight_surgeon_brief: str
    suggested_intervention: str
    source: str  # "watsonx" | "fallback_template" | "fallback_template:no_credentials"
                 # | "fallback_template:network_error" | "fallback_template:parse_error"


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


def _fallback_explanation(astronaut_name: str, result: DriftResult,
                          source: str = "fallback_template") -> Explanation:
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
        source=source,
    )


def explain_drift(astronaut_name: str, day: int, result: DriftResult) -> Explanation:
    if not (WATSONX_API_KEY and WATSONX_PROJECT_ID):
        return _fallback_explanation(astronaut_name, result,
                                     source="fallback_template:no_credentials")

    try:
        access_token = _get_access_token()
        prompt = _build_prompt(astronaut_name, day, result)
        generated_text = _call_inference(access_token, prompt, max_new_tokens=300)
        parsed = json.loads(generated_text.strip())

        if not _EXPLANATION_REQUIRED_KEYS.issubset(parsed.keys()):
            missing = _EXPLANATION_REQUIRED_KEYS - parsed.keys()
            _log_fallback_reason("missing required fields in explanation response",
                                 detail=str(missing))
            return _fallback_explanation(astronaut_name, result,
                                         source="fallback_template:parse_error")

        return Explanation(
            astronaut_message=parsed["astronaut_message"],
            flight_surgeon_brief=parsed["flight_surgeon_brief"],
            suggested_intervention=parsed["suggested_intervention"],
            source="watsonx",
        )
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        _log_fallback_reason("network/HTTP error during explanation", exc=exc)
        return _fallback_explanation(astronaut_name, result,
                                     source="fallback_template:network_error")
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        _log_fallback_reason("parse error during explanation", exc=exc)
        return _fallback_explanation(astronaut_name, result,
                                     source="fallback_template:parse_error")
    except Exception as exc:
        _log_fallback_reason("unexpected error during explanation", exc=exc)
        return _fallback_explanation(astronaut_name, result,
                                     source="fallback_template:network_error")


# ---------------------------------------------------------------------------
# ── NEW FEATURE ── Mission Decision Brief
# ---------------------------------------------------------------------------

@dataclass
class BriefContext:
    """
    Pre-assembled deterministic facts passed to Granite.
    Every field is a FACT computed by the deterministic backend before
    the prompt is constructed.  Granite never modifies these values.

    Optional fields are None when that context was not requested by the
    caller (e.g. no What-If was simulated, or no impact task was given).
    Granite is instructed to report unavailable data as unavailable rather
    than infer it.
    """
    # Subject
    astronaut_name: str
    astronaut_id: str
    day: int

    # Fatigue state [FACT — drift.py]
    drift_score: float
    risk_level: str           # "nominal" | "elevated" | "high" | "critical"
    sub_scores: dict          # {"reaction_time": float, "sleep_debt": float,
                              #  "circadian": float, "workload": float}

    # Mission-level risk snapshot [FACT — projection.py, optional]
    mission_overall_risk_level: Optional[str] = None
    mission_worst_day: Optional[int] = None
    mission_worst_drift_score: Optional[float] = None

    # What-If trajectory [FACT — projection.simulate_intervention(), optional]
    whatif_comparison: Optional[list[dict]] = None   # rows: {day, original_drift,
                                                     #  whatif_drift, delta,
                                                     #  original_risk_level,
                                                     #  whatif_risk_level}
    reassigned_to: Optional[str] = None

    # Feasibility result [FACT — feasibility.py, optional]
    feasibility_status: Optional[str] = None         # "feasible" | "feasible_with_caution"
                                                     # | "not_feasible"
    feasibility_reasons: Optional[list[str]] = None  # hard-failure reasons
    feasibility_warnings: Optional[list[str]] = None # soft advisory warnings

    # Receiver state at reassignment day [FACT — feasibility.py, optional]
    # Populated only when a feasibility check was run (whatif_reassign_to is set).
    # These are the raw numbers behind the feasibility verdict — Granite must
    # cite them as facts and must NOT use them to recalculate or override the
    # feasibility_status already provided above.
    receiver_drift_score: Optional[float] = None     # feasibility.FatigueFeasibility.drift_score
    receiver_risk_level: Optional[str] = None        # feasibility.FatigueFeasibility.risk_level
    workload_projected_ratio: Optional[float] = None # feasibility.WorkloadFeasibility.projected_ratio

    # Dependency impact [FACT — dependency_graph.py, optional]
    impacted_task_name: Optional[str] = None
    impacted_task_id: Optional[str] = None
    downstream_count: Optional[int] = None
    downstream_tasks: Optional[list[dict]] = None    # [{task_id, name, day,
                                                     #   astronaut_id, risk_level}]


@dataclass
class MissionDecisionBrief:
    """
    Structured AI-generated advisory document.

    human_review_required is ALWAYS True -- enforced in Python after
    parsing Granite's response, never trusted from the model output.
    """
    astronaut_message: str           # calm, jargon-free message to the astronaut
    executive_summary: str           # 2-3 sentences: what is happening, why it matters
    primary_drivers: list[str]       # deterministic sub-scores driving current risk
    mission_implications: list[str]  # downstream tasks / mission work exposed by risk
    intervention_assessment: str     # what changed before vs after in the What-If;
                                     # "No What-If context supplied." if absent
    recommended_actions: list[str]   # non-medical operational options for human consideration
    uncertainties: list[str]         # prototype limitations, data gaps, caveats
    human_review_required: bool      # always True
    source: str                      # "watsonx" | "fallback_template:*"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_fallback_reason(reason: str, exc: Exception | None = None,
                         detail: str | None = None) -> None:
    """
    Write a structured diagnostic line to stderr.
    Never logs WATSONX_API_KEY, WATSONX_PROJECT_ID, or any secret.
    Only logs the exception class name, not its repr() or message
    (which could contain partial credential echoes from HTTP errors).
    """
    parts = [f"[bob.py fallback] {reason}"]
    if exc is not None:
        parts.append(f"({type(exc).__name__})")
    if detail is not None:
        parts.append(f"detail={detail}")
    print(" ".join(parts), file=sys.stderr)


def _get_access_token() -> str:
    """
    Exchange the WATSONX_API_KEY for a short-lived IAM bearer token.
    Raises httpx.HTTPError or httpx.TimeoutException on failure.
    """
    resp = httpx.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": WATSONX_API_KEY,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _call_inference(access_token: str, prompt: str, max_new_tokens: int = 300) -> str:
    """
    Call the watsonx.ai text-generation endpoint and return the raw generated text.
    Raises httpx.HTTPError or httpx.TimeoutException on failure.
    """
    resp = httpx.post(
        f"{WATSONX_URL}/ml/v1/text/generation?version=2023-05-29",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={
            "model_id": WATSONX_MODEL_ID,
            "project_id": WATSONX_PROJECT_ID,
            "input": prompt,
            "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.2},
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["results"][0]["generated_text"]


# ---------------------------------------------------------------------------
# Mission Decision Brief -- prompt builder
# ---------------------------------------------------------------------------

def _build_brief_prompt(ctx: BriefContext) -> str:
    """
    Build the structured prompt sent to Granite for the Mission Decision Brief.

    Every section of the context is labelled [FACT — computed by deterministic model]
    so Granite understands clearly that these values must not be altered.
    The prompt instructs Granite to:
      - only use supplied facts
      - distinguish FACT / INTERPRETATION / ACTION OPTION in its output
      - explicitly state when information is unavailable rather than infer it
      - never diagnose, never recommend medication, never claim fitness-for-duty
      - never claim prototype thresholds are NASA-validated operational thresholds
    """
    # --- Sub-scores section ---
    ss = ctx.sub_scores
    sub_scores_block = (
        f"  Reaction-time sub-score: {ss.get('reaction_time', 'N/A')}\n"
        f"  Sleep-debt sub-score:    {ss.get('sleep_debt', 'N/A')}\n"
        f"  Circadian sub-score:     {ss.get('circadian', 'N/A')}\n"
        f"  Workload sub-score:      {ss.get('workload', 'N/A')}"
    )

    # --- Mission risk section (optional) ---
    if ctx.mission_overall_risk_level is not None:
        mission_block = (
            f"[FACT — computed by deterministic model]\n"
            f"Mission-level overall risk level: {ctx.mission_overall_risk_level}\n"
            f"Mission worst day: {ctx.mission_worst_day}\n"
            f"Mission worst drift score: {ctx.mission_worst_drift_score}"
        )
    else:
        mission_block = "Mission-level risk summary: NOT SUPPLIED — do not infer."

    # --- What-If section (optional) ---
    if ctx.whatif_comparison:
        mod_day_rows = [r for r in ctx.whatif_comparison if r.get("delta", 0) != 0]
        # Show the most informative days: the day modified and up to 2 subsequent days
        sample_rows = mod_day_rows[:3] if mod_day_rows else ctx.whatif_comparison[:3]
        rows_text = "\n".join(
            f"  Day {r['day']}: original drift={r['original_drift']} "
            f"({r['original_risk_level']}) → what-if drift={r['whatif_drift']} "
            f"({r['whatif_risk_level']}), delta={r['delta']:+.4f}"
            for r in sample_rows
        )
        receiver_line = (
            f"Proposed receiver: {ctx.reassigned_to}"
            if ctx.reassigned_to else "Task delayed (no receiver specified)"
        )
        whatif_block = (
            f"[FACT — computed by deterministic model]\n"
            f"{receiver_line}\n"
            f"Before/after drift trajectory (selected days):\n{rows_text}"
        )
    else:
        whatif_block = "What-If trajectory: NOT SUPPLIED — do not infer."

    # --- Feasibility section (optional) ---
    if ctx.feasibility_status is not None:
        reasons_text = (
            ("\n  Hard-failure reasons: " + "; ".join(ctx.feasibility_reasons))
            if ctx.feasibility_reasons else ""
        )
        warnings_text = (
            ("\n  Advisory warnings: " + "; ".join(ctx.feasibility_warnings))
            if ctx.feasibility_warnings else ""
        )
        feasibility_block = (
            f"[FACT — computed by deterministic model]\n"
            f"Feasibility status: {ctx.feasibility_status}{reasons_text}{warnings_text}\n"
            f"NOTE: The feasibility status is a deterministic prototype rule. "
            f"Do not override, soften, or reinterpret it."
        )
    else:
        feasibility_block = "Reassignment feasibility: NOT SUPPLIED — do not infer."

    # --- Receiver state block (optional — only present when feasibility was run) ---
    if (ctx.receiver_drift_score is not None
            and ctx.receiver_risk_level is not None
            and ctx.workload_projected_ratio is not None):
        receiver_block = (
            f"[FACT — computed by deterministic model]\n"
            f"Receiver fatigue state on reassignment day:\n"
            f"  Receiver drift score: {ctx.receiver_drift_score}\n"
            f"  Receiver risk level: {ctx.receiver_risk_level}\n"
            f"  Receiver projected workload ratio (post-reassignment): "
            f"{ctx.workload_projected_ratio}× rolling average\n"
            f"NOTE: These values are the raw inputs behind the feasibility verdict above. "
            f"Do NOT use them to recalculate or override that verdict. "
            f"Use them only to explain the tradeoff: what the source astronaut gains "
            f"vs. what fatigue/workload pressure the receiver would carry."
        )
    else:
        receiver_block = "Receiver fatigue/workload state: NOT SUPPLIED — do not infer."

    # --- Dependency impact section (optional) ---
    if ctx.impacted_task_name is not None and ctx.downstream_count is not None:
        if ctx.downstream_tasks:
            ds_text = "; ".join(
                f"{t['name']} (day {t['day']}, assigned {t['astronaut_id']}, "
                f"risk: {t.get('risk_level', 'unknown')})"
                for t in ctx.downstream_tasks[:5]  # cap at 5 to keep prompt bounded
            )
        else:
            ds_text = "details not supplied"
        dep_block = (
            f"[FACT — computed by deterministic model]\n"
            f"At-risk task: {ctx.impacted_task_name} (ID: {ctx.impacted_task_id})\n"
            f"Downstream tasks at risk if this task slips: {ctx.downstream_count}\n"
            f"Downstream task details: {ds_text}"
        )
    else:
        dep_block = "Dependency impact: NOT SUPPLIED — do not infer."

    return f"""You are a read-only decision-support narrator for spaceflight operations.

ROLE BOUNDARY
You do not calculate, modify, or override any value in this context.
Your only job is to interpret and synthesize the supplied deterministic facts
into a structured advisory document for human mission personnel.

HARD PROHIBITIONS — never do any of the following:
- Invent any score, task name, astronaut ID, dependency, or threshold not present below.
- Change or reinterpret a deterministic feasibility status.
- Diagnose any medical condition or recommend any medication.
- Claim an astronaut is safe or unsafe for duty.
- Claim prototype thresholds are NASA operational or clinical thresholds.
- If information is unavailable, say it is unavailable — do not infer it.
- Use authoritative medical or operational directive language. The following words and
  phrases are FORBIDDEN: "prescribe", "clear for duty", "must rest",
  "implement corrective rest protocols", "is required to", "is mandated to",
  "is ordered to", or any phrasing that frames a recommendation as a clinical
  or command-level directive.
  Use decision-support language instead: "consider additional rest opportunities",
  "consider workload reduction", "for mission-control / human review",
  "mission personnel may wish to consider".

OUTPUT LABELLING — use exactly these prefixes in free-text fields:
  FACT: <deterministic backend result — only cite values from the context below>
  INTERPRETATION: <your explanation of that fact>
  ACTION OPTION: <non-medical operational option for human consideration>

DETERMINISTIC CONTEXT
=====================

[FACT — computed by deterministic model]
Astronaut: {ctx.astronaut_name} (ID: {ctx.astronaut_id})
Mission day: {ctx.day}
Overall drift score (0-1): {ctx.drift_score} (risk level: {ctx.risk_level})
Sub-scores:
{sub_scores_block}

{mission_block}

{whatif_block}

{feasibility_block}

{receiver_block}

{dep_block}

TASK
====
Write a Mission Decision Brief that explicitly connects the supplied facts.

SCOPE SEPARATION — MANDATORY:
There are three distinct scopes in this brief. You must NEVER conflate them.
Treat each scope as a completely separate statement, even within the same field.

  SCOPE 1 — THIS ASTRONAUT, THIS DAY:
    The individual drift score and risk level above apply only to
    {ctx.astronaut_name} on day {ctx.day}. Use language like:
    "{ctx.astronaut_name} is currently {ctx.risk_level} on day {ctx.day} with drift {ctx.drift_score}."

  SCOPE 2 — MISSION-LEVEL PROJECTION (only if mission block is supplied):
    The mission-level overall risk level is a crew-wide worst-case snapshot,
    NOT derived from {ctx.astronaut_name}'s individual score. Introduce it
    with words such as "Separately, the mission-level projection shows..." or
    "Mission-wide, the worst projected day is...".

  SCOPE 3 — WHAT-IF TRAJECTORY (only if What-If block is supplied):
    The before/after drift values are a simulation result, not the current
    measured state. Introduce them as "The What-If simulation shows..." or
    "Under the proposed intervention, drift would change from X to Y."

HARD SCOPE RULE — the following sentence structure is FORBIDDEN:
  "The [individual] drift score of X places the mission at [mission-level risk]."
Any sentence that uses {ctx.astronaut_name}'s individual drift score as the
direct cause or evidence for a mission-level risk classification is incorrect
and must not appear. The mission-level risk comes from the mission block, not
from the individual astronaut block.

Additional requirements:
- Link the highest sub-score(s) to {ctx.astronaut_name}'s individual risk level (Scope 1 only).
- If What-If data is supplied, compare the before/after drift values by number (Scope 3).
  When describing a reassignment, always use the form
  "reassigning [task name] from [source astronaut] to [receiver astronaut]".
  Never use phrases like "swapping the receiver", "exchanging astronauts", or
  "the receiver takes over" — those obscure which person is the source and which is the receiver.
- If dependency impact is supplied, cite the downstream task count and names.
- If feasibility data is supplied, state the exact status word (FEASIBLE / FEASIBLE_WITH_CAUTION /
  NOT_FEASIBLE) and cite any reasons/warnings verbatim. Do not soften or reinterpret it.
- If receiver fatigue/workload state is supplied, use it to explain the reassignment tradeoff
  in this exact order: (1) benefit to source astronaut (drift reduction from What-If delta),
  (2) receiver's current fatigue state (drift score and risk level as supplied),
  (3) receiver's projected workload ratio (as supplied), (4) any later-day mission consequences
  visible from the downstream task list. Do NOT recalculate any of these values.
- Identify tradeoffs where a proposed intervention reduces one risk but may introduce another
  (based only on supplied data — do not speculate about conditions not in this context).
- Recommended actions must be non-medical and operational only.
- Keep uncertainties honest: note that all thresholds are prototype heuristics,
  not validated operational limits, and that final decisions rest with mission personnel.

Respond ONLY in this exact JSON shape, no other text, no markdown fences:
{{
  "astronaut_message": "<one calm, supportive sentence to the astronaut, plain language, no jargon, no diagnosis>",
  "executive_summary": "<sentences that SEPARATELY address each scope present: first sentence covers {ctx.astronaut_name}'s individual drift/risk on day {ctx.day}; if mission summary supplied, a second sentence covers mission-wide risk using 'Separately' or 'Mission-wide'; if What-If supplied, a third sentence introduces it as a simulation result — never blend individual drift into mission-level claims>",
  "primary_drivers": ["<sub-score name and value, referring to {ctx.astronaut_name}'s individual scores only>", "..."],
  "mission_implications": ["<one implication per item, citing task names or downstream counts where supplied>", "..."],
  "intervention_assessment": "<if What-If data supplied: introduce as simulation, compare before/after drift by number, state feasibility status verbatim; if not supplied: state 'No What-If context supplied.'>",
  "recommended_actions": ["ACTION OPTION: <non-medical operational action>", "..."],
  "uncertainties": ["<prototype limitation or data gap>", "..."],
  "human_review_required": true
}}"""


# ---------------------------------------------------------------------------
# Mission Decision Brief -- deterministic fallback
# ---------------------------------------------------------------------------

def _fallback_brief(ctx: BriefContext,
                    source: str = "fallback_template") -> MissionDecisionBrief:
    """
    Fully deterministic fallback for generate_mission_brief().
    Produces a complete MissionDecisionBrief from the supplied BriefContext
    without any LLM call.  Every statement is derived directly from ctx fields.
    """
    # Rank sub-scores highest to lowest
    scored = sorted(
        [
            ("reaction time", ctx.sub_scores.get("reaction_time", 0.0)),
            ("sleep debt",    ctx.sub_scores.get("sleep_debt", 0.0)),
            ("circadian misalignment", ctx.sub_scores.get("circadian", 0.0)),
            ("workload",      ctx.sub_scores.get("workload", 0.0)),
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    top_name, top_val = scored[0]
    second_name, second_val = scored[1]

    # --- astronaut_message ---
    astronaut_message = (
        f"Your current fatigue indicators are at the {ctx.risk_level} level on day {ctx.day}. "
        f"Please follow your scheduled rest plan and flag any concerns to the flight surgeon."
    )

    # --- executive_summary ---
    executive_summary = (
        f"FACT: {ctx.astronaut_name}'s drift score on day {ctx.day} is {ctx.drift_score} "
        f"({ctx.risk_level} risk). "
        f"INTERPRETATION: The two largest contributors are {top_name} "
        f"(sub-score {top_val:.3f}) and {second_name} (sub-score {second_val:.3f}), "
        f"indicating that fatigue accumulation is primarily driven by "
        f"{'impaired reaction time and cognitive performance' if top_name == 'reaction time' else 'insufficient recovery and scheduling pressure'}."
    )
    if ctx.mission_overall_risk_level:
        executive_summary += (
            f" FACT: Mission-wide risk is currently {ctx.mission_overall_risk_level} "
            f"(worst day: {ctx.mission_worst_day}, worst drift: {ctx.mission_worst_drift_score})."
        )

    # --- primary_drivers ---
    primary_drivers = [
        f"{name}: {val:.3f}" for name, val in scored if val > 0
    ]

    # --- mission_implications ---
    mission_implications = []
    if ctx.risk_level in ("high", "critical"):
        mission_implications.append(
            f"FACT: {ctx.astronaut_name} is at {ctx.risk_level} fatigue risk on day {ctx.day}. "
            f"INTERPRETATION: Tasks requiring high precision or safety-critical judgment "
            f"are at increased risk of error."
        )
    if ctx.impacted_task_name and ctx.downstream_count is not None:
        downstream_names = (
            ", ".join(t["name"] for t in (ctx.downstream_tasks or [])[:3])
            if ctx.downstream_tasks else "details not supplied"
        )
        mission_implications.append(
            f"FACT: Task '{ctx.impacted_task_name}' has {ctx.downstream_count} downstream "
            f"dependent task(s) at risk if it slips. "
            f"INTERPRETATION: Downstream tasks include: {downstream_names}. "
            f"A slip here cascades forward in the mission schedule."
        )
    if not mission_implications:
        mission_implications.append(
            "FACT: No specific task or dependency impact context was supplied. "
            "INTERPRETATION: Implications cannot be assessed without task context."
        )

    # --- intervention_assessment ---
    if ctx.whatif_comparison:
        # Find the day(s) with a non-zero delta
        changed = [r for r in ctx.whatif_comparison if r.get("delta", 0) != 0]
        if changed:
            r = changed[0]
            receiver_clause = (
                f"reassigning to {ctx.reassigned_to}" if ctx.reassigned_to else "delaying/removing the task"
            )
            intervention_assessment = (
                f"FACT: Simulating {receiver_clause} on day {r['day']} changes "
                f"{ctx.astronaut_name}'s drift from {r['original_drift']} "
                f"({r['original_risk_level']}) to {r['whatif_drift']} "
                f"({r['whatif_risk_level']}), a delta of {r['delta']:+.4f}. "
            )
            # Receiver tradeoff: state the other side of the coin
            if (ctx.receiver_drift_score is not None
                    and ctx.receiver_risk_level is not None
                    and ctx.workload_projected_ratio is not None):
                intervention_assessment += (
                    f"FACT: Receiver fatigue state on reassignment day — "
                    f"drift score {ctx.receiver_drift_score} ({ctx.receiver_risk_level} risk), "
                    f"projected workload ratio {ctx.workload_projected_ratio}× rolling average. "
                )
            if ctx.feasibility_status:
                intervention_assessment += (
                    f"FACT: Deterministic feasibility check result: {ctx.feasibility_status}."
                )
                if ctx.feasibility_reasons:
                    intervention_assessment += (
                        f" Hard-failure reason(s): {'; '.join(ctx.feasibility_reasons)}."
                    )
                if ctx.feasibility_warnings:
                    intervention_assessment += (
                        f" Advisory warning(s): {'; '.join(ctx.feasibility_warnings)}."
                    )
        else:
            intervention_assessment = (
                "FACT: A What-If simulation was supplied but showed no drift delta "
                "on any day. INTERPRETATION: The proposed change does not materially "
                "alter the drift trajectory under current model parameters."
            )
    elif ctx.feasibility_status:
        intervention_assessment = (
            f"FACT: A feasibility check was performed. "
            f"Result: {ctx.feasibility_status}."
        )
        if ctx.feasibility_reasons:
            intervention_assessment += f" Reasons: {'; '.join(ctx.feasibility_reasons)}."
        if ctx.feasibility_warnings:
            intervention_assessment += f" Warnings: {'; '.join(ctx.feasibility_warnings)}."
    else:
        intervention_assessment = "No What-If context supplied."

    # --- recommended_actions ---
    action_map = {
        "reaction time": "ACTION OPTION: Temporarily reassign any high-precision or safety-critical tasks to a crew member with a lower current drift score.",
        "sleep debt":    "ACTION OPTION: Schedule a protected recovery sleep period before the next high-load event.",
        "circadian misalignment": "ACTION OPTION: Adjust the lighting schedule and shift the next rest period earlier to support circadian re-entrainment.",
        "workload":      "ACTION OPTION: Identify non-critical tasks on the schedule and consider deferring or redistributing them.",
    }
    recommended_actions = [action_map[top_name], action_map[second_name]]
    if ctx.feasibility_status == "not_feasible" and ctx.reassigned_to:
        recommended_actions.append(
            f"ACTION OPTION: The proposed reassignment to {ctx.reassigned_to} was "
            f"assessed as not feasible by the deterministic checker. Consider an "
            f"alternative receiver or a task delay instead."
        )
    elif ctx.feasibility_status == "feasible_with_caution":
        recommended_actions.append(
            "ACTION OPTION: The proposed reassignment passed hard feasibility checks "
            "but carries a dependency conflict warning. Mission personnel should verify "
            "the receiver's critical-path task can absorb additional load before proceeding."
        )

    # --- uncertainties ---
    uncertainties = [
        "All fatigue thresholds in this system are prototype heuristics and are not "
        "NASA-validated operational or clinical limits.",
        "This brief is generated from synthetic mission data and is not based on "
        "real sensor readings or medical assessments.",
        "human_review_required: All decisions must be made by authorized mission personnel. "
        "This output is advisory only.",
    ]
    if ctx.whatif_comparison:
        uncertainties.append(
            "The What-If trajectory is a re-simulation of the deterministic model "
            "under modified inputs. It does not account for second-order effects "
            "such as the receiver's downstream fatigue accumulation."
        )

    return MissionDecisionBrief(
        astronaut_message=astronaut_message,
        executive_summary=executive_summary,
        primary_drivers=primary_drivers,
        mission_implications=mission_implications,
        intervention_assessment=intervention_assessment,
        recommended_actions=recommended_actions,
        uncertainties=uncertainties,
        human_review_required=True,   # enforced, not trusted from model
        source=source,
    )


# ---------------------------------------------------------------------------
# Mission Decision Brief -- public entry point
# ---------------------------------------------------------------------------

def _is_ollama_available() -> bool:
    """
    Probe whether the local Ollama server is reachable.
    Uses a short timeout so a missing Ollama instance fails fast rather than
    blocking the request thread for 30+ seconds.
    """
    try:
        resp = httpx.get(OLLAMA_BASE_URL + "/", timeout=3)
        return resp.is_success
    except Exception:
        return False


def _call_ollama_brief(ctx: BriefContext) -> str:
    """
    Call the local Ollama /api/generate endpoint with JSON mode enabled.

    "format": "json" instructs Ollama to grammar-sample the output so that
    the model emits valid JSON directly, suppressing chain-of-thought or
    free-text preamble.  "stream": false returns the full completion in one
    response object whose generated text is in response["response"].

    Raises httpx.HTTPError / httpx.TimeoutException on transport failure.
    """
    prompt = _build_brief_prompt(ctx)
    resp = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _parse_brief_json(parsed: dict, source: str, ctx: BriefContext
                      ) -> MissionDecisionBrief | None:
    """
    Validate a parsed JSON dict against _BRIEF_REQUIRED_KEYS and list-field
    types.  Returns a MissionDecisionBrief on success, or None (after logging)
    if validation fails.  Shared by both the watsonx and Ollama branches.
    """
    if not _BRIEF_REQUIRED_KEYS.issubset(parsed.keys()):
        missing = _BRIEF_REQUIRED_KEYS - parsed.keys()
        _log_fallback_reason("missing required fields in brief response",
                             detail=str(missing))
        return None

    for list_field in ("primary_drivers", "mission_implications",
                       "recommended_actions", "uncertainties"):
        if not isinstance(parsed[list_field], list):
            _log_fallback_reason(f"field '{list_field}' is not a list in brief response")
            return None

    return MissionDecisionBrief(
        astronaut_message=str(parsed["astronaut_message"]),
        executive_summary=str(parsed["executive_summary"]),
        primary_drivers=list(parsed["primary_drivers"]),
        mission_implications=list(parsed["mission_implications"]),
        intervention_assessment=str(parsed["intervention_assessment"]),
        recommended_actions=list(parsed["recommended_actions"]),
        uncertainties=list(parsed["uncertainties"]),
        human_review_required=True,   # enforced here, never trusted from model
        source=source,
    )


def generate_mission_brief(ctx: BriefContext) -> MissionDecisionBrief:
    """
    Generate a MissionDecisionBrief from a pre-assembled BriefContext.

    Provider order (approved):
      1. watsonx  — if credentials are present AND the call succeeds.
                    On any failure (network, parse, HTTP) fall through to (2).
      2. Ollama   — if the local Ollama server is reachable.
                    On any failure fall through to (3).
      3. Deterministic fallback — always available, no external dependencies.

    human_review_required is always forced to True in Python after parsing,
    regardless of what the model returns.
    """
    # ------------------------------------------------------------------
    # Branch 1: watsonx
    # ------------------------------------------------------------------
    if WATSONX_API_KEY and WATSONX_PROJECT_ID:
        try:
            access_token = _get_access_token()
            prompt = _build_brief_prompt(ctx)
            generated_text = _call_inference(access_token, prompt, max_new_tokens=600)
            parsed = json.loads(generated_text.strip())
            result = _parse_brief_json(parsed, source="watsonx", ctx=ctx)
            if result is not None:
                return result
            # Validation failed — fall through to Ollama
            _log_fallback_reason("watsonx brief validation failed; trying Ollama")
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            _log_fallback_reason("network/HTTP error during watsonx brief; trying Ollama",
                                 exc=exc)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            _log_fallback_reason("parse error during watsonx brief; trying Ollama", exc=exc)
        except Exception as exc:
            _log_fallback_reason("unexpected error during watsonx brief; trying Ollama",
                                 exc=exc)

    # ------------------------------------------------------------------
    # Branch 2: local Ollama / IBM Granite
    # ------------------------------------------------------------------
    if _is_ollama_available():
        try:
            generated_text = _call_ollama_brief(ctx)
            parsed = json.loads(generated_text.strip())
            result = _parse_brief_json(parsed, source="ollama_granite", ctx=ctx)
            if result is not None:
                return result
            # Validation failed — fall through to deterministic fallback
            _log_fallback_reason("ollama_granite brief validation failed; using fallback")
            return _fallback_brief(ctx, source="fallback_template:ollama_error")
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            _log_fallback_reason("network/HTTP error during ollama brief", exc=exc)
            return _fallback_brief(ctx, source="fallback_template:ollama_error")
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            _log_fallback_reason("parse error during ollama brief", exc=exc)
            return _fallback_brief(ctx, source="fallback_template:ollama_error")
        except Exception as exc:
            _log_fallback_reason("unexpected error during ollama brief", exc=exc)
            return _fallback_brief(ctx, source="fallback_template:ollama_error")

    # ------------------------------------------------------------------
    # Branch 3: deterministic fallback
    # ------------------------------------------------------------------
    return _fallback_brief(ctx, source="fallback_template:no_credentials")
