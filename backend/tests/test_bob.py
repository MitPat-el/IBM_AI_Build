"""
Unit tests for the bob.py explanation and mission-brief layer.

All watsonx / httpx calls are mocked -- no network access or real
credentials are required.  Tests cover both the original explain_drift()
path and the new generate_mission_brief() path.
"""

import json
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from drift import DriftResult
from bob import (
    BriefContext,
    Explanation,
    MissionDecisionBrief,
    _build_brief_prompt,
    _fallback_brief,
    _fallback_explanation,
    _log_fallback_reason,
    explain_drift,
    generate_mission_brief,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _drift(score: float = 0.72, level: str = "high") -> DriftResult:
    return DriftResult(
        reaction_time_score=0.60,
        sleep_debt_score=0.55,
        circadian_score=0.30,
        workload_score=0.20,
        drift_score=score,
        updated_sleep_debt_hours=2.5,
        risk_level=level,
    )


def _base_ctx(**overrides) -> BriefContext:
    defaults = dict(
        astronaut_name="Chen",
        astronaut_id="A1",
        day=4,
        drift_score=0.72,
        risk_level="high",
        sub_scores={
            "reaction_time": 0.60,
            "sleep_debt": 0.55,
            "circadian": 0.30,
            "workload": 0.20,
        },
    )
    defaults.update(overrides)
    return BriefContext(**defaults)


def _valid_brief_json(**overrides) -> str:
    """Returns a minimal valid MissionDecisionBrief JSON string."""
    base = {
        "astronaut_message": "Stay the course.",
        "executive_summary": "Drift is high on day 4.",
        "primary_drivers": ["reaction time: 0.600"],
        "mission_implications": ["Risk to EVA prep."],
        "intervention_assessment": "No What-If context supplied.",
        "recommended_actions": ["ACTION OPTION: Reassign high-precision tasks."],
        "uncertainties": ["Prototype thresholds only."],
        "human_review_required": True,
    }
    base.update(overrides)
    return json.dumps(base)


def _mock_httpx_post(token_json=None, infer_json=None, raise_on_token=None, raise_on_infer=None):
    """
    Returns a side_effect function for patching httpx.post.
    First call = IAM token exchange, second call = inference.
    """
    call_count = {"n": 0}

    def side_effect(url, **kwargs):
        call_count["n"] += 1
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        if "iam.cloud.ibm.com" in url:
            if raise_on_token:
                raise raise_on_token
            mock_resp.json.return_value = token_json or {"access_token": "test-token-abc"}
        else:
            if raise_on_infer:
                raise raise_on_infer
            payload = infer_json or {"results": [{"generated_text": _valid_brief_json()}]}
            mock_resp.json.return_value = payload

        return mock_resp

    return side_effect


# ---------------------------------------------------------------------------
# _log_fallback_reason
# ---------------------------------------------------------------------------

def test_log_fallback_reason_writes_to_stderr(capsys):
    _log_fallback_reason("test reason")
    captured = capsys.readouterr()
    assert "[bob.py fallback] test reason" in captured.err


def test_log_fallback_reason_includes_exception_class_not_message(capsys):
    exc = ValueError("secret-api-key-value-do-not-log")
    _log_fallback_reason("something failed", exc=exc)
    captured = capsys.readouterr()
    assert "ValueError" in captured.err
    # The exception message (which might contain sensitive content) must NOT appear
    assert "secret-api-key-value-do-not-log" not in captured.err


def test_log_fallback_reason_includes_detail_string(capsys):
    _log_fallback_reason("missing fields", detail="{'foo', 'bar'}")
    captured = capsys.readouterr()
    assert "detail=" in captured.err


# ---------------------------------------------------------------------------
# _fallback_explanation (original explain_drift path)
# ---------------------------------------------------------------------------

def test_fallback_explanation_returns_explanation_instance():
    result = _fallback_explanation("Chen", _drift())
    assert isinstance(result, Explanation)


def test_fallback_explanation_source_default():
    result = _fallback_explanation("Chen", _drift())
    assert result.source == "fallback_template"


def test_fallback_explanation_source_override():
    result = _fallback_explanation("Chen", _drift(), source="fallback_template:no_credentials")
    assert result.source == "fallback_template:no_credentials"


def test_fallback_explanation_all_fields_populated():
    result = _fallback_explanation("Chen", _drift())
    assert result.astronaut_message
    assert result.flight_surgeon_brief
    assert result.suggested_intervention


# ---------------------------------------------------------------------------
# explain_drift -- no credentials path
# ---------------------------------------------------------------------------

def test_explain_drift_no_credentials_returns_fallback():
    with patch.dict("os.environ", {}, clear=False):
        # Ensure WATSONX_API_KEY and WATSONX_PROJECT_ID are absent
        import bob
        original_key = bob.WATSONX_API_KEY
        original_pid = bob.WATSONX_PROJECT_ID
        bob.WATSONX_API_KEY = None
        bob.WATSONX_PROJECT_ID = None
        try:
            result = explain_drift("Chen", 4, _drift())
            assert isinstance(result, Explanation)
            assert result.source == "fallback_template:no_credentials"
        finally:
            bob.WATSONX_API_KEY = original_key
            bob.WATSONX_PROJECT_ID = original_pid


def test_explain_drift_with_credentials_and_valid_response():
    import bob
    bob.WATSONX_API_KEY = "fake-key"
    bob.WATSONX_PROJECT_ID = "fake-project"
    valid_json = json.dumps({
        "astronaut_message": "You're doing well.",
        "flight_surgeon_brief": "Drift is elevated due to sleep debt.",
        "suggested_intervention": "Schedule a rest block.",
    })
    try:
        with patch("bob.httpx.post", side_effect=_mock_httpx_post(
            infer_json={"results": [{"generated_text": valid_json}]}
        )):
            result = explain_drift("Chen", 4, _drift())
        assert result.source == "watsonx"
        assert result.astronaut_message == "You're doing well."
    finally:
        bob.WATSONX_API_KEY = None
        bob.WATSONX_PROJECT_ID = None


def test_explain_drift_http_error_falls_back():
    import bob
    import httpx
    bob.WATSONX_API_KEY = "fake-key"
    bob.WATSONX_PROJECT_ID = "fake-project"
    try:
        with patch("bob.httpx.post", side_effect=_mock_httpx_post(
            raise_on_infer=httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock()
            )
        )):
            result = explain_drift("Chen", 4, _drift())
        assert "fallback_template" in result.source
    finally:
        bob.WATSONX_API_KEY = None
        bob.WATSONX_PROJECT_ID = None


def test_explain_drift_missing_fields_falls_back(capsys):
    import bob
    bob.WATSONX_API_KEY = "fake-key"
    bob.WATSONX_PROJECT_ID = "fake-project"
    # Return JSON missing 'suggested_intervention'
    partial_json = json.dumps({
        "astronaut_message": "OK",
        "flight_surgeon_brief": "All good",
    })
    try:
        with patch("bob.httpx.post", side_effect=_mock_httpx_post(
            infer_json={"results": [{"generated_text": partial_json}]}
        )):
            result = explain_drift("Chen", 4, _drift())
        assert result.source == "fallback_template:parse_error"
    finally:
        bob.WATSONX_API_KEY = None
        bob.WATSONX_PROJECT_ID = None


# ---------------------------------------------------------------------------
# _fallback_brief
# ---------------------------------------------------------------------------

def test_fallback_brief_returns_mission_decision_brief():
    result = _fallback_brief(_base_ctx())
    assert isinstance(result, MissionDecisionBrief)


def test_fallback_brief_human_review_required_is_always_true():
    result = _fallback_brief(_base_ctx())
    assert result.human_review_required is True


def test_fallback_brief_all_required_fields_populated():
    result = _fallback_brief(_base_ctx())
    assert result.astronaut_message
    assert result.executive_summary
    assert isinstance(result.primary_drivers, list) and len(result.primary_drivers) > 0
    assert isinstance(result.mission_implications, list) and len(result.mission_implications) > 0
    assert result.intervention_assessment
    assert isinstance(result.recommended_actions, list) and len(result.recommended_actions) > 0
    assert isinstance(result.uncertainties, list) and len(result.uncertainties) > 0


def test_fallback_brief_executive_summary_cites_drift_score():
    ctx = _base_ctx(drift_score=0.72, risk_level="high", astronaut_name="Chen", day=4)
    result = _fallback_brief(ctx)
    assert "0.72" in result.executive_summary
    assert "high" in result.executive_summary.lower()


def test_fallback_brief_executive_summary_cites_top_two_sub_scores():
    # reaction_time=0.60 is top, sleep_debt=0.55 is second
    ctx = _base_ctx()
    result = _fallback_brief(ctx)
    assert "reaction time" in result.executive_summary.lower()
    assert "sleep debt" in result.executive_summary.lower()


def test_fallback_brief_primary_drivers_nonempty_and_nonzero():
    ctx = _base_ctx()
    result = _fallback_brief(ctx)
    # All four sub-scores are > 0, so all four should appear
    assert len(result.primary_drivers) == 4


def test_fallback_brief_intervention_assessment_no_whatif():
    ctx = _base_ctx()  # no whatif_comparison
    result = _fallback_brief(ctx)
    assert "No What-If context supplied" in result.intervention_assessment


def test_fallback_brief_intervention_assessment_with_whatif():
    ctx = _base_ctx(
        whatif_comparison=[
            {
                "day": 4,
                "original_drift": 0.72,
                "whatif_drift": 0.55,
                "delta": -0.17,
                "original_risk_level": "high",
                "whatif_risk_level": "elevated",
            }
        ],
        reassigned_to="A3",
    )
    result = _fallback_brief(ctx)
    assert "0.72" in result.intervention_assessment
    assert "0.55" in result.intervention_assessment
    assert "-0.17" in result.intervention_assessment or "-0.1700" in result.intervention_assessment
    assert "A3" in result.intervention_assessment


def test_fallback_brief_intervention_assessment_cites_feasibility_when_not_feasible():
    ctx = _base_ctx(
        whatif_comparison=[
            {
                "day": 4,
                "original_drift": 0.72,
                "whatif_drift": 0.50,
                "delta": -0.22,
                "original_risk_level": "high",
                "whatif_risk_level": "elevated",
            }
        ],
        reassigned_to="A2",
        feasibility_status="not_feasible",
        feasibility_reasons=["[PROTOTYPE RULE] Receiver is at HIGH fatigue risk."],
    )
    result = _fallback_brief(ctx)
    assert "not_feasible" in result.intervention_assessment
    assert "[PROTOTYPE RULE]" in result.intervention_assessment


def test_fallback_brief_mission_implications_cite_downstream_when_supplied():
    ctx = _base_ctx(
        impacted_task_name="EVA - External Repair",
        impacted_task_id="T7",
        downstream_count=5,
        downstream_tasks=[
            {"task_id": "T10", "name": "Cargo Transfer", "day": 4, "astronaut_id": "A2", "risk_level": "elevated"},
            {"task_id": "T13", "name": "Robotic Arm Ops", "day": 5, "astronaut_id": "A1", "risk_level": "high"},
        ],
    )
    result = _fallback_brief(ctx)
    implications_text = " ".join(result.mission_implications)
    assert "EVA - External Repair" in implications_text
    assert "5" in implications_text
    assert "Cargo Transfer" in implications_text


def test_fallback_brief_uncertainties_always_includes_prototype_disclaimer():
    result = _fallback_brief(_base_ctx())
    unc_text = " ".join(result.uncertainties).lower()
    assert "prototype" in unc_text


def test_fallback_brief_uncertainties_includes_whatif_caveat_when_whatif_supplied():
    ctx = _base_ctx(
        whatif_comparison=[
            {"day": 4, "original_drift": 0.72, "whatif_drift": 0.55, "delta": -0.17,
             "original_risk_level": "high", "whatif_risk_level": "elevated"}
        ]
    )
    result = _fallback_brief(ctx)
    unc_text = " ".join(result.uncertainties).lower()
    assert "what-if" in unc_text or "re-simulation" in unc_text


def test_fallback_brief_recommended_actions_not_feasible_includes_alternative_suggestion():
    ctx = _base_ctx(
        whatif_comparison=[
            {"day": 4, "original_drift": 0.72, "whatif_drift": 0.55, "delta": -0.17,
             "original_risk_level": "high", "whatif_risk_level": "elevated"}
        ],
        reassigned_to="A2",
        feasibility_status="not_feasible",
        feasibility_reasons=["Receiver is at HIGH fatigue risk."],
    )
    result = _fallback_brief(ctx)
    actions_text = " ".join(result.recommended_actions)
    assert "not feasible" in actions_text.lower() or "A2" in actions_text


def test_fallback_brief_mission_summary_in_executive_when_supplied():
    ctx = _base_ctx(
        mission_overall_risk_level="critical",
        mission_worst_day=5,
        mission_worst_drift_score=0.88,
    )
    result = _fallback_brief(ctx)
    assert "critical" in result.executive_summary.lower()
    assert "5" in result.executive_summary


def test_fallback_brief_source_passthrough():
    result = _fallback_brief(_base_ctx(), source="fallback_template:network_error")
    assert result.source == "fallback_template:network_error"


# ---------------------------------------------------------------------------
# generate_mission_brief -- no credentials
# ---------------------------------------------------------------------------

def test_generate_mission_brief_no_credentials_fallback():
    import bob
    bob.WATSONX_API_KEY = None
    bob.WATSONX_PROJECT_ID = None
    # Mock Ollama probe as unreachable so we reach the deterministic fallback
    with patch("bob.httpx.get", side_effect=Exception("no ollama")):
        result = generate_mission_brief(_base_ctx())
    assert isinstance(result, MissionDecisionBrief)
    assert result.source == "fallback_template:no_credentials"
    assert result.human_review_required is True


# ---------------------------------------------------------------------------
# generate_mission_brief -- with credentials, mocked watsonx
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def _fake_creds():
    """Temporarily inject fake watsonx credentials for tests that need them."""
    import bob
    bob.WATSONX_API_KEY = "fake-key"
    bob.WATSONX_PROJECT_ID = "fake-project"
    yield
    bob.WATSONX_API_KEY = None
    bob.WATSONX_PROJECT_ID = None


def test_generate_mission_brief_valid_response_from_watsonx(_fake_creds):
    with patch("bob.httpx.post", side_effect=_mock_httpx_post(
        infer_json={"results": [{"generated_text": _valid_brief_json()}]}
    )):
        result = generate_mission_brief(_base_ctx())
    assert result.source == "watsonx"
    assert result.astronaut_message == "Stay the course."
    assert result.executive_summary == "Drift is high on day 4."


def test_generate_mission_brief_human_review_required_forced_true_even_if_model_returns_false(_fake_creds):
    """Python must override the model's human_review_required value."""
    with patch("bob.httpx.post", side_effect=_mock_httpx_post(
        infer_json={"results": [{"generated_text": _valid_brief_json(human_review_required=False)}]}
    )):
        result = generate_mission_brief(_base_ctx())
    assert result.human_review_required is True


def test_generate_mission_brief_network_error_falls_back(_fake_creds, capsys):
    """
    watsonx token request times out → fall through to Ollama probe.
    Ollama also mocked as unreachable → deterministic fallback.
    Source is fallback_template:no_credentials (final branch reached).
    """
    import httpx
    with patch("bob.httpx.post", side_effect=_mock_httpx_post(
        raise_on_token=httpx.TimeoutException("timeout")
    )), patch("bob.httpx.get", side_effect=Exception("no ollama")):
        result = generate_mission_brief(_base_ctx())
    assert result.source == "fallback_template:no_credentials"
    assert result.human_review_required is True
    captured = capsys.readouterr()
    assert "[bob.py fallback]" in captured.err


def test_generate_mission_brief_http_error_on_inference_falls_back(_fake_creds, capsys):
    """watsonx inference returns HTTP 503 → fall through to Ollama (mocked unreachable) → fallback."""
    import httpx
    with patch("bob.httpx.post", side_effect=_mock_httpx_post(
        raise_on_infer=httpx.HTTPStatusError("503", request=MagicMock(), response=MagicMock())
    )), patch("bob.httpx.get", side_effect=Exception("no ollama")):
        result = generate_mission_brief(_base_ctx())
    assert result.source == "fallback_template:no_credentials"
    captured = capsys.readouterr()
    assert "[bob.py fallback]" in captured.err


def test_generate_mission_brief_malformed_json_falls_back(_fake_creds, capsys):
    """watsonx returns malformed JSON → fall through to Ollama (unreachable) → fallback."""
    with patch("bob.httpx.post", side_effect=_mock_httpx_post(
        infer_json={"results": [{"generated_text": "not valid JSON {{{{"}]}
    )), patch("bob.httpx.get", side_effect=Exception("no ollama")):
        result = generate_mission_brief(_base_ctx())
    assert result.source == "fallback_template:no_credentials"
    captured = capsys.readouterr()
    assert "[bob.py fallback]" in captured.err


def test_generate_mission_brief_missing_required_fields_falls_back(_fake_creds, capsys):
    """watsonx returns JSON missing 'executive_summary' → fall through to Ollama (unreachable) → fallback."""
    partial = json.dumps({
        "astronaut_message": "OK",
        "primary_drivers": [],
        "mission_implications": [],
        "intervention_assessment": "none",
        "recommended_actions": [],
        "uncertainties": [],
        "human_review_required": True,
        # executive_summary is missing
    })
    with patch("bob.httpx.post", side_effect=_mock_httpx_post(
        infer_json={"results": [{"generated_text": partial}]}
    )), patch("bob.httpx.get", side_effect=Exception("no ollama")):
        result = generate_mission_brief(_base_ctx())
    assert result.source == "fallback_template:no_credentials"
    captured = capsys.readouterr()
    assert "[bob.py fallback]" in captured.err


def test_generate_mission_brief_list_field_is_string_falls_back(_fake_creds):
    """primary_drivers returned as a string → validation fails → Ollama (unreachable) → fallback."""
    bad = _valid_brief_json(primary_drivers="reaction time: 0.600")  # string not list
    with patch("bob.httpx.post", side_effect=_mock_httpx_post(
        infer_json={"results": [{"generated_text": bad}]}
    )), patch("bob.httpx.get", side_effect=Exception("no ollama")):
        result = generate_mission_brief(_base_ctx())
    assert result.source == "fallback_template:no_credentials"


def test_generate_mission_brief_deterministic_context_in_response(_fake_creds):
    """
    When watsonx is mocked to return a valid response, the deterministic
    drift_score, risk_level, and astronaut_name from the context must not be
    silently dropped -- the brief's executive_summary should reference the
    day (we echo it from mock) but most critically human_review_required=True.
    """
    with patch("bob.httpx.post", side_effect=_mock_httpx_post(
        infer_json={"results": [{"generated_text": _valid_brief_json()}]}
    )):
        result = generate_mission_brief(_base_ctx(drift_score=0.72, risk_level="high"))
    assert result.human_review_required is True
    assert result.source == "watsonx"


# ---------------------------------------------------------------------------
# _build_brief_prompt -- structural checks (no LLM call)
# ---------------------------------------------------------------------------

def test_build_brief_prompt_contains_role_boundary():
    prompt = _build_brief_prompt(_base_ctx())
    assert "read-only decision-support narrator" in prompt
    assert "do not calculate" in prompt.lower() or "You do not calculate" in prompt


def test_build_brief_prompt_contains_all_prohibitions():
    prompt = _build_brief_prompt(_base_ctx())
    for phrase in ("Invent", "medical condition", "medication", "safe or unsafe for duty",
                   "NASA operational or clinical"):
        assert phrase in prompt, f"Missing prohibition: '{phrase}'"


def test_build_brief_prompt_labels_facts_as_deterministic():
    prompt = _build_brief_prompt(_base_ctx())
    assert "[FACT — computed by deterministic model]" in prompt


def test_build_brief_prompt_includes_sub_scores():
    ctx = _base_ctx()
    prompt = _build_brief_prompt(ctx)
    # Floats render as 0.6 (not 0.60) in Python's default str()
    assert "0.6" in prompt    # reaction_time (0.60 stored, displayed as 0.6)
    assert "0.55" in prompt   # sleep_debt


def test_build_brief_prompt_whatif_block_when_supplied():
    ctx = _base_ctx(
        whatif_comparison=[
            {"day": 4, "original_drift": 0.72, "whatif_drift": 0.55,
             "delta": -0.17, "original_risk_level": "high", "whatif_risk_level": "elevated"}
        ],
        reassigned_to="A3",
    )
    prompt = _build_brief_prompt(ctx)
    assert "0.72" in prompt
    assert "0.55" in prompt
    assert "A3" in prompt


def test_build_brief_prompt_whatif_absent_says_not_supplied():
    prompt = _build_brief_prompt(_base_ctx())
    assert "NOT SUPPLIED" in prompt


def test_build_brief_prompt_feasibility_block_when_supplied():
    ctx = _base_ctx(
        feasibility_status="not_feasible",
        feasibility_reasons=["Receiver at HIGH fatigue risk."],
    )
    prompt = _build_brief_prompt(ctx)
    assert "not_feasible" in prompt
    assert "Receiver at HIGH fatigue risk." in prompt
    assert "Do not override" in prompt


def test_build_brief_prompt_dependency_block_when_supplied():
    ctx = _base_ctx(
        impacted_task_name="EVA - External Repair",
        impacted_task_id="T7",
        downstream_count=5,
        downstream_tasks=[
            {"task_id": "T10", "name": "Cargo Transfer", "day": 4,
             "astronaut_id": "A2", "risk_level": "elevated"},
        ],
    )
    prompt = _build_brief_prompt(ctx)
    assert "EVA - External Repair" in prompt
    assert "5" in prompt
    assert "Cargo Transfer" in prompt


def test_build_brief_prompt_requires_json_output_shape():
    prompt = _build_brief_prompt(_base_ctx())
    assert "human_review_required" in prompt
    assert "executive_summary" in prompt
    assert "primary_drivers" in prompt
    assert "intervention_assessment" in prompt


def test_build_brief_prompt_contains_scope_separation_rules():
    """The new scope-separation section must be present and name all three scopes."""
    prompt = _build_brief_prompt(_base_ctx())
    assert "SCOPE SEPARATION" in prompt
    assert "SCOPE 1" in prompt
    assert "SCOPE 2" in prompt
    assert "SCOPE 3" in prompt


def test_build_brief_prompt_contains_hard_scope_rule():
    """The forbidden sentence pattern must be explicitly called out in the prompt."""
    prompt = _build_brief_prompt(_base_ctx())
    assert "HARD SCOPE RULE" in prompt
    assert "FORBIDDEN" in prompt


def test_build_brief_prompt_executive_summary_instruction_separates_scopes():
    """The executive_summary field instruction must require scope separation, not blending."""
    prompt = _build_brief_prompt(_base_ctx())
    # The instruction should explicitly tell Granite to keep scopes separate
    assert "SEPARATELY" in prompt or "separately" in prompt
    # The old instruction that invited conflation must be gone
    assert "synthesizing drift score, risk level" not in prompt


def test_build_brief_prompt_scope1_references_subject_astronaut():
    """SCOPE 1 instruction must be bound to the specific astronaut from the context."""
    ctx = _base_ctx(astronaut_name="Chen", day=4, drift_score=0.72, risk_level="high")
    prompt = _build_brief_prompt(ctx)
    # Scope 1 example sentence should contain the actual astronaut name and day
    assert "Chen" in prompt
    assert "day 4" in prompt or "day=4" in prompt or "0.72" in prompt


def test_build_brief_prompt_prohibits_authoritative_directive_language():
    """Forbidden directive phrases must be listed in the prompt's HARD PROHIBITIONS."""
    prompt = _build_brief_prompt(_base_ctx())
    for phrase in ("prescribe", "clear for duty", "must rest",
                   "implement corrective rest protocols"):
        assert phrase in prompt, f"Forbidden phrase not called out in prompt: '{phrase}'"


def test_build_brief_prompt_offers_decision_support_alternatives():
    """Prompt must supply decision-support language alternatives to directive phrasing."""
    prompt = _build_brief_prompt(_base_ctx())
    assert "consider additional rest opportunities" in prompt
    assert "consider workload reduction" in prompt
    assert "mission personnel may wish to consider" in prompt


def test_build_brief_prompt_reassignment_wording_rule_present():
    """Prompt must specify the required 'reassigning X from A to B' wording for What-If."""
    prompt = _build_brief_prompt(_base_ctx())
    assert "reassigning [task name] from [source astronaut] to [receiver astronaut]" in prompt


def test_build_brief_prompt_reassignment_forbidden_phrases_called_out():
    """Prompt must explicitly forbid ambiguous receiver-swap phrases."""
    prompt = _build_brief_prompt(_base_ctx())
    assert "swapping the receiver" in prompt   # cited as forbidden
    assert "exchanging astronauts" in prompt   # cited as forbidden
    assert "the receiver takes over" in prompt  # cited as forbidden


# ---------------------------------------------------------------------------
# New fields: receiver_drift_score, receiver_risk_level, workload_projected_ratio
# ---------------------------------------------------------------------------

def _base_ctx_with_receiver(**overrides) -> BriefContext:
    """Base context that includes a What-If + feasibility + all three receiver fields."""
    defaults = dict(
        astronaut_name="Chen",
        astronaut_id="A1",
        day=4,
        drift_score=0.72,
        risk_level="high",
        sub_scores={
            "reaction_time": 0.60,
            "sleep_debt": 0.55,
            "circadian": 0.30,
            "workload": 0.20,
        },
        whatif_comparison=[
            {
                "day": 4,
                "original_drift": 0.72,
                "whatif_drift": 0.50,
                "delta": -0.22,
                "original_risk_level": "high",
                "whatif_risk_level": "elevated",
            }
        ],
        reassigned_to="A2",
        feasibility_status="feasible_with_caution",
        feasibility_reasons=[],
        feasibility_warnings=["Receiver has a dependency-critical task on day 4."],
        receiver_drift_score=0.42,
        receiver_risk_level="elevated",
        workload_projected_ratio=1.43,
    )
    defaults.update(overrides)
    return BriefContext(**defaults)


# --- BriefContext accepts and stores the three new fields ---

def test_brief_context_stores_receiver_drift_score():
    ctx = _base_ctx_with_receiver()
    assert ctx.receiver_drift_score == 0.42


def test_brief_context_stores_receiver_risk_level():
    ctx = _base_ctx_with_receiver()
    assert ctx.receiver_risk_level == "elevated"


def test_brief_context_stores_workload_projected_ratio():
    ctx = _base_ctx_with_receiver()
    assert ctx.workload_projected_ratio == 1.43


def test_brief_context_receiver_fields_default_to_none():
    ctx = _base_ctx()  # uses the existing helper with no receiver fields
    assert ctx.receiver_drift_score is None
    assert ctx.receiver_risk_level is None
    assert ctx.workload_projected_ratio is None


# --- _build_brief_prompt includes the receiver block when fields are supplied ---

def test_build_brief_prompt_includes_receiver_block_when_supplied():
    ctx = _base_ctx_with_receiver()
    prompt = _build_brief_prompt(ctx)
    assert "Receiver fatigue state on reassignment day" in prompt
    assert "0.42" in prompt
    assert "elevated" in prompt
    assert "1.43" in prompt


def test_build_brief_prompt_receiver_block_labelled_as_deterministic_fact():
    ctx = _base_ctx_with_receiver()
    prompt = _build_brief_prompt(ctx)
    # The receiver block must be inside a [FACT — computed by deterministic model] section
    assert "[FACT — computed by deterministic model]" in prompt
    # Verify receiver values appear after that label (simple containment is sufficient)
    fact_idx = prompt.rfind("[FACT — computed by deterministic model]")
    receiver_idx = prompt.find("Receiver fatigue state on reassignment day")
    assert receiver_idx > fact_idx


def test_build_brief_prompt_receiver_block_instructs_no_recalculation():
    ctx = _base_ctx_with_receiver()
    prompt = _build_brief_prompt(ctx)
    assert "Do NOT use them to recalculate or override" in prompt


def test_build_brief_prompt_receiver_block_absent_when_fields_none():
    """When receiver fields are None, prompt must say NOT SUPPLIED not silently omit."""
    ctx = _base_ctx()  # no receiver fields
    prompt = _build_brief_prompt(ctx)
    assert "NOT SUPPLIED" in prompt
    assert "Receiver fatigue state on reassignment day" not in prompt


def test_build_brief_prompt_receiver_partial_fields_none_shows_not_supplied():
    """If only some receiver fields are present (e.g. drift but no ratio), treat as absent."""
    ctx = _base_ctx(receiver_drift_score=0.42, receiver_risk_level="elevated")
    # workload_projected_ratio is None → block should be NOT SUPPLIED
    prompt = _build_brief_prompt(ctx)
    assert "Receiver fatigue state on reassignment day" not in prompt
    assert "NOT SUPPLIED" in prompt


def test_build_brief_prompt_contains_tradeoff_instruction():
    """The prompt must instruct Granite on the ordered tradeoff explanation."""
    prompt = _build_brief_prompt(_base_ctx())
    assert "benefit to source astronaut" in prompt
    assert "receiver's current fatigue state" in prompt
    assert "receiver's projected workload ratio" in prompt


# --- _fallback_brief uses receiver fields in intervention_assessment ---

def test_fallback_brief_intervention_assessment_includes_receiver_tradeoff():
    ctx = _base_ctx_with_receiver()
    result = _fallback_brief(ctx)
    assert "0.42" in result.intervention_assessment
    assert "elevated" in result.intervention_assessment
    assert "1.43" in result.intervention_assessment


def test_fallback_brief_intervention_assessment_receiver_tradeoff_labelled_as_fact():
    ctx = _base_ctx_with_receiver()
    result = _fallback_brief(ctx)
    # The receiver tradeoff sentence must be prefixed with FACT:
    assert "FACT: Receiver fatigue state" in result.intervention_assessment


def test_fallback_brief_intervention_assessment_no_receiver_when_fields_none():
    """Without receiver fields the fallback must NOT mention receiver drift or ratio."""
    ctx = _base_ctx(
        whatif_comparison=[
            {"day": 4, "original_drift": 0.72, "whatif_drift": 0.55,
             "delta": -0.17, "original_risk_level": "high", "whatif_risk_level": "elevated"}
        ],
        reassigned_to="A2",
        feasibility_status="feasible",
    )
    result = _fallback_brief(ctx)
    # receiver-specific numbers must not appear
    assert "Receiver fatigue state" not in result.intervention_assessment


def test_fallback_brief_intervention_includes_source_delta_and_receiver_tradeoff_together():
    """Both the source drift reduction and receiver state must appear in the same assessment."""
    ctx = _base_ctx_with_receiver()
    result = _fallback_brief(ctx)
    ia = result.intervention_assessment
    # Source benefit — original and what-if drift values
    assert "0.72" in ia   # original drift
    assert "0.5" in ia    # what-if drift (Python renders 0.50 as 0.5)
    # Delta must appear as a positive magnitude, not as "reduction of -0.xxxx"
    assert "reduction of 0.2200" in ia or "increase of 0.2200" in ia
    assert "delta -0.2200" in ia   # signed delta still present in parenthetical
    # Receiver tradeoff
    assert "0.42" in ia
    assert "1.43" in ia


# --- Fix 1: intervention_assessment must cite BOTH receiver drift+risk AND workload ratio ---

def test_fallback_brief_intervention_cites_receiver_drift_and_risk_level():
    ctx = _base_ctx_with_receiver()
    result = _fallback_brief(ctx)
    ia = result.intervention_assessment
    assert "0.42" in ia               # receiver_drift_score
    assert "elevated" in ia           # receiver_risk_level


def test_fallback_brief_intervention_cites_workload_projected_ratio():
    ctx = _base_ctx_with_receiver()
    result = _fallback_brief(ctx)
    ia = result.intervention_assessment
    assert "1.43" in ia               # workload_projected_ratio


# --- Fix 2: wording must use "prototype workload threshold", not "validated capacity" ---

def test_feasibility_workload_reason_uses_prototype_workload_threshold_wording():
    """The workload fail reason must say 'prototype workload threshold', not 'validated capacity limit'."""
    from feasibility import check_workload
    from simulator import MissionDayRecord
    from drift import DriftResult

    drift = DriftResult(
        reaction_time_score=0.0, sleep_debt_score=0.0, circadian_score=0.0,
        workload_score=0.0, drift_score=0.3, updated_sleep_debt_hours=0.0,
        risk_level="nominal",
    )
    record = MissionDayRecord(
        day=3, astronaut_id="A2", hours_slept=7.0, pvt_lapses=3,
        minutes_phase_shift=0.0, task_load=8.0, rolling_avg_task_load=8.0, drift=drift,
    )
    result = check_workload(record, task_load_delta=-5.0)  # (8+5)/8 = 1.625 > 1.5 → fails
    assert result.passed is False
    assert "prototype workload threshold" in result.reason
    assert "not an operationally validated capacity limit" in result.reason
    # The old phrase must be gone
    assert "not a validated capacity limit" not in result.reason


def test_build_brief_prompt_intervention_assessment_instruction_forbids_negative_reduction():
    """The JSON output instruction for intervention_assessment must forbid 'a reduction of -X'."""
    prompt = _build_brief_prompt(_base_ctx())
    assert "never 'a reduction of -X.XXXX'" in prompt or "never" in prompt and "reduction of -" in prompt


def test_build_brief_prompt_intervention_assessment_requires_prototype_workload_label():
    """The JSON output instruction must require labelling the workload ratio as a prototype threshold."""
    prompt = _build_brief_prompt(_base_ctx())
    assert "prototype workload threshold" in prompt


def test_build_brief_prompt_receiver_tradeoff_requires_explicit_drift_and_ratio():
    """The tradeoff instruction must explicitly require stating both drift score/risk AND workload ratio."""
    prompt = _build_brief_prompt(_base_ctx())
    assert "receiver drift score" in prompt or "drift score and risk level" in prompt
    assert "projected workload ratio" in prompt


# --- Fix 3: fallback delta rendering ---

def test_fallback_brief_delta_not_rendered_as_reduction_of_negative():
    """'a reduction of -X.XXXX' must never appear — magnitude only for the reduction word."""
    ctx = _base_ctx(
        whatif_comparison=[
            {"day": 4, "original_drift": 0.72, "whatif_drift": 0.6621,
             "delta": -0.0579, "original_risk_level": "high", "whatif_risk_level": "high"}
        ],
        reassigned_to="A2",
    )
    result = _fallback_brief(ctx)
    ia = result.intervention_assessment
    assert "reduction of -" not in ia    # the broken pattern must be absent
    assert "increase of -" not in ia     # same check for increases (sign must never follow the word)
    # The magnitude should appear positively
    assert "0.0579" in ia


def test_fallback_brief_delta_positive_change_says_increase():
    """A positive delta (drift goes up) must say 'an increase of X.XXXX'."""
    ctx = _base_ctx(
        whatif_comparison=[
            {"day": 4, "original_drift": 0.60, "whatif_drift": 0.65,
             "delta": 0.05, "original_risk_level": "elevated", "whatif_risk_level": "elevated"}
        ],
    )
    result = _fallback_brief(ctx)
    ia = result.intervention_assessment
    assert "increase of 0.0500" in ia
    assert "delta +0.0500" in ia


def test_fallback_brief_delta_negative_change_says_reduction():
    """A negative delta (drift goes down) must say 'a reduction of X.XXXX' (no minus sign)."""
    ctx = _base_ctx(
        whatif_comparison=[
            {"day": 4, "original_drift": 0.72, "whatif_drift": 0.55,
             "delta": -0.17, "original_risk_level": "high", "whatif_risk_level": "elevated"}
        ],
    )
    result = _fallback_brief(ctx)
    ia = result.intervention_assessment
    assert "reduction of 0.1700" in ia
    assert "delta -0.1700" in ia
