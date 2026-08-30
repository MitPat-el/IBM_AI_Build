"""
Unit tests for the Ollama/IBM Granite provider path in bob.py.

All httpx calls (both the Ollama availability probe and the generate
endpoint) are mocked — no real network calls are made during pytest.

Provider order under test:
  1. watsonx  (absent — credentials set to None in all tests here)
  2. Ollama   ← this file tests this branch exhaustively
  3. deterministic fallback
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from bob import (
    BriefContext,
    MissionDecisionBrief,
    _is_ollama_available,
    _call_ollama_brief,
    generate_mission_brief,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _valid_ollama_response(**overrides) -> dict:
    """
    Simulates the dict returned by httpx.Response.json() for an Ollama
    /api/generate call.  The generated text lives in the "response" key.
    """
    brief = {
        "astronaut_message": "You are doing well, stay on schedule.",
        "executive_summary": "Drift is high on day 4 driven by reaction time.",
        "primary_drivers": ["reaction time: 0.600", "sleep debt: 0.550"],
        "mission_implications": ["High fatigue risk on precision tasks."],
        "intervention_assessment": "No What-If context supplied.",
        "recommended_actions": ["ACTION OPTION: Reassign high-precision tasks."],
        "uncertainties": ["Prototype thresholds only."],
        "human_review_required": True,
    }
    brief.update(overrides)
    return {"response": json.dumps(brief)}


def _mock_ollama_get_ok():
    """Returns a mock httpx.get response that passes the is_success probe."""
    mock = MagicMock()
    mock.is_success = True
    return mock


def _mock_ollama_post(response_json=None, raise_exc=None):
    """Returns a side_effect function for httpx.post that simulates Ollama."""
    def side_effect(url, **kwargs):
        if raise_exc:
            raise raise_exc
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = response_json or _valid_ollama_response()
        return mock_resp
    return side_effect


# ---------------------------------------------------------------------------
# Fixture: watsonx credentials always absent for this test file
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_watsonx_creds():
    """Ensure watsonx is never attempted in these tests."""
    import bob
    original_key = bob.WATSONX_API_KEY
    original_pid = bob.WATSONX_PROJECT_ID
    bob.WATSONX_API_KEY = None
    bob.WATSONX_PROJECT_ID = None
    yield
    bob.WATSONX_API_KEY = original_key
    bob.WATSONX_PROJECT_ID = original_pid


# ---------------------------------------------------------------------------
# _is_ollama_available
# ---------------------------------------------------------------------------

def test_is_ollama_available_returns_true_when_server_responds():
    with patch("bob.httpx.get", return_value=_mock_ollama_get_ok()):
        assert _is_ollama_available() is True


def test_is_ollama_available_returns_false_on_connection_error():
    import httpx
    with patch("bob.httpx.get", side_effect=httpx.ConnectError("refused")):
        assert _is_ollama_available() is False


def test_is_ollama_available_returns_false_on_timeout():
    import httpx
    with patch("bob.httpx.get", side_effect=httpx.TimeoutException("timeout")):
        assert _is_ollama_available() is False


def test_is_ollama_available_returns_false_when_response_not_success():
    mock = MagicMock()
    mock.is_success = False
    with patch("bob.httpx.get", return_value=mock):
        assert _is_ollama_available() is False


# ---------------------------------------------------------------------------
# _call_ollama_brief
# ---------------------------------------------------------------------------

def test_call_ollama_brief_returns_response_string():
    with patch("bob.httpx.post", side_effect=_mock_ollama_post()):
        text = _call_ollama_brief(_base_ctx())
    # Should be the JSON string embedded in the "response" key
    parsed = json.loads(text)
    assert "astronaut_message" in parsed


def test_call_ollama_brief_sends_json_format_flag():
    """Verify that "format": "json" is included in the request body."""
    captured_kwargs = {}

    def capture_post(url, **kwargs):
        captured_kwargs.update(kwargs)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _valid_ollama_response()
        return mock_resp

    with patch("bob.httpx.post", side_effect=capture_post):
        _call_ollama_brief(_base_ctx())

    assert captured_kwargs.get("json", {}).get("format") == "json"
    assert captured_kwargs.get("json", {}).get("stream") is False


def test_call_ollama_brief_sends_correct_model():
    captured_kwargs = {}

    def capture_post(url, **kwargs):
        captured_kwargs.update(kwargs)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _valid_ollama_response()
        return mock_resp

    import bob
    with patch("bob.httpx.post", side_effect=capture_post):
        _call_ollama_brief(_base_ctx())

    assert captured_kwargs.get("json", {}).get("model") == bob.OLLAMA_MODEL


# ---------------------------------------------------------------------------
# generate_mission_brief — Ollama success path
# ---------------------------------------------------------------------------

def test_generate_mission_brief_ollama_success_returns_ollama_granite():
    with patch("bob.httpx.get", return_value=_mock_ollama_get_ok()), \
         patch("bob.httpx.post", side_effect=_mock_ollama_post()):
        result = generate_mission_brief(_base_ctx())
    assert result.source == "ollama_granite"
    assert isinstance(result, MissionDecisionBrief)


def test_generate_mission_brief_ollama_human_review_forced_true_even_if_model_false():
    """Python must force human_review_required=True regardless of model output."""
    resp = _valid_ollama_response(human_review_required=False)
    with patch("bob.httpx.get", return_value=_mock_ollama_get_ok()), \
         patch("bob.httpx.post", side_effect=_mock_ollama_post(response_json=resp)):
        result = generate_mission_brief(_base_ctx())
    assert result.human_review_required is True
    assert result.source == "ollama_granite"


def test_generate_mission_brief_ollama_all_fields_populated():
    with patch("bob.httpx.get", return_value=_mock_ollama_get_ok()), \
         patch("bob.httpx.post", side_effect=_mock_ollama_post()):
        result = generate_mission_brief(_base_ctx())
    assert result.astronaut_message
    assert result.executive_summary
    assert isinstance(result.primary_drivers, list)
    assert isinstance(result.mission_implications, list)
    assert result.intervention_assessment
    assert isinstance(result.recommended_actions, list)
    assert isinstance(result.uncertainties, list)


def test_generate_mission_brief_ollama_astronaut_message_value():
    with patch("bob.httpx.get", return_value=_mock_ollama_get_ok()), \
         patch("bob.httpx.post", side_effect=_mock_ollama_post()):
        result = generate_mission_brief(_base_ctx())
    assert result.astronaut_message == "You are doing well, stay on schedule."


# ---------------------------------------------------------------------------
# generate_mission_brief — Ollama error paths
# ---------------------------------------------------------------------------

def test_generate_mission_brief_ollama_http_error_returns_ollama_error(capsys):
    import httpx
    with patch("bob.httpx.get", return_value=_mock_ollama_get_ok()), \
         patch("bob.httpx.post", side_effect=_mock_ollama_post(
             raise_exc=httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())
         )):
        result = generate_mission_brief(_base_ctx())
    assert result.source == "fallback_template:ollama_error"
    assert result.human_review_required is True
    captured = capsys.readouterr()
    assert "[bob.py fallback]" in captured.err


def test_generate_mission_brief_ollama_timeout_returns_ollama_error(capsys):
    import httpx
    with patch("bob.httpx.get", return_value=_mock_ollama_get_ok()), \
         patch("bob.httpx.post", side_effect=_mock_ollama_post(
             raise_exc=httpx.TimeoutException("timed out")
         )):
        result = generate_mission_brief(_base_ctx())
    assert result.source == "fallback_template:ollama_error"
    captured = capsys.readouterr()
    assert "[bob.py fallback]" in captured.err


def test_generate_mission_brief_ollama_malformed_json_returns_ollama_error(capsys):
    bad_resp = {"response": "not valid JSON {{{{"}
    with patch("bob.httpx.get", return_value=_mock_ollama_get_ok()), \
         patch("bob.httpx.post", side_effect=_mock_ollama_post(response_json=bad_resp)):
        result = generate_mission_brief(_base_ctx())
    assert result.source == "fallback_template:ollama_error"
    captured = capsys.readouterr()
    assert "[bob.py fallback]" in captured.err


def test_generate_mission_brief_ollama_missing_required_field_returns_ollama_error(capsys):
    """Ollama returns JSON missing 'executive_summary'."""
    brief = {
        "astronaut_message": "OK",
        "primary_drivers": [],
        "mission_implications": [],
        "intervention_assessment": "none",
        "recommended_actions": [],
        "uncertainties": [],
        "human_review_required": True,
        # executive_summary absent
    }
    bad_resp = {"response": json.dumps(brief)}
    with patch("bob.httpx.get", return_value=_mock_ollama_get_ok()), \
         patch("bob.httpx.post", side_effect=_mock_ollama_post(response_json=bad_resp)):
        result = generate_mission_brief(_base_ctx())
    assert result.source == "fallback_template:ollama_error"
    captured = capsys.readouterr()
    assert "[bob.py fallback]" in captured.err


def test_generate_mission_brief_ollama_list_field_wrong_type_returns_ollama_error():
    """Ollama returns primary_drivers as a string instead of a list."""
    bad_resp = {"response": json.dumps({
        "astronaut_message": "OK",
        "executive_summary": "OK",
        "primary_drivers": "reaction time: 0.600",  # should be a list
        "mission_implications": [],
        "intervention_assessment": "none",
        "recommended_actions": [],
        "uncertainties": [],
        "human_review_required": True,
    })}
    with patch("bob.httpx.get", return_value=_mock_ollama_get_ok()), \
         patch("bob.httpx.post", side_effect=_mock_ollama_post(response_json=bad_resp)):
        result = generate_mission_brief(_base_ctx())
    assert result.source == "fallback_template:ollama_error"


# ---------------------------------------------------------------------------
# generate_mission_brief — Ollama unavailable → deterministic fallback
# ---------------------------------------------------------------------------

def test_generate_mission_brief_ollama_unreachable_falls_to_deterministic():
    """When Ollama probe fails, skip Ollama and use deterministic fallback."""
    with patch("bob.httpx.get", side_effect=Exception("connection refused")):
        result = generate_mission_brief(_base_ctx())
    assert result.source == "fallback_template:no_credentials"
    assert result.human_review_required is True
