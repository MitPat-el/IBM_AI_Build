"""
tests/test_ai.py — Tests for the AI explanation endpoints.

These tests verify:
  - The AI endpoint never changes the numerical score
  - All three audience modes are accepted
  - Graceful stub response when Watson credentials are missing
  - Intervention explanation endpoint works
  - Disclaimer is always present
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

FATIGUE_RESULT = {
    "astronaut_id": "A01",
    "timestamp": "2025-01-15T14:30:00Z",
    "mission_day": 4,
    "fatigue_score": 78.0,
    "risk_level": "HIGH",
    "signal_breakdown": [
        {"signal": "pvt_risk",       "raw_value": 80.0, "weight": 0.30, "weighted_contribution": 24.0},
        {"signal": "sleep_risk",     "raw_value": 85.0, "weight": 0.30, "weighted_contribution": 25.5},
        {"signal": "circadian_risk", "raw_value": 70.0, "weight": 0.20, "weighted_contribution": 14.0},
        {"signal": "workload_risk",  "raw_value": 72.0, "weight": 0.20, "weighted_contribution": 14.4},
    ],
    "top_contributing_factors": ["sleep_risk", "pvt_risk", "workload_risk"],
    "trend": "RISING",
    "data_quality": {"score": 75.0, "level": "HIGH", "notes": []},
    "disclaimer": "This score is a decision-support tool for mission planning only.",
}

WHAT_IF_RESULT = {
    "before_fatigue_score": 78.0,
    "after_fatigue_score": 45.0,
    "before_mission_risk": 82.0,
    "after_mission_risk": 61.0,
    "risk_change": -21.0,
    "intervention": {
        "intervention_type": "REASSIGN_TASK",
        "task_id": "T-EVA-01",
        "from_astronaut_id": "A01",
        "to_astronaut_id": "A02",
    },
    "feasible": True,
    "constraint_violations": [],
    "explanation_data": {
        "intervention_type": "REASSIGN_TASK",
        "task_id": "T-EVA-01",
        "task_name": "EVA Solar Panel Repair",
        "before_fatigue_score": 78.0,
        "after_fatigue_score": 45.0,
        "before_mission_risk": 82.0,
        "after_mission_risk": 61.0,
        "risk_change": -21.0,
        "feasible": True,
        "constraint_violations": [],
    },
    "disclaimer": "This simulation result does not modify the real mission state.",
}


class TestAIExplain:

    def test_explain_returns_200(self):
        r = client.post("/ai/explain", json={"result": FATIGUE_RESULT})
        assert r.status_code == 200

    def test_explain_response_fields(self):
        r = client.post("/ai/explain", json={"result": FATIGUE_RESULT})
        body = r.json()
        assert "explanation" in body
        assert "astronaut_id" in body
        assert "fatigue_score" in body
        assert "risk_level" in body
        assert "disclaimer" in body

    def test_ai_does_not_change_score(self):
        """Critical: the score in the response must equal the input score."""
        r = client.post("/ai/explain", json={"result": FATIGUE_RESULT})
        assert r.json()["fatigue_score"] == FATIGUE_RESULT["fatigue_score"]

    def test_ai_does_not_change_risk_level(self):
        r = client.post("/ai/explain", json={"result": FATIGUE_RESULT})
        assert r.json()["risk_level"] == FATIGUE_RESULT["risk_level"]

    def test_stub_explanation_present_without_credentials(self):
        """Without WATSONX_API_KEY env var, a stub message must be returned."""
        r = client.post("/ai/explain", json={"result": FATIGUE_RESULT})
        explanation = r.json()["explanation"]
        assert len(explanation) > 0
        # Stub will say credentials not configured
        assert "watsonx" in explanation.lower() or "explanation" in explanation.lower()

    def test_astronaut_audience_mode(self):
        r = client.post("/ai/explain", json={
            "result": FATIGUE_RESULT, "audience": "ASTRONAUT"
        })
        assert r.status_code == 200
        assert r.json()["audience"] == "ASTRONAUT"

    def test_mission_team_audience_mode(self):
        r = client.post("/ai/explain", json={
            "result": FATIGUE_RESULT, "audience": "MISSION_TEAM"
        })
        assert r.status_code == 200
        assert r.json()["audience"] == "MISSION_TEAM"

    def test_flight_surgeon_audience_mode(self):
        r = client.post("/ai/explain", json={
            "result": FATIGUE_RESULT, "audience": "FLIGHT_SURGEON"
        })
        assert r.status_code == 200
        assert r.json()["audience"] == "FLIGHT_SURGEON"

    def test_invalid_audience_returns_422(self):
        r = client.post("/ai/explain", json={
            "result": FATIGUE_RESULT, "audience": "INVALID_MODE"
        })
        assert r.status_code == 422

    def test_disclaimer_always_present(self):
        r = client.post("/ai/explain", json={"result": FATIGUE_RESULT})
        assert len(r.json()["disclaimer"]) > 0

    def test_task_context_passed_through(self):
        r = client.post("/ai/explain", json={
            "result": FATIGUE_RESULT,
            "audience": "MISSION_TEAM",
            "task_context": "EVA Solar Panel Repair — criticality 5/5",
        })
        assert r.status_code == 200


class TestAIExplainIntervention:

    def test_explain_intervention_returns_200(self):
        r = client.post("/ai/explain-intervention", json={"what_if_result": WHAT_IF_RESULT})
        assert r.status_code == 200

    def test_explain_intervention_fields(self):
        r = client.post("/ai/explain-intervention", json={"what_if_result": WHAT_IF_RESULT})
        body = r.json()
        assert "summary" in body
        assert "why_it_helps" in body
        assert "limitations" in body
        assert "human_review_required" in body
        assert "disclaimer" in body

    def test_human_review_required_always_true(self):
        r = client.post("/ai/explain-intervention", json={"what_if_result": WHAT_IF_RESULT})
        assert r.json()["human_review_required"] is True

    def test_ai_does_not_change_risk_scores(self):
        """The explain-intervention endpoint must not alter the WhatIfResult scores."""
        r = client.post("/ai/explain-intervention", json={"what_if_result": WHAT_IF_RESULT})
        # Response fields are explanation only — no score fields to override
        body = r.json()
        assert "before_mission_risk" not in body  # explanation response, not a WhatIfResult
        assert "after_mission_risk"  not in body

    def test_stub_response_without_credentials(self):
        r = client.post("/ai/explain-intervention", json={"what_if_result": WHAT_IF_RESULT})
        summary = r.json()["summary"]
        assert len(summary) > 0
