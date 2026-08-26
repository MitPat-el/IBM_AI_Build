"""
tests/test_api.py — Integration tests for the FastAPI routes.

Uses FastAPI's built-in TestClient (backed by httpx).
Run with:
    pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

def test_root_returns_ok():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_returns_healthy():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# POST /fatigue/calculate — happy path
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "astronaut_id": "CDR-001",
    "timestamp": "2025-01-15T14:30:00Z",
    "pvt_risk": 72.0,
    "sleep_risk": 65.0,
    "circadian_risk": 80.0,
    "workload_risk": 55.0,
}


def test_calculate_returns_200():
    response = client.post("/fatigue/calculate", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_calculate_response_fields():
    response = client.post("/fatigue/calculate", json=VALID_PAYLOAD)
    body = response.json()
    assert body["astronaut_id"] == "CDR-001"
    assert "fatigue_score" in body
    assert "risk_level" in body
    assert "signal_breakdown" in body
    assert "top_contributing_factors" in body
    assert "disclaimer" in body


def test_calculate_score_value():
    # 72*0.30 + 65*0.30 + 80*0.20 + 55*0.20 = 68.1
    response = client.post("/fatigue/calculate", json=VALID_PAYLOAD)
    assert response.json()["fatigue_score"] == pytest.approx(68.1, abs=0.01)


def test_calculate_risk_level_moderate():
    response = client.post("/fatigue/calculate", json=VALID_PAYLOAD)
    assert response.json()["risk_level"] == "MODERATE"


def test_calculate_disclaimer_always_present():
    response = client.post("/fatigue/calculate", json=VALID_PAYLOAD)
    assert len(response.json()["disclaimer"]) > 0


# ---------------------------------------------------------------------------
# POST /fatigue/calculate — validation errors
# ---------------------------------------------------------------------------

def test_missing_field_returns_422():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "pvt_risk"}
    response = client.post("/fatigue/calculate", json=payload)
    assert response.status_code == 422


def test_risk_value_above_100_returns_422():
    payload = {**VALID_PAYLOAD, "pvt_risk": 101.0}
    response = client.post("/fatigue/calculate", json=payload)
    assert response.status_code == 422


def test_risk_value_below_0_returns_422():
    payload = {**VALID_PAYLOAD, "sleep_risk": -1.0}
    response = client.post("/fatigue/calculate", json=payload)
    assert response.status_code == 422


def test_blank_astronaut_id_returns_422():
    payload = {**VALID_PAYLOAD, "astronaut_id": "   "}
    response = client.post("/fatigue/calculate", json=payload)
    assert response.status_code == 422


def test_invalid_timestamp_returns_422():
    payload = {**VALID_PAYLOAD, "timestamp": "not-a-date"}
    response = client.post("/fatigue/calculate", json=payload)
    assert response.status_code == 422
