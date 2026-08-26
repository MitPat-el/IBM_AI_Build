"""
tests/test_history.py — Tests for the history storage and retrieval API.
"""
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

BASE_RECORD = {
    "mission_id": "ARTEMIS-TEST",
    "astronaut_id": "A01",
    "mission_day": 1,
    "timestamp": "2025-01-15T12:00:00Z",
    "pvt_risk": 50.0, "sleep_risk": 55.0,
    "circadian_risk": 40.0, "workload_risk": 45.0,
    "fatigue_score": 48.5,
    "fatigue_level": "MODERATE",
    "trend": "STABLE",
    "mission_risk": 55.0,
    "mission_risk_level": "MODERATE",
    "task_id": "T-EVA-01",
    "data_quality_score": 75.0,
    "data_quality_level": "HIGH",
}


class TestHistoryRecord:

    def test_record_returns_200(self):
        r = client.post("/history/record", json=BASE_RECORD)
        assert r.status_code == 200

    def test_record_echoes_fields(self):
        r = client.post("/history/record", json=BASE_RECORD)
        body = r.json()
        assert body["astronaut_id"] == "A01"
        assert body["mission_id"] == "ARTEMIS-TEST"
        assert body["fatigue_score"] == 48.5
        assert body["fatigue_level"] == "MODERATE"
        assert "id" in body

    def test_record_missing_required_field_returns_422(self):
        bad = {k: v for k, v in BASE_RECORD.items() if k != "fatigue_score"}
        r = client.post("/history/record", json=bad)
        assert r.status_code == 422


class TestHistoryRetrieval:

    def setup_method(self):
        """Seed two records for retrieval tests."""
        client.post("/history/record", json={**BASE_RECORD, "astronaut_id": "A01-HIST", "mission_day": 1})
        client.post("/history/record", json={**BASE_RECORD, "astronaut_id": "A01-HIST", "mission_day": 2,
                                              "fatigue_score": 62.0, "fatigue_level": "MODERATE"})

    def test_get_astronaut_history_returns_list(self):
        r = client.get("/history/astronaut/A01-HIST")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_get_astronaut_history_has_entries(self):
        r = client.get("/history/astronaut/A01-HIST")
        assert len(r.json()) >= 2

    def test_get_astronaut_history_ordered_by_timestamp(self):
        r = client.get("/history/astronaut/A01-HIST")
        entries = r.json()
        timestamps = [e["timestamp"] for e in entries]
        assert timestamps == sorted(timestamps)

    def test_get_mission_history_returns_list(self):
        r = client.get("/history/mission/ARTEMIS-TEST")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_filter_by_mission_day(self):
        r = client.get("/history/astronaut/A01-HIST?mission_day=1")
        entries = r.json()
        assert all(e["mission_day"] == 1 for e in entries)

    def test_unknown_astronaut_returns_empty_list(self):
        r = client.get("/history/astronaut/NOBODY-999")
        assert r.status_code == 200
        assert r.json() == []

    def test_timeline_entry_has_required_fields(self):
        r = client.get("/history/astronaut/A01-HIST")
        entry = r.json()[0]
        for field in ["id", "astronaut_id", "mission_id", "fatigue_score",
                      "fatigue_level", "timestamp", "pvt_risk", "sleep_risk"]:
            assert field in entry
