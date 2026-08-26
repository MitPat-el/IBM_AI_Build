"""
tests/test_fatigue_extended.py — Tests for trend, data quality, and top-3 factors.
"""
import pytest
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from app.main import app
from app.models.fatigue import FatigueInput
from app.services.fatigue_service import assess_fatigue, _calculate_trend
from app.services.data_quality_service import score_data_quality

client = TestClient(app)

TS = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)

def make_input(**overrides) -> FatigueInput:
    defaults = dict(
        astronaut_id="A01", timestamp=TS,
        pvt_risk=0.0, sleep_risk=0.0, circadian_risk=0.0, workload_risk=0.0,
    )
    defaults.update(overrides)
    return FatigueInput(**defaults)


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------
class TestTrend:
    def test_no_previous_gives_unknown(self):
        assert _calculate_trend(50.0, None) == "UNKNOWN"

    def test_stable_within_threshold(self):
        assert _calculate_trend(50.0, 47.0) == "STABLE"

    def test_rising(self):
        assert _calculate_trend(60.0, 50.0) == "RISING"

    def test_rapidly_rising(self):
        assert _calculate_trend(80.0, 60.0) == "RAPIDLY_RISING"

    def test_falling(self):
        assert _calculate_trend(40.0, 55.0) == "FALLING"

    def test_trend_in_full_pipeline(self):
        # pvt=50,sleep=50,circ=0,wl=0 → score=30; delta from 20 = +10 → RISING
        result = assess_fatigue(make_input(pvt_risk=50, sleep_risk=50, previous_fatigue_score=20.0))
        assert result.trend == "RISING"


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------
class TestDataQuality:
    def test_all_data_present_gives_high(self):
        payload = make_input(
            pvt_risk=50, sleep_risk=50, circadian_risk=50, workload_risk=50,
            baseline_available=True, task_info_available=True,
            previous_fatigue_score=40.0,
        )
        dq = score_data_quality(payload)
        assert dq.level == "HIGH"
        assert dq.score >= 80

    def test_missing_baseline_reduces_score(self):
        payload = make_input(pvt_risk=50, sleep_risk=50, baseline_available=False)
        dq = score_data_quality(payload)
        assert dq.score < 100

    def test_missing_task_info_reduces_score(self):
        payload = make_input(pvt_risk=50, sleep_risk=50, task_info_available=False)
        dq = score_data_quality(payload)
        assert dq.score < 100

    def test_no_previous_score_reduces_score(self):
        payload = make_input(pvt_risk=50, sleep_risk=50, previous_fatigue_score=None)
        dq = score_data_quality(payload)
        assert dq.score < 100

    def test_stale_reading_reduces_score(self):
        old_ts = datetime.now(timezone.utc) - timedelta(hours=3)
        payload = make_input(timestamp=old_ts, pvt_risk=50)
        dq = score_data_quality(payload)
        assert dq.score < 100
        assert any("old" in n.lower() or "freshness" in n.lower() for n in dq.notes)

    def test_low_confidence_note_present_when_score_low(self):
        old_ts = datetime.now(timezone.utc) - timedelta(hours=3)
        payload = make_input(
            timestamp=old_ts,
            pvt_risk=50,
            baseline_available=False,
            task_info_available=False,
        )
        dq = score_data_quality(payload)
        assert dq.level in ("MEDIUM", "LOW")


# ---------------------------------------------------------------------------
# Top-3 factors
# ---------------------------------------------------------------------------
class TestTopFactors:
    def test_returns_at_most_3_factors(self):
        result = assess_fatigue(make_input(pvt_risk=10, sleep_risk=90, circadian_risk=50, workload_risk=80))
        assert len(result.top_contributing_factors) <= 3

    def test_highest_contributor_is_first(self):
        result = assess_fatigue(make_input(pvt_risk=10, sleep_risk=90, circadian_risk=50, workload_risk=80))
        assert result.top_contributing_factors[0] == "sleep_risk"


# ---------------------------------------------------------------------------
# API response includes new fields
# ---------------------------------------------------------------------------
class TestFatigueAPIExtended:
    PAYLOAD = {
        "astronaut_id": "A01",
        "timestamp": "2025-01-15T14:30:00Z",
        "pvt_risk": 72.0, "sleep_risk": 65.0,
        "circadian_risk": 80.0, "workload_risk": 55.0,
    }

    def test_trend_field_present(self):
        r = client.post("/fatigue/calculate", json=self.PAYLOAD)
        assert "trend" in r.json()

    def test_data_quality_field_present(self):
        r = client.post("/fatigue/calculate", json=self.PAYLOAD)
        body = r.json()
        assert "data_quality" in body
        assert "score" in body["data_quality"]
        assert "level" in body["data_quality"]

    def test_trend_unknown_without_previous(self):
        r = client.post("/fatigue/calculate", json=self.PAYLOAD)
        assert r.json()["trend"] == "UNKNOWN"

    def test_trend_rising_with_previous(self):
        payload = {**self.PAYLOAD, "previous_fatigue_score": 30.0}
        r = client.post("/fatigue/calculate", json=payload)
        assert r.json()["trend"] in ("RISING", "RAPIDLY_RISING")

    def test_mission_day_echoed(self):
        payload = {**self.PAYLOAD, "mission_day": 3}
        r = client.post("/fatigue/calculate", json=payload)
        assert r.json()["mission_day"] == 3
