"""
tests/test_scoring.py — Unit tests for the deterministic scoring engine.

Run with:
    pytest tests/ -v

These tests cover:
  - Correct weighted-sum calculation
  - Correct risk-level classification for all four bands
  - Top contributing factors are ranked highest-first
  - Boundary values (0, 100)
  - All-zero and all-max inputs
"""

import pytest
from datetime import datetime, timezone
from app.models.fatigue import FatigueInput
from app.engine.scoring import calculate_fatigue_score, _classify_risk_level


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_input(**overrides) -> FatigueInput:
    """Build a FatigueInput with sensible defaults, accepting overrides."""
    defaults = dict(
        astronaut_id="TEST-001",
        timestamp=datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
        pvt_risk=0.0,
        sleep_risk=0.0,
        circadian_risk=0.0,
        workload_risk=0.0,
    )
    defaults.update(overrides)
    return FatigueInput(**defaults)


# ---------------------------------------------------------------------------
# Score calculation
# ---------------------------------------------------------------------------

class TestScoreCalculation:

    def test_all_zero_inputs_give_zero_score(self):
        result = calculate_fatigue_score(make_input())
        assert result.fatigue_score == 0.0

    def test_all_max_inputs_give_100_score(self):
        result = calculate_fatigue_score(
            make_input(pvt_risk=100, sleep_risk=100, circadian_risk=100, workload_risk=100)
        )
        assert result.fatigue_score == 100.0

    def test_known_weighted_sum(self):
        # pvt=72, sleep=65, circadian=80, workload=55
        # Score = 72*0.30 + 65*0.30 + 80*0.20 + 55*0.20
        #       = 21.6 + 19.5 + 16.0 + 11.0 = 68.1
        result = calculate_fatigue_score(
            make_input(pvt_risk=72, sleep_risk=65, circadian_risk=80, workload_risk=55)
        )
        assert result.fatigue_score == pytest.approx(68.1, abs=0.01)

    def test_single_signal_only(self):
        # Only pvt_risk=100, others=0 → score = 100*0.30 = 30.0
        result = calculate_fatigue_score(make_input(pvt_risk=100))
        assert result.fatigue_score == pytest.approx(30.0, abs=0.01)

    def test_breakdown_contributions_sum_to_score(self):
        result = calculate_fatigue_score(
            make_input(pvt_risk=50, sleep_risk=60, circadian_risk=70, workload_risk=80)
        )
        total = sum(b.weighted_contribution for b in result.signal_breakdown)
        assert total == pytest.approx(result.fatigue_score, abs=0.01)

    def test_astronaut_id_and_timestamp_echoed(self):
        ts = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        result = calculate_fatigue_score(make_input(astronaut_id="CDR-007", timestamp=ts))
        assert result.astronaut_id == "CDR-007"
        assert result.timestamp == ts


# ---------------------------------------------------------------------------
# Risk level classification
# ---------------------------------------------------------------------------

class TestRiskLevelClassification:

    @pytest.mark.parametrize("score,expected", [
        (0.0,  "LOW"),
        (20.0, "LOW"),
        (39.9, "LOW"),
        (40.0, "MODERATE"),
        (55.0, "MODERATE"),
        (69.9, "MODERATE"),
        (70.0, "HIGH"),
        (77.5, "HIGH"),
        (84.9, "HIGH"),
        (85.0, "CRITICAL"),
        (92.0, "CRITICAL"),
        (100.0,"CRITICAL"),
    ])
    def test_thresholds(self, score, expected):
        assert _classify_risk_level(score) == expected

    def test_full_pipeline_risk_level_moderate(self):
        # Score ≈ 68.1 → MODERATE
        result = calculate_fatigue_score(
            make_input(pvt_risk=72, sleep_risk=65, circadian_risk=80, workload_risk=55)
        )
        assert result.risk_level == "MODERATE"

    def test_full_pipeline_risk_level_critical(self):
        result = calculate_fatigue_score(
            make_input(pvt_risk=100, sleep_risk=100, circadian_risk=100, workload_risk=100)
        )
        assert result.risk_level == "CRITICAL"


# ---------------------------------------------------------------------------
# Top contributing factors
# ---------------------------------------------------------------------------

class TestTopContributingFactors:

    def test_factors_ranked_highest_first(self):
        # pvt=10 (contrib=3), sleep=90 (contrib=27), circadian=50 (contrib=10), workload=80 (contrib=16)
        # Expected order: sleep_risk, workload_risk, circadian_risk, pvt_risk
        result = calculate_fatigue_score(
            make_input(pvt_risk=10, sleep_risk=90, circadian_risk=50, workload_risk=80)
        )
        assert result.top_contributing_factors[0] == "sleep_risk"
        assert result.top_contributing_factors[-1] == "pvt_risk"

    def test_all_factors_present(self):
        result = calculate_fatigue_score(
            make_input(pvt_risk=10, sleep_risk=20, circadian_risk=30, workload_risk=40)
        )
        assert set(result.top_contributing_factors) == {
            "pvt_risk", "sleep_risk", "circadian_risk", "workload_risk"
        }


# ---------------------------------------------------------------------------
# Signal breakdown fields
# ---------------------------------------------------------------------------

class TestSignalBreakdown:

    def test_breakdown_has_all_signals(self):
        result = calculate_fatigue_score(
            make_input(pvt_risk=50, sleep_risk=50, circadian_risk=50, workload_risk=50)
        )
        names = {b.signal for b in result.signal_breakdown}
        assert names == {"pvt_risk", "sleep_risk", "circadian_risk", "workload_risk"}

    def test_breakdown_raw_values_match_input(self):
        result = calculate_fatigue_score(
            make_input(pvt_risk=33, sleep_risk=44, circadian_risk=55, workload_risk=66)
        )
        raw = {b.signal: b.raw_value for b in result.signal_breakdown}
        assert raw["pvt_risk"] == 33
        assert raw["sleep_risk"] == 44
        assert raw["circadian_risk"] == 55
        assert raw["workload_risk"] == 66
