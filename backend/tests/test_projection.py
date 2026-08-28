"""Tests for projection.py -- Mission Risk Map and What-If Simulator."""

import pytest

from simulator import AstronautProfile, simulate_mission
from projection import project_mission_risk, mission_risk_summary, simulate_intervention


@pytest.fixture
def crew_records():
    crew = [
        AstronautProfile(astronaut_id="A1", name="Chen", baseline_pvt_lapses=2.5, seed=1),
        AstronautProfile(astronaut_id="A2", name="Okafor", baseline_pvt_lapses=3.5, resilience=1.15, seed=2),
    ]
    return {p.astronaut_id: simulate_mission(p, num_days=6) for p in crew}


def test_mission_risk_map_uses_worst_case_astronaut_per_day(crew_records):
    snapshots = project_mission_risk(crew_records)
    for snap in snapshots:
        worst_actual = max(v["drift_score"] for v in snap.per_astronaut.values())
        assert snap.overall_drift_score == pytest.approx(worst_actual)
        assert snap.per_astronaut[snap.highest_risk_astronaut]["drift_score"] == pytest.approx(worst_actual)


def test_mission_risk_summary_picks_the_actual_worst_day(crew_records):
    """Regression test for a real tie-breaking bug: max() on risk_level
    alone picked an earlier day even when a later day had a strictly
    higher score within the same risk band."""
    summary = mission_risk_summary(crew_records)
    snapshots = project_mission_risk(crew_records)
    max_score_snapshot = max(snapshots, key=lambda s: s.overall_drift_score)
    assert summary["worst_day"] == max_score_snapshot.day
    assert summary["worst_drift_score"] == max_score_snapshot.overall_drift_score


def test_whatif_reassignment_reduces_drift_from_intervention_day_onward():
    profile = AstronautProfile(astronaut_id="A2", name="Okafor", baseline_pvt_lapses=3.5, resilience=1.15, seed=2)
    original_records = simulate_mission(profile, num_days=6)

    result = simulate_intervention(profile, original_records, day=4, task_load_delta=-8, reassign_to="A3")

    for row in result.comparison:
        if row["day"] < 4:
            assert row["delta"] == 0.0  # nothing before the intervention should change
        elif row["day"] == 4:
            assert row["delta"] < 0  # the intervention day itself should improve