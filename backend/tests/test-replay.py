"""Tests for replay.py -- Historical Replay and trend-based forward projection."""

from simulator import AstronautProfile, simulate_mission
from replay import get_replay, project_forward


def test_replay_produces_one_point_per_day():
    profile = AstronautProfile(astronaut_id="A1", name="Chen", seed=1)
    records = simulate_mission(profile, num_days=6)
    timeline = get_replay("A1", records)
    assert len(timeline.points) == 6
    assert [p.day for p in timeline.points] == [1, 2, 3, 4, 5, 6]


def test_projection_extends_the_correct_number_of_days():
    profile = AstronautProfile(astronaut_id="A1", name="Chen", seed=1)
    records = simulate_mission(profile, num_days=6)
    timeline = get_replay("A1", records)
    timeline = project_forward(timeline, num_days=3)

    assert len(timeline.projected_points) == 3
    assert [p.day for p in timeline.projected_points] == [7, 8, 9]


def test_projection_scores_stay_within_valid_range():
    """Extrapolation could overshoot past 1.0 with a steep trend --
    confirm it's clipped like the real drift score is."""
    profile = AstronautProfile(astronaut_id="A2", name="Okafor", resilience=1.3, seed=99)
    records = simulate_mission(profile, num_days=6)
    timeline = get_replay("A2", records)
    timeline = project_forward(timeline, num_days=5)

    for p in timeline.projected_points:
        assert 0.0 <= p.drift_score <= 1.0


def test_projection_is_a_noop_with_too_few_points():
    profile = AstronautProfile(astronaut_id="A1", name="Chen", seed=1)
    records = simulate_mission(profile, num_days=1)
    timeline = get_replay("A1", records)
    timeline = project_forward(timeline, num_days=3)
    assert timeline.projected_points == []