"""Tests for simulator.py -- mainly determinism, since the What-If
Simulator and DB-vs-resimulation consistency both depend on it."""

from simulator import AstronautProfile, simulate_mission, resolve_seed


def test_resolve_seed_is_stable_across_calls():
    """This was a real bug: Python's built-in hash() is randomized per
    process, so a naive fallback would produce different raw signals on
    every restart. resolve_seed() must not have that problem."""
    assert resolve_seed("A1") == resolve_seed("A1")
    assert resolve_seed("A1") != resolve_seed("A2")


def test_same_seed_produces_identical_mission():
    profile_a = AstronautProfile(astronaut_id="X1", name="Test", seed=42)
    profile_b = AstronautProfile(astronaut_id="X1", name="Test", seed=42)

    records_a = simulate_mission(profile_a, num_days=6)
    records_b = simulate_mission(profile_b, num_days=6)

    for ra, rb in zip(records_a, records_b):
        assert ra.hours_slept == rb.hours_slept
        assert ra.pvt_lapses == rb.pvt_lapses
        assert ra.drift.drift_score == rb.drift.drift_score


def test_explicit_seed_zero_is_respected_not_treated_as_falsy():
    """Real bug found earlier: `profile.seed or resolve_seed(...)` would
    silently ignore an explicit seed of 0. Confirm seed=0 actually
    produces different (deterministic) output than the resolved fallback."""
    profile_explicit_zero = AstronautProfile(astronaut_id="A1", name="Test", seed=0)
    profile_no_seed = AstronautProfile(astronaut_id="A1", name="Test", seed=None)

    records_zero = simulate_mission(profile_explicit_zero, num_days=3)
    records_fallback = simulate_mission(profile_no_seed, num_days=3)

    # seed=0 should NOT silently fall back to resolve_seed("A1") --
    # unless they coincidentally collide, these should differ.
    assert resolve_seed("A1") != 0  # sanity: the fallback isn't 0 anyway
    values_zero = [r.hours_slept for r in records_zero]
    values_fallback = [r.hours_slept for r in records_fallback]
    assert values_zero != values_fallback


def test_drift_generally_increases_over_a_demanding_mission():
    """Not a strict monotonic check (there's randomness), but the last
    day should be worse than the first for the default schedule, which
    front-loads rest and back-loads workload."""
    profile = AstronautProfile(astronaut_id="A2", name="Okafor", baseline_pvt_lapses=3.5, resilience=1.15, seed=2)
    records = simulate_mission(profile, num_days=6)
    assert records[-1].drift.drift_score > records[0].drift.drift_score