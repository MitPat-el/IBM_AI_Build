"""Unit tests for drift.py -- the deterministic formula, no AI, no DB."""

import pytest

from drift import (
    AstronautSignals, DriftWeights, compute_drift_score,
    reaction_time_score, sleep_debt_score, circadian_score, workload_score,
    classify_risk, EXPLANATION_TRIGGER_THRESHOLD,
)


def test_reaction_time_score_at_baseline_is_zero():
    assert reaction_time_score(pvt_lapses=3, baseline=3) == 0.0


def test_reaction_time_score_saturates_at_double_baseline():
    assert reaction_time_score(pvt_lapses=6, baseline=3) == 1.0
    assert reaction_time_score(pvt_lapses=100, baseline=3) == 1.0  # still clipped, not >1


def test_reaction_time_score_never_negative():
    """Fewer lapses than baseline (well-rested) should floor at 0, not go negative."""
    assert reaction_time_score(pvt_lapses=1, baseline=5) == 0.0


def test_sleep_debt_accumulates_with_deficit():
    score, updated_debt = sleep_debt_score(
        hours_slept_last_24=5.0, recommended=8.0, prior_debt=0.0
    )
    assert updated_debt == 3.0  # exact 3h deficit, no prior debt to carry
    assert score == pytest.approx(3.0 / 12.0)


def test_sleep_debt_recovers_on_adequate_sleep():
    score, updated_debt = sleep_debt_score(
        hours_slept_last_24=8.0, recommended=8.0, prior_debt=4.0, recovery_rate=0.7
    )
    assert updated_debt == pytest.approx(2.8)  # 4.0 * 0.7, no new deficit added


def test_sleep_debt_caps_at_max():
    score, updated_debt = sleep_debt_score(
        hours_slept_last_24=0.0, recommended=8.0, prior_debt=11.0, cap_hours=12.0
    )
    assert updated_debt == 12.0
    assert score == 1.0


def test_circadian_score_symmetric_around_zero_drift():
    assert circadian_score(0, max_tolerable_minutes=720) == 0.0
    assert circadian_score(360, max_tolerable_minutes=720) == 0.5
    assert circadian_score(-360, max_tolerable_minutes=720) == 0.5  # sign shouldn't matter
    assert circadian_score(1440, max_tolerable_minutes=720) == 1.0  # clipped


def test_workload_score_at_average_is_zero():
    assert workload_score(current_load=8, rolling_avg=8) == 0.0


def test_workload_score_saturates_at_double_average():
    assert workload_score(current_load=16, rolling_avg=8) == 1.0


def test_classify_risk_boundaries():
    assert classify_risk(0.0) == "nominal"
    assert classify_risk(0.34) == "nominal"
    assert classify_risk(0.35) == "elevated"
    assert classify_risk(0.54) == "elevated"
    assert classify_risk(0.55) == "high"
    assert classify_risk(0.74) == "high"
    assert classify_risk(0.75) == "critical"
    assert classify_risk(1.0) == "critical"


def test_compute_drift_score_default_weights_sum_matches_expectation():
    """A known, hand-computable input should produce a known output --
    if this ever changes, the formula changed, and that should be a
    deliberate decision, not a silent regression."""
    signals = AstronautSignals(
        pvt_lapses=6,               # 2x baseline -> rt score = 1.0
        pvt_baseline_lapses=3,
        hours_slept_last_24=8.0,    # meets recommended -> no new deficit
        minutes_phase_shift=0,      # no drift -> circadian score = 0.0
        current_task_load=8,
        rolling_avg_task_load=8,    # at average -> workload score = 0.0
        prior_sleep_debt=0.0,
    )
    result = compute_drift_score(signals)
    # only reaction_time contributes: 0.40 * 1.0 = 0.40
    assert result.drift_score == pytest.approx(0.40)
    assert result.risk_level == "elevated"


def test_weights_must_sum_to_one():
    bad_weights = DriftWeights(reaction_time=0.5, sleep_debt=0.5, circadian=0.5, workload=0.5)
    signals = AstronautSignals(
        pvt_lapses=3, pvt_baseline_lapses=3, hours_slept_last_24=8.0,
        minutes_phase_shift=0, current_task_load=8, rolling_avg_task_load=8,
    )
    with pytest.raises(ValueError):
        compute_drift_score(signals, weights=bad_weights)


def test_explanation_threshold_is_the_boundary_used_elsewhere():
    """Guards against someone changing this constant without noticing
    it's load-bearing for main.py's /explain gating logic."""
    assert EXPLANATION_TRIGGER_THRESHOLD == 0.55