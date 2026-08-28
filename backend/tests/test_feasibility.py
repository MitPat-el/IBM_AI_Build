"""
Unit tests for feasibility.py.

All tests are pure: they construct MissionDayRecord / DependencyGraph
objects in memory using existing dataclasses.  No database, no HTTP
client, no fixtures from conftest.py are needed or used.
"""

import pytest

from drift import DriftResult
from simulator import MissionDayRecord
from dependency_graph import DependencyGraph, TaskNode, GraphEdge
from feasibility import (
    ADVISORY_DISCLAIMER,
    WORKLOAD_RATIO_THRESHOLD,
    FatigueFeasibility,
    WorkloadFeasibility,
    DependencyConflict,
    FeasibilityResult,
    check_fatigue,
    check_workload,
    check_dependency_conflict,
    check_reassignment_feasibility,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_drift(drift_score: float, risk_level: str) -> DriftResult:
    return DriftResult(
        reaction_time_score=0.0,
        sleep_debt_score=0.0,
        circadian_score=0.0,
        workload_score=0.0,
        drift_score=drift_score,
        updated_sleep_debt_hours=0.0,
        risk_level=risk_level,
    )


def _make_record(
    day: int,
    drift_score: float,
    risk_level: str,
    task_load: float = 8.0,
    rolling_avg_task_load: float = 8.0,
    astronaut_id: str = "A2",
) -> MissionDayRecord:
    return MissionDayRecord(
        day=day,
        astronaut_id=astronaut_id,
        hours_slept=7.0,
        pvt_lapses=3,
        minutes_phase_shift=60.0,
        task_load=task_load,
        rolling_avg_task_load=rolling_avg_task_load,
        drift=_make_drift(drift_score, risk_level),
    )


def _empty_graph() -> DependencyGraph:
    return DependencyGraph(nodes=[], edges=[])


def _graph_with_tasks(nodes: list[TaskNode], edges: list[GraphEdge] = None) -> DependencyGraph:
    return DependencyGraph(nodes=nodes, edges=edges or [])


# ---------------------------------------------------------------------------
# check_fatigue
# ---------------------------------------------------------------------------

def test_fatigue_passes_for_nominal():
    record = _make_record(day=3, drift_score=0.20, risk_level="nominal")
    result = check_fatigue(record)
    assert result.passed is True
    assert result.reason is None
    assert result.risk_level == "nominal"
    assert result.drift_score == 0.20


def test_fatigue_passes_for_elevated():
    """
    PROTOTYPE RULE: 'elevated' is treated as a pass.
    This is a prototype simplification -- not a clinical clearance.
    """
    record = _make_record(day=3, drift_score=0.50, risk_level="elevated")
    result = check_fatigue(record)
    assert result.passed is True
    assert result.reason is None


def test_fatigue_fails_for_high():
    record = _make_record(day=3, drift_score=0.70, risk_level="high")
    result = check_fatigue(record)
    assert result.passed is False
    assert result.reason is not None
    assert "[PROTOTYPE RULE]" in result.reason
    assert "HIGH" in result.reason


def test_fatigue_fails_for_critical():
    record = _make_record(day=3, drift_score=0.90, risk_level="critical")
    result = check_fatigue(record)
    assert result.passed is False
    assert result.reason is not None
    assert "[PROTOTYPE RULE]" in result.reason
    assert "CRITICAL" in result.reason


def test_fatigue_reason_is_none_exactly_when_passed():
    """None, not empty string."""
    for level in ("nominal", "elevated"):
        record = _make_record(day=1, drift_score=0.30, risk_level=level)
        assert check_fatigue(record).reason is None

    for level in ("high", "critical"):
        record = _make_record(day=1, drift_score=0.80, risk_level=level)
        assert check_fatigue(record).reason is not None


# ---------------------------------------------------------------------------
# check_workload
# ---------------------------------------------------------------------------

def test_workload_passes_at_exactly_threshold():
    """projected_ratio == 1.5 is at the boundary and must pass (≤ threshold)."""
    # current=8, rolling=8, delta=-4 → (8+4)/8 = 1.5 exactly
    record = _make_record(day=3, drift_score=0.3, risk_level="nominal",
                          task_load=8.0, rolling_avg_task_load=8.0)
    result = check_workload(record, task_load_delta=-4.0)
    assert result.passed is True
    assert result.reason is None
    assert result.projected_ratio == pytest.approx(1.5)


def test_workload_fails_just_above_threshold():
    # current=8, rolling=8, delta=-5 → (8+5)/8 = 1.625
    record = _make_record(day=3, drift_score=0.3, risk_level="nominal",
                          task_load=8.0, rolling_avg_task_load=8.0)
    result = check_workload(record, task_load_delta=-5.0)
    assert result.passed is False
    assert result.reason is not None
    assert "[PROTOTYPE THRESHOLD]" in result.reason
    assert str(WORKLOAD_RATIO_THRESHOLD) in result.reason


def test_workload_uses_abs_of_negative_delta():
    """
    task_load_delta is always negative (load removed from source).
    The receiver absorbs the absolute value.  Confirm the projected load
    increases, not decreases.
    """
    record = _make_record(day=2, drift_score=0.3, risk_level="nominal",
                          task_load=6.0, rolling_avg_task_load=8.0)
    result = check_workload(record, task_load_delta=-6.0)
    # projected = 6 + abs(-6) = 12; ratio = 12/8 = 1.5
    assert result.projected_ratio == pytest.approx(1.5)
    assert result.receiver_load_that_day == 6.0


def test_workload_guards_zero_rolling_avg():
    """max(rolling_avg, 1.0) must prevent ZeroDivisionError."""
    record = _make_record(day=1, drift_score=0.2, risk_level="nominal",
                          task_load=4.0, rolling_avg_task_load=0.0)
    result = check_workload(record, task_load_delta=-2.0)
    # rolling_avg clamped to 1.0 → projected_ratio = (4+2)/1 = 6.0 → fails
    assert result.rolling_avg == pytest.approx(1.0)
    assert result.passed is False


def test_workload_projected_ratio_present_in_result():
    record = _make_record(day=2, drift_score=0.3, risk_level="nominal",
                          task_load=9.0, rolling_avg_task_load=9.0)
    result = check_workload(record, task_load_delta=-3.0)
    # projected = (9+3)/9 = 1.333...
    assert abs(result.projected_ratio - (12 / 9)) < 0.001


def test_workload_reason_is_none_when_passed():
    record = _make_record(day=1, drift_score=0.2, risk_level="nominal",
                          task_load=8.0, rolling_avg_task_load=8.0)
    result = check_workload(record, task_load_delta=-1.0)   # ratio = 9/8 = 1.125
    assert result.passed is True
    assert result.reason is None


# ---------------------------------------------------------------------------
# check_dependency_conflict
# ---------------------------------------------------------------------------

def test_dependency_no_tasks_that_day():
    """Graph has no tasks for receiver on the requested day."""
    nodes = [
        TaskNode(task_id="T1", name="Task One", day=2, astronaut_id="A2", load=3.0),
    ]
    edges = []
    graph = _graph_with_tasks(nodes, edges)
    result = check_dependency_conflict("A2", day=1, graph=graph)
    assert result.has_conflict is False
    assert result.warning is None
    assert result.conflicting_tasks == []


def test_dependency_all_leaf_tasks():
    """Receiver has tasks that day but none have downstream dependents."""
    nodes = [
        TaskNode(task_id="T1", name="Task One", day=3, astronaut_id="A2", load=2.0),
        TaskNode(task_id="T2", name="Task Two", day=3, astronaut_id="A2", load=2.0),
    ]
    # T1 and T2 have no tasks depending on them → leaf nodes
    edges = []
    graph = _graph_with_tasks(nodes, edges)
    result = check_dependency_conflict("A2", day=3, graph=graph)
    assert result.has_conflict is False
    assert result.warning is None


def test_dependency_one_critical_task():
    """Receiver has one task on that day with downstream dependents."""
    nodes = [
        TaskNode(task_id="T1", name="EVA Prep", day=3, astronaut_id="A2", load=4.0),
        TaskNode(task_id="T2", name="Airlock Dep", day=3, astronaut_id="A1", load=3.0),
        TaskNode(task_id="T3", name="EVA Repair", day=4, astronaut_id="A1", load=6.0),
    ]
    # T2 depends on T1 (T1 → T2), T3 depends on T2
    edges = [
        GraphEdge(task_id="T2", depends_on_id="T1"),
        GraphEdge(task_id="T3", depends_on_id="T2"),
    ]
    graph = _graph_with_tasks(nodes, edges)
    result = check_dependency_conflict("A2", day=3, graph=graph)
    assert result.has_conflict is True
    assert result.warning is not None
    assert len(result.conflicting_tasks) == 1
    assert result.conflicting_tasks[0]["task_id"] == "T1"
    assert result.conflicting_tasks[0]["downstream_count"] == 2  # T2 and T3


def test_dependency_multiple_critical_tasks():
    """Receiver has two tasks that day, both with dependents."""
    nodes = [
        TaskNode(task_id="T1", name="Task A", day=2, astronaut_id="A3", load=3.0),
        TaskNode(task_id="T2", name="Task B", day=2, astronaut_id="A3", load=2.0),
        TaskNode(task_id="T3", name="Task C", day=3, astronaut_id="A1", load=2.0),
        TaskNode(task_id="T4", name="Task D", day=3, astronaut_id="A1", load=2.0),
    ]
    edges = [
        GraphEdge(task_id="T3", depends_on_id="T1"),
        GraphEdge(task_id="T4", depends_on_id="T2"),
    ]
    graph = _graph_with_tasks(nodes, edges)
    result = check_dependency_conflict("A3", day=2, graph=graph)
    assert result.has_conflict is True
    assert len(result.conflicting_tasks) == 2


def test_dependency_conflict_has_no_passed_attribute():
    """DependencyConflict must not have a 'passed' field -- it is advisory, not a gate."""
    result = check_dependency_conflict("A2", day=1, graph=_empty_graph())
    assert not hasattr(result, "passed"), (
        "DependencyConflict must not have a 'passed' field. "
        "It is advisory; the system does not make a go/no-go decision on this check."
    )


def test_dependency_warning_text_is_advisory_not_prohibitive():
    """Warning language must be advisory, not prohibitive."""
    nodes = [
        TaskNode(task_id="T1", name="EVA Prep", day=3, astronaut_id="A2", load=4.0),
        TaskNode(task_id="T2", name="EVA", day=4, astronaut_id="A1", load=6.0),
    ]
    edges = [GraphEdge(task_id="T2", depends_on_id="T1")]
    graph = _graph_with_tasks(nodes, edges)
    result = check_dependency_conflict("A2", day=3, graph=graph)
    assert result.has_conflict is True
    for forbidden in ("blocked", "rejected", "forbidden", "not feasible", "not_feasible"):
        assert forbidden not in result.warning.lower(), (
            f"Warning text must not contain '{forbidden}'. Got: {result.warning}"
        )


# ---------------------------------------------------------------------------
# check_reassignment_feasibility -- overall status
# ---------------------------------------------------------------------------

def _make_single_record(
    day: int,
    drift_score: float,
    risk_level: str,
    task_load: float = 8.0,
    rolling_avg: float = 8.0,
    astronaut_id: str = "A2",
) -> list[MissionDayRecord]:
    return [_make_record(day, drift_score, risk_level, task_load, rolling_avg, astronaut_id)]


def test_status_feasible_all_checks_clear():
    records = _make_single_record(day=3, drift_score=0.3, risk_level="nominal",
                                  task_load=8.0, rolling_avg=8.0)
    result = check_reassignment_feasibility(
        source_id="A1", receiver_id="A2", day=3,
        task_load_delta=-2.0,          # projected ratio = (8+2)/8 = 1.25 ≤ 1.5
        receiver_records=records,
        graph=_empty_graph(),
    )
    assert result.status == "feasible"
    assert result.reasons == []
    assert result.warnings == []


def test_status_feasible_with_caution_dep_conflict_only():
    """Hard checks pass; only a dependency warning raises the status."""
    records = _make_single_record(day=3, drift_score=0.3, risk_level="nominal",
                                  task_load=8.0, rolling_avg=8.0)
    nodes = [
        TaskNode(task_id="T1", name="EVA Prep", day=3, astronaut_id="A2", load=4.0),
        TaskNode(task_id="T2", name="EVA", day=4, astronaut_id="A1", load=6.0),
    ]
    edges = [GraphEdge(task_id="T2", depends_on_id="T1")]
    graph = _graph_with_tasks(nodes, edges)

    result = check_reassignment_feasibility(
        source_id="A1", receiver_id="A2", day=3,
        task_load_delta=-2.0,
        receiver_records=records,
        graph=graph,
    )
    assert result.status == "feasible_with_caution"
    assert result.reasons == []
    assert len(result.warnings) == 1


def test_status_not_feasible_fatigue_only():
    records = _make_single_record(day=3, drift_score=0.70, risk_level="high")
    result = check_reassignment_feasibility(
        source_id="A1", receiver_id="A2", day=3,
        task_load_delta=-2.0,
        receiver_records=records,
        graph=_empty_graph(),
    )
    assert result.status == "not_feasible"
    assert len(result.reasons) == 1
    assert "HIGH" in result.reasons[0]
    assert result.warnings == []


def test_status_not_feasible_workload_only():
    records = _make_single_record(day=3, drift_score=0.30, risk_level="nominal",
                                  task_load=8.0, rolling_avg=8.0)
    result = check_reassignment_feasibility(
        source_id="A1", receiver_id="A2", day=3,
        task_load_delta=-8.0,          # projected ratio = (8+8)/8 = 2.0 > 1.5
        receiver_records=records,
        graph=_empty_graph(),
    )
    assert result.status == "not_feasible"
    assert len(result.reasons) == 1
    assert "[PROTOTYPE THRESHOLD]" in result.reasons[0]
    assert result.warnings == []


def test_status_not_feasible_both_hard_checks_fail():
    records = _make_single_record(day=3, drift_score=0.80, risk_level="critical",
                                  task_load=8.0, rolling_avg=8.0)
    result = check_reassignment_feasibility(
        source_id="A1", receiver_id="A2", day=3,
        task_load_delta=-8.0,          # workload also fails
        receiver_records=records,
        graph=_empty_graph(),
    )
    assert result.status == "not_feasible"
    assert len(result.reasons) == 2


def test_status_not_feasible_hard_fail_overrides_dep_conflict():
    """
    When a hard check fails AND a dep conflict exists:
      - status must be "not_feasible" (not "feasible_with_caution")
      - warnings should still be populated (dep warning coexists with hard failure)
    """
    records = _make_single_record(day=3, drift_score=0.70, risk_level="high")
    nodes = [
        TaskNode(task_id="T1", name="EVA Prep", day=3, astronaut_id="A2", load=4.0),
        TaskNode(task_id="T2", name="EVA", day=4, astronaut_id="A1", load=6.0),
    ]
    edges = [GraphEdge(task_id="T2", depends_on_id="T1")]
    graph = _graph_with_tasks(nodes, edges)

    result = check_reassignment_feasibility(
        source_id="A1", receiver_id="A2", day=3,
        task_load_delta=-2.0,
        receiver_records=records,
        graph=graph,
    )
    assert result.status == "not_feasible"
    assert len(result.reasons) >= 1
    assert len(result.warnings) == 1    # dep warning still reported


def test_advisory_disclaimer_always_present_feasible():
    records = _make_single_record(day=1, drift_score=0.2, risk_level="nominal")
    result = check_reassignment_feasibility(
        source_id="A1", receiver_id="A2", day=1,
        task_load_delta=-1.0,
        receiver_records=records,
        graph=_empty_graph(),
    )
    assert result.status == "feasible"
    assert result.advisory == ADVISORY_DISCLAIMER


def test_advisory_disclaimer_always_present_not_feasible():
    records = _make_single_record(day=1, drift_score=0.9, risk_level="critical")
    result = check_reassignment_feasibility(
        source_id="A1", receiver_id="A2", day=1,
        task_load_delta=-1.0,
        receiver_records=records,
        graph=_empty_graph(),
    )
    assert result.status == "not_feasible"
    assert result.advisory == ADVISORY_DISCLAIMER


def test_advisory_disclaimer_always_present_feasible_with_caution():
    records = _make_single_record(day=2, drift_score=0.3, risk_level="nominal",
                                  task_load=8.0, rolling_avg=8.0)
    nodes = [
        TaskNode(task_id="T1", name="Task A", day=2, astronaut_id="A2", load=3.0),
        TaskNode(task_id="T2", name="Task B", day=3, astronaut_id="A1", load=3.0),
    ]
    edges = [GraphEdge(task_id="T2", depends_on_id="T1")]
    graph = _graph_with_tasks(nodes, edges)
    result = check_reassignment_feasibility(
        source_id="A1", receiver_id="A2", day=2,
        task_load_delta=-1.0,
        receiver_records=records,
        graph=graph,
    )
    assert result.status == "feasible_with_caution"
    assert result.advisory == ADVISORY_DISCLAIMER


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_no_record_for_requested_day_returns_not_feasible():
    """Receiver has records for days 1–5 but day=6 is requested."""
    records = [_make_record(d, 0.3, "nominal") for d in range(1, 6)]
    result = check_reassignment_feasibility(
        source_id="A1", receiver_id="A2", day=6,
        task_load_delta=-2.0,
        receiver_records=records,
        graph=_empty_graph(),
    )
    assert result.status == "not_feasible"
    assert any("No mission record" in r for r in result.reasons)
    # checks dict should be empty (we couldn't run them)
    assert result.checks == {}


def test_result_has_required_top_level_keys():
    records = _make_single_record(day=1, drift_score=0.2, risk_level="nominal")
    result = check_reassignment_feasibility(
        source_id="A1", receiver_id="A2", day=1,
        task_load_delta=-1.0,
        receiver_records=records,
        graph=_empty_graph(),
    )
    assert hasattr(result, "status")
    assert hasattr(result, "receiver")
    assert hasattr(result, "checks")
    assert hasattr(result, "reasons")
    assert hasattr(result, "warnings")
    assert hasattr(result, "advisory")


def test_checks_dict_has_three_keys_when_record_found():
    records = _make_single_record(day=2, drift_score=0.3, risk_level="nominal")
    result = check_reassignment_feasibility(
        source_id="A1", receiver_id="A2", day=2,
        task_load_delta=-1.0,
        receiver_records=records,
        graph=_empty_graph(),
    )
    assert set(result.checks.keys()) == {"fatigue", "workload", "dependency_conflict"}


def test_reasons_and_warnings_are_separate_lists():
    """reasons and warnings must be independent list objects."""
    records = _make_single_record(day=1, drift_score=0.2, risk_level="nominal")
    result = check_reassignment_feasibility(
        source_id="A1", receiver_id="A2", day=1,
        task_load_delta=-1.0,
        receiver_records=records,
        graph=_empty_graph(),
    )
    assert result.reasons is not result.warnings
    assert isinstance(result.reasons, list)
    assert isinstance(result.warnings, list)