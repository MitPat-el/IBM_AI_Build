"""Tests for dependency_graph.py -- pure graph logic, no AI, no DB."""

import pytest

from dependency_graph import TaskRecord, build_graph, downstream_task_ids, compute_impact, at_risk_tasks


@pytest.fixture
def simple_chain_graph():
    tasks = [
        TaskRecord("T1", "Power Up Payload Bay", 1, "A1", 3),
        TaskRecord("T2", "Calibrate Spectrometer", 1, "A2", 2),
        TaskRecord("T5", "Airlock Depressurization", 2, "A2", 3),
        TaskRecord("T7", "EVA - External Repair", 3, "A1", 6),
    ]
    edges = [("T2", "T1"), ("T5", "T2"), ("T7", "T5")]  # T1 -> T2 -> T5 -> T7
    risk_lookup = {("A1", 3): "high", ("A1", 1): "nominal"}
    return build_graph(tasks, edges, risk_lookup)


def test_downstream_follows_multi_hop_chain(simple_chain_graph):
    edge_pairs = [(e.task_id, e.depends_on_id) for e in simple_chain_graph.edges]
    result = downstream_task_ids("T1", edge_pairs)
    assert result == ["T2", "T5", "T7"]


def test_downstream_of_leaf_task_is_empty(simple_chain_graph):
    edge_pairs = [(e.task_id, e.depends_on_id) for e in simple_chain_graph.edges]
    assert downstream_task_ids("T7", edge_pairs) == []


def test_cyclic_edges_terminate_instead_of_hanging():
    """Regression guard: a malformed/cyclic edge list must not infinite-loop."""
    cyclic_edges = [("T1", "T2"), ("T2", "T1")]
    result = downstream_task_ids("T1", cyclic_edges)
    assert result == ["T2"]


def test_compute_impact_returns_task_and_full_downstream(simple_chain_graph):
    impact = compute_impact("T1", simple_chain_graph)
    assert impact.task.task_id == "T1"
    assert [d.task_id for d in impact.downstream] == ["T2", "T5", "T7"]


def test_compute_impact_unknown_task_returns_none(simple_chain_graph):
    assert compute_impact("NOPE", simple_chain_graph) is None


def test_nodes_carry_risk_level_from_lookup(simple_chain_graph):
    node_by_id = {n.task_id: n for n in simple_chain_graph.nodes}
    assert node_by_id["T1"].risk_level == "nominal"
    assert node_by_id["T7"].risk_level == "high"
    assert node_by_id["T2"].risk_level is None  # no lookup entry for (A2, 1)


def test_at_risk_tasks_ranks_by_downstream_impact():
    tasks = [
        TaskRecord("A", "Task A", 1, "X1", 1),
        TaskRecord("B", "Task B", 1, "X1", 1),
        TaskRecord("C", "Task C", 1, "X1", 1),
        TaskRecord("D", "Task D", 1, "X1", 1),
    ]
    # A -> B -> C, D standalone. A has 2 downstream, D has 0.
    edges = [("B", "A"), ("C", "B")]
    risk_lookup = {("X1", 1): "critical"}
    graph = build_graph(tasks, edges, risk_lookup)

    ranked = at_risk_tasks(graph)
    assert ranked[0]["task_id"] == "A"
    assert ranked[0]["downstream_count"] == 2
    task_ids_in_order = [r["task_id"] for r in ranked]
    assert task_ids_in_order.index("A") < task_ids_in_order.index("D")


def test_at_risk_tasks_excludes_nominal_risk():
    tasks = [TaskRecord("A", "Task A", 1, "X1", 1)]
    graph = build_graph(tasks, [], {("X1", 1): "nominal"})
    assert at_risk_tasks(graph) == []