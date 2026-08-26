"""
Task dependency graph and cascading impact analysis.

Answers the question the What-If Simulator can't answer on its own:
"if this specific task slips, what else slips with it?" A task's drift
score tells you an astronaut is at risk; this tells you which downstream
work is at risk *because* of that.

Pure graph logic -- no AI, no DB. Takes plain records in, returns plain
records out, the same pattern as projection.py and replay.py.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskRecord:
    task_id: str
    name: str
    day: int
    astronaut_id: str
    load: float


@dataclass
class TaskNode:
    task_id: str
    name: str
    day: int
    astronaut_id: str
    load: float
    risk_level: Optional[str] = None  # the assigned astronaut's risk_level on this task's day, if known


@dataclass
class GraphEdge:
    task_id: str
    depends_on_id: str


@dataclass
class DependencyGraph:
    nodes: list[TaskNode]
    edges: list[GraphEdge]


@dataclass
class ImpactResult:
    task: TaskNode
    downstream: list[TaskNode]  # everything that would slip if `task` slips, in reachable order


def build_graph(
    tasks: list[TaskRecord],
    edges: list[tuple[str, str]],
    risk_lookup: Optional[dict[tuple[str, int], str]] = None,
) -> DependencyGraph:
    """
    risk_lookup maps (astronaut_id, day) -> risk_level, so each node can
    show whether the astronaut assigned to it is already at risk that day.
    """
    risk_lookup = risk_lookup or {}
    nodes = [
        TaskNode(
            task_id=t.task_id,
            name=t.name,
            day=t.day,
            astronaut_id=t.astronaut_id,
            load=t.load,
            risk_level=risk_lookup.get((t.astronaut_id, t.day)),
        )
        for t in tasks
    ]
    graph_edges = [GraphEdge(task_id=a, depends_on_id=b) for a, b in edges]
    return DependencyGraph(nodes=nodes, edges=graph_edges)


def downstream_task_ids(task_id: str, edges: list[tuple[str, str]]) -> list[str]:
    """
    All task_ids that transitively depend on task_id -- i.e. everything
    that would be delayed if task_id itself slips. edges are (task,
    depends_on) pairs, so we walk the reverse direction: task_id -> who
    depends on it -> who depends on THEM, etc. BFS with a seen-set, so a
    malformed/cyclic edge list can't cause an infinite loop.
    """
    children: dict[str, list[str]] = {}
    for task, depends_on in edges:
        children.setdefault(depends_on, []).append(task)

    result: list[str] = []
    seen = {task_id}
    frontier = [task_id]
    while frontier:
        current = frontier.pop(0)
        for child in children.get(current, []):
            if child not in seen:
                seen.add(child)
                result.append(child)
                frontier.append(child)
    return result


def compute_impact(task_id: str, graph: DependencyGraph) -> Optional[ImpactResult]:
    """The task itself plus everything downstream of it, as full nodes
    (not just ids) so the caller has names/astronauts/risk without a
    second lookup."""
    node_by_id = {n.task_id: n for n in graph.nodes}
    task = node_by_id.get(task_id)
    if task is None:
        return None

    edge_pairs = [(e.task_id, e.depends_on_id) for e in graph.edges]
    downstream_ids = downstream_task_ids(task_id, edge_pairs)
    downstream = [node_by_id[tid] for tid in downstream_ids if tid in node_by_id]

    return ImpactResult(task=task, downstream=downstream)


def at_risk_tasks(graph: DependencyGraph, risk_threshold: tuple[str, ...] = ("high", "critical")) -> list[dict]:
    """
    Ranks tasks whose assigned astronaut is currently at meaningful risk,
    by how much downstream work would be affected if that task slips --
    this is the "identify the task and explain why" piece: not just which
    astronaut is fatigued, but which of their tasks matters most to the
    rest of the mission if they can't do it.
    """
    edge_pairs = [(e.task_id, e.depends_on_id) for e in graph.edges]
    flagged = []
    for node in graph.nodes:
        if node.risk_level in risk_threshold:
            downstream_ids = downstream_task_ids(node.task_id, edge_pairs)
            flagged.append({
                "task_id": node.task_id,
                "name": node.name,
                "day": node.day,
                "astronaut_id": node.astronaut_id,
                "risk_level": node.risk_level,
                "downstream_count": len(downstream_ids),
            })
    flagged.sort(key=lambda f: f["downstream_count"], reverse=True)
    return flagged