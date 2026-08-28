"""
Reassignment Feasibility Checker.

Evaluates whether a proposed task reassignment from one astronaut to another
is operationally sensible given the current mission state.  All logic is
deterministic and rule-based — no LLM is involved.

Three checks are performed:

  1. Fatigue   -- is the receiver's drift score low enough to take on extra work?
  2. Workload  -- would the added load push the receiver beyond a safe multiplier
                  of their own rolling average?
  3. Dependency conflict -- does the receiver already own tasks on that day that
                  have downstream dependents?  (WARNING only, not a hard block.)

Possible outcomes
-----------------
  "feasible"              -- all hard checks pass, no dependency warning
  "feasible_with_caution" -- hard checks pass but a dependency warning was raised
  "not_feasible"          -- at least one hard check (fatigue or workload) failed

IMPORTANT — PROTOTYPE DISCLAIMER
----------------------------------
This module is a PROTOTYPE decision-support tool.  The thresholds and rules
defined here are engineering heuristics chosen to demonstrate the concept.
They are NOT NASA-validated operational limits, NOT clinically validated
impairment thresholds, and NOT a substitute for flight surgeon or flight
director judgment.  See ADVISORY_DISCLAIMER below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from dependency_graph import DependencyGraph, downstream_task_ids
from simulator import MissionDayRecord


# ---------------------------------------------------------------------------
# Advisory disclaimer -- included verbatim in every FeasibilityResult so the
# API consumer can never mistake this for an authoritative operational decision.
# ---------------------------------------------------------------------------
ADVISORY_DISCLAIMER = (
    "This feasibility assessment is a prototype decision-support tool only. "
    "It does not constitute operational clearance, medical authorization, or "
    "a substitute for flight surgeon or flight director judgment. "
    "All crew reassignment decisions must be made by authorized mission personnel."
)

# ---------------------------------------------------------------------------
# PROTOTYPE THRESHOLD: 1.5× rolling-average workload.
# The drift.py workload_score() sub-score saturates at 2.0× rolling average
# (score = 1.0).  1.5× sits midway between average and saturation and provides
# a prototype safety margin.
# This is NOT a validated NASA or aviation-medicine workload capacity limit.
# ---------------------------------------------------------------------------
WORKLOAD_RATIO_THRESHOLD: float = 1.5

# Risk levels that constitute a hard failure for fatigue.
# PROTOTYPE RULE: "high" and "critical" are treated as operationally blocking
# for task reassignment in this prototype.  "elevated" is treated as a pass
# because it represents the lowest named alert level (the same level at which
# the Bob/watsonx explanation layer is triggered for monitoring purposes).
# This boundary is NOT a clinically validated impairment cutoff.
_FATIGUE_BLOCKING_LEVELS = frozenset({"high", "critical"})

FeasibilityStatus = Literal["feasible", "feasible_with_caution", "not_feasible"]


# ---------------------------------------------------------------------------
# Per-check result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FatigueFeasibility:
    drift_score: float
    risk_level: str          # mirrors drift.classify_risk() output
    passed: bool
    reason: Optional[str]    # None when passed=True


@dataclass
class WorkloadFeasibility:
    receiver_load_that_day: float
    rolling_avg: float
    projected_ratio: float   # (receiver_load + abs(task_load_delta)) / max(rolling_avg, 1.0)
    passed: bool
    reason: Optional[str]    # None when passed=True


@dataclass
class DependencyConflict:
    """
    A soft advisory check only -- never a hard gate.

    `has_conflict` is True when the receiver already owns at least one task
    on the requested day that has downstream dependents.  Adding load on such
    a day increases the risk of that critical task slipping, which cascades to
    downstream work.  This is reported as a warning so the decision-maker can
    weigh it, but it does not automatically make the reassignment infeasible.
    """
    conflicting_tasks: list[dict]  # [{task_id, name, downstream_count}]
    has_conflict: bool
    warning: Optional[str]         # None when has_conflict=False


@dataclass
class FeasibilityResult:
    status: FeasibilityStatus
    receiver: str
    checks: dict                   # {"fatigue": ..., "workload": ..., "dependency_conflict": ...}
    reasons: list[str]             # reasons from failed HARD checks only (fatigue, workload)
    warnings: list[str]            # advisory notices from SOFT checks (dependency conflict)
    advisory: str                  # always == ADVISORY_DISCLAIMER


# ---------------------------------------------------------------------------
# Individual check functions (also importable directly for unit testing)
# ---------------------------------------------------------------------------

def check_fatigue(receiver_record: MissionDayRecord) -> FatigueFeasibility:
    """
    PROTOTYPE RULE: treat risk_level "high" or "critical" as a hard block.
    "nominal" and "elevated" are treated as a pass.

    The "elevated" pass-through is a prototype simplification.  "elevated" is
    the lowest named alert level in this system (drift_score >= 0.35).  Passing
    it here does NOT imply the receiver is clinically unimpaired at that level --
    it means this prototype does not block at that level.  The boundary between
    "elevated" and operationally impaired is not established by this system.
    """
    risk = receiver_record.drift.risk_level
    score = receiver_record.drift.drift_score

    if risk in _FATIGUE_BLOCKING_LEVELS:
        reason = (
            f"[PROTOTYPE RULE] Receiver is at {risk.upper()} fatigue risk on "
            f"day {receiver_record.day} (drift score {score:.2f}). "
            f"This prototype treats 'high' and 'critical' risk levels as "
            f"operationally blocking for task reassignment. "
            f"This threshold is not a clinically validated impairment cutoff."
        )
        return FatigueFeasibility(drift_score=score, risk_level=risk, passed=False, reason=reason)

    return FatigueFeasibility(drift_score=score, risk_level=risk, passed=True, reason=None)


def check_workload(
    receiver_record: MissionDayRecord,
    task_load_delta: float,
) -> WorkloadFeasibility:
    """
    PROTOTYPE THRESHOLD: 1.5× rolling-average workload (WORKLOAD_RATIO_THRESHOLD).

    task_load_delta is always negative (load removed from the source astronaut).
    The receiver absorbs the absolute value of that delta.
    """
    added_load = abs(task_load_delta)
    current_load = receiver_record.task_load
    rolling_avg = max(receiver_record.rolling_avg_task_load, 1.0)  # guard against 0

    projected_load = current_load + added_load
    projected_ratio = projected_load / rolling_avg

    if projected_ratio > WORKLOAD_RATIO_THRESHOLD:
        reason = (
            f"[PROTOTYPE THRESHOLD] Adding {added_load:.1f} load units would bring "
            f"receiver to {projected_ratio:.2f}× their rolling average on "
            f"day {receiver_record.day} "
            f"(threshold: {WORKLOAD_RATIO_THRESHOLD}×). "
            f"This threshold is a prototype operational heuristic, "
            f"not a validated capacity limit."
        )
        return WorkloadFeasibility(
            receiver_load_that_day=current_load,
            rolling_avg=rolling_avg,
            projected_ratio=round(projected_ratio, 4),
            passed=False,
            reason=reason,
        )

    return WorkloadFeasibility(
        receiver_load_that_day=current_load,
        rolling_avg=rolling_avg,
        projected_ratio=round(projected_ratio, 4),
        passed=True,
        reason=None,
    )


def check_dependency_conflict(
    receiver_id: str,
    day: int,
    graph: DependencyGraph,
) -> DependencyConflict:
    """
    Advisory check only -- not a hard block.

    Identifies tasks already assigned to the receiver on `day` that have at
    least one downstream dependent.  Adding load on such a day may increase
    the risk of a critical-path task slipping.  The caller decides what weight
    to give this warning.
    """
    edge_pairs = [(e.task_id, e.depends_on_id) for e in graph.edges]

    conflicting = []
    for node in graph.nodes:
        if node.astronaut_id != receiver_id or node.day != day:
            continue
        ds_count = len(downstream_task_ids(node.task_id, edge_pairs))
        if ds_count > 0:
            conflicting.append({
                "task_id": node.task_id,
                "name": node.name,
                "downstream_count": ds_count,
            })

    if not conflicting:
        return DependencyConflict(conflicting_tasks=[], has_conflict=False, warning=None)

    total_downstream = sum(t["downstream_count"] for t in conflicting)
    task_names = ", ".join(f'"{t["name"]}"' for t in conflicting)
    warning = (
        f"Receiver already has {len(conflicting)} dependency-critical task(s) on "
        f"day {day} ({task_names}), with {total_downstream} total downstream "
        f"task(s) at risk if any slip. "
        f"This is advisory -- mission personnel should assess whether the "
        f"receiver's existing tasks can absorb the additional load."
    )
    return DependencyConflict(
        conflicting_tasks=conflicting,
        has_conflict=True,
        warning=warning,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_reassignment_feasibility(
    source_id: str,
    receiver_id: str,
    day: int,
    task_load_delta: float,
    receiver_records: list[MissionDayRecord],
    graph: DependencyGraph,
) -> FeasibilityResult:
    """
    Run all three checks and return a FeasibilityResult with one of three statuses:

      "feasible"              -- all hard checks pass, no dependency warning
      "feasible_with_caution" -- hard checks pass, dependency warning raised
      "not_feasible"          -- fatigue or workload hard check failed

    Pre-conditions (caller must enforce before calling this function):
      - receiver_id != source_id
      - receiver_records is the full mission record list for receiver_id
      - receiver_records is already confirmed non-empty

    This function is pure: it reads from the passed-in data only, performs no
    DB access, and has no side effects.
    """
    # Find the receiver's record for the requested day.
    receiver_record = next((r for r in receiver_records if r.day == day), None)
    if receiver_record is None:
        return FeasibilityResult(
            status="not_feasible",
            receiver=receiver_id,
            checks={},
            reasons=[f"No mission record found for receiver '{receiver_id}' on day {day}."],
            warnings=[],
            advisory=ADVISORY_DISCLAIMER,
        )

    fatigue = check_fatigue(receiver_record)
    workload = check_workload(receiver_record, task_load_delta)
    dep = check_dependency_conflict(receiver_id, day, graph)

    hard_failed = not fatigue.passed or not workload.passed

    if hard_failed:
        status: FeasibilityStatus = "not_feasible"
    elif dep.has_conflict:
        status = "feasible_with_caution"
    else:
        status = "feasible"

    reasons = [c.reason for c in [fatigue, workload] if not c.passed]
    warnings = [dep.warning] if dep.has_conflict else []

    return FeasibilityResult(
        status=status,
        receiver=receiver_id,
        checks={
            "fatigue": fatigue,
            "workload": workload,
            "dependency_conflict": dep,
        },
        reasons=reasons,
        warnings=warnings,
        advisory=ADVISORY_DISCLAIMER,
    )
