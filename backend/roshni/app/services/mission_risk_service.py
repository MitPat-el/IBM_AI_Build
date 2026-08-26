"""
services/mission_risk_service.py — Deterministic Mission Risk Engine.

RULE: This module must never call an LLM or external service.
      All calculations are transparent weighted sums using only config.py values.

Formula (prototype assumptions — update config.py when research team validates):

  task_demand = (criticality_norm * 0.40) + (cognitive_norm * 0.30)
              + (physical_norm * 0.20)    + (dependency_factor * 0.10)

  mission_task_risk = (fatigue_score * 0.60) + (task_demand * 0.40)

  mission_risk = (highest_task_risk * 0.70) + (crew_average_risk * 0.30)

We do NOT simply average all risks because that hides a single critical task.
"""

from datetime import datetime, timezone
from typing import Optional

from app.core.config import (
    TASK_DEMAND_WEIGHTS,
    MISSION_TASK_RISK_WEIGHTS,
    MISSION_LEVEL_WEIGHTS,
    RISK_LEVELS,
)
from app.models.mission import Mission, MissionRiskResult, TaskRiskDetail
from app.models.astronaut import FatigueReading
from app.engine.scoring import calculate_fatigue_score, _classify_risk_level
from app.models.fatigue import FatigueInput


def _normalize_1_to_5(value: int) -> float:
    """Normalize a 1–5 scale value to 0–100."""
    return (value - 1) / 4.0 * 100.0


def _dependency_factor(task_id: str, all_tasks: list) -> float:
    """
    Calculate a dependency load factor (0–100) for a task.

    A task with many downstream dependents is riskier to delay or fail.
    Count how many other tasks depend on this task_id.
    """
    downstream_count = sum(
        1 for t in all_tasks if task_id in t.dependencies
    )
    # Cap at 5 dependents → 100
    return min(downstream_count / 5.0, 1.0) * 100.0


def _compute_task_demand(task, all_tasks: list) -> tuple[float, list[str]]:
    """
    Compute the task demand score for a single task.

    Returns:
        (task_demand_score 0–100, list of human-readable reason strings)
    """
    w = TASK_DEMAND_WEIGHTS
    crit_norm   = _normalize_1_to_5(task.criticality)
    cog_norm    = _normalize_1_to_5(task.cognitive_demand)
    phys_norm   = _normalize_1_to_5(task.physical_demand)
    dep_factor  = _dependency_factor(task.task_id, all_tasks)

    demand = (
        crit_norm  * w["criticality"] +
        cog_norm   * w["cognitive"]   +
        phys_norm  * w["physical"]    +
        dep_factor * w["dependency"]
    )
    demand = round(demand, 2)

    reasons: list[str] = []
    if task.criticality >= 4:
        reasons.append(f"Task criticality is {task.criticality}/5.")
    if task.cognitive_demand >= 4:
        reasons.append(f"Task cognitive demand is {task.cognitive_demand}/5.")
    if task.physical_demand >= 4:
        reasons.append(f"Task physical demand is {task.physical_demand}/5.")
    dep_count = int(dep_factor / 100 * 5)
    if dep_count > 0:
        reasons.append(f"Task has {dep_count} downstream dependent task(s).")

    return demand, reasons


def _fatigue_score_for_astronaut(
    astronaut_id: str,
    fatigue_readings: list[FatigueReading],
) -> Optional[float]:
    """Return the most recent fatigue score for an astronaut, or None."""
    readings = sorted(
        [r for r in fatigue_readings if r.astronaut_id == astronaut_id],
        key=lambda r: r.timestamp,
        reverse=True,
    )
    if not readings:
        return None
    latest = readings[0]
    fi = FatigueInput(
        astronaut_id=latest.astronaut_id,
        timestamp=latest.timestamp,
        pvt_risk=latest.pvt_risk,
        sleep_risk=latest.sleep_risk,
        circadian_risk=latest.circadian_risk,
        workload_risk=latest.workload_risk,
    )
    return calculate_fatigue_score(fi).fatigue_score


def calculate_mission_risk(
    mission: Mission,
    fatigue_readings: list[FatigueReading],
) -> MissionRiskResult:
    """
    Calculate the overall mission risk score.

    Steps:
      1. For each task, find the assigned astronaut's latest fatigue score.
      2. Compute task demand score.
      3. Compute mission_task_risk for each task.
      4. Mission score = (highest task risk * 0.70) + (crew average * 0.30).
      5. Collect human-readable contributing factors and warnings.

    Args:
        mission: Mission definition with tasks and astronaut profiles.
        fatigue_readings: List of fatigue readings for astronauts.

    Returns:
        MissionRiskResult with score, level, breakdown, and warnings.
    """
    task_details: list[TaskRiskDetail] = []
    all_astronaut_ids = {a.astronaut_id for a in mission.astronauts}
    warnings: list[str] = []

    # Build astronaut fatigue map
    fatigue_map: dict[str, float] = {}
    for astronaut in mission.astronauts:
        score = _fatigue_score_for_astronaut(astronaut.astronaut_id, fatigue_readings)
        if score is None:
            warnings.append(
                f"No fatigue reading for astronaut {astronaut.astronaut_id} — "
                "using default fatigue score of 50 (moderate)."
            )
            score = 50.0  # conservative default — never invent, but must proceed
        fatigue_map[astronaut.astronaut_id] = score

    # Compute per-task risk
    for task in mission.tasks:
        a_id = task.assigned_astronaut_id
        if a_id not in all_astronaut_ids:
            warnings.append(
                f"Task {task.task_id} assigned to unknown astronaut {a_id} — skipped."
            )
            continue

        fatigue_score = fatigue_map.get(a_id, 50.0)
        task_demand, demand_reasons = _compute_task_demand(task, mission.tasks)

        mw = MISSION_TASK_RISK_WEIGHTS
        mission_task_risk = round(
            fatigue_score * mw["fatigue"] + task_demand * mw["task_demand"], 2
        )
        risk_level = _classify_risk_level(mission_task_risk)

        reasons: list[str] = []
        reasons.append(f"Astronaut {a_id} fatigue score: {fatigue_score}.")
        reasons.extend(demand_reasons)
        if not task.can_delay and not task.can_reassign:
            reasons.append("Task cannot be delayed or reassigned.")
            warnings.append(
                f"Task {task.task_id} is non-deferrable and assigned to astronaut "
                f"{a_id} with fatigue score {fatigue_score}."
            )

        task_details.append(TaskRiskDetail(
            task_id=task.task_id,
            task_name=task.name,
            astronaut_id=a_id,
            fatigue_score=fatigue_score,
            task_demand_score=task_demand,
            mission_task_risk=mission_task_risk,
            risk_level=risk_level,
            reasons=reasons,
        ))

    if not task_details:
        # No tasks — base mission risk on crew fatigue only
        crew_scores = list(fatigue_map.values())
        mission_risk_score = round(sum(crew_scores) / len(crew_scores), 2) if crew_scores else 0.0
        highest_risk_astronaut = max(fatigue_map, key=fatigue_map.get) if fatigue_map else None
        highest_risk_task = None
        contributing_factors = ["No tasks defined — mission risk based on crew fatigue only."]
    else:
        task_risks = [t.mission_task_risk for t in task_details]
        highest_task_risk = max(task_risks)
        crew_avg = round(sum(task_risks) / len(task_risks), 2)

        lw = MISSION_LEVEL_WEIGHTS
        mission_risk_score = round(
            highest_task_risk * lw["highest_task"] + crew_avg * lw["crew_average"], 2
        )
        mission_risk_score = min(mission_risk_score, 100.0)

        highest_detail = max(task_details, key=lambda t: t.mission_task_risk)
        highest_risk_astronaut = highest_detail.astronaut_id
        highest_risk_task = highest_detail.task_id
        contributing_factors = highest_detail.reasons

    risk_level = _classify_risk_level(mission_risk_score)

    return MissionRiskResult(
        mission_id=mission.mission_id,
        timestamp=datetime.now(timezone.utc),
        mission_risk_score=mission_risk_score,
        risk_level=risk_level,
        highest_risk_astronaut=highest_risk_astronaut,
        highest_risk_task=highest_risk_task,
        task_risk_details=task_details,
        contributing_factors=contributing_factors,
        warnings=warnings,
    )
