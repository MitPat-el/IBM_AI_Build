"""
services/what_if_service.py — What-If Intervention Simulator.

CRITICAL RULE: This service NEVER modifies the real mission state.
It works on a deep copy of the input and returns a before/after comparison.

Feasibility is determined by the deterministic rules in this file.
AI never decides whether an intervention is allowed.

Supported interventions:
  REASSIGN_TASK    — move task from one astronaut to another
  DELAY_TASK       — push task start time forward
  REDUCE_WORKLOAD  — lower workload_risk for the assigned astronaut
"""

import copy
from datetime import datetime, timezone

from app.models.intervention import Intervention, WhatIfResult
from app.models.mission import Mission, MissionRiskResult
from app.models.astronaut import FatigueReading
from app.models.fatigue import FatigueInput
from app.engine.scoring import calculate_fatigue_score
from app.services.mission_risk_service import calculate_mission_risk


def _get_fatigue_score(astronaut_id: str, readings: list[FatigueReading]) -> float:
    """Return most-recent fatigue score for an astronaut, defaulting to 50."""
    sorted_r = sorted(
        [r for r in readings if r.astronaut_id == astronaut_id],
        key=lambda r: r.timestamp, reverse=True,
    )
    if not sorted_r:
        return 50.0
    r = sorted_r[0]
    fi = FatigueInput(
        astronaut_id=r.astronaut_id,
        timestamp=r.timestamp,
        pvt_risk=r.pvt_risk,
        sleep_risk=r.sleep_risk,
        circadian_risk=r.circadian_risk,
        workload_risk=r.workload_risk,
    )
    return calculate_fatigue_score(fi).fatigue_score


def simulate(
    intervention: Intervention,
    mission: Mission,
    fatigue_readings: list[FatigueReading],
) -> WhatIfResult:
    """
    Run a What-If simulation.

    1. Calculate BEFORE scores (original mission state).
    2. Deep copy mission + readings.
    3. Attempt to apply the intervention to the copy.
       - Collect constraint violations without touching original data.
    4. If feasible, calculate AFTER scores on the copy.
    5. Return WhatIfResult with before/after and explanation data.

    Args:
        intervention: The proposed change.
        mission: Current mission definition (NOT modified).
        fatigue_readings: Current fatigue readings (NOT modified).

    Returns:
        WhatIfResult with feasibility, risk delta, and explanation data.
    """
    # --- BEFORE ---
    before_mission_result: MissionRiskResult = calculate_mission_risk(mission, fatigue_readings)
    before_mission_risk = before_mission_result.mission_risk_score

    # Get target task
    target_task = next((t for t in mission.tasks if t.task_id == intervention.task_id), None)
    if target_task is None:
        return WhatIfResult(
            before_fatigue_score=0.0,
            after_fatigue_score=0.0,
            before_mission_risk=before_mission_risk,
            after_mission_risk=before_mission_risk,
            risk_change=0.0,
            intervention=intervention,
            feasible=False,
            constraint_violations=[f"Task '{intervention.task_id}' not found in mission."],
            explanation_data={},
        )

    before_fatigue_score = _get_fatigue_score(target_task.assigned_astronaut_id, fatigue_readings)

    # --- DEEP COPY (real state never touched) ---
    sim_mission = mission.model_copy(deep=True)
    sim_readings = copy.deepcopy(fatigue_readings)
    violations: list[str] = []

    # Find the task in the copy
    sim_task = next(t for t in sim_mission.tasks if t.task_id == intervention.task_id)

    # ----------------------------------------------------------------
    # Apply intervention to the COPY
    # ----------------------------------------------------------------
    itype = intervention.intervention_type

    if itype == "REASSIGN_TASK":
        to_id = intervention.to_astronaut_id
        if not to_id:
            violations.append("to_astronaut_id is required for REASSIGN_TASK.")
        else:
            to_astronaut = next((a for a in sim_mission.astronauts if a.astronaut_id == to_id), None)
            if to_astronaut is None:
                violations.append(f"Astronaut '{to_id}' not found in mission.")
            else:
                # Check qualifications
                missing_quals = [
                    q for q in sim_task.required_qualifications
                    if q not in to_astronaut.qualifications
                ]
                if missing_quals:
                    violations.append(
                        f"Astronaut {to_id} lacks required qualification(s): "
                        + ", ".join(missing_quals) + "."
                    )
                # Check if task can be reassigned
                if not sim_task.can_reassign:
                    violations.append(f"Task '{sim_task.task_id}' is marked as non-reassignable.")

                # Check target fatigue — warn but don't block
                target_fatigue = _get_fatigue_score(to_id, sim_readings)
                warning_added = False
                if target_fatigue >= 85:
                    violations.append(
                        f"Astronaut {to_id} has CRITICAL fatigue score ({target_fatigue}). "
                        "Reassignment would create unacceptable risk."
                    )

                if not violations:
                    # Apply: reassign the task
                    sim_task.assigned_astronaut_id = to_id

    elif itype == "DELAY_TASK":
        if not intervention.delay_minutes:
            violations.append("delay_minutes is required for DELAY_TASK.")
        elif not sim_task.can_delay:
            violations.append(f"Task '{sim_task.task_id}' is marked as non-delayable.")
        else:
            from datetime import timedelta
            sim_task.start_time = sim_task.start_time + timedelta(minutes=intervention.delay_minutes)
            # Check dependency conflicts: if any task that depends on this one
            # would now start before we finish, flag it.
            new_end = sim_task.start_time + __import__("datetime").timedelta(
                minutes=sim_task.duration_minutes
            )
            for other in sim_mission.tasks:
                if sim_task.task_id in other.dependencies:
                    if other.start_time < new_end:
                        violations.append(
                            f"Delaying task '{sim_task.task_id}' by {intervention.delay_minutes}m "
                            f"conflicts with dependent task '{other.task_id}' "
                            f"scheduled at {other.start_time.isoformat()}."
                        )

    elif itype == "REDUCE_WORKLOAD":
        if not intervention.workload_reduction:
            violations.append("workload_reduction is required for REDUCE_WORKLOAD.")
        else:
            # Apply to the most recent reading for the assigned astronaut
            a_id = sim_task.assigned_astronaut_id
            sorted_r = sorted(
                [r for r in sim_readings if r.astronaut_id == a_id],
                key=lambda r: r.timestamp, reverse=True,
            )
            if sorted_r:
                sorted_r[0].workload_risk = max(
                    0.0, sorted_r[0].workload_risk - intervention.workload_reduction
                )
            else:
                violations.append(f"No fatigue reading found for astronaut {a_id}.")

    # ----------------------------------------------------------------
    # AFTER scores — only if feasible
    # ----------------------------------------------------------------
    feasible = len(violations) == 0

    if feasible:
        after_mission_result = calculate_mission_risk(sim_mission, sim_readings)
        after_mission_risk = after_mission_result.mission_risk_score
        after_fatigue_score = _get_fatigue_score(sim_task.assigned_astronaut_id, sim_readings)
    else:
        after_mission_risk = before_mission_risk
        after_fatigue_score = before_fatigue_score

    risk_change = round(after_mission_risk - before_mission_risk, 2)

    explanation_data = {
        "intervention_type": itype,
        "task_id": intervention.task_id,
        "task_name": target_task.name,
        "before_fatigue_score": before_fatigue_score,
        "after_fatigue_score": after_fatigue_score,
        "before_mission_risk": before_mission_risk,
        "after_mission_risk": after_mission_risk,
        "risk_change": risk_change,
        "feasible": feasible,
        "constraint_violations": violations,
    }

    return WhatIfResult(
        before_fatigue_score=before_fatigue_score,
        after_fatigue_score=after_fatigue_score,
        before_mission_risk=before_mission_risk,
        after_mission_risk=after_mission_risk,
        risk_change=risk_change,
        intervention=intervention,
        feasible=feasible,
        constraint_violations=violations,
        explanation_data=explanation_data,
    )
