"""
Mission risk projection.

Two related "project forward" capabilities live here:

  1. Mission Risk Map: given the crew's simulated/real drift trajectories,
     roll them up into a per-day, whole-mission risk picture (worst-case
     astronaut drives the day's overall risk, since a mission is only as
     safe as its most fatigued crew member on a given day).

  2. What-If Simulator: re-run one astronaut's mission with a task-load
     change on a given day (simulating reassignment or delay) and return
     the before/after drift comparison, so an intervention's effect is
     visible before it's actually made.

This file does NOT call any LLM. It only re-runs the deterministic
formula (drift.py) against modified inputs.
"""

from dataclasses import dataclass
from typing import Optional

from drift import DriftWeights
from simulator import (
    AstronautProfile,
    MissionDayRecord,
    simulate_mission,
    DEFAULT_TASK_SCHEDULE,
)


# ---------------------------------------------------------------------------
# Mission Risk Map
# ---------------------------------------------------------------------------
@dataclass
class DayRiskSnapshot:
    day: int
    overall_risk_level: str
    overall_drift_score: float          # max across crew that day
    highest_risk_astronaut: str
    per_astronaut: dict[str, dict]      # astronaut_id -> {drift_score, risk_level}


RISK_ORDER = {"nominal": 0, "elevated": 1, "high": 2, "critical": 3}


def project_mission_risk(
    crew_records: dict[str, list[MissionDayRecord]],
) -> list[DayRiskSnapshot]:
    """
    Roll individual astronaut trajectories up into a whole-mission,
    day-by-day risk map. Drives the Mission Risk Map feature.
    """
    if not crew_records:
        return []

    num_days = max(len(records) for records in crew_records.values())
    snapshots: list[DayRiskSnapshot] = []

    for day_idx in range(num_days):
        day_number = day_idx + 1
        per_astronaut: dict[str, dict] = {}
        worst_astronaut: Optional[str] = None
        worst_score = -1.0
        worst_level = "nominal"

        for astronaut_id, records in crew_records.items():
            if day_idx >= len(records):
                continue
            record = records[day_idx]
            per_astronaut[astronaut_id] = {
                "drift_score": record.drift.drift_score,
                "risk_level": record.drift.risk_level,
            }
            if record.drift.drift_score > worst_score:
                worst_score = record.drift.drift_score
                worst_level = record.drift.risk_level
                worst_astronaut = astronaut_id

        snapshots.append(DayRiskSnapshot(
            day=day_number,
            overall_risk_level=worst_level,
            overall_drift_score=round(worst_score, 4),
            highest_risk_astronaut=worst_astronaut,
            per_astronaut=per_astronaut,
        ))

    return snapshots


def mission_risk_summary(crew_records: dict[str, list[MissionDayRecord]]) -> dict:
    """One-line mission-level verdict, e.g. for a dashboard header."""
    snapshots = project_mission_risk(crew_records)
    if not snapshots:
        return {"overall_risk_level": "nominal", "worst_day": None}

    worst_snapshot = max(
        snapshots,
        key=lambda s: (RISK_ORDER[s.overall_risk_level], s.overall_drift_score),
    )
    return {
        "overall_risk_level": worst_snapshot.overall_risk_level,
        "worst_day": worst_snapshot.day,
        "worst_astronaut": worst_snapshot.highest_risk_astronaut,
        "worst_drift_score": worst_snapshot.overall_drift_score,
    }


# ---------------------------------------------------------------------------
# What-If Simulator
# ---------------------------------------------------------------------------
@dataclass
class WhatIfResult:
    astronaut_id: str
    day_modified: int
    reassigned_to: Optional[str]
    comparison: list[dict]


def simulate_intervention(
    profile: AstronautProfile,
    original_records: list[MissionDayRecord],
    day: int,
    task_load_delta: float = -6.0,
    reassign_to: Optional[str] = None,
    weights: DriftWeights = DriftWeights(),
) -> WhatIfResult:
    """
    Re-run one astronaut's mission with reduced task load on `day` onward
    from that point (simulating handing a task off to `reassign_to`, or
    just delaying/dropping it if reassign_to is None), and diff against
    the original trajectory.
    """
    schedule = list(DEFAULT_TASK_SCHEDULE[:len(original_records)])
    if 1 <= day <= len(schedule):
        schedule[day - 1] = max(0.0, schedule[day - 1] + task_load_delta)

    new_records = simulate_mission(profile, num_days=len(schedule), task_schedule=schedule, weights=weights)

    comparison = [
        {
            "day": orig.day,
            "original_drift": orig.drift.drift_score,
            "whatif_drift": new.drift.drift_score,
            "delta": round(new.drift.drift_score - orig.drift.drift_score, 4),
            "original_risk_level": orig.drift.risk_level,
            "whatif_risk_level": new.drift.risk_level,
        }
        for orig, new in zip(original_records, new_records)
    ]

    return WhatIfResult(
        astronaut_id=profile.astronaut_id,
        day_modified=day,
        reassigned_to=reassign_to,
        comparison=comparison,
    )


if __name__ == "__main__":
    crew = [
        AstronautProfile(astronaut_id="A1", name="Chen", baseline_pvt_lapses=2.5, seed=1),
        AstronautProfile(astronaut_id="A2", name="Okafor", baseline_pvt_lapses=3.5, resilience=1.15, seed=2),
    ]
    records = {p.astronaut_id: simulate_mission(p) for p in crew}

    print("--- Mission Risk Map ---")
    for snap in project_mission_risk(records):
        print(snap)

    print("\n--- Mission summary ---")
    print(mission_risk_summary(records))

    print("\n--- What-if: reassign A2's day-4 task ---")
    result = simulate_intervention(crew[1], records["A2"], day=4, task_load_delta=-8, reassign_to="A1")
    for row in result.comparison:
        print(row)