"""
Historical Replay.

Feeds the "showcase astronaut's health before/during/after mission" +
"timeline slider" feature. Two things live here:

  1. get_replay(): flattens an astronaut's recorded drift trajectory into
     a scrubbable timeline (one entry per day, all sub-scores included so
     the frontend can plot each signal independently, not just the
     composite score).

  2. project_forward(): a simple, transparent trend-based projection of
     where drift is headed if nothing changes -- NOT another AI call, just
     linear extrapolation over the last few recorded days. This powers the
     "possible prediction" part of the timeline slider without pretending
     to be a clinical forecast.
"""

from dataclasses import dataclass
from typing import Optional

from simulator import MissionDayRecord


@dataclass
class TimelinePoint:
    day: int
    phase: str  # "pre_mission" | "in_mission" | "post_mission"
    drift_score: float
    risk_level: str
    hours_slept: float
    pvt_lapses: int
    minutes_phase_shift: float
    task_load: float
    sub_scores: dict


@dataclass
class ReplayTimeline:
    astronaut_id: str
    points: list[TimelinePoint]
    projected_points: list[TimelinePoint]  # empty unless project_forward() was run


def _phase_for_day(day: int, mission_start_day: int, mission_end_day: int) -> str:
    if day < mission_start_day:
        return "pre_mission"
    if day > mission_end_day:
        return "post_mission"
    return "in_mission"


def get_replay(
    astronaut_id: str,
    records: list[MissionDayRecord],
    mission_start_day: int = 1,
    mission_end_day: Optional[int] = None,
) -> ReplayTimeline:
    """
    Build the scrubbable timeline for one astronaut. Currently sources
    purely from in-mission simulated/real records -- pre/post-mission
    phases are labeled for when that data exists, so the frontend slider
    logic doesn't need to change later.
    """
    mission_end_day = mission_end_day or (records[-1].day if records else 0)

    points = [
        TimelinePoint(
            day=r.day,
            phase=_phase_for_day(r.day, mission_start_day, mission_end_day),
            drift_score=r.drift.drift_score,
            risk_level=r.drift.risk_level,
            hours_slept=r.hours_slept,
            pvt_lapses=r.pvt_lapses,
            minutes_phase_shift=r.minutes_phase_shift,
            task_load=r.task_load,
            sub_scores={
                "reaction_time": r.drift.reaction_time_score,
                "sleep_debt": r.drift.sleep_debt_score,
                "circadian": r.drift.circadian_score,
                "workload": r.drift.workload_score,
            },
        )
        for r in records
    ]

    return ReplayTimeline(astronaut_id=astronaut_id, points=points, projected_points=[])


def project_forward(timeline: ReplayTimeline, num_days: int = 3, lookback: int = 3) -> ReplayTimeline:
    """
    Simple linear trend extrapolation from the last `lookback` recorded
    days. Transparent and re-derivable by hand -- deliberately NOT a
    model, so it can't be mistaken for a clinical prediction. Clips to
    [0, 1] same as the real drift score.
    """
    points = timeline.points
    if len(points) < 2:
        return timeline

    recent = points[-lookback:] if len(points) >= lookback else points
    n = len(recent)
    avg_day = sum(p.day for p in recent) / n
    avg_score = sum(p.drift_score for p in recent) / n

    numerator = sum((p.day - avg_day) * (p.drift_score - avg_score) for p in recent)
    denominator = sum((p.day - avg_day) ** 2 for p in recent) or 1e-9
    slope = numerator / denominator

    last_day = points[-1].day
    last_score = points[-1].drift_score

    projected: list[TimelinePoint] = []
    for step in range(1, num_days + 1):
        day = last_day + step
        score = max(0.0, min(1.0, last_score + slope * step))
        level = "nominal" if score < 0.35 else "elevated" if score < 0.55 else "high" if score < 0.75 else "critical"
        projected.append(TimelinePoint(
            day=day,
            phase="in_mission",
            drift_score=round(score, 4),
            risk_level=level,
            hours_slept=points[-1].hours_slept,       # held constant -- projection, not simulation
            pvt_lapses=points[-1].pvt_lapses,
            minutes_phase_shift=points[-1].minutes_phase_shift,
            task_load=points[-1].task_load,
            sub_scores=points[-1].sub_scores,          # not extrapolated individually, composite only
        ))

    timeline.projected_points = projected
    return timeline


if __name__ == "__main__":
    from simulator import AstronautProfile, simulate_mission

    profile = AstronautProfile(astronaut_id="A2", name="Okafor", baseline_pvt_lapses=3.5, resilience=1.15, seed=2)
    records = simulate_mission(profile)

    timeline = get_replay("A2", records)
    timeline = project_forward(timeline, num_days=3)

    print("--- Recorded ---")
    for p in timeline.points:
        print(p.day, p.phase, p.drift_score, p.risk_level)

    print("--- Projected (trend only, not a model) ---")
    for p in timeline.projected_points:
        print(p.day, p.phase, p.drift_score, p.risk_level)