"""
Seed the database with fake mission data.

Reuses simulator.py (the same synthetic data generator already used for
in-memory demos) so the fake data in the DB is identical in shape and
realism to what's already been validated -- this isn't a second, separate
fake-data generator to maintain.

Run (from inside app/):
  python3 -m db.seed                  # wipes and reseeds with default 3-astronaut crew
  python3 -m db.seed --num-days 10    # longer mission
  python3 -m db.seed --keep-existing  # don't drop existing rows first
"""

import argparse

from db.session import init_db, get_session
from db.models import Astronaut, MissionDay, DriftScore, Explanation, Task, TaskDependency
from simulator import AstronautProfile, simulate_crew, resolve_seed
from drift import DriftWeights

DEFAULT_CREW = [
    AstronautProfile(astronaut_id="A1", name="Chen", baseline_pvt_lapses=2.5, seed=1),
    AstronautProfile(astronaut_id="A2", name="Okafor", baseline_pvt_lapses=3.5, resilience=1.15, seed=2),
    AstronautProfile(astronaut_id="A3", name="Ivanova", baseline_pvt_lapses=3.0, resilience=0.9, seed=3),
]

# A grounded, ISS-ops-style task DAG for the Dependency Graph / cascading
# What-If impact analysis. Deliberately crosses astronauts and days so a
# single slipped task can ripple through the rest of the mission -- e.g.
# T1 (A1, day 1) -> T2 -> T5 -> T7 -> T10 -> T13 -> T16 -> T17 -> T18 is
# one long critical-path chain spanning all six days and all three crew.
# Only seeded for the default 3-astronaut crew, and only up to num_days,
# since a custom crew or shorter mission won't have matching astronaut
# ids / days for every task.
DEFAULT_TASKS = [
    ("T1",  "Power Up External Payload Bay",  1, "A1", 3),
    ("T2",  "Calibrate Spectrometer",          1, "A2", 2),
    ("T3",  "Daily Systems Check",             1, "A3", 1),
    ("T4",  "EVA Prep - Suit Check",           2, "A1", 4),
    ("T5",  "Airlock Depressurization",        2, "A2", 3),
    ("T6",  "Sample Collection Log",           2, "A3", 2),
    ("T7",  "EVA - External Repair",           3, "A1", 6),
    ("T8",  "Telemetry Sync",                  3, "A2", 2),
    ("T9",  "Exercise Protocol",               3, "A3", 1),
    ("T10", "Cargo Transfer",                  4, "A2", 3),
    ("T11", "Medical Checkup - Crew",          4, "A1", 2),
    ("T12", "Data Downlink",                   4, "A3", 2),
    ("T13", "Robotic Arm Ops",                 5, "A1", 5),
    ("T14", "Experiment Monitoring",           5, "A2", 2),
    ("T15", "Waste Management",                5, "A3", 1),
    ("T16", "EVA - Panel Install",              6, "A1", 6),
    ("T17", "Final Systems Check",             6, "A2", 3),
    ("T18", "Mission Report Compile",          6, "A3", 2),
]

# (task_id, depends_on_id) -- task_id requires depends_on_id to finish first
DEFAULT_DEPENDENCIES = [
    ("T2", "T1"),
    ("T4", "T3"),
    ("T5", "T4"),
    ("T7", "T5"),
    ("T10", "T7"),
    ("T8", "T2"),
    ("T12", "T8"),
    ("T13", "T10"),
    ("T16", "T13"),
    ("T17", "T16"),
    ("T18", "T12"),
    ("T18", "T17"),
]


def task_derived_schedules(crew, num_days: int) -> dict[str, list[float]]:
    """
    Sums each astronaut's actual assigned task loads per day into a
    per-astronaut daily workload schedule, so the drift formula's
    workload signal reflects real task assignments instead of a single
    number shared identically across the whole crew. This is what makes
    the Dependency Graph and the fatigue model agree with each other --
    without it, a task's `load` (used for cascading impact analysis) and
    the day's `task_load` (used for the drift score) are two unrelated
    numbers that happen to share a name.

    Falls back to simulator.DEFAULT_TASK_SCHEDULE's value for any
    astronaut-day combination with no assigned tasks (keeps behavior
    sane for missions longer than the seeded task set, or gaps in it).
    """
    from simulator import DEFAULT_TASK_SCHEDULE

    crew_ids = {p.astronaut_id for p in crew}
    schedules = {aid: [0.0] * num_days for aid in crew_ids}
    covered = {aid: [False] * num_days for aid in crew_ids}

    for task_id, name, day, astronaut_id, load in DEFAULT_TASKS:
        if astronaut_id in schedules and day <= num_days:
            schedules[astronaut_id][day - 1] += load
            covered[astronaut_id][day - 1] = True

    for aid in crew_ids:
        for i in range(num_days):
            if not covered[aid][i]:
                schedules[aid][i] = DEFAULT_TASK_SCHEDULE[min(i, len(DEFAULT_TASK_SCHEDULE) - 1)]

    return schedules


def seed(num_days: int = 6, crew=None, wipe_existing: bool = True, weights: DriftWeights = DriftWeights()):
    crew = crew or DEFAULT_CREW
    init_db()

    with get_session() as db:
        if wipe_existing:
            db.query(Explanation).delete()  # clear cached AI explanations -- they'd reference stale drift scores otherwise
            db.query(TaskDependency).delete()
            db.query(Task).delete()
            db.query(DriftScore).delete()
            db.query(MissionDay).delete()
            db.query(Astronaut).delete()

        crew_ids = {p.astronaut_id for p in crew}
        default_ids = {"A1", "A2", "A3"}
        task_schedules = task_derived_schedules(crew, num_days) if default_ids.issubset(crew_ids) else None

        mission_data = simulate_crew(crew, num_days=num_days, weights=weights, task_schedules=task_schedules)

        for profile in crew:
            existing = db.get(Astronaut, profile.astronaut_id)
            if existing is None:
                resolved_seed = profile.seed if profile.seed is not None else resolve_seed(profile.astronaut_id)
                db.add(Astronaut(
                    astronaut_id=profile.astronaut_id,
                    name=profile.name,
                    baseline_pvt_lapses=profile.baseline_pvt_lapses,
                    resilience=profile.resilience,
                    seed=resolved_seed,
                ))

        db.flush()  # so FK inserts below can see the astronaut rows

        row_count = 0
        skipped_count = 0
        for astronaut_id, records in mission_data.items():
            for record in records:
                if not wipe_existing:
                    already_exists = db.query(MissionDay).filter_by(
                        astronaut_id=astronaut_id, day=record.day
                    ).first()
                    if already_exists:
                        skipped_count += 1
                        continue

                mission_day = MissionDay(
                    astronaut_id=astronaut_id,
                    day=record.day,
                    hours_slept=record.hours_slept,
                    pvt_lapses=record.pvt_lapses,
                    minutes_phase_shift=record.minutes_phase_shift,
                    task_load=record.task_load,
                    rolling_avg_task_load=record.rolling_avg_task_load,
                )
                db.add(mission_day)
                db.flush()  # get mission_day.id for the FK below

                db.add(DriftScore(
                    mission_day_id=mission_day.id,
                    reaction_time_score=record.drift.reaction_time_score,
                    sleep_debt_score=record.drift.sleep_debt_score,
                    circadian_score=record.drift.circadian_score,
                    workload_score=record.drift.workload_score,
                    drift_score=record.drift.drift_score,
                    risk_level=record.drift.risk_level,
                ))
                row_count += 1

    print(f"Seeded {len(crew)} astronauts x {num_days} days = {row_count} mission-day rows."
          + (f" Skipped {skipped_count} existing rows." if skipped_count else ""))

    _seed_tasks(crew, num_days, wipe_existing)


def _seed_tasks(crew, num_days: int, wipe_existing: bool):
    """Seeds the demo task DAG. Only applies when the crew includes the
    default astronaut ids (A1/A2/A3) the task list was written for, and
    only inserts tasks whose day fits within num_days -- a custom crew
    or a shorter mission just won't get a task graph, rather than
    inserting tasks that reference astronauts or days that don't exist."""
    crew_ids = {p.astronaut_id for p in crew}
    default_ids = {"A1", "A2", "A3"}
    if not default_ids.issubset(crew_ids):
        return

    with get_session() as db:
        if not wipe_existing and db.query(Task).first() is not None:
            return  # tasks already seeded and we're not wiping -- don't duplicate

        task_count = 0
        for task_id, name, day, astronaut_id, load in DEFAULT_TASKS:
            if day > num_days:
                continue
            db.add(Task(task_id=task_id, name=name, day=day, astronaut_id=astronaut_id, load=load))
            task_count += 1

        db.flush()
        seeded_task_ids = {t[0] for t in DEFAULT_TASKS if t[2] <= num_days}
        edge_count = 0
        for task_id, depends_on_id in DEFAULT_DEPENDENCIES:
            if task_id in seeded_task_ids and depends_on_id in seeded_task_ids:
                db.add(TaskDependency(task_id=task_id, depends_on_id=depends_on_id))
                edge_count += 1

    print(f"Seeded {task_count} tasks, {edge_count} dependency edges.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-days", type=int, default=6)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    seed(num_days=args.num_days, wipe_existing=not args.keep_existing)