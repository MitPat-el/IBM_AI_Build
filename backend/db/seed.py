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
from db.models import Astronaut, MissionDay, DriftScore, Explanation
from simulator import AstronautProfile, simulate_crew, resolve_seed
from drift import DriftWeights

DEFAULT_CREW = [
    AstronautProfile(astronaut_id="A1", name="Chen", baseline_pvt_lapses=2.5, seed=1),
    AstronautProfile(astronaut_id="A2", name="Okafor", baseline_pvt_lapses=3.5, resilience=1.15, seed=2),
    AstronautProfile(astronaut_id="A3", name="Ivanova", baseline_pvt_lapses=3.0, resilience=0.9, seed=3),
]


def seed(num_days: int = 6, crew=None, wipe_existing: bool = True, weights: DriftWeights = DriftWeights()):
    crew = crew or DEFAULT_CREW
    init_db()

    with get_session() as db:
        if wipe_existing:
            db.query(Explanation).delete()  # clear cached AI explanations -- they'd reference stale drift scores otherwise
            db.query(DriftScore).delete()
            db.query(MissionDay).delete()
            db.query(Astronaut).delete()

        mission_data = simulate_crew(crew, num_days=num_days, weights=weights)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-days", type=int, default=6)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    seed(num_days=args.num_days, wipe_existing=not args.keep_existing)