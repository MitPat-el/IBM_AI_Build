"""
Data access layer between the DB and the rest of the app.

Reconstructs DB rows into the exact same MissionDayRecord / DriftResult
dataclasses that simulator.py and drift.py already produce in-memory, so
projection.py, replay.py, and bob.py need zero changes to work
against the database instead of the old in-memory MISSION_CACHE.
"""

from sqlalchemy.orm import Session

from db.models import Astronaut, MissionDay, DriftScore, Explanation as ExplanationRow
from drift import DriftResult
from simulator import MissionDayRecord, AstronautProfile
from bob import Explanation


def load_crew(db: Session) -> list[Astronaut]:
    return db.query(Astronaut).order_by(Astronaut.astronaut_id).all()


def load_profile(db: Session, astronaut_id: str) -> AstronautProfile | None:
    """Reconstruct an AstronautProfile from the DB row, including the
    persisted seed -- required so re-simulation (What-If Simulator)
    reproduces the exact same raw signals as what's already stored."""
    row = db.get(Astronaut, astronaut_id)
    if row is None:
        return None
    return AstronautProfile(
        astronaut_id=row.astronaut_id,
        name=row.name,
        baseline_pvt_lapses=row.baseline_pvt_lapses,
        resilience=row.resilience,
        seed=row.seed,
    )


def load_mission_records(db: Session, astronaut_id: str | None = None) -> dict[str, list[MissionDayRecord]]:
    query = db.query(MissionDay, DriftScore).join(DriftScore, DriftScore.mission_day_id == MissionDay.id)
    if astronaut_id:
        query = query.filter(MissionDay.astronaut_id == astronaut_id)
    query = query.order_by(MissionDay.astronaut_id, MissionDay.day)

    result: dict[str, list[MissionDayRecord]] = {}
    for mission_day, drift_row in query.all():
        drift_result = DriftResult(
            reaction_time_score=drift_row.reaction_time_score,
            sleep_debt_score=drift_row.sleep_debt_score,
            circadian_score=drift_row.circadian_score,
            workload_score=drift_row.workload_score,
            drift_score=drift_row.drift_score,
            updated_sleep_debt_hours=0.0,  # not persisted -- only used mid-simulation, not needed on read
            risk_level=drift_row.risk_level,
        )
        record = MissionDayRecord(
            day=mission_day.day,
            astronaut_id=mission_day.astronaut_id,
            hours_slept=mission_day.hours_slept,
            pvt_lapses=mission_day.pvt_lapses,
            minutes_phase_shift=mission_day.minutes_phase_shift,
            task_load=mission_day.task_load,
            rolling_avg_task_load=mission_day.rolling_avg_task_load,
            drift=drift_result,
        )
        result.setdefault(mission_day.astronaut_id, []).append(record)

    return result


def get_cached_explanation(db: Session, astronaut_id: str, day: int) -> ExplanationRow | None:
    return db.query(ExplanationRow).filter_by(astronaut_id=astronaut_id, day=day).first()


def save_explanation(db: Session, astronaut_id: str, day: int, explanation: Explanation) -> ExplanationRow:
    row = get_cached_explanation(db, astronaut_id, day)
    if row is None:
        row = ExplanationRow(astronaut_id=astronaut_id, day=day)
        db.add(row)

    row.astronaut_message = explanation.astronaut_message
    row.flight_surgeon_brief = explanation.flight_surgeon_brief
    row.suggested_intervention = explanation.suggested_intervention
    row.source = explanation.source

    db.flush()
    return row