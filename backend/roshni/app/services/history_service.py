"""
services/history_service.py — Read/write mission history records.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session

from app.db.models import HistoryRecord


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

class HistoryRecordInput:
    """Simple data-transfer object for creating a history record."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def record_snapshot(db: Session, data: dict) -> HistoryRecord:
    """
    Persist one fatigue/mission snapshot to the database.

    Args:
        db:   SQLAlchemy session (injected via FastAPI dependency).
        data: Dict matching the HistoryRecord columns.

    Returns:
        The saved HistoryRecord ORM object.
    """
    record = HistoryRecord(
        mission_id=data["mission_id"],
        astronaut_id=data["astronaut_id"],
        mission_day=data.get("mission_day"),
        timestamp=data["timestamp"],
        pvt_risk=data["pvt_risk"],
        sleep_risk=data["sleep_risk"],
        circadian_risk=data["circadian_risk"],
        workload_risk=data["workload_risk"],
        fatigue_score=data["fatigue_score"],
        fatigue_level=data["fatigue_level"],
        trend=data.get("trend"),
        mission_risk=data.get("mission_risk"),
        mission_risk_level=data.get("mission_risk_level"),
        task_id=data.get("task_id"),
        intervention_type=data.get("intervention_type"),
        intervention_detail=data.get("intervention_detail"),
        data_quality_score=data.get("data_quality_score"),
        data_quality_level=data.get("data_quality_level"),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_astronaut_history(
    db: Session,
    astronaut_id: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    mission_day: Optional[int] = None,
) -> list[HistoryRecord]:
    """Return history records for one astronaut, ordered by timestamp."""
    q = db.query(HistoryRecord).filter(HistoryRecord.astronaut_id == astronaut_id)
    if start_time:
        q = q.filter(HistoryRecord.timestamp >= start_time)
    if end_time:
        q = q.filter(HistoryRecord.timestamp <= end_time)
    if mission_day is not None:
        q = q.filter(HistoryRecord.mission_day == mission_day)
    return q.order_by(HistoryRecord.timestamp.asc()).all()


def get_mission_history(
    db: Session,
    mission_id: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    mission_day: Optional[int] = None,
) -> list[HistoryRecord]:
    """Return history records for an entire mission, ordered by timestamp."""
    q = db.query(HistoryRecord).filter(HistoryRecord.mission_id == mission_id)
    if start_time:
        q = q.filter(HistoryRecord.timestamp >= start_time)
    if end_time:
        q = q.filter(HistoryRecord.timestamp <= end_time)
    if mission_day is not None:
        q = q.filter(HistoryRecord.mission_day == mission_day)
    return q.order_by(HistoryRecord.timestamp.asc()).all()
