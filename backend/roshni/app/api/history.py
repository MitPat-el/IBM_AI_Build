"""
api/history.py — Mission history routes.

Routes:
  POST /history/record                   — persist a snapshot
  GET  /history/astronaut/{astronaut_id} — get timeline for one astronaut
  GET  /history/mission/{mission_id}     — get timeline for a mission
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services import history_service

router = APIRouter(prefix="/history", tags=["Historical Replay"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class HistoryRecordRequest(BaseModel):
    """Body for POST /history/record."""
    mission_id: str
    astronaut_id: str
    mission_day: Optional[int] = None
    timestamp: datetime
    pvt_risk: float
    sleep_risk: float
    circadian_risk: float
    workload_risk: float
    fatigue_score: float
    fatigue_level: str
    trend: Optional[str] = None
    mission_risk: Optional[float] = None
    mission_risk_level: Optional[str] = None
    task_id: Optional[str] = None
    intervention_type: Optional[str] = None
    intervention_detail: Optional[str] = None
    data_quality_score: Optional[float] = None
    data_quality_level: Optional[str] = None


class TimelineEntry(BaseModel):
    """One point on the history timeline — format optimised for frontend graphing."""
    id: int
    mission_id: str
    astronaut_id: str
    mission_day: Optional[int]
    timestamp: datetime
    fatigue_score: float
    fatigue_level: str
    trend: Optional[str]
    mission_risk: Optional[float]
    mission_risk_level: Optional[str]
    task_id: Optional[str]
    intervention_type: Optional[str]
    pvt_risk: float
    sleep_risk: float
    circadian_risk: float
    workload_risk: float
    data_quality_score: Optional[float]
    data_quality_level: Optional[str]


def _to_timeline(record) -> TimelineEntry:
    return TimelineEntry(
        id=record.id,
        mission_id=record.mission_id,
        astronaut_id=record.astronaut_id,
        mission_day=record.mission_day,
        timestamp=record.timestamp,
        fatigue_score=record.fatigue_score,
        fatigue_level=record.fatigue_level,
        trend=record.trend,
        mission_risk=record.mission_risk,
        mission_risk_level=record.mission_risk_level,
        task_id=record.task_id,
        intervention_type=record.intervention_type,
        pvt_risk=record.pvt_risk,
        sleep_risk=record.sleep_risk,
        circadian_risk=record.circadian_risk,
        workload_risk=record.workload_risk,
        data_quality_score=record.data_quality_score,
        data_quality_level=record.data_quality_level,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/record",
    response_model=TimelineEntry,
    summary="Record a Mission History Snapshot",
    description="Persists one timestamped fatigue/mission-risk snapshot to the history database.",
)
def record(payload: HistoryRecordRequest, db: Session = Depends(get_db)) -> TimelineEntry:
    saved = history_service.record_snapshot(db, payload.model_dump())
    return _to_timeline(saved)


@router.get(
    "/astronaut/{astronaut_id}",
    response_model=list[TimelineEntry],
    summary="Get Astronaut History Timeline",
    description="Returns ordered timeline data for one astronaut. Supports optional date/day filters.",
)
def astronaut_history(
    astronaut_id: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    mission_day: Optional[int] = Query(None),
    db: Session = Depends(get_db),
) -> list[TimelineEntry]:
    records = history_service.get_astronaut_history(db, astronaut_id, start_time, end_time, mission_day)
    return [_to_timeline(r) for r in records]


@router.get(
    "/mission/{mission_id}",
    response_model=list[TimelineEntry],
    summary="Get Mission History Timeline",
    description="Returns ordered timeline data for an entire mission. Supports optional date/day filters.",
)
def mission_history(
    mission_id: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    mission_day: Optional[int] = Query(None),
    db: Session = Depends(get_db),
) -> list[TimelineEntry]:
    records = history_service.get_mission_history(db, mission_id, start_time, end_time, mission_day)
    return [_to_timeline(r) for r in records]
