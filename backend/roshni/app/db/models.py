"""
db/models.py — SQLAlchemy ORM models for persisting mission history.
"""

from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, Text, Boolean
from app.db.database import Base


class HistoryRecord(Base):
    """
    Stores one timestamped snapshot of astronaut fatigue and mission risk.
    Each API call to POST /history/record writes one row.
    """
    __tablename__ = "history_records"

    id = Column(Integer, primary_key=True, index=True)

    # Identity
    mission_id       = Column(String(64), index=True, nullable=False)
    astronaut_id     = Column(String(64), index=True, nullable=False)
    mission_day      = Column(Integer, nullable=True)
    timestamp        = Column(DateTime, nullable=False, index=True)

    # Fatigue inputs
    pvt_risk         = Column(Float, nullable=False)
    sleep_risk       = Column(Float, nullable=False)
    circadian_risk   = Column(Float, nullable=False)
    workload_risk    = Column(Float, nullable=False)

    # Computed outputs
    fatigue_score    = Column(Float, nullable=False)
    fatigue_level    = Column(String(16), nullable=False)
    trend            = Column(String(20), nullable=True)

    # Mission context
    mission_risk     = Column(Float, nullable=True)
    mission_risk_level = Column(String(16), nullable=True)
    task_id          = Column(String(64), nullable=True)

    # Intervention (if one was applied at this step)
    intervention_type   = Column(String(32), nullable=True)
    intervention_detail = Column(Text, nullable=True)  # JSON string

    # Data quality
    data_quality_score = Column(Float, nullable=True)
    data_quality_level = Column(String(8), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
