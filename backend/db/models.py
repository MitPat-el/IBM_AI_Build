"""
Database schema, via SQLAlchemy ORM.

Deliberately dialect-agnostic: no SQLite-only or Postgres-only types, so
swapping DATABASE_URL from sqlite:/// to postgresql:// or mysql:// later
requires zero changes here. That swap happens in session.py only.

Tables mirror exactly what the existing pipeline already produces:
  Astronaut     <- simulator.AstronautProfile
  MissionDay    <- simulator.MissionDayRecord (raw signals)
  DriftScore    <- drift.DriftResult (computed sub-scores + composite)
  Explanation   <- bob_client.Explanation (cached AI output, so Bob isn't
                   re-called every time someone re-views the same alert)
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, Float, String, ForeignKey, DateTime, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Astronaut(Base):
    __tablename__ = "astronauts"

    astronaut_id = Column(String(50), primary_key=True)  # e.g. "A1"
    name = Column(String(100), nullable=False)
    baseline_pvt_lapses = Column(Float, nullable=False, default=3.0)
    resilience = Column(Float, nullable=False, default=1.0)
    seed = Column(Integer, nullable=False)  # resolved RNG seed used to generate this astronaut's data -- persisted so re-simulation (What-If) reproduces identical raw signals

    mission_days = relationship("MissionDay", back_populates="astronaut", cascade="all, delete-orphan")


class MissionDay(Base):
    """Raw daily signals -- what a real sensor/wearable/scheduling feed would write."""
    __tablename__ = "mission_days"
    __table_args__ = (UniqueConstraint("astronaut_id", "day", name="uq_astronaut_day"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    astronaut_id = Column(String(50), ForeignKey("astronauts.astronaut_id"), nullable=False)
    day = Column(Integer, nullable=False)

    hours_slept = Column(Float, nullable=False)
    pvt_lapses = Column(Integer, nullable=False)
    minutes_phase_shift = Column(Float, nullable=False)
    task_load = Column(Float, nullable=False)
    rolling_avg_task_load = Column(Float, nullable=False)

    astronaut = relationship("Astronaut", back_populates="mission_days")
    drift_score = relationship("DriftScore", back_populates="mission_day", uselist=False, cascade="all, delete-orphan")


class DriftScore(Base):
    """Computed output of drift.compute_drift_score() for one astronaut-day."""
    __tablename__ = "drift_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mission_day_id = Column(Integer, ForeignKey("mission_days.id"), nullable=False, unique=True)

    reaction_time_score = Column(Float, nullable=False)
    sleep_debt_score = Column(Float, nullable=False)
    circadian_score = Column(Float, nullable=False)
    workload_score = Column(Float, nullable=False)
    drift_score = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)  # nominal | elevated | high | critical

    mission_day = relationship("MissionDay", back_populates="drift_score")


class Explanation(Base):
    """Cached Bob/watsonx output -- one row per astronaut-day that crossed threshold."""
    __tablename__ = "explanations"
    __table_args__ = (UniqueConstraint("astronaut_id", "day", name="uq_explanation_astronaut_day"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    astronaut_id = Column(String(50), ForeignKey("astronauts.astronaut_id"), nullable=False)
    day = Column(Integer, nullable=False)

    astronaut_message = Column(String(500), nullable=False)
    flight_surgeon_brief = Column(String(1000), nullable=False)
    suggested_intervention = Column(String(500), nullable=False)
    source = Column(String(20), nullable=False)  # "watsonx" | "fallback_template"
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))