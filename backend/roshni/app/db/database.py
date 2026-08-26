"""
db/database.py — SQLite database setup using SQLAlchemy.

Uses a local SQLite file for the hackathon prototype.
To switch to PostgreSQL later, change DATABASE_URL and install psycopg2.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# Default: SQLite file in the project root. Override via DATABASE_URL env var.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fatigue_mission.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Called once at application startup."""
    from app.db import models  # noqa: F401 — import to register ORM models
    Base.metadata.create_all(bind=engine)
