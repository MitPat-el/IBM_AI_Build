"""
Database engine + session setup.

This is the ONLY file that needs to change when you plug in your real
database. Right now DATABASE_URL defaults to a local SQLite file so
everyone on the team can run the app with zero setup.

To point at a real database, just set the env var, e.g.:
  export DATABASE_URL="postgresql://user:pass@host:5432/astrofatigue"
  export DATABASE_URL="mysql+pymysql://user:pass@host:3306/astrofatigue"

Nothing in db_models.py, seed.py, or any repository/query code needs to
change -- SQLAlchemy handles the dialect differences.
"""

import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///astrofatigue.db")

# check_same_thread=False is a SQLite-only requirement (FastAPI can use a
# request from a different thread than the one that opened the connection).
# It's a no-op / ignored for other dialects.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    """Create all tables if they don't exist yet. Safe to call repeatedly."""
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session():
    """Use as: `with get_session() as db: ...`"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db():
    """
    FastAPI dependency-injection version: `db: Session = Depends(get_db)`.
    Commits on successful request completion, rolls back on error --
    without this, writes made during a request (e.g. caching an
    explanation) are silently lost when the session closes.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()