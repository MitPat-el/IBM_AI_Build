"""
pytest configuration, shared fixtures.

Two things happen here that matter:

1. `backend/` (the parent of this tests/ folder) is added to sys.path so
   tests can `from drift import ...` etc, the same way main.py does (the
   app's modules use absolute imports assuming backend/ is the root, not
   a package).

2. DATABASE_URL is pointed at an isolated temp SQLite file BEFORE any
   app module is imported. db/session.py builds its engine at import
   time from that env var -- if this weren't set first, tests would
   silently read/write your real astrofatigue.db and corrupt dev data.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

TEST_DB_PATH = Path(tempfile.gettempdir()) / "test_astrofatigue.db"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"


@pytest.fixture(autouse=True)
def clean_db():
    """Wipe and reseed the test DB before every test, so tests never
    depend on execution order or leak state into each other."""
    from db.seed import seed

    seed(num_days=6, wipe_existing=True)
    yield


@pytest.fixture
def client():
    """FastAPI TestClient, used as a context manager so the app's
    startup event actually fires (auto-seed check, etc)."""
    from fastapi.testclient import TestClient
    from main import app

    with TestClient(app) as c:
        yield c