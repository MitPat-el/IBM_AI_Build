"""
app/config.py — Backwards-compatibility shim.

All configuration has moved to app/core/config.py.
This file re-exports everything so existing code (engine/scoring.py, models/)
continues to work without modification during the refactor.
"""
from app.core.config import (  # noqa: F401
    SIGNAL_WEIGHTS,
    RISK_LEVELS,
    MIN_RISK_VALUE,
    MAX_RISK_VALUE,
    TREND_THRESHOLDS,
    TASK_DEMAND_WEIGHTS,
    MISSION_TASK_RISK_WEIGHTS,
    MISSION_LEVEL_WEIGHTS,
    DATA_FRESHNESS_THRESHOLD_SECONDS,
)
