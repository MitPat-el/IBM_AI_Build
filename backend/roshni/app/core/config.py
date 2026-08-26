"""
core/config.py — Single source of truth for all model weights and thresholds.

IMPORTANT: All values here are PROTOTYPE ASSUMPTIONS.
When the research/data team provides validated values, update ONLY this file.

NASA research grounds the variables and simulated ranges used in this prototype.
The current score weights, thresholds, mission-risk formulas, and demonstration
records are prototype assumptions and are NOT NASA-validated operational models.
"""

# ===========================================================================
# FATIGUE DRIFT SCORE
# ===========================================================================

# Signal weights — must sum to 1.0
SIGNAL_WEIGHTS: dict[str, float] = {
    "pvt_risk":       0.30,   # PVT / reaction-time risk
    "sleep_risk":     0.30,   # Sleep quality/duration risk
    "circadian_risk": 0.20,   # Circadian-alignment risk
    "workload_risk":  0.20,   # Task workload risk
}

# Risk-level thresholds: (min_inclusive, max_inclusive, label)
# Evaluated top-to-bottom; first match wins.
RISK_LEVELS: list[tuple[float, float, str]] = [
    (85.0, 100.0, "CRITICAL"),
    (70.0,  84.9, "HIGH"),
    (40.0,  69.9, "MODERATE"),
    (0.0,   39.9, "LOW"),
]

# Input validation bounds
MIN_RISK_VALUE: float = 0.0
MAX_RISK_VALUE: float = 100.0

# Trend classification thresholds (delta between current and previous score)
TREND_THRESHOLDS: dict[str, float] = {
    "RAPIDLY_RISING": 15.0,   # score increased by >= 15 points
    "RISING":          5.0,   # score increased by >= 5 points
    "FALLING":        -5.0,   # score decreased by >= 5 points
    # else → STABLE
}

# ===========================================================================
# MISSION RISK ENGINE
# ===========================================================================

# Task demand sub-weights — must sum to 1.0
TASK_DEMAND_WEIGHTS: dict[str, float] = {
    "criticality":   0.40,
    "cognitive":     0.30,
    "physical":      0.20,
    "dependency":    0.10,
}

# Mission-task risk blend: fatigue vs task demand
MISSION_TASK_RISK_WEIGHTS: dict[str, float] = {
    "fatigue":      0.60,
    "task_demand":  0.40,
}

# Mission-level aggregation: highest-risk task vs crew average
MISSION_LEVEL_WEIGHTS: dict[str, float] = {
    "highest_task":  0.70,
    "crew_average":  0.30,
}

# ===========================================================================
# DATA QUALITY
# ===========================================================================

# Maximum age (seconds) of a reading before freshness penalty kicks in
DATA_FRESHNESS_THRESHOLD_SECONDS: int = 3600   # 1 hour

# ===========================================================================
# SANITY CHECKS — fail fast at import time if weights are misconfigured
# ===========================================================================

def _assert_weights(name: str, weights: dict[str, float]) -> None:
    total = round(sum(weights.values()), 6)
    assert total == 1.0, (
        f"{name} must sum to 1.0 — currently {total}. Fix core/config.py."
    )

_assert_weights("SIGNAL_WEIGHTS", SIGNAL_WEIGHTS)
_assert_weights("TASK_DEMAND_WEIGHTS", TASK_DEMAND_WEIGHTS)
_assert_weights("MISSION_TASK_RISK_WEIGHTS", MISSION_TASK_RISK_WEIGHTS)
_assert_weights("MISSION_LEVEL_WEIGHTS", MISSION_LEVEL_WEIGHTS)
