"""
Synthetic mission data generator.

Generates a 6-day, per-astronaut time series of the four raw signals
that drift.py consumes. This is NOT random noise — it follows the
shape of real fatigue accumulation curves from NASA/military PVT and
sleep-restriction studies:

  - PVT lapses grow roughly linearly with consecutive nights of
    partial sleep restriction (Van Dongen et al. 2003 dose-response
    curve is the classic reference: chronic sleep restriction at
    <6h/night causes cumulative, dose-dependent PVT lapse increases
    that don't fully resolve without recovery sleep).
  - Sleep debt compounds night over night unless a recovery night
    (>= recommended hours) occurs.
  - Circadian phase shift increases with each non-24h light/dark
    cycle exposure (ISS orbital period ~90 min = 16 cycles/day) and
    partially re-entrains during rest periods.
  - Workload follows a mission schedule you define (e.g. EVA days
    spike it).

Every astronaut gets an independent random seed per run, but the same
seed reproduces the same mission for demo purposes.
"""

import hashlib
import random
from dataclasses import dataclass, field
from typing import Optional

from drift import AstronautSignals, DriftWeights, compute_drift_score, DriftResult


@dataclass
class AstronautProfile:
    astronaut_id: str
    name: str
    baseline_pvt_lapses: float = 3.0      # healthy, well-rested baseline
    resilience: float = 1.0               # 0.8-1.2, individual variability multiplier
    seed: Optional[int] = None


@dataclass
class MissionDayRecord:
    day: int
    astronaut_id: str
    hours_slept: float
    pvt_lapses: int
    minutes_phase_shift: float
    task_load: float
    rolling_avg_task_load: float
    drift: DriftResult


# Mission schedule: task load index per day (relative scale, tune as needed).
# Spike on days 3-4 to simulate an EVA / high-workload event.
DEFAULT_TASK_SCHEDULE = [8, 9, 14, 15, 9, 8]  # 6-day mission


def resolve_seed(astronaut_id: str) -> int:
    """
    Deterministic seed derived from astronaut_id, stable across process
    restarts (unlike Python's built-in hash(), which is randomized per
    process by default via PYTHONHASHSEED). Used as the fallback when an
    AstronautProfile doesn't specify an explicit seed, and persisted to
    the DB so re-simulation (What-If Simulator) always reproduces the
    exact same raw signals as what's already stored.
    """
    digest = hashlib.md5(astronaut_id.encode()).digest()
    return int.from_bytes(digest[:4], byteorder="big")


def _rng_for(profile: AstronautProfile, day: int) -> random.Random:
    seed = (profile.seed if profile.seed is not None else resolve_seed(profile.astronaut_id)) + day
    return random.Random(seed)


def simulate_mission(
    profile: AstronautProfile,
    num_days: int = 6,
    task_schedule: Optional[list[float]] = None,
    weights: DriftWeights = DriftWeights(),
) -> list[MissionDayRecord]:
    task_schedule = task_schedule or DEFAULT_TASK_SCHEDULE[:num_days]

    records: list[MissionDayRecord] = []
    prior_sleep_debt = 0.0
    phase_shift = 0.0
    rolling_loads: list[float] = []

    for day in range(1, num_days + 1):
        rng = _rng_for(profile, day)

        # --- Sleep: gradually degrades mid-mission, dips further on high-workload days ---
        workload_today = task_schedule[min(day - 1, len(task_schedule) - 1)]
        base_sleep = 7.5 - 0.15 * day  # mild fatigue-driven decline over mission
        workload_penalty = max(0.0, (workload_today - 8) * 0.08)
        hours_slept = max(3.0, rng.gauss(base_sleep - workload_penalty, 0.6))

        # --- PVT lapses: baseline + cumulative sleep-debt-driven increase (dose-response) ---
        debt_effect = prior_sleep_debt * 0.9 * profile.resilience
        lapses = max(0, round(rng.gauss(profile.baseline_pvt_lapses + debt_effect, 1.2)))

        # --- Circadian phase shift: accumulates with 16 light/dark cycles/day,
        #     partially re-entrains if sleep that night met/exceeded recommended ---
        drift_increment = rng.uniform(35, 70)  # minutes of drift added per day in microgravity
        reentrainment = 40 if hours_slept >= 7.5 else 0
        phase_shift = max(0.0, phase_shift + drift_increment - reentrainment)

        rolling_loads.append(workload_today)
        rolling_avg = sum(rolling_loads) / len(rolling_loads)

        signals = AstronautSignals(
            pvt_lapses=lapses,
            pvt_baseline_lapses=profile.baseline_pvt_lapses,
            hours_slept_last_24=round(hours_slept, 2),
            minutes_phase_shift=round(phase_shift, 1),
            current_task_load=workload_today,
            rolling_avg_task_load=round(rolling_avg, 2),
            prior_sleep_debt=prior_sleep_debt,
        )
        result = compute_drift_score(signals, weights)
        prior_sleep_debt = result.updated_sleep_debt_hours

        records.append(MissionDayRecord(
            day=day,
            astronaut_id=profile.astronaut_id,
            hours_slept=signals.hours_slept_last_24,
            pvt_lapses=lapses,
            minutes_phase_shift=signals.minutes_phase_shift,
            task_load=workload_today,
            rolling_avg_task_load=signals.rolling_avg_task_load,
            drift=result,
        ))

    return records


def simulate_crew(
    profiles: list[AstronautProfile],
    num_days: int = 6,
    weights: DriftWeights = DriftWeights(),
) -> dict[str, list[MissionDayRecord]]:
    return {p.astronaut_id: simulate_mission(p, num_days=num_days, weights=weights) for p in profiles}


if __name__ == "__main__":
    crew = [
        AstronautProfile(astronaut_id="A1", name="Chen", baseline_pvt_lapses=2.5, seed=1),
        AstronautProfile(astronaut_id="A2", name="Okafor", baseline_pvt_lapses=3.5, resilience=1.15, seed=2),
    ]
    results = simulate_crew(crew)
    for aid, days in results.items():
        print(f"\n--- {aid} ---")
        for r in days:
            print(f"Day {r.day}: sleep={r.hours_slept}h lapses={r.pvt_lapses} "
                  f"phase_shift={r.minutes_phase_shift}min drift={r.drift.drift_score} "
                  f"risk={r.drift.risk_level}")