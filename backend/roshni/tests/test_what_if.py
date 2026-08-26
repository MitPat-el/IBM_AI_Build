"""
tests/test_what_if.py — Tests for the What-If intervention simulator.

Key invariants verified:
  - Real mission state is never modified
  - Feasibility is determined deterministically (not by AI)
  - Qualification check blocks reassignment
  - CRITICAL fatigue on target blocks reassignment
  - DELAY_TASK respects can_delay flag and dependency conflicts
  - REDUCE_WORKLOAD lowers fatigue and mission risk
"""
import pytest
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from app.main import app
from app.models.astronaut import Astronaut, FatigueReading
from app.models.task import Task
from app.models.mission import Mission
from app.models.intervention import Intervention
from app.services.what_if_service import simulate

client = TestClient(app)

TS      = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
T_START = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)


def _astronaut(aid, quals=None):
    return Astronaut(astronaut_id=aid, name=f"Astronaut {aid}",
                     role="Crew", qualifications=quals or [])

def _reading(aid, pvt=50.0, sleep=50.0, circ=40.0, wl=40.0):
    return FatigueReading(astronaut_id=aid, timestamp=TS,
                          pvt_risk=pvt, sleep_risk=sleep,
                          circadian_risk=circ, workload_risk=wl)

def _task(tid, aid, criticality=4, cognitive=4, physical=3,
          quals=None, can_delay=True, can_reassign=True, deps=None):
    return Task(task_id=tid, name=f"Task {tid}",
                assigned_astronaut_id=aid,
                start_time=T_START, duration_minutes=60,
                criticality=criticality, cognitive_demand=cognitive,
                physical_demand=physical,
                required_qualifications=quals or [],
                can_delay=can_delay, can_reassign=can_reassign,
                dependencies=deps or [])

def _mission(astronauts, tasks):
    return Mission(mission_id="SIM-01", mission_name="Sim Mission",
                   mission_day=4, total_days=14,
                   astronauts=astronauts, tasks=tasks)


# ---------------------------------------------------------------------------
# REASSIGN_TASK
# ---------------------------------------------------------------------------
class TestReassignTask:

    def _base(self):
        a1 = _astronaut("A01", quals=["EVA"])
        a2 = _astronaut("A02", quals=["EVA", "ROBOTICS"])
        a3 = _astronaut("A03", quals=["MEDICAL"])       # no EVA
        task = _task("T-EVA", "A01", quals=["EVA"])
        mission = _mission([a1, a2, a3], [task])
        r1 = _reading("A01", pvt=80, sleep=75)          # HIGH fatigue
        r2 = _reading("A02", pvt=30, sleep=25)          # LOW fatigue
        r3 = _reading("A03", pvt=30, sleep=25)
        return mission, [r1, r2, r3]

    def test_valid_reassign_is_feasible(self):
        mission, readings = self._base()
        iv = Intervention(intervention_type="REASSIGN_TASK",
                          task_id="T-EVA", from_astronaut_id="A01", to_astronaut_id="A02")
        result = simulate(iv, mission, readings)
        assert result.feasible is True
        assert result.constraint_violations == []

    def test_valid_reassign_reduces_risk(self):
        mission, readings = self._base()
        iv = Intervention(intervention_type="REASSIGN_TASK",
                          task_id="T-EVA", from_astronaut_id="A01", to_astronaut_id="A02")
        result = simulate(iv, mission, readings)
        assert result.after_mission_risk <= result.before_mission_risk

    def test_missing_qualification_is_infeasible(self):
        """A03 lacks EVA — reassignment must be blocked."""
        mission, readings = self._base()
        iv = Intervention(intervention_type="REASSIGN_TASK",
                          task_id="T-EVA", from_astronaut_id="A01", to_astronaut_id="A03")
        result = simulate(iv, mission, readings)
        assert result.feasible is False
        assert any("EVA" in v for v in result.constraint_violations)

    def test_infeasible_does_not_change_risk(self):
        mission, readings = self._base()
        iv = Intervention(intervention_type="REASSIGN_TASK",
                          task_id="T-EVA", from_astronaut_id="A01", to_astronaut_id="A03")
        result = simulate(iv, mission, readings)
        assert result.after_mission_risk == result.before_mission_risk

    def test_non_reassignable_task_is_infeasible(self):
        a1 = _astronaut("A01", quals=["EVA"])
        a2 = _astronaut("A02", quals=["EVA"])
        task = _task("T-LOCKED", "A01", quals=["EVA"], can_reassign=False)
        mission = _mission([a1, a2], [task])
        iv = Intervention(intervention_type="REASSIGN_TASK",
                          task_id="T-LOCKED", from_astronaut_id="A01", to_astronaut_id="A02")
        result = simulate(iv, mission, [_reading("A01"), _reading("A02")])
        assert result.feasible is False

    def test_unknown_astronaut_target_is_infeasible(self):
        mission, readings = self._base()
        iv = Intervention(intervention_type="REASSIGN_TASK",
                          task_id="T-EVA", from_astronaut_id="A01", to_astronaut_id="A99")
        result = simulate(iv, mission, readings)
        assert result.feasible is False

    def test_critical_fatigue_target_blocked(self):
        """Target astronaut with CRITICAL fatigue should be blocked."""
        a1 = _astronaut("A01", quals=["EVA"])
        a2 = _astronaut("A02", quals=["EVA"])
        task = _task("T-EVA", "A01", quals=["EVA"])
        mission = _mission([a1, a2], [task])
        r1 = _reading("A01", pvt=80, sleep=75)
        r2 = _reading("A02", pvt=95, sleep=95, circ=90, wl=90)  # CRITICAL
        result = simulate(
            Intervention(intervention_type="REASSIGN_TASK",
                         task_id="T-EVA", from_astronaut_id="A01", to_astronaut_id="A02"),
            mission, [r1, r2],
        )
        assert result.feasible is False

    def test_real_mission_state_unchanged_after_simulate(self):
        mission, readings = self._base()
        original_assigned = mission.tasks[0].assigned_astronaut_id
        original_readings = deepcopy(readings)
        iv = Intervention(intervention_type="REASSIGN_TASK",
                          task_id="T-EVA", from_astronaut_id="A01", to_astronaut_id="A02")
        simulate(iv, mission, readings)
        assert mission.tasks[0].assigned_astronaut_id == original_assigned
        for orig, after in zip(original_readings, readings):
            assert orig.workload_risk == after.workload_risk


# ---------------------------------------------------------------------------
# DELAY_TASK
# ---------------------------------------------------------------------------
class TestDelayTask:

    def test_valid_delay_is_feasible(self):
        a1 = _astronaut("A01")
        task = _task("T01", "A01", can_delay=True)
        mission = _mission([a1], [task])
        iv = Intervention(intervention_type="DELAY_TASK", task_id="T01", delay_minutes=120)
        result = simulate(iv, mission, [_reading("A01", pvt=70, sleep=70)])
        assert result.feasible is True

    def test_non_delayable_task_infeasible(self):
        a1 = _astronaut("A01")
        task = _task("T01", "A01", can_delay=False)
        mission = _mission([a1], [task])
        iv = Intervention(intervention_type="DELAY_TASK", task_id="T01", delay_minutes=60)
        result = simulate(iv, mission, [_reading("A01")])
        assert result.feasible is False
        assert any("non-delayable" in v.lower() or "non-delay" in v.lower()
                   for v in result.constraint_violations)

    def test_delay_with_dependency_conflict_is_infeasible(self):
        a1 = _astronaut("A01")
        # T01 finishes at T_START + 60min; T02 depends on T01 and starts at T_START + 30min
        t1 = _task("T01", "A01", can_delay=True)
        t2_start = T_START + timedelta(minutes=30)
        t2 = Task(task_id="T02", name="Task T02", assigned_astronaut_id="A01",
                  start_time=t2_start, duration_minutes=60,
                  criticality=3, cognitive_demand=3, physical_demand=2,
                  dependencies=["T01"], can_delay=True, can_reassign=True)
        mission = _mission([a1], [t1, t2])
        # Delay T01 by 60 mins → T01 now ends at T_START + 120min, AFTER T02 starts
        iv = Intervention(intervention_type="DELAY_TASK", task_id="T01", delay_minutes=60)
        result = simulate(iv, mission, [_reading("A01")])
        assert result.feasible is False
        assert any("T02" in v for v in result.constraint_violations)

    def test_real_state_unchanged_after_delay(self):
        a1 = _astronaut("A01")
        task = _task("T01", "A01", can_delay=True)
        mission = _mission([a1], [task])
        original_start = mission.tasks[0].start_time
        iv = Intervention(intervention_type="DELAY_TASK", task_id="T01", delay_minutes=120)
        simulate(iv, mission, [_reading("A01")])
        assert mission.tasks[0].start_time == original_start


# ---------------------------------------------------------------------------
# REDUCE_WORKLOAD
# ---------------------------------------------------------------------------
class TestReduceWorkload:

    def test_reduce_workload_is_feasible(self):
        a1 = _astronaut("A01")
        task = _task("T01", "A01")
        mission = _mission([a1], [task])
        iv = Intervention(intervention_type="REDUCE_WORKLOAD", task_id="T01", workload_reduction=20)
        result = simulate(iv, mission, [_reading("A01", wl=60)])
        assert result.feasible is True

    def test_reduce_workload_lowers_fatigue_score(self):
        a1 = _astronaut("A01")
        task = _task("T01", "A01")
        mission = _mission([a1], [task])
        iv = Intervention(intervention_type="REDUCE_WORKLOAD", task_id="T01", workload_reduction=30)
        result = simulate(iv, mission, [_reading("A01", wl=80)])
        assert result.after_fatigue_score <= result.before_fatigue_score

    def test_reduce_workload_does_not_go_below_zero(self):
        a1 = _astronaut("A01")
        task = _task("T01", "A01")
        mission = _mission([a1], [task])
        # Reduce by the max allowed (100) on a reading with workload_risk=10
        # Result workload_risk should be clamped to 0, not negative
        iv = Intervention(intervention_type="REDUCE_WORKLOAD", task_id="T01", workload_reduction=100)
        result = simulate(iv, mission, [_reading("A01", wl=10)])
        assert result.feasible is True
        assert result.after_fatigue_score >= 0

    def test_real_state_unchanged_after_reduce(self):
        a1 = _astronaut("A01")
        task = _task("T01", "A01")
        mission = _mission([a1], [task])
        readings = [_reading("A01", wl=70)]
        original_wl = readings[0].workload_risk
        iv = Intervention(intervention_type="REDUCE_WORKLOAD", task_id="T01", workload_reduction=20)
        simulate(iv, mission, readings)
        assert readings[0].workload_risk == original_wl


# ---------------------------------------------------------------------------
# WhatIfResult fields
# ---------------------------------------------------------------------------
class TestWhatIfResult:

    def test_risk_change_is_after_minus_before(self):
        a1 = _astronaut("A01", quals=["EVA"])
        a2 = _astronaut("A02", quals=["EVA"])
        task = _task("T-EVA", "A01", quals=["EVA"])
        mission = _mission([a1, a2], [task])
        iv = Intervention(intervention_type="REASSIGN_TASK",
                          task_id="T-EVA", from_astronaut_id="A01", to_astronaut_id="A02")
        result = simulate(iv, mission,
                          [_reading("A01", pvt=80, sleep=75),
                           _reading("A02", pvt=25, sleep=25)])
        assert abs(result.risk_change - (result.after_mission_risk - result.before_mission_risk)) < 0.01

    def test_explanation_data_populated(self):
        a1 = _astronaut("A01", quals=["EVA"])
        task = _task("T-EVA", "A01", quals=["EVA"])
        mission = _mission([a1], [task])
        iv = Intervention(intervention_type="DELAY_TASK", task_id="T-EVA", delay_minutes=60)
        result = simulate(iv, mission, [_reading("A01")])
        assert "intervention_type" in result.explanation_data
        assert "before_mission_risk" in result.explanation_data

    def test_disclaimer_present(self):
        a1 = _astronaut("A01")
        task = _task("T01", "A01")
        mission = _mission([a1], [task])
        iv = Intervention(intervention_type="REDUCE_WORKLOAD", task_id="T01", workload_reduction=10)
        result = simulate(iv, mission, [_reading("A01")])
        assert len(result.disclaimer) > 0


# ---------------------------------------------------------------------------
# API route
# ---------------------------------------------------------------------------
class TestWhatIfAPI:

    BASE = {
        "mission": {
            "mission_id": "API-SIM", "mission_name": "API Test",
            "mission_day": 4, "total_days": 14,
            "astronauts": [
                {"astronaut_id": "A01", "name": "A One", "role": "Crew",
                 "qualifications": ["EVA"]},
                {"astronaut_id": "A02", "name": "A Two", "role": "Crew",
                 "qualifications": ["EVA"]},
            ],
            "tasks": [{
                "task_id": "T-EVA", "name": "EVA Task",
                "assigned_astronaut_id": "A01",
                "start_time": "2025-01-15T14:00:00Z", "duration_minutes": 90,
                "criticality": 5, "cognitive_demand": 4, "physical_demand": 4,
                "required_qualifications": ["EVA"],
                "can_delay": True, "can_reassign": True, "dependencies": [],
            }],
        },
        "fatigue_readings": [
            {"astronaut_id": "A01", "timestamp": "2025-01-15T12:00:00Z",
             "pvt_risk": 80, "sleep_risk": 75, "circadian_risk": 65, "workload_risk": 70},
            {"astronaut_id": "A02", "timestamp": "2025-01-15T12:00:00Z",
             "pvt_risk": 25, "sleep_risk": 20, "circadian_risk": 15, "workload_risk": 20},
        ],
    }

    def test_valid_reassign_returns_200(self):
        payload = {**self.BASE, "intervention": {
            "intervention_type": "REASSIGN_TASK", "task_id": "T-EVA",
            "from_astronaut_id": "A01", "to_astronaut_id": "A02",
        }}
        r = client.post("/what-if/simulate", json=payload)
        assert r.status_code == 200

    def test_valid_reassign_feasible_true(self):
        payload = {**self.BASE, "intervention": {
            "intervention_type": "REASSIGN_TASK", "task_id": "T-EVA",
            "from_astronaut_id": "A01", "to_astronaut_id": "A02",
        }}
        r = client.post("/what-if/simulate", json=payload)
        assert r.json()["feasible"] is True

    def test_invalid_reassign_feasible_false(self):
        # A99 doesn't exist
        payload = {**self.BASE, "intervention": {
            "intervention_type": "REASSIGN_TASK", "task_id": "T-EVA",
            "from_astronaut_id": "A01", "to_astronaut_id": "A99",
        }}
        r = client.post("/what-if/simulate", json=payload)
        assert r.json()["feasible"] is False

    def test_response_has_required_fields(self):
        payload = {**self.BASE, "intervention": {
            "intervention_type": "REDUCE_WORKLOAD",
            "task_id": "T-EVA", "workload_reduction": 20,
        }}
        r = client.post("/what-if/simulate", json=payload)
        body = r.json()
        for field in ["before_mission_risk", "after_mission_risk",
                      "risk_change", "feasible", "constraint_violations", "disclaimer"]:
            assert field in body
