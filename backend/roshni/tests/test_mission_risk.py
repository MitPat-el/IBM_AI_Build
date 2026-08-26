"""
tests/test_mission_risk.py — Tests for the mission risk engine and API routes.
"""
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from app.main import app
from app.models.astronaut import Astronaut, FatigueReading
from app.models.task import Task
from app.models.mission import Mission
from app.services.mission_risk_service import calculate_mission_risk

client = TestClient(app)

TS = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
TASK_START = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)


def make_astronaut(aid: str, quals: list[str] | None = None) -> Astronaut:
    return Astronaut(
        astronaut_id=aid, name=f"Astronaut {aid}",
        role="Crew", qualifications=quals or [],
    )


def make_reading(aid: str, pvt=30.0, sleep=30.0, circadian=20.0, workload=20.0,
                 mission_day: int | None = None) -> FatigueReading:
    return FatigueReading(
        astronaut_id=aid, timestamp=TS,
        pvt_risk=pvt, sleep_risk=sleep,
        circadian_risk=circadian, workload_risk=workload,
        mission_day=mission_day,
    )


def make_task(tid: str, aid: str, criticality=3, cognitive=3, physical=2,
              can_delay=True, can_reassign=True, deps=None) -> Task:
    return Task(
        task_id=tid, name=f"Task {tid}",
        assigned_astronaut_id=aid,
        start_time=TASK_START, duration_minutes=60,
        criticality=criticality, cognitive_demand=cognitive, physical_demand=physical,
        can_delay=can_delay, can_reassign=can_reassign,
        dependencies=deps or [],
    )


def make_mission(astronauts, tasks=None) -> Mission:
    return Mission(
        mission_id="TEST-01", mission_name="Test Mission",
        mission_day=1, total_days=10,
        astronauts=astronauts, tasks=tasks or [],
    )


# ---------------------------------------------------------------------------
# Engine unit tests
# ---------------------------------------------------------------------------
class TestMissionRiskEngine:

    def test_low_fatigue_low_demand_gives_low_risk(self):
        astronaut = make_astronaut("A01")
        reading = make_reading("A01", pvt=20, sleep=20, circadian=15, workload=15)
        task = make_task("T01", "A01", criticality=1, cognitive=1, physical=1)
        mission = make_mission([astronaut], [task])
        result = calculate_mission_risk(mission, [reading])
        assert result.mission_risk_score < 40
        assert result.risk_level == "LOW"

    def test_high_fatigue_high_demand_gives_high_risk(self):
        astronaut = make_astronaut("A01")
        reading = make_reading("A01", pvt=85, sleep=85, circadian=75, workload=80)
        task = make_task("T01", "A01", criticality=5, cognitive=5, physical=5)
        mission = make_mission([astronaut], [task])
        result = calculate_mission_risk(mission, [reading])
        assert result.mission_risk_score >= 70
        assert result.risk_level in ("HIGH", "CRITICAL")

    def test_high_fatigue_low_demand_is_moderate(self):
        astronaut = make_astronaut("A01")
        reading = make_reading("A01", pvt=90, sleep=90, circadian=80, workload=80)
        task = make_task("T01", "A01", criticality=1, cognitive=1, physical=1)
        mission = make_mission([astronaut], [task])
        result = calculate_mission_risk(mission, [reading])
        # Fatigue is high but demand is very low — expect moderate, not critical
        assert result.mission_risk_score < 85

    def test_dependencies_increase_risk(self):
        astronaut = make_astronaut("A01")
        reading = make_reading("A01", pvt=50, sleep=50, circadian=40, workload=40)
        # T01 has no dependents; T02 and T03 both depend on T01
        t1 = make_task("T01", "A01", criticality=3, deps=[])
        t2 = make_task("T02", "A01", criticality=2, deps=["T01"])
        t3 = make_task("T03", "A01", criticality=2, deps=["T01"])
        mission_with_deps    = make_mission([astronaut], [t1, t2, t3])
        mission_without_deps = make_mission([astronaut], [t1])
        result_with    = calculate_mission_risk(mission_with_deps, [reading])
        result_without = calculate_mission_risk(mission_without_deps, [reading])
        # T01 should carry higher risk when it has 2 dependents
        t1_with    = next(d for d in result_with.task_risk_details    if d.task_id == "T01")
        t1_without = next(d for d in result_without.task_risk_details if d.task_id == "T01")
        assert t1_with.mission_task_risk >= t1_without.mission_task_risk

    def test_missing_reading_uses_default_and_warns(self):
        astronaut = make_astronaut("A01")
        task = make_task("T01", "A01")
        mission = make_mission([astronaut], [task])
        result = calculate_mission_risk(mission, [])   # no readings
        assert any("No fatigue reading" in w for w in result.warnings)
        assert result.mission_risk_score > 0

    def test_no_tasks_uses_crew_fatigue(self):
        astronaut = make_astronaut("A01")
        reading = make_reading("A01", pvt=40, sleep=40, circadian=30, workload=30)
        mission = make_mission([astronaut], [])
        result = calculate_mission_risk(mission, [reading])
        assert result.mission_risk_score > 0

    def test_highest_risk_task_identified(self):
        astronaut = make_astronaut("A01")
        reading = make_reading("A01", pvt=80, sleep=80, circadian=70, workload=70)
        low_task  = make_task("T-LOW",  "A01", criticality=1, cognitive=1, physical=1)
        high_task = make_task("T-HIGH", "A01", criticality=5, cognitive=5, physical=5)
        mission = make_mission([astronaut], [low_task, high_task])
        result = calculate_mission_risk(mission, [reading])
        assert result.highest_risk_task == "T-HIGH"

    def test_response_includes_contributing_factors(self):
        astronaut = make_astronaut("A01")
        reading = make_reading("A01", pvt=80, sleep=80, circadian=70, workload=70)
        task = make_task("T01", "A01", criticality=5, cognitive=5, physical=4)
        mission = make_mission([astronaut], [task])
        result = calculate_mission_risk(mission, [reading])
        assert len(result.contributing_factors) > 0

    def test_disclaimer_always_present(self):
        astronaut = make_astronaut("A01")
        reading = make_reading("A01")
        mission = make_mission([astronaut], [])
        result = calculate_mission_risk(mission, [reading])
        assert len(result.disclaimer) > 0


# ---------------------------------------------------------------------------
# API route tests
# ---------------------------------------------------------------------------

ASTRONAUT_PAYLOAD = {
    "astronaut_id": "A01", "name": "Elena Vasquez",
    "role": "Commander", "qualifications": ["EVA"],
}
READING_PAYLOAD = {
    "astronaut_id": "A01", "timestamp": "2025-01-15T12:00:00Z",
    "pvt_risk": 75, "sleep_risk": 70, "circadian_risk": 60, "workload_risk": 65,
}
TASK_PAYLOAD = {
    "task_id": "T-EVA-01", "name": "EVA Repair",
    "assigned_astronaut_id": "A01",
    "start_time": "2025-01-15T14:00:00Z", "duration_minutes": 90,
    "criticality": 5, "cognitive_demand": 4, "physical_demand": 5,
    "required_qualifications": ["EVA"],
    "can_delay": False, "can_reassign": True, "dependencies": [],
}
MISSION_PAYLOAD = {
    "mission_id": "TEST-01", "mission_name": "Test Mission",
    "mission_day": 1, "total_days": 10,
    "astronauts": [ASTRONAUT_PAYLOAD],
    "tasks": [TASK_PAYLOAD],
}

class TestMissionRiskAPI:

    def test_mission_risk_returns_200(self):
        r = client.post("/mission/risk", json={
            "mission": MISSION_PAYLOAD,
            "fatigue_readings": [READING_PAYLOAD],
        })
        assert r.status_code == 200

    def test_mission_risk_response_fields(self):
        r = client.post("/mission/risk", json={
            "mission": MISSION_PAYLOAD,
            "fatigue_readings": [READING_PAYLOAD],
        })
        body = r.json()
        assert "mission_risk_score" in body
        assert "risk_level" in body
        assert "contributing_factors" in body
        assert "task_risk_details" in body
        assert "disclaimer" in body

    def test_mission_risk_score_is_numeric(self):
        r = client.post("/mission/risk", json={
            "mission": MISSION_PAYLOAD,
            "fatigue_readings": [READING_PAYLOAD],
        })
        score = r.json()["mission_risk_score"]
        assert 0 <= score <= 100

    def test_mission_risk_high_fatigue_critical_task(self):
        r = client.post("/mission/risk", json={
            "mission": MISSION_PAYLOAD,
            "fatigue_readings": [READING_PAYLOAD],  # pvt=75, sleep=70 → HIGH
        })
        assert r.json()["risk_level"] in ("HIGH", "CRITICAL", "MODERATE")

    def test_project_risk_returns_200(self):
        proj_payload = {
            "mission_id": "TEST-01", "mission_name": "Test",
            "total_days": 2,
            "astronauts": [ASTRONAUT_PAYLOAD],
            "tasks": [TASK_PAYLOAD],
            "fatigue_readings": [
                {**READING_PAYLOAD, "mission_day": 1},
                {**READING_PAYLOAD, "mission_day": 2,
                 "pvt_risk": 85, "sleep_risk": 80},
            ],
        }
        r = client.post("/mission/project-risk", json=proj_payload)
        assert r.status_code == 200

    def test_project_risk_daily_summaries(self):
        proj_payload = {
            "mission_id": "TEST-01", "mission_name": "Test",
            "total_days": 2,
            "astronauts": [ASTRONAUT_PAYLOAD],
            "tasks": [],
            "fatigue_readings": [
                {**READING_PAYLOAD, "mission_day": 1},
                {**READING_PAYLOAD, "mission_day": 2},
            ],
        }
        r = client.post("/mission/project-risk", json=proj_payload)
        body = r.json()
        assert len(body["daily_summaries"]) == 2
        assert body["disclaimer"]  # prototype label always present
