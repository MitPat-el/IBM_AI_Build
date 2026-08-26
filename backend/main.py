"""
FastAPI backend for the astronaut fatigue drift-detection system.

Architecture:
  drift.py             -> deterministic math, no AI
  simulator.py          -> synthetic mission data (stand-in until real sensor
                            feeds exist -- same generator used by seed.py)
  db/models.py           \
  db/session.py            > SQL persistence layer (SQLite by default, swap
  db/repository.py        /  DATABASE_URL for Postgres/MySQL later, see README)
  db/seed.py             -> populates the DB with fake mission data + task DAG
  projection.py         -> Mission Risk Map + What-If Simulator (no AI)
  replay.py             -> Historical Replay + trend-based forward projection (no AI)
  dependency_graph.py   -> task dependency graph + cascading impact analysis (no AI)
  bob.py                -> IBM Bob / watsonx, ONLY called when drift_score crosses
                            EXPLANATION_TRIGGER_THRESHOLD. Results are cached in
                            the `explanations` table so the same alert is never
                            re-explained twice.

Run: uvicorn main:app --reload --port 8000
On first run (empty DB) the app auto-seeds a default 3-astronaut, 6-day
fake mission so there's never a blank/broken demo state.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from drift import DriftWeights, EXPLANATION_TRIGGER_THRESHOLD
from db.session import init_db, get_db, get_session
from db.models import Astronaut, Task, MissionDay
from db.repository import load_crew, load_profile, load_mission_records, get_cached_explanation, save_explanation, load_task_graph
from bob import explain_drift
from projection import project_mission_risk, mission_risk_summary, simulate_intervention
from replay import get_replay, project_forward
from dependency_graph import compute_impact, at_risk_tasks
from db import seed as seed_module

# Runtime state: the formula weights currently applied to the seeded data.
# Not persisted -- purely reflects what /mission/reset was last called with,
# for display purposes (e.g. showing "current weights" in a UI).
CURRENT_WEIGHTS = DriftWeights()

# Set FORCE_RESEED=1 to always wipe and reseed everything on startup --
# a convenience for active development, not needed for normal use since
# the self-healing check below already backfills missing tables.
FORCE_RESEED = os.environ.get("FORCE_RESEED", "").lower() in ("1", "true", "yes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    with get_session() as db:
        has_astronauts = db.query(Astronaut).first() is not None
        has_tasks = db.query(Task).first() is not None
        existing_max_day = db.query(func.max(MissionDay.day)).scalar()

    if FORCE_RESEED or not has_astronauts:
        # Fresh database, or an explicit forced reset -- seed everything.
        seed_module.seed(num_days=6, wipe_existing=True, weights=CURRENT_WEIGHTS)
    elif not has_tasks:
        # Self-heal: astronaut/mission data already exists (e.g. from
        # before the Dependency Graph feature was added) but the tasks
        # table is empty. Backfill just the task graph, matching the
        # existing mission length, without touching anything else --
        # no need to ever manually delete the database for this.
        seed_module._seed_tasks(seed_module.DEFAULT_CREW, existing_max_day or 6, wipe_existing=True)
    yield


app = FastAPI(title="Astronaut Fatigue Drift API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before demo-day if needed
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class WeightsIn(BaseModel):
    reaction_time: float = 0.40
    sleep_debt: float = 0.30
    circadian: float = 0.15
    workload: float = 0.15


class TaskReassignRequest(BaseModel):
    astronaut_id: str
    day: int
    reassign_to: Optional[str] = None  # None = just delay the task
    task_load_delta: float = -6.0      # how much load to remove from the source astronaut


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "astronaut-fatigue-drift-api"}


@app.get("/crew")
def get_crew(db: Session = Depends(get_db)):
    return [{"astronaut_id": a.astronaut_id, "name": a.name} for a in load_crew(db)]


@app.get("/mission")
def get_full_mission(db: Session = Depends(get_db)):
    """Whole-crew Mission Risk Map, day by day (worst-case astronaut drives each day)."""
    records = load_mission_records(db)
    snapshots = project_mission_risk(records)
    return [
        {
            "day": s.day,
            "overall_risk_level": s.overall_risk_level,
            "overall_drift_score": s.overall_drift_score,
            "highest_risk_astronaut": s.highest_risk_astronaut,
            "per_astronaut": s.per_astronaut,
        }
        for s in snapshots
    ]


@app.get("/mission/summary")
def get_mission_summary(db: Session = Depends(get_db)):
    """One-line mission-level verdict, e.g. for a dashboard header.
    NOTE: must stay declared before /mission/{astronaut_id} -- FastAPI
    matches routes in declaration order, and a dynamic path segment would
    otherwise swallow the literal 'summary' as an astronaut_id."""
    return mission_risk_summary(load_mission_records(db))


@app.get("/mission/{astronaut_id}")
def get_mission(astronaut_id: str, db: Session = Depends(get_db)):
    """Full drift history for one astronaut -- feeds Historical Replay."""
    records = load_mission_records(db, astronaut_id).get(astronaut_id)
    if not records:
        raise HTTPException(404, "Unknown astronaut_id")
    return [
        {
            "day": r.day,
            "hours_slept": r.hours_slept,
            "pvt_lapses": r.pvt_lapses,
            "minutes_phase_shift": r.minutes_phase_shift,
            "task_load": r.task_load,
            "drift_score": r.drift.drift_score,
            "risk_level": r.drift.risk_level,
            "sub_scores": {
                "reaction_time": r.drift.reaction_time_score,
                "sleep_debt": r.drift.sleep_debt_score,
                "circadian": r.drift.circadian_score,
                "workload": r.drift.workload_score,
            },
        }
        for r in records
    ]


@app.post("/mission/reset")
def reset_mission(
    weights: Optional[WeightsIn] = Body(None, embed=True),
    num_days: int = Body(6, embed=True),
):
    """
    Regenerate the fake mission data in the database, optionally with
    custom formula weights. This is your What-If Simulator hook for
    re-weighting the formula itself (as opposed to /whatif/reassign,
    which changes task load for one astronaut). Wipes and reseeds --
    also clears cached explanations, since they'd reference stale scores.
    """
    global CURRENT_WEIGHTS
    CURRENT_WEIGHTS = DriftWeights(**weights.model_dump()) if weights else DriftWeights()
    seed_module.seed(num_days=num_days, wipe_existing=True, weights=CURRENT_WEIGHTS)
    return {"status": "regenerated", "num_days": num_days, "weights": CURRENT_WEIGHTS.__dict__}


@app.get("/explain/{astronaut_id}/{day}")
def explain(astronaut_id: str, day: int, db: Session = Depends(get_db)):
    """
    Only fires the Bob/watsonx call if that day's drift score crossed
    threshold, and only if it hasn't already been explained (cached in
    the `explanations` table) -- keeps cost/latency down and matches
    "AI only explains, doesn't decide."
    """
    records = load_mission_records(db, astronaut_id).get(astronaut_id)
    if not records:
        raise HTTPException(404, "Unknown astronaut_id")
    record = next((r for r in records if r.day == day), None)
    if not record:
        raise HTTPException(404, "Unknown day")

    if record.drift.drift_score < EXPLANATION_TRIGGER_THRESHOLD:
        return {
            "triggered": False,
            "drift_score": record.drift.drift_score,
            "risk_level": record.drift.risk_level,
            "message": "Below alert threshold -- no explanation generated.",
        }

    cached = get_cached_explanation(db, astronaut_id, day)
    if cached:
        explanation_data = {
            "astronaut_message": cached.astronaut_message,
            "flight_surgeon_brief": cached.flight_surgeon_brief,
            "suggested_intervention": cached.suggested_intervention,
            "source": cached.source,
        }
    else:
        profile = load_profile(db, astronaut_id)
        explanation = explain_drift(profile.name, day, record.drift)
        save_explanation(db, astronaut_id, day, explanation)
        explanation_data = {
            "astronaut_message": explanation.astronaut_message,
            "flight_surgeon_brief": explanation.flight_surgeon_brief,
            "suggested_intervention": explanation.suggested_intervention,
            "source": explanation.source,
        }

    return {
        "triggered": True,
        "drift_score": record.drift.drift_score,
        "risk_level": record.drift.risk_level,
        **explanation_data,
    }


@app.post("/whatif/reassign")
def whatif_reassign(req: TaskReassignRequest, db: Session = Depends(get_db)):
    """
    What-If Simulator: re-run one astronaut's mission with a task load
    reduction (simulating reassignment/delay) and show the new drift
    trajectory from that day forward, so you can show before/after.
    """
    profile = load_profile(db, req.astronaut_id)
    if profile is None:
        raise HTTPException(404, "Unknown astronaut_id")

    original_records = load_mission_records(db, req.astronaut_id).get(req.astronaut_id, [])

    result = simulate_intervention(
        profile,
        original_records,
        day=req.day,
        task_load_delta=req.task_load_delta,
        reassign_to=req.reassign_to,
        weights=CURRENT_WEIGHTS,
    )
    return {
        "astronaut_id": result.astronaut_id,
        "day_modified": result.day_modified,
        "reassigned_to": result.reassigned_to,
        "comparison": result.comparison,
    }


@app.get("/replay/{astronaut_id}")
def replay(astronaut_id: str, project_days: int = 0, db: Session = Depends(get_db)):
    """
    Historical Replay timeline for one astronaut. Pass project_days > 0
    to also include a simple trend-based forward projection (timeline
    slider "possible prediction" feature) -- not another AI call, just
    linear extrapolation over the last few recorded days.
    """
    records = load_mission_records(db, astronaut_id).get(astronaut_id)
    if not records:
        raise HTTPException(404, "Unknown astronaut_id")

    timeline = get_replay(astronaut_id, records)
    if project_days > 0:
        timeline = project_forward(timeline, num_days=project_days)

    return {
        "astronaut_id": timeline.astronaut_id,
        "points": [p.__dict__ for p in timeline.points],
        "projected_points": [p.__dict__ for p in timeline.projected_points],
    }


@app.get("/tasks/graph")
def tasks_graph(db: Session = Depends(get_db)):
    """
    The full task dependency graph -- nodes (each task, annotated with
    its assigned astronaut's current risk_level on that task's day) and
    edges (task -> depends_on). Feeds the Dependency Graph visualization.
    """
    graph = load_task_graph(db)
    return {
        "nodes": [n.__dict__ for n in graph.nodes],
        "edges": [e.__dict__ for e in graph.edges],
    }


@app.get("/tasks/at-risk")
def tasks_at_risk(db: Session = Depends(get_db)):
    """
    Tasks whose assigned astronaut is currently high/critical risk,
    ranked by how much downstream work would slip if that task slips --
    the "identify the task and explain why" piece of the What-If /
    Dependency Graph feature.
    """
    graph = load_task_graph(db)
    return at_risk_tasks(graph)


@app.get("/tasks/{task_id}/impact")
def task_impact(task_id: str, db: Session = Depends(get_db)):
    """
    If this specific task slips, what else slips with it? Returns the
    task itself plus every downstream task reachable through the
    dependency graph, each with its own astronaut/day/risk context.
    """
    graph = load_task_graph(db)
    impact = compute_impact(task_id, graph)
    if impact is None:
        raise HTTPException(404, "Unknown task_id")

    return {
        "task": impact.task.__dict__,
        "downstream": [d.__dict__ for d in impact.downstream],
        "downstream_count": len(impact.downstream),
    }