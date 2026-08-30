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
  feasibility.py        -> reassignment feasibility checker (deterministic, advisory)
  bob.py                -> IBM Bob / watsonx.ai explanation layer:
                            (1) explain_drift()  -- called when drift_score crosses
                                EXPLANATION_TRIGGER_THRESHOLD; results cached in
                                the `explanations` table so the same alert is never
                                re-explained twice.
                            (2) generate_mission_brief()  -- called from POST
                                /mission-brief; synthesizes fatigue, mission risk,
                                What-If, feasibility, and dependency facts into a
                                structured MissionDecisionBrief; no caching (dynamic
                                per request context).

Run: uvicorn main:app --reload --port 8000
On first run (empty DB) the app auto-seeds a default 3-astronaut, 6-day
fake mission so there's never a blank/broken demo state.
"""

import dataclasses
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from drift import DriftWeights, EXPLANATION_TRIGGER_THRESHOLD
from db.session import init_db, get_db, get_session
from db.models import Astronaut, Task, MissionDay
from db.repository import load_crew, load_profile, load_mission_records, get_cached_explanation, save_explanation, load_task_graph
from bob import explain_drift, generate_mission_brief, BriefContext
from projection import project_mission_risk, mission_risk_summary, simulate_intervention
from replay import get_replay, project_forward
from dependency_graph import compute_impact, at_risk_tasks
from feasibility import check_reassignment_feasibility, ADVISORY_DISCLAIMER
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


class BriefRequest(BaseModel):
    """
    Request body for POST /mission-brief.

    Only astronaut_id + day are required.  All other fields opt-in to
    including richer context in the brief:
      - include_mission_summary: adds the mission-level worst-day snapshot
      - whatif_task_id / whatif_reassign_to / whatif_task_load_delta:
            re-runs the What-If Simulator and includes the trajectory diff
      - impact_task_id: includes the dependency impact for one specific task
    """
    astronaut_id: str
    day: int = Field(..., ge=1)

    # Mission summary context (default on)
    include_mission_summary: bool = True

    # Optional: What-If context to include in the brief
    whatif_task_id: Optional[str] = None
    whatif_reassign_to: Optional[str] = None
    whatif_task_load_delta: Optional[float] = Field(None, le=0.0)

    # Optional: dependency impact for one task
    impact_task_id: Optional[str] = None


class TaskReassignRequest(BaseModel):
    task_id: Optional[str] = None        # if given, astronaut_id/day/task_load_delta are derived from this task
    astronaut_id: Optional[str] = None   # required if task_id is not given
    day: Optional[int] = Field(None, ge=1)              # required if task_id is not given; must be a positive mission day
    reassign_to: Optional[str] = None    # None = just delay the task
    task_load_delta: Optional[float] = Field(None, le=0.0)  # manual mode only; must be negative if given; defaults to -6.0 if omitted


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

    This is where the Dependency Graph connects to the drift model: pass
    a real task_id (from /tasks/graph) instead of a manual astronaut_id/
    day/task_load_delta, and the astronaut, day, and load are derived
    from that actual task -- the response then includes not just the
    drift-score comparison but every downstream task that gets protected
    by the intervention, via the same graph used by /tasks/{id}/impact.

    When reassign_to is provided (task_id mode or manual mode), also runs
    a deterministic feasibility check (feasibility.py) to evaluate
    whether the receiving astronaut can safely absorb the load. The
    feasibility result is advisory only -- see ADVISORY_DISCLAIMER in
    feasibility.py. The trajectory comparison is always returned
    regardless of feasibility status.
    """
    if req.task_id:
        task_row = db.get(Task, req.task_id)
        if task_row is None:
            raise HTTPException(404, "Unknown task_id")
        astronaut_id = task_row.astronaut_id
        day = task_row.day
        task_load_delta = -task_row.load
    else:
        if not req.astronaut_id or req.day is None:
            raise HTTPException(400, "Provide either task_id, or astronaut_id and day")
        astronaut_id = req.astronaut_id
        day = req.day
        task_load_delta = req.task_load_delta if req.task_load_delta is not None else -6.0

    profile = load_profile(db, astronaut_id)
    if profile is None:
        raise HTTPException(404, "Unknown astronaut_id")

    original_records = load_mission_records(db, astronaut_id).get(astronaut_id, [])

    base_schedule = seed_module.task_derived_schedules([profile], len(original_records))[astronaut_id]

    # --- Feasibility check (only when a specific receiver is named) ---
    feasibility_result = None
    if req.reassign_to is not None:
        # Pre-guard 1: receiver must exist in the crew manifest.
        receiver_profile = load_profile(db, req.reassign_to)
        if receiver_profile is None:
            feasibility_result = {
                "status": "not_feasible",
                "receiver": req.reassign_to,
                "checks": {},
                "reasons": [f"Receiver '{req.reassign_to}' does not exist in the crew manifest."],
                "warnings": [],
                "advisory": ADVISORY_DISCLAIMER,
            }
        # Pre-guard 2: receiver must be different from source.
        elif req.reassign_to == astronaut_id:
            feasibility_result = {
                "status": "not_feasible",
                "receiver": req.reassign_to,
                "checks": {},
                "reasons": ["Source and receiver are the same astronaut."],
                "warnings": [],
                "advisory": ADVISORY_DISCLAIMER,
            }
        else:
            receiver_records = load_mission_records(db, req.reassign_to).get(req.reassign_to, [])
            graph = load_task_graph(db)
            fr = check_reassignment_feasibility(
                source_id=astronaut_id,
                receiver_id=req.reassign_to,
                day=day,
                task_load_delta=task_load_delta,
                receiver_records=receiver_records,
                graph=graph,
            )
            # Serialise dataclass fields to plain dicts for JSON response.
            feasibility_result = {
                "status": fr.status,
                "receiver": fr.receiver,
                "checks": {
                    k: dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v
                    for k, v in fr.checks.items()
                },
                "reasons": fr.reasons,
                "warnings": fr.warnings,
                "advisory": fr.advisory,
            }

    # Always run the trajectory simulation regardless of feasibility status.
    result = simulate_intervention(
        profile,
        original_records,
        day=day,
        task_load_delta=task_load_delta,
        reassign_to=req.reassign_to,
        weights=CURRENT_WEIGHTS,
        base_schedule=base_schedule,
    )

    dependency_impact = None
    if req.task_id:
        graph = load_task_graph(db)
        impact = compute_impact(req.task_id, graph)
        if impact is not None:
            dependency_impact = {
                "task": impact.task.__dict__,
                "downstream": [d.__dict__ for d in impact.downstream],
                "downstream_count": len(impact.downstream),
            }

    return {
        "astronaut_id": result.astronaut_id,
        "day_modified": result.day_modified,
        "reassigned_to": result.reassigned_to,
        "feasibility": feasibility_result,
        "comparison": result.comparison,
        "dependency_impact": dependency_impact,
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


@app.post("/mission-brief")
def mission_brief(req: BriefRequest, db: Session = Depends(get_db)):
    """
    Generate a Mission Decision Brief for a specific astronaut on a specific day.

    Assembles a BriefContext from deterministic backend results (fatigue state,
    optional mission summary, optional What-If trajectory, optional feasibility
    check, optional dependency impact) and passes it to generate_mission_brief()
    in bob.py, which calls IBM Granite via watsonx.ai or falls back to a
    deterministic template.

    Granite ONLY interprets the supplied computed facts.  It never calculates,
    modifies, or overrides any deterministic value.  human_review_required is
    always True in the response.

    No DB caching -- each request assembles fresh context from the current
    mission state.
    """
    # --- 1. Resolve the subject astronaut's fatigue record for the requested day ---
    profile = load_profile(db, req.astronaut_id)
    if profile is None:
        raise HTTPException(404, "Unknown astronaut_id")

    all_records = load_mission_records(db, req.astronaut_id).get(req.astronaut_id, [])
    record = next((r for r in all_records if r.day == req.day), None)
    if record is None:
        raise HTTPException(404, f"No mission record for day {req.day}")

    sub_scores = {
        "reaction_time": record.drift.reaction_time_score,
        "sleep_debt":    record.drift.sleep_debt_score,
        "circadian":     record.drift.circadian_score,
        "workload":      record.drift.workload_score,
    }

    # --- 2. Optional: mission-level risk summary ---
    mission_summary = None
    if req.include_mission_summary:
        mission_summary = mission_risk_summary(load_mission_records(db))

    # --- 3. Optional: What-If trajectory ---
    whatif_comparison = None
    whatif_reassign_to = None
    feasibility_status = None
    feasibility_reasons = None
    feasibility_warnings = None
    receiver_drift_score = None
    receiver_risk_level = None
    workload_projected_ratio = None

    if req.whatif_task_id or (req.whatif_reassign_to is not None and req.whatif_task_load_delta is not None):
        # Derive astronaut/day/load from task_id if given, else use req fields
        if req.whatif_task_id:
            task_row = db.get(Task, req.whatif_task_id)
            if task_row is None:
                raise HTTPException(404, f"Unknown whatif_task_id '{req.whatif_task_id}'")
            wi_astronaut_id = task_row.astronaut_id
            wi_day = task_row.day
            wi_delta = -task_row.load
        else:
            wi_astronaut_id = req.astronaut_id
            wi_day = req.day
            wi_delta = req.whatif_task_load_delta  # already validated le=0.0

        wi_profile = load_profile(db, wi_astronaut_id)
        if wi_profile is None:
            raise HTTPException(404, f"Unknown astronaut for whatif: {wi_astronaut_id}")

        wi_records = load_mission_records(db, wi_astronaut_id).get(wi_astronaut_id, [])
        base_schedule = seed_module.task_derived_schedules([wi_profile], len(wi_records))[wi_astronaut_id]

        wi_result = simulate_intervention(
            wi_profile,
            wi_records,
            day=wi_day,
            task_load_delta=wi_delta,
            reassign_to=req.whatif_reassign_to,
            weights=CURRENT_WEIGHTS,
            base_schedule=base_schedule,
        )
        whatif_comparison = wi_result.comparison
        whatif_reassign_to = req.whatif_reassign_to

        # Run feasibility check if a receiver is named
        if req.whatif_reassign_to:
            receiver_profile = load_profile(db, req.whatif_reassign_to)
            if receiver_profile is not None and req.whatif_reassign_to != wi_astronaut_id:
                receiver_records = load_mission_records(db, req.whatif_reassign_to).get(
                    req.whatif_reassign_to, []
                )
                graph = load_task_graph(db)
                fr = check_reassignment_feasibility(
                    source_id=wi_astronaut_id,
                    receiver_id=req.whatif_reassign_to,
                    day=wi_day,
                    task_load_delta=wi_delta,
                    receiver_records=receiver_records,
                    graph=graph,
                )
                feasibility_status = fr.status
                feasibility_reasons = fr.reasons
                feasibility_warnings = fr.warnings
                # Extract the raw check numbers so Granite can explain the tradeoff.
                # These are already-computed facts inside FeasibilityResult.checks —
                # no new calculations, just promoting them into BriefContext.
                _fc_fatigue = fr.checks.get("fatigue")
                _fc_workload = fr.checks.get("workload")
                receiver_drift_score = getattr(_fc_fatigue, "drift_score", None)
                receiver_risk_level = getattr(_fc_fatigue, "risk_level", None)
                workload_projected_ratio = getattr(_fc_workload, "projected_ratio", None)

    # --- 4. Optional: dependency impact for a specific task ---
    impacted_task_name = None
    impacted_task_id = None
    downstream_count = None
    downstream_tasks_list = None

    impact_task_id = req.impact_task_id or req.whatif_task_id
    if impact_task_id:
        graph = load_task_graph(db)
        impact = compute_impact(impact_task_id, graph)
        if impact is not None:
            impacted_task_name = impact.task.name
            impacted_task_id = impact.task.task_id
            downstream_count = len(impact.downstream)
            downstream_tasks_list = [d.__dict__ for d in impact.downstream]

    # --- 5. Assemble BriefContext from all collected deterministic facts ---
    ctx = BriefContext(
        astronaut_name=profile.name,
        astronaut_id=req.astronaut_id,
        day=req.day,
        drift_score=record.drift.drift_score,
        risk_level=record.drift.risk_level,
        sub_scores=sub_scores,
        mission_overall_risk_level=mission_summary.get("overall_risk_level") if mission_summary else None,
        mission_worst_day=mission_summary.get("worst_day") if mission_summary else None,
        mission_worst_drift_score=mission_summary.get("worst_drift_score") if mission_summary else None,
        whatif_comparison=whatif_comparison,
        reassigned_to=whatif_reassign_to,
        feasibility_status=feasibility_status,
        feasibility_reasons=feasibility_reasons,
        feasibility_warnings=feasibility_warnings,
        receiver_drift_score=receiver_drift_score,
        receiver_risk_level=receiver_risk_level,
        workload_projected_ratio=workload_projected_ratio,
        impacted_task_name=impacted_task_name,
        impacted_task_id=impacted_task_id,
        downstream_count=downstream_count,
        downstream_tasks=downstream_tasks_list,
    )

    # --- 6. Generate the brief (watsonx or deterministic fallback) ---
    brief = generate_mission_brief(ctx)

    return dataclasses.asdict(brief)
