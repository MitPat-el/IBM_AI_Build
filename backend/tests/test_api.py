"""
End-to-end API tests via FastAPI's TestClient, against the real (isolated
test) database. These exist specifically to catch the class of bug that
unit tests on individual modules miss -- wiring problems between main.py,
db/session.py, and the rest of the app. Several of these tests are direct
regression tests for bugs that were actually found and fixed during
development, not hypothetical edge cases.
"""

import sqlite3
import os


def test_crew_is_populated_on_startup(client):
    crew = client.get("/crew").json()
    assert len(crew) == 3
    assert {c["astronaut_id"] for c in crew} == {"A1", "A2", "A3"}


def test_mission_summary_route_not_shadowed_by_dynamic_route(client):
    """Regression test: /mission/summary was originally declared after
    /mission/{astronaut_id}, so FastAPI matched 'summary' as an
    astronaut_id and 404'd."""
    resp = client.get("/mission/summary")
    assert resp.status_code == 200
    assert "overall_risk_level" in resp.json()


def test_unknown_astronaut_returns_404_not_500(client):
    assert client.get("/mission/ZZZ").status_code == 404
    assert client.get("/explain/ZZZ/1").status_code == 404
    assert client.get("/replay/ZZZ").status_code == 404


def test_explain_does_not_trigger_below_threshold(client):
    resp = client.get("/explain/A1/1").json()
    assert resp["triggered"] is False
    assert "astronaut_message" not in resp


def test_explain_triggers_above_threshold(client):
    resp = client.get("/explain/A1/6").json()
    assert resp["triggered"] is True
    assert resp["astronaut_message"]
    assert resp["flight_surgeon_brief"]
    assert resp["suggested_intervention"]


def test_explanation_is_actually_persisted_not_just_returned(client):
    """Regression test: db/session.py's get_db() originally never called
    session.commit(), so the response looked correct but nothing was
    saved to disk. This checks the database file directly, not just the
    API response, so it can't be fooled by the same class of bug."""
    client.get("/explain/A1/6")

    db_path = os.environ["DATABASE_URL"].replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    count = conn.execute(
        "SELECT COUNT(*) FROM explanations WHERE astronaut_id='A1' AND day=6"
    ).fetchone()[0]
    conn.close()
    assert count == 1


def test_explanation_is_cached_not_regenerated_on_second_call(client):
    first = client.get("/explain/A1/6").json()
    second = client.get("/explain/A1/6").json()
    assert first["astronaut_message"] == second["astronaut_message"]

    db_path = os.environ["DATABASE_URL"].replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM explanations").fetchone()[0]
    conn.close()
    assert count == 1  # not 2 -- second call should hit the cache, not insert again


def test_mission_reset_actually_applies_custom_weights(client):
    """Regression test: /mission/reset originally silently ignored
    posted weights due to FastAPI's embed=True body-shape requirement,
    falling back to defaults with no error."""
    before = client.get("/mission/A1").json()[-1]["drift_score"]
    client.post("/mission/reset", json={
        "weights": {"reaction_time": 0.1, "sleep_debt": 0.1, "circadian": 0.1, "workload": 0.7}
    })
    after = client.get("/mission/A1").json()[-1]["drift_score"]
    assert before != after


def test_mission_reset_clears_stale_explanations(client):
    client.get("/explain/A1/6")
    client.post("/mission/reset", json={})

    db_path = os.environ["DATABASE_URL"].replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM explanations").fetchone()[0]
    conn.close()
    assert count == 0


def test_whatif_reassign_matches_stored_data_before_intervention_day(client):
    """Regression test: the What-If Simulator re-simulates from an
    AstronautProfile reconstructed from the DB row. If the persisted
    seed doesn't match what was used to generate the stored data, every
    day (not just the intervention day forward) would show spurious
    deltas."""
    resp = client.post("/whatif/reassign", json={
        "astronaut_id": "A2", "day": 4, "reassign_to": "A3", "task_load_delta": -8
    }).json()
    for row in resp["comparison"]:
        if row["day"] < 4:
            assert row["delta"] == 0.0


def test_replay_projection_extends_timeline(client):
    resp = client.get("/replay/A2", params={"project_days": 3}).json()
    assert len(resp["points"]) == 6
    assert len(resp["projected_points"]) == 3
    assert resp["projected_points"][0]["day"] == 7


def test_full_mission_risk_map_shape(client):
    resp = client.get("/mission").json()
    assert len(resp) == 6
    for day in resp:
        assert set(day["per_astronaut"].keys()) == {"A1", "A2", "A3"}


def test_tasks_graph_is_populated_on_startup(client):
    resp = client.get("/tasks/graph").json()
    assert len(resp["nodes"]) == 18
    assert len(resp["edges"]) == 12


def test_self_heals_missing_tasks_without_wiping_astronauts(client):
    """Regression test for a real reported issue: a database seeded
    before the Dependency Graph feature existed has astronauts/missions
    but no tasks. On next startup, the app must backfill just the tasks
    -- not wipe and reseed everything, and not require the person to
    manually delete the database file."""
    import sqlite3
    import os as os_module

    db_path = os_module.environ["DATABASE_URL"].replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM task_dependencies")
    conn.execute("DELETE FROM tasks")
    conn.commit()
    conn.close()

    # simulate a server restart against this now-stale db: a fresh
    # TestClient re-triggers the lifespan startup logic
    from fastapi.testclient import TestClient
    from main import app

    with TestClient(app) as client2:
        graph = client2.get("/tasks/graph").json()
        assert len(graph["nodes"]) == 18  # backfilled

        crew = client2.get("/crew").json()
        assert {c["astronaut_id"] for c in crew} == {"A1", "A2", "A3"}  # untouched, not reset


def test_self_heal_is_idempotent_no_duplicate_tasks_on_repeated_restarts(client):
    """A healthy database (tasks already present) must not get
    reseeded/duplicated on every restart."""
    from fastapi.testclient import TestClient
    from main import app

    with TestClient(app) as client2:
        pass  # trigger one more lifespan startup against the already-healthy db
    with TestClient(app) as client3:
        graph = client3.get("/tasks/graph").json()
        assert len(graph["nodes"]) == 18  # still 18, not 36 or 54


def test_task_impact_route_not_shadowed_by_graph_or_at_risk(client):
    """Regression guard: /tasks/graph and /tasks/at-risk must stay
    declared before /tasks/{task_id}/impact, the same route-ordering
    trap that broke /mission/summary earlier."""
    assert client.get("/tasks/graph").status_code == 200
    assert client.get("/tasks/at-risk").status_code == 200
    assert client.get("/tasks/T1/impact").status_code == 200


def test_task_impact_matches_the_seeded_critical_path(client):
    """End-to-end regression test: T1 sits at the head of the demo's
    critical path (T1 -> T2 -> T8 -> T12 -> T18). If the seeded task
    DAG or the impact logic ever changes in a way that breaks this
    chain, the demo's headline moment breaks with it."""
    resp = client.get("/tasks/T1/impact").json()
    downstream_ids = {d["task_id"] for d in resp["downstream"]}
    assert downstream_ids == {"T2", "T8", "T12", "T18"}
    assert resp["downstream_count"] == 4


def test_unknown_task_returns_404(client):
    assert client.get("/tasks/NOPE/impact").status_code == 404


def test_at_risk_tasks_only_includes_high_or_critical(client):
    resp = client.get("/tasks/at-risk").json()
    assert len(resp) > 0
    for task in resp:
        assert task["risk_level"] in ("high", "critical")


def test_at_risk_tasks_sorted_by_downstream_count_descending(client):
    resp = client.get("/tasks/at-risk").json()
    counts = [t["downstream_count"] for t in resp]
    assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------------------
# /whatif/reassign -- feasibility integration tests
# ---------------------------------------------------------------------------

def test_whatif_feasibility_present_when_reassign_to_given(client):
    resp = client.post("/whatif/reassign", json={
        "astronaut_id": "A1", "day": 2, "reassign_to": "A3", "task_load_delta": -2.0,
    }).json()
    assert "feasibility" in resp
    feas = resp["feasibility"]
    assert feas is not None
    for key in ("status", "receiver", "checks", "reasons", "warnings", "advisory"):
        assert key in feas, f"Missing key '{key}' in feasibility response"


def test_whatif_feasibility_null_when_reassign_to_is_none(client):
    """Task delay (no receiver) must return feasibility=null; comparison must still be present."""
    resp = client.post("/whatif/reassign", json={
        "astronaut_id": "A1", "day": 2, "task_load_delta": -3.0,
    }).json()
    assert resp["feasibility"] is None
    assert len(resp["comparison"]) > 0


def test_whatif_feasibility_not_feasible_unknown_receiver(client):
    """Unknown receiver -> not_feasible; comparison for source still returned."""
    resp = client.post("/whatif/reassign", json={
        "astronaut_id": "A1", "day": 2, "reassign_to": "ZZZ", "task_load_delta": -3.0,
    }).json()
    assert resp["feasibility"]["status"] == "not_feasible"
    assert any("does not exist" in r for r in resp["feasibility"]["reasons"])
    # Simulation still ran
    assert len(resp["comparison"]) > 0


def test_whatif_feasibility_not_feasible_self_reassignment(client):
    """Self-reassignment -> not_feasible."""
    resp = client.post("/whatif/reassign", json={
        "astronaut_id": "A2", "day": 3, "reassign_to": "A2", "task_load_delta": -4.0,
    }).json()
    assert resp["feasibility"]["status"] == "not_feasible"
    assert any("same astronaut" in r for r in resp["feasibility"]["reasons"])


def test_whatif_feasibility_advisory_disclaimer_in_every_response(client):
    """Advisory disclaimer must be present and untruncated in every feasibility response."""
    from feasibility import ADVISORY_DISCLAIMER
    resp = client.post("/whatif/reassign", json={
        "astronaut_id": "A1", "day": 2, "reassign_to": "A3", "task_load_delta": -2.0,
    }).json()
    assert resp["feasibility"]["advisory"] == ADVISORY_DISCLAIMER


def test_whatif_comparison_present_regardless_of_feasibility_status(client):
    """Comparison is never suppressed by a not_feasible result."""
    resp = client.post("/whatif/reassign", json={
        "astronaut_id": "A1", "day": 2, "reassign_to": "ZZZ", "task_load_delta": -3.0,
    }).json()
    assert resp["feasibility"]["status"] == "not_feasible"
    assert isinstance(resp["comparison"], list)
    assert len(resp["comparison"]) > 0


def test_whatif_day_field_rejects_zero(client):
    """Pydantic Field(ge=1) must reject day=0 with HTTP 422."""
    resp = client.post("/whatif/reassign", json={
        "astronaut_id": "A1", "day": 0, "task_load_delta": -3.0,
    })
    assert resp.status_code == 422


def test_whatif_day_field_rejects_negative(client):
    """Pydantic Field(ge=1) must reject day=-1 with HTTP 422."""
    resp = client.post("/whatif/reassign", json={
        "astronaut_id": "A1", "day": -1, "task_load_delta": -3.0,
    })
    assert resp.status_code == 422


def test_whatif_checks_dict_has_three_subkeys_for_valid_receiver(client):
    """When feasibility runs fully, checks must contain fatigue, workload, dependency_conflict."""
    resp = client.post("/whatif/reassign", json={
        "astronaut_id": "A1", "day": 1, "reassign_to": "A2", "task_load_delta": -2.0,
    }).json()
    checks = resp["feasibility"]["checks"]
    assert set(checks.keys()) == {"fatigue", "workload", "dependency_conflict"}


def test_whatif_feasibility_status_is_one_of_three_values(client):
    """Status must be one of the three defined values."""
    resp = client.post("/whatif/reassign", json={
        "astronaut_id": "A1", "day": 3, "reassign_to": "A3", "task_load_delta": -4.0,
    }).json()
    assert resp["feasibility"]["status"] in ("feasible", "feasible_with_caution", "not_feasible")
