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


def test_whatif_by_task_id_derives_astronaut_and_day(client):
    """T7 belongs to A1 on day 3 -- posting just the task_id should
    correctly derive both without the caller specifying them."""
    resp = client.post("/whatif/reassign", json={"task_id": "T7", "reassign_to": "A3"}).json()
    assert resp["astronaut_id"] == "A1"
    assert resp["day_modified"] == 3


def test_whatif_by_task_id_includes_dependency_impact(client):
    resp = client.post("/whatif/reassign", json={"task_id": "T7"}).json()
    assert resp["dependency_impact"] is not None
    assert resp["dependency_impact"]["task"]["task_id"] == "T7"
    downstream_ids = {d["task_id"] for d in resp["dependency_impact"]["downstream"]}
    assert downstream_ids == {"T10", "T13", "T16", "T17", "T18"}


def test_whatif_manual_mode_has_no_dependency_impact(client):
    """Manual mode (no task_id) shouldn't fabricate a dependency_impact
    section -- it should be explicitly null, not omitted or guessed."""
    resp = client.post("/whatif/reassign", json={
        "astronaut_id": "A2", "day": 4, "task_load_delta": -3
    }).json()
    assert resp["dependency_impact"] is None


def test_whatif_unknown_task_id_returns_404(client):
    assert client.post("/whatif/reassign", json={"task_id": "NOPE"}).status_code == 404


def test_whatif_missing_required_fields_returns_400_not_500(client):
    assert client.post("/whatif/reassign", json={}).status_code == 400


def test_task_load_matches_the_actual_task_the_day_it_belongs_to(client):
    """Regression test for the core Dependency Graph <-> Projection
    connection: a day's task_load (used by the drift formula) must
    equal the sum of that astronaut's actual assigned task loads that
    day -- not an unrelated shared number. T16 is A1's only day-6 task
    at load 6.0."""
    mission = client.get("/mission/A1").json()
    day6 = next(d for d in mission if d["day"] == 6)
    assert day6["task_load"] == 6.0


def test_whatif_on_a_task_that_is_above_astronaut_average_shows_real_drift_change(client):
    """Regression test for the actual reported issue: T16 (A1, day 6)
    is the top-ranked /tasks/at-risk result, and under the old shared
    global schedule its day was already at/below the crew-wide rolling
    average, so simulating its reassignment always showed delta=0.0 --
    a broken-looking result on the most natural demo path. With
    per-astronaut task-derived workload, A1's own day-6 load (6.0) is
    above A1's own rolling average, so the intervention must show a
    real negative delta on day 6."""
    resp = client.post("/whatif/reassign", json={"task_id": "T16", "reassign_to": "A3"}).json()
    day6 = next(r for r in resp["comparison"] if r["day"] == 6)
    assert day6["delta"] < 0


# ---------------------------------------------------------------------------
# POST /mission-brief -- integration tests
# ---------------------------------------------------------------------------

_BRIEF_REQUIRED_KEYS = {
    "astronaut_message", "executive_summary", "primary_drivers",
    "mission_implications", "intervention_assessment", "recommended_actions",
    "uncertainties", "human_review_required", "source",
}


def test_mission_brief_minimal_request_returns_valid_structure(client):
    """Minimum request: astronaut_id + day.  All optional context omitted."""
    resp = client.post("/mission-brief", json={"astronaut_id": "A1", "day": 4})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert _BRIEF_REQUIRED_KEYS.issubset(data.keys()), (
        f"Missing keys: {_BRIEF_REQUIRED_KEYS - data.keys()}"
    )


def test_mission_brief_human_review_required_always_true(client):
    """human_review_required must be True regardless of source."""
    resp = client.post("/mission-brief", json={"astronaut_id": "A2", "day": 3})
    assert resp.json()["human_review_required"] is True


def test_mission_brief_source_is_not_watsonx_without_credentials(client):
    """
    Without watsonx credentials the response must never claim source='watsonx'.
    It may come from the local Ollama provider (source='ollama_granite') or the
    deterministic fallback (source='fallback_template:*') depending on whether
    Ollama is running in the test environment.
    """
    resp = client.post("/mission-brief", json={"astronaut_id": "A3", "day": 2})
    source = resp.json()["source"]
    assert source != "watsonx", f"Got 'watsonx' source without credentials: {source}"
    assert source in ("ollama_granite", "fallback_template:no_credentials",
                      "fallback_template:ollama_error"), f"Unexpected source: {source}"


def test_mission_brief_primary_drivers_is_list(client):
    resp = client.post("/mission-brief", json={"astronaut_id": "A1", "day": 4})
    drivers = resp.json()["primary_drivers"]
    assert isinstance(drivers, list)
    assert len(drivers) > 0


def test_mission_brief_recommended_actions_is_list_of_strings(client):
    resp = client.post("/mission-brief", json={"astronaut_id": "A1", "day": 4})
    actions = resp.json()["recommended_actions"]
    assert isinstance(actions, list)
    assert all(isinstance(a, str) for a in actions)


def test_mission_brief_uncertainties_contains_prototype_disclaimer(client):
    resp = client.post("/mission-brief", json={"astronaut_id": "A1", "day": 4})
    unc_text = " ".join(resp.json()["uncertainties"]).lower()
    assert "prototype" in unc_text


def test_mission_brief_with_mission_summary_included(client):
    """include_mission_summary=True (default) should populate executive_summary
    with mission-level risk context."""
    resp = client.post("/mission-brief", json={
        "astronaut_id": "A1", "day": 4, "include_mission_summary": True
    })
    assert resp.status_code == 200
    # executive_summary must be non-empty
    assert resp.json()["executive_summary"]


def test_mission_brief_with_whatif_task_id(client):
    """Supplying whatif_task_id should include What-If context in the brief."""
    resp = client.post("/mission-brief", json={
        "astronaut_id": "A1", "day": 6,
        "whatif_task_id": "T16",
        "whatif_reassign_to": "A3",
    })
    assert resp.status_code == 200
    data = resp.json()
    # intervention_assessment must not say "No What-If context supplied"
    assert "No What-If context supplied" not in data["intervention_assessment"]


def test_mission_brief_with_impact_task_id(client):
    """Supplying impact_task_id=T7 should include downstream task context
    (T7 has 5 downstream tasks in the seeded DAG)."""
    resp = client.post("/mission-brief", json={
        "astronaut_id": "A1", "day": 3,
        "impact_task_id": "T7",
    })
    assert resp.status_code == 200
    implications = " ".join(resp.json()["mission_implications"])
    # The fallback brief should mention the task and downstream count
    assert "EVA - External Repair" in implications or "5" in implications


def test_mission_brief_unknown_astronaut_returns_404(client):
    resp = client.post("/mission-brief", json={"astronaut_id": "ZZZ", "day": 1})
    assert resp.status_code == 404


def test_mission_brief_unknown_day_returns_404(client):
    resp = client.post("/mission-brief", json={"astronaut_id": "A1", "day": 99})
    assert resp.status_code == 404


def test_mission_brief_invalid_day_zero_returns_422(client):
    resp = client.post("/mission-brief", json={"astronaut_id": "A1", "day": 0})
    assert resp.status_code == 422


def test_mission_brief_unknown_whatif_task_id_returns_404(client):
    resp = client.post("/mission-brief", json={
        "astronaut_id": "A1", "day": 1, "whatif_task_id": "NOPE"
    })
    assert resp.status_code == 404


def test_mission_brief_existing_explain_endpoint_still_works(client):
    """/explain must remain unbroken -- the brief is a parallel feature."""
    resp = client.get("/explain/A1/6")
    assert resp.status_code == 200
    data = resp.json()
    assert "astronaut_message" in data
    assert "flight_surgeon_brief" in data
    assert "suggested_intervention" in data


def test_mission_brief_does_not_affect_cached_explanations(client):
    """Calling /mission-brief must not write to or clear the explanations table."""
    import sqlite3, os
    # Seed an explanation
    client.get("/explain/A1/6")
    # Call mission-brief
    client.post("/mission-brief", json={"astronaut_id": "A1", "day": 6})
    # The explanation row must still be there, unmodified
    db_path = os.environ["DATABASE_URL"].replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    count = conn.execute(
        "SELECT COUNT(*) FROM explanations WHERE astronaut_id='A1' AND day=6"
    ).fetchone()[0]
    conn.close()
    assert count == 1
