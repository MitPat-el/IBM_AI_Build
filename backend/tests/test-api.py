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