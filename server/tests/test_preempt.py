"""Step 4b: a burn job stopped because a person came back is requeued,
not counted as the PC's error, a used attempt, or a completed job."""
import os
os.environ["POC_NO_BACKGROUND"] = "1"
import time

from fastapi.testclient import TestClient

from forecast_cache import NodeForecast
from server import (DEFAULT_ROOM_CODE, DEFAULT_ROOM_PASSWORD, create_app,
                    make_state, run_scheduler_cycle)


def make_test_state(tmp_path):
    return make_state(db_path=":memory:",
                      model_path=str(tmp_path / "no_model.pkl"),
                      settings_path=str(tmp_path / "settings.json"),
                      esg_path=str(tmp_path / "esg.json"))


def _join(client, node_name):
    r = client.post("/join", json={
        "room_code": DEFAULT_ROOM_CODE, "password": DEFAULT_ROOM_PASSWORD,
        "role": "worker", "node_name": node_name,
        "capabilities": {"cpu_cores": 4, "ram_gb": 8, "os": "test",
                         "has_gpu": False, "agent_version": "test"}})
    assert r.status_code == 201, r.text
    return r.json()["token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _ready_row(name, now):
    # A healthy, schedulable forecast row (put keeps live inflight/flags).
    return NodeForecast(
        node=name, state="READY", predicted_max_c=60.0, current_temp_c=50.0,
        cpu_util=10.0, power_w=20.0, power_source="sensor",
        inflight=0, max_concurrent=1, model_ready=True,
        last_sample_ts=now, idle_baseline_c=35.0,
        effective_threshold_c=75.0, p_idle_w=15.0, p_max_w=65.0)


def _setup(tmp_path, name="Node-P"):
    state = make_test_state(tmp_path)
    client = TestClient(create_app(state))
    tok = _join(client, name)
    state.rate_limiter = None
    now = time.time()
    state.forecast_cache.put(_ready_row(name, now))
    state.forecast_cache.patch(name, user_active=False, inflight=0)
    return state, client, tok


def _reserve_burn(state, name):
    job = state.balancer.enqueue_job(duration_s=10, cores=1)
    assert job is not None
    run_scheduler_cycle(state, time.time())
    reserved = [j for j in state.balancer.reserved() if j["id"] == job["id"]]
    assert reserved and reserved[0]["target"] == name
    return job["id"]


# wait=0 returns 204 before any claim on Linux (the deadline is already
# past by the first check), so these tests use a short real wait.
NEXT = "/jobs/next?wait=0.2"


def _pending(state, jid):
    return next((j for j in state.balancer.pending() if j["id"] == jid), None)


def _events(state, kind):
    return [e for e in state.store.esg_events() if e["event_type"] == kind]


def test_preempted_result_requeues_without_penalty(tmp_path):
    state, client, tok = _setup(tmp_path)
    jid = _reserve_burn(state, "Node-P")
    attempts_reserved = state.balancer.reserved()[0]["attempts"]
    r = client.get(NEXT, headers=_auth(tok))
    assert r.status_code == 200, r.text
    assert r.json()["id"] == jid
    active = state.balancer.peek_active("Node-P")
    errs_before = state.forecast_cache.get("Node-P").consecutive_errors
    completed_before = len(_events(state, "job_completed"))

    r = client.post("/jobs/result", headers=_auth(tok), json={
        "job_id": jid, "status": "preempted",
        "attempt_id": active["attempt_id"],
        "energy_j": 5.0, "energy_source": "model"})
    assert 200 <= r.status_code < 300, r.text
    assert r.json().get("requeued") is True

    job = _pending(state, jid)
    assert job is not None and job["target"] is None
    assert state.balancer.get_failed(jid) is None
    assert state.balancer.peek_active("Node-P") is None
    # The preemption gives back the attempt the reservation used.
    assert job["attempts"] == attempts_reserved - 1
    fc = state.forecast_cache.get("Node-P")
    assert fc.consecutive_errors == errs_before
    assert fc.inflight == 0
    assert fc.user_active is True
    assert len(_events(state, "job_preempted")) == 1
    assert len(_events(state, "job_completed")) == completed_before


def test_bogus_status_rejected(tmp_path):
    state, client, tok = _setup(tmp_path)
    r = client.post("/jobs/result", headers=_auth(tok), json={
        "job_id": "x", "status": "bogus"})
    assert r.status_code == 422, r.text


def test_user_active_node_gives_reservation_back(tmp_path):
    state, client, tok = _setup(tmp_path)
    jid = _reserve_burn(state, "Node-P")
    assert state.forecast_cache.get("Node-P").inflight == 1
    state.forecast_cache.patch("Node-P", user_active=True)

    r = client.get(NEXT, headers=_auth(tok))
    assert r.status_code == 204, r.text
    job = _pending(state, jid)
    assert job is not None and job["target"] is None
    assert state.forecast_cache.get("Node-P").inflight == 0
    assert state.balancer.peek_active("Node-P") is None

    # Person leaves: the next scheduler cycle may use this PC again.
    state.forecast_cache.patch("Node-P", user_active=False,
                               last_sample_ts=time.time())
    run_scheduler_cycle(state, time.time())
    reserved = [j for j in state.balancer.reserved() if j["id"] == jid]
    assert reserved and reserved[0]["target"] == "Node-P"
