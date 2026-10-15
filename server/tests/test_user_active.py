"""Step 4a: a PC with recent keyboard/mouse input gets no new jobs."""
import os
os.environ["POC_NO_BACKGROUND"] = "1"
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from forecast_cache import NodeForecast
from scheduler import filter_candidates
from server import (DEFAULT_ROOM_CODE, DEFAULT_ROOM_PASSWORD,
                    build_state_payload, create_app, make_state,
                    run_forecast_cycle)


# Helpers copied from test_server.py
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


def seed_ramp(state, node, start_ts, start_temp, slope_per_min, n=10, step=10.0):
    for i in range(n):
        ts = start_ts + i * step
        state.store.insert(
            node, ts,
            start_temp + slope_per_min * (i * step) / 60.0,
            None, 90.0, 40.0)


def _nf(name, **kw):
    defaults = dict(
        node=name, state="READY", predicted_max_c=60.0, current_temp_c=50.0,
        cpu_util=10.0, power_w=20.0, power_source="sensor",
        inflight=0, max_concurrent=1, model_ready=True,
        last_sample_ts=1000.0, idle_baseline_c=35.0,
        effective_threshold_c=75.0, p_idle_w=15.0, p_max_w=65.0,
    )
    defaults.update(kw)
    return NodeForecast(**defaults)


# --- scheduler ---------------------------------------------------------

def test_user_active_node_rejected_for_burn_and_chat():
    cache = {"A": _nf("A", user_active=True)}
    for job in ({"type": "burn"}, {"type": "chat"}, {}):
        cands, rejected = filter_candidates(cache, job, now=1000.0)
        assert cands == []
        assert rejected["A"] == "user_active"
    # round_robin mode too
    cands, rejected = filter_candidates(cache, {}, now=1000.0, skip_flagged=True)
    assert cands == [] and rejected["A"] == "user_active"


def test_idle_node_is_candidate():
    cache = {"A": _nf("A", user_active=False)}
    cands, _ = filter_candidates(cache, {"type": "burn"}, now=1000.0)
    assert cands == ["A"]


def test_fake_row_without_user_active_attribute_still_works():
    f = SimpleNamespace(
        state="READY", predicted_max_c=60.0, current_temp_c=50.0,
        inflight=0, max_concurrent=1, model_ready=True,
        last_sample_ts=1000.0)
    cands, _ = filter_candidates({"A": f}, {}, now=1000.0)
    assert cands == ["A"]


# --- API ---------------------------------------------------------------

def _setup(tmp_path, name="Node-U"):
    state = make_test_state(tmp_path)
    client = TestClient(create_app(state))
    tok = _join(client, name)
    # Ingest allows 2 posts per window; these tests post more often.
    state.rate_limiter = None
    return state, client, tok


def _ingest(client, tok, **extra):
    body = {"cpu_temp": 50, "cpu_util": 10}
    body.update(extra)
    return client.post("/ingest", json=body, headers=_auth(tok))


def test_ingest_sets_user_active(tmp_path):
    state, client, tok = _setup(tmp_path)
    r = _ingest(client, tok, seconds_since_input=10)
    assert 200 <= r.status_code < 300, r.text
    assert state.forecast_cache.get("Node-U").user_active is True
    r = _ingest(client, tok, seconds_since_input=500)
    assert 200 <= r.status_code < 300, r.text
    assert state.forecast_cache.get("Node-U").user_active is False
    _ingest(client, tok, seconds_since_input=10)
    r = _ingest(client, tok)  # field omitted (old agent / not Windows)
    assert 200 <= r.status_code < 300, r.text
    assert state.forecast_cache.get("Node-U").user_active is False


def test_ingest_rejects_bad_seconds(tmp_path):
    state, client, tok = _setup(tmp_path)
    assert _ingest(client, tok, seconds_since_input=True).status_code == 422
    assert _ingest(client, tok, seconds_since_input=-1).status_code == 422


def test_host_own_pc_is_exempt(tmp_path):
    state, client, tok = _setup(tmp_path)
    state.local_agent["node"] = "Node-U"
    r = _ingest(client, tok, seconds_since_input=10)
    assert 200 <= r.status_code < 300, r.text
    assert state.forecast_cache.get("Node-U").user_active is False


def test_flag_survives_forecast_rebuild_and_put(tmp_path):
    state, client, tok = _setup(tmp_path)
    _ingest(client, tok, seconds_since_input=10)
    now = time.time()
    seed_ramp(state, "Node-U", now - 90.0, 45.0, slope_per_min=0.0)
    run_forecast_cycle(state, now=now)
    assert state.forecast_cache.get("Node-U").user_active is True
    state.forecast_cache.put(_nf("Node-U", user_active=False))
    assert state.forecast_cache.get("Node-U").user_active is True


def test_state_payload_live_node_has_user_active(tmp_path):
    state, client, tok = _setup(tmp_path)
    _ingest(client, tok, seconds_since_input=10)
    now = time.time()
    seed_ramp(state, "Node-U", now - 90.0, 45.0, slope_per_min=0.0)
    run_forecast_cycle(state, now=now)
    payload = build_state_payload(state, now=now + 1.0)
    node = next(n for n in payload["nodes"] if n["name"] == "Node-U")
    assert node["live"] is True
    assert node["user_active"] is True
