"""After-hours run window and admin "Stop all jobs"."""
import json
import os
os.environ["POC_NO_BACKGROUND"] = "1"

import time

import pytest
from fastapi.testclient import TestClient

from server import (DEFAULT_ADMIN_PASSWORD, DEFAULT_ROOM_CODE,
                    DEFAULT_ROOM_PASSWORD, create_app, make_state,
                    run_forecast_cycle, run_scheduler_cycle)
from settings import Settings, in_run_window

NOW = 100.0


def _m(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _state(tmp_path):
    return make_state(db_path=":memory:",
                      model_path=str(tmp_path / "no_model.pkl"),
                      settings_path=str(tmp_path / "settings.json"),
                      esg_path=str(tmp_path / "esg.json"))


def _join(client, name="Node-A", role="worker"):
    pw = (DEFAULT_ADMIN_PASSWORD if role == "admin"
          else DEFAULT_ROOM_PASSWORD)
    body = {"room_code": DEFAULT_ROOM_CODE, "password": pw, "role": role}
    if role == "worker":
        body["node_name"] = name
        body["capabilities"] = {"cpu_cores": 4, "ram_gb": 8, "os": "t",
                                "has_gpu": False, "agent_version": "t"}
    r = client.post("/join", json=body)
    assert r.status_code == 201, r.text
    return r.json()["token"]


def _auth(tok):
    return {"Authorization": f"Bearer {tok}"}


def _seed_ready(state, name, now=NOW, temp=45.0):
    # Same setup as test_chat_g4.py: READY worker with the selected model.
    for i in range(10):
        state.store.insert(name, now - 90 + i * 10, temp, None, 10.0, 20.0)
    run_forecast_cycle(state, now)
    from room_assets import RUNTIME_ID, get_model
    model = get_model(state.settings.get()["model_id"])
    state.forecast_cache.patch(
        name, model_ready=True, model_id=model["model_id"],
        model_sha256=model["sha256"], model_generation=state.llm_generation,
        runtime_id=RUNTIME_ID)


def _setup(tmp_path):
    state = _state(tmp_path)
    c = TestClient(create_app(state))
    admin = _join(c, role="admin")
    _join(c, "Node-A")
    _seed_ready(state, "Node-A")
    return state, c, admin


def _reserved_ids(state):
    return {j["id"] for j in state.balancer.reserved() if j.get("target")}


def test_in_run_window():
    assert in_run_window("", "", 0) is True
    assert in_run_window("09:00", "", 0) is True
    day = ("09:00", "17:00")
    assert not in_run_window(*day, _m("08:59"))
    assert in_run_window(*day, _m("09:00"))
    assert in_run_window(*day, _m("16:59"))
    assert not in_run_window(*day, _m("17:00"))
    night = ("17:30", "06:30")
    assert not in_run_window(*night, _m("17:29"))
    assert in_run_window(*night, _m("17:30"))
    assert in_run_window(*night, _m("23:59"))
    assert in_run_window(*night, _m("00:00"))
    assert in_run_window(*night, _m("06:29"))
    assert not in_run_window(*night, _m("06:30"))


@pytest.mark.parametrize("start,end", [
    ("25:00", "06:00"), ("9:00x", "10:00"), ("09:00", "09:00"),
    ("09:00", ""), ("", "17:00"),
])
def test_set_run_window_rejects_bad_values(tmp_path, start, end):
    s = Settings(str(tmp_path / "settings.json"))
    with pytest.raises(ValueError):
        s.set_run_window(start, end)


def test_settings_persist_and_reload(tmp_path):
    path = str(tmp_path / "settings.json")
    s = Settings(path)
    s.set_run_window("17:30", "06:30")
    s.set_jobs_paused(True)
    again = Settings(path).get()
    assert again["run_window_start"] == "17:30"
    assert again["run_window_end"] == "06:30"
    assert again["jobs_paused"] is True


def test_bad_window_on_disk_loads_as_off(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"run_window_start": "99:99",
                                "run_window_end": "06:00"}))
    got = Settings(str(path)).get()
    assert got["run_window_start"] == "" and got["run_window_end"] == ""
    assert got["jobs_paused"] is False


def test_paused_blocks_handout_then_resume(tmp_path):
    state, _, _ = _setup(tmp_path)
    state.settings.set_jobs_paused(True)
    job = state.balancer.enqueue_job(job_type="burn", duration_s=5)
    run_scheduler_cycle(state, NOW)
    assert job["id"] not in _reserved_ids(state)
    state.settings.set_jobs_paused(False)
    run_scheduler_cycle(state, NOW)
    assert job["id"] in _reserved_ids(state)


def test_outside_window_holds_burn_but_hands_out_chat(tmp_path):
    state, c, admin = _setup(tmp_path)
    lt = time.localtime(NOW)
    minute = lt.tm_hour * 60 + lt.tm_min

    def fmt(m):
        m %= 1440
        return f"{m // 60:02d}:{m % 60:02d}"

    state.settings.set_run_window(fmt(minute + 60), fmt(minute + 120))
    burn = state.balancer.enqueue_job(job_type="burn", duration_s=5)
    r = c.post("/chat", headers=_auth(admin), json={"prompt": "hi"})
    assert r.status_code == 202, r.text
    chat_id = r.json()["job_id"]
    run_scheduler_cycle(state, NOW)
    reserved = _reserved_ids(state)
    assert chat_id in reserved
    assert burn["id"] not in reserved


def test_api_stop_all_and_bad_window(tmp_path):
    state, c, admin = _setup(tmp_path)
    running = state.balancer.enqueue_job(job_type="burn", duration_s=5)
    run_scheduler_cycle(state, NOW)
    assert state.balancer.claim("Node-A", now=NOW)["id"] == running["id"]
    _join(c, "Node-B")  # Node-A is busy; Node-B takes the next one
    _seed_ready(state, "Node-B")
    job = state.balancer.enqueue_job(job_type="burn", duration_s=5)
    run_scheduler_cycle(state, NOW)
    assert job["id"] in _reserved_ids(state)

    r = c.post("/api/settings", headers=_auth(admin),
               json={"jobs_paused": True})
    assert r.status_code == 200, r.text
    assert state.settings.get()["jobs_paused"] is True
    # Running job and reservation taken back at once
    assert not state.balancer.list_active()
    assert not _reserved_ids(state)
    assert state.forecast_cache.get("Node-A").inflight == 0
    assert state.forecast_cache.get("Node-B").inflight == 0

    r = c.post("/api/settings", headers=_auth(admin),
               json={"run_window_start": "25:00", "run_window_end": "06:00"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "BAD_SETTINGS"
