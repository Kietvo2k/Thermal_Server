"""Main server process: ingest API, pull-based job dispatch, forecast loop,
optional demo load generator, ESG accounting, dashboard + WebSocket. Run:
    python server.py                  (normal — no burn-job pump)
    python server.py --demo-load      (opt-in burn jobs every 4s)
    python server.py --calibrate      (calibration recording, see calibrate.py)
    python server.py --ab-benchmark   (A/B ESG trên cụm giả 10 node, docs/07)
"""
import asyncio
import contextlib
import io
import logging
from logging.handlers import RotatingFileHandler
import json
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

from fastapi import Depends, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    FileResponse, JSONResponse, RedirectResponse, Response,
)
from urllib.parse import quote, urlparse
from pydantic import BaseModel, Field, field_validator

from balancer import LoadBalancer
from chat_tracker import ChatTracker
from errors import ApiError, api_error_handler, validation_error_handler
from lifecycle import LifecycleGate, LifecycleWaitTimeout
from esg import EsgTracker, compute_report, load_esg_config
from features import MIN_SAMPLES, MIN_SPAN_S
from forecast_cache import (
    ForecastCache, NodeForecast, apply_hysteresis,
    effective_threshold, idle_baseline_from_samples, MAX_TEMP_MARGIN_C,
)
from forecaster import Forecaster
from llm import HOST_NODE
from power_model import PowerModel, load_power_model
from room import AuthContext, Room, make_default_room
from room_assets import (
    DEFAULT_MODEL_ID,
    RUNTIME_ID,
    UnknownModelError,
    get_model,
    list_models,
    room_config_llm,
)
from scheduler import (
    MAX_ATTEMPTS, RoundRobinCursor, reap_expired, schedule_once,
    SCHEDULE_EVERY_S,
)
from settings import Settings, in_run_window
from store import TelemetryStore
from weather import WeatherService, apply_weather_to_esg

STATE_PAYLOAD_TTL_S = 1.0
WS_UNAUTH_PER_IP = 5
WS_AUTH_PER_TOKEN = 2
WS_AUTH_PER_IP = 8
WS_ADMIN_EVENT_QUEUE_MAX = 64
# Dashboard and backend exchange this immutable value before using optional
# features, preventing a stale HTML file from failing later with a misleading
# 404. Bump it whenever a dashboard-facing API contract changes.
API_REVISION = "2026.08.03.3"
PASSWORD_MIN_LENGTH = 12


def configure_logging(log_file: str = "server.log") -> None:
    """UTF-8 cho stdout/stderr và mọi handler — không nhân đôi khi gọi lại.

    FileHandler encoding khác UTF-8 (vd cp1252) được thay bằng handler UTF-8.
    Giữ tối đa một tệp đang ghi và bốn bản xoay (10 MiB mỗi tệp).
    """
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s")
    has_stream = False
    expected_path = os.path.abspath(log_file)
    has_rotating_file = False
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.FileHandler):
            has_stream = True
            try:
                if hasattr(h.stream, "reconfigure"):
                    h.stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError, OSError):
                pass
            h.setFormatter(fmt)
        if isinstance(h, logging.FileHandler):
            enc = str(getattr(h, "encoding", "") or "").lower()
            actual_path = os.path.abspath(str(getattr(h, "baseFilename", "")))
            if (isinstance(h, RotatingFileHandler)
                    and "utf" in enc and actual_path == expected_path):
                has_rotating_file = True
                h.setFormatter(fmt)
            elif (not isinstance(h, RotatingFileHandler)
                  and "utf" in enc and actual_path != expected_path):
                # Không sở hữu handler UTF-8 của test runner hay thư viện
                # khác; giữ nguyên để configure_logging vẫn idempotent.
                h.setFormatter(fmt)
            else:
                # Handler cũ, sai encoding, hoặc trỏ sang log_file khác.
                try:
                    root.removeHandler(h)
                    h.close()
                except Exception:
                    pass
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    if not has_rotating_file:
        fh = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=4,
            encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


configure_logging()
log = logging.getLogger("server")

WINDOW_S = 180.0
FORECAST_EVERY_S = 5.0
GENERATOR_EVERY_S = 4.0
STALE_AFTER_S = 10.0
ACTIVE_WINDOW_S = 900.0
HYSTERESIS_C = 3.0
MIN_DWELL_S = 30.0
CLOCK_SKEW_S = 30.0
SKEW_LOG_EVERY_S = 300.0
WEATHER_EVERY_S = 15 * 60.0
TUNNEL_IDLE_CHECK_S = 60.0
SERVER_PORT = int(os.environ.get("POC_PORT", "8000"))

# Default room credentials for tests (override via make_state / CLI).
DEFAULT_ROOM_CODE = "THERMAL-LOCAL"
DEFAULT_ROOM_PASSWORD = "local-dev-password"       # worker
DEFAULT_ADMIN_PASSWORD = "local-admin-password"    # admin — khác worker

@dataclass
class AppState:
    store: TelemetryStore
    forecaster: Forecaster
    balancer: LoadBalancer
    esg: EsgTracker
    settings: Settings
    room: Room
    forecast_cache: ForecastCache = field(default_factory=ForecastCache)
    rr_cursor: RoundRobinCursor = field(default_factory=RoundRobinCursor)
    chat: ChatTracker = field(default_factory=ChatTracker)
    lifecycle: LifecycleGate = field(default_factory=LifecycleGate)
    calibrate: bool = False
    demo_load: bool = False
    tunnel: object | None = None
    weather: object | None = None
    rate_limiter: object | None = None
    power_model: PowerModel | None = None
    room_meta_path: str | None = None  # room.json — không plaintext password
    # Bumped atomically whenever the selected room model changes.  Agents
    # echo it in /nodes/ready so an old runtime cannot receive a new job.
    llm_generation: int = 0
    # Local Host agent lifecycle.  This is status only; the agent remains a
    # normal outbound worker and is never a server-side pseudo-node.
    local_agent: dict = field(default_factory=lambda: {
        "state": "missing", "message": "NodeAgent local chưa chạy",
        "node": None,
    })
    local_agent_launcher: object | None = None
    _skew_warned_at: dict = field(default_factory=dict)
    _job_polls: set = field(default_factory=set)  # node đang long-poll
    _ws_unauth_by_ip: dict = field(default_factory=dict)  # IP → số WS chưa auth
    _ws_unauth_lock: threading.Lock = field(default_factory=threading.Lock)
    _ws_auth_by_token: dict = field(default_factory=dict)  # token_hash → số WS
    _ws_auth_by_ip: dict = field(default_factory=dict)  # IP → số WS đã auth
    _ws_auth_lock: threading.Lock = field(default_factory=threading.Lock)
    _state_payload_cache: dict = field(default_factory=dict)
    _state_payload_lock: threading.Lock = field(default_factory=threading.Lock)
    # Admin dashboard subscribers.  Each queue is bounded so a slow browser
    # cannot retain an unbounded stream of node events or block forecasting.
    _ws_admin_subscribers: dict = field(default_factory=dict)
    _ws_admin_lock: threading.Lock = field(default_factory=threading.Lock)


def record_esg_event(state: AppState, ts: float, node: str, event_type: str,
                     detail: dict | None = None) -> None:
    """Ghi ledger kèm threshold_at_time (docs/07 §7.3)."""
    d = dict(detail) if detail else {}
    if "threshold_at_time" not in d:
        d["threshold_at_time"] = float(state.settings.get()["threshold_c"])
    state.store.insert_esg_event(ts, node, event_type, d)


def _legacy_database_paths() -> tuple[str, ...]:
    """Legacy DB chỉ được dò khi operator chưa chọn THERMAL_DATA_DIR.

    Một data directory tường minh là quyết định migration của operator. Quét lại
    DB cạnh repo lúc đó sẽ biến một lựa chọn hợp lệ thành xung đột giả và ngăn
    Host khởi động, dù code tuyệt đối không được tự hợp nhất dữ liệu lịch sử.
    """
    if (os.environ.get("THERMAL_DATA_DIR") or "").strip():
        return ()
    server_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(server_dir)
    return (
        os.path.join(repo_root, "telemetry.db"),
        os.path.join(server_dir, "telemetry.db"),
    )


def _runtime_data_path(path: str) -> str:
    """Đặt cấu hình vận hành tương đối cạnh DB khi data dir được chỉ định.

    Chỉ áp dụng với tên tệp một cấp để test/dev còn có thể truyền fixture hay
    đường dẫn tương đối riêng. Runtime installer luôn cung cấp
    ``THERMAL_DATA_DIR`` nên settings/room.json/model.pkl không còn phụ thuộc CWD.
    """
    if (os.environ.get("THERMAL_DATA_DIR") or "").strip() and (
            not os.path.isabs(path) and os.path.dirname(path) in ("", ".")):
        from data_paths import resolve_data_dir
        return os.path.join(resolve_data_dir(), path)
    return path


def make_state(db_path: str | None = None, model_path="model.pkl",
               settings_path="settings.json", esg_path="esg_config.json",
               power_model_path: str | None = None,
               weather_path: str | None = None,
               calibrate=False, demo_load=False,
               room_code=DEFAULT_ROOM_CODE,
               room_password=DEFAULT_ROOM_PASSWORD,
               admin_password=DEFAULT_ADMIN_PASSWORD,
               credentials_weak: bool = True,
               room_meta_path: str | None = None,
               bootstrap_open: bool = False):
    from rate_limit import TokenRateLimiter
    from tunnel import TunnelManager
    from room_persist import default_room_json_path, load_room_meta
    model_path = _runtime_data_path(model_path)
    if db_path is None:
        from data_paths import prepare_database_path
        db_path = prepare_database_path(legacy_paths=_legacy_database_paths())
        settings_path = _runtime_data_path(settings_path)
    store = TelemetryStore(db_path)
    if bootstrap_open:
        # Lần đầu Host: chưa mật khẩu — wizard/API bootstrap sẽ đặt.
        room = Room(room_code=room_code or "THERMAL-PENDING",
                    credentials_weak=False)
    else:
        room = make_default_room(
            room_code, room_password, admin_password,
            credentials_weak=credentials_weak)
    pm_path = power_model_path or os.path.join(
        os.path.dirname(__file__), "power_model.json")
    esg_cfg = load_esg_config(esg_path)
    pm = load_power_model(pm_path)
    from power_model import apply_leakage_to_esg_config
    apply_leakage_to_esg_config(esg_cfg, pm)
    settings = Settings(settings_path)
    meta_path = room_meta_path or default_room_json_path(settings_path)
    meta = load_room_meta(meta_path)
    if meta.get("display_name") or meta.get("name"):
        room.display_name = str(
            meta.get("display_name") or meta.get("name") or "")
    if meta.get("room_code") and bootstrap_open:
        room.room_code = str(meta["room_code"])
    # Meta password_set nhưng RAM chưa có hash (restart + bootstrap-open):
    # không tin tunnel_ready trên disk; bootstrap chỉ loopback (xem endpoint).
    if bootstrap_open and meta.get("password_set") and not room.has_password():
        log.warning(
            "[ROOM] meta password_set but RAM has no hash — "
            "bootstrap localhost only; ignore tunnel_ready on disk")
    wx_path = weather_path or os.path.join(
        os.path.dirname(__file__), "weather_local.json")
    weather = WeatherService(config_path=wx_path)
    _req_strong = (
        os.environ.get("POC_REQUIRE_STRONG_ROOM", "1") == "1")
    from room_persist import load_room_auth
    restored_room = load_room_auth(store)
    if restored_room is not None:
        room = restored_room
        bootstrap_open = False
        log.info("[ROOM] restored persistent password verifiers; tokens revoked")
    state = AppState(store=store,
                    forecaster=Forecaster(model_path),
                    balancer=LoadBalancer(),
                    esg=EsgTracker(esg_cfg),
                    settings=settings,
                    room=room,
                    forecast_cache=ForecastCache(),
                    chat=ChatTracker(),
                    calibrate=calibrate,
                    demo_load=demo_load,
                    rate_limiter=TokenRateLimiter(),
                    power_model=pm,
                    weather=weather,
                    room_meta_path=meta_path,
                    tunnel=TunnelManager(
                        local_url=f"http://127.0.0.1:{SERVER_PORT}",
                        allow_tunnel=bool(
                            settings.get().get("allow_tunnel", True)),
                        require_strong=_req_strong,
                        audit=lambda event, detail: store.insert_room_audit(
                            time.time(), "", event, detail,
                            source="tunnel")))
    return state


def _touch_tunnel_if_enabled(state: AppState, request: Request) -> None:
    """Chỉ tính hoạt động đã đi qua hostname tunnel đang READY.

    Header ``CF-Connecting-IP`` xuất hiện trong nhiều kiểu reverse proxy, nên
    không phải bằng chứng đủ để giữ tunnel sống. So sánh Host với hostname mà
    TunnelManager vừa công bố để polling LAN/local không vô tình reset idle.
    """
    if state.tunnel is None:
        return
    try:
        status = state.tunnel.status()
        public_url = status.get("public_url") or ""
        expected = (status.get("hostname")
                    or urlparse(public_url).hostname or "").lower()
        host = (request.headers.get("host") or "").split(":", 1)[0].lower()
        if status.get("state") == "ready" and expected and host == expected:
            state.tunnel.touch_activity()
    except Exception:
        pass


def detect_lan_url(port: int | None = None) -> str:
    """IP LAN của host — worker lưu cùng tunnel_url lúc /join."""
    import socket
    port = SERVER_PORT if port is None else int(port)
    ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        pass
    return f"http://{ip}:{port}"


def _local_agent_executable() -> str | None:
    """Find the packaged agent without assuming the current directory."""
    explicit = (os.environ.get("NODE_AGENT_EXE") or "").strip()
    if explicit and os.path.isfile(explicit):
        return os.path.abspath(explicit)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = (
        # Bản EXE giải nén NodeAgent thẳng vào payload/agent, khác với cây build.
        os.path.join(root, "agent", "NodeAgent.exe"),
        os.path.join(root, "agent", "bin", "Release", "net8.0-windows",
                     "NodeAgent.exe"),
        os.path.join(root, "agent", "bin", "Release", "net8.0",
                     "NodeAgent.exe"),
        os.path.join(root, "publish", "NodeAgent", "NodeAgent.exe"),
    )
    return next((path for path in candidates if os.path.isfile(path)), None)


def _local_agent_node_name() -> str:
    raw = os.environ.get("COMPUTERNAME") or "Host"
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", raw).strip("-._") or "Host"
    return f"Host-{safe}"[:64]


def _local_agent_config_path() -> str:
    """Đường dẫn DPAPI chuẩn, trùng với ``AgentConfig.ResolveConfigPath``.

    Không ghi cạnh executable: payload phiên bản có thể thay thế khi update và
    agent luôn ưu tiên config theo máy ở ProgramData.
    """
    root = (os.environ.get("THERMAL_AGENT_CONFIG_DIR") or "").strip()
    if not root:
        program_data = (os.environ.get("PROGRAMDATA") or r"C:\\ProgramData").strip()
        root = os.path.join(program_data, "ThermalOrchestrator", "agent")
    return os.path.join(root, "config.json")


def _set_local_agent_state(state: AppState, name: str, message: str,
                           node: str | None = None) -> None:
    state.local_agent = {"state": name, "message": message,
                         "node": node}
    _invalidate_state_cache(state)
    _publish_admin_event(state, {
        "type": "local_agent_state", **state.local_agent,
    })


def _write_local_agent_config(state: AppState, node: str, config_path: str,
                              worker_password: str) -> None:
    """Store the room credentials the Host worker joins with, DPAPI-protected.

    The agent binary owns the encryption, so the plaintext only crosses the pipe
    to that short-lived child: never a command line, JSON metadata, or an audit
    row.  Writing is also the only channel the Host needs for a rotation — a
    running worker reloads this file, so no secret ever goes over the network.
    """
    executable = _local_agent_executable()
    if executable is None:
        _set_local_agent_state(state, "error",
                               "Không tìm thấy NodeAgent.exe.", node)
        raise ApiError(503, "LOCAL_AGENT_MISSING",
                       "Không tìm thấy NodeAgent.exe. Hãy build/publish agent.")
    provision = json.dumps({
        "nodeName": node,
        "serverUrl": f"http://127.0.0.1:{SERVER_PORT}",
        "roomCode": state.room.room_code,
        "password": worker_password,
    })
    try:
        protected = subprocess.run(
            [executable, "--protect-config", config_path], input=provision,
            text=True, capture_output=True, timeout=20, check=False)
    except OSError as exc:
        _set_local_agent_state(state, "error",
                               "Không chạy được NodeAgent để tạo cấu hình.", node)
        raise ApiError(503, "LOCAL_AGENT_FAILED",
                       "Không chạy được NodeAgent để tạo cấu hình.") from exc
    if protected.returncode != 0:
        _set_local_agent_state(state, "error",
                               "NodeAgent từ chối cấu hình cục bộ.", node)
        raise ApiError(503, "LOCAL_AGENT_FAILED",
                       "NodeAgent từ chối cấu hình cục bộ.")


def _launch_local_agent(state: AppState, *, worker_password: str | None = None,
                        replace_process: bool = False) -> dict:
    """Provision a DPAPI config, then request an elevated NodeAgent launch.

    Password travels only through the short-lived child's standard input; it
    is never placed on a command line, in JSON metadata, or in an audit row.

    ``replace_process`` belongs to the operator's explicit retry: it also swaps
    a running worker so a build that predates config reloading cannot leave the
    Host without one.  Automatic paths never set it, because rewriting the
    config already delivers the credentials without a second UAC prompt.
    """
    if not state.room.has_password():
        raise ApiError(409, "ROOM_NOT_READY", "Phòng chưa có mật khẩu.")
    if os.environ.get("POC_NO_BACKGROUND") == "1" and state.local_agent_launcher is None:
        _set_local_agent_state(state, "missing",
                               "Test mode không khởi chạy NodeAgent.")
        return dict(state.local_agent)
    node = _local_agent_node_name()
    config_path = _local_agent_config_path()
    running = _local_agent_process_running()
    config_written = False
    if running and worker_password is not None:
        # Creating a room rotates its code and worker password, and the running
        # worker replays the previous pair until it reloads this file.  Writing
        # the config is therefore what delivers the new credentials: it needs no
        # elevation, so it works even when the Host cannot touch the elevated
        # agent at all.
        _write_local_agent_config(state, node, config_path, worker_password)
        config_written = True
        log.info("[AGENT] rewrote the protected config of the running Host "
                 "agent for the new room credentials")
        if replace_process:
            running = _stop_packaged_local_agent() == 0
    if running:
        # Retry/UAC can be clicked repeatedly while the first elevated process
        # is already alive.  A second NodeAgent would start a second
        # llama-server and make the Host contend with itself for CPU/RAM.
        message = "Host agent đã chạy; đang chờ telemetry và model readiness."
        if config_written:
            message = "Đã cập nhật cấu hình; Host agent đang tham gia phòng mới."
        _set_local_agent_state(state, "joining", message, node)
        return dict(state.local_agent)
    _set_local_agent_state(state, "starting", "Đang tạo cấu hình Host agent.",
                           node)
    if callable(state.local_agent_launcher):
        try:
            state.local_agent_launcher(state, node)
            _set_local_agent_state(state, "awaiting_uac",
                                   "Đang chờ quyền Administrator.", node)
            return dict(state.local_agent)
        except Exception:
            _set_local_agent_state(state, "error",
                                   "Không khởi chạy được NodeAgent.", node)
            raise ApiError(503, "LOCAL_AGENT_FAILED",
                           "Không khởi chạy được NodeAgent.")
    if worker_password is not None and not config_written:
        _write_local_agent_config(state, node, config_path, worker_password)
    elif worker_password is None and not os.path.isfile(config_path):
        _set_local_agent_state(state, "error",
                               "Thiếu cấu hình Host agent đã mã hóa.", node)
        raise ApiError(409, "LOCAL_AGENT_CREDENTIALS_UNAVAILABLE",
                       "Hãy tạo lại hoặc đổi mật khẩu phòng trên Host.")
    executable = _local_agent_executable()
    if executable is None:
        _set_local_agent_state(state, "error",
                               "Không tìm thấy NodeAgent.exe.", node)
        raise ApiError(503, "LOCAL_AGENT_MISSING",
                       "Không tìm thấy NodeAgent.exe. Hãy build/publish agent.")
    # ShellExecute/runas produces the UAC prompt.  It does not receive any
    # secret and uses the protected file beside the executable.
    escaped_executable = executable.replace("'", "''")
    command = f"Start-Process -Verb RunAs -FilePath '{escaped_executable}'"
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as exc:
        _set_local_agent_state(state, "error", "Không mở được yêu cầu UAC.",
                               node)
        raise ApiError(503, "LOCAL_AGENT_FAILED",
                       "Không mở được yêu cầu UAC.") from exc
    _set_local_agent_state(state, "awaiting_uac",
                           "Đã yêu cầu quyền Administrator cho NodeAgent.", node)
    return dict(state.local_agent)


def _local_agent_process_running() -> bool:
    """Best-effort guard so a Host restart never creates a second agent.

    Matching the image name alone is deliberate here: it errs towards "one is
    already running", and the cost of a false positive is a deferred start the
    dashboard can retry — never a terminated process, which is decided by
    ``_stop_packaged_local_agent`` from the image path instead.
    """
    if os.name != "nt":
        return False
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq NodeAgent.exe", "/NH"],
            text=True, capture_output=True, timeout=5, check=False)
    except OSError:
        return False
    # Test runners and restricted process queries can omit stdout; treat that as
    # "not found" instead of letting a retry endpoint fail with AttributeError.
    return "NodeAgent.exe" in (getattr(result, "stdout", "") or "")


def _local_agent_processes() -> list[tuple[int, str | None]]:
    """Every NodeAgent process with its image path when Windows discloses it.

    ``Win32_Process`` is used rather than ``Process.MainModule``: reading the
    path of an elevated agent from the unelevated Host works through the CIM
    provider, and a process whose path stays hidden is reported as ``None`` so
    callers can treat it as "not attributable" instead of guessing.
    """
    if os.name != "nt":
        return []
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name='NodeAgent.exe'\" | "
        "ForEach-Object { \"$($_.ProcessId)|$($_.ExecutablePath)\" }")
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            text=True, capture_output=True, timeout=20, check=False)
    except OSError:
        log.warning("[AGENT] could not enumerate NodeAgent processes")
        return []
    found: list[tuple[int, str | None]] = []
    for line in (getattr(result, "stdout", "") or "").splitlines():
        pid_text, _, path = line.strip().partition("|")
        if not pid_text.isdigit():
            continue
        found.append((int(pid_text), path.strip() or None))
    return found


def _stop_packaged_local_agent() -> int:
    """Stop only the NodeAgent started from this install, and its llama child.

    A NodeAgent whose image path is not this payload may be a worker joined to
    somebody else's room, so it is left running instead of being killed by image
    name.  Terminating an elevated agent also fails when the Host itself is not
    elevated; that is reported as "nothing stopped" so the caller falls back to
    the config it has already rewritten.  Returns the number of processes that
    really went away.
    """
    executable = _local_agent_executable()
    if executable is None:
        return 0
    target = os.path.normcase(os.path.abspath(executable))
    stopped = 0
    unattributed = 0
    for pid, path in _local_agent_processes():
        if path is None:
            unattributed += 1
            continue
        if os.path.normcase(os.path.abspath(path)) != target:
            continue
        try:
            killed = subprocess.run(
                ["taskkill", "/PID", str(pid), "/F", "/T"],
                capture_output=True, text=True, timeout=10, check=False)
        except OSError:
            log.warning("[AGENT] could not stop Host agent pid=%d", pid)
            continue
        if getattr(killed, "returncode", 1) != 0:
            log.warning("[AGENT] Windows refused to stop Host agent pid=%d — "
                        "keeping it and relying on the rewritten config", pid)
            continue
        stopped += 1
        log.info("[AGENT] stopped packaged Host agent pid=%d to apply the new "
                 "room credentials", pid)
    if unattributed:
        log.warning("[AGENT] left %d NodeAgent process(es) alone — no readable "
                    "image path to attribute them to this install", unattributed)
    return stopped


def _restore_local_agent(state: AppState) -> None:
    """Restart the persisted Host worker after a Host/server restart.

    The protected config is enough to start the normal outbound NodeAgent; the
    password never needs to be recovered by the server.  A running agent is
    retained and allowed to rejoin by itself instead of creating a duplicate.
    """
    if not state.room.has_password() or not os.path.isfile(_local_agent_config_path()):
        return
    node = _local_agent_node_name()
    if _local_agent_process_running():
        _set_local_agent_state(state, "joining",
                               "Host agent đang kết nối lại sau khi Host khởi động.",
                               node)
        return
    try:
        _launch_local_agent(state)
        log.info("[AGENT] restored local Host agent node=%s", node)
    except ApiError as exc:
        log.warning("[AGENT] local Host restore deferred code=%s", exc.code)


def _auth_failure_counts(state: AppState, now: float) -> tuple[int, int]:
    events = state.store.room_audit_events()
    total = 0
    hour = 0
    for ev in events:
        if ev.get("event_type") != "join_failed":
            continue
        total += 1
        ts = float(ev.get("ts") or 0)
        if now - ts <= 3600.0:
            hour += 1
    return total, hour


def refresh_weather(state: AppState, now: float) -> None:
    """Fetch thời tiết (blocking) rồi cập nhật ESG từ cache."""
    if state.weather is None:
        return
    site = (
        state.weather.config.get("default_site_id")
        or state.settings.get().get("site_id")
        or "hanoi-office-4f")
    try:
        state.weather.refresh_all(now)
        apply_weather_to_esg(
            state.weather, state.esg.config, site_id=site, now=now)
    except Exception:
        log.exception("[WEATHER] refresh failed — hệ thống vẫn chạy")
    try:
        state.store.maybe_wal_checkpoint(now)
    except Exception:
        log.exception("[STORE] WAL checkpoint failed")

def active_nodes(state, now):
    """Nodes with a sample within ACTIVE_WINDOW_S."""
    result = []
    for node in state.store.nodes():
        latest = state.store.latest(node)
        if latest is not None and (now - latest.get("ts", 0)) <= ACTIVE_WINDOW_S:
            result.append(node)
    return result


def nodes_for_payload(state, now):
    """Node active + STALE còn trong ACTIVE_WINDOW từ store + cache.

    Sau restart cache rỗng nhưng SQLite còn mẫu trong cửa sổ active → vẫn
    hiện STALE (không phụ thuộc forecast_cache). Ngoài ACTIVE_WINDOW = ghost
    (test_ghost_node) — không hiện.
    """
    names = set(active_nodes(state, now))
    for node in state.store.nodes():
        latest = state.store.latest(node)
        if latest is None:
            continue
        age = now - latest.get("ts", 0)
        # Trong cửa sổ active nhưng đã quá STALE_AFTER → vẫn liệt kê (offline)
        if 0 <= age <= ACTIVE_WINDOW_S:
            names.add(node)
    for node, fc in state.forecast_cache.all().items():
        if fc.state == "INACTIVE":
            continue
        names.add(node)
    return sorted(names)

def _base_state(samples, latest, now):
    if latest is None or (now - latest.get("ts", 0)) > STALE_AFTER_S:
        return "STALE"
    usable = [s for s in samples if s.get("cpu_temp") is not None]
    if len(usable) < MIN_SAMPLES:
        return "WARMING_UP"
    span = usable[-1]["ts"] - usable[0]["ts"]
    if span < MIN_SPAN_S:
        return "WARMING_UP"
    return "READY"

def run_forecast_cycle(state, now):
    """Predict once per node, write ForecastCache, flag/clear with hysteresis."""
    threshold = state.settings.get()["threshold_c"]
    preds = {}
    for node in active_nodes(state, now):
        prev = state.forecast_cache.get(node)
        # Node đã kick/leave — không gắn cờ lại từ mẫu cũ trong ACTIVE_WINDOW
        if prev is not None and prev.state == "INACTIVE":
            continue
        latest = state.store.latest(node) or {}
        samples = state.store.recent(node, WINDOW_S, now)
        base = _base_state(samples, latest, now)

        margin = state.settings.get().get("max_temp_margin_c", MAX_TEMP_MARGIN_C)
        if prev and prev.idle_baseline_c:
            baseline = prev.idle_baseline_c
        else:
            baseline = idle_baseline_from_samples(samples, fallback=40.0)
        eff = effective_threshold(threshold, baseline, margin)

        cur = latest.get("cpu_temp")
        pred = None
        delta = None
        if base == "READY":
            # pred = cur + ΔT — cùng số dùng để gắn cờ và hiển thị (bất biến 6)
            delta = state.forecaster.predict_delta_t(
                samples, idle_baseline=baseline)
            if delta is None or cur is None:
                base = "WARMING_UP"
                pred = None
                delta = None
            else:
                pred = cur + delta

        preds[node] = pred
        state_name, changed_at, reason = apply_hysteresis(
            prev, base_state=base, pred=pred,
            threshold_c=eff, now=now)

        was_flagged = state.balancer.is_flagged(node)
        pending_event = None
        if state_name == "STALE" and was_flagged:
            log.info("[FORECAST] %s: no fresh data (stale) -> clearing "
                     "stuck flag", node)
            state.balancer.set_flag(node, False)
            state.esg.node_cleared(node, now)
            reason = "stale clear stuck flag"
            record_esg_event(
                state, now, node, "cleared",
                {"reason": "stale", "pred": pred})
            changed_at = now
            pending_event = {
                "event": "cleared", "state_name": "STALE",
                "reason": reason, "predicted_max": pred,
                "decision_threshold_c": eff,
            }
        if state_name == "STALE":
            # Mirror leave/kick: nhả reservation ngay, không đợi reap ~20s.
            released = state.balancer.release_reservations_for(node)
            for jid in released:
                state.forecast_cache.adjust_inflight(node, -1)
                state.chat.mark_queued(jid, now=now)
            if released:
                log.info("[SCHED] released %d reservations — node %s STALE",
                         len(released), node)
            _release_active_on_stale(state, node, now)

        becoming_risk = state_name == "AT_RISK" and not was_flagged
        leaving_risk = (state_name != "AT_RISK" and was_flagged
                        and state_name != "STALE")
        if becoming_risk:
            log.info("[FORECAST] %s: predicted max %.1f°C in next 3min >= "
                     "eff_threshold %.1f (ΔT) -> flagged (%s)",
                     node, pred if pred is not None else -1, eff, reason)
            state.balancer.set_flag(node, True)
            state.esg.node_flagged(node, now)
            record_esg_event(
                state, now, node, "flagged",
                {"pred": pred, "threshold": eff, "reason": reason,
                 "predicted_max": pred})
            pending_event = {
                "event": "flagged", "state_name": state_name,
                "reason": reason, "predicted_max": pred,
                "decision_threshold_c": eff,
            }
        elif leaving_risk:
            log.info("[FORECAST] %s: predicted max %s < clear band -> cleared "
                     "(%s)", node,
                     f"{pred:.1f}°C" if pred is not None else "n/a", reason)
            state.balancer.set_flag(node, False)
            state.esg.node_cleared(node, now)
            record_esg_event(
                state, now, node, "cleared",
                {"pred": pred, "threshold": eff, "reason": reason,
                 "predicted_max": pred})
            pending_event = {
                "event": "cleared", "state_name": state_name,
                "reason": reason, "predicted_max": pred,
                "decision_threshold_c": eff,
            }

        flagged_since = None
        if state_name == "AT_RISK":
            flagged_since = (prev.flagged_since if prev and prev.flagged_since
                             else now)
            if becoming_risk:
                flagged_since = now

        temp_for_hr = pred if pred is not None else cur
        headroom = None
        if temp_for_hr is not None:
            span = max(eff - baseline, 1.0)
            headroom = max(0.0, min(1.0, (eff - temp_for_hr) / span))

        nf = NodeForecast(
            node=node,
            state=state_name,
            predicted_max_c=pred,
            delta_t_c=delta,
            current_temp_c=cur,
            headroom=headroom,
            cpu_util=latest.get("cpu_util"),
            power_w=latest.get("power_w"),
            power_source=(
                "none" if latest.get("power_w") is None
                else (prev.power_source if prev and prev.power_source
                      in ("sensor", "model") else "sensor")),
            inflight=prev.inflight if prev else 0,
            user_active=prev.user_active if prev else False,
            max_concurrent=prev.max_concurrent if prev else 1,
            model_ready=False if prev is None else prev.model_ready,
            model_id=prev.model_id if prev else None,
            model_sha256=prev.model_sha256 if prev else None,
            model_generation=(prev.model_generation if prev else
                              state.llm_generation),
            runtime_id=prev.runtime_id if prev else None,
            last_sample_ts=latest.get("ts", now),
            idle_baseline_c=baseline,
            effective_threshold_c=eff,
            p_idle_w=prev.p_idle_w if prev else 15.0,
            p_max_w=prev.p_max_w if prev else 65.0,
            recently_failed_until=(
                prev.recently_failed_until if prev else 0.0),
            consecutive_errors=prev.consecutive_errors if prev else 0,
            flagged_since=flagged_since,
            state_changed_at=changed_at,
            computed_at=now,
            reason=reason,
        )
        state.forecast_cache.put(nf)
        _invalidate_state_cache(state)
        if pending_event is not None:
            _publish_node_event(state, node, **pending_event)
    return preds


def _release_active(state, node, now, *, code: str, reason: str):
    """Rời active: chat requeue (hoặc fail nếu hết attempts); burn cancel.

    Chỉ giảm inflight khi release/cancel thành công (tránh đếm đôi).
    """
    peek = state.balancer.peek_active(node)
    if peek is None:
        return None
    jtype = peek.get("type") or "burn"
    attempts = int(peek.get("attempts") or 0)
    display = "host" if node == HOST_NODE else node

    if jtype == "chat":
        if attempts >= MAX_ATTEMPTS:
            job = state.balancer.cancel_active(node, code=code)
            if job is None:
                return None
            if node != HOST_NODE:
                state.forecast_cache.adjust_inflight(node, -1)
            state.chat.mark_error(
                job["id"], message=code, node=display, now=now)
            state.balancer.clear_prompt(job)
            log.info("[SCHED] active %s chat %s on %s → fail "
                     "(attempts %d/%d)",
                     reason, job["id"], node, attempts, MAX_ATTEMPTS)
            return job
        job = state.balancer.requeue_active_as_pending(node)
        if job is None:
            return None
        if node != HOST_NODE:
            state.forecast_cache.adjust_inflight(node, -1)
        pos = state.balancer.queue_position(job["id"]) or 1
        state.chat.mark_queued(
            job["id"], queue_position=pos,
            estimated_wait_s=float(pos) * 8.0, now=now)
        log.info("[SCHED] active %s chat %s on %s → requeue "
                 "(attempts %d/%d)",
                 reason, job["id"], node, attempts, MAX_ATTEMPTS)
        return job

    job = state.balancer.cancel_active(node, code=code)
    if job is None:
        return None
    if node != HOST_NODE:
        state.forecast_cache.adjust_inflight(node, -1)
    log.info("[SCHED] active %s burn %s on %s → cancel",
             reason, job["id"], node)
    return job


def _release_expired_active(state, now):
    """Active lease hết hạn: chat → requeue hoặc fail; burn → cancel."""
    for node, job in state.balancer.list_active():
        deadline = job.get("active_deadline")
        if deadline is None or now <= deadline:
            continue
        _release_active(state, node, now, code="LEASE_EXPIRED",
                        reason="lease hết hạn")


def _release_active_on_stale(state, node, now):
    """Node STALE còn job active — chat requeue/fail, burn cancel."""
    _release_active(state, node, now, code="NODE_STALE", reason="STALE")


class _ChatOnlyQueue:
    # Outside the run window: hand out chat only; burn jobs wait.
    def __init__(self, queue):
        self._queue = queue

    def pending(self):
        return [j for j in self._queue.pending()
                if (j.get("type") or "burn") == "chat"]

    def __getattr__(self, name):
        return getattr(self._queue, name)


def run_scheduler_cycle(state, now):
    """Thu hồi giữ chỗ hết hạn rồi gán job. Inflight qua ForecastCache API."""
    cfg = state.settings.get()
    live_before = ({j["id"] for j in state.balancer.pending()}
                   | {j["id"] for j in state.balancer.reserved()})

    reap_expired(state.forecast_cache, state.balancer, now)
    _release_expired_active(state, now)
    # Admin "Stop all jobs": hand out nothing.
    if not cfg.get("jobs_paused"):
        lt = time.localtime(now)
        minute = lt.tm_hour * 60 + lt.tm_min
        queue = state.balancer
        if not in_run_window(cfg.get("run_window_start", ""),
                             cfg.get("run_window_end", ""), minute):
            queue = _ChatOnlyQueue(state.balancer)
        schedule_once(
            state.forecast_cache, queue, state.settings.weights(), now,
            mode=cfg.get("scheduler_mode", "thermal_aware"),
            reservation_timeout_s=cfg.get("reservation_timeout_s", 20.0),
            rr_cursor=state.rr_cursor,
        )
    # Chat jobs that disappeared into fail()
    for jid in live_before:
        failed = state.balancer.get_failed(jid)
        if failed is not None:
            state.chat.mark_error(
                jid, message=failed.get("detail") or failed.get("code")
                or "NO_CAPACITY", now=now)


def _finite_nonneg(name, val, *, allow_none=True):
    if val is None:
        if allow_none:
            return None
        raise ApiError(422, "BAD_REQUEST", f"{name} bắt buộc.")
    if isinstance(val, bool):
        raise ApiError(422, "BAD_REQUEST", f"{name} không hợp lệ.")
    try:
        x = float(val)
    except (TypeError, ValueError) as e:
        raise ApiError(422, "BAD_REQUEST",
                       f"{name} không hợp lệ.") from e
    if not math.isfinite(x) or x < 0:
        raise ApiError(422, "BAD_REQUEST",
                       f"{name} phải là số hữu hạn ≥ 0.")
    return x


def _finite_optional(name, val, *, lo=None, hi=None):
    """None ok; từ chối NaN/Inf; tùy chọn khoảng [lo, hi]."""
    if val is None:
        return None
    if isinstance(val, bool):
        raise ApiError(422, "BAD_REQUEST", f"{name} không hợp lệ.")
    try:
        x = float(val)
    except (TypeError, ValueError) as e:
        raise ApiError(422, "BAD_REQUEST",
                       f"{name} không hợp lệ.") from e
    if not math.isfinite(x):
        raise ApiError(422, "BAD_REQUEST",
                       f"{name} phải là số hữu hạn.")
    if lo is not None and x < lo:
        raise ApiError(422, "BAD_REQUEST",
                       f"{name} phải ≥ {lo}.")
    if hi is not None and x > hi:
        raise ApiError(422, "BAD_REQUEST",
                       f"{name} phải ≤ {hi}.")
    return x


# Giới hạn độ dài (bảo vệ RAM/LLM). Cần duyệt cập nhật docs/03 nếu ship.
CHAT_PROMPT_MAX_CHARS = 32_768
JOB_RESULT_TEXT_MAX_CHARS = 65_536


def _room_mut_begin(state, auth, *, now: float) -> None:
    """Seam + revalidate — gọi bên trong mutation_section."""
    state.lifecycle._run_before_room_mut_hook()
    state.lifecycle.require_alive(state.room, auth, now=now)


def apply_job_result(state, node: str, body_or_dict, now: float):
    """Hoàn tất job (worker hoặc host P2). Trả về dict phản hồi HTTP."""
    if hasattr(body_or_dict, "model_dump"):
        data = body_or_dict.model_dump()
    else:
        data = dict(body_or_dict)
    job_id = data["job_id"]
    status = data.get("status") or "ok"
    if status not in ("ok", "error", "timeout", "preempted"):
        raise ApiError(422, "BAD_REQUEST", "status phải là ok|error|timeout.")
    energy_source = data.get("energy_source") or "none"
    energy_j = data.get("energy_j")
    if energy_source not in ("sensor", "model", "none"):
        raise ApiError(422, "BAD_REQUEST",
                       "energy_source phải là sensor|model|none.")
    if energy_source in ("sensor", "model") and energy_j is None:
        raise ApiError(422, "BAD_REQUEST",
                       f"energy_source={energy_source} đòi hỏi energy_j.")
    energy_j = _finite_nonneg("energy_j", energy_j, allow_none=True)
    tokens_out = data.get("tokens_out")
    tokens_in = data.get("tokens_in")
    if tokens_out is not None:
        tokens_out = _finite_nonneg("tokens_out", tokens_out)
    if tokens_in is not None:
        tokens_in = _finite_nonneg("tokens_in", tokens_in)
    duration_ms = _finite_nonneg("duration_ms", data.get("duration_ms"),
                                 allow_none=True)
    prompt_eval_ms = _finite_nonneg(
        "prompt_eval_ms", data.get("prompt_eval_ms"), allow_none=True)
    peak = _finite_optional("peak_temp_c", data.get("peak_temp_c"),
                            lo=-50.0, hi=150.0)
    min_clock_mhz = _finite_optional(
        "min_clock_mhz", data.get("min_clock_mhz"), lo=0.0, hi=20_000.0)
    cpu_util = _finite_optional("cpu_util", data.get("cpu_util"),
                                lo=0.0, hi=100.0)
    fan_rpm = _finite_optional("fan_rpm", data.get("fan_rpm"),
                               lo=0.0, hi=100_000.0)
    fan_rpm_hot = _finite_optional(
        "fan_rpm_hot", data.get("fan_rpm_hot"), lo=0.0, hi=100_000.0)
    fan_rpm_cool = _finite_optional(
        "fan_rpm_cool", data.get("fan_rpm_cool"), lo=0.0, hi=100_000.0)
    base_clock_mhz = _finite_optional(
        "base_clock_mhz", data.get("base_clock_mhz"), lo=0.0, hi=20_000.0)
    active = state.balancer.peek_active(node)
    if active is not None and active.get("id") == job_id:
        expected_attempt = active.get("attempt_id")
        received_attempt = data.get("attempt_id")
        # Transitional workers may omit the value for their first claim only.
        if (expected_attempt and received_attempt != expected_attempt
                and not (received_attempt is None
                         and int(active.get("attempts") or 0) <= 1)):
            raise ApiError(409, "STALE_ATTEMPT",
                           "Kết quả thuộc lần chạy đã hết hiệu lực.")
    done = state.balancer.complete(node, job_id)
    if done is None:
        log.info("[SCHED] bỏ qua kết quả muộn job %s từ %s", job_id, node)
        return {"ok": True, "ignored": True}
    # Seam: kích close ngay sau complete — commit ESG/chat phải abort nếu writer chờ
    after_c = getattr(state.lifecycle, "_after_complete_hook", None)
    if after_c is not None:
        state.lifecycle._after_complete_hook = None
        after_c()
    mode = (done.get("scheduler_mode")
            or state.settings.get().get("scheduler_mode", "thermal_aware"))
    err_msg = data.get("error_message")
    if err_msg is not None:
        err_msg = str(err_msg)[:512]
    display_node = "host" if node == HOST_NODE else node
    assign_temp = done.get("assign_temp_c")
    avoided_temp = done.get("avoided_temp_c")
    # docs/07: ΔT_tránh = nhiệt_node_bị_né − nhiệt_node_được_chọn lúc gán
    # (không dùng peak lúc chạy).
    delta_avoided = None
    if avoided_temp is not None and assign_temp is not None:
        try:
            delta_avoided = max(0.0, float(avoided_temp) - float(assign_temp))
        except (TypeError, ValueError):
            delta_avoided = None
    duration_h = None
    if duration_ms is not None:
        duration_h = duration_ms / 3_600_000.0
    detail = {
        "job_id": job_id,
        "status": status,
        "scheduler_mode": mode,
        "energy_j": energy_j,
        "energy_source": energy_source,
        "tokens_out": tokens_out,
        "tokens_in": tokens_in,
        "tokens_estimated": (
            bool(data.get("tokens_estimated")) or tokens_out is None),
        "peak_temp_c": peak,
        "min_clock_mhz": min_clock_mhz,
        "cpu_util": cpu_util,
        "fan_rpm": fan_rpm,
        "fan_rpm_hot": fan_rpm_hot,
        "fan_rpm_cool": fan_rpm_cool,
        "base_clock_mhz": base_clock_mhz,
        "prompt_eval_ms": prompt_eval_ms,
        "duration_ms": duration_ms,
        "duration_h": duration_h,
        "assign_temp_c": assign_temp,
        "avoided_temp_c": avoided_temp,
        "delta_t_avoided_c": delta_avoided,
    }
    job_type = done.get("type") or "burn"
    attempts = done.get("attempts", 0)

    # Giữ condition lock qua toàn bộ side-effect: writer không thể xen giữa
    # check và mark chat/ghi ESG; writer đã chờ thì abort trước khi ghi.
    with state.lifecycle.commit_section():
        return _finalize_job_result(
            state, job_id=job_id, node=node, data=data, now=now, done=done,
            status=status,
            detail=detail, job_type=job_type, attempts=attempts,
            tokens_in=tokens_in, tokens_out=tokens_out, duration_ms=duration_ms,
            display_node=display_node, err_msg=err_msg, mode=mode)


def _finalize_job_result(state, *, job_id, node, data, now, done, status, detail,
                         job_type, attempts, tokens_in, tokens_out,
                         duration_ms, display_node, err_msg, mode):
    """Ghi side-effect của result trong ``LifecycleGate.commit_section``."""
    if node != HOST_NODE:
        state.forecast_cache.adjust_inflight(node, -1)

    if status == "preempted":
        # Step 4b: a person came back to this PC. Not the PC's fault:
        # no error count, no used attempt, no ESG "job_completed".
        if node != HOST_NODE:
            # Telemetry comes every ~2 s; mark the PC busy now so the
            # scheduler does not hand the same job straight back.
            state.forecast_cache.patch(node, user_active=True)
        done["attempts"] = max(0, attempts - 1)
        state.balancer.return_job_to_pending(done)
        if job_type == "chat":
            pos = state.balancer.queue_position(job_id) or 1
            state.chat.mark_queued(
                job_id, queue_position=pos,
                estimated_wait_s=float(pos) * 8.0, now=now)
        record_esg_event(state, now, display_node, "job_preempted", detail)
        log.info("[SCHED] job %s preempted on %s (user active) -> requeued",
                 job_id, node)
        return {"ok": True, "requeued": True}

    if status != "ok":
        detail["error_message"] = err_msg
        if node != HOST_NODE:
            fc = state.forecast_cache.get(node)
            errs = (fc.consecutive_errors + 1) if fc else 1
            state.forecast_cache.patch(node, consecutive_errors=errs)
        # Chat: retry kể cả host P2 khi còn attempts (docs/03)
        if job_type == "chat" and attempts < MAX_ATTEMPTS:
            failed_nodes = set(done.get("failed_nodes") or ())
            if node != HOST_NODE:
                failed_nodes.add(node)
            done["failed_nodes"] = sorted(failed_nodes)
            state.balancer.return_job_to_pending(done)
            pos = state.balancer.queue_position(job_id) or 1
            state.chat.mark_queued(
                job_id, queue_position=pos,
                estimated_wait_s=float(pos) * 8.0, now=now)
            log.info("[SCHED] chat %s retry after %s on %s (attempt %d/%d)",
                     job_id, status, node, attempts, MAX_ATTEMPTS)
            record_esg_event(state, now, display_node, "job_retry", detail)
            return {"ok": True, "retried": True}
        state.chat.mark_error(job_id, message=err_msg or status,
                              node=display_node, now=now)
        state.balancer.clear_prompt(done)
        _publish_admin_event(state, {
            "type": "chat_result", "job_id": job_id,
            "attempt_id": done.get("attempt_id"), "status": "error",
            "node": display_node, "error_message": err_msg or status,
        })
    else:
        if node != HOST_NODE:
            state.forecast_cache.patch(node, consecutive_errors=0)
        state.chat.mark_done(
            job_id, node=display_node,
            text=(data.get("text") or (state.chat.get(job_id) or {}).get(
                "partial_text", "")),
            tokens_out=tokens_out,
            tokens_in=tokens_in,
            duration_ms=duration_ms, now=now)
        state.balancer.clear_prompt(done)
        _publish_admin_event(state, {
            "type": "chat_result", "job_id": job_id,
            "attempt_id": done.get("attempt_id"), "status": "done",
            "node": display_node,
            "text": data.get("text") or (state.chat.get(job_id) or {}).get(
                "text", ""),
            "duration_ms": duration_ms,
        })
    record_esg_event(state, now, display_node, "job_completed", detail)
    log.info("[SCHED] job %s completed on %s status=%s mode=%s tokens_out=%s",
             job_id, node, status, mode, data.get("tokens_out"))
    return {"ok": True}


def _job_payload_for_worker(job: dict) -> dict:
    """Hình dạng hợp đồng GET /jobs/next — không lộ field nội bộ thừa."""
    jtype = job.get("type") or "burn"
    if jtype == "chat":
        return {
            "id": job["id"],
            "type": "chat",
            "prompt": job.get("prompt") or "",
            "params": job.get("params") or {
                "max_tokens": 512, "temperature": 0.7},
            "deadline_s": job.get("deadline_s", 60),
            "reserved_until": job.get("reserved_until"),
            "attempt_id": job.get("attempt_id"),
        }
    return {
        "id": job["id"],
        "type": "burn",
        "duration_s": job.get("duration_s", 10),
        "cores": job.get("cores", 0),
    }


def _release_node_work(state, node, now=None):
    """Kick/leave: trả reservation; chat active → requeue; burn → hủy."""
    now = now if now is not None else time.time()
    released = state.balancer.release_reservations_for(node)
    for jid in released:
        state.forecast_cache.adjust_inflight(node, -1)
        state.chat.mark_queued(jid, now=now)
    cancelled_id = None
    # Kick/leave: luôn requeue chat (không fail theo attempts — user chủ động)
    peek = state.balancer.peek_active(node)
    if peek is not None:
        if (peek.get("type") or "burn") == "chat":
            job = state.balancer.requeue_active_as_pending(node)
            if job is not None:
                state.forecast_cache.adjust_inflight(node, -1)
                cancelled_id = job["id"]
                pos = state.balancer.queue_position(job["id"]) or 1
                state.chat.mark_queued(
                    job["id"], queue_position=pos,
                    estimated_wait_s=float(pos) * 8.0, now=now)
                log.info("[SCHED] chat %s requeued after leave/kick %s",
                         job["id"], node)
        else:
            cancelled = state.balancer.cancel_active(node)
            if cancelled is not None:
                state.forecast_cache.adjust_inflight(node, -1)
                cancelled_id = cancelled["id"]
                log.info("[SCHED] cancelled active job %s on %s — không requeue",
                         cancelled["id"], node)
    state.balancer.set_flag(node, False)
    state.forecast_cache.mark_inactive(node, now, reason="node gone")
    return {"released": released, "cancelled": cancelled_id}


def _purge_room_work(state, *, reason: str, now=None):
    """Đóng/đổi phòng: hủy mọi job + chat — không để rò sang phòng mới.

    Gọi TRONG lifecycle gate (sau set_accepting False). Không log prompt/token.
    """
    now = now if now is not None else time.time()
    # Hạ inflight trước khi drain (mọi node còn active/reserved)
    active = {n for n, _ in state.balancer.list_active()}
    reserved_nodes = {j["target"] for j in state.balancer.reserved()
                      if j.get("target")}
    for node in active | reserved_nodes:
        try:
            state.forecast_cache.adjust_inflight(node, -1)
        except Exception:
            pass
        state.forecast_cache.mark_inactive(node, now, reason=reason)
    n_jobs = len(state.balancer.drain_all(code="ROOM_CLOSED"))
    n_chat = state.chat.fail_all(
        f"Phòng đã đóng ({reason})", now=now)
    # Long-poll cũ không giữ slot tên node sang phòng mới
    state._job_polls.clear()
    _invalidate_state_cache(state)
    log.info("[ROOM] purge reason=%s jobs=%d chat=%d",
             reason, n_jobs, n_chat)
    log.info("[SCHED] room purge — cancelled jobs=%d chat=%d reason=%s",
             n_jobs, n_chat, reason)
    return {"jobs": n_jobs, "chat": n_chat}


def _set_work_accepting(state, accepting: bool) -> None:
    state.balancer.set_accepting(accepting)
    state.chat.set_accepting(accepting)


def _run_room_lifecycle(state, *, reason: str, finish_fn, now: float):
    """Gate chung: block → purge → [after_drain hook] → revoke/bump → mở."""
    def purge():
        _purge_room_work(state, reason=reason, now=now)

    state.lifecycle.run(
        purge_fn=purge,
        finish_fn=finish_fn,
        set_accepting=lambda a: _set_work_accepting(state, a),
    )
    _invalidate_state_cache(state)

def _selected_model_id(state) -> str:
    """Catalog model_id từ settings; id lạ → default + cảnh báo (không 500)."""
    raw = state.settings.get().get("model_id")
    try:
        return get_model(raw if raw is not None else None)["model_id"]
    except UnknownModelError:
        log.warning(
            "[LLM] settings model_id invalid %r → default %s",
            raw, DEFAULT_MODEL_ID)
        return DEFAULT_MODEL_ID


def build_state_payload(state, now, *, compact=False):
    """Read-only view of ForecastCache. Never re-predict. compact=worker view."""
    stats = state.balancer.stats()
    cache = state.forecast_cache.all()
    nodes = []
    for node in nodes_for_payload(state, now):
        latest = state.store.latest(node) or {}
        fc = cache.get(node)
        is_stale = (
            fc is not None and fc.state == "STALE"
        ) or (
            fc is None
            and (now - latest.get("ts", 0)) > STALE_AFTER_S
        )
        if compact:
            cpu = None if is_stale else (
                fc.current_temp_c if fc and fc.current_temp_c is not None
                else latest.get("cpu_temp"))
            nodes.append({
                "name": node,
                "state": fc.state if fc else (
                    "STALE" if is_stale else "WARMING_UP"),
                "cpu_temp": cpu,
            })
            continue
        if fc is None:
            nodes.append({
                "name": node,
                "cpu_temp": None if is_stale else latest.get("cpu_temp"),
                "gpu_temp": None if is_stale else latest.get("gpu_temp"),
                "cpu_util": None if is_stale else latest.get("cpu_util"),
                "power_w": None if is_stale else latest.get("power_w"),
                "predicted_max": None,
                "decision_threshold_c": state.settings.get()["threshold_c"],
                "flagged": False,
                "state": "STALE" if is_stale else "WARMING_UP",
                "reason": ("no fresh telemetry" if is_stale
                           else "awaiting forecast cycle"),
                "dispatched": stats["dispatched"].get(node, 0),
                "stale": is_stale,
                "live": not is_stale,
            })
            continue
        if is_stale or fc.state == "STALE":
            nodes.append({
                "name": node,
                "cpu_temp": None,
                "gpu_temp": None,
                "cpu_util": None,
                "power_w": None,
                "predicted_max": None,
                "decision_threshold_c": fc.effective_threshold_c,
                "delta_t": None,
                "flagged": False,
                "state": "STALE",
                "reason": fc.reason or "no fresh telemetry",
                "dispatched": stats["dispatched"].get(node, 0),
                "stale": True,
                "live": False,
            })
            continue
        nodes.append({
            "name": node,
            "cpu_temp": fc.current_temp_c if fc.current_temp_c is not None
                        else latest.get("cpu_temp"),
            "gpu_temp": latest.get("gpu_temp"),
            "cpu_util": fc.cpu_util if fc.cpu_util is not None
                        else latest.get("cpu_util"),
            "power_w": fc.power_w if fc.power_w is not None
                       else latest.get("power_w"),
            "predicted_max": fc.predicted_max_c,
            "decision_threshold_c": fc.effective_threshold_c,
            "delta_t": fc.delta_t_c,
            "flagged": fc.state == "AT_RISK",
            "user_active": fc.user_active,
            "state": fc.state,
            "reason": fc.reason,
            "dispatched": stats["dispatched"].get(node, 0),
            "stale": False,
            "live": True,
        })
    s = state.settings.get()
    mid = _selected_model_id(state)
    mdisp = get_model(mid)["display"]
    forecast_source = (
        "ml" if state.forecaster.model_loaded else "linear_fallback")
    payload = {"threshold_c": s["threshold_c"],
               "model_loaded": state.forecaster.model_loaded,
               "forecast_source": forecast_source,
               "llm_model_id": mid,
               "queue_len": stats["queue_len"],
               "jobs_paused": bool(s.get("jobs_paused")),
               "run_window_start": s.get("run_window_start", ""),
               "run_window_end": s.get("run_window_end", ""),
               "nodes": nodes}
    if not compact:
        payload["llm_model_display"] = mdisp
        ready_nodes = _matching_ready_nodes(state)
        payload["llm_readiness"] = {
            "selected_model_id": mid,
            "generation": state.llm_generation,
            "ready_nodes": ready_nodes,
            "ready": bool(ready_nodes),
        }
        payload["local_agent"] = dict(state.local_agent)
        events = state.store.esg_events()
        payload["esg"] = compute_report(events, state.esg.config, now)
        payload["room"] = _room_public_payload(state, now)
        payload["queue"] = state.chat.counts()
        if state.tunnel is not None:
            payload["tunnel"] = state.tunnel.status()
        total_af, hour_af = _auth_failure_counts(state, now)
        payload["auth_failures_total"] = total_af
        payload["auth_failures_1h"] = hour_af
        payload["allow_tunnel"] = bool(
            state.settings.get().get("allow_tunnel", True))
        if state.weather is not None:
            try:
                payload["weather"] = state.weather.snapshot(now)
            except Exception:
                payload["weather"] = {}
        # Phân biệt rõ với nhiệt CPU trên node cards
        payload["site_id"] = state.settings.get().get(
            "site_id", "hanoi-office-4f")
        payload["roles_hint"] = {
            "web_only": (
                "Chỉ mở web trên Host = quản trị + gửi câu hỏi; "
                "câu trả lời chạy trên worker đang online hoặc LLM local host."
            ),
            "share_compute": (
                "Muốn máy này đóng góp tính toán = cài/chạy NodeAgent "
                "(quyền Admin lúc chạy để đọc cảm biến)."
            ),
            "host_room": (
                "Muốn làm chủ phòng = chế độ Host (server), "
                "không cần Room Directory."
            ),
        }
    return payload


def _invalidate_state_cache(state: AppState) -> None:
    with state._state_payload_lock:
        state._state_payload_cache.clear()


def _publish_node_event(state: AppState, node: str, event: str, *,
                        state_name: str, reason: str | None = None,
                        predicted_max: float | None = None,
                        decision_threshold_c: float | None = None) -> None:
    """Đẩy sự kiện node tức thời tới các dashboard admin đang mở.

    Forecast chạy đồng bộ trong thread hoặc trong event loop; ``call_soon``
    giúp mọi thao tác với ``asyncio.Queue`` diễn ra đúng loop của trình duyệt.
    Hàng đợi có giới hạn và bỏ sự kiện cũ nhất khi dashboard bị nghẽn.
    """
    payload = {
        "type": "node_event",
        "node": node,
        "event": event,
        "state": state_name,
        "reason": reason or "",
        "predicted_max": predicted_max,
        "decision_threshold_c": decision_threshold_c,
    }

    _publish_admin_event(state, payload)


def _publish_admin_event(state: AppState, payload: dict) -> None:
    """Publish a bounded, already-sanitized event to admin dashboards."""

    def enqueue(queue):
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    with state._ws_admin_lock:
        subscribers = list(state._ws_admin_subscribers.items())
    stale = []
    for ident, (loop, queue) in subscribers:
        try:
            loop.call_soon_threadsafe(enqueue, queue)
        except RuntimeError:
            stale.append(ident)
    if stale:
        with state._ws_admin_lock:
            for ident in stale:
                state._ws_admin_subscribers.pop(ident, None)


def _subscribe_admin_events(state: AppState):
    """Tạo subscription (loop, queue), trả về id để hủy khi WS đóng."""
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue(maxsize=WS_ADMIN_EVENT_QUEUE_MAX)
    ident = id(queue)
    with state._ws_admin_lock:
        state._ws_admin_subscribers[ident] = (loop, queue)
    return ident, queue


def _unsubscribe_admin_events(state: AppState, ident: int) -> None:
    with state._ws_admin_lock:
        state._ws_admin_subscribers.pop(ident, None)


def _cached_state_payload(state, now, *, compact=False):
    """Cache TTL 1s — dùng chung WS và /api/state."""
    key = "compact" if compact else "full"
    with state._state_payload_lock:
        entry = state._state_payload_cache.get(key)
        if entry is not None and (now - entry[0]) < STATE_PAYLOAD_TTL_S:
            return entry[1]
    payload = build_state_payload(state, now, compact=compact)
    with state._state_payload_lock:
        state._state_payload_cache[key] = (now, payload)
    return payload


def _csv_safe_cell(value) -> str:
    """Chống formula injection khi mở bằng Excel."""
    if value is None:
        return ""
    s = str(value)
    if s and s[0] in ("=", "+", "-", "@"):
        return "'" + s
    return s


def _invite_urls(state: AppState) -> dict:
    """Link mời LAN (+ tunnel nếu có) — cùng số dùng để quyết định."""
    lan = detect_lan_url()
    code = state.room.room_code
    invite_lan = f"{lan}/join?code={code}"
    worker_setup_lan = f"{lan}/worker-setup?code={quote(code)}"
    invite_tunnel = None
    worker_setup_tunnel = None
    tunnel_url = None
    if state.tunnel is not None:
        st = state.tunnel.status()
        tunnel_url = st.get("public_url")
        if tunnel_url and state.tunnel.is_ready():
            invite_tunnel = f"{tunnel_url.rstrip('/')}/join?code={code}"
            worker_setup_tunnel = (
                f"{tunnel_url.rstrip('/')}/worker-setup?code={quote(code)}")
    return {
        "lan_url": lan,
        "tunnel_url": tunnel_url,
        "invite_lan": invite_lan,
        "invite_tunnel": invite_tunnel,
        "worker_setup_lan": worker_setup_lan,
        "worker_setup_tunnel": worker_setup_tunnel,
    }


def _room_public_payload(state: AppState, now: float) -> dict:
    inv = _invite_urls(state)
    return {
        "code": state.room.room_code,
        "display_name": state.room.display_name or "",
        "worker_count": state.room.worker_count(now),
        "capacity": state.room.capacity,
        "password_set": state.room.has_password(),
        "tunnel_ready": state.room.passwords_meet_tunnel_min(12),
        "credentials_weak": bool(state.room.credentials_weak),
        "bootstrap_open": not state.room.has_password(),
        **inv,
    }


def _persist_room_meta(state: AppState) -> None:
    from room_persist import save_room_auth, save_room_meta
    # The SQLite verifier is authoritative across a Host restart; the JSON
    # sidecar remains metadata-only so it cannot authenticate anyone.
    save_room_auth(state.store, state.room)
    path = state.room_meta_path
    if not path:
        return
    s = state.settings.get()
    inv = _invite_urls(state)
    save_room_meta(path, {
        "role": "host",
        "name": state.room.display_name,
        "display_name": state.room.display_name,
        "room_code": state.room.room_code,
        "site_id": s.get("site_id"),
        "model_id": s.get("model_id"),
        "threshold_c": s.get("threshold_c"),
        "password_set": state.room.has_password(),
        "tunnel_ready": state.room.passwords_meet_tunnel_min(12),
        "invite": inv.get("invite_lan"),
    })


_bootstrap_lock = threading.Lock()


def _is_loopback(ip: str) -> bool:
    # "testclient" = Starlette TestClient (pytest); không phải IP mạng thật.
    return ip in ("127.0.0.1", "::1", "localhost",
                  "::ffff:127.0.0.1", "testclient")


def _apply_room_bootstrap(state: AppState, body,
                          *, store_audit=None, ip: str = "") -> dict:
    """Đặt/đổi phòng Host — mật khẩu chỉ hash trong RAM, meta ra disk."""
    from room_assets import MODEL_CATALOG, DEFAULT_MODEL_ID
    wp = (body.worker_password or "").strip()
    ap = (body.admin_password or "").strip()
    if len(wp) < PASSWORD_MIN_LENGTH or len(ap) < PASSWORD_MIN_LENGTH:
        raise ApiError(400, "BAD_REQUEST",
                       "Mật khẩu worker và admin phải ≥12 ký tự Unicode.")
    if wp == ap:
        raise ApiError(400, "BAD_REQUEST",
                       "Mật khẩu worker và admin phải khác nhau.")
    # TOCTOU: một writer; đổi mật khẩu → thu hồi token cũ
    if not _bootstrap_lock.acquire(blocking=False):
        raise ApiError(409, "CONFLICT",
                       "Đang có thao tác tạo/đổi phòng khác. Thử lại.")
    try:
        had = state.room.has_password()
        if had:
            now = time.time()
            _run_room_lifecycle(
                state, reason="password_rotate", now=now,
                finish_fn=lambda: state.room.lifecycle_rotate_passwords(
                    wp, ap, now=now, store_audit=store_audit, ip=ip))
            log.info("[ROOM] đổi mật khẩu — thu hồi token (lifecycle)")
            _invalidate_state_cache(state)
        name = (body.display_name or body.name or "").strip()
        if name:
            state.room.display_name = name
        code = (body.room_code or "").strip().upper()
        if code and code.startswith("THERMAL-") and len(code) >= 10:
            state.room.room_code = code
        elif (not state.room.room_code
              or state.room.room_code in ("THERMAL-LOCAL", "THERMAL-PENDING")
              or body.regenerate_code):
            state.room.regenerate_code()
        if not had:
            state.room.set_worker_password(wp)
            state.room.set_admin_password(ap)
        state.room.credentials_weak = False
        _invalidate_state_cache(state)
        if body.site_id:
            try:
                state.settings.set_site_id(str(body.site_id).strip())
            except ValueError as e:
                raise ApiError(400, "BAD_REQUEST", str(e)) from e
        if body.threshold_c is not None:
            try:
                state.settings.set_threshold(float(body.threshold_c))
            except ValueError as e:
                raise ApiError(400, "BAD_REQUEST", str(e)) from e
        mid = (body.model_id or "").strip()
        if mid:
            if mid not in MODEL_CATALOG:
                mid = DEFAULT_MODEL_ID
            try:
                state.settings.set_model_id(mid)
            except ValueError as e:
                raise ApiError(400, "BAD_REQUEST", str(e)) from e
        _persist_room_meta(state)
        if body.host_contributes:
            try:
                _launch_local_agent(state, worker_password=wp)
            except ApiError as exc:
                # Room creation is durable even if UAC is denied or the agent
                # binary is missing; the dashboard exposes a retry action.
                log.warning("[AGENT] local Host start deferred code=%s", exc.code)
        # tunnel_ready chỉ từ len RAM — không đọc cờ disk
        log.info("[ROOM] bootstrap code=%s name=%r tunnel_ready=%s",
                 state.room.room_code, state.room.display_name,
                 state.room.passwords_meet_tunnel_min(12))
        result = _room_public_payload(state, time.time())
        result["local_agent"] = dict(state.local_agent)
        return result
    finally:
        _bootstrap_lock.release()


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"

def _bearer(request: Request) -> str | None:
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None

def _audit(state, ts, ip, event_type, detail):
    state.store.insert_room_audit(ts, ip, event_type, detail)

def _check_identity(auth: AuthContext, claimed: str | None, state, ip: str):
    if claimed is None or auth.node is None:
        return
    if claimed != auth.node:
        _audit(state, time.time(), ip, "identity_mismatch",
               {"token_node": auth.node, "claimed": claimed})
        raise ApiError(403, "IDENTITY_MISMATCH",
                       "Danh tính trong thân yêu cầu không khớp token.")

# --- request bodies ---

class JoinBody(BaseModel):
    room_code: str
    password: str
    node_name: str | None = None
    role: str = "worker"  # worker | admin
    capabilities: dict | None = None


class RoomBootstrapBody(BaseModel):
    display_name: str | None = None
    name: str | None = None  # alias
    worker_password: str
    admin_password: str
    site_id: str | None = None
    model_id: str | None = None
    threshold_c: float | None = None
    room_code: str | None = None
    regenerate_code: bool = False
    host_contributes: bool = True


class LocalAgentStartBody(BaseModel):
    """Worker password one-shot for recreating an unavailable local config."""
    worker_password: str | None = Field(default=None, max_length=4096)


class TunnelConfigBody(BaseModel):
    mode: Literal["quick", "named"]
    hostname: str | None = Field(default=None, max_length=253)
    # Write-only: never stored in JSON, state response, audit, or logs.
    token: str | None = Field(default=None, min_length=1, max_length=4096)


class TunnelEnableBody(BaseModel):
    mode: Literal["quick", "named"] | None = None
    hostname: str | None = Field(default=None, max_length=253)


class ReadyBody(BaseModel):
    model_id: str | None = None
    model_sha256: str | None = Field(default=None, max_length=128)
    model_generation: int | None = Field(default=None, ge=0)
    runtime_id: str | None = Field(default=None, max_length=128)
    runtime_ready: bool = False

class IngestBody(BaseModel):
    # node is NOT trusted — identity comes from token (invariant #2).
    node: str | None = None
    ts: float | None = Field(default=None, allow_inf_nan=False)
    cpu_temp: float | None = Field(default=None, allow_inf_nan=False)
    gpu_temp: float | None = Field(default=None, allow_inf_nan=False)
    cpu_util: float | None = Field(default=None, allow_inf_nan=False)
    power_w: float | None = Field(default=None, allow_inf_nan=False)
    cpu_clock_mhz: float | None = Field(default=None, allow_inf_nan=False)
    fan_rpm: float | None = Field(default=None, allow_inf_nan=False)
    seconds_since_input: float | None = Field(
        default=None, allow_inf_nan=False)
    power_source: Literal["sensor", "model", "none"] = "none"

    @field_validator(
        "ts", "cpu_temp", "gpu_temp", "cpu_util", "power_w",
        "cpu_clock_mhz", "fan_rpm", "seconds_since_input", mode="before")
    @classmethod
    def reject_boolean_metrics(cls, value):
        if isinstance(value, bool):
            raise ValueError("boolean không phải số đo")
        return value

class SettingsBody(BaseModel):
    threshold_c: float | None = Field(
        default=None, ge=40, le=100, allow_inf_nan=False)
    scheduler_mode: str | None = None
    w_cool: float | None = Field(default=None, allow_inf_nan=False)
    w_idle: float | None = Field(default=None, allow_inf_nan=False)
    w_power: float | None = Field(default=None, allow_inf_nan=False)
    w_load: float | None = Field(default=None, allow_inf_nan=False)
    model_id: str | None = None
    allow_tunnel: bool | None = None
    site_id: str | None = None
    run_window_start: str | None = None
    run_window_end: str | None = None
    jobs_paused: bool | None = None

    @field_validator(
        "threshold_c", "w_cool", "w_idle", "w_power", "w_load",
        mode="before")
    @classmethod
    def reject_boolean_numbers(cls, value):
        if isinstance(value, bool):
            raise ValueError("boolean không phải số")
        return value


class JobResultBody(BaseModel):
    job_id: str
    attempt_id: str | None = Field(default=None, max_length=64)
    status: Literal["ok", "error", "timeout", "preempted"] = "ok"
    text: str | None = Field(default=None, max_length=JOB_RESULT_TEXT_MAX_CHARS)
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens_estimated: bool = False
    duration_ms: float | None = Field(default=None, allow_inf_nan=False)
    prompt_eval_ms: float | None = Field(default=None, allow_inf_nan=False)
    energy_j: float | None = Field(default=None, allow_inf_nan=False)
    energy_source: Literal["sensor", "model", "none"] = "none"
    peak_temp_c: float | None = Field(default=None, allow_inf_nan=False)
    min_clock_mhz: float | None = Field(default=None, allow_inf_nan=False)
    fan_rpm: float | None = Field(default=None, allow_inf_nan=False)
    fan_rpm_hot: float | None = Field(default=None, allow_inf_nan=False)
    fan_rpm_cool: float | None = Field(default=None, allow_inf_nan=False)
    cpu_util: float | None = Field(default=None, allow_inf_nan=False)
    base_clock_mhz: float | None = Field(default=None, allow_inf_nan=False)
    error_message: str | None = Field(default=None, max_length=512)

    @field_validator(
        "tokens_in", "tokens_out", "duration_ms", "prompt_eval_ms",
        "energy_j", "peak_temp_c", "min_clock_mhz", "fan_rpm",
        "fan_rpm_hot", "fan_rpm_cool", "cpu_util", "base_clock_mhz",
        mode="before")
    @classmethod
    def reject_boolean_metrics(cls, value):
        if isinstance(value, bool):
            raise ValueError("boolean không phải số đo")
        return value


class JobEventBody(BaseModel):
    attempt_id: str = Field(min_length=1, max_length=64)
    seq: int = Field(ge=0)
    delta: str = Field(min_length=1, max_length=4096)
    timing: dict | None = None


class WeatherReadingBody(BaseModel):
    """Browser-fetched OWM snapshot — never includes the API key."""
    temp_c: float = Field(ge=-80.0, le=80.0)
    feels_like_c: float | None = Field(default=None, ge=-80.0, le=80.0)
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    label: str | None = Field(default=None, max_length=80)
    source: str = Field(default="owm_gps", max_length=32)
    site_id: str = Field(default="gps-host", max_length=64)

    @field_validator("temp_c", "feels_like_c", "lat", "lon", mode="before")
    @classmethod
    def reject_bool(cls, value):
        if isinstance(value, bool):
            raise ValueError("boolean is not a measurement")
        return value


def _check_default_quota(state, auth):
    from rate_limit import DEFAULT_MAX, DEFAULT_WINDOW_S
    if state.rate_limiter is not None:
        state.rate_limiter.check(
            "default", auth.token_hash,
            max_hits=DEFAULT_MAX, window_s=DEFAULT_WINDOW_S)


class ChatBody(BaseModel):
    prompt: str = Field(max_length=CHAT_PROMPT_MAX_CHARS)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False


def _matching_ready_nodes(state: AppState) -> list[str]:
    """Return nodes that proved they run the selected room generation."""
    model = get_model(_selected_model_id(state))
    expected_hash = str(model["sha256"]).upper()
    result = []
    for node, forecast in state.forecast_cache.all().items():
        if (forecast.model_ready
                and forecast.state in ("READY", "OK")
                and forecast.model_id == model["model_id"]
                and str(forecast.model_sha256 or "").upper() == expected_hash
                and forecast.model_generation == state.llm_generation
                and forecast.runtime_id == RUNTIME_ID):
            result.append(node)
    return sorted(result)


def _chat_deadline_s(max_tokens: int) -> float:
    """Bound inference lease by output size instead of one fixed 60s value.

    The conservative rate keeps 2B-class CPU models from being reclaimed
    while still bounding a wedged runtime.  Per-model benchmark tuning can
    override this through the catalog later without changing the API seam.
    """
    return float(max(60, min(600, 30 + int(max_tokens) * 2)))


def compute_ab_summary(events: list[dict]) -> dict:
    """Tổng hợp số liệu so sánh A/B giữa Thermal-Aware và Round-Robin từ esg_events."""
    stats = {
        "thermal_aware": {
            "total_jobs": 0,
            "max_temp_c": None,
            "total_energy_j": 0.0,
            "total_tokens_out": 0,
            "energy_per_token_j": None,
        },
        "round_robin": {
            "total_jobs": 0,
            "max_temp_c": None,
            "total_energy_j": 0.0,
            "total_tokens_out": 0,
            "energy_per_token_j": None,
        },
    }

    temps: dict[str, list[float]] = {"thermal_aware": [], "round_robin": []}

    for ev in events:
        etype = (ev.get("event_type") or "").strip().lower()
        if etype not in ("job_complete", "job_completed"):
            continue

        detail = ev.get("detail")
        if not isinstance(detail, dict):
            continue

        raw_mode = str(detail.get("scheduler_mode") or "").strip().lower().replace("-", "_")
        if raw_mode not in ("thermal_aware", "round_robin"):
            continue

        target = stats[raw_mode]
        target["total_jobs"] += 1

        # Nhiệt độ đỉnh (peak_temp_c)
        pt = detail.get("peak_temp_c")
        if pt is not None:
            try:
                val = float(pt)
                if not math.isnan(val) and not math.isinf(val):
                    temps[raw_mode].append(val)
            except (ValueError, TypeError):
                pass

        # Năng lượng tiêu thụ (energy_j)
        ej = detail.get("energy_j")
        if ej is not None:
            try:
                val = float(ej)
                if not math.isnan(val) and not math.isinf(val):
                    target["total_energy_j"] += val
            except (ValueError, TypeError):
                pass

        # Token xuất ra (tokens_out)
        to = detail.get("tokens_out")
        if to is not None:
            try:
                val = int(to)
                if val > 0:
                    target["total_tokens_out"] += val
            except (ValueError, TypeError):
                pass

    for mode in ("thermal_aware", "round_robin"):
        if temps[mode]:
            stats[mode]["max_temp_c"] = round(max(temps[mode]), 2)
        else:
            stats[mode]["max_temp_c"] = None

        tot_tok = stats[mode]["total_tokens_out"]
        tot_ej = stats[mode]["total_energy_j"]
        if tot_tok > 0 and tot_ej > 0:
            stats[mode]["energy_per_token_j"] = round(tot_ej / tot_tok, 4)
        else:
            stats[mode]["energy_per_token_j"] = None

        stats[mode]["total_energy_j"] = round(stats[mode]["total_energy_j"], 2)

    # Tính toán chênh lệch so sánh
    ta_jpt = stats["thermal_aware"]["energy_per_token_j"]
    rr_jpt = stats["round_robin"]["energy_per_token_j"]
    savings_pct = None
    if ta_jpt is not None and rr_jpt is not None and rr_jpt > 0:
        savings_pct = round(((rr_jpt - ta_jpt) / rr_jpt) * 100.0, 1)

    ta_max_t = stats["thermal_aware"]["max_temp_c"]
    rr_max_t = stats["round_robin"]["max_temp_c"]
    temp_diff_c = None
    if ta_max_t is not None and rr_max_t is not None:
        temp_diff_c = round(rr_max_t - ta_max_t, 1)

    return {
        "ok": True,
        "thermal_aware": stats["thermal_aware"],
        "round_robin": stats["round_robin"],
        "comparison": {
            "energy_savings_pct": savings_pct,
            "temp_reduction_c": temp_diff_c,
        },
    }


def create_app(state):
    @contextlib.asynccontextmanager
    async def lifespan(app):
        tasks = []
        if os.environ.get("POC_NO_BACKGROUND") != "1":
            _restore_local_agent(state)
            if state.calibrate:
                from calibrate import calibration_task
                tasks = [
                    asyncio.create_task(calibration_task(state)),
                    asyncio.create_task(_wal_loop(state)),
                ]
            else:
                tasks = [
                    asyncio.create_task(_forecast_loop(state)),
                    asyncio.create_task(_scheduler_loop(state)),
                    asyncio.create_task(_weather_loop(state)),
                    asyncio.create_task(_tunnel_idle_loop(state)),
                    asyncio.create_task(_wal_loop(state)),
                ]
                if state.demo_load:
                    tasks.append(asyncio.create_task(_generator_loop(state)))
        yield
        if state.tunnel is not None:
            try:
                state.tunnel.disable()
            except Exception:
                pass
        for t in tasks:
            t.cancel()

    app = FastAPI(lifespan=lifespan)
    app.state.thermal_state = state
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)

    def require_auth(request: Request) -> AuthContext:
        return state.room.resolve(_bearer(request))

    def require_admin(auth: AuthContext = Depends(require_auth)) -> AuthContext:
        if auth.role != "admin":
            raise ApiError(403, "FORBIDDEN",
                           "Cần quyền quản trị cho thao tác này.")
        return auth

    def require_worker(auth: AuthContext = Depends(require_auth)) -> AuthContext:
        if auth.role not in ("worker", "admin"):
            raise ApiError(403, "FORBIDDEN", "Token không hợp lệ cho worker.")
        return auth

    @app.get("/api/version")
    def api_version():
        """Public, secret-free compatibility revision for the static dashboard."""
        return {"api_revision": API_REVISION}

    @app.get("/api/ab_summary")
    def ab_summary():
        """Báo cáo so sánh A/B giữa Thermal-Aware và Round-Robin."""
        events = state.store.esg_events()
        return compute_ab_summary(events)

    @app.get("/join")
    def join_get(code: str | None = None):
        """QR / link mời mở bằng GET → landing (POST /join vẫn là API)."""
        loc = "/"
        if code:
            loc = f"/?code={quote(code, safe='')}"
        return RedirectResponse(url=loc, status_code=302)

    @app.post("/join", status_code=201)
    def join(body: JoinBody, request: Request):
        role = body.role if body.role in ("worker", "admin") else "worker"
        ip = _client_ip(request)
        token, expires = state.room.join(
            room_code=body.room_code, password=body.password,
            node_name=body.node_name, role=role, ip=ip,
            store_audit=lambda *a: _audit(state, *a))
        joined_name = (body.node_name.strip()
                       if role == "worker" and body.node_name else None)
        if role == "worker" and joined_name:
            joined_at = time.time()
            state.forecast_cache.mark_joining(joined_name, joined_at)
            if joined_name == state.local_agent.get("node"):
                _set_local_agent_state(
                    state, "joining", "Host agent đã tham gia phòng.",
                    joined_name)
            _invalidate_state_cache(state)
            _publish_node_event(
                state, joined_name, "joined", state_name="WARMING_UP",
                reason="worker joined",
                decision_threshold_c=state.settings.get()["threshold_c"])
        s = state.settings.get()
        lan_url = detect_lan_url()
        tunnel_url = None
        if state.tunnel is not None:
            st = state.tunnel.status()
            tunnel_url = st.get("public_url")
            _touch_tunnel_if_enabled(state, request)
        return {
            "token": token,
            "expires_at": expires,
            "node_name": joined_name,
            "role": role,
            "lan_url": lan_url,
            "tunnel_url": tunnel_url,
            "room_config": room_config_llm(
                threshold_c=s["threshold_c"],
                site_id=s.get("site_id") or "local",
                max_concurrent=1,
                telemetry_interval_s=2.0,
                model_id=_selected_model_id(state),
                model_generation=state.llm_generation,
            ),
        }

    @app.post("/leave")
    def leave(request: Request, auth: AuthContext = Depends(require_auth)):
        now = time.time()
        with state.lifecycle.mutation_section():
            _room_mut_begin(state, auth, now=now)
            info = None
            if auth.node:
                info = _release_node_work(state, auth.node)
                log.info("[SCHED] leave %s — %s", auth.node, info)
                _publish_node_event(
                    state, auth.node, "left", state_name="OFFLINE",
                    reason="worker left",
                    decision_threshold_c=state.settings.get()["threshold_c"])
            state.room.leave(auth, store_audit=lambda *a: _audit(state, *a),
                             ip=_client_ip(request))
            _invalidate_state_cache(state)
        return {"ok": True}

    @app.get("/worker-setup")
    def worker_setup_guide(request: Request, code: str = ""):
        """Hướng dẫn cài worker từ link mời, không cấp token hay lộ mật khẩu."""
        if (not state.room.has_password()
                or code.strip().upper() != state.room.room_code):
            raise ApiError(404, "NOT_FOUND", "Link máy khách không hợp lệ.")
        from html import escape
        origin = str(request.base_url).rstrip("/")
        setup_url = f"{origin}/worker-setup?code={quote(state.room.room_code)}"
        safe_url = escape(setup_url, quote=True)
        safe_code = escape(state.room.room_code, quote=True)
        page = f"""<!doctype html><html lang=\"vi\"><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Tham gia làm máy khách · Thermal Orchestrator</title>
<style>body{{margin:0;background:#0b1020;color:#f2f7ff;font:16px Segoe UI,system-ui,sans-serif}}
main{{max-width:680px;margin:7vh auto;padding:28px;background:#131b2f;border:1px solid #2c3a56;border-radius:16px}}
code{{display:block;overflow:auto;padding:12px;background:#0d1117;border-radius:8px;color:#b9e6ff}}
button{{min-height:44px;margin-top:14px;padding:8px 14px;color:#f2f7ff;background:#168dca;border:0;border-radius:7px;cursor:pointer}}
p{{line-height:1.55;color:#c5d2e6}}</style><main><h1>Tham gia làm máy khách</h1>
<p>Link này dành cho <strong>NodeAgent worker</strong>, không đăng nhập quản trị và không mở cổng vào máy khách.</p>
<p>1. Chép link dưới đây sang máy Windows muốn đóng góp tài nguyên.</p><code id=\"link\">{safe_url}</code>
<button onclick=\"navigator.clipboard?.writeText(document.getElementById('link').textContent)\">Sao chép link worker</button>
<p>2. Chạy <code>NodeAgent.exe --setup \"{safe_url}\"</code>. Wizard chỉ hỏi tên node và mật khẩu worker; mã phòng <strong>{safe_code}</strong> đã có trong link.</p>
<p>Mật khẩu worker không nằm trong link và không gửi tới trình duyệt.</p></main></html>"""
        return Response(page, media_type="text/html; charset=utf-8")

    @app.post("/nodes/ready")
    def nodes_ready(body: ReadyBody, auth: AuthContext = Depends(require_worker)):
        now = time.time()
        with state.lifecycle.mutation_section():
            _room_mut_begin(state, auth, now=now)
            if auth.role == "worker" and auth.node:
                selected = get_model(_selected_model_id(state))
                exact = (
                    bool(body.runtime_ready)
                    and body.model_id == selected["model_id"]
                    and str(body.model_sha256 or "").upper()
                    == str(selected["sha256"]).upper()
                    and body.model_generation == state.llm_generation
                    and body.runtime_id == RUNTIME_ID
                )
                state.forecast_cache.mark_warming_up(
                    auth.node, now, model_ready=exact,
                    model_id=body.model_id,
                    model_sha256=body.model_sha256,
                    model_generation=(body.model_generation
                                      if body.model_generation is not None
                                      else -1),
                    runtime_id=body.runtime_id)
                if auth.node == state.local_agent.get("node"):
                    _set_local_agent_state(
                        state,
                        "downloading_model" if not exact else "ready",
                        ("Host agent đang tải model đã chọn."
                         if not exact else "Host agent đã sẵn sàng."),
                        auth.node)
                _invalidate_state_cache(state)
                if bool(body.runtime_ready) and not exact:
                    raise ApiError(
                        409, "MODEL_MISMATCH",
                        "Runtime không khớp model/generation đang chọn.",
                        detail={"model_id": selected["model_id"],
                                "model_generation": state.llm_generation})
                return {"ok": True, "state": "WARMING_UP",
                        "model_ready": exact,
                        "model_generation": state.llm_generation}
            return {"ok": True, "state": "WARMING_UP"}

    @app.post("/ingest")
    def ingest(body: IngestBody, request: Request,
               auth: AuthContext = Depends(require_worker)):
        from rate_limit import INGEST_MAX, INGEST_WINDOW_S
        if state.rate_limiter is not None:
            state.rate_limiter.check(
                "ingest", auth.token_hash,
                max_hits=INGEST_MAX, window_s=INGEST_WINDOW_S)
        ip = _client_ip(request)
        _check_identity(auth, body.node, state, ip)
        _touch_tunnel_if_enabled(state, request)
        if auth.role == "worker":
            if not auth.node:
                raise ApiError(403, "FORBIDDEN", "Token worker thiếu node.")
            node = auth.node
        else:
            raise ApiError(403, "FORBIDDEN",
                           "Chỉ worker mới gửi telemetry.")
        if body.power_source not in ("sensor", "model", "none"):
            raise ApiError(400, "BAD_REQUEST",
                           "power_source phải là sensor|model|none.")
        # Boundary: finite + range (chặn NaN/Inf poison forecast/DB)
        cpu_temp = _finite_optional(
            "cpu_temp", body.cpu_temp, lo=-50.0, hi=150.0)
        gpu_temp = _finite_optional(
            "gpu_temp", body.gpu_temp, lo=-50.0, hi=150.0)
        cpu_util = _finite_optional(
            "cpu_util", body.cpu_util, lo=0.0, hi=100.0)
        power_w = _finite_optional(
            "power_w", body.power_w, lo=0.0, hi=2_000.0)
        client_ts = _finite_optional("ts", body.ts)
        secs = _finite_optional(
            "seconds_since_input", body.seconds_since_input, lo=0.0)
        server_ts = time.time()
        with state.lifecycle.mutation_section():
            _room_mut_begin(state, auth, now=server_ts)
            ts = client_ts if client_ts is not None else server_ts
            if abs(ts - server_ts) > CLOCK_SKEW_S:
                last = state._skew_warned_at.get(node, 0.0)
                if server_ts - last >= SKEW_LOG_EVERY_S:
                    log.warning(
                        "[TELEMETRY] %s clock skew %.1fs — dùng server_ts",
                        node, ts - server_ts)
                    state._skew_warned_at[node] = server_ts
                ts = server_ts
            power_source = body.power_source
            if (power_w is None and power_source == "model"
                    and state.power_model is not None):
                est = state.power_model.estimate_w(cpu_util, cpu_temp)
                if est is not None:
                    power_w = est
            state.store.insert(node, ts, cpu_temp, gpu_temp,
                               cpu_util, power_w)
            if node == state.local_agent.get("node"):
                local_state = state.local_agent.get("state")
                if local_state not in ("ready", "downloading_model"):
                    _set_local_agent_state(
                        state, "joining", "Host agent đang gửi telemetry.", node)
            if power_source in ("sensor", "model"):
                state.forecast_cache.patch(
                    node, power_w=power_w, power_source=power_source)
            # Step 4a: a PC with recent keyboard/mouse input gets no new
            # jobs. The Host's own PC is exempt.
            idle_after = float(
                state.settings.get().get("user_idle_after_s", 120.0))
            user_active = (secs is not None and secs < idle_after
                           and node != state.local_agent.get("node"))
            state.forecast_cache.patch(node, user_active=user_active)
            _invalidate_state_cache(state)
        return JSONResponse(
            status_code=202,
            content={"ok": True, "server_ts": server_ts,
                     "power_source": body.power_source})

    @app.get("/jobs/next")
    async def jobs_next(request: Request, node: str | None = None,
                        wait: float = 25.0,
                        auth: AuthContext = Depends(require_worker)):
        # S3: ?node= ignored. Long-poll vẫn outbound-only (worker khởi tạo).
        if auth.role != "worker" or not auth.node:
            raise ApiError(403, "FORBIDDEN", "Chỉ worker mới nhận job.")
        from rate_limit import JOBS_MAX, JOBS_WINDOW_S
        if state.rate_limiter is not None:
            state.rate_limiter.check(
                "jobs", auth.token_hash,
                max_hits=JOBS_MAX, window_s=JOBS_WINDOW_S)
        # Key theo token — không chặn node cùng tên phòng mới
        poll_key = auth.token_hash
        _touch_tunnel_if_enabled(state, request)
        _ = node  # discarded; identity comes from token only
        if poll_key in state._job_polls:
            raise ApiError(429, "TOO_MANY_POLLS",
                           "Đã có long-poll đang mở cho token này.")
        state._job_polls.add(poll_key)
        wait = max(0.0, min(30.0, float(wait)))
        deadline = time.monotonic() + wait
        who = auth.node
        gen = auth.generation
        th = auth.token_hash
        try:
            while True:
                now = time.time()
                remaining = deadline - time.monotonic()
                if remaining < 0:
                    return Response(status_code=204)
                job = None
                try:
                    with state.lifecycle.mutation_section(
                            timeout=max(0.0, remaining)):
                        if (threading.get_ident()
                                == state.lifecycle._runner_ident):
                            raise ApiError(
                                401, "TOKEN_REVOKED",
                                "Token đã bị thu hồi hoặc phòng đã đổi.")
                        if not state.room.token_alive(th, gen, now=now):
                            raise ApiError(
                                401, "TOKEN_REVOKED",
                                "Token đã bị thu hồi hoặc phòng đã đổi.")
                        state.lifecycle.ensure_commit_allowed()
                        hook = state.lifecycle._before_claim_hook
                        if hook is not None:
                            state.lifecycle._before_claim_hook = None
                            hook()
                            now = time.time()
                            if not state.room.token_alive(th, gen, now=now):
                                raise ApiError(
                                    401, "TOKEN_REVOKED",
                                    "Token đã bị thu hồi hoặc phòng đã đổi.")
                            state.lifecycle.ensure_commit_allowed()
                        fc = state.forecast_cache.get(who)
                        if state.settings.get().get("jobs_paused"):
                            job = None  # all jobs stopped by admin
                        elif fc is not None and fc.user_active:
                            # Step 4b: a person is using this PC; give its
                            # reserved job back to the queue.
                            for jid in state.balancer.release_reservations_for(
                                    who):
                                if who != HOST_NODE:
                                    state.forecast_cache.adjust_inflight(
                                        who, -1)
                                state.chat.mark_queued(jid, now=now)
                            job = None
                        else:
                            job = state.balancer.claim(who, now=now)
                        if job is not None:
                            if (job.get("type") or "burn") == "chat":
                                state.chat.mark_running(
                                    job["id"], who,
                                    attempt_id=job.get("attempt_id"), now=now)
                except LifecycleWaitTimeout:
                    return Response(status_code=204)
                if job is not None:
                    return _job_payload_for_worker(job)
                if time.monotonic() >= deadline:
                    return Response(status_code=204)
                await asyncio.sleep(0.05)
        finally:
            state._job_polls.discard(poll_key)

    @app.post("/jobs/result")
    def jobs_result(body: JobResultBody,
                    auth: AuthContext = Depends(require_worker)):
        if auth.role != "worker" or not auth.node:
            raise ApiError(403, "FORBIDDEN", "Chỉ worker mới trả kết quả.")
        _check_default_quota(state, auth)
        now = time.time()
        with state.lifecycle.mutation_section():
            state.lifecycle.require_alive(state.room, auth, now=now)
            hook = state.lifecycle._before_result_commit_hook
            if hook is not None:
                state.lifecycle._before_result_commit_hook = None
                hook()
                now = time.time()
                state.lifecycle.require_alive(state.room, auth, now=now)
            return apply_job_result(state, auth.node, body, now)

    @app.post("/jobs/{job_id}/events", status_code=202)
    def job_event(job_id: str, body: JobEventBody,
                  auth: AuthContext = Depends(require_worker)):
        """Accept ordered inference deltas from an outbound worker only."""
        if auth.role != "worker" or not auth.node:
            raise ApiError(403, "FORBIDDEN", "Chỉ worker mới gửi stream.")
        from rate_limit import JOB_EVENTS_MAX, JOB_EVENTS_WINDOW_S
        if state.rate_limiter is not None:
            state.rate_limiter.check(
                "job_events", auth.token_hash,
                max_hits=JOB_EVENTS_MAX, window_s=JOB_EVENTS_WINDOW_S)
        now = time.time()
        with state.lifecycle.mutation_section():
            state.lifecycle.require_alive(state.room, auth, now=now)
            active = state.balancer.peek_active(auth.node)
            if (active is None or active.get("id") != job_id
                    or active.get("attempt_id") != body.attempt_id):
                raise ApiError(409, "STALE_ATTEMPT",
                               "Stream thuộc lần chạy đã hết hiệu lực.")
            accepted = state.chat.append_delta(
                job_id, attempt_id=body.attempt_id, seq=body.seq,
                delta=body.delta, now=now)
            if not accepted:
                raise ApiError(409, "STREAM_SEQUENCE",
                               "Thứ tự chunk stream không hợp lệ.")
        _publish_admin_event(state, {
            "type": "chat_token", "job_id": job_id,
            "attempt_id": body.attempt_id, "seq": body.seq,
            "delta": body.delta,
        })
        return {"ok": True}

    @app.post("/chat")
    def post_chat(body: ChatBody,
                  auth: AuthContext = Depends(require_admin)):
        """202 ngay — không chờ suy luận (docs/03 §6)."""
        from rate_limit import CHAT_MAX, CHAT_WINDOW_S
        if state.rate_limiter is not None:
            state.rate_limiter.check(
                "chat", auth.token_hash,
                max_hits=CHAT_MAX, window_s=CHAT_WINDOW_S)
        prompt = (body.prompt or "").strip()
        if not prompt:
            raise ApiError(422, "BAD_REQUEST", "prompt không được rỗng.")
        now = time.time()
        # Enqueue dưới read-lock — không dùng 409 ROOM_LIFECYCLE
        with state.lifecycle.mutation_section():
            state.lifecycle.require_alive(state.room, auth, now=now)
            # The Host participates only through its normal outbound NodeAgent.
            # Check availability after lifecycle validation so a token revoked
            # while a room closes is consistently reported as TOKEN_REVOKED.
            if not _matching_ready_nodes(state):
                raise ApiError(409, "NO_LLM_READY",
                               "Chưa có node chạy đúng model đang chọn.")
            job = state.balancer.enqueue_job(
                job_type="chat",
                prompt=prompt,
                params={"max_tokens": body.max_tokens,
                        "temperature": body.temperature,
                        "stream": bool(body.stream)},
                deadline_s=_chat_deadline_s(body.max_tokens),
                duration_s=0,
                cores=0,
            )
            if job is None:
                raise ApiError(
                    429, "QUEUE_FULL",
                    "Hàng đợi chat đã đầy.",
                    detail={"max_queue": state.balancer.max_queue})
            pos = state.balancer.queue_position(job["id"]) or 1
            est = float(pos) * 8.0
            ok = state.chat.enqueue(
                job["id"], queue_position=pos,
                estimated_wait_s=est, now=now)
            if not ok:
                state.balancer.drop_pending(job["id"])
                raise ApiError(
                    429, "QUEUE_FULL",
                    "Hàng đợi chat đã đầy.",
                    detail={"max_queue": state.balancer.max_queue})
        log.info("[SCHED] chat job %s enqueued pos=%d (prompt_len=%d)",
                 job["id"], pos, len(prompt))
        return JSONResponse(
            status_code=202,
            content={"job_id": job["id"],
                     "queue_position": pos,
                     "estimated_wait_s": est,
                     "stream": bool(body.stream)},
        )

    @app.get("/chat/{job_id}")
    def get_chat(job_id: str, auth: AuthContext = Depends(require_admin)):
        _check_default_quota(state, auth)
        info = state.chat.get(job_id)
        if info is None:
            failed = state.balancer.get_failed(job_id)
            if failed is not None:
                return {"job_id": job_id, "status": "error",
                        "error_message": failed.get("detail")
                        or failed.get("code")}
            raise ApiError(404, "NOT_FOUND", "Không tìm thấy job chat.")
        out = {
            "job_id": info["job_id"],
            "status": info["status"],
        }
        if info["status"] == "done":
            out.update({
                "text": info.get("text"),
                "node": info.get("node"),
                "duration_ms": info.get("duration_ms"),
                "tokens_in": info.get("tokens_in"),
                "tokens_out": info.get("tokens_out"),
            })
        elif info["status"] == "error":
            out["error_message"] = info.get("error_message")
            out["node"] = info.get("node")
        elif info["status"] == "queued":
            pos = state.balancer.queue_position(job_id)
            if pos is None:
                pos = info.get("queue_position")
            out["queue_position"] = pos
            out["estimated_wait_s"] = (
                float(pos) * 8.0 if pos is not None
                else info.get("estimated_wait_s"))
        elif info["status"] == "running":
            out["node"] = info.get("node")
            out["attempt_id"] = info.get("attempt_id")
            out["partial_text"] = info.get("partial_text") or ""
        return out

    @app.get("/api/state")
    def api_state(request: Request, auth: AuthContext = Depends(require_auth)):
        _check_default_quota(state, auth)
        _touch_tunnel_if_enabled(state, request)
        compact = auth.role == "worker"
        payload = _cached_state_payload(state, time.time(), compact=compact)
        if not compact:
            payload["api_revision"] = API_REVISION
            payload["password_policy"] = {
                "min_length": PASSWORD_MIN_LENGTH,
                "unit": "unicode_characters",
            }
        return payload

    @app.post("/api/settings")
    def api_settings(body: SettingsBody,
                     auth: AuthContext = Depends(require_admin)):
        _check_default_quota(state, auth)
        now = time.time()
        with state.lifecycle.mutation_section():
            _room_mut_begin(state, auth, now=now)
            # S14 / E12: sàn theo idle_baseline thật của các node hoạt động.
            margin = float(state.esg.config.get(
                "threshold_margin_above_idle_c", 8.0) or 8.0)
            baselines = [
                fc.idle_baseline_c
                for fc in state.forecast_cache.all().values()
                if fc.state in ("READY", "WARMING_UP", "AT_RISK")
            ]
            idle_floor = 40.0
            if baselines:
                idle_floor = max(40.0, max(baselines) + margin)
            if body.threshold_c is not None:
                if body.threshold_c < idle_floor:
                    raise ApiError(
                        422, "THRESHOLD_TOO_LOW",
                        "Ngưỡng thấp hơn sàn cho phép.",
                        detail={"minimum": idle_floor})
                old_th = state.settings.get()["threshold_c"]
                state.settings.set_threshold(body.threshold_c)
                if body.threshold_c != old_th:
                    record_esg_event(
                        state, now, "system", "threshold_changed",
                        {"old": old_th, "new": body.threshold_c,
                         "threshold_at_time": body.threshold_c})
                log.info("[SETTINGS] threshold set to %.1f°C",
                         body.threshold_c)
            if body.scheduler_mode is not None:
                try:
                    state.settings.set_scheduler_mode(body.scheduler_mode)
                except ValueError as e:
                    raise ApiError(422, "BAD_SETTINGS", str(e)) from e
                log.info("[SETTINGS] scheduler_mode=%s", body.scheduler_mode)
            weights = [body.w_cool, body.w_idle, body.w_power, body.w_load]
            if any(w is not None for w in weights):
                if not all(w is not None for w in weights):
                    raise ApiError(422, "BAD_SETTINGS",
                                   "Phải gửi đủ bốn trọng số.")
                try:
                    state.settings.set_weights(*weights)
                except ValueError as e:
                    raise ApiError(422, "BAD_SETTINGS", str(e)) from e
                log.info("[SETTINGS] weights updated")
            if body.model_id is not None:
                mid = body.model_id.strip()
                if not mid:
                    raise ApiError(422, "UNKNOWN_MODEL", "empty model_id")
                try:
                    get_model(mid)
                except UnknownModelError as e:
                    raise ApiError(422, "UNKNOWN_MODEL", str(e)) from e
                old = _selected_model_id(state)
                try:
                    state.settings.set_model_id(mid)
                except ValueError as e:
                    raise ApiError(422, "BAD_SETTINGS", str(e)) from e
                if mid != old:
                    state.llm_generation += 1
                    state.forecast_cache.reset_model_readiness(
                        state.llm_generation)
                    log.info("[LLM] model_id %s → %s", old, mid)
                    log.info("[LLM] generation=%d; invalidated node readiness",
                             state.llm_generation)
                    record_esg_event(
                        state, now, "system", "model_changed",
                        {"old": old, "new": mid})
            if body.allow_tunnel is not None:
                if body.allow_tunnel is True and not bool(
                        state.settings.get().get("allow_tunnel", True)):
                    raise ApiError(
                        403, "FORBIDDEN",
                        "allow_tunnel=false (IT). Không bật lại qua API — "
                        "sửa settings.json trên máy host.")
                if body.allow_tunnel is False:
                    state.settings.set_allow_tunnel(False)
                    if state.tunnel is not None:
                        state.tunnel.set_allow_tunnel(False)
                    log.info("[SETTINGS] allow_tunnel=false (kill-switch)")
                else:
                    log.info("[SETTINGS] allow_tunnel đã là true — bỏ qua")
            if body.site_id is not None:
                try:
                    state.settings.set_site_id(body.site_id)
                except ValueError as e:
                    raise ApiError(422, "BAD_SETTINGS", str(e)) from e
                log.info("[SETTINGS] site_id=%s", body.site_id)
            if (body.run_window_start is not None
                    or body.run_window_end is not None):
                try:
                    state.settings.set_run_window(
                        body.run_window_start or "", body.run_window_end or "")
                except ValueError as e:
                    raise ApiError(422, "BAD_SETTINGS", str(e)) from e
                cfg = state.settings.get()
                log.info("[SETTINGS] run_window=%s-%s",
                         cfg["run_window_start"], cfg["run_window_end"])
            if body.jobs_paused is not None:
                state.settings.set_jobs_paused(body.jobs_paused)
                if body.jobs_paused:
                    # Take back running jobs and reservations at once.
                    for node, _ in state.balancer.list_active():
                        _release_active(state, node, now,
                                        code="STOPPED_BY_ADMIN",
                                        reason="stop all")
                    targets = {j.get("target")
                               for j in state.balancer.reserved()
                               if j.get("target")}
                    for node in targets:
                        released = state.balancer.release_reservations_for(node)
                        for jid in released:
                            if node != HOST_NODE:
                                state.forecast_cache.adjust_inflight(node, -1)
                            state.chat.mark_queued(jid, now=now)
                log.info("[SETTINGS] jobs_paused=%s", body.jobs_paused)
            _invalidate_state_cache(state)
            return {"ok": True, "settings": state.settings.get()}

    @app.get("/api/models")
    def api_models(auth: AuthContext = Depends(require_admin)):
        _check_default_quota(state, auth)
        selected = _selected_model_id(state)
        return {"models": list_models(), "selected": selected}

    @app.post("/api/nodes/{node}/kick")
    def kick_node(node: str, request: Request,
                  auth: AuthContext = Depends(require_admin)):
        _check_default_quota(state, auth)
        now = time.time()
        with state.lifecycle.mutation_section():
            _room_mut_begin(state, auth, now=now)
            info = _release_node_work(state, node)
            state.room.kick(node, store_audit=lambda *a: _audit(state, *a),
                            ip=_client_ip(request))
            log.info("[SCHED] kick %s — %s", node, info)
            return {"ok": True, "released_jobs": info["released"],
                    "cancelled_job": info["cancelled"]}

    @app.post("/api/tunnel/config")
    def tunnel_config(body: TunnelConfigBody,
                      auth: AuthContext = Depends(require_admin)):
        _check_default_quota(state, auth)
        if state.tunnel is None:
            raise ApiError(503, "TUNNEL_UNAVAILABLE", "Tunnel chưa khởi tạo.")
        state.tunnel.configure(mode=body.mode, hostname=body.hostname,
                               token=body.token)
        return {"ok": True, "tunnel": state.tunnel.status()}

    @app.post("/api/tunnel/enable")
    def tunnel_enable(body: TunnelEnableBody | None = None,
                      auth: AuthContext = Depends(require_admin)):
        _check_default_quota(state, auth)
        now = time.time()
        with state.lifecycle.mutation_section():
            _room_mut_begin(state, auth, now=now)
            if state.tunnel is None:
                raise ApiError(
                    503, "TUNNEL_UNAVAILABLE", "Tunnel chưa khởi tạo.")
            allowed = bool(state.settings.get().get("allow_tunnel", True))
            state.tunnel.set_allow_tunnel(allowed)
            st = state.tunnel.enable(
                state.room, mode=(body.mode if body else None),
                hostname=(body.hostname if body else None))
            from tunnel import TUNNEL_WARNING
            return {"ok": True, "tunnel": st, "warning": TUNNEL_WARNING}

    @app.post("/api/tunnel/disable")
    def tunnel_disable(auth: AuthContext = Depends(require_admin)):
        _check_default_quota(state, auth)
        now = time.time()
        with state.lifecycle.mutation_section():
            _room_mut_begin(state, auth, now=now)
            if state.tunnel is None:
                return {"ok": True, "tunnel": {"enabled": False}}
            return {"ok": True, "tunnel": state.tunnel.disable()}

    @app.get("/api/room/status")
    def room_status(request: Request):
        """Công khai tối thiểu. Mã/invite chỉ lộ khi loopback (Host local)."""
        open_ = not state.room.has_password()
        out: dict = {"bootstrap_open": open_}
        ip = _client_ip(request)
        if open_ and state.room.room_code not in (
                "", "THERMAL-PENDING", "THERMAL-LOCAL"):
            out["code_hint"] = state.room.room_code
        # Host vào lại sau logout: chỉ máy local thấy invite (không lộ ra mạng)
        if not open_ and _is_loopback(ip) and state.room.room_code:
            inv = _invite_urls(state)
            out["code"] = state.room.room_code
            out["invite_lan"] = inv.get("invite_lan")
        return out

    @app.post("/api/room/bootstrap")
    def room_bootstrap(body: RoomBootstrapBody, request: Request):
        """Tạo/cấu hình phòng Host. Chưa mật khẩu → chỉ loopback;
        đã có mật khẩu → cần admin. room.json không chứa plaintext password."""
        ip = _client_ip(request)
        if state.room.has_password():
            auth = require_admin(require_auth(request))
            _check_default_quota(state, auth)
        elif not _is_loopback(ip):
            raise ApiError(
                403, "FORBIDDEN",
                "Tạo phòng lần đầu chỉ từ máy Host (localhost).")
        room_info = _apply_room_bootstrap(
            state, body,
            store_audit=lambda *a: _audit(state, *a), ip=ip)
        _audit(state, time.time(), ip, "room_bootstrap",
               {"code": room_info["code"],
                "display_name": room_info.get("display_name")})
        return {"ok": True, "room": room_info}

    @app.post("/api/room/close")
    def room_close(request: Request,
                   auth: AuthContext = Depends(require_admin)):
        """Đóng phòng: thu hồi token + tắt tunnel (docs/10 §1)."""
        _check_default_quota(state, auth)
        now = time.time()
        ip = _client_ip(request)
        revoked = {"n": 0}

        def finish():
            revoked["n"] = state.room.lifecycle_close(
                now=now,
                store_audit=lambda *a: _audit(state, *a),
                ip=ip)

        _run_room_lifecycle(
            state, reason="close", now=now, finish_fn=finish)
        n = revoked["n"]
        tunnel_st = None
        if state.tunnel is not None:
            tunnel_st = state.tunnel.disable()
        state.room.credentials_weak = False
        _persist_room_meta(state)
        _invalidate_state_cache(state)
        log.info("[ROOM] closed — revoked=%d", n)
        return {"ok": True, "revoked": n, "tunnel": tunnel_st,
                "room": _room_public_payload(state, now)}

    @app.post("/api/local-agent/start")
    def local_agent_start(request: Request,
                          body: LocalAgentStartBody | None = None,
                          auth: AuthContext = Depends(require_admin)):
        """Retry the local Host worker only from the physical Host machine."""
        _check_default_quota(state, auth)
        if not _is_loopback(_client_ip(request)):
            raise ApiError(403, "FORBIDDEN",
                           "Chỉ máy Host cục bộ được khởi chạy NodeAgent.")
        worker_password = body.worker_password if body is not None else None
        if (worker_password is not None
                and len(worker_password.strip()) < PASSWORD_MIN_LENGTH):
            raise ApiError(400, "BAD_REQUEST",
                           "Mật khẩu worker phải ≥12 ký tự Unicode.")
        # An operator retrying with the worker password is asking for a worker
        # that works now, so this is the one path allowed to replace a running
        # agent instead of only handing it new credentials.
        status = _launch_local_agent(
            state, worker_password=worker_password,
            replace_process=worker_password is not None)
        return {"ok": True, "local_agent": status}

    @app.post("/api/room/reopen-bootstrap")
    def room_reopen_bootstrap(request: Request):
        """Mở lại tạo phòng từ Host (loopback) — không cần admin.

        Trust = truy cập vật lý máy Host (tương đương restart --bootstrap-open).
        Thu hồi token, xóa hash mật khẩu, tắt tunnel.
        """
        ip = _client_ip(request)
        if not _is_loopback(ip):
            raise ApiError(
                403, "FORBIDDEN",
                "Mở lại tạo phòng chỉ từ máy Host (localhost).")
        now = time.time()
        revoked = {"n": 0}

        def finish():
            revoked["n"] = state.room.lifecycle_close(
                now=now,
                store_audit=lambda *a: _audit(state, *a), ip=ip)

        _run_room_lifecycle(
            state, reason="reopen_bootstrap", now=now, finish_fn=finish)
        n = revoked["n"]
        tunnel_st = None
        if state.tunnel is not None:
            tunnel_st = state.tunnel.disable()
        state.room.credentials_weak = False
        _persist_room_meta(state)
        _invalidate_state_cache(state)
        _audit(state, now, ip, "room_reopen_bootstrap",
               {"revoked": n})
        log.info("[ROOM] reopen-bootstrap — revoked=%d", n)
        return {
            "ok": True,
            "revoked": n,
            "tunnel": tunnel_st,
            "bootstrap_open": not state.room.has_password(),
            "room": _room_public_payload(state, now),
        }

    @app.get("/api/room/invite")
    def room_invite(auth: AuthContext = Depends(require_admin)):
        """Invite URLs for LAN/tunnel — QR is generated client-side."""
        _check_default_quota(state, auth)
        return {
            "ok": True,
            "room": _room_public_payload(state, time.time()),
        }

    @app.post("/api/weather/reading")
    def weather_reading(body: WeatherReadingBody,
                        auth: AuthContext = Depends(require_admin)):
        """Ingest browser OWM reading (API key stays in localStorage)."""
        _check_default_quota(state, auth)
        if state.weather is None:
            raise ApiError(503, "WEATHER_UNAVAILABLE",
                           "Weather service is not available.")
        # Never accept or log an API key field if a client sends one.
        raw = body.model_dump()
        if "api_key" in raw or "appid" in raw:
            raise ApiError(422, "BAD_REQUEST",
                           "API key must not be sent to the server.")
        now = time.time()
        src = (body.source or "owm_gps").strip() or "owm_gps"
        if src not in ("owm_gps", "owm_manual", "cache"):
            src = "owm_gps"
        entry = state.weather.ingest_client_reading(
            temp_c=body.temp_c,
            feels_like_c=body.feels_like_c,
            lat=body.lat,
            lon=body.lon,
            now=now,
            site_id=(body.site_id or "gps-host").strip() or "gps-host",
            label=body.label,
            source=src,
        )
        apply_weather_to_esg(
            state.weather, state.esg.config,
            site_id=entry["site_id"], now=now)
        _invalidate_state_cache(state)
        return {
            "ok": True,
            "reading": {
                "site_id": entry["site_id"],
                "temp_c": entry["temp_c"],
                "feels_like_c": entry["feels_like_c"],
                "lat": entry["lat"],
                "lon": entry["lon"],
                "label": entry["label"],
                "source": entry["client_source"],
                "updated_at": entry["updated_at"],
            },
        }

    @app.get("/api/esg.csv")
    def esg_csv(auth: AuthContext = Depends(require_admin)):
        """Nhật ký sự kiện ESG — cột phẳng docs/03."""
        import csv
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow([
            "ts", "node", "event", "threshold_at_time", "predicted_max",
            "tokens_out", "energy_j", "scheduler_mode"])
        for ev in state.store.esg_events():
            d = ev.get("detail") or {}
            if not isinstance(d, dict):
                d = {}
            w.writerow([
                ev.get("ts", ""),
                _csv_safe_cell(ev.get("node", "")),
                _csv_safe_cell(ev.get("event_type", "")),
                d.get("threshold_at_time", ""),
                d.get("predicted_max", d.get("pred", "")),
                d.get("tokens_out", ""),
                d.get("energy_j", ""),
                _csv_safe_cell(d.get("scheduler_mode", "")),
            ])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition":
                                 "attachment; filename=esg_events.csv"})

    @app.get("/api/audit.csv")
    def audit_csv(auth: AuthContext = Depends(require_admin)):
        """Export operational audit metadata; store already strips secrets."""
        _check_default_quota(state, auth)
        return Response(
            state.store.room_audit_csv(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition":
                     "attachment; filename=room-audit.csv"},
        )

    @app.get("/api/esg/report.csv")
    def esg_report_csv(auth: AuthContext = Depends(require_admin),
                       from_ts: float | None = None,
                       to_ts: float | None = None):
        r = compute_report(state.store.esg_events(), state.esg.config,
                           time.time(), from_ts=from_ts, to_ts=to_ts)
        t1 = r["tier1_measured"]
        t2 = r["tier2_derived"]
        t3 = r["tier3_projected"]
        import csv
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["tier", "metric", "value"])
        w.writerow(["1", "j_per_token", t1.get("j_per_token")])
        w.writerow(["1", "improvement_pct", t1.get("improvement_pct")])
        w.writerow(["1", "samples", t1.get("samples")])
        w.writerow(["1", "confidence", t1.get("confidence")])
        w.writerow(["1", "latency_median_ms", t1.get("latency_median_ms")])
        w.writerow(["1", "latency_p95_ms", t1.get("latency_p95_ms")])
        w.writerow(["2", "kwh_leakage_saved", t2.get("kwh_leakage_saved")])
        w.writerow(["2", "kwh_fan_saved", t2.get("kwh_fan_saved")])
        w.writerow(["2", "kwh_cooling_saved", t2.get("kwh_cooling_saved")])
        w.writerow(["3", "kwh_projected_measured",
                    t3.get("kwh_projected_measured")])
        w.writerow(["3", "kwh_projected_derived",
                    t3.get("kwh_projected_derived")])
        w.writerow(["3", "vnd_projected_measured",
                    t3.get("vnd_projected_measured")])
        w.writerow(["3", "vnd_projected_derived",
                    t3.get("vnd_projected_derived")])
        w.writerow(["3", "scale_note",
                    _csv_safe_cell(t3.get("scale_note", ""))])
        w.writerow(["3", "label_measured", t3.get("label_measured")])
        w.writerow(["3", "label_derived", t3.get("label_derived")])
        if r.get("segments"):
            w.writerow(["meta", "segment_count", len(r["segments"])])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition":
                                 "attachment; filename=esg_report.csv"})

    def _dashboard_html():
        path = os.path.join(os.path.dirname(__file__), "static",
                            "dashboard.html")
        return FileResponse(path)

    @app.get("/dashboard.html")
    @app.get("/")
    def index():
        return _dashboard_html()

    @app.get("/landing")
    def landing_page():
        return _dashboard_html()

    @app.get("/login")
    def login_page():
        return _dashboard_html()

    @app.get("/app")
    def app_page():
        return _dashboard_html()

    @app.get("/business.html")
    @app.get("/business")
    def business_page():
        return FileResponse(os.path.join(os.path.dirname(__file__), "static",
                                         "business.html"))

    @app.get("/night-shift.html")
    @app.get("/night-shift")
    def night_shift_page():
        return FileResponse(os.path.join(os.path.dirname(__file__), "static",
                                         "night-shift.html"))

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        # Không nhận token trên query (tránh lộ access/proxy log).
        ip = (sock.client.host if sock.client else "unknown") or "unknown"
        await sock.accept()
        with state._ws_unauth_lock:
            n = state._ws_unauth_by_ip.get(ip, 0)
            if n >= WS_UNAUTH_PER_IP:
                await sock.close(code=4408)
                return
            state._ws_unauth_by_ip[ip] = n + 1
        slot_released = False
        auth_slot = False
        auth_th = None

        def _release_unauth_slot():
            nonlocal slot_released
            if slot_released:
                return
            slot_released = True
            with state._ws_unauth_lock:
                n = max(0, state._ws_unauth_by_ip.get(ip, 1) - 1)
                if n == 0:
                    state._ws_unauth_by_ip.pop(ip, None)
                else:
                    state._ws_unauth_by_ip[ip] = n

        def _release_auth_slot():
            nonlocal auth_slot
            if not auth_slot or auth_th is None:
                return
            auth_slot = False
            with state._ws_auth_lock:
                tn = max(0, state._ws_auth_by_token.get(auth_th, 1) - 1)
                if tn == 0:
                    state._ws_auth_by_token.pop(auth_th, None)
                else:
                    state._ws_auth_by_token[auth_th] = tn
                ipn = max(0, state._ws_auth_by_ip.get(ip, 1) - 1)
                if ipn == 0:
                    state._ws_auth_by_ip.pop(ip, None)
                else:
                    state._ws_auth_by_ip[ip] = ipn

        try:
            try:
                raw = await asyncio.wait_for(sock.receive_json(), timeout=5.0)
            except Exception:
                await sock.close(code=4401)
                return
            token = None
            if isinstance(raw, dict) and raw.get("type") == "auth":
                token = raw.get("token")
            try:
                auth = state.room.resolve(token)
            except ApiError as e:
                await sock.send_json(e.body())
                await sock.close(code=4401)
                return
            auth_th = auth.token_hash
            with state._ws_auth_lock:
                tn = state._ws_auth_by_token.get(auth_th, 0)
                ipn = state._ws_auth_by_ip.get(ip, 0)
                if tn >= WS_AUTH_PER_TOKEN or ipn >= WS_AUTH_PER_IP:
                    await sock.close(code=4408)
                    return
                state._ws_auth_by_token[auth_th] = tn + 1
                state._ws_auth_by_ip[ip] = ipn + 1
                auth_slot = True
            _release_unauth_slot()
            compact = auth.role == "worker"
            th = auth.token_hash
            gen = auth.generation
            admin_sub_id = None
            admin_queue = None
            if auth.role == "admin":
                admin_sub_id, admin_queue = _subscribe_admin_events(state)
            try:
                while True:
                    now = time.time()
                    if not state.room.token_alive(th, gen, now=now):
                        await sock.close(code=4401)
                        return
                    await sock.send_json(
                        _cached_state_payload(state, now, compact=compact))
                    if compact:
                        await asyncio.sleep(2.0)
                        continue
                    # Admin receives node_event immediately while retaining a
                    # 2-second snapshot heartbeat as a self-healing fallback.
                    event_task = asyncio.create_task(admin_queue.get())
                    sleep_task = asyncio.create_task(asyncio.sleep(2.0))
                    done, pending = await asyncio.wait(
                        (event_task, sleep_task),
                        return_when=asyncio.FIRST_COMPLETED)
                    for task in pending:
                        task.cancel()
                    if event_task in done:
                        await sock.send_json(event_task.result())
            except WebSocketDisconnect:
                pass
            finally:
                if admin_sub_id is not None:
                    _unsubscribe_admin_events(state, admin_sub_id)
        finally:
            _release_auth_slot()
            _release_unauth_slot()

    return app

async def _forecast_loop(state):
    while True:
        try:
            run_forecast_cycle(state, time.time())
        except Exception:
            log.exception("[FORECAST] cycle failed")
        await asyncio.sleep(FORECAST_EVERY_S)


async def _weather_loop(state):
    """Vòng thời tiết 15 phút — fetch trong thread, không chặn event loop."""
    while True:
        try:
            await asyncio.to_thread(refresh_weather, state, time.time())
        except Exception:
            log.exception("[WEATHER] loop failed")
        await asyncio.sleep(WEATHER_EVERY_S)


async def _wal_loop(state):
    """Checkpoint WAL định kỳ — chạy cả calibrate mode."""
    while True:
        try:
            state.store.maybe_wal_checkpoint(time.time())
        except Exception:
            log.exception("[STORE] WAL loop failed")
        await asyncio.sleep(300.0)


async def _tunnel_idle_loop(state):
    """Tự tắt tunnel sau 8 giờ không hoạt động."""
    while True:
        try:
            if state.tunnel is not None:
                state.tunnel.check_idle()
        except Exception:
            log.exception("[TUNNEL] idle check failed")
        await asyncio.sleep(TUNNEL_IDLE_CHECK_S)

async def _scheduler_loop(state):
    while True:
        try:
            run_scheduler_cycle(state, time.time())
        except Exception:
            log.exception("[SCHED] cycle failed")
        await asyncio.sleep(SCHEDULE_EVERY_S)

def host_claim_once(state, now: float | None = None):
    """Claim host dưới read-lock — linearizable revalidate → claim.

    Seam `_before_host_claim_hook` chạy trong lock (trước claim). Hook test
    chỉ được start lifecycle ở thread khác rồi chờ writer_waiting — không
    gọi run() đồng bộ (deadlock vì reader đang giữ).
    """
    now = time.time() if now is None else now
    with state.lifecycle.mutation_section():
        state.lifecycle.ensure_commit_allowed()
        hook = state.lifecycle._before_host_claim_hook
        if hook is not None:
            state.lifecycle._before_host_claim_hook = None
            hook()
            state.lifecycle.ensure_commit_allowed()
        if state.settings.get().get("jobs_paused"):
            job = None  # all jobs stopped by admin
        else:
            job = state.balancer.claim(HOST_NODE, now=now)
        if job is not None and (job.get("type") or "burn") == "chat":
            state.chat.mark_running(job["id"], "host", now=now)
        return job


def _host_apply_result(state, body: dict, now: float) -> None:
    """Commit kết quả host dưới read-lock (không giữ qua LLM)."""
    with state.lifecycle.mutation_section():
        state.lifecycle.ensure_commit_allowed()
        apply_job_result(state, HOST_NODE, body, now)


async def _host_infer_loop(state):
    """P2: claim job target=__host__ và suy luận cục bộ."""
    while True:
        try:
            try:
                job = host_claim_once(state)
            except ApiError as e:
                if e.code == "TOKEN_REVOKED":
                    await asyncio.sleep(0.05)
                    continue
                raise
            if job is None:
                await asyncio.sleep(0.2)
                continue
            if (job.get("type") or "burn") != "chat":
                try:
                    _host_apply_result(state, {
                        "job_id": job["id"], "status": "error",
                        "error_message": "host chỉ chạy job chat",
                        "energy_source": "none",
                    }, time.time())
                except ApiError:
                    pass
                continue
            prompt = job.get("prompt") or ""
            params = job.get("params") or {}
            t0 = time.perf_counter()
            try:
                reply = await asyncio.to_thread(
                    state.llm.generate, prompt, params)
                dur_ms = (time.perf_counter() - t0) * 1000.0
                try:
                    _host_apply_result(state, {
                        "job_id": job["id"],
                        "status": "ok",
                        "text": reply.text,
                        "tokens_in": reply.tokens_in,
                        "tokens_out": reply.tokens_out,
                        "tokens_estimated": reply.tokens_estimated,
                        "duration_ms": dur_ms,
                        "prompt_eval_ms": reply.prompt_ms,
                        "energy_source": "none",
                        "energy_j": None,
                    }, time.time())
                except ApiError:
                    pass
            except Exception as e:
                log.exception("[LLM] host infer failed job %s", job["id"])
                try:
                    _host_apply_result(state, {
                        "job_id": job["id"],
                        "status": "error",
                        "error_message": str(e)[:512],
                        "energy_source": "none",
                    }, time.time())
                except ApiError:
                    pass
        except Exception:
            log.exception("[LLM] host loop error")
            await asyncio.sleep(0.5)

async def _generator_loop(state):
    while True:
        job = state.balancer.enqueue_job(duration_s=10, cores=0)
        if job:
            log.info("[GENERATOR] enqueued job %s", job["id"])
        await asyncio.sleep(GENERATOR_EVERY_S)

def run_server(application, *, host: str = "0.0.0.0",
               port: int | None = None):
    """Entrypoint chạy uvicorn — test monkeypatch được."""
    import uvicorn
    from data_paths import resolve_data_dir
    from runtime_lock import HostInstanceLock
    lock = HostInstanceLock(os.path.join(resolve_data_dir(), "host.lock"))
    lock.acquire()
    try:
        uvicorn.run(application, host=host,
                    port=SERVER_PORT if port is None else int(port))
    finally:
        lock.release()

if __name__ == "__main__":
    import argparse
    import secrets as _secrets
    import sys
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true",
                        help="record a calibration load-cycle run (see calibrate.py)")
    parser.add_argument("--demo-load", action="store_true",
                        help="opt-in: enqueue burn jobs every 4s (off by default)")
    parser.add_argument("--ab-benchmark", action="store_true",
                        help="chạy A/B ESG 6 bước trên cụm giả 10 node (docs/07)")
    parser.add_argument("--cooldown-s", type=float, default=600.0,
                        help="nghỉ giữa 2 lượt A/B (mặc định 600s = 10 phút)")
    parser.add_argument("--ab-prompts", default=None,
                        help="đường dẫn bộ prompt cố định cho --ab-benchmark")
    parser.add_argument("--room-password", default=None,
                        help="mật khẩu worker (khuyến nghị: THERMAL_WORKER_PASSWORD)")
    parser.add_argument("--admin-password", default=None,
                        help="mật khẩu admin (khuyến nghị: THERMAL_ADMIN_PASSWORD)")
    parser.add_argument("--bootstrap-open", action="store_true",
                        help="chưa đặt mật khẩu — chờ POST /api/room/bootstrap")
    parser.add_argument("--write-credentials-once", action="store_true",
                        help="ghi mật khẩu một lần vào LOCALAPPDATA/config "
                             "(không ghi cạnh server/)")
    args = parser.parse_args()
    if args.ab_benchmark:
        from ab_benchmark import default_prompts_path, run_ab_benchmark
        from esg import load_esg_config
        prompts = args.ab_prompts or default_prompts_path()
        cfg = load_esg_config(
            os.path.join(os.path.dirname(__file__), "esg_config.json"))
        run_ab_benchmark(prompts, cooldown_s=args.cooldown_s,
                         esg_config=cfg)
        sys.exit(0)
    wp = (os.environ.get("THERMAL_WORKER_PASSWORD")
          or args.room_password)
    ap = (os.environ.get("THERMAL_ADMIN_PASSWORD")
          or args.admin_password)
    if args.room_password or args.admin_password:
        print("[ROOM] Cảnh báo: mật khẩu trên argv dễ lộ Process Explorer "
              "— dùng THERMAL_WORKER_PASSWORD / THERMAL_ADMIN_PASSWORD.")
    # Không có env/argv → landing tạo phòng (bootstrap-open)
    use_open = args.bootstrap_open or (wp is None and ap is None)
    if use_open:
        print(f"[ROOM] bootstrap-open — mở http://127.0.0.1:{SERVER_PORT}/ "
              "để tạo phòng (chỉ localhost)")
        app_state = make_state(
            calibrate=args.calibrate, demo_load=args.demo_load,
            bootstrap_open=True)
        application = create_app(app_state)
        run_server(application)
        sys.exit(0)
    weak = False
    if wp is None:
        wp = _secrets.token_urlsafe(12)
    if ap is None:
        ap = _secrets.token_urlsafe(12)
    print(f"[ROOM] code={DEFAULT_ROOM_CODE}")
    if args.write_credentials_once:
        local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        cred_dir = os.path.join(local, "ThermalOrchestrator", "config")
        try:
            os.makedirs(cred_dir, exist_ok=True)
            cred_path = os.path.join(cred_dir, "room_credentials.txt")
            with open(cred_path, "w", encoding="utf-8") as cf:
                cf.write(f"room_code={DEFAULT_ROOM_CODE}\n")
                cf.write(f"worker_password={wp}\n")
                cf.write(f"admin_password={ap}\n")
            print(f"[ROOM] credentials (một lần) → {cred_path}")
            print("[ROOM] Xóa tệp sau khi gửi mật khẩu qua kênh riêng.")
        except OSError:
            print("[ROOM] không ghi được credentials — đặt bằng env")
    else:
        print("[ROOM] Mật khẩu trong RAM/env — không ghi disk. "
              "Dùng --write-credentials-once nếu cần tệp tạm.")
    app_state = make_state(
        calibrate=args.calibrate, demo_load=args.demo_load,
        room_password=wp, admin_password=ap, credentials_weak=weak)
    application = create_app(app_state)
    run_server(application)
