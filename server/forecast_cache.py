"""ForecastCache — nguồn sự thật duy nhất cho dự báo và trạng thái node.

Vòng forecast (5s) GHI. Scheduler, dashboard, ESG, /api/state CHỈ ĐỌC
(trừ inflight / recently_failed do scheduler cập nhật).
Không gọi forecaster từ nơi hiển thị — đó là lỗi M6.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

HYSTERESIS_C = 3.0
MIN_DWELL_S = 30.0
MAX_TEMP_MARGIN_C = 40.0
MAX_CONCURRENT_DEFAULT = 1

STATES = ("JOINING", "WARMING_UP", "READY", "AT_RISK", "STALE", "INACTIVE")


@dataclass
class NodeForecast:
    node: str
    state: str = "JOINING"
    predicted_max_c: float | None = None
    delta_t_c: float | None = None
    current_temp_c: float | None = None
    headroom: float | None = None
    cpu_util: float | None = None
    power_w: float | None = None
    power_source: str = "none"  # sensor | model | none
    inflight: int = 0
    max_concurrent: int = MAX_CONCURRENT_DEFAULT
    # A node is never eligible for chat merely because it joined.  The agent
    # must prove that the exact room model/runtime generation is running.
    model_ready: bool = False
    model_id: str | None = None
    model_sha256: str | None = None
    model_generation: int = 0
    runtime_id: str | None = None
    last_sample_ts: float = 0.0
    idle_baseline_c: float = 40.0
    effective_threshold_c: float = 75.0
    p_idle_w: float = 15.0
    p_max_w: float = 65.0
    recently_failed_until: float = 0.0
    consecutive_errors: int = 0
    flagged_since: float | None = None
    state_changed_at: float = 0.0
    computed_at: float = 0.0
    reason: str = ""
    # Step 4a: recent keyboard/mouse input on this PC. Only /ingest sets it.
    user_active: bool = False

    @property
    def flagged(self) -> bool:
        return self.state == "AT_RISK"


class ForecastCache:
    """Thread-safe map node -> NodeForecast. Writers: forecast loop +
    scheduler (inflight / recently_failed). Everyone else reads snapshots."""

    def __init__(self):
        self._lock = threading.Lock()
        self._nodes: dict[str, NodeForecast] = {}

    def get(self, node: str) -> NodeForecast | None:
        with self._lock:
            f = self._nodes.get(node)
            return None if f is None else _copy(f)

    def all(self) -> dict[str, NodeForecast]:
        with self._lock:
            return {k: _copy(v) for k, v in self._nodes.items()}

    def put(self, forecast: NodeForecast) -> None:
        """Ghi forecast. Giữ inflight / recently_failed / consecutive_errors
        đang sống — tránh vòng forecast đè quyết định scheduler."""
        with self._lock:
            prev = self._nodes.get(forecast.node)
            nf = _copy(forecast)
            if prev is not None:
                nf.inflight = prev.inflight
                nf.user_active = prev.user_active
                nf.recently_failed_until = prev.recently_failed_until
                nf.consecutive_errors = prev.consecutive_errors
            self._nodes[forecast.node] = nf

    def patch(self, node: str, **fields) -> None:
        """Cập nhật từng field dưới lock — không thay cả hàng."""
        with self._lock:
            f = self._nodes.get(node)
            if f is None:
                return
            for k, v in fields.items():
                if k in f.__dataclass_fields__:
                    setattr(f, k, v)

    def mark_inactive(self, node: str, now: float, reason: str = "kicked") -> None:
        with self._lock:
            prev = self._nodes.get(node)
            self._nodes[node] = NodeForecast(
                node=node, state="INACTIVE",
                state_changed_at=now, computed_at=now,
                last_sample_ts=now, reason=reason,
                inflight=0,
                recently_failed_until=(
                    prev.recently_failed_until if prev else 0.0),
            )
    def remove(self, node: str) -> None:
        with self._lock:
            self._nodes.pop(node, None)

    def set_inflight(self, node: str, inflight: int) -> None:
        with self._lock:
            f = self._nodes.get(node)
            if f is not None:
                f.inflight = max(0, inflight)

    def adjust_inflight(self, node: str, delta: int) -> int:
        with self._lock:
            f = self._nodes.get(node)
            if f is None:
                return 0
            f.inflight = max(0, f.inflight + delta)
            return f.inflight

    def set_recently_failed_until(self, node: str, until: float) -> None:
        with self._lock:
            f = self._nodes.get(node)
            if f is not None:
                f.recently_failed_until = until

    def mark_joining(self, node: str, now: float) -> None:
        with self._lock:
            self._nodes[node] = NodeForecast(
                node=node, state="JOINING",
                state_changed_at=now, computed_at=now,
                last_sample_ts=now,
                reason="joined room; waiting for LLM readiness",
                model_ready=False)

    def mark_warming_up(self, node: str, now: float, *,
                        model_ready: bool = False,
                        model_id: str | None = None,
                        model_sha256: str | None = None,
                        model_generation: int = 0,
                        runtime_id: str | None = None) -> None:
        with self._lock:
            prev = self._nodes.get(node)
            self._nodes[node] = NodeForecast(
                node=node, state="WARMING_UP",
                state_changed_at=now, computed_at=now,
                last_sample_ts=now,
                reason=("model ready, warming up" if model_ready
                        else "joined; LLM runtime not ready"),
                current_temp_c=prev.current_temp_c if prev else None,
                idle_baseline_c=prev.idle_baseline_c if prev else 40.0,
                effective_threshold_c=(
                    prev.effective_threshold_c if prev else 75.0),
                inflight=prev.inflight if prev else 0,
                recently_failed_until=(
                    prev.recently_failed_until if prev else 0.0),
                consecutive_errors=prev.consecutive_errors if prev else 0,
                user_active=prev.user_active if prev else False,
                model_ready=bool(model_ready),
                model_id=model_id,
                model_sha256=model_sha256,
                model_generation=int(model_generation),
                runtime_id=runtime_id)

    def reset_model_readiness(self, generation: int) -> None:
        """Invalidate every node before a room model switch becomes visible."""
        with self._lock:
            for forecast in self._nodes.values():
                forecast.model_ready = False
                forecast.model_id = None
                forecast.model_sha256 = None
                forecast.model_generation = int(generation)
                forecast.runtime_id = None
                forecast.reason = "selected room model changed; waiting for agent"


def effective_threshold(cluster_threshold_c: float, idle_baseline_c: float,
                        margin_c: float = MAX_TEMP_MARGIN_C) -> float:
    return min(cluster_threshold_c, idle_baseline_c + margin_c)


def idle_baseline_from_samples(samples, fallback: float = 40.0) -> float:
    """Phân vị 5% nhiệt trong cửa sổ — khi chưa có hiệu chuẩn."""
    temps = sorted(
        s["cpu_temp"] for s in samples
        if s.get("cpu_temp") is not None)
    if not temps:
        return fallback
    idx = max(0, int(len(temps) * 0.05) - (0 if len(temps) > 1 else 0))
    # nearest-rank 5th percentile
    idx = min(len(temps) - 1, max(0, int(round(0.05 * (len(temps) - 1)))))
    return float(temps[idx])


def _copy(f: NodeForecast) -> NodeForecast:
    return NodeForecast(**{k: getattr(f, k) for k in f.__dataclass_fields__})


def apply_hysteresis(prev: NodeForecast | None, *,
                     base_state: str,
                     pred: float | None,
                     threshold_c: float,
                     now: float) -> tuple[str, float, str]:
    """Decide READY vs AT_RISK given hysteresis + MIN_DWELL.

    Returns (state, state_changed_at, reason).
    base_state is WARMING_UP / READY / STALE / INACTIVE — never AT_RISK.
    First flag from JOINING/WARMING_UP (or empty cache) is immediate; re-flag
    after clear (prev is READY) requires MIN_DWELL via state_changed_at.
    """
    if base_state in ("STALE", "INACTIVE", "JOINING", "WARMING_UP"):
        changed = now if (prev is None or prev.state != base_state) else prev.state_changed_at
        return base_state, changed, f"base={base_state}"

    # base_state == READY — may become AT_RISK
    was_risk = prev is not None and prev.state == "AT_RISK"
    last_change = prev.state_changed_at if prev else now
    dwell_ok = prev is None or (now - last_change) >= MIN_DWELL_S

    if was_risk:
        if pred is not None and pred <= threshold_c - HYSTERESIS_C and dwell_ok:
            return ("READY", now,
                    f"cleared pred={pred:.1f} <= {threshold_c - HYSTERESIS_C:.1f}")
        # stay AT_RISK (in band, or dwell not met, or no pred)
        reason = "still at risk"
        if pred is not None:
            reason = f"stay AT_RISK pred={pred:.1f}"
        return "AT_RISK", last_change, reason

    # not currently AT_RISK
    if pred is not None and pred >= threshold_c:
        # First entry into AT_RISK from JOINING/WARMING_UP (or no cache):
        # allow immediately. Re-flag after clear (prev.state == READY with
        # state_changed_at set at clear) requires MIN_DWELL — do NOT use
        # flagged_since is None (that is always true after clear).
        first_flag = prev is None or prev.state != "READY"
        if first_flag or dwell_ok:
            return ("AT_RISK", now,
                    f"flagged pred={pred:.1f} >= {threshold_c:.1f}")
        return ("READY", last_change,
                f"dwell block flag pred={pred:.1f}")

    changed = now if (prev is None or prev.state != "READY") else last_change
    return "READY", changed, "ready"
