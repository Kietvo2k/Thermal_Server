"""Scheduler: lọc → chấm điểm → giữ chỗ. Sửa S1/M8/M11/M14.

Khi nhận ForecastCache: mutate inflight/recently_failed qua API cache.
Khi nhận dict (test): mutate trực tiếp như trước.
Không chạm SQLite. Thời gian nhận qua tham số `now`.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("scheduler")

RESERVATION_TIMEOUT_S = 20.0
MAX_ATTEMPTS = 3
FAILED_PENALTY_WINDOW_S = 120.0
STALE_AFTER_S = 10.0
SCHEDULE_EVERY_S = 1.0

DEFAULT_WEIGHTS = {
    "cool": 0.40,
    "idle": 0.25,
    "power": 0.15,
    "load": 0.20,
}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def redistribute(weights: dict, drop: str = "power") -> dict:
    """Phân bổ lại trọng số khi bỏ một thành phần — tổng vẫn = 1."""
    w = dict(weights)
    dropped = w.get(drop, 0.0)
    remain = 1.0 - dropped
    w[drop] = 0.0
    if remain <= 0:
        return w
    for k in w:
        if k != drop:
            w[k] = w[k] / remain
    return w


def meets_capability(f, job) -> bool:
    """G2: burn/chat stub — luôn đủ trừ khi job yêu cầu RAM cụ thể."""
    need = (job or {}).get("min_ram_gb")
    if need is None:
        return True
    have = getattr(f, "ram_gb", None)
    if have is None:
        return True
    return have >= need


def _is_forecast_cache(obj) -> bool:
    return hasattr(obj, "all") and hasattr(obj, "adjust_inflight")


def _snapshot(cache_or_fc) -> dict:
    if _is_forecast_cache(cache_or_fc):
        return cache_or_fc.all()
    return cache_or_fc


def filter_candidates(cache: dict, job: dict, now: float, *,
                      skip_flagged: bool = False):
    """Giai đoạn 1. Trả về (ứng_viên, lý_do_loại).

    skip_flagged=True ở round_robin — node AT_RISK vẫn nhận job (đối chứng A/B).
    recently_failed chỉ phạt điểm (§4.6), không lọc (quyết định G2).
    model_ready chỉ bắt buộc với job chat.
    """
    job_type = (job or {}).get("type") or "burn"
    candidates, rejected = [], {}
    for node, f in cache.items():
        at_risk = f.state == "AT_RISK" or getattr(f, "flagged", False)
        # thermal_aware: loại flagged trước state chung (F1 → mã flagged)
        if at_risk and not skip_flagged:
            pred = f.predicted_max_c
            if pred is not None:
                rejected[node] = f"flagged (dự báo {pred:.1f}°C)"
            else:
                rejected[node] = "flagged"
            continue
        # round_robin: AT_RISK được phép; các state khác vẫn loại
        allowed = ("READY", "WARMING_UP")
        if skip_flagged:
            allowed = ("READY", "WARMING_UP", "AT_RISK")
        if f.state not in allowed:
            rejected[node] = f"state_{f.state.lower()}"
        elif getattr(f, "user_active", False):
            # Step 4a: a staff member is using this PC — no new jobs.
            rejected[node] = "user_active"
        elif f.inflight >= f.max_concurrent:
            rejected[node] = f"busy ({f.inflight}/{f.max_concurrent})"
        elif job_type == "chat" and not f.model_ready:
            rejected[node] = "model_not_ready"
        elif not meets_capability(f, job):
            rejected[node] = "insufficient_capability"
        elif (now - f.last_sample_ts) > STALE_AFTER_S:
            age = now - f.last_sample_ts
            rejected[node] = f"stale_telemetry ({age:.0f}s)"
        else:
            candidates.append(node)
    # A retried chat must prefer a different live node.  We only exclude a
    # previously failed node when another eligible candidate exists, so a
    # single-node room remains recoverable after a transient error.
    if job_type == "chat" and len(candidates) > 1:
        failed_nodes = set(job.get("failed_nodes") or ())
        fresh = [node for node in candidates if node not in failed_nodes]
        if fresh:
            for node in candidates:
                if node in failed_nodes:
                    rejected[node] = "failed_previous_attempt"
            candidates = fresh
    return candidates, rejected


def score_node(f, weights: dict, now: float):
    """Giai đoạn 2. Trả về (điểm, các_thành_phần). Headroom tuyệt đối — M11."""
    has_temp = (f.predicted_max_c is not None or f.current_temp_c is not None)
    temp = (f.predicted_max_c if f.predicted_max_c is not None
            else f.current_temp_c)
    span = max(f.effective_threshold_c - f.idle_baseline_c, 1.0)
    if not has_temp:
        # Không cảm biến nhiệt — trung lập, không được điểm headroom tối đa
        headroom = 0.5
    else:
        headroom = clamp((f.effective_threshold_c - temp) / span, 0.0, 1.0)

    idleness = 1.0 - clamp((f.cpu_util or 0.0) / 100.0, 0.0, 1.0)

    w = dict(weights)
    if f.power_source == "none":
        efficiency = 0.5
        w = redistribute(w, drop="power")
    else:
        p_span = max(f.p_max_w - f.p_idle_w, 1.0)
        pw = f.power_w if f.power_w is not None else f.p_idle_w
        efficiency = 1.0 - clamp((pw - f.p_idle_w) / p_span, 0.0, 1.0)

    availability = 1.0 - clamp(
        f.inflight / max(f.max_concurrent, 1), 0.0, 1.0)

    base = (w["cool"] * headroom
            + w["idle"] * idleness
            + w["power"] * efficiency
            + w["load"] * availability)

    penalty = 1.0
    if f.state == "WARMING_UP":
        penalty *= 0.7
    if now < f.recently_failed_until:
        penalty *= 0.5
    if f.consecutive_errors >= 2:
        penalty *= 0.3

    return base * penalty, {
        "headroom": headroom,
        "idleness": idleness,
        "efficiency": efficiency,
        "availability": availability,
        "penalty": penalty,
        "weights": w,
    }


def fmt_rejected(rejected: dict) -> str:
    if not rejected:
        return "(không)"
    return "; ".join(f"{n}: {r}" for n, r in sorted(rejected.items()))


class RoundRobinCursor:
    """Con trỏ xoay vòng cho chế độ A/B — tất định theo thứ tự tên."""

    def __init__(self):
        self._idx = 0

    def pick(self, candidates: list[str]) -> str:
        if not candidates:
            raise ValueError("no candidates")
        ordered = sorted(candidates)
        self._idx = self._idx % len(ordered)
        winner = ordered[self._idx]
        self._idx = (self._idx + 1) % len(ordered)
        return winner


def schedule_once(cache_or_fc, queue, weights: dict, now: float,
                  *, mode: str = "thermal_aware",
                  reservation_timeout_s: float = RESERVATION_TIMEOUT_S,
                  rr_cursor: RoundRobinCursor | None = None,
                  max_assign: int | None = None):
    """Một lượt scheduler. Gán giữ chỗ cho job pending (target is None).

    max_assign: giới hạn số job gán trong một vòng (None = không giới hạn).
    cache_or_fc: ForecastCache hoặc dict[str, NodeForecast] (test).
    """
    use_api = _is_forecast_cache(cache_or_fc)
    skip_flagged = mode == "round_robin"
    assigned = 0
    for job in list(queue.pending()):
        if max_assign is not None and assigned >= max_assign:
            break
        cache = _snapshot(cache_or_fc)
        candidates, rejected = filter_candidates(
            cache, job, now, skip_flagged=skip_flagged)

        if not candidates:
            # Giữ job pending để worker đang bận có thể lấy lượt kế tiếp.
            # Không có pseudo-host: một phòng một máy vẫn dùng chính máy đó
            # ngay khi nó hoàn thành job hiện tại.
            log.info("[SCHED] job %s chưa có worker khả dụng: %s",
                     job["id"], fmt_rejected(rejected))
            continue

        if mode == "round_robin":
            cursor = rr_cursor or RoundRobinCursor()
            winner = cursor.pick(candidates)
            best_score, parts = score_node(cache[winner], weights, now)
        else:
            scored = []
            for n in candidates:
                sc, parts = score_node(cache[n], weights, now)
                scored.append((sc, parts, n))
            scored.sort(key=lambda x: (-x[0], x[2]))
            best_score, parts, winner = scored[0]

        until = now + reservation_timeout_s
        wf = cache.get(winner)
        assign_temp = getattr(wf, "current_temp_c", None) if wf else None
        # ΔT_tránh (docs/07): nhiệt node bị né − nhiệt node được chọn lúc gán.
        # Chỉ stamp khi ≥2 ứng viên — không bịa ΔT khi chỉ một máy.
        avoided_temp = None
        if len(candidates) >= 2:
            rival_temps = []
            for n in candidates:
                if n == winner:
                    continue
                cf = cache.get(n)
                t = getattr(cf, "current_temp_c", None) if cf else None
                if t is not None:
                    try:
                        rival_temps.append(float(t))
                    except (TypeError, ValueError):
                        pass
            if rival_temps:
                avoided_temp = max(rival_temps)
                log.info("[SCHED] ΔT_tránh stamp job %s avoided=%.1f "
                         "assign=%s",
                         job["id"], avoided_temp,
                         assign_temp if assign_temp is not None else "?")
        reserved = queue.reserve(
            job["id"], winner, until, now=now,
            scheduler_mode=mode,
            assign_temp_c=assign_temp,
            assign_predicted_max_c=(
                getattr(wf, "predicted_max_c", None) if wf else None),
            avoided_temp_c=avoided_temp,
        )
        if reserved is None:
            continue
        if use_api:
            cache_or_fc.adjust_inflight(winner, 1)
        else:
            cache[winner].inflight = cache[winner].inflight + 1
        assigned += 1

        rivals = ""
        if mode == "thermal_aware" and len(candidates) > 1:
            others = [(score_node(cache[n], weights, now)[0], n)
                      for n in candidates if n != winner]
            others.sort(reverse=True)
            if others:
                rivals = (f" đối thủ: {others[0][1]} {others[0][0]:.3f}")

        log.info(
            "[SCHED] job %s -> %s (điểm %.3f) headroom %.2f | rảnh %.2f | "
            "điện %.2f | sẵn sàng %.2f | phạt %.2f | mode=%s | bị loại: %s%s",
            job["id"], winner, best_score, parts["headroom"], parts["idleness"],
            parts["efficiency"], parts["availability"], parts["penalty"],
            mode, fmt_rejected(rejected), rivals)


def reap_expired(cache_or_fc, queue, now: float):
    """Vòng thu hồi giữ chỗ quá hạn. BẮT BUỘC — ADR-001."""
    use_api = _is_forecast_cache(cache_or_fc)
    for job in list(queue.reserved()):
        until = job.get("reserved_until")
        if until is None or until >= now:
            continue
        node = job.get("target")
        attempts = job.get("attempts", 0)
        log.info("[SCHED] job %s giữ chỗ hết hạn ở %s sau timeout → trả về "
                 "hàng đợi (lần thử %d/%d)",
                 job["id"], node, attempts, MAX_ATTEMPTS)
        if node:
            if use_api:
                live = cache_or_fc.get(node)
                if live is not None:
                    cache_or_fc.adjust_inflight(node, -1)
                    cache_or_fc.set_recently_failed_until(
                        node, now + FAILED_PENALTY_WINDOW_S)
            else:
                cache = cache_or_fc
                if node in cache:
                    cache[node].inflight = max(0, cache[node].inflight - 1)
                    cache[node].recently_failed_until = (
                        now + FAILED_PENALTY_WINDOW_S)
        if attempts >= MAX_ATTEMPTS:
            queue.fail(job["id"], code="NO_CAPACITY",
                       detail=f"hết {MAX_ATTEMPTS} lần giữ chỗ")
        else:
            queue.requeue_front(job["id"])
