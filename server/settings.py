"""User-editable runtime settings, persisted to settings.json.

Ngưỡng cảnh báo, chế độ scheduler A/B, trọng số chấm điểm, timeout giữ chỗ.
"""
import json
import logging
import math
import os
import re

log = logging.getLogger("settings")

DEFAULTS = {
    "threshold_c": 75.0,
    "scheduler_mode": "thermal_aware",  # thermal_aware | round_robin
    "w_cool": 0.40,
    "w_idle": 0.25,
    "w_power": 0.15,
    "w_load": 0.20,
    "reservation_timeout_s": 20.0,
    "max_temp_margin_c": 40.0,
    # ADR-006 — catalog id; validate against room_assets.MODEL_CATALOG
    "model_id": "qwen2.5-0.5b-instruct-q4_k_m",
    # G6 / docs/06 §3 — IT kill-switch; false vô hiệu hóa tunnel ở tầng mã
    "allow_tunnel": True,
    # site_id cho thời tiết / cooling_factor (M12: không vào score)
    "site_id": "hanoi-office-4f",
    # After-hours window for burn jobs ("HH:MM", Host local time). Both "" = always on.
    "run_window_start": "",
    "run_window_end": "",
    # Admin "Stop all jobs": no job is handed out or claimed while true.
    "jobs_paused": False,
    # Step 4a: input within this many seconds = user active (0 = off). Edit settings.json to tune.
    "user_idle_after_s": 120.0,
}

WEIGHT_KEYS = ("w_cool", "w_idle", "w_power", "w_load")


def _hhmm_minutes(value):
    """"HH:MM" (two digits each) -> minute of day, or None if not valid."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{2}:[0-9]{2}", value):
        return None
    h, m = int(value[:2]), int(value[3:])
    if h > 23 or m > 59:
        return None
    return h * 60 + m


def in_run_window(start, end, minute_of_day):
    """True if burn jobs may run now. Empty start or end = always on."""
    if not start or not end:
        return True
    s, e = _hhmm_minutes(start), _hhmm_minutes(end)
    if s is None or e is None:
        return True
    if s < e:
        return s <= minute_of_day < e
    # Window crosses midnight (e.g. 17:30-06:30)
    return minute_of_day >= s or minute_of_day < e


class Settings:
    def __init__(self, path="settings.json"):
        self.path = path
        self._data = dict(DEFAULTS)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
                self._data.update({k: loaded[k] for k in DEFAULTS if k in loaded})
        # Chuẩn hóa kiểu bool cho allow_tunnel (JSON có thể là 0/1)
        self._data["allow_tunnel"] = bool(self._data.get("allow_tunnel", True))
        self._data["jobs_paused"] = bool(self._data.get("jobs_paused", False))
        if (_hhmm_minutes(self._data.get("run_window_start")) is None
                or _hhmm_minutes(self._data.get("run_window_end")) is None):
            self._data["run_window_start"] = ""
            self._data["run_window_end"] = ""
        self._sanitize_finite_values()
        self._sanitize_model_id()

    def _sanitize_finite_values(self):
        """Không để JSON cũ có NaN/Infinity làm hỏng response lúc khởi động."""
        changed = False
        numeric_keys = (
            "threshold_c", "w_cool", "w_idle", "w_power", "w_load",
            "reservation_timeout_s", "max_temp_margin_c", "user_idle_after_s",
        )
        for key in numeric_keys:
            value = self._data.get(key)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = math.nan
            if isinstance(value, bool) or not math.isfinite(numeric):
                log.warning("[SETTINGS] %s trên disk không hữu hạn → mặc định", key)
                self._data[key] = DEFAULTS[key]
                changed = True
            else:
                self._data[key] = numeric
        if changed and os.path.exists(self.path):
            try:
                self._persist()
            except OSError:
                pass

    def _sanitize_model_id(self):
        """Disk bẩn / id lạ → về default và persist."""
        from room_assets import DEFAULT_MODEL_ID, MODEL_CATALOG
        mid = str(self._data.get("model_id") or "").strip()
        if mid not in MODEL_CATALOG:
            log.warning(
                "[SETTINGS] model_id %r không thuộc catalog → %s",
                mid or "(empty)", DEFAULT_MODEL_ID)
            self._data["model_id"] = DEFAULT_MODEL_ID
            if os.path.exists(self.path) or mid:
                try:
                    self._persist()
                except OSError:
                    pass

    def get(self):
        return dict(self._data)

    def weights(self):
        return {
            "cool": self._data["w_cool"],
            "idle": self._data["w_idle"],
            "power": self._data["w_power"],
            "load": self._data["w_load"],
        }

    def _persist(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f)

    def set_threshold(self, value):
        value = float(value)
        if not math.isfinite(value) or not 40.0 <= value <= 100.0:
            raise ValueError(f"threshold_c must be 40-100, got {value}")
        self._data["threshold_c"] = value
        self._persist()

    def set_scheduler_mode(self, mode):
        if mode not in ("thermal_aware", "round_robin"):
            raise ValueError(f"scheduler_mode invalid: {mode}")
        self._data["scheduler_mode"] = mode
        self._persist()

    def set_weights(self, w_cool, w_idle, w_power, w_load):
        vals = [float(w_cool), float(w_idle), float(w_power), float(w_load)]
        if not all(math.isfinite(value) for value in vals):
            raise ValueError("weights must be finite")
        if abs(sum(vals) - 1.0) > 0.001:
            raise ValueError("weights must sum to 1.0 (±0.001)")
        for k, v in zip(WEIGHT_KEYS, vals):
            if v < 0:
                raise ValueError(f"{k} must be >= 0")
            self._data[k] = v
        self._persist()

    def set_reservation_timeout(self, seconds):
        seconds = float(seconds)
        if not math.isfinite(seconds) or not 5.0 <= seconds <= 300.0:
            raise ValueError("reservation_timeout_s must be 5-300")
        self._data["reservation_timeout_s"] = seconds
        self._persist()

    def set_model_id(self, model_id: str):
        """Persist catalog model_id. Caller validates against MODEL_CATALOG."""
        model_id = str(model_id).strip()
        if not model_id:
            raise ValueError("model_id empty")
        self._data["model_id"] = model_id
        self._persist()

    def set_allow_tunnel(self, allowed: bool):
        self._data["allow_tunnel"] = bool(allowed)
        self._persist()

    def set_site_id(self, site_id: str):
        site_id = str(site_id).strip()
        if not site_id:
            raise ValueError("site_id empty")
        self._data["site_id"] = site_id
        self._persist()

    def set_run_window(self, start, end):
        start = str(start or "").strip()
        end = str(end or "").strip()
        if start or end:
            s, e = _hhmm_minutes(start), _hhmm_minutes(end)
            if s is None or e is None:
                raise ValueError("run window must be two HH:MM times, or both empty")
            if s == e:
                raise ValueError("run window start and end must differ")
        self._data["run_window_start"] = start
        self._data["run_window_end"] = end
        self._persist()

    def set_jobs_paused(self, paused):
        self._data["jobs_paused"] = bool(paused)
        self._persist()
