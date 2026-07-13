"""
Web dashboard — local Flask app for live bet & P&L tracking.

Run:
    python -m src.bot web
    python -m src.bot web --port 8080
"""

import copy
import hashlib
import itertools
import json
import hmac
import logging
import math
import os
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import requests
from flask import Flask, jsonify, make_response, render_template, request

from src.betting_window import bet_window_status
from src.config import (
    ACTIVITY_ALERT_RETENTION_HOURS,
    GEMINI_TRACKER_CONFIDENCE_CAP,
    LOGS_DIR,
)
from src.web.alert_store import (
    ALERT_LEVELS,
    ALERT_STORE_FILENAME,
    load_recent_alerts,
    maybe_prune_alert_store,
)
from src.data.name_utils import canonical_fighter_display_name, normalize_cross_source_name
from src.polymarket.tracker import (
    BetLedger,
    _load_pnl_history,
    auto_reconcile_sold_positions,
    auto_redeem_positions_from_polymarket,
    auto_settle_from_polymarket,
    load_all_trader_ledgers,
    resolve_merged_bet_reference,
)
from src.polymarket.client import ClobClientWrapper
from src.polymarket.monitor import PositionDataPartialError, PositionMonitor

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def _load_fight_matrix_timezone():
    tz_name = str(os.getenv("DASHBOARD_EVENT_TIMEZONE", "America/New_York") or "").strip()
    if tz_name:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            logger.warning("Invalid DASHBOARD_EVENT_TIMEZONE=%s; falling back to UTC", tz_name)
    return timezone.utc


FIGHT_MATRIX_TIMEZONE = _load_fight_matrix_timezone()

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))


@app.after_request
def _apply_security_headers(response):
    """Apply browser protections to both HTML pages and JSON APIs."""
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; img-src 'self' data: https:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
    )
    return response

# Shared state — initialized in start_server()
_clob_client = None
_position_monitor = None
_monitor_lock = threading.Lock()
_server_host = "127.0.0.1"
_runtime_status_lock = threading.Lock()
_runtime_thread_lock = threading.Lock()
_runtime_threads: dict[str, threading.Thread] = {}

# Simple TTL cache for slow endpoints
_endpoint_cache = {}
_endpoint_inflight = {}
_cache_lock = threading.Lock()
_background_cache_refreshes = 0
_MAX_BACKGROUND_CACHE_REFRESHES = 2
_timed_call_lock = threading.Lock()
_timed_call_inflight: dict[str, threading.Event] = {}
SLOW_ENDPOINT_TTL = 300  # 5 minutes
LIMIT_ORDER_DISPLAY_TTL = 30
LIMIT_ORDER_CLOB_TIMEOUT_SECONDS = 8.0
LIMIT_ORDER_MARKET_METADATA_TTL = 300
LIMIT_ORDER_MARKET_LOOKUP_TIMEOUT_SECONDS = 2.0
LIVE_PNL_CACHE_TTL = 5
LIVE_PNL_TIMEOUT_SECONDS = 6.0
UPCOMING_EVENTS_CACHE_TTL = 300
UPCOMING_SNAPSHOT_SCAN_LIMIT = 1000
UPCOMING_SNAPSHOT_MAX_FILES = 2500
UPCOMING_SNAPSHOT_PRUNE_INTERVAL_SECONDS = 3600
TRACKER_DECISIONS_CACHE_TTL = 30
OPEN_BET_DISPLAY_SIZE_THRESHOLD = 0.01
PROFILE_BETS_CACHE_TTL = 30
PROFILE_TRADE_HISTORY_LIMIT = 1000
PROFILE_PNL_EPSILON = 1e-6
PROFILE_PAGE_CACHE_TTL = 30
PROFILE_PAGE_FAILURE_LOG_TTL = 300
PROFILE_PAGE_TIMEOUT_SECONDS = 10.0
BALANCE_CACHE_TTL = 60
GEOBLOCK_STATUS_CACHE_TTL = 60
GEOBLOCK_STATUS_TIMEOUT_SECONDS = 5.0
MARKET_INTEL_FILENAME = "market_intel_latest.json"
MARKET_INTEL_STALE_AFTER_SECONDS = 30 * 60
_last_upcoming_snapshot_prune = 0.0
_PROFILE_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json" crossorigin="anonymous">(.*?)</script>',
    re.DOTALL,
)
_PROFILE_NEXT_F_PUSH_PREFIX = "self.__next_f.push("
_profile_snapshot_warning_state: dict[tuple[str, str], float] = {}


def _sanitize_for_json(obj):
    """Replace NaN/Infinity floats with None so json.dumps produces valid JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj
LOG_LINE_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),?\d*\s*\[(\w+)]\s*([\w.]+):\s*(.*)"
)
HANDLED_ACTIVITY_WARNING_PATTERNS = (
    "trading restricted in your region",
    "clob/geoblock",
    "available regions",
)
RUNTIME_ACTIVITY_ERROR_STATES = {"degraded", "dead", "stale"}
RUNTIME_COMPONENT_LABELS = {
    "ufc_refresh_loop": "UFC Refresh",
    "betting_loop": "Betting Loop",
    "monitor_loop": "Monitor Loop",
    "clob": "CLOB",
    "prediction_refresh": "Prediction Refresh",
    "web": "Web",
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_runtime_status = {
    "service": "ufc-betting-bot",
    "startup_source": "web",
    "requested_live_mode": "off",
    "requested_live_mode_raw": "off",
    "effective_live_mode": "off",
    "trading_enabled": False,
    "trading_live": False,
    "model_name": "xgboost",
    "host": _server_host,
    "public_bind": False,
    "ready": True,
    "errors": [],
    "warnings": [],
    "checks": [],
    "components": {},
    "updated_at": _utcnow_iso(),
}


def get_runtime_status() -> dict:
    with _runtime_status_lock:
        return copy.deepcopy(_runtime_status)


def _parse_runtime_timestamp(value) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def set_runtime_status(status: dict) -> None:
    global _runtime_status
    with _runtime_status_lock:
        merged = copy.deepcopy(status)
        merged.setdefault("service", "ufc-betting-bot")
        merged.setdefault("startup_source", "web")
        merged.setdefault("requested_live_mode", "off")
        merged.setdefault("requested_live_mode_raw", merged["requested_live_mode"])
        merged.setdefault("effective_live_mode", "off")
        merged.setdefault("trading_enabled", False)
        merged.setdefault("trading_live", False)
        merged.setdefault("model_name", "xgboost")
        merged.setdefault("host", _server_host)
        merged.setdefault("public_bind", _dashboard_is_public_bind())
        merged.setdefault("ready", True)
        merged.setdefault("errors", [])
        merged.setdefault("warnings", [])
        merged.setdefault("checks", [])
        merged.setdefault("components", {})
        merged["updated_at"] = _utcnow_iso()
        _runtime_status = merged


def register_runtime_thread(component: str, thread: threading.Thread | None) -> None:
    if thread is None:
        return
    with _runtime_thread_lock:
        _runtime_threads[component] = thread


def _runtime_status_with_liveness() -> dict:
    status = get_runtime_status()
    components = copy.deepcopy(status.get("components") or {})
    errors = list(status.get("errors") or [])
    warnings = list(status.get("warnings") or [])
    now = datetime.now(timezone.utc)
    trading_enabled = bool(status.get("trading_enabled", False))

    with _runtime_thread_lock:
        thread_registry = dict(_runtime_threads)

    for component, payload in components.items():
        entry = dict(payload or {})
        thread = thread_registry.get(component)
        if thread is not None:
            entry["thread_alive"] = bool(thread.is_alive())
            if not thread.is_alive() and entry.get("state") not in {"disabled", "stopped", "starting"}:
                entry["state"] = "dead"
                entry["message"] = entry.get("message") or "Background thread is not alive."

        updated_at = _parse_runtime_timestamp(entry.get("updated_at"))
        stale_after = float(entry.get("stale_after_seconds") or 0.0)
        if updated_at is not None:
            entry["age_seconds"] = round((now - updated_at).total_seconds(), 1)
        if (
            updated_at is not None
            and stale_after > 0
            and (now - updated_at).total_seconds() > stale_after
            and entry.get("state") not in {"disabled", "stopped", "dead"}
        ):
            entry["state"] = "stale"
            entry["message"] = entry.get("message") or "Heartbeat is stale."

        components[component] = entry

    betting_loop = dict(components.get("betting_loop") or {})
    betting_state = str(betting_loop.get("state", "") or "").strip().lower()
    consecutive_failures = int(betting_loop.get("consecutive_failures") or 0)

    critical_loop_issue = False
    if trading_enabled:
        if betting_state in {"dead", "stale"}:
            errors.append(f"betting_loop_{betting_state}")
            critical_loop_issue = True
        elif consecutive_failures >= 3:
            if betting_state == "running":
                betting_loop["state"] = "degraded"
                components["betting_loop"] = betting_loop
            errors.append("betting_loop_repeated_cycle_failures")
            critical_loop_issue = True

    if str((components.get("monitor_loop") or {}).get("state", "")).strip().lower() in {"dead", "stale"}:
        warnings.append("monitor_loop_unhealthy")

    clob_state = str((components.get("clob") or {}).get("state", "")).strip().lower()
    if trading_enabled and clob_state in {"degraded", "dead", "stale", ""}:
        errors.append("clob_not_ready")
        critical_loop_issue = True

    status["components"] = components
    status["errors"] = sorted(set(errors))
    status["warnings"] = sorted(set(warnings))
    status["ready"] = bool(status.get("ready", False)) and not critical_loop_issue
    status["ok"] = not critical_loop_issue
    return status


def update_runtime_component(component: str, state: str, message: str = "", **metadata) -> None:
    global _runtime_status
    with _runtime_status_lock:
        snapshot = copy.deepcopy(_runtime_status)
        components = dict(snapshot.get("components") or {})
        existing = dict(components.get(component) or {})
        existing.update({
            "state": state,
            "message": message,
            "updated_at": _utcnow_iso(),
        })
        if metadata:
            existing.update(metadata)
        components[component] = existing
        snapshot["components"] = components
        snapshot["host"] = _server_host
        snapshot["public_bind"] = _dashboard_is_public_bind()
        snapshot["updated_at"] = _utcnow_iso()
        _runtime_status = snapshot


def _cached(key, ttl, compute_fn):
    """Return cached result if fresh, otherwise recompute once per cache key."""
    while True:
        with _cache_lock:
            entry = _endpoint_cache.get(key)
            if entry and time.time() - entry["ts"] < ttl:
                return entry["data"]

            pending = _endpoint_inflight.get(key)
            if pending is None:
                pending = {"event": threading.Event()}
                _endpoint_inflight[key] = pending
                break

            waiter = pending["event"]

        waiter.wait()

    try:
        data = compute_fn()
    finally:
        with _cache_lock:
            pending = _endpoint_inflight.pop(key, None)
            if "data" in locals():
                _endpoint_cache[key] = {"data": data, "ts": time.time()}
            if pending is not None:
                pending["event"].set()

    return data


def _cached_snapshot_data(key):
    """Return cached data even when the entry is stale."""
    with _cache_lock:
        entry = _endpoint_cache.get(key)
        if not entry:
            return None
        return copy.deepcopy(entry.get("data"))


def _fresh_cached_data(key: str, ttl: float):
    with _cache_lock:
        entry = _endpoint_cache.get(key)
        if entry and time.time() - entry["ts"] < ttl:
            return copy.deepcopy(entry.get("data"))
    return None


def _store_cache_data(key: str, data) -> None:
    with _cache_lock:
        _endpoint_cache[key] = {"data": copy.deepcopy(data), "ts": time.time()}


def _cache_key_secret_fragment(value: str) -> str:
    value = str(value or "")
    if not value:
        return "none"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _refresh_cache_key_in_background(key: str, compute_fn) -> bool:
    """Refresh a cache key in a daemon thread while callers use stale data."""
    global _background_cache_refreshes
    if not callable(compute_fn):
        return False

    with _cache_lock:
        if key in _endpoint_inflight:
            return False
        if _background_cache_refreshes >= _MAX_BACKGROUND_CACHE_REFRESHES:
            logger.debug("Skipping background cache refresh for %s; refresh workers are saturated", key)
            return False
        pending = {"event": threading.Event()}
        _endpoint_inflight[key] = pending
        _background_cache_refreshes += 1

    def _worker():
        global _background_cache_refreshes
        try:
            data = compute_fn()
        except Exception as exc:
            logger.warning("Background cache refresh failed for %s: %s", key, exc)
        finally:
            with _cache_lock:
                pending = _endpoint_inflight.pop(key, None)
                _background_cache_refreshes = max(0, _background_cache_refreshes - 1)
                if "data" in locals():
                    _endpoint_cache[key] = {"data": data, "ts": time.time()}
                if pending is not None:
                    pending["event"].set()

    thread = threading.Thread(
        target=_worker,
        name=f"cache-refresh-{re.sub(r'[^a-z0-9]+', '-', key.lower()).strip('-')[:40] or 'key'}",
        daemon=True,
    )
    try:
        thread.start()
    except RuntimeError as exc:
        with _cache_lock:
            pending = _endpoint_inflight.pop(key, None)
            _background_cache_refreshes = max(0, _background_cache_refreshes - 1)
            if pending is not None:
                pending["event"].set()
        logger.warning("Could not start background cache refresh for %s: %s", key, exc)
        return False
    return True


def _cached_stale_while_revalidate(key: str, ttl: float, compute_fn):
    """Return fresh cache, stale cache with background refresh, or compute once."""
    with _cache_lock:
        entry = _endpoint_cache.get(key)
        if entry and time.time() - entry["ts"] < ttl:
            return copy.deepcopy(entry.get("data")), "live"
        stale = copy.deepcopy(entry.get("data")) if entry else None

    if stale is not None:
        _refresh_cache_key_in_background(key, compute_fn)
        return stale, "stale"

    return copy.deepcopy(_cached(key, ttl, compute_fn)), "live"


def _normalize_activity_entry(entry: dict) -> dict:
    """Downgrade known handled order rejections so the UI stays actionable."""
    normalized = dict(entry)
    raw_level = str(normalized.get("level", "") or "").upper()
    source = str(normalized.get("source", "") or "")
    message = str(normalized.get("message", "") or "")
    msg_lower = message.lower()

    normalized["raw_level"] = raw_level

    if (
        raw_level == "WARNING"
        and source == "src.polymarket.executor"
        and "failed to place" in msg_lower
        and any(pattern in msg_lower for pattern in HANDLED_ACTIVITY_WARNING_PATTERNS)
    ):
        normalized["level"] = "INFO"
        normalized["activity_kind"] = "handled_order_rejection"

    return normalized


def _classify_activity_sport(entry: dict) -> str:
    """Classify an activity entry for server-side filtering."""
    source = str(entry.get("source", "") or "").lower()
    message = str(entry.get("message", "") or "").lower()
    component = str(entry.get("component", "") or "").lower()

    # werkzeug HTTP access logs are infrastructure — not sport-specific
    if source == "werkzeug":
        return "general"
    crypto_markers = (
        "btc5m",
        "btc 5m",
        "crypto 5m",
        "btc-updown",
        "eth-updown",
        "sol-updown",
        "rate_limit_signal",
    )
    if (
        source.startswith("runtime.btc5m_loop")
        or source == "src.polymarket.btc_5m"
        or source == "src.polymarket.btc5m_opportunity"
        or component.startswith("btc5m_loop")
        or any(marker in message for marker in crypto_markers)
    ):
        return "crypto"

    return "ufc"


def _normalize_sport_filter(raw_value) -> str:
    sport = str(raw_value or "all").strip().lower()
    return sport if sport in {"all", "ufc", "crypto"} else "all"


def _activity_entry_matches_sport(entry: dict, sport: str) -> bool:
    if sport == "all":
        return True
    entry_sport = str(entry.get("sport", "") or "").lower()
    return entry_sport in {sport, "general"}


def _filter_entries_by_sport(entries: list[dict], sport: str) -> list[dict]:
    if sport == "all":
        return entries
    return [entry for entry in entries if _activity_entry_matches_sport(entry, sport)]


def _activity_timestamp_sort_key(entry: dict) -> tuple[int, str]:
    raw = str(entry.get("timestamp", "") or "").strip()
    if not raw:
        return (0, "")
    try:
        return (1, datetime.strptime(raw, "%Y-%m-%d %H:%M:%S").isoformat())
    except ValueError:
        return (1, raw)


def _runtime_component_label(component: str) -> str:
    if component.startswith("btc5m_loop:"):
        profile = component.split(":", 1)[1].strip() or "profile"
        return f"Crypto 5m {profile}"
    if component == "btc5m_loop":
        return "Crypto 5m"
    return RUNTIME_COMPONENT_LABELS.get(component, component.replace("_", " ").title())


def _runtime_issue_activity_entries(status: dict | None = None) -> list[dict]:
    runtime_status = status or _runtime_status_with_liveness()
    entries: list[dict] = []
    components = runtime_status.get("components") or {}
    for component, payload in components.items():
        entry = dict(payload or {})
        state = str(entry.get("state", "") or "").strip().lower()
        if state not in RUNTIME_ACTIVITY_ERROR_STATES:
            continue

        updated_at = _parse_runtime_timestamp(entry.get("updated_at"))
        timestamp = (
            updated_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            if updated_at is not None
            else datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        )
        label = _runtime_component_label(component)
        coverage_alerts = [
            str(alert).strip()
            for alert in (entry.get("coverage_alerts") or [])
            if str(alert).strip()
        ]
        detail = " | ".join(coverage_alerts) if coverage_alerts else str(entry.get("message", "") or "").strip()
        message = f"{label} {state}"
        if detail:
            message = f"{message}: {detail}"

        normalized_entry = _normalize_activity_entry(
            {
                "timestamp": timestamp,
                "level": "ERROR",
                "source": f"runtime.{component}",
                "message": message,
                "activity_kind": "runtime_component_issue",
                "component": component,
                "component_state": state,
            }
        )
        normalized_entry["sport"] = _classify_activity_sport(normalized_entry)
        entries.append(normalized_entry)

    return entries


def _parse_log_entries(raw: str) -> list[dict]:
    """Parse structured log lines and fold continuation lines into the prior entry."""
    entries = []
    for line in raw.splitlines():
        match = LOG_LINE_RE.match(line)
        if match:
            entries.append({
                "timestamp": match.group(1),
                "level": match.group(2),
                "source": match.group(3),
                "message": match.group(4),
            })
        elif entries and line.strip():
            entries[-1]["message"] += " " + line.strip()

    normalized_entries = [_normalize_activity_entry(entry) for entry in entries]
    for entry in normalized_entries:
        entry["sport"] = _classify_activity_sport(entry)
    return normalized_entries


def _read_activity_entries(
    log_path: Path,
    *,
    limit: int = 500,
    sport: str = "all",
    runtime_status: dict | None = None,
) -> list[dict]:
    runtime_entries = _runtime_issue_activity_entries(runtime_status)
    filtered_runtime_entries = _filter_entries_by_sport(runtime_entries, sport)

    log_limit = max(limit, 1)
    log_entries = _read_recent_log_entries(log_path, limit=log_limit, sport=sport)
    if not filtered_runtime_entries:
        return log_entries[-limit:]

    retained_log_capacity = max(limit - len(filtered_runtime_entries), 0)
    if retained_log_capacity == 0:
        retained_log_entries: list[dict] = []
    else:
        retained_log_entries = log_entries[-retained_log_capacity:]

    combined = retained_log_entries + filtered_runtime_entries
    combined.sort(key=_activity_timestamp_sort_key)
    return combined[-limit:]


def _runtime_issue_significant_actions(status: dict | None = None) -> list[dict]:
    actions: list[dict] = []
    for entry in _runtime_issue_activity_entries(status):
        actions.append(
            {
                "timestamp": entry.get("timestamp", ""),
                "level": entry.get("level", "ERROR"),
                "source": entry.get("source", ""),
                "tag": "ALERT",
                "color": "red",
                "message": entry.get("message", ""),
                "sport": entry.get("sport", _classify_activity_sport(entry)),
            }
        )
    return actions


def _read_recent_log_entries(
    log_path: Path,
    limit: int = 500,
    chunk_bytes: int = 131_072,
    sport: str = "all",
) -> list[dict]:
    """
    Read backward through bot.log until we have the requested number of parsed entries.

    A fixed byte tail is not stable once prediction output gets verbose, because recent
    errors can disappear from the window long before they fall out of the latest 500
    parsed entries.
    """
    if not log_path.exists():
        return []

    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            position = f.tell()
            buffer = b""

            while position > 0:
                read_size = min(chunk_bytes, position)
                position -= read_size
                f.seek(position)
                buffer = f.read(read_size) + buffer

                raw = buffer.decode("utf-8", errors="replace")
                if position > 0:
                    first_newline = raw.find("\n")
                    if first_newline != -1:
                        raw = raw[first_newline + 1:]

                entries = _parse_log_entries(raw)
                filtered_entries = _filter_entries_by_sport(entries, sport)
                if len(filtered_entries) >= limit or position == 0:
                    return filtered_entries[-limit:]
    except Exception as e:
        logger.warning("Failed to read recent log entries from %s: %s", log_path, e)
        return []

    return []


def _json_no_store(payload, extra_headers: dict[str, str] | None = None):
    """Return JSON that bypasses browser and intermediary caches."""
    response = jsonify(payload)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    for key, value in (extra_headers or {}).items():
        response.headers[key] = value
    return response


@app.route("/healthz")
def healthz():
    status = _runtime_status_with_liveness()
    payload = {
        "ok": bool(status.get("ok", True)),
        "ready": bool(status.get("ready", False)),
        "startup_source": status.get("startup_source"),
        "effective_live_mode": status.get("effective_live_mode"),
        "trading_enabled": bool(status.get("trading_enabled", False)),
        "updated_at": status.get("updated_at"),
    }
    return _json_no_store(payload)


@app.route("/readyz")
def readyz():
    status = _runtime_status_with_liveness()
    payload = copy.deepcopy(status)
    payload["ok"] = bool(status.get("ready", False))
    response = _json_no_store(payload)
    response.status_code = 200 if payload["ok"] else 503
    return response


@app.route("/api/runtime-status")
def api_runtime_status():
    """Dashboard-friendly runtime status endpoint that always returns JSON."""
    status = _runtime_status_with_liveness()
    payload = copy.deepcopy(status)
    payload["ok"] = bool(status.get("ok", True))
    return _json_no_store(payload)


def _dashboard_mutation_token() -> str | None:
    token = str(os.environ.get("WEB_DASHBOARD_TOKEN", "") or "").strip()
    return token or None


def _dashboard_is_public_bind() -> bool:
    return str(_server_host or "").strip().lower() not in {"127.0.0.1", "localhost", "::1"}


def _request_dashboard_token() -> str | None:
    auth_header = str(request.headers.get("Authorization", "") or "").strip()
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            return token
    header_token = str(request.headers.get("X-Dashboard-Token", "") or "").strip()
    return header_token or None


def _dashboard_read_is_authorized() -> bool:
    """Local reads are trusted; public reads need the configured dashboard token."""
    if not _dashboard_is_public_bind():
        return True
    configured = _dashboard_mutation_token()
    provided = _request_dashboard_token()
    return bool(
        configured
        and provided
        and hmac.compare_digest(provided, configured)
    )


def _require_read_auth():
    """Pages retain a redacted public view; callers with the token receive full data."""
    return None


_PUBLIC_READ_REDACT_KEYS = frozenset(
    {
        "condition_id",
        "decision_key",
        "display_ledger_id",
        "existing_ledger_id",
        "existing_order_id",
        "existing_token_id",
        "ledger_bet_id",
        "ledger_id",
        "manifest_path",
        "order_id",
        "processed_dir",
        "response",
        "sources",
        "token_id",
        "token_id_no",
        "token_id_yes",
    }
)


def _redact_public_read_payload(value):
    if isinstance(value, dict):
        return {
            key: _redact_public_read_payload(item)
            for key, item in value.items()
            if str(key) not in _PUBLIC_READ_REDACT_KEYS
        }
    if isinstance(value, list):
        return [_redact_public_read_payload(item) for item in value]
    return value


def _dashboard_read_response(payload):
    authorized = _dashboard_read_is_authorized()
    response = _json_no_store(
        payload if authorized else _redact_public_read_payload(payload),
        {"X-Dashboard-Data-Scope": "full" if authorized else "redacted"},
    )
    return response


def _api_internal_error(code: str, message: str):
    response = _json_no_store({"ok": False, "error": code, "message": message})
    response.status_code = 500
    return response


def _require_mutation_auth():
    configured = _dashboard_mutation_token()
    if configured is None:
        if not _dashboard_is_public_bind():
            return None
        return _json_no_store(
            {
                "ok": False,
                "error": "dashboard_mutations_disabled",
                "message": "Set WEB_DASHBOARD_TOKEN to enable dashboard mutations on public binds.",
            }
        ), 503

    provided = _request_dashboard_token()
    if provided and hmac.compare_digest(provided, configured):
        return None

    return _json_no_store(
        {
            "ok": False,
            "error": "unauthorized",
            "message": "Missing or invalid dashboard mutation token.",
        }
    ), 401


def _html_no_store(template_name: str):
    """Render HTML pages without browser caching so deployed JS stays current."""
    response = make_response(render_template(template_name))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def _bot_activity_headers(log_path: Path, entries: list[dict]) -> dict[str, str]:
    """Expose snapshot metadata so the UI can prove what it is rendering."""
    headers = {
        "X-Bot-Activity-Server-Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "X-Bot-Activity-Last-Entry": entries[-1]["timestamp"] if entries else "",
    }
    if log_path.exists():
        headers["X-Bot-Activity-Log-MTime"] = datetime.fromtimestamp(
            log_path.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S")
    return headers


def _bot_activity_snapshot(log_path: Path, entries: list[dict]) -> dict:
    """Return a self-contained activity snapshot so metadata and rows stay in sync."""
    headers = _bot_activity_headers(log_path, entries)
    return {
        "server_time": headers.get("X-Bot-Activity-Server-Time", ""),
        "last_entry": headers.get("X-Bot-Activity-Last-Entry", ""),
        "log_mtime": headers.get("X-Bot-Activity-Log-MTime", ""),
        "entry_count": len(entries),
        "entries": entries,
    }


@app.route("/")
def index():
    return _html_no_store("dashboard.html")


@app.route("/ufc")
def ufc_page():
    return _html_no_store("dashboard.html")


BTC5M_EMERGENCY_STOP_FILENAME = "btc5m_emergency_stop.json"


def btc5m_emergency_stop_path() -> Path:
    return Path(LOGS_DIR) / BTC5M_EMERGENCY_STOP_FILENAME


def btc5m_emergency_stop_status(path: Path | None = None) -> dict:
    stop_path = Path(path) if path is not None else btc5m_emergency_stop_path()
    payload = {
        "active": False,
        "path": str(stop_path),
        "requested_at": None,
        "requested_by": None,
        "reason": "",
        "source": "",
    }
    if not stop_path.exists():
        return payload
    try:
        raw = json.loads(stop_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            payload.update(raw)
    except Exception as exc:
        payload.update(
            {
                "parse_error": str(exc),
                "reason": "Stop request file exists but could not be parsed.",
            }
        )
    payload["active"] = True
    payload["path"] = str(stop_path)
    return payload


def _classify_sport_from_market(market_title: str) -> str:
    """Classify a Polymarket position by sport."""
    title = str(market_title or "").strip().lower()
    if re.search(r"^\s*ufc\b", title) or "ultimate fighting championship" in title:
        return "ufc"
    return "other"


def _classify_sport_from_position(position: dict) -> str:
    title = position.get("market") or position.get("title") or position.get("question") or ""
    event_slug = str(position.get("event_slug") or position.get("eventSlug") or "").strip().lower()
    slug = str(position.get("slug") or "").strip().lower()
    icon = str(position.get("icon") or "").strip().lower()
    if (
        event_slug.startswith("ufc-")
        or slug.startswith("ufc-")
        or "ufc-logo" in icon
    ):
        return "ufc"
    return _classify_sport_from_market(str(title))


def _classify_sport_from_ledger_path(ledger_path: str) -> str:
    """Classify a bet by sport based on ledger path."""
    raw = str(ledger_path or "").lower()
    if any(token in raw for token in ("btc5m", "crypto", "bitcoin", "ethereum", "solana")):
        return "crypto"
    if "tennis" in raw:
        return "tennis"
    return "ufc"


def _trader_label_from_path(ledger_path: str) -> str:
    raw = str(ledger_path or "").lower()
    if "model_tracker" in raw:
        return "M"
    if "gemini_tracker" in raw:
        return "G"
    if "conviction" in raw:
        return "C"
    if "single" in raw:
        return "S"
    return "S"


def _trader_breakdown_specs():
    from src.strategy.duo_trader import get_all_trader_ledgers

    meta = {
        "S": ("Single (Value)", 0.30),
        "C": ("Conviction", None),
        "M": ("Model Tracker", 1.0),
        "G": ("Gemini Tracker", None),
    }
    specs = []
    for label, path in get_all_trader_ledgers():
        style, blend = meta.get(label, (f"Trader {label}", None))
        specs.append((label, style, path, blend))
    return specs


def _parse_upcoming_event_datetime(raw_value):
    if raw_value in (None, ""):
        return None

    raw = str(raw_value).strip()
    if not raw:
        return None

    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        pass

    for fmt in ("%B %d, %Y", "%b %d, %Y", "%b. %d, %Y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue

    return None


def _normalize_upcoming_event_datetime(parsed: datetime | None) -> datetime | None:
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _upcoming_event_day_key(raw_value) -> str | None:
    parsed = _parse_upcoming_event_datetime(raw_value)
    if parsed is None:
        return None
    return parsed.date().isoformat()


def _position_is_dashboard_open(position: dict) -> bool:
    """Mirror the Polymarket open-positions view for dashboard display."""
    if bool(position.get("redeemable")):
        return False
    return _safe_float(position.get("size"), 0.0) >= OPEN_BET_DISPLAY_SIZE_THRESHOLD


def _position_invested_value(position: dict) -> float:
    invested = _safe_float(position.get("invested"), math.nan)
    if math.isfinite(invested):
        return invested
    return _safe_float(position.get("initialValue"), 0.0)


def _position_current_value(position: dict) -> float:
    value = _safe_float(position.get("value"), math.nan)
    if math.isfinite(value):
        return value
    return _safe_float(position.get("currentValue"), 0.0)


def _position_unrealized_pnl(position: dict) -> float:
    value = _safe_float(position.get("unrealized_pnl"), math.nan)
    if math.isfinite(value):
        return value
    cash_pnl = _safe_float(position.get("cashPnl"), math.nan)
    realized = _safe_float(position.get("realized_pnl", position.get("realizedPnl")), 0.0)
    if math.isfinite(cash_pnl):
        return cash_pnl - realized
    return _position_current_value(position) - _position_invested_value(position)


def _position_realized_pnl(position: dict) -> float:
    return _safe_float(position.get("realized_pnl", position.get("realizedPnl")), 0.0)


def _dashboard_live_pnl_from_raw(raw_live_pnl: dict, *, sport: str = "all") -> dict:
    """Filter the displayed positions list; preserve raw aggregates to match Polymarket.

    monitor.compute_pnl() already sums invested / value / realized / unrealized
    across all open, closed, dust, and redeemable positions the same way
    Polymarket's profile page does. Only the per-row list shown in the UI is
    filtered — headline totals pass through unchanged so the dashboard PnL
    matches what Polymarket reports.
    """
    sport = _normalize_sport_filter(sport)
    live = copy.deepcopy(raw_live_pnl or {})
    raw_positions = [dict(pos) for pos in live.get("positions", [])]
    aggregate_positions = raw_positions
    if sport != "all":
        aggregate_positions = [
            pos for pos in raw_positions
            if _classify_sport_from_position(pos) == sport
        ]
        closed_positions = [
            dict(pos) for pos in live.get("closed_positions", [])
            if _classify_sport_from_position(pos) == sport
        ]
        open_realized = sum(_position_realized_pnl(pos) for pos in aggregate_positions)
        closed_realized = sum(_position_realized_pnl(pos) for pos in closed_positions)
        unrealized = sum(_position_unrealized_pnl(pos) for pos in aggregate_positions)
        live["total_invested"] = sum(_position_invested_value(pos) for pos in aggregate_positions)
        live["current_value"] = sum(_position_current_value(pos) for pos in aggregate_positions)
        live["unrealized_pnl"] = unrealized
        live["realized_pnl"] = open_realized + closed_realized
        live["total_pnl"] = live["unrealized_pnl"] + live["realized_pnl"]
        live["num_closed"] = len(closed_positions)
        live["closed_positions"] = closed_positions
        live["sport"] = sport

    visible_positions = [pos for pos in aggregate_positions if _position_is_dashboard_open(pos)]
    for pos in visible_positions:
        pos["sport"] = _classify_sport_from_position(pos)

    live["positions"] = visible_positions
    live["num_positions"] = len(visible_positions)
    live["hidden_positions"] = max(0, len(aggregate_positions) - len(visible_positions))
    live["excluded_positions"] = max(0, len(raw_positions) - len(aggregate_positions))
    live["visible_invested"] = sum(_safe_float(pos.get("invested"), 0.0) for pos in visible_positions)
    live["visible_value"] = sum(_safe_float(pos.get("value"), 0.0) for pos in visible_positions)
    live["open_position_size_threshold"] = OPEN_BET_DISPLAY_SIZE_THRESHOLD
    return live


def _empty_live_pnl_snapshot() -> dict:
    return {
        "total_invested": 0.0,
        "current_value": 0.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "total_pnl": 0.0,
        "num_positions": 0,
        "num_closed": 0,
        "positions": [],
        "closed_positions": [],
        "timestamp": _utcnow_iso(),
    }


def _get_position_monitor() -> PositionMonitor:
    global _position_monitor
    with _monitor_lock:
        if not _position_monitor:
            _position_monitor = PositionMonitor(clob_client=_clob_client)
        return _position_monitor


def _compute_live_pnl_snapshot() -> dict:
    monitor = _get_position_monitor()
    payload = _call_with_timeout(
        "fetching live dashboard P&L",
        monitor.compute_pnl,
        LIVE_PNL_TIMEOUT_SECONDS,
    )
    if payload is None:
        raise TimeoutError(
            f"Live dashboard P&L timed out after {LIVE_PNL_TIMEOUT_SECONDS:.1f}s"
        )
    return payload


def _load_live_pnl_snapshot() -> tuple[dict, str]:
    cache_key = "dashboard-live-pnl"
    try:
        snapshot, source = _cached_stale_while_revalidate(
            cache_key,
            LIVE_PNL_CACHE_TTL,
            _compute_live_pnl_snapshot,
        )
        return copy.deepcopy(snapshot or _empty_live_pnl_snapshot()), source
    except Exception as e:
        stale = _cached_snapshot_data(cache_key)
        if stale is not None:
            logger.warning("Using stale dashboard P&L snapshot: %s", e)
            return stale, "stale"
        logger.warning("Dashboard P&L snapshot unavailable: %s", e)
        return _empty_live_pnl_snapshot(), "unavailable"


def _profile_page_query_data(queries: list[dict], predicate) -> object | None:
    for query in queries:
        key = query.get("queryKey")
        if predicate(key):
            return query.get("state", {}).get("data")
    return None


def _profile_portfolio_pnl_history(queries: list[dict], preferred_range: str) -> list[dict]:
    preferred = str(preferred_range or "").strip().upper()
    for query in queries:
        key = query.get("queryKey")
        if not (isinstance(key, list) and key and key[0] == "portfolio-pnl"):
            continue
        range_key = str(key[-1] if len(key) >= 2 else "").strip().upper()
        if range_key != preferred:
            continue
        data = query.get("state", {}).get("data")
        return data if isinstance(data, list) else []
    return []


def _profile_terminal_pnl(pnl_history: list[dict]) -> float:
    for point in reversed(pnl_history):
        if not isinstance(point, dict):
            continue
        value = _safe_float(point.get("p"), math.nan)
        if math.isfinite(value):
            return value
    return math.nan


def _profile_any_portfolio_pnl_history(queries: list[dict]) -> list[dict]:
    longest: list[dict] = []
    for query in queries:
        key = query.get("queryKey")
        if not (isinstance(key, list) and key and key[0] == "portfolio-pnl"):
            continue
        data = query.get("state", {}).get("data")
        if isinstance(data, list) and len(data) > len(longest):
            longest = data
    return longest


def _json_string_values(value) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _json_string_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _json_string_values(item)


def _next_f_payload_text(html: str) -> str:
    """Return unescaped text chunks from Next's streamed flight payload."""
    decoder = json.JSONDecoder()
    chunks: list[str] = []
    idx = 0
    while True:
        start = html.find(_PROFILE_NEXT_F_PUSH_PREFIX, idx)
        if start < 0:
            break
        arg_start = start + len(_PROFILE_NEXT_F_PUSH_PREFIX)
        try:
            payload, end = decoder.raw_decode(html[arg_start:])
        except json.JSONDecodeError:
            idx = arg_start
            continue
        chunks.extend(_json_string_values(payload))
        idx = arg_start + end
    return "\n".join(chunks)


def _extract_dehydrated_queries_from_text(text: str) -> list[dict]:
    """Extract React Query dehydrated query arrays embedded in page text."""
    decoder = json.JSONDecoder()
    queries: list[dict] = []
    idx = 0
    needle = '"queries":'
    while True:
        start = text.find(needle, idx)
        if start < 0:
            break
        value_start = start + len(needle)
        value_text = text[value_start:]
        value_text = value_text.lstrip()
        try:
            value, end = decoder.raw_decode(value_text)
        except json.JSONDecodeError:
            idx = value_start
            continue
        if isinstance(value, list):
            queries.extend(query for query in value if isinstance(query, dict))
        idx = value_start + (len(text[value_start:]) - len(value_text)) + end
    return queries


def _profile_snapshot_from_queries(queries: list[dict], page_props: dict | None = None) -> dict:
    page_props = page_props or {}
    user_data = _profile_page_query_data(
        queries,
        lambda key: isinstance(key, list) and key and key[0] == "/api/profile/userData",
    ) or {}
    volume_data = _profile_page_query_data(
        queries,
        lambda key: isinstance(key, list) and key and key[0] == "/api/profile/volume",
    ) or {}
    stats_data = _profile_page_query_data(
        queries,
        lambda key: isinstance(key, list) and key and key[0] == "user-stats",
    ) or {}
    traded_data = _profile_page_query_data(
        queries,
        lambda key: isinstance(key, list) and key and key[0] == "/api/profile/marketsTraded",
    ) or {}
    positions_value = _profile_page_query_data(
        queries,
        lambda key: isinstance(key, list) and len(key) >= 2 and key[0] == "positions" and key[1] == "value",
    )
    all_pnl_history = _profile_portfolio_pnl_history(queries, "ALL")
    latest_all_chart_pnl = _profile_terminal_pnl(all_pnl_history)
    fallback_volume_pnl = _safe_float(volume_data.get("pnl"), math.nan)
    fallback_pnl_history = _profile_any_portfolio_pnl_history(queries)
    fallback_chart_pnl = _profile_terminal_pnl(fallback_pnl_history)
    total_pnl = latest_all_chart_pnl
    pnl_history_source = "portfolio-pnl:ALL"
    if not math.isfinite(total_pnl):
        total_pnl = fallback_volume_pnl
        pnl_history_source = "/api/profile/volume"
    if not math.isfinite(total_pnl):
        total_pnl = fallback_chart_pnl
        pnl_history_source = "portfolio-pnl:fallback"

    return {
        "username": (
            page_props.get("username")
            or user_data.get("name")
            or user_data.get("pseudonym")
        ),
        "profile_slug": page_props.get("profileSlug") or user_data.get("name") or user_data.get("pseudonym"),
        "proxy_address": page_props.get("proxyAddress") or user_data.get("proxyWallet"),
        "positions_value": _safe_float(positions_value, math.nan),
        # Match the profile headline's all-time P/L. Polymarket embeds several
        # short-range portfolio-pnl queries; selecting the first one can show a
        # 1D/1W/1M delta instead of the overall profile P/L.
        "total_pnl": total_pnl,
        "profile_volume": _safe_float(volume_data.get("amount"), math.nan),
        "largest_win": _safe_float(stats_data.get("largestWin"), math.nan),
        "predictions": int(traded_data.get("traded") or stats_data.get("trades") or 0),
        "views": int(stats_data.get("views") or 0),
        "join_date": stats_data.get("joinDate"),
        "pnl_history_all": all_pnl_history,
        "pnl_history_1d": _profile_portfolio_pnl_history(queries, "1D"),
        "pnl_history_source": pnl_history_source,
        "raw": {
            "user_data": user_data,
            "volume": volume_data,
            "stats": stats_data,
            "traded": traded_data,
        },
    }


def _extract_polymarket_profile_snapshot(html: str) -> dict:
    match = _PROFILE_NEXT_DATA_RE.search(html or "")
    if match:
        payload = json.loads(match.group(1))
        page_props = payload.get("props", {}).get("pageProps", {})
        queries = page_props.get("dehydratedState", {}).get("queries", [])
        return _profile_snapshot_from_queries(queries, page_props)

    streamed_text = _next_f_payload_text(html or "")
    queries = _extract_dehydrated_queries_from_text(streamed_text)
    if not queries:
        raise RuntimeError("Polymarket profile page did not contain profile query data")
    return _profile_snapshot_from_queries(queries)


def _compute_polymarket_profile_snapshot() -> dict:
    wallet_address = str(_get_position_monitor().wallet_address or "").strip()
    if not wallet_address:
        raise RuntimeError("No Polymarket wallet configured for profile snapshot")

    response = requests.get(
        f"https://polymarket.com/profile/{wallet_address}",
        timeout=PROFILE_PAGE_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    snapshot = _extract_polymarket_profile_snapshot(response.text)
    if not math.isfinite(snapshot.get("total_pnl", math.nan)):
        raise RuntimeError("Polymarket profile snapshot missing total P&L")
    return snapshot


def _log_polymarket_profile_snapshot_warning(prefix: str, exc: Exception) -> None:
    """Warn on profile scrape failures without spamming dashboard activity."""
    error_message = str(exc)
    now = time.monotonic()
    key = (str(prefix), error_message)
    with _cache_lock:
        last_logged_at = _profile_snapshot_warning_state.get(key)
        should_warn = (
            last_logged_at is None
            or now - last_logged_at >= PROFILE_PAGE_FAILURE_LOG_TTL
        )
        if should_warn:
            _profile_snapshot_warning_state[key] = now

    if should_warn:
        logger.warning("%s: %s", prefix, exc)
    else:
        logger.debug(
            "%s suppressed duplicate within %.0fs: %s",
            prefix,
            PROFILE_PAGE_FAILURE_LOG_TTL,
            exc,
        )


def _load_polymarket_profile_snapshot() -> tuple[dict, str]:
    cache_key = "dashboard-polymarket-profile"
    try:
        snapshot, source = _cached_stale_while_revalidate(
            cache_key,
            PROFILE_PAGE_CACHE_TTL,
            _compute_polymarket_profile_snapshot,
        )
        return copy.deepcopy(snapshot or {}), source
    except Exception as e:
        stale = _cached_snapshot_data(cache_key)
        if stale is not None:
            _log_polymarket_profile_snapshot_warning(
                "Using stale Polymarket profile snapshot",
                e,
            )
            return stale, "stale"
        _log_polymarket_profile_snapshot_warning(
            "Polymarket profile snapshot unavailable",
            e,
        )
        return {}, "unavailable"


@app.route("/api/summary")
def api_summary():
    """Return summary stats — merges ledger stats with live Polymarket data."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    sport = _normalize_sport_filter(request.args.get("sport", "all"))
    ledger = load_all_trader_ledgers()
    summary = ledger.get_summary()

    raw_live, live_source = _load_live_pnl_snapshot()
    try:
        if live_source == "unavailable":
            raise RuntimeError("live dashboard P&L unavailable")
        live = _dashboard_live_pnl_from_raw(raw_live, sport=sport)
        # Always overlay live P&L — it includes closed positions the
        # local ledger may never have tracked.
        summary["open_bets"] = live["num_positions"]
        summary["unrealized_pnl"] = live["unrealized_pnl"]
        summary["open_invested"] = live["total_invested"]
        summary["realized_pnl"] = live["realized_pnl"]
        summary["total_pnl"] = live["total_pnl"]
        # Fix settled count and ROI from live data
        num_closed = live.get("num_closed", 0)
        if num_closed > summary.get("settled_bets", 0):
            summary["settled_bets"] = num_closed
        total_deployed = live["total_invested"] + live["realized_pnl"]
        if total_deployed > 0:
            summary["roi"] = live["total_pnl"] / total_deployed
        if live_source != "live":
            summary["_pnl_degraded"] = True
            summary["_pnl_source"] = live_source
        summary["sport"] = sport
    except Exception as e:
        logger.warning("Live PnL merge failed — dashboard may show stale data: %s", e)
        summary["_pnl_degraded"] = True
        summary["_pnl_source"] = live_source

    profile_snapshot, profile_source = _load_polymarket_profile_snapshot()
    if profile_snapshot:
        profile_total_pnl = profile_snapshot.get("total_pnl")
        profile_positions_value = profile_snapshot.get("positions_value")
        profile_volume = profile_snapshot.get("profile_volume")
        if (
            profile_total_pnl is not None
            and math.isfinite(profile_total_pnl)
        ):
            summary["profile_total_pnl"] = profile_total_pnl
            if sport == "all":
                summary["total_pnl"] = profile_total_pnl
        if (
            sport == "all"
            and profile_positions_value is not None
            and math.isfinite(profile_positions_value)
        ):
            summary["positions_value"] = profile_positions_value
            summary["open_invested"] = profile_positions_value
        if profile_volume is not None and math.isfinite(profile_volume):
            summary["profile_volume"] = profile_volume
        largest_win = profile_snapshot.get("largest_win")
        if largest_win is not None and math.isfinite(largest_win):
            summary["profile_largest_win"] = largest_win
        predictions = profile_snapshot.get("predictions")
        if predictions:
            summary["profile_predictions"] = predictions
        username = profile_snapshot.get("username")
        if username:
            summary["profile_username"] = username
        summary["_profile_source"] = profile_source
        if profile_source != "live":
            summary["_pnl_degraded"] = True
    else:
        summary["_profile_source"] = profile_source

    return jsonify(summary)


@app.route("/api/bets")
def api_bets():
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    ledger = load_all_trader_ledgers()

    def _tag_sport(bets: list[dict]) -> list[dict]:
        for bet in bets:
            bet["sport"] = _classify_sport_from_ledger_path(bet.get("_ledger_path", ""))
        return bets

    return jsonify({
        "open": _tag_sport(ledger.open_bets),
        "settled": _tag_sport(ledger.settled_bets),
        "all": _tag_sport(ledger.bets),
    })


def _normalize_name(name):
    """Normalize fighter names for cross-source dashboard matching."""
    if not name:
        return ""
    return normalize_cross_source_name(name)


def _decision_context_aliases(value) -> list[str]:
    raw = str(value or "").strip().upper()
    if not raw:
        return []

    aliases: list[str] = []
    for candidate in (raw, raw.split(":", 1)[0]):
        if candidate and candidate not in aliases:
            aliases.append(candidate)
    return aliases


def _match_decision_to_bet(bet, decisions_index):
    """Find the best-matching operator decision for a bet.

    decisions_index is a dict keyed by (normalized_bet_on, event_date_prefix)
    with fallback keys of (frozenset({norm_a, norm_b}), event_date_prefix).
    Returns the matched decision dict or None.
    """
    norm_fighter = _normalize_name(
        bet.get("fighter") or bet.get("bet_on") or bet.get("side")
    )
    event_date = (
        _fight_matrix_event_group_date(
            bet.get("market_event_date") or bet.get("event_date") or "",
            card_date=_row_card_date(bet),
        )
        or ""
    )

    trader = str(
        bet.get("trader")
        or _trader_label_from_path(bet.get("_ledger_path", ""))
        or ""
    ).strip().upper()

    def _decision_matches_selected_fighter(decision: dict | None) -> bool:
        if not decision:
            return False
        decision_pick = (
            decision.get("bet_on")
            or decision.get("pick")
            or decision.get("fighter")
        )
        if not decision_pick:
            return False
        return _normalize_name(decision_pick) == norm_fighter

    for context in [*(_decision_context_aliases(trader) or []), ""]:
        # Primary: bet.fighter == decision.bet_on + event_date match
        key = (context, norm_fighter, event_date) if context else (norm_fighter, event_date)
        if key in decisions_index:
            return decisions_index[key]

        # Fallback: {fighter, opponent} == {fighter_a, fighter_b} + date
        norm_opponent = _normalize_name(
            bet.get("opponent") or bet.get("fighter_b") or bet.get("opposite_side")
        )
        if norm_fighter and norm_opponent:
            pair_key = (
                (context, frozenset({norm_fighter, norm_opponent}), event_date)
                if context
                else (frozenset({norm_fighter, norm_opponent}), event_date)
            )
            if pair_key in decisions_index:
                candidate = decisions_index[pair_key]
                if _decision_matches_selected_fighter(candidate):
                    return candidate

    return None


def _build_decisions_index(
    decisions,
    *,
    market_event_date_hints=None,
    card_date_hints=None,
):
    """Build lookup dicts from operator decisions for fast matching.

    Returns a single dict with both (norm_bet_on, date) keys and
    (frozenset({norm_a, norm_b}), date) keys. Most recent decision wins.
    """
    index = {}
    # Decisions are sorted newest-first from load_decision_log, so first match wins
    for d in decisions:
        market_event_date = _resolve_market_event_date_hint(
            fighter_a=str(d.get("fighter_a", "") or ""),
            fighter_b=str(d.get("fighter_b", "") or ""),
            event_date=str(d.get("event_date") or ""),
            market_event_date=str(d.get("market_event_date") or ""),
            hints=market_event_date_hints or {},
        )
        card_date = _resolve_card_date_hint(
            fighter_a=str(d.get("fighter_a", "") or ""),
            fighter_b=str(d.get("fighter_b", "") or ""),
            event_date=str(d.get("event_date") or ""),
            market_event_date=market_event_date or str(d.get("market_event_date") or ""),
            card_date=_row_card_date(d),
            hints=card_date_hints or {},
        )
        date = _fight_matrix_event_group_date(
            market_event_date or d.get("event_date") or "",
            card_date=card_date,
        )
        norm_bet_on = _normalize_name(d.get("bet_on"))
        if norm_bet_on and date:
            key = (norm_bet_on, date)
            if key not in index:
                index[key] = d
            for context in _decision_context_aliases(d.get("decision_context")):
                context_key = (context, norm_bet_on, date)
                if context_key not in index:
                    index[context_key] = d

        norm_a = _normalize_name(d.get("fighter_a"))
        norm_b = _normalize_name(d.get("fighter_b"))
        if norm_a and norm_b and date:
            pair_key = (frozenset({norm_a, norm_b}), date)
            if pair_key not in index:
                index[pair_key] = d
            for context in _decision_context_aliases(d.get("decision_context")):
                context_pair_key = (context, frozenset({norm_a, norm_b}), date)
                if context_pair_key not in index:
                    index[context_pair_key] = d

    return index


def _row_card_date(row) -> str:
    getter = getattr(row, "get", None)
    if not callable(getter):
        return ""

    raw = str(
        getter("card_date", "")
        or getter("official_event_date", "")
        or ""
    ).strip()
    if raw:
        return raw

    snapshot = getter("event_context_snapshot", None)
    if isinstance(snapshot, dict):
        return str(
            snapshot.get("card_date")
            or snapshot.get("event_date")
            or ""
        ).strip()
    return ""


def _coerce_fight_matrix_day(raw_value, *, allow_raw_prefix: bool = True) -> str:
    raw = str(raw_value or "").strip()
    if not raw:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw

    parsed = _parse_upcoming_event_datetime(raw)
    if parsed is None:
        return raw[:10] if allow_raw_prefix else ""
    if parsed.tzinfo is None:
        return parsed.date().isoformat()
    return parsed.astimezone(FIGHT_MATRIX_TIMEZONE).date().isoformat()


def _fight_matrix_key(
    fighter_a: str,
    fighter_b: str,
    event_date: str,
    *,
    card_date: str = "",
):
    event_group_date = _fight_matrix_event_group_date(event_date, card_date=card_date)
    return (
        frozenset({_normalize_name(fighter_a), _normalize_name(fighter_b)}),
        event_group_date,
    )


def _fight_matrix_event_group_date(raw_value, *, card_date: str = "") -> str:
    card_day = _coerce_fight_matrix_day(card_date, allow_raw_prefix=False)
    if card_day:
        return card_day

    raw = str(raw_value or "").strip()
    if not raw:
        return ""
    return _coerce_fight_matrix_day(raw)


def _raw_datetime_has_time(raw_value) -> bool:
    raw = str(raw_value or "").strip()
    return bool(re.search(r"(?:T|\d{1,2}:\d{2})", raw))


def _fight_relevance_datetime(raw_value) -> datetime | None:
    parsed = _normalize_upcoming_event_datetime(_parse_upcoming_event_datetime(raw_value))
    if parsed is None:
        return None
    if _raw_datetime_has_time(raw_value):
        return parsed

    day = _coerce_fight_matrix_day(raw_value, allow_raw_prefix=False)
    if not day:
        return parsed
    local_end_of_day = (
        datetime.fromisoformat(day).replace(tzinfo=FIGHT_MATRIX_TIMEZONE)
        + timedelta(days=1)
        - timedelta(microseconds=1)
    )
    return local_end_of_day.astimezone(timezone.utc)


def _select_fight_relevance_datetime(fight: dict) -> datetime | None:
    card_day = _coerce_fight_matrix_day(fight.get("card_date"), allow_raw_prefix=False)

    for key in ("event_date", "commence_time", "market_event_date"):
        raw = fight.get(key)
        if not _raw_datetime_has_time(raw):
            continue
        candidate = _fight_relevance_datetime(raw)
        if candidate is None:
            continue
        if card_day and candidate.astimezone(FIGHT_MATRIX_TIMEZONE).date().isoformat() != card_day:
            continue
        return candidate

    for key in ("card_date", "event_date", "commence_time", "market_event_date"):
        parsed = _fight_relevance_datetime(fight.get(key))
        if parsed is not None:
            return parsed
    return None


def _fight_is_relevant(fight: dict, cutoff: datetime) -> bool:
    """Return True if the fight's event date is >= cutoff.

    Fights with no parseable date are excluded — legitimate upcoming fights
    always have dates from predictions or market data.
    """
    parsed = _select_fight_relevance_datetime(fight)
    if parsed is None:
        return False
    normalized_cutoff = _normalize_upcoming_event_datetime(cutoff)
    if normalized_cutoff is None:
        return False
    return parsed >= normalized_cutoff


def _prediction_matrix_rows() -> list[dict]:
    try:
        payload = _load_prediction_payload(include_global_feature_importance=False)
        predictions = payload.get("predictions", [])
    except Exception as e:
        logger.warning("Failed to load predictions for tracker matrix: %s", e)
        predictions = []

    rows = []
    seen = set()
    for pred in predictions:
        fighter_a = str(pred.get("fighter_a", "") or "")
        fighter_b = str(pred.get("fighter_b", "") or "")
        event_date = str(pred.get("event_date") or pred.get("market_event_date") or "")
        card_date = _row_card_date(pred)
        key = _fight_matrix_key(fighter_a, fighter_b, event_date, card_date=card_date)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "event_date": event_date,
                "commence_time": str(pred.get("commence_time") or ""),
                "market_event_date": str(pred.get("market_event_date") or pred.get("event_date") or ""),
                "card_date": _coerce_fight_matrix_day(card_date, allow_raw_prefix=False),
                "event_group_date": _fight_matrix_event_group_date(
                    pred.get("market_event_date") or pred.get("event_date") or "",
                    card_date=card_date,
                ),
                "event_title": str(pred.get("event_title") or ""),
                "weight_class": str(pred.get("weight_class") or ""),
                "value_fighter": str(pred.get("value_fighter") or ""),
                "predicted_winner": str(pred.get("predicted_winner") or ""),
                "prob_a": _coerce_prediction_float(pred.get("prob_a")),
                "prob_b": _coerce_prediction_float(pred.get("prob_b")),
                "a_market_prob": _coerce_prediction_float(pred.get("a_market_prob")),
                "b_market_prob": _coerce_prediction_float(pred.get("b_market_prob")),
                "no_odds_prob_a": _coerce_prediction_float(pred.get("no_odds_prob_a")),
                "no_odds_prob_b": _coerce_prediction_float(pred.get("no_odds_prob_b")),
                "edge_a": _coerce_prediction_float(pred.get("edge_a")),
                "edge_b": _coerce_prediction_float(pred.get("edge_b")),
            }
        )
    return rows


def _build_market_event_date_hints(*row_groups) -> dict[frozenset[str], list[tuple[datetime, str]]]:
    hints: dict[frozenset[str], list[tuple[datetime, str]]] = defaultdict(list)

    for rows in row_groups:
        for row in rows:
            fighter_a = str(row.get("fighter_a", "") or "")
            fighter_b = str(row.get("fighter_b", "") or "")
            if not fighter_a or not fighter_b:
                continue

            raw_market_event_date = str(row.get("market_event_date") or "").strip()
            if not raw_market_event_date:
                continue

            parsed = _normalize_upcoming_event_datetime(
                _parse_upcoming_event_datetime(raw_market_event_date)
            )
            if parsed is None:
                continue

            pair_key = frozenset({_normalize_name(fighter_a), _normalize_name(fighter_b)})
            if not pair_key:
                continue

            candidates = hints[pair_key]
            if any(existing_raw == raw_market_event_date for _, existing_raw in candidates):
                continue
            candidates.append((parsed, raw_market_event_date))

    for candidates in hints.values():
        candidates.sort(key=lambda item: item[0])

    return hints


def _build_card_date_hints(*row_groups) -> dict[frozenset[str], list[tuple[datetime, str]]]:
    hints: dict[frozenset[str], list[tuple[datetime, str]]] = defaultdict(list)

    for rows in row_groups:
        for row in rows:
            fighter_a = str(row.get("fighter_a") or row.get("fighter") or "")
            fighter_b = str(row.get("fighter_b") or row.get("opponent") or "")
            if not fighter_a or not fighter_b:
                continue

            card_day = _coerce_fight_matrix_day(
                _row_card_date(row),
                allow_raw_prefix=False,
            )
            if not card_day:
                continue

            pair_key = frozenset({_normalize_name(fighter_a), _normalize_name(fighter_b)})
            if not pair_key:
                continue

            parsed_candidates: list[datetime] = []
            for raw_date in (
                row.get("market_event_date"),
                row.get("event_date"),
                row.get("commence_time"),
                card_day,
            ):
                parsed = _normalize_upcoming_event_datetime(
                    _parse_upcoming_event_datetime(raw_date)
                )
                if parsed is not None:
                    parsed_candidates.append(parsed)

            candidates = hints[pair_key]
            for parsed in parsed_candidates:
                if any(existing_raw == card_day and existing_ts == parsed for existing_ts, existing_raw in candidates):
                    continue
                candidates.append((parsed, card_day))

    for candidates in hints.values():
        candidates.sort(key=lambda item: item[0])

    return hints


def _resolve_card_date_hint(
    *,
    fighter_a: str,
    fighter_b: str,
    event_date: str,
    market_event_date: str,
    card_date: str,
    hints: dict[frozenset[str], list[tuple[datetime, str]]],
) -> str:
    explicit = _coerce_fight_matrix_day(card_date, allow_raw_prefix=False)
    if explicit:
        return explicit

    pair_key = frozenset({_normalize_name(fighter_a), _normalize_name(fighter_b)})
    if not pair_key:
        return ""

    candidates = hints.get(pair_key, [])
    if not candidates:
        return ""

    event_ts = _normalize_upcoming_event_datetime(
        _parse_upcoming_event_datetime(market_event_date or event_date)
    )
    unique_days = sorted({candidate_day for _, candidate_day in candidates})
    if event_ts is None:
        return unique_days[0] if len(unique_days) == 1 else ""

    best_day = ""
    best_gap = None
    for candidate_ts, candidate_day in candidates:
        gap_seconds = abs((candidate_ts - event_ts).total_seconds())
        if gap_seconds > 48 * 3600:
            continue
        if best_gap is None or gap_seconds < best_gap:
            best_gap = gap_seconds
            best_day = candidate_day
    return best_day


def _resolve_market_event_date_hint(
    *,
    fighter_a: str,
    fighter_b: str,
    event_date: str,
    market_event_date: str,
    hints: dict[frozenset[str], list[tuple[datetime, str]]],
) -> str:
    explicit = str(market_event_date or "").strip()
    if explicit:
        return explicit

    pair_key = frozenset({_normalize_name(fighter_a), _normalize_name(fighter_b)})
    if not pair_key:
        return ""

    event_ts = _normalize_upcoming_event_datetime(_parse_upcoming_event_datetime(event_date))
    if event_ts is None:
        return ""

    best_raw = ""
    best_gap = None
    for candidate_ts, candidate_raw in hints.get(pair_key, []):
        gap_seconds = abs((candidate_ts - event_ts).total_seconds())
        if gap_seconds > 36 * 3600:
            continue
        if best_gap is None or gap_seconds < best_gap:
            best_gap = gap_seconds
            best_raw = candidate_raw
    return best_raw


def _build_trader_bet_index(
    bets,
    *,
    market_event_date_hints=None,
    card_date_hints=None,
):
    index = {}
    ordered = sorted(
        (dict(bet) for bet in bets),
        key=lambda b: b.get("placed_at", ""),
        reverse=True,
    )
    for bet in ordered:
        trader = bet.get("trader") or _trader_label_from_path(bet.get("_ledger_path", ""))
        resolved_market_event_date = _resolve_market_event_date_hint(
            fighter_a=str(bet.get("fighter") or bet.get("fighter_a") or ""),
            fighter_b=str(bet.get("opponent") or bet.get("fighter_b") or ""),
            event_date=str(bet.get("event_date") or ""),
            market_event_date=str(bet.get("market_event_date") or ""),
            hints=market_event_date_hints or {},
        )
        if resolved_market_event_date and not bet.get("market_event_date"):
            bet["market_event_date"] = resolved_market_event_date
        card_date = _resolve_card_date_hint(
            fighter_a=str(bet.get("fighter") or bet.get("fighter_a") or ""),
            fighter_b=str(bet.get("opponent") or bet.get("fighter_b") or ""),
            event_date=str(bet.get("event_date") or ""),
            market_event_date=resolved_market_event_date or str(bet.get("market_event_date") or ""),
            card_date=_row_card_date(bet),
            hints=card_date_hints or {},
        )
        key = (
            trader,
            *_fight_matrix_key(
                str(bet.get("fighter") or bet.get("fighter_a") or ""),
                str(bet.get("opponent") or bet.get("fighter_b") or ""),
                str(bet.get("market_event_date") or bet.get("event_date") or ""),
                card_date=card_date,
            ),
        )
        if key not in index:
            index[key] = bet
    return index


def _build_prediction_row_index(rows):
    index = {}
    for row in rows:
        card_date = _row_card_date(row)
        key = _fight_matrix_key(
            str(row.get("fighter_a", "") or ""),
            str(row.get("fighter_b", "") or ""),
            str(row.get("market_event_date") or row.get("event_date") or ""),
            card_date=card_date,
        )
        if key not in index:
            index[key] = row
    return index


def _prediction_fields_for_fighter(prediction_row: dict | None, fighter_name: str) -> dict | None:
    if not prediction_row or not fighter_name:
        return None

    norm_fighter = _normalize_name(fighter_name)
    norm_a = _normalize_name(prediction_row.get("fighter_a"))
    norm_b = _normalize_name(prediction_row.get("fighter_b"))
    if norm_fighter == norm_a:
        return {
            "fighter": str(prediction_row.get("fighter_a") or fighter_name),
            "model_prob": _coerce_prediction_float(prediction_row.get("prob_a")),
            "market_prob": _coerce_prediction_float(prediction_row.get("a_market_prob")),
            "no_odds_prob": _coerce_prediction_float(prediction_row.get("no_odds_prob_a")),
            "edge": _coerce_prediction_float(prediction_row.get("edge_a")),
        }
    if norm_fighter == norm_b:
        return {
            "fighter": str(prediction_row.get("fighter_b") or fighter_name),
            "model_prob": _coerce_prediction_float(prediction_row.get("prob_b")),
            "market_prob": _coerce_prediction_float(prediction_row.get("b_market_prob")),
            "no_odds_prob": _coerce_prediction_float(prediction_row.get("no_odds_prob_b")),
            "edge": _coerce_prediction_float(prediction_row.get("edge_b")),
        }
    return None


def _fallback_matrix_trade_reason(
    *,
    trader: str,
    fighter_name: str,
    prediction_row: dict | None,
) -> str | None:
    fields = _prediction_fields_for_fighter(prediction_row, fighter_name)
    if not fields:
        return None

    fighter_label = fields["fighter"]
    model_prob = fields["model_prob"]
    market_prob = fields["market_prob"]
    edge = fields["edge"]
    no_odds_prob = fields["no_odds_prob"]

    if trader == "S" and model_prob is not None and market_prob is not None and edge is not None:
        return f"Model {model_prob:.0%} vs market {market_prob:.0%}, {edge:.1%} edge"
    if trader == "C" and model_prob is not None and market_prob is not None and no_odds_prob is not None:
        return (
            f"Conviction signal on {fighter_label}: model {model_prob:.0%}, "
            f"no-odds {no_odds_prob:.0%}, market {market_prob:.0%}, positive EV confirmed"
        )
    return None


def _build_operator_block_index(
    decisions,
    *,
    market_event_date_hints=None,
    card_date_hints=None,
):
    index = {}
    ordered = sorted(decisions, key=lambda d: d.get("timestamp", ""), reverse=True)
    for decision in ordered:
        if str(decision.get("verdict", "")).upper() != "BLOCK":
            continue
        resolved_market_event_date = _resolve_market_event_date_hint(
            fighter_a=str(decision.get("fighter_a", "") or ""),
            fighter_b=str(decision.get("fighter_b", "") or ""),
            event_date=str(decision.get("event_date") or ""),
            market_event_date=str(decision.get("market_event_date") or ""),
            hints=market_event_date_hints or {},
        )
        card_date = _resolve_card_date_hint(
            fighter_a=str(decision.get("fighter_a", "") or ""),
            fighter_b=str(decision.get("fighter_b", "") or ""),
            event_date=str(decision.get("event_date") or ""),
            market_event_date=resolved_market_event_date or str(decision.get("market_event_date") or ""),
            card_date=_row_card_date(decision),
            hints=card_date_hints or {},
        )
        key = _fight_matrix_key(
            str(decision.get("fighter_a", "") or ""),
            str(decision.get("fighter_b", "") or ""),
            resolved_market_event_date or str(decision.get("event_date") or ""),
            card_date=card_date,
        )
        if key not in index:
            index[key] = decision
        for context in _decision_context_aliases(decision.get("decision_context")):
            context_key = (context, *key)
            if context_key not in index:
                index[context_key] = decision
    return index


def _build_operator_pass_index(
    decisions,
    *,
    market_event_date_hints=None,
    card_date_hints=None,
):
    index = {}
    ordered = sorted(decisions, key=lambda d: d.get("timestamp", ""), reverse=True)
    for decision in ordered:
        if str(decision.get("verdict", "")).upper() != "PASS":
            continue
        resolved_market_event_date = _resolve_market_event_date_hint(
            fighter_a=str(decision.get("fighter_a", "") or ""),
            fighter_b=str(decision.get("fighter_b", "") or ""),
            event_date=str(decision.get("event_date") or ""),
            market_event_date=str(decision.get("market_event_date") or ""),
            hints=market_event_date_hints or {},
        )
        card_date = _resolve_card_date_hint(
            fighter_a=str(decision.get("fighter_a", "") or ""),
            fighter_b=str(decision.get("fighter_b", "") or ""),
            event_date=str(decision.get("event_date") or ""),
            market_event_date=resolved_market_event_date or str(decision.get("market_event_date") or ""),
            card_date=_row_card_date(decision),
            hints=card_date_hints or {},
        )
        key = _fight_matrix_key(
            str(decision.get("fighter_a", "") or ""),
            str(decision.get("fighter_b", "") or ""),
            resolved_market_event_date or str(decision.get("event_date") or ""),
            card_date=card_date,
        )
        if key not in index:
            index[key] = decision
        for context in _decision_context_aliases(decision.get("decision_context")):
            context_key = (context, *key)
            if context_key not in index:
                index[context_key] = decision
    return index


def _lookup_operator_matrix_decision(index: dict, key, trader: str) -> dict | None:
    for context in [*(_decision_context_aliases(trader) or []), ""]:
        lookup_key = (context, *key) if context else key
        if lookup_key in index:
            return index[lookup_key]
    return None


def _build_tracker_decision_index(records, *, card_date_hints=None):
    latest_decisions = {}
    latest_outcomes = {}
    ordered = sorted(records, key=lambda r: r.get("timestamp", ""), reverse=True)

    def _visible_decision_priority(record: dict) -> int:
        status = str(record.get("status") or "").strip().lower()
        if record.get("pick") or status == "eligible":
            return 3
        if status in {"no_pick", "invalid_pick", "missing_market_prob", "missing_model_prob"}:
            return 2
        if status in {"event_started", "outside_window", "too_close", "no_market"}:
            return 1
        return 0

    for record in ordered:
        decision_id = str(record.get("decision_id", "") or "").strip()
        if not decision_id:
            continue
        if record.get("type") == "outcome":
            latest_outcomes.setdefault(decision_id, record)
        elif record.get("type") == "decision":
            existing = latest_decisions.get(decision_id)
            if (
                existing is None
                or _visible_decision_priority(record) > _visible_decision_priority(existing)
            ):
                latest_decisions[decision_id] = record

    index = {}
    for decision_id, decision in latest_decisions.items():
        card_date = _resolve_card_date_hint(
            fighter_a=str(decision.get("fighter_a", "") or ""),
            fighter_b=str(decision.get("fighter_b", "") or ""),
            event_date=str(decision.get("event_date") or ""),
            market_event_date=str(decision.get("market_event_date") or ""),
            card_date=_row_card_date(decision),
            hints=card_date_hints or {},
        )
        key = (
            str(decision.get("trader", "") or ""),
            *_fight_matrix_key(
                str(decision.get("fighter_a", "") or ""),
                str(decision.get("fighter_b", "") or ""),
                str(decision.get("market_event_date") or decision.get("event_date") or ""),
                card_date=card_date,
            ),
        )
        merged = dict(decision)
        outcome = latest_outcomes.get(decision_id)
        if outcome:
            merged["outcome"] = outcome
        index.setdefault(key, merged)

    return index


def _format_sc_matrix_cell(
    *,
    trader: str,
    ledger_bet: dict | None,
    operator_decision: dict | None,
    block_decision: dict | None,
    decisions_index: dict,
    prediction_row: dict | None = None,
) -> dict:
    default_text = "No value edge" if trader == "S" else "No conviction signal"
    if ledger_bet:
        decision = _match_decision_to_bet(ledger_bet, decisions_index)
        trade_rationale = (
            ledger_bet.get("reason")
            or (decision or {}).get("trade_reason")
            or _fallback_matrix_trade_reason(
                trader=trader,
                fighter_name=str(
                    ledger_bet.get("fighter")
                    or ledger_bet.get("bet_on")
                    or ""
                ),
                prediction_row=prediction_row,
            )
        )
        return {
            "status": "bet",
            "text": ledger_bet.get("fighter") or ledger_bet.get("bet_on") or "Bet placed",
            "edge": ledger_bet.get("edge"),
            "rationale": trade_rationale or (decision or {}).get("rationale"),
            "operator_rationale": (decision or {}).get("rationale"),
            "operator_verdict": (decision or {}).get("verdict"),
            "operator_confidence": (decision or {}).get("confidence"),
        }
    if block_decision:
        return {
            "status": "blocked",
            "text": "Blocked",
            "rationale": block_decision.get("rationale"),
            "operator_verdict": block_decision.get("verdict"),
            "operator_confidence": block_decision.get("confidence"),
        }
    if operator_decision:
        trade_rationale = (
            operator_decision.get("trade_reason")
            or _fallback_matrix_trade_reason(
                trader=trader,
                fighter_name=str(operator_decision.get("bet_on") or ""),
                prediction_row=prediction_row,
            )
        )
        return {
            "status": "eligible",
            "text": operator_decision.get("bet_on") or "Candidate",
            "edge": operator_decision.get("edge"),
            "rationale": trade_rationale or operator_decision.get("rationale"),
            "operator_rationale": operator_decision.get("rationale"),
            "operator_verdict": operator_decision.get("verdict"),
            "operator_confidence": operator_decision.get("confidence"),
        }
    if prediction_row is not None and not _prediction_row_has_market(prediction_row):
        return {
            "status": "no_market",
            "text": "No market matched",
            "rationale": "No active Polymarket market was matched for this fight.",
        }
    return {"status": "no_signal", "text": default_text, "rationale": None}


def _prediction_row_has_market(row: dict | None) -> bool:
    if not row:
        return False
    return row.get("a_market_prob") is not None or row.get("b_market_prob") is not None


def _format_tracker_matrix_cell(
    entry: dict | None,
    *,
    fallback_text: str,
    ledger_bet: dict | None = None,
    prediction_row: dict | None = None,
) -> dict:
    if ledger_bet:
        return {
            "status": "bet",
            "text": ledger_bet.get("fighter") or ledger_bet.get("bet_on") or "Bet placed",
            "rationale": ledger_bet.get("reason"),
            "edge": ledger_bet.get("edge"),
            "bet_placed": True,
            "order_status": ledger_bet.get("placement_state") or ledger_bet.get("status"),
            "order_type": ledger_bet.get("order_type"),
        }

    if not entry:
        if prediction_row is not None and not _prediction_row_has_market(prediction_row):
            return {
                "status": "no_market",
                "text": "No market matched",
                "rationale": "No active Polymarket market was matched for this fight.",
            }
        return {"status": "pending", "text": fallback_text, "rationale": None}

    outcome = entry.get("outcome") or {}
    return {
        "status": entry.get("status") or "pending",
        "text": entry.get("pick") or entry.get("summary") or fallback_text,
        "rationale": entry.get("rationale"),
        "confidence": entry.get("confidence"),
        "edge": entry.get("edge"),
        "sources": entry.get("sources", []),
        "bet_placed": outcome.get("bet_placed"),
        "order_status": outcome.get("order_status"),
        "order_type": outcome.get("order_type"),
        "error": outcome.get("error"),
    }


def _first_present(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
            continue
        return value
    return None


def _unique_labels(values) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for value in values:
        label = str(value or "").strip()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def _bet_trader_labels(bet: dict) -> list[str]:
    explicit = bet.get("trader")
    if isinstance(explicit, list):
        labels = explicit
    else:
        labels = [explicit, _trader_label_from_path(bet.get("_ledger_path", ""))]
    return _unique_labels(str(label).strip().upper() for label in labels if label)


def _sanitize_open_bet_display_metrics(bet: dict) -> dict:
    normalized = dict(bet)
    if "G" not in _bet_trader_labels(normalized):
        return normalized

    signal_confidence = _coerce_prediction_float(normalized.get("signal_confidence"))
    model_prob = _coerce_prediction_float(normalized.get("model_prob"))
    if signal_confidence is None and model_prob is not None:
        # Legacy Gemini rows stored research confidence in model_prob.
        signal_confidence = model_prob
    if signal_confidence is not None:
        normalized["signal_confidence"] = min(
            max(signal_confidence, 0.0),
            GEMINI_TRACKER_CONFIDENCE_CAP,
        )
    return normalized


def _open_bet_model_label(*, model_prob, signal_confidence) -> str:
    if model_prob is None and signal_confidence is not None:
        return "Confidence"
    return "Model"


def _is_open_bet_metric_placeholder(bet: dict) -> bool:
    if str(bet.get("order_type") or "").strip().lower() != "imported":
        return False
    return (
        _coerce_prediction_float(bet.get("model_prob"), 0.0) == 0.0
        and _coerce_prediction_float(bet.get("edge"), 0.0) == 0.0
        and not str(bet.get("reason") or "").strip()
    )


def _weighted_open_bet_metric(
    bets: list[dict],
    field: str,
    *,
    primary_weight: str = "amount",
    fallback_weight: str = "shares",
):
    total_weight = 0.0
    weighted_sum = 0.0
    for bet in bets:
        if field in {"edge", "model_prob", "market_prob"} and _is_open_bet_metric_placeholder(bet):
            continue
        if field in {"edge", "model_prob"} and "G" in _bet_trader_labels(bet):
            continue
        raw_value = bet.get(field)
        if raw_value in (None, ""):
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        weight = _safe_float(bet.get(primary_weight), 0.0)
        if weight <= 0.0:
            weight = _safe_float(bet.get(fallback_weight), 0.0)
        if weight <= 0.0:
            continue
        total_weight += weight
        weighted_sum += value * weight

    if total_weight > 0.0:
        return weighted_sum / total_weight

    for bet in bets:
        if field in {"edge", "model_prob", "market_prob"} and _is_open_bet_metric_placeholder(bet):
            continue
        if field in {"edge", "model_prob"} and "G" in _bet_trader_labels(bet):
            continue
        raw_value = bet.get(field)
        if raw_value in (None, ""):
            continue
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            continue
    return None


def _aggregate_open_bet_position(pos: dict, matched_bets: list[dict]) -> dict:
    matched = [_sanitize_open_bet_display_metrics(bet) for bet in matched_bets if bet]
    latest = sorted(matched, key=lambda bet: str(bet.get("placed_at") or ""), reverse=True)
    earliest = list(reversed(latest))

    def _latest_value(field: str):
        return _first_present(*(bet.get(field) for bet in latest))

    trader_labels = _unique_labels(
        bet.get("trader") or _trader_label_from_path(bet.get("_ledger_path", ""))
        for bet in latest
    )
    tracked_shares = sum(_safe_float(bet.get("shares"), 0.0) for bet in matched)
    tracked_amount = sum(_safe_float(bet.get("amount"), 0.0) for bet in matched)
    live_size = _safe_float(pos.get("size"), 0.0)
    manual_shares = max(0.0, live_size - tracked_shares)
    manual_untracked = bool(matched) and manual_shares >= OPEN_BET_DISPLAY_SIZE_THRESHOLD

    fighter = _latest_value("fighter") or pos.get("side")
    opponent = _latest_value("opponent") or pos.get("opposite_side")
    event_date = _latest_value("event_date") or _upcoming_event_day_key(pos.get("end_date"))
    model_prob = _weighted_open_bet_metric(matched, "model_prob")
    signal_confidence = _weighted_open_bet_metric(matched, "signal_confidence")

    return {
        "id": _latest_value("id"),
        "merged_bet_ids": [bet.get("id") for bet in matched if bet.get("id") is not None],
        "fighter": fighter,
        "opponent": opponent,
        "side": _latest_value("side"),
        "amount": pos.get("invested") if pos.get("invested") is not None else (tracked_amount or None),
        "price": _weighted_open_bet_metric(matched, "price"),
        "shares": pos.get("size"),
        "model_prob": model_prob,
        "signal_confidence": signal_confidence,
        "market_prob": _weighted_open_bet_metric(matched, "market_prob"),
        "edge": _weighted_open_bet_metric(matched, "edge"),
        "reason": _latest_value("reason"),
        "placed_at": _first_present(*(bet.get("placed_at") for bet in earliest)),
        "event_date": event_date,
        "order_type": _latest_value("order_type"),
        "trader": trader_labels[0] if len(trader_labels) == 1 else None,
        "traders": trader_labels,
        "model_label": _open_bet_model_label(
            model_prob=model_prob,
            signal_confidence=signal_confidence,
        ),
        "token_id": str(pos.get("token_id") or "").strip() or None,
        "market_id": _latest_value("market_id"),
        "market": _first_present(pos.get("market"), _latest_value("market")),
        "sport": _latest_value("sport") or _classify_sport_from_position(pos),
        "cur_price": pos.get("cur_price"),
        "unrealized_pnl": pos.get("unrealized_pnl"),
        "pnl_pct": pos.get("pnl_pct"),
        "invested": pos.get("invested"),
        "value": pos.get("value"),
        "avg_price": pos.get("avg_price"),
        "size": pos.get("size"),
        "event_slug": pos.get("event_slug"),
        "tracked_amount": tracked_amount,
        "tracked_shares": tracked_shares,
        "manual_shares": manual_shares if manual_untracked else 0.0,
        "manual_untracked": manual_untracked,
        "matched_bet_count": len(matched),
        "unmatched": not matched,
    }


def _synthetic_open_position_from_bets(matched_bets: list[dict]) -> dict:
    latest = sorted(
        (dict(bet) for bet in matched_bets if bet),
        key=lambda bet: str(bet.get("placed_at") or ""),
        reverse=True,
    )
    if not latest:
        return {}

    tracked_shares = sum(_safe_float(bet.get("shares"), 0.0) for bet in latest)
    tracked_amount = sum(_safe_float(bet.get("amount"), 0.0) for bet in latest)
    avg_price = _weighted_open_bet_metric(latest, "price")
    current_price = _first_present(*(bet.get("cur_price") for bet in latest))
    current_price_value = _safe_float(current_price, math.nan)
    current_value = (
        tracked_shares * current_price_value
        if tracked_shares > 0.0 and math.isfinite(current_price_value)
        else None
    )
    unrealized_pnl = (
        current_value - tracked_amount if current_value is not None else 0.0
    )
    pnl_pct = (
        (unrealized_pnl / tracked_amount) * 100.0
        if tracked_amount > 0.0 and current_value is not None
        else 0.0
    )

    newest = latest[0]
    return {
        "token_id": str(newest.get("token_id") or "").strip() or None,
        "market": newest.get("market"),
        "side": newest.get("fighter") or newest.get("side"),
        "opposite_side": newest.get("opponent"),
        "size": tracked_shares,
        "avg_price": avg_price,
        "cur_price": current_price if current_price is not None else avg_price,
        "invested": tracked_amount if tracked_amount > 0.0 else None,
        "value": current_value,
        "unrealized_pnl": unrealized_pnl,
        "pnl_pct": pnl_pct,
        "realized_pnl": 0.0,
        "event_slug": newest.get("event_slug"),
        "end_date": newest.get("event_date"),
    }


def _compute_open_bets_enriched():
    """Build the dashboard open-bets payload from live Polymarket positions."""
    # 1. Open bets from ledger (exclude dry_run)
    ledger = load_all_trader_ledgers()
    open_bets = [b for b in ledger.open_bets if not b.get("dry_run")]
    for bet in open_bets:
        bet["sport"] = _classify_sport_from_ledger_path(bet.get("_ledger_path", ""))
        bet["trader"] = _trader_label_from_path(bet.get("_ledger_path", ""))

    # 2. Live positions from Polymarket
    raw_live_tids = None
    display_positions = []
    raw_pnl, live_source = _load_live_pnl_snapshot()
    if live_source != "unavailable":
        if live_source == "live":
            raw_live_tids = {
                str(pos.get("token_id") or "").strip()
                for pos in raw_pnl.get("positions", [])
                if str(pos.get("token_id") or "").strip()
            }
        display_pnl = _dashboard_live_pnl_from_raw(raw_pnl)
        display_positions = [dict(pos) for pos in display_pnl.get("positions", [])]

    # 2b. Reconcile only from a confirmed live snapshot. A stale cached snapshot
    # can legitimately miss newly opened positions and must not mutate the ledger.
    if raw_live_tids is not None:
        try:
            from src.strategy.duo_trader import get_all_trader_ledgers
            total_reconciled = 0
            for _, path in get_all_trader_ledgers():
                if Path(path).exists():
                    _ledger = BetLedger(path=path)
                    reconciled = auto_reconcile_sold_positions(_ledger, raw_live_tids)
                    if reconciled:
                        logger.info("Reconciled %d sold positions from %s", reconciled, path)
                        total_reconciled += reconciled
            if total_reconciled:
                # Reload ledger after reconciliation
                ledger = load_all_trader_ledgers()
                open_bets = [b for b in ledger.open_bets if not b.get("dry_run")]
                for bet in open_bets:
                    bet["sport"] = _classify_sport_from_ledger_path(bet.get("_ledger_path", ""))
                    bet["trader"] = _trader_label_from_path(bet.get("_ledger_path", ""))
        except Exception as e:
            logger.warning("Failed to reconcile sold positions: %s", e)

    # 3. Operator decisions (most recent 200)
    decisions_index = {}
    try:
        from src.strategy.llm_operator import load_decision_log
        all_decisions = load_decision_log()
        # Sort newest first, take last 200
        all_decisions.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
        decisions_index = _build_decisions_index(all_decisions[:200])
    except Exception as e:
        logger.warning("Failed to load operator decisions for enriched bets: %s", e)

    # 4. Match ledger metadata onto the live Polymarket positions.
    open_bets_by_token = defaultdict(list)
    for bet in open_bets:
        token_id = str(bet.get("token_id") or "").strip()
        if token_id:
            open_bets_by_token[token_id].append(bet)

    enriched = []
    if display_positions:
        for pos in display_positions:
            token_id = str(pos.get("token_id") or "").strip()
            matched_bets = open_bets_by_token.get(token_id, [])
            entry = _aggregate_open_bet_position(pos, matched_bets)

            decision = _match_decision_to_bet(entry, decisions_index)
            if decision:
                entry["operator_rationale"] = decision.get("rationale")
                entry["operator_verdict"] = decision.get("verdict")
                entry["operator_prob"] = decision.get("operator_prob")
                entry["operator_confidence"] = decision.get("confidence")
                entry["research_summary"] = decision.get("research_summary")
                entry["risk_flags"] = decision.get("risk_flags")

            enriched.append(entry)
    else:
        fallback_groups = defaultdict(list)
        for bet in open_bets:
            group_key = str(bet.get("token_id") or "").strip() or f"bet:{bet.get('id')}"
            fallback_groups[group_key].append(bet)

        for matched_bets in fallback_groups.values():
            synthetic_position = _synthetic_open_position_from_bets(matched_bets)
            entry = _aggregate_open_bet_position(synthetic_position, matched_bets)

            decision = _match_decision_to_bet(entry, decisions_index)
            if decision:
                entry["operator_rationale"] = decision.get("rationale")
                entry["operator_verdict"] = decision.get("verdict")
                entry["operator_prob"] = decision.get("operator_prob")
                entry["operator_confidence"] = decision.get("confidence")
                entry["research_summary"] = decision.get("research_summary")
                entry["risk_flags"] = decision.get("risk_flags")

            enriched.append(entry)

    return {"bets": enriched, "unmatched_positions": [], "_pnl_source": live_source}


def _profile_trade_timestamp_iso(trade: dict) -> str | None:
    raw_timestamp = trade.get("timestamp")
    parsed: datetime | None = None

    if raw_timestamp not in (None, ""):
        try:
            parsed = datetime.fromtimestamp(float(raw_timestamp), tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            parsed = None

    if parsed is None:
        raw_timestamp = str(raw_timestamp or "").strip()
        if not raw_timestamp:
            return None
        try:
            parsed = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
        except ValueError:
            return None

    return parsed.astimezone(timezone.utc).isoformat()


def _profile_trade_sort_key(trade: dict) -> str:
    return _profile_trade_timestamp_iso(trade) or ""


def _profile_trader_label(values) -> str | None:
    labels = _unique_labels(values)
    if not labels:
        return None
    if len(labels) == 1:
        return labels[0]
    return " / ".join(labels)


def _profile_position_status(realized_pnl: float) -> str:
    if realized_pnl > PROFILE_PNL_EPSILON:
        return "won"
    if realized_pnl < -PROFILE_PNL_EPSILON:
        return "lost"
    return "sold"


def _profile_activity_loss_rows(
    *,
    trades_by_asset: dict[str, list[dict]],
    active_token_ids: set[str],
    represented_token_ids: set[str],
) -> list[dict]:
    """Infer missing losses when Polymarket activity has buys but no position."""
    rows: list[dict] = []
    for token_id, entries in trades_by_asset.items():
        token_id = str(token_id or "").strip()
        if not token_id or token_id in active_token_ids or token_id in represented_token_ids:
            continue

        entries = sorted(entries, key=_profile_trade_sort_key)
        buy_trades = [
            trade for trade in entries
            if str(trade.get("type") or "").upper() == "TRADE"
            and str(trade.get("side") or "").upper() == "BUY"
        ]
        if not buy_trades:
            continue

        has_exit = any(
            (
                str(trade.get("type") or "").upper() == "REDEEM"
                or (
                    str(trade.get("type") or "").upper() == "TRADE"
                    and str(trade.get("side") or "").upper() == "SELL"
                )
            )
            for trade in entries
        )
        if has_exit:
            continue

        buy_amount = sum(_safe_float(trade.get("usdcSize"), 0.0) for trade in buy_trades)
        if buy_amount <= PROFILE_PNL_EPSILON:
            continue

        buy_shares = sum(_safe_float(trade.get("size"), 0.0) for trade in buy_trades)
        first_buy = buy_trades[0]
        latest = entries[-1]
        event = latest.get("title") or latest.get("market") or latest.get("question") or ""
        event_slug = latest.get("eventSlug") or latest.get("event_slug") or ""
        slug = latest.get("slug") or ""
        rows.append({
            "fighter": latest.get("outcome") or latest.get("asset") or event or "Unknown",
            "opponent": latest.get("oppositeOutcome") or "",
            "event": event,
            "amount": buy_amount,
            "shares": buy_shares,
            "result_pnl": -buy_amount,
            "status": "lost",
            "trade_side": "lost",
            "placed_at": _profile_trade_timestamp_iso(first_buy),
            "settled_at": latest.get("endDate") or latest.get("end_date"),
            "event_date": latest.get("endDate") or latest.get("end_date"),
            "token_id": token_id,
            "condition_id": str(latest.get("conditionId") or latest.get("condition_id") or "").strip() or None,
            "event_slug": event_slug,
            "slug": slug,
            "sport": _classify_sport_from_position({
                "market": event,
                "event_slug": event_slug,
                "slug": slug,
                "icon": latest.get("icon"),
            }),
            "trader_label": None,
            "traders": [],
            "avg_price": buy_amount / buy_shares if buy_shares > PROFILE_PNL_EPSILON else None,
            "source": "polymarket_activity",
        })

    return rows


def _index_profile_ledger_bets(bets: list[dict]) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    by_token: defaultdict[str, list[dict]] = defaultdict(list)
    by_condition: defaultdict[str, list[dict]] = defaultdict(list)

    for bet in bets:
        token_id = str(bet.get("token_id") or "").strip()
        condition_id = str(bet.get("condition_id") or bet.get("conditionId") or "").strip()
        if token_id:
            by_token[token_id].append(bet)
        if condition_id:
            by_condition[condition_id].append(bet)

    return dict(by_token), dict(by_condition)


def _match_profile_ledger_bets(
    *,
    token_id: str,
    condition_id: str,
    outcome: str,
    ledger_by_token: dict[str, list[dict]],
    ledger_by_condition: dict[str, list[dict]],
) -> list[dict]:
    matched: list[dict] = []
    seen: set[object] = set()

    for bet in ledger_by_token.get(token_id, []):
        marker = bet.get("id", id(bet))
        if marker in seen:
            continue
        seen.add(marker)
        matched.append(bet)

    outcome_norm = _normalize_name(outcome)
    for bet in ledger_by_condition.get(condition_id, []):
        marker = bet.get("id", id(bet))
        if marker in seen:
            continue

        fighter_norm = _normalize_name(bet.get("fighter"))
        side_norm = _normalize_name(bet.get("side"))
        if outcome_norm and (fighter_norm or side_norm):
            if fighter_norm and fighter_norm != outcome_norm:
                continue
            if not fighter_norm and side_norm and side_norm != outcome_norm:
                continue

        seen.add(marker)
        matched.append(bet)

    return matched


def _index_profile_trades(
    trades: list[dict],
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    by_asset: defaultdict[str, list[dict]] = defaultdict(list)
    redeem_by_condition: defaultdict[str, list[dict]] = defaultdict(list)

    for trade in trades:
        asset = str(trade.get("asset") or "").strip()
        condition_id = str(trade.get("conditionId") or trade.get("condition_id") or "").strip()
        trade_type = str(trade.get("type") or "").upper()

        if asset:
            by_asset[asset].append(trade)
        if trade_type == "REDEEM" and condition_id:
            redeem_by_condition[condition_id].append(trade)

    for entries in by_asset.values():
        entries.sort(key=_profile_trade_sort_key)
    for entries in redeem_by_condition.values():
        entries.sort(key=_profile_trade_sort_key)

    return dict(by_asset), dict(redeem_by_condition)


def _profile_closed_row(
    pos: dict,
    *,
    matched_bets: list[dict],
    trades: list[dict],
    redeem_trades: list[dict],
) -> dict:
    latest = sorted((dict(bet) for bet in matched_bets if bet), key=lambda bet: str(bet.get("placed_at") or ""), reverse=True)
    earliest = list(reversed(latest))

    def _latest_value(field: str):
        return _first_present(*(bet.get(field) for bet in latest))

    buy_trades = [trade for trade in trades if str(trade.get("side") or "").upper() == "BUY"]
    sell_trades = [trade for trade in trades if str(trade.get("side") or "").upper() == "SELL"]
    total_bought = _safe_float(pos.get("totalBought"), 0.0)
    avg_price = _safe_float(pos.get("avgPrice", pos.get("avg_price")), 0.0)
    buy_amount = sum(_safe_float(trade.get("usdcSize"), 0.0) for trade in buy_trades)
    if buy_amount <= 0.0:
        buy_amount = total_bought * avg_price
    if buy_amount <= 0.0:
        buy_amount = sum(_safe_float(bet.get("amount"), 0.0) for bet in matched_bets)

    buy_shares = sum(_safe_float(trade.get("size"), 0.0) for trade in buy_trades)
    if buy_shares <= 0.0:
        buy_shares = total_bought
    realized_pnl = _safe_float(pos.get("realizedPnl", pos.get("realized_pnl", pos.get("cashPnl"))), 0.0)
    placed_at = _profile_trade_timestamp_iso(buy_trades[0]) if buy_trades else _first_present(*(bet.get("placed_at") for bet in earliest))

    settled_at = None
    if redeem_trades:
        settled_at = _profile_trade_timestamp_iso(redeem_trades[-1])
    elif sell_trades:
        settled_at = _profile_trade_timestamp_iso(sell_trades[-1])
    if not settled_at:
        settled_at = _first_present(
            pos.get("closedAt"),
            pos.get("closed_at"),
            pos.get("endDate"),
            pos.get("end_date"),
        )

    trader_labels = _unique_labels(
        bet.get("trader") or _trader_label_from_path(bet.get("_ledger_path", ""))
        for bet in latest
    )
    event = pos.get("title") or pos.get("market") or pos.get("question") or ""
    event_slug = pos.get("eventSlug") or pos.get("event_slug") or ""
    slug = pos.get("slug") or ""
    fighter = _latest_value("fighter") or pos.get("outcome") or pos.get("asset") or event or "Unknown"
    opponent = _latest_value("opponent") or ""

    row = {
        "fighter": fighter,
        "opponent": opponent,
        "event": event,
        "amount": buy_amount,
        "shares": buy_shares,
        "result_pnl": realized_pnl,
        "status": _profile_position_status(realized_pnl),
        "trade_side": _profile_position_status(realized_pnl),
        "placed_at": placed_at,
        "settled_at": settled_at,
        "event_date": _latest_value("event_date") or pos.get("endDate") or pos.get("end_date"),
        "reason": _latest_value("reason"),
        "token_id": str(pos.get("asset") or "").strip() or None,
        "condition_id": str(pos.get("conditionId") or pos.get("condition_id") or "").strip() or None,
        "event_slug": event_slug,
        "slug": slug,
        "sport": _latest_value("sport") or _classify_sport_from_position({
            "market": event,
            "event_slug": event_slug,
            "slug": slug,
            "icon": pos.get("icon"),
        }),
        "trader_label": _profile_trader_label(trader_labels),
        "traders": trader_labels,
        "source": "polymarket",
    }

    odds = _latest_value("odds")
    if odds is not None:
        row["odds"] = odds
    clv = _latest_value("clv")
    if clv is not None:
        row["clv"] = clv

    return row


def _profile_open_row(open_bet: dict) -> dict:
    traders = _unique_labels(open_bet.get("traders") or [open_bet.get("trader")])
    amount = open_bet.get("invested")
    if amount in (None, ""):
        amount = open_bet.get("amount")

    row = {
        "id": open_bet.get("id"),
        "fighter": open_bet.get("fighter") or open_bet.get("side") or "Unknown",
        "opponent": open_bet.get("opponent") or open_bet.get("opposite_side") or "",
        "event": open_bet.get("market") or "",
        "amount": amount,
        "shares": open_bet.get("shares"),
        "status": "open",
        "trade_side": "open",
        "placed_at": open_bet.get("placed_at"),
        "event_date": open_bet.get("event_date"),
        "reason": open_bet.get("reason"),
        "token_id": str(open_bet.get("token_id") or "").strip() or None,
        "condition_id": str(open_bet.get("condition_id") or open_bet.get("conditionId") or "").strip() or None,
        "event_slug": open_bet.get("event_slug") or open_bet.get("eventSlug"),
        "slug": open_bet.get("slug"),
        "sport": open_bet.get("sport") or _classify_sport_from_position(open_bet),
        "trader_label": _profile_trader_label(traders),
        "traders": traders,
        "cur_price": open_bet.get("cur_price"),
        "price": open_bet.get("avg_price", open_bet.get("price")),
        "avg_price": open_bet.get("avg_price", open_bet.get("price")),
        "invested": open_bet.get("invested"),
        "value": open_bet.get("value"),
        "unrealized_pnl": open_bet.get("unrealized_pnl"),
        "pnl_pct": open_bet.get("pnl_pct"),
        "source": "polymarket",
    }

    for field in ("odds", "clv", "edge", "model_prob", "market_prob"):
        value = open_bet.get(field)
        if value is not None:
            row[field] = value

    return row


def _build_profile_summary(
    *,
    rows: list[dict],
    raw_live_pnl: dict,
    profile_snapshot: dict,
    live_source: str,
    degraded: bool,
) -> dict:
    open_rows = [row for row in rows if row.get("status") == "open"]
    settled_rows = [row for row in rows if row.get("status") != "open"]
    wins = sum(1 for row in settled_rows if row.get("status") == "won")
    losses = sum(1 for row in settled_rows if row.get("status") == "lost")
    total_wagered = sum(_safe_float(row.get("amount"), 0.0) for row in rows if row.get("amount") not in (None, ""))
    total_pnl = _safe_float(raw_live_pnl.get("total_pnl"), 0.0)
    profile_total_pnl = profile_snapshot.get("total_pnl")
    if profile_total_pnl is not None and math.isfinite(profile_total_pnl):
        total_pnl = profile_total_pnl
    realized_pnl = _safe_float(raw_live_pnl.get("realized_pnl"), 0.0)
    unrealized_pnl = _safe_float(raw_live_pnl.get("unrealized_pnl"), 0.0)
    positions_value = _safe_float(raw_live_pnl.get("current_value"), 0.0)
    profile_positions_value = profile_snapshot.get("positions_value")
    if profile_positions_value is not None and math.isfinite(profile_positions_value):
        positions_value = profile_positions_value
    profile_volume = profile_snapshot.get("profile_volume")

    summary = {
        "_canonical_profile": True,
        "_pnl_source": live_source,
        "_pnl_degraded": degraded,
        "total_bets": len(rows),
        "open_bets": len(open_rows),
        "settled_bets": len(settled_rows),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0.0,
        "total_wagered": total_wagered,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": total_pnl,
        "roi": total_pnl / total_wagered if total_wagered > 0 else 0.0,
        "open_invested": positions_value,
        "positions_value": positions_value,
    }
    if profile_total_pnl is not None and math.isfinite(profile_total_pnl):
        summary["profile_total_pnl"] = profile_total_pnl
    if profile_volume is not None and math.isfinite(profile_volume):
        summary["profile_volume"] = profile_volume
    largest_win = profile_snapshot.get("largest_win")
    if largest_win is not None and math.isfinite(largest_win):
        summary["profile_largest_win"] = largest_win
    predictions = profile_snapshot.get("predictions")
    if predictions:
        summary["profile_predictions"] = predictions
    username = profile_snapshot.get("username")
    if username:
        summary["profile_username"] = username
    return summary


def _compute_profile_bets_snapshot() -> dict:
    ledger = load_all_trader_ledgers()
    tracked_bets = [dict(bet) for bet in ledger.bets if not bet.get("dry_run")]
    ledger_by_token, ledger_by_condition = _index_profile_ledger_bets(tracked_bets)

    raw_live_pnl, live_source = _load_live_pnl_snapshot()
    degraded = live_source != "live"
    profile_snapshot, profile_source = _load_polymarket_profile_snapshot()
    if profile_source != "live":
        degraded = True

    open_snapshot = _cached("open-bets-enriched", PROFILE_BETS_CACHE_TTL, _compute_open_bets_enriched)
    open_rows = [_profile_open_row(row) for row in (open_snapshot or {}).get("bets", [])]

    monitor = _get_position_monitor()
    try:
        closed_positions = monitor.get_closed_positions(strict=(live_source == "live"))
    except PositionDataPartialError as exc:
        logger.warning("Profile bet history closed positions degraded: %s", exc)
        closed_positions = []
        degraded = True

    try:
        trades = monitor.get_trades(limit=PROFILE_TRADE_HISTORY_LIMIT)
    except Exception as exc:
        logger.warning("Profile bet history trade activity degraded: %s", exc)
        trades = []
        degraded = True

    trades_by_asset, redeem_by_condition = _index_profile_trades(trades)
    closed_rows: list[dict] = []
    represented_token_ids: set[str] = set()
    active_token_ids = {
        str(pos.get("token_id") or pos.get("asset") or "").strip()
        for pos in raw_live_pnl.get("positions", [])
        if str(pos.get("token_id") or pos.get("asset") or "").strip()
    }

    for pos in closed_positions:
        token_id = str(pos.get("asset") or "").strip()
        condition_id = str(pos.get("conditionId") or pos.get("condition_id") or "").strip()
        if token_id:
            represented_token_ids.add(token_id)
        matched_bets = _match_profile_ledger_bets(
            token_id=token_id,
            condition_id=condition_id,
            outcome=str(pos.get("outcome") or ""),
            ledger_by_token=ledger_by_token,
            ledger_by_condition=ledger_by_condition,
        )
        closed_rows.append(
            _profile_closed_row(
                pos,
                matched_bets=matched_bets,
                trades=trades_by_asset.get(token_id, []),
                redeem_trades=redeem_by_condition.get(condition_id, []),
            )
        )

    if live_source == "live":
        activity_loss_rows = _profile_activity_loss_rows(
            trades_by_asset=trades_by_asset,
            active_token_ids=active_token_ids,
            represented_token_ids=represented_token_ids,
        )
        closed_rows.extend(activity_loss_rows)

    if not closed_rows and live_source == "unavailable":
        fallback_rows = [dict(bet) for bet in tracked_bets]
        fallback_rows.sort(
            key=lambda row: str(row.get("settled_at") or row.get("placed_at") or ""),
            reverse=True,
        )
        fallback_summary = ledger.get_summary()
        fallback_summary["_canonical_profile"] = False
        fallback_summary["_pnl_source"] = live_source
        fallback_summary["_pnl_degraded"] = True
        return {"summary": fallback_summary, "bets": fallback_rows}

    rows = open_rows + closed_rows
    rows.sort(
        key=lambda row: str(row.get("settled_at") or row.get("placed_at") or row.get("event_date") or ""),
        reverse=True,
    )
    summary = _build_profile_summary(
        rows=rows,
        raw_live_pnl=raw_live_pnl,
        profile_snapshot=profile_snapshot,
        live_source=live_source,
        degraded=degraded,
    )
    summary["_profile_source"] = profile_source
    return {"summary": summary, "bets": rows}


def _profile_row_sport(row: dict) -> str:
    sport = str(row.get("sport") or "").strip().lower()
    if sport:
        return sport
    ledger_path = str(row.get("_ledger_path") or "").strip()
    if ledger_path:
        return _classify_sport_from_ledger_path(ledger_path)
    return _classify_sport_from_position({
        "market": row.get("event") or row.get("market") or row.get("title") or "",
        "event_slug": row.get("event_slug") or row.get("eventSlug"),
        "slug": row.get("slug"),
        "icon": row.get("icon"),
    })


def _scope_profile_bets_payload(payload: dict, sport: str) -> dict:
    sport = _normalize_sport_filter(sport)
    scoped = copy.deepcopy(payload or {})
    if sport == "all":
        return scoped

    rows = [
        row for row in scoped.get("bets", [])
        if _profile_row_sport(row) == sport
    ]
    base_summary = dict(scoped.get("summary") or {})
    profile_total_pnl = _safe_float(base_summary.get("profile_total_pnl"), math.nan)
    if not math.isfinite(profile_total_pnl) and base_summary.get("_canonical_profile"):
        profile_total_pnl = _safe_float(base_summary.get("total_pnl"), math.nan)
    open_rows = [row for row in rows if row.get("status") == "open"]
    settled_rows = [row for row in rows if row.get("status") != "open"]
    wins = sum(1 for row in settled_rows if row.get("status") == "won")
    losses = sum(1 for row in settled_rows if row.get("status") == "lost")
    realized = sum(_safe_float(row.get("result_pnl"), 0.0) for row in settled_rows)
    unrealized = sum(_safe_float(row.get("unrealized_pnl"), 0.0) for row in open_rows)
    open_invested = sum(
        _safe_float(row.get("invested", row.get("amount")), 0.0)
        for row in open_rows
    )
    positions_value = sum(
        _safe_float(row.get("value"), math.nan)
        if math.isfinite(_safe_float(row.get("value"), math.nan))
        else _safe_float(row.get("amount"), 0.0) + _safe_float(row.get("unrealized_pnl"), 0.0)
        for row in open_rows
    )
    total_wagered = sum(
        _safe_float(row.get("amount"), 0.0)
        for row in rows
        if row.get("amount") not in (None, "")
    )
    total_pnl = realized + unrealized
    base_summary.update({
        "sport": sport,
        "total_bets": len(rows),
        "open_bets": len(open_rows),
        "settled_bets": len(settled_rows),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0.0,
        "total_wagered": total_wagered,
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "total_pnl": total_pnl,
        "roi": total_pnl / total_wagered if total_wagered > 0 else 0.0,
        "open_invested": open_invested,
        "positions_value": positions_value,
    })
    if math.isfinite(profile_total_pnl):
        base_summary["profile_total_pnl"] = profile_total_pnl
    scoped["summary"] = base_summary
    scoped["bets"] = rows
    return scoped


@app.route("/api/open-bets-enriched")
def api_open_bets_enriched():
    """Open bets enriched with live positions and operator reasoning."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    return _json_no_store(_cached("open-bets-enriched", 30, _compute_open_bets_enriched))


@app.route("/api/profile-bets")
def api_profile_bets():
    """Canonical profile-backed history used by the dashboard and bet-history page."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    sport = _normalize_sport_filter(request.args.get("sport", "all"))
    payload = _cached("profile-bets", PROFILE_BETS_CACHE_TTL, _compute_profile_bets_snapshot)
    return _json_no_store(_scope_profile_bets_payload(payload, sport))


@app.route("/api/pnl-history")
def api_pnl_history():
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    history = _load_pnl_history()
    # Deduplicate by keeping one entry per unique timestamp (minute-level)
    seen = set()
    deduped = []
    for h in history:
        ts = h.get("timestamp", "")[:16]  # YYYY-MM-DDTHH:MM
        if ts not in seen:
            seen.add(ts)
            deduped.append(h)
    return jsonify(deduped[-200:])  # Last 200 points


@app.route("/api/refresh-prices", methods=["POST"])
def api_refresh_prices():
    """Fetch latest prices from Polymarket for open bets."""
    if not _clob_client:
        return jsonify({"status": "offline", "updated": 0})

    from src.strategy.duo_trader import get_all_trader_ledgers
    updated = 0
    skipped = 0
    for _, path in get_all_trader_ledgers():
        if Path(path).exists():
            ledger = BetLedger(path=path)
            for bet in ledger.get_open_bets(fresh=True):
                if bet.get("token_id"):
                    try:
                        price_data = _clob_client.get_price(bet["token_id"])
                        mid_price = price_data.get("mid")
                        if mid_price is None:
                            skipped += 1
                            logger.warning(
                                "Skipping dashboard price refresh for bet #%s (%s) because the orderbook is incomplete",
                                bet["id"],
                                bet["token_id"],
                            )
                            continue
                        result = ledger.update_current_price(bet["id"], mid_price)
                        if result.ok:
                            updated += 1
                        elif result.status != "not_found":
                            logger.info(
                                "Skipped dashboard price refresh for bet #%s because it is no longer open",
                                bet["id"],
                            )
                    except Exception as e:
                        skipped += 1
                        logger.warning(
                            "Dashboard price refresh failed for bet #%s (%s): %s",
                            bet.get("id"),
                            bet.get("token_id"),
                            e,
                        )

    return jsonify({"status": "ok", "updated": updated, "skipped": skipped})


@app.route("/api/positions")
def api_positions():
    """Fetch live positions directly from Polymarket's Data API."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    sport = _normalize_sport_filter(request.args.get("sport", "all"))
    raw_pnl, live_source = _load_live_pnl_snapshot()
    pnl = _dashboard_live_pnl_from_raw(raw_pnl, sport=sport)
    pnl["_pnl_degraded"] = live_source != "live"
    pnl["_pnl_source"] = live_source
    # Tag each position with its sport
    for pos in pnl.get("positions", []):
        pos["sport"] = _classify_sport_from_position(pos)
    return jsonify(pnl)


@app.route("/api/trade-history")
def api_trade_history():
    """Fetch trade history directly from Polymarket's activity API."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    global _position_monitor
    with _monitor_lock:
        if not _position_monitor:
            _position_monitor = PositionMonitor(clob_client=_clob_client)
        monitor = _position_monitor

    raw_limit = request.args.get("limit")
    limit: int | None
    if raw_limit is None:
        limit = 100
    elif raw_limit in ("", "0"):
        limit = None
    else:
        try:
            limit = max(int(raw_limit), 0) or None
        except ValueError:
            limit = 100
    try:
        trades = monitor.get_trades(limit=limit, page_size=500, strict=limit is None)
    except PositionDataPartialError as e:
        logger.warning("Failed to load trade history: %s", e)
        return jsonify({"error": "trade history unavailable"}), 503
    return jsonify(trades)


@app.route("/api/closed-positions")
def api_closed_positions():
    """Fetch settled/closed positions from Polymarket Data API."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    global _position_monitor
    with _monitor_lock:
        if not _position_monitor:
            _position_monitor = PositionMonitor(clob_client=_clob_client)
        monitor = _position_monitor

    # limit=0 (or missing) returns every closed position via pagination so the
    # UI doesn't silently truncate once the account passes 500 closed markets.
    raw_limit = request.args.get("limit")
    limit: int | None
    if raw_limit in (None, "", "0"):
        limit = None
    else:
        try:
            limit = max(int(raw_limit), 0) or None
        except ValueError:
            limit = None
    try:
        closed = monitor.get_closed_positions(limit=limit, strict=True)
    except PositionDataPartialError as e:
        logger.warning("Failed to load closed positions: %s", e)
        return jsonify({"error": "closed positions unavailable"}), 503
    return jsonify(closed)


@app.route("/api/settle-auto", methods=["POST"])
def api_settle_auto():
    """Auto-settle resolved markets across all trader ledgers."""
    auth_error = _require_mutation_auth()
    if auth_error is not None:
        return auth_error
    from src.strategy.duo_trader import get_all_trader_ledgers
    count = 0
    for _, path in get_all_trader_ledgers():
        if Path(path).exists():
            ledger = BetLedger(path=path)
            count += auto_settle_from_polymarket(ledger)
    return jsonify({"settled": count})


@app.route("/api/redeem-auto", methods=["POST"])
def api_redeem_auto():
    """Redeem any resolved Polymarket positions that are ready to claim."""
    auth_error = _require_mutation_auth()
    if auth_error is not None:
        return auth_error
    summary = auto_redeem_positions_from_polymarket(
        clob_client=_clob_client,
        wait=True,
    )
    return jsonify(summary)


@app.route("/api/settle/<int:bet_id>/<result>", methods=["POST"])
def api_settle_manual(bet_id: int, result: str):
    """Manually settle a bet across trader ledgers."""
    auth_error = _require_mutation_auth()
    if auth_error is not None:
        return auth_error
    result_lower = result.lower()
    valid_results = {"win", "won", "w", "loss", "lost", "l"}
    if result_lower not in valid_results:
        return jsonify({
            "ok": False,
            "error": f"Invalid result '{result}'. Must be one of: {', '.join(sorted(valid_results))}",
        }), 400
    won = result_lower in ("win", "won", "w")

    # bet_id is the renumbered merged ID — resolve to original trader ledger ID
    target = resolve_merged_bet_reference(bet_id, require_open=True)
    if not target:
        existing = resolve_merged_bet_reference(bet_id, require_open=False)
        if existing:
            return jsonify({"ok": False, "error": f"Bet #{bet_id} is not open"}), 409
        return jsonify({"ok": False, "error": f"Bet #{bet_id} not found"}), 404

    ledger = BetLedger(path=target["ledger_path"])
    mutation = ledger.settle_bet(target["original_id"], won)
    if not mutation.ok:
        if mutation.status == "not_open":
            return jsonify({"ok": False, "error": f"Bet #{bet_id} is not open"}), 409
        return jsonify({"ok": False, "error": f"Bet #{bet_id} not found"}), 404
    return jsonify({"ok": True, "bet_id": bet_id, "result": "won" if won else "lost"})


@app.route("/api/balance")
def api_balance():
    """Return wallet USDC balance and portfolio value (cached 60s)."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    payload, source = _cached_stale_while_revalidate(
        f"balance:{id(_clob_client)}",
        BALANCE_CACHE_TTL,
        _compute_balance,
    )
    payload = dict(payload or {})
    payload["_source"] = source
    return jsonify(payload)


def _compute_balance():
    balance = 0.0
    portfolio_value = 0.0

    if _clob_client:
        try:
            balance = _clob_client.get_cash_balance()
        except Exception as e:
            logger.warning("Failed to fetch cash balance: %s", e)
        try:
            portfolio_value = _clob_client.get_portfolio_value()
        except Exception as e:
            logger.warning("Failed to fetch portfolio value: %s", e)

    return {
        "cash_balance": balance,
        "portfolio_value": portfolio_value,
        "total_equity": balance + portfolio_value,
    }


def _compute_geoblock_status() -> dict:
    """Return the live geoblock verdict using the same shared CLOB transport."""
    import src.polymarket.client as polymarket_client_module
    from src.polymarket.client import ClobClientWrapper

    proxy_url = os.environ.get("CLOB_PROXY_URL", "")
    proxy_target = proxy_url.rsplit("@", 1)[-1] if proxy_url else ""

    clob = _clob_client
    if clob is None:
        clob = _call_with_timeout(
            "initializing CLOB geoblock client",
            ClobClientWrapper,
            GEOBLOCK_STATUS_TIMEOUT_SECONDS,
        )
        if clob is None:
            return {
                "available": False,
                "blocked": None,
                "ip": "",
                "country": "",
                "region": "",
                "status_code": None,
                "error": "Timed out initializing CLOB geoblock client",
                "proxy_configured": bool(proxy_url),
                "proxy_enabled": bool(polymarket_client_module._proxy_patched),
                "proxy_target": proxy_target,
            }

    status = _call_with_timeout(
        "checking Polymarket geoblock status",
        clob.get_geoblock_status,
        GEOBLOCK_STATUS_TIMEOUT_SECONDS,
    )
    if status is None:
        return {
            "available": False,
            "blocked": None,
            "ip": "",
            "country": "",
            "region": "",
            "status_code": None,
            "error": "Timed out checking Polymarket geoblock status",
            "proxy_configured": bool(proxy_url),
            "proxy_enabled": bool(polymarket_client_module._proxy_patched),
            "proxy_target": proxy_target,
        }
    return {
        "available": True,
        "blocked": status.get("blocked"),
        "ip": status.get("ip", ""),
        "country": status.get("country", ""),
        "region": status.get("region", ""),
        "status_code": status.get("status_code"),
        "error": status.get("error", ""),
        "proxy_configured": bool(proxy_url),
        "proxy_enabled": bool(polymarket_client_module._proxy_patched),
        "proxy_target": proxy_target,
    }


@app.route("/api/geoblock-status")
def api_geoblock_status():
    """Return Polymarket's live geoblock decision for this process."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    proxy_url = os.environ.get("CLOB_PROXY_URL", "")
    proxy_cache_key = _cache_key_secret_fragment(proxy_url)
    payload, source = _cached_stale_while_revalidate(
        f"geoblock-status:{id(_clob_client)}:{proxy_cache_key}",
        GEOBLOCK_STATUS_CACHE_TTL,
        _compute_geoblock_status,
    )
    payload = dict(payload or {})
    payload["_source"] = source
    return _json_no_store(payload)


@app.route("/api/bot-activity")
def api_bot_activity():
    """Return recent bot activity from bot.log."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    try:
        limit = min(max(1, int(request.args.get("limit", 500))), 10_000)
    except (ValueError, TypeError):
        limit = 500
    sport = _normalize_sport_filter(request.args.get("sport", "all"))
    log_path = LOGS_DIR / "bot.log"
    entries = _read_activity_entries(log_path, limit=limit, sport=sport)
    return _json_no_store(entries, extra_headers=_bot_activity_headers(log_path, entries))


@app.route("/api/bot-activity-snapshot")
def api_bot_activity_snapshot():
    """Return a single activity snapshot with metadata and entries together."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    try:
        limit = min(max(1, int(request.args.get("limit", 500))), 10_000)
    except (ValueError, TypeError):
        limit = 500
    sport = _normalize_sport_filter(request.args.get("sport", "all"))
    log_path = LOGS_DIR / "bot.log"
    entries = _read_activity_entries(log_path, limit=limit, sport=sport)
    snapshot = _bot_activity_snapshot(log_path, entries)
    return _json_no_store(snapshot, extra_headers=_bot_activity_headers(log_path, entries))


@app.route("/api/bot-alerts")
def api_bot_alerts():
    """Return durable WARNING/ERROR/CRITICAL alerts for the retention window.

    These are mirrored to ``alerts.jsonl`` independently of ``bot.log``'s INFO
    volume, so the Activity page's pinned alerts panel keeps showing them for the
    full retention window instead of letting them scroll out of the recent-log
    feed (e.g. an error logged overnight is still visible in the morning).
    """
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error

    sport = _normalize_sport_filter(request.args.get("sport", "all"))
    retention_hours = ACTIVITY_ALERT_RETENTION_HOURS
    alerts_path = LOGS_DIR / ALERT_STORE_FILENAME

    entries: list[dict] = []
    for record in load_recent_alerts(alerts_path, retention_hours):
        normalized = _normalize_activity_entry(
            {
                "timestamp": record.get("timestamp", ""),
                "level": record.get("level", ""),
                "source": record.get("source", ""),
                "message": record.get("message", ""),
            }
        )
        # Known handled rejections (e.g. geoblocked orders) get downgraded to
        # INFO by _normalize_activity_entry — they are not real alerts.
        if str(normalized.get("level", "")).upper() not in ALERT_LEVELS:
            continue
        normalized["sport"] = _classify_activity_sport(normalized)
        entries.append(normalized)

    entries = _filter_entries_by_sport(entries, sport)
    entries.sort(key=_activity_timestamp_sort_key)

    # Opportunistic, throttled retention prune so the sidecar file stays bounded.
    maybe_prune_alert_store(alerts_path, retention_hours)

    error_count = sum(
        1 for entry in entries if str(entry.get("level", "")).upper() in {"ERROR", "CRITICAL"}
    )
    warning_count = sum(
        1 for entry in entries if str(entry.get("level", "")).upper() == "WARNING"
    )
    payload = {
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "retention_hours": retention_hours,
        "entry_count": len(entries),
        "warning_count": warning_count,
        "error_count": error_count,
        "entries": entries,
    }
    return _json_no_store(payload)


@app.route("/api/significant-actions")
def api_significant_actions():
    """Return filtered high-value bot actions from bot.log."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    sport = _normalize_sport_filter(request.args.get("sport", "all"))
    log_path = LOGS_DIR / "bot.log"
    # Patterns that indicate significant bot activity
    sig_patterns = [
        (re.compile(r"order placed|Market order placed|Limit order placed", re.I), "ORDER", "green"),
        (re.compile(r"Auto-settled \d+ bets|Settled bet", re.I), "SETTLED", "blue"),
        (re.compile(r"value bet|conviction bet", re.I), "VALUE", "green"),
        (re.compile(r"sharp move|steam move|line tracking.*sharp", re.I), "SHARP", "orange"),
        (re.compile(r"SKIPPING.*injury|SKIPPING.*cancel|injury.*block", re.I), "INJURY", "red"),
        (re.compile(r"Stop-loss triggered", re.I), "STOP", "red"),
        (re.compile(r"Total orders: [1-9]", re.I), "SUMMARY", "blue"),
        (re.compile(r"Duo trader run complete", re.I), "RUN", "purple"),
    ]
    entries = []
    if log_path.exists():
        try:
            tail_bytes = 65_536
            with open(log_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - tail_bytes))
                if size > tail_bytes:
                    f.readline()
                raw = f.read().decode("utf-8", errors="replace")

            # Build log entries, joining continuation lines (no timestamp) with previous
            log_entries = []
            for line in raw.splitlines():
                m = re.match(
                    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),?\d*\s*\[(\w+)]\s*([\w.]+):\s*(.*)",
                    line,
                )
                if m:
                    log_entries.append({
                        "timestamp": m.group(1),
                        "level": m.group(2),
                        "source": m.group(3),
                        "message": m.group(4),
                    })
                elif log_entries and line.strip():
                    # Continuation line — append to previous entry's message
                    log_entries[-1]["message"] += " " + line.strip()

            for entry in log_entries:
                msg = entry["message"]
                for pattern, tag, color in sig_patterns:
                    if pattern.search(msg):
                        activity_entry = {
                            "timestamp": entry["timestamp"],
                            "level": entry["level"],
                            "source": entry.get("source", ""),
                            "tag": tag,
                            "color": color,
                            "message": msg.strip(),
                        }
                        activity_entry["sport"] = _classify_activity_sport(activity_entry)
                        if _activity_entry_matches_sport(activity_entry, sport):
                            entries.append(activity_entry)
                        break
        except Exception:
            logger.warning("Failed to parse significant actions from %s", log_path, exc_info=True)

    entries.extend(_filter_entries_by_sport(_runtime_issue_significant_actions(), sport))
    entries.sort(key=_activity_timestamp_sort_key)
    return jsonify(entries[-30:])


def _snapshot_upcoming_events() -> list[dict]:
    from src.config import RAW_DATA_DIR

    snapshot_dir = RAW_DATA_DIR / "snapshots"
    if not snapshot_dir.exists():
        return []

    _maybe_prune_upcoming_event_snapshots(snapshot_dir)

    # Each snapshot is a per-event file: { event, event_date, timestamp, fights }
    # Group by event name and take the most recent snapshot per event.
    # Only include events whose date is today or in the future.
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    seen = {}
    for path in _recent_upcoming_snapshot_paths(snapshot_dir):
        try:
            data = json.loads(path.read_text())
            event_name = data.get("event", "")
            if not event_name or event_name in seen:
                continue
            fights = data.get("fights", [])
            event_date = data.get("event_date", "")

            parsed = _normalize_upcoming_event_datetime(
                _parse_upcoming_event_datetime(event_date)
            )
            if parsed is not None and parsed < cutoff:
                continue

            seen[event_name] = {
                "event": event_name,
                "date": event_date,
                "fight_count": len(fights),
                "source": "snapshot",
                "sport": "ufc",
            }
        except Exception as e:
            logger.warning("Failed to load upcoming event snapshot %s: %s", path, e)
            continue

    return list(seen.values())


def _recent_upcoming_snapshot_paths(snapshot_dir: Path) -> list[Path]:
    paths = []
    for path in snapshot_dir.glob("*.json"):
        try:
            paths.append((path.stat().st_mtime, path))
        except OSError:
            continue
    paths.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in paths[:UPCOMING_SNAPSHOT_SCAN_LIMIT]]


def _maybe_prune_upcoming_event_snapshots(snapshot_dir: Path) -> None:
    global _last_upcoming_snapshot_prune
    now = time.time()
    if now - _last_upcoming_snapshot_prune < UPCOMING_SNAPSHOT_PRUNE_INTERVAL_SECONDS:
        return
    _last_upcoming_snapshot_prune = now

    try:
        paths = []
        for path in snapshot_dir.glob("*.json"):
            try:
                paths.append((path.stat().st_mtime, path))
            except OSError:
                continue
        if len(paths) <= UPCOMING_SNAPSHOT_MAX_FILES:
            return

        paths.sort(key=lambda item: item[0], reverse=True)
        removed = 0
        for _, path in paths[UPCOMING_SNAPSHOT_MAX_FILES:]:
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
        if removed:
            logger.info("Pruned %d old upcoming-event snapshot files from %s", removed, snapshot_dir)
    except Exception as exc:
        logger.warning("Failed to prune upcoming-event snapshots: %s", exc)


def _prediction_cache_upcoming_events() -> list[dict]:
    cache_path = LOGS_DIR / "predictions_cache.json"
    if not cache_path.exists():
        return []

    try:
        data = json.loads(cache_path.read_text())
    except Exception as e:
        logger.warning("Failed to load prediction cache upcoming events from %s: %s", cache_path, e)
        return []

    today_utc = datetime.now(timezone.utc).date()
    grouped: dict[str, dict] = {}
    for prediction in data.get("predictions", []):
        fighter_a = str(prediction.get("fighter_a", "") or "").strip()
        fighter_b = str(prediction.get("fighter_b", "") or "").strip()
        event_time = _select_fight_relevance_datetime(prediction)
        if event_time is None or event_time.date() < today_utc:
            continue
        if not fighter_a or not fighter_b:
            continue

        raw_card_date = prediction.get("card_date")
        day_key = _coerce_fight_matrix_day(raw_card_date, allow_raw_prefix=False)
        if day_key:
            display_date = day_key
            compare_date = event_time
        else:
            raw_date = prediction.get("event_date") or prediction.get("commence_time") or ""
            parsed_date = _parse_upcoming_event_datetime(raw_date)
            if parsed_date is None:
                continue
            compare_date = _normalize_upcoming_event_datetime(parsed_date) or event_time
            display_date = raw_date
            day_key = parsed_date.date().isoformat()

        entry = grouped.setdefault(
            day_key,
            {
                "event": "Live UFC odds card",
                "date": display_date,
                "fight_pairs": set(),
                "source": "predictions_cache",
                "sport": "ufc",
            },
        )
        current = _normalize_upcoming_event_datetime(
            _parse_upcoming_event_datetime(entry.get("date"))
        )
        if current is None or (compare_date is not None and compare_date < current):
            entry["date"] = display_date

        pair_key = "|".join(sorted([fighter_a.casefold(), fighter_b.casefold()]))
        entry["fight_pairs"].add(pair_key)

    events = []
    for entry in grouped.values():
        fight_pairs = entry.pop("fight_pairs", set())
        events.append({**entry, "fight_count": len(fight_pairs)})
    return events


def _merge_upcoming_events(snapshot_events: list[dict], prediction_events: list[dict]) -> list[dict]:
    merged = [dict(event) for event in snapshot_events]
    remaining_predictions = {
        day_key: dict(event)
        for event in prediction_events
        for day_key in [_upcoming_event_day_key(event.get("date"))]
        if day_key is not None
    }

    for event in merged:
        day_key = _upcoming_event_day_key(event.get("date"))
        if day_key is None:
            continue

        predicted = remaining_predictions.pop(day_key, None)
        if predicted is None:
            continue

        event["fight_count"] = max(
            int(event.get("fight_count") or 0),
            int(predicted.get("fight_count") or 0),
        )
        event["source"] = "snapshot+predictions"

    merged.extend(remaining_predictions.values())

    def _sort_key(event: dict):
        parsed = _normalize_upcoming_event_datetime(
            _parse_upcoming_event_datetime(event.get("date"))
        )
        if parsed is not None:
            return (0, parsed.isoformat(), str(event.get("event", "")))
        return (1, str(event.get("date", "")), str(event.get("event", "")))

    merged.sort(key=_sort_key)
    return merged[:10]


def _compute_upcoming_events_payload(sport: str) -> list[dict]:
    snapshot_events = _snapshot_upcoming_events()
    prediction_events = _prediction_cache_upcoming_events()
    merged = _merge_upcoming_events(snapshot_events, prediction_events)
    if sport != "all":
        merged = [
            event for event in merged
            if str(event.get("sport", "") or "").lower() == sport
        ]
    return merged


@app.route("/api/upcoming-events")
def api_upcoming_events():
    """Return upcoming UFC events from monitoring snapshots plus live predictions."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    sport = _normalize_sport_filter(request.args.get("sport", "all"))
    from src.config import RAW_DATA_DIR
    cache_key = f"upcoming-events:{sport}:{RAW_DATA_DIR}:{LOGS_DIR}"
    return jsonify(_cached(
        cache_key,
        UPCOMING_EVENTS_CACHE_TTL,
        lambda: _compute_upcoming_events_payload(sport),
    ))


@app.route("/api/predictions")
def api_predictions():
    """Return cached model predictions for the Model vs Market heatmap."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    try:
        return jsonify(_load_prediction_payload(include_global_feature_importance=False))
    except Exception as e:
        logger.error(f"Failed to load predictions cache: {e}")
        return jsonify(_empty_prediction_payload(
            include_global_feature_importance=False,
            cache_status="error",
        ))


@app.route("/api/trader-race")
def api_trader_race():
    """Return cumulative P&L timeline per trader for the race chart."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error

    result = {}
    for label, _, path, _ in _trader_breakdown_specs():
        ledger = BetLedger(path=path)
        # Build cumulative P&L from settled bets ordered by settled_at
        settled = [b for b in ledger.bets if b.get("settled_at") and b.get("result_pnl") is not None]
        settled.sort(key=lambda b: b["settled_at"])

        cum_pnl = 0.0
        points = [{"time": "", "pnl": 0.0}]  # Start at zero
        for b in settled:
            cum_pnl += b["result_pnl"]
            points.append({
                "time": b["settled_at"][:16],  # YYYY-MM-DDTHH:MM
                "pnl": round(cum_pnl, 2),
            })
        result[label] = points

    return jsonify(result)


@app.route("/api/injury-alerts")
def api_injury_alerts():
    """Return precomputed market alerts without scanning line history."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    return _json_no_store(_load_market_intel_artifact())


def _market_intel_artifact_path() -> Path:
    return LOGS_DIR / MARKET_INTEL_FILENAME


def _empty_market_intel_payload(status: str, message: str) -> dict:
    return {
        "status": status,
        "updated_at": None,
        "age_seconds": None,
        "stale_after_seconds": MARKET_INTEL_STALE_AFTER_SECONDS,
        "message": message,
        "alerts": [],
        "injury_alerts": [],
        "line_movements": [],
    }


def _market_intel_payload_from_line_summary(line_summary: dict | None) -> dict:
    summary = dict(line_summary or {})
    alerts = list(summary.get("injury_alerts") or [])
    line_movements = list(summary.get("line_movements") or [])
    updated_at = summary.get("timestamp") or _utcnow_iso()
    payload = {
        "status": "current",
        "updated_at": updated_at,
        "age_seconds": 0,
        "stale_after_seconds": MARKET_INTEL_STALE_AFTER_SECONDS,
        "message": "",
        "alerts": alerts,
        "injury_alerts": alerts,
        "line_movements": line_movements,
        "fights_analyzed": int(summary.get("fights_analyzed") or 0),
        "sharp_moves": int(summary.get("sharp_moves") or 0),
        "steam_moves": int(summary.get("steam_moves") or 0),
        "coverage": summary.get("coverage") or {},
    }
    return payload


def write_market_intel_artifact(line_summary: dict | None) -> dict:
    """Persist the latest market-intel scan for cheap dashboard reads."""
    payload = _market_intel_payload_from_line_summary(line_summary)
    path = _market_intel_artifact_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp_path.replace(path)
    except Exception as exc:
        logger.warning("Failed to write market intel artifact %s: %s", path, exc)
    return payload


def _load_market_intel_artifact() -> dict:
    path = _market_intel_artifact_path()
    if not path.exists():
        return _empty_market_intel_payload(
            "missing",
            "Market intel scan has not completed yet.",
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read market intel artifact %s: %s", path, exc)
        return _empty_market_intel_payload(
            "error",
            "Market intel artifact could not be read.",
        )
    if not isinstance(payload, dict):
        logger.warning("Market intel artifact %s had unexpected shape: %s", path, type(payload).__name__)
        return _empty_market_intel_payload(
            "error",
            "Market intel artifact had an unexpected shape.",
        )

    try:
        age_seconds = max(time.time() - path.stat().st_mtime, 0.0)
    except OSError:
        age_seconds = None
    status = str(payload.get("status") or "current")
    if age_seconds is not None and age_seconds > MARKET_INTEL_STALE_AFTER_SECONDS:
        status = "stale"

    alerts = list(payload.get("alerts") or payload.get("injury_alerts") or [])
    payload["status"] = status
    payload["age_seconds"] = age_seconds
    payload["stale_after_seconds"] = MARKET_INTEL_STALE_AFTER_SECONDS
    payload["alerts"] = alerts
    payload["injury_alerts"] = alerts
    if status == "stale" and not payload.get("message"):
        payload["message"] = "Market intel scan is stale."
    elif status == "current":
        payload.setdefault("message", "")
    return payload


def _load_latest_line_snapshot_fights():
    from src.config import RAW_DATA_DIR

    line_dir = RAW_DATA_DIR / "line_history"
    if not line_dir.exists():
        return None

    snapshots = sorted(line_dir.glob("odds_*.csv"), reverse=True)
    if not snapshots:
        return None

    import pandas as pd

    latest = pd.read_csv(snapshots[0])
    agg_spec = {
        "a_prob": ("a_fair_prob", "mean"),
        "b_prob": ("b_fair_prob", "mean"),
    }
    if "commence_time" in latest.columns:
        agg_spec["commence_time"] = ("commence_time", "first")

    return latest.groupby(["fighter_a", "fighter_b"]).agg(
        **agg_spec,
    ).reset_index()


def _compute_market_intel_bundle():
    try:
        from src.data.line_tracker import analyze_line_movement, detect_injury_or_cancellation

        fights = _load_latest_line_snapshot_fights()
        if fights is None:
            return {"injury_alerts": [], "line_movements": []}

        alerts = []
        results = []
        for _, fight in fights.iterrows():
            fighter_a = fight["fighter_a"]
            fighter_b = fight["fighter_b"]
            analysis = analyze_line_movement(fighter_a, fighter_b)
            analysis["fighter_a"] = fighter_a
            analysis["fighter_b"] = fighter_b
            results.append(analysis)

            alert = detect_injury_or_cancellation(
                fighter_a,
                fighter_b,
                current_odds={"a_prob": fight["a_prob"], "b_prob": fight["b_prob"]},
                analysis=analysis,
                commence_time=fight.get("commence_time"),
            )
            if alert["suspected"]:
                alerts.append({
                    "fighter_a": fighter_a,
                    "fighter_b": fighter_b,
                    "severity": alert["severity"],
                    "reason": alert["reason"],
                    "movement": alert.get("details", {}).get("movement"),
                    "steam_move": alert.get("details", {}).get("steam_move", False),
                })

        results.sort(key=lambda x: x.get("abs_movement", 0), reverse=True)
        return {"injury_alerts": alerts, "line_movements": results}
    except Exception as e:
        logger.error("Failed to build market intel bundle: %s", e)
        return {"injury_alerts": [], "line_movements": []}


def _market_intel_snapshot() -> dict:
    return _cached("market-intel", SLOW_ENDPOINT_TTL, _compute_market_intel_bundle)


def _compute_injury_alerts():
    return _compute_market_intel_bundle()["injury_alerts"]


@app.route("/api/filter-funnel")
def api_filter_funnel():
    """Run cached predictions through the filter pipeline and report funnel stats."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    cache_path = LOGS_DIR / "predictions_cache.json"
    if not cache_path.exists():
        return jsonify({"total": 0, "funnel": [], "fights": []})

    try:
        data = json.loads(cache_path.read_text())
        preds = data.get("predictions", [])
        if not preds:
            return jsonify({"total": 0, "funnel": [], "fights": []})

        from src.strategy.value import compute_independent_blend_probs, scaled_min_edge
        from src.config import (
            MIN_MODEL_PROB, MAX_DECIMAL_ODDS, MIN_EDGE_THRESHOLD,
            MIN_FIGHTER_FIGHTS, REQUIRE_MODEL_AGREEMENT,
            MODEL_AGREEMENT_MIN_EDGE, LINE_MOVEMENT_FILTER,
            LINE_AGAINST_EXTRA_EDGE, LINE_SHARP_BLOCK,
        )
        import math

        filter_names = [
            "Total Fights",
            "Experience",
            "Min Probability",
            "Max Odds",
            "Scaled Edge",
            "Model Agreement",
            "Line Movement",
            "Value Bets",
        ]
        counts = [len(preds), 0, 0, 0, 0, 0, 0, 0]
        fight_details = []

        for p in preds:
            model_a = p.get("prob_a", 0.5)
            model_b = p.get("prob_b", 0.5)
            market_a = p.get("a_market_prob", 0.5)
            market_b = p.get("b_market_prob", 0.5)
            a_fights = p.get("a_num_fights")
            b_fights = p.get("b_num_fights")
            no_odds_a = p.get("no_odds_prob_a")
            no_odds_b = p.get("no_odds_prob_b")
            line_mov = p.get("line_movement")
            line_sharp = p.get("line_is_sharp")
            line_steam = p.get("line_steam_move")

            blend_a, blend_b = compute_independent_blend_probs(
                model_a,
                market_a,
                no_odds_a,
                model_b,
                market_b,
                no_odds_b,
            )
            edge_a = blend_a - market_a
            edge_b = blend_b - market_b

            # Pick best side
            if edge_a >= edge_b:
                blend, market, edge, side = blend_a, market_a, edge_a, "a"
                no_odds = no_odds_a
            else:
                blend, market, edge, side = blend_b, market_b, edge_b, "b"
                no_odds = no_odds_b

            decimal_odds = 1.0 / market if market > 0 else 99.0
            fight_name = f"{p.get('fighter_a', '?')} vs {p.get('fighter_b', '?')}"
            stopped_at = None

            # Filter 0: Experience
            if a_fights is not None and a_fights < MIN_FIGHTER_FIGHTS:
                stopped_at = "Experience"
            elif b_fights is not None and b_fights < MIN_FIGHTER_FIGHTS:
                stopped_at = "Experience"

            if not stopped_at:
                counts[1] += 1  # passed experience
                # Filter 1: Min prob
                if blend < MIN_MODEL_PROB:
                    stopped_at = "Min Probability"

            if not stopped_at:
                counts[2] += 1  # passed min prob
                # Filter 2: Max odds
                if decimal_odds > MAX_DECIMAL_ODDS:
                    stopped_at = "Max Odds"

            if not stopped_at:
                counts[3] += 1  # passed max odds
                # Filter 3: Scaled edge
                required = scaled_min_edge(decimal_odds)
                if edge < required:
                    stopped_at = "Scaled Edge"

            if not stopped_at:
                counts[4] += 1  # passed scaled edge
                # Filter 4: Model agreement
                if REQUIRE_MODEL_AGREEMENT and no_odds is not None:
                    no_odds_edge = no_odds - market
                    if no_odds_edge < MODEL_AGREEMENT_MIN_EDGE:
                        stopped_at = "Model Agreement"

            if not stopped_at:
                counts[5] += 1  # passed agreement
                # Filter 5: Line movement
                if LINE_MOVEMENT_FILTER and line_mov is not None:
                    if not isinstance(line_mov, (int, float)) or math.isnan(line_mov):
                        line_mov = None
                if LINE_MOVEMENT_FILTER and line_mov is not None:
                    line_against = (side == "a" and line_mov < -0.02) or \
                                   (side == "b" and line_mov > 0.02)
                    if line_against:
                        required_edge = scaled_min_edge(decimal_odds) + LINE_AGAINST_EXTRA_EDGE
                        if edge < required_edge:
                            stopped_at = "Line Movement"
                        elif LINE_SHARP_BLOCK:
                            if (line_sharp == 1 or line_steam == 1):
                                stopped_at = "Line Movement"

            if not stopped_at:
                counts[6] += 1  # passed line movement
                # Also needs minimum edge
                if edge >= MIN_EDGE_THRESHOLD:
                    counts[7] += 1  # value bet!
                    stopped_at = "PASSED"
                else:
                    stopped_at = "Min Edge"

            fight_details.append({
                "fight": fight_name,
                "edge": round(edge, 4),
                "stopped_at": stopped_at,
            })

        funnel = [{"name": n, "count": c} for n, c in zip(filter_names, counts)]
        return jsonify({"total": len(preds), "funnel": funnel, "fights": fight_details})
    except Exception as e:
        logger.error(f"Failed to compute filter funnel: {e}")
        return jsonify({"total": 0, "funnel": [], "fights": []})



@app.route("/api/trader-breakdown")
def api_trader_breakdown():
    """Return per-trader P&L breakdown from individual ledgers."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error

    breakdown = []
    for label, style, path, blend in _trader_breakdown_specs():
        ledger = BetLedger(path=path)
        summary = ledger.get_summary()
        breakdown.append({
            "trader": label,
            "style": style,
            "blend_weight": blend,
            "total_bets": summary["total_bets"],
            "open_bets": summary["open_bets"],
            "wins": summary["wins"],
            "losses": summary["losses"],
            "win_rate": summary["win_rate"],
            "realized_pnl": summary["realized_pnl"],
            "unrealized_pnl": summary["unrealized_pnl"],
            "total_pnl": summary["total_pnl"],
            "total_wagered": summary["total_wagered"],
            "roi": summary["roi"],
        })

    return jsonify(breakdown)


@app.route("/api/tracker-decisions")
def api_tracker_decisions():
    """Return a per-fight decision matrix for S/C/M/G."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error

    show_history = request.args.get("history", "") == "1"
    cache_key = _tracker_decisions_cache_key(show_history)
    cached = _fresh_cached_data(cache_key, TRACKER_DECISIONS_CACHE_TTL)
    if cached is not None:
        return _json_no_store(cached)

    try:
        from src.strategy.llm_operator import load_decision_log, load_tracker_decision_log

        operator_decisions = load_decision_log()
        operator_decisions.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
        tracker_records = load_tracker_decision_log()
        ledger_view = load_all_trader_ledgers()

        prediction_rows = _prediction_matrix_rows()
        prediction_index = _build_prediction_row_index(prediction_rows)
        market_event_date_hints = _build_market_event_date_hints(
            prediction_rows,
            tracker_records,
            operator_decisions,
            ledger_view.bets,
        )
        card_date_hints = _build_card_date_hints(
            prediction_rows,
            tracker_records,
            operator_decisions,
            ledger_view.bets,
        )
        seen = {
            _fight_matrix_key(
                row.get("fighter_a", ""),
                row.get("fighter_b", ""),
                row.get("market_event_date") or row.get("event_date") or "",
                card_date=_row_card_date(row),
            )
            for row in prediction_rows
        }

        def _append_fight_row(
            fighter_a: str,
            fighter_b: str,
            event_date: str,
            *,
            market_event_date: str = "",
            **extra,
        ) -> None:
            if not fighter_a or not fighter_b:
                return
            resolved_market_event_date = _resolve_market_event_date_hint(
                fighter_a=fighter_a,
                fighter_b=fighter_b,
                event_date=event_date,
                market_event_date=market_event_date,
                hints=market_event_date_hints,
            )
            card_date = _resolve_card_date_hint(
                fighter_a=fighter_a,
                fighter_b=fighter_b,
                event_date=event_date,
                market_event_date=resolved_market_event_date or market_event_date,
                card_date=str(extra.get("card_date", "") or ""),
                hints=card_date_hints,
            )
            key_date = resolved_market_event_date or event_date
            key = _fight_matrix_key(fighter_a, fighter_b, key_date, card_date=card_date)
            if key in seen:
                return
            seen.add(key)
            prediction_rows.append(
                {
                    "fighter_a": fighter_a,
                    "fighter_b": fighter_b,
                    "event_date": event_date,
                    "market_event_date": resolved_market_event_date or event_date,
                    "card_date": card_date,
                    "event_group_date": _fight_matrix_event_group_date(
                        resolved_market_event_date or event_date,
                        card_date=card_date,
                    ),
                    "event_title": extra.get("event_title", ""),
                    "weight_class": extra.get("weight_class", ""),
                }
            )

        for decision in operator_decisions:
            _append_fight_row(
                str(decision.get("fighter_a", "") or ""),
                str(decision.get("fighter_b", "") or ""),
                str(decision.get("event_date", "") or ""),
                market_event_date=str(decision.get("market_event_date", "") or ""),
                card_date=_row_card_date(decision),
                event_title=str(decision.get("event_title", "") or ""),
            )
        for bet in ledger_view.bets:
            _append_fight_row(
                str(bet.get("fighter") or bet.get("fighter_a") or ""),
                str(bet.get("opponent") or bet.get("fighter_b") or ""),
                str(bet.get("event_date") or bet.get("market_event_date") or ""),
                market_event_date=str(bet.get("market_event_date") or ""),
                card_date=_row_card_date(bet),
            )

        decisions_index = _build_decisions_index(
            operator_decisions,
            market_event_date_hints=market_event_date_hints,
            card_date_hints=card_date_hints,
        )
        pass_index = _build_operator_pass_index(
            operator_decisions,
            market_event_date_hints=market_event_date_hints,
            card_date_hints=card_date_hints,
        )
        block_index = _build_operator_block_index(
            operator_decisions,
            market_event_date_hints=market_event_date_hints,
            card_date_hints=card_date_hints,
        )
        tracker_index = _build_tracker_decision_index(
            tracker_records,
            card_date_hints=card_date_hints,
        )
        ledger_index = _build_trader_bet_index(
            ledger_view.bets,
            market_event_date_hints=market_event_date_hints,
            card_date_hints=card_date_hints,
        )

        fights = []
        for row in prediction_rows:
            fighter_a = str(row.get("fighter_a", "") or "")
            fighter_b = str(row.get("fighter_b", "") or "")
            event_date = str(row.get("market_event_date") or row.get("event_date") or "")
            card_date = _row_card_date(row)
            key = _fight_matrix_key(fighter_a, fighter_b, event_date, card_date=card_date)
            prediction_row = prediction_index.get(key)

            fights.append(
                {
                    "fighter_a": fighter_a,
                    "fighter_b": fighter_b,
                    "event_date": str(row.get("event_date", "") or ""),
                    "commence_time": str(row.get("commence_time") or ""),
                    "market_event_date": str(row.get("market_event_date") or row.get("event_date") or ""),
                    "card_date": _coerce_fight_matrix_day(card_date, allow_raw_prefix=False),
                    "event_group_date": str(row.get("event_group_date") or ""),
                    "event_title": str(row.get("event_title", "") or ""),
                    "weight_class": str(row.get("weight_class", "") or ""),
                    "S": _format_sc_matrix_cell(
                        trader="S",
                        ledger_bet=ledger_index.get(("S", *key)),
                        operator_decision=_lookup_operator_matrix_decision(pass_index, key, "S"),
                        block_decision=_lookup_operator_matrix_decision(block_index, key, "S"),
                        decisions_index=decisions_index,
                        prediction_row=prediction_row,
                    ),
                    "C": _format_sc_matrix_cell(
                        trader="C",
                        ledger_bet=ledger_index.get(("C", *key)),
                        operator_decision=_lookup_operator_matrix_decision(pass_index, key, "C"),
                        block_decision=_lookup_operator_matrix_decision(block_index, key, "C"),
                        decisions_index=decisions_index,
                        prediction_row=prediction_row,
                    ),
                    "M": _format_tracker_matrix_cell(
                        tracker_index.get(("M", *key)),
                        fallback_text="Pending model tracker",
                        ledger_bet=ledger_index.get(("M", *key)),
                        prediction_row=prediction_row,
                    ),
                    "G": _format_tracker_matrix_cell(
                        tracker_index.get(("G", *key)),
                        fallback_text="Pending Gemini tracker",
                        ledger_bet=ledger_index.get(("G", *key)),
                        prediction_row=prediction_row,
                    ),
                }
            )

        if not show_history:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
            fights = [f for f in fights if _fight_is_relevant(f, cutoff)]

        def _sort_key(fight):
            parsed = _parse_upcoming_event_datetime(
                fight.get("market_event_date") or fight.get("event_date")
            )
            return (
                parsed.isoformat() if parsed else "9999-12-31T23:59:59",
                fight.get("fighter_a", ""),
                fight.get("fighter_b", ""),
            )

        fights.sort(key=_sort_key)
        payload = {"fights": fights, "count": len(fights), "showing_history": show_history}
        _store_cache_data(cache_key, payload)
        return _json_no_store(payload)
    except Exception as e:
        logger.error("Failed to build tracker decision matrix: %s", e)
        return _json_no_store({"fights": [], "count": 0, "error": str(e)})


_EXECUTION_ALREADY_BET_GATES = {
    "duplicate_open_position",
    "duplicate_open_limit_order",
    "duplicate_open_clob_order",
}

_EXECUTION_PATH_LABELS = {
    "S": "Single Trader",
    "C": "Conviction Trader",
    "M": "Model Tracker",
    "G": "Gemini Tracker",
}


def _execution_load_enrichment_context() -> dict:
    """Load ledger/operator/tracker records used to explain already-open bets."""
    context = {"ledger_bets": [], "operator_decisions": [], "tracker_records": []}
    try:
        ledger_view = load_all_trader_ledgers()
        context["ledger_bets"] = list(getattr(ledger_view, "bets", []) or [])
    except Exception as exc:
        logger.warning("Execution breakdown could not load trader ledgers: %s", exc)

    try:
        from src.strategy.llm_operator import load_decision_log, load_tracker_decision_log

        operator_decisions = load_decision_log()
        operator_decisions.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
        context["operator_decisions"] = operator_decisions

        tracker_records = load_tracker_decision_log()
        tracker_records.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
        context["tracker_records"] = tracker_records
    except Exception as exc:
        logger.warning("Execution breakdown could not load operator/tracker logs: %s", exc)
    return context


def _execution_id(value) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    try:
        numeric = float(raw)
    except ValueError:
        return raw
    if math.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return raw


def _execution_pair(row: dict | None) -> frozenset[str]:
    if not isinstance(row, dict):
        return frozenset()
    first = (
        row.get("fighter_a")
        or row.get("fighter")
        or row.get("bet_on")
        or row.get("pick")
        or ""
    )
    second = (
        row.get("fighter_b")
        or row.get("opponent")
        or row.get("opposite_side")
        or ""
    )
    names = {_normalize_name(first), _normalize_name(second)}
    names.discard("")
    return frozenset(names)


def _execution_pair_matches(left: dict | None, right: dict | None) -> bool:
    left_pair = _execution_pair(left)
    right_pair = _execution_pair(right)
    return len(left_pair) == 2 and left_pair == right_pair


def _execution_event_group_date(row: dict | None) -> str:
    if not isinstance(row, dict):
        return ""
    return _fight_matrix_event_group_date(
        row.get("market_event_date")
        or row.get("event_date")
        or row.get("commence_time")
        or "",
        card_date=_row_card_date(row),
    )


def _execution_event_matches(left: dict | None, right: dict | None) -> bool:
    left_date = _execution_event_group_date(left)
    right_date = _execution_event_group_date(right)
    return not left_date or not right_date or left_date == right_date


def _execution_trader_from_bet(bet: dict) -> str:
    raw = str(bet.get("trader") or "").strip().upper()
    if raw:
        return raw.split(":", 1)[0]
    ledger_path = str(bet.get("_ledger_path") or bet.get("ledger_path") or "")
    if ledger_path:
        return _trader_label_from_path(ledger_path)
    return ""


def _execution_ledger_ids(bet: dict) -> set[str]:
    return {
        value
        for value in (
            _execution_id(bet.get("id")),
            _execution_id(bet.get("_original_id")),
            _execution_id(bet.get("ledger_id")),
            _execution_id(bet.get("ledger_bet_id")),
        )
        if value
    }


def _execution_path_ledger_ids(path: dict) -> set[str]:
    numbers = path.get("numbers") if isinstance(path.get("numbers"), dict) else {}
    order = path.get("order") if isinstance(path.get("order"), dict) else {}
    return {
        value
        for value in (
            _execution_id(numbers.get("existing_ledger_id")),
            _execution_id(numbers.get("ledger_id")),
            _execution_id(order.get("ledger_id")),
            _execution_id(order.get("ledger_bet_id")),
        )
        if value
    }


def _execution_find_ledger_bet(
    fight: dict,
    trader: str,
    path: dict,
    ledger_bets: list[dict],
) -> dict | None:
    numbers = path.get("numbers") if isinstance(path.get("numbers"), dict) else {}
    order = path.get("order") if isinstance(path.get("order"), dict) else {}
    wanted_ids = _execution_path_ledger_ids(path)
    market_ids = {
        _execution_id(value)
        for value in (
            numbers.get("market_id"),
            numbers.get("existing_market_id"),
            order.get("market_id"),
        )
        if _execution_id(value)
    }
    token_ids = {
        str(value).strip()
        for value in (
            numbers.get("token_id"),
            numbers.get("existing_token_id"),
            order.get("token_id"),
        )
        if str(value or "").strip()
    }
    wanted_trader = str(trader or "").strip().upper()

    def _trader_matches(bet: dict) -> bool:
        bet_trader = _execution_trader_from_bet(bet)
        return not wanted_trader or not bet_trader or bet_trader == wanted_trader

    ordered = sorted(
        (bet for bet in ledger_bets if isinstance(bet, dict) and _trader_matches(bet)),
        key=lambda bet: str(bet.get("placed_at") or ""),
        reverse=True,
    )
    if wanted_ids:
        for bet in ordered:
            if wanted_ids & _execution_ledger_ids(bet):
                return bet

    for bet in ordered:
        bet_market_id = _execution_id(bet.get("market_id"))
        if market_ids and bet_market_id in market_ids:
            if (
                (_execution_pair_matches(fight, bet) and _execution_event_matches(fight, bet))
                or str(bet.get("token_id") or "").strip() in token_ids
            ):
                return bet

    for bet in ordered:
        if token_ids and str(bet.get("token_id") or "").strip() in token_ids:
            return bet

    for bet in ordered:
        if _execution_pair_matches(fight, bet) and _execution_event_matches(fight, bet):
            return bet
    return None


def _execution_find_operator_decision(
    fight: dict,
    trader: str,
    path: dict,
    ledger_bet: dict | None,
    operator_decisions: list[dict],
) -> dict | None:
    wanted_trader = str(trader or "").strip().upper()
    if wanted_trader not in {"S", "C"}:
        return None
    numbers = path.get("numbers") if isinstance(path.get("numbers"), dict) else {}
    pick = (
        (ledger_bet or {}).get("fighter")
        or (ledger_bet or {}).get("bet_on")
        or numbers.get("bet_on")
        or ""
    )
    norm_pick = _normalize_name(pick)
    reference = ledger_bet if isinstance(ledger_bet, dict) else fight

    for decision in operator_decisions:
        if not isinstance(decision, dict):
            continue
        aliases = _decision_context_aliases(decision.get("decision_context"))
        if aliases and wanted_trader not in aliases:
            continue
        if not _execution_event_matches(fight, decision):
            continue
        if not _execution_pair_matches(fight, decision) and not _execution_pair_matches(reference, decision):
            continue
        decision_pick = _normalize_name(
            decision.get("bet_on")
            or decision.get("pick")
            or decision.get("fighter")
            or ""
        )
        if norm_pick and decision_pick and norm_pick != decision_pick:
            continue
        return decision
    return None


def _execution_find_tracker_record(
    fight: dict,
    trader: str,
    path: dict,
    ledger_bet: dict | None,
    tracker_records: list[dict],
) -> dict | None:
    wanted_trader = str(trader or "").strip().upper()
    if wanted_trader not in {"M", "G"}:
        return None
    numbers = path.get("numbers") if isinstance(path.get("numbers"), dict) else {}
    pick = (
        (ledger_bet or {}).get("fighter")
        or (ledger_bet or {}).get("bet_on")
        or numbers.get("bet_on")
        or ""
    )
    norm_pick = _normalize_name(pick)
    reference = ledger_bet if isinstance(ledger_bet, dict) else fight

    latest_outcomes = {
        str(record.get("decision_id") or ""): record
        for record in tracker_records
        if isinstance(record, dict) and record.get("type") == "outcome" and record.get("decision_id")
    }
    for record in tracker_records:
        if not isinstance(record, dict) or record.get("type") == "outcome":
            continue
        if str(record.get("trader") or "").strip().upper() != wanted_trader:
            continue
        if not _execution_event_matches(fight, record):
            continue
        if not _execution_pair_matches(fight, record) and not _execution_pair_matches(reference, record):
            continue
        record_pick = _normalize_name(record.get("pick") or record.get("bet_on") or "")
        if norm_pick and record_pick and norm_pick != record_pick:
            continue
        merged = dict(record)
        outcome = latest_outcomes.get(str(record.get("decision_id") or ""))
        if outcome:
            merged["outcome"] = outcome
        return merged
    return None


def _execution_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _execution_format_money(value) -> str:
    parsed = _execution_float(value)
    return f"${parsed:.2f}" if parsed is not None else ""


def _execution_format_price(value) -> str:
    parsed = _execution_float(value)
    return f"{parsed:.4f}" if parsed is not None else ""


def _execution_stage_exists(path: dict, stage_name: str, gate: str) -> bool:
    stages = path.get("stages")
    if not isinstance(stages, list):
        return False
    return any(
        isinstance(stage, dict)
        and str(stage.get("stage") or "") == stage_name
        and str(stage.get("gate") or "") == gate
        for stage in stages
    )


def _execution_append_stage(path: dict, stage: dict) -> None:
    stages = path.setdefault("stages", [])
    if isinstance(stages, list) and not _execution_stage_exists(
        path,
        str(stage.get("stage") or ""),
        str(stage.get("gate") or ""),
    ):
        stages.append(stage)


def _execution_enrich_already_bet_path(
    fight: dict,
    trader: str,
    path: dict,
    enrichment: dict,
) -> None:
    ledger_bet = _execution_find_ledger_bet(
        fight,
        trader,
        path,
        enrichment.get("ledger_bets", []),
    )
    operator_decision = _execution_find_operator_decision(
        fight,
        trader,
        path,
        ledger_bet,
        enrichment.get("operator_decisions", []),
    )
    tracker_record = _execution_find_tracker_record(
        fight,
        trader,
        path,
        ledger_bet,
        enrichment.get("tracker_records", []),
    )

    numbers = path.setdefault("numbers", {})
    if not isinstance(numbers, dict):
        numbers = {}
        path["numbers"] = numbers
    order = path.setdefault("order", {})
    if not isinstance(order, dict):
        order = {}
        path["order"] = order

    if ledger_bet:
        ledger_id = ledger_bet.get("_original_id") or ledger_bet.get("ledger_id") or ledger_bet.get("id")
        display_ledger_id = ledger_bet.get("id")
        numbers.setdefault("ledger_id", ledger_id)
        numbers.setdefault("existing_ledger_id", ledger_id)
        numbers.setdefault("display_ledger_id", display_ledger_id)
        numbers.setdefault("bet_on", ledger_bet.get("fighter") or ledger_bet.get("bet_on"))
        numbers.setdefault("market_id", ledger_bet.get("market_id"))
        numbers.setdefault("token_id", ledger_bet.get("token_id"))
        numbers.setdefault("model_probability", _execution_float(ledger_bet.get("model_prob")))
        numbers.setdefault("market_probability", _execution_float(ledger_bet.get("market_prob")))
        numbers.setdefault("edge", _execution_float(ledger_bet.get("edge")))
        numbers.setdefault("amount", _execution_float(ledger_bet.get("amount")))
        numbers.setdefault("price", _execution_float(ledger_bet.get("price")))
        numbers.setdefault("shares", _execution_float(ledger_bet.get("shares")))
        numbers.setdefault("order_id", ledger_bet.get("order_id"))
        numbers.setdefault("order_type", ledger_bet.get("order_type"))
        numbers.setdefault("placement_state", ledger_bet.get("placement_state"))
        numbers.setdefault("placed_at", ledger_bet.get("placed_at"))
        if ledger_bet.get("reason"):
            path["original_reason"] = ledger_bet.get("reason")
        path["existing_bet"] = {
            "ledger_id": ledger_id,
            "display_ledger_id": display_ledger_id,
            "placed_at": ledger_bet.get("placed_at"),
            "status": ledger_bet.get("status"),
            "placement_state": ledger_bet.get("placement_state"),
        }
        order.update(
            {
                key: value
                for key, value in {
                    "ledger_id": ledger_id,
                    "display_ledger_id": display_ledger_id,
                    "order_id": ledger_bet.get("order_id"),
                    "order_type": ledger_bet.get("order_type"),
                    "amount": _execution_float(ledger_bet.get("amount")),
                    "price": _execution_float(ledger_bet.get("price")),
                    "shares": _execution_float(ledger_bet.get("shares")),
                    "token_id": ledger_bet.get("token_id"),
                    "dry_run": ledger_bet.get("dry_run"),
                    "placement_state": ledger_bet.get("placement_state"),
                    "status": ledger_bet.get("status"),
                    "error": ledger_bet.get("submission_error"),
                }.items()
                if value not in (None, "")
            }
        )

    if numbers.get("existing_reason") and not path.get("original_reason"):
        path["original_reason"] = numbers.get("existing_reason")
    existing_field_map = {
        "amount": "existing_amount",
        "price": "existing_price",
        "shares": "existing_shares",
        "token_id": "existing_token_id",
        "order_id": "existing_order_id",
        "order_type": "existing_order_type",
        "placement_state": "existing_placement_state",
    }
    for canonical, legacy in existing_field_map.items():
        if numbers.get(canonical) in (None, "") and numbers.get(legacy) not in (None, ""):
            numbers[canonical] = numbers.get(legacy)
    if not order:
        order.update(
            {
                key: value
                for key, value in {
                    "ledger_id": numbers.get("existing_ledger_id") or numbers.get("ledger_id"),
                    "order_id": numbers.get("order_id"),
                    "order_type": numbers.get("order_type"),
                    "amount": _execution_float(numbers.get("amount")),
                    "price": _execution_float(numbers.get("price")),
                    "shares": _execution_float(numbers.get("shares")),
                    "token_id": numbers.get("token_id"),
                    "placement_state": numbers.get("placement_state"),
                    "status": numbers.get("existing_status"),
                }.items()
                if value not in (None, "")
            }
        )

    if operator_decision:
        verdict = str(operator_decision.get("verdict") or "").upper() or "PASS"
        rationale = str(operator_decision.get("rationale") or "")
        path["operator"] = {
            "verdict": verdict,
            "rationale": rationale,
            "confidence": operator_decision.get("confidence"),
            "risk_flags": operator_decision.get("risk_flags") or [],
            "decision_key": operator_decision.get("decision_key") or "",
            "timestamp": operator_decision.get("timestamp") or "",
        }
        _execution_append_stage(
            path,
            {
                "stage": "operator",
                "status": "operator_pass" if verdict == "PASS" else "blocked",
                "gate": "llm_operator_pass" if verdict == "PASS" else "llm_operator_block",
                "explanation": (
                    f"LLM Operator {verdict}: {rationale}"
                    if rationale
                    else f"LLM Operator {verdict}."
                ),
                "numbers": {},
                "timestamp": operator_decision.get("timestamp") or "",
            },
        )

    if tracker_record:
        path["tracker_decision"] = {
            "status": tracker_record.get("status"),
            "rationale": tracker_record.get("rationale"),
            "summary": tracker_record.get("summary"),
            "confidence": tracker_record.get("confidence"),
            "decision_id": tracker_record.get("decision_id"),
            "timestamp": tracker_record.get("timestamp"),
            "outcome": tracker_record.get("outcome") or {},
        }
        _execution_append_stage(
            path,
            {
                "stage": "tracker_selection",
                "status": tracker_record.get("status") or "candidate",
                "gate": f"tracker_{tracker_record.get('status') or 'decision'}",
                "explanation": (
                    tracker_record.get("rationale")
                    or tracker_record.get("summary")
                    or "Tracker decision matched this existing bet."
                ),
                "numbers": {},
                "timestamp": tracker_record.get("timestamp") or "",
            },
        )

    label = path.get("label") or _EXECUTION_PATH_LABELS.get(str(trader), str(trader))
    amount = _execution_format_money(numbers.get("amount"))
    price = _execution_format_price(numbers.get("price"))
    ledger_id = numbers.get("existing_ledger_id") or numbers.get("ledger_id")
    order_id = numbers.get("order_id")
    market_id = numbers.get("market_id")
    reason = path.get("original_reason")
    operator = path.get("operator") if isinstance(path.get("operator"), dict) else {}
    tracker = path.get("tracker_decision") if isinstance(path.get("tracker_decision"), dict) else {}

    first_sentence = f"Already bet by {label}"
    if amount and price:
        first_sentence += f": {amount} at {price}"
    elif amount:
        first_sentence += f": {amount}"
    if market_id:
        first_sentence += f" on market {market_id}"
    refs = []
    if ledger_id:
        refs.append(f"ledger #{ledger_id}")
    if order_id:
        refs.append(f"order {order_id}")
    if refs:
        first_sentence += f" ({', '.join(refs)})"

    sentences = [
        first_sentence + ".",
        f"Current cycle did not place another order because {path.get('gate') or 'the duplicate gate'} found the existing position.",
    ]
    if reason:
        sentences.append(f"Original reason: {reason}")
    if operator:
        verdict = operator.get("verdict") or "PASS"
        rationale = operator.get("rationale")
        sentences.append(
            f"Operator {verdict}: {rationale}" if rationale else f"Operator {verdict}."
        )
    if tracker and (tracker.get("rationale") or tracker.get("summary")):
        sentences.append(f"Tracker rationale: {tracker.get('rationale') or tracker.get('summary')}")
    path["explanation"] = " ".join(sentences)


def _normalize_execution_breakdown_cycle(
    cycle: dict | None,
    enrichment: dict | None = None,
) -> dict | None:
    """Normalize older audit records whose duplicate/open-position gate said skipped."""
    if not isinstance(cycle, dict):
        return cycle
    normalized = copy.deepcopy(cycle)
    enrichment = enrichment or {"ledger_bets": [], "operator_decisions": [], "tracker_records": []}
    path_counts: dict[str, dict[str, int]] = {}
    for fight in normalized.get("fights", []) or []:
        paths = fight.get("paths", {})
        if not isinstance(paths, dict):
            continue
        for trader, path in paths.items():
            if not isinstance(path, dict):
                continue
            gate = str(path.get("gate") or "")
            status = str(path.get("status") or "")
            if gate in _EXECUTION_ALREADY_BET_GATES and status == "skipped":
                path["status"] = "already_bet"
                explanation = str(path.get("explanation") or "")
                if explanation.startswith("Skipped because already have "):
                    path["explanation"] = explanation.replace(
                        "Skipped because already have ",
                        "Already have ",
                        1,
                    )
                elif explanation.startswith("Skipped by executor because "):
                    path["explanation"] = explanation.replace(
                        "Skipped by executor because ",
                        "Already have ",
                        1,
                    )

            if (
                str(path.get("gate") or "") in _EXECUTION_ALREADY_BET_GATES
                or str(path.get("status") or "") == "already_bet"
            ):
                path["status"] = "already_bet"
                _execution_enrich_already_bet_path(fight, str(trader), path, enrichment)

            current_status = str(path.get("status") or "unknown")
            path_counts.setdefault(str(trader), {})
            path_counts[str(trader)][current_status] = (
                path_counts[str(trader)].get(current_status, 0) + 1
            )
    normalized["path_counts"] = path_counts
    return normalized


@app.route("/api/execution-breakdown")
def api_execution_breakdown():
    """Return the structured per-cycle execution decision audit."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error

    try:
        from src.strategy.execution_audit import (
            load_execution_audit_cycle,
            load_execution_audit_cycles,
            load_latest_execution_audit,
        )

        log_path = LOGS_DIR / "execution_decision_audit.jsonl"
        latest_path = LOGS_DIR / "execution_decision_audit_latest.json"
        history = request.args.get("history", "") == "1"
        cycle_id = str(request.args.get("cycle_id", "") or "").strip()
        offset_raw = request.args.get("offset", "0")
        limit_raw = request.args.get("limit", "20")
        try:
            limit = min(max(int(limit_raw), 1), 100)
        except (TypeError, ValueError):
            limit = 20
        try:
            offset = max(int(offset_raw), 0)
        except (TypeError, ValueError):
            offset = 0

        enrichment = _execution_load_enrichment_context()
        if history or cycle_id:
            page = load_execution_audit_cycles(
                limit=limit + 1,
                offset=offset,
                log_path=log_path,
            )
            has_more = len(page) > limit
            page = page[:limit]
            cycles = [
                {
                    key: cycle.get(key)
                    for key in (
                        "schema_version",
                        "cycle_id",
                        "started_at",
                        "completed_at",
                        "dry_run",
                        "event_title",
                        "fight_count",
                        "path_counts",
                    )
                }
                for cycle in page
                if isinstance(cycle, dict)
            ]
            if cycle_id:
                selected = load_execution_audit_cycle(
                    cycle_id,
                    log_path=log_path,
                )
            else:
                selected = load_latest_execution_audit(
                    latest_path=latest_path,
                    log_path=log_path,
                )
            selected = _normalize_execution_breakdown_cycle(selected, enrichment=enrichment)
            return _dashboard_read_response(
                _sanitize_for_json(
                    {
                        "cycle": selected,
                        "cycles": cycles,
                        "count": len(cycles),
                        "latest_available": selected is not None,
                        "offset": offset,
                        "next_offset": offset + len(cycles) if has_more else None,
                        "has_more": has_more,
                    }
                )
            )

        latest = load_latest_execution_audit(latest_path=latest_path, log_path=log_path)
        latest = _normalize_execution_breakdown_cycle(latest, enrichment=enrichment)
        return _dashboard_read_response(
            _sanitize_for_json(
                {
                    "cycle": latest,
                    "cycles": [latest] if latest else [],
                    "count": 1 if latest else 0,
                    "latest_available": latest is not None,
                }
            )
        )
    except Exception as e:
        logger.exception("Failed to load execution breakdown audit")
        return _api_internal_error(
            "execution_breakdown_unavailable",
            "Execution audit data could not be loaded.",
        )


def _tracker_decisions_cache_key(show_history: bool) -> str:
    try:
        from src.strategy.llm_operator import load_decision_log, load_tracker_decision_log
        decision_loader_key = _callable_cache_fingerprint(load_decision_log)
        tracker_loader_key = _callable_cache_fingerprint(load_tracker_decision_log)
    except Exception:
        decision_loader_key = "unavailable"
        tracker_loader_key = "unavailable"
    return (
        f"tracker-decisions:{int(show_history)}:{LOGS_DIR}:"
        f"{_callable_cache_fingerprint(_load_prediction_payload)}:"
        f"{_callable_cache_fingerprint(load_all_trader_ledgers)}:"
        f"{decision_loader_key}:{tracker_loader_key}"
    )


def _callable_cache_fingerprint(fn) -> str:
    if not callable(fn):
        return "unavailable"

    code = getattr(fn, "__code__", None)
    module = getattr(fn, "__module__", "")
    qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", type(fn).__name__))
    if code is None:
        return f"{module}:{qualname}:{type(fn).__name__}"

    closure_values = []
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            closure_values.append(repr(cell.cell_contents)[:160])
        except ValueError:
            closure_values.append("<empty>")

    return ":".join([
        str(module),
        str(qualname),
        str(code.co_filename),
        str(code.co_firstlineno),
        str(code.co_argcount),
        repr(code.co_consts),
        repr(code.co_names),
        "|".join(closure_values),
    ])


@app.route("/api/open-limit-orders")
def api_open_limit_orders():
    """Return live open limit orders, or 503 if the live source is unavailable."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    try:
        display_orders = _cached(
            "open-limit-orders-display",
            LIMIT_ORDER_DISPLAY_TTL,
            _compute_limit_orders_display,
        )
        _kickoff_limit_order_reconcile(open_order_ids=_extract_open_order_ids(display_orders))
        return _json_no_store(display_orders)
    except RuntimeError as e:
        logger.debug("Open limit order display unavailable: %s", e)
        return _json_no_store({"ok": False, "error": "live_open_orders_unavailable", "message": str(e)}), 503


@app.route("/api/reconcile-limit-orders", methods=["POST"])
def api_reconcile_limit_orders():
    """Force CLOB reconciliation of limit orders (manual trigger).

    Uses read auth — this only checks CLOB statuses and updates ledger,
    it doesn't place orders or move funds.
    """
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    # Bust the cache so it re-runs
    with _cache_lock:
        _endpoint_cache.pop("limit-order-clob-reconcile", None)
        _endpoint_cache.pop("open-limit-orders-display", None)
    try:
        result = _reconcile_limit_orders_with_clob()
        return _json_no_store({"status": "ok", "reconciled": result.get("reconciled", 0), "cancelled": result.get("cancelled", 0)})
    except Exception as e:
        logger.error("Manual limit order reconciliation failed: %s", e)
        return _json_no_store({"status": "error", "message": str(e)}), 500


def _extract_open_order_ids(orders: list[dict] | None) -> set[str]:
    ids = set()
    for order in orders or []:
        order_id = str((order or {}).get("order_id", "") or "").strip()
        if order_id:
            ids.add(order_id)
    return ids


def _kickoff_limit_order_reconcile(
    open_order_ids: set[str] | None = None,
    ttl_seconds: int = 21600,
) -> None:
    """Start a best-effort reconcile in the background without blocking the UI."""
    global _background_cache_refreshes
    key = "limit-order-clob-reconcile"

    with _cache_lock:
        entry = _endpoint_cache.get(key)
        if entry and time.time() - entry["ts"] < ttl_seconds:
            return
        if key in _endpoint_inflight:
            return
        if _background_cache_refreshes >= _MAX_BACKGROUND_CACHE_REFRESHES:
            logger.debug("Skipping limit order reconciliation; refresh workers are saturated")
            return
        _endpoint_inflight[key] = {"event": threading.Event()}
        _background_cache_refreshes += 1

    def _worker():
        global _background_cache_refreshes
        try:
            data = _reconcile_limit_orders_with_clob(open_order_ids=open_order_ids)
        except Exception as e:
            logger.warning("Background limit order reconciliation failed: %s", e)
            data = {"error": str(e)}
        finally:
            with _cache_lock:
                pending = _endpoint_inflight.pop(key, None)
                _background_cache_refreshes = max(0, _background_cache_refreshes - 1)
                _endpoint_cache[key] = {"data": data, "ts": time.time()}
                if pending is not None:
                    pending["event"].set()

    thread = threading.Thread(
        target=_worker,
        name="limit-order-reconcile",
        daemon=True,
    )
    try:
        thread.start()
    except RuntimeError as exc:
        with _cache_lock:
            pending = _endpoint_inflight.pop(key, None)
            _background_cache_refreshes = max(0, _background_cache_refreshes - 1)
            if pending is not None:
                pending["event"].set()
        logger.warning("Could not start limit order reconciliation worker: %s", exc)


def _build_token_to_fighter_map():
    """Build token_id -> {fighter, opponent, event_date} from live Polymarket markets."""
    from src.polymarket.markets import get_ufc_fight_markets
    try:
        markets = get_ufc_fight_markets()
        token_map = {}
        for _, m in markets.iterrows():
            tid_yes = m.get("token_id_yes", "")
            tid_no = m.get("token_id_no", "")
            end_date = m.get("end_date", "")
            cid = m.get("condition_id", "")
            if tid_yes:
                token_map[tid_yes] = {
                    "fighter": m.get("fighter_a", ""),
                    "opponent": m.get("fighter_b", ""),
                    "event_date": end_date,
                    "side": "a",
                    "condition_id": cid,
                }
            if tid_no:
                token_map[tid_no] = {
                    "fighter": m.get("fighter_b", ""),
                    "opponent": m.get("fighter_a", ""),
                    "event_date": end_date,
                    "side": "b",
                    "condition_id": cid,
                }
        return token_map
    except Exception as e:
        logger.warning(f"Failed to build token-to-fighter map: {e}")
        return {}


def _safe_float(value, default: float = 0.0) -> float:
    """Convert mixed API values to float without throwing."""
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _unwrap_clob_order(payload) -> dict:
    """Normalize CLOB responses that may wrap the order under `order` or `data`."""
    if not isinstance(payload, dict):
        return {}
    for key in ("order", "data"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return nested
    return payload


def _limit_order_types() -> tuple[str, ...]:
    return ("limit_bid", "limit", "near_miss_limit", "marketable_limit")


def _normalize_limit_status(raw_status) -> str:
    return str(raw_status or "").strip().lower().replace("-", "_").replace(" ", "_")


def _is_open_clob_order(order: dict) -> bool:
    """Keep only orders that are still actively open on Polymarket."""
    normalized_status = _normalize_limit_status(
        order.get("status") or order.get("order_status") or order.get("state")
    )

    if normalized_status:
        openish = any(token in normalized_status for token in ("live", "open", "rest", "unmatch", "active", "delay"))
        closedish = any(token in normalized_status for token in ("cancel", "expire", "reject", "void", "match", "fill", "execut", "complete", "close"))
        if closedish and not openish:
            return False

    original_size = _safe_float(order.get("original_size", order.get("size")), 0.0)
    size_matched = _safe_float(order.get("size_matched"), 0.0)
    if original_size > 0 and size_matched >= max(original_size - 1e-9, 0.0):
        return False

    return True


def _normalize_open_clob_orders(payload) -> list[dict]:
    normalized = []
    seen_ids = set()

    for raw_order in list(payload or []):
        order = _unwrap_clob_order(raw_order)
        if not order or not _is_open_clob_order(order):
            continue

        order_id = str(order.get("id", "") or "").strip()
        if order_id:
            if order_id in seen_ids:
                continue
            seen_ids.add(order_id)

        normalized.append(order)

    return normalized


def _call_with_timeout(action: str, fn, timeout_seconds: float):
    """Run a blocking read with a hard wait budget so dashboard requests stay responsive."""
    if not callable(fn):
        return None

    with _timed_call_lock:
        previous = _timed_call_inflight.get(action)
        if previous is not None and not previous.is_set():
            logger.debug("Skipping %s because a previous call is still running", action)
            return None

    result = {}
    error = {}
    done = threading.Event()
    with _timed_call_lock:
        _timed_call_inflight[action] = done

    def _worker():
        try:
            result["value"] = fn()
        except Exception as e:
            error["value"] = e
        finally:
            done.set()
            with _timed_call_lock:
                if _timed_call_inflight.get(action) is done:
                    _timed_call_inflight.pop(action, None)

    thread = threading.Thread(
        target=_worker,
        name=f"clob-{re.sub(r'[^a-z0-9]+', '-', action.lower()).strip('-') or 'call'}",
        daemon=True,
    )
    try:
        thread.start()
    except RuntimeError as exc:
        done.set()
        with _timed_call_lock:
            if _timed_call_inflight.get(action) is done:
                _timed_call_inflight.pop(action, None)
        logger.warning("Could not start worker while %s: %s", action, exc)
        return None

    if not done.wait(timeout_seconds):
        level = logging.DEBUG if action == "fetching open orders" else logging.WARNING
        logger.log(level, "Timed out after %.1fs while %s", timeout_seconds, action)
        return None

    if "value" in error:
        level = logging.DEBUG if action == "fetching open orders" else logging.WARNING
        logger.log(level, "%s failed: %s", action[:1].upper() + action[1:], error["value"])
        return None

    return result.get("value")


def _clob_call_with_timeout(action: str, fn, timeout_seconds: float):
    """Run a CLOB read with a hard wait budget so dashboard requests stay responsive."""
    return _call_with_timeout(action, fn, timeout_seconds)


def _get_open_clob_orders(timeout_seconds: float = LIMIT_ORDER_CLOB_TIMEOUT_SECONDS) -> list[dict] | None:
    if not _clob_client or not hasattr(_clob_client, "get_open_orders"):
        return None
    for attempt in range(2):
        started_at = time.monotonic()
        payload = _clob_call_with_timeout(
            "fetching open orders",
            _clob_client.get_open_orders,
            timeout_seconds,
        )
        if payload is not None:
            return _normalize_open_clob_orders(payload)
        if time.monotonic() - started_at >= timeout_seconds:
            break
        if attempt == 0:
            time.sleep(0.2)
    return None


def _get_clob_order(order_id: str, timeout_seconds: float = LIMIT_ORDER_CLOB_TIMEOUT_SECONDS) -> dict:
    if not _clob_client or not hasattr(_clob_client, "get_order") or not order_id:
        return {}
    payload = _clob_call_with_timeout(
        f"fetching order {order_id}",
        lambda: _clob_client.get_order(order_id),
        timeout_seconds,
    )
    return _unwrap_clob_order(payload) if payload else {}


def _collect_open_limit_ledger_entries():
    from src.strategy.duo_trader import get_all_trader_ledgers

    entries = []
    by_order_id = defaultdict(list)
    by_token_id = defaultdict(list)

    for label, path in get_all_trader_ledgers():
        ledger = BetLedger(path=path)
        for bet in ledger.bets:
            if bet.get("status") != "open":
                continue
            if bet.get("order_type") not in _limit_order_types():
                continue

            entry = {"bet": bet, "trader": label, "ledger_path": path}
            entries.append(entry)

            order_id = str(bet.get("order_id", "") or "").strip()
            if order_id:
                by_order_id[order_id].append(entry)

            token_id = str(bet.get("token_id", "") or "").strip()
            if token_id:
                by_token_id[token_id].append(entry)

    return entries, by_order_id, by_token_id


def _coerce_limit_order_timestamp(*values):
    for value in values:
        if value in (None, ""):
            continue
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
            except (OSError, OverflowError, ValueError):
                continue
        raw = str(value).strip()
        if not raw:
            continue
        try:
            if re.fullmatch(r"\d+(?:\.\d+)?", raw):
                return datetime.fromtimestamp(float(raw), tz=timezone.utc).isoformat()
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            return parsed.isoformat()
        except ValueError:
            return raw
    return None


def _parse_placed_at_sort_key(value) -> tuple[int, str]:
    """Return a stable sortable key for ledger/CLOB timestamps."""
    if value in (None, ""):
        return (0, "")
    raw = str(value)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return (1, parsed.isoformat())
    except ValueError:
        return (0, raw)


def _is_recovered_limit_placeholder(bet: dict | None) -> bool:
    """Detect ledger entries rebuilt from CLOB after a ledger wipe.

    These synthetic rows have no model context, so their `edge=0.0` means
    "unknown", not "true zero edge".
    """
    if not bet:
        return False
    return (
        _safe_float(bet.get("model_prob"), 0.0) == 0.0
        and _safe_float(bet.get("edge"), 0.0) == 0.0
        and str(bet.get("market_id", "")).strip() == ""
        and bet.get("order_type") in _limit_order_types()
    )


def _pick_best_limit_match(candidates: list[dict]) -> dict | None:
    """Prefer real ledger rows over recovery placeholders for the same order."""
    if not candidates:
        return None

    def _is_limit_type(bet: dict) -> bool:
        return bet.get("order_type") in _limit_order_types()

    def _score(bet: dict) -> tuple[int, int, tuple[int, str], int]:
        return (
            1 if _is_limit_type(bet) else 0,
            0 if _is_recovered_limit_placeholder(bet) else 1,
            1 if bet.get("order_id") else 0,
            _parse_placed_at_sort_key(bet.get("placed_at")),
            int(bet.get("id") or 0),
        )

    return max(candidates, key=_score)


def _display_edge_for_limit_order(bet: dict | None):
    """Hide placeholder 0.0 edges recovered from CLOB."""
    if not bet or _is_recovered_limit_placeholder(bet):
        return None
    return bet.get("edge")


def _match_live_limit_order_to_ledger(order: dict, by_order_id, by_token_id):
    candidates = []
    seen = set()
    order_id = str(order.get("id", "") or "").strip()
    token_id = str(order.get("asset_id", order.get("token_id", "")) or "").strip()

    for entry in itertools.chain(by_order_id.get(order_id, []), by_token_id.get(token_id, [])):
        key = (str(entry["ledger_path"]), int(entry["bet"].get("id") or 0))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(entry)

    if not candidates:
        return None

    best_bet = _pick_best_limit_match([entry["bet"] for entry in candidates])
    for entry in candidates:
        if entry["bet"] == best_bet:
            return entry
    return candidates[0]


def _title_parts_from_order(order: dict) -> tuple[str, str]:
    title = str(
        order.get("title")
        or order.get("question")
        or order.get("market")
        or order.get("slug")
        or ""
    ).strip()
    if not title:
        return "", ""
    if " vs " in title:
        left, right = title.split(" vs ", 1)
        return left.strip(), right.strip()
    return title, ""


def _fetch_limit_order_market_metadata(market_id: str) -> dict:
    market_id = str(market_id or "").strip()
    if not market_id:
        return {}

    try:
        from src.polymarket.client import GammaClient

        gamma = GammaClient()
        payload = _clob_call_with_timeout(
            f"fetching market metadata for {market_id[:12]}",
            lambda: gamma.get_market(market_id),
            LIMIT_ORDER_MARKET_LOOKUP_TIMEOUT_SECONDS,
        )
        if not isinstance(payload, dict):
            return {}
        return {
            "market_title": str(payload.get("question") or payload.get("title") or "").strip() or None,
            "event_slug": str(payload.get("event_slug") or payload.get("slug") or "").strip() or None,
        }
    except Exception as e:
        logger.warning("Failed to enrich limit order market metadata for %s: %s", market_id, e)
        return {}


def _market_metadata_by_market_id(orders: list[dict], token_map: dict[str, dict]) -> dict[str, dict]:
    market_ids = set()

    for order in orders:
        token_id = str(order.get("asset_id", order.get("token_id", "")) or "").strip()
        if token_id and token_id in token_map:
            continue
        if str(order.get("title") or order.get("question") or "").strip():
            continue

        market_id = str(order.get("market", "") or "").strip()
        if market_id:
            market_ids.add(market_id)

    metadata = {}
    for market_id in market_ids:
        metadata[market_id] = _cached(
            f"limit-order-market:{market_id}",
            LIMIT_ORDER_MARKET_METADATA_TTL,
            lambda market_id=market_id: _fetch_limit_order_market_metadata(market_id),
        )
    return metadata


def _serialize_live_limit_order(
    order: dict,
    ledger_entry,
    token_map: dict[str, dict],
    market_metadata: dict[str, dict] | None = None,
) -> dict:
    ledger_bet = ledger_entry["bet"] if ledger_entry else None
    token_id = str(order.get("asset_id", order.get("token_id", "")) or "").strip()
    market_id = str(order.get("market", "") or "").strip() or None
    token_info = token_map.get(token_id, {})
    market_info = (market_metadata or {}).get(market_id or "", {})
    fallback_fighter, fallback_opponent = _title_parts_from_order(order)
    resolved = _resolve_limit_order_state(order_data=order, ledger_bet=ledger_bet, on_clob=True)
    bid_price = _safe_float(order.get("price"), _safe_float((ledger_bet or {}).get("price"), 0.0))
    selection_label = str(order.get("outcome", "") or "").strip() or None
    market_title = (
        str(order.get("title") or order.get("question") or "").strip()
        or market_info.get("market_title")
    )

    return {
        "order_id": str(order.get("id", "") or (ledger_bet or {}).get("order_id") or "").strip() or None,
        "fighter": (
            (ledger_bet or {}).get("fighter")
            or token_info.get("fighter")
            or selection_label
            or fallback_fighter
            or (f"Token {token_id[:10]}..." if token_id else "Unknown")
        ),
        "opponent": (
            (ledger_bet or {}).get("opponent")
            or token_info.get("opponent")
            or fallback_opponent
        ),
        "selection_label": selection_label,
        "market_title": market_title or None,
        "market_id": market_id,
        "event_slug": market_info.get("event_slug"),
        "order_side": str(order.get("side", "") or "").strip() or None,
        "trader": ledger_entry["trader"] if ledger_entry else None,
        "bid_price": bid_price,
        "size_remaining": resolved["size_remaining"],
        "size_matched": resolved["size_matched"],
        "edge": _display_edge_for_limit_order(ledger_bet),
        "order_type": (ledger_bet or {}).get("order_type") or "limit_bid",
        "placed_at": _coerce_limit_order_timestamp(
            order.get("created_at"),
            order.get("timestamp"),
            (ledger_bet or {}).get("placed_at"),
        ),
        "event_date": (ledger_bet or {}).get("event_date") or token_info.get("event_date"),
        "on_clob": True,
        "status": resolved["status"],
        "status_note": None if ledger_entry else "Live on Polymarket but missing from the local ledger",
        "model_prob": (ledger_bet or {}).get("model_prob"),
        "market_prob": (ledger_bet or {}).get("market_prob", bid_price),
        "reason": (ledger_bet or {}).get("reason"),
    }


def _resolve_limit_order_state(order_data=None, ledger_bet=None, on_clob: bool = False) -> dict:
    """Derive a stable dashboard status from CLOB order data plus ledger fallback."""
    order = _unwrap_clob_order(order_data)
    raw_status = order.get("status") or order.get("order_status") or order.get("state")
    normalized_status = _normalize_limit_status(raw_status)

    shares_fallback = _safe_float((ledger_bet or {}).get("shares"), 0.0)
    original_size = _safe_float(
        order.get("original_size", order.get("size", shares_fallback)),
        shares_fallback,
    )
    size_matched = _safe_float(order.get("size_matched"), 0.0)
    size_remaining = max(original_size - size_matched, 0.0)

    openish = any(token in normalized_status for token in ("live", "open", "rest", "unmatch", "active", "delay"))
    filledish = any(token in normalized_status for token in ("match", "fill", "execut", "complete"))
    cancelled = any(token in normalized_status for token in ("cancel", "expire", "reject", "void"))

    if on_clob or openish:
        status = "partially_filled" if size_matched > 0 and size_remaining > 0 else "resting"
    elif cancelled:
        status = "partial_fill" if size_matched > 0 else "cancelled"
    elif size_matched > 0 and size_remaining <= 1e-9:
        status = "filled"
    elif filledish:
        status = "filled" if size_remaining <= 1e-9 else "partial_fill"
    elif size_matched > 0:
        status = "partial_fill"
    elif normalized_status:
        status = "closed"
    else:
        status = "unknown"

    if status == "filled" and size_matched <= 0 and shares_fallback > 0:
        size_matched = shares_fallback

    if status not in ("resting", "partially_filled"):
        size_remaining = 0.0

    return {
        "status": status,
        "raw_status": raw_status,
        "size_remaining": size_remaining,
        "size_matched": size_matched,
    }


def _compute_limit_orders_from_ledger():
    """Fast ledger-only limit order display — no CLOB calls.

    Returns open limit orders from the ledger enriched with operator decisions.
    CLOB reconciliation (which updates ledger statuses) runs separately on a 6h cadence.
    """
    results = []
    for label, _, path, _ in _trader_breakdown_specs():
        ledger = BetLedger(path=path)
        for bet in ledger.bets:
            is_open = bet.get("status") == "open"
            is_limit = bet.get("order_type") in _limit_order_types()
            if not (is_open and is_limit):
                continue

            results.append({
                "order_id": bet.get("order_id"),
                "fighter": bet.get("fighter"),
                "opponent": bet.get("opponent"),
                "trader": label,
                "bid_price": _safe_float(bet.get("price"), 0.0),
                "size_remaining": _safe_float(bet.get("shares"), 0.0),
                "size_matched": 0.0,
                "edge": _display_edge_for_limit_order(bet),
                "order_type": bet.get("order_type"),
                "placed_at": bet.get("placed_at"),
                "event_date": bet.get("event_date"),
                "on_clob": None,  # unknown until reconciliation runs
                "status": "resting",
                "status_note": None,
                "model_prob": bet.get("model_prob"),
                "market_prob": bet.get("market_prob"),
                "reason": bet.get("reason"),
            })

    # Enrich with operator decisions
    try:
        from src.strategy.llm_operator import load_decision_log
        all_decisions = load_decision_log()
        all_decisions.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
        decisions_index = _build_decisions_index(all_decisions[:200])

        for order in results:
            decision = _match_decision_to_bet(order, decisions_index)
            if decision:
                order["operator_rationale"] = decision.get("rationale")
                order["operator_verdict"] = decision.get("verdict")
                order["operator_prob"] = decision.get("operator_prob")
                order["research_summary"] = decision.get("research_summary")
                order["risk_flags"] = decision.get("risk_flags")
    except Exception as e:
        logger.warning("Failed to enrich limit orders with operator decisions: %s", e)

    # Sort by placed_at descending (newest first)
    results.sort(key=lambda x: x.get("placed_at") or "", reverse=True)
    return results


def _compute_limit_orders_display():
    """Return live open orders enriched from the ledger."""
    live_orders = _get_open_clob_orders()
    if live_orders is None:
        raise RuntimeError("Polymarket open orders are temporarily unavailable")

    _, by_order_id, by_token_id = _collect_open_limit_ledger_entries()
    token_map = _build_token_to_fighter_map()
    market_metadata = _market_metadata_by_market_id(live_orders, token_map)
    results = []

    for order in live_orders:
        ledger_entry = _match_live_limit_order_to_ledger(order, by_order_id, by_token_id)
        results.append(_serialize_live_limit_order(order, ledger_entry, token_map, market_metadata))

    try:
        from src.strategy.llm_operator import load_decision_log

        all_decisions = load_decision_log()
        all_decisions.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
        decisions_index = _build_decisions_index(all_decisions[:200])

        for order in results:
            decision = _match_decision_to_bet(order, decisions_index)
            if decision:
                order["operator_rationale"] = decision.get("rationale")
                order["operator_verdict"] = decision.get("verdict")
                order["operator_prob"] = decision.get("operator_prob")
                order["research_summary"] = decision.get("research_summary")
                order["risk_flags"] = decision.get("risk_flags")
    except Exception as e:
        logger.warning("Failed to enrich limit orders with operator decisions: %s", e)

    results.sort(key=lambda x: _parse_placed_at_sort_key(x.get("placed_at")), reverse=True)
    return results


def _reconcile_limit_orders_with_clob(open_order_ids: set[str] | None = None):
    """CLOB reconciliation — runs every 6h or on manual trigger.

    Checks Polymarket CLOB for ground-truth order statuses and updates
    ledger entries for orders that have been cancelled/filled externally.
    Returns stats about what was reconciled.
    """
    # Collect open limit bets from ledger
    ledger_orders = {}  # order_id -> (bet, trader_label, ledger_path)
    for label, _, path, _ in _trader_breakdown_specs():
        ledger = BetLedger(path=path)
        for bet in ledger.bets:
            is_open = bet.get("status") == "open"
            is_limit = bet.get("order_type") in _limit_order_types()
            oid = bet.get("order_id")
            if is_open and is_limit and oid:
                ledger_orders[str(oid)] = (bet, label, path)

    if not ledger_orders:
        return {"reconciled": 0, "cancelled": 0}

    # Fetch ground-truth open orders from CLOB
    clob_open_ids = set(open_order_ids or ())
    if _clob_client:
        if open_order_ids is None:
            open_orders = _get_open_clob_orders()
            if open_orders is None:
                logger.warning("CLOB reconciliation could not fetch open orders within %.1fs", LIMIT_ORDER_CLOB_TIMEOUT_SECONDS)
                return {"reconciled": 0, "cancelled": 0, "error": "open_orders_unavailable"}
            for order in open_orders:
                oid = order.get("id", "")
                if oid:
                    clob_open_ids.add(str(oid))

    reconciled = 0
    cancelled = 0

    # Check each ledger order against CLOB ground truth
    for oid, (bet, trader_label, ledger_path) in ledger_orders.items():
        if oid in clob_open_ids:
            continue  # Still open on CLOB, nothing to do

        # Order not in open orders — check if it was filled or cancelled
        closed_order = _get_clob_order(oid)

        resolved = _resolve_limit_order_state(order_data=closed_order, ledger_bet=bet, on_clob=False)

        bet_id = bet.get("id")
        if bet_id is None:
            continue

        if resolved["status"] in ("cancelled", "unknown", "closed"):
            # Order is gone from CLOB — cancel in ledger
            try:
                reason = "clob_reconciled_cancelled" if resolved["status"] == "cancelled" else "clob_reconciled_gone"
                BetLedger(path=ledger_path).cancel_bet(bet_id, reason=reason)
                logger.info(
                    "CLOB reconciliation: cancelled ledger bet #%s (%s) — %s",
                    bet_id, bet.get("fighter"), resolved["status"],
                )
                cancelled += 1
            except Exception as e:
                logger.debug("Failed to reconcile bet #%s: %s", bet_id, e)
        elif resolved["status"] == "filled":
            logger.info(
                "CLOB reconciliation: order #%s (%s) is filled — ledger will be updated by settlement",
                bet_id, bet.get("fighter"),
            )

        reconciled += 1

    # Bust the display cache so the next poll picks up changes
    with _cache_lock:
        _endpoint_cache.pop("open-limit-orders-display", None)

    logger.info("CLOB reconciliation complete: %d checked, %d cancelled", reconciled, cancelled)
    return {"reconciled": reconciled, "cancelled": cancelled}


@app.route("/predictions")
def predictions_page():
    return _html_no_store("predictions.html")


@app.route("/activity")
def activity_page():
    return _html_no_store("activity.html")


@app.route("/bet-history")
def bet_history_page():
    return _html_no_store("bet_history.html")


@app.route("/operator")
def operator_page():
    return _html_no_store("operator.html")


@app.route("/reasoning")
def reasoning_page():
    return _html_no_store("reasoning.html")


@app.route("/execution-breakdown")
def execution_breakdown_page():
    return _html_no_store("execution_breakdown.html")


@app.route("/api/operator-decisions")
def api_operator_decisions():
    """Return LLM Operator decision log entries."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    sport = str(request.args.get("sport", "all") or "all").strip().lower()
    try:
        raw_limit = request.args.get("limit")
        limit = None if raw_limit in (None, "") else min(max(1, int(raw_limit)), 1000)
    except (TypeError, ValueError):
        limit = None
    try:
        decisions = []
        from src.strategy.llm_operator import load_decision_log
        for d in load_decision_log():
            d["sport"] = "ufc"
            decisions.append(d)
        # Sort by timestamp descending (most recent first)
        decisions.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
        total_count = len(decisions)
        pass_verdicts = {"PASS", "NO_VETO"}
        block_verdicts = {"BLOCK", "AUTO_BLOCK", "AUTO_SKIP"}
        pass_count = sum(1 for decision in decisions if str(decision.get("verdict") or "").upper() in pass_verdicts)
        block_count = sum(1 for decision in decisions if str(decision.get("verdict") or "").upper() in block_verdicts)
        if limit is not None:
            decisions = decisions[:limit]
        return _dashboard_read_response({
            "decisions": decisions,
            "count": len(decisions),
            "total_count": total_count,
            "stats": {
                "total": total_count,
                "passed": pass_count,
                "blocked": block_count,
                "block_rate": (block_count / total_count) if total_count else 0.0,
            },
        })
    except Exception as e:
        logger.exception("Failed to load operator decisions")
        return _api_internal_error(
            "operator_decisions_unavailable",
            "Operator decisions could not be loaded.",
        )


def _reasoning_has_research(grounded_research: dict | None) -> bool:
    gr = grounded_research or {}
    return bool(
        str(gr.get("memo_text") or "").strip()
        or str(gr.get("recent_form") or "").strip()
        or str(gr.get("level_of_competition") or "").strip()
        or str(gr.get("style_matchup") or "").strip()
        or str(gr.get("paths_to_victory") or "").strip()
        or str(gr.get("model_pick_concerns") or "").strip()
        or (gr.get("key_flags") or [])
        or (gr.get("sources") or [])
    )


def _operator_context_label(decision_context: str) -> str:
    raw = str(decision_context or "").strip().upper()
    head = raw.split(":", 1)[0] if raw else ""
    names = {"S": "Single", "C": "Conviction"}
    if head in names:
        return f"Operator · {names[head]} ({head})"
    return "Operator"


def _reasoning_date_has_time(raw_value: object) -> bool:
    raw = str(raw_value or "").strip()
    if not raw:
        return False
    return bool(re.search(r"(?:T|\s)\d{1,2}:\d{2}", raw))


def _reasoning_current_fight_status(*, event_date: str, market_event_date: str) -> str:
    raw = str(market_event_date or event_date or "").strip()
    parsed = _normalize_upcoming_event_datetime(_parse_upcoming_event_datetime(raw))
    if parsed is None:
        return ""

    now = datetime.now(timezone.utc)
    if _reasoning_date_has_time(raw):
        return "completed" if parsed <= now else "upcoming"

    event_day = parsed.date()
    today = now.date()
    if event_day < today:
        return "completed"
    if event_day > today:
        return "upcoming"
    return "fight_day"


def _reasoning_display_fight_status(
    grounded_research: dict | None,
    *,
    event_date: str,
    market_event_date: str,
) -> str:
    research_status = str((grounded_research or {}).get("fight_status") or "").strip().lower()
    if research_status in {"cancelled", "completed"}:
        return research_status

    current_status = _reasoning_current_fight_status(
        event_date=event_date,
        market_event_date=market_event_date,
    )
    if current_status:
        return current_status
    return research_status


def _normalize_operator_reasoning_entry(d: dict) -> dict:
    research_summary = d.get("research_summary") or {}
    grounded = research_summary.get("grounded_research") or {}
    context = str(d.get("decision_context", "") or "")
    event_date = str(d.get("event_date", "") or "")
    market_event_date = str(d.get("market_event_date", "") or "")
    return {
        "source": "operator",
        "source_label": _operator_context_label(context),
        "decision_context": context,
        "fighter_a": str(d.get("fighter_a", "") or ""),
        "fighter_b": str(d.get("fighter_b", "") or ""),
        "bet_on": str(d.get("bet_on", "") or ""),
        "pick": str(d.get("bet_on", "") or ""),
        "event_title": str(d.get("event_title", "") or ""),
        "event_date": event_date,
        "market_event_date": market_event_date,
        "weight_class": "",
        "timestamp": str(d.get("timestamp", "") or ""),
        "fight_status": _reasoning_display_fight_status(
            grounded,
            event_date=event_date,
            market_event_date=market_event_date,
        ),
        "research_fight_status": str(grounded.get("fight_status", "") or ""),
        "verdict": str(d.get("verdict", "") or "") or None,
        "status": "",
        "confidence": d.get("confidence"),
        "model_prob": d.get("model_prob"),
        "market_prob": d.get("market_prob"),
        "operator_prob": d.get("operator_prob"),
        "edge": d.get("edge"),
        "rationale": str(d.get("rationale", "") or ""),
        "fighter_assessment": str(research_summary.get("fighter_assessment", "") or ""),
        "risk_flags": list(d.get("risk_flags") or []),
        "grounded_research": grounded,
        "verified_records": grounded.get("verified_records")
        or research_summary.get("verified_records")
        or {},
        "sources": _safe_research_urls(grounded.get("sources") or []),
        "model_used": str(grounded.get("model_used", "") or ""),
        "cached": bool(grounded.get("cached")),
        "has_research": _reasoning_has_research(grounded),
        "sport": "ufc",
    }


def _normalize_tracker_reasoning_entry(r: dict) -> dict:
    grounded = r.get("grounded_research") or {}
    confidence = r.get("confidence")
    if confidence is None:
        confidence = r.get("signal_confidence")
    event_date = str(r.get("event_date", "") or "")
    market_event_date = str(r.get("market_event_date", "") or "")
    return {
        "source": "tracker",
        "source_label": "Gemini Tracker (G)",
        "decision_context": "G",
        "fighter_a": str(r.get("fighter_a", "") or ""),
        "fighter_b": str(r.get("fighter_b", "") or ""),
        "bet_on": str(r.get("pick", "") or ""),
        "pick": str(r.get("pick", "") or ""),
        "event_title": str(r.get("event_title", "") or ""),
        "event_date": event_date,
        "market_event_date": market_event_date,
        "weight_class": str(r.get("weight_class", "") or ""),
        "timestamp": str(r.get("timestamp", "") or ""),
        "fight_status": _reasoning_display_fight_status(
            grounded,
            event_date=event_date,
            market_event_date=market_event_date,
        ),
        "research_fight_status": str(grounded.get("fight_status", "") or ""),
        "verdict": None,
        "status": str(r.get("status", "") or ""),
        "confidence": confidence,
        "model_prob": None,
        "market_prob": r.get("market_prob"),
        "operator_prob": None,
        "edge": r.get("edge"),
        "rationale": str(r.get("rationale", "") or ""),
        "fighter_assessment": str(r.get("fighter_assessment", "") or ""),
        "risk_flags": list(r.get("risk_flags") or []),
        "grounded_research": grounded,
        "verified_records": grounded.get("verified_records")
        or r.get("verified_records")
        or {},
        "sources": _safe_research_urls(grounded.get("sources") or r.get("sources") or []),
        "model_used": str(grounded.get("model_used", "") or ""),
        "cached": bool(r.get("cached")),
        "has_research": _reasoning_has_research(grounded),
        "sport": "ufc",
    }


def _tracker_reasoning_key(record: dict):
    decision_id = str(record.get("decision_id", "") or "").strip()
    if decision_id:
        return ("decision_id", decision_id)
    return (
        "fight",
        str(record.get("trader", "") or "").strip().upper(),
        *_fight_matrix_key(
            str(record.get("fighter_a", "") or ""),
            str(record.get("fighter_b", "") or ""),
            str(record.get("market_event_date") or record.get("event_date") or ""),
            card_date=_row_card_date(record),
        ),
    )


def _tracker_reasoning_record_priority(record: dict) -> tuple[int, str]:
    status = str(record.get("status") or "").strip().lower()
    has_pick = bool(str(record.get("pick") or "").strip()) or status == "eligible"
    has_research = _reasoning_has_research(record.get("grounded_research") or {})
    evaluated_status = status in {
        "no_pick",
        "invalid_pick",
        "missing_market_prob",
        "missing_model_prob",
    }

    if has_pick and has_research:
        quality = 5
    elif has_research:
        quality = 4
    elif has_pick:
        quality = 3
    elif evaluated_status:
        quality = 2
    else:
        quality = 1

    return quality, str(record.get("timestamp") or "")


def _dedupe_tracker_reasoning_records(records: list[dict]) -> list[dict]:
    """Keep one durable tracker conclusion per fight, preserving older research.

    The live loop can append later low-information rows such as outside-window
    or event-started skips. Those should not hide the earlier pick/research row
    that explains why the tracker made or skipped a bet.
    """
    selected: dict[object, dict] = {}
    for record in records:
        if str(record.get("trader", "") or "").strip().upper() != "G":
            continue
        if str(record.get("type", "decision") or "decision").strip().lower() == "outcome":
            continue
        key = _tracker_reasoning_key(record)
        current = selected.get(key)
        if (
            current is None
            or _tracker_reasoning_record_priority(record)
            > _tracker_reasoning_record_priority(current)
        ):
            selected[key] = record
    return list(selected.values())


_REASONING_SC_TRADERS = {"S", "C"}


def _reasoning_sc_trader_from_row(row: dict) -> str:
    raw = str(row.get("trader") or "").strip().upper()
    if raw:
        return raw.split(":", 1)[0]
    return _trader_label_from_path(
        str(row.get("_ledger_path") or row.get("ledger_path") or "")
    )


def _reasoning_fight_key(
    row: dict,
    *,
    market_event_date_hints=None,
    card_date_hints=None,
):
    fighter_a = str(row.get("fighter_a") or row.get("fighter") or "")
    fighter_b = str(row.get("fighter_b") or row.get("opponent") or "")
    if not fighter_a or not fighter_b:
        return None

    event_date = str(row.get("event_date") or row.get("market_event_date") or "")
    resolved_market_event_date = _resolve_market_event_date_hint(
        fighter_a=fighter_a,
        fighter_b=fighter_b,
        event_date=event_date,
        market_event_date=str(row.get("market_event_date") or ""),
        hints=market_event_date_hints or {},
    )
    card_date = _resolve_card_date_hint(
        fighter_a=fighter_a,
        fighter_b=fighter_b,
        event_date=event_date,
        market_event_date=resolved_market_event_date
        or str(row.get("market_event_date") or ""),
        card_date=_row_card_date(row),
        hints=card_date_hints or {},
    )
    key = _fight_matrix_key(
        fighter_a,
        fighter_b,
        resolved_market_event_date or event_date,
        card_date=card_date,
    )
    pair_key, event_group_date = key
    if len(pair_key) < 2 or not event_group_date:
        return None
    return key


def _reasoning_sc_candidate_keys(
    operator_decisions: list[dict],
    ledger_bets: list[dict],
    tracker_records: list[dict],
) -> tuple[set[tuple], dict, dict]:
    market_event_date_hints = _build_market_event_date_hints(
        operator_decisions,
        tracker_records,
        ledger_bets,
    )
    card_date_hints = _build_card_date_hints(
        operator_decisions,
        tracker_records,
        ledger_bets,
    )
    keys = set()

    for decision in operator_decisions:
        contexts = _decision_context_aliases(decision.get("decision_context"))
        if contexts and not (_REASONING_SC_TRADERS & set(contexts)):
            continue
        key = _reasoning_fight_key(
            decision,
            market_event_date_hints=market_event_date_hints,
            card_date_hints=card_date_hints,
        )
        if key:
            keys.add(key)

    for bet in ledger_bets:
        if _reasoning_sc_trader_from_row(bet) not in _REASONING_SC_TRADERS:
            continue
        key = _reasoning_fight_key(
            bet,
            market_event_date_hints=market_event_date_hints,
            card_date_hints=card_date_hints,
        )
        if key:
            keys.add(key)

    return keys, market_event_date_hints, card_date_hints


def _tracker_record_has_sc_candidate(
    record: dict,
    sc_candidate_keys: set[tuple],
    *,
    market_event_date_hints=None,
    card_date_hints=None,
) -> bool:
    key = _reasoning_fight_key(
        record,
        market_event_date_hints=market_event_date_hints,
        card_date_hints=card_date_hints,
    )
    return key in sc_candidate_keys if key else False


def _safe_research_urls(values: Iterable[object]) -> list[str]:
    """Only expose navigable HTTP(S) research links."""
    urls = []
    for value in values or []:
        url = str(value or "").strip()
        if re.match(r"^https?://", url, flags=re.IGNORECASE):
            urls.append(url)
    return urls


def _reasoning_feed_sort_key(entry: dict) -> str:
    return str(entry.get("timestamp") or "")


@app.route("/api/gemini-reasoning")
def api_gemini_reasoning():
    """Gemini reasoning for S/C operator candidates plus matching tracker research."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error

    source_filter = str(request.args.get("source", "all") or "all").strip().lower()
    if source_filter not in {"all", "operator", "tracker"}:
        source_filter = "all"
    try:
        raw_limit = request.args.get("limit")
        limit = None if raw_limit in (None, "") else min(max(1, int(raw_limit)), 1000)
    except (TypeError, ValueError):
        limit = None

    try:
        from src.strategy.llm_operator import (
            load_decision_log,
            load_tracker_decision_log,
        )

        entries: list[dict] = []
        operator_decisions = load_decision_log()
        tracker_records = load_tracker_decision_log()
        try:
            ledger_bets = list(getattr(load_all_trader_ledgers(), "bets", []) or [])
        except Exception as exc:
            logger.warning("Failed to load S/C ledger context for reasoning feed: %s", exc)
            ledger_bets = []
        (
            sc_candidate_keys,
            market_event_date_hints,
            card_date_hints,
        ) = _reasoning_sc_candidate_keys(
            operator_decisions,
            ledger_bets,
            tracker_records,
        )

        if source_filter in {"all", "operator"}:
            for d in operator_decisions:
                entries.append(_normalize_operator_reasoning_entry(d))

        if source_filter in {"all", "tracker"}:
            for r in _dedupe_tracker_reasoning_records(tracker_records):
                # M/G flat trackers can evaluate every fight; only show G rows
                # when the same fight has S/C operator or ledger context.
                if not _tracker_record_has_sc_candidate(
                    r,
                    sc_candidate_keys,
                    market_event_date_hints=market_event_date_hints,
                    card_date_hints=card_date_hints,
                ):
                    continue
                entries.append(_normalize_tracker_reasoning_entry(r))

        entries.sort(key=_reasoning_feed_sort_key, reverse=True)
        total_count = len(entries)
        operator_count = sum(1 for e in entries if e.get("source") == "operator")
        tracker_count = sum(1 for e in entries if e.get("source") == "tracker")
        research_count = sum(1 for e in entries if e.get("has_research"))
        if limit is not None:
            entries = entries[:limit]

        return _dashboard_read_response(
            {
                "entries": _sanitize_for_json(entries),
                "count": len(entries),
                "total_count": total_count,
                "operator_count": operator_count,
                "tracker_count": tracker_count,
                "research_count": research_count,
            }
        )
    except Exception as e:
        logger.exception("Failed to load Gemini reasoning feed")
        return _api_internal_error(
            "gemini_reasoning_unavailable",
            "Gemini reasoning data could not be loaded.",
        )


@app.route("/api/predictions-detail")
def api_predictions_detail():
    """Return enriched prediction data with SHAP values and feature highlights."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    try:
        return jsonify(_load_prediction_payload(include_global_feature_importance=True))
    except Exception as e:
        logger.error(f"Failed to load predictions detail: {e}")
        return jsonify(_empty_prediction_payload(
            include_global_feature_importance=True,
            cache_status="error",
        ))


PREDICTION_CACHE_STALE_AFTER_MINUTES = 180


def _empty_prediction_payload(*, include_global_feature_importance: bool, cache_status: str) -> dict:
    payload = {
        "timestamp": None,
        "data_timestamp": None,
        "timestamp_parse_failed": False,
        "prediction_count": 0,
        "freshness_age_minutes": None,
        "stale_after_minutes": PREDICTION_CACHE_STALE_AFTER_MINUTES,
        "is_stale": cache_status != "current",
        "cache_status": cache_status,
        "cache_available": False,
        "predictions": [],
    }
    if include_global_feature_importance:
        payload["global_feature_importance"] = []
    return payload


def _parse_prediction_timestamp(raw_value):
    if not raw_value:
        return None

    raw = str(raw_value).strip()
    if not raw:
        return None

    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _prediction_confidence_tier(confidence: float) -> str:
    if confidence >= 0.68:
        return "strong_lean"
    if confidence >= 0.57:
        return "lean"
    return "toss_up"


def _prediction_value_status(edge: float, minimum_edge: float) -> str:
    return "potential_value" if edge >= minimum_edge else "pass"


def _coerce_prediction_float(value, default=None):
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return default if parsed != parsed else parsed


def _coerce_prediction_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _prediction_market_disagreement_note(
    *,
    fighter_name: str | None,
    model_prob: float,
    market_prob: float,
    market_gap: float,
) -> str:
    fighter = fighter_name or "the model pick"
    gap_points = market_gap * 100.0
    if model_prob >= market_prob:
        return (
            f"The model is {gap_points:.1f} percentage points higher on "
            f"{fighter} than the market is."
        )
    return (
        f"The market is {gap_points:.1f} percentage points higher on "
        f"{fighter} than the model is."
    )


def _prediction_execution_status(
    *,
    event_date,
    blended_prob: float,
    market_prob: float,
    edge: float,
    fighter_name: str,
    no_odds_prob,
    bet_side: str,
    a_num_fights,
    b_num_fights,
    line_movement,
    line_is_sharp,
    line_steam_move,
    minimum_edge: float,
    prediction_is_stale: bool,
) -> dict:
    from src.strategy.value import _filter_rejection_reason

    if edge < minimum_edge:
        shortfall = minimum_edge - edge
        return {
            "status": "pass",
            "reason": "Edge below minimum",
            "detail": f"Edge {edge:.1%} needs +{shortfall:.1%} more to clear {minimum_edge:.1%}",
        }

    if prediction_is_stale:
        return {"status": "stale", "reason": None, "detail": None}

    window = bet_window_status(event_date)
    if window is not None and not window["open"]:
        return {
            "status": "pass",
            "reason": window["reason"],
            "detail": window["detail"],
        }

    rejection = _filter_rejection_reason(
        blended_prob,
        market_prob,
        edge,
        fighter_name,
        no_odds_prob,
        line_movement=line_movement,
        line_is_sharp=line_is_sharp,
        line_steam_move=line_steam_move,
        bet_side=bet_side,
        a_num_fights=a_num_fights,
        b_num_fights=b_num_fights,
    )
    if rejection is None:
        return {"status": "bettable_now", "reason": None, "detail": None}
    return {"status": "pass", "reason": rejection["reason"], "detail": rejection["detail"]}


_SC_TRADE_CANDIDATE_STATUSES = {"bet", "eligible", "blocked"}


def _prediction_trade_candidate_window_open(pred: dict) -> bool:
    if pred.get("prediction_is_stale"):
        return False
    window = bet_window_status(
        pred.get("event_date")
        or pred.get("market_event_date")
        or pred.get("commence_time")
    )
    return window is None or bool(window.get("open"))


def _prediction_trade_candidate_summary(cells: dict[str, dict]) -> dict:
    active_cells = {
        trader: cell
        for trader, cell in cells.items()
        if str(cell.get("status") or "").strip().lower() in _SC_TRADE_CANDIDATE_STATUSES
    }
    if not active_cells:
        return {
            "trade_candidate_active": False,
            "trade_candidate_status": None,
            "trade_candidate_label": None,
            "trade_candidate_traders": [],
            "trade_candidate_cells": cells,
        }

    statuses = {
        str(cell.get("status") or "").strip().lower()
        for cell in active_cells.values()
    }
    if "bet" in statuses:
        status = "already_bet"
        label = "Already bet"
    elif "blocked" in statuses:
        status = "operator_blocked"
        label = "Operator blocked"
    else:
        status = "operator_eligible"
        label = "Qualified"

    return {
        "trade_candidate_active": True,
        "trade_candidate_status": status,
        "trade_candidate_label": label,
        "trade_candidate_traders": sorted(active_cells),
        "trade_candidate_cells": cells,
    }


def _build_prediction_trade_candidate_index(rows: list[dict]) -> dict[tuple, dict]:
    if not rows:
        return {}

    try:
        from src.strategy.llm_operator import load_decision_log

        operator_decisions = load_decision_log()
        operator_decisions.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
    except Exception as e:
        logger.debug("Failed to load operator decisions for prediction candidates: %s", e)
        operator_decisions = []

    try:
        ledger_view = load_all_trader_ledgers()
        ledger_bets = list(getattr(ledger_view, "bets", []) or [])
    except Exception as e:
        logger.debug("Failed to load ledgers for prediction candidates: %s", e)
        ledger_bets = []

    if not operator_decisions and not ledger_bets:
        return {}

    market_event_date_hints = _build_market_event_date_hints(
        rows,
        operator_decisions,
        ledger_bets,
    )
    card_date_hints = _build_card_date_hints(
        rows,
        operator_decisions,
        ledger_bets,
    )
    decisions_index = _build_decisions_index(
        operator_decisions,
        market_event_date_hints=market_event_date_hints,
        card_date_hints=card_date_hints,
    )
    pass_index = _build_operator_pass_index(
        operator_decisions,
        market_event_date_hints=market_event_date_hints,
        card_date_hints=card_date_hints,
    )
    block_index = _build_operator_block_index(
        operator_decisions,
        market_event_date_hints=market_event_date_hints,
        card_date_hints=card_date_hints,
    )
    ledger_index = _build_trader_bet_index(
        ledger_bets,
        market_event_date_hints=market_event_date_hints,
        card_date_hints=card_date_hints,
    )

    result = {}
    for row in rows:
        if not _prediction_trade_candidate_window_open(row):
            continue
        fighter_a = str(row.get("fighter_a", "") or "")
        fighter_b = str(row.get("fighter_b", "") or "")
        event_date = str(row.get("market_event_date") or row.get("event_date") or "")
        card_date = _row_card_date(row)
        key = _fight_matrix_key(fighter_a, fighter_b, event_date, card_date=card_date)
        cells = {
            trader: _format_sc_matrix_cell(
                trader=trader,
                ledger_bet=ledger_index.get((trader, *key)),
                operator_decision=_lookup_operator_matrix_decision(pass_index, key, trader),
                block_decision=_lookup_operator_matrix_decision(block_index, key, trader),
                decisions_index=decisions_index,
                prediction_row=row,
            )
            for trader in ("S", "C")
        }
        summary = _prediction_trade_candidate_summary(cells)
        if summary["trade_candidate_active"]:
            result[key] = summary
    return result


def _prediction_cache_metadata(raw_timestamp) -> dict:
    parsed_timestamp = _parse_prediction_timestamp(raw_timestamp)
    freshness_age_minutes = None
    is_stale = True
    timestamp_parse_failed = bool(raw_timestamp) and parsed_timestamp is None

    if parsed_timestamp is not None:
        now = datetime.now(parsed_timestamp.tzinfo) if parsed_timestamp.tzinfo else datetime.now()
        freshness_age_minutes = max(0.0, (now - parsed_timestamp).total_seconds() / 60.0)
        is_stale = freshness_age_minutes >= PREDICTION_CACHE_STALE_AFTER_MINUTES

    return {
        "freshness_age_minutes": round(freshness_age_minutes, 1) if freshness_age_minutes is not None else None,
        "stale_after_minutes": PREDICTION_CACHE_STALE_AFTER_MINUTES,
        "is_stale": is_stale,
        "cache_status": "stale" if is_stale else "current",
        "cache_available": True,
        "timestamp_parse_failed": timestamp_parse_failed,
    }


def _prediction_payload_dedupe_key(pred: dict):
    event_date = (
        pred.get("market_event_date")
        or pred.get("event_date")
        or pred.get("commence_time")
        or ""
    )
    card_date = _row_card_date(pred)
    return _fight_matrix_key(
        str(pred.get("fighter_a", "") or ""),
        str(pred.get("fighter_b", "") or ""),
        str(event_date or ""),
        card_date=card_date,
    )


def _prediction_payload_quality(pred: dict) -> tuple:
    a_fights = _coerce_prediction_int(pred.get("a_num_fights")) or 0
    b_fights = _coerce_prediction_int(pred.get("b_num_fights")) or 0
    has_market = pred.get("a_market_prob") is not None or pred.get("b_market_prob") is not None
    return (
        0 if pred.get("low_experience") else 1,
        a_fights + b_fights,
        1 if has_market else 0,
        str(pred.get("prediction_generated_at") or ""),
    )


def _load_prediction_payload(*, include_global_feature_importance: bool) -> dict:
    cache_path = LOGS_DIR / "predictions_cache.json"
    if not cache_path.exists():
        return _empty_prediction_payload(
            include_global_feature_importance=include_global_feature_importance,
            cache_status="missing",
        )

    data = json.loads(cache_path.read_text())
    from src.config import MIN_EDGE_THRESHOLD
    from src.strategy.value import compute_independent_blend_probs, dynamic_blend_weight

    metadata = _prediction_cache_metadata(data.get("timestamp"))
    prediction_is_stale = metadata["is_stale"]
    enriched_predictions_by_key = {}
    for raw_pred in data.get("predictions", []):
        pred = dict(raw_pred)
        pred["fighter_a"] = canonical_fighter_display_name(pred.get("fighter_a"))
        pred["fighter_b"] = canonical_fighter_display_name(pred.get("fighter_b"))
        model_a = _coerce_prediction_float(pred.get("prob_a"), 0.5)
        model_b = _coerce_prediction_float(pred.get("prob_b"), 1.0 - model_a)
        market_a = _coerce_prediction_float(pred.get("a_market_prob"), 0.5)
        market_b = _coerce_prediction_float(pred.get("b_market_prob"), 1.0 - market_a)
        no_odds_a = _coerce_prediction_float(pred.get("no_odds_prob_a"))
        no_odds_b = _coerce_prediction_float(pred.get("no_odds_prob_b"))
        a_num_fights = _coerce_prediction_int(pred.get("a_num_fights"))
        b_num_fights = _coerce_prediction_int(pred.get("b_num_fights"))
        line_movement = _coerce_prediction_float(pred.get("line_movement"))
        line_is_sharp = _coerce_prediction_int(pred.get("line_is_sharp"))
        line_steam_move = _coerce_prediction_int(pred.get("line_steam_move"))

        weight_a = dynamic_blend_weight(model_a, market_a, no_odds_a)
        weight_b = dynamic_blend_weight(model_b, market_b, no_odds_b)
        blend_a, blend_b = compute_independent_blend_probs(
            model_a,
            market_a,
            no_odds_a,
            model_b,
            market_b,
            no_odds_b,
        )
        edge_a = blend_a - market_a
        edge_b = blend_b - market_b

        predicted_side = "a" if model_a >= model_b else "b"
        predicted_winner = pred.get("fighter_a") if predicted_side == "a" else pred.get("fighter_b")
        predicted_prob = model_a if predicted_side == "a" else model_b
        predicted_market_prob = market_a if predicted_side == "a" else market_b
        predicted_blended_prob = blend_a if predicted_side == "a" else blend_b
        predicted_edge = edge_a if predicted_side == "a" else edge_b

        value_side = "a" if edge_a >= edge_b else "b"
        value_fighter = pred.get("fighter_a") if value_side == "a" else pred.get("fighter_b")
        best_edge = edge_a if value_side == "a" else edge_b
        pick_value_status = _prediction_value_status(predicted_edge, MIN_EDGE_THRESHOLD)
        value_status = _prediction_value_status(best_edge, MIN_EDGE_THRESHOLD)
        pick_execution_status = _prediction_execution_status(
            event_date=pred.get("event_date") or pred.get("market_event_date") or pred.get("commence_time"),
            blended_prob=blend_a if predicted_side == "a" else blend_b,
            market_prob=market_a if predicted_side == "a" else market_b,
            edge=predicted_edge,
            fighter_name=predicted_winner or "Unknown",
            no_odds_prob=no_odds_a if predicted_side == "a" else no_odds_b,
            bet_side=predicted_side,
            a_num_fights=a_num_fights,
            b_num_fights=b_num_fights,
            line_movement=line_movement,
            line_is_sharp=line_is_sharp,
            line_steam_move=line_steam_move,
            minimum_edge=MIN_EDGE_THRESHOLD,
            prediction_is_stale=prediction_is_stale,
        )
        value_execution_status = _prediction_execution_status(
            event_date=pred.get("event_date") or pred.get("market_event_date") or pred.get("commence_time"),
            blended_prob=blend_a if value_side == "a" else blend_b,
            market_prob=market_a if value_side == "a" else market_b,
            edge=best_edge,
            fighter_name=value_fighter or "Unknown",
            no_odds_prob=no_odds_a if value_side == "a" else no_odds_b,
            bet_side=value_side,
            a_num_fights=a_num_fights,
            b_num_fights=b_num_fights,
            line_movement=line_movement,
            line_is_sharp=line_is_sharp,
            line_steam_move=line_steam_move,
            minimum_edge=MIN_EDGE_THRESHOLD,
            prediction_is_stale=prediction_is_stale,
        )

        market_pick = pred.get("fighter_a") if market_a >= market_b else pred.get("fighter_b")
        no_odds_pick = None
        if no_odds_a is not None and no_odds_b is not None:
            no_odds_pick = pred.get("fighter_a") if no_odds_a >= no_odds_b else pred.get("fighter_b")

        market_gap = abs(predicted_prob - predicted_market_prob)
        market_disagreement = market_gap >= 0.08
        confidence = float(pred["confidence"]) if pred.get("confidence") is not None else max(model_a, model_b)

        pred.update({
            "predicted_side": predicted_side,
            "predicted_winner": predicted_winner,
            "predicted_prob": round(predicted_prob, 4),
            "predicted_market_prob": round(predicted_market_prob, 4),
            "predicted_blended_prob": round(predicted_blended_prob, 4),
            "predicted_edge": round(predicted_edge, 4),
            "market_pick": market_pick,
            "no_odds_pick": no_odds_pick,
            "value_side": value_side,
            "value_fighter": value_fighter,
            "best_edge": round(best_edge, 4),
            "blended_prob_a": round(blend_a, 4),
            "blended_prob_b": round(blend_b, 4),
            "edge_a": round(edge_a, 4),
            "edge_b": round(edge_b, 4),
            "blend_weight": round(weight_a if predicted_side == "a" else weight_b, 3),
            "market_gap": round(market_gap, 4),
            "market_disagreement": market_disagreement,
            "market_disagreement_note": (
                _prediction_market_disagreement_note(
                    fighter_name=predicted_winner,
                    model_prob=predicted_prob,
                    market_prob=predicted_market_prob,
                    market_gap=market_gap,
                )
                if market_disagreement
                else None
            ),
            "confidence_tier": _prediction_confidence_tier(confidence),
            "prediction_is_stale": prediction_is_stale,
            "prediction_cache_status": metadata["cache_status"],
            "pick_value_status": pick_value_status,
            "pick_has_positive_edge": pick_value_status == "potential_value",
            "pick_execution_status": pick_execution_status["status"],
            "pick_is_bettable": pick_execution_status["status"] == "bettable_now",
            "pick_filter_reason": pick_execution_status.get("reason"),
            "pick_filter_detail": pick_execution_status.get("detail"),
            "value_status": value_status,
            "value_has_positive_edge": value_status == "potential_value",
            "value_execution_status": value_execution_status["status"],
            "value_is_bettable": value_execution_status["status"] == "bettable_now",
            "value_filter_reason": value_execution_status.get("reason"),
            "value_filter_detail": value_execution_status.get("detail"),
            "experience_flag": "low_sample" if pred.get("low_experience") else "normal",
        })
        dedupe_key = _prediction_payload_dedupe_key(pred)
        existing = enriched_predictions_by_key.get(dedupe_key)
        if existing is None or _prediction_payload_quality(pred) > _prediction_payload_quality(existing):
            enriched_predictions_by_key[dedupe_key] = pred

    enriched_predictions = list(enriched_predictions_by_key.values())
    trade_candidate_index = _build_prediction_trade_candidate_index(enriched_predictions)
    for pred in enriched_predictions:
        card_date = _row_card_date(pred)
        key = _fight_matrix_key(
            str(pred.get("fighter_a", "") or ""),
            str(pred.get("fighter_b", "") or ""),
            str(pred.get("market_event_date") or pred.get("event_date") or ""),
            card_date=card_date,
        )
        candidate = trade_candidate_index.get(key)
        if candidate is None:
            candidate = {
                "trade_candidate_active": False,
                "trade_candidate_status": None,
                "trade_candidate_label": None,
                "trade_candidate_traders": [],
                "trade_candidate_cells": {},
            }
        pred.update(candidate)

    payload = {
        "timestamp": data.get("timestamp"),
        "data_timestamp": data.get("timestamp"),
        "prediction_count": len(enriched_predictions),
        "predictions": enriched_predictions,
    }
    payload.update(metadata)
    if include_global_feature_importance:
        payload["global_feature_importance"] = data.get("global_feature_importance", [])
    return _sanitize_for_json(payload)


def _recover_ledger_from_clob(clob_client):
    """One-time recovery: rebuild ledger entries from live CLOB orders + market data.

    Only runs if all known trader ledgers are empty but CLOB has open orders.
    """
    from src.strategy.duo_trader import CONVICTION_LEDGER, get_all_trader_ledgers
    from datetime import datetime, timezone

    # Check if ledgers already have data
    for _, path in get_all_trader_ledgers():
        if path.exists():
            ledger = BetLedger(path=path)
            if ledger.bets:
                return  # ledger has data, nothing to recover

    if not clob_client:
        return

    try:
        clob_orders = clob_client.get_open_orders()
    except Exception as e:
        logger.warning(f"Ledger recovery: failed to fetch CLOB orders: {e}")
        return

    if not clob_orders:
        return

    token_map = _build_token_to_fighter_map()
    if not token_map:
        logger.warning("Ledger recovery: no market data available")
        return

    # Recover into conviction ledger (C) since we can't determine original trader
    ledger = BetLedger(path=CONVICTION_LEDGER)
    recovered = 0

    for order in clob_orders:
        asset_id = order.get("asset_id", "")
        info = token_map.get(asset_id)
        if not info:
            continue

        price = float(order.get("price", 0))
        original_size = float(order.get("original_size", order.get("size", 0)))
        size_matched = float(order.get("size_matched", 0))
        total_cost = size_matched * price if size_matched else original_size * price
        shares = size_matched if size_matched else original_size

        # Convert CLOB timestamp
        clob_ts = order.get("created_at") or order.get("timestamp")
        if isinstance(clob_ts, (int, float)):
            placed_at = datetime.fromtimestamp(clob_ts, tz=timezone.utc).isoformat()
        else:
            placed_at = clob_ts or datetime.now(timezone.utc).isoformat()

        ledger.add_bet(
            fighter=info["fighter"],
            opponent=info["opponent"],
            side=info.get("side", "a"),
            amount=round(total_cost, 2),
            price=price,
            shares=round(shares, 2),
            token_id=asset_id,
            market_id="",
            condition_id=info.get("condition_id", ""),
            model_prob=0.0,
            market_prob=price,
            edge=0.0,
            decimal_odds=round(1.0 / price, 4) if price > 0 else 0,
            dry_run=False,
            event_date=info.get("event_date", ""),
            order_type="limit_bid",
            order_id=order.get("id", ""),
        )
        recovered += 1

    if recovered:
        logger.info(f"Ledger recovery: rebuilt {recovered} bets from CLOB orders")


def set_clob_client(clob_client):
    """Hot-set the CLOB client after startup (called from background init thread)."""
    global _clob_client, _position_monitor
    _clob_client = clob_client
    _position_monitor = PositionMonitor(clob_client=clob_client)

    # Run ledger recovery now that CLOB is available
    try:
        _recover_ledger_from_clob(clob_client)
    except Exception as e:
        logger.warning(f"Ledger recovery failed (non-fatal): {e}")


def get_clob_client():
    """Return the process-wide CLOB client if startup initialization has completed."""
    return _clob_client


def _prediction_refresh_loop(interval_seconds: int) -> None:
    """Background loop: run predictions and (when armed) execute live trades."""
    import argparse

    from src.config import MIN_EDGE_THRESHOLD
    from src.live_control import resolve_live_mode_from_env, LIVE_MODE_REAL

    live_mode = resolve_live_mode_from_env()
    is_live = live_mode == LIVE_MODE_REAL

    args = argparse.Namespace(
        dry_run=not is_live,
        real=is_live,
        model="xgboost",
        min_edge=MIN_EDGE_THRESHOLD,
    )

    while True:
        cache_path = LOGS_DIR / "predictions_cache.json"
        needs_refresh = True
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text())
                meta = _prediction_cache_metadata(data.get("timestamp"))
                needs_refresh = meta["is_stale"] or meta["cache_status"] != "current"
            except Exception:
                needs_refresh = True

        if needs_refresh:
            logger.info("Background prediction refresh starting...")
            try:
                from src.bot import cmd_duo_live

                result = cmd_duo_live(args)
                logger.info(f"Background prediction refresh complete: {result}")
            except Exception as e:
                logger.error(f"Background prediction refresh failed: {e}")
        else:
            logger.debug("Prediction cache is fresh, skipping refresh")

        time.sleep(interval_seconds)


def start_server(
    port: int = 5050,
    debug: bool = False,
    clob_client=None,
    host: str = "127.0.0.1",
):
    """Start the Flask web dashboard."""
    global _clob_client, _position_monitor, _server_host
    _server_host = host
    status = get_runtime_status()
    status["host"] = host
    status["public_bind"] = _dashboard_is_public_bind()
    set_runtime_status(status)
    if clob_client:
        _clob_client = clob_client
        _position_monitor = PositionMonitor(clob_client=clob_client)
    else:
        _position_monitor = PositionMonitor(clob_client=None)

    # Start background prediction refresh thread only when the betting loop
    # is NOT already active.  Both loops call cmd_duo_live which triggers the
    # full LLM veto sweep.  Running them concurrently wastes Gemini
    # credits by evaluating every matchup twice (or more) per cycle.
    betting_loop_active = status.get("trading_enabled", False)
    if not betting_loop_active:
        refresh_interval = PREDICTION_CACHE_STALE_AFTER_MINUTES * 60
        refresh_thread = threading.Thread(
            target=_prediction_refresh_loop,
            args=(refresh_interval,),
            daemon=True,
            name="prediction-refresh",
        )
        refresh_thread.start()
        register_runtime_thread("prediction_refresh", refresh_thread)
        logger.info(
            f"Background prediction refresh enabled (every {PREDICTION_CACHE_STALE_AFTER_MINUTES} min)"
        )
    else:
        logger.info(
            "Background prediction refresh skipped — betting loop already handles prediction cache refresh"
        )

    logger.info(f"Starting web dashboard at http://{host}:{port}")
    print(f"\n  Dashboard running at: http://{host}:{port}\n")
    app.run(host=host, port=port, debug=debug)
