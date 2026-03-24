"""
Web dashboard — local Flask app for live bet & P&L tracking.

Run:
    python -m src.bot web
    python -m src.bot web --port 8080
"""

import copy
import json
import hmac
import logging
import math
import os
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, make_response, render_template, request

from src.config import LOGS_DIR
from src.polymarket.tracker import (
    BetLedger,
    _load_pnl_history,
    auto_redeem_positions_from_polymarket,
    auto_settle_from_polymarket,
    load_all_trader_ledgers,
    resolve_merged_bet_reference,
)
from src.polymarket.monitor import PositionMonitor

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

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
_cache_lock = threading.Lock()
SLOW_ENDPOINT_TTL = 300  # 5 minutes


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
    """Return cached result if fresh, otherwise recompute."""
    with _cache_lock:
        entry = _endpoint_cache.get(key)
        if entry and time.time() - entry["ts"] < ttl:
            return entry["data"]
    data = compute_fn()
    with _cache_lock:
        _endpoint_cache[key] = {"data": data, "ts": time.time()}
    return data


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


_TENNIS_ACTIVITY_PATTERNS = (
    "tennis",
    "atp",
    "wta",
    "surface_elo",
    "tennis-live",
    "tennis_veto",
    "tennis_decision",
    "tennis_automation",
    "tennis_market",
)


def _classify_activity_sport(entry: dict) -> str:
    """Classify an activity entry as UFC, tennis, or general for server-side filtering."""
    source = str(entry.get("source", "") or "").lower()
    message = str(entry.get("message", "") or "").lower()

    # Explicit tennis sources (e.g. src.model.tennis_model)
    if "tennis" in source:
        return "tennis"

    # werkzeug HTTP access logs are infrastructure — not sport-specific
    if source == "werkzeug":
        return "general"

    # Tennis keywords in message from ANY source, not just src.bot
    if any(pattern in message for pattern in _TENNIS_ACTIVITY_PATTERNS):
        return "tennis"

    return "ufc"


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
                filtered_entries = (
                    [entry for entry in entries if entry.get("sport") == sport]
                    if sport in {"ufc", "tennis"}
                    else entries
                )
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


def _require_read_auth():
    """Read-only dashboard endpoints remain public; mutations stay token-gated."""
    return None


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


@app.route("/tennis")
def tennis_page():
    return _html_no_store("dashboard.html")


_TENNIS_MARKET_KEYWORDS = (
    "tennis", "atp", "wta", "grand slam", "roland garros",
    "wimbledon", "us open tennis", "australian open tennis",
    "french open", "indian wells", "miami open",
)


def _classify_sport_from_market(market_title: str) -> str:
    """Classify a Polymarket position as 'ufc' or 'tennis' based on market title."""
    title_lower = (market_title or "").lower()
    for kw in _TENNIS_MARKET_KEYWORDS:
        if kw in title_lower:
            return "tennis"
    # Check for common tennis match patterns: "Player vs Player" with no UFC indicators
    # Default to ufc for ambiguous cases
    return "ufc"


def _classify_sport_from_ledger_path(ledger_path: str) -> str:
    """Classify a bet as 'ufc' or 'tennis' based on which ledger file it came from."""
    if "tennis" in str(ledger_path or "").lower():
        return "tennis"
    return "ufc"


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


def _upcoming_event_day_key(raw_value) -> str | None:
    parsed = _parse_upcoming_event_datetime(raw_value)
    if parsed is None:
        return None
    return parsed.date().isoformat()


@app.route("/api/summary")
def api_summary():
    """Return summary stats — merges ledger stats with live Polymarket data."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    ledger = load_all_trader_ledgers()
    summary = ledger.get_summary()

    # Overlay live Polymarket position data so dashboard matches Polymarket
    global _position_monitor
    with _monitor_lock:
        if not _position_monitor:
            _position_monitor = PositionMonitor(clob_client=_clob_client)
        monitor = _position_monitor

    try:
        live = monitor.compute_pnl()
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
    except Exception as e:
        logger.warning("Live PnL merge failed — dashboard may show stale data: %s", e)
        summary["_pnl_degraded"] = True

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

    from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER
    updated = 0
    skipped = 0
    for path in [SINGLE_LEDGER, CONVICTION_LEDGER]:
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
    global _position_monitor
    with _monitor_lock:
        if not _position_monitor:
            _position_monitor = PositionMonitor(clob_client=_clob_client)
        monitor = _position_monitor

    pnl = monitor.compute_pnl()
    # Tag each position with its sport
    for pos in pnl.get("positions", []):
        pos["sport"] = _classify_sport_from_market(pos.get("market", ""))
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

    trades = monitor.get_trades(limit=100)
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

    limit = min(int(request.args.get("limit", 200)), 500)
    closed = monitor.get_closed_positions(limit=limit)
    return jsonify(closed)


@app.route("/api/settle-auto", methods=["POST"])
def api_settle_auto():
    """Auto-settle resolved markets across all trader ledgers."""
    auth_error = _require_mutation_auth()
    if auth_error is not None:
        return auth_error
    from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER
    count = 0
    for path in [SINGLE_LEDGER, CONVICTION_LEDGER]:
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
    return jsonify(_cached("balance", 60, _compute_balance))


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
        try:
            clob = ClobClientWrapper()
        except Exception as e:
            return {
                "available": False,
                "blocked": None,
                "ip": "",
                "country": "",
                "region": "",
                "status_code": None,
                "error": str(e),
                "proxy_configured": bool(proxy_url),
                "proxy_enabled": bool(polymarket_client_module._proxy_patched),
                "proxy_target": proxy_target,
            }

    status = clob.get_geoblock_status()
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
    return _json_no_store(_compute_geoblock_status())


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
    sport = str(request.args.get("sport", "all") or "all").strip().lower()
    if sport not in {"all", "ufc", "tennis"}:
        sport = "all"
    log_path = LOGS_DIR / "bot.log"
    entries = _read_recent_log_entries(log_path, limit=limit, sport=sport)
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
    sport = str(request.args.get("sport", "all") or "all").strip().lower()
    if sport not in {"all", "ufc", "tennis"}:
        sport = "all"
    log_path = LOGS_DIR / "bot.log"
    entries = _read_recent_log_entries(log_path, limit=limit, sport=sport)
    snapshot = _bot_activity_snapshot(log_path, entries)
    return _json_no_store(snapshot, extra_headers=_bot_activity_headers(log_path, entries))


@app.route("/api/significant-actions")
def api_significant_actions():
    """Return filtered high-value bot actions from bot.log."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
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
                        "message": m.group(4),
                    })
                elif log_entries and line.strip():
                    # Continuation line — append to previous entry's message
                    log_entries[-1]["message"] += " " + line.strip()

            for entry in log_entries:
                msg = entry["message"]
                for pattern, tag, color in sig_patterns:
                    if pattern.search(msg):
                        entries.append({
                            "timestamp": entry["timestamp"],
                            "level": entry["level"],
                            "tag": tag,
                            "color": color,
                            "message": msg.strip(),
                        })
                        break
        except Exception:
            logger.warning("Failed to parse significant actions from %s", log_path, exc_info=True)

    return jsonify(entries[-30:])


def _snapshot_upcoming_events() -> list[dict]:
    from src.config import RAW_DATA_DIR

    snapshot_dir = RAW_DATA_DIR / "snapshots"
    if not snapshot_dir.exists():
        return []

    # Each snapshot is a per-event file: { event, event_date, timestamp, fights }
    # Group by event name and take the most recent snapshot per event.
    seen = {}
    for path in sorted(snapshot_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
            event_name = data.get("event", "")
            if not event_name or event_name in seen:
                continue
            fights = data.get("fights", [])
            event_date = data.get("event_date", "")

            seen[event_name] = {
                "event": event_name,
                "date": event_date,
                "fight_count": len(fights),
                "source": "snapshot",
            }
        except Exception as e:
            logger.warning("Failed to load upcoming event snapshot %s: %s", path, e)
            continue

    return list(seen.values())


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
        raw_date = prediction.get("event_date") or prediction.get("commence_time") or ""
        parsed_date = _parse_upcoming_event_datetime(raw_date)
        if parsed_date is None or parsed_date.date() < today_utc:
            continue
        if not fighter_a or not fighter_b:
            continue

        day_key = parsed_date.date().isoformat()
        entry = grouped.setdefault(
            day_key,
            {
                "event": "Live UFC odds card",
                "date": raw_date,
                "fight_pairs": set(),
                "source": "predictions_cache",
            },
        )
        current = _parse_upcoming_event_datetime(entry.get("date"))
        if current is None or parsed_date < current:
            entry["date"] = raw_date

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
        parsed = _parse_upcoming_event_datetime(event.get("date"))
        if parsed is not None:
            return (0, parsed.isoformat(), str(event.get("event", "")))
        return (1, str(event.get("date", "")), str(event.get("event", "")))

    merged.sort(key=_sort_key)
    return merged[:10]


@app.route("/api/upcoming-events")
def api_upcoming_events():
    """Return upcoming UFC events from monitoring snapshots plus live predictions."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error

    snapshot_events = _snapshot_upcoming_events()
    prediction_events = _prediction_cache_upcoming_events()
    return jsonify(_merge_upcoming_events(snapshot_events, prediction_events))


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
    from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER

    result = {}
    traders = [
        ("S", SINGLE_LEDGER),
        ("C", CONVICTION_LEDGER),
    ]

    for label, path in traders:
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
    """Check all tracked fights for injury/cancellation signals (cached)."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    return jsonify(_cached("injury-alerts", SLOW_ENDPOINT_TTL, _compute_injury_alerts))


def _compute_injury_alerts():
    from src.config import RAW_DATA_DIR

    line_dir = RAW_DATA_DIR / "line_history"
    if not line_dir.exists():
        return []

    snapshots = sorted(line_dir.glob("odds_*.csv"), reverse=True)
    if not snapshots:
        return []

    try:
        import pandas as pd
        from src.data.line_tracker import detect_injury_or_cancellation

        latest = pd.read_csv(snapshots[0])
        fights = latest.groupby(["fighter_a", "fighter_b"]).agg(
            a_prob=("a_fair_prob", "mean"),
            b_prob=("b_fair_prob", "mean"),
        ).reset_index()

        alerts = []
        for _, fight in fights.iterrows():
            a, b = fight["fighter_a"], fight["fighter_b"]
            result = detect_injury_or_cancellation(
                a, b,
                current_odds={"a_prob": fight["a_prob"], "b_prob": fight["b_prob"]},
            )
            if result["suspected"]:
                alerts.append({
                    "fighter_a": a,
                    "fighter_b": b,
                    "severity": result["severity"],
                    "reason": result["reason"],
                    "movement": result.get("details", {}).get("movement"),
                    "steam_move": result.get("details", {}).get("steam_move", False),
                })

        return alerts
    except Exception as e:
        logger.error(f"Failed to check injury alerts: {e}")
        return []


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


@app.route("/api/line-movements")
def api_line_movements():
    """Return line movement analysis for all tracked fights (cached)."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    return jsonify(_cached("line-movements", SLOW_ENDPOINT_TTL, _compute_line_movements))


def _compute_line_movements():
    from src.config import RAW_DATA_DIR

    line_dir = RAW_DATA_DIR / "line_history"
    if not line_dir.exists():
        return []

    snapshots = sorted(line_dir.glob("odds_*.csv"), reverse=True)
    if not snapshots:
        return []

    try:
        import pandas as pd
        from src.data.line_tracker import analyze_line_movement

        latest = pd.read_csv(snapshots[0])
        fights = latest.groupby(["fighter_a", "fighter_b"]).first().reset_index()

        results = []
        for _, fight in fights.iterrows():
            a, b = fight["fighter_a"], fight["fighter_b"]
            analysis = analyze_line_movement(a, b)
            analysis["fighter_a"] = a
            analysis["fighter_b"] = b
            results.append(analysis)

        results.sort(key=lambda x: x.get("abs_movement", 0), reverse=True)
        return results
    except Exception as e:
        logger.error(f"Failed to compute line movements: {e}")
        return []


@app.route("/api/trader-breakdown")
def api_trader_breakdown():
    """Return per-trader P&L breakdown from individual ledgers."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER

    breakdown = []
    traders = [
        ("S", "Single (Value)", SINGLE_LEDGER, 0.30),
        ("C", "Conviction", CONVICTION_LEDGER, None),
    ]

    for label, style, path, blend in traders:
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


@app.route("/api/open-limit-orders")
def api_open_limit_orders():
    """Return open limit orders cross-referenced with CLOB (cached 30s)."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    return jsonify(_cached("open-limit-orders", 30, _compute_open_limit_orders))


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
        and bet.get("order_type") in ("limit_bid", "limit", "near_miss_limit")
    )


def _pick_best_limit_match(candidates: list[dict]) -> dict | None:
    """Prefer real ledger rows over recovery placeholders for the same order."""
    if not candidates:
        return None

    def _is_limit_type(bet: dict) -> bool:
        return bet.get("order_type") in ("limit_bid", "limit", "near_miss_limit")

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


def _normalize_limit_status(raw_status) -> str:
    return str(raw_status or "").strip().lower().replace("-", "_").replace(" ", "_")


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


def _compute_open_limit_orders():
    from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER

    # Collect limit bids from both trader ledgers
    ledger_lookup = defaultdict(list)       # order_id -> enriched bet dicts
    token_lookup = defaultdict(list)        # token_id -> bet dicts (fallback)
    token_price_lookup = defaultdict(list)  # (token_id, price) -> bet dicts
    for label, path in [("S", SINGLE_LEDGER), ("C", CONVICTION_LEDGER)]:
        ledger = BetLedger(path=path)
        for bet in ledger.bets:
            enriched = {**bet, "trader": label}
            is_open = bet.get("status") == "open"
            is_limit = bet.get("order_type") in ("limit_bid", "limit", "near_miss_limit")

            tid = bet.get("token_id")
            if tid:
                if is_open:
                    token_lookup[tid].append(enriched)
                    price = _safe_float(bet.get("price"), None)
                    if price is not None:
                        token_price_lookup[(tid, round(price, 4))].append(enriched)

            if is_open and is_limit:
                oid = bet.get("order_id")
                if oid:
                    ledger_lookup[str(oid)].append(enriched)

    # Build market-data fallback: token_id -> fighter name (works even if ledger is empty)
    market_token_map = _build_token_to_fighter_map()

    # Fetch ground-truth open orders from CLOB
    clob_orders = []
    if _clob_client:
        try:
            clob_orders = _clob_client.get_open_orders()
        except Exception as e:
            logger.warning(f"Failed to fetch CLOB open orders: {e}")

    clob_order_ids = set()
    results = []

    # Enrich CLOB orders with ledger data, then token_lookup, then market data
    for raw_order in clob_orders:
        order = _unwrap_clob_order(raw_order)
        oid = order.get("id", "")
        asset_id = order.get("asset_id", "")
        if oid:
            clob_order_ids.add(str(oid))

        clob_price = round(_safe_float(order.get("price"), 0.0), 4)
        # Match: order_id (best) -> token+price -> token-only (worst)
        ledger_bet = _pick_best_limit_match(ledger_lookup.get(str(oid), []))
        if not ledger_bet:
            ledger_bet = _pick_best_limit_match(token_price_lookup.get((asset_id, clob_price), []))
        if not ledger_bet:
            ledger_bet = _pick_best_limit_match(token_lookup.get(asset_id, []))

        # Market data fallback for fighter/opponent/event_date
        market_info = market_token_map.get(asset_id, {})

        fighter = (ledger_bet["fighter"] if ledger_bet else None) or market_info.get("fighter")
        opponent = (ledger_bet.get("opponent") if ledger_bet else None) or market_info.get("opponent")
        event_date = (ledger_bet.get("event_date") if ledger_bet else None) or market_info.get("event_date")

        # Convert CLOB timestamp (Unix seconds) to ISO string
        raw_ts = ledger_bet.get("placed_at") if ledger_bet else None
        if not raw_ts:
            clob_ts = order.get("created_at") or order.get("timestamp")
            if isinstance(clob_ts, (int, float)):
                from datetime import datetime, timezone
                raw_ts = datetime.fromtimestamp(clob_ts, tz=timezone.utc).isoformat()
            else:
                raw_ts = clob_ts

        resolved = _resolve_limit_order_state(order_data=order, ledger_bet=ledger_bet, on_clob=True)

        results.append({
            "order_id": oid,
            "fighter": fighter,
            "opponent": opponent,
            "trader": ledger_bet["trader"] if ledger_bet else None,
            "bid_price": float(order.get("price", 0)),
            "size_remaining": resolved["size_remaining"],
            "size_matched": resolved["size_matched"],
            "edge": _display_edge_for_limit_order(ledger_bet),
            "order_type": ledger_bet.get("order_type") if ledger_bet else "limit",
            "placed_at": raw_ts,
            "event_date": event_date,
            "on_clob": True,
            "status": resolved["status"],
            "status_note": resolved["raw_status"],
        })

    # Resolve ledger limit bids not found on the open-order list.
    for oid, candidates in ledger_lookup.items():
        if oid not in clob_order_ids:
            bet = _pick_best_limit_match(candidates)
            if not bet:
                continue
            closed_order = {}
            if _clob_client and hasattr(_clob_client, "get_order"):
                try:
                    closed_order = _unwrap_clob_order(_clob_client.get_order(oid))
                except Exception as e:
                    logger.debug(f"Failed to fetch closed order {oid}: {e}")

            resolved = _resolve_limit_order_state(order_data=closed_order, ledger_bet=bet, on_clob=False)
            bid_price = _safe_float(closed_order.get("price", bet.get("price", 0.0)), 0.0)

            results.append({
                "order_id": oid,
                "fighter": bet["fighter"],
                "opponent": bet.get("opponent"),
                "trader": bet["trader"],
                "bid_price": bid_price,
                "size_remaining": resolved["size_remaining"],
                "size_matched": resolved["size_matched"],
                "edge": _display_edge_for_limit_order(bet),
                "order_type": bet.get("order_type"),
                "placed_at": bet.get("placed_at"),
                "event_date": bet.get("event_date"),
                "on_clob": False,
                "status": resolved["status"],
                "status_note": resolved["raw_status"] or "not found on open orders",
            })

    # Sort by placed_at descending (newest first)
    results.sort(key=lambda x: x.get("placed_at") or "", reverse=True)
    return results


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


@app.route("/api/operator-decisions")
def api_operator_decisions():
    """Return LLM Operator decision log entries for UFC and/or tennis."""
    auth_error = _require_read_auth()
    if auth_error is not None:
        return auth_error
    sport = str(request.args.get("sport", "all") or "all").strip().lower()
    try:
        decisions = []
        if sport in ("all", "ufc"):
            from src.strategy.llm_operator import load_decision_log
            for d in load_decision_log():
                d["sport"] = "ufc"
                decisions.append(d)
        if sport in ("all", "tennis"):
            try:
                from src.strategy.tennis_llm_operator import TENNIS_LLM_VETO_LOG_PATH
                if TENNIS_LLM_VETO_LOG_PATH.exists():
                    with open(TENNIS_LLM_VETO_LOG_PATH) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    d = json.loads(line)
                                    d["sport"] = "tennis"
                                    decisions.append(d)
                                except json.JSONDecodeError:
                                    pass
            except Exception as te:
                logger.warning("Failed to load tennis veto log: %s", te)
        # Sort by timestamp descending (most recent first)
        decisions.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
        return _json_no_store({"decisions": decisions, "count": len(decisions)})
    except Exception as e:
        logger.error(f"Failed to load operator decisions: {e}")
        return _json_no_store({"decisions": [], "count": 0, "error": str(e)})


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


def _prediction_execution_status(
    *,
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
) -> str:
    from src.strategy.value import _passes_filters

    if edge < minimum_edge:
        return "pass"

    if prediction_is_stale:
        return "stale"

    passed = _passes_filters(
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
    return "bettable_now" if passed else "pass"


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
    enriched_predictions = []
    for raw_pred in data.get("predictions", []):
        pred = dict(raw_pred)
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

        market_gap = abs(model_a - market_a)
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
            "market_disagreement": market_gap >= 0.08,
            "confidence_tier": _prediction_confidence_tier(confidence),
            "prediction_is_stale": prediction_is_stale,
            "prediction_cache_status": metadata["cache_status"],
            "pick_value_status": pick_value_status,
            "pick_has_positive_edge": pick_value_status == "potential_value",
            "pick_execution_status": pick_execution_status,
            "pick_is_bettable": pick_execution_status == "bettable_now",
            "value_status": value_status,
            "value_has_positive_edge": value_status == "potential_value",
            "value_execution_status": value_execution_status,
            "value_is_bettable": value_execution_status == "bettable_now",
            "experience_flag": "low_sample" if pred.get("low_experience") else "normal",
        })
        enriched_predictions.append(pred)

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

    Only runs if BOTH trader ledgers are empty but CLOB has open orders.
    """
    from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER
    from datetime import datetime, timezone

    # Check if ledgers already have data
    for path in [SINGLE_LEDGER, CONVICTION_LEDGER]:
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

    # Start background prediction refresh thread
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

    logger.info(f"Starting web dashboard at http://{host}:{port}")
    print(f"\n  Dashboard running at: http://{host}:{port}\n")
    app.run(host=host, port=port, debug=debug)
