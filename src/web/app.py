"""
Web dashboard — local Flask app for live bet & P&L tracking.

Run:
    python -m src.bot web
    python -m src.bot web --port 8080
"""

import json
import logging
import os
import re
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, make_response, render_template

from src.config import LOGS_DIR
from src.polymarket.tracker import BetLedger, _load_pnl_history, auto_settle_from_polymarket, load_all_trader_ledgers
from src.polymarket.monitor import PositionMonitor

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# Shared state — initialized in start_server()
_clob_client = None
_position_monitor = None
_monitor_lock = threading.Lock()

# Simple TTL cache for slow endpoints
_endpoint_cache = {}
_cache_lock = threading.Lock()
SLOW_ENDPOINT_TTL = 300  # 5 minutes
LOG_LINE_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),?\d*\s*\[(\w+)]\s*([\w.]+):\s*(.*)"
)
HANDLED_ACTIVITY_WARNING_PATTERNS = (
    "trading restricted in your region",
    "clob/geoblock",
    "available regions",
)


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
    return [_normalize_activity_entry(entry) for entry in entries]


def _read_recent_log_entries(log_path: Path, limit: int = 500, chunk_bytes: int = 131_072) -> list[dict]:
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
                if len(entries) >= limit or position == 0:
                    return entries[-limit:]
    except Exception:
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


@app.route("/api/summary")
def api_summary():
    """Return summary stats — merges ledger stats with live Polymarket data."""
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
        if live["num_positions"] > 0:
            summary["open_bets"] = live["num_positions"]
            summary["unrealized_pnl"] = live["unrealized_pnl"]
            summary["open_invested"] = live["total_invested"]
            summary["total_pnl"] = summary["realized_pnl"] + live["unrealized_pnl"]
    except Exception:
        pass

    return jsonify(summary)


@app.route("/api/bets")
def api_bets():
    ledger = load_all_trader_ledgers()
    return jsonify({
        "open": ledger.open_bets,
        "settled": ledger.settled_bets,
        "all": ledger.bets,
    })


@app.route("/api/pnl-history")
def api_pnl_history():
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
    for path in [SINGLE_LEDGER, CONVICTION_LEDGER]:
        if Path(path).exists():
            ledger = BetLedger(path=path)
            for bet in ledger.open_bets:
                if bet.get("token_id"):
                    try:
                        price_data = _clob_client.get_price(bet["token_id"])
                        ledger.update_current_price(bet["id"], price_data["mid"])
                        updated += 1
                    except Exception:
                        pass

    return jsonify({"status": "ok", "updated": updated})


@app.route("/api/positions")
def api_positions():
    """Fetch live positions directly from Polymarket's Data API."""
    global _position_monitor
    with _monitor_lock:
        if not _position_monitor:
            _position_monitor = PositionMonitor(clob_client=_clob_client)
        monitor = _position_monitor

    pnl = monitor.compute_pnl()
    return jsonify(pnl)


@app.route("/api/trade-history")
def api_trade_history():
    """Fetch trade history directly from Polymarket's activity API."""
    global _position_monitor
    with _monitor_lock:
        if not _position_monitor:
            _position_monitor = PositionMonitor(clob_client=_clob_client)
        monitor = _position_monitor

    trades = monitor.get_trades(limit=100)
    return jsonify(trades)


@app.route("/api/settle-auto", methods=["POST"])
def api_settle_auto():
    """Auto-settle resolved markets across all trader ledgers."""
    from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER
    count = 0
    for path in [SINGLE_LEDGER, CONVICTION_LEDGER]:
        if Path(path).exists():
            ledger = BetLedger(path=path)
            count += auto_settle_from_polymarket(ledger)
    return jsonify({"settled": count})


@app.route("/api/settle/<int:bet_id>/<result>", methods=["POST"])
def api_settle_manual(bet_id: int, result: str):
    """Manually settle a bet across trader ledgers."""
    from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER
    won = result.lower() in ("win", "won", "w")

    # bet_id is the renumbered merged ID — resolve to original trader ledger ID
    merged = load_all_trader_ledgers()
    target = next((b for b in merged.bets if b["id"] == bet_id), None)
    if not target:
        return jsonify({"ok": False, "error": f"Bet #{bet_id} not found"}), 404

    original_id = target.get("_original_id", bet_id)
    placed_at = target.get("placed_at")

    for path in [SINGLE_LEDGER, CONVICTION_LEDGER]:
        if Path(path).exists():
            ledger = BetLedger(path=path)
            for bet in ledger.bets:
                if bet["id"] == original_id and bet.get("placed_at") == placed_at and bet["status"] == "open":
                    ledger.settle_bet(original_id, won)
                    return jsonify({"ok": True, "bet_id": bet_id, "result": "won" if won else "lost"})
    return jsonify({"ok": False, "error": f"Bet #{bet_id} not found in any trader ledger"}), 404


@app.route("/api/balance")
def api_balance():
    """Return wallet USDC balance and portfolio value (cached 60s)."""
    return jsonify(_cached("balance", 60, _compute_balance))


def _compute_balance():
    balance = 0.0
    portfolio_value = 0.0

    if _clob_client:
        try:
            balance = _clob_client.get_cash_balance()
        except Exception:
            pass
        try:
            portfolio_value = _clob_client.get_portfolio_value()
        except Exception:
            pass

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
    return _json_no_store(_compute_geoblock_status())


@app.route("/api/bot-activity")
def api_bot_activity():
    """Return recent bot activity from bot.log."""
    log_path = LOGS_DIR / "bot.log"
    entries = _read_recent_log_entries(log_path, limit=500)
    return _json_no_store(entries, extra_headers=_bot_activity_headers(log_path, entries))


@app.route("/api/bot-activity-snapshot")
def api_bot_activity_snapshot():
    """Return a single activity snapshot with metadata and entries together."""
    log_path = LOGS_DIR / "bot.log"
    entries = _read_recent_log_entries(log_path, limit=500)
    snapshot = _bot_activity_snapshot(log_path, entries)
    return _json_no_store(snapshot, extra_headers=_bot_activity_headers(log_path, entries))


@app.route("/api/significant-actions")
def api_significant_actions():
    """Return filtered high-value bot actions from bot.log."""
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
            pass

    return jsonify(entries[-30:])


@app.route("/api/upcoming-events")
def api_upcoming_events():
    """Return upcoming UFC events from monitoring snapshots."""
    from src.config import RAW_DATA_DIR

    snapshot_dir = RAW_DATA_DIR / "snapshots"
    if not snapshot_dir.exists():
        return jsonify([])

    # Each snapshot is a per-event file: { event, event_date, timestamp, fights }
    # Group by event name and take the most recent snapshot per event
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
            }
        except Exception:
            continue

    events = list(seen.values())
    events.sort(key=lambda e: e.get("date", ""))
    return jsonify(events[:10])


@app.route("/api/predictions")
def api_predictions():
    """Return cached model predictions for the Model vs Market heatmap."""
    cache_path = LOGS_DIR / "predictions_cache.json"
    if not cache_path.exists():
        return jsonify({"timestamp": None, "predictions": []})

    try:
        data = json.loads(cache_path.read_text())
        # Compute blended probabilities and edges for the frontend
        from src.strategy.value import blend_probability, dynamic_blend_weight

        for p in data.get("predictions", []):
            model_a = p.get("prob_a", 0.5)
            market_a = p.get("a_market_prob", 0.5)
            no_odds_a = p.get("no_odds_prob_a")
            weight = dynamic_blend_weight(model_a, market_a, no_odds_a)
            blend_a = blend_probability(model_a, market_a, weight)
            p["blended_prob_a"] = round(blend_a, 4)
            p["blended_prob_b"] = round(1.0 - blend_a, 4)
            p["edge_a"] = round(blend_a - market_a, 4)
            p["edge_b"] = round((1.0 - blend_a) - p.get("b_market_prob", 0.5), 4)
            p["blend_weight"] = round(weight, 3)

        return jsonify(data)
    except Exception as e:
        logger.error(f"Failed to load predictions cache: {e}")
        return jsonify({"timestamp": None, "predictions": []})


@app.route("/api/trader-race")
def api_trader_race():
    """Return cumulative P&L timeline per trader for the race chart."""
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
    cache_path = LOGS_DIR / "predictions_cache.json"
    if not cache_path.exists():
        return jsonify({"total": 0, "funnel": [], "fights": []})

    try:
        data = json.loads(cache_path.read_text())
        preds = data.get("predictions", [])
        if not preds:
            return jsonify({"total": 0, "funnel": [], "fights": []})

        from src.strategy.value import (
            blend_probability, dynamic_blend_weight, scaled_min_edge,
        )
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

            weight = dynamic_blend_weight(model_a, market_a, no_odds_a)
            blend_a = blend_probability(model_a, market_a, weight)
            blend_b = 1.0 - blend_a
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
            if tid_yes:
                token_map[tid_yes] = {
                    "fighter": m.get("fighter_a", ""),
                    "opponent": m.get("fighter_b", ""),
                    "event_date": end_date,
                    "side": "a",
                }
            if tid_no:
                token_map[tid_no] = {
                    "fighter": m.get("fighter_b", ""),
                    "opponent": m.get("fighter_a", ""),
                    "event_date": end_date,
                    "side": "b",
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


@app.route("/api/predictions-detail")
def api_predictions_detail():
    """Return enriched prediction data with SHAP values and feature highlights."""
    cache_path = LOGS_DIR / "predictions_cache.json"
    if not cache_path.exists():
        return jsonify({"timestamp": None, "predictions": [], "global_feature_importance": []})
    try:
        data = json.loads(cache_path.read_text())
        return jsonify({
            "timestamp": data.get("timestamp"),
            "predictions": data.get("predictions", []),
            "global_feature_importance": data.get("global_feature_importance", []),
        })
    except Exception as e:
        logger.error(f"Failed to load predictions detail: {e}")
        return jsonify({"timestamp": None, "predictions": [], "global_feature_importance": []})


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


def start_server(port: int = 5050, debug: bool = False, clob_client=None):
    """Start the Flask web dashboard."""
    global _clob_client, _position_monitor
    if clob_client:
        _clob_client = clob_client
        _position_monitor = PositionMonitor(clob_client=clob_client)
    else:
        _position_monitor = PositionMonitor(clob_client=None)

    logger.info(f"Starting web dashboard at http://localhost:{port}")
    print(f"\n  Dashboard running at: http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
