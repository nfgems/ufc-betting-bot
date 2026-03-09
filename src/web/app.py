"""
Web dashboard — local Flask app for live bet & P&L tracking.

Run:
    python -m src.bot web
    python -m src.bot web --port 8080
"""

import json
import logging
import re
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template

from src.config import LOGS_DIR
from src.polymarket.tracker import BetLedger, _load_pnl_history, auto_settle_from_polymarket
from src.polymarket.monitor import PositionMonitor

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# Shared state — initialized in start_server()
_clob_client = None
_position_monitor = None
_monitor_lock = threading.Lock()


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/summary")
def api_summary():
    """Return summary stats — merges ledger stats with live Polymarket data."""
    ledger = BetLedger()
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
    ledger = BetLedger()
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

    ledger = BetLedger()
    updated = 0
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
    """Auto-settle resolved markets."""
    ledger = BetLedger()
    count = auto_settle_from_polymarket(ledger)
    return jsonify({"settled": count})


@app.route("/api/settle/<int:bet_id>/<result>", methods=["POST"])
def api_settle_manual(bet_id: int, result: str):
    """Manually settle a bet."""
    ledger = BetLedger()
    won = result.lower() in ("win", "won", "w")
    ledger.settle_bet(bet_id, won)
    return jsonify({"ok": True, "bet_id": bet_id, "result": "won" if won else "lost"})


@app.route("/api/balance")
def api_balance():
    """Return wallet USDC balance and portfolio value."""
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

    return jsonify({
        "cash_balance": balance,
        "portfolio_value": portfolio_value,
        "total_equity": balance + portfolio_value,
    })


@app.route("/api/bot-activity")
def api_bot_activity():
    """Return recent bot activity from bot.log."""
    log_path = LOGS_DIR / "bot.log"
    entries = []
    if log_path.exists():
        try:
            # Read only the tail of the log to avoid memory issues on large files
            tail_bytes = 32_768  # ~32KB ≈ last few hundred lines
            with open(log_path, "rb") as f:
                f.seek(0, 2)  # seek to end
                size = f.tell()
                f.seek(max(0, size - tail_bytes))
                if size > tail_bytes:
                    f.readline()  # skip partial first line
                raw = f.read().decode("utf-8", errors="replace")

            for line in raw.splitlines()[-100:]:
                # Format: 2024-01-15 12:30:45,123 [INFO] src.bot: message
                m = re.match(
                    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),?\d*\s*\[(\w+)]\s*([\w.]+):\s*(.*)",
                    line,
                )
                if m:
                    entries.append({
                        "timestamp": m.group(1),
                        "level": m.group(2),
                        "source": m.group(3),
                        "message": m.group(4),
                    })
        except Exception:
            pass

    return jsonify(entries[-50:])


@app.route("/api/upcoming-events")
def api_upcoming_events():
    """Return upcoming UFC events from monitoring snapshots."""
    from src.config import RAW_DATA_DIR

    snapshot_dir = RAW_DATA_DIR / "snapshots"
    events = []

    if snapshot_dir.exists():
        # Find the most recent snapshot
        snapshots = sorted(snapshot_dir.glob("*.json"), reverse=True)
        if snapshots:
            try:
                data = json.loads(snapshots[0].read_text())
                events = data if isinstance(data, list) else data.get("events", [])
            except Exception:
                pass

    # Also try the live_monitor signals file
    signals_path = LOGS_DIR / "monitor_signals.json"
    if not events and signals_path.exists():
        try:
            data = json.loads(signals_path.read_text())
            events = data.get("events", [])
        except Exception:
            pass

    return jsonify(events[:10])


@app.route("/api/trader-breakdown")
def api_trader_breakdown():
    """Return per-trader P&L breakdown from individual ledgers."""
    from src.strategy.triple_trader import TRADER_A_LEDGER, TRADER_B_LEDGER, TRADER_C_LEDGER

    breakdown = []
    traders = [
        ("A", "Conservative", TRADER_A_LEDGER, 0.20),
        ("B", "Aggressive", TRADER_B_LEDGER, 0.40),
        ("C", "Conviction", TRADER_C_LEDGER, None),
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


def start_server(port: int = 5050, debug: bool = False, clob_client=None):
    """Start the Flask web dashboard."""
    global _clob_client, _position_monitor
    _clob_client = clob_client
    _position_monitor = PositionMonitor(clob_client=clob_client)

    logger.info(f"Starting web dashboard at http://localhost:{port}")
    print(f"\n  Dashboard running at: http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
