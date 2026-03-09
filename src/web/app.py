"""
Web dashboard — local Flask app for live bet & P&L tracking.

Run:
    python -m src.bot web
    python -m src.bot web --port 8080
"""

import logging
from pathlib import Path

from flask import Flask, jsonify, render_template

from src.polymarket.tracker import BetLedger, _load_pnl_history, auto_settle_from_polymarket
from src.polymarket.monitor import PositionMonitor

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# Shared state — initialized in start_server()
_clob_client = None
_position_monitor = None


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/summary")
def api_summary():
    ledger = BetLedger()
    summary = ledger.get_summary()
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
    if not _position_monitor:
        _position_monitor = PositionMonitor(clob_client=_clob_client)

    pnl = _position_monitor.compute_pnl()
    return jsonify(pnl)


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


def start_server(port: int = 5050, debug: bool = False, clob_client=None):
    """Start the Flask web dashboard."""
    global _clob_client, _position_monitor
    _clob_client = clob_client
    _position_monitor = PositionMonitor(clob_client=clob_client)

    logger.info(f"Starting web dashboard at http://localhost:{port}")
    print(f"\n  Dashboard running at: http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
