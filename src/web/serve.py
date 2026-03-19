"""
Production entrypoint â€” runs the web dashboard + background monitor together.

Used by Railway/Docker for always-on deployment.
"""

import logging
import os
import sys
import threading
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import LOGS_DIR, MIN_EDGE_THRESHOLD
from src.live_control import (
    LIVE_MODE_DRY_RUN,
    LIVE_MODE_ENV,
    LIVE_MODE_OFF,
    LIVE_MODE_REAL,
    evaluate_live_startup,
    resolve_live_model_name,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "bot.log"),
    ],
)
logger = logging.getLogger(__name__)


def run_live_betting_loop(
    interval_minutes: float = 10.0,
    min_edge: float = MIN_EDGE_THRESHOLD,
    *,
    trading_mode: str = LIVE_MODE_DRY_RUN,
    model_name: str = "xgboost",
):
    """Run the live betting bot in a background loop."""
    import argparse
    from src.web.app import update_runtime_component

    # Wait for the web server to start
    time.sleep(15)

    if trading_mode not in {LIVE_MODE_DRY_RUN, LIVE_MODE_REAL}:
        update_runtime_component(
            "betting_loop",
            "disabled",
            f"Trading mode '{trading_mode}' does not start the betting loop.",
        )
        logger.info("Live betting loop disabled: unsupported trading mode %s", trading_mode)
        return

    update_runtime_component(
        "betting_loop",
        "running",
        f"Trading mode: {trading_mode}",
    )
    logger.info(
        f"Live betting loop started in {trading_mode} mode "
        f"(every {interval_minutes}m, min_edge={min_edge:.1%}, model={model_name})"
    )

    while True:
        # Cancel any limit bids for fights that have already started
        try:
            from src.polymarket.executor import cancel_all_stale_limit_bids

            cancelled = cancel_all_stale_limit_bids()
            if cancelled:
                logger.info(f"Pre-cycle cleanup: cancelled {cancelled} stale limit bid(s)")
        except Exception as e:
            logger.error(f"Limit bid cancellation error: {e}", exc_info=True)

        # Snapshot odds + Polymarket prices before each betting cycle so
        # Sharp Money Tracker has fresh line data every cycle
        try:
            from src.data.line_tracker import snapshot_odds, snapshot_polymarket_prices

            snapshot_odds()
            snapshot_polymarket_prices()
        except Exception as e:
            logger.warning(f"Pre-cycle line snapshot failed (non-fatal): {e}")

        try:
            from src.bot import cmd_duo_live

            args = argparse.Namespace(
                dry_run=trading_mode != LIVE_MODE_REAL,
                real=trading_mode == LIVE_MODE_REAL,
                model=model_name,
                min_edge=min_edge,
            )
            cmd_duo_live(args)
        except Exception as e:
            logger.error(f"Live betting error: {e}", exc_info=True)

        # Auto-settle any resolved markets each cycle
        try:
            from src.polymarket.tracker import BetLedger, auto_settle_from_polymarket
            from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER

            total_settled = 0
            for label, path in [("S", SINGLE_LEDGER), ("C", CONVICTION_LEDGER)]:
                if Path(path).exists():
                    ledger = BetLedger(path=path)
                    settled = auto_settle_from_polymarket(ledger)
                    if settled:
                        logger.info(f"Auto-settled {settled} bets for Trader {label}")
                        total_settled += settled
            if total_settled:
                logger.info(f"Auto-settled {total_settled} bets total")
        except Exception as e:
            logger.error(f"Auto-settle error: {e}")

        logger.info(f"Next betting cycle in {interval_minutes} minutes")
        time.sleep(interval_minutes * 60)


def run_background_monitor(interval_hours: float = 6.0):
    """Run the monitor + line tracker in a background loop."""
    from src.web.app import update_runtime_component

    # Wait for the web server to start before doing anything heavy
    time.sleep(10)

    update_runtime_component("monitor_loop", "running", "Background monitor active.")
    logger.info(f"Background monitor started (every {interval_hours}h)")

    while True:
        try:
            from src.polymarket.tracker import BetLedger, auto_settle_from_polymarket
            from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER

            total_settled = 0
            for label, path in [("S", SINGLE_LEDGER), ("C", CONVICTION_LEDGER)]:
                ledger = BetLedger(path=path)
                settled = auto_settle_from_polymarket(ledger)
                if settled:
                    logger.info(f"Auto-settled {settled} bets for Trader {label}")
                    total_settled += settled
            if total_settled:
                logger.info(f"Auto-settled {total_settled} bets total")
        except Exception as e:
            logger.error(f"Auto-settle error: {e}")

        try:
            from src.data.live_monitor import run_monitoring_pass

            signals = run_monitoring_pass()
            logger.info(f"Monitor pass: {len(signals.get('events', []))} events")
        except Exception as e:
            logger.error(f"Monitor error: {e}")

        try:
            from src.data.line_tracker import run_line_tracking_pass

            line_summary = run_line_tracking_pass()
            logger.info(f"Line tracking: {line_summary.get('sharp_moves', 0)} sharp moves")
        except Exception as e:
            logger.error(f"Line tracking error: {e}")

        time.sleep(interval_hours * 3600)


def main():
    port = int(os.environ.get("PORT", 5050))
    host = os.environ.get("WEB_HOST", "0.0.0.0")
    monitor_interval = float(os.environ.get("MONITOR_INTERVAL_HOURS", "6"))
    bet_interval = float(os.environ.get("BET_INTERVAL_MINUTES", "10"))
    min_edge = float(os.environ.get("MIN_EDGE", str(MIN_EDGE_THRESHOLD)))
    model_name = resolve_live_model_name(os.environ.get("LIVE_MODEL"))

    logger.info(f"Starting on {host}:{port}")

    from src.web.app import (
        set_clob_client,
        set_runtime_status,
        start_server,
        update_runtime_component,
    )

    runtime_status = evaluate_live_startup(
        requested_mode=os.environ.get(LIVE_MODE_ENV, LIVE_MODE_OFF),
        host=host,
        model_name=model_name,
        startup_source="serve",
    )
    runtime_status["components"] = {
        "web": {"state": "starting", "message": f"Binding {host}:{port}"},
        "monitor_loop": {"state": "starting", "message": "Monitor thread booting."},
        "clob": {"state": "starting", "message": "CLOB initialization pending."},
        "betting_loop": {
            "state": "starting" if runtime_status["trading_enabled"] else "disabled",
            "message": (
                f"Trading mode: {runtime_status['effective_live_mode']}"
                if runtime_status["trading_enabled"]
                else (
                    "Trading loop disabled by policy."
                    if runtime_status["requested_live_mode"] == LIVE_MODE_OFF
                    else "Trading loop blocked by readiness checks."
                )
            ),
        },
    }
    set_runtime_status(runtime_status)

    if runtime_status["errors"]:
        logger.error(
            "Hosted trading startup is not ready: %s",
            " | ".join(runtime_status["errors"]),
        )
    if runtime_status["warnings"]:
        logger.warning(
            "Hosted trading startup warnings: %s",
            " | ".join(runtime_status["warnings"]),
        )

    if runtime_status["trading_enabled"]:
        betting_thread = threading.Thread(
            target=run_live_betting_loop,
            kwargs={
                "interval_minutes": bet_interval,
                "min_edge": min_edge,
                "trading_mode": runtime_status["effective_live_mode"],
                "model_name": model_name,
            },
            daemon=True,
        )
        betting_thread.start()
    else:
        update_runtime_component(
            "betting_loop",
            "disabled",
            "Trading loop not started.",
        )

    monitor_thread = threading.Thread(
        target=run_background_monitor,
        args=(monitor_interval,),
        daemon=True,
    )
    monitor_thread.start()

    def _init_clob():
        time.sleep(2)  # let Flask bind the port first
        try:
            from src.polymarket.client import ClobClientWrapper

            clob = ClobClientWrapper()
            set_clob_client(clob)
            update_runtime_component("clob", "running", "Connected to Polymarket CLOB.")
            logger.info("Connected to Polymarket CLOB for live prices")
        except Exception as e:
            update_runtime_component("clob", "degraded", str(e))
            logger.warning(f"Running without CLOB (no live prices): {e}")

    clob_thread = threading.Thread(target=_init_clob, daemon=True)
    clob_thread.start()

    update_runtime_component("web", "running", f"Serving on http://{host}:{port}")
    start_server(port=port, debug=False, clob_client=None, host=host)


if __name__ == "__main__":
    main()
