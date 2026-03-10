"""
Production entrypoint — runs the web dashboard + background monitor together.

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "bot.log"),
    ],
)
logger = logging.getLogger(__name__)


def run_live_betting_loop(interval_minutes: float = 10.0, min_edge: float = MIN_EDGE_THRESHOLD):
    """Run the live betting bot in a background loop."""
    import argparse

    # Wait for the web server to start
    time.sleep(15)

    logger.info(
        f"Live betting loop started (every {interval_minutes}m, "
        f"min_edge={min_edge:.1%})"
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
                dry_run=False,
                real=True,
                model="xgboost",
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
    # Wait for the web server to start before doing anything heavy
    time.sleep(10)

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
    monitor_interval = float(os.environ.get("MONITOR_INTERVAL_HOURS", "6"))
    bet_interval = float(os.environ.get("BET_INTERVAL_MINUTES", "10"))
    min_edge = float(os.environ.get("MIN_EDGE", str(MIN_EDGE_THRESHOLD)))

    logger.info(f"Starting on port {port}")

    # Start live betting loop thread (daemon)
    betting_thread = threading.Thread(
        target=run_live_betting_loop,
        args=(bet_interval, min_edge),
        daemon=True,
    )
    betting_thread.start()

    # Start background monitor thread (daemon — won't block shutdown)
    monitor_thread = threading.Thread(
        target=run_background_monitor,
        args=(monitor_interval,),
        daemon=True,
    )
    monitor_thread.start()

    # Start web server (foreground — this must start quickly for Railway healthcheck)
    from src.web.app import start_server

    clob = None
    try:
        from src.polymarket.client import ClobClientWrapper
        clob = ClobClientWrapper()
        logger.info("Connected to Polymarket CLOB for live prices")
    except Exception as e:
        logger.warning(f"Running without CLOB (no live prices): {e}")

    start_server(port=port, debug=False, clob_client=clob)


if __name__ == "__main__":
    main()
