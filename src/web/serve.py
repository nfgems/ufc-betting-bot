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

from src.config import LOGS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "bot.log"),
    ],
)
logger = logging.getLogger(__name__)


def run_background_monitor(interval_hours: float = 6.0):
    """Run the monitor + line tracker in a background loop."""
    # Wait for the web server to start before doing anything heavy
    time.sleep(10)

    logger.info(f"Background monitor started (every {interval_hours}h)")

    while True:
        try:
            from src.polymarket.tracker import BetLedger, auto_settle_from_polymarket
            ledger = BetLedger()
            settled = auto_settle_from_polymarket(ledger)
            if settled:
                logger.info(f"Auto-settled {settled} bets")
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

    logger.info(f"Starting on port {port}")

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
