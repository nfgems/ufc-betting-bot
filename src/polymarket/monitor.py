"""
Position monitor — tracks open orders, fills, P&L, and active positions
on Polymarket using the Data API and CLOB API.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from src.config import POLYMARKET_CLOB_URL, POLYMARKET_PRIVATE_KEY, POLYMARKET_FUNDER_ADDRESS, LOGS_DIR

logger = logging.getLogger(__name__)

DATA_API_URL = "https://data-api.polymarket.com"
POSITIONS_LOG = LOGS_DIR / "positions.jsonl"
ORDERS_LOG = LOGS_DIR / "orders.jsonl"


class PositionMonitor:
    """
    Monitors positions, order fills, and P&L on Polymarket.

    Uses:
    - CLOB API (authenticated): open orders, order status
    - Data API (public): user positions, trades, activity
    """

    def __init__(self, clob_client=None, wallet_address: Optional[str] = None):
        """
        Args:
            clob_client: Authenticated ClobClientWrapper (for order management)
            wallet_address: Polygon wallet address (for Data API lookups).
                           If None, derived from private key.
        """
        self.clob = clob_client
        self.wallet_address = wallet_address
        self._positions_cache: dict = {}
        self._last_check: float = 0

        # Use funder (proxy) address first — this is what Polymarket's Data API indexes.
        # Priority: explicit funder env var > CLOB client proxy discovery > EOA derivation
        if not self.wallet_address and POLYMARKET_FUNDER_ADDRESS:
            self.wallet_address = POLYMARKET_FUNDER_ADDRESS.lower()
            logger.info(f"Wallet address (funder): {self.wallet_address}")
        elif not self.wallet_address and self.clob:
            # Use Gamma-based proxy discovery from ClobClientWrapper
            try:
                proxy = self.clob.proxy_address
                if proxy:
                    self.wallet_address = proxy.lower()
                    logger.info(f"Wallet address (proxy via CLOB): {self.wallet_address}")
            except Exception as e:
                logger.debug(f"Could not get proxy address from CLOB client: {e}")
        if not self.wallet_address and POLYMARKET_PRIVATE_KEY:
            try:
                from eth_account import Account
                acct = Account.from_key(POLYMARKET_PRIVATE_KEY)
                eoa = acct.address.lower()
                logger.warning(
                    f"Using EOA address {eoa} — Data API may not show positions. "
                    f"Set POLYMARKET_FUNDER_ADDRESS to the proxy wallet address."
                )
                self.wallet_address = eoa
            except ImportError:
                logger.warning(
                    "eth_account not installed — wallet address not derived. "
                    "Install with: pip install eth-account"
                )
            except Exception as e:
                logger.warning(f"Could not derive wallet address: {e}")

    # ------------------------------------------------------------------
    # Data API — public endpoints (no auth needed)
    # ------------------------------------------------------------------

    def get_positions(
        self,
        *,
        redeemable_only: bool = False,
        mergeable_only: bool = False,
        limit: int = 100,
    ) -> list[dict]:
        """
        Get all active positions for this wallet from the Data API.

        Returns list of position dicts with token_id, size, avg_price, etc.
        """
        if not self.wallet_address:
            logger.warning("No wallet address — cannot fetch positions")
            return []

        try:
            params = {
                "user": self.wallet_address,
                "limit": limit,
                # Keep dust positions visible for P&L and redeem checks.
                "sizeThreshold": 0,
            }
            if redeemable_only:
                params["redeemable"] = True
            if mergeable_only:
                params["mergeable"] = True
            resp = requests.get(
                f"{DATA_API_URL}/positions",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            positions = resp.json()

            # Filter to non-zero positions
            active = [
                p for p in positions
                if float(p.get("size", 0)) > 0
            ]

            logger.info(f"Found {len(active)} active positions")
            return active
        except Exception as e:
            logger.warning(f"Failed to fetch positions: {e}")
            return []

    def get_redeemable_positions(self, *, limit: int = 500) -> list[dict]:
        """Get all non-zero positions that Polymarket marks as redeemable."""
        return self.get_positions(redeemable_only=True, limit=limit)

    def get_trades(self, limit: int = 50) -> list[dict]:
        """Get recent trades for this wallet from the Data API."""
        if not self.wallet_address:
            return []

        try:
            resp = requests.get(
                f"{DATA_API_URL}/activity",
                params={"user": self.wallet_address, "limit": limit},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch trades: {e}")
            return []

    # ------------------------------------------------------------------
    # CLOB API — order management (requires auth)
    # ------------------------------------------------------------------

    def get_open_orders(self) -> list[dict]:
        """Get all open (unfilled) orders from the CLOB API."""
        if not self.clob:
            return []
        try:
            return self.clob.get_open_orders()
        except Exception as e:
            logger.warning(f"Failed to fetch open orders: {e}")
            return []

    def cancel_stale_orders(self, max_age_hours: float = 24.0) -> list[str]:
        """
        Cancel orders that have been open longer than max_age_hours.

        Returns list of cancelled order IDs.
        """
        if not self.clob:
            return []

        open_orders = self.get_open_orders()
        cancelled = []
        now = time.time()

        for order in open_orders:
            created = order.get("timestamp", order.get("created_at", ""))
            if not created:
                continue

            try:
                if isinstance(created, (int, float)):
                    order_time = created
                else:
                    order_time = datetime.fromisoformat(
                        str(created).replace("Z", "+00:00")
                    ).timestamp()

                age_hours = (now - order_time) / 3600
                if age_hours > max_age_hours:
                    order_id = order.get("id", "")
                    if order_id:
                        self.clob.cancel_order(order_id)
                        cancelled.append(order_id)
                        logger.info(
                            f"Cancelled stale order {order_id} "
                            f"(age: {age_hours:.1f}h)"
                        )
            except Exception as e:
                logger.warning(f"Error checking order age: {e}")

        if cancelled:
            logger.info(f"Cancelled {len(cancelled)} stale orders")
        return cancelled

    # ------------------------------------------------------------------
    # P&L tracking
    # ------------------------------------------------------------------

    def compute_pnl(self) -> dict:
        """
        Compute current P&L across all positions.

        Uses Polymarket's own P&L calculations (cashPnl, percentPnl) so the
        dashboard matches what Polymarket shows.  Falls back to manual
        calculation only when the API fields are missing.

        Returns dict with:
            - total_invested: total USDC spent on positions
            - current_value: current market value of positions
            - unrealized_pnl: current_value - total_invested
            - realized_pnl: profit from settled/sold positions
            - positions: list of per-position P&L details
        """
        positions = self.get_positions()

        total_invested = 0.0
        current_value = 0.0
        total_realized = 0.0
        position_details = []

        for pos in positions:
            size = float(pos.get("size", 0))
            avg_price = float(pos.get("avgPrice", pos.get("avg_price", 0)))
            cur_price = float(pos.get("curPrice", pos.get("cur_price", avg_price)))

            # Prefer Polymarket's own P&L values for 1:1 accuracy
            invested = float(pos.get("initialValue", size * avg_price))
            value = float(pos.get("currentValue", size * cur_price))
            cash_pnl = float(pos.get("cashPnl", value - invested))
            pct_pnl = float(pos.get("percentPnl", (cash_pnl / invested * 100) if invested > 0 else 0))
            realized = float(pos.get("realizedPnl", 0))

            total_invested += invested
            current_value += value
            total_realized += realized

            position_details.append({
                "token_id": pos.get("asset", pos.get("token_id", "")),
                "market": pos.get("title", pos.get("question", "Unknown")),
                "side": pos.get("outcome", ""),
                "opposite_side": pos.get("oppositeOutcome", ""),
                "size": size,
                "avg_price": avg_price,
                "cur_price": cur_price,
                "invested": invested,
                "value": value,
                "unrealized_pnl": cash_pnl,
                "pnl_pct": pct_pnl,
                "realized_pnl": realized,
                "event_slug": pos.get("eventSlug", ""),
                "end_date": pos.get("endDate", ""),
                "icon": pos.get("icon", ""),
                "redeemable": pos.get("redeemable", False),
            })

        result = {
            "total_invested": total_invested,
            "current_value": current_value,
            "unrealized_pnl": current_value - total_invested,
            "realized_pnl": total_realized,
            "total_pnl": (current_value - total_invested) + total_realized,
            "num_positions": len(position_details),
            "positions": position_details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return result

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_order(self, order_info: dict) -> None:
        """Append an order record to the persistent orders log."""
        record = {
            **order_info,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(ORDERS_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_positions_snapshot(self) -> None:
        """Save a snapshot of current positions to the positions log."""
        pnl = self.compute_pnl()
        with open(POSITIONS_LOG, "a") as f:
            f.write(json.dumps(pnl) + "\n")
        logger.info(
            f"Positions snapshot: {pnl['num_positions']} positions, "
            f"invested=${pnl['total_invested']:.2f}, "
            f"unrealized P&L=${pnl['unrealized_pnl']:+.2f}"
        )

    # ------------------------------------------------------------------
    # Status display
    # ------------------------------------------------------------------

    def print_status(self) -> None:
        """Print a summary of current positions and P&L."""
        pnl = self.compute_pnl()

        print(f"\n{'='*60}")
        print(f"  POLYMARKET POSITION MONITOR")
        print(f"{'='*60}")
        print(f"  Wallet: {self.wallet_address or 'unknown'}")
        print(f"  Positions: {pnl['num_positions']}")
        print(f"  Total invested: ${pnl['total_invested']:.2f}")
        print(f"  Current value:  ${pnl['current_value']:.2f}")
        print(f"  Unrealized P&L: ${pnl['unrealized_pnl']:+.2f}")
        print(f"  Realized P&L:   ${pnl['realized_pnl']:+.2f}")
        print(f"  Total P&L:      ${pnl['total_pnl']:+.2f}")

        if pnl['positions']:
            print(f"\n  {'Fighter/Market':<30} {'Size':>6} {'Avg':>6} {'Cur':>6} {'P&L':>8}")
            print(f"  {'-'*56}")
            for pos in pnl['positions']:
                name = pos['market'][:28]
                print(
                    f"  {name:<30} {pos['size']:>6.1f} "
                    f"${pos['avg_price']:.2f} ${pos['cur_price']:.2f} "
                    f"${pos['unrealized_pnl']:>+7.2f}"
                )

        open_orders = self.get_open_orders()
        if open_orders:
            print(f"\n  Open orders: {len(open_orders)}")
            for order in open_orders[:5]:
                print(
                    f"    {order.get('side', '?')} "
                    f"{order.get('size', '?')} shares "
                    f"@ ${order.get('price', '?')}"
                )
            if len(open_orders) > 5:
                print(f"    ... and {len(open_orders) - 5} more")

        print(f"{'='*60}\n")
