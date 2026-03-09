"""
Live bet & P&L tracker — persistent bet ledger with live-updating dashboard.

Tracks all bets placed (dry run or real), fetches current market prices,
computes unrealized/realized P&L, and provides a live terminal dashboard.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import LOGS_DIR, INITIAL_BANKROLL

logger = logging.getLogger(__name__)

LEDGER_PATH = LOGS_DIR / "bet_ledger.json"
PNL_HISTORY_PATH = LOGS_DIR / "pnl_history.jsonl"


class BetLedger:
    """
    Persistent bet ledger — survives restarts.

    Stores every bet with status tracking:
    - open: bet placed, waiting for event
    - won: event settled, fighter won
    - lost: event settled, fighter lost
    - cancelled: order was cancelled or never filled
    """

    def __init__(self, path: Path = LEDGER_PATH):
        self.path = path
        self.bets: list[dict] = []
        self._load()

    def _load(self):
        """Load ledger from disk."""
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self.bets = data.get("bets", [])
                logger.info(f"Loaded {len(self.bets)} bets from ledger")
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupt ledger file, starting fresh")
                self.bets = []
        else:
            self.bets = []

    def _save(self):
        """Persist ledger to disk."""
        data = {
            "bets": self.bets,
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def add_bet(
        self,
        fighter: str,
        opponent: str,
        side: str,
        amount: float,
        price: float,
        shares: float,
        token_id: str,
        market_id: str,
        model_prob: float,
        market_prob: float,
        edge: float,
        decimal_odds: float,
        dry_run: bool = True,
        event_date: Optional[str] = None,
    ) -> dict:
        """Record a new bet in the ledger."""
        bet = {
            "id": len(self.bets) + 1,
            "fighter": fighter,
            "opponent": opponent,
            "side": side,
            "amount": round(amount, 2),
            "price": round(price, 4),
            "shares": round(shares, 2),
            "token_id": token_id,
            "market_id": market_id,
            "model_prob": round(model_prob, 4),
            "market_prob": round(market_prob, 4),
            "edge": round(edge, 4),
            "decimal_odds": round(decimal_odds, 4),
            "dry_run": dry_run,
            "status": "open",
            "placed_at": datetime.now().isoformat(),
            "event_date": event_date,
            "settled_at": None,
            "result_pnl": None,
            "cur_price": None,
        }
        self.bets.append(bet)
        self._save()
        logger.info(
            f"Ledger: recorded bet #{bet['id']} — "
            f"${amount:.2f} on {fighter} @ {price:.4f}"
        )
        return bet

    def settle_bet(self, bet_id: int, won: bool) -> None:
        """Mark a bet as won or lost."""
        for bet in self.bets:
            if bet["id"] == bet_id:
                bet["status"] = "won" if won else "lost"
                bet["settled_at"] = datetime.now().isoformat()
                if won:
                    bet["result_pnl"] = round(
                        bet["shares"] * 1.0 - bet["amount"], 2
                    )  # shares pay out $1 each on win
                else:
                    bet["result_pnl"] = -bet["amount"]
                self._save()
                logger.info(
                    f"Ledger: settled bet #{bet_id} — "
                    f"{'WON' if won else 'LOST'} "
                    f"P&L: ${bet['result_pnl']:+.2f}"
                )
                return
        logger.warning(f"Bet #{bet_id} not found in ledger")

    def cancel_bet(self, bet_id: int) -> None:
        """Mark a bet as cancelled."""
        for bet in self.bets:
            if bet["id"] == bet_id:
                bet["status"] = "cancelled"
                bet["settled_at"] = datetime.now().isoformat()
                bet["result_pnl"] = 0.0
                self._save()
                return

    def update_current_price(self, bet_id: int, cur_price: float) -> None:
        """Update the current market price for an open bet."""
        for bet in self.bets:
            if bet["id"] == bet_id:
                bet["cur_price"] = round(cur_price, 4)
                self._save()
                return

    @property
    def open_bets(self) -> list[dict]:
        return [b for b in self.bets if b["status"] == "open"]

    @property
    def settled_bets(self) -> list[dict]:
        return [b for b in self.bets if b["status"] in ("won", "lost")]

    def get_summary(self) -> dict:
        """Compute aggregate P&L stats."""
        settled = self.settled_bets
        open_bets = self.open_bets

        total_wagered = sum(b["amount"] for b in settled)
        realized_pnl = sum(b["result_pnl"] for b in settled if b["result_pnl"] is not None)
        wins = sum(1 for b in settled if b["status"] == "won")

        # Unrealized P&L from open bets with current prices
        unrealized_pnl = 0.0
        open_invested = 0.0
        for bet in open_bets:
            open_invested += bet["amount"]
            if bet["cur_price"] is not None:
                cur_value = bet["shares"] * bet["cur_price"]
                unrealized_pnl += cur_value - bet["amount"]

        return {
            "total_bets": len(self.bets),
            "open_bets": len(open_bets),
            "settled_bets": len(settled),
            "wins": wins,
            "losses": len(settled) - wins,
            "win_rate": wins / len(settled) if settled else 0.0,
            "total_wagered": total_wagered,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": realized_pnl + unrealized_pnl,
            "roi": realized_pnl / total_wagered if total_wagered > 0 else 0.0,
            "open_invested": open_invested,
        }


def _log_pnl_snapshot(summary: dict) -> None:
    """Append a P&L snapshot to history for trend tracking."""
    record = {
        **summary,
        "timestamp": datetime.now().isoformat(),
    }
    with open(PNL_HISTORY_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _load_pnl_history() -> list[dict]:
    """Load P&L history for trend display."""
    history = []
    if PNL_HISTORY_PATH.exists():
        with open(PNL_HISTORY_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return history


def _clear_screen():
    """Clear terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def _format_pnl(value: float) -> str:
    """Format P&L with color indicators."""
    if value > 0:
        return f"+${value:.2f}"
    elif value < 0:
        return f"-${abs(value):.2f}"
    return "$0.00"


def run_live_dashboard(
    clob_client=None,
    refresh_seconds: int = 30,
    include_dry_runs: bool = True,
) -> None:
    """
    Run a live-updating terminal dashboard showing positions and P&L.

    Args:
        clob_client: Authenticated CLOB client (for fetching live prices)
        refresh_seconds: How often to refresh (default 30s)
        include_dry_runs: Whether to show dry-run bets
    """
    ledger = BetLedger()

    print(f"Starting live dashboard (refresh every {refresh_seconds}s)")
    print("Press Ctrl+C to stop\n")

    while True:
        try:
            # Update current prices for open bets
            if clob_client:
                for bet in ledger.open_bets:
                    if bet["token_id"]:
                        try:
                            price_data = clob_client.get_price(bet["token_id"])
                            ledger.update_current_price(
                                bet["id"], price_data["mid"]
                            )
                        except Exception:
                            pass

            summary = ledger.get_summary()
            _log_pnl_snapshot(summary)

            _clear_screen()
            _print_dashboard(ledger, summary, include_dry_runs)

            time.sleep(refresh_seconds)

        except KeyboardInterrupt:
            print("\nDashboard stopped.")
            break
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            time.sleep(5)


def _print_dashboard(
    ledger: BetLedger,
    summary: dict,
    include_dry_runs: bool = True,
) -> None:
    """Print the dashboard to terminal."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"{'=' * 70}")
    print(f"  UFC BETTING BOT — LIVE TRACKER        {now}")
    print(f"{'=' * 70}")

    # --- P&L Summary ---
    print(f"\n  P&L SUMMARY")
    print(f"  {'-' * 40}")
    print(f"  Realized P&L:    {_format_pnl(summary['realized_pnl']):>12}")
    print(f"  Unrealized P&L:  {_format_pnl(summary['unrealized_pnl']):>12}")
    print(f"  Total P&L:       {_format_pnl(summary['total_pnl']):>12}")
    if summary['total_wagered'] > 0:
        print(f"  ROI:             {summary['roi']:>11.1%}")
    print(f"  Total wagered:   {'${:.2f}'.format(summary['total_wagered']):>12}")
    print(f"  Win rate:        {summary['win_rate']:>11.1%}  "
          f"({summary['wins']}W / {summary['losses']}L)")

    # --- Open Positions ---
    open_bets = ledger.open_bets
    if not include_dry_runs:
        open_bets = [b for b in open_bets if not b.get("dry_run")]

    if open_bets:
        print(f"\n  OPEN POSITIONS ({len(open_bets)})")
        print(f"  {'-' * 66}")
        print(
            f"  {'#':>3}  {'Fighter':<22} {'Amt':>7} {'Entry':>6} "
            f"{'Cur':>6} {'P&L':>8} {'Edge':>6}"
        )
        print(f"  {'-' * 66}")

        for bet in open_bets:
            cur = bet.get("cur_price")
            if cur is not None:
                cur_value = bet["shares"] * cur
                pnl = cur_value - bet["amount"]
                pnl_str = _format_pnl(pnl)
                cur_str = f"${cur:.2f}"
            else:
                pnl_str = "   --"
                cur_str = "  --"

            dry = " (dry)" if bet.get("dry_run") else ""
            name = bet["fighter"][:20] + dry
            print(
                f"  {bet['id']:>3}  {name:<22} "
                f"${bet['amount']:>6.2f} ${bet['price']:.2f} "
                f"{cur_str:>6} {pnl_str:>8} "
                f"{bet['edge']:>+5.1%}"
            )

        total_open = sum(b["amount"] for b in open_bets)
        print(f"  {'-' * 66}")
        print(f"  {'Total open':>27} ${total_open:>6.2f}")

    # --- Recent Settled Bets ---
    settled = ledger.settled_bets
    if not include_dry_runs:
        settled = [b for b in settled if not b.get("dry_run")]

    recent_settled = settled[-10:]  # Last 10
    if recent_settled:
        print(f"\n  RECENT RESULTS ({len(settled)} total)")
        print(f"  {'-' * 56}")
        print(
            f"  {'#':>3}  {'Fighter':<22} {'Amt':>7} "
            f"{'Result':>6} {'P&L':>8}"
        )
        print(f"  {'-' * 56}")

        for bet in recent_settled:
            result = bet["status"].upper()
            pnl_str = _format_pnl(bet["result_pnl"]) if bet["result_pnl"] is not None else "--"
            print(
                f"  {bet['id']:>3}  {bet['fighter']:<22} "
                f"${bet['amount']:>6.2f} "
                f"{result:>6} {pnl_str:>8}"
            )

    # --- P&L Trend (mini sparkline from history) ---
    history = _load_pnl_history()
    if len(history) >= 2:
        recent = history[-20:]
        pnl_values = [h.get("total_pnl", 0) for h in recent]
        _print_pnl_trend(pnl_values)

    print(f"\n{'=' * 70}")


def _print_pnl_trend(values: list[float]) -> None:
    """Print a simple ASCII P&L trend chart."""
    if not values:
        return

    print(f"\n  P&L TREND (last {len(values)} snapshots)")
    print(f"  {'-' * 40}")

    min_val = min(values)
    max_val = max(values)
    spread = max_val - min_val

    if spread == 0:
        # Flat line
        for v in values:
            print(f"  ${v:>+8.2f} |{'=' * 20}|")
        return

    chart_width = 30
    for v in values:
        # Normalize to 0-chart_width
        pos = int((v - min_val) / spread * chart_width)
        zero_pos = int((0 - min_val) / spread * chart_width) if min_val < 0 else 0

        bar = list(" " * (chart_width + 1))
        if zero_pos <= chart_width:
            bar[zero_pos] = "|"

        if v >= 0:
            for i in range(zero_pos, pos + 1):
                if i <= chart_width:
                    bar[i] = "#"
        else:
            for i in range(pos, zero_pos + 1):
                if i <= chart_width:
                    bar[i] = "-"

        print(f"  ${v:>+8.2f} {''.join(bar)}")


def auto_settle_from_polymarket(ledger: BetLedger, clob_client=None) -> int:
    """
    Check Polymarket for settled markets and auto-settle matching bets.

    Returns number of bets settled.
    """
    from src.polymarket.client import GammaClient

    gamma = GammaClient()
    settled_count = 0

    for bet in ledger.open_bets:
        if not bet.get("market_id"):
            continue

        try:
            market = gamma.get_market(bet["market_id"])
            if not market:
                continue

            # Check if market is resolved
            resolved = market.get("resolved", False)
            if not resolved:
                continue

            # Determine outcome
            winning_token = market.get("winning_token_id", "")
            won = winning_token == bet["token_id"]

            ledger.settle_bet(bet["id"], won)
            settled_count += 1

            logger.info(
                f"Auto-settled: {bet['fighter']} "
                f"{'WON' if won else 'LOST'}"
            )
        except Exception as e:
            logger.warning(
                f"Could not check settlement for bet #{bet['id']}: {e}"
            )

    return settled_count
