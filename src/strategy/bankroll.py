"""
Bankroll management — Kelly criterion sizing and risk controls.

Auto-detects Polymarket balance on startup and syncs with the
persistent BetLedger so the bankroll reflects actual state across restarts.
"""

import logging

from src.config import KELLY_FRACTION, MAX_BET_FRACTION, STOP_LOSS_FRACTION, INITIAL_BANKROLL

logger = logging.getLogger(__name__)


def _fetch_polymarket_balance() -> float:
    """Query Polymarket CLOB API for actual cash balance. Returns 0 on failure."""
    try:
        from src.polymarket.client import ClobClientWrapper
        client = ClobClientWrapper()
        balance = client.get_cash_balance()
        if balance > 0:
            logger.info(f"Polymarket live balance: ${balance:.2f}")
            return balance
    except Exception as e:
        logger.debug(f"Could not fetch Polymarket balance: {e}")
    return 0.0


class BankrollManager:
    """Manages bankroll and bet sizing using fractional Kelly criterion."""

    def __init__(
        self,
        initial_bankroll: float = INITIAL_BANKROLL,
        kelly_fraction: float = KELLY_FRACTION,
        max_bet_fraction: float = MAX_BET_FRACTION,
        stop_loss_fraction: float = STOP_LOSS_FRACTION,
        auto_detect_balance: bool = True,
    ):
        # Try to auto-detect actual Polymarket balance
        if auto_detect_balance:
            live_balance = _fetch_polymarket_balance()
            if live_balance > 0:
                initial_bankroll = live_balance

        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.max_bet_fraction = max_bet_fraction
        self.stop_loss_fraction = stop_loss_fraction
        self.peak_bankroll = initial_bankroll
        self.history: list[dict] = []

        # Sync with persistent ledger so bankroll reflects past bets
        self._sync_from_ledger()

    def _sync_from_ledger(self) -> None:
        """Load past bets from the persistent BetLedger and adjust bankroll."""
        try:
            from src.polymarket.tracker import BetLedger
            ledger = BetLedger()
        except Exception:
            return

        if not ledger.bets:
            return

        # Replay all non-dry-run bets to reconstruct bankroll
        real_bets = [b for b in ledger.bets if not b.get("dry_run", True)]
        if not real_bets:
            return

        total_wagered = 0.0
        realized_pnl = 0.0

        for bet in real_bets:
            amount = bet.get("amount", 0)
            status = bet.get("status", "open")

            if status == "open":
                # Money is out — deduct from bankroll
                total_wagered += amount
            elif status == "won":
                # Won: shares pay $1 each
                pnl = bet.get("result_pnl", 0) or 0
                realized_pnl += pnl
            elif status == "lost":
                pnl = bet.get("result_pnl", 0) or 0
                realized_pnl += pnl
            # cancelled: money returned, no effect

        self.bankroll = self.initial_bankroll + realized_pnl - total_wagered
        self.peak_bankroll = max(self.peak_bankroll, self.initial_bankroll + realized_pnl)

        logger.info(
            f"Bankroll synced from ledger: ${self.bankroll:.2f} "
            f"(initial: ${self.initial_bankroll:.2f}, "
            f"realized P&L: ${realized_pnl:+.2f}, "
            f"open bets: ${total_wagered:.2f}, "
            f"{len(real_bets)} total bets)"
        )

    def refresh_balance(self) -> float:
        """Re-query Polymarket for the latest cash balance and update bankroll.

        Call this each cycle so the bankroll stays accurate after wins, losses,
        and manual deposits.  Returns the updated bankroll value.
        """
        live = _fetch_polymarket_balance()
        if live > 0:
            old = self.bankroll
            self.bankroll = live
            self.peak_bankroll = max(self.peak_bankroll, live)
            if abs(live - old) > 0.01:
                logger.info(f"Bankroll refreshed: ${old:.2f} -> ${live:.2f}")
        return self.bankroll

    @property
    def is_stopped(self) -> bool:
        """Check if stop-loss has been triggered."""
        return self.bankroll <= self.initial_bankroll * (1 - self.stop_loss_fraction)

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak."""
        if self.peak_bankroll <= 0:
            return 0.0
        return 1.0 - (self.bankroll / self.peak_bankroll)

    def kelly_bet_size(
        self,
        model_prob: float,
        decimal_odds: float,
    ) -> float:
        """
        Calculate bet size using fractional Kelly criterion.

        Full Kelly: f* = (bp - q) / b
        where b = decimal_odds - 1, p = win probability, q = 1 - p

        We use fractional Kelly (default 25%) for safety.
        """
        if self.is_stopped:
            logger.warning("Stop-loss triggered. No more bets.")
            return 0.0

        b = decimal_odds - 1.0  # net odds
        p = model_prob
        q = 1.0 - p

        if b <= 0 or p <= 0:
            return 0.0

        # Full Kelly fraction
        full_kelly = (b * p - q) / b

        if full_kelly <= 0:
            return 0.0  # Negative Kelly = no edge, don't bet

        # Apply fractional Kelly
        fraction = full_kelly * self.kelly_fraction

        # Apply max bet cap
        fraction = min(fraction, self.max_bet_fraction)

        bet_amount = self.bankroll * fraction

        # Minimum bet of $1 (or equivalent)
        if bet_amount < 1.0:
            return 0.0

        return round(bet_amount, 2)

    def place_bet(
        self,
        amount: float,
        fighter: str,
        decimal_odds: float,
        model_prob: float,
        market_prob: float,
    ) -> dict:
        """Record a bet placement."""
        if amount <= 0:
            return {}

        bet = {
            "fighter": fighter,
            "amount": amount,
            "decimal_odds": decimal_odds,
            "model_prob": model_prob,
            "market_prob": market_prob,
            "edge": model_prob - market_prob,
            "bankroll_before": self.bankroll,
            "potential_profit": amount * (decimal_odds - 1),
            "result": None,
            "profit": None,
        }

        self.bankroll -= amount
        self.history.append(bet)

        logger.info(
            f"BET: ${amount:.2f} on {fighter} @ {decimal_odds:.2f} "
            f"(model: {model_prob:.1%}, market: {market_prob:.1%}, edge: {model_prob - market_prob:.1%})"
        )

        return bet

    def settle_bet(self, bet_index: int, won: bool) -> None:
        """Settle a bet and update bankroll."""
        bet = self.history[bet_index]
        if won:
            profit = bet["amount"] * bet["decimal_odds"]  # Return includes stake
            self.bankroll += profit
            bet["profit"] = profit - bet["amount"]
            bet["result"] = "win"
        else:
            bet["profit"] = -bet["amount"]
            bet["result"] = "loss"

        self.peak_bankroll = max(self.peak_bankroll, self.bankroll)

        logger.info(
            f"SETTLED: {bet['fighter']} {'WON' if won else 'LOST'} | "
            f"P&L: ${bet['profit']:+.2f} | Bankroll: ${self.bankroll:.2f}"
        )

    def get_stats(self) -> dict:
        """Get current bankroll statistics."""
        settled = [b for b in self.history if b["result"] is not None]
        if not settled:
            return {
                "bankroll": self.bankroll,
                "initial_bankroll": self.initial_bankroll,
                "total_bets": 0,
                "roi": 0.0,
            }

        wins = sum(1 for b in settled if b["result"] == "win")
        total_wagered = sum(b["amount"] for b in settled)
        total_profit = sum(b["profit"] for b in settled)

        return {
            "bankroll": self.bankroll,
            "initial_bankroll": self.initial_bankroll,
            "total_bets": len(settled),
            "wins": wins,
            "losses": len(settled) - wins,
            "win_rate": wins / len(settled) if settled else 0,
            "total_wagered": total_wagered,
            "total_profit": total_profit,
            "roi": total_profit / total_wagered if total_wagered > 0 else 0,
            "bankroll_change": self.bankroll - self.initial_bankroll,
            "bankroll_change_pct": (self.bankroll - self.initial_bankroll) / self.initial_bankroll if self.initial_bankroll > 0 else 0.0,
            "peak_bankroll": self.peak_bankroll,
            "max_drawdown": 1 - (min(b["bankroll_before"] for b in settled) / self.peak_bankroll) if self.peak_bankroll > 0 else 0.0,
            "current_drawdown": self.current_drawdown,
            "avg_edge": sum(b["edge"] for b in settled) / len(settled),
            "avg_bet_size": total_wagered / len(settled),
        }
