"""
Bankroll management for Kelly sizing and risk controls.

For live trading, treat total account equity and available cash separately:
- total equity drives sizing and drawdown
- available cash determines whether a new order can actually be placed
"""

import logging
from datetime import datetime
from pathlib import Path

from src.config import INITIAL_BANKROLL, KELLY_FRACTION, MAX_BET_FRACTION, STOP_LOSS_FRACTION

logger = logging.getLogger(__name__)


def _fetch_polymarket_account_state(
    *,
    require_confirmed_cash: bool = False,
    require_portfolio_value: bool = False,
) -> dict[str, float | str | bool]:
    """Fetch live Polymarket cash, portfolio value, and total equity."""

    fallback = {
        "cash_balance": 0.0,
        "portfolio_value": 0.0,
        "total_equity": 0.0,
        "cash_source": "unavailable",
        "portfolio_source": "unavailable",
        "confirmed_cash": False,
        "confirmed_portfolio": False,
    }

    try:
        from src.polymarket.client import ClobClientWrapper

        client = ClobClientWrapper()
        cash_details = client.get_cash_balance_details(
            allow_onchain_fallback=not require_confirmed_cash
        )
        cash_balance = float(cash_details.get("balance") or 0.0)
        cash_source = str(cash_details.get("source", "unavailable") or "unavailable")
        confirmed_cash = cash_source == "clob"

        portfolio_details = client.get_portfolio_value_details()
        portfolio_value = float(portfolio_details.get("value") or 0.0)
        portfolio_source = str(
            portfolio_details.get("source", "unavailable") or "unavailable"
        )
        confirmed_portfolio = portfolio_source == "data_api"

        if confirmed_cash:
            logger.info("Polymarket live cash balance: $%.2f", cash_balance)
        elif cash_balance > 0 and cash_source == "onchain":
            logger.warning(
                "Polymarket CLOB cash balance unavailable; using on-chain USDC fallback: $%.2f",
                cash_balance,
            )

        if confirmed_portfolio:
            logger.info("Polymarket live portfolio value: $%.2f", portfolio_value)
        elif require_portfolio_value:
            logger.warning("Polymarket portfolio value could not be confirmed")

        if require_confirmed_cash and not confirmed_cash:
            result = dict(fallback)
            result["cash_source"] = cash_source
            result["portfolio_source"] = portfolio_source
            return result

        if require_portfolio_value and not confirmed_portfolio:
            result = dict(fallback)
            result["cash_balance"] = cash_balance
            result["cash_source"] = cash_source
            result["confirmed_cash"] = confirmed_cash
            result["portfolio_source"] = portfolio_source
            return result

        return {
            "cash_balance": cash_balance,
            "portfolio_value": portfolio_value,
            "total_equity": cash_balance + portfolio_value,
            "cash_source": cash_source,
            "portfolio_source": portfolio_source,
            "confirmed_cash": confirmed_cash,
            "confirmed_portfolio": confirmed_portfolio,
        }
    except Exception as exc:
        logger.warning("Could not fetch Polymarket account state: %s", exc)
    return fallback


def _fetch_polymarket_balance_details(
    *,
    require_confirmed_cash: bool = False,
) -> dict[str, float | str | bool]:
    """Backward-compatible helper exposing live cash balance details."""

    state = _fetch_polymarket_account_state(
        require_confirmed_cash=require_confirmed_cash,
        require_portfolio_value=False,
    )
    return {
        "balance": float(state.get("cash_balance") or 0.0),
        "source": str(state.get("cash_source", "unavailable") or "unavailable"),
        "confirmed_cash": bool(state.get("confirmed_cash")),
    }


def _fetch_polymarket_balance(*, require_confirmed_cash: bool = False) -> float:
    """Backward-compatible helper returning live available cash only."""

    return float(
        _fetch_polymarket_balance_details(
            require_confirmed_cash=require_confirmed_cash
        )["balance"]
    )


class BankrollManager:
    """Manages bankroll, equity, and bet sizing."""

    def __init__(
        self,
        initial_bankroll: float = INITIAL_BANKROLL,
        kelly_fraction: float = KELLY_FRACTION,
        max_bet_fraction: float = MAX_BET_FRACTION,
        stop_loss_fraction: float = STOP_LOSS_FRACTION,
        auto_detect_balance: bool = False,
        sync_ledger: bool = False,
        ledger_path: Path | None = None,
        *,
        available_cash: float | None = None,
        total_equity: float | None = None,
    ):
        used_live_snapshot = False
        if auto_detect_balance:
            state = _fetch_polymarket_account_state(
                require_confirmed_cash=True,
                require_portfolio_value=False,
            )
            live_cash = float(state.get("cash_balance") or 0.0)
            if live_cash > 0:
                initial_bankroll = float(state.get("total_equity") or live_cash)
                total_equity = float(state.get("total_equity") or live_cash)
                available_cash = live_cash
                used_live_snapshot = True

        if total_equity is None:
            total_equity = float(initial_bankroll)
        if available_cash is None:
            available_cash = float(total_equity)

        self.initial_bankroll = float(initial_bankroll)
        self.bankroll = float(available_cash)
        self.total_equity = float(total_equity)
        self.kelly_fraction = kelly_fraction
        self.max_bet_fraction = max_bet_fraction
        self.stop_loss_fraction = stop_loss_fraction
        self.peak_bankroll = float(total_equity)
        self.history: list[dict] = []
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None

        if sync_ledger and used_live_snapshot:
            logger.info(
                "Skipping ledger replay because bankroll was initialized from live account state"
            )
        elif sync_ledger:
            self._sync_from_ledger()

    @property
    def available_cash(self) -> float:
        return self.bankroll

    @property
    def sizing_bankroll(self) -> float:
        return self.total_equity

    def _sync_from_ledger(self) -> None:
        """Load past bets from the persistent BetLedger and adjust cash/equity."""

        try:
            from src.polymarket.tracker import BetLedger

            ledger = BetLedger(path=self.ledger_path) if self.ledger_path else BetLedger()
        except Exception as exc:
            logger.warning("Ledger sync failed - bankroll will use initial value: %s", exc)
            return

        if not ledger.bets:
            return

        real_bets = [bet for bet in ledger.bets if not bet.get("dry_run", True)]
        if not real_bets:
            return

        total_wagered = 0.0
        realized_pnl = 0.0

        for bet in real_bets:
            amount = float(bet.get("amount", 0) or 0.0)
            status = bet.get("status", "open")

            if status == "open":
                total_wagered += amount
            elif status in {"won", "lost"}:
                realized_pnl += float(bet.get("result_pnl", 0) or 0.0)

        self.total_equity = self.initial_bankroll + realized_pnl
        self.bankroll = self.total_equity - total_wagered
        self.peak_bankroll = max(self.peak_bankroll, self.total_equity)

        logger.info(
            "Bankroll synced from ledger: equity $%.2f, available cash $%.2f "
            "(initial: $%.2f, realized P&L: %+0.2f, open bets: $%.2f, %s total bets)",
            self.total_equity,
            self.bankroll,
            self.initial_bankroll,
            realized_pnl,
            total_wagered,
            len(real_bets),
        )

    def refresh_balance(self) -> float:
        """Refresh live cash/equity from Polymarket and return total equity."""

        state = _fetch_polymarket_account_state(
            require_confirmed_cash=True,
            require_portfolio_value=True,
        )
        if state.get("confirmed_cash") and state.get("confirmed_portfolio"):
            old_equity = self.total_equity
            old_cash = self.bankroll
            self.bankroll = float(state.get("cash_balance") or 0.0)
            self.total_equity = float(state.get("total_equity") or 0.0)
            self.peak_bankroll = max(self.peak_bankroll, self.total_equity)
            if (
                abs(self.total_equity - old_equity) > 0.01
                or abs(self.bankroll - old_cash) > 0.01
            ):
                logger.info(
                    "Bankroll refreshed: equity $%.2f -> $%.2f, cash $%.2f -> $%.2f",
                    old_equity,
                    self.total_equity,
                    old_cash,
                    self.bankroll,
                )
        return self.total_equity

    @property
    def is_stopped(self) -> bool:
        """Check if stop-loss has been triggered against total equity."""

        return self.total_equity <= self.initial_bankroll * (1 - self.stop_loss_fraction)

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak total equity."""

        if self.peak_bankroll <= 0:
            return 0.0
        return 1.0 - (self.total_equity / self.peak_bankroll)

    def kelly_bet_size(
        self,
        model_prob: float,
        decimal_odds: float,
    ) -> float:
        """Calculate bet size using total equity, not free cash."""

        if self.is_stopped:
            logger.warning("Stop-loss triggered. No more bets.")
            return 0.0

        b = decimal_odds - 1.0
        p = model_prob
        q = 1.0 - p

        if b <= 0 or p <= 0:
            return 0.0

        full_kelly = (b * p - q) / b
        if full_kelly <= 0:
            return 0.0

        fraction = min(full_kelly * self.kelly_fraction, self.max_bet_fraction)
        bet_amount = self.total_equity * fraction

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
        """Record a bet placement and reserve available cash."""

        if amount <= 0:
            return {}

        bet = {
            "fighter": fighter,
            "amount": amount,
            "decimal_odds": decimal_odds,
            "model_prob": model_prob,
            "market_prob": market_prob,
            "edge": model_prob - market_prob,
            "bankroll_before": self.total_equity,
            "available_cash_before": self.bankroll,
            "potential_profit": amount * (decimal_odds - 1),
            "result": None,
            "profit": None,
            "placed_at": datetime.now().isoformat(),
            "settled_at": None,
            "bankroll_after": None,
            "available_cash_after": None,
        }

        if self.is_stopped:
            logger.warning("Stop-loss triggered. Rejecting bet on %s.", fighter)
            return {}
        if amount > self.bankroll + 1e-9:
            logger.warning(
                "Rejecting bet on %s: amount $%.2f exceeds available cash $%.2f",
                fighter,
                amount,
                self.bankroll,
            )
            return {}

        self.bankroll -= amount
        bet["available_cash_after"] = self.bankroll
        self.history.append(bet)

        logger.info(
            "BET: $%.2f on %s @ %.2f (model: %.1f%%, market: %.1f%%, edge: %.1f%%) | "
            "equity $%.2f, cash after reserve $%.2f",
            amount,
            fighter,
            decimal_odds,
            model_prob * 100,
            market_prob * 100,
            (model_prob - market_prob) * 100,
            self.total_equity,
            self.bankroll,
        )

        return bet

    def release_bet(
        self,
        amount: float,
        fighter: str,
        reason: str = "",
    ) -> None:
        """Return reserved cash after an order is cancelled."""

        if amount <= 0:
            return

        self.bankroll += amount

        reason_suffix = f" ({reason})" if reason else ""
        logger.info(
            "RELEASED: $%.2f from %s%s | Available cash: $%.2f | Total equity: $%.2f",
            amount,
            fighter,
            reason_suffix,
            self.bankroll,
            self.total_equity,
        )

    def settle_bet(self, bet_index: int, won: bool) -> None:
        """Settle a bet and update both total equity and available cash."""

        bet = self.history[bet_index]
        stake = float(bet["amount"])

        if won:
            proceeds = stake * float(bet["decimal_odds"])
            profit = proceeds - stake
            self.bankroll += proceeds
            self.total_equity += profit
            bet["profit"] = profit
            bet["result"] = "win"
        else:
            self.total_equity -= stake
            bet["profit"] = -stake
            bet["result"] = "loss"

        bet["settled_at"] = datetime.now().isoformat()
        bet["bankroll_after"] = self.total_equity
        bet["available_cash_after"] = self.bankroll
        self.peak_bankroll = max(self.peak_bankroll, self.total_equity)

        logger.info(
            "SETTLED: %s %s | P&L: $%+.2f | Total equity: $%.2f | Available cash: $%.2f",
            bet["fighter"],
            "WON" if won else "LOST",
            bet["profit"],
            self.total_equity,
            self.bankroll,
        )

    def get_stats(self) -> dict:
        """Get bankroll statistics based on total equity."""

        settled = [bet for bet in self.history if bet["result"] is not None]
        if not settled:
            return {
                "bankroll": self.total_equity,
                "total_equity": self.total_equity,
                "available_cash": self.bankroll,
                "initial_bankroll": self.initial_bankroll,
                "total_bets": 0,
                "roi": 0.0,
            }

        wins = sum(1 for bet in settled if bet["result"] == "win")
        total_wagered = sum(float(bet["amount"]) for bet in settled)
        total_profit = sum(float(bet["profit"]) for bet in settled)
        ordered_settled = sorted(
            settled,
            key=lambda bet: bet.get("settled_at") or bet.get("placed_at") or "",
        )
        peak = self.initial_bankroll
        max_drawdown = 0.0
        for bet in ordered_settled:
            before = float(bet.get("bankroll_before", peak) or peak)
            if peak > 0:
                max_drawdown = max(max_drawdown, 1 - (before / peak))
            after = float(
                bet.get("bankroll_after")
                if bet.get("bankroll_after") is not None
                else before + float(bet.get("profit") or 0.0)
            )
            if peak > 0:
                max_drawdown = max(max_drawdown, 1 - (after / peak))
            peak = max(peak, after)

        return {
            "bankroll": self.total_equity,
            "total_equity": self.total_equity,
            "available_cash": self.bankroll,
            "initial_bankroll": self.initial_bankroll,
            "total_bets": len(settled),
            "wins": wins,
            "losses": len(settled) - wins,
            "win_rate": wins / len(settled) if settled else 0.0,
            "total_wagered": total_wagered,
            "total_profit": total_profit,
            "roi": total_profit / total_wagered if total_wagered > 0 else 0.0,
            "bankroll_change": self.total_equity - self.initial_bankroll,
            "bankroll_change_pct": (
                (self.total_equity - self.initial_bankroll) / self.initial_bankroll
                if self.initial_bankroll > 0
                else 0.0
            ),
            "peak_bankroll": self.peak_bankroll,
            "max_drawdown": max_drawdown,
            "current_drawdown": self.current_drawdown,
            "avg_edge": sum(float(bet["edge"]) for bet in settled) / len(settled),
            "avg_bet_size": total_wagered / len(settled),
        }
