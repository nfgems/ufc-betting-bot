"""
Duo-trader coordination layer for the shared Polymarket wallet.

Single Trader (S): value bets with Kelly sizing.
Conviction Trader (C): consensus bets sized conservatively.

Live-mode bankroll handling:
- size from total account equity
- gate order submission by available cash
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from src.config import (
    BLEND_WEIGHT,
    CONVICTION_MAX_BET_FRACTION,
    KELLY_FRACTION,
    LOGS_DIR,
    MAX_BET_FRACTION,
    MIN_EDGE_THRESHOLD,
    NEAR_MISS_MIN_EDGE,
    STOP_LOSS_FRACTION,
    TRADER_C_SHARE,
)
from src.polymarket.client import ClobClientWrapper
from src.polymarket.executor import OrderExecutor, assert_live_wallet_exposure_synced
from src.polymarket.tracker import BetLedger
from src.strategy.bankroll import BankrollManager, _fetch_polymarket_account_state
from src.strategy.value import conviction_bet_size, find_conviction_bets, find_value_bets

logger = logging.getLogger(__name__)

SINGLE_LEDGER = LOGS_DIR / "bet_ledger_single.json"
CONVICTION_LEDGER = LOGS_DIR / "bet_ledger_conviction.json"


@dataclass
class TraderProfile:
    """Configuration for a single trader instance."""

    name: str
    blend_weight: float
    ledger_path: str
    bankroll: Optional[BankrollManager] = None
    executor: Optional[OrderExecutor] = None
    value_bets: Optional[pd.DataFrame] = None


@dataclass(frozen=True)
class WalletBankrollBasis:
    """Authoritative account state for the current duo-trader cycle."""

    total_equity: float
    available_cash: float
    source: str


def _ledger_snapshot(ledger, *, open_only: bool = False, fresh: bool = False) -> list[dict]:
    getter_name = "get_open_bets" if open_only else "get_bets"
    attr_name = "open_bets" if open_only else "bets"
    getter = getattr(ledger, getter_name, None)
    if callable(getter):
        return getter(fresh=fresh)
    return [dict(bet) for bet in getattr(ledger, attr_name, [])]


def _bankroll_available_cash(bankroll) -> float:
    return float(
        getattr(
            bankroll,
            "available_cash",
            getattr(bankroll, "bankroll", 0.0),
        )
        or 0.0
    )


def _bankroll_total_equity(bankroll) -> float:
    return float(
        getattr(
            bankroll,
            "total_equity",
            getattr(bankroll, "bankroll", 0.0),
        )
        or 0.0
    )


def _resolve_total_bankroll(dry_run: bool = True) -> WalletBankrollBasis:
    """Resolve the wallet state the traders should use this cycle."""

    live_state = _fetch_polymarket_account_state(
        require_confirmed_cash=True,
        require_portfolio_value=True,
    )
    if live_state.get("confirmed_cash") and live_state.get("confirmed_portfolio"):
        return WalletBankrollBasis(
            total_equity=float(live_state.get("total_equity") or 0.0),
            available_cash=float(live_state.get("cash_balance") or 0.0),
            source="cash: Polymarket CLOB, portfolio: Polymarket Data API",
        )

    if dry_run:
        from src.config import INITIAL_BANKROLL

        return WalletBankrollBasis(
            total_equity=INITIAL_BANKROLL,
            available_cash=INITIAL_BANKROLL,
            source="INITIAL_BANKROLL fallback (dry-run only; live wallet state unavailable)",
        )

    raise RuntimeError(
        "Live mode: wallet cash balance and portfolio value could not be confirmed from "
        "Polymarket. Refusing to size bets against a synthetic bankroll."
    )


def _get_total_bankroll(dry_run: bool = True) -> float:
    """Return total equity for compatibility with older tests/callers."""

    basis = _resolve_total_bankroll(dry_run=dry_run)
    logger.info(
        "Wallet bankroll basis [%s]: equity $%.2f, cash $%.2f (source: %s)",
        "DRY RUN" if dry_run else "LIVE",
        basis.total_equity,
        basis.available_cash,
        basis.source,
    )
    return basis.total_equity


def _create_trader(
    profile: TraderProfile,
    allocation: float,
    clob: Optional[ClobClientWrapper],
    dry_run: bool,
    kelly_fraction: float = KELLY_FRACTION,
    max_bet_fraction: float = MAX_BET_FRACTION,
    sync_from_ledger: bool = False,
    available_cash: float | None = None,
    min_edge_threshold: float = MIN_EDGE_THRESHOLD,
    edge_scaling_base: float | None = None,
) -> TraderProfile:
    """Initialize bankroll and executor for a trader."""

    bankroll = BankrollManager(
        initial_bankroll=allocation,
        total_equity=allocation,
        available_cash=allocation if available_cash is None else available_cash,
        kelly_fraction=kelly_fraction,
        max_bet_fraction=max_bet_fraction,
        stop_loss_fraction=STOP_LOSS_FRACTION,
        auto_detect_balance=False,
    )

    ledger = BetLedger(path=profile.ledger_path)
    executor = OrderExecutor(
        bankroll=bankroll,
        clob_client=clob,
        dry_run=dry_run,
        min_edge_threshold=min_edge_threshold,
        edge_scaling_base=edge_scaling_base,
    )
    executor.ledger = ledger

    if sync_from_ledger:
        _sync_bankroll_from_trader_ledger(bankroll, ledger)

    profile.bankroll = bankroll
    profile.executor = executor
    return profile


def _sync_bankroll_from_trader_ledger(
    bankroll: BankrollManager, ledger: BetLedger
) -> None:
    """Replay a trader-specific ledger to adjust cash/equity."""

    real_bets = [
        bet for bet in _ledger_snapshot(ledger, fresh=True) if not bet.get("dry_run", True)
    ]
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

    bankroll.total_equity = bankroll.initial_bankroll + realized_pnl
    bankroll.bankroll = bankroll.total_equity - total_wagered
    bankroll.peak_bankroll = max(bankroll.peak_bankroll, bankroll.total_equity)

    logger.info(
        "  Synced from ledger: equity $%.2f, cash $%.2f "
        "(initial: $%.2f, realized P&L: %+0.2f, open bets: $%.2f)",
        bankroll.total_equity,
        bankroll.bankroll,
        bankroll.initial_bankroll,
        realized_pnl,
        total_wagered,
    )


def _fight_key(row) -> str:
    """Create a canonical fight key for conflict resolution."""

    fighters = sorted(
        [
            str(row.get("fighter_a", row.get("fighter", ""))).lower(),
            str(row.get("fighter_b", row.get("opponent", ""))).lower(),
        ]
    )
    return f"{fighters[0]}|{fighters[1]}"


def run_duo_traders(
    predictions: pd.DataFrame,
    markets: pd.DataFrame,
    clob: Optional[ClobClientWrapper] = None,
    dry_run: bool = True,
    min_edge: float = MIN_EDGE_THRESHOLD,
    features_by_fight: Optional[dict[str, dict]] = None,
    provenance_by_fight: Optional[dict[str, dict]] = None,
    event_title: str = "",
    existing_bets: Optional[list[dict]] = None,
    bankroll_basis: Optional[WalletBankrollBasis] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Run S+C duo traders on the same set of predictions and markets.

    S sizes from total equity, spends from available cash, and runs first.
    C then evaluates with its equity allocation and whatever cash remains free.
    """

    bankroll_basis = bankroll_basis or _resolve_total_bankroll(dry_run=dry_run)
    total_equity = bankroll_basis.total_equity
    available_cash = bankroll_basis.available_cash

    def _report_progress(message: str) -> None:
        if not callable(progress_callback):
            return
        try:
            progress_callback(message)
        except Exception as exc:
            logger.debug("Duo trader progress callback failed: %s", exc)

    logger.info(
        "Wallet bankroll basis [%s]: equity $%.2f, cash $%.2f (source: %s)",
        "DRY RUN" if dry_run else "LIVE",
        total_equity,
        available_cash,
        bankroll_basis.source,
    )

    if not dry_run and clob is not None:
        assert_live_wallet_exposure_synced(markets=markets, clob_client=clob)

    single = _create_trader(
        TraderProfile(
            name="Single Trader (S, blend=0.30)",
            blend_weight=BLEND_WEIGHT,
            ledger_path=SINGLE_LEDGER,
        ),
        allocation=total_equity,
        available_cash=available_cash,
        clob=clob,
        dry_run=dry_run,
        min_edge_threshold=min_edge,
        edge_scaling_base=min_edge,
    )

    logger.info(
        "\n%s\nDUO TRADER MODE (%s)\n  Wallet basis: equity $%.2f | cash $%.2f (%s)\n"
        "  %s: equity $%.2f | cash $%.2f (starting state)\n%s",
        "=" * 60,
        "DRY RUN" if dry_run else "LIVE",
        total_equity,
        available_cash,
        bankroll_basis.source,
        single.name,
        _bankroll_total_equity(single.bankroll),
        _bankroll_available_cash(single.bankroll),
        "=" * 60,
    )

    matched_s = single.executor._match_predictions_to_markets(predictions, markets)
    result = find_value_bets(
        matched_s,
        min_edge=min_edge,
        blend_weight=BLEND_WEIGHT,
        near_miss_min_edge=NEAR_MISS_MIN_EDGE,
        edge_scaling_base=min_edge,
    )
    if isinstance(result, tuple):
        value_bets, near_miss_bets = result
    else:
        value_bets, near_miss_bets = result, pd.DataFrame()

    # LLM Operator gate — evaluate value bets before execution
    from src.strategy.llm_operator import OPERATOR_ENABLED, evaluate_bets as operator_evaluate

    if OPERATOR_ENABLED and not value_bets.empty:
        logger.info("Running LLM Operator on %d value bets...", len(value_bets))
        _report_progress(f"Cycle active: running operator on {len(value_bets)} value bets")
        value_bets = operator_evaluate(
            value_bets,
            features_by_fight=features_by_fight,
            provenance_by_fight=provenance_by_fight,
            event_title=event_title,
            existing_bets=existing_bets,
            progress_callback=_report_progress,
            progress_label="value bets",
        )

    if OPERATOR_ENABLED and not near_miss_bets.empty:
        logger.info("Running LLM Operator on %d near-miss limit orders...", len(near_miss_bets))
        _report_progress(
            f"Cycle active: running operator on {len(near_miss_bets)} near-miss limit orders"
        )
        near_miss_bets = operator_evaluate(
            near_miss_bets,
            features_by_fight=features_by_fight,
            provenance_by_fight=provenance_by_fight,
            event_title=event_title,
            existing_bets=existing_bets,
            progress_callback=_report_progress,
            progress_label="near-miss limit orders",
        )

    single.executor.refresh_open_limit_orders(
        matched_predictions=matched_s,
        primary_bets=value_bets,
        limit_only_bets=near_miss_bets,
        trader_name=single.name,
    )

    logger.info("\n%s: %s value bets found", single.name, len(value_bets))
    _report_progress(f"Cycle active: executing {len(value_bets)} value bets for Single Trader")

    s_orders = []
    s_fight_keys = {
        _fight_key(bet)
        for bet in _ledger_snapshot(single.executor.ledger, open_only=True, fresh=True)
        if (not bet.get("dry_run")) or dry_run
    }

    if not value_bets.empty:
        logger.info("\n--- Executing %s ---", single.name)
        for _, bet in value_bets.iterrows():
            order = single.executor._place_bet(bet, markets)
            if order:
                order["trader"] = "S"
                s_orders.append(order)
                s_fight_keys.add(_fight_key(bet))

    nm_orders = []
    if not near_miss_bets.empty:
        logger.info(
            "\n--- %s: %s near-miss limit orders ---",
            single.name,
            len(near_miss_bets),
        )
        _report_progress(
            f"Cycle active: executing {len(near_miss_bets)} near-miss limit orders for Single Trader"
        )
        for _, bet in near_miss_bets.iterrows():
            if single.bankroll.is_stopped:
                logger.warning("  Stop-loss triggered - skipping remaining near-miss orders")
                break
            order = single.executor._place_near_miss_limit(bet, markets)
            if order:
                order["trader"] = "S"
                nm_orders.append(order)
                s_fight_keys.add(_fight_key(bet))

    remaining_cash = _bankroll_available_cash(single.bankroll)
    conv_equity_allocation = remaining_cash * TRADER_C_SHARE
    conv_cash_allocation = remaining_cash * TRADER_C_SHARE

    conv = _create_trader(
        TraderProfile(
            name="Conviction Trader (C)",
            blend_weight=0.0,
            ledger_path=CONVICTION_LEDGER,
        ),
        allocation=conv_equity_allocation,
        available_cash=conv_cash_allocation,
        clob=clob,
        dry_run=dry_run,
        kelly_fraction=1.0,
        max_bet_fraction=CONVICTION_MAX_BET_FRACTION,
        min_edge_threshold=min_edge,
        edge_scaling_base=min_edge,
    )

    logger.info(
        "\n  %s: equity $%.2f | cash $%.2f (after S reserved cash)",
        conv.name,
        _bankroll_total_equity(conv.bankroll),
        _bankroll_available_cash(conv.bankroll),
    )

    matched_c = conv.executor._match_predictions_to_markets(predictions, markets)
    conviction_bets = find_conviction_bets(matched_c, require_positive_ev=True)

    # LLM Operator gate — evaluate conviction bets before execution
    if OPERATOR_ENABLED and not conviction_bets.empty:
        logger.info("Running LLM Operator on %d conviction bets...", len(conviction_bets))
        _report_progress(f"Cycle active: running operator on {len(conviction_bets)} conviction bets")
        conviction_bets = operator_evaluate(
            conviction_bets,
            features_by_fight=features_by_fight,
            provenance_by_fight=provenance_by_fight,
            event_title=event_title,
            existing_bets=existing_bets,
            progress_callback=_report_progress,
            progress_label="conviction bets",
        )

    conv.executor.refresh_open_limit_orders(
        matched_predictions=matched_c,
        primary_bets=conviction_bets,
        trader_name=conv.name,
    )

    logger.info("  %s: %s conviction bets found", conv.name, len(conviction_bets))
    _report_progress(
        f"Cycle active: executing {len(conviction_bets)} conviction bets for Conviction Trader"
    )

    c_orders = []
    if not conviction_bets.empty:
        logger.info("\n--- Executing %s ---", conv.name)
        for _, bet in conviction_bets.iterrows():
            if _fight_key(bet) in s_fight_keys:
                logger.info(
                    "  Skipping conviction bet on %s - Single Trader already bet this fight",
                    bet.get("bet_on", "?"),
                )
                continue

            if conv.bankroll.is_stopped:
                logger.warning(
                    "  Conviction Trader stop-loss triggered - skipping remaining bets"
                )
                break

            bet_size = conviction_bet_size(
                model_prob=bet["model_prob"],
                bankroll=_bankroll_total_equity(conv.bankroll),
            )
            if bet_size <= 0:
                logger.info(
                    "  Skipping conviction bet on %s: bet size too small "
                    "(equity: $%.2f, cash: $%.2f)",
                    bet.get("bet_on", "?"),
                    _bankroll_total_equity(conv.bankroll),
                    _bankroll_available_cash(conv.bankroll),
                )
                continue

            bet_with_size = bet.copy()
            bet_with_size["override_bet_size"] = bet_size
            order = conv.executor._place_bet(bet_with_size, markets)
            if order:
                order["trader"] = "C"
                order["conviction_score"] = bet.get("conviction_score", 0)
                c_orders.append(order)

    total_orders = len(s_orders) + len(nm_orders) + len(c_orders)
    total_wagered_s = sum(order.get("bet_size_usd", 0) for order in s_orders)
    total_wagered_nm = sum(order.get("bet_size_usd", 0) for order in nm_orders)
    total_wagered_c = sum(order.get("bet_size_usd", 0) for order in c_orders)

    nm_line = ""
    if nm_orders:
        nm_line = (
            f"    Near-miss limits: {len(nm_orders)} | Reserved: ${total_wagered_nm:.2f}\n"
        )

    logger.info(
        "\n%s\nDUO TRADER EXECUTION SUMMARY\n%s\n"
        "  %s:\n"
        "    Orders: %s | Wagered: $%.2f | Cash remaining: $%.2f | Equity: $%.2f\n"
        "%s"
        "  %s:\n"
        "    Orders: %s | Wagered: $%.2f | Cash remaining: $%.2f | Equity: $%.2f\n"
        "  Combined: %s orders, $%.2f wagered\n%s",
        "=" * 60,
        "=" * 60,
        single.name,
        len(s_orders),
        total_wagered_s,
        _bankroll_available_cash(single.bankroll),
        _bankroll_total_equity(single.bankroll),
        nm_line,
        conv.name,
        len(c_orders),
        total_wagered_c,
        _bankroll_available_cash(conv.bankroll),
        _bankroll_total_equity(conv.bankroll),
        total_orders,
        total_wagered_s + total_wagered_nm + total_wagered_c,
        "=" * 60,
    )

    return {
        "trader_s": {
            "name": single.name,
            "blend_weight": BLEND_WEIGHT,
            "allocation": total_equity,
            "available_cash_start": available_cash,
            "orders": s_orders,
            "near_miss_orders": nm_orders,
            "total_wagered": total_wagered_s + total_wagered_nm,
            "bankroll_remaining": _bankroll_available_cash(single.bankroll),
            "total_equity": _bankroll_total_equity(single.bankroll),
            "stats": single.bankroll.get_stats(),
        },
        "trader_c": {
            "name": conv.name,
            "blend_weight": 0.0,
            "allocation": conv_equity_allocation,
            "available_cash_start": conv_cash_allocation,
            "orders": c_orders,
            "total_wagered": total_wagered_c,
            "bankroll_remaining": _bankroll_available_cash(conv.bankroll),
            "total_equity": _bankroll_total_equity(conv.bankroll),
            "stats": conv.bankroll.get_stats(),
        },
        "total_orders": total_orders,
    }
