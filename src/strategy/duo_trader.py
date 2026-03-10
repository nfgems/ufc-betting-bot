"""
Duo-trader coordination layer — runs two independent strategies (S + C)
on the same Polymarket wallet.

Single Trader (S):     BLEND_WEIGHT = 0.30 — Kelly-sized value bets on full bankroll
Conviction Trader (C): No blend — bets on fighters ALL signals agree will win

Coordination rules:
  1. S evaluates first with the full wallet balance
  2. C gets the remaining bankroll after S's bets are placed
  3. If S bets a fight, C skips that fight entirely (no double-betting)
  4. Each trader has its own ledger for tracking
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.polymarket.client import ClobClientWrapper
from src.polymarket.executor import OrderExecutor
from src.polymarket.tracker import BetLedger
from src.strategy.bankroll import BankrollManager, _fetch_polymarket_balance
from src.strategy.value import find_value_bets, find_conviction_bets, conviction_bet_size
from src.config import (
    BLEND_WEIGHT,
    MIN_EDGE_THRESHOLD,
    KELLY_FRACTION,
    MAX_BET_FRACTION,
    STOP_LOSS_FRACTION,
    CONVICTION_MAX_BET_FRACTION,
    TRADER_C_SHARE,
    LOGS_DIR,
)

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


def _get_total_bankroll(dry_run: bool = True) -> float:
    """Fetch the total Polymarket balance (no split — full amount)."""
    if dry_run:
        from src.config import INITIAL_BANKROLL
        total = INITIAL_BANKROLL
    else:
        total = _fetch_polymarket_balance()
        if total <= 0:
            from src.config import INITIAL_BANKROLL
            total = INITIAL_BANKROLL

    logger.info(f"Wallet balance: ${total:.2f} (full bankroll for Single Trader)")
    return total


def _create_trader(
    profile: TraderProfile,
    allocation: float,
    clob: Optional[ClobClientWrapper],
    dry_run: bool,
    kelly_fraction: float = KELLY_FRACTION,
    max_bet_fraction: float = MAX_BET_FRACTION,
) -> TraderProfile:
    """Initialize bankroll and executor for a trader."""
    bankroll = BankrollManager(
        initial_bankroll=allocation,
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
    )
    executor.ledger = ledger

    _sync_bankroll_from_trader_ledger(bankroll, ledger)

    profile.bankroll = bankroll
    profile.executor = executor
    return profile


def _sync_bankroll_from_trader_ledger(
    bankroll: BankrollManager, ledger: BetLedger
) -> None:
    """Replay a trader-specific ledger to adjust bankroll for open/settled bets."""
    real_bets = [b for b in ledger.bets if not b.get("dry_run", True)]
    if not real_bets:
        return

    total_wagered = 0.0
    realized_pnl = 0.0

    for bet in real_bets:
        amount = bet.get("amount", 0)
        status = bet.get("status", "open")

        if status == "open":
            total_wagered += amount
        elif status == "won":
            realized_pnl += bet.get("result_pnl", 0) or 0
        elif status == "lost":
            realized_pnl += bet.get("result_pnl", 0) or 0

    bankroll.bankroll = bankroll.initial_bankroll + realized_pnl - total_wagered
    bankroll.peak_bankroll = max(bankroll.peak_bankroll, bankroll.initial_bankroll + realized_pnl)

    logger.info(
        f"  Synced from ledger: ${bankroll.bankroll:.2f} "
        f"(initial: ${bankroll.initial_bankroll:.2f}, "
        f"realized P&L: ${realized_pnl:+.2f}, "
        f"open bets: ${total_wagered:.2f})"
    )


def _fight_key(row) -> str:
    """Create a canonical fight key for conflict resolution."""
    fighters = sorted([
        str(row.get("fighter_a", "")).lower(),
        str(row.get("fighter_b", "")).lower(),
    ])
    return f"{fighters[0]}|{fighters[1]}"


def run_duo_traders(
    predictions: pd.DataFrame,
    markets: pd.DataFrame,
    clob: Optional[ClobClientWrapper] = None,
    dry_run: bool = True,
    min_edge: float = MIN_EDGE_THRESHOLD,
) -> dict:
    """
    Run S+C duo traders on the same set of predictions and markets.

    1. Single Trader (S) evaluates first with the full wallet balance
    2. S finds and executes value bets (blend=0.30, Kelly sizing)
    3. Conviction Trader (C) gets remaining bankroll after S
    4. C finds conviction bets, skipping any fights S already bet
    5. C executes with conviction sizing

    Returns dict with results from both traders.
    """
    # 1. Get total balance (no split)
    total = _get_total_bankroll(dry_run=dry_run)

    # 2. Create Single Trader with full bankroll
    single = _create_trader(
        TraderProfile(
            name="Single Trader (S, blend=0.30)",
            blend_weight=BLEND_WEIGHT,
            ledger_path=SINGLE_LEDGER,
        ),
        allocation=total,
        clob=clob,
        dry_run=dry_run,
    )

    logger.info(
        f"\n{'='*60}\n"
        f"DUO TRADER MODE\n"
        f"  {single.name}: ${single.bankroll.bankroll:.2f} (full bankroll)\n"
        f"{'='*60}"
    )

    # 3. Match predictions to markets and find S value bets
    matched_s = single.executor._match_predictions_to_markets(predictions, markets)
    value_bets = find_value_bets(matched_s, min_edge=min_edge, blend_weight=BLEND_WEIGHT)

    logger.info(f"\n{single.name}: {len(value_bets)} value bets found")

    # 4. Execute S bets, track which fights were bet on
    s_orders = []
    s_fight_keys = set()

    if not value_bets.empty:
        logger.info(f"\n--- Executing {single.name} ---")
        for _, bet in value_bets.iterrows():
            order = single.executor._place_bet(bet, markets)
            if order:
                order["trader"] = "S"
                s_orders.append(order)
                s_fight_keys.add(_fight_key(bet))

    # 5. Create Conviction Trader with remaining bankroll
    remaining = single.bankroll.bankroll  # After S bets are deducted
    conv_allocation = remaining * TRADER_C_SHARE

    conv = _create_trader(
        TraderProfile(
            name="Conviction Trader (C)",
            blend_weight=0.0,  # Not used — conviction doesn't blend
            ledger_path=CONVICTION_LEDGER,
        ),
        allocation=conv_allocation,
        clob=clob,
        dry_run=dry_run,
        kelly_fraction=1.0,  # Not really used — conviction_bet_size handles sizing
        max_bet_fraction=CONVICTION_MAX_BET_FRACTION,
    )

    logger.info(
        f"\n  {conv.name}: ${conv.bankroll.bankroll:.2f} "
        f"(remaining after S bets)"
    )

    # 6. Find conviction bets, skip fights S already bet
    matched_c = conv.executor._match_predictions_to_markets(predictions, markets)
    conviction_bets = find_conviction_bets(matched_c)

    logger.info(f"  {conv.name}: {len(conviction_bets)} conviction bets found")

    c_orders = []

    if not conviction_bets.empty:
        logger.info(f"\n--- Executing {conv.name} ---")
        for _, bet in conviction_bets.iterrows():
            # Double-bet prevention: skip fights S already bet
            if _fight_key(bet) in s_fight_keys:
                logger.info(
                    f"  Skipping conviction bet on {bet.get('bet_on', '?')} "
                    f"— Single Trader already bet this fight"
                )
                continue

            # Stop-loss check
            if conv.bankroll.is_stopped:
                logger.warning(
                    "  Conviction Trader stop-loss triggered — skipping remaining bets"
                )
                break

            # Override bet size using conviction sizing instead of Kelly
            bet_size = conviction_bet_size(
                model_prob=bet["model_prob"],
                bankroll=conv.bankroll.bankroll,
            )
            if bet_size <= 0:
                logger.info(
                    f"  Skipping conviction bet on {bet.get('bet_on', '?')}: "
                    f"bet size too small (bankroll: ${conv.bankroll.bankroll:.2f})"
                )
                continue

            # Inject the conviction bet size so _place_bet uses it
            bet_with_size = bet.copy()
            bet_with_size["override_bet_size"] = bet_size
            order = conv.executor._place_bet(bet_with_size, markets)
            if order:
                order["trader"] = "C"
                order["conviction_score"] = bet.get("conviction_score", 0)
                c_orders.append(order)

    # 7. Summary
    total_orders = len(s_orders) + len(c_orders)
    total_wagered_s = sum(o.get("bet_size_usd", 0) for o in s_orders)
    total_wagered_c = sum(o.get("bet_size_usd", 0) for o in c_orders)

    logger.info(
        f"\n{'='*60}\n"
        f"DUO TRADER EXECUTION SUMMARY\n"
        f"{'='*60}\n"
        f"  {single.name}:\n"
        f"    Orders: {len(s_orders)} | Wagered: ${total_wagered_s:.2f} | "
        f"Bankroll remaining: ${single.bankroll.bankroll:.2f}\n"
        f"  {conv.name}:\n"
        f"    Orders: {len(c_orders)} | Wagered: ${total_wagered_c:.2f} | "
        f"Bankroll remaining: ${conv.bankroll.bankroll:.2f}\n"
        f"  Combined: {total_orders} orders, "
        f"${total_wagered_s + total_wagered_c:.2f} wagered\n"
        f"{'='*60}"
    )

    return {
        "trader_s": {
            "name": single.name,
            "blend_weight": BLEND_WEIGHT,
            "allocation": total,
            "orders": s_orders,
            "total_wagered": total_wagered_s,
            "bankroll_remaining": single.bankroll.bankroll,
            "stats": single.bankroll.get_stats(),
        },
        "trader_c": {
            "name": conv.name,
            "blend_weight": 0.0,
            "allocation": conv_allocation,
            "orders": c_orders,
            "total_wagered": total_wagered_c,
            "bankroll_remaining": conv.bankroll.bankroll,
            "stats": conv.bankroll.get_stats(),
        },
        "total_orders": total_orders,
    }
