"""
Dual-trader coordination layer — runs two independent blend-weight strategies
on the same Polymarket wallet with automatic bankroll splitting and conflict resolution.

Trader A ("Conservative"): BLEND_WEIGHT = 0.20 — fewer, higher-conviction bets
Trader B ("Aggressive"):   BLEND_WEIGHT = 0.40 — more bets, trusts model more

Coordination rules:
  1. Wallet is always split 50/50 regardless of balance
  2. Never bet opposite sides of the same fight
  3. If both want the same side, the one with higher edge takes it
  4. Each trader has its own ledger and bankroll tracking
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.polymarket.client import ClobClientWrapper
from src.polymarket.executor import OrderExecutor
from src.polymarket.tracker import BetLedger
from src.strategy.bankroll import BankrollManager, _fetch_polymarket_balance
from src.strategy.value import find_value_bets
from src.config import (
    MIN_EDGE_THRESHOLD,
    KELLY_FRACTION,
    MAX_BET_FRACTION,
    STOP_LOSS_FRACTION,
    LOGS_DIR,
)

logger = logging.getLogger(__name__)

TRADER_A_BLEND = 0.20
TRADER_B_BLEND = 0.40

TRADER_A_LEDGER = LOGS_DIR / "bet_ledger_trader_a.json"
TRADER_B_LEDGER = LOGS_DIR / "bet_ledger_trader_b.json"


@dataclass
class TraderProfile:
    """Configuration for a single trader instance."""
    name: str
    blend_weight: float
    ledger_path: str
    bankroll: Optional[BankrollManager] = None
    executor: Optional[OrderExecutor] = None
    value_bets: Optional[pd.DataFrame] = None


def _split_bankroll(dry_run: bool = True) -> tuple[float, float]:
    """
    Fetch the total Polymarket balance and split 50/50.

    Each trader gets exactly half. This is called fresh each run
    so deposits are automatically detected and split.
    """
    if dry_run:
        # In dry-run, use a default or config value
        from src.config import INITIAL_BANKROLL
        total = INITIAL_BANKROLL
    else:
        total = _fetch_polymarket_balance()
        if total <= 0:
            from src.config import INITIAL_BANKROLL
            total = INITIAL_BANKROLL

    half = round(total / 2, 2)
    logger.info(
        f"Wallet balance: ${total:.2f} -> "
        f"Trader A (conservative): ${half:.2f} | "
        f"Trader B (aggressive): ${half:.2f}"
    )
    return half, half


def _create_trader(
    profile: TraderProfile,
    allocation: float,
    clob: Optional[ClobClientWrapper],
    dry_run: bool,
) -> TraderProfile:
    """Initialize bankroll and executor for a trader."""
    # Create trader-specific bankroll (no auto-detect — we control the split)
    bankroll = BankrollManager(
        initial_bankroll=allocation,
        kelly_fraction=KELLY_FRACTION,
        max_bet_fraction=MAX_BET_FRACTION,
        stop_loss_fraction=STOP_LOSS_FRACTION,
        auto_detect_balance=False,
    )

    # Create trader-specific ledger and executor
    ledger = BetLedger(path=profile.ledger_path)
    executor = OrderExecutor(
        bankroll=bankroll,
        clob_client=clob,
        dry_run=dry_run,
    )
    # Override the executor's default ledger with the trader-specific one
    executor.ledger = ledger

    # Sync bankroll from this trader's own ledger
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


def _resolve_conflicts(
    bets_a: pd.DataFrame,
    bets_b: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Resolve betting conflicts between the two traders.

    Rules:
      1. If both want OPPOSITE sides of the same fight → neither bets (cancel both)
      2. If both want the SAME side → higher edge takes it (other skips)
      3. No conflict → both proceed independently
    """
    if bets_a.empty or bets_b.empty:
        return bets_a, bets_b

    # Build fight keys for matching
    def fight_key(row):
        fighters = sorted([
            str(row.get("fighter_a", "")).lower(),
            str(row.get("fighter_b", "")).lower(),
        ])
        return f"{fighters[0]}|{fighters[1]}"

    a_keys = {fight_key(row): idx for idx, (_, row) in enumerate(bets_a.iterrows())}
    b_keys = {fight_key(row): idx for idx, (_, row) in enumerate(bets_b.iterrows())}

    drop_a = set()
    drop_b = set()

    for key in set(a_keys.keys()) & set(b_keys.keys()):
        a_idx = a_keys[key]
        b_idx = b_keys[key]
        row_a = bets_a.iloc[a_idx]
        row_b = bets_b.iloc[b_idx]

        same_side = row_a["bet_side"] == row_b["bet_side"]

        if not same_side:
            # OPPOSITE SIDES — cancel both to avoid guaranteed loss
            logger.warning(
                f"CONFLICT: {row_a.get('bet_on', '?')} vs {row_b.get('bet_on', '?')} "
                f"on fight {key} — opposite sides, cancelling both"
            )
            drop_a.add(a_idx)
            drop_b.add(b_idx)
        else:
            # SAME SIDE — higher edge wins
            edge_a = row_a.get("edge", 0)
            edge_b = row_b.get("edge", 0)
            if edge_a >= edge_b:
                logger.info(
                    f"OVERLAP: Both want {row_a.get('bet_on', '?')} — "
                    f"Trader A edge {edge_a:.1%} >= Trader B {edge_b:.1%}, "
                    f"Trader A takes it"
                )
                drop_b.add(b_idx)
            else:
                logger.info(
                    f"OVERLAP: Both want {row_b.get('bet_on', '?')} — "
                    f"Trader B edge {edge_b:.1%} > Trader A {edge_a:.1%}, "
                    f"Trader B takes it"
                )
                drop_a.add(a_idx)

    # Drop conflicted rows
    if drop_a:
        bets_a = bets_a.drop(bets_a.index[list(drop_a)])
    if drop_b:
        bets_b = bets_b.drop(bets_b.index[list(drop_b)])

    return bets_a, bets_b


def run_dual_traders(
    predictions: pd.DataFrame,
    markets: pd.DataFrame,
    clob: Optional[ClobClientWrapper] = None,
    dry_run: bool = True,
    min_edge: float = MIN_EDGE_THRESHOLD,
) -> dict:
    """
    Run both traders on the same set of predictions and markets.

    1. Split wallet 50/50
    2. Each trader finds value bets using its own blend weight
    3. Resolve conflicts (opposite sides / overlaps)
    4. Execute remaining bets independently

    Returns dict with results from both traders.
    """
    # 1. Split the wallet
    alloc_a, alloc_b = _split_bankroll(dry_run=dry_run)

    # 2. Create trader profiles
    trader_a = TraderProfile(
        name="Trader A (Conservative, blend=0.20)",
        blend_weight=TRADER_A_BLEND,
        ledger_path=TRADER_A_LEDGER,
    )
    trader_b = TraderProfile(
        name="Trader B (Aggressive, blend=0.40)",
        blend_weight=TRADER_B_BLEND,
        ledger_path=TRADER_B_LEDGER,
    )

    trader_a = _create_trader(trader_a, alloc_a, clob, dry_run)
    trader_b = _create_trader(trader_b, alloc_b, clob, dry_run)

    logger.info(
        f"\n{'='*60}\n"
        f"DUAL TRADER MODE\n"
        f"  {trader_a.name}: ${trader_a.bankroll.bankroll:.2f}\n"
        f"  {trader_b.name}: ${trader_b.bankroll.bankroll:.2f}\n"
        f"{'='*60}"
    )

    # 3. Match predictions to markets for each trader
    #    We need to find value bets using each trader's blend weight
    matched_a = trader_a.executor._match_predictions_to_markets(predictions, markets)
    matched_b = trader_b.executor._match_predictions_to_markets(predictions, markets)

    # 4. Find value bets with each trader's blend weight
    bets_a = find_value_bets(matched_a, min_edge=min_edge, blend_weight=TRADER_A_BLEND)
    bets_b = find_value_bets(matched_b, min_edge=min_edge, blend_weight=TRADER_B_BLEND)

    logger.info(
        f"\nPre-coordination:\n"
        f"  {trader_a.name}: {len(bets_a)} value bets\n"
        f"  {trader_b.name}: {len(bets_b)} value bets"
    )

    # 5. Resolve conflicts
    bets_a, bets_b = _resolve_conflicts(bets_a, bets_b)

    logger.info(
        f"\nPost-coordination:\n"
        f"  {trader_a.name}: {len(bets_a)} bets\n"
        f"  {trader_b.name}: {len(bets_b)} bets"
    )

    # 6. Execute bets for each trader
    orders_a = []
    orders_b = []

    if not bets_a.empty:
        logger.info(f"\n--- Executing {trader_a.name} ---")
        for _, bet in bets_a.iterrows():
            order = trader_a.executor._place_bet(bet, markets)
            if order:
                order["trader"] = "A"
                orders_a.append(order)

    if not bets_b.empty:
        logger.info(f"\n--- Executing {trader_b.name} ---")
        for _, bet in bets_b.iterrows():
            order = trader_b.executor._place_bet(bet, markets)
            if order:
                order["trader"] = "B"
                orders_b.append(order)

    # 7. Summary
    total_orders = len(orders_a) + len(orders_b)
    total_wagered_a = sum(o.get("bet_size_usd", 0) for o in orders_a)
    total_wagered_b = sum(o.get("bet_size_usd", 0) for o in orders_b)

    logger.info(
        f"\n{'='*60}\n"
        f"DUAL TRADER EXECUTION SUMMARY\n"
        f"{'='*60}\n"
        f"  {trader_a.name}:\n"
        f"    Orders: {len(orders_a)} | Wagered: ${total_wagered_a:.2f} | "
        f"Bankroll remaining: ${trader_a.bankroll.bankroll:.2f}\n"
        f"  {trader_b.name}:\n"
        f"    Orders: {len(orders_b)} | Wagered: ${total_wagered_b:.2f} | "
        f"Bankroll remaining: ${trader_b.bankroll.bankroll:.2f}\n"
        f"  Combined: {total_orders} orders, "
        f"${total_wagered_a + total_wagered_b:.2f} wagered\n"
        f"{'='*60}"
    )

    return {
        "trader_a": {
            "name": trader_a.name,
            "blend_weight": TRADER_A_BLEND,
            "allocation": alloc_a,
            "orders": orders_a,
            "total_wagered": total_wagered_a,
            "bankroll_remaining": trader_a.bankroll.bankroll,
            "stats": trader_a.bankroll.get_stats(),
        },
        "trader_b": {
            "name": trader_b.name,
            "blend_weight": TRADER_B_BLEND,
            "allocation": alloc_b,
            "orders": orders_b,
            "total_wagered": total_wagered_b,
            "bankroll_remaining": trader_b.bankroll.bankroll,
            "stats": trader_b.bankroll.get_stats(),
        },
        "total_orders": total_orders,
        "conflicts_resolved": True,
    }
