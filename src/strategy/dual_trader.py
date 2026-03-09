"""
Triple-trader coordination layer — runs three independent strategies
on the same Polymarket wallet with automatic bankroll splitting and conflict resolution.

Trader A ("Conservative"): BLEND_WEIGHT = 0.20 — fewer, higher-conviction value bets
Trader B ("Aggressive"):   BLEND_WEIGHT = 0.40 — more value bets, trusts model more
Trader C ("Conviction"):   No blend — bets on fighters ALL signals agree will win

Coordination rules:
  1. Wallet is split 40/40/20 (A/B/C) regardless of balance
  2. Never bet opposite sides of the same fight across ANY traders
  3. If multiple traders want the same side, the one with higher edge/conviction takes it
  4. Each trader has its own ledger and bankroll tracking
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
    MIN_EDGE_THRESHOLD,
    KELLY_FRACTION,
    MAX_BET_FRACTION,
    STOP_LOSS_FRACTION,
    CONVICTION_MAX_BET_FRACTION,
    LOGS_DIR,
)

logger = logging.getLogger(__name__)

TRADER_A_BLEND = 0.20
TRADER_B_BLEND = 0.40

# Bankroll split: A gets 40%, B gets 40%, C gets 20%
TRADER_A_SHARE = 0.40
TRADER_B_SHARE = 0.40
TRADER_C_SHARE = 0.20

TRADER_A_LEDGER = LOGS_DIR / "bet_ledger_trader_a.json"
TRADER_B_LEDGER = LOGS_DIR / "bet_ledger_trader_b.json"
TRADER_C_LEDGER = LOGS_DIR / "bet_ledger_trader_c.json"


@dataclass
class TraderProfile:
    """Configuration for a single trader instance."""
    name: str
    blend_weight: float
    ledger_path: str
    bankroll: Optional[BankrollManager] = None
    executor: Optional[OrderExecutor] = None
    value_bets: Optional[pd.DataFrame] = None


def _split_bankroll(dry_run: bool = True) -> tuple[float, float, float]:
    """
    Fetch the total Polymarket balance and split across three traders.

    A: 40%, B: 40%, C: 20%. Called fresh each run so deposits are
    automatically detected and split.
    """
    if dry_run:
        from src.config import INITIAL_BANKROLL
        total = INITIAL_BANKROLL
    else:
        total = _fetch_polymarket_balance()
        if total <= 0:
            from src.config import INITIAL_BANKROLL
            total = INITIAL_BANKROLL

    alloc_a = round(total * TRADER_A_SHARE, 2)
    alloc_b = round(total * TRADER_B_SHARE, 2)
    alloc_c = round(total * TRADER_C_SHARE, 2)
    logger.info(
        f"Wallet balance: ${total:.2f} -> "
        f"Trader A (conservative): ${alloc_a:.2f} | "
        f"Trader B (aggressive): ${alloc_b:.2f} | "
        f"Trader C (conviction): ${alloc_c:.2f}"
    )
    return alloc_a, alloc_b, alloc_c


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


def _resolve_conflicts(
    bets_a: pd.DataFrame,
    bets_b: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Resolve betting conflicts between the two value traders.

    Rules:
      1. If both want OPPOSITE sides of the same fight -> neither bets (cancel both)
      2. If both want the SAME side -> higher edge takes it (other skips)
      3. No conflict -> both proceed independently
    """
    if bets_a.empty or bets_b.empty:
        return bets_a, bets_b

    a_keys = {_fight_key(row): idx for idx, (_, row) in enumerate(bets_a.iterrows())}
    b_keys = {_fight_key(row): idx for idx, (_, row) in enumerate(bets_b.iterrows())}

    drop_a = set()
    drop_b = set()

    for key in set(a_keys.keys()) & set(b_keys.keys()):
        a_idx = a_keys[key]
        b_idx = b_keys[key]
        row_a = bets_a.iloc[a_idx]
        row_b = bets_b.iloc[b_idx]

        same_side = row_a["bet_side"] == row_b["bet_side"]

        if not same_side:
            logger.warning(
                f"CONFLICT: {row_a.get('bet_on', '?')} vs {row_b.get('bet_on', '?')} "
                f"on fight {key} — opposite sides, cancelling both"
            )
            drop_a.add(a_idx)
            drop_b.add(b_idx)
        else:
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

    if drop_a:
        bets_a = bets_a.drop(bets_a.index[list(drop_a)])
    if drop_b:
        bets_b = bets_b.drop(bets_b.index[list(drop_b)])

    return bets_a, bets_b


def _resolve_conviction_conflicts(
    bets_c: pd.DataFrame,
    bets_a: pd.DataFrame,
    bets_b: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resolve conflicts between Trader C (conviction) and value traders A/B.

    Rules:
      - If C bets the SAME side as A or B on a fight, C proceeds (independent strategy).
        Both traders can bet the same side since they use separate bankrolls.
      - If C bets the OPPOSITE side from A or B, C is cancelled on that fight.
        Conviction bets should not contradict value signals.
    """
    if bets_c.empty:
        return bets_c

    # Collect all value bet fight keys and their sides
    value_fights = {}  # fight_key -> bet_side
    for bets_df, trader_label in [(bets_a, "A"), (bets_b, "B")]:
        if bets_df.empty:
            continue
        for _, row in bets_df.iterrows():
            key = _fight_key(row)
            value_fights[key] = {
                "side": row["bet_side"],
                "trader": trader_label,
                "fighter": row.get("bet_on", "?"),
            }

    drop_c = set()
    for idx, (_, row_c) in enumerate(bets_c.iterrows()):
        key = _fight_key(row_c)
        if key in value_fights:
            vf = value_fights[key]
            if row_c["bet_side"] != vf["side"]:
                logger.warning(
                    f"CONVICTION CONFLICT: Trader C wants {row_c.get('bet_on', '?')} "
                    f"but Trader {vf['trader']} wants {vf['fighter']} "
                    f"on fight {key} — cancelling conviction bet"
                )
                drop_c.add(idx)
            else:
                logger.info(
                    f"CONVICTION AGREEMENT: Trader C and Trader {vf['trader']} "
                    f"both want {row_c.get('bet_on', '?')} on fight {key} — "
                    f"both proceed independently"
                )

    if drop_c:
        bets_c = bets_c.drop(bets_c.index[list(drop_c)])

    return bets_c


def run_dual_traders(
    predictions: pd.DataFrame,
    markets: pd.DataFrame,
    clob: Optional[ClobClientWrapper] = None,
    dry_run: bool = True,
    min_edge: float = MIN_EDGE_THRESHOLD,
) -> dict:
    """
    Run all three traders on the same set of predictions and markets.

    1. Split wallet 40/40/20 (A/B/C)
    2. Traders A & B find value bets using their own blend weights
    3. Trader C finds conviction bets (triple-agreement, no edge required)
    4. Resolve conflicts across all traders
    5. Execute remaining bets independently

    Returns dict with results from all traders.
    """
    # 1. Split the wallet
    alloc_a, alloc_b, alloc_c = _split_bankroll(dry_run=dry_run)

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
    trader_c = TraderProfile(
        name="Trader C (Conviction)",
        blend_weight=0.0,  # Not used — conviction doesn't blend
        ledger_path=TRADER_C_LEDGER,
    )

    trader_a = _create_trader(trader_a, alloc_a, clob, dry_run)
    trader_b = _create_trader(trader_b, alloc_b, clob, dry_run)
    # Trader C uses flat sizing, so higher max_bet_fraction but no Kelly
    trader_c = _create_trader(
        trader_c, alloc_c, clob, dry_run,
        kelly_fraction=1.0,  # Not really used — conviction_bet_size handles sizing
        max_bet_fraction=CONVICTION_MAX_BET_FRACTION,
    )

    logger.info(
        f"\n{'='*60}\n"
        f"TRIPLE TRADER MODE\n"
        f"  {trader_a.name}: ${trader_a.bankroll.bankroll:.2f}\n"
        f"  {trader_b.name}: ${trader_b.bankroll.bankroll:.2f}\n"
        f"  {trader_c.name}: ${trader_c.bankroll.bankroll:.2f}\n"
        f"{'='*60}"
    )

    # 3. Match predictions to markets
    matched_a = trader_a.executor._match_predictions_to_markets(predictions, markets)
    matched_b = trader_b.executor._match_predictions_to_markets(predictions, markets)
    matched_c = trader_c.executor._match_predictions_to_markets(predictions, markets)

    # 4. Find bets for each trader
    bets_a = find_value_bets(matched_a, min_edge=min_edge, blend_weight=TRADER_A_BLEND)
    bets_b = find_value_bets(matched_b, min_edge=min_edge, blend_weight=TRADER_B_BLEND)
    bets_c = find_conviction_bets(matched_c)

    logger.info(
        f"\nPre-coordination:\n"
        f"  {trader_a.name}: {len(bets_a)} value bets\n"
        f"  {trader_b.name}: {len(bets_b)} value bets\n"
        f"  {trader_c.name}: {len(bets_c)} conviction bets"
    )

    # 5. Resolve conflicts — value traders first, then conviction vs value
    bets_a, bets_b = _resolve_conflicts(bets_a, bets_b)
    bets_c = _resolve_conviction_conflicts(bets_c, bets_a, bets_b)

    logger.info(
        f"\nPost-coordination:\n"
        f"  {trader_a.name}: {len(bets_a)} bets\n"
        f"  {trader_b.name}: {len(bets_b)} bets\n"
        f"  {trader_c.name}: {len(bets_c)} bets"
    )

    # 6. Execute bets for each trader
    orders_a = []
    orders_b = []
    orders_c = []

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

    if not bets_c.empty:
        logger.info(f"\n--- Executing {trader_c.name} ---")
        for _, bet in bets_c.iterrows():
            # Stop-loss check (conviction bets bypass Kelly, so check manually)
            if trader_c.bankroll.is_stopped:
                logger.warning(
                    f"  Trader C stop-loss triggered — skipping remaining conviction bets"
                )
                break

            # Override bet size using conviction sizing instead of Kelly
            bet_size = conviction_bet_size(
                model_prob=bet["model_prob"],
                bankroll=trader_c.bankroll.bankroll,
            )
            if bet_size <= 0:
                logger.info(
                    f"  Skipping conviction bet on {bet.get('bet_on', '?')}: "
                    f"bet size too small (bankroll: ${trader_c.bankroll.bankroll:.2f})"
                )
                continue

            # Inject the conviction bet size so _place_bet uses it
            bet_with_size = bet.copy()
            bet_with_size["override_bet_size"] = bet_size
            order = trader_c.executor._place_bet(bet_with_size, markets)
            if order:
                order["trader"] = "C"
                order["conviction_score"] = bet.get("conviction_score", 0)
                orders_c.append(order)

    # 7. Summary
    total_orders = len(orders_a) + len(orders_b) + len(orders_c)
    total_wagered_a = sum(o.get("bet_size_usd", 0) for o in orders_a)
    total_wagered_b = sum(o.get("bet_size_usd", 0) for o in orders_b)
    total_wagered_c = sum(o.get("bet_size_usd", 0) for o in orders_c)

    logger.info(
        f"\n{'='*60}\n"
        f"TRIPLE TRADER EXECUTION SUMMARY\n"
        f"{'='*60}\n"
        f"  {trader_a.name}:\n"
        f"    Orders: {len(orders_a)} | Wagered: ${total_wagered_a:.2f} | "
        f"Bankroll remaining: ${trader_a.bankroll.bankroll:.2f}\n"
        f"  {trader_b.name}:\n"
        f"    Orders: {len(orders_b)} | Wagered: ${total_wagered_b:.2f} | "
        f"Bankroll remaining: ${trader_b.bankroll.bankroll:.2f}\n"
        f"  {trader_c.name}:\n"
        f"    Orders: {len(orders_c)} | Wagered: ${total_wagered_c:.2f} | "
        f"Bankroll remaining: ${trader_c.bankroll.bankroll:.2f}\n"
        f"  Combined: {total_orders} orders, "
        f"${total_wagered_a + total_wagered_b + total_wagered_c:.2f} wagered\n"
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
        "trader_c": {
            "name": trader_c.name,
            "blend_weight": 0.0,
            "allocation": alloc_c,
            "orders": orders_c,
            "total_wagered": total_wagered_c,
            "bankroll_remaining": trader_c.bankroll.bankroll,
            "stats": trader_c.bankroll.get_stats(),
        },
        "total_orders": total_orders,
        "conflicts_resolved": True,
    }
