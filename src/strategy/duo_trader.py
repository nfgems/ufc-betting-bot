"""
Duo-trader coordination layer for the shared Polymarket wallet.

Single Trader (S): value bets with Kelly sizing.
Conviction Trader (C): consensus bets sized conservatively.

Live-mode bankroll handling:
- size from total account equity
- gate order submission by available cash
"""

import hashlib
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
    TRACKER_MIN_HOURS_BEFORE_EVENT,
    TRADER_C_SHARE,
)
from src.data.name_utils import same_person_name
from src.polymarket.client import ClobClientWrapper
from src.polymarket.executor import OrderExecutor, assert_live_wallet_exposure_synced
from src.polymarket.tracker import BetLedger
from src.strategy.bankroll import BankrollManager, _fetch_polymarket_account_state
from src.strategy.value import (
    conviction_bet_size,
    find_conviction_bets,
    find_value_bets,
    implied_prob_to_decimal_odds,
)

logger = logging.getLogger(__name__)

SINGLE_LEDGER = LOGS_DIR / "bet_ledger_single.json"
CONVICTION_LEDGER = LOGS_DIR / "bet_ledger_conviction.json"
MODEL_TRACKER_LEDGER = LOGS_DIR / "bet_ledger_model_tracker.json"
GEMINI_TRACKER_LEDGER = LOGS_DIR / "bet_ledger_gemini_tracker.json"
ALL_TRADER_LEDGERS = [
    ("S", SINGLE_LEDGER),
    ("C", CONVICTION_LEDGER),
    ("M", MODEL_TRACKER_LEDGER),
    ("G", GEMINI_TRACKER_LEDGER),
]
_STATIC_ALL_TRADER_LEDGERS = ALL_TRADER_LEDGERS


def get_all_trader_ledgers():
    registry = globals().get("ALL_TRADER_LEDGERS", _STATIC_ALL_TRADER_LEDGERS)
    if registry is _STATIC_ALL_TRADER_LEDGERS:
        return [
            ("S", SINGLE_LEDGER),
            ("C", CONVICTION_LEDGER),
            ("M", MODEL_TRACKER_LEDGER),
            ("G", GEMINI_TRACKER_LEDGER),
        ]
    return list(registry)


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
    skip_wallet_conflict_check: bool = False,
    force_market_order: bool = False,
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
        skip_wallet_conflict_check=skip_wallet_conflict_check,
        force_market_order=force_market_order,
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


def _coerce_probability(value, default: float | None = None) -> float | None:
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(prob):
        return default
    return prob


def _tracker_event_timestamp(row) -> pd.Timestamp | None:
    event_ts = pd.to_datetime(
        row.get("market_event_date") or row.get("event_date"),
        utc=True,
        errors="coerce",
    )
    if pd.isna(event_ts):
        return None
    return event_ts


def _tracker_hours_until_event(row) -> float | None:
    event_ts = _tracker_event_timestamp(row)
    if event_ts is None:
        return None
    return (event_ts - pd.Timestamp.now(tz="UTC")).total_seconds() / 3600.0


def _within_tracker_window(row) -> bool:
    hours_until = _tracker_hours_until_event(row)
    if hours_until is None:
        return False
    return 0 < hours_until <= TRACKER_MIN_HOURS_BEFORE_EVENT


def _tracker_decision_id(trader: str, row) -> str:
    raw = "||".join(
        [
            trader,
            str(row.get("fighter_a", "") or "").strip().casefold(),
            str(row.get("fighter_b", "") or "").strip().casefold(),
            str(row.get("market_event_date") or row.get("event_date") or "").strip().casefold(),
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{trader}_{digest}"


def _tracker_decision_record(
    *,
    trader: str,
    decision_id: str,
    row,
    event_title: str = "",
) -> dict:
    hours_until = _tracker_hours_until_event(row)
    return {
        "type": "decision",
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "trader": trader,
        "decision_id": decision_id,
        "fighter_a": str(row.get("fighter_a", "") or ""),
        "fighter_b": str(row.get("fighter_b", "") or ""),
        "event_date": str(row.get("event_date", "") or ""),
        "market_event_date": str(row.get("market_event_date", "") or ""),
        "event_title": event_title,
        "weight_class": str(row.get("weight_class", "") or ""),
        "market_id": str(row.get("market_id", "") or ""),
        "hours_until_event": round(hours_until, 2) if hours_until is not None else None,
    }


def _build_tracker_bet(
    row,
    *,
    bet_on: str,
    bet_side: str,
    model_prob: float,
    market_prob: float,
    edge: float,
    reason: str,
    decision_id: str,
) -> dict:
    bet = {
        "fighter_a": row.get("fighter_a", ""),
        "fighter_b": row.get("fighter_b", ""),
        "bet_on": bet_on,
        "bet_side": bet_side,
        "model_prob": model_prob,
        "blended_prob": model_prob,
        "market_prob": market_prob,
        "edge": edge,
        "decimal_odds": implied_prob_to_decimal_odds(market_prob),
        "event_date": row.get("market_event_date") or row.get("event_date"),
        "market_event_date": row.get("market_event_date"),
        "weight_class": row.get("weight_class", ""),
        "confidence": model_prob,
        "override_bet_size": 1.0,
        "reason": reason,
        "decision_id": decision_id,
    }
    for col in (
        "token_id_yes",
        "token_id_no",
        "market_id",
        "condition_id",
        "tick_size",
        "neg_risk",
        "volume",
        "liquidity",
    ):
        if row.get(col) is not None:
            bet[col] = row.get(col)
    return bet


def _append_tracker_outcome(trader: str, bet: pd.Series | dict, order: dict | None) -> None:
    from src.strategy.llm_operator import log_tracker_decision

    bet_dict = dict(bet)
    decision_id = str(bet_dict.get("decision_id", "") or "").strip()
    if not decision_id:
        return

    response = order.get("response") if isinstance(order, dict) else {}
    order_id = ""
    if isinstance(response, dict):
        order_id = str(
            response.get("orderID")
            or response.get("orderId")
            or response.get("id")
            or ""
        ).strip()

    order_status = str(order.get("status", "") or "").strip() if isinstance(order, dict) else "skipped"
    log_tracker_decision(
        {
            "type": "outcome",
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "trader": trader,
            "decision_id": decision_id,
            "fighter_a": str(bet_dict.get("fighter_a", "") or ""),
            "fighter_b": str(bet_dict.get("fighter_b", "") or ""),
            "pick": str(bet_dict.get("bet_on", "") or ""),
            "bet_side": str(bet_dict.get("bet_side", "") or ""),
            "market_id": str(bet_dict.get("market_id", "") or ""),
            "bet_placed": order_status in {"placed", "dry_run"},
            "order_status": order_status,
            "order_type": order.get("order_type") if isinstance(order, dict) else None,
            "order_id": order_id or None,
            "dry_run": bool(order.get("dry_run")) if isinstance(order, dict) else None,
            "bet_size_usd": float(order.get("bet_size_usd") or 0.0) if isinstance(order, dict) else 0.0,
            "price": order.get("price") if isinstance(order, dict) else None,
            "error": order.get("error") if isinstance(order, dict) else "skipped_by_executor",
        }
    )


def find_flat_model_bets(
    predictions: pd.DataFrame,
    *,
    event_title: str = "",
) -> pd.DataFrame:
    from src.strategy.llm_operator import log_tracker_decision

    bets = []
    for _, row in predictions.iterrows():
        decision_id = _tracker_decision_id("M", row)
        decision = _tracker_decision_record(
            trader="M",
            decision_id=decision_id,
            row=row,
            event_title=event_title,
        )

        hours_until = _tracker_hours_until_event(row)
        if hours_until is None:
            log_tracker_decision(
                {
                    **decision,
                    "status": "event_time_unavailable",
                    "summary": "Event time unavailable",
                    "rationale": "Model Tracker skipped this fight because the market event timestamp was unavailable.",
                }
            )
            continue
        if hours_until <= 0:
            log_tracker_decision(
                {
                    **decision,
                    "status": "event_started",
                    "summary": "Event already started",
                    "rationale": "Model Tracker skipped this fight because the market event time is no longer in the future.",
                }
            )
            continue
        if not _within_tracker_window(row):
            log_tracker_decision(
                {
                    **decision,
                    "status": "outside_window",
                    "summary": "Outside tracker window",
                    "rationale": (
                        f"Model Tracker skipped this fight because it is {hours_until:.1f}h away, "
                        f"outside the {TRACKER_MIN_HOURS_BEFORE_EVENT}h market-liquidity window."
                    ),
                }
            )
            continue

        model_a = _coerce_probability(row.get("prob_a"))
        model_b = _coerce_probability(row.get("prob_b"))
        if model_a is None or model_b is None:
            log_tracker_decision(
                {
                    **decision,
                    "status": "missing_model_prob",
                    "summary": "Missing model probabilities",
                    "rationale": "Model Tracker skipped this fight because one or both model probabilities were unavailable.",
                }
            )
            continue

        bet_side = "a" if model_a >= model_b else "b"
        bet_on = str(row.get("fighter_a" if bet_side == "a" else "fighter_b", "") or "")
        other_prob = model_b if bet_side == "a" else model_a
        model_prob = model_a if bet_side == "a" else model_b
        market_prob = _coerce_probability(
            row.get("a_market_prob" if bet_side == "a" else "b_market_prob")
        )
        if market_prob is None or market_prob <= 0:
            log_tracker_decision(
                {
                    **decision,
                    "status": "missing_market_prob",
                    "summary": "Missing market price",
                    "rationale": "Model Tracker skipped this fight because the matched market price was unavailable.",
                }
            )
            continue

        rationale = (
            f"Pure model pick: {bet_on} ({model_prob:.1%} vs {other_prob:.1%}). "
            "No blend, no edge threshold, no LLM gate."
        )
        log_tracker_decision(
            {
                **decision,
                "status": "eligible",
                "summary": f"Pick: {bet_on}",
                "pick": bet_on,
                "bet_side": bet_side,
                "model_prob": model_prob,
                "market_prob": market_prob,
                "edge": model_prob - market_prob,
                "rationale": rationale,
                "sources": [],
            }
        )
        bets.append(
            _build_tracker_bet(
                row,
                bet_on=bet_on,
                bet_side=bet_side,
                model_prob=model_prob,
                market_prob=market_prob,
                edge=model_prob - market_prob,
                reason=rationale,
                decision_id=decision_id,
            )
        )

    return pd.DataFrame(bets)


def find_flat_gemini_bets(
    predictions: pd.DataFrame,
    *,
    event_title: str = "",
) -> pd.DataFrame:
    from src.strategy.llm_operator import gemini_standalone_pick, log_tracker_decision

    bets = []
    for _, row in predictions.iterrows():
        decision_id = _tracker_decision_id("G", row)
        decision = _tracker_decision_record(
            trader="G",
            decision_id=decision_id,
            row=row,
            event_title=event_title,
        )

        hours_until = _tracker_hours_until_event(row)
        if hours_until is None:
            log_tracker_decision(
                {
                    **decision,
                    "status": "event_time_unavailable",
                    "summary": "Event time unavailable",
                    "rationale": "Gemini Tracker skipped this fight because the market event timestamp was unavailable.",
                }
            )
            continue
        if hours_until <= 0:
            log_tracker_decision(
                {
                    **decision,
                    "status": "event_started",
                    "summary": "Event already started",
                    "rationale": "Gemini Tracker skipped this fight because the market event time is no longer in the future.",
                }
            )
            continue
        if not _within_tracker_window(row):
            log_tracker_decision(
                {
                    **decision,
                    "status": "outside_window",
                    "summary": "Outside tracker window",
                    "rationale": (
                        f"Gemini Tracker skipped this fight because it is {hours_until:.1f}h away, "
                        f"outside the {TRACKER_MIN_HOURS_BEFORE_EVENT}h market-liquidity window."
                    ),
                }
            )
            continue

        fighter_a = str(row.get("fighter_a", "") or "")
        fighter_b = str(row.get("fighter_b", "") or "")
        pick = gemini_standalone_pick(
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            weight_class=str(row.get("weight_class", "") or ""),
            event_date=str(row.get("market_event_date") or row.get("event_date") or ""),
            event_title=event_title,
        )

        raw_pick = str(pick.get("pick") or "").strip()
        if not raw_pick:
            log_tracker_decision(
                {
                    **decision,
                    "status": "no_pick",
                    "summary": "No Gemini pick",
                    "pick": None,
                    "confidence": pick.get("confidence"),
                    "rationale": pick.get("rationale", ""),
                    "fighter_assessment": pick.get("fighter_assessment", ""),
                    "risk_flags": pick.get("risk_flags", []),
                    "verified_records": pick.get("verified_records", {}),
                    "sources": pick.get("sources", []),
                    "cached": bool(pick.get("cached")),
                }
            )
            continue

        if same_person_name(raw_pick, fighter_a):
            bet_side = "a"
            bet_on = fighter_a
        elif same_person_name(raw_pick, fighter_b):
            bet_side = "b"
            bet_on = fighter_b
        else:
            log_tracker_decision(
                {
                    **decision,
                    "status": "invalid_pick",
                    "summary": "Unrecognized Gemini pick",
                    "pick": raw_pick,
                    "confidence": pick.get("confidence"),
                    "rationale": pick.get("rationale", ""),
                    "fighter_assessment": pick.get("fighter_assessment", ""),
                    "risk_flags": pick.get("risk_flags", []),
                    "verified_records": pick.get("verified_records", {}),
                    "sources": pick.get("sources", []),
                    "cached": bool(pick.get("cached")),
                }
            )
            continue

        market_prob = _coerce_probability(
            row.get("a_market_prob" if bet_side == "a" else "b_market_prob")
        )
        if market_prob is None or market_prob <= 0:
            log_tracker_decision(
                {
                    **decision,
                    "status": "missing_market_prob",
                    "summary": "Missing market price",
                    "pick": bet_on,
                    "rationale": "Gemini Tracker got a pick but the matched market price was unavailable.",
                    "sources": pick.get("sources", []),
                }
            )
            continue

        confidence = _coerce_probability(pick.get("confidence"), 0.5)
        confidence = min(max(float(confidence or 0.5), 0.0), 1.0)
        rationale = str(pick.get("rationale", "") or "")
        log_tracker_decision(
            {
                **decision,
                "status": "eligible",
                "summary": f"Pick: {bet_on}",
                "pick": bet_on,
                "bet_side": bet_side,
                "confidence": confidence,
                "market_prob": market_prob,
                "edge": confidence - market_prob,
                "rationale": rationale,
                "fighter_assessment": pick.get("fighter_assessment", ""),
                "risk_flags": pick.get("risk_flags", []),
                "verified_records": pick.get("verified_records", {}),
                "sources": pick.get("sources", []),
                "cached": bool(pick.get("cached")),
            }
        )
        bets.append(
            _build_tracker_bet(
                row,
                bet_on=bet_on,
                bet_side=bet_side,
                model_prob=confidence,
                market_prob=market_prob,
                edge=confidence - market_prob,
                reason=rationale,
                decision_id=decision_id,
            )
        )

    return pd.DataFrame(bets)


def _create_tracker_trader(
    name: str,
    ledger_path,
    clob: Optional[ClobClientWrapper],
    dry_run: bool,
    *,
    available_cash: float,
) -> TraderProfile:
    return _create_trader(
        TraderProfile(
            name=name,
            blend_weight=1.0,
            ledger_path=ledger_path,
        ),
        allocation=max(float(available_cash or 0.0), 0.0),
        available_cash=max(float(available_cash or 0.0), 0.0),
        clob=clob,
        dry_run=dry_run,
        kelly_fraction=1.0,
        max_bet_fraction=1.0,
        min_edge_threshold=0.0,
        edge_scaling_base=0.0,
        skip_wallet_conflict_check=True,
        force_market_order=True,
    )


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
    if not conviction_bets.empty and s_fight_keys:
        conviction_keys = conviction_bets.apply(_fight_key, axis=1)
        already_owned_mask = conviction_keys.isin(s_fight_keys)
        skipped_owned = int(already_owned_mask.sum())
        if skipped_owned:
            logger.info(
                "Skipping %d conviction fights already covered by Single Trader before operator evaluation",
                skipped_owned,
            )
            conviction_bets = conviction_bets.loc[~already_owned_mask].reset_index(drop=True)

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

    total_wagered_s = sum(order.get("bet_size_usd", 0) for order in s_orders)
    total_wagered_nm = sum(order.get("bet_size_usd", 0) for order in nm_orders)
    total_wagered_c = sum(order.get("bet_size_usd", 0) for order in c_orders)

    def _tracker_cash_after_sc() -> float:
        if dry_run:
            spent = total_wagered_s + total_wagered_nm + total_wagered_c
            return max(0.0, float(available_cash or 0.0) - spent)
        live_state = _fetch_polymarket_account_state(
            require_confirmed_cash=True,
            require_portfolio_value=False,
        )
        return float(live_state.get("cash_balance") or 0.0)

    tracker_cash = _tracker_cash_after_sc()
    model_tracker = _create_tracker_trader(
        "Model Tracker (M)",
        MODEL_TRACKER_LEDGER,
        clob,
        dry_run,
        available_cash=tracker_cash,
    )
    logger.info(
        "\n  %s: equity $%.2f | cash $%.2f (post S+C wallet state)",
        model_tracker.name,
        _bankroll_total_equity(model_tracker.bankroll),
        _bankroll_available_cash(model_tracker.bankroll),
    )

    matched_m = model_tracker.executor._match_predictions_to_markets(predictions, markets)
    model_bets = find_flat_model_bets(matched_m, event_title=event_title)
    logger.info("  %s: %s flat bets found", model_tracker.name, len(model_bets))
    _report_progress(
        f"Cycle active: executing {len(model_bets)} flat bets for Model Tracker"
    )

    m_orders = []
    if not model_bets.empty:
        logger.info("\n--- Executing %s ---", model_tracker.name)
        for _, bet in model_bets.iterrows():
            if model_tracker.bankroll.is_stopped:
                logger.warning("  Model Tracker stop-loss triggered - skipping remaining bets")
                break
            order = model_tracker.executor._place_bet(bet, markets)
            _append_tracker_outcome("M", bet, order)
            if order:
                order["trader"] = "M"
                m_orders.append(order)

    tracker_cash_for_g = (
        _bankroll_available_cash(model_tracker.bankroll)
        if dry_run
        else _tracker_cash_after_sc()
    )
    gemini_tracker = _create_tracker_trader(
        "Gemini Tracker (G)",
        GEMINI_TRACKER_LEDGER,
        clob,
        dry_run,
        available_cash=tracker_cash_for_g,
    )
    logger.info(
        "  %s: equity $%.2f | cash $%.2f (post M wallet state)",
        gemini_tracker.name,
        _bankroll_total_equity(gemini_tracker.bankroll),
        _bankroll_available_cash(gemini_tracker.bankroll),
    )

    matched_g = gemini_tracker.executor._match_predictions_to_markets(predictions, markets)
    gemini_bets = find_flat_gemini_bets(matched_g, event_title=event_title)
    logger.info("  %s: %s flat bets found", gemini_tracker.name, len(gemini_bets))
    _report_progress(
        f"Cycle active: executing {len(gemini_bets)} flat bets for Gemini Tracker"
    )

    g_orders = []
    if not gemini_bets.empty:
        logger.info("\n--- Executing %s ---", gemini_tracker.name)
        for _, bet in gemini_bets.iterrows():
            if gemini_tracker.bankroll.is_stopped:
                logger.warning("  Gemini Tracker stop-loss triggered - skipping remaining bets")
                break
            order = gemini_tracker.executor._place_bet(bet, markets)
            _append_tracker_outcome("G", bet, order)
            if order:
                order["trader"] = "G"
                g_orders.append(order)

    total_wagered_m = sum(order.get("bet_size_usd", 0) for order in m_orders)
    total_wagered_g = sum(order.get("bet_size_usd", 0) for order in g_orders)
    total_orders = len(s_orders) + len(nm_orders) + len(c_orders) + len(m_orders) + len(g_orders)

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
        "  %s:\n"
        "    Orders: %s | Wagered: $%.2f | Cash remaining: $%.2f | Equity: $%.2f\n"
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
        model_tracker.name,
        len(m_orders),
        total_wagered_m,
        _bankroll_available_cash(model_tracker.bankroll),
        _bankroll_total_equity(model_tracker.bankroll),
        gemini_tracker.name,
        len(g_orders),
        total_wagered_g,
        _bankroll_available_cash(gemini_tracker.bankroll),
        _bankroll_total_equity(gemini_tracker.bankroll),
        total_orders,
        total_wagered_s + total_wagered_nm + total_wagered_c + total_wagered_m + total_wagered_g,
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
        "trader_m": {
            "name": model_tracker.name,
            "blend_weight": 1.0,
            "allocation": tracker_cash,
            "available_cash_start": tracker_cash,
            "orders": m_orders,
            "total_wagered": total_wagered_m,
            "bankroll_remaining": _bankroll_available_cash(model_tracker.bankroll),
            "total_equity": _bankroll_total_equity(model_tracker.bankroll),
            "stats": model_tracker.bankroll.get_stats(),
        },
        "trader_g": {
            "name": gemini_tracker.name,
            "blend_weight": None,
            "allocation": tracker_cash_for_g,
            "available_cash_start": tracker_cash_for_g,
            "orders": g_orders,
            "total_wagered": total_wagered_g,
            "bankroll_remaining": _bankroll_available_cash(gemini_tracker.bankroll),
            "total_equity": _bankroll_total_equity(gemini_tracker.bankroll),
            "stats": gemini_tracker.bankroll.get_stats(),
        },
        "total_orders": total_orders,
    }
