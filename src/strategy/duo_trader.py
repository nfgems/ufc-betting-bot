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
from datetime import timedelta
from typing import Callable, Optional

import pandas as pd

from src.betting_window import bet_window_status, parse_event_timestamp
from src.config import (
    BLEND_WEIGHT,
    CONVICTION_MAX_BET_FRACTION,
    GEMINI_TRACKER_CONFIDENCE_CAP,
    KELLY_FRACTION,
    LIMIT_BID_PRE_EVENT_HOURS,
    LOGS_DIR,
    MAX_BET_HOURS_BEFORE_EVENT,
    MAX_BET_FRACTION,
    MIN_EDGE_THRESHOLD,
    NEAR_MISS_MIN_EDGE,
    STOP_LOSS_FRACTION,
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


class WalletCashUnavailableError(RuntimeError):
    """Raised when a live cycle cannot confirm spendable wallet cash."""


SINGLE_LEDGER = LOGS_DIR / "bet_ledger_single.json"
CONVICTION_LEDGER = LOGS_DIR / "bet_ledger_conviction.json"
MODEL_TRACKER_LEDGER = LOGS_DIR / "bet_ledger_model_tracker.json"
GEMINI_TRACKER_LEDGER = LOGS_DIR / "bet_ledger_gemini_tracker.json"
TRACKER_FLAT_BET_USD = 2.0
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


_NON_CHARGEABLE_ORDER_STATUSES = {
    "failed",
    "unknown",
    "cancelled",
    "canceled",
    "skipped",
}


def _chargeable_order_amount(order: dict) -> float:
    """Return cash reserved by a successful/current order attempt."""
    if not isinstance(order, dict):
        return 0.0
    status = str(order.get("status", "placed") or "placed").strip().lower()
    if status in _NON_CHARGEABLE_ORDER_STATUSES:
        return 0.0
    try:
        return max(0.0, float(order.get("bet_size_usd") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _cash_after_chargeable_orders(starting_cash: float, *order_groups: list[dict]) -> float:
    spent = sum(
        _chargeable_order_amount(order)
        for group in order_groups
        for order in group
    )
    return max(0.0, float(starting_cash or 0.0) - spent)


def _open_buy_order_reserved_cash(order: dict) -> float:
    """Return remaining USDC notional reserved by an open BUY order."""
    if not isinstance(order, dict):
        return 0.0
    side = str(order.get("side", "") or "").strip().upper()
    if side and side != "BUY":
        return 0.0
    try:
        price = float(order.get("price") or 0.0)
        if "remaining_size" in order and order.get("remaining_size") is not None:
            remaining_size = float(order.get("remaining_size") or 0.0)
        else:
            original_size = float(order.get("original_size") or 0.0)
            size_matched = float(order.get("size_matched") or 0.0)
            remaining_size = original_size - size_matched
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, price * max(0.0, remaining_size))


def _reserved_open_buy_order_cash(clob: Optional[ClobClientWrapper] = None) -> float:
    """Fetch current open BUY order notional from the CLOB."""
    client = clob
    try:
        if client is None:
            client = ClobClientWrapper()
        if not hasattr(client, "get_open_orders"):
            return 0.0
        return sum(_open_buy_order_reserved_cash(order) for order in client.get_open_orders())
    except Exception as exc:
        logger.warning("Could not fetch open CLOB orders for cash reservation: %s", exc)
        return 0.0


def _cash_after_open_order_reservations(
    cash_balance: float,
    *,
    dry_run: bool,
    clob: Optional[ClobClientWrapper] = None,
) -> tuple[float, float]:
    """Adjust live CLOB cash by resting BUY orders that the balance endpoint may omit."""
    cash_balance = max(0.0, float(cash_balance or 0.0))
    if dry_run:
        return cash_balance, 0.0

    reserved = _reserved_open_buy_order_cash(clob)
    if reserved <= 0.01:
        return cash_balance, 0.0

    adjusted_cash = max(0.0, cash_balance - reserved)
    logger.info(
        "Open CLOB BUY orders reserve $%.2f; adjusted available cash $%.2f -> $%.2f",
        reserved,
        cash_balance,
        adjusted_cash,
    )
    return adjusted_cash, reserved


def _resolve_cash_after_order_groups(
    *,
    starting_cash: float,
    order_groups: tuple[list[dict], ...],
    dry_run: bool,
    label: str,
    clob: Optional[ClobClientWrapper] = None,
) -> float:
    """Resolve remaining cash, capping live reads by current-cycle reservations."""
    internal_cash = _cash_after_chargeable_orders(starting_cash, *order_groups)
    if dry_run:
        return internal_cash

    state_kwargs = {
        "require_confirmed_cash": True,
        "require_portfolio_value": False,
    }
    if clob is not None:
        state_kwargs["clob_client"] = clob
    live_state = _fetch_polymarket_account_state(**state_kwargs)
    live_cash = max(0.0, float(live_state.get("cash_balance") or 0.0))
    cash_gap = live_cash - internal_cash
    if cash_gap > 0.01:
        logger.info(
            "%s raw CLOB cash $%.2f is above reserved spendable cash $%.2f; "
            "using reserved spendable cash (difference $%.2f from open/current order reservations)",
            label,
            live_cash,
            internal_cash,
            cash_gap,
        )
    elif cash_gap < -0.01:
        logger.warning(
            "%s raw CLOB cash $%.2f is below reserved spendable cash $%.2f; "
            "using lower CLOB cash because the wallet has less free cash than internal reservations expected",
            label,
            live_cash,
            internal_cash,
        )
    return min(live_cash, internal_cash)


def _resolve_total_bankroll(
    dry_run: bool = True,
    clob: Optional[ClobClientWrapper] = None,
) -> WalletBankrollBasis:
    """Resolve the wallet state the traders should use this cycle."""

    state_kwargs = {
        "require_confirmed_cash": True,
        "require_portfolio_value": True,
    }
    if clob is not None:
        state_kwargs["clob_client"] = clob
    live_state = _fetch_polymarket_account_state(**state_kwargs)
    confirmed_cash = bool(live_state.get("confirmed_cash"))
    confirmed_portfolio = bool(live_state.get("confirmed_portfolio"))
    cash_balance = max(0.0, float(live_state.get("cash_balance") or 0.0))
    available_cash, open_order_reserve = _cash_after_open_order_reservations(
        cash_balance,
        dry_run=dry_run,
        clob=clob,
    )
    cash_source = (
        "cash: Polymarket CLOB less open BUY order reserves"
        if open_order_reserve > 0.01
        else "cash: Polymarket CLOB"
    )
    if confirmed_cash and confirmed_portfolio:
        return WalletBankrollBasis(
            total_equity=float(live_state.get("total_equity") or 0.0),
            available_cash=available_cash,
            source=f"{cash_source}, portfolio: Polymarket Data API",
        )
    if confirmed_cash:
        logger.warning(
            "Polymarket portfolio value could not be confirmed; sizing this cycle "
            "against confirmed CLOB cash only"
        )
        return WalletBankrollBasis(
            total_equity=cash_balance,
            available_cash=available_cash,
            source=f"{cash_source}, portfolio: unavailable (cash-only conservative fallback)",
        )

    if dry_run:
        from src.config import INITIAL_BANKROLL

        return WalletBankrollBasis(
            total_equity=INITIAL_BANKROLL,
            available_cash=INITIAL_BANKROLL,
            source="INITIAL_BANKROLL fallback (dry-run only; live wallet state unavailable)",
        )

    raise WalletCashUnavailableError(
        "Live mode: wallet cash balance could not be confirmed from Polymarket. "
        "Refusing to size bets without confirmed available cash."
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
    force_limit_order: bool = False,
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
        force_limit_order=force_limit_order,
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


def _bet_window_event_time(row) -> object | None:
    getter = getattr(row, "get", None)
    if not callable(getter):
        return None

    candidates = []
    for key in ("event_date", "commence_time", "market_event_date"):
        value = getter(key, None)
        if value is None or pd.isna(value):
            continue
        if str(value).strip():
            candidates.append(value)

    for value in candidates:
        if parse_event_timestamp(value) is not None:
            return value
    return candidates[0] if candidates else None


def _filter_bets_to_execution_window(
    bets: pd.DataFrame | None,
    *,
    label: str,
    close_buffer: timedelta | None = None,
    fail_closed: bool = False,
) -> pd.DataFrame:
    if bets is None or bets.empty:
        return bets if isinstance(bets, pd.DataFrame) else pd.DataFrame()

    keep_indices: list[object] = []
    skipped = 0

    for idx, row in bets.iterrows():
        window = bet_window_status(
            _bet_window_event_time(row),
            close_buffer=close_buffer,
            fail_closed=fail_closed,
        )
        if window is None or window["open"]:
            keep_indices.append(idx)
            continue
        skipped += 1

    if skipped:
        logger.info(
            "Skipping %d %s outside the live bet window (%sh before the event)",
            skipped,
            label,
            MAX_BET_HOURS_BEFORE_EVENT,
        )

    return bets.loc[keep_indices].reset_index(drop=True)


def _coerce_probability(value, default: float | None = None) -> float | None:
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(prob):
        return default
    return prob


def _tracker_event_timestamp(row) -> pd.Timestamp | None:
    event_ts = pd.to_datetime(_bet_window_event_time(row), utc=True, errors="coerce")
    if pd.isna(event_ts):
        return None
    return event_ts


def _tracker_hours_until_event(row) -> float | None:
    event_ts = _tracker_event_timestamp(row)
    if event_ts is None:
        return None
    return (event_ts - pd.Timestamp.now(tz="UTC")).total_seconds() / 3600.0


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
        "card_date": str(row.get("card_date", "") or ""),
        "event_title": event_title,
        "weight_class": str(row.get("weight_class", "") or ""),
        "market_id": str(row.get("market_id", "") or ""),
        "hours_until_event": round(hours_until, 2) if hours_until is not None else None,
    }


def _tracker_prediction_key(row) -> tuple[str, str]:
    return (
        _fight_key(row),
        str(row.get("event_date") or row.get("commence_time") or "").strip(),
    )


def _log_unmatched_tracker_decisions(
    *,
    trader: str,
    predictions: pd.DataFrame,
    matched_predictions: pd.DataFrame,
    event_title: str = "",
    audit_callback: Optional[Callable[[str, object, dict], None]] = None,
) -> None:
    """Write an explicit tracker decision for prediction rows with no executable market."""
    if predictions is None or predictions.empty:
        return

    from src.strategy.llm_operator import log_tracker_decision

    matched_keys: set[tuple[str, str]] = set()
    if matched_predictions is not None and not matched_predictions.empty:
        matched_keys = {
            _tracker_prediction_key(row)
            for _, row in matched_predictions.iterrows()
        }

    label = "Model Tracker" if trader == "M" else "Gemini Tracker"
    for _, row in predictions.iterrows():
        if _tracker_prediction_key(row) in matched_keys:
            continue

        row_event_title = str(row.get("event_title", "") or event_title or "")
        decision_id = _tracker_decision_id(trader, row)
        decision = _tracker_decision_record(
            trader=trader,
            decision_id=decision_id,
            row=row,
            event_title=row_event_title,
        )
        record = {
            **decision,
            "status": "no_market",
            "summary": "No market matched",
            "rationale": (
                f"{label} did not make its flat tracker bet because no active "
                "Polymarket market was matched for this fight."
            ),
        }
        log_tracker_decision(record)
        if callable(audit_callback):
            audit_callback(trader, row, record)


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
    signal_confidence: float | None = None,
    probability_source: str = "model",
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
        "event_date": (
            row.get("commence_time")
            or row.get("event_date")
            or row.get("market_event_date")
        ),
        "market_event_date": row.get("market_event_date"),
        "card_date": row.get("card_date"),
        "event_title": row.get("event_title", ""),
        "weight_class": row.get("weight_class", ""),
        "confidence": signal_confidence if signal_confidence is not None else model_prob,
        "override_bet_size": TRACKER_FLAT_BET_USD,
        "reason": reason,
        "decision_id": decision_id,
        "probability_source": probability_source,
    }
    if signal_confidence is not None:
        bet["signal_confidence"] = signal_confidence
        bet["signal_source"] = "gemini_research"
    for col in (
        "token_id_yes",
        "token_id_no",
        "market_id",
        "condition_id",
        "tick_size",
        "neg_risk",
        "fee_rate",
        "fee_exponent",
        "fee_source",
        "fee_schedule",
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
            "event_date": str(bet_dict.get("event_date", "") or ""),
            "market_event_date": str(bet_dict.get("market_event_date", "") or ""),
            "card_date": str(bet_dict.get("card_date", "") or ""),
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
    audit_callback: Optional[Callable[[str, object, dict], None]] = None,
) -> pd.DataFrame:
    from src.strategy.llm_operator import log_tracker_decision

    bets = []
    for _, row in predictions.iterrows():
        def _emit(record: dict) -> None:
            log_tracker_decision(record)
            if callable(audit_callback):
                audit_callback("M", row, record)

        row_event_title = str(row.get("event_title", "") or event_title or "")
        decision_id = _tracker_decision_id("M", row)
        decision = _tracker_decision_record(
            trader="M",
            decision_id=decision_id,
            row=row,
            event_title=row_event_title,
        )

        hours_until = _tracker_hours_until_event(row)
        if hours_until is None:
            _emit(
                {
                    **decision,
                    "status": "event_time_unavailable",
                    "summary": "Event time unavailable",
                    "rationale": "Model Tracker skipped this fight because the market event timestamp was unavailable.",
                }
            )
            continue
        if hours_until <= 0:
            _emit(
                {
                    **decision,
                    "status": "event_started",
                    "summary": "Event already started",
                    "rationale": "Model Tracker skipped this fight because the market event time is no longer in the future.",
                }
            )
            continue
        tracker_window = bet_window_status(_bet_window_event_time(row))
        if tracker_window is not None and not tracker_window["open"]:
            _emit(
                {
                    **decision,
                    "status": "outside_window"
                    if tracker_window.get("state") == "too_early"
                    else "too_close",
                    "summary": "Bet window not open"
                    if tracker_window.get("state") == "too_early"
                    else "Too close to event",
                    "rationale": (
                        "Model Tracker skipped this fight because "
                        f"{str(tracker_window.get('detail') or '').strip().rstrip('.')}"
                        "."
                    ),
                }
            )
            continue

        model_a = _coerce_probability(row.get("prob_a"))
        model_b = _coerce_probability(row.get("prob_b"))
        if model_a is None or model_b is None:
            _emit(
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
            _emit(
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
        _emit(
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
    audit_callback: Optional[Callable[[str, object, dict], None]] = None,
) -> pd.DataFrame:
    from src.strategy.llm_operator import gemini_standalone_pick, log_tracker_decision

    bets = []
    for _, row in predictions.iterrows():
        def _emit(record: dict) -> None:
            log_tracker_decision(record)
            if callable(audit_callback):
                audit_callback("G", row, record)

        row_event_title = str(row.get("event_title", "") or event_title or "")
        decision_id = _tracker_decision_id("G", row)
        decision = _tracker_decision_record(
            trader="G",
            decision_id=decision_id,
            row=row,
            event_title=row_event_title,
        )

        hours_until = _tracker_hours_until_event(row)
        if hours_until is None:
            _emit(
                {
                    **decision,
                    "status": "event_time_unavailable",
                    "summary": "Event time unavailable",
                    "rationale": "Gemini Tracker skipped this fight because the market event timestamp was unavailable.",
                }
            )
            continue
        if hours_until <= 0:
            _emit(
                {
                    **decision,
                    "status": "event_started",
                    "summary": "Event already started",
                    "rationale": "Gemini Tracker skipped this fight because the market event time is no longer in the future.",
                }
            )
            continue
        tracker_window = bet_window_status(_bet_window_event_time(row))
        if tracker_window is not None and not tracker_window["open"]:
            _emit(
                {
                    **decision,
                    "status": "outside_window"
                    if tracker_window.get("state") == "too_early"
                    else "too_close",
                    "summary": "Bet window not open"
                    if tracker_window.get("state") == "too_early"
                    else "Too close to event",
                    "rationale": (
                        "Gemini Tracker skipped this fight because "
                        f"{str(tracker_window.get('detail') or '').strip().rstrip('.')}"
                        "."
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
            event_title=row_event_title,
        )

        raw_pick = str(pick.get("pick") or "").strip()
        if not raw_pick:
            _emit(
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
                    "grounded_research": pick.get("grounded_research", {}),
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
            _emit(
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
                    "grounded_research": pick.get("grounded_research", {}),
                    "cached": bool(pick.get("cached")),
                }
            )
            continue

        market_prob = _coerce_probability(
            row.get("a_market_prob" if bet_side == "a" else "b_market_prob")
        )
        if market_prob is None or market_prob <= 0:
            _emit(
                {
                    **decision,
                    "status": "missing_market_prob",
                    "summary": "Missing market price",
                    "pick": bet_on,
                    "rationale": "Gemini Tracker got a pick but the matched market price was unavailable.",
                    "fighter_assessment": pick.get("fighter_assessment", ""),
                    "verified_records": pick.get("verified_records", {}),
                    "sources": pick.get("sources", []),
                    "grounded_research": pick.get("grounded_research", {}),
                }
            )
            continue

        confidence = _coerce_probability(pick.get("confidence"), 0.5)
        confidence = min(
            max(float(confidence or 0.5), 0.0),
            GEMINI_TRACKER_CONFIDENCE_CAP,
        )
        rationale = str(pick.get("rationale", "") or "")
        _emit(
            {
                **decision,
                "status": "eligible",
                "summary": f"Pick: {bet_on}",
                "pick": bet_on,
                "bet_side": bet_side,
                "confidence": confidence,
                "market_prob": market_prob,
                "signal_confidence": confidence,
                "rationale": rationale,
                "fighter_assessment": pick.get("fighter_assessment", ""),
                "risk_flags": pick.get("risk_flags", []),
                "verified_records": pick.get("verified_records", {}),
                "sources": pick.get("sources", []),
                "grounded_research": pick.get("grounded_research", {}),
                "cached": bool(pick.get("cached")),
            }
        )
        bets.append(
            _build_tracker_bet(
                row,
                bet_on=bet_on,
                bet_side=bet_side,
                model_prob=market_prob,
                market_prob=market_prob,
                edge=0.0,
                reason=rationale,
                decision_id=decision_id,
                signal_confidence=confidence,
                probability_source="market_neutral",
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
        min_edge_threshold=-1.0,
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

    from src.strategy.execution_audit import (
        ExecutionAuditCollector,
        audit_conviction_pipeline,
        audit_single_value_pipeline,
    )

    bankroll_basis = bankroll_basis or _resolve_total_bankroll(dry_run=dry_run, clob=clob)
    total_equity = bankroll_basis.total_equity
    available_cash = bankroll_basis.available_cash
    execution_audit = ExecutionAuditCollector(
        dry_run=dry_run,
        event_title=event_title,
        metadata={
            "bankroll_source": bankroll_basis.source,
            "total_equity": total_equity,
            "available_cash": available_cash,
            "min_edge": min_edge,
        },
    )

    def _report_progress(message: str) -> None:
        if not callable(progress_callback):
            return
        try:
            progress_callback(message)
        except Exception as exc:
            logger.debug("Duo trader progress callback failed: %s", exc)

    def _row_decision_key(row) -> tuple[str, str, str]:
        return (
            _fight_key(row),
            str(row.get("bet_on", "") or ""),
            str(row.get("bet_side", "") or ""),
        )

    def _audit_window_filtered(
        trader: str,
        before: pd.DataFrame | None,
        after: pd.DataFrame | None,
        *,
        label: str,
        close_buffer: timedelta | None = None,
    ) -> None:
        if before is None or before.empty:
            return
        after_keys = set()
        if after is not None and not after.empty:
            after_keys = {_row_decision_key(row) for _, row in after.iterrows()}
        for _, row in before.iterrows():
            if _row_decision_key(row) in after_keys:
                continue
            window = bet_window_status(
                _bet_window_event_time(row),
                close_buffer=close_buffer,
                fail_closed=not dry_run,
            )
            if window is None or window.get("open"):
                continue
            execution_audit.record_path(
                trader,
                row,
                status=(
                    "event_time_unavailable"
                    if window.get("state") == "event_time_unavailable"
                    else "outside_window"
                ),
                gate="bet_window",
                explanation=(
                    f"Skipped by {label} because {str(window.get('detail') or '').strip().rstrip('.')}."
                ),
                numbers={
                    "bet_on": row.get("bet_on"),
                    "bet_side": row.get("bet_side"),
                    "model_probability": _coerce_probability(row.get("model_prob")),
                    "no_odds_probability": _coerce_probability(row.get("no_odds_prob")),
                    "market_probability": _coerce_probability(row.get("market_prob")),
                    "blended_probability": _coerce_probability(row.get("blended_prob")),
                    "edge": _coerce_probability(row.get("edge")),
                    "market_id": str(row.get("market_id", "") or ""),
                    "token_id": str(
                        row.get("token_id_yes" if row.get("bet_side") == "a" else "token_id_no")
                        or ""
                    ),
                    "bet_window": window,
                },
            )

    def _audit_operator_results(trader: str, evaluated: pd.DataFrame) -> None:
        decisions_by_key = getattr(evaluated, "attrs", {}).get("operator_decisions_by_key", {})
        prepared_rows = getattr(evaluated, "attrs", {}).get("operator_prepared_keys", [])
        for bet, decision_key in prepared_rows:
            decision = decisions_by_key.get(decision_key)
            if decision is None:
                continue
            execution_audit.record_operator_decision(trader, bet, decision)

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
    execution_audit.bind_executor(single.executor, "S")

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
    audit_single_value_pipeline(
        execution_audit,
        predictions=predictions,
        matched_predictions=matched_s,
        min_edge=min_edge,
        blend_weight=BLEND_WEIGHT,
        edge_scaling_base=min_edge,
    )
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
    value_bets_before_window = value_bets.copy() if not value_bets.empty else value_bets
    value_bets = _filter_bets_to_execution_window(
        value_bets,
        label="value bets",
        fail_closed=not dry_run,
    )
    _audit_window_filtered(
        "S",
        value_bets_before_window,
        value_bets,
        label="Single Trader",
    )
    near_miss_bets_before_window = near_miss_bets.copy() if not near_miss_bets.empty else near_miss_bets
    near_miss_bets = _filter_bets_to_execution_window(
        near_miss_bets,
        label="near-miss limit orders",
        close_buffer=timedelta(hours=LIMIT_BID_PRE_EVENT_HOURS),
        fail_closed=not dry_run,
    )
    _audit_window_filtered(
        "S",
        near_miss_bets_before_window,
        near_miss_bets,
        label="Single Trader near-miss limit",
        close_buffer=timedelta(hours=LIMIT_BID_PRE_EVENT_HOURS),
    )

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
            decision_context="S",
        )
        _audit_operator_results("S", value_bets)

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
            decision_context="S",
        )
        _audit_operator_results("S", near_miss_bets)

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
                execution_audit.record_path(
                    "S",
                    bet,
                    status="blocked",
                    gate="stop_loss",
                    explanation="Skipped by Single Trader because stop-loss was triggered.",
                    numbers={
                        "bet_on": bet.get("bet_on"),
                        "bet_side": bet.get("bet_side"),
                        "bankroll_remaining": _bankroll_available_cash(single.bankroll),
                    },
                )
                continue
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
    execution_audit.bind_executor(conv.executor, "C")

    logger.info(
        "\n  %s: equity $%.2f | cash $%.2f (after S reserved cash)",
        conv.name,
        _bankroll_total_equity(conv.bankroll),
        _bankroll_available_cash(conv.bankroll),
    )

    matched_c = conv.executor._match_predictions_to_markets(predictions, markets)
    audit_conviction_pipeline(
        execution_audit,
        predictions=predictions,
        matched_predictions=matched_c,
        require_positive_ev=True,
    )
    conviction_bets = find_conviction_bets(matched_c, require_positive_ev=True)
    conviction_bets_before_window = conviction_bets.copy() if not conviction_bets.empty else conviction_bets
    conviction_bets = _filter_bets_to_execution_window(
        conviction_bets,
        label="conviction bets",
        fail_closed=not dry_run,
    )
    _audit_window_filtered(
        "C",
        conviction_bets_before_window,
        conviction_bets,
        label="Conviction Trader",
    )

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
            decision_context="C",
        )
        _audit_operator_results("C", conviction_bets)

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
                execution_audit.record_path(
                    "C",
                    bet,
                    status="blocked",
                    gate="single_trader_same_fight",
                    explanation=(
                        "Skipped by Conviction Trader because Single Trader already "
                        "has a current-cycle or open bet on this fight."
                    ),
                    numbers={
                        "bet_on": bet.get("bet_on"),
                        "bet_side": bet.get("bet_side"),
                        "model_probability": _coerce_probability(bet.get("model_prob")),
                        "no_odds_probability": _coerce_probability(bet.get("no_odds_prob")),
                        "market_probability": _coerce_probability(bet.get("market_prob")),
                        "edge": _coerce_probability(bet.get("edge")),
                        "market_id": str(bet.get("market_id", "") or ""),
                    },
                )
                continue

            if conv.bankroll.is_stopped:
                logger.warning(
                    "  Conviction Trader stop-loss triggered - skipping remaining bets"
                )
                execution_audit.record_path(
                    "C",
                    bet,
                    status="blocked",
                    gate="stop_loss",
                    explanation="Skipped by Conviction Trader because stop-loss was triggered.",
                    numbers={
                        "bet_on": bet.get("bet_on"),
                        "bet_side": bet.get("bet_side"),
                        "bankroll_remaining": _bankroll_available_cash(conv.bankroll),
                    },
                )
                continue

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
                execution_audit.record_path(
                    "C",
                    bet,
                    status="skipped",
                    gate="conviction_bet_size",
                    explanation=(
                        f"Skipped by Conviction Trader because calculated bet size was ${bet_size:.2f}."
                    ),
                    numbers={
                        "bet_on": bet.get("bet_on"),
                        "bet_side": bet.get("bet_side"),
                        "model_probability": _coerce_probability(bet.get("model_prob")),
                        "bankroll_equity": _bankroll_total_equity(conv.bankroll),
                        "available_cash": _bankroll_available_cash(conv.bankroll),
                        "bet_size_usd": bet_size,
                    },
                )
                continue

            bet_with_size = bet.copy()
            bet_with_size["override_bet_size"] = bet_size
            order = conv.executor._place_bet(bet_with_size, markets)
            if order:
                order["trader"] = "C"
                order["conviction_score"] = bet.get("conviction_score", 0)
                c_orders.append(order)

    total_wagered_s = sum(_chargeable_order_amount(order) for order in s_orders)
    total_wagered_nm = sum(_chargeable_order_amount(order) for order in nm_orders)
    total_wagered_c = sum(_chargeable_order_amount(order) for order in c_orders)

    tracker_cash = _resolve_cash_after_order_groups(
        starting_cash=available_cash,
        order_groups=(s_orders, nm_orders, c_orders),
        dry_run=dry_run,
        label="Model Tracker",
        clob=clob,
    )
    model_tracker = _create_tracker_trader(
        "Model Tracker (M)",
        MODEL_TRACKER_LEDGER,
        clob,
        dry_run,
        available_cash=tracker_cash,
    )
    execution_audit.bind_executor(model_tracker.executor, "M")
    logger.info(
        "\n  %s: equity $%.2f | cash $%.2f (post S+C wallet state)",
        model_tracker.name,
        _bankroll_total_equity(model_tracker.bankroll),
        _bankroll_available_cash(model_tracker.bankroll),
    )

    matched_m = model_tracker.executor._match_predictions_to_markets(predictions, markets)
    _log_unmatched_tracker_decisions(
        trader="M",
        predictions=predictions,
        matched_predictions=matched_m,
        event_title=event_title,
        audit_callback=execution_audit.record_tracker_decision,
    )
    model_bets = find_flat_model_bets(
        matched_m,
        event_title=event_title,
        audit_callback=execution_audit.record_tracker_decision,
    )
    model_bets_before_window = model_bets.copy() if not model_bets.empty else model_bets
    model_bets = _filter_bets_to_execution_window(
        model_bets,
        label="model tracker bets",
    )
    _audit_window_filtered(
        "M",
        model_bets_before_window,
        model_bets,
        label="Model Tracker",
    )
    model_tracker.executor.refresh_open_limit_orders(
        matched_predictions=matched_m,
        primary_bets=model_bets,
        trader_name=model_tracker.name,
    )
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
                execution_audit.record_path(
                    "M",
                    bet,
                    status="blocked",
                    gate="stop_loss",
                    explanation="Skipped by Model Tracker because stop-loss was triggered.",
                    numbers={
                        "bet_on": bet.get("bet_on"),
                        "bet_side": bet.get("bet_side"),
                        "bankroll_remaining": _bankroll_available_cash(model_tracker.bankroll),
                    },
                )
                continue
            order = model_tracker.executor._place_bet(bet, markets)
            _append_tracker_outcome("M", bet, order)
            if order:
                order["trader"] = "M"
                m_orders.append(order)

    tracker_cash_for_g = _resolve_cash_after_order_groups(
        starting_cash=available_cash,
        order_groups=(s_orders, nm_orders, c_orders, m_orders),
        dry_run=dry_run,
        label="Gemini Tracker",
        clob=clob,
    )
    gemini_tracker = _create_tracker_trader(
        "Gemini Tracker (G)",
        GEMINI_TRACKER_LEDGER,
        clob,
        dry_run,
        available_cash=tracker_cash_for_g,
    )
    execution_audit.bind_executor(gemini_tracker.executor, "G")
    logger.info(
        "  %s: equity $%.2f | cash $%.2f (post M wallet state)",
        gemini_tracker.name,
        _bankroll_total_equity(gemini_tracker.bankroll),
        _bankroll_available_cash(gemini_tracker.bankroll),
    )

    matched_g = gemini_tracker.executor._match_predictions_to_markets(predictions, markets)
    _log_unmatched_tracker_decisions(
        trader="G",
        predictions=predictions,
        matched_predictions=matched_g,
        event_title=event_title,
        audit_callback=execution_audit.record_tracker_decision,
    )
    gemini_bets = find_flat_gemini_bets(
        matched_g,
        event_title=event_title,
        audit_callback=execution_audit.record_tracker_decision,
    )
    gemini_bets_before_window = gemini_bets.copy() if not gemini_bets.empty else gemini_bets
    gemini_bets = _filter_bets_to_execution_window(
        gemini_bets,
        label="gemini tracker bets",
    )
    _audit_window_filtered(
        "G",
        gemini_bets_before_window,
        gemini_bets,
        label="Gemini Tracker",
    )
    gemini_tracker.executor.refresh_open_limit_orders(
        matched_predictions=matched_g,
        primary_bets=gemini_bets,
        trader_name=gemini_tracker.name,
    )
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
                execution_audit.record_path(
                    "G",
                    bet,
                    status="blocked",
                    gate="stop_loss",
                    explanation="Skipped by Gemini Tracker because stop-loss was triggered.",
                    numbers={
                        "bet_on": bet.get("bet_on"),
                        "bet_side": bet.get("bet_side"),
                        "bankroll_remaining": _bankroll_available_cash(gemini_tracker.bankroll),
                    },
                )
                continue
            order = gemini_tracker.executor._place_bet(bet, markets)
            _append_tracker_outcome("G", bet, order)
            if order:
                order["trader"] = "G"
                g_orders.append(order)

    total_wagered_m = sum(_chargeable_order_amount(order) for order in m_orders)
    total_wagered_g = sum(_chargeable_order_amount(order) for order in g_orders)
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

    try:
        audit_payload = execution_audit.persist()
    except Exception as exc:
        logger.error("Failed to persist execution decision audit: %s", exc)
        audit_payload = {
            "cycle_id": execution_audit.cycle_id,
            "fight_count": len(execution_audit._fights),
            "completed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }

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
        "execution_audit": {
            "cycle_id": audit_payload.get("cycle_id"),
            "fight_count": audit_payload.get("fight_count"),
            "completed_at": audit_payload.get("completed_at"),
        },
    }
