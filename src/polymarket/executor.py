"""
Order executor — places and manages bets on Polymarket based on model signals.
"""

import hashlib
import logging
import math
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests

from src.betting_window import bet_window_status, parse_event_timestamp
from src.data.name_utils import normalize_cross_source_name, same_person_name
from src.polymarket.client import (
    ClobClientWrapper,
    is_uncertain_clob_order_submission_error,
)
from src.polymarket.market_lookup import build_market_token_lookup
from src.polymarket.markets import get_ufc_fight_markets
from src.strategy.value import (
    conviction_bet_size,
    find_value_bets,
    implied_prob_to_decimal_odds,
    scaled_min_edge,
)
from src.strategy.bankroll import BankrollManager
from src.config import (
    MIN_EDGE_THRESHOLD,
    NEAR_MISS_MIN_EDGE,
    MIN_BOOK_LIQUIDITY,
    MAX_SLIPPAGE,
    MAX_BET_VS_BOOK_RATIO,
    LIMIT_BID_TTL_HOURS,
    LIMIT_BID_PRE_EVENT_HOURS,
    LIMIT_REPRICE_TICK_THRESHOLD,
    LIMIT_REPRICE_MIN_AGE_MINUTES,
    LIMIT_REPRICE_MAX_UPDATES,
)
from src.polymarket.tracker import BetLedger, _acquire_file_lock, _release_file_lock

logger = logging.getLogger(__name__)
_MARKETABLE_LIMIT_ORDER_TYPE = "marketable_limit"
_RESTING_LIMIT_ORDER_TYPES = frozenset(
    ("limit_bid", "limit", "near_miss_limit", _MARKETABLE_LIMIT_ORDER_TYPE)
)
_placement_locks: dict[str, threading.Lock] = {}
_placement_locks_guard = threading.Lock()
_WALLET_POSITION_CACHE_TTL_SECONDS = 60.0
_WALLET_POSITION_FETCH_CACHE: dict[str, tuple[float, list[dict]]] = {}
_WALLET_POSITION_RATE_LIMIT_UNTIL: dict[str, float] = {}
_WALLET_POSITION_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 530})
POST_CANCEL_CONFIRMATION_ATTEMPTS = 2
POST_CANCEL_CONFIRMATION_RETRY_SECONDS = 0.75
POLYMARKET_MIN_ORDER_USD = 2.0
POLYMARKET_LIMIT_SIZE_DECIMALS = 2
_EXECUTION_METADATA_FIELDS = (
    "tick_size",
    "neg_risk",
    "fee_rate",
    "fee_exponent",
    "fee_source",
)
_LEDGER_SIGNAL_METADATA_FIELDS = (
    "signal_confidence",
    "signal_source",
    "probability_source",
    "card_date",
)


def _ledger_entry_blocks_new_order(entry: dict, dry_run: bool) -> bool:
    """Decide whether an open ledger entry should block a new order attempt.

    Real-money runs should ignore historical dry-run entries, but repeated
    dry-run loops should still treat prior dry-run orders as duplicates.
    """
    return (not entry.get("dry_run")) or dry_run


def _order_failure_is_warning(exc: Exception) -> bool:
    """Treat expected API/order rejections as warnings instead of hard errors."""
    if is_uncertain_clob_order_submission_error(exc):
        return False
    msg = str(exc).lower()
    known_rejections = (
        "trading restricted in your region",
        "status_code=403",
        "status_code=400",
        "insufficient balance",
        "not enough balance",
        "not enough allowance",
        "invalid tick size",
        "minimum tick size",
    )
    return any(pattern in msg for pattern in known_rejections)


def _log_order_failure(action: str, fighter: str, exc: Exception) -> None:
    """Log handled order placement failures without promoting expected rejects to errors."""
    msg = f"{action} for {fighter}: {exc}"
    if _order_failure_is_warning(exc):
        logger.warning(msg)
    else:
        logger.error(msg)


def _extract_order_id(resp, warn: bool = False) -> Optional[str]:
    """Extract order ID from a CLOB post_order response.

    The CLOB client may return:
      - {"orderID": "0x..."} (single order)
      - {"orderIDs": ["0x..."]} (batch / newer client versions)
      - {"id": "0x..."}
    """
    if not isinstance(resp, dict):
        if warn and resp is not None:
            logger.warning(f"CLOB response is not a dict (got {type(resp).__name__}): {resp}")
        return None
    oid = resp.get("orderID") or resp.get("id")
    if oid:
        return oid
    ids = resp.get("orderIDs")
    if isinstance(ids, list) and ids:
        return ids[0]
    if warn:
        logger.warning(f"Could not extract order ID from CLOB response: {resp}")
    return None


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _market_price_or_default(value, default: float = math.nan) -> float:
    parsed = _safe_float(value, math.nan)
    if math.isnan(parsed):
        return default
    return parsed


def _valid_market_probability(value) -> bool:
    parsed = _safe_float(value, math.nan)
    return not math.isnan(parsed) and 0.0 <= parsed <= 1.0


def _metadata_missing(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"", "none", "nan", "nat"}


def _ledger_signal_metadata_from_bet(bet: pd.Series | dict) -> dict:
    metadata: dict = {}
    for field in _LEDGER_SIGNAL_METADATA_FIELDS:
        value = bet.get(field)
        if not _metadata_missing(value):
            metadata[field] = value
    return metadata


def _coerce_neg_risk(value) -> bool | None:
    if _metadata_missing(value):
        return None
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _fee_rate_from_mapping(values: dict) -> float | None:
    fee_details = values.get("fee_details")
    if isinstance(fee_details, dict):
        for key in ("r", "rate", "fee_rate", "feeRate"):
            if not _metadata_missing(fee_details.get(key)):
                return _safe_float(fee_details.get(key), 0.0)

    fee_schedule = values.get("fee_schedule") or values.get("feeSchedule")
    if isinstance(fee_schedule, dict):
        for key in ("taker_fee_rate", "takerFeeRate", "taker", "fee_rate", "feeRate"):
            if not _metadata_missing(fee_schedule.get(key)):
                return _safe_float(fee_schedule.get(key), 0.0)

    for key in ("fee_rate", "taker_fee_rate", "takerBaseFee", "taker_base_fee"):
        if not _metadata_missing(values.get(key)):
            return _safe_float(values.get(key), 0.0)
    for key in ("fee_rate_bps", "taker_fee_rate_bps", "takerBaseFeeBps"):
        if not _metadata_missing(values.get(key)):
            return _safe_float(values.get(key), 0.0) / 10_000.0
    return None


def _fee_exponent_from_mapping(values: dict) -> float | None:
    fee_details = values.get("fee_details")
    if isinstance(fee_details, dict):
        for key in ("e", "exponent", "fee_exponent", "feeExponent"):
            if not _metadata_missing(fee_details.get(key)):
                return _safe_float(fee_details.get(key), 1.0)
    for key in ("fee_exponent", "feeExponent"):
        if not _metadata_missing(values.get(key)):
            return _safe_float(values.get(key), 1.0)
    return None


def _expected_taker_fee_per_share(price: float, fee_rate: float, fee_exponent: float) -> float:
    if price <= 0 or price >= 1 or fee_rate <= 0:
        return 0.0
    base = max(price * (1.0 - price), 0.0)
    return fee_rate * (base ** max(fee_exponent, 0.0))


def _wallet_position_retry_wait_seconds(*, attempt: int, response: Optional[requests.Response] = None) -> float:
    retry_after = ""
    if response is not None:
        retry_after = str(getattr(response, "headers", {}).get("Retry-After", "")).strip()
    try:
        if retry_after:
            return max(float(retry_after), 1.0)
    except ValueError:
        pass
    if getattr(response, "status_code", None) == 429:
        return float(min(10 * attempt, 60))
    return float(min(2 ** (attempt - 1), 8))


def _fetch_wallet_positions_for_reconciliation(wallet_address: str) -> list[dict] | None:
    from src.config import POLYMARKET_DATA_API_URL

    wallet = str(wallet_address or "").strip()
    if not wallet:
        return None

    now = time.monotonic()
    cached_entry = _WALLET_POSITION_FETCH_CACHE.get(wallet)
    if cached_entry is not None:
        cached_at, cached_positions = cached_entry
        if (now - cached_at) <= _WALLET_POSITION_CACHE_TTL_SECONDS:
            return list(cached_positions)

    rate_limited_until = _WALLET_POSITION_RATE_LIMIT_UNTIL.get(wallet, 0.0)
    if now < rate_limited_until:
        if cached_entry is not None:
            cached_at, cached_positions = cached_entry
            logger.warning(
                "Polymarket positions endpoint still rate-limited; reusing cached wallet positions from %.0fs ago",
                now - cached_at,
            )
            return list(cached_positions)
        logger.warning(
            "Skipping wallet-position reconciliation fetch: Polymarket positions endpoint still rate-limited for %.0fs",
            rate_limited_until - now,
        )
        return None

    last_exc: Exception | None = None
    cooldown_seconds = 0.0
    for attempt in range(1, 4):
        response = None
        try:
            response = requests.get(
                f"{POLYMARKET_DATA_API_URL}/positions",
                params={"user": wallet},
                timeout=30,
            )
            if response.status_code == 429:
                cooldown_seconds = _wallet_position_retry_wait_seconds(
                    attempt=attempt,
                    response=response,
                )
                if attempt < 3:
                    time.sleep(cooldown_seconds)
                    continue
                _WALLET_POSITION_RATE_LIMIT_UNTIL[wallet] = time.monotonic() + cooldown_seconds
                break

            response.raise_for_status()
            live_positions = [
                position for position in response.json()
                if _safe_float(position.get("size"), 0.0) > 0
            ]
            _WALLET_POSITION_FETCH_CACHE[wallet] = (time.monotonic(), live_positions)
            _WALLET_POSITION_RATE_LIMIT_UNTIL.pop(wallet, None)
            return list(live_positions)
        except requests.RequestException as exc:
            last_exc = exc
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 429:
                cooldown_seconds = _wallet_position_retry_wait_seconds(
                    attempt=attempt,
                    response=getattr(exc, "response", None),
                )
                if attempt < 3:
                    time.sleep(cooldown_seconds)
                    continue
                _WALLET_POSITION_RATE_LIMIT_UNTIL[wallet] = time.monotonic() + cooldown_seconds
                break
            if attempt < 3 and (
                status_code is None or status_code in _WALLET_POSITION_RETRY_STATUSES
            ):
                time.sleep(_wallet_position_retry_wait_seconds(attempt=attempt, response=response))
                continue
            break
        except Exception as exc:
            last_exc = exc
            break

    cached_entry = _WALLET_POSITION_FETCH_CACHE.get(wallet)
    if cached_entry is not None:
        cached_at, cached_positions = cached_entry
        logger.warning(
            "Failed to fetch wallet positions for reconciliation; reusing cached snapshot from %.0fs ago: %s",
            time.monotonic() - cached_at,
            last_exc or "rate limited",
        )
        return list(cached_positions)

    if cooldown_seconds > 0:
        logger.warning(
            "Failed to fetch wallet positions for reconciliation: rate limited; skipping reconciliation for %.0fs",
            cooldown_seconds,
        )
        return None

    if last_exc is not None:
        logger.warning("Failed to fetch wallet positions for reconciliation: %s", last_exc)
    return None


def _available_cash(bankroll) -> float:
    return _safe_float(
        getattr(bankroll, "available_cash", getattr(bankroll, "bankroll", 0.0)),
        0.0,
    )


def _sizing_bankroll(bankroll) -> float:
    return _safe_float(
        getattr(bankroll, "total_equity", getattr(bankroll, "bankroll", 0.0)),
        0.0,
    )


def _candidate_key(entry) -> tuple[str, str]:
    market_id = str(entry.get("market_id", "") or "").strip()
    side = str(entry.get("bet_side", entry.get("side", "")) or "").strip().lower()
    fighter = str(entry.get("bet_on", entry.get("fighter", "")) or "").strip().lower()
    if market_id and side:
        return (market_id, side)
    return (fighter, side)


def _parse_event_timestamp(value) -> Optional[pd.Timestamp]:
    if value is None or str(value).strip() == "":
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if isinstance(ts, pd.Series):
        return None
    return ts


def _abbreviated_name_match(name1: str, name2: str) -> bool:
    tokens1 = normalize_cross_source_name(name1).split()
    tokens2 = normalize_cross_source_name(name2).split()
    if not tokens1 or not tokens2:
        return False
    if len(tokens1) == 1 and tokens1[0] == tokens2[-1]:
        return True
    if len(tokens2) == 1 and tokens2[0] == tokens1[-1]:
        return True
    return False


def _single_name_match_score(name1: str, name2: str) -> int:
    if same_person_name(name1, name2):
        return 3
    if _abbreviated_name_match(name1, name2):
        return 2
    if _name_match(name1, name2):
        return 1
    return 0


def _parse_placed_at(raw) -> Optional[datetime]:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _open_order_id(order: dict) -> Optional[str]:
    if not isinstance(order, dict):
        return None
    oid = order.get("id") or order.get("order_id") or order.get("orderID")
    return str(oid) if oid else None


def _unwrap_clob_order(payload) -> dict:
    if not isinstance(payload, dict):
        return {}
    for key in ("order", "data"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return nested
    return payload


def _normalize_order_status(raw_status) -> str:
    return str(raw_status or "").strip().lower().replace("-", "_").replace(" ", "_")


def _get_placement_process_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _placement_locks_guard:
        if key not in _placement_locks:
            _placement_locks[key] = threading.Lock()
        return _placement_locks[key]


def _coordinated_ledger_paths(ledger_path: Path) -> tuple[Path, ...]:
    resolved_path = Path(ledger_path).resolve()
    try:
        import src.strategy.duo_trader as duo_trader
    except ImportError as e:
        if getattr(e, "name", None) != "src.strategy.duo_trader":
            raise
        return (resolved_path,)

    trader_paths: list[Path] = []
    for attr_name in ("SINGLE_LEDGER", "CONVICTION_LEDGER"):
        raw_path = getattr(duo_trader, attr_name, None)
        if raw_path is None:
            continue
        resolved = Path(raw_path).resolve()
        if resolved != resolved_path and not resolved.exists():
            continue
        if resolved not in trader_paths:
            trader_paths.append(resolved)

    if resolved_path not in trader_paths:
        return (resolved_path,)

    return tuple(sorted(trader_paths, key=lambda path: str(path)))


def _placement_lock_scope(
    *,
    market_id: str,
    token_id: str,
    fighter: str,
    side: str,
    dry_run: bool,
) -> tuple[str, str, str]:
    normalized_side = str(side or "").strip().lower()
    run_mode = "dry_run" if dry_run else "live"
    normalized_market = str(market_id or "").strip()
    normalized_token = str(token_id or "").strip()
    normalized_fighter = str(fighter or "").strip().casefold()
    lock_side = normalized_side

    if normalized_market:
        candidate = f"market:{normalized_market}"
        lock_side = ""
    elif normalized_token:
        candidate = f"token:{normalized_token}"
    else:
        candidate = f"fighter:{normalized_fighter}"

    return run_mode, lock_side, candidate


def _placement_lock_path(ledger_path: Path, scope: tuple[str, str, str]) -> Path:
    coordinated_paths = _coordinated_ledger_paths(ledger_path)
    raw_key = "|".join((*[str(path) for path in coordinated_paths], *scope))
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    lock_root = coordinated_paths[0].parent
    return lock_root / f".{digest}.order.lock"


def _skip_for_insufficient_cash(bankroll, fighter: str, amount: float) -> bool:
    available = _available_cash(bankroll)
    if amount <= available + 1e-9:
        return False
    logger.info(
        "  Skipping %s: needs $%.2f but only $%.2f available cash is free",
        fighter,
        amount,
        available,
    )
    return True


def _skip_for_min_order_size(fighter: str, amount: float) -> bool:
    if amount >= POLYMARKET_MIN_ORDER_USD:
        return False
    logger.info(
        "  Skipping %s: order size $%.2f is below Polymarket $%.2f minimum",
        fighter,
        amount,
        POLYMARKET_MIN_ORDER_USD,
    )
    return True


def _round_down_decimal_places(value: float, places: int) -> float:
    factor = 10 ** places
    return math.floor((float(value) + 1e-12) * factor) / factor


def _round_up_decimal_places(value: float, places: int) -> float:
    factor = 10 ** places
    return math.ceil((float(value) - 1e-12) * factor) / factor


def _adjust_buy_limit_for_min_notional(
    fighter: str,
    *,
    price: float,
    amount: float,
) -> tuple[float, float]:
    """Return a limit BUY amount/share pair that survives CLOB size rounding.

    The CLOB rounds limit order size down to two decimal places before
    validating BUY notional. A nominal $2.00 order can therefore fall
    slightly below the minimum at submission time. Bump only near-minimum
    BUY limit orders.
    """
    if price <= 0:
        return amount, 0.0

    shares = amount / price
    if amount < POLYMARKET_MIN_ORDER_USD - 1e-9:
        return amount, shares

    rounded_submission_shares = _round_down_decimal_places(
        shares,
        POLYMARKET_LIMIT_SIZE_DECIMALS,
    )
    rounded_submission_amount = rounded_submission_shares * price
    if rounded_submission_amount >= POLYMARKET_MIN_ORDER_USD - 1e-9:
        return amount, shares

    min_shares = _round_up_decimal_places(
        POLYMARKET_MIN_ORDER_USD / price,
        POLYMARKET_LIMIT_SIZE_DECIMALS,
    )
    reserve_amount = _round_up_decimal_places(min_shares * price, 2)
    logger.info(
        "  Increasing BUY limit on %s from $%.4f to $%.2f "
        "(%.2f shares @ $%.4f) to clear Polymarket's $%.2f minimum",
        fighter,
        amount,
        reserve_amount,
        min_shares,
        price,
        POLYMARKET_MIN_ORDER_USD,
    )
    return max(amount, reserve_amount), max(shares, min_shares)


def _bet_size_multiplier(bet: pd.Series) -> float:
    raw = bet.get("size_multiplier", 1.0)
    try:
        multiplier = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if math.isnan(multiplier) or multiplier <= 0:
        return 1.0
    return multiplier


@contextmanager
def _placement_attempt_lock(
    ledger_path: Path,
    *,
    market_id: str,
    token_id: str,
    fighter: str,
    side: str,
    dry_run: bool,
):
    scope = _placement_lock_scope(
        market_id=market_id,
        token_id=token_id,
        fighter=fighter,
        side=side,
        dry_run=dry_run,
    )
    lock_path = _placement_lock_path(ledger_path, scope)
    process_lock = _get_placement_process_lock(lock_path)
    lock_handle = None

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with process_lock:
        try:
            lock_handle = open(lock_path, "a+b")
            _acquire_file_lock(lock_handle)
            yield
        finally:
            if lock_handle is not None:
                _release_file_lock(lock_handle)
                lock_handle.close()


class OrderExecutor:
    """Executes orders on Polymarket based on model predictions."""

    def __init__(
        self,
        bankroll: BankrollManager,
        clob_client: Optional[ClobClientWrapper] = None,
        dry_run: bool = True,
        *,
        min_edge_threshold: float = MIN_EDGE_THRESHOLD,
        edge_scaling_base: float | None = None,
        skip_wallet_conflict_check: bool = False,
        force_market_order: bool = False,
        force_limit_order: bool = False,
    ):
        """
        Args:
            bankroll: BankrollManager instance
            clob_client: Authenticated CLOB client (None = dry run)
            dry_run: If True, log orders but don't actually place them
        """
        self.bankroll = bankroll
        self.clob = clob_client or ClobClientWrapper()
        self.dry_run = dry_run
        self.order_log: list[dict] = []
        self.ledger = BetLedger()
        self.min_edge_threshold = float(min_edge_threshold)
        self.edge_scaling_base = (
            float(edge_scaling_base)
            if edge_scaling_base is not None
            else float(min_edge_threshold)
        )
        self.skip_wallet_conflict_check = bool(skip_wallet_conflict_check)
        self.force_market_order = bool(force_market_order)
        self.force_limit_order = bool(force_limit_order)
        self._live_positions_cache: tuple[float, list[dict]] | None = None
        self._open_orders_cache: tuple[float, list[dict]] | None = None
        self._execution_metadata_cache: dict[str, dict] = {}
        self.decision_audit_callback: Optional[Callable[[pd.Series | dict, dict], None]] = None
        self.decision_audit_trader: str = ""

    def _audit_order_decision(
        self,
        bet: pd.Series | dict,
        *,
        status: str,
        gate: str,
        explanation: str,
        order: Optional[dict] = None,
        numbers: Optional[dict] = None,
    ) -> None:
        callback = getattr(self, "decision_audit_callback", None)
        if not callable(callback):
            return
        try:
            callback(
                bet,
                {
                    "status": status,
                    "gate": gate,
                    "explanation": explanation,
                    "order": order or {},
                    "numbers": numbers or {},
                },
            )
        except Exception as exc:
            logger.debug("Decision audit callback failed: %s", exc)

    def execute_value_bets(
        self,
        predictions: pd.DataFrame,
        markets: pd.DataFrame,
        min_edge: float = MIN_EDGE_THRESHOLD,
    ) -> list[dict]:
        """
        Match model predictions to Polymarket markets and place orders.

        Args:
            predictions: DataFrame with prob_a, prob_b for each fight
            markets: DataFrame from get_ufc_fight_markets()
            min_edge: minimum edge to place a bet

        Returns list of order results.
        """
        if markets.empty:
            logger.warning("No markets available")
            return []

        # Match predictions to markets by fighter names
        matched = self._match_predictions_to_markets(predictions, markets)
        if matched.empty:
            logger.warning("No predictions matched to active markets")
            return []

        # Find value bets
        value_bets = find_value_bets(
            matched,
            min_edge=min_edge,
            edge_scaling_base=min_edge,
        )
        if value_bets.empty:
            logger.info("No value bets identified")
            return []

        orders = []
        for _, bet in value_bets.iterrows():
            order = self._place_bet(bet, markets)
            if order:
                orders.append(order)
            time.sleep(1)  # Rate limiting

        return orders

    def _cache_execution_metadata(self, keys: list[str], metadata: dict) -> None:
        cleaned = {
            key: value
            for key, value in metadata.items()
            if key in _EXECUTION_METADATA_FIELDS and not _metadata_missing(value)
        }
        if not cleaned:
            return
        for raw_key in keys:
            key = str(raw_key or "").strip()
            if key:
                self._execution_metadata_cache[key] = {
                    **self._execution_metadata_cache.get(key, {}),
                    **cleaned,
                }

    @staticmethod
    def _metadata_from_market_row(row) -> dict:
        values = row.to_dict() if hasattr(row, "to_dict") else dict(row or {})
        metadata: dict = {}
        if not _metadata_missing(values.get("tick_size")):
            metadata["tick_size"] = str(values.get("tick_size"))
        neg_risk = _coerce_neg_risk(values.get("neg_risk"))
        if neg_risk is not None:
            metadata["neg_risk"] = neg_risk
        fee_rate = _fee_rate_from_mapping(values)
        if fee_rate is not None:
            metadata["fee_rate"] = fee_rate
            metadata.setdefault("fee_source", "gamma")
        fee_exponent = _fee_exponent_from_mapping(values)
        if fee_exponent is not None:
            metadata["fee_exponent"] = fee_exponent
        return metadata

    @staticmethod
    def _metadata_from_clob_market_info(info: dict) -> tuple[dict, list[str]]:
        metadata: dict = {}
        if not isinstance(info, dict):
            return metadata, []

        tick_size = (
            info.get("mts")
            or info.get("minimum_tick_size")
            or info.get("minimumTickSize")
            or info.get("tick_size")
        )
        if not _metadata_missing(tick_size):
            metadata["tick_size"] = str(tick_size)

        neg_risk = _coerce_neg_risk(info.get("nr", info.get("neg_risk")))
        if neg_risk is not None:
            metadata["neg_risk"] = neg_risk

        fee_details = info.get("fd") or info.get("fee_details") or info.get("feeDetails") or {}
        if isinstance(fee_details, dict):
            fee_rate = _fee_rate_from_mapping({"fee_details": fee_details})
            if fee_rate is not None:
                metadata["fee_rate"] = fee_rate
                metadata["fee_source"] = "clob"
            fee_exponent = _fee_exponent_from_mapping({"fee_details": fee_details})
            if fee_exponent is not None:
                metadata["fee_exponent"] = fee_exponent

        token_ids: list[str] = []
        for token in info.get("t") or info.get("tokens") or []:
            if not isinstance(token, dict):
                continue
            token_id = token.get("t") or token.get("token_id") or token.get("asset_id")
            if token_id:
                token_ids.append(str(token_id))
        return metadata, token_ids

    def _hydrate_execution_metadata(
        self,
        bet: pd.Series,
        markets: pd.DataFrame,
        *,
        token_id: str,
        fighter: str,
    ) -> Optional[pd.Series]:
        """Resolve tick size, neg-risk, and fee parameters before live execution."""
        hydrated = bet.copy()
        keys = [
            str(hydrated.get("condition_id", "") or "").strip(),
            str(hydrated.get("market_id", "") or "").strip(),
            str(token_id or "").strip(),
        ]

        merged: dict = {}
        for key in keys:
            if key:
                merged.update(self._execution_metadata_cache.get(key, {}))

        merged.update(self._metadata_from_market_row(hydrated))

        if not markets.empty:
            for _, market in markets.iterrows():
                market_values = market.to_dict()
                market_keys = {
                    str(market_values.get("condition_id", "") or "").strip(),
                    str(market_values.get("market_id", "") or "").strip(),
                    str(market_values.get("token_id_yes", "") or "").strip(),
                    str(market_values.get("token_id_no", "") or "").strip(),
                }
                if any(key and key in market_keys for key in keys):
                    merged.update(self._metadata_from_market_row(market_values))
                    break

        needs_canonical = (
            _metadata_missing(merged.get("tick_size"))
            or _coerce_neg_risk(merged.get("neg_risk")) is None
            or _metadata_missing(merged.get("fee_rate"))
            or str(merged.get("fee_source", "")).lower() != "clob"
        )
        condition_id = str(hydrated.get("condition_id", "") or "").strip()
        if needs_canonical and condition_id and hasattr(self.clob, "get_clob_market_info"):
            try:
                clob_info = self.clob.get_clob_market_info(condition_id)
                clob_metadata, token_ids = self._metadata_from_clob_market_info(clob_info)
                if clob_metadata:
                    merged.update(clob_metadata)
                    self._cache_execution_metadata(
                        [condition_id, *token_ids, str(hydrated.get("market_id", "") or "")],
                        clob_metadata,
                    )
            except Exception as exc:
                logger.warning(
                    "Could not hydrate CLOB execution metadata for %s (%s): %s",
                    fighter,
                    condition_id,
                    exc,
                )

        tick_size = merged.get("tick_size")
        neg_risk = _coerce_neg_risk(merged.get("neg_risk"))
        if _metadata_missing(tick_size) or neg_risk is None:
            logger.warning(
                "Skipping %s: unresolved execution metadata after CLOB hydration "
                "(tick_size=%r, neg_risk=%r, condition_id=%s, token_id=%s)",
                fighter,
                tick_size,
                merged.get("neg_risk"),
                condition_id or "?",
                token_id or "?",
            )
            return None

        hydrated["tick_size"] = str(tick_size)
        hydrated["neg_risk"] = bool(neg_risk)
        fee_rate = merged.get("fee_rate")
        if not _metadata_missing(fee_rate):
            hydrated["fee_rate"] = _safe_float(fee_rate, 0.0)
        fee_exponent = merged.get("fee_exponent")
        if not _metadata_missing(fee_exponent):
            hydrated["fee_exponent"] = _safe_float(fee_exponent, 1.0)
        elif not _metadata_missing(fee_rate) and _safe_float(fee_rate, 0.0) > 0:
            hydrated["fee_exponent"] = 1.0
        if not _metadata_missing(merged.get("fee_source")):
            hydrated["fee_source"] = merged["fee_source"]

        self._cache_execution_metadata(keys, self._metadata_from_market_row(hydrated))
        return hydrated

    def _match_predictions_to_markets(
        self,
        predictions: pd.DataFrame,
        markets: pd.DataFrame,
    ) -> pd.DataFrame:
        """Match model predictions to Polymarket markets using fighter identity + event time."""
        matched_rows = []

        for _, pred in predictions.iterrows():
            pred_a = str(pred.get("fighter_a", "")).strip()
            pred_b = str(pred.get("fighter_b", "")).strip()
            pred_event_ts = _parse_event_timestamp(
                pred.get("event_date", pred.get("commence_time"))
            )
            best_match = None
            best_sort_key = None

            for _, market in markets.iterrows():
                mkt_a = str(market.get("fighter_a", "")).strip()
                mkt_b = str(market.get("fighter_b", "")).strip()

                direct_score = (
                    _single_name_match_score(pred_a, mkt_a),
                    _single_name_match_score(pred_b, mkt_b),
                )
                reverse_score = (
                    _single_name_match_score(pred_a, mkt_b),
                    _single_name_match_score(pred_b, mkt_a),
                )
                reverse = False
                if min(direct_score) > 0:
                    name_score = sum(direct_score)
                elif min(reverse_score) > 0:
                    name_score = sum(reverse_score)
                    reverse = True
                else:
                    continue

                market_event_ts = _parse_event_timestamp(market.get("event_date"))
                event_gap_days = 0
                if pred_event_ts is not None and market_event_ts is not None:
                    event_gap_days = abs((pred_event_ts.normalize() - market_event_ts.normalize()).days)
                    if event_gap_days > 3:
                        continue
                sort_key = (
                    name_score,
                    -(event_gap_days if pred_event_ts is not None and market_event_ts is not None else 99),
                    _safe_float(market.get("volume"), 0.0),
                    _safe_float(market.get("liquidity"), 0.0),
                )
                if best_sort_key is None or sort_key > best_sort_key:
                    best_sort_key = sort_key
                    best_match = (market, reverse)

            if best_match is None:
                continue

            market, reverse = best_match
            row = pred.to_dict()
            if not reverse:
                # Market YES token = fighter_a wins
                row["a_market_prob"] = _market_price_or_default(market.get("price_yes"))
                row["b_market_prob"] = _market_price_or_default(market.get("price_no"))
                row["token_id_yes"] = market.get("token_id_yes", "")
                row["token_id_no"] = market.get("token_id_no", "")
            else:
                # Swap: market YES = pred fighter_b
                row["a_market_prob"] = _market_price_or_default(market.get("price_no"))
                row["b_market_prob"] = _market_price_or_default(market.get("price_yes"))
                row["token_id_yes"] = market.get("token_id_no", "")
                row["token_id_no"] = market.get("token_id_yes", "")
            if not (
                _valid_market_probability(row["a_market_prob"])
                and _valid_market_probability(row["b_market_prob"])
            ):
                logger.warning(
                    "Skipping market match for %s vs %s because Polymarket price data is missing or invalid",
                    row.get("fighter_a", ""),
                    row.get("fighter_b", ""),
                )
                continue
            row["market_id"] = market.get("market_id", "")
            row["condition_id"] = market.get("condition_id", "")
            row["tick_size"] = market.get("tick_size")
            row["neg_risk"] = market.get("neg_risk")
            for fee_col in ("fee_rate", "fee_exponent", "fee_source", "fee_schedule"):
                if fee_col in market and not _metadata_missing(market.get(fee_col)):
                    row[fee_col] = market.get(fee_col)
            row["volume"] = market.get("volume", 0)
            row["liquidity"] = market.get("liquidity", 0)
            row["market_event_date"] = market.get("event_date", "")
            row["event_title"] = market.get("event_title", "")
            matched_rows.append(row)

        result = pd.DataFrame(matched_rows)
        logger.info(f"Matched {len(result)} predictions to markets")
        return result

    def _check_liquidity(
        self,
        token_id: str,
        price: float,
        desired_size_usd: float,
        fighter: str,
        *,
        allow_partial_remainder: bool = False,
    ) -> dict:
        """
        Check orderbook liquidity before placing an order.

        Returns dict with:
            - ok: whether the order should proceed
            - adjusted_size: recommended bet size (may be reduced)
            - available_liquidity: total USD available at or near price
            - slippage: estimated price impact
            - reason: why the order was blocked (if ok=False)
        """
        result = {
            "ok": True,
            "adjusted_size": desired_size_usd,
            "available_liquidity": 0.0,
            "best_ask_liquidity": 0.0,
            "slippage": 0.0,
            "best_ask": None,
            "reason": "",
        }

        try:
            book = self.clob.get_orderbook(token_id)
        except Exception as e:
            logger.warning(f"Could not fetch orderbook for {fighter}: {e}")
            result["ok"] = False
            result["reason"] = f"orderbook fetch failed: {e}"
            return result

        # We're buying, so we look at ask side (sellers)
        asks = book.get("asks", [])
        if not asks:
            result["ok"] = False
            result["reason"] = "no asks in orderbook"
            return result

        # Walk the ask side to calculate available liquidity and slippage
        total_shares = 0.0
        total_cost = 0.0
        best_ask = float(asks[0]["price"])
        best_ask_cost = 0.0
        result["best_ask"] = best_ask

        for level in asks:
            level_price = float(level["price"])
            level_size = float(level["size"])
            level_cost = level_price * level_size

            total_shares += level_size
            total_cost += level_cost
            if abs(level_price - best_ask) <= 1e-9:
                best_ask_cost += level_cost

            # Stop if we've found enough to fill our order
            if total_cost >= desired_size_usd * 1.5:
                break

        result["available_liquidity"] = total_cost
        result["best_ask_liquidity"] = best_ask_cost

        if allow_partial_remainder:
            if total_cost < MIN_BOOK_LIQUIDITY:
                logger.info(
                    "  %s: thin book ($%.0f sampled, $%.2f at best ask); "
                    "using marketable limit so any unfilled remainder can rest",
                    fighter,
                    total_cost,
                    best_ask_cost,
                )
            return result

        # Check 1: Minimum liquidity
        if total_cost < MIN_BOOK_LIQUIDITY:
            result["ok"] = False
            result["reason"] = f"insufficient liquidity (${total_cost:.0f} < ${MIN_BOOK_LIQUIDITY:.0f} min)"
            return result

        # Check 2: Don't take too much of the book
        max_size_from_book = total_cost * MAX_BET_VS_BOOK_RATIO
        if desired_size_usd > max_size_from_book:
            result["adjusted_size"] = max_size_from_book
            logger.info(
                f"  Reducing bet on {fighter}: ${desired_size_usd:.2f} -> "
                f"${max_size_from_book:.2f} (25% of ${total_cost:.0f} book)"
            )

        # Check 3: Estimate slippage (walk the book for our order size)
        filled_cost = 0.0
        filled_shares = 0.0
        worst_price = best_ask
        order_size = result["adjusted_size"]

        for level in asks:
            level_price = float(level["price"])
            level_size = float(level["size"])
            remaining = order_size - filled_cost

            if remaining <= 0:
                break

            take_cost = min(level_price * level_size, remaining)
            take_shares = take_cost / level_price
            filled_cost += take_cost
            filled_shares += take_shares
            worst_price = level_price

        if filled_shares > 0:
            avg_fill_price = filled_cost / filled_shares
            slippage = (avg_fill_price - best_ask) / best_ask if best_ask > 0 else 0
            result["slippage"] = slippage

            if slippage > MAX_SLIPPAGE:
                result["ok"] = False
                result["reason"] = (
                    f"slippage too high ({slippage:.1%} > {MAX_SLIPPAGE:.0%}) "
                    f"for ${order_size:.2f} order"
                )
                return result

        return result

    def _market_buy_fee_view(
        self,
        bet: pd.Series,
        *,
        price: float,
        amount: float,
        blended_prob: float,
    ) -> dict:
        fee_rate = _safe_float(bet.get("fee_rate"), 0.0)
        fee_exponent = _safe_float(bet.get("fee_exponent"), 1.0 if fee_rate > 0 else 0.0)
        fee_per_share = _expected_taker_fee_per_share(price, fee_rate, fee_exponent)
        shares = amount / price if price > 0 else 0.0
        fee_amount = shares * fee_per_share
        net_price = price + fee_per_share
        return {
            "fee_rate": fee_rate,
            "fee_exponent": fee_exponent,
            "fee_per_share": fee_per_share,
            "fee_amount": fee_amount,
            "gross_edge": blended_prob - price,
            "net_edge": blended_prob - net_price,
            "net_price": net_price,
        }

    def _build_limit_candidate_lookup(
        self,
        primary_bets: Optional[pd.DataFrame],
        limit_only_bets: Optional[pd.DataFrame],
    ) -> dict[tuple[str, str], dict]:
        lookup: dict[tuple[str, str], dict] = {}

        for mode, bets in (("primary", primary_bets), ("limit_only", limit_only_bets)):
            if bets is None or bets.empty:
                continue
            for _, bet in bets.iterrows():
                key = _candidate_key(bet)
                if not any(key):
                    continue
                if mode == "limit_only" and key in lookup:
                    continue
                lookup[key] = {"mode": mode, "bet": bet.copy()}

        return lookup

    def _resolve_open_clob_order(
        self,
        ledger_bet: dict,
        open_orders: list[dict],
    ) -> Optional[dict]:
        order_id = str(ledger_bet.get("order_id", "") or "").strip()
        if order_id:
            for order in open_orders:
                if _open_order_id(order) == order_id:
                    return order

        token_id = str(ledger_bet.get("token_id", "") or "").strip()
        if not token_id:
            return None

        target_price = round(_safe_float(ledger_bet.get("price"), -1.0), 4)
        target_shares = _safe_float(ledger_bet.get("shares"), 0.0)
        candidates = []
        for order in open_orders:
            if str(order.get("asset_id", "") or "").strip() != token_id:
                continue
            if round(_safe_float(order.get("price"), -1.0), 4) != target_price:
                continue
            candidates.append(order)

        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1 and target_shares > 0:
            size_matches = [
                order
                for order in candidates
                if abs(
                    _safe_float(
                        order.get("original_size", order.get("size")),
                        target_shares,
                    ) - target_shares
                ) <= 0.01
            ]
            if len(size_matches) == 1:
                return size_matches[0]

        return None

    def _order_has_partial_fill(self, ledger_bet: dict, open_order: dict) -> bool:
        metrics = self._order_fill_metrics(ledger_bet, open_order)
        return (
            metrics["size_matched"] > 1e-9
            and metrics["size_remaining"] > 1e-9
        )

    def _order_fill_metrics(self, ledger_bet: dict, order: dict) -> dict:
        order = _unwrap_clob_order(order)
        shares_fallback = _safe_float(ledger_bet.get("shares"), 0.0)
        original_size = _safe_float(
            order.get("original_size", order.get("size")),
            shares_fallback,
        )
        if original_size <= 0:
            original_size = shares_fallback

        size_matched = _safe_float(order.get("size_matched"), 0.0)
        status = _normalize_order_status(
            order.get("status") or order.get("order_status") or order.get("state")
        )
        filledish = any(
            token in status
            for token in ("match", "fill", "execut", "complete")
        )
        if filledish and size_matched <= 0 and shares_fallback > 0:
            size_matched = min(shares_fallback, original_size or shares_fallback)

        if original_size > 0:
            size_matched = min(size_matched, original_size)
        size_remaining = max(original_size - size_matched, 0.0)
        return {
            "order": order,
            "status": status,
            "original_size": original_size,
            "size_matched": size_matched,
            "size_remaining": size_remaining,
        }

    def _order_status_is_resting(self, status: str) -> bool:
        return any(
            token in status
            for token in ("live", "open", "rest", "unmatch", "active", "delay")
        )

    def _lookup_closed_clob_order(self, ledger_bet: dict) -> tuple[bool, Optional[dict]]:
        order_id = str(ledger_bet.get("order_id", "") or "").strip()
        if not order_id or not hasattr(self.clob, "get_order"):
            return False, None

        try:
            return True, _unwrap_clob_order(self.clob.get_order(order_id))
        except KeyError:
            return self._lookup_order_from_trade_history(ledger_bet, order_id)
        except Exception as e:
            msg = str(e).lower()
            if "404" in msg or "not found" in msg:
                return self._lookup_order_from_trade_history(ledger_bet, order_id)
            logger.warning(
                f"Failed to fetch closed order {order_id} for "
                f"{ledger_bet.get('fighter', '?')}: {e}"
            )
            return False, None

    def _lookup_order_from_trade_history(
        self,
        ledger_bet: dict,
        order_id: str,
    ) -> tuple[bool, Optional[dict]]:
        if not hasattr(self.clob, "get_trades"):
            return False, None

        token_id = str(ledger_bet.get("token_id", "") or "").strip() or None
        placed_at = _parse_placed_at(ledger_bet.get("placed_at"))
        after = None
        if placed_at is not None:
            # Give the trade query a small buffer so fills just before ledger write
            # or just after cancel aren't filtered out by a tight timestamp bound.
            after = max(int(placed_at.timestamp()) - 300, 0)

        try:
            from py_clob_client_v2.clob_types import TradeParams

            params = TradeParams(
                asset_id=token_id,
                after=after,
            )
        except Exception:
            params = None

        try:
            trades = self.clob.get_trades(params=params)
        except Exception as e:
            logger.warning(
                f"Failed to query trade history for order {order_id} "
                f"({ledger_bet.get('fighter', '?')}): {e}"
            )
            return False, None

        matched_shares = 0.0
        saw_non_final_trade = False
        for trade in trades or []:
            trade_status = _normalize_order_status(trade.get("status"))
            maker_orders = trade.get("maker_orders") or trade.get("makerOrders") or []
            for maker_order in maker_orders:
                maker_order_id = str(
                    maker_order.get("order_id")
                    or maker_order.get("orderID")
                    or maker_order.get("id")
                    or ""
                ).strip()
                if maker_order_id != order_id:
                    continue
                trade_matched = _safe_float(
                    maker_order.get("matched_amount", maker_order.get("matchedAmount")),
                    _safe_float(
                        maker_order.get("size_matched", maker_order.get("maker_amount")),
                        0.0,
                    ),
                )
                if any(token in trade_status for token in ("confirm", "complete", "success")):
                    matched_shares += trade_matched
                elif "fail" not in trade_status:
                    saw_non_final_trade = True

        if saw_non_final_trade:
            return False, None

        original_size = _safe_float(ledger_bet.get("shares"), 0.0)
        if original_size > 0:
            matched_shares = min(matched_shares, original_size)

        # Trade history can confirm fills, but an empty trade set does not prove
        # that an unfilled order was cancelled or disappeared from the book.
        if matched_shares <= 1e-9:
            return False, None

        status = "confirmed_via_trades"
        return True, {
            "id": order_id,
            "status": status,
            "price": ledger_bet.get("price"),
            "original_size": ledger_bet.get("shares"),
            "size_matched": matched_shares,
        }

    def _inspect_limit_order_state(
        self,
        ledger_bet: dict,
        open_orders: list[dict],
    ) -> dict:
        resolved_order = self._resolve_open_clob_order(ledger_bet, open_orders)
        resolved_order_id = _open_order_id(resolved_order) or str(
            ledger_bet.get("order_id", "") or ""
        ).strip() or None

        if resolved_order is not None:
            return {
                "state": "resting",
                "order": resolved_order,
                "order_id": resolved_order_id,
                "reason": None,
            }

        looked_up, closed_order = self._lookup_closed_clob_order(ledger_bet)
        if not looked_up or not closed_order:
            return {
                "state": "unknown",
                "order": None,
                "order_id": resolved_order_id,
                "reason": None,
            }

        metrics = self._order_fill_metrics(ledger_bet, closed_order)
        resolved_order_id = _open_order_id(closed_order) or resolved_order_id
        if self._order_status_is_resting(metrics["status"]):
            return {
                "state": "resting",
                "order": closed_order,
                "order_id": resolved_order_id,
                "reason": None,
            }

        return {
            "state": "closed",
            "order": closed_order,
            "order_id": resolved_order_id,
            "reason": metrics["status"] or "not_on_clob",
        }

    def _closed_limit_order_state_from_lookup(
        self,
        ledger_bet: dict,
        *,
        fallback_order_id: Optional[str] = None,
    ) -> Optional[dict]:
        looked_up, closed_order = self._lookup_closed_clob_order(ledger_bet)
        if not looked_up or not closed_order:
            return None

        metrics = self._order_fill_metrics(ledger_bet, closed_order)
        status = metrics["status"]
        if self._order_status_is_resting(status):
            return None
        if not status and metrics["size_remaining"] > 1e-9:
            return None

        return {
            "state": "closed",
            "order": closed_order,
            "order_id": _open_order_id(closed_order) or fallback_order_id,
            "reason": status or "not_on_clob",
        }

    def _release_reserved_cash(
        self,
        amount: float,
        fighter: str,
        reason: str,
        ledger: Optional[BetLedger] = None,
    ) -> None:
        if amount <= 0:
            return
        if ledger is not None and ledger is not self.ledger:
            return
        self.bankroll.release_bet(amount, fighter, reason=reason)

    def _ledger_bets(
        self,
        ledger: Optional[BetLedger] = None,
        *,
        fresh: bool = False,
    ) -> list[dict]:
        target = ledger or self.ledger
        getter = getattr(target, "get_bets", None)
        if callable(getter):
            return getter(fresh=fresh)
        return list(getattr(target, "bets", []))

    def _ledger_open_bets(
        self,
        ledger: Optional[BetLedger] = None,
        *,
        fresh: bool = False,
    ) -> list[dict]:
        target = ledger or self.ledger
        getter = getattr(target, "get_open_bets", None)
        if callable(getter):
            return getter(fresh=fresh)
        return list(getattr(target, "open_bets", []))

    def _coordinated_open_bets(self) -> list[dict]:
        current_path = self.ledger.path.resolve()
        coordinated_bets: list[dict] = []

        for ledger_path in _coordinated_ledger_paths(self.ledger.path):
            target_ledger = self.ledger if ledger_path == current_path else BetLedger(path=ledger_path)
            fresh = ledger_path == current_path
            for bet in self._ledger_open_bets(target_ledger, fresh=fresh):
                coordinated_bets.append(
                    {
                        **dict(bet),
                        "_ledger_path": str(ledger_path),
                    }
                )

        return coordinated_bets

    def _get_open_orders_cached(
        self,
        *,
        force_refresh: bool = False,
        ttl_seconds: float = 5.0,
    ) -> list[dict]:
        if self.dry_run:
            return []
        now = time.monotonic()
        if (
            not force_refresh
            and self._open_orders_cache is not None
            and now - self._open_orders_cache[0] <= ttl_seconds
        ):
            return list(self._open_orders_cache[1])
        open_orders = self.clob.get_open_orders()
        self._open_orders_cache = (now, list(open_orders))
        return list(open_orders)

    def _get_live_positions_cached(
        self,
        *,
        force_refresh: bool = False,
        ttl_seconds: float = 15.0,
    ) -> list[dict]:
        if self.dry_run:
            return []
        now = time.monotonic()
        if (
            not force_refresh
            and self._live_positions_cache is not None
            and now - self._live_positions_cache[0] <= ttl_seconds
        ):
            return list(self._live_positions_cache[1])
        from src.polymarket.monitor import PositionMonitor

        monitor = PositionMonitor(clob_client=self.clob)
        positions = list(monitor.get_positions(strict=True))
        self._live_positions_cache = (now, positions)
        return list(positions)

    def _authoritative_open_clob_order_conflict(
        self,
        *,
        token_ids: set[str],
        fighter: str,
        force_refresh: bool = True,
    ) -> tuple[bool, str]:
        if self.dry_run:
            return False, ""

        normalized_tokens = {
            str(token_id or "").strip()
            for token_id in token_ids
            if str(token_id or "").strip()
        }
        if not normalized_tokens:
            return False, ""

        if not hasattr(self.clob, "get_open_orders"):
            return False, ""

        try:
            clob_open = self._get_open_orders_cached(force_refresh=force_refresh)
        except Exception as exc:
            logger.warning(
                "  CLOB duplicate check failed for %s: %s — blocking new resting order until the exchange state is confirmed",
                fighter,
                exc,
            )
            return True, f"could not verify existing CLOB orders: {exc}"

        clob_dupes = []
        for order in clob_open:
            payload = _unwrap_clob_order(order)
            asset_id = str(
                payload.get("asset_id", payload.get("token_id", "")) or ""
            ).strip()
            if asset_id in normalized_tokens:
                clob_dupes.append(payload)

        if clob_dupes:
            matched_token = str(
                clob_dupes[0].get("asset_id", clob_dupes[0].get("token_id", "")) or ""
            ).strip() or next(iter(normalized_tokens))
            return True, (
                f"found {len(clob_dupes)} open CLOB order(s) on token "
                f"{matched_token[:16]}..."
            )

        return False, ""

    def _authoritative_wallet_conflict(
        self,
        *,
        token_ids: set[str],
        fighter: str,
    ) -> tuple[bool, str]:
        if self.dry_run:
            return False, ""

        normalized_tokens = {str(token_id or "").strip() for token_id in token_ids if str(token_id or "").strip()}
        if not normalized_tokens:
            return False, ""

        try:
            live_positions = self._get_live_positions_cached()
        except Exception as exc:
            logger.warning(
                "  Live position duplicate check failed for %s: %s — blocking new order until wallet state is confirmed",
                fighter,
                exc,
            )
            return True, f"could not verify live wallet positions: {exc}"

        for position in live_positions:
            asset_id = str(position.get("asset", position.get("token_id", "")) or "").strip()
            if asset_id in normalized_tokens:
                size = _safe_float(position.get("size"), 0.0)
                return True, (
                    f"wallet already holds a live position on token {asset_id[:16]}... "
                    f"(size {size:.4f})"
                )

        return self._authoritative_open_clob_order_conflict(
            token_ids=normalized_tokens,
            fighter=fighter,
            force_refresh=True,
        )

    def _invalidate_live_state_cache(self) -> None:
        self._live_positions_cache = None
        self._open_orders_cache = None

    def _ledger_for_entry(self, ledger_bet: dict) -> BetLedger:
        ledger_path = ledger_bet.get("_ledger_path")
        if not ledger_path:
            return self.ledger
        path = Path(ledger_path).resolve()
        if path == self.ledger.path.resolve():
            return self.ledger
        return BetLedger(path=path)

    def _pending_submission_reason(self, order_type: str, detail: str) -> str:
        return f"{order_type} submission unresolved: {detail}"

    def _journal_live_order_attempt(
        self,
        *,
        fighter: str,
        opponent: str,
        side: str,
        amount: float,
        price: float,
        shares: float,
        token_id: str,
        market_id: str,
        condition_id: str = "",
        model_prob: float,
        market_prob: float,
        edge: float,
        decimal_odds: float,
        event_date: str,
        market_event_date: str = "",
        order_type: str,
        reason: str = "",
        metadata: dict | None = None,
    ) -> dict:
        return self.ledger.add_bet(
            fighter=fighter,
            opponent=opponent,
            side=side,
            amount=amount,
            price=price,
            shares=shares,
            token_id=token_id,
            market_id=market_id,
            condition_id=condition_id,
            model_prob=model_prob,
            market_prob=market_prob,
            edge=edge,
            decimal_odds=decimal_odds,
            dry_run=False,
            event_date=event_date,
            market_event_date=market_event_date,
            order_type=order_type,
            order_id=None,
            placement_state="pending_submit",
            reason=reason,
            metadata=metadata,
        )

    def _update_submission_state(
        self,
        ledger_bet: dict,
        *,
        placement_state: str,
        submission_error: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> None:
        target_ledger = self._ledger_for_entry(ledger_bet)
        updates = {
            "placement_state": placement_state,
            "submission_error": submission_error,
        }
        if order_id is not None:
            updates["order_id"] = order_id
        result = target_ledger.update_bet_fields(int(ledger_bet["id"]), **updates)
        if not result.ok:
            self._log_ledger_mutation_blocked(
                result,
                fighter=str(ledger_bet.get("fighter", "?")),
                bet_id=int(ledger_bet["id"]),
                action=f"update submission state to {placement_state}",
            )

    def _cancel_submission_attempt(self, ledger_bet: dict, *, reason: str) -> None:
        target_ledger = self._ledger_for_entry(ledger_bet)
        result = target_ledger.cancel_bet(int(ledger_bet["id"]), reason=reason)
        if not result.ok:
            self._log_ledger_mutation_blocked(
                result,
                fighter=str(ledger_bet.get("fighter", "?")),
                bet_id=int(ledger_bet["id"]),
                action="cancel failed submission",
            )
            return
        target_ledger.update_bet_fields(
            int(ledger_bet["id"]),
            require_open=False,
            placement_state="failed",
            submission_error=reason,
        )

    def _reload_ledger_entry(self, ledger_bet: dict) -> dict:
        target_ledger = self._ledger_for_entry(ledger_bet)
        bet_id = int(ledger_bet["id"])
        for fresh_bet in self._ledger_bets(target_ledger, fresh=True):
            if int(fresh_bet.get("id", -1)) == bet_id:
                merged = dict(fresh_bet)
                if ledger_bet.get("_ledger_path"):
                    merged["_ledger_path"] = ledger_bet["_ledger_path"]
                return merged
        return dict(ledger_bet)

    def _reconcile_unresolved_submission(
        self,
        ledger_bet: dict,
        *,
        open_orders: Optional[list[dict]] = None,
    ) -> dict:
        placement_state = str(ledger_bet.get("placement_state", "") or "").strip().lower()
        order_id = str(ledger_bet.get("order_id", "") or "").strip()
        if self.dry_run or order_id or placement_state not in {"pending_submit", "unknown"}:
            return ledger_bet
        if ledger_bet.get("order_type") not in _RESTING_LIMIT_ORDER_TYPES:
            if placement_state == "pending_submit":
                self._update_submission_state(
                    ledger_bet,
                    placement_state="unknown",
                    submission_error=ledger_bet.get("submission_error")
                    or self._pending_submission_reason(
                        str(ledger_bet.get("order_type", "order") or "order"),
                        "could not confirm order on CLOB",
                    ),
                )
            return self._reload_ledger_entry(ledger_bet)

        try:
            clob_open_orders = open_orders if open_orders is not None else self.clob.get_open_orders()
        except Exception as exc:
            logger.warning(
                "Could not reconcile unresolved %s submission for %s: %s",
                ledger_bet.get("order_type", "order"),
                ledger_bet.get("fighter", "?"),
                exc,
            )
            return ledger_bet

        resolved_order = self._resolve_open_clob_order(ledger_bet, clob_open_orders)
        if resolved_order is not None:
            resolved_order_id = _open_order_id(resolved_order)
            if resolved_order_id:
                self._update_submission_state(
                    ledger_bet,
                    placement_state="submitted",
                    submission_error=None,
                    order_id=resolved_order_id,
                )
                logger.warning(
                    "Recovered order id %s for %s from the CLOB after an unresolved submission",
                    resolved_order_id,
                    ledger_bet.get("fighter", "?"),
                )
            return self._reload_ledger_entry(ledger_bet)

        if placement_state == "pending_submit":
            self._update_submission_state(
                ledger_bet,
                placement_state="unknown",
                submission_error=ledger_bet.get("submission_error")
                or self._pending_submission_reason(
                    str(ledger_bet.get("order_type", "order") or "order"),
                    "could not match a resting order on the CLOB",
                ),
            )
        return self._reload_ledger_entry(ledger_bet)

    @staticmethod
    def _log_ledger_mutation_blocked(
        result,
        *,
        fighter: str,
        bet_id: int,
        action: str,
    ) -> None:
        if result.status == "not_found":
            logger.info(
                "Skipping %s for %s: bet #%s was not found in the ledger",
                action,
                fighter,
                bet_id,
            )
            return
        if result.status == "not_open":
            logger.info(
                "Skipping %s for %s: bet #%s is no longer open",
                action,
                fighter,
                bet_id,
            )
            return
        if result.status == "invalid_order_type":
            logger.info(
                "Skipping %s for %s: bet #%s is no longer a resting limit order",
                action,
                fighter,
                bet_id,
            )
            return
        logger.info(
            "Skipping %s for %s: bet #%s returned ledger status %s",
            action,
            fighter,
            bet_id,
            getattr(result, "status", "unknown"),
        )

    def _reconcile_closed_limit_order(
        self,
        ledger_bet: dict,
        *,
        reason: str,
        order_data: Optional[dict] = None,
        ledger: Optional[BetLedger] = None,
    ) -> str:
        target_ledger = ledger or self.ledger
        fighter = str(ledger_bet.get("fighter", "?"))
        amount = _safe_float(ledger_bet.get("amount"), 0.0)
        price = _safe_float(ledger_bet.get("price"), 0.0)
        order_id = str(ledger_bet.get("order_id", "") or "").strip() or None

        metrics = self._order_fill_metrics(ledger_bet, order_data or {})
        size_matched = round(metrics["size_matched"], 2)

        if size_matched > 1e-9:
            filled_amount = round(size_matched * price, 2)
            refund_amount = max(round(amount - filled_amount, 2), 0.0)

            result = target_ledger.convert_limit_bet_to_position(
                ledger_bet["id"],
                filled_shares=size_matched,
                cancel_reason=reason if refund_amount > 0 else None,
            )
            if not result.ok:
                self._log_ledger_mutation_blocked(
                    result,
                    fighter=fighter,
                    bet_id=ledger_bet["id"],
                    action="filled-limit reconciliation",
                )
                return "unchanged"
            self._release_reserved_cash(
                refund_amount,
                fighter,
                reason=reason,
                ledger=target_ledger,
            )
            logger.info(
                f"Reconciled {fighter}: preserved {size_matched:.2f} filled shares"
                f"{f' and released ${refund_amount:.2f}' if refund_amount > 0 else ''}"
                f" ({reason})"
            )
            self.order_log.append(
                {
                    "fighter": fighter,
                    "status": "reconciled",
                    "order_type": ledger_bet.get("order_type"),
                    "cancel_reason": reason if refund_amount > 0 else None,
                    "bet_id": ledger_bet.get("id"),
                    "dry_run": self.dry_run,
                    "order_id": order_id,
                    "filled_shares": size_matched,
                    "released_amount": refund_amount,
                }
            )
            return "position"

        result = target_ledger.cancel_bet(
            ledger_bet["id"],
            reason=reason,
            expected_order_types=_RESTING_LIMIT_ORDER_TYPES,
        )
        if not result.ok:
            self._log_ledger_mutation_blocked(
                result,
                fighter=fighter,
                bet_id=ledger_bet["id"],
                action="limit cancellation reconciliation",
            )
            return "unchanged"
        self._release_reserved_cash(
            amount,
            fighter,
            reason=reason,
            ledger=target_ledger,
        )
        logger.info(
            f"Reconciled {fighter}: order is no longer resting on the CLOB ({reason})"
        )
        self.order_log.append(
            {
                "fighter": fighter,
                "status": "cancelled",
                "order_type": ledger_bet.get("order_type"),
                "cancel_reason": reason,
                "bet_id": ledger_bet.get("id"),
                "dry_run": self.dry_run,
                "order_id": order_id,
            }
        )
        return "cancelled"

    def _finalize_cancelled_limit_order(
        self,
        ledger_bet: dict,
        *,
        reason: str,
        ledger: Optional[BetLedger] = None,
    ) -> bool:
        target_ledger = ledger or self.ledger
        fighter = str(ledger_bet.get("fighter", "?"))
        order_id = str(ledger_bet.get("order_id", "") or "").strip() or None

        state = {
            "state": "unknown",
            "order": None,
            "order_id": order_id,
            "reason": None,
        }
        attempts = max(1, POST_CANCEL_CONFIRMATION_ATTEMPTS)
        for attempt in range(attempts):
            post_cancel_open_orders: list[dict] = []
            if hasattr(self.clob, "get_open_orders"):
                try:
                    post_cancel_open_orders = self.clob.get_open_orders()
                except Exception as e:
                    logger.warning(
                        f"Failed to refresh open orders after cancelling {order_id or '?'} "
                        f"for {fighter}: {e}"
                    )

            state = self._inspect_limit_order_state(ledger_bet, post_cancel_open_orders)
            if state["state"] == "resting":
                looked_up_state = self._closed_limit_order_state_from_lookup(
                    ledger_bet,
                    fallback_order_id=state["order_id"] or order_id,
                )
                if looked_up_state is not None:
                    state = looked_up_state

            if state["state"] != "resting" or attempt + 1 >= attempts:
                break

            time.sleep(POST_CANCEL_CONFIRMATION_RETRY_SECONDS)

        if state["state"] == "closed":
            outcome = self._reconcile_closed_limit_order(
                ledger_bet,
                reason=reason,
                order_data=state["order"],
                ledger=target_ledger,
            )
            return outcome in ("cancelled", "position")

        if state["state"] == "resting":
            logger.warning(
                f"Cancel for {fighter} was not confirmed: order {state['order_id'] or order_id or '?'} "
                f"still appears to be resting on the CLOB"
            )
        else:
            logger.warning(
                f"Cancel for {fighter} succeeded but the post-cancel state for order "
                f"{order_id or '?'} could not be confirmed; leaving the ledger unchanged"
            )
        return False

    def _count_prior_upward_reprices(self, ledger_bet: dict) -> int:
        market_id = str(ledger_bet.get("market_id", "") or "")
        fighter = str(ledger_bet.get("fighter", "") or "")
        return sum(
            1
            for bet in self._ledger_bets(fresh=True)
            if str(bet.get("market_id", "") or "") == market_id
            and str(bet.get("fighter", "") or "") == fighter
            and bet.get("cancel_reason") == "reprice_up"
            and _ledger_entry_blocks_new_order(bet, self.dry_run)
        )

    def _cancel_limit_order_for_refresh(
        self,
        ledger_bet: dict,
        reason: str,
        resolved_order_id: Optional[str] = None,
    ) -> bool:
        fighter = str(ledger_bet.get("fighter", "?"))
        amount = _safe_float(ledger_bet.get("amount"), 0.0)

        if self.dry_run:
            result = self.ledger.cancel_bet(
                ledger_bet["id"],
                reason=reason,
                expected_order_types=_RESTING_LIMIT_ORDER_TYPES,
            )
            if not result.ok:
                self._log_ledger_mutation_blocked(
                    result,
                    fighter=fighter,
                    bet_id=ledger_bet["id"],
                    action="dry-run limit cancellation",
                )
                return False
            self.bankroll.release_bet(amount, fighter, reason=reason)
            logger.info(
                f"Cancelled simulated limit order for {fighter}: "
                f"bet #{ledger_bet['id']} ({reason})"
            )
            self.order_log.append(
                {
                    "fighter": fighter,
                    "status": "cancelled",
                    "order_type": ledger_bet.get("order_type"),
                    "cancel_reason": reason,
                    "bet_id": ledger_bet.get("id"),
                    "dry_run": True,
                }
            )
            return True

        order_id = resolved_order_id or str(ledger_bet.get("order_id", "") or "").strip()
        if not order_id:
            logger.warning(
                f"Cannot refresh-manage limit order for {fighter}: "
                f"no open CLOB order ID (bet #{ledger_bet['id']})"
            )
            return False

        try:
            self.clob.cancel_order(order_id)
            self._invalidate_live_state_cache()
        except Exception as e:
            logger.warning(
                f"Failed to cancel order {order_id} for {fighter} during refresh: {e}"
            )
            return False

        return self._finalize_cancelled_limit_order(ledger_bet, reason=reason)

    def _plan_primary_limit_target(self, bet: pd.Series) -> dict:
        fighter = str(bet.get("bet_on", "?"))
        model_prob = bet["model_prob"]
        blended_prob = bet.get("blended_prob", model_prob)
        market_prob = bet["market_prob"]
        odds = bet.get("decimal_odds") or implied_prob_to_decimal_odds(market_prob)

        if bet["bet_side"] == "a":
            token_id = bet.get("token_id_yes", "")
        else:
            token_id = bet.get("token_id_no", "")

        if not token_id:
            return {"action": "keep", "reason": "missing token_id"}

        override = bet.get("override_bet_size")
        if override is not None and override > 0:
            desired_size = float(override)
        elif bet.get("conviction_score") is not None:
            desired_size = conviction_bet_size(
                model_prob=model_prob,
                bankroll=_sizing_bankroll(self.bankroll),
            )
        else:
            desired_size = self.bankroll.kelly_bet_size(blended_prob, odds)
            desired_size = round(desired_size * _bet_size_multiplier(bet), 2)
        if desired_size <= 0:
            return {"action": "none", "reason": "bet size <= 0"}

        liq = self._check_liquidity(token_id, market_prob, desired_size, fighter)
        if not liq["ok"]:
            return {"action": "keep", "reason": liq["reason"] or "liquidity unavailable"}

        live_ask = liq.get("best_ask")
        if live_ask is None or live_ask <= 0:
            return {"action": "keep", "reason": "live ask unavailable"}

        live_edge = blended_prob - live_ask
        if live_edge >= self.min_edge_threshold:
            return {
                "action": "market",
                "reason": f"live ask now offers edge {live_edge:.1%}",
                "live_ask": round(live_ask, 4),
            }

        tick = _safe_float(bet.get("tick_size"), math.nan)
        if math.isnan(tick) or tick <= 0:
            return {"action": "keep", "reason": "tick size unavailable"}
        bid_price = math.floor((blended_prob - self.min_edge_threshold) / tick) * tick
        bid_price = round(bid_price, 4)

        if bid_price <= 0 or bid_price >= live_ask:
            return {"action": "none", "reason": "no viable limit price"}

        return {"action": "limit", "price": bid_price, "tick_size": tick}

    def _plan_limit_only_target(self, bet: pd.Series) -> dict:
        blended_prob = bet.get("blended_prob", bet["model_prob"])
        market_prob = bet["market_prob"]
        tick = _safe_float(bet.get("tick_size"), math.nan)
        if math.isnan(tick) or tick <= 0:
            return {"action": "none", "reason": "tick size unavailable"}
        decimal_odds = implied_prob_to_decimal_odds(market_prob)
        required_edge = scaled_min_edge(decimal_odds, base=self.edge_scaling_base)
        bid_price = math.floor((blended_prob - required_edge) / tick) * tick
        bid_price = round(bid_price, 4)

        if bid_price <= 0:
            return {"action": "none", "reason": "bid price <= 0"}
        if bid_price >= market_prob:
            return {"action": "none", "reason": "bid would cross market"}

        bid_odds = implied_prob_to_decimal_odds(bid_price)
        bet_size = self.bankroll.kelly_bet_size(blended_prob, bid_odds)
        if bet_size <= 0:
            return {"action": "none", "reason": "kelly size <= 0"}

        return {"action": "limit", "price": bid_price, "tick_size": tick}

    def refresh_open_limit_orders(
        self,
        matched_predictions: pd.DataFrame,
        primary_bets: Optional[pd.DataFrame] = None,
        limit_only_bets: Optional[pd.DataFrame] = None,
        trader_name: str = "",
    ) -> dict:
        """
        Re-evaluate open resting limit orders against the latest model view.

        This is intentionally conservative:
        - never touch partially filled orders
        - reconcile orders that are no longer resting before managing replacements
        - only reprice after a meaningful price gap
        """
        summary = {
            "kept": 0,
            "cancelled": 0,
            "cancelled_thesis": 0,
            "cancelled_marketable": 0,
            "reconciled": 0,
            "repriced_up": 0,
            "repriced_down": 0,
        }

        open_limit_bets = [
            bet for bet in self._ledger_open_bets(fresh=True)
            if bet.get("order_type") in _RESTING_LIMIT_ORDER_TYPES
            and _ledger_entry_blocks_new_order(bet, self.dry_run)
        ]
        if not open_limit_bets:
            return summary

        has_model_view = matched_predictions is not None and not matched_predictions.empty
        if not has_model_view:
            logger.warning(
                "Limit-order refresh has no matched predictions; reconciling CLOB state "
                "only and leaving confirmed resting orders unchanged"
            )

        candidate_lookup = self._build_limit_candidate_lookup(primary_bets, limit_only_bets)

        clob_open_orders: list[dict] = []
        if not self.dry_run:
            try:
                clob_open_orders = self.clob.get_open_orders()
            except Exception as e:
                logger.warning(f"Skipping limit-order refresh: could not load open orders: {e}")
                summary["kept"] = len(open_limit_bets)
                return summary

        now = datetime.now(timezone.utc)
        age_floor = timedelta(minutes=LIMIT_REPRICE_MIN_AGE_MINUTES)

        for ledger_bet in list(open_limit_bets):
            fighter = str(ledger_bet.get("fighter", "?"))
            resolved_order_id = str(ledger_bet.get("order_id", "") or "").strip() or None

            if not self.dry_run:
                state = self._inspect_limit_order_state(ledger_bet, clob_open_orders)
                if state["state"] == "closed":
                    outcome = self._reconcile_closed_limit_order(
                        ledger_bet,
                        reason=state["reason"] or "not_on_clob",
                        order_data=state["order"],
                    )
                    if outcome == "cancelled":
                        summary["cancelled"] += 1
                    elif outcome == "position":
                        summary["reconciled"] += 1
                    continue

                if state["state"] == "unknown":
                    logger.info(
                        f"  Keeping {fighter}: order is not confirmed as resting on the CLOB"
                    )
                    summary["kept"] += 1
                    continue

                resolved_order = state["order"]
                resolved_order_id = state["order_id"] or resolved_order_id
                if self._order_has_partial_fill(ledger_bet, resolved_order):
                    logger.info(
                        f"  Keeping {fighter}: order is partially filled, leaving it alone"
                    )
                    summary["kept"] += 1
                    continue

            if not has_model_view:
                summary["kept"] += 1
                continue

            if ledger_bet.get("order_type") == _MARKETABLE_LIMIT_ORDER_TYPE:
                logger.info(
                    f"  Keeping {fighter}: marketable limit already submitted; "
                    "reconciliation handled above, repricing skipped"
                )
                summary["kept"] += 1
                continue

            candidate = candidate_lookup.get(_candidate_key(ledger_bet))
            if candidate is None:
                if self._cancel_limit_order_for_refresh(
                    ledger_bet,
                    reason="thesis_expired",
                    resolved_order_id=resolved_order_id,
                ):
                    summary["cancelled"] += 1
                    summary["cancelled_thesis"] += 1
                else:
                    summary["kept"] += 1
                continue

            if candidate["mode"] == "limit_only":
                plan = self._plan_limit_only_target(candidate["bet"])
            else:
                plan = self._plan_primary_limit_target(candidate["bet"])

            action = plan.get("action")
            if action == "keep":
                logger.info(f"  Keeping {fighter}: {plan.get('reason', 'refresh skipped')}")
                summary["kept"] += 1
                continue

            if action == "market":
                if self._cancel_limit_order_for_refresh(
                    ledger_bet,
                    reason="marketable_now",
                    resolved_order_id=resolved_order_id,
                ):
                    summary["cancelled"] += 1
                    summary["cancelled_marketable"] += 1
                else:
                    summary["kept"] += 1
                continue

            if action == "none":
                if self._cancel_limit_order_for_refresh(
                    ledger_bet,
                    reason="no_viable_limit",
                    resolved_order_id=resolved_order_id,
                ):
                    summary["cancelled"] += 1
                    summary["cancelled_thesis"] += 1
                else:
                    summary["kept"] += 1
                continue

            target_price = round(_safe_float(plan.get("price"), 0.0), 4)
            current_price = round(_safe_float(ledger_bet.get("price"), 0.0), 4)
            tick = _safe_float(plan.get("tick_size"), 0.0)
            if tick <= 0:
                logger.info(f"  Keeping {fighter}: tick size unavailable for refresh")
                summary["kept"] += 1
                continue
            diff_ticks = int(round((target_price - current_price) / tick))

            if abs(diff_ticks) < LIMIT_REPRICE_TICK_THRESHOLD:
                summary["kept"] += 1
                continue

            if diff_ticks < 0:
                if self._cancel_limit_order_for_refresh(
                    ledger_bet,
                    reason="reprice_down",
                    resolved_order_id=resolved_order_id,
                ):
                    summary["cancelled"] += 1
                    summary["repriced_down"] += 1
                else:
                    summary["kept"] += 1
                continue

            placed_at = _parse_placed_at(ledger_bet.get("placed_at"))
            if placed_at is None or now - placed_at < age_floor:
                logger.info(
                    f"  Keeping {fighter}: repricing up is gated until the order is at least "
                    f"{LIMIT_REPRICE_MIN_AGE_MINUTES}m old"
                )
                summary["kept"] += 1
                continue

            if self._count_prior_upward_reprices(ledger_bet) >= LIMIT_REPRICE_MAX_UPDATES:
                logger.info(
                    f"  Keeping {fighter}: already used {LIMIT_REPRICE_MAX_UPDATES} upward reprices"
                )
                summary["kept"] += 1
                continue

            if self._cancel_limit_order_for_refresh(
                ledger_bet,
                reason="reprice_up",
                resolved_order_id=resolved_order_id,
            ):
                summary["cancelled"] += 1
                summary["repriced_up"] += 1
            else:
                summary["kept"] += 1

        if trader_name:
            logger.info(
                f"{trader_name}: limit refresh kept {summary['kept']}, reconciled "
                f"{summary['reconciled']}, and cancelled {summary['cancelled']} "
                f"open limit order(s)"
            )

        return summary

    def _place_bet(self, bet: pd.Series, markets: pd.DataFrame) -> Optional[dict]:
        fighter = bet.get("bet_on", "")
        market_id = str(bet.get("market_id", ""))
        side = str(bet.get("bet_side", ""))
        if side == "a":
            token_id = bet.get("token_id_yes", "")
        else:
            token_id = bet.get("token_id_no", "")

        with _placement_attempt_lock(
            self.ledger.path,
            market_id=market_id,
            token_id=token_id,
            fighter=fighter,
            side=side,
            dry_run=self.dry_run,
        ):
            return self._place_bet_locked(bet, markets)

    @staticmethod
    def _bet_event_time(bet: pd.Series) -> object:
        candidates = [
            bet.get("event_date"),
            bet.get("commence_time"),
            bet.get("market_event_date"),
        ]
        for value in candidates:
            if parse_event_timestamp(value) is not None:
                return value
        return next((value for value in candidates if str(value or "").strip()), None)

    def _limit_bid_window_status(self, bet: pd.Series) -> dict | None:
        return bet_window_status(
            self._bet_event_time(bet),
            close_buffer=timedelta(hours=LIMIT_BID_PRE_EVENT_HOURS),
            fail_closed=not self.dry_run,
        )

    def _limit_bid_window_open(self, bet: pd.Series, fighter: str, *, label: str) -> bool:
        window = self._limit_bid_window_status(bet)
        if window is not None and not window["open"]:
            logger.info("  Skipping %s %s: %s", fighter, label, window["detail"])
            return False
        return True

    def _inside_limit_bid_pull_window(self, bet: pd.Series) -> bool:
        window = self._limit_bid_window_status(bet)
        return window is not None and window.get("state") == "too_late"

    def _place_bet_locked(self, bet: pd.Series, markets: pd.DataFrame) -> Optional[dict]:
        """Place a single bet on Polymarket."""
        fighter = bet["bet_on"]
        model_prob = bet["model_prob"]
        blended_prob = bet.get("blended_prob", model_prob)
        market_prob = bet["market_prob"]
        edge = bet["edge"]
        odds = bet["decimal_odds"]

        def _skip(
            gate: str,
            explanation: str,
            numbers: Optional[dict] = None,
            *,
            status: str = "skipped",
        ) -> None:
            self._audit_order_decision(
                bet,
                status=status,
                gate=gate,
                explanation=explanation,
                numbers=numbers,
            )
            return None

        def _existing_bet_audit_numbers(existing_bet: dict, **extra) -> dict:
            def _optional_existing_float(key: str):
                value = existing_bet.get(key)
                if value in (None, ""):
                    return None
                parsed = _safe_float(value, math.nan)
                return None if math.isnan(parsed) else parsed

            return {
                **{
                    "existing_ledger_id": existing_bet.get("id"),
                    "existing_order_id": existing_bet.get("order_id"),
                    "existing_price": _optional_existing_float("price"),
                    "existing_amount": _optional_existing_float("amount"),
                    "existing_shares": _optional_existing_float("shares"),
                    "existing_token_id": existing_bet.get("token_id"),
                    "existing_order_type": existing_bet.get("order_type"),
                    "existing_placement_state": existing_bet.get("placement_state"),
                    "existing_status": existing_bet.get("status"),
                    "existing_reason": existing_bet.get("reason"),
                    "existing_model_probability": _optional_existing_float("model_prob"),
                    "existing_market_probability": _optional_existing_float("market_prob"),
                    "existing_edge": _optional_existing_float("edge"),
                    "existing_placed_at": existing_bet.get("placed_at"),
                },
                **extra,
            }

        # Determine which token to buy
        if bet["bet_side"] == "a":
            token_id = bet.get("token_id_yes", "")
        else:
            token_id = bet.get("token_id_no", "")

        if not token_id:
            logger.warning(f"No token ID for {fighter}")
            return _skip(
                "executor_missing_token",
                f"Skipped by executor because no token id was available for {fighter}.",
            )

        hydrated_bet = self._hydrate_execution_metadata(
            bet,
            markets,
            token_id=token_id,
            fighter=fighter,
        )
        if hydrated_bet is None:
            return _skip(
                "executor_market_metadata",
                f"Skipped by executor because market metadata could not be hydrated for {fighter}.",
                {"token_id": token_id},
            )
        bet = hydrated_bet

        window = bet_window_status(
            self._bet_event_time(bet),
            close_buffer=(
                timedelta(hours=LIMIT_BID_PRE_EVENT_HOURS)
                if self.force_limit_order
                else None
            ),
            fail_closed=not self.dry_run,
        )
        if window is not None and not window["open"]:
            logger.info("  Skipping %s: %s", fighter, window["detail"])
            return _skip(
                (
                    "event_time_unavailable"
                    if window.get("state") == "event_time_unavailable"
                    else "bet_window"
                ),
                f"Skipped by executor because {window['detail']}",
                {"bet_window": window},
            )

        # Prevent duplicate positions on the same market
        mid = str(bet.get("market_id", ""))
        if mid:
            if self.skip_wallet_conflict_check:
                existing = [
                    b for b in self._ledger_open_bets(fresh=True)
                    if b.get("market_id") == mid
                    and _ledger_entry_blocks_new_order(b, self.dry_run)
                ]
            else:
                existing = [
                    b for b in self._coordinated_open_bets()
                    if b.get("market_id") == mid
                    and _ledger_entry_blocks_new_order(b, self.dry_run)
                ]
            if existing and not self.dry_run:
                reconciled = [self._reconcile_unresolved_submission(entry) for entry in existing]
                existing = [
                    b for b in reconciled
                    if b.get("market_id") == mid
                    and _ledger_entry_blocks_new_order(b, self.dry_run)
                ]
            if existing:
                logger.info(
                    f"  Skipping {fighter}: already have open bet on market {mid} "
                    f"(#{existing[0]['id']} @ ${existing[0]['price']:.4f})"
                )
                existing_bet = existing[0]
                return _skip(
                    "duplicate_open_position",
                    (
                        f"Already have open bet on market {mid}, "
                        f"ledger #{existing_bet.get('id')}."
                    ),
                    _existing_bet_audit_numbers(existing_bet, market_id=mid),
                    status="already_bet",
                )

        wallet_conflict = False
        conflict_reason = ""
        if not self.skip_wallet_conflict_check:
            conflict_tokens = set()
            if bet["bet_side"] == "a":
                conflict_tokens.add(str(bet.get("token_id_no", "") or "").strip())
            else:
                conflict_tokens.add(str(bet.get("token_id_yes", "") or "").strip())
            conflict_tokens.discard("")

            wallet_conflict, conflict_reason = self._authoritative_wallet_conflict(
                token_ids=conflict_tokens,
                fighter=fighter,
            )
            if wallet_conflict:
                logger.info("  Skipping %s: %s", fighter, conflict_reason)
                return _skip(
                    "wallet_conflict",
                    f"Skipped by executor because {conflict_reason}",
                    {"conflict_tokens": list(conflict_tokens)},
                )

        # Calculate preliminary bet size (using snapshot odds — may be recalculated below)
        override = bet.get("override_bet_size")
        if override is not None and override > 0:
            max_allowed = self.bankroll.total_equity * self.bankroll.max_bet_fraction
            bet_size = min(override, max_allowed)
        else:
            bet_size = self.bankroll.kelly_bet_size(blended_prob, odds)
            bet_size = round(bet_size * _bet_size_multiplier(bet), 2)
        if bet_size <= 0:
            return _skip(
                "bet_size",
                (
                    f"Skipped by executor because computed bet size was ${bet_size:.2f} "
                    f"for {fighter}."
                ),
                {"bet_size_usd": bet_size},
            )

        # Check orderbook liquidity before placing
        use_limit_bid = False
        tick = float(bet.get("tick_size"))
        market_fee_view: dict = {}
        best_ask_liquidity = 0.0
        if not self.dry_run:
            liq = self._check_liquidity(
                token_id,
                market_prob,
                bet_size,
                fighter,
                allow_partial_remainder=not self.force_limit_order,
            )
            if not liq["ok"]:
                logger.warning(f"Skipping {fighter}: {liq['reason']}")
                return _skip(
                    "liquidity",
                    f"Skipped by executor because {liq['reason']}",
                    {"liquidity": liq, "bet_size_usd": bet_size},
                )

            # Re-verify edge against the LIVE Polymarket ask price.
            # The edge was originally calculated against a snapshot price
            # that may be stale. The actual execution price is what matters.
            live_ask = liq.get("best_ask")
            if live_ask is None or live_ask <= 0:
                logger.warning(
                    f"Skipping {fighter}: could not get live ask price from orderbook"
                )
                return _skip(
                    "live_ask_unavailable",
                    "Skipped by executor because no live ask price was available from the orderbook.",
                    {"liquidity": liq, "bet_size_usd": bet_size},
                )

            if self.force_limit_order:
                use_limit_bid = True
                max_willing_price = math.floor(
                    (blended_prob - self.min_edge_threshold) / tick
                ) * tick
                max_willing_price = round(max_willing_price, 4)
                best_resting_price = round(live_ask - tick, 4)
                price = min(max_willing_price, best_resting_price)

                if price <= 0 or price >= live_ask:
                    logger.info(
                        f"  Skipping {fighter}: no viable resting limit price "
                        f"(max willing ${max_willing_price:.4f}, ask ${live_ask:.4f})"
                    )
                    return _skip(
                        "limit_price",
                        (
                            f"Skipped by executor because no viable resting limit price existed "
                            f"(max willing ${max_willing_price:.4f}, ask ${live_ask:.4f})."
                        ),
                        {
                            "max_willing_price": max_willing_price,
                            "live_ask": live_ask,
                            "tick_size": tick,
                        },
                    )

                edge = blended_prob - price
                odds = implied_prob_to_decimal_odds(price)
                logger.info(
                    f"  {fighter}: forcing resting limit bid @ ${price:.4f} "
                    f"(ask ${live_ask:.4f}, max willing ${max_willing_price:.4f}, "
                    f"edge if filled: {edge:+.1%})"
                )
            elif self.force_market_order:
                price = live_ask
                edge = blended_prob - live_ask
                odds = implied_prob_to_decimal_odds(live_ask)
                logger.info(
                    f"  {fighter}: live ask ${live_ask:.4f} "
                    f"(marketable limit forced), edge {edge:+.1%}"
                )
            else:
                live_edge = blended_prob - live_ask
                use_limit_bid = live_edge < self.min_edge_threshold

                if use_limit_bid:
                    # Don't place duplicate limit bids for the same fighter
                    existing = [
                        b for b in self._ledger_open_bets(fresh=True)
                        if b.get("fighter") == fighter
                        and b.get("order_type") in _RESTING_LIMIT_ORDER_TYPES
                        and _ledger_entry_blocks_new_order(b, self.dry_run)
                    ]
                    if existing:
                        logger.info(
                            f"  Skipping {fighter}: already have open limit bid "
                            f"(#{existing[0]['id']} @ ${existing[0]['price']:.4f})"
                        )
                        existing_bid = existing[0]
                        return _skip(
                            "duplicate_open_limit_order",
                            (
                                f"Already have open limit bid for {fighter}, "
                                f"ledger #{existing_bid.get('id')}."
                            ),
                            _existing_bet_audit_numbers(existing_bid),
                            status="already_bet",
                        )

                    # Ask is too expensive for a market buy — place a resting
                    # limit bid at a price that guarantees our minimum edge.
                    bid_price = math.floor((blended_prob - self.min_edge_threshold) / tick) * tick
                    bid_price = round(bid_price, 4)

                    if bid_price <= 0 or bid_price >= live_ask:
                        logger.info(
                            f"  Skipping {fighter}: no viable bid price "
                            f"(blended {blended_prob:.1%}, ask ${live_ask:.4f})"
                        )
                        return _skip(
                            "limit_price",
                            (
                                f"Skipped by executor because no viable bid price existed "
                                f"(blended {_safe_float(blended_prob, 0.0):.1%}, ask ${live_ask:.4f})."
                            ),
                            {
                                "bid_price": bid_price,
                                "live_ask": live_ask,
                                "blended_probability": blended_prob,
                                "required_edge": self.min_edge_threshold,
                            },
                        )

                    price = bid_price
                    edge = blended_prob - bid_price
                    odds = implied_prob_to_decimal_odds(bid_price)
                    logger.info(
                        f"  {fighter}: ask ${live_ask:.4f} too expensive "
                        f"(edge {live_edge:+.1%}), placing limit bid @ ${bid_price:.4f} "
                        f"(edge if filled: {edge:+.1%})"
                    )
                else:
                    # Ask price has edge — proceed with market buy
                    price = live_ask
                    edge = live_edge
                    odds = implied_prob_to_decimal_odds(live_ask)
                    logger.info(
                        f"  {fighter}: live ask ${live_ask:.4f} "
                        f"(snapshot was ${market_prob:.4f}), "
                        f"edge {live_edge:+.1%}"
                    )

            # Recalculate bet size with live odds (skip for override/conviction bets)
            if override is None or override <= 0:
                bet_size = self.bankroll.kelly_bet_size(blended_prob, odds)
                if bet_size <= 0:
                    return _skip(
                        "bet_size_live_odds",
                        (
                            f"Skipped by executor because live-odds Kelly sizing was "
                            f"${bet_size:.2f} for {fighter}."
                        ),
                        {"bet_size_usd": bet_size, "live_odds": odds},
                    )

            if not use_limit_bid:
                best_ask_liquidity = _safe_float(liq.get("best_ask_liquidity"), 0.0)
                if (
                    not self._inside_limit_bid_pull_window(bet)
                    and 0 < best_ask_liquidity < bet_size
                ):
                    logger.info(
                        "  %s: $%.2f immediately available at $%.4f; "
                        "$%.2f remainder can rest as a limit bid",
                        fighter,
                        best_ask_liquidity,
                        price,
                        max(bet_size - best_ask_liquidity, 0.0),
                    )

            if not use_limit_bid:
                market_fee_view = self._market_buy_fee_view(
                    bet,
                    price=price,
                    amount=bet_size,
                    blended_prob=blended_prob,
                )
                logger.info(
                    "  %s: market-buy edge gross %+0.1f%%, net %+0.1f%% "
                    "(fee_rate=%s, est_fee=$%.4f)",
                    fighter,
                    market_fee_view["gross_edge"] * 100,
                    market_fee_view["net_edge"] * 100,
                    market_fee_view["fee_rate"],
                    market_fee_view["fee_amount"],
                )
                if market_fee_view["net_edge"] < self.min_edge_threshold:
                    if self.force_market_order:
                        logger.info(
                            "  Skipping %s: market buy net edge %+0.1f%% below threshold %+0.1f%% after taker fees",
                            fighter,
                            market_fee_view["net_edge"] * 100,
                            self.min_edge_threshold * 100,
                        )
                        return _skip(
                            "taker_fee_net_edge",
                            (
                                f"Skipped by executor because market-buy net edge was "
                                f"{market_fee_view['net_edge']:+.1%}, needs "
                                f"{self.min_edge_threshold:+.1%} after taker fees."
                            ),
                            {
                                "gross_edge": market_fee_view["gross_edge"],
                                "net_edge": market_fee_view["net_edge"],
                                "fee_amount": market_fee_view["fee_amount"],
                                "required_edge": self.min_edge_threshold,
                            },
                        )

                    existing = [
                        b for b in self._ledger_open_bets(fresh=True)
                        if b.get("fighter") == fighter
                        and b.get("order_type") in _RESTING_LIMIT_ORDER_TYPES
                        and _ledger_entry_blocks_new_order(b, self.dry_run)
                    ]
                    if existing:
                        logger.info(
                            f"  Skipping {fighter}: already have open limit bid "
                            f"(#{existing[0]['id']} @ ${existing[0]['price']:.4f})"
                        )
                        existing_bid = existing[0]
                        return _skip(
                            "duplicate_open_limit_order",
                            (
                                f"Already have open limit bid for {fighter}, "
                                f"ledger #{existing_bid.get('id')}."
                            ),
                            _existing_bet_audit_numbers(existing_bid),
                            status="already_bet",
                        )

                    bid_price = math.floor((blended_prob - self.min_edge_threshold) / tick) * tick
                    bid_price = round(bid_price, 4)
                    if bid_price <= 0 or bid_price >= live_ask:
                        logger.info(
                            "  Skipping %s: no viable maker bid after taker-fee check "
                            "(blended %.1f%%, ask $%.4f, net edge %+0.1f%%)",
                            fighter,
                            blended_prob * 100,
                            live_ask,
                            market_fee_view["net_edge"] * 100,
                        )
                        return _skip(
                            "maker_bid_after_fee",
                            (
                                f"Skipped by executor because no viable maker bid remained after "
                                f"the taker-fee check (ask ${live_ask:.4f}, net edge "
                                f"{market_fee_view['net_edge']:+.1%})."
                            ),
                            {
                                "bid_price": bid_price,
                                "live_ask": live_ask,
                                "net_edge": market_fee_view["net_edge"],
                                "required_edge": self.min_edge_threshold,
                            },
                        )

                    use_limit_bid = True
                    price = bid_price
                    edge = blended_prob - bid_price
                    odds = implied_prob_to_decimal_odds(bid_price)
                    market_fee_view = {}
                    logger.info(
                        "  %s: market buy net edge below threshold after taker fees; "
                        "placing maker limit bid @ $%.4f (edge if filled: %+0.1f%%)",
                        fighter,
                        bid_price,
                        edge * 100,
                    )
                else:
                    edge = market_fee_view["net_edge"]
                    odds = implied_prob_to_decimal_odds(market_fee_view["net_price"])
        else:
            if self.force_limit_order:
                use_limit_bid = True
                max_willing_price = math.floor(
                    (blended_prob - self.min_edge_threshold) / tick
                ) * tick
                max_willing_price = round(max_willing_price, 4)
                best_resting_price = round(market_prob - tick, 4)
                candidate_prices = [value for value in (max_willing_price, best_resting_price) if value > 0]
                if not candidate_prices:
                    return _skip(
                        "limit_price",
                        (
                            f"Skipped by executor because no dry-run limit price was viable "
                            f"for {fighter}."
                        ),
                        {
                            "max_willing_price": max_willing_price,
                            "best_resting_price": best_resting_price,
                        },
                    )
                price = min(candidate_prices)
                edge = blended_prob - price
                odds = implied_prob_to_decimal_odds(price)
            else:
                price = market_prob
            # In dry run, still log what we'd check
            logger.info(
                f"  [DRY RUN] Would check orderbook for {fighter} "
                f"(token: {token_id[:16]}...)"
            )

        if use_limit_bid and not self._limit_bid_window_open(bet, fighter, label="limit bid"):
            window = self._limit_bid_window_status(bet)
            return _skip(
                "limit_bid_window",
                f"Skipped by executor because {window['detail'] if window else 'the limit-bid window is closed'}.",
                {"bet_window": window},
            )

        if price <= 0:
            return _skip(
                "execution_price",
                f"Skipped by executor because execution price was {price:.4f}.",
                {"price": price},
            )

        bet_size, shares = _adjust_buy_limit_for_min_notional(
            fighter,
            price=price,
            amount=bet_size,
        )
        if (
            not self.dry_run
            and not use_limit_bid
            and self._inside_limit_bid_pull_window(bet)
            and best_ask_liquidity + 1e-9 < bet_size
        ):
            logger.info(
                "  Skipping %s: only $%.2f available at best ask for $%.2f "
                "order inside the %sh limit-bid pull window; refusing to leave "
                "a resting remainder",
                fighter,
                best_ask_liquidity,
                bet_size,
                LIMIT_BID_PRE_EVENT_HOURS,
            )
            return _skip(
                "partial_liquidity_pull_window",
                (
                    f"Skipped by executor because only ${best_ask_liquidity:.2f} was available "
                    f"at best ask for a ${bet_size:.2f} order inside the "
                    f"{LIMIT_BID_PRE_EVENT_HOURS}h limit-bid pull window."
                ),
                {
                    "best_ask_liquidity": best_ask_liquidity,
                    "bet_size_usd": bet_size,
                    "limit_bid_pre_event_hours": LIMIT_BID_PRE_EVENT_HOURS,
                },
            )

        if _skip_for_insufficient_cash(self.bankroll, fighter, bet_size):
            return _skip(
                "insufficient_cash",
                (
                    f"Skipped by executor because available cash was "
                    f"${_available_cash(self.bankroll):.2f}, needs ${bet_size:.2f}."
                ),
                {
                    "available_cash": _available_cash(self.bankroll),
                    "bet_size_usd": bet_size,
                },
            )
        if _skip_for_min_order_size(fighter, bet_size):
            return _skip(
                "minimum_order_size",
                (
                    f"Skipped by executor because order size ${bet_size:.2f} is below "
                    f"the ${POLYMARKET_MIN_ORDER_USD:.2f} Polymarket minimum."
                ),
                {
                    "bet_size_usd": bet_size,
                    "min_order_usd": POLYMARKET_MIN_ORDER_USD,
                },
            )

        bankroll_charge_amount = bet_size

        if not self.dry_run and use_limit_bid:
            same_token_conflict, conflict_reason = self._authoritative_open_clob_order_conflict(
                token_ids={str(token_id or "")},
                fighter=fighter,
                force_refresh=False,
            )
            if same_token_conflict:
                logger.info("  Skipping %s limit bid: %s", fighter, conflict_reason)
                return _skip(
                    "duplicate_open_clob_order",
                    f"Already have an open CLOB order for {fighter}: {conflict_reason}",
                    {"token_id": token_id},
                    status="already_bet",
                )

        order_info = {
            "fighter": fighter,
            "side": "BUY",
            "token_id": token_id,
            "price": round(price, 4),
            "shares": round(shares, 2),
            "bet_size_usd": bet_size,
            "model_prob": model_prob,
            "market_prob": market_prob,
            "edge": edge,
            "dry_run": self.dry_run,
        }
        signal_metadata = _ledger_signal_metadata_from_bet(bet)
        if signal_metadata:
            order_info.update(signal_metadata)
        if market_fee_view:
            order_info.update(
                {
                    "gross_edge": market_fee_view["gross_edge"],
                    "net_edge": market_fee_view["net_edge"],
                    "estimated_taker_fee_usd": market_fee_view["fee_amount"],
                    "fee_rate": market_fee_view["fee_rate"],
                    "fee_exponent": market_fee_view["fee_exponent"],
                }
            )

        opponent = ""
        if bet["bet_side"] == "a":
            opponent = str(bet.get("fighter_b", ""))
        else:
            opponent = str(bet.get("fighter_a", ""))

        if self.dry_run:
            order_type = "limit_bid" if use_limit_bid else _MARKETABLE_LIMIT_ORDER_TYPE
            logger.info(
                f"[DRY RUN] Would place: {order_type.upper()} BUY {shares:.1f} shares "
                f"of {fighter} @ ${price:.4f} (${bet_size:.2f} total) | "
                f"Edge: {edge:.1%}"
            )
            order_info["status"] = "dry_run"
            order_info["order_type"] = order_type
            # Bankroll charged once in the common post-block handler below
            self.ledger.add_bet(
                fighter=fighter,
                opponent=opponent,
                side=bet["bet_side"],
                amount=bet_size,
                price=price,
                shares=shares,
                token_id=token_id,
                market_id=str(bet.get("market_id", "")),
                condition_id=str(bet.get("condition_id", "")),
                model_prob=model_prob,
                market_prob=market_prob,
                edge=edge,
                decimal_odds=odds,
                dry_run=True,
                event_date=str(bet.get("event_date", "")),
                market_event_date=str(bet.get("market_event_date", "")),
                order_type=order_type,
                order_id=None,
                reason=str(bet.get("reason", "")),
                metadata=signal_metadata,
            )
        elif use_limit_bid:
            pending_bet = self._journal_live_order_attempt(
                fighter=fighter,
                opponent=opponent,
                side=bet["bet_side"],
                amount=bet_size,
                price=price,
                shares=shares,
                token_id=token_id,
                market_id=str(bet.get("market_id", "")),
                condition_id=str(bet.get("condition_id", "")),
                model_prob=model_prob,
                market_prob=market_prob,
                edge=edge,
                decimal_odds=odds,
                event_date=str(bet.get("event_date", "")),
                market_event_date=str(bet.get("market_event_date", "")),
                order_type="limit_bid",
                reason=str(bet.get("reason", "")),
                metadata=signal_metadata,
            )
            order_info["ledger_bet_id"] = pending_bet["id"]
            # Place a resting limit bid — gets filled if price drops to our level
            try:
                tick_size = str(bet.get("tick_size"))
                response = self.clob.create_limit_order(
                    token_id=token_id,
                    side="BUY",
                    price=price,
                    size=shares,
                    tick_size=tick_size,
                    neg_risk=bool(bet.get("neg_risk")),
                )
                self._invalidate_live_state_cache()
                order_info["response"] = response
                order_info["order_type"] = "limit_bid"
                clob_order_id = _extract_order_id(response, warn=True)
                if clob_order_id:
                    order_info["status"] = "placed"
                    self._update_submission_state(
                        pending_bet,
                        placement_state="submitted",
                        submission_error=None,
                        order_id=clob_order_id,
                    )
                    logger.info(
                        f"Limit bid placed for {fighter}: "
                        f"BUY {shares:.1f} @ ${price:.4f} (${bet_size:.2f}) | "
                        f"Edge if filled: {edge:.1%} | {response}"
                    )
                else:
                    order_info["status"] = "unknown"
                    order_info["error"] = "limit bid response missing durable order id"
                    self._update_submission_state(
                        pending_bet,
                        placement_state="unknown",
                        submission_error=self._pending_submission_reason(
                            "limit bid",
                            "response missing durable order id",
                        ),
                    )
                    logger.error(
                        "Limit bid outcome is unknown for %s: CLOB response did not include an order id",
                        fighter,
                    )
            except Exception as e:
                if _order_failure_is_warning(e):
                    order_info["status"] = "failed"
                    order_info["error"] = str(e)
                    self._cancel_submission_attempt(
                        pending_bet,
                        reason=f"submit_failed: {e}",
                    )
                    _log_order_failure("Failed to place limit bid", fighter, e)
                else:
                    order_info["status"] = "unknown"
                    order_info["error"] = str(e)
                    self._update_submission_state(
                        pending_bet,
                        placement_state="unknown",
                        submission_error=self._pending_submission_reason("limit bid", str(e)),
                    )
                    logger.error(
                        "Limit bid outcome is unknown for %s: %s",
                        fighter,
                        e,
                    )
        else:
            pending_bet = self._journal_live_order_attempt(
                fighter=fighter,
                opponent=opponent,
                side=bet["bet_side"],
                amount=bet_size,
                price=price,
                shares=shares,
                token_id=token_id,
                market_id=str(bet.get("market_id", "")),
                condition_id=str(bet.get("condition_id", "")),
                model_prob=model_prob,
                market_prob=market_prob,
                edge=edge,
                decimal_odds=odds,
                event_date=str(bet.get("event_date", "")),
                market_event_date=str(bet.get("market_event_date", "")),
                order_type=_MARKETABLE_LIMIT_ORDER_TYPE,
                reason=str(bet.get("reason", "")),
                metadata=signal_metadata,
            )
            order_info["ledger_bet_id"] = pending_bet["id"]
            # Marketable limit: match immediately up to this price, then rest any remainder.
            try:
                tick_size = str(bet.get("tick_size"))
                response = self.clob.create_limit_order(
                    token_id=token_id,
                    side="BUY",
                    price=price,
                    size=shares,
                    tick_size=tick_size,
                    neg_risk=bool(bet.get("neg_risk")),
                )
                self._invalidate_live_state_cache()
                order_info["response"] = response
                order_info["order_type"] = _MARKETABLE_LIMIT_ORDER_TYPE
                order_info["requested_bet_size_usd"] = bet_size
                order_info["submitted_amount"] = round(bet_size, 2)
                clob_order_id = _extract_order_id(response, warn=True)
                if clob_order_id:
                    order_info["status"] = "placed"
                    amount_update = self._ledger_for_entry(pending_bet).update_bet_fields(
                        int(pending_bet["id"]),
                        requested_amount=round(bet_size, 2),
                        submitted_amount=round(bet_size, 2),
                    )
                    if not amount_update.ok:
                        self._log_ledger_mutation_blocked(
                            amount_update,
                            fighter=fighter,
                            bet_id=int(pending_bet["id"]),
                            action="update marketable-limit submitted amount",
                        )
                    self._update_submission_state(
                        pending_bet,
                        placement_state="submitted",
                        submission_error=None,
                        order_id=clob_order_id,
                    )
                    logger.info(
                        f"Marketable limit submitted for {fighter}: "
                        f"BUY {shares:.1f} @ max ${price:.4f} (${bet_size:.2f}) | "
                        f"Edge: {edge:.1%} | {response}"
                    )
                else:
                    order_info["status"] = "unknown"
                    order_info["error"] = "marketable limit response missing durable order id"
                    self._update_submission_state(
                        pending_bet,
                        placement_state="unknown",
                        submission_error=self._pending_submission_reason(
                            "marketable limit",
                            "response missing durable order id",
                        ),
                    )
                    logger.error(
                        "Marketable limit outcome is unknown for %s: CLOB response did not include an order id",
                        fighter,
                    )
            except Exception as e:
                if _order_failure_is_warning(e):
                    order_info["status"] = "failed"
                    order_info["error"] = str(e)
                    self._cancel_submission_attempt(
                        pending_bet,
                        reason=f"submit_failed: {e}",
                    )
                    _log_order_failure("Failed to place marketable limit", fighter, e)
                else:
                    order_info["status"] = "unknown"
                    order_info["error"] = str(e)
                    self._update_submission_state(
                        pending_bet,
                        placement_state="unknown",
                        submission_error=self._pending_submission_reason("marketable limit", str(e)),
                    )
                    logger.error(
                        "Marketable limit outcome is unknown for %s: %s. "
                        "Skipping automatic retry until the ledger is reconciled.",
                        fighter,
                        e,
                    )

        if order_info["status"] in ("placed", "dry_run"):
            self.bankroll.place_bet(
                amount=bankroll_charge_amount,
                fighter=fighter,
                decimal_odds=odds,
                model_prob=model_prob,
                market_prob=market_prob,
            )
        elif order_info["status"] == "unknown":
            logger.warning(
                "Order status UNKNOWN for %s ($%.2f) — bankroll NOT charged. "
                "Manual reconciliation required. Check exchange for fill status.",
                fighter,
                bet_size,
            )

        self.order_log.append(order_info)
        try:
            from src.strategy.execution_audit import summarize_order_for_explanation

            status_map = {
                "placed": "bet_placed",
                "dry_run": "dry_run",
                "failed": "order_failed",
                "unknown": "order_unknown",
            }
            self._audit_order_decision(
                bet,
                status=status_map.get(str(order_info.get("status") or ""), "order_result"),
                gate="executor_order_result",
                explanation=summarize_order_for_explanation(
                    self.decision_audit_trader,
                    order_info,
                ),
                order=order_info,
                numbers={
                    "price": price,
                    "shares": shares,
                    "bet_size_usd": bet_size,
                    "execution_edge": edge,
                    "order_type": order_info.get("order_type"),
                },
            )
        except Exception as exc:
            logger.debug("Failed to record order result in decision audit: %s", exc)
        return order_info

    def _place_near_miss_limit(self, bet: pd.Series, markets: pd.DataFrame) -> Optional[dict]:
        fighter = bet.get("bet_on", "")
        market_id = str(bet.get("market_id", ""))
        side = str(bet.get("bet_side", ""))
        if side == "a":
            token_id = bet.get("token_id_yes", "")
        else:
            token_id = bet.get("token_id_no", "")

        with _placement_attempt_lock(
            self.ledger.path,
            market_id=market_id,
            token_id=token_id,
            fighter=fighter,
            side=side,
            dry_run=self.dry_run,
        ):
            return self._place_near_miss_limit_locked(bet, markets)

    def _place_near_miss_limit_locked(self, bet: pd.Series, markets: pd.DataFrame) -> Optional[dict]:
        """Place a near-miss limit order — resting bid that guarantees MIN_EDGE if filled.

        Unlike _place_bet, this ONLY places limit bids (never market orders).
        Used for fights that pass all quality filters but barely miss the edge threshold.
        """
        fighter = bet["bet_on"]
        model_prob = bet["model_prob"]
        blended_prob = bet.get("blended_prob", model_prob)
        market_prob = bet["market_prob"]
        current_edge = bet["edge"]

        def _skip(
            gate: str,
            explanation: str,
            numbers: Optional[dict] = None,
            *,
            status: str = "skipped",
        ) -> None:
            self._audit_order_decision(
                bet,
                status=status,
                gate=gate,
                explanation=explanation,
                numbers=numbers,
            )
            return None

        # Determine which token to buy
        if bet["bet_side"] == "a":
            token_id = bet.get("token_id_yes", "")
        else:
            token_id = bet.get("token_id_no", "")

        if not token_id:
            logger.warning(f"  Near-miss skip {fighter}: no token ID")
            return _skip(
                "executor_missing_token",
                f"Skipped near-miss order because no token id was available for {fighter}.",
            )

        hydrated_bet = self._hydrate_execution_metadata(
            bet,
            markets,
            token_id=token_id,
            fighter=fighter,
        )
        if hydrated_bet is None:
            return _skip(
                "executor_market_metadata",
                f"Skipped near-miss order because market metadata could not be hydrated for {fighter}.",
                {"token_id": token_id},
            )
        bet = hydrated_bet

        window = bet_window_status(
            self._bet_event_time(bet),
            close_buffer=timedelta(hours=LIMIT_BID_PRE_EVENT_HOURS),
            fail_closed=not self.dry_run,
        )
        if window is not None and not window["open"]:
            logger.info("  Near-miss skip %s: %s", fighter, window["detail"])
            return _skip(
                (
                    "event_time_unavailable"
                    if window.get("state") == "event_time_unavailable"
                    else "limit_bid_window"
                ),
                f"Skipped near-miss order because {window['detail']}.",
                {"bet_window": window},
            )

        # Prevent duplicate positions on the same market
        mid = str(bet.get("market_id", ""))
        if mid:
            existing_market = [
                b for b in self._coordinated_open_bets()
                if b.get("market_id") == mid
                and _ledger_entry_blocks_new_order(b, self.dry_run)
            ]
            if existing_market and not self.dry_run:
                reconciled = [self._reconcile_unresolved_submission(entry) for entry in existing_market]
                existing_market = [
                    b for b in reconciled
                    if b.get("market_id") == mid
                    and _ledger_entry_blocks_new_order(b, self.dry_run)
                ]
            if existing_market:
                logger.info(
                    f"  Near-miss skip {fighter}: already have open bet on market {mid} "
                    f"(#{existing_market[0]['id']} @ ${existing_market[0]['price']:.4f})"
                )
                return _skip(
                    "duplicate_open_position",
                    f"Already have an open bet on market {mid}.",
                    {"market_id": mid, "existing_ledger_id": existing_market[0].get("id")},
                    status="already_bet",
                )

        wallet_conflict, conflict_reason = self._authoritative_wallet_conflict(
            token_ids={
                str(bet.get("token_id_yes", "") or ""),
                str(bet.get("token_id_no", "") or ""),
                str(token_id or ""),
            },
            fighter=fighter,
        )
        if wallet_conflict:
            logger.info("  Near-miss skip %s: %s", fighter, conflict_reason)
            return _skip(
                "wallet_conflict",
                f"Skipped near-miss order because {conflict_reason}",
            )

        # Duplicate check: ledger — any open limit-type order on same fighter
        existing = [
            b for b in self._ledger_open_bets(fresh=True)
            if b.get("fighter") == fighter
            and b.get("order_type") in _RESTING_LIMIT_ORDER_TYPES
            and _ledger_entry_blocks_new_order(b, self.dry_run)
        ]
        if existing:
            logger.info(
                f"  Near-miss skip {fighter}: already have open limit "
                f"(#{existing[0]['id']} @ ${existing[0]['price']:.4f})"
            )
            return _skip(
                "duplicate_open_limit_order",
                f"Already have an open limit order for {fighter}.",
                {"existing_ledger_id": existing[0].get("id")},
                status="already_bet",
            )

        same_token_conflict, conflict_reason = self._authoritative_open_clob_order_conflict(
            token_ids={str(token_id or "")},
            fighter=fighter,
            force_refresh=False,
        )
        if same_token_conflict:
            logger.info("  Near-miss skip %s: %s", fighter, conflict_reason)
            return _skip(
                "duplicate_open_clob_order",
                f"Already have an open CLOB order for {fighter}: {conflict_reason}",
                {"token_id": token_id},
                status="already_bet",
            )

        # Calculate bid price: guarantees scaled MIN_EDGE if filled
        tick = _safe_float(bet.get("tick_size"), math.nan)
        if math.isnan(tick) or tick <= 0:
            logger.info(f"  Near-miss skip {fighter}: tick size unavailable")
            return _skip(
                "tick_size",
                f"Skipped near-miss order because tick size was unavailable for {fighter}.",
            )
        decimal_odds = implied_prob_to_decimal_odds(market_prob)
        required_edge = scaled_min_edge(decimal_odds, base=self.edge_scaling_base)
        bid_price = math.floor((blended_prob - required_edge) / tick) * tick
        bid_price = round(bid_price, 4)

        if bid_price <= 0:
            logger.info(f"  Near-miss skip {fighter}: bid price <= 0")
            return _skip(
                "limit_price",
                f"Skipped near-miss order because computed bid price was {bid_price:.4f}.",
                {"bid_price": bid_price},
            )

        # Bid must be below current market (otherwise it would fill immediately
        # as a market order, which should have been caught by normal value betting)
        if bid_price >= market_prob:
            logger.info(
                f"  Near-miss skip {fighter}: bid ${bid_price:.4f} >= "
                f"market ${market_prob:.4f}"
            )
            return _skip(
                "limit_price",
                (
                    f"Skipped near-miss order because bid ${bid_price:.4f} was not below "
                    f"market ${market_prob:.4f}."
                ),
                {"bid_price": bid_price, "market_probability": market_prob},
            )

        edge_if_filled = blended_prob - bid_price
        bid_odds = implied_prob_to_decimal_odds(bid_price)

        # Size using Kelly at the bid price odds
        bet_size = self.bankroll.kelly_bet_size(blended_prob, bid_odds)
        bet_size = round(bet_size * _bet_size_multiplier(bet), 2)
        if bet_size <= 0:
            logger.info(f"  Near-miss skip {fighter}: Kelly size <= 0")
            return _skip(
                "bet_size",
                f"Skipped near-miss order because computed Kelly size was ${bet_size:.2f}.",
                {"bet_size_usd": bet_size},
            )

        bet_size, shares = _adjust_buy_limit_for_min_notional(
            fighter,
            price=bid_price,
            amount=bet_size,
        )

        if _skip_for_insufficient_cash(self.bankroll, fighter, bet_size):
            return _skip(
                "insufficient_cash",
                (
                    f"Skipped near-miss order because available cash was "
                    f"${_available_cash(self.bankroll):.2f}, needs ${bet_size:.2f}."
                ),
                {"available_cash": _available_cash(self.bankroll), "bet_size_usd": bet_size},
            )
        if _skip_for_min_order_size(fighter, bet_size):
            return _skip(
                "minimum_order_size",
                (
                    f"Skipped near-miss order because ${bet_size:.2f} is below the "
                    f"${POLYMARKET_MIN_ORDER_USD:.2f} minimum."
                ),
                {"bet_size_usd": bet_size, "min_order_usd": POLYMARKET_MIN_ORDER_USD},
            )

        logger.info(
            f"  NEAR-MISS LIMIT: {fighter} | current edge {current_edge:.1%} "
            f"(need {required_edge:.1%}) | bid @ ${bid_price:.4f} "
            f"(edge if filled: {edge_if_filled:.1%}) | ${bet_size:.2f}"
        )

        order_info = {
            "fighter": fighter,
            "side": "BUY",
            "token_id": token_id,
            "price": round(bid_price, 4),
            "shares": round(shares, 2),
            "bet_size_usd": bet_size,
            "model_prob": model_prob,
            "market_prob": market_prob,
            "edge": edge_if_filled,
            "dry_run": self.dry_run,
            "order_type": "near_miss_limit",
        }

        opponent = ""
        if bet["bet_side"] == "a":
            opponent = str(bet.get("fighter_b", ""))
        else:
            opponent = str(bet.get("fighter_a", ""))

        if self.dry_run:
            logger.info(
                f"  [DRY RUN] Would place: NEAR-MISS LIMIT BUY {shares:.1f} shares "
                f"of {fighter} @ ${bid_price:.4f} (${bet_size:.2f} total) | "
                f"Edge if filled: {edge_if_filled:.1%}"
            )
            order_info["status"] = "dry_run"
            # Bankroll charged once in the common post-block handler below
            self.ledger.add_bet(
                fighter=fighter,
                opponent=opponent,
                side=bet["bet_side"],
                amount=bet_size,
                price=bid_price,
                shares=shares,
                token_id=token_id,
                market_id=str(bet.get("market_id", "")),
                condition_id=str(bet.get("condition_id", "")),
                model_prob=model_prob,
                market_prob=market_prob,
                edge=edge_if_filled,
                decimal_odds=bid_odds,
                dry_run=True,
                event_date=str(bet.get("event_date", "")),
                market_event_date=str(bet.get("market_event_date", "")),
                order_type="near_miss_limit",
                order_id=None,
                reason=str(bet.get("reason", "")),
            )
        else:
            pending_bet = self._journal_live_order_attempt(
                fighter=fighter,
                opponent=opponent,
                side=bet["bet_side"],
                amount=bet_size,
                price=bid_price,
                shares=shares,
                token_id=token_id,
                market_id=str(bet.get("market_id", "")),
                condition_id=str(bet.get("condition_id", "")),
                model_prob=model_prob,
                market_prob=market_prob,
                edge=edge_if_filled,
                decimal_odds=bid_odds,
                event_date=str(bet.get("event_date", "")),
                market_event_date=str(bet.get("market_event_date", "")),
                order_type="near_miss_limit",
                reason=str(bet.get("reason", "")),
            )
            order_info["ledger_bet_id"] = pending_bet["id"]
            try:
                tick_size = str(bet.get("tick_size"))
                response = self.clob.create_limit_order(
                    token_id=token_id,
                    side="BUY",
                    price=bid_price,
                    size=shares,
                    tick_size=tick_size,
                    neg_risk=bool(bet.get("neg_risk")),
                )
                self._invalidate_live_state_cache()
                order_info["response"] = response
                clob_order_id = _extract_order_id(response, warn=True)
                if clob_order_id:
                    order_info["status"] = "placed"
                    self._update_submission_state(
                        pending_bet,
                        placement_state="submitted",
                        submission_error=None,
                        order_id=clob_order_id,
                    )
                    logger.info(
                        f"  Near-miss limit placed for {fighter}: "
                        f"BUY {shares:.1f} @ ${bid_price:.4f} (${bet_size:.2f}) | "
                        f"Edge if filled: {edge_if_filled:.1%} | {response}"
                    )
                else:
                    order_info["status"] = "unknown"
                    order_info["error"] = "near-miss limit response missing durable order id"
                    self._update_submission_state(
                        pending_bet,
                        placement_state="unknown",
                        submission_error=self._pending_submission_reason(
                            "near-miss limit",
                            "response missing durable order id",
                        ),
                    )
                    logger.error(
                        "Near-miss limit outcome is unknown for %s: CLOB response did not include an order id",
                        fighter,
                    )
            except Exception as e:
                if _order_failure_is_warning(e):
                    order_info["status"] = "failed"
                    order_info["error"] = str(e)
                    self._cancel_submission_attempt(
                        pending_bet,
                        reason=f"submit_failed: {e}",
                    )
                    _log_order_failure("Failed to place near-miss limit", fighter, e)
                else:
                    order_info["status"] = "unknown"
                    order_info["error"] = str(e)
                    self._update_submission_state(
                        pending_bet,
                        placement_state="unknown",
                        submission_error=self._pending_submission_reason("near-miss limit", str(e)),
                    )
                    logger.error(
                        "Near-miss limit outcome is unknown for %s: %s",
                        fighter,
                        e,
                    )

        if order_info["status"] in ("placed", "dry_run"):
            self.bankroll.place_bet(
                amount=bet_size,
                fighter=fighter,
                decimal_odds=bid_odds,
                model_prob=model_prob,
                market_prob=market_prob,
            )
        elif order_info["status"] == "unknown":
            logger.warning(
                "Near-miss limit status UNKNOWN for %s ($%.2f) — bankroll NOT charged. "
                "Manual reconciliation required.",
                fighter,
                bet_size,
            )

        self.order_log.append(order_info)
        try:
            from src.strategy.execution_audit import summarize_order_for_explanation

            status_map = {
                "placed": "bet_placed",
                "dry_run": "dry_run",
                "failed": "order_failed",
                "unknown": "order_unknown",
            }
            self._audit_order_decision(
                bet,
                status=status_map.get(str(order_info.get("status") or ""), "order_result"),
                gate="executor_near_miss_order_result",
                explanation=summarize_order_for_explanation(
                    self.decision_audit_trader,
                    order_info,
                ),
                order=order_info,
                numbers={
                    "price": bid_price,
                    "shares": shares,
                    "bet_size_usd": bet_size,
                    "execution_edge": edge_if_filled,
                    "required_edge": required_edge,
                    "current_edge": current_edge,
                    "order_type": order_info.get("order_type"),
                },
            )
        except Exception as exc:
            logger.debug("Failed to record near-miss order result in decision audit: %s", exc)
        return order_info

    def cancel_stale_limit_bids(self, ledger: Optional[BetLedger] = None) -> int:
        """
        Cancel open limit bids that are stale or approaching event time.

        A limit bid is cancelled if:
        - The event is within LIMIT_BID_PRE_EVENT_HOURS of starting, OR
        - The fight has already started (event_date <= now), OR
        - The bid has been resting longer than LIMIT_BID_TTL_HOURS

        Returns the number of orders cancelled.
        """
        # datetime already imported at module level

        target_ledger = ledger or self.ledger
        now = datetime.now(timezone.utc)
        ttl = timedelta(hours=LIMIT_BID_TTL_HOURS)
        pre_event_buffer = timedelta(hours=LIMIT_BID_PRE_EVENT_HOURS)
        cancelled = 0
        clob_open_orders: list[dict] = []

        try:
            clob_open_orders = self._get_open_orders_cached(ttl_seconds=15.0)
        except Exception as e:
            logger.warning(f"Could not load open orders for stale cleanup: {e}")

        for bet in list(self._ledger_bets(target_ledger, fresh=True)):
            if bet.get("status") != "open":
                continue
            if bet.get("order_type") not in _RESTING_LIMIT_ORDER_TYPES:
                continue
            if bet.get("dry_run"):
                continue

            fighter = bet.get("fighter", "?")
            order_id = bet.get("order_id")
            cancel_reason = None

            # Check 1: fight is about to start (cancel before event begins)
            event_date = bet.get("event_date")
            if event_date:
                try:
                    if "T" in str(event_date):
                        fight_time = datetime.fromisoformat(
                            str(event_date).replace("Z", "+00:00")
                        )
                    else:
                        fight_time = datetime.fromisoformat(str(event_date)).replace(
                            tzinfo=timezone.utc
                        )
                    if fight_time.tzinfo is None:
                        fight_time = fight_time.replace(tzinfo=timezone.utc)
                    cancel_deadline = fight_time - pre_event_buffer
                    if now >= cancel_deadline:
                        if now >= fight_time:
                            cancel_reason = "fight started"
                        else:
                            mins_left = int((fight_time - now).total_seconds() / 60)
                            cancel_reason = f"pre-event pull ({mins_left}min to event)"
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse event_date '{event_date}' for {fighter} "
                        f"(bet #{bet['id']}): {e} — pre-event check skipped"
                    )
            else:
                logger.warning(
                    f"Limit bid for {fighter} (bet #{bet['id']}) has no event_date — "
                    f"pre-event cancellation check skipped, relying on {LIMIT_BID_TTL_HOURS}h TTL"
                )

            # Check 2: bid has exceeded TTL
            if not cancel_reason:
                placed_at = bet.get("placed_at")
                if placed_at:
                    try:
                        placed_time = datetime.fromisoformat(str(placed_at))
                        if placed_time.tzinfo is None:
                            placed_time = placed_time.replace(tzinfo=timezone.utc)
                        if now - placed_time >= ttl:
                            cancel_reason = f"exceeded {LIMIT_BID_TTL_HOURS}h TTL"
                    except (ValueError, TypeError):
                        pass

            if not cancel_reason:
                continue

            state = self._inspect_limit_order_state(bet, clob_open_orders)
            if state["state"] == "closed":
                self._reconcile_closed_limit_order(
                    bet,
                    reason=state["reason"] or "not_on_clob",
                    order_data=state["order"],
                    ledger=target_ledger,
                )
                continue

            if state["state"] == "unknown":
                logger.info(
                    f"Keeping {fighter}: stale cleanup could not confirm the current "
                    f"order state on the CLOB"
                )
                continue

            resolved_order_id = state["order_id"]
            order_id = resolved_order_id or order_id
            if not order_id:
                logger.warning(
                    f"Cannot cancel limit bid for {fighter}: no order ID stored "
                    f"(bet #{bet['id']})"
                )
                continue

            try:
                self.clob.cancel_order(order_id)
                self._invalidate_live_state_cache()
                logger.info(
                    f"Cancelled limit bid for {fighter}: "
                    f"order {order_id} ({cancel_reason})"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to cancel order {order_id} for {fighter}: {e}"
                )
                continue

            if self._finalize_cancelled_limit_order(
                bet,
                reason=cancel_reason,
                ledger=target_ledger,
            ):
                cancelled += 1

        if cancelled:
            logger.info(f"Cancelled {cancelled} stale limit bid(s)")
        return cancelled

    def get_order_log(self) -> pd.DataFrame:
        """Get log of all orders placed."""
        return pd.DataFrame(self.order_log)


def _reconcile_import_positions(
    live_positions: list[dict],
    market_token_lookup: dict[str, dict],
    tracked_tokens: set[str],
    *,
    import_ledger_path: Path | None = None,
) -> int:
    """Import untracked wallet positions into the owning trader ledger."""
    from src.polymarket.tracker import BetLedger

    if import_ledger_path is None:
        from src.strategy.duo_trader import SINGLE_LEDGER

        import_ledger_path = SINGLE_LEDGER

    ledger = BetLedger(path=import_ledger_path)
    imported = 0

    for pos in live_positions:
        asset_id = str(pos.get("asset", pos.get("token_id", "")) or "").strip()
        if not asset_id or asset_id in tracked_tokens:
            continue
        token_mapping = market_token_lookup.get(asset_id)
        if token_mapping is None:
            # Expected for every non-UFC wallet position (we only map supported UFC
            # markets), so this fires constantly and is not actionable. Keep it at
            # DEBUG so it never reaches bot.log / the activity monitor at the normal
            # INFO level, while staying available when deep-debugging imports.
            logger.debug(
                "Skipping wallet position import for token %s: no unambiguous supported-market token mapping",
                asset_id,
            )
            continue

        size = _safe_float(pos.get("size"), 0.0)
        title = pos.get("title", pos.get("question", "unknown"))
        avg_price = _safe_float(pos.get("avgPrice", pos.get("avg_price", 0.5)), 0.5)
        market_id = str(
            token_mapping.get("market_id")
            or pos.get("market")
            or pos.get("condition_id")
            or ""
        ).strip()
        condition_id = str(
            token_mapping.get("condition_id")
            or pos.get("condition_id")
            or ""
        ).strip()
        fighter = str(token_mapping.get("fighter", "") or "").strip()
        opponent_raw = str(token_mapping.get("opponent", "") or "").strip()
        side = str(token_mapping.get("side", "") or "").strip().lower()
        if side not in {"a", "b"} or not fighter or not opponent_raw:
            logger.warning(
                "Skipping wallet position import for token %s: incomplete supported-market token metadata",
                asset_id,
            )
            continue

        amount = round(size * avg_price, 2)

        bet = ledger.add_bet(
            fighter=fighter,
            opponent=opponent_raw,
            side=side,
            amount=amount,
            price=avg_price,
            shares=round(size, 2),
            token_id=asset_id,
            market_id=market_id,
            model_prob=0.0,
            market_prob=avg_price,
            edge=0.0,
            decimal_odds=round(1.0 / avg_price, 4) if avg_price > 0 else 0,
            dry_run=False,
            order_type="imported",
            status="open",
            placement_state="filled",
            condition_id=condition_id,
            market_event_date=token_mapping.get("event_date", ""),
        )
        logger.info(
            "Auto-imported untracked position: %s [%s %s vs %s] (size=%.2f, price=%.4f) -> bet #%s",
            title,
            side,
            fighter,
            opponent_raw,
            size,
            avg_price,
            bet["id"],
        )
        imported += 1

    return imported


def _reconcile_closed_positions(
    live_position_tokens: set[str],
    ledger_open_bets: list[dict],
) -> int:
    """Mark ledger entries as cancelled when the wallet position is gone (manual sell)."""
    from src.polymarket.tracker import BetLedger

    closed = 0
    # Group bets by ledger path so we can update each ledger file
    bets_by_path: dict[str, list[dict]] = {}
    for bet in ledger_open_bets:
        token = str(bet.get("token_id", "") or "").strip()
        if not token:
            continue
        # Position is in the ledger but gone from the wallet
        if token not in live_position_tokens:
            path = str(bet.get("_ledger_path", ""))
            if path:
                bets_by_path.setdefault(path, []).append(bet)

    for path_str, bets in bets_by_path.items():
        ledger = BetLedger(path=Path(path_str))
        for bet in bets:
            bet_id = bet.get("_original_id", bet.get("id"))
            if bet_id is None:
                continue
            result = ledger.cancel_bet(
                int(bet_id),
                reason="position_gone_from_wallet",
            )
            if result.ok:
                logger.info(
                    "Auto-closed ledger bet #%s (%s): position no longer on wallet",
                    bet_id, bet.get("fighter", "unknown"),
                )
                closed += 1

    return closed


def assert_live_wallet_exposure_synced(
    markets: pd.DataFrame,
    clob_client: Optional[ClobClientWrapper] = None,
    *,
    import_ledger_path: Path | None = None,
) -> None:
    """Reconcile live wallet exposure with the trader ledgers.

    Instead of crashing on mismatch, this function:
    1. Imports untracked wallet positions into the owning ledger (handles manual buys)
    2. Marks ledger entries as cancelled when wallet position is gone (handles manual sells)
    """
    if markets is None or markets.empty:
        raise RuntimeError("Cannot reconcile live wallet exposure without active market metadata")

    market_token_lookup = build_market_token_lookup(markets)
    if not market_token_lookup:
        raise RuntimeError("Cannot reconcile live wallet exposure: market token mapping is empty")

    from src.polymarket.monitor import PositionMonitor
    from src.polymarket.tracker import load_all_trader_ledgers

    client = clob_client or ClobClientWrapper()
    ledger = load_all_trader_ledgers()
    ledger_open = [
        bet for bet in ledger.get_open_bets()
        if not bet.get("dry_run", True)
    ]
    tracked_tokens = {
        str(bet.get("token_id", "") or "").strip()
        for bet in ledger_open
        if str(bet.get("token_id", "") or "").strip()
    }

    monitor = PositionMonitor(clob_client=client)
    if not monitor.wallet_address:
        raise RuntimeError("Cannot reconcile live wallet positions: wallet address unavailable")

    live_positions = _fetch_wallet_positions_for_reconciliation(monitor.wallet_address)
    if live_positions is None:
        raise RuntimeError("Cannot reconcile live wallet positions: position API remained unavailable")
    live_position_tokens = {
        str(pos.get("asset", pos.get("token_id", "")) or "").strip()
        for pos in live_positions
    }

    # 1. Import untracked positions (manual buys or prior-session bets)
    imported = _reconcile_import_positions(
        live_positions,
        market_token_lookup,
        tracked_tokens,
        import_ledger_path=import_ledger_path,
    )

    # 2. Close ledger entries for positions gone from wallet (manual sells)
    closed = _reconcile_closed_positions(
        live_position_tokens, ledger_open,
    )

    if imported or closed:
        logger.info(
            "Wallet/ledger reconciliation: imported %d positions, closed %d stale entries",
            imported, closed,
        )


def cancel_all_stale_limit_bids(clob_client: Optional[ClobClientWrapper] = None) -> int:
    """
    Cancel stale limit bids across all trader ledgers.

    Called from the live betting loop before placing new bets.
    """
    from src.strategy.duo_trader import get_all_trader_ledgers
    from src.strategy.bankroll import BankrollManager

    client = clob_client or ClobClientWrapper()
    total = 0

    for label, path in get_all_trader_ledgers():
        ledger = BetLedger(path=path)
        executor = OrderExecutor(
            bankroll=BankrollManager(initial_bankroll=0, auto_detect_balance=False),
            clob_client=client,
            dry_run=False,
        )
        n = executor.cancel_stale_limit_bids(ledger=ledger)
        if n:
            logger.info(f"Trader {label}: cancelled {n} stale limit bid(s)")
        total += n

    return total


def _name_match(name1: str, name2: str) -> bool:
    """
    Fuzzy match two fighter names.
    Handles variations like "Jon Jones" vs "Jonathan Jones" or "Jon 'Bones' Jones".
    """
    if not name1 or not name2:
        return False

    # Exact match
    if name1 == name2:
        return True

    # Remove nicknames in quotes
    import re
    clean1 = re.sub(r"['\"].*?['\"]", "", name1).strip()
    clean2 = re.sub(r"['\"].*?['\"]", "", name2).strip()
    if clean1 == clean2:
        return True

    # Last name match (for cases like "Jon Jones" vs "Jonathan Jones")
    parts1 = clean1.split()
    parts2 = clean2.split()
    if parts1 and parts2 and parts1[-1] == parts2[-1]:
        # Same last name — require a prefix match from the start of the first token.
        if len(parts1) >= 2 and len(parts2) >= 2:
            first1 = parts1[0].lower()
            first2 = parts2[0].lower()
            if first1.startswith(first2) or first2.startswith(first1):
                return True

    return False
