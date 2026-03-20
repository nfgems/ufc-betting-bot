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
from typing import Optional

import pandas as pd

from src.polymarket.client import ClobClientWrapper
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
_RESTING_LIMIT_ORDER_TYPES = frozenset(("limit_bid", "limit", "near_miss_limit"))
_placement_locks: dict[str, threading.Lock] = {}
_placement_locks_guard = threading.Lock()


def _ledger_entry_blocks_new_order(entry: dict, dry_run: bool) -> bool:
    """Decide whether an open ledger entry should block a new order attempt.

    Real-money runs should ignore historical dry-run entries, but repeated
    dry-run loops should still treat prior dry-run orders as duplicates.
    """
    return (not entry.get("dry_run")) or dry_run


def _order_failure_is_warning(exc: Exception) -> bool:
    """Treat expected API/order rejections as warnings instead of hard errors."""
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

    The py_clob_client may return:
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


def _candidate_key(entry) -> tuple[str, str]:
    market_id = str(entry.get("market_id", "") or "").strip()
    side = str(entry.get("bet_side", entry.get("side", "")) or "").strip().lower()
    fighter = str(entry.get("bet_on", entry.get("fighter", "")) or "").strip().lower()
    if market_id and side:
        return (market_id, side)
    return (fighter, side)


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
        from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER
    except ImportError as e:
        if getattr(e, "name", None) != "src.strategy.duo_trader":
            raise
        return (resolved_path,)

    trader_paths: list[Path] = []
    for path in (Path(SINGLE_LEDGER), Path(CONVICTION_LEDGER)):
        resolved = path.resolve()
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
        value_bets = find_value_bets(matched, min_edge=min_edge)
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

    def _match_predictions_to_markets(
        self,
        predictions: pd.DataFrame,
        markets: pd.DataFrame,
    ) -> pd.DataFrame:
        """Match model predictions to Polymarket markets by fuzzy fighter name matching."""
        matched_rows = []

        for _, pred in predictions.iterrows():
            pred_a = str(pred.get("fighter_a", "")).lower().strip()
            pred_b = str(pred.get("fighter_b", "")).lower().strip()

            for _, market in markets.iterrows():
                mkt_a = str(market.get("fighter_a", "")).lower().strip()
                mkt_b = str(market.get("fighter_b", "")).lower().strip()

                # Check if fighters match (in either order)
                match_direct = (
                    _name_match(pred_a, mkt_a) and _name_match(pred_b, mkt_b)
                )
                match_reverse = (
                    _name_match(pred_a, mkt_b) and _name_match(pred_b, mkt_a)
                )

                if match_direct:
                    row = pred.to_dict()
                    # Market YES token = fighter_a wins
                    row["a_market_prob"] = market.get("price_yes") or 0.5
                    row["b_market_prob"] = market.get("price_no") or 0.5
                    row["token_id_yes"] = market.get("token_id_yes", "")
                    row["token_id_no"] = market.get("token_id_no", "")
                    row["market_id"] = market.get("market_id", "")
                    row["tick_size"] = market.get("tick_size", "0.01")
                    row["neg_risk"] = market.get("neg_risk", False)
                    row["volume"] = market.get("volume", 0)
                    matched_rows.append(row)
                    break

                elif match_reverse:
                    row = pred.to_dict()
                    # Swap: market YES = pred fighter_b
                    row["a_market_prob"] = market.get("price_no") or 0.5
                    row["b_market_prob"] = market.get("price_yes") or 0.5
                    row["token_id_yes"] = market.get("token_id_no", "")
                    row["token_id_no"] = market.get("token_id_yes", "")
                    row["market_id"] = market.get("market_id", "")
                    row["tick_size"] = market.get("tick_size", "0.01")
                    row["neg_risk"] = market.get("neg_risk", False)
                    row["volume"] = market.get("volume", 0)
                    matched_rows.append(row)
                    break

        result = pd.DataFrame(matched_rows)
        logger.info(f"Matched {len(result)} predictions to markets")
        return result

    def _check_liquidity(
        self,
        token_id: str,
        price: float,
        desired_size_usd: float,
        fighter: str,
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
        result["best_ask"] = best_ask

        for level in asks:
            level_price = float(level["price"])
            level_size = float(level["size"])
            level_cost = level_price * level_size

            total_shares += level_size
            total_cost += level_cost

            # Stop if we've found enough to fill our order
            if total_cost >= desired_size_usd * 1.5:
                break

        result["available_liquidity"] = total_cost

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
            from py_clob_client.clob_types import TradeParams

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

        status = "confirmed_via_trades" if matched_shares > 1e-9 else "canceled_via_trades"
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
        order_type: str,
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
            order_type=order_type,
            order_id=None,
            placement_state="pending_submit",
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
                bankroll=self.bankroll.bankroll,
            )
        else:
            desired_size = self.bankroll.kelly_bet_size(blended_prob, odds)
        if desired_size <= 0:
            return {"action": "none", "reason": "bet size <= 0"}

        liq = self._check_liquidity(token_id, market_prob, desired_size, fighter)
        if not liq["ok"]:
            return {"action": "keep", "reason": liq["reason"] or "liquidity unavailable"}

        live_ask = liq.get("best_ask")
        if live_ask is None or live_ask <= 0:
            return {"action": "keep", "reason": "live ask unavailable"}

        live_edge = blended_prob - live_ask
        if live_edge >= MIN_EDGE_THRESHOLD:
            return {
                "action": "market",
                "reason": f"live ask now offers edge {live_edge:.1%}",
                "live_ask": round(live_ask, 4),
            }

        tick = float(bet.get("tick_size", "0.01"))
        bid_price = math.floor((blended_prob - MIN_EDGE_THRESHOLD) / tick) * tick
        bid_price = round(bid_price, 4)

        if bid_price <= 0 or bid_price >= live_ask:
            return {"action": "none", "reason": "no viable limit price"}

        return {"action": "limit", "price": bid_price, "tick_size": tick}

    def _plan_limit_only_target(self, bet: pd.Series) -> dict:
        blended_prob = bet.get("blended_prob", bet["model_prob"])
        market_prob = bet["market_prob"]
        tick = float(bet.get("tick_size", "0.01"))
        decimal_odds = implied_prob_to_decimal_odds(market_prob)
        required_edge = scaled_min_edge(decimal_odds)
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
            if bet.get("order_type") in ("limit_bid", "limit", "near_miss_limit")
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
            tick = max(_safe_float(plan.get("tick_size"), 0.01), 0.0001)
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

    def _place_bet_locked(self, bet: pd.Series, markets: pd.DataFrame) -> Optional[dict]:
        """Place a single bet on Polymarket."""
        fighter = bet["bet_on"]
        model_prob = bet["model_prob"]
        blended_prob = bet.get("blended_prob", model_prob)
        market_prob = bet["market_prob"]
        edge = bet["edge"]
        odds = bet["decimal_odds"]

        # Determine which token to buy
        if bet["bet_side"] == "a":
            token_id = bet.get("token_id_yes", "")
        else:
            token_id = bet.get("token_id_no", "")

        if not token_id:
            logger.warning(f"No token ID for {fighter}")
            return None

        # Prevent duplicate positions on the same market
        mid = str(bet.get("market_id", ""))
        if mid:
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
                return None

        # Calculate preliminary bet size (using snapshot odds — may be recalculated below)
        override = bet.get("override_bet_size")
        if override is not None and override > 0:
            bet_size = override
        else:
            bet_size = self.bankroll.kelly_bet_size(blended_prob, odds)
        if bet_size <= 0:
            return None

        # Check orderbook liquidity before placing
        use_limit_bid = False
        if not self.dry_run:
            liq = self._check_liquidity(token_id, market_prob, bet_size, fighter)
            if not liq["ok"]:
                logger.warning(f"Skipping {fighter}: {liq['reason']}")
                return None

            # Re-verify edge against the LIVE Polymarket ask price.
            # The edge was originally calculated against a snapshot price
            # that may be stale. The actual execution price is what matters.
            live_ask = liq.get("best_ask")
            if live_ask is None or live_ask <= 0:
                logger.warning(
                    f"Skipping {fighter}: could not get live ask price from orderbook"
                )
                return None

            live_edge = blended_prob - live_ask
            use_limit_bid = live_edge < MIN_EDGE_THRESHOLD

            if use_limit_bid:
                # Don't place duplicate limit bids for the same fighter
                existing = [
                    b for b in self._ledger_open_bets(fresh=True)
                    if b.get("fighter") == fighter
                    and b.get("order_type") in ("limit_bid", "limit", "near_miss_limit")
                    and _ledger_entry_blocks_new_order(b, self.dry_run)
                ]
                if existing:
                    logger.info(
                        f"  Skipping {fighter}: already have open limit bid "
                        f"(#{existing[0]['id']} @ ${existing[0]['price']:.4f})"
                    )
                    return None

                # Authoritative CLOB check — catch orders the ledger missed
                try:
                    clob_open = self.clob.get_open_orders()
                    clob_dupes = [
                        o for o in clob_open
                        if o.get("asset_id") == token_id
                    ]
                    if clob_dupes:
                        logger.info(
                            f"  Skipping {fighter}: found {len(clob_dupes)} open "
                            f"CLOB order(s) on token {token_id[:16]}..."
                        )
                        return None
                except Exception as e:
                    logger.warning(
                        f"  CLOB duplicate check failed: {e} "
                        f"— proceeding with ledger-only check"
                    )

                # Ask is too expensive for a market buy — place a resting
                # limit bid at a price that guarantees our minimum edge.
                tick = float(bet.get("tick_size", "0.01"))
                bid_price = math.floor((blended_prob - MIN_EDGE_THRESHOLD) / tick) * tick
                bid_price = round(bid_price, 4)

                if bid_price <= 0 or bid_price >= live_ask:
                    logger.info(
                        f"  Skipping {fighter}: no viable bid price "
                        f"(blended {blended_prob:.1%}, ask ${live_ask:.4f})"
                    )
                    return None

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
                    return None

            # Apply liquidity adjustments from the orderbook check
            if not use_limit_bid:
                bet_size = min(bet_size, liq["adjusted_size"])
            if liq["slippage"] > 0 and not use_limit_bid:
                logger.info(
                    f"  {fighter}: ${liq['available_liquidity']:.0f} book liquidity, "
                    f"{liq['slippage']:.1%} est. slippage"
                )
        else:
            price = market_prob
            # In dry run, still log what we'd check
            logger.info(
                f"  [DRY RUN] Would check orderbook for {fighter} "
                f"(token: {token_id[:16]}...)"
            )

        # Calculate shares: bet_size / price
        shares = bet_size / price if price > 0 else 0

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

        opponent = ""
        if bet["bet_side"] == "a":
            opponent = str(bet.get("fighter_b", ""))
        else:
            opponent = str(bet.get("fighter_a", ""))

        if self.dry_run:
            order_type = "limit_bid" if use_limit_bid else "market"
            logger.info(
                f"[DRY RUN] Would place: {order_type.upper()} BUY {shares:.1f} shares "
                f"of {fighter} @ ${price:.4f} (${bet_size:.2f} total) | "
                f"Edge: {edge:.1%}"
            )
            order_info["status"] = "dry_run"
            order_info["order_type"] = order_type
            self.bankroll.place_bet(
                amount=bet_size,
                fighter=fighter,
                decimal_odds=odds,
                model_prob=model_prob,
                market_prob=market_prob,
            )
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
                order_type=order_type,
                order_id=None,
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
                order_type="limit_bid",
            )
            order_info["ledger_bet_id"] = pending_bet["id"]
            # Place a resting limit bid — gets filled if price drops to our level
            try:
                tick_size = str(bet.get("tick_size", "0.01"))
                response = self.clob.create_limit_order(
                    token_id=token_id,
                    side="BUY",
                    price=price,
                    size=shares,
                    tick_size=tick_size,
                    neg_risk=bet.get("neg_risk", False),
                )
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
                order_type="market",
            )
            order_info["ledger_bet_id"] = pending_bet["id"]
            # Market buy — ask price has edge
            try:
                tick_size = str(bet.get("tick_size", "0.01"))
                response = self.clob.create_market_order(
                    token_id=token_id,
                    side="BUY",
                    amount=bet_size,
                    tick_size=tick_size,
                    neg_risk=bet.get("neg_risk", False),
                )
                order_info["response"] = response
                order_info["order_type"] = "market"
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
                        f"Market order filled for {fighter}: "
                        f"${bet_size:.2f} | Edge: {edge:.1%} | {response}"
                    )
                else:
                    order_info["status"] = "unknown"
                    order_info["error"] = "market order response missing durable order id"
                    self._update_submission_state(
                        pending_bet,
                        placement_state="unknown",
                        submission_error=self._pending_submission_reason(
                            "market order",
                            "response missing durable order id",
                        ),
                    )
                    logger.error(
                        "Market order outcome is unknown for %s: CLOB response did not include an order id",
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
                    _log_order_failure("Failed to place market order", fighter, e)
                else:
                    order_info["status"] = "unknown"
                    order_info["error"] = str(e)
                    self._update_submission_state(
                        pending_bet,
                        placement_state="unknown",
                        submission_error=self._pending_submission_reason("market order", str(e)),
                    )
                    logger.error(
                        "Market order outcome is unknown for %s: %s. "
                        "Skipping automatic retry until the ledger is reconciled.",
                        fighter,
                        e,
                    )

        if order_info["status"] in ("placed", "dry_run"):
            self.bankroll.place_bet(
                amount=bet_size,
                fighter=fighter,
                decimal_odds=odds,
                model_prob=model_prob,
                market_prob=market_prob,
            )
        elif order_info["status"] == "unknown":
            logger.warning(
                "Market order status UNKNOWN for %s ($%.2f) — bankroll NOT charged. "
                "Manual reconciliation required. Check exchange for fill status.",
                fighter,
                bet_size,
            )

        self.order_log.append(order_info)
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

        # Determine which token to buy
        if bet["bet_side"] == "a":
            token_id = bet.get("token_id_yes", "")
        else:
            token_id = bet.get("token_id_no", "")

        if not token_id:
            logger.warning(f"  Near-miss skip {fighter}: no token ID")
            return None

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
                return None

        # Duplicate check: ledger — any open limit-type order on same fighter
        existing = [
            b for b in self._ledger_open_bets(fresh=True)
            if b.get("fighter") == fighter
            and b.get("order_type") in ("limit_bid", "limit", "near_miss_limit")
            and _ledger_entry_blocks_new_order(b, self.dry_run)
        ]
        if existing:
            logger.info(
                f"  Near-miss skip {fighter}: already have open limit "
                f"(#{existing[0]['id']} @ ${existing[0]['price']:.4f})"
            )
            return None

        # CLOB duplicate check — catch orders the ledger missed
        if not self.dry_run:
            try:
                clob_open = self.clob.get_open_orders()
                clob_dupes = [
                    o for o in clob_open
                    if o.get("asset_id") == token_id
                ]
                if clob_dupes:
                    logger.info(
                        f"  Near-miss skip {fighter}: found {len(clob_dupes)} open "
                        f"CLOB order(s) on token {token_id[:16]}..."
                    )
                    return None
            except Exception as e:
                logger.warning(
                    f"  CLOB duplicate check failed: {e} "
                    f"— proceeding with ledger-only check"
                )

        # Calculate bid price: guarantees scaled MIN_EDGE if filled
        tick = float(bet.get("tick_size", "0.01"))
        decimal_odds = implied_prob_to_decimal_odds(market_prob)
        required_edge = scaled_min_edge(decimal_odds)
        bid_price = math.floor((blended_prob - required_edge) / tick) * tick
        bid_price = round(bid_price, 4)

        if bid_price <= 0:
            logger.info(f"  Near-miss skip {fighter}: bid price <= 0")
            return None

        # Bid must be below current market (otherwise it would fill immediately
        # as a market order, which should have been caught by normal value betting)
        if bid_price >= market_prob:
            logger.info(
                f"  Near-miss skip {fighter}: bid ${bid_price:.4f} >= "
                f"market ${market_prob:.4f}"
            )
            return None

        edge_if_filled = blended_prob - bid_price
        bid_odds = implied_prob_to_decimal_odds(bid_price)

        # Size using Kelly at the bid price odds
        bet_size = self.bankroll.kelly_bet_size(blended_prob, bid_odds)
        if bet_size <= 0:
            logger.info(f"  Near-miss skip {fighter}: Kelly size <= 0")
            return None

        shares = bet_size / bid_price if bid_price > 0 else 0

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
            self.bankroll.place_bet(
                amount=bet_size,
                fighter=fighter,
                decimal_odds=bid_odds,
                model_prob=model_prob,
                market_prob=market_prob,
            )
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
                order_type="near_miss_limit",
                order_id=None,
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
                order_type="near_miss_limit",
            )
            order_info["ledger_bet_id"] = pending_bet["id"]
            try:
                tick_size = str(bet.get("tick_size", "0.01"))
                response = self.clob.create_limit_order(
                    token_id=token_id,
                    side="BUY",
                    price=bid_price,
                    size=shares,
                    tick_size=tick_size,
                    neg_risk=bet.get("neg_risk", False),
                )
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

        if order_info["status"] in ("placed", "dry_run", "unknown"):
            self.bankroll.place_bet(
                amount=bet_size,
                fighter=fighter,
                decimal_odds=bid_odds,
                model_prob=model_prob,
                market_prob=market_prob,
            )

        self.order_log.append(order_info)
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
            clob_open_orders = self.clob.get_open_orders()
        except Exception as e:
            logger.warning(f"Could not load open orders for stale cleanup: {e}")

        for bet in list(self._ledger_bets(target_ledger, fresh=True)):
            if bet.get("status") != "open":
                continue
            if bet.get("order_type") not in ("limit_bid", "limit", "near_miss_limit"):
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


def cancel_all_stale_limit_bids(clob_client: Optional[ClobClientWrapper] = None) -> int:
    """
    Cancel stale limit bids across all trader ledgers.

    Called from the live betting loop before placing new bets.
    """
    from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER
    from src.strategy.bankroll import BankrollManager

    client = clob_client or ClobClientWrapper()
    total = 0

    for label, path in [("S", SINGLE_LEDGER), ("C", CONVICTION_LEDGER)]:
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
