"""
Live bet & P&L tracker — persistent bet ledger with live-updating dashboard.

Tracks all bets placed (dry run or real), fetches current market prices,
computes unrealized/realized P&L, and provides a live terminal dashboard.
"""

import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from src.config import LOGS_DIR, INITIAL_BANKROLL
from src.strategy.value import calculate_closing_line_value

logger = logging.getLogger(__name__)

LEDGER_PATH = LOGS_DIR / "bet_ledger.json"
PNL_HISTORY_PATH = LOGS_DIR / "pnl_history.jsonl"
REDEEM_STATE_FILENAME = "polymarket_redeem_state.json"

_ledger_path_locks: dict[str, threading.Lock] = {}
_ledger_path_locks_guard = threading.Lock()
_LIMIT_ORDER_TYPES = {"limit_bid", "limit", "near_miss_limit", "marketable_limit"}
_CRYPTO_5M_ORDER_TYPE = "btc5m_marketable_limit"
_CRYPTO_5M_STRATEGY_NAME = "btc5m_momentum"
_REDEEM_SUCCESS_STATES = {"STATE_MINED", "STATE_CONFIRMED"}
_REDEEM_FAILURE_STATES = {"STATE_FAILED", "STATE_INVALID"}
_CLV_CAPTURE_WINDOW = timedelta(minutes=10)


def _lock_path_for_ledger(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


def _get_process_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _ledger_path_locks_guard:
        if key not in _ledger_path_locks:
            _ledger_path_locks[key] = threading.Lock()
        return _ledger_path_locks[key]


def _acquire_file_lock(lock_handle) -> None:
    lock_handle.seek(0, os.SEEK_END)
    if lock_handle.tell() == 0:
        lock_handle.write(b"0")
        lock_handle.flush()
    lock_handle.seek(0)

    if os.name == "nt":
        import msvcrt

        deadline = time.monotonic() + 30  # 30-second timeout
        while True:
            try:
                lock_handle.seek(0)
                msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
                lock_handle.seek(0, os.SEEK_END)
                if lock_handle.tell() > 1:
                    lock_handle.truncate(1)
                    lock_handle.flush()
                lock_handle.seek(0)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Could not acquire ledger file lock within 30 seconds"
                    )
                time.sleep(0.05)
    else:
        import fcntl

        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        lock_handle.seek(0, os.SEEK_END)
        if lock_handle.tell() > 1:
            lock_handle.truncate(1)
            lock_handle.flush()
        lock_handle.seek(0)


def _release_file_lock(lock_handle) -> None:
    lock_handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _ledger_write_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    process_lock = _get_process_lock(path)
    lock_path = _lock_path_for_ledger(path)
    lock_handle = None

    with process_lock:
        try:
            lock_handle = open(lock_path, "a+b")
            _acquire_file_lock(lock_handle)
            yield
        finally:
            if lock_handle is not None:
                _release_file_lock(lock_handle)
                lock_handle.close()


def _get_active_ledger_paths() -> list[Path]:
    import src.strategy.duo_trader as duo_trader

    registry_getter = getattr(duo_trader, "get_all_trader_ledgers", None)
    if callable(registry_getter):
        trader_paths = [Path(path) for _, path in registry_getter() if path is not None]
    else:
        trader_paths = [
            Path(path)
            for path in (
                getattr(duo_trader, "SINGLE_LEDGER", None),
                getattr(duo_trader, "CONVICTION_LEDGER", None),
            )
            if path is not None
        ]
    existing = [path for path in trader_paths if path.exists()]
    return existing if existing else [LEDGER_PATH]


def _redeem_state_path() -> Path:
    return LOGS_DIR / REDEEM_STATE_FILENAME


def _parse_iso_datetime(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _auto_redeem_cooldown_seconds() -> float:
    raw = str(os.getenv("POLYMARKET_AUTO_REDEEM_COOLDOWN_HOURS", "6") or "").strip()
    try:
        hours = float(raw)
    except ValueError:
        hours = 6.0
    return max(0.0, hours) * 3600.0


def _pending_redeem_stale_seconds() -> float:
    raw = str(os.getenv("POLYMARKET_AUTO_REDEEM_PENDING_TTL_HOURS", "24") or "").strip()
    try:
        hours = float(raw)
    except ValueError:
        hours = 24.0
    return max(0.0, hours) * 3600.0


def _empty_redeem_summary(
    *,
    wait: bool,
    reason: str = "",
    source: str = "manual",
) -> dict:
    return {
        "redeemable_positions": 0,
        "redeemable_conditions": 0,
        "submitted_conditions": 0,
        "submitted_positions": 0,
        "redeemed_conditions": 0,
        "redeemed_positions": 0,
        "results": [],
        "errors": [],
        "reason": reason,
        "waited_for_confirmation": bool(wait),
        "source": source,
    }


def _load_redeem_state(path: Path) -> dict:
    if not path.exists():
        return {"pending_transactions": []}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read redeem state from %s; resetting it", path)
        return {"pending_transactions": []}
    if not isinstance(payload, dict):
        return {"pending_transactions": []}
    pending = payload.get("pending_transactions")
    payload["pending_transactions"] = pending if isinstance(pending, list) else []
    return payload


def _save_redeem_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
        if os.name == "nt":
            deadline = time.monotonic() + 2.0
            while True:
                try:
                    os.replace(tmp_path, path)
                    break
                except PermissionError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
        else:
            os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _pending_redeem_record(result: dict, *, checked_at: Optional[datetime] = None) -> Optional[dict]:
    transaction_id = str(result.get("transaction_id") or "").strip()
    if not transaction_id:
        return None
    record = {
        "transaction_id": transaction_id,
        "transaction_hash": result.get("transaction_hash"),
        "condition_id": result.get("condition_id"),
        "title": result.get("title"),
        "position_count": int(result.get("position_count") or 0),
        "state": result.get("state"),
    }
    submitted_at = checked_at or datetime.now(timezone.utc)
    record["submitted_at"] = submitted_at.isoformat()
    return record


def _refresh_pending_redeem_transactions(
    state: dict,
    *,
    client,
    checked_at: Optional[datetime] = None,
) -> list[dict]:
    checked_at = checked_at or datetime.now(timezone.utc)
    active_transactions: list[dict] = []

    for raw_record in state.get("pending_transactions", []):
        if not isinstance(raw_record, dict):
            continue
        record = dict(raw_record)
        transaction_id = str(record.get("transaction_id") or "").strip()
        if not transaction_id:
            continue
        submitted_at = _parse_iso_datetime(record.get("submitted_at"))

        transaction = None
        try:
            get_tx = getattr(client, "get_relayer_transaction", None)
            if callable(get_tx):
                transaction = get_tx(transaction_id)
        except Exception as exc:
            logger.warning(
                "Could not refresh redeem transaction %s; keeping it pending: %s",
                transaction_id,
                exc,
            )

        if transaction:
            state_value = str(transaction.get("state", "") or "")
            if state_value:
                record["state"] = state_value
            transaction_hash = transaction.get("transactionHash")
            if transaction_hash:
                record["transaction_hash"] = transaction_hash
            record["last_checked_at"] = checked_at.isoformat()
            if state_value in _REDEEM_SUCCESS_STATES:
                logger.info(
                    "Redeem transaction %s confirmed in %s",
                    transaction_id,
                    state_value,
                )
                continue
            if state_value in _REDEEM_FAILURE_STATES:
                logger.warning(
                    "Redeem transaction %s ended in %s; allowing a new redeem attempt",
                    transaction_id,
                    state_value,
                )
                continue
        else:
            stale_after_seconds = _pending_redeem_stale_seconds()
            if (
                stale_after_seconds > 0
                and submitted_at is not None
                and (checked_at - submitted_at).total_seconds() >= stale_after_seconds
            ):
                logger.warning(
                    "Clearing stale redeem transaction %s after the relayer stopped returning it",
                    transaction_id,
                )
                continue
            record["last_checked_at"] = checked_at.isoformat()

        active_transactions.append(record)

    state["pending_transactions"] = active_transactions
    return active_transactions


_UNRESOLVED_PLACEMENT_STATES = {"pending_submit", "submitted", "unknown", "pending", "resting"}
_CONFIRMED_PLACEMENT_STATES = {"filled", "matched"}


def _open_bets_from(bets: list[dict] | tuple[dict, ...]) -> list[dict]:
    return [bet for bet in bets if bet.get("status") == "open"]


def _safe_positive_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed > 0.0 else 0.0


def _first_positive_float(*values: Any, default: float = 0.0) -> float:
    for value in values:
        parsed = _safe_positive_float(value)
        if parsed > 0.0:
            return parsed
    return float(default or 0.0)


def _actual_fill_amount(bet: dict) -> float:
    return _first_positive_float(
        bet.get("actual_fill_amount"),
        bet.get("fill_amount"),
    )


def _actual_filled_shares(bet: dict) -> float:
    return _first_positive_float(
        bet.get("actual_filled_shares"),
        bet.get("filled_shares"),
    )


def _has_actual_fill(bet: dict) -> bool:
    return _actual_fill_amount(bet) > 0.0 and _actual_filled_shares(bet) > 0.0


def _actual_fill_source(bet: dict) -> str:
    return str(bet.get("actual_fill_source") or "").strip().lower()


def _is_crypto_5m_bet(bet: dict) -> bool:
    if str(bet.get("order_type") or "") == _CRYPTO_5M_ORDER_TYPE:
        return True
    if str(bet.get("strategy") or "") == _CRYPTO_5M_STRATEGY_NAME:
        return True
    if str(bet.get("signal_source") or "") == _CRYPTO_5M_STRATEGY_NAME:
        return True
    slug = str(bet.get("market_slug") or "").strip().lower()
    return "-updown-5m-" in slug


def _has_polymarket_fill_evidence(bet: dict) -> bool:
    if not _has_actual_fill(bet):
        return False
    return _actual_fill_source(bet) in {"polymarket_activity", "clob_trade_history"}


def _placement_state(bet: dict) -> str:
    return str(bet.get("placement_state") or "").strip().lower()


def _has_confirmed_position(bet: dict) -> bool:
    if bet.get("dry_run"):
        return True
    if _is_crypto_5m_bet(bet):
        return _has_polymarket_fill_evidence(bet)
    if _has_actual_fill(bet):
        return True
    state = _placement_state(bet)
    if state in _CONFIRMED_PLACEMENT_STATES:
        return True
    if state in _UNRESOLVED_PLACEMENT_STATES:
        return False
    return True


def _settlement_amount(bet: dict) -> float:
    return _first_positive_float(
        bet.get("actual_fill_amount"),
        bet.get("fill_amount"),
        bet.get("amount"),
    )


def _settlement_shares(bet: dict) -> float:
    return _first_positive_float(
        bet.get("actual_filled_shares"),
        bet.get("filled_shares"),
        bet.get("shares"),
    )


def _funded_open_bets(bets: list[dict] | tuple[dict, ...]) -> list[dict]:
    """Open bets with confirmed fills — excludes unresolved submissions."""
    return [
        bet for bet in bets
        if bet.get("status") == "open"
        and _has_confirmed_position(bet)
    ]


def _settled_bets_from(bets: list[dict] | tuple[dict, ...]) -> list[dict]:
    return [bet for bet in bets if bet.get("status") in ("won", "lost")]


def _build_summary_from_bets(bets: list[dict] | tuple[dict, ...]) -> dict:
    settled = _settled_bets_from(bets)
    open_bets = _open_bets_from(bets)
    funded = _funded_open_bets(bets)

    total_wagered = sum(_settlement_amount(b) for b in settled)
    realized_pnl = sum(b["result_pnl"] for b in settled if b["result_pnl"] is not None)
    wins = sum(1 for b in settled if b["status"] == "won")

    # Only count funded (confirmed-fill) bets in exposure/unrealized P&L
    unrealized_pnl = 0.0
    open_invested = 0.0
    for bet in funded:
        invested = _settlement_amount(bet)
        shares = _settlement_shares(bet)
        open_invested += invested
        if bet["cur_price"] is not None:
            cur_value = shares * bet["cur_price"]
            unrealized_pnl += cur_value - invested

    valid_clv = [
        float(bet["clv"])
        for bet in bets
        if bet.get("clv") not in (None, "")
    ]

    return {
        "total_bets": len(bets),
        "open_bets": len(open_bets),
        "settled_bets": len(settled),
        "wins": wins,
        "losses": len(settled) - wins,
        "win_rate": wins / len(settled) if settled else 0.0,
        "total_wagered": total_wagered,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": realized_pnl + unrealized_pnl,
        "roi": realized_pnl / total_wagered if total_wagered > 0 else 0.0,
        "open_invested": open_invested,
        "avg_clv": sum(valid_clv) / len(valid_clv) if valid_clv else None,
        "clv_sample_size": len(valid_clv),
        "pct_positive_clv": (
            sum(1 for clv in valid_clv if clv > 0) / len(valid_clv)
            if valid_clv else None
        ),
    }


def _parse_event_timestamp(value: object) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _maybe_capture_clv_snapshot(bet: dict, cur_price: float) -> None:
    event_ts = _parse_event_timestamp(bet.get("event_date"))
    if event_ts is None:
        return

    now = datetime.now(timezone.utc)
    window_start = event_ts - _CLV_CAPTURE_WINDOW
    if now < window_start or now >= event_ts:
        return

    try:
        entry_prob = float(bet.get("market_prob"))
    except (TypeError, ValueError):
        return
    if entry_prob <= 0:
        return

    closing_prob = round(float(cur_price), 4)
    bet["closing_prob"] = closing_prob
    bet["clv"] = round(calculate_closing_line_value(entry_prob, closing_prob), 4)
    bet["clv_captured_at"] = now.isoformat()


@dataclass(frozen=True)
class LedgerMutationResult:
    status: str
    bet: Optional[dict] = None

    @property
    def ok(self) -> bool:
        return self.status == "updated"


@dataclass(frozen=True)
class ReadOnlyBetLedgerView:
    _bets: tuple[dict, ...]

    @classmethod
    def from_bets(cls, bets: list[dict]) -> "ReadOnlyBetLedgerView":
        return cls(tuple(dict(bet) for bet in bets))

    def get_bets(self, *, fresh: bool = False) -> list[dict]:
        return [dict(bet) for bet in self._bets]

    @property
    def bets(self) -> list[dict]:
        return self.get_bets()

    def get_open_bets(self, *, fresh: bool = False) -> list[dict]:
        return [dict(bet) for bet in _open_bets_from(self._bets)]

    @property
    def open_bets(self) -> list[dict]:
        return self.get_open_bets()

    def get_settled_bets(self, *, fresh: bool = False) -> list[dict]:
        return [dict(bet) for bet in _settled_bets_from(self._bets)]

    @property
    def settled_bets(self) -> list[dict]:
        return self.get_settled_bets()

    def get_summary(self, *, fresh: bool = False) -> dict:
        return _build_summary_from_bets(self._bets)


class BetLedger:
    """
    Persistent bet ledger — survives restarts.

    Stores every bet with status tracking:
    - open: bet placed, waiting for event
    - won: event settled, fighter won
    - lost: event settled, fighter lost
    - cancelled: order was cancelled or never filled
    """

    def __init__(self, path: Path = LEDGER_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._read_only = False
        self.bets: list[dict] = []
        self._load()

    def _read_bets_from_disk(self) -> list[dict]:
        """Read bets directly from disk without taking a write lock."""
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    data = json.load(f)
                bets = data.get("bets", [])
                logger.debug("Loaded %s bets from ledger %s", len(bets), self.path)
                return bets
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupt ledger file, starting fresh")
        return []

    def _load(self):
        """Load ledger from disk."""
        self.bets = self._read_bets_from_disk()

    def _snapshot_bets_locked(self) -> list[dict]:
        return [dict(bet) for bet in self.bets]

    def _save(self):
        """Persist ledger to disk atomically (write tmp → rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "bets": self.bets,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.path.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            if os.name == "nt":
                deadline = time.monotonic() + 2.0
                while True:
                    try:
                        os.replace(tmp_path, self.path)
                        break
                    except PermissionError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.05)
            else:
                os.replace(tmp_path, self.path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _mutate_locked(
        self,
        mutator: Callable[[list[dict]], tuple[Any, bool]],
    ) -> Any:
        if self._read_only:
            raise RuntimeError("Cannot mutate a read-only ledger view")
        with self._lock:
            with _ledger_write_lock(self.path):
                self.bets = self._read_bets_from_disk()
                result, changed = mutator(self.bets)
                if changed:
                    self._save()
                return result

    def refresh(self) -> list[dict]:
        with self._lock:
            self._load()
            return self._snapshot_bets_locked()

    def get_bets(self, *, fresh: bool = False) -> list[dict]:
        with self._lock:
            if fresh:
                self._load()
            return self._snapshot_bets_locked()

    def add_bet(
        self,
        fighter: str,
        opponent: str,
        side: str,
        amount: float,
        price: float,
        shares: float,
        token_id: str,
        market_id: str,
        model_prob: float,
        market_prob: float,
        edge: float,
        decimal_odds: float,
        dry_run: bool = True,
        event_date: Optional[str] = None,
        market_event_date: Optional[str] = None,
        order_type: Optional[str] = None,
        order_id: Optional[str] = None,
        status: str = "open",
        placement_state: Optional[str] = None,
        submission_error: Optional[str] = None,
        condition_id: Optional[str] = None,
        reason: str = "",
        metadata: Optional[dict] = None,
    ) -> dict:
        """Record a new bet in the ledger."""
        def _add(bets: list[dict]) -> tuple[dict, bool]:
            next_id = max((b.get("id", 0) for b in bets), default=0) + 1
            bet = {
                "id": next_id,
                "fighter": fighter,
                "opponent": opponent,
                "side": side,
                "amount": round(amount, 2),
                "price": round(price, 4),
                "shares": round(shares, 2),
                "token_id": token_id,
                "market_id": market_id,
                "condition_id": condition_id or "",
                "model_prob": round(model_prob, 4),
                "market_prob": round(market_prob, 4),
                "edge": round(edge, 4),
                "decimal_odds": round(decimal_odds, 4),
                "dry_run": dry_run,
                "status": status,
                "placed_at": datetime.now(timezone.utc).isoformat(),
                "event_date": event_date,
                "market_event_date": market_event_date,
                "settled_at": None,
                "result_pnl": None,
                "cur_price": None,
                "closing_prob": None,
                "clv": None,
                "clv_captured_at": None,
                "order_type": order_type,
                "order_id": order_id,
                "placement_state": (
                    placement_state
                    if placement_state is not None
                    else ("dry_run" if dry_run else "submitted")
                ),
                "submission_error": submission_error,
                "cancel_reason": None,
                "reason": reason,
            }
            if metadata:
                bet.update(dict(metadata))
            bets.append(bet)
            return bet, True

        bet = self._mutate_locked(_add)
        logger.info(
            f"Ledger: recorded bet #{bet['id']} — "
            f"${amount:.2f} on {fighter} @ {price:.4f}"
        )
        return bet

    def update_bet_fields(
        self,
        bet_id: int,
        *,
        require_open: bool = True,
        **updates,
    ) -> LedgerMutationResult:
        """Patch fields on a single ledger row."""
        if not updates:
            return LedgerMutationResult(status="not_found")

        def _update(bets: list[dict]) -> tuple[LedgerMutationResult, bool]:
            for bet in bets:
                if bet["id"] != bet_id:
                    continue
                if require_open and bet.get("status") != "open":
                    return LedgerMutationResult(status="not_open", bet=dict(bet)), False
                bet.update(updates)
                return LedgerMutationResult(status="updated", bet=dict(bet)), True
            return LedgerMutationResult(status="not_found"), False

        result = self._mutate_locked(_update)
        if result.ok:
            logger.info("Ledger: updated bet #%s fields %s", bet_id, sorted(updates))
        return result

    def settle_bet(self, bet_id: int, won: bool) -> LedgerMutationResult:
        """Mark a bet as won or lost."""
        def _settle(bets: list[dict]) -> tuple[LedgerMutationResult, bool]:
            for bet in bets:
                if bet["id"] != bet_id:
                    continue
                if bet.get("status") != "open":
                    return LedgerMutationResult(status="not_open", bet=dict(bet)), False
                amount = _settlement_amount(bet)
                shares = _settlement_shares(bet)
                bet["status"] = "won" if won else "lost"
                bet["settled_at"] = datetime.now(timezone.utc).isoformat()
                if won:
                    bet["result_pnl"] = round(
                        shares * 1.0 - amount, 2
                    )  # shares pay out $1 each on win
                else:
                    bet["result_pnl"] = -round(amount, 2)
                return LedgerMutationResult(status="updated", bet=dict(bet)), True
            return LedgerMutationResult(status="not_found"), False
        result = self._mutate_locked(_settle)
        if not result.ok:
            return result

        logger.info(
            f"Ledger: settled bet #{bet_id} - "
            f"{'WON' if won else 'LOST'} "
            f"P&L: ${result.bet['result_pnl']:+.2f}"
        )
        return result

    def cancel_bet(
        self,
        bet_id: int,
        reason: Optional[str] = None,
        *,
        expected_order_types: Optional[set[str]] = None,
    ) -> LedgerMutationResult:
        """Mark a bet as cancelled."""
        def _cancel(bets: list[dict]) -> tuple[LedgerMutationResult, bool]:
            for bet in bets:
                if bet["id"] != bet_id:
                    continue
                if bet.get("status") != "open":
                    return LedgerMutationResult(status="not_open", bet=dict(bet)), False
                if expected_order_types is not None and bet.get("order_type") not in expected_order_types:
                    return LedgerMutationResult(
                        status="invalid_order_type",
                        bet=dict(bet),
                    ), False
                bet["status"] = "cancelled"
                bet["settled_at"] = datetime.now(timezone.utc).isoformat()
                bet["result_pnl"] = 0.0
                if reason:
                    bet["cancel_reason"] = reason
                return LedgerMutationResult(status="updated", bet=dict(bet)), True
            return LedgerMutationResult(status="not_found"), False

        result = self._mutate_locked(_cancel)
        if result.ok:
            logger.info("Ledger: cancelled bet #%s", bet_id)
        return result

    def convert_limit_bet_to_position(
        self,
        bet_id: int,
        filled_shares: float,
        *,
        cancel_reason: Optional[str] = None,
    ) -> LedgerMutationResult:
        """Preserve matched shares after a resting limit order stops resting."""
        def _convert(bets: list[dict]) -> tuple[LedgerMutationResult, bool]:
            for bet in bets:
                if bet["id"] != bet_id:
                    continue
                if bet.get("status") != "open":
                    return LedgerMutationResult(status="not_open", bet=dict(bet)), False
                if bet.get("order_type") not in _LIMIT_ORDER_TYPES:
                    return LedgerMutationResult(
                        status="invalid_order_type",
                        bet=dict(bet),
                    ), False

                price = round(float(bet.get("price", 0.0) or 0.0), 4)
                shares = round(max(float(filled_shares or 0.0), 0.0), 2)
                amount = round(shares * price, 2)

                bet["status"] = "open"
                bet["amount"] = amount
                bet["shares"] = shares
                bet["settled_at"] = None
                bet["result_pnl"] = None
                bet["order_id"] = None
                bet["order_type"] = "filled_limit"
                if cancel_reason:
                    bet["cancel_reason"] = cancel_reason
                return LedgerMutationResult(status="updated", bet=dict(bet)), True
            return LedgerMutationResult(status="not_found"), False

        result = self._mutate_locked(_convert)
        if result.ok:
            logger.info("Ledger: converted bet #%s from resting limit to position", bet_id)
        return result

    def update_current_price(self, bet_id: int, cur_price: float) -> LedgerMutationResult:
        """Update the current market price for an open bet."""
        def _update(bets: list[dict]) -> tuple[LedgerMutationResult, bool]:
            for bet in bets:
                if bet["id"] == bet_id:
                    if bet.get("status") != "open":
                        return LedgerMutationResult(status="not_open", bet=dict(bet)), False
                    bet["cur_price"] = round(cur_price, 4)
                    _maybe_capture_clv_snapshot(bet, cur_price)
                    return LedgerMutationResult(status="updated", bet=dict(bet)), True
            return LedgerMutationResult(status="not_found"), False

        return self._mutate_locked(_update)

    def get_open_bets(self, *, fresh: bool = False) -> list[dict]:
        return _open_bets_from(self.get_bets(fresh=fresh))

    @property
    def open_bets(self) -> list[dict]:
        return self.get_open_bets()

    def get_settled_bets(self, *, fresh: bool = False) -> list[dict]:
        return _settled_bets_from(self.get_bets(fresh=fresh))

    @property
    def settled_bets(self) -> list[dict]:
        return self.get_settled_bets()

    def get_summary(self, *, fresh: bool = False) -> dict:
        """Compute aggregate P&L stats."""
        return _build_summary_from_bets(self.get_bets(fresh=fresh))


def load_all_trader_ledgers() -> ReadOnlyBetLedgerView:
    """Return a merged read-only ledger view spanning all duo-trader ledgers.

    Falls back to the default ledger if the duo-trader ledger files don't exist.
    """
    merged_bets: list[dict] = []
    seen_order_ids: set[str] = set()

    for path in _get_active_ledger_paths():
        trader = BetLedger(path=path)
        for bet in trader.bets:
            # Deduplicate across ledgers by order_id
            oid = str(bet.get("order_id") or "")
            if oid and oid in seen_order_ids:
                continue
            if oid:
                seen_order_ids.add(oid)
            merged_bets.append({
                **bet,
                "_original_id": bet["id"],
                "_ledger_path": str(path),
            })

    # Sort by placed_at for consistent ordering
    merged_bets.sort(key=lambda b: b.get("placed_at", ""))
    # Re-assign sequential IDs so settle-by-id still works
    for i, bet in enumerate(merged_bets, 1):
        bet["id"] = i

    return ReadOnlyBetLedgerView.from_bets(merged_bets)


def resolve_merged_bet_reference(
    bet_id: int,
    *,
    require_open: bool = True,
) -> Optional[dict]:
    """Resolve a displayed merged bet ID back to its ledger path and raw ID."""
    merged = load_all_trader_ledgers()
    target = next(
        (
            bet for bet in merged.bets
            if bet["id"] == bet_id and (not require_open or bet.get("status") == "open")
        ),
        None,
    )
    if target is None:
        return None

    return {
        "display_id": bet_id,
        "ledger_path": Path(target.get("_ledger_path", LEDGER_PATH)),
        "original_id": int(target.get("_original_id", bet_id)),
        "placed_at": target.get("placed_at"),
        "bet": target,
    }


_pnl_log_lock = threading.Lock()


def _log_pnl_snapshot(summary: dict) -> None:
    """Append a P&L snapshot to history for trend tracking."""
    record = {
        **summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with _pnl_log_lock:
        with open(PNL_HISTORY_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")


def _load_pnl_history() -> list[dict]:
    """Load P&L history for trend display."""
    history = []
    if PNL_HISTORY_PATH.exists():
        with open(PNL_HISTORY_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return history


def _clear_screen():
    """Clear terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def _format_pnl(value: float) -> str:
    """Format P&L with color indicators."""
    if value > 0:
        return f"+${value:.2f}"
    elif value < 0:
        return f"-${abs(value):.2f}"
    return "$0.00"


def run_live_dashboard(
    clob_client=None,
    refresh_seconds: int = 30,
    include_dry_runs: bool = True,
) -> None:
    """
    Run a live-updating terminal dashboard showing positions and P&L.

    Args:
        clob_client: Authenticated CLOB client (for fetching live prices)
        refresh_seconds: How often to refresh (default 30s)
        include_dry_runs: Whether to show dry-run bets
    """
    from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER

    # Load real trader ledgers for price updates, merged view for display
    def _load_trader_ledgers():
        ledgers = []
        for path in [SINGLE_LEDGER, CONVICTION_LEDGER]:
            if Path(path).exists():
                ledgers.append(BetLedger(path=path))
        return ledgers if ledgers else [BetLedger()]

    print(f"Starting live dashboard (refresh every {refresh_seconds}s)")
    print("Press Ctrl+C to stop\n")

    while True:
        try:
            trader_ledgers = _load_trader_ledgers()

            # Update current prices on real trader ledgers
            if clob_client:
                for ledger in trader_ledgers:
                    for bet in ledger.get_open_bets(fresh=True):
                        if bet.get("token_id"):
                            try:
                                price_data = clob_client.get_price(bet["token_id"])
                                mid_price = price_data.get("mid")
                                if mid_price is None:
                                    logger.warning(
                                        "Skipping live tracker price refresh for bet #%s (%s) because the orderbook is incomplete",
                                        bet["id"],
                                        bet["token_id"],
                                    )
                                    continue
                                result = ledger.update_current_price(
                                    bet["id"], mid_price
                                )
                                if not result.ok and result.status != "not_found":
                                    logger.info(
                                        "Skipped price refresh for bet #%s because it is no longer open",
                                        bet["id"],
                                    )
                            except Exception as e:
                                logger.warning(
                                    "Live tracker price refresh failed for bet #%s (%s): %s",
                                    bet.get("id"),
                                    bet.get("token_id"),
                                    e,
                                )

            # Use merged view for display
            ledger = load_all_trader_ledgers()
            summary = ledger.get_summary()
            _log_pnl_snapshot(summary)

            _clear_screen()
            _print_dashboard(ledger, summary, include_dry_runs)

            time.sleep(refresh_seconds)

        except KeyboardInterrupt:
            print("\nDashboard stopped.")
            break
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            time.sleep(5)


def _print_dashboard(
    ledger: BetLedger,
    summary: dict,
    include_dry_runs: bool = True,
) -> None:
    """Print the dashboard to terminal."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print(f"{'=' * 70}")
    print(f"  UFC BETTING BOT — LIVE TRACKER        {now}")
    print(f"{'=' * 70}")

    # --- P&L Summary ---
    print(f"\n  P&L SUMMARY")
    print(f"  {'-' * 40}")
    print(f"  Realized P&L:    {_format_pnl(summary['realized_pnl']):>12}")
    print(f"  Unrealized P&L:  {_format_pnl(summary['unrealized_pnl']):>12}")
    print(f"  Total P&L:       {_format_pnl(summary['total_pnl']):>12}")
    if summary['total_wagered'] > 0:
        print(f"  ROI:             {summary['roi']:>11.1%}")
    print(f"  Total wagered:   {'${:.2f}'.format(summary['total_wagered']):>12}")
    print(f"  Win rate:        {summary['win_rate']:>11.1%}  "
          f"({summary['wins']}W / {summary['losses']}L)")

    # --- Open Positions ---
    open_bets = ledger.open_bets
    if not include_dry_runs:
        open_bets = [b for b in open_bets if not b.get("dry_run")]

    if open_bets:
        print(f"\n  OPEN POSITIONS ({len(open_bets)})")
        print(f"  {'-' * 66}")
        print(
            f"  {'#':>3}  {'Fighter':<22} {'Amt':>7} {'Entry':>6} "
            f"{'Cur':>6} {'P&L':>8} {'Edge':>6}"
        )
        print(f"  {'-' * 66}")

        for bet in open_bets:
            cur = bet.get("cur_price")
            if cur is not None:
                cur_value = bet["shares"] * cur
                pnl = cur_value - bet["amount"]
                pnl_str = _format_pnl(pnl)
                cur_str = f"${cur:.2f}"
            else:
                pnl_str = "   --"
                cur_str = "  --"

            dry = " (dry)" if bet.get("dry_run") else ""
            name = bet["fighter"][:20] + dry
            print(
                f"  {bet['id']:>3}  {name:<22} "
                f"${bet['amount']:>6.2f} ${bet['price']:.2f} "
                f"{cur_str:>6} {pnl_str:>8} "
                f"{bet['edge']:>+5.1%}"
            )

        total_open = sum(b["amount"] for b in open_bets)
        print(f"  {'-' * 66}")
        print(f"  {'Total open':>27} ${total_open:>6.2f}")

    # --- Recent Settled Bets ---
    settled = ledger.settled_bets
    if not include_dry_runs:
        settled = [b for b in settled if not b.get("dry_run")]

    recent_settled = settled[-10:]  # Last 10
    if recent_settled:
        print(f"\n  RECENT RESULTS ({len(settled)} total)")
        print(f"  {'-' * 56}")
        print(
            f"  {'#':>3}  {'Fighter':<22} {'Amt':>7} "
            f"{'Result':>6} {'P&L':>8}"
        )
        print(f"  {'-' * 56}")

        for bet in recent_settled:
            result = bet["status"].upper()
            pnl_str = _format_pnl(bet["result_pnl"]) if bet["result_pnl"] is not None else "--"
            print(
                f"  {bet['id']:>3}  {bet['fighter']:<22} "
                f"${bet['amount']:>6.2f} "
                f"{result:>6} {pnl_str:>8}"
            )

    # --- P&L Trend (mini sparkline from history) ---
    history = _load_pnl_history()
    if len(history) >= 2:
        recent = history[-20:]
        pnl_values = [h.get("total_pnl", 0) for h in recent]
        _print_pnl_trend(pnl_values)

    print(f"\n{'=' * 70}")


def _print_pnl_trend(values: list[float]) -> None:
    """Print a simple ASCII P&L trend chart."""
    if not values:
        return

    print(f"\n  P&L TREND (last {len(values)} snapshots)")
    print(f"  {'-' * 40}")

    min_val = min(values)
    max_val = max(values)
    spread = max_val - min_val

    if spread == 0:
        # Flat line
        for v in values:
            print(f"  ${v:>+8.2f} |{'=' * 20}|")
        return

    chart_width = 30
    for v in values:
        # Normalize to 0-chart_width
        pos = int((v - min_val) / spread * chart_width)
        zero_pos = int((0 - min_val) / spread * chart_width) if min_val < 0 else 0

        bar = list(" " * (chart_width + 1))
        if zero_pos <= chart_width:
            bar[zero_pos] = "|"

        if v >= 0:
            for i in range(zero_pos, pos + 1):
                if i <= chart_width:
                    bar[i] = "#"
        else:
            for i in range(pos, zero_pos + 1):
                if i <= chart_width:
                    bar[i] = "-"

        print(f"  ${v:>+8.2f} {''.join(bar)}")


def auto_settle_from_polymarket(ledger: BetLedger, clob_client=None) -> int:
    """
    Check Polymarket for settled markets and auto-settle matching bets.

    Returns number of bets settled.
    """
    from src.polymarket.client import GammaClient

    gamma = GammaClient()
    settled_count = 0

    for bet in ledger.get_open_bets(fresh=True):
        if not _has_confirmed_position(bet):
            logger.info(
                "Skipped auto-settle for bet #%s because placement_state=%s has no confirmed fill",
                bet.get("id"),
                bet.get("placement_state"),
            )
            continue

        market_identifier = bet.get("condition_id") or bet.get("market_id")
        if not market_identifier:
            continue

        try:
            market = gamma.get_market(str(market_identifier))
            if not market:
                continue

            tokens = market.get("tokens", [])
            winning_token = ""
            if isinstance(tokens, list):
                for token in tokens:
                    if not isinstance(token, dict):
                        continue
                    if bool(token.get("winner")):
                        winning_token = str(
                            token.get("token_id")
                            or token.get("tokenId")
                            or ""
                        ).strip()
                        if winning_token:
                            break

            raw_resolved = market.get("resolved", False)
            resolved = raw_resolved if isinstance(raw_resolved, bool) else str(
                raw_resolved
            ).strip().lower() in {"1", "true", "yes"}
            resolved = resolved or bool(winning_token)
            if not resolved:
                continue

            if not winning_token:
                winning_token = str(market.get("winning_token_id", "") or "").strip()
            if not winning_token:
                continue

            won = winning_token == bet["token_id"]

            result = ledger.settle_bet(bet["id"], won)
            if result.ok:
                settled_count += 1
                logger.info(
                    f"Auto-settled: {bet['fighter']} "
                    f"{'WON' if won else 'LOST'}"
                )
            elif result.status == "not_open":
                logger.info(
                    "Skipped auto-settle for bet #%s because it is no longer open",
                    bet["id"],
                )
        except Exception as e:
            logger.warning(
                f"Could not check settlement for bet #{bet['id']}: {e}"
            )

    return settled_count


def auto_reconcile_sold_positions(
    ledger: BetLedger,
    live_token_ids: set[str],
) -> int:
    """
    Cancel ledger bets whose positions no longer exist on Polymarket.

    When a user sells/closes a position on Polymarket before the market
    resolves, the ledger still shows status="open".  This function detects
    those orphaned entries by comparing against *live_token_ids* (token IDs
    with size > 0 from the Polymarket Data API) and cancels them.

    Returns number of bets cancelled.
    """
    cancelled_count = 0
    for bet in ledger.get_open_bets(fresh=True):
        tid = bet.get("token_id")
        if not tid:
            continue
        # Skip unresolved limit orders — they may not have a position yet
        if bet.get("placement_state") in _UNRESOLVED_PLACEMENT_STATES:
            continue
        # Skip if Polymarket still shows a live position for this token
        if tid in live_token_ids:
            continue
        # This bet is "open" locally but has no position on Polymarket
        result = ledger.cancel_bet(bet["id"], reason="position_sold")
        if result.ok:
            cancelled_count += 1
            logger.info(
                "Reconciled sold position: %s (bet #%s)",
                bet.get("fighter", "?"),
                bet["id"],
            )
    return cancelled_count


def auto_redeem_positions_from_polymarket(
    clob_client=None,
    *,
    wait: bool = False,
    source: str = "manual",
) -> dict:
    """
    Redeem any resolved wallet positions that Polymarket marks as redeemable.

    Returns a summary dict so callers can decide whether to log, ignore,
    or surface configuration gaps without crashing the main loop.
    """
    from src.polymarket.client import ClobClientWrapper

    if source not in {"manual", "auto"}:
        raise ValueError("source must be 'manual' or 'auto'")

    client = clob_client or ClobClientWrapper()
    state_path = _redeem_state_path()
    checked_at = datetime.now(timezone.utc)

    with _ledger_write_lock(state_path):
        state = _load_redeem_state(state_path)
        pending_transactions = _refresh_pending_redeem_transactions(
            state,
            client=client,
            checked_at=checked_at,
        )
        if pending_transactions:
            state["last_checked_at"] = checked_at.isoformat()
            _save_redeem_state(state_path, state)
            summary = _empty_redeem_summary(
                wait=wait,
                reason="redeem_submission_pending",
                source=source,
            )
            summary["pending_transactions"] = pending_transactions
            return summary

        if source == "auto":
            cooldown_seconds = _auto_redeem_cooldown_seconds()
            last_auto_attempt_at = _parse_iso_datetime(state.get("last_auto_attempt_at"))
            if (
                cooldown_seconds > 0
                and last_auto_attempt_at is not None
                and (checked_at - last_auto_attempt_at).total_seconds() < cooldown_seconds
            ):
                next_attempt_at = last_auto_attempt_at.timestamp() + cooldown_seconds
                state["last_checked_at"] = checked_at.isoformat()
                _save_redeem_state(state_path, state)
                summary = _empty_redeem_summary(
                    wait=wait,
                    reason="auto_redeem_cooldown",
                    source=source,
                )
                summary["cooldown_remaining_seconds"] = max(
                    0,
                    int(next_attempt_at - checked_at.timestamp()),
                )
                summary["next_attempt_at"] = datetime.fromtimestamp(
                    next_attempt_at,
                    tz=timezone.utc,
                ).isoformat()
                return summary

        summary = client.redeem_redeemable_positions(wait=wait)
        summary["source"] = source

        pending_after_submit = [
            record
            for record in (
                _pending_redeem_record(result, checked_at=checked_at)
                for result in summary.get("results", [])
            )
            if record is not None
            and str(record.get("state") or "") not in (_REDEEM_SUCCESS_STATES | _REDEEM_FAILURE_STATES)
        ]

        state["pending_transactions"] = pending_after_submit
        state["last_checked_at"] = checked_at.isoformat()
        if source == "auto" and summary.get("reason") != "redeemer_not_configured":
            state["last_auto_attempt_at"] = checked_at.isoformat()
        _save_redeem_state(state_path, state)

    if summary.get("reason") == "redeemer_not_configured":
        logger.info(
            "Redeemable Polymarket positions detected but auto-redeem is not configured"
        )
    return summary
