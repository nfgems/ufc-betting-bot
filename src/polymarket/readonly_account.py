"""Strict account inspection for the orders-disabled refresh worker.

This adapter may derive an existing API key with the normal authenticated GET.
It never creates credentials, signs/submits an order, cancels or redeems. The SDK
itself is wrapped with an explicit read allowlist as a second boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path

from src.polymarket.client import (
    ClobClientWrapper, POLYMARKET_DATA_API_URL, POLYMARKET_GAMMA_URL,
    _clob_transport_lock,
)
from src.polymarket.data_api import request_json

logger = logging.getLogger(__name__)


class AccountReadUnavailable(RuntimeError):
    """A complete authenticated account observation could not be established."""


class AccountWritesDisabled(RuntimeError):
    pass


class _ReadOnlySdk:
    _reads = frozenset({
        "get_order_book", "get_clob_market_info", "get_open_orders", "get_order",
        "get_trades", "get_balance_allowance", "get_version", "get_neg_risk",
    })

    def __init__(self, sdk):
        self.__sdk = sdk

    def __getattr__(self, name):
        if name not in self._reads:
            raise AccountWritesDisabled("Orders-disabled account SDK permits explicit reads only")
        return getattr(self.__sdk, name)


class ReadOnlyClobClient(ClobClientWrapper):
    """Existing bounded/thread-safe CLOB reads with all writes unavailable."""

    def __getattribute__(self, name):
        if name.startswith(("create_", "cancel_", "redeem_", "submit_", "post_", "update_", "delete_")):
            raise AccountWritesDisabled("Orders-disabled account client cannot perform writes")
        return super().__getattribute__(name)

    def _relayer_request(self, *args, **kwargs):
        raise AccountWritesDisabled("Orders-disabled account client cannot access the relayer")

    def can_redeem_positions(self):
        return False

    def _discover_proxy_address(self):
        # The inherited implementation logs full wallet addresses. Keep account
        # routing internal and fail explicitly if there is no observed funder.
        if self.funder_address:
            self._proxy_address = self.funder_address
            return self.funder_address
        if self._proxy_address:
            return self._proxy_address
        from eth_account import Account
        import requests
        signer_address = Account.from_key(self.private_key).address
        response = requests.get(f"{POLYMARKET_GAMMA_URL}/public-profile",
                                params={"address": signer_address}, timeout=10)
        response.raise_for_status()
        value = response.json().get("proxyWallet")
        if not isinstance(value, str) or not value.strip():
            raise AccountReadUnavailable("Account funder could not be verified")
        self._proxy_address = value
        return value

    def _derive_or_create_api_key(self, client):
        # Despite the inherited method name, creation is deliberately absent.
        with _clob_transport_lock:
            return client.derive_api_key()

    def _ensure_client_locked(self):
        if not self.private_key:
            raise AccountReadUnavailable("Account signing credential is unavailable")
        from py_clob_client_v2.client import ClobClient
        self._configure_shared_transport()
        funder = self._discover_proxy_address()
        client = ClobClient(self.host, chain_id=self.chain_id, key=self.private_key,
                            signature_type=self.SIGNATURE_TYPE_GNOSIS_SAFE,
                            funder=funder, retry_on_error=False)
        credentials = self._derive_or_create_api_key(client)
        client.set_api_creds(credentials)
        self._api_creds = credentials
        self._client = _ReadOnlySdk(client)
        logger.info("Orders-disabled authenticated account client initialized")

    def get_positions(self, *, limit=500):
        """Paginate strictly; a missing, malformed or truncated page is unknown."""
        self._ensure_client()
        page_size = max(1, min(int(limit), 500))
        retained, seen = [], set()
        for page in range(10):
            rows = request_json(f"{POLYMARKET_DATA_API_URL}/positions", params={
                "user": self.proxy_address, "limit": page_size,
                "offset": page * page_size, "sizeThreshold": 0,
            }, timeout=15, attempts=1)
            if not isinstance(rows, list) or len(rows) > page_size:
                raise AccountReadUnavailable("Account positions page was not a bounded list")
            for row in rows:
                if not isinstance(row, dict):
                    raise AccountReadUnavailable("Account position was not a mapping")
                size = _nonnegative(row.get("size"), "position size")
                if size == 0:
                    continue
                asset = row.get("asset")
                if not isinstance(asset, str) or not asset or asset in seen:
                    raise AccountReadUnavailable("Account position identity is missing or duplicated")
                seen.add(asset)
                _nonnegative(row.get("currentValue"), "position value")
                retained.append(dict(row))
            if len(rows) < page_size:
                return retained
        raise AccountReadUnavailable("Account positions exceeded the bounded complete-read limit")


def _nonnegative(value, label):
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise AccountReadUnavailable(f"Observed {label} is unavailable") from exc
    if not math.isfinite(number) or number < 0:
        raise AccountReadUnavailable(f"Observed {label} is outside its valid range")
    return number


def _resolved_zero_position(row):
    # A zero quote on an unresolved market is not evidence of zero risk.
    if row.get("redeemable") is not True:
        return False
    try:
        return (_nonnegative(row.get("currentValue"), "position value") == 0
                and _nonnegative(row.get("curPrice"), "resolved position price") == 0)
    except AccountReadUnavailable:
        return False


@dataclass(frozen=True)
class AccountObservation:
    cash: float
    portfolio_value: float
    reserved_cash: float
    positions: tuple[dict, ...]
    open_orders: tuple[dict, ...]
    summary: dict

    @property
    def total_equity(self):
        return self.cash + self.portfolio_value

    @property
    def available_cash(self):
        return max(0.0, self.cash - self.reserved_cash)


def inspect_account(client):
    """Return a strict complete snapshot; unavailable never means an empty wallet."""
    from src.strategy.duo_trader import _open_buy_order_reserved_cash
    cash = client.get_cash_balance_details(allow_onchain_fallback=False)
    if not isinstance(cash, dict) or cash.get("source") != "clob":
        raise AccountReadUnavailable("Authenticated CLOB cash could not be confirmed")
    amount = _nonnegative(cash.get("balance"), "cash")
    positions = client.get_positions()
    orders = client.get_open_orders()
    if not isinstance(positions, list) or not isinstance(orders, list):
        raise AccountReadUnavailable("Complete position/order lists were not confirmed")
    value = sum(_nonnegative(row.get("currentValue"), "position value") for row in positions)
    try:
        reserve = sum(_open_buy_order_reserved_cash(row) for row in orders)
    except (TypeError, ValueError, KeyError) as exc:
        raise AccountReadUnavailable("Open-order cash reservations could not be confirmed") from exc
    summary = {
        "connected": True, "cash_confirmed": True, "positions_complete": True,
        "open_orders_complete": True, "position_count": len(positions),
        "open_order_count": len(orders), "open_buy_order_count": sum(
            str(row.get("side", "")).upper() == "BUY" for row in orders),
        "portfolio_value_usd": value, "open_buy_reserved_usd": reserve,
        "sizing_cash_usd": max(0.0, amount - reserve), "sizing_equity_usd": amount + value,
        "empty_account_proved": False, "orders_disabled": True,
    }
    return AccountObservation(amount, value, reserve, tuple(positions), tuple(orders), summary)


_LEDGERS = (
    ("legacy", "bet_ledger.json"), ("S", "bet_ledger_single.json"),
    ("C", "bet_ledger_conviction.json"), ("M", "bet_ledger_model_tracker.json"),
    ("G", "bet_ledger_gemini_tracker.json"), ("BTC", "bet_ledger_btc5m.json"),
    ("B", "bet_ledger_trader_b.json"),
)
_UNCERTAIN = {"pending_submit", "submitted", "unknown", "pending", "resting"}


def inspect_original_accounting(logs_dir, account=None):
    """Read original bytes only, preserving IDs and all uncertain intent records.

    Token overlap is limited attribution evidence. It does not assign every
    manual/shared holding to this strategy or mutate historical settlement.
    """
    files, owned_tokens, order_ids, uncertain, real_count = [], set(), set(), 0, 0
    historical_tokens, blockers = set(), []
    per_owner = {}
    ledgers = list(_LEDGERS)
    known = {filename for _, filename in ledgers}
    ledgers.extend((path.stem, path.name) for path in sorted(Path(logs_dir).glob("*ledger*.json"))
                   if path.name not in known)
    for owner, filename in ledgers:
        path = Path(logs_dir) / filename
        if not path.exists():
            files.append({"name": filename, "present": False})
            continue
        raw = path.read_bytes()
        try:
            payload = json.loads(raw)
            rows = payload.get("bets")
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError("Ledger bets are not a list of mappings")
        except (ValueError, TypeError, AttributeError) as exc:
            raise AccountReadUnavailable("Original ledger could not be read without ambiguity") from exc
        real = [row for row in rows if row.get("dry_run") is False]
        if any(row.get("status") == "open" and type(row.get("dry_run")) is not bool for row in rows):
            blockers.append("unclassified_original_open_ledger_row")
        active = [row for row in real if row.get("status") == "open"]
        real_count += len(real)
        per_owner[owner] = {"real_rows": len(real), "open_real_rows": len(active)}
        historical_tokens.update(str(row.get("token_id") or row.get("asset_id") or "") for row in real)
        for row in active:
            token = str(row.get("token_id") or row.get("asset_id") or "")
            if token:
                owned_tokens.add(token)
            order_id = str(row.get("order_id") or row.get("exchange_order_id") or "")
            if order_id:
                order_ids.add(order_id)
            placement = str(row.get("placement_state") or row.get("placement_status")
                            or row.get("order_status") or "").lower()
            if placement in _UNCERTAIN:
                uncertain += 1
                if not order_id:
                    blockers.append("original_uncertain_intent_without_exchange_id")
        files.append({"name": filename, "present": True,
                      "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                      "row_count": len(rows), "unclassified_rows": sum(
                          type(row.get("dry_run")) is not bool for row in rows)})
    result = {"read_only": True, "files": files, "real_rows": real_count,
              "by_owner": per_owner, "uncertain_original_intents_preserved": uncertain,
              "complete_ownership_reconciliation": False,
              "attribution_scope": "Observed token/order overlap; manual and shared holdings remain distinct"}
    if account is not None:
        result.update(
            positions_matching_original_ledger=sum(
                str(row.get("asset") or "") in owned_tokens for row in account.positions),
            positions_without_original_ledger_match=sum(
                str(row.get("asset") or "") not in owned_tokens for row in account.positions),
            open_orders_matching_original_ledger=sum(
                str(row.get("id") or row.get("order_id") or "") in order_ids for row in account.open_orders),
            open_orders_without_original_ledger_match=sum(
                str(row.get("id") or row.get("order_id") or "") not in order_ids for row in account.open_orders),
        )
        if result["open_orders_without_original_ledger_match"]:
            blockers.append("unattributed_open_exchange_order")
        if any(str(row.get("asset") or "") not in owned_tokens
               and not _resolved_zero_position(row)
               for row in account.positions):
            blockers.append("unattributed_current_position_exposure")
        result["zero_value_positions_not_adopted"] = sum(
            str(row.get("asset") or "") not in owned_tokens
            and _resolved_zero_position(row)
            for row in account.positions)
        result["positions_matching_historical_real_token"] = sum(
            str(row.get("asset") or "") in historical_tokens for row in account.positions)
    result["activation_blockers"] = sorted(set(blockers))
    return result
