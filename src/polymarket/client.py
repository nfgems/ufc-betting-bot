"""
Polymarket API client — wraps Gamma API (markets) and CLOB API (trading).
"""

import concurrent.futures
import logging
import os
import re
import time
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Optional

import requests

CLOB_ORDER_TIMEOUT_SECONDS = 30

from src.config import (
    POLYMARKET_PRIVATE_KEY,
    POLYMARKET_FUNDER_ADDRESS,
    POLYMARKET_CHAIN_ID,
    POLYMARKET_CLOB_URL,
    POLYMARKET_GAMMA_URL,
    POLYMARKET_DATA_API_URL,
)

logger = logging.getLogger(__name__)

GEOBLOCK_CHECK_URL = "https://polymarket.com/api/geoblock"
RELAYER_TIMEOUT_SECONDS = 30
ZERO_BYTES32 = "0x" + ("00" * 32)
BYTES32_HEX_RE = re.compile(r"^0x[a-f0-9]{64}$")


def _env_first_nonempty(*names: str, default: str = "") -> str:
    for name in names:
        raw = str(os.getenv(name, "") or "").strip()
        if raw:
            return raw
    return default


POLYMARKET_RELAYER_URL = _env_first_nonempty(
    "POLYMARKET_RELAYER_URL",
    "RELAYER_URL",
    default="https://relayer-v2.polymarket.com",
)
POLYMARKET_RELAYER_API_KEY = _env_first_nonempty(
    "POLYMARKET_RELAYER_API_KEY",
    "RELAYER_API_KEY",
)
POLYMARKET_RELAYER_API_KEY_ADDRESS = _env_first_nonempty(
    "POLYMARKET_RELAYER_API_KEY_ADDRESS",
    "RELAYER_API_KEY_ADDRESS",
)
POLYMARKET_BUILDER_CODE = _env_first_nonempty("POLYMARKET_BUILDER_CODE")
LEGACY_POLYGON_USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"


class GammaClient:
    """Client for Polymarket's public market data endpoints."""

    def __init__(
        self,
        base_url: str = POLYMARKET_GAMMA_URL,
        clob_base_url: str = POLYMARKET_CLOB_URL,
    ):
        self.base_url = base_url.rstrip("/")
        self.clob_base_url = clob_base_url.rstrip("/")

    @staticmethod
    def _http_status_code(exc: Exception) -> Optional[int]:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        try:
            return int(status_code) if status_code is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _is_non_retryable_http_error(cls, exc: Exception) -> bool:
        status_code = cls._http_status_code(exc)
        return status_code is not None and 400 <= status_code < 500 and status_code != 429

    @staticmethod
    def _looks_like_condition_id(identifier: str) -> bool:
        return bool(BYTES32_HEX_RE.fullmatch(str(identifier or "").strip().lower()))

    def _request_json(
        self,
        *,
        api_name: str,
        base_url: str,
        endpoint: str,
        params: Optional[dict] = None,
    ) -> dict | list:
        url = f"{base_url}/{endpoint.lstrip('/')}"
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                non_retryable = self._is_non_retryable_http_error(exc)
                logger.warning(
                    "%s request failed for %s (attempt %s/3%s): %s",
                    api_name,
                    endpoint,
                    attempt,
                    ", non-retryable" if non_retryable else "",
                    exc,
                )
                if non_retryable:
                    break
                if attempt < 3:
                    time.sleep(min(2 ** (attempt - 1), 4))
        raise last_exc or RuntimeError(f"Gamma API request failed for {endpoint}")

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict | list:
        return self._request_json(
            api_name="Gamma API",
            base_url=self.base_url,
            endpoint=endpoint,
            params=params,
        )

    def _clob_get(self, endpoint: str, params: Optional[dict] = None) -> dict | list:
        return self._request_json(
            api_name="CLOB API",
            base_url=self.clob_base_url,
            endpoint=endpoint,
            params=params,
        )

    def get_events(
        self,
        limit: int = 100,
        offset: int = 0,
        closed: bool = False,
        tag: Optional[str] = None,
    ) -> list[dict]:
        """Get events from Gamma API."""
        params = {"limit": limit, "offset": offset, "closed": closed}
        if tag:
            params["tag"] = tag
        return self._get("events", params=params)

    def get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        closed: bool = False,
    ) -> list[dict]:
        """Get markets from Gamma API."""
        params = {"limit": limit, "offset": offset, "closed": closed}
        return self._get("markets", params=params)

    def get_market(self, market_identifier: str) -> Optional[dict]:
        """Get a specific market by condition ID (CLOB) or numeric market ID (Gamma)."""
        identifier = str(market_identifier or "").strip()
        if not identifier:
            return None
        try:
            if self._looks_like_condition_id(identifier):
                payload = self._clob_get(f"markets/{identifier.lower()}")
            else:
                payload = self._get(f"markets/{identifier}")
        except requests.HTTPError as exc:
            if self._http_status_code(exc) in {404, 422}:
                return None
            raise
        return payload if isinstance(payload, dict) else None

    def search_events(self, query: str, limit: int = 50) -> list[dict]:
        """Search events by text query."""
        return self._get("events", params={"tag": query, "limit": limit})


_proxy_patched = False  # Module-level guard: patch CLOB proxy exactly once


class ClobClientWrapper:
    """
    Wrapper for Polymarket's CLOB API for trading.

    Requires py-clob-client-v2 to be installed and a funded Polygon wallet.
    Uses Gnosis Safe proxy wallet (signature_type=2) with auto-discovered
    funder address from Polymarket's Gamma API.
    """

    SIGNATURE_TYPE_GNOSIS_SAFE = 2

    def __init__(
        self,
        private_key: Optional[str] = None,
        funder_address: Optional[str] = None,
        chain_id: int = POLYMARKET_CHAIN_ID,
        host: str = POLYMARKET_CLOB_URL,
    ):
        self.private_key = private_key or POLYMARKET_PRIVATE_KEY
        self.funder_address = funder_address or POLYMARKET_FUNDER_ADDRESS
        self.chain_id = chain_id
        self.host = host
        self._client = None
        self._api_creds = None
        self._proxy_address = None  # Discovered from Gamma API
        self._init_lock = __import__("threading").Lock()

    def _configure_shared_transport(self):
        """Return the shared py-clob-client-v2 HTTP transport, applying proxy if configured."""
        global _proxy_patched

        import py_clob_client_v2.http_helpers.helpers as clob_helpers

        if not _proxy_patched:
            clob_proxy = os.environ.get("CLOB_PROXY_URL")
            if clob_proxy:
                import httpx

                clob_helpers._http_client = httpx.Client(http2=True, proxy=clob_proxy)
                _proxy_patched = True
                logger.info(f"CLOB proxy enabled: {clob_proxy.split('@')[-1]}")

        return clob_helpers._http_client

    @property
    def proxy_address(self) -> str:
        """The proxy wallet address (funder) for this account."""
        if self._proxy_address:
            return self._proxy_address
        if self.funder_address:
            return self.funder_address
        return ""

    def _discover_proxy_address(self) -> str:
        """Auto-discover the proxy wallet address from Polymarket's Gamma API."""
        if self.funder_address:
            self._proxy_address = self.funder_address
            return self.funder_address

        # Derive EOA address from private key
        try:
            from eth_account import Account
            eoa = Account.from_key(self.private_key).address
        except Exception:
            return ""

        # Query Gamma API for the profile (maps EOA → proxy wallet)
        try:
            resp = requests.get(
                f"{POLYMARKET_GAMMA_URL}/public-profile",
                params={"address": eoa},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                proxy = data.get("proxyWallet", "")
                if proxy:
                    self._proxy_address = proxy
                    logger.info(f"Discovered proxy wallet: {proxy} (EOA: {eoa})")
                    return proxy
        except Exception as e:
            logger.warning(f"Could not discover proxy wallet: {e}")

        return ""

    def _ensure_client(self):
        """Lazily initialize the CLOB client with API credentials and funder."""
        if self._client is not None:
            return

        with self._init_lock:
            # Double-check after acquiring lock
            if self._client is not None:
                return
            self._ensure_client_locked()

    def _ensure_client_locked(self):
        """Inner init — caller must hold _init_lock."""
        if not self.private_key:
            raise ValueError(
                "POLYMARKET_PRIVATE_KEY not set. "
                "Set it in your .env file to enable trading."
            )

        try:
            from py_clob_client_v2.client import ClobClient
        except ImportError:
            raise ImportError(
                "py-clob-client-v2 not installed. Run: pip install py-clob-client-v2"
            )

        # Patch the shared transport before any authenticated requests so
        # API key derivation and order traffic use the same egress path.
        self._configure_shared_transport()

        # Discover proxy wallet (funder) address
        funder = self._discover_proxy_address()

        # Create client and derive API credentials.
        # Build fully before assigning to self._client so that concurrent
        # threads calling _ensure_client() never see a half-initialised
        # client (missing API creds).
        client = ClobClient(
            self.host,
            chain_id=self.chain_id,
            key=self.private_key,
            signature_type=self.SIGNATURE_TYPE_GNOSIS_SAFE,
            funder=funder or None,
        )
        self._api_creds = self._derive_or_create_api_key(client)
        client.set_api_creds(self._api_creds)
        self._client = client

        logger.info(
            f"CLOB client initialized (signature_type=2/GnosisSafe, "
            f"funder={funder or 'none'})"
        )

    def _derive_or_create_api_key(self, client):
        """Prefer existing L2 credentials; create only for a first-time wallet."""
        try:
            return client.derive_api_key()
        except Exception:
            logger.info("No existing Polymarket API key could be derived; creating one.")
            return client.create_api_key()

    def get_geoblock_status(self) -> dict:
        """Query Polymarket's geoblock endpoint via the shared CLOB transport."""
        self._ensure_client()
        shared_client = self._configure_shared_transport()

        try:
            resp = shared_client.get(
                GEOBLOCK_CHECK_URL,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "py_clob_client_v2",
                },
            )
            payload = resp.json() if resp.content else {}
        except Exception as e:
            return {
                "status_code": None,
                "blocked": None,
                "ip": "",
                "country": "",
                "region": "",
                "error": str(e),
            }

        return {
            "status_code": resp.status_code,
            "blocked": payload.get("blocked"),
            "ip": payload.get("ip", ""),
            "country": payload.get("country", ""),
            "region": payload.get("region", ""),
            "error": "",
        }

    def _log_geoblock_status(self, action: str) -> None:
        """Log the exact geoblock decision for the current shared CLOB transport."""
        status = self.get_geoblock_status()

        if status.get("error"):
            logger.warning(
                f"Geoblock check before {action} failed: {status['error']}"
            )
            return

        msg = (
            f"Geoblock check before {action}: blocked={status.get('blocked')} "
            f"ip={status.get('ip') or '?'} "
            f"country={status.get('country') or '?'} "
            f"region={status.get('region') or '?'} "
            f"status={status.get('status_code')}"
        )
        if status.get("blocked") is True:
            logger.warning(msg)
        else:
            logger.info(msg)

    def _normalize_orderbook_side(self, book, side: str) -> list[dict]:
        raw_levels = (
            book.get(side, []) if isinstance(book, Mapping) else getattr(book, side, [])
        ) or []
        normalized = []

        for level in raw_levels:
            if isinstance(level, Mapping):
                price = level.get("price")
                size = level.get("size")
            else:
                price = getattr(level, "price", None)
                size = getattr(level, "size", None)

            if price is None or size is None:
                raise ValueError(f"malformed orderbook {side} level: missing price/size")

            try:
                Decimal(str(price))
                Decimal(str(size))
            except (InvalidOperation, ValueError) as exc:
                raise ValueError(
                    f"malformed orderbook {side} level: invalid price/size"
                ) from exc

            normalized.append({"price": price, "size": size})

        return sorted(
            normalized,
            key=lambda level: Decimal(str(level["price"])),
            reverse=side == "bids",
        )

    def _book_to_dict(self, book) -> dict:
        """Normalize py-clob-client orderbook responses to a plain dict."""
        result = dict(book) if isinstance(book, Mapping) else {}
        result["bids"] = self._normalize_orderbook_side(book, "bids")
        result["asks"] = self._normalize_orderbook_side(book, "asks")
        return result

    def get_orderbook(self, token_id: str) -> dict:
        """Get the current orderbook for a token."""
        self._ensure_client()
        book = self._client.get_order_book(token_id)
        return self._book_to_dict(book)

    def get_midpoint_price(self, token_id: str) -> Optional[float]:
        """Get the midpoint price (average of best bid and best ask).

        Returns None if the book is empty on either side, rather than
        fabricating a synthetic midpoint.
        """
        book = self.get_orderbook(token_id)
        bids = book["bids"]
        asks = book["asks"]

        if not bids or not asks:
            return None

        best_bid = float(bids[0]["price"])
        best_ask = float(asks[0]["price"])
        return (best_bid + best_ask) / 2.0

    def get_price(self, token_id: str) -> dict:
        """Get current bid/ask/mid prices for a token.

        Returns None values for missing sides instead of fabricating prices.
        Midpoint is only defined when both sides of the book are present.
        """
        book = self.get_orderbook(token_id)
        bids = book["bids"]
        asks = book["asks"]

        best_bid = float(bids[0]["price"]) if bids else None
        best_ask = float(asks[0]["price"]) if asks else None

        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0
        else:
            mid = None

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": mid,
            "bid_size": float(bids[0]["size"]) if bids else 0.0,
            "ask_size": float(asks[0]["size"]) if asks else 0.0,
        }

    def get_clob_market_info(self, condition_id: str) -> dict:
        """Get canonical CLOB market metadata for execution-time parameters."""
        self._ensure_client()
        return self._client.get_clob_market_info(condition_id)

    def create_limit_order(
        self,
        token_id: str,
        side: str,  # "BUY" or "SELL"
        price: float,
        size: float,
        tick_size: str | None = None,
        neg_risk: bool | None = None,
        builder_code: str | None = None,
        metadata: str | None = None,
    ) -> dict:
        """
        Create and submit a GTC limit order.

        Args:
            token_id: The CLOB token ID for the outcome
            side: "BUY" or "SELL"
            price: Limit price (0-1)
            size: Number of shares
            tick_size: Minimum price increment for this market
            neg_risk: True for multi-outcome markets
            builder_code: V2 builder attribution code
            metadata: Optional V2 order metadata bytes32

        Returns order response dict.
        """
        if not (0 < price < 1):
            raise ValueError(f"Limit order price must be in (0, 1), got {price}")
        if size <= 0:
            raise ValueError(f"Limit order size must be positive, got {size}")
        if side.upper() not in ("BUY", "SELL"):
            raise ValueError(f"Limit order side must be 'BUY' or 'SELL', got {side!r}")

        self._ensure_client()
        self._log_geoblock_status("limit order")
        from py_clob_client_v2.clob_types import (
            OrderArgsV2,
            OrderType,
            PartialCreateOrderOptions,
        )

        order_args = OrderArgsV2(
            price=price,
            size=size,
            side=side.upper(),
            token_id=token_id,
            builder_code=builder_code or POLYMARKET_BUILDER_CODE or ZERO_BYTES32,
            metadata=metadata or ZERO_BYTES32,
        )

        options = PartialCreateOrderOptions(
            tick_size=tick_size,
            neg_risk=neg_risk,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                self._client.create_and_post_order,
                order_args,
                options,
                OrderType.GTC,
            )
            response = future.result(timeout=CLOB_ORDER_TIMEOUT_SECONDS)

        logger.info(
            f"Order placed: {side} {size} shares @ {price} | "
            f"Token: {token_id[:16]}... | Response: {response}"
        )
        return response

    def create_market_order(
        self,
        token_id: str,
        side: str,
        amount: float,
        tick_size: str | None = None,
        neg_risk: bool | None = None,
        builder_code: str | None = None,
        metadata: str | None = None,
        user_usdc_balance: float | None = None,
    ) -> dict:
        """
        Create and submit a FOK market order.

        Args:
            token_id: The CLOB token ID
            side: "BUY" or "SELL"
            amount: Amount in USDC to spend (for BUY) or shares to sell
            tick_size: Minimum price increment for this market
            neg_risk: True for multi-outcome markets
            builder_code: V2 builder attribution code
            metadata: Optional V2 order metadata bytes32
            user_usdc_balance: Confirmed available collateral balance used by
                the SDK to leave room for V2 market-buy fees

        Returns order response dict.
        """
        if amount <= 0:
            raise ValueError(f"Market order amount must be positive, got {amount}")
        if side.upper() not in ("BUY", "SELL"):
            raise ValueError(f"Market order side must be 'BUY' or 'SELL', got {side!r}")

        self._ensure_client()
        self._log_geoblock_status("market order")
        from py_clob_client_v2.clob_types import (
            MarketOrderArgsV2,
            OrderType,
            PartialCreateOrderOptions,
        )

        requested_amount = float(amount)
        market_args = MarketOrderArgsV2(
            token_id=token_id,
            amount=requested_amount,
            side=side.upper(),
            order_type=OrderType.FOK,
            user_usdc_balance=float(user_usdc_balance or 0.0),
            builder_code=builder_code or POLYMARKET_BUILDER_CODE or ZERO_BYTES32,
            metadata=metadata or ZERO_BYTES32,
        )

        options = PartialCreateOrderOptions(
            tick_size=tick_size,
            neg_risk=neg_risk,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                self._client.create_and_post_market_order,
                market_args,
                options,
                OrderType.FOK,
            )
            response = future.result(timeout=CLOB_ORDER_TIMEOUT_SECONDS)

        response_payload = dict(response) if isinstance(response, dict) else {"response": response}
        response_payload["_requested_amount"] = requested_amount
        response_payload["_submitted_amount"] = float(market_args.amount)
        response_payload["_execution_price"] = float(market_args.price or 0.0)

        logger.info(
            "Market order placed: %s requested $%.2f submitted $%.2f | Token: %s...",
            side,
            requested_amount,
            float(market_args.amount),
            token_id[:16],
        )
        return response_payload

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order."""
        self._ensure_client()
        from py_clob_client_v2.clob_types import OrderPayload
        return self._client.cancel_order(OrderPayload(orderID=order_id))

    def cancel_all_orders(self) -> dict:
        """Cancel all open orders."""
        self._ensure_client()
        return self._client.cancel_all()

    def get_open_orders(self) -> list[dict]:
        """Get all open orders."""
        self._ensure_client()
        return self._client.get_open_orders()

    def get_order(self, order_id: str) -> dict:
        """Get a single order, including closed orders when available."""
        self._ensure_client()
        return self._client.get_order(order_id)

    def get_trades(self, params=None) -> list[dict]:
        """Get recent trades."""
        self._ensure_client()
        return self._client.get_trades(params=params)

    def get_balance_allowance(self) -> dict:
        """Get collateral balance and allowances from the CLOB API."""
        self._ensure_client()
        from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType
        return self._client.get_balance_allowance(
            BalanceAllowanceParams(
                asset_type=AssetType.COLLATERAL,
                signature_type=self.SIGNATURE_TYPE_GNOSIS_SAFE,
            )
        )

    @staticmethod
    def _parse_cash_balance_payload(payload: dict) -> float:
        balance_raw = (
            payload.get("balance")
            or payload.get("available")
            or payload.get("available_balance")
            or "0"
        )
        decimals = int(payload.get("decimals", 6) or 6)
        balance_text = str(balance_raw).strip()
        if not balance_text:
            return 0.0
        parsed = Decimal(balance_text)
        if any(ch in balance_text for ch in ".eE"):
            return float(parsed)
        return float(parsed / (Decimal(10) ** decimals))

    def get_cash_balance_details(
        self,
        allow_onchain_fallback: bool = True,
    ) -> dict[str, float | str]:
        """
        Get the account's available USDC cash balance on Polymarket.

        Queries the CLOB API balance endpoint which reflects the proxy wallet's
        deposited collateral in the exchange contract.
        """
        try:
            ba = self.get_balance_allowance()
            return {
                "balance": self._parse_cash_balance_payload(ba),
                "source": "clob",
            }
        except (InvalidOperation, ValueError, TypeError) as e:
            logger.warning(f"Could not parse CLOB balance payload: {e}")
        except Exception as e:
            logger.warning(f"Could not fetch CLOB balance: {e}")

        if allow_onchain_fallback:
            # Fallback: query the currently active collateral token on-chain.
            proxy = self.proxy_address
            if proxy:
                try:
                    balance, source = self._get_onchain_collateral_balance(proxy)
                    return {"balance": balance, "source": source}
                except Exception as e:
                    logger.warning(f"On-chain collateral balance check failed: {e}")

        return {"balance": 0.0, "source": "unavailable"}

    def get_cash_balance(self, allow_onchain_fallback: bool = True) -> float:
        return float(
            self.get_cash_balance_details(
                allow_onchain_fallback=allow_onchain_fallback
            )["balance"]
        )

    def _get_clob_backend_version(self) -> int:
        """Resolve CLOB backend version, defaulting conservatively for pre-cutover production."""
        try:
            self._ensure_client()
            version = int(self._client.get_version())
            if version in (1, 2):
                return version
        except Exception as exc:
            logger.warning("Could not resolve CLOB backend version for collateral fallback: %s", exc)

        # The pre-cutover V2 test host should use pUSD even before production cutover.
        if "clob-v2." in str(self.host).lower():
            return 2
        return 1

    def _collateral_token_for_fallback(self) -> tuple[str, str]:
        version = self._get_clob_backend_version()
        if version >= 2:
            from py_clob_client_v2.config import get_contract_config

            return (
                get_contract_config(self.chain_id).collateral,
                "onchain_v2_collateral",
            )
        return LEGACY_POLYGON_USDC_E, "onchain_v1_collateral"

    def _get_onchain_collateral_balance(self, address: str) -> tuple[float, str]:
        """Check the version-appropriate 6-decimal collateral token on Polygon."""
        collateral, source = self._collateral_token_for_fallback()
        data = "0x70a08231" + address[2:].lower().zfill(64)
        resp = requests.post(
            "https://polygon-rpc.com",
            json={
                "jsonrpc": "2.0",
                "method": "eth_call",
                "params": [{"to": collateral, "data": data}, "latest"],
                "id": 1,
            },
            timeout=10,
        )
        result = resp.json().get("result", "0x0")
        return int(result, 16) / 1e6, source

    def get_portfolio_value_details(self) -> dict[str, float | str]:
        """Get total portfolio value (positions only) from Data API."""
        proxy = self.proxy_address
        if not proxy:
            return {"value": 0.0, "source": "unavailable"}
        try:
            resp = requests.get(
                f"{POLYMARKET_DATA_API_URL}/value",
                params={"user": proxy},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data, list):
                    return {
                        "value": float(data[0].get("value", 0.0) or 0.0),
                        "source": "data_api",
                    }
        except Exception as e:
            logger.warning(f"Could not fetch portfolio value: {e}")
        return {"value": 0.0, "source": "unavailable"}

    def get_portfolio_value(self) -> float:
        details = self.get_portfolio_value_details()
        return float(details["value"])

    def _get_signer_address(self) -> str:
        if not self.private_key:
            raise RuntimeError(
                "POLYMARKET_PRIVATE_KEY is required to sign redeem transactions"
            )
        from eth_account import Account

        return str(Account.from_key(self.private_key).address)

    def _relayer_auth_headers(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        if POLYMARKET_RELAYER_API_KEY:
            owner_address = POLYMARKET_RELAYER_API_KEY_ADDRESS or self._get_signer_address()
            return {
                "RELAYER_API_KEY": POLYMARKET_RELAYER_API_KEY,
                "RELAYER_API_KEY_ADDRESS": owner_address,
            }

        raise RuntimeError(
            "Redeeming positions requires POLYMARKET_RELAYER_API_KEY"
        )

    def _relayer_request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
        auth_required: bool = False,
    ) -> dict | list:
        headers = {"Accept": "application/json"}
        if auth_required:
            headers.update(self._relayer_auth_headers(method, path, body))

        response = requests.request(
            method=method,
            url=f"{POLYMARKET_RELAYER_URL.rstrip('/')}{path}",
            params=params,
            json=body if body is not None else None,
            headers=headers,
            timeout=RELAYER_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            snippet = response.text[:400]
            raise RuntimeError(
                f"Relayer {method} {path} failed with {response.status_code}: {snippet}"
            )
        return response.json() if response.content else {}

    def get_positions(self, *, limit: int = 500) -> list[dict]:
        """Get active wallet positions from Polymarket's Data API."""
        proxy = self.proxy_address
        if not proxy and self.private_key:
            proxy = self._discover_proxy_address()
        if not proxy:
            return []

        response = requests.get(
            f"{POLYMARKET_DATA_API_URL}/positions",
            params={
                "user": proxy,
                "limit": limit,
                "sizeThreshold": 0,
            },
            timeout=30,
        )
        response.raise_for_status()
        positions = response.json()
        return [
            position
            for position in positions
            if float(position.get("size", 0) or 0) > 0
        ]

    def get_redeemable_positions(self, *, limit: int = 500) -> list[dict]:
        """Get non-zero positions that Polymarket marks as redeemable."""
        proxy = self.proxy_address
        if not proxy and self.private_key:
            proxy = self._discover_proxy_address()
        if not proxy:
            return []

        response = requests.get(
            f"{POLYMARKET_DATA_API_URL}/positions",
            params={
                "user": proxy,
                "limit": limit,
                "sizeThreshold": 0,
                "redeemable": True,
            },
            timeout=30,
        )
        response.raise_for_status()
        positions = response.json()
        return [
            position
            for position in positions
            if float(position.get("size", 0) or 0) > 0 and bool(position.get("redeemable"))
        ]

    @staticmethod
    def _normalize_bytes32(value: str, *, field_name: str) -> str:
        raw = str(value or "").strip().lower()
        if raw.startswith("0x"):
            raw = raw[2:]
        if len(raw) != 64:
            raise ValueError(f"{field_name} must be a 32-byte hex string")
        return f"0x{raw}"

    @staticmethod
    def _encode_contract_call(signature: str, arg_types: list[str], values: list[object]) -> str:
        from eth_abi import encode
        from eth_utils import keccak

        selector = keccak(text=signature)[:4]
        encoded_args = encode(arg_types, values)
        return "0x" + selector.hex() + encoded_args.hex()

    @staticmethod
    def _redeem_index_set(position: dict) -> int:
        raw_index_set = position.get("indexSet")
        if raw_index_set is not None and str(raw_index_set).strip():
            return int(raw_index_set)

        raw_outcome_index = position.get("outcomeIndex")
        if raw_outcome_index is None or str(raw_outcome_index).strip() == "":
            raise ValueError("Redeemable position is missing outcomeIndex")

        outcome_index = int(raw_outcome_index)
        if outcome_index < 0:
            raise ValueError("Redeemable position outcomeIndex must be non-negative")
        return 1 << outcome_index

    @classmethod
    def _group_redeemable_positions(cls, positions: list[dict]) -> list[dict]:
        groups: dict[str, dict] = {}
        for position in positions:
            if not bool(position.get("redeemable")):
                continue

            condition_id = cls._normalize_bytes32(
                position.get("conditionId", ""),
                field_name="conditionId",
            )
            index_set = cls._redeem_index_set(position)
            title = str(
                position.get("title")
                or position.get("question")
                or condition_id
            )

            bucket = groups.setdefault(
                condition_id,
                {
                    "condition_id": condition_id,
                    "title": title,
                    "index_sets": set(),
                    "positions": [],
                },
            )
            bucket["index_sets"].add(index_set)
            bucket["positions"].append(dict(position))

        grouped = []
        for bucket in groups.values():
            grouped.append(
                {
                    "condition_id": bucket["condition_id"],
                    "title": bucket["title"],
                    "index_sets": sorted(bucket["index_sets"]),
                    "positions": bucket["positions"],
                    "position_count": len(bucket["positions"]),
                }
            )
        grouped.sort(key=lambda item: (item["title"], item["condition_id"]))
        return grouped

    def can_redeem_positions(self) -> bool:
        return bool(self.private_key and POLYMARKET_RELAYER_API_KEY)

    def wait_for_relayer_transaction(
        self,
        transaction_id: str,
        *,
        max_polls: int = 30,
        poll_frequency_ms: int = 2000,
    ) -> Optional[dict]:
        for _ in range(max_polls):
            transaction = self.get_relayer_transaction(transaction_id)
            if not transaction:
                time.sleep(poll_frequency_ms / 1000)
                continue

            state = str(transaction.get("state", "") or "")
            if state in {"STATE_MINED", "STATE_CONFIRMED"}:
                return transaction
            if state in {"STATE_FAILED", "STATE_INVALID"}:
                raise RuntimeError(
                    f"Relayer transaction {transaction_id} ended in {state}"
                )
            time.sleep(poll_frequency_ms / 1000)

        return None

    def get_relayer_transaction(self, transaction_id: str) -> Optional[dict]:
        """Fetch a relayer transaction by id without polling."""
        payload = self._relayer_request(
            "GET",
            "/transaction",
            params={"id": transaction_id},
            auth_required=False,
        )
        transactions = payload if isinstance(payload, list) else [payload]
        return transactions[0] if transactions else None

    def redeem_positions(
        self,
        condition_id: str,
        index_sets: list[int],
        *,
        parent_collection_id: str = ZERO_BYTES32,
        metadata: Optional[str] = None,
        wait: bool = True,
    ) -> dict:
        if not self.can_redeem_positions():
            raise RuntimeError("Polymarket redeeming is not configured for this environment")

        normalized_condition_id = self._normalize_bytes32(
            condition_id,
            field_name="condition_id",
        )
        normalized_parent = self._normalize_bytes32(
            parent_collection_id,
            field_name="parent_collection_id",
        )
        cleaned_index_sets = sorted({int(index_set) for index_set in index_sets})
        if not cleaned_index_sets:
            raise ValueError("At least one redeem index set is required")

        from py_builder_relayer_client.builder.derive import derive
        from py_builder_relayer_client.builder.safe import build_safe_transaction_request
        from py_builder_relayer_client.config import get_contract_config as get_relayer_contract_config
        from py_builder_relayer_client.models import OperationType, SafeTransaction, SafeTransactionArgs
        from py_builder_relayer_client.signer import Signer
        from py_clob_client_v2.config import get_contract_config as get_clob_contract_config
        from eth_utils import to_checksum_address

        signer = Signer(self.private_key, self.chain_id)
        relayer_contract_config = get_relayer_contract_config(self.chain_id)
        clob_contract_config = get_clob_contract_config(self.chain_id)
        safe_address = str(derive(signer.address(), relayer_contract_config.safe_factory))
        known_proxy = self.proxy_address or self._discover_proxy_address()
        if known_proxy and known_proxy.lower() != safe_address.lower():
            raise RuntimeError(
                f"Derived relayer safe {safe_address} does not match Polymarket proxy {known_proxy}"
            )

        deployed_payload = self._relayer_request(
            "GET",
            "/deployed",
            params={"address": safe_address},
            auth_required=False,
        )
        if not bool(deployed_payload.get("deployed")):
            raise RuntimeError(f"Expected Polymarket safe {safe_address} is not deployed")

        nonce_payload = self._relayer_request(
            "GET",
            "/nonce",
            params={"address": signer.address(), "type": "SAFE"},
            auth_required=False,
        )
        nonce = str(nonce_payload.get("nonce", "") or "").strip()
        if not nonce:
            raise RuntimeError("Could not fetch relayer nonce for redeem transaction")

        call_data = self._encode_contract_call(
            "redeemPositions(address,bytes32,bytes32,uint256[])",
            ["address", "bytes32", "bytes32", "uint256[]"],
            [
                to_checksum_address(clob_contract_config.collateral),
                bytes.fromhex(normalized_parent[2:]),
                bytes.fromhex(normalized_condition_id[2:]),
                cleaned_index_sets,
            ],
        )

        safe_tx = SafeTransaction(
            to=clob_contract_config.conditional_tokens,
            operation=OperationType.Call,
            data=call_data,
            value="0",
        )
        request_body = build_safe_transaction_request(
            signer=signer,
            args=SafeTransactionArgs(
                from_address=signer.address(),
                nonce=nonce,
                chain_id=self.chain_id,
                transactions=[safe_tx],
            ),
            config=relayer_contract_config,
            metadata=metadata or "",
        ).to_dict()

        submit_response = self._relayer_request(
            "POST",
            "/submit",
            body=request_body,
            auth_required=True,
        )
        result = {
            "condition_id": normalized_condition_id,
            "index_sets": cleaned_index_sets,
            "safe_address": safe_address,
            "transaction_id": submit_response.get("transactionID"),
            "transaction_hash": submit_response.get("transactionHash"),
        }

        if wait and result["transaction_id"]:
            transaction = self.wait_for_relayer_transaction(result["transaction_id"])
            if transaction is not None:
                result["state"] = transaction.get("state")
                result["transaction_hash"] = (
                    transaction.get("transactionHash") or result["transaction_hash"]
                )

        return result

    def redeem_redeemable_positions(
        self,
        *,
        positions: Optional[list[dict]] = None,
        wait: bool = False,
        limit: int = 500,
    ) -> dict:
        redeemable_positions = list(
            positions if positions is not None else self.get_redeemable_positions(limit=limit)
        )
        grouped = self._group_redeemable_positions(redeemable_positions)
        summary = {
            "redeemable_positions": len(redeemable_positions),
            "redeemable_conditions": len(grouped),
            "submitted_conditions": 0,
            "submitted_positions": 0,
            "redeemed_conditions": 0,
            "redeemed_positions": 0,
            "results": [],
            "errors": [],
            "reason": "",
            "waited_for_confirmation": bool(wait),
        }

        if not grouped:
            summary["reason"] = "no_redeemable_positions"
            return summary
        if not self.can_redeem_positions():
            summary["reason"] = "redeemer_not_configured"
            return summary

        for group in grouped:
            try:
                result = self.redeem_positions(
                    group["condition_id"],
                    group["index_sets"],
                    metadata=f"Redeem {group['position_count']} Polymarket position(s) for {group['title']}",
                    wait=wait,
                )
                result["title"] = group["title"]
                result["position_count"] = group["position_count"]
                summary["results"].append(result)
                summary["submitted_conditions"] += 1
                summary["submitted_positions"] += group["position_count"]
                if result.get("state") in {"STATE_MINED", "STATE_CONFIRMED"}:
                    summary["redeemed_conditions"] += 1
                    summary["redeemed_positions"] += group["position_count"]
            except Exception as exc:
                summary["errors"].append(
                    {
                        "condition_id": group["condition_id"],
                        "title": group["title"],
                        "error": str(exc),
                    }
                )

        if summary["results"]:
            if summary["redeemed_conditions"]:
                logger.info(
                    "Redeem confirmed for %s position(s) across %s condition(s)",
                    summary["redeemed_positions"],
                    summary["redeemed_conditions"],
                )
            else:
                logger.info(
                    "Redeem submitted for %s position(s) across %s condition(s)",
                    summary["submitted_positions"],
                    summary["submitted_conditions"],
                )
        return summary
