"""
Polymarket API client — wraps Gamma API (markets) and CLOB API (trading).
"""

import concurrent.futures
import logging
import os
import time
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


class GammaClient:
    """Client for Polymarket's Gamma API (public market data, no auth required)."""

    def __init__(self, base_url: str = POLYMARKET_GAMMA_URL):
        self.base_url = base_url

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict | list:
        url = f"{self.base_url}/{endpoint}"
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Gamma API request failed for %s (attempt %s/3): %s",
                    endpoint,
                    attempt,
                    exc,
                )
                if attempt < 3:
                    time.sleep(min(2 ** (attempt - 1), 4))
        raise last_exc or RuntimeError(f"Gamma API request failed for {endpoint}")

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

    def get_market(self, condition_id: str) -> dict:
        """Get a specific market by condition ID."""
        return self._get(f"markets/{condition_id}")

    def search_events(self, query: str, limit: int = 50) -> list[dict]:
        """Search events by text query."""
        return self._get("events", params={"tag": query, "limit": limit})


_proxy_patched = False  # Module-level guard: patch CLOB proxy exactly once


class ClobClientWrapper:
    """
    Wrapper for Polymarket's CLOB API for trading.

    Requires py-clob-client to be installed and a funded Polygon wallet.
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

    def _configure_shared_transport(self):
        """Return the shared py-clob-client HTTP transport, applying proxy if configured."""
        global _proxy_patched

        import py_clob_client.http_helpers.helpers as clob_helpers

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

        if not self.private_key:
            raise ValueError(
                "POLYMARKET_PRIVATE_KEY not set. "
                "Set it in your .env file to enable trading."
            )

        try:
            from py_clob_client.client import ClobClient
        except ImportError:
            raise ImportError(
                "py-clob-client not installed. Run: pip install py-clob-client"
            )

        # Patch the shared transport before any authenticated requests so
        # API key derivation and order traffic use the same egress path.
        self._configure_shared_transport()

        # Discover proxy wallet (funder) address
        funder = self._discover_proxy_address()

        # Create client and derive API credentials
        self._client = ClobClient(
            self.host,
            chain_id=self.chain_id,
            key=self.private_key,
            signature_type=self.SIGNATURE_TYPE_GNOSIS_SAFE,
            funder=funder or None,
        )
        self._api_creds = self._client.create_or_derive_api_creds()
        self._client.set_api_creds(self._api_creds)

        logger.info(
            f"CLOB client initialized (signature_type=2/GnosisSafe, "
            f"funder={funder or 'none'})"
        )

    def get_geoblock_status(self) -> dict:
        """Query Polymarket's geoblock endpoint via the shared CLOB transport."""
        self._ensure_client()
        shared_client = self._configure_shared_transport()

        try:
            resp = shared_client.get(
                GEOBLOCK_CHECK_URL,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "py_clob_client",
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

    def _book_to_dict(self, book) -> dict:
        """Convert OrderBookSummary object to a plain dict."""
        return {
            "bids": [{"price": b.price, "size": b.size} for b in (book.bids or [])],
            "asks": [{"price": a.price, "size": a.size} for a in (book.asks or [])],
        }

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

    def create_limit_order(
        self,
        token_id: str,
        side: str,  # "BUY" or "SELL"
        price: float,
        size: float,
        tick_size: str = "0.01",
        neg_risk: bool = False,
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
        from py_clob_client.order_builder.constants import BUY, SELL
        from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions

        order_side = BUY if side.upper() == "BUY" else SELL

        order_args = OrderArgs(
            price=price,
            size=size,
            side=order_side,
            token_id=token_id,
        )

        options = PartialCreateOrderOptions(
            tick_size=tick_size,
            neg_risk=neg_risk,
        )

        signed_order = self._client.create_order(order_args, options)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._client.post_order, signed_order, OrderType.GTC)
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
        tick_size: str = "0.01",
        neg_risk: bool = False,
    ) -> dict:
        """
        Create and submit a FOK market order.

        Args:
            token_id: The CLOB token ID
            side: "BUY" or "SELL"
            amount: Amount in USDC to spend (for BUY) or shares to sell
            tick_size: Minimum price increment for this market
            neg_risk: True for multi-outcome markets

        Returns order response dict.
        """
        if amount <= 0:
            raise ValueError(f"Market order amount must be positive, got {amount}")
        if side.upper() not in ("BUY", "SELL"):
            raise ValueError(f"Market order side must be 'BUY' or 'SELL', got {side!r}")

        self._ensure_client()
        self._log_geoblock_status("market order")
        from py_clob_client.order_builder.constants import BUY, SELL
        from py_clob_client.clob_types import MarketOrderArgs, OrderType, PartialCreateOrderOptions

        order_side = BUY if side.upper() == "BUY" else SELL

        market_args = MarketOrderArgs(
            token_id=token_id,
            amount=amount,
            side=order_side,
        )

        options = PartialCreateOrderOptions(
            tick_size=tick_size,
            neg_risk=neg_risk,
        )

        signed_order = self._client.create_market_order(market_args, options)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._client.post_order, signed_order, OrderType.FOK)
            response = future.result(timeout=CLOB_ORDER_TIMEOUT_SECONDS)

        logger.info(f"Market order placed: {side} ${amount} | Token: {token_id[:16]}...")
        return response

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order."""
        self._ensure_client()
        return self._client.cancel(order_id)

    def cancel_all_orders(self) -> dict:
        """Cancel all open orders."""
        self._ensure_client()
        return self._client.cancel_all()

    def get_open_orders(self) -> list[dict]:
        """Get all open orders."""
        self._ensure_client()
        return self._client.get_orders()

    def get_order(self, order_id: str) -> dict:
        """Get a single order, including closed orders when available."""
        self._ensure_client()
        return self._client.get_order(order_id)

    def get_trades(self, params=None) -> list[dict]:
        """Get recent trades."""
        self._ensure_client()
        return self._client.get_trades(params=params)

    def get_balance_allowance(self) -> dict:
        """Get USDC balance and allowances from the CLOB API."""
        self._ensure_client()
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        return self._client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
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
            # Fallback: query on-chain USDC balance of proxy wallet
            proxy = self.proxy_address
            if proxy:
                try:
                    return {
                        "balance": self._get_onchain_usdc_balance(proxy),
                        "source": "onchain",
                    }
                except Exception as e:
                    logger.warning(f"On-chain balance check failed: {e}")

        return {"balance": 0.0, "source": "unavailable"}

    def get_cash_balance(self, allow_onchain_fallback: bool = True) -> float:
        return float(
            self.get_cash_balance_details(
                allow_onchain_fallback=allow_onchain_fallback
            )["balance"]
        )

    def _get_onchain_usdc_balance(self, address: str) -> float:
        """Check USDC.e balance on Polygon for an address."""
        usdc_e = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
        data = "0x70a08231" + address[2:].lower().zfill(64)
        resp = requests.post(
            "https://polygon-rpc.com",
            json={
                "jsonrpc": "2.0",
                "method": "eth_call",
                "params": [{"to": usdc_e, "data": data}, "latest"],
                "id": 1,
            },
            timeout=10,
        )
        result = resp.json().get("result", "0x0")
        return int(result, 16) / 1e6

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
