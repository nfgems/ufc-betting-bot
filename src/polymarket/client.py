"""
Polymarket API client — wraps Gamma API (markets) and CLOB API (trading).
"""

import logging
import os
from typing import Optional

import requests

from src.config import (
    POLYMARKET_PRIVATE_KEY,
    POLYMARKET_FUNDER_ADDRESS,
    POLYMARKET_CHAIN_ID,
    POLYMARKET_CLOB_URL,
    POLYMARKET_GAMMA_URL,
    POLYMARKET_DATA_API_URL,
)

logger = logging.getLogger(__name__)


class GammaClient:
    """Client for Polymarket's Gamma API (public market data, no auth required)."""

    def __init__(self, base_url: str = POLYMARKET_GAMMA_URL):
        self.base_url = base_url

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict | list:
        url = f"{self.base_url}/{endpoint}"
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

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

        # Route CLOB traffic through proxy (once per process)
        global _proxy_patched
        if not _proxy_patched:
            clob_proxy = os.environ.get("CLOB_PROXY_URL")
            if clob_proxy:
                import httpx
                import py_clob_client.http_helpers.helpers as clob_helpers

                clob_helpers._http_client = httpx.Client(http2=True, proxy=clob_proxy)
                _proxy_patched = True
                logger.info(f"CLOB proxy enabled: {clob_proxy.split('@')[-1]}")

        logger.info(
            f"CLOB client initialized (signature_type=2/GnosisSafe, "
            f"funder={funder or 'none'})"
        )

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

    def get_midpoint_price(self, token_id: str) -> float:
        """Get the midpoint price (average of best bid and best ask)."""
        book = self.get_orderbook(token_id)
        bids = book["bids"]
        asks = book["asks"]

        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0

        return (best_bid + best_ask) / 2.0

    def get_price(self, token_id: str) -> dict:
        """Get current bid/ask/mid prices for a token."""
        book = self.get_orderbook(token_id)
        bids = book["bids"]
        asks = book["asks"]

        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": (best_bid + best_ask) / 2.0,
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
        self._ensure_client()
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
        response = self._client.post_order(signed_order, OrderType.GTC)

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
        self._ensure_client()
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
        response = self._client.post_order(signed_order, OrderType.FOK)

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

    def get_trades(self) -> list[dict]:
        """Get recent trades."""
        self._ensure_client()
        return self._client.get_trades()

    def get_balance_allowance(self) -> dict:
        """Get USDC balance and allowances from the CLOB API."""
        self._ensure_client()
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        return self._client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )

    def get_cash_balance(self) -> float:
        """
        Get the account's available USDC cash balance on Polymarket.

        Queries the CLOB API balance endpoint which reflects the proxy wallet's
        deposited collateral in the exchange contract.
        """
        try:
            ba = self.get_balance_allowance()
            balance_str = ba.get("balance", "0")
            # CLOB API returns balance in USDC atomic units (6 decimals)
            balance = float(balance_str) / 1e6 if float(balance_str) > 1000 else float(balance_str)
            return balance
        except Exception as e:
            logger.warning(f"Could not fetch CLOB balance: {e}")

        # Fallback: query on-chain USDC balance of proxy wallet
        proxy = self.proxy_address
        if proxy:
            try:
                return self._get_onchain_usdc_balance(proxy)
            except Exception as e:
                logger.warning(f"On-chain balance check failed: {e}")

        return 0.0

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

    def get_portfolio_value(self) -> float:
        """Get total portfolio value (positions only) from Data API."""
        proxy = self.proxy_address
        if not proxy:
            return 0.0
        try:
            resp = requests.get(
                f"{POLYMARKET_DATA_API_URL}/value",
                params={"user": proxy},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data, list):
                    return data[0].get("value", 0.0)
        except Exception as e:
            logger.warning(f"Could not fetch portfolio value: {e}")
        return 0.0
