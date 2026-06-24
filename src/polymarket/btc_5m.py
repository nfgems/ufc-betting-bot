"""BTC 5-minute Polymarket momentum runner.

This module implements a local strategy inspired by the public shape of short
duration BTC Up/Down runners, without importing or copying external bot code.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from src.config import (
    BTC5M_BINANCE_API_BASE_URL,
    BTC5M_BINANCE_SYMBOL,
    BTC5M_COINBASE_API_BASE_URL,
    BTC5M_COINBASE_PRODUCT_ID,
    BTC5M_DEFAULT_PROFILE,
    BTC5M_EXIT_SHADOW_ENABLED,
    BTC5M_EXIT_SHADOW_LOG_PATH,
    BTC5M_LEDGER_PATH,
    BTC5M_MARKET_SLUG_PREFIX,
    BTC5M_POLL_SECONDS,
    BTC5M_PAPER_LEDGER_DIR,
    BTC5M_PAPER_PROFILES,
    BTC5M_PAPER_SETTLEMENT_DELAY_SECONDS,
    BTC5M_PAPER_SETTLEMENT_SOURCE,
    BTC5M_PRICE_SOURCE,
    BTC5M_PROFILES,
    BTC5M_REQUEST_RETRIES,
    BTC5M_REQUEST_TIMEOUT_SECONDS,
    BTC5M_SIGNAL_LOG_ENABLED,
    BTC5M_SIGNAL_LOG_PATH,
    INITIAL_BANKROLL,
    POLYMARKET_CLOB_URL,
    POLYMARKET_GAMMA_URL,
)
from src.live_control import assert_polymarket_real_trading_allowed
from src.polymarket.btc5m_exit import (
    Btc5mExitContext,
    evaluate_exit_policy,
    get_exit_shadow_state,
)
from src.polymarket.client import (
    ClobClientWrapper,
    POLYMARKET_MIN_BUY_ORDER_USD,
    is_uncertain_clob_order_submission_error,
)
from src.polymarket.tracker import BetLedger

logger = logging.getLogger(__name__)

BTC5M_WINDOW_SECONDS = 5 * 60
BTC5M_ORDER_TYPE = "btc5m_marketable_limit"
BTC5M_STRATEGY_NAME = "btc5m_momentum"
BTC5M_SIGNAL_SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Btc5mProfile:
    name: str
    trade_notional_usd: float
    max_notional_per_trade: float
    max_open_exposure_usd: float
    allocation_fraction: float
    max_trades_per_day: int
    daily_loss_limit_usd: float
    entry_seconds_left: float
    entry_tolerance_seconds: float
    min_move_usd: float
    min_supporting_prob: float
    max_entry_support_gap: float
    max_entry_price: float
    max_spread: float
    max_spread_fraction: float
    min_top_ask_notional: float
    min_total_ask_notional: float
    strategy_style: str
    min_entry_price: float
    order_style: str
    trading_hours_timezone: str
    trading_start_hour: float
    trading_end_hour: float
    weekdays_only: bool
    cooldown_windows_after_trade: int
    enable_extreme_skew_hedge: bool
    hedge_skew_threshold: float
    hedge_notional_usd: float
    hedge_max_price: float
    # Cheap-side take-profit (shadow exit) tuning. Defaults reproduce the
    # historical hardcoded constants in btc5m_exit._evaluate_cheap_side so any
    # profile that omits them behaves exactly as before.
    tp_fair_value_premium: float = 0.03
    tp_profit_target: float = 0.08
    tp_trail_arm: float = 0.10
    tp_trail_giveback: float = 0.06

    @classmethod
    def from_mapping(cls, name: str, values: dict) -> "Btc5mProfile":
        return cls(
            name=name,
            trade_notional_usd=float(values["trade_notional_usd"]),
            max_notional_per_trade=float(values["max_notional_per_trade"]),
            max_open_exposure_usd=float(values["max_open_exposure_usd"]),
            allocation_fraction=float(values["allocation_fraction"]),
            max_trades_per_day=int(values["max_trades_per_day"]),
            daily_loss_limit_usd=float(values["daily_loss_limit_usd"]),
            entry_seconds_left=float(values["entry_seconds_left"]),
            entry_tolerance_seconds=float(values["entry_tolerance_seconds"]),
            min_move_usd=float(values["min_move_usd"]),
            min_supporting_prob=float(values["min_supporting_prob"]),
            max_entry_support_gap=float(values.get("max_entry_support_gap", 0.0) or 0.0),
            max_entry_price=float(values["max_entry_price"]),
            max_spread=float(values["max_spread"]),
            max_spread_fraction=float(values.get("max_spread_fraction", 0.0) or 0.0),
            min_top_ask_notional=float(values["min_top_ask_notional"]),
            min_total_ask_notional=float(values["min_total_ask_notional"]),
            strategy_style=str(values.get("strategy_style", "momentum") or "momentum"),
            min_entry_price=float(values.get("min_entry_price", 0.0) or 0.0),
            order_style=str(values.get("order_style", "marketable_limit") or "marketable_limit"),
            trading_hours_timezone=str(values.get("trading_hours_timezone", "") or ""),
            trading_start_hour=float(values.get("trading_start_hour", 0.0) or 0.0),
            trading_end_hour=float(values.get("trading_end_hour", 24.0) or 24.0),
            weekdays_only=bool(values.get("weekdays_only", False)),
            cooldown_windows_after_trade=int(values.get("cooldown_windows_after_trade", 0) or 0),
            enable_extreme_skew_hedge=bool(values["enable_extreme_skew_hedge"]),
            hedge_skew_threshold=float(values["hedge_skew_threshold"]),
            hedge_notional_usd=float(values["hedge_notional_usd"]),
            hedge_max_price=float(values["hedge_max_price"]),
            tp_fair_value_premium=float(values.get("tp_fair_value_premium", 0.03) or 0.03),
            tp_profit_target=float(values.get("tp_profit_target", 0.08) or 0.08),
            tp_trail_arm=float(values.get("tp_trail_arm", 0.10) or 0.10),
            tp_trail_giveback=float(values.get("tp_trail_giveback", 0.06) or 0.06),
        )


@dataclass(frozen=True)
class Btc5mMarket:
    event_id: str
    market_id: str
    condition_id: str
    slug: str
    question: str
    window_start: datetime
    window_end: datetime
    token_up: str
    token_down: str
    price_up: float | None
    price_down: float | None
    tick_size: str
    order_min_size: float
    neg_risk: bool
    active: bool
    closed: bool
    accepting_orders: bool
    fee_schedule: dict


@dataclass(frozen=True)
class BtcPriceSnapshot:
    source: str
    symbol: str
    window_start_price: float
    current_price: float
    fetched_at: datetime

    @property
    def move_usd(self) -> float:
        return self.current_price - self.window_start_price


@dataclass(frozen=True)
class BtcReferencePrice:
    source: str
    symbol: str
    price: float
    timestamp: datetime
    fetched_at: datetime


@dataclass(frozen=True)
class Btc5mSettlementReference:
    source: str
    symbol: str
    winning_side: str
    timestamp: datetime
    fetched_at: datetime
    start_price: float | None = None
    final_price: float | None = None
    note: str = ""


@dataclass(frozen=True)
class OrderBookSummary:
    token_id: str
    best_bid: float | None
    best_ask: float | None
    bid_size: float
    ask_size: float
    spread: float | None
    top_ask_notional: float
    total_ask_notional: float
    min_order_size: float
    tick_size: str
    neg_risk: bool

    @property
    def midpoint(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        if self.best_bid > self.best_ask:
            return None
        return (self.best_bid + self.best_ask) / 2.0


def resolve_btc5m_profile(profile_name: str | None = None) -> Btc5mProfile:
    name = str(profile_name or BTC5M_DEFAULT_PROFILE or "conservative").strip().lower()
    values = BTC5M_PROFILES.get(name)
    if values is None:
        allowed = ", ".join(sorted(BTC5M_PROFILES))
        raise ValueError(f"Unknown BTC5M profile {name!r}. Allowed profiles: {allowed}")
    return Btc5mProfile.from_mapping(name, values)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_utc(value: datetime | None) -> datetime:
    if value is None:
        return utcnow()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def btc5m_window_start(now: datetime | None = None) -> datetime:
    current = _coerce_utc(now)
    epoch = int(current.timestamp())
    start_epoch = epoch - (epoch % BTC5M_WINDOW_SECONDS)
    return datetime.fromtimestamp(start_epoch, tz=timezone.utc)


def btc5m_slug_for_window(
    now: datetime | None = None,
    *,
    prefix: str = BTC5M_MARKET_SLUG_PREFIX,
) -> str:
    start = btc5m_window_start(now)
    return f"{prefix}-{int(start.timestamp())}"


def _parse_json_field(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def _safe_float(value, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    return parsed


def _round_optional(value, digits: int = 6):
    parsed = _safe_float(value, math.nan)
    if math.isnan(parsed):
        return None
    return round(parsed, digits)


def _parse_probability(value) -> float | None:
    parsed = _safe_float(value, math.nan)
    if math.isnan(parsed) or parsed < 0.0 or parsed > 1.0:
        return None
    return parsed


def _extract_slug_epoch(slug: str) -> int | None:
    raw = str(slug or "").strip().rstrip("/")
    if not raw:
        return None
    tail = raw.rsplit("-", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return None


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _market_fee_schedule(market: dict) -> dict:
    schedule = market.get("feeSchedule") or market.get("fee_schedule")
    return dict(schedule) if isinstance(schedule, dict) else {}


def _parse_btc5m_market(event: dict) -> Btc5mMarket | None:
    markets = event.get("markets")
    if not isinstance(markets, list):
        markets = []

    event_slug = str(event.get("slug", "") or "")
    event_id = str(event.get("id", "") or "")

    for market in markets:
        if not isinstance(market, dict):
            continue
        outcomes = [str(value).strip() for value in _parse_json_field(market.get("outcomes"))]
        normalized = [value.casefold() for value in outcomes]
        if "up" not in normalized or "down" not in normalized:
            continue

        tokens = _parse_json_field(market.get("clobTokenIds"))
        if len(tokens) < 2:
            continue

        prices = _parse_json_field(market.get("outcomePrices"))
        outcome_to_token = dict(zip(normalized, [str(token) for token in tokens]))
        outcome_to_price = {
            outcome: _parse_probability(prices[index]) if index < len(prices) else None
            for index, outcome in enumerate(normalized)
        }

        slug = str(market.get("slug") or event_slug or "")
        start_epoch = _extract_slug_epoch(slug)
        if start_epoch is None:
            continue
        window_start = datetime.fromtimestamp(start_epoch, tz=timezone.utc)
        window_end = datetime.fromtimestamp(start_epoch + BTC5M_WINDOW_SECONDS, tz=timezone.utc)

        return Btc5mMarket(
            event_id=event_id,
            market_id=str(market.get("id", "") or ""),
            condition_id=str(market.get("conditionId") or market.get("condition_id") or ""),
            slug=slug,
            question=str(market.get("question") or event.get("title") or ""),
            window_start=window_start,
            window_end=window_end,
            token_up=outcome_to_token["up"],
            token_down=outcome_to_token["down"],
            price_up=outcome_to_price.get("up"),
            price_down=outcome_to_price.get("down"),
            tick_size=str(market.get("orderPriceMinTickSize") or market.get("tick_size") or "0.01"),
            order_min_size=_safe_float(market.get("orderMinSize"), 5.0),
            neg_risk=_parse_bool(market.get("negRisk")),
            active=_parse_bool(market.get("active", event.get("active", True))),
            closed=_parse_bool(market.get("closed", event.get("closed", False))),
            accepting_orders=_parse_bool(market.get("acceptingOrders", True)),
            fee_schedule=_market_fee_schedule(market),
        )

    return None


def _find_btc5m_market_payload(event: dict, market_slug: str | None = None) -> dict | None:
    markets = event.get("markets")
    if not isinstance(markets, list):
        return None

    normalized_slug = str(market_slug or event.get("slug") or "").strip()
    for market in markets:
        if not isinstance(market, dict):
            continue
        slug = str(market.get("slug") or event.get("slug") or "").strip()
        if normalized_slug and slug and slug != normalized_slug:
            continue
        outcomes = [str(value).strip().casefold() for value in _parse_json_field(market.get("outcomes"))]
        if "up" in outcomes and "down" in outcomes:
            return market
    return None


def _official_winning_side_from_market_payload(market: dict) -> str | None:
    status = str(market.get("umaResolutionStatus") or "").strip().lower()
    closed = _parse_bool(market.get("closed"))
    accepting_orders = _parse_bool(market.get("acceptingOrders", False))
    if status and status != "resolved":
        return None
    if not closed and status != "resolved":
        return None
    if accepting_orders:
        return None

    outcomes = [str(value).strip().casefold() for value in _parse_json_field(market.get("outcomes"))]
    prices = _parse_json_field(market.get("outcomePrices"))
    if len(outcomes) != len(prices):
        return None

    parsed: dict[str, float] = {}
    for index, outcome in enumerate(outcomes):
        if outcome not in {"up", "down"}:
            continue
        probability = _parse_probability(prices[index])
        if probability is None:
            return None
        parsed[outcome] = probability

    if set(parsed) != {"up", "down"}:
        return None
    winners = [side for side, probability in parsed.items() if probability >= 0.99]
    losers = [side for side, probability in parsed.items() if probability <= 0.01]
    if len(winners) == 1 and len(losers) == 1 and winners[0] != losers[0]:
        return winners[0]
    return None


def official_btc5m_winning_side_from_event(event: dict, market_slug: str | None = None) -> str | None:
    market = _find_btc5m_market_payload(event, market_slug)
    if market is None:
        return None
    return _official_winning_side_from_market_payload(market)


class Btc5mMarketClient:
    def __init__(
        self,
        *,
        gamma_url: str = POLYMARKET_GAMMA_URL,
        timeout_seconds: float = BTC5M_REQUEST_TIMEOUT_SECONDS,
        request_retries: int = BTC5M_REQUEST_RETRIES,
        session: requests.Session | None = None,
    ):
        self.gamma_url = gamma_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.request_retries = max(int(request_retries), 0)
        self.session = session or requests.Session()

    def fetch_event_by_slug(self, slug: str) -> dict | None:
        url = f"{self.gamma_url}/events"
        for attempt in range(self.request_retries + 1):
            try:
                response = self.session.get(
                    url,
                    params={"slug": slug},
                    timeout=self.timeout_seconds,
                )
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                if attempt >= self.request_retries:
                    raise
                delay = min(0.5 * (attempt + 1), 2.0)
                logger.warning(
                    "Transient Polymarket Gamma request failure for %s; retrying in %.1fs (%s/%s): %s",
                    slug,
                    delay,
                    attempt + 1,
                    self.request_retries,
                    exc,
                )
                time.sleep(delay)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list) and payload:
            event = payload[0]
            return event if isinstance(event, dict) else None
        if isinstance(payload, dict):
            return payload
        return None

    def get_market(
        self,
        *,
        now: datetime | None = None,
        market_slug: str | None = None,
    ) -> Btc5mMarket | None:
        current = _coerce_utc(now)
        if market_slug:
            slugs = [market_slug]
        else:
            base_start = btc5m_window_start(current)
            slugs = [
                f"{BTC5M_MARKET_SLUG_PREFIX}-{int((base_start.timestamp() + offset))}"
                for offset in (0, -BTC5M_WINDOW_SECONDS, BTC5M_WINDOW_SECONDS)
            ]

        for slug in slugs:
            event = self.fetch_event_by_slug(slug)
            if not event:
                continue
            market = _parse_btc5m_market(event)
            if market is None:
                continue
            if market_slug:
                return market
            if market.window_start <= current < market.window_end:
                return market
        return None


class BinanceBtcPriceClient:
    def __init__(
        self,
        *,
        base_url: str = BTC5M_BINANCE_API_BASE_URL,
        symbol: str = BTC5M_BINANCE_SYMBOL,
        timeout_seconds: float = BTC5M_REQUEST_TIMEOUT_SECONDS,
        request_retries: int = BTC5M_REQUEST_RETRIES,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.symbol = symbol
        self.timeout_seconds = timeout_seconds
        self.request_retries = max(int(request_retries), 0)
        self.session = session or requests.Session()

    def _get_json(self, path: str, params: dict) -> object:
        url = f"{self.base_url}{path}"
        for attempt in range(self.request_retries + 1):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                return response.json()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                if attempt >= self.request_retries:
                    raise
                delay = min(0.5 * (attempt + 1), 2.0)
                logger.warning(
                    "Transient Binance request failure for %s; retrying in %.1fs (%s/%s): %s",
                    path,
                    delay,
                    attempt + 1,
                    self.request_retries,
                    exc,
                )
                time.sleep(delay)
        raise RuntimeError(f"Binance request failed unexpectedly for {path}")

    def get_snapshot(
        self,
        *,
        window_start: datetime,
        now: datetime | None = None,
    ) -> BtcPriceSnapshot:
        start_ms = int(_coerce_utc(window_start).timestamp() * 1000)
        klines = self._get_json(
            "/api/v3/klines",
            params={
                "symbol": self.symbol,
                "interval": "1m",
                "startTime": start_ms,
                "limit": 1,
            },
        )
        if not isinstance(klines, list) or not klines:
            raise RuntimeError(f"No Binance kline available for {self.symbol} at {start_ms}")
        first_kline = klines[0]
        if not isinstance(first_kline, list) or len(first_kline) < 2:
            raise RuntimeError(f"Malformed Binance kline payload for {self.symbol}: {first_kline!r}")
        window_start_price = float(first_kline[1])

        ticker = self._get_json(
            "/api/v3/ticker/price",
            params={"symbol": self.symbol},
        )
        if not isinstance(ticker, dict) or "price" not in ticker:
            raise RuntimeError(f"Malformed Binance ticker payload for {self.symbol}: {ticker!r}")

        return BtcPriceSnapshot(
            source="binance",
            symbol=self.symbol,
            window_start_price=window_start_price,
            current_price=float(ticker["price"]),
            fetched_at=_coerce_utc(now),
        )

    def get_price_at(self, timestamp: datetime, *, now: datetime | None = None) -> BtcReferencePrice:
        target = _coerce_utc(timestamp)
        target_ms = int(target.timestamp() * 1000)
        klines = self._get_json(
            "/api/v3/klines",
            params={
                "symbol": self.symbol,
                "interval": "1m",
                "startTime": target_ms,
                "limit": 1,
            },
        )
        if not isinstance(klines, list) or not klines:
            raise RuntimeError(f"No Binance settlement kline available for {self.symbol} at {target_ms}")
        first_kline = klines[0]
        if not isinstance(first_kline, list) or len(first_kline) < 2:
            raise RuntimeError(f"Malformed Binance settlement kline for {self.symbol}: {first_kline!r}")
        return BtcReferencePrice(
            source="binance",
            symbol=self.symbol,
            price=float(first_kline[1]),
            timestamp=target,
            fetched_at=_coerce_utc(now),
        )


class CoinbaseBtcPriceClient:
    def __init__(
        self,
        *,
        base_url: str = BTC5M_COINBASE_API_BASE_URL,
        product_id: str = BTC5M_COINBASE_PRODUCT_ID,
        timeout_seconds: float = BTC5M_REQUEST_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.product_id = product_id
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def _get_json(self, path: str, params: dict | None = None) -> object:
        response = self.session.get(
            f"{self.base_url}{path}",
            params=params or {},
            timeout=self.timeout_seconds,
            headers={"User-Agent": "ufc-betting-bot-btc5m/1.0"},
        )
        response.raise_for_status()
        return response.json()

    def get_snapshot(
        self,
        *,
        window_start: datetime,
        now: datetime | None = None,
    ) -> BtcPriceSnapshot:
        start = _coerce_utc(window_start)
        end = datetime.fromtimestamp(int(start.timestamp()) + 60, tz=timezone.utc)
        candles = self._get_json(
            f"/products/{self.product_id}/candles",
            params={
                "start": start.isoformat().replace("+00:00", "Z"),
                "end": end.isoformat().replace("+00:00", "Z"),
                "granularity": 60,
            },
        )
        if not isinstance(candles, list) or not candles:
            raise RuntimeError(f"No Coinbase candle available for {self.product_id} at {start.isoformat()}")

        start_epoch = int(start.timestamp())
        candle = min(
            (item for item in candles if isinstance(item, list) and len(item) >= 4),
            key=lambda item: abs(int(item[0]) - start_epoch),
            default=None,
        )
        if candle is None:
            raise RuntimeError(f"Malformed Coinbase candle payload for {self.product_id}: {candles!r}")
        window_start_price = float(candle[3])

        ticker = self._get_json(f"/products/{self.product_id}/ticker")
        if not isinstance(ticker, dict) or "price" not in ticker:
            raise RuntimeError(f"Malformed Coinbase ticker payload for {self.product_id}: {ticker!r}")

        return BtcPriceSnapshot(
            source="coinbase",
            symbol=self.product_id,
            window_start_price=window_start_price,
            current_price=float(ticker["price"]),
            fetched_at=_coerce_utc(now),
        )

    def get_price_at(self, timestamp: datetime, *, now: datetime | None = None) -> BtcReferencePrice:
        target = _coerce_utc(timestamp)
        end = datetime.fromtimestamp(int(target.timestamp()) + 60, tz=timezone.utc)
        candles = self._get_json(
            f"/products/{self.product_id}/candles",
            params={
                "start": target.isoformat().replace("+00:00", "Z"),
                "end": end.isoformat().replace("+00:00", "Z"),
                "granularity": 60,
            },
        )
        if not isinstance(candles, list) or not candles:
            raise RuntimeError(f"No Coinbase settlement candle available for {self.product_id} at {target.isoformat()}")

        target_epoch = int(target.timestamp())
        candle = min(
            (item for item in candles if isinstance(item, list) and len(item) >= 4),
            key=lambda item: abs(int(item[0]) - target_epoch),
            default=None,
        )
        if candle is None:
            raise RuntimeError(f"Malformed Coinbase settlement candle for {self.product_id}: {candles!r}")
        return BtcReferencePrice(
            source="coinbase",
            symbol=self.product_id,
            price=float(candle[3]),
            timestamp=target,
            fetched_at=_coerce_utc(now),
        )


def build_btc_price_client(price_source: str | None = None):
    source = str(price_source or BTC5M_PRICE_SOURCE or "coinbase").strip().lower()
    if source == "binance":
        return BinanceBtcPriceClient()
    if source == "coinbase":
        return CoinbaseBtcPriceClient()
    raise ValueError("Unknown BTC5M_PRICE_SOURCE {!r}. Allowed values: coinbase, binance".format(source))


class ProxyBtc5mSettlementClient:
    def __init__(self, price_client):
        self.price_client = price_client

    def settle_bet(
        self,
        bet: dict,
        *,
        window_end: datetime,
        now: datetime | None = None,
    ) -> Btc5mSettlementReference:
        start_price = _paper_settlement_start_price(bet)
        if math.isnan(start_price) or start_price <= 0:
            raise RuntimeError(f"bet #{bet.get('id')} missing BTC start price")
        reference = self.price_client.get_price_at(window_end, now=now)
        final_price = float(reference.price)
        return Btc5mSettlementReference(
            source=f"{reference.source}_proxy",
            symbol=reference.symbol,
            winning_side=_paper_outcome_for_prices(start_price, final_price),
            timestamp=reference.timestamp,
            fetched_at=reference.fetched_at,
            start_price=start_price,
            final_price=final_price,
            note=(
                "Proxy settlement uses the selected BTC price feed, not the official "
                "Polymarket Chainlink BTC/USD Data Stream resolution."
            ),
        )


class PolymarketBtc5mSettlementClient:
    def __init__(self, *, market_client: Btc5mMarketClient | None = None):
        self.market_client = market_client or Btc5mMarketClient()

    def settle_bet(
        self,
        bet: dict,
        *,
        window_end: datetime,
        now: datetime | None = None,
    ) -> Btc5mSettlementReference:
        slug = str(bet.get("market_slug") or "").strip()
        if not slug:
            raise RuntimeError(f"bet #{bet.get('id')} missing BTC 5m market slug")
        event = self.market_client.fetch_event_by_slug(slug)
        if not event:
            raise RuntimeError(f"official Polymarket resolution not found for {slug}")
        winning_side = official_btc5m_winning_side_from_event(event, slug)
        if winning_side is None:
            raise RuntimeError(f"official Polymarket resolution not available yet for {slug}")
        return Btc5mSettlementReference(
            source="polymarket_chainlink",
            symbol="BTC/USD Data Stream",
            winning_side=winning_side,
            timestamp=window_end,
            fetched_at=_coerce_utc(now),
            start_price=_paper_settlement_start_price(bet),
            final_price=None,
            note=(
                "Settled from Polymarket's resolved outcome. Polymarket BTC 5m "
                "rules resolve against the Chainlink BTC/USD Data Stream."
            ),
        )


def build_btc5m_settlement_client(
    *,
    settlement_source: str | None = None,
    price_client=None,
    market_client: Btc5mMarketClient | None = None,
):
    source = str(settlement_source or BTC5M_PAPER_SETTLEMENT_SOURCE or "polymarket").strip().lower()
    if source in {"polymarket", "official", "chainlink", "chainlink_aligned", "chainlink-aligned"}:
        return PolymarketBtc5mSettlementClient(market_client=market_client)
    if source in {"coinbase", "binance"}:
        return ProxyBtc5mSettlementClient(price_client or build_btc_price_client(source))
    if source in {"proxy", "price"}:
        return ProxyBtc5mSettlementClient(price_client or build_btc_price_client())
    allowed = "polymarket, coinbase, binance, proxy"
    raise ValueError(f"Unknown BTC5M_PAPER_SETTLEMENT_SOURCE {source!r}. Allowed values: {allowed}")


class PublicClobOrderBookClient:
    def __init__(
        self,
        *,
        clob_url: str = POLYMARKET_CLOB_URL,
        timeout_seconds: float = BTC5M_REQUEST_TIMEOUT_SECONDS,
        request_retries: int = BTC5M_REQUEST_RETRIES,
        session: requests.Session | None = None,
    ):
        self.clob_url = clob_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.request_retries = max(int(request_retries), 0)
        self.session = session or requests.Session()

    def get_orderbook(self, token_id: str) -> dict:
        url = f"{self.clob_url}/book"
        for attempt in range(self.request_retries + 1):
            try:
                response = self.session.get(
                    url,
                    params={"token_id": token_id},
                    timeout=self.timeout_seconds,
                )
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                if attempt >= self.request_retries:
                    raise
                delay = min(0.5 * (attempt + 1), 2.0)
                logger.warning(
                    "Transient Polymarket CLOB book request failure for token %s; retrying in %.1fs (%s/%s): %s",
                    token_id,
                    delay,
                    attempt + 1,
                    self.request_retries,
                    exc,
                )
                time.sleep(delay)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def summarize(self, token_id: str) -> OrderBookSummary:
        book = self.get_orderbook(token_id)
        bids = _normalize_book_side(book.get("bids"), reverse=True)
        asks = _normalize_book_side(book.get("asks"), reverse=False)

        best_bid = bids[0]["price"] if bids else None
        best_ask = asks[0]["price"] if asks else None
        bid_size = bids[0]["size"] if bids else 0.0
        ask_size = asks[0]["size"] if asks else 0.0
        spread = (
            best_ask - best_bid
            if best_bid is not None and best_ask is not None and best_ask >= best_bid
            else None
        )
        total_ask_notional = sum(level["price"] * level["size"] for level in asks)
        top_ask_notional = best_ask * ask_size if best_ask is not None else 0.0

        return OrderBookSummary(
            token_id=token_id,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_size=bid_size,
            ask_size=ask_size,
            spread=spread,
            top_ask_notional=top_ask_notional,
            total_ask_notional=total_ask_notional,
            min_order_size=_safe_float(book.get("min_order_size"), 5.0),
            tick_size=str(book.get("tick_size") or "0.01"),
            neg_risk=_parse_bool(book.get("neg_risk")),
        )


def _normalize_book_side(levels, *, reverse: bool) -> list[dict]:
    result = []
    for level in levels or []:
        if not isinstance(level, dict):
            continue
        price = _parse_probability(level.get("price"))
        size = _safe_float(level.get("size"), math.nan)
        if price is None or math.isnan(size) or size <= 0:
            continue
        result.append({"price": price, "size": size})
    return sorted(result, key=lambda item: item["price"], reverse=reverse)


def _book_for_direction(
    direction: str,
    *,
    up_book: OrderBookSummary,
    down_book: OrderBookSummary,
) -> OrderBookSummary:
    return up_book if direction == "up" else down_book


def _price_for_direction(direction: str, market: Btc5mMarket) -> float | None:
    return market.price_up if direction == "up" else market.price_down


def _order_style(profile: Btc5mProfile) -> str:
    return str(profile.order_style or "marketable_limit").strip().lower()


def _strategy_style(profile: Btc5mProfile) -> str:
    return str(profile.strategy_style or "momentum").strip().lower()


def _entry_price_for_book(profile: Btc5mProfile, book: OrderBookSummary) -> tuple[float | None, str]:
    style = _order_style(profile)
    if style == "maker_limit":
        return book.best_bid, "best_bid"
    return book.best_ask, "best_ask"


def _entry_window_skip(profile: Btc5mProfile, seconds_left: float) -> dict | None:
    entry_delta = abs(seconds_left - profile.entry_seconds_left)
    if entry_delta <= profile.entry_tolerance_seconds:
        return None
    return _skip(
        "outside_entry_window",
        (
            f"Outside entry window: {seconds_left:.0f}s left, target "
            f"{profile.entry_seconds_left:.0f}s +/- {profile.entry_tolerance_seconds:.0f}s."
        ),
        seconds_left=seconds_left,
    )


def _trading_hours_skip(profile: Btc5mProfile, current: datetime) -> dict | None:
    tz_name = str(profile.trading_hours_timezone or "").strip()
    if not tz_name and not profile.weekdays_only:
        return None

    local_now = current
    if tz_name:
        try:
            local_now = current.astimezone(ZoneInfo(tz_name))
        except ZoneInfoNotFoundError:
            return _skip(
                "invalid_trading_timezone",
                f"Unknown BTC 5m trading timezone {tz_name!r}.",
                timezone=tz_name,
            )

    if profile.weekdays_only and local_now.weekday() >= 5:
        return _skip(
            "outside_trading_hours",
            f"Outside BTC 5m trading schedule: {local_now:%A} is not a weekday.",
            local_time=local_now.isoformat(),
        )

    local_hour = local_now.hour + (local_now.minute / 60.0) + (local_now.second / 3600.0)
    if local_hour < profile.trading_start_hour or local_hour >= profile.trading_end_hour:
        return _skip(
            "outside_trading_hours",
            (
                f"Outside BTC 5m trading hours: local hour {local_hour:.2f}, "
                f"allowed {profile.trading_start_hour:g}-{profile.trading_end_hour:g}."
            ),
            local_time=local_now.isoformat(),
        )
    return None


def _entry_price_range_skip(
    *,
    profile: Btc5mProfile,
    direction: str,
    entry_price: float,
    entry_price_source: str,
) -> dict | None:
    if entry_price < profile.min_entry_price:
        return _skip(
            "entry_price_too_low",
            (
                f"{direction.upper()} {entry_price_source.replace('_', ' ')} ${entry_price:.4f} is below min entry "
                f"${profile.min_entry_price:.4f}."
            ),
            direction=direction,
            entry_price=entry_price,
            entry_price_source=entry_price_source,
        )
    if entry_price > profile.max_entry_price:
        return _skip(
            "entry_price_too_high",
            (
                f"{direction.upper()} {entry_price_source.replace('_', ' ')} ${entry_price:.4f} exceeds max entry "
                f"${profile.max_entry_price:.4f}."
            ),
            direction=direction,
            entry_price=entry_price,
            entry_price_source=entry_price_source,
        )
    return None


def _book_liquidity_skip(
    *,
    profile: Btc5mProfile,
    direction: str,
    book: OrderBookSummary,
    entry_price: float | None = None,
) -> dict | None:
    if book.spread is None:
        return _skip("missing_spread", f"Could not calculate {direction.upper()} spread.")
    if book.spread > profile.max_spread:
        return _skip(
            "spread_too_wide",
            f"{direction.upper()} spread {book.spread:.1%} exceeds max {profile.max_spread:.1%}.",
            direction=direction,
            spread=book.spread,
        )
    if profile.max_spread_fraction > 0 and entry_price and entry_price > 0:
        relative_threshold = entry_price * profile.max_spread_fraction
        if book.spread > relative_threshold:
            return _skip(
                "relative_spread_too_wide",
                (
                    f"{direction.upper()} spread {book.spread:.1%} exceeds relative cap "
                    f"{relative_threshold:.1%} ({profile.max_spread_fraction:.1%} of entry)."
                ),
                direction=direction,
                spread=book.spread,
                max_relative_spread=relative_threshold,
                entry_price=entry_price,
            )
    if book.top_ask_notional < profile.min_top_ask_notional:
        return _skip(
            "thin_top_ask",
            (
                f"{direction.upper()} top ask notional ${book.top_ask_notional:.2f} "
                f"is below ${profile.min_top_ask_notional:.2f}."
            ),
            direction=direction,
            top_ask_notional=book.top_ask_notional,
        )
    if book.total_ask_notional < profile.min_total_ask_notional:
        return _skip(
            "thin_book",
            (
                f"{direction.upper()} ask book ${book.total_ask_notional:.2f} "
                f"is below ${profile.min_total_ask_notional:.2f}."
            ),
            direction=direction,
            total_ask_notional=book.total_ask_notional,
        )
    return None


def _book_supporting_prob(direction: str, market: Btc5mMarket, book: OrderBookSummary) -> float | None:
    return book.midpoint if book.midpoint is not None else _price_for_direction(direction, market)


def _entry_support_gap_skip(
    *,
    profile: Btc5mProfile,
    direction: str,
    supporting_prob: float,
    entry_price: float,
) -> dict | None:
    max_gap = profile.max_entry_support_gap
    if max_gap <= 0:
        return None
    gap = entry_price - supporting_prob
    if gap <= max_gap:
        return None
    return _skip(
        "entry_support_gap_too_wide",
        (
            f"{direction.upper()} entry ${entry_price:.4f} is {gap:.1%} above "
            f"market support {supporting_prob:.1%}; max gap {max_gap:.1%}."
        ),
        direction=direction,
        supporting_prob=supporting_prob,
        entry_price=entry_price,
        entry_support_gap=gap,
        max_entry_support_gap=max_gap,
    )


def _book_snapshot(book: OrderBookSummary) -> dict:
    return {
        "token_id": book.token_id,
        "best_bid": _round_optional(book.best_bid),
        "best_ask": _round_optional(book.best_ask),
        "bid_size": round(book.bid_size, 4),
        "ask_size": round(book.ask_size, 4),
        "spread": _round_optional(book.spread),
        "midpoint": _round_optional(book.midpoint),
        "top_ask_notional": round(book.top_ask_notional, 4),
        "total_ask_notional": round(book.total_ask_notional, 4),
        "min_order_size": round(book.min_order_size, 4),
    }


def _build_signal_snapshot(
    *,
    market: Btc5mMarket,
    price_snapshot: BtcPriceSnapshot,
    up_book: OrderBookSummary,
    down_book: OrderBookSummary,
    profile: Btc5mProfile,
    signal: dict,
    result: dict,
    risk: dict | None = None,
    now: datetime | None = None,
) -> dict:
    current = _coerce_utc(now)
    price_to_beat = price_snapshot.window_start_price
    current_price = price_snapshot.current_price
    distance_usd = current_price - price_to_beat
    distance_pct = (distance_usd / price_to_beat) if price_to_beat else None
    seconds_left = (market.window_end - current).total_seconds()
    return {
        "schema_version": BTC5M_SIGNAL_SNAPSHOT_SCHEMA_VERSION,
        "recorded_at": current.isoformat(),
        "strategy": BTC5M_STRATEGY_NAME,
        "profile": profile.name,
        "strategy_style": _strategy_style(profile),
        "market": {
            "slug": market.slug,
            "market_id": market.market_id,
            "condition_id": market.condition_id,
            "window_start": market.window_start.isoformat(),
            "window_end": market.window_end.isoformat(),
            "seconds_left": round(seconds_left, 3),
            "active": market.active,
            "closed": market.closed,
            "accepting_orders": market.accepting_orders,
        },
        "price": {
            "source": price_snapshot.source,
            "symbol": price_snapshot.symbol,
            "price_to_beat": round(price_to_beat, 4),
            "current_price": round(current_price, 4),
            "distance_to_price_to_beat_usd": round(distance_usd, 4),
            "distance_to_price_to_beat_pct": _round_optional(distance_pct),
            "btc_above_price_to_beat": current_price >= price_to_beat,
        },
        "books": {
            "up": _book_snapshot(up_book),
            "down": _book_snapshot(down_book),
        },
        "portfolio": risk or {},
        "signal": {
            "action": signal.get("action"),
            "reason_code": signal.get("reason_code"),
            "reason": signal.get("reason"),
            "direction": signal.get("direction"),
            "supporting_prob": _round_optional(signal.get("supporting_prob")),
            "entry_price": _round_optional(signal.get("entry_price")),
            "entry_price_source": signal.get("entry_price_source"),
            "btc_move_usd": _round_optional(signal.get("btc_move_usd")),
        },
        "result": {
            "status": result.get("status"),
            "mode": result.get("mode"),
            "reason_code": result.get("reason_code"),
            "reason": result.get("reason"),
            "orders": [
                {
                    "status": order.get("status"),
                    "direction": order.get("direction"),
                    "amount": _round_optional(order.get("amount"), 4),
                    "price": _round_optional(order.get("price"), 6),
                    "shares": _round_optional(order.get("shares"), 4),
                    "order_style": order.get("order_style"),
                }
                for order in (result.get("orders") or [])
                if isinstance(order, dict)
            ],
        },
    }


def _evaluate_cheap_side_signal(
    *,
    market: Btc5mMarket,
    price_snapshot: BtcPriceSnapshot,
    up_book: OrderBookSummary,
    down_book: OrderBookSummary,
    profile: Btc5mProfile,
    seconds_left: float,
) -> dict:
    books = {"up": up_book, "down": down_book}
    candidates = [
        (direction, book)
        for direction, book in books.items()
        if book.best_ask is not None and book.best_ask > 0
    ]
    if not candidates:
        return _skip("no_ask", "No ask available for either BTC 5m token.")

    direction, book = min(candidates, key=lambda item: item[1].best_ask or 1.0)
    entry_price, entry_price_source = _entry_price_for_book(profile, book)
    if entry_price is None:
        return _skip(
            "no_entry_price",
            f"No {entry_price_source.replace('_', ' ')} available for {direction.upper()} token.",
            direction=direction,
        )
    if entry_price <= 0:
        return _skip(
            "invalid_entry_price",
            f"Invalid {direction.upper()} entry price ${entry_price:.4f}.",
            direction=direction,
            entry_price=entry_price,
        )

    price_skip = _entry_price_range_skip(
        profile=profile,
        direction=direction,
        entry_price=entry_price,
        entry_price_source=entry_price_source,
    )
    if price_skip is not None:
        price_skip["reason_code"] = "cheap_side_out_of_range"
        return price_skip

    liquidity_skip = _book_liquidity_skip(
        profile=profile,
        direction=direction,
        book=book,
        entry_price=entry_price,
    )
    if liquidity_skip is not None:
        return liquidity_skip

    supporting_prob = book.midpoint if book.midpoint is not None else _price_for_direction(direction, market)
    return {
        "action": "trade",
        "reason": (
            f"{direction.upper()} cheap-side entry: {entry_price_source.replace('_', ' ')} "
            f"${entry_price:.4f} within ${profile.min_entry_price:.2f}-${profile.max_entry_price:.2f}."
        ),
        "direction": direction,
        "seconds_left": seconds_left,
        "btc_move_usd": price_snapshot.move_usd,
        "supporting_prob": supporting_prob,
        "entry_price": entry_price,
        "entry_price_source": entry_price_source,
        "best_ask": book.best_ask,
        "best_bid": book.best_bid,
        "spread": book.spread,
        "strategy_style": _strategy_style(profile),
    }


def _evaluate_probability_capture_signal(
    *,
    market: Btc5mMarket,
    price_snapshot: BtcPriceSnapshot,
    up_book: OrderBookSummary,
    down_book: OrderBookSummary,
    profile: Btc5mProfile,
    seconds_left: float,
) -> dict:
    candidates: list[tuple[str, OrderBookSummary, float]] = []
    for direction, book in (("up", up_book), ("down", down_book)):
        supporting_prob = _book_supporting_prob(direction, market, book)
        if supporting_prob is not None and book.best_ask is not None:
            candidates.append((direction, book, supporting_prob))
    if not candidates:
        return _skip("missing_market_price", "Could not resolve market probability for either BTC 5m token.")

    direction, book, supporting_prob = max(candidates, key=lambda item: item[2])
    if supporting_prob < profile.min_supporting_prob:
        return _skip(
            "skew_not_supportive",
            (
                f"Market support {supporting_prob:.1%} is below probability-capture threshold "
                f"{profile.min_supporting_prob:.1%}."
            ),
            direction=direction,
            supporting_prob=supporting_prob,
            btc_move_usd=price_snapshot.move_usd,
        )

    entry_price, entry_price_source = _entry_price_for_book(profile, book)
    if entry_price is None:
        return _skip(
            "no_entry_price",
            f"No {entry_price_source.replace('_', ' ')} available for {direction.upper()} token.",
            direction=direction,
        )
    if entry_price <= 0:
        return _skip(
            "invalid_entry_price",
            f"Invalid {direction.upper()} entry price ${entry_price:.4f}.",
            direction=direction,
            entry_price=entry_price,
        )

    price_skip = _entry_price_range_skip(
        profile=profile,
        direction=direction,
        entry_price=entry_price,
        entry_price_source=entry_price_source,
    )
    if price_skip is not None:
        return price_skip

    support_gap_skip = _entry_support_gap_skip(
        profile=profile,
        direction=direction,
        supporting_prob=supporting_prob,
        entry_price=entry_price,
    )
    if support_gap_skip is not None:
        return support_gap_skip

    liquidity_skip = _book_liquidity_skip(
        profile=profile,
        direction=direction,
        book=book,
        entry_price=entry_price,
    )
    if liquidity_skip is not None:
        return liquidity_skip

    return {
        "action": "trade",
        "reason": (
            f"{direction.upper()} probability capture: market support {supporting_prob:.1%}; "
            f"{entry_price_source.replace('_', ' ')} ${entry_price:.4f}; "
            f"{seconds_left:.0f}s left."
        ),
        "direction": direction,
        "seconds_left": seconds_left,
        "btc_move_usd": price_snapshot.move_usd,
        "supporting_prob": supporting_prob,
        "entry_support_gap": entry_price - supporting_prob,
        "entry_price": entry_price,
        "entry_price_source": entry_price_source,
        "best_ask": book.best_ask,
        "best_bid": book.best_bid,
        "spread": book.spread,
        "strategy_style": _strategy_style(profile),
    }


def evaluate_signal(
    *,
    market: Btc5mMarket,
    price_snapshot: BtcPriceSnapshot,
    up_book: OrderBookSummary,
    down_book: OrderBookSummary,
    profile: Btc5mProfile,
    now: datetime | None = None,
    ignore_entry_window: bool = False,
) -> dict:
    current = _coerce_utc(now)
    seconds_left = (market.window_end - current).total_seconds()
    if seconds_left <= 0:
        return _skip("market_closed", "BTC 5m market window is already closed.")
    if not market.active or market.closed or not market.accepting_orders:
        return _skip("market_not_tradeable", "BTC 5m market is not accepting orders.")

    if not ignore_entry_window:
        entry_skip = _entry_window_skip(profile, seconds_left)
        if entry_skip is not None:
            return entry_skip

    hours_skip = _trading_hours_skip(profile, current)
    if hours_skip is not None:
        return hours_skip

    if _strategy_style(profile) == "cheap_side":
        return _evaluate_cheap_side_signal(
            market=market,
            price_snapshot=price_snapshot,
            up_book=up_book,
            down_book=down_book,
            profile=profile,
            seconds_left=seconds_left,
        )
    if _strategy_style(profile) == "probability_capture":
        return _evaluate_probability_capture_signal(
            market=market,
            price_snapshot=price_snapshot,
            up_book=up_book,
            down_book=down_book,
            profile=profile,
            seconds_left=seconds_left,
        )

    move = price_snapshot.move_usd
    if abs(move) < profile.min_move_usd:
        return _skip(
            "insufficient_impulse",
            f"BTC move ${move:+.2f} is below ${profile.min_move_usd:.2f} impulse threshold.",
            btc_move_usd=move,
        )
    if abs(move) <= 1e-9:
        return _skip("flat_impulse", "BTC move is flat; no momentum side selected.")

    direction = "up" if move > 0 else "down"
    book = _book_for_direction(direction, up_book=up_book, down_book=down_book)
    market_prob = _price_for_direction(direction, market)
    supporting_prob = book.midpoint if book.midpoint is not None else market_prob
    if supporting_prob is None:
        return _skip("missing_market_price", "Could not resolve market probability for momentum side.")
    if supporting_prob < profile.min_supporting_prob:
        return _skip(
            "skew_not_supportive",
            (
                f"Market skew {supporting_prob:.1%} does not support {direction.upper()} "
                f"momentum enough; need {profile.min_supporting_prob:.1%}."
            ),
            direction=direction,
            supporting_prob=supporting_prob,
            btc_move_usd=move,
        )
    if book.best_ask is None:
        return _skip("no_ask", f"No ask available for {direction.upper()} token.")
    entry_price, entry_price_source = _entry_price_for_book(profile, book)
    if entry_price is None:
        return _skip(
            "no_entry_price",
            f"No {entry_price_source.replace('_', ' ')} available for {direction.upper()} token.",
            direction=direction,
        )
    if entry_price <= 0:
        return _skip(
            "invalid_entry_price",
            f"Invalid {direction.upper()} entry price ${entry_price:.4f}.",
            direction=direction,
            entry_price=entry_price,
        )
    price_skip = _entry_price_range_skip(
        profile=profile,
        direction=direction,
        entry_price=entry_price,
        entry_price_source=entry_price_source,
    )
    if price_skip is not None:
        return price_skip

    liquidity_skip = _book_liquidity_skip(
        profile=profile,
        direction=direction,
        book=book,
        entry_price=entry_price,
    )
    if liquidity_skip is not None:
        return liquidity_skip

    return {
        "action": "trade",
        "reason": (
            f"{direction.upper()} momentum: BTC moved ${move:+.2f}; "
            f"market support {supporting_prob:.1%}; ask ${book.best_ask:.4f}."
        ),
        "direction": direction,
        "seconds_left": seconds_left,
        "btc_move_usd": move,
        "supporting_prob": supporting_prob,
        "entry_price": entry_price,
        "entry_price_source": entry_price_source,
        "best_ask": book.best_ask,
        "best_bid": book.best_bid,
        "spread": book.spread,
        "strategy_style": _strategy_style(profile),
    }


def _skip(reason_code: str, reason: str, **extra) -> dict:
    payload = {"action": "skip", "reason_code": reason_code, "reason": reason}
    payload.update(extra)
    return payload


def _parse_placed_at(raw) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return _coerce_utc(parsed)


def _ledger_risk_snapshot(ledger: BetLedger, *, now: datetime | None = None) -> dict:
    current = _coerce_utc(now)
    today = current.date()
    trade_count_today = 0
    open_exposure = 0.0
    realized_loss_today = 0.0

    for bet in ledger.get_bets(fresh=True):
        if str(bet.get("order_type", "") or "") != BTC5M_ORDER_TYPE:
            continue
        placed_at = _parse_placed_at(bet.get("placed_at"))
        if placed_at is not None and placed_at.date() == today and bet.get("status") != "cancelled":
            trade_count_today += 1
            pnl = bet.get("result_pnl")
            if pnl is not None:
                realized_loss_today += min(_safe_float(pnl), 0.0)
        if bet.get("status") == "open":
            open_exposure += max(_safe_float(bet.get("amount")), 0.0)

    return {
        "trade_count_today": trade_count_today,
        "open_exposure": round(open_exposure, 2),
        "realized_loss_today": round(abs(realized_loss_today), 2),
    }


def _duplicate_market_bet(ledger: BetLedger, market: Btc5mMarket, *, dry_run: bool) -> dict | None:
    for bet in ledger.get_bets(fresh=True):
        if str(bet.get("order_type", "") or "") != BTC5M_ORDER_TYPE:
            continue
        if str(bet.get("market_id", "") or "") != market.market_id:
            continue
        if bet.get("status") != "open":
            continue
        if bet.get("dry_run") and not dry_run:
            continue
        return dict(bet)
    return None


def _bet_window_start(bet: dict) -> datetime | None:
    for field in ("window_start", "market_event_date"):
        parsed = _parse_placed_at(bet.get(field))
        if parsed is not None:
            return parsed
    slug_epoch = _extract_slug_epoch(str(bet.get("market_slug", "") or ""))
    if slug_epoch is not None:
        return datetime.fromtimestamp(slug_epoch, tz=timezone.utc)
    return None


def _cooldown_market_bet(
    ledger: BetLedger,
    market: Btc5mMarket,
    profile: Btc5mProfile,
    *,
    dry_run: bool,
) -> dict | None:
    cooldown = max(int(profile.cooldown_windows_after_trade), 0)
    if cooldown <= 0:
        return None

    latest: tuple[datetime, dict, int] | None = None
    for bet in ledger.get_bets(fresh=True):
        if str(bet.get("order_type", "") or "") != BTC5M_ORDER_TYPE:
            continue
        if bet.get("status") == "cancelled":
            continue
        if bet.get("dry_run") and not dry_run:
            continue
        window_start = _bet_window_start(bet)
        if window_start is None:
            continue
        windows_since = int((market.window_start - window_start).total_seconds() // BTC5M_WINDOW_SECONDS)
        if windows_since <= 0 or windows_since > cooldown:
            continue
        if latest is None or window_start > latest[0]:
            latest = (window_start, dict(bet), windows_since)

    if latest is None:
        return None
    _, bet, windows_since = latest
    bet["cooldown_windows_since"] = windows_since
    return bet


def _risk_check(
    *,
    ledger: BetLedger,
    market: Btc5mMarket,
    profile: Btc5mProfile,
    dry_run: bool,
    now: datetime | None = None,
) -> dict:
    duplicate = _duplicate_market_bet(ledger, market, dry_run=dry_run)
    if duplicate is not None:
        return _skip(
            "duplicate_market",
            f"Already have an open BTC 5m ledger entry for market {market.slug}.",
            ledger_bet_id=duplicate.get("id"),
        )

    snapshot = _ledger_risk_snapshot(ledger, now=now)
    cooldown_bet = _cooldown_market_bet(ledger, market, profile, dry_run=dry_run)
    if cooldown_bet is not None:
        return _skip(
            "cooldown_after_trade",
            (
                f"BTC 5m cooldown active after ledger bet #{cooldown_bet.get('id')}; "
                f"skip {profile.cooldown_windows_after_trade} full window(s) after a trade."
            ),
            ledger_bet_id=cooldown_bet.get("id"),
            windows_since=cooldown_bet.get("cooldown_windows_since"),
            risk=snapshot,
        )
    if profile.max_trades_per_day and snapshot["trade_count_today"] >= profile.max_trades_per_day:
        return _skip(
            "daily_trade_limit",
            (
                f"BTC 5m daily trade limit reached: {snapshot['trade_count_today']} "
                f">= {profile.max_trades_per_day}."
            ),
            risk=snapshot,
        )
    if profile.daily_loss_limit_usd and snapshot["realized_loss_today"] >= profile.daily_loss_limit_usd:
        return _skip(
            "daily_loss_limit",
            (
                f"BTC 5m daily loss limit reached: ${snapshot['realized_loss_today']:.2f} "
                f">= ${profile.daily_loss_limit_usd:.2f}."
            ),
            risk=snapshot,
        )
    if profile.max_open_exposure_usd and snapshot["open_exposure"] >= profile.max_open_exposure_usd:
        return _skip(
            "open_exposure_limit",
            (
                f"BTC 5m open exposure limit reached: ${snapshot['open_exposure']:.2f} "
                f">= ${profile.max_open_exposure_usd:.2f}."
            ),
            risk=snapshot,
        )
    return {"action": "ok", "risk": snapshot}


def _available_cash(
    *,
    dry_run: bool,
    clob_client: ClobClientWrapper | None,
) -> float:
    if dry_run:
        return float(INITIAL_BANKROLL)
    if clob_client is None:
        return 0.0
    details = clob_client.get_cash_balance_details()
    return _safe_float(details.get("balance"), 0.0)


def _sized_notional(
    *,
    profile: Btc5mProfile,
    available_cash: float,
    open_exposure: float,
    price: float,
    min_order_size: float,
) -> tuple[float, str]:
    if available_cash <= 0:
        return 0.0, "available cash is zero"

    allocation_cap = available_cash * profile.allocation_fraction
    exposure_remaining = max(profile.max_open_exposure_usd - open_exposure, 0.0)
    amount = min(
        profile.trade_notional_usd,
        profile.max_notional_per_trade,
        allocation_cap,
        exposure_remaining,
        available_cash,
    )
    min_notional = max(POLYMARKET_MIN_BUY_ORDER_USD, min_order_size * price)
    if amount < min_notional:
        return 0.0, (
            f"sized notional ${amount:.2f} is below minimum ${min_notional:.2f} "
            f"({min_order_size:g} shares @ ${price:.4f})"
        )
    return round(amount, 2), ""


def _shares_for_amount(amount: float, price: float, min_order_size: float) -> tuple[float, float]:
    if price <= 0:
        return 0.0, amount
    shares = math.floor((amount / price) * 100.0) / 100.0
    shares = max(shares, float(min_order_size))
    adjusted_amount = round(shares * price, 2)
    if adjusted_amount < POLYMARKET_MIN_BUY_ORDER_USD:
        min_shares = math.ceil((POLYMARKET_MIN_BUY_ORDER_USD / price) * 100.0) / 100.0
        shares = max(shares, min_shares)
        adjusted_amount = round(shares * price, 2)
    return round(shares, 2), adjusted_amount


def _extract_order_id(response) -> Optional[str]:
    if not isinstance(response, dict):
        return None
    order_id = response.get("orderID") or response.get("orderId") or response.get("id")
    if order_id:
        return str(order_id)
    order_ids = response.get("orderIDs")
    if isinstance(order_ids, list) and order_ids:
        return str(order_ids[0])
    return None


def _response_float(response: dict, *keys: str) -> float:
    for key in keys:
        try:
            value = float(response.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return 0.0


def _limit_buy_fill_fields(response) -> dict:
    if not isinstance(response, dict):
        return {}

    filled_shares = _response_float(response, "takingAmount", "sizeMatched", "size_matched")
    fill_amount = _response_float(response, "makingAmount", "amountMatched", "amount_matched")
    if filled_shares <= 0.0 or fill_amount <= 0.0:
        return {}

    tx_hashes = response.get("transactionsHashes") or response.get("transactionHashes") or []
    if not isinstance(tx_hashes, list):
        tx_hashes = [str(tx_hashes)] if tx_hashes else []

    return {
        "actual_fill_price": round(fill_amount / filled_shares, 4),
        "actual_fill_amount": round(fill_amount, 6),
        "actual_filled_shares": round(filled_shares, 6),
        "actual_fill_source": "clob_order_response",
        "actual_fill_status": response.get("status"),
        "actual_fill_tx_hash": str(tx_hashes[0]) if tx_hashes else None,
        "actual_fill_tx_hashes": [str(value) for value in tx_hashes],
    }


def _order_metadata(
    *,
    market: Btc5mMarket,
    price_snapshot: BtcPriceSnapshot,
    signal: dict,
    profile: Btc5mProfile,
    supporting_prob: float | None,
) -> dict:
    return {
        "strategy": BTC5M_STRATEGY_NAME,
        "market_slug": market.slug,
        "window_start": market.window_start.isoformat(),
        "window_end": market.window_end.isoformat(),
        "btc_price_source": price_snapshot.source,
        "btc_symbol": price_snapshot.symbol,
        "btc_window_start_price": round(price_snapshot.window_start_price, 2),
        "btc_window_start_price_full": price_snapshot.window_start_price,
        "btc_current_price": round(price_snapshot.current_price, 2),
        "btc_current_price_full": price_snapshot.current_price,
        "btc_move_usd": round(price_snapshot.move_usd, 2),
        "seconds_left": round(_safe_float(signal.get("seconds_left")), 2),
        "supporting_prob": round(supporting_prob, 4) if supporting_prob is not None else None,
        "profile": profile.name,
        "strategy_style": _strategy_style(profile),
        "cooldown_windows_after_trade": profile.cooldown_windows_after_trade,
        "fee_schedule": market.fee_schedule,
        "signal_source": BTC5M_STRATEGY_NAME,
        "probability_source": "market_skew",
    }


def _direction_fields(direction: str, market: Btc5mMarket) -> tuple[str, str, str]:
    if direction == "up":
        return market.token_up, "BTC 5m Up", "BTC 5m Down"
    return market.token_down, "BTC 5m Down", "BTC 5m Up"


def _place_order(
    *,
    direction: str,
    amount: float,
    market: Btc5mMarket,
    book: OrderBookSummary,
    ledger: BetLedger,
    dry_run: bool,
    clob_client: ClobClientWrapper | None,
    price_snapshot: BtcPriceSnapshot,
    signal: dict,
    profile: Btc5mProfile,
    reason: str,
) -> dict:
    entry_price, entry_price_source = _entry_price_for_book(profile, book)
    if entry_price is None:
        return {"status": "skipped", "reason": f"No {entry_price_source} for {direction}"}

    token_id, label, opposite_label = _direction_fields(direction, market)
    shares, adjusted_amount = _shares_for_amount(
        amount,
        entry_price,
        max(market.order_min_size, book.min_order_size),
    )
    decimal_odds = round(1.0 / entry_price, 4) if entry_price > 0 else 0.0
    supporting_prob = book.midpoint if book.midpoint is not None else _price_for_direction(direction, market)
    metadata = _order_metadata(
        market=market,
        price_snapshot=price_snapshot,
        signal=signal,
        profile=profile,
        supporting_prob=supporting_prob,
    )

    common = {
        "fighter": label,
        "opponent": opposite_label,
        "side": direction,
        "amount": adjusted_amount,
        "price": entry_price,
        "shares": shares,
        "token_id": token_id,
        "market_id": market.market_id,
        "condition_id": market.condition_id,
        "model_prob": 0.0,
        "market_prob": supporting_prob or 0.0,
        "edge": 0.0,
        "decimal_odds": decimal_odds,
        "event_date": market.window_end.isoformat(),
        "market_event_date": market.window_start.isoformat(),
        "order_type": BTC5M_ORDER_TYPE,
        "reason": reason,
        "metadata": {
            **metadata,
            "order_style": _order_style(profile),
            "entry_price_source": entry_price_source,
            "submitted_entry_price": entry_price,
            "submitted_amount": adjusted_amount,
            "submitted_shares": shares,
        },
    }

    if dry_run:
        bet = ledger.add_bet(
            **common,
            dry_run=True,
            order_id=None,
            placement_state="dry_run",
        )
        return {
            "status": "dry_run",
            "ledger_bet_id": bet["id"],
            "direction": direction,
            "amount": adjusted_amount,
            "price": entry_price,
            "shares": shares,
            "order_style": _order_style(profile),
        }

    if clob_client is None:
        return {"status": "failed", "error": "CLOB client unavailable"}

    pending_bet = ledger.add_bet(
        **common,
        dry_run=False,
        order_id=None,
        placement_state="pending_submit",
    )
    order_info = {
        "status": "unknown",
        "ledger_bet_id": pending_bet["id"],
        "direction": direction,
        "amount": adjusted_amount,
        "price": entry_price,
        "shares": shares,
        "order_style": _order_style(profile),
    }
    try:
        response = clob_client.create_limit_order(
            token_id=token_id,
            side="BUY",
            price=entry_price,
            size=shares,
            tick_size=market.tick_size or book.tick_size,
            neg_risk=market.neg_risk or book.neg_risk,
        )
        order_id = _extract_order_id(response)
        order_info["response"] = response
        fill_fields = _limit_buy_fill_fields(response)
        if fill_fields:
            order_info.update(fill_fields)
        if order_id:
            ledger.update_bet_fields(
                int(pending_bet["id"]),
                placement_state="submitted",
                submission_error=None,
                order_id=order_id,
                **fill_fields,
            )
            order_info["status"] = "placed"
            order_info["order_id"] = order_id
        else:
            ledger.update_bet_fields(
                int(pending_bet["id"]),
                placement_state="unknown",
                submission_error="CLOB response missing durable order id",
            )
            order_info["error"] = "CLOB response missing durable order id"
    except Exception as exc:
        if is_uncertain_clob_order_submission_error(exc):
            ledger.update_bet_fields(
                int(pending_bet["id"]),
                placement_state="unknown",
                submission_error=f"order submission unresolved: {exc}",
            )
            order_info["status"] = "unknown"
            order_info["error"] = str(exc)
            logger.error(
                "BTC 5m order outcome is unknown for %s on %s: %s. "
                "Leaving ledger entry open to block automatic duplicate submission.",
                label,
                market.slug,
                exc,
            )
        else:
            ledger.cancel_bet(int(pending_bet["id"]), reason=f"submit_failed: {exc}")
            ledger.update_bet_fields(
                int(pending_bet["id"]),
                require_open=False,
                placement_state="failed",
                submission_error=str(exc),
            )
            order_info["status"] = "failed"
            order_info["error"] = str(exc)
    return order_info


class Btc5mRunner:
    def __init__(
        self,
        *,
        profile: Btc5mProfile | None = None,
        ledger: BetLedger | None = None,
        ledger_path: Path = BTC5M_LEDGER_PATH,
        market_client: Btc5mMarketClient | None = None,
        price_client: BinanceBtcPriceClient | CoinbaseBtcPriceClient | None = None,
        book_client: PublicClobOrderBookClient | None = None,
        clob_client: ClobClientWrapper | None = None,
        signal_log_path: Path = BTC5M_SIGNAL_LOG_PATH,
        record_signal_snapshots: bool = BTC5M_SIGNAL_LOG_ENABLED,
        exit_shadow_enabled: bool = BTC5M_EXIT_SHADOW_ENABLED,
        exit_shadow_log_path: Path = BTC5M_EXIT_SHADOW_LOG_PATH,
    ):
        self.profile = profile or resolve_btc5m_profile()
        self.ledger = ledger or BetLedger(path=ledger_path)
        self.ledger_path = ledger_path
        self.market_client = market_client or Btc5mMarketClient()
        self.price_client = price_client or build_btc_price_client()
        self.book_client = book_client or PublicClobOrderBookClient()
        self.clob_client = clob_client
        self.signal_log_path = signal_log_path
        self.record_signal_snapshots = bool(record_signal_snapshots)
        self.exit_shadow_enabled = bool(exit_shadow_enabled)
        self.exit_shadow_log_path = Path(exit_shadow_log_path)

    def _record_signal_snapshot(
        self,
        *,
        market: Btc5mMarket,
        price_snapshot: BtcPriceSnapshot,
        up_book: OrderBookSummary,
        down_book: OrderBookSummary,
        signal: dict,
        result: dict,
        risk: dict | None,
        now: datetime,
    ) -> None:
        if not self.record_signal_snapshots:
            return
        try:
            snapshot = _build_signal_snapshot(
                market=market,
                price_snapshot=price_snapshot,
                up_book=up_book,
                down_book=down_book,
                profile=self.profile,
                signal=signal,
                result=result,
                risk=risk,
                now=now,
            )
            self.signal_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.signal_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
        except Exception as exc:
            logger.warning("BTC 5m signal snapshot write failed: %s", exc)

    def _record_exit_shadow_decisions(
        self,
        *,
        market: Btc5mMarket,
        price_snapshot: BtcPriceSnapshot,
        up_book: OrderBookSummary,
        down_book: OrderBookSummary,
        now: datetime,
    ) -> int:
        if not self.exit_shadow_enabled:
            return 0

        rows: list[dict] = []
        seconds_left = (market.window_end - now).total_seconds()
        for bet in self.ledger.get_open_bets(fresh=True):
            if str(bet.get("order_type", "") or "") != BTC5M_ORDER_TYPE:
                continue
            if str(bet.get("market_slug", "") or "") != market.slug:
                continue
            side = str(bet.get("side", "") or "").strip().lower()
            if side not in {"up", "down"}:
                continue
            held_book = _book_for_direction(side, up_book=up_book, down_book=down_book)
            state_key = f"{self.ledger_path.resolve()}:{bet.get('id')}"
            context = Btc5mExitContext(
                bet=dict(bet),
                profile=self.profile,
                profile_name=self.profile.name,
                strategy_style=_strategy_style(self.profile),
                side=side,
                entry_price=_safe_float(bet.get("price")),
                shares=_safe_float(bet.get("shares")),
                amount=_safe_float(bet.get("amount")),
                btc_start_price=price_snapshot.window_start_price,
                btc_current_price=price_snapshot.current_price,
                seconds_left=seconds_left,
                held_book=held_book,
                fee_schedule=bet.get("fee_schedule") or market.fee_schedule,
                recorded_at=now,
                shadow_state=get_exit_shadow_state(state_key),
            )
            decision = evaluate_exit_policy(context)
            rows.append(
                {
                    "schema_version": 1,
                    "recorded_at": now.isoformat(),
                    "strategy": BTC5M_STRATEGY_NAME,
                    "profile": self.profile.name,
                    "strategy_style": _strategy_style(self.profile),
                    "ledger_bet_id": bet.get("id"),
                    "market": {
                        "slug": market.slug,
                        "market_id": market.market_id,
                        "condition_id": market.condition_id,
                        "window_start": market.window_start.isoformat(),
                        "window_end": market.window_end.isoformat(),
                        "seconds_left": round(seconds_left, 3),
                    },
                    "position": {
                        "side": side,
                        "entry_price": _round_optional(bet.get("price")),
                        "shares": _round_optional(bet.get("shares"), 4),
                        "amount": _round_optional(bet.get("amount"), 4),
                        "entry_btc_move_usd": _round_optional(bet.get("btc_move_usd"), 4),
                    },
                    "price": {
                        "source": price_snapshot.source,
                        "symbol": price_snapshot.symbol,
                        "price_to_beat": round(price_snapshot.window_start_price, 4),
                        "current_price": round(price_snapshot.current_price, 4),
                        "distance_to_price_to_beat_usd": round(price_snapshot.move_usd, 4),
                    },
                    "held_book": _book_snapshot(held_book),
                    "decision": decision.to_dict(),
                }
            )

        if not rows:
            return 0
        try:
            self.exit_shadow_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.exit_shadow_log_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception as exc:
            logger.warning("BTC 5m exit shadow write failed: %s", exc)
            return 0
        return len(rows)

    def run_once(
        self,
        *,
        dry_run: bool = True,
        market_slug: str | None = None,
        now: datetime | None = None,
    ) -> dict:
        current = _coerce_utc(now)
        if not dry_run:
            assert_polymarket_real_trading_allowed(
                host="127.0.0.1",
                startup_source="btc5m",
                ledger_paths=(self.ledger_path,),
            )
            if self.clob_client is None:
                self.clob_client = ClobClientWrapper()

        market = self.market_client.get_market(now=current, market_slug=market_slug)
        if market is None:
            return {"status": "idle", "reason": "no_btc5m_market_found"}

        if current < market.window_start:
            return {
                "status": "idle",
                "reason": "BTC 5m market window has not started yet.",
                "reason_code": "market_not_started",
                "market_slug": market.slug,
            }
        if current >= market.window_end:
            return {
                "status": "idle",
                "reason": "BTC 5m market window is already closed.",
                "reason_code": "market_closed",
                "market_slug": market.slug,
            }
        if not market.active or market.closed or not market.accepting_orders:
            return {
                "status": "idle",
                "reason": "BTC 5m market is not accepting orders.",
                "reason_code": "market_not_tradeable",
                "market_slug": market.slug,
            }
        seconds_left = (market.window_end - current).total_seconds()
        price_snapshot = None
        up_book = None
        down_book = None
        if self.exit_shadow_enabled:
            try:
                price_snapshot = self.price_client.get_snapshot(
                    window_start=market.window_start,
                    now=current,
                )
                up_book = self.book_client.summarize(market.token_up)
                down_book = self.book_client.summarize(market.token_down)
                self._record_exit_shadow_decisions(
                    market=market,
                    price_snapshot=price_snapshot,
                    up_book=up_book,
                    down_book=down_book,
                    now=current,
                )
            except Exception as exc:
                logger.warning("BTC 5m exit shadow data fetch failed: %s", exc)

        entry_skip = _entry_window_skip(self.profile, seconds_left)
        if entry_skip is not None:
            return {
                "status": "idle",
                "reason": entry_skip["reason"],
                "reason_code": entry_skip.get("reason_code"),
                "market_slug": market.slug,
            }
        hours_skip = _trading_hours_skip(self.profile, current)
        if hours_skip is not None:
            return {
                "status": "idle",
                "reason": hours_skip["reason"],
                "reason_code": hours_skip.get("reason_code"),
                "market_slug": market.slug,
            }

        if price_snapshot is None or up_book is None or down_book is None:
            try:
                price_snapshot = self.price_client.get_snapshot(
                    window_start=market.window_start,
                    now=current,
                )
                up_book = self.book_client.summarize(market.token_up)
                down_book = self.book_client.summarize(market.token_down)
            except Exception as exc:
                logger.warning("BTC 5m data fetch failed: %s", exc)
                return {
                    "status": "error",
                    "reason": f"data_fetch_failed: {exc}",
                    "market_slug": market.slug,
                }

        snapshot_risk = _ledger_risk_snapshot(self.ledger, now=current) if self.record_signal_snapshots else None
        signal = evaluate_signal(
            market=market,
            price_snapshot=price_snapshot,
            up_book=up_book,
            down_book=down_book,
            profile=self.profile,
            now=current,
        )
        if signal["action"] != "trade":
            result = {
                "status": "idle",
                "reason": signal["reason"],
                "reason_code": signal.get("reason_code"),
                "market_slug": market.slug,
                "signal": signal,
            }
            self._record_signal_snapshot(
                market=market,
                price_snapshot=price_snapshot,
                up_book=up_book,
                down_book=down_book,
                signal=signal,
                result=result,
                risk=snapshot_risk,
                now=current,
            )
            return result

        risk = _risk_check(
            ledger=self.ledger,
            market=market,
            profile=self.profile,
            dry_run=dry_run,
            now=current,
        )
        if risk["action"] != "ok":
            result = {
                "status": "idle",
                "reason": risk["reason"],
                "reason_code": risk.get("reason_code"),
                "market_slug": market.slug,
                "signal": signal,
                "risk": risk.get("risk"),
            }
            self._record_signal_snapshot(
                market=market,
                price_snapshot=price_snapshot,
                up_book=up_book,
                down_book=down_book,
                signal=signal,
                result=result,
                risk=risk.get("risk") or snapshot_risk,
                now=current,
            )
            return result

        direction = str(signal["direction"])
        chosen_book = _book_for_direction(direction, up_book=up_book, down_book=down_book)
        entry_price, entry_price_source = _entry_price_for_book(self.profile, chosen_book)
        if entry_price is None:
            result = {
                "status": "idle",
                "reason": f"No {entry_price_source.replace('_', ' ')} available for sizing.",
                "reason_code": "sizing_entry_price",
                "market_slug": market.slug,
                "signal": signal,
                "risk": risk["risk"],
            }
            self._record_signal_snapshot(
                market=market,
                price_snapshot=price_snapshot,
                up_book=up_book,
                down_book=down_book,
                signal=signal,
                result=result,
                risk=risk["risk"],
                now=current,
            )
            return result
        available_cash = _available_cash(dry_run=dry_run, clob_client=self.clob_client)
        amount, size_reason = _sized_notional(
            profile=self.profile,
            available_cash=available_cash,
            open_exposure=_safe_float(risk["risk"].get("open_exposure")),
            price=entry_price,
            min_order_size=max(market.order_min_size, chosen_book.min_order_size),
        )
        if amount <= 0:
            result = {
                "status": "idle",
                "reason": size_reason,
                "reason_code": "sizing",
                "market_slug": market.slug,
                "signal": signal,
                "risk": risk["risk"],
            }
            self._record_signal_snapshot(
                market=market,
                price_snapshot=price_snapshot,
                up_book=up_book,
                down_book=down_book,
                signal=signal,
                result=result,
                risk=risk["risk"],
                now=current,
            )
            return result

        primary_order = _place_order(
            direction=direction,
            amount=amount,
            market=market,
            book=chosen_book,
            ledger=self.ledger,
            dry_run=dry_run,
            clob_client=self.clob_client,
            price_snapshot=price_snapshot,
            signal=signal,
            profile=self.profile,
            reason=signal["reason"],
        )
        orders = [primary_order]

        if primary_order.get("status") in {"dry_run", "placed"}:
            hedge_order = self._maybe_place_hedge(
                direction=direction,
                market=market,
                up_book=up_book,
                down_book=down_book,
                ledger=self.ledger,
                dry_run=dry_run,
                price_snapshot=price_snapshot,
                signal=signal,
            )
            if hedge_order is not None:
                orders.append(hedge_order)

        placed = [order for order in orders if order.get("status") in {"dry_run", "placed"}]
        status = "ok" if placed else "error"
        result = {
            "status": status,
            "mode": "dry_run" if dry_run else "real",
            "market_slug": market.slug,
            "market_question": market.question,
            "profile": self.profile.name,
            "signal": signal,
            "price_snapshot": {
                "source": price_snapshot.source,
                "symbol": price_snapshot.symbol,
                "window_start_price": price_snapshot.window_start_price,
                "current_price": price_snapshot.current_price,
                "move_usd": price_snapshot.move_usd,
            },
            "risk": risk["risk"],
            "orders": orders,
        }
        self._record_signal_snapshot(
            market=market,
            price_snapshot=price_snapshot,
            up_book=up_book,
            down_book=down_book,
            signal=signal,
            result=result,
            risk=risk["risk"],
            now=current,
        )
        return result

    def _maybe_place_hedge(
        self,
        *,
        direction: str,
        market: Btc5mMarket,
        up_book: OrderBookSummary,
        down_book: OrderBookSummary,
        ledger: BetLedger,
        dry_run: bool,
        price_snapshot: BtcPriceSnapshot,
        signal: dict,
    ) -> dict | None:
        if not self.profile.enable_extreme_skew_hedge:
            return None

        chosen_book = _book_for_direction(direction, up_book=up_book, down_book=down_book)
        if chosen_book.best_ask is None or chosen_book.best_ask < self.profile.hedge_skew_threshold:
            return None

        hedge_direction = "down" if direction == "up" else "up"
        hedge_book = _book_for_direction(hedge_direction, up_book=up_book, down_book=down_book)
        if hedge_book.best_ask is None or hedge_book.best_ask > self.profile.hedge_max_price:
            return None

        min_notional = max(
            POLYMARKET_MIN_BUY_ORDER_USD,
            max(market.order_min_size, hedge_book.min_order_size) * hedge_book.best_ask,
        )
        hedge_amount = max(self.profile.hedge_notional_usd, min_notional)
        return _place_order(
            direction=hedge_direction,
            amount=hedge_amount,
            market=market,
            book=hedge_book,
            ledger=ledger,
            dry_run=dry_run,
            clob_client=self.clob_client,
            price_snapshot=price_snapshot,
            signal=signal,
            profile=self.profile,
            reason=f"Extreme-skew hedge against {direction.upper()} entry.",
        )


def run_btc5m_once(
    *,
    profile_name: str | None = None,
    dry_run: bool = True,
    market_slug: str | None = None,
    ledger_path: Path = BTC5M_LEDGER_PATH,
) -> dict:
    runner = Btc5mRunner(
        profile=resolve_btc5m_profile(profile_name),
        ledger_path=ledger_path,
    )
    return runner.run_once(dry_run=dry_run, market_slug=market_slug)


def run_btc5m_loop(
    *,
    profile_name: str | None = None,
    dry_run: bool = True,
    market_slug: str | None = None,
    poll_seconds: float = BTC5M_POLL_SECONDS,
    max_cycles: int | None = None,
    ledger_path: Path = BTC5M_LEDGER_PATH,
) -> dict:
    runner = Btc5mRunner(
        profile=resolve_btc5m_profile(profile_name),
        ledger_path=ledger_path,
    )
    cycle = 0
    last_result: dict = {"status": "idle", "reason": "not_started"}
    while max_cycles is None or cycle < max_cycles:
        cycle += 1
        last_result = runner.run_once(dry_run=dry_run, market_slug=market_slug)
        logger.info(
            "BTC 5m cycle %s: status=%s reason=%s market=%s",
            cycle,
            last_result.get("status"),
            last_result.get("reason") or last_result.get("signal", {}).get("reason"),
            last_result.get("market_slug"),
        )
        if max_cycles is not None and cycle >= max_cycles:
            break
        time.sleep(max(float(poll_seconds), 1.0))
    return last_result


def _parse_profile_names(profile_names: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if profile_names is None:
        raw_items = str(BTC5M_PAPER_PROFILES or "").split(",")
    elif isinstance(profile_names, str):
        raw_items = profile_names.split(",")
    else:
        raw_items = list(profile_names)

    names: list[str] = []
    for raw in raw_items:
        name = str(raw or "").strip().lower()
        if not name:
            continue
        if name == "all":
            for profile_name in BTC5M_PROFILES:
                if profile_name not in names:
                    names.append(profile_name)
            continue
        if name not in BTC5M_PROFILES:
            allowed = ", ".join(sorted(BTC5M_PROFILES))
            raise ValueError(f"Unknown BTC5M paper profile {name!r}. Allowed profiles: {allowed}, all")
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("No BTC5M paper profiles selected.")
    return names


def _paper_ledger_path(profile_name: str, ledger_dir: Path = BTC5M_PAPER_LEDGER_DIR) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in profile_name)
    return Path(ledger_dir) / f"{safe_name}.json"


def _paper_settlement_start_price(bet: dict) -> float:
    return _safe_float(
        bet.get("btc_window_start_price_full", bet.get("btc_window_start_price")),
        math.nan,
    )


def _paper_outcome_for_prices(start_price: float, final_price: float) -> str:
    return "up" if final_price >= start_price else "down"


def settle_btc5m_paper_ledger(
    *,
    ledger: BetLedger,
    price_client=None,
    settlement_client=None,
    now: datetime | None = None,
    settlement_delay_seconds: float = BTC5M_PAPER_SETTLEMENT_DELAY_SECONDS,
) -> dict:
    current = _coerce_utc(now)
    if settlement_client is None:
        if price_client is not None:
            settlement_client = ProxyBtc5mSettlementClient(price_client)
        else:
            settlement_client = build_btc5m_settlement_client()
    settled = 0
    skipped = 0
    errors: list[str] = []

    for bet in ledger.get_open_bets(fresh=True):
        if str(bet.get("order_type", "") or "") != BTC5M_ORDER_TYPE:
            continue
        window_end = _parse_placed_at(bet.get("event_date") or bet.get("window_end"))
        if window_end is None:
            skipped += 1
            continue
        if current < datetime.fromtimestamp(
            int(window_end.timestamp() + max(float(settlement_delay_seconds), 0.0)),
            tz=timezone.utc,
        ):
            skipped += 1
            continue

        try:
            reference = settlement_client.settle_bet(
                bet,
                window_end=window_end,
                now=current,
            )
        except Exception as exc:
            skipped += 1
            errors.append(f"bet #{bet.get('id')} settlement failed: {exc}")
            continue

        final_price = reference.final_price
        start_price = reference.start_price
        winning_side = reference.winning_side
        side = str(bet.get("side", "") or "").strip().lower()
        won = side == winning_side
        result = ledger.settle_bet(int(bet["id"]), won=won)
        if result.ok:
            ledger.update_bet_fields(
                int(bet["id"]),
                require_open=False,
                btc_paper_settlement_price=round(final_price, 4) if final_price is not None else None,
                btc_paper_settlement_source=reference.source,
                btc_paper_settlement_symbol=reference.symbol,
                btc_paper_settlement_timestamp=reference.timestamp.isoformat(),
                btc_paper_winning_side=winning_side,
                btc_paper_start_price=round(start_price, 4) if start_price is not None and not math.isnan(start_price) else None,
                btc_paper_settlement_note=reference.note,
            )
            settled += 1
        else:
            skipped += 1
            errors.append(f"bet #{bet.get('id')} settle failed: {result.status}")

    return {"settled": settled, "skipped": skipped, "errors": errors}


def _paper_profile_summary(profile_name: str, ledger: BetLedger, ledger_path: Path, settlement: dict, result: dict) -> dict:
    summary = ledger.get_summary(fresh=True)
    return {
        "profile": profile_name,
        "ledger_path": str(ledger_path),
        "last_status": result.get("status"),
        "last_reason": result.get("reason") or result.get("signal", {}).get("reason"),
        "last_reason_code": result.get("reason_code") or result.get("signal", {}).get("reason_code"),
        "orders": len(result.get("orders", []) or []),
        "settlement": settlement,
        "summary": {
            "total_bets": summary["total_bets"],
            "open_bets": summary["open_bets"],
            "settled_bets": summary["settled_bets"],
            "wins": summary["wins"],
            "losses": summary["losses"],
            "win_rate": round(summary["win_rate"], 4),
            "total_wagered": round(summary["total_wagered"], 2),
            "realized_pnl": round(summary["realized_pnl"], 2),
            "roi": round(summary["roi"], 4),
        },
    }


def _rank_paper_profiles(profile_results: dict[str, dict]) -> list[dict]:
    ranked = []
    for profile_name, payload in profile_results.items():
        summary = payload.get("summary", {})
        settled = int(summary.get("settled_bets", 0) or 0)
        ranked.append(
            {
                "profile": profile_name,
                "settled_bets": settled,
                "realized_pnl": float(summary.get("realized_pnl", 0.0) or 0.0),
                "roi": float(summary.get("roi", 0.0) or 0.0),
                "win_rate": float(summary.get("win_rate", 0.0) or 0.0),
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            item["settled_bets"] > 0,
            item["realized_pnl"],
            item["roi"],
            item["win_rate"],
        ),
        reverse=True,
    )


def run_btc5m_paper_compare_once(
    *,
    profile_names: str | list[str] | tuple[str, ...] | None = None,
    market_slug: str | None = None,
    ledger_dir: Path = BTC5M_PAPER_LEDGER_DIR,
    settlement_delay_seconds: float = BTC5M_PAPER_SETTLEMENT_DELAY_SECONDS,
    settlement_source: str | None = BTC5M_PAPER_SETTLEMENT_SOURCE,
    now: datetime | None = None,
    settle_only: bool = False,
    market_client: Btc5mMarketClient | None = None,
    price_client=None,
    settlement_client=None,
    book_client: PublicClobOrderBookClient | None = None,
) -> dict:
    current = _coerce_utc(now)
    names = _parse_profile_names(profile_names)
    price_client = price_client or build_btc_price_client()
    market_client = market_client or Btc5mMarketClient()
    settlement_client = settlement_client or build_btc5m_settlement_client(
        settlement_source=settlement_source,
        price_client=price_client,
        market_client=market_client,
    )
    book_client = book_client or PublicClobOrderBookClient()
    profile_results: dict[str, dict] = {}

    for name in names:
        ledger_path = _paper_ledger_path(name, Path(ledger_dir))
        ledger = BetLedger(path=ledger_path)
        settlement = settle_btc5m_paper_ledger(
            ledger=ledger,
            settlement_client=settlement_client,
            now=current,
            settlement_delay_seconds=settlement_delay_seconds,
        )
        if settle_only:
            result = {"status": "settle_only", "reason": "paper settlement only"}
        else:
            runner = Btc5mRunner(
                profile=resolve_btc5m_profile(name),
                ledger=ledger,
                ledger_path=ledger_path,
                market_client=market_client,
                price_client=price_client,
                book_client=book_client,
            )
            result = runner.run_once(
                dry_run=True,
                market_slug=market_slug,
                now=current,
            )
        profile_results[name] = _paper_profile_summary(name, ledger, ledger_path, settlement, result)

    ranking = _rank_paper_profiles(profile_results)
    return {
        "status": "ok",
        "mode": "paper_compare",
        "profiles": profile_results,
        "ranking": ranking,
        "best_profile": ranking[0]["profile"] if ranking and ranking[0]["settled_bets"] > 0 else None,
        "settlement_delay_seconds": settlement_delay_seconds,
        "settlement_source": str(settlement_source or BTC5M_PAPER_SETTLEMENT_SOURCE or "polymarket"),
        "settlement_note": (
            "Default paper settlement waits for Polymarket's official resolved outcome, "
            "which is based on the Chainlink BTC/USD Data Stream. Set "
            "BTC5M_PAPER_SETTLEMENT_SOURCE=coinbase or binance only for proxy settlement."
        ),
    }


def run_btc5m_paper_compare_loop(
    *,
    profile_names: str | list[str] | tuple[str, ...] | None = None,
    market_slug: str | None = None,
    ledger_dir: Path = BTC5M_PAPER_LEDGER_DIR,
    poll_seconds: float = BTC5M_POLL_SECONDS,
    max_cycles: int | None = None,
    settlement_delay_seconds: float = BTC5M_PAPER_SETTLEMENT_DELAY_SECONDS,
    settlement_source: str | None = BTC5M_PAPER_SETTLEMENT_SOURCE,
) -> dict:
    cycle = 0
    last_result: dict = {"status": "idle", "reason": "not_started"}
    while max_cycles is None or cycle < max_cycles:
        cycle += 1
        last_result = run_btc5m_paper_compare_once(
            profile_names=profile_names,
            market_slug=market_slug,
            ledger_dir=ledger_dir,
            settlement_delay_seconds=settlement_delay_seconds,
            settlement_source=settlement_source,
        )
        logger.info(
            "BTC 5m paper cycle %s: best=%s profiles=%s",
            cycle,
            last_result.get("best_profile"),
            ",".join(last_result.get("profiles", {}).keys()),
        )
        if max_cycles is not None and cycle >= max_cycles:
            break
        time.sleep(max(float(poll_seconds), 1.0))
    return last_result
