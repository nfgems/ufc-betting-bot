import json
from dataclasses import replace
from datetime import datetime, timezone

from src.polymarket.btc_5m import (
    Btc5mMarket,
    Btc5mRunner,
    BTC5M_ORDER_TYPE,
    PolymarketBtc5mSettlementClient,
    BtcReferencePrice,
    BtcPriceSnapshot,
    CoinbaseBtcPriceClient,
    FallbackBtcPriceClient,
    HyperliquidPriceClient,
    OrderBookSummary,
    build_btc_price_client_for_profile,
    official_btc5m_winning_side_from_event,
    _parse_btc5m_market,
    btc5m_slug_for_window,
    evaluate_signal,
    run_btc5m_paper_compare_once,
    resolve_btc5m_profile,
    settle_btc5m_paper_ledger,
)
from src.polymarket.tracker import BetLedger


def _market(start: datetime | None = None) -> Btc5mMarket:
    start = start or datetime(2026, 6, 20, 20, 10, tzinfo=timezone.utc)
    end = datetime.fromtimestamp(int(start.timestamp()) + 300, tz=timezone.utc)
    slug = f"btc-updown-5m-{int(start.timestamp())}"
    return Btc5mMarket(
        event_id="612888",
        market_id=f"market-{int(start.timestamp())}",
        condition_id=f"condition-{int(start.timestamp())}",
        slug=slug,
        question="Bitcoin Up or Down - June 20, 4:10PM-4:15PM ET",
        window_start=start,
        window_end=end,
        token_up="up-token",
        token_down="down-token",
        price_up=0.55,
        price_down=0.45,
        tick_size="0.01",
        order_min_size=5.0,
        neg_risk=False,
        active=True,
        closed=False,
        accepting_orders=True,
        fee_schedule={"rate": 0.07, "exponent": 1},
    )


def _book(token_id: str, bid: float, ask: float, size: float = 100.0) -> OrderBookSummary:
    return OrderBookSummary(
        token_id=token_id,
        best_bid=bid,
        best_ask=ask,
        bid_size=size,
        ask_size=size,
        spread=ask - bid,
        top_ask_notional=ask * size,
        total_ask_notional=ask * size,
        min_order_size=5.0,
        tick_size="0.01",
        neg_risk=False,
    )


def _add_open_btc5m_bet(ledger: BetLedger, market: Btc5mMarket, *, side: str = "up") -> dict:
    price = 0.50
    amount = 5.0
    shares = amount / price
    token_id = market.token_up if side == "up" else market.token_down
    return ledger.add_bet(
        fighter=f"BTC 5m {side.title()}",
        opponent="BTC 5m Down" if side == "up" else "BTC 5m Up",
        side=side,
        amount=amount,
        price=price,
        shares=shares,
        token_id=token_id,
        market_id=market.market_id,
        condition_id=market.condition_id,
        model_prob=0.0,
        market_prob=price,
        edge=0.0,
        decimal_odds=2.0,
        dry_run=True,
        event_date=market.window_end.isoformat(),
        market_event_date=market.window_start.isoformat(),
        order_type=BTC5M_ORDER_TYPE,
        reason="test open BTC5m position",
        metadata={
            "strategy": "btc5m_momentum",
            "market_slug": market.slug,
            "window_start": market.window_start.isoformat(),
            "window_end": market.window_end.isoformat(),
            "profile": "conservative",
            "strategy_style": "momentum",
            "fee_schedule": market.fee_schedule,
            "btc_move_usd": 90.0,
            "btc_window_start_price": 100_000.0,
            "btc_window_start_price_full": 100_000.0,
            "btc_current_price": 100_090.0,
            "btc_current_price_full": 100_090.0,
        },
    )


def test_btc5m_slug_uses_utc_five_minute_window():
    now = datetime(2026, 6, 20, 20, 12, 34, tzinfo=timezone.utc)

    assert btc5m_slug_for_window(now) == "btc-updown-5m-1781986200"


def test_late_profile_uses_final_seconds_entry_window():
    profile = resolve_btc5m_profile("late")

    assert profile.entry_seconds_left == 15.0
    assert profile.entry_tolerance_seconds == 10.0
    assert profile.min_move_usd == 10.0


def test_confidence_profile_uses_high_crowd_threshold_and_maker_limits():
    profile = resolve_btc5m_profile("confidence")

    assert profile.min_supporting_prob == 0.70
    assert profile.max_entry_price == 0.83
    assert profile.order_style == "maker_limit"


def test_cheap_side_profile_uses_price_range_without_schedule_limits():
    profile = resolve_btc5m_profile("cheap_side")

    assert profile.strategy_style == "cheap_side"
    assert profile.min_entry_price == 0.15
    assert profile.max_entry_price == 0.45
    assert profile.entry_seconds_left - profile.entry_tolerance_seconds == 90.0
    assert profile.trading_hours_timezone == ""
    assert profile.trading_start_hour == 0.0
    assert profile.trading_end_hour == 24.0
    assert profile.weekdays_only is False
    assert profile.cooldown_windows_after_trade == 0


def test_late_capture_profile_uses_final_minute_probability_capture():
    profile = resolve_btc5m_profile("late_capture")

    assert profile.strategy_style == "probability_capture"
    assert profile.min_supporting_prob == 0.85
    assert profile.min_entry_price == 0.85
    assert profile.max_entry_price == 0.97
    assert profile.max_entry_support_gap == 0.0
    assert profile.max_spread_fraction == 0.02
    assert profile.entry_seconds_left - profile.entry_tolerance_seconds == 5.0
    assert profile.entry_seconds_left + profile.entry_tolerance_seconds == 60.0


def test_late_capture_variant_profiles_resolve_expected_specs():
    min88 = resolve_btc5m_profile("late_capture_min88")
    gap005 = resolve_btc5m_profile("late_capture_gap005")
    cap94 = resolve_btc5m_profile("late_capture_cap94")
    cap93 = resolve_btc5m_profile("late_capture_cap93")
    window_30_60 = resolve_btc5m_profile("late_capture_30_60")
    window_45_60 = resolve_btc5m_profile("late_capture_45_60")
    v2 = resolve_btc5m_profile("late_capture_v2")
    mid_gap005 = resolve_btc5m_profile("late_capture_mid_gap005")
    full_gap005 = resolve_btc5m_profile("late_capture_full_gap005")
    full_gap010 = resolve_btc5m_profile("late_capture_full_gap010")
    full_min88 = resolve_btc5m_profile("late_capture_full_min88")
    full_min90 = resolve_btc5m_profile("late_capture_full_min90")
    full_gap005_liq = resolve_btc5m_profile("late_capture_full_gap005_liq")
    min86 = resolve_btc5m_profile("late_capture_min86")
    min90 = resolve_btc5m_profile("late_capture_min90")
    min92 = resolve_btc5m_profile("late_capture_min92")
    mid_min88 = resolve_btc5m_profile("late_capture_mid_min88")
    gap005_min88 = resolve_btc5m_profile("late_capture_gap005_min88")
    gap010_min88 = resolve_btc5m_profile("late_capture_gap010_min88")
    full_min88_liq = resolve_btc5m_profile("late_capture_full_min88_liq")

    for cap97_profile in (
        min86,
        min88,
        min90,
        min92,
        gap005,
        mid_gap005,
        mid_min88,
        full_min88,
        gap005_min88,
        gap010_min88,
        full_min88_liq,
    ):
        assert cap97_profile.max_entry_price == 0.97
    assert min88.min_supporting_prob == 0.88
    assert gap005.max_entry_support_gap == 0.005
    for live_sized_profile in (gap005, gap005_min88):
        assert live_sized_profile.trade_notional_usd == 50.0
        assert live_sized_profile.max_notional_per_trade == 55.0
        assert live_sized_profile.allocation_fraction == 1.0
        assert live_sized_profile.daily_loss_limit_usd == 200.0
    assert cap94.max_entry_price == 0.94
    assert cap93.max_entry_price == 0.93
    assert window_30_60.entry_seconds_left - window_30_60.entry_tolerance_seconds == 30.0
    assert window_30_60.entry_seconds_left + window_30_60.entry_tolerance_seconds == 60.0
    assert window_45_60.entry_seconds_left - window_45_60.entry_tolerance_seconds == 45.0
    assert window_45_60.entry_seconds_left + window_45_60.entry_tolerance_seconds == 60.0
    assert v2.min_supporting_prob == 0.88
    assert v2.max_entry_price == 0.94
    assert v2.max_entry_support_gap == 0.005
    assert v2.entry_seconds_left - v2.entry_tolerance_seconds == 30.0
    assert v2.entry_seconds_left + v2.entry_tolerance_seconds == 60.0
    assert mid_gap005.entry_seconds_left - mid_gap005.entry_tolerance_seconds == 30.0
    assert mid_gap005.entry_seconds_left + mid_gap005.entry_tolerance_seconds == 150.0
    assert mid_gap005.max_entry_support_gap == 0.005
    for full_window in (full_gap005, full_gap010, full_min88, full_min90, full_gap005_liq):
        assert full_window.entry_seconds_left - full_window.entry_tolerance_seconds == 0.0
        assert full_window.entry_seconds_left + full_window.entry_tolerance_seconds == 300.0
    assert full_gap005.max_entry_support_gap == 0.005
    assert full_gap010.max_entry_support_gap == 0.010
    assert full_min88.min_supporting_prob == 0.88
    assert full_min90.min_supporting_prob == 0.90
    assert full_gap005_liq.max_entry_support_gap == 0.005
    assert full_gap005_liq.min_top_ask_notional == 10.0
    assert full_gap005_liq.min_total_ask_notional == 40.0
    assert min86.min_supporting_prob == 0.86
    assert min90.min_supporting_prob == 0.90
    assert min92.min_supporting_prob == 0.92
    assert mid_min88.entry_seconds_left - mid_min88.entry_tolerance_seconds == 30.0
    assert mid_min88.entry_seconds_left + mid_min88.entry_tolerance_seconds == 150.0
    assert mid_min88.min_supporting_prob == 0.88
    assert gap005_min88.max_entry_support_gap == 0.005
    assert gap005_min88.min_supporting_prob == 0.88
    assert gap010_min88.max_entry_support_gap == 0.010
    assert gap010_min88.min_supporting_prob == 0.88
    assert full_min88_liq.entry_seconds_left - full_min88_liq.entry_tolerance_seconds == 0.0
    assert full_min88_liq.entry_seconds_left + full_min88_liq.entry_tolerance_seconds == 300.0
    assert full_min88_liq.min_supporting_prob == 0.88
    assert full_min88_liq.min_top_ask_notional == 10.0
    assert full_min88_liq.min_total_ask_notional == 40.0


def test_alt_asset_late_capture_profiles_resolve_expected_specs():
    assets = {
        "eth": ("ETH", "Ethereum", "eth-updown-5m", "binance", ("coinbase", "hyperliquid"), "ETH-USD", "ETH"),
        "sol": ("SOL", "Solana", "sol-updown-5m", "binance", ("coinbase", "hyperliquid"), "SOL-USD", "SOL"),
        "xrp": ("XRP", "XRP", "xrp-updown-5m", "binance", ("coinbase", "hyperliquid"), "XRP-USD", "XRP"),
        "doge": ("DOGE", "Dogecoin", "doge-updown-5m", "binance", ("coinbase", "hyperliquid"), "DOGE-USD", "DOGE"),
        "hype": ("HYPE", "Hyperliquid", "hype-updown-5m", "hyperliquid", (), "HYPE-USD", "@107"),
        "bnb": ("BNB", "BNB", "bnb-updown-5m", "binance", ("coinbase", "hyperliquid"), "BNB-USD", "BNB"),
    }

    for key, (symbol, name, prefix, price_source, fallbacks, coinbase_product_id, hyperliquid_coin) in assets.items():
        gap005 = resolve_btc5m_profile(f"{key}_late_capture_gap005")
        gap005_min88 = resolve_btc5m_profile(f"{key}_late_capture_gap005_min88")

        for profile in (gap005, gap005_min88):
            assert profile.trade_notional_usd == 50.0
            assert profile.max_notional_per_trade == 55.0
            assert profile.daily_loss_limit_usd == 200.0
            assert profile.allocation_fraction == 1.0
            assert profile.asset_symbol == symbol
            assert profile.asset_name == name
            assert profile.market_slug_prefix == prefix
            assert profile.price_source == price_source
            assert profile.price_source_fallbacks == fallbacks
            assert profile.coinbase_product_id == coinbase_product_id
            assert profile.hyperliquid_coin == hyperliquid_coin
            assert profile.max_entry_price == 0.97
            assert profile.max_entry_support_gap == 0.005

        assert gap005.min_supporting_prob == 0.85
        assert gap005_min88.min_supporting_prob == 0.88


def test_alt_asset_price_client_order_uses_binance_then_backups_except_hype():
    eth_client = build_btc_price_client_for_profile(resolve_btc5m_profile("eth_late_capture_gap005"))
    hype_client = build_btc_price_client_for_profile(resolve_btc5m_profile("hype_late_capture_gap005"))

    assert isinstance(eth_client, FallbackBtcPriceClient)
    assert [client.__class__.__name__ for client in eth_client.clients] == [
        "BinanceBtcPriceClient",
        "CoinbaseBtcPriceClient",
        "HyperliquidPriceClient",
    ]
    assert eth_client.clients[0].symbol == "ETHUSDT"
    assert eth_client.clients[1].product_id == "ETH-USD"
    assert eth_client.clients[2].coin == "ETH"
    assert isinstance(hype_client, HyperliquidPriceClient)
    assert hype_client.coin == "@107"


def test_cheap_side_variant_profiles_resolve_expected_specs():
    below30 = resolve_btc5m_profile("cheap_below30")
    below20 = resolve_btc5m_profile("cheap_below20")
    below10 = resolve_btc5m_profile("cheap_below10")
    below20_early = resolve_btc5m_profile("cheap_below20_early")
    below20_liq = resolve_btc5m_profile("cheap_below20_liq")

    assert below30.strategy_style == "cheap_side"
    assert below30.min_entry_price == 0.05
    assert below30.max_entry_price == 0.30
    assert below20.min_entry_price == 0.05
    assert below20.max_entry_price == 0.20
    assert below10.min_entry_price == 0.03
    assert below10.max_entry_price == 0.10
    assert below20_early.entry_seconds_left - below20_early.entry_tolerance_seconds == 180.0
    assert below20_early.entry_seconds_left + below20_early.entry_tolerance_seconds == 300.0
    assert below20_early.min_entry_price == 0.05
    assert below20_early.max_entry_price == 0.20
    assert below20_liq.min_entry_price == 0.05
    assert below20_liq.max_entry_price == 0.20
    assert below20_liq.min_top_ask_notional == 10.0
    assert below20_liq.min_total_ask_notional == 40.0


def test_cheap_early_take_profit_sweep_matrix_resolves():
    # Spot-check three corners of the cheap_<band>_<timing>_<tp> sweep cross-product.
    b10_ve_tp05 = resolve_btc5m_profile("cheap_b10_ve_tp05")
    b20_e_tp08 = resolve_btc5m_profile("cheap_b20_e_tp08")
    b30_em_tp12 = resolve_btc5m_profile("cheap_b30_em_tp12")

    # Price bands.
    assert (b10_ve_tp05.min_entry_price, b10_ve_tp05.max_entry_price) == (0.03, 0.10)
    assert (b20_e_tp08.min_entry_price, b20_e_tp08.max_entry_price) == (0.05, 0.20)
    assert (b30_em_tp12.min_entry_price, b30_em_tp12.max_entry_price) == (0.05, 0.30)

    # Early-entry windows (seconds_left +/- tolerance).
    assert (b10_ve_tp05.entry_seconds_left, b10_ve_tp05.entry_tolerance_seconds) == (270.0, 30.0)
    assert (b20_e_tp08.entry_seconds_left, b20_e_tp08.entry_tolerance_seconds) == (240.0, 60.0)
    assert (b30_em_tp12.entry_seconds_left, b30_em_tp12.entry_tolerance_seconds) == (210.0, 90.0)

    # Take-profit lock levels; tp08 reproduces the historical cheap-side defaults.
    assert b10_ve_tp05.tp_profit_target == 0.05
    assert b20_e_tp08.tp_profit_target == 0.08
    assert b30_em_tp12.tp_profit_target == 0.12
    assert (b30_em_tp12.tp_trail_arm, b30_em_tp12.tp_trail_giveback) == (0.14, 0.08)

    # Every profile keeps the cheap-side strategy + exit model.
    assert b10_ve_tp05.strategy_style == "cheap_side"


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _FakeCoinbaseSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        if url.endswith("/candles"):
            return _FakeResponse([[1782159000, 99990.0, 100020.0, 100000.0, 100010.0, 12.0]])
        if url.endswith("/ticker"):
            return _FakeResponse({"price": "100025.50"})
        return _FakeResponse({})


class _FakeHyperliquidSession:
    def __init__(self):
        self.calls = []

    def post(self, url, json=None, timeout=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if json.get("type") == "candleSnapshot":
            return _FakeResponse(
                [
                    {
                        "t": 1782159000000,
                        "T": 1782159059999,
                        "s": "@107",
                        "i": "1m",
                        "o": "61.250",
                        "c": "61.300",
                        "h": "61.400",
                        "l": "61.200",
                        "v": "1000.0",
                        "n": 10,
                    }
                ]
            )
        if json.get("type") == "allMids":
            return _FakeResponse({"@107": "61.375"})
        return _FakeResponse({})


def test_coinbase_price_client_reads_window_open_and_ticker():
    session = _FakeCoinbaseSession()
    client = CoinbaseBtcPriceClient(
        base_url="https://example.test",
        product_id="BTC-USD",
        session=session,
    )

    snapshot = client.get_snapshot(
        window_start=datetime(2026, 6, 22, 20, 10, tzinfo=timezone.utc),
        now=datetime(2026, 6, 22, 20, 12, tzinfo=timezone.utc),
    )

    assert snapshot.source == "coinbase"
    assert snapshot.symbol == "BTC-USD"
    assert snapshot.window_start_price == 100000.0
    assert snapshot.current_price == 100025.5
    assert session.calls[0]["params"]["granularity"] == 60


def test_coinbase_price_client_reads_reference_price_at_timestamp():
    session = _FakeCoinbaseSession()
    client = CoinbaseBtcPriceClient(
        base_url="https://example.test",
        product_id="BTC-USD",
        session=session,
    )

    reference = client.get_price_at(
        datetime(2026, 6, 22, 20, 15, tzinfo=timezone.utc),
        now=datetime(2026, 6, 22, 20, 16, tzinfo=timezone.utc),
    )

    assert reference.source == "coinbase"
    assert reference.symbol == "BTC-USD"
    assert reference.price == 100000.0
    assert reference.timestamp.isoformat() == "2026-06-22T20:15:00+00:00"


def test_hyperliquid_price_client_reads_spot_candle_and_mid():
    session = _FakeHyperliquidSession()
    client = HyperliquidPriceClient(
        base_url="https://example.test",
        coin="@107",
        session=session,
    )

    snapshot = client.get_snapshot(
        window_start=datetime(2026, 6, 22, 20, 10, tzinfo=timezone.utc),
        now=datetime(2026, 6, 22, 20, 12, tzinfo=timezone.utc),
    )

    assert snapshot.source == "hyperliquid"
    assert snapshot.symbol == "@107"
    assert snapshot.window_start_price == 61.25
    assert snapshot.current_price == 61.375
    assert session.calls[0]["json"]["req"]["coin"] == "@107"
    assert session.calls[0]["json"]["req"]["interval"] == "1m"
    assert session.calls[1]["json"]["type"] == "allMids"


def test_parse_btc5m_event_maps_up_down_tokens():
    event = {
        "id": "612888",
        "slug": "btc-updown-5m-1781986200",
        "markets": [
            {
                "id": "2610071",
                "question": "Bitcoin Up or Down - June 20, 4:10PM-4:15PM ET",
                "slug": "btc-updown-5m-1781986200",
                "conditionId": "0xcondition",
                "outcomes": '["Up", "Down"]',
                "outcomePrices": '["0.505", "0.495"]',
                "clobTokenIds": '["up-token", "down-token"]',
                "active": True,
                "closed": False,
                "acceptingOrders": True,
                "orderPriceMinTickSize": 0.01,
                "orderMinSize": 5,
                "feeSchedule": {"rate": 0.07, "exponent": 1},
            }
        ],
    }

    market = _parse_btc5m_market(event)

    assert market is not None
    assert market.token_up == "up-token"
    assert market.token_down == "down-token"
    assert market.price_up == 0.505
    assert market.price_down == 0.495
    assert market.window_end.isoformat() == "2026-06-20T20:15:00+00:00"


def _resolved_event(slug: str, outcome_prices: str = '["1", "0"]', *, closed: bool = True):
    return {
        "id": "612888",
        "slug": slug,
        "markets": [
            {
                "id": "2610071",
                "question": "Bitcoin Up or Down - June 20, 4:10PM-4:15PM ET",
                "slug": slug,
                "conditionId": "0xcondition",
                "outcomes": '["Up", "Down"]',
                "outcomePrices": outcome_prices,
                "clobTokenIds": '["up-token", "down-token"]',
                "active": True,
                "closed": closed,
                "acceptingOrders": False,
                "umaResolutionStatus": "resolved" if closed else "unresolved",
            }
        ],
    }


def test_official_btc5m_winning_side_uses_resolved_gamma_prices():
    slug = "btc-updown-5m-1781986200"

    assert official_btc5m_winning_side_from_event(_resolved_event(slug, '["1", "0"]'), slug) == "up"
    assert official_btc5m_winning_side_from_event(_resolved_event(slug, '["0", "1"]'), slug) == "down"
    assert official_btc5m_winning_side_from_event(_resolved_event(slug, '["0.49", "0.51"]'), slug) is None
    assert official_btc5m_winning_side_from_event(_resolved_event(slug, '["1", "0"]', closed=False), slug) is None


def test_evaluate_signal_requires_momentum_and_supporting_skew():
    market = _market()
    now = datetime(2026, 6, 20, 20, 13, tzinfo=timezone.utc)
    price = BtcPriceSnapshot(
        source="binance",
        symbol="BTCUSDT",
        window_start_price=100_000.0,
        current_price=100_085.0,
        fetched_at=now,
    )

    signal = evaluate_signal(
        market=market,
        price_snapshot=price,
        up_book=_book("up-token", 0.54, 0.55),
        down_book=_book("down-token", 0.44, 0.46),
        profile=resolve_btc5m_profile("conservative"),
        now=now,
    )

    assert signal["action"] == "trade"
    assert signal["direction"] == "up"
    assert signal["btc_move_usd"] == 85.0


def test_evaluate_signal_can_buy_cheap_side_without_momentum():
    start = datetime(2026, 6, 22, 20, 10, tzinfo=timezone.utc)
    market = _market(start)
    now = datetime(2026, 6, 22, 20, 12, tzinfo=timezone.utc)
    price = BtcPriceSnapshot(
        source="binance",
        symbol="BTCUSDT",
        window_start_price=100_000.0,
        current_price=100_000.0,
        fetched_at=now,
    )

    signal = evaluate_signal(
        market=market,
        price_snapshot=price,
        up_book=_book("up-token", 0.24, 0.26),
        down_book=_book("down-token", 0.34, 0.36),
        profile=resolve_btc5m_profile("cheap_side"),
        now=now,
    )

    assert signal["action"] == "trade"
    assert signal["direction"] == "up"
    assert signal["entry_price"] == 0.26
    assert signal["strategy_style"] == "cheap_side"


def test_cheap_side_profile_can_trade_on_weekends():
    market = _market()
    now = datetime(2026, 6, 20, 20, 12, tzinfo=timezone.utc)
    price = BtcPriceSnapshot(
        source="binance",
        symbol="BTCUSDT",
        window_start_price=100_000.0,
        current_price=100_000.0,
        fetched_at=now,
    )

    signal = evaluate_signal(
        market=market,
        price_snapshot=price,
        up_book=_book("up-token", 0.24, 0.26),
        down_book=_book("down-token", 0.34, 0.36),
        profile=resolve_btc5m_profile("cheap_side"),
        now=now,
    )

    assert signal["action"] == "trade"
    assert signal["direction"] == "up"


def test_evaluate_signal_can_capture_late_high_probability_side():
    start = datetime(2026, 6, 22, 20, 10, tzinfo=timezone.utc)
    market = _market(start)
    now = datetime(2026, 6, 22, 20, 14, 30, tzinfo=timezone.utc)
    price = BtcPriceSnapshot(
        source="coinbase",
        symbol="BTC-USD",
        window_start_price=100_000.0,
        current_price=100_000.0,
        fetched_at=now,
    )

    signal = evaluate_signal(
        market=market,
        price_snapshot=price,
        up_book=_book("up-token", 0.87, 0.88),
        down_book=_book("down-token", 0.11, 0.12),
        profile=resolve_btc5m_profile("late_capture"),
        now=now,
    )

    assert signal["action"] == "trade"
    assert signal["direction"] == "up"
    assert signal["entry_price"] == 0.88
    assert signal["strategy_style"] == "probability_capture"


def test_late_capture_profile_requires_tight_relative_spread():
    start = datetime(2026, 6, 22, 20, 10, tzinfo=timezone.utc)
    market = _market(start)
    now = datetime(2026, 6, 22, 20, 14, 30, tzinfo=timezone.utc)
    price = BtcPriceSnapshot(
        source="coinbase",
        symbol="BTC-USD",
        window_start_price=100_000.0,
        current_price=100_000.0,
        fetched_at=now,
    )

    signal = evaluate_signal(
        market=market,
        price_snapshot=price,
        up_book=_book("up-token", 0.86, 0.88),
        down_book=_book("down-token", 0.11, 0.12),
        profile=resolve_btc5m_profile("late_capture"),
        now=now,
    )

    assert signal["action"] == "skip"
    assert signal["reason_code"] == "relative_spread_too_wide"


def test_late_capture_gap_variant_rejects_ask_too_far_above_market_support():
    start = datetime(2026, 6, 22, 20, 10, tzinfo=timezone.utc)
    market = _market(start)
    now = datetime(2026, 6, 22, 20, 14, 30, tzinfo=timezone.utc)
    price = BtcPriceSnapshot(
        source="coinbase",
        symbol="BTC-USD",
        window_start_price=100_000.0,
        current_price=100_000.0,
        fetched_at=now,
    )

    signal = evaluate_signal(
        market=market,
        price_snapshot=price,
        up_book=_book("up-token", 0.889, 0.900),
        down_book=_book("down-token", 0.10, 0.11),
        profile=resolve_btc5m_profile("late_capture_gap005"),
        now=now,
    )

    assert signal["action"] == "skip"
    assert signal["reason_code"] == "entry_support_gap_too_wide"


class _FakeMarketClient:
    def __init__(self, market, event=None):
        self.market = market
        self.event = event
        self.market_slug_prefixes = []

    def get_market(self, *, now=None, market_slug=None, market_slug_prefix=None):
        self.market_slug_prefixes.append(market_slug_prefix)
        return self.market

    def fetch_event_by_slug(self, slug):
        return self.event


class _FakePriceClient:
    def get_snapshot(self, *, window_start, now=None):
        return BtcPriceSnapshot(
            source="binance",
            symbol="BTCUSDT",
            window_start_price=100_000.0,
            current_price=100_090.0,
            fetched_at=now,
        )

    def get_price_at(self, timestamp, *, now=None):
        return BtcReferencePrice(
            source="fake",
            symbol="BTCUSD",
            price=100_100.0,
            timestamp=timestamp,
            fetched_at=now,
        )


class _FakeBookClient:
    def summarize(self, token_id):
        if token_id == "up-token":
            return _book("up-token", 0.54, 0.55)
        return _book("down-token", 0.44, 0.46)


class _FakeLateCaptureBookClient:
    def summarize(self, token_id):
        if token_id == "up-token":
            return _book("up-token", 0.89, 0.895)
        return _book("down-token", 0.10, 0.11)


class _FakeClobClient:
    def __init__(self, response=None):
        self.response = response or {
            "orderID": "order-1",
            "takingAmount": "9.09",
            "makingAmount": "4.3632",
            "status": "matched",
            "transactionsHashes": ["0xtx"],
            "success": True,
        }
        self.limit_order = None
        self.limit_calls = 0

    def get_cash_balance_details(self):
        return {"balance": 1000.0}

    def create_limit_order(self, **kwargs):
        self.limit_calls += 1
        self.limit_order = kwargs
        return dict(self.response)


class _DeadlineExceededClobClient(_FakeClobClient):
    def create_limit_order(self, **kwargs):
        self.limit_calls += 1
        self.limit_order = kwargs
        exc = RuntimeError(
            "[py_clob_client_v2] request error status=500 "
            "url=https://clob.polymarket.com/order "
            "body={\"error\":\"rpc error: code = DeadlineExceeded "
            "desc = context deadline exceeded\"}"
        )
        exc.status_code = 500
        raise exc


class _FakeConfidenceBookClient:
    def summarize(self, token_id):
        if token_id == "up-token":
            return _book("up-token", 0.72, 0.74)
        return _book("down-token", 0.25, 0.27)


class _FakeCheapSideBookClient:
    def summarize(self, token_id):
        if token_id == "up-token":
            return _book("up-token", 0.24, 0.26)
        return _book("down-token", 0.34, 0.36)


def test_runner_dry_run_records_btc5m_ledger_without_private_key(tmp_path):
    market = _market()
    now = datetime(2026, 6, 20, 20, 13, tzinfo=timezone.utc)
    ledger_path = tmp_path / "btc5m_ledger.json"
    ledger = BetLedger(path=ledger_path)
    runner = Btc5mRunner(
        profile=resolve_btc5m_profile("conservative"),
        ledger=ledger,
        ledger_path=ledger_path,
        market_client=_FakeMarketClient(market),
        price_client=_FakePriceClient(),
        book_client=_FakeBookClient(),
    )

    result = runner.run_once(dry_run=True, now=now)

    assert result["status"] == "ok"
    assert result["orders"][0]["status"] == "dry_run"
    bets = ledger.get_bets(fresh=True)
    assert len(bets) == 1
    assert bets[0]["fighter"] == "BTC 5m Up"
    assert bets[0]["dry_run"] is True
    assert bets[0]["market_slug"] == market.slug


def test_runner_uses_asset_profile_prefix_and_labels(tmp_path):
    market = _market()
    now = datetime(2026, 6, 20, 20, 14, 30, tzinfo=timezone.utc)
    ledger_path = tmp_path / "eth_late_capture_gap005.json"
    ledger = BetLedger(path=ledger_path)
    market_client = _FakeMarketClient(market)
    runner = Btc5mRunner(
        profile=resolve_btc5m_profile("eth_late_capture_gap005"),
        ledger=ledger,
        ledger_path=ledger_path,
        market_client=market_client,
        price_client=_FakePriceClient(),
        book_client=_FakeLateCaptureBookClient(),
    )

    result = runner.run_once(dry_run=True, now=now)

    assert result["status"] == "ok"
    assert market_client.market_slug_prefixes == ["eth-updown-5m"]
    bets = ledger.get_bets(fresh=True)
    assert bets[0]["fighter"] == "ETH 5m Up"
    assert bets[0]["opponent"] == "ETH 5m Down"
    assert abs(bets[0]["amount"] - 50.0) <= 0.02
    assert bets[0]["asset_symbol"] == "ETH"
    assert bets[0]["profile_price_source"] == "binance"
    assert bets[0]["profile_price_source_fallbacks"] == ["coinbase", "hyperliquid"]


def test_runner_records_actual_fill_fields_from_matched_clob_response(monkeypatch, tmp_path):
    market = _market()
    now = datetime(2026, 6, 20, 20, 13, tzinfo=timezone.utc)
    ledger_path = tmp_path / "btc5m_ledger.json"
    ledger = BetLedger(path=ledger_path)
    clob = _FakeClobClient()
    monkeypatch.setattr(
        "src.polymarket.btc_5m.assert_polymarket_real_trading_allowed",
        lambda **_kwargs: None,
    )
    runner = Btc5mRunner(
        profile=resolve_btc5m_profile("conservative"),
        ledger=ledger,
        ledger_path=ledger_path,
        market_client=_FakeMarketClient(market),
        price_client=_FakePriceClient(),
        book_client=_FakeBookClient(),
        clob_client=clob,
    )

    result = runner.run_once(dry_run=False, now=now)

    assert result["status"] == "ok"
    assert result["orders"][0]["status"] == "placed"
    assert result["orders"][0]["actual_fill_price"] == 0.48
    assert clob.limit_order["price"] == 0.55
    bets = ledger.get_bets(fresh=True)
    assert bets[0]["price"] == 0.55
    assert bets[0]["submitted_entry_price"] == 0.55
    assert bets[0]["actual_fill_price"] == 0.48
    assert bets[0]["actual_fill_amount"] == 4.3632
    assert bets[0]["actual_filled_shares"] == 9.09
    assert bets[0]["actual_fill_tx_hash"] == "0xtx"


def test_runner_keeps_deadline_exceeded_order_unknown_and_blocks_resubmit(
    monkeypatch,
    tmp_path,
):
    market = _market()
    now = datetime(2026, 6, 20, 20, 13, tzinfo=timezone.utc)
    ledger_path = tmp_path / "btc5m_ledger.json"
    ledger = BetLedger(path=ledger_path)
    clob = _DeadlineExceededClobClient()
    monkeypatch.setattr(
        "src.polymarket.btc_5m.assert_polymarket_real_trading_allowed",
        lambda **_kwargs: None,
    )
    runner = Btc5mRunner(
        profile=resolve_btc5m_profile("conservative"),
        ledger=ledger,
        ledger_path=ledger_path,
        market_client=_FakeMarketClient(market),
        price_client=_FakePriceClient(),
        book_client=_FakeBookClient(),
        clob_client=clob,
    )

    result = runner.run_once(dry_run=False, now=now)

    assert result["status"] == "error"
    assert result["orders"][0]["status"] == "unknown"
    assert clob.limit_calls == 1
    bets = ledger.get_bets(fresh=True)
    assert len(bets) == 1
    assert bets[0]["status"] == "open"
    assert bets[0]["placement_state"] == "unknown"
    assert "DeadlineExceeded" in bets[0]["submission_error"]

    second = runner.run_once(dry_run=False, now=now)

    assert second["status"] == "idle"
    assert second["reason_code"] == "duplicate_market"
    assert clob.limit_calls == 1


def test_runner_does_not_place_hedge_after_unknown_primary_order(monkeypatch, tmp_path):
    market = _market()
    now = datetime(2026, 6, 20, 20, 13, tzinfo=timezone.utc)
    calls = []

    def fake_place_order(**kwargs):
        calls.append(kwargs["direction"])
        return {
            "status": "unknown",
            "direction": kwargs["direction"],
            "ledger_bet_id": 1,
        }

    monkeypatch.setattr("src.polymarket.btc_5m._place_order", fake_place_order)
    runner = Btc5mRunner(
        profile=replace(
            resolve_btc5m_profile("conservative"),
            enable_extreme_skew_hedge=True,
            hedge_skew_threshold=0.50,
            hedge_max_price=0.60,
            hedge_notional_usd=5.0,
        ),
        ledger=BetLedger(path=tmp_path / "btc5m_ledger.json"),
        ledger_path=tmp_path / "btc5m_ledger.json",
        market_client=_FakeMarketClient(market),
        price_client=_FakePriceClient(),
        book_client=_FakeBookClient(),
    )

    result = runner.run_once(dry_run=True, now=now)

    assert result["status"] == "error"
    assert calls == ["up"]
    assert len(result["orders"]) == 1


def test_confidence_profile_dry_run_records_resting_bid_entry(tmp_path):
    market = _market()
    now = datetime(2026, 6, 20, 20, 13, tzinfo=timezone.utc)
    ledger_path = tmp_path / "btc5m_confidence_ledger.json"
    ledger = BetLedger(path=ledger_path)
    runner = Btc5mRunner(
        profile=resolve_btc5m_profile("confidence"),
        ledger=ledger,
        ledger_path=ledger_path,
        market_client=_FakeMarketClient(market),
        price_client=_FakePriceClient(),
        book_client=_FakeConfidenceBookClient(),
    )

    result = runner.run_once(dry_run=True, now=now)

    assert result["status"] == "ok"
    assert result["orders"][0]["status"] == "dry_run"
    assert result["orders"][0]["price"] == 0.72
    bets = ledger.get_bets(fresh=True)
    assert bets[0]["price"] == 0.72
    assert bets[0]["order_style"] == "maker_limit"
    assert bets[0]["entry_price_source"] == "best_bid"


def test_runner_respects_profile_cooldown_after_trade(tmp_path):
    previous_market = _market(datetime(2026, 6, 22, 20, 10, tzinfo=timezone.utc))
    current_market = _market(datetime(2026, 6, 22, 20, 15, tzinfo=timezone.utc))
    market_client = _FakeMarketClient(previous_market)
    ledger_path = tmp_path / "btc5m_cooldown_ledger.json"
    ledger = BetLedger(path=ledger_path)
    runner = Btc5mRunner(
        # Cooldown is disabled in the shipped profiles, so inject a 1-window
        # cooldown to exercise the runner's cooldown gate directly.
        profile=replace(
            resolve_btc5m_profile("cheap_side"),
            cooldown_windows_after_trade=1,
        ),
        ledger=ledger,
        ledger_path=ledger_path,
        market_client=market_client,
        price_client=_FakePriceClient(),
        book_client=_FakeCheapSideBookClient(),
    )

    first = runner.run_once(
        dry_run=True,
        now=datetime(2026, 6, 22, 20, 12, tzinfo=timezone.utc),
    )
    market_client.market = current_market
    second = runner.run_once(
        dry_run=True,
        now=datetime(2026, 6, 22, 20, 17, tzinfo=timezone.utc),
    )

    assert first["status"] == "ok"
    assert second["status"] == "idle"
    assert second["reason_code"] == "cooldown_after_trade"


def test_runner_can_record_signal_snapshot_jsonl(tmp_path):
    market = _market()
    now = datetime(2026, 6, 20, 20, 13, tzinfo=timezone.utc)
    ledger_path = tmp_path / "btc5m_ledger.json"
    signal_log_path = tmp_path / "btc5m_signal_snapshots.jsonl"
    ledger = BetLedger(path=ledger_path)
    runner = Btc5mRunner(
        profile=resolve_btc5m_profile("conservative"),
        ledger=ledger,
        ledger_path=ledger_path,
        market_client=_FakeMarketClient(market),
        price_client=_FakePriceClient(),
        book_client=_FakeBookClient(),
        signal_log_path=signal_log_path,
        record_signal_snapshots=True,
    )

    result = runner.run_once(dry_run=True, now=now)

    assert result["status"] == "ok"
    rows = [json.loads(line) for line in signal_log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["schema_version"] == 1
    assert row["market"]["slug"] == market.slug
    assert row["price"]["price_to_beat"] == 100000.0
    assert row["price"]["distance_to_price_to_beat_usd"] == 90.0
    assert row["books"]["up"]["best_ask"] == 0.55
    assert row["signal"]["action"] == "trade"
    assert row["result"]["orders"][0]["status"] == "dry_run"


def test_exit_shadow_disabled_does_not_write_shadow_log(tmp_path):
    market = _market()
    now = datetime(2026, 6, 20, 20, 13, tzinfo=timezone.utc)
    ledger_path = tmp_path / "btc5m_ledger.json"
    shadow_log_path = tmp_path / "btc5m_exit_shadow.jsonl"
    ledger = BetLedger(path=ledger_path)
    runner = Btc5mRunner(
        profile=resolve_btc5m_profile("conservative"),
        ledger=ledger,
        ledger_path=ledger_path,
        market_client=_FakeMarketClient(market),
        price_client=_FakePriceClient(),
        book_client=_FakeBookClient(),
        exit_shadow_enabled=False,
        exit_shadow_log_path=shadow_log_path,
    )

    result = runner.run_once(dry_run=True, now=now)

    assert result["status"] == "ok"
    assert len(ledger.get_bets(fresh=True)) == 1
    assert not shadow_log_path.exists()


def test_exit_shadow_enabled_logs_without_mutating_ledger_and_still_settles(tmp_path):
    market = _market()
    now = datetime(2026, 6, 20, 20, 13, tzinfo=timezone.utc)
    ledger_path = tmp_path / "btc5m_ledger.json"
    shadow_log_path = tmp_path / "btc5m_exit_shadow.jsonl"
    ledger = BetLedger(path=ledger_path)
    _add_open_btc5m_bet(ledger, market, side="up")
    runner = Btc5mRunner(
        profile=resolve_btc5m_profile("conservative"),
        ledger=ledger,
        ledger_path=ledger_path,
        market_client=_FakeMarketClient(market),
        price_client=_FakePriceClient(),
        book_client=_FakeBookClient(),
        exit_shadow_enabled=True,
        exit_shadow_log_path=shadow_log_path,
    )

    runner.run_once(dry_run=True, now=now)

    bets_after_shadow = ledger.get_bets(fresh=True)
    assert len(bets_after_shadow) == 1
    assert bets_after_shadow[0]["status"] == "open"
    rows = [json.loads(line) for line in shadow_log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["decision"]["action"] == "hold"
    assert rows[0]["ledger_bet_id"] == bets_after_shadow[0]["id"]

    settlement = settle_btc5m_paper_ledger(
        ledger=ledger,
        price_client=_FakePriceClient(),
        now=datetime(2026, 6, 20, 20, 16, tzinfo=timezone.utc),
        settlement_delay_seconds=0,
    )

    bets_after_settlement = ledger.get_bets(fresh=True)
    assert settlement["settled"] == 1
    assert bets_after_settlement[0]["status"] == "won"


def test_exit_shadow_entry_window_skip_still_records_open_position_mark(tmp_path):
    market = _market()
    now = datetime(2026, 6, 20, 20, 10, 30, tzinfo=timezone.utc)
    ledger_path = tmp_path / "btc5m_ledger.json"
    shadow_log_path = tmp_path / "btc5m_exit_shadow.jsonl"
    ledger = BetLedger(path=ledger_path)
    _add_open_btc5m_bet(ledger, market, side="up")
    runner = Btc5mRunner(
        profile=resolve_btc5m_profile("conservative"),
        ledger=ledger,
        ledger_path=ledger_path,
        market_client=_FakeMarketClient(market),
        price_client=_FakePriceClient(),
        book_client=_FakeBookClient(),
        exit_shadow_enabled=True,
        exit_shadow_log_path=shadow_log_path,
    )

    result = runner.run_once(dry_run=True, now=now)

    assert result["status"] == "idle"
    assert result["reason_code"] == "outside_entry_window"
    assert len(ledger.get_open_bets(fresh=True)) == 1
    rows = [json.loads(line) for line in shadow_log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["market"]["seconds_left"] == 270.0


def test_exit_shadow_with_no_open_position_does_not_write_record(tmp_path):
    market = _market()
    now = datetime(2026, 6, 20, 20, 10, 30, tzinfo=timezone.utc)
    ledger_path = tmp_path / "btc5m_ledger.json"
    shadow_log_path = tmp_path / "btc5m_exit_shadow.jsonl"
    ledger = BetLedger(path=ledger_path)
    runner = Btc5mRunner(
        profile=resolve_btc5m_profile("conservative"),
        ledger=ledger,
        ledger_path=ledger_path,
        market_client=_FakeMarketClient(market),
        price_client=_FakePriceClient(),
        book_client=_FakeBookClient(),
        exit_shadow_enabled=True,
        exit_shadow_log_path=shadow_log_path,
    )

    result = runner.run_once(dry_run=True, now=now)

    assert result["status"] == "idle"
    assert result["reason_code"] == "outside_entry_window"
    assert not shadow_log_path.exists()


def test_paper_settlement_marks_proxy_winner(tmp_path):
    market = _market()
    now = datetime(2026, 6, 20, 20, 13, tzinfo=timezone.utc)
    ledger_path = tmp_path / "conservative.json"
    ledger = BetLedger(path=ledger_path)
    runner = Btc5mRunner(
        profile=resolve_btc5m_profile("conservative"),
        ledger=ledger,
        ledger_path=ledger_path,
        market_client=_FakeMarketClient(market),
        price_client=_FakePriceClient(),
        book_client=_FakeBookClient(),
    )
    result = runner.run_once(dry_run=True, now=now)

    settlement = settle_btc5m_paper_ledger(
        ledger=ledger,
        price_client=_FakePriceClient(),
        now=datetime(2026, 6, 20, 20, 16, tzinfo=timezone.utc),
        settlement_delay_seconds=0,
    )

    bets = ledger.get_bets(fresh=True)
    assert result["status"] == "ok"
    assert settlement["settled"] == 1
    assert bets[0]["status"] == "won"
    assert bets[0]["btc_paper_winning_side"] == "up"
    assert bets[0]["btc_paper_settlement_price"] == 100100.0


def test_paper_settlement_can_use_official_polymarket_resolution(tmp_path):
    market = _market()
    now = datetime(2026, 6, 20, 20, 13, tzinfo=timezone.utc)
    ledger_path = tmp_path / "conservative.json"
    ledger = BetLedger(path=ledger_path)
    runner = Btc5mRunner(
        profile=resolve_btc5m_profile("conservative"),
        ledger=ledger,
        ledger_path=ledger_path,
        market_client=_FakeMarketClient(market),
        price_client=_FakePriceClient(),
        book_client=_FakeBookClient(),
    )
    result = runner.run_once(dry_run=True, now=now)

    settlement = settle_btc5m_paper_ledger(
        ledger=ledger,
        settlement_client=PolymarketBtc5mSettlementClient(
            market_client=_FakeMarketClient(
                market,
                event=_resolved_event(market.slug, '["1", "0"]'),
            )
        ),
        now=datetime(2026, 6, 20, 20, 16, tzinfo=timezone.utc),
        settlement_delay_seconds=0,
    )

    bets = ledger.get_bets(fresh=True)
    assert result["status"] == "ok"
    assert settlement["settled"] == 1
    assert bets[0]["status"] == "won"
    assert bets[0]["btc_paper_winning_side"] == "up"
    assert bets[0]["btc_paper_settlement_source"] == "polymarket_chainlink"
    assert bets[0]["btc_paper_settlement_price"] is None


def test_paper_compare_runs_profiles_with_separate_ledgers(tmp_path):
    market = _market()
    result = run_btc5m_paper_compare_once(
        profile_names="conservative,confidence",
        ledger_dir=tmp_path,
        now=datetime(2026, 6, 20, 20, 13, tzinfo=timezone.utc),
        market_client=_FakeMarketClient(market),
        price_client=_FakePriceClient(),
        book_client=_FakeConfidenceBookClient(),
    )

    assert result["status"] == "ok"
    assert result["profiles"]["conservative"]["summary"]["total_bets"] == 1
    assert result["profiles"]["confidence"]["summary"]["total_bets"] == 1
    assert (tmp_path / "conservative.json").exists()
    assert (tmp_path / "confidence.json").exists()
