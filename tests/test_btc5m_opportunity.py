import json
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.polymarket.btc_5m import (
    Btc5mMarket,
    Btc5mSettlementReference,
    BtcPriceSnapshot,
    OrderBookSummary,
    resolve_btc5m_profile,
)
from src.polymarket.btc5m_opportunity import (
    ALL_SNAPSHOTS_MODE,
    HOLD_TO_SETTLEMENT_POLICY,
    LIVE_LIKE_MODE,
    TAKE_PROFIT_EXIT_POLICY,
    Btc5mOpportunityHarness,
    render_markdown_report,
    run_btc5m_opportunity,
)
from src.polymarket.tracker import BetLedger


def _market(start: datetime | None = None, *, suffix: str = "") -> Btc5mMarket:
    start = start or datetime(2026, 6, 20, 20, 10, tzinfo=timezone.utc)
    end = start + timedelta(minutes=5)
    epoch = int(start.timestamp())
    return Btc5mMarket(
        event_id=f"event-{epoch}{suffix}",
        market_id=f"market-{epoch}{suffix}",
        condition_id=f"condition-{epoch}{suffix}",
        slug=f"btc-updown-5m-{epoch}{suffix}",
        question="Bitcoin Up or Down",
        window_start=start,
        window_end=end,
        token_up=f"up-token{suffix}",
        token_down=f"down-token{suffix}",
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


def _book(token_id: str, bid: float = 0.54, ask: float = 0.55, size: float = 100.0) -> OrderBookSummary:
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


class _SequenceMarketClient:
    def __init__(self, markets):
        self.markets = list(markets)
        self.calls = []

    def get_market(self, *, now=None, market_slug=None, market_slug_prefix=None):
        self.calls.append(
            {"now": now, "market_slug": market_slug, "market_slug_prefix": market_slug_prefix}
        )
        index = min(len(self.calls) - 1, len(self.markets) - 1)
        return self.markets[index]

    def fetch_event_by_slug(self, slug):
        return None


class _StaticPriceClient:
    def __init__(self, *, current_price: float = 100_090.0):
        self.current_price = current_price
        self.calls = []

    def get_snapshot(self, *, window_start, now=None):
        self.calls.append({"window_start": window_start, "now": now})
        return BtcPriceSnapshot(
            source="fake",
            symbol="BTCUSD",
            window_start_price=100_000.0,
            current_price=self.current_price,
            fetched_at=now,
        )


class _CountingBookClient:
    def __init__(self, *, bid: float = 0.54, ask: float = 0.55):
        self.bid = bid
        self.ask = ask
        self.calls = []

    def summarize(self, token_id):
        self.calls.append(token_id)
        return _book(token_id, self.bid, self.ask)


class _SequencePriceClient:
    def __init__(self, prices):
        self.prices = list(prices)
        self.calls = []

    def get_snapshot(self, *, window_start, now=None):
        self.calls.append({"window_start": window_start, "now": now})
        index = min(len(self.calls) - 1, len(self.prices) - 1)
        return BtcPriceSnapshot(
            source="fake",
            symbol="BTCUSD",
            window_start_price=100_000.0,
            current_price=self.prices[index],
            fetched_at=now,
        )


class _SequenceBookClient:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)
        self.calls = []

    def summarize(self, token_id):
        self.calls.append(token_id)
        cycle = min((len(self.calls) - 1) // 2, len(self.snapshots) - 1)
        side = "up" if "up-token" in token_id else "down"
        bid, ask = self.snapshots[cycle][side]
        return _book(token_id, bid, ask)


class _FakeSettlementClient:
    def __init__(self, *, winning_side: str = "up"):
        self.winning_side = winning_side
        self.calls = []

    def settle_bet(self, bet, *, window_end, now=None):
        self.calls.append({"bet": dict(bet), "window_end": window_end, "now": now})
        return Btc5mSettlementReference(
            source="fake",
            symbol="BTCUSD",
            winning_side=self.winning_side,
            timestamp=window_end,
            fetched_at=now,
            start_price=100_000.0,
            final_price=100_100.0,
            note="test settlement",
        )


class _SequenceNow:
    def __init__(self, values):
        self.values = list(values)
        self.index = 0

    def __call__(self):
        if self.index >= len(self.values):
            return self.values[-1]
        value = self.values[self.index]
        self.index += 1
        return value


def _report(harness: Btc5mOpportunityHarness, now: datetime) -> dict:
    return harness.build_report(
        run_id="test",
        target_markets=1,
        status="running",
        started_at=now,
    )


def test_combined_opportunity_report_groups_late_capture_and_cheap_side(tmp_path):
    now = datetime(2026, 6, 20, 20, 10, tzinfo=timezone.utc)
    harness = Btc5mOpportunityHarness(
        profile_names="late_capture_min88,cheap_below20",
        ledger_dir=tmp_path / "ledgers",
    )

    report = _report(harness, now)
    markdown = render_markdown_report(report)

    assert report["profile_groups"]["late_capture"]["primary_exit_policy"] == HOLD_TO_SETTLEMENT_POLICY
    assert report["profile_groups"]["late_capture"]["profiles"] == ["late_capture_min88"]
    assert report["profile_groups"]["cheap_side"]["primary_exit_policy"] == TAKE_PROFIT_EXIT_POLICY
    assert report["profile_groups"]["cheap_side"]["profiles"] == ["cheap_below20"]
    assert report["group_ranking"][LIVE_LIKE_MODE]["late_capture"]["primary_exit_policy"] == HOLD_TO_SETTLEMENT_POLICY
    assert report["group_ranking"][LIVE_LIKE_MODE]["cheap_side"]["primary_exit_policy"] == TAKE_PROFIT_EXIT_POLICY
    assert "## Strategy Tracks" in markdown
    assert "Track A - late capture primary hold-to-settlement ranking" in markdown
    assert "Track B - cheap-side reversion primary take-profit ranking" in markdown
    assert "Reference only: all_snapshots ignores entry timing" in markdown


def test_opportunity_profiles_share_one_market_price_and_book_snapshot(tmp_path):
    market = _market()
    now = market.window_start + timedelta(minutes=3)
    market_client = _SequenceMarketClient([market])
    price_client = _StaticPriceClient()
    book_client = _CountingBookClient(bid=0.72, ask=0.74)
    harness = Btc5mOpportunityHarness(
        profile_names="conservative,confidence",
        ledger_dir=tmp_path / "ledgers",
        market_client=market_client,
        price_client=price_client,
        book_client=book_client,
        settlement_client=_FakeSettlementClient(),
        settlement_delay_seconds=0,
    )

    result = harness.run_once(now=now)
    report = _report(harness, now)

    assert result["status"] == "ok"
    assert len(market_client.calls) == 1
    assert len(price_client.calls) == 1
    assert book_client.calls == [market.token_up, market.token_down]
    for mode in (LIVE_LIKE_MODE, ALL_SNAPSHOTS_MODE):
        assert report["profiles"][mode]["conservative"]["snapshots_seen"] == 1
        assert report["profiles"][mode]["confidence"]["snapshots_seen"] == 1
        assert report["profiles"][mode]["conservative"]["actual_trade_markets"] == 1
        assert report["profiles"][mode]["confidence"]["actual_trade_markets"] == 1
        assert report["profiles"][mode]["conservative"]["exit_policies"][HOLD_TO_SETTLEMENT_POLICY]["total_trades"] == 1
        assert report["profiles"][mode]["conservative"]["exit_policies"][TAKE_PROFIT_EXIT_POLICY]["total_trades"] == 1
        assert len(report["take_profit_positions"][mode]["conservative"]) == 1


def test_all_snapshots_can_trade_where_live_like_timing_skips(tmp_path):
    market = _market()
    now = market.window_start + timedelta(seconds=30)
    harness = Btc5mOpportunityHarness(
        profile_names="conservative",
        ledger_dir=tmp_path / "ledgers",
        market_client=_SequenceMarketClient([market]),
        price_client=_StaticPriceClient(),
        book_client=_CountingBookClient(),
        settlement_client=_FakeSettlementClient(),
        settlement_delay_seconds=0,
    )

    harness.run_once(now=now)
    report = _report(harness, now)

    live = report["profiles"][LIVE_LIKE_MODE]["conservative"]
    all_snapshots = report["profiles"][ALL_SNAPSHOTS_MODE]["conservative"]
    assert live["outside_entry_window"] == 1
    assert live["signal_skips_by_reason"] == {"outside_entry_window": 1}
    assert live["actual_trade_markets"] == 0
    assert all_snapshots["outside_entry_window"] == 1
    assert all_snapshots["signal_skips_by_reason"] == {}
    assert all_snapshots["signal_trade_markets"] == 1
    assert all_snapshots["actual_trade_markets"] == 1
    assert all_snapshots["trade_outside_entry_window"] == 1


def test_signal_and_risk_skips_are_counted_separately(tmp_path):
    market = _market()
    now = market.window_start + timedelta(minutes=3)
    signal_skip_harness = Btc5mOpportunityHarness(
        profile_names="conservative",
        ledger_dir=tmp_path / "signal-ledgers",
        market_client=_SequenceMarketClient([market]),
        price_client=_StaticPriceClient(current_price=100_000.0),
        book_client=_CountingBookClient(),
        settlement_client=_FakeSettlementClient(),
        settlement_delay_seconds=0,
    )
    signal_skip_harness.run_once(now=now)
    signal_report = _report(signal_skip_harness, now)

    assert signal_report["profiles"][LIVE_LIKE_MODE]["conservative"]["signal_skips_by_reason"] == {
        "insufficient_impulse": 1
    }
    assert signal_report["profiles"][LIVE_LIKE_MODE]["conservative"]["risk_skips_by_reason"] == {}

    duplicate_harness = Btc5mOpportunityHarness(
        profile_names="conservative",
        ledger_dir=tmp_path / "duplicate-ledgers",
        market_client=_SequenceMarketClient([market, market]),
        price_client=_StaticPriceClient(),
        book_client=_CountingBookClient(),
        settlement_client=_FakeSettlementClient(),
        settlement_delay_seconds=999_999,
    )
    duplicate_harness.run_once(now=now)
    duplicate_harness.run_once(now=now)
    duplicate_report = _report(duplicate_harness, now)

    live = duplicate_report["profiles"][LIVE_LIKE_MODE]["conservative"]
    assert live["signal_skips_by_reason"] == {}
    assert live["risk_skips_by_reason"] == {"duplicate_market": 1}
    assert live["signal_trade_markets"] == 1
    assert live["actual_trade_markets"] == 1
    assert len(BetLedger(path=tmp_path / "duplicate-ledgers" / LIVE_LIKE_MODE / "conservative.json").get_bets(fresh=True)) == 1


def test_ignore_risk_limits_strips_all_throttles_except_one_per_market(tmp_path):
    # A throttled profile (like the pre-disable defaults) carries a per-day cap,
    # a cooldown, a loss limit, and a finite exposure cap. The shipped profiles no
    # longer set these, so build the throttled reference explicitly.
    throttled = replace(
        resolve_btc5m_profile("cheap_side"),
        max_trades_per_day=10,
        cooldown_windows_after_trade=1,
        daily_loss_limit_usd=25.0,
        max_open_exposure_usd=25.0,
    )
    assert throttled.max_trades_per_day > 0
    assert throttled.cooldown_windows_after_trade > 0
    assert throttled.daily_loss_limit_usd > 0

    unlimited = Btc5mOpportunityHarness(
        profile_names="cheap_side",
        ledger_dir=tmp_path / "unlimited",
        ignore_risk_limits=True,
    )
    profile = unlimited.profiles["cheap_side"]
    # Every portfolio risk throttle is removed...
    assert profile.max_trades_per_day == 0
    assert profile.cooldown_windows_after_trade == 0
    assert profile.daily_loss_limit_usd == 0.0
    # ...and exposure is widened (not zeroed) so position sizing still computes.
    assert profile.max_open_exposure_usd >= throttled.max_open_exposure_usd
    assert profile.max_open_exposure_usd > 0


def test_settlement_updates_opportunity_summary_stats(tmp_path):
    market1 = _market(datetime(2026, 6, 20, 20, 10, tzinfo=timezone.utc), suffix="-a")
    market2 = _market(datetime(2026, 6, 20, 20, 15, tzinfo=timezone.utc), suffix="-b")
    harness = Btc5mOpportunityHarness(
        profile_names="conservative",
        ledger_dir=tmp_path / "ledgers",
        market_client=_SequenceMarketClient([market1, market2]),
        price_client=_StaticPriceClient(),
        book_client=_CountingBookClient(),
        settlement_client=_FakeSettlementClient(winning_side="up"),
        settlement_delay_seconds=0,
    )

    harness.run_once(now=market1.window_start + timedelta(minutes=3))
    harness.run_once(now=market2.window_start + timedelta(minutes=1))
    report = _report(harness, market2.window_start + timedelta(minutes=1))

    live = report["profiles"][LIVE_LIKE_MODE]["conservative"]
    assert live["settled_trades"] == 1
    assert live["wins"] == 1
    assert live["losses"] == 0
    assert live["realized_pnl"] > 0
    assert live["roi"] > 0
    assert live["win_rate"] == 1.0


def test_take_profit_exit_can_close_before_hold_settlement_and_compute_delta(tmp_path):
    market = _market()
    entry_time = market.window_start + timedelta(minutes=3)
    exit_time = market.window_start + timedelta(minutes=3, seconds=20)
    settle_time = market.window_end + timedelta(seconds=1)
    harness = Btc5mOpportunityHarness(
        profile_names="confidence",
        ledger_dir=tmp_path / "ledgers",
        market_client=_SequenceMarketClient([market, market, market]),
        price_client=_SequencePriceClient([100_090.0, 99_990.0]),
        book_client=_SequenceBookClient(
            [
                {"up": (0.72, 0.74), "down": (0.25, 0.27)},
                {"up": (0.50, 0.56), "down": (0.44, 0.50)},
            ]
        ),
        settlement_client=_FakeSettlementClient(winning_side="up"),
        settlement_delay_seconds=0,
    )

    first = harness.run_once(now=entry_time)
    second = harness.run_once(now=exit_time)
    harness.run_exit_tracking_once(now=settle_time)
    report = _report(harness, settle_time)

    take_profit = report["profiles"][LIVE_LIKE_MODE]["confidence"]["exit_policies"][TAKE_PROFIT_EXIT_POLICY]
    hold = report["profiles"][LIVE_LIKE_MODE]["confidence"]["exit_policies"][HOLD_TO_SETTLEMENT_POLICY]
    position = report["take_profit_positions"][LIVE_LIKE_MODE]["confidence"][0]
    assert first["evaluations"][LIVE_LIKE_MODE]["confidence"]["status"] == "traded"
    assert second["take_profit_exits"] == 2
    assert take_profit["early_exits"] == 1
    assert take_profit["closed_trades"] == 1
    assert hold["settled_trades"] == 1
    assert position["status"] == "early_exited"
    assert position["hold_status"] == "won"
    assert position["delta_pnl_vs_hold"] == take_profit["delta_pnl_vs_hold"]
    assert take_profit["delta_pnl_vs_hold"] < 0
    assert take_profit["exit_reasons_by_reason"]


def test_take_profit_policy_falls_back_to_settlement_when_no_exit_triggers(tmp_path):
    market = _market()
    entry_time = market.window_start + timedelta(minutes=3)
    settle_time = market.window_end + timedelta(seconds=1)
    harness = Btc5mOpportunityHarness(
        profile_names="conservative",
        ledger_dir=tmp_path / "ledgers",
        market_client=_SequenceMarketClient([market, market]),
        price_client=_SequencePriceClient([100_090.0]),
        book_client=_SequenceBookClient([{"up": (0.54, 0.55), "down": (0.44, 0.46)}]),
        settlement_client=_FakeSettlementClient(winning_side="up"),
        settlement_delay_seconds=0,
    )

    harness.run_once(now=entry_time)
    harness.run_exit_tracking_once(now=settle_time)
    report = _report(harness, settle_time)

    take_profit = report["profiles"][LIVE_LIKE_MODE]["conservative"]["exit_policies"][TAKE_PROFIT_EXIT_POLICY]
    hold = report["profiles"][LIVE_LIKE_MODE]["conservative"]["exit_policies"][HOLD_TO_SETTLEMENT_POLICY]
    position = report["take_profit_positions"][LIVE_LIKE_MODE]["conservative"][0]
    assert take_profit["early_exits"] == 0
    assert take_profit["settlement_fallbacks"] == 1
    assert take_profit["realized_pnl"] == hold["realized_pnl"]
    assert take_profit["delta_pnl_vs_hold"] == 0
    assert position["status"] == "settled"


def test_full_run_tracks_take_profit_after_target_market_before_settlement(tmp_path):
    market = _market(datetime(2026, 6, 20, 20, 10, tzinfo=timezone.utc))
    now_fn = _SequenceNow(
        [
            market.window_start + timedelta(minutes=3),
            market.window_start + timedelta(minutes=3),
            market.window_start + timedelta(minutes=3, seconds=20),
            market.window_end + timedelta(seconds=1),
            market.window_end + timedelta(seconds=1),
        ]
    )

    result = run_btc5m_opportunity(
        profile_names="confidence",
        target_markets=1,
        poll_seconds=0,
        ledger_dir=tmp_path / "ledgers",
        report_path=tmp_path / "report.json",
        markdown_path=tmp_path / "report.md",
        market_client=_SequenceMarketClient([market, market, market]),
        price_client=_SequencePriceClient([100_090.0, 99_990.0]),
        book_client=_SequenceBookClient(
            [
                {"up": (0.72, 0.74), "down": (0.25, 0.27)},
                {"up": (0.50, 0.56), "down": (0.44, 0.50)},
            ]
        ),
        settlement_client=_FakeSettlementClient(winning_side="up"),
        settlement_delay_seconds=0,
        final_settlement_timeout_seconds=10,
        now_fn=now_fn,
        sleep_fn=lambda _: None,
        run_id="take-profit-after-target",
    )

    take_profit = result["profiles"][LIVE_LIKE_MODE]["confidence"]["exit_policies"][TAKE_PROFIT_EXIT_POLICY]
    assert result["status"] == "complete"
    assert take_profit["early_exits"] == 1
    assert take_profit["paired_hold_settled"] == 1


def test_opportunity_run_writes_json_markdown_and_requested_ledgers(tmp_path):
    market1 = _market(datetime(2026, 6, 20, 20, 10, tzinfo=timezone.utc), suffix="-a")
    market2 = _market(datetime(2026, 6, 20, 20, 15, tzinfo=timezone.utc), suffix="-b")
    run_dir = tmp_path / "opportunity-run"
    ledger_dir = run_dir / "ledgers"
    report_path = run_dir / "report.json"
    markdown_path = run_dir / "report.md"
    now_fn = _SequenceNow(
        [
            market1.window_start + timedelta(minutes=3),
            market1.window_start + timedelta(minutes=3),
            market2.window_start + timedelta(minutes=3),
            market2.window_start + timedelta(minutes=3),
        ]
    )

    result = run_btc5m_opportunity(
        profile_names="conservative",
        target_markets=2,
        poll_seconds=0,
        ledger_dir=ledger_dir,
        report_path=report_path,
        markdown_path=markdown_path,
        market_client=_SequenceMarketClient([market1, market2]),
        price_client=_StaticPriceClient(),
        book_client=_CountingBookClient(),
        settlement_client=_FakeSettlementClient(),
        settlement_delay_seconds=0,
        final_settlement_timeout_seconds=0,
        now_fn=now_fn,
        sleep_fn=lambda _: None,
        run_id="test-run",
    )

    assert result["status"] == "complete"
    assert result["distinct_markets_seen"] == 2
    assert report_path.exists()
    assert markdown_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "test-run"
    assert payload["profiles"][LIVE_LIKE_MODE]["conservative"]["markets_seen"] == 2
    assert "live_like" in markdown_path.read_text(encoding="utf-8")
    ledger_files = list(ledger_dir.rglob("*.json"))
    assert {path.relative_to(ledger_dir).as_posix() for path in ledger_files} == {
        "all_snapshots/conservative.json",
        "live_like/conservative.json",
    }
    assert not (tmp_path / "conservative.json").exists()


def test_opportunity_run_resumes_existing_report(tmp_path):
    market1 = _market(datetime(2026, 6, 20, 20, 10, tzinfo=timezone.utc), suffix="-a")
    market2 = _market(datetime(2026, 6, 20, 20, 15, tzinfo=timezone.utc), suffix="-b")
    run_dir = tmp_path / "resume-run"
    ledger_dir = run_dir / "ledgers"
    report_path = run_dir / "report.json"
    markdown_path = run_dir / "report.md"
    first_time = market1.window_start + timedelta(minutes=3)
    second_time = market2.window_start + timedelta(minutes=3)

    first = run_btc5m_opportunity(
        profile_names="conservative",
        target_markets=2,
        poll_seconds=0,
        ledger_dir=ledger_dir,
        report_path=report_path,
        markdown_path=markdown_path,
        market_client=_SequenceMarketClient([market1]),
        price_client=_StaticPriceClient(),
        book_client=_CountingBookClient(),
        settlement_client=_FakeSettlementClient(),
        settlement_delay_seconds=0,
        max_cycles=1,
        now_fn=_SequenceNow([first_time, first_time, first_time]),
        sleep_fn=lambda _: None,
        run_id="resume-test",
    )

    assert first["status"] == "stopped"
    assert first["distinct_markets_seen"] == 1
    first_started_at = first["started_at"]

    second = run_btc5m_opportunity(
        profile_names="conservative",
        target_markets=2,
        poll_seconds=0,
        ledger_dir=ledger_dir,
        report_path=report_path,
        markdown_path=markdown_path,
        market_client=_SequenceMarketClient([market2]),
        price_client=_StaticPriceClient(),
        book_client=_CountingBookClient(),
        settlement_client=_FakeSettlementClient(),
        settlement_delay_seconds=0,
        final_settlement_timeout_seconds=0,
        now_fn=_SequenceNow([second_time, second_time, second_time, second_time]),
        sleep_fn=lambda _: None,
        run_id="resume-test",
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert second["status"] == "complete"
    assert second["distinct_markets_seen"] == 2
    assert second["cycles"] == 2
    assert payload["started_at"] == first_started_at
    assert payload["profiles"][LIVE_LIKE_MODE]["conservative"]["markets_seen"] == 2
    assert len(payload["markets"]) == 2


def test_opportunity_run_attempts_final_settlement_after_target_market(tmp_path):
    market = _market(datetime(2026, 6, 20, 20, 10, tzinfo=timezone.utc))
    now_fn = _SequenceNow(
        [
            market.window_start + timedelta(minutes=3),
            market.window_start + timedelta(minutes=3),
            market.window_end + timedelta(seconds=1),
            market.window_end + timedelta(seconds=1),
        ]
    )

    result = run_btc5m_opportunity(
        profile_names="conservative",
        target_markets=1,
        poll_seconds=0,
        ledger_dir=tmp_path / "ledgers",
        report_path=tmp_path / "report.json",
        markdown_path=tmp_path / "report.md",
        market_client=_SequenceMarketClient([market]),
        price_client=_StaticPriceClient(),
        book_client=_CountingBookClient(),
        settlement_client=_FakeSettlementClient(),
        settlement_delay_seconds=0,
        final_settlement_timeout_seconds=0,
        now_fn=now_fn,
        sleep_fn=lambda _: None,
        run_id="final-settlement",
    )

    live = result["profiles"][LIVE_LIKE_MODE]["conservative"]
    assert result["status"] == "complete"
    assert live["settled_trades"] == 1
    assert live["wins"] == 1


def test_btc5m_opportunity_cli_parses_defaults_and_custom_args(monkeypatch, tmp_path):
    import src.bot as bot
    import src.polymarket.btc5m_opportunity as opportunity

    calls = []

    def fake_run_btc5m_opportunity(**kwargs):
        calls.append(kwargs)
        return {
            "status": "complete",
            "distinct_markets_seen": kwargs["target_markets"],
            "target_markets": kwargs["target_markets"],
            "report_path": str(kwargs["report_path"]),
            "ranking": {LIVE_LIKE_MODE: [], ALL_SNAPSHOTS_MODE: []},
        }

    monkeypatch.setattr(opportunity, "run_btc5m_opportunity", fake_run_btc5m_opportunity)
    monkeypatch.setattr(sys, "argv", ["bot.py", "btc5m-opportunity"])
    bot.main()

    assert calls[-1]["profile_names"] == "all"
    assert calls[-1]["target_markets"] == bot.BTC5M_OPPORTUNITY_TARGET_MARKETS
    assert calls[-1]["run_id"]
    assert Path(calls[-1]["ledger_dir"]).name == "ledgers"
    assert Path(calls[-1]["report_path"]).name == "report.json"
    assert Path(calls[-1]["markdown_path"]).name == "report.md"

    custom_ledger = tmp_path / "custom-ledgers"
    custom_report = tmp_path / "custom-report.json"
    custom_markdown = tmp_path / "custom-report.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bot.py",
            "btc5m-opportunity",
            "--profiles",
            "cheap_side,late",
            "--run-id",
            "custom-opportunity-run",
            "--target-markets",
            "12",
            "--poll-seconds",
            "0.5",
            "--settlement-source",
            "coinbase",
            "--ledger-dir",
            str(custom_ledger),
            "--report-path",
            str(custom_report),
            "--markdown-path",
            str(custom_markdown),
        ],
    )
    bot.main()

    assert calls[-1]["profile_names"] == "cheap_side,late"
    assert calls[-1]["run_id"] == "custom-opportunity-run"
    assert calls[-1]["target_markets"] == 12
    assert calls[-1]["poll_seconds"] == 0.5
    assert calls[-1]["settlement_source"] == "coinbase"
    assert calls[-1]["ledger_dir"] == custom_ledger
    assert calls[-1]["report_path"] == custom_report
    assert calls[-1]["markdown_path"] == custom_markdown
