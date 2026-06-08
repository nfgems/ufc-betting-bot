import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from src import betting_window
from src import config
from src.polymarket.executor import OrderExecutor
from src.polymarket.tracker import BetLedger
from src.strategy.bankroll import BankrollManager
from src.strategy import duo_trader


def _seed_live_bet(ledger: BetLedger, amount: float = 20.0) -> None:
    ledger.add_bet(
        fighter="Alpha",
        opponent="Beta",
        side="a",
        amount=amount,
        price=0.50,
        shares=amount / 0.50,
        token_id="token-1",
        market_id="market-1",
        model_prob=0.60,
        market_prob=0.50,
        edge=0.10,
        decimal_odds=2.0,
        dry_run=False,
        order_type="market",
    )


def test_resolve_total_bankroll_prefers_confirmed_clob_cash_in_dry_run(monkeypatch):
    monkeypatch.setattr(
        duo_trader,
        "_fetch_polymarket_account_state",
        lambda require_confirmed_cash=True, require_portfolio_value=True: {
            "cash_balance": 300.00,
            "portfolio_value": 312.34,
            "total_equity": 612.34,
            "cash_source": "clob",
            "portfolio_source": "data_api",
            "confirmed_cash": True,
            "confirmed_portfolio": True,
        },
    )

    basis = duo_trader._resolve_total_bankroll(dry_run=True)

    assert basis.total_equity == pytest.approx(612.34)
    assert basis.available_cash == pytest.approx(300.00)
    assert "Polymarket" in basis.source
    assert duo_trader._get_total_bankroll(dry_run=True) == pytest.approx(612.34)


def test_tracker_trader_uses_two_dollar_market_orders(tmp_path):
    trader = duo_trader._create_tracker_trader(
        "Model Tracker (M)",
        tmp_path / "tracker_ledger.json",
        clob=object(),
        dry_run=True,
        available_cash=25.0,
    )

    assert trader.executor.force_market_order is True
    assert trader.executor.force_limit_order is False
    assert trader.executor.min_edge_threshold == -1.0


def test_flat_trackers_use_fight_time_not_card_market_time(monkeypatch):
    now = datetime.now(timezone.utc)
    fight_time = now + timedelta(hours=2)
    card_market_time = now - timedelta(hours=1)
    row = {
        "fighter_a": "Alpha",
        "fighter_b": "Beta",
        "event_date": fight_time.isoformat(),
        "market_event_date": card_market_time.isoformat(),
        "card_date": "2026-06-14",
        "prob_a": 0.62,
        "prob_b": 0.38,
        "a_market_prob": 0.55,
        "b_market_prob": 0.45,
        "market_id": "market-1",
        "token_id_yes": "yes-1",
        "token_id_no": "no-1",
        "tick_size": "0.01",
        "neg_risk": False,
    }
    decisions = []
    monkeypatch.setattr(
        "src.strategy.llm_operator.log_tracker_decision",
        lambda record: decisions.append(record),
    )
    monkeypatch.setattr(
        "src.strategy.llm_operator.gemini_standalone_pick",
        lambda **kwargs: {
            "pick": "Beta",
            "confidence": 0.57,
            "rationale": "Gemini pick",
            "sources": [],
        },
    )

    model_bets = duo_trader.find_flat_model_bets(pd.DataFrame([row]))
    gemini_bets = duo_trader.find_flat_gemini_bets(pd.DataFrame([row]))

    assert len(model_bets) == 1
    assert len(gemini_bets) == 1
    assert model_bets.iloc[0]["override_bet_size"] == pytest.approx(2.0)
    assert gemini_bets.iloc[0]["override_bet_size"] == pytest.approx(2.0)
    assert model_bets.iloc[0]["event_date"] == fight_time.isoformat()
    assert model_bets.iloc[0]["market_event_date"] == card_market_time.isoformat()
    assert model_bets.iloc[0]["card_date"] == "2026-06-14"
    assert gemini_bets.iloc[0]["card_date"] == "2026-06-14"
    assert [record["status"] for record in decisions] == ["eligible", "eligible"]
    assert [record["card_date"] for record in decisions] == ["2026-06-14", "2026-06-14"]


def test_flat_gemini_tracker_keeps_confidence_separate_from_probability(monkeypatch):
    row = {
        "fighter_a": "Alpha",
        "fighter_b": "Beta",
        "event_date": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
        "prob_a": 0.62,
        "prob_b": 0.38,
        "a_market_prob": 0.55,
        "b_market_prob": 0.45,
        "market_id": "market-1",
        "token_id_yes": "yes-1",
        "token_id_no": "no-1",
    }
    records = []
    monkeypatch.setattr(
        "src.strategy.llm_operator.log_tracker_decision",
        lambda record: records.append(record),
    )
    monkeypatch.setattr(
        "src.strategy.llm_operator.gemini_standalone_pick",
        lambda **kwargs: {
            "pick": "Beta",
            "confidence": 1.0,
            "rationale": "Gemini pick",
            "sources": [],
        },
    )

    gemini_bets = duo_trader.find_flat_gemini_bets(pd.DataFrame([row]))

    assert len(gemini_bets) == 1
    assert gemini_bets.iloc[0]["model_prob"] == pytest.approx(0.45)
    assert gemini_bets.iloc[0]["blended_prob"] == pytest.approx(0.45)
    assert gemini_bets.iloc[0]["edge"] == pytest.approx(0.0)
    assert gemini_bets.iloc[0]["signal_confidence"] == pytest.approx(config.GEMINI_TRACKER_CONFIDENCE_CAP)
    assert gemini_bets.iloc[0]["probability_source"] == "market_neutral"
    assert records[0]["confidence"] == pytest.approx(config.GEMINI_TRACKER_CONFIDENCE_CAP)
    assert records[0]["signal_confidence"] == pytest.approx(config.GEMINI_TRACKER_CONFIDENCE_CAP)
    assert "edge" not in records[0]


def test_log_unmatched_tracker_decisions_records_no_market(monkeypatch):
    predictions = pd.DataFrame(
        [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "event_date": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
            }
        ]
    )
    records = []
    monkeypatch.setattr(
        "src.strategy.llm_operator.log_tracker_decision",
        lambda record: records.append(record),
    )

    duo_trader._log_unmatched_tracker_decisions(
        trader="M",
        predictions=predictions,
        matched_predictions=pd.DataFrame(),
        event_title="Test card",
    )

    assert len(records) == 1
    assert records[0]["status"] == "no_market"
    assert records[0]["summary"] == "No market matched"


def test_tracker_live_market_order_allows_negative_edge_flat_bet(tmp_path):
    class _FakeClob:
        def get_orderbook(self, token_id):
            return {"asks": [{"price": 0.80, "size": 1000}]}

        def create_limit_order(self, **kwargs):
            return {"orderID": "order-1"}

    bankroll = BankrollManager(
        initial_bankroll=100.0,
        total_equity=100.0,
        available_cash=100.0,
        kelly_fraction=1.0,
        max_bet_fraction=1.0,
        auto_detect_balance=False,
    )
    executor = OrderExecutor(
        bankroll=bankroll,
        clob_client=_FakeClob(),
        dry_run=False,
        min_edge_threshold=-1.0,
        skip_wallet_conflict_check=True,
        force_market_order=True,
    )
    executor.ledger = BetLedger(path=tmp_path / "tracker.json")
    bet = pd.Series(
        {
            "fighter_a": "Alpha",
            "fighter_b": "Beta",
            "bet_on": "Alpha",
            "bet_side": "a",
            "model_prob": 0.20,
            "blended_prob": 0.20,
            "market_prob": 0.80,
            "edge": -0.60,
            "decimal_odds": 1.25,
            "override_bet_size": 2.0,
            "event_date": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
            "token_id_yes": "yes-1",
            "token_id_no": "no-1",
            "market_id": "market-1",
            "condition_id": "condition-1",
            "tick_size": "0.01",
            "neg_risk": False,
            "signal_confidence": 0.85,
            "signal_source": "gemini_research",
            "probability_source": "market_neutral",
            "card_date": "2026-06-14",
        }
    )

    order = executor._place_bet(bet, pd.DataFrame())

    assert order["status"] == "placed"
    assert order["order_type"] == "marketable_limit"
    assert order["bet_size_usd"] >= 2.0
    ledger_bet = executor.ledger.get_bets(fresh=True)[0]
    assert ledger_bet["fighter"] == "Alpha"
    assert ledger_bet["signal_confidence"] == 0.85
    assert ledger_bet["probability_source"] == "market_neutral"
    assert ledger_bet["card_date"] == "2026-06-14"


def test_resolve_total_bankroll_falls_back_only_in_dry_run(monkeypatch):
    monkeypatch.setattr(
        duo_trader,
        "_fetch_polymarket_account_state",
        lambda require_confirmed_cash=True, require_portfolio_value=True: {
            "cash_balance": 0.0,
            "portfolio_value": 0.0,
            "total_equity": 0.0,
            "cash_source": "unavailable",
            "portfolio_source": "unavailable",
            "confirmed_cash": False,
            "confirmed_portfolio": False,
        },
    )
    monkeypatch.setattr(config, "INITIAL_BANKROLL", 777.0)

    basis = duo_trader._resolve_total_bankroll(dry_run=True)

    assert basis.total_equity == pytest.approx(777.0)
    assert basis.available_cash == pytest.approx(777.0)
    assert "INITIAL_BANKROLL fallback" in basis.source


def test_resolve_total_bankroll_live_rejects_unconfirmed_cash(monkeypatch):
    monkeypatch.setattr(
        duo_trader,
        "_fetch_polymarket_account_state",
        lambda require_confirmed_cash=True, require_portfolio_value=True: {
            "cash_balance": 0.0,
            "portfolio_value": 0.0,
            "total_equity": 0.0,
            "cash_source": "unavailable",
            "portfolio_source": "unavailable",
            "confirmed_cash": False,
            "confirmed_portfolio": False,
        },
    )

    with pytest.raises(RuntimeError, match="wallet cash balance"):
        duo_trader._resolve_total_bankroll(dry_run=False)


def test_resolve_total_bankroll_live_uses_confirmed_cash_when_portfolio_unavailable(monkeypatch):
    monkeypatch.setattr(
        duo_trader,
        "_fetch_polymarket_account_state",
        lambda require_confirmed_cash=True, require_portfolio_value=True: {
            "cash_balance": 157.82,
            "portfolio_value": 0.0,
            "total_equity": 0.0,
            "cash_source": "clob",
            "portfolio_source": "unavailable",
            "confirmed_cash": True,
            "confirmed_portfolio": False,
        },
    )

    basis = duo_trader._resolve_total_bankroll(dry_run=False)

    assert basis.total_equity == pytest.approx(157.82)
    assert basis.available_cash == pytest.approx(157.82)
    assert "cash-only" in basis.source


def test_resolve_total_bankroll_live_accepts_confirmed_zero_cash(monkeypatch):
    monkeypatch.setattr(
        duo_trader,
        "_fetch_polymarket_account_state",
        lambda require_confirmed_cash=True, require_portfolio_value=True: {
            "cash_balance": 0.0,
            "portfolio_value": 0.0,
            "total_equity": 0.0,
            "cash_source": "clob",
            "portfolio_source": "data_api",
            "confirmed_cash": True,
            "confirmed_portfolio": True,
        },
    )

    basis = duo_trader._resolve_total_bankroll(dry_run=False)

    assert basis.total_equity == pytest.approx(0.0)
    assert basis.available_cash == pytest.approx(0.0)
    assert "Polymarket" in basis.source


def test_resolve_cash_after_order_groups_caps_stale_live_cash(monkeypatch, caplog):
    monkeypatch.setattr(
        duo_trader,
        "_fetch_polymarket_account_state",
        lambda require_confirmed_cash=True, require_portfolio_value=False: {
            "cash_balance": 37.29,
            "confirmed_cash": True,
        },
    )

    orders = [
        {"status": "placed", "bet_size_usd": 61.07},
        {"status": "placed", "bet_size_usd": 58.0669},
        {"status": "placed", "bet_size_usd": 32.810375},
        {"status": "failed", "bet_size_usd": 99.0},
    ]

    with caplog.at_level(logging.WARNING, logger="src.strategy.duo_trader"):
        remaining = duo_trader._resolve_cash_after_order_groups(
            starting_cash=157.82,
            order_groups=(orders,),
            dry_run=False,
            label="Model Tracker",
        )

    assert remaining == pytest.approx(5.872725)
    assert "exceeds internally reserved remaining cash" in caplog.text


def test_resolve_cash_after_order_groups_logs_small_stale_cash_gap_below_warning(
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(
        duo_trader,
        "_fetch_polymarket_account_state",
        lambda require_confirmed_cash=True, require_portfolio_value=False: {
            "cash_balance": 1383.15,
            "confirmed_cash": True,
        },
    )

    with caplog.at_level(logging.WARNING, logger="src.strategy.duo_trader"):
        remaining = duo_trader._resolve_cash_after_order_groups(
            starting_cash=1383.15,
            order_groups=([{"status": "placed", "bet_size_usd": 2.01}],),
            dry_run=False,
            label="Gemini Tracker",
        )

    assert remaining == pytest.approx(1381.14)
    assert "exceeds internally reserved remaining cash" not in caplog.text


def test_create_trader_does_not_replay_live_ledger_by_default(tmp_path):
    ledger_path = tmp_path / "single.json"
    ledger = BetLedger(path=ledger_path)
    _seed_live_bet(ledger, amount=20.0)

    profile = duo_trader._create_trader(
        duo_trader.TraderProfile(
            name="Single Trader (S, blend=0.30)",
            blend_weight=0.30,
            ledger_path=str(ledger_path),
        ),
        allocation=100.0,
        available_cash=60.0,
        clob=SimpleNamespace(),
        dry_run=False,
    )

    assert profile.bankroll.bankroll == pytest.approx(60.0)
    assert profile.bankroll.total_equity == pytest.approx(100.0)


def test_create_trader_can_opt_in_to_ledger_replay(tmp_path):
    ledger_path = tmp_path / "single.json"
    ledger = BetLedger(path=ledger_path)
    _seed_live_bet(ledger, amount=20.0)

    profile = duo_trader._create_trader(
        duo_trader.TraderProfile(
            name="Single Trader (S, blend=0.30)",
            blend_weight=0.30,
            ledger_path=str(ledger_path),
        ),
        allocation=100.0,
        available_cash=100.0,
        clob=SimpleNamespace(),
        dry_run=False,
        sync_from_ledger=True,
    )

    assert profile.bankroll.bankroll == pytest.approx(80.0)
    assert profile.bankroll.total_equity == pytest.approx(100.0)


def test_bankroll_manager_sizes_from_total_equity_but_caps_spend_by_cash():
    bankroll = BankrollManager(
        initial_bankroll=600.0,
        total_equity=600.0,
        available_cash=300.0,
        kelly_fraction=1.0,
        max_bet_fraction=1.0,
        auto_detect_balance=False,
    )

    # At 2.0 odds and 75% win probability, full Kelly is 50% of bankroll.
    # This should use total equity ($600), not free cash ($300).
    assert bankroll.kelly_bet_size(0.75, 2.0) == pytest.approx(300.0)
    assert bankroll.place_bet(350.0, "Alpha", 2.0, 0.75, 0.50) == {}
    assert bankroll.bankroll == pytest.approx(300.0)
    assert bankroll.total_equity == pytest.approx(600.0)


def test_executor_skips_order_before_submit_when_cash_is_insufficient(tmp_path):
    class _NeverShouldSubmit:
        def __init__(self):
            self.market_calls = 0
            self.limit_calls = 0

        def get_open_orders(self):
            return []

        def get_orderbook(self, _token_id):
            return {"asks": [{"price": "0.50", "size": "10000"}], "bids": []}

        def create_market_order(self, **kwargs):
            self.market_calls += 1
            return {"orderID": "should-not-happen"}

        def create_limit_order(self, **kwargs):
            self.limit_calls += 1
            return {"orderID": "limit-1"}

    clob = _NeverShouldSubmit()
    bankroll = BankrollManager(
        initial_bankroll=600.0,
        total_equity=600.0,
        available_cash=300.0,
        auto_detect_balance=False,
    )
    executor = OrderExecutor(bankroll=bankroll, clob_client=clob, dry_run=False)
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")

    bet = pd.Series(
        {
            "fighter_a": "Alpha",
            "fighter_b": "Beta",
            "bet_on": "Alpha",
            "model_prob": 0.70,
            "blended_prob": 0.70,
            "market_prob": 0.50,
            "edge": 0.20,
            "decimal_odds": 2.0,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "token_id_no": "token-no",
            "market_id": "market-1",
            "tick_size": "0.01",
            "neg_risk": False,
            "override_bet_size": 350.0,
        }
    )

    # override_bet_size (350) is now capped by MAX_BET_FRACTION (4% of 600 = 24).
    # The capped size (24) fits within available_cash (300), so the bet proceeds.
    result = executor._place_bet(bet, pd.DataFrame())
    assert result is not None
    assert result["bet_size_usd"] == pytest.approx(24.0)
    assert result["order_type"] == "marketable_limit"
    assert clob.market_calls == 0
    assert clob.limit_calls == 1


def test_executor_skips_sub_dollar_market_buy_before_submit(tmp_path):
    class _NeverShouldSubmit:
        def __init__(self):
            self.market_calls = 0
            self.limit_calls = 0

        def get_open_orders(self):
            return []

        def get_orderbook(self, _token_id):
            return {"asks": [{"price": "0.50", "size": "10000"}], "bids": []}

        def create_market_order(self, **kwargs):
            self.market_calls += 1
            return {"orderID": "should-not-happen"}

        def create_limit_order(self, **kwargs):
            self.limit_calls += 1
            return {"orderID": "limit-1"}

    clob = _NeverShouldSubmit()
    bankroll = BankrollManager(
        initial_bankroll=0.04,
        total_equity=0.04,
        available_cash=0.04,
        max_bet_fraction=1.0,
        auto_detect_balance=False,
    )
    executor = OrderExecutor(bankroll=bankroll, clob_client=clob, dry_run=False)
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")

    bet = pd.Series(
        {
            "fighter_a": "Alpha",
            "fighter_b": "Beta",
            "bet_on": "Alpha",
            "model_prob": 0.70,
            "blended_prob": 0.70,
            "market_prob": 0.50,
            "edge": 0.20,
            "decimal_odds": 2.0,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "token_id_no": "token-no",
            "market_id": "market-1",
            "tick_size": "0.01",
            "neg_risk": False,
            "override_bet_size": 1.0,
        }
    )

    result = executor._place_bet(bet, pd.DataFrame())

    assert result is None
    assert clob.market_calls == 0
    assert clob.limit_calls == 0
    assert executor.ledger.bets == []


def test_executor_skips_sub_dollar_limit_bid_before_submit(tmp_path):
    class _NeverShouldSubmit:
        def __init__(self):
            self.market_calls = 0
            self.limit_calls = 0

        def get_open_orders(self):
            return []

        def get_orderbook(self, _token_id):
            return {"asks": [{"price": "0.50", "size": "10000"}], "bids": []}

        def create_market_order(self, **kwargs):
            self.market_calls += 1
            return {"orderID": "should-not-happen"}

        def create_limit_order(self, **kwargs):
            self.limit_calls += 1
            return {"orderID": "should-not-happen"}

    clob = _NeverShouldSubmit()
    bankroll = BankrollManager(
        initial_bankroll=0.04,
        total_equity=0.04,
        available_cash=0.04,
        max_bet_fraction=1.0,
        auto_detect_balance=False,
    )
    executor = OrderExecutor(
        bankroll=bankroll,
        clob_client=clob,
        dry_run=False,
        skip_wallet_conflict_check=True,
        force_limit_order=True,
    )
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")

    bet = pd.Series(
        {
            "fighter_a": "Alpha",
            "fighter_b": "Beta",
            "bet_on": "Alpha",
            "model_prob": 0.70,
            "blended_prob": 0.70,
            "market_prob": 0.50,
            "edge": 0.20,
            "decimal_odds": 2.0,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "token_id_no": "token-no",
            "market_id": "market-1",
            "tick_size": "0.01",
            "neg_risk": False,
            "override_bet_size": 1.0,
        }
    )

    result = executor._place_bet(bet, pd.DataFrame())

    assert result is None
    assert clob.market_calls == 0
    assert clob.limit_calls == 0
    assert executor.ledger.bets == []


def test_executor_skips_sub_dollar_near_miss_limit_before_submit(tmp_path):
    class _NeverShouldSubmit:
        def __init__(self):
            self.limit_calls = 0

        def get_open_orders(self):
            return []

        def create_limit_order(self, **kwargs):
            self.limit_calls += 1
            return {"orderID": "should-not-happen"}

    clob = _NeverShouldSubmit()
    bankroll = BankrollManager(
        initial_bankroll=500.0,
        total_equity=500.0,
        available_cash=500.0,
        auto_detect_balance=False,
    )
    executor = OrderExecutor(bankroll=bankroll, clob_client=clob, dry_run=False)
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")
    executor._authoritative_wallet_conflict = lambda **kwargs: (False, "")

    bet = pd.Series(
        {
            "fighter_a": "Alpha",
            "fighter_b": "Beta",
            "bet_on": "Alpha",
            "model_prob": 0.63,
            "blended_prob": 0.63,
            "market_prob": 0.62,
            "edge": 0.01,
            "decimal_odds": 1.6129,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "token_id_no": "token-no",
            "market_id": "market-1",
            "tick_size": "0.01",
            "neg_risk": False,
            "size_multiplier": 0.01,
        }
    )

    result = executor._place_near_miss_limit(bet, pd.DataFrame())

    assert result is None
    assert clob.limit_calls == 0
    assert executor.ledger.bets == []


def test_executor_blocks_resting_limit_bid_inside_two_hour_pull_window(monkeypatch, tmp_path):
    now = datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(betting_window, "_current_utc", lambda: now)

    class _NoLimitSubmit:
        def __init__(self):
            self.limit_calls = 0

        def create_limit_order(self, **kwargs):
            self.limit_calls += 1
            return {"orderID": "should-not-happen"}

    clob = _NoLimitSubmit()
    bankroll = BankrollManager(
        initial_bankroll=500.0,
        total_equity=500.0,
        available_cash=500.0,
        auto_detect_balance=False,
    )
    executor = OrderExecutor(
        bankroll=bankroll,
        clob_client=clob,
        dry_run=False,
        min_edge_threshold=0.05,
        skip_wallet_conflict_check=True,
    )
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")
    executor._check_liquidity = lambda *args, **kwargs: {
        "ok": True,
        "best_ask": 0.62,
        "best_ask_liquidity": 100.0,
    }

    bet = pd.Series(
        {
            "fighter_a": "Alpha",
            "fighter_b": "Beta",
            "bet_on": "Alpha",
            "model_prob": 0.66,
            "blended_prob": 0.66,
            "market_prob": 0.58,
            "edge": 0.08,
            "decimal_odds": 1.7241,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "token_id_no": "token-no",
            "market_id": "market-1",
            "tick_size": "0.01",
            "neg_risk": False,
            "override_bet_size": 10.0,
            "event_date": (now + timedelta(minutes=90)).isoformat(),
        }
    )

    result = executor._place_bet(bet, pd.DataFrame())

    assert result is None
    assert clob.limit_calls == 0
    assert executor.ledger.bets == []


def test_executor_allows_marketable_order_inside_two_hour_pull_window(monkeypatch, tmp_path):
    now = datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(betting_window, "_current_utc", lambda: now)

    class _MarketableSubmit:
        def __init__(self):
            self.limit_calls = 0

        def create_limit_order(self, **kwargs):
            self.limit_calls += 1
            return {"orderID": "marketable-order"}

    clob = _MarketableSubmit()
    bankroll = BankrollManager(
        initial_bankroll=500.0,
        total_equity=500.0,
        available_cash=500.0,
        auto_detect_balance=False,
    )
    executor = OrderExecutor(
        bankroll=bankroll,
        clob_client=clob,
        dry_run=False,
        min_edge_threshold=0.05,
        skip_wallet_conflict_check=True,
    )
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")
    executor._check_liquidity = lambda *args, **kwargs: {
        "ok": True,
        "best_ask": 0.50,
        "best_ask_liquidity": 100.0,
    }

    bet = pd.Series(
        {
            "fighter_a": "Alpha",
            "fighter_b": "Beta",
            "bet_on": "Alpha",
            "model_prob": 0.70,
            "blended_prob": 0.70,
            "market_prob": 0.50,
            "edge": 0.20,
            "decimal_odds": 2.0,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "token_id_no": "token-no",
            "market_id": "market-1",
            "tick_size": "0.01",
            "neg_risk": False,
            "override_bet_size": 10.0,
            "event_date": (now + timedelta(minutes=90)).isoformat(),
        }
    )

    result = executor._place_bet(bet, pd.DataFrame())

    assert result is not None
    assert result["status"] == "placed"
    assert result["order_type"] == "marketable_limit"
    assert clob.limit_calls == 1
    assert len(executor.ledger.open_bets) == 1


def test_executor_blocks_marketable_remainder_inside_two_hour_pull_window(monkeypatch, tmp_path):
    now = datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(betting_window, "_current_utc", lambda: now)

    class _NoSubmit:
        def __init__(self):
            self.limit_calls = 0

        def create_limit_order(self, **kwargs):
            self.limit_calls += 1
            return {"orderID": "should-not-happen"}

    clob = _NoSubmit()
    bankroll = BankrollManager(
        initial_bankroll=500.0,
        total_equity=500.0,
        available_cash=500.0,
        auto_detect_balance=False,
    )
    executor = OrderExecutor(
        bankroll=bankroll,
        clob_client=clob,
        dry_run=False,
        min_edge_threshold=0.05,
        skip_wallet_conflict_check=True,
    )
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")
    executor._check_liquidity = lambda *args, **kwargs: {
        "ok": True,
        "best_ask": 0.50,
        "best_ask_liquidity": 5.0,
    }

    bet = pd.Series(
        {
            "fighter_a": "Alpha",
            "fighter_b": "Beta",
            "bet_on": "Alpha",
            "model_prob": 0.70,
            "blended_prob": 0.70,
            "market_prob": 0.50,
            "edge": 0.20,
            "decimal_odds": 2.0,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "token_id_no": "token-no",
            "market_id": "market-1",
            "tick_size": "0.01",
            "neg_risk": False,
            "override_bet_size": 10.0,
            "event_date": (now + timedelta(minutes=90)).isoformat(),
        }
    )

    result = executor._place_bet(bet, pd.DataFrame())

    assert result is None
    assert clob.limit_calls == 0
    assert executor.ledger.bets == []


def test_executor_blocks_near_miss_limit_inside_two_hour_pull_window(monkeypatch, tmp_path):
    now = datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(betting_window, "_current_utc", lambda: now)

    class _NoLimitSubmit:
        def __init__(self):
            self.limit_calls = 0

        def create_limit_order(self, **kwargs):
            self.limit_calls += 1
            return {"orderID": "should-not-happen"}

    clob = _NoLimitSubmit()
    bankroll = BankrollManager(
        initial_bankroll=500.0,
        total_equity=500.0,
        available_cash=500.0,
        auto_detect_balance=False,
    )
    executor = OrderExecutor(bankroll=bankroll, clob_client=clob, dry_run=False)
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")

    bet = pd.Series(
        {
            "fighter_a": "Alpha",
            "fighter_b": "Beta",
            "bet_on": "Alpha",
            "model_prob": 0.63,
            "blended_prob": 0.63,
            "market_prob": 0.62,
            "edge": 0.01,
            "decimal_odds": 1.6129,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "token_id_no": "token-no",
            "market_id": "market-1",
            "tick_size": "0.01",
            "neg_risk": False,
            "event_date": (now + timedelta(minutes=90)).isoformat(),
        }
    )

    result = executor._place_near_miss_limit(bet, pd.DataFrame())

    assert result is None
    assert clob.limit_calls == 0
    assert executor.ledger.bets == []


def test_run_duo_traders_skips_value_bets_outside_live_bet_window(monkeypatch):
    now = datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc)
    inside_window = (now + timedelta(hours=24)).isoformat()
    inside_new_bet_window_but_inside_limit_pull_window = (now + timedelta(minutes=90)).isoformat()
    inside_limit_bid_window = (now + timedelta(hours=3)).isoformat()
    outside_window = (now + timedelta(days=4)).isoformat()
    monkeypatch.setattr(betting_window, "_current_utc", lambda: now)
    monkeypatch.setattr("src.strategy.llm_operator.OPERATOR_ENABLED", False)

    class _FakeBankroll:
        def __init__(self, bankroll):
            self.bankroll = bankroll
            self.is_stopped = False

        def get_stats(self):
            return {"bankroll": self.bankroll}

    class _FakeExecutor:
        def __init__(self):
            self.ledger = SimpleNamespace(open_bets=[])
            self.placed = []

        def _match_predictions_to_markets(self, predictions, markets):
            return pd.DataFrame()

        def refresh_open_limit_orders(self, **kwargs):
            return {}

        def _place_bet(self, bet, markets):
            self.placed.append(dict(bet))
            return {"bet_size_usd": 10.0}

        def _place_near_miss_limit(self, bet, markets):
            self.placed.append(dict(bet))
            return {"bet_size_usd": 10.0}

    single_exec = _FakeExecutor()
    conv_exec = _FakeExecutor()
    tracker_exec = _FakeExecutor()

    single = SimpleNamespace(
        name="Single Trader (S, blend=0.30)",
        bankroll=_FakeBankroll(100.0),
        executor=single_exec,
    )
    conv = SimpleNamespace(
        name="Conviction Trader (C)",
        bankroll=_FakeBankroll(100.0),
        executor=conv_exec,
    )

    created = iter([single, conv])

    monkeypatch.setattr(
        duo_trader,
        "_resolve_total_bankroll",
        lambda dry_run=True: duo_trader.WalletBankrollBasis(100.0, 100.0, "test"),
    )
    monkeypatch.setattr(duo_trader, "_create_trader", lambda *args, **kwargs: next(created))
    monkeypatch.setattr(
        duo_trader,
        "find_value_bets",
        lambda *args, **kwargs: (
            pd.DataFrame(
                [
                    {
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "bet_on": "Alpha",
                        "bet_side": "a",
                        "model_prob": 0.70,
                        "blended_prob": 0.70,
                        "market_prob": 0.55,
                        "edge": 0.15,
                        "decimal_odds": 1.82,
                        "event_date": inside_window,
                    },
                    {
                        "fighter_a": "Gamma",
                        "fighter_b": "Delta",
                        "bet_on": "Gamma",
                        "bet_side": "a",
                        "model_prob": 0.69,
                        "blended_prob": 0.69,
                        "market_prob": 0.54,
                        "edge": 0.15,
                        "decimal_odds": 1.85,
                        "event_date": outside_window,
                    },
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "fighter_a": "Near",
                        "fighter_b": "Closed",
                        "bet_on": "Near",
                        "bet_side": "a",
                        "model_prob": 0.63,
                        "blended_prob": 0.63,
                        "market_prob": 0.62,
                        "edge": 0.01,
                        "decimal_odds": 1.61,
                        "event_date": inside_new_bet_window_but_inside_limit_pull_window,
                    },
                    {
                        "fighter_a": "Near",
                        "fighter_b": "Open",
                        "bet_on": "Near Open",
                        "bet_side": "a",
                        "model_prob": 0.63,
                        "blended_prob": 0.63,
                        "market_prob": 0.62,
                        "edge": 0.01,
                        "decimal_odds": 1.61,
                        "event_date": inside_limit_bid_window,
                    },
                ]
            ),
        ),
    )
    monkeypatch.setattr(duo_trader, "find_conviction_bets", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        duo_trader,
        "_create_tracker_trader",
        lambda *args, **kwargs: SimpleNamespace(
            name="Tracker",
            bankroll=_FakeBankroll(100.0),
            executor=tracker_exec,
        ),
    )
    monkeypatch.setattr(duo_trader, "find_flat_model_bets", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(duo_trader, "find_flat_gemini_bets", lambda *args, **kwargs: pd.DataFrame())

    result = duo_trader.run_duo_traders(
        predictions=pd.DataFrame(),
        markets=pd.DataFrame(),
        clob=None,
        dry_run=True,
    )

    assert [bet["bet_on"] for bet in single_exec.placed] == ["Alpha", "Near Open"]
    assert result["total_orders"] == 2
