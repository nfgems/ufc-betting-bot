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


def test_tracker_trader_uses_one_dollar_market_orders(tmp_path):
    trader = duo_trader._create_tracker_trader(
        "Model Tracker (M)",
        tmp_path / "tracker_ledger.json",
        clob=object(),
        dry_run=True,
        available_cash=25.0,
    )

    assert trader.executor.force_market_order is True
    assert trader.executor.force_limit_order is False
    assert trader.executor.min_edge_threshold == 0.0


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

    with pytest.raises(RuntimeError, match="wallet cash balance and portfolio value"):
        duo_trader._resolve_total_bankroll(dry_run=False)


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

        def get_open_orders(self):
            return []

        def get_orderbook(self, _token_id):
            return {"asks": [{"price": "0.50", "size": "10000"}], "bids": []}

        def create_market_order(self, **kwargs):
            self.market_calls += 1
            return {"orderID": "should-not-happen"}

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
    assert clob.market_calls == 1


def test_run_duo_traders_skips_value_bets_outside_live_bet_window(monkeypatch):
    now = datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc)
    inside_window = (now + timedelta(hours=24)).isoformat()
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
            pd.DataFrame(),
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

    assert [bet["bet_on"] for bet in single_exec.placed] == ["Alpha"]
    assert result["total_orders"] == 1
