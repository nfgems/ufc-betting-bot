from datetime import datetime, timedelta, timezone

import pandas as pd

from src.polymarket.executor import OrderExecutor
from src.polymarket.tracker import BetLedger
from src.strategy.bankroll import BankrollManager


class _FakeClob:
    def __init__(
        self,
        *,
        open_orders=None,
        closed_orders=None,
        trades=None,
        open_orders_error=None,
    ):
        self._open_orders = list(open_orders or [])
        self._closed_orders = dict(closed_orders or {})
        self._trades = list(trades or [])
        self._open_orders_error = open_orders_error
        self.limit_calls = 0

    def get_open_orders(self):
        if self._open_orders_error is not None:
            raise self._open_orders_error
        return list(self._open_orders)

    def get_order(self, order_id):
        if order_id not in self._closed_orders:
            raise KeyError(order_id)
        return self._closed_orders[order_id]

    def get_trades(self, params=None):
        return list(self._trades)

    def create_limit_order(self, **kwargs):
        self.limit_calls += 1
        return {"orderID": "new-order"}


def _make_executor(tmp_path, fake_clob, *, bankroll_value=100.0):
    bankroll = BankrollManager(
        initial_bankroll=bankroll_value,
        auto_detect_balance=False,
    )
    bankroll.bankroll = bankroll_value
    executor = OrderExecutor(
        bankroll=bankroll,
        clob_client=fake_clob,
        dry_run=False,
    )
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")
    executor._get_live_positions_cached = lambda **kwargs: []
    return executor


def _seed_limit_bet(ledger: BetLedger):
    ledger.add_bet(
        fighter="Kelvin Gastelum",
        opponent="Vicente Luque",
        side="a",
        amount=20.0,
        price=0.58,
        shares=34.48,
        token_id="token-yes",
        market_id="1721390",
        model_prob=0.64,
        market_prob=0.62,
        edge=0.02,
        decimal_odds=1.6129,
        dry_run=False,
        event_date=(datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
        order_type="near_miss_limit",
        order_id="order-1",
    )
    ledger.bets[-1]["placed_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=60)
    ).isoformat()
    ledger._save()


def test_refresh_keeps_limit_order_when_closed_lookup_has_no_positive_evidence(tmp_path):
    fake_clob = _FakeClob(open_orders=[], trades=[])
    executor = _make_executor(tmp_path, fake_clob, bankroll_value=80.0)
    _seed_limit_bet(executor.ledger)

    summary = executor.refresh_open_limit_orders(
        matched_predictions=pd.DataFrame(),
        primary_bets=pd.DataFrame(),
        limit_only_bets=pd.DataFrame(),
        trader_name="Single Trader",
    )

    assert summary["kept"] == 1
    assert summary["cancelled"] == 0
    assert summary["reconciled"] == 0
    assert len(executor.ledger.open_bets) == 1
    assert executor.ledger.bets[0]["status"] == "open"
    assert executor.ledger.bets[0]["order_id"] == "order-1"
    assert executor.bankroll.bankroll == 80.0


def test_limit_bid_skips_when_same_token_order_is_already_resting_on_clob(tmp_path):
    fake_clob = _FakeClob(
        open_orders=[
            {
                "id": "resting-1",
                "asset_id": "token-yes",
                "price": "0.58",
                "original_size": "34.48",
                "size_matched": "0",
            }
        ]
    )
    executor = _make_executor(tmp_path, fake_clob)
    executor._check_liquidity = lambda *args, **kwargs: {
        "ok": True,
        "adjusted_size": 25.0,
        "available_liquidity": 100.0,
        "slippage": 0.0,
        "best_ask": 0.63,
        "reason": "",
    }

    bet = pd.Series(
        {
            "fighter_a": "Kelvin Gastelum",
            "fighter_b": "Vicente Luque",
            "bet_on": "Kelvin Gastelum",
            "model_prob": 0.64,
            "blended_prob": 0.64,
            "market_prob": 0.60,
            "edge": 0.04,
            "decimal_odds": 1.6667,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "token_id_no": "token-no",
            "market_id": "1721390",
            "tick_size": "0.01",
            "override_bet_size": 25.0,
        }
    )

    result = executor._place_bet(bet, pd.DataFrame())

    assert result is None
    assert fake_clob.limit_calls == 0
    assert executor.ledger.open_bets == []
    assert executor.bankroll.bankroll == 100.0


def test_near_miss_skips_when_clob_state_cannot_be_verified(tmp_path):
    fake_clob = _FakeClob(open_orders_error=RuntimeError("temporary CLOB outage"))
    executor = _make_executor(tmp_path, fake_clob)

    bet = pd.Series(
        {
            "fighter_a": "Kelvin Gastelum",
            "fighter_b": "Vicente Luque",
            "bet_on": "Kelvin Gastelum",
            "model_prob": 0.72,
            "blended_prob": 0.72,
            "market_prob": 0.70,
            "edge": 0.02,
            "decimal_odds": 1.4286,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "token_id_no": "token-no",
            "market_id": "1721390",
            "tick_size": "0.01",
        }
    )

    result = executor._place_near_miss_limit(bet, pd.DataFrame())

    assert result is None
    assert fake_clob.limit_calls == 0
    assert executor.ledger.open_bets == []
    assert executor.bankroll.bankroll == 100.0
