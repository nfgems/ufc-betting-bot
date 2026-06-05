import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd

from src.config import LIMIT_BID_TTL_HOURS
from src.polymarket.executor import OrderExecutor
from src.polymarket.tracker import BetLedger
from src.strategy import duo_trader
from src.strategy.bankroll import BankrollManager


class _FakeClob:
    def __init__(self, open_orders=None, closed_orders=None, post_cancel_orders=None, trades=None):
        self._open_orders = list(open_orders or [])
        self._closed_orders = dict(closed_orders or {})
        self._post_cancel_orders = dict(post_cancel_orders or {})
        self._trades = list(trades or [])
        self.cancelled = []
        self.open_order_calls = 0

    @staticmethod
    def _order_id(order):
        return order.get("id") or order.get("order_id") or order.get("orderID")

    def get_open_orders(self):
        self.open_order_calls += 1
        return list(self._open_orders)

    def get_order(self, order_id):
        if order_id not in self._closed_orders:
            raise KeyError(order_id)
        return self._closed_orders[order_id]

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        removed_order = None
        remaining_orders = []
        for order in self._open_orders:
            if self._order_id(order) == order_id and removed_order is None:
                removed_order = dict(order)
                continue
            remaining_orders.append(order)
        self._open_orders = remaining_orders

        if order_id in self._post_cancel_orders:
            closed_order = self._post_cancel_orders[order_id]
            if closed_order is None:
                self._closed_orders.pop(order_id, None)
            else:
                self._closed_orders[order_id] = dict(closed_order)
        elif removed_order is not None:
            synthesized = dict(removed_order)
            synthesized.setdefault("id", order_id)
            synthesized.setdefault("status", "CANCELED")
            synthesized.setdefault(
                "original_size",
                synthesized.get("size", synthesized.get("original_size")),
            )
            synthesized.setdefault("size_matched", "0")
            self._closed_orders[order_id] = synthesized

        return {"cancelled": order_id}

    def get_trades(self, params=None):
        return list(self._trades)


def _seed_limit_bet(
    ledger: BetLedger,
    *,
    fighter="Charles Johnson",
    opponent="Bruno Silva",
    market_id="1510646",
    token_id="token-yes",
    price=0.58,
    amount=20.0,
    shares=34.48,
    order_type="limit_bid",
    order_id="order-1",
    age_minutes=60,
    event_date=None,
):
    if event_date is None:
        # Keep the default event safely in the future so TTL-specific tests do
        # not drift into the "fight started" branch as calendar time moves on.
        event_date = datetime.now(timezone.utc) + timedelta(days=7)
    if isinstance(event_date, datetime):
        event_date = event_date.isoformat()

    ledger.add_bet(
        fighter=fighter,
        opponent=opponent,
        side="a",
        amount=amount,
        price=price,
        shares=shares,
        token_id=token_id,
        market_id=market_id,
        model_prob=0.64,
        market_prob=0.62,
        edge=0.02,
        decimal_odds=1.6129,
        dry_run=False,
        event_date=event_date,
        order_type=order_type,
        order_id=order_id,
    )
    ledger.bets[-1]["placed_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=age_minutes)
    ).isoformat()
    ledger._save()


def _make_executor(tmp_path, fake_clob):
    bankroll = BankrollManager(initial_bankroll=100, auto_detect_balance=False)
    bankroll.bankroll = 80.0
    executor = OrderExecutor(
        bankroll=bankroll,
        clob_client=fake_clob,
        dry_run=False,
    )
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")
    return executor


def test_refresh_open_limit_orders_cancels_marketable_limit(tmp_path):
    fake_clob = _FakeClob(
        open_orders=[
            {
                "id": "order-1",
                "asset_id": "token-yes",
                "price": "0.58",
                "original_size": "34.48",
                "size_matched": "0",
            }
        ]
    )
    executor = _make_executor(tmp_path, fake_clob)
    _seed_limit_bet(executor.ledger)

    executor._check_liquidity = lambda *args, **kwargs: {
        "ok": True,
        "adjusted_size": 20.0,
        "available_liquidity": 200.0,
        "slippage": 0.0,
        "best_ask": 0.60,
        "reason": "",
    }

    current_bet = pd.DataFrame(
        [
            {
                "fighter_a": "Charles Johnson",
                "fighter_b": "Bruno Silva",
                "bet_on": "Charles Johnson",
                "bet_side": "a",
                "model_prob": 0.68,
                "blended_prob": 0.64,
                "market_prob": 0.62,
                "edge": 0.02,
                "decimal_odds": 1.6129,
                "token_id_yes": "token-yes",
                "token_id_no": "token-no",
                "market_id": "1510646",
                "tick_size": "0.01",
            }
        ]
    )

    summary = executor.refresh_open_limit_orders(
        matched_predictions=current_bet,
        primary_bets=current_bet,
        trader_name="Single Trader",
    )

    assert summary["cancelled_marketable"] == 1
    assert fake_clob.cancelled == ["order-1"]
    assert executor.ledger.open_bets == []
    assert executor.ledger.bets[0]["status"] == "cancelled"
    assert executor.ledger.bets[0]["cancel_reason"] == "marketable_now"
    assert executor.bankroll.bankroll == 100.0


def test_refresh_open_limit_orders_keeps_submitted_marketable_limit_without_repricing(tmp_path):
    fake_clob = _FakeClob(
        open_orders=[
            {
                "id": "order-1",
                "asset_id": "token-yes",
                "price": "0.62",
                "original_size": "32.26",
                "size_matched": "0",
            }
        ]
    )
    executor = _make_executor(tmp_path, fake_clob)
    _seed_limit_bet(
        executor.ledger,
        price=0.62,
        amount=20.0,
        shares=32.26,
        order_type="marketable_limit",
    )

    current_bet = pd.DataFrame(
        [
            {
                "fighter_a": "Charles Johnson",
                "fighter_b": "Bruno Silva",
                "bet_on": "Charles Johnson",
                "bet_side": "a",
                "model_prob": 0.68,
                "blended_prob": 0.64,
                "market_prob": 0.62,
                "edge": 0.02,
                "decimal_odds": 1.6129,
                "token_id_yes": "token-yes",
                "token_id_no": "token-no",
                "market_id": "1510646",
                "tick_size": "0.01",
            }
        ]
    )

    summary = executor.refresh_open_limit_orders(
        matched_predictions=current_bet,
        primary_bets=current_bet,
        trader_name="Single Trader",
    )

    assert summary["kept"] == 1
    assert summary["cancelled"] == 0
    assert fake_clob.cancelled == []
    assert executor.ledger.bets[0]["status"] == "open"
    assert executor.ledger.bets[0]["order_type"] == "marketable_limit"


def test_refresh_open_limit_orders_reprices_up_only_when_stale(tmp_path):
    fake_clob = _FakeClob(
        open_orders=[
            {
                "id": "order-1",
                "asset_id": "token-yes",
                "price": "0.58",
                "original_size": "34.48",
                "size_matched": "0",
            }
        ]
    )
    executor = _make_executor(tmp_path, fake_clob)
    _seed_limit_bet(executor.ledger, order_type="near_miss_limit")

    matched = pd.DataFrame(
        [
            {
                "fighter_a": "Charles Johnson",
                "fighter_b": "Bruno Silva",
                "bet_on": "Charles Johnson",
                "bet_side": "a",
                "model_prob": 0.65,
                "blended_prob": 0.635,
                "market_prob": 0.62,
                "edge": 0.015,
                "decimal_odds": 1.6129,
                "token_id_yes": "token-yes",
                "token_id_no": "token-no",
                "market_id": "1510646",
                "tick_size": "0.01",
            }
        ]
    )

    summary = executor.refresh_open_limit_orders(
        matched_predictions=matched,
        limit_only_bets=matched,
        trader_name="Single Trader",
    )

    assert summary["repriced_up"] == 1
    assert fake_clob.cancelled == ["order-1"]
    assert executor.ledger.bets[0]["status"] == "cancelled"
    assert executor.ledger.bets[0]["cancel_reason"] == "reprice_up"
    assert executor.bankroll.bankroll == 100.0


def test_refresh_open_limit_orders_keeps_partially_filled_order(tmp_path):
    fake_clob = _FakeClob(
        open_orders=[
            {
                "id": "order-1",
                "asset_id": "token-yes",
                "price": "0.58",
                "original_size": "34.48",
                "size_matched": "10.00",
            }
        ]
    )
    executor = _make_executor(tmp_path, fake_clob)
    _seed_limit_bet(executor.ledger)

    matched = pd.DataFrame(
        [
            {
                "fighter_a": "Someone",
                "fighter_b": "Else",
                "bet_on": "Someone",
                "bet_side": "a",
                "model_prob": 0.60,
                "blended_prob": 0.58,
                "market_prob": 0.55,
                "edge": 0.03,
                "decimal_odds": 1.8182,
                "token_id_yes": "other-token",
                "token_id_no": "other-no",
                "market_id": "other-market",
                "tick_size": "0.01",
            }
        ]
    )

    summary = executor.refresh_open_limit_orders(
        matched_predictions=matched,
        primary_bets=pd.DataFrame(),
        trader_name="Single Trader",
    )

    assert summary["kept"] == 1
    assert fake_clob.cancelled == []
    assert len(executor.ledger.open_bets) == 1
    assert executor.ledger.bets[0]["status"] == "open"
    assert executor.bankroll.bankroll == 80.0


def test_refresh_open_limit_orders_reconciles_missing_cancelled_order(tmp_path):
    fake_clob = _FakeClob(
        open_orders=[],
        closed_orders={
            "order-1": {
                "id": "order-1",
                "status": "CANCELED",
                "price": "0.58",
                "original_size": "34.48",
                "size_matched": "0",
            }
        },
    )
    executor = _make_executor(tmp_path, fake_clob)
    _seed_limit_bet(executor.ledger)

    current_bet = pd.DataFrame(
        [
            {
                "fighter_a": "Charles Johnson",
                "fighter_b": "Bruno Silva",
                "bet_on": "Charles Johnson",
                "bet_side": "a",
                "model_prob": 0.68,
                "blended_prob": 0.64,
                "market_prob": 0.62,
                "edge": 0.02,
                "decimal_odds": 1.6129,
                "token_id_yes": "token-yes",
                "token_id_no": "token-no",
                "market_id": "1510646",
                "tick_size": "0.01",
            }
        ]
    )

    summary = executor.refresh_open_limit_orders(
        matched_predictions=current_bet,
        primary_bets=current_bet,
        trader_name="Single Trader",
    )

    assert summary["cancelled"] == 1
    assert summary["reconciled"] == 0
    assert executor.ledger.open_bets == []
    assert executor.ledger.bets[0]["status"] == "cancelled"
    assert executor.ledger.bets[0]["cancel_reason"] == "canceled"
    assert executor.bankroll.bankroll == 100.0


def test_refresh_open_limit_orders_cancels_expired_thesis(tmp_path):
    fake_clob = _FakeClob(
        open_orders=[
            {
                "id": "order-1",
                "asset_id": "token-yes",
                "price": "0.58",
                "original_size": "34.48",
                "size_matched": "0",
            }
        ]
    )
    executor = _make_executor(tmp_path, fake_clob)
    _seed_limit_bet(executor.ledger)

    matched = pd.DataFrame(
        [
            {
                "fighter_a": "Someone",
                "fighter_b": "Else",
                "bet_on": "Someone",
                "bet_side": "a",
                "model_prob": 0.60,
                "blended_prob": 0.58,
                "market_prob": 0.55,
                "edge": 0.03,
                "decimal_odds": 1.8182,
                "token_id_yes": "other-token",
                "token_id_no": "other-no",
                "market_id": "other-market",
                "tick_size": "0.01",
            }
        ]
    )

    summary = executor.refresh_open_limit_orders(
        matched_predictions=matched,
        primary_bets=pd.DataFrame(),
        trader_name="Single Trader",
    )

    assert summary["cancelled"] == 1
    assert summary["cancelled_thesis"] == 1
    assert executor.ledger.bets[0]["status"] == "cancelled"
    assert executor.ledger.bets[0]["cancel_reason"] == "thesis_expired"
    assert executor.bankroll.bankroll == 100.0


def test_refresh_open_limit_orders_preserves_fill_reported_after_cancel(tmp_path):
    fake_clob = _FakeClob(
        open_orders=[
            {
                "id": "order-1",
                "asset_id": "token-yes",
                "price": "0.58",
                "original_size": "34.48",
                "size_matched": "0",
            }
        ],
        post_cancel_orders={
            "order-1": {
                "id": "order-1",
                "status": "CANCELED",
                "price": "0.58",
                "original_size": "34.48",
                "size_matched": "10.00",
            }
        },
    )
    executor = _make_executor(tmp_path, fake_clob)
    _seed_limit_bet(executor.ledger)

    executor._check_liquidity = lambda *args, **kwargs: {
        "ok": True,
        "adjusted_size": 20.0,
        "available_liquidity": 200.0,
        "slippage": 0.0,
        "best_ask": 0.60,
        "reason": "",
    }

    current_bet = pd.DataFrame(
        [
            {
                "fighter_a": "Charles Johnson",
                "fighter_b": "Bruno Silva",
                "bet_on": "Charles Johnson",
                "bet_side": "a",
                "model_prob": 0.68,
                "blended_prob": 0.64,
                "market_prob": 0.62,
                "edge": 0.02,
                "decimal_odds": 1.6129,
                "token_id_yes": "token-yes",
                "token_id_no": "token-no",
                "market_id": "1510646",
                "tick_size": "0.01",
            }
        ]
    )

    summary = executor.refresh_open_limit_orders(
        matched_predictions=current_bet,
        primary_bets=current_bet,
        trader_name="Single Trader",
    )

    assert summary["cancelled"] == 1
    assert summary["cancelled_marketable"] == 1
    assert fake_clob.cancelled == ["order-1"]
    assert len(executor.ledger.open_bets) == 1
    assert executor.ledger.bets[0]["status"] == "open"
    assert executor.ledger.bets[0]["order_type"] == "filled_limit"
    assert executor.ledger.bets[0]["order_id"] is None
    assert executor.ledger.bets[0]["shares"] == 10.0
    assert executor.ledger.bets[0]["amount"] == 5.8
    assert executor.ledger.bets[0]["cancel_reason"] == "marketable_now"
    assert executor.bankroll.bankroll == 94.2


def test_finalize_cancelled_limit_order_does_not_cancel_stale_filled_limit_position(tmp_path):
    fake_clob = _FakeClob(
        open_orders=[],
        closed_orders={
            "order-1": {
                "id": "order-1",
                "status": "CANCELED",
                "price": "0.58",
                "original_size": "34.48",
                "size_matched": "0",
            }
        },
    )
    executor = _make_executor(tmp_path, fake_clob)
    _seed_limit_bet(executor.ledger)

    stale_bet = dict(executor.ledger.open_bets[0])
    converted = executor.ledger.convert_limit_bet_to_position(
        stale_bet["id"],
        filled_shares=10.0,
        cancel_reason="marketable_now",
    )

    finalized = executor._finalize_cancelled_limit_order(
        stale_bet,
        reason="marketable_now",
    )

    assert converted.ok is True
    assert finalized is False
    assert len(executor.ledger.open_bets) == 1
    assert executor.ledger.bets[0]["status"] == "open"
    assert executor.ledger.bets[0]["order_type"] == "filled_limit"
    assert executor.ledger.bets[0]["amount"] == 5.8
    assert executor.ledger.bets[0]["shares"] == 10.0
    assert executor.bankroll.bankroll == 80.0
    assert executor.order_log == []


def test_finalize_cancelled_limit_order_uses_direct_lookup_when_open_snapshot_is_stale(tmp_path):
    fake_clob = _FakeClob(
        open_orders=[
            {
                "id": "order-1",
                "asset_id": "token-yes",
                "price": "0.58",
                "original_size": "34.48",
                "size_matched": "0",
            }
        ],
        closed_orders={
            "order-1": {
                "id": "order-1",
                "status": "CANCELED",
                "price": "0.58",
                "original_size": "34.48",
                "size_matched": "0",
            }
        },
    )
    executor = _make_executor(tmp_path, fake_clob)
    _seed_limit_bet(executor.ledger)

    finalized = executor._finalize_cancelled_limit_order(
        executor.ledger.open_bets[0],
        reason="marketable_now",
    )

    assert finalized is True
    assert fake_clob.open_order_calls == 1
    assert executor.ledger.open_bets == []
    assert executor.ledger.bets[0]["status"] == "cancelled"
    assert executor.ledger.bets[0]["cancel_reason"] == "marketable_now"
    assert executor.bankroll.bankroll == 100.0


def test_cancel_stale_limit_bids_preserves_partial_fill(tmp_path):
    fake_clob = _FakeClob(
        open_orders=[
            {
                "id": "order-1",
                "asset_id": "token-yes",
                "price": "0.58",
                "original_size": "34.48",
                "size_matched": "10.00",
            }
        ]
    )
    executor = _make_executor(tmp_path, fake_clob)
    _seed_limit_bet(executor.ledger, age_minutes=(LIMIT_BID_TTL_HOURS * 60) + 5)

    cancelled = executor.cancel_stale_limit_bids()

    assert cancelled == 1
    assert fake_clob.cancelled == ["order-1"]
    assert len(executor.ledger.open_bets) == 1
    assert executor.ledger.bets[0]["status"] == "open"
    assert executor.ledger.bets[0]["order_type"] == "filled_limit"
    assert executor.ledger.bets[0]["order_id"] is None
    assert executor.ledger.bets[0]["shares"] == 10.0
    assert executor.ledger.bets[0]["amount"] == 5.8
    assert executor.ledger.bets[0]["cancel_reason"] == f"exceeded {LIMIT_BID_TTL_HOURS}h TTL"
    assert executor.bankroll.bankroll == 94.2


def test_refresh_open_limit_orders_reconciles_trade_history_without_predictions(tmp_path):
    fake_clob = _FakeClob(
        open_orders=[],
        trades=[
            {
                "status": "CONFIRMED",
                "maker_orders": [
                    {
                        "order_id": "order-1",
                        "matched_amount": "10.00",
                    }
                ]
            }
        ],
    )
    executor = _make_executor(tmp_path, fake_clob)
    _seed_limit_bet(executor.ledger)

    summary = executor.refresh_open_limit_orders(
        matched_predictions=pd.DataFrame(),
        primary_bets=pd.DataFrame(),
        trader_name="Single Trader",
    )

    assert summary["reconciled"] == 1
    assert summary["cancelled"] == 0
    assert summary["kept"] == 0
    assert len(executor.ledger.open_bets) == 1
    assert executor.ledger.bets[0]["status"] == "open"
    assert executor.ledger.bets[0]["order_type"] == "filled_limit"
    assert executor.ledger.bets[0]["order_id"] is None
    assert executor.ledger.bets[0]["shares"] == 10.0
    assert executor.ledger.bets[0]["amount"] == 5.8
    assert executor.bankroll.bankroll == 94.2


def test_refresh_open_limit_orders_reconciles_closed_marketable_limit(tmp_path):
    fake_clob = _FakeClob(
        open_orders=[],
        closed_orders={
            "order-1": {
                "id": "order-1",
                "status": "MATCHED",
                "price": "0.58",
                "original_size": "34.48",
                "size_matched": "10.00",
            }
        },
    )
    executor = _make_executor(tmp_path, fake_clob)
    _seed_limit_bet(executor.ledger, order_type="marketable_limit")

    summary = executor.refresh_open_limit_orders(
        matched_predictions=pd.DataFrame(),
        primary_bets=pd.DataFrame(),
        trader_name="Single Trader",
    )

    assert summary["reconciled"] == 1
    assert summary["cancelled"] == 0
    assert len(executor.ledger.open_bets) == 1
    assert executor.ledger.bets[0]["status"] == "open"
    assert executor.ledger.bets[0]["order_type"] == "filled_limit"
    assert executor.ledger.bets[0]["shares"] == 10.0
    assert executor.ledger.bets[0]["amount"] == 5.8
    assert executor.bankroll.bankroll == 94.2


def test_refresh_open_limit_orders_keeps_unknown_when_trade_history_is_not_final(tmp_path):
    fake_clob = _FakeClob(
        open_orders=[],
        trades=[
            {
                "status": "MATCHED",
                "maker_orders": [
                    {
                        "order_id": "order-1",
                        "matched_amount": "10.00",
                    }
                ],
            }
        ],
    )
    executor = _make_executor(tmp_path, fake_clob)
    _seed_limit_bet(executor.ledger)

    summary = executor.refresh_open_limit_orders(
        matched_predictions=pd.DataFrame(),
        primary_bets=pd.DataFrame(),
        trader_name="Single Trader",
    )

    assert summary["reconciled"] == 0
    assert summary["cancelled"] == 0
    assert summary["kept"] == 1
    assert len(executor.ledger.open_bets) == 1
    assert executor.ledger.bets[0]["status"] == "open"
    assert executor.ledger.bets[0]["order_type"] == "limit_bid"
    assert executor.ledger.bets[0]["order_id"] == "order-1"
    assert executor.bankroll.bankroll == 80.0


def test_cancel_stale_limit_bids_records_cancel_reason(tmp_path):
    fake_clob = _FakeClob(
        open_orders=[
            {
                "id": "order-1",
                "asset_id": "token-yes",
                "price": "0.58",
                "original_size": "34.48",
                "size_matched": "0",
            }
        ]
    )
    executor = _make_executor(tmp_path, fake_clob)
    _seed_limit_bet(executor.ledger, age_minutes=(LIMIT_BID_TTL_HOURS * 60) + 5)

    cancelled = executor.cancel_stale_limit_bids()

    assert cancelled == 1
    assert fake_clob.cancelled == ["order-1"]
    assert executor.ledger.open_bets == []
    assert executor.ledger.bets[0]["status"] == "cancelled"
    assert executor.ledger.bets[0]["cancel_reason"] == f"exceeded {LIMIT_BID_TTL_HOURS}h TTL"
    assert executor.bankroll.bankroll == 100.0


def test_cancel_stale_limit_bids_uses_recent_open_orders_cache(tmp_path):
    fake_clob = _FakeClob(
        open_orders=[
            {
                "id": "order-1",
                "asset_id": "token-yes",
                "price": "0.58",
                "original_size": "34.48",
                "size_matched": "0",
            }
        ]
    )
    executor = _make_executor(tmp_path, fake_clob)
    _seed_limit_bet(executor.ledger, age_minutes=(LIMIT_BID_TTL_HOURS * 60) + 5)
    executor._open_orders_cache = (time.monotonic(), list(fake_clob._open_orders))

    cancelled = executor.cancel_stale_limit_bids()

    assert cancelled == 1
    assert fake_clob.cancelled == ["order-1"]
    assert fake_clob.open_order_calls == 1
    assert executor.ledger.open_bets == []


def test_cancel_stale_limit_bids_preserves_fill_reported_after_cancel(tmp_path):
    fake_clob = _FakeClob(
        open_orders=[
            {
                "id": "order-1",
                "asset_id": "token-yes",
                "price": "0.58",
                "original_size": "34.48",
                "size_matched": "0",
            }
        ],
        post_cancel_orders={
            "order-1": {
                "id": "order-1",
                "status": "CANCELED",
                "price": "0.58",
                "original_size": "34.48",
                "size_matched": "10.00",
            }
        },
    )
    executor = _make_executor(tmp_path, fake_clob)
    _seed_limit_bet(executor.ledger, age_minutes=(LIMIT_BID_TTL_HOURS * 60) + 5)

    cancelled = executor.cancel_stale_limit_bids()

    assert cancelled == 1
    assert fake_clob.cancelled == ["order-1"]
    assert len(executor.ledger.open_bets) == 1
    assert executor.ledger.bets[0]["status"] == "open"
    assert executor.ledger.bets[0]["order_type"] == "filled_limit"
    assert executor.ledger.bets[0]["order_id"] is None
    assert executor.ledger.bets[0]["shares"] == 10.0
    assert executor.ledger.bets[0]["amount"] == 5.8
    assert executor.ledger.bets[0]["cancel_reason"] == f"exceeded {LIMIT_BID_TTL_HOURS}h TTL"
    assert executor.bankroll.bankroll == 94.2


def test_run_duo_traders_blocks_conviction_on_existing_single_open_bet(monkeypatch):
    class _FakeBankroll:
        def __init__(self, bankroll):
            self.bankroll = bankroll
            self.is_stopped = False

        def get_stats(self):
            return {"bankroll": self.bankroll}

    class _FakeExecutor:
        def __init__(self, open_bets=None):
            self.ledger = SimpleNamespace(open_bets=list(open_bets or []))
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

    single_exec = _FakeExecutor(
        open_bets=[
            {
                "fighter": "Charles Johnson",
                "opponent": "Bruno Silva",
                "status": "open",
                "order_type": "limit_bid",
            }
        ]
    )
    conv_exec = _FakeExecutor()

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
    tracker = lambda name: SimpleNamespace(
        name=name,
        bankroll=_FakeBankroll(100.0),
        executor=_FakeExecutor(),
    )

    monkeypatch.setattr(
        "src.strategy.llm_operator.OPERATOR_ENABLED",
        False,
    )
    monkeypatch.setattr(
        duo_trader,
        "_resolve_total_bankroll",
        lambda dry_run=True: duo_trader.WalletBankrollBasis(100.0, 100.0, "test"),
    )
    monkeypatch.setattr(duo_trader, "_create_trader", lambda *args, **kwargs: next(created))
    monkeypatch.setattr(
        duo_trader,
        "find_value_bets",
        lambda *args, **kwargs: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        duo_trader,
        "find_conviction_bets",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "fighter_a": "Charles Johnson",
                    "fighter_b": "Bruno Silva",
                    "bet_on": "Charles Johnson",
                    "bet_side": "a",
                    "model_prob": 0.72,
                }
            ]
        ),
    )
    monkeypatch.setattr(duo_trader, "_create_tracker_trader", lambda name, *_args, **_kwargs: tracker(name))
    monkeypatch.setattr(duo_trader, "find_flat_model_bets", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(duo_trader, "find_flat_gemini_bets", lambda *args, **kwargs: pd.DataFrame())

    result = duo_trader.run_duo_traders(
        predictions=pd.DataFrame(),
        markets=pd.DataFrame(),
        clob=None,
        dry_run=True,
    )

    assert conv_exec.placed == []
    assert result["total_orders"] == 0


def test_run_duo_traders_ignores_dry_run_single_open_bet_in_live_mode(monkeypatch):
    class _FakeBankroll:
        def __init__(self, bankroll):
            self.bankroll = bankroll
            self.is_stopped = False

        def get_stats(self):
            return {"bankroll": self.bankroll}

    class _FakeExecutor:
        def __init__(self, open_bets=None):
            self.ledger = SimpleNamespace(open_bets=list(open_bets or []))
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

    single_exec = _FakeExecutor(
        open_bets=[
            {
                "fighter": "Charles Johnson",
                "opponent": "Bruno Silva",
                "status": "open",
                "order_type": "limit_bid",
                "dry_run": True,
            }
        ]
    )
    conv_exec = _FakeExecutor()

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
    tracker = lambda name: SimpleNamespace(
        name=name,
        bankroll=_FakeBankroll(100.0),
        executor=_FakeExecutor(),
    )

    # Disable LLM Operator gate so it doesn't filter bets
    monkeypatch.setattr(
        "src.strategy.llm_operator.OPERATOR_ENABLED", False,
    )
    monkeypatch.setattr(
        duo_trader,
        "_resolve_total_bankroll",
        lambda dry_run=False: duo_trader.WalletBankrollBasis(100.0, 100.0, "test"),
    )
    monkeypatch.setattr(duo_trader, "_create_trader", lambda *args, **kwargs: next(created))
    monkeypatch.setattr(
        duo_trader,
        "find_value_bets",
        lambda *args, **kwargs: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        duo_trader,
        "find_conviction_bets",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "fighter_a": "Charles Johnson",
                    "fighter_b": "Bruno Silva",
                    "bet_on": "Charles Johnson",
                    "bet_side": "a",
                    "model_prob": 0.72,
                }
            ]
        ),
    )
    monkeypatch.setattr(duo_trader, "conviction_bet_size", lambda **kwargs: 10.0)
    monkeypatch.setattr(duo_trader, "_create_tracker_trader", lambda name, *_args, **_kwargs: tracker(name))
    monkeypatch.setattr(duo_trader, "find_flat_model_bets", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(duo_trader, "find_flat_gemini_bets", lambda *args, **kwargs: pd.DataFrame())

    result = duo_trader.run_duo_traders(
        predictions=pd.DataFrame(),
        markets=pd.DataFrame(),
        clob=None,
        dry_run=False,
    )

    assert len(conv_exec.placed) == 1
    assert result["total_orders"] == 1
