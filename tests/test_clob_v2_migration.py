import pytest
import pandas as pd

from src.polymarket.client import ClobClientWrapper, LEGACY_POLYGON_USDC_E
from src.polymarket.executor import OrderExecutor
from src.polymarket.tracker import BetLedger
from src.strategy.bankroll import BankrollManager


class _FakeResponse:
    def __init__(self, result_hex: str):
        self._result_hex = result_hex

    def json(self):
        return {"result": self._result_hex}


def _base_bet(**overrides):
    data = {
        "fighter_a": "Alpha",
        "fighter_b": "Beta",
        "bet_on": "Alpha",
        "model_prob": 0.70,
        "blended_prob": 0.70,
        "market_prob": 0.60,
        "edge": 0.10,
        "decimal_odds": 1.6667,
        "bet_side": "a",
        "token_id_yes": "token-yes",
        "token_id_no": "token-no",
        "market_id": "market-1",
        "condition_id": "condition-1",
        "tick_size": "0.01",
        "neg_risk": False,
        "override_bet_size": 25.0,
    }
    data.update(overrides)
    return pd.Series(data)


def test_v2_wrapper_compatibility_adapters():
    class _RawClient:
        def __init__(self):
            self.cancel_payload = None
            self.trade_params = None

        def cancel_order(self, payload):
            self.cancel_payload = payload
            return {"cancelled": payload.orderID}

        def cancel_all(self):
            return {"cancelled": "all"}

        def get_open_orders(self):
            return [{"id": "open-1"}]

        def get_trades(self, params=None):
            self.trade_params = params
            return [{"id": "trade-1"}]

    raw = _RawClient()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw

    assert wrapper.cancel_order("order-1") == {"cancelled": "order-1"}
    assert raw.cancel_payload.orderID == "order-1"
    assert wrapper.cancel_all_orders() == {"cancelled": "all"}
    assert wrapper.get_open_orders() == [{"id": "open-1"}]
    assert wrapper.get_trades(params={"asset_id": "token-1"}) == [{"id": "trade-1"}]
    assert raw.trade_params == {"asset_id": "token-1"}


def test_v2_wrapper_normalizes_raw_dict_orderbook():
    class _RawClient:
        def get_order_book(self, _token_id):
            return {
                "market": "market-1",
                "asset_id": "token-1",
                "bids": [
                    {"price": "0.61", "size": "10"},
                    {"price": "0.63", "size": "4"},
                ],
                "asks": [
                    {"price": "0.68", "size": "8"},
                    {"price": "0.66", "size": "6"},
                ],
            }

    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = _RawClient()

    book = wrapper.get_orderbook("token-1")

    assert book["market"] == "market-1"
    assert book["asset_id"] == "token-1"
    assert book["bids"] == [
        {"price": "0.63", "size": "4"},
        {"price": "0.61", "size": "10"},
    ]
    assert book["asks"] == [
        {"price": "0.66", "size": "6"},
        {"price": "0.68", "size": "8"},
    ]


def test_v2_wrapper_keeps_legacy_orderbook_object_compatibility():
    class _Level:
        def __init__(self, price, size):
            self.price = price
            self.size = size

    class _Book:
        bids = [_Level("0.58", "3"), _Level("0.62", "7")]
        asks = [_Level("0.71", "5"), _Level("0.69", "2")]

    class _RawClient:
        def get_order_book(self, _token_id):
            return _Book()

    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = _RawClient()

    book = wrapper.get_orderbook("token-1")

    assert book["bids"] == [
        {"price": "0.62", "size": "7"},
        {"price": "0.58", "size": "3"},
    ]
    assert book["asks"] == [
        {"price": "0.69", "size": "2"},
        {"price": "0.71", "size": "5"},
    ]


def test_market_buy_wrapper_returns_submitted_amount_metadata(monkeypatch):
    class _RawClient:
        def create_and_post_market_order(self, args, options, order_type):
            args.price = 0.62
            args.amount = 24.25
            return {"orderID": "order-1"}

    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = _RawClient()
    monkeypatch.setattr(
        wrapper,
        "get_geoblock_status",
        lambda: {"blocked": False, "status_code": 200, "ip": "", "country": "", "region": "", "error": ""},
    )

    response = wrapper.create_market_order(
        token_id="token-yes",
        side="BUY",
        amount=25.0,
        tick_size="0.01",
        neg_risk=False,
        user_usdc_balance=25.0,
    )

    assert response["orderID"] == "order-1"
    assert response["_requested_amount"] == pytest.approx(25.0)
    assert response["_submitted_amount"] == pytest.approx(24.25)
    assert response["_execution_price"] == pytest.approx(0.62)


def test_executor_uses_actual_submitted_market_amount_for_accounting(tmp_path):
    class _Clob:
        def get_open_orders(self):
            return []

        def create_market_order(self, **kwargs):
            return {
                "orderID": "order-1",
                "_requested_amount": kwargs["amount"],
                "_submitted_amount": 24.0,
                "_execution_price": 0.60,
            }

        def get_order(self, _order_id):
            return {"price": "0.60"}

    bankroll = BankrollManager(
        initial_bankroll=100.0,
        max_bet_fraction=1.0,
        auto_detect_balance=False,
    )
    executor = OrderExecutor(bankroll=bankroll, clob_client=_Clob(), dry_run=False)
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")
    executor._authoritative_wallet_conflict = lambda **_kwargs: (False, "")
    executor._check_liquidity = lambda *args, **kwargs: {
        "ok": True,
        "adjusted_size": 25.0,
        "available_liquidity": 100.0,
        "slippage": 0.0,
        "best_ask": 0.60,
        "reason": "",
    }

    result = executor._place_bet(_base_bet(), pd.DataFrame())

    assert result["status"] == "placed"
    assert result["requested_bet_size_usd"] == pytest.approx(25.0)
    assert result["bet_size_usd"] == pytest.approx(24.0)
    assert result["shares"] == pytest.approx(40.0)
    assert bankroll.bankroll == pytest.approx(76.0)

    ledger_bet = BetLedger(path=tmp_path / "ledger.json").bets[0]
    assert ledger_bet["amount"] == pytest.approx(24.0)
    assert ledger_bet["shares"] == pytest.approx(40.0)
    assert ledger_bet["requested_amount"] == pytest.approx(25.0)
    assert ledger_bet["submitted_amount"] == pytest.approx(24.0)


def test_version_aware_collateral_fallback_selects_token(monkeypatch):
    seen_tokens = []

    def fake_post(_url, *, json, timeout):
        seen_tokens.append(json["params"][0]["to"])
        return _FakeResponse("0x0f4240")

    monkeypatch.setattr("src.polymarket.client.requests.post", fake_post)

    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    monkeypatch.setattr(wrapper, "_get_clob_backend_version", lambda: 1)
    balance, source = wrapper._get_onchain_collateral_balance("0x" + "11" * 20)

    assert balance == pytest.approx(1.0)
    assert source == "onchain_v1_collateral"
    assert seen_tokens[-1] == LEGACY_POLYGON_USDC_E

    monkeypatch.setattr(wrapper, "_get_clob_backend_version", lambda: 2)
    balance, source = wrapper._get_onchain_collateral_balance("0x" + "11" * 20)

    assert balance == pytest.approx(1.0)
    assert source == "onchain_v2_collateral"
    assert seen_tokens[-1] != LEGACY_POLYGON_USDC_E


def test_market_buy_path_uses_fee_adjusted_net_edge(tmp_path):
    class _Clob:
        market_calls = 0

        def get_open_orders(self):
            return []

        def create_market_order(self, **_kwargs):
            self.market_calls += 1
            return {"orderID": "unexpected"}

    clob = _Clob()
    executor = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=100.0, auto_detect_balance=False),
        clob_client=clob,
        dry_run=False,
    )
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")
    executor._authoritative_wallet_conflict = lambda **_kwargs: (False, "")
    executor._check_liquidity = lambda *args, **kwargs: {
        "ok": True,
        "adjusted_size": 25.0,
        "available_liquidity": 100.0,
        "slippage": 0.0,
        "best_ask": 0.62,
        "reason": "",
    }

    result = executor._place_bet(
        _base_bet(blended_prob=0.65, model_prob=0.65, fee_rate=0.10, fee_exponent=1.0),
        pd.DataFrame(),
    )

    assert result is None
    assert clob.market_calls == 0


def test_maker_limit_path_is_not_suppressed_by_taker_fee_metadata(tmp_path):
    class _Clob:
        def __init__(self):
            self.limit_calls = 0

        def get_open_orders(self):
            return []

        def create_limit_order(self, **_kwargs):
            self.limit_calls += 1
            return {"orderID": "limit-1"}

    clob = _Clob()
    executor = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=100.0, auto_detect_balance=False),
        clob_client=clob,
        dry_run=False,
    )
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")
    executor._authoritative_wallet_conflict = lambda **_kwargs: (False, "")
    executor._check_liquidity = lambda *args, **kwargs: {
        "ok": True,
        "adjusted_size": 25.0,
        "available_liquidity": 100.0,
        "slippage": 0.0,
        "best_ask": 0.64,
        "reason": "",
    }

    result = executor._place_bet(
        _base_bet(blended_prob=0.65, model_prob=0.65, fee_rate=1.0, fee_exponent=1.0),
        pd.DataFrame(),
    )

    assert result["status"] == "placed"
    assert result["order_type"] == "limit_bid"
    assert clob.limit_calls == 1
