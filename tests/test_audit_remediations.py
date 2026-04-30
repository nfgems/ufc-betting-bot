import pandas as pd
import pytest
import src.strategy.value as value_module

from src.polymarket.client import ClobClientWrapper
from src.polymarket.executor import OrderExecutor, _name_match
from src.polymarket.tracker import BetLedger
from src.strategy.bankroll import BankrollManager
from src.strategy.value import (
    blend_probability,
    compute_independent_blend_probs,
    dynamic_blend_weight,
    find_conviction_bets,
    find_value_bets,
)
from src.web import app as web_app


class _UnknownMarketableLimitClob:
    def __init__(self):
        self.market_calls = 0
        self.limit_calls = 0

    def create_market_order(self, **kwargs):
        self.market_calls += 1
        return {'orderID': 'unexpected-market-order'}

    def create_limit_order(self, **kwargs):
        self.limit_calls += 1
        raise RuntimeError('socket timeout while posting order')


class _BalanceWrapper(ClobClientWrapper):
    def __init__(self, payload):
        self._payload = payload

    def get_balance_allowance(self):
        return dict(self._payload)


class _OrderbookPriceWrapper(ClobClientWrapper):
    def __init__(self, orderbook):
        self._orderbook = orderbook

    def get_orderbook(self, _token_id):
        return self._orderbook


@pytest.fixture(autouse=True)
def _reset_dashboard_host(monkeypatch):
    monkeypatch.setattr(web_app, '_server_host', '127.0.0.1')


def test_get_cash_balance_parses_atomic_units_by_format():
    wrapper = _BalanceWrapper({'balance': '750'})
    assert wrapper.get_cash_balance() == pytest.approx(0.00075)


def test_get_cash_balance_preserves_decimal_strings():
    wrapper = _BalanceWrapper({'balance': '12.34'})
    assert wrapper.get_cash_balance() == pytest.approx(12.34)


def test_get_cash_balance_details_report_source():
    wrapper = _BalanceWrapper({'balance': '12.34'})
    details = wrapper.get_cash_balance_details(allow_onchain_fallback=False)
    assert details["balance"] == pytest.approx(12.34)
    assert details["source"] == "clob"


def test_public_refresh_prices_is_open_but_other_mutations_require_token(monkeypatch):
    monkeypatch.setattr(web_app, '_server_host', '0.0.0.0')
    monkeypatch.delenv('WEB_DASHBOARD_TOKEN', raising=False)
    monkeypatch.setattr(
        web_app,
        'auto_redeem_positions_from_polymarket',
        lambda **_kwargs: {
            "redeemable_positions": 0,
            "redeemable_conditions": 0,
            "submitted_conditions": 0,
            "redeemed_conditions": 0,
            "redeemed_positions": 0,
            "results": [],
            "errors": [],
            "reason": "no_redeemable_positions",
        },
    )
    client = web_app.app.test_client()

    refresh = client.post('/api/refresh-prices')
    assert refresh.status_code == 200
    assert refresh.get_json()['status'] == 'offline'

    disabled = client.post('/api/settle-auto')
    assert disabled.status_code == 503
    assert disabled.get_json()['error'] == 'dashboard_mutations_disabled'

    monkeypatch.setenv('WEB_DASHBOARD_TOKEN', 'secret-token')
    unauthorized = client.post('/api/settle-auto')
    assert unauthorized.status_code == 401
    unauthorized_redeem = client.post('/api/redeem-auto')
    assert unauthorized_redeem.status_code == 401

    authorized = client.post(
        '/api/settle-auto',
        headers={'X-Dashboard-Token': 'secret-token'},
    )
    assert authorized.status_code == 200
    assert authorized.get_json()['settled'] == 0

    authorized_redeem = client.post(
        '/api/redeem-auto',
        headers={'X-Dashboard-Token': 'secret-token'},
    )
    assert authorized_redeem.status_code == 200
    assert authorized_redeem.get_json()['redeemed_positions'] == 0


def test_get_price_does_not_fabricate_midpoint_for_one_sided_book():
    wrapper = _OrderbookPriceWrapper(
        {
            "bids": [{"price": "0.62", "size": "15"}],
            "asks": [],
        }
    )

    price = wrapper.get_price("token-1")

    assert price["best_bid"] == pytest.approx(0.62)
    assert price["best_ask"] is None
    assert price["mid"] is None


def test_match_predictions_to_markets_preserves_zero_probability_prices():
    executor = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=100.0, auto_detect_balance=False),
        dry_run=True,
    )
    predictions = pd.DataFrame(
        [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "prob_a": 0.75,
                "prob_b": 0.25,
                "event_date": "2026-03-28",
            }
        ]
    )
    markets = pd.DataFrame(
        [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "price_yes": 0.0,
                "price_no": 1.0,
                "token_id_yes": "token-yes",
                "token_id_no": "token-no",
                "market_id": "market-1",
                "condition_id": "cond-1",
                "event_date": "2026-03-28",
            }
        ]
    )

    matched = executor._match_predictions_to_markets(predictions, markets)

    assert matched.loc[0, "a_market_prob"] == pytest.approx(0.0)
    assert matched.loc[0, "b_market_prob"] == pytest.approx(1.0)


def test_market_order_validation_rejects_invalid_side_before_client_use():
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = _UnknownMarketableLimitClob()

    with pytest.raises(ValueError, match="Market order side must be 'BUY' or 'SELL'"):
        wrapper.create_market_order(token_id="token-1", side="HOLD", amount=10)


def test_market_order_validation_rejects_non_positive_amount_before_client_use():
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = _UnknownMarketableLimitClob()

    with pytest.raises(ValueError, match="Market order amount must be positive"):
        wrapper.create_market_order(token_id="token-1", side="BUY", amount=0)


def test_market_order_validation_rejects_sub_dollar_buy_before_client_use():
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = _UnknownMarketableLimitClob()

    with pytest.raises(ValueError, match="Market BUY order amount must be at least"):
        wrapper.create_market_order(token_id="token-1", side="BUY", amount=0.99)


def test_limit_order_validation_rejects_rounded_sub_dollar_buy_before_client_use():
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = _UnknownMarketableLimitClob()

    with pytest.raises(ValueError, match="Limit BUY order notional must be at least"):
        wrapper.create_limit_order(
            token_id="token-1",
            side="BUY",
            price=0.57,
            size=1.75,
        )


def test_independent_blend_uses_both_side_weights():
    blend_a, blend_b = compute_independent_blend_probs(
        0.68,
        0.55,
        0.58,
        0.32,
        0.45,
        0.10,
        0.30,
    )
    legacy_weight = dynamic_blend_weight(0.68, 0.55, 0.58, 0.30)
    legacy_blend_a = blend_probability(0.68, 0.55, legacy_weight)

    assert blend_a + blend_b == pytest.approx(1.0)
    assert blend_a != pytest.approx(legacy_blend_a)


def test_find_conviction_bets_requires_positive_ev_by_default():
    predictions = pd.DataFrame(
        [
            {
                'fighter_a': 'Alpha',
                'fighter_b': 'Beta',
                'prob_a': 0.70,
                'prob_b': 0.30,
                'a_market_prob': 0.80,
                'b_market_prob': 0.20,
                'no_odds_prob_a': 0.60,
                'no_odds_prob_b': 0.40,
                'a_num_fights': 6,
                'b_num_fights': 7,
            }
        ]
    )

    assert find_conviction_bets(predictions).empty
    assert len(find_conviction_bets(predictions, require_positive_ev=False)) == 1


def test_find_value_bets_falls_back_to_other_side_when_larger_edge_fails(monkeypatch):
    predictions = pd.DataFrame(
        [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "prob_a": 0.70,
                "prob_b": 0.30,
                "a_market_prob": 0.60,
                "b_market_prob": 0.50,
                "no_odds_prob_a": 0.70,
                "no_odds_prob_b": 0.65,
                "a_num_fights": 6,
                "b_num_fights": 6,
            }
        ]
    )

    monkeypatch.setattr(
        value_module,
        "compute_independent_blend_probs",
        lambda *_args, **_kwargs: (0.75, 0.60),
    )
    monkeypatch.setattr(
        value_module,
        "_passes_filters",
        lambda blended_prob, market_prob, edge, fighter_name, no_odds_prob=None, **kwargs: kwargs.get("bet_side") == "b",
    )

    result = find_value_bets(predictions, min_edge=0.05)

    assert len(result) == 1
    assert result.iloc[0]["bet_side"] == "b"
    assert result.iloc[0]["bet_on"] == "Beta"


def test_bankroll_place_bet_rejects_overdraw_and_tracks_sequential_drawdown():
    bankroll = BankrollManager(initial_bankroll=100.0, auto_detect_balance=False)

    assert bankroll.place_bet(150.0, 'Alpha', 2.0, 0.60, 0.50) == {}
    assert bankroll.bankroll == pytest.approx(100.0)

    bankroll.place_bet(20.0, 'Alpha', 2.0, 0.60, 0.50)
    bankroll.settle_bet(0, won=True)
    bankroll.place_bet(30.0, 'Beta', 2.0, 0.60, 0.50)
    bankroll.settle_bet(1, won=False)

    stats = bankroll.get_stats()
    assert stats['max_drawdown'] == pytest.approx(0.25)


def test_marketable_limit_unknown_journals_state_without_charging_bankroll(tmp_path):
    clob = _UnknownMarketableLimitClob()
    executor = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=500.0, auto_detect_balance=False),
        clob_client=clob,
        dry_run=False,
    )
    executor.ledger = BetLedger(path=tmp_path / 'ledger.json')
    executor._check_liquidity = lambda *args, **kwargs: {
        'ok': True,
        'adjusted_size': 25.0,
        'available_liquidity': 100.0,
        'slippage': 0.0,
        'best_ask': 0.62,
        'reason': '',
    }

    bet = pd.Series(
        {
            'fighter_a': 'Alpha',
            'fighter_b': 'Beta',
            'bet_on': 'Alpha',
            'model_prob': 0.70,
            'blended_prob': 0.70,
            'market_prob': 0.62,
            'edge': 0.08,
            'decimal_odds': 1.6129,
            'bet_side': 'a',
            'token_id_yes': 'token-yes',
            'token_id_no': 'token-no',
            'market_id': 'market-1',
            'tick_size': '0.01',
            'neg_risk': False,
            'override_bet_size': 25.0,
        }
    )

    result = executor._place_bet(bet, pd.DataFrame())

    assert result is not None
    assert result['status'] == 'unknown'
    assert clob.market_calls == 0
    assert clob.limit_calls == 1
    # Unknown exchange submissions do NOT charge bankroll to prevent phantom drain.
    assert executor.bankroll.bankroll == pytest.approx(500.0)

    ledger_bet = BetLedger(path=tmp_path / 'ledger.json').bets[0]
    assert ledger_bet['status'] == 'open'
    assert ledger_bet['order_type'] == 'marketable_limit'
    assert ledger_bet['placement_state'] == 'unknown'
    assert ledger_bet['order_id'] is None


def test_name_match_requires_prefix_not_substring():
    assert _name_match('Jon Jones', 'Jonathan Jones') is True
    assert _name_match('Ian Garry', 'Brian Garry') is False
