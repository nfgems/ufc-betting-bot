import pandas as pd
import pytest

from src.polymarket.client import ClobClientWrapper
from src.polymarket.executor import OrderExecutor, _name_match
from src.polymarket.tracker import BetLedger
from src.strategy.bankroll import BankrollManager
from src.strategy.value import (
    blend_probability,
    compute_independent_blend_probs,
    dynamic_blend_weight,
    find_conviction_bets,
)
from src.web import app as web_app


class _UnknownMarketOrderClob:
    def __init__(self):
        self.market_calls = 0
        self.limit_calls = 0

    def create_market_order(self, **kwargs):
        self.market_calls += 1
        raise RuntimeError('socket timeout while posting order')

    def create_limit_order(self, **kwargs):
        self.limit_calls += 1
        return {'orderID': 'unexpected-limit-fallback'}


class _BalanceWrapper(ClobClientWrapper):
    def __init__(self, payload):
        self._payload = payload

    def get_balance_allowance(self):
        return dict(self._payload)


@pytest.fixture(autouse=True)
def _reset_dashboard_host(monkeypatch):
    monkeypatch.setattr(web_app, '_server_host', '127.0.0.1')


def test_get_cash_balance_parses_atomic_units_by_format():
    wrapper = _BalanceWrapper({'balance': '750'})
    assert wrapper.get_cash_balance() == pytest.approx(0.00075)


def test_get_cash_balance_preserves_decimal_strings():
    wrapper = _BalanceWrapper({'balance': '12.34'})
    assert wrapper.get_cash_balance() == pytest.approx(12.34)


def test_public_refresh_prices_is_open_but_other_mutations_require_token(monkeypatch):
    monkeypatch.setattr(web_app, '_server_host', '0.0.0.0')
    monkeypatch.delenv('WEB_DASHBOARD_TOKEN', raising=False)
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

    authorized = client.post(
        '/api/settle-auto',
        headers={'X-Dashboard-Token': 'secret-token'},
    )
    assert authorized.status_code == 200
    assert authorized.get_json()['settled'] == 0


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


def test_market_order_unknown_does_not_fall_back_to_limit_and_journals_state(tmp_path):
    clob = _UnknownMarketOrderClob()
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
            'override_bet_size': 25.0,
        }
    )

    result = executor._place_bet(bet, pd.DataFrame())

    assert result is not None
    assert result['status'] == 'unknown'
    assert clob.market_calls == 1
    assert clob.limit_calls == 0
    # C-4 fix: unknown market orders do NOT charge bankroll to prevent phantom drain
    assert executor.bankroll.bankroll == pytest.approx(500.0)

    ledger_bet = BetLedger(path=tmp_path / 'ledger.json').bets[0]
    assert ledger_bet['status'] == 'open'
    assert ledger_bet['order_type'] == 'market'
    assert ledger_bet['placement_state'] == 'unknown'
    assert ledger_bet['order_id'] is None


def test_name_match_requires_prefix_not_substring():
    assert _name_match('Jon Jones', 'Jonathan Jones') is True
    assert _name_match('Ian Garry', 'Brian Garry') is False
