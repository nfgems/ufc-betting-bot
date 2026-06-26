import requests
import pytest

from src.polymarket import client as client_module
from src.polymarket import tracker as tracker_module
from src.polymarket.client import GammaClient
from src.polymarket.tracker import BetLedger


class _FakeResponse:
    def __init__(self, *, payload=None, status_code: int = 200):
        self._payload = payload or {}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} error",
                response=self,
            )

    def json(self):
        return self._payload


def test_get_market_routes_condition_ids_to_clob(monkeypatch):
    condition_id = "0xd55659955387624f10617b915604220e01fdaff86e151218cb83d3bb194122f9"
    calls: list[str] = []

    def _fake_get(url, params=None, timeout=30):
        calls.append(url)
        return _FakeResponse(payload={"condition_id": condition_id, "tokens": []})

    monkeypatch.setattr(client_module.requests, "get", _fake_get)

    client = GammaClient(
        base_url="https://gamma-api.polymarket.com",
        clob_base_url="https://clob.polymarket.com",
    )

    market = client.get_market(condition_id)

    assert market == {"condition_id": condition_id, "tokens": []}
    assert calls == [f"https://clob.polymarket.com/markets/{condition_id}"]


def test_get_market_returns_none_without_retrying_non_retryable_422(monkeypatch):
    calls = 0

    def _fake_get(url, params=None, timeout=30):
        nonlocal calls
        calls += 1
        return _FakeResponse(status_code=422)

    monkeypatch.setattr(client_module.requests, "get", _fake_get)

    client = GammaClient(
        base_url="https://gamma-api.polymarket.com",
        clob_base_url="https://clob.polymarket.com",
    )

    assert client.get_market("bad-id") is None
    assert calls == 1


def test_auto_settle_from_polymarket_uses_market_token_winner_flags(monkeypatch, tmp_path):
    condition_id = "0xd55659955387624f10617b915604220e01fdaff86e151218cb83d3bb194122f9"
    seen_identifiers: list[str] = []

    class _FakeGammaClient:
        def get_market(self, market_identifier: str):
            seen_identifiers.append(market_identifier)
            return {
                "tokens": [
                    {"token_id": "winner-token", "winner": True},
                    {"token_id": "loser-token", "winner": False},
                ]
            }

    monkeypatch.setattr(client_module, "GammaClient", _FakeGammaClient)

    ledger = BetLedger(path=tmp_path / "ledger.json")
    ledger.add_bet(
        fighter="Sebastian Korda",
        opponent="Martin Landaluce",
        side="a",
        amount=20.0,
        price=0.775,
        shares=25.81,
        token_id="winner-token",
        market_id="1688367",
        condition_id=condition_id,
        model_prob=0.8710,
        market_prob=0.7602,
        edge=0.1108,
        decimal_odds=1.2903,
        dry_run=False,
        placement_state="filled",
    )

    settled = tracker_module.auto_settle_from_polymarket(ledger)
    bet = ledger.get_bets(fresh=True)[0]

    assert settled == 1
    assert seen_identifiers == [condition_id]
    assert bet["status"] == "won"
    assert bet["result_pnl"] == 5.81


@pytest.mark.parametrize("asset_slug", ["btc", "eth", "sol"])
def test_auto_settle_from_polymarket_skips_unfilled_submitted_orders(monkeypatch, tmp_path, asset_slug):
    condition_id = "0xd55659955387624f10617b915604220e01fdaff86e151218cb83d3bb194122f9"
    seen_identifiers: list[str] = []

    class _FakeGammaClient:
        def get_market(self, market_identifier: str):
            seen_identifiers.append(market_identifier)
            return {
                "tokens": [
                    {"token_id": "winner-token", "winner": True},
                    {"token_id": "loser-token", "winner": False},
                ]
            }

    monkeypatch.setattr(client_module, "GammaClient", _FakeGammaClient)

    ledger = BetLedger(path=tmp_path / "ledger.json")
    ledger.add_bet(
        fighter=f"{asset_slug.upper()} 5m Up",
        opponent=f"{asset_slug.upper()} 5m Down",
        side="up",
        amount=50.0,
        price=0.94,
        shares=53.19,
        token_id="winner-token",
        market_id="2684846",
        condition_id=condition_id,
        model_prob=0.0,
        market_prob=0.935,
        edge=0.0,
        decimal_odds=1.0638,
        dry_run=False,
        order_type="btc5m_marketable_limit",
        order_id="resting-order",
        placement_state="submitted",
        metadata={"market_slug": f"{asset_slug}-updown-5m-1782500100"},
    )

    settled = tracker_module.auto_settle_from_polymarket(ledger)
    bet = ledger.get_bets(fresh=True)[0]

    assert settled == 0
    assert seen_identifiers == []
    assert bet["status"] == "open"
    assert bet["result_pnl"] is None


@pytest.mark.parametrize("asset_slug", ["btc", "eth", "sol"])
def test_auto_settle_from_polymarket_skips_crypto_fill_without_polymarket_source(
    monkeypatch,
    tmp_path,
    asset_slug,
):
    condition_id = "0xd55659955387624f10617b915604220e01fdaff86e151218cb83d3bb194122f9"
    seen_identifiers: list[str] = []

    class _FakeGammaClient:
        def get_market(self, market_identifier: str):
            seen_identifiers.append(market_identifier)
            return {
                "tokens": [
                    {"token_id": "winner-token", "winner": True},
                    {"token_id": "loser-token", "winner": False},
                ]
            }

    monkeypatch.setattr(client_module, "GammaClient", _FakeGammaClient)

    ledger = BetLedger(path=tmp_path / "ledger.json")
    ledger.add_bet(
        fighter=f"{asset_slug.upper()} 5m Up",
        opponent=f"{asset_slug.upper()} 5m Down",
        side="up",
        amount=50.0,
        price=0.94,
        shares=53.19,
        token_id="winner-token",
        market_id="2684846",
        condition_id=condition_id,
        model_prob=0.0,
        market_prob=0.935,
        edge=0.0,
        decimal_odds=1.0638,
        dry_run=False,
        placement_state="filled",
        metadata={
            "market_slug": f"{asset_slug}-updown-5m-1782500100",
            "actual_fill_amount": 50.0,
            "actual_filled_shares": 53.19,
        },
    )

    settled = tracker_module.auto_settle_from_polymarket(ledger)
    bet = ledger.get_bets(fresh=True)[0]

    assert settled == 0
    assert seen_identifiers == []
    assert bet["status"] == "open"
    assert bet["result_pnl"] is None


def test_auto_settle_from_polymarket_settles_crypto_polymarket_activity_fill(monkeypatch, tmp_path):
    condition_id = "0xd55659955387624f10617b915604220e01fdaff86e151218cb83d3bb194122f9"

    class _FakeGammaClient:
        def get_market(self, market_identifier: str):
            return {
                "tokens": [
                    {"token_id": "winner-token", "winner": True},
                    {"token_id": "loser-token", "winner": False},
                ]
            }

    monkeypatch.setattr(client_module, "GammaClient", _FakeGammaClient)

    ledger = BetLedger(path=tmp_path / "ledger.json")
    ledger.add_bet(
        fighter="SOL 5m Up",
        opponent="SOL 5m Down",
        side="up",
        amount=50.0,
        price=0.94,
        shares=53.19,
        token_id="winner-token",
        market_id="2684846",
        condition_id=condition_id,
        model_prob=0.0,
        market_prob=0.935,
        edge=0.0,
        decimal_odds=1.0638,
        dry_run=False,
        order_type="btc5m_marketable_limit",
        placement_state="submitted",
        metadata={
            "actual_fill_amount": 4.7,
            "actual_filled_shares": 5.0,
            "actual_fill_source": "polymarket_activity",
        },
    )

    settled = tracker_module.auto_settle_from_polymarket(ledger)
    bet = ledger.get_bets(fresh=True)[0]

    assert settled == 1
    assert bet["status"] == "won"
    assert bet["result_pnl"] == 0.3


def test_auto_settle_from_polymarket_settles_crypto_clob_trade_history_fill(monkeypatch, tmp_path):
    condition_id = "0xd55659955387624f10617b915604220e01fdaff86e151218cb83d3bb194122f9"

    class _FakeGammaClient:
        def get_market(self, market_identifier: str):
            return {
                "tokens": [
                    {"token_id": "winner-token", "winner": True},
                    {"token_id": "loser-token", "winner": False},
                ]
            }

    monkeypatch.setattr(client_module, "GammaClient", _FakeGammaClient)

    ledger = BetLedger(path=tmp_path / "ledger.json")
    ledger.add_bet(
        fighter="SOL 5m Up",
        opponent="SOL 5m Down",
        side="up",
        amount=50.0,
        price=0.94,
        shares=53.19,
        token_id="winner-token",
        market_id="2684846",
        condition_id=condition_id,
        model_prob=0.0,
        market_prob=0.935,
        edge=0.0,
        decimal_odds=1.0638,
        dry_run=False,
        order_type="btc5m_marketable_limit",
        placement_state="submitted",
        metadata={
            "actual_fill_amount": 4.7,
            "actual_filled_shares": 5.0,
            "actual_fill_source": "clob_trade_history",
        },
    )

    settled = tracker_module.auto_settle_from_polymarket(ledger)
    bet = ledger.get_bets(fresh=True)[0]

    assert settled == 1
    assert bet["status"] == "won"
    assert bet["result_pnl"] == 0.3


def test_auto_settle_from_polymarket_skips_crypto_clob_order_response_fill(monkeypatch, tmp_path):
    condition_id = "0xd55659955387624f10617b915604220e01fdaff86e151218cb83d3bb194122f9"
    seen_identifiers: list[str] = []

    class _FakeGammaClient:
        def get_market(self, market_identifier: str):
            seen_identifiers.append(market_identifier)
            return {
                "tokens": [
                    {"token_id": "winner-token", "winner": True},
                    {"token_id": "loser-token", "winner": False},
                ]
            }

    monkeypatch.setattr(client_module, "GammaClient", _FakeGammaClient)

    ledger = BetLedger(path=tmp_path / "ledger.json")
    ledger.add_bet(
        fighter="SOL 5m Up",
        opponent="SOL 5m Down",
        side="up",
        amount=50.0,
        price=0.94,
        shares=53.19,
        token_id="winner-token",
        market_id="2684846",
        condition_id=condition_id,
        model_prob=0.0,
        market_prob=0.935,
        edge=0.0,
        decimal_odds=1.0638,
        dry_run=False,
        order_type="btc5m_marketable_limit",
        placement_state="submitted",
        metadata={
            "actual_fill_amount": 4.7,
            "actual_filled_shares": 5.0,
            "actual_fill_source": "clob_order_response",
        },
    )

    settled = tracker_module.auto_settle_from_polymarket(ledger)
    bet = ledger.get_bets(fresh=True)[0]

    assert settled == 0
    assert seen_identifiers == []
    assert bet["status"] == "open"
    assert bet["result_pnl"] is None


def test_settle_bet_uses_actual_fill_amount_for_pnl(tmp_path):
    ledger = BetLedger(path=tmp_path / "ledger.json")
    bet = ledger.add_bet(
        fighter="BTC 5m Up",
        opponent="BTC 5m Down",
        side="up",
        amount=50.0,
        price=0.94,
        shares=53.19,
        token_id="winner-token",
        market_id="2684846",
        model_prob=0.0,
        market_prob=0.935,
        edge=0.0,
        decimal_odds=1.0638,
        dry_run=False,
        placement_state="filled",
        metadata={
            "actual_fill_amount": 4.7,
            "actual_filled_shares": 5.0,
            "actual_fill_source": "polymarket_activity",
        },
    )

    result = ledger.settle_bet(bet["id"], won=True)

    assert result.ok
    assert ledger.get_bets(fresh=True)[0]["result_pnl"] == 0.3


def test_summary_uses_only_confirmed_crypto_fill_amounts_for_open_exposure(tmp_path):
    ledger = BetLedger(path=tmp_path / "ledger.json")
    ledger.add_bet(
        fighter="SOL 5m Up",
        opponent="SOL 5m Down",
        side="up",
        amount=50.0,
        price=0.94,
        shares=53.19,
        token_id="resting-token",
        market_id="2684846",
        model_prob=0.0,
        market_prob=0.935,
        edge=0.0,
        decimal_odds=1.0638,
        dry_run=False,
        order_type="btc5m_marketable_limit",
        order_id="resting-order",
        placement_state="submitted",
    )
    filled = ledger.add_bet(
        fighter="SOL 5m Up",
        opponent="SOL 5m Down",
        side="up",
        amount=50.0,
        price=0.94,
        shares=53.19,
        token_id="filled-token",
        market_id="2684846",
        model_prob=0.0,
        market_prob=0.935,
        edge=0.0,
        decimal_odds=1.0638,
        dry_run=False,
        order_type="btc5m_marketable_limit",
        order_id="filled-order",
        placement_state="submitted",
        metadata={
            "actual_fill_amount": 4.7,
            "actual_filled_shares": 5.0,
            "actual_fill_source": "polymarket_activity",
        },
    )
    ledger.update_bet_fields(filled["id"], cur_price=0.99)

    summary = ledger.get_summary(fresh=True)

    assert summary["open_bets"] == 2
    assert summary["open_invested"] == 4.7
    assert summary["unrealized_pnl"] == 0.25


def test_summary_excludes_crypto_clob_order_response_from_open_exposure(tmp_path):
    ledger = BetLedger(path=tmp_path / "ledger.json")
    filled = ledger.add_bet(
        fighter="SOL 5m Up",
        opponent="SOL 5m Down",
        side="up",
        amount=50.0,
        price=0.94,
        shares=53.19,
        token_id="filled-token",
        market_id="2684846",
        model_prob=0.0,
        market_prob=0.935,
        edge=0.0,
        decimal_odds=1.0638,
        dry_run=False,
        order_type="btc5m_marketable_limit",
        order_id="filled-order",
        placement_state="submitted",
        metadata={
            "actual_fill_amount": 4.7,
            "actual_filled_shares": 5.0,
            "actual_fill_source": "clob_order_response",
        },
    )
    ledger.update_bet_fields(filled["id"], cur_price=0.99)

    summary = ledger.get_summary(fresh=True)

    assert summary["open_bets"] == 1
    assert summary["open_invested"] == 0.0
    assert summary["unrealized_pnl"] == 0.0
