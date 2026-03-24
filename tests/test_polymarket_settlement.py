import requests

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
    )

    settled = tracker_module.auto_settle_from_polymarket(ledger)
    bet = ledger.get_bets(fresh=True)[0]

    assert settled == 1
    assert seen_identifiers == [condition_id]
    assert bet["status"] == "won"
    assert bet["result_pnl"] == 5.81
