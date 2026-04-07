import copy

import pytest

from src.strategy import duo_trader, llm_operator
from src.web import app as web_app


RAW_LIVE_PNL = {
    "total_invested": 28.3079,
    "current_value": 33.005,
    "unrealized_pnl": 3.8591,
    "realized_pnl": 0.8909,
    "total_pnl": 4.75,
    "num_positions": 4,
    "num_closed": 4,
    "positions": [
        {
            "token_id": "token-live",
            "market": "UFC 999: Alpha vs. Beta",
            "side": "Alpha",
            "opposite_side": "Beta",
            "size": 20.0,
            "avg_price": 0.5,
            "cur_price": 0.55,
            "invested": 10.0,
            "value": 11.0,
            "unrealized_pnl": 1.0,
            "pnl_pct": 10.0,
            "realized_pnl": 0.0,
            "event_slug": "ufc-alpha-beta",
            "end_date": "2026-04-12",
            "icon": "",
            "redeemable": False,
        },
        {
            "token_id": "token-untracked",
            "market": "Will Someone Fight Next?",
            "side": "Yes",
            "opposite_side": "No",
            "size": 50.0,
            "avg_price": 0.35,
            "cur_price": 0.4,
            "invested": 17.5,
            "value": 20.0,
            "unrealized_pnl": 2.5,
            "pnl_pct": 14.2857,
            "realized_pnl": 0.0,
            "event_slug": "someone-fight-next",
            "end_date": "2027-03-07",
            "icon": "",
            "redeemable": False,
        },
        {
            "token_id": "token-dust",
            "market": "UFC 999: Gamma vs. Delta",
            "side": "Gamma",
            "opposite_side": "Delta",
            "size": 0.005,
            "avg_price": 0.68,
            "cur_price": 0.685,
            "invested": 0.0049,
            "value": 0.005,
            "unrealized_pnl": 0.3591,
            "pnl_pct": 0.7353,
            "realized_pnl": -0.3590,
            "event_slug": "ufc-gamma-delta",
            "end_date": "2026-04-12",
            "icon": "",
            "redeemable": False,
        },
        {
            "token_id": "token-redeem",
            "market": "UFC 998: Old vs. Closed",
            "side": "Old",
            "opposite_side": "Closed",
            "size": 2.0,
            "avg_price": 0.4,
            "cur_price": 1.0,
            "invested": 0.803,
            "value": 2.0,
            "unrealized_pnl": 0.0,
            "pnl_pct": 0.0,
            "realized_pnl": 0.5,
            "event_slug": "ufc-old-closed",
            "end_date": "2026-03-29",
            "icon": "",
            "redeemable": True,
        },
    ],
    "timestamp": "2026-04-06T00:00:00+00:00",
}


class FakeMonitor:
    def __init__(self, pnl):
        self._pnl = copy.deepcopy(pnl)

    def compute_pnl(self):
        return copy.deepcopy(self._pnl)


class FakeLedgerView:
    def __init__(self, *, open_bets=None, summary=None):
        self.open_bets = list(open_bets or [])
        self.settled_bets = []
        self.bets = list(self.open_bets)
        self._summary = dict(summary or {})

    def get_summary(self):
        return dict(self._summary)


@pytest.fixture(autouse=True)
def _reset_dashboard_state(monkeypatch):
    web_app._endpoint_cache.clear()
    web_app._endpoint_inflight.clear()
    monkeypatch.setattr(web_app, "_require_read_auth", lambda: None)


def test_api_positions_filters_dust_and_redeemable_positions(monkeypatch):
    monkeypatch.setattr(web_app, "_position_monitor", FakeMonitor(RAW_LIVE_PNL))

    with web_app.app.test_client() as client:
        response = client.get("/api/positions")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["num_positions"] == 2
    assert [row["token_id"] for row in payload["positions"]] == ["token-live", "token-untracked"]
    assert payload["total_invested"] == pytest.approx(27.5)
    assert payload["current_value"] == pytest.approx(31.0)
    assert payload["unrealized_pnl"] == pytest.approx(3.5)
    assert payload["realized_pnl"] == pytest.approx(1.25)
    assert payload["total_pnl"] == pytest.approx(4.75)
    assert all(row["sport"] == "ufc" for row in payload["positions"])


def test_api_summary_uses_filtered_live_positions_for_open_metrics(monkeypatch):
    monkeypatch.setattr(web_app, "_position_monitor", FakeMonitor(RAW_LIVE_PNL))
    monkeypatch.setattr(
        web_app,
        "load_all_trader_ledgers",
        lambda: FakeLedgerView(
            summary={
                "open_bets": 99,
                "unrealized_pnl": 99.0,
                "open_invested": 99.0,
                "realized_pnl": -1.0,
                "total_pnl": -1.0,
                "settled_bets": 1,
                "roi": 0.0,
            }
        ),
    )

    with web_app.app.test_client() as client:
        response = client.get("/api/summary")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["open_bets"] == 2
    assert payload["open_invested"] == pytest.approx(27.5)
    assert payload["unrealized_pnl"] == pytest.approx(3.5)
    assert payload["realized_pnl"] == pytest.approx(1.25)
    assert payload["total_pnl"] == pytest.approx(4.75)
    assert payload["settled_bets"] == 4
    assert payload["roi"] == pytest.approx(4.75 / 28.75)


def test_open_bets_enriched_uses_live_positions_as_source_of_truth(monkeypatch, tmp_path):
    open_bets = [
        {
            "id": 7,
            "fighter": "Alpha",
            "opponent": "Beta",
            "side": "a",
            "amount": 9.5,
            "price": 0.48,
            "shares": 19.0,
            "model_prob": 0.6,
            "market_prob": 0.5,
            "edge": 0.1,
            "reason": "Model edge",
            "placed_at": "2026-04-06T12:00:00+00:00",
            "event_date": "2026-04-12",
            "order_type": "market",
            "token_id": "token-live",
            "market_id": "market-live",
            "_ledger_path": "bet_ledger_single.json",
            "dry_run": False,
            "status": "open",
        },
        {
            "id": 8,
            "fighter": "Stale",
            "opponent": "Gone",
            "side": "a",
            "amount": 4.0,
            "price": 0.4,
            "shares": 10.0,
            "model_prob": 0.52,
            "market_prob": 0.4,
            "edge": 0.12,
            "reason": "Should not render",
            "placed_at": "2026-04-01T12:00:00+00:00",
            "event_date": "2026-04-05",
            "order_type": "market",
            "token_id": "token-stale",
            "market_id": "market-stale",
            "_ledger_path": "bet_ledger_conviction.json",
            "dry_run": False,
            "status": "open",
        },
    ]

    monkeypatch.setattr(web_app, "_position_monitor", FakeMonitor(RAW_LIVE_PNL))
    monkeypatch.setattr(web_app, "load_all_trader_ledgers", lambda: FakeLedgerView(open_bets=open_bets))
    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", tmp_path / "single-missing.json")
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", tmp_path / "conviction-missing.json")
    monkeypatch.setattr(
        duo_trader,
        "ALL_TRADER_LEDGERS",
        [
            ("S", tmp_path / "single-missing.json"),
            ("C", tmp_path / "conviction-missing.json"),
            ("M", tmp_path / "model-missing.json"),
            ("G", tmp_path / "gemini-missing.json"),
        ],
    )
    monkeypatch.setattr(llm_operator, "load_decision_log", lambda: [])

    result = web_app._compute_open_bets_enriched()

    assert [row["token_id"] for row in result["bets"]] == ["token-live", "token-untracked"]
    assert result["unmatched_positions"] == []

    matched = result["bets"][0]
    assert matched["fighter"] == "Alpha"
    assert matched["edge"] == pytest.approx(0.1)
    assert matched["invested"] == pytest.approx(10.0)
    assert matched["avg_price"] == pytest.approx(0.5)
    assert matched["unmatched"] is False

    unmatched = result["bets"][1]
    assert unmatched["fighter"] == "Yes"
    assert unmatched["opponent"] == "No"
    assert unmatched["unmatched"] is True

    token_ids = {row["token_id"] for row in result["bets"]}
    assert "token-stale" not in token_ids
    assert "token-dust" not in token_ids
    assert "token-redeem" not in token_ids


def test_open_bets_enriched_splits_live_position_across_multiple_traders(monkeypatch, tmp_path):
    shared_pnl = copy.deepcopy(RAW_LIVE_PNL)
    shared_pnl["positions"] = [copy.deepcopy(RAW_LIVE_PNL["positions"][0])]
    shared_pnl["positions"][0]["size"] = 20.0
    shared_pnl["positions"][0]["invested"] = 10.0
    shared_pnl["positions"][0]["value"] = 12.0
    shared_pnl["positions"][0]["unrealized_pnl"] = 2.0
    shared_pnl["positions"][0]["pnl_pct"] = 20.0

    open_bets = [
        {
            "id": 7,
            "fighter": "Alpha",
            "opponent": "Beta",
            "side": "a",
            "amount": 4.0,
            "price": 0.5,
            "shares": 8.0,
            "model_prob": 0.6,
            "market_prob": 0.5,
            "edge": 0.1,
            "reason": "Single bet",
            "placed_at": "2026-04-06T12:00:00+00:00",
            "event_date": "2026-04-12",
            "order_type": "market",
            "token_id": "token-live",
            "market_id": "market-live",
            "_ledger_path": "bet_ledger_single.json",
            "dry_run": False,
            "status": "open",
        },
        {
            "id": 8,
            "fighter": "Alpha",
            "opponent": "Beta",
            "side": "a",
            "amount": 6.0,
            "price": 0.5,
            "shares": 12.0,
            "model_prob": 0.58,
            "market_prob": 0.5,
            "edge": 0.08,
            "reason": "Model tracker bet",
            "placed_at": "2026-04-06T12:05:00+00:00",
            "event_date": "2026-04-12",
            "order_type": "market",
            "token_id": "token-live",
            "market_id": "market-live",
            "_ledger_path": "bet_ledger_model_tracker.json",
            "dry_run": False,
            "status": "open",
        },
    ]

    monkeypatch.setattr(web_app, "_position_monitor", FakeMonitor(shared_pnl))
    monkeypatch.setattr(web_app, "load_all_trader_ledgers", lambda: FakeLedgerView(open_bets=open_bets))
    monkeypatch.setattr(
        duo_trader,
        "ALL_TRADER_LEDGERS",
        [
            ("S", tmp_path / "single-missing.json"),
            ("C", tmp_path / "conviction-missing.json"),
            ("M", tmp_path / "model-missing.json"),
            ("G", tmp_path / "gemini-missing.json"),
        ],
    )
    monkeypatch.setattr(llm_operator, "load_decision_log", lambda: [])

    result = web_app._compute_open_bets_enriched()

    assert len(result["bets"]) == 2
    by_trader = {row["trader"]: row for row in result["bets"]}

    assert by_trader["S"]["shares"] == pytest.approx(8.0)
    assert by_trader["S"]["invested"] == pytest.approx(4.0)
    assert by_trader["S"]["value"] == pytest.approx(4.8)
    assert by_trader["S"]["unrealized_pnl"] == pytest.approx(0.8)

    assert by_trader["M"]["shares"] == pytest.approx(12.0)
    assert by_trader["M"]["invested"] == pytest.approx(6.0)
    assert by_trader["M"]["value"] == pytest.approx(7.2)
    assert by_trader["M"]["unrealized_pnl"] == pytest.approx(1.2)
