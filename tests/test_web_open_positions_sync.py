import copy
import logging
import time

import pytest

from src.polymarket.monitor import PositionDataPartialError
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


class CountingMonitor(FakeMonitor):
    def __init__(self, pnl):
        super().__init__(pnl)
        self.calls = 0

    def compute_pnl(self):
        self.calls += 1
        return super().compute_pnl()


class ProfileMonitor(FakeMonitor):
    def __init__(self, pnl, *, closed_positions=None, trades=None):
        super().__init__(pnl)
        self._closed_positions = copy.deepcopy(closed_positions or [])
        self._trades = copy.deepcopy(trades or [])

    def get_closed_positions(self, *, limit=None, strict=False):
        rows = copy.deepcopy(self._closed_positions)
        if limit is None:
            return rows
        return rows[:limit]

    def get_trades(self, limit=50, *, page_size=500, strict=False):
        rows = copy.deepcopy(self._trades)
        if limit is None:
            return rows
        return rows[:limit]


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
    web_app._profile_snapshot_warning_state.clear()
    monkeypatch.setattr(web_app, "_require_read_auth", lambda: None)
    monkeypatch.setattr(web_app, "_load_polymarket_profile_snapshot", lambda: ({}, "unavailable"))
    monkeypatch.setattr(web_app, "_load_active_ufc_event_slugs", lambda: set())


def test_api_positions_filters_dust_and_redeemable_positions(monkeypatch):
    monkeypatch.setattr(web_app, "_position_monitor", FakeMonitor(RAW_LIVE_PNL))

    with web_app.app.test_client() as client:
        response = client.get("/api/positions")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["num_positions"] == 2
    assert [row["token_id"] for row in payload["positions"]] == ["token-live", "token-untracked"]
    # Headline totals pass through from monitor.compute_pnl() so they match Polymarket.
    assert payload["total_invested"] == pytest.approx(28.3079)
    assert payload["current_value"] == pytest.approx(33.005)
    assert payload["unrealized_pnl"] == pytest.approx(3.8591)
    assert payload["realized_pnl"] == pytest.approx(0.8909)
    assert payload["total_pnl"] == pytest.approx(4.75)
    assert payload["visible_invested"] == pytest.approx(27.5)
    assert payload["visible_value"] == pytest.approx(31.0)
    assert [row["sport"] for row in payload["positions"]] == ["ufc", "other"]


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
    # Totals reflect Polymarket's raw PnL across all positions (open + closed + dust + redeemable).
    assert payload["open_invested"] == pytest.approx(28.3079)
    assert payload["unrealized_pnl"] == pytest.approx(3.8591)
    assert payload["realized_pnl"] == pytest.approx(0.8909)
    assert payload["total_pnl"] == pytest.approx(4.75)
    assert payload["settled_bets"] == 4
    assert payload["roi"] == pytest.approx(4.75 / (28.3079 + 0.8909))


def test_api_summary_sport_ufc_excludes_crypto_and_other_markets(monkeypatch):
    raw_live = copy.deepcopy(RAW_LIVE_PNL)
    raw_live["realized_pnl"] = 104.141
    raw_live["total_pnl"] = 105.5001
    raw_live["closed_positions"] = [
        {
            "market": "UFC 997: Closed vs. Winner",
            "side": "Closed",
            "realized_pnl": 4.0,
            "event_slug": "ufc-closed-winner",
        },
        {
            "market": "Bitcoin Up or Down - June 27, 8:55AM-9:00AM ET",
            "side": "Up",
            "realized_pnl": 100.0,
            "event_slug": "btc-up-or-down-2026-06-27-0855",
        },
    ]
    monkeypatch.setattr(web_app, "_position_monitor", FakeMonitor(raw_live))
    monkeypatch.setattr(
        web_app,
        "load_all_trader_ledgers",
        lambda: FakeLedgerView(summary={"open_bets": 0, "settled_bets": 0}),
    )

    with web_app.app.test_client() as client:
        response = client.get("/api/summary?sport=ufc")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["sport"] == "ufc"
    assert payload["open_bets"] == 1
    assert payload["settled_bets"] == 1
    assert payload["open_invested"] == pytest.approx(10.8079)
    assert payload["unrealized_pnl"] == pytest.approx(1.3591)
    assert payload["realized_pnl"] == pytest.approx(4.141)
    assert payload["total_pnl"] == pytest.approx(5.5001)


def test_api_positions_sport_ufc_filters_non_ufc_markets(monkeypatch):
    monkeypatch.setattr(web_app, "_position_monitor", FakeMonitor(RAW_LIVE_PNL))

    with web_app.app.test_client() as client:
        response = client.get("/api/positions?sport=ufc")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["num_positions"] == 1
    assert payload["excluded_positions"] == 1
    assert [row["token_id"] for row in payload["positions"]] == ["token-live"]
    assert all(row["sport"] == "ufc" for row in payload["positions"])


def test_api_positions_uses_gamma_tags_for_ufc_fight_next_markets(monkeypatch):
    raw_live = copy.deepcopy(RAW_LIVE_PNL)
    raw_live["positions"] = [
        {
            "token_id": "yan-merab",
            "market": "Will Petr Yan fight Merab Dvalishvili next?",
            "side": "Yes",
            "size": 250.0,
            "avg_price": 0.78,
            "cur_price": 0.66,
            "invested": 195.0,
            "value": 165.0,
            "unrealized_pnl": -30.0,
            "event_slug": "who-will-petr-yan-fight-next-before-2027",
            "redeemable": False,
        },
        {
            "token_id": "merab-yan",
            "market": "Will Merab Dvalishvili fight Petr Yan next?",
            "side": "Yes",
            "size": 44.67,
            "avg_price": 0.7652,
            "cur_price": 0.555,
            "invested": 34.18,
            "value": 24.79,
            "unrealized_pnl": -9.39,
            "event_slug": "who-will-merab-dvalishivili-fight-next",
            "redeemable": False,
        },
        {
            "token_id": "boxing-next",
            "market": "Will Boxer A fight Boxer B next?",
            "side": "Yes",
            "size": 10.0,
            "avg_price": 0.5,
            "cur_price": 0.5,
            "invested": 5.0,
            "value": 5.0,
            "unrealized_pnl": 0.0,
            "event_slug": "who-will-boxer-a-fight-next",
            "redeemable": False,
        },
    ]
    monkeypatch.setattr(web_app, "_position_monitor", FakeMonitor(raw_live))
    monkeypatch.setattr(
        web_app,
        "_load_active_ufc_event_slugs",
        lambda: {
            "who-will-petr-yan-fight-next-before-2027",
            "who-will-merab-dvalishivili-fight-next",
        },
    )

    with web_app.app.test_client() as client:
        response = client.get("/api/positions?sport=ufc")

    assert response.status_code == 200
    payload = response.get_json()
    assert [row["token_id"] for row in payload["positions"]] == [
        "yan-merab",
        "merab-yan",
    ]
    assert all(row["sport"] == "ufc" for row in payload["positions"])


def test_api_summary_prefers_polymarket_profile_totals(monkeypatch):
    monkeypatch.setattr(web_app, "_position_monitor", FakeMonitor(RAW_LIVE_PNL))
    monkeypatch.setattr(
        web_app,
        "load_all_trader_ledgers",
        lambda: FakeLedgerView(summary={"total_pnl": 999.0, "open_invested": 1.0}),
    )
    monkeypatch.setattr(
        web_app,
        "_load_polymarket_profile_snapshot",
        lambda: (
            {
                "total_pnl": 49.61547168044103,
                "positions_value": 263.1572,
                "profile_volume": 26279.599464000006,
                "largest_win": 237.631007,
                "predictions": 85,
                "username": "chopboys",
            },
            "live",
        ),
    )

    with web_app.app.test_client() as client:
        response = client.get("/api/summary")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["total_pnl"] == pytest.approx(49.61547168044103)
    assert payload["profile_total_pnl"] == pytest.approx(49.61547168044103)
    assert payload["positions_value"] == pytest.approx(263.1572)
    assert payload["open_invested"] == pytest.approx(263.1572)
    assert payload["profile_volume"] == pytest.approx(26279.599464000006)
    assert payload["profile_largest_win"] == pytest.approx(237.631007)
    assert payload["profile_predictions"] == 85
    assert payload["profile_username"] == "chopboys"
    assert payload["_profile_source"] == "live"


def test_extract_polymarket_profile_snapshot_prefers_all_time_portfolio_pnl():
    next_data = {
        "props": {
            "pageProps": {
                "username": "chopboys",
                "profileSlug": "chopboys",
                "proxyAddress": "0xwallet",
                "dehydratedState": {
                    "queries": [
                        {
                            "queryKey": ["/api/profile/userData", "0xwallet"],
                            "state": {"data": {"name": "chopboys", "proxyWallet": "0xwallet"}},
                        },
                        {
                            "queryKey": ["/api/profile/volume", "0xwallet", "0xwallet"],
                            "state": {"data": {"amount": 26279.599464000006, "pnl": 639.971455}},
                        },
                        {
                            "queryKey": ["user-stats", "0xwallet"],
                            "state": {"data": {"largestWin": 237.631007, "views": 113, "joinDate": "2026-03-08T20:23:49.548000Z"}},
                        },
                        {
                            "queryKey": ["/api/profile/marketsTraded", "0xwallet", "0xwallet"],
                            "state": {"data": {"traded": 85}},
                        },
                        {
                            "queryKey": ["positions", "value", "0xwallet"],
                            "state": {"data": 263.5956},
                        },
                        {
                            "queryKey": ["portfolio-pnl", "chopboys", "0xwallet", "1W"],
                            "state": {"data": [{"t": 1, "p": 3515.12}, {"t": 2, "p": 3527.02}]},
                        },
                        {
                            "queryKey": ["portfolio-pnl", "chopboys", "0xwallet", "ALL"],
                            "state": {"data": [{"t": 1, "p": 646.84}, {"t": 2, "p": 667.17}]},
                        },
                        {
                            "queryKey": ["portfolio-pnl", "chopboys", "0xwallet", "1D"],
                            "state": {"data": [{"t": 1, "p": 612.68}]},
                        },
                    ]
                },
            }
        }
    }
    html = (
        '<script id="__NEXT_DATA__" type="application/json" crossorigin="anonymous">'
        f"{web_app.json.dumps(next_data)}"
        "</script>"
    )

    snapshot = web_app._extract_polymarket_profile_snapshot(html)

    assert snapshot["total_pnl"] == pytest.approx(667.17)
    assert snapshot["pnl_history_source"] == "portfolio-pnl:ALL"
    assert snapshot["pnl_history_all"][-1]["p"] == pytest.approx(667.17)
    assert snapshot["profile_volume"] == pytest.approx(26279.599464000006)
    assert snapshot["positions_value"] == pytest.approx(263.5956)
    assert snapshot["profile_slug"] == "chopboys"
    assert snapshot["username"] == "chopboys"


def test_extract_polymarket_profile_snapshot_reads_streamed_next_payload():
    queries = [
        {
            "queryKey": ["/api/profile/userData", "0xwallet"],
            "state": {"data": {"name": "chopboys", "proxyWallet": "0xwallet"}},
        },
        {
            "queryKey": ["/api/profile/volume", "0xwallet", "0xwallet"],
            "state": {"data": {"amount": 113556.790837, "pnl": 587.808666}},
        },
        {
            "queryKey": ["portfolio-pnl", "0xwallet", "1W"],
            "state": {"data": [{"t": 1782558000, "p": 3527.02}]},
        },
        {
            "queryKey": ["portfolio-pnl", "0xwallet", "ALL"],
            "state": {"data": [{"t": 1782558000, "p": 607.6828}]},
        },
        {
            "queryKey": ["/api/profile/marketsTraded", "0xwallet", "0xwallet"],
            "state": {"data": {"traded": 451}},
        },
        {
            "queryKey": ["user-stats", "0xwallet"],
            "state": {"data": {"largestWin": 536.532724, "views": 839}},
        },
    ]
    streamed = web_app.json.dumps({
        "dehydratedState": {
            "queries": queries,
        }
    })
    html = (
        "<script>self.__next_f.push("
        f"{web_app.json.dumps([1, streamed])}"
        ")</script>"
    )

    snapshot = web_app._extract_polymarket_profile_snapshot(html)

    assert snapshot["total_pnl"] == pytest.approx(607.6828)
    assert snapshot["pnl_history_source"] == "portfolio-pnl:ALL"
    assert snapshot["profile_volume"] == pytest.approx(113556.790837)
    assert snapshot["predictions"] == 451
    assert snapshot["largest_win"] == pytest.approx(536.532724)
    assert snapshot["username"] == "chopboys"


def test_extract_polymarket_profile_snapshot_uses_volume_before_short_range_chart():
    next_data = {
        "props": {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "queryKey": ["/api/profile/volume", "0xwallet", "0xwallet"],
                            "state": {"data": {"amount": 113998.970834, "pnl": 639.971455}},
                        },
                        {
                            "queryKey": ["portfolio-pnl", "0xwallet", "1W"],
                            "state": {"data": [{"t": 1, "p": 3527.02}]},
                        },
                    ]
                },
            }
        }
    }
    html = (
        '<script id="__NEXT_DATA__" type="application/json" crossorigin="anonymous">'
        f"{web_app.json.dumps(next_data)}"
        "</script>"
    )

    snapshot = web_app._extract_polymarket_profile_snapshot(html)

    assert snapshot["total_pnl"] == pytest.approx(639.971455)
    assert snapshot["pnl_history_source"] == "/api/profile/volume"


def test_profile_snapshot_warning_is_debounced(monkeypatch, caplog):
    now = [100.0]
    monkeypatch.setattr(web_app.time, "monotonic", lambda: now[0])
    caplog.set_level(logging.DEBUG, logger=web_app.logger.name)

    exc = RuntimeError("Polymarket profile page did not contain __NEXT_DATA__")
    prefix = "Polymarket profile snapshot unavailable"

    web_app._log_polymarket_profile_snapshot_warning(prefix, exc)
    web_app._log_polymarket_profile_snapshot_warning(prefix, exc)
    now[0] += web_app.PROFILE_PAGE_FAILURE_LOG_TTL + 0.1
    web_app._log_polymarket_profile_snapshot_warning(prefix, exc)

    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and prefix in record.getMessage()
    ]
    debug_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG and "suppressed duplicate" in record.getMessage()
    ]

    assert len(warning_messages) == 2
    assert len(debug_messages) == 1


def test_api_profile_bets_groups_partial_exit_into_single_closed_row(monkeypatch):
    raw_live = {
        "total_invested": 20.0,
        "current_value": 22.0,
        "unrealized_pnl": 2.0,
        "realized_pnl": 119.35,
        "total_pnl": 121.35,
        "num_positions": 1,
        "num_closed": 1,
        "positions": [],
        "timestamp": "2026-04-19T00:00:00+00:00",
    }
    closed_positions = [
        {
            "asset": "closed-token",
            "conditionId": "cond-lakers",
            "title": "Rockets vs. Lakers",
            "outcome": "Lakers",
            "realizedPnl": 119.35,
            "endDate": "2026-04-19T00:00:00Z",
        }
    ]
    trades = [
        {
            "asset": "closed-token",
            "conditionId": "cond-lakers",
            "title": "Rockets vs. Lakers",
            "outcome": "Lakers",
            "side": "BUY",
            "type": "TRADE",
            "usdcSize": 760.5732,
            "size": 824.7259,
            "price": 0.92,
            "timestamp": 1776567054,
        },
        {
            "asset": "closed-token",
            "conditionId": "cond-lakers",
            "title": "Rockets vs. Lakers",
            "outcome": "Lakers",
            "side": "BUY",
            "type": "TRADE",
            "usdcSize": 80.0,
            "size": 99.4,
            "price": 0.80,
            "timestamp": 1776566500,
        },
        {
            "asset": "closed-token",
            "conditionId": "cond-lakers",
            "title": "Rockets vs. Lakers",
            "outcome": "Lakers",
            "side": "SELL",
            "type": "TRADE",
            "usdcSize": 1021.92637,
            "size": 1022.98,
            "price": 0.999,
            "timestamp": 1776568660,
        },
    ]
    ledger_bets = [
        {
            "id": 101,
            "fighter": "Lakers",
            "opponent": "Rockets",
            "amount": 760.5732,
            "shares": 824.7259,
            "placed_at": "2026-04-18T00:10:54+00:00",
            "token_id": "closed-token",
            "_ledger_path": "bet_ledger_single.json",
            "status": "open",
            "reason": "First fill",
        },
        {
            "id": 102,
            "fighter": "Lakers",
            "opponent": "Rockets",
            "amount": 80.0,
            "shares": 99.4,
            "placed_at": "2026-04-18T00:01:40+00:00",
            "token_id": "closed-token",
            "_ledger_path": "bet_ledger_single.json",
            "status": "open",
            "reason": "Second fill",
        },
    ]
    profile_monitor = ProfileMonitor(raw_live, closed_positions=closed_positions, trades=trades)

    monkeypatch.setattr(web_app, "_position_monitor", profile_monitor)
    monkeypatch.setattr(
        web_app,
        "load_all_trader_ledgers",
        lambda: FakeLedgerView(open_bets=ledger_bets, summary={}),
    )
    monkeypatch.setattr(
        web_app,
        "_compute_open_bets_enriched",
        lambda: {
            "bets": [
                {
                    "fighter": "Open Fighter",
                    "opponent": "Open Opponent",
                    "market": "Open Market",
                    "invested": 20.0,
                    "shares": 40.0,
                    "placed_at": "2026-04-19T10:00:00+00:00",
                    "event_date": "2026-04-26",
                    "token_id": "open-token",
                    "trader": "S",
                    "traders": ["S"],
                    "cur_price": 0.55,
                    "avg_price": 0.50,
                    "unrealized_pnl": 2.0,
                    "pnl_pct": 10.0,
                }
            ],
            "_pnl_source": "live",
        },
    )
    monkeypatch.setattr(
        web_app,
        "_load_polymarket_profile_snapshot",
        lambda: (
            {
                "total_pnl": 49.61547168044103,
                "positions_value": 263.1572,
                "profile_volume": 26279.599464000006,
                "largest_win": 237.631007,
                "predictions": 85,
                "username": "chopboys",
            },
            "live",
        ),
    )

    with web_app.app.test_client() as client:
        response = client.get("/api/profile-bets")

    assert response.status_code == 200
    payload = response.get_json()
    summary = payload["summary"]
    assert summary["_canonical_profile"] is True
    assert summary["_pnl_degraded"] is False
    assert summary["total_bets"] == 2
    assert summary["open_bets"] == 1
    assert summary["settled_bets"] == 1
    assert summary["wins"] == 1
    assert summary["losses"] == 0
    assert summary["win_rate"] == pytest.approx(1.0)
    assert summary["realized_pnl"] == pytest.approx(119.35)
    assert summary["unrealized_pnl"] == pytest.approx(2.0)
    assert summary["total_pnl"] == pytest.approx(49.61547168044103)
    assert summary["profile_total_pnl"] == pytest.approx(49.61547168044103)
    assert summary["total_wagered"] == pytest.approx(860.5732)
    assert summary["positions_value"] == pytest.approx(263.1572)
    assert summary["profile_volume"] == pytest.approx(26279.599464000006)
    assert summary["profile_largest_win"] == pytest.approx(237.631007)
    assert summary["profile_predictions"] == 85
    assert summary["_profile_source"] == "live"

    closed_rows = [row for row in payload["bets"] if row["status"] == "won"]
    assert len(closed_rows) == 1
    closed_row = closed_rows[0]
    assert closed_row["fighter"] == "Lakers"
    assert closed_row["opponent"] == "Rockets"
    assert closed_row["amount"] == pytest.approx(840.5732)
    assert closed_row["result_pnl"] == pytest.approx(119.35)
    assert closed_row["token_id"] == "closed-token"
    assert closed_row["trader_label"] == "S"


def test_profile_bets_fallback_preserves_live_profile_headlines(monkeypatch):
    profile_snapshot = {
        "total_pnl": 331.77264,
        "positions_value": 273.0305,
        "profile_volume": 131808.648513,
        "largest_win": 536.532724,
        "predictions": 564,
        "username": "chopboys",
    }
    ledger = FakeLedgerView(
        open_bets=[
            {
                "id": 7,
                "status": "open",
                "fighter": "Stale Ledger Fighter",
                "amount": 4.0,
                "placed_at": "2026-07-16T21:18:58+00:00",
            }
        ],
        summary={"total_pnl": 0.0, "open_invested": 4.0},
    )

    monkeypatch.setattr(web_app, "load_all_trader_ledgers", lambda: ledger)
    monkeypatch.setattr(
        web_app,
        "_load_live_pnl_snapshot",
        lambda: (web_app._empty_live_pnl_snapshot(), "unavailable"),
    )
    monkeypatch.setattr(
        web_app,
        "_load_polymarket_profile_snapshot",
        lambda: (copy.deepcopy(profile_snapshot), "live"),
    )
    monkeypatch.setattr(
        web_app,
        "_compute_open_bets_enriched",
        lambda: {"bets": [], "_pnl_source": "unavailable"},
    )
    monkeypatch.setattr(
        web_app,
        "_position_monitor",
        ProfileMonitor(web_app._empty_live_pnl_snapshot()),
    )

    payload = web_app._compute_profile_bets_snapshot()
    summary = payload["summary"]

    assert summary["_canonical_profile"] is False
    assert summary["_pnl_degraded"] is True
    assert summary["_profile_source"] == "live"
    assert summary["profile_total_pnl"] == pytest.approx(331.77264)
    assert summary["positions_value"] == pytest.approx(273.0305)
    assert summary["profile_volume"] == pytest.approx(131808.648513)
    assert summary["profile_largest_win"] == pytest.approx(536.532724)
    assert summary["profile_predictions"] == 564
    assert summary["profile_username"] == "chopboys"


def test_scope_profile_bets_preserves_global_profile_total_pnl():
    payload = {
        "summary": {
            "_canonical_profile": True,
            "total_pnl": 667.17,
            "profile_total_pnl": 667.17,
        },
        "bets": [
            {
                "status": "won",
                "sport": "ufc",
                "amount": 10.0,
                "result_pnl": 4.0,
            },
            {
                "status": "open",
                "sport": "ufc",
                "amount": 20.0,
                "invested": 20.0,
                "unrealized_pnl": 1.5,
            },
            {
                "status": "won",
                "sport": "crypto",
                "amount": 50.0,
                "result_pnl": 100.0,
            },
        ],
    }

    scoped = web_app._scope_profile_bets_payload(payload, "ufc")
    summary = scoped["summary"]

    assert summary["total_pnl"] == pytest.approx(5.5)
    assert summary["profile_total_pnl"] == pytest.approx(667.17)
    assert summary["total_bets"] == 2


def test_profile_closed_row_uses_closed_position_cost_when_activity_is_unavailable():
    row = web_app._profile_closed_row(
        {
            "asset": "closed-token",
            "conditionId": "cond-alpha",
            "title": "Alpha vs. Beta",
            "outcome": "Alpha",
            "totalBought": 1201.458047,
            "avgPrice": 0.552432,
            "realizedPnl": 536.532724,
            "endDate": "2026-05-10T00:00:00Z",
        },
        matched_bets=[],
        trades=[],
        redeem_trades=[],
    )

    assert row["status"] == "won"
    assert row["fighter"] == "Alpha"
    assert row["amount"] == pytest.approx(663.723872)
    assert row["shares"] == pytest.approx(1201.458047)
    assert row["result_pnl"] == pytest.approx(536.532724)


def test_api_profile_bets_infers_activity_only_losing_tokens(monkeypatch):
    raw_live = {
        "total_invested": 0.0,
        "current_value": 0.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": -2.0,
        "total_pnl": -2.0,
        "num_positions": 0,
        "num_closed": 1,
        "positions": [],
        "timestamp": "2026-04-19T00:00:00+00:00",
    }
    closed_positions = [
        {
            "asset": "winner-token",
            "conditionId": "cond-fight",
            "title": "Alpha vs. Beta",
            "outcome": "Alpha",
            "realizedPnl": 6.0,
            "endDate": "2026-04-19T00:00:00Z",
        }
    ]
    trades = [
        {
            "asset": "winner-token",
            "conditionId": "cond-fight",
            "title": "Alpha vs. Beta",
            "outcome": "Alpha",
            "side": "BUY",
            "type": "TRADE",
            "usdcSize": 4.0,
            "size": 10.0,
            "price": 0.4,
            "timestamp": 1776567000,
        },
        {
            "asset": "loser-token",
            "conditionId": "cond-fight",
            "title": "Alpha vs. Beta",
            "outcome": "Beta",
            "side": "BUY",
            "type": "TRADE",
            "usdcSize": 8.0,
            "size": 16.0,
            "price": 0.5,
            "timestamp": 1776567100,
        },
    ]

    monkeypatch.setattr(web_app, "_position_monitor", ProfileMonitor(raw_live, closed_positions=closed_positions, trades=trades))
    monkeypatch.setattr(web_app, "load_all_trader_ledgers", lambda: FakeLedgerView(summary={}))
    monkeypatch.setattr(web_app, "_compute_open_bets_enriched", lambda: {"bets": [], "_pnl_source": "live"})

    with web_app.app.test_client() as client:
        response = client.get("/api/profile-bets")

    assert response.status_code == 200
    payload = response.get_json()
    summary = payload["summary"]
    assert summary["settled_bets"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["win_rate"] == pytest.approx(0.5)

    loss_rows = [row for row in payload["bets"] if row["status"] == "lost"]
    assert len(loss_rows) == 1
    loss_row = loss_rows[0]
    assert loss_row["fighter"] == "Beta"
    assert loss_row["token_id"] == "loser-token"
    assert loss_row["amount"] == pytest.approx(8.0)
    assert loss_row["result_pnl"] == pytest.approx(-8.0)
    assert loss_row["source"] == "polymarket_activity"


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


def test_open_bets_enriched_groups_multi_trader_positions(monkeypatch, tmp_path):
    raw_live = copy.deepcopy(RAW_LIVE_PNL)
    raw_live["positions"][0].update(
        {
            "size": 25.0,
            "invested": 12.5,
            "value": 14.0,
            "unrealized_pnl": 1.5,
            "pnl_pct": 12.0,
        }
    )

    open_bets = [
        {
            "id": 11,
            "fighter": "Alpha",
            "opponent": "Beta",
            "side": "a",
            "amount": 4.0,
            "price": 0.47,
            "shares": 10.0,
            "model_prob": 0.62,
            "market_prob": 0.5,
            "edge": 0.12,
            "reason": "Single edge",
            "placed_at": "2026-04-04T12:00:00+00:00",
            "event_date": "2026-04-12",
            "order_type": "market",
            "token_id": "token-live",
            "market_id": "market-live",
            "_ledger_path": "bet_ledger_single.json",
            "dry_run": False,
            "status": "open",
        },
        {
            "id": 12,
            "fighter": "Alpha",
            "opponent": "Beta",
            "side": "a",
            "amount": 3.0,
            "price": 0.49,
            "shares": 6.0,
            "model_prob": 0.58,
            "market_prob": 0.5,
            "edge": 0.08,
            "reason": "Conviction add",
            "placed_at": "2026-04-05T12:00:00+00:00",
            "event_date": "2026-04-12",
            "order_type": "market",
            "token_id": "token-live",
            "market_id": "market-live",
            "_ledger_path": "bet_ledger_conviction.json",
            "dry_run": False,
            "status": "open",
        },
        {
            "id": 13,
            "fighter": "Alpha",
            "opponent": "Beta",
            "side": "a",
            "amount": 2.0,
            "price": 0.51,
            "shares": 4.0,
            "model_prob": 0.57,
            "market_prob": 0.5,
            "edge": 0.07,
            "reason": "Tracker add",
            "placed_at": "2026-04-06T12:00:00+00:00",
            "event_date": "2026-04-12",
            "order_type": "market",
            "token_id": "token-live",
            "market_id": "market-live",
            "_ledger_path": "bet_ledger_model_tracker.json",
            "dry_run": False,
            "status": "open",
        },
    ]

    monkeypatch.setattr(web_app, "_position_monitor", FakeMonitor(raw_live))
    monkeypatch.setattr(web_app, "load_all_trader_ledgers", lambda: FakeLedgerView(open_bets=open_bets))
    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", tmp_path / "single-missing.json")
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", tmp_path / "conviction-missing.json")
    monkeypatch.setattr(llm_operator, "load_decision_log", lambda: [])

    result = web_app._compute_open_bets_enriched()

    token_live_rows = [row for row in result["bets"] if row["token_id"] == "token-live"]
    assert len(token_live_rows) == 1

    grouped = token_live_rows[0]
    assert grouped["fighter"] == "Alpha"
    assert grouped["avg_price"] == pytest.approx(0.5)
    assert grouped["invested"] == pytest.approx(12.5)
    assert grouped["shares"] == pytest.approx(25.0)
    assert grouped["tracked_shares"] == pytest.approx(20.0)
    assert grouped["manual_shares"] == pytest.approx(5.0)
    assert grouped["manual_untracked"] is True
    assert grouped["matched_bet_count"] == 3
    assert grouped["unmatched"] is False
    assert grouped["edge"] == pytest.approx((0.12 * 4.0 + 0.08 * 3.0 + 0.07 * 2.0) / 9.0)
    assert set(grouped["traders"]) == {"S", "C", "M"}


def test_operator_decision_match_does_not_attach_opposite_side_rationale():
    decisions_index = web_app._build_decisions_index(
        [
            {
                "timestamp": "2026-05-28T10:56:15+00:00",
                "decision_context": "S",
                "fighter_a": "Luis Felipe Dias",
                "fighter_b": "Yi Sak Lee",
                "event_date": "2026-05-30",
                "bet_on": "Yi Sak Lee",
                "rationale": "Lee rationale",
            }
        ]
    )
    bet = {
        "fighter": "Luis Felipe Dias",
        "opponent": "Yi Sak Lee",
        "event_date": "2026-05-30",
        "_ledger_path": "bet_ledger_single.json",
    }

    assert web_app._match_decision_to_bet(bet, decisions_index) is None


def test_operator_decision_match_accepts_alias_for_selected_fighter():
    decision = {
        "timestamp": "2026-05-28T10:56:15+00:00",
        "decision_context": "S",
        "fighter_a": "Luis Dias de Assis",
        "fighter_b": "Yi Sak Lee",
        "event_date": "2026-05-30",
        "bet_on": "Luis Dias de Assis",
        "rationale": "Luis rationale",
    }
    decisions_index = web_app._build_decisions_index([decision])
    bet = {
        "fighter": "Luis Felipe Dias",
        "opponent": "Yi Sak Lee",
        "event_date": "2026-05-30",
        "_ledger_path": "bet_ledger_single.json",
    }

    assert web_app._match_decision_to_bet(bet, decisions_index) == decision


def test_gemini_open_bet_confidence_is_capped_and_labeled_as_signal():
    position = {
        "side": "Loma Lookboonmee",
        "opposite_side": "Jaqueline Amorim",
        "size": 4.255,
        "invested": 2.0,
    }
    matched_bets = [
        {
            "fighter": "Loma Lookboonmee",
            "opponent": "Jaqueline Amorim",
            "amount": 2.0,
            "shares": 4.255,
            "model_prob": 0.0,
            "market_prob": 0.47,
            "edge": 0.0,
            "_ledger_path": "bet_ledger_single.json",
            "order_type": "imported",
            "status": "open",
        },
        {
            "fighter": "Loma Lookboonmee",
            "opponent": "Jaqueline Amorim",
            "amount": 2.0,
            "shares": 4.255,
            "model_prob": 0.465,
            "signal_confidence": 0.85,
            "market_prob": 0.465,
            "edge": 0.0,
            "_ledger_path": "bet_ledger_gemini_tracker.json",
            "probability_source": "market_neutral",
            "status": "open",
        }
    ]

    entry = web_app._aggregate_open_bet_position(position, matched_bets)

    assert entry["model_label"] == "Confidence"
    assert entry["model_prob"] is None
    assert entry["signal_confidence"] == pytest.approx(0.85)
    assert entry["edge"] is None


def test_dashboard_live_pnl_snapshot_is_cached_across_homepage_endpoints(monkeypatch):
    monitor = CountingMonitor(RAW_LIVE_PNL)
    monkeypatch.setattr(web_app, "_position_monitor", monitor)
    monkeypatch.setattr(
        web_app,
        "load_all_trader_ledgers",
        lambda: FakeLedgerView(summary={"open_bets": 0, "realized_pnl": 0.0, "total_pnl": 0.0}),
    )
    monkeypatch.setattr(duo_trader, "get_all_trader_ledgers", lambda: [])
    monkeypatch.setattr(llm_operator, "load_decision_log", lambda: [])

    with web_app.app.test_client() as client:
        assert client.get("/api/summary").status_code == 200
        assert client.get("/api/positions").status_code == 200

    result = web_app._compute_open_bets_enriched()

    assert result["_pnl_source"] == "live"
    assert monitor.calls == 1


def test_api_summary_uses_stale_snapshot_after_live_pnl_compute_error(monkeypatch):
    class FlakyMonitor(FakeMonitor):
        def __init__(self, pnl):
            super().__init__(pnl)
            self.calls = 0

        def compute_pnl(self):
            self.calls += 1
            if self.calls > 1:
                raise PositionDataPartialError("positions pagination failed")
            return super().compute_pnl()

    monitor = FlakyMonitor(RAW_LIVE_PNL)
    monkeypatch.setattr(web_app, "_position_monitor", monitor)
    monkeypatch.setattr(
        web_app,
        "load_all_trader_ledgers",
        lambda: FakeLedgerView(summary={"open_bets": 0, "realized_pnl": 0.0, "total_pnl": 0.0}),
    )

    with web_app.app.test_client() as client:
        first = client.get("/api/summary")
        assert first.status_code == 200
        web_app._endpoint_cache["dashboard-live-pnl"]["ts"] = 0
        second = client.get("/api/summary")

    payload = second.get_json()
    assert payload["open_bets"] == 2
    assert payload["total_pnl"] == pytest.approx(4.75)
    assert payload["_pnl_degraded"] is True
    assert payload["_pnl_source"] == "stale"

    deadline = time.time() + 1.0
    while monitor.calls < 2 and time.time() < deadline:
        time.sleep(0.01)
    assert monitor.calls == 2


def test_api_summary_preserves_ledger_values_when_live_pnl_is_unavailable(monkeypatch):
    ledger_summary = {
        "open_bets": 3,
        "unrealized_pnl": 7.5,
        "open_invested": 14.0,
        "realized_pnl": 2.0,
        "total_pnl": 9.5,
        "settled_bets": 8,
        "roi": 0.25,
    }
    monkeypatch.setattr(
        web_app,
        "_load_live_pnl_snapshot",
        lambda: (web_app._empty_live_pnl_snapshot(), "unavailable"),
    )
    monkeypatch.setattr(
        web_app,
        "load_all_trader_ledgers",
        lambda: FakeLedgerView(summary=ledger_summary),
    )

    with web_app.app.test_client() as client:
        response = client.get("/api/summary")

    assert response.status_code == 200
    payload = response.get_json()
    for key, value in ledger_summary.items():
        assert payload[key] == value
    assert payload["_pnl_degraded"] is True
    assert payload["_pnl_source"] == "unavailable"


def test_api_closed_positions_returns_503_on_partial_history(monkeypatch):
    class PartialClosedMonitor:
        def get_closed_positions(self, *, limit=None, strict=False):
            assert strict is True
            raise PositionDataPartialError("closed positions pagination failed")

    monkeypatch.setattr(web_app, "_position_monitor", PartialClosedMonitor())

    with web_app.app.test_client() as client:
        response = client.get("/api/closed-positions")

    assert response.status_code == 503
    assert response.get_json() == {"error": "closed positions unavailable"}


def test_api_trade_history_limit_zero_requests_full_activity(monkeypatch):
    calls = []

    class RecordingMonitor:
        def get_trades(self, limit=50, *, page_size=500, strict=False):
            calls.append({"limit": limit, "page_size": page_size, "strict": strict})
            return [{"type": "TRADE", "side": "BUY", "timestamp": 1776617160}]

    monkeypatch.setattr(web_app, "_position_monitor", RecordingMonitor())

    with web_app.app.test_client() as client:
        response = client.get("/api/trade-history?limit=0")

    assert response.status_code == 200
    assert response.get_json() == [{"type": "TRADE", "side": "BUY", "timestamp": 1776617160}]
    assert calls == [{"limit": None, "page_size": 500, "strict": True}]


def test_api_trade_history_returns_503_on_partial_activity(monkeypatch):
    class PartialTradeMonitor:
        def get_trades(self, limit=50, *, page_size=500, strict=False):
            assert limit is None
            assert strict is True
            raise PositionDataPartialError("activity pagination failed")

    monkeypatch.setattr(web_app, "_position_monitor", PartialTradeMonitor())

    with web_app.app.test_client() as client:
        response = client.get("/api/trade-history?limit=0")

    assert response.status_code == 503
    assert response.get_json() == {"error": "trade history unavailable"}


def test_open_bets_enriched_falls_back_to_ledger_when_live_positions_are_unavailable(monkeypatch):
    open_bets = [
        {
            "id": 41,
            "fighter": "Alpha",
            "opponent": "Beta",
            "side": "a",
            "amount": 9.5,
            "price": 0.48,
            "shares": 19.0,
            "cur_price": 0.52,
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
        }
    ]

    monkeypatch.setattr(
        web_app,
        "_load_live_pnl_snapshot",
        lambda: (web_app._empty_live_pnl_snapshot(), "unavailable"),
    )
    monkeypatch.setattr(web_app, "load_all_trader_ledgers", lambda: FakeLedgerView(open_bets=open_bets))
    monkeypatch.setattr(duo_trader, "get_all_trader_ledgers", lambda: [])
    monkeypatch.setattr(llm_operator, "load_decision_log", lambda: [])

    result = web_app._compute_open_bets_enriched()

    assert result["_pnl_source"] == "unavailable"
    assert [row["token_id"] for row in result["bets"]] == ["token-live"]
    entry = result["bets"][0]
    assert entry["fighter"] == "Alpha"
    assert entry["matched_bet_count"] == 1
    assert entry["unmatched"] is False
    assert entry["invested"] == pytest.approx(9.5)
    assert entry["unrealized_pnl"] == pytest.approx((19.0 * 0.52) - 9.5)


def test_open_bets_enriched_does_not_resurrect_ledger_rows_when_live_is_empty(monkeypatch):
    open_bets = [
        {
            "id": 42,
            "fighter": "Stale",
            "opponent": "Ledger",
            "amount": 5.0,
            "shares": 10.0,
            "token_id": "token-stale",
            "_ledger_path": "bet_ledger_single.json",
            "status": "open",
        }
    ]
    monkeypatch.setattr(
        web_app,
        "_load_live_pnl_snapshot",
        lambda: (web_app._empty_live_pnl_snapshot(), "live"),
    )
    monkeypatch.setattr(web_app, "load_all_trader_ledgers", lambda: FakeLedgerView(open_bets=open_bets))
    monkeypatch.setattr(duo_trader, "get_all_trader_ledgers", lambda: [])
    monkeypatch.setattr(llm_operator, "load_decision_log", lambda: [])

    result = web_app._compute_open_bets_enriched()

    assert result["_pnl_source"] == "live"
    assert result["bets"] == []


def test_open_bets_enriched_does_not_reconcile_from_stale_snapshot(monkeypatch):
    open_bets = [
        {
            "id": 51,
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
        }
    ]
    reconcile_calls = []

    monkeypatch.setattr(
        web_app,
        "_load_live_pnl_snapshot",
        lambda: (copy.deepcopy(RAW_LIVE_PNL), "stale"),
    )
    monkeypatch.setattr(web_app, "load_all_trader_ledgers", lambda: FakeLedgerView(open_bets=open_bets))
    monkeypatch.setattr(
        web_app,
        "auto_reconcile_sold_positions",
        lambda *args, **kwargs: reconcile_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(llm_operator, "load_decision_log", lambda: [])

    result = web_app._compute_open_bets_enriched()

    assert result["_pnl_source"] == "stale"
    assert reconcile_calls == []
    assert [row["token_id"] for row in result["bets"]] == ["token-live", "token-untracked"]
