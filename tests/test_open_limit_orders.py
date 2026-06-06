import json
import logging
import threading
import time

from src.polymarket.tracker import BetLedger
from src.strategy import duo_trader
from src.strategy import llm_operator
from src.web import app as web_app


def _write_ledger(path, bets):
    path.write_text(
        json.dumps({"bets": bets, "last_updated": "2026-03-11T04:47:00"}),
        encoding="utf-8",
    )


class FakeClobClient:
    def __init__(self, open_orders=None, closed_orders=None):
        self._open_orders = open_orders or []
        self._closed_orders = closed_orders or {}

    def get_open_orders(self):
        return list(self._open_orders)

    def get_order(self, order_id):
        return self._closed_orders[order_id]


def test_reconcile_limit_orders_detects_filled_order(tmp_path, monkeypatch):
    """Filled orders not on open-order list are logged (ledger updated by settlement)."""
    single = tmp_path / "bet_ledger_single.json"
    conviction = tmp_path / "bet_ledger_conviction.json"

    _write_ledger(single, [{
        "id": 1,
        "fighter": "Charles Johnson",
        "opponent": "Bruno Silva",
        "side": "a",
        "amount": 31.0,
        "price": 0.62,
        "shares": 50.0,
        "token_id": "token-1",
        "market_id": "market-1",
        "model_prob": 0.65,
        "market_prob": 0.62,
        "edge": 0.03,
        "decimal_odds": 1.6129,
        "dry_run": False,
        "status": "open",
        "placed_at": "2026-03-11T04:47:00",
        "event_date": "2026-03-15",
        "settled_at": None,
        "result_pnl": None,
        "cur_price": None,
        "order_type": "limit_bid",
        "order_id": "order-filled",
    }])
    _write_ledger(conviction, [])

    fake_clob = FakeClobClient(
        open_orders=[],
        closed_orders={
            "order-filled": {
                "id": "order-filled",
                "status": "MATCHED",
                "price": "0.62",
                "original_size": "50",
                "size_matched": "50",
            }
        },
    )

    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", single)
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", conviction)
    monkeypatch.setattr(web_app, "_clob_client", fake_clob)
    web_app._endpoint_cache.clear()

    result = web_app._reconcile_limit_orders_with_clob()

    # Filled orders are detected but not cancelled (settlement handles them)
    assert result["reconciled"] == 1
    assert result["cancelled"] == 0
    # Bet should still be open in ledger (settlement will close it)
    ledger = BetLedger(path=single)
    assert ledger.get_open_bets()[0]["status"] == "open"


def test_reconcile_limit_orders_cancels_cancelled_order(tmp_path, monkeypatch):
    """Orders cancelled on CLOB get cancelled in the ledger."""
    single = tmp_path / "bet_ledger_single.json"
    conviction = tmp_path / "bet_ledger_conviction.json"

    _write_ledger(single, [{
        "id": 2,
        "fighter": "Melissa Mullins",
        "opponent": "Luana Carolina",
        "side": "a",
        "amount": 7.2,
        "price": 0.47,
        "shares": 15.32,
        "token_id": "token-2",
        "market_id": "market-2",
        "model_prob": 0.50,
        "market_prob": 0.47,
        "edge": 0.03,
        "decimal_odds": 2.1277,
        "dry_run": False,
        "status": "open",
        "placed_at": "2026-03-11T04:47:00",
        "event_date": "2026-03-15",
        "settled_at": None,
        "result_pnl": None,
        "cur_price": None,
        "order_type": "near_miss_limit",
        "order_id": "order-cancelled",
    }])
    _write_ledger(conviction, [])

    fake_clob = FakeClobClient(
        open_orders=[],
        closed_orders={
            "order-cancelled": {
                "id": "order-cancelled",
                "status": "CANCELED",
                "price": "0.47",
                "original_size": "15.32",
                "size_matched": "0",
            }
        },
    )

    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", single)
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", conviction)
    monkeypatch.setattr(web_app, "_clob_client", fake_clob)
    web_app._endpoint_cache.clear()

    result = web_app._reconcile_limit_orders_with_clob()

    assert result["reconciled"] == 1
    assert result["cancelled"] == 1
    # Bet should be cancelled in ledger
    ledger = BetLedger(path=single)
    assert len(ledger.get_open_bets()) == 0
    cancelled = [b for b in ledger.bets if b["status"] == "cancelled"]
    assert len(cancelled) == 1


def test_reconcile_limit_orders_keeps_live_order(tmp_path, monkeypatch):
    """Orders still on CLOB open list should not be reconciled/cancelled."""
    single = tmp_path / "bet_ledger_single.json"
    conviction = tmp_path / "bet_ledger_conviction.json"

    _write_ledger(single, [{
        "id": 3,
        "fighter": "Gillian Robertson",
        "opponent": "Amanda Lemos",
        "side": "a",
        "amount": 24.11,
        "price": 0.63,
        "shares": 38.27,
        "token_id": "token-3",
        "market_id": "market-3",
        "model_prob": 0.63,
        "market_prob": 0.60,
        "edge": 0.03,
        "decimal_odds": 1.5873,
        "dry_run": False,
        "status": "open",
        "placed_at": "2026-03-11T04:47:00",
        "event_date": "2026-03-15",
        "settled_at": None,
        "result_pnl": None,
        "cur_price": None,
        "order_type": "limit_bid",
        "order_id": "order-live",
    }])
    _write_ledger(conviction, [])

    fake_clob = FakeClobClient(
        open_orders=[{
            "id": "order-live",
            "asset_id": "token-3",
            "status": "LIVE",
            "price": "0.63",
            "original_size": "38.27",
            "size_matched": "14.73",
            "created_at": 1741668420,
        }],
        closed_orders={},
    )

    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", single)
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", conviction)
    monkeypatch.setattr(web_app, "_clob_client", fake_clob)
    web_app._endpoint_cache.clear()

    result = web_app._reconcile_limit_orders_with_clob()

    # Order is still on CLOB, no reconciliation needed
    assert result["reconciled"] == 0
    assert result["cancelled"] == 0
    # Bet should still be open
    ledger = BetLedger(path=single)
    assert len(ledger.get_open_bets()) == 1


def test_resolve_limit_order_state_marks_filled():
    """_resolve_limit_order_state correctly identifies filled orders."""
    resolved = web_app._resolve_limit_order_state(
        order_data={
            "status": "MATCHED",
            "original_size": "50",
            "size_matched": "50",
        },
        ledger_bet={"shares": 50.0},
        on_clob=False,
    )
    assert resolved["status"] == "filled"
    assert resolved["raw_status"] == "MATCHED"
    assert resolved["size_remaining"] == 0.0
    assert resolved["size_matched"] == 50.0


def test_resolve_limit_order_state_marks_cancelled():
    """_resolve_limit_order_state correctly identifies cancelled orders."""
    resolved = web_app._resolve_limit_order_state(
        order_data={
            "status": "CANCELED",
            "original_size": "15.32",
            "size_matched": "0",
        },
        ledger_bet={"shares": 15.32},
        on_clob=False,
    )
    assert resolved["status"] == "cancelled"
    assert resolved["raw_status"] == "CANCELED"
    assert resolved["size_remaining"] == 0.0
    assert resolved["size_matched"] == 0.0


def test_resolve_limit_order_state_marks_partially_filled_on_clob():
    """_resolve_limit_order_state handles partial fills on active CLOB orders."""
    resolved = web_app._resolve_limit_order_state(
        order_data={
            "status": "LIVE",
            "original_size": "38.27",
            "size_matched": "14.73",
        },
        ledger_bet={"shares": 38.27},
        on_clob=True,
    )
    assert resolved["status"] == "partially_filled"
    assert round(resolved["size_remaining"], 2) == 23.54
    assert round(resolved["size_matched"], 2) == 14.73


def test_compute_limit_orders_display_surfaces_live_clob_order_missing_from_ledger(tmp_path, monkeypatch):
    single = tmp_path / "bet_ledger_single.json"
    conviction = tmp_path / "bet_ledger_conviction.json"

    _write_ledger(single, [{
        "id": 1,
        "fighter": "Tracked Position",
        "opponent": "Opponent",
        "side": "a",
        "amount": 10.0,
        "price": 0.55,
        "shares": 18.18,
        "token_id": "filled-token",
        "market_id": "market-1",
        "model_prob": 0.0,
        "market_prob": 0.55,
        "edge": 0.0,
        "decimal_odds": 1.8182,
        "dry_run": False,
        "status": "open",
        "placed_at": "2026-03-11T04:47:00",
        "event_date": "2026-03-15",
        "settled_at": None,
        "result_pnl": None,
        "cur_price": None,
        "order_type": "imported",
        "order_id": None,
    }])
    _write_ledger(conviction, [])

    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", single)
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", conviction)
    monkeypatch.setattr(web_app, "_get_open_clob_orders", lambda *_args, **_kwargs: [{
        "id": "order-live-only",
        "asset_id": "token-live",
        "status": "LIVE",
        "price": "0.63",
        "original_size": "50",
        "size_matched": "0",
        "created_at": 1741668420,
    }])
    monkeypatch.setattr(web_app, "_build_token_to_fighter_map", lambda: {
        "token-live": {
            "fighter": "Gillian Robertson",
            "opponent": "Amanda Lemos",
            "event_date": "2026-03-15",
        }
    })
    monkeypatch.setattr(llm_operator, "load_decision_log", lambda: [])

    results = web_app._compute_limit_orders_display()

    assert len(results) == 1
    assert results[0]["order_id"] == "order-live-only"
    assert results[0]["fighter"] == "Gillian Robertson"
    assert results[0]["opponent"] == "Amanda Lemos"
    assert results[0]["status"] == "resting"
    assert results[0]["size_remaining"] == 50.0
    assert results[0]["on_clob"] is True
    assert results[0]["status_note"] == "Live on Polymarket but missing from the local ledger"


def test_compute_limit_orders_display_uses_live_clob_truth_for_partial_fill(tmp_path, monkeypatch):
    single = tmp_path / "bet_ledger_single.json"
    conviction = tmp_path / "bet_ledger_conviction.json"

    _write_ledger(single, [{
        "id": 3,
        "fighter": "Gillian Robertson",
        "opponent": "Amanda Lemos",
        "side": "a",
        "amount": 24.11,
        "price": 0.63,
        "shares": 38.27,
        "token_id": "token-3",
        "market_id": "market-3",
        "model_prob": 0.63,
        "market_prob": 0.60,
        "edge": 0.03,
        "decimal_odds": 1.5873,
        "dry_run": False,
        "status": "open",
        "placed_at": "2026-03-11T04:47:00",
        "event_date": "2026-03-15",
        "settled_at": None,
        "result_pnl": None,
        "cur_price": None,
        "order_type": "limit_bid",
        "order_id": "order-live",
    }])
    _write_ledger(conviction, [])

    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", single)
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", conviction)
    monkeypatch.setattr(web_app, "_get_open_clob_orders", lambda *_args, **_kwargs: [{
        "id": "order-live",
        "asset_id": "token-3",
        "status": "LIVE",
        "price": "0.63",
        "original_size": "38.27",
        "size_matched": "14.73",
        "created_at": 1741668420,
    }])
    monkeypatch.setattr(web_app, "_build_token_to_fighter_map", lambda: {})
    monkeypatch.setattr(llm_operator, "load_decision_log", lambda: [])

    results = web_app._compute_limit_orders_display()

    assert len(results) == 1
    assert results[0]["fighter"] == "Gillian Robertson"
    assert results[0]["trader"] == "S"
    assert results[0]["status"] == "partially_filled"
    assert round(results[0]["size_remaining"], 2) == 23.54
    assert round(results[0]["size_matched"], 2) == 14.73
    assert results[0]["edge"] == 0.03


def test_compute_limit_orders_display_matches_empty_clob_exactly(tmp_path, monkeypatch):
    single = tmp_path / "bet_ledger_single.json"
    conviction = tmp_path / "bet_ledger_conviction.json"

    _write_ledger(single, [{
        "id": 4,
        "fighter": "Charles Johnson",
        "opponent": "Bruno Silva",
        "side": "a",
        "amount": 31.0,
        "price": 0.62,
        "shares": 50.0,
        "token_id": "token-1",
        "market_id": "market-1",
        "model_prob": 0.65,
        "market_prob": 0.62,
        "edge": 0.03,
        "decimal_odds": 1.6129,
        "dry_run": False,
        "status": "open",
        "placed_at": "2026-03-11T04:47:00",
        "event_date": "2026-03-15",
        "settled_at": None,
        "result_pnl": None,
        "cur_price": None,
        "order_type": "limit_bid",
        "order_id": "stale-order",
    }])
    _write_ledger(conviction, [])

    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", single)
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", conviction)
    monkeypatch.setattr(web_app, "_get_open_clob_orders", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(web_app, "_build_token_to_fighter_map", lambda: {})
    monkeypatch.setattr(llm_operator, "load_decision_log", lambda: [])

    assert web_app._compute_limit_orders_display() == []


def test_compute_limit_orders_display_raises_when_live_unavailable(tmp_path, monkeypatch):
    single = tmp_path / "bet_ledger_single.json"
    conviction = tmp_path / "bet_ledger_conviction.json"

    _write_ledger(single, [{
        "id": 5,
        "fighter": "Melissa Mullins",
        "opponent": "Luana Carolina",
        "side": "a",
        "amount": 7.2,
        "price": 0.47,
        "shares": 15.32,
        "token_id": "token-2",
        "market_id": "market-2",
        "model_prob": 0.50,
        "market_prob": 0.47,
        "edge": 0.03,
        "decimal_odds": 2.1277,
        "dry_run": False,
        "status": "open",
        "placed_at": "2026-03-11T04:47:00",
        "event_date": "2026-03-15",
        "settled_at": None,
        "result_pnl": None,
        "cur_price": None,
        "order_type": "near_miss_limit",
        "order_id": "order-cancelled",
    }])
    _write_ledger(conviction, [])

    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", single)
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", conviction)
    monkeypatch.setattr(web_app, "_get_open_clob_orders", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_operator, "load_decision_log", lambda: [])

    try:
        web_app._compute_limit_orders_display()
        assert False, "expected live-open-order failure"
    except RuntimeError as exc:
        assert "temporarily unavailable" in str(exc)


def test_api_open_limit_orders_returns_503_when_live_unavailable(monkeypatch):
    web_app.app.config["TESTING"] = True
    monkeypatch.setattr(web_app, "_require_read_auth", lambda: None)
    monkeypatch.setattr(web_app, "_kickoff_limit_order_reconcile", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_app, "_cached", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Polymarket open orders are temporarily unavailable")))

    with web_app.app.test_client() as client:
        response = client.get("/api/open-limit-orders")

    assert response.status_code == 503
    payload = response.get_json()
    assert payload["error"] == "live_open_orders_unavailable"


def test_api_open_limit_orders_unavailable_is_not_warning_noise(monkeypatch, caplog):
    web_app.app.config["TESTING"] = True
    monkeypatch.setattr(web_app, "_require_read_auth", lambda: None)
    monkeypatch.setattr(web_app, "_kickoff_limit_order_reconcile", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        web_app,
        "_cached",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("Polymarket open orders are temporarily unavailable")
        ),
    )
    caplog.set_level(logging.WARNING, logger="src.web.app")

    with web_app.app.test_client() as client:
        response = client.get("/api/open-limit-orders")

    assert response.status_code == 503
    assert not any("Open limit order display unavailable" in record.getMessage() for record in caplog.records)


def test_get_open_clob_orders_retries_once_and_filters_closed_entries(monkeypatch):
    fake_clob = FakeClobClient()
    attempts = {"count": 0}

    def fake_timeout(*_args, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return None
        return [
            {"id": "order-live", "status": "LIVE", "original_size": "10", "size_matched": "0"},
            {"id": "order-filled", "status": "MATCHED", "original_size": "10", "size_matched": "10"},
            {"id": "order-cancelled", "status": "CANCELED", "original_size": "10", "size_matched": "0"},
        ]

    monkeypatch.setattr(web_app, "_clob_client", fake_clob)
    monkeypatch.setattr(web_app, "_clob_call_with_timeout", fake_timeout)

    orders = web_app._get_open_clob_orders(timeout_seconds=0.01)

    assert attempts["count"] == 2
    assert [order["id"] for order in orders] == ["order-live"]


def test_call_with_timeout_does_not_start_duplicate_worker_after_timeout():
    release = threading.Event()
    calls = {"count": 0}
    action = "test duplicate open-order call"

    def slow_call():
        calls["count"] += 1
        release.wait(1)
        return []

    try:
        assert web_app._call_with_timeout(action, slow_call, 0.01) is None
        assert web_app._call_with_timeout(action, slow_call, 0.01) is None
        assert calls["count"] == 1
    finally:
        release.set()
        deadline = time.time() + 1
        while action in web_app._timed_call_inflight and time.time() < deadline:
            time.sleep(0.01)


def test_open_order_timeout_is_not_warning_noise(caplog):
    release = threading.Event()
    action = "fetching open orders"

    def slow_call():
        release.wait(1)
        return []

    caplog.set_level(logging.WARNING, logger="src.web.app")

    try:
        assert web_app._call_with_timeout(action, slow_call, 0.01) is None
        assert not any("Timed out after" in record.getMessage() for record in caplog.records)
    finally:
        release.set()
        deadline = time.time() + 1
        while action in web_app._timed_call_inflight and time.time() < deadline:
            time.sleep(0.01)


def test_reconcile_limit_orders_uses_supplied_open_order_ids_without_refetch(tmp_path, monkeypatch):
    single = tmp_path / "bet_ledger_single.json"
    conviction = tmp_path / "bet_ledger_conviction.json"

    _write_ledger(single, [{
        "id": 6,
        "fighter": "Kevin Holland",
        "opponent": "Randy Brown",
        "side": "a",
        "amount": 7.46,
        "price": 0.45,
        "shares": 16.57,
        "token_id": "token-live",
        "market_id": "market-live",
        "model_prob": 0.0,
        "market_prob": 0.45,
        "edge": 0.0,
        "decimal_odds": 2.2222,
        "dry_run": False,
        "status": "open",
        "placed_at": "2026-04-07T05:28:18+00:00",
        "event_date": "2026-04-11",
        "settled_at": None,
        "result_pnl": None,
        "cur_price": None,
        "order_type": "limit_bid",
        "order_id": "order-live",
    }])
    _write_ledger(conviction, [])

    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", single)
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", conviction)
    monkeypatch.setattr(web_app, "_clob_client", FakeClobClient())
    monkeypatch.setattr(
        web_app,
        "_get_open_clob_orders",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not refetch open orders")),
    )

    result = web_app._reconcile_limit_orders_with_clob(open_order_ids={"order-live"})

    assert result["reconciled"] == 0
    assert result["cancelled"] == 0


def test_api_open_limit_orders_kicks_off_reconcile_with_display_snapshot(monkeypatch):
    web_app.app.config["TESTING"] = True
    captured = {}

    monkeypatch.setattr(web_app, "_require_read_auth", lambda: None)
    monkeypatch.setattr(
        web_app,
        "_cached",
        lambda *_args, **_kwargs: [
            {"order_id": "order-1", "status": "resting"},
            {"order_id": "order-2", "status": "resting"},
            {"order_id": None, "status": "resting"},
        ],
    )
    monkeypatch.setattr(
        web_app,
        "_kickoff_limit_order_reconcile",
        lambda *args, **kwargs: captured.setdefault("open_order_ids", kwargs.get("open_order_ids")),
    )

    with web_app.app.test_client() as client:
        response = client.get("/api/open-limit-orders")

    assert response.status_code == 200
    assert captured["open_order_ids"] == {"order-1", "order-2"}


def test_compute_limit_orders_display_enriches_live_only_orders_with_market_metadata(tmp_path, monkeypatch):
    single = tmp_path / "bet_ledger_single.json"
    conviction = tmp_path / "bet_ledger_conviction.json"

    _write_ledger(single, [])
    _write_ledger(conviction, [])

    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", single)
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", conviction)
    monkeypatch.setattr(web_app, "_get_open_clob_orders", lambda *_args, **_kwargs: [{
        "id": "order-live-only",
        "market": "market-xyz",
        "asset_id": "token-live",
        "status": "LIVE",
        "price": "0.63",
        "original_size": "50",
        "size_matched": "0",
        "outcome": "No",
        "created_at": 1741668420,
    }])
    monkeypatch.setattr(web_app, "_build_token_to_fighter_map", lambda: {})
    monkeypatch.setattr(
        web_app,
        "_fetch_limit_order_market_metadata",
        lambda market_id: {"market_title": "Fight to Go the Distance?", "event_slug": "fight-to-go-distance"},
    )
    monkeypatch.setattr(llm_operator, "load_decision_log", lambda: [])
    web_app._endpoint_cache.clear()

    results = web_app._compute_limit_orders_display()

    assert len(results) == 1
    assert results[0]["fighter"] == "No"
    assert results[0]["market_title"] == "Fight to Go the Distance?"
    assert results[0]["event_slug"] == "fight-to-go-distance"
