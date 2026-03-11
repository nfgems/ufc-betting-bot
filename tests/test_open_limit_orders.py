import json

from src.strategy import duo_trader
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


def test_compute_open_limit_orders_marks_closed_match_as_filled(tmp_path, monkeypatch):
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
    monkeypatch.setattr(web_app, "_build_token_to_fighter_map", lambda: {})
    web_app._endpoint_cache.clear()

    orders = web_app._compute_open_limit_orders()

    assert len(orders) == 1
    assert orders[0]["status"] == "filled"
    assert orders[0]["status_note"] == "MATCHED"
    assert orders[0]["size_remaining"] == 0.0
    assert orders[0]["size_matched"] == 50.0
    assert orders[0]["on_clob"] is False


def test_compute_open_limit_orders_marks_cancelled_order_as_cancelled(tmp_path, monkeypatch):
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
    monkeypatch.setattr(web_app, "_build_token_to_fighter_map", lambda: {})
    web_app._endpoint_cache.clear()

    orders = web_app._compute_open_limit_orders()

    assert len(orders) == 1
    assert orders[0]["status"] == "cancelled"
    assert orders[0]["status_note"] == "CANCELED"
    assert orders[0]["size_remaining"] == 0.0
    assert orders[0]["size_matched"] == 0.0
    assert orders[0]["on_clob"] is False


def test_compute_open_limit_orders_marks_live_partial_as_partially_filled(tmp_path, monkeypatch):
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
    monkeypatch.setattr(web_app, "_build_token_to_fighter_map", lambda: {})
    web_app._endpoint_cache.clear()

    orders = web_app._compute_open_limit_orders()

    assert len(orders) == 1
    assert orders[0]["status"] == "partially_filled"
    assert round(orders[0]["size_remaining"], 2) == 23.54
    assert round(orders[0]["size_matched"], 2) == 14.73
    assert orders[0]["on_clob"] is True
