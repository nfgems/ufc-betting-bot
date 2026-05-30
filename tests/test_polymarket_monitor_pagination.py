import requests

import pytest

from src.polymarket import monitor as monitor_module
from src.polymarket.monitor import PositionDataPartialError, PositionMonitor


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def test_get_positions_strict_raises_on_later_page_failure(monkeypatch):
    def _fake_get(url, params=None, timeout=30):
        assert url == f"{monitor_module.DATA_API_URL}/positions"
        if params["offset"] == 0:
            return _FakeResponse(
                [
                    {"asset": "token-a", "size": 1, "avgPrice": 0.5, "curPrice": 0.6},
                    {"asset": "token-b", "size": 2, "avgPrice": 0.4, "curPrice": 0.5},
                ]
            )
        raise requests.Timeout("page 2 timed out")

    monkeypatch.setattr(monitor_module.requests, "get", _fake_get)

    monitor = PositionMonitor(wallet_address="0xwallet")

    with pytest.raises(PositionDataPartialError, match="positions page"):
        monitor.get_positions(page_size=2, strict=True)


def test_get_closed_positions_strict_raises_on_later_page_failure(monkeypatch):
    def _fake_get(url, params=None, timeout=30):
        assert url == f"{monitor_module.DATA_API_URL}/closed-positions"
        if params["offset"] == 0:
            return _FakeResponse(
                [
                    {"asset": "token-a", "realizedPnl": 1.25},
                    {"asset": "token-b", "realizedPnl": -0.5},
                ]
            )
        raise requests.Timeout("page 2 timed out")

    monkeypatch.setattr(monitor_module.requests, "get", _fake_get)

    monitor = PositionMonitor(wallet_address="0xwallet")

    with pytest.raises(PositionDataPartialError, match="closed positions page"):
        monitor.get_closed_positions(page_size=2, strict=True)


def test_get_trades_collects_multiple_activity_pages(monkeypatch):
    def _fake_get(url, params=None, timeout=30):
        assert url == f"{monitor_module.DATA_API_URL}/activity"
        assert params["user"] == "0xwallet"
        if params["offset"] == 0:
            return _FakeResponse(
                [
                    {"timestamp": 3, "side": "BUY"},
                    {"timestamp": 2, "side": "SELL"},
                ]
            )
        if params["offset"] == 2:
            return _FakeResponse(
                [
                    {"timestamp": 1, "type": "REDEEM"},
                ]
            )
        raise AssertionError(f"unexpected offset {params['offset']}")

    monkeypatch.setattr(monitor_module.requests, "get", _fake_get)

    monitor = PositionMonitor(wallet_address="0xwallet")
    rows = monitor.get_trades(limit=None, page_size=2, strict=True)

    assert [row["timestamp"] for row in rows] == [3, 2, 1]


def test_get_trades_collects_activity_beyond_legacy_thousand_row_cap(monkeypatch):
    seen_offsets = []

    def _fake_get(url, params=None, timeout=30):
        assert url == f"{monitor_module.DATA_API_URL}/activity"
        seen_offsets.append(params["offset"])
        if params["offset"] in {0, 500, 1000}:
            return _FakeResponse(
                [{"timestamp": params["offset"] + index} for index in range(500)]
            )
        if params["offset"] == 1500:
            return _FakeResponse([{"timestamp": 1500}])
        raise AssertionError(f"unexpected offset {params['offset']}")

    monkeypatch.setattr(monitor_module.requests, "get", _fake_get)

    monitor = PositionMonitor(wallet_address="0xwallet")
    rows = monitor.get_trades(limit=None, page_size=500, strict=True)

    assert seen_offsets == [0, 500, 1000, 1500]
    assert len(rows) == 1501


def test_get_trades_strict_raises_on_later_page_failure(monkeypatch):
    def _fake_get(url, params=None, timeout=30):
        assert url == f"{monitor_module.DATA_API_URL}/activity"
        if params["offset"] == 0:
            return _FakeResponse(
                [
                    {"timestamp": 3, "side": "BUY"},
                    {"timestamp": 2, "side": "SELL"},
                ]
            )
        raise requests.Timeout("page 2 timed out")

    monkeypatch.setattr(monitor_module.requests, "get", _fake_get)

    monitor = PositionMonitor(wallet_address="0xwallet")

    with pytest.raises(PositionDataPartialError, match="activity page"):
        monitor.get_trades(limit=None, page_size=2, strict=True)


def test_compute_pnl_requests_strict_full_pagination():
    seen = {}

    class _RecordingMonitor(PositionMonitor):
        def __init__(self):
            super().__init__(wallet_address="0xwallet")

        def get_positions(self, *args, **kwargs):
            seen["positions"] = dict(kwargs)
            return []

        def get_closed_positions(self, *args, **kwargs):
            seen["closed"] = dict(kwargs)
            return []

    monitor = _RecordingMonitor()

    payload = monitor.compute_pnl()

    assert payload["total_pnl"] == 0.0
    assert seen["positions"]["strict"] is True
    assert seen["closed"]["strict"] is True
