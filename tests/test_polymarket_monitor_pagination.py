from datetime import datetime, timezone

import requests

import pytest

from src.polymarket import monitor as monitor_module
from src.polymarket import data_api as data_api_module
from src.polymarket.monitor import PositionDataPartialError, PositionMonitor


@pytest.mark.parametrize(
    ("retry_after", "expected_seconds"),
    [
        ("12.5", 12.5),
        ("Wed, 12 Aug 2026 12:00:15 GMT", 15.0),
    ],
)
def test_data_api_retry_after_accepts_seconds_and_http_date(
    retry_after,
    expected_seconds,
):
    class _FakeResponse:
        headers = {"Retry-After": retry_after}

    wait = data_api_module._retry_after_seconds(
        _FakeResponse(),
        now=datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc),
    )
    assert wait == pytest.approx(expected_seconds)


def test_data_api_429_uses_exponential_floor_and_bounded_jitter(monkeypatch):
    sleeps: list[float] = []

    class _FakeResponse:
        headers = {"Retry-After": "1"}

        def __init__(self, status_code: int):
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} error", response=self)

        def json(self):
            return [{"asset": "token-a", "size": 1}]

    responses = [_FakeResponse(429), _FakeResponse(429), _FakeResponse(200)]
    monkeypatch.setattr(
        data_api_module.requests,
        "get",
        lambda *args, **kwargs: responses.pop(0),
    )
    monkeypatch.setattr(data_api_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(data_api_module.random, "uniform", lambda _low, _high: 0.25)

    rows = data_api_module.request_json("https://data-api.polymarket.com/positions")

    assert rows == [{"asset": "token-a", "size": 1}]
    assert sleeps == pytest.approx([10.25, 20.25])


@pytest.mark.parametrize("status_code", [408, 530])
def test_data_api_request_json_retries_transient_status(monkeypatch, status_code):
    sleeps: list[float] = []

    class _FakeResponse:
        headers: dict[str, str] = {}

        def __init__(self, status_code: int, payload: list[dict] | None = None):
            self.status_code = status_code
            self._payload = payload or []

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} error", response=self)

        def json(self):
            return self._payload

    responses = [
        _FakeResponse(status_code),
        _FakeResponse(200, [{"asset": "token-a", "size": 1}]),
    ]

    def _fake_get(url, params=None, timeout=30):
        assert url == f"{monitor_module.DATA_API_URL}/positions"
        assert params == {"user": "0xwallet"}
        return responses.pop(0)

    monkeypatch.setattr(data_api_module.requests, "get", _fake_get)
    monkeypatch.setattr(data_api_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    rows = data_api_module.request_json(
        f"{monitor_module.DATA_API_URL}/positions",
        params={"user": "0xwallet"},
    )

    assert rows == [{"asset": "token-a", "size": 1}]
    assert sleeps == [0.5]


def test_get_positions_strict_raises_on_later_page_failure(monkeypatch):
    def _fake_request_json(url, *, params=None, timeout=30):
        assert url == f"{monitor_module.DATA_API_URL}/positions"
        if params["offset"] == 0:
            return [
                {"asset": "token-a", "size": 1, "avgPrice": 0.5, "curPrice": 0.6},
                {"asset": "token-b", "size": 2, "avgPrice": 0.4, "curPrice": 0.5},
            ]
        raise requests.Timeout("page 2 timed out")

    monkeypatch.setattr(monitor_module, "request_data_api_json", _fake_request_json)

    monitor = PositionMonitor(wallet_address="0xwallet")

    with pytest.raises(PositionDataPartialError, match="positions page"):
        monitor.get_positions(page_size=2, strict=True)


@pytest.mark.parametrize(
    "payload",
    [None, {}, [{"asset": "token-a", "size": 1}, None]],
)
def test_get_positions_strict_rejects_malformed_success_payload(monkeypatch, payload):
    monkeypatch.setattr(
        monitor_module,
        "request_data_api_json",
        lambda *args, **kwargs: payload,
    )
    monitor = PositionMonitor(wallet_address="0xwallet")

    with pytest.raises(PositionDataPartialError, match="positions page at offset=0"):
        monitor.get_positions(strict=True)


@pytest.mark.parametrize(
    "row",
    [
        {"size": 1},
        {"asset": "", "size": 1},
        {"asset": 123, "size": 1},
        {"asset": "token-a", "size": None},
        {"asset": "token-a", "size": "not-a-number"},
        {"asset": "token-a", "size": float("nan")},
        {"asset": "token-a", "size": float("inf")},
        {"asset": "token-a", "size": -0.01},
        {"asset": "token-a", "size": True},
    ],
)
def test_get_positions_strict_rejects_semantically_invalid_rows(monkeypatch, row):
    monkeypatch.setattr(
        monitor_module,
        "request_data_api_json",
        lambda *args, **kwargs: [row],
    )
    monitor = PositionMonitor(wallet_address="0xwallet")

    with pytest.raises(PositionDataPartialError, match="positions page at offset=0"):
        monitor.get_positions(strict=True)


def test_get_positions_allows_zero_size_rows_without_assets_and_normalizes_active_ids(
    monkeypatch,
):
    monkeypatch.setattr(
        monitor_module,
        "request_data_api_json",
        lambda *args, **kwargs: [
            {"size": 0},
            {"asset": None, "size": "0"},
            {"asset": " token-a ", "size": "1.25"},
            {"token_id": " token-b ", "size": 2},
        ],
    )
    monitor = PositionMonitor(wallet_address="0xwallet")

    rows = monitor.get_positions(strict=True)

    assert rows == [
        {"asset": "token-a", "size": "1.25"},
        {"token_id": "token-b", "size": 2},
    ]


def test_get_closed_positions_strict_raises_on_later_page_failure(monkeypatch):
    def _fake_request_json(url, *, params=None, timeout=30):
        assert url == f"{monitor_module.DATA_API_URL}/closed-positions"
        if params["offset"] == 0:
            return [
                {"asset": "token-a", "realizedPnl": 1.25},
                {"asset": "token-b", "realizedPnl": -0.5},
            ]
        raise requests.Timeout("page 2 timed out")

    monkeypatch.setattr(monitor_module, "request_data_api_json", _fake_request_json)

    monitor = PositionMonitor(wallet_address="0xwallet")

    with pytest.raises(PositionDataPartialError, match="closed positions page"):
        monitor.get_closed_positions(page_size=2, strict=True)


def test_get_closed_positions_clamps_requests_to_data_api_maximum(monkeypatch):
    seen_params = []

    def _fake_request_json(url, *, params=None, timeout=30):
        assert url == f"{monitor_module.DATA_API_URL}/closed-positions"
        seen_params.append(dict(params))
        if params["offset"] == 0:
            return [
                {"asset": f"token-{index}", "realizedPnl": 0.0}
                for index in range(50)
            ]
        if params["offset"] == 50:
            return [
                {"asset": f"token-{index}", "realizedPnl": 0.0}
                for index in range(50, 75)
            ]
        raise AssertionError(f"unexpected params {params}")

    monkeypatch.setattr(monitor_module, "request_data_api_json", _fake_request_json)

    monitor = PositionMonitor(wallet_address="0xwallet")
    rows = monitor.get_closed_positions(limit=75, page_size=500, strict=True)

    assert len(rows) == 75
    assert seen_params == [
        {"user": "0xwallet", "limit": 50, "offset": 0},
        {"user": "0xwallet", "limit": 25, "offset": 50},
    ]


def test_get_trades_collects_multiple_activity_pages(monkeypatch):
    def _fake_request_json(url, *, params=None, timeout=30):
        assert url == f"{monitor_module.DATA_API_URL}/activity"
        assert params["user"] == "0xwallet"
        if params["offset"] == 0:
            return [
                {"timestamp": 3, "side": "BUY"},
                {"timestamp": 2, "side": "SELL"},
            ]
        if params["offset"] == 2:
            return [
                {"timestamp": 1, "type": "REDEEM"},
            ]
        raise AssertionError(f"unexpected offset {params['offset']}")

    monkeypatch.setattr(monitor_module, "request_data_api_json", _fake_request_json)

    monitor = PositionMonitor(wallet_address="0xwallet")
    rows = monitor.get_trades(limit=None, page_size=2, strict=True)

    assert [row["timestamp"] for row in rows] == [3, 2, 1]


def test_get_trades_collects_activity_beyond_legacy_thousand_row_cap(monkeypatch):
    seen_offsets = []

    def _fake_request_json(url, *, params=None, timeout=30):
        assert url == f"{monitor_module.DATA_API_URL}/activity"
        seen_offsets.append(params["offset"])
        if params["offset"] in {0, 500, 1000}:
            return [{"timestamp": params["offset"] + index} for index in range(500)]
        if params["offset"] == 1500:
            return [{"timestamp": 1500}]
        raise AssertionError(f"unexpected offset {params['offset']}")

    monkeypatch.setattr(monitor_module, "request_data_api_json", _fake_request_json)

    monitor = PositionMonitor(wallet_address="0xwallet")
    rows = monitor.get_trades(limit=None, page_size=500, strict=True)

    assert seen_offsets == [0, 500, 1000, 1500]
    assert len(rows) == 1501


def test_get_trades_rolls_activity_window_after_offset_cap(monkeypatch):
    monkeypatch.setattr(monitor_module, "ACTIVITY_MAX_OFFSET", 500)
    seen = []

    def _rows(start, stop):
        return [{"timestamp": ts, "transactionHash": f"0x{ts:x}"} for ts in range(start, stop, -1)]

    def _fake_request_json(url, *, params=None, timeout=30):
        assert url == f"{monitor_module.DATA_API_URL}/activity"
        seen.append({"offset": params["offset"], "end": params.get("end")})
        if params["offset"] == 0 and "end" not in params:
            return _rows(2000, 1500)
        if params["offset"] == 500 and "end" not in params:
            return _rows(1500, 1000)
        if params["offset"] == 0 and params.get("end") == 1000:
            return _rows(1000, 998)
        raise AssertionError(f"unexpected params {params}")

    monkeypatch.setattr(monitor_module, "request_data_api_json", _fake_request_json)

    monitor = PositionMonitor(wallet_address="0xwallet")
    rows = monitor.get_trades(limit=None, page_size=500, strict=True)

    assert seen == [
        {"offset": 0, "end": None},
        {"offset": 500, "end": None},
        {"offset": 0, "end": 1000},
    ]
    assert len(rows) == 1002
    assert rows[-1]["timestamp"] == 999


def test_get_trades_strict_raises_on_later_page_failure(monkeypatch):
    def _fake_request_json(url, *, params=None, timeout=30):
        assert url == f"{monitor_module.DATA_API_URL}/activity"
        if params["offset"] == 0:
            return [
                {"timestamp": 3, "side": "BUY"},
                {"timestamp": 2, "side": "SELL"},
            ]
        raise requests.Timeout("page 2 timed out")

    monkeypatch.setattr(monitor_module, "request_data_api_json", _fake_request_json)

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
