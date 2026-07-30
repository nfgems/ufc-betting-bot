from datetime import datetime, timedelta, timezone

import logging
import threading
import time
import pytest
import pandas as pd
import httpx
from py_clob_client_v2.exceptions import PolyApiException

from src.polymarket.client import ClobClientWrapper, LEGACY_POLYGON_USDC_E
import src.polymarket.client as client_mod
from src.polymarket.executor import OrderExecutor
from src.polymarket.tracker import BetLedger
from src.strategy.bankroll import BankrollManager
from src.web.alert_store import DurableAlertHandler, load_alert_incidents


class _FakeResponse:
    def __init__(self, result_hex: str):
        self._result_hex = result_hex

    def json(self):
        return {"result": self._result_hex}


def _base_bet(**overrides):
    data = {
        "fighter_a": "Alpha",
        "fighter_b": "Beta",
        "bet_on": "Alpha",
        "model_prob": 0.70,
        "blended_prob": 0.70,
        "market_prob": 0.60,
        "edge": 0.10,
        "decimal_odds": 1.6667,
        "bet_side": "a",
        "token_id_yes": "token-yes",
        "token_id_no": "token-no",
        "market_id": "market-1",
        "condition_id": "condition-1",
        "tick_size": "0.01",
        "neg_risk": False,
        "override_bet_size": 25.0,
        "event_date": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
    }
    data.update(overrides)
    return pd.Series(data)


def test_v2_wrapper_compatibility_adapters():
    class _RawClient:
        def __init__(self):
            self.cancel_payload = None
            self.trade_params = None

        def cancel_order(self, payload):
            self.cancel_payload = payload
            return {"cancelled": payload.orderID}

        def cancel_all(self):
            return {"cancelled": "all"}

        def get_open_orders(self):
            return [{"id": "open-1"}]

        def get_trades(self, params=None):
            self.trade_params = params
            return [{"id": "trade-1"}]

    raw = _RawClient()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw

    assert wrapper.cancel_order("order-1") == {"cancelled": "order-1"}
    assert raw.cancel_payload.orderID == "order-1"
    assert wrapper.cancel_all_orders() == {"cancelled": "all"}
    assert wrapper.get_open_orders() == [{"id": "open-1"}]
    assert wrapper.get_trades(params={"asset_id": "token-1"}) == [{"id": "trade-1"}]
    assert raw.trade_params == {"asset_id": "token-1"}


def test_v2_wrapper_retries_transient_cancel_order_425(monkeypatch):
    class _RawClient:
        def __init__(self):
            self.calls = 0

        def cancel_order(self, payload):
            self.calls += 1
            if self.calls == 1:
                response = httpx.Response(
                    425,
                    text="service not ready",
                    request=httpx.Request("POST", "https://clob.polymarket.com/order"),
                )
                raise PolyApiException(resp=response)
            return {"cancelled": payload.orderID}

    raw = _RawClient()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(client_mod.time, "sleep", lambda _seconds: None)

    assert wrapper.cancel_order("order-1") == {"cancelled": "order-1"}
    assert raw.calls == 2


def test_v2_wrapper_does_not_retry_non_retryable_cancel_order_error(monkeypatch):
    class _RawClient:
        def __init__(self):
            self.calls = 0

        def cancel_order(self, _payload):
            self.calls += 1
            response = httpx.Response(
                400,
                text="bad request",
                request=httpx.Request("POST", "https://clob.polymarket.com/order"),
            )
            raise PolyApiException(resp=response)

    raw = _RawClient()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(client_mod.time, "sleep", lambda _seconds: None)

    with pytest.raises(PolyApiException):
        wrapper.cancel_order("order-1")
    assert raw.calls == 1


def test_v2_wrapper_retries_transient_get_open_orders(monkeypatch):
    class _RawClient:
        def __init__(self):
            self.calls = 0

        def get_open_orders(self):
            self.calls += 1
            if self.calls == 1:
                raise PolyApiException(error_msg="Request exception!")
            return [{"id": "open-1"}]

    raw = _RawClient()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(client_mod.time, "sleep", lambda _seconds: None)

    assert wrapper.get_open_orders() == [{"id": "open-1"}]
    assert raw.calls == 2


def test_v2_wrapper_retries_transient_get_balance_allowance(monkeypatch, caplog):
    helper_logger = logging.getLogger(client_mod._CLOB_HELPER_LOGGER_NAME)

    class _RawClient:
        def __init__(self):
            self.calls = 0

        def get_balance_allowance(self, _params):
            self.calls += 1
            if self.calls == 1:
                helper_logger.error(
                    "[py_clob_client_v2] request error: read timed out"
                )
                raise PolyApiException(error_msg="Request exception!")
            return {"balance": "12500000"}

    raw = _RawClient()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(client_mod, "POLYMARKET_BALANCE_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(client_mod.time, "sleep", lambda _seconds: None)

    with caplog.at_level(logging.INFO):
        details = wrapper.get_cash_balance_details(allow_onchain_fallback=False)

    assert details == {"balance": pytest.approx(12.5), "source": "clob"}
    assert raw.calls == 2
    helper_records = [
        record
        for record in caplog.records
        if record.name == client_mod._CLOB_HELPER_LOGGER_NAME
    ]
    assert len(helper_records) == 1
    assert helper_records[0].levelno == logging.INFO
    assert not any(
        record.name == "src.polymarket.client"
        and record.levelno >= logging.WARNING
        for record in caplog.records
    )


def test_v2_wrapper_bounds_transient_balance_retries(monkeypatch, caplog):
    helper_logger = logging.getLogger(client_mod._CLOB_HELPER_LOGGER_NAME)

    class _RawClient:
        def __init__(self):
            self.calls = 0

        def get_balance_allowance(self, _params):
            self.calls += 1
            helper_logger.error(
                "[py_clob_client_v2] request error: read timed out"
            )
            raise PolyApiException(error_msg="Request exception!")

    raw = _RawClient()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(client_mod, "POLYMARKET_BALANCE_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(client_mod.time, "sleep", lambda _seconds: None)

    with caplog.at_level(logging.INFO):
        details = wrapper.get_cash_balance_details(allow_onchain_fallback=False)

    assert details == {"balance": 0.0, "source": "unavailable"}
    assert raw.calls == client_mod.POLYMARKET_BALANCE_MAX_ATTEMPTS
    helper_records = [
        record
        for record in caplog.records
        if record.name == client_mod._CLOB_HELPER_LOGGER_NAME
    ]
    assert len(helper_records) == client_mod.POLYMARKET_BALANCE_MAX_ATTEMPTS
    assert {record.levelno for record in helper_records} == {logging.INFO}
    client_warnings = [
        record
        for record in caplog.records
        if record.name == "src.polymarket.client"
        and record.levelno == logging.WARNING
        and "Could not fetch CLOB balance" in record.getMessage()
    ]
    assert len(client_warnings) == 1


def test_v2_wrapper_does_not_retry_non_transient_balance_error(monkeypatch):
    class _RawClient:
        def __init__(self):
            self.calls = 0

        def get_balance_allowance(self, _params):
            self.calls += 1
            response = httpx.Response(
                401,
                text="unauthorized",
                request=httpx.Request(
                    "GET",
                    "https://clob.polymarket.com/balance-allowance",
                ),
            )
            raise PolyApiException(resp=response)

    raw = _RawClient()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(client_mod.time, "sleep", lambda _seconds: None)

    details = wrapper.get_cash_balance_details(allow_onchain_fallback=False)

    assert details == {"balance": 0.0, "source": "unavailable"}
    assert raw.calls == 1


@pytest.mark.parametrize("status_code", [408, 520, 521, 522, 523, 524, 530])
def test_v2_wrapper_retries_safe_transient_balance_statuses(
    monkeypatch,
    status_code,
):
    class _RawClient:
        def __init__(self):
            self.calls = 0

        def get_balance_allowance(self, _params):
            self.calls += 1
            if self.calls == 1:
                response = httpx.Response(
                    status_code,
                    text="transient read failure",
                    request=httpx.Request(
                        "GET",
                        "https://clob.polymarket.com/balance-allowance",
                    ),
                )
                raise PolyApiException(resp=response)
            return {"balance": "12500000"}

    raw = _RawClient()
    shared_client = httpx.Client()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(wrapper, "_configure_shared_transport", lambda: shared_client)
    monkeypatch.setattr(client_mod.time, "sleep", lambda _seconds: None)

    try:
        payload = wrapper.get_balance_allowance(
            max_attempts=2,
            total_budget_seconds=25.0,
        )
        assert payload == {"balance": "12500000"}
        assert raw.calls == 2
    finally:
        shared_client.close()


def test_v2_wrapper_balance_uses_bounded_attempts_and_restores_timeout(
    monkeypatch,
):
    attempt_timeouts = []

    class _RawClient:
        def __init__(self):
            self.calls = 0

        def get_balance_allowance(self, _params):
            self.calls += 1
            attempt_timeouts.append(shared_client.timeout)
            raise PolyApiException(error_msg="Request exception!")

    raw = _RawClient()
    shared_client = httpx.Client(timeout=4.0)
    original_timeout = shared_client.timeout
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(wrapper, "_configure_shared_transport", lambda: shared_client)
    monkeypatch.setattr(client_mod.time, "sleep", lambda _seconds: None)

    try:
        with pytest.raises(client_mod.ClobBalanceUnavailableError):
            wrapper.get_balance_allowance(
                max_attempts=2,
                read_timeout_seconds=10.0,
                total_budget_seconds=25.0,
            )
        assert raw.calls == 2
        assert [timeout.read for timeout in attempt_timeouts] == [10.0, 10.0]
        assert [timeout.connect for timeout in attempt_timeouts] == [3.0, 3.0]
        assert [timeout.write for timeout in attempt_timeouts] == [5.0, 5.0]
        assert [timeout.pool for timeout in attempt_timeouts] == [2.0, 2.0]
        assert shared_client.timeout.connect == original_timeout.connect
        assert shared_client.timeout.read == original_timeout.read
        assert shared_client.timeout.write == original_timeout.write
        assert shared_client.timeout.pool == original_timeout.pool
    finally:
        shared_client.close()


def test_v2_wrapper_quarantines_transport_when_timeout_restore_fails(
    monkeypatch,
):
    import py_clob_client_v2.http_helpers.helpers as clob_helpers

    class _RestoreRejectingTransport:
        def __init__(self, timeout):
            self._timeout = timeout
            self.assignments = 0
            self.closed = False

        @property
        def timeout(self):
            return self._timeout

        @timeout.setter
        def timeout(self, value):
            self.assignments += 1
            if self.assignments == 2:
                raise RuntimeError("restore setter failed")
            self._timeout = value

        def close(self):
            self.closed = True

    class _RawClient:
        def __init__(self):
            self.calls = 0

        def get_balance_allowance(self, _params):
            self.calls += 1
            return {"balance": "12500000"}

    original_timeout = httpx.Timeout(4.0)
    poisoned_transport = _RestoreRejectingTransport(original_timeout)
    replacement_transport = httpx.Client(timeout=7.0)
    raw = _RawClient()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(clob_helpers, "_http_client", poisoned_transport)
    monkeypatch.setattr(client_mod, "_proxy_patched", True)
    monkeypatch.setattr(
        client_mod,
        "_new_clob_http_client",
        lambda: replacement_transport,
    )

    try:
        with pytest.raises(client_mod.ClobBalanceUnavailableError):
            wrapper.get_balance_allowance(
                max_attempts=2,
                total_budget_seconds=25.0,
            )

        assert raw.calls == 1
        assert poisoned_transport.assignments == 2
        assert poisoned_transport.closed is True
        assert poisoned_transport.timeout is not original_timeout
        assert clob_helpers._http_client is replacement_transport

        assert wrapper.get_balance_allowance(max_attempts=1) == {
            "balance": "12500000"
        }
        assert raw.calls == 2
        assert clob_helpers._http_client is replacement_transport
    finally:
        replacement_transport.close()


def test_v2_wrapper_balance_budget_prevents_late_retry(monkeypatch):
    clock = {"now": 0.0}

    class _RawClient:
        def __init__(self):
            self.calls = 0

        def get_balance_allowance(self, _params):
            self.calls += 1
            clock["now"] = 25.0
            raise PolyApiException(error_msg="Request exception!")

    raw = _RawClient()
    shared_client = httpx.Client()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(wrapper, "_configure_shared_transport", lambda: shared_client)
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(client_mod.time, "sleep", lambda _seconds: None)

    try:
        with pytest.raises(client_mod.ClobBalanceUnavailableError):
            wrapper.get_balance_allowance(
                max_attempts=3,
                total_budget_seconds=25.0,
            )
        assert raw.calls == 1
    finally:
        shared_client.close()


def test_v2_wrapper_balance_budget_starts_after_cold_initialization(
    monkeypatch,
):
    import py_clob_client_v2.clob_types as clob_types

    clock = {"now": 0.0}

    class _RawClient:
        def __init__(self):
            self.calls = 0

        def get_balance_allowance(self, _params):
            self.calls += 1
            return {"balance": "12500000"}

    raw = _RawClient()
    shared_client = httpx.Client()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    real_params_type = clob_types.BalanceAllowanceParams

    def _cold_start():
        clock["now"] += 30.0
        wrapper._client = raw

    def _construct_params(*args, **kwargs):
        clock["now"] += 20.0
        return real_params_type(*args, **kwargs)

    monkeypatch.setattr(wrapper, "_ensure_client", _cold_start)
    monkeypatch.setattr(wrapper, "_configure_shared_transport", lambda: shared_client)
    monkeypatch.setattr(
        clob_types,
        "BalanceAllowanceParams",
        _construct_params,
    )
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: clock["now"])

    try:
        assert wrapper.get_balance_allowance(
            max_attempts=1,
            total_budget_seconds=0.5,
        ) == {"balance": "12500000"}
        assert raw.calls == 1
        assert clock["now"] == 50.0
    finally:
        shared_client.close()


def test_v2_wrapper_balance_budget_bounds_transport_lock_wait(monkeypatch):
    class _RawClient:
        calls = 0

        def get_balance_allowance(self, _params):
            self.calls += 1
            return {"balance": "12500000"}

    raw = _RawClient()
    shared_client = httpx.Client()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    configure_calls = []
    monkeypatch.setattr(
        wrapper,
        "_configure_shared_transport",
        lambda: configure_calls.append(1) or shared_client,
    )

    acquired = threading.Event()
    release = threading.Event()

    def _hold_transport_lock():
        with client_mod._clob_transport_lock:
            acquired.set()
            release.wait(1.0)

    holder = threading.Thread(target=_hold_transport_lock, daemon=True)
    holder.start()
    assert acquired.wait(1.0)

    started_at = time.monotonic()
    try:
        with pytest.raises(client_mod.ClobBalanceUnavailableError):
            wrapper.get_balance_allowance(
                max_attempts=1,
                total_budget_seconds=0.05,
            )
        assert time.monotonic() - started_at < 0.5
        assert raw.calls == 0
        assert configure_calls == []
    finally:
        release.set()
        holder.join(timeout=1.0)
        shared_client.close()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"balance": None},
        {"balance": ""},
        {"balance": "-1"},
        {"balance": float("nan")},
        {"balance": float("inf")},
        {"balance": float("-inf")},
        {"balance": "100", "decimals": float("inf")},
    ],
)
def test_v2_wrapper_rejects_invalid_clob_balance_payloads(
    monkeypatch,
    caplog,
    payload,
):
    class _RawClient:
        def __init__(self):
            self.calls = 0

        def get_balance_allowance(self, _params):
            self.calls += 1
            return payload

    raw = _RawClient()
    shared_client = httpx.Client()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(wrapper, "_configure_shared_transport", lambda: shared_client)

    try:
        with caplog.at_level(logging.INFO):
            details = wrapper.get_cash_balance_details(
                allow_onchain_fallback=False
            )
        assert details == {"balance": 0.0, "source": "unavailable"}
        assert raw.calls == 1
        terminal_warnings = [
            record
            for record in caplog.records
            if record.name == "src.polymarket.client"
            and record.levelno == logging.WARNING
            and "Could not fetch CLOB balance" in record.getMessage()
        ]
        assert len(terminal_warnings) == 1
    finally:
        shared_client.close()


def test_v2_wrapper_accepts_confirmed_zero_clob_balance(monkeypatch):
    class _RawClient:
        def get_balance_allowance(self, _params):
            return {"balance": 0}

    shared_client = httpx.Client()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = _RawClient()
    monkeypatch.setattr(wrapper, "_configure_shared_transport", lambda: shared_client)

    try:
        assert wrapper.get_cash_balance_details(
            allow_onchain_fallback=False
        ) == {"balance": 0.0, "source": "clob"}
    finally:
        shared_client.close()


def test_v2_wrapper_rejects_balance_success_after_budget(monkeypatch):
    clock = {"now": 0.0}

    class _RawClient:
        calls = 0

        def get_balance_allowance(self, _params):
            self.calls += 1
            clock["now"] = 25.1
            return {"balance": "12500000"}

    raw = _RawClient()
    shared_client = httpx.Client()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(wrapper, "_configure_shared_transport", lambda: shared_client)
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: clock["now"])

    try:
        with pytest.raises(client_mod.ClobBalanceUnavailableError):
            wrapper.get_balance_allowance(
                max_attempts=1,
                total_budget_seconds=25.0,
            )
        assert raw.calls == 1
    finally:
        shared_client.close()


def test_v2_wrapper_open_orders_backoff_does_not_hold_transport_lock(
    monkeypatch,
):
    class _RawClient:
        def __init__(self):
            self.calls = 0

        def get_open_orders(self):
            self.calls += 1
            if self.calls == 1:
                raise PolyApiException(error_msg="Request exception!")
            return []

    raw = _RawClient()
    shared_client = httpx.Client()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(wrapper, "_configure_shared_transport", lambda: shared_client)
    lock_was_available = []

    def _assert_lock_available_during_backoff(_seconds):
        acquired = threading.Event()

        def _probe_lock():
            with client_mod._clob_transport_lock:
                acquired.set()

        probe = threading.Thread(target=_probe_lock, daemon=True)
        probe.start()
        lock_was_available.append(acquired.wait(0.5))
        probe.join(timeout=0.5)

    monkeypatch.setattr(client_mod.time, "sleep", _assert_lock_available_during_backoff)

    try:
        assert wrapper.get_open_orders(max_attempts=2) == []
        assert lock_was_available == [True]
    finally:
        shared_client.close()


def test_v2_wrapper_open_orders_uses_bounded_attempts_and_restores_timeout(
    monkeypatch,
):
    attempt_timeouts = []

    class _RawClient:
        def __init__(self):
            self.calls = 0

        def get_open_orders(self):
            self.calls += 1
            attempt_timeouts.append(shared_client.timeout)
            raise PolyApiException(error_msg="Request exception!")

    raw = _RawClient()
    shared_client = httpx.Client(timeout=4.0)
    original_timeout = shared_client.timeout
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(wrapper, "_configure_shared_transport", lambda: shared_client)
    monkeypatch.setattr(client_mod.time, "sleep", lambda _seconds: None)

    try:
        with pytest.raises(client_mod.ClobOpenOrdersUnavailableError):
            wrapper.get_open_orders(
                max_attempts=2,
                read_timeout_seconds=10.0,
                total_budget_seconds=25.0,
            )
        assert raw.calls == 2
        assert [timeout.read for timeout in attempt_timeouts] == [10.0, 10.0]
        assert [timeout.connect for timeout in attempt_timeouts] == [3.0, 3.0]
        assert [timeout.write for timeout in attempt_timeouts] == [5.0, 5.0]
        assert [timeout.pool for timeout in attempt_timeouts] == [2.0, 2.0]
        assert shared_client.timeout.connect == original_timeout.connect
        assert shared_client.timeout.read == original_timeout.read
        assert shared_client.timeout.write == original_timeout.write
        assert shared_client.timeout.pool == original_timeout.pool
    finally:
        shared_client.close()


def test_v2_wrapper_open_orders_does_not_retry_non_transient_error(monkeypatch):
    class _RawClient:
        def __init__(self):
            self.calls = 0

        def get_open_orders(self):
            self.calls += 1
            response = httpx.Response(
                400,
                text="bad request",
                request=httpx.Request(
                    "GET",
                    "https://clob.polymarket.com/data/orders",
                ),
            )
            raise PolyApiException(resp=response)

    raw = _RawClient()
    shared_client = httpx.Client()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(wrapper, "_configure_shared_transport", lambda: shared_client)
    monkeypatch.setattr(client_mod.time, "sleep", lambda _seconds: None)

    try:
        with pytest.raises(client_mod.ClobOpenOrdersUnavailableError):
            wrapper.get_open_orders(max_attempts=2)
        assert raw.calls == 1
    finally:
        shared_client.close()


def test_v2_wrapper_open_orders_single_attempt_does_not_retry_transient_error(
    monkeypatch,
):
    class _RawClient:
        def __init__(self):
            self.calls = 0

        def get_open_orders(self):
            self.calls += 1
            raise PolyApiException(error_msg="Request exception!")

    raw = _RawClient()
    shared_client = httpx.Client()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(wrapper, "_configure_shared_transport", lambda: shared_client)
    monkeypatch.setattr(client_mod.time, "sleep", lambda _seconds: None)

    try:
        with pytest.raises(client_mod.ClobOpenOrdersUnavailableError):
            wrapper.get_open_orders(max_attempts=1)
        assert raw.calls == 1
    finally:
        shared_client.close()


def test_v2_wrapper_open_orders_budget_prevents_late_retry(monkeypatch):
    clock = {"now": 0.0}

    class _RawClient:
        def __init__(self):
            self.calls = 0

        def get_open_orders(self):
            self.calls += 1
            clock["now"] = 25.0
            raise PolyApiException(error_msg="Request exception!")

    raw = _RawClient()
    shared_client = httpx.Client()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(wrapper, "_configure_shared_transport", lambda: shared_client)
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(client_mod.time, "sleep", lambda _seconds: None)

    try:
        with pytest.raises(client_mod.ClobOpenOrdersUnavailableError):
            wrapper.get_open_orders(
                max_attempts=3,
                total_budget_seconds=25.0,
            )
        assert raw.calls == 1
    finally:
        shared_client.close()


def test_v2_wrapper_open_orders_budget_bounds_transport_lock_wait(monkeypatch):
    class _RawClient:
        calls = 0

        def get_open_orders(self):
            self.calls += 1
            return []

    raw = _RawClient()
    shared_client = httpx.Client()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(wrapper, "_configure_shared_transport", lambda: shared_client)

    acquired = threading.Event()
    release = threading.Event()

    def _hold_transport_lock():
        with client_mod._clob_transport_lock:
            acquired.set()
            release.wait(1.0)

    holder = threading.Thread(target=_hold_transport_lock, daemon=True)
    holder.start()
    assert acquired.wait(1.0)

    started_at = time.monotonic()
    try:
        with pytest.raises(client_mod.ClobOpenOrdersUnavailableError):
            wrapper.get_open_orders(
                max_attempts=1,
                total_budget_seconds=0.05,
            )
        assert time.monotonic() - started_at < 0.5
        assert raw.calls == 0
    finally:
        release.set()
        holder.join(timeout=1.0)
        shared_client.close()


def test_v2_wrapper_rejects_success_that_finishes_after_budget(monkeypatch):
    clock = {"now": 0.0}

    class _RawClient:
        calls = 0

        def get_open_orders(self):
            self.calls += 1
            clock["now"] = 25.1
            return []

    raw = _RawClient()
    shared_client = httpx.Client()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(wrapper, "_configure_shared_transport", lambda: shared_client)
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: clock["now"])

    try:
        with pytest.raises(client_mod.ClobOpenOrdersUnavailableError):
            wrapper.get_open_orders(
                max_attempts=1,
                total_budget_seconds=25.0,
            )
        assert raw.calls == 1
    finally:
        shared_client.close()


def test_direct_balance_read_failure_then_success_recovers_incident(
    tmp_path,
    monkeypatch,
):
    client_logger = logging.getLogger("src.polymarket.client")
    previous_client_level = client_logger.level
    client_logger.setLevel(logging.INFO)
    path = tmp_path / "alerts.jsonl"
    durable_handler = DurableAlertHandler(path)
    client_logger.addHandler(durable_handler)

    class _RawClient:
        def __init__(self):
            self.fail = True

        def get_balance_allowance(self, _params):
            if self.fail:
                raise PolyApiException(error_msg="Request exception!")
            return {"balance": "12500000"}

    raw = _RawClient()
    shared_client = httpx.Client()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(wrapper, "_configure_shared_transport", lambda: shared_client)
    monkeypatch.setattr(client_mod, "_balance_recovery_probe_emitted", False)
    monkeypatch.setattr(client_mod, "_balance_read_sequence", 0)
    monkeypatch.setattr(client_mod, "_balance_incident_state_sequence", 0)

    try:
        with pytest.raises(client_mod.ClobBalanceUnavailableError):
            wrapper.get_balance_allowance(max_attempts=1)
        active = load_alert_incidents(path, max_age_hours=1)
        assert len(active) == 1
        assert active[0]["status"] == "active"

        raw.fail = False
        assert wrapper.get_balance_allowance(max_attempts=1) == {
            "balance": "12500000"
        }
        recovered = load_alert_incidents(path, max_age_hours=1)
        assert len(recovered) == 1
        assert recovered[0]["status"] == "recovered"
        assert recovered[0]["recovery_count"] == 1
    finally:
        client_logger.removeHandler(durable_handler)
        client_logger.setLevel(previous_client_level)
        shared_client.close()


def test_stale_earlier_balance_success_cannot_recover_newer_failure(
    tmp_path,
    monkeypatch,
):
    client_logger = logging.getLogger("src.polymarket.client")
    previous_client_level = client_logger.level
    client_logger.setLevel(logging.INFO)
    path = tmp_path / "alerts.jsonl"
    durable_handler = DurableAlertHandler(path)
    client_logger.addHandler(durable_handler)

    class _FailingRawClient:
        def get_balance_allowance(self, _params):
            raise PolyApiException(error_msg="Request exception!")

    class _SuccessfulRawClient:
        def get_balance_allowance(self, _params):
            return {"balance": "12500000"}

    shared_client = httpx.Client()
    newer_wrapper = ClobClientWrapper(
        private_key="dummy",
        funder_address="0xnewer",
    )
    newer_wrapper._client = _FailingRawClient()
    earlier_wrapper = ClobClientWrapper(
        private_key="dummy",
        funder_address="0xearlier",
    )

    def _finish_earlier_initialization():
        try:
            newer_wrapper.get_balance_allowance(max_attempts=1)
        except client_mod.ClobBalanceUnavailableError:
            pass
        earlier_wrapper._client = _SuccessfulRawClient()

    monkeypatch.setattr(
        earlier_wrapper,
        "_ensure_client",
        _finish_earlier_initialization,
    )
    monkeypatch.setattr(
        earlier_wrapper,
        "_configure_shared_transport",
        lambda: shared_client,
    )
    monkeypatch.setattr(
        newer_wrapper,
        "_configure_shared_transport",
        lambda: shared_client,
    )
    monkeypatch.setattr(client_mod, "_balance_recovery_probe_emitted", False)
    monkeypatch.setattr(client_mod, "_balance_read_sequence", 0)
    monkeypatch.setattr(client_mod, "_balance_incident_state_sequence", 0)

    try:
        assert earlier_wrapper.get_balance_allowance(max_attempts=1) == {
            "balance": "12500000"
        }
        active = load_alert_incidents(path, max_age_hours=1)
        assert len(active) == 1
        assert active[0]["status"] == "active"
        assert active[0]["recovery_count"] == 0
    finally:
        client_logger.removeHandler(durable_handler)
        client_logger.setLevel(previous_client_level)
        shared_client.close()


def test_balance_alert_is_aggregated_and_next_confirmed_read_recovers(
    tmp_path,
    monkeypatch,
):
    helper_logger = logging.getLogger(client_mod._CLOB_HELPER_LOGGER_NAME)
    client_logger = logging.getLogger("src.polymarket.client")
    previous_helper_level = helper_logger.level
    previous_client_level = client_logger.level
    helper_logger.setLevel(logging.INFO)
    client_logger.setLevel(logging.INFO)

    path = tmp_path / "alerts.jsonl"
    durable_handler = DurableAlertHandler(path)
    helper_records = []
    client_records = []

    class _CaptureHandler(logging.Handler):
        def __init__(self, records):
            super().__init__()
            self.records = records

        def emit(self, record):
            self.records.append(record)

    helper_capture_handler = _CaptureHandler(helper_records)
    client_capture_handler = _CaptureHandler(client_records)
    helper_logger.addHandler(durable_handler)
    helper_logger.addHandler(helper_capture_handler)
    client_logger.addHandler(durable_handler)
    client_logger.addHandler(client_capture_handler)

    class _RawClient:
        def __init__(self):
            self.fail = True

        def get_balance_allowance(self, _params):
            if self.fail:
                helper_logger.error(
                    "[py_clob_client_v2] request error: read timed out"
                )
                raise PolyApiException(error_msg="Request exception!")
            return {"balance": "12500000"}

    raw = _RawClient()
    shared_client = httpx.Client()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(wrapper, "_configure_shared_transport", lambda: shared_client)
    monkeypatch.setattr(client_mod.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(client_mod, "_balance_recovery_probe_emitted", False)
    monkeypatch.setattr(client_mod, "_balance_read_sequence", 0)
    monkeypatch.setattr(client_mod, "_balance_incident_state_sequence", 0)

    try:
        unavailable = wrapper.get_cash_balance_details(
            allow_onchain_fallback=False
        )

        assert unavailable == {"balance": 0.0, "source": "unavailable"}
        active = load_alert_incidents(path, max_age_hours=1)
        assert len(active) == 1
        assert active[0]["status"] == "active"
        assert active[0]["occurrence_count"] == 1
        assert active[0]["incident_key"] == client_mod.CLOB_BALANCE_INCIDENT_KEY
        assert helper_records
        assert {record.levelno for record in helper_records} == {logging.INFO}
        terminal_warnings = [
            record
            for record in client_records
            if record.levelno == logging.WARNING
            and "Could not fetch CLOB balance" in record.getMessage()
        ]
        assert len(terminal_warnings) == 1

        raw.fail = False
        confirmed = wrapper.get_cash_balance_details(
            allow_onchain_fallback=False
        )
        repeated = wrapper.get_cash_balance_details(
            allow_onchain_fallback=False
        )

        assert confirmed == {"balance": pytest.approx(12.5), "source": "clob"}
        assert repeated == {"balance": pytest.approx(12.5), "source": "clob"}
        recovered = load_alert_incidents(path, max_age_hours=1)
        assert len(recovered) == 1
        assert recovered[0]["status"] == "recovered"
        assert recovered[0]["occurrence_count"] == 1
        assert recovered[0]["recovery_count"] == 1
    finally:
        helper_logger.removeHandler(durable_handler)
        helper_logger.removeHandler(helper_capture_handler)
        client_logger.removeHandler(durable_handler)
        client_logger.removeHandler(client_capture_handler)
        helper_logger.setLevel(previous_helper_level)
        client_logger.setLevel(previous_client_level)
        shared_client.close()


def test_open_orders_alert_is_aggregated_and_next_success_recovers(
    tmp_path,
    monkeypatch,
):
    helper_logger = logging.getLogger(client_mod._CLOB_HELPER_LOGGER_NAME)
    client_logger = logging.getLogger("src.polymarket.client")
    previous_helper_level = helper_logger.level
    previous_client_level = client_logger.level
    helper_logger.setLevel(logging.INFO)
    client_logger.setLevel(logging.INFO)

    path = tmp_path / "alerts.jsonl"
    durable_handler = DurableAlertHandler(path)
    helper_records = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            helper_records.append(record)

    capture_handler = _CaptureHandler()
    helper_logger.addHandler(durable_handler)
    helper_logger.addHandler(capture_handler)
    client_logger.addHandler(durable_handler)

    class _RawClient:
        def __init__(self):
            self.fail = True

        def get_open_orders(self):
            if self.fail:
                helper_logger.error(
                    "[py_clob_client_v2] request error: read timed out"
                )
                raise PolyApiException(error_msg="Request exception!")
            return []

    raw = _RawClient()
    shared_client = httpx.Client()
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = raw
    monkeypatch.setattr(wrapper, "_configure_shared_transport", lambda: shared_client)
    monkeypatch.setattr(client_mod.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(client_mod, "_open_orders_recovery_probe_emitted", False)

    try:
        with pytest.raises(client_mod.ClobOpenOrdersUnavailableError):
            wrapper.get_open_orders(max_attempts=2)

        active = load_alert_incidents(path, max_age_hours=1)
        assert len(active) == 1
        assert active[0]["status"] == "active"
        assert active[0]["occurrence_count"] == 1
        assert active[0]["incident_key"] == client_mod.CLOB_OPEN_ORDERS_INCIDENT_KEY
        assert helper_records
        assert {record.levelno for record in helper_records} == {logging.INFO}

        raw.fail = False
        assert wrapper.get_open_orders(max_attempts=1) == []
        assert wrapper.get_open_orders(max_attempts=1) == []

        recovered = load_alert_incidents(path, max_age_hours=1)
        assert len(recovered) == 1
        assert recovered[0]["status"] == "recovered"
        assert recovered[0]["occurrence_count"] == 1
        assert recovered[0]["recovery_count"] == 1

        helper_logger.error("[py_clob_client_v2] request error outside open orders")
        assert helper_records[-1].levelno == logging.ERROR
        after_external_error = load_alert_incidents(path, max_age_hours=1)
        assert len(after_external_error) == 2
        assert any(
            incident["source"] == client_mod._CLOB_HELPER_LOGGER_NAME
            and incident["status"] == "active"
            for incident in after_external_error
        )
    finally:
        helper_logger.removeHandler(durable_handler)
        helper_logger.removeHandler(capture_handler)
        client_logger.removeHandler(durable_handler)
        helper_logger.setLevel(previous_helper_level)
        client_logger.setLevel(previous_client_level)
        shared_client.close()


def test_v2_wrapper_normalizes_raw_dict_orderbook():
    class _RawClient:
        def get_order_book(self, _token_id):
            return {
                "market": "market-1",
                "asset_id": "token-1",
                "bids": [
                    {"price": "0.61", "size": "10"},
                    {"price": "0.63", "size": "4"},
                ],
                "asks": [
                    {"price": "0.68", "size": "8"},
                    {"price": "0.66", "size": "6"},
                ],
            }

    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = _RawClient()

    book = wrapper.get_orderbook("token-1")

    assert book["market"] == "market-1"
    assert book["asset_id"] == "token-1"
    assert book["bids"] == [
        {"price": "0.63", "size": "4"},
        {"price": "0.61", "size": "10"},
    ]
    assert book["asks"] == [
        {"price": "0.66", "size": "6"},
        {"price": "0.68", "size": "8"},
    ]


def test_v2_wrapper_keeps_legacy_orderbook_object_compatibility():
    class _Level:
        def __init__(self, price, size):
            self.price = price
            self.size = size

    class _Book:
        bids = [_Level("0.58", "3"), _Level("0.62", "7")]
        asks = [_Level("0.71", "5"), _Level("0.69", "2")]

    class _RawClient:
        def get_order_book(self, _token_id):
            return _Book()

    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = _RawClient()

    book = wrapper.get_orderbook("token-1")

    assert book["bids"] == [
        {"price": "0.62", "size": "7"},
        {"price": "0.58", "size": "3"},
    ]
    assert book["asks"] == [
        {"price": "0.69", "size": "2"},
        {"price": "0.71", "size": "5"},
    ]


def test_market_buy_wrapper_returns_submitted_amount_metadata(monkeypatch):
    class _RawClient:
        def create_and_post_market_order(self, args, options, order_type):
            args.price = 0.62
            args.amount = 24.25
            return {"orderID": "order-1"}

    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = _RawClient()
    monkeypatch.setattr(
        wrapper,
        "get_geoblock_status",
        lambda: {"blocked": False, "status_code": 200, "ip": "", "country": "", "region": "", "error": ""},
    )

    response = wrapper.create_market_order(
        token_id="token-yes",
        side="BUY",
        amount=25.0,
        tick_size="0.01",
        neg_risk=False,
        user_usdc_balance=25.0,
    )

    assert response["orderID"] == "order-1"
    assert response["_requested_amount"] == pytest.approx(25.0)
    assert response["_submitted_amount"] == pytest.approx(24.25)
    assert response["_execution_price"] == pytest.approx(0.62)


def test_executor_uses_marketable_limit_for_immediate_buy_accounting(tmp_path):
    class _Clob:
        def __init__(self):
            self.limit_order = None

        def get_open_orders(self):
            return []

        def create_limit_order(self, **kwargs):
            self.limit_order = kwargs
            return {"orderID": "order-1", "status": "live"}

    bankroll = BankrollManager(
        initial_bankroll=100.0,
        max_bet_fraction=1.0,
        auto_detect_balance=False,
    )
    clob = _Clob()
    executor = OrderExecutor(bankroll=bankroll, clob_client=clob, dry_run=False)
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")
    executor._authoritative_wallet_conflict = lambda **_kwargs: (False, "")
    executor._check_liquidity = lambda *args, **kwargs: {
        "ok": True,
        "adjusted_size": 25.0,
        "available_liquidity": 100.0,
        "best_ask_liquidity": 10.0,
        "slippage": 0.0,
        "best_ask": 0.60,
        "reason": "",
    }

    result = executor._place_bet(_base_bet(), pd.DataFrame())

    assert result["status"] == "placed"
    assert result["requested_bet_size_usd"] == pytest.approx(25.0)
    assert result["bet_size_usd"] == pytest.approx(25.0)
    assert result["shares"] == pytest.approx(41.67, rel=1e-3)
    assert result["order_type"] == "marketable_limit"
    assert clob.limit_order["price"] == pytest.approx(0.60)
    assert clob.limit_order["size"] == pytest.approx(25.0 / 0.60)
    assert bankroll.bankroll == pytest.approx(75.0)

    ledger_bet = BetLedger(path=tmp_path / "ledger.json").bets[0]
    assert ledger_bet["amount"] == pytest.approx(25.0)
    assert ledger_bet["shares"] == pytest.approx(41.67)
    assert ledger_bet["requested_amount"] == pytest.approx(25.0)
    assert ledger_bet["submitted_amount"] == pytest.approx(25.0)
    assert ledger_bet["order_type"] == "marketable_limit"


def test_marketable_limit_uses_full_size_when_best_ask_liquidity_is_partial(tmp_path):
    class _Clob:
        def __init__(self):
            self.limit_order = None

        def get_open_orders(self):
            return []

        def get_orderbook(self, _token_id):
            return {"asks": [{"price": "0.50", "size": "60"}], "bids": []}

        def create_limit_order(self, **kwargs):
            self.limit_order = kwargs
            return {"orderID": "order-1", "status": "live"}

    clob = _Clob()
    bankroll = BankrollManager(
        initial_bankroll=500.0,
        max_bet_fraction=1.0,
        auto_detect_balance=False,
    )
    executor = OrderExecutor(bankroll=bankroll, clob_client=clob, dry_run=False)
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")
    executor._authoritative_wallet_conflict = lambda **_kwargs: (False, "")

    result = executor._place_bet(
        _base_bet(
            market_prob=0.50,
            decimal_odds=2.0,
            override_bet_size=50.0,
        ),
        pd.DataFrame(),
    )

    assert result["status"] == "placed"
    assert result["order_type"] == "marketable_limit"
    assert result["bet_size_usd"] == pytest.approx(50.0)
    assert clob.limit_order["price"] == pytest.approx(0.50)
    assert clob.limit_order["size"] == pytest.approx(100.0)
    assert bankroll.bankroll == pytest.approx(450.0)


def test_marketable_limit_two_dollar_order_clears_rounded_minimum(tmp_path):
    class _Clob:
        def __init__(self):
            self.limit_order = None

        def get_open_orders(self):
            return []

        def get_orderbook(self, _token_id):
            return {"asks": [{"price": "0.57", "size": "100"}], "bids": []}

        def create_limit_order(self, **kwargs):
            self.limit_order = kwargs
            return {"orderID": "order-1", "status": "live"}

    clob = _Clob()
    bankroll = BankrollManager(
        initial_bankroll=10.0,
        max_bet_fraction=1.0,
        auto_detect_balance=False,
    )
    executor = OrderExecutor(bankroll=bankroll, clob_client=clob, dry_run=False)
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")
    executor._authoritative_wallet_conflict = lambda **_kwargs: (False, "")

    result = executor._place_bet(
        _base_bet(
            blended_prob=0.70,
            model_prob=0.70,
            market_prob=0.57,
            decimal_odds=1.7544,
            override_bet_size=2.0,
        ),
        pd.DataFrame(),
    )

    assert result["status"] == "placed"
    assert result["order_type"] == "marketable_limit"
    assert result["bet_size_usd"] == pytest.approx(2.01)
    assert clob.limit_order["price"] == pytest.approx(0.57)
    assert clob.limit_order["size"] == pytest.approx(3.51)
    assert clob.limit_order["price"] * clob.limit_order["size"] >= 2.0
    assert bankroll.bankroll == pytest.approx(7.99)

    ledger_bet = BetLedger(path=tmp_path / "ledger.json").bets[0]
    assert ledger_bet["amount"] == pytest.approx(2.01)
    assert ledger_bet["shares"] == pytest.approx(3.51)


def test_resting_limit_two_dollar_order_clears_rounded_minimum(tmp_path):
    class _Clob:
        def __init__(self):
            self.limit_order = None

        def get_open_orders(self):
            return []

        def get_orderbook(self, _token_id):
            return {"asks": [{"price": "0.58", "size": "100"}], "bids": []}

        def create_limit_order(self, **kwargs):
            self.limit_order = kwargs
            return {"orderID": "limit-1", "status": "live"}

    clob = _Clob()
    bankroll = BankrollManager(
        initial_bankroll=10.0,
        max_bet_fraction=1.0,
        auto_detect_balance=False,
    )
    executor = OrderExecutor(
        bankroll=bankroll,
        clob_client=clob,
        dry_run=False,
        force_limit_order=True,
    )
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")
    executor._authoritative_wallet_conflict = lambda **_kwargs: (False, "")

    result = executor._place_bet(
        _base_bet(
            blended_prob=0.70,
            model_prob=0.70,
            market_prob=0.58,
            decimal_odds=1.7241,
            override_bet_size=2.0,
        ),
        pd.DataFrame(),
    )

    assert result["status"] == "placed"
    assert result["order_type"] == "limit_bid"
    assert result["bet_size_usd"] == pytest.approx(2.01)
    assert clob.limit_order["price"] == pytest.approx(0.57)
    assert clob.limit_order["size"] == pytest.approx(3.51)
    assert clob.limit_order["price"] * clob.limit_order["size"] >= 2.0
    assert bankroll.bankroll == pytest.approx(7.99)


def test_near_miss_limit_two_dollar_order_clears_rounded_minimum(tmp_path):
    class _Clob:
        def __init__(self):
            self.limit_order = None

        def get_open_orders(self):
            return []

        def create_limit_order(self, **kwargs):
            self.limit_order = kwargs
            return {"orderID": "near-miss-1", "status": "live"}

    clob = _Clob()
    bankroll = BankrollManager(
        initial_bankroll=10.0,
        max_bet_fraction=1.0,
        auto_detect_balance=False,
    )
    bankroll.kelly_bet_size = lambda *_args, **_kwargs: 2.0
    executor = OrderExecutor(bankroll=bankroll, clob_client=clob, dry_run=False)
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")
    executor._authoritative_wallet_conflict = lambda **_kwargs: (False, "")
    executor._authoritative_open_clob_order_conflict = lambda **_kwargs: (False, "")

    result = executor._place_near_miss_limit(
        _base_bet(
            blended_prob=0.59,
            model_prob=0.59,
            market_prob=0.58,
            edge=0.01,
            decimal_odds=1.7241,
            override_bet_size=None,
        ),
        pd.DataFrame(),
    )

    assert result["status"] == "placed"
    assert result["order_type"] == "near_miss_limit"
    assert result["bet_size_usd"] == pytest.approx(2.01)
    assert clob.limit_order["price"] == pytest.approx(0.56)
    assert clob.limit_order["size"] == pytest.approx(3.58)
    assert clob.limit_order["price"] * clob.limit_order["size"] >= 2.0
    assert bankroll.bankroll == pytest.approx(7.99)


def test_version_aware_collateral_fallback_selects_token(monkeypatch):
    seen_tokens = []

    def fake_post(_url, *, json, timeout):
        seen_tokens.append(json["params"][0]["to"])
        return _FakeResponse("0x0f4240")

    monkeypatch.setattr("src.polymarket.client.requests.post", fake_post)

    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    monkeypatch.setattr(wrapper, "_get_clob_backend_version", lambda: 1)
    balance, source = wrapper._get_onchain_collateral_balance("0x" + "11" * 20)

    assert balance == pytest.approx(1.0)
    assert source == "onchain_v1_collateral"
    assert seen_tokens[-1] == LEGACY_POLYGON_USDC_E

    monkeypatch.setattr(wrapper, "_get_clob_backend_version", lambda: 2)
    balance, source = wrapper._get_onchain_collateral_balance("0x" + "11" * 20)

    assert balance == pytest.approx(1.0)
    assert source == "onchain_v2_collateral"
    assert seen_tokens[-1] != LEGACY_POLYGON_USDC_E


def test_market_buy_path_uses_fee_adjusted_net_edge(tmp_path):
    class _Clob:
        market_calls = 0

        def get_open_orders(self):
            return []

        def create_market_order(self, **_kwargs):
            self.market_calls += 1
            return {"orderID": "unexpected"}

    clob = _Clob()
    executor = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=100.0, auto_detect_balance=False),
        clob_client=clob,
        dry_run=False,
    )
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")
    executor._authoritative_wallet_conflict = lambda **_kwargs: (False, "")
    executor._check_liquidity = lambda *args, **kwargs: {
        "ok": True,
        "adjusted_size": 25.0,
        "available_liquidity": 100.0,
        "slippage": 0.0,
        "best_ask": 0.62,
        "reason": "",
    }

    result = executor._place_bet(
        _base_bet(blended_prob=0.65, model_prob=0.65, fee_rate=0.10, fee_exponent=1.0),
        pd.DataFrame(),
    )

    assert result is None
    assert clob.market_calls == 0


def test_maker_limit_path_is_not_suppressed_by_taker_fee_metadata(tmp_path):
    class _Clob:
        def __init__(self):
            self.limit_calls = 0

        def get_open_orders(self):
            return []

        def create_limit_order(self, **_kwargs):
            self.limit_calls += 1
            return {"orderID": "limit-1"}

    clob = _Clob()
    executor = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=100.0, auto_detect_balance=False),
        clob_client=clob,
        dry_run=False,
    )
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")
    executor._authoritative_wallet_conflict = lambda **_kwargs: (False, "")
    executor._check_liquidity = lambda *args, **kwargs: {
        "ok": True,
        "adjusted_size": 25.0,
        "available_liquidity": 100.0,
        "slippage": 0.0,
        "best_ask": 0.64,
        "reason": "",
    }

    result = executor._place_bet(
        _base_bet(blended_prob=0.65, model_prob=0.65, fee_rate=1.0, fee_exponent=1.0),
        pd.DataFrame(),
    )

    assert result["status"] == "placed"
    assert result["order_type"] == "limit_bid"
    assert clob.limit_calls == 1
