import logging

import py_clob_client_v2.http_helpers.helpers as clob_helpers
from py_clob_client_v2.exceptions import PolyApiException

import src.polymarket.client as client_mod
from src.polymarket.client import ClobClientWrapper, is_uncertain_clob_order_submission_error


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.content = b"x"

    def json(self):
        return self._payload


class _FakeSharedClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, headers, timeout))
        return _FakeResponse(self.payload, status_code=200)


class _FakeClobClient:
    def __init__(self):
        self.limit_orders = []
        self.market_orders = []
        self.posted = []

    def create_and_post_order(self, order_args, options, order_type):
        self.limit_orders.append((order_args, options))
        self.posted.append((order_args, order_type))
        return {"orderID": "stub-order"}

    def create_and_post_market_order(self, market_args, options, order_type):
        self.market_orders.append((market_args, options))
        self.posted.append((market_args, order_type))
        market_args.price = market_args.price or 0.5
        return {"orderID": "stub-order"}


def test_get_geoblock_status_uses_shared_clob_transport(monkeypatch):
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    shared_client = _FakeSharedClient(
        {
            "blocked": False,
            "ip": "163.176.191.39",
            "country": "BR",
            "region": "Sao Paulo",
        }
    )

    monkeypatch.setattr(wrapper, "_ensure_client", lambda: None)
    monkeypatch.setattr(wrapper, "_configure_shared_transport", lambda: shared_client)

    result = wrapper.get_geoblock_status()

    assert result["blocked"] is False
    assert result["ip"] == "163.176.191.39"
    assert result["country"] == "BR"
    assert result["region"] == "Sao Paulo"
    assert shared_client.calls[0][0] == client_mod.GEOBLOCK_CHECK_URL
    assert shared_client.calls[0][2] == client_mod.GEOBLOCK_CHECK_TIMEOUT_SECONDS


def test_limit_order_logs_geoblock_status_before_post(caplog, monkeypatch):
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = _FakeClobClient()

    monkeypatch.setattr(
        wrapper,
        "get_geoblock_status",
        lambda: {
            "status_code": 200,
            "blocked": False,
            "ip": "163.176.191.39",
            "country": "BR",
            "region": "Sao Paulo",
            "error": "",
        },
    )

    with caplog.at_level(logging.INFO, logger="src.polymarket.client"):
        response = wrapper.create_limit_order(
            token_id="1234567890abcdef",
            side="BUY",
            price=0.73,
            size=10,
        )

    assert response["orderID"] == "stub-order"
    assert any(
        "Geoblock check before limit order: blocked=False ip=163.176.191.39 country=BR region=Sao Paulo status=200"
        in record.message
        for record in caplog.records
    )
    assert wrapper._client.posted


def test_market_order_logs_blocked_geoblock_status_as_warning(caplog, monkeypatch):
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._client = _FakeClobClient()

    monkeypatch.setattr(
        wrapper,
        "get_geoblock_status",
        lambda: {
            "status_code": 200,
            "blocked": True,
            "ip": "151.197.34.204",
            "country": "US",
            "region": "PA",
            "error": "",
        },
    )

    with caplog.at_level(logging.INFO, logger="src.polymarket.client"):
        response = wrapper.create_market_order(
            token_id="1234567890abcdef",
            side="BUY",
            amount=25,
        )

    assert response["orderID"] == "stub-order"
    assert any(
        record.levelname == "WARNING"
        and "Geoblock check before market order: blocked=True ip=151.197.34.204 country=US region=PA status=200"
        in record.message
        for record in caplog.records
    )
    assert wrapper._client.posted


def test_proxy_is_applied_before_api_key_derivation(monkeypatch):
    marker_client = object()
    client_kwargs = {}

    class _CtorClobClient:
        def __init__(self, *args, **kwargs):
            pass

        def create_or_derive_api_key(self):
            raise AssertionError("wrapper should use derive-first auth")

        def derive_api_key(self):
            assert clob_helpers._http_client is marker_client
            return {"apiKey": "k"}

        def create_api_key(self):
            raise AssertionError("create_api_key should not run when derive succeeds")

        def set_api_creds(self, creds):
            self.creds = creds

    monkeypatch.setenv("CLOB_PROXY_URL", "http://user:pass@163.176.191.39:3128")
    monkeypatch.setattr(client_mod, "_proxy_patched", False)

    import httpx
    import py_clob_client_v2.client as py_client

    def _fake_httpx_client(**kwargs):
        client_kwargs.update(kwargs)
        return marker_client

    monkeypatch.setattr(httpx, "Client", _fake_httpx_client)
    monkeypatch.setattr(py_client, "ClobClient", _CtorClobClient)
    monkeypatch.setattr(
        ClobClientWrapper,
        "_discover_proxy_address",
        lambda self: "0xFf9747b606699347895B14C6405Df21e9170E7cA",
    )

    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    wrapper._ensure_client()

    assert wrapper._client is not None
    assert client_mod._proxy_patched is True
    assert client_kwargs["proxy"] == "http://user:pass@163.176.191.39:3128"
    assert client_kwargs["http2"] is False


def test_proxy_http2_can_be_enabled_explicitly(monkeypatch):
    marker_client = object()
    client_kwargs = {}

    monkeypatch.setenv("CLOB_PROXY_URL", "http://user:pass@163.176.191.39:3128")
    monkeypatch.setenv("CLOB_PROXY_HTTP2_ENABLED", "1")
    monkeypatch.setattr(client_mod, "_proxy_patched", False)

    import httpx

    def _fake_httpx_client(**kwargs):
        client_kwargs.update(kwargs)
        return marker_client

    monkeypatch.setattr(httpx, "Client", _fake_httpx_client)

    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")
    assert wrapper._configure_shared_transport() is marker_client
    assert client_kwargs["http2"] is True


def test_uncertain_order_submission_includes_clob_transport_state_error():
    exc = RuntimeError(
        "[py_clob_client_v2] request error: Invalid input "
        "StreamInputs.SEND_HEADERS in state 5"
    )

    assert is_uncertain_clob_order_submission_error(exc) is True


def test_api_key_bootstrap_creates_only_after_derive_fails():
    calls = []

    class _Client:
        def derive_api_key(self):
            calls.append("derive")
            raise RuntimeError("missing key")

        def create_api_key(self):
            calls.append("create")
            return {"apiKey": "new"}

    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")

    assert wrapper._derive_or_create_api_key(_Client()) == {"apiKey": "new"}
    assert calls == ["derive", "create"]


def test_api_key_bootstrap_retries_transient_derive_before_success(monkeypatch):
    calls = []

    class _Client:
        def derive_api_key(self):
            calls.append("derive")
            if len(calls) == 1:
                raise PolyApiException(error_msg="Request exception!")
            return {"apiKey": "existing"}

        def create_api_key(self):
            raise AssertionError("create_api_key should not run after transient derive")

    monkeypatch.setattr(client_mod.time, "sleep", lambda _seconds: None)
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")

    assert wrapper._derive_or_create_api_key(_Client()) == {"apiKey": "existing"}
    assert calls == ["derive", "derive"]


def test_api_key_bootstrap_does_not_create_after_transient_derive_failures(monkeypatch):
    calls = []

    class _Client:
        def derive_api_key(self):
            calls.append("derive")
            raise PolyApiException(error_msg="Request exception!")

        def create_api_key(self):
            calls.append("create")
            raise AssertionError("create_api_key should not run on transient derive errors")

    monkeypatch.setattr(client_mod.time, "sleep", lambda _seconds: None)
    wrapper = ClobClientWrapper(private_key="dummy", funder_address="0xabc")

    try:
        wrapper._derive_or_create_api_key(_Client())
    except PolyApiException:
        pass
    else:
        raise AssertionError("expected transient derive failure to be re-raised")

    assert calls == ["derive", "derive", "derive"]
