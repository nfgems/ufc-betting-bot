import logging

import py_clob_client.http_helpers.helpers as clob_helpers

import src.polymarket.client as client_mod
from src.polymarket.client import ClobClientWrapper


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

    def get(self, url, headers=None):
        self.calls.append((url, headers))
        return _FakeResponse(self.payload, status_code=200)


class _FakeClobClient:
    def __init__(self):
        self.limit_orders = []
        self.market_orders = []
        self.posted = []

    def create_order(self, order_args, options):
        self.limit_orders.append((order_args, options))
        return {"signed": "limit"}

    def create_market_order(self, market_args, options):
        self.market_orders.append((market_args, options))
        return {"signed": "market"}

    def post_order(self, signed_order, order_type):
        self.posted.append((signed_order, order_type))
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

    class _CtorClobClient:
        def __init__(self, *args, **kwargs):
            pass

        def create_or_derive_api_creds(self):
            assert clob_helpers._http_client is marker_client
            return {"apiKey": "k"}

        def set_api_creds(self, creds):
            self.creds = creds

    monkeypatch.setenv("CLOB_PROXY_URL", "http://user:pass@163.176.191.39:3128")
    monkeypatch.setattr(client_mod, "_proxy_patched", False)

    import httpx
    import py_clob_client.client as py_client

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: marker_client)
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
