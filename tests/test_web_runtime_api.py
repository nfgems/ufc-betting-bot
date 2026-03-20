import pytest

from src.web import app as web_app
from src.web import serve as web_serve


def _runtime_status(**overrides):
    status = {
        "service": "ufc-betting-bot",
        "startup_source": "test",
        "requested_live_mode": "off",
        "requested_live_mode_raw": "off",
        "effective_live_mode": "off",
        "trading_enabled": False,
        "trading_live": False,
        "model_name": "xgboost",
        "host": "127.0.0.1",
        "public_bind": False,
        "ready": True,
        "errors": [],
        "warnings": [],
        "checks": [],
        "components": {},
    }
    status.update(overrides)
    return status


@pytest.fixture(autouse=True)
def _reset_runtime_state(monkeypatch):
    monkeypatch.setattr(web_app, "_server_host", "127.0.0.1")
    web_app.set_runtime_status(_runtime_status())


def test_healthz_stays_up_even_when_not_ready():
    web_app.set_runtime_status(
        _runtime_status(
            ready=False,
            errors=["missing startup checks"],
        )
    )
    client = web_app.app.test_client()

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["ready"] is False
    assert payload["effective_live_mode"] == "off"


def test_readyz_returns_503_with_runtime_status_when_not_ready():
    web_app.set_runtime_status(
        _runtime_status(
            ready=False,
            requested_live_mode="real",
            requested_live_mode_raw="real",
            errors=["real trading is blocked"],
        )
    )
    client = web_app.app.test_client()

    response = client.get("/readyz")

    assert response.status_code == 503
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["ready"] is False
    assert payload["errors"] == ["real trading is blocked"]


def test_production_boot_does_not_start_betting_thread_by_default(monkeypatch):
    threads = []
    statuses = []
    components = []
    start_calls = []

    class _FakeThread:
        def __init__(self, target, args=(), kwargs=None, daemon=None):
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}
            self.daemon = daemon
            threads.append(self)

        def start(self):
            return None

    monkeypatch.delenv("LIVE_TRADING_MODE", raising=False)
    monkeypatch.setenv("PORT", "5050")
    monkeypatch.setattr(web_serve.threading, "Thread", _FakeThread)
    monkeypatch.setattr(web_app, "set_runtime_status", lambda status: statuses.append(status))
    monkeypatch.setattr(web_app, "update_runtime_component", lambda *args: components.append(args))
    monkeypatch.setattr(web_app, "set_clob_client", lambda client: None)
    monkeypatch.setattr(web_app, "start_server", lambda **kwargs: start_calls.append(kwargs))

    web_serve.main()

    assert any(thread.target == web_serve.run_background_monitor for thread in threads)
    assert not any(thread.target == web_serve.run_live_betting_loop for thread in threads)
    assert statuses[0]["requested_live_mode"] == "off"
    assert start_calls[0]["host"] == "127.0.0.1"


def test_production_boot_starts_betting_thread_when_policy_allows(monkeypatch):
    threads = []

    class _FakeThread:
        def __init__(self, target, args=(), kwargs=None, daemon=None):
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}
            self.daemon = daemon
            threads.append(self)

        def start(self):
            return None

    monkeypatch.setenv("PORT", "5050")
    monkeypatch.setattr(web_serve.threading, "Thread", _FakeThread)
    monkeypatch.setattr(
        web_serve,
        "evaluate_live_startup",
        lambda **kwargs: _runtime_status(
            startup_source="serve",
            requested_live_mode="dry-run",
            requested_live_mode_raw="dry-run",
            effective_live_mode="dry-run",
            trading_enabled=True,
            ready=True,
        ),
    )
    monkeypatch.setattr(web_app, "set_runtime_status", lambda status: None)
    monkeypatch.setattr(web_app, "update_runtime_component", lambda *args: None)
    monkeypatch.setattr(web_app, "set_clob_client", lambda client: None)
    monkeypatch.setattr(web_app, "start_server", lambda **kwargs: None)

    web_serve.main()

    betting_threads = [thread for thread in threads if thread.target == web_serve.run_live_betting_loop]
    assert len(betting_threads) == 1
    assert betting_threads[0].kwargs["trading_mode"] == "dry-run"
