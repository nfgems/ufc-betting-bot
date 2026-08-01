import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from src.polymarket.tracker import ReadOnlyBetLedgerView
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
    monkeypatch.delenv("BTC5M_LIVE_PROFILES", raising=False)
    monkeypatch.delenv("BTC5M_LIVE_LEDGER_DIR", raising=False)
    web_app._endpoint_cache.clear()
    web_app._endpoint_inflight.clear()
    web_app._background_cache_refreshes = 0
    web_app._timed_call_inflight.clear()
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


def test_api_runtime_status_returns_components_without_probe_failure():
    web_app.set_runtime_status(
        _runtime_status(
            ready=True,
            components={
                "ufc_refresh_loop": {
                    "state": "degraded",
                    "message": "Last UFC refresh failed",
                    "coverage_alerts": ["refresh failure: boom"],
                }
            },
        )
    )
    client = web_app.app.test_client()

    response = client.get("/api/runtime-status")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"
    payload = response.get_json()
    assert payload["components"]["ufc_refresh_loop"]["state"] == "degraded"
    assert payload["components"]["ufc_refresh_loop"]["coverage_alerts"] == ["refresh failure: boom"]


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
    monkeypatch.setattr(web_app, "update_runtime_component", lambda *args, **kwargs: components.append(args))
    monkeypatch.setattr(web_app, "set_clob_client", lambda client: None)
    monkeypatch.setattr(web_app, "start_server", lambda **kwargs: start_calls.append(kwargs))

    web_serve.main()

    assert any(thread.target == web_serve.run_background_monitor for thread in threads)
    assert not any(
        thread.target == web_serve._run_live_betting_loop_guarded for thread in threads
    )
    assert statuses[0]["requested_live_mode"] == "off"
    assert start_calls[0]["host"] == "0.0.0.0"


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
    monkeypatch.setattr(web_app, "update_runtime_component", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_app, "set_clob_client", lambda client: None)
    monkeypatch.setattr(web_app, "start_server", lambda **kwargs: None)

    web_serve.main()

    betting_threads = [
        thread
        for thread in threads
        if thread.target == web_serve._run_live_betting_loop_guarded
    ]
    assert len(betting_threads) == 1
    assert betting_threads[0].kwargs["trading_mode"] == "dry-run"


def test_production_boot_gates_refresh_behind_first_betting_cycle(monkeypatch):
    threads = []
    started = []

    class _FakeThread:
        def __init__(self, target, args=(), kwargs=None, daemon=None):
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}
            self.daemon = daemon
            threads.append(self)

        def start(self):
            started.append(self.target)

    monkeypatch.setenv("PORT", "5050")
    monkeypatch.setenv("UFC_REFRESH_ENABLED", "1")
    monkeypatch.setenv("UFC_REFRESH_INITIAL_DELAY_MINUTES", "0")
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
    monkeypatch.setattr(web_app, "update_runtime_component", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_app, "set_clob_client", lambda client: None)
    monkeypatch.setattr(web_app, "start_server", lambda **kwargs: None)

    web_serve.main()

    betting_thread = next(
        thread
        for thread in threads
        if thread.target == web_serve._run_live_betting_loop_guarded
    )
    refresh_thread = next(
        thread
        for thread in threads
        if thread.target == web_serve.run_background_ufc_refresh_loop
    )
    gate = betting_thread.kwargs["first_cycle_complete_event"]

    assert started.index(web_serve._run_live_betting_loop_guarded) < started.index(
        web_serve.run_background_ufc_refresh_loop
    )
    assert isinstance(gate, threading.Event)
    assert refresh_thread.kwargs["startup_gate"] is gate


def test_guarded_betting_thread_releases_refresh_gate_on_unexpected_crash(monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(
        web_serve,
        "run_live_betting_loop",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("startup crash")),
    )

    with pytest.raises(RuntimeError, match="startup crash"):
        web_serve._run_live_betting_loop_guarded(
            first_cycle_complete_event=gate,
        )

    assert gate.is_set()


def test_production_boot_starts_btc5m_threads_for_promoted_dry_run_profiles(monkeypatch, tmp_path):
    threads = []
    statuses = []
    live_ledger_dir = tmp_path / "live_ledgers"

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
    monkeypatch.setenv("LIVE_TRADING_MODE", "dry-run")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture,cheap_side")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_ledger_dir))
    monkeypatch.setattr(web_serve.threading, "Thread", _FakeThread)
    monkeypatch.setattr(
        web_serve,
        "evaluate_live_startup",
        lambda **kwargs: _runtime_status(
            startup_source="serve",
            requested_live_mode="off",
            requested_live_mode_raw="off",
            effective_live_mode="off",
            trading_enabled=False,
            ready=True,
        ),
    )
    monkeypatch.setattr(web_app, "set_runtime_status", lambda status: statuses.append(status))
    monkeypatch.setattr(web_app, "update_runtime_component", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_app, "set_clob_client", lambda client: None)
    monkeypatch.setattr(web_app, "start_server", lambda **kwargs: None)

    web_serve.main()

    btc_threads = [thread for thread in threads if thread.target == web_serve.run_btc5m_live_loop]
    assert len(btc_threads) == 2
    assert [thread.kwargs["profile_name"] for thread in btc_threads] == ["late_capture", "cheap_side"]
    assert all(thread.kwargs["trading_mode"] == "dry-run" for thread in btc_threads)
    assert {thread.kwargs["ledger_path"].name for thread in btc_threads} == {
        "late_capture.json",
        "cheap_side.json",
    }
    assert statuses[0]["components"]["btc5m_loop"]["state"] == "starting"
    assert statuses[0]["btc5m_live_startup"]["effective_live_mode"] == "dry-run"
    assert live_ledger_dir.exists()


def test_production_boot_starts_btc5m_threads_paused_when_emergency_stop_active(monkeypatch, tmp_path):
    threads = []
    statuses = []
    live_ledger_dir = tmp_path / "live_ledgers"
    stop_status = {
        "active": True,
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "path": str(tmp_path / "logs" / "btc5m_emergency_stop.json"),
    }

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
    monkeypatch.setenv("LIVE_TRADING_MODE", "dry-run")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_ledger_dir))
    monkeypatch.setattr(web_app, "btc5m_emergency_stop_status", lambda: dict(stop_status))
    monkeypatch.setattr(web_serve.threading, "Thread", _FakeThread)
    monkeypatch.setattr(
        web_serve,
        "evaluate_live_startup",
        lambda **kwargs: _runtime_status(
            startup_source="serve",
            requested_live_mode="off",
            requested_live_mode_raw="off",
            effective_live_mode="off",
            trading_enabled=False,
            ready=True,
        ),
    )
    monkeypatch.setattr(web_app, "set_runtime_status", lambda status: statuses.append(status))
    monkeypatch.setattr(web_app, "update_runtime_component", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_app, "set_clob_client", lambda client: None)
    monkeypatch.setattr(web_app, "start_server", lambda **kwargs: None)

    web_serve.main()

    btc_threads = [thread for thread in threads if thread.target == web_serve.run_btc5m_live_loop]
    assert len(btc_threads) == 1
    assert statuses[0]["components"]["btc5m_loop"]["state"] == "paused"
    assert statuses[0]["components"]["btc5m_loop:late_capture"]["state"] == "paused"
    assert statuses[0]["btc5m_emergency_stop"]["active"] is True


def test_production_boot_blocks_btc5m_real_without_arming(monkeypatch, tmp_path):
    threads = []
    statuses = []

    class _FakeThread:
        def __init__(self, target, args=(), kwargs=None, daemon=None):
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}
            self.daemon = daemon
            threads.append(self)

        def start(self):
            return None

    for env_name in [
        "POLYMARKET_PRIVATE_KEY",
        "WEB_DASHBOARD_TOKEN",
        "LIVE_TRADING_ARMED",
        "LIVE_TRADING_CONFIRMATION",
    ]:
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("PORT", "5050")
    monkeypatch.setenv("LIVE_TRADING_MODE", "real")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(tmp_path))
    monkeypatch.setattr(web_serve.threading, "Thread", _FakeThread)
    monkeypatch.setattr(
        web_serve,
        "evaluate_live_startup",
        lambda **kwargs: _runtime_status(
            startup_source="serve",
            requested_live_mode="off",
            requested_live_mode_raw="off",
            effective_live_mode="off",
            trading_enabled=False,
            ready=True,
        ),
    )
    monkeypatch.setattr(web_app, "set_runtime_status", lambda status: statuses.append(status))
    monkeypatch.setattr(web_app, "update_runtime_component", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_app, "set_clob_client", lambda client: None)
    monkeypatch.setattr(web_app, "start_server", lambda **kwargs: None)

    web_serve.main()

    assert not any(thread.target == web_serve.run_btc5m_live_loop for thread in threads)
    assert statuses[0]["components"]["btc5m_loop"]["state"] == "disabled"
    assert statuses[0]["btc5m_live_startup"]["effective_live_mode"] == "off"
    errors = " ".join(statuses[0]["btc5m_live_startup"]["errors"])
    assert "POLYMARKET_PRIVATE_KEY" in errors
    assert "LIVE_TRADING_ARMED" in errors
    assert "LIVE_TRADING_CONFIRMATION" in errors


def test_btc5m_live_loop_uses_shared_clob_client(monkeypatch, tmp_path):
    from src.polymarket import btc_5m

    class _LoopExit(Exception):
        pass

    shared_clob = object()
    created = {}
    run_calls = []
    runtime_updates = []

    class _FakeRunner:
        def __init__(
            self,
            *,
            profile,
            ledger_path,
            clob_client=None,
            record_signal_snapshots=False,
        ):
            self.profile = profile
            self.ledger_path = ledger_path
            self.clob_client = clob_client
            created["clob_client"] = clob_client
            created["ledger_path"] = ledger_path
            created["record_signal_snapshots"] = record_signal_snapshots

        def run_once(self, *, dry_run, market_slug=None):
            run_calls.append({"dry_run": dry_run, "market_slug": market_slug})
            return {
                "status": "ok",
                "reason": "test cycle complete",
                "market_slug": "btc-updown-5m-test",
                "orders": [],
            }

    monkeypatch.setattr(btc_5m, "Btc5mRunner", _FakeRunner)
    monkeypatch.setattr(btc_5m, "resolve_btc5m_profile", lambda profile_name: {"name": profile_name})
    monkeypatch.setattr(web_app, "get_clob_client", lambda: shared_clob)
    monkeypatch.setattr(
        web_app,
        "update_runtime_component",
        lambda *args, **kwargs: runtime_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(web_serve.time, "sleep", lambda _seconds: (_ for _ in ()).throw(_LoopExit()))

    with pytest.raises(_LoopExit):
        web_serve.run_btc5m_live_loop(
            profile_name="late_capture",
            ledger_path=tmp_path / "late_capture.json",
            poll_seconds=1,
            trading_mode="dry-run",
            startup_delay_seconds=0,
        )

    assert created["clob_client"] is shared_clob
    assert created["ledger_path"] == tmp_path / "late_capture.json"
    assert created["record_signal_snapshots"] is True
    assert run_calls == [{"dry_run": True, "market_slug": None}]
    assert any(args[0] == "btc5m_loop:late_capture" for args, _kwargs in runtime_updates)


def test_btc5m_live_loop_reports_rate_limit_errors_immediately(monkeypatch, tmp_path):
    from src.polymarket import btc_5m

    class _LoopExit(Exception):
        pass

    runtime_updates = []

    class _FakeRunner:
        def __init__(
            self,
            *,
            profile,
            ledger_path,
            clob_client=None,
            record_signal_snapshots=False,
        ):
            self.profile = profile
            self.ledger_path = ledger_path
            self.clob_client = clob_client
            self.record_signal_snapshots = record_signal_snapshots
            self.ledger = None

        def run_once(self, *, dry_run, market_slug=None):
            return {
                "status": "error",
                "reason": "http_status_429: clob.polymarket.com/book",
                "reason_code": "upstream_rate_limited",
                "http_status": 429,
                "http_endpoint": "clob.polymarket.com/book",
                "retry_after": "2",
                "market_slug": "btc-updown-5m-test",
                "orders": [],
            }

    monkeypatch.setattr(btc_5m, "Btc5mRunner", _FakeRunner)
    monkeypatch.setattr(btc_5m, "resolve_btc5m_profile", lambda profile_name: {"name": profile_name})
    monkeypatch.setattr(web_app, "get_clob_client", lambda: object())
    monkeypatch.setattr(
        web_app,
        "update_runtime_component",
        lambda *args, **kwargs: runtime_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(web_serve.time, "sleep", lambda _seconds: (_ for _ in ()).throw(_LoopExit()))

    with pytest.raises(_LoopExit):
        web_serve.run_btc5m_live_loop(
            profile_name="late_capture",
            ledger_path=tmp_path / "late_capture.json",
            poll_seconds=1,
            trading_mode="dry-run",
            startup_delay_seconds=0,
        )

    degraded = [
        (args, kwargs)
        for args, kwargs in runtime_updates
        if args[0] == "btc5m_loop:late_capture" and args[1] == "degraded"
    ]
    assert degraded
    args, kwargs = degraded[-1]
    assert "rate-limited" in args[2]
    assert kwargs["last_result_reason_code"] == "upstream_rate_limited"
    assert kwargs["last_http_status"] == 429
    assert kwargs["last_http_endpoint"] == "clob.polymarket.com/book"
    assert kwargs["poll_seconds"] == 1.0


def test_btc5m_live_loop_auto_settles_due_ledger(monkeypatch, tmp_path):
    from src.polymarket import btc_5m, tracker as polymarket_tracker

    class _LoopExit(Exception):
        pass

    due_time = datetime.now(timezone.utc) - timedelta(seconds=5)
    settle_calls = []
    runtime_updates = []

    class _FakeLedger:
        def get_open_bets(self, *, fresh=False):
            return [{"id": 1, "event_date": due_time.isoformat()}]

    class _FakeRunner:
        def __init__(
            self,
            *,
            profile,
            ledger_path,
            clob_client=None,
            record_signal_snapshots=False,
        ):
            self.profile = profile
            self.ledger_path = ledger_path
            self.clob_client = clob_client
            self.record_signal_snapshots = record_signal_snapshots
            self.ledger = _FakeLedger()

        def run_once(self, *, dry_run, market_slug=None):
            return {
                "status": "idle",
                "reason": "test cycle complete",
                "market_slug": "btc-updown-5m-test",
                "orders": [],
            }

    monkeypatch.setattr(btc_5m, "Btc5mRunner", _FakeRunner)
    monkeypatch.setattr(btc_5m, "resolve_btc5m_profile", lambda profile_name: {"name": profile_name})
    monkeypatch.setattr(web_app, "get_clob_client", lambda: object())
    monkeypatch.setattr(
        polymarket_tracker,
        "auto_settle_from_polymarket",
        lambda ledger: settle_calls.append(ledger) or 1,
    )
    monkeypatch.setattr(
        web_app,
        "update_runtime_component",
        lambda *args, **kwargs: runtime_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(web_serve.time, "sleep", lambda _seconds: (_ for _ in ()).throw(_LoopExit()))

    with pytest.raises(_LoopExit):
        web_serve.run_btc5m_live_loop(
            profile_name="late_capture",
            ledger_path=tmp_path / "late_capture.json",
            poll_seconds=1,
            trading_mode="dry-run",
            startup_delay_seconds=0,
        )

    assert len(settle_calls) == 1
    assert any(kwargs.get("last_settled_count") == 1 for _args, kwargs in runtime_updates)


def test_btc5m_daily_loss_metadata_auto_resumes_next_utc_day():
    profile = type("_Profile", (), {"daily_loss_limit_usd": 25.0})()
    current = datetime(2026, 6, 23, 21, 30, tzinfo=timezone.utc)

    metadata = web_serve._btc5m_daily_loss_metadata(
        reason_code="daily_loss_limit",
        result={"risk": {"realized_loss_today": 27.5}},
        profile=profile,
        hit_at="2026-06-23T21:30:00+00:00",
        current=current,
    )

    assert metadata["daily_loss_limit_hit_at"] == "2026-06-23T21:30:00+00:00"
    assert metadata["daily_loss_limit_auto_resume_at"] == "2026-06-24T00:00:00+00:00"
    assert metadata["daily_loss_limit_usd"] == 25.0
    assert metadata["daily_loss_realized_today"] == 27.5


def test_btc5m_live_loop_pauses_after_cycle_until_emergency_resume(monkeypatch, tmp_path):
    from src.polymarket import btc_5m

    class _LoopExit(Exception):
        pass

    stop_state = {"active": False}
    run_calls = []
    runtime_updates = []

    class _FakeRunner:
        def __init__(
            self,
            *,
            profile,
            ledger_path,
            clob_client=None,
            record_signal_snapshots=False,
        ):
            self.profile = profile
            self.ledger_path = ledger_path
            self.clob_client = clob_client
            self.record_signal_snapshots = record_signal_snapshots
            self.ledger = None

        def run_once(self, *, dry_run, market_slug=None):
            run_calls.append({"dry_run": dry_run, "market_slug": market_slug})
            if len(run_calls) == 1:
                stop_state["active"] = True
                return {
                    "status": "ok",
                    "reason": "first cycle complete",
                    "market_slug": "btc-updown-5m-test",
                    "orders": [],
                }
            raise _LoopExit()

    def fake_stop_status():
        return {
            "active": bool(stop_state["active"]),
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "path": str(tmp_path / "logs" / "btc5m_emergency_stop.json"),
        }

    def fake_sleep(_seconds):
        if stop_state["active"]:
            stop_state["active"] = False
            return None
        raise _LoopExit()

    monkeypatch.setattr(btc_5m, "Btc5mRunner", _FakeRunner)
    monkeypatch.setattr(btc_5m, "resolve_btc5m_profile", lambda profile_name: {"name": profile_name})
    monkeypatch.setattr(web_app, "get_clob_client", lambda: object())
    monkeypatch.setattr(web_app, "btc5m_emergency_stop_status", fake_stop_status)
    monkeypatch.setattr(
        web_app,
        "update_runtime_component",
        lambda *args, **kwargs: runtime_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(web_serve.time, "sleep", fake_sleep)

    with pytest.raises(_LoopExit):
        web_serve.run_btc5m_live_loop(
            profile_name="late_capture",
            ledger_path=tmp_path / "late_capture.json",
            poll_seconds=1,
            trading_mode="dry-run",
            startup_delay_seconds=0,
        )

    assert run_calls == [
        {"dry_run": True, "market_slug": None},
        {"dry_run": True, "market_slug": None},
    ]
    states = [args[1] for args, _kwargs in runtime_updates if args[0] == "btc5m_loop:late_capture"]
    assert "paused" in states
    assert states.count("running") >= 2


def test_live_betting_loop_does_not_auto_redeem(monkeypatch, tmp_path):
    from src import bot
    from src.data import line_tracker
    from src.polymarket import executor, tracker as polymarket_tracker
    from src.strategy import duo_trader

    class _LoopExit(Exception):
        pass

    sleep_calls = {"count": 0}

    def fake_sleep(_seconds):
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 2:
            raise _LoopExit()

    redeem_calls = []
    runtime_updates = []
    shared_clob = object()
    cleanup_clients = []
    monkeypatch.setattr(web_serve.time, "sleep", fake_sleep)
    monkeypatch.setattr(
        web_app,
        "update_runtime_component",
        lambda *args, **kwargs: runtime_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(web_app, "get_clob_client", lambda: shared_clob)
    monkeypatch.setattr(
        executor,
        "cancel_all_stale_limit_bids",
        lambda **kwargs: cleanup_clients.append(kwargs.get("clob_client")) or 0,
    )
    monkeypatch.setattr(line_tracker, "snapshot_odds", lambda: None)
    monkeypatch.setattr(line_tracker, "snapshot_polymarket_prices", lambda: None)

    def fake_cmd_duo_live(args):
        assert callable(getattr(args, "progress_callback", None))
        assert args.clob_client is shared_clob
        args.progress_callback("Cycle active: test progress callback")
        return {"status": "ok"}

    monkeypatch.setattr(bot, "cmd_duo_live", fake_cmd_duo_live)
    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", str(tmp_path / "single-ledger.json"))
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", str(tmp_path / "conviction-ledger.json"))
    monkeypatch.setattr(
        polymarket_tracker,
        "auto_redeem_positions_from_polymarket",
        lambda **kwargs: redeem_calls.append(kwargs),
    )

    first_cycle_complete = threading.Event()
    with pytest.raises(_LoopExit):
        web_serve.run_live_betting_loop(
            interval_minutes=0.01,
            trading_mode="dry-run",
            model_name="xgboost",
            first_cycle_complete_event=first_cycle_complete,
        )

    assert redeem_calls == []
    assert cleanup_clients == [shared_clob]
    assert first_cycle_complete.is_set()
    assert any(
        args[2] == "Cycle active: test progress callback"
        for args, _kwargs in runtime_updates
    )


def test_live_betting_loop_marks_degraded_cycles_as_failures(monkeypatch, tmp_path):
    from src import bot
    from src.data import line_tracker
    from src.polymarket import executor, tracker as polymarket_tracker
    from src.strategy import duo_trader

    class _LoopExit(Exception):
        pass

    cycles = {"count": 0}

    def fake_cmd_duo_live(args):
        cycles["count"] += 1
        return {"status": "degraded", "reason": "all tradeable fights skipped"}

    def fake_sleep(_seconds):
        if cycles["count"] >= 3:
            raise _LoopExit()

    runtime_updates = []
    monkeypatch.setattr(web_serve.time, "sleep", fake_sleep)
    monkeypatch.setattr(
        web_app,
        "update_runtime_component",
        lambda *args, **kwargs: runtime_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(executor, "cancel_all_stale_limit_bids", lambda **_kwargs: 0)
    monkeypatch.setattr(line_tracker, "snapshot_odds", lambda: None)
    monkeypatch.setattr(line_tracker, "snapshot_polymarket_prices", lambda: None)
    monkeypatch.setattr(bot, "cmd_duo_live", fake_cmd_duo_live)
    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", str(tmp_path / "single-ledger.json"))
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", str(tmp_path / "conviction-ledger.json"))
    monkeypatch.setattr(
        polymarket_tracker,
        "auto_redeem_positions_from_polymarket",
        lambda **kwargs: None,
    )

    with pytest.raises(_LoopExit):
        web_serve.run_live_betting_loop(
            interval_minutes=0.01,
            trading_mode="dry-run",
            model_name="xgboost",
        )

    assert cycles["count"] == 3
    failure_updates = [
        (args, kwargs)
        for args, kwargs in runtime_updates
        if len(args) >= 3 and "Live cycle reported degraded" in str(args[2])
    ]
    assert len(failure_updates) == 3
    assert [kwargs.get("consecutive_failures") for _args, kwargs in failure_updates] == [1, 2, 3]
    # Third consecutive degraded cycle flips the component state itself to degraded.
    assert failure_updates[-1][0][1] == "degraded"
    assert all(args[1] == "running" for args, _kwargs in failure_updates[:2])


def test_live_betting_loop_treats_refresh_hash_mismatch_as_pause(monkeypatch, tmp_path):
    from src import bot
    from src.data import line_tracker
    from src.polymarket import executor, tracker as polymarket_tracker
    from src.strategy import duo_trader

    class _LoopExit(Exception):
        pass

    cycles = {"count": 0}

    def fake_cmd_duo_live(_args):
        cycles["count"] += 1
        raise RuntimeError(
            "Production bundle processed fights snapshot hash mismatch: "
            "manifest expects old, artifact is new."
        )

    def fake_sleep(_seconds):
        if cycles["count"] >= 3:
            raise _LoopExit()

    runtime_updates = []
    monkeypatch.setattr(web_serve.time, "sleep", fake_sleep)
    monkeypatch.setattr(
        web_app,
        "update_runtime_component",
        lambda *args, **kwargs: runtime_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(web_serve, "_ufc_refresh_cycle_in_progress", lambda: True)
    monkeypatch.setattr(executor, "cancel_all_stale_limit_bids", lambda **_kwargs: 0)
    monkeypatch.setattr(line_tracker, "snapshot_odds", lambda: None)
    monkeypatch.setattr(line_tracker, "snapshot_polymarket_prices", lambda: None)
    monkeypatch.setattr(bot, "cmd_duo_live", fake_cmd_duo_live)
    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", str(tmp_path / "single-ledger.json"))
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", str(tmp_path / "conviction-ledger.json"))
    monkeypatch.setattr(
        polymarket_tracker,
        "auto_redeem_positions_from_polymarket",
        lambda **kwargs: None,
    )

    with pytest.raises(_LoopExit):
        web_serve.run_live_betting_loop(
            interval_minutes=0.01,
            trading_mode="dry-run",
            model_name="xgboost",
        )

    assert cycles["count"] == 3
    pause_updates = [
        (args, kwargs)
        for args, kwargs in runtime_updates
        if len(args) >= 3 and "Live trading paused while scheduled UFC refresh" in str(args[2])
    ]
    assert len(pause_updates) == 3
    assert all(args[1] == "running" for args, _kwargs in pause_updates)
    assert all(kwargs.get("consecutive_failures") == 0 for _args, kwargs in pause_updates)


def test_background_monitor_auto_redeem_uses_auto_source(monkeypatch, tmp_path):
    from src.data import line_tracker, live_monitor
    from src.polymarket import tracker as polymarket_tracker
    from src.strategy import duo_trader

    class _LoopExit(Exception):
        pass

    sleep_calls = {"count": 0}

    def fake_sleep(_seconds):
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 2:
            raise _LoopExit()

    redeem_calls = []
    monkeypatch.setenv("POLYMARKET_AUTO_REDEEM", "1")
    monkeypatch.setattr(web_serve.time, "sleep", fake_sleep)
    monkeypatch.setattr(web_app, "update_runtime_component", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(
        line_tracker,
        "run_line_tracking_pass",
        lambda progress_callback=None: {"sharp_moves": 0},
    )
    monkeypatch.setattr(live_monitor, "run_monitoring_pass", lambda: {"events": []})
    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", str(tmp_path / "single-ledger.json"))
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", str(tmp_path / "conviction-ledger.json"))
    monkeypatch.setattr(
        polymarket_tracker,
        "auto_redeem_positions_from_polymarket",
        lambda **kwargs: redeem_calls.append(kwargs) or {
            "submitted_conditions": 1,
            "submitted_positions": 2,
            "redeemed_conditions": 0,
            "redeemed_positions": 0,
            "errors": [],
            "reason": "",
        },
    )

    with pytest.raises(_LoopExit):
        web_serve.run_background_monitor(interval_hours=0.01)

    assert redeem_calls == [{"wait": False, "source": "auto"}]


def test_background_monitor_passes_heartbeat_callback_to_line_tracking(monkeypatch, tmp_path):
    from src.data import line_tracker, live_monitor
    from src.strategy import duo_trader

    class _LoopExit(Exception):
        pass

    sleep_calls = {"count": 0}

    def fake_sleep(_seconds):
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 2:
            raise _LoopExit()

    runtime_updates = []
    monkeypatch.setattr(web_serve.time, "sleep", fake_sleep)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(
        web_app,
        "update_runtime_component",
        lambda *args, **kwargs: runtime_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(live_monitor, "run_monitoring_pass", lambda: {"events": [{}, {}]})

    def fake_run_line_tracking_pass(progress_callback=None):
        assert callable(progress_callback)
        progress_callback("Line tracking: analyzing 1/2 (Alpha Fighter vs Beta Fighter)")
        return {"sharp_moves": 0}

    monkeypatch.setattr(line_tracker, "run_line_tracking_pass", fake_run_line_tracking_pass)
    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", str(tmp_path / "single-ledger.json"))
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", str(tmp_path / "conviction-ledger.json"))

    with pytest.raises(_LoopExit):
        web_serve.run_background_monitor(interval_hours=0.01)

    assert any(
        args[2] == "Line tracking: analyzing 1/2 (Alpha Fighter vs Beta Fighter)"
        for args, _kwargs in runtime_updates
    )


def test_background_monitor_reports_method_odds_fallback_in_runtime(monkeypatch, tmp_path):
    from src.data import line_tracker, live_monitor
    from src.strategy import duo_trader

    class _LoopExit(Exception):
        pass

    sleep_calls = {"count": 0}

    def fake_sleep(_seconds):
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 2:
            raise _LoopExit()

    runtime_updates = []
    monkeypatch.setattr(web_serve.time, "sleep", fake_sleep)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(
        web_app,
        "update_runtime_component",
        lambda *args, **kwargs: runtime_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        live_monitor,
        "run_monitoring_pass",
        lambda: {
            "events": [{}],
            "method_odds_snapshot": {
                "status": "failed",
                "record_count": 0,
                "snapshot_time": "2026-06-28T20:22:40",
                "latest_usable_snapshot": {
                    "snapshot_time": "2026-06-28T17:11:27",
                    "snapshot_path": "/app/logs/raw/method_odds/method_odds_20260628_171127.json",
                    "record_count": 1,
                    "is_stale": False,
                },
            },
        },
    )
    monkeypatch.setattr(
        line_tracker,
        "run_line_tracking_pass",
        lambda progress_callback=None: {"sharp_moves": 0},
    )
    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", str(tmp_path / "single-ledger.json"))
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", str(tmp_path / "conviction-ledger.json"))

    with pytest.raises(_LoopExit):
        web_serve.run_background_monitor(interval_hours=0.01)

    fallback_updates = [
        (args, kwargs)
        for args, kwargs in runtime_updates
        if kwargs.get("method_odds_effective_status") == "fresh_fallback"
    ]
    assert len(fallback_updates) == 1
    args, kwargs = fallback_updates[0]
    assert args[0] == "monitor_loop"
    assert args[1] == "running"
    assert "latest usable snapshot is fresh" in args[2]
    assert "latest usable snapshot is fresh" in kwargs["method_odds_status_message"]
    assert kwargs["method_odds_snapshot"]["latest_usable_snapshot"]["record_count"] == 1


def test_method_odds_runtime_metadata_reports_unpublished_props_without_failure():
    suffix, metadata = web_serve._method_odds_runtime_metadata(
        {
            "method_odds_snapshot": {
                "status": "unavailable",
                "record_count": 0,
                "availability_expected": False,
            }
        }
    )

    assert "not currently published" in suffix
    assert metadata["method_odds_effective_status"] == "unavailable"
    assert "refresh failed" not in metadata["method_odds_status_message"].lower()


def test_method_odds_runtime_metadata_reports_partial_coverage_without_failure():
    suffix, metadata = web_serve._method_odds_runtime_metadata(
        {
            "method_odds_snapshot": {
                "status": "partial",
                "record_count": 12,
                "tracked_fight_count": 13,
                "covered_fight_count": 12,
                "expected_fight_count": 13,
                "expected_covered_fight_count": 12,
            }
        }
    )

    assert "partial" in suffix
    assert metadata["method_odds_effective_status"] == "partial"
    assert "12/13" in metadata["method_odds_status_message"]
    assert "failed" not in metadata["method_odds_status_message"].lower()


def test_cached_deduplicates_concurrent_compute_calls():
    compute_calls = {"count": 0}
    barrier = threading.Barrier(3)
    results = []

    def compute():
        compute_calls["count"] += 1
        time.sleep(0.05)
        return {"value": 42}

    def worker():
        barrier.wait()
        results.append(web_app._cached("shared-key", 60, compute))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert compute_calls["count"] == 1
    assert results == [{"value": 42}, {"value": 42}, {"value": 42}]


def test_stale_while_revalidate_serves_stale_when_refresh_thread_cannot_start(monkeypatch, caplog):
    class _FailingThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    compute_calls = {"count": 0}

    def compute():
        compute_calls["count"] += 1
        return {"value": "fresh"}

    web_app._endpoint_cache["stale-key"] = {"data": {"value": "stale"}, "ts": 0}
    monkeypatch.setattr(web_app.threading, "Thread", _FailingThread)
    caplog.set_level("WARNING", logger="src.web.app")

    payload, source = web_app._cached_stale_while_revalidate("stale-key", 1, compute)

    assert payload == {"value": "stale"}
    assert source == "stale"
    assert compute_calls["count"] == 0
    assert "stale-key" not in web_app._endpoint_inflight
    assert web_app._background_cache_refreshes == 0
    assert any("Could not start background cache refresh for stale-key" in record.getMessage() for record in caplog.records)


def test_call_with_timeout_returns_none_when_worker_thread_cannot_start(monkeypatch, caplog):
    class _FailingThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(web_app.threading, "Thread", _FailingThread)
    caplog.set_level("WARNING", logger="src.web.app")

    result = web_app._call_with_timeout("checking Polymarket geoblock status", lambda: {"ok": True}, 1)

    assert result is None
    assert web_app._timed_call_inflight == {}
    assert any(
        "Could not start worker while checking Polymarket geoblock status" in record.getMessage()
        for record in caplog.records
    )


def test_secret_cache_fragment_does_not_expose_proxy_credentials():
    raw_proxy = "http://proxyuser:super-secret-token@163.176.191.39:3128"

    fragment = web_app._cache_key_secret_fragment(raw_proxy)

    assert fragment
    assert fragment != raw_proxy
    assert "proxyuser" not in fragment
    assert "super-secret-token" not in fragment
    assert "163.176.191.39" not in fragment


def test_limit_order_reconcile_skips_cleanly_when_worker_thread_cannot_start(monkeypatch, caplog):
    class _FailingThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    reconcile_calls = {"count": 0}

    def fake_reconcile(**_kwargs):
        reconcile_calls["count"] += 1
        return {"reconciled": 1}

    monkeypatch.setattr(web_app.threading, "Thread", _FailingThread)
    monkeypatch.setattr(web_app, "_reconcile_limit_orders_with_clob", fake_reconcile)
    caplog.set_level("WARNING", logger="src.web.app")

    web_app._kickoff_limit_order_reconcile(open_order_ids={"order-1"}, ttl_seconds=0)

    assert reconcile_calls["count"] == 0
    assert "limit-order-clob-reconcile" not in web_app._endpoint_inflight
    assert web_app._background_cache_refreshes == 0
    assert any(
        "Could not start limit order reconciliation worker" in record.getMessage()
        for record in caplog.records
    )


def test_api_injury_alerts_reads_precomputed_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    web_app.write_market_intel_artifact({
        "timestamp": "2026-06-04T18:00:00+00:00",
        "injury_alerts": [{"fighter_a": "A", "fighter_b": "B", "severity": "warning"}],
        "line_movements": [{"fighter_a": "A", "fighter_b": "B", "abs_movement": 0.1}],
        "fights_analyzed": 1,
    })
    client = web_app.app.test_client()

    injury_response = client.get("/api/injury-alerts")

    assert injury_response.status_code == 200
    payload = injury_response.get_json()
    assert payload["status"] == "current"
    assert payload["alerts"] == [{"fighter_a": "A", "fighter_b": "B", "severity": "warning"}]
    assert payload["line_movements"] == [{"fighter_a": "A", "fighter_b": "B", "abs_movement": 0.1}]


def test_api_injury_alerts_missing_artifact_returns_fast_status(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/injury-alerts")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "missing"
    assert payload["alerts"] == []
    assert "not completed" in payload["message"]


def test_api_bets_exposes_clv_fields(monkeypatch):
    monkeypatch.setattr(
        web_app,
        "load_all_trader_ledgers",
        lambda: ReadOnlyBetLedgerView.from_bets(
            [
                {
                    "id": 1,
                    "fighter": "Alpha",
                    "opponent": "Beta",
                    "status": "won",
                    "amount": 10.0,
                    "price": 0.5,
                    "shares": 20.0,
                    "model_prob": 0.6,
                    "market_prob": 0.5,
                    "edge": 0.1,
                    "decimal_odds": 2.0,
                    "clv": 0.04,
                    "closing_prob": 0.48,
                    "clv_captured_at": "2026-03-25T20:00:00+00:00",
                    "_ledger_path": "bet_ledger_single.json",
                }
            ]
        ),
    )
    client = web_app.app.test_client()

    response = client.get("/api/bets")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["all"][0]["clv"] == pytest.approx(0.04)
    assert payload["all"][0]["closing_prob"] == pytest.approx(0.48)
    assert payload["all"][0]["clv_captured_at"] == "2026-03-25T20:00:00+00:00"
