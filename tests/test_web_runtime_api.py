import threading
import time

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
    web_app._endpoint_cache.clear()
    web_app._endpoint_inflight.clear()
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
    monkeypatch.setattr(web_app, "update_runtime_component", lambda *args: components.append(args))
    monkeypatch.setattr(web_app, "set_clob_client", lambda client: None)
    monkeypatch.setattr(web_app, "start_server", lambda **kwargs: start_calls.append(kwargs))

    web_serve.main()

    assert any(thread.target == web_serve.run_background_monitor for thread in threads)
    assert not any(thread.target == web_serve.run_live_betting_loop for thread in threads)
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
    monkeypatch.setattr(web_app, "update_runtime_component", lambda *args: None)
    monkeypatch.setattr(web_app, "set_clob_client", lambda client: None)
    monkeypatch.setattr(web_app, "start_server", lambda **kwargs: None)

    web_serve.main()

    betting_threads = [thread for thread in threads if thread.target == web_serve.run_live_betting_loop]
    assert len(betting_threads) == 1
    assert betting_threads[0].kwargs["trading_mode"] == "dry-run"


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
    monkeypatch.setattr(web_serve.time, "sleep", fake_sleep)
    monkeypatch.setattr(
        web_app,
        "update_runtime_component",
        lambda *args, **kwargs: runtime_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(executor, "cancel_all_stale_limit_bids", lambda: 0)
    monkeypatch.setattr(line_tracker, "snapshot_odds", lambda: None)
    monkeypatch.setattr(line_tracker, "snapshot_polymarket_prices", lambda: None)

    def fake_cmd_duo_live(args):
        assert callable(getattr(args, "progress_callback", None))
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

    with pytest.raises(_LoopExit):
        web_serve.run_live_betting_loop(
            interval_minutes=0.01,
            trading_mode="dry-run",
            model_name="xgboost",
        )

    assert redeem_calls == []
    assert any(
        args[2] == "Cycle active: test progress callback"
        for args, _kwargs in runtime_updates
    )


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
