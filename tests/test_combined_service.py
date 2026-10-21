from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import threading

import pytest

from src import combined_service
from src import live_control
from src.data.refresh_service import create_app


@pytest.fixture(autouse=True)
def clean_combined_controls(monkeypatch):
    for name in ("LIVE_TRADING_MODE", "LIVE_TRADING_ARMED", "LIVE_TRADING_CONFIRMATION", "LIVE_MODEL",
                 "BTC5M_LIVE_PROFILES", "UFC_MONTHLY_CANDIDATE_ENABLED", "UFC_V7_PROMOTION_MONTH_UTC"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def real_startup(tmp_path, monkeypatch):
    logs = tmp_path / "original"
    models = tmp_path / "models"
    logs.mkdir()
    models.mkdir()
    for name in ("xgboost_model.pkl", "xgboost_no_odds_model.pkl"):
        (models / name).write_bytes(b"startup-existence-fixture")
    monkeypatch.setattr(live_control, "LOGS_DIR", logs)
    monkeypatch.setattr(live_control, "MODELS_DIR", models)
    monkeypatch.setattr("src.config.LOGS_DIR", logs)
    for name, value in {"LIVE_TRADING_MODE": "real", "LIVE_MODEL": "xgboost",
                        "LIVE_TRADING_ARMED": "1", "LIVE_TRADING_CONFIRMATION": "REAL_TRADING_ENABLED",
                        "ODDS_API_KEY": "fixture-odds", "POLYMARKET_PRIVATE_KEY": "fixture-private",
                        "WEB_DASHBOARD_TOKEN": "fixture-dashboard", "UFC_LOGS_DIR": str(logs),
                        "UFC_DATA_DIR": str(tmp_path / "original-data")}.items():
        monkeypatch.setenv(name, value)
    return logs, models


class Process:
    pid = 12345

    def __init__(self, code=None):
        self.code = code
        self.terminated = False
        self.waited = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = 0

    def wait(self, **kwargs):
        self.waited = True
        return self.code


class Service:
    def __init__(self, tmp_path):
        self.root = tmp_path / "refresh"
        self.baseline = tmp_path / "accepted"
        self.root.mkdir()
        self.baseline.mkdir()
        self.published_path = self.root / "published.json"
        self.published_path.write_text("{}")
        self.status_path = self.root / "status.json"
        self.status_path.write_text("{}")
        self.ticked = threading.Event()
        self.tick_threads = []

    def published_pair(self):
        return ({"generation_id": "accepted"}, {"total_rows": 9503},
                {"accepted_generation": "accepted", "prediction_count": 13, "predictions": []})

    def current_snapshot(self):
        return {"generation_id": "accepted"}

    def tick(self):
        self.tick_threads.append(threading.current_thread().name)
        self.ticked.set()


@pytest.fixture
def process_state(tmp_path):
    path = tmp_path / "paper"
    path.mkdir()
    saved = {"ready": True, "pid": Process.pid, "orders_disabled": True,
             "trading_enabled": False, "updated_at_utc": datetime.now(timezone.utc).isoformat()}
    (path / "status.json").write_text(json.dumps(saved))
    (path / "latest.json").write_text(json.dumps({"predictions": [{"fighter_a": "A", "fighter_b": "B"}]}))
    return path, saved


@pytest.mark.parametrize("mode", ["dry-run", "invalid", "OFF", "live"])
def test_combined_rejects_unsupported_mode_before_creating_workers(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("LIVE_TRADING_MODE", mode)
    monkeypatch.setattr(combined_service.subprocess, "Popen", lambda *a, **kw: pytest.fail("consumer started"))
    with pytest.raises(RuntimeError, match="LIVE_TRADING_MODE=off"):
        combined_service.serve(Service(tmp_path), state_dir=tmp_path / "paper", host="127.0.0.1")
    assert not (tmp_path / "paper").exists()


@pytest.mark.parametrize("token", ["", "   ", "\n\t"])
def test_public_combined_requires_nonblank_dashboard_token(monkeypatch, token):
    monkeypatch.setenv("LIVE_TRADING_MODE", "off")
    monkeypatch.setenv("WEB_DASHBOARD_TOKEN", token)
    with pytest.raises(RuntimeError, match="WEB_DASHBOARD_TOKEN"):
        combined_service.require_combined_startup(host="0.0.0.0")
    combined_service.require_combined_startup(host="127.0.0.1")


@pytest.mark.parametrize("name,value,expected", [
    ("LIVE_TRADING_ARMED", "0", "LIVE_TRADING_ARMED"),
    ("LIVE_TRADING_CONFIRMATION", "enabled", "LIVE_TRADING_CONFIRMATION"),
    ("ODDS_API_KEY", "", "ODDS_API_KEY"),
    ("POLYMARKET_PRIVATE_KEY", "", "POLYMARKET_PRIVATE_KEY"),
    ("LIVE_MODEL", "", "LIVE_MODEL"),
    ("LIVE_MODEL", "unknown", "Missing required model"),
    ("BTC5M_LIVE_PROFILES", "profile", "BTC5M_LIVE_PROFILES"),
    ("UFC_MONTHLY_CANDIDATE_ENABLED", "1", "Monthly candidate"),
    ("UFC_V7_PROMOTION_MONTH_UTC", "2026-10", "automatic promotion"),
])
def test_real_startup_obeys_existing_gates_before_starting_any_worker(real_startup, tmp_path, monkeypatch,
                                                                    name, value, expected):
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(combined_service.subprocess, "Popen", lambda *a, **kw: pytest.fail("consumer started"))
    with pytest.raises(RuntimeError, match=expected):
        combined_service.serve(Service(tmp_path), state_dir=tmp_path / "live", host="0.0.0.0")
    assert not (tmp_path / "live").exists()


def test_real_startup_rejects_hard_disabled_mode_even_when_control_flags_are_set(real_startup, monkeypatch):
    monkeypatch.setattr(live_control, "LIVE_TRADING_DISABLED", True)
    with pytest.raises(RuntimeError, match="not enabled"):
        combined_service.require_combined_startup(host="0.0.0.0")


def test_real_startup_does_not_require_funding_or_contact_exchange(real_startup):
    status = combined_service.require_combined_startup(host="0.0.0.0")
    assert status["ready"] and status["trading_enabled"]
    assert status["effective_live_mode"] == "real"


def test_real_consumer_environment_retains_arming_original_accounting_and_data(real_startup, tmp_path):
    original, _models = real_startup
    live = tmp_path / "live"
    child = combined_service.child_environment(live, mode="real", real_logs_dir=original)
    assert child["LIVE_TRADING_MODE"] == "real"
    assert child["LIVE_TRADING_ARMED"] == "1"
    assert child["LIVE_TRADING_CONFIRMATION"] == "REAL_TRADING_ENABLED"
    assert child["LIVE_MODEL"] == "xgboost"
    assert Path(child["UFC_LOGS_DIR"]) == original
    assert Path(child["UFC_DATA_DIR"]) == tmp_path / "original-data"
    assert Path(child["UFC_REFRESH_TRADING_STATE_DIR"]) == live


def test_consumer_environment_is_unarmed_and_keeps_original_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVE_TRADING_MODE", "real")
    monkeypatch.setenv("LIVE_TRADING_ARMED", "1")
    monkeypatch.setenv("LIVE_TRADING_CONFIRMATION", "enabled")
    monkeypatch.setenv("UFC_LOGS_DIR", str(tmp_path / "original"))
    monkeypatch.setenv("UFC_DATA_DIR", str(tmp_path / "original-data"))
    child = combined_service.child_environment(tmp_path / "paper")
    assert child["LIVE_TRADING_MODE"] == "off"
    assert child["LIVE_TRADING_ARMED"] == "0"
    assert child["LIVE_TRADING_CONFIRMATION"] == ""
    assert Path(child["UFC_LOGS_DIR"]) == tmp_path / "paper" / "logs"
    assert Path(child["UFC_DATA_DIR"]) == tmp_path / "paper" / "data"
    assert child["UFC_REFRESH_SUPERVISOR_PID"] == str(os.getpid())
    assert os.environ["LIVE_TRADING_MODE"] == "real"
    assert os.environ["UFC_LOGS_DIR"] == str(tmp_path / "original")


def test_lifetime_lock_excludes_duplicate_supervisor_and_releases_after_interruption(tmp_path):
    path = tmp_path / "owner.lock"
    with pytest.raises(SystemExit):
        with combined_service.supervisor_lock(path):
            with pytest.raises(RuntimeError, match="already owns"):
                with combined_service.supervisor_lock(path):
                    pytest.fail("duplicate owner acquired lease")
            raise SystemExit("interrupted")
    with combined_service.supervisor_lock(path):
        pass


@pytest.mark.parametrize("change", ["stale", "future", "pid", "not_ready", "armed", "naive", "dead"])
def test_readiness_requires_current_ready_status_from_actual_orders_off_owner(process_state, change):
    path, saved = process_state
    process = Process()
    if change == "stale":
        saved["updated_at_utc"] = (datetime.now(timezone.utc) - timedelta(minutes=11)).isoformat()
    elif change == "future":
        saved["updated_at_utc"] = (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat()
    elif change == "pid":
        saved["pid"] += 1
    elif change == "not_ready":
        saved["ready"] = False
    elif change == "armed":
        saved["orders_disabled"] = False
    elif change == "naive":
        saved["updated_at_utc"] = datetime.now().isoformat()
    else:
        process.code = 1
    (path / "status.json").write_text(json.dumps(saved))
    with pytest.raises(RuntimeError):
        combined_service.TradingProcess(process, path).readiness()


def test_existing_auth_protects_current_trading_report_and_status_and_combined_readiness(tmp_path, process_state, monkeypatch):
    monkeypatch.setenv("WEB_DASHBOARD_TOKEN", "fixture-access")
    path, saved = process_state
    process = Process()
    client = create_app(Service(tmp_path), trading=combined_service.TradingProcess(process, path)).test_client()
    for endpoint in ("/api/trading-status", "/api/trading-predictions"):
        assert client.get(endpoint).status_code == 401
        assert client.get(endpoint, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get(endpoint, headers={"Authorization": "Bearer fixture-access"}).status_code == 200
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json["components"]["trading_consumer"]["orders_disabled"] is True
    saved["ready"] = False
    (path / "status.json").write_text(json.dumps(saved))
    assert client.get("/readyz").status_code == 503
    assert client.get("/healthz").status_code == 200
    process.code = 1
    assert client.get("/readyz").status_code == 503


@pytest.mark.parametrize("missing", ["startup_verified", "accounting_reconciled", "effective_live_mode",
                                      "orders_disabled", "trading_enabled"])
def test_real_readiness_requires_live_gates_and_original_accounting_confirmation(process_state, missing):
    path, saved = process_state
    saved.update(orders_disabled=False, trading_enabled=True, effective_live_mode="real",
                 startup_verified=True, accounting_reconciled=True)
    saved.pop(missing)
    (path / "status.json").write_text(json.dumps(saved))
    with pytest.raises(RuntimeError):
        combined_service.TradingProcess(Process(), path, mode="real").readiness()


def test_authenticated_pause_resume_is_durable_and_live_health_reports_actual_mode(process_state, tmp_path, monkeypatch):
    path, saved = process_state
    saved.update(orders_disabled=False, trading_enabled=True, effective_live_mode="real",
                 startup_verified=True, accounting_reconciled=True)
    (path / "status.json").write_text(json.dumps(saved))
    monkeypatch.setenv("WEB_DASHBOARD_TOKEN", "fixture-access")
    trading = combined_service.TradingProcess(Process(), path, mode="real")
    client = create_app(Service(tmp_path), trading=trading).test_client()
    headers = {"Authorization": "Bearer fixture-access"}
    for endpoint in ("/healthz", "/readyz"):
        current = client.get(endpoint)
        assert current.status_code == 200
        assert current.json["trading_enabled"] is True
        assert current.json["effective_live_mode"] == "real"
    # Runtime mode alone does not establish current observations. This fixture
    # has no coherent cycle, so the page must retain its availability warning.
    assert b"status needs attention" in client.get("/predictions", headers=headers).data
    assert client.post("/api/trading-control", json={"paused": True}).status_code == 401
    assert client.post("/api/trading-control", json={"paused": True},
                       headers={"Authorization": "Bearer wrong"}).status_code == 401
    for bad in ({}, {"paused": "false"}, {"paused": 0}, {"paused": True, "arm": True}, []):
        assert client.post("/api/trading-control", json=bad, headers=headers).status_code == 400
    paused = client.post("/api/trading-control", json={"paused": True}, headers=headers)
    assert paused.status_code == 200 and paused.json["paused"] is True
    assert json.loads((path / "control.json").read_text())["paused"] is True
    restarted_view = combined_service.TradingProcess(Process(), path, mode="real")
    assert restarted_view.readiness()["paused"] is True
    assert restarted_view.readiness()["trading_enabled"] is False
    assert client.get("/readyz").status_code == 200
    assert client.get("/healthz").json["trading_enabled"] is False
    assert b"trading is paused" in client.get("/predictions", headers=headers).data
    resumed = client.post("/api/trading-control", json={"paused": False}, headers=headers)
    assert resumed.status_code == 200 and resumed.json["paused"] is False
    assert restarted_view.readiness()["trading_enabled"] is True
    assert os.environ.get("LIVE_TRADING_MODE") is None


def test_original_order_block_without_observations_does_not_pass_release_readiness(process_state, tmp_path, monkeypatch):
    path, saved = process_state
    saved.update(ready=False, status="blocked", orders_disabled=False, trading_enabled=False,
                 effective_live_mode="real", startup_verified=True, accounting_reconciled=False,
                 blocking_conditions=["original_order_unresolved"])
    (path / "status.json").write_text(json.dumps(saved))
    monkeypatch.setenv("WEB_DASHBOARD_TOKEN", "fixture-access")
    trading = combined_service.TradingProcess(Process(), path, mode="real")
    client = create_app(Service(tmp_path), trading=trading).test_client()
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 503
    assert client.get("/healthz").json["trading_enabled"] is False
    page = client.get("/predictions", headers={"Authorization": "Bearer fixture-access"})
    assert b"new orders are blocked pending recovery" in page.data


@pytest.fixture
def current_blocked_consumer(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv("WEB_DASHBOARD_TOKEN", "fixture-access")
    now = datetime.now(timezone.utc)
    started = (now - timedelta(minutes=2)).isoformat()
    observed = (now - timedelta(seconds=15)).isoformat()
    completed = (now - timedelta(seconds=5)).isoformat()
    blocker = ["original_order_unresolved"]
    state_dir = tmp_path / "consumer"
    state_dir.mkdir()
    status = {"status": "blocked", "ready": False, "pid": Process.pid,
              "orders_disabled": False, "trading_enabled": False, "effective_live_mode": "real",
              "startup_verified": True, "accounting_reconciled": False, "paused": False,
              "blocking_conditions": list(blocker), "cycle_in_progress": False,
              "cycle_id": "current-cycle", "started_at_utc": started,
              "last_completed_at_utc": completed, "updated_at_utc": now.isoformat()}
    report = {"orders_disabled": False, "trading_enabled": False, "effective_live_mode": "real",
              "startup_verified": True, "accounting_reconciled": False, "paused": False,
              "blocking_conditions": list(blocker), "real_submissions": 0,
              "worker_pid": Process.pid, "worker_started_at_utc": started,
              "cycle_id": "current-cycle", "completed_at_utc": completed,
              "cycle": {"status": "blocked", "blocking_conditions": list(blocker),
                        "orders_disabled": False, "trading_enabled": False, "total_orders": 0,
                        "confirmed_placed_orders": 0, "accepted_generation": "accepted",
                        "publication_job_id": "final-job", "snapshot_binding": {"final_report_sha256": "f" * 64},
                        "bookmaker_observed_at": observed, "market_observed_at": observed}}
    publication = {"generation_id": "accepted", "job_id": "final-job",
                   "prediction_report": {"sha256": "f" * 64}}
    service = Service(tmp_path)
    service.published_pair = lambda: (publication, {"total_rows": 9503},
        {"accepted_generation": "accepted", "prediction_count": 13, "predictions": []})
    process = Process()
    trading = combined_service.TradingProcess(process, state_dir, mode="real")
    def save():
        (state_dir / "status.json").write_text(json.dumps(status))
        (state_dir / "latest.json").write_text(json.dumps(report))
    save()
    return SimpleNamespace(status=status, report=report, publication=publication, service=service,
                           process=process, trading=trading, save=save, now=now)


def test_current_order_block_is_ready_to_serve_but_never_ready_to_trade(current_blocked_consumer):
    view = current_blocked_consumer
    with pytest.raises(RuntimeError, match="not ready"):
        view.trading.readiness()
    service_ready = view.trading.service_readiness(view.publication)
    assert service_ready["ready"] is True and service_ready["trading_ready"] is False
    assert service_ready["trading_enabled"] is False and service_ready["orders_disabled"] is False
    assert service_ready["blocking_conditions"] == ["original_order_unresolved"]
    client = create_app(view.service, trading=view.trading).test_client()
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json["ready"] is True and response.json["trading_ready"] is False
    assert response.json["trading_enabled"] is False
    assert response.json["effective_live_mode"] == "real" and response.json["orders_disabled"] is False
    assert response.json["components"]["trading_consumer"]["blocking_conditions"] == ["original_order_unresolved"]
    assert view.trading.status()["ready"] is False


@pytest.mark.parametrize("source,key,value", [
    ("status", "pid", Process.pid + 1),
    ("status", "status", "degraded"),
    ("status", "status", "evaluating"),
    ("status", "cycle_in_progress", True),
    ("status", "ready", True),
    ("status", "startup_verified", False),
    ("status", "accounting_reconciled", True),
    ("status", "blocking_conditions", ["original_order_unresolved", "account_read_unavailable"]),
    ("report", "worker_pid", Process.pid + 1),
    ("report", "worker_started_at_utc", "2020-01-01T00:00:00+00:00"),
    ("report", "cycle_id", "previous-cycle"),
    ("report", "startup_verified", False),
    ("report", "accounting_reconciled", True),
    ("report", "effective_live_mode", "off"),
    ("report", "orders_disabled", True),
    ("report", "trading_enabled", True),
    ("report", "blocking_conditions", []),
    ("report", "real_submissions", None),
    ("report", "real_submissions", False),
    ("report", "real_submissions", 1),
    ("cycle", "status", "degraded"),
    ("cycle", "blocking_conditions", ["unknown_blocker"]),
    ("cycle", "trading_enabled", True),
    ("cycle", "orders_disabled", True),
    ("cycle", "total_orders", False),
    ("cycle", "total_orders", 1),
    ("cycle", "confirmed_placed_orders", 1),
    ("cycle", "accepted_generation", "previous-generation"),
    ("cycle", "publication_job_id", "previous-job"),
    ("cycle", "snapshot_binding", {"final_report_sha256": "a" * 64}),
])
def test_service_readiness_rejects_unverified_or_conflicting_blocked_state(current_blocked_consumer,
                                                                          source, key, value):
    view = current_blocked_consumer
    target = {"status": view.status, "report": view.report, "cycle": view.report["cycle"]}[source]
    target[key] = value
    view.save()
    with pytest.raises(RuntimeError):
        view.trading.service_readiness(view.publication)


@pytest.mark.parametrize("damage", ["dead", "stale_heartbeat", "future_heartbeat", "naive_heartbeat",
                                    "stale_observation", "future_observation", "naive_observation",
                                    "missing_observation", "invalid_observation",
                                    "previous_worker_observation", "observation_after_completion",
                                    "mismatched_completion", "incomplete_publication"])
def test_service_readiness_rejects_dead_stale_or_unbound_observations(current_blocked_consumer, damage):
    view = current_blocked_consumer
    if damage == "dead":
        view.process.code = 1
    elif damage in {"stale_heartbeat", "future_heartbeat", "naive_heartbeat"}:
        stamp = view.now + timedelta(minutes=2 if damage == "future_heartbeat" else -11)
        view.status["updated_at_utc"] = (stamp.replace(tzinfo=None) if damage == "naive_heartbeat" else stamp).isoformat()
    elif damage in {"stale_observation", "future_observation", "naive_observation"}:
        stamp = view.now + timedelta(minutes=2 if damage == "future_observation" else -11)
        view.report["cycle"]["market_observed_at"] = (
            stamp.replace(tzinfo=None) if damage == "naive_observation" else stamp).isoformat()
    elif damage == "missing_observation":
        view.report["cycle"].pop("market_observed_at")
    elif damage == "invalid_observation":
        view.report["cycle"]["market_observed_at"] = "private-invalid-time"
    elif damage == "previous_worker_observation":
        restarted = view.now.isoformat()
        view.status["started_at_utc"] = view.report["worker_started_at_utc"] = restarted
    elif damage == "observation_after_completion":
        view.report["cycle"]["bookmaker_observed_at"] = view.now.isoformat()
    elif damage == "mismatched_completion":
        view.status["last_completed_at_utc"] = view.now.isoformat()
    else:
        view.publication.pop("prediction_report")
    view.save()
    client = create_app(view.service, trading=view.trading).test_client()
    response = client.get("/readyz")
    assert response.status_code == 503
    assert "private-invalid-time" not in response.get_data(as_text=True)


def test_corrupt_pause_record_blocks_orders_and_deceased_consumer_reports_disabled(process_state):
    path, saved = process_state
    saved.update(orders_disabled=False, trading_enabled=True, effective_live_mode="real",
                 startup_verified=True, accounting_reconciled=True)
    (path / "status.json").write_text(json.dumps(saved))
    (path / "control.json").write_text('{"paused": "invalid"}')
    trading = combined_service.TradingProcess(Process(), path, mode="real")
    assert trading.readiness()["paused"] is True
    assert trading.readiness()["trading_enabled"] is False
    trading.process.code = 9
    assert trading.status()["trading_enabled"] is False
    with pytest.raises(RuntimeError, match="not running"):
        trading.readiness()


@pytest.mark.parametrize("mode", ["off", "real"])
def test_one_refresh_thread_and_consumer_are_shutdown_together_with_original_accounting_root(tmp_path, monkeypatch,
                                                                                           real_startup, mode):
    monkeypatch.setenv("LIVE_TRADING_MODE", mode)
    monkeypatch.setenv("UFC_LOGS_DIR", str(tmp_path / "original"))
    service = Service(tmp_path)
    stop = threading.Event()
    process = Process()
    starts = []

    def start(command, **kwargs):
        starts.append((command, kwargs))
        return process

    class Server:
        closed = False

        def handle_request(self):
            assert service.ticked.wait(timeout=2)
            stop.set()

        def server_close(self):
            self.closed = True

    server = Server()
    monkeypatch.setattr(combined_service.subprocess, "Popen", start)
    monkeypatch.setattr("werkzeug.serving.make_server", lambda *a, **kw: server)
    state = tmp_path / ("paper" if mode == "off" else "live") / "nested"
    combined_service.serve(service, state_dir=state, host="127.0.0.1", stop=stop)
    assert len(starts) == 1
    command, options = starts[0]
    assert command[1:3] == ["-m", "src.trading_worker"]
    assert command[command.index("--real-logs-dir") + 1] == str(tmp_path / "original")
    assert options["env"]["UFC_LOGS_DIR"] == str(state / "logs" if mode == "off" else tmp_path / "original")
    assert options["env"]["LIVE_TRADING_MODE"] == mode
    if mode == "real":
        assert not (state / "logs").exists()
        assert not (state / "data").exists()
        assert options["env"]["LIVE_TRADING_ARMED"] == "1"
    assert service.tick_threads == ["ufc-refresh-calendar"]
    assert process.terminated and process.waited and server.closed
    assert not any(t.name == "ufc-refresh-calendar" for t in threading.enumerate())
    with combined_service.supervisor_lock(service.root / "combined-supervisor.lock"):
        pass


def test_bind_failure_terminates_child_and_releases_supervisor_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_MODE", "off")
    service = Service(tmp_path)
    process = Process()
    monkeypatch.setattr(combined_service.subprocess, "Popen", lambda *a, **kw: process)

    def unavailable(*args, **kwargs):
        raise OSError("address already in use")

    monkeypatch.setattr("werkzeug.serving.make_server", unavailable)
    with pytest.raises(OSError, match="address already"):
        combined_service.serve(service, state_dir=tmp_path / "paper", host="127.0.0.1")
    assert process.terminated and process.waited
    with combined_service.supervisor_lock(service.root / "combined-supervisor.lock"):
        pass


def test_dead_consumer_stops_supervisor_without_starting_replacement(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_MODE", "off")
    service = Service(tmp_path)
    process = Process(code=9)
    starts = []
    monkeypatch.setattr(combined_service.subprocess, "Popen", lambda *a, **kw: starts.append(a) or process)

    class Server:
        def handle_request(self):
            pytest.fail("accepting after consumer exit")

        def server_close(self):
            pass

    monkeypatch.setattr("werkzeug.serving.make_server", lambda *a, **kw: Server())
    with pytest.raises(RuntimeError, match="consumer exited"):
        combined_service.serve(service, state_dir=tmp_path / "paper", host="127.0.0.1")
    assert len(starts) == 1


@pytest.mark.parametrize("which", ["refresh", "baseline"])
def test_overlapping_paper_root_is_rejected_before_child_start(tmp_path, monkeypatch, which):
    monkeypatch.setenv("LIVE_TRADING_MODE", "off")
    service = Service(tmp_path)
    monkeypatch.setattr(combined_service.subprocess, "Popen", lambda *a, **kw: pytest.fail("consumer started"))
    paper = service.root if which == "refresh" else service.baseline / "paper"
    with pytest.raises(RuntimeError, match="separate roots"):
        combined_service.serve(service, state_dir=paper, host="127.0.0.1")


@pytest.mark.skipif(os.name == "nt", reason="Linux entrypoint and symlink permissions")
def test_paper_log_symlink_cannot_redirect_writes_into_original_accounting(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_MODE", "off")
    service = Service(tmp_path)
    original = tmp_path / "original"
    original.mkdir()
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "logs").symlink_to(original, target_is_directory=True)
    monkeypatch.setattr(combined_service.subprocess, "Popen", lambda *a, **kw: pytest.fail("consumer started"))
    with pytest.raises(RuntimeError, match="inside the paper root"):
        combined_service.serve(service, state_dir=paper, host="127.0.0.1")
    assert not list(original.iterdir())


@pytest.mark.skipif(os.name == "nt", reason="Linux shell entrypoint")
@pytest.mark.parametrize("settings,expected", [
    ({"WEB_DASHBOARD_TOKEN": ""}, "WEB_DASHBOARD_TOKEN is required"),
    ({"WEB_DASHBOARD_TOKEN": "fixture-only", "LIVE_TRADING_MODE": "invalid"}, "LIVE_TRADING_MODE=off or real"),
    ({"WEB_DASHBOARD_TOKEN": "fixture-only", "LIVE_TRADING_MODE": "real"}, "mode-specific persistent namespace"),
    ({"WEB_DASHBOARD_TOKEN": "fixture-only", "UFC_REFRESH_TRADING_STATE_DIR": "/tmp/escaped"}, "Consumer state must be"),
    ({"WEB_DASHBOARD_TOKEN": "fixture-only", "UFC_INCREMENTAL_REFRESH_ROOT": "/tmp/escaped"}, "Refresh state must be"),
    ({"WEB_DASHBOARD_TOKEN": "fixture-only", "PORT": "99999"}, "port must be an integer"),
])
def test_entrypoint_rejects_unsafe_configuration_before_seed_or_workers(settings, expected):
    environment = {**os.environ, "LIVE_TRADING_MODE": "off", "UFC_INCREMENTAL_REFRESH_ROOT": "/app/logs/incremental_refresh",
                   "UFC_REFRESH_TRADING_STATE_DIR": "/app/logs/refresh_trading_paper", "PORT": "8080", **settings}
    entrypoint = Path(__file__).resolve().parents[1] / "refresh-entrypoint.sh"
    result = subprocess.run(["/bin/bash", str(entrypoint)], env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert expected in result.stderr
    assert "fixture-only" not in result.stdout + result.stderr
