from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import requests

from src import trading_worker
from src.polymarket import readonly_account


def _account(*, positions=(), orders=(), cash=100):
    client = SimpleNamespace(
        get_cash_balance_details=lambda **kw: {"source": "clob", "balance": cash},
        get_positions=lambda: list(positions), get_open_orders=lambda: list(orders),
    )
    return readonly_account.inspect_account(client)


def test_account_authoritative_sizing_deducts_shared_open_buy_reserve():
    account = _account(positions=[{"asset": "token", "size": 4, "currentValue": 20}], orders=[
        {"id": "existing", "side": "BUY", "price": .5, "original_size": 40, "size_matched": 10},
        {"id": "sell", "side": "SELL"},
    ])
    assert account.total_equity == 120
    assert account.available_cash == 85
    assert account.summary["position_count"] == 1
    assert account.summary["open_order_count"] == 2
    assert account.summary["empty_account_proved"] is False


@pytest.mark.parametrize("cash", [{"source": "unavailable", "balance": 0},
                                  {"source": "onchain_v2_collateral", "balance": 100},
                                  {"source": "clob", "balance": float("nan")}])
def test_account_unavailable_or_fallback_never_means_empty(cash):
    client = SimpleNamespace(get_cash_balance_details=lambda **kw: cash)
    with pytest.raises(readonly_account.AccountReadUnavailable):
        readonly_account.inspect_account(client)


@pytest.mark.parametrize("stage, method", [
    ("cash", "get_cash_balance_details"),
    ("positions", "get_positions"),
    ("open_orders", "get_open_orders"),
])
def test_account_read_failure_identifies_stage_and_wrapped_http_status(stage, method):
    response = requests.Response()
    response.status_code = 429
    vendor_error = requests.HTTPError("https://vendor.invalid/?token=secret", response=response)
    def unavailable(*args, **kwargs):
        raise RuntimeError("vendor response contains secret") from vendor_error
    client = SimpleNamespace(get_cash_balance_details=lambda **kw: {"source": "clob", "balance": 100},
                             get_positions=lambda: [], get_open_orders=lambda: [])
    setattr(client, method, unavailable)
    with pytest.raises(readonly_account.AccountReadUnavailable) as raised:
        readonly_account.inspect_account(client)
    details = readonly_account.account_read_diagnostics(raised.value)
    assert details == {"read_stage": stage, "error_class": "RuntimeError", "http_status": 429}
    assert "secret" not in json.dumps(details)


@pytest.mark.parametrize("stage, value", [
    ("cash", {"source": "unavailable", "balance": 0}),
    ("positions", [{"currentValue": float("nan")}]),
    ("open_orders", None),
])
def test_unconfirmed_account_values_retain_stage_without_inventing_http_status(stage, value):
    client = SimpleNamespace(
        get_cash_balance_details=lambda **kw: value if stage == "cash" else {"source": "clob", "balance": 100},
        get_positions=lambda: value if stage == "positions" else [],
        get_open_orders=lambda: value if stage == "open_orders" else [],
    )
    with pytest.raises(readonly_account.AccountReadUnavailable) as raised:
        readonly_account.inspect_account(client)
    details = readonly_account.account_read_diagnostics(raised.value)
    assert details["read_stage"] == stage and details["http_status"] is None


@pytest.mark.parametrize("status", [None, True, 99, 600, 503.5, "secret", "503?token=secret"])
def test_account_diagnostics_reject_invalid_status_and_cyclic_causes(status):
    error = RuntimeError("private vendor details")
    error.status_code = status
    error.__cause__ = error
    assert readonly_account.account_read_diagnostics(error) == {
        "read_stage": "unknown", "error_class": "RuntimeError", "http_status": None}


def test_mutating_client_methods_and_sdk_are_denied_before_network():
    client = readonly_account.ReadOnlyClobClient(private_key="fixture")
    for name in ("create_limit_order", "create_market_order", "cancel_order", "cancel_all_orders",
                 "redeem_positions", "redeem_redeemable_positions", "create_api_key", "post_order"):
        with pytest.raises(readonly_account.AccountWritesDisabled):
            getattr(client, name)
    with pytest.raises(readonly_account.AccountWritesDisabled):
        client._relayer_request("POST", "/submit")
    sdk = readonly_account._ReadOnlySdk(SimpleNamespace(get_order=lambda _: {"status": "LIVE"}))
    assert sdk.get_order("fixture") == {"status": "LIVE"}
    with pytest.raises(readonly_account.AccountWritesDisabled):
        sdk.create_api_key()


def test_derive_existing_key_has_no_create_fallback():
    client = readonly_account.ReadOnlyClobClient(private_key="fixture")
    sdk = SimpleNamespace(derive_api_key=lambda: (_ for _ in ()).throw(ValueError("not found")),
                          create_api_key=lambda: pytest.fail("key creation forbidden"))
    with pytest.raises(ValueError):
        client._derive_or_create_api_key(sdk)


def test_actual_client_initialization_wraps_sdk_after_derive_only(monkeypatch):
    import py_clob_client_v2.client as sdk_module
    events = []
    class SDK:
        def __init__(self, *args, **kwargs):
            pass
        def derive_api_key(self):
            events.append("derive_existing")
            return SimpleNamespace(api_key="fixture", api_secret="fixture", api_passphrase="fixture")
        def create_api_key(self):
            pytest.fail("credential provision forbidden")
        def set_api_creds(self, credentials):
            events.append("set_local_credentials")
        def get_order(self, order_id):
            return {"id": order_id, "status": "LIVE"}
        def post_order(self, *args):
            pytest.fail("order POST forbidden")
    monkeypatch.setattr(sdk_module, "ClobClient", SDK)
    client = readonly_account.ReadOnlyClobClient(private_key="fixture", funder_address="fixture")
    monkeypatch.setattr(client, "_configure_shared_transport", lambda: None)
    client._ensure_client()
    assert events == ["derive_existing", "set_local_credentials"]
    assert client.get_order("saved-original") == {"id": "saved-original", "status": "LIVE"}
    with pytest.raises(readonly_account.AccountWritesDisabled):
        client._client.post_order("forbidden")


def test_positions_pagination_proves_all_pages(monkeypatch):
    client = readonly_account.ReadOnlyClobClient(private_key="fixture", funder_address="fixture")
    monkeypatch.setattr(client, "_ensure_client", lambda: None)
    calls = []
    def request(url, **kwargs):
        calls.append(kwargs)
        offset = kwargs["params"]["offset"]
        return ([{"asset": "A", "size": 1, "currentValue": 2},
                 {"asset": "B", "size": 1, "currentValue": 3}] if offset == 0 else
                [{"asset": "C", "size": 1, "currentValue": 4}])
    monkeypatch.setattr(readonly_account, "request_json", request)
    assert len(client.get_positions(limit=2)) == 3
    assert [row["params"]["offset"] for row in calls] == [0, 2]
    assert all(row["attempts"] == 1 for row in calls)


@pytest.mark.parametrize("payload", [None, {}, [{"asset": "A", "size": 1}],
                                    [{"asset": "A", "size": float("nan"), "currentValue": 1}]])
def test_positions_malformed_is_unknown(monkeypatch, payload):
    client = readonly_account.ReadOnlyClobClient(private_key="fixture", funder_address="fixture")
    monkeypatch.setattr(client, "_ensure_client", lambda: None)
    monkeypatch.setattr(readonly_account, "request_json", lambda *args, **kw: payload)
    with pytest.raises(readonly_account.AccountReadUnavailable):
        client.get_positions()


def test_original_accounting_preserves_bytes_and_uncertainty(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    path = original / "bet_ledger_single.json"
    raw = json.dumps({"bets": [
        {"id": 12, "dry_run": False, "status": "open", "token_id": "owned",
         "order_id": "original-order", "placement_state": "unknown"},
        {"id": 13, "dry_run": True, "status": "open", "token_id": "paper"},
    ]}).encode()
    path.write_bytes(raw)
    before = path.stat().st_mtime_ns
    account = _account(positions=[{"asset": "owned", "currentValue": 4},
                                  {"asset": "manual", "currentValue": 8}], orders=[
        {"id": "original-order", "side": "SELL"}, {"id": "manual-order", "side": "SELL"}])
    result = readonly_account.inspect_original_accounting(original, account)
    assert result["real_rows"] == 1
    assert result["uncertain_original_intents_preserved"] == 1
    assert result["positions_without_original_ledger_match"] == 1
    assert result["open_orders_without_original_ledger_match"] == 1
    assert result["complete_ownership_reconciliation"] is False
    assert result["files"][1]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert path.read_bytes() == raw and path.stat().st_mtime_ns == before
    assert sorted(item.name for item in original.iterdir()) == [path.name]


def test_corrupt_original_accounting_does_not_reset_or_write(tmp_path):
    path = tmp_path / "bet_ledger_single.json"
    path.write_bytes(b"not-json")
    with pytest.raises(readonly_account.AccountReadUnavailable):
        readonly_account.inspect_original_accounting(tmp_path)
    assert path.read_bytes() == b"not-json"


def _worker(tmp_path, *, account_reader=None, cycle_runner=None):
    original = tmp_path / "original"
    original.mkdir(exist_ok=True)
    state = original / "paper"
    return trading_worker.TradingWorker(
        root=tmp_path / "refresh", baseline=tmp_path / "baseline",
        state_dir=state, real_logs_dir=original, client=object(),
        account_reader=account_reader or (lambda _: _account()),
        cycle_runner=cycle_runner or (lambda args: {"status": "ok", "orders_disabled": True,
            "trading_enabled": False, "accepted_generation": "good", "prediction_count": 1,
            "matched_market_count": 1, "predictions": [
                {"fighter_a": "A", "prob_a": .7, "private_key": "forbidden"}]}),
    )


def test_worker_real_consumer_call_is_orders_off_and_state_atomic(tmp_path):
    observed = []
    def cycle(args):
        observed.append(args)
        args.progress_callback("vendor secret should not reach status")
        return {"status": "ok", "orders_disabled": True, "trading_enabled": False,
                "accepted_generation": "published-final", "prediction_count": 13}
    worker = _worker(tmp_path, cycle_runner=cycle)
    report = worker.run_once()
    assert observed[0].dry_run is True
    assert observed[0].refresh_root == tmp_path / "refresh"
    assert observed[0].readonly_clob is worker.client
    assert observed[0].bankroll_basis.total_equity == 100
    assert report["real_submissions"] == 0
    assert not report["blocking_conditions"]
    status = json.loads((worker.state_dir / "status.json").read_text())
    assert status["ready"] is True and status["pid"] == os.getpid()
    assert datetime.fromisoformat(status["updated_at_utc"]).tzinfo is not None
    assert json.loads((worker.state_dir / "latest.json").read_text()) == report
    assert json.loads((worker.state_dir / "report.json").read_text()) == report
    assert len(list((worker.state_dir / "cycles").glob("*.json"))) == 1


def test_account_failure_runs_monitor_with_zero_basis_and_degraded(tmp_path):
    observed = []
    def failed(_):
        raise RuntimeError("credential secret must not be recorded")
    worker = _worker(tmp_path, account_reader=failed, cycle_runner=lambda args:
                     observed.append(args) or {"status": "idle", "prediction_count": 13,
                        "accepted_generation": "good", "orders_disabled": True, "trading_enabled": False})
    report = worker.run_once()
    assert observed[0].bankroll_basis.total_equity == 0
    assert observed[0].bankroll_basis.available_cash == 0
    assert report["account"]["connected"] is False
    assert report["blocking_conditions"] == ["account_read_unavailable"]
    assert json.loads((worker.state_dir / "status.json").read_text())["ready"] is False
    assert "credential secret" not in (worker.state_dir / "latest.json").read_text()


@pytest.mark.parametrize("real", [False, True])
def test_account_http_failure_logs_safe_diagnostics_and_next_cycle_recovers(monkeypatch, tmp_path, caplog, real):
    from src import live_control
    monkeypatch.setenv("LIVE_TRADING_MODE", "real" if real else "off")
    monkeypatch.setattr(live_control, "assert_real_trading_allowed", lambda **kw: {"ready": True})
    response = requests.Response()
    response.status_code = 503
    response.url = "https://vendor.invalid/positions?user=private-wallet"
    response._content = b"private-response-body"
    def failed_positions():
        raise requests.HTTPError("vendor-secret " + response.url, response=response)
    client = SimpleNamespace(get_cash_balance_details=lambda **kw: {"source": "clob", "balance": 100},
                             get_positions=failed_positions, get_open_orders=lambda: [])
    seen = []
    def cycle(args):
        seen.append(args)
        return {"status": "evaluated", "orders_disabled": not real, "trading_enabled": real,
                "accepted_generation": "final", "real_submissions": 0}
    worker = _worker(tmp_path, account_reader=readonly_account.inspect_account, cycle_runner=cycle)
    worker.client = client
    with caplog.at_level(logging.WARNING, logger="src.trading_worker"):
        report = worker.run_once()
    assert "Account reads unavailable (stage=positions, error=HTTPError, http_status=503)" in caplog.text
    assert "Orders-disabled" not in caplog.text
    assert report["account"]["read_stage"] == "positions"
    assert report["account"]["http_status"] == 503
    assert report["account"]["error_class"] == "HTTPError"
    assert report["account"]["read_only"] is True
    assert not report["trading_enabled"] and not report["accounting_reconciled"]
    assert "account_read_unavailable" in report["blocking_conditions"]
    assert json.loads((worker.state_dir / "status.json").read_text())["ready"] is False
    assert len(seen) == (0 if real else 1)
    if not real:
        assert seen[0].bankroll_basis.available_cash == seen[0].bankroll_basis.total_equity == 0
    retained = caplog.text + "".join(path.read_text() for path in worker.state_dir.rglob("*.json"))
    for private in ("vendor-secret", "private-wallet", "vendor.invalid", "private-response-body"):
        assert private not in retained
    client.get_positions = lambda: []
    recovered = worker.run_once()
    assert not recovered["blocking_conditions"] and recovered["accounting_reconciled"]
    assert recovered["trading_enabled"] is real
    assert json.loads((worker.state_dir / "status.json").read_text())["ready"] is True


def test_worker_market_failure_is_durable_and_next_cycle_recovers(tmp_path):
    worker = _worker(tmp_path, cycle_runner=lambda _: (_ for _ in ()).throw(RuntimeError("unavailable")))
    first = worker.run_once()
    assert "market_or_prediction_cycle_unavailable" in first["blocking_conditions"]
    worker.cycle_runner = lambda _: {"status": "idle", "orders_disabled": True,
                                   "trading_enabled": False, "predictions": [], "accepted_generation": "good"}
    second = worker.run_once()
    assert not second["blocking_conditions"]
    assert first["cycle_id"] != second["cycle_id"]
    assert len(list((worker.state_dir / "cycles").glob("*.json"))) == 2


def test_worker_failure_stage_is_safe_and_repeat_warnings_resume_after_recovery(tmp_path, caplog):
    from src import refresh_live
    snapshot = SimpleNamespace(snapshot={"models": {"primary": {"bytes": 1, "sha256": "fixture"}}})
    def failed(_):
        refresh_live._assert_selected_live_model(snapshot, str(tmp_path / "private-model-path.pkl"))
    worker = _worker(tmp_path, cycle_runner=failed)
    with caplog.at_level(logging.WARNING, logger="src.trading_worker"):
        first = worker.run_once()
        second = worker.run_once()
        worker.cycle_runner = lambda _: {"status": "idle", "orders_disabled": True,
            "trading_enabled": False, "accepted_generation": "good"}
        recovered = worker.run_once()
        worker.cycle_runner = failed
        worker.run_once()
    assert first["cycle"]["failure_stage"] == "selected_live_model"
    assert second["cycle"]["error_class"] == "RuntimeError"
    assert caplog.text.count("Trading evaluation unavailable (stage=selected_live_model, error=RuntimeError)") == 2
    assert not recovered["blocking_conditions"]
    assert len(list((worker.state_dir / "cycles").glob("*.json"))) == 4
    retained = caplog.text + "".join(path.read_text() for path in worker.state_dir.rglob("*.json"))
    assert "private-model-path" not in retained
    assert "LIVE_MODEL does not match" not in retained


def test_worker_original_order_block_preserves_observations_without_claiming_ready(monkeypatch, tmp_path, caplog):
    from src import live_control
    monkeypatch.setenv("LIVE_TRADING_MODE", "real")
    monkeypatch.setattr(live_control, "assert_real_trading_allowed", lambda **kw: {"ready": True})
    observed = {"status": "blocked", "blocking_conditions": ["original_order_unresolved"],
                "orders_disabled": False, "trading_enabled": False, "accepted_generation": "final",
                "real_submissions": 0, "total_orders": 0, "prediction_count": 2,
                "bookmaker_observed_at": "2026-10-20T20:00:00+00:00",
                "market_observed_at": "2026-10-20T20:00:00+00:00",
                "mapped_markets": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
                "predictions": [{"fighter_a": "Alpha", "prob_a": .6}]}
    worker = _worker(tmp_path, cycle_runner=lambda _: observed)
    with caplog.at_level(logging.WARNING, logger="src.trading_worker"):
        report = worker.run_once()
        worker.run_once()
    assert report["blocking_conditions"] == ["original_order_unresolved"]
    assert report["cycle"]["blocking_conditions"] == ["original_order_unresolved"]
    assert report["cycle"]["prediction_count"] == 2
    assert report["cycle"]["mapped_markets"] == observed["mapped_markets"]
    assert report["cycle"]["market_observed_at"] == observed["market_observed_at"]
    assert not report["trading_enabled"] and not report["orders_disabled"]
    assert not report["accounting_reconciled"] and report["real_submissions"] == 0
    status = json.loads((worker.state_dir / "status.json").read_text())
    assert status["status"] == "blocked" and not status["ready"] and not status["trading_enabled"]
    assert caplog.text.count("Trading evaluation blocked (original_order_unresolved)") == 1
    worker.cycle_runner = lambda _: {**observed, "status": "idle", "blocking_conditions": [], "trading_enabled": True}
    recovered = worker.run_once()
    assert not recovered["blocking_conditions"] and recovered["trading_enabled"]
    assert json.loads((worker.state_dir / "status.json").read_text())["ready"] is True


@pytest.mark.parametrize("blockers", [["private-vendor-query"], [], "original_order_unresolved"])
def test_worker_unknown_block_contract_never_claims_ready(tmp_path, blockers):
    worker = _worker(tmp_path, cycle_runner=lambda _: {"status": "blocked", "blocking_conditions": blockers,
        "orders_disabled": True, "trading_enabled": False, "accepted_generation": "final"})
    report = worker.run_once()
    assert "cycle_blocker_unconfirmed" in report["blocking_conditions"]
    assert json.loads((worker.state_dir / "status.json").read_text())["ready"] is False
    assert "private-vendor-query" not in json.dumps(report)


def test_worker_receipts_survive_new_instance_without_reusing_cycle_id(tmp_path):
    first = _worker(tmp_path).run_once()
    second = _worker(tmp_path).run_once()
    assert first["cycle_id"] != second["cycle_id"]
    assert len(list((tmp_path / "original" / "paper" / "cycles").glob("*.json"))) == 2
    assert "forbidden" not in json.dumps(second)


def test_worker_unconfirmed_consumer_contract_cannot_claim_ready(tmp_path):
    report = _worker(tmp_path, cycle_runner=lambda _: {}).run_once()
    assert "orders_disabled_result_contract_unconfirmed" in report["blocking_conditions"]
    status = json.loads((tmp_path / "original" / "paper" / "status.json").read_text())
    assert status["ready"] is False


def test_worker_rejects_real_mode_before_reads(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVE_TRADING_MODE", "real")
    worker = _worker(tmp_path, account_reader=lambda _: pytest.fail("read before off gate"))
    with pytest.raises(RuntimeError, match="Real-money live trading blocked"):
        worker.run_once()


@pytest.mark.parametrize("cadence", [float("nan"), float("inf"), 0, -1])
def test_worker_invalid_poll_rejected_before_reads(tmp_path, cadence):
    worker = _worker(tmp_path, account_reader=lambda _: pytest.fail("read before poll validation"))
    with pytest.raises(ValueError, match="finite"):
        worker.run(poll_seconds=cadence)


def test_worker_lifetime_lock_interrupt_releases_and_second_owner_refused(tmp_path):
    from src.refresh_live import trading_consumer_lock, TradingConsumerBusy
    worker = _worker(tmp_path)
    lock = worker.state_dir / "trading-worker.lock"
    with trading_consumer_lock(lock):
        with pytest.raises(TradingConsumerBusy):
            worker.run(once=True)
    worker.cycle_runner = lambda _: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        worker.run(once=True)
    with trading_consumer_lock(lock):
        pass
    assert json.loads((worker.state_dir / "status.json").read_text())["status"] == "stopped"


def test_environment_config_separates_paper_before_import(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVE_TRADING_MODE", "off")
    original, state = tmp_path / "original", tmp_path / "original" / "paper"
    trading_worker.configure_environment(state, tmp_path / "refresh", tmp_path / "baseline", original)
    assert Path(os.environ["UFC_LOGS_DIR"]) == (state / "logs").resolve()
    assert Path(os.environ["UFC_DATA_DIR"]) == (state / "data").resolve()
    assert os.environ["LIVE_TRADING_ARMED"] == "0"
    assert os.environ["LIVE_TRADING_CONFIRMATION"] == ""
    monkeypatch.setenv("LIVE_TRADING_MODE", "real")
    monkeypatch.setenv("UFC_LOGS_DIR", str(original))
    monkeypatch.setenv("UFC_DATA_DIR", str(original))
    monkeypatch.setenv("LIVE_TRADING_ARMED", "1")
    monkeypatch.setenv("LIVE_TRADING_CONFIRMATION", "REAL_TRADING_ENABLED")
    trading_worker.configure_environment(state, tmp_path / "refresh", tmp_path / "baseline", original)
    assert Path(os.environ["UFC_LOGS_DIR"]) == original.resolve()
    assert Path(os.environ["UFC_DATA_DIR"]) == original.resolve()
    assert os.environ["LIVE_TRADING_ARMED"] == "1"
    assert os.environ["LIVE_TRADING_CONFIRMATION"] == "REAL_TRADING_ENABLED"


def test_real_worker_calls_real_consumer_only_after_complete_account(monkeypatch, tmp_path):
    from src import live_control
    monkeypatch.setenv("LIVE_TRADING_MODE", "real")
    monkeypatch.setattr(live_control, "assert_real_trading_allowed", lambda **kw: {"ready": True})
    seen = []
    worker = _worker(tmp_path, cycle_runner=lambda args: seen.append(args) or {
        "status": "evaluated", "orders_disabled": False, "trading_enabled": True,
        "accepted_generation": "final", "real_submissions": 0})
    report = worker.run_once()
    assert seen[0].dry_run is False
    assert report["startup_verified"] and report["accounting_reconciled"]
    assert report["trading_enabled"] and not report["orders_disabled"]
    assert report["real_submissions"] == 0
    worker.account_reader = lambda _: (_ for _ in ()).throw(ValueError("unavailable"))
    second = worker.run_once()
    assert len(seen) == 1
    assert not second["trading_enabled"] and not second["accounting_reconciled"]
    assert "account_read_unavailable" in second["blocking_conditions"]


def test_real_worker_preserves_unattributed_exposure_and_denies_execution(monkeypatch, tmp_path):
    from src import live_control
    monkeypatch.setenv("LIVE_TRADING_MODE", "real")
    monkeypatch.setattr(live_control, "assert_real_trading_allowed", lambda **kw: {"ready": True})
    worker = _worker(tmp_path, account_reader=lambda _: _account(positions=[
        {"asset": "manual", "size": 10, "currentValue": 8}]),
        cycle_runner=lambda _: pytest.fail("must not execute unattributed exposure"))
    report = worker.run_once()
    assert "unattributed_current_position_exposure" in report["blocking_conditions"]
    assert not report["trading_enabled"]
    assert not list(worker.real_logs_dir.glob("*ledger*.json"))


def test_durable_real_pause_stops_consumer_and_bad_record_fails_closed(monkeypatch, tmp_path):
    from src import live_control, trading_controls
    monkeypatch.setenv("LIVE_TRADING_MODE", "real")
    monkeypatch.setattr(live_control, "assert_real_trading_allowed", lambda **kw: {"ready": True})
    worker = _worker(tmp_path, cycle_runner=lambda _: pytest.fail("paused consumer called"))
    trading_controls.set_paused(worker.state_dir, True)
    report = worker.run_once()
    assert report["paused"] and not report["trading_enabled"]
    assert not report["blocking_conditions"]
    (worker.state_dir / "control.json").write_text("torn")
    assert trading_controls.paused_status(worker.state_dir)["paused"]
    monkeypatch.setenv("UFC_REFRESH_TRADING_STATE_DIR", str(worker.state_dir))
    with pytest.raises(RuntimeError, match="paused"):
        trading_controls.require_unpaused()


def test_all_original_ledgers_include_trader_b_and_do_not_adopt_zero_position(tmp_path):
    path = tmp_path / "bet_ledger_trader_b.json"
    raw = json.dumps({"bets": [{"dry_run": False, "status": "open", "token_id": "old-B",
                               "order_id": "original-B"}]}).encode()
    path.write_bytes(raw)
    account = _account(positions=[{"asset": "external-lost", "size": 24, "currentValue": 0,
                                  "redeemable": True, "curPrice": 0}])
    result = readonly_account.inspect_original_accounting(tmp_path, account)
    assert result["by_owner"]["B"]["real_rows"] == 1
    assert result["zero_value_positions_not_adopted"] == 1
    assert not result["activation_blockers"]
    assert path.read_bytes() == raw


def test_unresolved_zero_quote_is_unattributed_risk(tmp_path):
    account = _account(positions=[{"asset": "unresolved", "size": 24, "currentValue": 0,
                                  "redeemable": False, "curPrice": 0}])
    result = readonly_account.inspect_original_accounting(tmp_path, account)
    assert result["activation_blockers"] == ["unattributed_current_position_exposure"]
    assert result["zero_value_positions_not_adopted"] == 0


def test_environment_rejects_source_state_writes(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVE_TRADING_MODE", "off")
    with pytest.raises(RuntimeError, match="overlap"):
        trading_worker.configure_environment(tmp_path / "refresh" / "paper", tmp_path / "refresh",
                                            tmp_path / "baseline", tmp_path / "original")


def test_parent_guard_rejects_missing_parent_before_any_network(monkeypatch):
    monkeypatch.setenv("UFC_REFRESH_SUPERVISOR_PID", "999999999")
    with pytest.raises(RuntimeError, match="no longer"):
        trading_worker.install_parent_death_guard()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux kernel parent-death signal")
def test_linux_supervisor_sigkill_terminates_guarded_child(tmp_path):
    ready, terminated = tmp_path / "child-ready", tmp_path / "child-terminated"
    child = (
        "from pathlib import Path\nimport os,signal,time,sys\n"
        "from src.trading_worker import install_parent_death_guard\n"
        "def stop(*_):\n Path(sys.argv[2]).write_text('terminated')\n sys.exit(0)\n"
        "signal.signal(signal.SIGTERM,stop)\ninstall_parent_death_guard()\n"
        "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "while True: time.sleep(.05)\n"
    )
    parent = (
        "import os,subprocess,sys,time\n"
        "env=dict(os.environ,UFC_REFRESH_SUPERVISOR_PID=str(os.getpid()))\n"
        "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]],env=env)\n"
        "while True: time.sleep(.05)\n"
    )
    process = subprocess.Popen([sys.executable, "-c", parent, child, str(ready), str(terminated)])
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists(), "Guarded test child did not finish startup"
        process.kill()
        process.wait(timeout=5)
        deadline = time.monotonic() + 5
        while not terminated.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert terminated.read_text() == "terminated"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if ready.exists() and not terminated.exists():
            try:
                os.kill(int(ready.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_log_filter_redacts_env_secret_address_and_traceback(monkeypatch):
    monkeypatch.setenv("WEB_DASHBOARD_TOKEN", "fixture-secret")
    record = logging.LogRecord("fixture", logging.ERROR, __file__, 1,
                               "credential %s at %s", ("fixture-secret", "0x" + "a" * 40),
                               (ValueError, ValueError("fixture-secret"), None))
    assert trading_worker._SecretFilter().filter(record)
    assert "fixture-secret" not in record.getMessage() and "0x" not in record.getMessage()
    assert record.exc_info is None


def test_log_filter_redacts_derived_api_credentials_and_query_string():
    record = logging.LogRecord("fixture", logging.ERROR, __file__, 1,
                               "derived-key derived-secret https://example.invalid/path?apiKey=other", (), None)
    assert trading_worker._SecretFilter(lambda: ("derived-key", "derived-secret")).filter(record)
    assert "derived-key" not in record.getMessage() and "derived-secret" not in record.getMessage()
    assert "apiKey=other" not in record.getMessage()
