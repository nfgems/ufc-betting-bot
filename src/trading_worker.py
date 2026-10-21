"""One guarded UFC consumer with durable cycle reports and original real ledgers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import sys
import tempfile
import threading
from types import SimpleNamespace
import uuid

logger = logging.getLogger(__name__)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, allow_nan=False, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def configure_environment(state_dir, root, baseline, real_logs_dir):
    """Configure namespace before any module imports environment-backed paths."""
    mode = str(os.environ.get("LIVE_TRADING_MODE", "off"))
    if mode not in {"off", "real"}:
        raise RuntimeError("Combined worker requires exact off or real trading mode")
    state = Path(state_dir).resolve()
    original = Path(real_logs_dir).resolve()
    if state == original:
        raise RuntimeError("Paper state must be separate from original accounting")
    if mode == "real":
        if state == original or not state.is_relative_to(original):
            raise RuntimeError("Real cycle reports require a separate original-volume namespace")
        for protected in (Path(root).resolve(), Path(baseline).resolve()):
            if state == protected or state.is_relative_to(protected) or protected.is_relative_to(state):
                raise RuntimeError("Real cycle reports overlap accepted state")
        if Path(os.environ.get("UFC_LOGS_DIR", original)).resolve() != original:
            raise RuntimeError("Real trading must use original accounting")
        os.environ.update(UFC_LOGS_DIR=str(original),
                          UFC_INCREMENTAL_REFRESH_ROOT=str(Path(root).resolve()),
                          UFC_INCREMENTAL_REFRESH_BASELINE=str(Path(baseline).resolve()),
                          UFC_REFRESH_TRADING_STATE_DIR=str(state))
        state.mkdir(parents=True, exist_ok=True)
        return
    writable = ((state / "logs").resolve(), (state / "data").resolve())
    protected = (Path(root).resolve(), Path(baseline).resolve(), original)
    if any(path == target or path in target.parents for path in writable for target in protected):
        raise RuntimeError("Paper writable namespaces overlap protected state")
    if any(target in path.parents for path in writable for target in protected[:2]):
        raise RuntimeError("Paper writable namespaces overlap protected state")
    os.environ.update(LIVE_TRADING_MODE="off", LIVE_TRADING_ARMED="0",
                      LIVE_TRADING_CONFIRMATION="", UFC_LOGS_DIR=str(writable[0]),
                      UFC_DATA_DIR=str(writable[1]),
                      UFC_INCREMENTAL_REFRESH_ROOT=str(Path(root).resolve()),
                      UFC_INCREMENTAL_REFRESH_BASELINE=str(Path(baseline).resolve()))
    for path in writable:
        path.mkdir(parents=True, exist_ok=True)


def install_parent_death_guard():
    """A Linux supervisor crash must release the trading owner immediately."""
    raw = os.environ.get("UFC_REFRESH_SUPERVISOR_PID")
    if not raw:
        return None
    expected = int(raw)
    if expected <= 0 or os.getppid() != expected:
        raise RuntimeError("Trading worker supervisor is no longer its parent")
    if sys.platform.startswith("linux"):
        import ctypes
        library = ctypes.CDLL(None, use_errno=True)
        if library.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise RuntimeError("Trading worker could not install its parent-death guard")
        # Close the race where the parent exited between getppid and prctl.
        if os.getppid() != expected:
            raise RuntimeError("Trading worker supervisor exited during startup")
    return expected


class _SecretFilter(logging.Filter):
    def __init__(self, secret_supplier=None):
        super().__init__()
        self.secret_supplier = secret_supplier

    def filter(self, record):
        message = record.getMessage()
        for key, value in os.environ.items():
            if value and any(part in key.upper() for part in
                             ("SECRET", "TOKEN", "PRIVATE_KEY", "API_KEY", "PASSWORD", "PROXY_URL")):
                message = message.replace(value, "[redacted]")
        if self.secret_supplier is not None:
            for value in self.secret_supplier():
                if isinstance(value, str) and value:
                    message = message.replace(value, "[redacted]")
        message = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[redacted-query]", message)
        message = re.sub(r"0x[a-fA-F0-9]{40,64}", "[redacted-address]", message)
        record.msg, record.args = message, ()
        # Traceback exception text may include request URLs/authorization.
        if record.exc_info:
            record.exc_text = f"Exception class: {record.exc_info[0].__name__}"
            record.exc_info = None
        return True


def _safe_predictions(rows):
    allowed = {
        "fighter_a", "fighter_b", "event_date", "official_event_date", "prob_a", "prob_b",
        "no_odds_prob_a", "no_odds_prob_b", "a_market_prob", "b_market_prob",
        "a_fair_prob_avg", "b_fair_prob_avg", "trade_blocked", "trade_block_reason",
        "market_inputs_available", "market_observed_at", "primary_model_available",
        "bet_window",
        "model_contract", "prediction_timestamp", "runtime_signature", "prediction_source",
    }
    return [{key: value for key, value in row.items() if key in allowed} for row in rows]


def _evaluation_failure_stage(exc):
    """Classify trusted call sites without retaining paths or exception text."""
    stages = {
        ("src.refresh_live", "_run_cycle"): "refresh_cycle",
        ("src.refresh_live", "_assert_selected_live_model"): "selected_live_model",
        ("src.refresh_live", "_maintain_original_accounting"): "original_accounting",
        ("src.data.refresh_trading", "read"): "published_snapshot",
        ("src.data.refresh_trading", "score"): "prediction",
        ("src.data.refresh_live_observations", "observe_publication"): "official_card_observation",
        ("src.data.odds_client", "get_live_odds"): "bookmaker_observation",
        ("src.polymarket.markets", "get_ufc_fight_markets"): "market_observation",
        ("src.strategy.duo_trader", "run_duo_traders"): "portfolio_execution",
    }
    stage, trace = "evaluation", exc.__traceback__
    while trace is not None:
        key = (trace.tb_frame.f_globals.get("__name__"), trace.tb_frame.f_code.co_name)
        stage = stages.get(key, stage)
        trace = trace.tb_next
    return stage


class TradingWorker:
    def __init__(self, *, root, baseline, state_dir, real_logs_dir, client=None,
                 cycle_runner=None, account_reader=None, accounting_reader=None):
        self.root, self.baseline = Path(root), Path(baseline)
        self.state_dir, self.real_logs_dir = Path(state_dir), Path(real_logs_dir)
        self.client = client
        self.cycle_runner, self.account_reader = cycle_runner, account_reader
        self.accounting_reader = accounting_reader
        self.stop_event = threading.Event()
        self.cycle_count = 0
        self.started_at = _now()
        self.last_completed = None
        self.supervisor_pid = None
        self._last_evaluation_failure = None

    def _status(self, *, status, ready=False, **extra):
        real = os.environ.get("LIVE_TRADING_MODE", "off") == "real"
        from src.trading_controls import paused_status
        paused = paused_status(self.state_dir).get("paused", False) if real else False
        payload = {"schema_version": 1, "pid": os.getpid(), "status": status,
                   "ready": ready, "orders_disabled": not real,
                   "trading_enabled": real and ready and not paused,
                   "effective_live_mode": "real" if real else "off", "paused": paused,
                   "started_at_utc": self.started_at, "updated_at_utc": _now(),
                   "cycle_count": self.cycle_count, "last_completed_at_utc": self.last_completed,
                   **extra}
        _atomic_json(self.state_dir / "status.json", payload)
        return payload

    def _progress(self, _message):
        # A current heartbeat demonstrates this sole process still owns work.
        # Vendor text is not retained in the public status record.
        self._status(status="evaluating", cycle_in_progress=True)

    def run_once(self):
        mode = os.environ.get("LIVE_TRADING_MODE", "off")
        if mode not in {"off", "real"}:
            raise RuntimeError("Combined worker requires exact off or real trading mode")
        real = mode == "real"
        startup_verified = False
        if real:
            from src.live_control import assert_real_trading_allowed
            assert_real_trading_allowed(model_name=os.environ.get("LIVE_MODEL"),
                                        host="0.0.0.0", startup_source="refresh-consumer")
            startup_verified = True
        from src.polymarket.readonly_account import (ReadOnlyClobClient, inspect_account,
                                                    inspect_original_accounting,
                                                    account_read_diagnostics)
        from src.polymarket.executor import POLYMARKET_MIN_ORDER_USD
        from src.strategy.duo_trader import WalletBankrollBasis
        if self.client is None:
            self.client = ReadOnlyClobClient()
        reader = self.account_reader or inspect_account
        accounting_reader = self.accounting_reader or inspect_original_accounting
        cycle_runner = self.cycle_runner
        if cycle_runner is None:
            from src.bot import cmd_duo_live
            cycle_runner = cmd_duo_live
        self.cycle_count += 1
        cycle_id = uuid.uuid4().hex
        self._status(status="evaluating", cycle_id=cycle_id, cycle_in_progress=True)
        account, failures = None, []
        try:
            account = reader(self.client)
            account_summary = dict(account.summary)
            account_summary.pop("orders_disabled", None)
            account_summary["read_only"] = True
        except Exception as exc:
            failures.append("account_read_unavailable")
            diagnostics = account_read_diagnostics(exc)
            account_summary = {"connected": False, "cash_confirmed": False,
                               "positions_complete": False, "open_orders_complete": False,
                               "empty_account_proved": False, **diagnostics,
                               "read_only": True}
            logger.warning("Account reads unavailable (stage=%s, error=%s, http_status=%s)",
                           diagnostics["read_stage"], diagnostics["error_class"],
                           diagnostics["http_status"] if diagnostics["http_status"] is not None else "unknown")
        try:
            accounting = accounting_reader(self.real_logs_dir, account)
        except Exception as exc:
            failures.append("original_accounting_read_unavailable")
            accounting = {"read_only": True, "complete_ownership_reconciliation": False,
                          "error_class": type(exc).__name__}
        basis = WalletBankrollBasis(
            total_equity=account.total_equity if account is not None and not failures else 0.0,
            available_cash=account.available_cash if account is not None and not failures else 0.0,
            source=("Confirmed readonly cash + complete positions; open BUY reserves deducted"
                    if account is not None and not failures else "Unconfirmed account/accounting; zero paper sizing"))
        # The consumer reconciles durable exchange IDs before any new submission.
        # This read-only gate denies unattributed current exposure/order activity.
        reconciled = not failures
        if real and accounting.get("activation_blockers"):
            failures.extend(accounting["activation_blockers"])
            reconciled = False
        from src.trading_controls import paused_status
        paused = paused_status(self.state_dir).get("paused", False) if real else False
        args = SimpleNamespace(dry_run=not real, refresh_root=self.root,
                               refresh_baseline=self.baseline, progress_callback=self._progress,
                               readonly_clob=self.client, bankroll_basis=basis,
                               observation_dir=self.state_dir / "observations",
                               model=os.environ.get("LIVE_MODEL"), host="0.0.0.0")
        evaluation_failure = None
        try:
            if real and (failures or paused):
                result = {"status": "paused" if paused and not failures else "degraded",
                          "orders_disabled": False, "trading_enabled": False,
                          "accepted_generation": None}
            else:
                result = cycle_runner(args)
            if not isinstance(result, dict):
                raise TypeError("Trading cycle did not return a report")
        except Exception as exc:
            result = {"status": "degraded", "reason": "market_or_prediction_cycle_unavailable",
                      "error_class": type(exc).__name__, "failure_stage": _evaluation_failure_stage(exc)}
            failures.append("market_or_prediction_cycle_unavailable")
            evaluation_failure = ("unavailable", result["failure_stage"], result["error_class"])
        reported_blockers = result.get("blocking_conditions", [])
        cycle_blockers = [code for code in reported_blockers
                          if isinstance(code, str) and code == "original_order_unresolved"] \
            if isinstance(reported_blockers, list) else []
        if not isinstance(reported_blockers, list) or len(cycle_blockers) != len(reported_blockers):
            failures.append("cycle_blocker_unconfirmed")
        failures.extend(cycle_blockers)
        if result.get("status") == "blocked" and not cycle_blockers:
            failures.append("cycle_blocker_unconfirmed")
        if cycle_blockers:
            if (not real or result.get("status") != "blocked" or result.get("trading_enabled") is not False
                    or result.get("orders_disabled") is not False or not result.get("accepted_generation")
                    or result.get("real_submissions") != 0 or result.get("total_orders") != 0):
                failures.append("blocked_result_contract_unconfirmed")
            evaluation_failure = ("blocked", *sorted(set(cycle_blockers)))
        if evaluation_failure != self._last_evaluation_failure:
            if evaluation_failure is not None:
                if evaluation_failure[0] == "unavailable":
                    logger.warning("Trading evaluation unavailable (stage=%s, error=%s)",
                                   evaluation_failure[1], evaluation_failure[2])
                else:
                    logger.warning("Trading evaluation blocked (%s)", ", ".join(evaluation_failure[1:]))
            self._last_evaluation_failure = evaluation_failure
        if not real and (result.get("trading_enabled") is True or result.get("orders_disabled") is False):
            raise RuntimeError("Orders-disabled cycle returned incompatible trading state")
        if not real and (result.get("trading_enabled") is not False or result.get("orders_disabled") is not True
                or not result.get("accepted_generation")):
            failures.append("orders_disabled_result_contract_unconfirmed")
        if real and not failures and not paused and (
                result.get("orders_disabled") is not False or result.get("trading_enabled") is not True
                or not result.get("accepted_generation")):
            failures.append("real_result_contract_unconfirmed")
        if result.get("status") in {"error", "busy", "degraded"}:
            failures.append(str(result.get("status")))
        reconciled = reconciled and not failures
        allowed = {"status", "accepted_generation", "publication_job_id", "bookmaker_observed_at",
                   "market_observed_at", "prediction_count", "executable_prediction_count",
                   "matched_market_count", "total_orders", "orders_disabled", "trading_enabled",
                   "error_class", "failure_stage", "freshness", "execution_audit", "current_card", "mapped_markets",
                   "snapshot_binding", "original_accounting_maintenance", "confirmed_placed_orders"}
        safe_result = {key: result[key] for key in allowed if key in result}
        if cycle_blockers:
            safe_result["blocking_conditions"] = sorted(set(cycle_blockers))
        order_fields = {"fighter", "opponent", "status", "dry_run", "bet_size_usd",
                        "price", "shares", "order_type", "side"}
        for name in ("trader_s", "trader_c", "trader_m"):
            trader = result.get(name)
            if isinstance(trader, dict):
                safe_result[name] = {key: trader[key] for key in
                    ("name", "allocation", "available_cash_start", "total_wagered", "total_equity")
                    if key in trader}
                for group in ("orders", "near_miss_orders"):
                    safe_result[name][group] = [{key: value for key, value in order.items()
                                                if key in order_fields}
                                               for order in trader.get(group, [])]
        # Reason text can include vendor URLs. Keep only internal classification.
        safe_result["predictions"] = _safe_predictions(result.get("predictions", []))
        self.last_completed = _now()
        report = {"schema_version": 1, "cycle_id": cycle_id,
                  "worker_pid": os.getpid(), "worker_started_at_utc": self.started_at,
                  "completed_at_utc": self.last_completed, "orders_disabled": not real,
                  "trading_enabled": real and not failures and not paused,
                  "effective_live_mode": mode, "paused": paused,
                  "startup_verified": startup_verified, "accounting_reconciled": reconciled,
                  "accounting_reconciliation_scope": "Complete current account reads and no unattributed active exposure; original-ID recovery precedes submissions",
                  "real_submissions": result.get("real_submissions") if real else 0,
                  "paper_state_separate": not real, "account": account_summary,
                  "original_accounting": accounting, "cycle": safe_result,
                  "blocking_conditions": sorted(set(failures)),
                  "ownership_reconciliation_required_before_activation": not real,
                  "warnings": ["Cash below existing minimum buy; insufficient-cash checks remain active"]
                    if account is not None and account.available_cash < POLYMARKET_MIN_ORDER_USD else []}
        _atomic_json(self.state_dir / "latest.json", report)
        _atomic_json(self.state_dir / "report.json", report)
        # Durable minimal receipt supports restart proof without retaining full
        # credential/account payloads or rewriting earlier successful records.
        _atomic_json(self.state_dir / "cycles" / f"{cycle_id}.json", {
            key: value for key, value in report.items() if key not in {"cycle", "original_accounting"}
        } | {"accepted_generation": result.get("accepted_generation"),
             "prediction_count": result.get("prediction_count"),
             "matched_market_count": result.get("matched_market_count")})
        blocked = bool(cycle_blockers) and set(failures) == set(cycle_blockers)
        self._status(status="blocked" if blocked else "degraded" if failures else "waiting", ready=not failures,
                     cycle_in_progress=False, cycle_id=cycle_id,
                     startup_verified=startup_verified, accounting_reconciled=reconciled,
                     blocking_conditions=sorted(set(failures)))
        return report

    def run(self, *, poll_seconds=300, once=False):
        from src.refresh_live import trading_consumer_lock
        if not math.isfinite(float(poll_seconds)) or float(poll_seconds) < 1:
            raise ValueError("Trading worker poll seconds must be finite and at least one")
        with trading_consumer_lock(self.state_dir / "trading-worker.lock"):
            self._status(status="starting")
            try:
                while not self.stop_event.is_set():
                    if self.supervisor_pid is not None and os.getppid() != self.supervisor_pid:
                        break
                    self.run_once()
                    if once:
                        break
                    self.stop_event.wait(max(1, float(poll_seconds)))
            finally:
                self._status(status="stopped", ready=False, cycle_in_progress=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--real-logs-dir")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float)
    args = parser.parse_args(argv)
    if args.poll_seconds is not None and (
            not math.isfinite(args.poll_seconds) or args.poll_seconds < 1):
        parser.error("--poll-seconds must be finite and at least one")
    real_logs = args.real_logs_dir or str(Path(args.state_dir).resolve().parent)
    configure_environment(args.state_dir, args.root, args.baseline, real_logs)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from src.config import REFRESH_TRADING_POLL_SECONDS
    worker = TradingWorker(root=args.root, baseline=args.baseline,
                           state_dir=args.state_dir, real_logs_dir=real_logs)
    def runtime_secrets():
        credentials = getattr(worker.client, "_api_creds", None)
        return tuple(getattr(credentials, name, "") for name in
                     ("api_key", "api_secret", "api_passphrase"))
    for handler in logging.getLogger().handlers:
        handler.addFilter(_SecretFilter(runtime_secrets))
    def stop(*_args):
        worker.stop_event.set()
        raise SystemExit(0)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    worker.supervisor_pid = install_parent_death_guard()
    worker.run(poll_seconds=(REFRESH_TRADING_POLL_SECONDS if args.poll_seconds is None
                             else args.poll_seconds), once=args.once)


if __name__ == "__main__":
    main()
