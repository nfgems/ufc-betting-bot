"""One refresh owner and one guarded UFC market/execution consumer.

Off-mode accounting is isolated. Real-mode execution uses the original ledgers
only after the existing live-control gates pass; reports use a separate root.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

from src.data.refresh_service import RefreshService, create_app
from src.live_control import evaluate_live_startup, is_public_bind

logger = logging.getLogger(__name__)


def require_combined_startup(*, host):
    mode = str(os.getenv("LIVE_TRADING_MODE", "off"))
    if mode not in {"off", "real"}:
        raise RuntimeError("Combined service requires LIVE_TRADING_MODE=off or real")
    if is_public_bind(host) and not str(os.getenv("WEB_DASHBOARD_TOKEN", "")).strip():
        raise RuntimeError("WEB_DASHBOARD_TOKEN is required for the public combined service")
    if str(os.getenv("BTC5M_LIVE_PROFILES", "")).strip():
        raise RuntimeError("BTC5M_LIVE_PROFILES must remain empty for the UFC combined service")
    monthly = str(os.getenv("UFC_MONTHLY_CANDIDATE_ENABLED", "0")).strip().lower()
    if monthly in {"1", "true", "yes", "on", "armed"} or str(os.getenv("UFC_V7_PROMOTION_MONTH_UTC", "")).strip():
        raise RuntimeError("Monthly candidate training and automatic promotion must remain disabled")
    if mode == "real" and not str(os.getenv("LIVE_MODEL", "")).strip():
        raise RuntimeError("LIVE_MODEL must explicitly select the accepted model for real startup")
    status = evaluate_live_startup(host=host, startup_source="combined_service")
    if (status.get("ready") is not True or status.get("effective_live_mode") != mode
            or status.get("trading_enabled") is not (mode == "real")):
        errors = status.get("errors") or ["Requested live mode was not enabled by live_control"]
        raise RuntimeError("Combined live startup blocked: " + "; ".join(errors))
    return status


def validate_roots(root, baseline, state_dir):
    root, baseline, state_dir = (Path(value).resolve() for value in (root, baseline, state_dir))
    for left, right in ((root, baseline), (root, state_dir), (baseline, state_dir)):
        if left == right or left.is_relative_to(right) or right.is_relative_to(left):
            raise RuntimeError("Refresh, accepted baseline, and consumer state must have separate roots")
    return root, baseline, state_dir


@contextmanager
def supervisor_lock(path):
    """Nonblocking lifetime lease, released by the OS after interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        acquired = False
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0, os.SEEK_END)
                if stream.tell() == 0:
                    stream.write(b"\0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise RuntimeError("Another combined supervisor already owns this refresh root") from exc
        try:
            yield
        finally:
            if acquired:
                if os.name == "nt":
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def child_environment(state_dir, *, mode="off", real_logs_dir=None):
    state_dir = Path(state_dir).resolve()
    environment = os.environ.copy()
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "UFC_REFRESH_SUPERVISOR_PID": str(os.getpid()),
        "UFC_REFRESH_TRADING_STATE_DIR": str(state_dir),
    })
    if mode == "off":
        environment.update({
            "LIVE_TRADING_MODE": "off", "LIVE_TRADING_ARMED": "0",
            "LIVE_TRADING_CONFIRMATION": "", "UFC_DATA_DIR": str(state_dir / "data"),
            "UFC_LOGS_DIR": str(state_dir / "logs"),
        })
    elif mode == "real":
        if not real_logs_dir:
            raise RuntimeError("Real consumer requires the original accounting root")
        environment.update(LIVE_TRADING_MODE="real", UFC_LOGS_DIR=str(Path(real_logs_dir).resolve()))
    else:
        raise RuntimeError("Unsupported combined consumer mode")
    return environment


class TradingProcess:
    """HTTP view of the one child; reports are immutable between atomic saves."""

    def __init__(self, process, state_dir, *, max_age_seconds=600, mode="off"):
        self.process = process
        self.state_dir = Path(state_dir)
        self.max_age_seconds = max_age_seconds
        if mode not in {"off", "real"}:
            raise ValueError("Unsupported combined consumer mode")
        self.mode = mode

    def status(self):
        saved = json.loads((self.state_dir / "status.json").read_text(encoding="utf-8"))
        saved["process_alive"] = self.process.poll() is None
        from src.trading_controls import paused_status
        saved["paused"] = paused_status(self.state_dir).get("paused", False)
        if saved["paused"] or not saved["process_alive"]:
            saved["trading_enabled"] = False
        return saved

    def report(self):
        return json.loads((self.state_dir / "latest.json").read_text(encoding="utf-8"))

    def _running_status(self):
        saved = self.status()
        if not saved["process_alive"] or saved.get("pid") != self.process.pid:
            raise RuntimeError("Trading consumer owner is not running")
        instant = datetime.fromisoformat(saved["updated_at_utc"])
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise RuntimeError("Trading heartbeat has no timezone")
        age = (datetime.now(timezone.utc) - instant).total_seconds()
        if age < -30 or age > self.max_age_seconds:
            raise RuntimeError("Trading consumer heartbeat is stale")
        return saved

    def readiness(self):
        """Strict execution readiness, independent of HTTP service readiness."""
        saved = self._running_status()
        real = self.mode == "real"
        paused = saved.get("paused") is True
        if (saved.get("ready") is not True or saved.get("orders_disabled") is not (not real)
                or saved.get("trading_enabled") is not (real and not paused)):
            raise RuntimeError("Trading consumer is not ready in the configured mode")
        if real and (saved.get("effective_live_mode") != "real"
                     or saved.get("startup_verified") is not True
                     or saved.get("accounting_reconciled") is not True):
            raise RuntimeError("Real trading startup or original accounting is unverified")
        return {"ready": True, "process_alive": True, "orders_disabled": not real,
                "trading_enabled": real and not paused, "effective_live_mode": self.mode,
                "paused": paused}

    def service_readiness(self, publication):
        """Serve verified observations while a known order blocks execution.

        This does not change ``readiness`` or any execution control. A stopped,
        stale, failed or unverified consumer still fails the deployment check.
        """
        try:
            return {**self.readiness(), "trading_ready": True}
        except RuntimeError:
            pass
        saved = self._running_status()
        report = self.report()
        cycle = report.get("cycle") or {}
        blocker = ["original_order_unresolved"]
        if (self.mode != "real" or saved.get("status") != "blocked"
                or saved.get("cycle_in_progress") is not False
                or saved.get("ready") is not False or saved.get("accounting_reconciled") is not False
                or saved.get("blocking_conditions") != blocker
                or report.get("blocking_conditions") != blocker
                or cycle.get("blocking_conditions") != blocker or cycle.get("status") != "blocked"
                or any(source.get("trading_enabled") is not False or source.get("orders_disabled") is not False
                       for source in (saved, report, cycle))
                or any(source.get("startup_verified") is not True or source.get("effective_live_mode") != "real"
                       for source in (saved, report))
                or report.get("accounting_reconciled") is not False
                or any(type(value) is not int or value != 0 for value in (
                    report.get("real_submissions"), cycle.get("total_orders"), cycle.get("confirmed_placed_orders")))):
            raise RuntimeError("Consumer is not safely observing with a known execution block")
        if (report.get("worker_pid") != self.process.pid
                or not saved.get("started_at_utc")
                or report.get("worker_started_at_utc") != saved["started_at_utc"]
                or not saved.get("cycle_id") or report.get("cycle_id") != saved["cycle_id"]
                or report.get("completed_at_utc") != saved.get("last_completed_at_utc")):
            raise RuntimeError("Observation report does not belong to the current worker cycle")
        generation, job = publication.get("generation_id"), publication.get("job_id")
        pin = (publication.get("prediction_report") or {}).get("sha256")
        if (not generation or not job or not pin or cycle.get("accepted_generation") != generation
                or cycle.get("publication_job_id") != job
                or (cycle.get("snapshot_binding") or {}).get("final_report_sha256") != pin):
            raise RuntimeError("Observation report is not bound to the current publication")
        now = datetime.now(timezone.utc)
        times = {}
        for name, value in (("started", saved["started_at_utc"]),
                            ("completed", report.get("completed_at_utc")),
                            ("bookmaker", cycle.get("bookmaker_observed_at")),
                            ("market", cycle.get("market_observed_at"))):
            instant = datetime.fromisoformat(value)
            if instant.tzinfo is None or instant.utcoffset() is None:
                raise RuntimeError("Observation time has no timezone")
            if name != "started" and not -30 <= (now - instant).total_seconds() <= self.max_age_seconds:
                raise RuntimeError("Consumer observations are stale")
            times[name] = instant
        if not (times["started"] <= times["bookmaker"] <= times["completed"]
                and times["started"] <= times["market"] <= times["completed"]):
            raise RuntimeError("Observation times do not belong to this worker cycle")
        return {"ready": True, "trading_ready": False, "process_alive": True,
                "orders_disabled": False, "trading_enabled": False, "effective_live_mode": "real",
                "paused": saved.get("paused") is True, "blocking_conditions": blocker,
                "observation_status": "current", "cycle_id": saved["cycle_id"]}


def serve(service, *, state_dir, host="0.0.0.0", port=5050,
          poll_seconds=60, trading_poll_seconds=300, real_logs_dir=None, stop=None):
    """Own both workers until shutdown; never use Flask's reloader or legacy loop."""
    from werkzeug.serving import make_server

    startup = require_combined_startup(host=host)
    mode = startup["effective_live_mode"]
    root, baseline, state_dir = validate_roots(service.root, service.baseline, state_dir)
    real_logs_dir = Path(real_logs_dir or os.getenv("UFC_LOGS_DIR", "/app/logs")).resolve()
    if real_logs_dir == state_dir or real_logs_dir.is_relative_to(state_dir):
        raise RuntimeError("Original accounting root must remain outside the consumer state root")
    if mode == "off":
        for name in ("logs", "data"):
            if not (state_dir / name).resolve().is_relative_to(state_dir):
                raise RuntimeError("Paper data and logs must remain inside the paper root")
    else:
        from src.config import LOGS_DIR
        if real_logs_dir != LOGS_DIR.resolve():
            raise RuntimeError("Real accounting root differs from the live_control configured root")
    for interval in (poll_seconds, trading_poll_seconds):
        if not math.isfinite(interval) or interval < 1:
            raise ValueError("Worker poll intervals must be finite and at least one second")
    stop = stop if stop is not None else threading.Event()
    with supervisor_lock(root / "combined-supervisor.lock"):
        directories = (state_dir, state_dir / "logs", state_dir / "data") if mode == "off" else (state_dir,)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen([
            sys.executable, "-m", "src.trading_worker",
            "--root", str(root), "--baseline", str(baseline),
            "--state-dir", str(state_dir), "--real-logs-dir", str(real_logs_dir),
            "--poll-seconds", str(trading_poll_seconds),
        ], env=child_environment(state_dir, mode=mode, real_logs_dir=real_logs_dir))
        trading = TradingProcess(process, state_dir,
                                 max_age_seconds=max(600, trading_poll_seconds * 2), mode=mode)

        def work():
            while not stop.is_set():
                try:
                    service.tick()
                    if not service.published_path.exists():
                        service.initialize_predictions()
                except Exception:
                    logger.exception("Refresh worker retained its original state")
                stop.wait(poll_seconds)

        worker = threading.Thread(target=work, name="ufc-refresh-calendar", daemon=True)
        server = None
        previous_handlers = {}
        try:
            worker.start()
            server = make_server(host, port, create_app(service, worker=worker, trading=trading), threaded=True)
            server.timeout = 0.5
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGTERM, signal.SIGINT):
                    previous_handlers[signum] = signal.getsignal(signum)
                    signal.signal(signum, lambda _signum, _frame: stop.set())
            logger.info("Combined service started: one refresh owner and one %s UFC consumer", mode)
            while not stop.is_set():
                if process.poll() is not None:
                    raise RuntimeError("Trading consumer exited")
                if not worker.is_alive():
                    raise RuntimeError("Refresh calendar worker exited")
                server.handle_request()
        finally:
            stop.set()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            if server is not None:
                server.server_close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            worker.join(timeout=5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--real-logs-dir", type=Path, default=os.getenv("UFC_LOGS_DIR", "/app/logs"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5050")))
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--trading-poll-seconds", type=float, default=300)
    args = parser.parse_args(argv)
    require_combined_startup(host=args.host)
    root, baseline, state_dir = validate_roots(args.root, args.baseline, args.state_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    service = RefreshService(root=root, baseline=baseline)
    serve(service, state_dir=state_dir, host=args.host, port=args.port,
          poll_seconds=args.poll_seconds, trading_poll_seconds=args.trading_poll_seconds,
          real_logs_dir=args.real_logs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
