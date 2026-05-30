"""Durable alert store for the dashboard Activity view.

`bot.log` is a single chronological stream dominated by INFO chatter (werkzeug
HTTP access logs, the background monitor loop, prediction output). The Activity
page only reads the most recent ~1000 log lines, so an overnight WARNING/ERROR
scrolls out of that window long before it is useful to read — it is still on
disk, just no longer surfaced.

This module mirrors only high-severity records (WARNING/ERROR/CRITICAL) into a
compact, time-bounded sidecar file (``alerts.jsonl``) that the UI reads
separately. That guarantees alerts stay visible for the configured retention
window regardless of how much INFO traffic happens afterwards.

Each line is a self-describing JSON object::

    {"ts": 1716950043.12, "timestamp": "2026-05-29 02:14:03",
     "level": "ERROR", "source": "src.data.scraper", "message": "..."}

``ts`` (epoch seconds) drives retention; ``timestamp`` is the human display form
matching bot.log's local-time format so the two feeds line up visually.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

ALERT_STORE_FILENAME = "alerts.jsonl"

# Severities we persist. Everything below WARNING stays only in bot.log.
ALERT_LEVELS = frozenset({"WARNING", "ERROR", "CRITICAL"})

# Per-line message cap keeps each JSONL record bounded even when a traceback is
# huge, and keeps lines small enough that concurrent appends from separate
# processes (web + scheduled bot) stay unlikely to interleave.
_MAX_MESSAGE_CHARS = 1500

_TS_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOCK_POLL_SECONDS = 0.05
_LOCK_STALE_SECONDS = 30.0
_LOCK_TIMEOUT_SECONDS = 10.0

logger = logging.getLogger(__name__)

# Serializes file mutations within this process before taking the on-disk lock
# below. Reads are intentionally lock-free: os.replace keeps them on a complete
# old or new file, never a partial rewrite.
_file_lock = threading.Lock()

# Throttle state so a read path prunes a given file at most once per interval per
# process. Keyed by resolved path so distinct files (incl. test fixtures) don't
# share a throttle clock.
_throttle_lock = threading.Lock()
_last_prune_by_path: dict[str, float] = {}


def _format_timestamp(epoch: float) -> str:
    return time.strftime(_TS_FORMAT, time.localtime(epoch))


def _truncate(message: str) -> str:
    if len(message) <= _MAX_MESSAGE_CHARS:
        return message
    return message[:_MAX_MESSAGE_CHARS] + " …(truncated)"


def _lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


@contextmanager
def _cross_process_lock(path: Path, *, timeout_seconds: float = _LOCK_TIMEOUT_SECONDS):
    """Acquire an atomic lock file shared by append and prune writers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path(path)
    start = time.monotonic()
    fd: int | None = None
    token = f"{os.getpid()} {threading.get_ident()} {time.monotonic_ns()}\n"

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, token.encode("ascii"))
            break
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
            except OSError:
                age = 0.0
            if age > _LOCK_STALE_SECONDS:
                try:
                    lock_path.unlink()
                except OSError:
                    pass
                continue
            if time.monotonic() - start >= timeout_seconds:
                raise TimeoutError(f"Timed out waiting for alert store lock: {lock_path}")
            time.sleep(_LOCK_POLL_SECONDS)

    try:
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            if lock_path.read_text(encoding="ascii") == token:
                lock_path.unlink()
        except OSError:
            pass


@contextmanager
def _locked_alert_store(path: Path):
    with _file_lock:
        with _cross_process_lock(path):
            yield


class DurableAlertHandler(logging.Handler):
    """Logging handler that appends WARNING+ records to ``alerts.jsonl``."""

    def __init__(self, path: Path | str, level: int = logging.WARNING) -> None:
        super().__init__(level=level)
        self.path = Path(path)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            if record.exc_info:
                message = (
                    f"{message} | "
                    f"{logging.Formatter().formatException(record.exc_info)}"
                )
            entry = {
                "ts": float(record.created),
                "timestamp": _format_timestamp(record.created),
                "level": record.levelname,
                "source": record.name,
                "message": _truncate(message),
            }
            line = json.dumps(entry, ensure_ascii=False)
            with _locked_alert_store(self.path):
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except Exception:  # pragma: no cover - logging must never raise
            self.handleError(record)


def install_alert_handler(
    logs_dir: Path | str, *, level: int = logging.WARNING
) -> DurableAlertHandler:
    """Attach a :class:`DurableAlertHandler` to the root logger exactly once.

    Idempotent: calling it again for the same target path returns the existing
    handler instead of double-writing every alert.
    """
    root = logging.getLogger()
    path = Path(logs_dir) / ALERT_STORE_FILENAME
    for existing in root.handlers:
        if isinstance(existing, DurableAlertHandler) and existing.path == path:
            return existing
    handler = DurableAlertHandler(path, level=level)
    root.addHandler(handler)
    return handler


def _parse_timestamp_string(value) -> float | None:
    if not value:
        return None
    try:
        return time.mktime(time.strptime(str(value), _TS_FORMAT))
    except (ValueError, OverflowError):
        return None


def _entry_epoch(obj: dict) -> float | None:
    ts = obj.get("ts")
    if isinstance(ts, (int, float)):
        return float(ts)
    return _parse_timestamp_string(obj.get("timestamp"))


def _iter_alert_records(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict):
                yield obj


def load_recent_alerts(
    path: Path | str,
    max_age_hours: float,
    *,
    now: float | None = None,
    limit: int | None = None,
    _log_errors: bool = True,
    _raise_errors: bool = False,
) -> list[dict]:
    """Return persisted alert records within the retention window, oldest first.

    Malformed or out-of-window lines are skipped; a missing file yields ``[]``.
    """
    path = Path(path)
    if not path.exists():
        return []
    cutoff = (now if now is not None else time.time()) - max_age_hours * 3600.0
    records: list[dict] = []
    try:
        for obj in _iter_alert_records(path):
            epoch = _entry_epoch(obj)
            if epoch is None or epoch < cutoff:
                continue
            obj["ts"] = epoch
            records.append(obj)
    except OSError as exc:
        if _raise_errors:
            raise
        if _log_errors:
            logger.warning("Failed to read alert store %s: %s", path, exc)
        return []
    records.sort(key=lambda item: item.get("ts") or 0.0)
    if limit is not None and limit > 0:
        records = records[-limit:]
    return records


def prune_alert_store(
    path: Path | str, max_age_hours: float, *, now: float | None = None
) -> int:
    """Rewrite the alert store keeping only in-window records.

    Returns the number of records retained. The rewrite is atomic (temp file +
    ``os.replace``) so a concurrent reader never sees a partial file, and the
    whole read→rewrite→replace runs under the same in-process and on-disk locks
    as handler ``emit`` so concurrent appends are not lost across the swap.
    """
    path = Path(path)
    if not path.exists():
        return 0
    kept: list[dict] = []
    error: OSError | TimeoutError | None = None
    tmp_name = None
    # Hold the write locks across read+write+replace so emit cannot append a new
    # record after we read but before we swap the rewritten file in.
    try:
        with _locked_alert_store(path):
            kept = load_recent_alerts(
                path,
                max_age_hours,
                now=now,
                _log_errors=False,
                _raise_errors=True,
            )
            fd, tmp_name = tempfile.mkstemp(
                dir=str(path.parent), prefix=".alerts-", suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for obj in kept:
                    handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
            os.replace(tmp_name, path)
            tmp_name = None
    except (OSError, TimeoutError) as exc:
        error = exc
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass
    if error is not None:
        logger.warning("Failed to prune alert store %s: %s", path, error)
    return len(kept)


def maybe_prune_alert_store(
    path: Path | str,
    max_age_hours: float,
    *,
    min_interval_seconds: float = 3600.0,
    now: float | None = None,
) -> None:
    """Best-effort, throttled prune. Never raises into the caller."""
    key = str(Path(path).resolve())
    monotonic = time.monotonic()
    with _throttle_lock:
        last = _last_prune_by_path.get(key)
        if last is not None and (monotonic - last) < min_interval_seconds:
            return
        _last_prune_by_path[key] = monotonic
    try:
        prune_alert_store(path, max_age_hours, now=now)
    except Exception as exc:  # pragma: no cover - pruning must never break reads
        logger.warning("Alert store prune skipped: %s", exc)
