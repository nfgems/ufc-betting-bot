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

The file is an append-only event stream. Alert observations are coalesced into
incidents at read time. Incidents emitted with a stable lifecycle key remain
*active* until an explicit recovery event is written; quiet time alone never
implies recovery for those managed incidents. Unkeyed warnings retain the
historical time-window behavior. Legacy JSONL lines without an ``event`` or
``fingerprint`` field are still valid alert observations.

Each line is a self-describing JSON object::

    {"ts": 1716950043.12, "timestamp": "2026-05-29 02:14:03",
     "level": "ERROR", "source": "src.data.scraper", "message": "..."}

``ts`` (epoch seconds) drives retention; ``timestamp`` is the human display form
matching bot.log's local-time format so the two feeds line up visually.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
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

_LEVEL_RANK = {"WARNING": 1, "ERROR": 2, "CRITICAL": 3}
_WHITESPACE_RE = re.compile(r"\s+")


def _format_timestamp(epoch: float) -> str:
    return time.strftime(_TS_FORMAT, time.localtime(epoch))


def _truncate(message: str) -> str:
    if len(message) <= _MAX_MESSAGE_CHARS:
        return message
    return message[:_MAX_MESSAGE_CHARS] + " …(truncated)"


def normalize_alert_message(message: str) -> str:
    """Return the conservative form used to identify repeated incidents.

    Only case and whitespace are normalized. Values such as fighter names,
    URLs, HTTP statuses, and event timestamps are deliberately preserved so
    distinct operational failures are not accidentally merged.
    """
    return _WHITESPACE_RE.sub(" ", str(message or "").strip()).casefold()


def alert_fingerprint(
    *, source: str = "", message: str = "", incident_key: str | None = None
) -> str:
    """Build a stable incident fingerprint.

    Producers that know the lifecycle of an incident should pass a stable
    ``incident_key`` on both the alert and recovery. Existing logging calls need
    no changes: their source plus conservatively normalized message is used.
    """
    if incident_key is not None:
        normalized_key = _WHITESPACE_RE.sub(" ", str(incident_key).strip()).casefold()
        if not normalized_key:
            raise ValueError("incident_key must not be empty")
        material = f"key\0{normalized_key}"
    else:
        normalized_source = str(source or "").strip().casefold()
        normalized_message = normalize_alert_message(message)
        if not normalized_source or not normalized_message:
            raise ValueError("source and message are required without an incident_key")
        material = f"message\0{normalized_source}\0{normalized_message}"
    return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()


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


def _append_store_entry(path: Path | str, entry: dict) -> None:
    path = Path(path)
    line = json.dumps(entry, ensure_ascii=False)
    with _locked_alert_store(path):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def record_alert(
    path: Path | str,
    *,
    level: str,
    source: str,
    message: str,
    incident_key: str | None = None,
    created: float | None = None,
) -> dict:
    """Append one alert observation and return the stored event.

    ``incident_key`` is optional for backwards-compatible logging. Lifecycle-
    aware producers should supply one so a recovery can close the incident even
    when individual occurrence messages contain changing diagnostic detail.
    """
    normalized_level = str(level or "").upper()
    if normalized_level not in ALERT_LEVELS:
        raise ValueError(f"Unsupported alert level: {level!r}")
    normalized_source = str(source or "").strip()
    normalized_message = _truncate(str(message or ""))
    fingerprint = alert_fingerprint(
        source=normalized_source,
        message=normalized_message,
        incident_key=incident_key,
    )
    epoch = float(time.time() if created is None else created)
    entry = {
        "schema_version": 2,
        "event": "alert",
        "ts": epoch,
        "timestamp": _format_timestamp(epoch),
        "level": normalized_level,
        "source": normalized_source,
        "message": normalized_message,
        "fingerprint": fingerprint,
    }
    if incident_key is not None:
        entry["incident_key"] = str(incident_key)
    _append_store_entry(path, entry)
    return entry


def record_recovery(
    path: Path | str,
    *,
    fingerprint: str | None = None,
    source: str = "",
    message: str = "",
    incident_key: str | None = None,
    recovery_message: str = "Recovered",
    created: float | None = None,
) -> dict:
    """Append an explicit recovery event for exactly one incident.

    Identify the incident with its returned ``fingerprint``, the same stable
    ``incident_key`` used by :func:`record_alert`, or the original source and
    message. A recovery never uses elapsed quiet time and cannot wildcard-close
    unrelated alerts.
    """
    if fingerprint is None:
        fingerprint = alert_fingerprint(
            source=source,
            message=message,
            incident_key=incident_key,
        )
    fingerprint = str(fingerprint or "").strip()
    if not fingerprint:
        raise ValueError("fingerprint must not be empty")
    epoch = float(time.time() if created is None else created)
    entry = {
        "schema_version": 2,
        "event": "recovery",
        "ts": epoch,
        "timestamp": _format_timestamp(epoch),
        "source": str(source or "").strip(),
        "message": _truncate(str(recovery_message or "Recovered")),
        "recovery_message": _truncate(str(recovery_message or "Recovered")),
        "fingerprint": fingerprint,
    }
    if incident_key is not None:
        entry["incident_key"] = str(incident_key)
    _append_store_entry(path, entry)
    return entry


def record_recovery_if_active(
    path: Path | str,
    *,
    fingerprint: str | None = None,
    source: str = "",
    message: str = "",
    incident_key: str | None = None,
    recovery_message: str = "Recovered",
    created: float | None = None,
) -> dict | None:
    """Atomically append a recovery only when that incident is active.

    Health checks may emit the same recovery signal on every successful probe
    so a first success after process restart can close a durable pre-restart
    alert.  This operation keeps that restart-safe behavior from producing an
    endless stream of recovery-only JSONL records during healthy operation.
    """
    if fingerprint is None:
        fingerprint = alert_fingerprint(
            source=source,
            message=message,
            incident_key=incident_key,
        )
    fingerprint = str(fingerprint or "").strip()
    if not fingerprint:
        raise ValueError("fingerprint must not be empty")

    path = Path(path)
    epoch = float(time.time() if created is None else created)
    entry = {
        "schema_version": 2,
        "event": "recovery",
        "ts": epoch,
        "timestamp": _format_timestamp(epoch),
        "source": str(source or "").strip(),
        "message": _truncate(str(recovery_message or "Recovered")),
        "recovery_message": _truncate(str(recovery_message or "Recovered")),
        "fingerprint": fingerprint,
    }
    if incident_key is not None:
        entry["incident_key"] = str(incident_key)

    with _locked_alert_store(path):
        active = False
        if path.exists():
            for record in _iter_alert_records(path):
                if _fingerprint_for_record(record) != fingerprint:
                    continue
                event = str(record.get("event", "alert") or "alert").strip().lower()
                if event == "recovery":
                    active = False
                elif event == "alert" and str(record.get("level", "")).upper() in ALERT_LEVELS:
                    active = True
        if not active:
            return None
        line = json.dumps(entry, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return entry


# Intent-revealing alias for callers that update a known source lifecycle.
mark_alert_recovered = record_recovery


class DurableAlertHandler(logging.Handler):
    """Persist WARNING+ observations plus explicit INFO-level recoveries."""

    def __init__(self, path: Path | str, level: int = logging.INFO) -> None:
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
            recovery_keys = getattr(record, "alert_recovered_incident_keys", ()) or ()
            if isinstance(recovery_keys, str):
                recovery_keys = (recovery_keys,)
            for incident_key in recovery_keys:
                if str(incident_key or "").strip():
                    record_recovery_if_active(
                        self.path,
                        source=record.name,
                        incident_key=str(incident_key),
                        recovery_message=message,
                        created=record.created,
                    )

            if record.levelno < logging.WARNING:
                return
            record_alert(
                self.path,
                level=record.levelname,
                source=record.name,
                message=message,
                incident_key=getattr(record, "alert_incident_key", None),
                created=record.created,
            )
        except Exception:  # pragma: no cover - logging must never raise
            self.handleError(record)


def install_alert_handler(
    logs_dir: Path | str, *, level: int = logging.INFO
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


def _fingerprint_for_record(record: dict) -> str | None:
    fingerprint = str(record.get("fingerprint", "") or "").strip()
    if fingerprint:
        return fingerprint
    if str(record.get("event", "alert") or "alert").lower() == "recovery":
        return None
    legacy_incident_key = _legacy_incident_key_for_record(record)
    try:
        return alert_fingerprint(
            source=str(record.get("source", "") or ""),
            message=str(record.get("message", "") or ""),
            incident_key=record.get("incident_key") or legacy_incident_key,
        )
    except ValueError:
        return None


def _incident_key_for_record(record: dict) -> str | None:
    explicit = str(record.get("incident_key", "") or "").strip()
    if explicit:
        return explicit
    # A stored fingerprint identifies a record written by the current alert
    # schema. Unkeyed current-schema warnings intentionally age out; applying
    # the legacy text migration here would make them lifecycle-managed under a
    # different fingerprint that no keyed recovery can close.
    if str(record.get("fingerprint", "") or "").strip():
        return None
    return _legacy_incident_key_for_record(record)


def _legacy_incident_key_for_record(record: dict) -> str | None:
    """Map pre-schema lifecycle warnings to their new stable incident keys.

    Production may already contain legacy alert observations when this release
    starts. Without this narrow migration, the first healthy probe can close the
    new keyed incident but leave the old warning active forever. These patterns
    intentionally cover only the operational clusters that now have explicit
    producer lifecycles.
    """
    source = str(record.get("source", "") or "").casefold()
    message = normalize_alert_message(str(record.get("message", "") or ""))

    if "newest method-odds snapshot is stale" in message:
        return "method-odds:expected-coverage"
    if "ufc.com events fetch failed" in message:
        return "upstream-fetch:https://www.ufc.com/events"
    if "official ufc roster sync failed" in message:
        return "ufc-active-roster:live-sync-failed"
    if "scheduled ufc refresh alerts:" in message or "scheduled ufc refresh failed:" in message:
        return "scheduled-ufc-refresh:degraded"

    if "tapology" in message and (
        "reader" in message
        or "railway runtime tapology fetch disabled" in message
    ) and any(marker in message for marker in ("unavailable", "failed", "status 403", "disabled")):
        return "external-source:tapology:reader-circuit-opened"

    fightdx_transient_markers = (
        "timed out",
        "timeout",
        "connection",
        "status 401",
        "status 403",
        "status 408",
        "status 429",
        "status 500",
        "status 502",
        "status 503",
        "status 504",
    )
    if (
        "fightdx" in message
        and any(marker in message for marker in fightdx_transient_markers)
        and (
            "fallback_scrapers" in source
            or "profile_supplement" in source
            or "profile lookup failed" in message
            or "profile scrape failed" in message
        )
    ):
        return "external-source:fightdx:transient-request-failure"
    return None


def load_alert_incidents(
    path: Path | str,
    max_age_hours: float,
    *,
    now: float | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Resolve recent JSONL events into coalesced alert incidents.

    Repeated observations with the same fingerprint increment
    ``occurrence_count``. Only an explicit recovery event changes an incident to
    ``recovered``; a later observation reactivates it. Lifecycle-managed active
    incidents persist until recovery, while unkeyed observations and recovered
    history use the requested retention window. Legacy records are fingerprinted
    from their source and normalized message on read.
    """
    # Lifecycle must be resolved before applying the recovered-history window.
    # Otherwise an unresolved alert silently disappears when its observation
    # ages out, and a recent recovery whose alert is older than the window is
    # misread as an orphan. Lifecycle-managed active incidents intentionally
    # survive until explicit recovery; unmanaged observations age by last-seen
    # time and recovered incidents age by recovery time.
    incidents: dict[str, dict] = {}
    for record in load_recent_alerts(path, float("inf"), now=now):
        event = str(record.get("event", "alert") or "alert").strip().lower()
        fingerprint = _fingerprint_for_record(record)
        if fingerprint is None:
            continue
        lifecycle_managed = _incident_key_for_record(record) is not None

        epoch = _entry_epoch(record)
        if epoch is None:
            continue

        if event == "recovery":
            incident = incidents.get(fingerprint)
            # A recovery with no retained alert observation is not an incident
            # by itself. This also keeps a pruned alert from reappearing as an
            # unexplained recovery-only row.
            if incident is None:
                continue
            incident["status"] = "recovered"
            incident["active_occurrence_count"] = 0
            incident["recovery_count"] += 1
            incident["recovered_ts"] = epoch
            incident["recovered_at"] = str(
                record.get("timestamp") or _format_timestamp(epoch)
            )
            incident["recovery_message"] = str(
                record.get("recovery_message")
                or record.get("message")
                or "Recovered"
            )
            continue

        level = str(record.get("level", "") or "").upper()
        if event != "alert" or level not in ALERT_LEVELS:
            continue

        incident = incidents.get(fingerprint)
        if incident is None:
            incident = {
                "fingerprint": fingerprint,
                "status": "active",
                "level": level,
                "source": str(record.get("source", "") or ""),
                "message": str(record.get("message", "") or ""),
                "timestamp": str(record.get("timestamp") or _format_timestamp(epoch)),
                "ts": epoch,
                "first_seen": str(record.get("timestamp") or _format_timestamp(epoch)),
                "first_seen_ts": epoch,
                "last_seen": str(record.get("timestamp") or _format_timestamp(epoch)),
                "last_seen_ts": epoch,
                "occurrence_count": 0,
                "active_occurrence_count": 0,
                "episode_count": 1,
                "recovery_count": 0,
                "recovered_at": None,
                "recovered_ts": None,
                "recovery_message": None,
                "lifecycle_managed": lifecycle_managed,
                "_active_level_rank": 0,
            }
            incidents[fingerprint] = incident
        elif incident["status"] == "recovered":
            incident["status"] = "active"
            incident["active_occurrence_count"] = 0
            incident["episode_count"] += 1
            incident["_active_level_rank"] = 0
        if lifecycle_managed:
            incident["lifecycle_managed"] = True

        incident["occurrence_count"] += 1
        incident["active_occurrence_count"] += 1
        incident["source"] = str(record.get("source", "") or incident["source"])
        incident["message"] = str(record.get("message", "") or incident["message"])
        incident["timestamp"] = str(
            record.get("timestamp") or _format_timestamp(epoch)
        )
        incident["ts"] = epoch
        incident["last_seen"] = incident["timestamp"]
        incident["last_seen_ts"] = epoch
        if record.get("incident_key") is not None:
            incident["incident_key"] = str(record["incident_key"])

        level_rank = _LEVEL_RANK[level]
        if level_rank >= incident["_active_level_rank"]:
            incident["level"] = level
            incident["_active_level_rank"] = level_rank

    cutoff = (now if now is not None else time.time()) - max_age_hours * 3600.0
    resolved = [
        incident
        for incident in incidents.values()
        if (
            incident.get("status") == "active"
            and (
                bool(incident.get("lifecycle_managed"))
                or float(incident.get("last_seen_ts") or 0.0) >= cutoff
            )
        )
        or float(incident.get("recovered_ts") or 0.0) >= cutoff
    ]
    for incident in resolved:
        incident.pop("_active_level_rank", None)
    resolved.sort(
        key=lambda item: (
            float(
                item.get("recovered_ts")
                if item.get("status") == "recovered"
                else item.get("last_seen_ts")
                or 0.0
            ),
            str(item.get("fingerprint", "")),
        )
    )
    if limit is not None and limit > 0:
        resolved = resolved[-limit:]
    return resolved


def prune_alert_store(
    path: Path | str, max_age_hours: float, *, now: float | None = None
) -> int:
    """Compact the event stream without silently resolving active incidents.

    All in-window events are retained. For an older lifecycle-managed active
    incident, the newest alert observation is retained as a compact state
    anchor. A recent recovery also retains the newest alert that it closes, even
    when that alert predates the history window. Unmanaged observations and
    fully recovered incidents age out normally.

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
            current_time = now if now is not None else time.time()
            cutoff = current_time - max_age_hours * 3600.0
            records: list[dict] = []
            for obj in _iter_alert_records(path):
                epoch = _entry_epoch(obj)
                if epoch is None:
                    continue
                obj["ts"] = epoch
                records.append(obj)
            records.sort(key=lambda item: item.get("ts") or 0.0)

            keep_indices = {
                index
                for index, record in enumerate(records)
                if float(record.get("ts") or 0.0) >= cutoff
            }
            lifecycle: dict[str, dict[str, object]] = {}
            for index, record in enumerate(records):
                fingerprint = _fingerprint_for_record(record)
                if fingerprint is None:
                    continue
                state = lifecycle.setdefault(
                    fingerprint,
                    {
                        "active": False,
                        "lifecycle_managed": False,
                        "latest_alert_index": None,
                        "latest_recovery_ts": None,
                    },
                )
                event = str(record.get("event", "alert") or "alert").strip().lower()
                level = str(record.get("level", "") or "").upper()
                if _incident_key_for_record(record) is not None:
                    state["lifecycle_managed"] = True
                if event == "alert" and level in ALERT_LEVELS:
                    state["active"] = True
                    state["latest_alert_index"] = index
                elif event == "recovery":
                    state["active"] = False
                    state["latest_recovery_ts"] = float(record.get("ts") or 0.0)

            for state in lifecycle.values():
                latest_alert_index = state.get("latest_alert_index")
                if latest_alert_index is None:
                    continue
                recent_recovery = float(state.get("latest_recovery_ts") or 0.0) >= cutoff
                if (
                    bool(state.get("active"))
                    and bool(state.get("lifecycle_managed"))
                ) or recent_recovery:
                    keep_indices.add(int(latest_alert_index))

            kept = [records[index] for index in sorted(keep_indices)]
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
