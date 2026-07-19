"""Small, dependency-free helpers for bounding persistent runtime files."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable


def _last_complete_line(path: Path, *, block_size: int = 64 * 1024) -> bytes:
    """Return the final complete line, including a trailing newline."""
    size = path.stat().st_size
    if size <= 0:
        return b""

    pending = b""
    with path.open("rb") as handle:
        position = size
        while position > 0:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            pending = handle.read(read_size) + pending
            stripped = pending.rstrip(b"\r\n")
            boundary = stripped.rfind(b"\n")
            if boundary >= 0:
                return stripped[boundary + 1 :] + b"\n"
        return pending.rstrip(b"\r\n") + b"\n"


def compact_file_tail(
    path: str | os.PathLike[str],
    max_bytes: int,
    *,
    target_ratio: float = 0.8,
) -> int:
    """Atomically retain the newest complete lines when a file exceeds its cap.

    Returns the number of bytes reclaimed. A non-positive cap disables
    compaction. The target is below the hard cap to avoid rewriting the file on
    every subsequent append.
    """
    resolved = Path(path)
    cap = int(max_bytes)
    if cap <= 0:
        return 0

    try:
        original_stat = resolved.stat()
    except FileNotFoundError:
        return 0
    original_size = original_stat.st_size
    if original_size <= cap:
        return 0

    ratio = min(max(float(target_ratio), 0.1), 1.0)
    target_bytes = max(1, int(cap * ratio))
    start = max(0, original_size - target_bytes)
    with resolved.open("rb") as handle:
        handle.seek(start)
        if start:
            handle.readline()  # Drop the partial oldest line.
        retained = handle.read()

    if not retained:
        retained = _last_complete_line(resolved)

    fd, temporary_name = tempfile.mkstemp(
        dir=str(resolved.parent),
        prefix=f".{resolved.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(retained)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, stat.S_IMODE(original_stat.st_mode))
        os.replace(temporary_name, resolved)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise

    return max(0, original_size - len(retained))


def _snapshot_timestamp(value: object) -> datetime | None:
    """Parse a snapshot timestamp as UTC without guessing from its filename."""
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _retention_now(value: datetime | float | int | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime.fromtimestamp(float(value), tz=timezone.utc)


def prune_json_snapshot_history(
    directory: str | os.PathLike[str],
    pattern: str,
    *,
    retention_days: int,
    full_resolution_days: int,
    daily_keep: int = 1,
    timestamp_field: str = "snapshot_time",
    protected_paths: Iterable[str | os.PathLike[str]] = (),
    protect_latest: bool = True,
    protect_latest_matching: Callable[[dict], bool] | None = None,
    prefer_daily_matching: Callable[[dict], bool] | None = None,
    now: datetime | float | int | None = None,
) -> list[Path]:
    """Thin timestamped JSON snapshots while preserving safe fallbacks.

    Valid snapshots inside ``full_resolution_days`` are retained at their
    original cadence. Older snapshots are thinned to ``daily_keep`` per UTC
    day until ``retention_days`` and then removed. Malformed snapshots and
    snapshots without a parseable embedded timestamp are deliberately left in
    place: retention must never turn a read problem into silent data loss.

    The newest valid snapshot and the newest snapshot accepted by
    ``protect_latest_matching`` can be retained beyond the normal horizon.
    This lets callers preserve their current and last-known-good fallbacks.
    When ``prefer_daily_matching`` is supplied, matching snapshots are kept
    ahead of non-matching snapshots during daily thinning, with recency used
    to break ties. If no snapshot from a day matches, the newest snapshots are
    retained as usual.
    A non-positive retention period disables pruning.
    """
    retention = int(retention_days)
    if retention <= 0:
        return []

    root = Path(directory)
    if not root.exists():
        return []

    current = _retention_now(now)
    full_days = min(max(int(full_resolution_days), 0), retention)
    per_day = max(int(daily_keep), 0)
    retention_cutoff = current - timedelta(days=retention)
    full_resolution_cutoff = current - timedelta(days=full_days)

    entries: list[tuple[datetime, Path, dict]] = []
    for path in root.glob(pattern):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        captured_at = _snapshot_timestamp(payload.get(timestamp_field))
        if captured_at is None:
            continue
        entries.append((captured_at, path, payload))

    if not entries:
        return []

    protected = {Path(path) for path in protected_paths}
    if protect_latest:
        protected.add(max(entries, key=lambda item: item[0])[1])
    if protect_latest_matching is not None:
        matching = []
        for entry in entries:
            try:
                if protect_latest_matching(entry[2]):
                    matching.append(entry)
            except Exception:
                # A caller predicate must not make retention destructive.
                continue
        if matching:
            protected.add(max(matching, key=lambda item: item[0])[1])

    remove: set[Path] = set()
    daily_entries: dict[object, list[tuple[datetime, Path, dict]]] = {}
    for entry in entries:
        captured_at, path, _payload = entry
        if path in protected:
            continue
        if captured_at < retention_cutoff:
            remove.add(path)
            continue
        if captured_at >= full_resolution_cutoff:
            continue
        daily_entries.setdefault(captured_at.date(), []).append(entry)

    for day_entries in daily_entries.values():
        ordered_entries = sorted(day_entries, key=lambda item: item[0], reverse=True)
        if prefer_daily_matching is not None:
            preferred_entries = []
            other_entries = []
            predicate_failed = False
            for entry in ordered_entries:
                try:
                    is_preferred = bool(prefer_daily_matching(entry[2]))
                except Exception:
                    # A caller predicate must not make retention destructive.
                    predicate_failed = True
                    break
                (preferred_entries if is_preferred else other_entries).append(entry)
            if predicate_failed:
                continue
            ordered_entries = preferred_entries + other_entries

        for _captured_at, path, _payload in ordered_entries[per_day:]:
            if path not in protected:
                remove.add(path)

    removed: list[Path] = []
    for path in sorted(remove, key=lambda item: str(item)):
        try:
            path.unlink()
            removed.append(path)
        except FileNotFoundError:
            continue
        except OSError:
            # Best effort: preserving a file is safer than breaking collection.
            continue
    return removed
