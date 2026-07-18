"""Small, dependency-free helpers for bounding persistent runtime files."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


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
