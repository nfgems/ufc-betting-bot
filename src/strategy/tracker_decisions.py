"""Persistent decision and outcome history for the model tracker.

The tracker history used to live in the retired external-analysis module even
though the model tracker does not use an LLM.  Keeping it here lets the
model-only tracker retain its audit trail without that dependency.
"""

from __future__ import annotations

import json
import logging
import os
import threading

from src.config import DATA_DIR
from src.storage_retention import compact_file_tail

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = str(os.getenv(name, default) or "").strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = int(default)
    if minimum is not None:
        value = max(minimum, value)
    return value


# Preserve the existing path so historical Model Tracker records remain
# visible after removing the operator.  The directory name is legacy only.
TRACKER_DECISION_LOG_PATH = DATA_DIR / "operator" / "tracker_decision_log.jsonl"
TRACKER_DECISION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
TRACKER_DECISION_READ_LIMIT = _env_int(
    "TRACKER_DECISION_READ_LIMIT",
    25_000,
    minimum=1_000,
)
TRACKER_DECISION_LOG_MAX_BYTES = _env_int(
    "TRACKER_DECISION_LOG_MAX_BYTES",
    50 * 1024 * 1024,
    minimum=1,
)
_tracker_decision_log_lock = threading.Lock()


def log_tracker_decision(record: dict) -> None:
    """Append a Model Tracker decision/outcome record to its JSONL log."""
    try:
        with _tracker_decision_log_lock:
            with open(TRACKER_DECISION_LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
            reclaimed = compact_file_tail(
                TRACKER_DECISION_LOG_PATH,
                TRACKER_DECISION_LOG_MAX_BYTES,
            )
        if reclaimed:
            logger.info(
                "Compacted tracker decision history; reclaimed %.1f MiB",
                reclaimed / 1024 / 1024,
            )
    except Exception as exc:
        logger.error("Failed to log tracker decision: %s", exc)


def load_tracker_decision_log(
    limit: int | None = TRACKER_DECISION_READ_LIMIT,
) -> list[dict]:
    """Read recent tracker records in their original chronological order."""
    if not TRACKER_DECISION_LOG_PATH.exists():
        return []

    if limit is not None:
        record_limit = max(1, int(limit))
        records = []
        with open(TRACKER_DECISION_LOG_PATH, "rb") as handle:
            handle.seek(0, 2)
            position = handle.tell()
            pending = b""
            while position > 0 and len(records) < record_limit:
                read_size = min(64 * 1024, position)
                position -= read_size
                handle.seek(position)
                pending = handle.read(read_size) + pending
                lines = pending.split(b"\n")
                pending = lines[0]
                for raw_line in reversed(lines[1:]):
                    raw = raw_line.strip()
                    if not raw:
                        continue
                    try:
                        records.append(json.loads(raw.decode("utf-8")))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if len(records) >= record_limit:
                        break
            if position == 0 and pending.strip() and len(records) < record_limit:
                try:
                    records.append(json.loads(pending.decode("utf-8")))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
        records.reverse()
        return records

    records = []
    with open(TRACKER_DECISION_LOG_PATH, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
