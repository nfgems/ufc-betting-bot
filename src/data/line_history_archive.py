"""Compressed object-storage archive for expired line-history snapshots."""

from __future__ import annotations

import gzip
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from src.config import (
    LINE_HISTORY_ARCHIVE_ACCESS_KEY_ID,
    LINE_HISTORY_ARCHIVE_BUCKET,
    LINE_HISTORY_ARCHIVE_ENDPOINT,
    LINE_HISTORY_ARCHIVE_PREFIX,
    LINE_HISTORY_ARCHIVE_REGION,
    LINE_HISTORY_ARCHIVE_SECRET_ACCESS_KEY,
    LINE_HISTORY_ARCHIVE_URL_STYLE,
)


def archive_enabled() -> bool:
    """Return whether durable line-history archiving is configured."""
    return bool(LINE_HISTORY_ARCHIVE_BUCKET)


def _validate_archive_config() -> None:
    missing = [
        name
        for name, value in (
            ("LINE_HISTORY_ARCHIVE_ENDPOINT", LINE_HISTORY_ARCHIVE_ENDPOINT),
            ("LINE_HISTORY_ARCHIVE_ACCESS_KEY_ID", LINE_HISTORY_ARCHIVE_ACCESS_KEY_ID),
            (
                "LINE_HISTORY_ARCHIVE_SECRET_ACCESS_KEY",
                LINE_HISTORY_ARCHIVE_SECRET_ACCESS_KEY,
            ),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Line-history archive is enabled but missing: " + ", ".join(missing)
        )
    if LINE_HISTORY_ARCHIVE_URL_STYLE not in {"auto", "path", "virtual"}:
        raise RuntimeError(
            "LINE_HISTORY_ARCHIVE_URL_STYLE must be auto, path, or virtual"
        )


@lru_cache(maxsize=1)
def _archive_client():
    _validate_archive_config()
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=LINE_HISTORY_ARCHIVE_ENDPOINT,
        region_name=LINE_HISTORY_ARCHIVE_REGION,
        aws_access_key_id=LINE_HISTORY_ARCHIVE_ACCESS_KEY_ID,
        aws_secret_access_key=LINE_HISTORY_ARCHIVE_SECRET_ACCESS_KEY,
        config=Config(
            connect_timeout=5,
            read_timeout=30,
            retries={"max_attempts": 3, "mode": "standard"},
            s3={"addressing_style": LINE_HISTORY_ARCHIVE_URL_STYLE},
        ),
    )


def _archive_key(path: Path) -> str:
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    category = "polymarket" if path.name.startswith("polymarket_") else "odds"
    segments = [
        LINE_HISTORY_ARCHIVE_PREFIX,
        category,
        modified.strftime("%Y"),
        modified.strftime("%m"),
        f"{path.name}.gz",
    ]
    return "/".join(segment for segment in segments if segment)


def archive_line_history_snapshot(path: Path, *, client=None) -> str:
    """Gzip and upload one snapshot, returning its deterministic object key."""
    if not archive_enabled():
        raise RuntimeError("Line-history archive is not configured")

    resolved = Path(path)
    source_stat = resolved.stat()
    object_key = _archive_key(resolved)
    compressed = gzip.compress(resolved.read_bytes(), compresslevel=6, mtime=0)
    archive_client = _archive_client() if client is None else client
    archive_client.put_object(
        Bucket=LINE_HISTORY_ARCHIVE_BUCKET,
        Key=object_key,
        Body=compressed,
        ContentType="text/csv",
        ContentEncoding="gzip",
        Metadata={
            "source-size-bytes": str(source_stat.st_size),
            "source-mtime-ns": str(source_stat.st_mtime_ns),
        },
    )
    return object_key
