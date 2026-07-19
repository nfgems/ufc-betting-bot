"""Compressed object-storage archive for expired line-history snapshots."""

from __future__ import annotations

import gzip
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path, PurePosixPath

from src.config import (
    DATA_DIR,
    LINE_HISTORY_ARCHIVE_ACCESS_KEY_ID,
    LINE_HISTORY_ARCHIVE_BUCKET,
    LINE_HISTORY_ARCHIVE_ENDPOINT,
    LINE_HISTORY_ARCHIVE_PREFIX,
    LINE_HISTORY_ARCHIVE_REGION,
    LINE_HISTORY_ARCHIVE_SECRET_ACCESS_KEY,
    LINE_HISTORY_ARCHIVE_URL_STYLE,
)


DEFAULT_RESTORE_DIR = DATA_DIR / "restored_line_history"
MAX_RESTORE_BYTES = 512 * 1024 * 1024
_RESTORE_CHUNK_BYTES = 1024 * 1024
_MAX_LIST_RESULTS = 10_000
_ARCHIVE_FILENAME_PATTERN = re.compile(
    r"(?P<category>odds|polymarket)_(?P<date>\d{8})(?:_|T)"
    r"(?P<time>\d{6})\.csv\.gz\Z"
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ArchivedSnapshot:
    """Compact metadata returned by an archive listing."""

    key: str
    compressed_size_bytes: int
    last_modified: datetime
    etag: str


@dataclass(frozen=True)
class ArchivePage:
    """One bounded archive listing and the cursor for the next listing."""

    objects: tuple[ArchivedSnapshot, ...]
    next_cursor: str | None


def archive_enabled() -> bool:
    """Return whether durable line-history archiving is configured."""
    return bool(LINE_HISTORY_ARCHIVE_BUCKET)


def _validate_archive_config() -> None:
    missing = [
        name
        for name, value in (
            ("LINE_HISTORY_ARCHIVE_BUCKET", LINE_HISTORY_ARCHIVE_BUCKET),
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
    _archive_prefix_parts()


def _archive_prefix_parts() -> tuple[str, ...]:
    """Return a validated configured prefix as POSIX path components."""
    prefix = LINE_HISTORY_ARCHIVE_PREFIX
    if not prefix:
        return ()
    if "\\" in prefix or prefix.startswith("/") or prefix.endswith("/"):
        raise RuntimeError("LINE_HISTORY_ARCHIVE_PREFIX must be a relative POSIX prefix")
    raw_parts = prefix.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise RuntimeError("LINE_HISTORY_ARCHIVE_PREFIX contains an unsafe path segment")
    return tuple(raw_parts)


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


def _require_archive_bucket() -> None:
    if not archive_enabled():
        raise RuntimeError("Line-history archive is not configured")


def _listing_prefix(
    *,
    category: str | None,
    year: int | None,
    month: int | None,
) -> str:
    if category not in {None, "odds", "polymarket"}:
        raise ValueError("category must be odds or polymarket")
    if year is not None:
        if category is None:
            raise ValueError("--year requires --category")
        if isinstance(year, bool) or not 1000 <= int(year) <= 9999:
            raise ValueError("year must be a four-digit year")
    if month is not None:
        if year is None:
            raise ValueError("--month requires --year")
        if isinstance(month, bool) or not 1 <= int(month) <= 12:
            raise ValueError("month must be between 1 and 12")

    segments = list(_archive_prefix_parts())
    if category is not None:
        segments.append(category)
    if year is not None:
        segments.append(f"{int(year):04d}")
    if month is not None:
        segments.append(f"{int(month):02d}")
    return "/".join(segments) + ("/" if segments else "")


def list_line_history_archive(
    *,
    category: str | None = None,
    year: int | None = None,
    month: int | None = None,
    limit: int = 100,
    cursor: str | None = None,
    client=None,
) -> ArchivePage:
    """List a bounded page of archived snapshots using S3 continuation tokens."""
    _require_archive_bucket()
    if isinstance(limit, bool) or not 1 <= int(limit) <= _MAX_LIST_RESULTS:
        raise ValueError(f"limit must be between 1 and {_MAX_LIST_RESULTS}")
    if cursor is not None and (not isinstance(cursor, str) or not cursor):
        raise ValueError("cursor must be a non-empty string")

    prefix = _listing_prefix(category=category, year=year, month=month)
    archive_client = _archive_client() if client is None else client
    objects: list[ArchivedSnapshot] = []
    continuation = cursor
    next_cursor: str | None = None

    while len(objects) < int(limit):
        request = {
            "Bucket": LINE_HISTORY_ARCHIVE_BUCKET,
            "Prefix": prefix,
            "MaxKeys": min(1000, int(limit) - len(objects)),
        }
        if continuation is not None:
            request["ContinuationToken"] = continuation
        response = archive_client.list_objects_v2(**request)

        for item in response.get("Contents", ()):
            key = str(item["Key"])
            # Ignore unrelated or malformed objects even if a bucket provider
            # returns entries outside the requested prefix.
            try:
                validate_line_history_archive_key(key)
            except ValueError:
                continue
            objects.append(
                ArchivedSnapshot(
                    key=key,
                    compressed_size_bytes=int(item.get("Size", 0)),
                    last_modified=item["LastModified"],
                    etag=str(item.get("ETag", "")).strip('"'),
                )
            )
            if len(objects) >= int(limit):
                break

        if not response.get("IsTruncated"):
            next_cursor = None
            break
        next_token = response.get("NextContinuationToken")
        if not next_token or next_token == continuation:
            raise RuntimeError("Archive listing was truncated without a usable next cursor")
        next_cursor = str(next_token)
        if len(objects) >= int(limit):
            break
        continuation = next_cursor

    return ArchivePage(objects=tuple(objects), next_cursor=next_cursor)


def validate_line_history_archive_key(key: str) -> str:
    """Validate an exact archive key and return its safe restored CSV basename."""
    if not isinstance(key, str) or not key or key != key.strip():
        raise ValueError("archive key must be a non-empty exact key")
    if "\\" in key or key.startswith("/") or "//" in key:
        raise ValueError("archive key must be a relative POSIX key")

    raw_parts = key.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("archive key contains an unsafe path segment")
    parts = PurePosixPath(key).parts
    prefix_parts = _archive_prefix_parts()
    if tuple(parts[: len(prefix_parts)]) != prefix_parts:
        raise ValueError("archive key is outside LINE_HISTORY_ARCHIVE_PREFIX")
    remainder = parts[len(prefix_parts) :]
    if len(remainder) != 4:
        raise ValueError("archive key does not match the line-history key schema")

    category, year, month, filename = remainder
    match = _ARCHIVE_FILENAME_PATTERN.fullmatch(filename)
    if category not in {"odds", "polymarket"} or match is None:
        raise ValueError("archive key is not an odds or Polymarket CSV snapshot")
    if match.group("category") != category:
        raise ValueError("archive key category does not match its filename")
    if not re.fullmatch(r"\d{4}", year) or not 1000 <= int(year) <= 9999:
        raise ValueError("archive key year must contain four digits")
    if not re.fullmatch(r"(?:0[1-9]|1[0-2])", month):
        raise ValueError("archive key month must be between 01 and 12")
    try:
        datetime.strptime(match.group("date") + match.group("time"), "%Y%m%d%H%M%S")
    except ValueError as exc:
        raise ValueError("archive key filename contains an invalid timestamp") from exc
    return filename.removesuffix(".gz")


def _metadata_int(metadata: dict[str, str], name: str) -> int | None:
    raw = metadata.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Archive object has invalid {name} metadata") from exc
    if value < 0:
        raise ValueError(f"Archive object has invalid {name} metadata")
    return value


def restore_line_history_snapshot(
    key: str,
    *,
    output_dir: Path | None = None,
    overwrite: bool = False,
    max_uncompressed_bytes: int = MAX_RESTORE_BYTES,
    client=None,
) -> Path:
    """Download and safely restore one archived CSV without touching the object."""
    _require_archive_bucket()
    restored_name = validate_line_history_archive_key(key)
    if isinstance(max_uncompressed_bytes, bool) or int(max_uncompressed_bytes) <= 0:
        raise ValueError("max_uncompressed_bytes must be positive")

    destination_dir = Path(output_dir or DEFAULT_RESTORE_DIR).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    if not destination_dir.is_dir():
        raise NotADirectoryError(f"Restore output is not a directory: {destination_dir}")
    destination = destination_dir / restored_name
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            raise FileExistsError(
                f"Restore destination already exists: {destination}; use --force to replace it"
            )
        if destination.is_dir():
            raise IsADirectoryError(f"Restore destination is a directory: {destination}")

    archive_client = _archive_client() if client is None else client
    response = archive_client.get_object(
        Bucket=LINE_HISTORY_ARCHIVE_BUCKET,
        Key=key,
    )
    body = response.get("Body")
    if body is None or not hasattr(body, "read"):
        raise RuntimeError("Archive provider returned an object without a readable body")

    content_encoding = str(response.get("ContentEncoding", "gzip") or "").lower()
    if content_encoding != "gzip":
        close = getattr(body, "close", None)
        if close is not None:
            close()
        raise ValueError("Archive object is not gzip encoded")

    try:
        metadata = {
            str(name).lower(): str(value)
            for name, value in (response.get("Metadata") or {}).items()
        }
        expected_size = _metadata_int(metadata, "source-size-bytes")
        source_mtime_ns = _metadata_int(metadata, "source-mtime-ns")
        expected_sha256 = metadata.get("source-sha256")
        if expected_sha256 is not None:
            expected_sha256 = expected_sha256.lower()
            if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
                raise ValueError("Archive object has invalid source-sha256 metadata")
        if expected_size is not None and expected_size > int(max_uncompressed_bytes):
            raise ValueError("Archived snapshot exceeds the restore size limit")
    except Exception:
        close = getattr(body, "close", None)
        if close is not None:
            close()
        raise

    temporary_path: Path | None = None
    published = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{restored_name}.",
            suffix=".tmp",
            dir=destination_dir,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            restored_size = 0
            digest = hashlib.sha256()
            with gzip.GzipFile(fileobj=body, mode="rb") as decompressed:
                while True:
                    chunk = decompressed.read(_RESTORE_CHUNK_BYTES)
                    if not chunk:
                        break
                    restored_size += len(chunk)
                    if restored_size > int(max_uncompressed_bytes):
                        raise ValueError("Archived snapshot exceeds the restore size limit")
                    temporary.write(chunk)
                    digest.update(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())

        if expected_size is not None and restored_size != expected_size:
            raise ValueError("Restored snapshot size does not match archive metadata")
        if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
            raise ValueError("Restored snapshot checksum does not match archive metadata")
        if source_mtime_ns is not None:
            os.utime(temporary_path, ns=(source_mtime_ns, source_mtime_ns))

        if overwrite:
            os.replace(temporary_path, destination)
        else:
            # A hard link publishes the already-validated inode atomically and
            # fails instead of overwriting if another process won the race.
            os.link(temporary_path, destination)
            temporary_path.unlink()
        published = True
        return destination
    finally:
        close = getattr(body, "close", None)
        if close is not None:
            close()
        if not published and temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def archive_line_history_snapshot(path: Path, *, client=None) -> str:
    """Gzip and upload one snapshot, returning its deterministic object key."""
    if not archive_enabled():
        raise RuntimeError("Line-history archive is not configured")

    resolved = Path(path)
    source_stat = resolved.stat()
    object_key = _archive_key(resolved)
    # Keep the writer and operator tooling on one strict schema. If an
    # unexpected filename reaches the pruner, raising here makes pruning fail
    # closed and preserves the local file instead of creating an object that
    # the supported list/restore commands cannot recover.
    validate_line_history_archive_key(object_key)
    source_bytes = resolved.read_bytes()
    final_source_stat = resolved.stat()
    if (
        source_stat.st_size != final_source_stat.st_size
        or source_stat.st_mtime_ns != final_source_stat.st_mtime_ns
    ):
        raise RuntimeError(f"Line-history snapshot changed while archiving: {resolved}")
    compressed = gzip.compress(source_bytes, compresslevel=6, mtime=0)
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
            "source-sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
    )
    return object_key
