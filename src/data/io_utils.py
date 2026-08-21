"""Small I/O helpers for safe data artifact writes."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

_REPLACE_ATTEMPTS = 6
_REPLACE_BACKOFF_SECONDS = 0.5
CSV_PAIR_MANIFEST_FORMAT = "csv_pair_commit"
CSV_PAIR_MANIFEST_VERSION = 1


class CSVPairIntegrityError(RuntimeError):
    """Raised when two CSVs do not match their committed pair manifest."""


def _replace_with_retry(tmp_path: Path, target: Path) -> None:
    # On Windows, antivirus/indexer scans briefly hold freshly written files,
    # making os.replace fail with a transient PermissionError.
    delay = _REPLACE_BACKOFF_SECONDS
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp_path, target)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(delay)
            delay *= 2


def csv_pair_manifest_path(first_path: Path | str, second_path: Path | str) -> Path:
    """Return the stable commit marker stored beside a related CSV pair."""
    first = Path(first_path)
    second = Path(second_path)
    return first.parent / f"{first.stem}--{second.stem}.pair-commit.json"


def _read_csv_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise CSVPairIntegrityError(f"Could not read CSV pair artifact {path}: {exc}") from exc


def _parse_csv_bytes(path: Path, content: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise CSVPairIntegrityError(f"Could not parse CSV pair artifact {path}: {exc}") from exc


def _csv_record(path: Path, content: bytes, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "path": str(path.resolve(strict=False)),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "rows": int(len(frame)),
        "schema": [str(column) for column in frame.columns],
    }


def _manifest_payload(
    artifacts: Sequence[tuple[Path, bytes, pd.DataFrame]],
    *,
    state: str = "committed",
) -> dict[str, object]:
    return {
        "format": CSV_PAIR_MANIFEST_FORMAT,
        "version": CSV_PAIR_MANIFEST_VERSION,
        "state": state,
        "artifacts": [_csv_record(path, content, frame) for path, content, frame in artifacts],
    }


def _validate_manifest(
    manifest_path: Path,
    manifest_bytes: bytes,
    actual: Sequence[dict[str, object]],
) -> dict[str, object]:
    try:
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as exc:
        raise CSVPairIntegrityError(f"Could not parse CSV pair manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CSVPairIntegrityError(f"CSV pair manifest {manifest_path} is not an object")
    expected_header = (
        payload.get("format") == CSV_PAIR_MANIFEST_FORMAT
        and payload.get("version") == CSV_PAIR_MANIFEST_VERSION
        and payload.get("state") == "committed"
    )
    if not expected_header:
        raise CSVPairIntegrityError(
            f"CSV pair manifest {manifest_path} is not a supported committed manifest"
        )
    declared = payload.get("artifacts")
    if not isinstance(declared, list) or len(declared) != 2:
        raise CSVPairIntegrityError(
            f"CSV pair manifest {manifest_path} must describe exactly two artifacts"
        )
    mismatches: list[str] = []
    for index, (expected, observed) in enumerate(zip(declared, actual, strict=True)):
        if not isinstance(expected, dict):
            mismatches.append(f"artifact[{index}] metadata is not an object")
            continue
        for field in ("path", "sha256", "bytes", "rows", "schema"):
            if expected.get(field) != observed[field]:
                mismatches.append(
                    f"artifact[{index}].{field} expected={expected.get(field)!r} "
                    f"actual={observed[field]!r}"
                )
    if mismatches:
        raise CSVPairIntegrityError(
            f"CSV pair manifest mismatch for {manifest_path}: " + "; ".join(mismatches)
        )
    return payload


def verify_csv_pair_manifest(
    first_path: Path | str,
    second_path: Path | str,
    *,
    manifest_path: Path | str | None = None,
) -> dict[str, object] | None:
    """Verify a committed pair, permitting a legacy pair with no manifest."""
    first, second = Path(first_path), Path(second_path)
    manifest = Path(manifest_path) if manifest_path else csv_pair_manifest_path(first, second)
    if not manifest.exists():
        return None
    before = manifest.read_bytes()
    records: list[dict[str, object]] = []
    for path in (first, second):
        content = _read_csv_bytes(path)
        records.append(_csv_record(path, content, _parse_csv_bytes(path, content)))
    try:
        after = manifest.read_bytes()
    except OSError as exc:
        raise CSVPairIntegrityError(
            f"CSV pair manifest {manifest} disappeared during verification: {exc}"
        ) from exc
    if after != before:
        raise CSVPairIntegrityError(f"CSV pair manifest {manifest} changed during verification")
    return _validate_manifest(manifest, before, records)


def read_csv_pair_verified(
    first_path: Path | str,
    second_path: Path | str,
    *,
    manifest_path: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read exact byte snapshots, failing closed around an interleaved commit."""
    first, second = Path(first_path), Path(second_path)
    manifest = Path(manifest_path) if manifest_path else csv_pair_manifest_path(first, second)
    try:
        manifest_before = manifest.read_bytes()
    except FileNotFoundError:
        manifest_before = None
    except OSError as exc:
        raise CSVPairIntegrityError(f"Could not read CSV pair manifest {manifest}: {exc}") from exc

    first_bytes = _read_csv_bytes(first)
    second_bytes = _read_csv_bytes(second)

    try:
        manifest_after = manifest.read_bytes()
    except FileNotFoundError:
        manifest_after = None
    except OSError as exc:
        raise CSVPairIntegrityError(f"Could not reread CSV pair manifest {manifest}: {exc}") from exc
    if manifest_before is not None and manifest_after != manifest_before:
        state = "disappeared" if manifest_after is None else "changed"
        raise CSVPairIntegrityError(f"CSV pair manifest {manifest} {state} during verified read")

    first_frame = _parse_csv_bytes(first, first_bytes)
    second_frame = _parse_csv_bytes(second, second_bytes)
    observed_manifest = manifest_before if manifest_before is not None else manifest_after
    if observed_manifest is not None:
        _validate_manifest(
            manifest,
            observed_manifest,
            [
                _csv_record(first, first_bytes, first_frame),
                _csv_record(second, second_bytes, second_frame),
            ],
        )
    return first_frame, second_frame


def write_csv_atomically(
    df: pd.DataFrame,
    path: Path | str,
    *,
    refuse_empty: bool = False,
) -> Path:
    """Write a CSV through a temp file and rename it into place.

    When ``refuse_empty`` is true, an empty dataframe is rejected before any
    existing artifact is replaced.
    """
    target = Path(path)
    if refuse_empty and df.empty:
        raise ValueError(f"Refusing to overwrite {target} with an empty dataframe")

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_csv(tmp_path, index=False)
        _replace_with_retry(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return target


def write_csvs_atomically(
    writes: Sequence[tuple[pd.DataFrame, Path | str]],
    *,
    refuse_empty: bool = False,
    manifest_path: Path | str | None = None,
) -> tuple[Path, ...]:
    """Commit related CSVs together and publish their manifest last.

    Existing targets are backed up before replacement. If either CSV or the
    final manifest cannot be replaced, every changed target is restored before
    the original exception is raised.
    """
    entries = [(frame, Path(path)) for frame, path in writes]
    if not entries:
        return ()
    manifest = Path(manifest_path) if manifest_path is not None else None
    if manifest is not None and len(entries) != 2:
        raise ValueError("A CSV pair commit manifest requires exactly two writes")

    normalized: set[str] = set()
    for frame, target in entries:
        if refuse_empty and frame.empty:
            raise ValueError(f"Refusing to overwrite {target} with an empty dataframe")
        identity = os.path.normcase(str(target.resolve(strict=False)))
        if identity in normalized:
            raise ValueError(f"Duplicate grouped CSV target: {target}")
        normalized.add(identity)
    if manifest is not None and os.path.normcase(str(manifest.resolve(strict=False))) in normalized:
        raise ValueError(f"CSV pair manifest cannot replace a CSV target: {manifest}")

    staged: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    committed: list[Path] = []
    preserve_backups: set[Path] = set()
    staged_manifest: Path | None = None
    manifest_backup: Path | None = None
    manifest_created_for_commit = False
    commit_started = False
    manifest_replace_started = False
    try:
        staged_artifacts: list[tuple[Path, bytes, pd.DataFrame]] = []
        for frame, target in entries:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(dir=target.parent, suffix=".pair.tmp")
            os.close(fd)
            staged[target] = Path(name)
            frame.to_csv(staged[target], index=False)
            content = staged[target].read_bytes()
            staged_artifacts.append((target, content, _parse_csv_bytes(target, content)))

        if manifest is not None:
            manifest.parent.mkdir(parents=True, exist_ok=True)
            if manifest.exists():
                verify_csv_pair_manifest(entries[0][1], entries[1][1], manifest_path=manifest)
            else:
                exists = [target.exists() for _frame, target in entries]
                if any(exists) and not all(exists):
                    raise CSVPairIntegrityError(
                        "Cannot bootstrap a CSV pair manifest when only one target exists"
                    )
                manifest_created_for_commit = True
                if all(exists):
                    current: list[tuple[Path, bytes, pd.DataFrame]] = []
                    for _frame, target in entries:
                        content = _read_csv_bytes(target)
                        current.append((target, content, _parse_csv_bytes(target, content)))
                    write_json_atomically(_manifest_payload(current), manifest)
                else:
                    write_json_atomically(
                        {
                            "format": CSV_PAIR_MANIFEST_FORMAT,
                            "version": CSV_PAIR_MANIFEST_VERSION,
                            "state": "initializing",
                            "artifacts": [
                                {"path": str(target.resolve(strict=False))}
                                for _frame, target in entries
                            ],
                        },
                        manifest,
                    )

            fd, name = tempfile.mkstemp(dir=manifest.parent, suffix=".rollback.bak")
            os.close(fd)
            manifest_backup = Path(name)
            shutil.copyfile(manifest, manifest_backup)
            fd, name = tempfile.mkstemp(dir=manifest.parent, suffix=".manifest.tmp")
            os.close(fd)
            staged_manifest = Path(name)
            staged_manifest.write_text(
                json.dumps(_manifest_payload(staged_artifacts), indent=2) + "\n",
                encoding="utf-8",
            )

        for _frame, target in entries:
            if target.exists():
                fd, name = tempfile.mkstemp(dir=target.parent, suffix=".rollback.bak")
                os.close(fd)
                backups[target] = Path(name)
                shutil.copyfile(target, backups[target])
            else:
                backups[target] = None

        try:
            commit_started = True
            for _frame, target in entries:
                _replace_with_retry(staged[target], target)
                committed.append(target)
            if manifest is not None and staged_manifest is not None:
                manifest_replace_started = True
                _replace_with_retry(staged_manifest, manifest)
        except Exception as commit_error:
            rollback_errors: list[str] = []
            if manifest_replace_started and manifest_backup is not None:
                try:
                    _replace_with_retry(manifest_backup, manifest)
                    manifest_backup = None
                except Exception as exc:
                    if manifest_backup.exists():
                        preserve_backups.add(manifest_backup)
                    rollback_errors.append(f"{manifest}: {exc}")
            for target in reversed(committed):
                try:
                    backup = backups[target]
                    if backup is None:
                        target.unlink(missing_ok=True)
                    else:
                        _replace_with_retry(backup, target)
                except Exception as exc:
                    if backup is not None and backup.exists():
                        preserve_backups.add(backup)
                    rollback_errors.append(f"{target}: {exc}")
            if manifest_created_for_commit and not rollback_errors:
                try:
                    manifest.unlink(missing_ok=True)
                except Exception as exc:
                    rollback_errors.append(f"{manifest}: {exc}")
            if rollback_errors:
                raise RuntimeError(
                    "Grouped CSV/manifest replacement failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from commit_error
            raise
        return tuple(target for _frame, target in entries)
    finally:
        if (
            manifest is not None
            and manifest_created_for_commit
            and not commit_started
        ):
            manifest.unlink(missing_ok=True)
        cleanup = [
            *staged.values(),
            *[
                path
                for path in backups.values()
                if path is not None and path not in preserve_backups
            ],
        ]
        if staged_manifest is not None:
            cleanup.append(staged_manifest)
        if manifest_backup is not None and manifest_backup not in preserve_backups:
            cleanup.append(manifest_backup)
        for path in cleanup:
            path.unlink(missing_ok=True)


def write_json_atomically(
    payload: object,
    path: Path | str,
    *,
    indent: int = 2,
) -> Path:
    """Write JSON through a temp file and rename it into place."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=indent) + "\n",
            encoding="utf-8",
        )
        _replace_with_retry(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return target


def copy_file_atomically(src: Path | str, dst: Path | str) -> Path:
    """Copy a file through a temp file and rename it into place."""
    source = Path(src)
    target = Path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copyfile(source, tmp_path)
        _replace_with_retry(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return target
