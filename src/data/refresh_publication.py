"""Fail-closed staging and publication for scheduled raw-data refreshes.

The refresh pipeline builds a candidate tree away from the accepted raw-data
tree.  This module provides the narrow boundary between those trees:

* accepted artifacts can be copied into an empty candidate tree;
* candidate artifacts are fully staged and hashed before accepted data moves;
* every accepted target has a byte-for-byte rollback copy;
* the results/statistics pair manifest is published last; and
* a caller can publish its processed-data manifest inside the transaction.  If
  that caller step raises, every raw artifact is restored before the exception
  escapes.

Only explicitly declared relative file paths are accepted.  The helpers reject
absolute paths, parent traversal, overlapping roots, duplicate paths, and
symlink escapes from either declared root.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.data.io_utils import (
    CSV_PAIR_MANIFEST_FORMAT,
    CSV_PAIR_MANIFEST_VERSION,
    verify_csv_pair_manifest,
    write_json_atomically,
)


class RefreshPublicationError(RuntimeError):
    """Base error for raw refresh staging and publication."""


class ArtifactPathError(RefreshPublicationError, ValueError):
    """Raised when an artifact path escapes, overlaps, or is duplicated."""


class MissingArtifactError(RefreshPublicationError, FileNotFoundError):
    """Raised when a required accepted or candidate artifact is absent."""


class PublicationVerificationError(RefreshPublicationError):
    """Raised when copied or published bytes do not match their source."""


class PublicationRollbackError(RefreshPublicationError):
    """Raised when publication fails and exact rollback cannot be completed."""


@dataclass(frozen=True)
class RawArtifact:
    """One raw artifact, addressed relative to both declared roots."""

    relative_path: str
    required: bool = True


@dataclass(frozen=True)
class ResultsStatsPair:
    """The committed results/statistics pair and its last-published marker."""

    results_path: str
    stats_path: str
    manifest_path: str


@dataclass(frozen=True)
class PublicationPointer:
    """A prevalidated final pointer committed with the accepted raw snapshot."""

    target_path: Path | str
    content: bytes
    journal_path: Path | str | None = None


@dataclass(frozen=True)
class FileIdentity:
    """Content identity used in staging and publication receipts."""

    sha256: str
    bytes: int

    def as_dict(self) -> dict[str, object]:
        return {"sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True)
class StageArtifactReceipt:
    relative_path: str
    required: bool
    status: str
    accepted: FileIdentity | None
    candidate: FileIdentity | None

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "required": self.required,
            "status": self.status,
            "accepted": self.accepted.as_dict() if self.accepted else None,
            "candidate": self.candidate.as_dict() if self.candidate else None,
        }


@dataclass(frozen=True)
class StageReceipt:
    accepted_root: str
    candidate_root: str
    artifacts: tuple[StageArtifactReceipt, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted_root": self.accepted_root,
            "candidate_root": self.candidate_root,
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
        }


@dataclass(frozen=True)
class PublishedArtifactReceipt:
    relative_path: str
    required: bool
    status: str
    before: FileIdentity | None
    candidate: FileIdentity | None
    after: FileIdentity | None

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "required": self.required,
            "status": self.status,
            "before": self.before.as_dict() if self.before else None,
            "candidate": self.candidate.as_dict() if self.candidate else None,
            "after": self.after.as_dict() if self.after else None,
        }


@dataclass(frozen=True)
class CleanupFailure:
    """One non-fatal cleanup failure after a successful commit."""

    path: str
    error_type: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass(frozen=True)
class PointerPublicationReceipt:
    target_path: str
    before: FileIdentity | None
    after: FileIdentity

    def as_dict(self) -> dict[str, object]:
        return {
            "target_path": self.target_path,
            "before": self.before.as_dict() if self.before else None,
            "after": self.after.as_dict(),
        }


@dataclass
class PublicationReceipt:
    """Verified before/after identities for one publication attempt."""

    accepted_root: str
    candidate_root: str
    artifacts: tuple[PublishedArtifactReceipt, ...]
    commit_order: tuple[str, ...]
    pair_manifest_path: str
    status: str = "published"
    cleanup_status: str = "pending"
    cleanup_failures: tuple[CleanupFailure, ...] = ()
    pointer: PointerPublicationReceipt | None = None
    journal_path: str = ""
    recovery_action: str = "none"

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted_root": self.accepted_root,
            "candidate_root": self.candidate_root,
            "status": self.status,
            "cleanup_status": self.cleanup_status,
            "cleanup_failures": [failure.as_dict() for failure in self.cleanup_failures],
            "pointer": self.pointer.as_dict() if self.pointer else None,
            "journal_path": self.journal_path,
            "recovery_action": self.recovery_action,
            "commit_order": list(self.commit_order),
            "pair_manifest_path": self.pair_manifest_path,
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
        }


@dataclass(frozen=True)
class _PlannedArtifact:
    spec: RawArtifact
    relative_path: str
    accepted_path: Path
    candidate_path: Path


@dataclass
class _PreparedArtifact:
    plan: _PlannedArtifact
    before: FileIdentity | None
    candidate: FileIdentity
    after: FileIdentity
    staged_path: Path
    backup_path: Path | None


@dataclass
class _PreparedPointer:
    target_path: Path
    before: FileIdentity | None
    after: FileIdentity
    staged_path: Path
    backup_path: Path | None


_REPLACE_ATTEMPTS = 6
_REPLACE_BACKOFF_SECONDS = 0.1


def _identity_bytes(content: bytes) -> FileIdentity:
    return FileIdentity(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))


def _identity_file(path: Path) -> FileIdentity:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise PublicationVerificationError(f"Could not read artifact {path}: {exc}") from exc
    return FileIdentity(sha256=digest.hexdigest(), bytes=size)


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left)) == os.path.normcase(str(right))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_roots(accepted_root: Path | str, candidate_root: Path | str) -> tuple[Path, Path]:
    accepted = Path(accepted_root).resolve(strict=False)
    candidate = Path(candidate_root).resolve(strict=False)
    if _is_within(candidate, accepted) or _is_within(accepted, candidate):
        raise ArtifactPathError(
            "Accepted and candidate roots must be isolated and may not overlap: "
            f"accepted={accepted}, candidate={candidate}"
        )
    if accepted.exists() and not accepted.is_dir():
        raise ArtifactPathError(f"Accepted root is not a directory: {accepted}")
    if candidate.exists() and not candidate.is_dir():
        raise ArtifactPathError(f"Candidate root is not a directory: {candidate}")
    return accepted, candidate


def _normalize_relative_path(raw_path: str) -> str:
    path = Path(raw_path)
    if not raw_path or path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ArtifactPathError(f"Artifact path must be a non-empty relative file path: {raw_path!r}")
    return path.as_posix()


def _member(root: Path, relative_path: str, *, label: str) -> Path:
    lexical = root / Path(relative_path)
    resolved = lexical.resolve(strict=False)
    if not _is_within(resolved, root):
        raise ArtifactPathError(
            f"{label} artifact escapes its declared root: {relative_path!r} -> {resolved}"
        )
    return resolved


def _build_plan(
    accepted_root: Path | str,
    candidate_root: Path | str,
    artifacts: Sequence[RawArtifact],
    pair: ResultsStatsPair,
) -> tuple[Path, Path, tuple[_PlannedArtifact, ...], ResultsStatsPair]:
    accepted, candidate = _validate_roots(accepted_root, candidate_root)
    if not artifacts:
        raise ArtifactPathError("At least one raw artifact must be declared")

    planned: list[_PlannedArtifact] = []
    seen_accepted: set[str] = set()
    seen_candidate: set[str] = set()
    by_relative: dict[str, _PlannedArtifact] = {}
    for spec in artifacts:
        relative = _normalize_relative_path(spec.relative_path)
        accepted_path = _member(accepted, relative, label="Accepted")
        candidate_path = _member(candidate, relative, label="Candidate")
        accepted_key = os.path.normcase(str(accepted_path))
        candidate_key = os.path.normcase(str(candidate_path))
        relative_key = os.path.normcase(relative)
        if accepted_key in seen_accepted or candidate_key in seen_candidate or relative_key in by_relative:
            raise ArtifactPathError(f"Duplicate raw artifact path: {relative!r}")
        seen_accepted.add(accepted_key)
        seen_candidate.add(candidate_key)
        item = _PlannedArtifact(
            spec=spec,
            relative_path=relative,
            accepted_path=accepted_path,
            candidate_path=candidate_path,
        )
        by_relative[relative_key] = item
        planned.append(item)

    normalized_pair = ResultsStatsPair(
        results_path=_normalize_relative_path(pair.results_path),
        stats_path=_normalize_relative_path(pair.stats_path),
        manifest_path=_normalize_relative_path(pair.manifest_path),
    )
    pair_paths = (
        normalized_pair.results_path,
        normalized_pair.stats_path,
        normalized_pair.manifest_path,
    )
    if len({os.path.normcase(path) for path in pair_paths}) != 3:
        raise ArtifactPathError("Results, statistics, and pair-manifest paths must be distinct")
    for relative in pair_paths:
        item = by_relative.get(os.path.normcase(relative))
        if item is None:
            raise ArtifactPathError(f"Pair artifact is not in the declared artifact set: {relative}")
        if relative != normalized_pair.manifest_path and not item.spec.required:
            raise ArtifactPathError(f"Pair artifact must be required: {relative}")

    manifest_key = os.path.normcase(normalized_pair.manifest_path)
    ordered = [item for item in planned if os.path.normcase(item.relative_path) != manifest_key]
    ordered.append(by_relative[manifest_key])
    return accepted, candidate, tuple(ordered), normalized_pair


def _atomic_replace(source: Path, target: Path) -> None:
    """Replace one target, retrying transient Windows file locks."""
    source_arg: str | Path = source
    target_arg: str | Path = target
    if os.name == "nt":
        def _extended(path: Path) -> str:
            value = str(path.resolve(strict=False))
            if value.startswith("\\\\?\\"):
                return value
            if value.startswith("\\\\"):
                return "\\\\?\\UNC\\" + value[2:]
            return "\\\\?\\" + value

        source_arg = _extended(source)
        target_arg = _extended(target)
    delay = _REPLACE_BACKOFF_SECONDS
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(source_arg, target_arg)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(delay)
            delay *= 2


def _cleanup_path(path: Path) -> None:
    """Remove a transaction temporary; split out for focused failure testing."""
    path.unlink(missing_ok=True)


def publication_journal_path(pointer_path: Path | str) -> Path:
    """Return the stable recovery journal beside a production pointer."""
    target = Path(pointer_path).resolve(strict=False)
    return target.with_name(f".{target.name}.refresh-publication-journal.json")


def _publication_boundary(_name: str) -> None:
    """No-op crash boundary used by kill-equivalent recovery tests."""


def _prepare_pointer(pointer: PublicationPointer) -> tuple[_PreparedPointer, Path]:
    target = Path(pointer.target_path).resolve(strict=False)
    journal = (
        Path(pointer.journal_path).resolve(strict=False)
        if pointer.journal_path is not None
        else publication_journal_path(target)
    )
    if _same_path(target, journal):
        raise ArtifactPathError("Publication pointer and recovery journal must be distinct")
    if journal.parent != target.parent:
        raise ArtifactPathError("Publication recovery journal must be beside its pointer")
    if target.exists():
        _ensure_regular_file(target, label="Publication pointer")
    before = _identity_file(target) if target.exists() else None
    staged = _write_temp_bytes(target, bytes(pointer.content), suffix=".pointer.tmp")
    after = _identity_bytes(bytes(pointer.content))
    backup: Path | None = None
    try:
        if target.exists():
            backup, backup_identity = _copy_to_temp_verified(
                target, target, suffix=".pointer.rollback.bak"
            )
            current = _identity_file(target)
            if backup_identity != before or current != before:
                raise PublicationVerificationError(
                    f"Publication pointer changed during preparation: {target}"
                )
        return (
            _PreparedPointer(
                target_path=target,
                before=before,
                after=after,
                staged_path=staged,
                backup_path=backup,
            ),
            journal,
        )
    except BaseException:
        staged.unlink(missing_ok=True)
        if backup is not None:
            backup.unlink(missing_ok=True)
        raise


def _durable_backup(source: Path, destination: Path, expected: FileIdentity) -> None:
    temp, copied = _copy_to_temp_verified(source, destination, suffix=".journal.tmp")
    try:
        if copied != expected:
            raise PublicationVerificationError(
                f"Durable journal backup identity mismatch for {source}"
            )
        _atomic_replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def _journal_identity(identity: FileIdentity | None) -> dict[str, object] | None:
    return identity.as_dict() if identity is not None else None


def _create_publication_journal(
    *,
    journal_path: Path,
    accepted: Path,
    candidate: Path,
    prepared: Sequence[_PreparedArtifact],
    pair: ResultsStatsPair,
    pointer: _PreparedPointer,
) -> tuple[Path, dict[str, object]]:
    if journal_path.exists():
        raise RefreshPublicationError(
            f"Pending publication journal must be recovered before staging: {journal_path}"
        )
    # Keep this name deliberately short for Windows MAX_PATH headroom.  The
    # journal itself binds the directory's exact resolved path.
    backup_root = journal_path.parent / f".ufcr-{uuid4().hex[:12]}"
    backup_root.mkdir(parents=True, exist_ok=False)
    artifact_rows: list[dict[str, object]] = []
    try:
        for index, item in enumerate(prepared):
            durable_backup: Path | None = None
            if item.before is not None:
                if item.backup_path is None:
                    raise PublicationVerificationError(
                        f"Missing rollback snapshot for {item.plan.accepted_path}"
                    )
                durable_backup = backup_root / f"raw-{index:03d}.bak"
                _durable_backup(item.backup_path, durable_backup, item.before)
            artifact_rows.append(
                {
                    "relative_path": item.plan.relative_path,
                    "target_path": str(item.plan.accepted_path),
                    "before": _journal_identity(item.before),
                    "after": _journal_identity(item.after),
                    "backup_path": str(durable_backup) if durable_backup else None,
                }
            )
        pointer_backup: Path | None = None
        if pointer.before is not None:
            if pointer.backup_path is None:
                raise PublicationVerificationError(
                    f"Missing rollback snapshot for pointer {pointer.target_path}"
                )
            pointer_backup = backup_root / "pointer.bak"
            _durable_backup(pointer.backup_path, pointer_backup, pointer.before)
        payload: dict[str, object] = {
            "format": "ufc_refresh_publication_journal",
            "version": 1,
            "state": "prepared",
            "transaction_id": uuid4().hex,
            "accepted_root": str(accepted),
            "candidate_root": str(candidate),
            "backup_root": str(backup_root),
            "pair": {
                "results_path": pair.results_path,
                "stats_path": pair.stats_path,
                "manifest_path": pair.manifest_path,
            },
            "artifacts": artifact_rows,
            "pointer": {
                "target_path": str(pointer.target_path),
                "before": _journal_identity(pointer.before),
                "after": _journal_identity(pointer.after),
                "backup_path": str(pointer_backup) if pointer_backup else None,
            },
        }
        write_json_atomically(payload, journal_path)
        return backup_root, payload
    except BaseException:
        for path in backup_root.glob("*"):
            path.unlink(missing_ok=True)
        backup_root.rmdir()
        raise


def _identity_from_journal(value: object, *, label: str) -> FileIdentity | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise PublicationVerificationError(f"Journal {label} identity is not an object")
    sha256 = str(value.get("sha256") or "").lower()
    byte_count = value.get("bytes")
    if (
        len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
        or not isinstance(byte_count, int)
        or byte_count < 0
    ):
        raise PublicationVerificationError(f"Journal {label} identity is invalid")
    return FileIdentity(sha256=sha256, bytes=byte_count)


def _observed_identity(path: Path) -> FileIdentity | None:
    return _identity_file(path) if path.exists() else None


def _restore_durable_record(
    *,
    target: Path,
    before: FileIdentity | None,
    backup: Path | None,
) -> None:
    if before is None:
        target.unlink(missing_ok=True)
    else:
        if backup is None or not backup.is_file():
            raise PublicationRollbackError(f"Durable rollback backup is missing: {backup}")
        temp, copied = _copy_to_temp_verified(backup, target, suffix=".recover.tmp")
        try:
            if copied != before:
                raise PublicationRollbackError(
                    f"Durable rollback backup identity mismatch: {backup}"
                )
            _atomic_replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
    observed = _observed_identity(target)
    if observed != before:
        raise PublicationRollbackError(
            f"Recovered target identity mismatch for {target}: "
            f"expected={before}, actual={observed}"
        )


def _cleanup_recovered_journal(
    journal: Path,
    backup_root: Path,
) -> tuple[str, ...]:
    """Remove the journal first, then best-effort orphaned durable backups."""
    try:
        journal.unlink()
    except OSError as exc:
        return (f"{journal}:{type(exc).__name__}:{exc}",)
    failures: list[str] = []
    if backup_root.exists():
        for path in backup_root.iterdir():
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                failures.append(f"{path}:{type(exc).__name__}:{exc}")
        try:
            backup_root.rmdir()
        except OSError as exc:
            failures.append(f"{backup_root}:{type(exc).__name__}:{exc}")
    return tuple(failures)


def recover_pending_publication(
    journal_path: Path | str,
    *,
    force_rollback: bool = False,
) -> dict[str, object]:
    """Recover a killed raw-plus-pointer publication from its durable journal.

    A fully observed new raw snapshot, new pair manifest, and new processed
    pointer is rolled forward.  Every other boundary is rolled back to the
    exact pre-publication snapshot while an invalid pair guard keeps raw readers
    fail closed.  The operation is idempotent across repeated process kills.
    """
    journal = Path(journal_path).resolve(strict=False)
    if not journal.exists():
        return {
            "status": "none",
            "action": "none",
            "journal_path": str(journal),
            "cleanup_failures": [],
        }
    try:
        payload = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationVerificationError(
            f"Pending publication journal is unreadable: {journal}: {exc}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "ufc_refresh_publication_journal"
        or payload.get("version") != 1
    ):
        raise PublicationVerificationError(f"Unsupported publication journal: {journal}")

    accepted = Path(str(payload.get("accepted_root") or "")).resolve(strict=False)
    backup_root = Path(str(payload.get("backup_root") or "")).resolve(strict=False)
    if not str(payload.get("accepted_root") or "") or not str(payload.get("backup_root") or ""):
        raise PublicationVerificationError("Publication journal roots are missing")
    if backup_root.parent != journal.parent or not backup_root.name.startswith(".ufcr-"):
        raise ArtifactPathError(f"Unsafe publication journal backup root: {backup_root}")

    pair_payload = payload.get("pair")
    if not isinstance(pair_payload, dict):
        raise PublicationVerificationError("Publication journal pair metadata is missing")
    pair = ResultsStatsPair(
        results_path=_normalize_relative_path(str(pair_payload.get("results_path") or "")),
        stats_path=_normalize_relative_path(str(pair_payload.get("stats_path") or "")),
        manifest_path=_normalize_relative_path(str(pair_payload.get("manifest_path") or "")),
    )
    artifact_payload = payload.get("artifacts")
    if not isinstance(artifact_payload, list) or not artifact_payload:
        raise PublicationVerificationError("Publication journal artifact inventory is empty")
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw_record in enumerate(artifact_payload):
        if not isinstance(raw_record, dict):
            raise PublicationVerificationError(f"Journal artifact[{index}] is invalid")
        relative = _normalize_relative_path(str(raw_record.get("relative_path") or ""))
        target = _member(accepted, relative, label="Journal target")
        declared_target = Path(str(raw_record.get("target_path") or "")).resolve(strict=False)
        if not _same_path(target, declared_target):
            raise ArtifactPathError(f"Journal target does not match accepted root: {declared_target}")
        key = os.path.normcase(str(target))
        if key in seen:
            raise ArtifactPathError(f"Duplicate target in publication journal: {target}")
        seen.add(key)
        backup_raw = raw_record.get("backup_path")
        backup = Path(str(backup_raw)).resolve(strict=False) if backup_raw else None
        if backup is not None and not _is_within(backup, backup_root):
            raise ArtifactPathError(f"Journal backup escapes its root: {backup}")
        records.append(
            {
                "relative_path": relative,
                "target": target,
                "before": _identity_from_journal(raw_record.get("before"), label=f"artifact[{index}].before"),
                "after": _identity_from_journal(raw_record.get("after"), label=f"artifact[{index}].after"),
                "backup": backup,
            }
        )

    pointer_payload = payload.get("pointer")
    if not isinstance(pointer_payload, dict):
        raise PublicationVerificationError("Publication journal pointer metadata is missing")
    pointer_target = Path(str(pointer_payload.get("target_path") or "")).resolve(strict=False)
    if (
        pointer_target.parent != journal.parent
        or _same_path(pointer_target, journal)
        or _is_within(pointer_target, backup_root)
    ):
        raise ArtifactPathError(f"Unsafe publication pointer target: {pointer_target}")
    pointer_before = _identity_from_journal(pointer_payload.get("before"), label="pointer.before")
    pointer_after = _identity_from_journal(pointer_payload.get("after"), label="pointer.after")
    if pointer_after is None:
        raise PublicationVerificationError("Publication journal pointer.after is missing")
    pointer_backup_raw = pointer_payload.get("backup_path")
    pointer_backup = (
        Path(str(pointer_backup_raw)).resolve(strict=False) if pointer_backup_raw else None
    )
    if pointer_backup is not None and not _is_within(pointer_backup, backup_root):
        raise ArtifactPathError(f"Pointer backup escapes journal root: {pointer_backup}")

    all_new = all(
        record["after"] is not None
        and _observed_identity(record["target"]) == record["after"]
        for record in records
    ) and _observed_identity(pointer_target) == pointer_after
    if all_new:
        try:
            verify_csv_pair_manifest(
                _member(accepted, pair.results_path, label="Recovered results"),
                _member(accepted, pair.stats_path, label="Recovered statistics"),
                manifest_path=_member(accepted, pair.manifest_path, label="Recovered manifest"),
            )
        except Exception:
            all_new = False

    if force_rollback:
        all_new = False
    action = "roll_forward" if all_new else "roll_back"
    if not all_new:
        manifest_key = os.path.normcase(pair.manifest_path)
        manifest_record = next(
            (
                record
                for record in records
                if os.path.normcase(str(record["relative_path"])) == manifest_key
            ),
            None,
        )
        if manifest_record is None:
            raise PublicationVerificationError("Journal does not include the pair manifest")
        guard = _bootstrap_guard_bytes(accepted, pair)
        guard_temp = _write_temp_bytes(
            manifest_record["target"], guard, suffix=".recovery-guard.tmp"
        )
        try:
            _atomic_replace(guard_temp, manifest_record["target"])
        finally:
            guard_temp.unlink(missing_ok=True)

        try:
            for record in reversed(records):
                if record is manifest_record:
                    continue
                _restore_durable_record(
                    target=record["target"],
                    before=record["before"],
                    backup=record["backup"],
                )
            _restore_durable_record(
                target=pointer_target,
                before=pointer_before,
                backup=pointer_backup,
            )
            _restore_durable_record(
                target=manifest_record["target"],
                before=manifest_record["before"],
                backup=manifest_record["backup"],
            )
            for record in records:
                if _observed_identity(record["target"]) != record["before"]:
                    raise PublicationRollbackError(
                        f"Journal rollback verification failed: {record['target']}"
                    )
            if _observed_identity(pointer_target) != pointer_before:
                raise PublicationRollbackError(
                    f"Journal pointer rollback verification failed: {pointer_target}"
                )
        except BaseException as exc:
            raise PublicationRollbackError(
                f"Durable publication recovery failed; journal retained at {journal}"
            ) from exc

    cleanup_failures = _cleanup_recovered_journal(journal, backup_root)
    return {
        "status": "recovered",
        "action": action,
        "journal_path": str(journal),
        "cleanup_failures": list(cleanup_failures),
    }


def _write_temp_bytes(target: Path, content: bytes, *, suffix: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=target.parent, suffix=suffix)
    temp_path = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _copy_to_temp_verified(source: Path, target: Path, *, suffix: str) -> tuple[Path, FileIdentity]:
    before = _identity_file(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=target.parent, suffix=suffix)
    os.close(fd)
    temp_path = Path(name)
    try:
        shutil.copyfile(source, temp_path)
        copied = _identity_file(temp_path)
        after = _identity_file(source)
        if copied != before or after != before:
            raise PublicationVerificationError(
                f"Artifact changed while it was copied: {source}"
            )
        return temp_path, copied
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _ensure_regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise ArtifactPathError(f"{label} artifact may not be a symlink: {path}")
    if not path.is_file():
        raise MissingArtifactError(f"{label} artifact is not a regular file: {path}")


def _verify_existing_pair(root: Path, pair: ResultsStatsPair) -> None:
    if not _member(root, pair.manifest_path, label="Pair manifest").exists():
        return
    verify_csv_pair_manifest(
        _member(root, pair.results_path, label="Results"),
        _member(root, pair.stats_path, label="Statistics"),
        manifest_path=_member(root, pair.manifest_path, label="Pair manifest"),
    )


def _with_provider_reference_artifacts(artifacts, *roots):
    from src.data.provider_consumer_references import provider_reference_artifacts
    from src.data.observation_consumer_references import observation_reference_artifacts

    result = list(artifacts)
    declared = {artifact.relative_path for artifact in result}
    for root in roots:
        for relative in sorted(set(provider_reference_artifacts(Path(root))) | set(observation_reference_artifacts(Path(root)))):
            if relative not in declared:
                result.append(RawArtifact(relative, required=False))
                declared.add(relative)
    return tuple(result)


def stage_accepted_raw_artifacts(
    *,
    accepted_root: Path | str,
    candidate_root: Path | str,
    artifacts: Sequence[RawArtifact],
    pair: ResultsStatsPair,
) -> StageReceipt:
    """Copy declared accepted files into an isolated candidate raw-data tree.

    Required files are preflighted before any candidate file is created.
    Optional absences are returned as ``missing_optional`` receipt entries.
    Existing declared targets are rejected so staging never erases a candidate
    left by another run.
    """
    artifacts = _with_provider_reference_artifacts(artifacts, accepted_root)
    accepted, candidate, plan, normalized_pair = _build_plan(
        accepted_root, candidate_root, artifacts, pair
    )
    _verify_existing_pair(accepted, normalized_pair)

    for item in plan:
        if item.accepted_path.exists():
            _ensure_regular_file(item.accepted_path, label="Accepted")
        elif item.spec.required:
            raise MissingArtifactError(f"Required accepted artifact is missing: {item.accepted_path}")
        if item.candidate_path.exists():
            raise ArtifactPathError(
                f"Candidate artifact target already exists; use an isolated staging tree: "
                f"{item.candidate_path}"
            )

    manifest_bytes, manifest_was_synthesized = _rewritten_manifest_bytes(
        accepted,
        candidate,
        normalized_pair,
        allow_synthesize=True,
    )
    manifest_relative = os.path.normcase(normalized_pair.manifest_path)
    candidate.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    receipts: list[StageArtifactReceipt] = []
    try:
        for item in plan:
            is_manifest = os.path.normcase(item.relative_path) == manifest_relative
            if not item.accepted_path.exists() and not is_manifest:
                receipts.append(
                    StageArtifactReceipt(
                        relative_path=item.relative_path,
                        required=item.spec.required,
                        status="missing_optional",
                        accepted=None,
                        candidate=None,
                    )
                )
                continue
            accepted_identity = (
                _identity_file(item.accepted_path) if item.accepted_path.exists() else None
            )
            if is_manifest:
                temp_path = _write_temp_bytes(
                    item.candidate_path, manifest_bytes, suffix=".stage.tmp"
                )
                identity = _identity_bytes(manifest_bytes)
            else:
                temp_path, identity = _copy_to_temp_verified(
                    item.accepted_path, item.candidate_path, suffix=".stage.tmp"
                )
            try:
                _atomic_replace(temp_path, item.candidate_path)
            finally:
                temp_path.unlink(missing_ok=True)
            created.append(item.candidate_path)
            candidate_identity = _identity_file(item.candidate_path)
            if candidate_identity != identity:
                raise PublicationVerificationError(
                    f"Staged candidate bytes failed verification: {item.relative_path}"
                )
            receipts.append(
                StageArtifactReceipt(
                    relative_path=item.relative_path,
                    required=item.spec.required,
                    status=(
                        "synthesized"
                        if is_manifest and manifest_was_synthesized
                        else "staged"
                    ),
                    accepted=accepted_identity,
                    candidate=candidate_identity,
                )
            )
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise

    return StageReceipt(
        accepted_root=str(accepted),
        candidate_root=str(candidate),
        artifacts=tuple(receipts),
    )


def _stable_file_bytes(path: Path) -> bytes:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise PublicationVerificationError(f"Could not read artifact {path}: {exc}") from exc
    if _identity_file(path) != _identity_bytes(content):
        raise PublicationVerificationError(f"Artifact changed while it was read: {path}")
    return content


def _synthesized_pair_payload(source_root: Path, pair: ResultsStatsPair) -> dict[str, object]:
    source_manifest = _member(source_root, pair.manifest_path, label="Source manifest")
    if source_manifest.exists():
        raise PublicationVerificationError(
            f"Pair manifest appeared during legacy bootstrap: {source_manifest}"
        )
    records: list[dict[str, object]] = []
    for relative in (pair.results_path, pair.stats_path):
        path = _member(source_root, relative, label="Legacy pair artifact")
        content = _stable_file_bytes(path)
        try:
            frame = pd.read_csv(io.BytesIO(content))
        except Exception as exc:
            raise PublicationVerificationError(
                f"Could not parse legacy pair artifact {path}: {exc}"
            ) from exc
        identity = _identity_bytes(content)
        records.append(
            {
                "path": str(path.resolve(strict=False)),
                "sha256": identity.sha256,
                "bytes": identity.bytes,
                "rows": int(len(frame)),
                "schema": [str(column) for column in frame.columns],
            }
        )
    if source_manifest.exists():
        raise PublicationVerificationError(
            f"Pair manifest appeared during legacy bootstrap: {source_manifest}"
        )
    return {
        "format": CSV_PAIR_MANIFEST_FORMAT,
        "version": CSV_PAIR_MANIFEST_VERSION,
        "state": "committed",
        "artifacts": records,
    }


def _rewritten_manifest_bytes(
    source_root: Path,
    destination_root: Path,
    pair: ResultsStatsPair,
    *,
    allow_synthesize: bool = False,
) -> tuple[bytes, bool]:
    source_results = _member(source_root, pair.results_path, label="Source results")
    source_stats = _member(source_root, pair.stats_path, label="Source statistics")
    source_manifest = _member(source_root, pair.manifest_path, label="Source manifest")
    if source_manifest.exists():
        payload = verify_csv_pair_manifest(
            source_results,
            source_stats,
            manifest_path=source_manifest,
        )
        synthesized = False
    elif allow_synthesize:
        payload = _synthesized_pair_payload(source_root, pair)
        synthesized = True
    else:
        raise MissingArtifactError(
            f"Candidate results/statistics pair has no commit manifest: {source_manifest}"
        )
    rewritten = copy.deepcopy(payload)
    records = rewritten.get("artifacts")
    if not isinstance(records, list) or len(records) != 2:
        raise PublicationVerificationError(
            f"Source pair manifest has an invalid artifact list: {source_manifest}"
        )
    for record, relative in zip(records, (pair.results_path, pair.stats_path), strict=True):
        if not isinstance(record, dict):
            raise PublicationVerificationError(
                f"Source pair manifest has an invalid artifact record: {source_manifest}"
            )
        record["path"] = str(
            _member(destination_root, relative, label="Destination pair").resolve(False)
        )
    return (json.dumps(rewritten, indent=2) + "\n").encode("utf-8"), synthesized


def _prepare_publication(
    accepted: Path,
    candidate: Path,
    plan: tuple[_PlannedArtifact, ...],
    pair: ResultsStatsPair,
) -> tuple[list[_PreparedArtifact], tuple[PublishedArtifactReceipt, ...]]:
    _verify_existing_pair(accepted, pair)

    manifest_relative = os.path.normcase(pair.manifest_path)
    for item in plan:
        if item.candidate_path.exists():
            _ensure_regular_file(item.candidate_path, label="Candidate")
        elif item.spec.required or os.path.normcase(item.relative_path) == manifest_relative:
            raise MissingArtifactError(f"Required candidate artifact is missing: {item.candidate_path}")
        if item.accepted_path.exists():
            _ensure_regular_file(item.accepted_path, label="Accepted")

    manifest_bytes, _synthesized = _rewritten_manifest_bytes(candidate, accepted, pair)
    prepared: list[_PreparedArtifact] = []
    receipt_rows: list[PublishedArtifactReceipt] = []
    cleanup: list[Path] = []
    try:
        for item in plan:
            before = _identity_file(item.accepted_path) if item.accepted_path.exists() else None
            if not item.candidate_path.exists():
                receipt_rows.append(
                    PublishedArtifactReceipt(
                        relative_path=item.relative_path,
                        required=item.spec.required,
                        status=("preserved_optional" if before is not None else "missing_optional"),
                        before=before,
                        candidate=None,
                        after=before,
                    )
                )
                continue

            if os.path.normcase(item.relative_path) == manifest_relative:
                candidate_identity = _identity_file(item.candidate_path)
                staged_path = _write_temp_bytes(
                    item.accepted_path, manifest_bytes, suffix=".publish.tmp"
                )
                after = _identity_bytes(manifest_bytes)
            else:
                staged_path, candidate_identity = _copy_to_temp_verified(
                    item.candidate_path, item.accepted_path, suffix=".publish.tmp"
                )
                after = candidate_identity
            cleanup.append(staged_path)

            backup_path: Path | None = None
            if item.accepted_path.exists():
                backup_path, backup_identity = _copy_to_temp_verified(
                    item.accepted_path, item.accepted_path, suffix=".rollback.bak"
                )
                cleanup.append(backup_path)
                current = _identity_file(item.accepted_path)
                if backup_identity != before or current != before:
                    raise PublicationVerificationError(
                        f"Accepted artifact changed while rollback bytes were captured: "
                        f"{item.accepted_path}"
                    )
            prepared.append(
                _PreparedArtifact(
                    plan=item,
                    before=before,
                    candidate=candidate_identity,
                    after=after,
                    staged_path=staged_path,
                    backup_path=backup_path,
                )
            )
            receipt_rows.append(
                PublishedArtifactReceipt(
                    relative_path=item.relative_path,
                    required=item.spec.required,
                    status="replace" if before is not None else "create",
                    before=before,
                    candidate=candidate_identity,
                    after=after,
                )
            )
    except BaseException:
        for path in cleanup:
            path.unlink(missing_ok=True)
        raise
    return prepared, tuple(receipt_rows)


def _verify_published(prepared: Sequence[_PreparedArtifact], accepted: Path, pair: ResultsStatsPair) -> None:
    for item in prepared:
        observed = _identity_file(item.plan.accepted_path)
        if observed != item.after:
            raise PublicationVerificationError(
                f"Published artifact does not match staged bytes: {item.plan.accepted_path}"
            )
    verify_csv_pair_manifest(
        _member(accepted, pair.results_path, label="Accepted results"),
        _member(accepted, pair.stats_path, label="Accepted statistics"),
        manifest_path=_member(accepted, pair.manifest_path, label="Accepted manifest"),
    )


def _verify_before_snapshots(prepared: Sequence[_PreparedArtifact]) -> None:
    """Refuse to overwrite an accepted file that changed during preparation."""
    for item in prepared:
        observed = (
            _identity_file(item.plan.accepted_path)
            if item.plan.accepted_path.exists()
            else None
        )
        if observed != item.before:
            raise PublicationVerificationError(
                f"Accepted artifact changed before publication: {item.plan.accepted_path}; "
                f"expected={item.before}, actual={observed}"
            )


def _bootstrap_guard_bytes(accepted: Path, pair: ResultsStatsPair) -> bytes:
    """Build an intentionally non-committed pair marker for legacy bootstrap."""
    payload = {
        "format": CSV_PAIR_MANIFEST_FORMAT,
        "version": CSV_PAIR_MANIFEST_VERSION,
        "state": "updating",
        "artifacts": [
            {
                "path": str(
                    _member(accepted, relative, label="Bootstrap pair").resolve(False)
                )
            }
            for relative in (pair.results_path, pair.stats_path)
        ],
    }
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def _restore_item(item: _PreparedArtifact) -> None:
    target = item.plan.accepted_path
    if item.backup_path is None:
        target.unlink(missing_ok=True)
    else:
        _atomic_replace(item.backup_path, target)
        item.backup_path = None
    restored = _identity_file(target) if target.exists() else None
    if restored != item.before:
        raise PublicationVerificationError(
            f"Restored identity mismatch for {target}: expected={item.before}, "
            f"actual={restored}"
        )


def _rollback(
    attempted: Sequence[_PreparedArtifact],
    all_prepared: Sequence[_PreparedArtifact],
    *,
    manifest_relative_path: str,
    bootstrap_guard: bytes | None,
) -> None:
    errors: list[str] = []
    attempted_ids = {id(item) for item in attempted}
    manifest_key = os.path.normcase(manifest_relative_path)
    manifest_item = next(
        item
        for item in all_prepared
        if os.path.normcase(item.plan.relative_path) == manifest_key
    )
    bootstrap = manifest_item.before is None

    if bootstrap:
        if bootstrap_guard is None:
            errors.append("legacy bootstrap rollback has no fail-closed guard bytes")
        else:
            guard_temp: Path | None = None
            try:
                guard_temp = _write_temp_bytes(
                    manifest_item.plan.accepted_path,
                    bootstrap_guard,
                    suffix=".rollback-guard.tmp",
                )
                _atomic_replace(guard_temp, manifest_item.plan.accepted_path)
                if _identity_file(manifest_item.plan.accepted_path) != _identity_bytes(
                    bootstrap_guard
                ):
                    raise PublicationVerificationError(
                        "Legacy bootstrap rollback guard failed verification"
                    )
            except BaseException as exc:
                errors.append(
                    f"{manifest_item.plan.accepted_path}: could not install rollback guard: {exc}"
                )
            finally:
                if guard_temp is not None:
                    guard_temp.unlink(missing_ok=True)

    # With a legacy bootstrap, the invalid guard remains present while every
    # non-manifest file is restored.  With an existing manifest, reverse order
    # restores that old valid marker first and causes readers to reject the
    # still-new pair until both CSVs are back.
    restore_items = [
        item
        for item in reversed(attempted)
        if not bootstrap or os.path.normcase(item.plan.relative_path) != manifest_key
    ]
    for item in restore_items:
        target = item.plan.accepted_path
        try:
            _restore_item(item)
        except BaseException as exc:
            errors.append(f"{target}: {exc}")

    # Targets that were prepared but never attempted must still equal the
    # before snapshot; this catches an unexpected outside writer during commit.
    for item in all_prepared:
        if id(item) in attempted_ids:
            continue
        if bootstrap and os.path.normcase(item.plan.relative_path) == manifest_key:
            continue
        try:
            observed = (
                _identity_file(item.plan.accepted_path)
                if item.plan.accepted_path.exists()
                else None
            )
            if observed != item.before:
                errors.append(
                    f"{item.plan.accepted_path}: unattempted target changed "
                    f"(expected={item.before}, actual={observed})"
                )
        except BaseException as exc:
            errors.append(f"{item.plan.accepted_path}: {exc}")

    if bootstrap and not errors:
        try:
            manifest_item.plan.accepted_path.unlink(missing_ok=True)
            if manifest_item.plan.accepted_path.exists():
                raise PublicationVerificationError(
                    f"Bootstrap manifest still exists after rollback: "
                    f"{manifest_item.plan.accepted_path}"
                )
        except BaseException as exc:
            errors.append(f"{manifest_item.plan.accepted_path}: {exc}")

    if errors:
        raise PublicationRollbackError(
            "Raw publication failed and byte-exact rollback was incomplete: " + "; ".join(errors)
        )


def _cleanup_failure_from_text(path: Path, message: str) -> CleanupFailure:
    return CleanupFailure(
        path=str(path.resolve(strict=False)),
        error_type="JournalCleanupError",
        message=message,
    )


@contextmanager
def _publish_raw_and_pointer_transaction(
    *,
    accepted_root: Path | str,
    candidate_root: Path | str,
    artifacts: Sequence[RawArtifact],
    pair: ResultsStatsPair,
    pointer: PublicationPointer,
) -> Iterator[PublicationReceipt]:
    """Durably commit raw artifacts and their processed-generation pointer."""
    accepted, candidate, plan, normalized_pair = _build_plan(
        accepted_root, candidate_root, artifacts, pair
    )
    prepared, receipt_rows = _prepare_publication(
        accepted, candidate, plan, normalized_pair
    )
    prepared_pointer: _PreparedPointer | None = None
    guard_temp: Path | None = None
    journal: Path | None = None
    receipt: PublicationReceipt | None = None
    interrupted = False
    try:
        prepared_pointer, journal = _prepare_pointer(pointer)
        if journal.exists():
            raise RefreshPublicationError(
                f"Pending publication journal requires recovery before a new candidate: {journal}"
            )
        manifest_key = os.path.normcase(normalized_pair.manifest_path)
        manifest_item = next(
            item
            for item in prepared
            if os.path.normcase(item.plan.relative_path) == manifest_key
        )
        non_manifest = [item for item in prepared if item is not manifest_item]
        receipt = PublicationReceipt(
            accepted_root=str(accepted),
            candidate_root=str(candidate),
            artifacts=receipt_rows,
            commit_order=tuple(
                [item.plan.relative_path for item in non_manifest]
                + [f"pointer:{prepared_pointer.target_path}", manifest_item.plan.relative_path]
            ),
            pair_manifest_path=normalized_pair.manifest_path,
            pointer=PointerPublicationReceipt(
                target_path=str(prepared_pointer.target_path),
                before=prepared_pointer.before,
                after=prepared_pointer.after,
            ),
            journal_path=str(journal),
        )
        _verify_before_snapshots(prepared)
        if _observed_identity(prepared_pointer.target_path) != prepared_pointer.before:
            raise PublicationVerificationError(
                f"Publication pointer changed before commit: {prepared_pointer.target_path}"
            )
        _backup_root, _journal_payload = _create_publication_journal(
            journal_path=journal,
            accepted=accepted,
            candidate=candidate,
            prepared=prepared,
            pair=normalized_pair,
            pointer=prepared_pointer,
        )
        _publication_boundary("after_journal")

        guard = _bootstrap_guard_bytes(accepted, normalized_pair)
        guard_temp = _write_temp_bytes(
            manifest_item.plan.accepted_path,
            guard,
            suffix=".coordinated-guard.tmp",
        )
        _atomic_replace(guard_temp, manifest_item.plan.accepted_path)
        if _identity_file(manifest_item.plan.accepted_path) != _identity_bytes(guard):
            raise PublicationVerificationError(
                "Coordinated publication guard failed verification"
            )
        _publication_boundary("after_guard")

        for item in non_manifest:
            _atomic_replace(item.staged_path, item.plan.accepted_path)
            if _identity_file(item.plan.accepted_path) != item.after:
                raise PublicationVerificationError(
                    f"Replacement verification failed for {item.plan.accepted_path}"
                )
        _publication_boundary("after_raw_artifacts")

        _atomic_replace(prepared_pointer.staged_path, prepared_pointer.target_path)
        if _identity_file(prepared_pointer.target_path) != prepared_pointer.after:
            raise PublicationVerificationError(
                f"Pointer replacement verification failed: {prepared_pointer.target_path}"
            )
        _publication_boundary("after_pointer")

        _atomic_replace(manifest_item.staged_path, manifest_item.plan.accepted_path)
        _verify_published(prepared, accepted, normalized_pair)
        _publication_boundary("after_pair_manifest")

        try:
            yield receipt
        except (SystemExit, KeyboardInterrupt):
            interrupted = True
            receipt.status = "interrupted"
            receipt.cleanup_status = "recovery_required"
            raise
        except BaseException as caller_error:
            try:
                recovery = recover_pending_publication(journal, force_rollback=True)
                receipt.recovery_action = str(recovery["action"])
            except BaseException as recovery_error:
                receipt.status = "rollback_failed"
                raise PublicationRollbackError(
                    f"Coordinated publication rollback failed; journal retained at {journal}"
                ) from recovery_error
            receipt.status = "rolled_back"
            raise caller_error

        receipt.status = "committed"
        recovery = recover_pending_publication(journal)
        receipt.recovery_action = str(recovery["action"])
        journal_failures = list(recovery.get("cleanup_failures") or [])
        receipt.cleanup_failures = tuple(
            _cleanup_failure_from_text(journal, str(message))
            for message in journal_failures
        )
    except (SystemExit, KeyboardInterrupt):
        interrupted = True
        if receipt is not None:
            receipt.status = "interrupted"
            receipt.cleanup_status = "recovery_required"
        raise
    except BaseException as publish_error:
        if journal is not None and journal.exists():
            try:
                recovery = recover_pending_publication(journal, force_rollback=True)
                if receipt is not None:
                    receipt.status = "rolled_back"
                    receipt.recovery_action = str(recovery["action"])
            except BaseException as recovery_error:
                if receipt is not None:
                    receipt.status = "rollback_failed"
                raise PublicationRollbackError(
                    f"Coordinated publication rollback failed; journal retained at {journal}"
                ) from recovery_error
        raise publish_error
    finally:
        if not interrupted:
            cleanup_paths: list[Path] = []
            if guard_temp is not None:
                cleanup_paths.append(guard_temp)
            for item in prepared:
                cleanup_paths.append(item.staged_path)
                if item.backup_path is not None:
                    cleanup_paths.append(item.backup_path)
            if prepared_pointer is not None:
                cleanup_paths.append(prepared_pointer.staged_path)
                if prepared_pointer.backup_path is not None:
                    cleanup_paths.append(prepared_pointer.backup_path)

            if receipt is not None and receipt.status == "committed":
                failures = list(receipt.cleanup_failures)
                for path in cleanup_paths:
                    try:
                        _cleanup_path(path)
                    except OSError as exc:
                        failures.append(
                            CleanupFailure(
                                path=str(path.resolve(strict=False)),
                                error_type=type(exc).__name__,
                                message=str(exc),
                            )
                        )
                receipt.cleanup_failures = tuple(failures)
                receipt.cleanup_status = "incomplete" if failures else "complete"
            else:
                for path in cleanup_paths:
                    _cleanup_path(path)
                if receipt is not None and receipt.status != "rollback_failed":
                    receipt.cleanup_status = "complete"


@contextmanager
def publish_raw_artifacts_transaction(
    *,
    accepted_root: Path | str,
    candidate_root: Path | str,
    artifacts: Sequence[RawArtifact],
    pair: ResultsStatsPair,
    pointer: PublicationPointer | None = None,
) -> Iterator[PublicationReceipt]:
    """Publish candidate raw files and keep rollback live across caller work.

    The context is entered only after every candidate has been copied to a
    same-directory temporary file, every accepted target has a verified
    rollback copy, all replacements have succeeded, and the committed pair has
    verified.  The pair manifest is always the last replacement.

    Publish a processed-data manifest (or any other final commit marker) inside
    the ``with`` block.  An exception from that work restores all accepted raw
    targets byte-for-byte and then re-raises the original exception.  A rollback
    failure instead raises :class:`PublicationRollbackError` chained from the
    publication or caller failure.
    """
    artifacts = _with_provider_reference_artifacts(artifacts, accepted_root, candidate_root)
    if pointer is not None:
        with _publish_raw_and_pointer_transaction(
            accepted_root=accepted_root,
            candidate_root=candidate_root,
            artifacts=artifacts,
            pair=pair,
            pointer=pointer,
        ) as coordinated_receipt:
            yield coordinated_receipt
        return

    accepted, candidate, plan, normalized_pair = _build_plan(
        accepted_root, candidate_root, artifacts, pair
    )
    prepared, receipt_rows = _prepare_publication(
        accepted, candidate, plan, normalized_pair
    )
    manifest_key = os.path.normcase(normalized_pair.manifest_path)
    manifest_item = next(
        item
        for item in prepared
        if os.path.normcase(item.plan.relative_path) == manifest_key
    )
    bootstrap_guard = (
        _bootstrap_guard_bytes(accepted, normalized_pair)
        if manifest_item.before is None
        else None
    )
    bootstrap_started = False
    guard_temp: Path | None = None
    attempted: list[_PreparedArtifact] = []
    receipt = PublicationReceipt(
        accepted_root=str(accepted),
        candidate_root=str(candidate),
        artifacts=receipt_rows,
        commit_order=tuple(item.plan.relative_path for item in prepared),
        pair_manifest_path=normalized_pair.manifest_path,
    )
    try:
        try:
            _verify_before_snapshots(prepared)
            if bootstrap_guard is not None:
                guard_temp = _write_temp_bytes(
                    manifest_item.plan.accepted_path,
                    bootstrap_guard,
                    suffix=".bootstrap-guard.tmp",
                )
                bootstrap_started = True
                _atomic_replace(guard_temp, manifest_item.plan.accepted_path)
                if _identity_file(manifest_item.plan.accepted_path) != _identity_bytes(
                    bootstrap_guard
                ):
                    raise PublicationVerificationError(
                        "Legacy bootstrap publication guard failed verification"
                    )
            for item in prepared:
                attempted.append(item)
                _atomic_replace(item.staged_path, item.plan.accepted_path)
                observed = _identity_file(item.plan.accepted_path)
                if observed != item.after:
                    raise PublicationVerificationError(
                        f"Replacement verification failed for {item.plan.accepted_path}"
                    )
            _verify_published(prepared, accepted, normalized_pair)
        except BaseException as publish_error:
            if not attempted and not bootstrap_started:
                receipt.status = "not_published"
                raise
            try:
                _rollback(
                    attempted,
                    prepared,
                    manifest_relative_path=normalized_pair.manifest_path,
                    bootstrap_guard=bootstrap_guard,
                )
            except PublicationRollbackError as rollback_error:
                receipt.status = "rollback_failed"
                raise rollback_error from publish_error
            receipt.status = "rolled_back"
            raise

        try:
            yield receipt
        except BaseException as caller_error:
            try:
                _rollback(
                    attempted,
                    prepared,
                    manifest_relative_path=normalized_pair.manifest_path,
                    bootstrap_guard=bootstrap_guard,
                )
            except PublicationRollbackError as rollback_error:
                receipt.status = "rollback_failed"
                raise rollback_error from caller_error
            receipt.status = "rolled_back"
            raise
        receipt.status = "committed"
    finally:
        cleanup_paths: list[Path] = []
        if guard_temp is not None:
            cleanup_paths.append(guard_temp)
        for item in prepared:
            cleanup_paths.append(item.staged_path)
            if item.backup_path is not None and receipt.status != "rollback_failed":
                cleanup_paths.append(item.backup_path)

        if receipt.status == "committed":
            failures: list[CleanupFailure] = []
            for path in cleanup_paths:
                try:
                    _cleanup_path(path)
                except OSError as exc:
                    failures.append(
                        CleanupFailure(
                            path=str(path.resolve(strict=False)),
                            error_type=type(exc).__name__,
                            message=str(exc),
                        )
                    )
            receipt.cleanup_failures = tuple(failures)
            receipt.cleanup_status = "incomplete" if failures else "complete"
        else:
            # Failure-path cleanup remains strict.  In particular, a cleanup
            # error must never be mistaken for a successful rollback.
            for path in cleanup_paths:
                _cleanup_path(path)
            receipt.cleanup_status = (
                "recovery_backups_preserved"
                if receipt.status == "rollback_failed"
                else "complete"
            )


__all__ = [
    "ArtifactPathError",
    "CleanupFailure",
    "FileIdentity",
    "MissingArtifactError",
    "PublicationPointer",
    "PublicationReceipt",
    "PublicationRollbackError",
    "PublicationVerificationError",
    "RawArtifact",
    "RefreshPublicationError",
    "ResultsStatsPair",
    "StageReceipt",
    "publication_journal_path",
    "publish_raw_artifacts_transaction",
    "recover_pending_publication",
    "stage_accepted_raw_artifacts",
]
