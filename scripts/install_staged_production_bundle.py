"""Install an approved staging bundle into an immutable, atomic bundle store.

The store deliberately keeps artifacts in versioned release directories.  A
single small ``active_bundle.json`` file contains both the active and rollback
release pointers; replacing that file is the only activation operation.  This
avoids publishing models, processed data, and their manifest independently.

This tool has no default source or target paths.  Initializing a store,
promoting a release, and rolling back are separate explicit commands.  It does
not deploy, restart a service, or modify repository canonical artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import sys
import tempfile
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


STORE_MARKER_NAME = ".production_bundle_store.json"
STATE_NAME = "active_bundle.json"
LOCK_NAME = ".production_bundle_store.lock"
RELEASES_DIR_NAME = "releases"
LOOKUPS_DIR_NAME = "lookups"
RECEIPT_NAME = "install_receipt.json"
INSTALLED_MANIFEST_NAME = "manifest.json"
SOURCE_MANIFEST_NAMES = ("staging_manifest.json", "manifest.json")
STORE_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 2
RECEIPT_SCHEMA_VERSION = 1
STORE_KIND = "ufc-production-bundle-store"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BUNDLE_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,180}$")
logger = logging.getLogger(__name__)

_MODEL_FILENAMES = {
    "primary": "xgboost_model.pkl",
    "no_odds": "xgboost_no_odds_model.pkl",
    "logistic": "logistic_model.pkl",
}
_LOOKUP_FILENAMES = ("fights_cleaned.csv", "features.csv")
_RUNTIME_MANIFEST_NAME = "runtime_manifest.json"
_LOOKUP_ACTIVE_ROLE = "active_mutable"
_LOOKUP_ROLLBACK_ROLE = "rollback_immutable_snapshot"
_LOOKUP_BINDING_RUNTIME_MANIFEST = "runtime_manifest"
_LOOKUP_BINDING_RELEASE_SEED = "release_seed"
_CORE_PATH_FIELDS = {
    "model_path": Path("models/xgboost_model.pkl"),
    "no_odds_model_path": Path("models/xgboost_no_odds_model.pkl"),
    "logistic_model_path": Path("models/logistic_model.pkl"),
    "processed_dir": Path("processed"),
}


class BundleInstallError(RuntimeError):
    """Raised when an isolated bundle operation cannot be proven safe."""


@dataclass(frozen=True)
class ValidatedSource:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    payload: dict[str, Any]

    @property
    def bundle_id(self) -> str:
        return str(self.payload["bundle_id"])


@dataclass(frozen=True)
class ValidatedLegacyCapture:
    source_manifest: Path
    runtime_manifest: Path
    capture_manifest: Path
    primary_model: Path
    no_odds_model: Path
    logistic_model: Path
    saved_spec: Path
    processed_fights: Path
    processed_features: Path
    payload: dict[str, Any]
    source_manifest_sha256: str

    @property
    def bundle_id(self) -> str:
        return str(self.payload["bundle_id"])


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise BundleInstallError(f"{label} is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleInstallError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BundleInstallError(f"{label} must contain a JSON object: {path}")
    return payload


def _write_json_file(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Durably replace one pointer/marker file in its existing directory."""
    parent = path.parent.resolve(strict=True)
    if parent.is_symlink():
        raise BundleInstallError(f"Refusing atomic write through symlink: {parent}")
    temporary = parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            _fsync_directory(parent)
        except OSError as exc:
            raise BundleInstallError(
                "Atomic pointer replacement completed, but directory durability could not "
                f"be confirmed; inspect {path} before any retry: {exc}"
            ) from exc
        if path.read_bytes() != _canonical_json_bytes(payload):
            raise BundleInstallError(
                f"Atomic pointer replacement read-back failed; inspect before retry: {path}"
            )
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _safe_relative_path(raw: object, *, label: str) -> Path:
    text = str(raw or "").strip()
    candidate = Path(text)
    if not text or candidate.is_absolute() or ".." in candidate.parts:
        raise BundleInstallError(f"{label} is not a safe relative path: {raw!r}")
    return candidate


def _confined_file(root: Path, relative: object, *, label: str) -> Path:
    rel = _safe_relative_path(relative, label=label)
    candidate = root / rel
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise BundleInstallError(f"{label} escapes or is missing from {root}: {rel}") from exc
    if _is_link_like(candidate) or not resolved.is_file():
        raise BundleInstallError(f"{label} is not a regular confined file: {candidate}")
    return resolved


def _is_link_like(path: Path) -> bool:
    """Reject symlinks and Windows directory junction/reparse indirection."""
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction):
        try:
            if is_junction():
                return True
        except OSError:
            return True
    # ``Path.is_junction`` was added after Python 3.11.  Railway verification
    # also runs on Windows hosts where a junction is represented by the
    # reparse-point file attribute, so use lstat rather than following it.
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, FileNotFoundError, OSError):
        return False
    if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
        return True
    return False


def _assert_path_components_no_links(path: Path, *, label: str) -> None:
    """Reject indirection in every existing component before canonicalization."""
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        # Check link identity independently of exists(): a broken symlink has
        # exists()==False but can still redirect a target created beneath it.
        if _is_link_like(current):
            raise BundleInstallError(
                f"{label} traverses a symlink, junction, or reparse point: {current}"
            )


def _assert_no_symlinks(root: Path, *, label: str) -> None:
    if _is_link_like(root):
        raise BundleInstallError(f"{label} root cannot be a symlink or junction: {root}")
    for path in root.rglob("*"):
        if _is_link_like(path):
            raise BundleInstallError(f"{label} contains a symlink or junction: {path}")


def _resolve_source_manifest(root: Path) -> Path:
    present = [root / name for name in SOURCE_MANIFEST_NAMES if (root / name).is_file()]
    if len(present) != 1:
        raise BundleInstallError(
            "Source root must contain exactly one staging_manifest.json or manifest.json: "
            f"{root}"
        )
    return present[0]


def _require_manifest_layout(
    root: Path,
    payload: dict[str, Any],
    *,
    require_core_paths: bool = True,
) -> None:
    if payload.get("manifest_version") != 3 or payload.get("staging_schema_version") != 1:
        raise BundleInstallError("Source must be a rich manifest_version=3 staging bundle")

    bundle_id = str(payload.get("bundle_id") or "")
    if not _BUNDLE_ID_RE.fullmatch(bundle_id):
        raise BundleInstallError(
            "bundle_id must use only letters, digits, period, underscore, or hyphen"
        )

    for field, relative in _CORE_PATH_FIELDS.items():
        expected = (root / relative).resolve(strict=True)
        if require_core_paths:
            declared = Path(str(payload.get(field) or ""))
            try:
                actual = declared.resolve(strict=True)
            except OSError as exc:
                raise BundleInstallError(f"Manifest core path {field} is missing: {declared}") from exc
            if actual != expected:
                raise BundleInstallError(
                    f"Manifest core path {field} is not confined to its source bundle: "
                    f"expected {expected}, got {actual}"
                )

    model_artifacts = payload.get("model_artifacts")
    if not isinstance(model_artifacts, dict) or set(model_artifacts) != set(_MODEL_FILENAMES):
        raise BundleInstallError("Manifest must describe exactly primary, no_odds, and logistic models")
    for label, filename in _MODEL_FILENAMES.items():
        record = model_artifacts[label]
        if not isinstance(record, dict) or record.get("staged_path") != f"models/{filename}":
            raise BundleInstallError(f"Manifest {label} model path is not exact")
        _confined_file(root, record["staged_path"], label=f"{label} model")

    saved_spec = payload.get("saved_fullfit_spec")
    if not isinstance(saved_spec, dict):
        raise BundleInstallError("Manifest saved_fullfit_spec is missing")
    _confined_file(root, saved_spec.get("staged_path"), label="saved full-fit spec")

    snapshot = payload.get("immutable_training_snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("immutable") is not True:
        raise BundleInstallError("Manifest immutable_training_snapshot is missing")
    for label, expected in (
        ("fights", "processed/fights_cleaned.csv"),
        ("features", "processed/features.csv"),
    ):
        record = snapshot.get(label)
        if not isinstance(record, dict) or record.get("staged_path") != expected:
            raise BundleInstallError(f"Manifest immutable {label} path is not exact")
        _confined_file(root, expected, label=f"processed {label}")
    test_set = snapshot.get("test_set")
    if not isinstance(test_set, dict):
        raise BundleInstallError("Manifest test-set record is missing")
    _confined_file(root, test_set.get("staged_path"), label="test set")
    _confined_file(root, test_set.get("metadata_staged_path"), label="test-set metadata")

    source_identity = payload.get("source_identity")
    if not isinstance(source_identity, dict):
        raise BundleInstallError("Manifest source_identity is missing")
    for key in ("pretraining_inventory_artifact", "assembly_inventory_artifact"):
        record = source_identity.get(key)
        if not isinstance(record, dict):
            raise BundleInstallError(f"Manifest {key} is missing")
        _confined_file(root, record.get("staged_path"), label=key)

    raw_provenance = payload.get("raw_input_provenance")
    bfo = raw_provenance.get("bfo_ledger") if isinstance(raw_provenance, dict) else None
    if not isinstance(bfo, dict):
        raise BundleInstallError("Manifest BFO provenance ledger is missing")
    _confined_file(root, bfo.get("staged_path"), label="BFO provenance ledger")

    evidence = payload.get("selection_evidence")
    evidence_files = evidence.get("files") if isinstance(evidence, dict) else None
    if not isinstance(evidence_files, list) or not evidence_files:
        raise BundleInstallError("Manifest selection evidence is missing")
    for index, record in enumerate(evidence_files):
        if not isinstance(record, dict):
            raise BundleInstallError(f"Selection evidence record {index} is invalid")
        _confined_file(root, record.get("staged_path"), label=f"selection evidence {index}")

    invocation = payload.get("training_invocation")
    audit = invocation.get("independent_audit_snapshot") if isinstance(invocation, dict) else None
    if not isinstance(audit, dict):
        raise BundleInstallError("Manifest independent audit snapshot is missing")
    for label in ("fights", "features"):
        record = audit.get(label)
        if not isinstance(record, dict):
            raise BundleInstallError(f"Manifest independent audit {label} is missing")
        _confined_file(root, record.get("staged_path"), label=f"independent audit {label}")

    previous = payload.get("previous_rollback_identity")
    readyz = previous.get("readyz_evidence") if isinstance(previous, dict) else None
    if not isinstance(readyz, dict):
        raise BundleInstallError("Manifest previous rollback readiness evidence is missing")
    _confined_file(root, readyz.get("staged_path"), label="previous readiness evidence")


def _required_rich_source_files(payload: dict[str, Any], manifest_name: str) -> set[str]:
    """Return the exact portable file allowlist declared by a rich v3 bundle."""
    snapshot = payload["immutable_training_snapshot"]
    source = payload["source_identity"]
    audit = payload["training_invocation"]["independent_audit_snapshot"]
    required = {
        manifest_name,
        "models/xgboost_model.pkl",
        "models/xgboost_no_odds_model.pkl",
        "models/logistic_model.pkl",
        str(payload["saved_fullfit_spec"]["staged_path"]),
        str(snapshot["fights"]["staged_path"]),
        str(snapshot["features"]["staged_path"]),
        str(snapshot["test_set"]["staged_path"]),
        str(snapshot["test_set"]["metadata_staged_path"]),
        str(source["pretraining_inventory_artifact"]["staged_path"]),
        str(source["assembly_inventory_artifact"]["staged_path"]),
        str(payload["raw_input_provenance"]["bfo_ledger"]["staged_path"]),
        str(audit["fights"]["staged_path"]),
        str(audit["features"]["staged_path"]),
        str(payload["previous_rollback_identity"]["readyz_evidence"]["staged_path"]),
    }
    required.update(
        str(record["staged_path"])
        for record in payload["selection_evidence"]["files"]
    )
    normalized: set[str] = set()
    for index, raw in enumerate(sorted(required)):
        relative = _safe_relative_path(raw, label=f"rich source allowlist item {index}")
        normalized.add(relative.as_posix())
    return normalized


def _assert_exact_source_files(
    root: Path,
    *,
    expected_files: set[str],
    label: str,
) -> None:
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected_directories = {"."}
    for relative in expected_files:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    actual_directories: set[str] = set()
    special: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            actual_directories.add(relative)
        elif not path.is_file():
            special.append(relative)
    missing = sorted(expected_files - actual)
    extra = sorted(actual - expected_files)
    extra_directories = sorted(actual_directories - expected_directories)
    if missing or extra or extra_directories or special:
        raise BundleInstallError(
            f"{label} must contain exactly its manifest-declared files; "
            f"missing={missing}, extra={extra}, extra_directories={extra_directories}, "
            f"special={special}"
        )


def _portable_semantic_validate_source(source: ValidatedSource) -> None:
    """Validate a moved bundle without trusting its machine-specific core paths."""
    with tempfile.TemporaryDirectory(prefix="ufc-bundle-source-validation-") as temp_dir:
        validation_root = Path(temp_dir).resolve(strict=True)
        shutil.copytree(
            source.root,
            validation_root,
            dirs_exist_ok=True,
            copy_function=shutil.copy2,
        )
        validation_manifest = validation_root / source.manifest_path.name
        rebased = _rebase_manifest(source.payload, validation_root)
        _write_json_file(validation_manifest, rebased)
        _require_manifest_layout(validation_root, rebased, require_core_paths=True)
        _semantic_validate_manifest(validation_manifest, validation_root, rich=True)


def _verify_declared_file_identity(
    root: Path,
    record: Mapping[str, object],
    *,
    staged_path_key: str,
    sha256_key: str,
    bytes_key: str,
    label: str,
) -> None:
    path = _confined_file(root, record.get(staged_path_key), label=label)
    expected_sha = str(record.get(sha256_key) or "").lower()
    expected_bytes = record.get(bytes_key)
    if not _SHA256_RE.fullmatch(expected_sha) or not isinstance(expected_bytes, int):
        raise BundleInstallError(f"{label} has an incomplete declared identity")
    if _sha256_file(path) != expected_sha or int(path.stat().st_size) != expected_bytes:
        raise BundleInstallError(f"{label} does not match its declared identity")


def _verify_rich_file_identities(root: Path, payload: dict[str, Any]) -> None:
    """Generically bind every copied rich-v3 file without selection-specific constants."""
    artifacts = payload["model_artifacts"]
    for label in _MODEL_FILENAMES:
        _verify_declared_file_identity(
            root,
            artifacts[label],
            staged_path_key="staged_path",
            sha256_key="sha256",
            bytes_key="bytes",
            label=f"{label} model",
        )
    core_hash_fields = {
        "primary": "model_sha256",
        "no_odds": "no_odds_model_sha256",
        "logistic": "logistic_model_sha256",
    }
    for label, field in core_hash_fields.items():
        if artifacts[label].get("sha256") != payload.get(field):
            raise BundleInstallError(f"{label} model rich identity disagrees with {field}")

    saved_spec = payload["saved_fullfit_spec"]
    _verify_declared_file_identity(
        root,
        saved_spec,
        staged_path_key="staged_path",
        sha256_key="sha256",
        bytes_key="bytes",
        label="saved full-fit spec",
    )
    if saved_spec.get("payload") is not None:
        spec_path = _confined_file(root, saved_spec["staged_path"], label="saved full-fit spec")
        if _load_json_object(spec_path, label="saved full-fit spec") != saved_spec["payload"]:
            raise BundleInstallError("Saved full-fit spec payload does not match its staged file")

    snapshot = payload["immutable_training_snapshot"]
    for label, core_sha, core_bytes in (
        ("fights", "processed_fights_sha256", "processed_fights_bytes"),
        ("features", "processed_features_sha256", "processed_features_bytes"),
    ):
        record = snapshot[label]
        _verify_declared_file_identity(
            root,
            record,
            staged_path_key="staged_path",
            sha256_key="sha256",
            bytes_key="bytes",
            label=f"processed {label}",
        )
        if record.get("sha256") != payload.get(core_sha) or record.get("bytes") != payload.get(
            core_bytes
        ):
            raise BundleInstallError(f"Processed {label} identity disagrees with core manifest")
    test_set = snapshot["test_set"]
    _verify_declared_file_identity(
        root,
        test_set,
        staged_path_key="staged_path",
        sha256_key="sha256",
        bytes_key="bytes",
        label="test set",
    )
    _verify_declared_file_identity(
        root,
        test_set,
        staged_path_key="metadata_staged_path",
        sha256_key="metadata_sha256",
        bytes_key="metadata_bytes",
        label="test-set metadata",
    )

    source = payload["source_identity"]
    for key in ("pretraining_inventory_artifact", "assembly_inventory_artifact"):
        _verify_declared_file_identity(
            root,
            source[key],
            staged_path_key="staged_path",
            sha256_key="sha256",
            bytes_key="bytes",
            label=key,
        )
    _verify_declared_file_identity(
        root,
        payload["raw_input_provenance"]["bfo_ledger"],
        staged_path_key="staged_path",
        sha256_key="sha256",
        bytes_key="bytes",
        label="BFO provenance ledger",
    )
    for index, record in enumerate(payload["selection_evidence"]["files"]):
        _verify_declared_file_identity(
            root,
            record,
            staged_path_key="staged_path",
            sha256_key="sha256",
            bytes_key="bytes",
            label=f"selection evidence {index}",
        )
    audit = payload["training_invocation"]["independent_audit_snapshot"]
    for label in ("fights", "features"):
        _verify_declared_file_identity(
            root,
            audit[label],
            staged_path_key="staged_path",
            sha256_key="sha256",
            bytes_key="bytes",
            label=f"independent audit {label}",
        )
    _verify_declared_file_identity(
        root,
        payload["previous_rollback_identity"]["readyz_evidence"],
        staged_path_key="staged_path",
        sha256_key="sha256",
        bytes_key="bytes",
        label="previous readiness evidence",
    )


def _semantic_validate_manifest(manifest_path: Path, root: Path, *, rich: bool) -> None:
    """Run generic strict validators without mutating canonical files."""
    try:
        from src.model.production_bundle import load_production_bundle, validate_production_bundle

        payload = _load_json_object(manifest_path, label="Bundle manifest")
        if rich:
            _verify_rich_file_identities(root, payload)
        bundle = load_production_bundle(manifest_path)
        validate_production_bundle(
            bundle,
            expected_models_dir=root / "models",
            expected_processed_dir=root / "processed",
        )
    except Exception as exc:
        raise BundleInstallError(f"Strict bundle validation failed: {exc}") from exc


def _validate_source(
    source_root: Path,
    *,
    expected_bundle_id: str,
    expected_manifest_sha256: str,
) -> ValidatedSource:
    if not _SHA256_RE.fullmatch(expected_manifest_sha256):
        raise BundleInstallError("Expected source manifest SHA-256 must be 64 lowercase hex characters")
    _assert_path_components_no_links(source_root, label="Source root")
    if _is_link_like(source_root):
        raise BundleInstallError(f"Source root cannot be a symlink or junction: {source_root}")
    root = source_root.resolve(strict=True)
    if not root.is_dir() or _is_link_like(root):
        raise BundleInstallError(f"Source root must be a real directory: {root}")
    _assert_no_symlinks(root, label="Source bundle")
    manifest_path = _resolve_source_manifest(root)
    actual_sha256 = _sha256_file(manifest_path)
    if actual_sha256 != expected_manifest_sha256:
        raise BundleInstallError(
            "Source manifest hash mismatch: "
            f"expected {expected_manifest_sha256}, got {actual_sha256}"
        )
    payload = _load_json_object(manifest_path, label="Source staging manifest")
    if payload.get("bundle_id") != expected_bundle_id:
        raise BundleInstallError(
            f"Source bundle_id mismatch: expected {expected_bundle_id!r}, "
            f"got {payload.get('bundle_id')!r}"
        )
    # Core paths in the signed staging manifest may describe the machine where
    # it was assembled.  Every deployable file is instead located by its safe
    # staged_path and exact hash; a private validation copy is rebased below.
    _require_manifest_layout(root, payload, require_core_paths=False)
    expected_files = _required_rich_source_files(payload, manifest_path.name)
    _assert_exact_source_files(
        root,
        expected_files=expected_files,
        label="Source staging bundle",
    )
    source = ValidatedSource(root, manifest_path, actual_sha256, payload)
    _portable_semantic_validate_source(source)
    return source


def _validated_explicit_file(path: Path, *, expected_sha256: str, label: str) -> Path:
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise BundleInstallError(f"Expected {label} SHA-256 is invalid")
    _assert_path_components_no_links(path, label=label)
    if _is_link_like(path):
        raise BundleInstallError(f"{label} cannot be a symlink or junction: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or _is_link_like(resolved):
        raise BundleInstallError(f"{label} must be an explicit regular file: {resolved}")
    actual = _sha256_file(resolved)
    if actual != expected_sha256:
        raise BundleInstallError(
            f"{label} hash mismatch: expected {expected_sha256}, got {actual}"
        )
    return resolved


def _validate_legacy_capture(
    *,
    source_manifest: Path,
    runtime_manifest: Path,
    capture_manifest: Path,
    primary_model: Path,
    no_odds_model: Path,
    logistic_model: Path,
    saved_spec: Path,
    processed_fights: Path,
    processed_features: Path,
    expected_bundle_id: str,
    expected_model_spec_name: str,
    expected_snapshot_max_event_date: str,
    expected_source_manifest_sha256: str,
    expected_runtime_manifest_sha256: str,
    expected_capture_manifest_sha256: str,
    expected_saved_spec_sha256: str,
    expected_primary_model_sha256: str,
    expected_no_odds_model_sha256: str,
    expected_logistic_model_sha256: str,
    expected_processed_fights_sha256: str,
    expected_processed_features_sha256: str,
) -> ValidatedLegacyCapture:
    """Validate an explicitly enumerated capture of the currently served legacy unit."""
    paths = {
        "source manifest": _validated_explicit_file(
            source_manifest,
            expected_sha256=expected_source_manifest_sha256,
            label="legacy source manifest",
        ),
        "runtime manifest": _validated_explicit_file(
            runtime_manifest,
            expected_sha256=expected_runtime_manifest_sha256,
            label="legacy runtime manifest",
        ),
        "capture manifest": _validated_explicit_file(
            capture_manifest,
            expected_sha256=expected_capture_manifest_sha256,
            label="legacy capture manifest",
        ),
        "primary model": _validated_explicit_file(
            primary_model,
            expected_sha256=expected_primary_model_sha256,
            label="legacy primary model",
        ),
        "no-odds model": _validated_explicit_file(
            no_odds_model,
            expected_sha256=expected_no_odds_model_sha256,
            label="legacy no-odds model",
        ),
        "logistic model": _validated_explicit_file(
            logistic_model,
            expected_sha256=expected_logistic_model_sha256,
            label="legacy logistic model",
        ),
        "saved spec": _validated_explicit_file(
            saved_spec,
            expected_sha256=expected_saved_spec_sha256,
            label="legacy saved spec",
        ),
        "processed fights": _validated_explicit_file(
            processed_fights,
            expected_sha256=expected_processed_fights_sha256,
            label="legacy processed fights",
        ),
        "processed features": _validated_explicit_file(
            processed_features,
            expected_sha256=expected_processed_features_sha256,
            label="legacy processed features",
        ),
    }
    if len(set(paths.values())) != len(paths):
        raise BundleInstallError("Legacy capture inputs must be distinct explicit files")
    if paths["primary model"].name != "xgboost_model.pkl" or paths[
        "no-odds model"
    ].name != "xgboost_no_odds_model.pkl" or paths["logistic model"].name != "logistic_model.pkl":
        raise BundleInstallError("Legacy model inputs must use the three canonical alias filenames")
    if len({paths[label].parent for label in ("primary model", "no-odds model", "logistic model", "saved spec")}) != 1:
        raise BundleInstallError("Legacy model artifacts and saved spec must share one explicit directory")
    if paths["processed fights"].name != "fights_cleaned.csv" or paths[
        "processed features"
    ].name != "features.csv" or paths["processed fights"].parent != paths[
        "processed features"
    ].parent:
        raise BundleInstallError("Legacy processed inputs must be the exact two canonical CSV names")

    payload = _load_json_object(paths["source manifest"], label="Legacy source manifest")
    if payload.get("manifest_version") not in (1, 2):
        raise BundleInstallError("Legacy source manifest must be manifest version 1 or 2")
    expected_top_level = {
        "bundle_id": expected_bundle_id,
        "model_spec_name": expected_model_spec_name,
        "snapshot_max_event_date": expected_snapshot_max_event_date,
        "model_sha256": expected_primary_model_sha256,
        "no_odds_model_sha256": expected_no_odds_model_sha256,
        "logistic_model_sha256": expected_logistic_model_sha256,
        "processed_fights_sha256": expected_processed_fights_sha256,
        "processed_features_sha256": expected_processed_features_sha256,
        "processed_fights_bytes": int(paths["processed fights"].stat().st_size),
        "processed_features_bytes": int(paths["processed features"].stat().st_size),
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected_top_level.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise BundleInstallError(f"Legacy source manifest identity mismatch: {mismatches}")

    runtime = _load_json_object(paths["runtime manifest"], label="Captured runtime manifest")
    for key in (
        "bundle_id",
        "model_spec_name",
        "snapshot_max_event_date",
        "processed_fights_sha256",
        "processed_features_sha256",
        "processed_fights_bytes",
        "processed_features_bytes",
        "model_sha256",
        "no_odds_model_sha256",
    ):
        if runtime.get(key) != expected_top_level.get(key):
            raise BundleInstallError(f"Captured runtime manifest field {key} is inconsistent")

    capture = _load_json_object(paths["capture manifest"], label="Rollback capture manifest")
    if (
        capture.get("bundle_id") != expected_bundle_id
        or capture.get("model_spec_name") != expected_model_spec_name
        or capture.get("snapshot_max_event_date") != expected_snapshot_max_event_date
        or capture.get("verification", {}).get(
            "all_downloaded_and_copied_files_match_recorded_hashes"
        )
        is not True
    ):
        raise BundleInstallError("Rollback capture manifest identity or verification is invalid")
    capture_expected = {
        "primary_model": expected_primary_model_sha256,
        "no_odds_model": expected_no_odds_model_sha256,
        "logistic_model": expected_logistic_model_sha256,
        "saved_spec": expected_saved_spec_sha256,
        "processed_fights": expected_processed_fights_sha256,
        "processed_features": expected_processed_features_sha256,
        "runtime_manifest": expected_runtime_manifest_sha256,
    }
    capture_files = capture.get("files")
    if not isinstance(capture_files, dict) or any(
        not isinstance(capture_files.get(label), dict)
        or capture_files[label].get("sha256") != expected_hash
        for label, expected_hash in capture_expected.items()
    ):
        raise BundleInstallError("Rollback capture manifest file identities are incomplete")

    try:
        import joblib
        from src.model.production_bundle import _expected_no_odds_spec_payload

        primary_result = joblib.load(paths["primary model"])
        no_odds_result = joblib.load(paths["no-odds model"])
        logistic_result = joblib.load(paths["logistic model"])
    except Exception as exc:
        raise BundleInstallError(f"Could not load pinned legacy model artifacts: {exc}") from exc
    if any(not isinstance(result, dict) for result in (primary_result, no_odds_result, logistic_result)):
        raise BundleInstallError("Legacy model artifacts must contain model-result mappings")
    primary_spec = primary_result.get("training_spec")
    no_odds_spec = no_odds_result.get("training_spec")
    logistic_spec = logistic_result.get("training_spec")
    if not isinstance(primary_spec, dict) or primary_spec.get("name") != expected_model_spec_name:
        raise BundleInstallError("Legacy primary model spec identity is invalid")
    if primary_result.get("feature_cols") != primary_spec.get("feature_cols"):
        raise BundleInstallError("Legacy primary model feature contract is invalid")
    if logistic_spec != primary_spec or logistic_result.get("feature_cols") != primary_spec.get(
        "feature_cols"
    ):
        raise BundleInstallError("Legacy logistic model contract differs from primary")
    if no_odds_spec != _expected_no_odds_spec_payload(primary_spec) or no_odds_result.get(
        "feature_cols"
    ) != no_odds_spec.get("feature_cols"):
        raise BundleInstallError("Legacy no-odds model contract is invalid")
    if _load_json_object(paths["saved spec"], label="Legacy saved spec") != primary_spec:
        raise BundleInstallError("Legacy saved spec differs from the embedded primary contract")

    try:
        from src.model.production_bundle import get_processed_snapshot_max_event_date

        actual_max_date = get_processed_snapshot_max_event_date(paths["processed fights"].parent)
    except Exception as exc:
        raise BundleInstallError(f"Could not inspect legacy processed snapshot: {exc}") from exc
    if actual_max_date != expected_snapshot_max_event_date:
        raise BundleInstallError(
            f"Legacy processed max date mismatch: expected {expected_snapshot_max_event_date}, "
            f"got {actual_max_date}"
        )

    return ValidatedLegacyCapture(
        source_manifest=paths["source manifest"],
        runtime_manifest=paths["runtime manifest"],
        capture_manifest=paths["capture manifest"],
        primary_model=paths["primary model"],
        no_odds_model=paths["no-odds model"],
        logistic_model=paths["logistic model"],
        saved_spec=paths["saved spec"],
        processed_fights=paths["processed fights"],
        processed_features=paths["processed features"],
        payload=payload,
        source_manifest_sha256=expected_source_manifest_sha256,
    )


def _safe_target_root(path: Path, *, source_root: Path | None = None) -> Path:
    _assert_path_components_no_links(path, label="Target root")
    if _is_link_like(path):
        raise BundleInstallError(f"Target root cannot be a symlink or junction: {path}")
    target = path.resolve(strict=False)
    if target == Path(target.anchor) or not target.name:
        raise BundleInstallError(f"Refusing filesystem root as bundle store: {target}")
    if target.exists() and (not target.is_dir() or _is_link_like(target)):
        raise BundleInstallError(f"Target root must be a real directory: {target}")
    parent = target.parent.resolve(strict=True)
    if not parent.is_dir() or _is_link_like(parent):
        raise BundleInstallError(f"Target parent must be a real existing directory: {parent}")
    if source_root is not None:
        source = source_root.resolve(strict=True)
        try:
            target.relative_to(source)
        except ValueError:
            pass
        else:
            raise BundleInstallError("Target store cannot be inside the source bundle")
        try:
            source.relative_to(target)
        except ValueError:
            pass
        else:
            raise BundleInstallError("Source bundle cannot be inside the target store")
    return target


def _store_marker_payload() -> dict[str, object]:
    return {
        "schema_version": STORE_SCHEMA_VERSION,
        "kind": STORE_KIND,
        "layout": "immutable-releases-with-generation-lookups-and-atomic-state",
    }


def _validate_store_top_level(target_root: Path, *, allow_empty: bool) -> None:
    if not target_root.exists():
        if allow_empty:
            return
        raise BundleInstallError(f"Bundle store does not exist: {target_root}")
    if _is_link_like(target_root) or not target_root.is_dir():
        raise BundleInstallError(f"Bundle store must be a real directory: {target_root}")
    entries = {path.name for path in target_root.iterdir()}
    if not entries and allow_empty:
        return
    allowed = {
        STORE_MARKER_NAME,
        STATE_NAME,
        LOCK_NAME,
        RELEASES_DIR_NAME,
        LOOKUPS_DIR_NAME,
    }
    unexpected = sorted(entries - allowed)
    if unexpected:
        raise BundleInstallError(f"Bundle store contains unexpected top-level entries: {unexpected}")
    marker_path = target_root / STORE_MARKER_NAME
    releases = target_root / RELEASES_DIR_NAME
    if allow_empty and STATE_NAME not in entries and (
        STORE_MARKER_NAME not in entries
        or RELEASES_DIR_NAME not in entries
        or LOOKUPS_DIR_NAME not in entries
    ):
        # Recoverable initialization-only shapes are exact and contain no
        # release data.  They can result from a process loss between creating
        # the two owned layout entries.
        if STORE_MARKER_NAME in entries:
            marker = _load_json_object(marker_path, label="Bundle store marker")
            if marker != _store_marker_payload():
                raise BundleInstallError("Partial bundle store marker is invalid")
        if RELEASES_DIR_NAME in entries:
            if not releases.is_dir() or _is_link_like(releases) or any(releases.iterdir()):
                raise BundleInstallError("Partial bundle store releases directory is not empty")
        lookups = target_root / LOOKUPS_DIR_NAME
        if LOOKUPS_DIR_NAME in entries:
            if not lookups.is_dir() or _is_link_like(lookups) or any(lookups.iterdir()):
                raise BundleInstallError("Partial bundle store lookups directory is not empty")
        return
    marker = _load_json_object(marker_path, label="Bundle store marker")
    if marker != _store_marker_payload():
        raise BundleInstallError("Bundle store marker does not match the supported layout")
    if not releases.is_dir() or _is_link_like(releases):
        raise BundleInstallError(f"Bundle store releases directory is invalid: {releases}")
    lookups = target_root / LOOKUPS_DIR_NAME
    if not lookups.is_dir() or _is_link_like(lookups):
        raise BundleInstallError(f"Bundle store lookups directory is invalid: {lookups}")


def _initialize_store_layout(target_root: Path) -> None:
    entries = {path.name for path in target_root.iterdir()}
    if entries - {LOCK_NAME, STORE_MARKER_NAME, RELEASES_DIR_NAME, LOOKUPS_DIR_NAME}:
        _validate_store_top_level(target_root, allow_empty=False)
        return
    releases = target_root / RELEASES_DIR_NAME
    lookups = target_root / LOOKUPS_DIR_NAME
    marker = target_root / STORE_MARKER_NAME
    if marker.exists():
        if _load_json_object(marker, label="Bundle store marker") != _store_marker_payload():
            raise BundleInstallError("Partial bundle store marker is invalid")
    if releases.exists():
        if not releases.is_dir() or _is_link_like(releases) or any(releases.iterdir()):
            raise BundleInstallError("Partial bundle store releases directory is not empty")
    else:
        releases.mkdir(exist_ok=False)
    if lookups.exists():
        if not lookups.is_dir() or _is_link_like(lookups) or any(lookups.iterdir()):
            raise BundleInstallError("Partial bundle store lookups directory is not empty")
    else:
        lookups.mkdir(exist_ok=False)
    if not marker.exists():
        _atomic_write_json(marker, _store_marker_payload())
    _fsync_directory(target_root)


@contextmanager
def _store_lock(target_root: Path) -> Iterator[None]:
    lock_path = target_root / LOCK_NAME
    descriptor: int | None = None
    acquired = False
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        acquired = True
        payload = _canonical_json_bytes(
            {"pid": os.getpid(), "created_at": _utcnow(), "nonce": uuid.uuid4().hex}
        )
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        yield
    except FileExistsError as exc:
        raise BundleInstallError(
            f"Another bundle-store operation may be active; lock exists: {lock_path}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if acquired:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError as exc:
                # A stale lock fails later mutations closed.  Do not turn a
                # successfully committed state swap into an ambiguous failure.
                logger.error("Could not release bundle-store lock %s: %s", lock_path, exc)


def _tree_inventory(root: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if _is_link_like(path):
            raise BundleInstallError(f"Release contains a symlink or junction: {path}")
        if not path.is_file() or path == root / RECEIPT_NAME:
            continue
        relative = path.relative_to(root).as_posix()
        rows.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    return {
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "aggregate_sha256": _canonical_sha256(rows),
        "files": rows,
    }


def _fsync_tree(root: Path) -> None:
    """Flush every release file and directory before publishing its name."""
    directories = [root]
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if _is_link_like(path):
            raise BundleInstallError(f"Cannot fsync a release containing indirection: {path}")
        if path.is_file():
            # Windows' CRT rejects fsync on a read-only descriptor.  These are
            # newly owned temporary copies, so opening without truncation in
            # update mode is safe and works on both deployment and dev hosts.
            with path.open("r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())
        elif path.is_dir():
            directories.append(path)
    for directory in sorted(set(directories), key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _seal_immutable_release(root: Path) -> None:
    """Make a published release non-writable while keeping it runtime-readable."""
    if os.name == "nt":
        return
    directories = [root]
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            directories.append(path)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        directory.chmod(0o555)


def _make_owned_tree_deletable(root: Path) -> None:
    if os.name == "nt" or not root.exists():
        return
    root.chmod(0o755)
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            path.chmod(0o644)


def _make_lookup_writable(root: Path) -> None:
    if os.name == "nt":
        return
    root.chmod(0o750)
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o750)
        elif path.is_file():
            path.chmod(0o640)


def _release_id(bundle_id: str, source_manifest_sha256: str) -> str:
    if not _BUNDLE_ID_RE.fullmatch(bundle_id):
        raise BundleInstallError(f"Unsafe bundle_id: {bundle_id!r}")
    # Keep Windows development/verification paths below legacy MAX_PATH even
    # when evidence retains deep repository-relative names.  The receipt pins
    # the full digest and rejects the vanishingly unlikely 80-bit collision.
    return f"r-{source_manifest_sha256[:20]}"


def _rebase_manifest(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    rebased = deepcopy(payload)
    for field, relative in _CORE_PATH_FIELDS.items():
        rebased[field] = str((root / relative).resolve(strict=False))
    # Runtime reconciliation may add this field.  It must never retain a path
    # to the upload/extraction source after installation.
    rebased["rich_release_root"] = str(root.resolve(strict=False))
    return rebased


def _receipt_payload(
    release_root: Path,
    *,
    release_id: str,
    source: ValidatedSource,
) -> dict[str, object]:
    manifest_path = release_root / INSTALLED_MANIFEST_NAME
    inventory = _tree_inventory(release_root)
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "release_kind": "rich_v3",
        "release_id": release_id,
        "bundle_id": source.bundle_id,
        "source_manifest_sha256": source.manifest_sha256,
        "installed_manifest_sha256": _sha256_file(manifest_path),
        "source_manifest_copy": "provenance/source_staging_manifest.json",
        "core_path_rewrites": sorted((*_CORE_PATH_FIELDS, "rich_release_root")),
        "tree_inventory": inventory,
    }


def _validate_receipt(
    release_root: Path,
    *,
    expected_release_id: str | None = None,
) -> dict[str, Any]:
    receipt_path = release_root / RECEIPT_NAME
    receipt = _load_json_object(receipt_path, label="Release install receipt")
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise BundleInstallError("Unsupported release receipt schema")
    release_id = expected_release_id or release_root.name
    if receipt.get("release_id") != release_id:
        raise BundleInstallError("Release receipt id does not match its directory")
    bundle_id = str(receipt.get("bundle_id") or "")
    source_hash = str(receipt.get("source_manifest_sha256") or "")
    if not _BUNDLE_ID_RE.fullmatch(bundle_id) or not _SHA256_RE.fullmatch(source_hash):
        raise BundleInstallError("Release receipt source identity is invalid")
    if _release_id(bundle_id, source_hash) != release_id:
        raise BundleInstallError("Release directory does not derive from its source identity")
    release_kind = receipt.get("release_kind")
    expected_rewrites = (
        sorted((*_CORE_PATH_FIELDS, "rich_release_root"))
        if release_kind == "rich_v3"
        else sorted(_CORE_PATH_FIELDS)
    )
    if release_kind not in {"rich_v3", "legacy_runtime_capture"}:
        raise BundleInstallError("Unsupported release receipt kind")
    if receipt.get("core_path_rewrites") != expected_rewrites:
        raise BundleInstallError("Release receipt core-path rewrite policy is invalid")

    source_copy = _confined_file(
        release_root,
        receipt.get("source_manifest_copy"),
        label="source staging manifest copy",
    )
    if _sha256_file(source_copy) != source_hash:
        raise BundleInstallError("Preserved source staging manifest hash mismatch")
    installed_manifest = release_root / INSTALLED_MANIFEST_NAME
    if _sha256_file(installed_manifest) != receipt.get("installed_manifest_sha256"):
        raise BundleInstallError("Installed manifest hash does not match its receipt")

    expected_inventory = receipt.get("tree_inventory")
    actual_inventory = _tree_inventory(release_root)
    if expected_inventory != actual_inventory:
        raise BundleInstallError("Release tree no longer matches its exact install inventory")
    return receipt


def _validate_legacy_installed_release(
    root: Path,
    payload: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    if payload.get("manifest_version") not in (1, 2):
        raise BundleInstallError("Installed legacy runtime capture has an invalid manifest version")
    for field, relative in _CORE_PATH_FIELDS.items():
        if Path(str(payload.get(field) or "")).resolve(strict=True) != (
            root / relative
        ).resolve(strict=True):
            raise BundleInstallError(f"Installed legacy core path {field} is not exact")
    capture_record = payload.get("legacy_runtime_capture")
    if not isinstance(capture_record, dict) or capture_record.get(
        "processed_role"
    ) != "mutable_runtime_lookup_snapshot" or capture_record.get(
        "claims_immutable_training_snapshot"
    ) is not False:
        raise BundleInstallError("Installed legacy runtime snapshot is mislabeled")
    capture_manifest = _load_json_object(
        root / "provenance/rollback_capture_manifest.json",
        label="Installed rollback capture manifest",
    )
    capture_files = capture_manifest.get("files")
    if not isinstance(capture_files, dict):
        raise BundleInstallError("Installed rollback capture file identities are missing")
    saved_spec_record = capture_files.get("saved_spec")
    runtime_record = capture_files.get("runtime_manifest")
    if not isinstance(saved_spec_record, dict) or not isinstance(runtime_record, dict):
        raise BundleInstallError("Installed rollback capture spec/runtime identities are missing")
    _validate_legacy_capture(
        source_manifest=root / str(receipt["source_manifest_copy"]),
        runtime_manifest=root / "provenance/runtime_manifest.json",
        capture_manifest=root / "provenance/rollback_capture_manifest.json",
        primary_model=root / "models/xgboost_model.pkl",
        no_odds_model=root / "models/xgboost_no_odds_model.pkl",
        logistic_model=root / "models/logistic_model.pkl",
        saved_spec=root / f"models/{payload['model_spec_name']}_spec.json",
        processed_fights=root / "processed/fights_cleaned.csv",
        processed_features=root / "processed/features.csv",
        expected_bundle_id=str(payload["bundle_id"]),
        expected_model_spec_name=str(payload["model_spec_name"]),
        expected_snapshot_max_event_date=str(payload["snapshot_max_event_date"]),
        expected_source_manifest_sha256=str(receipt["source_manifest_sha256"]),
        expected_runtime_manifest_sha256=str(runtime_record["sha256"]),
        expected_capture_manifest_sha256=_sha256_file(
            root / "provenance/rollback_capture_manifest.json"
        ),
        expected_saved_spec_sha256=str(saved_spec_record["sha256"]),
        expected_primary_model_sha256=str(payload["model_sha256"]),
        expected_no_odds_model_sha256=str(payload["no_odds_model_sha256"]),
        expected_logistic_model_sha256=str(payload["logistic_model_sha256"]),
        expected_processed_fights_sha256=str(payload["processed_fights_sha256"]),
        expected_processed_features_sha256=str(payload["processed_features_sha256"]),
    )


def _validate_release(release_root: Path, *, rich: bool) -> dict[str, Any]:
    if _is_link_like(release_root):
        raise BundleInstallError(f"Release root cannot be a symlink or junction: {release_root}")
    root = release_root.resolve(strict=True)
    if not root.is_dir() or _is_link_like(root):
        raise BundleInstallError(f"Release root is invalid: {root}")
    _assert_no_symlinks(root, label="Installed release")
    receipt = _validate_receipt(root)
    manifest_path = root / INSTALLED_MANIFEST_NAME
    payload = _load_json_object(manifest_path, label="Installed manifest")
    if payload.get("bundle_id") != receipt.get("bundle_id"):
        raise BundleInstallError("Installed manifest and receipt bundle_id disagree")
    if receipt.get("release_kind") == "rich_v3":
        _require_manifest_layout(root, payload)
        if Path(str(payload.get("rich_release_root") or "")).resolve(strict=True) != root:
            raise BundleInstallError("Installed rich_release_root is not the immutable release root")
        _semantic_validate_manifest(manifest_path, root, rich=rich)
    else:
        _validate_legacy_installed_release(root, payload, receipt)
    return {
        "release_id": root.name,
        "bundle_id": payload["bundle_id"],
        "manifest_sha256": _sha256_file(manifest_path),
        "receipt_sha256": _sha256_file(root / RECEIPT_NAME),
        "tree_aggregate_sha256": receipt["tree_inventory"]["aggregate_sha256"],
    }


def _record_for_release(target_root: Path, release_id: str, *, rich: bool) -> dict[str, Any]:
    release_rel = Path(RELEASES_DIR_NAME) / release_id
    unresolved_release_root = target_root / release_rel
    _assert_path_components_no_links(unresolved_release_root, label="Release root")
    if _is_link_like(unresolved_release_root):
        raise BundleInstallError("Release pointer cannot traverse a symlink or junction")
    release_root = unresolved_release_root.resolve(strict=True)
    try:
        release_root.relative_to((target_root / RELEASES_DIR_NAME).resolve(strict=True))
    except ValueError as exc:
        raise BundleInstallError("Release path escapes its store") from exc
    validation = _validate_release(release_root, rich=rich)
    return {
        **validation,
        "release_path": release_rel.as_posix(),
        "manifest_path": (release_rel / INSTALLED_MANIFEST_NAME).as_posix(),
    }


def _lookup_seed_fields_for_release(
    target_root: Path,
    release_record: Mapping[str, object],
) -> dict[str, object]:
    release_root = target_root / str(release_record["release_path"])
    manifest = _load_json_object(release_root / INSTALLED_MANIFEST_NAME, label="Release manifest")
    return {
        "lookup_seed_fights_sha256": manifest.get("processed_fights_sha256"),
        "lookup_seed_features_sha256": manifest.get("processed_features_sha256"),
        "lookup_seed_fights_bytes": manifest.get("processed_fights_bytes"),
        "lookup_seed_features_bytes": manifest.get("processed_features_bytes"),
    }


def _lookup_identity(
    lookup_root: Path,
    *,
    exact_tree: bool = False,
    include_runtime_manifest: bool = False,
) -> dict[str, object]:
    if _is_link_like(lookup_root) or not lookup_root.is_dir():
        raise BundleInstallError(f"Generation lookup directory is invalid: {lookup_root}")
    _assert_no_symlinks(lookup_root, label="Generation lookup")
    if exact_tree:
        entries = {path.name for path in lookup_root.iterdir()}
        expected_entries = set(_LOOKUP_FILENAMES)
        if include_runtime_manifest:
            expected_entries.add(_RUNTIME_MANIFEST_NAME)
        if entries != expected_entries:
            raise BundleInstallError(
                "Generation lookup must contain exactly "
                f"{sorted(expected_entries)}; found {sorted(entries)}"
            )
    identity: dict[str, object] = {}
    for stem, filename in (
        ("fights", "fights_cleaned.csv"),
        ("features", "features.csv"),
    ):
        path = lookup_root / filename
        if not path.is_file() or _is_link_like(path):
            raise BundleInstallError(f"Generation lookup is missing {filename}: {lookup_root}")
        identity[f"{stem}_sha256"] = _sha256_file(path)
        identity[f"{stem}_bytes"] = int(path.stat().st_size)
    return identity


def _identity_from_seed_fields(fields: Mapping[str, object]) -> dict[str, object]:
    return {
        "fights_sha256": fields.get("lookup_seed_fights_sha256"),
        "features_sha256": fields.get("lookup_seed_features_sha256"),
        "fights_bytes": fields.get("lookup_seed_fights_bytes"),
        "features_bytes": fields.get("lookup_seed_features_bytes"),
    }


def _snapshot_fields(identity: Mapping[str, object]) -> dict[str, object]:
    return {
        "lookup_snapshot_fights_sha256": identity.get("fights_sha256"),
        "lookup_snapshot_features_sha256": identity.get("features_sha256"),
        "lookup_snapshot_fights_bytes": identity.get("fights_bytes"),
        "lookup_snapshot_features_bytes": identity.get("features_bytes"),
    }


def _identity_from_snapshot_fields(fields: Mapping[str, object]) -> dict[str, object]:
    return {
        "fights_sha256": fields.get("lookup_snapshot_fights_sha256"),
        "features_sha256": fields.get("lookup_snapshot_features_sha256"),
        "fights_bytes": fields.get("lookup_snapshot_fights_bytes"),
        "features_bytes": fields.get("lookup_snapshot_features_bytes"),
    }


def _validate_lookup_identity(identity: Mapping[str, object], *, label: str) -> None:
    for stem in ("fights", "features"):
        sha = identity.get(f"{stem}_sha256")
        size = identity.get(f"{stem}_bytes")
        if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
            raise BundleInstallError(f"{label} {stem} SHA-256 is invalid")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise BundleInstallError(f"{label} {stem} byte count is invalid")


def _require_lookup_identity(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    _validate_lookup_identity(expected, label=f"Expected {label}")
    if dict(actual) != dict(expected):
        raise BundleInstallError(
            f"{label} identity mismatch: expected {dict(expected)}, actual {dict(actual)}"
        )


def _runtime_manifest_binding(
    target_root: Path,
    lookup_root: Path,
    release_record: Mapping[str, object],
    *,
    required: bool,
) -> dict[str, object] | None:
    """Read one atomic runtime-manifest generation and bind it to its release."""
    path = lookup_root / _RUNTIME_MANIFEST_NAME
    if _is_link_like(path):
        raise BundleInstallError(f"Runtime lookup manifest cannot be a link: {path}")
    if not path.exists():
        if required:
            raise BundleInstallError(f"Runtime lookup manifest is missing: {path}")
        return None
    if not path.is_file():
        raise BundleInstallError(f"Runtime lookup manifest is not a regular file: {path}")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleInstallError(f"Runtime lookup manifest is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BundleInstallError(f"Runtime lookup manifest must contain an object: {path}")

    release_manifest = _load_json_object(
        target_root / str(release_record["manifest_path"]),
        label="Rollback release manifest",
    )
    for field in (
        "bundle_id",
        "model_sha256",
        "no_odds_model_sha256",
        "logistic_model_sha256",
    ):
        if payload.get(field) != release_manifest.get(field):
            raise BundleInstallError(
                f"Runtime lookup manifest {field} does not match its immutable release"
            )
    identity = {
        "fights_sha256": payload.get("processed_fights_sha256"),
        "features_sha256": payload.get("processed_features_sha256"),
        "fights_bytes": payload.get("processed_fights_bytes"),
        "features_bytes": payload.get("processed_features_bytes"),
    }
    _validate_lookup_identity(identity, label="Runtime manifest lookup")
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "identity": identity,
    }


def _lookup_relative_path(target_root: Path, lookup_root: Path) -> Path:
    lookups_root = (target_root / LOOKUPS_DIR_NAME).resolve(strict=True)
    _assert_path_components_no_links(lookup_root, label="Generation lookup")
    if _is_link_like(lookup_root):
        raise BundleInstallError(f"Generation lookup cannot be a link: {lookup_root}")
    resolved = lookup_root.resolve(strict=True)
    if resolved.parent != lookups_root:
        raise BundleInstallError(f"Generation lookup is not a direct child of {lookups_root}")
    return Path(LOOKUPS_DIR_NAME) / resolved.name


def _lookup_root_from_record(target_root: Path, record: Mapping[str, object]) -> Path:
    relative = _safe_relative_path(record.get("lookup_path"), label="Generation lookup path")
    if len(relative.parts) != 2 or relative.parts[0] != LOOKUPS_DIR_NAME:
        raise BundleInstallError("Generation lookup path must name one direct store lookup")
    unresolved = target_root / relative
    _assert_path_components_no_links(unresolved, label="Generation lookup path")
    if _is_link_like(unresolved):
        raise BundleInstallError("Generation lookup path cannot name a link")
    resolved = unresolved.resolve(strict=True)
    if resolved.parent != (target_root / LOOKUPS_DIR_NAME).resolve(strict=True):
        raise BundleInstallError("Generation lookup path escapes its store")
    return resolved


def _active_lookup_record(
    target_root: Path,
    release_record: dict[str, Any],
    *,
    lookup_root: Path,
) -> dict[str, Any]:
    return {
        **release_record,
        **_lookup_seed_fields_for_release(target_root, release_record),
        "lookup_path": _lookup_relative_path(target_root, lookup_root).as_posix(),
        "lookup_role": _LOOKUP_ACTIVE_ROLE,
    }


def _rollback_lookup_record(
    target_root: Path,
    release_record: dict[str, Any],
    *,
    lookup_root: Path,
    identity: Mapping[str, object],
    binding_kind: str,
    runtime_binding: Mapping[str, object] | None,
) -> dict[str, Any]:
    _validate_lookup_identity(identity, label="Rollback snapshot")
    record = {
        **release_record,
        **_lookup_seed_fields_for_release(target_root, release_record),
        "lookup_path": _lookup_relative_path(target_root, lookup_root).as_posix(),
        "lookup_role": _LOOKUP_ROLLBACK_ROLE,
        "lookup_snapshot_binding": binding_kind,
        **_snapshot_fields(identity),
    }
    if binding_kind == _LOOKUP_BINDING_RUNTIME_MANIFEST:
        if runtime_binding is None:
            raise BundleInstallError("Runtime-manifest snapshot binding is missing")
        record.update(
            {
                "lookup_snapshot_runtime_manifest_sha256": runtime_binding.get("sha256"),
                "lookup_snapshot_runtime_manifest_bytes": runtime_binding.get("bytes"),
            }
        )
    elif binding_kind != _LOOKUP_BINDING_RELEASE_SEED or runtime_binding is not None:
        raise BundleInstallError(f"Unsupported rollback snapshot binding: {binding_kind}")
    return record


def _publish_active_lookup(
    target_root: Path,
    release_record: dict[str, Any],
    *,
    source_lookup_root: Path,
    expected_identity: Mapping[str, object],
) -> dict[str, Any]:
    """Copy lookup bytes into a fresh path that no prior process can mutate."""
    actual_source = _lookup_identity(source_lookup_root)
    _require_lookup_identity(
        actual_source,
        expected_identity,
        label="Active lookup source",
    )

    lookups_root = (target_root / LOOKUPS_DIR_NAME).resolve(strict=True)
    temp_root = Path(tempfile.mkdtemp(prefix=".active-incoming-", dir=lookups_root)).resolve(
        strict=True
    )
    final_root = lookups_root / (
        f"a-{release_record['release_id']}-{uuid.uuid4().hex[:12]}"
    )
    published = False
    try:
        for filename in _LOOKUP_FILENAMES:
            shutil.copy2(source_lookup_root / filename, temp_root / filename)
        _make_lookup_writable(temp_root)
        _require_lookup_identity(
            _lookup_identity(temp_root, exact_tree=True),
            expected_identity,
            label="Copied active lookup",
        )
        _fsync_tree(temp_root)
        if final_root.exists():
            raise BundleInstallError(f"Generation lookup appeared concurrently: {final_root}")
        os.rename(temp_root, final_root)
        published = True
        _fsync_directory(lookups_root)
        _require_lookup_identity(
            _lookup_identity(final_root, exact_tree=True),
            expected_identity,
            label="Published active lookup",
        )
        return _active_lookup_record(
            target_root,
            release_record,
            lookup_root=final_root,
        )
    except Exception:
        cleanup = final_root if published else temp_root
        if cleanup.exists():
            resolved = cleanup.resolve(strict=True)
            if resolved.parent != lookups_root or (
                not published and not resolved.name.startswith(".active-incoming-")
            ):
                raise BundleInstallError(
                    f"Refusing cleanup of unexpected generation lookup: {resolved}"
                )
            _make_owned_tree_deletable(resolved)
            shutil.rmtree(resolved)
        raise


def _publish_rollback_snapshot(
    target_root: Path,
    active_record: Mapping[str, object],
    *,
    expected_identity: Mapping[str, object] | None,
) -> dict[str, Any]:
    """Stability-copy an active lookup into an immutable, exactly pinned snapshot."""
    source_root = _lookup_root_from_record(target_root, active_record)
    release = _record_for_release(target_root, str(active_record["release_id"]), rich=False)
    lookups_root = (target_root / LOOKUPS_DIR_NAME).resolve(strict=True)
    temp_root = Path(tempfile.mkdtemp(prefix=".snapshot-incoming-", dir=lookups_root)).resolve(
        strict=True
    )
    final_root: Path | None = None
    published = False
    try:
        copied_identity: dict[str, object] | None = None
        copied_binding_kind: str | None = None
        copied_runtime_binding: dict[str, object] | None = None
        seed_identity = _identity_from_seed_fields(
            _lookup_seed_fields_for_release(target_root, release)
        )
        _validate_lookup_identity(seed_identity, label="Release seed lookup")
        for _attempt in range(3):
            for filename in (*_LOOKUP_FILENAMES, _RUNTIME_MANIFEST_NAME):
                (temp_root / filename).unlink(missing_ok=True)
            before = _lookup_identity(source_root)
            manifest_before = _runtime_manifest_binding(
                target_root,
                source_root,
                release,
                required=False,
            )
            for filename in _LOOKUP_FILENAMES:
                shutil.copy2(source_root / filename, temp_root / filename)
            if manifest_before is not None:
                shutil.copy2(
                    source_root / _RUNTIME_MANIFEST_NAME,
                    temp_root / _RUNTIME_MANIFEST_NAME,
                )
            copied = _lookup_identity(
                temp_root,
                exact_tree=True,
                include_runtime_manifest=manifest_before is not None,
            )
            manifest_copied = _runtime_manifest_binding(
                target_root,
                temp_root,
                release,
                required=manifest_before is not None,
            )
            after = _lookup_identity(source_root)
            manifest_after = _runtime_manifest_binding(
                target_root,
                source_root,
                release,
                required=False,
            )
            expected_matches = expected_identity is None or copied == dict(expected_identity)
            if manifest_before is None:
                if (
                    manifest_copied is None
                    and manifest_after is None
                    and before == copied == after == seed_identity
                    and expected_matches
                ):
                    copied_identity = copied
                    copied_binding_kind = _LOOKUP_BINDING_RELEASE_SEED
                    copied_runtime_binding = None
                    break
                continue
            manifest_stable = (
                manifest_copied is not None
                and manifest_after is not None
                and manifest_before["sha256"] == manifest_copied["sha256"]
                == manifest_after["sha256"]
                and manifest_before["bytes"] == manifest_copied["bytes"]
                == manifest_after["bytes"]
                and manifest_before["identity"] == copied
            )
            if before == copied == after and manifest_stable and expected_matches:
                copied_identity = copied
                copied_binding_kind = _LOOKUP_BINDING_RUNTIME_MANIFEST
                copied_runtime_binding = manifest_copied
                break
        if copied_identity is None:
            qualifier = " and match approved predecessor evidence" if expected_identity else ""
            raise BundleInstallError(
                "Could not capture a coherent outgoing active lookup snapshot bound to a "
                "stable runtime manifest or exact release seed" + qualifier
            )
        assert copied_binding_kind is not None
        if expected_identity is not None:
            _require_lookup_identity(
                copied_identity,
                expected_identity,
                label="Approved outgoing rollback snapshot",
            )

        snapshot_digest = _canonical_sha256(
            {
                "release_id": release["release_id"],
                "binding": copied_binding_kind,
                "runtime_manifest_sha256": (
                    copied_runtime_binding.get("sha256")
                    if copied_runtime_binding is not None
                    else None
                ),
                **copied_identity,
            }
        )
        final_root = lookups_root / (
            f"s-{release['release_id']}-{snapshot_digest[:20]}"
        )
        _fsync_tree(temp_root)
        _seal_immutable_release(temp_root)
        if final_root.exists():
            existing_identity = _lookup_identity(
                final_root,
                exact_tree=True,
                include_runtime_manifest=(
                    copied_binding_kind == _LOOKUP_BINDING_RUNTIME_MANIFEST
                ),
            )
            _require_lookup_identity(
                existing_identity,
                copied_identity,
                label="Existing rollback snapshot",
            )
            if copied_binding_kind == _LOOKUP_BINDING_RUNTIME_MANIFEST:
                existing_runtime = _runtime_manifest_binding(
                    target_root,
                    final_root,
                    release,
                    required=True,
                )
                if existing_runtime != copied_runtime_binding:
                    raise BundleInstallError(
                        "Existing rollback snapshot runtime-manifest binding differs"
                    )
            _make_owned_tree_deletable(temp_root)
            shutil.rmtree(temp_root)
            return _rollback_lookup_record(
                target_root,
                release,
                lookup_root=final_root,
                identity=copied_identity,
                binding_kind=copied_binding_kind,
                runtime_binding=copied_runtime_binding,
            )
        os.rename(temp_root, final_root)
        published = True
        _fsync_directory(lookups_root)
        _require_lookup_identity(
            _lookup_identity(
                final_root,
                exact_tree=True,
                include_runtime_manifest=(
                    copied_binding_kind == _LOOKUP_BINDING_RUNTIME_MANIFEST
                ),
            ),
            copied_identity,
            label="Published rollback snapshot",
        )
        return _rollback_lookup_record(
            target_root,
            release,
            lookup_root=final_root,
            identity=copied_identity,
            binding_kind=copied_binding_kind,
            runtime_binding=copied_runtime_binding,
        )
    except Exception:
        cleanup = final_root if published and final_root is not None else temp_root
        if cleanup.exists():
            resolved = cleanup.resolve(strict=True)
            if resolved.parent != lookups_root or (
                not published and not resolved.name.startswith(".snapshot-incoming-")
            ):
                raise BundleInstallError(
                    f"Refusing cleanup of unexpected rollback snapshot: {resolved}"
                )
            _make_owned_tree_deletable(resolved)
            shutil.rmtree(resolved)
        raise


def _validate_record(
    target_root: Path,
    record: object,
    *,
    label: str,
    expected_lookup_role: str,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise BundleInstallError(f"{label} release pointer is not an object")
    release_id = str(record.get("release_id") or "")
    if not release_id or Path(release_id).name != release_id:
        raise BundleInstallError(f"{label} release id is unsafe")
    actual_core = _record_for_release(target_root, release_id, rich=False)
    lookup_root = _lookup_root_from_record(target_root, record)
    actual = {
        **actual_core,
        **_lookup_seed_fields_for_release(target_root, actual_core),
        "lookup_path": _lookup_relative_path(target_root, lookup_root).as_posix(),
        "lookup_role": expected_lookup_role,
    }
    if expected_lookup_role == _LOOKUP_ROLLBACK_ROLE:
        snapshot_identity = _identity_from_snapshot_fields(record)
        _validate_lookup_identity(snapshot_identity, label=f"{label} rollback snapshot")
        binding_kind = record.get("lookup_snapshot_binding")
        actual.update(
            {
                "lookup_snapshot_binding": binding_kind,
                **_snapshot_fields(snapshot_identity),
            }
        )
        if binding_kind == _LOOKUP_BINDING_RUNTIME_MANIFEST:
            runtime_sha = record.get("lookup_snapshot_runtime_manifest_sha256")
            runtime_bytes = record.get("lookup_snapshot_runtime_manifest_bytes")
            if not isinstance(runtime_sha, str) or not _SHA256_RE.fullmatch(runtime_sha):
                raise BundleInstallError(f"{label} runtime manifest SHA-256 is invalid")
            if (
                isinstance(runtime_bytes, bool)
                or not isinstance(runtime_bytes, int)
                or runtime_bytes < 0
            ):
                raise BundleInstallError(f"{label} runtime manifest byte count is invalid")
            actual.update(
                {
                    "lookup_snapshot_runtime_manifest_sha256": runtime_sha,
                    "lookup_snapshot_runtime_manifest_bytes": runtime_bytes,
                }
            )
        elif binding_kind != _LOOKUP_BINDING_RELEASE_SEED:
            raise BundleInstallError(f"{label} rollback snapshot binding is unsupported")
    elif expected_lookup_role != _LOOKUP_ACTIVE_ROLE:
        raise BundleInstallError(f"Unsupported expected lookup role: {expected_lookup_role}")
    if actual != record:
        raise BundleInstallError(f"{label} release pointer does not match its immutable release")
    lookup_identity = _lookup_identity(
        lookup_root,
        exact_tree=expected_lookup_role == _LOOKUP_ROLLBACK_ROLE,
        include_runtime_manifest=(
            expected_lookup_role == _LOOKUP_ROLLBACK_ROLE
            and actual.get("lookup_snapshot_binding") == _LOOKUP_BINDING_RUNTIME_MANIFEST
        ),
    )
    if expected_lookup_role == _LOOKUP_ROLLBACK_ROLE:
        _require_lookup_identity(
            lookup_identity,
            _identity_from_snapshot_fields(actual),
            label=f"{label} immutable rollback snapshot",
        )
        if actual["lookup_snapshot_binding"] == _LOOKUP_BINDING_RUNTIME_MANIFEST:
            runtime_binding = _runtime_manifest_binding(
                target_root,
                lookup_root,
                actual_core,
                required=True,
            )
            if (
                runtime_binding is None
                or runtime_binding["sha256"]
                != actual["lookup_snapshot_runtime_manifest_sha256"]
                or runtime_binding["bytes"]
                != actual["lookup_snapshot_runtime_manifest_bytes"]
                or runtime_binding["identity"] != lookup_identity
            ):
                raise BundleInstallError(
                    f"{label} immutable rollback runtime-manifest binding differs"
                )
        else:
            _require_lookup_identity(
                lookup_identity,
                _identity_from_seed_fields(actual),
                label=f"{label} release-seed rollback snapshot",
            )
    return actual


def _read_validated_state(target_root: Path, *, required: bool) -> dict[str, Any] | None:
    state_path = target_root / STATE_NAME
    if not state_path.exists():
        if required:
            raise BundleInstallError(f"Bundle store has no active state: {state_path}")
        return None
    state = _load_json_object(state_path, label="Active bundle state")
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise BundleInstallError("Unsupported active bundle state schema")
    active = _validate_record(
        target_root,
        state.get("active"),
        label="Active",
        expected_lookup_role=_LOOKUP_ACTIVE_ROLE,
    )
    rollback_raw = state.get("rollback")
    rollback = None
    if rollback_raw is not None:
        rollback = _validate_record(
            target_root,
            rollback_raw,
            label="Rollback",
            expected_lookup_role=_LOOKUP_ROLLBACK_ROLE,
        )
        if rollback["release_id"] == active["release_id"]:
            raise BundleInstallError("Active and rollback pointers cannot name the same release")
    normalized = {
        "schema_version": STATE_SCHEMA_VERSION,
        "updated_at": state.get("updated_at"),
        "action": state.get("action"),
        "active": active,
        "rollback": rollback,
    }
    if state != normalized or not isinstance(state.get("updated_at"), str):
        raise BundleInstallError("Active bundle state contains unsupported or malformed fields")
    return normalized


def _publish_release(
    source: ValidatedSource,
    *,
    target_root: Path,
) -> dict[str, Any]:
    release_id = _release_id(source.bundle_id, source.manifest_sha256)
    releases_root = (target_root / RELEASES_DIR_NAME).resolve(strict=True)
    final_root = releases_root / release_id
    if final_root.exists():
        record = _record_for_release(target_root, release_id, rich=True)
        receipt = _load_json_object(final_root / RECEIPT_NAME, label="Existing release receipt")
        if receipt.get("source_manifest_sha256") != source.manifest_sha256:
            raise BundleInstallError("Existing release id has a different source identity")
        return record

    temp_root = Path(tempfile.mkdtemp(prefix=".incoming-", dir=releases_root)).resolve(strict=True)
    published = False
    try:
        shutil.copytree(source.root, temp_root, dirs_exist_ok=True, copy_function=shutil.copy2)
        _assert_no_symlinks(temp_root, label="Copied release")
        copied_manifest = temp_root / source.manifest_path.name
        _assert_exact_source_files(
            temp_root,
            expected_files=_required_rich_source_files(
                source.payload,
                source.manifest_path.name,
            ),
            label="Copied staging bundle",
        )

        # First validate the copied bytes at their temporary paths.  Only the
        # four core absolute paths differ from the already validated source.
        temp_payload = _rebase_manifest(source.payload, temp_root)
        _write_json_file(copied_manifest, temp_payload)
        _require_manifest_layout(temp_root, temp_payload)
        _semantic_validate_manifest(copied_manifest, temp_root, rich=True)

        preserved_source = temp_root / "provenance/source_staging_manifest.json"
        if preserved_source.exists():
            raise BundleInstallError(
                "Source bundle already contains reserved provenance/source_staging_manifest.json"
            )
        preserved_source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.manifest_path, preserved_source)
        if _sha256_file(preserved_source) != source.manifest_sha256:
            raise BundleInstallError("Could not preserve the exact source staging manifest")

        installed_manifest = temp_root / INSTALLED_MANIFEST_NAME
        final_payload = _rebase_manifest(source.payload, final_root)
        _write_json_file(installed_manifest, final_payload)
        if copied_manifest != installed_manifest:
            copied_manifest.unlink()

        receipt = _receipt_payload(
            temp_root,
            release_id=release_id,
            source=source,
        )
        _write_json_file(temp_root / RECEIPT_NAME, receipt)
        _validate_receipt(temp_root, expected_release_id=release_id)
        _fsync_tree(temp_root)

        if final_root.exists():
            raise BundleInstallError(f"Release appeared concurrently: {final_root}")
        os.rename(temp_root, final_root)
        published = True
        _fsync_directory(releases_root)

        # The release is still inert here.  Validate again at its final paths
        # before the atomic state pointer is allowed to mention it.
        record = _record_for_release(target_root, release_id, rich=True)
        _seal_immutable_release(final_root)
        _fsync_directory(releases_root)
        return record
    except Exception:
        cleanup = final_root if published else temp_root
        if cleanup.exists():
            resolved = cleanup.resolve(strict=True)
            if resolved.parent != releases_root or (
                not published and not resolved.name.startswith(".incoming-")
            ):
                raise BundleInstallError(f"Refusing cleanup of unexpected path: {resolved}")
            _make_owned_tree_deletable(resolved)
            shutil.rmtree(resolved)
        raise


def _publish_legacy_release(
    source: ValidatedLegacyCapture,
    *,
    target_root: Path,
) -> dict[str, Any]:
    release_id = _release_id(source.bundle_id, source.source_manifest_sha256)
    releases_root = (target_root / RELEASES_DIR_NAME).resolve(strict=True)
    final_root = releases_root / release_id
    if final_root.exists():
        record = _record_for_release(target_root, release_id, rich=False)
        receipt = _load_json_object(final_root / RECEIPT_NAME, label="Existing legacy receipt")
        if (
            receipt.get("release_kind") != "legacy_runtime_capture"
            or receipt.get("source_manifest_sha256") != source.source_manifest_sha256
        ):
            raise BundleInstallError("Existing legacy release id has a different identity")
        return record

    temp_root = Path(tempfile.mkdtemp(prefix=".incoming-", dir=releases_root)).resolve(strict=True)
    published = False
    try:
        copies = {
            source.primary_model: temp_root / "models/xgboost_model.pkl",
            source.no_odds_model: temp_root / "models/xgboost_no_odds_model.pkl",
            source.logistic_model: temp_root / "models/logistic_model.pkl",
            source.saved_spec: temp_root / f"models/{source.payload['model_spec_name']}_spec.json",
            source.processed_fights: temp_root / "processed/fights_cleaned.csv",
            source.processed_features: temp_root / "processed/features.csv",
            source.source_manifest: temp_root / "provenance/source_staging_manifest.json",
            source.runtime_manifest: temp_root / "provenance/runtime_manifest.json",
            source.capture_manifest: temp_root / "provenance/rollback_capture_manifest.json",
        }
        for origin, destination in copies.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, destination)
            if _sha256_file(destination) != _sha256_file(origin):
                raise BundleInstallError(f"Legacy capture copy failed identity check: {origin}")

        payload = deepcopy(source.payload)
        for field, relative in _CORE_PATH_FIELDS.items():
            payload[field] = str((final_root / relative).resolve(strict=False))
        payload["legacy_runtime_capture"] = {
            "schema_version": 1,
            "processed_role": "mutable_runtime_lookup_snapshot",
            "claims_immutable_training_snapshot": False,
            "statement": (
                "These processed bytes capture the mutable lookup snapshot served by the "
                "predecessor deployment; they are not relabeled as its immutable training data."
            ),
            "source_manifest_path": "provenance/source_staging_manifest.json",
            "runtime_manifest_path": "provenance/runtime_manifest.json",
            "capture_manifest_path": "provenance/rollback_capture_manifest.json",
        }
        _write_json_file(temp_root / INSTALLED_MANIFEST_NAME, payload)
        inventory = _tree_inventory(temp_root)
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "release_kind": "legacy_runtime_capture",
            "release_id": release_id,
            "bundle_id": source.bundle_id,
            "source_manifest_sha256": source.source_manifest_sha256,
            "installed_manifest_sha256": _sha256_file(
                temp_root / INSTALLED_MANIFEST_NAME
            ),
            "source_manifest_copy": "provenance/source_staging_manifest.json",
            "core_path_rewrites": sorted(_CORE_PATH_FIELDS),
            "tree_inventory": inventory,
        }
        _write_json_file(temp_root / RECEIPT_NAME, receipt)
        _validate_receipt(temp_root, expected_release_id=release_id)
        _fsync_tree(temp_root)
        if final_root.exists():
            raise BundleInstallError(f"Legacy release appeared concurrently: {final_root}")
        os.rename(temp_root, final_root)
        published = True
        _fsync_directory(releases_root)
        record = _record_for_release(target_root, release_id, rich=False)
        _seal_immutable_release(final_root)
        _fsync_directory(releases_root)
        return record
    except Exception:
        cleanup = final_root if published else temp_root
        if cleanup.exists():
            resolved = cleanup.resolve(strict=True)
            if resolved.parent != releases_root or (
                not published and not resolved.name.startswith(".incoming-")
            ):
                raise BundleInstallError(f"Refusing cleanup of unexpected path: {resolved}")
            _make_owned_tree_deletable(resolved)
            shutil.rmtree(resolved)
        raise


def _state_payload(
    *,
    action: str,
    active: dict[str, Any],
    rollback: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "updated_at": _utcnow(),
        "action": action,
        "active": active,
        "rollback": rollback,
    }


def _require_expected_release(
    record: Mapping[str, object],
    *,
    expected_bundle_id: str,
    expected_release_id: str,
    expected_manifest_sha256: str,
    label: str,
) -> None:
    if not _SHA256_RE.fullmatch(expected_manifest_sha256):
        raise BundleInstallError(f"Expected {label} manifest SHA-256 is invalid")
    expected = {
        "bundle_id": expected_bundle_id,
        "release_id": expected_release_id,
        "manifest_sha256": expected_manifest_sha256,
    }
    mismatches = {
        key: {"expected": value, "actual": record.get(key)}
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatches:
        raise BundleInstallError(
            f"{label} release changed or does not match the explicit expectation: {mismatches}"
        )


def _require_candidate_predecessor(
    candidate_payload: Mapping[str, object],
    *,
    active_manifest_path: Path,
    active_lookup_path: Path,
) -> dict[str, object]:
    """Bind staged rollback evidence to the exact complete active predecessor."""
    active = _load_json_object(active_manifest_path, label="Active predecessor manifest")
    rollback = candidate_payload.get("previous_rollback_identity")
    if not isinstance(rollback, dict):
        raise BundleInstallError("Candidate has no previous_rollback_identity")
    source_manifest = rollback.get("source_manifest")
    source_payload = source_manifest.get("payload") if isinstance(source_manifest, dict) else None
    if not isinstance(source_payload, dict) or source_payload.get("bundle_id") != active.get(
        "bundle_id"
    ):
        raise BundleInstallError("Candidate rollback source bundle does not match active predecessor")

    models = rollback.get("local_model_artifacts")
    expected_models = {
        "primary": "model_sha256",
        "no_odds": "no_odds_model_sha256",
        "logistic": "logistic_model_sha256",
    }
    if not isinstance(models, dict):
        raise BundleInstallError("Candidate rollback model identities are missing")
    for label, active_field in expected_models.items():
        record = models.get(label)
        if not isinstance(record, dict) or record.get("sha256") != active.get(active_field):
            raise BundleInstallError(
                f"Candidate rollback {label} model does not match active predecessor"
            )

    runtime_hashes = rollback.get("runtime_lookup_hashes")
    readyz = rollback.get("readyz_evidence")
    ready_payload = readyz.get("payload") if isinstance(readyz, dict) else None
    ready_bundle = ready_payload.get("production_bundle") if isinstance(ready_payload, dict) else None
    if not isinstance(runtime_hashes, dict) or not isinstance(ready_bundle, dict):
        raise BundleInstallError("Candidate rollback runtime evidence is missing")
    approved_identity = {
        "fights_sha256": runtime_hashes.get("processed_fights_sha256"),
        "features_sha256": runtime_hashes.get("processed_features_sha256"),
        "fights_bytes": ready_bundle.get("processed_fights_bytes"),
        "features_bytes": ready_bundle.get("processed_features_bytes"),
    }
    _validate_lookup_identity(approved_identity, label="Candidate approved rollback lookup")
    expected_ready = {
        "bundle_id": active.get("bundle_id"),
        "processed_fights_sha256": approved_identity["fights_sha256"],
        "processed_features_sha256": approved_identity["features_sha256"],
        "processed_fights_bytes": approved_identity["fights_bytes"],
        "processed_features_bytes": approved_identity["features_bytes"],
    }
    if any(ready_bundle.get(field) != value for field, value in expected_ready.items()):
        raise BundleInstallError(
            "Candidate rollback readiness evidence does not match active predecessor"
        )
    _require_lookup_identity(
        _lookup_identity(active_lookup_path),
        approved_identity,
        label="Candidate rollback runtime snapshot",
    )
    return approved_identity


def initialize_store(
    *,
    source_root: Path,
    target_root: Path,
    expected_bundle_id: str,
    expected_source_manifest_sha256: str,
    allow_no_rollback_test_only: bool = False,
) -> dict[str, Any]:
    """Test helper for rich-store initialization; production must seed the live predecessor."""
    if not allow_no_rollback_test_only:
        raise BundleInstallError(
            "A rich candidate cannot initialize an active store without a rollback release; "
            "use initialize-legacy for the captured live predecessor, then promote."
        )
    source = _validate_source(
        source_root,
        expected_bundle_id=expected_bundle_id,
        expected_manifest_sha256=expected_source_manifest_sha256,
    )
    target = _safe_target_root(target_root, source_root=source.root)
    _validate_store_top_level(target, allow_empty=True)

    if not target.exists():
        target.mkdir()
    with _store_lock(target):
        _initialize_store_layout(target)
        _validate_store_top_level(target, allow_empty=False)
        if _read_validated_state(target, required=False) is not None:
            raise BundleInstallError("Initialize requires a store with no active bundle")
        release = _publish_release(source, target_root=target)
        seed_fields = _lookup_seed_fields_for_release(target, release)
        active = _publish_active_lookup(
            target,
            release,
            source_lookup_root=target / str(release["release_path"]) / "processed",
            expected_identity=_identity_from_seed_fields(seed_fields),
        )
        state = _state_payload(action="initialize", active=active, rollback=None)
        _atomic_write_json(target / STATE_NAME, state)

    return resolve_store(target_root=target)


def initialize_legacy_store(
    *,
    source_manifest: Path,
    runtime_manifest: Path,
    capture_manifest: Path,
    primary_model: Path,
    no_odds_model: Path,
    logistic_model: Path,
    saved_spec: Path,
    processed_fights: Path,
    processed_features: Path,
    target_root: Path,
    expected_bundle_id: str,
    expected_model_spec_name: str,
    expected_snapshot_max_event_date: str,
    expected_source_manifest_sha256: str,
    expected_runtime_manifest_sha256: str,
    expected_capture_manifest_sha256: str,
    expected_saved_spec_sha256: str,
    expected_primary_model_sha256: str,
    expected_no_odds_model_sha256: str,
    expected_logistic_model_sha256: str,
    expected_processed_fights_sha256: str,
    expected_processed_features_sha256: str,
) -> dict[str, Any]:
    """Seed the exact currently served unit as the safe promotion predecessor."""
    source = _validate_legacy_capture(
        source_manifest=source_manifest,
        runtime_manifest=runtime_manifest,
        capture_manifest=capture_manifest,
        primary_model=primary_model,
        no_odds_model=no_odds_model,
        logistic_model=logistic_model,
        saved_spec=saved_spec,
        processed_fights=processed_fights,
        processed_features=processed_features,
        expected_bundle_id=expected_bundle_id,
        expected_model_spec_name=expected_model_spec_name,
        expected_snapshot_max_event_date=expected_snapshot_max_event_date,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_runtime_manifest_sha256=expected_runtime_manifest_sha256,
        expected_capture_manifest_sha256=expected_capture_manifest_sha256,
        expected_saved_spec_sha256=expected_saved_spec_sha256,
        expected_primary_model_sha256=expected_primary_model_sha256,
        expected_no_odds_model_sha256=expected_no_odds_model_sha256,
        expected_logistic_model_sha256=expected_logistic_model_sha256,
        expected_processed_fights_sha256=expected_processed_fights_sha256,
        expected_processed_features_sha256=expected_processed_features_sha256,
    )
    target = _safe_target_root(target_root)
    for path in (
        source.source_manifest,
        source.runtime_manifest,
        source.capture_manifest,
        source.primary_model,
        source.no_odds_model,
        source.logistic_model,
        source.saved_spec,
        source.processed_fights,
        source.processed_features,
    ):
        try:
            path.relative_to(target)
        except ValueError:
            pass
        else:
            raise BundleInstallError("Legacy capture inputs cannot be inside the target store")
        try:
            target.relative_to(path.parent)
        except ValueError:
            pass
        else:
            raise BundleInstallError(
                f"Target store cannot be inside a legacy capture input directory: {path.parent}"
            )
    _validate_store_top_level(target, allow_empty=True)
    if not target.exists():
        target.mkdir()
    with _store_lock(target):
        _initialize_store_layout(target)
        _validate_store_top_level(target, allow_empty=False)
        if _read_validated_state(target, required=False) is not None:
            raise BundleInstallError("Legacy initialization requires a store with no active bundle")
        release = _publish_legacy_release(source, target_root=target)
        seed_fields = _lookup_seed_fields_for_release(target, release)
        active = _publish_active_lookup(
            target,
            release,
            source_lookup_root=target / str(release["release_path"]) / "processed",
            expected_identity=_identity_from_seed_fields(seed_fields),
        )
        state = _state_payload(
            action="initialize_legacy_capture",
            active=active,
            rollback=None,
        )
        _atomic_write_json(target / STATE_NAME, state)
    return resolve_store(target_root=target)


def promote_bundle(
    *,
    source_root: Path,
    target_root: Path,
    expected_bundle_id: str,
    expected_source_manifest_sha256: str,
    expected_active_bundle_id: str,
    expected_active_release_id: str,
    expected_active_manifest_sha256: str,
) -> dict[str, Any]:
    """Atomically activate one release while retaining the complete current release."""
    source = _validate_source(
        source_root,
        expected_bundle_id=expected_bundle_id,
        expected_manifest_sha256=expected_source_manifest_sha256,
    )
    target = _safe_target_root(target_root, source_root=source.root)
    _validate_store_top_level(target, allow_empty=False)
    preflight = _read_validated_state(target, required=True)
    assert preflight is not None
    _require_expected_release(
        preflight["active"],
        expected_bundle_id=expected_active_bundle_id,
        expected_release_id=expected_active_release_id,
        expected_manifest_sha256=expected_active_manifest_sha256,
        label="Active predecessor",
    )
    approved_predecessor_identity = _require_candidate_predecessor(
        source.payload,
        active_manifest_path=(target / preflight["active"]["manifest_path"]).resolve(
            strict=True
        ),
        active_lookup_path=(target / preflight["active"]["lookup_path"]).resolve(
            strict=True
        ),
    )

    with _store_lock(target):
        _validate_store_top_level(target, allow_empty=False)
        current = _read_validated_state(target, required=True)
        assert current is not None
        if current["active"] != preflight["active"]:
            raise BundleInstallError("Active release changed after preflight; refusing promotion")
        release = _publish_release(source, target_root=target)
        if release["release_id"] == current["active"]["release_id"]:
            return resolve_store(target_root=target)
        seed_fields = _lookup_seed_fields_for_release(target, release)
        new_active = _publish_active_lookup(
            target,
            release,
            source_lookup_root=target / str(release["release_path"]) / "processed",
            expected_identity=_identity_from_seed_fields(seed_fields),
        )
        rollback_snapshot = _publish_rollback_snapshot(
            target,
            current["active"],
            expected_identity=approved_predecessor_identity,
        )
        # Re-run the complete candidate predecessor contract against the
        # disconnected immutable copy, then verify its state record directly
        # before publishing the one atomic pointer file.
        _require_candidate_predecessor(
            source.payload,
            active_manifest_path=(target / rollback_snapshot["manifest_path"]).resolve(
                strict=True
            ),
            active_lookup_path=_lookup_root_from_record(target, rollback_snapshot),
        )
        rollback_snapshot = _validate_record(
            target,
            rollback_snapshot,
            label="Prepared rollback",
            expected_lookup_role=_LOOKUP_ROLLBACK_ROLE,
        )
        state = _state_payload(
            action="promote",
            active=new_active,
            rollback=rollback_snapshot,
        )
        _atomic_write_json(target / STATE_NAME, state)

    return resolve_store(target_root=target)


def rollback_bundle(
    *,
    target_root: Path,
    expected_active_bundle_id: str,
    expected_rollback_bundle_id: str,
    expected_active_release_id: str,
    expected_rollback_release_id: str,
    expected_active_manifest_sha256: str,
    expected_rollback_manifest_sha256: str,
) -> dict[str, Any]:
    """Atomically swap the active and rollback release pointers."""
    target = _safe_target_root(target_root)
    _validate_store_top_level(target, allow_empty=False)
    preflight = _read_validated_state(target, required=True)
    assert preflight is not None
    if preflight["rollback"] is None:
        raise BundleInstallError("No rollback release is recorded")
    _require_expected_release(
        preflight["active"],
        expected_bundle_id=expected_active_bundle_id,
        expected_release_id=expected_active_release_id,
        expected_manifest_sha256=expected_active_manifest_sha256,
        label="Active rollback source",
    )
    _require_expected_release(
        preflight["rollback"],
        expected_bundle_id=expected_rollback_bundle_id,
        expected_release_id=expected_rollback_release_id,
        expected_manifest_sha256=expected_rollback_manifest_sha256,
        label="Rollback target",
    )

    with _store_lock(target):
        current = _read_validated_state(target, required=True)
        assert current is not None and current["rollback"] is not None
        if current["active"] != preflight["active"] or current["rollback"] != preflight["rollback"]:
            raise BundleInstallError("Bundle state changed after preflight; refusing rollback")
        outgoing_snapshot = _publish_rollback_snapshot(
            target,
            current["active"],
            expected_identity=None,
        )
        rollback_identity = _identity_from_snapshot_fields(current["rollback"])
        rollback_release = _record_for_release(
            target,
            str(current["rollback"]["release_id"]),
            rich=False,
        )
        new_active = _publish_active_lookup(
            target,
            rollback_release,
            source_lookup_root=_lookup_root_from_record(target, current["rollback"]),
            expected_identity=rollback_identity,
        )
        outgoing_snapshot = _validate_record(
            target,
            outgoing_snapshot,
            label="Prepared roll-forward",
            expected_lookup_role=_LOOKUP_ROLLBACK_ROLE,
        )
        state = _state_payload(
            action="rollback",
            active=new_active,
            rollback=outgoing_snapshot,
        )
        _atomic_write_json(target / STATE_NAME, state)

    return resolve_store(target_root=target)


def resolve_store(*, target_root: Path) -> dict[str, Any]:
    """Read and validate the active/rollback state without changing it."""
    target = _safe_target_root(target_root)
    _validate_store_top_level(target, allow_empty=False)
    state = _read_validated_state(target, required=True)
    assert state is not None
    active_manifest = (target / state["active"]["manifest_path"]).resolve(strict=True)
    active_release = active_manifest.parent
    active_lookup = (target / state["active"]["lookup_path"]).resolve(strict=True)
    rollback = state["rollback"]
    return {
        "store_root": str(target),
        "state_path": str(target / STATE_NAME),
        "state_sha256": _sha256_file(target / STATE_NAME),
        "active_bundle_id": state["active"]["bundle_id"],
        "active_release_id": state["active"]["release_id"],
        "active_manifest_sha256": state["active"]["manifest_sha256"],
        "active_receipt_sha256": state["active"]["receipt_sha256"],
        "active_tree_aggregate_sha256": state["active"]["tree_aggregate_sha256"],
        "active_manifest_path": str(active_manifest),
        "active_models_dir": str(active_release / "models"),
        "active_processed_dir": str(active_release / "processed"),
        "active_lookup_dir": str(active_lookup),
        "rollback_bundle_id": rollback["bundle_id"] if rollback else None,
        "rollback_release_id": rollback["release_id"] if rollback else None,
        "rollback_manifest_sha256": rollback["manifest_sha256"] if rollback else None,
        "rollback_receipt_sha256": rollback["receipt_sha256"] if rollback else None,
        "rollback_tree_aggregate_sha256": (
            rollback["tree_aggregate_sha256"] if rollback else None
        ),
        "rollback_manifest_path": (
            str((target / rollback["manifest_path"]).resolve(strict=True)) if rollback else None
        ),
        "rollback_lookup_dir": (
            str((target / rollback["lookup_path"]).resolve(strict=True)) if rollback else None
        ),
        "rollback_ready": rollback is not None,
    }


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--expected-bundle-id", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize_legacy = subparsers.add_parser(
        "initialize-legacy",
        help="Seed the explicitly captured currently served bundle as the predecessor.",
    )
    initialize_legacy.add_argument("--target-root", type=Path, required=True)
    for argument in (
        "source-manifest",
        "runtime-manifest",
        "capture-manifest",
        "primary-model",
        "no-odds-model",
        "logistic-model",
        "saved-spec",
        "processed-fights",
        "processed-features",
    ):
        initialize_legacy.add_argument(f"--{argument}", type=Path, required=True)
    for argument in (
        "expected-bundle-id",
        "expected-model-spec-name",
        "expected-snapshot-max-event-date",
        "expected-source-manifest-sha256",
        "expected-runtime-manifest-sha256",
        "expected-capture-manifest-sha256",
        "expected-saved-spec-sha256",
        "expected-primary-model-sha256",
        "expected-no-odds-model-sha256",
        "expected-logistic-model-sha256",
        "expected-processed-fights-sha256",
        "expected-processed-features-sha256",
    ):
        initialize_legacy.add_argument(f"--{argument}", required=True)

    promote = subparsers.add_parser(
        "promote",
        help="Atomically activate a v3 bundle and retain the current complete release.",
    )
    _add_source_arguments(promote)
    promote.add_argument("--expected-active-bundle-id", required=True)
    promote.add_argument("--expected-active-release-id", required=True)
    promote.add_argument("--expected-active-manifest-sha256", required=True)

    rollback = subparsers.add_parser(
        "rollback",
        help="Atomically swap the active and rollback complete releases.",
    )
    rollback.add_argument("--target-root", type=Path, required=True)
    rollback.add_argument("--expected-active-bundle-id", required=True)
    rollback.add_argument("--expected-rollback-bundle-id", required=True)
    rollback.add_argument("--expected-active-release-id", required=True)
    rollback.add_argument("--expected-rollback-release-id", required=True)
    rollback.add_argument("--expected-active-manifest-sha256", required=True)
    rollback.add_argument("--expected-rollback-manifest-sha256", required=True)

    resolve = subparsers.add_parser(
        "resolve",
        help="Validate the store and print active runtime paths without mutation.",
    )
    resolve.add_argument("--target-root", type=Path, required=True)
    resolve.add_argument(
        "--field",
        choices=(
            "active_manifest_path",
            "active_models_dir",
            "active_processed_dir",
            "active_lookup_dir",
            "active_bundle_id",
            "active_release_id",
            "active_manifest_sha256",
            "active_receipt_sha256",
            "active_tree_aggregate_sha256",
            "rollback_manifest_path",
            "rollback_lookup_dir",
            "rollback_bundle_id",
            "rollback_release_id",
            "rollback_manifest_sha256",
            "rollback_receipt_sha256",
            "rollback_tree_aggregate_sha256",
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "initialize-legacy":
        result = initialize_legacy_store(
            source_manifest=args.source_manifest,
            runtime_manifest=args.runtime_manifest,
            capture_manifest=args.capture_manifest,
            primary_model=args.primary_model,
            no_odds_model=args.no_odds_model,
            logistic_model=args.logistic_model,
            saved_spec=args.saved_spec,
            processed_fights=args.processed_fights,
            processed_features=args.processed_features,
            target_root=args.target_root,
            expected_bundle_id=args.expected_bundle_id,
            expected_model_spec_name=args.expected_model_spec_name,
            expected_snapshot_max_event_date=args.expected_snapshot_max_event_date,
            expected_source_manifest_sha256=args.expected_source_manifest_sha256,
            expected_runtime_manifest_sha256=args.expected_runtime_manifest_sha256,
            expected_capture_manifest_sha256=args.expected_capture_manifest_sha256,
            expected_saved_spec_sha256=args.expected_saved_spec_sha256,
            expected_primary_model_sha256=args.expected_primary_model_sha256,
            expected_no_odds_model_sha256=args.expected_no_odds_model_sha256,
            expected_logistic_model_sha256=args.expected_logistic_model_sha256,
            expected_processed_fights_sha256=args.expected_processed_fights_sha256,
            expected_processed_features_sha256=args.expected_processed_features_sha256,
        )
    elif args.command == "promote":
        result = promote_bundle(
            source_root=args.source_root,
            target_root=args.target_root,
            expected_bundle_id=args.expected_bundle_id,
            expected_source_manifest_sha256=args.expected_source_manifest_sha256,
            expected_active_bundle_id=args.expected_active_bundle_id,
            expected_active_release_id=args.expected_active_release_id,
            expected_active_manifest_sha256=args.expected_active_manifest_sha256,
        )
    elif args.command == "rollback":
        result = rollback_bundle(
            target_root=args.target_root,
            expected_active_bundle_id=args.expected_active_bundle_id,
            expected_rollback_bundle_id=args.expected_rollback_bundle_id,
            expected_active_release_id=args.expected_active_release_id,
            expected_rollback_release_id=args.expected_rollback_release_id,
            expected_active_manifest_sha256=args.expected_active_manifest_sha256,
            expected_rollback_manifest_sha256=args.expected_rollback_manifest_sha256,
        )
    else:
        result = resolve_store(target_root=args.target_root)
        if args.field:
            value = result[args.field]
            if value is None:
                raise BundleInstallError(f"Resolved field {args.field} is not available")
            print(value)
            return 0
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BundleInstallError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
