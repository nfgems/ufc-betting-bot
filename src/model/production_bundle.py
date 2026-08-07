from __future__ import annotations

import hashlib
import json
import logging
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from src.data.io_utils import write_json_atomically

logger = logging.getLogger(__name__)

HOSTED_PROJECT_ROOT = Path("/app")
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "models" / "current_production_model.json"
PRODUCTION_BUNDLE_ENV = "UFC_PRODUCTION_BUNDLE_MANIFEST"


class ProductionBundleError(RuntimeError):
    """Raised when the active production bundle is missing or inconsistent."""


@dataclass(frozen=True)
class ProductionBundle:
    manifest_path: Path
    bundle_id: str
    model_spec_name: str
    model_path: Path
    no_odds_model_path: Path
    processed_dir: Path
    snapshot_max_event_date: str
    built_at: str
    git_sha: str
    training_source_git_sha: str
    deployed_git_sha: str | None = None
    no_odds_model_spec_name: str | None = None
    logistic_model_path: Path | None = None
    processed_fights_sha256: str | None = None
    processed_features_sha256: str | None = None
    processed_fights_bytes: int | None = None
    processed_features_bytes: int | None = None
    manifest_updated_at: str | None = None
    manifest_version: int = 1
    model_sha256: str | None = None
    no_odds_model_sha256: str | None = None
    logistic_model_sha256: str | None = None


def is_hosted_runtime(*, project_root: Path | None = None) -> bool:
    root = project_root or PROJECT_ROOT
    if root == HOSTED_PROJECT_ROOT:
        return True
    for env_name in (
        "RAILWAY_VOLUME_MOUNT_PATH",
        "RAILWAY_ENVIRONMENT_NAME",
        "RAILWAY_SERVICE_NAME",
    ):
        if str(os.getenv(env_name, "") or "").strip():
            return True
    return False


def _resolved_path(path: Path) -> Path:
    return path.resolve(strict=False)


def _path_mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return -1


def _required_string(payload: dict[str, Any], key: str) -> str:
    raw = payload.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ProductionBundleError(f"Production bundle manifest is missing required field '{key}'.")
    return raw.strip()


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    raw = payload.get(key)
    text = str(raw or "").strip()
    return text or None


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    raw = payload.get(key)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _token_values() -> dict[str, str]:
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "DATA_DIR": str(DATA_DIR),
        "UFC_DATA_DIR": str(DATA_DIR),
        "MODELS_DIR": str(MODELS_DIR),
        "UFC_MODELS_DIR": str(MODELS_DIR),
        "PROCESSED_DATA_DIR": str(PROCESSED_DATA_DIR),
    }


def _expand_tokens(raw: str) -> str:
    expanded = raw
    for key, value in _token_values().items():
        expanded = expanded.replace(f"${{{key}}}", value)
    return expanded


def _resolve_bundle_path(raw: str) -> Path:
    expanded = _expand_tokens(raw)
    candidate = Path(expanded)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _resolve_manifest_path(manifest_path: str | Path | None = None) -> Path:
    if manifest_path is not None:
        candidate = Path(manifest_path)
    else:
        raw_env = str(os.getenv(PRODUCTION_BUNDLE_ENV, "") or "").strip()
        candidate = Path(raw_env) if raw_env else DEFAULT_MANIFEST_PATH
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _load_manifest_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProductionBundleError(f"Production bundle manifest is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProductionBundleError(
            f"Production bundle manifest must contain a JSON object: {path}."
        )
    return payload


def _runtime_timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_bundle_id(*, model_spec_name: str, snapshot_max_event_date: str) -> str:
    date_token = snapshot_max_event_date.replace("-", "")
    return f"ufc-production-{date_token}-{model_spec_name}"


def _determine_training_source_git_sha(base_payload: dict[str, Any]) -> str:
    """Resolve the commit whose checkout produced the trained artifacts.

    ``git_sha`` used to carry this value in CI and the deployed commit at
    Railway runtime.  Reading it as a final fallback keeps old manifests
    loadable while new manifests retain the two identities separately.
    """
    github_sha = str(os.getenv("GITHUB_SHA", "") or "").strip()
    if github_sha:
        return github_sha

    existing = _optional_string(base_payload, "training_source_git_sha")
    if existing:
        return existing

    generic_sha = str(os.getenv("GIT_SHA", "") or "").strip()
    railway_sha = str(os.getenv("RAILWAY_GIT_COMMIT_SHA", "") or "").strip()
    if generic_sha and not railway_sha:
        return generic_sha

    legacy = _optional_string(base_payload, "git_sha")
    if legacy:
        return legacy

    return generic_sha or "unknown"


def _determine_deployed_git_sha(base_payload: dict[str, Any]) -> str | None:
    """Resolve the commit Railway is currently executing, when knowable."""
    railway_sha = str(os.getenv("RAILWAY_GIT_COMMIT_SHA", "") or "").strip()
    if railway_sha:
        return railway_sha

    # A GitHub refit creates new generated files after GITHUB_SHA.  Carrying a
    # prior deployment SHA into that new source manifest would be false, so CI
    # deliberately clears it until Railway reconciles the runtime manifest.
    if str(os.getenv("GITHUB_SHA", "") or "").strip():
        return None

    return _optional_string(base_payload, "deployed_git_sha")


def _determine_git_sha(base_payload: dict[str, Any]) -> str:
    """Return the documented legacy alias for older consumers.

    At Railway runtime this aliases ``deployed_git_sha``; everywhere else it
    aliases ``training_source_git_sha``.  New callers should use the explicit
    fields instead.
    """
    training_source_git_sha = _determine_training_source_git_sha(base_payload)
    deployed_git_sha = _determine_deployed_git_sha(base_payload)
    return deployed_git_sha or training_source_git_sha


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_text_file_sha256(path: Path) -> str:
    """Hash text content independently of Git's LF/CRLF checkout policy."""

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _file_size_bytes(path: Path) -> int:
    return int(path.stat().st_size)


_STABLE_SOURCE_METADATA_KEYS = (
    "line",
    "status",
    "live_model_alias",
    "live_no_odds_alias",
)

_IDENTITY_SOURCE_METADATA_KEYS = (
    "as_of_date",
    "promoted_from",
    "promoted_alias_targets",
    "selection_basis",
    "rollback_backup_dir",
    "prior_promotion",
)

RICH_MANIFEST_VERSION = 3
RICH_STAGING_SCHEMA_VERSION = 1
_RICH_MANIFEST_SECTIONS = (
    "source_identity",
    "registered_training_specs",
    "model_artifacts",
    "saved_fullfit_spec",
    "immutable_training_snapshot",
    "raw_input_provenance",
    "selection_evidence",
    "training_invocation",
    "assembly_validation_environment",
    "finite_inference",
    "previous_rollback_identity",
)
_RICH_MODEL_FIELDS = (
    "model_sha256",
    "no_odds_model_sha256",
    "logistic_model_sha256",
)
_RICH_PROCESSED_FIELDS = (
    "processed_fights_sha256",
    "processed_features_sha256",
)
_SCHEDULED_REFIT_POLICY_KEYS = frozenset(
    {
        "policy_id",
        "sha256",
        "root_bundle_id",
        "parent_bundle_id",
        "parent_model_spec_name",
        "parent_model_sha256",
        "parent_no_odds_model_sha256",
        "parent_logistic_model_sha256",
        "parent_processed_fights_sha256",
        "parent_processed_features_sha256",
    }
)


def _is_sha256(value: object) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_rich_manifest_shape(
    payload: dict[str, Any],
    *,
    label: str,
) -> bool:
    """Validate the schema marker and required identity sections when present.

    Legacy manifests deliberately remain supported. Once any rich-schema marker
    appears, however, accepting a partial payload would turn missing provenance
    into an apparently valid runtime manifest, so the check is fail closed.
    """
    rich_marked = (
        _optional_int(payload, "manifest_version") == RICH_MANIFEST_VERSION
        or "staging_schema_version" in payload
        or "scheduled_refit_policy" in payload
        or any(key in payload for key in _RICH_MANIFEST_SECTIONS)
    )
    if not rich_marked:
        return False
    if _optional_int(payload, "manifest_version") != RICH_MANIFEST_VERSION:
        raise ProductionBundleError(
            f"{label} rich manifest must use manifest_version "
            f"{RICH_MANIFEST_VERSION}."
        )
    if _optional_int(payload, "staging_schema_version") != RICH_STAGING_SCHEMA_VERSION:
        raise ProductionBundleError(
            f"{label} rich manifest must use staging_schema_version "
            f"{RICH_STAGING_SCHEMA_VERSION}."
        )
    missing_sections = [
        key for key in _RICH_MANIFEST_SECTIONS if not isinstance(payload.get(key), dict)
    ]
    if missing_sections:
        raise ProductionBundleError(
            f"{label} rich manifest is missing required object sections: "
            f"{missing_sections}."
        )
    invalid_hashes = [
        key
        for key in _RICH_MODEL_FIELDS + _RICH_PROCESSED_FIELDS
        if not _is_sha256(payload.get(key))
    ]
    if invalid_hashes:
        raise ProductionBundleError(
            f"{label} rich manifest has missing or invalid SHA-256 fields: "
            f"{invalid_hashes}."
        )
    for key in ("processed_fights_bytes", "processed_features_bytes"):
        value = _optional_int(payload, key)
        if value is None or value < 0:
            raise ProductionBundleError(
                f"{label} rich manifest has an invalid byte count in {key}."
            )
    return True


def _validated_scheduled_refit_policy(
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    value = payload.get("scheduled_refit_policy")
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _SCHEDULED_REFIT_POLICY_KEYS:
        raise ProductionBundleError(
            "Production bundle scheduled_refit_policy has missing or unknown fields."
        )
    text_fields = (
        "policy_id",
        "root_bundle_id",
        "parent_bundle_id",
        "parent_model_spec_name",
    )
    if any(not str(value.get(field) or "").strip() for field in text_fields):
        raise ProductionBundleError(
            "Production bundle scheduled_refit_policy has an empty identity field."
        )
    hash_fields = _SCHEDULED_REFIT_POLICY_KEYS - set(text_fields)
    invalid_hashes = [field for field in sorted(hash_fields) if not _is_sha256(value.get(field))]
    if invalid_hashes:
        raise ProductionBundleError(
            "Production bundle scheduled_refit_policy has invalid SHA-256 fields: "
            f"{invalid_hashes}."
        )
    return deepcopy(value)


def _manifest_relative_file(
    manifest_path: Path,
    relative_path: object,
    *,
    label: str,
    release_root: Path | None = None,
) -> Path:
    raw = Path(str(relative_path or ""))
    if raw.is_absolute() or not raw.parts or ".." in raw.parts:
        raise ProductionBundleError(
            f"{label} has an unsafe manifest-relative path: {relative_path!r}."
        )
    root = _resolved_path(release_root or manifest_path.parent)
    if not root.is_dir():
        raise ProductionBundleError(
            f"{label} release root is missing or not a directory: {root}."
        )
    resolved = _resolved_path(root / raw)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ProductionBundleError(
            f"{label} escapes the manifest release root: {relative_path!r}."
        ) from exc
    _require_existing_file(resolved, label=label)
    return resolved


def _rich_release_root(payload: dict[str, Any], manifest_path: Path) -> Path:
    raw = _optional_string(payload, "rich_release_root")
    if raw is None:
        return _resolved_path(manifest_path.parent)
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise ProductionBundleError(
            "Production bundle rich_release_root must be an absolute path."
        )
    resolved = _resolved_path(candidate)
    if not resolved.is_dir():
        raise ProductionBundleError(
            f"Production bundle rich_release_root is missing: {resolved}."
        )
    return resolved


def _independent_audit_summary(
    payload: dict[str, Any],
    *,
    manifest_path: Path,
    release_root: Path,
) -> dict[str, str | None]:
    result: dict[str, str | None] = {
        "independent_audit_fights_canonical_sha256": None,
        "independent_audit_features_canonical_sha256": None,
    }
    invocation = payload.get("training_invocation")
    audit = (
        invocation.get("independent_audit_snapshot")
        if isinstance(invocation, dict)
        else None
    )
    if audit is None:
        return result
    if not isinstance(audit, dict):
        raise ProductionBundleError(
            "Production bundle independent audit snapshot must be an object."
        )
    for label, expected_staged_path in (
        ("fights", "provenance/independent_audit_snapshot/fights_cleaned.csv"),
        ("features", "provenance/independent_audit_snapshot/features.csv"),
    ):
        record = audit.get(label)
        if (
            not isinstance(record, dict)
            or record.get("staged_path") != expected_staged_path
            or not _is_sha256(record.get("sha256"))
        ):
            raise ProductionBundleError(
                f"Production bundle independent audit {label} identity is invalid."
            )
        path = _manifest_relative_file(
            manifest_path,
            record["staged_path"],
            label=f"independent audit {label}",
            release_root=release_root,
        )
        if _file_sha256(path) != str(record["sha256"]).lower():
            raise ProductionBundleError(
                f"Production bundle independent audit {label} hash mismatch."
            )
        result[f"independent_audit_{label}_canonical_sha256"] = (
            _canonical_text_file_sha256(path)
        )
    return result


def _scheduled_bfo_lineage_summary(
    payload: dict[str, Any],
    *,
    manifest_path: Path,
    release_root: Path,
    required: bool,
) -> dict[str, str | int | None]:
    """Validate the sealed BFO carry-forward package exposed by a rich bundle."""

    result: dict[str, str | int | None] = {
        "scheduled_bfo_lineage_manifest_sha256": None,
        "scheduled_bfo_lineage_batch_count": 0,
    }
    raw_provenance = payload.get("raw_input_provenance")
    record = (
        raw_provenance.get("scheduled_bfo_lineage")
        if isinstance(raw_provenance, dict)
        else None
    )
    if record is None:
        if required:
            raise ProductionBundleError(
                "Scheduled production bundle is missing its BFO lineage package."
            )
        return result
    if not isinstance(record, dict) or set(record) != {
        "manifest_staged_path",
        "manifest_sha256",
        "manifest_bytes",
        "batch_count",
        "batches",
    }:
        raise ProductionBundleError(
            "Production bundle scheduled BFO lineage identity is invalid."
        )

    staged_path = "provenance/bfo_lineage/manifest.json"
    manifest_sha256 = str(record.get("manifest_sha256") or "").lower()
    manifest_bytes = record.get("manifest_bytes")
    batch_count = record.get("batch_count")
    batches = record.get("batches")
    if (
        record.get("manifest_staged_path") != staged_path
        or not _is_sha256(manifest_sha256)
        or isinstance(manifest_bytes, bool)
        or not isinstance(manifest_bytes, int)
        or manifest_bytes <= 0
        or isinstance(batch_count, bool)
        or not isinstance(batch_count, int)
        or batch_count < 0
        or not isinstance(batches, list)
        or batch_count != len(batches)
    ):
        raise ProductionBundleError(
            "Production bundle scheduled BFO lineage identity is invalid."
        )

    lineage_manifest_path = _manifest_relative_file(
        manifest_path,
        staged_path,
        label="scheduled BFO lineage manifest",
        release_root=release_root,
    )
    if (
        _file_sha256(lineage_manifest_path) != manifest_sha256
        or _file_size_bytes(lineage_manifest_path) != manifest_bytes
    ):
        raise ProductionBundleError(
            "Production bundle scheduled BFO lineage manifest hash or size mismatch."
        )
    # The workflow restore command and production readiness deliberately use
    # the same validator, so their schema checks cannot drift apart.
    try:
        import scripts.bfo_lineage as bfo_lineage
    except ImportError as exc:
        raise ProductionBundleError(
            "Production runtime cannot load the BFO lineage validator."
        ) from exc
    try:
        lineage_payload = bfo_lineage.validate_package(
            lineage_manifest_path,
            expected_manifest_sha256=manifest_sha256,
        )
    except (bfo_lineage.BfoLineageError, OSError) as exc:
        raise ProductionBundleError(
            f"Production bundle scheduled BFO lineage package is invalid: {exc}"
        ) from exc
    if (
        lineage_payload.get("batches") != batches
        or len(lineage_payload["batches"]) != batch_count
        or (
            required
            and lineage_payload.get("parent_bundle_id")
            != payload["scheduled_refit_policy"].get("parent_bundle_id")
        )
    ):
        raise ProductionBundleError(
            "Production bundle scheduled BFO lineage manifest does not match its identity."
        )

    result["scheduled_bfo_lineage_manifest_sha256"] = manifest_sha256
    result["scheduled_bfo_lineage_batch_count"] = batch_count
    return result


def _rich_artifact_record(
    payload: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    artifacts = payload.get("model_artifacts")
    record = artifacts.get(label) if isinstance(artifacts, dict) else None
    if not isinstance(record, dict):
        raise ProductionBundleError(
            f"Rich production bundle is missing model_artifacts.{label}."
        )
    return record


def _validate_rich_model_identity(
    payload: dict[str, Any],
    *,
    manifest_path: Path,
    model_paths: dict[str, Path],
    actual_fingerprints: dict[str, str],
    label: str,
    release_root: Path | None = None,
) -> None:
    """Bind rich provenance to the exact three model binaries and sidecar."""
    artifacts_payload = payload.get("model_artifacts")
    if not isinstance(artifacts_payload, dict) or set(artifacts_payload) != {
        "primary",
        "no_odds",
        "logistic",
    }:
        raise ProductionBundleError(
            f"{label} rich model_artifacts must describe exactly primary, "
            "no_odds, and logistic."
        )
    hash_fields = {
        "primary": "model_sha256",
        "no_odds": "no_odds_model_sha256",
        "logistic": "logistic_model_sha256",
    }
    embedded_specs: dict[str, dict[str, Any]] = {}
    for artifact_label, hash_field in hash_fields.items():
        expected_hash = str(payload.get(hash_field) or "").strip().lower()
        actual_hash = actual_fingerprints.get(hash_field)
        if actual_hash is None or expected_hash != actual_hash:
            raise ProductionBundleError(
                f"{label} rich identity does not match the active {artifact_label} "
                f"model: manifest={expected_hash!r}, actual={actual_hash!r}."
            )
        record = _rich_artifact_record(payload, artifact_label)
        if str(record.get("sha256") or "").strip().lower() != actual_hash:
            raise ProductionBundleError(
                f"{label} model_artifacts.{artifact_label}.sha256 disagrees with "
                f"{hash_field}."
            )
        model_path = model_paths[artifact_label]
        if _optional_int(record, "bytes") != _file_size_bytes(model_path):
            raise ProductionBundleError(
                f"{label} model_artifacts.{artifact_label}.bytes does not match "
                "the active artifact."
            )
        spec, _ = _embedded_training_spec_payload(
            model_path,
            label=f"{artifact_label} model",
        )
        embedded_specs[artifact_label] = spec
        if (
            record.get("embedded_training_spec") != spec
            or record.get("embedded_training_spec_sha256")
            != _canonical_json_sha256(spec)
        ):
            raise ProductionBundleError(
                f"{label} model_artifacts.{artifact_label} embeds stale training "
                "spec identity."
            )

    if embedded_specs["logistic"] != embedded_specs["primary"]:
        raise ProductionBundleError(
            f"{label} rich logistic training spec does not match the primary spec."
        )
    if embedded_specs["no_odds"] != _expected_no_odds_spec_payload(
        embedded_specs["primary"]
    ):
        raise ProductionBundleError(
            f"{label} rich no-odds training spec is not the deterministic no-odds "
            "variant of the primary spec."
        )
    if (
        _optional_string(payload, "model_spec_name")
        != str(embedded_specs["primary"].get("name") or "").strip()
        or _optional_string(payload, "no_odds_model_spec_name")
        != str(embedded_specs["no_odds"].get("name") or "").strip()
    ):
        raise ProductionBundleError(
            f"{label} rich core spec names disagree with the active model contracts."
        )

    saved = payload.get("saved_fullfit_spec")
    if not isinstance(saved, dict):
        raise ProductionBundleError(f"{label} rich saved_fullfit_spec is missing.")
    sidecar_path = _manifest_relative_file(
        manifest_path,
        saved.get("staged_path"),
        label="saved full-fit training spec",
        release_root=release_root,
    )
    if (
        str(saved.get("sha256") or "").strip().lower() != _file_sha256(sidecar_path)
        or _optional_int(saved, "bytes") != _file_size_bytes(sidecar_path)
    ):
        raise ProductionBundleError(
            f"{label} rich saved_fullfit_spec fingerprint does not match its file."
        )
    try:
        sidecar_payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProductionBundleError(
            f"{label} rich saved full-fit training spec is invalid JSON: "
            f"{sidecar_path}: {exc}"
        ) from exc
    if saved.get("payload") != sidecar_payload or sidecar_payload != embedded_specs["primary"]:
        raise ProductionBundleError(
            f"{label} rich saved full-fit training spec does not match the primary "
            "model."
        )

    registered = payload.get("registered_training_specs")
    selected_fullfit = (
        registered.get("selected_fullfit") if isinstance(registered, dict) else None
    )
    selected_evaluation = (
        registered.get("selected_evaluation") if isinstance(registered, dict) else None
    )
    for record_label, record in (
        ("selected_fullfit", selected_fullfit),
        ("selected_evaluation", selected_evaluation),
    ):
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("payload"), dict)
            or record.get("sha256") != _canonical_json_sha256(record.get("payload"))
        ):
            raise ProductionBundleError(
                f"{label} rich registered_training_specs.{record_label} identity "
                "is invalid."
            )
    if _registered_contract_payload(selected_fullfit["payload"]) != _registered_contract_payload(
        embedded_specs["primary"]
    ):
        raise ProductionBundleError(
            f"{label} rich selected full-fit contract does not match the active model."
        )


def _validate_rich_source_snapshot_identity(
    payload: dict[str, Any],
    *,
    label: str,
) -> None:
    """Bind a source manifest's immutable snapshot section to its core fields."""
    snapshot = payload.get("immutable_training_snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("immutable") is not True:
        raise ProductionBundleError(
            f"{label} rich immutable_training_snapshot is not marked immutable."
        )
    if snapshot.get("snapshot_max_event_date") != payload.get("snapshot_max_event_date"):
        raise ProductionBundleError(
            f"{label} rich immutable snapshot date disagrees with the source core."
        )
    bindings = (
        ("fights", "processed_fights_sha256", "processed_fights_bytes"),
        ("features", "processed_features_sha256", "processed_features_bytes"),
    )
    for record_label, hash_field, bytes_field in bindings:
        record = snapshot.get(record_label)
        if (
            not isinstance(record, dict)
            or str(record.get("sha256") or "").strip().lower()
            != str(payload.get(hash_field) or "").strip().lower()
            or _optional_int(record, "bytes") != _optional_int(payload, bytes_field)
        ):
            raise ProductionBundleError(
                f"{label} rich immutable {record_label} identity disagrees with "
                "the source core fields."
            )


def _validate_authoritative_legacy_source_identity(
    payload: dict[str, Any],
    *,
    processed_fingerprints: dict[str, int | str],
    model_fingerprints: dict[str, str],
) -> None:
    """Require complete hashes before a legacy release may replace rich state."""
    expected = {
        **processed_fingerprints,
        **model_fingerprints,
    }
    required = (
        "model_sha256",
        "no_odds_model_sha256",
        "logistic_model_sha256",
        "processed_fights_sha256",
        "processed_features_sha256",
        "processed_fights_bytes",
        "processed_features_bytes",
    )
    missing = [key for key in required if payload.get(key) in (None, "")]
    if missing:
        raise ProductionBundleError(
            "Authoritative legacy source manifest is not fully pinned; missing: "
            f"{missing}."
        )
    mismatches: list[str] = []
    for key in required:
        declared: object = payload[key]
        actual = expected.get(key)
        if isinstance(actual, int):
            try:
                declared = int(declared)
            except (TypeError, ValueError):
                mismatches.append(key)
                continue
        else:
            declared = str(declared).strip().lower()
        if declared != actual:
            mismatches.append(key)
    if mismatches:
        raise ProductionBundleError(
            "Authoritative legacy source manifest does not match its exact "
            f"artifacts: {mismatches}."
        )


def _manifest_model_identity_matches(
    payload: dict[str, Any],
    *,
    model_spec_name: str,
    no_odds_model_spec_name: str | None,
) -> bool:
    if _optional_string(payload, "model_spec_name") != model_spec_name:
        return False
    payload_no_odds_spec = _optional_string(payload, "no_odds_model_spec_name")
    if (
        payload_no_odds_spec
        and no_odds_model_spec_name
        and payload_no_odds_spec != no_odds_model_spec_name
    ):
        return False
    return True


def _drop_identity_metadata(payload: dict[str, Any]) -> None:
    for key in _IDENTITY_SOURCE_METADATA_KEYS:
        payload.pop(key, None)


def _merge_source_metadata(
    payload: dict[str, Any],
    source_payload: dict[str, Any],
    *,
    model_spec_name: str,
    no_odds_model_spec_name: str | None,
) -> None:
    for key in _STABLE_SOURCE_METADATA_KEYS:
        if key in source_payload:
            payload[key] = source_payload[key]

    if not _manifest_model_identity_matches(
        source_payload,
        model_spec_name=model_spec_name,
        no_odds_model_spec_name=no_odds_model_spec_name,
    ):
        return

    for key in _IDENTITY_SOURCE_METADATA_KEYS:
        if key in source_payload:
            payload[key] = source_payload[key]


def _should_preserve_bundle_id(
    base_payload: dict[str, Any],
    *,
    model_spec_name: str,
    no_odds_model_spec_name: str | None,
    snapshot_max_event_date: str,
    model_sha256: str | None = None,
    no_odds_model_sha256: str | None = None,
    logistic_model_sha256: str | None = None,
) -> bool:
    bundle_id = _optional_string(base_payload, "bundle_id")
    if not bundle_id:
        return False
    if _optional_string(base_payload, "model_spec_name") != model_spec_name:
        return False
    if _optional_string(base_payload, "snapshot_max_event_date") != snapshot_max_event_date:
        return False
    existing_no_odds_spec_name = _optional_string(base_payload, "no_odds_model_spec_name")
    if existing_no_odds_spec_name and no_odds_model_spec_name and existing_no_odds_spec_name != no_odds_model_spec_name:
        return False
    # Same spec + snapshot date does not mean same model: a same-spec refit (or a
    # rollback to older binaries) changes the artifact bytes while keeping both.
    # When the base manifest pins model hashes, a mismatch is an identity change,
    # which lets built_at follow the artifact actually being served.
    existing_model_sha256 = _optional_string(base_payload, "model_sha256")
    if existing_model_sha256 and model_sha256 and existing_model_sha256 != model_sha256:
        return False
    existing_no_odds_sha256 = _optional_string(base_payload, "no_odds_model_sha256")
    if existing_no_odds_sha256 and no_odds_model_sha256 and existing_no_odds_sha256 != no_odds_model_sha256:
        return False
    existing_logistic_sha256 = _optional_string(base_payload, "logistic_model_sha256")
    if existing_logistic_sha256 and logistic_model_sha256 and existing_logistic_sha256 != logistic_model_sha256:
        return False
    if existing_logistic_sha256 and logistic_model_sha256 is None:
        return False
    return True


def _parse_manifest_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _resolve_bundle_built_at(
    *,
    base_payload: dict[str, Any],
    source_payload: dict[str, Any],
    preserve_bundle_id: bool,
    model_spec_name: str,
) -> str:
    source_built_at = _optional_string(source_payload, "built_at")
    source_model_spec_name = _optional_string(source_payload, "model_spec_name")
    source_spec_matches = bool(source_built_at) and (
        source_model_spec_name is None or source_model_spec_name == model_spec_name
    )

    if preserve_bundle_id:
        existing = _optional_string(base_payload, "built_at")
        if existing:
            # built_at must move forward monotonically: a redeployed image carrying a
            # same-spec refit has to roll the runtime manifest's built_at even though
            # bundle identity is preserved, or the freshness guard keeps reading the
            # pre-refit timestamp forever. Daily volume reconciles are unaffected
            # (their source built_at is never newer than the runtime's).
            if source_spec_matches:
                existing_ts = _parse_manifest_timestamp(existing)
                source_ts = _parse_manifest_timestamp(source_built_at)
                if existing_ts is not None and source_ts is not None and source_ts > existing_ts:
                    return source_built_at
            return existing

    if source_spec_matches:
        return source_built_at

    existing = _optional_string(base_payload, "built_at")
    existing_model_spec_name = _optional_string(base_payload, "model_spec_name")
    if existing and existing_model_spec_name == model_spec_name:
        return existing

    return _runtime_timestamp_now()


def load_production_bundle(manifest_path: str | Path | None = None) -> ProductionBundle:
    path = _resolve_manifest_path(manifest_path)
    if not path.exists():
        raise ProductionBundleError(f"Production bundle manifest not found: {path}")
    payload = _load_manifest_payload(path)
    _validate_rich_manifest_shape(payload, label="Production bundle")

    # Legacy manifests have only git_sha.  Historically that value meant the
    # training source in CI-authored manifests and the deployed commit after
    # Railway reconciliation, so it is safe only as a compatibility fallback
    # for training provenance.  deployed_git_sha remains unknown until an
    # explicit field is written by a hosted runtime.
    legacy_git_sha = _required_string(payload, "git_sha")
    training_source_git_sha = (
        _optional_string(payload, "training_source_git_sha") or legacy_git_sha
    )

    return ProductionBundle(
        manifest_path=path,
        bundle_id=_required_string(payload, "bundle_id"),
        model_spec_name=_required_string(payload, "model_spec_name"),
        model_path=_resolve_bundle_path(_required_string(payload, "model_path")),
        no_odds_model_path=_resolve_bundle_path(_required_string(payload, "no_odds_model_path")),
        processed_dir=_resolve_bundle_path(_required_string(payload, "processed_dir")),
        snapshot_max_event_date=_required_string(payload, "snapshot_max_event_date"),
        built_at=_required_string(payload, "built_at"),
        git_sha=legacy_git_sha,
        training_source_git_sha=training_source_git_sha,
        deployed_git_sha=_optional_string(payload, "deployed_git_sha"),
        no_odds_model_spec_name=str(payload.get("no_odds_model_spec_name") or "").strip() or None,
        logistic_model_path=(
            _resolve_bundle_path(str(payload["logistic_model_path"]).strip())
            if str(payload.get("logistic_model_path") or "").strip()
            else None
        ),
        processed_fights_sha256=_optional_string(payload, "processed_fights_sha256"),
        processed_features_sha256=_optional_string(payload, "processed_features_sha256"),
        processed_fights_bytes=_optional_int(payload, "processed_fights_bytes"),
        processed_features_bytes=_optional_int(payload, "processed_features_bytes"),
        manifest_updated_at=_optional_string(payload, "manifest_updated_at"),
        manifest_version=int(payload.get("manifest_version") or 1),
        model_sha256=_optional_string(payload, "model_sha256"),
        no_odds_model_sha256=_optional_string(payload, "no_odds_model_sha256"),
        logistic_model_sha256=_optional_string(payload, "logistic_model_sha256"),
    )


def load_production_bundle_or_none(
    manifest_path: str | Path | None = None,
) -> ProductionBundle | None:
    try:
        return load_production_bundle(manifest_path)
    except ProductionBundleError:
        return None


@lru_cache(maxsize=32)
def _cached_snapshot_max_event_date(
    processed_dir_str: str,
    fights_mtime_ns: int,
    features_mtime_ns: int,
) -> str | None:
    import pandas as pd

    processed_dir = Path(processed_dir_str)
    max_dates: list[str] = []

    for filename in ("fights_cleaned.csv", "features.csv"):
        path = processed_dir / filename
        if not path.exists():
            continue
        try:
            event_dates = pd.read_csv(path, usecols=["event_date"])["event_date"]
        except ValueError:
            continue
        parsed = pd.to_datetime(event_dates, errors="coerce").dropna()
        if parsed.empty:
            continue
        max_dates.append(parsed.max().date().isoformat())

    if not max_dates:
        return None
    return max(max_dates)


def get_processed_snapshot_max_event_date(processed_dir: Path) -> str | None:
    resolved_dir = _resolved_path(Path(processed_dir))
    fights_path = resolved_dir / "fights_cleaned.csv"
    features_path = resolved_dir / "features.csv"
    return _cached_snapshot_max_event_date(
        str(resolved_dir),
        _path_mtime_ns(fights_path),
        _path_mtime_ns(features_path),
    )


def get_model_artifact_fingerprints(
    model_path: Path,
    no_odds_model_path: Path,
    logistic_model_path: Path | None = None,
) -> dict[str, str]:
    _require_existing_file(model_path, label="primary model")
    _require_existing_file(no_odds_model_path, label="no-odds model")
    fingerprints = {
        "model_sha256": _file_sha256(model_path),
        "no_odds_model_sha256": _file_sha256(no_odds_model_path),
    }
    if logistic_model_path is not None:
        _require_existing_file(logistic_model_path, label="logistic model")
        fingerprints["logistic_model_sha256"] = _file_sha256(logistic_model_path)
    return fingerprints


def get_processed_snapshot_fingerprints(processed_dir: Path) -> dict[str, int | str]:
    resolved_dir = _resolved_path(Path(processed_dir))
    fights_path = resolved_dir / "fights_cleaned.csv"
    features_path = resolved_dir / "features.csv"
    _require_existing_file(fights_path, label="processed fights snapshot")
    _require_existing_file(features_path, label="processed features snapshot")
    return {
        "processed_fights_sha256": _file_sha256(fights_path),
        "processed_features_sha256": _file_sha256(features_path),
        "processed_fights_bytes": _file_size_bytes(fights_path),
        "processed_features_bytes": _file_size_bytes(features_path),
    }


def _embedded_training_spec_name(path: Path, model_result: dict[str, Any] | None = None) -> str:
    if model_result is None:
        import joblib

        model_result = joblib.load(path)
    spec = model_result.get("training_spec")
    if not isinstance(spec, dict):
        raise ProductionBundleError(f"Model artifact {path} is missing an embedded training_spec.")
    spec_name = str(spec.get("name") or "").strip()
    if not spec_name:
        raise ProductionBundleError(f"Model artifact {path} is missing training_spec.name.")
    return spec_name


def _artifact_path_from_model_result(model_result: dict[str, Any] | None) -> Path | None:
    if not isinstance(model_result, dict):
        return None
    raw = str(model_result.get("artifact_path") or "").strip()
    return Path(raw) if raw else None


def _require_existing_file(path: Path, *, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise ProductionBundleError(f"Production bundle {label} is missing: {path}")


def _require_matching_path(*, label: str, expected: Path, actual: Path) -> None:
    if _resolved_path(expected) != _resolved_path(actual):
        raise ProductionBundleError(
            f"Production bundle {label} mismatch: expected {expected}, got {actual}."
        )


def _resolve_validation_path(path: str | Path) -> Path:
    candidate = Path(_expand_tokens(str(path)))
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return _resolved_path(candidate)


def _validation_directories(
    *,
    expected_models_dir: str | Path | None,
    expected_processed_dir: str | Path | None,
) -> tuple[Path, Path, bool]:
    """Return the exact directories allowed for this validation.

    Omitting both arguments retains the active-runtime contract: canonical
    model aliases under ``MODELS_DIR`` plus the manifest-pinned runtime
    snapshot (which may live on a mounted volume). Supplying both opts into
    strict staged validation and also requires the exact processed directory.
    Requiring the pair prevents a candidate model from being accidentally
    validated against mutable runtime data (or vice versa).
    """
    if (expected_models_dir is None) != (expected_processed_dir is None):
        raise ProductionBundleError(
            "Staged production bundle validation requires both "
            "expected_models_dir and expected_processed_dir."
        )
    if expected_models_dir is None:
        return _resolved_path(MODELS_DIR), _resolved_path(PROCESSED_DATA_DIR), False
    staged_models_dir = _resolve_validation_path(expected_models_dir)
    staged_processed_dir = _resolve_validation_path(expected_processed_dir)
    if staged_models_dir == _resolved_path(MODELS_DIR):
        raise ProductionBundleError(
            "Staged production bundle models directory must be isolated from MODELS_DIR."
        )
    if staged_processed_dir == _resolved_path(PROCESSED_DATA_DIR):
        raise ProductionBundleError(
            "Staged production bundle processed directory must be isolated from "
            "PROCESSED_DATA_DIR."
        )
    return staged_models_dir, staged_processed_dir, True


def _require_staged_manifest_fingerprints(bundle: ProductionBundle) -> None:
    required = {
        "model_sha256": bundle.model_sha256,
        "no_odds_model_sha256": bundle.no_odds_model_sha256,
        "logistic_model_sha256": bundle.logistic_model_sha256,
        "processed_fights_sha256": bundle.processed_fights_sha256,
        "processed_features_sha256": bundle.processed_features_sha256,
        "processed_fights_bytes": bundle.processed_fights_bytes,
        "processed_features_bytes": bundle.processed_features_bytes,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ProductionBundleError(
            "Staged production bundle manifest is missing required fingerprints: "
            f"{missing}."
        )


def _load_model_result(
    path: Path,
    *,
    label: str,
    model_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if model_result is None:
        import joblib

        model_result = joblib.load(path)
    if not isinstance(model_result, dict):
        raise ProductionBundleError(
            f"Production bundle {label} artifact is not a model-result mapping: {path}."
        )
    return model_result


def _embedded_training_spec_payload(
    path: Path,
    *,
    label: str,
    model_result: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _load_model_result(path, label=label, model_result=model_result)
    spec = result.get("training_spec")
    if not isinstance(spec, dict):
        raise ProductionBundleError(
            f"Production bundle {label} artifact {path} is missing an embedded training_spec."
        )
    feature_cols = result.get("feature_cols")
    spec_feature_cols = spec.get("feature_cols")
    if not isinstance(feature_cols, list) or not isinstance(spec_feature_cols, list):
        raise ProductionBundleError(
            f"Production bundle {label} artifact {path} has an invalid feature contract."
        )
    if feature_cols != spec_feature_cols:
        raise ProductionBundleError(
            f"Production bundle {label} artifact {path} feature_cols do not exactly "
            "match embedded training_spec.feature_cols."
        )
    return deepcopy(spec), result


def _spec_payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _registered_contract_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(payload)
    # These values identify a particular fit, not the model-affecting named
    # contract registered in source. They must still match the saved sidecar.
    normalized.pop("git_hash", None)
    normalized.pop("trained_at", None)
    return normalized


def _expected_no_odds_spec_payload(primary_spec: dict[str, Any]) -> dict[str, Any]:
    from src.features.build_features import exclude_market_derived_features

    expected = deepcopy(primary_spec)
    primary_name = str(expected.get("name") or "").strip()
    primary_description = str(expected.get("description") or "").strip()
    primary_feature_cols = expected.get("feature_cols")
    if not primary_name or not isinstance(primary_feature_cols, list):
        raise ProductionBundleError(
            "Production bundle primary model has an invalid embedded training_spec."
        )
    expected["name"] = f"{primary_name}_no_odds"
    expected["description"] = (
        f"{primary_description} (no-odds variant)"
        if primary_description
        else "No-odds variant"
    )
    expected["feature_cols"] = exclude_market_derived_features(primary_feature_cols)
    return expected


def _validate_staged_training_specs(
    bundle: ProductionBundle,
    *,
    expected_models_dir: Path,
    saved_training_spec_path: str | Path | None,
    primary_model_result: dict[str, Any] | None,
    no_odds_model_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if bundle.logistic_model_path is None:
        raise ProductionBundleError(
            "Staged production bundle manifest must include logistic_model_path."
        )

    primary_spec, _ = _embedded_training_spec_payload(
        bundle.model_path,
        label="primary model",
        model_result=primary_model_result,
    )
    no_odds_spec, _ = _embedded_training_spec_payload(
        bundle.no_odds_model_path,
        label="no-odds model",
        model_result=no_odds_model_result,
    )
    logistic_spec, _ = _embedded_training_spec_payload(
        bundle.logistic_model_path,
        label="logistic model",
    )

    if logistic_spec != primary_spec:
        raise ProductionBundleError(
            "Staged production bundle logistic training_spec does not exactly "
            "match the primary model training_spec."
        )

    expected_no_odds_spec = _expected_no_odds_spec_payload(primary_spec)
    if no_odds_spec != expected_no_odds_spec:
        raise ProductionBundleError(
            "Staged production bundle no-odds training_spec differs from the "
            "primary contract beyond its documented no-odds variant fields."
        )

    try:
        from src.model.training_spec import resolve_named_training_spec

        registered_spec = asdict(
            resolve_named_training_spec(str(primary_spec.get("name") or ""))
        )
    except (TypeError, ValueError) as exc:
        raise ProductionBundleError(
            "Staged production bundle primary training_spec is not registered: "
            f"{exc}"
        ) from exc
    if (
        set(primary_spec) != set(registered_spec)
        or _registered_contract_payload(primary_spec)
        != _registered_contract_payload(registered_spec)
    ):
        raise ProductionBundleError(
            "Staged production bundle primary training_spec is incomplete or does "
            "not match the registered named training contract."
        )

    if saved_training_spec_path is None:
        spec_path = expected_models_dir / f"{bundle.model_spec_name}_spec.json"
    else:
        spec_path = _resolve_validation_path(saved_training_spec_path)
    _require_matching_path(
        label="saved training spec directory",
        expected=spec_path.parent,
        actual=expected_models_dir,
    )
    _require_existing_file(spec_path, label="saved training spec")
    try:
        saved_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProductionBundleError(
            f"Production bundle saved training spec is invalid JSON: {spec_path}: {exc}"
        ) from exc
    if not isinstance(saved_spec, dict) or saved_spec != primary_spec:
        raise ProductionBundleError(
            "Staged production bundle saved training spec does not exactly match "
            "the primary model embedded training_spec."
        )

    return {
        "saved_training_spec_path": str(spec_path),
        "saved_training_spec_sha256": _file_sha256(spec_path),
        "embedded_training_spec_sha256": _spec_payload_sha256(primary_spec),
        "embedded_no_odds_training_spec_sha256": _spec_payload_sha256(no_odds_spec),
        "embedded_logistic_training_spec_sha256": _spec_payload_sha256(logistic_spec),
        "registered_training_spec_name": str(primary_spec.get("name") or ""),
    }


def _validate_runtime_processed_snapshot(bundle: ProductionBundle) -> dict[str, Any]:
    _require_existing_file(bundle.manifest_path, label="manifest")

    fights_path = bundle.processed_dir / "fights_cleaned.csv"
    features_path = bundle.processed_dir / "features.csv"
    _require_existing_file(fights_path, label="processed fights snapshot")
    _require_existing_file(features_path, label="processed features snapshot")

    actual_snapshot_max_event_date = get_processed_snapshot_max_event_date(bundle.processed_dir)
    if actual_snapshot_max_event_date is None:
        raise ProductionBundleError(
            f"Production bundle processed snapshot in {bundle.processed_dir} has no usable event_date column."
        )
    if actual_snapshot_max_event_date < bundle.snapshot_max_event_date:
        raise ProductionBundleError(
            "Production bundle processed snapshot is older than expected: "
            f"{actual_snapshot_max_event_date} < {bundle.snapshot_max_event_date}."
        )

    actual_processed_fingerprints = get_processed_snapshot_fingerprints(bundle.processed_dir)
    if (
        bundle.processed_fights_sha256 is not None
        and actual_processed_fingerprints["processed_fights_sha256"] != bundle.processed_fights_sha256
    ):
        raise ProductionBundleError(
            "Production bundle processed fights snapshot hash mismatch: "
            f"manifest expects {bundle.processed_fights_sha256}, "
            f"artifact is {actual_processed_fingerprints['processed_fights_sha256']}."
        )
    if (
        bundle.processed_features_sha256 is not None
        and actual_processed_fingerprints["processed_features_sha256"] != bundle.processed_features_sha256
    ):
        raise ProductionBundleError(
            "Production bundle processed features snapshot hash mismatch: "
            f"manifest expects {bundle.processed_features_sha256}, "
            f"artifact is {actual_processed_fingerprints['processed_features_sha256']}."
        )
    if (
        bundle.processed_fights_bytes is not None
        and actual_processed_fingerprints["processed_fights_bytes"] != bundle.processed_fights_bytes
    ):
        raise ProductionBundleError(
            "Production bundle processed fights snapshot size mismatch: "
            f"manifest expects {bundle.processed_fights_bytes}, "
            f"artifact is {actual_processed_fingerprints['processed_fights_bytes']}."
        )
    if (
        bundle.processed_features_bytes is not None
        and actual_processed_fingerprints["processed_features_bytes"] != bundle.processed_features_bytes
    ):
        raise ProductionBundleError(
            "Production bundle processed features snapshot size mismatch: "
            f"manifest expects {bundle.processed_features_bytes}, "
            f"artifact is {actual_processed_fingerprints['processed_features_bytes']}."
        )

    return {
        "manifest_snapshot_max_event_date": bundle.snapshot_max_event_date,
        "processed_snapshot_max_event_date": actual_snapshot_max_event_date,
        **actual_processed_fingerprints,
    }


def reconcile_production_bundle_manifest(
    *,
    target_manifest_path: str | Path | None = None,
    source_manifest_path: str | Path | None = None,
    model_path: str | Path | None = None,
    no_odds_model_path: str | Path | None = None,
    logistic_model_path: str | Path | None = None,
    processed_dir: str | Path | None = None,
    authoritative_source_manifest: bool = False,
) -> dict[str, Any]:
    target_path = _resolve_manifest_path(target_manifest_path)

    base_payload: dict[str, Any] = {}
    target_payload: dict[str, Any] = {}
    source_payload: dict[str, Any] = {}
    source_path: Path | None = None
    if target_path.exists():
        try:
            target_payload = _load_manifest_payload(target_path)
        except ProductionBundleError:
            target_payload = {}
    if source_manifest_path is not None:
        source_path = _resolve_manifest_path(source_manifest_path)
        if not source_path.is_file():
            raise ProductionBundleError(
                f"Source production bundle manifest not found: {source_path}"
            )
        source_payload = _load_manifest_payload(source_path)
    elif DEFAULT_MANIFEST_PATH.exists() and _resolved_path(target_path) != _resolved_path(DEFAULT_MANIFEST_PATH):
        source_path = DEFAULT_MANIFEST_PATH
        source_payload = _load_manifest_payload(DEFAULT_MANIFEST_PATH)
    if authoritative_source_manifest and (source_path is None or not source_payload):
        raise ProductionBundleError(
            "An authoritative source activation requires an explicit source manifest."
        )

    resolved_model_path = _resolved_path(Path(model_path) if model_path is not None else MODELS_DIR / "xgboost_model.pkl")
    resolved_no_odds_model_path = _resolved_path(
        Path(no_odds_model_path) if no_odds_model_path is not None else MODELS_DIR / "xgboost_no_odds_model.pkl"
    )
    resolved_processed_dir = _resolved_path(Path(processed_dir) if processed_dir is not None else PROCESSED_DATA_DIR)

    resolved_logistic_model_path: Path | None
    if logistic_model_path is not None:
        candidate = _resolved_path(Path(logistic_model_path))
        resolved_logistic_model_path = candidate if candidate.exists() else None
    else:
        candidate = MODELS_DIR / "logistic_model.pkl"
        resolved_logistic_model_path = _resolved_path(candidate) if candidate.exists() else None

    model_spec_name = _embedded_training_spec_name(resolved_model_path)
    no_odds_spec_name = _embedded_training_spec_name(resolved_no_odds_model_path)
    snapshot_max_event_date = get_processed_snapshot_max_event_date(resolved_processed_dir)
    if snapshot_max_event_date is None:
        raise ProductionBundleError(
            f"Production bundle processed snapshot in {resolved_processed_dir} has no usable event_date column."
        )
    processed_fingerprints = get_processed_snapshot_fingerprints(resolved_processed_dir)
    model_fingerprints = get_model_artifact_fingerprints(
        resolved_model_path,
        resolved_no_odds_model_path,
        resolved_logistic_model_path,
    )

    source_is_rich = False
    if source_payload:
        source_is_rich = _validate_rich_manifest_shape(
            source_payload,
            label="Source production bundle",
        )
    if (
        (source_is_rich or authoritative_source_manifest)
        and source_path is not None
        and _resolved_path(source_path) == _resolved_path(target_path)
    ):
        raise ProductionBundleError(
            "Production release manifests are immutable; reconcile into a separate "
            "runtime manifest path."
        )
    # A complete, validated rich source is authoritative and may safely replace
    # even a corrupt/stale rich runtime payload. Without that replacement, a
    # rich target must itself be structurally valid and bound to the binaries.
    target_is_rich = False
    if target_payload and not source_is_rich and not authoritative_source_manifest:
        target_is_rich = _validate_rich_manifest_shape(
            target_payload,
            label="Runtime production bundle",
        )

    model_paths = {
        "primary": resolved_model_path,
        "no_odds": resolved_no_odds_model_path,
    }
    if resolved_logistic_model_path is not None:
        model_paths["logistic"] = resolved_logistic_model_path
    if source_is_rich:
        if resolved_logistic_model_path is None:
            raise ProductionBundleError(
                "Source rich production bundle requires an active logistic model."
            )
        _validate_rich_source_snapshot_identity(
            source_payload,
            label="Source production bundle",
        )
        _validate_rich_model_identity(
            source_payload,
            manifest_path=source_path or target_path,
            model_paths=model_paths,
            actual_fingerprints=model_fingerprints,
            label="Source production bundle",
            release_root=(source_path or target_path).parent,
        )
    elif authoritative_source_manifest:
        _validate_authoritative_legacy_source_identity(
            source_payload,
            processed_fingerprints=processed_fingerprints,
            model_fingerprints=model_fingerprints,
        )
    elif target_is_rich:
        if resolved_logistic_model_path is None:
            raise ProductionBundleError(
                "Runtime rich production bundle requires an active logistic model."
            )
        _validate_rich_model_identity(
            target_payload,
            manifest_path=target_path,
            model_paths=model_paths,
            actual_fingerprints=model_fingerprints,
            label="Runtime production bundle",
            release_root=_rich_release_root(target_payload, target_path),
        )

    base_payload = target_payload or source_payload
    manifest_updated_at = _runtime_timestamp_now()
    preserve_bundle_id = _should_preserve_bundle_id(
        base_payload,
        model_spec_name=model_spec_name,
        no_odds_model_spec_name=no_odds_spec_name,
        snapshot_max_event_date=snapshot_max_event_date,
        model_sha256=model_fingerprints["model_sha256"],
        no_odds_model_sha256=model_fingerprints["no_odds_model_sha256"],
        logistic_model_sha256=model_fingerprints.get("logistic_model_sha256"),
    )
    if source_is_rich:
        # A legacy target with missing hashes cannot prove it represents this
        # model generation. Treat it as an identity change so built_at and the
        # bundle identity come from the authoritative rich release instead of
        # carrying forward a same-spec timestamp by name alone.
        target_model_identity_matches = all(
            _optional_string(target_payload, field) == model_fingerprints.get(field)
            for field in _RICH_MODEL_FIELDS
        )
        preserve_bundle_id = preserve_bundle_id and target_model_identity_matches

    if source_is_rich or authoritative_source_manifest:
        # Rich provenance belongs to a particular set of model bytes. Replace
        # it wholesale; piecemeal merging is exactly how a same-spec refit can
        # retain stale selection evidence, inventories, or rollback identity.
        payload = deepcopy(source_payload)
        if source_is_rich:
            payload["rich_release_root"] = str(
                _resolved_path((source_path or target_path).parent)
            )
        else:
            payload.pop("rich_release_root", None)
    elif target_is_rich:
        payload = deepcopy(target_payload)
    else:
        payload = dict(base_payload)
        if not _manifest_model_identity_matches(
            base_payload,
            model_spec_name=model_spec_name,
            no_odds_model_spec_name=no_odds_spec_name,
        ):
            _drop_identity_metadata(payload)
        if source_payload:
            _merge_source_metadata(
                payload,
                source_payload,
                model_spec_name=model_spec_name,
                no_odds_model_spec_name=no_odds_spec_name,
            )
    provenance_payload = source_payload or base_payload
    training_source_git_sha = _determine_training_source_git_sha(provenance_payload)
    deployed_git_sha = _determine_deployed_git_sha(provenance_payload)
    payload.update(
        {
            "manifest_version": int(payload.get("manifest_version") or 1),
            "bundle_id": (
                _optional_string(source_payload, "bundle_id")
                if source_is_rich or authoritative_source_manifest
                else _optional_string(base_payload, "bundle_id")
                if preserve_bundle_id
                else _default_bundle_id(
                    model_spec_name=model_spec_name,
                    snapshot_max_event_date=snapshot_max_event_date,
                )
            ),
            "model_spec_name": model_spec_name,
            "no_odds_model_spec_name": no_odds_spec_name,
            "model_path": str(resolved_model_path),
            "no_odds_model_path": str(resolved_no_odds_model_path),
            "processed_dir": str(resolved_processed_dir),
            "snapshot_max_event_date": snapshot_max_event_date,
            "built_at": _resolve_bundle_built_at(
                base_payload=base_payload,
                source_payload=source_payload,
                preserve_bundle_id=preserve_bundle_id,
                model_spec_name=model_spec_name,
            ),
            "training_source_git_sha": training_source_git_sha,
            # Keep git_sha for older readers.  Its precise compatibility
            # semantics are documented by _determine_git_sha above.
            "git_sha": deployed_git_sha or training_source_git_sha,
            "manifest_updated_at": manifest_updated_at,
            **processed_fingerprints,
            **model_fingerprints,
        }
    )
    if resolved_logistic_model_path is not None:
        payload["logistic_model_path"] = str(resolved_logistic_model_path)
    else:
        payload.pop("logistic_model_path", None)
        payload.pop("logistic_model_sha256", None)
    if deployed_git_sha is not None:
        payload["deployed_git_sha"] = deployed_git_sha
    else:
        payload.pop("deployed_git_sha", None)

    write_json_atomically(payload, target_path)
    summary = validate_production_bundle(load_production_bundle(target_path))
    summary["source_manifest_path"] = str(source_path) if source_path is not None else None
    return summary


def runtime_manifest_needs_source_bootstrap(
    *,
    target_manifest_path: str | Path | None = None,
    source_manifest_path: str | Path | None = None,
) -> bool:
    if source_manifest_path is not None:
        load_production_bundle(source_manifest_path)
    target_bundle = load_production_bundle_or_none(target_manifest_path)
    if target_bundle is None:
        return True
    try:
        _validate_runtime_processed_snapshot(target_bundle)
    except ProductionBundleError:
        return True
    return False


def validate_production_bundle(
    bundle: ProductionBundle,
    *,
    primary_model_result: dict[str, Any] | None = None,
    no_odds_model_result: dict[str, Any] | None = None,
    expected_models_dir: str | Path | None = None,
    expected_processed_dir: str | Path | None = None,
    saved_training_spec_path: str | Path | None = None,
) -> dict[str, Any]:
    validation_models_dir, validation_processed_dir, staged_validation = (
        _validation_directories(
            expected_models_dir=expected_models_dir,
            expected_processed_dir=expected_processed_dir,
        )
    )
    if staged_validation:
        _require_matching_path(
            label="processed snapshot path",
            expected=bundle.processed_dir,
            actual=validation_processed_dir,
        )
    processed_snapshot_summary = _validate_runtime_processed_snapshot(bundle)
    _require_existing_file(bundle.model_path, label="primary model")
    _require_existing_file(bundle.no_odds_model_path, label="no-odds model")
    if bundle.logistic_model_path is not None:
        _require_existing_file(bundle.logistic_model_path, label="logistic model")

    _require_matching_path(
        label="primary alias path",
        expected=bundle.model_path,
        actual=validation_models_dir / "xgboost_model.pkl",
    )
    _require_matching_path(
        label="no-odds alias path",
        expected=bundle.no_odds_model_path,
        actual=validation_models_dir / "xgboost_no_odds_model.pkl",
    )
    if bundle.logistic_model_path is not None:
        _require_matching_path(
            label="logistic alias path",
            expected=bundle.logistic_model_path,
            actual=validation_models_dir / "logistic_model.pkl",
        )

    if staged_validation:
        _require_staged_manifest_fingerprints(bundle)
        if bundle.logistic_model_path is None:
            raise ProductionBundleError(
                "Staged production bundle manifest must include logistic_model_path."
            )

    actual_model_sha256 = _file_sha256(bundle.model_path)
    actual_no_odds_sha256 = _file_sha256(bundle.no_odds_model_path)
    actual_logistic_sha256 = (
        _file_sha256(bundle.logistic_model_path)
        if bundle.logistic_model_path is not None
        else None
    )
    if bundle.model_sha256:
        if actual_model_sha256 != bundle.model_sha256:
            raise ProductionBundleError(
                "Production bundle primary model hash mismatch: "
                f"manifest pins {bundle.model_sha256}, artifact is {actual_model_sha256}."
            )
    if bundle.no_odds_model_sha256:
        if actual_no_odds_sha256 != bundle.no_odds_model_sha256:
            raise ProductionBundleError(
                "Production bundle no-odds model hash mismatch: "
                f"manifest pins {bundle.no_odds_model_sha256}, artifact is {actual_no_odds_sha256}."
            )
    if bundle.logistic_model_sha256:
        if bundle.logistic_model_path is None:
            raise ProductionBundleError(
                "Production bundle pins a logistic model hash but has no logistic model path."
            )
        if actual_logistic_sha256 != bundle.logistic_model_sha256:
            raise ProductionBundleError(
                "Production bundle logistic model hash mismatch: "
                f"manifest pins {bundle.logistic_model_sha256}, artifact is {actual_logistic_sha256}."
            )

    manifest_payload = _load_manifest_payload(bundle.manifest_path)
    rich_manifest = _validate_rich_manifest_shape(
        manifest_payload,
        label="Production bundle",
    )
    scheduled_refit_policy = _validated_scheduled_refit_policy(manifest_payload)
    if rich_manifest:
        if bundle.logistic_model_path is None or actual_logistic_sha256 is None:
            raise ProductionBundleError(
                "Rich production bundle requires an active logistic model."
            )
        _validate_rich_model_identity(
            manifest_payload,
            manifest_path=bundle.manifest_path,
            model_paths={
                "primary": bundle.model_path,
                "no_odds": bundle.no_odds_model_path,
                "logistic": bundle.logistic_model_path,
            },
            actual_fingerprints={
                "model_sha256": actual_model_sha256,
                "no_odds_model_sha256": actual_no_odds_sha256,
                "logistic_model_sha256": actual_logistic_sha256,
            },
            label="Production bundle",
            release_root=_rich_release_root(manifest_payload, bundle.manifest_path),
        )

    loaded_primary_path = _artifact_path_from_model_result(primary_model_result)
    if loaded_primary_path is not None:
        _require_matching_path(
            label="loaded primary model",
            expected=bundle.model_path,
            actual=loaded_primary_path,
        )
    loaded_no_odds_path = _artifact_path_from_model_result(no_odds_model_result)
    if loaded_no_odds_path is not None:
        _require_matching_path(
            label="loaded no-odds model",
            expected=bundle.no_odds_model_path,
            actual=loaded_no_odds_path,
        )

    embedded_model_spec_name = _embedded_training_spec_name(
        bundle.model_path,
        model_result=primary_model_result,
    )
    if embedded_model_spec_name != bundle.model_spec_name:
        raise ProductionBundleError(
            "Production bundle model spec mismatch: "
            f"manifest expects {bundle.model_spec_name}, artifact embeds {embedded_model_spec_name}."
        )

    embedded_no_odds_model_spec_name = _embedded_training_spec_name(
        bundle.no_odds_model_path,
        model_result=no_odds_model_result,
    )
    expected_no_odds_spec = bundle.no_odds_model_spec_name
    if expected_no_odds_spec and embedded_no_odds_model_spec_name != expected_no_odds_spec:
        raise ProductionBundleError(
            "Production bundle no-odds model spec mismatch: "
            f"manifest expects {expected_no_odds_spec}, artifact embeds {embedded_no_odds_model_spec_name}."
        )

    staged_spec_summary: dict[str, Any] = {}
    if staged_validation:
        staged_spec_summary = _validate_staged_training_specs(
            bundle,
            expected_models_dir=validation_models_dir,
            saved_training_spec_path=saved_training_spec_path,
            primary_model_result=primary_model_result,
            no_odds_model_result=no_odds_model_result,
        )

    rich_release_identity: dict[str, Any] = {
        "rich_release_id": None,
        "installed_manifest_path": None,
        "installed_manifest_sha256": None,
        "source_manifest_sha256": None,
        "immutable_training_fights_sha256": None,
        "immutable_training_features_sha256": None,
        "immutable_training_snapshot_max_event_date": None,
        "independent_audit_fights_canonical_sha256": None,
        "independent_audit_features_canonical_sha256": None,
        "scheduled_bfo_lineage_manifest_sha256": None,
        "scheduled_bfo_lineage_batch_count": 0,
    }
    if rich_manifest:
        release_root = _rich_release_root(manifest_payload, bundle.manifest_path)
        installed_manifest_path = release_root / "manifest.json"
        source_manifest_path = (
            release_root / "provenance" / "source_staging_manifest.json"
        )
        immutable_snapshot = manifest_payload.get("immutable_training_snapshot")
        immutable_fights = (
            immutable_snapshot.get("fights")
            if isinstance(immutable_snapshot, dict)
            else None
        )
        immutable_features = (
            immutable_snapshot.get("features")
            if isinstance(immutable_snapshot, dict)
            else None
        )
        rich_release_identity = {
            "rich_release_id": release_root.name,
            "installed_manifest_path": (
                str(installed_manifest_path)
                if installed_manifest_path.is_file()
                else None
            ),
            "installed_manifest_sha256": (
                _file_sha256(installed_manifest_path)
                if installed_manifest_path.is_file()
                else None
            ),
            "source_manifest_sha256": (
                _file_sha256(source_manifest_path)
                if source_manifest_path.is_file()
                else None
            ),
            "immutable_training_fights_sha256": (
                str(immutable_fights.get("sha256") or "").strip().lower()
                if isinstance(immutable_fights, dict)
                else None
            ),
            "immutable_training_features_sha256": (
                str(immutable_features.get("sha256") or "").strip().lower()
                if isinstance(immutable_features, dict)
                else None
            ),
            "immutable_training_snapshot_max_event_date": (
                immutable_snapshot.get("snapshot_max_event_date")
                if isinstance(immutable_snapshot, dict)
                else None
            ),
            **_independent_audit_summary(
                manifest_payload,
                manifest_path=bundle.manifest_path,
                release_root=release_root,
            ),
            **_scheduled_bfo_lineage_summary(
                manifest_payload,
                manifest_path=bundle.manifest_path,
                release_root=release_root,
                required=scheduled_refit_policy is not None,
            ),
        }

    return {
        "bundle_id": bundle.bundle_id,
        "manifest_path": str(bundle.manifest_path),
        "manifest_version": bundle.manifest_version,
        "model_spec_name": bundle.model_spec_name,
        "embedded_model_spec_name": embedded_model_spec_name,
        "model_path": str(bundle.model_path),
        "no_odds_model_spec_name": expected_no_odds_spec or embedded_no_odds_model_spec_name,
        "embedded_no_odds_model_spec_name": embedded_no_odds_model_spec_name,
        "no_odds_model_path": str(bundle.no_odds_model_path),
        "model_sha256": actual_model_sha256,
        "no_odds_model_sha256": actual_no_odds_sha256,
        "logistic_model_path": str(bundle.logistic_model_path) if bundle.logistic_model_path else None,
        "logistic_model_sha256": actual_logistic_sha256,
        "model_hashes": {
            "primary_sha256": actual_model_sha256,
            "no_odds_sha256": actual_no_odds_sha256,
            "logistic_sha256": actual_logistic_sha256,
        },
        "rich_manifest_validated": rich_manifest,
        "scheduled_refit_policy": scheduled_refit_policy,
        "staging_schema_version": (
            _optional_int(manifest_payload, "staging_schema_version")
            if rich_manifest
            else None
        ),
        "rich_release_root": (
            str(_rich_release_root(manifest_payload, bundle.manifest_path))
            if rich_manifest
            else None
        ),
        **rich_release_identity,
        "processed_dir": str(bundle.processed_dir),
        **processed_snapshot_summary,
        "built_at": bundle.built_at,
        "manifest_updated_at": bundle.manifest_updated_at,
        "git_sha": bundle.git_sha,
        "training_source_git_sha": bundle.training_source_git_sha,
        "deployed_git_sha": bundle.deployed_git_sha,
        "validation_scope": "staged" if staged_validation else "canonical_runtime",
        "expected_models_dir": str(validation_models_dir),
        "expected_processed_dir": (
            str(validation_processed_dir) if staged_validation else None
        ),
        **staged_spec_summary,
    }
