"""Fail-closed identity binding for a candidate and its installed manifest.

This module only validates already-created manifests and the install receipt.
It does not install, activate, rewrite, or otherwise mutate a production bundle.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class CandidateActiveParityError(RuntimeError):
    """Raised when candidate-to-active immutable parity cannot be proved."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_MODEL_HASH_FIELDS = {
    "primary": "model_sha256",
    "no_odds": "no_odds_model_sha256",
    "logistic": "logistic_model_sha256",
}
_DATA_RECORDS = {
    "fights": ("processed_fights_sha256", "processed_fights_bytes"),
    "features": ("processed_features_sha256", "processed_features_bytes"),
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CandidateActiveParityError(f"Manifest identity is unreadable: {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise CandidateActiveParityError(f"Duplicate JSON key: {key!r}")
        payload[key] = value
    return payload


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CandidateActiveParityError(f"{label} is missing: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        CandidateActiveParityError,
    ) as exc:
        raise CandidateActiveParityError(f"{label} is unreadable JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CandidateActiveParityError(f"{label} must be a JSON object: {path}")
    return payload


def _mapping(payload: Mapping[str, Any], field: str, *, label: str) -> Mapping[str, Any]:
    value = payload.get(field)
    if not isinstance(value, Mapping):
        raise CandidateActiveParityError(f"{label}.{field} is missing or not an object")
    return value


def _string(payload: Mapping[str, Any], field: str, *, label: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CandidateActiveParityError(f"{label}.{field} is missing or blank")
    return value.strip()


def _integer(payload: Mapping[str, Any], field: str, *, label: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CandidateActiveParityError(f"{label}.{field} is missing or not a nonnegative integer")
    return value


def _sha256(payload: Mapping[str, Any], field: str, *, label: str) -> str:
    value = _string(payload, field, label=label).lower()
    if not _SHA256_RE.fullmatch(value):
        raise CandidateActiveParityError(f"{label}.{field} is not a SHA-256")
    return value


def _git_sha(payload: Mapping[str, Any], field: str, *, label: str) -> str:
    value = _string(payload, field, label=label).lower()
    if not _GIT_SHA_RE.fullmatch(value):
        raise CandidateActiveParityError(f"{label}.{field} is not a full Git SHA")
    return value


def _validated_spec_record(
    payload: Mapping[str, Any],
    field: str,
    *,
    label: str,
) -> Mapping[str, Any]:
    record = _mapping(payload, field, label=label)
    spec_payload = _mapping(record, "payload", label=f"{label}.{field}")
    declared_sha = _sha256(record, "sha256", label=f"{label}.{field}")
    actual_sha = _canonical_sha256(spec_payload)
    if declared_sha != actual_sha:
        raise CandidateActiveParityError(
            f"{label}.{field}.sha256 does not bind its specification payload"
        )
    return record


def _manifest_parity_identity(payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if payload.get("manifest_version") != 3 or payload.get("staging_schema_version") != 1:
        raise CandidateActiveParityError(
            f"{label} must be a rich manifest_version=3, staging_schema_version=1 bundle"
        )

    source_identity = _mapping(payload, "source_identity", label=label)
    source_base_git_sha = _git_sha(
        source_identity,
        "base_git_sha",
        label=f"{label}.source_identity",
    )
    source_diff_sha256 = _sha256(
        source_identity,
        "tracked_diff_sha256",
        label=f"{label}.source_identity",
    )

    model_artifacts = _mapping(payload, "model_artifacts", label=label)
    model_identity: dict[str, Any] = {}
    for role, top_level_hash_field in _MODEL_HASH_FIELDS.items():
        artifact = _mapping(model_artifacts, role, label=f"{label}.model_artifacts")
        top_level_sha = _sha256(payload, top_level_hash_field, label=label)
        artifact_sha = _sha256(artifact, "sha256", label=f"{label}.model_artifacts.{role}")
        if artifact_sha != top_level_sha:
            raise CandidateActiveParityError(
                f"{label}.model_artifacts.{role}.sha256 disagrees with {top_level_hash_field}"
            )
        embedded_spec = _mapping(
            artifact,
            "embedded_training_spec",
            label=f"{label}.model_artifacts.{role}",
        )
        embedded_spec_sha = _sha256(
            artifact,
            "embedded_training_spec_sha256",
            label=f"{label}.model_artifacts.{role}",
        )
        if embedded_spec_sha != _canonical_sha256(embedded_spec):
            raise CandidateActiveParityError(
                f"{label}.model_artifacts.{role}.embedded_training_spec_sha256 "
                "does not bind its specification payload"
            )
        model_identity[role] = {
            "sha256": top_level_sha,
            "bytes": _integer(artifact, "bytes", label=f"{label}.model_artifacts.{role}"),
            "feature_count": _integer(
                artifact,
                "feature_count",
                label=f"{label}.model_artifacts.{role}",
            ),
            "embedded_training_spec_sha256": embedded_spec_sha,
        }

    registered_specs = _mapping(payload, "registered_training_specs", label=label)
    selected_evaluation = _validated_spec_record(
        registered_specs,
        "selected_evaluation",
        label=f"{label}.registered_training_specs",
    )
    selected_fullfit = _validated_spec_record(
        registered_specs,
        "selected_fullfit",
        label=f"{label}.registered_training_specs",
    )
    saved_spec = _mapping(payload, "saved_fullfit_spec", label=label)
    saved_spec_payload = _mapping(saved_spec, "payload", label=f"{label}.saved_fullfit_spec")
    saved_spec_name = _string(
        saved_spec_payload,
        "name",
        label=f"{label}.saved_fullfit_spec.payload",
    )
    model_spec_name = _string(payload, "model_spec_name", label=label)
    if saved_spec_name != model_spec_name:
        raise CandidateActiveParityError(
            f"{label}.saved_fullfit_spec.payload.name disagrees with model_spec_name"
        )
    saved_spec_payload_sha = _canonical_sha256(saved_spec_payload)
    if _string(
        selected_fullfit["payload"],
        "name",
        label=f"{label}.registered_training_specs.selected_fullfit.payload",
    ) != saved_spec_name:
        raise CandidateActiveParityError(
            f"{label} selected full-fit specification name does not match saved_fullfit_spec.payload"
        )
    if model_identity["primary"]["embedded_training_spec_sha256"] != saved_spec_payload_sha:
        raise CandidateActiveParityError(
            f"{label} primary embedded specification does not match saved_fullfit_spec.payload"
        )
    if model_identity["logistic"]["embedded_training_spec_sha256"] != saved_spec_payload_sha:
        raise CandidateActiveParityError(
            f"{label} logistic embedded specification does not match saved_fullfit_spec.payload"
        )
    no_odds_embedded = _mapping(
        _mapping(model_artifacts, "no_odds", label=f"{label}.model_artifacts"),
        "embedded_training_spec",
        label=f"{label}.model_artifacts.no_odds",
    )
    if _string(
        no_odds_embedded,
        "name",
        label=f"{label}.model_artifacts.no_odds.embedded_training_spec",
    ) != _string(payload, "no_odds_model_spec_name", label=label):
        raise CandidateActiveParityError(
            f"{label} no-odds embedded specification disagrees with no_odds_model_spec_name"
        )

    immutable_snapshot = _mapping(payload, "immutable_training_snapshot", label=label)
    if immutable_snapshot.get("immutable") is not True:
        raise CandidateActiveParityError(
            f"{label}.immutable_training_snapshot is not marked immutable"
        )
    snapshot_date = _string(
        immutable_snapshot,
        "snapshot_max_event_date",
        label=f"{label}.immutable_training_snapshot",
    )
    manifest_snapshot_date = _string(payload, "snapshot_max_event_date", label=label)
    if snapshot_date != manifest_snapshot_date:
        raise CandidateActiveParityError(
            f"{label}.immutable_training_snapshot date disagrees with snapshot_max_event_date"
        )
    data_identity: dict[str, Any] = {}
    for record_name, (top_level_sha_field, top_level_bytes_field) in _DATA_RECORDS.items():
        record = _mapping(
            immutable_snapshot,
            record_name,
            label=f"{label}.immutable_training_snapshot",
        )
        top_level_sha = _sha256(payload, top_level_sha_field, label=label)
        top_level_bytes = _integer(payload, top_level_bytes_field, label=label)
        record_label = f"{label}.immutable_training_snapshot.{record_name}"
        if _sha256(record, "sha256", label=record_label) != top_level_sha:
            raise CandidateActiveParityError(
                f"{label}.immutable_training_snapshot.{record_name}.sha256 disagrees "
                f"with {top_level_sha_field}"
            )
        if _integer(record, "bytes", label=record_label) != top_level_bytes:
            raise CandidateActiveParityError(
                f"{label}.immutable_training_snapshot.{record_name}.bytes disagrees "
                f"with {top_level_bytes_field}"
            )
        data_identity[record_name] = {
            "sha256": top_level_sha,
            "bytes": top_level_bytes,
            "rows": _integer(
                record,
                "rows",
                label=f"{label}.immutable_training_snapshot.{record_name}",
            ),
            "columns": _integer(
                record,
                "columns",
                label=f"{label}.immutable_training_snapshot.{record_name}",
            ),
            "column_names_sha256": _sha256(
                record,
                "column_names_sha256",
                label=f"{label}.immutable_training_snapshot.{record_name}",
            ),
            "max_event_date": _string(
                record,
                "max_event_date",
                label=f"{label}.immutable_training_snapshot.{record_name}",
            ),
        }

    return {
        "release": {
            "bundle_id": _string(payload, "bundle_id", label=label),
            "snapshot_max_event_date": manifest_snapshot_date,
            "git_sha": _git_sha(payload, "git_sha", label=label),
            "training_source_git_sha": _git_sha(payload, "training_source_git_sha", label=label),
            "source_base_git_sha": source_base_git_sha,
            "source_tracked_diff_sha256": source_diff_sha256,
            "source_identity_sha256": _canonical_sha256(source_identity),
        },
        "models": {
            "model_spec_name": model_spec_name,
            "no_odds_model_spec_name": _string(payload, "no_odds_model_spec_name", label=label),
            "artifacts": model_identity,
            "model_artifacts_sha256": _canonical_sha256(model_artifacts),
        },
        "data": {
            "artifacts": data_identity,
            "immutable_training_snapshot_sha256": _canonical_sha256(immutable_snapshot),
        },
        "specs": {
            "selected_evaluation_sha256": _sha256(
                selected_evaluation,
                "sha256",
                label=f"{label}.registered_training_specs.selected_evaluation",
            ),
            "selected_fullfit_sha256": _sha256(
                selected_fullfit,
                "sha256",
                label=f"{label}.registered_training_specs.selected_fullfit",
            ),
            "registered_training_specs_sha256": _canonical_sha256(registered_specs),
            "saved_fullfit_spec_sha256": _sha256(
                saved_spec,
                "sha256",
                label=f"{label}.saved_fullfit_spec",
            ),
            "saved_fullfit_spec_bytes": _integer(
                saved_spec,
                "bytes",
                label=f"{label}.saved_fullfit_spec",
            ),
            "saved_fullfit_payload_sha256": saved_spec_payload_sha,
            "saved_fullfit_spec_record_sha256": _canonical_sha256(saved_spec),
        },
    }


def validate_candidate_active_parity(
    candidate_manifest_path: str | Path,
    active_manifest_path: str | Path,
    install_receipt_path: str | Path,
) -> dict[str, Any]:
    """Prove exact immutable parity for one candidate/active manifest pair.

    The standard install receipt must bind the exact candidate and active
    manifest bytes. Both manifests must then describe the same complete rich
    release, model, immutable data, and specification identities.
    """

    candidate_path = Path(candidate_manifest_path)
    active_path = Path(active_manifest_path)
    receipt_path = Path(install_receipt_path)
    candidate = _load_json_object(candidate_path, label="Candidate manifest")
    active = _load_json_object(active_path, label="Active manifest")
    receipt = _load_json_object(receipt_path, label="Install receipt")

    if receipt.get("schema_version") != 1 or receipt.get("release_kind") != "rich_v3":
        raise CandidateActiveParityError(
            "Install receipt must be schema_version=1 and release_kind='rich_v3'"
        )
    release_id = _string(receipt, "release_id", label="Install receipt")
    receipt_bundle_id = _string(receipt, "bundle_id", label="Install receipt")
    candidate_manifest_sha = _file_sha256(candidate_path)
    active_manifest_sha = _file_sha256(active_path)
    receipt_candidate_sha = _sha256(
        receipt,
        "source_manifest_sha256",
        label="Install receipt",
    )
    if receipt_candidate_sha != candidate_manifest_sha:
        raise CandidateActiveParityError(
            "Install receipt source_manifest_sha256 does not match candidate manifest bytes"
        )
    receipt_active_sha = _sha256(
        receipt,
        "installed_manifest_sha256",
        label="Install receipt",
    )
    if receipt_active_sha != active_manifest_sha:
        raise CandidateActiveParityError(
            "Install receipt installed_manifest_sha256 does not match active manifest bytes"
        )

    candidate_identity = _manifest_parity_identity(candidate, label="Candidate manifest")
    active_identity = _manifest_parity_identity(active, label="Active manifest")
    candidate_bundle_id = candidate_identity["release"]["bundle_id"]
    active_bundle_id = active_identity["release"]["bundle_id"]
    if receipt_bundle_id != candidate_bundle_id or receipt_bundle_id != active_bundle_id:
        raise CandidateActiveParityError(
            "Install receipt bundle_id does not match candidate and active manifests"
        )
    if candidate_identity != active_identity:
        mismatches = [
            section
            for section in ("release", "models", "data", "specs")
            if candidate_identity[section] != active_identity[section]
        ]
        raise CandidateActiveParityError(
            "Candidate and active immutable parity mismatch: " + ", ".join(mismatches)
        )

    return {
        "schema_version": 1,
        "passed": True,
        "release_id": release_id,
        "bundle_id": candidate_bundle_id,
        "candidate_manifest_sha256": candidate_manifest_sha,
        "active_manifest_sha256": active_manifest_sha,
        "immutable_parity_sha256": _canonical_sha256(candidate_identity),
    }
