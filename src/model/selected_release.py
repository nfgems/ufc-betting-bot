"""Portable file identities for an approved, already-trained selected release.

This schema carries the actual monthly-candidate lineage. It does not pretend
that the older production-refit builder created the selected candidate.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
REQUIRED_PROVENANCE = frozenset({
    "original_selection", "monthly_selection", "candidate_receipt", "monthly_run",
    "build_run", "independent_review", "independent_review_manifest",
})


def confined_file(root: Path, relative: object) -> Path:
    text = str(relative or "")
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts or "\\" in text or ":" in text:
        raise ValueError(f"Unsafe selected release path: {text!r}")
    resolved = (root / path).resolve(strict=True)
    resolved.relative_to(root.resolve(strict=True))
    if not resolved.is_file() or any(p.is_symlink() for p in [root / path, *(root / path).parents] if p != p.parent):
        raise ValueError(f"Selected release file is not regular: {text}")
    return resolved


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_selected_release(
    payload: dict[str, Any], root: Path, *,
    expected_predecessor_manifest_sha256: str = "3c61cba634d4bf64ccc62f7f66d5855cb8fcc0da3201f5fad66378bf8074aa9d",
    expected_selection_approval_sha256: str | None = None,
    expected_owner_dob_correction_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate v1 relocation or an externally approved, narrow v2 revision.

    Trust arguments come from the authorized release plan or a previously
    verified installation receipt, never fields copied from candidate bytes.
    """
    version = payload.get("selected_candidate_schema_version")
    if type(version) is int and version == 2:
        from src.model.selected_inference_revision import validate_inference_revision

        return validate_inference_revision(
            payload, root,
            expected_predecessor_manifest_sha256=expected_predecessor_manifest_sha256,
            expected_selection_approval_sha256=expected_selection_approval_sha256,
            expected_owner_dob_correction_sha256=expected_owner_dob_correction_sha256,
        )
    return _validate_selected_release_v1(payload, root)


def _validate_selected_release_v1(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    """Bind every declared file, the source selector and the release identity."""
    if type(payload.get("selected_candidate_schema_version")) is not int or payload["selected_candidate_schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported selected candidate release schema")
    if payload.get("manifest_version") != 3 or payload.get("staging_schema_version") != 1:
        raise ValueError("Selected candidate requires rich production manifest v3")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,180}", str(payload.get("bundle_id", ""))):
        raise ValueError("Selected candidate bundle ID is invalid")
    inventory = payload.get("release_file_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Selected release file inventory is missing")
    records = {}
    for record in inventory:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise ValueError("Invalid selected release inventory record")
        relative = record["path"]
        if relative in records or relative in {"staging_manifest.json", "manifest.json"}:
            raise ValueError("Duplicate or recursive selected release file inventory")
        path = confined_file(root, relative)
        if type(record["bytes"]) is not int or record["bytes"] != path.stat().st_size or record["sha256"] != file_hash(path):
            raise ValueError(f"Selected release file identity mismatch: {relative}")
        records[relative] = record
    provenance = payload.get("selected_provenance")
    if not isinstance(provenance, dict) or set(provenance) != REQUIRED_PROVENANCE:
        raise ValueError("Selected release must retain its original selection, monthly and independent-review evidence")
    if any(value not in records for value in provenance.values()):
        raise ValueError("Selected release provenance is not in the pinned file inventory")
    required = {
        "models/xgboost_model.pkl", "models/xgboost_no_odds_model.pkl",
        "models/logistic_model.pkl", "processed/features.csv", "processed/fights_cleaned.csv",
        str(payload.get("saved_fullfit_spec", {}).get("staged_path", "")),
        str(payload.get("selected_runtime_manifest_path", "")),
    }
    if not required.issubset(records):
        raise ValueError("Selected release is missing a required model, processed pair, spec or selector")
    runtime_relative = payload["selected_runtime_manifest_path"]
    runtime_record = records[runtime_relative]
    if runtime_record["sha256"] != payload.get("selected_runtime_manifest_sha256"):
        raise ValueError("Selected runtime manifest hash disagrees with release inventory")
    runtime_path = confined_file(root, runtime_relative)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if runtime.get("release_id") != payload.get("bundle_id"):
        raise ValueError("Selected runtime manifest belongs to a different release")
    selection = runtime.get("selection", {})
    selected_record = records.get(selection.get("path"))
    if selected_record != selection:
        raise ValueError("Selected source manifest does not match the exact release inventory")
    if runtime.get("schema_version") != 1 or runtime.get("approved_history_cutoff") != payload.get("snapshot_max_event_date"):
        raise ValueError("Selected runtime schema or history cutoff disagrees with production manifest")
    for role, hash_field, filename in (
        ("primary", "model_sha256", "xgboost_model.pkl"),
        ("no_odds", "no_odds_model_sha256", "xgboost_no_odds_model.pkl"),
        ("logistic", "logistic_model_sha256", "logistic_model.pkl"),
    ):
        record = records[f"models/{filename}"]
        if runtime.get("models", {}).get(role) != record or record["sha256"] != payload.get(hash_field):
            raise ValueError(f"Selected runtime {role} model disagrees with production identity")
    if set(runtime.get("models", {})) != {"primary", "no_odds", "logistic"}:
        raise ValueError("Selected runtime model roles are not exact")
    processed = runtime.get("processed", {})
    if processed.get("directory") != "processed" or set(processed.get("files", {})) != {"features.csv", "fights_cleaned.csv"}:
        raise ValueError("Selected processed directory or file set is not exact")
    for filename, stem in (("features.csv", "features"), ("fights_cleaned.csv", "fights")):
        record = records[f"processed/{filename}"]
        if (processed["files"][filename] != record or record["sha256"] != payload.get(f"processed_{stem}_sha256")
                or record["bytes"] != payload.get(f"processed_{stem}_bytes")):
            raise ValueError("Selected runtime processed pair disagrees with production identity")
    selected = json.loads(confined_file(root, selection["path"]).read_text(encoding="utf-8"))
    original = json.loads(confined_file(root, provenance["original_selection"]).read_text(encoding="utf-8"))
    if set(selected.get("files", {})) != set(original.get("files", {})) or not selected.get("files"):
        raise ValueError("Relocated selection changed its original selected input roles")
    for role, item in selected["files"].items():
        original_item = original["files"][role]
        if {k:v for k,v in item.items() if k != "path"} != {k:v for k,v in original_item.items() if k != "path"}:
            raise ValueError(f"Relocated input differs from approved source identity: {role}")
        if item.get("absent") is True:
            if (root / "explicitly-absent-inputs" / role).exists():
                raise ValueError(f"Absent selected input unexpectedly exists: {role}")
        elif item.get("directory") is True:
            directory = str(item.get("path", "")).rstrip("/")
            expected_files = item.get("files", {})
            declared = {name[len(directory)+1:]: value["sha256"] for name,value in records.items() if name.startswith(directory + "/")}
            actual_directory = (root / directory).resolve(strict=True)
            actual_directory.relative_to(root.resolve(strict=True))
            actual = {p.relative_to(actual_directory).as_posix(): file_hash(p) for p in actual_directory.rglob("*") if p.is_file()}
            if declared != expected_files or actual != expected_files:
                raise ValueError(f"Selected directory is not exactly covered by release inventory: {role}")
        else:
            record = records.get(item.get("path"))
            if record is None or record["sha256"] != item.get("sha256") or record["bytes"] != item.get("bytes"):
                raise ValueError(f"Selected input is not covered by release inventory: {role}")
    recovery = payload.get("previous_rollback_identity", {})
    if recovery.get("automatic_resume_permitted") is not False or recovery.get("recovery_mode") != "paused":
        raise ValueError("Selected release recovery must pause without automatically resuming the old system")
    return {
        "selected_runtime_manifest_path": str(runtime_path),
        "selected_runtime_manifest_sha256": runtime_record["sha256"],
        "selected_release_id": runtime["release_id"],
        "selected_candidate_schema_version": SCHEMA_VERSION,
        "selected_release_file_count": len(records),
    }


def validate_selected_monthly_spec(payload: dict[str, Any], root: Path, primary_spec: dict) -> None:
    """Use the completed monthly adapter's exact resolved recipe, not defaults.

    The named recipe has a default cutoff and an implicit RNG seed. Monthly05
    already resolved both in its independently reviewed receipt. No additional
    model parameter difference is permitted here.
    """
    from src.model.monthly_evaluator_adapters import validate_primary_spec
    from src.model.training_preflight import resolved_spec_identity
    from src.model.training_spec import NamedModelTrainingSpec, resolve_named_training_spec

    receipt = json.loads(confined_file(root, payload["selected_provenance"]["candidate_receipt"]).read_text(encoding="utf-8"))
    contract = receipt["contract"]
    policy = contract["monthly_policy"]
    name = primary_spec["name"]
    if policy.get("adapter") != "durability_monthly_refit_v1" or contract.get("name") != name:
        raise ValueError("Selected model does not match the completed monthly adapter")
    cutoff = datetime.fromisoformat(policy["input_completion_cutoff_utc"].replace("Z", "+00:00"))
    if cutoff.tzinfo is None or cutoff.utcoffset() != timedelta(0):
        raise ValueError("Monthly receipt completion cutoff must be explicit UTC")
    expected_end = (cutoff.date() + timedelta(days=1)).isoformat()
    if policy["train_cutoff_date"] != expected_end:
        raise ValueError("Monthly receipt resolved cutoff differs from its accepted input completion")
    validate_primary_spec(primary_spec, {
        "start_inclusive": policy["modeling_start_date"], "end_exclusive": expected_end,
    })
    expected_named = {"name": name, "sha256": resolved_spec_identity(resolve_named_training_spec(name))}
    expected_resolved = {"name": name, "sha256": resolved_spec_identity(NamedModelTrainingSpec.from_json(json.dumps(primary_spec)))}
    if policy["named_training_spec"] != expected_named or policy["training_spec"] != expected_resolved:
        raise ValueError("Selected recipe differs from the independently reviewed monthly identities")
