"""Narrow, externally authorized inference revision of one retained v1 release.

This module only reads package bytes. It never imports an application entrypoint,
loads models, obtains source data, or turns a JSON approval claim into authority.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from src.model.selected_release import REQUIRED_PROVENANCE, confined_file, file_hash

BASELINE_MANIFEST_SHA256 = "3c61cba634d4bf64ccc62f7f66d5855cb8fcc0da3201f5fad66378bf8074aa9d"
ROUTE_BASELINE_SHA256 = "607bd897a17bb21cc5743e20de95663c4992ca146f469ce2b1fbb3f6af27c7f3"
# Independently reviewed local additions, excluding only generation rebinding.
# These constants are deliberately narrow: editing any observation, owner text,
# original timestamp, evidence pin or policy cell needs a new reviewed contract.
APPENDED_ROWS_SHA256 = {
    "profiles": "6b4f7db16d2faee4ec95b4451840db056fabe94589fe839423da1edb9d39f324",
    "profile_decisions": "7071cf522467abeeb3d7ea23692828ebcd657f7baf2ff1c394a6f514dda7501e",
}
ROUTE_ADDITIONS_SHA256 = "4a81b40f66720937fb20c697619d033395bbfcce04a1a258e57eb29290f248c4"
ROUTE_CONTENT_SHA256 = "ec0a4caeb673a750669bbdaf510b4db0ffb63ee28c5e67bed376ca5928954368"
ROUTE_GENERATION_FIELDS = {"selection_sha256", "selected_sources_sha256", "manifest_sha256", "release_id"}
KIND = "profile_only_inference_revision_v1"
OWNER_DOB_KIND = "profile_only_inference_revision_lucia_owner_dob_v1"
# Separate exact anchors for the current owner's narrowly scoped DOB correction.
OWNER_DOB_DECISIONS_SHA256 = "a6ee64a3b22c6ca5fc2cc6a0d5e19264748c7a081321ea4ea0a5a525d79aab69"
OWNER_DOB_ROUTE_ADDITIONS_SHA256 = "58c0718ff8695406feaa48244e034fbcdd3b2c79dfb2a5567a61a3ad6d2341f2"
OWNER_DOB_ROUTE_CONTENT_SHA256 = "19bd818921d945f7ad876cc3ce8e1c394b3e84f3989b148022e5ddf976a1a919"
CHANGED_ROLES = frozenset({"profiles", "profile_decisions"})
SUBJECT_IDS = ("184f6ec4ee712e9f", "e9f9add1481fc11c")
SUBJECT_NAMES = dict(zip(SUBJECT_IDS, ("Lucia Szabova", "Javad Mahjoub")))
NAMESPACE = "provenance/inference-revision/"
COMPATIBILITY_CHECKS = frozenset({
    "real_model_inference", "missing_values", "strict_method_coverage",
    "all_saved_routes", "negative_identity_pin_approval_cases",
})
REVIEW_CHECKS = frozenset({
    "complete_release_inventory", "complete_source_inventory", "preserved_training_lineage",
    "all_saved_routes", "real_model_missing_values", "negative_identity_pin_approval_cases",
})
EXCLUDED_AUTHORITIES = ["hosted_execution", "production_activation", "recurrence", "source_acquisition", "trading"]
CORE_PATHS = {"model_path": "models/xgboost_model.pkl", "no_odds_model_path": "models/xgboost_no_odds_model.pkl",
              "logistic_model_path": "models/logistic_model.pkl", "processed_dir": "processed"}
# Owner packaging decision A (2026-09-22): carry the already approved two-profile
# selection and the Lucia DOB correction onto a separately reviewed retrained
# integer-1 predecessor. The original approval, its request and the correction are
# carried byte-for-byte and re-verified against their original anchors; they never
# approve the new data/model bytes, which the external predecessor anchor binds.
CARRIED_KIND = "carried_owner_inference_revision_v1"
CARRIED_APPLICATION_KIND = "carried_application_of_existing_owner_authority"
CARRIED_APPROVED_MANIFEST_SHA256 = "1271a974e41c6496a3c787ee65495af2404334b6411d17c98eedbda74660b048"
CARRIED_ARCHIVE = NAMESPACE + "carried-approved-revision/"
CARRIED_POLICY = "carried owner inference revision; reviewed retrained selection retained unchanged"
CARRIED_ROUTE_FIELDS = ROUTE_GENERATION_FIELDS | {"processed_fights_sha256"}
# Only the exact original directions below authorize local use of the reviewed
# retrained predecessor. Retained technical chain evidence has a separate role.
# Package records cannot choose these expectations or turn re-serialized old
# approvals, corrections, requests or this revision's evidence into authority.
CARRIED_NEW_BYTES_AUTHORITY = NAMESPACE + "new-bytes-authority/"
CARRIED_COMPATIBILITY_EVIDENCE = NAMESPACE + "compatibility-evidence/"
CARRIED_REVIEW_EVIDENCE = NAMESPACE + "technical-review-evidence/"
# Source-reviewed contract: current owner direction plus original packaging A;
# old approval and correction remain bound solely to their original graph.
CARRIED_NEW_BYTES_CONTRACT = {'predecessor_manifest_sha256': '7a13048b5aeb1179a1ea0b29a7eba100e60c94e8d159ea131d72775b541f27ea',
 'authority_records': [{'bytes': 2106,
                        'path': 'provenance/owner-direction.json',
                        'sha256': 'd22c05f2a672e0e4238ad161e26a93e8aff2e341bd21ecb066989cbb447984f1'},
                       {'bytes': 2870,
                        'sha256': '3a62dacec708e37c319f015689d88a4bf632477f7b545cbde311d450dca04125',
                        'path': 'provenance/inference-revision/new-bytes-authority/owner-decision-packaging-A-01.json'}],
 'chain_evidence': [{'bytes': 9813,
                     'path': 'provenance/closed-evidence/C/Temp/s14-review01/final01/actual_results.json',
                     'sha256': '316297984f6b97291d12b928b4f81191916d06344e56ae418ab6d9d1f3b027a3'},
                    {'bytes': 1251,
                     'path': 'provenance/closed-evidence/C/Temp/s14-review01/final01/journal.jsonl',
                     'sha256': 'c773eed063afb26ac099bf551886c68826db43f427d1d4694d574468c209b30b'},
                    {'bytes': 7988,
                     'path': 'provenance/closed-evidence/C/Temp/s14-review01/final01/preservation_scope.json',
                     'sha256': '7caed563263d0a9d38ab4d570011ce331dd3cad73edad43d45ecc184b3987010'},
                    {'bytes': 1646272,
                     'path': 'provenance/completed-chain-index.json',
                     'sha256': 'ee542e9c2c6a2f44af79ae89b17814da51a2954b73a81f60efcdd3dc3970a05c'},
                    {'bytes': 213107,
                     'path': 'provenance/full-chain-readback.json',
                     'sha256': '1a45740f30f9f98e63f35b1418340f088d41d7b1557c7cfeebc7b146a1e20364'},
                    {'bytes': 1296,
                     'path': 'provenance/independent_review.json',
                     'sha256': 'd93ecb794a31df2fceadb8df62bae05878579524fd6457f8993db5cace9b41eb'},
                    {'bytes': 6297,
                     'path': 'provenance/independent_review_manifest.json',
                     'sha256': '3fdd3e28c2077d67145ee424e02b0f7e4d659a809144c892ed2fd24c5f254a4d'},
                    {'bytes': 3819,
                     'path': 'provenance/local-chain-checkpoint.json',
                     'sha256': 'faac405966be56436ab69ad0fabd812e82a614603629089ec913e843d719206a'}]}
CARRIED_KEYS = frozenset({
    "schema_version", "kind", "predecessor_release_id", "predecessor_manifest", "predecessor_selection",
    "predecessor_runtime", "carried_manifest", "derivation", "technical_review", "selection_request",
    "selection_approval", "base_selection_request", "owner_dob_correction",
})
OWNER_DOB_KEYS = frozenset({
    "schema_version", "kind", "predecessor_release_id", "predecessor_manifest", "predecessor_selection",
    "predecessor_runtime", "derivation", "technical_review", "selection_request", "selection_approval",
    "base_selection_request", "owner_dob_correction",
})


def _same(actual, expected, message):
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError(message)


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _without(value, keys):
    return {key: item for key, item in value.items() if key not in keys}


def _hash(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid external {label} trust anchor")


def _relative(value):
    if not isinstance(value, str) or not value or Path(value).as_posix() != value:
        raise ValueError("Revision paths must be canonical relative strings")
    if Path(value).is_absolute() or ".." in Path(value).parts or "\\" in value or ":" in value:
        raise ValueError("Unsafe inference revision path")
    return value


def _record(record):
    if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
        raise ValueError("Invalid inference revision file record")
    _relative(record["path"])
    if type(record["bytes"]) is not int or record["bytes"] < 0:
        raise ValueError("Invalid inference revision byte count")
    if not isinstance(record["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
        raise ValueError("Invalid inference revision file hash")
    return record


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate revision JSON key: {key}")
        result[key] = value
    return result


def _read(root, relative):
    result = json.loads(confined_file(root, _relative(relative)).read_text(encoding="utf-8"),
                        object_pairs_hook=_unique_object)
    if not isinstance(result, dict):
        raise ValueError("Revision evidence must be a JSON object")
    return result


def _pin(pin, records, label):
    _record(pin)
    _same(pin, records.get(pin["path"]), f"{label} is not pinned by the complete release inventory")
    return pin


def _evidence(value, records, label):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} requires actual completed evidence")
    seen = set()
    for pin in value:
        _pin(pin, records, label)
        if pin["bytes"] <= 0 or pin["path"] in seen:
            raise ValueError(f"{label} evidence is empty or duplicated")
        seen.add(pin["path"])


def _unrebase(payload, reference, root):
    """Admit only the installer's exact four-path plus root projection."""
    if all(payload.get(key) == reference.get(key) for key in (*CORE_PATHS, "rich_release_root")):
        return payload
    for key, relative in CORE_PATHS.items():
        if payload.get(key) != str((root / relative).resolve(strict=True)):
            raise ValueError("Inference revision core paths are not the exact installer projection")
    if payload.get("rich_release_root") != str(root.resolve(strict=True)):
        raise ValueError("Inference revision rich release root is not exact")
    restored = dict(payload)
    for key in (*CORE_PATHS, "rich_release_root"):
        if key in reference:
            restored[key] = reference[key]
        else:
            restored.pop(key, None)
    return restored


def _installed_envelope(payload, root, actual, records):
    if {"staging_manifest.json", "manifest.json"}.issubset(actual):
        raise ValueError("Inference revision has ambiguous outer manifests")
    extras = actual - set(records) - {"staging_manifest.json", "manifest.json"}
    if not extras:
        return
    if extras != {"install_receipt.json", "provenance/source_staging_manifest.json"} or "staging_manifest.json" in actual:
        raise ValueError("Inference revision inventory does not exactly cover the release tree")
    source = _read(root, "provenance/source_staging_manifest.json")
    installed = _read(root, "manifest.json")
    _same(payload, installed, "Validated manifest differs from installed manifest bytes")
    _same(_unrebase(payload, source, root), source, "Installed inference manifest is not an exact source projection")
    source_hash = file_hash(confined_file(root, "provenance/source_staging_manifest.json"))
    release_id = "r-" + source_hash[:20]
    if root.name != release_id:
        raise ValueError("Installed inference directory does not derive from source staging identity")
    rows = [{"path": relative, "bytes": confined_file(root, relative).stat().st_size,
             "sha256": file_hash(confined_file(root, relative))} for relative in sorted(actual - {"install_receipt.json"})]
    aggregate = hashlib.sha256((json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()).hexdigest()
    receipt = _read(root, "install_receipt.json")
    _same(receipt, {
        "schema_version": 1, "release_kind": "rich_v3", "release_id": release_id,
        "bundle_id": payload["bundle_id"], "source_manifest_sha256": source_hash,
        "installed_manifest_sha256": file_hash(confined_file(root, "manifest.json")),
        "source_manifest_copy": "provenance/source_staging_manifest.json",
        "core_path_rewrites": sorted((*CORE_PATHS, "rich_release_root")),
        "tree_inventory": {"file_count": len(rows), "total_bytes": sum(row["bytes"] for row in rows),
                           "aggregate_sha256": aggregate, "files": rows},
    }, "Installed inference receipt does not bind its exact source, projection and complete tree")


def _csv(root, pin):
    with confined_file(root, pin["path"]).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames
        if not headers or len(headers) != len(set(headers)):
            raise ValueError("Revision CSV header is missing or duplicated")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Revision CSV rows do not match their header")
    return headers, rows


def _append_semantics(root, before, after, role, *, owner_dob_correction=False):
    old_headers, old_rows = _csv(root, before)
    headers, rows = _csv(root, after)
    if headers[:len(old_headers)] != old_headers or (role == "profiles" and headers != old_headers):
        raise ValueError(f"Revision changed prior {role} CSV columns")
    if len(rows) != len(old_rows) + 2:
        raise ValueError(f"Revision must append exactly two {role} rows")
    for old, new in zip(old_rows, rows):
        _same({key: new[key] for key in old_headers}, old, f"Revision changed prior {role} row values")
        if any(new[key] != "" for key in headers[len(old_headers):]):
            raise ValueError("Revision populated new columns on a prior decision")
    added = rows[len(old_rows):]
    identity_key = "fighter_url" if role == "profiles" else "stable_ufcstats_id"
    if identity_key not in headers:
        raise ValueError(f"Revision {role} lacks its stable ID column")

    def identity(row):
        if role == "profiles":
            match = re.fullmatch(r"http://ufcstats.com/fighter-details/([0-9a-f]{16})", row[identity_key])
            return match.group(1) if match else ""
        return row[identity_key]

    if sorted(identity(row) for row in added) != list(SUBJECT_IDS):
        raise ValueError("Revision appended the wrong or duplicate subject IDs")
    if any(identity(row) in SUBJECT_IDS for row in old_rows):
        raise ValueError("Revision subject already existed in the predecessor")
    additions = {identity(row): row for row in added}
    expected_rows = (OWNER_DOB_DECISIONS_SHA256 if owner_dob_correction and role == "profile_decisions"
                     else APPENDED_ROWS_SHA256[role])
    if _canonical_hash([additions[subject] for subject in SUBJECT_IDS]) != expected_rows:
        raise ValueError("Revision appended rows differ from exact independently reviewed cell identities")
    for subject, row in additions.items():
        name_key = "name" if role == "profiles" else "canonical_name"
        if row.get(name_key) != SUBJECT_NAMES[subject]:
            raise ValueError("Revision subject name disagrees with its stable ID")
    lucia, javad = (additions[subject] for subject in SUBJECT_IDS)
    if role == "profiles":
        if lucia.get("dob") != "Dec 23, 1997" or javad.get("dob") != "May 26, 1992":
            raise ValueError("Revision changed retained subject DOB observations")
        if any(value for row in added for key, value in row.items() if key not in {"name", "fighter_url", "dob"}):
            raise ValueError("Revision admitted unreviewed profile measurements")
    else:
        if owner_dob_correction:
            if (lucia.get("field") != "dob" or lucia.get("decision_status") != "owner_adjudicated"
                    or lucia.get("model_safe_value") != "1997-12-23"):
                raise ValueError("Lucia DOB correction must select the exact owner-adjudicated date")
        elif (lucia.get("field") != "dob" or lucia.get("decision_status") != "unresolved"
                or lucia.get("model_safe_value") != ""):
            raise ValueError("Lucia DOB must remain unresolved and model unknown")
        if (javad.get("field") != "dob" or javad.get("decision_status") != "owner_adjudicated"
                or javad.get("model_safe_value") != "1992-05-26"
                or "1991-05-26" not in javad.get("retained_disagreement_note", "")
                or javad.get("owner_adjudication_text") != "# Javad Mahjoub was born May 26, 1992"):
            raise ValueError("Javad DOB must retain the owner decision and official disagreement")
        if any(row.get("automatic_replacement_allowed") != "False" for row in added):
            raise ValueError("Inference revision must preserve automatic publisher restrictions")


def _routes(root, derivation, records, runtime, selected, *, owner_dob_correction=False):
    prior_pin = _pin(derivation.get("predecessor_route_artifact"), records, "Predecessor routes")
    if prior_pin["sha256"] != ROUTE_BASELINE_SHA256:
        raise ValueError("Predecessor routes differ from the independent route trust anchor")
    route_pin = _pin(derivation.get("route_artifact"), records, "Successor routes")
    prior, route = _read(root, prior_pin["path"]), _read(root, route_pin["path"])
    old_bindings, bindings = prior.get("bindings"), route.get("bindings")
    if not isinstance(old_bindings, list) or len(old_bindings) != 96 or not isinstance(bindings, list) or len(bindings) != 98:
        raise ValueError("Inference revision must preserve 96 routes and append exactly two")
    _same(bindings[:96], old_bindings, "Inference revision changed an earlier route binding")
    if not isinstance(prior.get("aliases"), list) or len(prior["aliases"]) != 2:
        raise ValueError("Predecessor route aliases must contain exactly two records")
    _same(route.get("aliases"), prior["aliases"], "Inference revision changed an earlier alias")
    added = bindings[96:]
    expected_additions = OWNER_DOB_ROUTE_ADDITIONS_SHA256 if owner_dob_correction else ROUTE_ADDITIONS_SHA256
    if _canonical_hash(added) != expected_additions:
        raise ValueError("Inference revision changed exact reviewed route additions")
    if sorted(item.get("fighter_id", "") for item in added) != list(SUBJECT_IDS):
        raise ValueError("Inference revision route additions have the wrong subject IDs")
    if len({item.get("athlete_url") for item in bindings}) != len(bindings):
        raise ValueError("Inference revision introduced duplicate route ownership")
    for item in added:
        subject = item["fighter_id"]
        _same(item.get("selected_owner"), {"name": SUBJECT_NAMES[subject], "athlete_url": "", "fighter_id": subject},
              "Inference revision changed a literal blank selected-owner route anchor")
        expected_slug = "lucia-szabova" if subject == SUBJECT_IDS[0] else "javad-mahjoub"
        if item.get("name") != SUBJECT_NAMES[subject] or item.get("athlete_url") != "https://www.ufc.com/athlete/" + expected_slug:
            raise ValueError("Inference revision route identity mismatch")
    _same(route.get("selection_sha256"), runtime["selection"]["sha256"], "Route/source identity mismatch")
    _same(route.get("manifest_sha256"), derivation["successor_runtime"]["sha256"], "Route/runtime identity mismatch")
    _same(route.get("release_id"), runtime["release_id"], "Route/release identity mismatch")
    _same(route.get("processed_fights_sha256"), runtime["processed"]["files"]["fights_cleaned.csv"]["sha256"],
          "Route/processed identity mismatch")
    _same(route.get("selected_sources_sha256"),
          {role: selected["files"][role]["sha256"] for role in ("profiles", "active_roster", "identity_aliases")},
          "Route selected input identities changed")
    source_pin = _pin(derivation.get("source_candidate_manifest"), records, "Source candidate manifest")
    source = _read(root, source_pin["path"])
    source_records = {}
    for record in source.get("files", []):
        _record(record)
        if record["path"] in source_records:
            raise ValueError("Source candidate inventory contains duplicate paths")
        source_records[record["path"]] = record
    source_route = source_records.get("config/runtime_fighter_bindings_v1.json", {})
    _same(_without(source_route, {"path"}), _without(route_pin, {"path"}),
          "Source candidate does not contain the exact successor route artifact")
    expected_content = OWNER_DOB_ROUTE_CONTENT_SHA256 if owner_dob_correction else ROUTE_CONTENT_SHA256
    if _canonical_hash(_without(route, ROUTE_GENERATION_FIELDS)) != expected_content:
        raise ValueError("Inference revision changed reviewed route metadata or retained evidence")
    for item in route.get("evidence_files", []):
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("Invalid route evidence inventory record")
        path = "config/" + _relative(item["path"])
        if source_records.get(path, {}).get("sha256") != item["sha256"]:
            raise ValueError("Source candidate inventory omitted or changed route evidence")


def _original_timestamp(record, label):
    timestamp = record.get("original_timestamp")
    if not isinstance(timestamp, str):
        raise ValueError(f"{label} must preserve the actual original timestamp")
    match = re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.(\d{1,9}))?(?:Z|[+-]\d{2}:\d{2})", timestamp)
    if not match or type(record.get("timestamp_precision")) is not int or record["timestamp_precision"] != len(match.group(1) or ""):
        raise ValueError(f"{label} must preserve original timestamp offset and precision")
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def _owner_correction(root, meta, expected_hash):
    _hash(expected_hash, "owner DOB correction")
    if meta["owner_dob_correction"]["sha256"] != expected_hash:
        raise ValueError("Owner DOB correction differs from its independent external trust anchor")
    correction = _read(root, meta["owner_dob_correction"]["path"])
    for key in ("correction_id", "verbatim_decision", "attribution"):
        if not isinstance(correction.get(key), str) or not correction[key].strip():
            raise ValueError("Owner DOB correction must retain the actual attributed owner instruction")
    _original_timestamp(correction, "Owner DOB correction")
    _same(correction, {
        "schema_version": 1, "kind": "lucia_owner_dob_correction_v1",
        "correction_id": correction["correction_id"],
        "base_selection_request_sha256": meta["base_selection_request"]["sha256"],
        "base_selection_approval_sha256": meta["selection_approval"]["sha256"],
        "subject_id": SUBJECT_IDS[0], "canonical_name": SUBJECT_NAMES[SUBJECT_IDS[0]],
        "field": "dob", "previous_model_safe_value": None, "selected_model_safe_value": "1997-12-23",
        "scope": "inference_selection_only", "automatic_profile_policy_changed": False,
        "excluded_authorities": EXCLUDED_AUTHORITIES,
        **{key: correction[key] for key in ("verbatim_decision", "original_timestamp", "timestamp_precision", "attribution")},
    }, "Owner DOB correction changed its exact identity, original authority or permitted scope")
    return correction


def _correction_base(root, meta, derivation, records, selected, runtime, correction):
    """Bind old approval to its old request; amend exactly one approved row."""
    base_request = _read(root, meta["base_selection_request"]["path"])
    for key in ("request_id", "description"):
        if not isinstance(base_request.get(key), str) or not base_request[key].strip():
            raise ValueError("Owner correction requires the exact original selection request")
    _same(base_request, {
        "schema_version": 1, "request_id": base_request["request_id"], "description": base_request["description"],
        "scope": "inference_selection_only",
        **{key: base_request.get(key) for key in ("derivation", "technical_review", "source_candidate_manifest",
                                               "route_artifact", "successor_selection", "successor_runtime")},
        "subject_ids": list(SUBJECT_IDS), "retained_models": runtime["models"], "retained_processed": runtime["processed"],
        "strict_method_coverage": True, "excluded_authorities": EXCLUDED_AUTHORITIES,
    }, "Owner correction widened or replaced the original approved request")
    for key in ("derivation", "technical_review", "source_candidate_manifest", "route_artifact", "successor_selection", "successor_runtime"):
        _record(base_request[key])
    archive = Path(meta["base_selection_request"]["path"]).parent
    for key, filename in (
        ("derivation", "derivation.json"), ("technical_review", "technical-review.json"),
        ("source_candidate_manifest", "source-candidate-manifest.json"), ("route_artifact", "route-artifact.json"),
        ("successor_selection", "selected-source-manifest.json"), ("successor_runtime", "selected-runtime-manifest.json"),
    ):
        relative = (archive / filename).as_posix()
        archived = _pin(records.get(relative), records, "Archived original approved " + key)
        _same(_without(archived, {"path"}), _without(base_request[key], {"path"}),
              "Archived original approved graph evidence changed")
    old_derivation = _read(root, (archive / "derivation.json").as_posix())
    old_compatibility = _pin(records.get((archive / "compatibility-result.json").as_posix()), records,
                             "Archived original compatibility result")
    _same(_without(old_compatibility, {"path"}), _without(old_derivation["compatibility_result"], {"path"}),
          "Archived original compatibility result changed")
    base_selection_pin = _pin(derivation.get("base_source_selection"), records, "Original approved source selection")
    _same(_without(base_selection_pin, {"path"}), _without(base_request["successor_selection"], {"path"}),
          "Archived original approved source selection changed")
    base_selection = _read(root, base_selection_pin["path"])
    _same(_without(selected, {"files", "inference_revision"}), _without(base_selection, {"files", "inference_revision"}),
          "Owner DOB correction changed original approved source metadata")
    _same(_without(selected["inference_revision"], {"owner_dob_correction_sha256"}), base_selection.get("inference_revision"),
          "Owner DOB correction changed original inference revision metadata")
    if set(base_selection.get("files", {})) != set(selected["files"]):
        raise ValueError("Owner DOB correction changed original approved input roles")
    for role, item in selected["files"].items():
        if role != "profile_decisions":
            _same(item, base_selection["files"][role], "Owner DOB correction changed an unrelated approved input")
    base_decisions_pin = _pin(derivation.get("base_profile_decisions"), records, "Original approved profile decisions")
    _same(_without(base_decisions_pin, {"path"}), _without(base_selection["files"]["profile_decisions"], {"path"}),
          "Archived original unresolved DOB decision changed")
    _append_semantics(root, derivation["predecessor_inputs"]["profile_decisions"], base_decisions_pin, "profile_decisions")
    headers, prior_rows = _csv(root, base_decisions_pin)
    new_headers, rows = _csv(root, selected["files"]["profile_decisions"])
    if headers != new_headers or len(prior_rows) != len(rows):
        raise ValueError("Owner DOB correction may only update an existing approved decision row")
    changed = [(before, after) for before, after in zip(prior_rows, rows) if before != after]
    if len(changed) != 1 or any(row.get("stable_ufcstats_id") != SUBJECT_IDS[0] or row.get("field") != "dob" for row in changed[0]):
        raise ValueError("Owner DOB correction must change only Lucia's existing DOB decision")
    row = changed[0][1]
    if (row.get("owner_adjudication_text") != correction["verbatim_decision"]
            or row.get("owner_adjudicated_at_utc") != correction["original_timestamp"]
            or row.get("owner_adjudication_date") != correction["original_timestamp"]
            or row.get("owner_authority_sha256") != meta["owner_dob_correction"]["sha256"]):
        raise ValueError("Corrected profile decision must retain the exact owner text and original timestamp")
    return base_request


def _approval(root, meta, request):
    """The original owner approval, unchanged, still approving only its original request."""
    approval = _read(root, meta["selection_approval"]["path"])
    for key in ("approval_id", "verbatim_decision", "original_timestamp", "attribution"):
        if not isinstance(approval.get(key), str) or not approval[key].strip():
            raise ValueError("Selection approval must retain the actual attributed owner decision")
    timestamp = approval["original_timestamp"]
    match = re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.(\d{1,9}))?(?:Z|[+-]\d{2}:\d{2})", timestamp)
    if not match or type(approval.get("timestamp_precision")) is not int or approval["timestamp_precision"] != len(match.group(1) or ""):
        raise ValueError("Selection approval must preserve original timestamp offset and precision")
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    _same(approval, {
        "schema_version": 1, "approval_id": approval["approval_id"],
        "request_sha256": meta["base_selection_request"]["sha256"],
        "decision": "approved", "scope": "inference_selection_only",
        **{key: approval[key] for key in ("verbatim_decision", "original_timestamp", "timestamp_precision", "attribution")},
    }, "Selection approval is stale or does not approve this exact request")
    for key in ("request_id", "description"):
        if not isinstance(request.get(key), str) or not request[key].strip():
            raise ValueError("Selection request must identify and explain the exact proposed decision")
    return approval


def _carried_authority(root, meta, records, expected_selection_approval_sha256, expected_owner_dob_correction_sha256):
    """Re-verify the carried owner-approved revision exactly as it was originally validated.

    Only the authority sub-graph is carried: the approved staging manifest (a fixed
    identity), its selection/runtime manifests, the two profile-role inputs and every
    inference-revision evidence file, each byte-identical to that manifest's inventory.
    """
    carried_pin = meta["carried_manifest"]
    if (carried_pin["path"] != CARRIED_ARCHIVE + "staging_manifest.json"
            or carried_pin["sha256"] != CARRIED_APPROVED_MANIFEST_SHA256):
        raise ValueError("Carried authority is not the exact owner-approved release")
    archive_root = root / CARRIED_ARCHIVE.rstrip("/")
    if archive_root.is_symlink() or not archive_root.is_dir():
        raise ValueError("Carried authority archive is missing")
    carried = _read(root, carried_pin["path"])
    inventory = carried.get("release_file_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Carried authority inventory is missing")
    carried_records = {}
    for record in inventory:
        _record(record)
        if record["path"] in carried_records or record["path"] in {"staging_manifest.json", "manifest.json"}:
            raise ValueError("Carried authority inventory contains duplicate records")
        carried_records[record["path"]] = record
    archived = {path[len(CARRIED_ARCHIVE):]: record for path, record in records.items() if path.startswith(CARRIED_ARCHIVE)}
    for relative, record in archived.items():
        expected = carried_pin if relative == "staging_manifest.json" else carried_records.get(relative)
        if expected is None or _without(record, {"path"}) != _without(expected, {"path"}):
            raise ValueError("Carried authority archive differs from the approved release inventory")
    c_meta = carried.get("selected_inference_revision")
    if not isinstance(c_meta, dict) or set(c_meta) != OWNER_DOB_KEYS or c_meta.get("kind") != OWNER_DOB_KIND:
        raise ValueError("Carried authority must be the exact owner-corrected inference revision")
    c_runtime_relative = carried.get("selected_runtime_manifest_path")
    c_runtime_record = carried_records.get(c_runtime_relative)
    if c_runtime_record is None or c_runtime_record["sha256"] != carried.get("selected_runtime_manifest_sha256"):
        raise ValueError("Carried authority runtime identity mismatch")
    if c_runtime_relative not in archived:
        raise ValueError("Carried authority archive omits the approved runtime manifest")
    c_runtime = _read(archive_root, c_runtime_relative)
    c_selection = c_runtime.get("selection")
    _record(c_selection)
    _same(c_selection, carried_records.get(c_selection["path"]), "Carried authority selection identity mismatch")
    if c_selection["path"] not in archived:
        raise ValueError("Carried authority archive omits the approved source selection")
    c_selected = _read(archive_root, c_selection["path"])
    c_files = c_selected.get("files")
    if not isinstance(c_files, dict) or not CHANGED_ROLES.issubset(c_files):
        raise ValueError("Carried authority selection lacks the approved profile roles")
    required = {"staging_manifest.json", c_runtime_relative, c_selection["path"],
                *(c_files[role]["path"] for role in CHANGED_ROLES),
                *(path for path in carried_records if path.startswith(NAMESPACE))}
    if set(archived) != required:
        raise ValueError("Carried authority archive is not the exact approved authority sub-graph")
    archive_records = {relative: carried_records[relative] for relative in required - {"staging_manifest.json"}}
    for key in OWNER_DOB_KEYS - {"schema_version", "kind", "predecessor_release_id"}:
        _pin(c_meta[key], archive_records, "Carried " + key)
    for key in ("selection_approval", "owner_dob_correction", "base_selection_request"):
        if (meta[key]["path"] != CARRIED_ARCHIVE + c_meta[key]["path"]
                or _without(meta[key], {"path"}) != _without(c_meta[key], {"path"})):
            raise ValueError("Carried owner authority must be the archived original record")
    if c_meta["selection_approval"]["sha256"] != expected_selection_approval_sha256:
        raise ValueError("Selection approval differs from the external trust anchor")
    # The original validator's owner-authority checks, on the archived original graph. The
    # archived derivation, compatibility and review shapes are fixed transitively: every archived
    # byte must equal an inventory entry of the constant-pinned approved manifest.
    c_correction = _owner_correction(archive_root, c_meta, expected_owner_dob_correction_sha256)
    c_derivation = _read(archive_root, c_meta["derivation"]["path"])
    c_base_request = _correction_base(archive_root, c_meta, c_derivation, archive_records, c_selected, c_runtime, c_correction)
    _routes(archive_root, c_derivation, archive_records, c_runtime, c_selected, owner_dob_correction=True)
    c_request = _read(archive_root, c_meta["selection_request"]["path"])
    _same(c_request, {
        "schema_version": 1, "request_id": c_request.get("request_id"), "description": c_request.get("description"),
        "scope": "inference_selection_only", "derivation": c_meta["derivation"], "technical_review": c_meta["technical_review"],
        **{key: c_derivation.get(key) for key in ("source_candidate_manifest", "route_artifact")},
        "successor_selection": c_selection, "successor_runtime": c_runtime_record,
        "subject_ids": list(SUBJECT_IDS), "retained_models": c_runtime["models"], "retained_processed": c_runtime["processed"],
        "strict_method_coverage": True, "excluded_authorities": EXCLUDED_AUTHORITIES,
        "kind": "application_of_existing_owner_authority", "base_selection_request": c_meta["base_selection_request"],
        "owner_dob_correction": c_meta["owner_dob_correction"],
    }, "Carried application record changed")
    c_approval = _approval(archive_root, c_meta, c_request)
    identities = [c_approval["approval_id"], c_base_request["request_id"], c_correction["correction_id"], c_request["request_id"]]
    if len(set(identities)) != len(identities):
        raise ValueError("Carried owner authority identities are not distinct")
    # All three historical identities come from the constant-pinned archive,
    # including the base two-profile release approved by the original request.
    base_archive = Path(c_meta["base_selection_request"]["path"]).parent
    base_runtime = _read(archive_root, (base_archive / "selected-runtime-manifest.json").as_posix())
    original_predecessor = _read(archive_root, c_meta["predecessor_manifest"]["path"])
    reserved_release_ids = {carried["bundle_id"], c_meta["predecessor_release_id"],
                            original_predecessor["bundle_id"], base_runtime["release_id"]}
    return {"selected": c_selected, "selection": c_selection, "derivation": c_derivation, "base_request": c_base_request,
            "correction": c_correction, "request": c_request, "approval": c_approval,
            "bundle_id": carried.get("bundle_id"), "records": carried_records,
            "reserved_release_ids": reserved_release_ids}


def _carried_fresh_evidence(value, records, forbidden_hashes, prefix, label):
    """Refuse stale graph bytes as new proof; completed semantics need actual review."""
    _evidence(value, records, label)
    for item in value:
        if not item["path"].startswith(prefix) or item["sha256"] in forbidden_hashes:
            raise ValueError(label + " must cite fresh evidence for this carried revision")


def _validate_carried_revision(payload, root, records, meta, expected_predecessor_manifest_sha256,
                               expected_selection_approval_sha256, expected_owner_dob_correction_sha256):
    """Reviewed retrained integer-1 predecessor plus exactly the carried owner inference authority."""
    if set(meta) != CARRIED_KEYS:
        raise ValueError("Carried inference revision metadata has missing or unexpected fields")
    _same(meta["schema_version"], 1, "Invalid inference revision metadata schema")
    for key in CARRIED_KEYS - {"schema_version", "kind", "predecessor_release_id"}:
        _pin(meta[key], records, key)
    # The retrained predecessor is trusted only through an explicit reviewed anchor; the
    # code's default predecessor is the older release already covered by the Lucia kind.
    if expected_predecessor_manifest_sha256 in {BASELINE_MANIFEST_SHA256, CARRIED_APPROVED_MANIFEST_SHA256}:
        raise ValueError("Carried inference revision requires the reviewed retrained predecessor anchor")
    if meta["predecessor_manifest"]["sha256"] != expected_predecessor_manifest_sha256:
        raise ValueError("Predecessor manifest differs from the external trust anchor")
    if meta["selection_approval"]["sha256"] != expected_selection_approval_sha256:
        raise ValueError("Selection approval differs from the external trust anchor")
    authority = _carried_authority(root, meta, records, expected_selection_approval_sha256,
                                   expected_owner_dob_correction_sha256)
    correction = _owner_correction(root, meta, expected_owner_dob_correction_sha256)
    previous = _read(root, meta["predecessor_manifest"]["path"])
    _same(previous.get("selected_candidate_schema_version"), 1, "Inference revision must derive directly from integer 1")
    if (payload["bundle_id"] == previous.get("bundle_id") or meta["predecessor_release_id"] != previous.get("bundle_id")
            or payload["bundle_id"] in authority["reserved_release_ids"]):
        raise ValueError("Inference revision predecessor/successor release identity mismatch")
    if not isinstance(previous.get("release_file_inventory"), list):
        raise ValueError("Predecessor inventory is missing")
    changes = {"bundle_id", "selected_candidate_schema_version", "selected_runtime_manifest_sha256",
               "release_file_inventory", "selected_inference_revision"}
    _same(_without(_unrebase(payload, previous, root), changes), _without(previous, changes),
          "Inference revision changed retained training or production metadata")
    prior_records = {}
    for record in previous["release_file_inventory"]:
        _record(record)
        if record["path"] in prior_records:
            raise ValueError("Predecessor inventory contains duplicate records")
        prior_records[record["path"]] = record
    runtime_relative = payload.get("selected_runtime_manifest_path")
    if runtime_relative not in records or runtime_relative not in prior_records:
        raise ValueError("Inference runtime manifest is not pinned by both release inventories")
    _same(_without(meta["predecessor_runtime"], {"path"}), _without(prior_records[runtime_relative], {"path"}),
          "Archived predecessor runtime changed")
    prior_runtime = _read(root, meta["predecessor_runtime"]["path"])
    _same(_without(meta["predecessor_selection"], {"path"}), _without(prior_runtime.get("selection"), {"path"}),
          "Archived predecessor selection changed")
    prior_selected = _read(root, meta["predecessor_selection"]["path"])
    runtime_record = records[runtime_relative]
    _same(runtime_record["sha256"], payload.get("selected_runtime_manifest_sha256"), "Inference runtime inventory identity mismatch")
    runtime = _read(root, runtime_relative)
    _same(_without(runtime, {"release_id", "selection"}), _without(prior_runtime, {"release_id", "selection"}),
          "Inference revision changed runtime models, processed pair, history or observations")
    _same(runtime.get("schema_version"), 1, "Runtime schema must remain integer 1")
    _same(runtime.get("release_id"), payload["bundle_id"], "Inference runtime belongs to another release")
    _same(runtime.get("approved_history_cutoff"), payload.get("snapshot_max_event_date"), "Inference history cutoff changed")
    selection = _pin(runtime.get("selection"), records, "Inference source selection")
    _same(selection["path"], prior_runtime["selection"]["path"], "Inference selector path changed")
    selected = _read(root, selection["path"])
    if (not isinstance(selected.get("files"), dict) or not isinstance(prior_selected.get("files"), dict)
            or not isinstance(prior_selected.get("release_relocation"), dict)):
        raise ValueError("Inference selector lacks its selected inputs or relocation record")
    carried_selected = authority["selected"]
    carried_selection_sha256 = authority["selection"]["sha256"]
    _same(selected.get("status"), "inference_selection_revision", "Inference selector status is incorrect")
    _same(selected.get("inference_revision"), {
        "predecessor_selection_sha256": prior_runtime["selection"]["sha256"],
        "changed_roles": sorted(CHANGED_ROLES), "subject_ids": list(SUBJECT_IDS),
        "predecessor_release_relocation": prior_selected.get("release_relocation"),
        "owner_dob_correction_sha256": meta["owner_dob_correction"]["sha256"],
        "carried_selection_sha256": carried_selection_sha256,
    }, "Inference selector revision metadata is incorrect")
    _same(selected.get("release_relocation"), {
        "original_selection_sha256": prior_selected["release_relocation"]["original_selection_sha256"],
        "policy": CARRIED_POLICY,
    }, "Inference selector misstates original training selection")
    source_changes = {"status", "files", "release_relocation", "inference_revision"}
    _same(_without(selected, source_changes), _without(prior_selected, source_changes),
          "Inference selector changed historical metadata")
    files, prior_files = selected["files"], prior_selected["files"]
    if set(files) != set(prior_files) or {role for role in files if files[role] != prior_files[role]} != CHANGED_ROLES:
        raise ValueError("Inference revision may change exactly profiles and profile_decisions")
    changed_paths = {runtime_relative, selection["path"]}
    for role in CHANGED_ROLES:
        _pin(files[role], records, role)
        _same(files[role]["path"], prior_files[role]["path"], "Inference revision changed a profile role path")
        changed_paths.add(files[role]["path"])
        # Carried rows: the exact approved appended files, on the exact base they were reviewed on.
        _same(_without(files[role], {"path"}), _without(carried_selected["files"][role], {"path"}),
              "Carried profile inputs differ from the owner-approved selection")
        _same(_without(prior_files[role], {"path"}), _without(authority["derivation"]["predecessor_inputs"][role], {"path"}),
              "Retrained predecessor profile inputs differ from the approved revision's reviewed base")
    for relative, record in prior_records.items():
        if relative not in changed_paths:
            _same(records.get(relative), record, f"Retained predecessor file changed: {relative}")
    if any(not path.startswith(NAMESPACE) for path in set(records) - set(prior_records)):
        raise ValueError("New inference evidence must use the revision provenance namespace")
    for relative in (NAMESPACE + "selection-approval.json", NAMESPACE + "owner-dob-correction.json"):
        if relative in records and records[relative] != prior_records.get(relative):
            raise ValueError("Conventional owner-authority paths may only retain exact predecessor records")
    provenance = payload.get("selected_provenance")
    if not isinstance(provenance, dict) or set(provenance) != REQUIRED_PROVENANCE:
        raise ValueError("Inference revision must retain all seven original provenance roles")
    required = {"models/xgboost_model.pkl", "models/xgboost_no_odds_model.pkl", "models/logistic_model.pkl",
                "processed/features.csv", "processed/fights_cleaned.csv", payload["saved_fullfit_spec"]["staged_path"],
                *provenance.values()}
    if not required.issubset(records):
        raise ValueError("Inference revision is missing retained model, processed, spec or provenance evidence")
    for role, item in files.items():
        if item.get("absent") is True:
            if (root / "explicitly-absent-inputs" / role).exists():
                raise ValueError("An explicitly absent inference input now exists")
        elif item.get("directory") is True:
            directory = _relative(item["path"])
            actual_directory = (root / directory).resolve(strict=True)
            actual_directory.relative_to(root.resolve(strict=True))
            declared = {path[len(directory) + 1:]: rec["sha256"] for path, rec in records.items() if path.startswith(directory + "/")}
            _same(declared, item["files"], "Inference input directory is not covered exactly")
        else:
            _pin({key: item[key] for key in ("path", "bytes", "sha256")}, records, role)
    recovery = payload.get("previous_rollback_identity", {})
    if recovery.get("automatic_resume_permitted") is not False or recovery.get("recovery_mode") != "paused":
        raise ValueError("Inference revision recovery must remain paused")
    derivation = _read(root, meta["derivation"]["path"])
    predecessor_inputs = derivation.get("predecessor_inputs", {})
    if not isinstance(predecessor_inputs, dict) or set(predecessor_inputs) != CHANGED_ROLES:
        raise ValueError("Inference revision requires both archived predecessor input files")
    for role, pin in predecessor_inputs.items():
        _pin(pin, records, "Archived predecessor " + role)
        if not pin["path"].startswith(NAMESPACE) or pin["path"].startswith(CARRIED_ARCHIVE):
            raise ValueError("Archived predecessor input must be the retrained predecessor's own copy")
        _same(_without(pin, {"path"}), _without(prior_files[role], {"path"}), "Archived predecessor profile input changed")
        _append_semantics(root, pin, files[role], role, owner_dob_correction=True)
    for key in ("source_candidate_manifest", "route_artifact", "carried_route_artifact", "compatibility_result"):
        _pin(derivation.get(key), records, key)
    _same(derivation, {
        "schema_version": 1, "kind": CARRIED_KIND, "new_training_performed": False,
        **{key: meta[key] for key in ("predecessor_manifest", "predecessor_selection", "predecessor_runtime",
                                      "carried_manifest", "base_selection_request", "owner_dob_correction")},
        "predecessor_inputs": predecessor_inputs, "successor_release_id": payload["bundle_id"],
        "successor_selection": selection, "successor_runtime": runtime_record,
        "changed_roles": {role: {"before": prior_files[role], "after": files[role]} for role in CHANGED_ROLES},
        "retained_provenance": {role: records[path] for role, path in provenance.items()},
        "unchanged_models": prior_runtime["models"], "unchanged_processed": prior_runtime["processed"],
        **{key: derivation[key] for key in ("source_candidate_manifest", "route_artifact", "carried_route_artifact", "compatibility_result")},
    }, "Derivation does not bind exact retrained predecessor and carried inference identities")
    # Routes: the approved route content, rebound only to this generation.
    carried_derivation = authority["derivation"]
    route_pin = derivation["route_artifact"]
    carried_route_pin = derivation["carried_route_artifact"]
    if (carried_route_pin["path"] != CARRIED_ARCHIVE + carried_derivation["route_artifact"]["path"]
            or _without(carried_route_pin, {"path"}) != _without(carried_derivation["route_artifact"], {"path"})):
        raise ValueError("Carried route artifact is not the approved route evidence")
    route, carried_route = _read(root, route_pin["path"]), _read(root, carried_route_pin["path"])
    if not CARRIED_ROUTE_FIELDS.issubset(route) or not CARRIED_ROUTE_FIELDS.issubset(carried_route):
        raise ValueError("Route artifact lacks its generation identity")
    _same(_without(route, CARRIED_ROUTE_FIELDS), _without(carried_route, CARRIED_ROUTE_FIELDS),
          "Carried inference revision changed reviewed route bindings, aliases or evidence")
    _same(route.get("selection_sha256"), runtime["selection"]["sha256"], "Route/source identity mismatch")
    _same(route.get("manifest_sha256"), runtime_record["sha256"], "Route/runtime identity mismatch")
    _same(route.get("release_id"), runtime["release_id"], "Route/release identity mismatch")
    _same(route.get("processed_fights_sha256"), runtime["processed"]["files"]["fights_cleaned.csv"]["sha256"],
          "Route/processed identity mismatch")
    _same(route.get("selected_sources_sha256"),
          {role: selected["files"][role]["sha256"] for role in ("profiles", "active_roster", "identity_aliases")},
          "Route selected input identities changed")
    source = _read(root, derivation["source_candidate_manifest"]["path"])
    source_records = {}
    for record in source.get("files", []):
        _record(record)
        if record["path"] in source_records:
            raise ValueError("Source candidate inventory contains duplicate paths")
        source_records[record["path"]] = record
    _same(_without(source_records.get("config/runtime_fighter_bindings_v1.json", {}), {"path"}), _without(route_pin, {"path"}),
          "Source candidate does not contain the exact successor route artifact")
    for item in route.get("evidence_files", []):
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("Invalid route evidence inventory record")
        path = "config/" + _relative(item["path"])
        if source_records.get(path, {}).get("sha256") != item["sha256"]:
            raise ValueError("Source candidate inventory omitted or changed route evidence")
    # A previous campaign, approval or self-summary cannot stand in for this
    # generation's executed compatibility job or independent technical review.
    stale_evidence_hashes = {record["sha256"] for record in authority["records"].values()}
    stale_evidence_hashes |= {record["sha256"] for record in prior_records.values()}
    stale_evidence_hashes |= {meta[key]["sha256"] for key in CARRIED_KEYS
                             - {"schema_version", "kind", "predecessor_release_id"}}
    stale_evidence_hashes |= {derivation[key]["sha256"] for key in
                             ("source_candidate_manifest", "route_artifact", "carried_route_artifact", "compatibility_result")}
    stale_evidence_hashes |= {selection["sha256"], runtime_record["sha256"]}
    stale_evidence_hashes |= {record["sha256"] for record in CARRIED_NEW_BYTES_CONTRACT["authority_records"]}
    compatibility = _read(root, derivation["compatibility_result"]["path"])
    _carried_fresh_evidence(compatibility.get("evidence"), records, stale_evidence_hashes,
                            CARRIED_COMPATIBILITY_EVIDENCE, "Compatibility")
    _same(compatibility, {
        "schema_version": 1, "kind": "inference_revision_compatibility", "status": "passed",
        "successor_release_id": payload["bundle_id"], "successor_selection": selection, "successor_runtime": runtime_record,
        **{key: derivation[key] for key in ("source_candidate_manifest", "route_artifact")},
        "models": runtime["models"], "processed": runtime["processed"],
        "checks": {key: True for key in COMPATIBILITY_CHECKS}, "evidence": compatibility["evidence"],
    }, "Compatibility results are incomplete or belong to another candidate")
    review = _read(root, meta["technical_review"]["path"])
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("Technical review must identify its independent reviewer")
    _carried_fresh_evidence(review.get("evidence"), records, stale_evidence_hashes,
                            CARRIED_REVIEW_EVIDENCE, "Technical review")
    _same(review, {
        "schema_version": 1, "scope": "inference_revision_technical_review", "status": "completed",
        "reviewer": review["reviewer"], "derivation": meta["derivation"], "compatibility_result": derivation["compatibility_result"],
        "checks": {key: True for key in REVIEW_CHECKS}, "evidence": review["evidence"], "findings": [],
    }, "Technical review is stale, unfinished or has unresolved findings")
    request = _read(root, meta["selection_request"]["path"])
    _evidence(request.get("new_bytes_authority"), records, "New data and model authority")
    _evidence(request.get("new_bytes_chain_evidence"), records, "Reviewed predecessor chain evidence")
    contract = CARRIED_NEW_BYTES_CONTRACT
    if expected_predecessor_manifest_sha256 != contract["predecessor_manifest_sha256"]:
        raise ValueError("Carried inference revision has no reviewed new-bytes contract for this predecessor")
    _same(request["new_bytes_authority"], contract["authority_records"],
          "New data and model authority differs from the reviewed source contract")
    _same(request["new_bytes_chain_evidence"], contract["chain_evidence"],
          "Reviewed predecessor chain evidence differs from the reviewed source contract")
    # Chain evidence is already bound by P': cite it in place, never copy or
    # re-encode it and describe it as new owner authority.
    for item in request["new_bytes_chain_evidence"]:
        _same(item, prior_records.get(item["path"]), "Reviewed chain evidence must be retained predecessor bytes")
    # Explicitly exclude own compatibility/review proofs and successor manifests
    # even if a future source-contract amendment accidentally lists them.
    self_evidence_hashes = {item["sha256"] for item in compatibility["evidence"] + review["evidence"]}
    self_evidence_hashes |= {selection["sha256"], runtime_record["sha256"], derivation["compatibility_result"]["sha256"],
                             meta["technical_review"]["sha256"], meta["derivation"]["sha256"], meta["selection_request"]["sha256"]}
    if any(item["sha256"] in self_evidence_hashes
           for item in request["new_bytes_authority"] + request["new_bytes_chain_evidence"]):
        raise ValueError("This revision's own evidence cannot authorize its data or model bytes")
    _same(request, {
        "schema_version": 1, "request_id": request.get("request_id"), "kind": CARRIED_APPLICATION_KIND,
        "description": request.get("description"), "scope": "inference_selection_only",
        "derivation": meta["derivation"], "technical_review": meta["technical_review"],
        **{key: derivation[key] for key in ("source_candidate_manifest", "route_artifact")},
        "successor_selection": selection, "successor_runtime": runtime_record,
        "carried_manifest": meta["carried_manifest"], "base_selection_request": meta["base_selection_request"],
        "owner_dob_correction": meta["owner_dob_correction"],
        "subject_ids": list(SUBJECT_IDS), "retained_models": runtime["models"], "retained_processed": runtime["processed"],
        "strict_method_coverage": True, "excluded_authorities": EXCLUDED_AUTHORITIES,
        "new_bytes_authority": request["new_bytes_authority"],
        "new_bytes_chain_evidence": request["new_bytes_chain_evidence"],
    }, "Carried application record does not cover the exact scoped inference revision")
    approval = _approval(root, meta, request)
    identities = [approval["approval_id"], authority["base_request"]["request_id"], correction["correction_id"],
                  authority["request"]["request_id"], request["request_id"]]
    if len(set(identities)) != len(identities):
        raise ValueError("Owner correction and its applications require distinct fresh identities")
    return {
        "selected_runtime_manifest_path": str(confined_file(root, runtime_relative)),
        "selected_runtime_manifest_sha256": runtime_record["sha256"], "selected_release_id": payload["bundle_id"],
        "selected_candidate_schema_version": 2, "selected_release_file_count": len(records),
        "selected_source_manifest_path": str(confined_file(root, selection["path"])),
        "selected_source_manifest_sha256": selection["sha256"], "selected_inference_revision_verified": True,
        "selected_inference_revision_predecessor_sha256": expected_predecessor_manifest_sha256,
        "selected_inference_revision_approval_sha256": expected_selection_approval_sha256,
        "selected_owner_dob_correction_verified": True,
        "selected_owner_dob_correction_sha256": expected_owner_dob_correction_sha256,
        "selected_inference_revision_base_request_sha256": meta["base_selection_request"]["sha256"],
        "selected_inference_revision_kind": CARRIED_KIND,
        "selected_inference_revision_carried_manifest_sha256": CARRIED_APPROVED_MANIFEST_SHA256,
    }


def validate_inference_revision(payload, root, *, expected_predecessor_manifest_sha256=BASELINE_MANIFEST_SHA256,
                                expected_selection_approval_sha256=None, expected_owner_dob_correction_sha256=None):
    """Read an exact graph with independent predecessor and owner-approval pins."""
    _hash(expected_predecessor_manifest_sha256, "predecessor")
    _hash(expected_selection_approval_sha256, "selection approval")
    root = Path(root)
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("Inference revision root cannot traverse symlinks")
    _same(payload.get("selected_candidate_schema_version"), 2, "Unsupported inference revision schema")
    _same(payload.get("manifest_version"), 3, "Inference revision requires rich production manifest v3")
    _same(payload.get("staging_schema_version"), 1, "Inference revision requires staging schema 1")
    if not isinstance(payload.get("bundle_id"), str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,180}", payload["bundle_id"]):
        raise ValueError("Invalid inference revision release ID")
    inventory = payload.get("release_file_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Inference revision inventory is missing")
    records = {}
    for record in inventory:
        _record(record)
        relative = record["path"]
        if relative in records or relative in {"staging_manifest.json", "manifest.json"}:
            raise ValueError("Duplicate or recursive inference revision inventory")
        path = confined_file(root, relative)
        if record["bytes"] != path.stat().st_size or record["sha256"] != file_hash(path):
            raise ValueError(f"Inference revision file identity mismatch: {relative}")
        records[relative] = record
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Inference revision tree cannot contain symlinks")
    actual = {p.relative_to(root).as_posix() for p in paths if p.is_file()}
    _installed_envelope(payload, root, actual, records)
    meta = payload.get("selected_inference_revision")
    if isinstance(meta, dict) and meta.get("kind") == CARRIED_KIND:
        return _validate_carried_revision(payload, root, records, meta, expected_predecessor_manifest_sha256,
                                          expected_selection_approval_sha256, expected_owner_dob_correction_sha256)
    amended = isinstance(meta, dict) and meta.get("kind") == OWNER_DOB_KIND
    keys = {"schema_version", "kind", "predecessor_release_id", "predecessor_manifest", "predecessor_selection",
            "predecessor_runtime", "derivation", "technical_review", "selection_request", "selection_approval"}
    if amended:
        keys |= {"base_selection_request", "owner_dob_correction"}
    if not isinstance(meta, dict) or set(meta) != keys:
        raise ValueError("Inference revision metadata has missing or unexpected fields")
    _same(meta["schema_version"], 1, "Invalid inference revision metadata schema")
    _same(meta["kind"], OWNER_DOB_KIND if amended else KIND, "Unsupported inference revision kind")
    for key in keys - {"schema_version", "kind", "predecessor_release_id"}:
        _pin(meta[key], records, key)
    if meta["predecessor_manifest"]["sha256"] != expected_predecessor_manifest_sha256:
        raise ValueError("Predecessor manifest differs from the external trust anchor")
    if meta["selection_approval"]["sha256"] != expected_selection_approval_sha256:
        raise ValueError("Selection approval differs from the external trust anchor")
    correction = _owner_correction(root, meta, expected_owner_dob_correction_sha256) if amended else None
    previous = _read(root, meta["predecessor_manifest"]["path"])
    _same(previous.get("selected_candidate_schema_version"), 1, "Inference revision must derive directly from integer 1")
    if payload["bundle_id"] == previous.get("bundle_id") or meta["predecessor_release_id"] != previous.get("bundle_id"):
        raise ValueError("Inference revision predecessor/successor release identity mismatch")
    changes = {"bundle_id", "selected_candidate_schema_version", "selected_runtime_manifest_sha256",
               "release_file_inventory", "selected_inference_revision"}
    _same(_without(_unrebase(payload, previous, root), changes), _without(previous, changes),
          "Inference revision changed retained training or production metadata")
    prior_records = {}
    for record in previous["release_file_inventory"]:
        _record(record)
        if record["path"] in prior_records:
            raise ValueError("Predecessor inventory contains duplicate records")
        prior_records[record["path"]] = record
    runtime_relative = payload["selected_runtime_manifest_path"]
    _same(_without(meta["predecessor_runtime"], {"path"}), _without(prior_records[runtime_relative], {"path"}),
          "Archived predecessor runtime changed")
    prior_runtime = _read(root, meta["predecessor_runtime"]["path"])
    _same(_without(meta["predecessor_selection"], {"path"}), _without(prior_runtime["selection"], {"path"}),
          "Archived predecessor selection changed")
    prior_selected = _read(root, meta["predecessor_selection"]["path"])
    runtime_record = records[runtime_relative]
    _same(runtime_record["sha256"], payload["selected_runtime_manifest_sha256"], "Inference runtime inventory identity mismatch")
    runtime = _read(root, runtime_relative)
    _same(_without(runtime, {"release_id", "selection"}), _without(prior_runtime, {"release_id", "selection"}),
          "Inference revision changed runtime models, processed pair, history or observations")
    _same(runtime.get("schema_version"), 1, "Runtime schema must remain integer 1")
    _same(runtime.get("release_id"), payload["bundle_id"], "Inference runtime belongs to another release")
    _same(runtime.get("approved_history_cutoff"), payload.get("snapshot_max_event_date"), "Inference history cutoff changed")
    selection = _pin(runtime.get("selection"), records, "Inference source selection")
    _same(selection["path"], prior_runtime["selection"]["path"], "Inference selector path changed")
    selected = _read(root, selection["path"])
    _same(selected.get("status"), "inference_selection_revision", "Inference selector status is incorrect")
    _same(selected.get("inference_revision"), {
        "predecessor_selection_sha256": prior_runtime["selection"]["sha256"],
        "changed_roles": sorted(CHANGED_ROLES), "subject_ids": list(SUBJECT_IDS),
        "predecessor_release_relocation": prior_selected.get("release_relocation"),
        **({"owner_dob_correction_sha256": meta["owner_dob_correction"]["sha256"]} if amended else {}),
    }, "Inference selector revision metadata is incorrect")
    _same(selected.get("release_relocation"), {
        "original_selection_sha256": prior_selected["release_relocation"]["original_selection_sha256"],
        "policy": "profile-only inference revision; original model-training selection retained unchanged",
    }, "Inference selector misstates original training selection")
    source_changes = {"status", "files", "release_relocation", "inference_revision"}
    _same(_without(selected, source_changes), _without(prior_selected, source_changes),
          "Inference selector changed historical metadata")
    files, prior_files = selected["files"], prior_selected["files"]
    if set(files) != set(prior_files) or {role for role in files if files[role] != prior_files[role]} != CHANGED_ROLES:
        raise ValueError("Inference revision may change exactly profiles and profile_decisions")
    changed_paths = {runtime_relative, selection["path"]}
    for role in CHANGED_ROLES:
        _pin(files[role], records, role)
        _same(files[role]["path"], prior_files[role]["path"], "Inference revision changed a profile role path")
        changed_paths.add(files[role]["path"])
    for relative, record in prior_records.items():
        if relative not in changed_paths:
            _same(records.get(relative), record, f"Retained predecessor file changed: {relative}")
    if any(not path.startswith(NAMESPACE) for path in set(records) - set(prior_records)):
        raise ValueError("New inference evidence must use the revision provenance namespace")
    provenance = payload.get("selected_provenance")
    if not isinstance(provenance, dict) or set(provenance) != REQUIRED_PROVENANCE:
        raise ValueError("Inference revision must retain all seven original provenance roles")
    required = {"models/xgboost_model.pkl", "models/xgboost_no_odds_model.pkl", "models/logistic_model.pkl",
                "processed/features.csv", "processed/fights_cleaned.csv", payload["saved_fullfit_spec"]["staged_path"],
                *provenance.values()}
    if not required.issubset(records):
        raise ValueError("Inference revision is missing retained model, processed, spec or provenance evidence")
    for role, item in files.items():
        if item.get("absent") is True:
            if (root / "explicitly-absent-inputs" / role).exists():
                raise ValueError("An explicitly absent inference input now exists")
        elif item.get("directory") is True:
            directory = _relative(item["path"])
            actual_directory = (root / directory).resolve(strict=True)
            actual_directory.relative_to(root.resolve(strict=True))
            declared = {path[len(directory) + 1:]: rec["sha256"] for path, rec in records.items() if path.startswith(directory + "/")}
            _same(declared, item["files"], "Inference input directory is not covered exactly")
        else:
            _pin({key: item[key] for key in ("path", "bytes", "sha256")}, records, role)
    recovery = payload.get("previous_rollback_identity", {})
    if recovery.get("automatic_resume_permitted") is not False or recovery.get("recovery_mode") != "paused":
        raise ValueError("Inference revision recovery must remain paused")
    derivation = _read(root, meta["derivation"]["path"])
    predecessor_inputs = derivation.get("predecessor_inputs", {})
    if not isinstance(predecessor_inputs, dict) or set(predecessor_inputs) != CHANGED_ROLES:
        raise ValueError("Inference revision requires both archived predecessor input files")
    for role, pin in predecessor_inputs.items():
        _pin(pin, records, "Archived predecessor " + role)
        _same(_without(pin, {"path"}), _without(prior_files[role], {"path"}), "Archived predecessor profile input changed")
        _append_semantics(root, pin, files[role], role, owner_dob_correction=amended)
    for key in ("source_candidate_manifest", "route_artifact", "predecessor_route_artifact", "compatibility_result"):
        _pin(derivation.get(key), records, key)
    _same(derivation, {
        "schema_version": 1, "kind": OWNER_DOB_KIND if amended else KIND, "new_training_performed": False,
        **{key: meta[key] for key in ("predecessor_manifest", "predecessor_selection", "predecessor_runtime")},
        "predecessor_inputs": predecessor_inputs, "successor_release_id": payload["bundle_id"],
        "successor_selection": selection, "successor_runtime": runtime_record,
        "changed_roles": {role: {"before": prior_files[role], "after": files[role]} for role in CHANGED_ROLES},
        "retained_provenance": {role: records[path] for role, path in provenance.items()},
        "unchanged_models": prior_runtime["models"], "unchanged_processed": prior_runtime["processed"],
        **{key: derivation[key] for key in ("source_candidate_manifest", "route_artifact", "predecessor_route_artifact", "compatibility_result")},
        **({"base_selection_request": meta["base_selection_request"], "owner_dob_correction": meta["owner_dob_correction"],
            "base_source_selection": derivation.get("base_source_selection"), "base_profile_decisions": derivation.get("base_profile_decisions")}
           if amended else {}),
    }, "Derivation does not bind exact original training and revised inference identities")
    base_request = _correction_base(root, meta, derivation, records, selected, runtime, correction) if amended else None
    _routes(root, derivation, records, runtime, selected, owner_dob_correction=amended)
    compatibility = _read(root, derivation["compatibility_result"]["path"])
    _evidence(compatibility.get("evidence"), records, "Compatibility")
    _same(compatibility, {
        "schema_version": 1, "kind": "inference_revision_compatibility", "status": "passed",
        "successor_release_id": payload["bundle_id"], "successor_selection": selection, "successor_runtime": runtime_record,
        **{key: derivation[key] for key in ("source_candidate_manifest", "route_artifact")},
        "models": runtime["models"], "processed": runtime["processed"],
        "checks": {key: True for key in COMPATIBILITY_CHECKS}, "evidence": compatibility["evidence"],
    }, "Compatibility results are incomplete or belong to another candidate")
    review = _read(root, meta["technical_review"]["path"])
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("Technical review must identify its independent reviewer")
    _evidence(review.get("evidence"), records, "Technical review")
    _same(review, {
        "schema_version": 1, "scope": "inference_revision_technical_review", "status": "completed",
        "reviewer": review["reviewer"], "derivation": meta["derivation"], "compatibility_result": derivation["compatibility_result"],
        "checks": {key: True for key in REVIEW_CHECKS}, "evidence": review["evidence"], "findings": [],
    }, "Technical review is stale, unfinished or has unresolved findings")
    request = _read(root, meta["selection_request"]["path"])
    for key in ("request_id", "description"):
        if not isinstance(request.get(key), str) or not request[key].strip():
            raise ValueError("Selection request must identify and explain the exact proposed decision")
    _same(request, {
        "schema_version": 1, "request_id": request["request_id"], "description": request["description"],
        "scope": "inference_selection_only", "derivation": meta["derivation"], "technical_review": meta["technical_review"],
        **{key: derivation[key] for key in ("source_candidate_manifest", "route_artifact")},
        "successor_selection": selection, "successor_runtime": runtime_record,
        "subject_ids": list(SUBJECT_IDS), "retained_models": runtime["models"], "retained_processed": runtime["processed"],
        "strict_method_coverage": True, "excluded_authorities": EXCLUDED_AUTHORITIES,
        **({"kind": "application_of_existing_owner_authority", "base_selection_request": meta["base_selection_request"],
            "owner_dob_correction": meta["owner_dob_correction"]} if amended else {}),
    }, "Selection request does not cover the exact scoped inference revision")
    approval = _read(root, meta["selection_approval"]["path"])
    for key in ("approval_id", "verbatim_decision", "original_timestamp", "attribution"):
        if not isinstance(approval.get(key), str) or not approval[key].strip():
            raise ValueError("Selection approval must retain the actual attributed owner decision")
    timestamp = approval["original_timestamp"]
    match = re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.(\d{1,9}))?(?:Z|[+-]\d{2}:\d{2})", timestamp)
    if not match or type(approval.get("timestamp_precision")) is not int or approval["timestamp_precision"] != len(match.group(1) or ""):
        raise ValueError("Selection approval must preserve original timestamp offset and precision")
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if approval["approval_id"] == request["request_id"]:
        raise ValueError("Selection approval requires a fresh decision identity")
    if amended and (approval["approval_id"] == base_request["request_id"]
                    or request["request_id"] == base_request["request_id"] or correction["correction_id"] in
                    {approval["approval_id"], request["request_id"], base_request["request_id"]}):
        raise ValueError("Owner correction and its application require distinct fresh identities")
    _same(approval, {
        "schema_version": 1, "approval_id": approval["approval_id"],
        "request_sha256": meta["base_selection_request" if amended else "selection_request"]["sha256"],
        "decision": "approved", "scope": "inference_selection_only",
        **{key: approval[key] for key in ("verbatim_decision", "original_timestamp", "timestamp_precision", "attribution")},
    }, "Selection approval is stale or does not approve this exact request")
    return {
        "selected_runtime_manifest_path": str(confined_file(root, runtime_relative)),
        "selected_runtime_manifest_sha256": runtime_record["sha256"], "selected_release_id": payload["bundle_id"],
        "selected_candidate_schema_version": 2, "selected_release_file_count": len(records),
        "selected_source_manifest_path": str(confined_file(root, selection["path"])),
        "selected_source_manifest_sha256": selection["sha256"], "selected_inference_revision_verified": True,
        "selected_inference_revision_predecessor_sha256": expected_predecessor_manifest_sha256,
        "selected_inference_revision_approval_sha256": expected_selection_approval_sha256,
        **({"selected_owner_dob_correction_verified": True,
            "selected_owner_dob_correction_sha256": expected_owner_dob_correction_sha256,
            "selected_inference_revision_base_request_sha256": meta["base_selection_request"]["sha256"]}
           if amended else {}),
    }
