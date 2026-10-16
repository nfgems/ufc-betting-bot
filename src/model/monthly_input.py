"""Registration-time, independently accepted monthly inputs; no model work.

The small request is an external acceptance pin and an unused save root. A save
never invents that authority by hashing the active publication. Recovery uses
only saved originals and the same outer pins. Partial roots are consumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd

from src.data.method_observation_history import _attached, _digest, _json_bytes
from src.data.publication_identity import (
    APPROVED_V7_BASELINE_LINEAGE, accepted_data_generation_from_artifacts,
    _accepted_raw_generation_identity, _validated_calendar_claim,
)
from src.model import prepared_input as prepared
from src.model.monthly_candidate import build_monthly_modeling_view, _dataframe_csv_bytes
from src.model.training_preflight import preflight_training_data, resolved_spec_identity
from src.model.training_spec import resolve_named_training_spec

ACCEPTANCE_KIND = "accepted_monthly_input_v1"
TRANSPORT_KIND = "saved_monthly_input_transport_v1"
TEMPORAL_CONTRACT = "existing_monthly_view_v1"
ACCEPTANCE_KIND_V2 = "accepted_monthly_input_v2"
TRANSPORT_KIND_V2 = "saved_monthly_input_transport_v2"
CONTEXT_FIELDS = frozenset({"utc_month", "accepted_update_id", "accepted_at_utc", "iso_week",
                            "data_slot", "schedule_timezone", "promotion_month_utc", "claim_id"})


class MonthlyInputRefusal(ValueError):
    """Original accepted monthly identity or complete saved closure is unavailable."""


def _require(value, message):
    if not value:
        raise MonthlyInputRefusal(message)


def _time(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise MonthlyInputRefusal("invalid monthly timestamp") from exc
    _require(result.tzinfo is not None and result.utcoffset() is not None,
             "monthly timestamps must be timezone-aware")
    return result.astimezone(timezone.utc)


def _context(context):
    _require(isinstance(context, dict) and set(context) == CONTEXT_FIELDS,
             "monthly context must contain all immutable selection bindings")
    _require(all(isinstance(value, str) for value in context.values()), "invalid monthly context text")
    _require(context["utc_month"] == _time(context["accepted_at_utc"]).strftime("%Y-%m"),
             "monthly cutoff differs from selected UTC month")
    _require(context["accepted_update_id"] and context["schedule_timezone"]
             and context["iso_week"] and context["data_slot"] in {"tuesday", "wednesday"},
             "missing monthly generation or schedule binding")
    return dict(context)


def _document(body, context, *, resolve_spec=True, policy_body=None):
    value = prepared._json(body)
    fields = {"schema", "kind", "context", "training_spec", "publication", "completion", "features",
              "prepared_acceptance", "processed_fights", "raw_root", "temporal_contract"}
    version = value.get("schema") if isinstance(value, dict) else None
    if type(version) is int and version == 2:
        fields.add("policy")
    _require(isinstance(value, dict) and set(value) == fields and type(version) is int
             and version in {1, 2} and value["kind"] == {1: ACCEPTANCE_KIND, 2: ACCEPTANCE_KIND_V2}[version],
             "invalid independently accepted monthly input descriptor")
    _require(value["context"] == _context(context), "monthly acceptance context conflict")
    policy = None
    if version == 2:
        from src.model.monthly_policy import ADAPTER, load_monthly_policy, validate_policy_context
        _require(value["temporal_contract"] == ADAPTER, "unapproved monthly temporal contract")
        policy = validate_policy_context(load_monthly_policy(value["policy"], body=policy_body, context=context), context)
    else:
        _require(value["temporal_contract"] == TEMPORAL_CONTRACT, "unapproved monthly temporal contract")
    for name in ("publication", "completion", "features", "prepared_acceptance", "processed_fights"):
        prepared._pin(value[name])
    _require(Path(value["raw_root"]).is_absolute(), "monthly accepted raw root must be absolute")
    prepared._path(value["raw_root"])
    identity = value["training_spec"]
    _require(isinstance(identity, dict) and set(identity) == {"name", "sha256"}, "invalid monthly named spec")
    _require(isinstance(identity["name"], str) and identity["name"].strip(), "missing monthly named spec identity")
    prepared._sha(identity["sha256"])
    if policy is not None:
        _require(identity == policy.binding["training_spec"], "monthly policy/accepted recipe conflict")
        return value, policy.spec if resolve_spec else None
    if not resolve_spec:
        return value, None
    spec = resolve_named_training_spec(identity["name"])
    _require(identity == {"name": spec.name, "sha256": resolved_spec_identity(spec)},
             "monthly resolved named spec differs from independent acceptance")
    return value, spec


def _pin_key(pin):
    prepared._pin(pin)
    return _digest(_json_bytes(pin))


def _collect(pin, attachments):
    """Capture only explicitly pinned direct ancestry, once, including nested JSON."""
    key = _pin_key(pin)
    if key in attachments:
        return
    body = prepared._capture(pin)
    attachments[key] = prepared._attach(body, pin)
    if str(pin["path"]).lower().endswith(".json"):
        _collect_children(prepared._json(body), attachments)


def _collect_children(value, attachments):
    if isinstance(value, dict):
        if {"state_root", "store_present", "objects", "artifacts", "closure_sha256", "references"}.issubset(value):
            for pin in _closure_pins(value).values():
                _collect(pin, attachments)
        if set(value) == {"path", "bytes", "sha256"}:
            _collect(value, attachments)
        elif set(value) == {"pin", "body_base64"}:
            # Already embedded originals do not require their former paths.
            body = _attached(value)
            attachments[_pin_key(value["pin"])] = value
            if str(value["pin"]["path"]).lower().endswith(".json"):
                _collect_children(prepared._json(body), attachments)
        else:
            for child in value.values():
                _collect_children(child, attachments)
    elif isinstance(value, list):
        for child in value:
            _collect_children(child, attachments)


def _get(pin, attachments):
    attachment = attachments.get(_pin_key(pin))
    _require(isinstance(attachment, dict) and attachment.get("pin") == pin,
             "missing original monthly evidence attachment")
    return _attached(attachment)


def _raw_pins(document, publication):
    inventory = publication.get("refresh_input_inventory")
    _require(isinstance(inventory, dict) and set(inventory) == {"schema_version", "artifacts"},
             "missing complete refresh input inventory")
    _accepted_raw_generation_identity(inventory)
    pins = {}
    root = prepared._path(document["raw_root"])
    for name, identity in inventory["artifacts"].items():
        _require(isinstance(name, str) and name and not Path(name).is_absolute()
                 and ".." not in Path(name).parts and "\\" not in name,
                 "unsafe refresh inventory relative path")
        _require(isinstance(identity, dict) and set(identity) == {"bytes", "sha256"},
                 "invalid refresh inventory identity")
        pins[name] = prepared._pin({"path": str(root / name), **identity})
    return pins


def _transaction_pins(transaction):
    root = prepared._path(transaction["accepted_root"])
    result = {}
    for row in transaction.get("artifacts") or []:
        name = row.get("relative_path")
        _require(isinstance(name, str) and name and not Path(name).is_absolute()
                 and ".." not in Path(name).parts and "\\" not in name, "unsafe transaction artifact path")
        if row.get("after") is not None:
            result[name] = prepared._pin({"path": str(root / name), **row["after"]})
    return result


def _candidate_pair_pin(transaction):
    pair = transaction.get("pair_manifest_path")
    _require(isinstance(pair, str) and pair and not Path(pair).is_absolute()
             and ".." not in Path(pair).parts and "\\" not in pair, "unsafe original candidate pair path")
    rows = [row for row in transaction.get("artifacts") or [] if row.get("relative_path") == pair]
    _require(len(rows) == 1 and isinstance(rows[0].get("candidate"), dict),
             "original candidate pair identity missing")
    root = prepared._path(transaction.get("candidate_root"))
    return prepared._pin({"path": str(root / pair), **rows[0]["candidate"]})


def _collect_transaction_children(value, attachments):
    if isinstance(value, dict):
        if {"accepted_root", "artifacts", "pair_manifest_path", "commit_order"}.issubset(value):
            for pin in _transaction_pins(value).values():
                _collect(pin, attachments)
            _collect(_candidate_pair_pin(value), attachments)
        for child in value.values():
            _collect_transaction_children(child, attachments)
    elif isinstance(value, list):
        for child in value:
            _collect_transaction_children(child, attachments)


def _transaction(transaction, document, publication, attachments=None):
    _require(isinstance(transaction, dict) and transaction.get("status") == "committed",
             "monthly publication transaction did not commit")
    pointer = transaction.get("pointer")
    _require(isinstance(pointer, dict) and pointer.get("target_path") == document["publication"]["path"]
             and pointer.get("after") == {key: document["publication"][key] for key in ("bytes", "sha256")},
             "monthly transaction pointer differs from accepted original publication")
    _require(transaction.get("accepted_root") == document["raw_root"], "monthly transaction raw root conflict")
    artifacts = transaction.get("artifacts")
    _require(isinstance(artifacts, list) and artifacts, "monthly transaction artifact evidence missing")
    found = set()
    inventory = publication["refresh_input_inventory"]["artifacts"]
    pair = transaction.get("pair_manifest_path")
    for artifact in artifacts:
        _require(isinstance(artifact, dict), "invalid monthly transaction artifact")
        name = artifact.get("relative_path")
        _require(isinstance(name, str) and name not in found, "duplicate monthly transaction artifact")
        found.add(name)
        if artifact.get("status") == "missing_optional":
            _require(artifact.get("required") is False and name not in inventory,
                     "required monthly transaction artifact skipped")
            continue
        _require(artifact.get("status") in {"create", "replace", "preserved_optional"}
                 and (artifact.get("candidate") == artifact.get("after") or
                      artifact.get("status") == "preserved_optional" or name == pair)
                 and isinstance(artifact.get("after"), dict), "failed monthly transaction artifact")
        if artifact.get("status") == "preserved_optional":
            _require(artifact.get("required") is False and artifact.get("before") == artifact.get("after"),
                     "preserved optional transaction artifact changed")
        if name in inventory:
            _require(artifact["after"] == inventory[name], "transaction differs from refresh inventory")
    replaced = [row["relative_path"] for row in artifacts if row.get("status") in {"create", "replace"}]
    expected_order = [name for name in replaced if name != pair] + [
        "pointer:" + document["publication"]["path"], pair]
    _require(isinstance(pair, str) and pair in replaced and transaction.get("commit_order") == expected_order,
        "monthly transaction commit order contradicts artifacts")
    pins = _transaction_pins(transaction)
    # Read-only rebuild inputs can legitimately be outside the transaction.
    # Every declared transaction entry must nevertheless have exact saved bytes.
    bodies = {name: (_get(pin, attachments) if attachments is not None else prepared._capture(pin))
              for name, pin in pins.items()}
    from src.data.io_utils import _validate_manifest, _csv_record, _parse_csv_bytes
    pair_body = bodies[pair]
    pair_document = prepared._json(pair_body)
    records = []
    declared = pair_document.get("artifacts") if isinstance(pair_document, dict) else None
    _require(isinstance(declared, list) and len(declared) == 2, "transaction pair evidence incomplete")
    for row in declared:
        path = prepared._path(row.get("path"))
        relative = path.relative_to(prepared._path(document["raw_root"])).as_posix()
        _require(relative in bodies and relative in inventory, "pair artifact absent from accepted raw inventory")
        body = bodies[relative]
        records.append(_csv_record(path, body, _parse_csv_bytes(path, body)))
    _validate_manifest(Path(pins[pair]["path"]), pair_body, records)
    # The real publisher rewrites only the pair's two absolute artifact paths.
    # Its original candidate marker therefore has a different hash from the
    # accepted marker. Retain and validate both, rather than treating a marker
    # hash difference as permission for any other transaction artifact change.
    candidate_pin = _candidate_pair_pin(transaction)
    candidate_body = _get(candidate_pin, attachments) if attachments is not None else prepared._capture(candidate_pin)
    candidate_root = prepared._path(transaction["candidate_root"])
    source_records = []
    for record in records:
        relative = Path(record["path"]).relative_to(prepared._path(document["raw_root"])).as_posix()
        source_path = candidate_root / relative
        source_records.append(_csv_record(source_path, bodies[relative], _parse_csv_bytes(source_path, bodies[relative])))
    source_payload = _validate_manifest(Path(candidate_pin["path"]), candidate_body, source_records)
    for row, accepted in zip(source_payload["artifacts"], records, strict=True):
        row["path"] = accepted["path"]
    _require((json.dumps(source_payload, indent=2) + "\n").encode("utf-8") == pair_body,
             "accepted pair differs from original publisher path-only rewrite")


def _quality(evidence, attachments=None):
    get = (lambda pin: _get(pin, attachments)) if attachments is not None else prepared._capture
    quality = prepared._json(get(prepared._pin(evidence.get("quality"))))
    _require(type(quality.get("schema")) is int and quality["schema"] == 1
             and quality.get("kind") == "roster_refresh_quality_decision_v1"
             and quality.get("passed") is True and quality.get("execution_failed") is False
             and quality.get("errors") == [] and isinstance(quality.get("checks"), dict)
             and quality["checks"] and all(value is True for value in quality["checks"].values())
             and isinstance(quality.get("processed_identity_guard"), dict)
             and not quality["processed_identity_guard"].get("violations"), "roster quality decision failed")
    report = prepared._json(get(prepared._pin(quality.get("report"))))
    checks = report.get("checks")
    _require(report.get("passed") is True and report.get("errors") == [] and isinstance(checks, dict)
             and checks and quality["checks"] == {
                 name: isinstance(check, dict) and check.get("passed") is True for name, check in checks.items()},
             "roster quality report failed or contradicts decision")
    if "processing" in evidence:
        _require(quality.get("processing") == evidence["processing"], "quality/processing original pins conflict")
    return quality


def _closure_pins(closure):
    _require(isinstance(closure.get("objects"), dict) and isinstance(closure.get("artifacts"), list)
             and sorted(closure["objects"]) == closure["artifacts"], "identity closure inventory conflict")
    root = prepared._path(closure["state_root"])
    pins = {}
    for name, identity in closure["objects"].items():
        _require(isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts
                 and "\\" not in name, "unsafe identity closure path")
        pins[name] = prepared._pin({"path": str(root / name), **identity})
    return pins


def _identity_closure(evidence, publication, attachments=None):
    declared = publication.get("identity_evidence_closure_sha256")
    if "processing" not in evidence:
        _require(declared in (None, "") and publication.get("identity_evidence_artifact_count", 0) == 0,
                 "published identity evidence closure has no original processing binding")
        return
    get = (lambda pin: _get(pin, attachments)) if attachments is not None else prepared._capture
    processing = prepared._json(get(prepared._pin(evidence["processing"])))
    _require(type(processing.get("schema")) is int and processing["schema"] == 1
             and processing.get("kind") == "roster_refresh_processing_v1",
             "invalid original processing receipt schema or kind")
    closure = processing.get("identity_evidence_closure")
    _require(isinstance(closure, dict), "original processing identity closure missing")
    pins = _closure_pins(closure)
    bodies = {name: get(pin) for name, pin in pins.items()}
    if closure.get("store_present") is False:
        _require(not pins and closure.get("references") == [] and closure["closure_sha256"] == _digest(b""),
                 "absent identity store contradicts closure")
    else:
        _require(closure.get("store_present") is True and pins, "invalid original identity store closure")
        canonical = json.dumps(closure["objects"], sort_keys=True, separators=(",", ":")).encode("utf-8")
        _require(closure["closure_sha256"] == _digest(canonical), "original identity closure hash conflict")
        from pathlib import PurePosixPath
        from types import SimpleNamespace
        from src.data.rosterwatch_identity_evidence import (
            IdentityEvidenceStore, STORE_DIRECTORY, _FORMAT, MAX_MANIFEST_BYTES,
        )
        _require(all(name.startswith(STORE_DIRECTORY + "/")
                     and PurePosixPath(name).as_posix() == name for name in bodies),
                 "identity closure objects must use the original store layout")
        directories = {STORE_DIRECTORY, STORE_DIRECTORY + "/objects"}
        for name in bodies:
            directories.update(str(parent) for parent in PurePosixPath(name).parents if str(parent) != ".")
        _require(not set(bodies).intersection(directories), "identity closure file/directory conflict")

        class SavedEvidencePath:
            """Read-only directory view of already captured originals.

            The existing store verifier receives its original directory API.
            No active path is reopened and no replacement store is written.
            """
            def __init__(self, name):
                self.key = name
            def __truediv__(self, component):
                return SavedEvidencePath(self.key + "/" + component)
            @property
            def name(self):
                return PurePosixPath(self.key).name
            @property
            def suffix(self):
                return PurePosixPath(self.key).suffix
            @property
            def stem(self):
                return PurePosixPath(self.key).stem
            def exists(self):
                return self.key in bodies or self.key in directories
            def is_symlink(self):
                return False  # Captured byte objects have no link behavior.
            def iterdir(self):
                _require(self.key in directories, "missing captured identity directory")
                names = {name for name in set(bodies) | directories
                         if str(PurePosixPath(name).parent) == self.key}
                return iter(SavedEvidencePath(name) for name in sorted(names))
            def stat(self):
                _require(self.key in bodies, "missing captured identity control file")
                return SimpleNamespace(st_size=len(bodies[self.key]))

        class SavedIdentityEvidence(IdentityEvidenceStore):
            def __init__(self):
                self.root = SavedEvidencePath(STORE_DIRECTORY)
                _require(self._read(self.root / "format.json", MAX_MANIFEST_BYTES) == _FORMAT,
                         "unknown captured identity evidence store version")
                # Reuse every original verify rule: layout, object names and
                # hashes, all manifests (including unreferenced history),
                # legacy-originals pins, count/size/control-byte budgets.
                self.verify()
            def _read(self, path, maximum):
                name = path.key
                _require(name in bodies and len(bodies[name]) <= maximum, "missing or oversized original identity object")
                return bodies[name]
        store = SavedIdentityEvidence()
        _require(isinstance(closure.get("references"), list), "missing identity closure references")
        for reference in closure["references"]:
            store.get_bytes(reference)
    _require(evidence.get("identity_evidence_closure_sha256") == closure["closure_sha256"]
             or (not pins and evidence.get("identity_evidence_closure_sha256") is None),
             "completion identity evidence closure differs from original processing")
    if pins and evidence.get("outcome") == "published" and publication.get("published_by") == "roster_refresh_integration":
        _require(declared == closure["closure_sha256"], "bridge publication cannot downgrade required identity closure")
    if declared in (None, ""):
        _require(publication.get("identity_evidence_artifact_count", 0) == 0,
                 "absent published identity closure contradicts artifact count")
    if declared not in (None, ""):
        _require(declared == closure["closure_sha256"]
                 and publication.get("identity_evidence_artifact_count") == len(pins),
                 "published identity closure hash or count conflict")


def _unchanged(evidence, publication, document, attachments=None):
    comparison = evidence.get("comparison") or {}
    checks = comparison.get("checks") or {}
    required = {"pair_marker_unchanged", "accepted_generation_id_matches", "processed_identity_matches",
                "operational_inventory_matches"}
    _require(evidence.get("original_publication_preserved") is True
             and evidence.get("publication_performed") is False and comparison.get("unchanged") is True
             and isinstance(checks, dict) and required.issubset(checks)
             and all(value is True for value in checks.values()) and comparison.get("operational_differences") == []
             and comparison.get("candidate_generation_id") == publication["accepted_generation_id"],
             "unchanged completion contradicts original publication")
    pin = prepared._pin(evidence.get("accepted_pair_marker"))
    body = _get(pin, attachments) if attachments is not None else prepared._capture(pin)
    pair = prepared._json(body)
    from src.data.io_utils import _validate_manifest, _csv_record, _parse_csv_bytes
    pins = _raw_pins(document, publication)
    records = []
    _require(isinstance(pair.get("artifacts"), list) and len(pair["artifacts"]) == 2,
             "unchanged accepted pair marker missing")
    for row in pair["artifacts"]:
        path = prepared._path(row.get("path"))
        name = path.relative_to(prepared._path(document["raw_root"])).as_posix()
        _require(name in pins, "unchanged pair artifact is outside accepted raw inventory")
        captured = _get(pins[name], attachments) if attachments is not None else prepared._capture(pins[name])
        records.append(_csv_record(path, captured, _parse_csv_bytes(path, captured)))
    _validate_manifest(Path(pin["path"]), body, records)


def _recovered_publication(evidence, publication, attachments=None):
    _require(evidence.get("recovered_from") == {"action": "accepted_pointer_already_addresses_cycle"}
             and evidence.get("publication_performed") is True and evidence.get("republished") is False,
             "unsupported or unsuccessful original publication recovery evidence")
    get = (lambda pin: _get(pin, attachments)) if attachments is not None else prepared._capture
    processing = prepared._json(get(prepared._pin(evidence.get("processing"))))
    _require(type(processing.get("schema")) is int and processing["schema"] == 1
             and processing.get("kind") == "roster_refresh_processing_v1"
             and processing.get("cycle_id") == evidence.get("cycle_id") == publication.get("bridge_cycle_id")
             and processing.get("staged_processed_root") == publication.get("processed_dir")
             and processing.get("candidate_inventory") == publication.get("refresh_input_inventory"),
             "recovered publication does not bind original processing generation")
    fingerprints = {field: publication.get(field) for field in (
        "processed_fights_sha256", "processed_fights_bytes", "processed_features_sha256", "processed_features_bytes")}
    _require(processing.get("staged_processed_identity") == fingerprints,
             "recovered publication processed identity mismatch")


def _validate_publication(document, attachments, *, metadata_only=False):
    manifest = prepared._json(_get(document["publication"], attachments))
    publication = manifest.get("accepted_data_publication") if isinstance(manifest, dict) else None
    _require(isinstance(publication, dict) and type(publication.get("schema_version")) is int
             and publication["schema_version"] in {1, 2}, "unsupported monthly publication schema")
    context = document["context"]
    _require(publication.get("accepted_generation_id") == context["accepted_update_id"],
             "monthly publication generation conflict")
    _require(_time(publication.get("accepted_at_utc")) <= _time(context["accepted_at_utc"]),
             "publication time follows qualifying completion")
    directory = str(Path(document["features"]["path"]).parent)
    _require(publication.get("processed_dir") == manifest.get("processed_dir") == directory
             and Path(document["processed_fights"]["path"]).parent == Path(directory)
             and Path(document["features"]["path"]).name == "features.csv"
             and Path(document["processed_fights"]["path"]).name == "fights_cleaned.csv",
             "monthly publication processed paths conflict")
    artifacts = {}
    for key, field, filename in (("features", "processed_features", "features.csv"),
                                 ("processed_fights", "processed_fights", "fights_cleaned.csv")):
        if metadata_only:
            # Independently accepted original metadata is enough to reconcile
            # data completion; missing feature bytes remain candidate failure.
            identity = {field: document[key][field] for field in ("bytes", "sha256")}
        else:
            body = _get(document[key], attachments)
            identity = {"bytes": len(body), "sha256": _digest(body)}
        artifacts[filename] = identity
        for name, actual in identity.items():
            _require(manifest.get(f"{field}_{name}") == publication.get(f"{field}_{name}") == actual,
                     "monthly processed artifact differs from publication")
    _require(accepted_data_generation_from_artifacts(artifacts)["accepted_generation_id"]
             == context["accepted_update_id"], "monthly processed generation identity invalid")
    report = publication.get("update_report")
    _require(isinstance(report, dict) and type(report.get("schema_version")) is int and report["schema_version"] == 1
             and manifest.get("update_report") == report, "monthly accepted update report conflict")
    _require((publication.get("baseline_lineage") == APPROVED_V7_BASELINE_LINEAGE
              and type(publication["baseline_lineage"].get("schema_version")) is int)
             or (publication["schema_version"] == 1 and "baseline_lineage" not in publication),
             "monthly accepted baseline lineage conflict")
    pins = _raw_pins(document, publication)
    if not metadata_only:
        for pin in pins.values():
            _get(pin, attachments)
    raw_generation = _accepted_raw_generation_identity(publication["refresh_input_inventory"])["accepted_raw_generation_id"]
    recorded = publication.get("accepted_raw_generation_id")
    _require(recorded == raw_generation or (publication["schema_version"] == 1 and not recorded),
             "monthly accepted raw generation identity invalid")
    completion = prepared._json(_get(document["completion"], attachments))
    _require(isinstance(completion, dict), "monthly completion evidence must be an object")
    kind = completion.get("kind")
    if kind == "roster_refresh_cycle_completion_v1":
        _require(type(completion.get("schema")) is int and completion["schema"] == 1
                 and completion.get("outcome") in {"published", "unchanged"}
                 and completion.get("claim_id", "") == context["claim_id"], "unsuccessful roster completion")
        _require(completion.get("qualifying_cycle_time_utc") == context["accepted_at_utc"],
                 "roster completion cutoff conflict")
        stage = "publication" if completion["outcome"] == "published" else "unchanged"
        evidence = prepared._json(_get(completion["stages"][stage], attachments))
        _require(completion.get("accepted_generation_id") == context["accepted_update_id"]
                 and completion.get("accepted_at_utc") == publication["accepted_at_utc"]
                 and completion.get("cycle_id") == evidence.get("cycle_id")
                 and completion.get("outcome") == evidence.get("outcome")
                 and completion.get("publication_performed") is (completion["outcome"] == "published"),
                 "completed cycle contradicts original publication stage")
    else:
        evidence = completion
    if evidence.get("kind") in {"roster_refresh_publication_v1", "roster_refresh_unchanged_success_v1"}:
        action = evidence.get("outcome")
        _require(type(evidence.get("schema")) is int and evidence["schema"] == 1
                 and evidence.get("kind") == {"published": "roster_refresh_publication_v1",
                                             "unchanged": "roster_refresh_unchanged_success_v1"}.get(action)
                 and evidence.get("accepted_manifest") == document["publication"]
                 and evidence.get("accepted_generation_id") == context["accepted_update_id"]
                 and evidence.get("accepted_raw_generation_id") == recorded
                 and evidence.get("accepted_at_utc") == publication["accepted_at_utc"],
                 "roster publication evidence contradicts accepted original")
        cutoff = evidence["accepted_at_utc"] if action == "published" else completion.get(
            "qualifying_cycle_time_utc", evidence.get("completed_at_utc"))
        _require(cutoff == context["accepted_at_utc"], "roster qualifying cutoff conflict")
        _quality(evidence, attachments)
        _identity_closure(evidence, publication, attachments)
        if action == "published":
            if "transaction" in evidence:
                _transaction(evidence["transaction"], document, publication, attachments)
            else:
                _recovered_publication(evidence, publication, attachments)
        else:
            _unchanged(evidence, publication, document, None if metadata_only else attachments)
        calendar = evidence.get("calendar_claim")
    else:
        _require(completion.get("continuity_green") is True and completion.get("published") is True
                 and (completion.get("publication_quality") or {}).get("passed") is True,
                 "scheduled publication completion failed")
        quality = completion["publication_quality"]
        _require(quality.get("errors") == [] and isinstance(quality.get("checks"), dict)
                 and quality["checks"] and all(isinstance(check, dict) and check.get("passed") is True
                                               for check in quality["checks"].values()),
                 "scheduled publication quality evidence contradicts success")
        evidence = completion.get("data_publication_receipt")
        _require(isinstance(evidence, dict) and evidence.get("accepted_generation_id") == context["accepted_update_id"]
                 and evidence.get("accepted_at_utc") == publication["accepted_at_utc"],
                 "scheduled publication generation or original timestamp conflict")
        _require(evidence.get("refresh_input_inventory") == publication["refresh_input_inventory"]
                 and evidence.get("accepted_raw_generation_id") == recorded,
                 "scheduled publication raw inventory conflict")
        _require(evidence.get("schema_version") == "ufc_accepted_data_generation_v1"
                 and evidence.get("artifacts") == artifacts
                 and evidence.get("processed_dir") == publication["processed_dir"]
                 and evidence.get("update_report") == publication["update_report"]
                 and evidence.get("baseline_lineage") == publication.get("baseline_lineage"),
                 "scheduled publication processed or lineage evidence conflict")
        _require(evidence.get("publication_action") == "committed"
                 and completion.get("publication_action") == "committed", "unsupported scheduled completion action")
        _transaction(evidence.get("raw"), document, publication, attachments)
        _require(context["accepted_at_utc"] == evidence["accepted_at_utc"], "scheduled cutoff conflict")
        calendar = evidence.get("calendar_claim")
    if context["claim_id"]:
        _require(isinstance(calendar, dict) and type(calendar.get("schema_version")) is int,
                 "monthly calendar schema must be an integer version")
        claim = _validated_calendar_claim(calendar)
        _require(claim is not None and claim["claim_id"] == context["claim_id"]
                 and claim["iso_week"] == context["iso_week"] and claim["slot"] == context["data_slot"],
                 "monthly calendar identity conflict")
        if evidence.get("outcome") != "unchanged":
            _require(publication.get("calendar_claim") == claim, "publication calendar evidence conflict")
    return publication


def _view(selection, spec, context, policy=None):
    arguments = {"feature_columns": spec.feature_cols}
    if policy is not None:
        arguments["policy"] = policy
    view, summary = build_monthly_modeling_view(selection.dataframe, **arguments)
    # Existing date-only event interpretation: UTC midnight, inclusive cutoff.
    cutoff = pd.Timestamp(_time(context["accepted_at_utc"]))
    dates = pd.to_datetime(view["event_date"], utc=True)
    _require(not dates.gt(cutoff).any(), "monthly modeling view contains rows after qualifying completion")
    result = preflight_training_data(view, spec)
    report = {"temporal_contract": policy.binding["adapter"] if policy else TEMPORAL_CONTRACT,
              "timezone": "UTC", "cutoff_inclusive": True,
              "event_date_interpretation": "date_only_utc_midnight",
              "qualifying_completion_cutoff": context["accepted_at_utc"],
              "view": summary, "shared_preflight": result.report()}
    if policy is not None:
        report["policy_binding"] = policy.binding
    return view, report


@dataclass(frozen=True)
class LoadedMonthlyInput:
    selection: prepared.PreparedInputSelection
    modeling_dataframe: pd.DataFrame
    spec: object
    publication: dict
    receipt: dict
    transport: dict
    policy: object = None


def _transport(root, document, acceptance_pin, receipt_body):
    result = {"schema": 1, "kind": TRANSPORT_KIND, "root": str(root), "context": document["context"],
            "acceptance_sha256": acceptance_pin["sha256"], "training_spec": document["training_spec"],
            "inventory_sha256": _digest(prepared._path(root / "complete.json").read_bytes()),
            "receipt_sha256": _digest(receipt_body), **{f"{name}_sha256": document[name]["sha256"]
                for name in ("publication", "completion", "features", "prepared_acceptance")}}
    if document["schema"] == 2:
        from src.model.monthly_policy import load_monthly_policy, validate_policy_context
        receipt = prepared._json(receipt_body)
        policy = validate_policy_context(load_monthly_policy(document["policy"],
            body=_attached(receipt["policy"]), context=document["context"]), document["context"])
        _require(receipt["policy"]["pin"] == document["policy"], "saved policy pin conflict")
        result.update(schema=2, kind=TRANSPORT_KIND_V2, policy_binding=policy.binding)
    return result


def inspect_monthly_completion(request):
    """Recover original completed data work without requiring candidate preflight.

    This checks independently pinned successful producer evidence. It does not
    authorize monthly dispatch, which requires the complete save and reload.
    """
    _require(isinstance(request, dict) and set(request) == {"acceptance", "output_root"},
             "missing independently accepted monthly recovery request")
    body = prepared._capture(prepared._pin(request["acceptance"]))
    accepted = prepared._json(body)
    context = _context(accepted.get("context") if isinstance(accepted, dict) else None)
    document, _ = _document(body, context, resolve_spec=False)
    attachments = {}
    for name in ("publication", "completion"):
        _collect(document[name], attachments)
    for attachment in list(attachments.values()):
        if attachment["pin"]["path"].lower().endswith(".json"):
            _collect_transaction_children(prepared._json(_attached(attachment)), attachments)
    _validate_publication(document, attachments, metadata_only=True)
    return context


def prepare_monthly_input(request, *, context):
    """Verify and save before schedule registration; complete duplicates only."""
    _require(isinstance(request, dict) and set(request) == {"acceptance", "output_root"},
             "monthly input requires independent acceptance pin and explicit output root")
    acceptance_pin = prepared._pin(request["acceptance"])
    root = prepared._path(request["output_root"])
    if root.exists():
        receipt_path = prepared._path(root / "monthly-receipt.json")
        _require(receipt_path.is_file() and (root / "complete.json").is_file(),
                 "monthly input root incomplete; preserve it and its original intent")
        body = receipt_path.read_bytes()
        receipt = prepared._json(body)
        _require(receipt.get("acceptance", {}).get("pin") == acceptance_pin,
                 "monthly saved original acceptance path or pin conflict")
        acceptance_body = _attached(receipt["acceptance"])
        prepared._check_bytes(acceptance_body, acceptance_pin)
        original_document = prepared._json(acceptance_body)
        saved_policy_body = None
        if original_document.get("schema") == 2:
            # Completed retry is a saved-only path even when a receipt is damaged.
            policy_path = prepared._path(root / "policy.json")
            _require(policy_path.is_file(), "saved original monthly policy missing")
            saved_policy_body = policy_path.read_bytes()
        document, _ = _document(acceptance_body, context,
            policy_body=saved_policy_body)
        transport = _transport(root, document, acceptance_pin, body)
        load_monthly_input(transport, context=context)
        return transport
    acceptance_body = prepared._capture(acceptance_pin)
    document, spec = _document(acceptance_body, context)
    policy = None
    if document["schema"] == 2:
        from src.model.monthly_policy import load_monthly_policy, validate_policy_context
        policy = validate_policy_context(load_monthly_policy(document["policy"], context=context), context)
    selection = prepared.select_prepared_input(document["features"]["path"],
        expected_sha256=document["features"]["sha256"], acceptance_path=document["prepared_acceptance"]["path"],
        expected_acceptance_sha256=document["prepared_acceptance"]["sha256"], spec=spec)
    _require(selection.document["artifact"] == document["features"], "monthly and prepared artifact pins conflict")
    attachments = {}
    for name in ("publication", "completion", "features", "prepared_acceptance", "processed_fights"):
        _collect(document[name], attachments)
    # Completion and nested stage documents may carry publication transactions.
    for attachment in list(attachments.values()):
        if attachment["pin"]["path"].lower().endswith(".json"):
            _collect_transaction_children(prepared._json(_attached(attachment)), attachments)
    manifest = prepared._json(_get(document["publication"], attachments))
    for pin in _raw_pins(document, manifest.get("accepted_data_publication") or {}).values():
        _collect(pin, attachments)
    publication = _validate_publication(document, attachments)
    view, report = _view(selection, spec, context, policy)
    protected = [prepared._path(acceptance_pin["path"]), prepared._path(document["raw_root"]),
                 *prepared._paths(attachments), *prepared._paths(document)]
    prepared.validate_prepared_output_root(selection, root, protected_roots=protected)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    # The intent exists before any copied input. A failure consumes this root.
    intent = {"request": request, "context": context}
    if policy is not None:
        intent.update(policy=prepared._attach(policy.body, policy.pin), policy_binding=policy.binding)
    prepared._write_new(root / "intent.json", _json_bytes(intent))
    prepared.save_prepared_input(selection, root / "prepared", preflight_report=report,
                                 protected_roots=protected)
    receipt = {"schema": 1, "kind": "saved_monthly_input_v1", "context": context,
               "acceptance": prepared._attach(acceptance_body, acceptance_pin), "attachments": attachments,
               "report": report, "pruning_authorized": False, "model_execution_performed": False}
    if policy is not None:
        receipt.update(schema=2, kind="saved_monthly_input_v2", policy=prepared._attach(policy.body, policy.pin),
                       policy_binding=policy.binding)
    files = {"monthly-receipt.json": _json_bytes(receipt), "modeling_dataset.csv": _dataframe_csv_bytes(view)}
    if policy is not None:
        files["policy.json"] = policy.body
    for name, body in files.items():
        prepared._write_new(root / name, body)
    inventory = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            body = prepared._path(path).read_bytes()
            inventory[path.relative_to(root).as_posix()] = {"bytes": len(body), "sha256": _digest(body)}
    prepared._fsync_directory(root)
    version = document["schema"]
    prepared._write_new(root / "complete.json", _json_bytes({"schema": version,
        "kind": f"complete_monthly_input_v{version}", "files": inventory}))
    prepared._fsync_directory(root)
    transport = _transport(root, document, acceptance_pin, files["monthly-receipt.json"])
    load_monthly_input(transport, context=context)
    return transport


def load_monthly_input(transport, *, context):
    """Validate all immutable claim pins against saved originals without active reads."""
    _require(isinstance(transport, dict) and type(transport.get("schema")) is int
             and transport["schema"] in {1, 2}
             and transport.get("kind") == {1: TRANSPORT_KIND, 2: TRANSPORT_KIND_V2}[transport["schema"]],
             "monthly claim has no strict saved-input transport")
    version = transport["schema"]
    _require(transport.get("context") == _context(context), "monthly claim context differs from saved selection")
    root = prepared._path(transport.get("root"))
    complete = prepared._json(prepared._path(root / "complete.json").read_bytes())
    _require(isinstance(complete, dict) and set(complete) == {"schema", "kind", "files"}
             and type(complete["schema"]) is int and complete["schema"] == version
             and complete["kind"] == f"complete_monthly_input_v{version}"
             and isinstance(complete["files"], dict), "invalid monthly saved completion")
    names = set(complete["files"])
    required = {"intent.json", "monthly-receipt.json", "modeling_dataset.csv", "prepared/complete.json",
                "prepared/input.csv", "prepared/selection.json", "prepared/preflight.json"}
    if version == 2:
        required.add("policy.json")
    allowed = required | {"prepared/input.csv" + module.SUFFIX for module in prepared._REFERENCE_MODULES.values()}
    _require(required.issubset(names) and names.issubset(allowed), "incomplete monthly saved inventory")
    actual_names = set()
    for path in root.rglob("*"):
        prepared._path(path)
        if path.is_file():
            actual_names.add(path.relative_to(root).as_posix())
    _require(actual_names == names | {"complete.json"}, "unexpected or missing monthly saved files")
    bodies = {name: prepared._capture({"path": str(root / name), **identity})
              for name, identity in complete["files"].items()}
    receipt_body = bodies["monthly-receipt.json"]
    _require(_digest(receipt_body) == transport.get("receipt_sha256"), "monthly receipt outer pin conflict")
    receipt = prepared._json(receipt_body)
    _require(type(receipt.get("schema")) is int and receipt["schema"] == version
             and receipt.get("kind") == f"saved_monthly_input_v{version}"
             and receipt.get("context") == context and receipt.get("pruning_authorized") is False
             and receipt.get("model_execution_performed") is False, "invalid saved monthly receipt")
    acceptance_body = _attached(receipt["acceptance"])
    document, spec = _document(acceptance_body, context, policy_body=bodies.get("policy.json"))
    _require(document["schema"] == version, "saved monthly input version conflict")
    policy = None
    if version == 2:
        from src.model.monthly_policy import load_monthly_policy, validate_policy_context
        policy = validate_policy_context(load_monthly_policy(document["policy"], body=bodies["policy.json"], context=context), context)
        _require(receipt.get("policy_binding") == policy.binding
                 and receipt.get("policy", {}).get("pin") == document["policy"]
                 and _attached(receipt["policy"]) == bodies["policy.json"], "saved monthly policy originals conflict")
    _require(_transport(root, document, receipt["acceptance"]["pin"], receipt_body) == transport,
             "monthly original outer acceptance/publication/input/spec pins conflict")
    intent = prepared._json(bodies["intent.json"])
    expected_intent = {"request": {"acceptance": receipt["acceptance"]["pin"], "output_root": str(root)},
                       "context": context}
    if policy is not None:
        expected_intent.update(policy=receipt["policy"], policy_binding=policy.binding)
    _require(intent == expected_intent, "monthly original first-selection intent conflict")
    publication = _validate_publication(document, receipt["attachments"])
    selection = prepared.load_saved_prepared_input(root / "prepared", expected_sha256=document["features"]["sha256"],
        expected_acceptance_sha256=document["prepared_acceptance"]["sha256"], spec=spec)
    view, report = _view(selection, spec, context, policy)
    _require(report == receipt["report"]
             and prepared._json(bodies["prepared/preflight.json"])["checks"] == report
             and _dataframe_csv_bytes(view) == bodies["modeling_dataset.csv"],
             "monthly modeling view or real shared preflight changed")
    return LoadedMonthlyInput(selection, view, spec, publication, receipt, dict(transport), policy)
