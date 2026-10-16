"""Exact, portable references for the bounded provider consumers.

These manifests retain accepted input bytes, not new acceptance decisions. Their
outer pins belong to the accepting caller. No manifest here authorizes pruning;
training/evaluation/monthly and general uncertainty owners remain unenumerated.
"""
from __future__ import annotations

import json
import os
import csv
import io
from pathlib import Path

from src.data.method_observation_history import (
    _attached, _attachment, _json_bytes, _safe_path, _write_immutable,
)
from src.data.provider_observation_history import ProviderObservationError, _digest

SUFFIX = ".provider-references.json"
STORE = "provider_consumer_manifests_v1"
KIND = "provider_consumer_references_v1"


def _require(value, message):
    if not value:
        raise ProviderObservationError(message)


def _pin(path):
    path = _safe_path(Path(path))
    body = path.read_bytes()
    return {"path": str(path), "bytes": len(body), "sha256": _digest(body)}


def sidecar_path(path):
    return Path(str(path) + SUFFIX)


def _marked(path):
    if not path.is_file() or path.suffix not in {".csv", ".json"}:
        return False
    body = path.read_bytes()
    if b"provider_reference_schema" not in body:
        return False
    if path.suffix == ".csv":
        return "provider_reference_schema" in next(csv.reader(io.StringIO(body.decode("utf-8"))), [])
    document = json.loads(body)
    return isinstance(document, dict) and "provider_reference_schema" in document


def _validate(document):
    _require(isinstance(document, dict) and set(document) == {
        "schema", "kind", "consumer_kind", "artifact", "authority", "objects", "parents",
        "reference_coverage", "pruning_authorized",
    }, "incomplete provider consumer reference manifest")
    _require(type(document["schema"]) is int and document["schema"] == 1 and document["kind"] == KIND and
             isinstance(document["consumer_kind"], str) and bool(document["consumer_kind"].strip()) and
             document["reference_coverage"] == "direct_provider_consumers_only" and
             document["pruning_authorized"] is False, "unknown provider reference policy")
    _attached(document["artifact"])
    objects = document["objects"]
    _require(isinstance(objects, dict), "missing provider object references")
    for path, attachment in objects.items():
        _attached(attachment)
        _require(path == attachment["pin"]["path"], "provider reference path differs")
    authority = document["authority"]
    parents = document["parents"]
    _require(isinstance(parents, list), "invalid provider parent references")
    for parent in parents:
        _validate(json.loads(_attached(parent)))
    if authority is None:
        _require(parents, "provider references lack independently supplied authority")
    else:
        _require(isinstance(authority, dict) and set(authority) == {"membership", "membership_bytes", "decisions"},
                 "incomplete provider authority references")
        membership = authority["membership"]
        _require(isinstance(membership, dict) and set(membership) == {"path", "sha256", "generation"},
                 "missing original accepted membership pin")
        body = _attached(authority["membership_bytes"])
        _require(_digest(body) == membership["sha256"] and isinstance(membership["generation"], str) and membership["generation"] and
                 authority["membership_bytes"]["pin"]["path"] == membership["path"],
                 "accepted membership reference differs from original pin")
        from src.data.provider_membership import load_accepted_rosterwatch_membership_bytes
        load_accepted_rosterwatch_membership_bytes(body, expected_sha256=membership["sha256"],
                                                 expected_generation=membership["generation"])
        _require(isinstance(authority["decisions"], list) and authority["decisions"],
                 "missing independent binding/field-decision references")
        for decision in authority["decisions"]:
            _attached(decision)
    return document


def _objects(document):
    result = dict(document["objects"])
    for parent in document["parents"]:
        inherited = _objects(json.loads(_attached(parent)))
        for key, value in inherited.items():
            _require(key not in result or result[key] == value, "conflicting inherited provider reference")
            result[key] = value
    return result


def _dependency_identity(document):
    """Compare complete inherited inputs without growing duplicate ancestry."""
    atoms = set()
    def visit(value):
        atoms.add(_digest(_json_bytes(value["artifact"])))
        if value["authority"] is not None:
            atoms.add(_digest(_json_bytes(value["authority"])))
        for item in value["objects"].values():
            atoms.add(_digest(_json_bytes(item)))
        for parent in value["parents"]:
            visit(json.loads(_attached(parent)))
    visit(document)
    return atoms


def read_provider_reference(path, *, references=None):
    """Read an original path through a caller-verified portable manifest."""
    key = str(path)
    if references is not None:
        objects = _objects(_validate(references))
        _require(key in objects, "original provider evidence missing from consumer references")
        return _attached(objects[key])
    return _safe_path(Path(path)).read_bytes()


def verify_provider_consumer_artifact(path, *, required=False):
    """Verify an artifact against its supplied sidecar; absence never permits pruning."""
    path = _safe_path(Path(path))
    sidecar = sidecar_path(path)
    if not sidecar.exists():
        required = required or _marked(path)
        store = path.parent / STORE
        if store.exists():
            for saved in store.iterdir():
                document = _validate(json.loads(_safe_path(saved).read_bytes()))
                if Path(document["artifact"]["pin"]["path"]).name == path.name:
                    required = True
        _require(not required, "provider consumer reference manifest is missing")
        return None
    try:
        document = _validate(json.loads(_safe_path(sidecar).read_bytes()))
        _objects(document)
        _require(path.read_bytes() == _attached(document["artifact"]),
                 "provider consumer artifact differs from its retained reference")
        return document
    except ProviderObservationError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ProviderObservationError(f"provider consumer reference refused: {exc}") from exc


def record_provider_consumer_artifact(path, *, evidence_paths=(), accepted_membership=None,
                                     decision_paths=(), history_root, kind, parent_artifacts=(), parent_references=(),
                                     capture_descriptors=(), expected_decision_pins=()):
    """Retain an exact artifact and its authority/evidence before reporting success.

The artifact's directory owns a copyable immutable manifest store. history_root
is an explicit caller scope, not permission to scan arbitrary directories.
Failures retain the already written artifact and all immutable originals.
"""
    from src.data.provider_membership import load_accepted_rosterwatch_membership

    _safe_path(Path(history_root))
    path = _safe_path(Path(path))
    objects, parents = {}, list(parent_references)
    for parent in parent_artifacts:
        snapshot = snapshot_provider_references(parent)
        _require(snapshot is not None, "missing provider parent manifest")
        parents.append(snapshot)
    inherited_objects = {}
    for parent in parents:
        for key, value in _objects(_validate(json.loads(_attached(parent)))).items():
            _require(key not in inherited_objects or inherited_objects[key] == value,
                     "conflicting inherited provider input")
            inherited_objects[key] = value
    authority = None
    if accepted_membership is not None:
        load_accepted_rosterwatch_membership(Path(accepted_membership["path"]),
            expected_sha256=accepted_membership["sha256"], expected_generation=accepted_membership["generation"])
        membership = dict(accepted_membership)
        membership["path"] = str(_safe_path(Path(membership["path"])))
        decisions = [_attachment(_pin(p)) for p in decision_paths]
        if expected_decision_pins:
            _require([value["pin"] for value in decisions] == list(expected_decision_pins),
                     "independent binding/field-decision inputs changed during processing")
        authority = {"membership": membership, "membership_bytes": _attachment(_pin(membership["path"])),
                     "decisions": decisions}

    def collect(original):
        key = str(_safe_path(Path(original)))
        if key in objects:
            return
        attachment = inherited_objects.get(key)
        if attachment is None:
            attachment = _attachment(_pin(key))
        objects[key] = attachment
        document = json.loads(_attached(attachment))
        if document.get("kind") == "provider_history_attachment_v1":
            row = document["row"]
            collect(row["source_evidence_path"])
        elif document.get("kind") == "provider_observations_v1":
            # Observation documents already retain exact body/envelope bytes.
            _attached(document["response"])
            _attached(document["envelope"])
        else:
            raise ProviderObservationError("unknown provider observation/attachment reference")

    for evidence in evidence_paths:
        collect(evidence)
    for capture in capture_descriptors:
        _require(isinstance(capture, dict) and set(capture) == {"schema", "source", "raw_response", "response_envelope"}
                 and type(capture["schema"]) is int and capture["schema"] == 1,
                 "invalid failed acquisition reference")
        for key in ("raw_response", "response_envelope"):
            value = _attachment(capture[key])
            objects[value["pin"]["path"]] = value
    document = {"schema": 1, "kind": KIND, "consumer_kind": str(kind),
                "artifact": _attachment(_pin(path)), "authority": authority, "objects": objects,
                "parents": parents, "reference_coverage": "direct_provider_consumers_only",
                "pruning_authorized": False}
    _validate(document)
    previous_path = sidecar_path(path)
    if previous_path.is_file():
        previous = _validate(json.loads(previous_path.read_bytes()))
        if (previous["artifact"] == document["artifact"] and
                previous["consumer_kind"] == document["consumer_kind"] and
                _dependency_identity(previous) == _dependency_identity(document)):
            verify_provider_consumer_artifact(path, required=True)
            return previous_path
    body = _json_bytes(document)
    immutable = path.parent / STORE / f"{_digest(body)}.json"
    _write_immutable(immutable, body)
    # This marker is last: a crash before replacement leaves a missing or stale
    # sidecar, which readers/publication refuse rather than using partial refs.
    destination = sidecar_path(path)
    temporary = destination.with_name(destination.name + ".pending")
    try:
        with temporary.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        if os.name != "nt":
            fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    except OSError as exc:
        raise ProviderObservationError(f"provider reference write incomplete: {exc}") from exc
    verify_provider_consumer_artifact(path, required=True)
    return destination


def provider_reference_artifacts(root):
    """Verify and inventory complete sidecars and immutable older manifests."""
    root = _safe_path(Path(root))
    result = set()
    if not root.exists():
        return []
    for pending in root.rglob("*" + SUFFIX + ".pending"):
        raise ProviderObservationError(f"incomplete provider reference write: {pending}")
    for path in root.rglob("*"):
        if _marked(path):
            verify_provider_consumer_artifact(path, required=True)
    for sidecar in root.rglob("*" + SUFFIX):
        artifact = Path(str(sidecar)[:-len(SUFFIX)])
        verify_provider_consumer_artifact(artifact, required=True)
        result.add(sidecar.relative_to(root).as_posix())
        result.add(artifact.relative_to(root).as_posix())
    for store in root.rglob(STORE):
        _safe_path(store)
        for path in store.iterdir():
            body = _safe_path(path).read_bytes()
            _require(path.name == _digest(body) + ".json", "corrupt immutable provider references")
            _validate(json.loads(body))
            result.add(path.relative_to(root).as_posix())
    return sorted(result)


def inherit_provider_references(artifact, raw_root):
    """Connect a real rebuilt output to exact staged raw consumer manifests."""
    parents = []
    for relative in provider_reference_artifacts(raw_root):
        if relative.endswith(SUFFIX):
            parents.append(Path(raw_root) / relative[:-len(SUFFIX)])
    if parents:
        return record_provider_consumer_artifact(artifact, parent_artifacts=parents,
            history_root=Path(artifact).parent, kind="rebuilt_artifact")
    return None


def snapshot_provider_references(artifact):
    document = verify_provider_consumer_artifact(artifact)
    if document is not None:
        snapshot = _attachment(_pin(sidecar_path(artifact)))
        _require(json.loads(_attached(snapshot)) == document and
                 Path(artifact).read_bytes() == _attached(document["artifact"]),
                 "provider parent changed while pinning references")
        return snapshot
    return None


def provider_reference_inventory(root):
    return {relative: {key: value for key, value in _pin(Path(root) / relative).items() if key != "path"}
            for relative in provider_reference_artifacts(root)}


def provider_pruning_decision(root):
    """An explicit honest fence, including unknown or malformed reference state."""
    try:
        artifacts = provider_reference_artifacts(root)
        reason = "full reference-owner inventory remains incomplete"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        artifacts, reason = [], f"unverified references: {exc}"
    return {"pruning_authorized": False, "preserve_files": True,
            "verified_artifacts": artifacts, "reason": reason,
            "uncovered_owners": ["all accepted generations", "training", "evaluation", "monthly", "general recovery", "uncertainty"]}
