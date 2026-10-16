"""Portable exact references for direct method/rankings consumers.

These records retain caller-selected originals; they do not grant acceptance or
global deletion authority. Different versions of one path remain independent.
Provider membership/field authority keeps its separate existing schema.
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path

from src.data.method_observation_history import (
    ObservationRefusal, _attached, _attachment, _digest, _json_bytes, _safe_path,
    _write_immutable,
)

SUFFIX = ".observation-references.json"
STORE = "observation_consumer_manifests_v1"
KIND = "method_rankings_consumer_references_v1"


def _require(value, message):
    if not value:
        raise ObservationRefusal(message)


def _pin(path):
    path = _safe_path(Path(path))
    body = path.read_bytes()
    return {"path": str(path), "bytes": len(body), "sha256": _digest(body)}


def sidecar_path(path):
    return Path(str(path) + SUFFIX)


def _key(attachment):
    return _digest(_json_bytes(attachment["pin"]))


def _marked(path):
    _safe_path(path)
    if not path.is_file() or path.suffix not in {".csv", ".json"}:
        return False
    body = path.read_bytes()
    if b"observation_reference_schema" not in body:
        return False
    if path.suffix == ".csv":
        return "observation_reference_schema" in next(csv.reader(io.StringIO(body.decode("utf-8"))), [])
    value = json.loads(body)
    return isinstance(value, dict) and "observation_reference_schema" in value


def _validate(document):
    _require(isinstance(document, dict) and set(document) == {
        "schema", "kind", "consumer_kind", "artifact", "inputs",
        "reference_coverage", "pruning_authorized",
    }, "incomplete observation consumer references")
    _require(type(document["schema"]) is int and document["schema"] == 1
             and document["kind"] == KIND and isinstance(document["consumer_kind"], str)
             and bool(document["consumer_kind"].strip())
             and document["reference_coverage"] == "direct_method_rankings_consumers_only"
             and document["pruning_authorized"] is False, "unknown observation reference policy")
    _attached(document["artifact"])
    _require(isinstance(document["inputs"], dict) and document["inputs"], "missing original observation inputs")
    for key, value in document["inputs"].items():
        _attached(value)
        _require(key == _key(value), "observation reference identity differs")
    return document


def verify_observation_consumer_artifact(path, *, required=False):
    path = _safe_path(Path(path))
    sidecar = sidecar_path(path)
    if not sidecar.exists():
        required = required or _marked(path)
        store = path.parent / STORE
        if store.exists():
            for saved in store.glob("*.json"):
                document = _validate(json.loads(_safe_path(saved).read_bytes()))
                if Path(document["artifact"]["pin"]["path"]).name == path.name:
                    required = True
        _require(not required, "observation consumer reference manifest is missing")
        return None
    document = _validate(json.loads(_safe_path(sidecar).read_bytes()))
    _require(path.read_bytes() == _attached(document["artifact"]),
             "observation consumer artifact differs from its original reference")
    return document


def snapshot_observation_references(path):
    document = verify_observation_consumer_artifact(path)
    if document is None:
        return None
    snapshot = _attachment(_pin(sidecar_path(path)))
    _require(json.loads(_attached(snapshot)) == document
             and Path(path).read_bytes() == _attached(document["artifact"]),
             "observation references changed during selection")
    return snapshot


def record_observation_consumer_artifact(path, *, kind, evidence_paths=(),
                                         parent_artifacts=(), parent_references=(), evidence_references=()):
    """Save exact inputs and immutable prior versions, then replace the sidecar.

    Flatten original inputs across generations to reuse unchanged references.
    Original paths and independently selected authority bytes are never rewritten.
    A failed write preserves originals and fails closed; it is not a source retry.
    """
    path = _safe_path(Path(path))
    inputs = {}
    parents = list(parent_references)
    for parent in parent_artifacts:
        snapshot = snapshot_observation_references(parent)
        _require(snapshot is not None, "missing observation parent references")
        parents.append(snapshot)
    for parent in parents:
        document = _validate(json.loads(_attached(parent)))
        inputs.update(document["inputs"])
        value = document["artifact"]
        inputs[_key(value)] = value
    for evidence in evidence_paths:
        value = _attachment(_pin(evidence))
        inputs[_key(value)] = value
    for value in evidence_references:
        _attached(value)
        inputs[_key(value)] = value
    document = {"schema": 1, "kind": KIND, "consumer_kind": kind,
                "artifact": _attachment(_pin(path)), "inputs": inputs,
                "reference_coverage": "direct_method_rankings_consumers_only",
                "pruning_authorized": False}
    _validate(document)
    body = _json_bytes(document)
    _write_immutable(path.parent / STORE / (_digest(body) + ".json"), body)
    destination = sidecar_path(path)
    if destination.exists() and destination.read_bytes() == body:
        verify_observation_consumer_artifact(path, required=True)
        return destination
    temporary = destination.with_name(destination.name + ".pending")
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
    verify_observation_consumer_artifact(path, required=True)
    return destination


def read_observation_reference(pin, *, references):
    """Resolve a specific original version, including after its path disappears."""
    document = _validate(references)
    for value in [document["artifact"], *document["inputs"].values()]:
        if value["pin"] == pin:
            return _attached(value)
    raise ObservationRefusal("exact original observation is absent from consumer references")


def observation_reference_artifacts(root):
    root = _safe_path(Path(root))
    if not root.exists():
        return []
    result = set()
    for pending in root.rglob("*" + SUFFIX + ".pending"):
        raise ObservationRefusal(f"incomplete observation reference write: {pending}")
    for path in root.rglob("*"):
        if _marked(path):
            verify_observation_consumer_artifact(path, required=True)
    for sidecar in root.rglob("*" + SUFFIX):
        artifact = Path(str(sidecar)[:-len(SUFFIX)])
        verify_observation_consumer_artifact(artifact, required=True)
        result.update((artifact.relative_to(root).as_posix(), sidecar.relative_to(root).as_posix()))
        # Publication recovery can leave preparation originals beside an owned
        # snapshot/output. They remain evidence, never a substitute for a valid
        # canonical artifact or sidecar and never deletion authority.
        for pattern in ("*.rollback.bak", "*.publish.tmp"):
            for retained in sidecar.parent.glob(pattern):
                _safe_path(retained)
                _require(retained.is_file(), "non-file publication preparation evidence")
                result.add(retained.relative_to(root).as_posix())
    for store in root.rglob(STORE):
        _safe_path(store)
        for path in store.iterdir():
            body = _safe_path(path).read_bytes()
            if path.suffix == ".json":
                _require(path.name == _digest(body) + ".json", "corrupt immutable observation references")
                _validate(json.loads(body))
            # Noncanonical preparation/uncertainty originals are carried by
            # exact byte pin; they cannot grant positive consumer authority.
            result.add(path.relative_to(root).as_posix())
    # Final histories are self-contained original evidence. Include actual files
    # even if the temporary collector snapshot is no longer in this generation.
    from src.data.method_observation_history import HISTORY_SUBDIR as METHOD_HISTORY, validate_finalized_document as validate_method
    from src.data.rankings_observation_history import HISTORY_SUBDIR as RANK_HISTORY, validate_finalized_document as validate_rankings
    for name, validate in ((METHOD_HISTORY, validate_method), (RANK_HISTORY, validate_rankings)):
        for directory in root.rglob(name):
            _safe_path(directory)
            for path in directory.iterdir():
                _safe_path(path)
                _require(path.is_file(), "non-file finalized observation member")
                if path.suffix != ".json":
                    result.add(path.relative_to(root).as_posix())
                    continue
                if path.name.endswith(SUFFIX):
                    continue
                document = json.loads(_safe_path(path).read_bytes())
                validate(document)
                _require(document.get("snapshot_sha256") == path.stem,
                         "finalized observation filename differs from original snapshot")
                result.add(path.relative_to(root).as_posix())
    # Attempts, including incomplete captures and downstream failures, are owned
    # by these bounded source directories. Preserve them verbatim, without
    # claiming that incomplete transport context can supply a feature.
    for name in ("method_response_captures_v1", "rankings_response_captures_v1",
                 "method_collection_attempts_v1", "rankings_collection_attempts_v1"):
        for directory in root.rglob(name):
            _safe_path(directory)
            for path in directory.rglob("*"):
                _safe_path(path)
                if path.is_file():
                    result.add(path.relative_to(root).as_posix())
    return sorted(result)


def observation_reference_inventory(root):
    return {relative: {key: value for key, value in _pin(Path(root) / relative).items() if key != "path"}
            for relative in observation_reference_artifacts(root)}


def freeze_observation_inputs(root):
    inventory = observation_reference_inventory(root)
    parents = []
    for relative, expected in inventory.items():
        if relative.endswith(SUFFIX):
            snapshot = snapshot_observation_references(Path(root) / relative[:-len(SUFFIX)])
            _require(snapshot is not None and {key: snapshot["pin"][key] for key in ("bytes", "sha256")} == expected,
                     "observation references changed while selecting rebuild inputs")
            parents.append(snapshot)
    originals = []
    for relative, expected in inventory.items():
        if relative.endswith(SUFFIX) or (STORE in Path(relative).parts and Path(relative).suffix == ".json"):
            continue
        original = _attachment(_pin(Path(root) / relative))
        _require({key: original["pin"][key] for key in ("bytes", "sha256")} == expected,
                 "observation original changed while selecting rebuild inputs")
        originals.append(original)
    _require(observation_reference_inventory(root) == inventory,
             "observation inputs changed while freezing original bytes")
    return inventory, parents, originals


def observation_pruning_decision(root):
    try:
        artifacts = observation_reference_artifacts(root)
        reason = "complete accepted/training/evaluation/monthly/recovery/uncertainty ownership remains unproved"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        artifacts, reason = [], f"unverified observation references: {exc}"
    return {"pruning_authorized": False, "preserve_files": True,
            "verified_artifacts": artifacts, "reason": reason}
