"""Select exact, independently accepted feature bytes for offline preparation.

The accepting caller supplies both outer hashes; this module cannot approve a
new table or upgrade legacy evidence. Saved selections preserve direct inputs
only, and confer no cleanup, training or production authority.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data import observation_consumer_references as observation_refs
from src.data import provider_consumer_references as provider_refs
from src.data.method_observation_history import _attached, _digest, _json_bytes, _safe_path
from src.model.training_preflight import resolved_spec_identity

ACCEPTANCE_KIND = "accepted_prepared_feature_input_v1"
SELECTION_KIND = "saved_prepared_feature_input_v1"
REQUIRED_COLUMNS = frozenset({
    "event_date", "fighter_a", "fighter_b", "target", "a_num_fights", "b_num_fights",
})
_REFERENCE_MODULES = {"provider": provider_refs, "observation": observation_refs}


class PreparedInputRefusal(ValueError):
    """The requested accepted input or isolated output cannot be established."""


def _require(condition, message):
    if not condition:
        raise PreparedInputRefusal(message)


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
             "an explicit lowercase SHA-256 pin is required")
    return value


def _path(value):
    _require(isinstance(value, (str, Path)) and str(value), "missing prepared input path")
    _require(".." not in Path(value).parts, "parent traversal in prepared input path")
    return _safe_path(Path(value))


def _pin(value):
    _require(isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"},
             "invalid prepared input pin")
    _require(isinstance(value["path"], str) and Path(value["path"]).is_absolute(),
             "prepared input pin path must be absolute")
    _require(".." not in Path(value["path"]).parts, "parent traversal in prepared input pin")
    _require(type(value["bytes"]) is int and value["bytes"] >= 0,
             "invalid prepared input byte length")
    _sha(value["sha256"])
    return value


def _check_bytes(body, pin):
    _pin(pin)
    _require(len(body) == pin["bytes"] and _digest(body) == pin["sha256"],
             "prepared input pin mismatch")
    return body


def _capture(pin):
    # Exactly one read; all later validation/parsing uses these immutable bytes.
    _pin(pin)
    return _check_bytes(_path(pin["path"]).read_bytes(), pin)


def _attach(body, pin):
    _check_bytes(body, pin)
    return {"pin": dict(pin), "body_base64": base64.b64encode(body).decode("ascii")}


def _json(body):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON field in prepared input evidence")
            result[key] = value
        return result
    return json.loads(body, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          PreparedInputRefusal("nonfinite JSON in prepared input evidence")))


def _acceptance(body, *, expected_sha256, expected_acceptance_sha256, spec):
    _sha(expected_sha256)
    _sha(expected_acceptance_sha256)
    _require(not spec.add_line_movement,
             "prepared inputs do not support external line-movement materialization")
    _require(_digest(body) == expected_acceptance_sha256, "accepted input manifest hash mismatch")
    document = _json(body)
    _require(isinstance(document, dict) and set(document) == {
        "schema", "kind", "input_id", "artifact", "training_spec", "references",
    }, "incomplete accepted prepared input manifest")
    _require(type(document["schema"]) is int and document["schema"] == 1
             and document["kind"] == ACCEPTANCE_KIND
             and isinstance(document["input_id"], str) and document["input_id"].strip(),
             "unknown prepared input acceptance identity")
    artifact = _pin(document["artifact"])
    _require(artifact["sha256"] == expected_sha256, "accepted table differs from independent input hash")
    _require(document["training_spec"] == {"name": spec.name, "sha256": resolved_spec_identity(spec)},
             "accepted input training spec identity mismatch")
    references = document["references"]
    _require(isinstance(references, dict) and set(references) == set(_REFERENCE_MODULES),
             "provider and method/rankings evidence policies must both be explicit")
    for name, module in _REFERENCE_MODULES.items():
        descriptor = references[name]
        if descriptor == {"status": "legacy_absent"}:
            continue
        _require(isinstance(descriptor, dict) and set(descriptor) == {"status", "pin"}
                 and descriptor["status"] == "required", "unknown prepared reference policy")
        pin = _pin(descriptor["pin"])
        _require(Path(pin["path"]) == module.sidecar_path(artifact["path"]),
                 "accepted reference is not the selected artifact's direct sidecar")
    return document


def _load_frame(body):
    try:
        text = body.decode("utf-8-sig")
        rows = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(rows)
        _require(header and all(column and column.strip() == column for column in header),
                 "prepared CSV has empty or ambiguous columns")
        _require(len(header) == len(set(header)), "prepared CSV has duplicate columns")
        for row in rows:
            _require(len(row) == len(header), "prepared CSV row width differs from columns")
        _require(REQUIRED_COLUMNS.issubset(header),
                 "prepared CSV is missing required training columns")
        # Identity text is not numeric data: retain leading zeroes and literal
        # labels such as "NA". A blank identity remains unknown, not a default.
        identity_columns = {column for column in header if column.endswith("_id")}
        identity_columns.update({"fighter_a", "fighter_b", "winner", "event", "event_name"} & set(header))
        converters = {column: lambda value: value if value != "" else float("nan")
                      for column in identity_columns}
        frame = pd.read_csv(io.BytesIO(body), encoding="utf-8-sig", low_memory=False,
                            converters=converters, float_precision="round_trip")
        _require(frame.columns.tolist() == header, "prepared CSV column identity changed while parsing")
        # Date parsing is the only loader conversion beyond ordinary CSV types.
        # Preserve the supplied row order; the shared chronological split owns sorting.
        dates = pd.to_datetime(frame["event_date"], errors="coerce", format="mixed")
        _require(not dates.isna().any(), "prepared CSV contains invalid event_date values")
        frame["event_date"] = dates
        return frame
    except (UnicodeError, StopIteration, csv.Error, pd.errors.ParserError) as exc:
        raise PreparedInputRefusal(f"prepared CSV cannot be parsed: {exc}") from exc


def _validate_references(document, body, frame, attachments):
    _require(isinstance(attachments, dict) and set(attachments) == set(_REFERENCE_MODULES),
             "incomplete saved prepared references")
    for name, module in _REFERENCE_MODULES.items():
        descriptor, attachment = document["references"][name], attachments[name]
        if descriptor["status"] == "legacy_absent":
            _require(attachment is None and f"{name}_reference_schema" not in frame.columns,
                     "marked prepared input cannot use legacy absence")
            continue
        _require(isinstance(attachment, dict) and attachment.get("pin") == descriptor["pin"],
                 "saved reference differs from accepted original")
        captured = _attached(attachment)
        # Invoke the original schema validators on the selected bytes, without
        # reopening the active sidecar or any original response location.
        reference = module._validate(_json(captured))
        if name == "provider":
            module._objects(reference)
        _require(_attached(reference["artifact"]) == body,
                 "prepared table differs from retained consumer artifact")
        _require(reference["artifact"]["pin"] == document["artifact"],
                 "prepared consumer reference artifact identity differs")


@dataclass(frozen=True)
class PreparedInputSelection:
    dataframe: pd.DataFrame
    artifact_bytes: bytes
    acceptance_bytes: bytes
    acceptance_pin: dict
    references: dict
    document: dict
    saved_root: Path | None = None

    @property
    def input_id(self):
        return self.document["input_id"]

    @property
    def reference_status(self):
        return {name: value["status"] for name, value in self.document["references"].items()}


def select_prepared_input(path, *, expected_sha256, acceptance_path,
                          expected_acceptance_sha256, spec):
    """Freeze one accepted table and its exact direct evidence, then parse once."""
    _sha(expected_sha256)
    _sha(expected_acceptance_sha256)
    _require(not spec.add_line_movement,
             "prepared inputs do not support external line-movement materialization")
    acceptance_path, path = _path(acceptance_path), _path(path)
    acceptance_body = acceptance_path.read_bytes()
    document = _acceptance(acceptance_body, expected_sha256=expected_sha256,
        expected_acceptance_sha256=expected_acceptance_sha256, spec=spec)
    _require(path == _path(document["artifact"]["path"]), "selected path differs from accepted input identity")
    body = _capture(document["artifact"])
    attachments = {}
    for name, module in _REFERENCE_MODULES.items():
        descriptor = document["references"][name]
        if descriptor["status"] == "legacy_absent":
            sidecar = _path(module.sidecar_path(path))
            _require(not sidecar.exists() and not _path(str(sidecar) + ".pending").exists(),
                     "existing or incomplete evidence cannot be relabelled legacy absent")
            # A missing marker does not erase concrete evidence retained beside
            # this selected artifact. Match the existing consumer-reader fence;
            # this is only its local manifest store, never an archive census.
            store = _path(path.parent / module.STORE)
            if store.exists():
                _require(store.is_dir(), "malformed local consumer manifest store")
                for saved in store.iterdir():
                    _require(saved.is_file(), "malformed local consumer manifest entry")
                    retained = module._validate(_json(_path(saved).read_bytes()))
                    _require(Path(retained["artifact"]["pin"]["path"]).name != path.name,
                             "retained evidence cannot be relabelled legacy absent")
            attachments[name] = None
        else:
            _require(not _path(descriptor["pin"]["path"] + ".pending").exists(),
                     "incomplete selected evidence write")
            attachments[name] = _attach(_capture(descriptor["pin"]), descriptor["pin"])
    frame = _load_frame(body)
    _validate_references(document, body, frame, attachments)
    acceptance_pin = {"path": str(acceptance_path), "bytes": len(acceptance_body),
                      "sha256": expected_acceptance_sha256}
    return PreparedInputSelection(frame, body, acceptance_body, acceptance_pin,
                                  attachments, document)


def _paths(value):
    """Only concrete original dependencies; this is not a directory census."""
    if isinstance(value, dict):
        if set(value) == {"pin", "body_base64"}:
            body = _attached(value)
            # Consumer manifests retain nested original attachments. Decode
            # only the already captured JSON, never scan a source directory.
            if body.lstrip().startswith((b"{", b"[")):
                try:
                    nested = _json(body)
                except (ValueError, UnicodeError):
                    nested = None
                if nested is not None:
                    yield from _paths(nested)
        if set(value) == {"path", "bytes", "sha256"}:
            yield _path(value["path"])
        for child in value.values():
            yield from _paths(child)
    elif isinstance(value, list):
        for child in value:
            yield from _paths(child)


def _overlaps(left, right):
    return left == right or left in right.parents or right in left.parents


def validate_prepared_output_root(selection, output_root, *, protected_roots=()):
    """Require a never-used directory disjoint from the selected source scope."""
    root = _path(output_root)
    _require(not root.exists(), "prepared output root must be fresh; preserve existing or incomplete roots")
    protected = [_path(value) for value in protected_roots]
    originals = [*_paths(selection.document), *_paths(selection.references),
                 _path(selection.acceptance_pin["path"])]
    protected.extend(originals)
    protected.extend(value.parent for value in originals)
    if selection.saved_root is not None:
        protected.append(_path(selection.saved_root))
    _require(not any(_overlaps(root, value) for value in protected),
             "prepared output root overlaps selected input, evidence or protected runtime")
    return root


def _write_new(path, body):
    path = _path(path)
    with path.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(root):
    if os.name != "nt":
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def save_prepared_input(selection, output_root, *, preflight_report, protected_roots=()):
    """Save exact originals and a direct preflight record in a fresh output root.

    Failure preserves every partial original and consumes this output root.
    Explicit saved-input resume requires a complete record and original outer pins.
    """
    root = validate_prepared_output_root(selection, output_root, protected_roots=protected_roots)
    # Revalidate frozen fields before writes; no active path is consulted.
    document = _json(selection.acceptance_bytes)
    _require(document == selection.document, "frozen acceptance document changed")
    _check_bytes(selection.acceptance_bytes, selection.acceptance_pin)
    _check_bytes(selection.artifact_bytes, document["artifact"])
    frame = _load_frame(selection.artifact_bytes)
    _require(selection.dataframe.equals(frame), "selected dataframe differs from frozen artifact bytes")
    _validate_references(document, selection.artifact_bytes, frame, selection.references)
    _require(isinstance(preflight_report, dict), "missing explicit preflight report")
    record = {"schema": 1, "kind": SELECTION_KIND, "input_id": selection.input_id,
              "acceptance": _attach(selection.acceptance_bytes, selection.acceptance_pin),
              "artifact": document["artifact"], "references": selection.references,
              "reference_coverage": "selected_prepared_input_direct_dependencies_only",
              "pruning_authorized": False}
    report = {"schema": 1, "kind": "prepared_feature_preflight_record_v1",
              "input_id": selection.input_id, "acceptance_sha256": selection.acceptance_pin["sha256"],
              "artifact_sha256": document["artifact"]["sha256"],
              "training_spec": document["training_spec"], "reference_status": selection.reference_status,
              "checks": preflight_report, "model_execution_performed": False,
              "reference_coverage": record["reference_coverage"], "pruning_authorized": False}
    files = {"input.csv": selection.artifact_bytes, "selection.json": _json_bytes(record),
             "preflight.json": _json_bytes(report)}
    for name, module in _REFERENCE_MODULES.items():
        if selection.references[name] is not None:
            files["input.csv" + module.SUFFIX] = _attached(selection.references[name])
    completion = {"schema": 1, "kind": "complete_prepared_input_save_v1",
                  "files": {name: {"bytes": len(body), "sha256": _digest(body)}
                            for name, body in files.items()}}
    # Parent directories may be established, but the selected leaf is exclusive.
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    for name, body in files.items():
        _write_new(root / name, body)
    _fsync_directory(root)
    _write_new(root / "complete.json", _json_bytes(completion))
    _fsync_directory(root)
    return root


def load_saved_prepared_input(root, *, expected_sha256, expected_acceptance_sha256, spec):
    """Resume only saved originals, with the independently accepted hashes again."""
    _sha(expected_sha256)
    _sha(expected_acceptance_sha256)
    root = _path(root)
    complete_path = _path(root / "complete.json")
    _require(complete_path.is_file(), "saved prepared input is incomplete; original root must be preserved")
    complete = _json(complete_path.read_bytes())
    _require(isinstance(complete, dict) and set(complete) == {"schema", "kind", "files"}
             and type(complete["schema"]) is int and complete["schema"] == 1
             and complete["kind"] == "complete_prepared_input_save_v1"
             and isinstance(complete["files"], dict), "invalid prepared input completion record")
    required = {"input.csv", "selection.json", "preflight.json"}
    allowed = required | {"input.csv" + module.SUFFIX for module in _REFERENCE_MODULES.values()}
    _require(required.issubset(complete["files"]) and set(complete["files"]).issubset(allowed),
             "incomplete or unknown saved input files")
    bodies = {}
    for name, pin in complete["files"].items():
        _require(isinstance(pin, dict) and set(pin) == {"bytes", "sha256"}, "invalid saved file pin")
        bodies[name] = _capture({"path": str(root / name), **pin})
    record = _json(bodies["selection.json"])
    _require(isinstance(record, dict) and set(record) == {
        "schema", "kind", "input_id", "acceptance", "artifact", "references",
        "reference_coverage", "pruning_authorized",
    } and type(record["schema"]) is int and record["schema"] == 1
        and record["kind"] == SELECTION_KIND
        and record["reference_coverage"] == "selected_prepared_input_direct_dependencies_only"
        and record["pruning_authorized"] is False, "invalid saved input selection")
    acceptance_body = _attached(record["acceptance"])
    document = _acceptance(acceptance_body, expected_sha256=expected_sha256,
        expected_acceptance_sha256=expected_acceptance_sha256, spec=spec)
    _require(record["input_id"] == document["input_id"] and record["artifact"] == document["artifact"],
             "saved input identity differs from original acceptance")
    body = _check_bytes(bodies["input.csv"], document["artifact"])
    frame = _load_frame(body)
    _validate_references(document, body, frame, record["references"])
    for name, module in _REFERENCE_MODULES.items():
        filename = "input.csv" + module.SUFFIX
        attachment = record["references"][name]
        if attachment is None:
            _require(filename not in bodies and not _path(root / filename).exists(),
                     "saved legacy input has unexpected evidence")
        else:
            _require(filename in bodies and bodies[filename] == _attached(attachment),
                     "saved direct reference differs from original selection")
        _require(not _path(root / (filename + ".pending")).exists(), "saved reference write is incomplete")
    report = _json(bodies["preflight.json"])
    _require(isinstance(report, dict) and set(report) == {
        "schema", "kind", "input_id", "acceptance_sha256", "artifact_sha256", "training_spec",
        "reference_status", "checks", "model_execution_performed", "reference_coverage", "pruning_authorized",
    } and type(report["schema"]) is int and report["schema"] == 1
             and report["kind"] == "prepared_feature_preflight_record_v1"
             and isinstance(report["checks"], dict)
             and report["reference_coverage"] == record["reference_coverage"]
             and report["reference_status"] == {name: value["status"]
                                                for name, value in document["references"].items()}
             and report.get("acceptance_sha256") == expected_acceptance_sha256
             and report.get("artifact_sha256") == expected_sha256
             and report.get("training_spec") == document["training_spec"]
             and report.get("input_id") == document["input_id"]
             and report.get("model_execution_performed") is False
             and report.get("pruning_authorized") is False,
             "saved preflight record differs from original acceptance")
    return PreparedInputSelection(frame, body, acceptance_body, record["acceptance"]["pin"],
                                  record["references"], document, root)
