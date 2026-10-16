"""One preserved Roster Watch collection and exact same-identity replay.

The caller supplies a reviewed identity registry. This module publishes only
rankings observations; it does not update membership, models or schedules.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
from pathlib import Path

from src.data.method_observation_history import (
    ObservationRefusal, _attached, _attachment, _digest, _json_bytes, _read_pin,
    _require, _safe_path, _time, _write_immutable,
)
from src.data import rosterwatch_rankings as provider

CAPTURE_SUBDIR = "rosterwatch_rankings_response_captures_v1"
HISTORY_KIND = "retained_rosterwatch_rankings_observations_v1"
READY_KIND = "rosterwatch_rankings_collection_ready_v1"


def _pin(path):
    path = _safe_path(Path(path))
    body = path.read_bytes()
    return {"path": str(path), "bytes": len(body), "sha256": _digest(body)}


def _preserve_page(page, capture_root):
    body = page["content"]
    _require(isinstance(body, bytes), "original Roster Watch page bytes required")
    envelope = {key: value for key, value in page.items() if key != "content"}
    envelope_body = _json_bytes(envelope)
    body_path = capture_root / "bodies" / (_digest(body) + ".body")
    envelope_path = capture_root / "envelopes" / (_digest(envelope_body) + ".json")
    _write_immutable(body_path, body)
    _write_immutable(envelope_path, envelope_body)
    return body, envelope, {"raw_response": _pin(body_path), "response_envelope": _pin(envelope_path)}


def _original_observations(snapshot_body, evidence):
    snapshot = json.loads(snapshot_body)
    _require(set(evidence) == {"registry", "response", "envelope"}, "incomplete Roster Watch evidence")
    registry_body = _attached(evidence["registry"])
    registry = provider.validate_registry(json.loads(registry_body))
    _require(registry_body == _json_bytes(registry), "noncanonical reviewed Roster Watch registry")
    body, envelope_body = _attached(evidence["response"]), _attached(evidence["envelope"])
    envelope = json.loads(envelope_body)
    _require(envelope_body == _json_bytes(envelope), "noncanonical preserved Roster Watch receipt")
    _require(snapshot.get("schema_version") == 2 and snapshot.get("observation_reference_schema") == 1
             and snapshot.get("source") == "roster.watch" and snapshot.get("status") == "success"
             and snapshot.get("acquisition_failed") is False
             and snapshot.get("collection_id") == registry["collection_id"]
             and snapshot.get("reviewed_context") == evidence["registry"]["pin"]
             and snapshot.get("response_capture") == {"raw_response": evidence["response"]["pin"],
                                                       "response_envelope": evidence["envelope"]["pin"]},
             "Roster Watch snapshot provenance differs")
    _require(snapshot.get("captured_at") == envelope["retrieved_at_utc"]
             and _time(snapshot["snapshot_time"]) >= _time(snapshot["captured_at"]),
             "Roster Watch original receipt differs")
    rows = provider.observations_from_capture(body, envelope, registry)
    _require(snapshot.get("rankings_observations") == rows, "Roster Watch values differ from original source")
    return rows


def validate_finalized_document(document):
    _require(isinstance(document, dict) and set(document) == {
        "schema", "kind", "availability_policy", "snapshot_sha256", "snapshot_body_base64", "evidence", "observations"
    } and type(document["schema"]) is int and document["schema"] == 1
        and document["kind"] == HISTORY_KIND and document["availability_policy"] == provider.POLICY,
        "unsupported retained Roster Watch history")
    try:
        body = base64.b64decode(document["snapshot_body_base64"], validate=True)
    except (TypeError, ValueError) as exc:
        raise ObservationRefusal("invalid retained Roster Watch snapshot") from exc
    _require(_digest(body) == document["snapshot_sha256"], "corrupt retained Roster Watch snapshot")
    rows = _original_observations(body, document["evidence"])
    _require(rows == document["observations"], "retained Roster Watch observations differ")
    return rows


def _save(snapshot, snapshot_root, finalized_root):
    from src.data.observation_consumer_references import record_observation_consumer_artifact

    body = _json_bytes(snapshot)
    path = _safe_path(Path(snapshot_root)) / ("rosterwatch_rankings_" + _digest(body) + ".json")
    _write_immutable(path, body)
    evidence_paths = [Path(snapshot["reviewed_context"]["path"]), Path(snapshot["collection_intent"]["path"])]
    capture = snapshot.get("response_capture")
    if capture:
        evidence_paths.extend(Path(capture[key]["path"]) for key in ("raw_response", "response_envelope"))
    if snapshot["status"] == "success":
        evidence = {"registry": _attachment(snapshot["reviewed_context"]),
                    "response": _attachment(capture["raw_response"]),
                    "envelope": _attachment(capture["response_envelope"])}
        rows = _original_observations(body, evidence)
        document = {"schema": 1, "kind": HISTORY_KIND, "availability_policy": provider.POLICY,
                    "snapshot_sha256": _digest(body), "snapshot_body_base64": base64.b64encode(body).decode("ascii"),
                    "evidence": evidence, "observations": rows}
        final = _safe_path(Path(finalized_root)) / (_digest(body) + ".json")
        _write_immutable(final, _json_bytes(document))
        evidence_paths.append(final)
    record_observation_consumer_artifact(path, kind="rosterwatch_rankings_snapshot", evidence_paths=evidence_paths)
    return {**snapshot, "snapshot_path": str(path)}


def _load_ready(context, registry, collection):
    original = _read_pin(context)
    intent = {"schema": 1, "reviewed_context": context, "context_sha256": _digest(original),
              "registry_original": _attachment(context)}
    intent_path, ready_path = collection / "intent.json", collection / "ready.json"
    _require(intent_path.exists() and ready_path.exists(), "Roster Watch collection incomplete; reacquisition refused")
    _require(_safe_path(intent_path).read_bytes() == _json_bytes(intent), "Roster Watch collection identity changed")
    ready = json.loads(_safe_path(ready_path).read_bytes())
    _require(set(ready) == {"schema", "kind", "snapshot"} and type(ready["schema"]) is int
             and ready["schema"] == 1 and ready["kind"] == READY_KIND, "invalid Roster Watch ready checkpoint")
    snapshot = ready["snapshot"]
    _require(snapshot.get("collection_id") == registry["collection_id"]
             and snapshot.get("reviewed_context") == context and snapshot.get("collection_intent") == _pin(intent_path),
             "Roster Watch ready checkpoint binding differs")
    return snapshot


def inspect_replay(context, *, snapshot_root, finalized_root):
    """Read-only admission for complete success; no missing finalization repair."""
    registry = provider.load_reviewed_registry(context)
    collection = _safe_path(Path(snapshot_root) / CAPTURE_SUBDIR / "collections" / _digest(registry["collection_id"].encode()))
    snapshot = _load_ready(context, registry, collection)
    _require(snapshot.get("status") == "success" and snapshot.get("acquisition_failed") is False,
             "Roster Watch collection has no verified successful checkpoint")
    for value in snapshot["response_capture"].values():
        _read_pin(value)
    body = _json_bytes(snapshot)
    snapshot_path = _safe_path(Path(snapshot_root) / ("rosterwatch_rankings_" + _digest(body) + ".json"))
    _require(snapshot_path.read_bytes() == body, "Roster Watch saved snapshot missing or changed")
    from src.data.observation_consumer_references import verify_observation_consumer_artifact
    verify_observation_consumer_artifact(snapshot_path, required=True)
    document = json.loads(_safe_path(Path(finalized_root) / (_digest(body) + ".json")).read_bytes())
    rows = validate_finalized_document(document)
    _require(document["snapshot_sha256"] == _digest(body), "Roster Watch retained checkpoint differs")
    _require(_time(snapshot["captured_at"]) <= datetime.now(timezone.utc), "Roster Watch receipt is future-dated")
    return {"collection_id": registry["collection_id"], "captured_at": snapshot["captured_at"],
            "snapshot_sha256": _digest(body), "observation_count": len(rows), "mode": "validated_same_identity_replay"}


def collect_rankings_snapshot(*, reviewed_context, snapshot_root, finalized_root, captured_page=None, session=None):
    """Reuse a shared roster response if supplied; otherwise one ordinary GET.

    A ready checkpoint never sends another GET. An interrupted claim refuses.
    Supplied pages retain their actual receipt; the caller owns acquisition.
    """
    registry = provider.load_reviewed_registry(reviewed_context)
    root = _safe_path(Path(snapshot_root))
    capture_root = root / CAPTURE_SUBDIR
    collection = _safe_path(capture_root / "collections" / _digest(registry["collection_id"].encode()))
    if collection.exists():
        _require(captured_page is None, "cannot replace a consumed Roster Watch response")
        snapshot = _load_ready(reviewed_context, registry, collection)
        return _save(snapshot, root, finalized_root)
    collection.parent.mkdir(parents=True, exist_ok=True)
    try:
        collection.mkdir()
    except FileExistsError as exc:
        raise ObservationRefusal("Roster Watch collection is concurrently claimed") from exc
    original = _read_pin(reviewed_context)
    intent = {"schema": 1, "reviewed_context": reviewed_context, "context_sha256": _digest(original),
              "registry_original": _attachment(reviewed_context)}
    intent_path = collection / "intent.json"
    _write_immutable(intent_path, _json_bytes(intent))
    snapshot = {"schema_version": 2, "observation_reference_schema": 1,
                "status": "failed", "source": "roster.watch", "acquisition_failed": True,
                "reviewed_context": reviewed_context, "collection_id": registry["collection_id"],
                "collection_intent": _pin(intent_path), "sources": [],
                "acquisition_mode": "collection_request" if captured_page is None else "shared_roster_response"}
    try:
        if captured_page is None:
            from src.data.rosterwatch_site import fetch_rosterwatch_site_page
            captured_page = fetch_rosterwatch_site_page(session=session)
        body, envelope, snapshot["response_capture"] = _preserve_page(captured_page, capture_root)
        rows = provider.observations_from_capture(body, envelope, registry)
        _require(_time(envelope["retrieved_at_utc"]) <= datetime.now(timezone.utc), "future Roster Watch receipt refused")
        snapshot.update(status="success", acquisition_failed=False, captured_at=envelope["retrieved_at_utc"],
                        rankings_observations=rows)
        snapshot["sources"].append({"source": "roster.watch", "status": "success", "captured_at": snapshot["captured_at"]})
    except Exception as exc:
        rejected_page = getattr(exc, "page", None)
        if snapshot.get("response_capture") is None and isinstance(rejected_page, dict):
            # Original rejected/redirect/challenge bytes are evidence only.
            _, _, snapshot["response_capture"] = _preserve_page(rejected_page, capture_root)
        snapshot.update(status="processing_refused" if snapshot.get("response_capture") else "failed",
                        error_type=type(exc).__name__,
                        source_response_unavailable=snapshot.get("response_capture") is None)
        snapshot["sources"].append({"source": "roster.watch", "status": "failed", "error_type": type(exc).__name__})
    snapshot["snapshot_time"] = datetime.now(timezone.utc).isoformat()
    _write_immutable(collection / "ready.json", _json_bytes({"schema": 1, "kind": READY_KIND, "snapshot": snapshot}))
    return _save(snapshot, root, finalized_root)
