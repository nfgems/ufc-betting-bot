"""Finalize explicitly bound saved BFO observations before temporary retention.

This first history slice accepts trusted caller-selected identity/cutoff and raw
response pins. It does not acquire evidence or upgrade legacy provenance. The
ordinary schema-1 collector lacks those pins, so its snapshots remain retained
until a trusted evidence producer supplies the explicit finalization context.
Old CSV history is separate and is never rewritten or relabelled here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from src.data.roster_refresh_files import publish_new

HISTORY_SUBDIR = "finalized_observations_v1"
POLICY = "method_prefight_exact_response_v1"
PARSER = "bestfightodds_exact_matchup_v1"
ORDINARY_PARSER = "bestfightodds_bound_matchup_v2"
ORDINARY_POLICY = "method_original_receipt_prefight_v2"
REFERENCE_ROLES = frozenset({"accepted", "training", "evaluation", "monthly", "recovery", "uncertainty"})
FEATURES = tuple(f"{side}_{method}_odds_prob" for side in ("a", "b") for method in ("ko", "sub", "dec"))


class ObservationRefusal(ValueError):
    """Evidence cannot support finalization, use, or deletion."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ObservationRefusal(message)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _time(value: object) -> datetime:
    _require(isinstance(value, str), "timestamp must be explicit text")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ObservationRefusal("unparseable timestamp") from exc
    _require(result.tzinfo is not None, "timestamp has no timezone")
    return result.astimezone(timezone.utc)


def _safe_path(path: Path) -> Path:
    path = Path(path).absolute()
    _require(not any(member.is_symlink() for member in (path, *path.parents)), "linked evidence/history path")
    return path


def _read_pin(pin: dict) -> bytes:
    _require(isinstance(pin, dict) and set(pin) == {"path", "bytes", "sha256"}, "invalid evidence pin")
    _require(isinstance(pin["path"], str) and Path(pin["path"]).is_absolute(), "evidence path must be absolute")
    body = _safe_path(Path(pin["path"])).read_bytes()
    _require(type(pin["bytes"]) is int and pin["bytes"] == len(body) and pin["sha256"] == _digest(body), "evidence pin mismatch")
    return body


def _attachment(pin: dict) -> dict:
    body = _read_pin(pin)
    return {"pin": pin, "body_base64": base64.b64encode(body).decode("ascii")}


def _attached(value: dict) -> bytes:
    _require(isinstance(value, dict) and set(value) == {"pin", "body_base64"}, "invalid saved evidence")
    try:
        body = base64.b64decode(value["body_base64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise ObservationRefusal("invalid saved response bytes") from exc
    pin = value["pin"]
    _require(isinstance(pin, dict) and set(pin) == {"path", "bytes", "sha256"}, "invalid saved evidence pin")
    _require(type(pin["bytes"]) is int and len(body) == pin["bytes"] and _digest(body) == pin["sha256"], "corrupt saved response")
    return body


def _observations(snapshot_body: bytes, evidence: list[dict]) -> list[dict]:
    from bs4 import BeautifulSoup
    from src.data.method_odds import _parse_bfo_method_odds

    snapshot = json.loads(snapshot_body)
    _require(isinstance(snapshot, dict) and type(snapshot.get("schema_version")) is int and snapshot["schema_version"] in (1, 2), "unknown snapshot schema")
    _require(snapshot.get("status") in {"success", "partial"}, "failed snapshot is not eligible")
    records = snapshot.get("records")
    _require(isinstance(records, list) and records and len(records) == len(evidence), "every observation needs exact evidence")
    snapshot_time = _time(snapshot.get("snapshot_time"))
    result, seen = [], set()
    for record, proof in zip(records, evidence):
        _require(isinstance(record, dict) and isinstance(proof, dict), "malformed observation")
        if set(proof) == {"binding", "responses", "reviewed_context"}:
            observation = _ordinary_observation(record, proof, snapshot_time)
            key = (observation["event_id"], *sorted((observation["fighter_a_id"], observation["fighter_b_id"])))
            _require(key not in seen, "ambiguous duplicate fight observation")
            seen.add(key)
            result.append(observation)
            continue
        _require(set(proof) == {"binding", "response", "envelope"}, "unknown observation proof")
        binding = proof["binding"]
        required = {"event_id", "event_date", "fighter_a", "fighter_b", "fighter_a_id", "fighter_b_id", "commence_time", "cutoff_at", "source_url", "parser_revision"}
        _require(isinstance(binding, dict) and set(binding) == required, "incomplete trusted identity/cutoff binding")
        _require(binding["parser_revision"] == PARSER, "unknown parser revision")
        for field in ("fighter_a_id", "fighter_b_id"):
            _require(isinstance(binding[field], str) and re.fullmatch(r"[0-9a-f]{16}", binding[field]), "unknown stable fighter identity")
        _require(binding["fighter_a_id"] != binding["fighter_b_id"], "conflicting fighter identity")
        _require(all(isinstance(binding[field], str) and binding[field].strip() for field in ("event_id", "fighter_a", "fighter_b")), "missing event/fighter identity")
        _require(record.get("source") == "bestfightodds", "unsupported source provenance")
        for field in ("event_id", "fighter_a", "fighter_b", "commence_time"):
            _require(record.get(field) == binding[field], f"observation identity differs: {field}")
        observed = _time(record.get("captured_at"))
        cutoff, commence = _time(binding["cutoff_at"]), _time(binding["commence_time"])
        _require(observed <= snapshot_time <= cutoff < commence, "late observation or invalid eligibility cutoff")
        _require(commence.date().isoformat() == binding["event_date"], "event date does not match exact commence time")
        body, envelope_body = _attached(proof["response"]), _attached(proof["envelope"])
        envelope = json.loads(envelope_body)
        _require(isinstance(envelope, dict) and envelope.get("schema") == 1 and envelope.get("status_code") == 200, "unproved successful response")
        url = binding["source_url"]
        parsed_url = urlparse(url)
        _require(parsed_url.scheme == "https" and parsed_url.hostname in {"bestfightodds.com", "www.bestfightodds.com"} and not parsed_url.username and not parsed_url.password, "unsupported response source")
        _require(envelope.get("requested_url") == url and envelope.get("response_sha256") == _digest(body) and envelope.get("response_bytes") == len(body), "response envelope does not bind exact body/source")
        _require(_time(envelope.get("received_at")) == observed, "original observation time is not response time")
        parsed = _parse_bfo_method_odds(BeautifulSoup(body, "html.parser"), binding["fighter_a"], binding["fighter_b"])
        _require(parsed is not None, "ambiguous or unusable raw method observation")
        values = {name: None if math.isnan(parsed[name]) else float(parsed[name]) for name in FEATURES}
        _require(record.get("method_odds") == values, "parsed values differ from exact retained response")
        key = (binding["event_id"], *sorted((binding["fighter_a_id"], binding["fighter_b_id"])))
        _require(key not in seen, "ambiguous duplicate fight observation")
        seen.add(key)
        result.append({**binding, "observed_at": record["captured_at"], "eligibility_policy": POLICY, "method_odds": values,
                       "response_sha256": _digest(body), "response_envelope_sha256": _digest(envelope_body)})
    return result


def validate_ordinary_binding(binding: dict) -> None:
    """Validate caller-reviewed ownership; discovery cannot construct this input."""
    required = {"event_id", "event_date", "fighter_a", "fighter_b", "fighter_a_id", "fighter_b_id",
                "commence_time", "cutoff_at", "source_urls", "fighter_a_source_url", "fighter_b_source_url",
                "parser_revision", "eligibility_policy"}
    _require(isinstance(binding, dict) and set(binding) == required, "incomplete reviewed method binding")
    _require(binding["parser_revision"] == ORDINARY_PARSER and binding["eligibility_policy"] == ORDINARY_POLICY,
             "unknown ordinary method parser/timing policy")
    for side in ("a", "b"):
        _require(isinstance(binding[f"fighter_{side}_id"], str) and re.fullmatch(r"[0-9a-f]{16}", binding[f"fighter_{side}_id"]), "unknown stable fighter identity")
        _require(isinstance(binding[f"fighter_{side}"], str) and binding[f"fighter_{side}"].strip(), "missing fighter display identity")
        _bfo_owned_url(binding[f"fighter_{side}_source_url"], "/fighters/")
    _require(binding["fighter_a_id"] != binding["fighter_b_id"] and binding["fighter_a_source_url"] != binding["fighter_b_source_url"], "conflicting fighter identity")
    _require(isinstance(binding["event_id"], str) and binding["event_id"].strip(), "missing event identity")
    _require(isinstance(binding["source_urls"], list) and binding["source_urls"] and len(set(binding["source_urls"])) == len(binding["source_urls"]), "missing/duplicate approved event pages")
    for url in binding["source_urls"]:
        _bfo_owned_url(url, "/events/")
    commence, cutoff = _time(binding["commence_time"]), _time(binding["cutoff_at"])
    _require(cutoff < commence and commence.date().isoformat() == binding["event_date"], "invalid method commencement/cutoff")


def _bfo_owned_url(url, prefix):
    _require(isinstance(url, str), "missing approved source URL")
    parsed = urlparse(url)
    _require(parsed.scheme == "https" and parsed.hostname in {"bestfightodds.com", "www.bestfightodds.com"}
             and not parsed.username and not parsed.password and parsed.path.startswith(prefix)
             and len(parsed.path) > len(prefix) and not parsed.query and not parsed.fragment,
             "unsupported approved source URL")
    return url


def parse_bound_response(body: bytes, envelope: dict, binding: dict) -> dict | None:
    """Reproduce one page only after exact page and athlete ownership checks."""
    from bs4 import BeautifulSoup
    from src.data.method_odds import _absolute_bfo_url, _bfo_exact_matchup_rows, _bfo_matchup_fighter_name, _bfo_names_match, _parse_bfo_method_odds

    validate_ordinary_binding(binding)
    url = envelope.get("requested_url")
    _require(envelope.get("schema") == 1 and envelope.get("status_code") == 200 and envelope.get("format") == "html", "unproved successful original response")
    _require(url in binding["source_urls"] and envelope.get("issued_url") == url and envelope.get("effective_url") == url and envelope.get("redirect_chain") == [], "wrong event page or unapproved redirect")
    _require(envelope.get("response_bytes") == len(body) and envelope.get("response_sha256") == _digest(body), "original response envelope differs")
    _time(envelope.get("received_at"))
    soup = BeautifulSoup(body, "html.parser")
    rows = _bfo_exact_matchup_rows(soup, binding["fighter_a"], binding["fighter_b"])
    _require(rows is not None and bool(rows), "wrong person or ambiguous source matchup")
    found = set()
    for row in rows:
        label = _bfo_matchup_fighter_name(row)
        for side in ("a", "b"):
            if not _bfo_names_match(binding[f"fighter_{side}"], label):
                continue
            links = {_absolute_bfo_url(str(link.get("href"))) for link in row.select('a[href*="/fighters/"]')}
            _require(links == {binding[f"fighter_{side}_source_url"]}, "wrong or missing source fighter identity")
            found.add(side)
    _require(found == {"a", "b"}, "source fighter ownership unavailable")
    parsed = _parse_bfo_method_odds(soup, binding["fighter_a"], binding["fighter_b"])
    if parsed is None:
        return None
    return {name: None if math.isnan(parsed[name]) else float(parsed[name]) for name in FEATURES}


def merge_bound_values(parts: list[dict]) -> dict:
    values = {name: None for name in FEATURES}
    for part in parts:
        for name in FEATURES:
            incoming = part[name]
            _require(values[name] is None or incoming is None or incoming == values[name], "conflicting original method markets")
            if incoming is not None:
                values[name] = incoming
    return values


def _ordinary_observation(record, proof, snapshot_time):
    binding = proof["binding"]
    validate_ordinary_binding(binding)
    context = validate_reviewed_context(json.loads(_attached(proof["reviewed_context"])))
    _require(binding in context["bindings"], "observation is outside independent reviewed binding context")
    _require(record.get("source") == "bestfightodds", "unsupported source provenance")
    for field in ("event_id", "fighter_a", "fighter_b", "fighter_a_id", "fighter_b_id", "commence_time"):
        _require(record.get(field) == binding[field], f"observation identity differs: {field}")
    responses = proof["responses"]
    _require(isinstance(responses, list) and responses, "missing contributing original responses")
    parts, received, response_pins, urls = [], [], [], []
    for response in responses:
        _require(isinstance(response, dict) and set(response) == {"response", "envelope"}, "invalid original response evidence")
        body, envelope_body = _attached(response["response"]), _attached(response["envelope"])
        envelope = json.loads(envelope_body)
        parsed = parse_bound_response(body, envelope, binding)
        _require(parsed is not None, "noncontributing response cannot finalize method values")
        parts.append(parsed)
        received.append(_time(envelope["received_at"]))
        urls.append(envelope["requested_url"])
        response_pins.append({"response_sha256": _digest(body), "response_envelope_sha256": _digest(envelope_body), "received_at": envelope["received_at"], "source_url": envelope["requested_url"]})
    _require(len(set(urls)) == len(urls), "duplicate contributing source page")
    observed = _time(record.get("captured_at"))
    _require(observed == max(received) <= snapshot_time <= _time(binding["cutoff_at"]) < _time(binding["commence_time"]), "late original receipt or invalid availability cutoff")
    values = merge_bound_values(parts)
    _require(record.get("method_odds") == values, "parsed values differ from contributing original responses")
    return {**binding, "observed_at": record["captured_at"], "method_odds": values, "responses": response_pins}


def validate_reviewed_context(context):
    _require(isinstance(context, dict) and set(context) == {"schema", "kind", "collection_id", "bindings"}
             and type(context["schema"]) is int and context["schema"] == 1
             and context["kind"] == "method_reviewed_bindings_v1", "invalid independently reviewed method context")
    _require(isinstance(context["collection_id"], str) and bool(context["collection_id"].strip()), "missing original method collection identity")
    _require(isinstance(context["bindings"], list) and bool(context["bindings"]), "missing independently reviewed method bindings")
    seen = set()
    for binding in context["bindings"]:
        validate_ordinary_binding(binding)
        key = (binding["event_id"], *sorted((binding["fighter_a_id"], binding["fighter_b_id"])))
        _require(key not in seen, "duplicate reviewed fight binding")
        seen.add(key)
    return context


def load_reviewed_bindings(pin):
    return validate_reviewed_context(json.loads(_read_pin(pin)))


def _sync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_immutable(path: Path, body: bytes) -> None:
    path = _safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _sync_directory(path.parent.parent)
    if path.exists():
        _require(path.read_bytes() == body, "conflicting finalization for snapshot")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".finalizing-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444) if hasattr(os, "fchmod") else None
            os.fsync(handle.fileno())
        try:
            publish_new(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == body, "conflicting concurrent finalization")
        _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def finalize_snapshot(path: Path, *, history_root: Path, proof: list[dict]) -> Path:
    """Persist original snapshot, raw responses, envelopes and trusted bindings together."""
    path = _safe_path(path)
    body = path.read_bytes()
    _require(isinstance(proof, list), "missing trusted snapshot proof")
    evidence = []
    for item in proof:
        if isinstance(item, dict) and set(item) == {"binding", "responses", "reviewed_context"}:
            _require(isinstance(item["responses"], list), "missing original response list")
            evidence.append({"binding": item["binding"], "reviewed_context": _attachment(item["reviewed_context"]), "responses": [
                {"response": _attachment(response["raw_response"]), "envelope": _attachment(response["response_envelope"])}
                for response in item["responses"]]})
            continue
        _require(isinstance(item, dict) and set(item) == {"binding", "raw_response", "response_envelope"}, "invalid trusted proof")
        evidence.append({"binding": item["binding"], "response": _attachment(item["raw_response"]), "envelope": _attachment(item["response_envelope"])})
    observations = _observations(body, evidence)
    document = {"schema": 1, "kind": HISTORY_SUBDIR, "eligibility_policy": POLICY, "snapshot_sha256": _digest(body),
                "snapshot_body_base64": base64.b64encode(body).decode("ascii"), "evidence": evidence, "observations": observations}
    snapshot = json.loads(body)
    if snapshot.get("observation_reference_schema") == 1:
        document["attempts"] = [{"response": _attachment(descriptor["raw_response"]),
                                 "envelope": _attachment(descriptor["response_envelope"])}
                                for source in snapshot.get("sources", []) for descriptor in source.get("captures", [])]
        _validate_saved_attempts(snapshot, document["attempts"], evidence)
    destination = Path(history_root) / f"{_digest(body)}.json"
    _write_immutable(destination, _json_bytes(document))
    _require(path.read_bytes() == body, "snapshot changed during finalization")
    return destination


def _validate_saved_attempts(snapshot, attempts, evidence):
    descriptors = [descriptor for source in snapshot.get("sources", []) for descriptor in source.get("captures", [])]
    _require(isinstance(attempts, list) and bool(attempts) and len(attempts) == len(descriptors), "missing complete original method attempts")
    for descriptor, attempt in zip(descriptors, attempts):
        _require(isinstance(attempt, dict) and set(attempt) == {"response", "envelope"}, "invalid saved method attempt")
        body, envelope_body = _attached(attempt["response"]), _attached(attempt["envelope"])
        envelope = json.loads(envelope_body)
        _require(descriptor == {"schema": 1, "source": "bestfightodds", "raw_response": attempt["response"]["pin"],
                                "response_envelope": attempt["envelope"]["pin"]}, "original method attempt descriptor differs")
        _require(envelope.get("response_bytes") == len(body) and envelope.get("response_sha256") == _digest(body), "corrupt original method attempt")
        _time(envelope.get("received_at"))
    for proof in evidence:
        if "responses" in proof:
            _require(all(response in attempts for response in proof["responses"]), "contributing method response omitted from original attempt inventory")


def _unreferenced(path: Path, references: object) -> bool:
    """The caller must supply a complete trusted inventory for every consumer role."""
    if not isinstance(references, dict) or set(references) != {"schema", "complete", "held_paths"} or type(references["schema"]) is not int or references["schema"] != 1:
        return False
    complete, held = references["complete"], references["held_paths"]
    if not isinstance(complete, dict) or not isinstance(held, dict) or set(complete) != REFERENCE_ROLES or set(held) != REFERENCE_ROLES:
        return False
    for role in REFERENCE_ROLES:
        if complete[role] is not True or not isinstance(held[role], list):
            return False
        for item in held[role]:
            if not isinstance(item, str) or not Path(item).is_absolute():
                return False
            selected_path = Path(item)
            if not selected_path.exists() or any(member.is_symlink() for member in (selected_path, *selected_path.parents)):
                return False
            selected = selected_path.resolve(strict=True)
            resolved = path.resolve(strict=False)
            if selected == resolved or selected in resolved.parents:
                return False
    return True


def finalize_before_prune(path: Path, context: object) -> bool:
    """Collector cleanup hook: absence/uncertainty preserves the temporary source."""
    if not isinstance(context, dict) or set(context) != {"schema", "history_root", "snapshot_proofs", "references"} or type(context["schema"]) is not int or context["schema"] != 1:
        return False
    if not _unreferenced(path, context["references"]):
        return False
    proofs = context["snapshot_proofs"]
    if not isinstance(proofs, dict):
        return False
    proof = proofs.get(_digest(path.read_bytes()))
    if proof is None:
        return False
    try:
        finalize_snapshot(path, history_root=Path(context["history_root"]), proof=proof)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ObservationRefusal("malformed finalization evidence") from exc
    return True


def read_finalized_observations(history_root: Path) -> list[dict]:
    """Read only intact observations whose original bytes still reproduce the values."""
    root = _safe_path(history_root)
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.glob("*.json")):
        _safe_path(path)
        rows.extend(validate_finalized_document(json.loads(path.read_bytes()), filename_digest=path.stem))
    return rows


def validate_finalized_document(document, filename_digest=None):
    _require(isinstance(document, dict) and type(document.get("schema")) is int and document.get("schema") == 1 and document.get("kind") == HISTORY_SUBDIR and document.get("eligibility_policy") == POLICY, "unknown finalized history schema")
    try:
        body = base64.b64decode(document["snapshot_body_base64"], validate=True)
    except (ValueError, TypeError, KeyError) as exc:
        raise ObservationRefusal("invalid retained snapshot") from exc
    _require(document.get("snapshot_sha256") == _digest(body), "corrupt finalized snapshot")
    _require(filename_digest is None or filename_digest == _digest(body), "corrupt finalized snapshot filename")
    snapshot = json.loads(body)
    if snapshot.get("observation_reference_schema") == 1:
        _validate_saved_attempts(snapshot, document.get("attempts"), document.get("evidence", []))
    observations = _observations(body, document.get("evidence", []))
    _require(document.get("observations") == observations, "finalized fields differ from verified evidence")
    return observations


def select_finalized_observation(rows: list[dict], *, fighter_a_id: str, fighter_b_id: str, event_date: str,
                                 event_id: str | None = None, commence_time: str | None = None, as_of: str | None = None) -> dict | None:
    """Require stable identities and an unambiguous event; preserve missing values."""
    if not isinstance(fighter_a_id, str) or not isinstance(fighter_b_id, str) or not re.fullmatch(r"[0-9a-f]{16}", fighter_a_id) or not re.fullmatch(r"[0-9a-f]{16}", fighter_b_id) or fighter_a_id == fighter_b_id:
        return None
    matches = [row for row in rows if row["event_date"] == event_date and {row["fighter_a_id"], row["fighter_b_id"]} == {fighter_a_id, fighter_b_id}]
    if event_id is not None:
        matches = [row for row in matches if row["event_id"] == event_id]
    if commence_time is not None:
        matches = [row for row in matches if _time(row["commence_time"]) == _time(commence_time)]
    if as_of is not None:
        matches = [row for row in matches if _time(row["observed_at"]) <= _time(as_of)]
    if not matches:
        return None
    _require(len({row["event_id"] for row in matches}) == 1, "ambiguous finalized event identity")
    _require(len({_time(row["commence_time"]) for row in matches}) == 1, "ambiguous rescheduled commencement")
    latest = max(_time(row["observed_at"]) for row in matches)
    oriented = []
    for row in matches:
        if _time(row["observed_at"]) != latest:
            continue
        reverse = row["fighter_a_id"] != fighter_a_id
        oriented.append({name: row["method_odds"][("b" if name[0] == "a" else "a") + name[1:]] if reverse else row["method_odds"][name] for name in FEATURES})
    _require(all(value == oriented[0] for value in oriented), "conflicting finalized observations")
    return oriented[0]
