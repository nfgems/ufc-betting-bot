"""Account for every saved monitor bout without granting collection authority.

This report is separate from method_reviewed_bindings_v1. A saved card is an
inventory observation, never proof of stable fighter identity or market absence.
Preparation and inspection are read-only and do not replace collection preflight.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from src.data.method_observation_history import (
    _json_bytes, _read_pin, _require, _safe_path, _time, validate_reviewed_context,
)


KIND = "method_inventory_coverage_v1"
RUNTIME_KIND = "method_inventory_coverage_v2"
IDENTITY_FIELDS = ("event_id", "fighter_a", "fighter_b", "fighter_a_id", "fighter_b_id", "commence_time")


def _external_pin(path, expected_sha256):
    _require(isinstance(expected_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha256),
             "independently supplied SHA256 is required")
    selected = Path(path)
    _require(selected.is_absolute(), "coverage input path must be absolute")
    body = _safe_path(selected).read_bytes()
    _require(hashlib.sha256(body).hexdigest() == expected_sha256, "coverage input pin differs")
    return {"path": str(selected), "bytes": len(body), "sha256": expected_sha256}


def _inventory_rows(inventory_pin, *, use_runtime=False):
    saved = json.loads(_read_pin(inventory_pin))
    _require(isinstance(saved, dict) and isinstance(saved.get("events"), list)
             and saved["events"], "coverage needs the complete saved monitor event inventory")
    rows = []
    for event_index, event in enumerate(saved["events"]):
        _require(isinstance(event, dict) and isinstance(event.get("fights"), list), "invalid saved event")
        if "num_fights" in event:
            _require(type(event["num_fights"]) is int and event["num_fights"] == len(event["fights"]),
                     "saved event fight count differs")
        event_fields = {key: value for key, value in event.items() if key != "fights"}
        for fight_index, fight in enumerate(event["fights"]):
            _require(isinstance(fight, dict) and all(isinstance(fight.get(key), str) and fight[key].strip()
                     for key in ("fighter_a", "fighter_b")), "invalid saved fight")
            # Match the existing monitor's event fallback, without adding any ID.
            tracked = {key: fight.get(key, "") for key in IDENTITY_FIELDS}
            for key in ("event_id", "commence_time"):
                tracked[key] = fight.get(key, "") or event.get(key, "")
            rows.append({"event_index": event_index, "fight_index": fight_index,
                         "inventory_row_sha256": hashlib.sha256(_json_bytes({"event": event_fields, "fight": fight})).hexdigest(),
                         "tracked": tracked})
    _require(bool(rows), "coverage inventory has no fights")
    if use_runtime:
        rows = _runtime_inventory_rows(saved, rows)
    return saved, rows


def _runtime_inventory_rows(saved, rows):
    """Account for the saved collector view without re-running its enrichment.

    This checks correspondence to the raw cards, not the authority of a saved
    identity binding. Independent reviewed method bindings remain separate.
    """
    runtime = saved.get("method_tracked_fights")
    _require(isinstance(runtime, list) and len(runtime) == len(rows),
             "saved runtime inventory differs from complete raw card count")
    result = []
    for row, enriched in zip(rows, runtime):
        _require(isinstance(enriched, dict), "invalid saved runtime fight")
        event = saved["events"][row["event_index"]]
        fight = event["fights"][row["fight_index"]]
        original = {
            "event_title": event.get("title", ""),
            "event_date": event.get("date", ""),
            "event_url": event.get("url", ""),
            **{key: fight.get(key, "") for key in (
                "fighter_a", "fighter_b", "fighter_a_id", "fighter_b_id",
                "fighter_a_athlete_url", "fighter_b_athlete_url", "weight_class")},
            **{key: fight.get(key, "") or event.get(key, "")
               for key in ("event_id", "commence_time")},
        }
        _require(all(key in enriched for key in original), "saved runtime fight omits original fields")
        mutable = {"fighter_a_id", "fighter_b_id", "event_id", "commence_time"}
        _require(all(enriched[key] == value for key, value in original.items() if key not in mutable),
                 "saved runtime fight changes raw order, names, routes or event")
        provenance = enriched.get("runtime_identity_binding")
        if "runtime_identity_binding" in enriched:
            _require(isinstance(provenance, dict) and isinstance(provenance.get("sides"), dict)
                     and set(provenance["sides"]) == {"a", "b"}, "invalid saved runtime identity binding")
        refusals = {}
        for side in ("a", "b"):
            field = f"fighter_{side}_id"
            prior, current = original[field], enriched[field]
            detail = provenance["sides"][side] if provenance is not None else None
            if provenance is not None:
                _require(isinstance(detail, dict)
                         and all(key in detail and detail[key] == expected for key, expected in (
                             ("submitted_name", original[f"fighter_{side}"]),
                             ("submitted_id", prior),
                             ("submitted_athlete_url", original[f"fighter_{side}_athlete_url"]))),
                         "saved runtime binding changes submitted identity")
                if detail.get("status") == "withheld":
                    _require(current is None and isinstance(detail.get("reason"), str) and detail["reason"].strip(),
                             "withheld runtime identity cannot supply a fighter ID")
                    refusals[side] = {"submitted_id": prior, "reason": detail["reason"]}
                else:
                    _require(detail.get("status") == "bound" and isinstance(current, str)
                             and re.fullmatch(r"[0-9a-f]{16}", current)
                             and detail.get("fighter_id") == current, "invalid saved bound runtime identity")
            # A genuine recorded refusal may clear a conflicting submitted ID.
            # It cannot replace that ID with another identity or hide the row.
            withheld = detail is not None and detail.get("status") == "withheld" and current is None
            _require(prior in (None, "") or current == prior or withheld,
                     "saved runtime fight changes a known fighter ID")
        for field in ("event_id", "commence_time"):
            _require(not original[field] or enriched[field] == original[field],
                     "saved runtime fight changes known event identity or timing")
        result.append({**row,
                       "runtime_inventory_row_sha256": hashlib.sha256(_json_bytes(enriched)).hexdigest(),
                       "runtime_identity_refusals": refusals,
                       "tracked": {key: enriched[key] for key in IDENTITY_FIELDS}})
    return result


def _bindings(pin):
    if pin is None:
        return []
    return validate_reviewed_context(json.loads(_read_pin(pin)))["bindings"]


def _observation_interval(observed_at, not_after):
    if observed_at is None and not_after is None:
        return None
    _require(observed_at is not None and not_after is not None,
             "unknown inventory observation interval requires two null values")
    observed, expires = _time(observed_at), _time(not_after)
    _require(observed < expires, "invalid inventory observation interval")
    return observed, expires


def _decision(row, bindings):
    tracked = row["tracked"]
    reasons = []
    ids = [tracked[key] for key in ("fighter_a_id", "fighter_b_id")]
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{16}", value) is None for value in ids) or ids[0] == ids[1]:
        reasons.append("missing_or_invalid_saved_stable_identity")
    if not isinstance(tracked["event_id"], str) or not tracked["event_id"].strip():
        reasons.append("missing_saved_event_identity")
    try:
        _time(tracked["commence_time"])
    except ValueError:
        reasons.append("missing_or_invalid_saved_commencement")
    matches = [index for index, binding in enumerate(bindings)
               if all(tracked[key] == binding[key] for key in IDENTITY_FIELDS)]
    _require(len(matches) <= 1, "ambiguous reviewed coverage binding")
    if not matches:
        reasons.append("no_matching_reviewed_method_binding")
    return {"decision": "blocked" if reasons else "collect",
            "binding_index": matches[0] if matches and not reasons else None,
            "reasons": reasons}


def prepare_inventory_coverage(*, inventory_path, inventory_sha256, observed_at,
                               not_after, evidence_pins, reviewed_bindings_pin=None):
    """Return an unapproved draft; saving or independently reviewing it is separate.

    Evidence pins preserve the review inputs. They do not establish that a
    provider lacks a page or market, and no such claim is accepted in this schema.
    """
    inventory_pin = _external_pin(inventory_path, inventory_sha256)
    _observation_interval(observed_at, not_after)
    _require(isinstance(evidence_pins, list) and evidence_pins, "coverage review evidence is required")
    for pin in evidence_pins:
        _read_pin(pin)
    _require(len({_json_bytes(pin) for pin in evidence_pins}) == len(evidence_pins), "duplicate coverage evidence pin")
    saved, rows = _inventory_rows(inventory_pin)
    use_runtime = "method_tracked_fights" in saved
    if use_runtime:
        rows = _runtime_inventory_rows(saved, rows)
    bindings = _bindings(reviewed_bindings_pin)
    decisions = [{**row, **_decision(row, bindings)} for row in rows]
    return {"schema": 2 if use_runtime else 1, "kind": RUNTIME_KIND if use_runtime else KIND,
            "inventory": {"pin": inventory_pin, "observed_at": observed_at, "not_after": not_after},
            "evidence": evidence_pins, "reviewed_bindings": reviewed_bindings_pin, "rows": decisions}


def inspect_inventory_coverage(*, report_path, report_sha256, now=None):
    """Verify exact accounting against pinned originals without admitting a run."""
    report_pin = _external_pin(report_path, report_sha256)
    document = json.loads(_read_pin(report_pin))
    _require(isinstance(document, dict) and set(document) == {
        "schema", "kind", "inventory", "evidence", "reviewed_bindings", "rows"}
        and type(document["schema"]) is int
        and (document["schema"], document["kind"]) in ((1, KIND), (2, RUNTIME_KIND)),
        "invalid method inventory coverage schema")
    inventory = document["inventory"]
    _require(isinstance(inventory, dict) and set(inventory) == {"pin", "observed_at", "not_after"},
             "invalid coverage inventory provenance")
    interval = _observation_interval(inventory["observed_at"], inventory["not_after"])
    evidence = document["evidence"]
    _require(isinstance(evidence, list) and evidence, "coverage review evidence is required")
    for pin in evidence:
        _read_pin(pin)
    _require(len({_json_bytes(pin) for pin in evidence}) == len(evidence), "duplicate coverage evidence pin")
    use_runtime = document["schema"] == 2
    saved, original_rows = _inventory_rows(inventory["pin"], use_runtime=use_runtime)
    bindings = _bindings(document["reviewed_bindings"])
    rows = document["rows"]
    expected = [{**row, **_decision(row, bindings)} for row in original_rows]
    # Canonical byte comparison also refuses bool-for-int indices and extra keys.
    _require(isinstance(rows, list) and _json_bytes(rows) == _json_bytes(expected),
             "coverage rows differ from the complete pinned inventory or reviewed decisions")
    current = now if now is not None else datetime.now(timezone.utc)
    _require(isinstance(current, datetime) and current.tzinfo is not None, "coverage inspection needs aware time")
    current = current.astimezone(timezone.utc)
    eligible = [row for row in rows if row["decision"] == "collect"]
    blocked = [row for row in rows if row["decision"] == "blocked"]
    cutoffs_current = bool(eligible) and all(current < _time(bindings[row["binding_index"]]["cutoff_at"])
                                            for row in eligible)
    inspection = {"schema": 2 if use_runtime else 1,
            "kind": "method_inventory_coverage_inspection_v2" if use_runtime else "method_inventory_coverage_inspection_v1",
            "report": report_pin,
            "inventory": inventory, "checked_at": current.isoformat(), "accounting_complete": True,
            "tracked_card_count": len(saved["events"]), "tracked_fight_count": len(rows),
            "eligible_binding_count": len(eligible), "blocked_fight_count": len(blocked),
            "all_rows_have_bindings": not blocked, "observation_interval_known": interval is not None,
            "inventory_current": interval is not None and interval[0] <= current < interval[1],
            "eligible_binding_cutoffs_current": cutoffs_current, "rows": rows,
            "ready": False, "collection_ready": False, "collection_claimed": False,
            "source_requests_made": False, "source_authority_created": False,
            "qualification": "Complete accounting of the pinned saved inventory only. An unknown observation interval remains unknown; artifact read time is not source freshness. Missing saved timing is not proof that temporary runtime enrichment lacked timing. Collect decisions identify existing matching bindings, not permission or admission to run. Existing whole-inventory collection preflight remains required. No market absence or prediction eligibility is inferred."}
    if use_runtime:
        inspection["inventory_view"] = "saved_method_tracked_fights"
        inspection["qualification"] = (
            "Complete accounting of the pinned saved collector inventory, matched in order to every raw card bout. "
            "Saved identity metadata is checked for consistency, not independently authenticated here. "
            "Enriched IDs and timing do not grant source authority or establish observation freshness. "
            "Unknown observation timing stays unknown. Collect decisions identify matching reviewed bindings only; "
            "whole-inventory collection preflight remains separate. Missing method values may remain NaN for "
            "otherwise-valid prediction; this report grants neither prediction eligibility nor collection success "
            "and does not establish market absence.")
    return inspection
