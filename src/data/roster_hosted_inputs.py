"""Recover saved roster inputs from external accepted and journal anchors.

An installed, server-owned pointer binds the complete immutable BASE_ROLES
manifest. Its hash, release hash and minimum journal checkpoint hash must come
from outside this preparation. This module writes proposals only; its output
cannot install a pointer, authorize collection or make consumed work executable.
Historical stages are independently recomputed at their original preparation
times before fresh needs are derived at the current server time.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.data.roster_hosted_needs import BASE_ROLES, derive_need_evidence, prepare_need_manifest
from src.data.roster_hosted_stages import validate_stage_chain
from src.data.roster_hosted_store import (
    _fsync_dir, _json, _new_file, _read, _require, _safe, _time, file_pin, verify_pin,
)


def _load(pin):
    verify_pin(pin)
    return json.loads(_read(pin["path"]))


def _write(output, name, value):
    path = output / name
    _new_file(path, _json(value), immutable=True)
    return file_pin(path)


def _base(pins):
    return {key: pins[key] for key in sorted(BASE_ROLES)}


def _manifest(pin):
    value = _load(pin)
    _require(isinstance(value, dict) and set(value) == {"schema", "input_pins"}
             and value["schema"] == 1 and isinstance(value["input_pins"], dict)
             and set(value["input_pins"]) == BASE_ROLES,
             "Complete immutable accepted BASE_ROLES manifest required")
    for item in value["input_pins"].values():
        verify_pin(item)
    return value["input_pins"]


def _accepted(store, pointer_pin, expected_pointer_sha256, expected_release_sha256):
    pointer = _load(pointer_pin)
    _require(pointer_pin["sha256"] == expected_pointer_sha256,
             "External accepted pointer hash mismatch")
    _require(isinstance(pointer, dict) and set(pointer) == {
        "schema", "kind", "input_manifest", "store_manifest_sha256"}
        and pointer["schema"] == 1 and pointer["kind"] == "accepted_roster_pointer_v1",
        "Exact server-owned accepted pointer required")
    base = _manifest(pointer["input_manifest"])
    _require(base["release"]["sha256"] == expected_release_sha256,
             "External accepted release hash mismatch")
    return pointer, base


def _checkpoint(store, state, pointer_pin, checkpoint_pin, expected_checkpoint_sha256, now):
    checkpoint = _load(checkpoint_pin)
    _require(checkpoint_pin["sha256"] == expected_checkpoint_sha256,
             "External recovery checkpoint hash mismatch")
    _require(isinstance(checkpoint, dict) and set(checkpoint) == {
        "schema", "kind", "accepted_pointer", "store_root", "store_head"}
        and checkpoint["schema"] == 1 and checkpoint["kind"] == "roster_input_checkpoint_v1"
        and checkpoint["accepted_pointer"] == pointer_pin
        and checkpoint["store_root"] == store.root.as_posix(),
        "Checkpoint accepted state or accounting namespace changed")
    head = checkpoint["store_head"]
    _require(isinstance(head, dict) and set(head) == {"sequence", "sha256", "manifest_sha256"}
             and type(head["sequence"]) is int
             and 0 <= head["sequence"] <= state["head"]["sequence"]
             and head["manifest_sha256"] == state["head"]["manifest_sha256"],
             "Accounting rollback or checkpoint manifest mismatch")
    # snapshot() independently validates every event, charge, receipt and claim.
    # Read that exact validated chain and then require an unchanged snapshot.
    events = [json.loads(line) for line in _read(store.root / "journal.jsonl").splitlines()]
    manifest = json.loads(_read(store.root / "manifest.json"))
    _require(_time(manifest["created_at"]) <= now, "Server time precedes accounting initialization")
    digest = events[head["sequence"] - 1]["sha256"] if head["sequence"] else "0" * 64
    _require(head["sha256"] == digest, "Accounting rollback or checkpoint ancestry conflict")
    if events:
        _require(_time(events[-1]["at_utc"]) <= now, "Server time precedes durable accounting")
    _require(store.snapshot() == state, "Accounting changed during checkpoint verification")
    return checkpoint


def _recover_cycles(store, state, accepted, now, root_evidence, *, adoption_context=None):
    """Only journal-anchored transitions participate; directory names do not."""
    cycles, recovered, represented_collections = [], [], set()
    _require(isinstance(root_evidence, dict) and set(root_evidence) <= set(state["plan_slots"]),
             "Root evidence must identify original consumed slots")
    for slot_id, records in state["plan_stages"].items():
        first, last = _load(records[0]["stage_pin"]), _load(records[-1]["stage_pin"])
        root_pin, review_pin = records[0]["root_plan_pin"], first["root_review_pin"]
        root = _load(root_pin)
        final_time = _time(last["prepared_at"])
        _require(final_time <= now, "Future historical cycle cannot provide inputs")
        for record in records:
            stage = _load(record["stage_pin"])
            _require(stage["source_plan_id"] not in represented_collections,
                     "Consumed source collection cannot supply multiple historical phases")
            represented_collections.add(stage["source_plan_id"])
            _require(stage["prepared_at"] == record["at_utc"]
                     and _time(root["issued_at"]) <= _time(stage["prepared_at"])
                         <= _time(root["expires_at"]),
                     "Historical stage escaped its original reservation time or lifetime")
            directory = Path(record["stage_pin"]["path"]).parent
            # Interrupted post-reservation metadata is reported explicitly. A
            # fresh complete recomputation can recover the immutable stage; no
            # old terminal or consumed preparation is reconstructed or replayed.
            missing = [name for name in ("review.json", "terminal.json")
                       if not (directory / name).is_file()]
            if missing:
                recovered.append({"stage": record["stage_pin"], "missing_outputs": missing,
                                  "replay_authorized": False})
        review = validate_stage_chain(store, root_pin, review_pin, now=final_time)
        initial, final = _base(root["content"]["input_pins"]), _base(last["input_pins"])
        failed_pin = root_evidence.get(slot_id, {}).get("failed_membership")
        if failed_pin is not None:
            from src.data.roster_hosted_recurrence import verify_membership_failure
            failed = verify_membership_failure(store, failed_pin, now=now)
            _require(failed["root_plan"] == root_pin and failed["root_review"] == review_pin
                     and failed["prior_inputs"] == final, "Failed cycle original lineage changed")
            _require(failed["source_plan_id"] not in represented_collections,
                     "Failed source collection cannot supply multiple cycles")
            represented_collections.add(failed["source_plan_id"])
        cycles.append({"slot_id": slot_id, "root_plan": root_pin, "root_review": review_pin,
            "initial_inputs": initial, "final_inputs": final,
            "root_sequence": state["plan_slots"][slot_id]["sequence"],
            "final_sequence": records[-1]["sequence"], "final_stage": records[-1]["stage_pin"],
            "review": review})
    for slot_id in sorted(set(state["plan_slots"]) - set(state["plan_stages"])):
        evidence = root_evidence.get(slot_id)
        _require(isinstance(evidence, dict) and set(evidence) in ({"plan", "review"}, {"plan", "review", "failed_membership"}),
                 "Unadvanced consumed slot requires original root recovery evidence")
        root = _load(evidence["plan"])
        _require(root["slot_id"] == slot_id and _time(root["issued_at"]) <= now,
                 "Unadvanced root evidence belongs to another or future slot")
        review = validate_stage_chain(store, evidence["plan"], evidence["review"],
                                      now=_time(root["issued_at"]))
        initial = _base(root["content"]["input_pins"])
        if root["content"]["proposed_requests"]:
            from src.data.roster_hosted_recurrence import verify_membership_failure
            _require("failed_membership" in evidence,
                     "Unadvanced consumed source phase requires receipt recovery before new inputs")
            failed = verify_membership_failure(store, evidence["failed_membership"], now=now)
            _require(failed["root_plan"] == evidence["plan"] and failed["root_review"] == evidence["review"]
                     and failed["prior_inputs"] == initial, "Failed cycle original lineage changed")
            _require(failed["source_plan_id"] not in represented_collections,
                     "Failed source collection cannot supply multiple cycles")
            represented_collections.add(failed["source_plan_id"])
        sequence = state["plan_slots"][slot_id]["sequence"]
        cycles.append({"slot_id": slot_id, "root_plan": evidence["plan"], "root_review": evidence["review"],
            "initial_inputs": initial, "final_inputs": initial, "root_sequence": sequence,
            "final_sequence": sequence, "final_stage": None, "review": review})
    # A complete result can exist after the latest stage when the controller
    # stops before advancing it. Do not silently discard such failed/conflicting
    # captures or reclaim a consumed-but-not-sent source collection under a new
    # cycle. Imported historical accounting has no fresh plan_claimed events.
    events = [json.loads(line) for line in _read(store.root / "journal.jsonl").splitlines()]
    source_claims = {row["plan_id"] for row in events if row["event"] == "plan_claimed"
                     and _load(row["plan_pin"]).get("kind") == "roster_source_collection_v1"}
    source_claims |= {row["plan_id"] for row in state["requests"].values()}
    _require(source_claims == represented_collections,
             "Unattributed consumed source evidence requires original phase recovery")
    cycles.sort(key=lambda item: item["root_sequence"])
    selected, lineage, excluded, sequence = accepted, [], [], 0
    transitions = {} if adoption_context is None else adoption_context["transitions"]
    consumed_transitions = set()
    for cycle in cycles:
        initial = cycle["initial_inputs"]
        if adoption_context is None and (_load(initial["owners"]) != _load(accepted["owners"])
                or initial["release"]["sha256"] != accepted["release"]["sha256"]):
            excluded.append({"slot_id": cycle["slot_id"],
                             "reason": "different_externally_selected_accepted_seed_or_release"})
            continue
        _require(initial == selected and cycle["root_sequence"] > sequence,
                 "Conflicting or stale prior-cycle input lineage; no candidate may be dropped")
        selected, sequence = cycle["final_inputs"], cycle["final_sequence"]
        lineage.append(cycle)
        if cycle["slot_id"] in transitions:
            edge = transitions[cycle["slot_id"]]
            _require(edge["final_inputs"] == selected
                     and _load(edge["binding"])["root_plan"] == cycle["root_plan"],
                     "Accepted adoption edge differs from original journal lineage")
            selected = edge["accepted_inputs"]
            consumed_transitions.add(cycle["slot_id"])
    _require(consumed_transitions == set(transitions), "Unattributed accepted adoption transition")
    return selected, lineage, excluded, recovered


def prepare_starting_inputs(store, accepted_pointer_pin, output_directory, *,
                           expected_pointer_sha256, expected_release_sha256,
                           checkpoint_pin, expected_checkpoint_sha256, now=None, root_evidence=None,
                           adoption_configuration_pin=None, expected_adoption_configuration_sha256=None):
    """Select complete saved inputs and write a fresh disabled need manifest.

The checkpoint is a minimum retained journal anchor, not an automatic backup or
proof against rollback after that anchor. Its hash and accepted/release hashes
must be supplied by the trusted server configuration outside these artifacts.
Unadvanced slots block selection unless original root_evidence independently
proves a completed no-request root. The optional mapping gives discovery pins
only; the journal binding and semantic proof are recomputed. No filenames, mtimes, reviewer
booleans or partial outputs grant selection or execution authority.
"""
    now = _time(now)
    output = _safe(output_directory)
    _require(not output.exists() and output.parent.is_dir(), "Fresh starting-input output required")
    output.mkdir(mode=0o700)
    _fsync_dir(output.parent)
    try:
        pointer, accepted = _accepted(store, accepted_pointer_pin, expected_pointer_sha256,
                                     expected_release_sha256)
        state = store.snapshot()
        _require(pointer["store_manifest_sha256"] == state["head"]["manifest_sha256"],
                 "Accepted pointer belongs to a different accounting manifest")
        _require(not state["uncertain_requests"], "Uncertain accounting prevents input recovery")
        _checkpoint(store, state, accepted_pointer_pin, checkpoint_pin, expected_checkpoint_sha256, now)
        adoption_context = None
        origin = accepted
        if adoption_configuration_pin is not None:
            from src.data.roster_refresh_adoption import resolve_adoption_context
            adoption_context = resolve_adoption_context(store, adoption_configuration_pin,
                expected_configuration_sha256=expected_adoption_configuration_sha256, now=now)
            _require(adoption_context["accepted_pointer"] == accepted_pointer_pin
                     and adoption_context["checkpoint"] == checkpoint_pin
                     and adoption_context["expected_release_sha256"] == expected_release_sha256,
                     "Controller accepted anchors differ from independently validated adoption")
            origin = adoption_context["genesis_inputs"]
        else:
            _require(expected_adoption_configuration_sha256 is None, "Adoption hash requires its configuration")
        selected, lineage, excluded, recovered = _recover_cycles(store, state, origin, now,
            {} if root_evidence is None else root_evidence, adoption_context=adoption_context)
        # The mandatory manifest preparation below fully derives and verifies
        # selected inputs. When these are the exact accepted pins, a separate
        # identical derivation would replay the entire owner history twice.
        # Distinct accepted evidence must still receive its own full check.
        if selected != accepted:
            derive_need_evidence(accepted, now=now)
        base_pin = _write(output, "base-inputs-manifest.json", {"schema": 1, "input_pins": selected})
        needs = prepare_need_manifest(selected, output / "needs", now=now)
        _require(store.snapshot() == state, "Accounting changed during starting-input derivation")
        _accepted(store, accepted_pointer_pin, expected_pointer_sha256, expected_release_sha256)
        _load(checkpoint_pin)
        proof = {"schema": 1, "status": "starting_inputs_validated_execution_disabled",
            "selected_at": now.isoformat(), "accepted_pointer": accepted_pointer_pin,
            "accepted_input_manifest": pointer["input_manifest"], "checkpoint": checkpoint_pin,
            "store_root": store.root.as_posix(), "store_head": state["head"],
            "base_inputs_manifest": base_pin, "inputs_manifest": needs["inputs_manifest"],
            "cycles": lineage, "excluded_cycles": excluded,
            "recovered_incomplete_outputs": recovered,
            "unadvanced_slots": [],
            "source_requests_authorized": 0, "activation_authorized": False,
            "publication_authorized": False, "replay_authorized": False}
        selection_pin = _write(output, "selection.json", proof)
        result = {**needs, "base_inputs_manifest": base_pin, "selection": selection_pin,
            "store_head": state["head"], "publication_authorized": False,
            "replay_authorized": False}
        _write(output, "terminal.json", result)
        return result
    except BaseException as exc:
        _write(output, "failed.json", {"status": "failed_execution_disabled",
            "error": type(exc).__name__ + ": " + str(exc), "replay_authorized": False,
            "source_requests_authorized": 0, "activation_authorized": False,
            "publication_authorized": False})
        raise
