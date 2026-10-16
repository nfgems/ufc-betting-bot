"""Disabled server controller for accepted inputs and finite saved-response cycles.

The optional collection path reads an immutable synthetic response file. There
is no live callback, URL argument, scheduler, candidate writer or publisher.
Restart recovery validates already consumed receipts and never resends a plan.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from src.data.roster_hosted_broker import SourceBroker, MEMBERSHIP_URL
from src.data.roster_hosted_needs import BASE_ROLES
from src.data.roster_hosted_plan import _policy, _slot, issue_candidate_plan
from src.data.roster_hosted_stages import prepare_next_stage, validate_stage_chain
from src.data.roster_hosted_store import (
    StoreError, _fsync_dir, _json, _new_file, _read, _require, _safe, _time,
    file_pin, verify_pin,
)


def _load(pin):
    return json.loads(_read(verify_pin(pin)["path"]))


def _save(parent, name, value):
    path = parent / name
    _new_file(path, _json(value), immutable=True)
    return file_pin(path)


def _directory(path):
    path = _safe(path)
    _require(not path.exists() and path.parent.is_dir(), "Unused controller directory required")
    path.mkdir(mode=0o700)
    _fsync_dir(path.parent)
    return path


def _protected(binding):
    """Recheck complete accepted state immediately before issue/send/advance."""
    from src.data.roster_hosted_candidate import tree_inventory
    verify_pin(binding["accepted_pointer"])
    verify_pin(binding["checkpoint"])
    pointer = _load(binding["accepted_pointer"])
    accepted = _load(pointer["input_manifest"])["input_pins"]
    for base in (accepted, binding["base_inputs"]):
        _require(set(base) == BASE_ROLES, "Complete protected input roles required")
        for pin in base.values():
            verify_pin(pin)
        membership = _load(base["membership"])
        verify_pin(membership["body"])
        verify_pin(membership["envelope"])
        owners = _load(base["owners"])
        _require(tree_inventory(owners["seed_root"]) == _load(owners["seed_inventory"]),
                 "Accepted seed changed during controller cycle")
    _require(binding["base_inputs"]["release"]["sha256"] == binding["expected_release_sha256"],
             "Controller release authority changed")
    if binding["synthetic_responses"] is not None:
        verify_pin(binding["synthetic_responses"])


def _saved_wire(pin):
    """Read-only synthetic exchange; pins are rechecked on every invocation."""
    value = _load(pin)
    _require(isinstance(value, dict) and set(value) == {"schema", "mode", "responses"}
             and value["schema"] == 1 and value["mode"] == "saved_synthetic_only"
             and isinstance(value["responses"], list) and len(value["responses"]) <= 12,
             "Exact bounded synthetic response fixture required")
    responses = {}
    for row in value["responses"]:
        _require(isinstance(row, dict) and set(row) == {"method", "url", "status", "headers",
                  "body_base64", "complete", "error"} and row["method"] == "GET",
                 "Only explicit saved GET responses supported")
        _require(row["url"] not in responses and type(row["complete"]) is bool,
                 "Repeated or ambiguous synthetic response")
        raw = base64.b64decode(row["body_base64"], validate=True)
        _require(len(raw) <= 8388608, "Synthetic response body ceiling")
        responses[row["url"]] = dict(row, body=raw)

    def exchange(method, url, headers, body, scope):
        verify_pin(pin)
        _require(method == "GET" and body is None and url in responses,
                 "No saved synthetic response for derived request; no send performed")
        row = responses[url]
        return {key: row[key] for key in ("status", "headers", "body", "complete", "error")} | {
            "wire_metadata": {"synthetic": True, "fixture": pin, "socket_created": False}}
    return exchange


def _discover_roots(store, parent):
    """Find original root pins; the selector validates their journal authority.

    Names are discovery only. They never rank/select a cycle or establish a
    completed result. Missing bindings for an unresolved slot still block.
    """
    roots = {}
    entries = list(parent.iterdir())
    _require(len(entries) <= 10000, "Controller history entry ceiling")
    for entry in entries:
        _safe(entry)
        if not entry.is_dir() or not (entry / "binding.json").exists():
            continue
        binding = _load(file_pin(entry / "binding.json"))
        _require(binding.get("mode") == "disabled_controller_v1"
                 and binding.get("store_root") == store.root.as_posix()
                 and binding.get("output_root") == entry.as_posix(),
                 "Controller history binding namespace changed")
        root = _load(binding["root_plan"])
        pins = {"plan": binding["root_plan"], "review": binding["root_review"]}
        failure = entry / "reconciled-membership-failure.json"
        if failure.exists():
            pins["failed_membership"] = file_pin(failure)
        _require(root["slot_id"] not in roots or roots[root["slot_id"]] == pins,
                 "Conflicting controller roots for one consumed slot")
        roots[root["slot_id"]] = pins
    return roots


def _cycle_state(store, binding, now):
    validate_stage_chain(store, binding["root_plan"], binding["root_review"], now=now)
    root = _load(binding["root_plan"])
    state = store.snapshot()
    chain = state["plan_stages"].get(root["slot_id"], [])
    previous = _load(chain[-1]["stage_pin"]) if chain else None
    requests = previous["proposed_requests"] if previous else root["content"]["proposed_requests"]
    return root, state, chain, previous, requests


def _persist(store, callback, pin):
    """Consume an explicit custody acknowledgment on the connected local path."""
    strict = store.mutation_guard is not None or getattr(callback, "strict_checkpoint_ack", False)
    if strict:
        _require(callable(callback) and getattr(callback, "strict_checkpoint_ack", False)
                 and store.mutation_guard is callback,
                 "Connected controller requires the same independent mutation and checkpoint custodian")
    if not callback:
        return None
    result = callback(pin)
    if not strict:
        return None  # Historical preservation-only callback grants no custody claim.
    head = store.snapshot()["head"]
    _require(isinstance(result, dict) and result.get("requested") == pin
             and result.get("retained_store_head") == head
             and result.get("current_store_head") == head
             and result.get("unretained_store_tail") is False
             and isinstance(result.get("custody_ack"), dict),
             "Exact independently retained checkpoint acknowledgment required")
    _require(callback.confirm_checkpoint(result, pin, head) is True,
             "Independent current-head acknowledgment was not confirmed")
    verify_pin(result["checkpoint"])
    return result


def _drive(store, binding_pin, *, clock, persist_checkpoint, recovering,
           checkpoint_pin=None, expected_checkpoint_sha256=None, synthetic_helper=None):
    binding = _load(binding_pin)
    _require(binding.get("schema") == 1 and binding.get("mode") == "disabled_controller_v1"
             and binding.get("store_root") == store.root.as_posix()
             and binding.get("activation_authorized") is False
             and binding.get("publication_authorized") is False,
             "Exact disabled controller binding required")
    output = _safe(binding["output_root"])
    _require(binding_pin == file_pin(output / "binding.json"), "Controller binding location changed")
    if store.mutation_guard is not None or getattr(persist_checkpoint, "strict_checkpoint_ack", False):
        _require(store.mutation_guard is persist_checkpoint
                 and getattr(persist_checkpoint, "strict_checkpoint_ack", False),
                 "Connected controller custody must be established before recovery mutations")
        controls = persist_checkpoint.controller_configuration()
        _require(controls == {"policy": _load(binding["root_plan"])["content"]["policy_pin"],
                              "synthetic_responses": binding["synthetic_responses"]},
                 "Original controller policy or fixture differs from provisioned custody controls")
    _protected(binding)
    if checkpoint_pin is None:
        checkpoint_pin = binding["checkpoint"]
    if expected_checkpoint_sha256 is None:
        expected_checkpoint_sha256 = binding["checkpoint"]["sha256"]
    wire = _saved_wire(binding["synthetic_responses"]) if binding["synthetic_responses"] else None
    _require(bool(binding.get("synthetic_helper_required", False)) == (synthetic_helper is not None),
             "Original controller helper boundary must be preserved on recovery")
    if synthetic_helper is not None:
        _require(store.mutation_guard is persist_checkpoint
                 and getattr(persist_checkpoint, "strict_checkpoint_ack", False)
                 and synthetic_helper.synthetic_responses_pin == binding["synthetic_responses"],
                 "Saved helper requires exact fixture and connected independent custody")
    for _ in range(13):  # Every nonempty phase consumes at least one of 12 reserved charges.
        now = _time(clock())
        _protected(binding)
        from src.data.roster_hosted_inputs import _checkpoint
        _checkpoint(store, store.snapshot(), binding["accepted_pointer"], checkpoint_pin,
                    expected_checkpoint_sha256, now)
        root, state, chain, previous, requests = _cycle_state(store, binding, now)
        failure_path = output / "reconciled-membership-failure.json"
        if failure_path.exists():
            # Completed membership failures have no advanced success stage.
            # Recover their exact original receipts after the ordinary current
            # checkpoint, root lifetime and custody checks above. Rebuilding the
            # failed stage would attempt to overwrite its immutable descriptor.
            _require(recovering, "Existing membership failure requires original recovery")
            from src.data.roster_hosted_recurrence import validated_cycle_status
            outcome = validated_cycle_status(store, binding_pin, now=now)
            _require(outcome["status"] == "failed"
                     and outcome["root_plan"] == binding["root_plan"]
                     and outcome["root_review"] == binding["root_review"]
                     and store.snapshot() == state,
                     "Completed failure recovery changed original lineage or accounting")
            return {"schema": 1, "status": "cycle_failed_membership_reconciled_execution_disabled",
                "binding": binding_pin, "root_plan": binding["root_plan"],
                "root_review": binding["root_review"], "failure": file_pin(failure_path),
                "outcome": outcome, "store_head": state["head"], "source_requests": 0,
                "activation_authorized": False, "publication_authorized": False,
                "schedule_enabled": False, "replay_authorized": False,
                "recovered_without_replay": True, "original_failure_preserved": True}
        if not requests or wire is None:
            final_inputs = previous["input_pins"] if previous else root["content"]["input_pins"]
            proof = _load(final_inputs["need_evidence"])
            status = "cycle_prepared_execution_disabled" if not requests else "awaiting_collection_execution_disabled"
            result = {"schema": 1, "status": status, "binding": binding_pin,
                "root_plan": binding["root_plan"], "root_review": binding["root_review"],
                "stage_count": len(chain), "store_head": state["head"], "input_pins": final_inputs,
                "source_requests": 0, "activation_authorized": False, "publication_authorized": False,
                "schedule_enabled": False, "synthetic_only": wire is not None,
                "recovered_without_replay": recovering,
                "budget_deferred_requests": previous["budget_deferred_requests"] if previous else [],
                "automatic_retry_withheld": previous["automatic_retry_withheld"] if previous else [],
                "identity_blockers": proof["blockers"],
                "pending_identity_rows": [row for row in proof["rows"] if row["status"] not in
                                          {"resolved", "existing_identity_preserved"}]}
            result["recovery_checkpoint"] = _save(output, "recovery-checkpoint-" + uuid.uuid4().hex + ".json", {
                "schema": 1, "kind": "roster_input_checkpoint_v1", "accepted_pointer": binding["accepted_pointer"],
                "store_root": store.root.as_posix(), "store_head": state["head"]})
            acknowledged = _persist(store, persist_checkpoint, result["recovery_checkpoint"])
            if acknowledged:
                result["recovery_checkpoint"] = acknowledged["checkpoint"]
                result["retained_checkpoint_acknowledgment"] = acknowledged
            terminal = _save(output, "terminal-" + uuid.uuid4().hex + ".json", result)
            return dict(result, terminal=terminal)
        _require(callable(persist_checkpoint), "Persistent preclaim checkpoint callback required for synthetic collection")
        ordinal = len(chain) + 1
        phase = output / ("collection-" + str(ordinal))
        source_id = root["plan_id"] + "-source-" + str(ordinal)
        if phase.exists():
            # Never resume the broker, even when a crash preceded its first charge.
            _require(recovering, "Existing collection preparation cannot be executed twice")
            claim = store.claimed_authority(source_id)
            preclaim = _load(file_pin(phase / "preclaim.json"))
            _require(preclaim["binding"] == binding_pin and preclaim["plan"] == claim["plan_pin"]
                     and preclaim["review"] == claim["review_pin"], "Consumed collection recovery binding drift")
        else:
            _directory(phase)
            _require(source_id not in state["consumed_plans"], "Source stage is already consumed")
            source = {"kind": "roster_source_collection_v1", "plan_id": source_id,
                "store_root": store.root.as_posix(), "execution_mode": "synthetic", "publication_requested": False,
                "membership": any(r["url"] == MEMBERSHIP_URL for r in requests),
                "documents": [r["url"] for r in requests if r["url"] != MEMBERSHIP_URL],
                "max_requests": len(requests), "challenge": {"enabled": False},
                "user_agent": "ufc-roster-disabled-saved-controller/1",
                "observed_at": now.isoformat(), "issued_at": now.isoformat(), "expires_at": root["expires_at"]}
            plan_pin = _save(phase, "plan.json", source)
            # Reproduce semantics before creating the synthetic-only broker review.
            _, current, _, _, checked = _cycle_state(store, binding, _time(clock()))
            _require(checked == requests and current["head"] == state["head"],
                     "Controller accounting or semantic needs changed before source claim")
            review_pin = _save(phase, "review.json", {"status": "passed", "blockers": [], "plan": plan_pin,
                "scope": "saved_synthetic_only", "binding": binding_pin, "source_requests_authorized": 0,
                "activation_authorized": False, "publication_authorized": False})
            checkpoint = _save(phase, "preclaim.json", {"binding": binding_pin, "plan": plan_pin,
                "review": review_pin, "store_head": state["head"], "replay_authorized": False})
            # The harness/installed volume must persist these original bytes before a claim.
            acknowledged = _persist(store, persist_checkpoint, checkpoint)
            if acknowledged:
                checkpoint_pin = acknowledged["checkpoint"]
                expected_checkpoint_sha256 = checkpoint_pin["sha256"]
            _protected(binding)
            _require(store.snapshot()["head"] == state["head"], "Accounting changed during preclaim checkpoint")
            store.claim_plan(source_id, plan_pin, review_pin, now=_time(clock()))
            broker = SourceBroker(store, plan_pin, review_pin,
                synthetic_wire=synthetic_helper.exchange(plan_pin, review_pin) if synthetic_helper else wire,
                protected_verifier=lambda: _protected(binding), clock=clock)
            try:
                collected = broker.collect()
                receipt = _save(phase, "collection-complete.json", collected)
            except StoreError as exc:
                receipt = _save(phase, "collection-failed.json", {"error": str(exc), "source_requests": 0,
                    "replay_authorized": False, "store_head": store.snapshot()["head"]})
                # Fully received non-200 evidence can still yield explicit pending needs.
                # Missing/uncertain receipts are rejected by the stage adapter below.
            acknowledged = _persist(store, persist_checkpoint, receipt)
            if acknowledged:
                checkpoint_pin = acknowledged["checkpoint"]
                expected_checkpoint_sha256 = checkpoint_pin["sha256"]
        _protected(binding)
        stage_output = output / ("stage-" + str(ordinal) + "-" + uuid.uuid4().hex)
        stage_time = _time(clock())
        try:
            stage = prepare_next_stage(store, binding["root_plan"], binding["root_review"], source_id,
                stage_output, now=stage_time)
        except (StoreError, ValueError):
            # Only independently complete authoritative membership failure is
            # eligible. Unknown/missing delivery and other phase errors escape.
            if (store.snapshot()["uncertain_requests"]
                    or not (stage_output / "membership-input.json").is_file()
                    or not (stage_output / "complete-catalog.json").is_file()):
                raise
            from src.data.roster_hosted_recurrence import verify_membership_failure
            prior_inputs = previous["input_pins"] if previous else root["content"]["input_pins"]
            _require(len(requests) == 1 and requests[0]["url"] == MEMBERSHIP_URL,
                     "Only membership-only semantic failure has a reconciled data outcome")
            failed_inputs = {key: prior_inputs[key] for key in BASE_ROLES}
            failed_inputs["membership"] = file_pin(stage_output / "membership-input.json")
            failed_inputs["catalog"] = file_pin(stage_output / "complete-catalog.json")
            discovery = _save(stage_output, "failure-discovery.json", {
                "schema": 1, "kind": "complete_membership_failure_v1",
                "root_plan": binding["root_plan"], "root_review": binding["root_review"],
                "source_plan_id": source_id, "prepared_at": stage_time.isoformat(),
                "input_pins": failed_inputs})
            outcome = verify_membership_failure(store, discovery, now=_time(clock()))
            failure_pin = _save(output, "reconciled-membership-failure.json", _load(discovery))
            acknowledged = _persist(store, persist_checkpoint, failure_pin)
            result = {"schema": 1, "status": "cycle_failed_membership_reconciled_execution_disabled",
                "binding": binding_pin, "root_plan": binding["root_plan"], "root_review": binding["root_review"],
                "failure": failure_pin, "outcome": outcome, "store_head": store.snapshot()["head"],
                "retained_checkpoint_acknowledgment": acknowledged,
                "source_requests": 0, "activation_authorized": False, "publication_authorized": False,
                "schedule_enabled": False, "replay_authorized": False}
            return dict(result, terminal=_save(output, "terminal-" + uuid.uuid4().hex + ".json", result))
        acknowledged = _persist(store, persist_checkpoint, stage["stage"])
        if acknowledged:
            checkpoint_pin = acknowledged["checkpoint"]
            expected_checkpoint_sha256 = checkpoint_pin["sha256"]
    raise StoreError("Finite controller phase bound exhausted")


def prepare_controller(store, accepted_pointer_pin, output_directory, *, expected_pointer_sha256,
                       expected_release_sha256, checkpoint_pin, expected_checkpoint_sha256,
                       policy_pin, expected_policy_sha256, synthetic_responses_pin=None,
                       persist_checkpoint=None, recovery_evidence_pin=None,
                       synthetic_helper=None,
                       adoption_configuration_pin=None, expected_adoption_configuration_sha256=None,
                       clock=lambda: datetime.now(timezone.utc)):
    """Select, derive and issue once; optionally drive only pinned saved responses."""
    from src.data.roster_hosted_inputs import prepare_starting_inputs
    output = _directory(output_directory)
    try:
        if (store.mutation_guard is not None or synthetic_helper is not None
                or getattr(persist_checkpoint, "strict_checkpoint_ack", False)):
            _require(store.mutation_guard is persist_checkpoint
                     and getattr(persist_checkpoint, "strict_checkpoint_ack", False),
                     "Connected controller requires independent checkpoint custody before issuance")
            _require(persist_checkpoint.controller_configuration() == {
                "policy": policy_pin, "synthetic_responses": synthetic_responses_pin},
                "Controller policy or fixture differs from independently provisioned custody controls")
        now = _time(clock())
        # Refuse ineligible calendar invocations before expensive seed validation.
        # This grants no slot: the issuer rechecks policy and current time later.
        _slot(_policy(policy_pin, expected_policy_sha256), now, recovery_evidence_pin)
        selection = prepare_starting_inputs(store, accepted_pointer_pin, output / "starting-inputs",
            expected_pointer_sha256=expected_pointer_sha256, expected_release_sha256=expected_release_sha256,
            checkpoint_pin=checkpoint_pin, expected_checkpoint_sha256=expected_checkpoint_sha256, now=now,
            root_evidence=_discover_roots(store, output.parent),
            adoption_configuration_pin=adoption_configuration_pin,
            expected_adoption_configuration_sha256=expected_adoption_configuration_sha256)
        inputs = _load(selection["inputs_manifest"])["input_pins"]
        binding = {"schema": 1, "mode": "disabled_controller_v1", "store_root": store.root.as_posix(),
            "output_root": output.as_posix(), "accepted_pointer": accepted_pointer_pin,
            "checkpoint": checkpoint_pin, "expected_release_sha256": expected_release_sha256,
            "selection": selection, "base_inputs": {key: inputs[key] for key in BASE_ROLES},
            "synthetic_responses": synthetic_responses_pin,
            "activation_authorized": False, "publication_authorized": False}
        if adoption_configuration_pin is not None:
            binding["adoption_configuration"] = adoption_configuration_pin
            binding["expected_adoption_configuration_sha256"] = expected_adoption_configuration_sha256
        if synthetic_helper is not None:
            _require(synthetic_responses_pin is not None
                     and synthetic_helper.synthetic_responses_pin == synthetic_responses_pin,
                     "Exact saved helper response fixture required")
            binding["synthetic_helper_required"] = True
        if synthetic_responses_pin is not None:
            _saved_wire(synthetic_responses_pin)
            _require(callable(persist_checkpoint), "Persistent checkpoint callback required before issuance")
        _protected(binding)
        plans = output / "plans"
        plans.mkdir(mode=0o700)
        _fsync_dir(output)
        _require(store.snapshot()["head"] == selection["store_head"], "Accounting changed after input selection")
        root = issue_candidate_plan(store, plans, policy_pin, expected_policy_sha256=expected_policy_sha256,
            input_pins=inputs, proposed_requests=selection["proposed_requests"],
            recovery_evidence_pin=recovery_evidence_pin, now=_time(clock()))
        binding.update(root_plan=root["plan"], root_review=root["review"])
        binding_pin = _save(output, "binding.json", binding)
        acknowledged = _persist(store, persist_checkpoint, binding_pin)
        return _drive(store, binding_pin, clock=clock, persist_checkpoint=persist_checkpoint, recovering=False,
            checkpoint_pin=acknowledged["checkpoint"] if acknowledged else None,
            expected_checkpoint_sha256=acknowledged["checkpoint"]["sha256"] if acknowledged else None,
            synthetic_helper=synthetic_helper)
    except BaseException as exc:
        _save(output, "failed-" + uuid.uuid4().hex + ".json", {"error": type(exc).__name__ + ": " + str(exc),
            "source_requests": 0, "replay_authorized": False, "activation_authorized": False})
        raise


def recover_controller(store, binding_pin, *, expected_binding_sha256, persist_checkpoint=None,
                       checkpoint_pin, expected_checkpoint_sha256,
                       synthetic_helper=None,
                       clock=lambda: datetime.now(timezone.utc)):
    """Recover complete receipts within original lifetime; never restart a broker."""
    verify_pin(binding_pin)
    _require(binding_pin["sha256"] == expected_binding_sha256, "External controller binding hash mismatch")
    _require(checkpoint_pin is not None and expected_checkpoint_sha256 is not None,
             "Current externally retained recovery checkpoint required")
    if store.mutation_guard is not None or getattr(persist_checkpoint, "strict_checkpoint_ack", False):
        _require(store.mutation_guard is persist_checkpoint
                 and getattr(persist_checkpoint, "strict_checkpoint_ack", False),
                 "Recovery requires connected independent custody")
        current = persist_checkpoint.current()
        _require(current["checkpoint"] == checkpoint_pin
                 and checkpoint_pin["sha256"] == expected_checkpoint_sha256,
                 "Recovery checkpoint is not the exact latest custodian head")
    return _drive(store, binding_pin, clock=clock, persist_checkpoint=persist_checkpoint, recovering=True,
                  checkpoint_pin=checkpoint_pin, expected_checkpoint_sha256=expected_checkpoint_sha256,
                  synthetic_helper=synthetic_helper)


def recover_retained_controller(store, binding_pin, *, expected_binding_sha256, custodian,
                                synthetic_helper=None, clock=lambda: datetime.now(timezone.utc)):
    """Obtain the recovery head from its independent owner, never a caller filename."""
    _require(store.mutation_guard is custodian and getattr(custodian, "strict_checkpoint_ack", False),
             "Exact connected recovery custodian required")
    current = custodian.current()
    return recover_controller(store, binding_pin, expected_binding_sha256=expected_binding_sha256,
        persist_checkpoint=custodian, checkpoint_pin=current["checkpoint"],
        expected_checkpoint_sha256=current["checkpoint"]["sha256"],
        synthetic_helper=synthetic_helper, clock=clock)
