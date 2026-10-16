"""Reconcile disabled recurring cycles from original charged evidence.

No terminal status grants an outcome. Complete membership failures retain their
original receipts but never replace valid starting inputs. Pending identities
are an expected data state, not failure of an otherwise valid membership refresh.
This module does not publish data, register a monthly model source or claim work.
"""
from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from src.data.roster_hosted_needs import BASE_ROLES, MEMBERSHIP_URL, derive_need_evidence
from src.data.roster_hosted_plan import _policy, _slot
from src.data.roster_hosted_stages import _receipts, validate_stage_chain
from src.data.roster_hosted_store import StoreError, _read, _require, _time, verify_pin


def _load(pin):
    return json.loads(_read(verify_pin(pin)["path"]))


def verify_membership_failure(store, failure_pin, *, now):
    """Recompute one complete, unadvanced membership-only failed collection.

The evidence descriptor is discovery only. Original reservation, all source
claims/charges/results and both before/after semantic inputs are revalidated.
Incomplete delivery, missing claims and unexpected input corruption raise.
"""
    from src.data import roster_hosted_candidate as candidate
    from src.data import rosterwatch_active_roster as ordinary
    from src.data.roster_hosted_candidate import CandidateError
    from src.data.rosterwatch_site import RosterWatchSiteError, parse_rosterwatch_site_roster, rosterwatch_csv_bytes
    from src.data.roster_hosted_store import _sha

    failure = _load(failure_pin)
    _require(set(failure) == {"schema", "kind", "root_plan", "root_review", "source_plan_id",
                            "prepared_at", "input_pins"}
             and failure["schema"] == 1 and failure["kind"] == "complete_membership_failure_v1",
             "Exact original failed membership descriptor required")
    prepared = _time(failure["prepared_at"])
    _require(prepared <= _time(now), "Future failed membership evidence")
    root = _load(failure["root_plan"])
    validate_stage_chain(store, failure["root_plan"], failure["root_review"], now=prepared)
    state = store.snapshot()
    _require(not state["uncertain_requests"], "Uncertain accounting is not a reconciled failure")
    chain = state["plan_stages"].get(root["slot_id"], [])
    _require(failure["source_plan_id"] == root["plan_id"] + "-source-" + str(len(chain) + 1),
             "Failure must bind the exact original next source phase")
    previous = _load(chain[-1]["stage_pin"]) if chain else None
    prior = previous["input_pins"] if previous else root["content"]["input_pins"]
    requests = previous["proposed_requests"] if previous else root["content"]["proposed_requests"]
    since = _time(previous["prepared_at"] if previous else root["issued_at"])
    _require(len(requests) == 1 and requests[0]["url"] == MEMBERSHIP_URL,
             "Only a complete authoritative membership failure can fail this data cycle")
    _require(all(_load(row["stage_pin"])["source_plan_id"] != failure["source_plan_id"] for row in chain),
             "Failed membership phase was already advanced")
    receipts, captures, membership = _receipts(store, state, failure["source_plan_id"], requests,
                                               since=since, now=prepared)
    _require(membership is not None and captures == [], "Exact membership-only receipts required")
    base = failure["input_pins"]
    _require(set(base) == BASE_ROLES and all(base[key] == prior[key] for key in ("owners", "release")),
             "Failed collection changed accepted owner or release authority")
    _require(_load(base["catalog"]) == _load(prior["catalog"]), "Membership failure changed capture history")
    descriptor = _load(base["membership"])
    _require(set(descriptor) == {"schema", "body", "envelope"} and descriptor["schema"] == 1
             and descriptor["body"] == membership[0] and _load(descriptor["envelope"]) == membership[1],
             "Failed membership inputs differ from complete original receipt")
    # First prove the unchanged original authority. Only the new membership
    # differs during the second derivation; generic I/O/runtime errors escape.
    prior_base = {key: prior[key] for key in BASE_ROLES}
    derive_need_evidence(prior_base, now=since)
    owners = _load(prior_base["owners"])
    seed = Path(owners["seed_root"])
    accepted_state = ordinary._read_previous_state(seed / "rosterwatch" / candidate.STATE_FILE)
    previous_count = len(candidate._records(seed / "rosterwatch" / ordinary.ROSTERWATCH_SOURCE_LEDGER_FILENAME))
    raw = _read(verify_pin(descriptor["body"])["path"])
    try:
        # Restrict failure classification to authoritative membership validation;
        # no identity resolution, owner validation or catalog work occurs here.
        _, requested, observed = candidate._membership({"membership_envelope": descriptor["envelope"],
            "evidence_mode": "offline_regression"}, raw)
        _require(requested <= observed <= prepared, "Future membership")
        _require(observed.date().isoformat() >= accepted_state["source_cutoff"], "Membership cutoff regressed")
        rows, _ = parse_rosterwatch_site_roster(raw)
        snapshot = _sha(rosterwatch_csv_bytes(rows))
        _require(observed.date().isoformat() != accepted_state["source_cutoff"]
                 or snapshot == accepted_state["snapshot_sha256"], "Same-cutoff membership conflicts")
        _require(len(rows) <= previous_count * ordinary.ROSTERWATCH_RECURRING_GROWTH_RATIO,
                 "Membership growth guard failed")
        _require(not (len(rows) < previous_count * ordinary.ROSTERWATCH_RECURRING_SHRINK_RATIO
                     and previous_count-len(rows) >= ordinary.ROSTERWATCH_RECURRING_SHRINK_ABSOLUTE),
                 "Membership shrink guard failed")
    except (StoreError, CandidateError, RosterWatchSiteError) as exc:
        error = type(exc).__name__ + ": " + str(exc)
    else:
        raise StoreError("Valid membership cannot be relabelled as a failed data cycle")
    _require(store.snapshot() == state, "Accounting changed during failure reconciliation")
    return {"status": "failed", "slot_id": root["slot_id"], "source_plan_id": failure["source_plan_id"],
            "root_plan": failure["root_plan"], "root_review": failure["root_review"],
            "prior_inputs": prior_base, "receipts": receipts, "prepared_at": failure["prepared_at"],
            "reason": error, "retry_authorized": False, "publication_authorized": False}


def validated_cycle_status(store, binding_pin, *, now):
    """Return only a recomputed completed cycle result; unfinished work blocks."""
    from src.data.roster_hosted_store import file_pin
    binding = _load(binding_pin)
    _require(binding.get("mode") == "disabled_controller_v1" and binding["store_root"] == store.root.as_posix(),
             "Original controller binding required")
    root = _load(binding["root_plan"])
    failed_path = Path(binding["output_root"]) / "reconciled-membership-failure.json"
    if failed_path.exists():
        failed = verify_membership_failure(store, file_pin(failed_path), now=now)
        _require(failed["root_plan"] == binding["root_plan"] and failed["root_review"] == binding["root_review"],
                 "Failure belongs to another original cycle")
        return failed
    state = store.snapshot()
    _require(not state["uncertain_requests"], "Uncertain work cannot supply a calendar outcome")
    chain = state["plan_stages"].get(root["slot_id"], [])
    last = _load(chain[-1]["stage_pin"]) if chain else None
    original_time = _time(last["prepared_at"] if last else root["issued_at"])
    _require(original_time <= _time(now), "Future original cycle")
    validate_stage_chain(store, binding["root_plan"], binding["root_review"], now=original_time)
    requests = last["proposed_requests"] if last else root["content"]["proposed_requests"]
    _require(requests == [], "Unfinished original cycle requires original recovery")
    pins = last["input_pins"] if last else root["content"]["input_pins"]
    proof = derive_need_evidence({key: pins[key] for key in BASE_ROLES}, now=original_time)
    _require(proof["phase"] != "membership_refresh_required", "Stale membership is not data success")
    _require(store.snapshot() == state, "Accounting changed during outcome verification")
    return {"status": "succeeded", "slot_id": root["slot_id"], "root_plan": binding["root_plan"],
            "root_review": binding["root_review"], "pending_identity_rows": [row for row in proof["rows"]
                if row["status"] not in {"resolved", "existing_identity_preserved"}],
            "input_pins": {key: pins[key] for key in BASE_ROLES}, "publication_authorized": False}


def validate_successor_cycle(store, previous_binding_pin, policy_pin, *, now, recovery_evidence_pin=None,
                             adoption_configuration_pin=None, expected_adoption_configuration_sha256=None):
    """Apply existing Tuesday/conditional-Wednesday rules to retained originals.

This is the data-only decision boundary: no mutable ordinary schedule or monthly
registration is touched. Existing atomic store slot reservations remain the
sole issuer authority. A supplied failed/missed status is never trusted.
"""
    _require(recovery_evidence_pin is None, "Successor recovery evidence must be independently derived")
    now = _time(now)
    policy = _policy(policy_pin, policy_pin["sha256"])
    from src.data.roster_hosted_controller import _discover_roots
    from src.data.roster_hosted_inputs import _recover_cycles
    binding = _load(previous_binding_pin)
    accepted = _load(_load(binding["accepted_pointer"])["input_manifest"])["input_pins"]
    initial_state = store.snapshot()
    adoption_context = None
    configured = binding.get("adoption_configuration")
    if configured is not None:
        _require(adoption_configuration_pin in (None, configured), "Successor adoption configuration changed")
        adoption_configuration_pin = configured
        pinned_hash = binding["expected_adoption_configuration_sha256"]
        _require(expected_adoption_configuration_sha256 in (None, pinned_hash), "Successor adoption authority changed")
        expected_adoption_configuration_sha256 = pinned_hash
    if adoption_configuration_pin is not None:
        from src.data.roster_refresh_adoption import resolve_adoption_context
        adoption_context = resolve_adoption_context(store, adoption_configuration_pin,
            expected_configuration_sha256=expected_adoption_configuration_sha256, now=now)
        _require(Path(binding["output_root"]).parent == Path(adoption_context["configuration"]["controller_history_root"]),
                 "Original successor history differs from trusted adoption namespace")
        accepted = adoption_context["genesis_inputs"]
    else:
        _require(expected_adoption_configuration_sha256 is None, "Adoption hash requires its configuration")
    _, lineage, excluded, _ = _recover_cycles(store, initial_state, accepted, now,
        _discover_roots(store, Path(binding["output_root"]).parent), adoption_context=adoption_context)
    _require(lineage and not excluded and lineage[-1]["root_plan"] == binding["root_plan"],
             "Successor must follow latest complete original lineage")
    prior = validated_cycle_status(store, previous_binding_pin, now=now)
    state = store.snapshot()
    _require(not state["uncertain_requests"], "Uncertain accounting blocks every later cycle")
    local = now.astimezone(ZoneInfo(policy["timezone"]))
    recovery = None
    if local.weekday() == policy["primary_weekday"]:
        slot = _slot(policy, now, None)
    else:
        _require(local.weekday() == policy["recovery_weekday"], "No current data calendar slot")
        start = local.replace(hour=policy["hour"], minute=0, second=0, microsecond=0)
        _require(0 <= (local-start).total_seconds() < policy["slot_lateness_seconds"], "Outside recovery window")
        previous = (local-timedelta(days=1)).strftime("%Y%m%d") + "-primary"
        slot = local.strftime("%Y%m%d") + "-recovery"
        if previous in state["plan_slots"]:
            _require(prior["slot_id"] == previous and prior["status"] == "failed",
                     "Wednesday requires independently completed failed Tuesday; success suppresses recovery")
            status = "failed"
        else:
            status = "missed"
        recovery = {"slot_id": previous, "status": status, "accounting_reconciled": True,
                    "uncertain_requests": [], "retry_authorized": False,
                    "original_previous_binding": previous_binding_pin, "store_head": state["head"]}
    _require(slot not in state["plan_slots"], "Current data slot is already consumed")
    _require(slot > prior["slot_id"], "Successor must be a later original calendar slot")
    _require(store.snapshot() == state, "Accounting changed during successor calendar selection")
    return {"slot_id": slot, "previous_status": prior["status"], "recovery_evidence": recovery}
