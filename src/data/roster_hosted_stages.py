"""Disabled machine advancement of membership/listing/profile evidence phases.

One original server slot owns every phase. Exact durable synthetic receipts
advance its input manifest; no caller supplies a new URL list or deletes an old
capture. The stage chain is consumed durably and has one nonrefundable finite
worst-case preparation budget. Neither root nor subordinate plans grant sends.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from src.data.roster_hosted_needs import (
    BASE_ROLES, MEMBERSHIP_URL, prepare_need_manifest, validate_need_evidence,
)
from src.data.roster_hosted_plan import _policy, _requests, _slot
from src.data.roster_hosted_store import (
    _fsync_dir, _json, _new_file, _read, _require, _safe, _sha, _time, file_pin, verify_pin,
)


def _load(pin):
    verify_pin(pin)
    return json.loads(_read(pin["path"]))


def _write(output, name, value):
    path = output / name
    _new_file(path, value if isinstance(value, bytes) else _json(value), immutable=True)
    return file_pin(path)


def _root(store, root_pin, review_pin, now):
    root, review = _load(root_pin), _load(review_pin)
    state = store.snapshot()
    _require(root.get("mode") == "disabled_preparation" and root.get("network_requests") == 0
             and root.get("activation_authorized") is False and root.get("publication_requested") is False
             and root["content"].get("store_root") == store.root.as_posix(),
             "Exact disabled root must belong to this accounting store")
    slot = state["plan_slots"].get(root["slot_id"])
    _require(slot == root["slot_reservation"] and slot is not None
             and slot["content_sha256"] == _sha(_json(root["content"]))
             and root["plan_id"] == root["slot_id"] + "-" + slot["nonce"],
             "Original root slot/content binding drift")
    _require(review.get("plan") == root_pin and review.get("semantic_needs_verified") is True
             and review.get("source_requests_authorized") == 0
             and review.get("activation_authorized") is False
             and review.get("publication_authorized") is False
             and review.get("status") == "preparation_validated_execution_disabled"
             and review.get("store_head", {}).get("sha256") == slot["sha256"],
             "Original independently recomputed disabled semantic review required")
    _require(_time(root["issued_at"]) <= now <= _time(root["expires_at"]),
             "Original root lifetime expired; no new slot alias")
    _require(not state["uncertain_requests"], "Uncertain delivery prevents stage advancement")
    policy_pin = root["content"]["policy_pin"]
    policy = _policy(policy_pin, policy_pin["sha256"])
    _require(policy["store_policy"] == state["policy"], "Root stage/store policy drift")
    issued, expires = _time(root["issued_at"]), _time(root["expires_at"])
    cost, rw = _requests(root["content"]["proposed_requests"], policy)
    before = root["content"]["accounting_before"]
    prior_total = state["total_charged"] - len(state["requests"])
    prior_rw = state["rosterwatch_charged"] - sum(r["source"] == "roster.watch" for r in state["requests"].values())
    _require(set(root) == {"schema", "mode", "plan_id", "slot_id", "nonce", "issued_at", "expires_at",
                          "network_requests", "publication_requested", "activation_authorized",
                          "slot_reservation", "content", "maximum_future_charges"}
             and root["schema"] == 1 and type(root["network_requests"]) is int
             and root["nonce"] == slot["nonce"] and root["issued_at"] == slot["at_utc"]
             and (expires - issued).total_seconds() == policy["plan_ttl_seconds"]
             and _slot(policy, issued, root["content"]["recovery_evidence_pin"]) == root["slot_id"]
             and type(root["maximum_future_charges"]) is int and root["maximum_future_charges"] == cost
             and before["sha256"] == slot["previous"] and before["sequence"] + 1 == slot["sequence"]
             and before["manifest_sha256"] == state["head"]["manifest_sha256"]
             and root["content"]["total_charged"] == prior_total + sum(
                 charge["sequence"] < slot["sequence"] for charge in state["requests"].values())
             and root["content"]["rosterwatch_charged"] == prior_rw + sum(
                 charge["sequence"] < slot["sequence"] and charge["source"] == "roster.watch"
                 for charge in state["requests"].values())
             and review["policy"] == policy_pin and review["store_head"] == {
                 "sequence": slot["sequence"], "sha256": slot["sha256"],
                 "manifest_sha256": before["manifest_sha256"]},
             "Original root lifetime, cost, nonce or accounting binding changed")
    validate_need_evidence(root["content"]["input_pins"], root["content"]["proposed_requests"],
                           now=_time(root["issued_at"]))
    return root, state, policy


def _receipts(store, state, source_plan_id, proposed_requests, *, since, now):
    """Read every result for one exact consumed synthetic source collection."""
    from src.data.ufcstats_identity_access import POST_URL, _document_supported, is_challenge

    claim = store.claimed_authority(source_plan_id)
    source_plan, source_review = _load(claim["plan_pin"]), _load(claim["review_pin"])
    _require(source_plan.get("kind") == "roster_source_collection_v1"
             and source_plan.get("execution_mode") == "synthetic"
             and source_plan.get("publication_requested") is False
             and source_plan.get("store_root") == store.root.as_posix()
             and source_plan.get("plan_id") == source_plan_id
             and source_review.get("status") == "passed" and source_review.get("blockers") == []
             and source_review.get("plan") == claim["plan_pin"],
             "Only exact consumed synthetic broker evidence may advance disabled stages")
    issued, expires = _time(source_plan.get("issued_at")), _time(source_plan.get("expires_at"))
    _require(since <= issued <= now and 0 < (expires - issued).total_seconds() <= 86400,
             "Original source collection lifetime or predecessor time changed")
    wanted = {request["url"] for request in proposed_requests}
    _require(bool(wanted), "Completed preparation has no collection phase to replay")
    planned = set(source_plan.get("documents", []))
    if source_plan.get("membership") is True:
        planned.add(MEMBERSHIP_URL)
    _require(planned == wanted and len(source_plan.get("documents", [])) == len(planned - {MEMBERSHIP_URL}),
             "Collected source plan differs from machine-derived prior needs")
    challenge = source_plan.get("challenge", {})
    _require(type(source_plan.get("membership")) is bool and type(challenge.get("enabled")) is bool
             and type(source_plan.get("max_requests")) is int
             and source_plan["max_requests"] == len(planned) + 2 * int(challenge["enabled"])
             and 0 < source_plan["max_requests"] <= state["policy"]["plan_ceiling"],
             "Exact finite optional-branch source scope required")
    if challenge["enabled"]:
        verify_pin(challenge["template_pin"])
    records, captures, membership = [], [], None
    for request_id, result in state["results"].items():
        outer = _load(result["envelope"]) if "envelope" in result else None
        if outer is None:
            continue
        envelope = outer["metadata"]
        if envelope.get("plan_id") != source_plan_id:
            continue
        request, response = envelope.get("request", {}), envelope.get("response", {})
        _require(envelope.get("plan") == claim["plan_pin"] and envelope.get("review") == claim["review_pin"]
                 and request.get("request_id") == request_id and request.get("method") == outer["method"]
                 and request.get("url") == outer["url"] and outer["request_id"] == request_id
                 and result["outcome"] == outer["outcome"] == "complete",
                 "Receipt does not bind the exact durable source charge and claim")
        raw = _read(verify_pin(result["blob"])["path"])
        started, completed = _time(request.get("started_at")), _time(response.get("completed_at"))
        _require(request.get("started_at") == state["requests"][request_id]["at_utc"]
                 and response.get("completed_at") == result["at_utc"],
                 "Receipt timestamps differ from durable charge/result events")
        _require(outer["raw_body_sha256"] == response.get("body_sha256") == _sha(raw)
                 and response.get("complete") is True and not response.get("error")
                 and since <= started <= completed <= now,
                 "Incomplete, future or old receipt cannot advance current stage")
        _require(issued <= started <= completed < expires,
                 "Receipt escaped original source collection lifetime")
        url = request["url"]
        if request["method"] == "POST":
            _require(url == POST_URL and source_plan.get("challenge", {}).get("enabled") is True,
                     "Unexpected conditional challenge receipt")
        else:
            _require(request["method"] == "GET" and url in wanted, "Unexpected receipt URL or method")
            if url == MEMBERSHIP_URL:
                _require(membership is None, "Repeated membership receipt cannot advance a stage")
                membership = (result["blob"], envelope)
            else:
                try:
                    raw.decode("utf-8", errors="strict")
                    decoded = True
                except UnicodeDecodeError:
                    decoded = False
                captures.append({"url": url, "status_code": response["status"],
                    "body_base64": base64.b64encode(raw).decode("ascii"), "decoded_sha256": _sha(raw),
                    "decoded_size": len(raw), "captured_at": response["completed_at"],
                    "capture_complete": True, "decode_complete": decoded,
                    "ufcstats_challenge_detected": is_challenge(raw),
                    "semantic_evidence_eligible": decoded and response["status"] == 200
                        and _document_supported(url, raw, response["headers"]),
                    "error": response.get("error")})
        records.append({"request_id": request_id, "receipt": result})
    _require(records and {(_load(row["receipt"]["envelope"])["method"],
                          _load(row["receipt"]["envelope"])["url"]) for row in records
                         if _load(row["receipt"]["envelope"])["method"] == "GET"}
             == {("GET", url) for url in wanted}, "Complete prior phase receipt coverage required")
    _require({row["request_id"] for row in records} == {key for key, charge in state["requests"].items()
                                                       if charge["plan_id"] == source_plan_id},
             "Every phase charge requires complete retained response evidence")
    _require(len(records) <= source_plan["max_requests"], "Receipt phase exceeded finite source scope")
    methods = [_load(row["receipt"]["envelope"])["method"] for row in records]
    _require(methods.count("POST") <= 1 and (not methods.count("POST") or challenge["enabled"]),
             "Repeated conditional challenge evidence")
    return records, captures, membership


def _select(proof, attempted, remaining):
    unmet = proof["proposed_requests"] + proof["deferred_requests"]
    available = [r for r in unmet if r["url"] not in attempted]
    withheld = [r for r in unmet if r["url"] in attempted]
    proposed, deferred = [], []
    for request in available:
        cost = 1 if request["source"] == "roster.watch" else 3
        if cost <= remaining:
            proposed.append(request)
            remaining -= cost
        else:
            deferred.append(request)
    return proposed, deferred, withheld


def validate_stage_chain(store, root_plan_pin, root_review_pin, *, now=None):
    """Recompute every saved transition, including receipts and deferred needs."""
    now = _time(now)
    root, state, policy = _root(store, root_plan_pin, root_review_pin, now)
    previous, attempted, reserved, collections = None, set(), root["maximum_future_charges"], set()
    for record in state["plan_stages"].get(root["slot_id"], []):
        stage = _load(record["stage_pin"])
        prepared = _time(stage["prepared_at"])
        prior_inputs = previous["input_pins"] if previous else root["content"]["input_pins"]
        prior_requests = previous["proposed_requests"] if previous else root["content"]["proposed_requests"]
        since = _time(previous["prepared_at"] if previous else root["issued_at"])
        _require(stage["source_plan_id"] not in collections and stage["root_plan_pin"] == root_plan_pin
                 and stage["root_review_pin"] == root_review_pin and stage["store_root"] == store.root.as_posix()
                 and prepared <= now and stage["publication_authorized"] is False,
                 "Stage predecessor, authority or collection replay mismatch")
        collections.add(stage["source_plan_id"])
        receipts, captures, membership = _receipts(store, state, stage["source_plan_id"], prior_requests,
                                                   since=since, now=prepared)
        _require(stage["receipts"] == receipts, "Stage omitted or altered complete receipt evidence")
        base = stage["input_pins"]
        _require(all(base[key] == prior_inputs[key] for key in ("owners", "release")),
                 "Stage changed complete ownership or release authority")
        old_catalog = _load(prior_inputs["catalog"])
        _require(_load(base["catalog"]) == dict(old_catalog, captures=old_catalog["captures"] + captures),
                 "Stage removed/reordered old captures or substituted new evidence")
        if membership is None:
            _require(base["membership"] == prior_inputs["membership"], "Stage changed uncollected membership")
        else:
            descriptor = _load(base["membership"])
            _require(descriptor == {"schema": 1, "body": membership[0], "envelope": descriptor["envelope"]}
                     and _load(descriptor["envelope"]) == membership[1], "Stage membership receipt mismatch")
        proof = _load(base["need_evidence"])
        validate_need_evidence(base, proof["proposed_requests"], now=prepared)
        attempted |= {r["url"] for r in prior_requests}
        # Bind selection to the true accounting snapshot at preparation, not a
        # later total that could retroactively change the historical decision.
        _require(type(stage["actual_total_charged"]) is int and stage["actual_total_charged"] >= 0,
                 "Stage actual accounting witness required")
        historical_total = root["content"]["total_charged"] + sum(
            root["slot_reservation"]["sequence"] < charge["sequence"] < record["sequence"]
            for charge in state["requests"].values())
        _require(stage["actual_total_charged"] == historical_total, "Stage charged accounting witness mismatch")
        remaining = min(policy["store_policy"]["plan_ceiling"] - reserved,
                        policy["store_policy"]["total_ceiling"] - historical_total - reserved)
        proposed, deferred, withheld = _select(proof, attempted, remaining)
        cost, rw = _requests(proposed, policy)
        _require(stage["attempted_urls"] == sorted(attempted) and stage["proposed_requests"] == proposed
                 and stage["budget_deferred_requests"] == deferred and stage["automatic_retry_withheld"] == withheld
                 and stage["maximum_future_charges"] == cost and stage["maximum_rosterwatch_charges"] == rw
                 and stage["pending_identity_rows"] == proof["blockers"], "Stage derived need or budget selection mismatch")
        reserved += cost
        previous = stage
    return {"status": "stage_chain_validated_execution_disabled", "stages": len(collections),
            "maximum_reserved_charges": reserved, "source_requests_authorized": 0,
            "activation_authorized": False, "publication_authorized": False}


def prepare_next_stage(store, root_plan_pin, root_review_pin, source_plan_id, output_directory, *, now=None):
    """Advance one consumed root slot from durable receipts, saving failures.

    The collection ID identifies already consumed *synthetic* broker evidence;
    this function never runs that collection. No stage may repeat a source URL
    attempted earlier in the same root cycle. Pending fighters remain pending.
    """
    now = _time(now)
    output = _safe(output_directory)
    _require(not output.exists() and output.parent.is_dir(), "Fresh stage evidence directory required")
    output.mkdir(mode=0o700)
    _fsync_dir(output.parent)
    try:
        root, state, policy = _root(store, root_plan_pin, root_review_pin, now)
        validate_stage_chain(store, root_plan_pin, root_review_pin, now=now)
        chain = state["plan_stages"].get(root["slot_id"], [])
        previous = _load(chain[-1]["stage_pin"]) if chain else None
        _require(not any(_load(row["stage_pin"])["source_plan_id"] == source_plan_id for row in chain),
                 "Consumed collection phase cannot advance twice")
        prior_inputs = previous["input_pins"] if previous else root["content"]["input_pins"]
        prior_requests = previous["proposed_requests"] if previous else root["content"]["proposed_requests"]
        since = _time(previous["prepared_at"] if previous else root["issued_at"])
        records, captures, membership = _receipts(store, state, source_plan_id, prior_requests, since=since, now=now)
        base = {key: prior_inputs[key] for key in BASE_ROLES}
        old_catalog = _load(base["catalog"])
        base["catalog"] = _write(output, "complete-catalog.json", dict(old_catalog,
            captures=old_catalog["captures"] + captures))
        if membership is not None:
            body, envelope = membership
            envelope_pin = _write(output, "membership-envelope.json", envelope)
            base["membership"] = _write(output, "membership-input.json", {"schema": 1, "body": body,
                                                                        "envelope": envelope_pin})
        needs = prepare_need_manifest(base, output / "needs", now=now)
        input_pins = _load(needs["inputs_manifest"])["input_pins"]
        proof = validate_need_evidence(input_pins, needs["proposed_requests"], now=now)
        attempted = set(previous["attempted_urls"] if previous else []) | {r["url"] for r in prior_requests}
        reserved = root["maximum_future_charges"] + sum(r["maximum_charges"] for r in chain)
        remaining = min(state["policy"]["plan_ceiling"] - reserved,
                        state["policy"]["total_ceiling"] - state["total_charged"] - reserved)
        proposed, deferred, withheld = _select(proof, attempted, remaining)
        cost, rw = _requests(proposed, policy)
        ordinal = len(chain) + 1
        stage = {"schema": 1, "mode": "disabled_stage_preparation", "stage_id": root["plan_id"] + "-stage-" + str(ordinal),
            "ordinal": ordinal, "root_plan_pin": root_plan_pin, "root_review_pin": root_review_pin,
            "store_root": store.root.as_posix(), "prepared_at": now.isoformat(), "source_plan_id": source_plan_id,
            "previous_stage_sha256": chain[-1]["sha256"] if chain else root["slot_reservation"]["sha256"],
            "network_requests": 0, "activation_authorized": False, "publication_authorized": False,
            "input_pins": input_pins, "proposed_requests": proposed, "attempted_urls": sorted(attempted),
            "automatic_retry_withheld": withheld, "budget_deferred_requests": deferred, "receipts": records,
            "maximum_future_charges": cost, "maximum_rosterwatch_charges": rw,
            "actual_total_charged": state["total_charged"],
            "pending_identity_rows": proof["blockers"]}
        stage_pin = _write(output, "stage.json", stage)
        reservation = store.reserve_plan_stage(root_plan_pin, stage_pin, now=now)
        review = validate_stage_chain(store, root_plan_pin, root_review_pin, now=now)
        review_pin = _write(output, "review.json", review)
        # A failure here leaves the root-stage reservation consumed, never replayed.
        result = {"status": "stage_prepared_execution_disabled", "stage": stage_pin,
            "reservation": reservation, "review": review_pin, "source_requests_authorized": 0, "activation_authorized": False,
            "next_collection_needed": bool(proposed), "current_source_claim": False}
        _write(output, "terminal.json", result)
        return result
    except BaseException as exc:
        _write(output, "failed.json", {"status": "failed_execution_disabled", "error": type(exc).__name__ + ": " + str(exc),
            "replay_authorized": False, "source_requests_authorized": 0})
        raise
