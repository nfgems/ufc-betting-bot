"""Server-side plan preparation, with no scheduler or execution authority.

The server computes slots, fresh nonces, content/accounting bindings and a
machine review. This module accepts disabled policies only. Activating recurring
source collection requires a separately approved policy and integration; a
generated preparation review cannot authorize network, publication or betting.
"""
from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from src.data.roster_hosted_store import (
    StoreError, PROPOSED_POLICY, _fsync_dir, _json, _new_file, _read, _require,
    _safe, _sha, _time, file_pin, verify_pin,
)


_FLAGS = ("activation_authorized", "schedule_enabled", "source_network_enabled",
          "publication_enabled", "betting_enabled", "model_operations_enabled")
PROPOSED_RECURRING_POLICY = {
    "schema": 1, "mode": "disabled_preparation", **{key: False for key in _FLAGS},
    "timezone": "America/New_York", "primary_weekday": 1, "recovery_weekday": 2,
    "hour": 11, "slot_lateness_seconds": 86400, "plan_ttl_seconds": 3600,
    "max_proposed_requests": 12, "max_rosterwatch_requests": 1,
    "store_policy": dict(PROPOSED_POLICY),
    "publication_requested": False, "catch_up_multiple_missed_slots": False,
    "automatic_uncertain_send_retry": False,
}
_INPUT_ROLES = {"owners", "membership", "catalog", "release", "need_evidence"}


def _policy(pin, expected_sha256):
    verify_pin(pin)
    _require(pin["sha256"] == expected_sha256, "External policy hash mismatch")
    value = json.loads(_read(pin["path"]))
    _require(isinstance(value, dict) and set(value) == set(PROPOSED_RECURRING_POLICY),
             "Exact disabled recurring policy required")
    for key in _FLAGS + ("publication_requested", "catch_up_multiple_missed_slots",
                         "automatic_uncertain_send_retry"):
        _require(value[key] is False, "Recurring activation is not authorized")
    _require(value["schema"] == 1 and value["mode"] == "disabled_preparation"
             and value["timezone"] == "America/New_York"
             and value["primary_weekday"] == 1 and value["recovery_weekday"] == 2
             and value["hour"] == 11, "Unreviewed calendar policy")
    for key, maximum in (("slot_lateness_seconds", 86400), ("plan_ttl_seconds", 86400),
                         ("max_proposed_requests", 12), ("max_rosterwatch_requests", 1)):
        _require(type(value[key]) is int and 0 < value[key] <= maximum, "Invalid recurring limit")
    # The policy is compared to the already initialized store by both phases.
    from src.data.roster_hosted_store import _policy as store_policy
    store_policy(value["store_policy"])
    return value


def _slot(policy, now, recovery_evidence_pin):
    local = now.astimezone(ZoneInfo(policy["timezone"]))
    day = local.weekday()
    _require(day in {policy["primary_weekday"], policy["recovery_weekday"]},
             "No eligible current calendar slot; no catch-up")
    start = local.replace(hour=policy["hour"], minute=0, second=0, microsecond=0)
    _require(0 <= (local - start).total_seconds() < policy["slot_lateness_seconds"],
             "Outside current slot window")
    kind = "primary" if day == policy["primary_weekday"] else "recovery"
    slot_id = local.strftime("%Y%m%d") + "-" + kind
    if kind == "primary":
        _require(recovery_evidence_pin is None, "Primary cannot carry recovery override")
    else:
        _require(recovery_evidence_pin is not None, "Recovery requires pinned reconciliation")
        verify_pin(recovery_evidence_pin)
        proof = json.loads(_read(recovery_evidence_pin["path"]))
        previous = (local - timedelta(days=1)).strftime("%Y%m%d") + "-primary"
        _require(proof.get("slot_id") == previous and proof.get("status") in {"failed", "missed"}
                 and proof.get("accounting_reconciled") is True
                 and proof.get("uncertain_requests") == [] and proof.get("retry_authorized") is False,
                 "Recovery reconciliation is incomplete or authorizes a retry")
    return slot_id


def _inputs(pins):
    _require(isinstance(pins, dict) and set(pins) == _INPUT_ROLES,
             "Complete owners, membership, catalog, release and need evidence required")
    for pin in pins.values():
        verify_pin(pin)


def _requests(requests, policy):
    _require(isinstance(requests, list) and len(requests) <= policy["max_proposed_requests"],
             "Proposed request ceiling")
    seen, rosterwatch = set(), 0
    for request in requests:
        _require(isinstance(request, dict) and set(request) == {"request_id", "source", "method", "url"},
                 "Invalid proposed request")
        key = request["request_id"]
        _require(isinstance(key, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", key)
                 and key not in seen, "Repeated/invalid proposed request ID")
        seen.add(key)
        parsed = urlsplit(request["url"])
        _require(not parsed.username and not parsed.password and parsed.port is None
                 and not parsed.fragment, "Invalid proposed URL")
        source = request["source"]
        hosts = {"roster.watch", "www.roster.watch"} if source == "roster.watch" else {source}
        _require(parsed.hostname in hosts, "Source host mismatch")
        if source == "roster.watch":
            _require(parsed.scheme == "https" and parsed.path in {"", "/"}
                     and not parsed.query and request["method"] == "GET", "Invalid membership request proposal")
            rosterwatch += 1
        elif source == "ufcstats.com":
            _require(parsed.scheme == "http" and request["method"] == "GET"
                     and ((re.fullmatch(r"/fighter-details/[a-f0-9]{16}", parsed.path) and not parsed.query)
                          or (parsed.path == "/statistics/fighters"
                              and re.fullmatch(r"char=[a-z]&page=all", parsed.query))),
                     "Only canonical UFCStats listing/profile proposals are supported")
        else:
            raise StoreError("Unsupported proposed source; pause for review")
    _require(rosterwatch <= policy["max_rosterwatch_requests"], "Membership proposal ceiling")
    # A document may cost three charges if the separately reviewed challenge is
    # needed. Reserve budget for that branch even though proposals contain GETs.
    return rosterwatch + 3 * (len(requests) - rosterwatch), rosterwatch


def _needs(pins, requests, *, now=None):
    proof = json.loads(_read(pins["need_evidence"]["path"]))
    if proof.get("schema") == 2:
        from src.data.roster_hosted_needs import validate_need_evidence
        validate_need_evidence(pins, requests, now=now)
        return True
    _require(proof.get("schema") == 1 and proof.get("unknown_semantics") is False
             and proof.get("proposed_requests") == requests,
             "Request proposals require exact saved need evidence")
    return False  # Historical preparation format; never semantic or live authority.


def issue_candidate_plan(store, output_parent, policy_pin, *, expected_policy_sha256,
                         input_pins, proposed_requests, now=None, recovery_evidence_pin=None):
    """Prepare a fresh slot-bound artifact from server inputs; never execute it."""
    now = _time(now)
    policy = _policy(policy_pin, expected_policy_sha256)
    input_pins = json.loads(_json(input_pins))
    proposed_requests = json.loads(_json(proposed_requests))
    _inputs(input_pins)
    required_total, required_rw = _requests(proposed_requests, policy)
    _needs(input_pins, proposed_requests, now=now)
    slot_id = _slot(policy, now, recovery_evidence_pin)
    before = store.snapshot()
    _require(before["policy"] == policy["store_policy"], "Store/issuer policy drift")
    _require(not before["uncertain_requests"], "Uncertain accounting prevents issuance")
    limits = policy["store_policy"]
    _require(required_total <= limits["plan_ceiling"]
             and before["total_charged"] + required_total <= limits["total_ceiling"]
             and before["rosterwatch_charged"] + required_rw <= limits["rosterwatch_ceiling"],
             "Insufficient finite carried budget; never reset")
    parent = _safe(output_parent)
    _require(parent.is_dir(), "Existing preparation parent required")
    content = {"policy_pin": policy_pin, "input_pins": input_pins, "store_root": store.root.as_posix(),
               "proposed_requests": proposed_requests, "recovery_evidence_pin": recovery_evidence_pin,
               "accounting_before": before["head"], "total_charged": before["total_charged"],
               "rosterwatch_charged": before["rosterwatch_charged"]}
    reservation = store.reserve_plan_slot(slot_id, policy_pin, content_sha256=_sha(_json(content)), now=now)
    # Exclusive output and durable reservation remain consumed on every failure.
    directory = parent / (slot_id + "-" + reservation["nonce"])
    directory.mkdir(mode=0o700)
    _fsync_dir(parent)
    plan = {"schema": 1, "mode": "disabled_preparation", "plan_id": directory.name,
            "slot_id": slot_id, "nonce": reservation["nonce"], "issued_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=policy["plan_ttl_seconds"])).isoformat(),
            "network_requests": 0, "publication_requested": False,
            "activation_authorized": False, "slot_reservation": reservation,
            "content": content, "maximum_future_charges": required_total}
    path = directory / "plan.json"
    _new_file(path, _json(plan), immutable=True)
    plan_pin = file_pin(path)
    review = validate_candidate_plan(store, plan_pin, expected_policy_sha256=expected_policy_sha256, now=now)
    review_path = directory / "review.json"
    _new_file(review_path, _json(review), immutable=True)
    return {"plan": plan_pin, "review": file_pin(review_path), "source_requests": 0,
            "activation_authorized": False}


def validate_candidate_plan(store, plan_pin, *, expected_policy_sha256, now=None):
    """Independently recompute preparation semantics and exact durable binding."""
    now = _time(now)
    verify_pin(plan_pin)
    plan = json.loads(_read(plan_pin["path"]))
    _require(set(plan) == {"schema", "mode", "plan_id", "slot_id", "nonce", "issued_at", "expires_at",
                           "network_requests", "publication_requested", "activation_authorized",
                           "slot_reservation", "content", "maximum_future_charges"}, "Invalid preparation plan")
    _require(plan["schema"] == 1 and plan["mode"] == "disabled_preparation"
             and plan["network_requests"] == 0 and plan["publication_requested"] is False
             and plan["activation_authorized"] is False, "Preparation grants no execution authority")
    content = plan["content"]
    policy = _policy(content["policy_pin"], expected_policy_sha256)
    _inputs(content["input_pins"])
    required_total, required_rw = _requests(content["proposed_requests"], policy)
    semantic_needs_verified = _needs(content["input_pins"], content["proposed_requests"], now=now)
    _require(not semantic_needs_verified or content.get("store_root") == store.root.as_posix(),
             "Semantic plan is bound to another accounting store")
    issued, expires = _time(plan["issued_at"]), _time(plan["expires_at"])
    _require(issued <= now <= expires and (expires - issued).total_seconds() == policy["plan_ttl_seconds"],
             "Future, expired or changed plan lifetime")
    _require(_slot(policy, issued, content["recovery_evidence_pin"]) == plan["slot_id"], "Slot drift")
    state = store.snapshot()
    reservation = state["plan_slots"].get(plan["slot_id"])
    _require(reservation == plan["slot_reservation"] and reservation is not None
             and reservation["nonce"] == plan["nonce"]
             and reservation["policy_pin"] == content["policy_pin"]
             and reservation["content_sha256"] == _sha(_json(content))
             and reservation["previous"] == content["accounting_before"]["sha256"]
             and reservation["sequence"] == content["accounting_before"]["sequence"] + 1
             and reservation["at_utc"] == plan["issued_at"]
             and plan["plan_id"] == plan["slot_id"] + "-" + plan["nonce"], "Durable slot/content binding mismatch")
    _require(state["head"]["sha256"] == reservation["sha256"]
             and state["policy"] == policy["store_policy"] and not state["uncertain_requests"]
             and state["total_charged"] == content["total_charged"]
             and state["rosterwatch_charged"] == content["rosterwatch_charged"], "Accounting changed since issuance")
    _require(plan["maximum_future_charges"] == required_total
             and required_total <= policy["store_policy"]["plan_ceiling"]
             and state["total_charged"] + required_total <= policy["store_policy"]["total_ceiling"]
             and state["rosterwatch_charged"] + required_rw <= policy["store_policy"]["rosterwatch_ceiling"],
             "Finite budget binding mismatch")
    return {"schema": 1, "status": "preparation_validated_execution_disabled", "plan": plan_pin,
            "policy": content["policy_pin"], "store_head": state["head"],
            "source_requests_authorized": 0, "activation_authorized": False,
            "publication_authorized": False, "semantic_needs_verified": semantic_needs_verified,
            "validated_at": now.isoformat()}
