"""Derive disabled collection needs from complete pinned saved evidence.

Membership comes only from the canonical roster.watch response. This module
does not assign identities or mutate any roster. It reuses the ordinary matcher
with every historical, conflicting and failed capture. URLs are derived by the
matcher from source names or observed links, never from a caller's URL list.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re

from src.data.roster_hosted_store import (
    StoreError, _json, _new_file, _read, _require, _safe, _sha, _time, file_pin, verify_pin,
)

SEMANTIC_POLICY = {"schema": 1, "max_membership_age_seconds": 86400,
                   "max_identity_age_seconds": 604800, "source_requests_authorized": 0,
                   "publication_authorized": False, "activation_authorized": False,
                   "max_preparation_charges": 12}
BASE_ROLES = {"owners", "membership", "catalog", "release"}
MEMBERSHIP_URL = "https://www.roster.watch/"


def _pinned_json(pin):
    verify_pin(pin)
    return json.loads(_read(pin["path"]))


def _catalog(value):
    _require(isinstance(value, dict) and value.get("schema_version") == 1
             and isinstance(value.get("captures"), list), "Complete saved capture catalog required")
    for capture in value["captures"]:
        _require(isinstance(capture, dict)
                 and any(key in capture for key in ("body_text", "body_base64"))
                 and not any(key in capture for key in ("capture", "decoded_path", "raw_path", "path")),
                 "Capture bytes must be embedded; external paths cannot grant authority")
    return value["captures"]


def _request(url):
    if url == MEMBERSHIP_URL:
        return {"request_id": "membership", "source": "roster.watch", "method": "GET", "url": url}
    _require(re.fullmatch(r"http://ufcstats\.com/(?:fighter-details/[a-f0-9]{16}|"
                         r"statistics/fighters\?char=[a-z]&page=all)", url),
             "Unsupported semantic collection URL")
    return {"request_id": "document-" + _sha(url.encode())[:24], "source": "ufcstats.com",
            "method": "GET", "url": url}


def derive_need_evidence(input_pins, *, now=None):
    """Pure saved-input derivation; malformed/incomplete input fails closed.

owners pins a JSON {schema:1, seed_root, seed_inventory}; membership pins a
JSON {schema:1, body, envelope}. catalog pins the complete new saved captures;
the seed's prior catalog is also included. release pins the reviewed release
manifest. All four files and nested pins must already be readonly and exact.
Stale membership permits only a proposed refresh, never identity work based on
an old roster. Unsupported needs remain explicit blockers, not omitted rows.
"""
    from src.data import roster_hosted_candidate as candidate
    from src.data import rosterwatch_active_roster as ordinary
    from src.data.name_utils import normalize_person_name
    from src.data.rosterwatch_identity_matching import CapturedIdentityProvider, surname_listing_url
    from src.data.rosterwatch_site import parse_rosterwatch_site_roster, rosterwatch_csv_bytes
    import pandas as pd

    now = _time(now)
    _require(isinstance(input_pins, dict) and set(input_pins) == BASE_ROLES,
             "Complete semantic input roles required")
    values = {key: _pinned_json(pin) for key, pin in input_pins.items()}
    owners_spec, membership_spec = values["owners"], values["membership"]
    _require(isinstance(owners_spec, dict) and set(owners_spec) == {"schema", "seed_root", "seed_inventory"}
             and owners_spec["schema"] == 1, "Exact complete-owner seed specification required")
    _require(isinstance(membership_spec, dict) and set(membership_spec) == {"schema", "body", "envelope"}
             and membership_spec["schema"] == 1, "Exact membership evidence specification required")
    seed = _safe(owners_spec["seed_root"])
    before = candidate.tree_inventory(seed)
    _require(before == _pinned_json(owners_spec["seed_inventory"]), "Complete seed inventory drift")
    _require(all(name == candidate.ROSTER_FILE or name.startswith("rosterwatch/")
                 for name in before["files"]), "Roster-only complete seed required")
    owners = candidate._owner_coverage(seed, ordinary)
    state = ordinary._read_previous_state(seed / "rosterwatch" / candidate.STATE_FILE)
    roster = pd.DataFrame(candidate._records(seed / candidate.ROSTER_FILE))
    ledger = candidate._records(seed / "rosterwatch" / ordinary.ROSTERWATCH_SOURCE_LEDGER_FILENAME)
    from src.data.rosterwatch_identity_evidence import resolve_records
    ledger = resolve_records(ledger, seed / "rosterwatch")
    known = ordinary._known_identity_pages(seed / "rosterwatch" / ordinary.ROSTERWATCH_LEGACY_LEDGER_FILENAME,
                                            ledger, roster)
    ordinary._verify_identity_aliases(list(owners.values()),
        seed / "rosterwatch" / ordinary.ROSTERWATCH_LEARNED_ALIASES_FILENAME, known)
    # _verify_identity_aliases replays _validated_identity_bindings for every
    # learned decision, including historical owners. Non-learned baseline
    # carriers must be independently represented in the complete source ledgers.
    for owner, row in owners.items():
        if row.get("external_carrier_status") == "resolved":
            carrier = str(row.get("preferred_carrier_type")) + ":" + str(row.get("preferred_carrier_value"))
            _require(known.get(carrier) == owner or bool(row.get("identity_binding_json")),
                     "Resolved owner lacks independent ledger or replayable binding")
    occupied = ordinary._occupied_identity_carriers(list(owners.values()), known)
    raw = _read(verify_pin(membership_spec["body"])["path"])
    verify_pin(membership_spec["envelope"])
    _, requested_at, observed = candidate._membership(
        {"membership_envelope": membership_spec["envelope"], "evidence_mode": "offline_regression"}, raw)
    _require(requested_at <= observed <= now, "Future membership cannot generate needs")
    _require(observed.date().isoformat() >= state["source_cutoff"], "Membership cutoff regressed")
    rows, parse = parse_rosterwatch_site_roster(raw)
    membership = rosterwatch_csv_bytes(rows)
    snapshot_hash = _sha(membership)
    _require(observed.date().isoformat() != state["source_cutoff"]
             or snapshot_hash == state["snapshot_sha256"], "Same-cutoff membership conflicts with accepted evidence")
    captures = _catalog(values["catalog"])
    previous_path = seed / "rosterwatch" / candidate.CATALOG_FILE
    prior = _catalog(json.loads(previous_path.read_bytes())) if previous_path.exists() else []
    provider = CapturedIdentityProvider(prior + captures)
    _require(all(_time(c["captured_at"]) <= now for c in provider.captures), "Future capture cannot generate needs")
    proof = {"schema": 2, "generator": "roster_hosted_needs_v1", "generated_at": now.isoformat(),
             "semantic_policy": dict(SEMANTIC_POLICY), "input_pins": input_pins,
             "unknown_semantics": False, "activation_authorized": False,
             "source_requests_authorized": 0, "publication_authorized": False,
             "membership_observed_at": observed.isoformat(), "membership_parse": parse,
             "membership_snapshot_sha256": snapshot_hash, "complete_owner_count": len(owners),
             "historical_owner_count": sum(not candidate._current(row) for row in owners.values()),
             "preserved_capture_count": len(prior) + len(captures),
             "normalized_capture_count": len(provider.captures),
             "capture_fingerprints": sorted(_sha(_json(c)) for c in provider.captures),
             "rows": [], "blockers": [], "proposed_requests": [], "deferred_requests": []}
    if (now - observed).total_seconds() > SEMANTIC_POLICY["max_membership_age_seconds"]:
        proof["phase"] = "membership_refresh_required"
        proof["proposed_requests"] = [_request(MEMBERSHIP_URL)]
    else:
        previous_count = len(ledger)
        _require(len(rows) <= previous_count * ordinary.ROSTERWATCH_RECURRING_GROWTH_RATIO,
                 "Membership growth guard failed")
        _require(not (len(rows) < previous_count * ordinary.ROSTERWATCH_RECURRING_SHRINK_RATIO
                     and previous_count - len(rows) >= ordinary.ROSTERWATCH_RECURRING_SHRINK_ABSOLUTE),
                 "Membership shrink guard failed")
        by_identity = defaultdict(set)
        for owner, row in owners.items():
            by_identity[(normalize_person_name(row.get("current_source_name", "")),
                         row.get("current_source_dob", ""))].add(owner)
        current = []
        for ordinal, row in enumerate(rows, 1):
            matches = by_identity[(normalize_person_name(row["fighter"]), row["date_of_birth"])]
            reviewed = ordinary.ROSTERWATCH_REVIEWED_SOURCE_OWNER_TRANSITIONS.get(
                (row["fighter"], row["date_of_birth"]), "")
            if reviewed:
                _require(reviewed in owners and (not matches or matches == {reviewed}),
                         "Reviewed continuity conflicts with complete owners")
                matches = {reviewed}
            _require(len(matches) <= 1, "Ambiguous exact source ownership; no URL authority")
            owner = next(iter(matches)) if matches else ordinary._provisional_owner_id(
                source_sha256=snapshot_hash, row_ordinal=ordinal, fighter=row["fighter"], dob=row["date_of_birth"])
            current.append((owner, row))
        _require(len({owner for owner, _ in current}) == len(current), "Duplicate current source owner")
        current_owners = {owner for owner, _ in current}
        registry = [dict(row, is_current=owner in current_owners) for owner, row in owners.items()]
        profiles = ordinary._confirmed_identity_profiles(ledger, roster)
        opponents = ordinary._confirmed_identity_opponents(registry, profiles)
        fresh = candidate.FreshIdentityProvider(provider, clock=lambda: now,
            max_age_seconds=SEMANTIC_POLICY["max_identity_age_seconds"], minimum_time=now)
        proposed = set()
        for owner, row in current:
            record = owners.get(owner, {"internal_owner_id": owner,
                "current_source_name": row["fighter"], "current_source_dob": row["date_of_birth"]})
            item = {"internal_owner_id": owner, "fighter": row["fighter"], "date_of_birth": row["date_of_birth"]}
            if record.get("external_carrier_status") == "resolved":
                proof["rows"].append(dict(item, status="existing_identity_preserved", requested_urls=[]))
                continue
            decision = fresh.decide(record, source_context={**row, "snapshot_sha256": snapshot_hash},
                confirmed_opponents=opponents, occupied_carriers=occupied)
            urls = set(decision["requested_urls"])
            reasons = decision["reason_codes"]
            if decision["status"] == "resolved":
                urls.clear()  # No additional collection is needed for a proved fresh binding.
            elif any("conflict" in reason or "ambiguous" in reason or "owned_by_another" in reason
                     for reason in reasons):
                proof["blockers"].append(dict(item, reason="conflicting_identity_evidence", reasons=reasons))
                urls.clear()
            else:
                # Failed and old captures stay in the matcher. Refresh their
                # exact evidenced URLs; no old capture is removed to force a match.
                relevant = {page["page"] for page in decision["candidate_pages"]}
                listing = surname_listing_url(row["fighter"])
                if listing:
                    relevant.add(listing)
                for url in relevant:
                    related = [c for c in provider.captures if c["url"] == url]
                    if related:
                        latest = max(related, key=lambda c: _time(c["captured_at"]))
                        if not latest["semantic_evidence_eligible"] or (
                                now - _time(latest["captured_at"])).total_seconds() > SEMANTIC_POLICY["max_identity_age_seconds"]:
                            urls.add(url)
            supported = []
            for url in sorted(urls):
                try:
                    _request(url)
                except StoreError:
                    proof["blockers"].append(dict(item, reason="unsupported_source_need", url=url))
                else:
                    proposed.add(url)
                    supported.append(url)
            if decision["status"] != "resolved" and not urls and not any(
                    b["internal_owner_id"] == owner for b in proof["blockers"]):
                proof["blockers"].append(dict(item, reason="unresolved_without_supported_collection", reasons=reasons))
            proof["rows"].append(dict(item, status=decision["status"], requested_urls=supported,
                reason_codes=reasons, decision_sha256=decision["decision_sha256"]))
        # An unlinked fighter remains pending. Membership automation does not
        # halt for a per-fighter identity conflict or unsupported evidence route.
        # Structural manifest/owner ambiguity already raised above; those are
        # the cases where machine semantics cannot be trusted at all.
        proof["phase"] = "identity_evidence_required" if proposed else "saved_candidate_ready"
        def priority(url):
            attempts = [_time(c["captured_at"]) for c in provider.captures if c["url"] == url]
            # Never-attempted evidence precedes old attempts; latest-attempt time
            # makes repeated failures yield to every older unmet need.
            return (bool(attempts), max(attempts).isoformat() if attempts else "", url)
        ordered = [_request(url) for url in sorted(proposed, key=priority)]
        cap = SEMANTIC_POLICY["max_preparation_charges"] // 3
        proof["proposed_requests"] = ordered[:cap]
        proof["deferred_requests"] = ordered[cap:]
    _require(candidate.tree_inventory(seed) == before, "Complete seed changed during derivation")
    for pin in input_pins.values():
        verify_pin(pin)
    verify_pin(owners_spec["seed_inventory"])
    verify_pin(membership_spec["body"])
    verify_pin(membership_spec["envelope"])
    return proof


def validate_need_evidence(input_pins, proposed_requests, *, now=None):
    """Recompute semantic evidence; caller-written booleans are not authority."""
    proof = _pinned_json(input_pins["need_evidence"])
    _require(proof.get("schema") == 2, "Trusted generated semantic need evidence required")
    generated = _time(proof.get("generated_at"))
    now = _time(now)
    _require(0 <= (now - generated).total_seconds() <= 3600, "Need evidence future or expired")
    base = {key: value for key, value in input_pins.items() if key != "need_evidence"}
    expected = derive_need_evidence(base, now=generated)
    _require(proof == expected, "Semantic need evidence does not reproduce from pinned sources")
    _require(not proof["unknown_semantics"] and proof["proposed_requests"] == proposed_requests,
             "Unresolved or changed semantic needs cannot issue a plan")
    if proof["phase"] != "membership_refresh_required":
        _require((now - _time(proof["membership_observed_at"])).total_seconds()
                 <= SEMANTIC_POLICY["max_membership_age_seconds"], "Membership expired after derivation")
        if now != generated:
            current = derive_need_evidence(base, now=now)
            def semantic_state(value):
                return {"phase": value["phase"], "proposed_requests": value["proposed_requests"],
                        "deferred_requests": value["deferred_requests"],
                        "unknown_semantics": value["unknown_semantics"],
                        "rows": [{key: item.get(key) for key in
                                  ("internal_owner_id", "status", "requested_urls", "reason_codes")}
                                 for item in value["rows"]]}
            _require(semantic_state(proof) == semantic_state(current),
                     "Identity evidence freshness changed after derivation")
    return proof


def prepare_need_manifest(input_pins, output_directory, *, now=None):
    """Save an exclusive disabled derivation and complete issuer input manifest."""
    output = _safe(output_directory)
    _require(not output.exists() and output.parent.is_dir(), "Fresh semantic output required")
    output.mkdir(mode=0o700)
    from src.data.roster_hosted_store import _fsync_dir
    _fsync_dir(output.parent)
    try:
        proof = derive_need_evidence(input_pins, now=now)
        _new_file(output / "need-evidence.json", _json(proof), immutable=True)
        pins = dict(input_pins, need_evidence=file_pin(output / "need-evidence.json"))
        _new_file(output / "inputs-manifest.json", _json({"schema": 1, "input_pins": pins}), immutable=True)
        return {"need_evidence": pins["need_evidence"], "inputs_manifest": file_pin(output / "inputs-manifest.json"),
                "source_requests_authorized": 0, "activation_authorized": False,
                "unknown_semantics": proof["unknown_semantics"], "proposed_requests": proof["proposed_requests"]}
    except BaseException as exc:
        _new_file(output / "failed.json", _json({"status": "failed_execution_disabled",
            "error": type(exc).__name__ + ": " + str(exc), "replay_authorized": False}), immutable=True)
        raise
