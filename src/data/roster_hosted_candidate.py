"""Prepare an isolated ordinary Roster Watch candidate from saved evidence.

This module has no source transport, scheduler or publication operation.  Its
specification is a pinned *input*, not an approval.  The trusted supervisor owns
plan consumption, containment and final evidence sealing.  New output roots are
exclusive and failed output is retained.  Historical identity decisions are
revalidated by the ordinary adapter; their age never deletes an existing owner.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import stat
from urllib.parse import urlsplit


SCHEMA_VERSION = 1
ROSTER_FILE = "ufc_active_roster_official.csv"
REGISTRY_FILE = "rosterwatch_owner_registry.csv"
STATE_FILE = "rosterwatch_state.json"
CATALOG_FILE = "rosterwatch_identity_captures.json"


class CandidateError(ValueError):
    """Saved evidence or candidate isolation failed validation."""


def _require(condition, message):
    if not condition:
        raise CandidateError(message)


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def _time(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        _require(parsed.tzinfo is not None, "timestamps require timezone")
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError) as exc:
        raise CandidateError("invalid zoned timestamp") from exc


def _path(value, *, existing=True):
    path = Path(value)
    _require(path.is_absolute() and ".." not in path.parts, "absolute non-traversing path required")
    for part in (path, *path.parents):
        _require(not part.is_symlink(), "symbolic links are forbidden")
        if part.exists():
            info = part.lstat()
            _require(not getattr(info, "st_file_attributes", 0) & 0x400,
                     "reparse points are forbidden")
    resolved = path.resolve(strict=existing)
    if existing and resolved.is_file():
        _require(resolved.stat().st_nlink == 1, "hard-linked input is forbidden")
    return resolved


def file_pin(path):
    path = _path(path)
    _require(path.is_file(), "pin requires regular file")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path.read_bytes())}


def _read_pin(pin):
    _require(isinstance(pin, dict), "file pin required")
    path = _path(pin["path"])
    _require(path.is_file(), "pinned input is not a file")
    content = path.read_bytes()
    _require(type(pin.get("bytes")) is int and len(content) == pin["bytes"]
             and _sha(content) == pin.get("sha256"), "input pin mismatch")
    return content


def tree_inventory(root):
    """Exact bytes and write-relevant metadata; excludes access time."""
    root = _path(root)
    _require(root.is_dir(), "input tree must be a directory")
    files, directories = {}, {}
    for path in [root, *sorted(root.rglob("*"))]:
        path = _path(path)
        info = path.stat()
        relative = path.relative_to(root).as_posix()
        item = {"mode": stat.S_IMODE(info.st_mode), "mtime_ns": info.st_mtime_ns}
        if path.is_dir():
            directories[relative] = item
        else:
            _require(path.is_file() and info.st_nlink == 1, "only regular unaliased seed files are allowed")
            files[relative] = {**item, "bytes": info.st_size, "sha256": _sha(path.read_bytes())}
    return {"files": files, "directories": directories}


def _write_new(path, content):
    """Worker-side byte write only; the trusted parent seals after exit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _records(path):
    # Embedded replayable identity evidence can exceed csv's 128 KiB default.
    # Bound the explicit read and restore the caller's process setting afterwards.
    previous_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(134217728)
        return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"), newline="")))
    finally:
        csv.field_size_limit(previous_limit)


def _owners(root):
    from src.data.rosterwatch_identity_evidence import resolve_records
    records = resolve_records(_records(root / "rosterwatch" / REGISTRY_FILE), root / "rosterwatch")
    owners = {record.get("internal_owner_id", ""): record for record in records}
    _require("" not in owners and len(owners) == len(records), "owner registry is incomplete or duplicated")
    return owners


def _current(record):
    return str(record.get("is_current", "")).casefold() == "true"


def _owner_coverage(root, ordinary):
    state = ordinary._read_previous_state(root / "rosterwatch" / STATE_FILE)
    _require(state is not None, "ordinary accepted seed state is required; no migration")
    owners = _owners(root)
    roster = _records(root / ROSTER_FILE)
    ledger = _records(root / "rosterwatch" / ordinary.ROSTERWATCH_SOURCE_LEDGER_FILENAME)
    roster_owners = {row.get("internal_owner_id", "") for row in ledger}
    _require(len(owners) == state.get("owner_registry_rows"), "full historical owner count mismatch")
    _require(len(roster) == len(ledger) == len(roster_owners) and "" not in roster_owners
             and [str(index + 1) for index in range(len(ledger))]
             == [row.get("rw_row_ordinal") for row in ledger],
             "current roster owner coverage is incomplete")
    _require(roster_owners == {owner for owner, row in owners.items() if _current(row)},
             "current roster and complete registry disagree")
    _require(_sha((root / ROSTER_FILE).read_bytes()) == state.get("operational_roster_sha256"),
             "ordinary roster hash mismatch")
    return owners


def _portable_candidate_state(candidate, output):
    """Normalize two relative Windows snapshot references only in the copy."""
    path = candidate / "rosterwatch" / STATE_FILE
    original = path.read_bytes()
    state = json.loads(original)
    changes = {}
    for key in ("snapshot_path", "initial_legacy_snapshot_path"):
        value = state.get(key)
        _require(isinstance(value, str), "snapshot reference must be text")
        normalized = value.replace("\\", "/")
        parts = normalized.split("/")
        _require(len(parts) == 2 and parts[0] == "snapshots"
                 and parts[1] not in {"", ".", ".."} and ":" not in parts[1],
                 "snapshot reference must be snapshots/<filename>")
        if value != normalized:
            changes[key] = {"before": value, "after": normalized}
            state[key] = normalized
    if changes:
        _write_new(output / "evidence" / "original-seed-state.json", original)
        path.unlink()
        _write_new(path, _json_bytes(state))
    return changes


def _logical_url(url):
    parsed = urlsplit(url)
    return (parsed.hostname or "").removeprefix("www.").casefold() + parsed.path.rstrip("/")


class FreshIdentityProvider:
    """Run unchanged matching on all captures, then withhold stale new bindings.

    All contrary, historical and failed captures remain in the matcher.  Dropping
    old captures before matching could erase a DOB conflict.  Only unresolved
    owners reach this provider through the ordinary adapter.
    """
    def __init__(self, provider, *, clock, max_age_seconds, minimum_time):
        self.provider = provider
        self.clock = clock
        self.max_age_seconds = max_age_seconds
        self.freshness_withheld = []
        self.last_time = _time(minimum_time.isoformat())

    def lookup(self, *args, **kwargs):
        return self.provider.lookup(*args, **kwargs)

    def decide(self, record, observations=None, **kwargs):
        now = _time(self.clock().isoformat())
        _require(now >= self.last_time, "identity decision clock moved backwards")
        self.last_time = now
        decision = self.provider.decide(record, observations, **dict(kwargs, checked_at=now.isoformat()))
        if decision.get("status") != "resolved":
            return decision
        selected = _logical_url(decision["page"])
        captures = decision["evidence"]["captures"]
        related = [capture for capture in captures if _logical_url(capture["url"]) == selected]
        used_hashes = {page["evidence"]["capture_sha256"] for page in decision["candidate_pages"]
                       if _logical_url(page["page"]) == selected and not page["reason_not_bound"]}
        eligible = [capture for capture in related if capture.get("semantic_evidence_eligible") is True
                    and capture["decoded_sha256"] in used_hashes]
        fresh = [capture for capture in eligible
                 if 0 <= (now - _time(capture["captured_at"])).total_seconds() <= self.max_age_seconds]
        latest_eligible = max((_time(capture["captured_at"]) for capture in fresh), default=None)
        later_unavailable = latest_eligible is not None and any(
            _time(capture["captured_at"]) > latest_eligible
            and not capture.get("semantic_evidence_eligible") for capture in related)
        if fresh and not later_unavailable:
            return decision  # Exact canonical decision; ordinary replay must still validate it.
        reason = "new_identity_source_stale_or_later_unavailable"
        self.freshness_withheld.append({"internal_owner_id": record["internal_owner_id"], "reason": reason})
        pending = dict(decision, status="pending", carrier_type="", carrier_value="",
                       matching_basis="", ladder_rung=None, learned_aliases=[],
                       reason_codes=sorted(set(decision.get("reason_codes", []) + [reason])))
        pending.pop("decision_sha256", None)
        pending["decision_sha256"] = _sha(json.dumps(pending, sort_keys=True,
            separators=(",", ":"), ensure_ascii=False).encode())
        return pending


def _membership(spec, raw):
    from src.data.rosterwatch_site import looks_like_bot_challenge
    envelope = json.loads(_read_pin(spec["membership_envelope"]))
    request, response = envelope.get("request", {}), envelope.get("response", {})
    _require(envelope.get("schema_version") == 1, "membership envelope schema mismatch")
    _require(request.get("method") == "GET" and request.get("url") == "https://www.roster.watch/",
             "membership must be the single canonical roster.watch GET")
    _require(response.get("status") == 200 and response.get("complete") is True
             and not response.get("error"), "membership response incomplete or failed")
    _require(response.get("body_sha256") == _sha(raw), "membership envelope/body mismatch")
    headers = response.get("headers")
    _require(isinstance(headers, list) and all(isinstance(row, list) and len(row) == 2
             and all(isinstance(value, str) for value in row) for row in headers), "response headers required")
    lowered = {}
    for key, value in headers:
        lowered.setdefault(key.casefold(), []).append(value)
    _require(len(lowered.get("content-type", [])) == 1
             and lowered["content-type"][0].split(";", 1)[0].strip().casefold() == "text/html",
             "membership media type must be unambiguous text/html")
    if "content-length" in lowered:
        _require(len(lowered["content-length"]) == 1 and lowered["content-length"][0].isdigit()
                 and int(lowered["content-length"][0]) == len(raw), "membership body length mismatch")
    _require(not looks_like_bot_challenge(200, dict(headers), raw), "membership challenge is not a roster")
    requested, observed = _time(request.get("started_at")), _time(response.get("completed_at"))
    _require(requested <= observed, "membership timestamps are reversed")
    if spec["evidence_mode"] == "current_candidate":
        _require(envelope.get("plan_id") == spec.get("acquisition_plan_id")
                 and isinstance(spec.get("acquisition_plan_id"), str)
                 and bool(spec["acquisition_plan_id"]), "current membership must bind this acquisition plan")
        _require(requested >= _time(spec["run_started_at"]), "current membership must be observed in this run")
    return envelope, requested, observed


def prepare_candidate(spec_path, spec_sha256, *, clock=None):
    """Create one isolated candidate; raise on failure and retain terminal evidence.

    Required spec fields: schema_version=1, publication_requested=false,
    network_requests=0, evidence_mode (offline_regression/current_candidate),
    seed_root, seed_inventory pin, output_root, membership_body pin,
    membership_envelope pin, identity_catalog pin, max_membership_age_seconds,
    max_identity_age_seconds, identity_check_cap, run_started_at and expires_at.
    Current mode additionally binds acquisition_plan_id.  Both freshness ceilings
    are explicit proposed policy values, bounded to 24 hours / seven days here.
    """
    clock = clock or (lambda: datetime.now(timezone.utc))
    spec_path = _path(spec_path)
    content = spec_path.read_bytes()
    _require(_sha(content) == spec_sha256, "candidate specification hash mismatch")
    spec = json.loads(content)
    _require(spec.get("schema_version") == SCHEMA_VERSION, "candidate schema mismatch")
    _require(spec.get("publication_requested") is False and type(spec.get("network_requests")) is int
             and spec["network_requests"] == 0, "candidate publication/network must be explicitly disabled")
    _require(spec.get("evidence_mode") in {"offline_regression", "current_candidate"}, "unknown evidence mode")
    storage = spec.get("identity_evidence_storage", "legacy_inline")
    _require(storage in {"legacy_inline", "shared_v1"}, "unknown identity evidence storage version")
    for field, ceiling in (("max_membership_age_seconds", 86400), ("max_identity_age_seconds", 604800)):
        _require(type(spec.get(field)) is int and 0 < spec[field] <= ceiling, "invalid freshness ceiling")
    _require(type(spec.get("identity_check_cap")) is int and 0 <= spec["identity_check_cap"] <= 40,
             "invalid offline identity check cap")
    _require(not any(os.getenv(name) for name in ("UFC_ROSTERWATCH_SOURCE_PATH", "UFC_ROSTERWATCH_SOURCE_URL")),
             "ambient roster source overrides are forbidden")
    seed = _path(spec["seed_root"])
    from src.data.rosterwatch_identity_evidence import STORE_DIRECTORY
    if (seed / "rosterwatch" / STORE_DIRECTORY).exists():
        _require(spec.get("identity_evidence_storage") != "legacy_inline",
                 "shared identity evidence cannot be downgraded to inline storage")
        storage = "shared_v1"
    output = _path(spec["output_root"], existing=False)
    _require(output != seed and output not in seed.parents and seed not in output.parents,
             "candidate output must be disjoint from seed")
    _require(not output.exists() and output.parent.is_dir(), "candidate output must be fresh under an existing parent")
    pins = [spec[key] for key in ("seed_inventory", "membership_body", "membership_envelope", "identity_catalog")]
    for pin in pins:
        path = _path(pin["path"])
        _require(output != path and output not in path.parents, "candidate cannot contain source authority")
        _read_pin(pin)
    inventory = json.loads(_read_pin(spec["seed_inventory"]))
    before = tree_inventory(seed)
    _require(before == inventory, "seed inventory mismatch; complete seed tree required")
    _require(ROSTER_FILE in before["files"] and "rosterwatch/" + REGISTRY_FILE in before["files"],
             "seed must contain operational roster and complete owner registry")
    _require(all(path == ROSTER_FILE or path.startswith("rosterwatch/") for path in before["files"]),
             "seed contains files outside the roster-only allowlist")
    started = _time(clock().isoformat())
    issued, expires = _time(spec["run_started_at"]), _time(spec["expires_at"])
    _require(issued <= started < expires and 0 < (expires - issued).total_seconds() <= 86400,
             "candidate run timing invalid or expired")
    output.mkdir(mode=0o700)
    report = {"schema_version": 1, "status": "failed", "publication_requested": False,
              "publication_performed": False, "publication_eligible": False,
              "network_requests": 0, "spec_sha256": spec_sha256,
              "evidence_mode": spec["evidence_mode"], "started_at": started.isoformat(),
              "identity_evidence_storage": storage,
              "replay_permitted": False, "limitations": [
                  "Candidate preparation is not publication or a full application quality audit.",
                  "Existing data quality limitations and training/live consistency remain unresolved.",
                  "Local evidence does not prove hosted volume durability or rollback detection."]}
    try:
        from src.data import rosterwatch_active_roster as ordinary
        from src.data.rosterwatch_site import parse_rosterwatch_site_roster, rosterwatch_csv_bytes
        _require(ordinary.ROSTERWATCH_IDENTITY_CAPTURE_FILENAME == CATALOG_FILE,
                 "ordinary capture catalog filename changed")
        raw = _read_pin(spec["membership_body"])
        envelope, requested, observed = _membership(spec, raw)
        _require(observed <= started, "membership observation is in the future")
        if spec["evidence_mode"] == "current_candidate":
            _require((started - observed).total_seconds() <= spec["max_membership_age_seconds"],
                     "current membership observation is stale")
        rows, parse = parse_rosterwatch_site_roster(raw)
        membership = rosterwatch_csv_bytes(rows)
        cutoff = observed.date().isoformat()
        _require(cutoff >= json.loads((seed / "rosterwatch" / STATE_FILE).read_bytes())["source_cutoff"],
                 "membership cutoff regressed")
        candidate = output / "candidate"
        candidate.mkdir()
        for relative in sorted(before["directories"]):
            if relative != ".":
                (candidate / relative).mkdir(parents=True, exist_ok=True)
        for relative in sorted(before["files"]):
            _write_new(candidate / relative, (seed / relative).read_bytes())
        report["candidate_snapshot_path_adaptation"] = _portable_candidate_state(candidate, output)
        prior_owners = _owner_coverage(candidate, ordinary)
        state_root = candidate / "rosterwatch"
        catalog_content = _read_pin(spec["identity_catalog"])
        catalog = json.loads(catalog_content)
        _require(catalog.get("schema_version") == 1 and isinstance(catalog.get("captures"), list),
                 "saved identity catalog schema mismatch")
        catalog_path = state_root / CATALOG_FILE
        # Every historical capture remains active matching evidence. Saving an
        # old catalog separately without merging it could erase a known conflict.
        if catalog_path.exists():
            previous_content = catalog_path.read_bytes()
            previous_catalog = json.loads(previous_content)
            _require(previous_catalog.get("schema_version") == 1
                     and isinstance(previous_catalog.get("captures"), list), "previous catalog schema mismatch")
            _write_new(output / "evidence" / "previous-identity-catalog.json", previous_content)
            catalog = dict(catalog, captures=previous_catalog["captures"] + catalog["captures"],
                           previous_catalog_sha256=_sha(previous_content),
                           collected_catalog_sha256=_sha(catalog_content))
            catalog_content = _json_bytes(catalog)
            catalog_path.unlink()
        _write_new(catalog_path, catalog_content)
        provider = ordinary.load_rosterwatch_identity_provider(state_root)
        _require(provider is not None, "saved identity provider required")
        for capture in provider.captures:
            _require(_time(capture["captured_at"]) <= started, "identity capture is in the future")
        fresh_provider = FreshIdentityProvider(provider, clock=clock,
                                              max_age_seconds=spec["max_identity_age_seconds"],
                                              minimum_time=started)
        _write_new(output / "evidence" / "membership-body.bin", raw)
        _write_new(output / "evidence" / "membership-envelope.json", _json_bytes(envelope))
        source = output / "evidence" / "membership.csv"
        _write_new(source, membership)
        snapshot = state_root / "snapshots" / f"{cutoff}--{_sha(membership)}.csv"
        if snapshot.exists():
            _require(snapshot.read_bytes() == membership, "candidate snapshot collision")
        else:
            # Ordinary helper sees matching existing bytes; no forbidden chmod is
            # attempted by the contained worker. Trusted parent seals all output.
            _write_new(snapshot, membership)
        def sync(check_cap):
            frame = ordinary.sync_rosterwatch_active_roster(
                output_path=candidate / ROSTER_FILE, state_root=state_root,
                source_path=source, source_site=False, source_cutoff=cutoff,
                allow_initial_migration=False, identity_provider=fresh_provider,
                identity_request_cap=check_cap,
                shared_identity_evidence=storage == "shared_v1")
            _require(frame.attrs.get("sync_complete") is True
                     and frame.attrs.get("sync_fallback_used") is False
                     and frame.attrs.get("request_count") == 0,
                     "ordinary candidate sync failed: " + str(frame.attrs.get("sync_error", frame.attrs)))
            return frame
        first = sync(spec["identity_check_cap"])
        next_owners = _owner_coverage(candidate, ordinary)
        _require(set(prior_owners).issubset(next_owners), "historical owner was lost")
        for owner, prior in prior_owners.items():
            after = next_owners[owner]
            if prior.get("external_carrier_status") == "resolved":
                for key in ("preferred_carrier_type", "preferred_carrier_value", "identity_binding_json"):
                    _require(after.get(key, "") == prior.get(key, ""), "existing identity carrier or binding changed")
            if not _current(prior) and not _current(after):
                _require(all(after.get(key, "") == value for key, value in prior.items()),
                         "historical owner evidence changed")
        for relative in before["files"]:
            if relative.startswith("rosterwatch/snapshots/") or relative.endswith("rosterwatch_legacy_reconciliation_ledger.csv"):
                _require((candidate / relative).read_bytes() == (seed / relative).read_bytes(),
                         "historical snapshot or baseline ledger changed")
        delta = "rosterwatch/rosterwatch_delta_log.jsonl"
        _require((candidate / delta).read_bytes().startswith((seed / delta).read_bytes()),
                 "historical delta log was rewritten")
        first_tree = tree_inventory(candidate)
        reloaded = sync(0)  # Reload validates persisted decisions without fresh checks.
        reload_tree = tree_inventory(candidate)
        changed = sorted(path for path in set(first_tree["files"]) | set(reload_tree["files"])
                         if first_tree["files"].get(path) != reload_tree["files"].get(path))
        _require(set(changed).issubset({"rosterwatch/rosterwatch_identity_review.json"}),
                 "ordinary reload changed roster, owner or evidence bytes/metadata")
        _require(reloaded.attrs.get("idempotent_snapshot") is True
                 and reloaded.attrs.get("identity_recheck", {}).get("checks_performed") == 0,
                 "ordinary reload was not idempotent")
        completed = _time(clock().isoformat())
        _require(completed >= fresh_provider.last_time and completed < expires,
                 "candidate deadline or clock rollback")
        age = (completed - observed).total_seconds()
        if spec["evidence_mode"] == "current_candidate":
            _require(age <= spec["max_membership_age_seconds"], "membership expired during candidate assessment")
        report.update(status="prepared_candidate_only", completed_at=completed.isoformat(),
                      membership_observed_at=observed.isoformat(), membership_requested_at=requested.isoformat(),
                      membership_age_seconds=age, current_source_claim=False,
                      current_timing_eligible=spec["evidence_mode"] == "current_candidate",
                      observation_kind=("saved_evidence_current_timing_check"
                                        if spec["evidence_mode"] == "current_candidate"
                                        else "saved_offline_regression"),
                      membership_parse=parse, current_members=len(reloaded), total_owners=len(next_owners),
                      historical_owners=sum(not _current(row) for row in next_owners.values()),
                      pending_members=sum(_current(row) and row.get("external_carrier_status") != "resolved"
                                          for row in next_owners.values()),
                      freshness_withheld=fresh_provider.freshness_withheld,
                      first_identity_recheck=first.attrs.get("identity_recheck"),
                      reload={"idempotent": True, "new_identity_checks": 0, "changed_files": changed},
                      candidate_inventory=reload_tree)
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        try:
            report["seed_preserved"] = tree_inventory(seed) == before
            report["authority_inputs_preserved"] = all(_read_pin(pin) is not None for pin in pins)
        except Exception as exc:
            report["seed_preserved"] = False
            report["authority_inputs_preserved"] = False
            report["preservation_error"] = type(exc).__name__ + ": " + str(exc)
        if not report["seed_preserved"] or not report["authority_inputs_preserved"]:
            report["status"] = "failed"
            report["error"] = "protected seed changed"
        _write_new(output / "candidate-terminal.json", _json_bytes(report))
        _require(report["seed_preserved"] and report["authority_inputs_preserved"], "protected input changed")
    return report
