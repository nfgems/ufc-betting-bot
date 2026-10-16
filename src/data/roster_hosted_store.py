"""Standalone Linux durable controls for a future reviewed roster worker.

This module opens no sockets and installs no recurring job. The existing
acquisition CLI is deliberately unchanged: an audited adapter must call
``charge_request`` before each possible send and carry the returned receipt.
Creation requires an explicit policy and pinned imported accounting evidence.

The filesystem must provide Linux flock, atomic rename and working file and
directory fsync. Recovery never truncates a journal, reclaims a charge, repairs
an orphan, or replays a plan. Hash chaining detects accidental drift; it is not
protection against a privileged writer or rollback of the entire volume. Such
recovery requires independently retained head anchors and operator review.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat


class StoreError(RuntimeError):
    """A durable-control requirement failed; no send is authorized."""


class StoreBusy(StoreError):
    """Another process owns the store lock; do not start another worker."""


_HASH = re.compile(r"[a-f0-9]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
_SOURCES = {"ufcstats.com", "ufc.com", "roster.watch"}
_ZERO = "0" * 64
_WINDOWS = {"day": 86400, "week": 604800, "month": 2592000}

# Review material only. No caller or production configuration uses this policy.
PROPOSED_POLICY = {
    "total_ceiling": 200, "rosterwatch_ceiling": 80,
    "plan_ceiling": 12, "day_ceiling": 50, "week_ceiling": 100,
    "month_ceiling": 200, "rosterwatch_day_ceiling": 2,
    "max_source_age_seconds": 86400, "min_archive_age_seconds": 7776000,
}


def _require(value, message):
    if not value:
        raise StoreError(message)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha(body):
    return hashlib.sha256(body).hexdigest()


def _time(value=None):
    if value is None:
        value = datetime.now(timezone.utc)
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise StoreError("Invalid UTC timestamp") from exc
    _require(isinstance(value, datetime) and value.tzinfo is not None,
             "Timezone-aware timestamp required")
    return value.astimezone(timezone.utc)


def _safe(path):
    path = Path(path).absolute()
    _require(path.resolve() == path, "Linked or noncanonical path")
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            _require(not stat.S_ISLNK(info.st_mode), "Symbolic link forbidden")
    return path


def _read(path):
    path = _safe(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                 "Regular unshared file required")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(fd)


def file_pin(path):
    path = _safe(path)
    body = _read(path)
    return {"path": path.as_posix(), "bytes": len(body), "sha256": _sha(body)}


def verify_pin(pin, *, readonly=True):
    _require(isinstance(pin, dict) and set(pin) == {"path", "bytes", "sha256"},
             "Invalid file pin")
    _require(type(pin["bytes"]) is int and pin["bytes"] >= 0
             and isinstance(pin["sha256"], str) and _HASH.fullmatch(pin["sha256"]),
             "Invalid file pin values")
    _require(file_pin(pin["path"]) == pin, "Pinned file drift")
    _require(not readonly or not (Path(pin["path"]).stat().st_mode & 0o222),
             "Pinned evidence must be read-only")
    return pin


def _fsync_dir(path):
    fd = os.open(_safe(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _new_file(path, body, *, immutable=False):
    path = _safe(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(body)
            stream.flush()
            if immutable:
                os.fchmod(fd, 0o400)
            os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(path.parent)


def _policy(value):
    _require(isinstance(value, dict) and set(value) == set(PROPOSED_POLICY),
             "Complete explicit policy required")
    _require(all(type(x) is int and x > 0 for x in value.values()),
             "Policy limits must be positive integers")
    _require(value["total_ceiling"] <= 200 and value["rosterwatch_ceiling"] <= 80,
             "Carried source ceilings may not increase")
    _require(value["plan_ceiling"] <= value["total_ceiling"], "Invalid plan ceiling")
    return dict(value)


def _imported_accounting(pins):
    """Reconcile the existing request_started format, including prepaid intents."""
    from urllib.parse import urlsplit
    ordinals, rosterwatch, plan_ids = set(), 0, set()
    for pin in pins:
        verify_pin(pin)
        body = _read(pin["path"])
        _require(not body or body.endswith(b"\n"), "Torn imported ledger")
        for line in body.splitlines():
            row = json.loads(line)
            _require(isinstance(row, dict), "Invalid imported ledger row")
            if row.get("plan_id"):
                plan_ids.add(row["plan_id"])
            if row.get("event") != "request_started":
                continue
            ordinal = row.get("cumulative_ordinal", row.get("prior_sends", 0) + row["ordinal"])
            _require(type(ordinal) is int and ordinal > 0 and ordinal not in ordinals,
                     "Duplicate or invalid imported charge ordinal")
            source = (urlsplit(row["url"]).hostname or "").removeprefix("www.")
            _require(source in _SOURCES, "Unknown imported source")
            ordinals.add(ordinal)
            rosterwatch += source == "roster.watch"
    _require(ordinals == set(range(1, len(ordinals) + 1)), "Imported charge sequence has gaps")
    return len(ordinals), rosterwatch, sorted(plan_ids)


def assess_source_freshness(observed_at, *, now=None, max_age_seconds):
    """Assess actual observation chronology; never substitute a replay clock."""
    _require(type(max_age_seconds) is int and max_age_seconds > 0, "Invalid maximum age")
    age = (_time(now) - _time(observed_at)).total_seconds()
    return {"eligible": 0 <= age <= max_age_seconds, "age_seconds": age,
            "reason": "future_observation" if age < 0 else
                      "stale_observation" if age > max_age_seconds else "within_policy"}


def _consumed_catalog(catalog_pin, review_pin, evidence_pins, manifest):
    """Validate a reviewed mapping of already charged ledgers, never add charges.

    Original catalog bytes retain Windows paths. The explicit relocation map
    changes access paths only; hashes, sizes and original evidence stay exact.
    The independent review is supplied by the trusted caller, not the worker.
    """
    verify_pin(catalog_pin)
    verify_pin(review_pin)
    catalog = json.loads(_read(catalog_pin["path"]))
    review = json.loads(_read(review_pin["path"]))
    _require(review.get("status") == "passed" and review.get("blockers") == []
             and bool(review.get("checks"))
             and all(v is True for v in review["checks"].values()), "Catalog review did not pass")
    _require(all(review["catalog"][k] == catalog_pin[k] for k in ("bytes", "sha256")),
             "Catalog review binding mismatch")
    _require(catalog.get("schema_version") == 1 and catalog.get("new_charges") == 0
             and catalog.get("source_requests") == 0 and catalog.get("executed_plans") == 0,
             "Catalog cannot grant new execution or charges")
    _require(isinstance(evidence_pins, dict), "Explicit relocated evidence map required")

    def relocated(original):
        pin = evidence_pins[original["path"]]
        _require(all(pin[k] == original[k] for k in ("bytes", "sha256")), "Migration evidence drift")
        verify_pin(pin)
        return pin

    prior = {(p["sha256"], p["bytes"]) for p in manifest["prior_ledger_pins"]}
    seen_ledgers, consumed, hashes = set(), set(), set()
    total = 0
    for record in catalog["records"]:
        plan_id, aliases = record["consumed_plan_id"], record["blocked_aliases"]
        _require(isinstance(aliases, list) and record["replay_permitted"] is False,
                 "Invalid consumed record")
        ids = [plan_id, *aliases]
        _require(all(isinstance(x, str) and _ID.fullmatch(x) for x in ids)
                 and len(set(ids)) == len(ids) and not set(ids) & consumed,
                 "Duplicate or invalid consumed identity")
        consumed.update(ids)
        ledger_pin = relocated(record["ledger"])
        identity = ledger_pin["sha256"], ledger_pin["bytes"]
        _require(identity in prior and identity not in seen_ledgers,
                 "Catalog ledger must be an already counted original exactly once")
        seen_ledgers.add(identity)
        rows = [json.loads(x) for x in _read(ledger_pin["path"]).splitlines()]
        charges = [r for r in rows if r.get("event") == "request_started"]
        _require(len(charges) == record["charged_intents"]
                 and sum(not r.get("plan_id") for r in charges) == record["missing_row_plan_id_count"]
                 and all(not r.get("plan_id") or r["plan_id"] in ids for r in rows),
                 "Catalog ledger identity/count mismatch")
        for charge in charges:
            if charge.get("accounting_only") is True:
                _require(isinstance(charge.get("original_ledger"), dict)
                         and charge.get("original_plan") == record["plan"],
                         "Accounting bridge must bind original ledger and consumed plan")
                original = relocated(charge["original_ledger"])
                _require((original["sha256"], original["bytes"]) not in prior,
                         "Original membership ledger must not also be charged")
                # Preserve all explicitly pinned bridge evidence, including the
                # original membership ledger, without interpreting it as charges.
                for key, value in charge.items():
                    if key.startswith("original_") and isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"}:
                        relocated(value)
        total += len(charges)
        plan_pin = relocated(record["plan"])
        plan = json.loads(_read(plan_pin["path"]))
        _require(plan.get("plan_id", plan.get("id")) == plan_id, "Consumed plan identity mismatch")
        _require(plan_pin["sha256"] not in hashes, "Duplicate consumed plan bytes")
        hashes.add(plan_pin["sha256"])
        _require(bool(record["consumed_evidence"]), "Consumed evidence required")
        for original in record["consumed_evidence"]:
            proof = json.loads(_read(relocated(original)["path"]))
            proof_plan = proof.get("plan", proof.get("plan_pin", {}))
            exact_nested = all(proof_plan.get(k) == record["plan"][k] for k in ("bytes", "sha256"))
            exact_flat = (proof.get("plan_id") == plan_id
                          and proof.get("plan_sha256") == record["plan"]["sha256"])
            _require(proof.get("rerun_permitted") is False and (exact_nested or exact_flat),
                     "Consumed proof does not bind exact original plan")
        if record["terminal"] is not None:
            relocated(record["terminal"])
    _require(seen_ledgers == prior and total == manifest["prior_total"] == catalog["charged_total"],
             "Catalog must cover exact carried ledger scope")
    return sorted(consumed), sorted(hashes)


class HostedControlStore:
    """One-volume control store; all public mutations take an exclusive flock."""

    def __init__(self, root, *, mutation_guard=None):
        _require(os.name == "posix", "Linux/POSIX durable store required")
        self.root = _safe(root)
        _require(self.root.is_dir(), "Existing initialized store required")
        self._poisoned = False
        # Explicit local custody connection only; existing callers remain unchanged.
        # The guard must use independent storage and must not reenter this store.
        self.mutation_guard = mutation_guard

    def _confirm_mutation(self, method, *args, expected):
        if self.mutation_guard is None:
            return
        # Callback-owned objects must never alias the authoritative event/head.
        arguments = json.loads(_json(args))
        expected = json.loads(_json(expected))
        result = getattr(self.mutation_guard, method)(*arguments)
        _require(isinstance(result, dict) and all(result.get(k) == v for k, v in expected.items()),
                 "Exact independent mutation acknowledgment required")

    @classmethod
    def create(cls, root, *, prior_total, prior_rosterwatch, prior_ledger_pins,
               policy, now=None):
        """Create once; caller must independently reconcile the imported counts.

        Imported charges count in every rolling window from import time, a
        conservative fallback until timestamped historical migration is reviewed.
        No implicit zero baseline or reusable historical execution authority exists.
        """
        _require(os.name == "posix", "Linux/POSIX durable store required")
        policy = _policy(policy)
        _require(type(prior_total) is int and type(prior_rosterwatch) is int
                 and 0 <= prior_rosterwatch <= prior_total <= policy["total_ceiling"]
                 and prior_rosterwatch <= policy["rosterwatch_ceiling"], "Invalid carried counts")
        _require(isinstance(prior_ledger_pins, list) and bool(prior_ledger_pins),
                 "Pinned accounting evidence required even for an empty baseline")
        _require(len({x["path"] for x in prior_ledger_pins}) == len(prior_ledger_pins),
                 "Duplicate imported evidence")
        total, rosterwatch, imported_plan_ids = _imported_accounting(prior_ledger_pins)
        _require((total, rosterwatch) == (prior_total, prior_rosterwatch),
                 "Imported ledger counts do not match carried accounting")
        root = _safe(root)
        _require(not root.exists() and root.parent.is_dir(), "Fresh store root required")
        root.mkdir(mode=0o700)
        _fsync_dir(root.parent)
        for name in ("claims", "blobs"):
            (root / name).mkdir(mode=0o700)
        manifest = {"schema": 1, "prior_total": prior_total,
                    "prior_rosterwatch": prior_rosterwatch,
                    "prior_ledger_pins": prior_ledger_pins,
                    "imported_plan_ids": imported_plan_ids,
                    "policy": policy, "created_at": _time(now).isoformat()}
        _new_file(root / "manifest.json", _json(manifest), immutable=True)
        _new_file(root / "lock", b"")
        _new_file(root / "journal.jsonl", b"")
        _new_file(root / "head.json", _json({"sequence": 0, "sha256": _ZERO,
                                            "manifest_sha256": _sha(_json(manifest))}))
        _fsync_dir(root)
        store = cls(root)
        store.snapshot()
        return store

    @contextmanager
    def _locked(self):
        import fcntl
        _require(not self._poisoned, "Prior write failure: reopen and inspect recovery")
        fd = os.open(_safe(self.root / "lock"), os.O_RDWR | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Invalid lock file")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StoreBusy("Another writer owns the durable store") from exc
            yield
        finally:
            os.close(fd)

    def _load(self):
        try:
            for name in ("claims", "blobs"):
                _require(_safe(self.root / name).is_dir(), "Required store directory is absent")
            _require({p.name for p in self.root.iterdir()} ==
                     {"claims", "blobs", "lock", "manifest.json", "journal.jsonl", "head.json"},
                     "Unexpected store entry or unfinished write; preserve and review")
            _require(not (self.root / "head.next").exists(), "Unfinished head write; preserve and review")
            manifest_body = _read(self.root / "manifest.json")
            manifest = json.loads(manifest_body)
            _require(manifest["schema"] == 1, "Unsupported store schema")
            _policy(manifest["policy"])
            imported = _imported_accounting(manifest["prior_ledger_pins"])
            _require(imported == (manifest["prior_total"], manifest["prior_rosterwatch"],
                                  manifest["imported_plan_ids"]), "Imported accounting drift")
            body = _read(self.root / "journal.jsonl")
            _require(not body or body.endswith(b"\n"), "Torn journal; preserve and review")
            events, previous, claims, charges, results = [], _ZERO, {}, {}, {}
            consumed_ids, consumed_hashes, migrations, slots = set(manifest["imported_plan_ids"]), set(), [], {}
            stages = {}
            last_time = _time(manifest["created_at"])
            for sequence, line in enumerate(body.splitlines(), 1):
                row = json.loads(line)
                digest = row.pop("sha256")
                _require(row["sequence"] == sequence and row["previous"] == previous
                         and digest == _sha(_json(row)), "Journal chain mismatch")
                _require(_time(row["at_utc"]) >= last_time, "Journal time moved backwards")
                last_time, previous = _time(row["at_utc"]), digest
                event = row["event"]
                if event == "plan_claimed":
                    plan_id = row["plan_id"]
                    _require(plan_id not in claims and plan_id not in consumed_ids
                             and row["plan_pin"]["sha256"] not in consumed_hashes,
                             "Duplicate consumed plan")
                    claims[plan_id] = row
                elif event == "request_charged":
                    request_id = row["request_id"]
                    _require(row["plan_id"] in claims and request_id not in charges,
                             "Invalid or repeated request charge")
                    _require(row["source"] in _SOURCES, "Unknown charged source")
                    charges[request_id] = row
                elif event == "request_result":
                    request_id = row["request_id"]
                    _require(request_id in charges and request_id not in results,
                             "Invalid or duplicate request result")
                    blob = row["blob"]
                    _require(blob["path"] == (self.root / "blobs" / (blob["sha256"] + ".bin")).as_posix(),
                             "Blob path outside store")
                    verify_pin(blob)
                    _require(row["outcome"] in {"complete", "failed", "partial", "unknown"},
                             "Invalid saved outcome")
                    if "raw_response" in row:
                        raw_pin = row["raw_response"]
                        _require(raw_pin["path"] == (self.root / "blobs" / (raw_pin["sha256"] + ".bin")).as_posix(),
                                 "Raw response path outside store")
                        verify_pin(raw_pin)
                        _require(raw_pin["bytes"] <= 8388609, "Saved raw response exceeds limit")
                    if "envelope" in row:
                        pin = row["envelope"]
                        _require(pin["path"] == (self.root / "blobs" / (pin["sha256"] + ".bin")).as_posix(),
                                 "Envelope path outside store")
                        verify_pin(pin)
                        envelope = json.loads(_read(pin["path"]))
                        charge = charges[request_id]
                        expected = {"schema": 1, "request_id": request_id,
                            "source": charge["source"], "method": charge["method"], "url": charge["url"],
                            "outcome": row["outcome"], "raw_body_sha256": blob["sha256"],
                            "metadata": envelope["metadata"]}
                        if "raw_response" in row:
                            expected["raw_response_sha256"] = row["raw_response"]["sha256"]
                        _require(envelope == expected, "Saved envelope charge/body binding mismatch")
                    results[request_id] = row
                elif event == "consumed_catalog_migrated":
                    _require(not migrations, "Consumed catalog already migrated")
                    ids, hashes = _consumed_catalog(row["catalog_pin"], row["review_pin"],
                                                    row["evidence_pins"], manifest)
                    _require(row["consumed_ids"] == ids and row["consumed_hashes"] == hashes
                             and not set(ids) & set(claims)
                             and not set(hashes) & {r["plan_pin"]["sha256"] for r in claims.values()},
                             "Migration conflicts with consumed state")
                    consumed_ids.update(ids)
                    consumed_hashes.update(hashes)
                    migrations.append(row)
                elif event == "plan_slot_reserved":
                    _require(row["slot_id"] not in slots and _ID.fullmatch(row["slot_id"])
                             and _HASH.fullmatch(row["content_sha256"])
                             and re.fullmatch(r"[a-f0-9]{32}", row["nonce"]), "Invalid/repeated plan slot")
                    verify_pin(row["policy_pin"])
                    slots[row["slot_id"]] = dict(row, sha256=digest)
                elif event == "plan_stage_reserved":
                    self._validate_stage_reservation(row, slots, stages, manifest, charges)
                    stages.setdefault(row["slot_id"], []).append(dict(row, sha256=digest))
                    _require(row["stage_id"] not in consumed_ids and row["stage_id"] not in claims,
                             "Prepared stage identity already consumed")
                    consumed_ids.add(row["stage_id"])
                else:
                    raise StoreError("Unknown journal event")
                events.append(dict(row, sha256=digest))
            head = json.loads(_read(self.root / "head.json"))
            _require(head == {"sequence": len(events), "sha256": previous,
                              "manifest_sha256": _sha(manifest_body)},
                     "Head/journal mismatch; preserve and review")
            expected_claims = {key + ".json" for key in claims}
            _require({p.name for p in (self.root / "claims").iterdir()} == expected_claims,
                     "Orphan or missing consumed marker; preserve and review")
            for plan_id, row in claims.items():
                marker = json.loads(_read(self.root / "claims" / (plan_id + ".json")))
                _require(marker == {key: row[key] for key in ("plan_id", "plan_pin", "review_pin", "at_utc")},
                         "Consumed marker drift")
            expected_blobs = {r[key]["sha256"] + ".bin" for r in results.values()
                              for key in ("blob", "envelope", "raw_response") if key in r}
            _require({p.name for p in (self.root / "blobs").iterdir()} == expected_blobs,
                     "Orphan or missing blob; preserve and review")
            self._confirm_mutation("check", head, expected={"phase": "current", "store_head": head})
            return {"manifest": manifest, "head": head, "events": events,
                    "claims": claims, "charges": charges, "results": results,
                    "consumed_ids": consumed_ids, "consumed_hashes": consumed_hashes,
                    "migrations": migrations, "slots": slots, "stages": stages,
                    "last_time": last_time}
        except StoreError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise StoreError("Invalid durable state; preserve all files for review") from exc

    def _append(self, state, event, now, *, prepare_write=None):
        _require(now >= state["last_time"], "Runtime clock moved backwards")
        row = dict(event, sequence=state["head"]["sequence"] + 1,
                   previous=state["head"]["sha256"], at_utc=now.isoformat())
        digest = _sha(_json(row))
        row["sha256"] = digest
        try:
            self._confirm_mutation("before", state["head"], row, expected={"phase": "intent",
                "previous_head": state["head"], "event_sha256": digest})
            if prepare_write is not None:
                prepare_write()
            fd = os.open(_safe(self.root / "journal.jsonl"), os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
            try:
                with os.fdopen(fd, "ab", closefd=False) as stream:
                    stream.write(_json(row) + b"\n")
                    stream.flush()
                    os.fsync(fd)
            finally:
                os.close(fd)
            head = {"sequence": row["sequence"], "sha256": digest,
                    "manifest_sha256": state["head"]["manifest_sha256"]}
            _new_file(self.root / "head.next", _json(head))
            os.replace(self.root / "head.next", self.root / "head.json")
            _fsync_dir(self.root)
            self._confirm_mutation("after", head, row, expected={"phase": "committed",
                "store_head": head, "event_sha256": digest})
        except BaseException:
            self._poisoned = True
            raise
        return row

    def snapshot(self):
        """Validate recovery and return conservative accounting without mutation."""
        with self._locked():
            state = self._load()
            manifest = state["manifest"]
            total = manifest["prior_total"] + len(state["charges"])
            rosterwatch = manifest["prior_rosterwatch"] + sum(
                r["source"] == "roster.watch" for r in state["charges"].values())
            return {"head": state["head"], "total_charged": total,
                    "rosterwatch_charged": rosterwatch,
                    "consumed_plans": sorted(set(state["claims"]) | state["consumed_ids"]),
                    "historical_migrations": state["migrations"], "plan_slots": state["slots"],
                    "plan_stages": state["stages"],
                    "pending_requests": sorted(set(state["charges"]) - set(state["results"])),
                    "uncertain_requests": sorted(self._uncertain(state)),
                    "results": state["results"], "requests": state["charges"], "policy": manifest["policy"]}

    @staticmethod
    def _uncertain(state):
        return ((set(state["charges"]) - set(state["results"])) |
                {key for key, row in state["results"].items() if row["outcome"] == "unknown"})

    def claimed_authority(self, plan_id):
        """Return exact fresh claim pins for an independently controlled broker."""
        with self._locked():
            state = self._load()
            _require(plan_id in state["claims"], "No fresh durable claim for this plan")
            row = state["claims"][plan_id]
            return {"plan_id": plan_id, "plan_pin": row["plan_pin"], "review_pin": row["review_pin"]}

    def claim_plan(self, plan_id, plan_pin, review_pin, *, now=None):
        """Consume an exact plan once before any child work; failures stay consumed.

        The supervisor, not this storage primitive, verifies review semantics,
        owner authority, code/input pins and the complete plan request allowlist.
        """
        _require(isinstance(plan_id, str) and _ID.fullmatch(plan_id), "Invalid plan ID")
        verify_pin(plan_pin)
        verify_pin(review_pin)
        now = _time(now)
        with self._locked():
            state = self._load()
            _require(not self._uncertain(state), "Uncertain prior request; stop for reviewed recovery")
            _require(plan_id not in state["claims"]
                     and plan_id not in state["consumed_ids"], "Plan already consumed")
            _require(plan_pin["sha256"] not in state["consumed_hashes"], "Historical plan bytes already consumed")
            _require(not any(row["plan_pin"]["sha256"] == plan_pin["sha256"]
                             for row in state["claims"].values()), "Plan bytes already consumed")
            _require(now >= state["last_time"], "Runtime clock moved backwards")
            marker = {"plan_id": plan_id, "plan_pin": plan_pin,
                      "review_pin": review_pin, "at_utc": now.isoformat()}
            try:
                if self.mutation_guard is None:
                    _new_file(self.root / "claims" / (plan_id + ".json"), _json(marker), immutable=True)
                    return self._append(state, dict(marker, event="plan_claimed"), now)
                return self._append(state, dict(marker, event="plan_claimed"), now,
                    prepare_write=lambda: _new_file(self.root / "claims" / (plan_id + ".json"),
                        _json(marker), immutable=True))
            except BaseException:
                self._poisoned = True
                raise

    def charge_request(self, plan_id, request_id, source, method, url, *, now=None):
        """Return only after durable pre-send charge; no failure ever refunds it."""
        from urllib.parse import urlsplit
        _require(isinstance(request_id, str) and _ID.fullmatch(request_id), "Invalid request ID")
        _require(source in _SOURCES and method in {"GET", "POST"}, "Invalid source/method")
        parsed = urlsplit(url)
        hosts = {"roster.watch", "www.roster.watch"} if source == "roster.watch" else {source}
        _require(parsed.hostname in hosts and parsed.scheme in {"http", "https"}
                 and not parsed.username and not parsed.password and not parsed.fragment
                 and parsed.port is None, "Request URL/source mismatch")
        _require(source != "roster.watch" or (parsed.scheme == "https" and method == "GET"),
                 "Rosterwatch requires HTTPS GET")
        now = _time(now)
        with self._locked():
            state = self._load()
            _require(not self._uncertain(state), "Uncertain prior request; stop for reviewed recovery")
            _require(plan_id in state["claims"], "Plan must be consumed first")
            _require(request_id not in state["charges"], "Request already charged; do not resend")
            claim = state["claims"][plan_id]
            verify_pin(claim["plan_pin"])
            verify_pin(claim["review_pin"])
            manifest, charges = state["manifest"], list(state["charges"].values())
            policy = manifest["policy"]
            _require(manifest["prior_total"] + len(charges) < policy["total_ceiling"], "Total request ceiling")
            rw = manifest["prior_rosterwatch"] + sum(r["source"] == "roster.watch" for r in charges)
            _require(source != "roster.watch" or rw < policy["rosterwatch_ceiling"], "Rosterwatch request ceiling")
            _require(sum(r["plan_id"] == plan_id for r in charges) < policy["plan_ceiling"], "Plan request ceiling")
            for label, seconds in _WINDOWS.items():
                cutoff = now - timedelta(seconds=seconds)
                prior = manifest["prior_total"] if _time(manifest["created_at"]) > cutoff else 0
                used = prior + sum(_time(r["at_utc"]) > cutoff for r in charges)
                _require(used < policy[label + "_ceiling"], "Rolling " + label + " request ceiling")
            if source == "roster.watch":
                cutoff = now - timedelta(days=1)
                prior = manifest["prior_rosterwatch"] if _time(manifest["created_at"]) > cutoff else 0
                used = prior + sum(r["source"] == source and _time(r["at_utc"]) > cutoff for r in charges)
                _require(used < policy["rosterwatch_day_ceiling"], "Rolling rosterwatch day ceiling")
            return self._append(state, {"event": "request_charged", "plan_id": plan_id,
                "request_id": request_id, "source": source, "method": method, "url": url}, now)

    def record_result(self, request_id, outcome, raw_body, *, envelope=None, raw_response=None, now=None):
        """Preserve successful, partial and failed response bytes identically."""
        _require(outcome in {"complete", "failed", "partial", "unknown"}, "Invalid request outcome")
        # One extra retained byte is an oversize-response witness, not permission
        # to accept an oversized successful document.
        _require(isinstance(raw_body, bytes) and len(raw_body) <= 8388609
                 and (len(raw_body) <= 8388608 or outcome != "complete"), "Invalid raw response bytes")
        _require(raw_response is None or isinstance(raw_response, bytes) and len(raw_response) <= 8388609,
                 "Invalid raw HTTP response bytes")
        if envelope is not None:
            _require(isinstance(envelope, dict), "Response metadata must be a JSON object")
            try:
                envelope = json.loads(_json(envelope))
            except (ValueError, TypeError) as exc:
                raise StoreError("Invalid response metadata") from exc
            _require(len(_json(envelope)) <= 2097152, "Response metadata exceeds limit")
        now = _time(now)
        with self._locked():
            state = self._load()
            _require(request_id in state["charges"] and request_id not in state["results"],
                     "Uncharged request or result already recorded")
            _require(now >= state["last_time"], "Runtime clock moved backwards")
            try:
                bodies = {}
                def planned_blob(raw):
                    path = self.root / "blobs" / (_sha(raw) + ".bin")
                    bodies[path] = raw
                    return {"path": path.as_posix(), "bytes": len(raw), "sha256": _sha(raw)}
                pin = planned_blob(raw_body)
                event = {"event": "request_result", "request_id": request_id,
                         "outcome": outcome, "blob": pin}
                if raw_response is not None:
                    event["raw_response"] = planned_blob(raw_response)
                if envelope is not None:
                    charge = state["charges"][request_id]
                    wrapped = {"schema": 1, "request_id": request_id,
                        "source": charge["source"], "method": charge["method"], "url": charge["url"],
                        "outcome": outcome, "raw_body_sha256": pin["sha256"], "metadata": envelope}
                    if "raw_response" in event:
                        wrapped["raw_response_sha256"] = event["raw_response"]["sha256"]
                    event["envelope"] = planned_blob(_json(wrapped))
                def write_blobs():
                    for path, raw in bodies.items():
                        if not path.exists():
                            _new_file(path, raw, immutable=True)
                        actual = verify_pin(file_pin(path))
                        _require(actual["sha256"] == _sha(raw) and actual["bytes"] == len(raw), "Blob drift")
                if self.mutation_guard is None:
                    write_blobs()
                    return self._append(state, event, now)
                return self._append(state, event, now, prepare_write=write_blobs)
            except BaseException:
                self._poisoned = True
                raise

    def migrate_consumed_catalog(self, catalog_pin, review_pin, *, evidence_pins, now=None):
        """Block reviewed historical identities/bytes once without charging again."""
        now = _time(now)
        with self._locked():
            state = self._load()
            _require(not state["migrations"] and not self._uncertain(state),
                     "Prior migration or uncertain accounting prevents migration")
            ids, hashes = _consumed_catalog(catalog_pin, review_pin, evidence_pins, state["manifest"])
            _require(not set(ids) & set(state["claims"])
                     and not set(hashes) & {r["plan_pin"]["sha256"] for r in state["claims"].values()},
                     "Migration conflicts with fresh claims")
            return self._append(state, {"event": "consumed_catalog_migrated",
                "catalog_pin": catalog_pin, "review_pin": review_pin, "evidence_pins": evidence_pins,
                "consumed_ids": ids, "consumed_hashes": hashes, "new_charges": 0}, now)

    def reserve_plan_slot(self, slot_id, policy_pin, *, content_sha256, now=None):
        """Persist a unique preparation slot and nonce; this grants no send authority.

        Failed issuance stays reserved. Reconciliation uses a fresh eligible
        slot; no API removes reservations or resets cumulative accounting.
        """
        _require(isinstance(slot_id, str) and _ID.fullmatch(slot_id), "Invalid slot ID")
        _require(isinstance(content_sha256, str) and _HASH.fullmatch(content_sha256),
                 "Invalid slot content hash")
        verify_pin(policy_pin)
        now = _time(now)
        with self._locked():
            state = self._load()
            _require(not self._uncertain(state), "Uncertain accounting prevents plan issuance")
            _require(slot_id not in state["slots"], "Plan slot already reserved")
            return self._append(state, {"event": "plan_slot_reserved", "slot_id": slot_id,
                "policy_pin": policy_pin, "content_sha256": content_sha256,
                "nonce": os.urandom(16).hex()}, now)

    def _validate_stage_reservation(self, row, slots, stages, manifest, charges):
        """Replay the disabled preparation chain and its nonrefundable ceiling."""
        verify_pin(row["root_plan_pin"])
        verify_pin(row["stage_pin"])
        root = json.loads(_read(row["root_plan_pin"]["path"]))
        stage = json.loads(_read(row["stage_pin"]["path"]))
        slot = slots.get(row["slot_id"])
        prior = stages.get(row["slot_id"], [])
        _require(slot is not None and root["slot_id"] == row["slot_id"]
                 and root["slot_reservation"] == slot
                 and root["content"]["policy_pin"] == slot["policy_pin"]
                 and _sha(_json(root["content"])) == slot["content_sha256"]
                 and root["plan_id"] == row["slot_id"] + "-" + slot["nonce"]
                 and root["network_requests"] == 0 and root["activation_authorized"] is False,
                 "Stage root is not the original disabled reserved slot")
        from src.data.roster_hosted_plan import _policy as recurring_policy, _requests, _slot
        policy = recurring_policy(slot["policy_pin"], slot["policy_pin"]["sha256"])
        root_cost, root_rw = _requests(root["content"]["proposed_requests"], policy)
        cost, rw_cost = _requests(stage["proposed_requests"], policy)
        _require(root["schema"] == 1 and root["mode"] == "disabled_preparation"
                 and root["publication_requested"] is False and root["nonce"] == slot["nonce"]
                 and root["issued_at"] == slot["at_utc"]
                 and (_time(root["expires_at"]) - _time(root["issued_at"])).total_seconds() == policy["plan_ttl_seconds"]
                 and _slot(policy, _time(root["issued_at"]), root["content"]["recovery_evidence_pin"]) == row["slot_id"]
                 and type(root["maximum_future_charges"]) is int and root["maximum_future_charges"] == root_cost
                 and root["content"]["accounting_before"]["sha256"] == slot["previous"]
                 and root["content"]["accounting_before"]["sequence"] + 1 == slot["sequence"]
                 and root["content"].get("store_root") == self.root.as_posix()
                 and policy["store_policy"] == manifest["policy"]
                 and stage.get("schema") == 1 and stage.get("mode") == "disabled_stage_preparation"
                 and stage.get("publication_authorized") is False and stage.get("store_root") == self.root.as_posix()
                 and row["maximum_charges"] == cost and row["maximum_rosterwatch"] == rw_cost,
                 "Stage/root disabled policy, lifetime, namespace or recomputed cost drift")
        expected_previous = prior[-1]["sha256"] if prior else slot["sha256"]
        _require(row["ordinal"] == len(prior) + 1 and row["previous_stage_sha256"] == expected_previous
                 and row["stage_id"] == root["plan_id"] + "-stage-" + str(row["ordinal"])
                 and stage["stage_id"] == row["stage_id"] and stage["ordinal"] == row["ordinal"]
                 and stage["previous_stage_sha256"] == expected_previous
                 and stage["root_plan_pin"] == row["root_plan_pin"]
                 and stage["network_requests"] == 0 and stage["activation_authorized"] is False
                 and stage["maximum_future_charges"] == row["maximum_charges"]
                 and stage["maximum_rosterwatch_charges"] == row["maximum_rosterwatch"],
                 "Stage sequence/content binding drift")
        for key in ("maximum_charges", "maximum_rosterwatch"):
            _require(type(row[key]) is int and row[key] >= 0, "Invalid stage budget")
        _require(row["maximum_rosterwatch"] <= row["maximum_charges"], "Invalid stage source budget")
        total = root["maximum_future_charges"] + sum(r["maximum_charges"] for r in prior) + row["maximum_charges"]
        rw = sum(r["source"] == "roster.watch" for r in root["content"]["proposed_requests"])
        rw += sum(r["maximum_rosterwatch"] for r in prior) + row["maximum_rosterwatch"]
        policy = manifest["policy"]
        _require(total <= policy["plan_ceiling"]
                 and manifest["prior_total"] + len(charges) + total <= policy["total_ceiling"]
                 and manifest["prior_rosterwatch"] + sum(r["source"] == "roster.watch" for r in charges.values())
                     + rw <= policy["rosterwatch_ceiling"],
                 "Shared root-stage finite budget exhausted; reservations never reset")
        _require(_time(root["issued_at"]) <= _time(row["at_utc"]) <= _time(root["expires_at"]),
                 "Stage outside original root lifetime")

    def reserve_plan_stage(self, root_plan_pin, stage_pin, *, now=None):
        """Consume one disabled subordinate preparation without a new slot.

        Worst-case proposed charges reserve the root's shared finite ceiling;
        they are separate from actual charged sends and are never refunded here.
        Prepared stage IDs cannot be claimed as executable source plans.
        This is trusted-caller bookkeeping, like claim_plan. The stage adapter
        must verify semantic receipts/needs; this primitive verifies only the
        durable chain, disabled envelope and budgets. A semantic forgery can
        consume its preparation ID/budget but cannot grant execution authority.
        """
        verify_pin(root_plan_pin)
        verify_pin(stage_pin)
        now = _time(now)
        root = json.loads(_read(root_plan_pin["path"]))
        stage = json.loads(_read(stage_pin["path"]))
        with self._locked():
            state = self._load()
            _require(not self._uncertain(state), "Uncertain accounting prevents stage advancement")
            row = {"event": "plan_stage_reserved", "slot_id": root["slot_id"],
                   "root_plan_pin": root_plan_pin, "stage_pin": stage_pin,
                   "stage_id": stage["stage_id"], "ordinal": stage["ordinal"],
                   "previous_stage_sha256": stage["previous_stage_sha256"],
                   "maximum_charges": stage["maximum_future_charges"],
                   "maximum_rosterwatch": stage["maximum_rosterwatch_charges"], "at_utc": now.isoformat()}
            self._validate_stage_reservation(row, state["slots"], state["stages"], state["manifest"], state["charges"])
            _require(row["stage_id"] not in state["claims"] and row["stage_id"] not in state["consumed_ids"],
                     "Stage identity already consumed")
            return self._append(state, row, now)

    def retention_eligibility(self, *, now=None, held_digests=()):
        """Archive-review inventory only: no deletion or ledger compaction API.

        Journals, imported evidence, consumed markers, failures, ownership/history
        evidence and active source pins stay retained. Every raw blob remains
        necessary for this store's recovery until a separate archive protocol is
        reviewed and implemented.
        """
        now = _time(now)
        with self._locked():
            state = self._load()
            output = []
            for request_id, row in state["results"].items():
                age = (now - _time(row["at_utc"])).total_seconds()
                held = any(row[key]["sha256"] in held_digests
                           for key in ("blob", "envelope", "raw_response") if key in row)
                old = age >= state["manifest"]["policy"]["min_archive_age_seconds"]
                output.append({"request_id": request_id, "blob": row["blob"],
                    "envelope": row.get("envelope"),
                    "raw_response": row.get("raw_response"),
                    "eligible_for_archive_review": bool(old and not held),
                    "deletion_allowed": False, "held": held,
                    "reason": "journal_evidence_required"})
            return output
