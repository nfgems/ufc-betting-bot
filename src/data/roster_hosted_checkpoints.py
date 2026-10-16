"""Bounded local retention callback for the disabled roster controller.

The caller separately provisions and pins the binding and empty directory layout.
This module does not install controls or establish independent custody: same-UID
files and even separate filesystems are not an independent security boundary.
It protects accounting ancestry at retained callbacks, not every intervening
claim/charge, and is not a complete-volume backup or a replay mechanism.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import stat

from src.data.roster_hosted_inputs import _accepted, _checkpoint
from src.data.roster_hosted_store import (
    StoreBusy, _fsync_dir, _json, _new_file, _read, _require, _safe, _sha, _time,
    file_pin, verify_pin,
)

# Conservative bounds for this new namespace, no larger than existing worker
# output bounds. They do not extend any worker/job budget or prove host capacity.
MAX_FILE_BYTES = 134217728
MAX_TOTAL_BYTES = 1073741824
MAX_ENTRIES = 10000
FREE_RESERVE_BYTES = 536870912
_ZERO = "0" * 64
_LINKS = {"binding", "plan", "review", "root_plan", "root_review",
          "root_plan_pin", "root_review_pin"}
_FLAGS = {"source_requests_authorized": 0, "activation_authorized": False,
          "publication_authorized": False, "replay_authorized": False,
          "independent_custody_proved": False, "hosted_durability_proved": False,
          "complete_volume_backup": False, "every_mutation_retained": False}


def native_identity(path, *, directory):
    """Describe a checked native object, without certifying its trust owner."""
    path = _safe(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | (os.O_DIRECTORY if directory else 0))
    try:
        info = os.fstat(fd)
        _require(stat.S_ISDIR(info.st_mode) if directory else
                 stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Invalid native retention object")
        named = path.lstat()
        _require((info.st_dev, info.st_ino) == (named.st_dev, named.st_ino),
                 "Native retention object replaced while opening")
        _require(not info.st_mode & 0o077, "Private native retention object required")
        return {"path": path.as_posix(), "device": info.st_dev, "inode": info.st_ino,
                "uid": info.st_uid, "gid": info.st_gid, "mode": stat.S_IMODE(info.st_mode)}
    finally:
        os.close(fd)


def _bounded_pin(pin):
    _exact_body(pin)
    return pin


def _exact_body(pin):
    _require(isinstance(pin, dict) and set(pin) == {"path", "bytes", "sha256"}
             and type(pin.get("bytes")) is int
             and 0 <= pin["bytes"] <= MAX_FILE_BYTES, "Retention file byte ceiling")
    fd = os.open(_safe(pin["path"]), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                 and not info.st_mode & 0o222, "Immutable unshared retention evidence required")
        _require(info.st_size == pin["bytes"], "Pinned retention file drift")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(pin["bytes"] + 1)
        _require(len(raw) == pin["bytes"] and _sha(raw) == pin["sha256"],
                 "Pinned retention bytes drift")
        return raw
    finally:
        os.close(fd)


class RetainedCheckpoints:
    """Explicitly pinned local callback; no CLI/default or provisioning change.

    Provision a private root containing private `objects` and `records`
    directories, an empty regular `lock`, and mutable `head.json` containing
    {"sequence":0,"sha256":"0"*64}. The external binding pins their native
    identities, the mutable volume root, store path, accepted pointer/release,
    and initial checkpoint. The binding itself must be outside both roots.

    The separate expected_retention_head must be the exact latest externally
    acknowledged record head. An unacknowledged newer local head fails closed;
    explicit review is needed to supply a newer head after an interrupted ack.
    This API does not install that external acknowledgment plumbing.

    `retain(pin)` is compatible with persist_checkpoint; `current()` returns an
    explicitly pinned checkpoint for existing recovery APIs. Original byte
    copies can be inspected after source paths disappear, but are never restored
    or made executable here. Existing recovery still requires complete evidence.
    """

    def __init__(self, store, binding_pin, *, expected_binding_sha256,
                 expected_retention_head, clock=_time):
        self.store, self.binding_pin, self.clock = store, dict(binding_pin), clock
        self.expected_binding_sha256 = expected_binding_sha256
        _require(isinstance(expected_retention_head, dict)
                 and set(expected_retention_head) == {"sequence", "sha256"}
                 and type(expected_retention_head["sequence"]) is int
                 and 0 <= expected_retention_head["sequence"] <= MAX_ENTRIES
                 and isinstance(expected_retention_head["sha256"], str)
                 and len(expected_retention_head["sha256"]) == 64,
                 "Explicit external retained head required")
        self.expected_retention_head = dict(expected_retention_head)
        self._poisoned = False
        _bounded_pin(binding_pin)
        _require(binding_pin["sha256"] == expected_binding_sha256, "External retention binding hash mismatch")
        self.binding = json.loads(_exact_body(binding_pin))
        _require(isinstance(self.binding, dict) and set(self.binding) == {
            "schema", "kind", "directories", "lock", "mutable_root", "store_root",
            "accepted_pointer", "expected_pointer_sha256", "expected_release_sha256",
            "initial_checkpoint", "expected_checkpoint_sha256"}
            and self.binding["schema"] == 1
            and self.binding["kind"] == "local_roster_checkpoint_retention_v1",
            "Exact local retention binding required")
        directories = self.binding["directories"]
        _require(isinstance(directories, dict) and set(directories) == {"root", "objects", "records"},
                 "Exact retention directory identities required")
        self.root = _safe(directories["root"]["path"])
        self._boundary()

    def _boundary(self):
        _bounded_pin(self.binding_pin)
        _require(self.binding_pin["sha256"] == self.expected_binding_sha256
                 and json.loads(_exact_body(self.binding_pin)) == self.binding,
                 "Retention binding changed")
        mutable = _safe(self.binding["mutable_root"]["path"])
        _require(native_identity(mutable, directory=True) == self.binding["mutable_root"],
                 "Mutable volume native identity changed")
        _require(self.root != mutable and self.root not in mutable.parents and mutable not in self.root.parents,
                 "Retention and mutable volume must be disjoint")
        bound_path = _safe(self.binding_pin["path"])
        _require(not any(root == bound_path or root in bound_path.parents for root in (mutable, self.root)),
                 "Retention binding must be outside both roots")
        _require(self.store.root.as_posix() == self.binding["store_root"]
                 and (self.store.root == mutable or mutable in self.store.root.parents),
                 "Retention accounting namespace changed")
        for name in ("root", "objects", "records"):
            identity = self.binding["directories"][name]
            target = self.root if name == "root" else self.root / name
            _require(identity["path"] == target.as_posix()
                     and native_identity(target, directory=True) == identity,
                     "Retention directory native identity changed")
        _require(self.binding["lock"]["path"] == (self.root / "lock").as_posix()
                 and native_identity(self.root / "lock", directory=False) == self.binding["lock"],
                 "Retention lock native identity changed")

    @contextmanager
    def _locked(self):
        import fcntl
        _require(not self._poisoned, "Prior retention write failed; preserve and review")
        self._boundary()
        fd = os.open(self.root / "lock", os.O_RDWR | os.O_NOFOLLOW)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StoreBusy("Another writer owns checkpoint retention") from exc
            info = os.fstat(fd)
            _require((info.st_dev, info.st_ino) ==
                     (self.binding["lock"]["device"], self.binding["lock"]["inode"]),
                     "Retention lock replaced while opening")
            self._boundary()
            yield
        finally:
            os.close(fd)

    def _inventory(self):
        _require({p.name for p in self.root.iterdir()} == {"objects", "records", "head.json", "lock"},
                 "Unexpected retention entry or interrupted write; preserve and review")
        files, total, entries = {}, 0, 3
        for parent in (self.root, self.root / "objects", self.root / "records"):
            for path in parent.iterdir():
                if parent == self.root and path.name in {"objects", "records"}:
                    continue
                _safe(path)
                info = path.lstat()
                _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Unshared regular retention file required")
                _require(info.st_size <= MAX_FILE_BYTES, "Retention file byte ceiling")
                entries += 1
                total += info.st_size
                _require(entries <= MAX_ENTRIES and total <= MAX_TOTAL_BYTES, "Retention capacity exhausted")
                files[path] = info.st_size
        return files, total, entries

    def _load(self, now):
        files, total, entries = self._inventory()
        head = json.loads(_read(self.root / "head.json"))
        _require(isinstance(head, dict) and set(head) == {"sequence", "sha256"}
                 and type(head["sequence"]) is int and 0 <= head["sequence"] <= MAX_ENTRIES,
                 "Invalid retained head")
        _require(head == self.expected_retention_head,
                 "Retention rollback or unacknowledged head conflicts with externally supplied exact head")
        state = self.store.snapshot()
        pointer, _ = _accepted(self.store, self.binding["accepted_pointer"],
            self.binding["expected_pointer_sha256"], self.binding["expected_release_sha256"])
        _require(pointer["store_manifest_sha256"] == state["head"]["manifest_sha256"],
                 "Retention accepted accounting manifest changed")
        checkpoint_pin = self.binding["initial_checkpoint"]
        _checkpoint(self.store, state, self.binding["accepted_pointer"], checkpoint_pin,
                    self.binding["expected_checkpoint_sha256"], now)
        previous, expected, original_paths, last = _ZERO, set(), {}, None
        for sequence in range(1, head["sequence"] + 1):
            path = self.root / "records" / (str(sequence).zfill(8) + ".json")
            _require(path in files, "Missing retained record; preserve and review")
            record_pin = _bounded_pin(file_pin(path))
            record = json.loads(_exact_body(record_pin))
            _require(isinstance(record, dict) and set(record) == {
                "schema", "sequence", "previous", "binding", "checkpoint", "requested", "originals", "scope"}
                and record["schema"] == 1 and record["sequence"] == sequence
                and record["previous"] == previous and record["binding"] == self.binding_pin
                and record["scope"] == _FLAGS and isinstance(record["originals"], list),
                "Retained record chain or scope changed")
            expected.add(path)
            checkpoint_pin = record["checkpoint"]
            for retained in [checkpoint_pin, *(row["saved"] for row in record["originals"])]:
                target = self.root / "objects" / (retained["sha256"] + ".bin")
                _require(retained["path"] == target.as_posix(), "Retained object namespace changed")
                _bounded_pin(retained)
                expected.add(target)
            originals = {}
            for row in record["originals"]:
                _require(isinstance(row, dict) and set(row) == {"original", "saved"}, "Invalid retained original")
                original, saved = row["original"], row["saved"]
                _require(isinstance(original, dict) and set(original) == {"path", "bytes", "sha256"}
                         and original["bytes"] == saved["bytes"] and original["sha256"] == saved["sha256"]
                         and original["path"] not in originals, "Retained original bytes or duplicate changed")
                _require(original["path"] not in original_paths or original_paths[original["path"]] == original,
                         "Original path has conflicting retained bytes")
                originals[original["path"]] = original
                original_paths[original["path"]] = original
            _require(originals.get(record["requested"]["path"]) == record["requested"],
                     "Requested original is missing from retention")
            saved_by_path = {row["original"]["path"]: row["saved"] for row in record["originals"]}
            pending, visited = [record["requested"]], set()
            while pending:
                item = pending.pop()
                _require(originals.get(item["path"]) == item, "Required retained original relationship missing")
                if item["path"] in visited:
                    continue
                visited.add(item["path"])
                value = json.loads(_exact_body(saved_by_path[item["path"]]))
                if isinstance(value, dict):
                    for key in _LINKS & value.keys():
                        nested = value[key]
                        if nested is not None:
                            _require(isinstance(nested, dict) and set(nested) == {"path", "bytes", "sha256"},
                                     "Malformed retained original relationship")
                            pending.append(nested)
            _require(visited == set(originals), "Unrelated retained originals changed traversal scope")
            checkpoint = _checkpoint(self.store, state, self.binding["accepted_pointer"],
                checkpoint_pin, checkpoint_pin["sha256"], now)
            if last:
                prior = json.loads(_exact_body(last["checkpoint"]))["store_head"]
                _require(checkpoint["store_head"]["sequence"] >= prior["sequence"], "Retained accounting head decreased")
            previous, last = record_pin["sha256"], dict(record, record_pin=record_pin)
            if sequence == self.expected_retention_head["sequence"]:
                _require(previous == self.expected_retention_head["sha256"],
                         "Retention conflicts with externally supplied head")
        if self.expected_retention_head["sequence"] == 0:
            _require(self.expected_retention_head["sha256"] == _ZERO, "Invalid external initial retained head")
        _require(head["sha256"] == previous, "Retained head does not match complete records")
        _require(set(files) == expected | {self.root / "head.json", self.root / "lock"},
                 "Orphan or unanchored retained evidence; preserve and review")
        return state, head, checkpoint_pin, last, original_paths, total, entries

    def current(self):
        """Validate retained ancestry against current complete accounting."""
        with self._locked():
            state, head, checkpoint, last, _, _, _ = self._load(_time(self.clock()))
            self._boundary()
            retained_head = json.loads(_exact_body(checkpoint))["store_head"]
            return {"checkpoint": checkpoint, "retention_head": head,
                    "record": last["record_pin"] if last else None,
                    "retained_store_head": retained_head, "current_store_head": state["head"],
                    "unretained_store_tail": retained_head != state["head"], "scope": dict(_FLAGS)}

    def retain(self, pin):
        """Fsync exact originals and current anchor before acknowledging callback."""
        with self._locked():
            now = _time(self.clock())
            state, head, _, _, original_paths, total, entries = self._load(now)
            pending, originals, bodies = [pin], {}, {}
            while pending:
                item = pending.pop()
                _bounded_pin(item)
                _require(item["path"] not in original_paths or original_paths[item["path"]] == item,
                         "Original path has conflicting retained bytes")
                if item["path"] in originals:
                    _require(originals[item["path"]]["original"] == item, "Conflicting original traversal")
                    continue
                raw = _exact_body(item)
                _require(len(originals) + entries + 3 <= MAX_ENTRIES, "Retention traversal entry ceiling")
                target = self.root / "objects" / (_sha(raw) + ".bin")
                saved = {"path": target.as_posix(), "bytes": len(raw), "sha256": _sha(raw)}
                originals[item["path"]] = {"original": dict(item), "saved": saved}
                bodies[target] = raw
                _require(sum(map(len, bodies.values())) + total <= MAX_TOTAL_BYTES, "Retention traversal byte ceiling")
                value = json.loads(raw)
                if isinstance(value, dict):
                    for key in _LINKS & value.keys():
                        nested = value[key]
                        if nested is not None:
                            _require(isinstance(nested, dict) and set(nested) == {"path", "bytes", "sha256"},
                                     "Malformed required original relationship")
                            pending.append(nested)
            checkpoint_raw = _json({"schema": 1, "kind": "roster_input_checkpoint_v1",
                "accepted_pointer": self.binding["accepted_pointer"], "store_root": self.store.root.as_posix(),
                "store_head": state["head"]})
            checkpoint = {"path": (self.root / "objects" / (_sha(checkpoint_raw) + ".bin")).as_posix(),
                          "bytes": len(checkpoint_raw), "sha256": _sha(checkpoint_raw)}
            bodies[Path(checkpoint["path"])] = checkpoint_raw
            sequence = head["sequence"] + 1
            record_raw = _json({"schema": 1, "sequence": sequence, "previous": head["sha256"],
                "binding": self.binding_pin, "checkpoint": checkpoint, "requested": pin,
                "originals": [originals[key] for key in sorted(originals)], "scope": _FLAGS})
            record_path = self.root / "records" / (str(sequence).zfill(8) + ".json")
            new_head = {"sequence": sequence, "sha256": _sha(record_raw)}
            bodies[record_path] = record_raw
            new = {path: raw for path, raw in bodies.items() if not path.exists()}
            for path in set(bodies) - set(new):
                _require(_read(path) == bodies[path], "Retained immutable object collision")
            added = sum(map(len, new.values())) + len(_json(new_head))
            _require(all(len(raw) <= MAX_FILE_BYTES for raw in bodies.values()), "Retention file byte ceiling")
            _require(total + added <= MAX_TOTAL_BYTES and entries + len(new) + 1 <= MAX_ENTRIES,
                     "Retention capacity exhausted")
            capacity = os.statvfs(self.root)
            _require(capacity.f_bavail * capacity.f_frsize >= added + FREE_RESERVE_BYTES,
                     "Retention free disk reserve exhausted")
            _require(self.store.snapshot() == state, "Accounting changed before retention write")
            self._boundary()
            try:
                for path, raw in new.items():
                    _new_file(path, raw, immutable=True)
                _new_file(self.root / "head.next", _json(new_head))
                _require(self.store.snapshot() == state, "Accounting changed during retention write")
                self._boundary()
                os.replace(self.root / "head.next", self.root / "head.json")
                _fsync_dir(self.root)
                self._boundary()
                _require(self.store.snapshot() == state, "Accounting changed before retention acknowledgment")
                record_pin = file_pin(record_path)
            except BaseException:
                self._poisoned = True
                raise
            self.expected_retention_head = dict(new_head)
            return {"checkpoint": checkpoint, "record": record_pin,
                    "retention_head": new_head, "retained_store_head": state["head"],
                    "current_store_head": state["head"], "unretained_store_tail": False,
                    "scope": dict(_FLAGS)}

    __call__ = retain
