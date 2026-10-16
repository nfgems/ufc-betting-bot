"""Explicit native-root custody for the disabled saved-input controller.

Provisioning and the authority run as native root; the controller runs as a
different UID and uses pipes opened before its process/network guard. No source
request, subprocess, socket, scheduler, restore, or default behavior is provided.
An unresolved write-ahead intent is deliberately a recovery blocker. This local
owner boundary does not establish hosted durability or complete-volume backup.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import select
import stat
import threading
import time

from src.data.roster_hosted_checkpoints import (
    RetainedCheckpoints, MAX_ENTRIES, MAX_FILE_BYTES, MAX_TOTAL_BYTES,
    FREE_RESERVE_BYTES, native_identity,
)
from src.data.roster_hosted_store import (
    HostedControlStore, StoreError, _fsync_dir, _json, _new_file, _read,
    _require, _safe, _sha, _time, file_pin, verify_pin,
)

_ZERO = {"sequence": 0, "sha256": "0" * 64}
MAX_MESSAGE_BYTES = 16777216
IPC_TIMEOUT_SECONDS = 30


def _copy(value):
    return json.loads(_json(value))


def _root():
    _require(os.name == "posix" and os.geteuid() == 0, "Native root custody authority required")


def _trusted(path, *, directory):
    """The controller cannot chmod or replace any component of this path."""
    path = _safe(path)
    for parent in path.parents:
        info = parent.lstat()
        _require(info.st_uid == 0 and stat.S_ISDIR(info.st_mode),
                 "Custody ancestry must be native root owned")
        # Root-owned sticky /tmp cannot remove a child owned by root. Every
        # other writable ancestor would permit replacement of the trust root.
        _require(not info.st_mode & 0o022 or bool(info.st_mode & stat.S_ISVTX),
                 "Writable custody ancestry")
    info = path.lstat()
    _require(info.st_uid == 0 and not info.st_mode & 0o022,
             "Custody object must be independently root owned")
    _require(stat.S_ISDIR(info.st_mode) if directory else
             stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
             "Native unshared custody object required")
    if not directory:
        _require(not info.st_mode & 0o222, "Immutable custody evidence required")
    return {"path": path.as_posix(), "device": info.st_dev, "inode": info.st_ino,
            "uid": info.st_uid, "gid": info.st_gid, "mode": stat.S_IMODE(info.st_mode)}


def _inside(path, roots):
    path = _safe(path)
    _require(any(path == root or root in path.parents for root in roots),
             "Accepted closure escaped explicit provisioning roots")
    return path


def _worker_absent(worker_uid):
    """Provision before the first worker exists, including writable descriptors."""
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            status = (path / "status").read_text()
        except FileNotFoundError:
            continue
        uid_line = next(line for line in status.splitlines() if line.startswith("Uid:"))
        _require(worker_uid not in [int(value) for value in uid_line.split()[1:]],
                 "Provisioning requires no existing controller UID process or inherited writer")


def _sync_file(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _accepted_closure(pointer, checkpoint, roots):
    # Follow the accepted-input schema, never arbitrary pins embedded in raw
    # captured envelopes/catalog/release provenance. Such historical paths can
    # identify consumed predecessor plans and are inert exact saved bytes.
    from src.data.roster_hosted_needs import BASE_ROLES
    visited = {}
    def accept(pin):
        path = _inside(pin["path"], roots)
        verify_pin(pin)
        _require(pin["bytes"] <= MAX_FILE_BYTES, "Accepted file byte ceiling")
        if pin["path"] in visited:
            _require(visited[pin["path"]] == pin, "Conflicting accepted closure pin")
        visited[pin["path"]] = pin
        _require(len(visited) <= MAX_ENTRIES, "Accepted closure entry ceiling")
        return path
    value = json.loads(_read(accept(pointer)))
    manifest = json.loads(_read(accept(value["input_manifest"])))
    _require(set(manifest["input_pins"]) == BASE_ROLES, "Complete accepted base roles required")
    base = manifest["input_pins"]
    for pin in base.values():
        accept(pin)
    owners = json.loads(_read(base["owners"]["path"]))
    membership = json.loads(_read(base["membership"]["path"]))
    # Tiny contract-only tests are not accepted semantic fixtures. Real roster
    # provisioning requires these structure links; no default or fabricated
    # ownership/membership data are supplied here.
    _require(set(owners) >= {"seed_root", "seed_inventory"}
             and set(membership) >= {"body", "envelope"}, "Complete accepted owner/membership closure required")
    seed = _inside(owners["seed_root"], roots)
    _require(seed.is_dir(), "Complete accepted seed directory required")
    for pin in (owners["seed_inventory"], membership["body"], membership["envelope"], checkpoint):
        accept(pin)
    return visited


def provision_native_custody(authority_root, *, store_root, retention_binding_pin,
                            worker_uid, accepted_roots, expected_pointer_sha256,
                            expected_release_sha256, clock=_time, controller_controls=None):
    """Freeze fresh explicitly supplied native inputs and install disabled controls.

    This does not relocate or rewrite pins or inventory metadata. Supply fresh
    root-owned trees already readable/traversable by the worker, with readonly
    files and intended native paths; never point it at accepted historical
    evidence. Parents of each tree must already be root owned.
    The mutable store/retention roots must be separately worker owned.
    """
    _root()
    _require(type(worker_uid) is int and worker_uid > 0, "Distinct non-root controller UID required")
    _worker_absent(worker_uid)
    authority_root = _safe(authority_root)
    _require(not authority_root.exists(), "Unused custody authority directory required")
    _trusted(authority_root.parent, directory=True)
    roots = [_safe(path) for path in accepted_roots]
    _require(bool(roots) and len(set(roots)) == len(roots), "Explicit accepted provisioning roots required")
    binding = json.loads(_read(verify_pin(retention_binding_pin)["path"]))
    _require(binding["store_root"] == _safe(store_root).as_posix()
             and binding["expected_pointer_sha256"] == expected_pointer_sha256
             and binding["expected_release_sha256"] == expected_release_sha256,
             "Externally accepted controls mismatch")
    mutable = _safe(binding["mutable_root"]["path"])
    retained = _safe(binding["directories"]["root"]["path"])
    _require(mutable.stat().st_uid == worker_uid and retained.stat().st_uid == worker_uid,
             "Explicit separate controller-owned mutable namespaces required")
    for root in roots:
        _trusted(root.parent, directory=True)
        _require(all(root != target and root not in target.parents and target not in root.parents
                     for target in (mutable, retained, authority_root)),
                 "Accepted and mutable/custody namespaces must be disjoint")
    closure = _accepted_closure(binding["accepted_pointer"], binding["initial_checkpoint"], roots)
    if controller_controls is not None:
        _require(isinstance(controller_controls, dict)
                 and set(controller_controls) == {"policy", "synthetic_responses"}
                 and controller_controls["policy"] is not None,
                 "Exact named controller policy and saved response controls required")
        for pin in controller_controls.values():
            if pin is not None:
                _inside(pin["path"], roots)
                verify_pin(pin)
                _require(pin["bytes"] <= MAX_FILE_BYTES, "Named controller control byte ceiling")
                closure[pin["path"]] = _copy(pin)
    inventory, total = [], 0
    for root in roots:
        for path in [root, *root.rglob("*")]:
            path = _safe(path)
            info = path.lstat()
            _require(stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                     "Accepted tree contains a link or special object")
            _require(info.st_uid == 0 and not info.st_mode & 0o022,
                     "Trusted root must prepare accepted input objects before controller launch")
            _require(bool(info.st_mode & (0o001 if path.is_dir() else 0o004))
                     and (path.is_dir() or not info.st_mode & 0o222),
                     "Prepare worker-readable immutable native inputs before pinning inventories")
            if path.is_file():
                _require(info.st_size <= MAX_FILE_BYTES, "Accepted file byte ceiling")
                total += info.st_size
            inventory.append(path)
            _require(len(inventory) <= MAX_ENTRIES and total <= MAX_TOTAL_BYTES,
                     "Accepted provisioning capacity exhausted")
    # All validation above precedes the explicit ownership mutation.
    for path in inventory:
        os.chown(path, 0, 0, follow_symlinks=False)
        if path.is_file():
            _sync_file(path)
    for path in reversed(inventory):
        if path.is_dir():
            _fsync_dir(path)
    identities = [_trusted(path, directory=path.is_dir()) for path in inventory]
    for pin in closure.values():
        verify_pin(pin)
    binding_path = _safe(retention_binding_pin["path"])
    _trusted(binding_path.parent, directory=True)
    os.chown(binding_path, 0, 0)
    binding_path.chmod(0o444)
    _sync_file(binding_path)
    _trusted(binding_path, directory=False)
    store = HostedControlStore(store_root)
    retention = RetainedCheckpoints(store, retention_binding_pin,
        expected_binding_sha256=retention_binding_pin["sha256"], expected_retention_head=_ZERO, clock=clock)
    current = retention.current()
    _require(not current["unretained_store_tail"], "Provisioning requires the exact initial accounting head")
    authority_root.mkdir(mode=0o700)
    (authority_root / "records").mkdir(mode=0o700)
    _new_file(authority_root / "lock", b"")
    controls = {"schema": 1, "kind": "native_root_roster_custody_v1", "worker_uid": worker_uid,
        "root": native_identity(authority_root, directory=True),
        "records": native_identity(authority_root / "records", directory=True),
        "lock": native_identity(authority_root / "lock", directory=False),
        "accepted_roots": [_trusted(path, directory=True) for path in roots],
        "accepted_inventory": identities, "accepted_pins": list(closure.values()),
        "controller_controls": _copy(controller_controls),
        "retention_binding": retention_binding_pin,
        "initial_store_head": current["current_store_head"],
        "initial_checkpoint": current["checkpoint"], "initial_retention_head": _ZERO,
        "activation_authorized": False, "publication_authorized": False,
        "schedule_enabled": False, "hosted_durability_proved": False}
    _new_file(authority_root / "controls.json", _json(controls), immutable=True)
    _new_file(authority_root / "head.json", _json(_ZERO))
    _fsync_dir(authority_root)
    _fsync_dir(authority_root.parent)
    return file_pin(authority_root / "controls.json")


class NativeCustodyAuthority:
    """Root-owned current acknowledgment and write-ahead uncertainty journal."""

    def __init__(self, controls_pin, *, expected_controls_sha256, clock=_time,
                 archive_checkpoint=None):
        _root()
        self.controls_pin = _copy(controls_pin)
        _trusted(controls_pin["path"], directory=False)
        verify_pin(controls_pin)
        _require(controls_pin["sha256"] == expected_controls_sha256, "External custody controls hash mismatch")
        self.controls = json.loads(_read(controls_pin["path"]))
        _require(self.controls.get("schema") == 1
                 and self.controls.get("kind") == "native_root_roster_custody_v1"
                 and type(self.controls.get("worker_uid")) is int and self.controls["worker_uid"] > 0,
                 "Exact native custody controls required")
        self.root = _safe(self.controls["root"]["path"])
        self.clock, self._poisoned = clock, False
        self.archive_checkpoint = archive_checkpoint
        self._boundary()
        for identity in self.controls["accepted_inventory"]:
            path = Path(identity["path"])
            _require(_trusted(path, directory=path.is_dir()) == identity,
                     "Provisioned accepted native object changed")
        for pin in self.controls["accepted_pins"]:
            verify_pin(pin)

    def _boundary(self):
        _root()
        _require(not self._poisoned, "Prior custody write failed; preserve and review")
        _trusted(self.controls_pin["path"], directory=False)
        verify_pin(self.controls_pin)
        for name in ("root", "records", "lock"):
            identity = self.controls[name]
            _require(native_identity(identity["path"], directory=name != "lock") == identity,
                     "Custody native identity changed")
            _require(identity["uid"] == 0, "Independent custody owner changed")
        _trusted(self.root, directory=True)
        for identity in self.controls["accepted_roots"]:
            _require(_trusted(identity["path"], directory=True) == identity,
                     "Accepted provisioning root changed")

    @contextmanager
    def _locked(self):
        import fcntl
        self._boundary()
        fd = os.open(self.root / "lock", os.O_RDWR | os.O_NOFOLLOW)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            info = os.fstat(fd)
            _require((info.st_dev, info.st_ino) ==
                     (self.controls["lock"]["device"], self.controls["lock"]["inode"]),
                     "Custody lock replaced")
            self._boundary()
            yield
        finally:
            os.close(fd)

    def _load(self):
        _require({p.name for p in self.root.iterdir()} == {"controls.json", "records", "lock", "head.json"},
                 "Unfinished custody write; preserve and review")
        head = json.loads(_read(self.root / "head.json"))
        _require(set(head) == {"sequence", "sha256"} and type(head["sequence"]) is int
                 and 0 <= head["sequence"] <= MAX_ENTRIES, "Invalid custody head")
        paths = list((self.root / "records").iterdir())
        _require(len(paths) == head["sequence"], "Unacknowledged custody tail or rollback; preserve and review")
        state = {"store_head": self.controls["initial_store_head"], "pending": None,
                 "retention_head": self.controls["initial_retention_head"], "retained": None}
        previous, last = _ZERO["sha256"], None
        for sequence in range(1, head["sequence"] + 1):
            path = self.root / "records" / (str(sequence).zfill(8) + ".json")
            _trusted(path, directory=False)
            raw = _read(path)
            _require(len(raw) <= MAX_FILE_BYTES, "Custody record byte ceiling")
            record = json.loads(raw)
            _require(set(record) == {"schema", "sequence", "previous", "controls", "event", "state"}
                     and record["schema"] == 1 and record["sequence"] == sequence
                     and record["previous"] == previous and record["controls"] == self.controls_pin,
                     "Custody record chain changed")
            # Independently replay transitions; state cannot erase an intent or
            # replace a committed head even when JSON remains hash consistent.
            state = self._transition(state, record["event"])
            _require(state == record["state"], "Custody state transition changed")
            previous, last = _sha(raw), file_pin(path)
        _require(head["sha256"] == previous, "Custody journal/head mismatch")
        return state, head, last

    def _transition(self, state, event):
        result = _copy(state)
        kind = event["kind"]
        if kind == "intent":
            row, previous = event["row"], event["previous_head"]
            controls = self.controls["controller_controls"]
            if controls is not None and row.get("event") == "plan_slot_reserved":
                _require(row.get("policy_pin") == controls["policy"],
                         "Reserved slot policy differs from independently provisioned controller policy")
            unsigned = {key: value for key, value in row.items() if key != "sha256"}
            _require(state["pending"] is None and previous == state["store_head"]
                     and row["sequence"] == previous["sequence"] + 1
                     and row["previous"] == previous["sha256"] and row["sha256"] == _sha(_json(unsigned)),
                     "Custody intent conflicts with exact committed head")
            result["pending"] = {"previous_head": previous, "row": row}
        elif kind == "commit":
            _require(state["pending"] is not None and state["pending"]["row"] == event["row"],
                     "Commit has no exact independent pending intent")
            expected = {"sequence": event["row"]["sequence"], "sha256": event["row"]["sha256"],
                        "manifest_sha256": state["store_head"]["manifest_sha256"]}
            _require(event["store_head"] == expected, "Committed accounting head conflicts with intent")
            result.update(store_head=expected, pending=None)
        elif kind == "retained":
            retained = event["result"]
            _require(state["pending"] is None and retained["current_store_head"] == state["store_head"]
                     and retained["retained_store_head"] == state["store_head"]
                     and retained["unretained_store_tail"] is False
                     and retained["retention_head"]["sequence"] == state["retention_head"]["sequence"] + 1,
                     "Retention acknowledgment conflicts with independent accounting head")
            result.update(retention_head=retained["retention_head"], retained=retained)
        else:
            raise StoreError("Unknown custody transition")
        return result

    def _append(self, state, head, event):
        next_state = self._transition(state, event)
        sequence = head["sequence"] + 1
        raw = _json({"schema": 1, "sequence": sequence, "previous": head["sha256"],
                     "controls": self.controls_pin, "event": event, "state": next_state})
        files = [p for p in self.root.rglob("*") if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        added = len(raw) + 256
        _require(len(raw) <= MAX_FILE_BYTES and total + added <= MAX_TOTAL_BYTES
                 and len(files) + 4 <= MAX_ENTRIES, "Custody capacity exhausted")
        capacity = os.statvfs(self.root)
        _require(capacity.f_bavail * capacity.f_frsize >= added + FREE_RESERVE_BYTES,
                 "Custody free disk reserve exhausted")
        record_path = self.root / "records" / (str(sequence).zfill(8) + ".json")
        new_head = {"sequence": sequence, "sha256": _sha(raw)}
        try:
            _new_file(record_path, raw, immutable=True)
            _new_file(self.root / "head.next", _json(new_head))
            os.replace(self.root / "head.next", self.root / "head.json")
            _fsync_dir(self.root)
            self._boundary()
        except BaseException:
            self._poisoned = True
            raise
        return {"head": new_head, "record": file_pin(record_path)}

    def _native_head(self):
        binding = json.loads(_read(self.controls["retention_binding"]["path"]))
        return json.loads(_read(Path(binding["store_root"]) / "head.json"))

    def controller_configuration(self):
        """Return exactly provisioned named controls, never a worker-selected pin."""
        with self._locked():
            controls = self.controls["controller_controls"]
            _require(isinstance(controls, dict) and set(controls) == {"policy", "synthetic_responses"},
                     "Named independent controller controls were not provisioned")
            for pin in controls.values():
                if pin is not None:
                    _trusted(pin["path"], directory=False)
                    verify_pin(pin)
            return _copy(controls)

    def check(self, head):
        with self._locked():
            state, authority_head, record = self._load()
            _require(state["pending"] is None, "Unresolved independent mutation intent; preserve uncertainty")
            _require(head == state["store_head"] == self._native_head(),
                     "Accounting rollback or unanchored tail conflicts with independent current head")
            return {"phase": "current", "store_head": _copy(head),
                    "custody_ack": {"head": authority_head, "record": record}}

    def before(self, previous_head, row):
        with self._locked():
            state, head, _ = self._load()
            _require(previous_head == self._native_head(), "Native accounting changed before intent")
            ack = self._append(state, head, {"kind": "intent", "previous_head": previous_head, "row": row})
            return {"phase": "intent", "previous_head": _copy(previous_head),
                    "event_sha256": row["sha256"], "custody_ack": ack}

    def after(self, new_head, row):
        with self._locked():
            state, head, _ = self._load()
            _require(new_head == self._native_head(), "Native accounting changed before commit")
            binding = json.loads(_read(self.controls["retention_binding"]["path"]))
            body = _read(Path(binding["store_root"]) / "journal.jsonl")
            _require(body.endswith(b"\n") and json.loads(body.splitlines()[-1]) == row,
                     "Exact native journal mutation absent before commit")
            ack = self._append(state, head, {"kind": "commit", "store_head": new_head, "row": row})
            return {"phase": "committed", "store_head": _copy(new_head),
                    "event_sha256": row["sha256"], "custody_ack": ack}

    def _retention(self, state):
        _require(state["pending"] is None, "Unresolved independent mutation intent; preserve uncertainty")
        binding_pin = self.controls["retention_binding"]
        binding = json.loads(_read(binding_pin["path"]))
        store = HostedControlStore(binding["store_root"])
        _require(store.snapshot()["head"] == state["store_head"],
                 "Accounting rollback or unanchored tail conflicts with independent current head")
        return RetainedCheckpoints(store, binding_pin, expected_binding_sha256=binding_pin["sha256"],
            expected_retention_head=state["retention_head"], clock=self.clock)

    def retain(self, pin):
        with self._locked():
            state, head, _ = self._load()
            if self.archive_checkpoint is not None:
                self.archive_checkpoint(pin)
            result = self._retention(state).retain(pin)
            # The existing selector reads this exact local retained checkpoint.
            # This namespace is explicitly controller owned; its independent
            # authority is the protected root acknowledgment, not chmod alone.
            os.chown(result["checkpoint"]["path"], self.controls["worker_uid"], -1)
            _sync_file(result["checkpoint"]["path"])
            result["requested"] = _copy(pin)
            ack = self._append(state, head, {"kind": "retained", "result": result})
            return dict(result, custody_ack=ack)

    __call__ = retain

    def current(self):
        with self._locked():
            state, head, record = self._load()
            result = self._retention(state).current()
            _require(result["retention_head"] == state["retention_head"], "Independent retained head changed")
            if state["retained"] is not None:
                _require(result["checkpoint"] == state["retained"]["checkpoint"],
                         "Recovery must use exact independently acknowledged checkpoint")
            return dict(result, custody_ack={"head": head, "record": record})

    def confirm_checkpoint(self, result, pin, store_head):
        with self._locked():
            state, head, record = self._load()
            expected = dict(state["retained"] or {}, custody_ack={"head": head, "record": record})
            _require(result == expected and result.get("requested") == pin
                     and state["store_head"] == store_head == self._native_head()
                     and state["pending"] is None, "Exact latest independent checkpoint acknowledgment required")
            self._retention(state).current()
            return True


def _pipe(fd):
    _require(type(fd) is int and fd >= 0 and stat.S_ISFIFO(os.fstat(fd).st_mode),
             "Preopened native anonymous pipe required")
    _require(os.readlink("/proc/self/fd/" + str(fd)).startswith("pipe:["),
             "Named or substituted custody pipe forbidden")
    os.set_blocking(fd, False)
    info = os.fstat(fd)
    return info.st_dev, info.st_ino


def _receive(fd, deadline):
    parts = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        _require(remaining > 0 and bool(select.select([fd], [], [], max(0, remaining))[0]),
                 "Custody pipe deadline exhausted")
        # Single-byte framing avoids reading bytes belonging to the next call;
        # the protocol is small and strictly request/response serialized.
        try:
            chunk = os.read(fd, 1)
        except BlockingIOError:
            continue
        _require(bool(chunk), "Custody pipe closed; preserve uncertain work")
        if chunk == b"\n":
            return json.loads(parts)
        parts.extend(chunk)
        _require(len(parts) <= MAX_MESSAGE_BYTES, "Custody pipe message ceiling")


def _send(fd, value, deadline):
    raw = _json(value) + b"\n"
    _require(len(raw) <= MAX_MESSAGE_BYTES, "Custody pipe message ceiling")
    position = 0
    while position < len(raw):
        remaining = deadline - time.monotonic()
        _require(remaining > 0 and bool(select.select([], [fd], [], max(0, remaining))[1]),
                 "Custody pipe deadline exhausted")
        try:
            position += os.write(fd, raw[position:position + 4096])
        except BlockingIOError:
            continue


class CustodyClient:
    """Only inherited-pipe I/O; safe to use after the actual parent guard."""
    strict_checkpoint_ack = True

    def __init__(self, request_fd, response_fd):
        self.identities = (_pipe(request_fd), _pipe(response_fd))
        _require(self.identities[0] != self.identities[1], "Distinct custody pipes required")
        self.request_fd, self.response_fd = request_fd, response_fd
        self._lock, self._sequence, self._poisoned = threading.Lock(), 0, False

    def _call(self, method, *args):
        with self._lock:
            _require(not self._poisoned, "Prior custody transport failure; preserve uncertain work")
            _require((_pipe(self.request_fd), _pipe(self.response_fd)) == self.identities,
                     "Custody pipe descriptor replaced")
            self._sequence += 1
            try:
                deadline = time.monotonic() + IPC_TIMEOUT_SECONDS
                _send(self.request_fd, {"sequence": self._sequence, "method": method, "args": args}, deadline)
                response = _receive(self.response_fd, deadline)
                _require(response.get("sequence") == self._sequence, "Custody response sequence changed")
                _require(response.get("ok") is True, "Custody authority refused: " + str(response.get("error")))
                return response["result"]
            except BaseException:
                self._poisoned = True
                raise

    def check(self, head):
        return self._call("check", head)

    def before(self, previous_head, row):
        return self._call("before", previous_head, row)

    def after(self, new_head, row):
        return self._call("after", new_head, row)

    def retain(self, pin):
        return self._call("retain", pin)

    __call__ = retain

    def current(self):
        return self._call("current")

    def controller_configuration(self):
        return self._call("controller_configuration")

    def confirm_checkpoint(self, result, pin, head):
        return self._call("confirm_checkpoint", result, pin, head)


def serve_custody(authority, request_fd, response_fd):
    """Serve one preopened controller channel; never spawn or open a socket."""
    _root()
    _pipe(request_fd)
    _pipe(response_fd)
    sequence = 0
    methods = {name: getattr(authority, name) for name in
               ("check", "before", "after", "retain", "current", "confirm_checkpoint",
                "controller_configuration")}
    while True:
        # Controller validation may take several minutes between messages.
        # A bounded frame timeout starts only once an actual request arrives.
        select.select([request_fd], [], [])
        request = _receive(request_fd, time.monotonic() + IPC_TIMEOUT_SECONDS)
        sequence += 1
        try:
            _require(isinstance(request, dict) and set(request) == {"sequence", "method", "args"}
                     and request["sequence"] == sequence and request["method"] in methods
                     and isinstance(request["args"], list), "Exact serialized custody request required")
            result = methods[request["method"]](*request["args"])
            _send(response_fd, {"sequence": sequence, "ok": True, "result": result},
                  time.monotonic() + IPC_TIMEOUT_SECONDS)
        except BaseException as exc:
            _send(response_fd, {"sequence": sequence, "ok": False,
                               "error": type(exc).__name__ + ": " + str(exc)},
                  time.monotonic() + IPC_TIMEOUT_SECONDS)
            raise
