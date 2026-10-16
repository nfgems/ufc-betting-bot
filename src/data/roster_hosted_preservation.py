"""Explicit native preservation connection; custody still owns acknowledgments.

Provision a separate empty native destination before launching the controller.
Capsules are the existing original-byte format, not a backup/restore system.
Neither an anchor nor a capsule grants recovery, replay or source authority.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import stat
import uuid

from src.data.roster_hosted_store import (
    _fsync_dir, _json, _new_file, _read, _require, _safe, _sha, _time,
    file_pin, verify_pin,
)


def persistent_checkpointer(directory):
    """Preserve exact original plans before claims; copies grant no replay."""
    parent = _safe(directory)
    _require(parent.is_dir(), "Existing persistent checkpoint directory required")

    def persist(pin):
        saved, pending = {}, [pin]
        while pending:
            item = pending.pop()
            if item["path"] in saved:
                continue
            raw = _read(verify_pin(item)["path"])
            saved[item["path"]] = {"pin": item, "body_base64": base64.b64encode(raw).decode("ascii")}
            value = json.loads(raw)
            if isinstance(value, dict):
                for key in ("binding", "plan", "review", "root_plan", "root_review"):
                    nested = value.get(key)
                    if isinstance(nested, dict) and set(nested) == {"path", "bytes", "sha256"}:
                        pending.append(nested)
        _new_file(parent / ("preservation-" + uuid.uuid4().hex + ".json"), _json({
            "schema": 1, "checkpoint": pin, "originals": list(saved.values()),
            "replay_authorized": False, "source_requests_authorized": 0,
            "complete_volume_backup": False}), immutable=True)
    return persist


def _native(path, *, directory):
    from src.data.roster_hosted_custody import _trusted
    _require(Path(path).is_absolute(), "Explicit absolute native preservation path required")
    path = _safe(path)
    # Only the explicit protected reservation can use the managed log volume.
    # This grants placement admission, never recovery or accounting authority.
    from src.data.roster_railway_paths import RESERVED, VOLUME, persistent_path
    reserved = path == RESERVED or RESERVED in path.parents
    if reserved:
        persistent_path(path)
    # All other managed production and volatile/Windows paths remain excluded.
    for excluded in ("/tmp", "/var/tmp", "/mnt", "/app/logs", "/app/data"):
        root = Path(excluded)
        _require((reserved and root == VOLUME) or (path != root and root not in path.parents),
                 "Unsuitable preservation destination")
    for parent in path.parents:
        info = parent.lstat()
        _require(stat.S_ISDIR(info.st_mode) and info.st_uid == 0 and
                 (not info.st_mode & 0o022 or
                  (reserved and parent == VOLUME and bool(info.st_mode & stat.S_ISVTX))),
                 "Preservation requires protected root-owned ancestry")
    identity = _trusted(path, directory=directory)
    # Explicit ownership on a Windows mount is not native role separation.
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        mount = Path(left.split()[4].replace("\\040", " ").replace("\\134", "\\"))
        if path == mount or mount in path.parents:
            mounts.append((len(mount.parts), right.split()[0]))
    _require(bool(mounts) and max(mounts)[1] not in {"9p", "drvfs", "tmpfs", "ramfs"},
             "Native nonvolatile preservation filesystem required")
    return identity


def _public_original(path, value):
    from src.data.roster_hosted_custody import _sync_file
    _new_file(path, _json(value), immutable=True)
    path.chmod(0o444)
    _sync_file(path)
    _fsync_dir(path.parent)
    return file_pin(path)


def provision_server_preservation(destination, *, controls_pin,
                                  expected_controls_sha256, controller_binding_path):
    """Bind an already provisioned empty destination to one original cycle.

    This function creates only its named controls/capsules inside that destination.
    It never creates/fixes ownership, supplies a default path or restores evidence.
    """
    from src.data.roster_hosted_custody import _root, _trusted, _worker_absent
    _root()
    _trusted(controls_pin["path"], directory=False)
    verify_pin(controls_pin)
    _require(controls_pin["sha256"] == expected_controls_sha256,
             "External custody controls hash mismatch")
    controls = json.loads(_read(controls_pin["path"]))
    _require(controls.get("kind") == "native_root_roster_custody_v1", "Native custody controls required")
    _worker_absent(controls["worker_uid"])
    from src.data.roster_hosted_custody import NativeCustodyAuthority
    authority = NativeCustodyAuthority(controls_pin, expected_controls_sha256=expected_controls_sha256)
    with authority._locked():
        state, _, _ = authority._load()
        _require(state["retained"] is None and state["retention_head"]["sequence"] == 0
                 and state["pending"] is None,
                 "Fresh preservation provisioning cannot adopt earlier acknowledged or uncertain work")
    root_identity = _native(destination, directory=True)
    root = Path(root_identity["path"])
    _require(not list(root.iterdir()), "Unused separately provisioned preservation destination required")
    retention = json.loads(_read(verify_pin(controls["retention_binding"])["path"]))
    others = [Path(controls["root"]["path"]), Path(retention["mutable_root"]["path"]),
              Path(retention["directories"]["root"]["path"]),
              *[Path(item["path"]) for item in controls["accepted_roots"]]]
    _require(all(root != other and root not in other.parents and other not in root.parents
                 for other in others), "Preservation must be disjoint from all existing role trees")
    _require(Path(controller_binding_path).is_absolute(), "Explicit original controller binding path required")
    binding = _safe(controller_binding_path)
    _require(binding.name == "binding.json" and not binding.exists()
             and Path(retention["mutable_root"]["path"]) in binding.parents,
             "Unused original controller binding in provisioned mutable namespace required")
    capsules = root / "capsules"
    capsules.mkdir(mode=0o700)
    value = {"schema": 1, "kind": "native_roster_server_preservation_v1",
             "destination": root_identity, "capsules": _native(capsules, directory=True),
             "controls": controls_pin, "expected_controls_sha256": expected_controls_sha256,
             "controller_binding_path": binding.as_posix(),
             "activation_authorized": False, "publication_authorized": False,
             "schedule_enabled": False, "replay_authorized": False}
    result = _public_original(root / "preservation-controls.json", value)
    _fsync_dir(capsules)
    _fsync_dir(root)
    return result


def provision_server_successor(destination, *, previous_configuration_pin,
                               expected_previous_config_sha256, controller_binding_path,
                               recovery_evidence_pin=None, clock=_time):
    """Bind a fresh cycle to the same independently acknowledged custody history.

    This creates preservation metadata only. The ordinary issuer still consumes
    the current slot and enforces the original per-cycle authority and lifetime.
    Nothing is copied, reset, acknowledged retroactively, or made executable.
    """
    from src.data.roster_hosted_custody import _root, _worker_absent
    from src.data.roster_hosted_recurrence import validate_successor_cycle
    from src.data.roster_hosted_store import HostedControlStore

    _root()
    previous = NativeServerPreservation(previous_configuration_pin,
        expected_config_sha256=expected_previous_config_sha256)
    ancestors = previous._lineage()
    previous._acknowledged_capsules()
    _require(not (previous.root / "successor-intent.json").exists(),
             "Original successor provision already attempted; preserve its exact result")
    controls = json.loads(_read(previous.config["controls"]["path"]))
    _worker_absent(controls["worker_uid"])
    _require(previous.helper_configuration() is not None,
             "Successor requires the exact original independently owned helper association")
    authority = open_server_custody(previous_configuration_pin,
        expected_config_sha256=expected_previous_config_sha256, clock=clock)
    current = authority.current()
    retention = json.loads(_read(verify_pin(controls["retention_binding"])["path"]))
    store = HostedControlStore(retention["store_root"])
    state = store.snapshot()
    _require(not state["uncertain_requests"]
             and current["current_store_head"] == state["head"]
             and current["retained_store_head"] == state["head"]
             and current["unretained_store_tail"] is False,
             "Successor requires the complete independently retained accounting head")
    anchor = previous.startup_anchor()
    now = _time(clock())
    decision = validate_successor_cycle(store, anchor["binding"],
        controls["controller_controls"]["policy"], now=now,
        recovery_evidence_pin=recovery_evidence_pin)
    slot = decision["slot_id"]
    _require(slot not in state["plan_slots"], "Current scheduled cycle is already consumed")
    history_parent = Path(previous.config["controller_binding_path"]).parent.parent
    binding = _safe(controller_binding_path)
    _require(Path(controller_binding_path).is_absolute() and binding.name == "binding.json"
             and binding.parent.parent == history_parent and not binding.parent.exists(),
             "Fresh sibling original controller binding required")
    root_identity = _native(destination, directory=True)
    root = Path(root_identity["path"])
    others = [Path(controls["root"]["path"]), Path(retention["mutable_root"]["path"]),
              Path(retention["directories"]["root"]["path"]),
              *[Path(item["path"]) for item in controls["accepted_roots"]]]
    for ancestor in ancestors:
        others.append(ancestor.root)
        helper = ancestor.root / "launcher-helper-binding.json"
        if helper.exists():
            others.append(Path(json.loads(_read(helper))["journal"]["path"]))
    _require(not list(root.iterdir()) and all(root != other and root not in other.parents
             and other not in root.parents for other in others),
             "Unused disjoint native successor preservation destination required")
    proof = decision.get("recovery_evidence")
    if proof is not None:
        # The guarded worker later verifies this public original by exact pin.
        # Admission never changes the caller's provisioned role permissions.
        _require(all(path.stat().st_mode & 0o001 for path in (root, *root.parents)),
                 "Successor calendar evidence requires an already worker-traversable protected destination")
    # The checks above authorize metadata creation only while the exact custody
    # head and previous independent acknowledgments remain unchanged.
    _require(authority.current() == current and store.snapshot() == state,
             "Custody changed during successor admission")
    intent = {"schema": 1, "kind": "native_roster_successor_intent_v1",
        "previous_configuration": previous_configuration_pin,
        "destination": root_identity, "controller_binding_path": binding.as_posix(),
        "admitted_slot": slot, "retained_store_head": state["head"],
        "replay_authorized": False, "activation_authorized": False}
    # Exclusive creation is the irreversible admission fence. A crash leaves
    # this exact attempt consumed, including before a new config is complete.
    intent_pin = _public_original(previous.root / "successor-intent.json", intent)
    recovery_evidence_pin = (_public_original(root / "calendar-reconciliation.json", proof)
                             if proof is not None else None)
    capsules = root / "capsules"
    capsules.mkdir(mode=0o700)
    value = {"schema": 2, "kind": "native_roster_server_preservation_v2",
        "destination": root_identity, "capsules": _native(capsules, directory=True),
        "controls": previous.config["controls"],
        "expected_controls_sha256": previous.config["expected_controls_sha256"],
        "controller_binding_path": binding.as_posix(),
        "previous_configuration": previous_configuration_pin, "successor_intent": intent_pin,
        "admitted_slot": slot, "admitted_at": now.isoformat(),
        "recovery_evidence": recovery_evidence_pin,
        "activation_authorized": False, "publication_authorized": False,
        "schedule_enabled": False, "replay_authorized": False}
    result = _public_original(root / "preservation-controls.json", value)
    _fsync_dir(capsules)
    _fsync_dir(root)
    return result


class NativeServerPreservation:
    """Mandatory callback for the explicit server custody path only."""

    def __init__(self, config_pin, *, expected_config_sha256):
        from src.data.roster_hosted_custody import _root
        _root()
        _native(config_pin["path"], directory=False)
        verify_pin(config_pin)
        _require(config_pin["sha256"] == expected_config_sha256,
                 "External preservation configuration hash mismatch")
        self.pin = json.loads(_json(config_pin))
        self.config = json.loads(_read(config_pin["path"]))
        fields = {"schema", "kind", "destination", "capsules", "controls",
                 "expected_controls_sha256", "controller_binding_path", "activation_authorized",
                 "publication_authorized", "schedule_enabled", "replay_authorized"}
        schema = self.config.get("schema")
        if schema == 2:
            fields |= {"previous_configuration", "successor_intent", "admitted_slot", "admitted_at", "recovery_evidence"}
        _require(set(self.config) == fields and type(schema) is int and schema in {1, 2}
                 and self.config["kind"] == "native_roster_server_preservation_v" + str(schema)
                 and all(self.config[key] is False for key in ("activation_authorized",
                    "publication_authorized", "schedule_enabled", "replay_authorized")),
                 "Exact disabled native preservation configuration required")
        self.root = Path(self.config["destination"]["path"])
        self.capsules = self.root / "capsules"
        self.poisoned = False
        self._boundary()
        self.persist = persistent_checkpointer(self.capsules)

    def _lineage(self):
        """Validate original disjoint roots without copying old evidence."""
        chain, current, seen, entries, total = [], self, set(), 0, 0
        from src.data.roster_hosted_checkpoints import MAX_ENTRIES, MAX_TOTAL_BYTES
        while True:
            _require(current.pin["path"] not in seen and len(chain) < 64,
                     "Repeated or excessive preservation lineage")
            seen.add(current.pin["path"])
            _require(current.config["controls"] == self.config["controls"]
                     and current.config["expected_controls_sha256"] == self.config["expected_controls_sha256"],
                     "Successor cannot replace original custody controls")
            _require(all(current.root != item.root and current.root not in item.root.parents
                         and item.root not in current.root.parents for item in chain),
                     "Preservation lineage roots must be disjoint")
            members = list(current.root.rglob("*"))
            entries += len(members) + 1
            total += sum(path.stat().st_size for path in members if path.is_file())
            _require(entries <= MAX_ENTRIES and total <= MAX_TOTAL_BYTES,
                     "Combined preservation lineage capacity exhausted")
            chain.append(current)
            if current.config["schema"] == 1:
                break
            prior = current.config["previous_configuration"]
            current = NativeServerPreservation(prior, expected_config_sha256=prior["sha256"])
        return chain

    def helper_configuration(self):
        """Reuse the original association and its complete append-only journal."""
        from src.data.roster_hosted_helper_binding import NAME, validate_helper_binding
        associated = [item for item in self._lineage() if (item.root / NAME).exists()]
        _require(len(associated) <= 1, "Conflicting original helper associations")
        if not associated:
            return None
        original = associated[0]
        validate_helper_binding(original.pin,
            json.loads(_read(verify_pin(self.config["controls"])["path"])))
        return original.pin

    def preparation_output(self, *, recovery_evidence_pin=None, clock=_time):
        """Use a successor only in its admitted slot, under the original policy."""
        if self.config["schema"] == 2:
            from src.data.roster_hosted_plan import _policy, _slot
            controls = json.loads(_read(self.config["controls"]["path"]))
            policy_pin = controls["controller_controls"]["policy"]
            recovery_evidence_pin = recovery_evidence_pin or self.config["recovery_evidence"]
            _require(recovery_evidence_pin == self.config["recovery_evidence"]
                     and _slot(_policy(policy_pin, policy_pin["sha256"]), _time(clock()),
                               recovery_evidence_pin) == self.config["admitted_slot"],
                     "Successor preparation escaped its original admitted slot")
        output = Path(self.config["controller_binding_path"]).parent
        _require(not output.exists(), "Unused original preparation output required")
        return output

    def _acknowledged_capsules(self):
        # On fresh role opening, a capsule left by a failed callback (including
        # failure after marker removal) must not be mistaken for an ack. Reuse
        # the existing independently validated custody journal as the authority.
        # Do not ask current() to validate time here: ordinary inspection and
        # recovery perform that validation using their unchanged caller clock.
        from src.data.roster_hosted_custody import NativeCustodyAuthority
        authority = NativeCustodyAuthority(self.config["controls"],
            expected_controls_sha256=self.config["expected_controls_sha256"])
        with authority._locked():
            _, head, _ = authority._load()
            acknowledged = []
            for number in range(1, head["sequence"] + 1):
                record = json.loads(_read(authority.root / "records" / (str(number).zfill(8) + ".json")))
                if record["event"]["kind"] == "retained":
                    acknowledged.append(_json(record["event"]["result"]["requested"]))
            preserved = []
            capsule_paths = [path for item in self._lineage() for path in item.capsules.iterdir()]
            for path in capsule_paths:
                _require(path.is_file() and path.name.startswith("preservation-")
                         and path.suffix == ".json", "Unexpected server preservation object")
                capsule = json.loads(_read(path))
                _require(set(capsule) == {"schema", "checkpoint", "originals", "replay_authorized",
                         "source_requests_authorized", "complete_volume_backup"}
                         and capsule["schema"] == 1 and capsule["replay_authorized"] is False
                         and capsule["source_requests_authorized"] == 0
                         and capsule["complete_volume_backup"] is False,
                         "Exact original preservation capsule required")
                bodies = {}
                for row in capsule["originals"]:
                    item = row["pin"]
                    raw = base64.b64decode(row["body_base64"], validate=True)
                    _require(item["path"] not in bodies and len(raw) == item["bytes"]
                             and _sha(raw) == item["sha256"], "Preserved original bytes changed")
                    bodies[item["path"]] = (item, raw)
                pending, visited = [capsule["checkpoint"]], set()
                while pending:
                    item = pending.pop()
                    _require(item["path"] in bodies and bodies[item["path"]][0] == item,
                             "Missing exact preserved original relationship")
                    if item["path"] in visited:
                        continue
                    visited.add(item["path"])
                    value = json.loads(bodies[item["path"]][1])
                    if isinstance(value, dict):
                        for key in ("binding", "plan", "review", "root_plan", "root_review"):
                            nested = value.get(key)
                            if isinstance(nested, dict) and set(nested) == {"path", "bytes", "sha256"}:
                                pending.append(nested)
                _require(visited == set(bodies), "Unexpected preserved original relationship")
                preserved.append(_json(capsule["checkpoint"]))
            _require(sorted(preserved) == sorted(acknowledged),
                     "Preservation lacks exact independent custody acknowledgment; preserve and review")

    def _boundary(self):
        from src.data.roster_hosted_checkpoints import (
            MAX_ENTRIES, MAX_FILE_BYTES, MAX_TOTAL_BYTES, FREE_RESERVE_BYTES,
        )
        from src.data.roster_hosted_custody import _root, _trusted
        _root()
        _require(not self.poisoned, "Prior server preservation failed; preserve and review")
        _native(self.pin["path"], directory=False)
        verify_pin(self.pin)
        _require(json.loads(_read(self.pin["path"])) == self.config
                 and Path(self.pin["path"]) == self.root / "preservation-controls.json",
                 "Preservation configuration changed")
        for key, path in (("destination", self.root), ("capsules", self.capsules)):
            _require(_native(path, directory=True) == self.config[key],
                     "Provisioned native preservation destination changed")
        _trusted(self.config["controls"]["path"], directory=False)
        verify_pin(self.config["controls"])
        _require(self.config["controls"]["sha256"] == self.config["expected_controls_sha256"],
                 "Original custody controls changed")
        _require({p.name for p in self.root.iterdir()} <= {
            "capsules", "preservation-controls.json", "startup-anchor.json", "launcher-helper-binding.json",
            "successor-intent.json", "calendar-reconciliation.json"},
            "Unfinished server preservation; preserve and review")
        if self.config["schema"] == 2:
            intent_pin = self.config["successor_intent"]
            prior = self.config["previous_configuration"]
            _require(Path(intent_pin["path"]) == Path(prior["path"]).parent / "successor-intent.json",
                     "Successor intent must remain at the original predecessor")
            _native(intent_pin["path"], directory=False)
            intent = json.loads(_read(verify_pin(intent_pin)["path"]))
            _require(intent["previous_configuration"] == prior
                     and intent["destination"] == self.config["destination"]
                     and intent["controller_binding_path"] == self.config["controller_binding_path"]
                     and intent["admitted_slot"] == self.config["admitted_slot"],
                     "Successor configuration escaped its original provision intent")
            _require(not (self.root / "launcher-helper-binding.json").exists(),
                     "Successor cannot replace the original helper association")
        proof = self.config.get("recovery_evidence")
        proof_path = self.root / "calendar-reconciliation.json"
        _require(proof_path.exists() == (proof is not None), "Unexpected calendar reconciliation")
        if proof is not None:
            _require(Path(proof["path"]) == proof_path, "Original successor calendar evidence required")
            verify_pin(proof)
        intent_path = self.root / "successor-intent.json"
        if intent_path.exists():
            intent = json.loads(_read(intent_path))
            _require(set(intent) == {"schema", "kind", "previous_configuration", "destination",
                     "controller_binding_path", "admitted_slot", "retained_store_head",
                     "replay_authorized", "activation_authorized"}
                     and intent["schema"] == 1 and intent["kind"] == "native_roster_successor_intent_v1"
                     and intent["previous_configuration"] == self.pin
                     and intent["activation_authorized"] is False and intent["replay_authorized"] is False,
                     "Original successor intent changed")
        if (self.root / "launcher-helper-binding.json").exists():
            from src.data.roster_hosted_helper_binding import validate_helper_binding
            controls = json.loads(_read(self.config["controls"]["path"]))
            validate_helper_binding(self.pin, controls)
        entries = list(self.root.rglob("*"))
        total = 0
        for path in entries:
            _native(path, directory=path.is_dir())
            if path.is_file():
                size = path.stat().st_size
                _require(size <= MAX_FILE_BYTES, "Preservation file ceiling")
                total += size
        _require(len(entries) <= MAX_ENTRIES and total <= MAX_TOTAL_BYTES,
                 "Preservation capacity exhausted")
        capacity = os.statvfs(self.root)
        _require(capacity.f_bavail * capacity.f_frsize >= FREE_RESERVE_BYTES,
                 "Preservation free space reserve exhausted")

    def __call__(self, original):
        self._boundary()
        _require(not (self.root / "successor-intent.json").exists(),
                 "Earlier cycle is closed by its original successor intent")
        verify_pin(original)
        marker = self.root / "preservation-pending.json"
        try:
            # An interrupted/failed preservation cannot be retried implicitly
            # after reopening. This marker grants no acknowledgment or replay.
            _new_file(marker, _json({"original": original, "replay_authorized": False}), immutable=True)
            self.persist(original)
            if original["path"] == self.config["controller_binding_path"]:
                binding = json.loads(_read(original["path"]))
                _require(binding.get("mode") == "disabled_controller_v1"
                         and binding.get("activation_authorized") is False
                         and binding.get("publication_authorized") is False,
                         "Exact disabled original controller binding required")
                anchor = {"schema": 1, "kind": "native_roster_server_startup_anchor_v1",
                          "binding": original, "expected_binding_sha256": original["sha256"],
                          "controls": self.config["controls"], "preservation_configuration": self.pin,
                          "replay_authorized": False}
                _public_original(self.root / "startup-anchor.json", anchor)
            else:
                self.startup_anchor()
            marker.unlink()
            _fsync_dir(self.root)
            self._boundary()
        except BaseException:
            self.poisoned = True
            raise

    def startup_anchor(self):
        """Read the original anchor; existing inspection/recovery still decides."""
        path = self.root / "startup-anchor.json"
        _native(path, directory=False)
        value = json.loads(_read(path))
        _require(set(value) == {"schema", "kind", "binding", "expected_binding_sha256", "controls",
                 "preservation_configuration", "replay_authorized"}
                 and value["schema"] == 1 and value["kind"] == "native_roster_server_startup_anchor_v1"
                 and value["controls"] == self.config["controls"]
                 and value["preservation_configuration"] == self.pin
                 and value["replay_authorized"] is False
                 and value["binding"]["path"] == self.config["controller_binding_path"]
                 and value["expected_binding_sha256"] == value["binding"]["sha256"],
                 "Original server startup anchor changed")
        verify_pin(value["binding"])
        return value


def open_server_custody(config_pin, *, expected_config_sha256, clock=_time):
    from src.data.roster_hosted_custody import NativeCustodyAuthority
    preservation = NativeServerPreservation(config_pin, expected_config_sha256=expected_config_sha256)
    preservation._acknowledged_capsules()
    return NativeCustodyAuthority(preservation.config["controls"],
        expected_controls_sha256=preservation.config["expected_controls_sha256"],
        clock=clock, archive_checkpoint=preservation)


def read_server_startup_anchor(config_pin, *, expected_config_sha256):
    # Read-only provenance, not custody admission. Only open_server_custody may
    # admit the new server path after checking actual independent acknowledgments.
    # Avoid competing with that role's startup lock merely to read an anchor.
    preservation = NativeServerPreservation(config_pin, expected_config_sha256=expected_config_sha256)
    return preservation.startup_anchor()
