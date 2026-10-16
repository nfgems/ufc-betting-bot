"""File-only worker IPC; privileged storage/sealing remains in the parent.

No sockets, subprocesses, flock or chmod are performed by MailboxClient.
ParentMailbox is an explicit synthetic-only supervisor adapter. It never
accepts worker-supplied source targets or transport instructions.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
import time


def _require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def _body(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _pin(path):
    body = path.read_bytes()
    return {"path": path.as_posix(), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(path, body, *, seal=False):
    # Worker uses the named path API so the audit boundary can scope the open.
    # Only the trusted parent uses descriptor-based immutable creation.
    stream = os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o400), "wb") if seal else path.open("xb")
    with stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_dir(path.parent)


class MailboxClient:
    def __init__(self, root, plan_sha256, *, timeout_seconds=120):
        self.root = Path(root).absolute()
        _require(self.root == self.root.resolve(), "Linked mailbox")
        self.plan_sha256 = plan_sha256
        self.timeout = timeout_seconds
        self.sequence = 0

    def _call(self, operation, payload):
        self.sequence += 1
        prefix = f"{self.sequence:06d}"
        message = {"sequence": self.sequence, "plan_sha256": self.plan_sha256,
                   "operation": operation, "payload": payload}
        temporary = self.root / (prefix + "-request.tmp")
        _write(temporary, _body(message))
        os.rename(temporary, self.root / (prefix + "-request.json"))
        _sync_dir(self.root)
        deadline = time.monotonic() + self.timeout
        response = self.root / (prefix + "-response.json")
        while not response.exists():
            _require(time.monotonic() < deadline, "Parent mailbox deadline exceeded")
            time.sleep(.025)
        result = json.loads(response.read_bytes())
        _require(result.get("sequence") == self.sequence
                 and result.get("plan_sha256") == self.plan_sha256, "Unbound parent response")
        _require(result.get("status") == "passed", "Parent operation failed: " + str(result.get("error")))
        return result["result"]

    def collect(self):
        return self._call("collect", {})

    def immutable(self, path, body):
        relative = Path(path).absolute().relative_to(self.root.parent).as_posix()
        result = self._call("immutable", {"relative": relative,
                            "body_base64": base64.b64encode(body).decode()})
        _require(_pin(Path(result["path"])) == result, "Sealed parent artifact drift")
        return result

    def seal_existing(self, path):
        path = Path(path).absolute()
        return self._call("seal_existing", {"relative": path.relative_to(self.root.parent).as_posix(),
                                           "pin": _pin(path)})


class ParentMailbox:
    """Explicit in-process adapter for supervise(args, broker=...).

    Collection runs on a parent thread so the existing deadline/output watchdog
    continues. Cancellation prevents later sends. A callback that fails to stop
    is reported as a failure; this preparation adapter is not a hostile-code or
    live transport sandbox. All provided callbacks in its tests are synthetic.
    """

    execution_mode = "synthetic"

    def __init__(self, broker, *, artifact_roots=("evidence",)):
        _require(broker.plan.get("execution_mode") == "synthetic", "Synthetic broker required")
        self.broker = broker
        self.plan_pin = dict(broker.plan_pin)
        self.artifact_roots = tuple(artifact_roots)
        _require(self.artifact_roots and all(isinstance(r, str) and r.isidentifier() for r in self.artifact_roots),
                 "Simple artifact roots required")
        self.work = self.root = None
        self.sequence = 0
        self.thread = None
        self.error = None
        self.response = None
        self.collected = False
        # Kept only by the trusted parent. Read-only mode alone cannot revoke a
        # worker's already-open descriptor or prevent directory replacement.
        self.custody = {}

    def bind(self, work, plan):
        _require(self.work is None, "Mailbox cannot be rebound")
        _require(plan.get("broker_plan") == self.plan_pin
                 and self.plan_pin in plan["input_pins"]
                 and plan.get("broker_artifact_roots") == list(self.artifact_roots), "Supervisor does not bind broker authority")
        self.work = Path(work).resolve()
        self.root = self.work / "broker-mailbox"
        self.root.mkdir(mode=0o700)

    def _target(self, relative):
        _require(isinstance(relative, str), "Artifact path required")
        path = self.work / relative
        _require(path == path.resolve() and self.work in path.parents
                 and path.relative_to(self.work).parts[0] in self.artifact_roots,
                 "Artifact outside allowed work roots")
        for item in (path, *path.parents):
            _require(not item.is_symlink(), "Artifact symlink")
        _require(path.parent.is_dir(), "Artifact parent must exist")
        return path

    def _execute(self, message):
        try:
            operation, payload = message["operation"], message["payload"]
            if operation == "collect":
                _require(payload == {} and not self.collected, "Collection cannot replay or accept worker request targets")
                self.collected = True
                result = self.broker.collect()
                if result["membership"] is not None:
                    raw = result["membership"]["raw"]
                    result["membership"]["body_base64"] = base64.b64encode(Path(raw["path"]).read_bytes()).decode()
            elif operation == "immutable":
                _require(set(payload) == {"relative", "body_base64"}, "Invalid artifact message")
                path = self._target(payload["relative"])
                body = base64.b64decode(payload["body_base64"], validate=True)
                _require(len(body) <= 33554432, "Artifact size bound")
                _require(str(path) not in self.custody, "Artifact already acknowledged")
                _write(path, body, seal=True)
                result = {"path": str(path), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
                self.custody[str(path)] = result
            elif operation == "seal_existing":
                _require(set(payload) == {"relative", "pin"}, "Invalid sealing message")
                path = self._target(payload["relative"])
                _require(str(path) not in self.custody, "Artifact already acknowledged")
                info = path.lstat()
                _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                         and _pin(path) == payload["pin"], "Artifact drift or alias")
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    os.fchmod(fd, 0o400)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                result = dict(payload["pin"])
                self.custody[str(path)] = result
            else:
                raise RuntimeError("Unsupported parent operation")
            self.response = {"status": "passed", "result": result}
        except BaseException as exc:
            self.error = type(exc).__name__ + ": " + str(exc)
            self.response = {"status": "failed", "error": self.error}

    def poll(self):
        if self.thread is not None:
            if self.thread.is_alive():
                return
            self.thread.join()
            self.thread = None
            value = {"sequence": self.sequence, "plan_sha256": self.plan_pin["sha256"], **self.response}
            temporary = self.root / f"{self.sequence:06d}-response.tmp"
            response_body = _body(value)
            _write(temporary, response_body, seal=True)
            response_path = self.root / f"{self.sequence:06d}-response.json"
            os.rename(temporary, response_path)
            _sync_dir(self.root)
            self.custody[str(response_path)] = {"path": str(response_path), "bytes": len(response_body),
                                               "sha256": hashlib.sha256(response_body).hexdigest()}
            if self.error:
                raise RuntimeError(self.error)
        _require(self.sequence < 1024, "Mailbox operation limit")
        request = self.root / f"{self.sequence + 1:06d}-request.json"
        if not request.exists():
            return
        info = request.lstat()
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size <= 50331648,
                 "Mailbox request type/size")
        raw = request.read_bytes()
        message = json.loads(raw)
        _require(set(message) == {"sequence", "plan_sha256", "operation", "payload"}
                 and message["sequence"] == self.sequence + 1
                 and message["plan_sha256"] == self.plan_pin["sha256"], "Unbound mailbox request")
        self.sequence += 1
        self.custody[str(request)] = {"path": str(request), "bytes": len(raw),
                                     "sha256": hashlib.sha256(raw).hexdigest()}
        self.thread = threading.Thread(target=self._execute, args=(message,), daemon=True)
        self.thread.start()

    def close(self):
        self.broker.cancel()
        if self.thread is not None:
            self.thread.join(timeout=25)
            _require(not self.thread.is_alive(), "Parent callback did not stop; preserve pending accounting")
        # Called by the supervisor only after the child has exited and been
        # reaped. No live worker descriptor remains at this final comparison.
        for name, expected in self.custody.items():
            path = Path(name)
            _require(path == path.resolve() and self.work in path.parents, "Acknowledged artifact moved")
            info = path.lstat()
            _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                     and _pin(path) == expected, "Acknowledged evidence changed: " + name)
        self.broker.store.snapshot()  # Revalidate durable bodies/envelopes/head.
        _require(self.error is None, "Parent broker failed: " + str(self.error))
        return {"execution_mode": "synthetic", "source_requests": 0,
                "synthetic_charges": len(self.broker.records), "operations": self.sequence,
                "collected": self.collected, "acknowledged_artifacts": list(self.custody.values())}
