"""Bounded, append-only storage of exact legacy identity evidence bytes.

References are a persistence encoding only: matching always receives the exact
original UTF-8 string. Content-defined chunks share evidence even when earlier
captures differ. No capture, whitespace, failed response or historical decision
is parsed away. The trusted candidate supervisor seals output after execution;
this worker-compatible writer needs no chmod, lock or widened syscall policy.

A failed write is retained. Torn/corrupt objects fail closed on reopen. Complete
unreferenced objects are harmless retained evidence and may be reused. This is
not a replacement for the ordinary transaction or full-volume rollback backup.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat


STORE_DIRECTORY = "identity-evidence-v1"
REFERENCE_PREFIX = "roster-identity-evidence:v1:"
FIELDS = ("identity_decision_json", "identity_binding_json")
MAX_DOCUMENT_BYTES = 134_217_728
MAX_STORE_BYTES = 268_435_456
MAX_STORE_OBJECTS = 8192
MAX_MANIFEST_BYTES = 4_194_304
MAX_DOCUMENT_CHUNKS = 32768
MIN_CHUNK_BYTES = 4096
MAX_CHUNK_BYTES = 65536
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_GEAR = tuple(int.from_bytes(hashlib.sha256(bytes([i])).digest()[:8], "big") for i in range(256))
_FORMAT = b'{"chunker":"gear-sha256-v1-4k-16k-64k","schema_version":1}\n'


class IdentityEvidenceError(ValueError):
    """An evidence reference or retained object cannot be trusted."""


def _require(condition, message):
    if not condition:
        raise IdentityEvidenceError(message)


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _json(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def _regular(path):
    for parent in (path, *path.parents):
        if parent.exists() or parent.is_symlink():
            info = parent.lstat()
            _require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400,
                     "evidence paths cannot use symbolic links or reparse points")
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "evidence must be an unaliased regular file")
    return info


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _chunks(content):
    start = 0
    while start < len(content):
        end = min(start + MIN_CHUNK_BYTES, len(content))
        ceiling = min(start + MAX_CHUNK_BYTES, len(content))
        rolling = 0
        while end < ceiling:
            rolling = ((rolling << 1) + _GEAR[content[end]]) & ((1 << 64) - 1)
            end += 1
            if rolling & 16383 == 0:
                break
        yield content[start:end]
        start = end


class IdentityEvidenceStore:
    def __init__(self, root, *, create=False):
        self.root = Path(root).absolute()
        for parent in (self.root, *self.root.parents):
            _require(not parent.is_symlink(), "evidence root cannot use symbolic links")
            if parent.exists():
                _require(not getattr(parent.lstat(), "st_file_attributes", 0) & 0x400,
                         "evidence root cannot use reparse points")
        if not self.root.exists():
            _require(create, "identity evidence store is absent")
            self.root.mkdir()
            _sync_directory(self.root.parent)
            (self.root / "objects").mkdir()
            _sync_directory(self.root)
            self._write_new(self.root / "format.json", _FORMAT)
        _require(self.root.is_dir() and (self.root / "objects").is_dir(), "evidence store layout is invalid")
        _require(self._read(self.root / "format.json", MAX_MANIFEST_BYTES) == _FORMAT,
                 "unknown identity evidence store version")
        self.verify()

    @classmethod
    def for_state(cls, state_root, *, create=False):
        root = Path(state_root) / STORE_DIRECTORY
        if not root.exists() and not root.is_symlink() and not create:
            return None
        return cls(root, create=create)

    def _read(self, path, limit):
        info = _regular(path)
        _require(info.st_size <= limit, "evidence object exceeds byte limit")
        with path.open("rb") as stream:
            content = stream.read(limit + 1)
        _require(len(content) == info.st_size and len(content) <= limit, "evidence changed or exceeded byte limit")
        return content

    def _write_new(self, path, content):
        if path.exists() or path.is_symlink():
            _require(self._read(path, max(len(content), MAX_CHUNK_BYTES)) == content,
                     "immutable evidence collision or interrupted write")
            return
        # Retain incomplete files on any error: never silently repair evidence.
        with path.open("xb") as stream:
            written = stream.write(content)
            _require(written == len(content), "short evidence write")
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(path.parent)

    def verify(self):
        expected = {"format.json", "objects", "legacy-originals.json"}
        _require(all(path.name in expected for path in self.root.iterdir()), "unknown evidence store entry")
        _require(not (self.root / "objects").is_symlink(), "evidence objects directory is a symlink")
        objects = list((self.root / "objects").iterdir())
        _require(len(objects) <= MAX_STORE_OBJECTS, "evidence object count exhausted")
        total = 0
        for path in objects:
            digest, separator, suffix = path.name.partition(".")
            _require(_HEX.fullmatch(digest) and separator and suffix in {"bin", "json"}, "invalid evidence object name")
            limit = MAX_CHUNK_BYTES if suffix == "bin" else MAX_MANIFEST_BYTES
            content = self._read(path, limit)
            _require(_sha(content) == digest, "corrupt or interrupted immutable evidence object")
            total += len(content)
            _require(total <= MAX_STORE_BYTES, "evidence store byte budget exhausted")
        self.object_count, self.object_bytes = len(objects), total
        # Validate complete manifests including currently unreferenced history.
        for path in objects:
            if path.suffix == ".json":
                self.get_bytes(REFERENCE_PREFIX + path.stem)
        originals = self.root / "legacy-originals.json"
        if originals.exists() or originals.is_symlink():
            payload = json.loads(self._read(originals, MAX_MANIFEST_BYTES))
            _require(isinstance(payload, dict) and set(payload) == {"schema_version", "files"}
                     and type(payload["schema_version"]) is int and payload["schema_version"] == 1
                     and isinstance(payload["files"], dict), "legacy originals manifest is invalid")
            for name, pin in payload["files"].items():
                _require(isinstance(name, str) and Path(name).name == name and "/" not in name and "\\" not in name
                         and isinstance(pin, dict) and type(pin.get("present")) is bool,
                         "invalid legacy original pin")
                if not pin["present"]:
                    _require(set(pin) == {"present"}, "invalid absent legacy original pin")
                    continue
                _require(set(pin) == {"present", "bytes", "sha256", "reference"}, "invalid legacy original pin")
                content = self.get_bytes(pin["reference"])
                _require(type(pin["bytes"]) is int and len(content) == pin["bytes"] and _sha(content) == pin["sha256"],
                         "legacy original bytes disagree with pin")
        _require(total + self._control_bytes() <= MAX_STORE_BYTES, "evidence store byte budget exhausted")
        return {"objects": self.object_count, "object_bytes": self.object_bytes,
                "total_bytes": self.object_bytes + self._control_bytes()}

    def _control_bytes(self):
        return sum(path.stat().st_size for path in
                   (self.root / "format.json", self.root / "legacy-originals.json") if path.exists())

    def _put_object(self, content, suffix):
        digest = _sha(content)
        path = self.root / "objects" / f"{digest}.{suffix}"
        if not path.exists() and not path.is_symlink():
            _require(self.object_count + 1 <= MAX_STORE_OBJECTS, "evidence object count exhausted")
            _require(self.object_bytes + self._control_bytes() + len(content) <= MAX_STORE_BYTES,
                     "evidence store byte budget exhausted")
            self._write_new(path, content)
            self.object_count += 1
            self.object_bytes += len(content)
        else:
            self._write_new(path, content)
        return {"sha256": digest, "bytes": len(content)}

    def put_bytes(self, content):
        _require(isinstance(content, bytes) and len(content) <= MAX_DOCUMENT_BYTES, "evidence document exceeds byte limit")
        chunks = []
        for chunk in _chunks(content):
            _require(len(chunks) < MAX_DOCUMENT_CHUNKS, "evidence document chunk limit exhausted")
            chunks.append(self._put_object(chunk, "bin"))
        manifest = _json({"schema_version": 1, "bytes": len(content), "sha256": _sha(content), "chunks": chunks})
        _require(len(manifest) <= MAX_MANIFEST_BYTES, "evidence manifest exceeds byte limit")
        pin = self._put_object(manifest, "json")
        return REFERENCE_PREFIX + pin["sha256"]

    def get_bytes(self, reference):
        _require(isinstance(reference, str) and reference.startswith(REFERENCE_PREFIX), "unknown identity evidence reference")
        digest = reference[len(REFERENCE_PREFIX):]
        _require(_HEX.fullmatch(digest), "invalid identity evidence reference")
        raw = self._read(self.root / "objects" / f"{digest}.json", MAX_MANIFEST_BYTES)
        _require(_sha(raw) == digest, "evidence manifest hash mismatch")
        payload = json.loads(raw)
        _require(isinstance(payload, dict) and set(payload) == {"schema_version", "bytes", "sha256", "chunks"}
                 and type(payload["schema_version"]) is int and payload["schema_version"] == 1,
                 "unknown identity evidence manifest version")
        size, chunks = payload["bytes"], payload["chunks"]
        _require(type(size) is int and 0 <= size <= MAX_DOCUMENT_BYTES
                 and isinstance(payload["sha256"], str) and _HEX.fullmatch(payload["sha256"])
                 and isinstance(chunks, list) and len(chunks) <= MAX_DOCUMENT_CHUNKS,
                 "invalid evidence document bounds")
        result = bytearray()
        for pin in chunks:
            _require(isinstance(pin, dict) and set(pin) == {"sha256", "bytes"}
                     and isinstance(pin["sha256"], str) and _HEX.fullmatch(pin["sha256"])
                     and type(pin["bytes"]) is int and 0 < pin["bytes"] <= MAX_CHUNK_BYTES,
                     "invalid evidence chunk pin")
            _require(len(result) + pin["bytes"] <= size, "evidence expansion exceeds document bound")
            body = self._read(self.root / "objects" / f"{pin['sha256']}.bin", MAX_CHUNK_BYTES)
            _require(len(body) == pin["bytes"] and _sha(body) == pin["sha256"], "evidence chunk hash or size mismatch")
            result.extend(body)
        _require(len(result) == size and _sha(result) == payload["sha256"], "restored legacy evidence hash mismatch")
        return bytes(result)

    def archive_originals(self, files):
        """Save complete original CSV/alias bytes before first persistence change."""
        files = [Path(path) for path in files]
        _require(len({path.name for path in files}) == len(files), "legacy original filenames must be unique")
        destination = self.root / "legacy-originals.json"
        if destination.exists():
            self.verify()
            payload = json.loads(self._read(destination, MAX_MANIFEST_BYTES))
            _require(set(payload["files"]) == {path.name for path in files},
                     "legacy original archive does not cover the requested persistence files")
            return
        records = {}
        # A lost migration receipt must not relabel references as exact legacy
        # originals. Only an interrupted, still-inline migration may resume.
        for path in files:
            if path.exists():
                content = self._read(path, MAX_DOCUMENT_BYTES)
                _require(b"roster-identity-evidence:" not in content,
                         "legacy originals manifest is missing for already referenced evidence")
        for path in files:
            if path.exists():
                content = self._read(path, MAX_DOCUMENT_BYTES)
                records[path.name] = {"present": True, "bytes": len(content), "sha256": _sha(content),
                                      "reference": self.put_bytes(content)}
            else:
                _require(not path.is_symlink(), "legacy original cannot be a dangling symbolic link")
                records[path.name] = {"present": False}
        payload = _json({"schema_version": 1, "files": records})
        _require(len(payload) <= MAX_MANIFEST_BYTES, "legacy original manifest exceeds byte limit")
        _require(self.object_bytes + self._control_bytes() + len(payload) <= MAX_STORE_BYTES,
                 "evidence store byte budget exhausted")
        self._write_new(destination, payload)

    def compact_records(self, records):
        result, references = [], {}
        for record in records:
            row = dict(record)
            for field in FIELDS:
                value = row.get(field)
                if isinstance(value, str) and value:
                    if value.startswith("roster-identity-evidence:"):
                        self.get_bytes(value)
                    else:
                        digest = _sha(value.encode("utf-8"))
                        if digest not in references:
                            references[digest] = self.put_bytes(value.encode("utf-8"))
                        row[field] = references[digest]
            result.append(row)
        return result

    def resolve_records(self, records):
        result = []
        for record in records:
            row = dict(record)
            for field in FIELDS:
                value = row.get(field)
                if isinstance(value, str) and value.startswith("roster-identity-evidence:"):
                    row[field] = self.get_bytes(value).decode("utf-8")
            result.append(row)
        return result


def resolve_records(records, state_root):
    referenced = any(isinstance(row.get(field), str) and row[field].startswith("roster-identity-evidence:")
                     for row in records for field in FIELDS)
    store = IdentityEvidenceStore.for_state(state_root)
    if store is not None:
        _require(not referenced or (store.root / "legacy-originals.json").is_file(),
                 "referenced state is missing its exact legacy originals manifest")
        return store.resolve_records(records)
    _require(not referenced, "referenced identity evidence store is absent")
    return records
