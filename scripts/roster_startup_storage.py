"""Read-only startup admission for an explicitly provisioned refresh namespace.

This does not provision custody, repair ownership, start a workflow or grant
request authority. An absent namespace retains legacy startup behavior. The
normal launcher and all role-specific checks remain separate requirements.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import sys

RESERVED_NAME = "roster_refresh"
MAX_INSPECTION_ENTRIES = 10000


class StorageAdmissionError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise StorageAdmissionError(message)


def _absolute(value):
    raw = os.fspath(value)
    path = Path(raw)
    _require(path.is_absolute() and raw == os.path.normpath(raw),
             "Refresh storage requires canonical absolute paths")
    return path


def _overlaps(left, right):
    return left == right or left in right.parents or right in left.parents


def _no_symlink_components(path):
    for item in (*reversed(path.parents), path):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        _require(not stat.S_ISLNK(info.st_mode),
                 "Refresh storage cannot share startup paths containing symlinks")


def _trusted_directory(path, *, writable_sticky=False):
    info = path.lstat()
    _require(stat.S_ISDIR(info.st_mode) and info.st_uid == 0,
             "Refresh storage ancestry must be native root owned")
    _require(not info.st_mode & 0o022 or
             (writable_sticky and bool(info.st_mode & stat.S_ISVTX)),
             "Refresh storage ancestry permits replacement")


def preflight(data_dir, log_dir, volume_dir, *, models_dir="/app/models",
              manifest_path=None, app_root="/app", extra_paths=()):
    """Validate existing storage without writing or changing any permissions.

    ``app_root`` is an explicit component-test seam; the CLI always uses /app.
    No application environment switch relocates the protected namespace. This
    is a quiescent startup check, not a concurrent-writer transaction. Mutable
    recursive chown must separately use -hR -P to avoid following links.
    """
    # lexists includes dangling links, which must fail admission when present.
    # Check the fixed location independently of overrides: changing data_dir
    # must not hide existing protected state from a sweep of the log volume.
    candidates = [Path(app_root) / "logs" / RESERVED_NAME,
                  Path(data_dir) / RESERVED_NAME, Path(log_dir) / RESERVED_NAME]
    if volume_dir:
        candidates.append(Path(volume_dir) / RESERVED_NAME)
    if not any(os.path.lexists(path) for path in candidates):
        return {"protected_namespace_present": False}
    _require(os.name == "posix" and os.geteuid() == 0,
             "Refresh storage admission requires native root")
    app = _absolute(app_root)
    data = _absolute(data_dir)
    logs = _absolute(log_dir)
    volume = _absolute(volume_dir or "")
    reserved = data / RESERVED_NAME
    _require(data == app / "logs" and logs == data and volume == data,
             "Protected refresh storage requires the reviewed shared Railway log volume layout")
    _no_symlink_components(reserved)
    for parent in reserved.parents:
        _trusted_directory(parent, writable_sticky=True)
    _trusted_directory(reserved)

    mutable = [data / name for name in ("raw", "processed", "operator", "tmp", "production_bundle")]
    mutable.extend((app / "data", app / "models", _absolute(models_dir)))
    if manifest_path:
        mutable.append(_absolute(manifest_path))
    mutable.extend(_absolute(value) for value in extra_paths if value)
    for path in mutable:
        _require(not _overlaps(path, reserved),
                 "Mutable startup path overlaps protected refresh storage")
        _no_symlink_components(path)
        _require(not _overlaps(path.resolve(strict=False), reserved),
                 "Resolved startup path overlaps protected refresh storage")

    # Hard-linked aliases would let chown on a mutable sibling change protected
    # metadata. Refuse shared files and links; never try to repair their owner.
    count = 1
    root_info = reserved.lstat()
    protected_identities = {(root_info.st_dev, root_info.st_ino)}
    pending = [reserved]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                count += 1
                _require(count <= MAX_INSPECTION_ENTRIES,
                         "Protected refresh storage inspection entry limit")
                info = entry.stat(follow_symlinks=False)
                identity = (info.st_dev, info.st_ino)
                _require(identity not in protected_identities,
                         "Protected refresh storage repeats a native object")
                protected_identities.add(identity)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                else:
                    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                             "Protected refresh storage requires unshared regular files")
    # Legacy migration, touch and nonrecursive chown can follow links even
    # though recursive chown does not. Refuse mutable links before those paths
    # execute. This deliberately does not support symlink-based shared layouts.
    pending = [data, *mutable]
    seen = set()
    while pending:
        path = pending.pop()
        if path == reserved:
            continue
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        identity = (info.st_dev, info.st_ino)
        _require(identity not in protected_identities,
                 "Mutable startup path aliases a protected native object")
        if identity in seen:
            continue
        seen.add(identity)
        count += 1
        _require(count <= MAX_INSPECTION_ENTRIES,
                 "Protected refresh storage inspection entry limit")
        _require(not stat.S_ISLNK(info.st_mode),
                 "Mutable startup tree contains a symlink beside protected refresh storage")
        if stat.S_ISDIR(info.st_mode):
            with os.scandir(path) as entries:
                for entry in entries:
                    _require(count + len(pending) < MAX_INSPECTION_ENTRIES,
                             "Protected refresh storage inspection entry limit")
                    pending.append(Path(entry.path))
    return {"protected_namespace_present": True, "inspected_entries": count}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--volume-dir", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--manifest-path")
    parser.add_argument("--extra-path", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        preflight(args.data_dir, args.log_dir, args.volume_dir,
                  models_dir=args.models_dir, manifest_path=args.manifest_path,
                  extra_paths=args.extra_path)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print("[startup] protected refresh storage refused: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
