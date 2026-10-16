"""Assemble an explicit, externally pinned offline component; never run its code.

The trusted caller reviews the complete specification and quiesces its inputs.
Failure retains every created byte; a partial destination cannot be reused.
This packages explicit files, not a deployable service or all system libraries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys

MAX_FILE, MAX_TOTAL, MAX_ENTRIES, RESERVE = 128 << 20, 1 << 30, 10000, 512 << 20
PATHS = dict(source="source", python="python/bin/python3.11", dependencies="dependencies",
             libc="native/libc.so", seccomp="native/libseccomp.so")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute(value):
    require(isinstance(value, str) and value.startswith("/") and not value.startswith("//") and "\\" not in value,
            "Explicit canonical absolute path required")
    path = Path(value)
    require(str(path) == value and ".." not in path.parts, "Noncanonical path")
    return path


def regular(path):
    require(path.resolve(strict=True) == path, "Symbolic input path refused")
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Unshared regular input required")
    require(info.st_size <= MAX_FILE, "Input exceeds file bound")
    return info


def exact_read(path, expected, destination=None):
    before = regular(path)
    require(before.st_size == expected["bytes"], "Input size differs from pin")
    digest, count, chunks = hashlib.sha256(), 0, []
    identity = lambda value: tuple(getattr(value, "st_" + name) for name in
                                  ("dev", "ino", "mode", "uid", "gid", "nlink", "size", "mtime_ns", "ctime_ns"))
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        require(identity(os.fstat(stream.fileno())) == identity(before), "Input identity changed before read")
        while chunk := stream.read(1 << 20):
            count += len(chunk)
            require(count <= expected["bytes"], "Input grew beyond pin")
            digest.update(chunk)
            if destination is not None:
                destination.write(chunk)
            elif expected.get("collect"):
                chunks.append(chunk)
        require(identity(os.fstat(stream.fileno())) == identity(before) and identity(path.lstat()) == identity(before),
                "Input metadata changed during read")
    require(count == expected["bytes"] and digest.hexdigest() == expected["sha256"],
            "Input SHA256 differs from pin")
    return b"".join(chunks)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def package(spec_path, spec_sha256, destination):
    require(sys.platform == "linux" and os.geteuid() == 0, "Native Linux root required")
    spec_path, root = absolute(spec_path), absolute(destination)
    require(not os.path.lexists(root), "Unused destination required; partials are preserved")
    for parent in root.parents:
        info = parent.lstat()
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == 0 and not info.st_mode & 0o022,
                "Destination requires protected root-owned ancestry")
    require(not any(root == Path(p) or Path(p) in root.parents for p in ("/app", "/tmp", "/var/tmp", "/mnt")),
            "Volatile or Windows destination refused")
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), left.split()[4]))
        if root == mount or mount in root.parents:
            mounts.append((len(mount.parts), right.split()[0]))
    require(mounts and max(mounts)[1] in {"ext2", "ext3", "ext4", "xfs", "btrfs", "overlay", "zfs"},
            "Native nonvolatile destination filesystem required")
    require(isinstance(spec_sha256, str) and re.fullmatch("[0-9a-f]{64}", spec_sha256), "SHA256 required")
    spec = json.loads(exact_read(spec_path, dict(bytes=regular(spec_path).st_size,
                      sha256=spec_sha256, collect=True)), object_pairs_hook=unique_object)
    require(set(spec) == {"schema", "kind", "files"} and type(spec["schema"]) is int and spec["schema"] == 1
            and spec["kind"] == "roster_runtime_package_spec_v1", "Exact package specification required")
    require(isinstance(spec["files"], list) and 0 < len(spec["files"]) < MAX_ENTRIES, "Bounded files required")
    files, directories, total = {}, set(), 0
    for row in spec["files"]:
        require(isinstance(row, dict) and set(row) == {"path", "relative", "bytes", "sha256"}, "Exact file pin required")
        relative = row["relative"]
        require(isinstance(relative, str) and "\\" not in relative and not any(ord(c) < 32 for c in relative), "Unsafe relative path")
        target = PurePosixPath(relative)
        require(not target.is_absolute() and str(target) == relative and ".." not in target.parts
                and len(target.parts) > 1 and target.parts[0] in {"source", "web-source", "python", "dependencies", "native"}, "Invalid package layout")
        require(target.parts[0] != "python" or relative == PATHS["python"]
                or (len(target.parts) > 2 and target.parts[1] == "lib"), "Invalid Python runtime layout")
        require(target.parts[0] != "native" or relative in {PATHS["libc"], PATHS["seccomp"]}, "Unexpected native library path")
        require(relative not in files and relative != "runtime-manifest.json", "Duplicate or reserved destination")
        require(type(row["bytes"]) is int and 0 <= row["bytes"] <= MAX_FILE
                and isinstance(row["sha256"], str) and re.fullmatch("[0-9a-f]{64}", row["sha256"]), "Invalid file pin")
        source = absolute(row["path"])
        require(root != source and root not in source.parents and source not in root.parents, "Input overlaps destination")
        files[relative] = row
        directories.update(str(p) for p in target.parents if str(p) != ".")
        total += row["bytes"]
        require(total <= MAX_TOTAL and len(files) + len(directories) + 2 <= MAX_ENTRIES,
                "Package specification exceeds bounds")
        exact_read(source, row)
    require(not directories.intersection(files), "File/directory collision")
    if "web-source" in directories:
        require({"web-source/src/web/serve.py", "web-source/src/web/app.py",
                 "web-source/src/config.py"} <= files.keys(), "Incomplete separate web source component")
    require({PATHS["python"], PATHS["libc"], PATHS["seccomp"], "python/lib/libpython3.11.so.1.0",
             "python/lib/python3.11/encodings/__init__.py", "source/scripts/launch_roster_hosted_server.py",
             "source/scripts/roster_railway_startup.py", "source/scripts/run_roster_hosted_server.py",
             "source/scripts/run_roster_hosted_saved_helper.py"} <= files.keys()
            and {"source", "python", "dependencies", "native"} <= directories, "Required runtime layout missing")
    rows = [{key: files[name][key] for key in ("relative", "bytes", "sha256")} for name in sorted(files)]
    body = (json.dumps(dict(schema=1, kind="roster_runtime_package_v1", paths=PATHS, files=rows), sort_keys=True) + "\n").encode()
    require(len(files) + len(directories) + 2 <= MAX_ENTRIES and len(body) <= MAX_FILE
            and total + len(body) <= MAX_TOTAL, "Package exceeds entry/output bounds")
    require(shutil.disk_usage(root.parent).free >= total + len(body) + RESERVE, "Free-space reserve required")
    root.mkdir(mode=0o700)
    for name in sorted(directories, key=lambda p: (len(PurePosixPath(p).parts), p)):
        (root/name).mkdir(mode=0o700)
    for name, row in sorted(files.items()):
        require(shutil.disk_usage(root).free >= row["bytes"] + RESERVE, "Free-space reserve exhausted")
        with (root/name).open("xb") as stream:
            exact_read(absolute(row["path"]), row, stream)
            stream.flush()
            os.fchmod(stream.fileno(), 0o555 if name == PATHS["python"] else 0o444)
            os.fsync(stream.fileno())
    manifest = root/"runtime-manifest.json"
    require(shutil.disk_usage(root).free >= len(body) + RESERVE, "Manifest free-space reserve exhausted")
    with manifest.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fchmod(stream.fileno(), 0o444)
        os.fsync(stream.fileno())
    for name in sorted(directories, key=lambda p: len(PurePosixPath(p).parts), reverse=True):
        (root/name).chmod(0o555)
        sync_directory(root/name)
    root.chmod(0o555)
    sync_directory(root)
    sync_directory(root.parent)
    return dict(path=str(manifest), bytes=len(body), sha256=hashlib.sha256(body).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("spec", "spec-sha256", "destination"):
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.spec, args.spec_sha256, args.destination), sort_keys=True))
