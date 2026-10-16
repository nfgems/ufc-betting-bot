"""Assemble an inert local Docker context from explicitly reviewed component pins.

This tool never builds an image, imports the application, loads a model, creates
request authority or accesses the network. Its input JSON and runtime archive
must be independently reviewed and quiescent. Failures retain all output; never
reuse a partial destination. Windows output is an artifact, not native custody.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile

MAX_FILE, MAX_TOTAL, MAX_ENTRIES, RESERVE = 128 << 20, 1 << 30, 10000, 512 << 20
PATHS = {"source": "source", "python": "python/bin/python3.11", "dependencies": "dependencies",
         "libc": "native/libc.so", "seccomp": "native/libseccomp.so"}
STARTUP = {"entrypoint.sh", "scripts/start_web_service.sh", "scripts/roster_startup_storage.py"}
RECIPES = {"deploy/roster-service/Dockerfile", "railway.toml"}
ASSETS = {
    "models/current_production_model.json", "models/xgboost_model.pkl",
    "models/xgboost_no_odds_model.pkl", "models/logistic_model.pkl",
    "models/full_live_contract_v6_durability_fullfit_spec.json",
    "data/processed/fights_cleaned.csv", "data/processed/features.csv",
    "data/raw/ufc_active_roster_official.csv", "data/raw/ufc_fighters_scraped.csv",
    "data/raw/ufc-fighter-details.csv",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def sha(body):
    return hashlib.sha256(body).hexdigest()


def relative(value):
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value
            and not any(ord(c) < 32 for c in value), "Unsafe relative path")
    p = PurePosixPath(value)
    require(not p.is_absolute() and str(p) == value and ".." not in p.parts
            and all(x.rstrip(" .") == x for x in p.parts), "Noncanonical relative path")
    require(not any(x.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(10)],
                    *[f"LPT{i}" for i in range(10)]} for x in p.parts), "Reserved host path")
    return value


def safe_path(value, exists=True):
    p = Path(value)
    require(p.is_absolute() and ".." not in p.parts, "Absolute path required")
    require(p.resolve(strict=exists) == p, "Linked or noncanonical input refused")
    for q in (p, *p.parents):
        if not q.exists():
            continue
        s = q.lstat()
        require(not stat.S_ISLNK(s.st_mode) and not getattr(s, "st_file_attributes", 0) & 0x400,
                "Linked/reparse ancestry refused")
    return p


def identity(s):
    # Windows path stat and descriptor stat can expose different ctime semantics.
    # Compare shared identity here; require each API's own ctime stable separately.
    # Windows path stat also synthesizes executable bits for .bat/.exe names;
    # descriptor stat does not. Neither represents native artifact permissions.
    mode = s.st_mode & ~0o111 if os.name == "nt" else s.st_mode
    return (s.st_dev, s.st_ino, mode, s.st_nlink, s.st_size, s.st_mtime_ns,
            getattr(s, "st_birthtime_ns", 0), getattr(s, "st_file_attributes", 0),
            getattr(s, "st_uid", 0), getattr(s, "st_gid", 0))


def pin_shape(row):
    require(type(row["bytes"]) is int and 0 <= row["bytes"] <= MAX_FILE
            and isinstance(row["sha256"], str) and re.fullmatch("[0-9a-f]{64}", row["sha256"]), "Invalid pin")


def read_exact(path, row, destination=None, collect=False):
    pin_shape(row)
    p = safe_path(path)
    before = p.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size == row["bytes"],
            "Unshared pinned regular file required")
    digest, count, chunks = hashlib.sha256(), 0, []
    with p.open("rb") as f:
        opened = os.fstat(f.fileno())
        require(identity(opened) == identity(before), "Input changed before read")
        while chunk := f.read(1 << 20):
            count += len(chunk)
            require(count <= row["bytes"], "Input grew")
            digest.update(chunk)
            if destination is not None:
                destination.write(chunk)
            if collect:
                chunks.append(chunk)
        after_fd, after_path = os.fstat(f.fileno()), p.lstat()
        require(identity(after_fd) == identity(before) == identity(after_path)
                and opened.st_ctime_ns == after_fd.st_ctime_ns
                and before.st_ctime_ns == after_path.st_ctime_ns
                and opened.st_mode == after_fd.st_mode
                and before.st_mode == after_path.st_mode, "Input changed during read")
    require(count == row["bytes"] and digest.hexdigest() == row["sha256"], "Input pin mismatch")
    return b"".join(chunks)


def member_read(archive, member, row, destination=None, collect=False):
    require(member.isfile() and member.size == row["bytes"] <= MAX_FILE, "Invalid archive member size/type")
    digest, count, chunks = hashlib.sha256(), 0, []
    with archive.extractfile(member) as f:
        while chunk := f.read(1 << 20):
            count += len(chunk)
            require(count <= row["bytes"], "Archive member grew")
            digest.update(chunk)
            if destination is not None:
                destination.write(chunk)
            if collect:
                chunks.append(chunk)
    require(count == row["bytes"] and digest.hexdigest() == row["sha256"], "Archive member pin mismatch")
    return b"".join(chunks)


def budget(root):
    total, entries = 0, 0
    for p in root.rglob("*"):
        s = p.lstat()
        require(not p.is_symlink() and not getattr(s, "st_file_attributes", 0) & 0x400, "Linked evidence refused")
        entries += 1
        if stat.S_ISREG(s.st_mode):
            require(s.st_size <= MAX_FILE, "Evidence file bound exceeded")
            total += s.st_size
        else:
            require(stat.S_ISDIR(s.st_mode), "Special evidence entry")
    require(total <= MAX_TOTAL and entries <= MAX_ENTRIES, "Combined evidence bound exceeded")
    require(shutil.disk_usage(root).free >= RESERVE, "Free-space reserve required")
    return total, entries


def context_report(files, spec, spec_sha256, runtime, groups, artifact_format):
    rows = [{"relative": n, **{k: row[k] for k in ("bytes", "sha256", "mode")}} for n, row in sorted(files.items())]
    return {"schema": 1, "kind": "roster_service_context_inventory_v1", "purpose": spec["purpose"],
            "artifact_format": artifact_format, "spec_sha256": spec_sha256, "base_image": spec["base_image"], "base_reviewed": False,
            "image_built": False, "native_startup_tested": False, "model_compatibility_tested": False,
            "deployment_authorized": False, "runtime_manifest_sha256": runtime["manifest_sha256"],
            "files": rows, "component_counts": {k: len(v) for k, v in groups.items()},
            "qualification": "Inert existing-asset context only. POSIX modes are intended image modes; host metadata is not native custody. Base/OS/dependencies/browser review and image build remain outstanding. No V9 release or model compatibility claim."}


def archive_context(output, root, files, directories, archive, members, report):
    """Stream a new compressed context; never read a failed context or extract it."""
    require(output.name.endswith(".tar.gz"), "Compressed context name must end .tar.gz")
    sidecar = output.with_name(output.name + ".manifest.json")
    require(not os.path.lexists(sidecar), "Fresh archive manifest required")
    body = (json.dumps(report, indent=2, sort_keys=True)+"\n").encode()
    require(len(body) <= 8 << 20, "Bounded context manifest required")
    require(sum(row["bytes"] for row in files.values()) + len(body) <= MAX_TOTAL
            and len(files) + len(directories) + 2 <= MAX_ENTRIES, "Decoded context bounds exceeded")
    total, entries = budget(root)
    require(total + MAX_FILE + len(body) <= MAX_TOTAL and entries + 2 <= MAX_ENTRIES,
            "Compressed context would exceed retained evidence bounds")
    require(shutil.disk_usage(root).free >= MAX_FILE + len(body) + RESERVE, "Archive reserve required")

    class BoundedWriter:
        def __init__(self, stream):
            self.stream, self.count = stream, 0

        def write(self, chunk):
            require(self.count + len(chunk) <= MAX_FILE, "Compressed context file cap reached; partial retained")
            self.count += len(chunk)
            return self.stream.write(chunk)

        def flush(self):
            self.stream.flush()

    def directory_mode(name):
        return 0o555 if name == "bundle/roster-runtime" or name.startswith("bundle/roster-runtime/") else 0o755

    errors = []
    with output.open("xb") as sink:
        try:
            with gzip.GzipFile(filename="", mode="wb", fileobj=BoundedWriter(sink), compresslevel=9, mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as target:
                    for name in sorted(directories):
                        info = tarfile.TarInfo(name)
                        info.type, info.mode = tarfile.DIRTYPE, directory_mode(name)
                        target.addfile(info)
                    for name, row in sorted(files.items()):
                        if "member" in row:
                            data = member_read(archive, members[row["member"]], row, collect=True)
                        else:
                            data = read_exact(row["path"], row, collect=True)
                        info = tarfile.TarInfo(name)
                        info.size, info.mode = len(data), row["mode"]
                        target.addfile(info, io.BytesIO(data))
                    info = tarfile.TarInfo("context-manifest.json")
                    info.size, info.mode = len(body), 0o644
                    target.addfile(info, io.BytesIO(body))
        except BaseException as exc:
            errors.append(exc)
        finally:
            for operation in (sink.flush, lambda: os.fsync(sink.fileno())):
                try:
                    operation()
                except BaseException as exc:
                    errors.append(exc)
    if errors:
        raise BaseExceptionGroup("Compressed context errors; every partial retained", errors)
    expected = dict(files)
    expected["context-manifest.json"] = {"bytes": len(body), "sha256": sha(body), "mode": 0o644}
    seen_files, seen_dirs = set(), set()
    # Sequential readback of the complete inert archive, without extraction.
    with tarfile.open(output, "r|gz") as check:
        for member in check:
            name = relative(member.name)
            require(member.uid == member.gid == 0 and set(member.pax_headers) <= {"path"}
                    and member.pax_headers.get("path", name) == name, "Output archive metadata differs")
            if member.isdir():
                require(name in directories and name not in seen_dirs and member.mode == directory_mode(name), "Output archive directory differs")
                seen_dirs.add(name)
            else:
                require(member.isfile() and name in expected and name not in seen_files, "Output archive file set differs")
                row = expected[name]
                require(member.mode == row["mode"], "Output archive file mode differs")
                member_read(check, member, row)
                seen_files.add(name)
    require(seen_files == set(expected) and seen_dirs == directories, "Complete compressed context required")
    # tarfile's streaming reader does not validate the gzip trailer. Consume
    # the entire wrapper separately so CRC, ISIZE and truncated EOF also refuse.
    decoded_bytes = 0
    with gzip.open(output, "rb") as decoded:
        while chunk := decoded.read(1 << 20):
            decoded_bytes += len(chunk)
            require(decoded_bytes <= MAX_TOTAL, "Decoded archive byte cap exceeded")
    with sidecar.open("xb") as stream:
        stream.write(body); stream.flush(); os.fsync(stream.fileno())
    read_exact(sidecar, {"bytes": len(body), "sha256": sha(body)})
    with output.open("rb") as stream:
        output_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(report, archive={"path": str(output), "bytes": output.stat().st_size, "sha256": output_sha}, decoded_tar_bytes=decoded_bytes,
                manifest={"path": str(sidecar), "bytes": len(body), "sha256": sha(body)})


def assemble(spec_path, spec_sha256, destination, evidence_root, artifact_format="directory"):
    require(artifact_format in {"directory", "tar-gzip"}, "Unknown context artifact format")
    root = safe_path(evidence_root)
    output = safe_path(destination, exists=False)
    require(root in output.parents and output.parent.is_dir() and not os.path.lexists(output),
            "Fresh destination beneath evidence root required")
    require(re.fullmatch("[0-9a-f]{64}", spec_sha256 or ""), "Specification SHA256 required")
    spec_file = safe_path(spec_path)
    spec = json.loads(read_exact(spec_file, {"bytes": spec_file.stat().st_size, "sha256": spec_sha256}, collect=True), object_pairs_hook=unique)
    require(set(spec) == {"schema", "kind", "purpose", "base_image", "runtime", "files"}
            and type(spec["schema"]) is int and spec["schema"] == 1
            and spec["kind"] == "roster_service_context_v1"
            and spec["purpose"] == "existing_assets_local_baseline", "Exact baseline context specification required")
    base = spec["base_image"]
    require(base is None or isinstance(base, str) and re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", base),
            "Base must be absent or immutable digest reference")
    runtime = spec["runtime"]
    require(set(runtime) == {"path", "bytes", "sha256", "manifest_sha256"}, "Exact runtime input required")
    read_exact(runtime["path"], runtime)
    archive_path = safe_path(runtime["path"])
    require(output not in archive_path.parents, "Runtime input overlaps destination")
    archive_stat = archive_path.stat()
    archive_before = (identity(archive_stat), archive_stat.st_ctime_ns, archive_stat.st_mode)
    files, groups = {}, {k: set() for k in ("web", "startup", "asset", "recipe")}
    require(isinstance(spec["files"], list) and 0 < len(spec["files"]) < MAX_ENTRIES, "Explicit files required")
    for row in spec["files"]:
        require(set(row) == {"path", "relative", "bytes", "sha256", "component"}, "Exact context file row required")
        name, group = relative(row["relative"]), row["component"]
        require(group in groups and name.casefold() not in {x.casefold() for x in files}, "Unknown component/duplicate destination")
        if group == "recipe":
            require(name in RECIPES, "Unknown build recipe")
            component_name = name
        else:
            require(name.startswith("bundle/app/"), "Application path required")
            component_name = name[len("bundle/app/"):]
        groups[group].add(component_name)
        p = safe_path(row["path"])
        require(output not in p.parents, "Source overlaps destination")
        read_exact(p, row)
        files[name] = {**row, "mode": 0o755 if component_name == "entrypoint.sh" else 0o644}
    require(groups["startup"] == STARTUP and groups["recipe"] == RECIPES
            and groups["asset"] == ASSETS, "Incomplete or excessive baseline startup/assets/recipes")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {}
        for m in archive:
            require(len(members) < MAX_ENTRIES and (m.isdir() or m.isfile()), "Unexpected archive entry")
            name = relative(m.name)
            require(set(m.pax_headers) <= {"mtime", "path"}
                    and m.pax_headers.get("path", name) == name, "Unexpected archive metadata")
            if "mtime" in m.pax_headers:
                timestamp = m.pax_headers["mtime"]
                require(re.fullmatch(r"[0-9]{1,12}(\.[0-9]{1,9})?", timestamp)
                        and float(timestamp) <= 253402300799, "Unexpected archive timestamp")
            require(name.casefold() not in {x.casefold() for x in members}, "Duplicate archive path")
            require(m.uid == 0 and m.gid == 0, "Archive ownership differs")
            require(name == "runtime" or name.startswith("runtime/"), "Unexpected runtime archive root")
            require(m.mode == (0o555 if m.isdir() or name == "runtime/python/bin/python3.11" else 0o444), "Runtime mode differs")
            members[name] = m
        require("runtime/runtime-manifest.json" in members, "Runtime manifest missing")
        mm = members["runtime/runtime-manifest.json"]
        manifest = json.loads(member_read(archive, mm, {"bytes": mm.size, "sha256": runtime["manifest_sha256"]}, collect=True), object_pairs_hook=unique)
        require(set(manifest) == {"schema", "kind", "paths", "files"} and type(manifest["schema"]) is int and manifest["schema"] == 1
                and manifest["kind"] == "roster_runtime_package_v1", "Normal runtime package manifest required")
        require(manifest["paths"] == PATHS and isinstance(manifest["files"], list)
                and 0 < len(manifest["files"]) < MAX_ENTRIES, "Exact runtime layout required")
        expected = {}
        for row in manifest["files"]:
            require(set(row) == {"relative", "bytes", "sha256"}, "Invalid runtime manifest row")
            name = relative(row["relative"])
            pin_shape(row)
            require(name not in expected, "Repeated runtime manifest member")
            expected[name] = row
        actual = {n[len("runtime/"):] for n, m in members.items() if m.isfile() and n != "runtime/runtime-manifest.json"}
        require(actual == set(expected), "Runtime exact file set differs")
        require({"python/bin/python3.11", "python/lib/libpython3.11.so.1.0", "python/lib/python3.11/encodings/__init__.py",
                 "native/libc.so", "native/libseccomp.so", "source/scripts/launch_roster_hosted_server.py",
                 "source/scripts/roster_railway_startup.py", "source/scripts/run_roster_hosted_server.py",
                 "source/scripts/run_roster_hosted_saved_helper.py"} <= expected.keys(), "Required runtime members missing")
        expected_dirs = {str(p) for n,m in members.items() if m.isfile() for p in PurePosixPath(n).parents if str(p) != "."}
        require({n for n,m in members.items() if m.isdir()} == expected_dirs, "Runtime exact directory set differs")
        web = {n[len("web-source/"):] for n in expected if n.startswith("web-source/")}
        require(groups["web"] == web and {"src/web/app.py", "src/web/serve.py", "src/config.py"} <= web,
                "Separate complete web component differs")
        for name in groups["web"] | STARTUP:
            prefix = "web-source/" if name in web else "source/"
            require(prefix + name in expected, "Startup source absent from normal package")
            row, packaged = files["bundle/app/" + name], expected[prefix + name]
            require((row["bytes"], row["sha256"]) == (packaged["bytes"], packaged["sha256"]), "Application component differs from runtime")
        for name in RECIPES:
            packaged = expected["source/deploy/roster-service/" + PurePosixPath(name).name]
            require((files[name]["bytes"], files[name]["sha256"]) == (packaged["bytes"], packaged["sha256"]), "Recipe differs from reviewed runtime")
        for name, row in expected.items():
            member_read(archive, members["runtime/" + name], row)
            files["bundle/roster-runtime/" + name] = {**row, "mode": members["runtime/" + name].mode, "member": "runtime/" + name}
        files["bundle/roster-runtime/runtime-manifest.json"] = {"bytes": mm.size, "sha256": runtime["manifest_sha256"], "mode": 0o444, "member": mm.name}
        directories = {str(p) for n in files for p in PurePosixPath(n).parents if str(p) != "."}
        require(not directories.intersection(files), "File/directory collision")
        report = context_report(files, spec, spec_sha256, runtime, groups, artifact_format)
        if artifact_format == "tar-gzip":
            result = archive_context(output, root, files, directories, archive, members, report)
            archive_after = archive_path.stat()
            require((identity(archive_after), archive_after.st_ctime_ns, archive_after.st_mode) == archive_before, "Runtime archive changed during assembly")
            read_exact(archive_path, runtime)
            budget(root)
            return result
        total, entries = budget(root)
        require(total + sum(r["bytes"] for r in files.values()) + (8 << 20) <= MAX_TOTAL
                and entries + len(files) + len(directories) + 2 <= MAX_ENTRIES, "Context exceeds combined evidence bounds")
        require(shutil.disk_usage(root).free >= sum(r["bytes"] for r in files.values()) + RESERVE, "Context reserve required")
        output.mkdir()
        for name in sorted(directories, key=lambda n: (len(PurePosixPath(n).parts), n)):
            (output/name).mkdir()
        for name, row in sorted(files.items()):
            require(shutil.disk_usage(root).free >= row["bytes"] + RESERVE, "Reserve exhausted")
            with (output/name).open("xb") as f:
                if "member" in row:
                    member_read(archive, members[row["member"]], row, destination=f)
                else:
                    read_exact(row["path"], row, destination=f)
                f.flush()
                os.fsync(f.fileno())
        archive_after = archive_path.stat()
        require((identity(archive_after), archive_after.st_ctime_ns, archive_after.st_mode) == archive_before, "Runtime archive changed during assembly")
    read_exact(archive_path, runtime)
    with (output/"context-manifest.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    for row in report["files"]:
        read_exact(output/row["relative"], row)
    require({p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
            == set(files) | {"context-manifest.json"}, "Output file set differs")
    budget(root)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("spec", "spec-sha256", "destination", "evidence-root"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--artifact-format", choices=("directory", "tar-gzip"), default="directory")
    args = parser.parse_args()
    result = assemble(args.spec, args.spec_sha256, args.destination, args.evidence_root, args.artifact_format)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, sort_keys=True))
