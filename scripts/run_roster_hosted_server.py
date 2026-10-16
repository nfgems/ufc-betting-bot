"""Explicit native server preservation roles for the disabled roster controller.

A separately trusted launcher provisions the role trees, supplies exact external
pins and opens anonymous custody/helper pipes before starting these roles. This
command neither starts children nor selects default storage. Provisioning and
custody run as root. Startup reads the independently protected original anchor,
then irreversibly drops to the provisioned worker before the existing guarded
connected startup path. The existing UTC calendar and execution-disabled rules
remain in force; there is no scheduler, source activation or clock option.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _external(path, digest):
    from src.data.roster_hosted_store import _require, file_pin, verify_pin

    _require(path is not None and digest is not None, "Exact path and external hash required")
    value = verify_pin(file_pin(path))
    _require(value["sha256"] == digest, "External server role input hash mismatch")
    return value


def _guard_library(path, digest):
    from src.data.roster_hosted_store import _require, _safe

    target = _safe(path)
    for parent in target.parents:
        info = parent.lstat()
        _require(stat.S_ISDIR(info.st_mode) and info.st_uid == 0
                 and (not info.st_mode & 0o022 or info.st_mode & stat.S_ISVTX),
                 "Native guard library requires trusted root ancestry")
    info = target.lstat()
    _require(stat.S_ISREG(info.st_mode) and info.st_uid == 0
             and not info.st_mode & 0o022,
             "Native guard library requires independent root owner")
    return _external(path, digest)


def _pipe_ends(pairs):
    """Admit only the exact directional, preopened anonymous role channels."""
    import fcntl
    from src.data.roster_hosted_store import _require

    handles = []
    identities = []
    for fd, direction in pairs:
        _require(type(fd) is int and fd > 2 and fd not in handles,
                 "Distinct preopened role pipe descriptors required")
        info = os.fstat(fd)
        _require(stat.S_ISFIFO(info.st_mode)
                 and os.readlink("/proc/self/fd/" + str(fd)).startswith("pipe:["),
                 "Anonymous role pipe required")
        _require(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == direction,
                 "Role pipe direction mismatch")
        identity = (info.st_dev, info.st_ino)
        _require(identity not in identities, "Separate request and response channels required")
        handles.append(fd)
        identities.append(identity)
    for entry in Path("/proc/self/fd").iterdir():
        fd = int(entry.name)
        try:
            os.fstat(fd)
        except OSError:
            continue  # The descriptor used to enumerate this directory is closed.
        _require(fd <= 2 or fd in handles, "Unexpected inherited role descriptor")


def _drop_worker(worker_uid):
    from src.data.roster_hosted_store import _require

    _require(type(worker_uid) is int and worker_uid > 0, "Distinct provisioned worker UID required")
    os.setgroups([])
    os.setgid(worker_uid)
    os.setuid(worker_uid)
    _require(os.getresuid() == (worker_uid,) * 3 and os.getresgid() == (worker_uid,) * 3
             and os.getgroups() == [], "Irreversible worker ownership transition required")


def _connected_startup(arguments):
    # Fixed packaged implementation; never an arbitrary root command or module.
    from scripts.run_roster_hosted_connected import main

    return main(arguments)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("provision", "successor", "custody", "prepare", "startup"))
    for name in ("seccomp-library", "seccomp-library-sha256", "libc-library", "libc-library-sha256"):
        parser.add_argument("--" + name, required=True)
    for name in ("destination", "controls", "controls-sha256", "controller-binding-path",
                 "configuration", "configuration-sha256", "recovery-evidence", "recovery-evidence-sha256"):
        parser.add_argument("--" + name)
    for name in ("custody-request-fd", "custody-response-fd", "helper-read-fd", "helper-write-fd"):
        parser.add_argument("--" + name, type=int)
    args = parser.parse_args(argv)
    from src.data.roster_hosted_store import StoreError, _read, _require, verify_pin

    guard = None
    try:
        _require(sys.platform == "linux" and os.geteuid() == 0,
                 "Explicit native root server role required")
        _require((args.recovery_evidence is None) == (args.recovery_evidence_sha256 is None),
                 "Calendar reconciliation evidence and external hash required together")
        _require(args.mode in {"prepare", "successor"} or args.recovery_evidence is None,
                 "Calendar reconciliation evidence is preparation-only")
        provisioning = (args.destination, args.controls, args.controls_sha256, args.controller_binding_path)
        configuration = (args.configuration, args.configuration_sha256)
        descriptors = (args.custody_request_fd, args.custody_response_fd,
                       args.helper_read_fd, args.helper_write_fd)
        if args.mode == "provision":
            _require(all(value is not None for value in provisioning)
                     and all(value is None for value in configuration + descriptors),
                     "Provision requires only explicit destination, controls and future original binding path")
            _pipe_ends([])
        elif args.mode == "successor":
            _require(args.destination is not None and args.controller_binding_path is not None
                     and args.controls is None and args.controls_sha256 is None
                     and all(value is not None for value in configuration)
                     and all(value is None for value in descriptors),
                     "Successor requires prior configuration and fresh destination/binding only")
            _pipe_ends([])
        else:
            _require(all(value is None for value in provisioning)
                     and all(value is not None for value in configuration)
                     and args.custody_request_fd is not None and args.custody_response_fd is not None,
                     "Serving requires exact provisioned configuration and both custody pipes")
            _require((args.helper_read_fd is None) == (args.helper_write_fd is None),
                     "Both preopened helper pipe ends required")
            _require(args.mode in {"prepare", "startup"} or args.helper_read_fd is None,
                     "Custody must not inherit helper pipe descriptors")
            pairs = [(args.custody_request_fd, os.O_RDONLY if args.mode == "custody" else os.O_WRONLY),
                     (args.custody_response_fd, os.O_WRONLY if args.mode == "custody" else os.O_RDONLY)]
            if args.helper_read_fd is not None:
                pairs.extend(((args.helper_read_fd, os.O_RDONLY), (args.helper_write_fd, os.O_WRONLY)))
            _pipe_ends(pairs)
        from src.data.roster_hosted_wire import install_parent_network_deny

        guard = install_parent_network_deny(
            seccomp_library_pin=_guard_library(args.seccomp_library, args.seccomp_library_sha256),
            libc_library_pin=_guard_library(args.libc_library, args.libc_library_sha256))
        from src.data.roster_hosted_preservation import (
            open_server_custody, provision_server_preservation, provision_server_successor, read_server_startup_anchor,
        )

        if args.mode == "provision":
            result = provision_server_preservation(args.destination,
                controls_pin=_external(args.controls, args.controls_sha256),
                expected_controls_sha256=args.controls_sha256,
                controller_binding_path=args.controller_binding_path)
            print(json.dumps({"status": "server_preservation_provisioned_execution_disabled",
                "configuration": result, "expected_configuration_sha256": result["sha256"],
                "parent_network_guard": guard}, sort_keys=True))
            return 0
        config_pin = _external(args.configuration, args.configuration_sha256)
        if args.mode == "successor":
            result = provision_server_successor(args.destination,
                previous_configuration_pin=config_pin,
                expected_previous_config_sha256=args.configuration_sha256,
                controller_binding_path=args.controller_binding_path,
                recovery_evidence_pin=_external(args.recovery_evidence, args.recovery_evidence_sha256)
                    if args.recovery_evidence is not None else None)
            print(json.dumps({"status": "server_successor_provisioned_execution_disabled",
                "configuration": result, "expected_configuration_sha256": result["sha256"],
                "parent_network_guard": guard}, sort_keys=True))
            return 0
        if args.mode == "custody":
            from src.data.roster_hosted_custody import serve_custody

            authority = open_server_custody(config_pin, expected_config_sha256=args.configuration_sha256)
            try:
                serve_custody(authority, args.custody_request_fd, args.custody_response_fd)
            except StoreError as exc:
                if str(exc) != "Custody pipe closed; preserve uncertain work":
                    raise
            print(json.dumps({"status": "server_custody_exited_on_pipe_eof",
                "parent_network_guard": guard, "source_requests": 0}, sort_keys=True))
            return 0
        if args.mode == "prepare":
            from src.data.roster_hosted_preservation import NativeServerPreservation
            preservation = NativeServerPreservation(config_pin,
                expected_config_sha256=args.configuration_sha256)
            controls = json.loads(_read(verify_pin(preservation.config["controls"])["path"]))
            recovery = preservation.config.get("recovery_evidence")
            if args.recovery_evidence is not None:
                from src.data.roster_hosted_custody import _trusted
                _trusted(args.recovery_evidence, directory=False)
                recovery = _external(args.recovery_evidence, args.recovery_evidence_sha256)
            output = preservation.preparation_output(recovery_evidence_pin=recovery)
            arguments = ["prepare", "--output", str(output)]
            if recovery is not None:
                arguments.extend(("--recovery-evidence", recovery["path"],
                                  "--recovery-evidence-sha256", recovery["sha256"]))
        else:
            anchor = read_server_startup_anchor(config_pin,
                expected_config_sha256=args.configuration_sha256)
            controls = json.loads(_read(verify_pin(anchor["controls"])["path"]))
            arguments = ["startup", "--binding", anchor["binding"]["path"],
                         "--binding-sha256", anchor["expected_binding_sha256"]]
        _drop_worker(controls["worker_uid"])
        for name in ("custody_request_fd", "custody_response_fd", "seccomp_library",
                     "seccomp_library_sha256", "libc_library", "libc_library_sha256",
                     "helper_read_fd", "helper_write_fd"):
            value = getattr(args, name)
            if value is not None:
                arguments.extend(("--" + name.replace("_", "-"), str(value)))
        return _connected_startup(arguments)
    except (RuntimeError, OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "failed_execution_disabled",
            "error": type(exc).__name__ + ": " + str(exc), "source_requests": 0,
            "server_role": args.mode, "effective_uid": os.geteuid(), "parent_network_guard": guard,
            "activation_authorized": False, "publication_authorized": False,
            "replay_authorized": False, "schedule_enabled": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
