"""One bounded local connection to the existing disabled server roles.

The trusted caller supplies the exact reviewed runtime, original native custody
controls and role trees. Initial saved preparation binds an unused protected
helper journal and launches the existing independent saved helper. Startup uses
that same journal; older externally owned helper channels remain supported for
original configurations without this association. A successor retains the original custody, capsule lineage and helper journal. Test-original collection and Windows archival remain the enclosing
test harness's responsibility. Role guards, custody admission and calendar/time
validation stay in the existing application roles.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time

from src.data.roster_hosted_store import (
    _fsync_dir, _json, _new_file, _read, _require, _safe, file_pin, verify_pin,
)


def _role_arguments(mode, libraries, *, destination=None, controls_pin=None,
                    controller_binding_path=None, configuration_pin=None,
                    custody_request_fd=None, custody_response_fd=None,
                    helper_read_fd=None, helper_write_fd=None, recovery_evidence_pin=None):
    """Fixed packaged roles only; there is no arbitrary executable or clock arg."""
    arguments = [mode]
    _require(mode in {"prepare", "successor"} or recovery_evidence_pin is None,
             "Calendar reconciliation evidence is preparation-only")
    for name in ("seccomp", "libc"):
        item = libraries[name + "_library_pin"]
        arguments.extend(("--" + name + "-library", item["path"],
                          "--" + name + "-library-sha256", item["sha256"]))
    if mode == "provision":
        arguments.extend(("--destination", str(destination), "--controls", controls_pin["path"],
                          "--controls-sha256", controls_pin["sha256"],
                          "--controller-binding-path", str(controller_binding_path)))
    elif mode == "successor":
        arguments.extend(("--destination", str(destination), "--configuration", configuration_pin["path"],
                          "--configuration-sha256", configuration_pin["sha256"],
                          "--controller-binding-path", str(controller_binding_path)))
        if recovery_evidence_pin is not None:
            arguments.extend(("--recovery-evidence", recovery_evidence_pin["path"],
                              "--recovery-evidence-sha256", recovery_evidence_pin["sha256"]))
    else:
        arguments.extend(("--configuration", configuration_pin["path"],
                          "--configuration-sha256", configuration_pin["sha256"],
                          "--custody-request-fd", str(custody_request_fd),
                          "--custody-response-fd", str(custody_response_fd)))
        if helper_read_fd is not None:
            arguments.extend(("--helper-read-fd", str(helper_read_fd),
                              "--helper-write-fd", str(helper_write_fd)))
        if recovery_evidence_pin is not None:
            arguments.extend(("--recovery-evidence", recovery_evidence_pin["path"],
                              "--recovery-evidence-sha256", recovery_evidence_pin["sha256"]))
    return arguments


def _protected_pin(item):
    from src.data.roster_hosted_custody import _trusted
    _trusted(item["path"], directory=False)
    return verify_pin(item)


def _controls(mode, controls_pin, configuration_pin):
    if mode in {"prepare", "startup", "successor"}:
        from src.data.roster_hosted_preservation import NativeServerPreservation
        # Boundary validation does not admit custody or read/replace the anchor.
        preservation = NativeServerPreservation(configuration_pin,
            expected_config_sha256=configuration_pin["sha256"])
        controls_pin = preservation.config["controls"]
    _protected_pin(controls_pin)
    controls = json.loads(_read(controls_pin["path"]))
    _require(controls.get("kind") == "native_root_roster_custody_v1"
             and controls.get("worker_uid") == 10001,
             "Original native custody controls for worker UID10001 required")
    retention = json.loads(_read(_protected_pin(controls["retention_binding"])["path"]))
    return controls, retention


def _disjoint(evidence, controls, retention, destination):
    roots = [controls["root"]["path"], retention["mutable_root"]["path"],
             retention["directories"]["root"]["path"], destination,
             *[row["path"] for row in controls["accepted_roots"]]]
    for value in roots:
        other = _safe(value)
        _require(evidence != other and evidence not in other.parents and other not in evidence.parents,
                 "Launcher evidence must be separate from every provisioned role tree")


def _helper_ends(read_fd, write_fd):
    import fcntl
    _require((read_fd is None) == (write_fd is None), "Both original helper pipe ends required")
    identities = []
    for fd, direction in ((read_fd, os.O_RDONLY), (write_fd, os.O_WRONLY)):
        if fd is None:
            continue
        _require(type(fd) is int and fd > 2, "Preopened helper descriptor required")
        info = os.fstat(fd)
        _require(stat.S_ISFIFO(info.st_mode)
                 and os.readlink("/proc/self/fd/" + str(fd)).startswith("pipe:[")
                 and fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == direction,
                 "Directional anonymous helper pipe required")
        identity = (info.st_dev, info.st_ino)
        _require(identity not in identities, "Distinct original helper pipes required")
        identities.append(identity)


def _runtime_paths(retention):
    from src.data.roster_hosted_checkpoints import native_identity
    mutable = _safe(retention["mutable_root"]["path"])
    _require(native_identity(mutable, directory=True) == retention["mutable_root"]
             and mutable.stat().st_uid == 10001,
             "Original provisioned worker namespace changed")
    runtime = mutable / "launcher-runtime"
    paths = {name: str(runtime / suffix) for name, suffix in (
        ("UFC_DATA_DIR", "data"), ("UFC_PROCESSED_DIR", "processed"),
        ("UFC_MODELS_DIR", "models"), ("UFC_LOGS_DIR", "logs"))}
    for path in (runtime, *map(Path, paths.values())):
        _safe(path)
        if path.exists():
            info = path.lstat()
            _require(stat.S_ISDIR(info.st_mode) and info.st_uid == 10001,
                     "Existing worker runtime directory has different ownership")
    # No root mkdir/chown. Existing src.config initializes these explicit paths
    # only when the guarded startup process imports it after dropping to worker.
    return paths


def _capacity(evidence):
    from src.data.roster_hosted_checkpoints import (
        MAX_ENTRIES, MAX_FILE_BYTES, MAX_TOTAL_BYTES, FREE_RESERVE_BYTES,
    )
    entries = list(evidence.iterdir())
    total = 0
    for path in entries:
        info = path.lstat()
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == 0,
                 "Launcher evidence object changed")
        _require(info.st_size <= MAX_FILE_BYTES, "Launcher evidence file ceiling")
        total += info.st_size
    _require(len(entries) <= MAX_ENTRIES and total <= MAX_TOTAL_BYTES,
             "Launcher evidence capacity exhausted")
    capacity = os.statvfs(evidence)
    _require(capacity.f_bavail * capacity.f_frsize >= FREE_RESERVE_BYTES,
             "Launcher evidence reserve exhausted")


def _group_members(group):
    members = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            # comm may contain spaces and parentheses; fields follow its last ')'.
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group:
                members.append(int(path.name))
        except FileNotFoundError:
            continue
    return members


def _finished(process):
    # Preserve the unreaped original leader until its final group signal. Its
    # reserved PID prevents a recycled, unrelated group becoming cleanup scope.
    if process.returncode is not None:
        return True
    return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def _cleanup(processes):
    """Stop exact child groups even if a leader already exited; never retry work."""
    errors, remaining = [], []
    for process in processes.values():
        if process.returncode is not None:
            continue
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            errors.append(type(exc).__name__ + ": " + str(exc))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            if all(_finished(process) for process in processes.values()):
                break
        except OSError as exc:
            errors.append(type(exc).__name__ + ": " + str(exc))
            break
        time.sleep(.05)
    for process in processes.values():
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                errors.append(type(exc).__name__ + ": " + str(exc))
        try:
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(type(exc).__name__ + ": " + str(exc))
        # Guarded roles cannot create descendants. Any unexpected remaining
        # member makes this invocation fail; the enclosing test subreaper owns
        # adopted descendants and final Windows archival.
        try:
            remaining.extend(_group_members(process.pid))
        except (OSError, ValueError, IndexError) as exc:
            errors.append(type(exc).__name__ + ": " + str(exc))
    return {"quiescent": not errors and not remaining,
            "remaining_group_members": remaining, "errors": errors}


def launch_server(mode, *, evidence_directory, dependencies, libraries,
                  timeout_seconds=300, destination=None, controls_pin=None,
                  controller_binding_path=None, configuration_pin=None,
                  helper_read_fd=None, helper_write_fd=None, helper_journal_root=None,
                  recovery_evidence_pin=None):
    """Run one explicit provision, preparation or startup with finite cleanup.

    Helper descriptors are borrowed, never fabricated. Their trusted owner must
    keep the original helper/current-head authority alive and reap it separately.
    Owned helper mode reserves the original protected journal and cleans up its
    helper process separately alongside custody and worker. The CLI closes its
    own borrowed copies on exit. CPU/memory/output bounds are
    inherited from the separately reviewed local harness, not provider capacity.
    """
    from scripts.run_roster_hosted_server import _guard_library
    from src.data.roster_hosted_custody import _trusted
    from src.data.roster_hosted_preservation import _native

    _require(sys.platform == "linux" and os.geteuid() == 0
             and sys.version_info[:3] == (3, 11, 16), "Trusted native root CPython3.11.16 launcher required")
    _require(mode in {"provision", "successor", "prepare", "startup"}, "Explicit provision, prepare or startup mode required")
    _require(mode in {"prepare", "successor"} or recovery_evidence_pin is None,
             "Calendar reconciliation evidence is preparation-only")
    if recovery_evidence_pin is not None:
        _protected_pin(recovery_evidence_pin)
    _require(type(timeout_seconds) in (int, float) and 0 < timeout_seconds <= 1200,
             "Explicit bounded local role deadline required")
    _require(set(libraries) == {"seccomp_library_pin", "libc_library_pin"}, "Both exact native guard pins required")
    for item in libraries.values():
        _require(_guard_library(item["path"], item["sha256"]) == item, "Exact native guard pin required")
    _helper_ends(helper_read_fd, helper_write_fd)
    _require(helper_journal_root is None or helper_read_fd is None,
             "Owned helper journal and borrowed helper pipes are mutually exclusive")
    if mode == "provision":
        valid = (destination is not None and controls_pin is not None
                 and controller_binding_path is not None and configuration_pin is None)
    elif mode == "successor":
        valid = (destination is not None and controls_pin is None
                 and controller_binding_path is not None and configuration_pin is not None)
    else:
        valid = (destination is None and controls_pin is None
                 and controller_binding_path is None and configuration_pin is not None)
    _require(valid and (mode not in {"provision", "successor"}
             or (helper_read_fd is None and helper_journal_root is None)),
             "Exact mode-specific launch inputs required")
    evidence = Path(_native(evidence_directory, directory=True)["path"])
    _require(not list(evidence.iterdir()), "Unused separately provisioned launcher evidence directory required")
    _require(Path(dependencies).is_absolute(), "Explicit matching trusted runtime dependencies required")
    dependencies = _safe(dependencies)
    _require(dependencies.is_dir(), "Explicit matching trusted runtime dependencies required")
    application = Path(__file__).resolve().parents[2]
    script = application / "scripts/run_roster_hosted_server.py"
    _trusted(script, directory=False)
    controls, retention = _controls(mode, controls_pin, configuration_pin)
    runtime_paths = _runtime_paths(retention)
    native_destination = destination if mode in {"provision", "successor"} else str(Path(configuration_pin["path"]).parent)
    _disjoint(evidence, controls, retention, native_destination)
    helper_binding = None
    if mode in {"prepare", "startup"}:
        from src.data.roster_hosted_helper_binding import NAME, provision_helper_binding, validate_helper_binding
        config = json.loads(_read(configuration_pin["path"]))
        from src.data.roster_hosted_preservation import NativeServerPreservation
        preservation = NativeServerPreservation(configuration_pin,
            expected_config_sha256=configuration_pin["sha256"])
        helper_config = preservation.helper_configuration()
        association = (Path(helper_config["path"]).parent / NAME if helper_config else
                       Path(configuration_pin["path"]).parent / NAME)
        if mode == "prepare":
            recovery_evidence_pin = recovery_evidence_pin or config.get("recovery_evidence")
            preservation.preparation_output(recovery_evidence_pin=recovery_evidence_pin)
            _require(helper_read_fd is None and helper_journal_root is not None,
                     "Initial local saved preparation requires its independently owned helper journal")
        if association.exists():
            _require(helper_journal_root is not None and helper_read_fd is None,
                     "Original owned helper boundary cannot be replaced by borrowed channels")
            helper_binding = validate_helper_binding(helper_config, controls,
                journal_root=helper_journal_root)
            _require(mode == "startup" or config["schema"] == 2,
                     "Original helper preparation already attempted; preserve it")
        elif mode == "startup":
            _require(helper_journal_root is None, "Startup requires the original helper association")
        if helper_journal_root is not None:
            journal = Path(_native(helper_journal_root, directory=True)["path"])
            _disjoint(journal, controls, retention, native_destination)
            _require(journal != evidence and journal not in evidence.parents and evidence not in journal.parents,
                     "Helper journal and launcher diagnostics must be disjoint")
            _require(controls["controller_controls"]["synthetic_responses"] is not None,
                     "Owned saved helper requires the original pinned fixture")
            if mode == "prepare" and helper_binding is None:
                helper_binding = provision_helper_binding(configuration_pin, controls, journal)
    _capacity(evidence)

    def save(name, value):
        _new_file(evidence / name, _json(value), immutable=True)

    processes, streams, owned_fds, commands = {}, [], [], {}
    error = None
    began = time.monotonic()
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(dependencies),
           "PYTHON_DOTENV_DISABLED": "1", "OPENBLAS_NUM_THREADS": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "UFC_REFRESH_ENABLED": "0", "UFC_MONTHLY_CANDIDATE_ENABLED": "0",
           "UFC_UPDATE_CALENDAR_ENABLED": "0", "LIVE_TRADING_MODE": "off", "LIVE_TRADING_ARMED": "0"}
    try:
        save("launch-original.json", {"schema": 1, "mode": mode, "python": sys.executable,
            "dependencies": str(dependencies), "libraries": libraries, "script": file_pin(script),
            "destination": str(destination) if destination is not None else None,
            "controls": controls_pin, "configuration": configuration_pin,
            "controller_binding_path": str(controller_binding_path) if controller_binding_path is not None else None,
            "helper_endpoints_supplied": helper_read_fd is not None, "timeout_seconds": timeout_seconds,
            "owned_helper_binding": helper_binding,
            "recovery_evidence": recovery_evidence_pin,
            "worker_uid": 10001, "activation_authorized": False, "schedule_enabled": False,
            "runtime_paths": runtime_paths,
            "publication_authorized": False, "replay_authorized": False})
        if mode in {"provision", "successor"}:
            roles = [(mode, (), dict(destination=destination, controls_pin=controls_pin,
                controller_binding_path=controller_binding_path, configuration_pin=configuration_pin,
                recovery_evidence_pin=recovery_evidence_pin))]
        else:
            request_read, request_write = os.pipe()
            owned_fds.extend((request_read, request_write))
            response_read, response_write = os.pipe()
            owned_fds.extend((response_read, response_write))
            roles = [("custody", (request_read, response_write), dict(configuration_pin=configuration_pin,
                custody_request_fd=request_read, custody_response_fd=response_write))]
            if helper_journal_root is not None:
                helper_request_read, helper_write_fd = os.pipe()
                owned_fds.extend((helper_request_read, helper_write_fd))
                helper_read_fd, helper_response_write = os.pipe()
                owned_fds.extend((helper_read_fd, helper_response_write))
                roles.append(("helper", (helper_request_read, helper_response_write), {}))
            roles.append((mode, (request_write, response_read) + (() if helper_read_fd is None else
                    (helper_read_fd, helper_write_fd)), dict(configuration_pin=configuration_pin,
                custody_request_fd=request_write, custody_response_fd=response_read,
                helper_read_fd=helper_read_fd, helper_write_fd=helper_write_fd,
                recovery_evidence_pin=recovery_evidence_pin)))
        for role, descriptors, options in roles:
            if role == "helper":
                helper_script = application / "scripts/run_roster_hosted_saved_helper.py"
                _trusted(helper_script, directory=False)
                arguments = ["--controls", config["controls"]["path"],
                    "--controls-sha256", config["controls"]["sha256"],
                    "--journal-root", str(helper_journal_root), "--request-fd", str(descriptors[0]),
                    "--response-fd", str(descriptors[1])]
                for name in ("seccomp", "libc"):
                    item = libraries[name + "_library_pin"]
                    arguments.extend(("--" + name + "-library", item["path"],
                        "--" + name + "-library-sha256", item["sha256"]))
                command = [sys.executable, "-s", "-S", "-P", "-B", str(helper_script), *arguments]
            else:
                command = [sys.executable, "-s", "-S", "-P", "-B", str(script), *_role_arguments(role, libraries, **options)]
            commands[role] = {"argv": command, "inherited_fds": list(descriptors)}
            save(role + "-command.json", commands[role])
            out = (evidence / (role + ".stdout")).open("xb")
            streams.append(out)
            err = (evidence / (role + ".stderr")).open("xb")
            streams.append(err)
            processes[role] = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                close_fds=True, pass_fds=descriptors, start_new_session=True, cwd=application,
                env=dict(env, **runtime_paths) if role in {"prepare", "startup"} else env)
        for fd in owned_fds:
            os.close(fd)
        owned_fds.clear()
        while True:
            _capacity(evidence)
            _require(time.monotonic() - began < timeout_seconds, "Bounded local server role deadline")
            if all(_finished(process) for process in processes.values()):
                break
            # Allow peers to observe EOF and write their exact refusal naturally;
            # this is finite and the common deadline still owns cleanup.
            time.sleep(.05)
    except BaseException as exc:
        error = type(exc).__name__ + ": " + str(exc)
    finally:
        # A second stop signal must not skip the other child or its diagnostics.
        # This finite cleanup is for these exact packaged role groups only.
        finalization_errors, prior_signals = [], {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                prior_signals[signum] = signal.signal(signum, signal.SIG_IGN)
            except (OSError, ValueError) as exc:
                finalization_errors.append(type(exc).__name__ + ": " + str(exc))
        for fd in owned_fds:
            try:
                os.close(fd)
            except OSError as exc:
                finalization_errors.append(type(exc).__name__ + ": " + str(exc))
        try:
            cleanup = _cleanup(processes)
        except BaseException as exc:
            cleanup = {"quiescent": False, "remaining_group_members": None,
                       "errors": [type(exc).__name__ + ": " + str(exc)]}
        diagnostics_errors = []
        for stream in streams:
            for operation in (stream.flush, lambda: os.fsync(stream.fileno()), stream.close):
                try:
                    operation()
                except (OSError, ValueError) as exc:
                    diagnostics_errors.append(type(exc).__name__ + ": " + str(exc))
        try:
            _fsync_dir(evidence)
        except OSError as exc:
            diagnostics_errors.append(type(exc).__name__ + ": " + str(exc))
        finalization_errors.extend(diagnostics_errors)
    try:
        failed = {role: process.returncode for role, process in processes.items() if process.returncode != 0}
        if failed and error is None:
            error = "Server roles refused or failed: " + json.dumps(failed, sort_keys=True)
        try:
            _capacity(evidence)
        except (RuntimeError, OSError, ValueError) as exc:
            finalization_errors.append(type(exc).__name__ + ": " + str(exc))
        result = {"schema": 1,
            "status": "passed" if error is None and cleanup["quiescent"] and not finalization_errors else "failed",
            "mode": mode, "error": error, "cleanup": cleanup, "seconds": time.monotonic() - began,
            "roles": {name: {"pid": process.pid, "exit_code": process.returncode,
                      "stdout": file_pin(evidence / (name + ".stdout")),
                      "stderr": file_pin(evidence / (name + ".stderr"))} for name, process in processes.items()},
            "role_commands": commands, "activation_authorized": False, "publication_authorized": False,
            "schedule_enabled": False, "replay_authorized": False,
            "finalization_errors": finalization_errors, "diagnostics_fsynced": not diagnostics_errors,
            "windows_test_archival_owned_by_enclosing_harness": True}
        save("launch-result.json", result)
        return result
    finally:
        for signum, handler in prior_signals.items():
            signal.signal(signum, handler)
