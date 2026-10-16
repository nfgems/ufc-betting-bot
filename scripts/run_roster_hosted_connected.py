"""Explicit disabled local controller using provisioned native custody pipes.

A trusted native launcher must provision fresh immutable starting inputs and
controls, start the independent custodian/helper, then pass ONLY the controller
ends of their anonymous pipes to a non-root controller. This command never
starts those services, provisions custody, enables scheduling, or requests live
sources. The existing native parent guard is installed before application imports.
The inspect mode reports saved accounting and charged uncertainty without a
helper, controller execution, state writes or recovery authorization.
Startup mode inspects and pauses uncertainty or routes the explicitly pinned
original binding through existing retained-controller recovery validation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "recover", "inspect", "startup"))
    for name in ("custody-request-fd", "custody-response-fd"):
        parser.add_argument("--" + name, type=int, required=True)
    for name in ("seccomp-library", "seccomp-library-sha256", "libc-library", "libc-library-sha256"):
        parser.add_argument("--" + name, required=True)
    for name in ("output", "policy", "policy-sha256", "binding", "binding-sha256",
                 "synthetic-responses", "synthetic-responses-sha256",
                 "recovery-evidence", "recovery-evidence-sha256"):
        parser.add_argument("--" + name)
    parser.add_argument("--helper-read-fd", type=int)
    parser.add_argument("--helper-write-fd", type=int)
    args = parser.parse_args(argv)
    from src.data.roster_hosted_store import HostedControlStore, StoreError, _read, _require, _safe, file_pin, verify_pin
    guard = None
    try:
        _require(os.name == "posix" and os.geteuid() != 0, "Provisioned non-root controller required")
        _require((args.recovery_evidence is None) == (args.recovery_evidence_sha256 is None),
                 "Calendar reconciliation evidence and external hash required together")
        _require(args.mode == "prepare" or args.recovery_evidence is None,
                 "Calendar reconciliation evidence is preparation-only")
        def guarded_library(path, digest):
            target = _safe(path)
            for parent in target.parents:
                info = parent.stat()
                _require(info.st_uid == 0 and (not info.st_mode & 0o022 or info.st_mode & stat.S_ISVTX),
                         "Native guard library requires trusted root ancestry")
            _require(target.stat().st_uid == 0, "Native guard library requires independent root owner")
            pin = verify_pin(file_pin(target))
            _require(pin["sha256"] == digest, "External guard library hash mismatch")
            return pin
        def external(path, digest, *, trusted=False):
            _require(path is not None and digest is not None, "Exact path and external hash required")
            if trusted:
                _trusted(path, directory=False)
            pin = verify_pin(file_pin(path))
            _require(pin["sha256"] == digest, "External connected controller input hash mismatch")
            return pin
        from src.data.roster_hosted_wire import install_parent_network_deny
        guard = install_parent_network_deny(
            seccomp_library_pin=guarded_library(args.seccomp_library, args.seccomp_library_sha256),
            libc_library_pin=guarded_library(args.libc_library, args.libc_library_sha256))
        from src.data.roster_hosted_custody import CustodyClient, _trusted
        recovery = (external(args.recovery_evidence, args.recovery_evidence_sha256, trusted=True)
                    if args.recovery_evidence is not None else None)
        if args.mode == "inspect":
            _require(all(value is None for value in (
                args.output, args.policy, args.policy_sha256, args.binding,
                args.binding_sha256, args.synthetic_responses, args.synthetic_responses_sha256,
                args.helper_read_fd, args.helper_write_fd)),
                "Inspection accepts custody pipes only; no work or helper arguments")
        custodian = CustodyClient(args.custody_request_fd, args.custody_response_fd)
        current = custodian.current()
        checkpoint = current["checkpoint"]
        cp = json.loads(_read(verify_pin(checkpoint)["path"]))
        store = HostedControlStore(cp["store_root"], mutation_guard=custodian)
        if args.mode == "inspect":
            from src.data.roster_hosted_inspection import inspect_retained_state
            result = inspect_retained_state(store, custodian=custodian)
            print(json.dumps(dict(result, parent_network_guard=guard), sort_keys=True))
            return 0
        configuration = custodian.controller_configuration()
        fixture = configuration["synthetic_responses"]
        helper = None
        startup_binding = None
        if args.mode == "startup":
            _require(args.binding and args.binding_sha256 and args.output is None
                     and args.policy is None and args.policy_sha256 is None,
                     "Startup requires exact original binding and independently retained current head")
            startup_binding = external(args.binding, args.binding_sha256)
            original = json.loads(_read(verify_pin(startup_binding)["path"]))
            _require(bool(original.get("synthetic_helper_required", False))
                     == (args.helper_read_fd is not None),
                     "Original controller helper boundary must be preserved on startup")
        _require(bool(args.synthetic_responses) == bool(args.synthetic_responses_sha256),
                 "Exact saved fixture and hash required together")
        _require((args.helper_read_fd is None) == (args.helper_write_fd is None),
                 "Both preopened helper pipe ends required")
        if args.synthetic_responses:
            _require(external(args.synthetic_responses, args.synthetic_responses_sha256, trusted=True) == fixture,
                     "Saved fixture differs from independently provisioned control")
        if fixture is not None:
            _trusted(fixture["path"], directory=False)
            verify_pin(fixture)
            if args.mode != "startup":
                _require(args.helper_read_fd is not None, "Connected saved collection requires independent helper")
        if args.helper_read_fd is not None:
            _require(fixture is not None, "Saved-only helper requires pinned response fixture")
            from src.data.roster_hosted_helper import SavedHelperClient
            helper = SavedHelperClient(args.helper_read_fd, args.helper_write_fd,
                store_root=store.root, synthetic_responses_pin=fixture)
        if args.mode == "startup":
            from src.data.roster_hosted_startup import startup_retained_controller
            result = startup_retained_controller(store, startup_binding,
                expected_binding_sha256=args.binding_sha256,
                custodian=custodian, synthetic_helper=helper)
            print(json.dumps(dict(result, parent_network_guard=guard), sort_keys=True))
            return 0
        from src.data.roster_hosted_controller import prepare_controller, recover_retained_controller
        if args.mode == "prepare":
            _require(args.output and bool(args.policy) == bool(args.policy_sha256)
                     and args.binding is None and args.binding_sha256 is None,
                     "Preparation requires output and immutable policy only")
            policy = configuration["policy"]
            _trusted(policy["path"], directory=False)
            verify_pin(policy)
            if args.policy:
                _require(external(args.policy, args.policy_sha256, trusted=True) == policy,
                         "Policy differs from independently provisioned control")
            pointer = cp["accepted_pointer"]
            accepted = json.loads(_read(verify_pin(pointer)["path"]))
            manifest = json.loads(_read(verify_pin(accepted["input_manifest"])["path"]))
            release = manifest["input_pins"]["release"]
            result = prepare_controller(store, pointer, args.output,
                expected_pointer_sha256=pointer["sha256"], expected_release_sha256=release["sha256"],
                checkpoint_pin=checkpoint, expected_checkpoint_sha256=checkpoint["sha256"],
                policy_pin=policy, expected_policy_sha256=policy["sha256"], synthetic_responses_pin=fixture,
                persist_checkpoint=custodian, synthetic_helper=helper, recovery_evidence_pin=recovery)
        else:
            _require(args.binding and args.binding_sha256 and args.output is None
                     and args.policy is None and args.policy_sha256 is None,
                     "Recovery requires exact original binding and independently retained current head")
            result = recover_retained_controller(store,
                external(args.binding, args.binding_sha256), expected_binding_sha256=args.binding_sha256,
                custodian=custodian, synthetic_helper=helper)
        print(json.dumps(dict(result, parent_network_guard=guard), sort_keys=True))
        return 0
    except (RuntimeError, OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "failed_execution_disabled", "error": type(exc).__name__ + ": " + str(exc),
            "source_requests": 0, "activation_authorized": False, "publication_authorized": False,
            "effective_uid": os.geteuid(), "parent_network_guard": guard,
            "replay_authorized": False, "schedule_enabled": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
