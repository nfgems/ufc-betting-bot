"""Prepare a disabled roster cycle from externally pinned server-owned inputs.

No schedule, live transport, publication or clock override is provided. The
mandatory native parent guard refuses sockets before application imports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.roster_hosted_store import (
    HostedControlStore, StoreError, _json, _new_file, _read, _require, _safe, file_pin, verify_pin,
)


def external(path, digest):
    pin = verify_pin(file_pin(path))
    _require(pin["sha256"] == digest, "External controller input hash mismatch")
    return pin


from src.data.roster_hosted_preservation import persistent_checkpointer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "output", "accepted-pointer", "accepted-pointer-sha256",
                 "release-sha256", "checkpoint", "checkpoint-sha256", "policy", "policy-sha256",
                 "seccomp-library", "seccomp-library-sha256", "libc-library", "libc-library-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--synthetic-responses")
    parser.add_argument("--synthetic-responses-sha256")
    parser.add_argument("--persistent-checkpoints")
    parser.add_argument("--recovery-evidence")
    parser.add_argument("--recovery-evidence-sha256")
    args = parser.parse_args(argv)
    try:
        _require(bool(args.synthetic_responses) == bool(args.synthetic_responses_sha256),
                 "Saved response fixture and external hash required together")
        _require(bool(args.recovery_evidence) == bool(args.recovery_evidence_sha256),
                 "Calendar reconciliation evidence and external hash required together")
        _require(not args.synthetic_responses or args.persistent_checkpoints,
                 "Synthetic collection needs persistent preclaim evidence")
        from src.data.roster_hosted_wire import install_parent_network_deny
        guard = install_parent_network_deny(
            seccomp_library_pin=external(args.seccomp_library, args.seccomp_library_sha256),
            libc_library_pin=external(args.libc_library, args.libc_library_sha256))
        from src.data.roster_hosted_controller import prepare_controller
        result = prepare_controller(HostedControlStore(args.store),
            external(args.accepted_pointer, args.accepted_pointer_sha256), args.output,
            expected_pointer_sha256=args.accepted_pointer_sha256, expected_release_sha256=args.release_sha256,
            checkpoint_pin=external(args.checkpoint, args.checkpoint_sha256),
            expected_checkpoint_sha256=args.checkpoint_sha256,
            policy_pin=external(args.policy, args.policy_sha256), expected_policy_sha256=args.policy_sha256,
            synthetic_responses_pin=external(args.synthetic_responses, args.synthetic_responses_sha256)
                if args.synthetic_responses else None,
            persist_checkpoint=persistent_checkpointer(args.persistent_checkpoints)
                if args.persistent_checkpoints else None,
            recovery_evidence_pin=external(args.recovery_evidence, args.recovery_evidence_sha256)
                if args.recovery_evidence else None)
        print(json.dumps(dict(result, parent_network_guard=guard), sort_keys=True))
        return 0
    except (StoreError, OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "failed_execution_disabled", "error": type(exc).__name__ + ": " + str(exc),
            "source_requests": 0, "activation_authorized": False, "publication_authorized": False,
            "replay_authorized": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
