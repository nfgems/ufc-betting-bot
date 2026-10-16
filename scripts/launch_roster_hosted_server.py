"""Explicit LOCAL trusted parent for disabled provision, prepare and startup.

Supply already-provisioned native role trees and the exact original control
pins. Prepare requires an unused protected helper journal; startup preserves
that original journal. Older helper-bound configurations use both preopened
ends from their independently provisioned saved helper. This command does not
create initial accounting/accepted authority, replace helper custody or alter original calendar/lifetime rules. Run only in the separately
reviewed and bounded local runtime; it is not an installed production entrypoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("provision", "successor", "prepare", "startup"))
    for name in ("evidence-directory", "dependencies", "seccomp-library", "seccomp-library-sha256",
                 "libc-library", "libc-library-sha256"):
        parser.add_argument("--" + name, required=True)
    for name in ("destination", "controls", "controls-sha256", "controller-binding-path",
                 "configuration", "configuration-sha256", "recovery-evidence", "recovery-evidence-sha256"):
        parser.add_argument("--" + name)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--helper-read-fd", type=int)
    parser.add_argument("--helper-write-fd", type=int)
    parser.add_argument("--helper-journal-root")
    args = parser.parse_args(argv)
    from scripts.run_roster_hosted_server import _external, _pipe_ends
    from src.data.roster_hosted_launcher import launch_server
    from src.data.roster_hosted_store import _require

    def interrupted(signum, frame):
        raise InterruptedError("Local launcher interrupted by signal " + str(signum))

    prior_handlers = {}
    try:
        _require((args.recovery_evidence is None) == (args.recovery_evidence_sha256 is None),
                 "Calendar reconciliation evidence and external hash required together")
        _require(args.mode in {"prepare", "successor"} or args.recovery_evidence is None,
                 "Calendar reconciliation evidence is preparation-only")
        _require((args.helper_read_fd is None) == (args.helper_write_fd is None), "Both original helper endpoints required")
        _pipe_ends([] if args.helper_read_fd is None else
                   [(args.helper_read_fd, os.O_RDONLY), (args.helper_write_fd, os.O_WRONLY)])
        _require((args.controls is None) == (args.controls_sha256 is None)
                 and (args.configuration is None) == (args.configuration_sha256 is None),
                 "Exact external path and hash pairs required")
        for signum in (signal.SIGINT, signal.SIGTERM):
            prior_handlers[signum] = signal.signal(signum, interrupted)
        result = launch_server(args.mode, evidence_directory=args.evidence_directory,
            dependencies=args.dependencies, timeout_seconds=args.timeout_seconds,
            libraries={name + "_library_pin": _external(getattr(args, name + "_library"),
                getattr(args, name + "_library_sha256")) for name in ("seccomp", "libc")},
            destination=args.destination,
            controls_pin=_external(args.controls, args.controls_sha256) if args.controls else None,
            controller_binding_path=args.controller_binding_path,
            configuration_pin=_external(args.configuration, args.configuration_sha256) if args.configuration else None,
            helper_read_fd=args.helper_read_fd, helper_write_fd=args.helper_write_fd,
            helper_journal_root=args.helper_journal_root,
            recovery_evidence_pin=_external(args.recovery_evidence, args.recovery_evidence_sha256)
                if args.recovery_evidence is not None else None)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "passed" else 1
    except (RuntimeError, OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "failed_execution_disabled", "error": type(exc).__name__ + ": " + str(exc),
            "activation_authorized": False, "publication_authorized": False,
            "schedule_enabled": False, "replay_authorized": False}), file=sys.stderr)
        return 1
    finally:
        for signum, handler in prior_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
