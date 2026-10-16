"""Explicit independent saved-response role for the disabled local launcher.

The launcher supplies original protected custody controls, a separately
provisioned native root-owned journal and exactly two anonymous pipe endpoints.
Each response uses the existing independent custody journal to confirm the
current charge. This role creates no authority, child process, socket or retry;
its enclosing launcher owns finite waiting, cleanup and original journal binding.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_roster_hosted_server import _external, _guard_library, _pipe_ends


def _open_helper(controls_pin, journal_root):
    from src.data.roster_hosted_custody import NativeCustodyAuthority, _trusted
    from src.data.roster_hosted_preservation import _native
    from src.data.roster_hosted_store import HostedControlStore, _read, _require, _safe, verify_pin

    authority = NativeCustodyAuthority(controls_pin,
        expected_controls_sha256=controls_pin["sha256"])
    controls = authority.controls
    _require(controls["worker_uid"] == 10001, "Original worker UID10001 controls required")
    retention_pin = controls["retention_binding"]
    _trusted(retention_pin["path"], directory=False)
    retention = json.loads(_read(verify_pin(retention_pin)["path"]))
    journal = Path(_native(journal_root, directory=True)["path"])
    others = [controls["root"]["path"], retention["mutable_root"]["path"],
              retention["directories"]["root"]["path"],
              *[item["path"] for item in controls["accepted_roots"]]]
    for value in others:
        other = _safe(value)
        _require(journal != other and journal not in other.parents and other not in journal.parents,
                 "Helper journal must be separate from every custody role tree")
    # These original named pins are immutable. Reading them does not compete
    # with the custody process's transient mutation lock or admit recovery.
    named = controls["controller_controls"]
    _require(isinstance(named, dict) and set(named) == {"policy", "synthetic_responses"}
             and named["policy"] is not None and named["synthetic_responses"] is not None,
             "Independently provisioned saved response controls required")
    for item in named.values():
        _trusted(item["path"], directory=False)
        verify_pin(item)
    from src.data.roster_hosted_helper import SavedResponseAuthority

    # check() reads and replays the root-owned custody journal under its own
    # lock for EVERY charged request, then compares the exact native store head.
    # A checkpoint or worker-provided head is never its own acknowledgment.
    return SavedResponseAuthority(HostedControlStore(retention["store_root"]),
        named["synthetic_responses"], journal, controller_uid=10001,
        confirmed_head=authority.check)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("controls", "controls-sha256", "journal-root", "seccomp-library",
                 "seccomp-library-sha256", "libc-library", "libc-library-sha256"):
        parser.add_argument("--" + name, required=True)
    for name in ("request-fd", "response-fd"):
        parser.add_argument("--" + name, type=int, required=True)
    args = parser.parse_args(argv)
    from src.data.roster_hosted_store import _require

    guard = None
    try:
        _require(sys.platform == "linux" and os.geteuid() == 0
                 and sys.version_info[:3] == (3, 11, 16),
                 "Independent native root CPython3.11.16 saved helper required")
        _pipe_ends([(args.request_fd, os.O_RDONLY), (args.response_fd, os.O_WRONLY)])
        from src.data.roster_hosted_wire import install_parent_network_deny

        guard = install_parent_network_deny(
            seccomp_library_pin=_guard_library(args.seccomp_library, args.seccomp_library_sha256),
            libc_library_pin=_guard_library(args.libc_library, args.libc_library_sha256))
        authority = _open_helper(_external(args.controls, args.controls_sha256), args.journal_root)
        from src.data.roster_hosted_helper import serve_saved_helper

        result = serve_saved_helper(authority, args.request_fd, args.response_fd)
        print(json.dumps(dict(result, status="saved_helper_exited_execution_disabled",
            effective_uid=os.geteuid(), parent_network_guard=guard,
            activation_authorized=False, publication_authorized=False,
            replay_authorized=False, schedule_enabled=False), sort_keys=True))
        return 0
    except (RuntimeError, OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "failed_execution_disabled",
            "error": type(exc).__name__ + ": " + str(exc), "source_requests": 0,
            "server_role": "saved_helper", "effective_uid": os.geteuid(),
            "parent_network_guard": guard, "activation_authorized": False,
            "publication_authorized": False, "replay_authorized": False,
            "schedule_enabled": False}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
