"""Prepare one disabled server roster plan from pinned machine-saved inputs.

This command never fetches sources, executes a worker, publishes or schedules.
There is no activation switch or clock override. The server supplies the actual
clock, durable state and a pinned input manifest; request proposals are read only
from that manifest's saved need evidence rather than a per-run manual URL list.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.roster_hosted_plan import issue_candidate_plan
from src.data.roster_hosted_store import (
    HostedControlStore, StoreError, _read, _require, file_pin, verify_pin,
)


def _external_pin(path, expected_sha256):
    pin = verify_pin(file_pin(path))
    _require(pin["sha256"] == expected_sha256, "External preparation input hash mismatch")
    return pin


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, help="Existing reconciled durable state directory")
    parser.add_argument("--output-parent", required=True, help="Existing immutable-plan parent directory")
    parser.add_argument("--policy", required=True, help="Exact disabled policy JSON file")
    parser.add_argument("--policy-sha256", required=True, help="Externally bound policy hash")
    parser.add_argument("--inputs-manifest", required=True, help="Machine-saved complete input-pins manifest")
    parser.add_argument("--inputs-manifest-sha256", required=True, help="Externally bound input manifest hash")
    parser.add_argument("--recovery-evidence", help="Pinned previous Tuesday reconciliation, Wednesday only")
    parser.add_argument("--recovery-evidence-sha256", help="Externally bound recovery evidence hash")
    args = parser.parse_args(argv)
    try:
        _require(bool(args.recovery_evidence) == bool(args.recovery_evidence_sha256),
                 "Recovery path and external hash must be supplied together")
        policy = _external_pin(args.policy, args.policy_sha256)
        manifest_pin = _external_pin(args.inputs_manifest, args.inputs_manifest_sha256)
        manifest = json.loads(_read(manifest_pin["path"]))
        _require(isinstance(manifest, dict) and set(manifest) == {"schema", "input_pins"}
                 and manifest["schema"] == 1, "Complete machine input manifest required")
        inputs = manifest["input_pins"]
        _require(isinstance(inputs, dict) and "need_evidence" in inputs, "Saved need evidence is required")
        verify_pin(inputs["need_evidence"])
        need = json.loads(_read(inputs["need_evidence"]["path"]))
        _require(isinstance(need, dict) and isinstance(need.get("proposed_requests"), list),
                 "Machine saved request proposals required")
        recovery = (_external_pin(args.recovery_evidence, args.recovery_evidence_sha256)
                    if args.recovery_evidence else None)
        result = issue_candidate_plan(HostedControlStore(args.store), args.output_parent, policy,
            expected_policy_sha256=args.policy_sha256, input_pins=inputs,
            proposed_requests=need["proposed_requests"], recovery_evidence_pin=recovery)
        print(json.dumps({"status": "prepared_execution_disabled", "inputs_manifest": manifest_pin, **result},
                         sort_keys=True))
        return 0
    except (StoreError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed_execution_disabled", "error": type(exc).__name__ + ": " + str(exc),
                          "source_requests": 0, "activation_authorized": False,
                          "preserve_all_existing_evidence": True}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
