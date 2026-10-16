"""Derive and issue one disabled roster preparation from server-owned inputs.

No scheduler, source transport, publication, clock override or activation switch
exists here. A future approved service can call this command with its current
immutable input pointer; it does not require manual weekly request URL lists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.roster_hosted_needs import BASE_ROLES, prepare_need_manifest
from src.data.roster_hosted_plan import issue_candidate_plan
from src.data.roster_hosted_store import (
    HostedControlStore, StoreError, _read, _require, file_pin, verify_pin,
)


def _external_pin(path, digest):
    pin = verify_pin(file_pin(path))
    _require(pin["sha256"] == digest, "External recurring input hash mismatch")
    return pin


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--needs-output", required=True, help="Unused evidence directory; failures remain retained")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--base-inputs-manifest", required=True)
    parser.add_argument("--base-inputs-manifest-sha256", required=True)
    parser.add_argument("--recovery-evidence")
    parser.add_argument("--recovery-evidence-sha256")
    args = parser.parse_args(argv)
    try:
        _require(bool(args.recovery_evidence) == bool(args.recovery_evidence_sha256),
                 "Recovery evidence and external hash are required together")
        policy = _external_pin(args.policy, args.policy_sha256)
        manifest_pin = _external_pin(args.base_inputs_manifest, args.base_inputs_manifest_sha256)
        manifest = json.loads(_read(manifest_pin["path"]))
        _require(isinstance(manifest, dict) and set(manifest) == {"schema", "input_pins"}
                 and manifest["schema"] == 1 and isinstance(manifest["input_pins"], dict)
                 and set(manifest["input_pins"]) == BASE_ROLES,
                 "Complete machine base-input manifest required")
        store = HostedControlStore(args.store)
        _require(not store.snapshot()["uncertain_requests"], "Uncertain requests require reconciliation")
        result = prepare_need_manifest(manifest["input_pins"], args.needs_output)
        _require(not result["unknown_semantics"], "Semantic blockers retained; no plan issued")
        inputs = json.loads(_read(result["inputs_manifest"]["path"]))["input_pins"]
        recovery = (_external_pin(args.recovery_evidence, args.recovery_evidence_sha256)
                    if args.recovery_evidence else None)
        issued = issue_candidate_plan(store, args.output_parent, policy,
            expected_policy_sha256=args.policy_sha256, input_pins=inputs,
            proposed_requests=result["proposed_requests"], recovery_evidence_pin=recovery)
        print(json.dumps({"status": "semantic_preparation_execution_disabled", "needs": result,
                          **issued}, sort_keys=True))
        return 0
    except (StoreError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed_execution_disabled", "error": type(exc).__name__ + ": " + str(exc),
                          "source_requests": 0, "activation_authorized": False,
                          "preserve_all_existing_evidence": True}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
