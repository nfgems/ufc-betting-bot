"""Validate separately supplied reviewed context pins and propose configuration.

This command never fetches sources, invents identity/scope/collection decisions,
installs files, changes process/hosted environment, or starts a collector.
"""
import argparse
import json
from pathlib import Path

from src.data.observation_collection import (
    CONTEXT_PREFIXES, observation_context_preflight, observation_collection_preflight,
    validate_observation_context,
)


def proposed_configuration(contexts):
    """Require both explicit outer hashes; the input file cannot approve itself."""
    if set(contexts) != set(CONTEXT_PREFIXES):
        raise ValueError("Both reviewed source contexts are required")
    validated = {}
    variables = {}
    for source, prefix in CONTEXT_PREFIXES.items():
        value = contexts[source]
        if set(value) != {"path", "sha256"}:
            raise ValueError("Each reviewed context needs an explicit path and independently supplied SHA256")
        pin = validate_observation_context(source, value["path"], value["sha256"])
        validated[source] = pin
        variables[prefix + "_REVIEWED_BINDINGS_PATH"] = pin["path"]
        variables[prefix + "_REVIEWED_BINDINGS_SHA256"] = value["sha256"]
    return {"record_kind": "validated_existing_context_local_configuration_proposal",
        "contexts": validated, "validated_local_settings": variables,
        "path_scope": "local_validation_only_not_hosted_environment", "deployment_ready": False,
        "new_identity_or_collection_authority_created": False,
        "installed_or_configured": False, "source_requests_made": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("inspect-configured")
    coverage = actions.add_parser("inspect-method-coverage")
    coverage.add_argument("--report", required=True)
    coverage.add_argument("--report-sha256", required=True)
    admission = actions.add_parser("inspect-collection")
    admission.add_argument("--inventory", required=True)
    admission.add_argument("--inventory-sha256", required=True)
    admission.add_argument("--inventory-observed-at", required=True)
    admission.add_argument("--inventory-not-after", required=True)
    prepare = actions.add_parser("prepare")
    for source in CONTEXT_PREFIXES:
        prepare.add_argument("--" + source + "-context", required=True)
        prepare.add_argument("--" + source + "-sha256", required=True)
    prepare.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "inspect-method-coverage":
        from src.data.method_inventory_coverage import inspect_inventory_coverage
        result = inspect_inventory_coverage(report_path=args.report, report_sha256=args.report_sha256)
        print(json.dumps(result, indent=2))
        # A complete accounting report never replaces collection admission.
        return 2
    if args.action == "inspect-collection":
        result = observation_collection_preflight(inventory_path=args.inventory,
            inventory_sha256=args.inventory_sha256, observed_at=args.inventory_observed_at,
            not_after=args.inventory_not_after)
        print(json.dumps(result, indent=2))
        return 0 if result["ready"] else 2
    if args.action == "inspect-configured":
        result = observation_context_preflight()
        print(json.dumps(result, indent=2))
        return 0 if result["ready"] else 2
    result = proposed_configuration({source: {"path": getattr(args, source + "_context"), "sha256": getattr(args, source + "_sha256")} for source in CONTEXT_PREFIXES})
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps({"validated": True, "proposal": args.output, "installed_or_configured": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
