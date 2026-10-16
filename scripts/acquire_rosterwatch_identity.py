"""Acquire reviewed UFCStats evidence and replay ordinary pending identities.

The explicit CLI consumes a sealed one-use plan. It is not a scheduler and has
no default HTTP activity. Its output is an unpublished registry and a portable
catalog consumable by the ordinary Roster Watch candidate adapter. A supervising
launcher must install write/network containment before importing project code.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import stat


def _pin(path: Path) -> dict:
    body = path.read_bytes()
    return {"path": path.as_posix(), "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest()}


def _immutable_json(path: Path, value: object) -> dict:
    body = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(stat.S_IREAD)
    result = _pin(path)
    if result["sha256"] != hashlib.sha256(body).hexdigest():
        raise RuntimeError("Persisted matching artifact differs")
    return result


def _workflow(plan: dict) -> dict:
    item = plan.get("workflow_input")
    if not isinstance(item, dict) or item not in plan.get("evidence_pins", []):
        raise ValueError("Workflow input must be an exact reviewed evidence pin")
    path = Path(item["path"])
    if _pin(path) != {key: item[key] for key in ("path", "bytes", "sha256")}:
        raise ValueError("Workflow input pin drift")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Unsupported acquisition workflow schema")
    registry, captures = value.get("owner_registry"), value.get("captures")
    if not isinstance(registry, list) or not registry or not all(isinstance(row, dict) for row in registry):
        raise ValueError("Workflow requires the complete candidate owner registry")
    owners = [row.get("internal_owner_id") for row in registry]
    if not all(isinstance(owner, str) and owner for owner in owners) or len(set(owners)) != len(owners):
        raise ValueError("Workflow registry owner IDs are missing or duplicated")
    if not isinstance(captures, list) or not all(isinstance(row, dict) for row in captures):
        raise ValueError("Workflow requires embedded captured evidence")
    for capture in captures:
        if not any(key in capture for key in ("body_text", "body_base64")) or any(
            key in capture for key in ("capture", "decoded_path", "raw_path", "path")
        ):
            raise ValueError("Workflow captures must contain portable embedded bytes")
    for key in ("source_contexts", "confirmed_profiles", "occupied_carriers"):
        if not isinstance(value.get(key, {}), dict):
            raise ValueError("Invalid workflow context mapping: " + key)
    cap = value.get("request_cap")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
        raise ValueError("Workflow captured-source check cap must be a nonnegative integer")
    if not isinstance(value.get("checked_at"), str):
        raise ValueError("Workflow requires its reviewed zoned matching timestamp")
    return value


def run_acquisition_matching(plan_path, review_path, *, wire=None, verifier=None) -> dict:
    """Perform bounded acquisition and ordinary deterministic recheck together.

    ``wire`` is a synthetic testing seam; the command line always uses the
    counted real transport. ``verifier`` supplies the supervisor's additional
    preservation/containment checks. Every source send is separately accounted
    for; the existing captured provider still reports zero network requests.
    Input records (including historical owners) are never modified in place.
    """
    from src.data.ufcstats_identity_access import acquire_reviewed, verify_authority

    plan = verify_authority(plan_path, review_path)
    workflow = _workflow(plan)
    # These imports occur only after authority/pins have passed; the caller's
    # pre-import guard and isolated environment must already be active.
    from src.data.rosterwatch_active_roster import recheck_rosterwatch_identities
    from src.data.rosterwatch_identity_matching import CapturedIdentityProvider

    def recheck(registry, captures):
        return recheck_rosterwatch_identities(
            registry, identity_provider=CapturedIdentityProvider(captures),
            source_contexts=workflow.get("source_contexts", {}),
            confirmed_profiles=workflow.get("confirmed_profiles", {}),
            occupied_carriers=workflow.get("occupied_carriers", {}),
            request_cap=workflow["request_cap"], checked_at=workflow["checked_at"],
        )

    discovery = recheck(copy.deepcopy(workflow["owner_registry"]), workflow["captures"])
    if discovery["errors"]:
        raise ValueError("Offline discovery failed; no acquisition is permitted")
    needs = discovery.get("requested_urls", [])
    planned = [request["url"] for request in plan["requests"]]
    if not set(planned).issubset(needs):
        raise ValueError("Reviewed request is not an evidenced unmet identity need")
    acquisition = acquire_reviewed(
        plan_path, review_path, needs, wire=wire, verifier=verifier,
    )
    # Even a stopped acquisition retains rejected captures and safe pending
    # decisions. Transport failure is not relabeled as acquisition success.
    catalog = {"schema_version": 1, "captures": [
        *workflow["captures"], *acquisition["captures"],
    ]}
    registry = copy.deepcopy(workflow["owner_registry"])
    matching = recheck(registry, catalog["captures"])
    if matching["network_requests"] != 0:
        raise RuntimeError("Captured matching violated its offline contract")
    if _workflow(plan) != workflow:
        raise RuntimeError("Workflow input changed during acquisition/matching")
    checked = set(matching["checked_members"])
    outcomes = []
    for row in registry:
        if row["internal_owner_id"] not in checked:
            continue
        decision = json.loads(row.get("identity_decision_json") or "{}")
        outcomes.append({
            "internal_owner_id": row["internal_owner_id"],
            "fighter": row.get("current_source_name") or row.get("rw_fighter_raw") or "",
            "status": decision.get("status", "pending"),
            "reason_codes": decision.get("reason_codes", []),
            "carrier_type": decision.get("carrier_type", ""),
            "carrier_value": decision.get("carrier_value", ""),
            "matching_basis": decision.get("matching_basis", ""),
            "decision_sha256": decision.get("decision_sha256", ""),
        })
    output = Path(plan["evidence_root"]) / "matching"
    if output.is_symlink() or output.exists():
        raise ValueError("Matching output must be fresh")
    output.mkdir()
    catalog_pin = _immutable_json(output / "identity-catalog.json", catalog)
    registry_pin = _immutable_json(output / "owner-registry.json", {
        "schema_version": 1, "owner_registry": registry,
    })
    report_pin = _immutable_json(output / "matching-report.json", {
        "schema_version": 1, "publication_requested": False,
        "status": "passed" if acquisition["status"] == "passed" and not matching["errors"] else "failed",
        "acquisition_status": acquisition["status"],
        "acquisition_new_sends": acquisition["new_sends"],
        "acquisition_cumulative_sends": acquisition["cumulative_sends"],
        "matching": matching, "outcomes": outcomes,
        "deferred_requested_urls": sorted(set(needs) - set(planned)),
        "catalog": catalog_pin, "owner_registry": registry_pin,
        "workflow_input": plan["workflow_input"],
    })
    return {"status": "passed" if acquisition["status"] == "passed" and not matching["errors"] else "failed",
            "acquisition": acquisition, "catalog": catalog_pin,
            "owner_registry": registry_pin, "matching_report": report_pin}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--review", required=True)
    args = parser.parse_args(argv)
    # A real CLI invocation requires the reviewed supervisor, which provides
    # the additional per-send checks. There is no direct unguarded wire route.
    import sys
    supervisor = sys.modules.get("ufcstats_access_supervisor")
    verifier = getattr(supervisor, "verify_before_send", None)
    if not callable(verifier):
        parser.error("Run through the reviewed source-access supervisor")
    result = run_acquisition_matching(args.plan, args.review, verifier=verifier)
    acquisition = result["acquisition"]
    print(json.dumps({"status": result["status"], "acquisition_status": acquisition["status"], "new_sends": acquisition["new_sends"],
                      "cumulative_sends": acquisition["cumulative_sends"],
                      "matching_report": result["matching_report"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
