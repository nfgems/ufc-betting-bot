"""Contained synthetic collection to ordinary candidate workflow.

The parent chooses every request from sealed authority. This worker only asks
the bound mailbox to collect, persists its saved evidence, and runs the explicitly
publication-disabled ordinary candidate driver. Invoke through the reviewed Linux
supervisor; there is no live-source, scheduling or publication mode.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path


TEMPLATE_KEYS = frozenset({
    "schema_version", "publication_requested", "network_requests", "evidence_mode",
    "acquisition_plan_id", "seed_root", "seed_inventory", "max_membership_age_seconds",
    "max_identity_age_seconds", "identity_check_cap", "run_started_at", "expires_at",
})


def run_worker(template_path, template_sha256, source_plan_sha256):
    from src.data.roster_hosted_candidate import CandidateError, file_pin, prepare_candidate
    from src.data.roster_hosted_mailbox import MailboxClient

    template_path = Path(template_path)
    if file_pin(template_path)["sha256"] != template_sha256:
        raise CandidateError("workflow template hash mismatch")
    template = json.loads(template_path.read_bytes())
    if (set(template) not in (TEMPLATE_KEYS, TEMPLATE_KEYS | {"identity_evidence_storage"})
            or template.get("schema_version") != 1
            or template.get("publication_requested") is not False
            or type(template.get("network_requests")) is not int
            or template["network_requests"] != 0):
        raise CandidateError("workflow requires exact publication-disabled template")
    if template.get("identity_evidence_storage", "legacy_inline") not in ("legacy_inline", "shared_v1"):
        raise CandidateError("unknown identity evidence storage version")
    work = Path.cwd().resolve()
    if (work / "evidence").exists() or (work / "prepared").exists():
        raise CandidateError("workflow output already exists; never resume")
    client = MailboxClient(work / "broker-mailbox", source_plan_sha256)
    collected = client.collect()
    if (collected.get("status") != "passed" or collected.get("execution_mode") != "synthetic"
            or collected.get("source_requests") != 0
            or collected.get("publication_requested") is not False):
        raise CandidateError("workflow accepts completed synthetic collection only")
    membership = collected.get("membership")
    if not isinstance(membership, dict):
        raise CandidateError("workflow requires complete membership evidence")
    envelope = membership.get("envelope", {})
    if (envelope.get("plan_id") != template["acquisition_plan_id"]
            or envelope.get("plan", {}).get("sha256") != source_plan_sha256):
        raise CandidateError("collected membership is not bound to this source plan")
    raw = base64.b64decode(membership["body_base64"], validate=True)
    if (hashlib.sha256(raw).hexdigest() != membership.get("raw", {}).get("sha256")
            or len(raw) != membership.get("raw", {}).get("bytes")):
        raise CandidateError("membership bytes disagree with durable store receipt")
    evidence = work / "evidence"
    evidence.mkdir()
    def save(name, content):
        return client.immutable(evidence / name, content)
    def save_json(name, payload):
        return save(name, (json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode())
    body_pin = save("membership.bin", raw)
    envelope_pin = save_json("membership-envelope.json", envelope)
    catalog_pin = save_json("identity-catalog.json", collected["catalog"])
    # Keep the complete broker operation/receipt record. Raw membership bytes are
    # already saved separately and hash-bound; identity bodies stay embedded.
    receipt = dict(collected, membership={key: value for key, value in membership.items()
                                        if key != "body_base64"})
    receipt_pin = save_json("collection-receipt.json", receipt)
    specification = dict(template, output_root=str(work / "prepared"),
                         membership_body=body_pin, membership_envelope=envelope_pin,
                         identity_catalog=catalog_pin)
    spec_pin = save_json("candidate-spec.json", specification)
    report = prepare_candidate(spec_pin["path"], spec_pin["sha256"])
    summary = {"schema_version": 1, "status": report["status"], "execution_mode": "synthetic",
               "source_requests": 0, "synthetic_charges": collected["synthetic_charges"],
               "publication_performed": False, "current_source_claim": False,
               "template": file_pin(template_path), "candidate_spec": spec_pin,
               "collection_receipt": receipt_pin,
               "candidate_terminal": file_pin(work / "prepared/candidate-terminal.json"),
               "current_members": report["current_members"], "total_owners": report["total_owners"],
               "historical_owners": report["historical_owners"], "pending_members": report["pending_members"]}
    save_json("workflow-terminal.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--template-sha256", required=True)
    parser.add_argument("--source-plan-sha256", required=True)
    args = parser.parse_args()
    result = run_worker(args.template, args.template_sha256, args.source_plan_sha256)
    print(json.dumps({key: result[key] for key in (
        "status", "execution_mode", "source_requests", "synthetic_charges",
        "publication_performed", "current_members", "total_owners")}, sort_keys=True))


if __name__ == "__main__":
    main()
