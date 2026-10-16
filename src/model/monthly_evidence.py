"""Pure monthly evaluator protocol checks, without executing an evaluator.

Validated JSON describes evidence and its bindings. It is not proof that an
evaluator ran: genuine report production and independent artifact acceptance
remain responsibilities of the separately authorized evaluator workflow.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
from typing import Any, Mapping


EVALUATOR_CONTRACT = "monthly_evaluator_evidence_v1"
REQUEST_KIND = "monthly_evaluator_request_v1"
PRODUCER_REQUEST_KIND = "monthly_evaluator_request_v2"
REPORT_KIND = "monthly_evaluator_report_v1"
REQUIRED_CHECKS = (
    "fixed_evaluation", "production_load", "exact_parity", "prefight_parity",
    "coverage_report", "current_live_comparison",
)
MODEL_FIELDS = frozenset({"primary", "logistic", "no_odds", "spec"})


class MonthlyEvidenceRefusal(ValueError):
    """An evaluator request/report is missing or contradicts its original input."""


def _require(value: bool, message: str) -> None:
    if not value:
        raise MonthlyEvidenceRefusal(message)


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MonthlyEvidenceRefusal("evaluator evidence must be finite JSON") from exc


def evidence_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def parse_evaluator_json(body: str | bytes) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate evaluator JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise MonthlyEvidenceRefusal("nonfinite evaluator JSON number")

    try:
        value = json.loads(body, object_pairs_hook=unique, parse_constant=nonfinite)
    except (ValueError, UnicodeError) as exc:
        raise MonthlyEvidenceRefusal("invalid evaluator JSON") from exc
    _require(isinstance(value, dict), "evaluator JSON must be an object")
    return value


def _hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _models(value: Any) -> None:
    _require(isinstance(value, dict) and set(value) == MODEL_FIELDS
             and all(_hash(item) for item in value.values()),
             "exact evaluator model identities are required")


def _absolute_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    normalized = value.replace("\\", "/")
    return (normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized) is not None) \
        and ".." not in normalized.split("/")


def _instant(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise MonthlyEvidenceRefusal("invalid evaluator cutoff") from exc
    _require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
             "evaluator cutoff must be timezone aware")
    return parsed.astimezone(timezone.utc)


def _bindings(input_receipt: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(input_receipt, dict) and isinstance(contract, dict),
             "evaluator input and contract must be objects")
    _require(input_receipt.get("contract") == contract,
             "evaluator input contract conflict")
    generation = input_receipt.get("accepted_generation_id")
    _require(isinstance(generation, str) and bool(generation.strip()),
             "evaluator accepted generation missing")
    cutoff = input_receipt.get("weekly_update_completed_at")
    _instant(cutoff)
    _require(_hash(input_receipt.get("source_sha256"))
             and _hash(input_receipt.get("derived_sha256")), "evaluator input hashes missing")
    policy = contract.get("monthly_policy")
    transport = input_receipt.get("monthly_input")
    if policy is not None:
        _require(isinstance(policy, dict) and isinstance(transport, dict),
                 "successor evaluator needs saved policy/input")
        policy_fields = {"policy_id", "policy_version", "policy", "adapter", "training_spec",
                         "named_training_spec", "modeling_start_date", "train_cutoff_date",
                         "input_completion_cutoff_utc", "adoption_id", "model_id", "promotion_month_utc",
                         "evaluator_contract", "evaluation_windows"}
        _require(set(policy) == policy_fields, "evaluator policy bindings incomplete or unknown")
        for field in ("policy_id", "adoption_id", "model_id"):
            _require(isinstance(policy[field], str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", policy[field])
                     is not None, "evaluator original policy/adoption identity missing")
        policy_pin = policy["policy"]
        _require(isinstance(policy_pin, dict) and set(policy_pin) == {"path", "bytes", "sha256"}
                 and _absolute_path(policy_pin["path"])
                 and type(policy_pin["bytes"]) is int and 0 < policy_pin["bytes"] <= 128 * 1024
                 and _hash(policy_pin["sha256"]), "evaluator original policy pin missing")
        hash_fields = {"acceptance_sha256", "inventory_sha256", "receipt_sha256", "publication_sha256",
                       "completion_sha256", "features_sha256", "prepared_acceptance_sha256"}
        _require(set(transport) == {"schema", "kind", "root", "context", "training_spec", "policy_binding"}
                 | hash_fields and type(transport["schema"]) is int and transport["schema"] == 2
                 and transport["kind"] == "saved_monthly_input_transport_v2"
                 and _absolute_path(transport["root"])
                 and all(_hash(transport[field]) for field in hash_fields),
                 "evaluator saved original input identities missing")
        _require(transport["features_sha256"] == input_receipt["source_sha256"],
                 "evaluator original feature identity conflicts")
        _require(policy.get("evaluator_contract") == EVALUATOR_CONTRACT,
                 "unsupported evaluator contract")
        _require(type(policy.get("policy_version")) is int and policy["policy_version"] == 1
                 and policy.get("adapter") == "durability_monthly_refit_v1",
                 "unsupported evaluator monthly policy version")
        contract_fields = {"name", "primary_feature_count", "logistic_feature_count", "no_odds_feature_count",
                           "primary_feature_sha256", "logistic_feature_sha256", "no_odds_feature_sha256",
                           "modeling_start_date", "unsupported_contract_start_date", "monthly_policy"}
        _require(set(contract) == contract_fields
                 and all(type(contract[key]) is int and contract[key] > 0
                         for key in ("primary_feature_count", "logistic_feature_count", "no_odds_feature_count"))
                 and contract["primary_feature_count"] == contract["logistic_feature_count"]
                 and contract["no_odds_feature_count"] <= contract["primary_feature_count"]
                 and all(_hash(contract[key]) for key in
                         ("primary_feature_sha256", "logistic_feature_sha256", "no_odds_feature_sha256"))
                 and contract["primary_feature_sha256"] == contract["logistic_feature_sha256"]
                 and contract["modeling_start_date"] == policy["modeling_start_date"]
                 and contract["unsupported_contract_start_date"] == policy["train_cutoff_date"],
                 "evaluator exact feature contract fields conflict")
        for identity_name in ("training_spec", "named_training_spec"):
            identity = policy.get(identity_name)
            _require(isinstance(identity, dict) and set(identity) == {"name", "sha256"}
                     and identity["name"] == "full_live_contract_v6_durability_2022"
                     and _hash(identity["sha256"]), "invalid evaluator recipe identity")
        _require(contract.get("name") == policy["training_spec"]["name"],
                 "evaluator contract recipe identity conflict")
        _require(_instant(cutoff).year == 2026 and policy.get("modeling_start_date") == "2022-01-01"
                 and policy.get("train_cutoff_date") ==
                 (_instant(cutoff).date() + timedelta(days=1)).isoformat(),
                 "evaluator refit window conflict")
        windows = policy.get("evaluation_windows")
        fixed = {"timezone": "UTC", "train_start_inclusive": "2022-01-01",
                 "train_end_exclusive": "2025-01-01", "validation_start_inclusive": "2025-01-01",
                 "validation_end_exclusive": "2026-01-01", "validation_report_only": True,
                 "evaluation_start_inclusive": "2026-01-01", "evaluation_end_inclusive": True}
        _require(isinstance(windows, dict) and set(windows) == set(fixed) | {"common_supported_end_utc"}
                 and all(windows[key] == value and type(windows[key]) is type(value)
                         for key, value in fixed.items()), "evaluator fixed windows conflict")
        common_end = _instant(windows["common_supported_end_utc"])
        _require(common_end.year == 2026 and common_end <= _instant(cutoff),
                 "evaluator common supported cutoff conflict")
        _require(transport.get("policy_binding") == policy,
                 "evaluator saved policy conflict")
        _require(_instant(policy.get("input_completion_cutoff_utc")) == _instant(cutoff),
                 "evaluator policy completion cutoff conflict")
        _require(transport.get("training_spec") == policy.get("training_spec"),
                 "evaluator saved recipe conflict")
        context = transport.get("context")
        _require(isinstance(context, dict)
                 and set(context) == {"utc_month", "accepted_update_id", "accepted_at_utc", "iso_week",
                                      "data_slot", "schedule_timezone", "promotion_month_utc", "claim_id"}
                 and all(isinstance(value, str) and bool(value.strip()) for value in context.values())
                 and context.get("accepted_update_id") == generation
                 and _instant(context.get("accepted_at_utc")) == _instant(cutoff)
                 and context.get("promotion_month_utc") == policy.get("promotion_month_utc"),
                 "evaluator original context conflict")
        _require(context["utc_month"] == _instant(cutoff).strftime("%Y-%m")
                 and re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", context["promotion_month_utc"]) is not None
                 and context["utc_month"] > context["promotion_month_utc"]
                 and context["data_slot"] in {"tuesday", "wednesday"}
                 and context["schedule_timezone"] == "America/New_York"
                 and re.fullmatch(r"[0-9]{4}-W(0[1-9]|[1-4][0-9]|5[0-3])", context["iso_week"]) is not None,
                 "evaluator original month/slot/adoption facts conflict")
    return {
        "contract_sha256": evidence_sha256(contract),
        "input_receipt_sha256": evidence_sha256(input_receipt),
        "accepted_generation_id": generation,
        "input_completion_cutoff_utc": cutoff,
        "source_sha256": input_receipt["source_sha256"],
        "derived_sha256": input_receipt["derived_sha256"],
        "monthly_input_sha256": evidence_sha256(transport) if transport is not None else None,
        "monthly_policy": deepcopy(policy),
    }


def build_evaluator_request(*, candidate_dir: str, input_receipt: Mapping[str, Any],
                            contract: Mapping[str, Any], model_hashes: Mapping[str, str],
                            accepted_update_report: Mapping[str, Any] | None = None,
                            producer_inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    request = {
        "schema_version": 2, "kind": REQUEST_KIND,
        "evaluator_contract": EVALUATOR_CONTRACT, "candidate_dir": candidate_dir,
        "input_receipt": deepcopy(dict(input_receipt)), "contract": deepcopy(dict(contract)),
        "model_hashes": deepcopy(dict(model_hashes)),
        "accepted_update_report": deepcopy(accepted_update_report),
        "bindings": _bindings(input_receipt, contract),
        "candidate_only": True, "automatic_promotion_allowed": False,
    }
    if producer_inputs is not None:
        request.update(schema_version=3, kind=PRODUCER_REQUEST_KIND,
                       producer_inputs=deepcopy(dict(producer_inputs)))
    return validate_evaluator_request(request)


def validate_evaluator_request(request: Any) -> dict[str, Any]:
    fields = {"schema_version", "kind", "evaluator_contract", "candidate_dir", "input_receipt",
              "contract", "model_hashes", "accepted_update_report", "bindings", "candidate_only",
              "automatic_promotion_allowed"}
    producer = isinstance(request, dict) and type(request.get("schema_version")) is int and request["schema_version"] == 3
    if producer:
        fields.add("producer_inputs")
    _require(isinstance(request, dict) and set(request) == fields,
             "invalid evaluator request fields")
    canonical_bytes(request)
    _require(type(request["schema_version"]) is int and request["schema_version"] == (3 if producer else 2)
             and request["kind"] == (PRODUCER_REQUEST_KIND if producer else REQUEST_KIND)
             and request["evaluator_contract"] == EVALUATOR_CONTRACT,
             "unknown evaluator request version")
    _require(isinstance(request["candidate_dir"], str) and bool(request["candidate_dir"]),
             "evaluator candidate identity missing")
    _models(request["model_hashes"])
    _require(request["bindings"] == _bindings(request["input_receipt"], request["contract"]),
             "evaluator request binding conflict")
    _require(request["candidate_only"] is True and request["automatic_promotion_allowed"] is False,
             "evaluator request permits unsafe promotion")
    expected_report = request["input_receipt"].get("accepted_update_report")
    _require(request["accepted_update_report"] == expected_report,
             "evaluator original update report conflict")
    if producer:
        from src.model.monthly_evaluator_inputs import validate_transport
        validate_transport(request["producer_inputs"])
        _require(request["bindings"]["monthly_policy"] is not None,
                 "producer request requires original accepted monthly policy")
    return deepcopy(request)


def _successful_tree(value: Any) -> None:
    """Known correctness decisions cannot be hidden under another success flag."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"passed", "succeeded", "compatible", "correct", "complete", "healthy",
                       "loaded", "parity_matches", "coverage_complete"}:
                _require(item is True, f"nested evaluator {key} decision failed")
            elif key in {"failed", "execution_failed"}:
                _require(item is False, f"nested evaluator {key} decision failed")
            elif key in {"errors", "violations", "failures", "missing"}:
                _require(item == [], f"nested evaluator {key} present")
            elif key == "status":
                _require(item in {"passed", "complete", "succeeded", "compatible"},
                         "nested evaluator status failed or unknown")
            elif key == "checks":
                _require(isinstance(item, dict) and bool(item), "nested evaluator checks missing")
                for decision in item.values():
                    _require(decision is True or isinstance(decision, dict),
                             "nested evaluator check failed")
                    if isinstance(decision, dict):
                        _require(any(decision.get(field) is True for field in
                                     ("passed", "succeeded", "compatible", "correct", "healthy")),
                                 "nested evaluator check lacks explicit success decision")
            _successful_tree(item)
    elif isinstance(value, list):
        for item in value:
            _successful_tree(item)


def _positive_count(value: Any, label: str) -> None:
    _require(type(value) is int and value > 0, f"{label} must be a positive integer")


def validate_correctness_mapping(value: Any) -> dict[str, Any]:
    """Check nested decisions in a correctness report, never comparison metrics."""
    _require(isinstance(value, dict) and value.get("passed") is True,
             "correctness report lacks explicit success")
    canonical_bytes(value)
    _successful_tree(value)
    return deepcopy(value)


def _evidence(name: str, envelope: Any, request: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(envelope, dict) and set(envelope) == {"identity", "bytes", "sha256", "body"},
             f"{name} original evidence envelope missing")
    _require(isinstance(envelope["identity"], str) and bool(envelope["identity"].strip()),
             f"{name} evidence identity missing")
    body = envelope["body"]
    content = canonical_bytes(body)
    _require(type(envelope["bytes"]) is int and envelope["bytes"] == len(content)
             and envelope["sha256"] == hashlib.sha256(content).hexdigest(),
             f"{name} original evidence bytes conflict")
    _require(isinstance(body, dict) and body.get("check") == name
             and body.get("request_sha256") == evidence_sha256(request)
             and body.get("model_hashes") == request["model_hashes"],
             f"{name} evidence binding conflict")
    specific = {
        "fixed_evaluation": {"windows", "honest_historical_replicas", "historical_replica_model_hashes",
                             "replica_training_window", "evaluation_coverage", "report_guided_selection_performed"},
        "production_load": {"loaded_model_hashes"},
        "exact_parity": {"compared_rows", "mismatch_count"},
        "prefight_parity": {"compared_rows", "mismatch_count"},
        "coverage_report": {"evaluated_rows", "missing_required_rows"},
        "current_live_comparison": {"reference_model_hashes", "comparison_scope", "metrics"},
    }
    _require(set(body) == {"check", "request_sha256", "model_hashes", "passed", "errors", "checks"}
             | specific[name], f"{name} evidence fields missing or unknown")
    _require(body.get("passed") is True and body.get("errors") == []
             and isinstance(body.get("checks"), dict) and bool(body["checks"]),
             f"{name} correctness evidence missing")
    _successful_tree({key: value for key, value in body.items()
                      if not (name == "current_live_comparison" and key == "metrics")})
    if name == "fixed_evaluation":
        policy = request["bindings"]["monthly_policy"]
        expected = policy.get("evaluation_windows") if isinstance(policy, dict) else None
        _require(expected is not None and body.get("windows") == expected,
                 "fixed evaluation requires exact independently accepted windows")
        _require(body.get("honest_historical_replicas") is True,
                 "fixed evaluation historical replica evidence missing")
        _models(body.get("historical_replica_model_hashes"))
        _require(all(body["historical_replica_model_hashes"][key] != request["model_hashes"][key]
                     for key in ("primary", "logistic", "no_odds")),
                 "later-outcome refit cannot substitute for historical evaluation replica")
        _require(body.get("replica_training_window") == {
            "train_start_inclusive": expected["train_start_inclusive"],
            "train_end_exclusive": expected["train_end_exclusive"]},
            "historical replica training dates conflict with fixed evaluation")
        _require(body.get("evaluation_coverage") == {
            "start_inclusive": expected["evaluation_start_inclusive"],
            "end_inclusive": expected["common_supported_end_utc"]},
            "fixed evaluation coverage conflicts with common supported cutoff")
        _require(body.get("report_guided_selection_performed") is False,
                 "report-only validation cannot guide recipe selection or tuning")
    elif name == "production_load":
        _require(body.get("loaded_model_hashes") == request["model_hashes"],
                 "production load identities conflict")
    elif name in {"exact_parity", "prefight_parity"}:
        _positive_count(body.get("compared_rows"), f"{name} compared rows")
        _require(type(body.get("mismatch_count")) is int and body["mismatch_count"] == 0,
                 f"{name} mismatches present")
    elif name == "coverage_report":
        _positive_count(body.get("evaluated_rows"), "coverage evaluated rows")
        _require(type(body.get("missing_required_rows")) is int and body["missing_required_rows"] == 0,
                 "coverage required rows missing")
    else:
        _models(body.get("reference_model_hashes"))
        _require(body.get("comparison_scope") == "descriptive_only"
                 and isinstance(body.get("metrics"), dict) and bool(body["metrics"]),
                 "descriptive current model comparison missing")
    return body


def validate_evaluator_report(result: Any, *, request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate protocol semantics, never inferred model performance or execution."""
    selected = validate_evaluator_request(dict(request))
    fields = {"schema_version", "kind", "passed", "request_sha256", "bindings", "model_hashes",
              "checks", "evidence", "comparison_with_current_live", "owner_review_required",
              "auto_promotion_performed", "live_model_replaced"}
    _require(isinstance(result, dict) and set(result) == fields, "invalid evaluator report fields")
    canonical_bytes(result)
    _require(type(result["schema_version"]) is int and result["schema_version"] == 2
             and result["kind"] == REPORT_KIND and result["passed"] is True,
             "evaluator report did not pass supported protocol")
    _require(result["request_sha256"] == evidence_sha256(selected)
             and result["bindings"] == selected["bindings"]
             and result["model_hashes"] == selected["model_hashes"],
             "evaluator report original request binding conflict")
    _require(isinstance(result["checks"], dict) and set(result["checks"]) == set(REQUIRED_CHECKS)
             and all(value is True for value in result["checks"].values()),
             "required evaluator checks failed or missing")
    _require(isinstance(result["evidence"], dict) and set(result["evidence"]) == set(REQUIRED_CHECKS),
             "required evaluator evidence missing")
    identities = set()
    bodies = {}
    for name in REQUIRED_CHECKS:
        bodies[name] = _evidence(name, result["evidence"][name], selected)
        identity = result["evidence"][name]["identity"]
        _require(identity not in identities, "evaluator evidence identities reused")
        identities.add(identity)
    _require(result["comparison_with_current_live"] == bodies["current_live_comparison"]["metrics"],
             "evaluator descriptive comparison conflicts with original evidence")
    _require(result["owner_review_required"] is True
             and result["auto_promotion_performed"] is False
             and result["live_model_replaced"] is False, "evaluator report permits unsafe promotion")
    return deepcopy(result)
