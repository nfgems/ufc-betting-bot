"""Explicit accepted monthly rules; resolving these rules performs no model work.

Only the implemented durability adapter is supported. Original named definitions
remain unchanged. Policy bytes are supplied under an independent outer pin; this
module never creates policy acceptance or chooses real promotion/cutoff values.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re

POLICY_KIND = "accepted_monthly_policy_v1"
ADAPTER = "durability_monthly_refit_v1"
SUPPORTED_NAMED_SPEC = "full_live_contract_v6_durability_2022"
EVALUATOR_CONTRACT = "monthly_evaluator_evidence_v1"
MAX_POLICY_BYTES = 128 * 1024


class MonthlyPolicyRefusal(ValueError):
    """An independently accepted implemented policy cannot be established."""


def _require(condition, message):
    if not condition:
        raise MonthlyPolicyRefusal(message)


def _time(value):
    _require(isinstance(value, str), "policy timestamp must be explicit text")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MonthlyPolicyRefusal("invalid policy timestamp") from exc
    _require(result.tzinfo is not None and result.utcoffset() is not None,
             "policy timestamp must be timezone-aware")
    _require(result.utcoffset() == timedelta(0), "policy timestamps must use UTC")
    return result.astimezone(timezone.utc)


def _month(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", value),
             "invalid explicit policy promotion month")
    _require(int(value[:4]) >= 1, "invalid policy promotion year")
    return value


def _identity(value):
    from src.model import prepared_input as prepared
    _require(isinstance(value, dict) and set(value) == {"name", "sha256"}
             and isinstance(value["name"], str) and value["name"], "invalid policy spec identity")
    prepared._sha(value["sha256"])
    return value


def feature_order_sha256(columns):
    """Same canonical ordered feature hashing used by candidate receipts."""
    return hashlib.sha256(json.dumps(list(columns), separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LoadedMonthlyPolicy:
    document: dict
    pin: dict
    body: bytes
    spec: object
    binding: dict
    context: dict | None = None

    @property
    def promotion_month_utc(self):
        return self.binding["promotion_month_utc"]


def load_monthly_policy(pin, *, body=None, context=None):
    """Resolve pinned original bytes, or already-saved originals, without writes."""
    from src.model import prepared_input as prepared
    from src.model.training_preflight import resolved_spec_identity
    from src.model.training_spec import resolve_named_training_spec
    pin = prepared._pin(pin)
    _require(0 < pin["bytes"] <= MAX_POLICY_BYTES, "monthly policy exceeds bounded record size")
    if body is None:
        path = prepared._path(pin["path"])
        _require(path.stat().st_size == pin["bytes"], "monthly policy actual length differs from accepted pin")
        with path.open("rb") as stream:
            body = stream.read(MAX_POLICY_BYTES + 1)
        prepared._check_bytes(body, pin)
    else:
        prepared._check_bytes(body, pin)
    value = prepared._json(body)
    _require(isinstance(value, dict) and set(value) == {
        "schema", "kind", "policy_id", "policy_version", "adapter", "named_training_spec",
        "features", "adoption", "temporal", "evaluator_contract"},
        "invalid explicit monthly policy schema")
    _require(type(value["schema"]) is int and value["schema"] == 1 and value["kind"] == POLICY_KIND
             and type(value["policy_version"]) is int and value["policy_version"] == 1,
             "unsupported monthly policy version")
    _require(isinstance(value["policy_id"], str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value["policy_id"]),
             "invalid explicit monthly policy identity")
    _require(value["adapter"] == ADAPTER and value["evaluator_contract"] == EVALUATOR_CONTRACT,
             "unsupported monthly policy adapter or evaluator contract")
    named = _identity(value["named_training_spec"])
    _require(named["name"] == SUPPORTED_NAMED_SPEC, "unsupported explicit monthly named recipe")
    base = resolve_named_training_spec(named["name"])
    _require(named == {"name": base.name, "sha256": resolved_spec_identity(base)},
             "monthly original named recipe identity conflict")
    features = value["features"]
    _require(isinstance(features, dict) and set(features) == {"columns", "sha256", "count"}
             and type(features["count"]) is int and features["count"] == len(base.feature_cols)
             and isinstance(features["columns"], list) and features["columns"] == base.feature_cols
             and len(set(base.feature_cols)) == len(base.feature_cols)
             and features["sha256"] == feature_order_sha256(base.feature_cols),
             "monthly named feature order/count/hash conflict")
    adoption = value["adoption"]
    _require(isinstance(adoption, dict) and set(adoption) == {
        "model_id", "adoption_id", "promotion_month_utc", "stable_promoted_at_utc"},
        "missing explicit stable adoption facts")
    for field in ("model_id", "adoption_id"):
        _require(isinstance(adoption[field], str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", adoption[field]),
                 "invalid explicit stable adoption identity")
    month = _month(adoption["promotion_month_utc"])
    promoted = _time(adoption["stable_promoted_at_utc"])
    _require(promoted.strftime("%Y-%m") == month, "stable adoption time/month conflict")
    temporal = value["temporal"]
    _require(isinstance(temporal, dict) and set(temporal) == {
        "version", "timezone", "event_date_interpretation", "modeling_start_date",
        "cutoff_inclusive", "refit_train_start_date", "refit_end_rule",
        "refit_cutoff_inclusive", "evaluation_windows", "supported_completion_year"},
        "missing explicit monthly temporal facts")
    _require(temporal["version"] == ADAPTER and temporal["timezone"] == "UTC"
             and temporal["event_date_interpretation"] == "date_only_utc_midnight"
             and temporal["modeling_start_date"] == temporal["refit_train_start_date"] == "2022-01-01"
             and temporal["refit_end_rule"] == "exclusive_day_after_qualifying_completion"
             and type(temporal["supported_completion_year"]) is int and temporal["supported_completion_year"] == 2026
             and temporal["cutoff_inclusive"] is True and temporal["refit_cutoff_inclusive"] is False,
             "unsupported monthly temporal rule or inclusivity")
    windows = temporal["evaluation_windows"]
    fixed = {"timezone": "UTC", "train_start_inclusive": "2022-01-01", "train_end_exclusive": "2025-01-01",
             "validation_start_inclusive": "2025-01-01", "validation_end_exclusive": "2026-01-01",
             "validation_report_only": True, "evaluation_start_inclusive": "2026-01-01",
             "evaluation_end_inclusive": True}
    _require(isinstance(windows, dict) and set(windows) == set(fixed) | {"common_supported_end_utc"}
             and all(windows[key] == expected and type(windows[key]) is type(expected)
                     for key, expected in fixed.items()), "fixed monthly evaluation windows conflict")
    common_end = _time(windows["common_supported_end_utc"])
    _require(common_end.year == 2026, "common evaluation cutoff must be within 2026")
    binding = {"policy_id": value["policy_id"], "policy_version": value["policy_version"], "policy": dict(pin),
        "adapter": ADAPTER, "named_training_spec": named, "modeling_start_date": "2022-01-01",
        "adoption_id": adoption["adoption_id"], "model_id": adoption["model_id"],
        "promotion_month_utc": month, "evaluator_contract": EVALUATOR_CONTRACT,
        "evaluation_windows": windows}
    spec = None
    if context is not None:
        _require(isinstance(context, dict) and set(context) == {
            "utc_month", "accepted_update_id", "accepted_at_utc", "iso_week", "data_slot",
            "schedule_timezone", "promotion_month_utc", "claim_id"}
            and all(isinstance(item, str) and bool(item.strip()) for item in context.values()),
            "invalid monthly policy selection context")
        _require(context["data_slot"] in {"tuesday", "wednesday"}
                 and context["schedule_timezone"] == "America/New_York"
                 and re.fullmatch(r"[0-9]{4}-W[0-9]{2}", context["iso_week"]) is not None,
                 "invalid original monthly policy calendar facts")
        try:
            datetime.fromisocalendar(int(context["iso_week"][:4]), int(context["iso_week"][6:]), 1)
        except ValueError as exc:
            raise MonthlyPolicyRefusal("invalid original monthly policy ISO week") from exc
        cutoff = _time(context["accepted_at_utc"])
        _require(context["promotion_month_utc"] == month and context["utc_month"] == cutoff.strftime("%Y-%m")
                 and context["utc_month"] > month and cutoff > promoted,
                 "monthly selection must match stable adoption and occur in a later month")
        _require(cutoff.year == temporal["supported_completion_year"],
                 "this explicit monthly adapter supports 2026 completion only")
        _require(common_end <= cutoff, "common evaluation cutoff follows original completion")
        exclusive = (cutoff.date() + timedelta(days=1)).isoformat()
        # Context comes from the separately pinned original input descriptor.
        # The descriptor must independently accept this realized spec identity.
        spec = replace(base, train_start_date="2022-01-01", train_end_date="", train_cutoff_date=exclusive)
        binding.update(training_spec={"name": spec.name, "sha256": resolved_spec_identity(spec)},
                       train_cutoff_date=exclusive, input_completion_cutoff_utc=context["accepted_at_utc"])
    return LoadedMonthlyPolicy(deepcopy(value), dict(pin), bytes(body), spec, deepcopy(binding), deepcopy(context))


def validate_policy_context(policy, context):
    """The accepted policy cannot select another completion, adoption or month."""
    _require(context["promotion_month_utc"] == policy.promotion_month_utc,
             "monthly policy adoption month differs from original selection")
    _require(context["accepted_at_utc"] == policy.binding["input_completion_cutoff_utc"],
             "monthly policy cutoff differs from original selection")
    _require(context["utc_month"] == _time(context["accepted_at_utc"]).strftime("%Y-%m")
             and context["utc_month"] > policy.promotion_month_utc,
             "monthly policy selection is not eligible")
    return policy
