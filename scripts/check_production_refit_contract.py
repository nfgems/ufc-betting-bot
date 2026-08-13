"""Fail-closed contract gate for the fixed scheduled UFC refit policy.

The scheduled job is not a model-selection job.  Its policy names one approved
evaluation recipe and one versioned full-fit recipe.  This module validates the
checked-in registry, refreshed feature snapshot, active production lineage, and
the staged candidate without deriving any contract identity from a mutable
production manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.training_spec import NamedModelTrainingSpec, resolve_named_training_spec
from scripts import bfo_lineage


class ContractInputError(ValueError):
    """Raised for malformed, missing, or ambiguous contract evidence."""


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_TOP_LEVEL_KEYS = {
    "schema_version",
    "policy_id",
    "contract",
    "evaluation",
    "baseline",
    "health_limits",
    "root_release",
}
_CONTRACT_KEYS = {
    "evaluation_spec_name",
    "evaluation_spec_payload_sha256",
    "fullfit_spec_name",
    "fullfit_spec_payload_sha256",
    "allowed_fullfit_differences",
    "exclusive_train_cutoff_date",
    "minimum_cutoff_buffer_days",
    "feature_count",
}
_EVALUATION_KEYS = {
    "model_seed",
    "odds_noise_seed",
    "retrain_months",
    "initial_train_years",
    "min_train_test_fights",
    "bet_start_date",
    "execution_mode",
    "entry_offset_days",
    "entry_offset_for_features",
    "strategy_name",
    "initial_bankroll",
    "min_edge",
    "kelly_fraction",
    "max_bet_fraction",
    "blend_weight",
    "require_agreement",
    "agreement_model",
    "model_agreement_min_edge",
    "min_model_probability",
    "max_decimal_odds",
    "dynamic_blend_min",
    "dynamic_blend_max",
    "dynamic_blend_confidence",
    "dynamic_blend_agreement_boost",
    "newbie_mode",
    "minimum_fighter_fights",
    "line_movement_filter",
    "execution_assumptions",
}
_EXECUTION_ASSUMPTION_KEYS = {
    "min_book_liquidity",
    "max_slippage",
    "max_bet_vs_book_ratio",
    "assumed_half_spread",
    "synthetic_liquidity_floor",
    "synthetic_liquidity_peak",
    "synthetic_price_step",
    "synthetic_depth_notional_shares",
}
_BASELINE_KEYS = {
    "comparison_role",
    "evidence_path",
    "evidence_sha256",
    "evidence_root_source_manifest_sha256",
    "evaluation_spec_name",
    "features_sha256",
    "evidence_protocol_sha256",
    "scheduled_protocol_sha256",
    "evaluation_sample_sha256",
    "model_seed",
    "odds_noise_seed",
    "execution_mode",
    "entry_offset_days",
    "entry_offset_for_features",
    "evaluation_start_date",
    "evaluation_end_date",
    "evaluation_n_fights",
    "evaluation_n_folds",
    "model_brier_score",
    "model_ece",
    "strategy_total_bets",
    "strategy_roi",
    "strategy_total_profit",
    "strategy_avg_clv",
    "strategy_max_drawdown_pct",
}
_HEALTH_LIMIT_KEYS = {
    "maximum_brier_relative_regression",
    "maximum_ece_absolute_regression",
    "minimum_evaluation_fights",
    "minimum_evaluation_folds",
    "minimum_strategy_bets",
    "minimum_strategy_roi",
    "minimum_strategy_total_profit",
    "maximum_drawdown_multiplier",
    "maximum_clv_absolute_regression",
    "maximum_snapshot_age_days",
}
_ROOT_RELEASE_KEYS = {
    "bundle_id",
    "release_id",
    "source_manifest_sha256",
    "installed_manifest_sha256",
    "promotion_git_sha",
    "training_source_git_sha",
    "model_sha256",
    "no_odds_model_sha256",
    "logistic_model_sha256",
    "processed_fights_sha256",
    "processed_features_sha256",
}
_SCHEDULED_MANIFEST_POLICY_KEYS = {
    "policy_id",
    "sha256",
    "root_bundle_id",
    "parent_bundle_id",
    "parent_model_spec_name",
    "parent_model_sha256",
    "parent_no_odds_model_sha256",
    "parent_logistic_model_sha256",
    "parent_processed_fights_sha256",
    "parent_processed_features_sha256",
}


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractInputError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractInputError(f"cannot read {label} {path}: {exc}") from exc
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    except (json.JSONDecodeError, ContractInputError) as exc:
        raise ContractInputError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractInputError(f"{label} must contain one JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ContractInputError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractInputError(f"{label} must be an object")
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ContractInputError(f"{label} has invalid fields: {'; '.join(details)}")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractInputError(f"{label} must be a nonempty string")
    return value.strip()


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ContractInputError(f"{label} must be a boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractInputError(f"{label} must be an integer >= {minimum}")
    return value


def _require_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractInputError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        suffix = f" >= {minimum}" if minimum is not None else " and finite"
        raise ContractInputError(f"{label} must be finite{suffix}")
    return parsed


def _require_sha256(value: Any, label: str) -> str:
    parsed = _require_string(value, label).lower()
    if not _SHA256_RE.fullmatch(parsed):
        raise ContractInputError(f"{label} must be a lowercase SHA-256 digest")
    return parsed


def _require_git_sha(value: Any, label: str) -> str:
    parsed = _require_string(value, label).lower()
    if not _GIT_SHA_RE.fullmatch(parsed):
        raise ContractInputError(f"{label} must be a lowercase 40-character Git SHA")
    return parsed


def _require_iso_date(value: Any, label: str) -> str:
    parsed = _require_string(value, label)
    try:
        converted = datetime.strptime(parsed, "%Y-%m-%d")
    except ValueError as exc:
        raise ContractInputError(f"{label} must use YYYY-MM-DD") from exc
    if converted.strftime("%Y-%m-%d") != parsed:
        raise ContractInputError(f"{label} must use canonical YYYY-MM-DD")
    return parsed


def load_policy(path: Path) -> dict[str, Any]:
    """Load and strictly validate the complete v1 policy schema."""

    policy = _require_exact_keys(read_json_object(path, label="scheduled refit policy"), _TOP_LEVEL_KEYS, "policy")
    if policy["schema_version"] != 1 or isinstance(policy["schema_version"], bool):
        raise ContractInputError("policy.schema_version must be exactly 1")
    _require_string(policy["policy_id"], "policy.policy_id")

    contract = _require_exact_keys(policy["contract"], _CONTRACT_KEYS, "policy.contract")
    for field in ("evaluation_spec_name", "fullfit_spec_name"):
        _require_string(contract[field], f"policy.contract.{field}")
    if contract["evaluation_spec_name"] == contract["fullfit_spec_name"]:
        raise ContractInputError("evaluation and full-fit spec names must differ")
    for field in ("evaluation_spec_payload_sha256", "fullfit_spec_payload_sha256"):
        _require_sha256(contract[field], f"policy.contract.{field}")
    if contract["allowed_fullfit_differences"] != ["description", "name", "train_cutoff_date"]:
        raise ContractInputError(
            "policy.contract.allowed_fullfit_differences must be exactly "
            "['description', 'name', 'train_cutoff_date']"
        )
    _require_iso_date(contract["exclusive_train_cutoff_date"], "policy.contract.exclusive_train_cutoff_date")
    _require_int(contract["minimum_cutoff_buffer_days"], "policy.contract.minimum_cutoff_buffer_days", minimum=1)
    _require_int(contract["feature_count"], "policy.contract.feature_count", minimum=1)

    evaluation = _require_exact_keys(policy["evaluation"], _EVALUATION_KEYS, "policy.evaluation")
    for field in ("model_seed", "odds_noise_seed", "retrain_months", "initial_train_years", "min_train_test_fights", "minimum_fighter_fights"):
        _require_int(evaluation[field], f"policy.evaluation.{field}", minimum=1)
    _require_iso_date(evaluation["bet_start_date"], "policy.evaluation.bet_start_date")
    if evaluation["execution_mode"] != "realistic":
        raise ContractInputError("policy.evaluation.execution_mode must be 'realistic'")
    if _require_number(evaluation["entry_offset_days"], "policy.evaluation.entry_offset_days", minimum=0.0) != 1.0:
        raise ContractInputError("policy.evaluation.entry_offset_days must be exactly 1.0")
    if not _require_bool(evaluation["entry_offset_for_features"], "policy.evaluation.entry_offset_for_features"):
        raise ContractInputError("policy must feed the T-1 entry snapshot into model features")
    if evaluation["strategy_name"] != "production_gated":
        raise ContractInputError("policy.evaluation.strategy_name must be 'production_gated'")
    if evaluation["agreement_model"] != "xgboost_no_odds":
        raise ContractInputError("policy.evaluation.agreement_model must be 'xgboost_no_odds'")
    if not _require_bool(evaluation["require_agreement"], "policy.evaluation.require_agreement"):
        raise ContractInputError("scheduled evaluation must require model agreement")
    _require_bool(evaluation["line_movement_filter"], "policy.evaluation.line_movement_filter")
    for field in (
        "initial_bankroll", "min_edge", "kelly_fraction", "max_bet_fraction",
        "blend_weight", "model_agreement_min_edge", "min_model_probability",
        "max_decimal_odds", "dynamic_blend_min", "dynamic_blend_max",
        "dynamic_blend_confidence", "dynamic_blend_agreement_boost",
    ):
        _require_number(evaluation[field], f"policy.evaluation.{field}", minimum=0.0)
    if evaluation["newbie_mode"] != "hard_skip":
        raise ContractInputError("policy.evaluation.newbie_mode must be 'hard_skip'")
    assumptions = _require_exact_keys(
        evaluation["execution_assumptions"],
        _EXECUTION_ASSUMPTION_KEYS,
        "policy.evaluation.execution_assumptions",
    )
    for field in _EXECUTION_ASSUMPTION_KEYS - {"synthetic_depth_notional_shares"}:
        _require_number(assumptions[field], f"policy.evaluation.execution_assumptions.{field}", minimum=0.0)
    shares = assumptions["synthetic_depth_notional_shares"]
    if not isinstance(shares, list) or not shares:
        raise ContractInputError("synthetic_depth_notional_shares must be a nonempty array")
    for index, share in enumerate(shares):
        _require_number(share, f"synthetic_depth_notional_shares[{index}]", minimum=0.0)

    baseline = _require_exact_keys(policy["baseline"], _BASELINE_KEYS, "policy.baseline")
    if baseline["comparison_role"] != "root_release_reference_thresholds":
        raise ContractInputError(
            "policy.baseline.comparison_role must be "
            "'root_release_reference_thresholds'"
        )
    evidence_path = Path(_require_string(baseline["evidence_path"], "policy.baseline.evidence_path"))
    if evidence_path.is_absolute() or ".." in evidence_path.parts:
        raise ContractInputError("policy.baseline.evidence_path must be a safe relative path")
    for field in (
        "evidence_sha256",
        "evidence_root_source_manifest_sha256",
        "features_sha256",
        "evidence_protocol_sha256",
        "scheduled_protocol_sha256",
        "evaluation_sample_sha256",
    ):
        _require_sha256(baseline[field], f"policy.baseline.{field}")
    if baseline["evidence_protocol_sha256"] != baseline["scheduled_protocol_sha256"]:
        raise ContractInputError(
            "root-release evidence and scheduled protocol identities must match"
        )
    if baseline["evaluation_spec_name"] != contract["evaluation_spec_name"]:
        raise ContractInputError(
            "root-release baseline spec does not match the fixed evaluation spec"
        )
    for field in ("model_seed", "odds_noise_seed"):
        _require_int(baseline[field], f"policy.baseline.{field}", minimum=1)
        if baseline[field] != evaluation[field]:
            raise ContractInputError(
                f"root-release baseline {field} does not match scheduled evaluation"
            )
    if baseline["execution_mode"] != evaluation["execution_mode"]:
        raise ContractInputError(
            "root-release baseline execution mode does not match scheduled evaluation"
        )
    if _require_number(baseline["entry_offset_days"], "policy.baseline.entry_offset_days", minimum=0.0) != evaluation["entry_offset_days"]:
        raise ContractInputError(
            "root-release baseline entry offset does not match scheduled evaluation"
        )
    if _require_bool(baseline["entry_offset_for_features"], "policy.baseline.entry_offset_for_features") != evaluation["entry_offset_for_features"]:
        raise ContractInputError(
            "root-release baseline feature-entry offset does not match scheduled evaluation"
        )
    for field in ("evaluation_start_date", "evaluation_end_date"):
        _require_iso_date(baseline[field], f"policy.baseline.{field}")
    for field in ("evaluation_n_fights", "evaluation_n_folds", "strategy_total_bets"):
        _require_int(baseline[field], f"policy.baseline.{field}", minimum=1)
    for field in (
        "model_brier_score", "model_ece", "strategy_roi", "strategy_total_profit",
        "strategy_avg_clv", "strategy_max_drawdown_pct",
    ):
        _require_number(baseline[field], f"policy.baseline.{field}")

    limits = _require_exact_keys(policy["health_limits"], _HEALTH_LIMIT_KEYS, "policy.health_limits")
    for field in ("minimum_evaluation_fights", "minimum_evaluation_folds", "minimum_strategy_bets", "maximum_snapshot_age_days"):
        _require_int(limits[field], f"policy.health_limits.{field}", minimum=1)
    for field in _HEALTH_LIMIT_KEYS - {
        "minimum_evaluation_fights", "minimum_evaluation_folds",
        "minimum_strategy_bets", "maximum_snapshot_age_days",
    }:
        _require_number(limits[field], f"policy.health_limits.{field}", minimum=0.0)

    root = _require_exact_keys(policy["root_release"], _ROOT_RELEASE_KEYS, "policy.root_release")
    for field in ("bundle_id", "release_id"):
        _require_string(root[field], f"policy.root_release.{field}")
    for field in ("promotion_git_sha", "training_source_git_sha"):
        _require_git_sha(root[field], f"policy.root_release.{field}")
    for field in _ROOT_RELEASE_KEYS - {
        "bundle_id", "release_id", "promotion_git_sha", "training_source_git_sha"
    }:
        _require_sha256(root[field], f"policy.root_release.{field}")
    if baseline["scheduled_protocol_sha256"] != scheduled_protocol_sha256(policy):
        raise ContractInputError(
            "policy baseline scheduled_protocol_sha256 does not match the fixed protocol"
        )
    if baseline["evidence_protocol_sha256"] != scheduled_protocol_sha256(policy):
        raise ContractInputError(
            "policy baseline evidence_protocol_sha256 does not match the fixed protocol"
        )
    if (
        baseline["evidence_root_source_manifest_sha256"]
        != root["source_manifest_sha256"]
    ):
        raise ContractInputError(
            "root-release baseline evidence is not bound to the fixed root source manifest"
        )
    return policy


def policy_file_identity(path: Path, policy: dict[str, Any]) -> dict[str, str]:
    return {
        "policy_id": str(policy["policy_id"]),
        "sha256": file_sha256(path),
    }


def scheduled_protocol(policy: dict[str, Any]) -> dict[str, Any]:
    """Return the complete fixed evaluation protocol recorded by Track C."""

    contract = policy["contract"]
    return {
        "schema_version": 1,
        "policy_id": policy["policy_id"],
        "evaluation_spec_name": contract["evaluation_spec_name"],
        "evaluation_spec_payload_sha256": contract["evaluation_spec_payload_sha256"],
        "evaluation": policy["evaluation"],
    }


def scheduled_protocol_sha256(policy: dict[str, Any]) -> str:
    return canonical_json_sha256(scheduled_protocol(policy))


def _effective_spec_payload(spec: NamedModelTrainingSpec) -> dict[str, Any]:
    payload = asdict(spec)
    payload.pop("git_hash", None)
    payload.pop("trained_at", None)
    if payload.get("odds_noise_seed") is None:
        params = payload.get("xgb_params")
        seed = params.get("random_state", 42) if isinstance(params, dict) else 42
        payload["odds_noise_seed"] = int(seed)
    return payload


def validate_policy_registry(policy: dict[str, Any]) -> tuple[list[str], NamedModelTrainingSpec | None, NamedModelTrainingSpec | None]:
    errors: list[str] = []
    contract = policy["contract"]
    try:
        evaluation = resolve_named_training_spec(contract["evaluation_spec_name"])
        fullfit = resolve_named_training_spec(contract["fullfit_spec_name"])
    except (KeyError, TypeError, ValueError) as exc:
        return [f"cannot resolve fixed scheduled training contract: {exc}"], None, None

    evaluation_payload = asdict(evaluation)
    fullfit_payload = asdict(fullfit)
    if canonical_json_sha256(evaluation_payload) != contract["evaluation_spec_payload_sha256"]:
        errors.append("fixed evaluation spec payload hash does not match the policy")
    if canonical_json_sha256(fullfit_payload) != contract["fullfit_spec_payload_sha256"]:
        errors.append("fixed full-fit spec payload hash does not match the policy")
    if len(fullfit.feature_cols) != contract["feature_count"]:
        errors.append(
            f"fixed full-fit feature count is {len(fullfit.feature_cols)}, "
            f"policy requires {contract['feature_count']}"
        )
    if fullfit.train_cutoff_date != contract["exclusive_train_cutoff_date"]:
        errors.append("fixed full-fit cutoff does not match the policy")

    evaluation_effective = _effective_spec_payload(evaluation)
    fullfit_effective = _effective_spec_payload(fullfit)
    changed = sorted(
        key
        for key in set(evaluation_effective) | set(fullfit_effective)
        if evaluation_effective.get(key) != fullfit_effective.get(key)
    )
    if changed != contract["allowed_fullfit_differences"]:
        errors.append(
            "evaluation/full-fit contracts differ outside the reviewed mapping: "
            f"actual={changed}, allowed={contract['allowed_fullfit_differences']}"
        )
    for label, spec in (("evaluation", evaluation), ("full-fit", fullfit)):
        violations = spec.validate_feature_contract()
        if violations:
            errors.append(f"{label} spec has live feature-contract violations: {violations}")
    return errors, evaluation, fullfit


def inspect_features(
    path: Path,
    *,
    policy: dict[str, Any],
    fullfit_spec: NamedModelTrainingSpec,
    now: datetime | None = None,
) -> tuple[dict[str, Any], list[str]]:
    if not path.is_file():
        raise ContractInputError(f"features snapshot is missing: {path}")
    try:
        columns = list(pd.read_csv(path, nrows=0).columns)
        dates = pd.read_csv(path, usecols=["event_date"])["event_date"]
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ContractInputError(f"cannot load features snapshot {path}: {exc}") from exc
    errors: list[str] = []
    missing = [column for column in fullfit_spec.feature_cols if column not in columns]
    if missing:
        errors.append(f"features snapshot is missing {len(missing)} fixed contract columns: {missing[:12]}")
    parsed_dates = pd.to_datetime(dates, format="mixed", errors="coerce")
    if parsed_dates.isna().all():
        raise ContractInputError(f"features snapshot has no usable event_date: {path}")
    if parsed_dates.isna().any():
        errors.append("features snapshot contains invalid event_date values")
    snapshot_day = pd.Timestamp(parsed_dates.max()).normalize()
    reference = pd.Timestamp(now or datetime.now(timezone.utc)).tz_localize(None).normalize()
    age_days = int((reference - snapshot_day).days)
    maximum_age = int(policy["health_limits"]["maximum_snapshot_age_days"])
    if age_days < 0:
        errors.append(f"features snapshot max {snapshot_day.date()} is in the future")
    elif age_days > maximum_age:
        errors.append(
            f"features snapshot max {snapshot_day.date()} is {age_days} days old; "
            f"maximum allowed is {maximum_age} days"
        )
    cutoff_day = pd.Timestamp(policy["contract"]["exclusive_train_cutoff_date"])
    if cutoff_day <= snapshot_day:
        errors.append(
            f"fixed cutoff {cutoff_day.date()} does not include snapshot max "
            f"{snapshot_day.date()}"
        )
    buffer_days = int((cutoff_day - snapshot_day).days)
    minimum_buffer = int(policy["contract"]["minimum_cutoff_buffer_days"])
    if buffer_days < minimum_buffer:
        errors.append(
            f"fixed cutoff {cutoff_day.date()} is only {buffer_days} days after "
            f"snapshot max; minimum required buffer is {minimum_buffer} days"
        )
    return {
        "path": str(path.resolve(strict=False)),
        "sha256": file_sha256(path),
        "bytes": int(path.stat().st_size),
        "column_count": len(columns),
        "snapshot_max_event_date": snapshot_day.strftime("%Y-%m-%d"),
        "snapshot_age_days": age_days,
        "cutoff_buffer_days": buffer_days,
    }, errors


def _production_bundle(readyz: dict[str, Any]) -> dict[str, Any]:
    bundle = readyz.get("production_bundle")
    if not isinstance(bundle, dict):
        raise ContractInputError("readyz evidence has no production_bundle object")
    return bundle


def _bundle_hash(bundle: dict[str, Any], direct: str, nested: str) -> Any:
    value = bundle.get(direct)
    hashes = bundle.get("model_hashes")
    if value in (None, "") and isinstance(hashes, dict):
        value = hashes.get(nested)
    return value


def checked_out_git_sha(*, repo_root: Path = REPO_ROOT) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractInputError(f"cannot resolve checked-out workflow Git SHA: {exc}") from exc
    return _require_git_sha(completed.stdout.strip(), "checked-out workflow Git SHA")


def validate_workflow_git_lineage(
    policy: dict[str, Any],
    *,
    workflow_git_sha: str,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    workflow_git_sha = _require_git_sha(workflow_git_sha, "workflow Git SHA")
    promotion_sha = policy["root_release"]["promotion_git_sha"]
    try:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", promotion_sha, workflow_git_sha],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ContractInputError(f"cannot validate workflow Git ancestry: {exc}") from exc
    if completed.returncode == 0:
        return []
    if completed.returncode == 1:
        return [
            f"workflow Git SHA {workflow_git_sha} does not descend from fixed "
            f"promotion SHA {promotion_sha}"
        ]
    detail = completed.stderr.strip() or f"git exited {completed.returncode}"
    raise ContractInputError(f"cannot validate workflow Git ancestry: {detail}")


def validate_active_lineage(
    readyz: dict[str, Any],
    *,
    policy: dict[str, Any],
    policy_sha256: str,
    workflow_git_sha: str,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    if readyz.get("ready") is not True:
        errors.append("production readiness evidence does not report ready=true")
    bundle = _production_bundle(readyz)
    expected_spec = policy["contract"]["fullfit_spec_name"]
    if bundle.get("model_spec_name") != expected_spec:
        errors.append(
            f"active production spec is {bundle.get('model_spec_name')!r}; "
            f"policy requires {expected_spec!r}"
        )
    if bundle.get("deployed_git_sha") != workflow_git_sha:
        errors.append(
            "active production deployed_git_sha does not match the checked-out "
            f"workflow HEAD {workflow_git_sha}"
        )

    root = policy["root_release"]
    if bundle.get("bundle_id") == root["bundle_id"]:
        expected = {
            "rich_release_id": root["release_id"],
            "source_manifest_sha256": root["source_manifest_sha256"],
            "installed_manifest_sha256": root["installed_manifest_sha256"],
            "training_source_git_sha": root["training_source_git_sha"],
            "model_sha256": root["model_sha256"],
            "no_odds_model_sha256": root["no_odds_model_sha256"],
            "logistic_model_sha256": root["logistic_model_sha256"],
            "immutable_training_fights_sha256": root["processed_fights_sha256"],
            "immutable_training_features_sha256": root["processed_features_sha256"],
        }
        actual = {
            "rich_release_id": bundle.get("rich_release_id"),
            "source_manifest_sha256": bundle.get("source_manifest_sha256"),
            "installed_manifest_sha256": bundle.get("installed_manifest_sha256"),
            "training_source_git_sha": bundle.get("training_source_git_sha"),
            "model_sha256": _bundle_hash(bundle, "model_sha256", "primary_sha256"),
            "no_odds_model_sha256": _bundle_hash(bundle, "no_odds_model_sha256", "no_odds_sha256"),
            "logistic_model_sha256": _bundle_hash(bundle, "logistic_model_sha256", "logistic_sha256"),
            "immutable_training_fights_sha256": bundle.get(
                "immutable_training_fights_sha256"
            ),
            "immutable_training_features_sha256": bundle.get(
                "immutable_training_features_sha256"
            ),
        }
        for field, expected_value in expected.items():
            if actual[field] != expected_value:
                errors.append(f"active root release {field} does not match the fixed policy")
    else:
        embedded = bundle.get("scheduled_refit_policy")
        if not isinstance(embedded, dict):
            embedded = {
                "policy_id": bundle.get("scheduled_refit_policy_id"),
                "sha256": bundle.get("scheduled_refit_policy_sha256"),
                "root_bundle_id": bundle.get("scheduled_refit_root_bundle_id"),
            }
        if embedded.get("policy_id") != policy["policy_id"]:
            errors.append("active non-root release has a different scheduled policy id")
        if embedded.get("sha256") != policy_sha256:
            errors.append("active non-root release has a different scheduled policy hash")
        if embedded.get("root_bundle_id") != root["bundle_id"]:
            errors.append("active non-root release is not descended from the fixed root bundle")
    for field in (
        "independent_audit_fights_canonical_sha256",
        "independent_audit_features_canonical_sha256",
    ):
        if not _SHA256_RE.fullmatch(str(bundle.get(field) or "")):
            errors.append(f"active production {field} is missing or invalid")
    source_manifest_sha = str(bundle.get("source_manifest_sha256") or "")
    lineage_sha = bundle.get("scheduled_bfo_lineage_manifest_sha256")
    if lineage_sha is None:
        if source_manifest_sha != policy["root_release"]["source_manifest_sha256"]:
            errors.append(
                "active non-root production is missing scheduled_bfo_lineage_manifest_sha256"
            )
    elif not _SHA256_RE.fullmatch(str(lineage_sha)):
        errors.append("active scheduled_bfo_lineage_manifest_sha256 is invalid")
    return errors, bundle


def evaluate_preflight(
    *,
    policy_path: Path,
    features_path: Path,
    readyz_path: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    policy = load_policy(policy_path)
    identity = policy_file_identity(policy_path, policy)
    errors, _evaluation, fullfit = validate_policy_registry(policy)
    workflow_git_sha = checked_out_git_sha()
    errors.extend(
        validate_workflow_git_lineage(policy, workflow_git_sha=workflow_git_sha)
    )
    if fullfit is None:
        return {"errors": errors, "policy": identity}
    features, feature_errors = inspect_features(features_path, policy=policy, fullfit_spec=fullfit, now=now)
    readyz = read_json_object(readyz_path, label="production readiness evidence")
    lineage_errors, bundle = validate_active_lineage(
        readyz,
        policy=policy,
        policy_sha256=identity["sha256"],
        workflow_git_sha=workflow_git_sha,
    )
    return {
        "errors": [*errors, *feature_errors, *lineage_errors],
        "policy": identity,
        "fixed_contract": {
            "evaluation_spec_name": policy["contract"]["evaluation_spec_name"],
            "fullfit_spec_name": policy["contract"]["fullfit_spec_name"],
            "protocol_sha256": scheduled_protocol_sha256(policy),
            "workflow_git_sha": workflow_git_sha,
        },
        "features": features,
        "active_bundle_id": bundle.get("bundle_id"),
        "active_independent_audit": {
            "fights_canonical_sha256": bundle.get(
                "independent_audit_fights_canonical_sha256"
            ),
            "features_canonical_sha256": bundle.get(
                "independent_audit_features_canonical_sha256"
            ),
        },
    }


def _confined_manifest_path(root: Path, value: Any, *, label: str) -> Path:
    raw = Path(_require_string(value, label))
    root = root.resolve(strict=False)
    resolved = raw.resolve(strict=False) if raw.is_absolute() else (root / raw).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ContractInputError(f"{label} escapes the candidate root: {value!r}") from exc
    return resolved


def _validate_file_record(root: Path, record: Any, *, label: str, expected_path: str) -> tuple[Path, list[str]]:
    errors: list[str] = []
    if not isinstance(record, dict):
        raise ContractInputError(f"{label} must be an object")
    if record.get("staged_path") != expected_path:
        errors.append(f"{label}.staged_path must be {expected_path!r}")
    path = _confined_manifest_path(root, record.get("staged_path"), label=f"{label}.staged_path")
    if not path.is_file():
        raise ContractInputError(f"{label} file is missing: {path}")
    expected_sha = _require_sha256(record.get("sha256"), f"{label}.sha256")
    expected_bytes = _require_int(record.get("bytes"), f"{label}.bytes", minimum=1)
    if file_sha256(path) != expected_sha:
        errors.append(f"{label} SHA-256 does not match its file")
    if path.stat().st_size != expected_bytes:
        errors.append(f"{label} byte count does not match its file")
    return path, errors


def evaluate_candidate(
    *,
    policy_path: Path,
    manifest_path: Path,
    parent_readyz_path: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    policy = load_policy(policy_path)
    identity = policy_file_identity(policy_path, policy)
    errors, evaluation_spec, fullfit_spec = validate_policy_registry(policy)
    workflow_git_sha = checked_out_git_sha()
    errors.extend(
        validate_workflow_git_lineage(policy, workflow_git_sha=workflow_git_sha)
    )
    if fullfit_spec is None or evaluation_spec is None:
        return {"errors": errors, "policy": identity}
    parent_readyz = read_json_object(parent_readyz_path, label="parent production readiness evidence")
    parent_errors, parent = validate_active_lineage(
        parent_readyz,
        policy=policy,
        policy_sha256=identity["sha256"],
        workflow_git_sha=workflow_git_sha,
    )
    errors.extend(parent_errors)

    manifest = read_json_object(manifest_path, label="candidate staging manifest")
    root = manifest_path.resolve(strict=False).parent
    if manifest.get("manifest_version") != 3 or manifest.get("staging_schema_version") != 1:
        errors.append("candidate must use rich manifest_version=3 and staging_schema_version=1")
    contract = policy["contract"]
    if manifest.get("model_spec_name") != contract["fullfit_spec_name"]:
        errors.append("candidate manifest model_spec_name does not match the fixed policy")
    expected_no_odds = f"{contract['fullfit_spec_name']}_no_odds"
    if manifest.get("no_odds_model_spec_name") != expected_no_odds:
        errors.append("candidate manifest no_odds_model_spec_name does not match the fixed policy")

    binding = _require_exact_keys(
        manifest.get("scheduled_refit_policy"),
        _SCHEDULED_MANIFEST_POLICY_KEYS,
        "candidate.scheduled_refit_policy",
    )
    expected_binding = {
        "policy_id": policy["policy_id"],
        "sha256": identity["sha256"],
        "root_bundle_id": policy["root_release"]["bundle_id"],
        "parent_bundle_id": parent.get("bundle_id"),
        "parent_model_spec_name": parent.get("model_spec_name"),
        "parent_model_sha256": _bundle_hash(parent, "model_sha256", "primary_sha256"),
        "parent_no_odds_model_sha256": _bundle_hash(parent, "no_odds_model_sha256", "no_odds_sha256"),
        "parent_logistic_model_sha256": _bundle_hash(parent, "logistic_model_sha256", "logistic_sha256"),
        "parent_processed_fights_sha256": parent.get(
            "immutable_training_fights_sha256"
        ),
        "parent_processed_features_sha256": parent.get(
            "immutable_training_features_sha256"
        ),
    }
    for field, expected_value in expected_binding.items():
        if binding.get(field) != expected_value:
            errors.append(f"candidate scheduled policy binding mismatch: {field}")

    registered = manifest.get("registered_training_specs")
    if not isinstance(registered, dict):
        raise ContractInputError("candidate has no registered_training_specs object")
    expected_records = {
        "selected_evaluation": (asdict(evaluation_spec), contract["evaluation_spec_payload_sha256"]),
        "selected_fullfit": (asdict(fullfit_spec), contract["fullfit_spec_payload_sha256"]),
    }
    for key, (expected_payload, expected_sha) in expected_records.items():
        record = registered.get(key)
        if not isinstance(record, dict):
            errors.append(f"candidate has no registered_training_specs.{key} object")
            continue
        if record.get("payload") != expected_payload or record.get("sha256") != expected_sha:
            errors.append(f"candidate registered_training_specs.{key} does not match policy")
    if registered.get("allowed_differences") != contract["allowed_fullfit_differences"]:
        errors.append("candidate registered allowed_differences does not match policy")

    saved = manifest.get("saved_fullfit_spec")
    expected_sidecar = f"models/{contract['fullfit_spec_name']}_spec.json"
    sidecar_path, sidecar_errors = _validate_file_record(root, saved, label="candidate.saved_fullfit_spec", expected_path=expected_sidecar)
    errors.extend(sidecar_errors)
    saved_payload = read_json_object(sidecar_path, label="candidate full-fit sidecar")
    if not isinstance(saved, dict) or saved.get("payload") != saved_payload:
        errors.append("candidate saved_fullfit_spec payload does not match its sidecar")
    try:
        saved_spec = NamedModelTrainingSpec.from_json(json.dumps(saved_payload))
        if _effective_spec_payload(saved_spec) != _effective_spec_payload(fullfit_spec):
            errors.append("candidate saved full-fit sidecar changed the fixed training contract")
    except (TypeError, ValueError) as exc:
        raise ContractInputError(f"invalid candidate full-fit sidecar: {exc}") from exc

    artifacts = manifest.get("model_artifacts")
    if not isinstance(artifacts, dict):
        raise ContractInputError("candidate has no model_artifacts object")
    artifact_rules = {
        "primary": ("models/xgboost_model.pkl", "model_sha256"),
        "no_odds": ("models/xgboost_no_odds_model.pkl", "no_odds_model_sha256"),
        "logistic": ("models/logistic_model.pkl", "logistic_model_sha256"),
    }
    for role, (staged_path, top_hash) in artifact_rules.items():
        record = artifacts.get(role)
        _path, record_errors = _validate_file_record(root, record, label=f"candidate.model_artifacts.{role}", expected_path=staged_path)
        errors.extend(record_errors)
        if isinstance(record, dict) and manifest.get(top_hash) != record.get("sha256"):
            errors.append(f"candidate {top_hash} does not match model_artifacts.{role}")

    snapshot = manifest.get("immutable_training_snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("immutable") is not True:
        raise ContractInputError("candidate has no immutable training snapshot")
    snapshot_paths: dict[str, Path] = {}
    for role, staged_path in (("fights", "processed/fights_cleaned.csv"), ("features", "processed/features.csv")):
        path, record_errors = _validate_file_record(root, snapshot.get(role), label=f"candidate.immutable_training_snapshot.{role}", expected_path=staged_path)
        snapshot_paths[role] = path
        errors.extend(record_errors)
        top_hash = f"processed_{role}_sha256"
        if isinstance(snapshot.get(role), dict) and manifest.get(top_hash) != snapshot[role].get("sha256"):
            errors.append(f"candidate {top_hash} does not match immutable snapshot")
    features_info, feature_errors = inspect_features(snapshot_paths["features"], policy=policy, fullfit_spec=fullfit_spec, now=now)
    errors.extend(feature_errors)
    if manifest.get("snapshot_max_event_date") != features_info["snapshot_max_event_date"]:
        errors.append("candidate snapshot_max_event_date does not match immutable features")
    candidate_snapshot_date = _require_iso_date(
        manifest.get("snapshot_max_event_date"),
        "candidate.snapshot_max_event_date",
    )
    parent_snapshot_date = _require_iso_date(
        parent.get("immutable_training_snapshot_max_event_date"),
        "parent.immutable_training_snapshot_max_event_date",
    )
    if candidate_snapshot_date < parent_snapshot_date:
        errors.append("candidate snapshot predates the immediate parent snapshot")

    raw_provenance = manifest.get("raw_input_provenance")
    ledger = raw_provenance.get("bfo_ledger") if isinstance(raw_provenance, dict) else None
    if not isinstance(ledger, dict):
        errors.append("candidate has no BFO provenance ledger record")
    else:
        _require_sha256(ledger.get("sha256"), "candidate.raw_input_provenance.bfo_ledger.sha256")
        corrected = ledger.get("corrected_csv_files")
        if not isinstance(corrected, list) or not corrected:
            errors.append("candidate BFO provenance has no corrected CSV records")
        else:
            for index, record in enumerate(corrected):
                if not isinstance(record, dict):
                    raise ContractInputError(f"candidate BFO corrected_csv_files[{index}] must be an object")
                _require_sha256(record.get("sha256"), f"candidate BFO corrected_csv_files[{index}].sha256")
                _require_int(record.get("rows"), f"candidate BFO corrected_csv_files[{index}].rows", minimum=1)
                if ledger.get("provenance_mode") == "scheduled_recovery_batch":
                    _path, record_errors = _validate_file_record(
                        root,
                        record,
                        label=f"candidate BFO corrected_csv_files[{index}]",
                        expected_path=f"provenance/{record.get('path')}",
                    )
                    errors.extend(record_errors)

    lineage = (
        raw_provenance.get("scheduled_bfo_lineage")
        if isinstance(raw_provenance, dict)
        else None
    )
    expected_lineage_keys = {
        "manifest_staged_path",
        "manifest_sha256",
        "manifest_bytes",
        "batch_count",
        "batches",
    }
    if not isinstance(lineage, dict) or set(lineage) != expected_lineage_keys:
        errors.append("candidate has no exact scheduled BFO lineage identity")
    else:
        expected_path = "provenance/bfo_lineage/manifest.json"
        if lineage.get("manifest_staged_path") != expected_path:
            errors.append("candidate scheduled BFO lineage path is not exact")
        lineage_path = _confined_manifest_path(
            root,
            lineage.get("manifest_staged_path"),
            label="candidate scheduled BFO lineage manifest path",
        )
        if not lineage_path.is_file():
            raise ContractInputError(
                f"candidate scheduled BFO lineage manifest is missing: {lineage_path}"
            )
        lineage_sha = _require_sha256(
            lineage.get("manifest_sha256"),
            "candidate scheduled BFO lineage manifest_sha256",
        )
        lineage_bytes = _require_int(
            lineage.get("manifest_bytes"),
            "candidate scheduled BFO lineage manifest_bytes",
            minimum=1,
        )
        if file_sha256(lineage_path) != lineage_sha:
            errors.append("candidate scheduled BFO lineage manifest SHA-256 mismatch")
        if lineage_path.stat().st_size != lineage_bytes:
            errors.append("candidate scheduled BFO lineage manifest byte-count mismatch")
        try:
            lineage_payload = bfo_lineage.validate_package(
                lineage_path,
                expected_manifest_sha256=lineage_sha,
            )
        except (bfo_lineage.BfoLineageError, OSError) as exc:
            raise ContractInputError(
                f"candidate scheduled BFO lineage package is invalid: {exc}"
            ) from exc
        if (
            lineage.get("batches") != lineage_payload.get("batches")
            or lineage.get("batch_count") != len(lineage_payload["batches"])
        ):
            errors.append("candidate scheduled BFO lineage batches do not reconcile")
        parent_lineage_sha = parent.get("scheduled_bfo_lineage_manifest_sha256")
        expected_previous_sha = parent_lineage_sha or None
        if (
            lineage_payload.get("parent_bundle_id") != parent.get("bundle_id")
            or lineage_payload.get("parent_source_manifest_sha256")
            != parent.get("source_manifest_sha256")
            or lineage_payload.get("previous_lineage_manifest_sha256")
            != expected_previous_sha
        ):
            errors.append("candidate scheduled BFO lineage does not chain to its parent")

    return {
        "errors": errors,
        "policy": identity,
        "candidate_manifest": {
            "path": str(manifest_path.resolve(strict=False)),
            "sha256": file_sha256(manifest_path),
            "bundle_id": manifest.get("bundle_id"),
            "model_spec_name": manifest.get("model_spec_name"),
            "features": features_info,
        },
        "parent_bundle_id": parent.get("bundle_id"),
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(encoded)
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def _finish(report: Path, result: dict[str, Any]) -> int:
    errors = list(result.pop("errors", []))
    status = "HEALTHY" if not errors else "UNHEALTHY"
    payload = {"status": status, "healthy": not errors, "errors": errors, **result}
    _write_report(report, payload)
    if errors:
        print("Scheduled refit contract is UNHEALTHY:", file=sys.stderr)
        for error in errors:
            print(f" - {error}", file=sys.stderr)
        return 1
    print("Scheduled refit contract is HEALTHY.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Validate policy, refreshed features, and active lineage")
    preflight.add_argument("--policy", type=Path, required=True)
    preflight.add_argument("--features", type=Path, required=True)
    preflight.add_argument("--readyz-json", type=Path, required=True)
    preflight.add_argument("--report", type=Path, required=True)

    candidate = subparsers.add_parser("candidate", help="Validate a staged candidate against the fixed policy")
    candidate.add_argument("--policy", type=Path, required=True)
    candidate.add_argument("--manifest", type=Path, required=True)
    candidate.add_argument("--parent-readyz-json", type=Path, required=True)
    candidate.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "preflight":
            result = evaluate_preflight(
                policy_path=args.policy,
                features_path=args.features,
                readyz_path=args.readyz_json,
            )
        else:
            result = evaluate_candidate(
                policy_path=args.policy,
                manifest_path=args.manifest,
                parent_readyz_path=args.parent_readyz_json,
            )
        return _finish(args.report, result)
    except Exception as exc:
        failure = {
            "status": "ERROR",
            "healthy": False,
            "errors": [str(exc)],
            "command": args.command,
        }
        try:
            _write_report(args.report, failure)
        except Exception as report_exc:
            print(f"Could not write required contract report: {report_exc}", file=sys.stderr)
        print(f"Scheduled refit contract ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
