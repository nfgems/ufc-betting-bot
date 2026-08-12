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
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.training_spec import NamedModelTrainingSpec, resolve_named_training_spec
from scripts import bfo_lineage
from scripts import recover_post_cutoff_method_odds as method_recovery


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
_PERFORMANCE_CONFIRMATION_KEYS = {
    "schema_version",
    "state",
    "parent_policy_path",
    "parent_policy_sha256",
    "confirmation_result_path",
    "confirmation_result_sha256",
    "global_claim_key",
    "global_claim_path",
    "global_claim_sha256",
    "package_id",
    "candidate_package_contract_sha256",
    "evaluation_spec_name",
    "evaluation_spec_payload_sha256",
    "fullfit_spec_name",
    "fullfit_spec_payload_sha256",
    "strategy_config",
    "strategy_config_sha256",
}
_FINAL_TRACK_C_PASS_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "healthy",
    "final_policy_path",
    "final_policy_sha256",
    "confirmation_result_path",
    "confirmation_result_sha256",
    "model_spec_name",
    "model_spec_payload_sha256",
    "strategy_config_sha256",
    "dataset_fights_path",
    "dataset_fights_sha256",
    "source_dataset_fights_path",
    "source_dataset_fights_sha256",
    "features_path",
    "features_sha256",
    "features_value_sha256",
    "source_features_path",
    "source_features_sha256",
    "feature_contract_count",
    "feature_contract_sha256",
    "confirmation_evaluation_input_value_sha256",
    "odds_source_inventory_path",
    "odds_source_inventory_sha256",
    "source_fingerprint",
    "source_inventory_path",
    "source_inventory_sha256",
    "source_inventory_artifact_sha256",
    "environment_path",
    "environment_artifact_sha256",
    "environment_payload_sha256",
    "protocol_sha256",
    "evaluation_protocol_sha256",
    "evaluation_sample_sha256",
    "artifacts",
    "health_metrics",
    "derived_limits",
}
_FINAL_TRACK_C_ARTIFACT_KEYS = {
    "summary",
    "evaluation_index",
    "data_quality",
    "predictive",
    "production_bets",
}
_FINAL_TRACK_C_ARTIFACT_RECORD_KEYS = {"path", "sha256"}
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
_CONTRACT_V2_KEYS = _CONTRACT_KEYS | {
    "method_contract_decision_path",
    "method_contract_decision_sha256",
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
_EVALUATION_V2_KEYS = _EVALUATION_KEYS | {
    "require_entry_odds",
    "quality_recent_window_days",
    "quality_minimum_recent_rows",
    "quality_minimum_entry_coverage",
    "quality_maximum_entry_lag_days",
    "quality_allowed_prefight_sources",
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
    "policy_schema_version",
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
    """Load and strictly validate the complete scheduled-refit policy."""

    policy = read_json_object(path, label="scheduled refit policy")
    policy_keys = set(policy)
    allowed_policy_keys = _TOP_LEVEL_KEYS | {"performance_confirmation"}
    if policy_keys != _TOP_LEVEL_KEYS and policy_keys != allowed_policy_keys:
        _require_exact_keys(policy, _TOP_LEVEL_KEYS, "policy")
    schema_version = policy["schema_version"]
    if schema_version not in {1, 2} or isinstance(schema_version, bool):
        raise ContractInputError("policy.schema_version must be 1 or 2")
    _require_string(policy["policy_id"], "policy.policy_id")

    if "performance_confirmation" in policy:
        confirmation = _require_exact_keys(
            policy["performance_confirmation"],
            _PERFORMANCE_CONFIRMATION_KEYS,
            "policy.performance_confirmation",
        )
        if confirmation["schema_version"] != 1:
            raise ContractInputError(
                "policy.performance_confirmation.schema_version must be exactly 1"
            )
        if confirmation["state"] not in {
            "bootstrap_pending_track_c",
            "final_track_c_bound",
        }:
            raise ContractInputError(
                "policy.performance_confirmation.state must be "
                "'bootstrap_pending_track_c' or 'final_track_c_bound'"
            )
        for field in (
            "parent_policy_path",
            "confirmation_result_path",
            "global_claim_key",
            "global_claim_path",
            "package_id",
            "evaluation_spec_name",
            "fullfit_spec_name",
        ):
            _require_string(
                confirmation[field], f"policy.performance_confirmation.{field}"
            )
        for field in (
            "parent_policy_sha256",
            "confirmation_result_sha256",
            "global_claim_sha256",
            "candidate_package_contract_sha256",
            "evaluation_spec_payload_sha256",
            "fullfit_spec_payload_sha256",
            "strategy_config_sha256",
        ):
            _require_sha256(
                confirmation[field], f"policy.performance_confirmation.{field}"
            )
        from src.strategy.runtime_strategy import validate_locked_strategy_config

        try:
            strategy = validate_locked_strategy_config(
                confirmation["strategy_config"]
            )
        except ValueError as exc:
            raise ContractInputError(
                f"policy.performance_confirmation.strategy_config is invalid: {exc}"
            ) from exc
        if canonical_json_sha256(strategy) != confirmation["strategy_config_sha256"]:
            raise ContractInputError(
                "policy.performance_confirmation strategy_config hash does not reproduce"
            )

    contract_keys = _CONTRACT_V2_KEYS if schema_version == 2 else _CONTRACT_KEYS
    contract = _require_exact_keys(policy["contract"], contract_keys, "policy.contract")
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
    if schema_version == 2:
        decision_path = Path(
            _require_string(
                contract["method_contract_decision_path"],
                "policy.contract.method_contract_decision_path",
            )
        )
        if decision_path.is_absolute() or ".." in decision_path.parts:
            raise ContractInputError(
                "policy.contract.method_contract_decision_path must be a safe relative path"
            )
        _require_sha256(
            contract["method_contract_decision_sha256"],
            "policy.contract.method_contract_decision_sha256",
        )

    evaluation_keys = _EVALUATION_V2_KEYS if schema_version == 2 else _EVALUATION_KEYS
    evaluation = _require_exact_keys(policy["evaluation"], evaluation_keys, "policy.evaluation")
    for field in ("model_seed", "odds_noise_seed", "retrain_months", "initial_train_years", "min_train_test_fights", "minimum_fighter_fights"):
        _require_int(evaluation[field], f"policy.evaluation.{field}", minimum=1)
    _require_iso_date(evaluation["bet_start_date"], "policy.evaluation.bet_start_date")
    if evaluation["execution_mode"] != "realistic":
        raise ContractInputError("policy.evaluation.execution_mode must be 'realistic'")
    if _require_number(evaluation["entry_offset_days"], "policy.evaluation.entry_offset_days", minimum=0.0) != 1.0:
        raise ContractInputError("policy.evaluation.entry_offset_days must be exactly 1.0")
    if not _require_bool(evaluation["entry_offset_for_features"], "policy.evaluation.entry_offset_for_features"):
        raise ContractInputError("policy must feed the T-1 entry snapshot into model features")
    if schema_version == 2:
        if not _require_bool(evaluation["require_entry_odds"], "policy.evaluation.require_entry_odds"):
            raise ContractInputError("policy v2 must require verified entry odds")
        if _require_int(
            evaluation["quality_recent_window_days"],
            "policy.evaluation.quality_recent_window_days",
            minimum=1,
        ) != 60:
            raise ContractInputError("policy v2 recent quality window must be exactly 60 days")
        if _require_int(
            evaluation["quality_minimum_recent_rows"],
            "policy.evaluation.quality_minimum_recent_rows",
            minimum=1,
        ) != 50:
            raise ContractInputError("policy v2 must require exactly 50 recent rows")
        if _require_number(
            evaluation["quality_minimum_entry_coverage"],
            "policy.evaluation.quality_minimum_entry_coverage",
            minimum=0.0,
        ) != 0.70:
            raise ContractInputError("policy v2 entry coverage must be exactly 0.70")
        if _require_number(
            evaluation["quality_maximum_entry_lag_days"],
            "policy.evaluation.quality_maximum_entry_lag_days",
            minimum=0.0,
        ) != 35.0:
            raise ContractInputError("policy v2 maximum entry lag must be exactly 35 days")
        if evaluation["quality_allowed_prefight_sources"] != ["line_history", "odds_api"]:
            raise ContractInputError(
                "policy v2 allowed pre-fight sources must be exactly "
                "['line_history', 'odds_api']"
            )
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
    performance_state = (
        policy.get("performance_confirmation", {}).get("state")
        if isinstance(policy.get("performance_confirmation"), dict)
        else None
    )
    bootstrap_pending = performance_state == "bootstrap_pending_track_c"
    expected_comparison_role = (
        "confirmed_winner_track_c_baseline"
        if performance_state == "final_track_c_bound"
        else "root_release_reference_thresholds"
    )
    if baseline["comparison_role"] != expected_comparison_role:
        raise ContractInputError(
            "policy.baseline.comparison_role must be " + repr(expected_comparison_role)
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
    if (
        not bootstrap_pending
        and baseline["evaluation_spec_name"] != contract["evaluation_spec_name"]
    ):
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
    if (
        not bootstrap_pending
        and baseline["scheduled_protocol_sha256"] != scheduled_protocol_sha256(policy)
    ):
        raise ContractInputError(
            "policy baseline scheduled_protocol_sha256 does not match the fixed protocol"
        )
    if (
        not bootstrap_pending
        and baseline["evidence_protocol_sha256"] != scheduled_protocol_sha256(policy)
    ):
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


def _resolve_durable_evidence_path(
    value: Any,
    *,
    label: str,
    repo_root: Path,
) -> Path:
    """Resolve one canonical repository-relative file below ``evidence/``."""

    raw_value = _require_string(value, label)
    if "\\" in raw_value:
        raise ContractInputError(f"{label} must use repository-relative POSIX separators")
    raw = Path(raw_value)
    if raw.is_absolute() or ".." in raw.parts or raw.as_posix() != raw_value:
        raise ContractInputError(f"{label} must be a canonical repository-relative path")
    root = repo_root.resolve(strict=False)
    evidence_root = (root / "evidence").resolve(strict=False)
    resolved = (root / raw).resolve(strict=False)
    try:
        resolved.relative_to(evidence_root)
    except ValueError as exc:
        raise ContractInputError(f"{label} must be preserved below evidence/") from exc
    if not resolved.is_file():
        raise ContractInputError(f"{label} is missing: {resolved}")
    return resolved


def _performance_transition_errors(
    *,
    parent: dict[str, Any],
    child: dict[str, Any],
    strategy_config: dict[str, Any],
) -> list[str]:
    """Validate the narrowly permitted pre-winner -> confirmed-child transition."""

    errors: list[str] = []
    if "performance_confirmation" in parent:
        errors.append("immutable parent policy must be the pre-winner policy")
    for field in ("schema_version", "policy_id", "health_limits", "root_release"):
        if child.get(field) != parent.get(field):
            errors.append(f"confirmed child changed immutable policy field: {field}")

    mutable_contract = {
        "evaluation_spec_name",
        "evaluation_spec_payload_sha256",
        "fullfit_spec_name",
        "fullfit_spec_payload_sha256",
    }
    for field, parent_value in parent["contract"].items():
        if field not in mutable_contract and child["contract"].get(field) != parent_value:
            errors.append(f"confirmed child changed immutable contract field: {field}")

    strategy_evaluation_bindings = {
        "min_edge": "s_min_edge",
        "blend_weight": "s_blend_weight",
        "model_agreement_min_edge": "s_model_agreement_min_edge",
    }
    for field, parent_value in parent["evaluation"].items():
        if field not in strategy_evaluation_bindings and child["evaluation"].get(field) != parent_value:
            errors.append(f"confirmed child changed immutable evaluation field: {field}")
    for policy_field, strategy_field in strategy_evaluation_bindings.items():
        if child["evaluation"].get(policy_field) != strategy_config[strategy_field]:
            errors.append(
                "confirmed child evaluation/strategy mismatch: "
                f"{policy_field} != {strategy_field}"
            )

    if (
        child["performance_confirmation"]["state"]
        == "bootstrap_pending_track_c"
    ):
        if child["baseline"] != parent["baseline"]:
            errors.append(
                "bootstrap winner policy must retain the exact frozen parent baseline"
            )
        return errors

    immutable_baseline = _BASELINE_KEYS - {
        "comparison_role",
        "evidence_path",
        "evidence_sha256",
        "evaluation_spec_name",
        "evidence_protocol_sha256",
        "scheduled_protocol_sha256",
        "model_brier_score",
        "model_ece",
        "strategy_total_bets",
        "strategy_roi",
        "strategy_total_profit",
        "strategy_avg_clv",
        "strategy_max_drawdown_pct",
    }
    for field in immutable_baseline:
        if child["baseline"].get(field) != parent["baseline"].get(field):
            errors.append(f"confirmed child changed immutable baseline field: {field}")

    baseline = child["baseline"]
    limits = child["health_limits"]
    parent_baseline = parent["baseline"]
    if baseline["evaluation_n_fights"] < limits["minimum_evaluation_fights"]:
        errors.append("confirmed child baseline has too few evaluation fights")
    if baseline["evaluation_n_folds"] < limits["minimum_evaluation_folds"]:
        errors.append("confirmed child baseline has too few evaluation folds")
    if baseline["strategy_total_bets"] < limits["minimum_strategy_bets"]:
        errors.append("confirmed child baseline has too few strategy bets")
    if not (
        baseline["strategy_roi"] > 0.0
        and baseline["strategy_roi"] >= limits["minimum_strategy_roi"]
    ):
        errors.append("confirmed child baseline ROI is not strictly positive and healthy")
    if not (
        baseline["strategy_total_profit"] > 0.0
        and baseline["strategy_total_profit"]
        >= limits["minimum_strategy_total_profit"]
    ):
        errors.append(
            "confirmed child baseline total profit is not strictly positive and healthy"
        )
    if baseline["model_brier_score"] > parent_baseline["model_brier_score"] * (
        1.0 + limits["maximum_brier_relative_regression"]
    ):
        errors.append("confirmed child baseline Brier exceeds the frozen health limit")
    if baseline["model_ece"] > (
        parent_baseline["model_ece"] + limits["maximum_ece_absolute_regression"]
    ):
        errors.append("confirmed child baseline ECE exceeds the frozen health limit")
    if baseline["strategy_max_drawdown_pct"] > (
        parent_baseline["strategy_max_drawdown_pct"]
        * limits["maximum_drawdown_multiplier"]
    ):
        errors.append("confirmed child baseline drawdown exceeds the frozen health limit")
    if baseline["strategy_avg_clv"] < (
        parent_baseline["strategy_avg_clv"]
        - limits["maximum_clv_absolute_regression"]
    ):
        errors.append("confirmed child baseline CLV exceeds the frozen regression limit")
    return errors


def validate_performance_confirmation(
    policy: dict[str, Any],
    *,
    repo_root: Path | None = None,
    confirmation_validator: Any | None = None,
) -> tuple[list[str], dict[str, Any] | None]:
    """Validate a durable PASS and its strict post-confirmation child policy."""

    confirmation = policy.get("performance_confirmation")
    if confirmation is None:
        return [], None
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    errors: list[str] = []
    try:
        parent_path = _resolve_durable_evidence_path(
            confirmation["parent_policy_path"],
            label="policy.performance_confirmation.parent_policy_path",
            repo_root=root,
        )
        result_path = _resolve_durable_evidence_path(
            confirmation["confirmation_result_path"],
            label="policy.performance_confirmation.confirmation_result_path",
            repo_root=root,
        )
        claim_path = _resolve_durable_evidence_path(
            confirmation["global_claim_path"],
            label="policy.performance_confirmation.global_claim_path",
            repo_root=root,
        )
        if file_sha256(parent_path) != confirmation["parent_policy_sha256"]:
            raise ContractInputError("immutable parent policy hash changed")
        if file_sha256(result_path) != confirmation["confirmation_result_sha256"]:
            raise ContractInputError("authoritative confirmation result hash changed")
        if file_sha256(claim_path) != confirmation["global_claim_sha256"]:
            raise ContractInputError("global confirmation claim hash changed")
        claim_root = (root / "evidence" / "confirmation_claims").resolve(strict=False)
        try:
            relative_result = result_path.relative_to(claim_root)
            relative_claim = claim_path.relative_to(claim_root)
        except ValueError as exc:
            raise ContractInputError(
                "confirmation result and claim must be below evidence/confirmation_claims/"
            ) from exc
        expected_key = confirmation["global_claim_key"]
        if (
            len(relative_result.parts) < 2
            or relative_result.parts[0] != expected_key
            or relative_result.name != "confirmation_evaluation_result.json"
            or len(relative_claim.parts) < 2
            or relative_claim.parts[0] != expected_key
            or relative_claim.name != "confirmation_evaluation_claim.json"
        ):
            raise ContractInputError(
                "confirmation result/claim paths do not reproduce the global claim key"
            )

        parent = load_policy(parent_path)
        # DIR-AUD-P2-024: load_policy never opens the parent's baseline
        # evidence CSV, so a child could otherwise bind health anchors that
        # were measured under a stale parent protocol. Reuse the single strict
        # baseline-evidence validator against the preserved parent.
        from scripts.check_scheduled_refit_quality import (
            _validate_v2_baseline_evidence,
        )

        parent_baseline_errors = _validate_v2_baseline_evidence(
            parent, repo_root=root
        )
        if parent_baseline_errors:
            errors.append(
                "immutable parent policy baseline evidence is stale: "
                + "; ".join(parent_baseline_errors)
            )
        result = read_json_object(result_path, label="authoritative confirmation result")
        claim = read_json_object(claim_path, label="global confirmation claim")
        if confirmation_validator is None:
            from src.strategy.run_evaluation import (
                validate_confirmation_result_for_promotion,
            )

            confirmation_validator = validate_confirmation_result_for_promotion
        validated = confirmation_validator(result_path)
        if not isinstance(validated, dict):
            raise ContractInputError("confirmation validator returned no package binding")

        expected_binding = {
            "package_id": validated.get("package_id"),
            "candidate_package_contract_sha256": validated.get(
                "package_contract_sha256"
            ),
            "evaluation_spec_name": validated.get("model_spec_name"),
            "evaluation_spec_payload_sha256": validated.get(
                "model_spec_payload_sha256"
            ),
            "fullfit_spec_name": validated.get("fullfit_model_spec_name"),
            "fullfit_spec_payload_sha256": validated.get(
                "fullfit_model_spec_payload_sha256"
            ),
            "strategy_config": validated.get("strategy_config"),
            "strategy_config_sha256": validated.get("strategy_config_sha256"),
        }
        for field, expected in expected_binding.items():
            if confirmation.get(field) != expected:
                errors.append(f"performance confirmation PASS binding mismatch: {field}")
        if validated.get("runtime_invariants") != {"near_miss_enabled": False}:
            errors.append("confirmed package does not disable the near-miss runtime path")
        if validated.get("policy_sha256") != confirmation["parent_policy_sha256"]:
            errors.append("PASS result is not bound to the immutable parent policy")
        if result.get("global_claim_key") != confirmation["global_claim_key"]:
            errors.append("authoritative result global claim key differs")
        if result.get("claim_path") != confirmation["global_claim_path"]:
            errors.append("authoritative result global claim path differs")
        if result.get("claim_sha256") != confirmation["global_claim_sha256"]:
            errors.append("authoritative result global claim hash differs")
        if claim.get("global_claim_key") != confirmation["global_claim_key"]:
            errors.append("global claim artifact key differs")

        contract = policy["contract"]
        for field in (
            "evaluation_spec_name",
            "evaluation_spec_payload_sha256",
            "fullfit_spec_name",
            "fullfit_spec_payload_sha256",
        ):
            if contract.get(field) != confirmation.get(field):
                errors.append(f"confirmed child contract does not match PASS: {field}")
        errors.extend(
            _performance_transition_errors(
                parent=parent,
                child=policy,
                strategy_config=confirmation["strategy_config"],
            )
        )
        baseline_path = _resolve_durable_evidence_path(
            policy["baseline"]["evidence_path"],
            label="policy.baseline.evidence_path",
            repo_root=root,
        )
        if file_sha256(baseline_path) != policy["baseline"]["evidence_sha256"]:
            errors.append("confirmed child baseline evidence hash changed")
    except (ContractInputError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid performance confirmation: {exc}")
        return errors, None
    return errors, validated


def _resolve_canonical_repo_file(
    value: Any,
    *,
    label: str,
    repo_root: Path,
    require_evidence: bool = False,
) -> Path:
    raw_value = _require_string(value, label)
    if "\\" in raw_value:
        raise ContractInputError(f"{label} must use POSIX separators")
    raw = Path(raw_value)
    if raw.is_absolute() or ".." in raw.parts or raw.as_posix() != raw_value:
        raise ContractInputError(f"{label} must be a canonical repository-relative path")
    root = repo_root.resolve(strict=False)
    resolved = (root / raw).resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ContractInputError(f"{label} escapes the repository") from exc
    if require_evidence and (not relative.parts or relative.parts[0] != "evidence"):
        raise ContractInputError(f"{label} must be durable below evidence/")
    if not resolved.is_file():
        raise ContractInputError(f"{label} is missing: {resolved}")
    return resolved


def validate_final_track_c_pass_receipt(
    receipt_path: Path,
    *,
    expected_policy_path: Path | None = None,
    expected_fullfit_spec: NamedModelTrainingSpec | None = None,
    repo_root: Path | None = None,
    quality_validator: Any | None = None,
    confirmation_validator: Any | None = None,
) -> dict[str, Any]:
    """Recheck the post-freeze Track-C PASS required before any full fit.

    The receipt lives outside the policy because the final policy hash cannot be
    embedded in the policy itself. Every load-bearing Track-C artifact is
    rehashed, and the health gate is rerun against the exact final child bytes.
    """

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    raw_receipt_path = Path(receipt_path)
    receipt_path = (
        raw_receipt_path.resolve(strict=False)
        if raw_receipt_path.is_absolute()
        else (root / raw_receipt_path).resolve(strict=False)
    )
    evidence_root = (root / "evidence").resolve(strict=False)
    try:
        receipt_path.relative_to(evidence_root)
    except ValueError as exc:
        raise ContractInputError(
            "final Track-C PASS receipt must be durable below evidence/"
        ) from exc
    if not receipt_path.is_file():
        raise ContractInputError(f"final Track-C PASS receipt is missing: {receipt_path}")
    receipt = _require_exact_keys(
        read_json_object(receipt_path, label="final Track-C PASS receipt"),
        _FINAL_TRACK_C_PASS_RECEIPT_KEYS,
        "final Track-C PASS receipt",
    )
    if (
        receipt["schema_version"] != 1
        or receipt["status"] != "PASS"
        or receipt["healthy"] is not True
    ):
        raise ContractInputError("final Track-C receipt is not an explicit healthy PASS")
    for field in (
        "final_policy_sha256",
        "confirmation_result_sha256",
        "model_spec_payload_sha256",
        "strategy_config_sha256",
        "dataset_fights_sha256",
        "source_dataset_fights_sha256",
        "features_sha256",
        "features_value_sha256",
        "source_features_sha256",
        "feature_contract_sha256",
        "confirmation_evaluation_input_value_sha256",
        "odds_source_inventory_sha256",
        "source_fingerprint",
        "source_inventory_sha256",
        "source_inventory_artifact_sha256",
        "environment_artifact_sha256",
        "environment_payload_sha256",
        "protocol_sha256",
        "evaluation_protocol_sha256",
        "evaluation_sample_sha256",
    ):
        _require_sha256(receipt[field], f"final Track-C receipt.{field}")

    policy_path = _resolve_canonical_repo_file(
        receipt["final_policy_path"],
        label="final Track-C receipt.final_policy_path",
        repo_root=root,
    )
    if expected_policy_path is not None and policy_path != Path(
        expected_policy_path
    ).resolve(strict=False):
        raise ContractInputError("final Track-C receipt binds a different policy path")
    if file_sha256(policy_path) != receipt["final_policy_sha256"]:
        raise ContractInputError("final Track-C receipt policy hash is stale")
    policy = load_policy(policy_path)
    if "performance_confirmation" not in policy:
        raise ContractInputError(
            "final Track-C receipt policy has no locked performance confirmation"
        )
    if policy["performance_confirmation"].get("state") != "final_track_c_bound":
        raise ContractInputError(
            "only final_track_c_bound policy state can authorize a full fit"
        )
    registry_errors, evaluation_spec, fullfit_spec = validate_policy_registry(
        policy,
        repo_root=root,
    )
    if registry_errors or evaluation_spec is None or fullfit_spec is None:
        raise ContractInputError(
            "final Track-C receipt policy is invalid: " + "; ".join(registry_errors)
        )
    if expected_fullfit_spec is not None and (
        fullfit_spec.name != expected_fullfit_spec.name
        or canonical_json_sha256(asdict(fullfit_spec))
        != canonical_json_sha256(asdict(expected_fullfit_spec))
    ):
        raise ContractInputError("final Track-C receipt full-fit spec differs from trainer")

    confirmation = policy["performance_confirmation"]
    performance_errors, performance_binding = validate_performance_confirmation(
        policy,
        repo_root=root,
        confirmation_validator=confirmation_validator,
    )
    if performance_errors or not isinstance(performance_binding, dict):
        raise ContractInputError(
            "final Track-C receipt performance confirmation is invalid: "
            + "; ".join(performance_errors)
        )
    confirmation_path = _resolve_canonical_repo_file(
        receipt["confirmation_result_path"],
        label="final Track-C receipt.confirmation_result_path",
        repo_root=root,
        require_evidence=True,
    )
    if (
        receipt["confirmation_result_path"]
        != confirmation["confirmation_result_path"]
        or receipt["confirmation_result_sha256"]
        != confirmation["confirmation_result_sha256"]
        or file_sha256(confirmation_path) != receipt["confirmation_result_sha256"]
    ):
        raise ContractInputError("final Track-C receipt confirmation result is stale")
    expected_scalar_bindings = {
        "model_spec_name": confirmation["evaluation_spec_name"],
        "model_spec_payload_sha256": confirmation[
            "evaluation_spec_payload_sha256"
        ],
        "strategy_config_sha256": confirmation["strategy_config_sha256"],
        "protocol_sha256": policy["baseline"]["scheduled_protocol_sha256"],
        "evaluation_protocol_sha256": performance_binding[
            "evaluation_protocol_sha256"
        ],
        "evaluation_sample_sha256": policy["baseline"][
            "evaluation_sample_sha256"
        ],
        "feature_contract_count": performance_binding["feature_contract_count"],
        "feature_contract_sha256": performance_binding["feature_contract_sha256"],
        "confirmation_evaluation_input_value_sha256": performance_binding[
            "confirmation_evaluation_input_value_sha256"
        ],
        "features_value_sha256": performance_binding["features_value_sha256"],
        "source_fingerprint": performance_binding["source_fingerprint"],
        "source_inventory_sha256": performance_binding["source_inventory_sha256"],
        "environment_payload_sha256": performance_binding[
            "environment_payload_sha256"
        ],
    }
    for field, expected in expected_scalar_bindings.items():
        if receipt[field] != expected:
            raise ContractInputError(f"final Track-C receipt binding mismatch: {field}")

    dataset_fights_path = _resolve_canonical_repo_file(
        receipt["dataset_fights_path"],
        label="final Track-C receipt.dataset_fights_path",
        repo_root=root,
        require_evidence=True,
    )
    source_dataset_fights_path = _resolve_canonical_repo_file(
        receipt["source_dataset_fights_path"],
        label="final Track-C receipt.source_dataset_fights_path",
        repo_root=root,
        require_evidence=True,
    )
    for prefix, path in (
        ("dataset_fights", dataset_fights_path),
        ("source_dataset_fights", source_dataset_fights_path),
    ):
        if (
            receipt[f"{prefix}_path"] != performance_binding.get(f"{prefix}_path")
            or receipt[f"{prefix}_sha256"]
            != performance_binding.get(f"{prefix}_sha256")
            or file_sha256(path) != receipt[f"{prefix}_sha256"]
        ):
            raise ContractInputError(
                f"final Track-C receipt {prefix} binding is stale"
            )
    if (
        dataset_fights_path != source_dataset_fights_path
        or receipt["dataset_fights_sha256"]
        != receipt["source_dataset_fights_sha256"]
    ):
        raise ContractInputError(
            "final Track-C receipt evaluated fights differ from the preserved source"
        )

    immutable_input_bindings = (
        (
            "features",
            "features_path",
            "features_sha256",
            "features_artifact_path",
            "features_artifact_sha256",
            False,
        ),
        (
            "source features",
            "source_features_path",
            "source_features_sha256",
            "source_features_path",
            "source_features_sha256",
            True,
        ),
        (
            "odds-source inventory",
            "odds_source_inventory_path",
            "odds_source_inventory_sha256",
            "odds_source_inventory_path",
            "odds_source_inventory_sha256",
            True,
        ),
        (
            "source inventory",
            "source_inventory_path",
            "source_inventory_artifact_sha256",
            "source_inventory_path",
            "source_inventory_artifact_sha256",
            True,
        ),
        (
            "environment inventory",
            "environment_path",
            "environment_artifact_sha256",
            "environment_path",
            "environment_artifact_sha256",
            True,
        ),
    )
    resolved_inputs: dict[str, Path] = {}
    for (
        label,
        receipt_path_field,
        receipt_sha_field,
        performance_path_field,
        performance_sha_field,
        require_evidence,
    ) in immutable_input_bindings:
        path = _resolve_canonical_repo_file(
            receipt[receipt_path_field],
            label=f"final Track-C receipt.{receipt_path_field}",
            repo_root=root,
            require_evidence=require_evidence,
        )
        if (
            receipt[receipt_path_field] != performance_binding.get(performance_path_field)
            or receipt[receipt_sha_field]
            != performance_binding.get(performance_sha_field)
            or file_sha256(path) != receipt[receipt_sha_field]
        ):
            raise ContractInputError(f"final Track-C receipt {label} binding is stale")
        resolved_inputs[receipt_path_field] = path
    features_path = resolved_inputs["features_path"]
    if receipt["features_sha256"] != policy["baseline"]["features_sha256"]:
        raise ContractInputError("final Track-C receipt features differ from final policy")

    artifacts = _require_exact_keys(
        receipt["artifacts"],
        _FINAL_TRACK_C_ARTIFACT_KEYS,
        "final Track-C receipt.artifacts",
    )
    artifact_paths: dict[str, Path] = {}
    for name, record_value in artifacts.items():
        record = _require_exact_keys(
            record_value,
            _FINAL_TRACK_C_ARTIFACT_RECORD_KEYS,
            f"final Track-C receipt.artifacts.{name}",
        )
        expected_sha = _require_sha256(
            record["sha256"], f"final Track-C receipt.artifacts.{name}.sha256"
        )
        path = _resolve_canonical_repo_file(
            record["path"],
            label=f"final Track-C receipt.artifacts.{name}.path",
            repo_root=root,
            require_evidence=True,
        )
        if file_sha256(path) != expected_sha:
            raise ContractInputError(f"final Track-C artifact changed: {name}")
        artifact_paths[name] = path
    if len(set(artifact_paths.values())) != len(artifact_paths):
        raise ContractInputError("final Track-C artifact paths are not unique")
    artifact_dir = artifact_paths["summary"].parent
    expected_names = {
        "summary": "track_c_summary.csv",
        "evaluation_index": f"{evaluation_spec.name}_evaluation_index.csv",
        "data_quality": f"{evaluation_spec.name}_data_quality.csv",
        "predictive": f"{evaluation_spec.name}_predictive.json",
        "production_bets": f"{evaluation_spec.name}_production_gated_bets.csv",
    }
    for name, expected_name in expected_names.items():
        if artifact_paths[name].parent != artifact_dir or artifact_paths[name].name != expected_name:
            raise ContractInputError(
                f"final Track-C artifact path is not canonical for {name}"
            )

    summary = pd.read_csv(artifact_paths["summary"])
    if len(summary) != 1:
        raise ContractInputError("final Track-C summary must contain exactly one row")
    summary_row = summary.iloc[0]
    if str(summary_row.get("strategy_config_sha256") or "").strip() != receipt[
        "strategy_config_sha256"
    ]:
        raise ContractInputError(
            "final Track-C summary does not bind the confirmed strategy config"
        )
    exact_summary = {
        "policy_sha256": receipt["final_policy_sha256"],
        "spec": receipt["model_spec_name"],
        "spec_payload_sha256": receipt["model_spec_payload_sha256"],
        "features_sha256": receipt["features_sha256"],
        "protocol_sha256": receipt["protocol_sha256"],
        "evaluation_sample_sha256": receipt["evaluation_sample_sha256"],
    }
    for field, expected in exact_summary.items():
        if str(summary_row.get(field) or "").strip() != str(expected):
            raise ContractInputError(f"final Track-C summary binding mismatch: {field}")
    baseline_metric_fields = (
        "model_brier_score",
        "model_ece",
        "strategy_total_bets",
        "strategy_roi",
        "strategy_total_profit",
        "strategy_avg_clv",
        "strategy_max_drawdown_pct",
    )
    for field in baseline_metric_fields:
        actual = _require_number(summary_row.get(field), f"final Track-C summary.{field}")
        expected = float(policy["baseline"][field])
        if actual != expected:
            raise ContractInputError(
                f"final Track-C rerun does not reproduce frozen baseline metric: {field}"
            )

    if quality_validator is None:
        from scripts.check_scheduled_refit_quality import evaluate_quality

        quality_validator = evaluate_quality
    quality = quality_validator(
        policy_path=policy_path,
        features_path=features_path,
        artifacts_dir=artifact_dir,
    )
    if not isinstance(quality, dict) or quality.get("errors") != []:
        raise ContractInputError("final Track-C health gate is not a clean PASS")
    if quality.get("metrics") != receipt["health_metrics"]:
        raise ContractInputError("final Track-C receipt health metrics are stale")
    if quality.get("derived_limits") != receipt["derived_limits"]:
        raise ContractInputError("final Track-C receipt derived limits are stale")
    return {
        "receipt_path": receipt_path,
        "receipt_sha256": file_sha256(receipt_path),
        "policy_path": policy_path,
        "policy_sha256": receipt["final_policy_sha256"],
        "confirmation_result_path": confirmation_path,
        "confirmation_result_sha256": receipt["confirmation_result_sha256"],
        "evaluation_spec": evaluation_spec,
        "fullfit_spec": fullfit_spec,
        "strategy_config": confirmation["strategy_config"],
        "strategy_config_sha256": receipt["strategy_config_sha256"],
        "dataset_fights_path": dataset_fights_path,
        "dataset_fights_sha256": receipt["dataset_fights_sha256"],
        "source_dataset_fights_path": source_dataset_fights_path,
        "source_dataset_fights_sha256": receipt[
            "source_dataset_fights_sha256"
        ],
        "features_path": features_path,
        "features_sha256": receipt["features_sha256"],
        "features_value_sha256": receipt["features_value_sha256"],
        "source_features_path": resolved_inputs["source_features_path"],
        "source_features_sha256": receipt["source_features_sha256"],
        "feature_contract_count": receipt["feature_contract_count"],
        "feature_contract_sha256": receipt["feature_contract_sha256"],
        "confirmation_evaluation_input_value_sha256": receipt[
            "confirmation_evaluation_input_value_sha256"
        ],
        "odds_source_inventory_path": resolved_inputs[
            "odds_source_inventory_path"
        ],
        "odds_source_inventory_sha256": receipt["odds_source_inventory_sha256"],
        "source_fingerprint": receipt["source_fingerprint"],
        "source_inventory_path": resolved_inputs["source_inventory_path"],
        "source_inventory_sha256": receipt["source_inventory_sha256"],
        "source_inventory_artifact_sha256": receipt[
            "source_inventory_artifact_sha256"
        ],
        "environment_path": resolved_inputs["environment_path"],
        "environment_artifact_sha256": receipt["environment_artifact_sha256"],
        "environment_payload_sha256": receipt["environment_payload_sha256"],
        "evaluation_protocol_sha256": receipt["evaluation_protocol_sha256"],
        "artifacts": artifact_paths,
        "health_metrics": receipt["health_metrics"],
    }


def scheduled_protocol(policy: dict[str, Any]) -> dict[str, Any]:
    """Return the complete fixed evaluation protocol recorded by Track C."""

    contract = policy["contract"]
    protocol = {
        "schema_version": policy["schema_version"],
        "policy_id": policy["policy_id"],
        "evaluation_spec_name": contract["evaluation_spec_name"],
        "evaluation_spec_payload_sha256": contract["evaluation_spec_payload_sha256"],
        "evaluation": policy["evaluation"],
    }
    if policy["schema_version"] == 2:
        protocol["method_contract_decision_sha256"] = contract[
            "method_contract_decision_sha256"
        ]
    return protocol


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


def _validate_method_contract_descendant(
    selected: NamedModelTrainingSpec,
    base: NamedModelTrainingSpec,
    *,
    label: str,
    expected_model_seed: int,
    expected_odds_noise_seed: int,
) -> list[str]:
    """Keep the method decision as the immutable feature/data base contract.

    A performance-confirmed descendant may change registered fit/calibration
    metadata, but it cannot change the ordered feature contract, data window,
    T-1 transform/noise semantics, or deterministic seeds selected before
    model metrics were consulted.
    """
    errors: list[str] = []
    if list(selected.feature_cols) != list(base.feature_cols):
        errors.append(f"{label} descendant changes the method-selected feature columns")
    invariant_fields = (
        "dataset_variant",
        "train_start_date",
        "train_end_date",
        "train_cutoff_date",
        "odds_noise_std",
        "odds_noise_seed",
        "odds_noise_mode",
        "add_rematch_features",
        "add_line_movement",
        "impute_strategy",
        "impute_with_indicators",
    )
    changed = [
        field
        for field in invariant_fields
        if getattr(selected, field) != getattr(base, field)
    ]
    if changed:
        errors.append(
            f"{label} descendant changes frozen data/T-1/noise semantics: {changed}"
        )
    selected_seed = (selected.xgb_params or {}).get("random_state")
    base_seed = (base.xgb_params or {}).get("random_state")
    if selected_seed != base_seed or selected_seed != expected_model_seed:
        errors.append(f"{label} descendant changes the fixed model seed")
    if selected.odds_noise_seed != expected_odds_noise_seed:
        errors.append(f"{label} descendant changes the fixed odds-noise seed")
    return errors


def validate_policy_registry(
    policy: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> tuple[
    list[str],
    NamedModelTrainingSpec | None,
    NamedModelTrainingSpec | None,
]:
    errors: list[str] = []
    policy_root = Path(repo_root) if repo_root is not None else REPO_ROOT
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
    if policy["schema_version"] == 2:
        decision_path = policy_root / contract["method_contract_decision_path"]
        if not decision_path.is_file():
            errors.append(f"method-contract decision artifact is missing: {decision_path}")
        elif file_sha256(decision_path) != contract["method_contract_decision_sha256"]:
            errors.append("method-contract decision artifact hash does not match the policy")
        else:
            try:
                decision = read_json_object(decision_path, label="method-contract decision")
                required_passes = ["revalidate", "direct", "playwright", "wayback"]
                targets_total = int(decision.get("targets_total", -1))
                targets_terminal = int(decision.get("targets_terminal", -2))
                if decision.get("schema_version") != 1:
                    errors.append("method-contract decision schema_version must be 1")
                if decision.get("recovery_complete") is not True:
                    errors.append("method-contract recovery is not complete")
                if decision.get("all_targets_terminal") is not True:
                    errors.append("method-contract decision has non-terminal targets")
                if decision.get("all_targets_recovery_complete") is not True:
                    errors.append("method-contract decision has incomplete target recovery")
                if decision.get("passes_completed") != required_passes:
                    errors.append(
                        "method-contract recovery passes must be exactly "
                        f"{required_passes}"
                    )
                if targets_total < 1 or targets_terminal != targets_total:
                    errors.append("method-contract terminal target counts are invalid")
                if int(decision.get("targets_recovery_complete", -3)) != targets_total:
                    errors.append("method-contract recovery target counts are invalid")
                if int(decision.get("precondition_failures", -1)) != 0:
                    errors.append("method-contract recovery has failed preconditions")
                if int(decision.get("attempt_ledger_failures", -1)) != 0:
                    errors.append("method-contract recovery has invalid attempt ledgers")
                if int(decision.get("selected_feature_count", -1)) != contract["feature_count"]:
                    errors.append("method-contract selected feature count differs from policy")
                base_evaluation_name = decision.get("selected_evaluation_contract")
                base_fullfit_name = decision.get("selected_fullfit_contract")
                if decision.get("selected_spec_name") != base_evaluation_name:
                    errors.append(
                        "method-contract selected spec and evaluation contract differ"
                    )
                base_evaluation = resolve_named_training_spec(
                    str(base_evaluation_name or "")
                )
                base_fullfit = resolve_named_training_spec(str(base_fullfit_name or ""))
                if len(base_evaluation.feature_cols) != contract["feature_count"]:
                    errors.append(
                        "method-contract base feature count differs from policy"
                    )
                errors.extend(
                    _validate_method_contract_descendant(
                        evaluation,
                        base_evaluation,
                        label="evaluation",
                        expected_model_seed=int(policy["evaluation"]["model_seed"]),
                        expected_odds_noise_seed=int(
                            policy["evaluation"]["odds_noise_seed"]
                        ),
                    )
                )
                errors.extend(
                    _validate_method_contract_descendant(
                        fullfit,
                        base_fullfit,
                        label="full-fit",
                        expected_model_seed=int(policy["evaluation"]["model_seed"]),
                        expected_odds_noise_seed=int(
                            policy["evaluation"]["odds_noise_seed"]
                        ),
                    )
                )
                if decision.get("decision_phase") != "pre_model_metrics":
                    errors.append("method-contract decision phase is not pre-model-metrics")
                if decision.get("model_metrics_consulted") is not False:
                    errors.append("method-contract decision consulted model metrics")

                supplied_decision_sha = decision.get("decision_sha256")
                unsigned = dict(decision)
                unsigned.pop("decision_sha256", None)
                computed_decision_sha = hashlib.sha256(
                    json.dumps(
                        unsigned, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest()
                if supplied_decision_sha != computed_decision_sha:
                    errors.append("method-contract internal decision SHA-256 is invalid")

                terminal_locator = decision.get("terminal_report_path")
                terminal_sha256 = decision.get("terminal_report_sha256")
                terminal_path = None
                if not isinstance(terminal_locator, str) or not terminal_locator.strip():
                    errors.append("method-contract terminal report path is missing")
                elif Path(terminal_locator).is_absolute():
                    errors.append("method-contract terminal report path must be repository-relative")
                else:
                    terminal_path = (policy_root / terminal_locator).resolve()
                    try:
                        terminal_path.relative_to(policy_root.resolve())
                    except ValueError:
                        errors.append("method-contract terminal report path escapes the repository")
                        terminal_path = None
                if not isinstance(terminal_sha256, str) or not _SHA256_RE.fullmatch(
                    terminal_sha256
                ):
                    errors.append("method-contract terminal report SHA-256 is invalid")
                elif terminal_path is not None:
                    if not terminal_path.is_file():
                        errors.append("method-contract terminal report is missing")
                    elif file_sha256(terminal_path) != terminal_sha256:
                        errors.append("method-contract terminal report hash is invalid")
                    else:
                        terminal_frame = pd.read_csv(terminal_path)
                        recomputed = method_recovery.decide_feature_contract(
                            terminal_frame,
                        )
                        recomputed["terminal_report_sha256"] = terminal_sha256
                        recomputed["terminal_report_path"] = terminal_locator
                        method_recovery._refresh_decision_sha256(recomputed)
                        mismatched = [
                            key
                            for key in sorted(set(decision) | set(recomputed))
                            if decision.get(key) != recomputed.get(key)
                        ]
                        if mismatched:
                            errors.append(
                                "method-contract decision does not reproduce from the "
                                f"terminal report: {mismatched}"
                            )
            except (ContractInputError, TypeError, ValueError) as exc:
                errors.append(f"invalid method-contract decision artifact: {exc}")
    if "performance_confirmation" in policy:
        performance_errors, _binding = validate_performance_confirmation(
            policy,
            repo_root=policy_root,
        )
        errors.extend(performance_errors)
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


def validate_training_input_evidence(
    evidence: Any,
    *,
    features_path: Path,
    fights_path: Path,
    policy: dict[str, Any],
    policy_sha256: str,
    performance_confirmation: Any,
) -> list[str]:
    """Validate the manifest receipt for the exact verified T-1 trainer input."""
    errors: list[str] = []
    if not isinstance(evidence, dict):
        return ["candidate has no policy-bound training_input_evidence object"]
    expected_keys = {
        "schema_version",
        "artifact_receipt_sha256",
        "artifact_roles",
        "policy_id",
        "policy_sha256",
        "fullfit_spec_name",
        "entry_offset_days",
        "prepared_features_sha256",
        "prepared_features_bytes",
        "source_fights_path",
        "source_fights_sha256",
        "source_fights_bytes",
        "final_track_c_pass_receipt_sha256",
        "performance_confirmation_result_sha256",
        "confirmed_strategy_config_sha256",
        "confirmation_evaluation_input_value_sha256",
        "feature_contract_count",
        "feature_contract_sha256",
        "confirmation_features_value_sha256",
        "confirmation_source_features_sha256",
        "confirmation_odds_source_inventory_sha256",
        "confirmation_source_fingerprint",
        "confirmation_source_inventory_sha256",
        "confirmation_source_inventory_artifact_sha256",
        "confirmation_environment_artifact_sha256",
        "confirmation_environment_payload_sha256",
        "confirmation_evaluation_protocol_sha256",
        "row_count",
        "rows_with_verified_t_minus_entry",
        "rows_missing_t_minus_entry",
        "prepared_odds_columns",
        "provenance_columns",
    }
    if set(evidence) != expected_keys:
        errors.append("candidate training_input_evidence keys are not exact")
        return errors
    performance_keys = {
        "schema_version",
        "policy_sha256",
        "confirmation_result_sha256",
        "final_track_c_pass_receipt_sha256",
        "evaluation_spec_name",
        "evaluation_spec_payload_sha256",
        "fullfit_spec_name",
        "fullfit_spec_payload_sha256",
        "strategy_config_sha256",
        "dataset_fights_sha256",
        "features_sha256",
        "features_value_sha256",
        "feature_contract_sha256",
        "confirmation_evaluation_input_value_sha256",
        "evaluation_protocol_sha256",
        "odds_source_inventory_sha256",
        "source_fingerprint",
        "source_inventory_sha256",
        "source_inventory_artifact_sha256",
        "environment_artifact_sha256",
        "environment_payload_sha256",
        "performance_evidence_aggregate_sha256",
    }
    if (
        not isinstance(performance_confirmation, dict)
        or set(performance_confirmation) != performance_keys
    ):
        return [
            "candidate has no exact performance_confirmation for training-input evidence"
        ]
    policy_confirmation = policy.get("performance_confirmation")
    if (
        not isinstance(policy_confirmation, dict)
        or policy_confirmation.get("state") != "final_track_c_bound"
    ):
        errors.append(
            "candidate training-input evidence is not bound to a final Track-C policy"
        )
    else:
        policy_performance_expected = {
            "confirmation_result_sha256": policy_confirmation.get(
                "confirmation_result_sha256"
            ),
            "strategy_config_sha256": policy_confirmation.get(
                "strategy_config_sha256"
            ),
        }
        for field, expected_value in policy_performance_expected.items():
            if performance_confirmation.get(field) != expected_value:
                errors.append(
                    "candidate performance_confirmation/final policy "
                    f"mismatch: {field}"
                )
    expected = {
        "schema_version": 1,
        "artifact_roles": ["logistic", "no_odds", "primary"],
        "policy_id": policy["policy_id"],
        "policy_sha256": policy_sha256,
        "fullfit_spec_name": policy["contract"]["fullfit_spec_name"],
        "entry_offset_days": float(policy["evaluation"]["entry_offset_days"]),
        "prepared_features_sha256": file_sha256(features_path),
        "prepared_features_bytes": int(features_path.stat().st_size),
        "feature_contract_count": int(policy["contract"]["feature_count"]),
    }
    for field, expected_value in expected.items():
        if evidence.get(field) != expected_value:
            errors.append(f"candidate training_input_evidence mismatch: {field}")
    if not _SHA256_RE.fullmatch(str(evidence.get("artifact_receipt_sha256") or "")):
        errors.append("candidate training_input_evidence artifact receipt is not SHA-256")
    source_fights_value = str(evidence.get("source_fights_path") or "").strip()
    if not source_fights_value:
        errors.append("candidate training_input_evidence source_fights_path is empty")
    else:
        source_fights_path = Path(source_fights_value).resolve(strict=False)
        if not source_fights_path.is_file():
            errors.append(
                "candidate training_input_evidence source_fights_path is missing"
            )
        else:
            if evidence.get("source_fights_sha256") != file_sha256(
                source_fights_path
            ):
                errors.append(
                    "candidate training_input_evidence source_fights_sha256 is stale"
                )
            if evidence.get("source_fights_bytes") != int(
                source_fights_path.stat().st_size
            ):
                errors.append(
                    "candidate training_input_evidence source_fights_bytes is stale"
                )

    evidence_hash_fields = expected_keys - {
        "schema_version",
        "artifact_roles",
        "policy_id",
        "fullfit_spec_name",
        "entry_offset_days",
        "prepared_features_bytes",
        "source_fights_path",
        "source_fights_bytes",
        "feature_contract_count",
        "row_count",
        "rows_with_verified_t_minus_entry",
        "rows_missing_t_minus_entry",
        "prepared_odds_columns",
        "provenance_columns",
    }
    for field in sorted(evidence_hash_fields):
        if not _SHA256_RE.fullmatch(str(evidence.get(field) or "")):
            errors.append(f"candidate training_input_evidence {field} is not SHA-256")

    performance_expected = {
        "schema_version": 1,
        "policy_sha256": evidence["policy_sha256"],
        "confirmation_result_sha256": evidence[
            "performance_confirmation_result_sha256"
        ],
        "final_track_c_pass_receipt_sha256": evidence[
            "final_track_c_pass_receipt_sha256"
        ],
        "evaluation_spec_name": policy["contract"]["evaluation_spec_name"],
        "evaluation_spec_payload_sha256": policy["contract"][
            "evaluation_spec_payload_sha256"
        ],
        "fullfit_spec_name": evidence["fullfit_spec_name"],
        "fullfit_spec_payload_sha256": policy["contract"][
            "fullfit_spec_payload_sha256"
        ],
        "strategy_config_sha256": evidence["confirmed_strategy_config_sha256"],
        "dataset_fights_sha256": evidence["source_fights_sha256"],
        "features_value_sha256": evidence["confirmation_features_value_sha256"],
        "feature_contract_sha256": evidence["feature_contract_sha256"],
        "confirmation_evaluation_input_value_sha256": evidence[
            "confirmation_evaluation_input_value_sha256"
        ],
        "evaluation_protocol_sha256": evidence[
            "confirmation_evaluation_protocol_sha256"
        ],
        "odds_source_inventory_sha256": evidence[
            "confirmation_odds_source_inventory_sha256"
        ],
        "source_fingerprint": evidence["confirmation_source_fingerprint"],
        "source_inventory_sha256": evidence[
            "confirmation_source_inventory_sha256"
        ],
        "source_inventory_artifact_sha256": evidence[
            "confirmation_source_inventory_artifact_sha256"
        ],
        "environment_artifact_sha256": evidence[
            "confirmation_environment_artifact_sha256"
        ],
        "environment_payload_sha256": evidence[
            "confirmation_environment_payload_sha256"
        ],
    }
    for field, expected_value in performance_expected.items():
        if performance_confirmation.get(field) != expected_value:
            errors.append(
                "candidate training_input_evidence/performance_confirmation "
                f"mismatch: {field}"
            )
    for field in performance_keys - {
        "schema_version",
        "evaluation_spec_name",
        "fullfit_spec_name",
    }:
        if not _SHA256_RE.fullmatch(
            str(performance_confirmation.get(field) or "")
        ):
            errors.append(f"candidate performance_confirmation {field} is not SHA-256")

    odds_columns = ["a_implied_prob", "b_implied_prob", "diff_implied_prob"]
    provenance_columns = [
        "model_odds_source_kind",
        "model_odds_source_file",
        "model_odds_observed_at",
        "model_odds_commence_time",
        "model_odds_hours_to_start",
        "model_odds_verified_prefight",
    ]
    if evidence.get("prepared_odds_columns") != odds_columns:
        errors.append("candidate training_input_evidence odds columns are not exact")
    if evidence.get("provenance_columns") != provenance_columns:
        errors.append("candidate training_input_evidence provenance columns are not exact")
    try:
        features = pd.read_csv(
            features_path,
            usecols=[*odds_columns, *provenance_columns],
            low_memory=False,
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return [*errors, f"cannot inspect policy-bound training odds evidence: {exc}"]
    odds_present = features[odds_columns].notna()
    populated = odds_present.all(axis=1)
    if (odds_present.any(axis=1) & ~populated).any():
        errors.append("candidate prepared features contain partial implied-probability rows")
    verified = features["model_odds_verified_prefight"].map(
        lambda value: value is True or (isinstance(value, np.bool_) and bool(value))
    )
    allowed_sources = set(policy["evaluation"]["quality_allowed_prefight_sources"])
    minimum_hours = float(policy["evaluation"]["entry_offset_days"]) * 24.0
    hours = pd.to_numeric(features["model_odds_hours_to_start"], errors="coerce")
    invalid = populated & (
        ~verified
        | ~features["model_odds_source_kind"].isin(allowed_sources)
        | ~hours.ge(minimum_hours)
    )
    if invalid.any() or verified[~populated].any():
        errors.append("candidate prepared model odds are not exclusively verified T-1 evidence")
    if populated.any() and not np.allclose(
        features.loc[populated, "diff_implied_prob"].to_numpy(dtype=float),
        (
            features.loc[populated, "a_implied_prob"]
            - features.loc[populated, "b_implied_prob"]
        ).to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        errors.append("candidate prepared diff_implied_prob is inconsistent")
    populated_count = int(populated.sum())
    missing_count = int((~populated).sum())
    if evidence.get("row_count") != len(features):
        errors.append("candidate training_input_evidence row_count is stale")
    if evidence.get("rows_with_verified_t_minus_entry") != populated_count:
        errors.append("candidate training_input_evidence verified T-1 count is stale")
    if evidence.get("rows_missing_t_minus_entry") != missing_count:
        errors.append("candidate training_input_evidence missing T-1 count is stale")
    return errors


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
        # The fixed root predates this policy contract and is identified by
        # its exact bundle/release/manifest/model/data hashes above. Only a
        # scheduled descendant is required to use the policy's new full-fit
        # spec; otherwise the old root could never bootstrap that lineage.
        if bundle.get("model_spec_name") != expected_spec:
            errors.append(
                f"active production spec is {bundle.get('model_spec_name')!r}; "
                f"policy requires {expected_spec!r}"
            )
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
    if policy.get("schema_version") != 2:
        errors.append("scheduled refit gate requires policy.schema_version=2")
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
    if policy.get("schema_version") != 2:
        errors.append("scheduled refit gate requires policy.schema_version=2")
    if (
        policy.get("performance_confirmation", {}).get("state")
        != "final_track_c_bound"
    ):
        errors.append(
            "candidate full fit requires "
            "policy.performance_confirmation.state='final_track_c_bound'"
        )
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
        "policy_schema_version": policy["schema_version"],
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
    if policy.get("schema_version") == 2:
        errors.extend(
            validate_training_input_evidence(
                manifest.get("training_input_evidence"),
                features_path=snapshot_paths["features"],
                fights_path=snapshot_paths["fights"],
                policy=policy,
                policy_sha256=identity["sha256"],
                performance_confirmation=manifest.get(
                    "performance_confirmation"
                ),
            )
        )
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
