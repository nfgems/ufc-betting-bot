"""Assemble and validate one immutable local production-candidate bundle.

This script is intentionally unable to publish into the canonical ``models`` or
``data/processed`` directories.  It copies a fully trained candidate into a new
repository-local staging root, records all identities needed for review and
rollback, and invokes the strict staged production-bundle validator before it
reports success.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.io_utils import copy_file_atomically, write_json_atomically
from src.model.production_bundle import (
    ProductionBundleError,
    _expected_no_odds_spec_payload,
    load_production_bundle,
    validate_production_bundle,
)
from src.model.training_spec import resolve_named_training_spec
from src.strategy.runtime_strategy import (
    build_confirmed_strategy_payload,
    validate_confirmed_strategy_payload,
)

import scripts.build_model_input_inventory as model_input_inventory
import scripts.bfo_lineage as bfo_lineage
import scripts.check_production_refit_contract as production_refit_contract


MODEL_FILENAMES = {
    "primary": "xgboost_model.pkl",
    "no_odds": "xgboost_no_odds_model.pkl",
    "logistic": "logistic_model.pkl",
}
PROCESSED_FILENAMES = (
    "fights_cleaned.csv",
    "features.csv",
    "test_set.csv",
    "test_set.csv.metadata.json",
)
FIT_ONLY_SPEC_FIELDS = frozenset({"git_hash", "trained_at"})
FULLFIT_ALLOWED_DIFFERENCES = frozenset(
    {"name", "description", "train_cutoff_date"}
)
SCHEDULED_POLICY_STAGED_PATH = "provenance/scheduled_refit_policy.json"
SCHEDULED_POLICY_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "policy_id",
        "contract",
        "evaluation",
        "baseline",
        "health_limits",
        "root_release",
    }
)
SCHEDULED_POLICY_CONTRACT_KEYS = frozenset(
    {
        "evaluation_spec_name",
        "evaluation_spec_payload_sha256",
        "fullfit_spec_name",
        "fullfit_spec_payload_sha256",
        "allowed_fullfit_differences",
        "exclusive_train_cutoff_date",
        "minimum_cutoff_buffer_days",
        "feature_count",
    }
)
SCHEDULED_POLICY_CONTRACT_V2_KEYS = SCHEDULED_POLICY_CONTRACT_KEYS | frozenset(
    {"method_contract_decision_path", "method_contract_decision_sha256"}
)
SCHEDULED_POLICY_ROOT_RELEASE_KEYS = frozenset(
    {
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
)
SCHEDULED_POLICY_EVALUATION_KEYS = frozenset(
    {
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
)
SCHEDULED_POLICY_EVALUATION_V2_KEYS = SCHEDULED_POLICY_EVALUATION_KEYS | frozenset(
    {
        "require_entry_odds",
        "quality_recent_window_days",
        "quality_minimum_recent_rows",
        "quality_minimum_entry_coverage",
        "quality_maximum_entry_lag_days",
        "quality_allowed_prefight_sources",
    }
)
SCHEDULED_POLICY_EXECUTION_ASSUMPTION_KEYS = frozenset(
    {
        "min_book_liquidity",
        "max_slippage",
        "max_bet_vs_book_ratio",
        "assumed_half_spread",
        "synthetic_liquidity_floor",
        "synthetic_liquidity_peak",
        "synthetic_price_step",
        "synthetic_depth_notional_shares",
    }
)
SCHEDULED_POLICY_BASELINE_KEYS = frozenset(
    {
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
)
SCHEDULED_POLICY_HEALTH_LIMIT_KEYS = frozenset(
    {
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
)
SCHEDULED_REFIT_MANIFEST_KEYS = frozenset(
    {
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
)
MAX_EVIDENCE_FILE_BYTES = 10 * 1024 * 1024
MAX_EVIDENCE_TOTAL_BYTES = 50 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
RUNTIME_HASH_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
APPROVED_EVALUATION_SPEC = "full_live_contract_v6_durability"
APPROVED_FULLFIT_SPEC = (
    "full_live_contract_v6_durability_corrected_20260805_fullfit"
)
APPROVED_EVALUATION_PAYLOAD_SHA256 = (
    "68f2fd6d851224ab395fe469b17a9974d87b8b48d812e2108636d6b889352f45"
)
APPROVED_FIGHTS_SHA256 = (
    "8b4d068df13e3d8440f819c7d2021a94d72abdb3f36de2aecd4a6326d8c4d8b3"
)
APPROVED_FEATURES_SHA256 = (
    "7bb8b1f6594d0844740cb5e2c11e873f0a7b23bea6880d3c80f88509f2eedcc5"
)
APPROVED_TRAIN_FIGHTS_SHA256 = (
    "a81c05b9c674dbacfa647c2cf744888aba1a2dd04d033cc58a4f23f061e95894"
)
APPROVED_TRAIN_FEATURES_SHA256 = (
    "38d399c8fdc04a1338ff47226e7908aeaeed5efad5ff759062acb1500a188b1f"
)
SEMANTIC_EQUIVALENCE_ATOL = 1e-12
ASSEMBLY_INVENTORY_ALLOWED_CHANGED_PATHS = frozenset(
    {
        "scripts/build_staged_production_bundle.py",
        "scripts/parity_replay.py",
    }
)
APPROVED_BFO_LEDGER_NAME = (
    "bfo_revalidation_20260805_head_source.provenance.jsonl"
)
APPROVED_BFO_LEDGER_SHA256 = (
    "2bc0e3797b427607498e1714f6e890a13c90c24063c4606f0d631deebcc4a217"
)
APPROVED_BFO_CSVS = {
    "historical_odds_bfo_recovered_20260319.csv": (
        45,
        "e5be8fcae5935dc0fa780c2bdb3b724d7350fca71db726b6aa8e354153abee3b",
    ),
    "historical_odds_bfo_recovered_20260529_fullfit_gap.csv": (
        90,
        "faddf329d55f2be402a18ca58477b039a20fdc1f6d4a01f11a8298909bbbfddc",
    ),
    "historical_odds_bfo_recovered_20260711_guard_gap.csv": (
        53,
        "45df9112970682e02e4ffe37d4b557d5b434ebb364a9eb1a695186a7d082c309",
    ),
    "historical_odds_bfo_recovered_auto_20260722_run29887204421_1.csv": (
        22,
        "439fbed0733372f5b60d87ad91505f91a47be509de0f71d567b488de87256ea7",
    ),
    "historical_odds_bfo_recovered_auto_20260728_run30341844205_1.csv": (
        11,
        "11a38f9e3e748e702a4bfbc318dcb781265177c582db480fa2b6069aff0c8ee7",
    ),
    "historical_odds_bfo_recovered_auto_20260804_run30891790168_1.csv": (
        13,
        "bd877d41b85e1057782a0d134be7e5d9116455e8d52a6d3c3829807663bf6c1c",
    ),
}
APPROVED_BFO_PARSER_CANONICAL_SHA256 = {
    "scripts/recover_bfo_moneyline_gaps.py": (
        "6d0a50ec26b4d41e458ac18ecfce2bb556236d3ed09c76aa2e1d5a07762c97e7"
    ),
    "scripts/revalidate_bfo_recovery_file.py": (
        "ed8e3147dadf310c7a94e6a9e194049e0f708f5b27ccf720cf73f42373e0cd3e"
    ),
}
SAFE_EVIDENCE_SUFFIXES = frozenset({".csv", ".json", ".md", ".txt"})
SENSITIVE_EVIDENCE_TOKENS = (
    ".env",
    "credential",
    "secret",
    "private_key",
    "private-key",
    "account_ledger",
    "account-ledger",
    "bet_ledger",
    "bet-ledger",
)


class StagingBundleError(RuntimeError):
    """Raised when a candidate cannot be staged without weakening a gate."""


@dataclass(frozen=True)
class BundleInputs:
    staging_root: Path
    candidate_models_dir: Path
    candidate_processed_dir: Path
    evaluation_spec_name: str
    input_inventory_path: Path
    assembly_inventory_path: Path
    bfo_provenance_path: Path
    selection_evidence_paths: tuple[Path, ...]
    previous_manifest_path: Path | None
    previous_readyz_path: Path
    previous_deployed_git_sha: str | None
    previous_runtime_lookup_hashes: dict[str, str]
    expected_fights_sha256: str
    expected_features_sha256: str
    training_argv: tuple[str, ...]
    bundle_id: str | None = None
    inference_sample_rows: int = 32
    scheduled_refit_policy_path: Path | None = None
    previous_bfo_lineage_manifest_path: Path | None = None
    final_track_c_pass_receipt_path: Path | None = None


@dataclass(frozen=True)
class ScheduledRefitPolicy:
    path: Path
    sha256: str
    payload: dict[str, Any]

    @property
    def contract(self) -> dict[str, Any]:
        return self.payload["contract"]

    @property
    def root_release(self) -> dict[str, Any]:
        return self.payload["root_release"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_text_sha256(path: Path) -> str:
    """Hash reviewed text content independent of Git's LF/CRLF checkout form."""
    raw = path.read_bytes()
    canonical = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return _sha256_bytes(canonical)


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _file_identity(path: Path) -> dict[str, object]:
    return {"sha256": _sha256_file(path), "bytes": int(path.stat().st_size)}


def _aggregate_identities(rows: Iterable[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: str(value["path"])):
        digest.update(str(row["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).lower().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _repo_path(
    value: Path,
    *,
    repo_root: Path,
    label: str,
    must_exist: bool = True,
) -> Path:
    raw = value if value.is_absolute() else repo_root / value
    try:
        resolved = raw.resolve(strict=must_exist)
    except OSError as exc:
        raise StagingBundleError(f"{label} cannot be resolved: {raw}: {exc}") from exc
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise StagingBundleError(f"{label} must stay inside the repository: {raw}") from exc
    return resolved


def _existing_file(value: Path, *, repo_root: Path, label: str) -> Path:
    path = _repo_path(value, repo_root=repo_root, label=label)
    if not path.is_file():
        raise StagingBundleError(f"{label} is not a file: {path}")
    return path


def _strict_candidate_dir(
    value: Path,
    *,
    repo_root: Path,
    allowed_root: Path,
    label: str,
) -> Path:
    path = _repo_path(value, repo_root=repo_root, label=label)
    if not path.is_dir():
        raise StagingBundleError(f"{label} is not a directory: {path}")
    try:
        relative = path.relative_to(allowed_root.resolve(strict=False))
    except ValueError as exc:
        raise StagingBundleError(
            f"{label} must be below {allowed_root.resolve(strict=False)}: {path}"
        ) from exc
    if not relative.parts:
        raise StagingBundleError(f"{label} must name an isolated candidate run: {path}")
    return path


def _exact_child_file(directory: Path, filename: str, *, label: str) -> Path:
    path = directory / filename
    if not path.is_file():
        raise StagingBundleError(f"Missing exact {label} path: {path}")
    if path.resolve(strict=True).parent != directory.resolve(strict=True):
        raise StagingBundleError(f"{label} must be a real file directly under {directory}")
    return path.resolve(strict=True)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StagingBundleError(f"{label} is not valid UTF-8 JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StagingBundleError(f"{label} must contain a JSON object: {path}")
    return payload


def _load_json_object_without_duplicates(
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StagingBundleError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except StagingBundleError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StagingBundleError(
            f"{label} is not valid duplicate-free UTF-8 JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise StagingBundleError(f"{label} must contain a JSON object: {path}")
    return payload


def _require_exact_keys(
    payload: dict[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    observed = set(payload)
    if observed != expected:
        raise StagingBundleError(
            f"{label} keys are not exact: missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )


def _load_scheduled_refit_policy(
    path: Path,
    *,
    repo_root: Path,
) -> ScheduledRefitPolicy:
    resolved = _existing_file(path, repo_root=repo_root, label="Scheduled refit policy")
    try:
        payload = production_refit_contract.load_policy(resolved)
    except (production_refit_contract.ContractInputError, OSError) as exc:
        raise StagingBundleError(f"Scheduled refit policy is invalid: {exc}") from exc
    _require_exact_keys(
        payload,
        SCHEDULED_POLICY_TOP_LEVEL_KEYS
        | ({"performance_confirmation"} if "performance_confirmation" in payload else set()),
        label="Scheduled refit policy",
    )
    schema_version = payload.get("schema_version")
    if schema_version not in {1, 2} or isinstance(schema_version, bool):
        raise StagingBundleError("Scheduled refit policy must use schema_version 1 or 2")
    policy_id = payload.get("policy_id")
    if (
        not isinstance(policy_id, str)
        or not policy_id.strip()
        or policy_id != policy_id.strip()
        or any(character.isspace() for character in policy_id)
    ):
        raise StagingBundleError("Scheduled refit policy_id must be a nonempty token")

    contract = payload.get("contract")
    evaluation = payload.get("evaluation")
    baseline = payload.get("baseline")
    health_limits = payload.get("health_limits")
    root_release = payload.get("root_release")
    if not all(
        isinstance(section, dict)
        for section in (contract, evaluation, baseline, health_limits, root_release)
    ):
        raise StagingBundleError("Scheduled refit policy sections must be objects")
    _require_exact_keys(
        contract,
        (
            SCHEDULED_POLICY_CONTRACT_V2_KEYS
            if schema_version == 2
            else SCHEDULED_POLICY_CONTRACT_KEYS
        ),
        label="Scheduled refit policy contract",
    )
    _require_exact_keys(
        evaluation,
        (
            SCHEDULED_POLICY_EVALUATION_V2_KEYS
            if schema_version == 2
            else SCHEDULED_POLICY_EVALUATION_KEYS
        ),
        label="Scheduled refit policy evaluation",
    )
    assumptions = evaluation.get("execution_assumptions")
    if not isinstance(assumptions, dict):
        raise StagingBundleError("Scheduled refit execution_assumptions must be an object")
    _require_exact_keys(
        assumptions,
        SCHEDULED_POLICY_EXECUTION_ASSUMPTION_KEYS,
        label="Scheduled refit execution assumptions",
    )
    _require_exact_keys(
        baseline,
        SCHEDULED_POLICY_BASELINE_KEYS,
        label="Scheduled refit policy baseline",
    )
    _require_exact_keys(
        health_limits,
        SCHEDULED_POLICY_HEALTH_LIMIT_KEYS,
        label="Scheduled refit policy health limits",
    )
    _require_exact_keys(
        root_release,
        SCHEDULED_POLICY_ROOT_RELEASE_KEYS,
        label="Scheduled refit policy root release",
    )

    for field in (
        "evaluation_spec_payload_sha256",
        "fullfit_spec_payload_sha256",
    ):
        if not SHA256_RE.fullmatch(str(contract.get(field) or "")):
            raise StagingBundleError(f"Scheduled refit contract {field} is not SHA-256")
    if schema_version == 2:
        decision_path = Path(str(contract.get("method_contract_decision_path") or ""))
        if (
            not str(decision_path)
            or decision_path.is_absolute()
            or ".." in decision_path.parts
            or not SHA256_RE.fullmatch(
                str(contract.get("method_contract_decision_sha256") or "")
            )
        ):
            raise StagingBundleError(
                "Scheduled refit method-contract decision binding is invalid"
            )
        resolved_decision_path = _existing_file(
            decision_path,
            repo_root=repo_root,
            label="Scheduled refit method-contract decision",
        )
        if _sha256_file(resolved_decision_path) != str(
            contract["method_contract_decision_sha256"]
        ).lower():
            raise StagingBundleError(
                "Scheduled refit method-contract decision SHA-256 does not match policy"
            )
        registry_errors, _, _ = production_refit_contract.validate_policy_registry(
            payload,
            repo_root=repo_root,
        )
        if registry_errors:
            raise StagingBundleError(
                "Scheduled refit policy-v2 registry validation failed: "
                + "; ".join(registry_errors)
            )
        try:
            entry_offset_days = float(evaluation.get("entry_offset_days", -1.0))
            minimum_entry_coverage = float(
                evaluation.get("quality_minimum_entry_coverage", -1.0)
            )
            maximum_entry_lag_days = float(
                evaluation.get("quality_maximum_entry_lag_days", -1.0)
            )
        except (TypeError, ValueError) as exc:
            raise StagingBundleError(
                "Scheduled refit policy v2 honest-odds quality settings are invalid"
            ) from exc
        if (
            evaluation.get("require_entry_odds") is not True
            or evaluation.get("entry_offset_for_features") is not True
            or entry_offset_days != 1.0
            or evaluation.get("quality_recent_window_days") != 60
            or evaluation.get("quality_minimum_recent_rows") != 50
            or minimum_entry_coverage != 0.70
            or maximum_entry_lag_days != 35.0
            or evaluation.get("quality_allowed_prefight_sources")
            != ["line_history", "odds_api"]
        ):
            raise StagingBundleError(
                "Scheduled refit policy v2 honest-odds quality settings are invalid"
            )
    if set(contract.get("allowed_fullfit_differences") or []) != set(
        FULLFIT_ALLOWED_DIFFERENCES
    ) or len(contract.get("allowed_fullfit_differences") or []) != len(
        FULLFIT_ALLOWED_DIFFERENCES
    ):
        raise StagingBundleError(
            "Scheduled refit allowed_fullfit_differences must contain exactly "
            "description, name, and train_cutoff_date"
        )
    for field in ("minimum_cutoff_buffer_days", "feature_count"):
        if not isinstance(contract.get(field), int) or int(contract[field]) < 1:
            raise StagingBundleError(f"Scheduled refit contract {field} must be positive")
    try:
        datetime.fromisoformat(str(contract["exclusive_train_cutoff_date"])).date()
    except (TypeError, ValueError) as exc:
        raise StagingBundleError(
            "Scheduled refit exclusive_train_cutoff_date must be an ISO date"
        ) from exc
    for field in (
        "source_manifest_sha256",
        "installed_manifest_sha256",
        "model_sha256",
        "no_odds_model_sha256",
        "logistic_model_sha256",
        "processed_fights_sha256",
        "processed_features_sha256",
    ):
        if not SHA256_RE.fullmatch(str(root_release.get(field) or "")):
            raise StagingBundleError(f"Scheduled refit root_release.{field} is not SHA-256")
    for field in (
        "evidence_sha256",
        "evidence_root_source_manifest_sha256",
        "features_sha256",
        "evidence_protocol_sha256",
        "scheduled_protocol_sha256",
        "evaluation_sample_sha256",
    ):
        if not SHA256_RE.fullmatch(str(baseline.get(field) or "")):
            raise StagingBundleError(f"Scheduled refit baseline.{field} is not SHA-256")
    expected_baseline_role = (
        "confirmed_winner_track_c_baseline"
        if payload.get("performance_confirmation", {}).get("state")
        == "final_track_c_bound"
        else "root_release_reference_thresholds"
    )
    if baseline.get("comparison_role") != expected_baseline_role:
        raise StagingBundleError(
            "Scheduled refit baseline role does not match its policy state"
        )
    if baseline.get("evidence_protocol_sha256") != baseline.get(
        "scheduled_protocol_sha256"
    ):
        raise StagingBundleError(
            "Scheduled refit evidence protocol does not match the fixed protocol"
        )
    if (
        baseline.get("evidence_root_source_manifest_sha256")
        != root_release.get("source_manifest_sha256")
    ):
        raise StagingBundleError("Scheduled refit baseline evidence is not root-manifest bound")
    for field in ("promotion_git_sha", "training_source_git_sha"):
        if not GIT_SHA_RE.fullmatch(str(root_release.get(field) or "")):
            raise StagingBundleError(f"Scheduled refit root_release.{field} is not a Git SHA")
    for field in ("bundle_id", "release_id"):
        value = root_release.get(field)
        if not isinstance(value, str) or not value or value != Path(value).name:
            raise StagingBundleError(
                f"Scheduled refit root_release.{field} must be a safe nonempty token"
            )

    return ScheduledRefitPolicy(
        path=resolved,
        sha256=_sha256_file(resolved),
        payload=payload,
    )


def _validate_inventory_payload(
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    payload = _load_json_object(path, label=label)
    if payload.get("schema_version") != 1:
        raise StagingBundleError(f"{label} must use schema_version 1")
    if not str(payload.get("run_id") or "").strip() or not str(
        payload.get("generated_at_utc") or ""
    ).strip():
        raise StagingBundleError(f"{label} is missing run/timestamp identity")
    rows = payload.get("files")
    if not isinstance(rows, list) or not rows or not all(
        isinstance(row, dict) for row in rows
    ):
        raise StagingBundleError(f"{label} files must be a nonempty object list")
    paths = [str(row.get("path") or "") for row in rows]
    if any(not path for path in paths) or len(paths) != len(set(paths)):
        raise StagingBundleError(f"{label} paths must be nonempty and unique")
    for row in rows:
        expected_category = (
            "raw_input" if str(row["path"]).startswith("data/raw/") else "source"
        )
        if (
            row.get("category") != expected_category
            or not isinstance(row.get("bytes"), int)
            or int(row["bytes"]) < 0
            or not SHA256_RE.fullmatch(str(row.get("sha256") or ""))
        ):
            raise StagingBundleError(
                f"{label} row identity is invalid for {row['path']}"
            )
    source_count = sum(row.get("category") == "source" for row in rows)
    raw_count = sum(row.get("category") == "raw_input" for row in rows)
    if (
        payload.get("file_count") != len(rows)
        or payload.get("source_file_count") != source_count
        or payload.get("raw_input_file_count") != raw_count
        or str(payload.get("inventory_sha256") or "").lower()
        != _aggregate_identities(rows)
    ):
        raise StagingBundleError(f"{label} counts or aggregate are invalid")
    status_lines = payload.get("git_status")
    if not isinstance(status_lines, list) or not all(
        isinstance(line, str) for line in status_lines
    ):
        raise StagingBundleError(f"{label} git_status must be a string list")
    status_bytes = ("\n".join(status_lines) + ("\n" if status_lines else "")).encode()
    if _sha256_bytes(status_bytes) != payload.get("git_status_sha256"):
        raise StagingBundleError(f"{label} git_status hash is internally invalid")
    if not GIT_SHA_RE.fullmatch(str(payload.get("git_head") or "")) or not SHA256_RE.fullmatch(
        str(payload.get("git_diff_sha256") or "")
    ):
        raise StagingBundleError(f"{label} git identity is invalid")
    return payload


def _raw_inventory(payload: dict[str, Any]) -> dict[str, object]:
    raw_rows = [
        row
        for row in payload.get("files", [])
        if isinstance(row, dict) and row.get("category") == "raw_input"
    ]
    return {
        "file_count": len(raw_rows),
        "inventory_sha256": _aggregate_identities(raw_rows),
        "files": raw_rows,
    }


def _validate_input_inventories(
    pretraining_path: Path,
    assembly_path: Path,
    *,
    repo_root: Path,
    scheduled_refit: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, object],
    dict[str, object],
]:
    pretraining = _validate_inventory_payload(
        pretraining_path,
        label="Pretraining model input inventory",
    )
    assembly = _validate_inventory_payload(
        assembly_path,
        label="Assembly model input inventory",
    )

    module_root = model_input_inventory.REPO_ROOT.resolve(strict=True)
    if module_root != repo_root:
        raise StagingBundleError(
            "Model inventory helper root does not match the staging repository"
        )
    current = model_input_inventory.build_inventory(
        run_id=str(assembly.get("run_id") or "staging-verification")
    )
    current_comparison = model_input_inventory.compare_inventories(assembly, current)
    if not current_comparison.get("ok"):
        raise StagingBundleError(
            "Assembly model input inventory scoped files changed after it was captured: "
            f"added={current_comparison.get('added')}, "
            f"removed={current_comparison.get('removed')}, "
            f"changed={current_comparison.get('changed')}"
        )
    if str(assembly.get("inventory_sha256") or "").lower() != str(
        current.get("inventory_sha256") or ""
    ).lower():
        raise StagingBundleError("Assembly model input inventory aggregate hash is invalid")
    for field in ("git_head", "git_diff_sha256"):
        if str(assembly.get(field) or "").lower() != str(current.get(field) or "").lower():
            raise StagingBundleError(
                f"Assembly model input inventory {field} no longer matches the repository"
            )

    pretraining_to_assembly = model_input_inventory.compare_inventories(
        pretraining, assembly
    )
    added = list(pretraining_to_assembly.get("added") or [])
    removed = list(pretraining_to_assembly.get("removed") or [])
    changed = list(pretraining_to_assembly.get("changed") or [])
    allowed_change = set(changed).issubset(
        ASSEMBLY_INVENTORY_ALLOWED_CHANGED_PATHS
    )
    if added or removed or not allowed_change or not pretraining_to_assembly.get(
        "git_head_matches"
    ):
        raise StagingBundleError(
            "Pretraining-to-assembly scoped diff must contain only the allowlisted "
            "posttraining validation/packaging changes: "
            f"added={added}, removed={removed}, changed={changed}, "
            f"git_head_matches={pretraining_to_assembly.get('git_head_matches')}"
        )
    pretraining_raw = _raw_inventory(pretraining)
    assembly_raw = _raw_inventory(assembly)
    if pretraining_raw != assembly_raw:
        raise StagingBundleError(
            "Raw inputs must be byte-identical between pretraining and assembly inventories"
        )
    inventory_delta: dict[str, object] = {
        "allowlisted_changed_paths": sorted(
            ASSEMBLY_INVENTORY_ALLOWED_CHANGED_PATHS
        ),
        "added": added,
        "removed": removed,
        "changed": changed,
        "git_head_matches": True,
        "git_diff_hash_matches": bool(
            pretraining_to_assembly.get("git_diff_matches")
        ),
        "all_raw_inputs_identical": True,
        "only_allowlisted_assembly_change": True,
        "pretraining_inventory_sha256": pretraining["inventory_sha256"],
        "assembly_inventory_sha256": assembly["inventory_sha256"],
    }
    if scheduled_refit:
        inventory_delta["scheduled_refit_mode"] = True
    return pretraining, assembly, pretraining_raw, inventory_delta


def _validate_fixed_bfo_ledger(
    path: Path,
    *,
    repo_root: Path,
    inventory_payload: dict[str, Any],
    allowed_additional_csv_names: set[str] | None = None,
) -> dict[str, object]:
    from scripts.recover_bfo_moneyline_gaps import SPORTSBOOK_DISPLAY_NAMES

    raw_root = (repo_root / "data" / "raw").resolve(strict=True)
    try:
        relative_raw = path.relative_to(raw_root)
    except ValueError as exc:
        raise StagingBundleError(
            f"BFO provenance ledger must be under data/raw: {path}"
        ) from exc
    if (
        path.name != APPROVED_BFO_LEDGER_NAME
        or _canonical_text_sha256(path) != APPROVED_BFO_LEDGER_SHA256
    ):
        raise StagingBundleError("BFO provenance ledger is not the approved corrected ledger")

    repo_relative = path.relative_to(repo_root).as_posix()
    inventory_rows = {
        str(row.get("path")): row
        for row in inventory_payload.get("files", [])
        if isinstance(row, dict)
    }
    inventory_row = inventory_rows.get(repo_relative)
    if inventory_row is None or inventory_row.get("category") != "raw_input":
        raise StagingBundleError(
            "BFO provenance ledger is missing from the raw input inventory"
        )

    ledger_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StagingBundleError(
                    f"BFO provenance ledger line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise StagingBundleError(
                    f"BFO provenance ledger line {line_number} is not an object"
                )
            ledger_rows.append(row)
    if len(ledger_rows) != 244:
        raise StagingBundleError("BFO provenance ledger must contain exactly 244 records")

    odds_root = path.parent
    actual_csv_names = {item.name for item in odds_root.glob("historical_odds_bfo_recovered_*.csv")}
    expected_csv_names = set(APPROVED_BFO_CSVS) | set(
        allowed_additional_csv_names or set()
    )
    if actual_csv_names != expected_csv_names:
        raise StagingBundleError(
            "BFO corrected CSV set is not exactly the approved baseline plus "
            "authenticated scheduled lineage"
        )
    csv_rows: dict[str, dict[tuple[object, ...], dict[str, str]]] = {}
    csv_identities: list[dict[str, object]] = []
    key_fields = ("event_date", "fighter_a", "fighter_b", "query_date", "offset_days")
    for filename, (expected_rows, expected_sha) in APPROVED_BFO_CSVS.items():
        csv_path = odds_root / filename
        if _canonical_text_sha256(csv_path) != expected_sha:
            raise StagingBundleError(f"Corrected BFO CSV hash is not approved: {filename}")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_rows:
            raise StagingBundleError(f"Corrected BFO CSV row count is invalid: {filename}")
        indexed: dict[tuple[object, ...], dict[str, str]] = {}
        for row in rows:
            try:
                key = tuple(row[field] for field in key_fields[:-1]) + (
                    int(row["offset_days"]),
                )
            except (KeyError, ValueError) as exc:
                raise StagingBundleError(f"Corrected BFO CSV schema is invalid: {filename}") from exc
            if key in indexed:
                raise StagingBundleError(f"Corrected BFO CSV has a duplicate recovery key: {key}")
            indexed[key] = row
        csv_rows[filename] = indexed
        csv_identities.append(
            {
                "path": csv_path.relative_to(repo_root).as_posix(),
                "rows": len(rows),
                "canonical_content_sha256": expected_sha,
                **_file_identity(csv_path),
            }
        )

    required_top = {
        "schema_version",
        "decision",
        "recovery_key",
        "requested_fighters",
        "input_batch",
        "parser",
        "thresholds",
        "event_page",
        "matched_bfo_rows",
        "paired_quotes",
        "consensus",
        "csv_values",
        "rejection_reason",
    }
    accepted_keys: dict[str, set[tuple[object, ...]]] = {
        filename: set() for filename in APPROVED_BFO_CSVS
    }
    decisions = {"accepted": 0, "rejected": 0}
    parser_matches_current_inventory = True
    for index, row in enumerate(ledger_rows, start=1):
        if not required_top.issubset(row) or row.get("schema_version") != 1:
            raise StagingBundleError(f"BFO ledger record {index} has an invalid schema")
        decision = row.get("decision")
        if decision not in decisions:
            raise StagingBundleError(f"BFO ledger record {index} has an invalid decision")
        decisions[decision] += 1
        parser = row.get("parser")
        parser_files = parser.get("file_sha256") if isinstance(parser, dict) else None
        if (
            not isinstance(parser, dict)
            or not GIT_SHA_RE.fullmatch(str(parser.get("git_head") or ""))
            or not SHA256_RE.fullmatch(str(parser.get("dirty_diff_sha256") or ""))
            or not isinstance(parser_files, dict)
            or set(parser_files) != set(APPROVED_BFO_PARSER_CANONICAL_SHA256)
            or not all(SHA256_RE.fullmatch(str(value or "")) for value in parser_files.values())
        ):
            raise StagingBundleError(f"BFO ledger record {index} lacks parser identity")
        for parser_path, parser_sha in parser_files.items():
            source_path = repo_root / parser_path
            source_identity = _file_identity(source_path)
            inventoried = inventory_rows.get(parser_path, {})
            if (
                inventoried.get("sha256") != source_identity["sha256"]
                or inventoried.get("bytes") != source_identity["bytes"]
                or _canonical_text_sha256(source_path)
                != APPROVED_BFO_PARSER_CANONICAL_SHA256[parser_path]
            ):
                raise StagingBundleError(
                    f"BFO ledger record {index} parser content does not match "
                    "the approved inventoried source"
                )
            if inventoried.get("sha256") != parser_sha:
                parser_matches_current_inventory = False
        recovery = row.get("recovery_key")
        requested = row.get("requested_fighters")
        if not isinstance(recovery, dict) or not isinstance(requested, dict):
            raise StagingBundleError(f"BFO ledger record {index} lacks fight identity")
        try:
            key = tuple(recovery[field] for field in key_fields[:-1]) + (
                int(recovery["offset_days"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StagingBundleError(f"BFO ledger record {index} has a bad recovery key") from exc
        if (
            requested.get("fighter_a") != recovery.get("fighter_a")
            or requested.get("fighter_b") != recovery.get("fighter_b")
        ):
            raise StagingBundleError(f"BFO ledger record {index} fighter identity disagrees")
        batch_marker = ":data/raw/historical_odds/"
        input_batch = str(row.get("input_batch") or "")
        if batch_marker not in input_batch:
            raise StagingBundleError(f"BFO ledger record {index} has a bad input batch")
        filename = input_batch.split(batch_marker, 1)[1]
        if filename not in APPROVED_BFO_CSVS:
            raise StagingBundleError(f"BFO ledger record {index} names an unapproved input")

        thresholds = row.get("thresholds")
        if not isinstance(thresholds, dict) or thresholds.get("minimum_paired_sportsbooks") != 3:
            raise StagingBundleError(f"BFO ledger record {index} has invalid thresholds")
        if decision == "rejected":
            if not str(row.get("rejection_reason") or "").strip():
                raise StagingBundleError(f"Rejected BFO ledger record {index} lacks a reason")
            continue

        event_page = row.get("event_page")
        consensus = row.get("consensus")
        csv_values = row.get("csv_values")
        quotes = row.get("paired_quotes")
        if (
            not isinstance(event_page, dict)
            or not str(event_page.get("url") or "").startswith(("http://", "https://"))
            or not SHA256_RE.fullmatch(str(event_page.get("content_sha256") or ""))
            or not str(event_page.get("fetched_at_utc") or "").strip()
            or not isinstance(consensus, dict)
            or not isinstance(csv_values, dict)
            or not isinstance(quotes, list)
            or not row.get("matched_bfo_rows")
            or row.get("rejection_reason") not in ("", None)
        ):
            raise StagingBundleError(f"Accepted BFO ledger record {index} lacks provenance")
        try:
            datetime.fromisoformat(str(event_page["fetched_at_utc"]).replace("Z", "+00:00"))
            a_prob = float(consensus["a_fair_prob"])
            b_prob = float(consensus["b_fair_prob"])
            books = int(consensus["num_bookmakers"])
            accepted_quotes = [quote for quote in quotes if quote.get("accepted") is True]
            market_ids = [int(quote["market_id"]) for quote in accepted_quotes]
        except (KeyError, TypeError, ValueError) as exc:
            raise StagingBundleError(f"Accepted BFO ledger record {index} is malformed") from exc
        if (
            not math.isfinite(a_prob)
            or not math.isfinite(b_prob)
            or not 0.0 < a_prob < 1.0
            or not 0.0 < b_prob < 1.0
            or abs(a_prob + b_prob - 1.0) > 1e-9
            or books < 3
            or len(accepted_quotes) != books
            or len(market_ids) != len(set(market_ids))
            or any(market_id not in SPORTSBOOK_DISPLAY_NAMES for market_id in market_ids)
            or any(
                quote.get("book_name") != SPORTSBOOK_DISPLAY_NAMES[int(quote["market_id"])]
                for quote in accepted_quotes
            )
        ):
            raise StagingBundleError(f"Accepted BFO ledger record {index} fails quote semantics")
        for quote in accepted_quotes:
            try:
                overround = float(quote["overround"])
                quote_prob = float(quote["a_fair_prob"])
            except (KeyError, TypeError, ValueError) as exc:
                raise StagingBundleError(f"Accepted BFO quote in record {index} is malformed") from exc
            if (
                not str(quote.get("book_name") or "").strip()
                or not math.isfinite(overround)
                or not float(thresholds["minimum_book_overround"])
                <= overround
                <= float(thresholds["maximum_book_overround"])
                or not 0.0 < quote_prob < 1.0
            ):
                raise StagingBundleError(f"Accepted BFO quote in record {index} is invalid")
        csv_row = csv_rows[filename].get(key)
        if csv_row is None or key in accepted_keys[filename]:
            raise StagingBundleError(f"BFO ledger record {index} does not reconcile uniquely")
        for field in (
            "a_fair_prob",
            "b_fair_prob",
            "a_decimal_odds",
            "b_decimal_odds",
            "num_bookmakers",
        ):
            if float(csv_row[field]) != float(csv_values[field]):
                raise StagingBundleError(f"BFO ledger record {index} CSV values disagree")
        if csv_row.get("source_url") != event_page.get("url"):
            raise StagingBundleError(f"BFO ledger record {index} source URL disagrees")
        accepted_keys[filename].add(key)

    if decisions != {"accepted": 234, "rejected": 10}:
        raise StagingBundleError(f"BFO ledger decisions are incomplete: {decisions}")
    for filename, indexed in csv_rows.items():
        if accepted_keys[filename] != set(indexed):
            raise StagingBundleError(f"BFO ledger does not exactly reconcile {filename}")
    identity = _file_identity(path)
    return {
        "source_path": repo_relative,
        "raw_relative_path": relative_raw.as_posix(),
        "line_count": len(ledger_rows),
        "accepted_records": decisions["accepted"],
        "rejected_records": decisions["rejected"],
        "corrected_csv_files": csv_identities,
        "corrected_csv_aggregate_sha256": _aggregate_identities(csv_identities),
        "canonical_content_sha256": APPROVED_BFO_LEDGER_SHA256,
        "historical_parser_matches_current_inventory": parser_matches_current_inventory,
        **identity,
    }


def _scheduled_bfo_output_path(ledger_path: Path, *, repo_root: Path) -> Path:
    suffix = ".provenance.jsonl"
    if not ledger_path.name.endswith(suffix):
        raise StagingBundleError(
            "Scheduled BFO provenance must use the companion .provenance.jsonl suffix"
        )
    output = ledger_path.with_name(f"{ledger_path.name[:-len(suffix)]}.csv").resolve(
        strict=True
    )
    raw_root = (repo_root / "data" / "raw").resolve(strict=True)
    try:
        output.relative_to(raw_root)
    except ValueError as exc:
        raise StagingBundleError(
            "Scheduled BFO recovered output must stay under data/raw"
        ) from exc
    if not output.is_file():
        raise StagingBundleError("Scheduled BFO recovered output is not a file")
    return output


def _validate_scheduled_bfo_ledger(
    path: Path,
    *,
    repo_root: Path,
    inventory_payload: dict[str, Any],
) -> dict[str, object]:
    from scripts.recover_bfo_moneyline_gaps import SPORTSBOOK_DISPLAY_NAMES

    raw_root = (repo_root / "data" / "raw").resolve(strict=True)
    try:
        relative_raw = path.relative_to(raw_root)
    except ValueError as exc:
        raise StagingBundleError(
            f"Scheduled BFO provenance ledger must be under data/raw: {path}"
        ) from exc
    output = _scheduled_bfo_output_path(path, repo_root=repo_root)
    inventory_rows = {
        str(row.get("path")): row
        for row in inventory_payload.get("files", [])
        if isinstance(row, dict)
    }
    ledger_relative = path.relative_to(repo_root).as_posix()
    output_relative = output.relative_to(repo_root).as_posix()
    for evidence_path, relative in ((path, ledger_relative), (output, output_relative)):
        row = inventory_rows.get(relative)
        if (
            not isinstance(row, dict)
            or row.get("category") != "raw_input"
            or row.get("sha256") != _sha256_file(evidence_path)
            or row.get("bytes") != int(evidence_path.stat().st_size)
        ):
            raise StagingBundleError(
                f"Scheduled BFO evidence is not exactly bound by the input inventory: {relative}"
            )

    ledger_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StagingBundleError(
                    f"Scheduled BFO provenance line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(parsed, dict):
                raise StagingBundleError(
                    f"Scheduled BFO provenance line {line_number} is not an object"
                )
            ledger_rows.append(parsed)

    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        output_rows = list(csv.DictReader(handle))
    key_fields = ("event_date", "fighter_a", "fighter_b", "query_date", "offset_days")
    indexed_output: dict[tuple[object, ...], dict[str, str]] = {}
    for row in output_rows:
        try:
            key = tuple(row[field] for field in key_fields[:-1]) + (
                int(row["offset_days"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StagingBundleError("Scheduled BFO recovered CSV schema is invalid") from exc
        if key in indexed_output:
            raise StagingBundleError("Scheduled BFO recovered CSV has a duplicate fight key")
        indexed_output[key] = row

    required_top = {
        "schema_version",
        "decision",
        "recovery_key",
        "requested_fighters",
        "parser",
        "thresholds",
        "event_page",
        "matched_bfo_rows",
        "paired_quotes",
        "consensus",
        "csv_values",
        "rejection_reason",
        "output_batch",
    }
    accepted_keys: set[tuple[object, ...]] = set()
    decisions = {"accepted": 0, "rejected": 0}
    expected_output = output.resolve(strict=True)
    for index, row in enumerate(ledger_rows, start=1):
        decision = row.get("decision")
        if (
            row.get("schema_version") != 1
            or not required_top.issubset(row)
            or decision not in decisions
        ):
            raise StagingBundleError(
                f"Scheduled BFO provenance record {index} has an invalid schema or decision"
            )
        decisions[str(decision)] += 1
        output_batch = Path(str(row.get("output_batch") or ""))
        resolved_output_batch = (
            output_batch.resolve(strict=False)
            if output_batch.is_absolute()
            else (repo_root / output_batch).resolve(strict=False)
        )
        if resolved_output_batch != expected_output:
            raise StagingBundleError(
                f"Scheduled BFO provenance record {index} names a different recovered output"
            )

        parser = row.get("parser")
        parser_files = parser.get("file_sha256") if isinstance(parser, dict) else None
        if (
            not isinstance(parser, dict)
            or not GIT_SHA_RE.fullmatch(str(parser.get("git_head") or ""))
            or not SHA256_RE.fullmatch(str(parser.get("dirty_diff_sha256") or ""))
            or not isinstance(parser_files, dict)
            or set(parser_files)
            != {
                "scripts/recover_bfo_moneyline_gaps.py",
                "scripts/revalidate_bfo_recovery_file.py",
            }
            or any(
                inventory_rows.get(parser_path, {}).get("sha256") != parser_sha
                for parser_path, parser_sha in parser_files.items()
            )
        ):
            raise StagingBundleError(
                f"Scheduled BFO provenance record {index} parser identity is invalid"
            )
        recovery = row.get("recovery_key")
        requested = row.get("requested_fighters")
        if not isinstance(recovery, dict) or not isinstance(requested, dict):
            raise StagingBundleError(
                f"Scheduled BFO provenance record {index} has no fight identity"
            )
        try:
            key = tuple(str(recovery[field]) for field in key_fields[:-1]) + (
                int(recovery["offset_days"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StagingBundleError(
                f"Scheduled BFO provenance record {index} has a malformed fight key"
            ) from exc
        if (
            str(requested.get("fighter_a") or "") != str(recovery.get("fighter_a") or "")
            or str(requested.get("fighter_b") or "")
            != str(recovery.get("fighter_b") or "")
        ):
            raise StagingBundleError(
                f"Scheduled BFO provenance record {index} fighter identity disagrees"
            )
        thresholds = row.get("thresholds")
        if (
            not isinstance(thresholds, dict)
            or thresholds.get("minimum_paired_sportsbooks") != 3
        ):
            raise StagingBundleError(
                f"Scheduled BFO provenance record {index} weakens the sportsbook threshold"
            )
        if decision == "rejected":
            if not str(row.get("rejection_reason") or "").strip() or key in indexed_output:
                raise StagingBundleError(
                    f"Scheduled rejected BFO record {index} is not quarantined"
                )
            continue

        event_page = row.get("event_page")
        consensus = row.get("consensus")
        csv_values = row.get("csv_values")
        quotes = row.get("paired_quotes")
        if (
            not isinstance(event_page, dict)
            or not str(event_page.get("url") or "").startswith(("http://", "https://"))
            or not SHA256_RE.fullmatch(str(event_page.get("content_sha256") or ""))
            or not str(event_page.get("fetched_at_utc") or "").strip()
            or not isinstance(consensus, dict)
            or not isinstance(csv_values, dict)
            or not isinstance(quotes, list)
            or row.get("rejection_reason") not in ("", None)
        ):
            raise StagingBundleError(
                f"Scheduled accepted BFO record {index} lacks exact provenance"
            )
        accepted_quotes = [quote for quote in quotes if quote.get("accepted") is True]
        try:
            books = int(consensus["num_bookmakers"])
            a_prob = float(consensus["a_fair_prob"])
            b_prob = float(consensus["b_fair_prob"])
            market_ids = [int(quote["market_id"]) for quote in accepted_quotes]
        except (KeyError, TypeError, ValueError) as exc:
            raise StagingBundleError(
                f"Scheduled accepted BFO record {index} is malformed"
            ) from exc
        if (
            books < 3
            or len(accepted_quotes) != books
            or len(market_ids) != len(set(market_ids))
            or any(market_id not in SPORTSBOOK_DISPLAY_NAMES for market_id in market_ids)
            or not math.isfinite(a_prob)
            or not math.isfinite(b_prob)
            or not 0.0 < a_prob < 1.0
            or not 0.0 < b_prob < 1.0
            or abs(a_prob + b_prob - 1.0) > 1e-9
        ):
            raise StagingBundleError(
                f"Scheduled accepted BFO record {index} fails consensus semantics"
            )
        csv_row = indexed_output.get(key)
        if csv_row is None or key in accepted_keys:
            raise StagingBundleError(
                f"Scheduled accepted BFO record {index} does not reconcile uniquely"
            )
        for field in (
            "a_fair_prob",
            "b_fair_prob",
            "a_decimal_odds",
            "b_decimal_odds",
            "num_bookmakers",
        ):
            try:
                matches = float(csv_row[field]) == float(csv_values[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise StagingBundleError(
                    f"Scheduled accepted BFO record {index} has malformed CSV values"
                ) from exc
            if not matches:
                raise StagingBundleError(
                    f"Scheduled accepted BFO record {index} CSV values disagree"
                )
        if csv_row.get("source_url") != event_page.get("url"):
            raise StagingBundleError(
                f"Scheduled accepted BFO record {index} source URL disagrees"
            )
        accepted_keys.add(key)
    if accepted_keys != set(indexed_output):
        raise StagingBundleError(
            "Scheduled BFO recovered CSV contains a row without accepted provenance"
        )

    output_identity = {
        "path": output_relative,
        "staged_path": f"provenance/{output_relative}",
        "rows": len(output_rows),
        **_file_identity(output),
    }
    return {
        "source_path": ledger_relative,
        "raw_relative_path": relative_raw.as_posix(),
        "line_count": len(ledger_rows),
        "accepted_records": decisions["accepted"],
        "rejected_records": decisions["rejected"],
        "corrected_csv_files": [output_identity],
        "corrected_csv_aggregate_sha256": _aggregate_identities([output_identity]),
        "provenance_mode": "scheduled_recovery_batch",
        **_file_identity(path),
    }


def _validate_bfo_ledger(
    path: Path,
    *,
    repo_root: Path,
    inventory_payload: dict[str, Any],
    scheduled_refit: bool = False,
    allowed_additional_csv_names: set[str] | None = None,
) -> dict[str, object]:
    if not scheduled_refit:
        return _validate_fixed_bfo_ledger(
            path,
            repo_root=repo_root,
            inventory_payload=inventory_payload,
            allowed_additional_csv_names=allowed_additional_csv_names,
        )
    if (
        path.name == APPROVED_BFO_LEDGER_NAME
        and _canonical_text_sha256(path) == APPROVED_BFO_LEDGER_SHA256
    ):
        baseline = _validate_fixed_bfo_ledger(
            path,
            repo_root=repo_root,
            inventory_payload=inventory_payload,
            allowed_additional_csv_names=allowed_additional_csv_names,
        )
        baseline["provenance_mode"] = "fixed_corrected_baseline"
        return baseline
    return _validate_scheduled_bfo_ledger(
        path,
        repo_root=repo_root,
        inventory_payload=inventory_payload,
    )


def _validate_previous_scheduled_bfo_lineage(
    manifest_path: Path | None,
    *,
    readyz_path: Path,
    policy: ScheduledRefitPolicy,
    repo_root: Path,
    inventory_payload: dict[str, Any],
) -> dict[str, Any]:
    readyz = _load_json_object_without_duplicates(
        readyz_path,
        label="Previous /readyz BFO lineage evidence",
    )
    bundle = readyz.get("production_bundle")
    if readyz.get("ready") is not True or not isinstance(bundle, dict):
        raise StagingBundleError("Previous /readyz has no ready production bundle")
    parent_bundle_id = str(bundle.get("bundle_id") or "")
    parent_source_sha = str(bundle.get("source_manifest_sha256") or "").lower()
    expected_lineage_sha = str(
        bundle.get("scheduled_bfo_lineage_manifest_sha256") or ""
    ).lower()
    if not parent_bundle_id or not SHA256_RE.fullmatch(parent_source_sha):
        raise StagingBundleError("Previous /readyz BFO parent identity is incomplete")

    if not expected_lineage_sha:
        if parent_source_sha != policy.root_release["source_manifest_sha256"]:
            raise StagingBundleError(
                "Non-root parent is missing its scheduled BFO lineage identity"
            )
        if manifest_path is not None:
            raise StagingBundleError(
                "Policy-root parent cannot accept an unbound BFO lineage manifest"
            )
        batches: list[dict[str, Any]] = []
        validated_manifest_path = None
    else:
        if not SHA256_RE.fullmatch(expected_lineage_sha):
            raise StagingBundleError("Previous /readyz BFO lineage hash is invalid")
        if manifest_path is None:
            raise StagingBundleError("Previous BFO lineage manifest is required")
        validated_manifest_path = _existing_file(
            manifest_path,
            repo_root=repo_root,
            label="Previous BFO lineage manifest",
        )
        try:
            payload = bfo_lineage.validate_package(
                validated_manifest_path,
                expected_manifest_sha256=expected_lineage_sha,
            )
        except (bfo_lineage.BfoLineageError, OSError) as exc:
            raise StagingBundleError(f"Previous BFO lineage is invalid: {exc}") from exc
        batches = deepcopy(payload["batches"])

    inventory_rows = {
        str(row.get("path")): row
        for row in inventory_payload.get("files", [])
        if isinstance(row, dict)
    }
    for batch in batches:
        for label, record in (("CSV", batch["csv"]), ("ledger", batch["provenance"])):
            try:
                raw_path = (repo_root / str(record["raw_path"])).resolve(strict=True)
            except OSError as exc:
                raise StagingBundleError(
                    f"Previous BFO lineage {label} is missing from restored raw odds"
                ) from exc
            try:
                raw_path.relative_to((repo_root / "data/raw/historical_odds").resolve(strict=True))
            except ValueError as exc:
                raise StagingBundleError(
                    f"Previous BFO lineage {label} escaped raw odds"
                ) from exc
            inventory = inventory_rows.get(str(record["raw_path"]))
            if (
                not raw_path.is_file()
                or raw_path.stat().st_size != record["bytes"]
                or _sha256_file(raw_path) != record["sha256"]
                or not isinstance(inventory, dict)
                or inventory.get("category") != "raw_input"
                or inventory.get("sha256") != record["sha256"]
                or inventory.get("bytes") != record["bytes"]
            ):
                raise StagingBundleError(
                    f"Previous BFO lineage {label} is not bound by the input inventory"
                )
    return {
        "parent_bundle_id": parent_bundle_id,
        "parent_source_manifest_sha256": parent_source_sha,
        "previous_lineage_manifest_sha256": expected_lineage_sha or None,
        "manifest_path": validated_manifest_path,
        "batches": batches,
    }


def _current_bfo_lineage_batch(
    identity: dict[str, object],
) -> dict[str, Any] | None:
    if identity.get("provenance_mode") != "scheduled_recovery_batch":
        return None
    accepted = int(identity.get("accepted_records") or 0)
    if accepted <= 0:
        return None
    corrected = identity.get("corrected_csv_files")
    if not isinstance(corrected, list) or len(corrected) != 1 or not isinstance(corrected[0], dict):
        raise StagingBundleError("Scheduled BFO lineage requires one recovered CSV")
    csv_record = corrected[0]
    csv_name = Path(str(csv_record.get("path") or "")).name
    ledger_path = str(identity.get("source_path") or "")
    ledger_name = Path(ledger_path).name
    return {
        "accepted_records": accepted,
        "rejected_records": int(identity.get("rejected_records") or 0),
        "csv": {
            "raw_path": str(csv_record["path"]),
            "artifact_path": f"batches/{csv_name}",
            "sha256": str(csv_record["sha256"]),
            "bytes": int(csv_record["bytes"]),
            "rows": int(csv_record["rows"]),
        },
        "provenance": {
            "raw_path": ledger_path,
            "artifact_path": f"batches/{ledger_name}",
            "sha256": str(identity["sha256"]),
            "bytes": int(identity["bytes"]),
            "line_count": int(identity["line_count"]),
        },
    }


def _next_scheduled_bfo_lineage(
    previous: dict[str, Any],
    current_bfo_identity: dict[str, object],
) -> dict[str, Any]:
    batches = deepcopy(previous["batches"])
    current = _current_bfo_lineage_batch(current_bfo_identity)
    if current is not None:
        batches.append(current)
    payload = {
        "schema_version": bfo_lineage.SCHEMA_VERSION,
        "parent_bundle_id": previous["parent_bundle_id"],
        "parent_source_manifest_sha256": previous[
            "parent_source_manifest_sha256"
        ],
        "previous_lineage_manifest_sha256": previous[
            "previous_lineage_manifest_sha256"
        ],
        "batches": batches,
    }
    try:
        bfo_lineage.validate_manifest_payload(payload)
    except bfo_lineage.BfoLineageError as exc:
        raise StagingBundleError(f"Next BFO lineage is invalid: {exc}") from exc
    return payload


def _registered_payload(spec_name: str, *, label: str) -> dict[str, Any]:
    try:
        payload = asdict(resolve_named_training_spec(spec_name))
    except (TypeError, ValueError) as exc:
        raise StagingBundleError(f"{label} is not a registered training spec: {exc}") from exc
    return payload


def _without_fit_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(payload)
    for field in FIT_ONLY_SPEC_FIELDS:
        result.pop(field, None)
    return result


def _effective_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a deferred odds seed to its documented effective XGB seed."""
    result = _without_fit_metadata(payload)
    if result.get("odds_noise_seed") is None:
        xgb_params = result.get("xgb_params")
        if not isinstance(xgb_params, dict) or xgb_params.get("random_state") is None:
            raise StagingBundleError(
                "A deferred odds_noise_seed requires an explicit XGBoost random_state"
            )
        result["odds_noise_seed"] = int(xgb_params["random_state"])
    return result


def _load_model_artifact(path: Path, *, label: str) -> dict[str, Any]:
    try:
        import joblib

        result = joblib.load(path)
    except Exception as exc:
        raise StagingBundleError(f"Unable to load {label} artifact {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise StagingBundleError(f"{label} artifact is not a model-result mapping: {path}")
    spec = result.get("training_spec")
    feature_cols = result.get("feature_cols")
    if not isinstance(spec, dict) or not isinstance(feature_cols, list):
        raise StagingBundleError(f"{label} artifact has no complete embedded contract")
    if feature_cols != spec.get("feature_cols"):
        raise StagingBundleError(
            f"{label} feature_cols do not exactly match its embedded training spec"
        )
    return result


def _validate_contracts(
    *,
    model_paths: dict[str, Path],
    sidecar_path: Path,
    evaluation_spec_name: str,
    expected_git_head: str,
    scheduled_policy: ScheduledRefitPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    results = {
        label: _load_model_artifact(path, label=label)
        for label, path in model_paths.items()
    }
    embedded = {
        label: deepcopy(result["training_spec"]) for label, result in results.items()
    }
    for label, spec in embedded.items():
        git_hash = str(spec.get("git_hash") or "").strip().lower()
        trained_at = str(spec.get("trained_at") or "").strip()
        if git_hash != expected_git_head.lower():
            raise StagingBundleError(
                f"{label} embedded git_hash does not match the inventoried training source HEAD"
            )
        try:
            datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise StagingBundleError(
                f"{label} embedded trained_at is missing or invalid"
            ) from exc
    primary = embedded["primary"]
    fullfit_name = str(primary.get("name") or "").strip()
    if not fullfit_name:
        raise StagingBundleError("Primary embedded spec has no name")
    registered_fullfit = _registered_payload(fullfit_name, label="Full-fit spec")
    registered_evaluation = _registered_payload(
        evaluation_spec_name, label="Evaluation spec"
    )
    if scheduled_policy is None:
        required_evaluation_name = APPROVED_EVALUATION_SPEC
        required_fullfit_name = APPROVED_FULLFIT_SPEC
        required_evaluation_sha = APPROVED_EVALUATION_PAYLOAD_SHA256
        required_fullfit_sha: str | None = None
        allowed_differences = FULLFIT_ALLOWED_DIFFERENCES
        required_feature_count: int | None = None
        required_cutoff: str | None = None
        required_model_seed = 42
        required_odds_seed = 42
    else:
        contract = scheduled_policy.contract
        required_evaluation_name = str(contract["evaluation_spec_name"])
        required_fullfit_name = str(contract["fullfit_spec_name"])
        required_evaluation_sha = str(
            contract["evaluation_spec_payload_sha256"]
        ).lower()
        required_fullfit_sha = str(contract["fullfit_spec_payload_sha256"]).lower()
        allowed_differences = frozenset(contract["allowed_fullfit_differences"])
        required_feature_count = int(contract["feature_count"])
        required_cutoff = str(contract["exclusive_train_cutoff_date"])
        required_model_seed = int(scheduled_policy.payload["evaluation"]["model_seed"])
        required_odds_seed = int(
            scheduled_policy.payload["evaluation"]["odds_noise_seed"]
        )

    if (
        evaluation_spec_name != required_evaluation_name
        or fullfit_name != required_fullfit_name
    ):
        raise StagingBundleError(
            "Evaluation/full-fit pair does not match the selected immutable policy"
        )
    if _canonical_json_sha256(registered_evaluation) != required_evaluation_sha:
        raise StagingBundleError(
            "Registered evaluation payload no longer matches the selected policy"
        )
    if required_fullfit_sha is not None and _canonical_json_sha256(
        registered_fullfit
    ) != required_fullfit_sha:
        raise StagingBundleError(
            "Registered full-fit payload no longer matches the selected policy"
        )
    if _without_fit_metadata(primary) != _without_fit_metadata(registered_fullfit):
        raise StagingBundleError(
            "Primary embedded spec differs from the registered full-fit spec beyond "
            "git_hash/trained_at"
        )

    saved = _load_json_object(sidecar_path, label="Saved full-fit spec sidecar")
    if saved != primary:
        raise StagingBundleError(
            "Saved full-fit spec sidecar does not exactly match the primary embedded spec"
        )
    if embedded["logistic"] != primary:
        raise StagingBundleError(
            "Logistic embedded spec does not exactly match the primary embedded spec"
        )
    expected_no_odds = _expected_no_odds_spec_payload(primary)
    if embedded["no_odds"] != expected_no_odds:
        raise StagingBundleError(
            "No-odds embedded spec is not the exact derived name/description/features variant"
        )

    if registered_fullfit.get("odds_noise_seed") != required_odds_seed or (
        registered_fullfit.get("xgb_params") or {}
    ).get("random_state") != required_model_seed:
        raise StagingBundleError(
            "The selected full-fit contract does not pin the policy model and odds-noise seeds"
        )
    if required_feature_count is not None and (
        len(registered_fullfit.get("feature_cols") or []) != required_feature_count
        or len(registered_evaluation.get("feature_cols") or [])
        != required_feature_count
    ):
        raise StagingBundleError(
            "Evaluation/full-fit feature count does not match the scheduled policy"
        )
    if required_cutoff is not None and str(
        registered_fullfit.get("train_cutoff_date") or ""
    ) != required_cutoff:
        raise StagingBundleError(
            "Full-fit exclusive cutoff does not match the scheduled policy"
        )
    eval_compare = _effective_contract(registered_evaluation)
    fullfit_compare = _effective_contract(registered_fullfit)
    differences = {
        key
        for key in set(eval_compare) | set(fullfit_compare)
        if eval_compare.get(key) != fullfit_compare.get(key)
    }
    if differences != allowed_differences:
        raise StagingBundleError(
            "Evaluation/full-fit registered specs must differ exactly in name, "
            f"description, and cutoff; observed={sorted(differences)}"
        )
    return registered_evaluation, registered_fullfit, results


def _csv_identity(path: Path) -> dict[str, object]:
    import pandas as pd

    try:
        columns = list(pd.read_csv(path, nrows=0).columns)
        if "event_date" not in columns:
            raise StagingBundleError(f"CSV is missing event_date: {path}")
        dates = pd.read_csv(path, usecols=["event_date"])["event_date"]
    except StagingBundleError:
        raise
    except Exception as exc:
        raise StagingBundleError(f"Unable to inspect CSV {path}: {exc}") from exc
    parsed = pd.to_datetime(dates, errors="coerce")
    if len(dates) and parsed.isna().any():
        raise StagingBundleError(f"CSV contains unparseable event_date values: {path}")
    max_date = None if parsed.empty else parsed.max().date().isoformat()
    return {
        **_file_identity(path),
        "rows": int(len(dates)),
        "columns": len(columns),
        "column_names_sha256": _canonical_json_sha256(columns),
        "max_event_date": max_date,
    }


def _validate_test_set_metadata(
    *,
    test_set_path: Path,
    metadata_path: Path,
    primary_spec: dict[str, Any],
    expected_test_frame,
    expected_training_input_evidence: dict[str, Any] | None = None,
) -> dict[str, object]:
    import pandas as pd

    metadata = _load_json_object(metadata_path, label="Test-set metadata")
    test_identity = _file_identity(test_set_path)
    if str(metadata.get("test_set_sha256") or "").lower() != test_identity["sha256"]:
        raise StagingBundleError("Test-set metadata hash does not match test_set.csv")
    if metadata.get("training_spec") != primary_spec:
        raise StagingBundleError(
            "Test-set metadata training_spec does not exactly match the primary artifact"
        )
    if metadata.get("training_input_evidence") != expected_training_input_evidence:
        raise StagingBundleError(
            "Test-set metadata training_input_evidence does not match the primary artifact"
        )
    feature_cols = primary_spec.get("feature_cols")
    expected_feature_hash = _canonical_json_sha256(feature_cols)
    if metadata.get("spec_name") != primary_spec.get("name"):
        raise StagingBundleError("Test-set metadata spec_name is incorrect")
    if metadata.get("feature_count") != len(feature_cols):
        raise StagingBundleError("Test-set metadata feature_count is incorrect")
    if str(metadata.get("feature_hash") or "").lower() != expected_feature_hash:
        raise StagingBundleError("Test-set metadata feature_hash is incorrect")
    try:
        frame = pd.read_csv(test_set_path)
    except Exception as exc:
        raise StagingBundleError(f"Unable to read test_set.csv: {exc}") from exc
    if metadata.get("row_count") != len(frame):
        raise StagingBundleError("Test-set metadata row_count is incorrect")
    expected_csv = expected_test_frame.to_csv(index=False).encode("utf-8")
    if test_set_path.read_bytes() != expected_csv:
        raise StagingBundleError(
            "test_set.csv does not exactly match the split reconstructed from features.csv"
        )
    return {
        **test_identity,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "metadata_sha256": _sha256_file(metadata_path),
        "metadata_bytes": int(metadata_path.stat().st_size),
    }


def _finite_inference(
    *,
    features_path: Path,
    primary_spec: dict[str, Any],
    model_results: dict[str, dict[str, Any]],
    sample_rows: int,
) -> tuple[dict[str, object], object, object]:
    import numpy as np
    import pandas as pd

    from src.model.predict import predict_batch
    from src.model.train import _mirror_augment_training_rows, prepare_train_test
    from src.model.training_spec import materialize_and_validate_spec_features

    if sample_rows < 1:
        raise StagingBundleError("inference_sample_rows must be at least one")
    try:
        features = pd.read_csv(features_path, low_memory=False)
    except Exception as exc:
        raise StagingBundleError(f"Unable to read training features: {exc}") from exc
    declared = list(primary_spec.get("feature_cols") or [])
    missing = [column for column in declared if column not in features.columns]
    if missing:
        raise StagingBundleError(
            f"Training features are missing declared contract columns: {missing}"
        )
    try:
        registered_spec = resolve_named_training_spec(str(primary_spec.get("name") or ""))
        materialized = materialize_and_validate_spec_features(features, registered_spec)
        training, test, _ = prepare_train_test(
            materialized,
            cutoff_date=str(primary_spec.get("train_cutoff_date") or ""),
            feature_cols=declared,
            start_date=str(primary_spec.get("train_start_date") or "") or None,
            end_date=str(primary_spec.get("train_end_date") or "") or None,
        )
    except Exception as exc:
        raise StagingBundleError(f"Unable to reconstruct the training sample: {exc}") from exc
    if training.empty:
        raise StagingBundleError("No eligible training rows are available for inference")
    row_reconciliation: dict[str, object] = {}
    for label, result in model_results.items():
        observed = result.get("observed_training_rows")
        effective = result.get("effective_training_rows")
        _, mirrored = _mirror_augment_training_rows(training, result["feature_cols"])
        expected_effective = len(training) * (2 if mirrored else 1)
        if observed != len(training) or effective != expected_effective:
            raise StagingBundleError(
                f"{label} artifact training-row metadata does not match reconstructed data: "
                f"observed={observed}/{len(training)}, effective={effective}/{expected_effective}"
            )
        row_reconciliation[label] = {
            "observed_training_rows": int(observed),
            "effective_training_rows": int(effective),
            "mirror_augmentation": mirrored,
        }
    count = min(int(sample_rows), len(training))
    positions = np.linspace(0, len(training) - 1, num=count, dtype=int)
    sample = training.iloc[positions].copy()
    summary: dict[str, object] = {
        "sample_rows": count,
        "eligible_training_rows": int(len(training)),
        "reconstructed_test_rows": int(len(test)),
        "artifact_row_reconciliation": row_reconciliation,
        "sample_event_date_min": sample["event_date"].min().date().isoformat(),
        "sample_event_date_max": sample["event_date"].max().date().isoformat(),
    }
    for label, result in model_results.items():
        try:
            predicted = predict_batch(sample, model_result=result)
            probabilities = predicted["prob_a"].to_numpy(dtype=float)
        except Exception as exc:
            raise StagingBundleError(f"Finite inference failed for {label}: {exc}") from exc
        if (
            len(probabilities) != count
            or not np.isfinite(probabilities).all()
            or (probabilities < 0.0).any()
            or (probabilities > 1.0).any()
        ):
            raise StagingBundleError(
                f"{label} did not emit finite probabilities in [0, 1]"
            )
        summary[label] = {
            "finite_probability_count": int(len(probabilities)),
            "probability_min": float(probabilities.min()),
            "probability_max": float(probabilities.max()),
        }
    return summary, training, test


def _files_byte_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(1024 * 1024)
            right_chunk = right_handle.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _read_semantic_csv(path: Path, *, label: str):
    import pandas as pd

    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise StagingBundleError(f"Unable to read {label} CSV: {path}: {exc}") from exc


def _semantic_frame_equivalence(
    audit_frame: object,
    trainer_frame: object,
    *,
    label: str,
    atol: float = SEMANTIC_EQUIVALENCE_ATOL,
) -> dict[str, object]:
    import numpy as np
    import pandas as pd

    if atol < 0:
        raise StagingBundleError("Semantic-equivalence atol must be nonnegative")
    if audit_frame.shape != trainer_frame.shape:
        raise StagingBundleError(
            f"{label} semantic shape mismatch: "
            f"audit={audit_frame.shape}, trainer={trainer_frame.shape}"
        )
    audit_columns = list(audit_frame.columns)
    trainer_columns = list(trainer_frame.columns)
    if audit_columns != trainer_columns:
        raise StagingBundleError(f"{label} semantic column order mismatch")
    audit_dtypes = [str(dtype) for dtype in audit_frame.dtypes]
    trainer_dtypes = [str(dtype) for dtype in trainer_frame.dtypes]
    if audit_dtypes != trainer_dtypes:
        raise StagingBundleError(f"{label} semantic dtype mismatch")

    required_exact = ["event_date", "fighter_a", "fighter_b", "winner", "target"]
    missing_required = [column for column in required_exact if column not in audit_columns]
    if missing_required:
        raise StagingBundleError(
            f"{label} semantic identity is missing key/target columns: {missing_required}"
        )
    audit_missing = audit_frame.isna().to_numpy(dtype=bool)
    trainer_missing = trainer_frame.isna().to_numpy(dtype=bool)
    if not np.array_equal(audit_missing, trainer_missing):
        raise StagingBundleError(f"{label} semantic NaN-mask mismatch")

    for column in required_exact:
        if not audit_frame[column].equals(trainer_frame[column]):
            kind = "key" if column in {"event_date", "fighter_a", "fighter_b"} else "target"
            raise StagingBundleError(
                f"{label} semantic {kind} mismatch in column {column}"
            )

    numeric_columns: list[str] = []
    exact_nonfloat_columns: list[str] = []
    max_abs_delta = 0.0
    max_abs_delta_column: str | None = None
    numerically_changed_cells = 0
    for column in audit_columns:
        if column in required_exact:
            continue
        if pd.api.types.is_float_dtype(audit_frame[column].dtype):
            numeric_columns.append(column)
            audit_values = audit_frame[column].to_numpy(dtype=float, copy=False)
            trainer_values = trainer_frame[column].to_numpy(dtype=float, copy=False)
            if not np.array_equal(np.isposinf(audit_values), np.isposinf(trainer_values)) or not np.array_equal(
                np.isneginf(audit_values), np.isneginf(trainer_values)
            ):
                raise StagingBundleError(
                    f"{label} semantic infinity-mask mismatch in column {column}"
                )
            finite = np.isfinite(audit_values) & np.isfinite(trainer_values)
            deltas = np.abs(audit_values[finite] - trainer_values[finite])
            column_max = float(deltas.max()) if deltas.size else 0.0
            if column_max > atol:
                raise StagingBundleError(
                    f"{label} semantic numeric drift exceeds atol={atol:g} in "
                    f"column {column}: max_abs_delta={column_max:.17g}"
                )
            numerically_changed_cells += int(np.count_nonzero(deltas))
            if column_max > max_abs_delta:
                max_abs_delta = column_max
                max_abs_delta_column = column
        else:
            exact_nonfloat_columns.append(column)
            if not audit_frame[column].equals(trainer_frame[column]):
                value_kind = (
                    "integer/bool"
                    if pd.api.types.is_numeric_dtype(audit_frame[column].dtype)
                    else "nonnumeric"
                )
                raise StagingBundleError(
                    f"{label} semantic {value_kind} mismatch in column {column}"
                )

    packed_missing = np.packbits(audit_missing.reshape(-1)).tobytes()
    column_order_sha = _canonical_json_sha256(audit_columns)
    dtype_sequence_sha = _canonical_json_sha256(audit_dtypes)
    key_rows = audit_frame[["event_date", "fighter_a", "fighter_b"]]
    target_rows = audit_frame[["winner", "target"]]
    return {
        "equivalent": True,
        "rows": int(len(audit_frame)),
        "columns": int(len(audit_columns)),
        "column_order_exact": True,
        "column_order_sha256": column_order_sha,
        "dtypes_exact": True,
        "dtype_sequence_sha256": dtype_sequence_sha,
        "key_columns": ["event_date", "fighter_a", "fighter_b"],
        "key_values_and_order_exact": True,
        "key_rows_sha256": _sha256_bytes(
            key_rows.to_csv(index=False).encode("utf-8")
        ),
        "target_columns": ["winner", "target"],
        "target_values_and_order_exact": True,
        "target_rows_sha256": _sha256_bytes(
            target_rows.to_csv(index=False).encode("utf-8")
        ),
        "nan_masks_exact": True,
        "nan_mask_sha256": _sha256_bytes(packed_missing),
        "integer_bool_and_nonnumeric_values_exact": True,
        "exact_nonfloat_column_count": len(exact_nonfloat_columns),
        "numeric_column_count": len(numeric_columns),
        "numeric_rtol": 0.0,
        "numeric_atol": float(atol),
        "numeric_max_abs_delta": max_abs_delta,
        "numeric_max_abs_delta_column": max_abs_delta_column,
        "numerically_changed_cells": numerically_changed_cells,
    }


def _semantic_csv_equivalence(
    audit_path: Path,
    trainer_path: Path,
    *,
    label: str,
    atol: float = SEMANTIC_EQUIVALENCE_ATOL,
) -> dict[str, object]:
    audit_frame = _read_semantic_csv(audit_path, label=f"audit {label}")
    trainer_frame = _read_semantic_csv(trainer_path, label=f"trainer {label}")
    report = _semantic_frame_equivalence(
        audit_frame,
        trainer_frame,
        label=label,
        atol=atol,
    )
    return {
        **report,
        "audit_sha256": _sha256_file(audit_path),
        "trainer_sha256": _sha256_file(trainer_path),
        "byte_equal": _files_byte_equal(audit_path, trainer_path),
    }


def _eligible_split_equivalence(
    audit_features: object,
    trainer_features: object,
    *,
    primary_spec: dict[str, Any],
) -> dict[str, object]:
    from src.model.train import prepare_train_test
    from src.model.training_spec import materialize_and_validate_spec_features

    try:
        registered_spec = resolve_named_training_spec(str(primary_spec.get("name") or ""))
        declared = list(primary_spec.get("feature_cols") or [])

        def split(frame):
            materialized = materialize_and_validate_spec_features(frame, registered_spec)
            training, test, _ = prepare_train_test(
                materialized,
                cutoff_date=str(primary_spec.get("train_cutoff_date") or ""),
                feature_cols=declared,
                start_date=str(primary_spec.get("train_start_date") or "") or None,
                end_date=str(primary_spec.get("train_end_date") or "") or None,
            )
            return training, test

        audit_training, audit_test = split(audit_features)
        trainer_training, trainer_test = split(trainer_features)
    except Exception as exc:
        raise StagingBundleError(
            f"Unable to reconstruct audit/trainer eligibility: {exc}"
        ) from exc

    identity_columns = ["event_date", "fighter_a", "fighter_b", "winner", "target"]
    for split_label, audit_split, trainer_split in (
        ("training", audit_training, trainer_training),
        ("test", audit_test, trainer_test),
    ):
        audit_indices = [int(index) for index in audit_split.index]
        trainer_indices = [int(index) for index in trainer_split.index]
        if audit_indices != trainer_indices or not audit_split[
            identity_columns
        ].reset_index(drop=True).equals(
            trainer_split[identity_columns].reset_index(drop=True)
        ):
            raise StagingBundleError(
                f"Audit/trainer {split_label} eligibility or row identity mismatch"
            )
    training_indices = [int(index) for index in audit_training.index]
    test_indices = [int(index) for index in audit_test.index]
    return {
        "equivalent": True,
        "training_rows": len(training_indices),
        "test_rows": len(test_indices),
        "training_eligible_indices_sha256": _canonical_json_sha256(training_indices),
        "test_eligible_indices_sha256": _canonical_json_sha256(test_indices),
        "identity_columns": identity_columns,
        "identity_values_and_order_exact": True,
    }


def _probability_sha256(values: object) -> str:
    import numpy as np

    canonical = np.asarray(values, dtype="<f8")
    return _sha256_bytes(np.ascontiguousarray(canonical).tobytes())


def _prediction_invariance(
    audit_features: object,
    trainer_features: object,
    *,
    model_results: dict[str, dict[str, Any]],
) -> dict[str, object]:
    import numpy as np

    from src.model.predict import _predict_prob_a_symmetrized

    if len(audit_features) != len(trainer_features):
        raise StagingBundleError("Prediction-invariance frames have different row counts")
    report: dict[str, object] = {
        "equivalent": True,
        "rows": int(len(audit_features)),
        "xgboost_policy": "bit_identical",
        "logistic_policy": {"rtol": 0.0, "atol": SEMANTIC_EQUIVALENCE_ATOL},
    }
    for label in ("primary", "no_odds", "logistic"):
        result = model_results.get(label)
        if not isinstance(result, dict):
            raise StagingBundleError(f"Prediction-invariance model is missing: {label}")
        try:
            audit_probabilities = np.asarray(
                _predict_prob_a_symmetrized(audit_features, result), dtype=float
            )
            trainer_probabilities = np.asarray(
                _predict_prob_a_symmetrized(trainer_features, result), dtype=float
            )
        except Exception as exc:
            raise StagingBundleError(
                f"Audit/trainer prediction invariance failed for {label}: {exc}"
            ) from exc
        if (
            audit_probabilities.shape != trainer_probabilities.shape
            or audit_probabilities.shape != (len(audit_features),)
            or not np.isfinite(audit_probabilities).all()
            or not np.isfinite(trainer_probabilities).all()
            or (audit_probabilities < 0.0).any()
            or (audit_probabilities > 1.0).any()
            or (trainer_probabilities < 0.0).any()
            or (trainer_probabilities > 1.0).any()
        ):
            raise StagingBundleError(
                f"Audit/trainer prediction invariance produced invalid probabilities for {label}"
            )
        deltas = np.abs(audit_probabilities - trainer_probabilities)
        max_abs_delta = float(deltas.max()) if deltas.size else 0.0
        bit_identical = bool(np.array_equal(audit_probabilities, trainer_probabilities))
        if label in {"primary", "no_odds"}:
            if not bit_identical:
                raise StagingBundleError(
                    f"{label} audit/trainer predictions are not bit-identical; "
                    f"max_abs_delta={max_abs_delta:.17g}"
                )
        elif not np.allclose(
            audit_probabilities,
            trainer_probabilities,
            rtol=0.0,
            atol=SEMANTIC_EQUIVALENCE_ATOL,
        ):
            raise StagingBundleError(
                "logistic audit/trainer prediction drift exceeds "
                f"atol={SEMANTIC_EQUIVALENCE_ATOL:g}; "
                f"max_abs_delta={max_abs_delta:.17g}"
            )
        report[label] = {
            "probability_count": int(len(audit_probabilities)),
            "audit_probability_sha256": _probability_sha256(audit_probabilities),
            "trainer_probability_sha256": _probability_sha256(trainer_probabilities),
            "bit_identical": bit_identical,
            "max_abs_delta": max_abs_delta,
            "required_atol": 0.0
            if label in {"primary", "no_odds"}
            else SEMANTIC_EQUIVALENCE_ATOL,
        }
    return report


def _replay_trainer_preprocessing(
    *,
    audit_fights_path: Path,
    trainer_fights_path: Path,
    trainer_features_path: Path,
    fullfit_spec_name: str,
) -> dict[str, object]:
    from src.bot import (
        _load_training_dataframe,
        _prepare_policy_bound_fullfit_training_features,
    )
    from src.data.kaggle_loader import save_processed
    from src.features.build_features import build_features, save_features

    spec = resolve_named_training_spec(fullfit_spec_name)
    with tempfile.TemporaryDirectory(prefix="ufc-preprocessing-replay-") as temporary:
        replay_root = Path(temporary)
        replay_fights = replay_root / "fights_cleaned.csv"
        replay_features = replay_root / "features.csv"
        try:
            fights_frame = _load_training_dataframe(
                data_path=audit_fights_path,
                spec=spec,
            )
            save_processed(fights_frame, filename=replay_fights)
            features_frame = build_features(fights_frame)
            features_frame, _training_input_evidence = (
                _prepare_policy_bound_fullfit_training_features(
                    features_frame,
                    spec=spec,
                )
            )
            save_features(features_frame, filename=replay_features)
        except Exception as exc:
            raise StagingBundleError(
                f"Unable to replay the trainer preprocessing path: {exc}"
            ) from exc
        fights_match = _files_byte_equal(replay_fights, trainer_fights_path)
        features_match = _files_byte_equal(replay_features, trainer_features_path)
        replay_fights_identity = _file_identity(replay_fights)
        replay_features_identity = _file_identity(replay_features)
        if not fights_match or not features_match:
            raise StagingBundleError(
                "Trainer preprocessing replay does not byte-match the completed train "
                "outputs: "
                f"fights={fights_match} ({replay_fights_identity['sha256']}), "
                f"features={features_match} ({replay_features_identity['sha256']})"
            )

    audit_fights_sha = _sha256_file(audit_fights_path)
    trainer_fights_sha = _sha256_file(trainer_fights_path)
    audit_source_equals_trainer_output = audit_fights_sha == trainer_fights_sha
    if audit_source_equals_trainer_output:
        raise StagingBundleError(
            "Approved audit source must remain distinct from the completed trainer output"
        )
    return {
        "preprocessing_replay_byte_match": True,
        "audit_source_equals_trainer_output": False,
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "replay_path": [
            "src.bot._load_training_dataframe",
            "src.data.kaggle_loader.save_processed",
            "src.features.build_features.build_features",
            "src.bot._prepare_policy_bound_fullfit_training_features",
            "src.features.build_features.save_features",
        ],
        "fights": {
            "audit_source_sha256": audit_fights_sha,
            "trainer_output_sha256": trainer_fights_sha,
            "replay_output_sha256": replay_fights_identity["sha256"],
            "replay_output_bytes": replay_fights_identity["bytes"],
            "byte_match": True,
        },
        "features": {
            "trainer_output_sha256": _sha256_file(trainer_features_path),
            "replay_output_sha256": replay_features_identity["sha256"],
            "replay_output_bytes": replay_features_identity["bytes"],
            "byte_match": True,
        },
        "explanation": (
            "The audit CSV is the direct --data input. The trainer reads, parses, "
            "sorts, and reserializes it before feature construction, so benign "
            "floating-point CSV text round-trips change bytes. Replaying that exact "
            "load/build/save path reproduces the completed train files byte-for-byte."
        ),
    }


def _audit_trainer_relationship(
    *,
    audit_fights_path: Path,
    audit_features_path: Path,
    trainer_fights_path: Path,
    trainer_features_path: Path,
    primary_spec: dict[str, Any],
    model_results: dict[str, dict[str, Any]],
) -> dict[str, object]:
    from io import StringIO

    import pandas as pd

    from src.bot import _prepare_policy_bound_fullfit_training_features

    fights_report = _semantic_csv_equivalence(
        audit_fights_path,
        trainer_fights_path,
        label="fights",
    )
    audit_features = _read_semantic_csv(
        audit_features_path, label="audit features eligibility"
    )
    trainer_features = _read_semantic_csv(
        trainer_features_path, label="trainer features eligibility"
    )
    try:
        registered_spec = resolve_named_training_spec(str(primary_spec.get("name") or ""))
        audit_features, preparation_evidence = (
            _prepare_policy_bound_fullfit_training_features(
                audit_features,
                spec=registered_spec,
            )
        )
    except Exception as exc:
        raise StagingBundleError(
            f"Unable to prepare the audit feature matrix with trainer odds semantics: {exc}"
        ) from exc
    if preparation_evidence is not None:
        # Normalize through the same CSV boundary used by the completed trainer
        # before enforcing dtype/NaN/value equivalence.
        audit_features = pd.read_csv(
            StringIO(audit_features.to_csv(index=False)),
            low_memory=False,
        )
    features_report = _semantic_frame_equivalence(
        audit_features,
        trainer_features,
        label="features",
    )
    features_report.update(
        {
            "audit_sha256": _sha256_file(audit_features_path),
            "trainer_sha256": _sha256_file(trainer_features_path),
            "byte_equal": _files_byte_equal(audit_features_path, trainer_features_path),
        }
    )
    if fights_report["byte_equal"] or features_report["byte_equal"]:
        raise StagingBundleError(
            "Approved audit and trainer snapshots must retain both distinct exact identities"
        )
    eligibility = _eligible_split_equivalence(
        audit_features,
        trainer_features,
        primary_spec=primary_spec,
    )
    predictions = _prediction_invariance(
        audit_features,
        trainer_features,
        model_results=model_results,
    )
    return {
        "equivalent": True,
        "policy": {
            "numeric_rtol": 0.0,
            "numeric_atol": SEMANTIC_EQUIVALENCE_ATOL,
            "exact": [
                "shape",
                "column_order",
                "dtypes",
                "key_values_and_order",
                "target_values_and_order",
                "eligibility",
                "nan_masks",
                "nonnumeric_values",
            ],
        },
        "fights": fights_report,
        "features": features_report,
        "eligibility": eligibility,
        "prediction_invariance": predictions,
        "policy_bound_training_preparation": preparation_evidence,
    }


def _validate_policy_bound_training_input_evidence(
    *,
    features_path: Path,
    model_results: dict[str, dict[str, Any]],
    scheduled_policy: ScheduledRefitPolicy | None,
    final_track_c_binding: dict[str, Any] | None = None,
) -> dict[str, object] | None:
    """Validate the model receipts against the exact prepared feature CSV."""
    import numpy as np
    import pandas as pd

    primary_spec = model_results.get("primary", {}).get("training_spec")
    spec_name = str((primary_spec or {}).get("name") or "")
    policy_bound = bool(
        scheduled_policy is not None
        and scheduled_policy.payload.get("schema_version") == 2
        and spec_name == scheduled_policy.contract.get("fullfit_spec_name")
    )
    receipts = {
        role: result.get("training_input_evidence")
        for role, result in model_results.items()
    }
    if not policy_bound:
        if any(receipt is not None for receipt in receipts.values()):
            raise StagingBundleError(
                "Only policy-bound integrity full-fit artifacts may carry training-input evidence"
            )
        return None
    if scheduled_policy is None:
        raise StagingBundleError(
            "Policy-bound integrity full-fit artifacts require a scheduled refit policy"
        )
    if final_track_c_binding is None:
        raise StagingBundleError(
            "Policy-bound full-fit artifacts require a final Track-C PASS binding"
        )
    if set(receipts) != set(MODEL_FILENAMES) or any(
        not isinstance(receipt, dict) for receipt in receipts.values()
    ):
        raise StagingBundleError(
            "Every policy-bound model artifact must embed training-input evidence"
        )
    primary_receipt = deepcopy(receipts["primary"])
    if any(receipt != primary_receipt for receipt in receipts.values()):
        raise StagingBundleError("Policy-bound model training-input receipts do not match")

    receipt_payload = deepcopy(primary_receipt)
    receipt_sha256 = str(receipt_payload.pop("receipt_sha256", "") or "").lower()
    if receipt_sha256 != _canonical_json_sha256(receipt_payload):
        raise StagingBundleError("Policy-bound training-input receipt hash is invalid")
    evaluation = scheduled_policy.payload["evaluation"]
    expected_values = {
        "schema_version": 1,
        "preparation": "verified_t_minus_entry_model_odds",
        "policy_id": scheduled_policy.payload["policy_id"],
        "policy_sha256": scheduled_policy.sha256,
        "fullfit_spec_name": scheduled_policy.contract["fullfit_spec_name"],
        "fullfit_spec_payload_sha256": scheduled_policy.contract[
            "fullfit_spec_payload_sha256"
        ],
        "entry_offset_days": float(evaluation["entry_offset_days"]),
        "entry_offset_for_features": True,
        "require_entry_odds": True,
        "allowed_prefight_sources": list(
            evaluation["quality_allowed_prefight_sources"]
        ),
    }
    for field, expected in expected_values.items():
        if receipt_payload.get(field) != expected:
            raise StagingBundleError(
                f"Policy-bound training-input receipt differs from policy: {field}"
            )
    final_receipt_expected = {
        "final_track_c_pass_receipt_path": str(
            final_track_c_binding["receipt_path"]
        ),
        "final_track_c_pass_receipt_sha256": final_track_c_binding[
            "receipt_sha256"
        ],
        "performance_confirmation_result_sha256": final_track_c_binding[
            "confirmation_result_sha256"
        ],
        "confirmed_strategy_config_sha256": final_track_c_binding[
            "strategy_config_sha256"
        ],
        "confirmation_evaluation_input_value_sha256": final_track_c_binding[
            "confirmation_evaluation_input_value_sha256"
        ],
        "feature_contract_count": final_track_c_binding["feature_contract_count"],
        "feature_contract_sha256": final_track_c_binding[
            "feature_contract_sha256"
        ],
        "confirmation_features_value_sha256": final_track_c_binding[
            "features_value_sha256"
        ],
        "confirmation_source_features_path": str(
            final_track_c_binding["source_features_path"]
        ),
        "confirmation_source_features_sha256": final_track_c_binding[
            "source_features_sha256"
        ],
        "confirmation_odds_source_inventory_path": str(
            final_track_c_binding["odds_source_inventory_path"]
        ),
        "confirmation_odds_source_inventory_sha256": final_track_c_binding[
            "odds_source_inventory_sha256"
        ],
        "confirmation_source_fingerprint": final_track_c_binding[
            "source_fingerprint"
        ],
        "confirmation_source_inventory_path": str(
            final_track_c_binding["source_inventory_path"]
        ),
        "confirmation_source_inventory_sha256": final_track_c_binding[
            "source_inventory_sha256"
        ],
        "confirmation_source_inventory_artifact_sha256": final_track_c_binding[
            "source_inventory_artifact_sha256"
        ],
        "confirmation_environment_path": str(
            final_track_c_binding["environment_path"]
        ),
        "confirmation_environment_artifact_sha256": final_track_c_binding[
            "environment_artifact_sha256"
        ],
        "confirmation_environment_payload_sha256": final_track_c_binding[
            "environment_payload_sha256"
        ],
        "confirmation_evaluation_protocol_sha256": final_track_c_binding[
            "evaluation_protocol_sha256"
        ],
    }
    for field, expected in final_receipt_expected.items():
        if receipt_payload.get(field) != expected:
            raise StagingBundleError(
                f"Policy-bound training-input receipt differs from final PASS: {field}"
            )

    source_fights_record = receipt_payload.get("source_fights_csv")
    if not isinstance(source_fights_record, dict):
        raise StagingBundleError(
            "Policy-bound training-input receipt has no source fights binding"
        )
    source_fights_path = Path(
        str(source_fights_record.get("path") or "")
    ).resolve(strict=False)
    if (
        source_fights_path != final_track_c_binding["dataset_fights_path"]
        or source_fights_record.get("sha256")
        != final_track_c_binding["dataset_fights_sha256"]
        or not source_fights_path.is_file()
        or _sha256_file(source_fights_path)
        != final_track_c_binding["dataset_fights_sha256"]
        or source_fights_record.get("bytes") != source_fights_path.stat().st_size
    ):
        raise StagingBundleError(
            "Policy-bound training source fights differ from final Track-C PASS"
        )

    features_identity = _csv_identity(features_path)
    features_record = receipt_payload.get("features_csv")
    if (
        not isinstance(features_record, dict)
        or features_record.get("sha256") != features_identity["sha256"]
        or features_record.get("bytes") != features_identity["bytes"]
    ):
        raise StagingBundleError(
            "Policy-bound training-input receipt is not bound to prepared features.csv"
        )
    try:
        features = pd.read_csv(features_path, low_memory=False)
    except Exception as exc:
        raise StagingBundleError(
            f"Unable to inspect policy-bound training features: {exc}"
        ) from exc
    if receipt_payload.get("row_count") != len(features):
        raise StagingBundleError("Policy-bound training-input row count is stale")

    odds_columns = ["a_implied_prob", "b_implied_prob", "diff_implied_prob"]
    provenance_columns = [
        "model_odds_source_kind",
        "model_odds_source_file",
        "model_odds_observed_at",
        "model_odds_commence_time",
        "model_odds_hours_to_start",
        "model_odds_verified_prefight",
    ]
    if receipt_payload.get("prepared_odds_columns") != odds_columns or receipt_payload.get(
        "provenance_columns"
    ) != provenance_columns:
        raise StagingBundleError("Policy-bound training-input column receipt is invalid")
    missing = [
        column for column in [*odds_columns, *provenance_columns] if column not in features
    ]
    if missing:
        raise StagingBundleError(
            f"Policy-bound prepared features omit odds provenance columns: {missing}"
        )

    odds_present = features[odds_columns].notna()
    populated = odds_present.all(axis=1)
    if (odds_present.any(axis=1) & ~populated).any():
        raise StagingBundleError("Policy-bound prepared features contain partial odds rows")
    strict_verified = features["model_odds_verified_prefight"].map(
        lambda value: value is True or (isinstance(value, np.bool_) and bool(value))
    )
    allowed_sources = set(expected_values["allowed_prefight_sources"])
    hours = pd.to_numeric(features["model_odds_hours_to_start"], errors="coerce")
    minimum_hours = float(expected_values["entry_offset_days"]) * 24.0
    invalid = populated & (
        ~strict_verified
        | ~features["model_odds_source_kind"].isin(allowed_sources)
        | ~hours.ge(minimum_hours)
    )
    if invalid.any() or strict_verified[~populated].any():
        raise StagingBundleError(
            "Policy-bound prepared odds are not exclusively verified T-1 observations"
        )
    if populated.any() and not np.allclose(
        features.loc[populated, "diff_implied_prob"].to_numpy(dtype=float),
        (
            features.loc[populated, "a_implied_prob"]
            - features.loc[populated, "b_implied_prob"]
        ).to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise StagingBundleError("Policy-bound prepared odds differential is invalid")
    populated_count = int(populated.sum())
    missing_count = int((~populated).sum())
    if (
        receipt_payload.get("rows_with_verified_t_minus_entry") != populated_count
        or receipt_payload.get("rows_missing_t_minus_entry") != missing_count
    ):
        raise StagingBundleError("Policy-bound training-input coverage receipt is stale")

    return {
        "schema_version": 1,
        "artifact_receipt_sha256": receipt_sha256,
        "artifact_roles": sorted(receipts),
        "policy_id": expected_values["policy_id"],
        "policy_sha256": expected_values["policy_sha256"],
        "fullfit_spec_name": spec_name,
        "entry_offset_days": expected_values["entry_offset_days"],
        "prepared_features_sha256": features_identity["sha256"],
        "prepared_features_bytes": features_identity["bytes"],
        "source_fights_path": str(source_fights_path),
        "source_fights_sha256": source_fights_record["sha256"],
        "source_fights_bytes": source_fights_record["bytes"],
        "final_track_c_pass_receipt_sha256": final_track_c_binding[
            "receipt_sha256"
        ],
        "performance_confirmation_result_sha256": final_track_c_binding[
            "confirmation_result_sha256"
        ],
        "confirmed_strategy_config_sha256": final_track_c_binding[
            "strategy_config_sha256"
        ],
        "confirmation_evaluation_input_value_sha256": final_track_c_binding[
            "confirmation_evaluation_input_value_sha256"
        ],
        "feature_contract_count": final_track_c_binding["feature_contract_count"],
        "feature_contract_sha256": final_track_c_binding[
            "feature_contract_sha256"
        ],
        "confirmation_features_value_sha256": final_track_c_binding[
            "features_value_sha256"
        ],
        "confirmation_source_features_sha256": final_track_c_binding[
            "source_features_sha256"
        ],
        "confirmation_odds_source_inventory_sha256": final_track_c_binding[
            "odds_source_inventory_sha256"
        ],
        "confirmation_source_fingerprint": final_track_c_binding[
            "source_fingerprint"
        ],
        "confirmation_source_inventory_sha256": final_track_c_binding[
            "source_inventory_sha256"
        ],
        "confirmation_source_inventory_artifact_sha256": final_track_c_binding[
            "source_inventory_artifact_sha256"
        ],
        "confirmation_environment_artifact_sha256": final_track_c_binding[
            "environment_artifact_sha256"
        ],
        "confirmation_environment_payload_sha256": final_track_c_binding[
            "environment_payload_sha256"
        ],
        "confirmation_evaluation_protocol_sha256": final_track_c_binding[
            "evaluation_protocol_sha256"
        ],
        "row_count": int(len(features)),
        "rows_with_verified_t_minus_entry": populated_count,
        "rows_missing_t_minus_entry": missing_count,
        "prepared_odds_columns": odds_columns,
        "provenance_columns": provenance_columns,
    }


def _argv_option(argv: Sequence[str], option: str) -> str:
    matches: list[str] = []
    for index, token in enumerate(argv):
        if token == option:
            if index + 1 >= len(argv):
                raise StagingBundleError(f"Training argv option {option} has no value")
            matches.append(argv[index + 1])
        elif token.startswith(f"{option}="):
            matches.append(token.split("=", 1)[1])
    if len(matches) != 1 or not str(matches[0]).strip():
        raise StagingBundleError(
            f"Training argv must contain exactly one {option} value"
        )
    return matches[0]


def _validate_training_argv(
    argv: Sequence[str],
    *,
    repo_root: Path,
    fullfit_spec_name: str,
    candidate_models_dir: Path,
    candidate_processed_dir: Path,
    expected_fights_sha256: str,
    expected_features_sha256: str,
    scheduled_refit: bool = False,
    final_track_c_binding: dict[str, Any] | None = None,
) -> dict[str, object]:
    if not argv or any(not isinstance(token, str) or not token for token in argv):
        raise StagingBundleError("Exact training argv must be a nonempty string array")
    executable_token = argv[0]
    executable_path = Path(executable_token)
    if not executable_path.is_absolute():
        direct = (repo_root / executable_path).resolve(strict=False)
        executable_path = direct if direct.exists() else Path(
            shutil.which(executable_token) or direct
        )
    try:
        same_python = os.path.samefile(executable_path, Path(sys.executable))
    except OSError:
        same_python = False
    if not same_python:
        raise StagingBundleError(
            "Training argv Python executable must match the interpreter building the bundle: "
            f"argv={executable_path}, builder={sys.executable}"
        )
    expected_options = ["--data", "--spec", "--output-subdir"]
    if final_track_c_binding is not None:
        expected_options.append("--final-track-c-pass-receipt")
    if len(argv) != 4 + 2 * len(expected_options) or list(argv[1:4]) != [
        "-m",
        "src.bot",
        "train",
    ] or list(argv[4::2]) != expected_options:
        raise StagingBundleError(
            "Training argv must exactly be: PYTHON -m src.bot train --data PATH "
            "--spec NAME --output-subdir PATH"
            + (
                " --final-track-c-pass-receipt PATH"
                if final_track_c_binding is not None
                else ""
            )
        )
    if _argv_option(argv, "--spec") != fullfit_spec_name:
        raise StagingBundleError("Training argv --spec does not match the primary artifact")

    output_subdir_raw = _argv_option(argv, "--output-subdir")
    output_subdir = Path(output_subdir_raw)
    if output_subdir.is_absolute() or ".." in output_subdir.parts:
        raise StagingBundleError("Training --output-subdir must be safe and relative")
    expected_models = (repo_root / "models" / output_subdir).resolve(strict=False)
    expected_processed = (
        repo_root / "data" / "processed" / output_subdir
    ).resolve(strict=False)
    if expected_models != candidate_models_dir or expected_processed != candidate_processed_dir:
        raise StagingBundleError(
            "Candidate directories do not exactly match training --output-subdir"
        )

    data_raw = Path(_argv_option(argv, "--data"))
    data_path = _existing_file(data_raw, repo_root=repo_root, label="Training --data input")
    if final_track_c_binding is not None:
        receipt_argument = _existing_file(
            Path(_argv_option(argv, "--final-track-c-pass-receipt")),
            repo_root=repo_root,
            label="Training final Track-C PASS receipt",
        )
        if (
            receipt_argument != final_track_c_binding["receipt_path"]
            or _sha256_file(receipt_argument)
            != final_track_c_binding["receipt_sha256"]
            or data_path != final_track_c_binding["dataset_fights_path"]
            or _sha256_file(data_path)
            != final_track_c_binding["dataset_fights_sha256"]
        ):
            raise StagingBundleError(
                "Training argv does not bind the exact final Track-C PASS/fights"
            )
        audit_features = final_track_c_binding["features_path"]
    else:
        audit_features = None
    if data_path.name != "fights_cleaned.csv" or data_path.parent == candidate_processed_dir:
        raise StagingBundleError(
            "Training --data must be the independent audit rebuild's fights_cleaned.csv"
        )
    if final_track_c_binding is None:
        candidates_root = (repo_root / "data" / "processed" / "candidates").resolve(
            strict=True
        )
        try:
            audit_relative = data_path.parent.relative_to(candidates_root)
        except ValueError as exc:
            raise StagingBundleError(
                "Training --data audit directory must be strictly below data/processed/candidates"
            ) from exc
        if not audit_relative.parts:
            raise StagingBundleError("Training audit rebuild must use a distinct run directory")
        audit_features = _exact_child_file(
            data_path.parent,
            "features.csv",
            label="independent audit features",
        )
    candidate_fights = candidate_processed_dir / "fights_cleaned.csv"
    candidate_features = candidate_processed_dir / "features.csv"
    audit_fights_sha = _sha256_file(data_path)
    audit_features_sha = _sha256_file(audit_features)
    candidate_fights_sha = _sha256_file(candidate_fights)
    candidate_features_sha = _sha256_file(candidate_features)
    if not SHA256_RE.fullmatch(expected_fights_sha256) or not SHA256_RE.fullmatch(
        expected_features_sha256
    ):
        raise StagingBundleError("Expected corrected snapshot hashes must be 64 hex characters")
    if not scheduled_refit and (
        expected_fights_sha256.lower() != APPROVED_FIGHTS_SHA256
        or expected_features_sha256.lower() != APPROVED_FEATURES_SHA256
    ):
        raise StagingBundleError(
            "Caller snapshot hashes do not match the allowlisted corrected comparison snapshot"
        )
    if audit_fights_sha != expected_fights_sha256.lower():
        raise StagingBundleError(
            "Independent audit fights do not match the approved controlling snapshot hash"
        )
    if audit_features_sha != expected_features_sha256.lower():
        raise StagingBundleError(
            "Independent audit features do not match the approved controlling snapshot hash"
        )
    if not scheduled_refit and candidate_fights_sha != APPROVED_TRAIN_FIGHTS_SHA256:
        raise StagingBundleError(
            "Completed trainer fights do not match the approved train-output hash"
        )
    if not scheduled_refit and candidate_features_sha != APPROVED_TRAIN_FEATURES_SHA256:
        raise StagingBundleError(
            "Completed trainer features do not match the approved train-output hash"
        )
    if audit_fights_sha == candidate_fights_sha or audit_features_sha == candidate_features_sha:
        raise StagingBundleError(
            "Approved audit and completed trainer outputs must retain distinct exact identities"
        )
    return {
        "argv": list(argv),
        "argv_sha256": _canonical_json_sha256(list(argv)),
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "data_argument": str(data_raw),
        "data_input_path": data_path.relative_to(repo_root).as_posix(),
        "data_input_sha256": _sha256_file(data_path),
        "data_input_bytes": int(data_path.stat().st_size),
        "output_subdir": output_subdir.as_posix(),
        "independent_audit_snapshot": {
            "processed_dir": data_path.parent.relative_to(repo_root).as_posix(),
            "fights": {
                "path": data_path.relative_to(repo_root).as_posix(),
                "staged_path": "provenance/independent_audit_snapshot/fights_cleaned.csv",
                "sha256": audit_fights_sha,
                "bytes": int(data_path.stat().st_size),
            },
            "features": {
                "path": audit_features.relative_to(repo_root).as_posix(),
                "staged_path": "provenance/independent_audit_snapshot/features.csv",
                "sha256": audit_features_sha,
                "bytes": int(audit_features.stat().st_size),
            },
            "audit_source_equals_trainer_output": False,
            "controlling_corrected_snapshot": {
                "fights_sha256": expected_fights_sha256.lower(),
                "features_sha256": expected_features_sha256.lower(),
                "append_only_delta_used": scheduled_refit,
            },
            "completed_trainer_snapshot": {
                "fights_sha256": candidate_fights_sha,
                "features_sha256": candidate_features_sha,
                "fights_bytes": int(candidate_fights.stat().st_size),
                "features_bytes": int(candidate_features.stat().st_size),
            },
        },
    }


def _package_versions() -> dict[str, object]:
    distributions = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
        "xgboost": "xgboost",
        "joblib": "joblib",
    }
    versions: dict[str, str] = {}
    for label, distribution in distributions.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise StagingBundleError(
                f"Required training package is not installed: {distribution}"
            ) from exc
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve(strict=True)),
            "full_version": sys.version,
        },
        "packages": versions,
    }


def _selection_evidence(
    paths: Sequence[Path], *, repo_root: Path
) -> tuple[list[dict[str, object]], int]:
    if not paths:
        raise StagingBundleError("At least one selection evidence file is required")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    total = 0
    for raw_path in paths:
        path = _existing_file(raw_path, repo_root=repo_root, label="Selection evidence")
        relative = path.relative_to(repo_root).as_posix()
        lowered = relative.lower()
        if (
            not (relative.startswith("logs/") or relative.startswith("docs/"))
            or path.suffix.lower() not in SAFE_EVIDENCE_SUFFIXES
            or any(token in lowered for token in SENSITIVE_EVIDENCE_TOKENS)
        ):
            raise StagingBundleError(
                f"Selection evidence path is not in the approved safe scope: {relative}"
            )
        if relative in seen:
            raise StagingBundleError(f"Duplicate selection evidence path: {relative}")
        seen.add(relative)
        size = int(path.stat().st_size)
        if size > MAX_EVIDENCE_FILE_BYTES:
            raise StagingBundleError(
                f"Selection evidence exceeds {MAX_EVIDENCE_FILE_BYTES} bytes: {relative}"
            )
        total += size
        rows.append({"path": relative, "bytes": size, "sha256": _sha256_file(path)})
    if total > MAX_EVIDENCE_TOTAL_BYTES:
        raise StagingBundleError(
            f"Selection evidence exceeds {MAX_EVIDENCE_TOTAL_BYTES} total bytes"
        )
    return rows, total


def _portable_performance_evidence(
    binding: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Inventory the complete immutable PASS proof copied into a release.

    Original artifacts remain byte-for-byte unchanged.  The records provide
    release-relative locators so validation never depends on an ephemeral CI
    checkout after the bundle is installed.
    """

    root = repo_root.resolve(strict=True)
    roles_by_path: dict[Path, set[str]] = {}

    def add(source: Path, role: str, *, require_evidence: bool = False) -> None:
        path = Path(source).resolve(strict=True)
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise StagingBundleError(
                f"Performance evidence {role} is outside the repository: {path}"
            ) from exc
        if require_evidence and (not relative.parts or relative.parts[0] != "evidence"):
            raise StagingBundleError(
                f"Performance evidence {role} is not durable below evidence/: {relative}"
            )
        roles_by_path.setdefault(path, set()).add(role)

    add(binding["receipt_path"], "final_track_c_pass_receipt", require_evidence=True)
    add(binding["policy_path"], "final_child_policy")
    for name, path in sorted(binding["artifacts"].items()):
        add(path, f"final_track_c_artifact:{name}", require_evidence=True)

    confirmation_result_path = Path(binding["confirmation_result_path"]).resolve(
        strict=True
    )
    try:
        confirmation_result = _load_json_object(
            confirmation_result_path,
            label="Authoritative confirmation result",
        )
        claim_path = _repo_path(
            Path(str(confirmation_result["claim_path"])),
            repo_root=root,
            label="Authoritative confirmation claim",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StagingBundleError(
            f"Authoritative confirmation result has no durable claim: {exc}"
        ) from exc
    claim_root = (root / "evidence" / "confirmation_claims").resolve(strict=True)
    try:
        claim_path.relative_to(claim_root)
    except ValueError as exc:
        raise StagingBundleError(
            "Authoritative confirmation claim is outside evidence/confirmation_claims/"
        ) from exc
    claim_dir = claim_path.parent
    claim_files = sorted(path for path in claim_dir.rglob("*") if path.is_file())
    if not claim_files or confirmation_result_path not in claim_files:
        raise StagingBundleError(
            "Authoritative confirmation evidence directory is incomplete"
        )
    for path in claim_files:
        relative = path.relative_to(claim_dir).as_posix()
        role = (
            "confirmation_result"
            if path == confirmation_result_path
            else f"confirmation_dependency:{relative}"
        )
        add(path, role, require_evidence=True)

    files: list[dict[str, Any]] = []
    for source in sorted(roles_by_path, key=lambda value: value.relative_to(root).as_posix()):
        source_relative = source.relative_to(root).as_posix()
        files.append(
            {
                "roles": sorted(roles_by_path[source]),
                "source_path": source_relative,
                "staged_path": (
                    "evidence/performance_confirmation/repository/" + source_relative
                ),
                "sha256": _sha256_file(source),
                "bytes": int(source.stat().st_size),
            }
        )
    staged_paths = [str(record["staged_path"]) for record in files]
    if len(staged_paths) != len(set(staged_paths)):
        raise StagingBundleError("Portable performance evidence paths are not unique")
    role_counts: dict[str, int] = {}
    for record in files:
        for role in record["roles"]:
            role_counts[role] = role_counts.get(role, 0) + 1
    required_roles = {
        "final_track_c_pass_receipt",
        "final_child_policy",
        "confirmation_result",
        *(f"final_track_c_artifact:{name}" for name in binding["artifacts"]),
    }
    if any(role_counts.get(role) != 1 for role in required_roles):
        raise StagingBundleError("Portable performance evidence has ambiguous required roles")
    return {
        "schema_version": 1,
        "aggregate_sha256": _canonical_json_sha256(files),
        "file_count": len(files),
        "total_bytes": sum(int(record["bytes"]) for record in files),
        "files": files,
    }


def _validate_legacy_previous_rollback(
    *,
    manifest_path: Path,
    readyz_path: Path,
    repo_root: Path,
    deployed_git_sha: str,
    runtime_lookup_hashes: dict[str, str],
) -> dict[str, object]:
    if not GIT_SHA_RE.fullmatch(deployed_git_sha):
        raise StagingBundleError("Previous deployed Git SHA must be exactly 40 hex characters")
    required_runtime_hashes = {
        "processed_fights_sha256",
        "processed_features_sha256",
    }
    if set(runtime_lookup_hashes) != required_runtime_hashes:
        raise StagingBundleError(
            "Previous runtime lookup hashes must contain exactly processed fights/features"
        )
    normalized_runtime: dict[str, str] = {}
    for key, value in runtime_lookup_hashes.items():
        if not RUNTIME_HASH_KEY_RE.fullmatch(key) or not SHA256_RE.fullmatch(value):
            raise StagingBundleError(
                f"Invalid previous runtime lookup hash {key}={value!r}"
            )
        normalized_runtime[key] = value.lower()

    expected_manifest_path = (repo_root / "models/current_production_model.json").resolve(
        strict=True
    )
    if manifest_path != expected_manifest_path:
        raise StagingBundleError(
            "Previous source manifest must be models/current_production_model.json"
        )
    payload = _load_json_object(manifest_path, label="Previous source manifest")
    required_manifest_fields = (
        "bundle_id",
        "model_spec_name",
        "snapshot_max_event_date",
        "built_at",
        "git_sha",
        "processed_fights_sha256",
        "processed_features_sha256",
        "processed_fights_bytes",
        "processed_features_bytes",
        "model_sha256",
        "no_odds_model_sha256",
    )
    missing_fields = [field for field in required_manifest_fields if payload.get(field) in (None, "")]
    if missing_fields:
        raise StagingBundleError(
            f"Previous source manifest is missing rollback fields: {missing_fields}"
        )
    canonical_models = (repo_root / "models").resolve(strict=True)
    local_models: dict[str, dict[str, object]] = {}
    manifest_hash_fields = {
        "primary": "model_sha256",
        "no_odds": "no_odds_model_sha256",
        "logistic": "logistic_model_sha256",
    }
    model_results: dict[str, dict[str, Any]] = {}
    for label, filename in MODEL_FILENAMES.items():
        path = _exact_child_file(
            canonical_models, filename, label=f"previous local {label} model"
        )
        model_results[label] = _load_model_artifact(path, label=f"previous {label}")
        identity = _file_identity(path)
        pinned = str(payload.get(manifest_hash_fields[label]) or "").strip().lower()
        if pinned and pinned != identity["sha256"]:
            raise StagingBundleError(
                f"Previous source manifest {manifest_hash_fields[label]} does not match "
                f"the local rollback artifact"
            )
        local_models[label] = {
            "path": path.relative_to(repo_root).as_posix(),
            **identity,
        }
    primary_spec = deepcopy(model_results["primary"]["training_spec"])
    if model_results["logistic"]["training_spec"] != primary_spec:
        raise StagingBundleError("Previous logistic contract does not match previous primary")
    if model_results["no_odds"]["training_spec"] != _expected_no_odds_spec_payload(
        primary_spec
    ):
        raise StagingBundleError("Previous no-odds contract is not the exact derived variant")
    if primary_spec.get("name") != payload["model_spec_name"]:
        raise StagingBundleError("Previous source manifest spec does not match local models")
    previous_sidecar = _exact_child_file(
        canonical_models,
        f"{payload['model_spec_name']}_spec.json",
        label="previous saved spec sidecar",
    )
    previous_saved_spec = _load_json_object(previous_sidecar, label="Previous saved spec")
    if previous_saved_spec != primary_spec:
        raise StagingBundleError("Previous saved spec does not exactly match local models")

    local_processed: dict[str, object] = {"available": False, "mutable_lookup": True}
    processed_dir = repo_root / "data/processed"
    local_fights = processed_dir / "fights_cleaned.csv"
    local_features = processed_dir / "features.csv"
    if local_fights.is_file() and local_features.is_file():
        fights_identity = _file_identity(local_fights)
        features_identity = _file_identity(local_features)
        local_processed = {
            "available": True,
            "mutable_lookup": True,
            "fights": {
                "path": local_fights.relative_to(repo_root).as_posix(),
                **fights_identity,
                "matches_source_manifest": fights_identity["sha256"]
                == str(payload["processed_fights_sha256"]).lower(),
            },
            "features": {
                "path": local_features.relative_to(repo_root).as_posix(),
                **features_identity,
                "matches_source_manifest": features_identity["sha256"]
                == str(payload["processed_features_sha256"]).lower(),
            },
        }

    readyz = _load_json_object(readyz_path, label="Previous /readyz evidence")
    ready_bundle = readyz.get("production_bundle")
    if readyz.get("ready") is not True or not isinstance(ready_bundle, dict):
        raise StagingBundleError("Previous /readyz evidence is not a ready bundle response")
    ready_deployed = str(
        ready_bundle.get("deployed_git_sha") or ready_bundle.get("git_sha") or ""
    ).lower()
    expected_ready_fields = {
        "bundle_id": str(payload["bundle_id"]),
        "model_spec_name": str(payload["model_spec_name"]),
        **normalized_runtime,
    }
    if ready_deployed != deployed_git_sha.lower():
        raise StagingBundleError("Previous /readyz deployed SHA does not match caller identity")
    for field, expected in expected_ready_fields.items():
        if str(ready_bundle.get(field) or "").lower() != expected.lower():
            raise StagingBundleError(
                f"Previous /readyz production_bundle.{field} does not match rollback identity"
            )
    artifact_training_sha = str(primary_spec.get("git_hash") or "").lower()
    return {
        "source_manifest": {
            "path": manifest_path.relative_to(repo_root).as_posix(),
            **_file_identity(manifest_path),
            "payload_sha256": _canonical_json_sha256(payload),
            "payload": payload,
        },
        "saved_training_spec": {
            "path": previous_sidecar.relative_to(repo_root).as_posix(),
            **_file_identity(previous_sidecar),
            "payload": previous_saved_spec,
        },
        "local_model_artifacts": local_models,
        "source_manifest_immutable_training_snapshot": {
            "snapshot_max_event_date": payload["snapshot_max_event_date"],
            "processed_fights_sha256": payload["processed_fights_sha256"],
            "processed_features_sha256": payload["processed_features_sha256"],
            "processed_fights_bytes": payload["processed_fights_bytes"],
            "processed_features_bytes": payload["processed_features_bytes"],
        },
        "local_processed_lookup_observation": local_processed,
        "identity_labels": {
            "source_manifest_git_sha": str(payload["git_sha"]),
            "artifact_training_git_sha": artifact_training_sha,
            "deployed_git_sha": deployed_git_sha.lower(),
            "source_manifest_git_sha_is_stale": str(payload["git_sha"]).lower()
            != artifact_training_sha,
        },
        "deployed_git_sha": deployed_git_sha.lower(),
        "runtime_lookup_hashes": dict(sorted(normalized_runtime.items())),
        "readyz_evidence": {
            "source_path": readyz_path.relative_to(repo_root).as_posix(),
            "staged_path": "rollback/previous_readyz.json",
            **_file_identity(readyz_path),
            "payload": readyz,
            "attests_model_hashes": False,
            "model_hash_attestation_limitation": (
                "The legacy live /readyz response does not expose model hashes; local "
                "rollback artifacts are pinned separately."
            ),
        },
    }


def _validate_rich_previous_rollback(
    *,
    readyz_path: Path,
    repo_root: Path,
    scheduled_policy: ScheduledRefitPolicy,
    expected_deployed_git_sha: str,
) -> dict[str, object]:
    readyz = _load_json_object(readyz_path, label="Previous /readyz evidence")
    ready_bundle = readyz.get("production_bundle")
    if readyz.get("ready") is not True or not isinstance(ready_bundle, dict):
        raise StagingBundleError("Previous /readyz evidence is not a ready bundle response")
    required_text = (
        "bundle_id",
        "model_spec_name",
        "rich_release_id",
        "rich_release_root",
        "installed_manifest_path",
        "immutable_training_snapshot_max_event_date",
    )
    if any(not str(ready_bundle.get(field) or "").strip() for field in required_text):
        raise StagingBundleError("Previous /readyz omits rich predecessor identity fields")
    hash_fields = (
        "installed_manifest_sha256",
        "source_manifest_sha256",
        "model_sha256",
        "no_odds_model_sha256",
        "logistic_model_sha256",
        "immutable_training_fights_sha256",
        "immutable_training_features_sha256",
        "processed_fights_sha256",
        "processed_features_sha256",
    )
    if any(
        not SHA256_RE.fullmatch(str(ready_bundle.get(field) or ""))
        for field in hash_fields
    ):
        raise StagingBundleError("Previous /readyz contains an invalid predecessor hash")
    deployed_git_sha = str(ready_bundle.get("deployed_git_sha") or "").lower()
    training_source_git_sha = str(
        ready_bundle.get("training_source_git_sha")
        or ready_bundle.get("git_sha")
        or ""
    ).lower()
    if not GIT_SHA_RE.fullmatch(deployed_git_sha) or not GIT_SHA_RE.fullmatch(
        training_source_git_sha
    ):
        raise StagingBundleError("Previous /readyz omits deployed/training Git identity")
    if deployed_git_sha != expected_deployed_git_sha.lower():
        raise StagingBundleError(
            "Previous /readyz deployed Git SHA does not match the inventoried workflow checkout"
        )
    release_id = str(ready_bundle["rich_release_id"])
    release_root = str(ready_bundle["rich_release_root"])
    installed_manifest_path = str(ready_bundle["installed_manifest_path"])
    release_root_absolute = Path(release_root).is_absolute() or release_root.startswith("/")
    installed_path_absolute = Path(installed_manifest_path).is_absolute() or (
        installed_manifest_path.startswith("/")
    )
    if (
        release_id != Path(release_id).name
        or not release_root_absolute
        or not installed_path_absolute
    ):
        raise StagingBundleError("Previous /readyz rich release paths or id are unsafe")

    root = scheduled_policy.root_release
    if ready_bundle.get("bundle_id") == root["bundle_id"]:
        root_expectations = {
            "release_id": release_id,
            "source_manifest_sha256": ready_bundle["source_manifest_sha256"],
            "installed_manifest_sha256": ready_bundle[
                "installed_manifest_sha256"
            ],
            "training_source_git_sha": training_source_git_sha,
            "model_sha256": ready_bundle["model_sha256"],
            "no_odds_model_sha256": ready_bundle["no_odds_model_sha256"],
            "logistic_model_sha256": ready_bundle["logistic_model_sha256"],
            "processed_fights_sha256": ready_bundle[
                "immutable_training_fights_sha256"
            ],
            "processed_features_sha256": ready_bundle[
                "immutable_training_features_sha256"
            ],
        }
        if any(
            str(root.get(field) or "").lower() != str(value or "").lower()
            for field, value in root_expectations.items()
        ):
            raise StagingBundleError(
                "Ready root predecessor does not match scheduled policy root_release"
            )

    immutable_snapshot = {
        "snapshot_max_event_date": ready_bundle[
            "immutable_training_snapshot_max_event_date"
        ],
        "processed_fights_sha256": ready_bundle[
            "immutable_training_fights_sha256"
        ],
        "processed_features_sha256": ready_bundle[
            "immutable_training_features_sha256"
        ],
    }
    return {
        "source_manifest": {
            "sha256": str(ready_bundle["source_manifest_sha256"]).lower(),
            "attested_by": "previous_readyz",
            "bytes_local": False,
        },
        "installed_manifest": {
            "runtime_path": installed_manifest_path,
            "sha256": str(ready_bundle["installed_manifest_sha256"]).lower(),
            "attested_by": "previous_readyz",
            "bytes_local": False,
        },
        "installed_release_identity": {
            "release_id": release_id,
            "release_root": release_root,
            "source_manifest_sha256": str(
                ready_bundle["source_manifest_sha256"]
            ).lower(),
            "installed_manifest_sha256": str(
                ready_bundle["installed_manifest_sha256"]
            ).lower(),
            "store_bytes_retained": True,
        },
        "parent_model_artifacts": {
            "primary": {"sha256": str(ready_bundle["model_sha256"]).lower()},
            "no_odds": {
                "sha256": str(ready_bundle["no_odds_model_sha256"]).lower()
            },
            "logistic": {
                "sha256": str(ready_bundle["logistic_model_sha256"]).lower()
            },
        },
        "source_manifest_immutable_training_snapshot": immutable_snapshot,
        "local_processed_lookup_observation": {
            "available": False,
            "mutable_lookup": True,
            "reason": "runtime lookup is attested by readyz, separate from immutable release data",
        },
        "identity_labels": {
            "artifact_training_git_sha": training_source_git_sha,
            "deployed_git_sha": deployed_git_sha,
        },
        "deployed_git_sha": deployed_git_sha,
        "runtime_lookup_hashes": {
            "processed_features_sha256": str(
                ready_bundle["processed_features_sha256"]
            ).lower(),
            "processed_fights_sha256": str(
                ready_bundle["processed_fights_sha256"]
            ).lower(),
        },
        "readyz_evidence": {
            "source_path": readyz_path.relative_to(repo_root).as_posix(),
            "staged_path": "rollback/previous_readyz.json",
            **_file_identity(readyz_path),
            "payload": readyz,
            "attests_model_hashes": True,
            "sole_parent_identity_source": True,
        },
    }


def _validate_previous_rollback(
    *,
    manifest_path: Path | None,
    readyz_path: Path,
    repo_root: Path,
    deployed_git_sha: str | None,
    runtime_lookup_hashes: dict[str, str],
    scheduled_policy: ScheduledRefitPolicy | None = None,
    expected_scheduled_deployed_git_sha: str | None = None,
) -> dict[str, object]:
    if scheduled_policy is None:
        if manifest_path is None or deployed_git_sha is None:
            raise StagingBundleError(
                "Legacy bundle assembly requires previous manifest and deployed Git SHA"
            )
        return _validate_legacy_previous_rollback(
            manifest_path=manifest_path,
            readyz_path=readyz_path,
            repo_root=repo_root,
            deployed_git_sha=deployed_git_sha,
            runtime_lookup_hashes=runtime_lookup_hashes,
        )
    if manifest_path is not None or deployed_git_sha is not None or runtime_lookup_hashes:
        raise StagingBundleError(
            "Scheduled refit parent identity must come only from frozen /readyz; "
            "do not pass legacy predecessor manifest/hash arguments"
        )
    if not GIT_SHA_RE.fullmatch(str(expected_scheduled_deployed_git_sha or "")):
        raise StagingBundleError("Scheduled refit has no inventoried checkout Git SHA")
    return _validate_rich_previous_rollback(
        readyz_path=readyz_path,
        repo_root=repo_root,
        scheduled_policy=scheduled_policy,
        expected_deployed_git_sha=str(expected_scheduled_deployed_git_sha),
    )


def _scheduled_refit_manifest_identity(
    policy: ScheduledRefitPolicy,
    rollback: dict[str, object],
) -> dict[str, object]:
    readyz = rollback.get("readyz_evidence")
    ready_payload = readyz.get("payload") if isinstance(readyz, dict) else None
    ready_bundle = (
        ready_payload.get("production_bundle")
        if isinstance(ready_payload, dict)
        else None
    )
    if not isinstance(ready_bundle, dict):
        raise StagingBundleError(
            "Scheduled refit identity requires rich readyz predecessor evidence"
        )
    identity = {
        "policy_schema_version": policy.payload["schema_version"],
        "policy_id": policy.payload["policy_id"],
        "sha256": policy.sha256,
        "root_bundle_id": policy.root_release["bundle_id"],
        "parent_bundle_id": ready_bundle["bundle_id"],
        "parent_model_spec_name": ready_bundle["model_spec_name"],
        "parent_model_sha256": ready_bundle["model_sha256"],
        "parent_no_odds_model_sha256": ready_bundle["no_odds_model_sha256"],
        "parent_logistic_model_sha256": ready_bundle["logistic_model_sha256"],
        "parent_processed_fights_sha256": ready_bundle[
            "immutable_training_fights_sha256"
        ],
        "parent_processed_features_sha256": ready_bundle[
            "immutable_training_features_sha256"
        ],
    }
    _require_exact_keys(
        identity,
        SCHEDULED_REFIT_MANIFEST_KEYS,
        label="Scheduled refit manifest identity",
    )
    return identity


def _copy_with_identity(source: Path, destination: Path, expected_sha256: str) -> None:
    copy_file_atomically(source, destination)
    if _sha256_file(destination) != expected_sha256:
        raise StagingBundleError(
            f"Copied artifact hash mismatch: source={source}, destination={destination}"
        )


def _remove_owned_temp(path: Path, *, expected_parent: Path) -> None:
    if not path.exists():
        return
    resolved_parent = path.resolve(strict=False).parent
    if resolved_parent != expected_parent.resolve(strict=True) or not path.name.startswith(
        ".bundle-build-"
    ):
        raise StagingBundleError(f"Refusing to clean unexpected temporary path: {path}")
    shutil.rmtree(path)


def _rich_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise StagingBundleError(f"Staging manifest section {key!r} must be an object")
    return value


def _staged_file(staging_root: Path, relative: object, *, label: str) -> Path:
    raw = Path(str(relative or ""))
    if raw.is_absolute() or not raw.parts or ".." in raw.parts:
        raise StagingBundleError(f"{label} has an unsafe staged path: {relative!r}")
    path = (staging_root / raw).resolve(strict=True)
    try:
        path.relative_to(staging_root)
    except ValueError as exc:
        raise StagingBundleError(f"{label} escapes the staging root") from exc
    if not path.is_file():
        raise StagingBundleError(f"{label} is not a staged file: {path}")
    return path


def _verify_rich_file(path: Path, record: dict[str, Any], *, label: str) -> None:
    if (
        str(record.get("sha256") or "").lower() != _sha256_file(path)
        or record.get("bytes") != int(path.stat().st_size)
    ):
        raise StagingBundleError(f"{label} rich identity does not match its staged file")


def _validate_portable_performance_evidence(
    payload: dict[str, Any],
    *,
    staging_root: Path,
) -> dict[str, dict[str, Any]]:
    """Rehash the self-contained confirmation/Track-C evidence package."""

    evidence = _rich_object(payload, "performance_evidence")
    if set(evidence) != {
        "schema_version",
        "aggregate_sha256",
        "file_count",
        "total_bytes",
        "files",
    } or evidence.get("schema_version") != 1:
        raise StagingBundleError("Portable performance evidence schema is invalid")
    records = evidence.get("files")
    if not isinstance(records, list) or not records:
        raise StagingBundleError("Portable performance evidence file list is empty")
    expected_record_keys = {"roles", "source_path", "staged_path", "sha256", "bytes"}
    staged_paths: set[str] = set()
    source_paths: set[str] = set()
    role_records: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for record in records:
        if not isinstance(record, dict) or set(record) != expected_record_keys:
            raise StagingBundleError("Portable performance evidence record is malformed")
        roles = record.get("roles")
        source_path = str(record.get("source_path") or "")
        staged_path = str(record.get("staged_path") or "")
        if (
            not isinstance(roles, list)
            or not roles
            or roles != sorted(set(roles))
            or source_path in source_paths
            or staged_path in staged_paths
            or not staged_path.startswith("evidence/performance_confirmation/repository/")
        ):
            raise StagingBundleError("Portable performance evidence identity is ambiguous")
        source_raw = Path(source_path)
        staged_raw = Path(staged_path)
        if (
            source_raw.is_absolute()
            or not source_raw.parts
            or ".." in source_raw.parts
            or source_raw.as_posix() != source_path
            or staged_raw.is_absolute()
            or ".." in staged_raw.parts
            or staged_raw.as_posix() != staged_path
        ):
            raise StagingBundleError("Portable performance evidence path is unsafe")
        source_paths.add(source_path)
        staged_paths.add(staged_path)
        artifact = _staged_file(
            staging_root,
            staged_path,
            label="portable performance evidence",
        )
        _verify_rich_file(artifact, record, label="portable performance evidence")
        total_bytes += int(record["bytes"])
        for role in roles:
            if not isinstance(role, str) or not role or role in role_records:
                raise StagingBundleError("Portable performance evidence role is ambiguous")
            role_records[role] = record
    package_root = staging_root / "evidence" / "performance_confirmation" / "repository"
    actual_paths = {
        path.relative_to(staging_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != staged_paths:
        raise StagingBundleError("Portable performance evidence file allowlist differs")
    if (
        evidence.get("file_count") != len(records)
        or evidence.get("total_bytes") != total_bytes
        or evidence.get("aggregate_sha256") != _canonical_json_sha256(records)
    ):
        raise StagingBundleError("Portable performance evidence aggregate is invalid")
    required_roles = {
        "final_track_c_pass_receipt",
        "final_child_policy",
        "confirmation_result",
        "final_track_c_artifact:summary",
        "final_track_c_artifact:evaluation_index",
        "final_track_c_artifact:data_quality",
        "final_track_c_artifact:predictive",
        "final_track_c_artifact:production_bets",
    }
    if not required_roles.issubset(role_records):
        raise StagingBundleError("Portable performance evidence is incomplete")

    receipt_record = role_records["final_track_c_pass_receipt"]
    receipt = _load_json_object(
        _staged_file(staging_root, receipt_record["staged_path"], label="final receipt"),
        label="Portable final Track-C receipt",
    )
    performance = _rich_object(payload, "performance_confirmation")
    exact_hashes = {
        "final_policy_sha256": performance.get("policy_sha256"),
        "confirmation_result_sha256": performance.get(
            "confirmation_result_sha256"
        ),
        "strategy_config_sha256": performance.get("strategy_config_sha256"),
        "features_sha256": performance.get("features_sha256"),
        "features_value_sha256": performance.get("features_value_sha256"),
        "feature_contract_sha256": performance.get("feature_contract_sha256"),
        "confirmation_evaluation_input_value_sha256": performance.get(
            "confirmation_evaluation_input_value_sha256"
        ),
        "evaluation_protocol_sha256": performance.get(
            "evaluation_protocol_sha256"
        ),
        "odds_source_inventory_sha256": performance.get(
            "odds_source_inventory_sha256"
        ),
        "source_fingerprint": performance.get("source_fingerprint"),
        "source_inventory_sha256": performance.get("source_inventory_sha256"),
        "source_inventory_artifact_sha256": performance.get(
            "source_inventory_artifact_sha256"
        ),
        "environment_artifact_sha256": performance.get(
            "environment_artifact_sha256"
        ),
        "environment_payload_sha256": performance.get(
            "environment_payload_sha256"
        ),
    }
    if any(receipt.get(field) != expected for field, expected in exact_hashes.items()):
        raise StagingBundleError("Portable final Track-C receipt cross-binding differs")
    if receipt_record["sha256"] != performance.get("final_track_c_pass_receipt_sha256"):
        raise StagingBundleError("Portable final Track-C receipt hash differs")

    source_index = {record["source_path"]: record for record in records}
    required_source_hashes = {
        receipt["final_policy_path"]: receipt["final_policy_sha256"],
        receipt["confirmation_result_path"]: receipt[
            "confirmation_result_sha256"
        ],
        receipt["dataset_fights_path"]: receipt["dataset_fights_sha256"],
        receipt["source_dataset_fights_path"]: receipt[
            "source_dataset_fights_sha256"
        ],
        receipt["features_path"]: receipt["features_sha256"],
        receipt["source_features_path"]: receipt["source_features_sha256"],
        receipt["odds_source_inventory_path"]: receipt[
            "odds_source_inventory_sha256"
        ],
        receipt["source_inventory_path"]: receipt[
            "source_inventory_artifact_sha256"
        ],
        receipt["environment_path"]: receipt["environment_artifact_sha256"],
    }
    for artifact in (receipt.get("artifacts") or {}).values():
        if not isinstance(artifact, dict):
            raise StagingBundleError("Portable final Track-C artifact record is invalid")
        required_source_hashes[artifact.get("path")] = artifact.get("sha256")
    if any(
        not isinstance(source_index.get(path), dict)
        or source_index[path].get("sha256") != sha256
        for path, sha256 in required_source_hashes.items()
    ):
        raise StagingBundleError("Portable performance evidence dependency is missing")
    return role_records


def validate_rich_staged_manifest(
    manifest_path: Path,
    *,
    expected_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Read back and verify every schema-v1 rich staging identity."""
    path = manifest_path.resolve(strict=True)
    staging_root = path.parent
    payload = _load_json_object(path, label="Staging manifest")
    if expected_payload is not None and payload != expected_payload:
        raise StagingBundleError("Read-back staging manifest differs from assembled payload")
    if payload.get("staging_schema_version") != 1 or payload.get("manifest_version") != 3:
        raise StagingBundleError("Unsupported or missing rich staging manifest schema")

    scheduled_record = payload.get("scheduled_refit_policy")
    scheduled_policy: ScheduledRefitPolicy | None = None
    if scheduled_record is not None:
        if not isinstance(scheduled_record, dict):
            raise StagingBundleError("Rich scheduled_refit_policy must be an object")
        _require_exact_keys(
            scheduled_record,
            SCHEDULED_REFIT_MANIFEST_KEYS,
            label="Rich scheduled_refit_policy",
        )
        scheduled_policy_path = _staged_file(
            staging_root,
            SCHEDULED_POLICY_STAGED_PATH,
            label="scheduled refit policy",
        )
        scheduled_policy = _load_scheduled_refit_policy(
            scheduled_policy_path,
            repo_root=(
                REPO_ROOT
                if json.loads(scheduled_policy_path.read_text(encoding="utf-8")).get(
                    "schema_version"
                )
                == 2
                else staging_root
            ),
        )
        if (
            scheduled_record.get("policy_id")
            != scheduled_policy.payload["policy_id"]
            or scheduled_record.get("sha256") != scheduled_policy.sha256
            or scheduled_record.get("root_bundle_id")
            != scheduled_policy.root_release["bundle_id"]
        ):
            raise StagingBundleError(
                "Rich scheduled refit policy identity does not match its staged policy file"
            )

    expected_core_paths = {
        "model_path": staging_root / "models/xgboost_model.pkl",
        "no_odds_model_path": staging_root / "models/xgboost_no_odds_model.pkl",
        "logistic_model_path": staging_root / "models/logistic_model.pkl",
        "processed_dir": staging_root / "processed",
    }
    for field, expected in expected_core_paths.items():
        if Path(str(payload.get(field) or "")).resolve(strict=True) != expected.resolve(
            strict=True
        ):
            raise StagingBundleError(f"Rich manifest core path {field} is not exact")

    artifacts = _rich_object(payload, "model_artifacts")
    if set(artifacts) != set(MODEL_FILENAMES):
        raise StagingBundleError("Rich manifest must describe exactly all three models")
    core_model_hashes = {
        "primary": "model_sha256",
        "no_odds": "no_odds_model_sha256",
        "logistic": "logistic_model_sha256",
    }
    embedded_specs: dict[str, dict[str, Any]] = {}
    staged_model_results: dict[str, dict[str, Any]] = {}
    used_staged_paths: set[str] = set()
    if scheduled_policy is not None:
        used_staged_paths.add(SCHEDULED_POLICY_STAGED_PATH)
    for label, filename in MODEL_FILENAMES.items():
        record = artifacts[label]
        if not isinstance(record, dict) or record.get("staged_path") != f"models/{filename}":
            raise StagingBundleError(f"Rich {label} model staged path is not exact")
        model_path = _staged_file(staging_root, record["staged_path"], label=label)
        _verify_rich_file(model_path, record, label=label)
        if record["sha256"] != payload.get(core_model_hashes[label]):
            raise StagingBundleError(f"Rich {label} hash disagrees with the core manifest")
        result = _load_model_artifact(model_path, label=f"staged {label}")
        staged_model_results[label] = result
        spec = deepcopy(result["training_spec"])
        embedded_specs[label] = spec
        if (
            record.get("embedded_training_spec") != spec
            or record.get("embedded_training_spec_sha256")
            != _canonical_json_sha256(spec)
            or record.get("feature_count") != len(result["feature_cols"])
        ):
            raise StagingBundleError(f"Rich {label} embedded contract identity is invalid")
        used_staged_paths.add(str(record["staged_path"]))
    if embedded_specs["logistic"] != embedded_specs["primary"] or embedded_specs[
        "no_odds"
    ] != _expected_no_odds_spec_payload(embedded_specs["primary"]):
        raise StagingBundleError("Rich model embedded contracts do not reconcile")
    if payload.get("model_spec_name") != embedded_specs["primary"].get("name") or payload.get(
        "no_odds_model_spec_name"
    ) != embedded_specs["no_odds"].get("name"):
        raise StagingBundleError("Rich model spec names disagree with embedded contracts")

    saved_spec = _rich_object(payload, "saved_fullfit_spec")
    spec_path = _staged_file(staging_root, saved_spec.get("staged_path"), label="saved spec")
    _verify_rich_file(spec_path, saved_spec, label="saved spec")
    if (
        _load_json_object(spec_path, label="Staged saved spec") != embedded_specs["primary"]
        or saved_spec.get("payload") != embedded_specs["primary"]
    ):
        raise StagingBundleError("Rich saved full-fit spec does not reconcile")
    used_staged_paths.add(str(saved_spec["staged_path"]))

    registered = _rich_object(payload, "registered_training_specs")
    for label in ("selected_evaluation", "selected_fullfit"):
        item = registered.get(label)
        if not isinstance(item, dict) or item.get("sha256") != _canonical_json_sha256(
            item.get("payload")
        ):
            raise StagingBundleError(f"Rich registered {label} hash is invalid")
    expected_evaluation_sha = (
        str(scheduled_policy.contract["evaluation_spec_payload_sha256"]).lower()
        if scheduled_policy is not None
        else APPROVED_EVALUATION_PAYLOAD_SHA256
    )
    expected_fullfit_sha = (
        str(scheduled_policy.contract["fullfit_spec_payload_sha256"]).lower()
        if scheduled_policy is not None
        else None
    )
    allowed_differences = (
        sorted(scheduled_policy.contract["allowed_fullfit_differences"])
        if scheduled_policy is not None
        else sorted(FULLFIT_ALLOWED_DIFFERENCES)
    )
    if (
        registered["selected_evaluation"]["sha256"] != expected_evaluation_sha
        or (
            expected_fullfit_sha is not None
            and registered["selected_fullfit"]["sha256"] != expected_fullfit_sha
        )
        or _without_fit_metadata(registered["selected_fullfit"]["payload"])
        != _without_fit_metadata(embedded_specs["primary"])
    ):
        raise StagingBundleError("Rich registered full-fit contract is invalid")
    if registered.get("allowed_differences") != allowed_differences:
        raise StagingBundleError("Rich registered spec difference policy is invalid")
    if scheduled_policy is not None:
        contract = scheduled_policy.contract
        if (
            registered["selected_evaluation"]["payload"].get("name")
            != contract["evaluation_spec_name"]
            or registered["selected_fullfit"]["payload"].get("name")
            != contract["fullfit_spec_name"]
            or embedded_specs["primary"].get("name") != contract["fullfit_spec_name"]
            or embedded_specs["primary"].get("train_cutoff_date")
            != contract["exclusive_train_cutoff_date"]
            or len(embedded_specs["primary"].get("feature_cols") or [])
            != contract["feature_count"]
        ):
            raise StagingBundleError(
                "Rich model contracts do not match the scheduled refit policy"
            )

    snapshot = _rich_object(payload, "immutable_training_snapshot")
    if snapshot.get("immutable") is not True or snapshot.get(
        "snapshot_max_event_date"
    ) != payload.get("snapshot_max_event_date"):
        raise StagingBundleError("Rich immutable snapshot identity is invalid")
    for label, filename, core_hash, core_bytes in (
        ("fights", "fights_cleaned.csv", "processed_fights_sha256", "processed_fights_bytes"),
        ("features", "features.csv", "processed_features_sha256", "processed_features_bytes"),
    ):
        record = snapshot.get(label)
        if not isinstance(record, dict) or record.get("staged_path") != f"processed/{filename}":
            raise StagingBundleError(f"Rich training {label} path is invalid")
        artifact_path = _staged_file(staging_root, record["staged_path"], label=label)
        actual = _csv_identity(artifact_path)
        if any(record.get(key) != value for key, value in actual.items()):
            raise StagingBundleError(f"Rich training {label} CSV identity is invalid")
        if record["sha256"] != payload.get(core_hash) or record["bytes"] != payload.get(
            core_bytes
        ):
            raise StagingBundleError(f"Rich training {label} disagrees with core fields")
        used_staged_paths.add(str(record["staged_path"]))
    if snapshot["fights"]["max_event_date"] != snapshot["features"]["max_event_date"]:
        raise StagingBundleError("Rich training snapshot dates disagree")
    rich_final_track_c_binding: dict[str, Any] | None = None
    if scheduled_policy is not None and scheduled_policy.payload["schema_version"] == 2:
        portable_performance_roles = _validate_portable_performance_evidence(
            payload,
            staging_root=staging_root,
        )
        embedded_receipt = staged_model_results["primary"].get(
            "training_input_evidence"
        )
        if not isinstance(embedded_receipt, dict):
            raise StagingBundleError(
                "Rich policy-v2 model omits final Track-C training evidence"
            )
        try:
            rich_final_track_c_binding = (
                production_refit_contract.validate_final_track_c_pass_receipt(
                    Path(str(embedded_receipt["final_track_c_pass_receipt_path"])),
                    repo_root=REPO_ROOT,
                )
            )
            observed_confirmed_strategy = validate_confirmed_strategy_payload(
                payload.get("confirmed_strategy")
            )
        except (KeyError, ValueError, OSError) as exc:
            raise StagingBundleError(
                f"Rich final Track-C/strategy binding is invalid: {exc}"
            ) from exc
        if (
            observed_confirmed_strategy["strategy_config_sha256"]
            != rich_final_track_c_binding["strategy_config_sha256"]
        ):
            raise StagingBundleError(
                "Rich confirmed strategy differs from final Track-C PASS"
            )
        performance_record = payload.get("performance_confirmation")
        if not isinstance(performance_record, dict) or any(
            performance_record.get(field) != expected
            for field, expected in {
                "policy_sha256": rich_final_track_c_binding["policy_sha256"],
                "confirmation_result_sha256": rich_final_track_c_binding[
                    "confirmation_result_sha256"
                ],
                "final_track_c_pass_receipt_sha256": rich_final_track_c_binding[
                    "receipt_sha256"
                ],
                "strategy_config_sha256": rich_final_track_c_binding[
                    "strategy_config_sha256"
                ],
                "dataset_fights_sha256": rich_final_track_c_binding[
                    "dataset_fights_sha256"
                ],
                "features_sha256": rich_final_track_c_binding["features_sha256"],
                "features_value_sha256": rich_final_track_c_binding[
                    "features_value_sha256"
                ],
                "feature_contract_sha256": rich_final_track_c_binding[
                    "feature_contract_sha256"
                ],
                "confirmation_evaluation_input_value_sha256": rich_final_track_c_binding[
                    "confirmation_evaluation_input_value_sha256"
                ],
                "evaluation_protocol_sha256": rich_final_track_c_binding[
                    "evaluation_protocol_sha256"
                ],
                "odds_source_inventory_sha256": rich_final_track_c_binding[
                    "odds_source_inventory_sha256"
                ],
                "source_fingerprint": rich_final_track_c_binding[
                    "source_fingerprint"
                ],
                "source_inventory_sha256": rich_final_track_c_binding[
                    "source_inventory_sha256"
                ],
                "source_inventory_artifact_sha256": rich_final_track_c_binding[
                    "source_inventory_artifact_sha256"
                ],
                "environment_artifact_sha256": rich_final_track_c_binding[
                    "environment_artifact_sha256"
                ],
                "environment_payload_sha256": rich_final_track_c_binding[
                    "environment_payload_sha256"
                ],
                "performance_evidence_aggregate_sha256": payload[
                    "performance_evidence"
                ]["aggregate_sha256"],
            }.items()
        ):
            raise StagingBundleError(
                "Rich performance confirmation does not match final Track-C PASS"
            )
    observed_training_input_evidence = _validate_policy_bound_training_input_evidence(
        features_path=_staged_file(
            staging_root,
            snapshot["features"]["staged_path"],
            label="prepared training features",
        ),
        model_results=staged_model_results,
        scheduled_policy=scheduled_policy,
        final_track_c_binding=rich_final_track_c_binding,
    )
    if payload.get("training_input_evidence") != observed_training_input_evidence:
        raise StagingBundleError(
            "Rich policy-bound training-input evidence does not reconcile"
        )
    cutoff = snapshot.get("cutoff_safety")
    required_minimum_buffer = (
        int(scheduled_policy.contract["minimum_cutoff_buffer_days"])
        if scheduled_policy is not None
        else 60
    )
    try:
        cutoff_date = datetime.fromisoformat(
            str(embedded_specs["primary"]["train_cutoff_date"])
        ).date()
        snapshot_date = datetime.fromisoformat(
            str(snapshot["features"]["max_event_date"])
        ).date()
    except (KeyError, TypeError, ValueError) as exc:
        raise StagingBundleError("Rich cutoff safety identity is invalid") from exc
    snapshot_buffer_days = (cutoff_date - snapshot_date).days
    if (
        not isinstance(cutoff, dict)
        or cutoff.get("exclusive_train_cutoff_date") != cutoff_date.isoformat()
        or cutoff.get("snapshot_buffer_days") != snapshot_buffer_days
        or cutoff.get("required_minimum_buffer_days") != required_minimum_buffer
        or cutoff.get("effective_buffer_days") != snapshot_buffer_days
        or snapshot_buffer_days < required_minimum_buffer
    ):
        raise StagingBundleError("Rich cutoff safety identity is invalid")

    test_record = snapshot.get("test_set")
    if not isinstance(test_record, dict):
        raise StagingBundleError("Rich test-set identity is missing")
    test_path = _staged_file(staging_root, test_record.get("staged_path"), label="test set")
    metadata_path = _staged_file(
        staging_root, test_record.get("metadata_staged_path"), label="test metadata"
    )
    if (
        _sha256_file(test_path) != test_record.get("sha256")
        or int(test_path.stat().st_size) != test_record.get("bytes")
        or _sha256_file(metadata_path) != test_record.get("metadata_sha256")
        or int(metadata_path.stat().st_size) != test_record.get("metadata_bytes")
    ):
        raise StagingBundleError("Rich staged test-set identity is invalid")
    test_metadata_payload = _load_json_object(metadata_path, label="Staged test metadata")
    if test_metadata_payload.get("training_input_evidence") != staged_model_results[
        "primary"
    ].get("training_input_evidence"):
        raise StagingBundleError(
            "Rich staged test metadata training-input evidence does not reconcile"
        )
    used_staged_paths.update(
        {str(test_record["staged_path"]), str(test_record["metadata_staged_path"])}
    )

    source = _rich_object(payload, "source_identity")
    inventory_records = {
        "pretraining": (
            source.get("pretraining_inventory_artifact"),
            source.get("complete_pretraining_inventory"),
            "provenance/pretraining_model_input_inventory.json",
        ),
        "assembly": (
            source.get("assembly_inventory_artifact"),
            source.get("complete_assembly_inventory"),
            "provenance/assembly_model_input_inventory.json",
        ),
    }
    staged_inventories: dict[str, dict[str, Any]] = {}
    for inventory_label, (record, embedded, expected_staged_path) in inventory_records.items():
        if not isinstance(record, dict) or record.get("staged_path") != expected_staged_path:
            raise StagingBundleError(
                f"Rich {inventory_label} source inventory artifact is missing or misplaced"
            )
        inventory_path = _staged_file(
            staging_root,
            record.get("staged_path"),
            label=f"{inventory_label} source inventory",
        )
        _verify_rich_file(
            inventory_path,
            record,
            label=f"{inventory_label} source inventory",
        )
        inventory_payload = _validate_inventory_payload(
            inventory_path,
            label=f"Staged {inventory_label} source inventory",
        )
        if inventory_payload != embedded:
            raise StagingBundleError(
                f"Rich embedded and copied {inventory_label} inventories disagree"
            )
        staged_inventories[inventory_label] = inventory_payload
        used_staged_paths.add(str(record["staged_path"]))

    inventory_comparison = model_input_inventory.compare_inventories(
        staged_inventories["pretraining"],
        staged_inventories["assembly"],
    )
    observed_delta = {
        "allowlisted_changed_paths": sorted(
            ASSEMBLY_INVENTORY_ALLOWED_CHANGED_PATHS
        ),
        "added": list(inventory_comparison.get("added") or []),
        "removed": list(inventory_comparison.get("removed") or []),
        "changed": list(inventory_comparison.get("changed") or []),
        "git_head_matches": bool(inventory_comparison.get("git_head_matches")),
        "git_diff_hash_matches": bool(inventory_comparison.get("git_diff_matches")),
        "all_raw_inputs_identical": _raw_inventory(staged_inventories["pretraining"])
        == _raw_inventory(staged_inventories["assembly"]),
        "only_allowlisted_assembly_change": True,
        "pretraining_inventory_sha256": staged_inventories["pretraining"][
            "inventory_sha256"
        ],
        "assembly_inventory_sha256": staged_inventories["assembly"][
            "inventory_sha256"
        ],
    }
    if scheduled_policy is not None:
        observed_delta["scheduled_refit_mode"] = True
    observed_changed = set(observed_delta["changed"])
    allowed_observed_change = observed_changed.issubset(
        ASSEMBLY_INVENTORY_ALLOWED_CHANGED_PATHS
    )
    if (
        not allowed_observed_change
        or observed_delta["added"]
        or observed_delta["removed"]
        or observed_delta["git_head_matches"] is not True
        or observed_delta["all_raw_inputs_identical"] is not True
        or observed_delta != source.get("pretraining_to_assembly_delta")
    ):
        raise StagingBundleError("Rich dual source inventories do not reconcile")

    inventory_rows = staged_inventories["pretraining"]["files"]

    raw = _rich_object(payload, "raw_input_provenance")
    expected_raw_rows = [
        row for row in inventory_rows if row.get("category") == "raw_input"
    ]
    raw_inventory = raw.get("complete_raw_inventory")
    if not isinstance(raw_inventory, dict) or raw_inventory.get("files") != expected_raw_rows or raw_inventory.get(
        "inventory_sha256"
    ) != _aggregate_identities(expected_raw_rows):
        raise StagingBundleError("Rich raw-input inventory does not reconcile")
    ledger = raw.get("bfo_ledger")
    if not isinstance(ledger, dict):
        raise StagingBundleError("Rich BFO provenance ledger identity is missing")
    ledger_path = _staged_file(staging_root, ledger.get("staged_path"), label="BFO ledger")
    _verify_rich_file(ledger_path, ledger, label="BFO ledger")
    parsed_lines = sum(
        1
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and isinstance(json.loads(line), dict)
    )
    if parsed_lines != ledger.get("line_count"):
        raise StagingBundleError("Rich BFO provenance row count is invalid")
    corrected_csvs = ledger.get("corrected_csv_files")
    raw_by_path = {row.get("path"): row for row in expected_raw_rows}
    common_bfo_invalid = (
        not isinstance(corrected_csvs, list)
        or any(
            not isinstance(row, dict)
            or raw_by_path.get(row.get("path"), {}).get("sha256")
            != row.get("sha256")
            or raw_by_path.get(row.get("path"), {}).get("bytes")
            != row.get("bytes")
            for row in (corrected_csvs or [])
        )
        or ledger.get("corrected_csv_aggregate_sha256")
        != _aggregate_identities(corrected_csvs or [])
    )
    if scheduled_policy is None:
        bfo_invalid = (
            common_bfo_invalid
            or ledger.get("canonical_content_sha256")
            != APPROVED_BFO_LEDGER_SHA256
            or _canonical_text_sha256(ledger_path) != APPROVED_BFO_LEDGER_SHA256
            or ledger.get("accepted_records") != 234
            or ledger.get("rejected_records") != 10
            or {Path(str(row.get("path"))).name for row in corrected_csvs}
            != set(APPROVED_BFO_CSVS)
            or any(
                APPROVED_BFO_CSVS[Path(str(row.get("path"))).name][1]
                != row.get("canonical_content_sha256")
                for row in corrected_csvs
            )
        )
    else:
        bfo_invalid = (
            common_bfo_invalid
            or ledger.get("provenance_mode")
            not in {"fixed_corrected_baseline", "scheduled_recovery_batch"}
        )
        if not bfo_invalid and ledger.get("provenance_mode") == "scheduled_recovery_batch":
            if len(corrected_csvs) != 1:
                bfo_invalid = True
            else:
                record = corrected_csvs[0]
                staged_csv = _staged_file(
                    staging_root,
                    record.get("staged_path"),
                    label="scheduled BFO recovered CSV",
                )
                _verify_rich_file(
                    staged_csv,
                    record,
                    label="scheduled BFO recovered CSV",
                )
                if str(record["staged_path"]) in used_staged_paths:
                    bfo_invalid = True
                used_staged_paths.add(str(record["staged_path"]))
    if bfo_invalid:
        raise StagingBundleError("Rich corrected BFO CSV identities do not reconcile")
    used_staged_paths.add(str(ledger["staged_path"]))

    lineage_payload: dict[str, Any] | None = None
    lineage_record = raw.get("scheduled_bfo_lineage")
    if scheduled_policy is None:
        if lineage_record is not None:
            raise StagingBundleError(
                "Rich BFO lineage is allowed only in scheduled-refit mode"
            )
    else:
        expected_lineage_keys = {
            "manifest_staged_path",
            "manifest_sha256",
            "manifest_bytes",
            "batch_count",
            "batches",
        }
        if not isinstance(lineage_record, dict) or set(lineage_record) != expected_lineage_keys:
            raise StagingBundleError("Rich scheduled BFO lineage identity is missing or invalid")
        if lineage_record.get("manifest_staged_path") != (
            "provenance/bfo_lineage/manifest.json"
        ):
            raise StagingBundleError("Rich scheduled BFO lineage manifest path is not exact")
        lineage_path = _staged_file(
            staging_root,
            lineage_record["manifest_staged_path"],
            label="scheduled BFO lineage manifest",
        )
        if (
            _sha256_file(lineage_path) != lineage_record.get("manifest_sha256")
            or lineage_path.stat().st_size != lineage_record.get("manifest_bytes")
        ):
            raise StagingBundleError("Rich scheduled BFO lineage manifest identity is invalid")
        try:
            lineage_payload = bfo_lineage.validate_package(
                lineage_path,
                expected_manifest_sha256=str(lineage_record["manifest_sha256"]),
            )
        except (bfo_lineage.BfoLineageError, OSError) as exc:
            raise StagingBundleError(f"Rich scheduled BFO lineage package is invalid: {exc}") from exc
        if (
            lineage_payload.get("batches") != lineage_record.get("batches")
            or lineage_record.get("batch_count") != len(lineage_payload["batches"])
        ):
            raise StagingBundleError("Rich scheduled BFO lineage batches do not reconcile")
        used_staged_paths.add(str(lineage_record["manifest_staged_path"]))
        for batch in lineage_payload["batches"]:
            for label, record in (
                ("CSV", batch["csv"]),
                ("ledger", batch["provenance"]),
            ):
                raw_record = raw_by_path.get(record["raw_path"])
                if (
                    not isinstance(raw_record, dict)
                    or raw_record.get("sha256") != record["sha256"]
                    or raw_record.get("bytes") != record["bytes"]
                ):
                    raise StagingBundleError(
                        f"Rich scheduled BFO lineage {label} is not in the raw inventory"
                    )
                staged_path = f"provenance/bfo_lineage/{record['artifact_path']}"
                staged_batch_file = _staged_file(
                    staging_root,
                    staged_path,
                    label=f"scheduled BFO lineage {label}",
                )
                _verify_rich_file(
                    staged_batch_file,
                    record,
                    label=f"scheduled BFO lineage {label}",
                )
                if staged_path in used_staged_paths:
                    raise StagingBundleError(
                        "Rich manifest duplicates a scheduled BFO lineage artifact path"
                    )
                used_staged_paths.add(staged_path)

    evidence = _rich_object(payload, "selection_evidence")
    evidence_files = evidence.get("files")
    if not isinstance(evidence_files, list) or not evidence_files:
        raise StagingBundleError("Rich selection evidence is missing")
    evidence_identities = []
    for record in evidence_files:
        if not isinstance(record, dict):
            raise StagingBundleError("Rich selection evidence record is invalid")
        evidence_path = _staged_file(
            staging_root, record.get("staged_path"), label="selection evidence"
        )
        _verify_rich_file(evidence_path, record, label="selection evidence")
        evidence_identities.append(
            {"path": record.get("path"), "bytes": record.get("bytes"), "sha256": record.get("sha256")}
        )
        if str(record["staged_path"]) in used_staged_paths:
            raise StagingBundleError("Rich manifest duplicates a staged artifact path")
        used_staged_paths.add(str(record["staged_path"]))
    if (
        evidence.get("file_count") != len(evidence_files)
        or evidence.get("total_bytes") != sum(item["bytes"] for item in evidence_identities)
        or evidence.get("aggregate_sha256") != _aggregate_identities(evidence_identities)
    ):
        raise StagingBundleError("Rich selection evidence aggregate is invalid")

    invocation = _rich_object(payload, "training_invocation")
    if invocation.get("argv_sha256") != _canonical_json_sha256(invocation.get("argv")):
        raise StagingBundleError("Rich training argv identity is invalid")
    audit = invocation.get("independent_audit_snapshot")
    if not isinstance(audit, dict) or audit.get("audit_source_equals_trainer_output") is not False:
        raise StagingBundleError("Rich independent audit identity is invalid")
    controlling = audit.get("controlling_corrected_snapshot")
    expected_audit_hashes = {
        "fights": (
            str(controlling.get("fights_sha256") or "")
            if isinstance(controlling, dict)
            else ""
        ),
        "features": (
            str(controlling.get("features_sha256") or "")
            if isinstance(controlling, dict)
            else ""
        ),
    }
    expected_controlling = {
        "fights_sha256": expected_audit_hashes["fights"],
        "features_sha256": expected_audit_hashes["features"],
        "append_only_delta_used": scheduled_policy is not None,
    }
    if (
        controlling != expected_controlling
        or not all(SHA256_RE.fullmatch(value) for value in expected_audit_hashes.values())
        or (
            scheduled_policy is None
            and expected_controlling
            != {
                "fights_sha256": APPROVED_FIGHTS_SHA256,
                "features_sha256": APPROVED_FEATURES_SHA256,
                "append_only_delta_used": False,
            }
        )
    ):
        raise StagingBundleError("Rich controlling corrected snapshot identity is invalid")
    audit_paths: dict[str, Path] = {}
    for audit_label, filename, approved_hash in (
        ("fights", "fights_cleaned.csv", expected_audit_hashes["fights"]),
        ("features", "features.csv", expected_audit_hashes["features"]),
    ):
        record = audit.get(audit_label)
        expected_staged_path = f"provenance/independent_audit_snapshot/{filename}"
        if (
            not isinstance(record, dict)
            or record.get("staged_path") != expected_staged_path
            or record.get("sha256") != approved_hash
        ):
            raise StagingBundleError(
                f"Rich independent audit {audit_label} identity is invalid"
            )
        audit_path = _staged_file(
            staging_root,
            record.get("staged_path"),
            label=f"independent audit {audit_label}",
        )
        _verify_rich_file(
            audit_path,
            record,
            label=f"independent audit {audit_label}",
        )
        audit_paths[audit_label] = audit_path
        used_staged_paths.add(str(record["staged_path"]))

    completed = audit.get("completed_trainer_snapshot")
    expected_completed = {
        "fights_sha256": snapshot["fights"]["sha256"],
        "features_sha256": snapshot["features"]["sha256"],
        "fights_bytes": snapshot["fights"]["bytes"],
        "features_bytes": snapshot["features"]["bytes"],
    }
    if (
        completed != expected_completed
        or payload.get("processed_fights_sha256") != expected_completed["fights_sha256"]
        or payload.get("processed_features_sha256")
        != expected_completed["features_sha256"]
        or (
            scheduled_policy is None
            and (
                expected_completed["fights_sha256"] != APPROVED_TRAIN_FIGHTS_SHA256
                or expected_completed["features_sha256"]
                != APPROVED_TRAIN_FEATURES_SHA256
            )
        )
    ):
        raise StagingBundleError("Rich completed trainer snapshot identity is invalid")

    replay = audit.get("preprocessing_replay")
    if (
        not isinstance(replay, dict)
        or replay.get("preprocessing_replay_byte_match") is not True
        or replay.get("audit_source_equals_trainer_output") is not False
        or replay.get("fights", {}).get("audit_source_sha256")
        != expected_audit_hashes["fights"]
        or replay.get("fights", {}).get("trainer_output_sha256")
        != expected_completed["fights_sha256"]
        or replay.get("fights", {}).get("replay_output_sha256")
        != expected_completed["fights_sha256"]
        or replay.get("fights", {}).get("replay_output_bytes")
        != snapshot["fights"]["bytes"]
        or replay.get("features", {}).get("trainer_output_sha256")
        != expected_completed["features_sha256"]
        or replay.get("features", {}).get("replay_output_sha256")
        != expected_completed["features_sha256"]
        or replay.get("features", {}).get("replay_output_bytes")
        != snapshot["features"]["bytes"]
        or replay.get("fights", {}).get("byte_match") is not True
        or replay.get("features", {}).get("byte_match") is not True
    ):
        raise StagingBundleError("Rich preprocessing replay receipt is invalid")

    observed_relationship = _audit_trainer_relationship(
        audit_fights_path=audit_paths["fights"],
        audit_features_path=audit_paths["features"],
        trainer_fights_path=staging_root / "processed/fights_cleaned.csv",
        trainer_features_path=staging_root / "processed/features.csv",
        primary_spec=embedded_specs["primary"],
        model_results=staged_model_results,
    )
    if observed_relationship != audit.get("semantic_equivalence"):
        raise StagingBundleError(
            "Rich audit/trainer semantic or prediction diagnostics do not reconcile"
        )

    environment = _rich_object(payload, "assembly_validation_environment")
    if environment.get("provenance_level") != "deterministic_preprocessing_replay_same_interpreter" or set(
        (environment.get("packages") or {}).keys()
    ) != {"numpy", "pandas", "scikit-learn", "xgboost", "joblib"}:
        raise StagingBundleError("Rich assembly validation environment is incomplete")
    inference = _rich_object(payload, "finite_inference")
    sample_rows = inference.get("sample_rows")
    if not isinstance(sample_rows, int) or sample_rows < 1:
        raise StagingBundleError("Rich finite-inference sample identity is invalid")
    for label in MODEL_FILENAMES:
        result = inference.get(label)
        if not isinstance(result, dict) or result.get("finite_probability_count") != sample_rows:
            raise StagingBundleError(f"Rich finite inference is incomplete for {label}")

    rollback = _rich_object(payload, "previous_rollback_identity")
    source_manifest = rollback.get("source_manifest")
    readyz = rollback.get("readyz_evidence")
    if not isinstance(source_manifest, dict) or not isinstance(readyz, dict):
        raise StagingBundleError("Rich previous rollback identity is invalid")
    readyz_path = _staged_file(
        staging_root, readyz.get("staged_path"), label="previous readyz evidence"
    )
    _verify_rich_file(readyz_path, readyz, label="previous readyz evidence")
    if _load_json_object(readyz_path, label="Staged readyz evidence") != readyz.get("payload"):
        raise StagingBundleError("Rich copied /readyz evidence payload is invalid")
    used_staged_paths.add(str(readyz["staged_path"]))
    runtime_hashes = rollback.get("runtime_lookup_hashes")
    ready_bundle = readyz["payload"].get("production_bundle")
    if not isinstance(runtime_hashes, dict) or not isinstance(ready_bundle, dict) or any(
        ready_bundle.get(key) != value for key, value in runtime_hashes.items()
    ):
        raise StagingBundleError("Rich mutable runtime lookup hashes do not reconcile")

    local_lookup = rollback.get("local_processed_lookup_observation")
    if (
        not isinstance(local_lookup, dict)
        or local_lookup.get("mutable_lookup") is not True
        or "local_immutable_processed_snapshot" in rollback
    ):
        raise StagingBundleError(
            "Rich rollback identity mixes immutable training and mutable lookup data"
        )
    if scheduled_policy is None:
        if source_manifest.get("payload_sha256") != _canonical_json_sha256(
            source_manifest.get("payload")
        ):
            raise StagingBundleError("Rich previous rollback source manifest is invalid")
        old_manifest_payload = source_manifest["payload"]
        expected_old_snapshot = {
            "snapshot_max_event_date": old_manifest_payload.get("snapshot_max_event_date"),
            "processed_fights_sha256": old_manifest_payload.get("processed_fights_sha256"),
            "processed_features_sha256": old_manifest_payload.get("processed_features_sha256"),
            "processed_fights_bytes": old_manifest_payload.get("processed_fights_bytes"),
            "processed_features_bytes": old_manifest_payload.get("processed_features_bytes"),
        }
        if rollback.get("source_manifest_immutable_training_snapshot") != expected_old_snapshot:
            raise StagingBundleError(
                "Rich rollback immutable snapshot does not match its source manifest"
            )
    else:
        installed_manifest = rollback.get("installed_manifest")
        release_identity = rollback.get("installed_release_identity")
        parent_models = rollback.get("parent_model_artifacts")
        immutable_snapshot = rollback.get("source_manifest_immutable_training_snapshot")
        if not all(
            isinstance(record, dict)
            for record in (
                installed_manifest,
                release_identity,
                parent_models,
                immutable_snapshot,
            )
        ):
            raise StagingBundleError(
                "Rich scheduled rollback omits readyz-attested release identity"
            )
        if (
            source_manifest.get("bytes_local") is not False
            or source_manifest.get("attested_by") != "previous_readyz"
            or source_manifest.get("sha256")
            != ready_bundle.get("source_manifest_sha256")
            or installed_manifest.get("bytes_local") is not False
            or installed_manifest.get("attested_by") != "previous_readyz"
            or installed_manifest.get("sha256")
            != ready_bundle.get("installed_manifest_sha256")
            or installed_manifest.get("runtime_path")
            != ready_bundle.get("installed_manifest_path")
            or release_identity.get("release_id")
            != ready_bundle.get("rich_release_id")
            or release_identity.get("release_root")
            != ready_bundle.get("rich_release_root")
            or release_identity.get("source_manifest_sha256")
            != source_manifest.get("sha256")
            or release_identity.get("installed_manifest_sha256")
            != installed_manifest.get("sha256")
            or release_identity.get("store_bytes_retained") is not True
        ):
            raise StagingBundleError(
                "Rich scheduled rollback release identities do not reconcile"
            )
        if lineage_payload is None:
            raise StagingBundleError("Rich scheduled bundle omits its BFO lineage package")
        parent_lineage_sha = str(
            ready_bundle.get("scheduled_bfo_lineage_manifest_sha256") or ""
        )
        expected_previous_lineage_sha = parent_lineage_sha or None
        if (
            lineage_payload.get("parent_bundle_id") != ready_bundle.get("bundle_id")
            or lineage_payload.get("parent_source_manifest_sha256")
            != ready_bundle.get("source_manifest_sha256")
            or lineage_payload.get("previous_lineage_manifest_sha256")
            != expected_previous_lineage_sha
            or (
                not parent_lineage_sha
                and ready_bundle.get("source_manifest_sha256")
                != scheduled_policy.root_release["source_manifest_sha256"]
            )
        ):
            raise StagingBundleError(
                "Rich scheduled BFO lineage is not chained to the exact ready predecessor"
            )
        expected_parent_policy = {
            "policy_id": scheduled_policy.payload["policy_id"],
            "policy_schema_version": scheduled_policy.payload["schema_version"],
            "sha256": scheduled_policy.sha256,
            "root_bundle_id": scheduled_policy.root_release["bundle_id"],
            "parent_bundle_id": ready_bundle["bundle_id"],
            "parent_model_spec_name": ready_bundle["model_spec_name"],
            "parent_model_sha256": ready_bundle["model_sha256"],
            "parent_no_odds_model_sha256": ready_bundle["no_odds_model_sha256"],
            "parent_logistic_model_sha256": ready_bundle["logistic_model_sha256"],
            "parent_processed_fights_sha256": ready_bundle[
                "immutable_training_fights_sha256"
            ],
            "parent_processed_features_sha256": ready_bundle[
                "immutable_training_features_sha256"
            ],
        }
        if scheduled_record != expected_parent_policy:
            raise StagingBundleError(
                "Rich scheduled_refit_policy does not bind the exact ready predecessor"
            )
        expected_immutable = {
            "snapshot_max_event_date": ready_bundle[
                "immutable_training_snapshot_max_event_date"
            ],
            "processed_fights_sha256": ready_bundle[
                "immutable_training_fights_sha256"
            ],
            "processed_features_sha256": ready_bundle[
                "immutable_training_features_sha256"
            ],
        }
        expected_models = {
            "primary": {"sha256": ready_bundle["model_sha256"]},
            "no_odds": {"sha256": ready_bundle["no_odds_model_sha256"]},
            "logistic": {"sha256": ready_bundle["logistic_model_sha256"]},
        }
        if immutable_snapshot != expected_immutable or parent_models != expected_models:
            raise StagingBundleError(
                "Rich scheduled immutable/model hashes do not match readyz"
            )

    return {
        "manifest_path": str(path),
        "manifest_sha256": _sha256_file(path),
        "verified_staged_file_count": len(used_staged_paths),
        "pretraining_source_inventory_sha256": staged_inventories["pretraining"][
            "inventory_sha256"
        ],
        "assembly_source_inventory_sha256": staged_inventories["assembly"][
            "inventory_sha256"
        ],
        "selection_evidence_sha256": evidence["aggregate_sha256"],
        "rollback_readyz_sha256": readyz["sha256"],
    }


def assemble_staged_bundle(
    inputs: BundleInputs,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Build one new staging root and return its strict validation summary."""
    root = repo_root.resolve(strict=True)
    staging_root = _repo_path(
        inputs.staging_root,
        repo_root=root,
        label="Staging root",
        must_exist=False,
    )
    if staging_root == root or staging_root.exists():
        raise StagingBundleError(
            f"Staging root must be a new, non-repository directory: {staging_root}"
        )
    allowed_staging_roots = [
        (root / ".codex_stage").resolve(strict=False),
        (root / "logs").resolve(strict=False),
    ]
    if not any(
        staging_root != namespace and namespace in staging_root.parents
        for namespace in allowed_staging_roots
    ):
        raise StagingBundleError(
            "Staging root must be strictly below .codex_stage/ or logs/"
        )
    if not staging_root.parent.is_dir():
        raise StagingBundleError(
            f"Staging root parent must already exist: {staging_root.parent}"
        )
    scheduled_policy = (
        _load_scheduled_refit_policy(
            inputs.scheduled_refit_policy_path,
            repo_root=root,
        )
        if inputs.scheduled_refit_policy_path is not None
        else None
    )
    final_track_c_binding: dict[str, Any] | None = None
    if scheduled_policy is not None and scheduled_policy.payload["schema_version"] == 2:
        if inputs.final_track_c_pass_receipt_path is None:
            raise StagingBundleError(
                "Policy-v2 staging requires the final-policy Track-C PASS receipt"
            )
        try:
            final_track_c_binding = (
                production_refit_contract.validate_final_track_c_pass_receipt(
                    inputs.final_track_c_pass_receipt_path,
                    expected_policy_path=scheduled_policy.path,
                    repo_root=root,
                )
            )
        except (
            production_refit_contract.ContractInputError,
            OSError,
            ValueError,
        ) as exc:
            raise StagingBundleError(
                f"Final-policy Track-C PASS receipt is invalid: {exc}"
            ) from exc
    elif inputs.final_track_c_pass_receipt_path is not None:
        raise StagingBundleError(
            "Final-policy Track-C PASS receipt is allowed only in policy-v2 mode"
        )
    confirmed_strategy = (
        build_confirmed_strategy_payload(
            final_track_c_binding["strategy_config"],
            expected_sha256=final_track_c_binding["strategy_config_sha256"],
        )
        if final_track_c_binding is not None
        else None
    )
    performance_evidence = (
        _portable_performance_evidence(final_track_c_binding, repo_root=root)
        if final_track_c_binding is not None
        else None
    )

    candidate_models = _strict_candidate_dir(
        inputs.candidate_models_dir,
        repo_root=root,
        allowed_root=root / "models" / "candidates",
        label="Candidate models directory",
    )
    candidate_processed = _strict_candidate_dir(
        inputs.candidate_processed_dir,
        repo_root=root,
        allowed_root=root / "data" / "processed" / "candidates",
        label="Candidate processed directory",
    )
    if candidate_models == (root / "models").resolve(strict=False):
        raise StagingBundleError("Canonical models directory cannot be staged from")
    if candidate_processed == (root / "data" / "processed").resolve(strict=False):
        raise StagingBundleError("Canonical processed directory cannot be staged from")
    for source_dir in (candidate_models, candidate_processed):
        if staging_root == source_dir or source_dir in staging_root.parents:
            raise StagingBundleError("Staging root must be isolated from candidate sources")

    model_paths = {
        label: _exact_child_file(candidate_models, filename, label=f"candidate {label}")
        for label, filename in MODEL_FILENAMES.items()
    }
    processed_paths = {
        filename: _exact_child_file(
            candidate_processed, filename, label=f"candidate {filename}"
        )
        for filename in PROCESSED_FILENAMES
    }

    primary_result = _load_model_artifact(model_paths["primary"], label="primary")
    fullfit_name = str(primary_result["training_spec"].get("name") or "").strip()
    sidecar_path = _exact_child_file(
        candidate_models,
        f"{fullfit_name}_spec.json",
        label="candidate full-fit spec sidecar",
    )
    pretraining_inventory_path = _existing_file(
        inputs.input_inventory_path,
        repo_root=root,
        label="Pretraining model input inventory",
    )
    assembly_inventory_path = _existing_file(
        inputs.assembly_inventory_path,
        repo_root=root,
        label="Assembly model input inventory",
    )
    (
        pretraining_inventory,
        assembly_inventory,
        raw_inventory,
        inventory_delta,
    ) = _validate_input_inventories(
        pretraining_inventory_path,
        assembly_inventory_path,
        repo_root=root,
        scheduled_refit=scheduled_policy is not None,
    )
    registered_eval, registered_fullfit, model_results = _validate_contracts(
        model_paths=model_paths,
        sidecar_path=sidecar_path,
        evaluation_spec_name=inputs.evaluation_spec_name,
        expected_git_head=str(pretraining_inventory["git_head"]),
        scheduled_policy=scheduled_policy,
    )
    previous_readyz = _existing_file(
        inputs.previous_readyz_path,
        repo_root=root,
        label="Previous /readyz evidence",
    )
    previous_bfo_lineage: dict[str, Any] | None = None
    allowed_lineage_csv_names: set[str] = set()
    if scheduled_policy is not None:
        previous_bfo_lineage = _validate_previous_scheduled_bfo_lineage(
            inputs.previous_bfo_lineage_manifest_path,
            readyz_path=previous_readyz,
            policy=scheduled_policy,
            repo_root=root,
            inventory_payload=pretraining_inventory,
        )
        allowed_lineage_csv_names = {
            Path(str(batch["csv"]["raw_path"])).name
            for batch in previous_bfo_lineage["batches"]
        }
    elif inputs.previous_bfo_lineage_manifest_path is not None:
        raise StagingBundleError(
            "Previous BFO lineage is allowed only in scheduled-refit mode"
        )
    bfo_path = _existing_file(
        inputs.bfo_provenance_path,
        repo_root=root,
        label="BFO provenance ledger",
    )
    bfo_identity = _validate_bfo_ledger(
        bfo_path,
        repo_root=root,
        inventory_payload=pretraining_inventory,
        scheduled_refit=scheduled_policy is not None,
        allowed_additional_csv_names=allowed_lineage_csv_names,
    )
    next_bfo_lineage: dict[str, Any] | None = None
    bfo_lineage_identity: dict[str, Any] | None = None
    if previous_bfo_lineage is not None:
        next_bfo_lineage = _next_scheduled_bfo_lineage(
            previous_bfo_lineage,
            bfo_identity,
        )
        expected_recovered_csvs = set(APPROVED_BFO_CSVS) | {
            Path(str(batch["csv"]["raw_path"])).name
            for batch in next_bfo_lineage["batches"]
        } | {
            Path(str(record["path"])).name
            for record in bfo_identity["corrected_csv_files"]
        }
        actual_recovered_csvs = {
            path.name
            for path in (root / "data/raw/historical_odds").glob(
                "historical_odds_bfo_recovered_*.csv"
            )
        }
        if actual_recovered_csvs != expected_recovered_csvs:
            raise StagingBundleError(
                "Scheduled BFO recovered CSV set is not exactly baseline plus lineage"
            )
        lineage_content = bfo_lineage.manifest_bytes(next_bfo_lineage)
        bfo_lineage_identity = {
            "manifest_staged_path": "provenance/bfo_lineage/manifest.json",
            "manifest_sha256": _sha256_bytes(lineage_content),
            "manifest_bytes": len(lineage_content),
            "batch_count": len(next_bfo_lineage["batches"]),
            "batches": deepcopy(next_bfo_lineage["batches"]),
        }
    evidence_rows, evidence_total = _selection_evidence(
        inputs.selection_evidence_paths, repo_root=root
    )
    previous_manifest = (
        _existing_file(
            inputs.previous_manifest_path,
            repo_root=root,
            label="Previous source manifest",
        )
        if inputs.previous_manifest_path is not None
        else None
    )
    rollback = _validate_previous_rollback(
        manifest_path=previous_manifest,
        readyz_path=previous_readyz,
        repo_root=root,
        deployed_git_sha=inputs.previous_deployed_git_sha,
        runtime_lookup_hashes=inputs.previous_runtime_lookup_hashes,
        scheduled_policy=scheduled_policy,
        expected_scheduled_deployed_git_sha=str(pretraining_inventory["git_head"]),
    )
    scheduled_refit_identity: dict[str, object] | None = None
    if scheduled_policy is not None:
        scheduled_refit_identity = _scheduled_refit_manifest_identity(
            scheduled_policy,
            rollback,
        )
    invocation = _validate_training_argv(
        inputs.training_argv,
        repo_root=root,
        fullfit_spec_name=fullfit_name,
        candidate_models_dir=candidate_models,
        candidate_processed_dir=candidate_processed,
        expected_fights_sha256=inputs.expected_fights_sha256,
        expected_features_sha256=inputs.expected_features_sha256,
        scheduled_refit=scheduled_policy is not None,
        final_track_c_binding=final_track_c_binding,
    )
    audit_snapshot = invocation["independent_audit_snapshot"]
    audit_fights_path = root / str(audit_snapshot["fights"]["path"])
    audit_features_path = root / str(audit_snapshot["features"]["path"])
    relationship = _audit_trainer_relationship(
        audit_fights_path=audit_fights_path,
        audit_features_path=audit_features_path,
        trainer_fights_path=processed_paths["fights_cleaned.csv"],
        trainer_features_path=processed_paths["features.csv"],
        primary_spec=model_results["primary"]["training_spec"],
        model_results=model_results,
    )
    replay = _replay_trainer_preprocessing(
        audit_fights_path=audit_fights_path,
        trainer_fights_path=processed_paths["fights_cleaned.csv"],
        trainer_features_path=processed_paths["features.csv"],
        fullfit_spec_name=fullfit_name,
    )
    audit_snapshot["semantic_equivalence"] = relationship
    audit_snapshot["preprocessing_replay"] = replay

    fights_identity = _csv_identity(processed_paths["fights_cleaned.csv"])
    features_identity = _csv_identity(processed_paths["features.csv"])
    training_input_evidence = _validate_policy_bound_training_input_evidence(
        features_path=processed_paths["features.csv"],
        model_results=model_results,
        scheduled_policy=scheduled_policy,
        final_track_c_binding=final_track_c_binding,
    )
    if not fights_identity["max_event_date"] or not features_identity["max_event_date"]:
        raise StagingBundleError("Training fights/features snapshots must not be empty")
    if fights_identity["max_event_date"] != features_identity["max_event_date"]:
        raise StagingBundleError(
            "Training fights/features snapshots must have the same maximum event date"
        )
    snapshot_max_date = str(fights_identity["max_event_date"])
    try:
        cutoff_date = datetime.fromisoformat(
            str(registered_fullfit["train_cutoff_date"])
        ).date()
        snapshot_date = datetime.fromisoformat(snapshot_max_date).date()
    except (KeyError, ValueError) as exc:
        raise StagingBundleError("Training cutoff/snapshot dates must be valid ISO dates") from exc
    current_date = datetime.now(timezone.utc).date()
    snapshot_buffer_days = (cutoff_date - snapshot_date).days
    current_buffer_days = (cutoff_date - current_date).days
    required_buffer_days = snapshot_buffer_days
    minimum_buffer_days = (
        int(scheduled_policy.contract["minimum_cutoff_buffer_days"])
        if scheduled_policy is not None
        else 60
    )
    if required_buffer_days < minimum_buffer_days:
        raise StagingBundleError(
            "Full-fit training cutoff must be at least "
            f"{minimum_buffer_days} day(s) after the newest training snapshot"
        )
    inference_summary, reconstructed_training, reconstructed_test = _finite_inference(
        features_path=processed_paths["features.csv"],
        primary_spec=model_results["primary"]["training_spec"],
        model_results=model_results,
        sample_rows=inputs.inference_sample_rows,
    )
    test_identity = _validate_test_set_metadata(
        test_set_path=processed_paths["test_set.csv"],
        metadata_path=processed_paths["test_set.csv.metadata.json"],
        primary_spec=model_results["primary"]["training_spec"],
        expected_test_frame=reconstructed_test,
        expected_training_input_evidence=model_results["primary"].get(
            "training_input_evidence"
        ),
    )
    eligible_training_rows = len(reconstructed_training)

    model_identities = {
        label: {
            "source_path": path.relative_to(root).as_posix(),
            "staged_path": f"models/{MODEL_FILENAMES[label]}",
            **_file_identity(path),
            "embedded_training_spec": deepcopy(
                model_results[label]["training_spec"]
            ),
            "embedded_training_spec_sha256": _canonical_json_sha256(
                model_results[label]["training_spec"]
            ),
            "feature_count": len(model_results[label]["feature_cols"]),
        }
        for label, path in model_paths.items()
    }
    built_at = datetime.now(timezone.utc).isoformat()
    bundle_id = inputs.bundle_id or f"ufc-production-{snapshot_max_date.replace('-', '')}-{fullfit_name}"
    if not bundle_id.strip() or any(character.isspace() for character in bundle_id):
        raise StagingBundleError("bundle_id must be nonempty and contain no whitespace")

    final_models = staging_root / "models"
    final_processed = staging_root / "processed"
    manifest_path = staging_root / "staging_manifest.json"
    manifest: dict[str, object] = {
        "manifest_version": 3,
        "staging_schema_version": 1,
        "bundle_id": bundle_id,
        "model_spec_name": fullfit_name,
        "no_odds_model_spec_name": f"{fullfit_name}_no_odds",
        "model_path": str(final_models / MODEL_FILENAMES["primary"]),
        "no_odds_model_path": str(final_models / MODEL_FILENAMES["no_odds"]),
        "logistic_model_path": str(final_models / MODEL_FILENAMES["logistic"]),
        "processed_dir": str(final_processed),
        "snapshot_max_event_date": snapshot_max_date,
        "built_at": built_at,
        "manifest_updated_at": built_at,
        "git_sha": str(pretraining_inventory["git_head"]),
        "training_source_git_sha": str(pretraining_inventory["git_head"]),
        "model_sha256": model_identities["primary"]["sha256"],
        "no_odds_model_sha256": model_identities["no_odds"]["sha256"],
        "logistic_model_sha256": model_identities["logistic"]["sha256"],
        "processed_fights_sha256": fights_identity["sha256"],
        "processed_features_sha256": features_identity["sha256"],
        "processed_fights_bytes": fights_identity["bytes"],
        "processed_features_bytes": features_identity["bytes"],
        "source_identity": {
            "base_git_sha": pretraining_inventory["git_head"],
            "tracked_diff_sha256": pretraining_inventory["git_diff_sha256"],
            "pre_training_dirty_status": {
                "git_dirty": pretraining_inventory.get("git_dirty"),
                "git_status_sha256": pretraining_inventory.get("git_status_sha256"),
                "git_status": pretraining_inventory.get("git_status"),
            },
            "assembly_dirty_status": {
                "git_dirty": assembly_inventory.get("git_dirty"),
                "git_status_sha256": assembly_inventory.get("git_status_sha256"),
                "git_status": assembly_inventory.get("git_status"),
            },
            "pretraining_inventory_artifact": {
                "role": "frozen_model_and_raw_input_provenance",
                "source_path": pretraining_inventory_path.relative_to(root).as_posix(),
                "staged_path": "provenance/pretraining_model_input_inventory.json",
                **_file_identity(pretraining_inventory_path),
            },
            "assembly_inventory_artifact": {
                "role": "current_assembly_source_and_raw_input_identity",
                "source_path": assembly_inventory_path.relative_to(root).as_posix(),
                "staged_path": "provenance/assembly_model_input_inventory.json",
                **_file_identity(assembly_inventory_path),
            },
            "complete_pretraining_inventory": pretraining_inventory,
            "complete_assembly_inventory": assembly_inventory,
            "pretraining_to_assembly_delta": inventory_delta,
        },
        "registered_training_specs": {
            "selected_evaluation": {
                "payload": registered_eval,
                "sha256": _canonical_json_sha256(registered_eval),
            },
            "selected_fullfit": {
                "payload": registered_fullfit,
                "sha256": _canonical_json_sha256(registered_fullfit),
            },
            "allowed_differences": sorted(
                scheduled_policy.contract["allowed_fullfit_differences"]
                if scheduled_policy is not None
                else FULLFIT_ALLOWED_DIFFERENCES
            ),
        },
        "model_artifacts": model_identities,
        "saved_fullfit_spec": {
            "source_path": sidecar_path.relative_to(root).as_posix(),
            "staged_path": f"models/{sidecar_path.name}",
            **_file_identity(sidecar_path),
            "payload": model_results["primary"]["training_spec"],
        },
        "immutable_training_snapshot": {
            "immutable": True,
            "snapshot_max_event_date": snapshot_max_date,
            "eligible_training_rows": eligible_training_rows,
            "cutoff_safety": {
                "exclusive_train_cutoff_date": cutoff_date.isoformat(),
                "validation_current_utc_date": current_date.isoformat(),
                "snapshot_buffer_days": snapshot_buffer_days,
                "current_date_buffer_days": current_buffer_days,
                "required_minimum_buffer_days": minimum_buffer_days,
                "effective_buffer_days": required_buffer_days,
            },
            "fights": {
                "source_path": processed_paths["fights_cleaned.csv"].relative_to(root).as_posix(),
                "staged_path": "processed/fights_cleaned.csv",
                **fights_identity,
            },
            "features": {
                "source_path": processed_paths["features.csv"].relative_to(root).as_posix(),
                "staged_path": "processed/features.csv",
                **features_identity,
            },
            "test_set": {
                "source_path": processed_paths["test_set.csv"].relative_to(root).as_posix(),
                "staged_path": "processed/test_set.csv",
                "metadata_staged_path": "processed/test_set.csv.metadata.json",
                **test_identity,
            },
        },
        "raw_input_provenance": {
            "complete_raw_inventory": raw_inventory,
            "bfo_ledger": {
                **bfo_identity,
                "staged_path": f"provenance/{bfo_path.relative_to(root).as_posix()}",
            },
            **(
                {"scheduled_bfo_lineage": bfo_lineage_identity}
                if bfo_lineage_identity is not None
                else {}
            ),
        },
        "selection_evidence": {
            "aggregate_sha256": _aggregate_identities(evidence_rows),
            "file_count": len(evidence_rows),
            "total_bytes": evidence_total,
            "files": [
                {
                    **row,
                    "source_path": row["path"],
                    "staged_path": f"evidence/{row['path']}",
                }
                for row in evidence_rows
            ],
        },
        "training_invocation": invocation,
        "assembly_validation_environment": {
            **_package_versions(),
            "provenance_level": "deterministic_preprocessing_replay_same_interpreter",
            "statement": (
                "The assembler replayed the trainer's exact load/build/save preprocessing "
                "path with this interpreter and required byte-identical train outputs."
            ),
        },
        "finite_inference": inference_summary,
        "previous_rollback_identity": rollback,
    }
    if training_input_evidence is not None:
        manifest["training_input_evidence"] = training_input_evidence
    if scheduled_refit_identity is not None:
        manifest["scheduled_refit_policy"] = scheduled_refit_identity
    if confirmed_strategy is not None and final_track_c_binding is not None:
        manifest["confirmed_strategy"] = confirmed_strategy
        manifest["performance_evidence"] = performance_evidence
        manifest["performance_confirmation"] = {
            "schema_version": 1,
            "policy_sha256": final_track_c_binding["policy_sha256"],
            "confirmation_result_sha256": final_track_c_binding[
                "confirmation_result_sha256"
            ],
            "final_track_c_pass_receipt_sha256": final_track_c_binding[
                "receipt_sha256"
            ],
            "evaluation_spec_name": final_track_c_binding[
                "evaluation_spec"
            ].name,
            "evaluation_spec_payload_sha256": _canonical_json_sha256(
                asdict(final_track_c_binding["evaluation_spec"])
            ),
            "fullfit_spec_name": final_track_c_binding["fullfit_spec"].name,
            "fullfit_spec_payload_sha256": _canonical_json_sha256(
                asdict(final_track_c_binding["fullfit_spec"])
            ),
            "strategy_config_sha256": final_track_c_binding[
                "strategy_config_sha256"
            ],
            "dataset_fights_sha256": final_track_c_binding[
                "dataset_fights_sha256"
            ],
            "features_sha256": final_track_c_binding["features_sha256"],
            "features_value_sha256": final_track_c_binding[
                "features_value_sha256"
            ],
            "feature_contract_sha256": final_track_c_binding[
                "feature_contract_sha256"
            ],
            "confirmation_evaluation_input_value_sha256": final_track_c_binding[
                "confirmation_evaluation_input_value_sha256"
            ],
            "evaluation_protocol_sha256": final_track_c_binding[
                "evaluation_protocol_sha256"
            ],
            "odds_source_inventory_sha256": final_track_c_binding[
                "odds_source_inventory_sha256"
            ],
            "source_fingerprint": final_track_c_binding["source_fingerprint"],
            "source_inventory_sha256": final_track_c_binding[
                "source_inventory_sha256"
            ],
            "source_inventory_artifact_sha256": final_track_c_binding[
                "source_inventory_artifact_sha256"
            ],
            "environment_artifact_sha256": final_track_c_binding[
                "environment_artifact_sha256"
            ],
            "environment_payload_sha256": final_track_c_binding[
                "environment_payload_sha256"
            ],
            "performance_evidence_aggregate_sha256": performance_evidence[
                "aggregate_sha256"
            ],
        }

    temp_root = Path(
        tempfile.mkdtemp(prefix=".bundle-build-", dir=staging_root.parent)
    ).resolve(strict=True)
    try:
        temp_models = temp_root / "models"
        temp_processed = temp_root / "processed"
        for label, source in model_paths.items():
            identity = model_identities[label]
            destination = temp_models / MODEL_FILENAMES[label]
            _copy_with_identity(source, destination, str(identity["sha256"]))
        _copy_with_identity(
            sidecar_path,
            temp_models / sidecar_path.name,
            _sha256_file(sidecar_path),
        )
        for filename, source in processed_paths.items():
            _copy_with_identity(
                source, temp_processed / filename, _sha256_file(source)
            )
        _copy_with_identity(
            pretraining_inventory_path,
            temp_root / "provenance" / "pretraining_model_input_inventory.json",
            _sha256_file(pretraining_inventory_path),
        )
        _copy_with_identity(
            assembly_inventory_path,
            temp_root / "provenance" / "assembly_model_input_inventory.json",
            _sha256_file(assembly_inventory_path),
        )
        if scheduled_policy is not None:
            _copy_with_identity(
                scheduled_policy.path,
                temp_root / SCHEDULED_POLICY_STAGED_PATH,
                scheduled_policy.sha256,
            )
        _copy_with_identity(
            audit_fights_path,
            temp_root / "provenance" / "independent_audit_snapshot" / "fights_cleaned.csv",
            _sha256_file(audit_fights_path),
        )
        _copy_with_identity(
            audit_features_path,
            temp_root / "provenance" / "independent_audit_snapshot" / "features.csv",
            _sha256_file(audit_features_path),
        )
        _copy_with_identity(
            bfo_path,
            temp_root / "provenance" / bfo_path.relative_to(root),
            _sha256_file(bfo_path),
        )
        if bfo_identity.get("provenance_mode") == "scheduled_recovery_batch":
            for record in bfo_identity["corrected_csv_files"]:
                source = root / str(record["path"])
                _copy_with_identity(
                    source,
                    temp_root / str(record["staged_path"]),
                    str(record["sha256"]),
                )
        if next_bfo_lineage is not None and bfo_lineage_identity is not None:
            lineage_root = temp_root / "provenance" / "bfo_lineage"
            written_manifest = bfo_lineage.write_manifest(
                next_bfo_lineage,
                lineage_root / bfo_lineage.MANIFEST_NAME,
            )
            if (
                _sha256_file(written_manifest)
                != bfo_lineage_identity["manifest_sha256"]
                or written_manifest.stat().st_size
                != bfo_lineage_identity["manifest_bytes"]
            ):
                raise StagingBundleError("Written BFO lineage manifest identity changed")
            for batch in next_bfo_lineage["batches"]:
                for record in (batch["csv"], batch["provenance"]):
                    _copy_with_identity(
                        root / str(record["raw_path"]),
                        lineage_root / str(record["artifact_path"]),
                        str(record["sha256"]),
                    )
        for row in evidence_rows:
            source = root / str(row["path"])
            _copy_with_identity(
                source,
                temp_root / "evidence" / str(row["path"]),
                str(row["sha256"]),
            )
        if performance_evidence is not None:
            for record in performance_evidence["files"]:
                _copy_with_identity(
                    root / str(record["source_path"]),
                    temp_root / str(record["staged_path"]),
                    str(record["sha256"]),
                )
        _copy_with_identity(
            previous_readyz,
            temp_root / "rollback" / "previous_readyz.json",
            _sha256_file(previous_readyz),
        )
        write_json_atomically(manifest, temp_root / "staging_manifest.json")

        if staging_root.exists():
            raise StagingBundleError(
                f"Staging root appeared during assembly; refusing overwrite: {staging_root}"
            )
        os.rename(temp_root, staging_root)
    except Exception:
        _remove_owned_temp(temp_root, expected_parent=staging_root.parent)
        raise

    try:
        rich_validation = validate_rich_staged_manifest(
            manifest_path, expected_payload=manifest
        )
        bundle = load_production_bundle(manifest_path)
        validation = validate_production_bundle(
            bundle,
            expected_models_dir=final_models,
            expected_processed_dir=final_processed,
        )
    except (ProductionBundleError, Exception) as exc:
        # The final directory is wholly owned by this failed invocation and did
        # not exist before it.  Removing it preserves the fail-closed promise.
        try:
            shutil.rmtree(staging_root)
        except OSError as cleanup_exc:
            raise StagingBundleError(
                f"Staged validation failed ({exc}) and cleanup failed ({cleanup_exc})"
            ) from exc
        raise StagingBundleError(f"Strict staged validation failed: {exc}") from exc

    return {
        "staging_root": str(staging_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "bundle_id": bundle_id,
        "model_spec_name": fullfit_name,
        "snapshot_max_event_date": snapshot_max_date,
        "validation": validation,
        "rich_validation": rich_validation,
        "finite_inference": inference_summary,
    }


def _parse_runtime_hash(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("runtime lookup hashes must use NAME=SHA256")
    key, digest = value.split("=", 1)
    if not RUNTIME_HASH_KEY_RE.fullmatch(key) or not SHA256_RE.fullmatch(digest):
        raise argparse.ArgumentTypeError(f"invalid runtime lookup hash: {value!r}")
    return key, digest.lower()


def _parse_training_argv(value: str) -> tuple[str, ...]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"training argv must be a JSON string array: {exc}"
        ) from exc
    if not isinstance(payload, list) or not payload or not all(
        isinstance(token, str) and token for token in payload
    ):
        raise argparse.ArgumentTypeError("training argv must be a nonempty JSON string array")
    return tuple(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--candidate-models-dir", type=Path, required=True)
    parser.add_argument("--candidate-processed-dir", type=Path, required=True)
    parser.add_argument("--evaluation-spec", required=True)
    parser.add_argument(
        "--input-inventory",
        type=Path,
        required=True,
        help="Frozen pretraining model/source/raw-input inventory.",
    )
    parser.add_argument("--assembly-inventory", type=Path, required=True)
    parser.add_argument("--bfo-provenance", type=Path, required=True)
    parser.add_argument(
        "--selection-evidence", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--previous-manifest",
        type=Path,
        help="Legacy one-off mode only; scheduled mode binds its parent from /readyz.",
    )
    parser.add_argument("--previous-readyz", type=Path, required=True)
    parser.add_argument(
        "--previous-bfo-lineage-manifest",
        type=Path,
        help=(
            "Scheduled mode only: validated manifest restored from the active "
            "production BFO lineage artifact."
        ),
    )
    parser.add_argument("--previous-deployed-git-sha")
    parser.add_argument("--expected-fights-sha256", required=True)
    parser.add_argument("--expected-features-sha256", required=True)
    parser.add_argument(
        "--previous-runtime-lookup-hash",
        action="append",
        type=_parse_runtime_hash,
        default=[],
        metavar="NAME=SHA256",
    )
    parser.add_argument(
        "--training-argv-json",
        type=_parse_training_argv,
        required=True,
        help="Exact training process argv as a JSON string array, including Python.",
    )
    parser.add_argument("--bundle-id")
    parser.add_argument("--inference-sample-rows", type=int, default=32)
    parser.add_argument(
        "--scheduled-refit-policy",
        type=Path,
        help=(
            "Enable strict scheduled-refit mode using this immutable policy JSON. "
            "The frozen /readyz response is then the sole parent identity evidence; "
            "legacy predecessor manifest/hash arguments are forbidden."
        ),
    )
    parser.add_argument(
        "--final-track-c-pass-receipt",
        type=Path,
        help=(
            "Required in scheduled policy-v2 mode; exact durable PASS for the "
            "final child policy and confirmed package."
        ),
    )
    args = parser.parse_args(argv)

    legacy_parent_args_present = bool(
        args.previous_manifest
        or args.previous_deployed_git_sha
        or args.previous_runtime_lookup_hash
    )
    if args.scheduled_refit_policy is not None and legacy_parent_args_present:
        parser.error(
            "--scheduled-refit-policy takes parent identity only from --previous-readyz; "
            "do not pass --previous-manifest, --previous-deployed-git-sha, or "
            "--previous-runtime-lookup-hash"
        )
    if args.scheduled_refit_policy is None and (
        args.previous_manifest is None
        or args.previous_deployed_git_sha is None
        or not args.previous_runtime_lookup_hash
    ):
        parser.error(
            "legacy one-off mode requires --previous-manifest, "
            "--previous-deployed-git-sha, and --previous-runtime-lookup-hash"
        )
    if (
        args.scheduled_refit_policy is None
        and args.previous_bfo_lineage_manifest is not None
    ):
        parser.error(
            "--previous-bfo-lineage-manifest is allowed only with "
            "--scheduled-refit-policy"
        )

    runtime_hashes: dict[str, str] = {}
    for key, digest in args.previous_runtime_lookup_hash:
        if key in runtime_hashes:
            parser.error(f"duplicate previous runtime lookup hash key: {key}")
        runtime_hashes[key] = digest

    inputs = BundleInputs(
        staging_root=args.staging_root,
        candidate_models_dir=args.candidate_models_dir,
        candidate_processed_dir=args.candidate_processed_dir,
        evaluation_spec_name=args.evaluation_spec,
        input_inventory_path=args.input_inventory,
        assembly_inventory_path=args.assembly_inventory,
        bfo_provenance_path=args.bfo_provenance,
        selection_evidence_paths=tuple(args.selection_evidence),
        previous_manifest_path=args.previous_manifest,
        previous_readyz_path=args.previous_readyz,
        previous_deployed_git_sha=args.previous_deployed_git_sha,
        previous_runtime_lookup_hashes=runtime_hashes,
        expected_fights_sha256=args.expected_fights_sha256,
        expected_features_sha256=args.expected_features_sha256,
        training_argv=args.training_argv_json,
        bundle_id=args.bundle_id,
        inference_sample_rows=args.inference_sample_rows,
        scheduled_refit_policy_path=args.scheduled_refit_policy,
        previous_bfo_lineage_manifest_path=args.previous_bfo_lineage_manifest,
        final_track_c_pass_receipt_path=args.final_track_c_pass_receipt,
    )
    try:
        result = assemble_staged_bundle(inputs)
    except StagingBundleError as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
