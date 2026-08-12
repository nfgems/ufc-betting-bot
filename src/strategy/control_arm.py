"""Frozen control arm management.

Provides utilities to freeze, validate, and load control arm artifacts
used as the baseline for selection gate and promotion gate evaluation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FROZEN_DIR = REPO_ROOT / "data" / "frozen"
INTEGRITY_V2_POLICY_PATH = REPO_ROOT / "config" / "scheduled_refit_policy_v2.json"
INTEGRITY_V2_APPROVED_PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
INTEGRITY_V2_APPROVED_ODDS_ROOT = REPO_ROOT / "data" / "raw"
INTEGRITY_V2_CORRECTED_ROOT = (
    INTEGRITY_V2_APPROVED_PROCESSED_ROOT / "audit_remediation_integrity_v2_final"
)
INTEGRITY_V2_CORRECTED_FIGHTS_PATH = INTEGRITY_V2_CORRECTED_ROOT / "fights_cleaned.csv"
INTEGRITY_V2_CORRECTED_FEATURES_PATH = INTEGRITY_V2_CORRECTED_ROOT / "features.csv"
INTEGRITY_V2_CORRECTED_FIGHTS_SHA256 = (
    "0c4d616474155ccc611bd88e77309c4e954cfc64b04a691d92473c529caae8d1"
)
REQUIRED_METRICS = ("accuracy", "brier", "log_loss", "ece")
INTEGRITY_V2_FREEZE_PREFIX = "control_arm_integrity_v2"
INTEGRITY_V2_RECEIPT_FILENAME = "fixed_control_bootstrap_receipt.json"
INTEGRITY_V2_EVALUATION_INDEX_FILENAME = "fixed_control_evaluation_index.csv"
INTEGRITY_V2_EVALUATION_INDEX_NA_REP = "<NA>"
INTEGRITY_V2_INPUT_PROVENANCE_FILENAME = "fixed_control_input_provenance.json"
INTEGRITY_V2_ODDS_SOURCE_INVENTORY_FILENAME = (
    "fixed_control_odds_source_inventory.json"
)
INTEGRITY_V2_SOURCE_INVENTORY_FILENAME = "evaluation_source_inventory.json"
INTEGRITY_V2_ENVIRONMENT_FILENAME = "evaluation_environment.json"
INTEGRITY_V2_PRODUCTION_RESULT_FILENAME = "production_result.json"
INTEGRITY_V2_PRODUCTION_BET_LOG_FILENAME = "production_bet_log.csv"
INTEGRITY_V2_PRODUCTION_BANKROLL_FILENAME = "production_bankroll_history.csv"
INTEGRITY_V2_PRODUCTION_ROW_FILENAME = "production_row.json"
INTEGRITY_V2_RECEIPT_ARTIFACT_FILENAMES = (
    INTEGRITY_V2_PRODUCTION_RESULT_FILENAME,
    INTEGRITY_V2_PRODUCTION_BET_LOG_FILENAME,
    INTEGRITY_V2_PRODUCTION_BANKROLL_FILENAME,
    INTEGRITY_V2_PRODUCTION_ROW_FILENAME,
    INTEGRITY_V2_EVALUATION_INDEX_FILENAME,
    INTEGRITY_V2_INPUT_PROVENANCE_FILENAME,
    INTEGRITY_V2_ODDS_SOURCE_INVENTORY_FILENAME,
    INTEGRITY_V2_SOURCE_INVENTORY_FILENAME,
    INTEGRITY_V2_ENVIRONMENT_FILENAME,
)
INTEGRITY_V2_FROZEN_ARTIFACT_FILENAMES = frozenset(
    {
        "control_metrics.json",
        "control_sweep_summary.json",
        "control_bet_log.csv",
        "control_bankroll_history.csv",
        INTEGRITY_V2_RECEIPT_FILENAME,
        *INTEGRITY_V2_RECEIPT_ARTIFACT_FILENAMES,
    }
)
_EVALUATION_INDEX_COLUMNS = (
    "event_date",
    "fighter_a",
    "fighter_b",
    "target",
    "fold",
    "train_end",
    "test_end",
)
_INTEGRITY_V2_PARTITION_BINDING_FIELDS = (
    "selection_fold_ids",
    "confirmation_fold_ids",
    "evaluation_sample_sha256",
    "n_predictions",
    "n_folds",
    "confirmation_evaluation_sample_sha256",
    "confirmation_evaluation_n_fights",
    "confirmation_evaluation_n_folds",
    "evaluation_input_value_sha256",
    "full_evaluation_input_value_sha256",
    "confirmation_evaluation_input_value_sha256",
    "evaluation_fight_identity_sha256",
    "full_evaluation_fight_identity_sha256",
    "confirmation_evaluation_fight_identity_sha256",
    "prediction_rows_sha256",
    "prediction_values_sha256",
    "evaluation_end_date",
    "fresh_window_cutoff",
)
_INTEGRITY_V2_DYNAMIC_PROVENANCE_FIELDS = (
    "input_provenance_path",
    "input_provenance_sha256",
    "input_provenance_payload_sha256",
    "odds_source_inventory_path",
    "odds_source_inventory_sha256",
    "source_inventory_path",
    "source_inventory_artifact_sha256",
    "source_inventory_sha256",
    "environment_path",
    "environment_artifact_sha256",
    "environment_payload_sha256",
    "source_dataset_fights_path",
    "source_dataset_fights_sha256",
    "source_features_path",
    "source_features_sha256",
    "dataset_fights_path",
    "dataset_fights_sha256",
    "features_artifact_path",
    "features_artifact_sha256",
    "features_value_sha256",
    "feature_contract_columns",
)
_INTEGRITY_V2_INPUT_VALUE_FIELDS = (
    "evaluation_input_value_sha256",
    "full_evaluation_input_value_sha256",
    "confirmation_evaluation_input_value_sha256",
)
_INTEGRITY_V2_STAGE1_MATERIALIZED_PATH_FIELDS = frozenset(
    {"input_provenance_path", "odds_source_inventory_path"}
)
_INTEGRITY_V2_REQUIRED_SOURCE_SCRIPTS = (
    "scripts/bfo_lineage.py",
    "scripts/check_production_refit_contract.py",
    "scripts/check_scheduled_refit_quality.py",
    "scripts/check_tracked_artifact_integrity.py",
    "scripts/recover_post_cutoff_method_odds.py",
    "scripts/track_c_batch.py",
)
_INTEGRITY_V2_REQUIRED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
)


def _sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_integrity_v2_freeze(freeze_id: str) -> bool:
    return _normalize_freeze_id(freeze_id).startswith(INTEGRITY_V2_FREEZE_PREFIX)


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_approved_path(
    value: object,
    *,
    approved_root: Path,
    label: str,
) -> tuple[Path | None, str | None]:
    if not isinstance(value, str) or not value:
        return None, f"{label} must be a canonical absolute path"
    candidate = Path(value)
    if not candidate.is_absolute():
        return None, f"{label} must be a canonical absolute path"
    try:
        resolved = candidate.resolve(strict=True)
        root = approved_root.resolve(strict=True)
    except OSError as exc:
        return None, f"{label} cannot be resolved: {exc}"
    if str(candidate) != str(resolved):
        return None, f"{label} is not canonical: {candidate}"
    if not resolved.is_relative_to(root):
        return None, f"{label} is outside approved root {root}"
    return resolved, None


def _load_policy_bound_integrity_v2_contract(
    *,
    require_final_validation: bool = True,
) -> tuple[
    dict[str, Any],
    object,
    dict[str, Any],
    dict[str, Any],
]:
    """Resolve the control contract from the final policy and live registry."""
    from scripts import check_production_refit_contract as contract_gate

    try:
        policy = contract_gate.load_policy(INTEGRITY_V2_POLICY_PATH)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot load integrity-v2 policy: {exc}") from exc
    registry_errors, registered_spec, _fullfit_spec = (
        contract_gate.validate_policy_registry(policy, repo_root=REPO_ROOT)
    )
    if registered_spec is None:
        raise RuntimeError(
            "Integrity-v2 policy/registry validation failed: "
            + "; ".join(registry_errors)
        )
    if require_final_validation and registry_errors:
        raise RuntimeError(
            "Integrity-v2 final policy is not promotion-ready: "
            + "; ".join(registry_errors)
        )
    try:
        contract = policy["contract"]
        evaluation = policy["evaluation"]
        baseline = policy["baseline"]
        spec_name = str(contract["evaluation_spec_name"])
        declared_spec_sha = str(contract["evaluation_spec_payload_sha256"])
        declared_feature_count = int(contract["feature_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Integrity-v2 policy contract is incomplete: {exc}") from exc

    registered_spec_sha = _canonical_json_sha256(asdict(registered_spec))
    feature_contract_columns = list(registered_spec.feature_cols)
    feature_contract_sha = _canonical_json_sha256(feature_contract_columns)
    if registered_spec_sha != declared_spec_sha:
        raise RuntimeError("Integrity-v2 policy spec hash differs from registry")
    if len(feature_contract_columns) != declared_feature_count:
        raise RuntimeError("Integrity-v2 policy feature count differs from registry")

    scheduled_protocol_sha = contract_gate.scheduled_protocol_sha256(policy)
    if (
        baseline.get("scheduled_protocol_sha256") != scheduled_protocol_sha
        or baseline.get("evidence_protocol_sha256") != scheduled_protocol_sha
    ):
        raise RuntimeError("Integrity-v2 policy scheduled protocol hash is inconsistent")

    if require_final_validation:
        # Parent-policy finality (DIR-AUD-P2-017): a control may only be frozen
        # against a policy whose bound baseline evidence CSV is byte-current and
        # embeds the protocol recomputed above (the checks above pin both policy
        # protocol fields to that recomputed hash, so the strict checker's
        # embedded-protocol comparison closes the chain). Reuses the quality
        # gate's checker so there is exactly one baseline-evidence validator.
        from scripts import check_scheduled_refit_quality as quality_gate

        baseline_evidence_errors = quality_gate._validate_v2_baseline_evidence(
            policy
        )
        if baseline_evidence_errors:
            raise RuntimeError(
                "Integrity-v2 policy baseline evidence is stale: "
                + "; ".join(baseline_evidence_errors)
            )

    protocol_kwargs = {
        "bet_start_date": evaluation.get("bet_start_date"),
        "execution_mode": evaluation.get("execution_mode"),
        "entry_offset_days": evaluation.get("entry_offset_days"),
        "entry_offset_for_features": evaluation.get("entry_offset_for_features"),
        "require_entry_odds": evaluation.get("require_entry_odds"),
        "allow_closing_odds": False,
        "reserved_confirmation_folds": 2,
        "bootstrap": 5000,
        "retrain_months": evaluation.get("retrain_months"),
    }
    if require_final_validation:
        # Use the same builder as the evaluator after module initialization so
        # additions to the load-bearing protocol fail closed here too.
        from src.strategy.run_evaluation import _evaluation_protocol_payload

        evaluation_protocol = _evaluation_protocol_payload(
            **protocol_kwargs,
            policy_evaluation=evaluation,
        )
    else:
        # ``run_evaluation`` imports this module, so bootstrap the current shape
        # without creating an import cycle. Every freeze/readiness check above
        # re-resolves it through the shared builder.
        evaluation_protocol = {
            **protocol_kwargs,
            "initial_train_years": int(evaluation.get("initial_train_years")),
            "minimum_fighter_fights": int(
                evaluation.get("minimum_fighter_fights")
            ),
            "minimum_train_rows": 100,
            "minimum_test_rows": 5,
            "initial_bankroll": float(evaluation.get("initial_bankroll")),
            "model_seed": evaluation.get("model_seed"),
            "odds_noise_seed": evaluation.get("odds_noise_seed"),
            "policy_min_train_test_fights": evaluation.get(
                "min_train_test_fights"
            ),
            "honest_odds_role": "entry",
            "historical_odds_selection": (
                "closest_verified_snapshot_at_or_before_offset"
            ),
            "allowed_prefight_sources": list(
                evaluation.get("quality_allowed_prefight_sources") or []
            ),
            "closing_odds_fallback": False,
            "agreement_model": evaluation.get("agreement_model"),
            "fresh_window_rule": (
                "six_calendar_months_before_selection_partition_max_event_date"
            ),
            "strategy": {
                field: evaluation.get(field)
                for field in (
                    "strategy_name",
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
                )
            },
            "execution_assumptions": evaluation.get("execution_assumptions"),
        }
    if (
        baseline.get("evaluation_spec_name") != spec_name
        or baseline.get("evaluation_sample_sha256") is None
    ):
        raise RuntimeError("Integrity-v2 policy baseline does not bind the evaluation spec")

    bindings = {
        "model_spec_name": spec_name,
        "model_spec_payload_sha256": registered_spec_sha,
        "feature_contract_count": len(feature_contract_columns),
        "feature_contract_sha256": feature_contract_sha,
        "source_dataset_fights_sha256": INTEGRITY_V2_CORRECTED_FIGHTS_SHA256,
        "source_features_sha256": baseline.get("features_sha256"),
        "policy_sha256": contract_gate.file_sha256(INTEGRITY_V2_POLICY_PATH),
        "scheduled_protocol_sha256": scheduled_protocol_sha,
        "evaluation_protocol_sha256": _canonical_json_sha256(evaluation_protocol),
        "evaluation_partition": "selection",
        "reserved_confirmation_folds": 2,
        "full_evaluation_sample_sha256": baseline.get(
            "evaluation_sample_sha256"
        ),
        "full_evaluation_n_fights": baseline.get("evaluation_n_fights"),
        "full_evaluation_n_folds": baseline.get("evaluation_n_folds"),
    }
    for field in (
        "model_spec_payload_sha256",
        "feature_contract_sha256",
        "source_dataset_fights_sha256",
        "source_features_sha256",
        "policy_sha256",
        "scheduled_protocol_sha256",
        "evaluation_protocol_sha256",
        "full_evaluation_sample_sha256",
    ):
        if not _valid_sha256(bindings.get(field)):
            raise RuntimeError(f"Integrity-v2 policy binding {field} is not SHA-256")
    return policy, registered_spec, evaluation_protocol, bindings


(
    _INTEGRITY_V2_POLICY,
    _INTEGRITY_V2_REGISTERED_SPEC,
    INTEGRITY_V2_EVALUATION_PROTOCOL,
    _INTEGRITY_V2_BINDINGS,
) = _load_policy_bound_integrity_v2_contract(require_final_validation=False)
INTEGRITY_V2_SPEC_NAME = str(_INTEGRITY_V2_BINDINGS["model_spec_name"])


def _canonical_evaluation_sample_sha256(frame: pd.DataFrame) -> str:
    missing = [column for column in _EVALUATION_INDEX_COLUMNS if column not in frame]
    if frame.empty or missing:
        raise ValueError(
            f"evaluation index cannot be bound; missing columns: {missing}"
        )
    canonical = frame[list(_EVALUATION_INDEX_COLUMNS)].copy()
    for column in ("event_date", "train_end", "test_end"):
        canonical[column] = pd.to_datetime(
            canonical[column], errors="raise"
        ).dt.strftime("%Y-%m-%d")
    canonical["target"] = pd.to_numeric(
        canonical["target"], errors="raise"
    ).astype(int)
    canonical["fold"] = pd.to_numeric(
        canonical["fold"], errors="raise"
    ).astype(int)
    if canonical.duplicated(["event_date", "fighter_a", "fighter_b"]).any():
        raise ValueError("evaluation index contains duplicate fight identities")
    canonical = canonical.sort_values(
        list(_EVALUATION_INDEX_COLUMNS), kind="stable"
    ).reset_index(drop=True)
    return hashlib.sha256(
        canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _validate_exact_integrity_v2_bindings(
    payload: object,
    *,
    label: str,
    expected_bindings: dict[str, Any],
) -> list[str]:
    if not isinstance(payload, dict):
        return [f"{label} must be a JSON object"]
    errors: list[str] = []
    for field, expected in expected_bindings.items():
        actual = payload.get(field)
        if actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
            errors.append(
                f"{label}.{field} must equal {expected!r}; got {actual!r}"
            )
    return errors


def _validate_integrity_v2_partition_bindings(
    *,
    metrics_payload: object,
    fixed_evidence: object,
    receipt_payload: object,
) -> list[str]:
    errors: list[str] = []
    payloads = (
        ("control_metrics", metrics_payload),
        ("fixed_control_evidence", fixed_evidence),
        ("fixed_control_bootstrap_receipt", receipt_payload),
    )
    for label, payload in payloads:
        if not isinstance(payload, dict):
            continue
        for field in _INTEGRITY_V2_PARTITION_BINDING_FIELDS:
            if field not in payload:
                errors.append(f"{label}.{field} is required")
        for field in (
            "evaluation_sample_sha256",
            "confirmation_evaluation_sample_sha256",
            *_INTEGRITY_V2_INPUT_VALUE_FIELDS,
            "evaluation_fight_identity_sha256",
            "full_evaluation_fight_identity_sha256",
            "confirmation_evaluation_fight_identity_sha256",
            "prediction_rows_sha256",
            "prediction_values_sha256",
        ):
            if not _valid_sha256(payload.get(field)):
                errors.append(f"{label}.{field} must be a SHA-256 digest")

    if not all(isinstance(payload, dict) for _, payload in payloads):
        return errors
    for field in _INTEGRITY_V2_PARTITION_BINDING_FIELDS:
        receipt_value = receipt_payload.get(field)
        for label, payload in payloads[:2]:
            if payload.get(field) != receipt_value:
                errors.append(f"{label}.{field} differs from receipt")

    selection_ids = receipt_payload.get("selection_fold_ids")
    confirmation_ids = receipt_payload.get("confirmation_fold_ids")
    if (
        not isinstance(selection_ids, list)
        or not selection_ids
        or any(isinstance(value, bool) or not isinstance(value, int) for value in selection_ids)
        or selection_ids != sorted(set(selection_ids))
    ):
        errors.append("selection_fold_ids must be a non-empty ordered integer list")
        selection_ids = []
    if (
        not isinstance(confirmation_ids, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in confirmation_ids
        )
        or confirmation_ids != sorted(set(confirmation_ids))
        or len(confirmation_ids) != 2
    ):
        errors.append("confirmation_fold_ids must contain the final two fold IDs")
        confirmation_ids = []
    all_ids = [*selection_ids, *confirmation_ids]
    if all_ids and (
        all_ids != sorted(all_ids)
        or len(all_ids) != receipt_payload.get("full_evaluation_n_folds")
    ):
        errors.append("selection/confirmation fold IDs do not form the full schedule")

    expected_counts = {
        "n_folds": len(selection_ids),
        "confirmation_evaluation_n_folds": len(confirmation_ids),
    }
    for field, expected in expected_counts.items():
        actual = receipt_payload.get(field)
        if isinstance(actual, bool) or actual != expected:
            errors.append(f"{field} must equal {expected}; got {actual!r}")
    selection_rows = receipt_payload.get("n_predictions")
    confirmation_rows = receipt_payload.get("confirmation_evaluation_n_fights")
    full_rows = receipt_payload.get("full_evaluation_n_fights")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (selection_rows, confirmation_rows, full_rows)
    ):
        errors.append("selection/confirmation/full evaluation counts must be positive")
    elif selection_rows + confirmation_rows != full_rows:
        errors.append("selection and confirmation row counts do not sum to full sample")
    try:
        evaluation_end = pd.Timestamp(receipt_payload.get("evaluation_end_date"))
        fresh_cutoff = pd.Timestamp(receipt_payload.get("fresh_window_cutoff"))
    except (TypeError, ValueError):
        errors.append("evaluation_end_date/fresh_window_cutoff must be ISO dates")
    else:
        if (
            str(evaluation_end.date()) != receipt_payload.get("evaluation_end_date")
            or str(fresh_cutoff.date()) != receipt_payload.get("fresh_window_cutoff")
        ):
            errors.append("evaluation_end_date/fresh_window_cutoff must be canonical dates")
    return errors


def _validate_integrity_v2_run_binding(
    *,
    metrics_payload: object,
    fixed_evidence: object,
    receipt_payload: object,
    expected_protocol: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    payloads = (
        ("control_metrics", metrics_payload),
        ("fixed_control_evidence", fixed_evidence),
        ("fixed_control_bootstrap_receipt", receipt_payload),
    )
    for label, payload in payloads:
        if not isinstance(payload, dict):
            continue
        source_fingerprint = payload.get("source_fingerprint")
        if not _valid_sha256(source_fingerprint):
            errors.append(f"{label}.source_fingerprint must be a SHA-256 digest")
        protocol = payload.get("evaluation_protocol")
        if not isinstance(protocol, dict):
            errors.append(f"{label}.evaluation_protocol must be a JSON object")
            continue
        missing = sorted(set(expected_protocol).difference(protocol))
        unexpected = sorted(set(protocol).difference(expected_protocol))
        if missing:
            errors.append(f"{label}.evaluation_protocol is missing fields: {missing}")
        if unexpected:
            errors.append(
                f"{label}.evaluation_protocol has unexpected fields: {unexpected}"
            )
        for field, expected in expected_protocol.items():
            actual = protocol.get(field)
            if actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
                errors.append(
                    f"{label}.evaluation_protocol.{field} must equal "
                    f"{expected!r}; got {actual!r}"
                )

    if all(isinstance(payload, dict) for _, payload in payloads):
        receipt_fingerprint = receipt_payload.get("source_fingerprint")
        receipt_protocol = receipt_payload.get("evaluation_protocol")
        for label, payload in payloads[:2]:
            if payload.get("source_fingerprint") != receipt_fingerprint:
                errors.append(f"{label}.source_fingerprint differs from receipt")
            if payload.get("evaluation_protocol") != receipt_protocol:
                errors.append(f"{label}.evaluation_protocol differs from receipt")
    return errors


def _validate_integrity_v2_input_provenance(
    *,
    metrics_payload: object,
    fixed_evidence: object,
    receipt_payload: object,
    input_provenance_path: Path,
    odds_inventory_path: Path,
    source_inventory_path: Path,
    environment_path: Path,
    verify_external_sources: bool,
    expected_protocol: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    payloads = (
        ("control_metrics", metrics_payload),
        ("fixed_control_evidence", fixed_evidence),
        ("fixed_control_bootstrap_receipt", receipt_payload),
    )
    for label, payload in payloads:
        if not isinstance(payload, dict):
            continue
        for field in _INTEGRITY_V2_DYNAMIC_PROVENANCE_FIELDS:
            if field not in payload:
                errors.append(f"{label}.{field} is required")
        for field in (
            "input_provenance_sha256",
            "odds_source_inventory_sha256",
            "source_inventory_artifact_sha256",
            "source_inventory_sha256",
            "environment_artifact_sha256",
            "environment_payload_sha256",
            "source_dataset_fights_sha256",
            "source_features_sha256",
            "dataset_fights_sha256",
            "features_artifact_sha256",
            "features_value_sha256",
        ):
            if not _valid_sha256(payload.get(field)):
                errors.append(f"{label}.{field} must be a SHA-256 digest")
        for field, expected_name in (
            ("input_provenance_path", INTEGRITY_V2_INPUT_PROVENANCE_FILENAME),
            (
                "odds_source_inventory_path",
                INTEGRITY_V2_ODDS_SOURCE_INVENTORY_FILENAME,
            ),
            ("source_inventory_path", INTEGRITY_V2_SOURCE_INVENTORY_FILENAME),
            ("environment_path", INTEGRITY_V2_ENVIRONMENT_FILENAME),
        ):
            path_value = payload.get(field)
            if not isinstance(path_value, str) or Path(path_value).name != expected_name:
                errors.append(f"{label}.{field} must name {expected_name}")
        columns = payload.get("feature_contract_columns")
        if (
            not isinstance(columns, list)
            or not columns
            or any(not isinstance(column, str) or not column for column in columns)
            or len(set(columns)) != len(columns)
        ):
            errors.append(
                f"{label}.feature_contract_columns must be an ordered unique string list"
            )
        elif (
            len(columns) != payload.get("feature_contract_count")
            or _canonical_json_sha256(columns)
            != payload.get("feature_contract_sha256")
        ):
            errors.append(
                f"{label}.feature_contract_columns does not match its count/hash"
            )

    if all(isinstance(payload, dict) for _, payload in payloads):
        for field in _INTEGRITY_V2_DYNAMIC_PROVENANCE_FIELDS:
            receipt_value = receipt_payload.get(field)
            for label, payload in payloads[:2]:
                if payload.get(field) != receipt_value:
                    errors.append(f"{label}.{field} differs from receipt")

    input_provenance_sha = (
        _sha256(input_provenance_path) if input_provenance_path.is_file() else None
    )
    odds_inventory_sha = (
        _sha256(odds_inventory_path) if odds_inventory_path.is_file() else None
    )
    source_inventory_artifact_sha = (
        _sha256(source_inventory_path) if source_inventory_path.is_file() else None
    )
    environment_artifact_sha = (
        _sha256(environment_path) if environment_path.is_file() else None
    )
    for label, payload in payloads:
        if not isinstance(payload, dict):
            continue
        if payload.get("input_provenance_sha256") != input_provenance_sha:
            errors.append(f"{label}.input_provenance_sha256 does not match artifact")
        if payload.get("odds_source_inventory_sha256") != odds_inventory_sha:
            errors.append(
                f"{label}.odds_source_inventory_sha256 does not match artifact"
            )
        if payload.get("source_inventory_artifact_sha256") != (
            source_inventory_artifact_sha
        ):
            errors.append(
                f"{label}.source_inventory_artifact_sha256 does not match artifact"
            )
        if payload.get("environment_artifact_sha256") != environment_artifact_sha:
            errors.append(
                f"{label}.environment_artifact_sha256 does not match artifact"
            )

    try:
        input_provenance = json.loads(
            input_provenance_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Invalid {INTEGRITY_V2_INPUT_PROVENANCE_FILENAME}: {exc}")
        input_provenance = None
    try:
        odds_inventory = json.loads(odds_inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Invalid {INTEGRITY_V2_ODDS_SOURCE_INVENTORY_FILENAME}: {exc}")
        odds_inventory = None
    try:
        source_inventory = json.loads(
            source_inventory_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Invalid {INTEGRITY_V2_SOURCE_INVENTORY_FILENAME}: {exc}")
        source_inventory = None
    try:
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Invalid {INTEGRITY_V2_ENVIRONMENT_FILENAME}: {exc}")
        environment = None

    receipt = receipt_payload if isinstance(receipt_payload, dict) else {}
    if not isinstance(input_provenance, dict):
        if input_provenance is not None:
            errors.append("fixed-control input provenance must be a JSON object")
    else:
        if input_provenance.get("schema_version") != 1:
            errors.append("fixed-control input provenance schema_version must equal 1")
        provenance_binding_fields = (
            "model_spec_name",
            "model_spec_payload_sha256",
            "source_dataset_fights_path",
            "source_dataset_fights_sha256",
            "source_features_path",
            "source_features_sha256",
            "dataset_fights_path",
            "dataset_fights_sha256",
            "features_artifact_path",
            "features_artifact_sha256",
            "features_value_sha256",
            "feature_contract_columns",
            "feature_contract_count",
            "feature_contract_sha256",
            "policy_sha256",
            "scheduled_protocol_sha256",
            "evaluation_protocol_sha256",
            "source_fingerprint",
            "evaluation_protocol",
            "evaluation_partition",
            "reserved_confirmation_folds",
            "selection_fold_ids",
            "confirmation_fold_ids",
            "evaluation_sample_sha256",
            "n_predictions",
            "n_folds",
            "full_evaluation_sample_sha256",
            "full_evaluation_n_fights",
            "full_evaluation_n_folds",
            "confirmation_evaluation_sample_sha256",
            "confirmation_evaluation_n_fights",
            "confirmation_evaluation_n_folds",
            "odds_source_inventory_sha256",
            "source_inventory_sha256",
            "source_inventory_path",
            "source_inventory_artifact_sha256",
            "environment_path",
            "environment_artifact_sha256",
            "environment_payload_sha256",
        )
        for field in provenance_binding_fields:
            if input_provenance.get(field) != receipt.get(field):
                errors.append(
                    f"fixed-control input provenance {field} differs from receipt"
                )
        for field in _INTEGRITY_V2_INPUT_VALUE_FIELDS:
            if not _valid_sha256(input_provenance.get(field)):
                errors.append(
                    f"fixed-control input provenance {field} must be a SHA-256 digest"
                )
            elif input_provenance.get(field) != receipt.get(field):
                errors.append(
                    f"fixed-control input provenance {field} differs from receipt"
                )
        unsigned_provenance = dict(input_provenance)
        supplied_payload_sha = unsigned_provenance.pop(
            "input_provenance_payload_sha256",
            None,
        )
        if supplied_payload_sha != _canonical_json_sha256(unsigned_provenance):
            errors.append(
                "fixed-control input provenance payload SHA-256 does not reproduce"
            )
        if supplied_payload_sha != receipt.get("input_provenance_payload_sha256"):
            errors.append(
                "fixed-control input provenance payload SHA-256 differs from receipt"
            )
        if verify_external_sources:
            expected_corrected_paths = {
                "source_dataset_fights_path": INTEGRITY_V2_CORRECTED_FIGHTS_PATH,
                "dataset_fights_path": INTEGRITY_V2_CORRECTED_FIGHTS_PATH,
                "source_features_path": INTEGRITY_V2_CORRECTED_FEATURES_PATH,
                "features_artifact_path": INTEGRITY_V2_CORRECTED_FEATURES_PATH,
            }
            resolved_input_paths: dict[str, Path] = {}
            for path_field, sha_field in (
                ("source_dataset_fights_path", "source_dataset_fights_sha256"),
                ("dataset_fights_path", "dataset_fights_sha256"),
                ("source_features_path", "source_features_sha256"),
                ("features_artifact_path", "features_artifact_sha256"),
            ):
                source_path, path_error = _canonical_approved_path(
                    input_provenance.get(path_field),
                    approved_root=INTEGRITY_V2_APPROVED_PROCESSED_ROOT,
                    label=f"fixed-control input provenance {path_field}",
                )
                if path_error:
                    errors.append(path_error)
                    continue
                resolved_input_paths[path_field] = source_path
                if source_path != expected_corrected_paths[path_field].resolve(
                    strict=True
                ):
                    errors.append(
                        "fixed-control input provenance does not use the final "
                        f"corrected source: {path_field}"
                    )
                if _sha256(source_path) != receipt.get(sha_field):
                    errors.append(
                        "fixed-control input provenance source hash mismatch: "
                        f"{path_field}"
                    )
            if (
                receipt.get("dataset_fights_sha256")
                != receipt.get("source_dataset_fights_sha256")
            ):
                errors.append(
                    "evaluated fights artifact differs from the final corrected source"
                )
            if (
                receipt.get("features_artifact_sha256")
                != receipt.get("source_features_sha256")
            ):
                errors.append(
                    "evaluated features artifact differs from the final corrected source"
                )
            features_path = resolved_input_paths.get("features_artifact_path")
            if features_path is not None:
                try:
                    from src.strategy.run_evaluation import (
                        _canonical_frame_value_sha256,
                    )

                    features_frame = pd.read_csv(features_path)
                    actual_features_value_sha = _canonical_frame_value_sha256(
                        features_frame
                    )
                except Exception as exc:
                    errors.append(
                        "fixed-control features value hash cannot be recomputed: "
                        f"{exc}"
                    )
                else:
                    if actual_features_value_sha != receipt.get(
                        "features_value_sha256"
                    ):
                        errors.append(
                            "fixed-control input provenance features value hash mismatch"
                        )
            raw_policy_path = input_provenance.get("policy_path")
            if not isinstance(raw_policy_path, str) or Path(
                raw_policy_path
            ).resolve(strict=False) != INTEGRITY_V2_POLICY_PATH.resolve(strict=True):
                errors.append(
                    "fixed-control input provenance policy_path is not the final policy"
                )
            elif str(Path(raw_policy_path)) != str(
                INTEGRITY_V2_POLICY_PATH.resolve(strict=True)
            ):
                errors.append(
                    "fixed-control input provenance policy_path is not canonical"
                )
            elif _sha256(Path(raw_policy_path)) != receipt.get("policy_sha256"):
                errors.append(
                    "fixed-control input provenance source hash mismatch: policy_path"
                )

    expected_source_contract = {
        "python_glob": "src/**/*.py",
        "required_scripts": list(_INTEGRITY_V2_REQUIRED_SOURCE_SCRIPTS),
    }
    if not isinstance(source_inventory, dict):
        if source_inventory is not None:
            errors.append("evaluation source inventory must be a JSON object")
    else:
        if set(source_inventory) != {
            "schema_version",
            "source_contract",
            "sources",
            "environment_payload_sha256",
        }:
            errors.append("evaluation source inventory has unexpected fields")
        if source_inventory.get("schema_version") != 2:
            errors.append("evaluation source inventory schema_version must equal 2")
        if source_inventory.get("source_contract") != expected_source_contract:
            errors.append("evaluation source inventory source contract is not exact")
        if source_inventory.get("environment_payload_sha256") != receipt.get(
            "environment_payload_sha256"
        ):
            errors.append(
                "evaluation source inventory environment hash differs from receipt"
            )
        sources = source_inventory.get("sources")
        if not isinstance(sources, list) or not sources:
            errors.append("evaluation source inventory sources must be non-empty")
        else:
            malformed_sources = [
                index
                for index, entry in enumerate(sources)
                if not isinstance(entry, dict)
                or set(entry) != {"path", "sha256"}
                or not isinstance(entry.get("path"), str)
                or not _valid_sha256(entry.get("sha256"))
            ]
            source_paths = [
                entry.get("path") for entry in sources if isinstance(entry, dict)
            ]
            if malformed_sources:
                errors.append(
                    "evaluation source inventory has malformed entries: "
                    f"{malformed_sources}"
                )
            if len(source_paths) != len(set(source_paths)):
                errors.append("evaluation source inventory contains duplicate paths")
            inventory_fingerprint = _canonical_json_sha256(source_inventory)
            if inventory_fingerprint != receipt.get("source_inventory_sha256"):
                errors.append(
                    "evaluation source inventory fingerprint differs from receipt"
                )
            if inventory_fingerprint != receipt.get("source_fingerprint"):
                errors.append(
                    "evaluation source inventory fingerprint differs from source_fingerprint"
                )
            if verify_external_sources and not malformed_sources:
                expected_paths = list((REPO_ROOT / "src").rglob("*.py"))
                expected_paths.extend(
                    REPO_ROOT / relative
                    for relative in _INTEGRITY_V2_REQUIRED_SOURCE_SCRIPTS
                )
                current_sources = [
                    {
                        "path": path.relative_to(REPO_ROOT).as_posix(),
                        "sha256": _sha256(path),
                    }
                    for path in sorted(
                        set(expected_paths), key=lambda value: value.as_posix()
                    )
                    if path.is_file()
                ]
                if sources != current_sources:
                    observed_by_path = {
                        entry["path"]: entry["sha256"] for entry in sources
                    }
                    current_by_path = {
                        entry["path"]: entry["sha256"] for entry in current_sources
                    }
                    missing_paths = sorted(
                        set(current_by_path).difference(observed_by_path)
                    )
                    extra_paths = sorted(
                        set(observed_by_path).difference(current_by_path)
                    )
                    changed_paths = sorted(
                        path
                        for path in set(observed_by_path).intersection(current_by_path)
                        if observed_by_path[path] != current_by_path[path]
                    )
                    errors.append(
                        "evaluation source inventory differs from current load-bearing "
                        f"sources (missing={missing_paths}, extra={extra_paths}, "
                        f"changed={changed_paths})"
                    )

    if not isinstance(environment, dict):
        if environment is not None:
            errors.append("evaluation environment must be a JSON object")
    else:
        if set(environment) != {
            "schema_version",
            "python",
            "packages",
            "dependency_files",
        }:
            errors.append("evaluation environment has unexpected fields")
        if environment.get("schema_version") != 1:
            errors.append("evaluation environment schema_version must equal 1")
        python_info = environment.get("python")
        expected_python = {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        }
        if not isinstance(python_info, dict) or set(python_info) != {
            "implementation",
            "version",
        }:
            errors.append("evaluation environment python identity is malformed")
        elif verify_external_sources and python_info != expected_python:
            errors.append("evaluation environment Python identity changed")
        packages = environment.get("packages")
        if not isinstance(packages, dict) or set(packages) != set(
            _INTEGRITY_V2_REQUIRED_PACKAGES
        ):
            errors.append("evaluation environment package inventory is not exact")
        elif verify_external_sources:
            current_packages = {
                package: importlib.metadata.version(package)
                for package in _INTEGRITY_V2_REQUIRED_PACKAGES
            }
            if packages != current_packages:
                errors.append("evaluation environment package versions changed")
        dependencies = environment.get("dependency_files")
        expected_dependency_paths = ["requirements.txt"]
        if (
            not isinstance(dependencies, list)
            or len(dependencies) != 1
            or not isinstance(dependencies[0], dict)
            or set(dependencies[0]) != {"path", "sha256"}
            or dependencies[0].get("path") != "requirements.txt"
            or not _valid_sha256(dependencies[0].get("sha256"))
        ):
            errors.append("evaluation environment dependency inventory is not exact")
        elif verify_external_sources:
            dependency_path = REPO_ROOT / expected_dependency_paths[0]
            if (
                not dependency_path.is_file()
                or _sha256(dependency_path) != dependencies[0]["sha256"]
            ):
                errors.append("evaluation environment dependency bytes changed")
        environment_payload_sha = _canonical_json_sha256(environment)
        if environment_payload_sha != receipt.get("environment_payload_sha256"):
            errors.append("evaluation environment payload hash differs from receipt")

    if not isinstance(odds_inventory, dict):
        if odds_inventory is not None:
            errors.append("fixed-control odds source inventory must be a JSON object")
    else:
        if set(odds_inventory) != {"schema_version", "evaluation_protocol", "entries"}:
            errors.append("fixed-control odds source inventory has unexpected fields")
        if odds_inventory.get("schema_version") != 1:
            errors.append("fixed-control odds source inventory schema_version must equal 1")
        if odds_inventory.get("evaluation_protocol") != expected_protocol:
            errors.append(
                "fixed-control odds source inventory does not bind the exact T-1 protocol"
            )
        if isinstance(input_provenance, dict):
            if _canonical_json_sha256(odds_inventory) != input_provenance.get(
                "odds_source_inventory_payload_sha256"
            ):
                errors.append(
                    "fixed-control odds source inventory semantic hash does not reproduce"
                )
        entries = odds_inventory.get("entries")
        if not isinstance(entries, list) or not entries:
            errors.append("fixed-control odds source inventory entries must be non-empty")
        else:
            resolved_paths: list[str] = []
            for index, entry in enumerate(entries):
                if not isinstance(entry, dict) or set(entry) != {
                    "source_file",
                    "resolved_path",
                    "sha256",
                }:
                    errors.append(
                        f"fixed-control odds source inventory entry {index} is malformed"
                    )
                    continue
                source_file = entry.get("source_file")
                resolved_path = entry.get("resolved_path")
                entry_sha = entry.get("sha256")
                if not isinstance(source_file, str) or not source_file:
                    errors.append(
                        f"fixed-control odds source inventory entry {index} has no source_file"
                    )
                if not isinstance(resolved_path, str) or not resolved_path:
                    errors.append(
                        f"fixed-control odds source inventory entry {index} has no resolved_path"
                    )
                    continue
                if resolved_path in resolved_paths:
                    errors.append(
                        f"fixed-control odds source inventory duplicates {resolved_path}"
                    )
                resolved_paths.append(resolved_path)
                if not _valid_sha256(entry_sha):
                    errors.append(
                        f"fixed-control odds source inventory entry {index} has invalid SHA256"
                    )
                    continue
                if verify_external_sources:
                    source_path, path_error = _canonical_approved_path(
                        resolved_path,
                        approved_root=INTEGRITY_V2_APPROVED_ODDS_ROOT,
                        label=(
                            "fixed-control odds source inventory "
                            f"entry {index} resolved_path"
                        ),
                    )
                    if path_error:
                        errors.append(path_error)
                        continue
                    if source_file != source_path.name:
                        errors.append(
                            "fixed-control odds source inventory entry "
                            f"{index} source_file differs from resolved_path"
                        )
                    if _sha256(source_path) != entry_sha:
                        errors.append(
                            f"fixed-control odds source inventory entry {index} source mismatch"
                        )
            if verify_external_sources:
                from src.data.historical_backfill import BACKFILL_DIR

                expected_paths = sorted(
                    {
                        path.resolve()
                        for path in (
                            BACKFILL_DIR / "historical_odds_pre2022_from_cleaned.csv",
                            BACKFILL_DIR / "historical_odds_bfo.csv",
                            *sorted(
                                BACKFILL_DIR.glob(
                                    "historical_odds_bfo_recovered_*.csv"
                                )
                            ),
                            BACKFILL_DIR / "historical_odds.csv",
                            *sorted(
                                (BACKFILL_DIR.parent / "line_history").glob(
                                    "odds_*.csv"
                                )
                            ),
                        )
                        if path.is_file()
                    },
                    key=lambda path: str(path).lower(),
                )
                observed_paths = [Path(path).resolve(strict=False) for path in resolved_paths]
                if observed_paths != expected_paths:
                    errors.append(
                        "fixed-control odds source inventory is incomplete or contains "
                        "unlisted inputs"
                    )
    return errors


def _validate_integrity_v2_index(
    path: Path,
    *,
    expected_bindings: object,
) -> list[str]:
    if not path.is_file():
        return [f"Missing {INTEGRITY_V2_EVALUATION_INDEX_FILENAME}"]
    if not isinstance(expected_bindings, dict):
        return ["Integrity-v2 evaluation index bindings must be a JSON object"]
    try:
        frame = pd.read_csv(
            path,
            float_precision="round_trip",
            keep_default_na=False,
            na_values=[INTEGRITY_V2_EVALUATION_INDEX_NA_REP],
        )
    except Exception as exc:
        return [f"Invalid integrity-v2 evaluation index: {exc}"]

    errors: list[str] = []
    required = set(_EVALUATION_INDEX_COLUMNS) | {"evaluation_partition"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        return [f"Integrity-v2 evaluation index is missing columns: {missing}"]

    try:
        folds = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    except (TypeError, ValueError) as exc:
        return [f"Integrity-v2 evaluation index has invalid folds: {exc}"]
    frame = frame.copy()
    frame["fold"] = folds
    for date_column in ("event_date", "train_end", "test_end"):
        frame[date_column] = pd.to_datetime(
            frame[date_column],
            errors="raise",
            utc=True,
        )
    observed_evaluation_end = str(frame["event_date"].max().date())
    if observed_evaluation_end != expected_bindings.get("evaluation_end_date"):
        errors.append(
            "Integrity-v2 evaluation index end date differs from the bound "
            f"evaluation_end_date: {observed_evaluation_end}"
        )
    selection_fold_ids = list(expected_bindings.get("selection_fold_ids") or [])
    confirmation_fold_ids = list(
        expected_bindings.get("confirmation_fold_ids") or []
    )
    expected_fold_ids = selection_fold_ids + confirmation_fold_ids
    expected_partition = frame["fold"].map(
        lambda fold: (
            "selection"
            if fold in selection_fold_ids
            else "confirmation"
            if fold in confirmation_fold_ids
            else "invalid"
        )
    )
    if not frame["evaluation_partition"].astype(str).eq(expected_partition).all():
        errors.append(
            "Integrity-v2 evaluation index partition labels do not match the "
            "bound selection/confirmation fold IDs"
        )

    observed_fold_ids = sorted(frame["fold"].unique().tolist())
    if observed_fold_ids != sorted(expected_fold_ids):
        errors.append(
            f"Integrity-v2 evaluation index fold IDs are {observed_fold_ids!r}, "
            f"expected {sorted(expected_fold_ids)!r}"
        )

    try:
        partitions = {
            "full": (
                frame,
                int(expected_bindings["full_evaluation_n_fights"]),
                str(expected_bindings["full_evaluation_sample_sha256"]),
            ),
            "selection": (
                frame[frame["fold"].isin(selection_fold_ids)],
                int(expected_bindings["n_predictions"]),
                str(expected_bindings["evaluation_sample_sha256"]),
            ),
            "confirmation": (
                frame[frame["fold"].isin(confirmation_fold_ids)],
                int(expected_bindings["confirmation_evaluation_n_fights"]),
                str(expected_bindings["confirmation_evaluation_sample_sha256"]),
            ),
        }
    except (KeyError, TypeError, ValueError) as exc:
        return [*errors, f"Integrity-v2 evaluation index bindings are invalid: {exc}"]
    selection_partition = partitions["selection"][0]
    if selection_partition.empty:
        errors.append("Integrity-v2 selection index cannot anchor the fresh window")
    else:
        observed_fresh_cutoff = (
            selection_partition["event_date"].max().normalize()
            - pd.DateOffset(months=6)
        )
        expected_fresh_cutoff = expected_bindings.get("fresh_window_cutoff")
        if str(observed_fresh_cutoff.date()) != expected_fresh_cutoff:
            errors.append(
                "Integrity-v2 fresh-window cutoff differs from six calendar months "
                "before the selection partition end: "
                f"expected {expected_fresh_cutoff!r}, "
                f"got {str(observed_fresh_cutoff.date())!r}"
            )
    for label, (partition, expected_rows, expected_hash) in partitions.items():
        if len(partition) != expected_rows:
            errors.append(
                f"Integrity-v2 {label} index has {len(partition)} rows, "
                f"expected {expected_rows}"
            )
            continue
        try:
            actual_hash = _canonical_evaluation_sample_sha256(partition)
        except (TypeError, ValueError) as exc:
            errors.append(f"Integrity-v2 {label} index is invalid: {exc}")
            continue
        if actual_hash != expected_hash:
            errors.append(
                f"Integrity-v2 {label} sample hash mismatch: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    feature_contract_columns = expected_bindings.get("feature_contract_columns")
    if not isinstance(feature_contract_columns, list) or not feature_contract_columns:
        errors.append("Integrity-v2 evaluation index has no bound feature contract")
    else:
        try:
            from src.strategy.run_evaluation import _manifest_input_value_sha256

            value_hashes = {
                "evaluation_input_value_sha256": _manifest_input_value_sha256(
                    partitions["selection"][0],
                    feature_contract_columns=feature_contract_columns,
                ),
                "full_evaluation_input_value_sha256": _manifest_input_value_sha256(
                    partitions["full"][0],
                    feature_contract_columns=feature_contract_columns,
                ),
                "confirmation_evaluation_input_value_sha256": (
                    _manifest_input_value_sha256(
                        partitions["confirmation"][0],
                        feature_contract_columns=feature_contract_columns,
                    )
                ),
            }
        except Exception as exc:
            errors.append(
                "Integrity-v2 evaluation input value hashes cannot be recomputed: "
                f"{exc}"
            )
        else:
            for field, actual_hash in value_hashes.items():
                expected_hash = expected_bindings.get(field)
                if actual_hash != expected_hash:
                    errors.append(
                        f"Integrity-v2 {field} mismatch: expected "
                        f"{expected_hash}, got {actual_hash}"
                    )
    return errors


def _validate_integrity_v2_evidence_set(
    *,
    metrics_payload: object,
    sweep_payload: object,
    receipt_payload: object,
    index_path: Path,
    artifact_sources: dict[str, Path] | None = None,
    verify_external_sources: bool = False,
) -> list[str]:
    artifact_sources = artifact_sources or {}
    try:
        (
            current_policy,
            current_registered_spec,
            current_evaluation_protocol,
            current_bindings,
        ) = _load_policy_bound_integrity_v2_contract()
    except RuntimeError as exc:
        return [str(exc)]
    errors = _validate_exact_integrity_v2_bindings(
        metrics_payload,
        label="control_metrics",
        expected_bindings=current_bindings,
    )
    fixed_evidence = (
        sweep_payload.get("fixed_control_evidence")
        if isinstance(sweep_payload, dict)
        else None
    )
    errors.extend(
        _validate_exact_integrity_v2_bindings(
            fixed_evidence,
            label="fixed_control_evidence",
            expected_bindings=current_bindings,
        )
    )
    errors.extend(
        _validate_exact_integrity_v2_bindings(
            receipt_payload,
            label="fixed_control_bootstrap_receipt",
            expected_bindings=current_bindings,
        )
    )
    errors.extend(
        _validate_integrity_v2_partition_bindings(
            metrics_payload=metrics_payload,
            fixed_evidence=fixed_evidence,
            receipt_payload=receipt_payload,
        )
    )
    errors.extend(
        _validate_integrity_v2_run_binding(
            metrics_payload=metrics_payload,
            fixed_evidence=fixed_evidence,
            receipt_payload=receipt_payload,
            expected_protocol=current_evaluation_protocol,
        )
    )
    errors.extend(
        _validate_integrity_v2_input_provenance(
            metrics_payload=metrics_payload,
            fixed_evidence=fixed_evidence,
            receipt_payload=receipt_payload,
            input_provenance_path=artifact_sources.get(
                INTEGRITY_V2_INPUT_PROVENANCE_FILENAME,
                Path(),
            ),
            odds_inventory_path=artifact_sources.get(
                INTEGRITY_V2_ODDS_SOURCE_INVENTORY_FILENAME,
                Path(),
            ),
            source_inventory_path=artifact_sources.get(
                INTEGRITY_V2_SOURCE_INVENTORY_FILENAME,
                Path(),
            ),
            environment_path=artifact_sources.get(
                INTEGRITY_V2_ENVIRONMENT_FILENAME,
                Path(),
            ),
            verify_external_sources=verify_external_sources,
            expected_protocol=current_evaluation_protocol,
        )
    )

    if isinstance(metrics_payload, dict):
        try:
            from src.model.training_spec import resolve_named_training_spec

            registered_spec = resolve_named_training_spec(
                str(metrics_payload.get("model_spec_name"))
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"control_metrics model spec cannot be resolved: {exc}")
        else:
            registered_spec_sha = _canonical_json_sha256(asdict(registered_spec))
            if registered_spec_sha != metrics_payload.get(
                "model_spec_payload_sha256"
            ):
                errors.append(
                    "control_metrics model spec payload hash does not match registry"
                )
            registered_feature_cols = list(registered_spec.feature_cols)
            if registered_feature_cols != metrics_payload.get(
                "feature_contract_columns"
            ):
                errors.append(
                    "control_metrics ordered feature contract differs from registry"
                )
            if len(registered_feature_cols) != metrics_payload.get(
                "feature_contract_count"
            ):
                errors.append(
                    "control_metrics feature contract count does not match registry"
                )
            if _canonical_json_sha256(registered_feature_cols) != metrics_payload.get(
                "feature_contract_sha256"
            ):
                errors.append(
                    "control_metrics feature contract hash does not match registry"
                )

    expected_metrics = {
        "model_variant": "baseline",
        "dataset_variant": current_registered_spec.dataset_variant,
        "feature_family": "production",
        "calibration_method": current_registered_spec.calibration_method,
        "retrain_months": current_policy["evaluation"]["retrain_months"],
    }
    if isinstance(metrics_payload, dict):
        for field, expected in expected_metrics.items():
            if metrics_payload.get(field) != expected:
                errors.append(
                    f"control_metrics.{field} must equal {expected!r}; "
                    f"got {metrics_payload.get(field)!r}"
                )

    for label, payload in (
        ("fixed_control_evidence", fixed_evidence),
        ("fixed_control_bootstrap_receipt", receipt_payload),
    ):
        if not isinstance(payload, dict):
            continue
        if payload.get("role") != "fixed_control_construction":
            errors.append(f"{label}.role must be 'fixed_control_construction'")
        if payload.get("stage2_bypassed") is not True:
            errors.append(f"{label}.stage2_bypassed must be true")
        if payload.get("research_grids_evaluated") is not False:
            errors.append(f"{label}.research_grids_evaluated must be false")

    if not isinstance(receipt_payload, dict):
        return errors
    if receipt_payload.get("schema_version") != 1:
        errors.append("fixed_control_bootstrap_receipt.schema_version must equal 1")
    expected_receipt = {
        "candidate_id": (
            f"baseline_{current_registered_spec.dataset_variant}_production"
        ),
        "dataset_variant": current_registered_spec.dataset_variant,
        "feature_family": "production",
        "calibration_method": current_registered_spec.calibration_method,
        "retrain_months": current_policy["evaluation"]["retrain_months"],
        "execution_mode": current_evaluation_protocol["execution_mode"],
        "entry_offset_days": current_evaluation_protocol[
            "entry_offset_days"
        ],
        "entry_offset_for_features": current_evaluation_protocol[
            "entry_offset_for_features"
        ],
        "require_entry_odds": current_evaluation_protocol[
            "require_entry_odds"
        ],
    }
    for field, expected in expected_receipt.items():
        if receipt_payload.get(field) != expected:
            errors.append(
                f"fixed_control_bootstrap_receipt.{field} must equal "
                f"{expected!r}; got {receipt_payload.get(field)!r}"
            )

    index_hash = _sha256(index_path) if index_path.is_file() else None
    if index_hash is not None:
        for label, payload in (
            ("control_metrics", metrics_payload),
            ("fixed_control_evidence", fixed_evidence),
            ("fixed_control_bootstrap_receipt", receipt_payload),
        ):
            if not isinstance(payload, dict):
                continue
            if payload.get("evaluation_index_sha256") != index_hash:
                errors.append(
                    f"{label}.evaluation_index_sha256 does not match "
                    f"{INTEGRITY_V2_EVALUATION_INDEX_FILENAME}"
                )
            path_value = payload.get("evaluation_index_path")
            if not isinstance(path_value, str) or Path(path_value).name != (
                INTEGRITY_V2_EVALUATION_INDEX_FILENAME
            ):
                errors.append(
                    f"{label}.evaluation_index_path must name "
                    f"{INTEGRITY_V2_EVALUATION_INDEX_FILENAME}"
                )

    artifacts = receipt_payload.get("artifacts")
    expected_receipt_artifacts = set(INTEGRITY_V2_RECEIPT_ARTIFACT_FILENAMES)
    if not isinstance(artifacts, dict):
        errors.append("fixed_control_bootstrap_receipt.artifacts must be an object")
    elif set(artifacts) != expected_receipt_artifacts:
        errors.append(
            "fixed_control_bootstrap_receipt artifact allowlist mismatch: "
            f"missing={sorted(expected_receipt_artifacts.difference(artifacts))}, "
            f"unexpected={sorted(set(artifacts).difference(expected_receipt_artifacts))}"
        )
    for artifact_name in INTEGRITY_V2_RECEIPT_ARTIFACT_FILENAMES:
        artifact_entry = (
            artifacts.get(artifact_name) if isinstance(artifacts, dict) else None
        )
        if not isinstance(artifact_entry, dict):
            errors.append(
                "fixed_control_bootstrap_receipt.artifacts is missing "
                f"{artifact_name}"
            )
            continue
        receipt_path = artifact_entry.get("path")
        if not isinstance(receipt_path, str) or Path(receipt_path).name != artifact_name:
            errors.append(
                "fixed_control_bootstrap_receipt artifact path mismatch for "
                f"{artifact_name}"
            )
        receipt_sha = artifact_entry.get("sha256")
        if not _valid_sha256(receipt_sha):
            errors.append(
                "fixed_control_bootstrap_receipt artifact has invalid SHA256 for "
                f"{artifact_name}"
            )
            continue
        source_path = artifact_sources.get(artifact_name)
        if source_path is None or not source_path.is_file():
            errors.append(
                f"Integrity-v2 freeze source is missing {artifact_name}"
            )
            continue
        actual_sha = _sha256(source_path)
        if actual_sha != receipt_sha:
            errors.append(
                "fixed_control_bootstrap_receipt artifact hash mismatch for "
                f"{artifact_name}: expected {receipt_sha}, got {actual_sha}"
            )

    errors.extend(
        _validate_integrity_v2_index(
            index_path,
            expected_bindings=receipt_payload,
        )
    )
    return errors


def _normalize_freeze_id(freeze_id: str) -> str:
    """Accept either ``20260313`` or ``control_arm_20260313``."""
    freeze_id = freeze_id.strip()
    if not freeze_id:
        raise ValueError("freeze_id must be a non-empty string")
    return freeze_id if freeze_id.startswith("control_arm_") else f"control_arm_{freeze_id}"


def _control_arm_dir(freeze_id: str) -> Path:
    """Return the directory for a given freeze ID."""
    return FROZEN_DIR / _normalize_freeze_id(freeze_id)


def _collect_checksums(arm_dir: Path) -> dict[str, str]:
    """Return checksums for every frozen artifact except metadata files."""
    checksums: dict[str, str] = {}
    for artifact in sorted(arm_dir.iterdir()):
        if not artifact.is_file():
            continue
        if artifact.name in {"checksums.json", "MANIFEST.md"}:
            continue
        checksums[artifact.name] = _sha256(artifact)
    return checksums


def _write_manifest(
    arm_dir: Path,
    freeze_id: str,
    checksums: dict[str, str],
    source_metrics: dict[str, Any],
) -> None:
    """Write a human-readable manifest for a frozen control arm."""
    normalized_id = _normalize_freeze_id(freeze_id)
    source_fields = []
    for label, field in (
        ("spec", "model_spec_name"),
        ("calibration", "calibration_method"),
        ("retrain_months", "retrain_months"),
        ("partition", "evaluation_partition"),
    ):
        value = source_metrics.get(field)
        if value is not None:
            source_fields.append(f"{label}={value}")
    source_description = "frozen control evaluation"
    if source_fields:
        source_description += f" ({', '.join(source_fields)})"
    lines = [
        f"# Control Arm Freeze: {normalized_id}",
        "",
        f"Frozen on: {normalized_id[-8:-4]}-{normalized_id[-4:-2]}-{normalized_id[-2:]}",
        f"Source: {source_description}",
        "",
        "## Artifacts",
    ]
    for name in sorted(checksums):
        lines.append(f"- {name}")
    lines += ["", "## Checksums", "See checksums.json for SHA256 integrity verification.", ""]
    (arm_dir / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")


def _coerce_artifact_frame(
    artifact: pd.DataFrame | list[dict[str, Any]] | list[float] | None,
    *,
    default_column: str,
) -> pd.DataFrame | None:
    """Normalize optional frozen trading artifacts to a DataFrame."""
    if artifact is None:
        return None
    if isinstance(artifact, pd.DataFrame):
        return artifact.copy()
    if isinstance(artifact, list):
        if not artifact:
            return pd.DataFrame(columns=[default_column])
        if isinstance(artifact[0], dict):
            return pd.DataFrame(artifact)
        return pd.DataFrame({default_column: list(artifact)})
    raise TypeError(f"Unsupported artifact type: {type(artifact)!r}")


def _load_optional_csv(path: Path) -> pd.DataFrame | None:
    """Load an optional CSV artifact and preserve event dates when present."""
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "event_date" in frame.columns:
        frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    return frame


def _extract_fresh_window_brier(metrics_payload: dict[str, Any]) -> float | None:
    """Return the frozen fresh-window Brier metric when present."""
    top_level = metrics_payload.get("fresh_data_brier")
    if top_level is not None:
        try:
            return float(top_level)
        except (TypeError, ValueError):
            return None
    nested = (
        metrics_payload.get("sliced_metrics", {})
        .get("fresh_window", {})
        .get("brier")
    )
    if nested is None:
        return None
    try:
        return float(nested)
    except (TypeError, ValueError):
        return None


def _extract_year_by_year_metrics(metrics_payload: dict[str, Any]) -> Any:
    """Return frozen year-by-year metrics when present."""
    raw = metrics_payload.get("year_by_year")
    if raw:
        return raw
    nested = metrics_payload.get("sliced_metrics", {}).get("by_year")
    return nested if nested else None


def _csv_has_rows(path: Path) -> bool:
    """Return True when a CSV artifact exists and contains at least one row."""
    frame = _load_optional_csv(path)
    return bool(frame is not None and not frame.empty)


def _make_json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats so frozen JSON stays standards-compliant."""
    if isinstance(value, dict):
        return {key: _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def freeze_control_arm(
    source_metrics: dict[str, Any],
    freeze_id: str,
    sweep_summary: dict[str, Any] | None = None,
    artifact_sources: dict[str, str | Path] | None = None,
    bet_log: pd.DataFrame | list[dict[str, Any]] | None = None,
    bankroll_history: pd.DataFrame | list[dict[str, Any]] | list[float] | None = None,
) -> Path:
    """Create a new dated frozen control arm.

    Parameters
    ----------
    source_metrics : dict
        Model evaluation metrics (must include accuracy, brier, log_loss, ece).
    freeze_id : str
        Date-based identifier, e.g. "20260313".
    sweep_summary : dict | None
        Optional trading sweep summary to freeze alongside model metrics.
    artifact_sources : dict[str, str | Path] | None
        Optional extra artifacts to copy into the freeze directory. Keys are the
        frozen filenames to create and values are source paths.
    bet_log : pd.DataFrame | list[dict[str, Any]] | None
        Optional baseline bet log to freeze as ``control_bet_log.csv``.
    bankroll_history : pd.DataFrame | list[dict[str, Any]] | list[float] | None
        Optional baseline bankroll history to freeze as
        ``control_bankroll_history.csv``.

    Returns
    -------
    Path
        Directory containing the frozen artifacts.
    """
    missing = [m for m in REQUIRED_METRICS if m not in source_metrics]
    if missing:
        raise ValueError(f"Source metrics missing required keys: {missing}")
    nonfinite = [
        metric for metric in REQUIRED_METRICS if not _finite_number(source_metrics[metric])
    ]
    if nonfinite:
        raise ValueError(f"Source metrics must be finite numbers: {nonfinite}")

    normalized_id = _normalize_freeze_id(freeze_id)
    integrity_v2_freeze = _is_integrity_v2_freeze(normalized_id)
    if integrity_v2_freeze:
        artifact_sources = dict(artifact_sources or {})
        expected_artifact_sources = {
            INTEGRITY_V2_RECEIPT_FILENAME,
            *INTEGRITY_V2_RECEIPT_ARTIFACT_FILENAMES,
        }
        unexpected_artifact_sources = sorted(
            set(artifact_sources).difference(expected_artifact_sources)
        )
        receipt_source = artifact_sources.get(INTEGRITY_V2_RECEIPT_FILENAME)
        index_source = artifact_sources.get(INTEGRITY_V2_EVALUATION_INDEX_FILENAME)
        receipt_path = Path(receipt_source) if receipt_source is not None else Path()
        index_path = Path(index_source) if index_source is not None else Path()
        if not receipt_source or not receipt_path.is_file():
            raise ValueError(
                f"Integrity-v2 freeze requires {INTEGRITY_V2_RECEIPT_FILENAME}"
            )
        if not index_source or not index_path.is_file():
            raise ValueError(
                f"Integrity-v2 freeze requires {INTEGRITY_V2_EVALUATION_INDEX_FILENAME}"
            )
        try:
            receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid fixed-control bootstrap receipt: {exc}") from exc
        evidence_errors = _validate_integrity_v2_evidence_set(
            metrics_payload=source_metrics,
            sweep_payload=sweep_summary,
            receipt_payload=receipt_payload,
            index_path=index_path,
            artifact_sources={
                name: Path(path) for name, path in artifact_sources.items()
            },
            verify_external_sources=True,
        )
        if unexpected_artifact_sources:
            evidence_errors.append(
                "Integrity-v2 freeze received unexpected artifacts: "
                f"{unexpected_artifact_sources}"
            )
        production_result_source = artifact_sources.get(
            INTEGRITY_V2_PRODUCTION_RESULT_FILENAME
        )
        if production_result_source is not None:
            production_result_path = Path(production_result_source)
            try:
                production_result_payload = json.loads(
                    production_result_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                evidence_errors.append(f"Invalid production_result.json: {exc}")
            else:
                if production_result_payload != _make_json_safe(sweep_summary):
                    evidence_errors.append(
                        "production_result.json does not match the supplied sweep summary"
                    )
        if evidence_errors:
            raise ValueError(
                "Integrity-v2 control evidence is invalid: "
                + "; ".join(evidence_errors)
            )

    arm_dir = _control_arm_dir(normalized_id)
    if arm_dir.exists():
        raise FileExistsError(f"Freeze directory already exists: {arm_dir}")
    arm_dir.mkdir(parents=True, exist_ok=False)

    metrics_path = arm_dir / "control_metrics.json"
    metrics_payload = _make_json_safe(dict(source_metrics))
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2, sort_keys=True, allow_nan=False)

    if integrity_v2_freeze:
        shutil.copy2(
            Path(artifact_sources[INTEGRITY_V2_PRODUCTION_RESULT_FILENAME]),
            arm_dir / "control_sweep_summary.json",
        )
    elif sweep_summary is not None:
        sweep_path = arm_dir / "control_sweep_summary.json"
        sweep_payload = _make_json_safe(dict(sweep_summary))
        with open(sweep_path, "w") as f:
            json.dump(sweep_payload, f, indent=4, allow_nan=False)

    if artifact_sources:
        for target_name, source_path in artifact_sources.items():
            shutil.copy2(Path(source_path), arm_dir / target_name)

    if integrity_v2_freeze:
        shutil.copy2(
            Path(artifact_sources[INTEGRITY_V2_PRODUCTION_BET_LOG_FILENAME]),
            arm_dir / "control_bet_log.csv",
        )
        shutil.copy2(
            Path(artifact_sources[INTEGRITY_V2_PRODUCTION_BANKROLL_FILENAME]),
            arm_dir / "control_bankroll_history.csv",
        )
    else:
        bet_log_frame = _coerce_artifact_frame(bet_log, default_column="profit")
        if bet_log_frame is not None:
            bet_log_frame.to_csv(arm_dir / "control_bet_log.csv", index=False)

        bankroll_frame = _coerce_artifact_frame(
            bankroll_history,
            default_column="combined",
        )
        if bankroll_frame is not None:
            bankroll_frame.to_csv(arm_dir / "control_bankroll_history.csv", index=False)

    checksums = _collect_checksums(arm_dir)

    with open(arm_dir / "checksums.json", "w") as f:
        json.dump(checksums, f, indent=4)

    _write_manifest(arm_dir, normalized_id, checksums, metrics_payload)

    logger.info("Frozen control arm %s at %s", normalized_id, arm_dir)
    return arm_dir


def validate_frozen_control_arm(freeze_id: str) -> dict[str, Any]:
    """Validate a frozen control arm's integrity.

    Checks directory existence, required files, and SHA256 checksums.

    Returns
    -------
    dict with keys: valid (bool), errors (list[str]), freeze_id, path
    """
    normalized_id = _normalize_freeze_id(freeze_id)
    errors: list[str] = []
    arm_dir = _control_arm_dir(normalized_id)

    if not arm_dir.exists():
        return {
            "valid": False,
            "passes": False,
            "errors": [f"Control arm directory not found: {arm_dir}"],
            "freeze_id": normalized_id,
            "freeze_dir": str(arm_dir),
            "path": str(arm_dir),
        }

    for fname in ["control_metrics.json", "checksums.json", "MANIFEST.md"]:
        if not (arm_dir / fname).exists():
            errors.append(f"Missing required file: {fname}")

    if errors:
        return {
            "valid": False,
            "passes": False,
            "errors": errors,
            "freeze_id": normalized_id,
            "freeze_dir": str(arm_dir),
            "path": str(arm_dir),
        }

    # Verify checksums
    try:
        with open(arm_dir / "checksums.json") as f:
            expected = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        expected = {}
        errors.append(f"Invalid checksums.json: {exc}")
    if not isinstance(expected, dict):
        errors.append("checksums.json must contain a filename-to-SHA256 object")
        expected = {}

    actual_artifact_names = {
        path.name
        for path in arm_dir.iterdir()
        if path.is_file() and path.name not in {"checksums.json", "MANIFEST.md"}
    }
    checksum_names = set(expected)
    if checksum_names != actual_artifact_names:
        errors.append(
            "checksums.json artifact allowlist mismatch: "
            f"missing={sorted(actual_artifact_names - checksum_names)}, "
            f"unexpected={sorted(checksum_names - actual_artifact_names)}"
        )
    if _is_integrity_v2_freeze(normalized_id):
        if actual_artifact_names != INTEGRITY_V2_FROZEN_ARTIFACT_FILENAMES:
            errors.append(
                "Integrity-v2 frozen artifact allowlist mismatch: "
                f"missing={sorted(INTEGRITY_V2_FROZEN_ARTIFACT_FILENAMES - actual_artifact_names)}, "
                f"unexpected={sorted(actual_artifact_names - INTEGRITY_V2_FROZEN_ARTIFACT_FILENAMES)}"
            )
        unexpected_directories = sorted(
            path.name for path in arm_dir.iterdir() if path.is_dir()
        )
        if unexpected_directories:
            errors.append(
                "Integrity-v2 freeze contains unexpected directories: "
                f"{unexpected_directories}"
            )

    for fname, expected_hash in expected.items():
        if not isinstance(fname, str) or Path(fname).name != fname:
            errors.append(f"Invalid checksummed artifact name: {fname!r}")
            continue
        if not _valid_sha256(expected_hash):
            errors.append(f"Invalid checksum for {fname}: {expected_hash!r}")
            continue
        fpath = arm_dir / fname
        if not fpath.exists():
            errors.append(f"Checksummed file missing: {fname}")
            continue
        actual_hash = _sha256(fpath)
        if actual_hash != expected_hash:
            errors.append(
                f"Checksum mismatch for {fname}: "
                f"expected {expected_hash[:12]}..., got {actual_hash[:12]}..."
            )

    # Validate metrics payload
    metrics_path = arm_dir / "control_metrics.json"
    payload: dict[str, Any] | None = None
    try:
        with open(metrics_path) as f:
            payload = json.load(f)
        missing = [k for k in REQUIRED_METRICS if k not in payload]
        if missing:
            errors.append(f"Metrics missing required keys: {missing}")
        nonfinite = [
            metric
            for metric in REQUIRED_METRICS
            if metric in payload and not _finite_number(payload.get(metric))
        ]
        if nonfinite:
            errors.append(f"Metrics must be finite numbers: {nonfinite}")
    except (json.JSONDecodeError, ValueError) as exc:
        errors.append(f"Invalid control metrics: {exc}")

    if _is_integrity_v2_freeze(normalized_id):
        for fname in (
            "control_sweep_summary.json",
            "control_bet_log.csv",
            "control_bankroll_history.csv",
            INTEGRITY_V2_RECEIPT_FILENAME,
            *INTEGRITY_V2_RECEIPT_ARTIFACT_FILENAMES,
        ):
            if not (arm_dir / fname).is_file():
                errors.append(f"Missing required integrity-v2 file: {fname}")
        if not errors and payload is not None:
            try:
                sweep_payload = json.loads(
                    (arm_dir / "control_sweep_summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                receipt_payload = json.loads(
                    (arm_dir / INTEGRITY_V2_RECEIPT_FILENAME).read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"Invalid integrity-v2 control evidence: {exc}")
            else:
                for control_name, production_name in (
                    ("control_sweep_summary.json", INTEGRITY_V2_PRODUCTION_RESULT_FILENAME),
                    ("control_bet_log.csv", INTEGRITY_V2_PRODUCTION_BET_LOG_FILENAME),
                    (
                        "control_bankroll_history.csv",
                        INTEGRITY_V2_PRODUCTION_BANKROLL_FILENAME,
                    ),
                ):
                    if _sha256(arm_dir / control_name) != _sha256(
                        arm_dir / production_name
                    ):
                        errors.append(
                            f"{control_name} does not match receipt-bound {production_name}"
                        )
                errors.extend(
                    _validate_integrity_v2_evidence_set(
                        metrics_payload=payload,
                        sweep_payload=sweep_payload,
                        receipt_payload=receipt_payload,
                        index_path=(
                            arm_dir / INTEGRITY_V2_EVALUATION_INDEX_FILENAME
                        ),
                        artifact_sources={
                            name: arm_dir / name
                            for name in INTEGRITY_V2_RECEIPT_ARTIFACT_FILENAMES
                        },
                        verify_external_sources=True,
                    )
                )

    passes = len(errors) == 0
    return {
        "valid": passes,
        "passes": passes,
        "errors": errors,
        "freeze_id": normalized_id,
        "freeze_dir": str(arm_dir),
        "path": str(arm_dir),
    }


def load_frozen_control_metrics(freeze_id: str) -> dict[str, Any]:
    """Load metrics from a frozen control arm.

    Validates integrity before loading.

    Raises
    ------
    FileNotFoundError
        If the frozen control arm doesn't exist.
    ValueError
        If integrity validation fails.
    """
    validation = validate_frozen_control_arm(freeze_id)
    if not validation["valid"]:
        if any("not found" in e for e in validation["errors"]):
            raise FileNotFoundError(
                f"Frozen control arm {freeze_id} not found at {validation['path']}"
            )
        raise ValueError(
            f"Frozen control arm {freeze_id} integrity check failed: {validation['errors']}"
        )

    with open(_control_arm_dir(freeze_id) / "control_metrics.json") as f:
        payload = json.load(f)
    loaded = dict(payload)
    for metric in REQUIRED_METRICS:
        loaded[metric] = float(payload[metric])
    return loaded


def load_frozen_sweep_summary(freeze_id: str) -> dict[str, Any] | None:
    """Load sweep summary from a frozen control arm, if present."""
    validation = validate_frozen_control_arm(freeze_id)
    if not validation["valid"]:
        raise ValueError(
            f"Frozen control arm {freeze_id} integrity check failed: {validation['errors']}"
        )

    sweep_path = _control_arm_dir(freeze_id) / "control_sweep_summary.json"
    if not sweep_path.exists():
        return None

    with open(sweep_path) as f:
        return json.load(f)


def load_frozen_control_trading_artifacts(freeze_id: str) -> dict[str, pd.DataFrame]:
    """Load optional baseline trading artifacts from a frozen control arm."""
    validation = validate_frozen_control_arm(freeze_id)
    if not validation["valid"]:
        raise ValueError(
            f"Frozen control arm {freeze_id} integrity check failed: {validation['errors']}"
        )

    arm_dir = _control_arm_dir(freeze_id)
    artifacts: dict[str, pd.DataFrame] = {}
    bet_log = _load_optional_csv(arm_dir / "control_bet_log.csv")
    if bet_log is not None:
        artifacts["bet_log"] = bet_log
    bankroll_history = _load_optional_csv(arm_dir / "control_bankroll_history.csv")
    if bankroll_history is not None:
        artifacts["bankroll_history"] = bankroll_history
    return artifacts


def validate_frozen_control_arm_for_selection_gate(freeze_id: str) -> dict[str, Any]:
    """Validate that a frozen control arm is rich enough for stage-2 gating."""
    validation = validate_frozen_control_arm(freeze_id)
    errors = list(validation["errors"])
    if not validation["valid"]:
        return {
            **validation,
            "ready": False,
            "passes": False,
            "errors": errors,
        }

    with open(_control_arm_dir(freeze_id) / "control_metrics.json") as f:
        metrics_payload = json.load(f)

    fresh_window_brier = _extract_fresh_window_brier(metrics_payload)
    if fresh_window_brier is None:
        errors.append(
            "control_metrics.json is missing fresh-window Brier "
            "(expected fresh_data_brier or sliced_metrics.fresh_window.brier)."
        )
    elif not math.isfinite(fresh_window_brier):
        errors.append("control_metrics.json fresh-window Brier must be finite.")

    ready = len(errors) == 0
    return {
        **validation,
        "ready": ready,
        "passes": ready,
        "errors": errors,
    }


def main() -> None:
    """CLI entry point: freeze a promoted candidate as a new control arm.

    Usage::

        python -m src.strategy.control_arm \\
          --run-dir data/logs/evaluation/run_20260316_044304 \\
          --candidate-id rematch_features_append_only_2026_production \\
          --freeze-id 20260316
    """
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Freeze a promoted model as a new control arm")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Evaluation run directory (e.g. data/logs/evaluation/run_20260316_044304)",
    )
    parser.add_argument(
        "--candidate-id",
        required=True,
        help=(
            "Promoted candidate ID using single underscores "
            "(e.g. rematch_features_append_only_2026_production)"
        ),
    )
    parser.add_argument(
        "--freeze-id",
        required=True,
        help="Date-based freeze identifier (e.g. 20260316)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = Path(__file__).resolve().parent.parent.parent / run_dir

    # Resolve cell metrics JSON.
    # candidate_id uses single underscores:  rematch_features_append_only_2026_production
    # filename uses double underscores:       rematch_features__append_only_2026__production__isotonic_metrics.json
    cells_dir = run_dir / "cells"
    if not cells_dir.exists():
        logger.error("cells/ directory not found under %s", run_dir)
        sys.exit(1)

    cell_path: Path | None = None
    for f in sorted(cells_dir.glob("*_metrics.json")):
        simplified = (
            f.stem
            .replace("__isotonic_metrics", "")
            .replace("__sigmoid_metrics", "")
            .replace("__", "_")
        )
        if simplified == args.candidate_id:
            cell_path = f
            break

    if cell_path is None:
        candidates = [
            f.stem
            .replace("__isotonic_metrics", "")
            .replace("__sigmoid_metrics", "")
            .replace("__", "_")
            for f in sorted(cells_dir.glob("*_metrics.json"))
        ]
        logger.error("No cell metrics found for candidate %r", args.candidate_id)
        logger.error("Available candidates:\n  %s", "\n  ".join(candidates[:20]))
        sys.exit(1)

    logger.info("Using cell metrics: %s", cell_path.name)
    with open(cell_path) as fh:
        cell_data = json.load(fh)

    m = cell_data["metrics"]
    sliced = cell_data.get("sliced_metrics", {})
    source_metrics: dict[str, Any] = {
        "accuracy": m["accuracy"],
        "brier": m["brier"],
        "log_loss": m["log_loss"],
        "ece": m["ece"],
        "fresh_data_brier": sliced.get("fresh_window", {}).get("brier"),
        "year_by_year": sliced.get("by_year"),
        "bootstrap_ci": m.get("bootstrap_ci"),
        "n_predictions": cell_data.get("n_predictions"),
        "n_folds": cell_data.get("n_folds"),
        "model_variant": cell_data.get("model_variant"),
        "feature_family": cell_data.get("feature_family"),
        "dataset_variant": cell_data.get("dataset_variant"),
        "calibration_method": cell_data.get("calibration_method"),
        "retrain_months": cell_data.get("retrain_months"),
        "evaluated_at": cell_data.get("evaluated_at"),
        "evaluation_sample_sha256": cell_data.get("evaluation_sample_sha256"),
        "model_spec_name": cell_data.get("model_spec_name"),
        "model_spec_payload_sha256": cell_data.get(
            "model_spec_payload_sha256"
        ),
        "feature_contract_count": cell_data.get("feature_contract_count"),
        "feature_contract_sha256": cell_data.get("feature_contract_sha256"),
    }
    for field in _INTEGRITY_V2_BINDINGS:
        source_metrics[field] = cell_data.get(field)
    for field in _INTEGRITY_V2_PARTITION_BINDING_FIELDS:
        source_metrics[field] = cell_data.get(field)

    # Freeze only the predeclared fixed production controls. Broad/narrow
    # winners are research evidence and are never eligible to become a control.
    sweep_dir = run_dir / "stage3_sweeps" / args.candidate_id
    production_result_path = sweep_dir / "production_result.json"
    if not production_result_path.exists():
        logger.error("Fixed production result is missing: %s", production_result_path)
        sys.exit(1)
    with open(production_result_path) as fh:
        sweep_summary = json.load(fh)
    if sweep_summary.get("promotion_eligible") is not True:
        logger.error(
            "Refusing non-production/research sweep artifact: %s",
            production_result_path,
        )
        sys.exit(1)
    fixed_evidence = sweep_summary.get("fixed_control_evidence")
    integrity_v2_freeze = _is_integrity_v2_freeze(args.freeze_id)
    if integrity_v2_freeze and fixed_evidence is None:
        logger.error(
            "Integrity-v2 control requires fixed_control_evidence: %s",
            production_result_path,
        )
        sys.exit(1)
    if fixed_evidence is not None:
        if not isinstance(fixed_evidence, dict):
            logger.error("Fixed-control evidence is malformed: %s", production_result_path)
            sys.exit(1)
        if integrity_v2_freeze:
            expected_evidence = {
                field: cell_data.get(field)
                for field in (
                    *_INTEGRITY_V2_BINDINGS,
                    *_INTEGRITY_V2_PARTITION_BINDING_FIELDS,
                )
            }
            expected_evidence["source_fingerprint"] = cell_data.get(
                "source_fingerprint"
            )
            expected_evidence["evaluation_protocol"] = cell_data.get(
                "evaluation_protocol"
            )
            for field in _INTEGRITY_V2_DYNAMIC_PROVENANCE_FIELDS:
                if field not in _INTEGRITY_V2_STAGE1_MATERIALIZED_PATH_FIELDS:
                    expected_evidence[field] = cell_data.get(field)
        else:
            expected_evidence = {
                "model_spec_name": cell_data.get("model_spec_name"),
                "model_spec_payload_sha256": cell_data.get(
                    "model_spec_payload_sha256"
                ),
                "evaluation_sample_sha256": cell_data.get(
                    "evaluation_sample_sha256"
                ),
                "evaluation_n_fights": cell_data.get("n_predictions"),
                "evaluation_n_folds": cell_data.get("n_folds"),
            }
        mismatched = [
            field
            for field, expected in expected_evidence.items()
            if fixed_evidence.get(field) != expected
        ]
        if mismatched:
            logger.error(
                "Fixed-control production evidence differs from Stage 1: %s",
                mismatched,
            )
            sys.exit(1)
        if integrity_v2_freeze:
            for field in _INTEGRITY_V2_DYNAMIC_PROVENANCE_FIELDS:
                source_metrics[field] = fixed_evidence.get(field)
            source_metrics["source_fingerprint"] = fixed_evidence.get(
                "source_fingerprint"
            )
            source_metrics["evaluation_protocol"] = fixed_evidence.get(
                "evaluation_protocol"
            )
            source_metrics["evaluation_index_path"] = fixed_evidence.get(
                "evaluation_index_path"
            )
            source_metrics["evaluation_index_sha256"] = fixed_evidence.get(
                "evaluation_index_sha256"
            )
    if not integrity_v2_freeze:
        sweep_summary.setdefault(
            "evaluation_sample_sha256",
            cell_data.get("evaluation_sample_sha256"),
        )
    logger.info("Loaded fixed production controls from %s", production_result_path)

    # Load bet log and bankroll history
    bet_log_path = sweep_dir / "production_bet_log.csv"
    if not bet_log_path.exists():
        logger.error("Fixed production bet log is missing: %s", bet_log_path)
        sys.exit(1)
    bet_log = pd.read_csv(bet_log_path)
    logger.info("Loaded fixed production bet log: %d rows", len(bet_log))

    bh_path = sweep_dir / "production_bankroll_history.csv"
    if not bh_path.exists():
        logger.error("Fixed production bankroll history is missing: %s", bh_path)
        sys.exit(1)
    bankroll_history = pd.read_csv(bh_path)
    logger.info("Loaded fixed production bankroll history: %d rows", len(bankroll_history))

    artifact_sources = None
    if integrity_v2_freeze:
        receipt_path = run_dir / INTEGRITY_V2_RECEIPT_FILENAME
        evaluation_index_path = run_dir / INTEGRITY_V2_EVALUATION_INDEX_FILENAME
        input_provenance_path = run_dir / INTEGRITY_V2_INPUT_PROVENANCE_FILENAME
        odds_inventory_path = run_dir / INTEGRITY_V2_ODDS_SOURCE_INVENTORY_FILENAME
        source_inventory_path = run_dir / INTEGRITY_V2_SOURCE_INVENTORY_FILENAME
        environment_path = run_dir / INTEGRITY_V2_ENVIRONMENT_FILENAME
        artifact_sources = {
            INTEGRITY_V2_RECEIPT_FILENAME: receipt_path,
            INTEGRITY_V2_EVALUATION_INDEX_FILENAME: evaluation_index_path,
            INTEGRITY_V2_PRODUCTION_RESULT_FILENAME: production_result_path,
            INTEGRITY_V2_PRODUCTION_BET_LOG_FILENAME: bet_log_path,
            INTEGRITY_V2_PRODUCTION_BANKROLL_FILENAME: bh_path,
            INTEGRITY_V2_PRODUCTION_ROW_FILENAME: (
                sweep_dir / INTEGRITY_V2_PRODUCTION_ROW_FILENAME
            ),
            INTEGRITY_V2_INPUT_PROVENANCE_FILENAME: input_provenance_path,
            INTEGRITY_V2_ODDS_SOURCE_INVENTORY_FILENAME: odds_inventory_path,
            INTEGRITY_V2_SOURCE_INVENTORY_FILENAME: source_inventory_path,
            INTEGRITY_V2_ENVIRONMENT_FILENAME: environment_path,
        }

    arm_dir = freeze_control_arm(
        source_metrics=source_metrics,
        freeze_id=args.freeze_id,
        sweep_summary=sweep_summary,
        artifact_sources=artifact_sources,
        bet_log=bet_log,
        bankroll_history=bankroll_history,
    )
    logger.info("Frozen to: %s", arm_dir)

    result = validate_frozen_control_arm(args.freeze_id)
    if result["valid"]:
        logger.info("Integrity check: PASS")
    else:
        logger.error("Integrity check: FAIL — %s", result["errors"])
        sys.exit(1)


if __name__ == "__main__":
    main()


def validate_frozen_control_arm_for_promotion_gate(freeze_id: str) -> dict[str, Any]:
    """Validate that a frozen control arm supports a trustworthy stage-4 verdict."""
    selection_validation = validate_frozen_control_arm_for_selection_gate(freeze_id)
    errors = list(selection_validation["errors"])
    if not selection_validation["valid"]:
        return {
            **selection_validation,
            "ready": False,
            "passes": False,
            "errors": errors,
        }

    arm_dir = _control_arm_dir(freeze_id)
    with open(arm_dir / "control_metrics.json") as f:
        metrics_payload = json.load(f)

    if _extract_year_by_year_metrics(metrics_payload) is None:
        errors.append(
            "control_metrics.json is missing year-by-year metrics "
            "(expected year_by_year or sliced_metrics.by_year)."
        )

    sweep_path = arm_dir / "control_sweep_summary.json"
    if not sweep_path.exists():
        errors.append("Missing control_sweep_summary.json for the trading bar baseline.")
    else:
        try:
            with open(sweep_path) as f:
                sweep_payload = json.load(f)
            from src.strategy.promotion_gate import _canonicalize_sweep_payload

            canonical_sweep = _canonicalize_sweep_payload(sweep_payload, sweep_path)
            nonfinite_trading_metrics = [
                field
                for field in ("roi", "total_profit", "max_drawdown", "avg_clv")
                if not _finite_number(canonical_sweep.get(field))
            ]
            if nonfinite_trading_metrics:
                errors.append(
                    "control_sweep_summary.json has missing/non-finite trading "
                    f"metrics: {nonfinite_trading_metrics}."
                )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append(
                "control_sweep_summary.json is not a usable trading baseline: "
                f"{exc}"
            )

    bet_log_path = arm_dir / "control_bet_log.csv"
    if not _csv_has_rows(bet_log_path):
        errors.append("Missing non-empty control_bet_log.csv for baseline trading-volume checks.")

    bankroll_path = arm_dir / "control_bankroll_history.csv"
    if not _csv_has_rows(bankroll_path):
        errors.append("Missing non-empty control_bankroll_history.csv for baseline trading artifacts.")

    ready = len(errors) == 0
    return {
        **selection_validation,
        "ready": ready,
        "passes": ready,
        "errors": errors,
    }
