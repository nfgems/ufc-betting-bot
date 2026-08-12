"""Full UFC model evaluation orchestrator.

Runs the phased UFC evaluation pipeline end to end:
1. Matrix evaluation across model variants, dataset variants, and feature families.
2. Selection gate finalist filtering against a frozen control arm.
3. Finalist trading sweeps.
4. Promotion gate verdicts (report only; no deployment).
5. Consolidated markdown report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata as importlib_metadata
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src.config import (
    BETSAPI_MMA_PROCESSED_DIR,
    BETSAPI_MMA_RAW_DIR,
    INITIAL_BANKROLL,
    LOGS_DIR,
    RAW_DATA_DIR,
    TRAIN_CUTOFF_DATE,
)
from src.data.betsapi_mma import (
    augment_features_with_betsapi_mma,
    summarize_saved_betsapi_mma_backfill,
)
from src.data.kaggle_loader import load_kaggle_dataset
from src.data.name_utils import canonical_fighter_name_key
from src.data.ufc_refresh import TRAINING_DATASET_VARIANTS, build_training_dataset_variants
from src.features.build_features import (
    BETSAPI_CHALLENGER_FEATURE_NAMES,
    BETSAPI_HISTORICAL_FEATURE_NAMES,
)
from src.model.training_spec import resolve_named_training_spec
from src.strategy.compare_matrix import ALL_FEATURE_FAMILIES, BETSAPI_FEATURE_FAMILIES
from src.strategy.confirmation_ledger import (
    require_recorded_anchor_in_history,
    require_remotely_anchored_git_artifacts,
)
from src.strategy.control_arm import (
    INTEGRITY_V2_EVALUATION_INDEX_NA_REP,
    _validate_integrity_v2_index,
    load_frozen_control_metrics,
    load_frozen_control_trading_artifacts,
    validate_frozen_control_arm,
    validate_frozen_control_arm_for_promotion_gate,
    validate_frozen_control_arm_for_selection_gate,
)
from src.strategy.duo_trader_sweep import (
    SweepConfig,
    _evaluate_config,
    _generate_walk_forward_predictions,
    _production_sweep_config,
    _sort_summary_rows,
    _summary_row_from_result,
    build_sweep_configs,
)
from src.strategy.finalist_sweep import (
    _row_to_sweep_config,
    compare_finalist_sweeps,
    narrow_sweep_around,
)
from src.strategy.lab_stats import compute_ece
from src.strategy.model_lab import (
    DEFAULT_CONFIRMATION_FOLD_COUNT,
    build_variant_features,
    generate_variant_fold_predictions,
    partition_walk_forward_folds,
    resolve_variant_feature_columns,
)
from src.strategy.model_variants import ALL_VARIANTS, variant_from_named_training_spec
from src.strategy.promotion_gate import (
    _canonicalize_sweep_payload,
    evaluate_for_promotion,
    generate_promotion_report,
)
from src.strategy.selection_gate import CandidateResult, SelectionGate, SweepTargetSpec

logger = logging.getLogger(__name__)

EVALUATION_DIR = LOGS_DIR / "evaluation"
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

FRESH_WINDOW_CUTOFF = pd.Timestamp.now() - pd.DateOffset(months=6)
FEATURE_CACHE_DIRNAME = "features"
DATASET_CACHE_DIRNAME = "datasets"
CELL_OUTPUT_DIRNAME = "cells"
SWEEP_OUTPUT_DIRNAME = "stage3_sweeps"
PROMOTION_OUTPUT_DIRNAME = "stage4_promotions"
STAGE3_STATE_FILENAME = "stage3_state.json"
CONFIRMATION_LOCK_FILENAME = "confirmation_package_lock.json"
CONFIRMATION_CLAIM_FILENAME = "confirmation_evaluation_claim.json"
CONFIRMATION_RESULT_FILENAME = "confirmation_evaluation_result.json"
CONFIRMATION_COMPLETION_FILENAME = "confirmation_evaluation_completion.json"
CONFIRMATION_CLAIM_REFERENCE_FILENAME = "confirmation_evaluation_claim_reference.json"
GLOBAL_CONFIRMATION_CLAIMS_DIR = Path("evidence/confirmation_claims")
GLOBAL_SELECTION_EXPOSURES_DIRNAME = "_selection_exposures"
BOUNDED_SELECTION_SUMMARY_FILENAME = "bounded_selection_summary.json"
BOUNDED_SELECTION_RESULTS_DIRNAME = "bounded_selection_results"
BOUNDED_SELECTION_RANKING_POLICY = (
    "selection_gate_composite_desc_then_strategy_roi_desc_then_"
    "drawdown_asc_then_package_id"
)
FIXED_CONTROL_INPUT_PROVENANCE_FILENAME = "fixed_control_input_provenance.json"
FIXED_CONTROL_ODDS_INVENTORY_FILENAME = "fixed_control_odds_source_inventory.json"
SOURCE_INVENTORY_FILENAME = "evaluation_source_inventory.json"
ENVIRONMENT_INVENTORY_FILENAME = "evaluation_environment.json"
EVALUATION_SCRIPT_SOURCE_FILES = (
    "scripts/bfo_lineage.py",
    "scripts/check_production_refit_contract.py",
    "scripts/check_scheduled_refit_quality.py",
    "scripts/check_tracked_artifact_integrity.py",
    "scripts/recover_post_cutoff_method_odds.py",
    "scripts/track_c_batch.py",
)
EVALUATION_DEPENDENCY_FILES = ("requirements.txt",)
SCHEDULED_REFIT_POLICY_PATH = Path("config/scheduled_refit_policy_v2.json")
FINAL_CORRECTED_FIGHTS_PATH = Path(
    "data/processed/audit_remediation_integrity_v2_final/fights_cleaned.csv"
)
FINAL_CORRECTED_FEATURES_PATH = Path(
    "data/processed/audit_remediation_integrity_v2_final/features.csv"
)
FINAL_CORRECTED_FIGHTS_SHA256 = (
    "0c4d616474155ccc611bd88e77309c4e954cfc64b04a691d92473c529caae8d1"
)
DEFAULT_FREEZE_ID = "20260320"
DEFAULT_BOOTSTRAP = 5000
DEFAULT_RETRAIN_MONTHS = 4
DEFAULT_MAX_FINALISTS = 3
HISTORICAL_MATRIX_EXCLUDED_VARIANTS = frozenset({"betsapi_challenger"})
HISTORICAL_MATRIX_EXCLUDED_FAMILIES = frozenset(
    {
        "production_betsapi_expanded",
        "ufcstats_betsapi_expanded",
    }
)
MATERIAL_RESUME_FIELDS = (
    "variants",
    "datasets",
    "families",
    "freeze_id",
    "bootstrap",
    "calibration_method",
    "retrain_months",
    "max_finalists",
    "run_narrow",
    "execution_mode",
    "entry_offset_days",
    "entry_offset_for_features",
    "require_entry_odds",
    "allow_closing_odds",
    "fixed_control_bootstrap",
    "selected_model_spec_name",
    "selected_model_spec_payload_sha256",
    "expected_evaluation_sample_sha256",
    "expected_evaluation_n_fights",
    "expected_evaluation_n_folds",
    "reserved_confirmation_folds",
)
FINGERPRINT_SOURCE_FILES = (
    "src/strategy/run_evaluation.py",
    "src/strategy/model_lab.py",
    "src/strategy/duo_trader_sweep.py",
    "src/strategy/finalist_sweep.py",
    "src/strategy/selection_gate.py",
    "src/strategy/promotion_gate.py",
    "src/strategy/model_variants.py",
    "src/strategy/compare_matrix.py",
    "src/data/ufc_refresh.py",
    "src/data/betsapi_mma.py",
    "src/features/build_features.py",
    "src/model/train.py",
    "src/strategy/robustness.py",
    "src/strategy/control_arm.py",
    "src/config.py",
)


@dataclass(frozen=True)
class CellSpec:
    """One matrix cell in the full evaluation grid."""

    model_variant: str
    dataset_variant: str
    feature_family: str
    calibration_method: str
    retrain_months: int
    bootstrap: int

    @property
    def key(self) -> str:
        return "|".join(
            (
                self.model_variant,
                self.dataset_variant,
                self.feature_family,
                self.calibration_method,
            )
        )

    @property
    def filename_stem(self) -> str:
        return "__".join(
            _safe_name(part)
            for part in (
                self.model_variant,
                self.dataset_variant,
                self.feature_family,
                self.calibration_method,
            )
        )


def build_cell_specs(
    *,
    variants: list[str],
    datasets: list[str],
    families: list[str],
    calibration_method: str | None,
    retrain_months: int,
    bootstrap: int,
) -> list[CellSpec]:
    """Return the full model x dataset x feature-family grid."""
    return [
        CellSpec(
            model_variant=model_variant,
            dataset_variant=dataset_variant,
            feature_family=feature_family,
            calibration_method=_resolve_variant_calibration(
                model_variant,
                calibration_method,
            ),
            retrain_months=retrain_months,
            bootstrap=bootstrap,
        )
        for dataset_variant in datasets
        for feature_family in families
        for model_variant in variants
    ]


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_git_command(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=_repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _collect_runtime_code_metadata() -> dict[str, Any]:
    repo_root = _repo_root()
    source_paths = list(sorted((repo_root / "src").rglob("*.py")))
    for relative_path in EVALUATION_SCRIPT_SOURCE_FILES:
        path = repo_root / relative_path
        if not path.is_file():
            raise ValueError(f"required evaluation source is missing: {relative_path}")
        source_paths.append(path)
    source_entries = []
    for path in sorted(set(source_paths), key=lambda value: value.as_posix()):
        relative_path = path.relative_to(repo_root).as_posix()
        source_entries.append(
            {"path": relative_path, "sha256": _file_sha256(path)}
        )
    if not source_entries:
        raise ValueError("evaluation source inventory is empty")
    dependency_files = []
    for relative_path in EVALUATION_DEPENDENCY_FILES:
        path = repo_root / relative_path
        if not path.is_file():
            raise ValueError(f"required dependency contract is missing: {relative_path}")
        dependency_files.append(
            {"path": relative_path, "sha256": _file_sha256(path)}
        )
    package_versions = {
        distribution: importlib_metadata.version(distribution)
        for distribution in ("numpy", "pandas", "scikit-learn", "xgboost")
    }
    environment = {
        "schema_version": 1,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": package_versions,
        "dependency_files": dependency_files,
    }
    environment_payload_sha256 = _canonical_json_sha256(environment)
    source_inventory = {
        "schema_version": 2,
        "source_contract": {
            "python_glob": "src/**/*.py",
            "required_scripts": list(EVALUATION_SCRIPT_SOURCE_FILES),
        },
        "sources": source_entries,
        "environment_payload_sha256": environment_payload_sha256,
    }
    source_fingerprint = _canonical_json_sha256(source_inventory)

    dirty_output = _run_git_command("status", "--short")
    return {
        "git_sha": _run_git_command("rev-parse", "HEAD"),
        "git_dirty": bool(dirty_output) if dirty_output is not None else None,
        "source_fingerprint": source_fingerprint,
        "source_inventory_sha256": source_fingerprint,
        "source_inventory": source_inventory,
        "environment": environment,
        "environment_payload_sha256": environment_payload_sha256,
    }


def _feature_cache_format() -> str:
    if importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet"):
        return "parquet"
    return "pickle"


def _feature_cache_suffix(cache_format: str) -> str:
    return ".parquet" if cache_format == "parquet" else ".pkl"


def _feature_cache_path(
    run_dir: Path,
    *,
    dataset_variant: str,
    profile_key: str,
    include_betsapi: bool,
    cache_format: str,
) -> Path:
    flavor = "betsapi" if include_betsapi else "base"
    suffix = _feature_cache_suffix(cache_format)
    filename = f"{_safe_name(dataset_variant)}__{_safe_name(profile_key)}__{flavor}{suffix}"
    return run_dir / FEATURE_CACHE_DIRNAME / filename


def _dataset_cache_path(
    run_dir: Path,
    *,
    dataset_variant: str,
    cache_format: str,
) -> Path:
    return (
        run_dir
        / DATASET_CACHE_DIRNAME
        / f"{_safe_name(dataset_variant)}{_feature_cache_suffix(cache_format)}"
    )


def _save_cached_frame(df: pd.DataFrame, path: Path, cache_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if cache_format == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_pickle(path)


def _load_cached_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_pickle(path)


def _cell_output_path(run_dir: Path, cell: CellSpec) -> Path:
    return run_dir / CELL_OUTPUT_DIRNAME / f"{cell.filename_stem}_metrics.json"


def _stage3_candidate_dir(run_dir: Path, candidate_id: str) -> Path:
    return run_dir / SWEEP_OUTPUT_DIRNAME / _safe_name(candidate_id)


def _stage4_report_path(run_dir: Path, candidate_id: str) -> Path:
    return run_dir / PROMOTION_OUTPUT_DIRNAME / f"{_safe_name(candidate_id)}_promotion.md"


def _normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return _normalize_json(value.to_dict(orient="records"))
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _normalize_json(value.item())
    if isinstance(value, np.ndarray):
        return _normalize_json(value.tolist())
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _normalize_json(asdict(value) if hasattr(value, "__dataclass_fields__") else vars(value))
    return value


def _evaluation_sample_sha256(predictions: pd.DataFrame) -> str:
    columns = [
        "event_date", "fighter_a", "fighter_b", "target",
        "fold", "train_end", "test_end",
    ]
    missing = [column for column in columns if column not in predictions.columns]
    if predictions.empty or missing:
        raise ValueError(f"evaluation sample cannot be bound; missing columns: {missing}")
    sample = predictions[columns].copy()
    for column in ("event_date", "train_end", "test_end"):
        sample[column] = pd.to_datetime(sample[column], errors="raise").dt.strftime("%Y-%m-%d")
    sample["target"] = pd.to_numeric(sample["target"], errors="raise").astype(int)
    sample["fold"] = pd.to_numeric(sample["fold"], errors="raise").astype(int)
    if sample.duplicated(["event_date", "fighter_a", "fighter_b"]).any():
        raise ValueError("evaluation sample contains duplicate fight identities")
    sample = sample.sort_values(columns, kind="stable").reset_index(drop=True)
    return hashlib.sha256(
        sample.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _canonical_evaluation_fight_identities(
    predictions: pd.DataFrame,
) -> list[dict[str, str]]:
    """Return orientation-invariant bout identities without labels or inputs."""
    source_columns = ["event_date", "fighter_a", "fighter_b"]
    if predictions.empty:
        raise ValueError("evaluation fight identity cannot bind an empty frame")
    missing = [column for column in source_columns if column not in predictions.columns]
    if missing:
        raise ValueError(f"evaluation fight identity is missing columns: {missing}")
    identity = predictions.copy()
    identity["event_date"] = pd.to_datetime(
        identity["event_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    identity["fighter_a_name_key"] = identity["fighter_a"].map(
        canonical_fighter_name_key
    )
    identity["fighter_b_name_key"] = identity["fighter_b"].map(
        canonical_fighter_name_key
    )
    if (identity[["fighter_a_name_key", "fighter_b_name_key"]] == "").any().any():
        raise ValueError("evaluation fight identity contains an empty normalized name")
    records: list[dict[str, str]] = []
    for row in identity.to_dict(orient="records"):
        name_pair = sorted((row["fighter_a_name_key"], row["fighter_b_name_key"]))
        records.append(
            {
                "event_date": row["event_date"],
                "fighter_name_low": name_pair[0],
                "fighter_name_high": name_pair[1],
            }
        )
    records.sort(
        key=lambda row: (
            row["event_date"],
            row["fighter_name_low"],
            row["fighter_name_high"],
        )
    )
    if len({_canonical_json_sha256(record) for record in records}) != len(records):
        raise ValueError("evaluation fight identities contain duplicates")
    return records


def _evaluation_fight_identity_sha256(predictions: pd.DataFrame) -> str:
    return _canonical_json_sha256(_canonical_evaluation_fight_identities(predictions))


def _prediction_rows_sha256(predictions: pd.DataFrame) -> str:
    """Hash the exact ordered rows on which model predictions are reported."""
    columns = [
        "event_date",
        "fighter_a",
        "fighter_b",
        "target",
        "fold",
        "train_end",
        "test_end",
    ]
    return _canonical_frame_value_sha256(predictions, columns=columns)


def _prediction_values_sha256(predictions: pd.DataFrame) -> str:
    """Hash exact row identities plus every load-bearing scored probability."""
    required = ["prob_a", "prob_b", "no_odds_prob_a", "no_odds_prob_b"]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"prediction values cannot be bound; missing columns: {missing}")
    identity = [
        "event_date",
        "fighter_a",
        "fighter_b",
        "target",
        "fold",
        "train_end",
        "test_end",
    ]
    value_columns = sorted(
        column
        for column in predictions.columns
        if column.startswith(("prob_", "no_odds_prob_", "entry_", "market_"))
        or column
        in {
            "a_implied_prob",
            "b_implied_prob",
            "diff_implied_prob",
            "a_fair_prob_avg",
            "b_fair_prob_avg",
            "bet_eligible",
        }
    )
    return _canonical_frame_value_sha256(
        predictions,
        columns=list(dict.fromkeys(identity + value_columns)),
    )


def _post_cutoff_predictions(
    fold_predictions: list[tuple[int, pd.DataFrame]],
) -> pd.DataFrame:
    """Concatenate one fold partition and apply the common betting cutoff."""
    predictions = (
        pd.concat([frame for _fold, frame in fold_predictions], ignore_index=True)
        if fold_predictions
        else pd.DataFrame()
    )
    if predictions.empty:
        return predictions
    predictions["event_date"] = pd.to_datetime(
        predictions["event_date"], errors="coerce"
    )
    return predictions[
        predictions["event_date"] >= pd.Timestamp(TRAIN_CUTOFF_DATE)
    ].copy()


def _partition_evaluation_folds(
    fold_predictions: list[tuple[int, pd.DataFrame]],
    *,
    reserved_confirmation_folds: int,
) -> tuple[
    list[tuple[int, pd.DataFrame]],
    list[tuple[int, pd.DataFrame]],
]:
    """Return selection/confirmation folds under the run's immutable policy."""
    if isinstance(reserved_confirmation_folds, bool) or not isinstance(
        reserved_confirmation_folds, int
    ):
        raise ValueError("reserved_confirmation_folds must be an integer")
    if reserved_confirmation_folds < 0:
        raise ValueError("reserved_confirmation_folds cannot be negative")
    if reserved_confirmation_folds == 0:
        return list(fold_predictions), []
    if reserved_confirmation_folds != DEFAULT_CONFIRMATION_FOLD_COUNT:
        raise ValueError(
            "the performance-recovery protocol requires exactly "
            f"{DEFAULT_CONFIRMATION_FOLD_COUNT} reserved confirmation folds"
        )
    return partition_walk_forward_folds(
        fold_predictions,
        confirmation_fold_count=reserved_confirmation_folds,
    )


def _fold_partition_evidence(
    fold_manifest: list[tuple[int, pd.DataFrame]],
    selection_folds: list[tuple[int, pd.DataFrame]],
    *,
    reserved_confirmation_folds: int,
    feature_contract_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Bind partitions from a metadata-only manifest and scored selection rows."""
    selection_manifest, confirmation_manifest = _partition_evaluation_folds(
        fold_manifest,
        reserved_confirmation_folds=reserved_confirmation_folds,
    )
    full_predictions = _post_cutoff_predictions(fold_manifest)
    selection_predictions = _post_cutoff_predictions(selection_folds)
    selection_index = _post_cutoff_predictions(selection_manifest)
    confirmation_predictions = _post_cutoff_predictions(confirmation_manifest)
    if full_predictions.empty or selection_predictions.empty:
        raise ValueError("walk-forward evaluation produced an empty bound partition")
    selection_hash = _evaluation_sample_sha256(selection_predictions)
    selection_index_hash = _evaluation_sample_sha256(selection_index)
    if selection_hash != selection_index_hash:
        raise ValueError(
            "scored selection rows differ from the metadata-only fold manifest"
        )
    evaluation_anchor = pd.to_datetime(
        full_predictions["event_date"], errors="raise"
    ).max().normalize()
    selection_anchor = pd.to_datetime(
        selection_index["event_date"], errors="raise"
    ).max().normalize()
    fresh_window_cutoff = selection_anchor - pd.DateOffset(months=6)
    evidence = {
        "evaluation_partition": (
            "selection" if reserved_confirmation_folds else "all"
        ),
        "reserved_confirmation_folds": reserved_confirmation_folds,
        "selection_fold_ids": [
            fold_id for fold_id, _frame in selection_manifest
        ],
        "confirmation_fold_ids": [
            fold_id for fold_id, _frame in confirmation_manifest
        ],
        "evaluation_sample_sha256": selection_hash,
        "evaluation_fight_identity_sha256": _evaluation_fight_identity_sha256(
            selection_index
        ),
        "selection_fight_identities": _canonical_evaluation_fight_identities(
            selection_index
        ),
        "n_predictions": int(len(selection_predictions)),
        "n_folds": len(selection_folds),
        "full_evaluation_sample_sha256": _evaluation_sample_sha256(full_predictions),
        "full_evaluation_fight_identity_sha256": _evaluation_fight_identity_sha256(
            full_predictions
        ),
        "full_evaluation_n_fights": int(len(full_predictions)),
        "full_evaluation_n_folds": len(fold_manifest),
        "confirmation_evaluation_sample_sha256": (
            _evaluation_sample_sha256(confirmation_predictions)
            if not confirmation_predictions.empty
            else None
        ),
        "confirmation_evaluation_fight_identity_sha256": (
            _evaluation_fight_identity_sha256(confirmation_predictions)
            if not confirmation_predictions.empty
            else None
        ),
        "confirmation_fight_identities": (
            _canonical_evaluation_fight_identities(confirmation_predictions)
            if not confirmation_predictions.empty
            else []
        ),
        "confirmation_evaluation_n_fights": int(len(confirmation_predictions)),
        "confirmation_evaluation_n_folds": len(confirmation_manifest),
        "evaluation_end_date": str(evaluation_anchor.date()),
        "fresh_window_cutoff": str(fresh_window_cutoff.date()),
    }
    if feature_contract_columns is not None:
        evidence.update(_ordered_feature_contract_evidence(feature_contract_columns))
        evidence.update(
            {
                "evaluation_input_value_sha256": _manifest_input_value_sha256(
                    selection_index,
                    feature_contract_columns=feature_contract_columns,
                ),
                "full_evaluation_input_value_sha256": _manifest_input_value_sha256(
                    full_predictions,
                    feature_contract_columns=feature_contract_columns,
                ),
                "confirmation_evaluation_input_value_sha256": (
                    _manifest_input_value_sha256(
                        confirmation_predictions,
                        feature_contract_columns=feature_contract_columns,
                    )
                    if not confirmation_predictions.empty
                    else None
                ),
            }
        )
    return evidence


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_artifact_sha256(value: object) -> str:
    encoded = json.dumps(
        _normalize_json(value),
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ordered_feature_contract_evidence(columns: list[str]) -> dict[str, Any]:
    """Return the exact ordered feature contract used by the trainer."""
    if not columns or any(not isinstance(column, str) or not column for column in columns):
        raise ValueError("feature contract must contain non-empty column names")
    if len(set(columns)) != len(columns):
        raise ValueError("feature contract contains duplicate columns")
    ordered = list(columns)
    return {
        "feature_contract_columns": ordered,
        "feature_contract_count": len(ordered),
        "feature_contract_sha256": _canonical_json_sha256(ordered),
    }


def _canonical_frame_value_sha256(
    frame: pd.DataFrame,
    *,
    columns: list[str] | None = None,
) -> str:
    """Hash a dataframe's schema and values using a stable text encoding."""
    selected_columns = list(columns) if columns is not None else sorted(frame.columns)
    missing = [column for column in selected_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"cannot hash dataframe values; missing columns: {missing}")
    canonical = frame[selected_columns].copy()
    identity = [
        column
        for column in ("event_date", "fighter_a", "fighter_b", "fold", "train_end", "test_end")
        if column in canonical.columns
    ]
    if identity:
        canonical = canonical.sort_values(identity, kind="stable", na_position="last")
    canonical = canonical.reset_index(drop=True)
    for column in canonical.columns:
        series = canonical[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            canonical[column] = pd.to_datetime(series, errors="raise", utc=True).dt.strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )
    digest = hashlib.sha256()
    digest.update(
        json.dumps(selected_columns, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    digest.update(b"\n")
    digest.update(
        canonical.to_csv(
            index=False,
            lineterminator="\n",
            na_rep="<NA>",
            float_format="%.17g",
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _manifest_input_value_sha256(
    predictions: pd.DataFrame,
    *,
    feature_contract_columns: list[str],
) -> str:
    identity_columns = [
        "event_date",
        "fighter_a",
        "fighter_b",
        "target",
        "fold",
        "train_end",
        "test_end",
    ]
    provenance_columns = sorted(
        column
        for column in predictions.columns
        if column.startswith(("entry_", "opening_", "model_odds_", "market_"))
        or column in {"a_implied_prob", "b_implied_prob", "diff_implied_prob"}
    )
    columns = list(
        dict.fromkeys(identity_columns + feature_contract_columns + provenance_columns)
    )
    return _canonical_frame_value_sha256(predictions, columns=columns)


def _evaluation_protocol_payload(
    *,
    bet_start_date: str,
    execution_mode: str,
    entry_offset_days: float | None,
    entry_offset_for_features: bool,
    require_entry_odds: bool,
    allow_closing_odds: bool,
    reserved_confirmation_folds: int,
    bootstrap: int,
    retrain_months: int,
    policy_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze every load-bearing evaluation setting used by this workflow."""
    policy_evaluation = dict(policy_evaluation or {})
    if policy_evaluation:
        expected_runtime = {
            "bet_start_date": str(pd.Timestamp(bet_start_date).date()),
            "execution_mode": execution_mode,
            "entry_offset_days": entry_offset_days,
            "entry_offset_for_features": bool(entry_offset_for_features),
            "require_entry_odds": bool(require_entry_odds),
            "retrain_months": int(retrain_months),
            "initial_train_years": 5,
        }
        mismatches = [
            field
            for field, actual in expected_runtime.items()
            if policy_evaluation.get(field) != actual
        ]
        if mismatches:
            raise ValueError(
                "evaluation runtime differs from the final policy: "
                + ", ".join(mismatches)
            )
    strategy_fields = (
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
    payload = {
        "bet_start_date": str(pd.Timestamp(bet_start_date).date()),
        "execution_mode": execution_mode,
        "entry_offset_days": entry_offset_days,
        "entry_offset_for_features": bool(entry_offset_for_features),
        "require_entry_odds": bool(require_entry_odds),
        "allow_closing_odds": bool(allow_closing_odds),
        "reserved_confirmation_folds": int(reserved_confirmation_folds),
        "bootstrap": int(bootstrap),
        "retrain_months": int(retrain_months),
        "initial_train_years": 5,
        "minimum_fighter_fights": 2,
        "minimum_train_rows": 100,
        "minimum_test_rows": 5,
        "initial_bankroll": float(
            policy_evaluation.get("initial_bankroll", INITIAL_BANKROLL)
        ),
        "model_seed": policy_evaluation.get("model_seed", 42),
        "odds_noise_seed": policy_evaluation.get("odds_noise_seed", 42),
        "policy_min_train_test_fights": policy_evaluation.get(
            "min_train_test_fights"
        ),
        "honest_odds_role": "entry",
        "historical_odds_selection": "closest_verified_snapshot_at_or_before_offset",
        "allowed_prefight_sources": ["line_history", "odds_api"],
        "closing_odds_fallback": False,
        "agreement_model": "xgboost_no_odds",
        "fresh_window_rule": (
            "six_calendar_months_before_selection_partition_max_event_date"
        ),
        "strategy": {
            field: policy_evaluation.get(field) for field in strategy_fields
        },
        "execution_assumptions": policy_evaluation.get("execution_assumptions"),
    }
    if payload["initial_bankroll"] != float(INITIAL_BANKROLL):
        raise ValueError("policy initial bankroll differs from evaluator runtime")
    return payload


def _load_policy_provenance(
    *,
    expected_spec_name: str | None = None,
    expected_spec_sha256: str | None = None,
) -> dict[str, Any]:
    path = (_repo_root() / SCHEDULED_REFIT_POLICY_PATH).resolve()
    if not path.is_file():
        raise ValueError(f"scheduled refit policy is missing: {path}")
    from scripts import check_production_refit_contract as contract_gate

    try:
        policy = contract_gate.load_policy(path)
    except Exception as exc:
        raise ValueError(f"scheduled refit policy is structurally invalid: {exc}") from exc
    registry_errors, _evaluation_spec, _fullfit_spec = contract_gate.validate_policy_registry(
        policy,
        repo_root=_repo_root(),
    )
    if registry_errors:
        raise ValueError(
            "scheduled refit policy/registry/method decision is not ready: "
            + "; ".join(registry_errors)
        )
    baseline = policy.get("baseline") if isinstance(policy, dict) else None
    contract = policy.get("contract") if isinstance(policy, dict) else None
    if not isinstance(baseline, dict) or not isinstance(contract, dict):
        raise ValueError("scheduled refit policy is missing baseline/contract")
    scheduled_protocol_sha256 = baseline.get("scheduled_protocol_sha256")
    if not _valid_sha256(scheduled_protocol_sha256):
        raise ValueError("scheduled refit policy has no valid scheduled protocol SHA-256")
    if expected_spec_name is not None and contract.get("evaluation_spec_name") != expected_spec_name:
        raise ValueError("named evaluation spec differs from the final policy contract")
    if expected_spec_sha256 is not None and (
        str(contract.get("evaluation_spec_payload_sha256") or "").lower()
        != expected_spec_sha256.lower()
    ):
        raise ValueError("named evaluation spec hash differs from the final policy contract")
    corrected_fights_path = (_repo_root() / FINAL_CORRECTED_FIGHTS_PATH).resolve()
    corrected_features_path = (_repo_root() / FINAL_CORRECTED_FEATURES_PATH).resolve()
    if not corrected_fights_path.is_file() or not corrected_features_path.is_file():
        raise ValueError("final corrected fights/features artifacts are missing")
    if _file_sha256(corrected_fights_path) != FINAL_CORRECTED_FIGHTS_SHA256:
        raise ValueError("final corrected fights artifact differs from the handoff contract")
    corrected_features_sha256 = _file_sha256(corrected_features_path)
    if corrected_features_sha256 != baseline.get("features_sha256"):
        raise ValueError("final corrected features artifact differs from the final policy")
    return {
        "policy_path": str(path),
        "policy_sha256": _file_sha256(path),
        "scheduled_protocol_sha256": str(scheduled_protocol_sha256).lower(),
        "policy_features_sha256": baseline.get("features_sha256"),
        "policy_processed_fights_sha256": (policy.get("root_release") or {}).get(
            "processed_fights_sha256"
        ),
        "policy_processed_features_sha256": (policy.get("root_release") or {}).get(
            "processed_features_sha256"
        ),
        "policy_evaluation": policy["evaluation"],
        "corrected_fights_path": str(corrected_fights_path),
        "corrected_fights_sha256": FINAL_CORRECTED_FIGHTS_SHA256,
        "corrected_features_path": str(corrected_features_path),
        "corrected_features_sha256": corrected_features_sha256,
        "source_dataset_fights_path": str(corrected_fights_path),
        "source_dataset_fights_sha256": FINAL_CORRECTED_FIGHTS_SHA256,
        "source_features_path": str(corrected_features_path),
        "source_features_sha256": corrected_features_sha256,
    }


def _validate_registered_challenger_spec(
    *,
    spec_name: str,
    spec_sha256: str,
    policy_contract: dict[str, Any],
) -> Any:
    """Validate a registered challenger without changing the frozen data contract.

    The final policy chooses the ordered feature/data contract.  A bounded
    experiment may vary the registered model/calibration contract, but it may
    not change the source data, temporal cutoff, T-1/noise semantics, feature
    columns, or fixed random seed selected by that policy.
    """
    try:
        candidate = resolve_named_training_spec(spec_name)
        base = resolve_named_training_spec(policy_contract["evaluation_spec_name"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"registered challenger spec cannot be resolved: {exc}") from exc
    registered_sha256 = _canonical_json_sha256(asdict(candidate))
    if not _valid_sha256(spec_sha256) or registered_sha256 != spec_sha256.lower():
        raise ValueError("challenger spec hash differs from the registered payload")
    base_sha256 = _canonical_json_sha256(asdict(base))
    if base_sha256 != policy_contract.get("evaluation_spec_payload_sha256"):
        raise ValueError("final policy evaluation spec hash is invalid")
    if list(candidate.feature_cols) != list(base.feature_cols):
        raise ValueError("challenger feature columns differ from the final policy contract")
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
        if getattr(candidate, field) != getattr(base, field)
    ]
    if changed:
        raise ValueError(
            "challenger changes frozen data/T-1/noise semantics: "
            + ", ".join(changed)
        )
    candidate_seed = (candidate.xgb_params or {}).get("random_state")
    base_seed = (base.xgb_params or {}).get("random_state")
    if candidate_seed != base_seed:
        raise ValueError("challenger changes the fixed model seed")
    return candidate


def _historical_odds_inventory_payload(
    evaluation_protocol: dict[str, Any],
) -> dict[str, Any]:
    from src.data.historical_backfill import BACKFILL_DIR

    source_paths = [
        BACKFILL_DIR / "historical_odds_pre2022_from_cleaned.csv",
        BACKFILL_DIR / "historical_odds_bfo.csv",
        *sorted(BACKFILL_DIR.glob("historical_odds_bfo_recovered_*.csv")),
        BACKFILL_DIR / "historical_odds.csv",
        *sorted((BACKFILL_DIR.parent / "line_history").glob("odds_*.csv")),
    ]
    unique_paths = sorted(
        {path.resolve() for path in source_paths if path.is_file()},
        key=lambda path: str(path).lower(),
    )
    if evaluation_protocol.get("require_entry_odds") is True and not unique_paths:
        raise ValueError("honest-odds evaluation has no immutable source files")
    entries = [
        {
            "source_file": path.name,
            "resolved_path": str(path),
            "sha256": _file_sha256(path),
        }
        for path in unique_paths
    ]
    return {
        "schema_version": 1,
        "evaluation_protocol": evaluation_protocol,
        "entries": entries,
    }


def _build_input_provenance_payload(
    *,
    dataset_fights_path: Path,
    features_artifact_path: Path,
    features_frame: pd.DataFrame,
    feature_contract_columns: list[str],
    partition_evidence: dict[str, Any],
    policy_provenance: dict[str, Any],
    evaluation_protocol: dict[str, Any],
    odds_source_inventory: dict[str, Any],
    source_fingerprint: str,
    source_inventory_sha256: str,
    source_inventory_path: Path,
    source_inventory_artifact_sha256: str,
    environment_path: Path,
    environment_artifact_sha256: str,
    environment_payload_sha256: str,
    model_spec_name: str | None,
    model_spec_payload_sha256: str | None,
) -> dict[str, Any]:
    for label, path in (
        ("dataset fights", dataset_fights_path),
        ("features artifact", features_artifact_path),
    ):
        if not path.is_file():
            raise ValueError(f"{label} artifact is missing: {path}")
    if dataset_fights_path.resolve() != Path(
        policy_provenance["corrected_fights_path"]
    ).resolve():
        raise ValueError("evaluation dataset is not the final corrected fights artifact")
    if features_artifact_path.resolve() != Path(
        policy_provenance["corrected_features_path"]
    ).resolve():
        raise ValueError("evaluation features are not the final corrected features artifact")
    if _file_sha256(dataset_fights_path) != policy_provenance[
        "corrected_fights_sha256"
    ]:
        raise ValueError("evaluation fights bytes differ from final corrected input")
    if _file_sha256(features_artifact_path) != policy_provenance[
        "corrected_features_sha256"
    ]:
        raise ValueError("evaluation feature bytes differ from final corrected input")
    contract = _ordered_feature_contract_evidence(feature_contract_columns)
    expected_contract = {
        key: partition_evidence.get(key)
        for key in (
            "feature_contract_columns",
            "feature_contract_count",
            "feature_contract_sha256",
        )
    }
    if expected_contract != contract:
        raise ValueError("partition evidence does not match the actual feature contract")
    if not _valid_sha256(source_fingerprint) or not _valid_sha256(
        source_inventory_sha256
    ):
        raise ValueError("input provenance requires the complete source inventory hash")
    if not source_inventory_path.is_file() or _file_sha256(
        source_inventory_path
    ) != source_inventory_artifact_sha256:
        raise ValueError("complete source inventory artifact is missing or changed")
    if (
        not environment_path.is_file()
        or _file_sha256(environment_path) != environment_artifact_sha256
    ):
        raise ValueError("evaluation environment artifact is missing or changed")
    environment = _read_json(environment_path)
    if _canonical_json_sha256(environment) != environment_payload_sha256:
        raise ValueError("evaluation environment payload hash is invalid")
    odds_inventory_payload_sha256 = _canonical_json_sha256(odds_source_inventory)
    odds_inventory_sha256 = _json_artifact_sha256(odds_source_inventory)
    evaluation_protocol_sha256 = _canonical_json_sha256(evaluation_protocol)
    payload = {
        "schema_version": 1,
        "dataset_fights_path": str(dataset_fights_path.resolve()),
        "dataset_fights_sha256": _file_sha256(dataset_fights_path),
        "source_dataset_fights_path": policy_provenance[
            "source_dataset_fights_path"
        ],
        "source_dataset_fights_sha256": policy_provenance[
            "source_dataset_fights_sha256"
        ],
        "features_artifact_path": str(features_artifact_path.resolve()),
        "features_artifact_sha256": _file_sha256(features_artifact_path),
        "features_value_sha256": _canonical_frame_value_sha256(features_frame),
        "source_features_path": policy_provenance["source_features_path"],
        "source_features_sha256": policy_provenance["source_features_sha256"],
        "model_spec_name": model_spec_name,
        "model_spec_payload_sha256": model_spec_payload_sha256,
        **partition_evidence,
        **contract,
        "evaluation_sample_sha256": partition_evidence["evaluation_sample_sha256"],
        "full_evaluation_sample_sha256": partition_evidence[
            "full_evaluation_sample_sha256"
        ],
        "confirmation_evaluation_sample_sha256": partition_evidence[
            "confirmation_evaluation_sample_sha256"
        ],
        "evaluation_input_value_sha256": partition_evidence[
            "evaluation_input_value_sha256"
        ],
        "full_evaluation_input_value_sha256": partition_evidence[
            "full_evaluation_input_value_sha256"
        ],
        "confirmation_evaluation_input_value_sha256": partition_evidence[
            "confirmation_evaluation_input_value_sha256"
        ],
        "selection_fold_ids": partition_evidence["selection_fold_ids"],
        "confirmation_fold_ids": partition_evidence["confirmation_fold_ids"],
        "evaluation_end_date": partition_evidence["evaluation_end_date"],
        "fresh_window_cutoff": partition_evidence["fresh_window_cutoff"],
        "odds_source_inventory_sha256": odds_inventory_sha256,
        "odds_source_inventory_payload_sha256": odds_inventory_payload_sha256,
        "evaluation_protocol": evaluation_protocol,
        "evaluation_protocol_sha256": evaluation_protocol_sha256,
        "source_fingerprint": source_fingerprint,
        "source_inventory_sha256": source_inventory_sha256,
        "source_inventory_path": str(source_inventory_path.resolve()),
        "source_inventory_artifact_sha256": source_inventory_artifact_sha256,
        "environment_path": str(environment_path.resolve()),
        "environment_artifact_sha256": environment_artifact_sha256,
        "environment_payload_sha256": environment_payload_sha256,
        **policy_provenance,
    }
    payload["input_provenance_payload_sha256"] = _canonical_json_sha256(payload)
    return payload


def _validate_input_provenance_files(
    provenance: object,
    odds_source_inventory: object,
    *,
    source_inventory_path_override: Path | None = None,
    environment_path_override: Path | None = None,
) -> pd.DataFrame:
    """Rehash every load-bearing local input before selection/confirmation."""
    if not isinstance(provenance, dict) or not isinstance(
        odds_source_inventory, dict
    ):
        raise ValueError("input provenance and odds inventory must be objects")
    unsigned = dict(provenance)
    supplied_payload_sha256 = unsigned.pop("input_provenance_payload_sha256", None)
    if supplied_payload_sha256 != _canonical_json_sha256(unsigned):
        raise ValueError("input provenance payload hash is invalid")
    dataset_path = Path(str(provenance.get("dataset_fights_path") or ""))
    features_path = Path(str(provenance.get("features_artifact_path") or ""))
    for path, field in (
        (dataset_path, "dataset_fights_sha256"),
        (features_path, "features_artifact_sha256"),
    ):
        if not path.is_file() or _file_sha256(path) != provenance.get(field):
            raise ValueError(f"input provenance artifact changed: {field}")
    features = _load_cached_frame(features_path)
    if _canonical_frame_value_sha256(features) != provenance.get(
        "features_value_sha256"
    ):
        raise ValueError("input feature values changed after selection")
    if _json_artifact_sha256(odds_source_inventory) != provenance.get(
        "odds_source_inventory_sha256"
    ):
        raise ValueError("odds source inventory differs from input provenance")
    expected_inventory = _historical_odds_inventory_payload(
        provenance["evaluation_protocol"]
    )
    if expected_inventory != odds_source_inventory:
        raise ValueError("historical odds inventory is incomplete or changed")
    current_policy = _load_policy_provenance()
    policy_payload = _read_json(Path(current_policy["policy_path"]))
    candidate_spec = _validate_registered_challenger_spec(
        spec_name=str(provenance.get("model_spec_name") or ""),
        spec_sha256=str(provenance.get("model_spec_payload_sha256") or ""),
        policy_contract=policy_payload["contract"],
    )
    contract_evidence = _ordered_feature_contract_evidence(
        list(candidate_spec.feature_cols)
    )
    for field, expected in contract_evidence.items():
        if provenance.get(field) != expected:
            raise ValueError(f"input provenance differs from challenger contract for {field}")
    for field in (
        "policy_sha256",
        "scheduled_protocol_sha256",
        "source_dataset_fights_sha256",
        "source_features_sha256",
    ):
        if provenance.get(field) != current_policy.get(field):
            raise ValueError(f"input provenance differs from final policy for {field}")
    source_inventory_path = (
        source_inventory_path_override.resolve()
        if source_inventory_path_override is not None
        else Path(str(provenance.get("source_inventory_path") or ""))
    )
    if (
        not source_inventory_path.is_file()
        or _file_sha256(source_inventory_path)
        != provenance.get("source_inventory_artifact_sha256")
    ):
        raise ValueError("evaluation source inventory artifact changed")
    source_inventory = _read_json(source_inventory_path)
    current_code = _collect_runtime_code_metadata()
    if (
        source_inventory != current_code["source_inventory"]
        or provenance.get("source_fingerprint")
        != current_code["source_fingerprint"]
        or provenance.get("source_inventory_sha256")
        != current_code["source_inventory_sha256"]
    ):
        raise ValueError("evaluation source-code inventory changed after selection")
    environment_path = (
        environment_path_override.resolve()
        if environment_path_override is not None
        else Path(str(provenance.get("environment_path") or ""))
    )
    if (
        not environment_path.is_file()
        or _file_sha256(environment_path)
        != provenance.get("environment_artifact_sha256")
    ):
        raise ValueError("evaluation environment artifact changed")
    environment = _read_json(environment_path)
    if (
        environment != current_code["environment"]
        or provenance.get("environment_payload_sha256")
        != current_code["environment_payload_sha256"]
    ):
        raise ValueError("evaluation runtime environment changed after selection")
    return features


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_normalize_json(payload), handle, indent=2, sort_keys=True)


def _write_or_validate_json(path: Path, payload: Any) -> None:
    normalized = _normalize_json(payload)
    if path.exists():
        if _read_json(path) != normalized:
            raise ValueError(f"immutable JSON artifact already differs: {path}")
        return
    _write_json(path, normalized)


def _write_json_exclusive_fsync(path: Path, payload: Any) -> None:
    """Create an immutable JSON artifact and durably flush it before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_normalize_json(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_artifact_exclusive_fsync(path: Path, payload: Any) -> None:
    """Create the canonical no-trailing-newline JSON artifact durably.

    `_json_artifact_sha256` is the byte contract used by odds-source
    inventories, so this writer intentionally matches `_write_json` exactly
    while retaining exclusive-create and fsync semantics.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _normalize_json(payload),
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_csv_exclusive_fsync(path: Path, frame: pd.DataFrame) -> None:
    """Create a byte-stable immutable CSV artifact and durably flush it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _copy_file_exclusive_fsync(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("xb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle)
        target_handle.flush()
        os.fsync(target_handle.fileno())


def _repo_relative_artifact_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(_repo_root()).as_posix()
    except ValueError as exc:
        raise ValueError(f"durable evidence path is outside the repository: {path}") from exc


def _resolve_repo_artifact_path(raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("durable evidence path must be a non-empty repo-relative path")
    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError("durable evidence paths must be repository-relative")
    resolved = (_repo_root() / path).resolve()
    try:
        resolved.relative_to(_repo_root())
    except ValueError as exc:
        raise ValueError("durable evidence path escapes the repository") from exc
    return resolved


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_csv_list(arg_value: str | None, *, default_values: list[str]) -> list[str]:
    if not arg_value:
        return list(default_values)
    return [item.strip() for item in arg_value.split(",") if item.strip()]


def _parse_stage_range(stage_arg: str | None) -> tuple[int, int]:
    if not stage_arg:
        return (1, 5)
    if "-" in stage_arg:
        start_str, end_str = stage_arg.split("-", 1)
        start = int(start_str)
        end = int(end_str)
    else:
        start = int(stage_arg)
        end = start
    if start < 1 or end > 5 or start > end:
        raise ValueError(f"Invalid stage range: {stage_arg!r}")
    return start, end


def _resolve_run_dir(*, resume: str | None) -> Path:
    """Return the run directory for a fresh run or a validated resume path."""
    if resume:
        run_dir = Path(resume)
        if not run_dir.exists() or not run_dir.is_dir():
            raise ValueError(f"Resume run directory does not exist: {run_dir}")
        return run_dir

    run_dir = EVALUATION_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _build_explicit_resume_overrides(
    args: argparse.Namespace,
    *,
    variants: list[str],
    datasets: list[str],
    families: list[str],
    run_narrow: bool,
) -> dict[str, Any]:
    """Build the explicit CLI override map used for safe resume handling."""
    return {
        "variants": variants if args.variants is not None else None,
        "datasets": datasets if args.datasets is not None else None,
        "families": families if args.families is not None else None,
        "freeze_id": args.freeze_id if args.freeze_id != DEFAULT_FREEZE_ID else None,
        "bootstrap": args.bootstrap if args.bootstrap != DEFAULT_BOOTSTRAP else None,
        "calibration_method": args.calibration_method,
        "retrain_months": (
            args.retrain_months
            if args.retrain_months != DEFAULT_RETRAIN_MONTHS
            else None
        ),
        "max_finalists": (
            args.max_finalists
            if args.max_finalists != DEFAULT_MAX_FINALISTS
            else None
        ),
        "run_narrow": run_narrow if args.no_narrow else None,
        "execution_mode": (
            getattr(args, "execution_mode", "legacy")
            if getattr(args, "execution_mode", "legacy") != "legacy"
            else None
        ),
        "entry_offset_days": getattr(args, "entry_offset_days", None),
        "entry_offset_for_features": (
            True if getattr(args, "entry_offset_for_features", False) else None
        ),
        "require_entry_odds": (
            True if getattr(args, "require_entry_odds", False) else None
        ),
        "allow_closing_odds": (
            True if getattr(args, "allow_closing_odds", False) else None
        ),
    }


def _resolve_variant_calibration(
    variant_name: str,
    calibration_method: str | None,
) -> str:
    """Resolve the actual calibration method for a variant/cell."""
    if calibration_method in {None, "", "variant_default"}:
        return ALL_VARIANTS[variant_name]().calibration_method
    return calibration_method


def _variant_needs_betsapi(variant_name: str) -> bool:
    """Return whether a model variant depends on BetsAPI MMA augmentation."""
    if variant_name not in ALL_VARIANTS:
        return False
    variant = ALL_VARIANTS[variant_name]()
    builder_name = (
        variant.feature_builder_fn.__name__
        if variant.feature_builder_fn is not None
        else ""
    )
    return "betsapi" in variant_name or "betsapi" in builder_name


def _has_saved_betsapi_mma_backfill(
    raw_root: Path = BETSAPI_MMA_RAW_DIR,
    processed_root: Path = BETSAPI_MMA_PROCESSED_DIR,
) -> bool:
    """Return True when the local BetsAPI MMA backfill is complete and usable."""
    return bool(
        summarize_saved_betsapi_mma_backfill(
            raw_root=raw_root,
            processed_root=processed_root,
        )["ready"]
    )


def _default_variant_names(*, has_betsapi_backfill: bool) -> list[str]:
    """Return the default stage-1 variants for the historical evaluation matrix."""
    variants = [
        name for name in sorted(ALL_VARIANTS)
        if name not in HISTORICAL_MATRIX_EXCLUDED_VARIANTS
    ]
    if has_betsapi_backfill:
        return variants
    return [name for name in variants if not _variant_needs_betsapi(name)]


def _default_feature_families(*, has_betsapi_backfill: bool) -> list[str]:
    """Return the default stage-1 families for the historical evaluation matrix."""
    families = [
        family for family in ALL_FEATURE_FAMILIES
        if family not in HISTORICAL_MATRIX_EXCLUDED_FAMILIES
    ]
    if has_betsapi_backfill:
        return families
    return [family for family in families if not _family_needs_betsapi(family)]


def _ensure_historical_matrix_scope(
    *,
    variants: list[str],
    families: list[str],
) -> None:
    """Block live snapshot-dependent BetsAPI scope from the historical matrix."""
    blocked_variants = sorted(set(variants) & set(HISTORICAL_MATRIX_EXCLUDED_VARIANTS))
    blocked_families = sorted(set(families) & set(HISTORICAL_MATRIX_EXCLUDED_FAMILIES))
    if not blocked_variants and not blocked_families:
        return

    requested: list[str] = []
    if blocked_variants:
        requested.append(f"variants={blocked_variants}")
    if blocked_families:
        requested.append(f"families={blocked_families}")

    raise ValueError(
        "The UFC evaluation matrix is historical-only and excludes live "
        "snapshot-dependent BetsAPI scope sourced from bet365/upcoming. "
        f"Remove these requests and rerun: {', '.join(requested)}."
    )


def _requested_families_need_live_snapshot_features(
    feature_families: list[str],
) -> bool:
    """Return whether the requested BetsAPI families require live snapshot features."""
    return bool(set(feature_families) & set(HISTORICAL_MATRIX_EXCLUDED_FAMILIES))


def _ensure_betsapi_backfill_available(
    *,
    variants: list[str],
    families: list[str],
    has_betsapi_backfill: bool,
) -> None:
    """Fail fast when BetsAPI-labeled cells are requested without local backfill."""
    if has_betsapi_backfill:
        return

    betsapi_variants = [name for name in variants if _variant_needs_betsapi(name)]
    betsapi_families = [name for name in families if _family_needs_betsapi(name)]
    if not betsapi_variants and not betsapi_families:
        return

    requested: list[str] = []
    if betsapi_variants:
        requested.append(f"variants={betsapi_variants}")
    if betsapi_families:
        requested.append(f"families={betsapi_families}")
    raise ValueError(
        "BetsAPI MMA backfill is not available locally, but BetsAPI-labeled "
        f"cells were requested ({'; '.join(requested)}). "
        "Populate data/raw/betsapi/mma first or run without BetsAPI variants/families."
    )


def _variant_profile_key(variant_name: str) -> str:
    variant = ALL_VARIANTS[variant_name]()
    builder_name = variant.feature_builder_fn.__name__ if variant.feature_builder_fn is not None else "build_features"
    parts = [builder_name]
    if variant.add_rematch_features:
        parts.append("rematch_features")
    return "__".join(parts)


def _selected_profile_variants(variant_names: list[str]) -> dict[str, str]:
    representatives: dict[str, str] = {}
    for variant_name in variant_names:
        profile_key = _variant_profile_key(variant_name)
        representatives.setdefault(profile_key, variant_name)
    return representatives


def _family_needs_betsapi(feature_family: str) -> bool:
    return feature_family in BETSAPI_FEATURE_FAMILIES


def _family_added_betsapi_columns(
    features_df: pd.DataFrame,
    feature_family: str,
) -> list[str]:
    """Return the BetsAPI columns uniquely added by a family."""
    if feature_family == "production_betsapi":
        source_cols = BETSAPI_HISTORICAL_FEATURE_NAMES
    elif feature_family in {"production_betsapi_expanded", "ufcstats_betsapi_expanded"}:
        source_cols = BETSAPI_CHALLENGER_FEATURE_NAMES
    else:
        return []
    return [column for column in source_cols if column in features_df.columns]


def _validate_requested_betsapi_families_have_signal(
    features_df: pd.DataFrame,
    *,
    requested_families: list[str],
    dataset_variant: str,
    profile_key: str,
) -> None:
    """Fail fast when a requested BetsAPI family would be placeholder-shaped or invalid."""
    failures: list[str] = []
    for feature_family in requested_families:
        added_cols = _family_added_betsapi_columns(features_df, feature_family)
        if not added_cols:
            failures.append(f"{feature_family}: no BetsAPI feature columns were present")
            continue
        if not features_df[added_cols].notna().any().any():
            failures.append(
                f"{feature_family}: all {len(added_cols)} added BetsAPI columns are null"
            )
            continue

        implied_prob_cols = [
            column for column in added_cols
            if "implied_prob" in column
            and "diff_" not in column
            and "_move_" not in column
        ]
        invalid_prob_cols: list[str] = []
        for column in implied_prob_cols:
            series = pd.to_numeric(features_df[column], errors="coerce")
            invalid_mask = series.notna() & ((series < 0.0) | (series > 1.0))
            if invalid_mask.any():
                invalid_prob_cols.append(column)
        if invalid_prob_cols:
            failures.append(
                f"{feature_family}: implied-probability columns out of range [0, 1] "
                f"({', '.join(invalid_prob_cols)})"
            )

    if failures:
        raise ValueError(
            "Requested BetsAPI feature families are not usable for "
            f"dataset={dataset_variant} profile={profile_key}: {'; '.join(failures)}"
        )


def _seed_from_key(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if n_bootstrap <= 0 or len(y_true) < 2:
        return {}

    rng = np.random.RandomState(seed)
    metrics: dict[str, list[float]] = {
        "accuracy": [],
        "brier": [],
        "log_loss": [],
        "ece": [],
    }

    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        boot_y = y_true[idx]
        boot_p = y_prob[idx]
        metrics["accuracy"].append(float(accuracy_score(boot_y, (boot_p > 0.5).astype(int))))
        metrics["brier"].append(float(brier_score_loss(boot_y, boot_p)))
        metrics["log_loss"].append(float(log_loss(boot_y, boot_p, labels=[0, 1])))
        metrics["ece"].append(float(compute_ece(boot_y, boot_p)))

    return {
        metric_name: {
            "ci_lo": float(np.percentile(values, 2.5)),
            "ci_hi": float(np.percentile(values, 97.5)),
        }
        for metric_name, values in metrics.items()
    }


def _compute_slice_metrics(slice_df: pd.DataFrame) -> dict[str, Any]:
    if slice_df.empty:
        return {"n_samples": 0}

    y_true = slice_df["target"].astype(int).to_numpy()
    y_prob = slice_df["prob_a"].astype(float).to_numpy()
    return {
        "n_samples": int(len(slice_df)),
        "accuracy": float(accuracy_score(y_true, (y_prob > 0.5).astype(int))),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "ece": float(compute_ece(y_true, y_prob)),
    }


def _betsapi_coverage_mask(
    predictions_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    feature_family: str | None = None,
) -> pd.Series:
    feature_names = (
        BETSAPI_HISTORICAL_FEATURE_NAMES
        if feature_family == "production_betsapi"
        else BETSAPI_CHALLENGER_FEATURE_NAMES
    )
    betsapi_cols = [
        column for column in feature_names
        if column in features_df.columns
    ]
    if not betsapi_cols or predictions_df.empty:
        return pd.Series(False, index=predictions_df.index)

    coverage_df = features_df[
        ["event_date", "fighter_a", "fighter_b"] + betsapi_cols
    ].copy()
    coverage_df["event_date"] = pd.to_datetime(coverage_df["event_date"], errors="coerce")
    coverage_df["_has_betsapi"] = coverage_df[betsapi_cols].notna().any(axis=1)
    coverage_df = coverage_df.drop_duplicates(subset=["event_date", "fighter_a", "fighter_b"])

    prediction_keys = predictions_df[["event_date", "fighter_a", "fighter_b"]].copy()
    prediction_keys["event_date"] = pd.to_datetime(
        prediction_keys["event_date"], errors="coerce"
    )

    merged = prediction_keys.merge(
        coverage_df[["event_date", "fighter_a", "fighter_b", "_has_betsapi"]],
        on=["event_date", "fighter_a", "fighter_b"],
        how="left",
    )
    mask = merged["_has_betsapi"].fillna(False).astype(bool)
    mask.index = predictions_df.index
    return mask


def _compute_sliced_metrics(
    predictions_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    feature_family: str | None = None,
    fresh_window_cutoff: pd.Timestamp | None = None,
) -> dict[str, Any]:
    if predictions_df.empty:
        return {"by_year": {}, "fresh_window": {"n_samples": 0}, "by_coverage": {}}

    preds = predictions_df.copy()
    preds["event_date"] = pd.to_datetime(preds["event_date"], errors="coerce")

    by_year: dict[str, Any] = {}
    for year in sorted(preds["event_date"].dt.year.dropna().astype(int).unique()):
        year_slice = preds[preds["event_date"].dt.year == year]
        by_year[str(year)] = _compute_slice_metrics(year_slice)

    if fresh_window_cutoff is None:
        anchor = preds["event_date"].max()
        if pd.isna(anchor):
            raise ValueError("fresh-window anchor cannot be derived from evaluation rows")
        fresh_window_cutoff = pd.Timestamp(anchor).normalize() - pd.DateOffset(months=6)
    fresh_window = _compute_slice_metrics(
        preds[preds["event_date"] >= pd.Timestamp(fresh_window_cutoff)]
    )

    coverage_mask = _betsapi_coverage_mask(preds, features_df, feature_family=feature_family)
    if len(coverage_mask) != len(preds):
        raise ValueError("BetsAPI coverage mask length does not match predictions")

    aligned_mask = coverage_mask.fillna(False).astype(bool).to_numpy()
    if aligned_mask.any():
        by_coverage = {
            "betsapi_high_coverage": _compute_slice_metrics(preds.loc[aligned_mask]),
            "betsapi_low_coverage": _compute_slice_metrics(preds.loc[~aligned_mask]),
        }
    else:
        by_coverage = {
            "all_data": _compute_slice_metrics(preds),
        }

    return {
        "by_year": by_year,
        "fresh_window": fresh_window,
        "by_coverage": by_coverage,
    }


def _compute_model_metrics(
    predictions_df: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    y_true = predictions_df["target"].astype(int).to_numpy()
    y_prob = predictions_df["prob_a"].astype(float).to_numpy()
    metrics = {
        "accuracy": float(accuracy_score(y_true, (y_prob > 0.5).astype(int))),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "ece": float(compute_ece(y_true, y_prob)),
    }
    bootstrap_ci = _compute_bootstrap_ci(
        y_true,
        y_prob,
        n_bootstrap=bootstrap,
        seed=seed,
    )
    if bootstrap_ci:
        metrics["bootstrap_ci"] = bootstrap_ci

    # Name the cached feature column by its actual role. Evaluation preparation
    # replaces it with verified pre-fight input; calling it "closing" was both
    # misleading and encouraged look-ahead use.
    for label, col in [("model_input_odds", "a_implied_prob"), ("opening_odds", "a_opening_implied_prob")]:
        if col in predictions_df.columns:
            odds_prob = pd.to_numeric(predictions_df[col], errors="coerce")
            valid = odds_prob.notna()
            if valid.sum() >= 10:
                y_sub = y_true[valid.to_numpy()]
                p_sub = odds_prob[valid].to_numpy()
                metrics[f"{label}_baseline"] = {
                    "n": int(valid.sum()),
                    "accuracy": float(accuracy_score(y_sub, (p_sub > 0.5).astype(int))),
                    "brier": float(brier_score_loss(y_sub, p_sub)),
                    "log_loss": float(log_loss(y_sub, p_sub, labels=[0, 1])),
                }

    return metrics


def _evaluate_cell_worker(task: dict[str, Any]) -> dict[str, Any]:
    if "reserved_confirmation_folds" not in task:
        raise ValueError(
            "Stage 1 task is missing the reserved_confirmation_folds contract"
        )
    raw_reserved_confirmation_folds = task["reserved_confirmation_folds"]
    if (
        isinstance(raw_reserved_confirmation_folds, bool)
        or not isinstance(raw_reserved_confirmation_folds, int)
        or raw_reserved_confirmation_folds < 0
    ):
        raise ValueError(
            "Stage 1 reserved_confirmation_folds must be a non-negative integer"
        )
    reserved_confirmation_folds = raw_reserved_confirmation_folds

    cell = CellSpec(**task["cell"])
    features_path = Path(task["features_path"])
    dataset_fights_path = Path(task.get("dataset_fights_path", features_path))
    cached_features_df = _load_cached_frame(features_path)
    features_df = cached_features_df.copy()

    model_spec_name = task.get("model_spec_name")
    model_spec_payload_sha256 = task.get("model_spec_payload_sha256")
    if model_spec_name:
        named_spec = resolve_named_training_spec(model_spec_name)
        actual_spec_hash = _canonical_json_sha256(asdict(named_spec))
        if actual_spec_hash != model_spec_payload_sha256:
            raise ValueError("Stage 1 named model spec hash differs from the registry")
        variant = variant_from_named_training_spec(
            model_spec_name,
            variant_name=cell.model_variant,
        )
        if cell.calibration_method != named_spec.calibration_method:
            raise ValueError("Stage 1 calibration differs from the named model spec")
    else:
        variant = ALL_VARIANTS[cell.model_variant]()
        variant.calibration_method = cell.calibration_method

    # Stage 3 materializes the promoted contract transforms before fold
    # generation; stage 1 must do the same or the two paths can train on
    # different column sets (a no-op when the columns already exist).
    from src.strategy.model_lab import _materialize_variant_contract_features

    features_df = _materialize_variant_contract_features(features_df, variant)
    feature_contract_columns, _no_odds_columns = resolve_variant_feature_columns(
        features_df,
        variant,
        feature_family=(None if model_spec_name else cell.feature_family),
        feature_cols=(list(named_spec.feature_cols) if model_spec_name else None),
    )

    generated = generate_variant_fold_predictions(
        features_df,
        variant,
        retrain_months=cell.retrain_months,
        initial_train_years=5,
        bet_start_date=TRAIN_CUTOFF_DATE,
        feature_family=cell.feature_family,
        feature_cols=feature_contract_columns,
        entry_offset_days=task.get("entry_offset_days"),
        entry_offset_for_features=bool(task.get("entry_offset_for_features", False)),
        require_entry_odds=bool(task.get("require_entry_odds", False)),
        allow_closing_odds=bool(task.get("allow_closing_odds", False)),
        evaluation_partition=(
            "selection" if reserved_confirmation_folds else "all"
        ),
        confirmation_fold_count=(
            reserved_confirmation_folds or DEFAULT_CONFIRMATION_FOLD_COUNT
        ),
        return_fold_manifest=True,
        allow_all_folds=not bool(reserved_confirmation_folds),
    )
    selection_folds, fold_manifest = generated
    partition_evidence = _fold_partition_evidence(
        fold_manifest,
        selection_folds,
        reserved_confirmation_folds=reserved_confirmation_folds,
        feature_contract_columns=feature_contract_columns,
    )
    predictions_df = _post_cutoff_predictions(selection_folds)
    if predictions_df.empty:
        raise ValueError(f"No walk-forward predictions produced for {cell.key}")

    seed = _seed_from_key(cell.key)
    metrics = _compute_model_metrics(
        predictions_df,
        bootstrap=cell.bootstrap,
        seed=seed,
    )
    prediction_rows_sha256 = _prediction_rows_sha256(predictions_df)
    prediction_values_sha256 = _prediction_values_sha256(predictions_df)
    metrics.update(
        {
            "evaluation_input_value_sha256": partition_evidence.get(
                "evaluation_input_value_sha256"
            ),
            "prediction_rows_sha256": prediction_rows_sha256,
            "prediction_values_sha256": prediction_values_sha256,
            "model_spec_name": model_spec_name,
            "model_spec_payload_sha256": model_spec_payload_sha256,
            "calibration_method": cell.calibration_method,
        }
    )
    sliced_metrics = _compute_sliced_metrics(
        predictions_df,
        features_df,
        feature_family=cell.feature_family,
        fresh_window_cutoff=pd.Timestamp(partition_evidence["fresh_window_cutoff"]),
    )
    evaluation_protocol = dict(
        task.get(
            "evaluation_protocol",
            _evaluation_protocol_payload(
                bet_start_date=TRAIN_CUTOFF_DATE,
                execution_mode="legacy",
                entry_offset_days=task.get("entry_offset_days"),
                entry_offset_for_features=bool(
                    task.get("entry_offset_for_features", False)
                ),
                require_entry_odds=bool(task.get("require_entry_odds", False)),
                allow_closing_odds=bool(task.get("allow_closing_odds", False)),
                reserved_confirmation_folds=reserved_confirmation_folds,
                bootstrap=cell.bootstrap,
                retrain_months=cell.retrain_months,
            ),
        )
    )
    input_provenance = None
    if reserved_confirmation_folds:
        input_provenance = _build_input_provenance_payload(
            dataset_fights_path=dataset_fights_path,
            features_artifact_path=features_path,
            features_frame=cached_features_df,
            feature_contract_columns=feature_contract_columns,
            partition_evidence=partition_evidence,
            policy_provenance=dict(task["policy_provenance"]),
            evaluation_protocol=evaluation_protocol,
            odds_source_inventory=dict(task["odds_source_inventory"]),
            source_fingerprint=str(task["source_fingerprint"]),
            source_inventory_sha256=str(task["source_inventory_sha256"]),
            source_inventory_path=Path(task["source_inventory_path"]),
            source_inventory_artifact_sha256=str(
                task["source_inventory_artifact_sha256"]
            ),
            environment_path=Path(task["environment_path"]),
            environment_artifact_sha256=str(
                task["environment_artifact_sha256"]
            ),
            environment_payload_sha256=str(task["environment_payload_sha256"]),
            model_spec_name=model_spec_name,
            model_spec_payload_sha256=model_spec_payload_sha256,
        )

    return {
        "status": "success",
        "cell_key": cell.key,
        "model_variant": cell.model_variant,
        "dataset_variant": cell.dataset_variant,
        "feature_family": cell.feature_family,
        "calibration_method": cell.calibration_method,
        "retrain_months": cell.retrain_months,
        "bootstrap": cell.bootstrap,
        **partition_evidence,
        "evaluation_protocol": evaluation_protocol,
        "model_spec_name": model_spec_name,
        "model_spec_payload_sha256": model_spec_payload_sha256,
        "prediction_rows_sha256": prediction_rows_sha256,
        "prediction_values_sha256": prediction_values_sha256,
        **(
            {
                "input_provenance": input_provenance,
                "input_provenance_payload_sha256": input_provenance[
                    "input_provenance_payload_sha256"
                ],
                "input_provenance_sha256": _json_artifact_sha256(input_provenance),
                "odds_source_inventory": task["odds_source_inventory"],
                "odds_source_inventory_sha256": input_provenance[
                    "odds_source_inventory_sha256"
                ],
                **{
                    field: input_provenance[field]
                    for field in (
                        "dataset_fights_sha256",
                        "dataset_fights_path",
                        "source_dataset_fights_path",
                        "source_dataset_fights_sha256",
                        "features_artifact_sha256",
                        "features_artifact_path",
                        "features_value_sha256",
                        "source_features_path",
                        "source_features_sha256",
                        "policy_sha256",
                        "scheduled_protocol_sha256",
                        "evaluation_protocol_sha256",
                        "source_inventory_sha256",
                        "source_inventory_path",
                        "source_inventory_artifact_sha256",
                        "environment_path",
                        "environment_artifact_sha256",
                        "environment_payload_sha256",
                    )
                },
            }
            if input_provenance is not None
            else {}
        ),
        "metrics": metrics,
        "sliced_metrics": sliced_metrics,
        "source_fingerprint": task.get("source_fingerprint"),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _cell_summary_row(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics", {})
    bootstrap_ci = metrics.get("bootstrap_ci", {})
    brier_ci = bootstrap_ci.get("brier", {})
    log_loss_ci = bootstrap_ci.get("log_loss", {})
    return {
        "cell_key": result["cell_key"],
        "model_variant": result["model_variant"],
        "dataset_variant": result["dataset_variant"],
        "feature_family": result["feature_family"],
        "calibration_method": result["calibration_method"],
        "evaluation_partition": result.get("evaluation_partition", "all"),
        "reserved_confirmation_folds": result.get(
            "reserved_confirmation_folds", 0
        ),
        "n_folds": result.get("n_folds", 0),
        "n_predictions": result.get("n_predictions", 0),
        "full_evaluation_n_folds": result.get("full_evaluation_n_folds"),
        "full_evaluation_n_fights": result.get("full_evaluation_n_fights"),
        "accuracy": metrics.get("accuracy"),
        "brier": metrics.get("brier"),
        "log_loss": metrics.get("log_loss"),
        "ece": metrics.get("ece"),
        "brier_ci_lo": brier_ci.get("ci_lo"),
        "brier_ci_hi": brier_ci.get("ci_hi"),
        "log_loss_ci_lo": log_loss_ci.get("ci_lo"),
        "log_loss_ci_hi": log_loss_ci.get("ci_hi"),
        "model_input_odds_baseline_accuracy": (metrics.get("model_input_odds_baseline") or {}).get("accuracy"),
        "model_input_odds_baseline_brier": (metrics.get("model_input_odds_baseline") or {}).get("brier"),
        "model_input_odds_baseline_log_loss": (metrics.get("model_input_odds_baseline") or {}).get("log_loss"),
        "model_input_odds_baseline_n": (metrics.get("model_input_odds_baseline") or {}).get("n"),
        "opening_odds_baseline_accuracy": (metrics.get("opening_odds_baseline") or {}).get("accuracy"),
        "opening_odds_baseline_brier": (metrics.get("opening_odds_baseline") or {}).get("brier"),
        "opening_odds_baseline_log_loss": (metrics.get("opening_odds_baseline") or {}).get("log_loss"),
        "opening_odds_baseline_n": (metrics.get("opening_odds_baseline") or {}).get("n"),
    }


def _load_checkpoint(run_dir: Path) -> dict[str, Any]:
    checkpoint_path = run_dir / "checkpoint.json"
    if checkpoint_path.exists():
        return _read_json(checkpoint_path)
    return {"completed_cells": [], "updated_at": None}


def _save_checkpoint(run_dir: Path, completed_cells: set[str]) -> None:
    _write_json(
        run_dir / "checkpoint.json",
        {
            "completed_cells": sorted(completed_cells),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def _load_dataset_variants(dataset_names: list[str]) -> dict[str, pd.DataFrame]:
    legacy_df = load_kaggle_dataset(RAW_DATA_DIR / "ufc-master.csv")
    all_variants = build_training_dataset_variants(legacy_df=legacy_df)
    missing = sorted(set(dataset_names) - set(all_variants))
    if missing:
        raise ValueError(
            f"Unknown dataset variants: {missing}. Available: {list(all_variants.keys())}"
        )
    return {name: all_variants[name] for name in dataset_names}


def _prepare_feature_caches(
    *,
    run_dir: Path,
    dataset_names: list[str],
    variant_names: list[str],
    feature_families: list[str],
    cache_format: str,
) -> dict[tuple[str, str, bool], str]:
    raw_datasets = _load_dataset_variants(dataset_names)
    profile_representatives = _selected_profile_variants(variant_names)
    needs_any_betsapi = any(_family_needs_betsapi(family) for family in feature_families)
    requested_betsapi_families = [
        family for family in feature_families if _family_needs_betsapi(family)
    ]
    include_live_snapshot_features = _requested_families_need_live_snapshot_features(
        requested_betsapi_families,
    )

    cache_index: dict[tuple[str, str, bool], str] = {}

    for dataset_variant, training_df in raw_datasets.items():
        dataset_path = _dataset_cache_path(
            run_dir,
            dataset_variant=dataset_variant,
            cache_format=cache_format,
        )
        if not dataset_path.exists():
            _save_cached_frame(training_df, dataset_path, cache_format)
        for profile_key, representative_name in profile_representatives.items():
            variant = ALL_VARIANTS[representative_name]()
            base_path = _feature_cache_path(
                run_dir,
                dataset_variant=dataset_variant,
                profile_key=profile_key,
                include_betsapi=False,
                cache_format=cache_format,
            )

            if not base_path.exists():
                logger.info(
                    "Building features for dataset=%s profile=%s (variant=%s)",
                    dataset_variant,
                    profile_key,
                    representative_name,
                )
                base_features = build_variant_features(
                    training_df,
                    variant,
                    save_artifacts=False,
                )
                _save_cached_frame(base_features, base_path, cache_format)
            else:
                logger.info(
                    "Reusing cached features for dataset=%s profile=%s",
                    dataset_variant,
                    profile_key,
                )
            cache_index[(dataset_variant, profile_key, False)] = str(base_path)

            if not needs_any_betsapi:
                continue

            betsapi_path = _feature_cache_path(
                run_dir,
                dataset_variant=dataset_variant,
                profile_key=profile_key,
                include_betsapi=True,
                cache_format=cache_format,
            )
            if not betsapi_path.exists():
                base_features = _load_cached_frame(base_path)
                if any(column.startswith("betsapi_") for column in base_features.columns):
                    betsapi_features = base_features
                else:
                    logger.info(
                        "Augmenting BetsAPI features for dataset=%s profile=%s",
                        dataset_variant,
                        profile_key,
                    )
                    betsapi_features = augment_features_with_betsapi_mma(
                        base_features,
                        save_artifacts=False,
                        include_live_snapshot_features=include_live_snapshot_features,
                    )
            else:
                betsapi_features = _load_cached_frame(betsapi_path)

            _validate_requested_betsapi_families_have_signal(
                betsapi_features,
                requested_families=requested_betsapi_families,
                dataset_variant=dataset_variant,
                profile_key=profile_key,
            )
            if not betsapi_path.exists():
                _save_cached_frame(betsapi_features, betsapi_path, cache_format)
            cache_index[(dataset_variant, profile_key, True)] = str(betsapi_path)

    return cache_index


def validate_betsapi_matrix_runtime_readiness(
    *,
    dataset_names: list[str],
    variant_names: list[str],
    feature_families: list[str],
    cache_format: str | None = None,
) -> dict[str, Any]:
    """Run the same BetsAPI family signal checks used at stage 1 without side effects."""
    requested_betsapi_families = [
        family for family in feature_families if _family_needs_betsapi(family)
    ]
    representative_profiles = _selected_profile_variants(variant_names)
    result = {
        "ready": True,
        "checked": bool(requested_betsapi_families),
        "datasets": list(dataset_names),
        "families": requested_betsapi_families,
        "representative_profiles": representative_profiles,
    }
    if not requested_betsapi_families:
        return result

    resolved_cache_format = cache_format or _feature_cache_format()
    temp_dir = Path(tempfile.mkdtemp(prefix="ufc_betsapi_preflight_"))
    try:
        _prepare_feature_caches(
            run_dir=temp_dir,
            dataset_names=list(dataset_names),
            variant_names=list(variant_names),
            feature_families=requested_betsapi_families,
            cache_format=resolved_cache_format,
        )
    except Exception as exc:
        result["ready"] = False
        result["error"] = str(exc)
        return result
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return result


def validate_betsapi_preflight_readiness(
    *,
    dataset_names: list[str],
    feature_families: list[str],
    raw_root: Path = BETSAPI_MMA_RAW_DIR,
) -> dict[str, Any]:
    """Run a lightweight dataset-level BetsAPI readiness check for preflight.

    This intentionally avoids full feature-profile builds. The exact stage-1
    runtime guard still validates the real requested profiles during matrix
    execution; preflight only needs to prove that each dataset variant can join
    to usable BetsAPI signal with the current local artifacts.
    """
    requested_betsapi_families = [
        family for family in feature_families if _family_needs_betsapi(family)
    ]
    include_live_snapshot_features = _requested_families_need_live_snapshot_features(
        requested_betsapi_families,
    )
    result = {
        "ready": True,
        "checked": bool(requested_betsapi_families),
        "datasets": list(dataset_names),
        "families": requested_betsapi_families,
        "mode": "dataset_preflight",
        "representative_profiles": {"dataset_preflight": "lightweight_join"},
    }
    if not requested_betsapi_families:
        return result

    raw_datasets = _load_dataset_variants(dataset_names)
    try:
        for dataset_variant, training_df in raw_datasets.items():
            minimal = training_df[["event_date", "fighter_a", "fighter_b"]].copy()
            for column in ("a_implied_prob", "b_implied_prob", "diff_implied_prob"):
                minimal[column] = (
                    training_df[column]
                    if column in training_df.columns
                    else np.nan
                )
            betsapi_features = augment_features_with_betsapi_mma(
                minimal,
                raw_root=raw_root,
                save_artifacts=False,
                include_live_snapshot_features=include_live_snapshot_features,
            )
            _validate_requested_betsapi_families_have_signal(
                betsapi_features,
                requested_families=requested_betsapi_families,
                dataset_variant=dataset_variant,
                profile_key="dataset_preflight",
            )
    except Exception as exc:
        result["ready"] = False
        result["error"] = str(exc)
        return result

    return result


def _run_stage1(
    *,
    run_dir: Path,
    cells: list[CellSpec],
    max_workers: int,
    cache_format: str,
    current_source_fingerprint: str | None = None,
    entry_offset_days: float | None = None,
    entry_offset_for_features: bool = False,
    require_entry_odds: bool = False,
    allow_closing_odds: bool = False,
    execution_mode: str = "legacy",
    model_spec_name: str | None = None,
    model_spec_payload_sha256: str | None = None,
    reserved_confirmation_folds: int = 0,
    policy_provenance: dict[str, Any] | None = None,
    odds_source_inventory: dict[str, Any] | None = None,
    source_inventory_sha256: str | None = None,
) -> list[dict[str, Any]]:
    resolved_policy_provenance = (
        policy_provenance
        or (
            _load_policy_provenance(
                expected_spec_name=model_spec_name,
                expected_spec_sha256=model_spec_payload_sha256,
            )
            if reserved_confirmation_folds
            else {}
        )
    )
    evaluation_protocol = _evaluation_protocol_payload(
        bet_start_date=TRAIN_CUTOFF_DATE,
        execution_mode=execution_mode,
        entry_offset_days=entry_offset_days,
        entry_offset_for_features=entry_offset_for_features,
        require_entry_odds=require_entry_odds,
        allow_closing_odds=allow_closing_odds,
        reserved_confirmation_folds=reserved_confirmation_folds,
        bootstrap=cells[0].bootstrap if cells else DEFAULT_BOOTSTRAP,
        retrain_months=cells[0].retrain_months if cells else DEFAULT_RETRAIN_MONTHS,
        policy_evaluation=resolved_policy_provenance.get("policy_evaluation"),
    )
    resolved_odds_inventory = (
        odds_source_inventory
        or (
            _historical_odds_inventory_payload(evaluation_protocol)
            if reserved_confirmation_folds
            else {}
        )
    )
    resolved_source_inventory_sha256 = (
        source_inventory_sha256 or current_source_fingerprint
    )
    source_inventory_path = run_dir / SOURCE_INVENTORY_FILENAME
    source_inventory_artifact_sha256 = (
        _file_sha256(source_inventory_path)
        if source_inventory_path.is_file()
        else None
    )
    environment_path = run_dir / ENVIRONMENT_INVENTORY_FILENAME
    environment_artifact_sha256 = (
        _file_sha256(environment_path) if environment_path.is_file() else None
    )
    environment_payload_sha256 = (
        _canonical_json_sha256(_read_json(environment_path))
        if environment_path.is_file()
        else None
    )
    logger.info("Stage 1: resolving immutable feature inputs")
    if reserved_confirmation_folds:
        corrected_features_path = str(
            Path(resolved_policy_provenance["corrected_features_path"])
        )
        cache_index = {
            (
                cell.dataset_variant,
                _variant_profile_key(cell.model_variant),
                _family_needs_betsapi(cell.feature_family),
            ): corrected_features_path
            for cell in cells
        }
    else:
        cache_index = _prepare_feature_caches(
            run_dir=run_dir,
            dataset_names=sorted({cell.dataset_variant for cell in cells}),
            variant_names=sorted({cell.model_variant for cell in cells}),
            feature_families=sorted({cell.feature_family for cell in cells}),
            cache_format=cache_format,
        )

    checkpoint = _load_checkpoint(run_dir)
    completed_cells = set(checkpoint.get("completed_cells", []))
    existing_success_keys: set[str] = set()

    def task_for_cell(cell: CellSpec) -> dict[str, Any]:
        profile_key = _variant_profile_key(cell.model_variant)
        features_path = cache_index[
            (
                cell.dataset_variant,
                profile_key,
                _family_needs_betsapi(cell.feature_family),
            )
        ]
        return {
            "cell": asdict(cell),
            "features_path": features_path,
            "dataset_fights_path": (
                str(resolved_policy_provenance["corrected_fights_path"])
                if reserved_confirmation_folds
                else str(
                    _dataset_cache_path(
                        run_dir,
                        dataset_variant=cell.dataset_variant,
                        cache_format=cache_format,
                    )
                )
            ),
            "source_fingerprint": current_source_fingerprint,
            "source_inventory_sha256": resolved_source_inventory_sha256,
            "source_inventory_path": str(source_inventory_path),
            "source_inventory_artifact_sha256": source_inventory_artifact_sha256,
            "environment_path": str(environment_path),
            "environment_artifact_sha256": environment_artifact_sha256,
            "environment_payload_sha256": environment_payload_sha256,
            "policy_provenance": resolved_policy_provenance,
            "odds_source_inventory": resolved_odds_inventory,
            "evaluation_protocol": {
                **evaluation_protocol,
                "retrain_months": cell.retrain_months,
                "bootstrap": cell.bootstrap,
            },
            "entry_offset_days": entry_offset_days,
            "entry_offset_for_features": entry_offset_for_features,
            "require_entry_odds": require_entry_odds,
            "allow_closing_odds": allow_closing_odds,
            "model_spec_name": model_spec_name,
            "model_spec_payload_sha256": model_spec_payload_sha256,
            "reserved_confirmation_folds": reserved_confirmation_folds,
        }

    results: list[dict[str, Any]] = []
    for cell in cells:
        output_path = _cell_output_path(run_dir, cell)
        if output_path.exists():
            payload = _read_json(output_path)
            if (
                payload.get("status") == "success"
                and payload.get("source_fingerprint") == current_source_fingerprint
                and int(payload.get("reserved_confirmation_folds", 0))
                == reserved_confirmation_folds
                and _valid_sha256(payload.get("input_provenance_payload_sha256"))
                and payload.get("source_inventory_sha256")
                == resolved_source_inventory_sha256
                and payload.get("policy_sha256")
                == resolved_policy_provenance.get("policy_sha256")
                and (
                    model_spec_payload_sha256 is None
                    or payload.get("model_spec_payload_sha256")
                    == model_spec_payload_sha256
                )
            ):
                results.append(payload)
                existing_success_keys.add(cell.key)

    completed_cells.update(existing_success_keys)
    pending_cells = [cell for cell in cells if cell.key not in existing_success_keys]
    logger.info(
        "Stage 1: %d total cells, %d already complete, %d pending",
        len(cells),
        len(existing_success_keys),
        len(pending_cells),
    )

    if pending_cells:
        if max_workers <= 1:
            for cell in pending_cells:
                try:
                    payload = _evaluate_cell_worker(task_for_cell(cell))
                except Exception as exc:  # pragma: no cover - exercised in integration
                    payload = {
                        "status": "failed",
                        "cell_key": cell.key,
                        "model_variant": cell.model_variant,
                        "dataset_variant": cell.dataset_variant,
                        "feature_family": cell.feature_family,
                        "calibration_method": cell.calibration_method,
                        "source_fingerprint": current_source_fingerprint,
                        "error": str(exc),
                    }
                _write_json(_cell_output_path(run_dir, cell), payload)
                if payload.get("status") == "success":
                    completed_cells.add(cell.key)
                    results.append(payload)
                    _save_checkpoint(run_dir, completed_cells)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(_evaluate_cell_worker, task_for_cell(cell)): cell
                    for cell in pending_cells
                }
                for future in as_completed(future_map):
                    cell = future_map[future]
                    try:
                        payload = future.result()
                    except Exception as exc:  # pragma: no cover - exercised in integration
                        payload = {
                            "status": "failed",
                            "cell_key": cell.key,
                            "model_variant": cell.model_variant,
                            "dataset_variant": cell.dataset_variant,
                            "feature_family": cell.feature_family,
                            "calibration_method": cell.calibration_method,
                            "source_fingerprint": current_source_fingerprint,
                            "error": str(exc),
                        }
                    _write_json(_cell_output_path(run_dir, cell), payload)
                    if payload.get("status") == "success":
                        completed_cells.add(cell.key)
                        results.append(payload)
                        _save_checkpoint(run_dir, completed_cells)

    summary_rows = [_cell_summary_row(result) for result in results]
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["brier", "ece", "log_loss", "accuracy"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)
    summary_df.to_csv(run_dir / "stage1_summary.csv", index=False)
    return results


def _validate_selection_evidence_payload(
    payload: dict[str, Any],
    *,
    reserved_confirmation_folds: int,
    label: str,
) -> None:
    """Reject legacy/all-fold evidence before it can enter selection."""
    if reserved_confirmation_folds == 0:
        return
    expected = {
        "evaluation_partition": "selection",
        "reserved_confirmation_folds": reserved_confirmation_folds,
        "confirmation_evaluation_n_folds": reserved_confirmation_folds,
    }
    mismatches = [
        field
        for field, expected_value in expected.items()
        if payload.get(field) != expected_value
    ]
    selection_ids = payload.get("selection_fold_ids")
    confirmation_ids = payload.get("confirmation_fold_ids")
    if not isinstance(selection_ids, list) or not selection_ids:
        mismatches.append("selection_fold_ids")
    if (
        not isinstance(confirmation_ids, list)
        or len(confirmation_ids) != reserved_confirmation_folds
    ):
        mismatches.append("confirmation_fold_ids")
    for field in (
        "evaluation_sample_sha256",
        "evaluation_fight_identity_sha256",
        "full_evaluation_sample_sha256",
        "full_evaluation_fight_identity_sha256",
        "confirmation_evaluation_sample_sha256",
        "confirmation_evaluation_fight_identity_sha256",
        "evaluation_input_value_sha256",
        "full_evaluation_input_value_sha256",
        "confirmation_evaluation_input_value_sha256",
        "feature_contract_sha256",
        "input_provenance_payload_sha256",
        "odds_source_inventory_sha256",
        "dataset_fights_sha256",
        "source_dataset_fights_sha256",
        "features_artifact_sha256",
        "features_value_sha256",
        "source_features_sha256",
        "policy_sha256",
        "scheduled_protocol_sha256",
        "evaluation_protocol_sha256",
        "source_fingerprint",
        "source_inventory_sha256",
        "source_inventory_artifact_sha256",
        "environment_artifact_sha256",
        "environment_payload_sha256",
    ):
        if not _valid_sha256(payload.get(field)):
            mismatches.append(field)
    feature_columns = payload.get("feature_contract_columns")
    if (
        not isinstance(feature_columns, list)
        or not feature_columns
        or payload.get("feature_contract_count") != len(feature_columns)
        or payload.get("feature_contract_sha256")
        != _canonical_json_sha256(feature_columns)
    ):
        mismatches.append("feature_contract")
    confirmation_identities = payload.get("confirmation_fight_identities")
    if (
        not isinstance(confirmation_identities, list)
        or len(confirmation_identities)
        != payload.get("confirmation_evaluation_n_fights")
        or _canonical_json_sha256(confirmation_identities)
        != payload.get("confirmation_evaluation_fight_identity_sha256")
    ):
        mismatches.append("confirmation_fight_identities")
    selection_identities = payload.get("selection_fight_identities")
    if (
        not isinstance(selection_identities, list)
        or len(selection_identities) != payload.get("n_predictions")
        or _canonical_json_sha256(selection_identities)
        != payload.get("evaluation_fight_identity_sha256")
    ):
        mismatches.append("selection_fight_identities")
    input_provenance = payload.get("input_provenance")
    if (
        not isinstance(input_provenance, dict)
        or input_provenance.get("input_provenance_payload_sha256")
        != payload.get("input_provenance_payload_sha256")
    ):
        mismatches.append("input_provenance")
    if mismatches:
        raise ValueError(
            f"{label} is not valid selection-only evidence: "
            + ", ".join(sorted(set(mismatches)))
        )


def _load_stage1_results(
    run_dir: Path,
    *,
    reserved_confirmation_folds: int = 0,
) -> list[dict[str, Any]]:
    cell_dir = run_dir / CELL_OUTPUT_DIRNAME
    if not cell_dir.exists():
        return []
    results = []
    for path in sorted(cell_dir.glob("*_metrics.json")):
        payload = _read_json(path)
        if payload.get("status") == "success":
            _validate_selection_evidence_payload(
                payload,
                reserved_confirmation_folds=reserved_confirmation_folds,
                label=f"Stage 1 artifact {path}",
            )
            results.append(payload)
    return results


def _candidate_from_cell_result(result: dict[str, Any]) -> CandidateResult:
    return CandidateResult(
        name=result["model_variant"],
        dataset_variant=result["dataset_variant"],
        feature_family=result["feature_family"],
        calibration=result["calibration_method"],
        retrain_months=result.get("retrain_months"),
        metrics=result.get("metrics", {}),
        sliced_metrics=result.get("sliced_metrics", {}),
    )


def _run_stage2(
    *,
    run_dir: Path,
    freeze_id: str,
    max_finalists: int,
    stage1_results: list[dict[str, Any]],
    reserved_confirmation_folds: int = 0,
) -> list[dict[str, Any]]:
    logger.info("Stage 2: selection gate")
    readiness = validate_frozen_control_arm_for_selection_gate(freeze_id)
    if not readiness["ready"]:
        raise ValueError(
            f"Frozen control arm {freeze_id} is not ready for stage 2: {readiness['errors']}"
        )
    control_metrics = load_frozen_control_metrics(freeze_id)
    for result in stage1_results:
        _validate_selection_evidence_payload(
            result,
            reserved_confirmation_folds=reserved_confirmation_folds,
            label=f"Stage 1 candidate {result.get('cell_key')}",
        )
    uses_confirmation_reserve = reserved_confirmation_folds > 0
    if uses_confirmation_reserve:
        expected_control_fields = {
            "evaluation_partition": "selection",
            "reserved_confirmation_folds": reserved_confirmation_folds,
        }
        mismatched_control_fields = [
            field
            for field, expected in expected_control_fields.items()
            if control_metrics.get(field) != expected
        ]
        if mismatched_control_fields:
            raise ValueError(
                "Frozen control is not bound to the required selection/confirmation "
                f"partition: {mismatched_control_fields}"
            )
        for result in stage1_results:
            binding_fields = tuple(_selection_binding_from_payload(result))
            mismatches = [
                field
                for field in binding_fields
                if result.get(field) != control_metrics.get(field)
            ]
            if mismatches:
                raise ValueError(
                    "Stage 2 candidate/control fold binding differs for "
                    f"{result.get('cell_key')}: {mismatches}"
                )
    gate = SelectionGate(control_metrics)

    candidates = [_candidate_from_cell_result(result) for result in stage1_results]
    evidence_by_candidate = {
        id(candidate): result for candidate, result in zip(candidates, stage1_results)
    }
    finalist_pairs = gate.select_finalists_with_specs(candidates, max_finalists=max_finalists)
    finalists = [candidate for candidate, _spec in finalist_pairs]

    report = gate.generate_report(candidates, finalists)
    (run_dir / "stage2_selection_report.md").write_text(report, encoding="utf-8")

    finalist_payloads = []
    for candidate, spec in finalist_pairs:
        source_evidence = evidence_by_candidate[id(candidate)]
        finalist_payloads.append(
            {
                "candidate_id": spec.candidate_id,
                "model_variant": spec.model_variant,
                "dataset_variant": spec.dataset_variant,
                "feature_family": spec.feature_family,
                "calibration_method": spec.calibration_method,
                "retrain_months": spec.retrain_months,
                "metrics": candidate.metrics,
                "sliced_metrics": candidate.sliced_metrics,
                "evaluation_sample_sha256": source_evidence["evaluation_sample_sha256"],
                "evaluation_partition": source_evidence.get(
                    "evaluation_partition", "all"
                ),
                "reserved_confirmation_folds": source_evidence.get(
                    "reserved_confirmation_folds", 0
                ),
                "selection_fold_ids": source_evidence.get("selection_fold_ids", []),
                "confirmation_fold_ids": source_evidence.get(
                    "confirmation_fold_ids", []
                ),
                "confirmation_evaluation_sample_sha256": source_evidence.get(
                    "confirmation_evaluation_sample_sha256"
                ),
                "full_evaluation_sample_sha256": source_evidence.get(
                    "full_evaluation_sample_sha256"
                ),
                "n_predictions": source_evidence.get("n_predictions"),
                "n_folds": source_evidence.get("n_folds"),
                "full_evaluation_n_fights": source_evidence.get(
                    "full_evaluation_n_fights"
                ),
                "full_evaluation_n_folds": source_evidence.get(
                    "full_evaluation_n_folds"
                ),
                "confirmation_evaluation_n_fights": source_evidence.get(
                    "confirmation_evaluation_n_fights"
                ),
                "confirmation_evaluation_n_folds": source_evidence.get(
                    "confirmation_evaluation_n_folds"
                ),
                "model_spec_name": source_evidence.get("model_spec_name"),
                "model_spec_payload_sha256": source_evidence.get(
                    "model_spec_payload_sha256"
                ),
                "input_provenance": source_evidence.get("input_provenance"),
                "input_provenance_payload_sha256": source_evidence.get(
                    "input_provenance_payload_sha256"
                ),
                "odds_source_inventory": source_evidence.get(
                    "odds_source_inventory"
                ),
                **{
                    field: source_evidence.get(field)
                    for field in _selection_binding_from_payload(source_evidence)
                },
                "gate_result": gate.evaluate(candidate),
            }
        )
    _write_json(
        run_dir / "stage2_finalists.json",
        {
            "freeze_id": freeze_id,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "finalists": finalist_payloads,
        },
    )
    return finalist_payloads


def _load_stage2_finalists(
    run_dir: Path,
    *,
    reserved_confirmation_folds: int = 0,
) -> list[dict[str, Any]]:
    path = run_dir / "stage2_finalists.json"
    if not path.exists():
        return []
    payload = _read_json(path)
    finalists = payload.get("finalists", [])
    for finalist in finalists:
        _validate_selection_evidence_payload(
            finalist,
            reserved_confirmation_folds=reserved_confirmation_folds,
            label=f"Stage 2 finalist {finalist.get('candidate_id')}",
        )
    return finalists


def _spec_from_finalist_payload(payload: dict[str, Any]) -> SweepTargetSpec:
    return SweepTargetSpec(
        candidate_id=payload["candidate_id"],
        model_variant=payload["model_variant"],
        dataset_variant=payload["dataset_variant"],
        feature_family=payload["feature_family"],
        calibration_method=_resolve_variant_calibration(
            payload["model_variant"],
            payload.get("calibration_method"),
        ),
        retrain_months=int(payload.get("retrain_months", 4)),
    )


def _serialize_sweep_result(result: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in result.items()
        if key not in {"bet_log", "bankroll_history"}
    }
    return _normalize_json(payload)


def _save_raw_sweep_result(candidate_dir: Path, label: str, result: dict[str, Any]) -> None:
    _write_json(candidate_dir / f"{label}_result.json", _serialize_sweep_result(result))
    bet_log = result.get("bet_log")
    if isinstance(bet_log, pd.DataFrame):
        bet_log.to_csv(candidate_dir / f"{label}_bet_log.csv", index=False)
    bankroll_history = result.get("bankroll_history")
    if isinstance(bankroll_history, pd.DataFrame):
        bankroll_history.to_csv(candidate_dir / f"{label}_bankroll_history.csv", index=False)


def _load_raw_sweep_result(
    candidate_dir: Path,
    label: str,
    *,
    require_bet_log: bool = False,
) -> dict[str, Any]:
    payload = _read_json(candidate_dir / f"{label}_result.json")
    bet_log_path = candidate_dir / f"{label}_bet_log.csv"
    bankroll_path = candidate_dir / f"{label}_bankroll_history.csv"
    if bet_log_path.exists():
        payload["bet_log"] = pd.read_csv(bet_log_path, parse_dates=["event_date"])
    elif require_bet_log:
        raise ValueError(
            f"Required {label} bet log artifact is missing: {bet_log_path}"
        )
    else:
        payload["bet_log"] = pd.DataFrame()
    payload["bankroll_history"] = pd.read_csv(bankroll_path) if bankroll_path.exists() else pd.DataFrame()
    return payload


def _stage3_state_path(candidate_dir: Path) -> Path:
    return candidate_dir / STAGE3_STATE_FILENAME


def _stage3_context_from_payload(
    payload: dict[str, Any],
    *,
    run_narrow: bool,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    retrain_months = int(payload.get("retrain_months", metadata.get("retrain_months", 4)))
    return {
        "candidate_id": payload["candidate_id"],
        "model_variant": payload["model_variant"],
        "dataset_variant": payload["dataset_variant"],
        "feature_family": payload["feature_family"],
        "calibration_method": _resolve_variant_calibration(
            payload["model_variant"],
            payload.get("calibration_method"),
        ),
        "retrain_months": retrain_months,
        "run_narrow": run_narrow,
        "execution_mode": str(metadata.get("execution_mode", "legacy")),
        "entry_offset_days": metadata.get("entry_offset_days"),
        "entry_offset_for_features": bool(
            metadata.get("entry_offset_for_features", False)
        ),
        "require_entry_odds": bool(metadata.get("require_entry_odds", False)),
        "allow_closing_odds": bool(metadata.get("allow_closing_odds", False)),
        "reserved_confirmation_folds": int(
            metadata.get("reserved_confirmation_folds", 0)
        ),
        "source_fingerprint": metadata.get("source_fingerprint"),
    }


def _new_stage3_state(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "context": context,
        "phases": {
            "production_complete": False,
            "broad_complete": False,
            "narrow_complete": False,
            "final_complete": False,
        },
        "production_row": None,
        "broad_summary": [],
        "narrow_summary": [],
        "best_config": None,
        "primary_summary": [],
    }


def _validate_stage3_context(candidate_dir: Path, state: dict[str, Any], expected_context: dict[str, Any]) -> None:
    saved_context = state.get("context", {})
    mismatches = [
        f"{field} saved={saved_context.get(field)!r} current={expected_context.get(field)!r}"
        for field in expected_context
        if saved_context.get(field) != expected_context.get(field)
    ]
    if mismatches:
        raise ValueError(
            "Stage 3 artifacts for "
            f"{candidate_dir.name} were created with different settings: {', '.join(mismatches)}"
        )


def _hydrate_legacy_stage3_state(candidate_dir: Path, context: dict[str, Any]) -> dict[str, Any] | None:
    summary_path = candidate_dir / "summary.json"
    if not summary_path.exists():
        return None
    if int(context.get("reserved_confirmation_folds", 0)) > 0:
        raise ValueError(
            "Legacy Stage 3 artifacts cannot be rebound to a reserved-fold run; "
            "start a fresh selection-only Stage 3"
        )
    if not (candidate_dir / "production_result.json").exists():
        return None
    if not (candidate_dir / "best_result.json").exists():
        return None

    summary = _read_json(summary_path)
    state = _new_stage3_state(context)
    state["production_row"] = summary.get("production_row")
    state["broad_summary"] = summary.get("broad_summary", [])
    state["narrow_summary"] = summary.get("narrow_summary") or []
    state["best_config"] = summary.get("best_config")
    state["primary_summary"] = summary.get("primary_summary", [])
    state["phases"] = {
        "production_complete": True,
        "broad_complete": bool(summary.get("broad_summary")),
        "narrow_complete": True,
        "final_complete": True,
    }
    _write_json(_stage3_state_path(candidate_dir), state)
    return state


def _load_or_initialize_stage3_state(candidate_dir: Path, expected_context: dict[str, Any]) -> dict[str, Any]:
    state_path = _stage3_state_path(candidate_dir)
    if state_path.exists():
        state = _read_json(state_path)
    else:
        state = _hydrate_legacy_stage3_state(candidate_dir, expected_context) or _new_stage3_state(expected_context)
    _validate_stage3_context(candidate_dir, state, expected_context)
    return state


def _save_stage3_state(candidate_dir: Path, state: dict[str, Any]) -> None:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    _write_json(_stage3_state_path(candidate_dir), state)


def _annotate_sweep_row(
    row: dict[str, Any],
    spec: SweepTargetSpec,
    *,
    promotion_eligible: bool = False,
) -> dict[str, Any]:
    row["candidate_id"] = spec.candidate_id
    row["model_variant"] = spec.model_variant
    row["dataset_variant"] = spec.dataset_variant
    row["feature_family"] = spec.feature_family
    row["calibration_method"] = spec.calibration_method
    row["promotion_eligible"] = bool(promotion_eligible)
    return row


def _persist_stage3_production_phase(
    candidate_dir: Path,
    state: dict[str, Any],
    *,
    production_row: dict[str, Any],
    production_result: dict[str, Any],
) -> None:
    state["production_row"] = production_row
    state["phases"]["production_complete"] = True
    _write_json(candidate_dir / "production_row.json", production_row)
    _save_raw_sweep_result(candidate_dir, "production", production_result)
    _save_stage3_state(candidate_dir, state)


def _persist_stage3_broad_phase(
    candidate_dir: Path,
    state: dict[str, Any],
    *,
    broad_summary: list[dict[str, Any]],
    broad_best_result: dict[str, Any],
) -> None:
    state["broad_summary"] = broad_summary
    state["phases"]["broad_complete"] = True
    pd.DataFrame(broad_summary).to_csv(candidate_dir / "broad_summary.csv", index=False)
    _save_raw_sweep_result(candidate_dir, "broad_best", broad_best_result)
    _save_stage3_state(candidate_dir, state)


def _persist_stage3_narrow_phase(
    candidate_dir: Path,
    state: dict[str, Any],
    *,
    narrow_summary: list[dict[str, Any]],
    narrow_best_result: dict[str, Any] | None,
) -> None:
    state["narrow_summary"] = narrow_summary
    state["phases"]["narrow_complete"] = True
    pd.DataFrame(narrow_summary).to_csv(candidate_dir / "narrow_summary.csv", index=False)
    if narrow_summary and narrow_best_result is not None:
        _save_raw_sweep_result(candidate_dir, "narrow_best", narrow_best_result)
    _save_stage3_state(candidate_dir, state)


def _stage3_candidate_is_complete(candidate_dir: Path, state: dict[str, Any]) -> bool:
    return state.get("phases", {}).get("final_complete", False) and all(
        (candidate_dir / path_name).exists()
        for path_name in ("summary.json", "production_result.json", "best_result.json")
    )


def _load_optional_raw_sweep_result(candidate_dir: Path, label: str) -> dict[str, Any] | None:
    result_path = candidate_dir / f"{label}_result.json"
    if not result_path.exists():
        return None
    return _load_raw_sweep_result(candidate_dir, label)


def _persist_stage3_candidate_result(
    candidate_dir: Path,
    result: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
) -> None:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        key: value
        for key, value in result.items()
        if key not in {
            "production_result", "best_result", "research_best_result",
            "broad_top_results", "narrow_top_results",
        }
    }
    _write_json(candidate_dir / "summary.json", summary_payload)
    pd.DataFrame(result.get("broad_summary", [])).to_csv(candidate_dir / "broad_summary.csv", index=False)
    narrow_summary = result.get("narrow_summary")
    if narrow_summary is not None:
        pd.DataFrame(narrow_summary).to_csv(candidate_dir / "narrow_summary.csv", index=False)
    _save_raw_sweep_result(candidate_dir, "production", result["production_result"])
    _save_raw_sweep_result(candidate_dir, "best", result["best_result"])
    _save_raw_sweep_result(
        candidate_dir,
        "research_best",
        result.get("research_best_result", result["best_result"]),
    )
    if state is None:
        state = _new_stage3_state(
            {
                "candidate_id": result["candidate_id"],
                "model_variant": result["model_variant"],
                "dataset_variant": result["dataset_variant"],
                "feature_family": result["feature_family"],
                "calibration_method": result["calibration_method"],
                "retrain_months": result.get("retrain_months", 4),
                "run_narrow": bool(result.get("narrow_summary")),
                "source_fingerprint": None,
            }
        )
    state["production_row"] = result.get("production_row")
    state["broad_summary"] = result.get("broad_summary", [])
    state["narrow_summary"] = result.get("narrow_summary") or []
    state["best_config"] = result.get("best_config")
    state["primary_summary"] = result.get("primary_summary", [])
    state["phases"] = {
        "production_complete": True,
        "broad_complete": True,
        "narrow_complete": True,
        "final_complete": True,
    }
    _save_stage3_state(candidate_dir, state)


def _load_stage3_candidate_result(
    candidate_dir: Path,
    *,
    expected_context: dict[str, Any],
) -> dict[str, Any]:
    state = _load_or_initialize_stage3_state(candidate_dir, expected_context)
    if not _stage3_candidate_is_complete(candidate_dir, state):
        raise ValueError(f"Stage 3 artifacts for {candidate_dir.name} are incomplete.")
    summary = _read_json(candidate_dir / "summary.json")
    loaded = {
        **summary,
        "production_result": _load_raw_sweep_result(
            candidate_dir,
            "production",
            require_bet_log=True,
        ),
        "best_result": _load_raw_sweep_result(candidate_dir, "best"),
    }
    research_best = _load_optional_raw_sweep_result(candidate_dir, "research_best")
    if research_best is not None:
        loaded["research_best_result"] = research_best
    return loaded


def _run_stage3_candidate(
    *,
    candidate_dir: Path,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    run_narrow: bool,
    initial_bankroll: float = INITIAL_BANKROLL,
    bet_start_date: str = TRAIN_CUTOFF_DATE,
    top_n_broad: int = 5,
) -> dict[str, Any]:
    _validate_selection_evidence_payload(
        payload,
        reserved_confirmation_folds=int(
            metadata.get("reserved_confirmation_folds", 0)
        ),
        label=f"Stage 3 finalist {payload.get('candidate_id')}",
    )
    spec = _spec_from_finalist_payload(payload)
    reserved_confirmation_folds = int(
        metadata.get("reserved_confirmation_folds", 0)
    )
    feature_contract_columns = payload.get("feature_contract_columns")
    if reserved_confirmation_folds and (
        not isinstance(feature_contract_columns, list) or not feature_contract_columns
    ):
        raise ValueError("Stage 3 finalist is missing its exact feature contract")
    contract_evidence = (
        _ordered_feature_contract_evidence(feature_contract_columns)
        if feature_contract_columns
        else {}
    )
    if reserved_confirmation_folds and any(
        payload.get(field) != value for field, value in contract_evidence.items()
    ):
        raise ValueError("Stage 3 finalist feature-contract binding is invalid")
    model_spec_name = payload.get("model_spec_name")
    bound_features = (
        _validate_input_provenance_files(
            payload.get("input_provenance"),
            payload.get("odds_source_inventory"),
        )
        if reserved_confirmation_folds
        else None
    )
    original_factory = None
    if model_spec_name:
        named_spec = resolve_named_training_spec(model_spec_name)
        named_spec_sha256 = _canonical_json_sha256(asdict(named_spec))
        if named_spec_sha256 != payload.get("model_spec_payload_sha256"):
            raise ValueError("Stage 3 named spec changed after Stage 1")
        if feature_contract_columns and list(named_spec.feature_cols) != feature_contract_columns:
            raise ValueError("Stage 3 named spec differs from the selected feature contract")
        original_factory = ALL_VARIANTS[spec.model_variant]
        ALL_VARIANTS[spec.model_variant] = lambda: variant_from_named_training_spec(
            model_spec_name,
            variant_name=spec.model_variant,
        )
    context = _stage3_context_from_payload(
        payload,
        run_narrow=run_narrow,
        metadata=metadata,
    )
    state = _load_or_initialize_stage3_state(candidate_dir, context)
    if _stage3_candidate_is_complete(candidate_dir, state):
        return _load_stage3_candidate_result(candidate_dir, expected_context=context)

    variant_name = spec.model_variant
    execution_mode = str(metadata.get("execution_mode", "legacy"))
    logger.info(
        "Stage 3: generating walk-forward predictions for %s (execution_mode=%s)",
        spec.candidate_id, execution_mode,
    )
    try:
        generated = _generate_walk_forward_predictions(
            **({"features_df": bound_features} if bound_features is not None else {}),
            bet_start_date=bet_start_date,
            variant_name=variant_name,
            dataset_variant=spec.dataset_variant,
            feature_family=spec.feature_family,
            feature_cols=feature_contract_columns,
            calibration_method=spec.calibration_method,
            retrain_months=spec.retrain_months,
            entry_offset_days=metadata.get("entry_offset_days"),
            entry_offset_for_features=bool(
                metadata.get("entry_offset_for_features", False)
            ),
            require_entry_odds=bool(metadata.get("require_entry_odds", False)),
            allow_closing_odds=bool(metadata.get("allow_closing_odds", False)),
            evaluation_partition=(
                "selection" if reserved_confirmation_folds else "all"
            ),
            confirmation_fold_count=(
                reserved_confirmation_folds or DEFAULT_CONFIRMATION_FOLD_COUNT
            ),
            return_fold_manifest=True,
            allow_all_folds=not bool(reserved_confirmation_folds),
        )
    finally:
        if original_factory is not None:
            ALL_VARIANTS[spec.model_variant] = original_factory
    if reserved_confirmation_folds == 0 and isinstance(generated, list):
        # Compatibility for legacy test seams/callers; reserved runs require
        # the metadata-only full manifest and never enter this branch.
        fold_predictions = generated
        fold_manifest = generated
    else:
        fold_predictions, fold_manifest = generated
    partition_evidence = _fold_partition_evidence(
        fold_manifest,
        fold_predictions,
        reserved_confirmation_folds=reserved_confirmation_folds,
        feature_contract_columns=(
            feature_contract_columns if reserved_confirmation_folds else None
        ),
    )
    stage3_predictions = _post_cutoff_predictions(fold_predictions)
    expected_sample_sha256 = payload.get("evaluation_sample_sha256")
    evaluation_sample_sha256 = partition_evidence["evaluation_sample_sha256"]
    if expected_sample_sha256:
        if evaluation_sample_sha256 != expected_sample_sha256:
            raise ValueError(
                f"Stage 3 evaluation sample drift for {spec.candidate_id}: "
                f"stage1={expected_sample_sha256}, stage3={evaluation_sample_sha256}"
            )
    for binding_field in partition_evidence:
        expected = payload.get(binding_field)
        if expected is not None and partition_evidence.get(binding_field) != expected:
            raise ValueError(
                f"Stage 3 {binding_field} drift for {spec.candidate_id}: "
                f"stage1={expected}, stage3={partition_evidence.get(binding_field)}"
            )

    selection_only = bool(reserved_confirmation_folds)

    production_result_path = candidate_dir / "production_result.json"
    if state["phases"].get("production_complete") and production_result_path.exists() and state.get("production_row"):
        production_row = state["production_row"]
        production_result = _load_raw_sweep_result(
            candidate_dir,
            "production",
            require_bet_log=True,
        )
    else:
        production_config = _production_sweep_config(
            variant_name=variant_name,
            name=f"{variant_name}_production_controls",
            is_baseline=True,
        )
        production_result = _evaluate_config(
            fold_predictions,
            production_config,
            initial_bankroll=initial_bankroll,
            bet_start_date=bet_start_date,
            execution_mode=execution_mode,
        )
        production_row = _annotate_sweep_row(
            _summary_row_from_result(production_config, production_result),
            spec,
            promotion_eligible=not selection_only,
        )
        production_result["promotion_eligible"] = not selection_only
        _persist_stage3_production_phase(
            candidate_dir,
            state,
            production_row=production_row,
            production_result=production_result,
        )
    production_row["promotion_eligible"] = not selection_only
    production_result["promotion_eligible"] = not selection_only

    broad_best_path = candidate_dir / "broad_best_result.json"
    if state["phases"].get("broad_complete") and state.get("broad_summary"):
        broad_rows = list(state.get("broad_summary") or [])
        broad_best_row = broad_rows[0] if broad_rows else production_row
        if broad_best_row.get("config") == production_row.get("config"):
            broad_best_result = production_result
        elif broad_best_path.exists():
            broad_best_result = _load_raw_sweep_result(candidate_dir, "broad_best")
        else:
            state["phases"]["broad_complete"] = False
            _save_stage3_state(candidate_dir, state)
            broad_rows = []
            broad_best_result = production_result
    else:
        broad_rows = []
        broad_best_result = production_result

    if not state["phases"].get("broad_complete"):
        broad_configs = build_sweep_configs(
            variant_name=variant_name,
            include_production_controls=False,
        )
        logger.info("Stage 3: broad sweep for %s across %d configs", spec.candidate_id, len(broad_configs))
        broad_rows = [production_row]
        broad_results: dict[str, dict[str, Any]] = {
            production_row["config"]: production_result,
        }

        for index, config in enumerate(broad_configs, 1):
            result = _evaluate_config(
                fold_predictions,
                config,
                initial_bankroll=initial_bankroll,
                bet_start_date=bet_start_date,
                execution_mode=execution_mode,
            )
            broad_results[config.name] = result
            result["promotion_eligible"] = False
            broad_rows.append(
                _annotate_sweep_row(
                    _summary_row_from_result(config, result),
                    spec,
                )
            )
            if index % 50 == 0:
                logger.info(
                    "Stage 3: broad sweep %s progress %d/%d",
                    spec.candidate_id,
                    index,
                    len(broad_configs),
                )

        _sort_summary_rows(broad_rows)
        broad_best_row = broad_rows[0] if broad_rows else production_row
        broad_best_result = broad_results.get(
            broad_best_row["config"],
            production_result,
        )
        _persist_stage3_broad_phase(
            candidate_dir,
            state,
            broad_summary=broad_rows,
            broad_best_result=broad_best_result,
        )

    non_baseline_rows = [row for row in broad_rows if not row.get("is_baseline")]
    for row in non_baseline_rows:
        row["promotion_eligible"] = False
    top_broad_rows = non_baseline_rows[:top_n_broad]
    top_broad_configs = [
        _row_to_sweep_config(row, variant_name)
        for row in top_broad_rows
    ]

    narrow_best_path = candidate_dir / "narrow_best_result.json"
    if state["phases"].get("narrow_complete"):
        narrow_rows = list(state.get("narrow_summary") or [])
        narrow_best_result = (
            _load_raw_sweep_result(candidate_dir, "narrow_best")
            if narrow_rows and narrow_best_path.exists()
            else None
        )
    else:
        narrow_rows = []
        narrow_best_result = None

    if not state["phases"].get("narrow_complete"):
        if run_narrow and top_broad_configs:
            narrow_configs = narrow_sweep_around(
                top_broad_configs,
                variant_name=variant_name,
            )
            if narrow_configs:
                logger.info(
                    "Stage 3: narrow sweep for %s across %d configs",
                    spec.candidate_id,
                    len(narrow_configs),
                )
                narrow_results: dict[str, dict[str, Any]] = {}
                for index, config in enumerate(narrow_configs, 1):
                    result = _evaluate_config(
                        fold_predictions,
                        config,
                        initial_bankroll=initial_bankroll,
                        bet_start_date=bet_start_date,
                        execution_mode=execution_mode,
                    )
                    narrow_results[config.name] = result
                    result["promotion_eligible"] = False
                    narrow_rows.append(
                        _annotate_sweep_row(
                            _summary_row_from_result(config, result),
                            spec,
                        )
                    )
                    if index % 50 == 0:
                        logger.info(
                            "Stage 3: narrow sweep %s progress %d/%d",
                            spec.candidate_id,
                            index,
                            len(narrow_configs),
                        )
                _sort_summary_rows(narrow_rows)
                narrow_best_result = (
                    narrow_results.get(narrow_rows[0]["config"])
                    if narrow_rows
                    else None
                )
            _persist_stage3_narrow_phase(
                candidate_dir,
                state,
                narrow_summary=narrow_rows,
                narrow_best_result=narrow_best_result,
            )
        else:
            _persist_stage3_narrow_phase(
                candidate_dir,
                state,
                narrow_summary=[],
                narrow_best_result=None,
            )
            narrow_rows = []
            narrow_best_result = None

    all_candidate_rows = list(broad_rows)
    if narrow_rows:
        all_candidate_rows.extend(narrow_rows)
    _sort_summary_rows(all_candidate_rows)
    best_row = all_candidate_rows[0] if all_candidate_rows else production_row

    if best_row.get("config") == production_row.get("config"):
        best_result = production_result
    elif narrow_rows and best_row.get("config") == narrow_rows[0].get("config"):
        best_result = narrow_best_result or broad_best_result
    else:
        best_result = broad_best_result

    # --- Selection-only stability diagnostic ---
    # This uses the last *selection* fold. Reserved confirmation folds never
    # enter ranking, sweeps, diagnostics, or persisted strategy results.
    holdout_row = None
    if len(fold_predictions) >= 3:
        holdout_folds = fold_predictions[-1:]
        best_sweep_config = _row_to_sweep_config(best_row, variant_name)
        holdout_result = _evaluate_config(
            holdout_folds,
            best_sweep_config,
            initial_bankroll=initial_bankroll,
            bet_start_date=bet_start_date,
            execution_mode=execution_mode,
        )
        holdout_row = _annotate_sweep_row(
            _summary_row_from_result(best_sweep_config, holdout_result),
            spec,
        )
        holdout_row["is_holdout"] = True
        n_holdout_bets = holdout_row.get("total_bets", 0)
        holdout_roi = holdout_row.get("roi", 0.0)
        sweep_roi = best_row.get("roi", 0.0)
        logger.info(
            "Stage 3: holdout validation for %s — sweep ROI %.3f vs holdout ROI %.3f (%d bets)",
            spec.candidate_id,
            sweep_roi,
            holdout_roi,
            n_holdout_bets,
        )

    result = {
        "candidate_id": spec.candidate_id,
        "model_variant": spec.model_variant,
        "production_row": production_row,
        "production_result": production_result,
        "broad_summary": broad_rows,
        "broad_top_results": {},
        "narrow_summary": narrow_rows,
        "narrow_top_results": {},
        "best_config": best_row,
        # ``best_result`` remains as a compatibility alias, but points to the
        # fixed production controls. Research sweep winners are evidence only.
        "best_result": production_result,
        "research_best_result": best_result,
        "holdout_row": holdout_row,
        "primary_summary": narrow_rows if narrow_rows else broad_rows,
        "dataset_variant": spec.dataset_variant,
        "feature_family": spec.feature_family,
        "calibration_method": spec.calibration_method,
        "retrain_months": spec.retrain_months,
        **partition_evidence,
    }
    _persist_stage3_candidate_result(
        candidate_dir,
        result,
        state=state,
    )
    return result


_FIXED_CONTROL_RESEARCH_ARTIFACT_PREFIXES = (
    "best_",
    "broad_",
    "narrow_",
    "research_",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_fixed_control_stage3_candidate(
    *,
    run_dir: Path,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    initial_bankroll: float = INITIAL_BANKROLL,
    bet_start_date: str = TRAIN_CUTOFF_DATE,
) -> dict[str, Any]:
    """Build only the predeclared production result for a frozen control.

    This deliberately bypasses candidate selection because it constructs the
    comparison control itself. It never evaluates broad/narrow research grids
    and binds both model fits to the predeclared evaluation index.
    """
    expected_prediction_hashes = _fixed_control_prediction_hashes(
        payload,
        label="Fixed-control Stage 3 payload",
    )
    candidate_dir = _stage3_candidate_dir(run_dir, payload["candidate_id"])
    candidate_dir.mkdir(parents=True, exist_ok=True)

    forbidden = sorted(
        path.name
        for path in candidate_dir.iterdir()
        if path.name.startswith(_FIXED_CONTROL_RESEARCH_ARTIFACT_PREFIXES)
    )
    if forbidden:
        raise ValueError(
            "Fixed-control bootstrap refuses research-grid artifacts: "
            f"{forbidden}"
        )
    for forbidden_path in (
        run_dir / "stage2_finalists.json",
        run_dir / PROMOTION_OUTPUT_DIRNAME,
    ):
        if forbidden_path.exists():
            raise ValueError(
                "Fixed-control bootstrap refuses selection/promotion artifacts: "
                f"{forbidden_path}"
            )

    spec = _spec_from_finalist_payload(payload)
    named_spec = resolve_named_training_spec(metadata["selected_model_spec_name"])
    named_spec_sha256 = _canonical_json_sha256(asdict(named_spec))
    if named_spec_sha256 != metadata["selected_model_spec_payload_sha256"]:
        raise ValueError("Fixed-control named spec changed after Stage 1")
    feature_contract_columns = list(named_spec.feature_cols)
    contract_evidence = _ordered_feature_contract_evidence(feature_contract_columns)
    if any(payload.get(field) != value for field, value in contract_evidence.items()):
        raise ValueError("Fixed-control Stage 1 did not use the exact named feature contract")
    context = _stage3_context_from_payload(
        payload,
        run_narrow=False,
        metadata=metadata,
    )
    context.update(
        {
            "fixed_control_bootstrap": True,
            "selected_model_spec_name": metadata["selected_model_spec_name"],
            "selected_model_spec_payload_sha256": metadata[
                "selected_model_spec_payload_sha256"
            ],
            "expected_evaluation_sample_sha256": metadata[
                "expected_evaluation_sample_sha256"
            ],
            "expected_evaluation_n_fights": metadata[
                "expected_evaluation_n_fights"
            ],
            "expected_evaluation_n_folds": metadata[
                "expected_evaluation_n_folds"
            ],
        }
    )
    state = _load_or_initialize_stage3_state(candidate_dir, context)

    expected_sample = str(metadata["expected_evaluation_sample_sha256"])
    expected_fights = int(metadata["expected_evaluation_n_fights"])
    expected_folds = int(metadata["expected_evaluation_n_folds"])
    production_path = candidate_dir / "production_result.json"
    if state["phases"].get("production_complete"):
        if not production_path.exists():
            raise ValueError("Fixed-control state is missing production_result.json")
        existing = _load_raw_sweep_result(
            candidate_dir,
            "production",
            require_bet_log=True,
        )
        existing_evidence = existing.get("fixed_control_evidence", {})
        if existing_evidence.get("full_evaluation_sample_sha256") != expected_sample:
            raise ValueError(
                "Existing fixed-control result has a different full sample hash"
            )
        return existing
    if production_path.exists():
        raise ValueError(
            "Incomplete fixed-control state already contains production_result.json"
        )

    input_provenance = payload.get("input_provenance")
    odds_source_inventory = payload.get("odds_source_inventory")
    if not isinstance(input_provenance, dict) or not isinstance(
        odds_source_inventory, dict
    ):
        raise ValueError("Fixed-control Stage 1 is missing immutable input provenance")
    fixed_features = _validate_input_provenance_files(
        input_provenance,
        odds_source_inventory,
    )
    reserved_confirmation_folds = int(metadata["reserved_confirmation_folds"])
    generated = _generate_walk_forward_predictions(
        features_df=fixed_features,
        bet_start_date=bet_start_date,
        variant_name=spec.model_variant,
        dataset_variant=spec.dataset_variant,
        feature_family=spec.feature_family,
        feature_cols=feature_contract_columns,
        calibration_method=spec.calibration_method,
        retrain_months=spec.retrain_months,
        entry_offset_days=metadata["entry_offset_days"],
        entry_offset_for_features=bool(metadata["entry_offset_for_features"]),
        require_entry_odds=bool(metadata["require_entry_odds"]),
        allow_closing_odds=bool(metadata["allow_closing_odds"]),
        evaluation_partition="selection",
        confirmation_fold_count=reserved_confirmation_folds,
        return_fold_manifest=True,
    )
    fold_predictions, fold_manifest = generated
    partition_evidence = _fold_partition_evidence(
        fold_manifest,
        fold_predictions,
        reserved_confirmation_folds=reserved_confirmation_folds,
        feature_contract_columns=feature_contract_columns,
    )
    selection_predictions = _post_cutoff_predictions(fold_predictions)
    prediction_rows_sha256 = _prediction_rows_sha256(selection_predictions)
    prediction_values_sha256 = _prediction_values_sha256(selection_predictions)
    mismatches: list[str] = []
    if partition_evidence["full_evaluation_sample_sha256"] != expected_sample:
        mismatches.append(
            "full sample expected="
            f"{expected_sample} actual={partition_evidence['full_evaluation_sample_sha256']}"
        )
    if partition_evidence["full_evaluation_n_fights"] != expected_fights:
        mismatches.append(
            "full fights expected="
            f"{expected_fights} actual={partition_evidence['full_evaluation_n_fights']}"
        )
    if partition_evidence["full_evaluation_n_folds"] != expected_folds:
        mismatches.append(
            "full folds expected="
            f"{expected_folds} actual={partition_evidence['full_evaluation_n_folds']}"
        )
    for binding_field in _selection_binding_from_payload(partition_evidence):
        expected_binding = payload.get(binding_field)
        if (
            binding_field in partition_evidence
            and expected_binding is not None
            and partition_evidence.get(binding_field) != expected_binding
        ):
            mismatches.append(f"{binding_field} differs from Stage 1")
    for binding_field, actual in (
        ("prediction_rows_sha256", prediction_rows_sha256),
        ("prediction_values_sha256", prediction_values_sha256),
    ):
        if expected_prediction_hashes[binding_field] != actual:
            mismatches.append(f"{binding_field} differs from Stage 1")
    if mismatches:
        raise ValueError(
            "Fixed-control Stage 3 evaluation index mismatch: " + "; ".join(mismatches)
        )

    production_config = _production_sweep_config(
        variant_name=spec.model_variant,
        name=f"{spec.model_variant}_production_controls",
        is_baseline=True,
    )
    production_result = _evaluate_config(
        fold_predictions,
        production_config,
        initial_bankroll=initial_bankroll,
        bet_start_date=bet_start_date,
        execution_mode=str(metadata["execution_mode"]),
    )
    selection_manifest, confirmation_manifest = _partition_evaluation_folds(
        fold_manifest,
        reserved_confirmation_folds=reserved_confirmation_folds,
    )
    index_parts: list[pd.DataFrame] = []
    for partition_name, partition_folds in (
        ("selection", selection_manifest),
        ("confirmation", confirmation_manifest),
    ):
        index_frame = _post_cutoff_predictions(partition_folds)
        index_frame["evaluation_partition"] = partition_name
        index_parts.append(index_frame)
    evaluation_index_path = run_dir / "fixed_control_evaluation_index.csv"
    pd.concat(index_parts, ignore_index=True).to_csv(
        evaluation_index_path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        date_format="%Y-%m-%dT%H:%M:%S.%fZ",
        na_rep=INTEGRITY_V2_EVALUATION_INDEX_NA_REP,
    )
    index_errors = _validate_integrity_v2_index(
        evaluation_index_path,
        expected_bindings=partition_evidence,
    )
    if index_errors:
        raise ValueError(
            "Fixed-control evaluation index failed strict validation: "
            + "; ".join(index_errors)
        )

    for field, actual in _selection_binding_from_payload(partition_evidence).items():
        if (
            field in partition_evidence
            and input_provenance.get(field) is not None
            and input_provenance.get(field) != actual
        ):
            raise ValueError(f"Fixed-control input provenance differs for {field}")
    input_provenance_path = run_dir / FIXED_CONTROL_INPUT_PROVENANCE_FILENAME
    odds_inventory_path = run_dir / FIXED_CONTROL_ODDS_INVENTORY_FILENAME
    _write_or_validate_json(input_provenance_path, input_provenance)
    _write_or_validate_json(odds_inventory_path, odds_source_inventory)
    input_provenance_sha256 = _file_sha256(input_provenance_path)
    odds_source_inventory_sha256 = _file_sha256(odds_inventory_path)
    if _json_artifact_sha256(odds_source_inventory) != payload.get(
        "odds_source_inventory_sha256"
    ):
        raise ValueError("Fixed-control odds inventory semantic hash differs from Stage 1")

    fixed_control_evidence = {
        "role": "fixed_control_construction",
        "stage2_bypassed": True,
        "research_grids_evaluated": False,
        "model_spec_name": metadata["selected_model_spec_name"],
        "model_spec_payload_sha256": metadata[
            "selected_model_spec_payload_sha256"
        ],
        "source_fingerprint": metadata.get("source_fingerprint"),
        "evaluation_protocol": input_provenance["evaluation_protocol"],
        "input_provenance_path": str(input_provenance_path),
        "input_provenance_sha256": input_provenance_sha256,
        "odds_source_inventory_path": str(odds_inventory_path),
        "odds_source_inventory_sha256": odds_source_inventory_sha256,
        "input_provenance_payload_sha256": input_provenance[
            "input_provenance_payload_sha256"
        ],
        **{
            field: input_provenance[field]
            for field in (
                "dataset_fights_sha256",
                "dataset_fights_path",
                "source_dataset_fights_path",
                "source_dataset_fights_sha256",
                "features_artifact_sha256",
                "features_artifact_path",
                "features_value_sha256",
                "source_features_path",
                "source_features_sha256",
                "feature_contract_columns",
                "feature_contract_count",
                "feature_contract_sha256",
                "policy_sha256",
                "scheduled_protocol_sha256",
                "evaluation_protocol_sha256",
                "source_inventory_sha256",
                "source_inventory_path",
                "source_inventory_artifact_sha256",
                "environment_path",
                "environment_artifact_sha256",
                "environment_payload_sha256",
                "model_spec_name",
                "model_spec_payload_sha256",
            )
        },
        **partition_evidence,
        "evaluation_n_fights": partition_evidence["n_predictions"],
        "evaluation_n_folds": partition_evidence["n_folds"],
        "evaluation_index_path": str(evaluation_index_path),
        "evaluation_index_sha256": _file_sha256(evaluation_index_path),
        "prediction_rows_sha256": prediction_rows_sha256,
        "prediction_values_sha256": prediction_values_sha256,
    }
    production_result["promotion_eligible"] = True
    production_result["fixed_control_evidence"] = fixed_control_evidence
    production_row = _annotate_sweep_row(
        _summary_row_from_result(production_config, production_result),
        spec,
        promotion_eligible=True,
    )
    production_row["fixed_control_evidence"] = fixed_control_evidence
    _persist_stage3_production_phase(
        candidate_dir,
        state,
        production_row=production_row,
        production_result=production_result,
    )

    artifact_names = (
        "production_result.json",
        "production_bet_log.csv",
        "production_bankroll_history.csv",
        "production_row.json",
    )
    receipt = {
        "schema_version": 1,
        **fixed_control_evidence,
        "candidate_id": spec.candidate_id,
        "dataset_variant": spec.dataset_variant,
        "feature_family": spec.feature_family,
        "calibration_method": spec.calibration_method,
        "retrain_months": spec.retrain_months,
        "execution_mode": metadata["execution_mode"],
        "entry_offset_days": metadata["entry_offset_days"],
        "entry_offset_for_features": metadata["entry_offset_for_features"],
        "require_entry_odds": metadata["require_entry_odds"],
        "artifacts": {
            name: {
                "path": str(candidate_dir / name),
                "sha256": _file_sha256(candidate_dir / name),
            }
            for name in artifact_names
            if (candidate_dir / name).exists()
        },
    }
    receipt["artifacts"][evaluation_index_path.name] = {
        "path": str(evaluation_index_path),
        "sha256": _file_sha256(evaluation_index_path),
    }
    source_inventory_path = Path(input_provenance["source_inventory_path"])
    environment_path = Path(input_provenance["environment_path"])
    for provenance_path in (
        input_provenance_path,
        odds_inventory_path,
        source_inventory_path,
        environment_path,
    ):
        receipt["artifacts"][provenance_path.name] = {
            "path": str(provenance_path),
            "sha256": _file_sha256(provenance_path),
        }
    _write_json(run_dir / "fixed_control_bootstrap_receipt.json", receipt)
    return production_result


def _run_stage3(
    *,
    run_dir: Path,
    finalists: list[dict[str, Any]],
    run_narrow: bool,
    metadata: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    logger.info("Stage 3: finalist sweeps")
    sweep_results: dict[str, dict[str, Any]] = {}

    for payload in finalists:
        candidate_id = payload["candidate_id"]
        candidate_dir = _stage3_candidate_dir(run_dir, candidate_id)
        context = _stage3_context_from_payload(
            payload,
            run_narrow=run_narrow,
            metadata=metadata,
        )
        state = _load_or_initialize_stage3_state(candidate_dir, context)
        if _stage3_candidate_is_complete(candidate_dir, state):
            logger.info("Stage 3: reusing saved finalist sweep for %s", candidate_id)
            sweep_results[candidate_id] = _load_stage3_candidate_result(
                candidate_dir,
                expected_context=context,
            )
        else:
            logger.info("Stage 3: sweeping finalist %s", candidate_id)
            sweep_results[candidate_id] = _run_stage3_candidate(
                candidate_dir=candidate_dir,
                payload=payload,
                metadata=metadata,
                run_narrow=run_narrow,
            )

    comparison_df = compare_finalist_sweeps(sweep_results)
    comparison_df.to_csv(run_dir / "stage3_comparison.csv", index=False)
    return sweep_results


def _load_stage3_best_results(
    run_dir: Path,
    finalists: list[dict[str, Any]],
    *,
    run_narrow: bool,
    metadata: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for payload in finalists:
        candidate_id = payload["candidate_id"]
        candidate_dir = _stage3_candidate_dir(run_dir, candidate_id)
        if not candidate_dir.exists():
            raise ValueError(
                f"Stage 3 artifacts for {candidate_id} are missing. "
                "Run stage 3 before resuming later stages."
            )
        context = _stage3_context_from_payload(
            payload,
            run_narrow=run_narrow,
            metadata=metadata,
        )
        state = _load_or_initialize_stage3_state(candidate_dir, context)
        if not _stage3_candidate_is_complete(candidate_dir, state):
            raise ValueError(
                f"Stage 3 artifacts for {candidate_id} are incomplete. "
                "Resume stage 3 before loading later stages."
            )
        results[candidate_id] = _load_stage3_candidate_result(
            candidate_dir,
            expected_context=context,
        )
    return results


def _load_control_arm_payloads(freeze_id: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    validation = validate_frozen_control_arm(freeze_id)
    if not validation["valid"]:
        raise ValueError(
            f"Frozen control arm {freeze_id} integrity check failed: {validation['errors']}"
        )
    freeze_dir = Path(validation["path"])
    metrics_payload = _read_json(freeze_dir / "control_metrics.json")
    sweep_payload = None
    sweep_path = freeze_dir / "control_sweep_summary.json"
    if sweep_path.exists():
        sweep_payload = _read_json(sweep_path)
    trading_artifacts = load_frozen_control_trading_artifacts(freeze_id)
    if trading_artifacts:
        sweep_payload = dict(sweep_payload or {})
        sweep_payload.update(trading_artifacts)
    return metrics_payload, sweep_payload


def _control_sweep_is_placeholder(sweep_payload: dict[str, Any] | None) -> bool:
    if not sweep_payload:
        return True
    total_bets = float(sweep_payload.get("total_bets", 0) or 0)
    roi = abs(float(sweep_payload.get("roi", 0.0) or 0.0))
    profit = abs(float(sweep_payload.get("total_profit", 0.0) or 0.0))
    max_drawdown = abs(float(sweep_payload.get("max_drawdown", 0.0) or 0.0))
    avg_clv = abs(float(sweep_payload.get("avg_clv", 0.0) or 0.0))
    return total_bets <= 0 and roi == 0.0 and profit == 0.0 and max_drawdown == 0.0 and avg_clv == 0.0


def _canonicalize_promotion_model_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", payload)
    sliced_metrics = payload.get("sliced_metrics", {})
    fresh = payload.get("fresh_window", sliced_metrics.get("fresh_window", {}))
    raw_by_year = payload.get("year_by_year", sliced_metrics.get("by_year", {}))

    if isinstance(raw_by_year, list):
        year_by_year = [
            {
                "year": int(values["year"]),
                "accuracy": values.get("accuracy"),
                "brier_score": values.get("brier_score", values.get("brier")),
                "ece": values.get("ece"),
                "log_loss": values.get("log_loss"),
                "n_samples": values.get("n_samples", values.get("n")),
                "roi": values.get("roi"),
            }
            for values in raw_by_year
        ]
    else:
        year_by_year = [
            {
                "year": int(year),
                "accuracy": values.get("accuracy"),
                "brier_score": values.get("brier_score", values.get("brier")),
                "ece": values.get("ece"),
                "log_loss": values.get("log_loss"),
                "n_samples": values.get("n_samples", values.get("n")),
                "roi": values.get("roi"),
            }
            for year, values in sorted(raw_by_year.items(), key=lambda item: int(item[0]))
        ]

    return {
        "accuracy": metrics.get("accuracy"),
        "win_rate": metrics.get("accuracy"),
        "brier_score": metrics.get("brier_score", metrics.get("brier")),
        "ece": metrics.get("ece"),
        "log_loss": metrics.get("log_loss"),
        "fresh_data_brier": payload.get("fresh_data_brier", fresh.get("brier")),
        "fresh_data_ece": payload.get("fresh_data_ece", fresh.get("ece")),
        "year_by_year": year_by_year,
        "evaluation_sample_sha256": payload.get(
            "evaluation_sample_sha256", metrics.get("evaluation_sample_sha256")
        ),
        "evaluation_input_value_sha256": payload.get(
            "evaluation_input_value_sha256",
            metrics.get("evaluation_input_value_sha256"),
        ),
        "prediction_rows_sha256": payload.get(
            "prediction_rows_sha256", metrics.get("prediction_rows_sha256")
        ),
        "prediction_values_sha256": payload.get(
            "prediction_values_sha256", metrics.get("prediction_values_sha256")
        ),
        "model_spec_name": payload.get(
            "model_spec_name", metrics.get("model_spec_name")
        ),
        "model_spec_payload_sha256": payload.get(
            "model_spec_payload_sha256", metrics.get("model_spec_payload_sha256")
        ),
        "calibration_method": payload.get(
            "calibration_method", metrics.get("calibration_method")
        ),
        "model_unchanged_declared": payload.get(
            "model_unchanged_declared", metrics.get("model_unchanged_declared")
        ),
    }


MAX_PREDECLARED_PERFORMANCE_PACKAGES = 8
_CONFIRMATION_PROTOCOL = "integrity_v2_bounded_performance_recovery"
_LOCKED_STRATEGY_FIELDS = (
    "name",
    "s_min_edge",
    "s_blend_weight",
    "s_model_agreement_min_edge",
    "c_min_model_prob",
    "c_min_no_odds_prob",
    "c_share",
    "c_bet_fraction",
    "c_max_bet_fraction",
    "c_min_edge",
    "c_max_decimal_odds",
    "m_enabled",
    "m_min_model_prob",
    "m_min_no_odds_prob",
    "m_bet_fraction",
    "m_share",
)


def _resolve_bound_path(raw_path: object, *, relative_to: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("bound artifact path must be a non-empty string")
    path = Path(raw_path)
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _load_hash_bound_json(
    raw_path: object,
    raw_sha256: object,
    *,
    relative_to: Path,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    path = _resolve_bound_path(raw_path, relative_to=relative_to)
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    expected_sha256 = str(raw_sha256 or "").lower()
    if not _valid_sha256(expected_sha256):
        raise ValueError(f"{label} SHA-256 is missing or malformed")
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return path, payload


def _parse_contract_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc


def _validate_locked_strategy_config(payload: object, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} strategy_config must be a JSON object")
    missing = [field for field in _LOCKED_STRATEGY_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"{label} strategy_config is incomplete: {missing}")
    unknown = sorted(set(payload) - set(_LOCKED_STRATEGY_FIELDS))
    if unknown:
        raise ValueError(f"{label} strategy_config has unknown fields: {unknown}")
    normalized = dict(payload)
    if not isinstance(normalized.get("name"), str) or not normalized["name"].strip():
        raise ValueError(f"{label} strategy_config name must be non-empty")
    if not isinstance(normalized.get("m_enabled"), bool):
        raise ValueError(f"{label} strategy_config m_enabled must be boolean")
    if normalized["m_enabled"] is not False:
        raise ValueError(f"{label} strategy_config must disable the near-miss trader")
    probability_fields = (
        "s_min_edge",
        "s_blend_weight",
        "s_model_agreement_min_edge",
        "c_min_model_prob",
        "c_min_no_odds_prob",
        "c_share",
        "c_bet_fraction",
        "c_max_bet_fraction",
        "c_min_edge",
        "m_min_model_prob",
        "m_min_no_odds_prob",
        "m_bet_fraction",
        "m_share",
    )
    for field in probability_fields:
        value = normalized.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} strategy_config {field} must be numeric")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{label} strategy_config {field} must be finite in [0,1]")
    max_odds = normalized.get("c_max_decimal_odds")
    if max_odds is not None and (
        isinstance(max_odds, bool)
        or not isinstance(max_odds, (int, float))
        or not math.isfinite(float(max_odds))
        or float(max_odds) <= 1.0
    ):
        raise ValueError(
            f"{label} strategy_config c_max_decimal_odds must be null or finite > 1"
        )
    if float(normalized["c_bet_fraction"]) > float(normalized["c_max_bet_fraction"]):
        raise ValueError(f"{label} conviction bet fraction exceeds its cap")
    # The near-miss trader is forbidden by this protocol, so its dormant share
    # must not reduce the independently allocated conviction share.
    if normalized.get("c_max_decimal_odds") is None:
        normalized["c_max_decimal_odds"] = float("inf")
    normalized.setdefault("variant_name", "baseline")
    normalized.setdefault("is_baseline", False)
    try:
        SweepConfig(**normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} strategy_config is invalid: {exc}") from exc
    return dict(payload)


def _validate_locked_model_package(
    package: object,
    *,
    label: str,
    expected_spec_name: str | None = None,
    expected_spec_sha256: str | None = None,
    expected_feature_contract: list[str] | None = None,
    expected_retrain_months: int | None = None,
    expected_fullfit_cutoff_date: str | None = None,
) -> dict[str, Any]:
    if not isinstance(package, dict):
        raise ValueError(f"{label} must be a JSON object")
    required = (
        "model_variant",
        "dataset_variant",
        "feature_family",
        "calibration_method",
        "retrain_months",
        "model_spec_name",
        "model_spec_payload_sha256",
        "fullfit_model_spec_name",
        "fullfit_model_spec_payload_sha256",
        "strategy_config",
        "runtime_invariants",
    )
    missing = [field for field in required if package.get(field) is None]
    if missing:
        raise ValueError(f"{label} is incomplete: {missing}")
    model_variant = str(package["model_variant"])
    if model_variant not in ALL_VARIANTS:
        raise ValueError(f"{label} has unknown model_variant {model_variant!r}")
    if isinstance(package["retrain_months"], bool) or int(package["retrain_months"]) <= 0:
        raise ValueError(f"{label} retrain_months must be positive")
    try:
        named_spec = resolve_named_training_spec(str(package["model_spec_name"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} named spec cannot be resolved: {exc}") from exc
    registered_hash = _canonical_json_sha256(asdict(named_spec))
    declared_hash = str(package["model_spec_payload_sha256"]).lower()
    if not _valid_sha256(declared_hash) or registered_hash != declared_hash:
        raise ValueError(
            f"{label} named-spec hash does not match the registered contract"
        )
    base_spec = None
    if expected_spec_name is not None:
        base_spec = resolve_named_training_spec(expected_spec_name)
        base_hash = _canonical_json_sha256(asdict(base_spec))
        if expected_spec_sha256 is not None and base_hash != expected_spec_sha256:
            raise ValueError(f"{label} final policy base spec hash is invalid")
    if expected_feature_contract is not None and list(named_spec.feature_cols) != list(
        expected_feature_contract
    ):
        raise ValueError(f"{label} feature columns differ from the final policy")
    if str(package["dataset_variant"]) != named_spec.dataset_variant:
        raise ValueError(f"{label} dataset_variant differs from its named spec")
    if base_spec is not None:
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
            if getattr(named_spec, field) != getattr(base_spec, field)
        ]
        if changed:
            raise ValueError(
                f"{label} challenger changes frozen data/T-1/noise semantics: {changed}"
            )
        candidate_seed = int((named_spec.xgb_params or {}).get("random_state", -1))
        base_seed = int((base_spec.xgb_params or {}).get("random_state", -2))
        if candidate_seed != base_seed:
            raise ValueError(f"{label} challenger changes the fixed model seed")
    if str(package["calibration_method"]) != named_spec.calibration_method:
        raise ValueError(f"{label} calibration differs from its named spec")
    if str(package["feature_family"]) != "production":
        raise ValueError(f"{label} must use the policy-compatible production family")
    if expected_retrain_months is not None and int(package["retrain_months"]) != int(
        expected_retrain_months
    ):
        raise ValueError(f"{label} retrain cadence differs from the final protocol")
    try:
        fullfit_spec = resolve_named_training_spec(
            str(package["fullfit_model_spec_name"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} fullfit spec cannot be resolved: {exc}") from exc
    fullfit_hash = _canonical_json_sha256(asdict(fullfit_spec))
    if (
        not _valid_sha256(package["fullfit_model_spec_payload_sha256"])
        or fullfit_hash != str(package["fullfit_model_spec_payload_sha256"]).lower()
    ):
        raise ValueError(f"{label} fullfit spec hash differs from the registry")
    eval_payload = asdict(named_spec)
    fullfit_payload = asdict(fullfit_spec)
    allowed_fullfit_differences = {"name", "description", "train_cutoff_date"}
    unexpected_fullfit_differences = [
        field
        for field in eval_payload
        if field not in allowed_fullfit_differences
        and eval_payload[field] != fullfit_payload[field]
    ]
    if unexpected_fullfit_differences:
        raise ValueError(
            f"{label} fullfit spec changes frozen evaluation semantics: "
            + ", ".join(unexpected_fullfit_differences)
        )
    if (
        fullfit_spec.name == named_spec.name
        or fullfit_spec.train_cutoff_date == named_spec.train_cutoff_date
        or list(fullfit_spec.feature_cols) != list(named_spec.feature_cols)
    ):
        raise ValueError(f"{label} fullfit spec is not a distinct exact-feature descendant")
    if (
        expected_fullfit_cutoff_date is not None
        and fullfit_spec.train_cutoff_date != expected_fullfit_cutoff_date
    ):
        raise ValueError(
            f"{label} fullfit cutoff differs from the immutable policy cutoff"
        )
    if package.get("runtime_invariants") != {"near_miss_enabled": False}:
        raise ValueError(f"{label} must bind near_miss_enabled=false")
    _validate_locked_strategy_config(package["strategy_config"], label=label)
    return dict(package)


def _flatten_package_contract(value: object, *, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(
                _flatten_package_contract(value[key], prefix=child_prefix)
            )
        return flattened
    return {prefix: value}


def _package_changed_fields(
    candidate_package: dict[str, Any],
    control_package: dict[str, Any],
) -> list[str]:
    candidate = _flatten_package_contract(candidate_package)
    control = _flatten_package_contract(control_package)
    return sorted(
        field
        for field in set(candidate) | set(control)
        if candidate.get(field) != control.get(field)
    )


def _selection_binding_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "evaluation_partition",
        "reserved_confirmation_folds",
        "selection_fold_ids",
        "confirmation_fold_ids",
        "evaluation_sample_sha256",
        "evaluation_fight_identity_sha256",
        "selection_fight_identities",
        "n_predictions",
        "n_folds",
        "full_evaluation_sample_sha256",
        "full_evaluation_fight_identity_sha256",
        "full_evaluation_n_fights",
        "full_evaluation_n_folds",
        "confirmation_evaluation_sample_sha256",
        "confirmation_evaluation_fight_identity_sha256",
        "confirmation_fight_identities",
        "confirmation_evaluation_n_fights",
        "confirmation_evaluation_n_folds",
        "evaluation_input_value_sha256",
        "full_evaluation_input_value_sha256",
        "confirmation_evaluation_input_value_sha256",
        "feature_contract_columns",
        "feature_contract_count",
        "feature_contract_sha256",
        "dataset_fights_sha256",
        "source_dataset_fights_sha256",
        "features_artifact_sha256",
        "features_value_sha256",
        "source_features_sha256",
        "odds_source_inventory_sha256",
        "policy_sha256",
        "scheduled_protocol_sha256",
        "evaluation_protocol_sha256",
        "source_fingerprint",
        "source_inventory_sha256",
        "source_inventory_artifact_sha256",
        "environment_artifact_sha256",
        "environment_payload_sha256",
        "evaluation_end_date",
        "fresh_window_cutoff",
    )
    return {field: payload.get(field) for field in fields}


def _selection_binding_for_control_metrics_comparison(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Return the immutable binding representable in reduced control metrics."""
    binding = _selection_binding_from_payload(payload)
    binding.pop("selection_fight_identities")
    binding.pop("confirmation_fight_identities")
    return binding


def _strategy_performance_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    """Return exact trading outcomes for a diagnostic bet slice."""
    if rows.empty:
        return {
            "n_bets": 0,
            "wins": 0,
            "total_wagered": 0.0,
            "total_profit": 0.0,
            "roi": 0.0,
        }
    required = ("bet_size", "profit", "won")
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(
            "selection strategy diagnostics are missing settled bet fields: "
            + ", ".join(missing)
        )
    wagered = pd.to_numeric(rows["bet_size"], errors="raise")
    profit = pd.to_numeric(rows["profit"], errors="raise")
    if (
        not np.isfinite(wagered.to_numpy(dtype=float)).all()
        or not np.isfinite(profit.to_numpy(dtype=float)).all()
        or (wagered <= 0).any()
    ):
        raise ValueError("selection strategy diagnostics contain invalid settled bets")
    total_wagered = float(wagered.sum())
    total_profit = float(profit.sum())
    return {
        "n_bets": int(len(rows)),
        "wins": int(rows["won"].astype(bool).sum()),
        "total_wagered": total_wagered,
        "total_profit": total_profit,
        "roi": total_profit / total_wagered,
    }


def _strategy_performance_slices(
    *,
    bet_log: pd.DataFrame,
    predictions: pd.DataFrame,
    selection_index: pd.DataFrame,
    feature_contract_columns: list[str],
) -> dict[str, Any]:
    """Join settled selection bets to exact prediction/input rows and slice P&L."""
    dimensions = (
        "fold",
        "event_month",
        "trader",
        "side",
        "favorite_underdog",
        "edge_bucket",
        "odds_source",
        "missingness_bucket",
    )
    if bet_log.empty:
        return {
            "joined_bet_rows_sha256": _canonical_json_sha256([]),
            "overall": _strategy_performance_metrics(bet_log),
            **{f"by_{dimension}": {} for dimension in dimensions},
        }

    keys = ["event_date", "fighter_a", "fighter_b", "fold"]
    for label, frame in (
        ("selection bet log", bet_log),
        ("selection predictions", predictions),
        ("selection input index", selection_index),
    ):
        missing = [column for column in keys if column not in frame.columns]
        if missing:
            raise ValueError(f"{label} is missing diagnostic identity fields: {missing}")

    bets = bet_log.copy()
    bets["event_date"] = pd.to_datetime(bets["event_date"], errors="raise")
    bets["fold"] = pd.to_numeric(bets["fold"], errors="raise").astype(int)
    scored = predictions.copy()
    scored["event_date"] = pd.to_datetime(scored["event_date"], errors="raise")
    scored["fold"] = pd.to_numeric(scored["fold"], errors="raise").astype(int)
    inputs = selection_index.copy()
    inputs["event_date"] = pd.to_datetime(inputs["event_date"], errors="raise")
    inputs["fold"] = pd.to_numeric(inputs["fold"], errors="raise").astype(int)
    if scored.duplicated(keys).any() or inputs.duplicated(keys).any():
        raise ValueError("selection diagnostics cannot join duplicate fight identities")

    context_columns = keys + [
        column
        for column in ("a_market_prob", "b_market_prob", "odds_source")
        if column in scored.columns
    ]
    context = scored[context_columns].rename(
        columns={
            "a_market_prob": "_diag_a_market_prob",
            "b_market_prob": "_diag_b_market_prob",
            "odds_source": "_diag_odds_source",
        }
    )
    joined = bets.merge(context, on=keys, how="left", validate="many_to_one")
    if len(joined) != len(bets) or joined[["fighter_a", "fighter_b"]].isna().any().any():
        raise ValueError("selection bets could not be bound to prediction identities")

    missing_contract = [
        column for column in feature_contract_columns if column not in inputs.columns
    ]
    if missing_contract:
        raise ValueError(
            "selection strategy diagnostics cannot bind contract columns: "
            + ", ".join(missing_contract)
        )
    input_context = inputs[keys].copy()
    input_context["feature_missing_count"] = inputs[feature_contract_columns].isna().sum(axis=1)
    input_context["feature_missing_fraction"] = (
        input_context["feature_missing_count"] / len(feature_contract_columns)
    )
    joined = joined.merge(input_context, on=keys, how="left", validate="many_to_one")
    if joined["feature_missing_fraction"].isna().any():
        raise ValueError("selection bets could not be bound to exact input missingness")

    side = joined["side"].astype(str).str.lower()
    if not side.isin({"a", "b"}).all():
        raise ValueError("selection bet log contains an invalid fighter side")
    selected_market = np.where(
        side == "a",
        pd.to_numeric(joined.get("_diag_a_market_prob"), errors="coerce"),
        pd.to_numeric(joined.get("_diag_b_market_prob"), errors="coerce"),
    )
    opponent_market = np.where(
        side == "a",
        pd.to_numeric(joined.get("_diag_b_market_prob"), errors="coerce"),
        pd.to_numeric(joined.get("_diag_a_market_prob"), errors="coerce"),
    )
    joined["favorite_underdog"] = np.select(
        [
            np.isfinite(selected_market) & (selected_market > opponent_market),
            np.isfinite(selected_market) & (selected_market < opponent_market),
            np.isfinite(selected_market),
        ],
        ["favorite", "underdog", "pickem"],
        default="<missing>",
    )
    joined["event_month"] = joined["event_date"].dt.strftime("%Y-%m")
    joined["edge_bucket"] = pd.cut(
        pd.to_numeric(joined["edge"], errors="raise"),
        bins=[-np.inf, 0.0, 0.025, 0.05, 0.10, np.inf],
        labels=["non_positive", "0_2.5pct", "2.5_5pct", "5_10pct", "10pct_plus"],
        right=False,
    ).astype(str)
    odds_source = joined.get("odds_source")
    if odds_source is None:
        odds_source = joined.get("_diag_odds_source")
    elif "_diag_odds_source" in joined.columns:
        odds_source = odds_source.where(odds_source.notna(), joined["_diag_odds_source"])
    if odds_source is None:
        raise ValueError("selection strategy diagnostics have no bound odds source")
    joined["odds_source"] = odds_source.fillna("<missing>").astype(str)
    joined["missingness_bucket"] = pd.cut(
        joined["feature_missing_fraction"],
        bins=[-np.inf, 0.0, 0.10, 0.25, np.inf],
        labels=["none", "up_to_10pct", "10_25pct", "over_25pct"],
        include_lowest=True,
    ).astype(str)

    slices: dict[str, Any] = {"overall": _strategy_performance_metrics(joined)}
    for dimension in dimensions:
        slices[f"by_{dimension}"] = {
            str(value): _strategy_performance_metrics(group)
            for value, group in joined.groupby(dimension, sort=True, dropna=False)
        }
    bound_columns = keys + [
        "trader",
        "side",
        "bet_on",
        "bet_size",
        "profit",
        "won",
        "edge",
        "favorite_underdog",
        "edge_bucket",
        "odds_source",
        "feature_missing_count",
        "feature_missing_fraction",
    ]
    slices["joined_bet_rows_sha256"] = _canonical_frame_value_sha256(
        joined,
        columns=bound_columns,
    )
    return slices


def _bounded_selection_diagnostics(
    *,
    predictions: pd.DataFrame,
    selection_index: pd.DataFrame,
    feature_contract_columns: list[str],
    sliced_metrics: dict[str, Any],
    sweep: dict[str, Any],
) -> dict[str, Any]:
    """Build hash-bound selection-only diagnostics for a bounded package."""
    if predictions.empty:
        raise ValueError("bounded selection diagnostics require predictions")
    frame = predictions.copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise")
    by_fold = {
        str(int(fold_id)): _compute_slice_metrics(fold_frame)
        for fold_id, fold_frame in frame.groupby("fold", sort=True)
    }
    side_a = _compute_slice_metrics(frame)
    side_b_frame = pd.DataFrame(
        {
            "target": 1 - frame["target"].astype(int),
            "prob_a": frame["prob_b"].astype(float),
        }
    )
    side_b = _compute_slice_metrics(side_b_frame)

    favorite_underdog: dict[str, Any] = {"available": False}
    edge_buckets: dict[str, Any] = {"available": False}
    if {"a_market_prob", "b_market_prob"}.issubset(frame.columns):
        market_mask = frame["a_market_prob"].notna() & frame["b_market_prob"].notna()
        market = frame.loc[market_mask].copy()
        if not market.empty:
            a_favorite = market["a_market_prob"].astype(float) >= market[
                "b_market_prob"
            ].astype(float)
            favorite = pd.DataFrame(
                {
                    "target": np.where(
                        a_favorite,
                        market["target"].astype(int),
                        1 - market["target"].astype(int),
                    ),
                    "prob_a": np.where(
                        a_favorite,
                        market["prob_a"].astype(float),
                        market["prob_b"].astype(float),
                    ),
                }
            )
            underdog = pd.DataFrame(
                {
                    "target": 1 - favorite["target"].astype(int),
                    "prob_a": 1.0 - favorite["prob_a"].astype(float),
                }
            )
            favorite_underdog = {
                "available": True,
                "favorite": _compute_slice_metrics(favorite),
                "underdog": _compute_slice_metrics(underdog),
            }
            side_rows = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "target": market["target"].astype(int),
                            "prob_a": market["prob_a"].astype(float),
                            "edge": market["prob_a"].astype(float)
                            - market["a_market_prob"].astype(float),
                        }
                    ),
                    pd.DataFrame(
                        {
                            "target": 1 - market["target"].astype(int),
                            "prob_a": market["prob_b"].astype(float),
                            "edge": market["prob_b"].astype(float)
                            - market["b_market_prob"].astype(float),
                        }
                    ),
                ],
                ignore_index=True,
            )
            side_rows["edge_bucket"] = pd.cut(
                side_rows["edge"],
                bins=[-np.inf, 0.0, 0.025, 0.05, 0.10, np.inf],
                labels=["non_positive", "0_2.5pct", "2.5_5pct", "5_10pct", "10pct_plus"],
                right=False,
            )
            edge_buckets = {
                "available": True,
                "buckets": {
                    str(bucket): _compute_slice_metrics(bucket_frame)
                    for bucket, bucket_frame in side_rows.groupby(
                        "edge_bucket", observed=True, sort=True
                    )
                },
            }

    calibration_bins: dict[str, Any] = {}
    calibrated = frame[["target", "prob_a"]].copy()
    calibrated["bin"] = pd.cut(
        calibrated["prob_a"],
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
    )
    for bucket, bucket_frame in calibrated.groupby("bin", observed=True, sort=True):
        calibration_bins[str(bucket)] = {
            "n_samples": int(len(bucket_frame)),
            "mean_probability": float(bucket_frame["prob_a"].mean()),
            "observed_rate": float(bucket_frame["target"].mean()),
        }

    odds_source_counts: dict[str, Any] = {}
    for column in sorted(col for col in frame.columns if "source" in col.lower()):
        counts = frame[column].fillna("<missing>").astype(str).value_counts(dropna=False)
        odds_source_counts[column] = {
            str(value): int(count) for value, count in counts.items()
        }

    missing_columns = [
        column for column in feature_contract_columns if column not in selection_index.columns
    ]
    if missing_columns:
        raise ValueError(
            "selection diagnostics cannot bind missing contract columns: "
            + ", ".join(missing_columns)
        )
    missingness = selection_index[feature_contract_columns].isna().mean()

    mirror_columns = (
        "mirror_swapped_prob_a",
        "mirror_swapped_no_odds_prob_a",
    )
    missing_mirror = [column for column in mirror_columns if column not in frame.columns]
    if missing_mirror:
        raise ValueError(
            "bounded package lacks actual swapped-input inference: "
            + ", ".join(missing_mirror)
        )
    model_mirror_error = (
        frame["prob_a"].astype(float)
        + frame["mirror_swapped_prob_a"].astype(float)
        - 1.0
    ).abs()
    no_odds_mirror_error = (
        frame["no_odds_prob_a"].astype(float)
        + frame["mirror_swapped_no_odds_prob_a"].astype(float)
        - 1.0
    ).abs()
    mirror_symmetry = {
        "check": "actual_ab_swapped_feature_inference",
        "n_rows": int(len(frame)),
        "max_abs_model_probability_complement_error": float(model_mirror_error.max()),
        "max_abs_no_odds_probability_complement_error": float(
            no_odds_mirror_error.max()
        ),
        "inference_receipt_sha256": _canonical_frame_value_sha256(
            frame,
            columns=[
                "event_date",
                "fighter_a",
                "fighter_b",
                "fold",
                "prob_a",
                "no_odds_prob_a",
                "mirror_swapped_prob_a",
                "mirror_swapped_no_odds_prob_a",
            ],
        ),
    }
    mirror_symmetry["passed"] = (
        mirror_symmetry["max_abs_model_probability_complement_error"] <= 1e-12
        and mirror_symmetry["max_abs_no_odds_probability_complement_error"] <= 1e-12
    )
    if not mirror_symmetry["passed"]:
        raise ValueError("bounded package failed actual A/B-swapped inference parity")

    bet_log = sweep.get("bet_log")
    if not isinstance(bet_log, pd.DataFrame):
        raise ValueError("bounded selection strategy did not return a bet-log dataframe")
    if bet_log.empty:
        raise ValueError("bounded selection strategy produced no settled diagnostic bets")
    strategy_performance = _strategy_performance_slices(
        bet_log=bet_log,
        predictions=frame,
        selection_index=selection_index,
        feature_contract_columns=feature_contract_columns,
    )

    return {
        "schema_version": 1,
        "protocol": _CONFIRMATION_PROTOCOL,
        "evaluation_partition": "selection",
        "evaluation_sample_sha256": _evaluation_sample_sha256(frame),
        "by_fold": by_fold,
        "by_time": sliced_metrics.get("by_year", {}),
        "by_side": {"fighter_a": side_a, "fighter_b": side_b},
        "favorite_underdog": favorite_underdog,
        "edge_buckets": edge_buckets,
        "calibration_bins": calibration_bins,
        "odds_source_counts": odds_source_counts,
        "feature_missingness": {
            column: float(missingness[column]) for column in feature_contract_columns
        },
        "mirror_symmetry": mirror_symmetry,
        "strategy_performance": strategy_performance,
    }


def _run_bounded_selection_experiment(
    *,
    run_dir: Path,
    experiment_manifest_path: Path,
    freeze_id: str,
) -> tuple[Path, dict[str, Any]]:
    """Evaluate at most eight predeclared strategy packages on folds 1..N-2."""
    experiment_manifest_path = experiment_manifest_path.resolve()
    if not experiment_manifest_path.is_file():
        raise ValueError(f"bounded experiment manifest is missing: {experiment_manifest_path}")
    experiment_sha256 = _file_sha256(experiment_manifest_path)
    experiment = _read_json(experiment_manifest_path)
    if (
        not isinstance(experiment, dict)
        or experiment.get("schema_version") != 1
        or experiment.get("protocol") != _CONFIRMATION_PROTOCOL
        or experiment.get("reserved_confirmation_folds")
        != DEFAULT_CONFIRMATION_FOLD_COUNT
        or experiment.get("ranking_policy") != BOUNDED_SELECTION_RANKING_POLICY
    ):
        raise ValueError("bounded experiment manifest has an invalid fixed protocol")
    _parse_contract_timestamp(
        experiment.get("created_at"),
        label="bounded experiment created_at",
    )

    runtime_code = _collect_runtime_code_metadata()
    policy_provenance = _load_policy_provenance()
    policy = _read_json(Path(policy_provenance["policy_path"]))
    policy_contract = policy["contract"]
    policy_evaluation = policy["evaluation"]
    protocol = _evaluation_protocol_payload(
        bet_start_date=policy_evaluation["bet_start_date"],
        execution_mode=policy_evaluation["execution_mode"],
        entry_offset_days=policy_evaluation["entry_offset_days"],
        entry_offset_for_features=policy_evaluation["entry_offset_for_features"],
        require_entry_odds=policy_evaluation["require_entry_odds"],
        allow_closing_odds=False,
        reserved_confirmation_folds=DEFAULT_CONFIRMATION_FOLD_COUNT,
        bootstrap=DEFAULT_BOOTSTRAP,
        retrain_months=policy_evaluation["retrain_months"],
        policy_evaluation=policy_evaluation,
    )
    expected_manifest_bindings = {
        "source_fingerprint": runtime_code["source_fingerprint"],
        "policy_sha256": policy_provenance["policy_sha256"],
        "scheduled_protocol_sha256": policy_provenance[
            "scheduled_protocol_sha256"
        ],
        "evaluation_protocol_sha256": _canonical_json_sha256(protocol),
    }
    stale = [
        field
        for field, expected in expected_manifest_bindings.items()
        if experiment.get(field) != expected
    ]
    if stale:
        raise ValueError("bounded experiment manifest has stale bindings: " + ", ".join(stale))
    packages = experiment.get("packages")
    if (
        not isinstance(packages, list)
        or not packages
        or len(packages) > MAX_PREDECLARED_PERFORMANCE_PACKAGES
    ):
        raise ValueError("bounded selection requires between one and eight packages")
    declared_packages: list[tuple[str, dict[str, Any], str, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    policy_spec = resolve_named_training_spec(policy_contract["evaluation_spec_name"])
    for declaration in packages:
        if not isinstance(declaration, dict):
            raise ValueError("bounded package declaration must be an object")
        package_id = str(declaration.get("package_id") or "").strip()
        if not package_id or package_id in seen_ids:
            raise ValueError("bounded package IDs must be non-empty and unique")
        seen_ids.add(package_id)
        package = _validate_locked_model_package(
            declaration.get("package"),
            label=f"Bounded package {package_id}",
            expected_spec_name=policy_contract["evaluation_spec_name"],
            expected_spec_sha256=policy_contract["evaluation_spec_payload_sha256"],
            expected_feature_contract=list(policy_spec.feature_cols),
            expected_retrain_months=policy_evaluation["retrain_months"],
            expected_fullfit_cutoff_date=policy_contract[
                "exclusive_train_cutoff_date"
            ],
        )
        contract_sha256 = _canonical_json_sha256(package)
        if declaration.get("package_contract_sha256") != contract_sha256:
            raise ValueError(f"bounded package {package_id} contract hash is invalid")
        hypothesis = str(declaration.get("hypothesis") or "").strip()
        rationale = str(declaration.get("rationale") or "").strip()
        changed_fields = declaration.get("changed_fields")
        if (
            not hypothesis
            or not rationale
            or not isinstance(changed_fields, list)
            or not changed_fields
            or any(not isinstance(field, str) or not field for field in changed_fields)
            or len(set(changed_fields)) != len(changed_fields)
        ):
            raise ValueError(
                f"bounded package {package_id} lacks a justified changed-fields declaration"
            )
        declaration_evidence = {
            "hypothesis": hypothesis,
            "rationale": rationale,
            "changed_fields": sorted(changed_fields),
        }
        declared_packages.append(
            (package_id, package, contract_sha256, declaration_evidence)
        )

    readiness = validate_frozen_control_arm_for_selection_gate(freeze_id)
    if not readiness["ready"]:
        raise ValueError(f"frozen control is not ready for bounded selection: {readiness['errors']}")
    control_metrics, control_sweep_payload = _load_control_arm_payloads(freeze_id)
    if not isinstance(control_sweep_payload, dict) or not isinstance(
        control_sweep_payload.get("config"), dict
    ):
        raise ValueError("frozen control has no complete strategy package")
    raw_control_strategy = control_sweep_payload["config"]
    control_strategy = {
        field: raw_control_strategy[field] for field in _LOCKED_STRATEGY_FIELDS
    }
    control_package = _validate_locked_model_package(
        {
            "model_variant": control_metrics["model_variant"],
            "dataset_variant": control_metrics["dataset_variant"],
            "feature_family": control_metrics["feature_family"],
            "calibration_method": control_metrics["calibration_method"],
            "retrain_months": control_metrics["retrain_months"],
            "model_spec_name": control_metrics["model_spec_name"],
            "model_spec_payload_sha256": control_metrics[
                "model_spec_payload_sha256"
            ],
            "fullfit_model_spec_name": policy_contract["fullfit_spec_name"],
            "fullfit_model_spec_payload_sha256": policy_contract[
                "fullfit_spec_payload_sha256"
            ],
            "strategy_config": control_strategy,
            "runtime_invariants": {"near_miss_enabled": False},
        },
        label="Bounded frozen control package",
        expected_spec_name=policy_contract["evaluation_spec_name"],
        expected_spec_sha256=policy_contract["evaluation_spec_payload_sha256"],
        expected_feature_contract=list(policy_spec.feature_cols),
        expected_retrain_months=policy_evaluation["retrain_months"],
        expected_fullfit_cutoff_date=policy_contract[
            "exclusive_train_cutoff_date"
        ],
    )
    for package_id, package, _contract_sha256, declaration in declared_packages:
        actual_changed_fields = _package_changed_fields(package, control_package)
        if declaration["changed_fields"] != actual_changed_fields:
            raise ValueError(
                f"bounded package {package_id} changed_fields differ from the frozen control: "
                f"declared={declaration['changed_fields']} actual={actual_changed_fields}"
            )
    source_inventory_path = run_dir / SOURCE_INVENTORY_FILENAME
    _write_or_validate_json(source_inventory_path, runtime_code["source_inventory"])
    environment_path = run_dir / ENVIRONMENT_INVENTORY_FILENAME
    _write_or_validate_json(environment_path, runtime_code["environment"])
    corrected_features_path = Path(policy_provenance["corrected_features_path"])
    corrected_features = _load_cached_frame(corrected_features_path)
    odds_inventory = _historical_odds_inventory_payload(protocol)

    from src.strategy.model_lab import _materialize_variant_contract_features

    feature_contract_columns = list(policy_spec.feature_cols)
    model_cache: dict[str, dict[str, Any]] = {}
    evaluated: list[dict[str, Any]] = []
    selection_gate = SelectionGate(control_metrics)
    durable_selection_identity_sha256: str | None = None

    def _preflight_selection_exposure(
        fold_manifest: list[tuple[int, pd.DataFrame]],
    ) -> None:
        nonlocal durable_selection_identity_sha256
        selection_manifest, _confirmation_manifest = _partition_evaluation_folds(
            fold_manifest,
            reserved_confirmation_folds=DEFAULT_CONFIRMATION_FOLD_COUNT,
        )
        selection_index = _post_cutoff_predictions(selection_manifest)
        identities = _canonical_evaluation_fight_identities(selection_index)
        identity_sha256 = _canonical_json_sha256(identities)
        if (
            durable_selection_identity_sha256 is not None
            and durable_selection_identity_sha256 != identity_sha256
        ):
            raise ValueError("bounded packages resolved different selection identities")
        durable_selection_identity_sha256 = identity_sha256

    for package_id, package, contract_sha256, declaration in declared_packages:
        spec_sha256 = package["model_spec_payload_sha256"]
        model_result = model_cache.get(spec_sha256)
        if model_result is None:
            named_spec = _validate_registered_challenger_spec(
                spec_name=package["model_spec_name"],
                spec_sha256=spec_sha256,
                policy_contract=policy_contract,
            )
            variant = variant_from_named_training_spec(
                named_spec.name,
                variant_name=package["model_variant"],
            )
            prepared_features = _materialize_variant_contract_features(
                corrected_features.copy(), variant
            )
            resolved_columns, _no_odds = resolve_variant_feature_columns(
                prepared_features,
                variant,
                feature_family=package["feature_family"],
                feature_cols=feature_contract_columns,
            )
            if resolved_columns != feature_contract_columns:
                raise ValueError(
                    f"bounded package {package_id} did not resolve the exact policy feature contract"
                )
            generated = generate_variant_fold_predictions(
                prepared_features,
                variant,
                retrain_months=int(package["retrain_months"]),
                initial_train_years=int(policy_evaluation["initial_train_years"]),
                bet_start_date=policy_evaluation["bet_start_date"],
                feature_family=package["feature_family"],
                feature_cols=feature_contract_columns,
                entry_offset_days=policy_evaluation["entry_offset_days"],
                entry_offset_for_features=policy_evaluation[
                    "entry_offset_for_features"
                ],
                require_entry_odds=policy_evaluation["require_entry_odds"],
                allow_closing_odds=False,
                evaluation_partition="selection",
                confirmation_fold_count=DEFAULT_CONFIRMATION_FOLD_COUNT,
                return_fold_manifest=True,
                include_mirror_diagnostics=True,
                pre_fit_manifest_callback=_preflight_selection_exposure,
            )
            fold_predictions, fold_manifest = generated
            if durable_selection_identity_sha256 is None:
                raise ValueError(
                    "bounded selection generator skipped the durable exposure preflight"
                )
            partition_evidence = _fold_partition_evidence(
                fold_manifest,
                fold_predictions,
                reserved_confirmation_folds=DEFAULT_CONFIRMATION_FOLD_COUNT,
                feature_contract_columns=feature_contract_columns,
            )
            input_provenance = _build_input_provenance_payload(
                dataset_fights_path=Path(policy_provenance["corrected_fights_path"]),
                features_artifact_path=corrected_features_path,
                features_frame=corrected_features,
                feature_contract_columns=feature_contract_columns,
                partition_evidence=partition_evidence,
                policy_provenance=policy_provenance,
                evaluation_protocol=protocol,
                odds_source_inventory=odds_inventory,
                source_fingerprint=runtime_code["source_fingerprint"],
                source_inventory_sha256=runtime_code["source_inventory_sha256"],
                source_inventory_path=source_inventory_path,
                source_inventory_artifact_sha256=_file_sha256(source_inventory_path),
                environment_path=environment_path,
                environment_artifact_sha256=_file_sha256(environment_path),
                environment_payload_sha256=runtime_code[
                    "environment_payload_sha256"
                ],
                model_spec_name=named_spec.name,
                model_spec_payload_sha256=spec_sha256,
            )
            selection_evidence = {
                **partition_evidence,
                **{
                    field: input_provenance[field]
                    for field in _selection_binding_from_payload(input_provenance)
                    if field in input_provenance
                },
                "input_provenance": input_provenance,
                "input_provenance_payload_sha256": input_provenance[
                    "input_provenance_payload_sha256"
                ],
                "odds_source_inventory": odds_inventory,
                "model_spec_name": named_spec.name,
                "model_spec_payload_sha256": spec_sha256,
            }
            _validate_selection_evidence_payload(
                selection_evidence,
                reserved_confirmation_folds=DEFAULT_CONFIRMATION_FOLD_COUNT,
                label=f"bounded package {package_id} selection evidence",
            )
            candidate_selection = _selection_binding_for_control_metrics_comparison(
                selection_evidence
            )
            control_selection = _selection_binding_for_control_metrics_comparison(
                control_metrics
            )
            if candidate_selection != control_selection:
                raise ValueError(
                    f"bounded package {package_id}/control immutable selection input differs"
                )
            predictions = _post_cutoff_predictions(fold_predictions)
            model_metrics = _compute_model_metrics(
                predictions,
                bootstrap=DEFAULT_BOOTSTRAP,
                seed=_seed_from_key(f"bounded:{experiment_sha256}:{spec_sha256}"),
            )
            prediction_rows_sha256 = _prediction_rows_sha256(predictions)
            prediction_values_sha256 = _prediction_values_sha256(predictions)
            model_metrics.update(
                {
                    "evaluation_input_value_sha256": partition_evidence[
                        "evaluation_input_value_sha256"
                    ],
                    "prediction_rows_sha256": prediction_rows_sha256,
                    "prediction_values_sha256": prediction_values_sha256,
                    "model_spec_name": named_spec.name,
                    "model_spec_payload_sha256": spec_sha256,
                    "calibration_method": named_spec.calibration_method,
                }
            )
            sliced_metrics = _compute_sliced_metrics(
                predictions,
                prepared_features,
                feature_family=package["feature_family"],
                fresh_window_cutoff=pd.Timestamp(
                    partition_evidence["fresh_window_cutoff"]
                ),
            )
            strict_gate_result = selection_gate.evaluate(
                CandidateResult(
                    name=named_spec.name,
                    dataset_variant=named_spec.dataset_variant,
                    feature_family=package["feature_family"],
                    calibration=named_spec.calibration_method,
                    retrain_months=int(package["retrain_months"]),
                    metrics=model_metrics,
                    sliced_metrics=sliced_metrics,
                )
            )
            model_unchanged_declared = all(
                (
                    candidate_value is not None
                    and control_metrics.get(field) == candidate_value
                )
                for field, candidate_value in (
                    ("model_spec_name", named_spec.name),
                    ("model_spec_payload_sha256", spec_sha256),
                    ("calibration_method", named_spec.calibration_method),
                    ("prediction_rows_sha256", prediction_rows_sha256),
                    ("prediction_values_sha256", prediction_values_sha256),
                )
            )
            if model_unchanged_declared:
                for metric_name in ("brier", "ece"):
                    value = model_metrics.get(metric_name)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        raise ValueError(
                            f"unchanged model has invalid {metric_name} selection metric"
                        )
                gate_result = {
                    **strict_gate_result,
                    "passed": True,
                    "route": "exact_model_identity_strategy_only",
                    "model_improvement_claimed": False,
                    "model_identity": {
                        "passed": True,
                        "prediction_rows_sha256": prediction_rows_sha256,
                        "prediction_values_sha256": prediction_values_sha256,
                    },
                }
            else:
                gate_result = {
                    **strict_gate_result,
                    "route": "registered_challenger_strict_improvement",
                    "model_improvement_claimed": True,
                }
            composite_score = float(gate_result.get("composite_score"))
            if not math.isfinite(composite_score):
                raise ValueError(
                    f"bounded package {package_id} produced a non-finite selection score"
                )
            selection_manifest, _confirmation_manifest = _partition_evaluation_folds(
                fold_manifest,
                reserved_confirmation_folds=DEFAULT_CONFIRMATION_FOLD_COUNT,
            )
            model_result = {
                "fold_predictions": fold_predictions,
                "predictions": predictions,
                "selection_index": _post_cutoff_predictions(selection_manifest),
                "prepared_features": prepared_features,
                "partition_evidence": partition_evidence,
                "input_provenance": input_provenance,
                "selection_evidence": selection_evidence,
                "model_metrics": model_metrics,
                "sliced_metrics": sliced_metrics,
                "selection_gate": gate_result,
                "model_unchanged_declared": model_unchanged_declared,
                "composite_score": composite_score,
            }
            model_cache[spec_sha256] = model_result

        sweep_config = _sweep_config_from_locked_package(package)
        sweep = _evaluate_config(
            model_result["fold_predictions"],
            sweep_config,
            initial_bankroll=float(protocol["initial_bankroll"]),
            bet_start_date=protocol["bet_start_date"],
            execution_mode=protocol["execution_mode"],
        )
        combined_strategy = sweep.get("combined")
        if not isinstance(combined_strategy, dict):
            raise ValueError(f"bounded package {package_id} has no combined strategy result")
        roi = float(combined_strategy.get("roi"))
        drawdown = float(
            combined_strategy.get(
                "max_drawdown_pct",
                combined_strategy.get("max_drawdown"),
            )
        )
        if not math.isfinite(roi) or not math.isfinite(drawdown):
            raise ValueError(f"bounded package {package_id} produced non-finite ranking metrics")
        diagnostics = _bounded_selection_diagnostics(
            predictions=model_result["predictions"],
            selection_index=model_result["selection_index"],
            feature_contract_columns=feature_contract_columns,
            sliced_metrics=model_result["sliced_metrics"],
            sweep=sweep,
        )
        evaluated.append(
            {
                "package_id": package_id,
                "package": package,
                "package_contract_sha256": contract_sha256,
                "predeclared_justification": declaration,
                **model_result,
                "sweep": sweep,
                "strategy_metrics": _serialize_sweep_result(sweep),
                "diagnostics": diagnostics,
                "roi": roi,
                "drawdown": drawdown,
            }
        )
    eligible = [row for row in evaluated if row["selection_gate"].get("passed") is True]
    eligible.sort(
        key=lambda row: (
            -row["composite_score"],
            -row["roi"],
            row["drawdown"],
            row["package_id"],
        )
    )
    rank_by_package_id = {
        row["package_id"]: rank for rank, row in enumerate(eligible, start=1)
    }

    result_dir = run_dir / BOUNDED_SELECTION_RESULTS_DIRNAME
    result_dir.mkdir(parents=True, exist_ok=False)
    completed_at = datetime.now(timezone.utc).isoformat()
    winner_path: Path | None = None
    winner_payload: dict[str, Any] | None = None
    summary_rows = []
    for row in evaluated:
        rank = rank_by_package_id.get(row["package_id"])
        safe_package_id = _safe_name(row["package_id"])
        bet_log = row["sweep"].get("bet_log")
        if not isinstance(bet_log, pd.DataFrame):
            bet_log = pd.DataFrame()
        bet_log_path = result_dir / f"{safe_package_id}_selection_bet_log.csv"
        _write_csv_exclusive_fsync(bet_log_path, bet_log)
        diagnostics_path = result_dir / f"{safe_package_id}_selection_diagnostics.json"
        diagnostics_payload = {
            **row["diagnostics"],
            "package_id": row["package_id"],
            "candidate_package": row["package"],
            "package_contract_sha256": row["package_contract_sha256"],
            "predeclared_justification": row["predeclared_justification"],
            "bet_log_path": str(bet_log_path.resolve()),
            "bet_log_sha256": _file_sha256(bet_log_path),
        }
        _write_json_exclusive_fsync(diagnostics_path, diagnostics_payload)
        result_payload = {
            "schema_version": 1,
            "protocol": _CONFIRMATION_PROTOCOL,
            "selection_runner": "bounded_selection_v1",
            "generic_stage3_search_used": False,
            "evaluation_partition": "selection",
            "package_id": row["package_id"],
            "package_contract_sha256": row["package_contract_sha256"],
            "candidate_package": row["package"],
            "predeclared_justification": row["predeclared_justification"],
            "predeclared_experiment_manifest_path": str(experiment_manifest_path),
            "predeclared_experiment_manifest_sha256": experiment_sha256,
            "ranking_policy": experiment["ranking_policy"],
            "ranking": {
                "selected": rank == 1,
                "rank": rank,
                "n_packages": len(evaluated),
                "n_eligible": len(eligible),
            },
            "model_spec_name": row["package"]["model_spec_name"],
            "model_spec_payload_sha256": row["package"][
                "model_spec_payload_sha256"
            ],
            "selection_gate": row["selection_gate"],
            "model_unchanged_declared": row["model_unchanged_declared"],
            "candidate_metrics": {
                "model": row["model_metrics"],
                "sliced": row["sliced_metrics"],
                "strategy": row["strategy_metrics"],
            },
            "selection_evidence": row["selection_evidence"],
            "input_provenance": row["input_provenance"],
            "odds_source_inventory": odds_inventory,
            "selection_diagnostics_path": str(diagnostics_path.resolve()),
            "selection_diagnostics_sha256": _file_sha256(diagnostics_path),
            "selection_bet_log_path": str(bet_log_path.resolve()),
            "selection_bet_log_sha256": _file_sha256(bet_log_path),
            "source_fingerprint": runtime_code["source_fingerprint"],
            "completed_at": completed_at,
        }
        result_path = result_dir / f"{safe_package_id}_selection_result.json"
        _write_json_exclusive_fsync(result_path, result_payload)
        summary_rows.append(
            {
                "package_id": row["package_id"],
                "predeclared_justification": row["predeclared_justification"],
                "rank": rank,
                "selection_gate_passed": row["selection_gate"].get("passed") is True,
                "composite_score": row["composite_score"],
                "roi": row["roi"],
                "drawdown": row["drawdown"],
                "result_path": str(result_path.resolve()),
                "result_sha256": _file_sha256(result_path),
                "diagnostics_path": str(diagnostics_path.resolve()),
                "diagnostics_sha256": _file_sha256(diagnostics_path),
                "bet_log_path": str(bet_log_path.resolve()),
                "bet_log_sha256": _file_sha256(bet_log_path),
            }
        )
        if rank == 1:
            winner_path, winner_payload = result_path, result_payload
    summary_path = run_dir / BOUNDED_SELECTION_SUMMARY_FILENAME
    _write_json_exclusive_fsync(
        summary_path,
        {
            "schema_version": 1,
            "protocol": _CONFIRMATION_PROTOCOL,
            "experiment_manifest_path": str(experiment_manifest_path.resolve()),
            "experiment_manifest_sha256": experiment_sha256,
            "ranking_policy": experiment["ranking_policy"],
            "n_packages": len(evaluated),
            "n_eligible": len(eligible),
            "results": summary_rows,
            "winner_package_id": eligible[0]["package_id"] if eligible else None,
            "completed_at": completed_at,
        },
    )
    if winner_path is None or winner_payload is None:
        raise ValueError("no predeclared package passed the frozen-control selection gate")
    return winner_path, winner_payload


def _validate_bounded_selection_summary(
    *,
    summary_path: Path,
    summary: dict[str, Any],
    experiment_path: Path,
    experiment_manifest: dict[str, Any],
    winner_path: Path,
    winner_result: dict[str, Any],
) -> None:
    """Prove the supplied winner came from the complete bounded result set."""
    if summary_path.name != BOUNDED_SELECTION_SUMMARY_FILENAME:
        raise ValueError("bounded selection summary does not use the canonical filename")
    result_dir = summary_path.parent / BOUNDED_SELECTION_RESULTS_DIRNAME
    if winner_path.parent != result_dir:
        raise ValueError("bounded selection winner is outside the canonical result directory")
    declarations = experiment_manifest.get("packages")
    entries = summary.get("results")
    if not isinstance(declarations, list) or not isinstance(entries, list):
        raise ValueError("bounded selection summary is missing declarations/results")
    declared = {
        item.get("package_id"): item for item in declarations if isinstance(item, dict)
    }
    if (
        summary.get("schema_version") != 1
        or summary.get("protocol") != _CONFIRMATION_PROTOCOL
        or summary.get("experiment_manifest_sha256") != _file_sha256(experiment_path)
        or summary.get("ranking_policy") != BOUNDED_SELECTION_RANKING_POLICY
        or summary.get("n_packages") != len(declarations)
        or len(entries) != len(declarations)
        or {entry.get("package_id") for entry in entries if isinstance(entry, dict)}
        != set(declared)
    ):
        raise ValueError("bounded selection summary does not cover the declared experiment")

    verified: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("bounded selection summary result entry is malformed")
        package_id = entry.get("package_id")
        declaration = declared.get(package_id)
        result_path, result = _load_hash_bound_json(
            entry.get("result_path"),
            entry.get("result_sha256"),
            relative_to=summary_path.parent,
            label=f"bounded result {package_id}",
        )
        expected_result_path = result_dir / (
            f"{_safe_name(str(package_id))}_selection_result.json"
        )
        if result_path != expected_result_path:
            raise ValueError(f"bounded result {package_id} is not at its canonical path")
        ranking = result.get("ranking")
        gate = result.get("selection_gate")
        if (
            result.get("schema_version") != 1
            or result.get("protocol") != _CONFIRMATION_PROTOCOL
            or result.get("selection_runner") != "bounded_selection_v1"
            or result.get("evaluation_partition") != "selection"
            or result.get("generic_stage3_search_used") is not False
            or result.get("package_id") != package_id
            or result.get("package_contract_sha256")
            != declaration.get("package_contract_sha256")
            or result.get("predeclared_justification")
            != {
                "hypothesis": str(declaration.get("hypothesis") or "").strip(),
                "rationale": str(declaration.get("rationale") or "").strip(),
                "changed_fields": sorted(declaration.get("changed_fields") or []),
            }
            or entry.get("predeclared_justification")
            != result.get("predeclared_justification")
            or result.get("predeclared_experiment_manifest_sha256")
            != _file_sha256(experiment_path)
            or result.get("ranking_policy") != BOUNDED_SELECTION_RANKING_POLICY
            or not isinstance(ranking, dict)
            or ranking.get("n_packages") != len(declarations)
            or not isinstance(gate, dict)
            or gate.get("passed") is not entry.get("selection_gate_passed")
            or ranking.get("rank") != entry.get("rank")
            or ranking.get("selected") is not (entry.get("rank") == 1)
            or float(gate.get("composite_score")) != float(entry.get("composite_score"))
        ):
            raise ValueError(f"bounded result {package_id} differs from its summary")
        if declaration.get("package") != result.get("candidate_package"):
            raise ValueError(f"bounded result {package_id} package contract differs")
        diagnostics_path, diagnostics = _load_hash_bound_json(
            result.get("selection_diagnostics_path"),
            result.get("selection_diagnostics_sha256"),
            relative_to=result_dir,
            label=f"bounded diagnostics {package_id}",
        )
        bet_log_path = Path(str(result.get("selection_bet_log_path") or "")).resolve()
        if (
            diagnostics_path
            != result_dir / f"{_safe_name(str(package_id))}_selection_diagnostics.json"
            or diagnostics_path != Path(str(entry.get("diagnostics_path") or "")).resolve()
            or _file_sha256(diagnostics_path) != entry.get("diagnostics_sha256")
            or bet_log_path
            != result_dir / f"{_safe_name(str(package_id))}_selection_bet_log.csv"
            or not bet_log_path.is_file()
            or _file_sha256(bet_log_path) != result.get("selection_bet_log_sha256")
            or _file_sha256(bet_log_path) != entry.get("bet_log_sha256")
        ):
            raise ValueError(f"bounded result {package_id} diagnostics are not hash-bound")
        bet_log = pd.read_csv(bet_log_path)
        mirror = diagnostics.get("mirror_symmetry")
        strategy_performance = diagnostics.get("strategy_performance")
        required_strategy_slices = {
            "overall",
            "by_fold",
            "by_event_month",
            "by_trader",
            "by_side",
            "by_favorite_underdog",
            "by_edge_bucket",
            "by_odds_source",
            "by_missingness_bucket",
            "joined_bet_rows_sha256",
        }
        if (
            diagnostics.get("schema_version") != 1
            or diagnostics.get("protocol") != _CONFIRMATION_PROTOCOL
            or diagnostics.get("evaluation_partition") != "selection"
            or diagnostics.get("evaluation_sample_sha256")
            != (result.get("selection_evidence") or {}).get(
                "evaluation_sample_sha256"
            )
            or diagnostics.get("package_id") != package_id
            or diagnostics.get("candidate_package") != declaration.get("package")
            or diagnostics.get("package_contract_sha256")
            != declaration.get("package_contract_sha256")
            or diagnostics.get("predeclared_justification")
            != result.get("predeclared_justification")
            or Path(str(diagnostics.get("bet_log_path") or "")).resolve()
            != bet_log_path
            or diagnostics.get("bet_log_sha256") != _file_sha256(bet_log_path)
            or bet_log.empty
            or not isinstance(mirror, dict)
            or mirror.get("check") != "actual_ab_swapped_feature_inference"
            or mirror.get("passed") is not True
            or not _valid_sha256(mirror.get("inference_receipt_sha256"))
            or not isinstance(strategy_performance, dict)
            or not required_strategy_slices.issubset(strategy_performance)
            or not _valid_sha256(
                strategy_performance.get("joined_bet_rows_sha256")
            )
            or (strategy_performance.get("overall") or {}).get("n_bets")
            != len(bet_log)
        ):
            raise ValueError(
                f"bounded result {package_id} lacks substantive selection diagnostics"
            )
        for error_field in (
            "max_abs_model_probability_complement_error",
            "max_abs_no_odds_probability_complement_error",
        ):
            error = mirror.get(error_field)
            if (
                isinstance(error, bool)
                or not isinstance(error, (int, float))
                or not math.isfinite(float(error))
                or float(error) > 1e-12
            ):
                raise ValueError(
                    f"bounded result {package_id} mirror inference is not reproducible"
                )
        verified.append(entry)

    eligible = [entry for entry in verified if entry["selection_gate_passed"] is True]
    eligible.sort(
        key=lambda entry: (
            -float(entry["composite_score"]),
            -float(entry["roi"]),
            float(entry["drawdown"]),
            str(entry["package_id"]),
        )
    )
    expected_ranks = {
        entry["package_id"]: rank for rank, entry in enumerate(eligible, start=1)
    }
    if any(entry.get("rank") != expected_ranks.get(entry["package_id"]) for entry in verified):
        raise ValueError("bounded selection summary ranking is not reproducible")
    if (
        not eligible
        or summary.get("n_eligible") != len(eligible)
        or summary.get("winner_package_id") != eligible[0]["package_id"]
        or winner_result.get("package_id") != eligible[0]["package_id"]
        or winner_result.get("ranking", {}).get("rank") != 1
        or _file_sha256(winner_path) != eligible[0].get("result_sha256")
    ):
        raise ValueError("bounded selection winner is not the unique rank-one package")


def _create_or_load_confirmation_lock(
    *,
    run_dir: Path,
    package_manifest_path: Path,
    metadata: dict[str, Any],
    freeze_id: str,
    control_metrics: dict[str, Any],
    control_sweep_payload: dict[str, Any],
    control_validation: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Create the immutable one-package confirmation lock after full preflight."""
    package_manifest_path = package_manifest_path.resolve()
    if not package_manifest_path.is_file():
        raise ValueError(
            f"Predeclared confirmation package manifest is missing: {package_manifest_path}"
        )
    package_manifest_sha256 = _file_sha256(package_manifest_path)
    package_manifest = _read_json(package_manifest_path)
    if not isinstance(package_manifest, dict):
        raise ValueError("Confirmation package manifest must be a JSON object")
    if package_manifest.get("schema_version") != 1:
        raise ValueError("Confirmation package manifest schema_version must be 1")
    if package_manifest.get("protocol") != _CONFIRMATION_PROTOCOL:
        raise ValueError("Confirmation package manifest has the wrong protocol")
    if package_manifest.get("generic_stage3_search_used") is not False:
        raise ValueError(
            "Confirmation refuses generic broad/narrow Stage 3 search artifacts"
        )
    if package_manifest.get("prior_all_fold_baseline_visibility") is not True:
        raise ValueError(
            "Confirmation manifest must record prior all-fold baseline visibility"
        )
    limitation = str(package_manifest.get("baseline_visibility_limitation") or "").strip()
    if not limitation:
        raise ValueError("Confirmation manifest must describe the baseline visibility limitation")

    policy_provenance = _load_policy_provenance()
    policy_evaluation = policy_provenance["policy_evaluation"]
    policy_path = Path(policy_provenance["policy_path"])
    policy_payload = _read_json(policy_path)
    policy_contract = policy_payload["contract"]
    policy_named_spec = resolve_named_training_spec(
        policy_contract["evaluation_spec_name"]
    )
    policy_feature_contract = list(policy_named_spec.feature_cols)
    protocol = _evaluation_protocol_payload(
        bet_start_date=TRAIN_CUTOFF_DATE,
        execution_mode=str(metadata.get("execution_mode")),
        entry_offset_days=metadata.get("entry_offset_days"),
        entry_offset_for_features=bool(metadata.get("entry_offset_for_features")),
        require_entry_odds=bool(metadata.get("require_entry_odds")),
        allow_closing_odds=bool(metadata.get("allow_closing_odds")),
        reserved_confirmation_folds=int(
            metadata.get("reserved_confirmation_folds", -1)
        ),
        bootstrap=int(metadata.get("bootstrap", -1)),
        retrain_months=int(policy_evaluation["retrain_months"]),
        policy_evaluation=policy_evaluation,
    )
    if (
        protocol["reserved_confirmation_folds"]
        != DEFAULT_CONFIRMATION_FOLD_COUNT
        or protocol["allow_closing_odds"] is not False
        or protocol["bootstrap"] != DEFAULT_BOOTSTRAP
    ):
        raise ValueError("Run metadata is not the locked honest confirmation protocol")

    experiment_path, experiment_manifest = _load_hash_bound_json(
        package_manifest.get("predeclared_experiment_manifest_path"),
        package_manifest.get("predeclared_experiment_manifest_sha256"),
        relative_to=package_manifest_path.parent,
        label="Predeclared experiment manifest",
    )
    if (
        experiment_manifest.get("schema_version") != 1
        or experiment_manifest.get("protocol") != _CONFIRMATION_PROTOCOL
        or experiment_manifest.get("reserved_confirmation_folds")
        != DEFAULT_CONFIRMATION_FOLD_COUNT
    ):
        raise ValueError("Predeclared experiment manifest has an invalid protocol binding")
    experiment_created_at = _parse_contract_timestamp(
        experiment_manifest.get("created_at"),
        label="Predeclared experiment created_at",
    )
    expected_experiment_bindings = {
        "source_fingerprint": metadata.get("source_fingerprint"),
        "policy_sha256": policy_provenance["policy_sha256"],
        "scheduled_protocol_sha256": policy_provenance[
            "scheduled_protocol_sha256"
        ],
        "evaluation_protocol_sha256": _canonical_json_sha256(protocol),
    }
    experiment_mismatches = [
        field
        for field, expected in expected_experiment_bindings.items()
        if experiment_manifest.get(field) != expected
    ]
    if experiment_mismatches:
        raise ValueError(
            "Predeclared experiment manifest has stale provenance: "
            + ", ".join(experiment_mismatches)
        )
    experiments = experiment_manifest.get("packages")
    if (
        not isinstance(experiments, list)
        or not experiments
        or len(experiments) > MAX_PREDECLARED_PERFORMANCE_PACKAGES
    ):
        raise ValueError(
            "Predeclared experiment manifest must contain between 1 and "
            f"{MAX_PREDECLARED_PERFORMANCE_PACKAGES} packages"
        )
    package_ids = [
        item.get("package_id") for item in experiments if isinstance(item, dict)
    ]
    if len(package_ids) != len(experiments) or len(set(package_ids)) != len(package_ids):
        raise ValueError("Predeclared experiment package IDs must be unique and complete")

    package_id = str(package_manifest.get("package_id") or "").strip()
    candidate_package = _validate_locked_model_package(
        package_manifest.get("candidate_package"),
        label="Candidate package",
        expected_spec_name=policy_contract["evaluation_spec_name"],
        expected_spec_sha256=policy_contract["evaluation_spec_payload_sha256"],
        expected_feature_contract=policy_feature_contract,
        expected_retrain_months=int(policy_evaluation["retrain_months"]),
        expected_fullfit_cutoff_date=policy_contract[
            "exclusive_train_cutoff_date"
        ],
    )
    candidate_contract_sha256 = _canonical_json_sha256(candidate_package)
    matching_experiment = next(
        (
            item
            for item in experiments
            if item.get("package_id") == package_id
        ),
        None,
    )
    if (
        not isinstance(matching_experiment, dict)
        or matching_experiment.get("package_contract_sha256")
        != candidate_contract_sha256
        or matching_experiment.get("package") != candidate_package
    ):
        raise ValueError(
            "Locked candidate package was not byte-contract predeclared before selection"
        )

    selection_result_path, selection_result = _load_hash_bound_json(
        package_manifest.get("selection_result_path"),
        package_manifest.get("selection_result_sha256"),
        relative_to=package_manifest_path.parent,
        label="Bounded candidate selection result",
    )
    if (
        selection_result.get("schema_version") != 1
        or selection_result.get("protocol") != _CONFIRMATION_PROTOCOL
        or selection_result.get("selection_runner") != "bounded_selection_v1"
        or selection_result.get("generic_stage3_search_used") is not False
        or selection_result.get("evaluation_partition") != "selection"
        or selection_result.get("package_id") != package_id
        or selection_result.get("package_contract_sha256")
        != candidate_contract_sha256
        or selection_result.get("predeclared_experiment_manifest_sha256")
        != _file_sha256(experiment_path)
    ):
        raise ValueError("Candidate selection result is not the bounded predeclared result")
    selection_completed_at = _parse_contract_timestamp(
        selection_result.get("completed_at"),
        label="Candidate selection completed_at",
    )
    if experiment_created_at >= selection_completed_at:
        raise ValueError("Experiment manifest was not predeclared before candidate selection")
    if experiment_path.stat().st_mtime_ns > selection_result_path.stat().st_mtime_ns:
        raise ValueError("Experiment manifest file postdates the selection result")
    ranking = selection_result.get("ranking")
    if (
        not isinstance(ranking, dict)
        or ranking.get("selected") is not True
        or isinstance(ranking.get("rank"), bool)
        or not isinstance(ranking.get("rank"), int)
        or ranking["rank"] != 1
        or ranking.get("n_packages") != len(experiments)
        or not isinstance(selection_result.get("selection_gate"), dict)
        or selection_result["selection_gate"].get("passed") is not True
    ):
        raise ValueError("Candidate selection result does not prove the unique rank-one winner")
    if not isinstance(selection_result.get("candidate_metrics"), dict):
        raise ValueError("Candidate selection result is missing selection-only metrics")
    summary_path, selection_summary = _load_hash_bound_json(
        package_manifest.get("bounded_selection_summary_path"),
        package_manifest.get("bounded_selection_summary_sha256"),
        relative_to=package_manifest_path.parent,
        label="Bounded selection summary",
    )
    _validate_bounded_selection_summary(
        summary_path=summary_path,
        summary=selection_summary,
        experiment_path=experiment_path,
        experiment_manifest=experiment_manifest,
        winner_path=selection_result_path,
        winner_result=selection_result,
    )

    declared_source_fingerprint = package_manifest.get("source_fingerprint")
    if declared_source_fingerprint != metadata.get("source_fingerprint"):
        raise ValueError("Candidate package source fingerprint differs from this run")
    selection_evidence = selection_result.get("selection_evidence")
    if not isinstance(selection_evidence, dict):
        raise ValueError("Candidate selection result is missing selection_evidence")
    _validate_selection_evidence_payload(
        selection_evidence,
        reserved_confirmation_folds=DEFAULT_CONFIRMATION_FOLD_COUNT,
        label="Candidate package selection evidence",
    )
    if package_manifest.get("selection_evidence") not in (None, selection_evidence):
        raise ValueError("Package manifest selection evidence differs from the bound result")
    candidate_input_provenance = selection_result.get("input_provenance")
    candidate_odds_inventory = selection_result.get("odds_source_inventory")
    _validate_input_provenance_files(
        candidate_input_provenance,
        candidate_odds_inventory,
    )
    if not isinstance(candidate_input_provenance, dict):
        raise ValueError("Candidate selection result has no input provenance")
    for field, expected in _selection_binding_from_payload(selection_evidence).items():
        if candidate_input_provenance.get(field) is not None and (
            candidate_input_provenance.get(field) != expected
        ):
            raise ValueError(f"Candidate input provenance differs for {field}")
    control_selection = _selection_binding_for_control_metrics_comparison(
        control_metrics
    )
    if (
        _selection_binding_for_control_metrics_comparison(selection_evidence)
        != control_selection
    ):
        raise ValueError(
            "Candidate and frozen control do not have identical selection/confirmation indices"
        )

    control_strategy = control_sweep_payload.get("config")
    if not isinstance(control_strategy, dict):
        raise ValueError("Frozen control is missing its complete strategy config")
    control_package = _validate_locked_model_package(
        {
            "model_variant": control_metrics.get("model_variant"),
            "dataset_variant": control_metrics.get("dataset_variant"),
            "feature_family": control_metrics.get("feature_family"),
            "calibration_method": control_metrics.get("calibration_method"),
            "retrain_months": control_metrics.get("retrain_months"),
            "model_spec_name": control_metrics.get("model_spec_name"),
            "model_spec_payload_sha256": control_metrics.get(
                "model_spec_payload_sha256"
            ),
            "fullfit_model_spec_name": policy_contract["fullfit_spec_name"],
            "fullfit_model_spec_payload_sha256": policy_contract[
                "fullfit_spec_payload_sha256"
            ],
            "strategy_config": control_strategy,
            "runtime_invariants": {"near_miss_enabled": False},
        },
        label="Frozen control package",
        expected_spec_name=policy_contract["evaluation_spec_name"],
        expected_spec_sha256=policy_contract["evaluation_spec_payload_sha256"],
        expected_feature_contract=policy_feature_contract,
        expected_retrain_months=int(policy_evaluation["retrain_months"]),
        expected_fullfit_cutoff_date=policy_contract[
            "exclusive_train_cutoff_date"
        ],
    )
    candidate_selection_model = selection_result["candidate_metrics"].get("model")
    if not isinstance(candidate_selection_model, dict):
        raise ValueError("bounded winner is missing its selection model payload")
    expected_candidate_model_fields = {
        "evaluation_input_value_sha256": selection_evidence[
            "evaluation_input_value_sha256"
        ],
        "model_spec_name": candidate_package["model_spec_name"],
        "model_spec_payload_sha256": candidate_package[
            "model_spec_payload_sha256"
        ],
        "calibration_method": candidate_package["calibration_method"],
    }
    for field, expected in expected_candidate_model_fields.items():
        if candidate_selection_model.get(field) != expected:
            raise ValueError(f"bounded winner model payload differs for {field}")
    for field in ("prediction_rows_sha256", "prediction_values_sha256"):
        if not _valid_sha256(candidate_selection_model.get(field)):
            raise ValueError(f"bounded winner has no valid {field}")
        if not _valid_sha256(control_metrics.get(field)):
            raise ValueError(f"frozen control has no valid {field}")
    same_registered_model = all(
        candidate_package[field] == control_package[field]
        for field in (
            "model_spec_name",
            "model_spec_payload_sha256",
            "calibration_method",
        )
    )
    exact_prediction_identity = all(
        candidate_selection_model[field] == control_metrics[field]
        for field in ("prediction_rows_sha256", "prediction_values_sha256")
    )
    declared_unchanged = selection_result.get("model_unchanged_declared") is True
    if same_registered_model:
        if not exact_prediction_identity or not declared_unchanged:
            raise ValueError(
                "strategy-only winner does not prove exact control prediction identity"
            )
    elif declared_unchanged:
        raise ValueError("registered challenger cannot declare the control model unchanged")

    control_dir = Path(control_validation["path"])
    control_receipt_path = control_dir / "fixed_control_bootstrap_receipt.json"
    if not control_receipt_path.is_file():
        raise ValueError("Frozen control is missing fixed_control_bootstrap_receipt.json")
    control_receipt = _read_json(control_receipt_path)
    if not isinstance(control_receipt, dict):
        raise ValueError("Frozen control bootstrap receipt is malformed")
    for field in (
        *_selection_binding_for_control_metrics_comparison(control_metrics).keys(),
        "evaluation_index_sha256",
        "input_provenance_sha256",
        "odds_source_inventory_sha256",
        "model_spec_name",
        "model_spec_payload_sha256",
    ):
        if control_receipt.get(field) != control_metrics.get(field):
            raise ValueError(f"Frozen control receipt/metrics differ for {field}")
    control_input_path = control_dir / FIXED_CONTROL_INPUT_PROVENANCE_FILENAME
    control_odds_path = control_dir / FIXED_CONTROL_ODDS_INVENTORY_FILENAME
    control_source_inventory_path = control_dir / SOURCE_INVENTORY_FILENAME
    control_environment_path = control_dir / ENVIRONMENT_INVENTORY_FILENAME
    if (
        not control_input_path.is_file()
        or not control_odds_path.is_file()
        or not control_environment_path.is_file()
    ):
        raise ValueError("Frozen control is missing input-provenance artifacts")
    control_input_provenance = _read_json(control_input_path)
    control_odds_inventory = _read_json(control_odds_path)
    _validate_input_provenance_files(
        control_input_provenance,
        control_odds_inventory,
        source_inventory_path_override=control_source_inventory_path,
        environment_path_override=control_environment_path,
    )
    if control_input_provenance.get("evaluation_protocol") != protocol:
        raise ValueError("Frozen control protocol differs from confirmation protocol")
    if candidate_input_provenance.get("evaluation_protocol") != protocol:
        raise ValueError("Candidate selection protocol differs from confirmation protocol")
    if _selection_binding_from_payload(candidate_input_provenance) != (
        _selection_binding_from_payload(control_input_provenance)
    ):
        raise ValueError("Candidate/control immutable input provenance differs")

    lock_payload = {
        "schema_version": 1,
        "protocol": _CONFIRMATION_PROTOCOL,
        "package_id": package_id,
        "package_manifest_path": str(package_manifest_path),
        "package_manifest_sha256": package_manifest_sha256,
        "predeclared_experiment_manifest_path": str(experiment_path),
        "predeclared_experiment_manifest_sha256": _file_sha256(experiment_path),
        "selection_result_path": str(selection_result_path),
        "selection_result_sha256": _file_sha256(selection_result_path),
        "bounded_selection_summary_path": str(summary_path),
        "bounded_selection_summary_sha256": _file_sha256(summary_path),
        "selection_diagnostics_path": selection_result[
            "selection_diagnostics_path"
        ],
        "selection_diagnostics_sha256": selection_result[
            "selection_diagnostics_sha256"
        ],
        "selection_bet_log_path": selection_result["selection_bet_log_path"],
        "selection_bet_log_sha256": selection_result["selection_bet_log_sha256"],
        "candidate_package": candidate_package,
        "candidate_package_contract_sha256": candidate_contract_sha256,
        "selection_evidence": _selection_binding_from_payload(selection_evidence),
        "candidate_input_provenance": candidate_input_provenance,
        "candidate_input_provenance_sha256": _json_artifact_sha256(
            candidate_input_provenance
        ),
        "candidate_odds_source_inventory": candidate_odds_inventory,
        "candidate_odds_source_inventory_sha256": _json_artifact_sha256(
            candidate_odds_inventory
        ),
        "frozen_control": {
            "freeze_id": freeze_id,
            "path": str(control_dir),
            "checksums_sha256": _file_sha256(control_dir / "checksums.json"),
            "bootstrap_receipt_path": str(control_receipt_path),
            "bootstrap_receipt_sha256": _file_sha256(control_receipt_path),
            "evaluation_index_sha256": control_metrics.get(
                "evaluation_index_sha256"
            ),
            "input_provenance_path": str(control_input_path),
            "input_provenance_sha256": _file_sha256(control_input_path),
            "odds_source_inventory_path": str(control_odds_path),
            "odds_source_inventory_sha256": _file_sha256(control_odds_path),
            "source_inventory_path": str(control_source_inventory_path),
            "source_inventory_artifact_sha256": _file_sha256(
                control_source_inventory_path
            ),
            "environment_path": str(control_environment_path),
            "environment_artifact_sha256": _file_sha256(
                control_environment_path
            ),
            "environment_payload_sha256": control_input_provenance[
                "environment_payload_sha256"
            ],
            "package": control_package,
        },
        "evaluation_protocol": protocol,
        "evaluation_protocol_sha256": _canonical_json_sha256(protocol),
        "policy_path": str(policy_path),
        "policy_sha256": policy_provenance["policy_sha256"],
        "scheduled_protocol_sha256": policy_provenance[
            "scheduled_protocol_sha256"
        ],
        "source_fingerprint": metadata.get("source_fingerprint"),
        "git_sha": metadata.get("git_sha"),
        "prior_all_fold_baseline_visibility": True,
        "baseline_visibility_limitation": limitation,
        "locked_at": datetime.now().isoformat(timespec="seconds"),
    }
    lock_path = run_dir / CONFIRMATION_LOCK_FILENAME
    if lock_path.exists():
        existing = _read_json(lock_path)
        immutable_fields = (
            "package_id",
            "package_manifest_path",
            "package_manifest_sha256",
            "predeclared_experiment_manifest_path",
            "predeclared_experiment_manifest_sha256",
            "selection_result_path",
            "selection_result_sha256",
            "bounded_selection_summary_path",
            "bounded_selection_summary_sha256",
            "selection_diagnostics_path",
            "selection_diagnostics_sha256",
            "selection_bet_log_path",
            "selection_bet_log_sha256",
            "candidate_package_contract_sha256",
            "selection_evidence",
            "candidate_input_provenance_sha256",
            "candidate_odds_source_inventory_sha256",
            "frozen_control",
            "evaluation_protocol",
            "evaluation_protocol_sha256",
            "policy_sha256",
            "scheduled_protocol_sha256",
            "source_fingerprint",
            "git_sha",
            "prior_all_fold_baseline_visibility",
            "baseline_visibility_limitation",
        )
        if any(existing.get(field) != lock_payload.get(field) for field in immutable_fields):
            raise ValueError("Existing confirmation lock binds a different package or protocol")
    else:
        _write_json_exclusive_fsync(lock_path, lock_payload)
    return _read_json(lock_path), _file_sha256(lock_path)


def _validated_fight_identity_rows(
    identities: object,
    *,
    label: str,
) -> list[dict[str, str]]:
    """Validate the canonical, orientation-invariant identity ledger shape."""
    if not isinstance(identities, list) or not identities:
        raise ValueError(f"{label} requires a non-empty fight-identity ledger")
    expected_keys = {"event_date", "fighter_name_low", "fighter_name_high"}
    rows: list[dict[str, str]] = []
    for raw in identities:
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise ValueError(f"{label} contains a malformed fight identity")
        row = {key: str(raw[key]).strip() for key in expected_keys}
        if not row["fighter_name_low"] or not row["fighter_name_high"]:
            raise ValueError(f"{label} contains an empty fighter identity")
        if row["fighter_name_low"] >= row["fighter_name_high"]:
            raise ValueError(f"{label} fighter identities are not canonically oriented")
        try:
            normalized_date = pd.Timestamp(row["event_date"]).strftime("%Y-%m-%d")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains an invalid event date") from exc
        if normalized_date != row["event_date"]:
            raise ValueError(f"{label} event dates must use YYYY-MM-DD")
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["event_date"],
            row["fighter_name_low"],
            row["fighter_name_high"],
        )
    )
    if rows != identities:
        raise ValueError(f"{label} fight identities are not canonically ordered")
    if len({_canonical_json_sha256(row) for row in rows}) != len(rows):
        raise ValueError(f"{label} contains duplicate fight identities")
    return rows


def _global_confirmation_store() -> Path:
    return (_repo_root() / GLOBAL_CONFIRMATION_CLAIMS_DIR).resolve()


def _global_store_files() -> list[Path]:
    store = _global_confirmation_store()
    return sorted(path.resolve() for path in store.rglob("*") if path.is_file())


def _require_global_store_durable() -> None:
    files = _global_store_files()
    if files:
        require_remotely_anchored_git_artifacts(
            _repo_root(),
            files,
            label="global confirmation ledger",
        )


def _selection_exposure_key_material(
    identities: list[dict[str, str]],
) -> dict[str, str]:
    rows = _validated_fight_identity_rows(
        identities,
        label="selection exposure",
    )
    return {
        "key_contract": "selection_outcome_exposure_v1",
        "selection_evaluation_fight_identity_sha256": _canonical_json_sha256(rows),
    }


def _selection_exposure_path(identities: list[dict[str, str]]) -> Path:
    key = _canonical_json_sha256(_selection_exposure_key_material(identities))
    return (
        _global_confirmation_store()
        / GLOBAL_SELECTION_EXPOSURES_DIRNAME
        / f"{key}.json"
    )


def _validate_selection_exposure_payload(
    payload: object,
    *,
    path: Path,
) -> list[dict[str, str]]:
    if not isinstance(payload, dict):
        raise ValueError("selection exposure ledger is malformed")
    identities = _validated_fight_identity_rows(
        payload.get("selection_fight_identities"),
        label="selection exposure",
    )
    key_material = _selection_exposure_key_material(identities)
    exposure_key = _canonical_json_sha256(key_material)
    expected_path = (
        _global_confirmation_store()
        / GLOBAL_SELECTION_EXPOSURES_DIRNAME
        / f"{exposure_key}.json"
    ).resolve()
    dates = [row["event_date"] for row in identities]
    if (
        path.resolve() != expected_path
        or payload.get("schema_version") != 1
        or payload.get("protocol") != _CONFIRMATION_PROTOCOL
        or payload.get("status") != "selection_exposed"
        or payload.get("selection_exposure_key") != exposure_key
        or payload.get("selection_exposure_key_material") != key_material
        or payload.get("selection_evaluation_fight_identity_sha256")
        != key_material["selection_evaluation_fight_identity_sha256"]
        or payload.get("selection_min_event_date") != min(dates)
        or payload.get("selection_max_event_date") != max(dates)
    ):
        raise ValueError("selection exposure ledger binding is invalid")
    _parse_contract_timestamp(
        payload.get("recorded_at"),
        label="selection exposure recorded_at",
    )
    return identities


def _record_global_selection_exposure(
    identities: list[dict[str, str]],
) -> tuple[dict[str, Any], Path]:
    """Persist selection identities before fitting and require a pushed rerun."""
    identities = _validated_fight_identity_rows(
        identities,
        label="selection exposure",
    )
    _require_global_store_durable()
    key_material = _selection_exposure_key_material(identities)
    exposure_key = _canonical_json_sha256(key_material)
    path = _selection_exposure_path(identities)
    if path.exists():
        payload = _read_json(path)
        _validate_selection_exposure_payload(payload, path=path)
        # DIR-NRA-P3-012: the durable-store sweep above already covers this
        # file, but the existing-ledger branch must not depend on that sweep
        # staying upstream — re-anchor the exact ledger path before reuse.
        require_remotely_anchored_git_artifacts(
            _repo_root(),
            [path],
            label="selection exposure ledger",
        )
        return payload, path
    dates = [row["event_date"] for row in identities]
    payload = {
        "schema_version": 1,
        "protocol": _CONFIRMATION_PROTOCOL,
        "status": "selection_exposed",
        "selection_exposure_key": exposure_key,
        "selection_exposure_key_material": key_material,
        "selection_evaluation_fight_identity_sha256": key_material[
            "selection_evaluation_fight_identity_sha256"
        ],
        "selection_fight_identities": identities,
        "selection_min_event_date": min(dates),
        "selection_max_event_date": max(dates),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_exclusive_fsync(path, payload)
    # This intentionally stops the first invocation.  Selection may proceed
    # only after an operator commits and pushes the exact exposure ledger.
    require_remotely_anchored_git_artifacts(
        _repo_root(),
        [path],
        label="selection exposure ledger",
    )
    return payload, path


def _preflight_selection_fold_manifest(
    fold_manifest: list[tuple[int, pd.DataFrame]],
    *,
    confirmation_fold_count: int = DEFAULT_CONFIRMATION_FOLD_COUNT,
) -> str:
    """Anchor every selection identity before a model may fit or score it."""
    selection_manifest, _confirmation_manifest = partition_walk_forward_folds(
        fold_manifest,
        confirmation_fold_count=confirmation_fold_count,
    )
    selection_index = _post_cutoff_predictions(selection_manifest)
    identities = _canonical_evaluation_fight_identities(selection_index)
    _record_global_selection_exposure(identities)
    return _canonical_json_sha256(identities)


def _bound_selection_exposure(
    selection_evidence: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    identities = _validated_fight_identity_rows(
        selection_evidence.get("selection_fight_identities"),
        label="confirmation-bound selection exposure",
    )
    if _canonical_json_sha256(identities) != selection_evidence.get(
        "evaluation_fight_identity_sha256"
    ):
        raise ValueError("confirmation selection-exposure identity hash differs")
    path = _selection_exposure_path(identities)
    if not path.is_file():
        raise ValueError("confirmation requires the durable selection-exposure ledger")
    payload = _read_json(path)
    if _validate_selection_exposure_payload(payload, path=path) != identities:
        raise ValueError("confirmation selection-exposure ledger differs")
    return payload, path


def _prior_exposed_fight_identities(
    store: Path,
    *,
    requested_hashes: set[str] | None = None,
) -> list[dict[str, str]]:
    prior: dict[str, dict[str, str]] = {}
    exposure_dir = store / GLOBAL_SELECTION_EXPOSURES_DIRNAME
    for exposure_path in exposure_dir.glob("*.json"):
        for identity in _validate_selection_exposure_payload(
            _read_json(exposure_path),
            path=exposure_path,
        ):
            prior[_canonical_json_sha256(identity)] = identity
    for prior_claim_path in store.glob(f"*/{CONFIRMATION_CLAIM_FILENAME}"):
        prior_claim = _read_json(prior_claim_path)
        if (
            prior_claim.get("schema_version") != 1
            or prior_claim.get("protocol") != _CONFIRMATION_PROTOCOL
            or prior_claim.get("status") != "claimed"
        ):
            raise ValueError("prior confirmation claim has invalid semantics")
        for identity in _validated_fight_identity_rows(
            prior_claim.get("confirmation_fight_identities"),
            label="prior confirmation claim",
        ):
            prior[_canonical_json_sha256(identity)] = identity
    reservation_dir = store / "_fight_reservations"
    for marker_path in reservation_dir.glob("*.json"):
        marker = _read_json(marker_path)
        identity = marker.get("fight_identity") if isinstance(marker, dict) else None
        rows = _validated_fight_identity_rows(
            [identity] if isinstance(identity, dict) else identity,
            label="confirmation reservation",
        )
        identity_sha256 = _canonical_json_sha256(rows[0])
        if (
            marker.get("schema_version") != 1
            or marker.get("protocol") != _CONFIRMATION_PROTOCOL
            or marker.get("fight_identity_sha256") != identity_sha256
            or marker_path.name != f"{identity_sha256}.json"
        ):
            raise ValueError("confirmation reservation marker is malformed")
        # A directly overlapping orphan marker is deliberately left to the
        # O_EXCL write below.  That atomic path is the race authority; disjoint
        # orphan dates still participate in the forward-window check.
        if identity_sha256 not in (requested_hashes or set()):
            prior[identity_sha256] = rows[0]
    return sorted(
        prior.values(),
        key=lambda row: (
            row["event_date"],
            row["fighter_name_low"],
            row["fighter_name_high"],
        ),
    )


def _reserve_global_confirmation_fights(
    *,
    global_claim_key: str,
    identities: list[dict[str, str]],
) -> None:
    """Atomically consume every bout identity and reject any prior overlap."""
    identities = _validated_fight_identity_rows(
        identities,
        label="confirmation claim",
    )
    store = _global_confirmation_store()
    requested = {_canonical_json_sha256(row) for row in identities}
    prior_identities = _prior_exposed_fight_identities(
        store,
        requested_hashes=requested,
    )
    prior_hashes = {_canonical_json_sha256(row) for row in prior_identities}
    if requested.intersection(prior_hashes):
        raise ValueError(
            "confirmation fight set overlaps a previously exposed selection or claim"
        )
    if prior_identities:
        requested_min_date = min(row["event_date"] for row in identities)
        prior_max_date = max(row["event_date"] for row in prior_identities)
        if requested_min_date <= prior_max_date:
            raise ValueError(
                "confirmation window must be strictly forward of all prior exposures"
            )
    reservation_dir = store / "_fight_reservations"
    reservation_dir.mkdir(parents=True, exist_ok=True)
    for identity in identities:
        identity_sha256 = _canonical_json_sha256(identity)
        marker_path = reservation_dir / f"{identity_sha256}.json"
        try:
            _write_json_exclusive_fsync(
                marker_path,
                {
                    "schema_version": 1,
                    "protocol": _CONFIRMATION_PROTOCOL,
                    "global_claim_key": global_claim_key,
                    "fight_identity_sha256": identity_sha256,
                    "fight_identity": identity,
                },
            )
        except FileExistsError as exc:
            raise ValueError(
                "confirmation fight identity was already consumed"
            ) from exc


def _claim_confirmation_once(
    *,
    run_dir: Path,
    lock_payload: dict[str, Any],
    lock_sha256: str,
) -> tuple[dict[str, Any], str, Path]:
    """Consume the sole confirmation attempt before any fit or prediction."""
    local_lock_path = run_dir / CONFIRMATION_LOCK_FILENAME
    if not local_lock_path.is_file() or _file_sha256(local_lock_path) != lock_sha256:
        raise ValueError("confirmation claim requires the exact local package lock")
    if (run_dir / CONFIRMATION_RESULT_FILENAME).exists():
        raise ValueError("Confirmation result already exists; the reserved sample is consumed")
    local_reference_path = run_dir / CONFIRMATION_CLAIM_REFERENCE_FILENAME
    _require_global_store_durable()
    # Consumption is keyed only to the orientation-invariant bout identities,
    # never to fold metadata, labels, inputs, code, policy, candidate, or
    # control.  The per-fight reservations below additionally refuse subsets,
    # supersets, and any other overlap with a prior confirmation window.
    key_material = {
        "key_contract": "confirmation_sample_consumption_v1",
        "confirmation_evaluation_fight_identity_sha256": lock_payload[
            "selection_evidence"
        ]["confirmation_evaluation_fight_identity_sha256"],
    }
    global_claim_key = _canonical_json_sha256(key_material)
    selection_exposure, selection_exposure_path = _bound_selection_exposure(
        lock_payload["selection_evidence"]
    )
    confirmation_identities = lock_payload["selection_evidence"][
        "confirmation_fight_identities"
    ]
    claim_dir = (_global_confirmation_store() / global_claim_key).resolve()
    claim_path = claim_dir / CONFIRMATION_CLAIM_FILENAME
    global_lock_path = claim_dir / CONFIRMATION_LOCK_FILENAME
    global_result_path = claim_dir / CONFIRMATION_RESULT_FILENAME
    if local_reference_path.exists():
        reference = _read_json(local_reference_path)
        if (
            reference.get("schema_version") != 1
            or reference.get("protocol") != _CONFIRMATION_PROTOCOL
            or reference.get("global_claim_key") != global_claim_key
            or _resolve_repo_artifact_path(reference.get("claim_path")) != claim_path
            or not claim_path.is_file()
            or _file_sha256(claim_path) != reference.get("claim_sha256")
            or not global_lock_path.is_file()
            or _file_sha256(global_lock_path) != lock_sha256
            or global_result_path.exists()
        ):
            raise ValueError("existing confirmation claim reference is not resumable")
        claim = _read_json(claim_path)
        if (
            claim.get("schema_version") != 1
            or claim.get("protocol") != _CONFIRMATION_PROTOCOL
            or claim.get("status") != "claimed"
            or claim.get("global_claim_key") != global_claim_key
            or claim.get("lock_sha256") != lock_sha256
            or claim.get("selection_exposure_key")
            != selection_exposure["selection_exposure_key"]
            or _resolve_repo_artifact_path(claim.get("selection_exposure_path"))
            != selection_exposure_path
            or claim.get("selection_exposure_sha256")
            != _file_sha256(selection_exposure_path)
        ):
            raise ValueError("existing durable confirmation claim changed")
        _require_global_store_durable()
        return claim, _file_sha256(claim_path), claim_path
    _reserve_global_confirmation_fights(
        global_claim_key=global_claim_key,
        identities=confirmation_identities,
    )
    claim_dir.mkdir(parents=True, exist_ok=True)
    if claim_path.exists() or global_result_path.exists():
        raise ValueError(
            "The canonical confirmation sample/control claim is already consumed"
        )
    if global_lock_path.exists():
        if _file_sha256(global_lock_path) != lock_sha256:
            raise ValueError(
                "The canonical confirmation sample was prelocked to a different package"
            )
    else:
        _copy_file_exclusive_fsync(local_lock_path, global_lock_path)
    claim = {
        "schema_version": 1,
        "protocol": _CONFIRMATION_PROTOCOL,
        "status": "claimed",
        "global_claim_key": global_claim_key,
        "global_claim_key_material": key_material,
        "claim_path": _repo_relative_artifact_path(claim_path),
        "global_result_path": _repo_relative_artifact_path(global_result_path),
        "lock_path": _repo_relative_artifact_path(global_lock_path),
        "lock_sha256": lock_sha256,
        "package_id": lock_payload["package_id"],
        "source_fingerprint": lock_payload["source_fingerprint"],
        "frozen_control_receipt_sha256": lock_payload["frozen_control"][
            "bootstrap_receipt_sha256"
        ],
        "full_evaluation_sample_sha256": lock_payload["selection_evidence"][
            "full_evaluation_sample_sha256"
        ],
        "selection_evaluation_sample_sha256": lock_payload["selection_evidence"][
            "evaluation_sample_sha256"
        ],
        "confirmation_evaluation_sample_sha256": lock_payload[
            "selection_evidence"
        ]["confirmation_evaluation_sample_sha256"],
        "confirmation_evaluation_fight_identity_sha256": lock_payload[
            "selection_evidence"
        ]["confirmation_evaluation_fight_identity_sha256"],
        "confirmation_fight_identities": confirmation_identities,
        "selection_exposure_key": selection_exposure["selection_exposure_key"],
        "selection_exposure_path": _repo_relative_artifact_path(
            selection_exposure_path
        ),
        "selection_exposure_sha256": _file_sha256(selection_exposure_path),
        "confirmation_evaluation_input_value_sha256": lock_payload[
            "selection_evidence"
        ]["confirmation_evaluation_input_value_sha256"],
        "confirmation_evaluation_n_fights": lock_payload["selection_evidence"][
            "confirmation_evaluation_n_fights"
        ],
        "confirmation_fold_ids": lock_payload["selection_evidence"][
            "confirmation_fold_ids"
        ],
        "claimed_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json_exclusive_fsync(claim_path, claim)
    claim_sha256 = _file_sha256(claim_path)
    _write_json_exclusive_fsync(
        local_reference_path,
        {
            "schema_version": 1,
            "protocol": _CONFIRMATION_PROTOCOL,
            "global_claim_key": global_claim_key,
            "claim_path": _repo_relative_artifact_path(claim_path),
            "claim_sha256": claim_sha256,
        },
    )
    # The initial invocation stops here until the global claim, lock, and
    # reservations are committed and pushed.  A rerun resumes only this exact
    # reference, so no confirmation fit can precede the durable claim.
    _require_global_store_durable()
    return claim, claim_sha256, claim_path


def _sweep_config_from_locked_package(package: dict[str, Any]) -> SweepConfig:
    config_payload = dict(package["strategy_config"])
    if config_payload.get("c_max_decimal_odds") is None:
        config_payload["c_max_decimal_odds"] = float("inf")
    config_payload["variant_name"] = package["model_variant"]
    return SweepConfig(**config_payload)


def _generate_locked_confirmation_predictions(
    *,
    package: dict[str, Any],
    protocol: dict[str, Any],
    claim_path: Path,
    claim_sha256: str,
    features_df: pd.DataFrame,
) -> tuple[list[tuple[int, pd.DataFrame]], list[tuple[int, pd.DataFrame]]]:
    """Fit one named package only on the claimed confirmation folds."""
    named_spec = resolve_named_training_spec(package["model_spec_name"])
    registered_hash = _canonical_json_sha256(asdict(named_spec))
    if registered_hash != package["model_spec_payload_sha256"]:
        raise ValueError("Locked package named spec changed after the package lock")

    variant_name = package["model_variant"]
    original_factory = ALL_VARIANTS[variant_name]
    ALL_VARIANTS[variant_name] = lambda: variant_from_named_training_spec(
        named_spec.name,
        variant_name=variant_name,
    )
    try:
        generated = _generate_walk_forward_predictions(
            features_df=features_df,
            bet_start_date=protocol["bet_start_date"],
            variant_name=variant_name,
            dataset_variant=package["dataset_variant"],
            feature_family=package["feature_family"],
            feature_cols=list(named_spec.feature_cols),
            calibration_method=package["calibration_method"],
            retrain_months=int(package["retrain_months"]),
            entry_offset_days=protocol["entry_offset_days"],
            entry_offset_for_features=bool(
                protocol["entry_offset_for_features"]
            ),
            require_entry_odds=bool(protocol["require_entry_odds"]),
            allow_closing_odds=bool(protocol["allow_closing_odds"]),
            evaluation_partition="confirmation",
            confirmation_fold_count=int(
                protocol["reserved_confirmation_folds"]
            ),
            return_fold_manifest=True,
            confirmation_claim_sha256=claim_sha256,
            confirmation_claim_path=claim_path,
        )
    finally:
        ALL_VARIANTS[variant_name] = original_factory
    confirmation_predictions, fold_manifest = generated
    return confirmation_predictions, fold_manifest


def _validate_confirmation_prediction_binding(
    *,
    confirmation_predictions: list[tuple[int, pd.DataFrame]],
    fold_manifest: list[tuple[int, pd.DataFrame]],
    expected_selection_evidence: dict[str, Any],
    feature_contract_columns: list[str],
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selection_manifest, confirmation_manifest = _partition_evaluation_folds(
        fold_manifest,
        reserved_confirmation_folds=DEFAULT_CONFIRMATION_FOLD_COUNT,
    )
    full_index = _post_cutoff_predictions(fold_manifest)
    selection_index = _post_cutoff_predictions(selection_manifest)
    confirmation_index = _post_cutoff_predictions(confirmation_manifest)
    scored_confirmation = _post_cutoff_predictions(confirmation_predictions)
    if scored_confirmation.empty:
        raise ValueError(f"{label} produced no confirmation predictions")
    actual = _fold_partition_evidence(
        fold_manifest,
        selection_manifest,
        reserved_confirmation_folds=DEFAULT_CONFIRMATION_FOLD_COUNT,
        feature_contract_columns=feature_contract_columns,
    )
    expected = _selection_binding_from_payload(expected_selection_evidence)
    expected.pop("evaluation_partition", None)
    expected.pop("reserved_confirmation_folds", None)
    mismatches = [
        field
        for field, expected_value in expected.items()
        if field in actual and actual.get(field) != expected_value
    ]
    scored_hash = _evaluation_sample_sha256(scored_confirmation)
    if scored_hash != actual["confirmation_evaluation_sample_sha256"]:
        mismatches.append("scored_confirmation_index")
    scored_fold_ids = [fold_id for fold_id, _frame in confirmation_predictions]
    if scored_fold_ids != actual["confirmation_fold_ids"]:
        mismatches.append("scored_confirmation_fold_ids")
    if mismatches:
        raise ValueError(
            f"{label} confirmation binding differs from the locked manifest: "
            + ", ".join(mismatches)
        )
    return scored_confirmation, actual


def _confirmation_model_payload(
    predictions: pd.DataFrame,
    *,
    sample_sha256: str,
    fold_ids: list[int],
    bootstrap: int,
    seed_key: str,
    evaluation_input_value_sha256: str,
    model_spec_name: str,
    model_spec_payload_sha256: str,
    calibration_method: str,
) -> dict[str, Any]:
    metrics = _compute_model_metrics(
        predictions,
        bootstrap=bootstrap,
        seed=_seed_from_key(seed_key),
    )
    sliced_metrics = _compute_sliced_metrics(predictions, predictions)
    prediction_rows_sha256 = _prediction_rows_sha256(predictions)
    prediction_values_sha256 = _prediction_values_sha256(predictions)
    evidence = {
        "evaluation_input_value_sha256": evaluation_input_value_sha256,
        "prediction_rows_sha256": prediction_rows_sha256,
        "prediction_values_sha256": prediction_values_sha256,
        "model_spec_name": model_spec_name,
        "model_spec_payload_sha256": model_spec_payload_sha256,
        "calibration_method": calibration_method,
    }
    metrics.update(evidence)
    return {
        "metrics": metrics,
        "sliced_metrics": sliced_metrics,
        "evaluation_partition": "confirmation",
        "evaluation_sample_sha256": sample_sha256,
        "confirmation_fold_ids": fold_ids,
        "n_predictions": len(predictions),
        "n_folds": len(fold_ids),
        **evidence,
    }


def _persist_durable_confirmation_dependencies(
    *,
    claim_payload: dict[str, Any],
    claim_path: Path,
    lock_payload: dict[str, Any],
    confirmation_dir: Path,
    confirmation_artifact_names: set[str],
) -> tuple[Path, dict[str, dict[str, str]]]:
    """Copy every ignored/run-local confirmation dependency into global evidence."""
    claim_dir = claim_path.parent
    global_lock_path = _resolve_repo_artifact_path(claim_payload["lock_path"])
    allowed_existing = {claim_path.resolve(), global_lock_path.resolve()}
    existing = {path.resolve() for path in claim_dir.rglob("*") if path.is_file()}
    if existing != allowed_existing:
        raise ValueError(
            "canonical confirmation evidence directory contains stale files before publish"
        )

    sources: list[tuple[str, Path, str]] = []
    for name in sorted(confirmation_artifact_names):
        path = (confirmation_dir / name).resolve()
        sources.append((f"confirmation/{name}", path, _file_sha256(path)))
    for field, hash_field, logical_name in (
        ("package_manifest_path", "package_manifest_sha256", "selection/package_manifest.json"),
        ("predeclared_experiment_manifest_path", "predeclared_experiment_manifest_sha256", "selection/experiment_manifest.json"),
        ("selection_result_path", "selection_result_sha256", "selection/winner_result.json"),
        ("bounded_selection_summary_path", "bounded_selection_summary_sha256", "selection/summary.json"),
        ("selection_diagnostics_path", "selection_diagnostics_sha256", "selection/winner_diagnostics.json"),
        ("selection_bet_log_path", "selection_bet_log_sha256", "selection/winner_bet_log.csv"),
        ("policy_path", "policy_sha256", "policy/scheduled_refit_policy.json"),
    ):
        source = Path(str(lock_payload[field])).resolve()
        sources.append((logical_name, source, str(lock_payload[hash_field])))
    provenance = lock_payload["candidate_input_provenance"]
    for field, hash_field, logical_name in (
        ("dataset_fights_path", "dataset_fights_sha256", "inputs/fights_cleaned.csv"),
        ("features_artifact_path", "features_artifact_sha256", "inputs/features.csv"),
        ("source_inventory_path", "source_inventory_artifact_sha256", "inputs/evaluation_source_inventory.json"),
        ("environment_path", "environment_artifact_sha256", "inputs/evaluation_environment.json"),
    ):
        source = Path(str(provenance[field])).resolve()
        sources.append((logical_name, source, str(provenance[hash_field])))
    for index, entry in enumerate(
        lock_payload["candidate_odds_source_inventory"].get("entries", [])
    ):
        source = Path(str(entry["resolved_path"])).resolve()
        sources.append(
            (
                f"inputs/odds/{index:04d}_{_safe_name(source.name)}",
                source,
                str(entry["sha256"]),
            )
        )
    frozen = lock_payload["frozen_control"]
    for field, hash_field, logical_name in (
        ("checksums_path", "checksums_sha256", "control/checksums.json"),
        ("bootstrap_receipt_path", "bootstrap_receipt_sha256", "control/fixed_control_bootstrap_receipt.json"),
        ("input_provenance_path", "input_provenance_sha256", "control/fixed_control_input_provenance.json"),
        ("odds_source_inventory_path", "odds_source_inventory_sha256", "control/fixed_control_odds_source_inventory.json"),
        ("source_inventory_path", "source_inventory_artifact_sha256", "control/evaluation_source_inventory.json"),
        ("environment_path", "environment_artifact_sha256", "control/evaluation_environment.json"),
    ):
        raw = frozen.get(field)
        if raw is None and field == "checksums_path":
            raw = str(Path(frozen["path"]) / "checksums.json")
        source = Path(str(raw)).resolve()
        sources.append((logical_name, source, str(frozen[hash_field])))

    seen_destinations: set[str] = set()
    inventory_entries = [
        {
            "logical_name": "confirmation_claim",
            "path": _repo_relative_artifact_path(claim_path),
            "sha256": _file_sha256(claim_path),
        },
        {
            "logical_name": "confirmation_lock",
            "path": _repo_relative_artifact_path(global_lock_path),
            "sha256": _file_sha256(global_lock_path),
        },
    ]
    durable_odds_inventory_path = (
        claim_dir / "artifacts" / "inputs" / "odds_source_inventory.json"
    )
    durable_odds_inventory = lock_payload["candidate_odds_source_inventory"]
    _write_json_artifact_exclusive_fsync(
        durable_odds_inventory_path,
        durable_odds_inventory,
    )
    durable_odds_sha256 = _file_sha256(durable_odds_inventory_path)
    expected_odds_sha256 = lock_payload["selection_evidence"].get(
        "odds_source_inventory_sha256"
    )
    if durable_odds_sha256 != expected_odds_sha256:
        raise ValueError("durable candidate odds-source inventory hash differs")
    odds_destination = "artifacts/inputs/odds_source_inventory.json"
    seen_destinations.add(odds_destination)
    inventory_entries.append(
        {
            "logical_name": "inputs/odds_source_inventory.json",
            "path": _repo_relative_artifact_path(durable_odds_inventory_path),
            "sha256": durable_odds_sha256,
        }
    )
    published_artifacts: dict[str, dict[str, str]] = {}
    for logical_name, source, expected_sha256 in sources:
        if not source.is_file() or _file_sha256(source) != expected_sha256:
            raise ValueError(f"durable confirmation dependency changed: {logical_name}")
        relative_destination = f"artifacts/{logical_name}"
        if relative_destination in seen_destinations:
            raise ValueError(f"duplicate durable evidence destination: {relative_destination}")
        seen_destinations.add(relative_destination)
        destination = claim_dir / Path(relative_destination)
        _copy_file_exclusive_fsync(source, destination)
        binding = {
            "path": _repo_relative_artifact_path(destination),
            "sha256": _file_sha256(destination),
        }
        inventory_entries.append(
            {"logical_name": logical_name, **binding}
        )
        if logical_name.startswith("confirmation/"):
            published_artifacts[Path(logical_name).name] = binding

    inventory = {
        "schema_version": 1,
        "protocol": _CONFIRMATION_PROTOCOL,
        "global_claim_key": claim_payload["global_claim_key"],
        "entries": inventory_entries,
    }
    inventory_path = claim_dir / "confirmation_evidence_inventory.json"
    _write_json_exclusive_fsync(inventory_path, inventory)
    return inventory_path, published_artifacts


def _run_locked_confirmation_stage4(
    *,
    run_dir: Path,
    freeze_id: str,
    metadata: dict[str, Any],
    package_manifest_path: Path,
) -> list[dict[str, Any]]:
    """Evaluate one locked package/control pair once on the two-fold reserve."""
    readiness = validate_frozen_control_arm_for_promotion_gate(freeze_id)
    if not readiness["ready"]:
        raise ValueError(
            f"Frozen control {freeze_id} is not ready for confirmation: "
            f"{readiness['errors']}"
        )
    control_model_payload, raw_control_sweep = _load_control_arm_payloads(freeze_id)
    if not isinstance(raw_control_sweep, dict):
        raise ValueError("Frozen control is missing its strategy package")
    confirmation_dir = run_dir / PROMOTION_OUTPUT_DIRNAME / "locked_confirmation"
    if confirmation_dir.exists():
        raise ValueError(
            "Locked confirmation output directory already exists; refuse stale artifacts"
        )
    lock_payload, lock_sha256 = _create_or_load_confirmation_lock(
        run_dir=run_dir,
        package_manifest_path=package_manifest_path,
        metadata=metadata,
        freeze_id=freeze_id,
        control_metrics=control_model_payload,
        control_sweep_payload=raw_control_sweep,
        control_validation=readiness,
    )

    # All static/package/control checks above complete before this exclusive,
    # durable claim. Any failure after this line consumes the sole attempt.
    claim_payload, claim_sha256, claim_path = _claim_confirmation_once(
        run_dir=run_dir,
        lock_payload=lock_payload,
        lock_sha256=lock_sha256,
    )
    protocol = lock_payload["evaluation_protocol"]
    candidate_package = lock_payload["candidate_package"]
    control_package = lock_payload["frozen_control"]["package"]
    locked_features = _validate_input_provenance_files(
        lock_payload["candidate_input_provenance"],
        lock_payload["candidate_odds_source_inventory"],
    )

    candidate_folds, candidate_manifest = _generate_locked_confirmation_predictions(
        package=candidate_package,
        protocol=protocol,
        claim_path=claim_path,
        claim_sha256=claim_sha256,
        features_df=locked_features,
    )
    control_folds, control_manifest = _generate_locked_confirmation_predictions(
        package=control_package,
        protocol=protocol,
        claim_path=claim_path,
        claim_sha256=claim_sha256,
        features_df=locked_features,
    )
    expected_selection = lock_payload["selection_evidence"]
    candidate_predictions, candidate_binding = _validate_confirmation_prediction_binding(
        confirmation_predictions=candidate_folds,
        fold_manifest=candidate_manifest,
        expected_selection_evidence=expected_selection,
        feature_contract_columns=list(expected_selection["feature_contract_columns"]),
        label="Candidate",
    )
    control_predictions, control_binding = _validate_confirmation_prediction_binding(
        confirmation_predictions=control_folds,
        fold_manifest=control_manifest,
        expected_selection_evidence=expected_selection,
        feature_contract_columns=list(expected_selection["feature_contract_columns"]),
        label="Control",
    )
    if candidate_binding != control_binding:
        raise ValueError("Candidate/control confirmation manifests are not identical")

    confirmation_sha256 = candidate_binding[
        "confirmation_evaluation_sample_sha256"
    ]
    confirmation_fold_ids = candidate_binding["confirmation_fold_ids"]
    candidate_model = _confirmation_model_payload(
        candidate_predictions,
        sample_sha256=confirmation_sha256,
        fold_ids=confirmation_fold_ids,
        bootstrap=int(protocol["bootstrap"]),
        seed_key=f"confirmation:candidate:{lock_payload['package_id']}",
        evaluation_input_value_sha256=candidate_binding[
            "confirmation_evaluation_input_value_sha256"
        ],
        model_spec_name=candidate_package["model_spec_name"],
        model_spec_payload_sha256=candidate_package["model_spec_payload_sha256"],
        calibration_method=candidate_package["calibration_method"],
    )
    control_model = _confirmation_model_payload(
        control_predictions,
        sample_sha256=confirmation_sha256,
        fold_ids=confirmation_fold_ids,
        bootstrap=int(protocol["bootstrap"]),
        seed_key=f"confirmation:control:{freeze_id}",
        evaluation_input_value_sha256=control_binding[
            "confirmation_evaluation_input_value_sha256"
        ],
        model_spec_name=control_package["model_spec_name"],
        model_spec_payload_sha256=control_package["model_spec_payload_sha256"],
        calibration_method=control_package["calibration_method"],
    )
    candidate_model["model_unchanged_declared"] = all(
        candidate_model[field] == control_model[field]
        for field in (
            "model_spec_name",
            "model_spec_payload_sha256",
            "calibration_method",
            "evaluation_input_value_sha256",
            "prediction_rows_sha256",
            "prediction_values_sha256",
        )
    )
    candidate_model["metrics"]["model_unchanged_declared"] = candidate_model[
        "model_unchanged_declared"
    ]

    candidate_config = _sweep_config_from_locked_package(candidate_package)
    control_config = _sweep_config_from_locked_package(control_package)
    candidate_sweep = _evaluate_config(
        candidate_folds,
        candidate_config,
        initial_bankroll=INITIAL_BANKROLL,
        bet_start_date=protocol["bet_start_date"],
        execution_mode=protocol["execution_mode"],
    )
    control_sweep = _evaluate_config(
        control_folds,
        control_config,
        initial_bankroll=INITIAL_BANKROLL,
        bet_start_date=protocol["bet_start_date"],
        execution_mode=protocol["execution_mode"],
    )
    for sweep in (candidate_sweep, control_sweep):
        sweep["evaluation_partition"] = "confirmation"
        sweep["evaluation_sample_sha256"] = confirmation_sha256
        sweep["confirmation_fold_ids"] = confirmation_fold_ids
        sweep["confirmation_lock_sha256"] = lock_sha256
        sweep["confirmation_claim_sha256"] = claim_sha256
        sweep["promotion_eligible"] = True

    candidate_gate_metrics = _canonicalize_promotion_model_metrics(candidate_model)
    control_gate_metrics = _canonicalize_promotion_model_metrics(control_model)
    candidate_gate_sweep = _canonicalize_sweep_payload(
        candidate_sweep,
        source_path="locked_confirmation:candidate",
    )
    control_gate_sweep = _canonicalize_sweep_payload(
        control_sweep,
        source_path="locked_confirmation:control",
    )
    verdict = evaluate_for_promotion(
        candidate_gate_metrics,
        control_gate_metrics,
        candidate_gate_sweep,
        control_gate_sweep,
    )

    confirmation_dir.mkdir(parents=True, exist_ok=True)
    _write_json(confirmation_dir / "candidate_model_metrics.json", candidate_model)
    _write_json(confirmation_dir / "control_model_metrics.json", control_model)
    _save_raw_sweep_result(confirmation_dir, "candidate", candidate_sweep)
    _save_raw_sweep_result(confirmation_dir, "control", control_sweep)
    report = generate_promotion_report(
        candidate_gate_metrics,
        control_gate_metrics,
        candidate_gate_sweep,
        control_gate_sweep,
    )
    report_path = confirmation_dir / "promotion_report.md"
    report_path.write_text(report, encoding="utf-8")

    expected_artifact_names = {
        "candidate_model_metrics.json",
        "control_model_metrics.json",
        "candidate_result.json",
        "candidate_bet_log.csv",
        "candidate_bankroll_history.csv",
        "control_result.json",
        "control_bet_log.csv",
        "control_bankroll_history.csv",
        "promotion_report.md",
    }
    actual_artifact_names = {
        path.name for path in confirmation_dir.iterdir() if path.is_file()
    }
    if actual_artifact_names != expected_artifact_names:
        raise ValueError(
            "Locked confirmation artifact set is incomplete or contains extras: "
            f"expected={sorted(expected_artifact_names)}, "
            f"actual={sorted(actual_artifact_names)}"
        )
    artifact_paths = sorted(confirmation_dir / name for name in expected_artifact_names)
    evidence_inventory_path, durable_artifacts = (
        _persist_durable_confirmation_dependencies(
            claim_payload=claim_payload,
            claim_path=claim_path,
            lock_payload=lock_payload,
            confirmation_dir=confirmation_dir,
            confirmation_artifact_names=expected_artifact_names,
        )
    )
    confirmation_passed = verdict.get("verdict") == "PROMOTE"
    durable_inputs_dir = claim_path.parent / "artifacts" / "inputs"
    durable_fights_path = durable_inputs_dir / "fights_cleaned.csv"
    durable_features_path = durable_inputs_dir / "features.csv"
    durable_odds_inventory_path = durable_inputs_dir / "odds_source_inventory.json"
    durable_source_inventory_path = (
        durable_inputs_dir / "evaluation_source_inventory.json"
    )
    durable_environment_path = durable_inputs_dir / "evaluation_environment.json"
    # DIR-NRA-P3-013: record which pushed origin default-branch tip anchored
    # the consumed claim/lock so post-hoc validation can pin the completion to
    # real remote history rather than the local checkout alone.
    anchored_origin_tip_sha256 = require_remotely_anchored_git_artifacts(
        _repo_root(),
        [
            _resolve_repo_artifact_path(claim_payload["claim_path"]),
            _resolve_repo_artifact_path(claim_payload["lock_path"]),
        ],
        label="confirmation completion",
    )
    result_payload = {
        "schema_version": 1,
        "protocol": _CONFIRMATION_PROTOCOL,
        "status": "complete",
        "confirmation_passed": confirmation_passed,
        "confirmation_outcome": "pass" if confirmation_passed else "fail",
        "package_id": lock_payload["package_id"],
        "lock_path": claim_payload["lock_path"],
        "lock_sha256": lock_sha256,
        "global_claim_key": claim_payload["global_claim_key"],
        "claim_path": claim_payload["claim_path"],
        "claim_sha256": claim_sha256,
        "claim": claim_payload,
        "anchored_origin_tip_sha256": anchored_origin_tip_sha256,
        "evaluation_partition": "confirmation",
        "evaluation_sample_sha256": confirmation_sha256,
        "confirmation_evaluation_fight_identity_sha256": expected_selection[
            "confirmation_evaluation_fight_identity_sha256"
        ],
        "confirmation_fold_ids": confirmation_fold_ids,
        "candidate_control_index_identical": True,
        "selection_evidence": expected_selection,
        "candidate_package_contract_sha256": lock_payload[
            "candidate_package_contract_sha256"
        ],
        "candidate_package": candidate_package,
        "candidate_model_spec_name": candidate_package["model_spec_name"],
        "candidate_model_spec_payload_sha256": candidate_package[
            "model_spec_payload_sha256"
        ],
        "candidate_fullfit_model_spec_name": candidate_package[
            "fullfit_model_spec_name"
        ],
        "candidate_fullfit_model_spec_payload_sha256": candidate_package[
            "fullfit_model_spec_payload_sha256"
        ],
        "candidate_strategy_config_sha256": _canonical_json_sha256(
            candidate_package["strategy_config"]
        ),
        "candidate_calibration_method": candidate_package["calibration_method"],
        "candidate_runtime_invariants": candidate_package["runtime_invariants"],
        "candidate_model_unchanged_declared": candidate_model[
            "model_unchanged_declared"
        ],
        "candidate_prediction_rows_sha256": candidate_model[
            "prediction_rows_sha256"
        ],
        "candidate_prediction_values_sha256": candidate_model[
            "prediction_values_sha256"
        ],
        "control_model_spec_name": control_model["model_spec_name"],
        "control_model_spec_payload_sha256": control_model[
            "model_spec_payload_sha256"
        ],
        "control_calibration_method": control_model["calibration_method"],
        "control_prediction_rows_sha256": control_model[
            "prediction_rows_sha256"
        ],
        "control_prediction_values_sha256": control_model[
            "prediction_values_sha256"
        ],
        "candidate_selection_result_sha256": lock_payload[
            "selection_result_sha256"
        ],
        "feature_contract_count": expected_selection["feature_contract_count"],
        "feature_contract_sha256": expected_selection[
            "feature_contract_sha256"
        ],
        "dataset_fights_path": _repo_relative_artifact_path(
            durable_fights_path
        ),
        "dataset_fights_sha256": expected_selection["dataset_fights_sha256"],
        "source_dataset_fights_path": _repo_relative_artifact_path(
            durable_fights_path
        ),
        "source_dataset_fights_sha256": expected_selection[
            "source_dataset_fights_sha256"
        ],
        "features_artifact_path": _repo_relative_artifact_path(
            durable_features_path
        ),
        "features_artifact_sha256": expected_selection[
            "features_artifact_sha256"
        ],
        "features_value_sha256": expected_selection["features_value_sha256"],
        "source_features_path": _repo_relative_artifact_path(
            durable_features_path
        ),
        "source_features_sha256": expected_selection["source_features_sha256"],
        "confirmation_evaluation_input_value_sha256": expected_selection[
            "confirmation_evaluation_input_value_sha256"
        ],
        "odds_source_inventory_path": _repo_relative_artifact_path(
            durable_odds_inventory_path
        ),
        "odds_source_inventory_sha256": expected_selection[
            "odds_source_inventory_sha256"
        ],
        "policy_sha256": lock_payload["policy_sha256"],
        "scheduled_protocol_sha256": lock_payload[
            "scheduled_protocol_sha256"
        ],
        "evaluation_protocol": lock_payload["evaluation_protocol"],
        "evaluation_protocol_sha256": lock_payload[
            "evaluation_protocol_sha256"
        ],
        "frozen_control_receipt_sha256": lock_payload["frozen_control"][
            "bootstrap_receipt_sha256"
        ],
        "source_fingerprint": lock_payload["source_fingerprint"],
        "source_inventory_path": _repo_relative_artifact_path(
            durable_source_inventory_path
        ),
        "source_inventory_sha256": expected_selection["source_inventory_sha256"],
        "source_inventory_artifact_sha256": expected_selection[
            "source_inventory_artifact_sha256"
        ],
        "environment_path": _repo_relative_artifact_path(
            durable_environment_path
        ),
        "environment_artifact_sha256": expected_selection[
            "environment_artifact_sha256"
        ],
        "environment_payload_sha256": expected_selection[
            "environment_payload_sha256"
        ],
        "prior_all_fold_baseline_visibility": True,
        "baseline_visibility_limitation": lock_payload[
            "baseline_visibility_limitation"
        ],
        "verdict": verdict,
        "evidence_inventory_path": _repo_relative_artifact_path(
            evidence_inventory_path
        ),
        "evidence_inventory_sha256": _file_sha256(evidence_inventory_path),
        "artifacts": durable_artifacts,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    result_path = run_dir / CONFIRMATION_RESULT_FILENAME
    global_result_path = _resolve_repo_artifact_path(
        claim_payload["global_result_path"]
    )
    _write_json_exclusive_fsync(global_result_path, result_payload)
    completion_path = global_result_path.with_name(CONFIRMATION_COMPLETION_FILENAME)
    completion_payload = {
        "schema_version": 1,
        "protocol": _CONFIRMATION_PROTOCOL,
        "global_claim_key": claim_payload["global_claim_key"],
        "claim_sha256": claim_sha256,
        "lock_sha256": lock_sha256,
        "result_path": _repo_relative_artifact_path(global_result_path),
        "result_sha256": _file_sha256(global_result_path),
        "verdict_sha256": _canonical_json_sha256(verdict),
        "completed_at": result_payload["completed_at"],
    }
    _write_json_exclusive_fsync(completion_path, completion_payload)
    _write_json_exclusive_fsync(result_path, result_payload)
    verdict_row = {
        "candidate_id": lock_payload["package_id"],
        "model_variant": candidate_package["model_variant"],
        "blocked": False,
        "confirmation_lock_sha256": lock_sha256,
        "confirmation_claim_sha256": claim_sha256,
        **verdict,
    }
    _write_json(run_dir / "stage4_verdicts.json", [verdict_row])
    return [verdict_row]


def validate_confirmation_result_for_promotion(
    result_path: Path,
) -> dict[str, Any]:
    """Return the exact confirmed package binding or fail closed.

    Downstream full-fit/bundle code can consume this interface without
    interpreting exploratory Stage 1-3 artifacts.
    """
    result_path = result_path.resolve()
    if not result_path.is_file():
        raise ValueError(f"confirmation result is missing: {result_path}")
    try:
        result_relative = result_path.relative_to(_global_confirmation_store())
    except ValueError as exc:
        raise ValueError("confirmation result is outside the canonical global store") from exc
    if (
        len(result_relative.parts) != 2
        or result_relative.name != CONFIRMATION_RESULT_FILENAME
    ):
        raise ValueError("confirmation result does not use the canonical global path")
    _require_global_store_durable()
    result = _read_json(result_path)
    if (
        result.get("schema_version") != 1
        or result.get("protocol") != _CONFIRMATION_PROTOCOL
        or result.get("status") != "complete"
    ):
        raise ValueError("confirmation result has invalid completion semantics")
    require_recorded_anchor_in_history(
        _repo_root(),
        result.get("anchored_origin_tip_sha256"),
        label="confirmation result",
    )
    lock_path = _resolve_repo_artifact_path(result.get("lock_path"))
    claim_path = _resolve_repo_artifact_path(result.get("claim_path"))
    if (
        not lock_path.is_file()
        or _file_sha256(lock_path) != result.get("lock_sha256")
        or not claim_path.is_file()
        or _file_sha256(claim_path) != result.get("claim_sha256")
    ):
        raise ValueError("confirmation result lock/claim artifacts changed")
    lock = _read_json(lock_path)
    claim = _read_json(claim_path)
    selection = lock.get("selection_evidence")
    frozen_control = lock.get("frozen_control")
    if not isinstance(selection, dict) or not isinstance(frozen_control, dict):
        raise ValueError("confirmation lock is missing immutable selection/control evidence")
    expected_key_material = {
        "key_contract": "confirmation_sample_consumption_v1",
        "confirmation_evaluation_fight_identity_sha256": selection.get(
            "confirmation_evaluation_fight_identity_sha256"
        ),
    }
    expected_global_key = _canonical_json_sha256(expected_key_material)
    selection_exposure, selection_exposure_path = _bound_selection_exposure(selection)
    confirmed_identities = _validated_fight_identity_rows(
        selection.get("confirmation_fight_identities"),
        label="confirmed result",
    )
    if min(row["event_date"] for row in confirmed_identities) <= selection_exposure.get(
        "selection_max_event_date"
    ):
        raise ValueError("confirmed window is not strictly forward of selection exposure")
    expected_claim_bindings = {
        "global_claim_key": expected_global_key,
        "global_claim_key_material": expected_key_material,
        "lock_sha256": result.get("lock_sha256"),
        "package_id": lock.get("package_id"),
        "source_fingerprint": lock.get("source_fingerprint"),
        "frozen_control_receipt_sha256": frozen_control.get(
            "bootstrap_receipt_sha256"
        ),
        "full_evaluation_sample_sha256": selection.get(
            "full_evaluation_sample_sha256"
        ),
        "selection_evaluation_sample_sha256": selection.get(
            "evaluation_sample_sha256"
        ),
        "confirmation_evaluation_sample_sha256": selection.get(
            "confirmation_evaluation_sample_sha256"
        ),
        "confirmation_evaluation_fight_identity_sha256": selection.get(
            "confirmation_evaluation_fight_identity_sha256"
        ),
        "confirmation_fight_identities": selection.get(
            "confirmation_fight_identities"
        ),
        "selection_exposure_key": selection_exposure.get(
            "selection_exposure_key"
        ),
        "selection_exposure_path": _repo_relative_artifact_path(
            selection_exposure_path
        ),
        "selection_exposure_sha256": _file_sha256(selection_exposure_path),
        "confirmation_evaluation_input_value_sha256": selection.get(
            "confirmation_evaluation_input_value_sha256"
        ),
        "confirmation_evaluation_n_fights": selection.get(
            "confirmation_evaluation_n_fights"
        ),
        "confirmation_fold_ids": selection.get("confirmation_fold_ids"),
    }
    if (
        claim.get("schema_version") != 1
        or claim.get("protocol") != _CONFIRMATION_PROTOCOL
        or claim.get("status") != "claimed"
        or result.get("claim") != claim
        or result.get("global_claim_key") != expected_global_key
        or any(claim.get(field) != value for field, value in expected_claim_bindings.items())
    ):
        raise ValueError("confirmation result does not reproduce its lock/claim")
    reservation_dir = (
        _repo_root() / GLOBAL_CONFIRMATION_CLAIMS_DIR / "_fight_reservations"
    ).resolve()
    identities = selection.get("confirmation_fight_identities")
    if not isinstance(identities, list) or not identities:
        raise ValueError("confirmation lock has no consumed fight-identity ledger")
    for identity in identities:
        identity_sha256 = _canonical_json_sha256(identity)
        marker_path = reservation_dir / f"{identity_sha256}.json"
        expected_marker = {
            "schema_version": 1,
            "protocol": _CONFIRMATION_PROTOCOL,
            "global_claim_key": expected_global_key,
            "fight_identity_sha256": identity_sha256,
            "fight_identity": identity,
        }
        if not marker_path.is_file() or _read_json(marker_path) != expected_marker:
            raise ValueError("confirmation fight-consumption reservation changed")
    inventory_path = _resolve_repo_artifact_path(
        result.get("evidence_inventory_path")
    )
    if (
        not inventory_path.is_file()
        or _file_sha256(inventory_path) != result.get("evidence_inventory_sha256")
    ):
        raise ValueError("confirmation evidence inventory changed")
    inventory = _read_json(inventory_path)
    inventory_entries = inventory.get("entries") if isinstance(inventory, dict) else None
    if (
        inventory.get("schema_version") != 1
        or inventory.get("protocol") != _CONFIRMATION_PROTOCOL
        or inventory.get("global_claim_key") != result.get("global_claim_key")
        or not isinstance(inventory_entries, list)
        or not inventory_entries
    ):
        raise ValueError("confirmation evidence inventory is malformed")
    inventory_paths: set[Path] = set()
    for entry in inventory_entries:
        path = _resolve_repo_artifact_path((entry or {}).get("path"))
        if (
            path in inventory_paths
            or not path.is_file()
            or _file_sha256(path) != (entry or {}).get("sha256")
        ):
            raise ValueError("confirmation evidence inventory entry changed")
        inventory_paths.add(path)
    global_result_path = _resolve_repo_artifact_path(claim["global_result_path"])
    if not global_result_path.is_file() or _read_json(global_result_path) != result:
        raise ValueError("authoritative global confirmation result differs")
    completion_path = global_result_path.with_name(CONFIRMATION_COMPLETION_FILENAME)
    expected_completion = {
        "schema_version": 1,
        "protocol": _CONFIRMATION_PROTOCOL,
        "global_claim_key": expected_global_key,
        "claim_sha256": result.get("claim_sha256"),
        "lock_sha256": result.get("lock_sha256"),
        "result_path": _repo_relative_artifact_path(global_result_path),
        "result_sha256": _file_sha256(global_result_path),
        "verdict_sha256": _canonical_json_sha256(result.get("verdict")),
        "completed_at": result.get("completed_at"),
    }
    if not completion_path.is_file() or _read_json(completion_path) != expected_completion:
        raise ValueError("confirmation completion marker/result digest changed")
    actual_global_files = {
        path.resolve() for path in claim_path.parent.rglob("*") if path.is_file()
    }
    expected_global_files = inventory_paths | {
        inventory_path.resolve(),
        global_result_path.resolve(),
        completion_path.resolve(),
    }
    if actual_global_files != expected_global_files:
        raise ValueError("canonical confirmation evidence allowlist differs")
    expected_artifacts = {
        "candidate_model_metrics.json",
        "control_model_metrics.json",
        "candidate_result.json",
        "candidate_bet_log.csv",
        "candidate_bankroll_history.csv",
        "control_result.json",
        "control_bet_log.csv",
        "control_bankroll_history.csv",
        "promotion_report.md",
    }
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ValueError("confirmation result artifact allowlist is invalid")
    for name, binding in artifacts.items():
        path = _resolve_repo_artifact_path((binding or {}).get("path"))
        if path.name != name or not path.is_file() or _file_sha256(path) != binding.get(
            "sha256"
        ):
            raise ValueError(f"confirmation result artifact changed: {name}")
    package = lock["candidate_package"]
    expected_result_bindings = {
        "package_id": lock.get("package_id"),
        "lock_path": claim.get("lock_path"),
        "lock_sha256": claim.get("lock_sha256"),
        "global_claim_key": expected_global_key,
        "claim_path": claim.get("claim_path"),
        "claim_sha256": _file_sha256(claim_path),
        "claim": claim,
        "evaluation_partition": "confirmation",
        "evaluation_sample_sha256": selection.get(
            "confirmation_evaluation_sample_sha256"
        ),
        "confirmation_evaluation_fight_identity_sha256": selection.get(
            "confirmation_evaluation_fight_identity_sha256"
        ),
        "confirmation_fold_ids": selection.get("confirmation_fold_ids"),
        "candidate_control_index_identical": True,
        "selection_evidence": selection,
        "candidate_package_contract_sha256": lock.get(
            "candidate_package_contract_sha256"
        ),
        "candidate_package": package,
        "candidate_model_spec_name": package.get("model_spec_name"),
        "candidate_model_spec_payload_sha256": package.get(
            "model_spec_payload_sha256"
        ),
        "candidate_fullfit_model_spec_name": package.get(
            "fullfit_model_spec_name"
        ),
        "candidate_fullfit_model_spec_payload_sha256": package.get(
            "fullfit_model_spec_payload_sha256"
        ),
        "candidate_strategy_config_sha256": _canonical_json_sha256(
            package.get("strategy_config")
        ),
        "candidate_calibration_method": package.get("calibration_method"),
        "candidate_runtime_invariants": package.get("runtime_invariants"),
        "candidate_selection_result_sha256": lock.get("selection_result_sha256"),
        "feature_contract_count": selection.get("feature_contract_count"),
        "feature_contract_sha256": selection.get("feature_contract_sha256"),
        "dataset_fights_path": _repo_relative_artifact_path(
            claim_path.parent / "artifacts" / "inputs" / "fights_cleaned.csv"
        ),
        "dataset_fights_sha256": selection.get("dataset_fights_sha256"),
        "source_dataset_fights_path": _repo_relative_artifact_path(
            claim_path.parent / "artifacts" / "inputs" / "fights_cleaned.csv"
        ),
        "source_dataset_fights_sha256": selection.get(
            "source_dataset_fights_sha256"
        ),
        "features_artifact_path": _repo_relative_artifact_path(
            claim_path.parent / "artifacts" / "inputs" / "features.csv"
        ),
        "features_artifact_sha256": selection.get("features_artifact_sha256"),
        "features_value_sha256": selection.get("features_value_sha256"),
        "source_features_path": _repo_relative_artifact_path(
            claim_path.parent / "artifacts" / "inputs" / "features.csv"
        ),
        "source_features_sha256": selection.get("source_features_sha256"),
        "confirmation_evaluation_input_value_sha256": selection.get(
            "confirmation_evaluation_input_value_sha256"
        ),
        "odds_source_inventory_path": _repo_relative_artifact_path(
            claim_path.parent
            / "artifacts"
            / "inputs"
            / "odds_source_inventory.json"
        ),
        "odds_source_inventory_sha256": selection.get(
            "odds_source_inventory_sha256"
        ),
        "policy_sha256": lock.get("policy_sha256"),
        "scheduled_protocol_sha256": lock.get("scheduled_protocol_sha256"),
        "evaluation_protocol": lock.get("evaluation_protocol"),
        "evaluation_protocol_sha256": lock.get("evaluation_protocol_sha256"),
        "frozen_control_receipt_sha256": frozen_control.get(
            "bootstrap_receipt_sha256"
        ),
        "source_fingerprint": lock.get("source_fingerprint"),
        "source_inventory_path": _repo_relative_artifact_path(
            claim_path.parent
            / "artifacts"
            / "inputs"
            / "evaluation_source_inventory.json"
        ),
        "source_inventory_sha256": selection.get("source_inventory_sha256"),
        "source_inventory_artifact_sha256": selection.get(
            "source_inventory_artifact_sha256"
        ),
        "environment_path": _repo_relative_artifact_path(
            claim_path.parent
            / "artifacts"
            / "inputs"
            / "evaluation_environment.json"
        ),
        "environment_artifact_sha256": selection.get(
            "environment_artifact_sha256"
        ),
        "environment_payload_sha256": selection.get(
            "environment_payload_sha256"
        ),
        "prior_all_fold_baseline_visibility": True,
        "baseline_visibility_limitation": lock.get(
            "baseline_visibility_limitation"
        ),
    }
    if any(result.get(field) != expected for field, expected in expected_result_bindings.items()):
        raise ValueError("confirmed package/spec/strategy binding is invalid")
    durable_odds_inventory_path = _resolve_repo_artifact_path(
        result["odds_source_inventory_path"]
    )
    durable_source_inventory_path = _resolve_repo_artifact_path(
        result["source_inventory_path"]
    )
    durable_environment_path = _resolve_repo_artifact_path(
        result["environment_path"]
    )
    if (
        not durable_odds_inventory_path.is_file()
        or _file_sha256(durable_odds_inventory_path)
        != result["odds_source_inventory_sha256"]
        or _read_json(durable_odds_inventory_path)
        != lock.get("candidate_odds_source_inventory")
    ):
        raise ValueError("confirmed durable odds-source inventory changed")
    if _historical_odds_inventory_payload(lock["evaluation_protocol"]) != _read_json(
        durable_odds_inventory_path
    ):
        raise ValueError("current historical odds inventory differs from confirmation")
    if (
        not durable_source_inventory_path.is_file()
        or _file_sha256(durable_source_inventory_path)
        != result["source_inventory_artifact_sha256"]
        or not durable_environment_path.is_file()
        or _file_sha256(durable_environment_path)
        != result["environment_artifact_sha256"]
    ):
        raise ValueError("confirmed durable source/environment inventory changed")
    current_code = _collect_runtime_code_metadata()
    if (
        current_code.get("source_fingerprint") != result["source_fingerprint"]
        or current_code.get("source_inventory_sha256")
        != result["source_inventory_sha256"]
        or current_code.get("source_inventory")
        != _read_json(durable_source_inventory_path)
        or current_code.get("environment_payload_sha256")
        != result["environment_payload_sha256"]
        or current_code.get("environment") != _read_json(durable_environment_path)
    ):
        raise ValueError("current source/environment differs from confirmation")
    candidate_model_artifact = _read_json(
        _resolve_repo_artifact_path(artifacts["candidate_model_metrics.json"]["path"])
    )
    control_model_artifact = _read_json(
        _resolve_repo_artifact_path(artifacts["control_model_metrics.json"]["path"])
    )
    for prefix, model_payload in (
        ("candidate", candidate_model_artifact),
        ("control", control_model_artifact),
    ):
        for field in (
            "evaluation_input_value_sha256",
            "prediction_rows_sha256",
            "prediction_values_sha256",
            "model_spec_payload_sha256",
        ):
            if (
                not _valid_sha256(model_payload.get(field))
                or (
                    field != "evaluation_input_value_sha256"
                    and field != "model_spec_payload_sha256"
                    and result.get(f"{prefix}_{field}") != model_payload.get(field)
                )
            ):
                raise ValueError(f"confirmed {prefix} {field} binding is invalid")
        if (
            model_payload.get("evaluation_sample_sha256")
            != selection.get("confirmation_evaluation_sample_sha256")
            or model_payload.get("evaluation_input_value_sha256")
            != selection.get("confirmation_evaluation_input_value_sha256")
            or model_payload.get("confirmation_fold_ids")
            != selection.get("confirmation_fold_ids")
        ):
            raise ValueError(f"confirmed {prefix} model evaluated a different reserve")
    control_package = frozen_control.get("package")
    if not isinstance(control_package, dict):
        raise ValueError("confirmation lock has no frozen-control package")
    for prefix, model_payload, bound_package in (
        ("candidate", candidate_model_artifact, package),
        ("control", control_model_artifact, control_package),
    ):
        for field in (
            "model_spec_name",
            "model_spec_payload_sha256",
            "calibration_method",
        ):
            if model_payload.get(field) != bound_package.get(field):
                raise ValueError(f"confirmed {prefix} model differs from locked {field}")
            result_field = f"{prefix}_{field}"
            if result_field in result and result.get(result_field) != model_payload.get(field):
                raise ValueError(f"confirmed {prefix} result differs for {field}")
    if result.get("candidate_model_unchanged_declared") != candidate_model_artifact.get(
        "model_unchanged_declared"
    ):
        raise ValueError("confirmed unchanged-model declaration was altered")
    if result.get("candidate_model_unchanged_declared") is True and any(
        candidate_model_artifact.get(field) != control_model_artifact.get(field)
        for field in (
            "model_spec_name",
            "model_spec_payload_sha256",
            "calibration_method",
            "evaluation_input_value_sha256",
            "prediction_rows_sha256",
            "prediction_values_sha256",
        )
    ):
        raise ValueError("unchanged-model confirmation does not have exact prediction identity")

    def _load_durable_sweep(prefix: str) -> dict[str, Any]:
        sweep_payload = _read_json(
            _resolve_repo_artifact_path(artifacts[f"{prefix}_result.json"]["path"])
        )
        bet_log = pd.read_csv(
            _resolve_repo_artifact_path(artifacts[f"{prefix}_bet_log.csv"]["path"])
        )
        bankroll_history = pd.read_csv(
            _resolve_repo_artifact_path(
                artifacts[f"{prefix}_bankroll_history.csv"]["path"]
            )
        )
        if bet_log.empty or bankroll_history.empty or "combined" not in bankroll_history.columns:
            raise ValueError(
                f"confirmed {prefix} trading evidence lacks settled bets/bankroll history"
            )
        if "event_date" not in bet_log.columns:
            raise ValueError(f"confirmed {prefix} bet log has no event dates")
        bet_log["event_date"] = pd.to_datetime(
            bet_log["event_date"], errors="raise", utc=True
        )
        sweep_payload["bet_log"] = bet_log
        sweep_payload["bankroll_history"] = bankroll_history
        if (
            sweep_payload.get("evaluation_partition") != "confirmation"
            or sweep_payload.get("evaluation_sample_sha256")
            != selection.get("confirmation_evaluation_sample_sha256")
            or sweep_payload.get("confirmation_fold_ids")
            != selection.get("confirmation_fold_ids")
            or sweep_payload.get("confirmation_lock_sha256")
            != result.get("lock_sha256")
            or sweep_payload.get("confirmation_claim_sha256")
            != result.get("claim_sha256")
            or sweep_payload.get("promotion_eligible") is not True
        ):
            raise ValueError(f"confirmed {prefix} sweep binding is invalid")
        return sweep_payload

    candidate_sweep = _load_durable_sweep("candidate")
    control_sweep = _load_durable_sweep("control")
    recomputed_verdict = evaluate_for_promotion(
        _canonicalize_promotion_model_metrics(candidate_model_artifact),
        _canonicalize_promotion_model_metrics(control_model_artifact),
        _canonicalize_sweep_payload(
            candidate_sweep,
            source_path="durable_confirmation:candidate",
        ),
        _canonicalize_sweep_payload(
            control_sweep,
            source_path="durable_confirmation:control",
        ),
    )
    if _normalize_json(result.get("verdict")) != _normalize_json(recomputed_verdict):
        raise ValueError("confirmation verdict does not reproduce durable gate evidence")
    recomputed_passed = recomputed_verdict.get("verdict") == "PROMOTE"
    if (
        result.get("confirmation_passed") is not recomputed_passed
        or result.get("confirmation_outcome")
        != ("pass" if recomputed_passed else "fail")
        or not recomputed_passed
    ):
        raise ValueError("confirmation result is not a recomputed passing locked result")
    return {
        "package_id": result["package_id"],
        "package_contract_sha256": result["candidate_package_contract_sha256"],
        "model_spec_name": result["candidate_model_spec_name"],
        "model_spec_payload_sha256": result[
            "candidate_model_spec_payload_sha256"
        ],
        "fullfit_model_spec_name": result["candidate_fullfit_model_spec_name"],
        "fullfit_model_spec_payload_sha256": result[
            "candidate_fullfit_model_spec_payload_sha256"
        ],
        "strategy_config_sha256": result["candidate_strategy_config_sha256"],
        "strategy_config": package["strategy_config"],
        "calibration_method": result["candidate_calibration_method"],
        "runtime_invariants": result["candidate_runtime_invariants"],
        "model_unchanged_declared": result[
            "candidate_model_unchanged_declared"
        ],
        "prediction_rows_sha256": result["candidate_prediction_rows_sha256"],
        "prediction_values_sha256": result[
            "candidate_prediction_values_sha256"
        ],
        "confirmation_evaluation_input_value_sha256": result[
            "confirmation_evaluation_input_value_sha256"
        ],
        "feature_contract_sha256": result["feature_contract_sha256"],
        "feature_contract_count": result["feature_contract_count"],
        "dataset_fights_path": result["dataset_fights_path"],
        "dataset_fights_sha256": result["dataset_fights_sha256"],
        "source_dataset_fights_path": result["source_dataset_fights_path"],
        "source_dataset_fights_sha256": result[
            "source_dataset_fights_sha256"
        ],
        "features_artifact_path": result["features_artifact_path"],
        "features_artifact_sha256": result["features_artifact_sha256"],
        "features_value_sha256": result["features_value_sha256"],
        "source_features_path": result["source_features_path"],
        "source_features_sha256": result["source_features_sha256"],
        "odds_source_inventory_path": result["odds_source_inventory_path"],
        "odds_source_inventory_sha256": result[
            "odds_source_inventory_sha256"
        ],
        "policy_sha256": result["policy_sha256"],
        "scheduled_protocol_sha256": result["scheduled_protocol_sha256"],
        "evaluation_protocol_sha256": result["evaluation_protocol_sha256"],
        "source_fingerprint": result["source_fingerprint"],
        "source_inventory_path": result["source_inventory_path"],
        "source_inventory_sha256": result["source_inventory_sha256"],
        "source_inventory_artifact_sha256": result[
            "source_inventory_artifact_sha256"
        ],
        "environment_path": result["environment_path"],
        "environment_artifact_sha256": result[
            "environment_artifact_sha256"
        ],
        "environment_payload_sha256": result[
            "environment_payload_sha256"
        ],
        "frozen_control_receipt_sha256": result[
            "frozen_control_receipt_sha256"
        ],
        "global_claim_key": result["global_claim_key"],
        "confirmation_result_path": str(result_path),
        "confirmation_result_sha256": _file_sha256(result_path),
    }


def _run_stage4(
    *,
    run_dir: Path,
    freeze_id: str,
    finalists: list[dict[str, Any]],
    sweep_results: dict[str, dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    confirmation_package_manifest: Path | None = None,
) -> list[dict[str, Any]]:
    reserved_confirmation_folds = int(
        (metadata or {}).get("reserved_confirmation_folds", 0)
    )
    if reserved_confirmation_folds:
        if confirmation_package_manifest is None:
            raise ValueError(
                "Reserved-fold Stage 4 requires one predeclared "
                "--confirmation-package-manifest"
            )
        return _run_locked_confirmation_stage4(
            run_dir=run_dir,
            freeze_id=freeze_id,
            metadata=dict(metadata or {}),
            package_manifest_path=confirmation_package_manifest,
        )
    logger.info("Stage 4: promotion gate")
    readiness = validate_frozen_control_arm_for_promotion_gate(freeze_id)
    if not readiness["ready"]:
        raise ValueError(
            f"Frozen control arm {freeze_id} is not ready for stage 4: {readiness['errors']}"
        )
    control_model_payload, control_sweep_payload = _load_control_arm_payloads(freeze_id)
    control_metrics = _canonicalize_promotion_model_metrics(control_model_payload)

    blocked = _control_sweep_is_placeholder(control_sweep_payload)
    verdicts: list[dict[str, Any]] = []
    promotion_dir = run_dir / PROMOTION_OUTPUT_DIRNAME
    promotion_dir.mkdir(parents=True, exist_ok=True)

    for payload in finalists:
        candidate_id = payload["candidate_id"]
        if blocked:
            result = {
                "candidate_id": candidate_id,
                "model_variant": payload["model_variant"],
                "verdict": "BLOCKED",
                "blocked": True,
                "reason": (
                    f"Frozen control sweep for {freeze_id} is placeholder data "
                    "and cannot support a trustworthy promotion verdict."
                ),
            }
            report_lines = [
                "# Promotion Gate Report",
                "",
                f"Candidate: {candidate_id}",
                "",
                "Verdict: BLOCKED",
                "",
                result["reason"],
            ]
            _stage4_report_path(run_dir, candidate_id).write_text(
                "\n".join(report_lines),
                encoding="utf-8",
            )
            verdicts.append(result)
            continue

        candidate_model_metrics = _canonicalize_promotion_model_metrics(payload)
        promotion_result = sweep_results.get(candidate_id, {}).get("production_result")
        if not isinstance(promotion_result, dict):
            raise ValueError(
                f"Stage 4 requires a fixed production_result for {candidate_id}"
            )
        if promotion_result.get("promotion_eligible") is not True:
            raise ValueError(
                "Stage 4 requires production_result promotion_eligible=true "
                f"for {candidate_id}"
            )
        candidate_sweep = _canonicalize_sweep_payload(
            promotion_result,
            source_path=f"stage3:{candidate_id}:production_result",
        )
        baseline_sweep = _canonicalize_sweep_payload(
            dict(control_sweep_payload or {}),
            source_path=f"frozen_control:{freeze_id}",
        )
        candidate_sweep.setdefault("bankroll_history", pd.DataFrame())
        baseline_sweep.setdefault("bankroll_history", pd.DataFrame())

        verdict = evaluate_for_promotion(
            candidate_model_metrics,
            control_metrics,
            candidate_sweep,
            baseline_sweep,
        )
        report = generate_promotion_report(
            candidate_model_metrics,
            control_metrics,
            candidate_sweep,
            baseline_sweep,
        )
        _stage4_report_path(run_dir, candidate_id).write_text(report, encoding="utf-8")
        verdicts.append(
            {
                "candidate_id": candidate_id,
                "model_variant": payload["model_variant"],
                "blocked": False,
                **verdict,
            }
        )

    _write_json(run_dir / "stage4_verdicts.json", verdicts)
    return verdicts


def _load_stage4_verdicts(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "stage4_verdicts.json"
    if not path.exists():
        return []
    return _read_json(path)


def _table_block(df: pd.DataFrame) -> str:
    if df.empty:
        return "```text\n(no rows)\n```"
    return f"```text\n{df.to_string(index=False)}\n```"


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    """Load a CSV artifact when present, tolerating intentionally empty files."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _run_stage5(
    *,
    run_dir: Path,
    metadata: dict[str, Any],
    finalists: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
) -> Path:
    summary_path = run_dir / "stage1_summary.csv"
    summary_df = _read_csv_or_empty(summary_path)
    comparison_path = run_dir / "stage3_comparison.csv"
    comparison_df = _read_csv_or_empty(comparison_path)

    lines = [
        "# Full Model Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Run Metadata",
        "",
        f"- Freeze ID: {metadata['freeze_id']}",
        f"- Selected variants: {', '.join(metadata['variants'])}",
        f"- Selected datasets: {', '.join(metadata['datasets'])}",
        f"- Selected feature families: {', '.join(metadata['families'])}",
        f"- Matrix size: {metadata['matrix_cells']} cells",
        f"- Feature cache format: {metadata['feature_cache_format']}",
        f"- Narrow finalist sweep enabled: {metadata['run_narrow']}",
        f"- Git SHA: {metadata.get('git_sha')}",
        f"- Git dirty: {metadata.get('git_dirty')}",
        "",
        "## Top Matrix Cells",
        "",
        _table_block(summary_df.head(20)),
        "",
        "## Finalists",
        "",
    ]

    if finalists:
        finalists_df = pd.DataFrame(
            [
                {
                    "candidate_id": payload["candidate_id"],
                    "model_variant": payload["model_variant"],
                    "dataset_variant": payload["dataset_variant"],
                    "feature_family": payload["feature_family"],
                    "calibration_method": payload["calibration_method"],
                    "brier": payload.get("metrics", {}).get("brier"),
                    "ece": payload.get("metrics", {}).get("ece"),
                    "composite_score": payload.get("gate_result", {}).get("composite_score"),
                }
                for payload in finalists
            ]
        )
        lines.extend([_table_block(finalists_df), ""])
    else:
        lines.extend(["No finalists passed the selection gate.", ""])

    lines.extend(
        [
            "## Sweep Comparison",
            "",
            _table_block(comparison_df),
            "",
            "## Promotion Verdicts",
            "",
        ]
    )

    if verdicts:
        verdict_df = pd.DataFrame(
            [
                {
                    "candidate_id": verdict.get("candidate_id"),
                    "model_variant": verdict.get("model_variant"),
                    "verdict": verdict.get("verdict"),
                    "blocked": verdict.get("blocked"),
                    "reason": verdict.get("reason", ""),
                }
                for verdict in verdicts
            ]
        )
        lines.extend([_table_block(verdict_df), ""])
    else:
        lines.extend(["Promotion stage was not run.", ""])

    lines.extend(
        [
            "## Notes",
            "",
            "- This run is report-only. No model artifacts were deployed or promoted automatically.",
        ]
    )
    if any(verdict.get("blocked") for verdict in verdicts):
        lines.append(
            "- Promotion verdicts are blocked when the frozen control sweep is still placeholder data."
        )
    lines.append("")

    final_report_path = run_dir / "final_report.md"
    final_report_path.write_text("\n".join(lines), encoding="utf-8")
    return final_report_path


def _build_run_metadata(
    *,
    variants: list[str],
    datasets: list[str],
    families: list[str],
    freeze_id: str,
    max_workers: int,
    bootstrap: int,
    calibration_method: str | None,
    retrain_months: int,
    max_finalists: int,
    run_narrow: bool,
    feature_cache_format: str,
    runtime_code: dict[str, Any],
    execution_mode: str = "legacy",
    entry_offset_days: float | None = None,
    entry_offset_for_features: bool = False,
    require_entry_odds: bool = False,
    allow_closing_odds: bool = False,
    fixed_control_bootstrap: bool = False,
    selected_model_spec_name: str | None = None,
    selected_model_spec_payload_sha256: str | None = None,
    expected_evaluation_sample_sha256: str | None = None,
    expected_evaluation_n_fights: int | None = None,
    expected_evaluation_n_folds: int | None = None,
    reserved_confirmation_folds: int = DEFAULT_CONFIRMATION_FOLD_COUNT,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "freeze_id": freeze_id,
        "variants": variants,
        "datasets": datasets,
        "families": families,
        "max_workers": max_workers,
        "bootstrap": bootstrap,
        "calibration_method": calibration_method or "variant_default",
        "retrain_months": retrain_months,
        "max_finalists": max_finalists,
        "run_narrow": run_narrow,
        "feature_cache_format": feature_cache_format,
        "execution_mode": execution_mode,
        "entry_offset_days": entry_offset_days,
        "entry_offset_for_features": bool(entry_offset_for_features),
        "require_entry_odds": bool(require_entry_odds),
        "allow_closing_odds": bool(allow_closing_odds),
        "fixed_control_bootstrap": bool(fixed_control_bootstrap),
        "selected_model_spec_name": selected_model_spec_name,
        "selected_model_spec_payload_sha256": selected_model_spec_payload_sha256,
        "expected_evaluation_sample_sha256": expected_evaluation_sample_sha256,
        "expected_evaluation_n_fights": expected_evaluation_n_fights,
        "expected_evaluation_n_folds": expected_evaluation_n_folds,
        "reserved_confirmation_folds": reserved_confirmation_folds,
        "matrix_cells": len(variants) * len(datasets) * len(families),
        **runtime_code,
    }


def _load_or_initialize_metadata(
    *,
    run_dir: Path,
    proposed_metadata: dict[str, Any],
    explicit_overrides: dict[str, Any],
    allow_material_mismatch: bool = False,
) -> dict[str, Any]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        _write_json(metadata_path, proposed_metadata)
        return proposed_metadata

    metadata = _read_json(metadata_path)
    for field_name, value in proposed_metadata.items():
        metadata.setdefault(field_name, value)

    mismatches: list[str] = []
    for field_name in MATERIAL_RESUME_FIELDS:
        cli_value = explicit_overrides.get(field_name)
        if cli_value is None:
            cli_value = proposed_metadata.get(field_name)
        saved_value = metadata.get(field_name)
        if cli_value != saved_value:
            mismatches.append(
                f"{field_name}: saved={saved_value!r}, requested={cli_value!r}"
            )

    if metadata.get("source_fingerprint") != proposed_metadata.get("source_fingerprint"):
        mismatches.append(
            "source_fingerprint: "
            f"saved={metadata.get('source_fingerprint')!r}, "
            f"current={proposed_metadata.get('source_fingerprint')!r}"
        )

    if mismatches and not allow_material_mismatch:
        raise ValueError(
            "Resume metadata mismatch detected. "
            "Use the original run settings or start a new run. "
            f"Mismatches: {'; '.join(mismatches)}"
        )

    if mismatches:
        for mismatch in mismatches:
            logger.warning(
                "Resume mismatch allowed for %s: %s",
                run_dir,
                mismatch,
            )

    if metadata.get("git_sha") != proposed_metadata.get("git_sha") or metadata.get("git_dirty") != proposed_metadata.get("git_dirty"):
        logger.warning(
            "Current git state differs from the saved run metadata for %s "
            "(saved sha=%r dirty=%r, current sha=%r dirty=%r).",
            run_dir,
            metadata.get("git_sha"),
            metadata.get("git_dirty"),
            proposed_metadata.get("git_sha"),
            proposed_metadata.get("git_dirty"),
        )

    _write_json(metadata_path, metadata)
    return metadata


def _persist_source_inventory(run_dir: Path, metadata: dict[str, Any]) -> None:
    inventory = metadata.get("source_inventory")
    environment = metadata.get("environment")
    if not isinstance(inventory, dict):
        raise ValueError("run metadata is missing the complete source inventory")
    if not isinstance(environment, dict):
        raise ValueError("run metadata is missing the evaluation environment")
    expected = metadata.get("source_inventory_sha256")
    if expected != _canonical_json_sha256(inventory):
        raise ValueError("run source inventory does not reproduce its fingerprint")
    if metadata.get("environment_payload_sha256") != _canonical_json_sha256(
        environment
    ):
        raise ValueError("run environment does not reproduce its payload hash")
    path = run_dir / SOURCE_INVENTORY_FILENAME
    environment_path = run_dir / ENVIRONMENT_INVENTORY_FILENAME
    _write_or_validate_json(path, inventory)
    _write_or_validate_json(environment_path, environment)
    metadata["source_inventory_path"] = str(path)
    metadata["source_inventory_artifact_sha256"] = _file_sha256(path)
    metadata["environment_path"] = str(environment_path)
    metadata["environment_artifact_sha256"] = _file_sha256(environment_path)
    _write_json(run_dir / "metadata.json", metadata)


def _configure_logging(run_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(run_dir / "run_evaluation.log"),
        ],
        force=True,
    )


def _valid_sha256(value: str | None) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


_FIXED_CONTROL_PREDICTION_HASH_FIELDS = (
    "prediction_rows_sha256",
    "prediction_values_sha256",
)


def _fixed_control_prediction_hashes(
    payload: dict[str, Any],
    *,
    label: str,
) -> dict[str, str]:
    """Require the fixed-control Stage 1 prediction identity at top level."""
    hashes: dict[str, str] = {}
    for field in _FIXED_CONTROL_PREDICTION_HASH_FIELDS:
        value = payload.get(field)
        if not _valid_sha256(value):
            raise ValueError(f"{label} is missing a valid top-level {field}")
        hashes[field] = value
    return hashes


def _run_fixed_control_bootstrap_cli(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Execute the explicitly constrained one-cell control construction path."""
    variants = _parse_csv_list(args.variants, default_values=[])
    datasets = _parse_csv_list(args.datasets, default_values=[])
    families = _parse_csv_list(args.families, default_values=[])
    required_shape = {
        "--variants": (variants, ["baseline"]),
        "--datasets": (datasets, ["pulled_all_plus_legacy_market"]),
        "--families": (families, ["production"]),
    }
    for label, (actual, expected) in required_shape.items():
        if actual != expected:
            parser.error(
                f"--bootstrap-fixed-control requires {label} "
                f"{','.join(expected)}"
            )
    if args.stage != "1-3":
        parser.error("--bootstrap-fixed-control requires --stage 1-3")
    if args.max_workers != 1:
        parser.error("--bootstrap-fixed-control requires --max-workers 1")
    if args.max_finalists != 1:
        parser.error("--bootstrap-fixed-control requires --max-finalists 1")
    if args.retrain_months != 6:
        parser.error("--bootstrap-fixed-control requires --retrain-months 6")
    if args.bootstrap != DEFAULT_BOOTSTRAP:
        parser.error(
            f"--bootstrap-fixed-control requires --bootstrap {DEFAULT_BOOTSTRAP}"
        )
    if not args.no_narrow:
        parser.error("--bootstrap-fixed-control requires --no-narrow")
    if args.execution_mode != "realistic":
        parser.error("--bootstrap-fixed-control requires --execution-mode realistic")
    if args.entry_offset_days != 1.0:
        parser.error("--bootstrap-fixed-control requires --entry-offset-days 1")
    if not args.entry_offset_for_features:
        parser.error("--bootstrap-fixed-control requires --entry-offset-for-features")
    if not args.require_entry_odds:
        parser.error("--bootstrap-fixed-control requires --require-entry-odds")
    if args.allow_closing_odds:
        parser.error("--bootstrap-fixed-control refuses --allow-closing-odds")
    if args.calibration_method is not None:
        parser.error(
            "--bootstrap-fixed-control uses the named spec calibration and refuses "
            "--calibration-method"
        )
    if args.allow_resume_mismatch:
        parser.error("--bootstrap-fixed-control refuses --allow-resume-mismatch")
    if args.freeze_id == DEFAULT_FREEZE_ID:
        parser.error("--bootstrap-fixed-control requires an explicit new --freeze-id")

    if not args.fixed_control_spec_name:
        parser.error("--bootstrap-fixed-control requires --fixed-control-spec-name")
    if not _valid_sha256(args.fixed_control_spec_payload_sha256):
        parser.error(
            "--bootstrap-fixed-control requires a SHA-256 "
            "--fixed-control-spec-payload-sha256"
        )
    if not _valid_sha256(args.expected_evaluation_sample_sha256):
        parser.error(
            "--bootstrap-fixed-control requires a SHA-256 "
            "--expected-evaluation-sample-sha256"
        )
    if not args.expected_evaluation_fights or args.expected_evaluation_fights <= 0:
        parser.error(
            "--bootstrap-fixed-control requires positive --expected-evaluation-fights"
        )
    if not args.expected_evaluation_folds or args.expected_evaluation_folds <= 0:
        parser.error(
            "--bootstrap-fixed-control requires positive --expected-evaluation-folds"
        )

    try:
        named_spec = resolve_named_training_spec(args.fixed_control_spec_name)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(f"cannot resolve fixed-control spec: {exc}")
    actual_spec_hash = _canonical_json_sha256(asdict(named_spec))
    expected_spec_hash = args.fixed_control_spec_payload_sha256.lower()
    if actual_spec_hash != expected_spec_hash:
        parser.error(
            "fixed-control spec payload hash mismatch: "
            f"expected {expected_spec_hash}, registry has {actual_spec_hash}"
        )
    if (
        named_spec.odds_noise_mode != "antithetic"
        or named_spec.odds_noise_seed != 42
        or named_spec.odds_noise_std != 0.06
        or int((named_spec.xgb_params or {}).get("random_state", -1)) != 42
    ):
        parser.error(
            "fixed-control spec must pin antithetic odds noise std=0.06, "
            "odds-noise seed=42, and XGBoost random_state=42"
        )

    run_dir = _resolve_run_dir(resume=args.resume)
    _configure_logging(run_dir)
    runtime_code = _collect_runtime_code_metadata()
    cache_format = _feature_cache_format()
    proposed_metadata = _build_run_metadata(
        variants=variants,
        datasets=datasets,
        families=families,
        freeze_id=args.freeze_id,
        max_workers=args.max_workers,
        bootstrap=args.bootstrap,
        calibration_method=None,
        retrain_months=args.retrain_months,
        max_finalists=args.max_finalists,
        run_narrow=False,
        feature_cache_format=cache_format,
        runtime_code=runtime_code,
        execution_mode=args.execution_mode,
        entry_offset_days=args.entry_offset_days,
        entry_offset_for_features=args.entry_offset_for_features,
        require_entry_odds=args.require_entry_odds,
        allow_closing_odds=False,
        fixed_control_bootstrap=True,
        selected_model_spec_name=named_spec.name,
        selected_model_spec_payload_sha256=actual_spec_hash,
        expected_evaluation_sample_sha256=(
            args.expected_evaluation_sample_sha256.lower()
        ),
        expected_evaluation_n_fights=args.expected_evaluation_fights,
        expected_evaluation_n_folds=args.expected_evaluation_folds,
    )
    explicit_overrides = {
        field: proposed_metadata.get(field)
        for field in MATERIAL_RESUME_FIELDS
    }
    metadata = _load_or_initialize_metadata(
        run_dir=run_dir,
        proposed_metadata=proposed_metadata,
        explicit_overrides=explicit_overrides,
        allow_material_mismatch=False,
    )
    _persist_source_inventory(run_dir, metadata)
    original_baseline_factory = ALL_VARIANTS["baseline"]
    ALL_VARIANTS["baseline"] = lambda: variant_from_named_training_spec(
        named_spec.name,
        variant_name="baseline",
    )
    try:
        cells = build_cell_specs(
            variants=variants,
            datasets=datasets,
            families=families,
            calibration_method=None,
            retrain_months=args.retrain_months,
            bootstrap=args.bootstrap,
        )
        if len(cells) != 1:
            raise ValueError(
                f"Fixed-control bootstrap requires exactly one cell, got {len(cells)}"
            )
        stage1_results = _run_stage1(
            run_dir=run_dir,
            cells=cells,
            max_workers=1,
            cache_format=metadata["feature_cache_format"],
            current_source_fingerprint=metadata.get("source_fingerprint"),
            entry_offset_days=metadata["entry_offset_days"],
            entry_offset_for_features=True,
            require_entry_odds=True,
            allow_closing_odds=False,
            execution_mode=metadata["execution_mode"],
            model_spec_name=named_spec.name,
            model_spec_payload_sha256=actual_spec_hash,
            reserved_confirmation_folds=int(
                metadata["reserved_confirmation_folds"]
            ),
        )
        if len(stage1_results) != 1:
            raise ValueError(
                "Fixed-control bootstrap requires exactly one successful Stage 1 cell, "
                f"got {len(stage1_results)}"
            )
        cell_result = stage1_results[0]
        stage1_prediction_hashes = _fixed_control_prediction_hashes(
            cell_result,
            label="Fixed-control Stage 1 cell",
        )
        expected_sample = metadata["expected_evaluation_sample_sha256"]
        expected_fights = int(metadata["expected_evaluation_n_fights"])
        expected_folds = int(metadata["expected_evaluation_n_folds"])
        stage1_mismatches: list[str] = []
        if cell_result.get("full_evaluation_sample_sha256") != expected_sample:
            stage1_mismatches.append("full evaluation sample SHA-256")
        if int(cell_result.get("full_evaluation_n_fights", -1)) != expected_fights:
            stage1_mismatches.append("full evaluation fight count")
        if int(cell_result.get("full_evaluation_n_folds", -1)) != expected_folds:
            stage1_mismatches.append("full evaluation fold count")
        if cell_result.get("evaluation_partition") != "selection":
            stage1_mismatches.append("selection partition")
        if int(cell_result.get("reserved_confirmation_folds", -1)) != int(
            metadata["reserved_confirmation_folds"]
        ):
            stage1_mismatches.append("reserved confirmation fold count")
        if cell_result.get("model_spec_name") != named_spec.name:
            stage1_mismatches.append("model spec name")
        if cell_result.get("model_spec_payload_sha256") != actual_spec_hash:
            stage1_mismatches.append("model spec payload SHA-256")
        if stage1_mismatches:
            raise ValueError(
                "Fixed-control Stage 1 evidence mismatch: "
                + ", ".join(stage1_mismatches)
            )

        candidate_id = "baseline_pulled_all_plus_legacy_market_production"
        payload = {
            "candidate_id": candidate_id,
            "model_variant": "baseline",
            "dataset_variant": "pulled_all_plus_legacy_market",
            "feature_family": "production",
            "calibration_method": cell_result["calibration_method"],
            "retrain_months": 6,
            "metrics": cell_result["metrics"],
            "sliced_metrics": cell_result["sliced_metrics"],
            "evaluation_sample_sha256": cell_result["evaluation_sample_sha256"],
            "full_evaluation_sample_sha256": expected_sample,
            "full_evaluation_n_fights": expected_fights,
            "full_evaluation_n_folds": expected_folds,
            "n_predictions": cell_result["n_predictions"],
            "n_folds": cell_result["n_folds"],
            "evaluation_partition": "selection",
            "reserved_confirmation_folds": cell_result[
                "reserved_confirmation_folds"
            ],
            "selection_fold_ids": cell_result["selection_fold_ids"],
            "confirmation_fold_ids": cell_result["confirmation_fold_ids"],
            "confirmation_evaluation_sample_sha256": cell_result[
                "confirmation_evaluation_sample_sha256"
            ],
            "model_spec_name": named_spec.name,
            "model_spec_payload_sha256": actual_spec_hash,
            "input_provenance": cell_result["input_provenance"],
            "input_provenance_payload_sha256": cell_result[
                "input_provenance_payload_sha256"
            ],
            "odds_source_inventory": cell_result["odds_source_inventory"],
            **{
                field: cell_result.get(field)
                for field in _selection_binding_from_payload(cell_result)
            },
            **stage1_prediction_hashes,
            "control_construction": True,
        }
        _run_fixed_control_stage3_candidate(
            run_dir=run_dir,
            payload=payload,
            metadata=metadata,
        )
    finally:
        ALL_VARIANTS["baseline"] = original_baseline_factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full UFC evaluation orchestrator")
    parser.add_argument("--variants", type=str, default=None, help="Comma-separated model variants")
    parser.add_argument("--families", type=str, default=None, help="Comma-separated feature families")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset variants")
    parser.add_argument("--freeze-id", type=str, default=DEFAULT_FREEZE_ID)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None, help="Existing run directory to resume")
    parser.add_argument("--stage", type=str, default=None, help="Single stage N or inclusive range N-M")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--calibration-method", type=str, default=None)
    parser.add_argument("--retrain-months", type=int, default=DEFAULT_RETRAIN_MONTHS)
    parser.add_argument("--no-narrow", action="store_true", help="Skip narrow finalist sweep")
    parser.add_argument("--max-finalists", type=int, default=DEFAULT_MAX_FINALISTS)
    parser.add_argument(
        "--execution-mode",
        type=str,
        choices=["legacy", "realistic"],
        default="legacy",
        help="Stage 3/4 sweep execution model: frictionless 'legacy' or "
        "spread/liquidity/event-batched 'realistic'",
    )
    parser.add_argument(
        "--entry-offset-days",
        type=float,
        default=None,
        help="Use the closest verified snapshot at least this many days before start.",
    )
    parser.add_argument(
        "--entry-offset-for-features",
        action="store_true",
        help="Feed the selected entry snapshot into model odds features before scoring.",
    )
    parser.add_argument(
        "--require-entry-odds",
        action="store_true",
        help="Make rows without verified entry odds ineligible for strategy betting.",
    )
    parser.add_argument(
        "--allow-closing-odds",
        action="store_true",
        help="Research-only: allow closing odds when verified pre-fight roles are absent.",
    )
    parser.add_argument(
        "--allow-resume-mismatch",
        action="store_true",
        help="Allow resuming with material metadata or code mismatches",
    )
    parser.add_argument(
        "--bootstrap-fixed-control",
        action="store_true",
        help=(
            "Explicitly build one fixed production control cell while bypassing "
            "candidate selection; research grids and Stage 4 are forbidden."
        ),
    )
    parser.add_argument("--fixed-control-spec-name", type=str, default=None)
    parser.add_argument(
        "--fixed-control-spec-payload-sha256",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--expected-evaluation-sample-sha256",
        type=str,
        default=None,
    )
    parser.add_argument("--expected-evaluation-fights", type=int, default=None)
    parser.add_argument("--expected-evaluation-folds", type=int, default=None)
    parser.add_argument(
        "--confirmation-package-manifest",
        type=Path,
        default=None,
        help=(
            "Hash-bound one-package manifest for the separate, one-shot "
            "two-fold confirmation run."
        ),
    )
    parser.add_argument(
        "--bounded-selection-manifest",
        type=Path,
        default=None,
        help="Run only the predeclared <=8-package selection-fold executor.",
    )
    args = parser.parse_args()

    if args.bounded_selection_manifest is not None:
        conflicts = []
        for label, present in (
            ("--resume", args.resume is not None),
            ("--stage", args.stage is not None),
            ("--bootstrap-fixed-control", args.bootstrap_fixed_control),
            ("--confirmation-package-manifest", args.confirmation_package_manifest is not None),
            ("--allow-resume-mismatch", args.allow_resume_mismatch),
            ("--variants", args.variants is not None),
            ("--datasets", args.datasets is not None),
            ("--families", args.families is not None),
            ("--calibration-method", args.calibration_method is not None),
            ("--allow-closing-odds", args.allow_closing_odds),
        ):
            if present:
                conflicts.append(label)
        if conflicts:
            parser.error(
                "--bounded-selection-manifest refuses other experiment/resume inputs: "
                + ", ".join(conflicts)
            )
        if args.max_workers != 1:
            parser.error("bounded selection requires --max-workers 1")
        if args.freeze_id == DEFAULT_FREEZE_ID:
            parser.error("bounded selection requires an explicit integrity-v2 --freeze-id")
        run_dir = _resolve_run_dir(resume=None)
        _configure_logging(run_dir)
        winner_path, winner = _run_bounded_selection_experiment(
            run_dir=run_dir,
            experiment_manifest_path=args.bounded_selection_manifest,
            freeze_id=args.freeze_id,
        )
        logger.info(
            "Bounded selection winner %s: %s (sha256=%s)",
            winner["package_id"],
            winner_path,
            _file_sha256(winner_path),
        )
        return

    if args.entry_offset_for_features and args.entry_offset_days is None:
        parser.error("--entry-offset-for-features requires --entry-offset-days")
    if args.require_entry_odds and args.entry_offset_days is None:
        parser.error("--require-entry-odds requires --entry-offset-days")
    if args.entry_offset_days is not None and args.entry_offset_days <= 0.0:
        parser.error("--entry-offset-days must be greater than zero")
    if args.require_entry_odds and args.allow_closing_odds:
        parser.error("--require-entry-odds cannot be combined with --allow-closing-odds")
    if args.allow_resume_mismatch:
        raise ValueError(
            "--allow-resume-mismatch is only valid with --resume and is disabled "
            "for the reserved-fold performance-recovery protocol"
        )

    requested_stage_start, requested_stage_end = _parse_stage_range(args.stage)
    includes_confirmation_stage = (
        requested_stage_start <= 4 <= requested_stage_end
    )
    if includes_confirmation_stage:
        if args.resume is None:
            parser.error(
                "reserved confirmation must be a separate --resume run after "
                "selection; use --stage 4 or --stage 4-5"
            )
        if requested_stage_start != 4 or args.confirmation_package_manifest is None:
            parser.error(
                "reserved confirmation requires --stage 4 or 4-5 and exactly "
                "one --confirmation-package-manifest"
            )
    elif args.confirmation_package_manifest is not None:
        parser.error("--confirmation-package-manifest requires Stage 4")

    if args.bootstrap_fixed_control:
        if args.confirmation_package_manifest is not None:
            parser.error(
                "--bootstrap-fixed-control refuses confirmation package inputs"
            )
        _run_fixed_control_bootstrap_cli(args, parser)
        return
    fixed_control_only_args = (
        args.fixed_control_spec_name,
        args.fixed_control_spec_payload_sha256,
        args.expected_evaluation_sample_sha256,
        args.expected_evaluation_fights,
        args.expected_evaluation_folds,
    )
    if any(value is not None for value in fixed_control_only_args):
        parser.error(
            "fixed-control spec/index arguments require --bootstrap-fixed-control"
        )

    run_dir = _resolve_run_dir(resume=args.resume)
    _configure_logging(run_dir)

    # --- Fresh-run CLI guards ---
    if args.resume is None:
        if args.allow_resume_mismatch:
            raise ValueError("--allow-resume-mismatch is only valid with --resume")
        if args.freeze_id == DEFAULT_FREEZE_ID:
            logger.warning(
                "Fresh run using the default --freeze-id (%s); "
                "pass an explicit --freeze-id to override.",
                DEFAULT_FREEZE_ID,
            )
        if args.calibration_method is not None:
            logger.warning(
                "Global --calibration-method override (%s) on a fresh run; "
                "per-variant defaults will be ignored.",
                args.calibration_method,
            )

    has_betsapi_backfill = _has_saved_betsapi_mma_backfill()
    default_variants = _default_variant_names(has_betsapi_backfill=has_betsapi_backfill)
    default_families = _default_feature_families(has_betsapi_backfill=has_betsapi_backfill)
    variants = _parse_csv_list(args.variants, default_values=default_variants)
    datasets = _parse_csv_list(args.datasets, default_values=list(TRAINING_DATASET_VARIANTS))
    families = _parse_csv_list(args.families, default_values=default_families)
    _ensure_historical_matrix_scope(
        variants=variants,
        families=families,
    )
    if not has_betsapi_backfill and (len(default_variants) < len(ALL_VARIANTS) or len(default_families) < len(ALL_FEATURE_FAMILIES)):
        logger.warning(
            "Local BetsAPI MMA backfill is unavailable; default run selections exclude BetsAPI variants/families."
        )
    _ensure_betsapi_backfill_available(
        variants=variants,
        families=families,
        has_betsapi_backfill=has_betsapi_backfill,
    )

    cache_format = _feature_cache_format()
    run_narrow = not args.no_narrow
    runtime_code = _collect_runtime_code_metadata()
    proposed_metadata = _build_run_metadata(
        variants=variants,
        datasets=datasets,
        families=families,
        freeze_id=args.freeze_id,
        max_workers=args.max_workers,
        bootstrap=args.bootstrap,
        calibration_method=args.calibration_method,
        retrain_months=args.retrain_months,
        max_finalists=args.max_finalists,
        run_narrow=run_narrow,
        feature_cache_format=cache_format,
        runtime_code=runtime_code,
        execution_mode=args.execution_mode,
        entry_offset_days=args.entry_offset_days,
        entry_offset_for_features=args.entry_offset_for_features,
        require_entry_odds=args.require_entry_odds,
        allow_closing_odds=args.allow_closing_odds,
    )
    metadata = _load_or_initialize_metadata(
        run_dir=run_dir,
        proposed_metadata=proposed_metadata,
        explicit_overrides=_build_explicit_resume_overrides(
            args,
            variants=variants,
            datasets=datasets,
            families=families,
            run_narrow=run_narrow,
        ),
        allow_material_mismatch=args.allow_resume_mismatch,
    )
    _persist_source_inventory(run_dir, metadata)

    variants = list(metadata["variants"])
    datasets = list(metadata["datasets"])
    families = list(metadata["families"])
    _ensure_historical_matrix_scope(
        variants=variants,
        families=families,
    )
    freeze_id = metadata["freeze_id"]
    calibration_method = metadata["calibration_method"]
    calibration_override = None if calibration_method == "variant_default" else calibration_method
    retrain_months = int(metadata["retrain_months"])
    bootstrap = int(metadata["bootstrap"])
    max_finalists = int(metadata["max_finalists"])
    run_narrow = bool(metadata["run_narrow"])

    unknown_variants = sorted(set(variants) - set(ALL_VARIANTS))
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}")
    unknown_datasets = sorted(set(datasets) - set(TRAINING_DATASET_VARIANTS))
    if unknown_datasets:
        raise ValueError(f"Unknown dataset variants: {unknown_datasets}")
    unknown_families = sorted(set(families) - set(ALL_FEATURE_FAMILIES))
    if unknown_families:
        raise ValueError(f"Unknown feature families: {unknown_families}")
    _ensure_betsapi_backfill_available(
        variants=variants,
        families=families,
        has_betsapi_backfill=has_betsapi_backfill,
    )

    stage_start, stage_end = requested_stage_start, requested_stage_end
    locked_confirmation_run = (
        stage_start == 4 and args.confirmation_package_manifest is not None
    )
    cells = build_cell_specs(
        variants=variants,
        datasets=datasets,
        families=families,
        calibration_method=calibration_override,
        retrain_months=retrain_months,
        bootstrap=bootstrap,
    )

    stage1_results: list[dict[str, Any]] = []
    finalists: list[dict[str, Any]] = []
    sweep_results: dict[str, dict[str, Any]] = {}
    verdicts: list[dict[str, Any]] = []

    if locked_confirmation_run:
        stage1_results = []
    elif stage_start <= 1 <= stage_end:
        stage1_results = _run_stage1(
            run_dir=run_dir,
            cells=cells,
            max_workers=args.max_workers,
            cache_format=metadata["feature_cache_format"],
            current_source_fingerprint=metadata.get("source_fingerprint"),
            entry_offset_days=metadata.get("entry_offset_days"),
            entry_offset_for_features=bool(
                metadata.get("entry_offset_for_features", False)
            ),
            require_entry_odds=bool(metadata.get("require_entry_odds", False)),
            allow_closing_odds=bool(metadata.get("allow_closing_odds", False)),
            execution_mode=str(metadata.get("execution_mode", "legacy")),
            reserved_confirmation_folds=int(
                metadata["reserved_confirmation_folds"]
            ),
        )
    else:
        stage1_results = _load_stage1_results(
            run_dir,
            reserved_confirmation_folds=int(
                metadata["reserved_confirmation_folds"]
            ),
        )

    if locked_confirmation_run:
        finalists = []
    elif stage_start <= 2 <= stage_end:
        finalists = _run_stage2(
            run_dir=run_dir,
            freeze_id=freeze_id,
            max_finalists=max_finalists,
            stage1_results=stage1_results,
            reserved_confirmation_folds=int(
                metadata["reserved_confirmation_folds"]
            ),
        )
    else:
        finalists = _load_stage2_finalists(
            run_dir,
            reserved_confirmation_folds=int(
                metadata["reserved_confirmation_folds"]
            ),
        )

    if locked_confirmation_run:
        sweep_results = {}
    elif stage_start <= 3 <= stage_end:
        sweep_results = _run_stage3(
            run_dir=run_dir,
            finalists=finalists,
            run_narrow=run_narrow,
            metadata=metadata,
        )
    elif stage_end >= 4:
        sweep_results = _load_stage3_best_results(
            run_dir,
            finalists,
            run_narrow=run_narrow,
            metadata=metadata,
        )
    else:
        sweep_results = {}

    if stage_start <= 4 <= stage_end:
        verdicts = _run_stage4(
            run_dir=run_dir,
            freeze_id=freeze_id,
            finalists=finalists,
            sweep_results=sweep_results,
            metadata=metadata,
            confirmation_package_manifest=args.confirmation_package_manifest,
        )
    elif stage_end >= 5:
        verdicts = _load_stage4_verdicts(run_dir)
    else:
        verdicts = []

    if stage_start <= 5 <= stage_end:
        final_report = _run_stage5(
            run_dir=run_dir,
            metadata=metadata,
            finalists=finalists,
            verdicts=verdicts,
        )
        logger.info("Final report written to %s", final_report)


if __name__ == "__main__":
    main()
