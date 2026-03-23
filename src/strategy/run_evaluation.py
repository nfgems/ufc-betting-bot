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
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
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
from src.data.ufc_refresh import TRAINING_DATASET_VARIANTS, build_training_dataset_variants
from src.features.build_features import (
    BETSAPI_CHALLENGER_FEATURE_NAMES,
    BETSAPI_HISTORICAL_FEATURE_NAMES,
)
from src.strategy.compare_matrix import ALL_FEATURE_FAMILIES, BETSAPI_FEATURE_FAMILIES
from src.strategy.control_arm import (
    load_frozen_control_metrics,
    load_frozen_control_trading_artifacts,
    validate_frozen_control_arm,
    validate_frozen_control_arm_for_promotion_gate,
    validate_frozen_control_arm_for_selection_gate,
)
from src.strategy.duo_trader_sweep import (
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
from src.strategy.model_lab import build_variant_features, generate_variant_fold_predictions
from src.strategy.model_variants import ALL_VARIANTS
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
CELL_OUTPUT_DIRNAME = "cells"
SWEEP_OUTPUT_DIRNAME = "stage3_sweeps"
PROMOTION_OUTPUT_DIRNAME = "stage4_promotions"
STAGE3_STATE_FILENAME = "stage3_state.json"
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
    digest = hashlib.sha256()
    repo_root = _repo_root()
    for relative_path in FINGERPRINT_SOURCE_FILES:
        path = repo_root / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        if path.exists():
            digest.update(path.read_bytes())
        digest.update(b"\0")

    dirty_output = _run_git_command("status", "--short")
    return {
        "git_sha": _run_git_command("rev-parse", "HEAD"),
        "git_dirty": bool(dirty_output) if dirty_output is not None else None,
        "source_fingerprint": digest.hexdigest(),
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


def _save_cached_frame(df: pd.DataFrame, path: Path, cache_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if cache_format == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_pickle(path)


def _load_cached_frame(path: Path) -> pd.DataFrame:
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
        return value.to_dict(orient="records")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _normalize_json(asdict(value) if hasattr(value, "__dataclass_fields__") else vars(value))
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_normalize_json(payload), handle, indent=2, sort_keys=True)


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
) -> dict[str, Any]:
    if predictions_df.empty:
        return {"by_year": {}, "fresh_window": {"n_samples": 0}, "by_coverage": {}}

    preds = predictions_df.copy()
    preds["event_date"] = pd.to_datetime(preds["event_date"], errors="coerce")

    by_year: dict[str, Any] = {}
    for year in sorted(preds["event_date"].dt.year.dropna().astype(int).unique()):
        year_slice = preds[preds["event_date"].dt.year == year]
        by_year[str(year)] = _compute_slice_metrics(year_slice)

    fresh_window = _compute_slice_metrics(preds[preds["event_date"] >= FRESH_WINDOW_CUTOFF])

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

    # Dual odds baselines: closing and opening
    for label, col in [("closing_odds", "a_implied_prob"), ("opening_odds", "a_opening_implied_prob")]:
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
    cell = CellSpec(**task["cell"])
    features_df = _load_cached_frame(Path(task["features_path"]))

    variant = ALL_VARIANTS[cell.model_variant]()
    variant.calibration_method = cell.calibration_method

    fold_predictions = generate_variant_fold_predictions(
        features_df,
        variant,
        retrain_months=cell.retrain_months,
        initial_train_years=5,
        bet_start_date=TRAIN_CUTOFF_DATE,
        feature_family=cell.feature_family,
    )

    predictions_df = (
        pd.concat([predictions for _, predictions in fold_predictions], ignore_index=True)
        if fold_predictions
        else pd.DataFrame()
    )
    if predictions_df.empty:
        raise ValueError(f"No walk-forward predictions produced for {cell.key}")

    predictions_df["event_date"] = pd.to_datetime(predictions_df["event_date"], errors="coerce")
    predictions_df = predictions_df[predictions_df["event_date"] >= pd.Timestamp(TRAIN_CUTOFF_DATE)].copy()
    if predictions_df.empty:
        raise ValueError(f"No eligible post-cutoff predictions produced for {cell.key}")

    seed = _seed_from_key(cell.key)
    metrics = _compute_model_metrics(
        predictions_df,
        bootstrap=cell.bootstrap,
        seed=seed,
    )
    sliced_metrics = _compute_sliced_metrics(predictions_df, features_df, feature_family=cell.feature_family)

    return {
        "status": "success",
        "cell_key": cell.key,
        "model_variant": cell.model_variant,
        "dataset_variant": cell.dataset_variant,
        "feature_family": cell.feature_family,
        "calibration_method": cell.calibration_method,
        "retrain_months": cell.retrain_months,
        "bootstrap": cell.bootstrap,
        "n_folds": len(fold_predictions),
        "n_predictions": int(len(predictions_df)),
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
        "n_folds": result.get("n_folds", 0),
        "n_predictions": result.get("n_predictions", 0),
        "accuracy": metrics.get("accuracy"),
        "brier": metrics.get("brier"),
        "log_loss": metrics.get("log_loss"),
        "ece": metrics.get("ece"),
        "brier_ci_lo": brier_ci.get("ci_lo"),
        "brier_ci_hi": brier_ci.get("ci_hi"),
        "log_loss_ci_lo": log_loss_ci.get("ci_lo"),
        "log_loss_ci_hi": log_loss_ci.get("ci_hi"),
        "closing_odds_baseline_accuracy": (metrics.get("closing_odds_baseline") or {}).get("accuracy"),
        "closing_odds_baseline_brier": (metrics.get("closing_odds_baseline") or {}).get("brier"),
        "closing_odds_baseline_log_loss": (metrics.get("closing_odds_baseline") or {}).get("log_loss"),
        "closing_odds_baseline_n": (metrics.get("closing_odds_baseline") or {}).get("n"),
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
) -> list[dict[str, Any]]:
    logger.info("Stage 1: preparing cached feature frames")
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
            "source_fingerprint": current_source_fingerprint,
        }

    results: list[dict[str, Any]] = []
    for cell in cells:
        output_path = _cell_output_path(run_dir, cell)
        if output_path.exists():
            payload = _read_json(output_path)
            if (
                payload.get("status") == "success"
                and payload.get("source_fingerprint") == current_source_fingerprint
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


def _load_stage1_results(run_dir: Path) -> list[dict[str, Any]]:
    cell_dir = run_dir / CELL_OUTPUT_DIRNAME
    if not cell_dir.exists():
        return []
    results = []
    for path in sorted(cell_dir.glob("*_metrics.json")):
        payload = _read_json(path)
        if payload.get("status") == "success":
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
) -> list[dict[str, Any]]:
    logger.info("Stage 2: selection gate")
    readiness = validate_frozen_control_arm_for_selection_gate(freeze_id)
    if not readiness["ready"]:
        raise ValueError(
            f"Frozen control arm {freeze_id} is not ready for stage 2: {readiness['errors']}"
        )
    control_metrics = load_frozen_control_metrics(freeze_id)
    gate = SelectionGate(control_metrics)

    candidates = [_candidate_from_cell_result(result) for result in stage1_results]
    finalist_pairs = gate.select_finalists_with_specs(candidates, max_finalists=max_finalists)
    finalists = [candidate for candidate, _spec in finalist_pairs]

    report = gate.generate_report(candidates, finalists)
    (run_dir / "stage2_selection_report.md").write_text(report, encoding="utf-8")

    finalist_payloads = []
    for candidate, spec in finalist_pairs:
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


def _load_stage2_finalists(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "stage2_finalists.json"
    if not path.exists():
        return []
    payload = _read_json(path)
    return payload.get("finalists", [])


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


def _load_raw_sweep_result(candidate_dir: Path, label: str) -> dict[str, Any]:
    payload = _read_json(candidate_dir / f"{label}_result.json")
    bet_log_path = candidate_dir / f"{label}_bet_log.csv"
    bankroll_path = candidate_dir / f"{label}_bankroll_history.csv"
    payload["bet_log"] = pd.read_csv(bet_log_path, parse_dates=["event_date"]) if bet_log_path.exists() else pd.DataFrame()
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


def _annotate_sweep_row(row: dict[str, Any], spec: SweepTargetSpec) -> dict[str, Any]:
    row["candidate_id"] = spec.candidate_id
    row["model_variant"] = spec.model_variant
    row["dataset_variant"] = spec.dataset_variant
    row["feature_family"] = spec.feature_family
    row["calibration_method"] = spec.calibration_method
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
        if key not in {"production_result", "best_result", "broad_top_results", "narrow_top_results"}
    }
    _write_json(candidate_dir / "summary.json", summary_payload)
    pd.DataFrame(result.get("broad_summary", [])).to_csv(candidate_dir / "broad_summary.csv", index=False)
    narrow_summary = result.get("narrow_summary")
    if narrow_summary is not None:
        pd.DataFrame(narrow_summary).to_csv(candidate_dir / "narrow_summary.csv", index=False)
    _save_raw_sweep_result(candidate_dir, "production", result["production_result"])
    _save_raw_sweep_result(candidate_dir, "best", result["best_result"])
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
    return {
        **summary,
        "production_result": _load_raw_sweep_result(candidate_dir, "production"),
        "best_result": _load_raw_sweep_result(candidate_dir, "best"),
    }


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
    spec = _spec_from_finalist_payload(payload)
    context = _stage3_context_from_payload(
        payload,
        run_narrow=run_narrow,
        metadata=metadata,
    )
    state = _load_or_initialize_stage3_state(candidate_dir, context)
    if _stage3_candidate_is_complete(candidate_dir, state):
        return _load_stage3_candidate_result(candidate_dir, expected_context=context)

    variant_name = spec.model_variant
    logger.info("Stage 3: generating walk-forward predictions for %s", spec.candidate_id)
    fold_predictions = _generate_walk_forward_predictions(
        bet_start_date=bet_start_date,
        variant_name=variant_name,
        dataset_variant=spec.dataset_variant,
        feature_family=spec.feature_family,
        calibration_method=spec.calibration_method,
        retrain_months=spec.retrain_months,
    )

    production_result_path = candidate_dir / "production_result.json"
    if state["phases"].get("production_complete") and production_result_path.exists() and state.get("production_row"):
        production_row = state["production_row"]
        production_result = _load_raw_sweep_result(candidate_dir, "production")
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
        )
        production_row = _annotate_sweep_row(
            _summary_row_from_result(production_config, production_result),
            spec,
        )
        _persist_stage3_production_phase(
            candidate_dir,
            state,
            production_row=production_row,
            production_result=production_result,
        )

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
            )
            broad_results[config.name] = result
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
                    )
                    narrow_results[config.name] = result
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

    # --- Post-sweep holdout validation ---
    # Re-evaluate the best config on only the last walk-forward fold to detect
    # overfitting to the sweep set.  The holdout metrics are advisory — they do
    # not override selection, but Stage 4 can inspect them.
    holdout_row = None
    if len(fold_predictions) >= 3:
        holdout_folds = fold_predictions[-1:]
        best_sweep_config = _row_to_sweep_config(best_row, variant_name)
        holdout_result = _evaluate_config(
            holdout_folds,
            best_sweep_config,
            initial_bankroll=initial_bankroll,
            bet_start_date=bet_start_date,
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
        "best_result": best_result,
        "holdout_row": holdout_row,
        "primary_summary": narrow_rows if narrow_rows else broad_rows,
        "dataset_variant": spec.dataset_variant,
        "feature_family": spec.feature_family,
        "calibration_method": spec.calibration_method,
        "retrain_months": spec.retrain_months,
    }
    _persist_stage3_candidate_result(
        candidate_dir,
        result,
        state=state,
    )
    return result


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
    }


def _run_stage4(
    *,
    run_dir: Path,
    freeze_id: str,
    finalists: list[dict[str, Any]],
    sweep_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
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
        candidate_sweep = _canonicalize_sweep_payload(
            sweep_results[candidate_id]["best_result"],
            source_path=f"stage3:{candidate_id}:best_result",
        )
        baseline_sweep = _canonicalize_sweep_payload(
            dict(control_sweep_payload or {}),
            source_path=f"frozen_control:{freeze_id}",
        )
        candidate_sweep.setdefault("bet_log", pd.DataFrame())
        candidate_sweep.setdefault("bankroll_history", pd.DataFrame())
        baseline_sweep.setdefault("bet_log", pd.DataFrame())
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
        "--allow-resume-mismatch",
        action="store_true",
        help="Allow resuming with material metadata or code mismatches",
    )
    args = parser.parse_args()

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

    stage_start, stage_end = _parse_stage_range(args.stage)
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

    if stage_start <= 1 <= stage_end:
        stage1_results = _run_stage1(
            run_dir=run_dir,
            cells=cells,
            max_workers=args.max_workers,
            cache_format=metadata["feature_cache_format"],
            current_source_fingerprint=metadata.get("source_fingerprint"),
        )
    else:
        stage1_results = _load_stage1_results(run_dir)

    if stage_start <= 2 <= stage_end:
        finalists = _run_stage2(
            run_dir=run_dir,
            freeze_id=freeze_id,
            max_finalists=max_finalists,
            stage1_results=stage1_results,
        )
    else:
        finalists = _load_stage2_finalists(run_dir)

    if stage_start <= 3 <= stage_end:
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
