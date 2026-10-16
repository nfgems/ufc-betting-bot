"""Shared feature-contract preparation and checks, without model imports or fitting.

The named-spec route is self-contained by default. Legacy training may explicitly
allow its existing external line-movement transform; prepared inputs cannot use
that transform because it selects data outside the accepted prepared artifact.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.model.training_spec import (
    NamedModelTrainingSpec,
    compute_feature_family_coverage,
    materialize_and_validate_spec_features,
)

logger = logging.getLogger(__name__)


def resolved_spec_identity(spec: NamedModelTrainingSpec) -> str:
    """Hash the resolved recipe, excluding save metadata and resolving its seed.

    The trainer records the effective noise seed when the recipe leaves it
    unspecified. Normalize that existing default here without mutating the spec
    so later recording of the same effective value cannot change this identity.
    """
    payload = asdict(spec)
    payload.pop("trained_at", None)
    payload.pop("git_hash", None)
    if payload["odds_noise_seed"] is None:
        params = payload["xgb_params"]
        random_state = params.get("random_state", 42) if isinstance(params, dict) else 42
        payload["odds_noise_seed"] = int(random_state)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TrainingPreflightResult:
    """Materialized copies and the exact chronological rows used by training."""

    features_df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_cols: list[str]
    spec_name: str
    spec_sha256: str
    dead_train_columns: list[str]
    allowed_all_missing_train_columns: list[str]
    family_coverage: dict[str, dict[str, object]]

    def report(self) -> dict[str, object]:
        """Return a JSON-safe account of completed checks; no fit is implied."""
        return {
            "schema": "training_preflight_v1",
            "training_spec": {"name": self.spec_name, "sha256": self.spec_sha256},
            "materialized_rows": int(len(self.features_df)),
            "train_rows": int(len(self.train_df)),
            "test_rows": int(len(self.test_df)),
            "feature_columns": list(self.feature_cols),
            "dead_train_columns": list(self.dead_train_columns),
            "allowed_all_missing_train_columns": list(self.allowed_all_missing_train_columns),
            "required_family_coverage": self.family_coverage,
            "checks": [
                "feature_contract", "materialization", "temporal_split",
                "dead_train_columns", "required_feature_families",
            ],
            "model_execution": False,
        }


def _dead_train_feature_columns(train_df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Return declared feature columns that are entirely missing on trainable rows."""
    dead_columns: list[str] = []
    for column in feature_cols:
        if column not in train_df.columns:
            continue
        if train_df[column].isna().all():
            dead_columns.append(column)
    return dead_columns


def prepare_train_test(
    features_df: pd.DataFrame,
    cutoff_date: Optional[str] = None,
    min_fights: int = 2,
    feature_cols: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Split features into train/test sets by date.
    Filters out fights where either fighter has fewer than min_fights prior bouts.

    If *start_date* is provided, training rows before that date are dropped.
    If *end_date* is provided, training rows on/after that date are dropped.
    The test set is unaffected.

    Returns (train_df, test_df, feature_columns).
    """
    if cutoff_date is None or not cutoff_date:
        from src.config import TRAIN_CUTOFF_DATE

        cutoff_date = TRAIN_CUTOFF_DATE
    cutoff = pd.Timestamp(cutoff_date)
    if not feature_cols:
        from src.features.build_features import get_feature_columns

        feature_cols = get_feature_columns(features_df)

    # Filter to fights where we have enough data
    df = features_df.copy()
    if "a_num_fights" in df.columns and "b_num_fights" in df.columns:
        df = df[
            (df["a_num_fights"] >= min_fights) & (df["b_num_fights"] >= min_fights)
        ]

    # Drop rows with missing target
    df = df.dropna(subset=["target"])

    # Drop rows where all features are NaN
    df = df.dropna(subset=feature_cols, how="all")

    if "event_date" not in df.columns:
        raise ValueError("features_df must include event_date for temporal train/test split")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    if df["event_date"].isna().any():
        raise ValueError("features_df contains unparseable event_date values after filtering")
    df["_input_order"] = np.arange(len(df))
    df = df.sort_values(["event_date", "_input_order"], kind="mergesort").drop(columns="_input_order")
    if not df["event_date"].is_monotonic_increasing:
        raise ValueError("features_df event_date ordering is not monotonic after temporal sort")

    train = df[df["event_date"] < cutoff].copy()
    test = df[df["event_date"] >= cutoff].copy()

    # Apply optional lower-bound filter to training data only
    if start_date:
        start = pd.Timestamp(start_date)
        pre_filter_count = len(train)
        train = train[train["event_date"] >= start].copy()
        logger.info(
            f"train_start_date={start.date()}: dropped {pre_filter_count - len(train)} "
            f"pre-{start.date()} training rows"
        )
    if end_date:
        end = pd.Timestamp(end_date)
        pre_filter_count = len(train)
        train = train[train["event_date"] < end].copy()
        logger.info(
            f"train_end_date={end.date()}: dropped {pre_filter_count - len(train)} "
            f"training rows on/after {end.date()}"
        )

    logger.info(
        f"Train: {len(train)} fights (before {cutoff.date()}), "
        f"Test: {len(test)} fights (after {cutoff.date()})"
    )
    logger.info(f"Using {len(feature_cols)} features")

    return train, test, feature_cols


def preflight_training_data(
    features_df: pd.DataFrame,
    spec: NamedModelTrainingSpec,
    *,
    allow_external_transforms: bool = False,
) -> TrainingPreflightResult:
    """Run the trainer's existing prefit checks without mutating inputs/spec.

    No filling, odds noise, orientation augmentation, model import or fit occurs
    here. Those remain in the fitting functions. Required-family percentages and
    dead-column checks use the same filtered training rows as actual fitting.
    """
    if spec is None:
        raise ValueError("Training preflight requires an explicit named training spec")
    if spec.add_line_movement and not allow_external_transforms:
        raise ValueError(
            "Prepared preflight requires self-contained transforms; this spec "
            "requires external line-movement data"
        )
    violations = spec.validate_feature_contract()
    if violations:
        for violation in violations:
            logger.error(violation)
        raise ValueError(f"Training spec has {len(violations)} contract violations")

    transformed = materialize_and_validate_spec_features(features_df, spec)
    feature_cols = list(spec.feature_cols)
    for column in feature_cols:
        supplied = transformed[column]
        numeric = pd.to_numeric(supplied, errors="coerce")
        if (supplied.notna() & numeric.isna()).any() or np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"Training feature '{column}' contains malformed or infinite values")
    train_df, test_df, _ = prepare_train_test(
        transformed,
        cutoff_date=spec.train_cutoff_date,
        feature_cols=feature_cols,
        start_date=spec.train_start_date or None,
        end_date=getattr(spec, "train_end_date", "") or None,
    )
    if train_df.empty:
        raise ValueError(f"Training spec '{spec.name}' has no eligible training rows")
    dead_train_columns = _dead_train_feature_columns(train_df, feature_cols)
    allowed_all_missing = set(spec.all_missing_feature_allowlist)
    allowed_dead_columns = [column for column in dead_train_columns if column in allowed_all_missing]
    forbidden_dead_columns = [column for column in dead_train_columns if column not in allowed_all_missing]
    if forbidden_dead_columns:
        raise ValueError(
            f"Training spec '{spec.name}' has dead train-time contract columns "
            f"for dataset variant '{spec.dataset_variant}': {forbidden_dead_columns}. "
            "Refusing to train until the dataset or spec is fixed."
        )
    required_family_coverage = dict(
        getattr(spec, "required_feature_family_coverage_pct", {}) or {}
    )
    coverage_summary: dict[str, dict[str, object]] = {}
    if required_family_coverage:
        coverage_summary = compute_feature_family_coverage(
            train_df,
            feature_cols=feature_cols,
            family_names=list(required_family_coverage.keys()),
        )
        coverage_failures = []
        for family_name, min_pct in required_family_coverage.items():
            family_summary = coverage_summary.get(family_name)
            if family_summary is None:
                coverage_failures.append(
                    f"{family_name}: no active feature columns found in the training frame"
                )
                continue
            actual_pct = float(family_summary["coverage_pct"])
            if actual_pct + 1e-9 < float(min_pct):
                coverage_failures.append(
                    f"{family_name}: {actual_pct:.2f}% complete coverage "
                    f"({family_summary['rows_complete']}/{family_summary['rows_total']}) "
                    f"< required {float(min_pct):.2f}%"
                )
        if coverage_failures:
            joined = "; ".join(coverage_failures)
            raise ValueError(
                f"Training spec '{spec.name}' failed required external-family coverage gates: {joined}"
            )
    return TrainingPreflightResult(
        features_df=transformed,
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        spec_name=spec.name,
        spec_sha256=resolved_spec_identity(spec),
        dead_train_columns=dead_train_columns,
        allowed_all_missing_train_columns=allowed_dead_columns,
        family_coverage=coverage_summary,
    )
