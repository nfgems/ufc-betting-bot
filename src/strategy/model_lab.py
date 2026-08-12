"""
Model Lab — sandbox A/B testing framework for betting model variants.

Runs walk-forward backtests of model/strategy variants against the production
baseline. All output goes to logs/model_lab/ — no production files are modified.

Usage:
    python -m src.strategy.model_lab --variants baseline,blend_b_fix,temporal_sigmoid_cal
    python -m src.strategy.model_lab --all
    python -m src.strategy.model_lab --list
"""

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import MISSING, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.config import (
    INITIAL_BANKROLL,
    MIN_EDGE_THRESHOLD,
    KELLY_FRACTION,
    MAX_BET_FRACTION,
    BLEND_WEIGHT,
    PROCESSED_DATA_DIR,
    LOGS_DIR,
    TRAIN_CUTOFF_DATE,
    CONVICTION_MIN_MODEL_PROB,
    CONVICTION_MIN_NO_ODDS_PROB,
    MIN_FIGHTER_FIGHTS,
)
from src.strategy.value import (
    blend_probability,
    dynamic_blend_weight,
    implied_prob_to_decimal_odds,
    _passes_filters,
    calculate_closing_line_value,
)
from src.strategy.bankroll import BankrollManager
from src.strategy.backtest import (
    _merge_historical_odds,  # compatibility seam for existing lab callers/tests
    _prepare_evaluation_odds,
    _resolve_market_odds,
)
from src.strategy.model_variants import (
    VariantConfig,
    apply_variant_feature_transforms,
    train_variant_model,
    ALL_VARIANTS,
)
from src.strategy.confirmation_ledger import (
    require_remotely_anchored_git_artifacts,
)
from src.model.predict import _ordered_feature_frame
from src.model.training_spec import materialize_contract_transforms
from src.strategy.lab_stats import (
    compare_variants,
    plot_comparison,
    compute_ece,
)
from src.features.build_features import (
    get_betsapi_challenger_feature_columns,
    get_feature_columns,
    get_feature_columns_no_odds,
    get_feature_family_columns,
    build_features,
    exclude_market_derived_features,
)

logger = logging.getLogger(__name__)

# Output directory — never writes to models/ or data/processed/
LAB_DIR = LOGS_DIR / "model_lab"
LAB_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIRMATION_FOLD_COUNT = 2


def partition_walk_forward_folds(
    fold_predictions: list[tuple[int, pd.DataFrame]],
    *,
    confirmation_fold_count: int = DEFAULT_CONFIRMATION_FOLD_COUNT,
) -> tuple[list[tuple[int, pd.DataFrame]], list[tuple[int, pd.DataFrame]]]:
    """Split a chronological walk-forward run into selection and confirmation.

    The final ``confirmation_fold_count`` folds are always the confirmation
    partition.  Callers must opt into the returned confirmation frames only
    after a package has been locked; ordinary ranking and strategy experiments
    consume the selection return value exclusively.
    """
    if (
        isinstance(confirmation_fold_count, bool)
        or not isinstance(confirmation_fold_count, int)
        or confirmation_fold_count < 1
    ):
        raise ValueError("confirmation_fold_count must be a positive integer")
    if len(fold_predictions) <= confirmation_fold_count:
        raise ValueError(
            "walk-forward run must contain at least one selection fold plus "
            f"{confirmation_fold_count} confirmation folds"
        )

    fold_ids = [fold_id for fold_id, _frame in fold_predictions]
    if any(isinstance(fold_id, bool) or not isinstance(fold_id, int) for fold_id in fold_ids):
        raise ValueError("walk-forward fold identifiers must be integers")
    if len(set(fold_ids)) != len(fold_ids) or fold_ids != sorted(fold_ids):
        raise ValueError("walk-forward folds must have unique chronological identifiers")

    for fold_id, frame in fold_predictions:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"walk-forward fold {fold_id} is missing prediction rows")
        if "fold" not in frame.columns:
            raise ValueError(f"walk-forward fold {fold_id} is missing its fold column")
        frame_fold_ids = pd.to_numeric(frame["fold"], errors="coerce")
        if frame_fold_ids.isna().any() or not frame_fold_ids.eq(fold_id).all():
            raise ValueError(
                f"walk-forward fold {fold_id} contains inconsistent fold bindings"
            )

    split_at = len(fold_predictions) - confirmation_fold_count
    return list(fold_predictions[:split_at]), list(fold_predictions[split_at:])


def _walk_forward_fold_windows(
    dates: pd.Series,
    *,
    initial_train_years: int,
    retrain_months: int,
    bet_start_date: str,
) -> list[tuple[int, pd.Timestamp, pd.Timestamp, np.ndarray, np.ndarray]]:
    """Derive the complete fold schedule without fitting or predicting."""
    min_date = dates.min()
    max_date = dates.max()
    train_end = min_date + pd.DateOffset(years=initial_train_years)
    bet_start = pd.Timestamp(bet_start_date)
    windows: list[
        tuple[int, pd.Timestamp, pd.Timestamp, np.ndarray, np.ndarray]
    ] = []
    fold_num = 0
    while train_end < max_date:
        test_end = train_end + pd.DateOffset(months=retrain_months)
        if test_end > max_date:
            test_end = max_date + pd.Timedelta(days=1)
        if pd.Timestamp(test_end) <= bet_start:
            train_end = test_end
            continue

        train_mask = dates < train_end
        test_mask = (dates >= train_end) & (dates < test_end)
        if int(train_mask.sum()) < 100 or int(test_mask.sum()) < 5:
            train_end = test_end
            continue

        fold_num += 1
        windows.append(
            (
                fold_num,
                pd.Timestamp(train_end),
                pd.Timestamp(test_end),
                np.flatnonzero(train_mask.to_numpy()),
                np.flatnonzero(test_mask.to_numpy()),
            )
        )
        train_end = test_end
    return windows


def _variant_feature_columns(
    features_df: pd.DataFrame,
    variant: VariantConfig,
) -> list[str]:
    """Resolve the exact feature columns a variant should train on."""
    if variant.feature_cols is not None:
        missing = [column for column in variant.feature_cols if column not in features_df.columns]
        if missing:
            raise ValueError(
                f"Variant '{variant.name}' is missing declared feature columns after materialization: {missing}"
            )
        return list(variant.feature_cols)
    production = ALL_VARIANTS["baseline"]()
    if production.feature_cols is not None:
        missing = [column for column in production.feature_cols if column not in features_df.columns]
        if missing:
            raise ValueError(
                "Promoted baseline contract is missing declared feature columns after "
                f"materialization: {missing}"
            )
        return list(production.feature_cols)
    return get_feature_columns(features_df)


def _variant_default(field_def):
    if field_def.default is not MISSING:
        return field_def.default
    if field_def.default_factory is not MISSING:
        return field_def.default_factory()
    return None


def _resolve_variant_against_promoted_baseline(
    variant: VariantConfig,
) -> VariantConfig:
    """Apply a variant as an explicit delta on top of the promoted baseline."""
    if variant.name == "baseline":
        return variant

    production = ALL_VARIANTS["baseline"]()
    overrides = {}
    for field_def in fields(VariantConfig):
        if field_def.name in {"name", "description"}:
            continue
        value = getattr(variant, field_def.name)
        if value != _variant_default(field_def):
            overrides[field_def.name] = value

    resolved = replace(
        production,
        name=variant.name,
        description=variant.description,
        **overrides,
    )

    use_native_nan = getattr(production, "_native_nan", False)
    if variant.impute_with_indicators:
        use_native_nan = False
    if hasattr(variant, "_native_nan"):
        use_native_nan = bool(getattr(variant, "_native_nan"))

    if use_native_nan:
        resolved._native_nan = True  # type: ignore[attr-defined]
    elif hasattr(resolved, "_native_nan"):
        delattr(resolved, "_native_nan")

    return resolved


def _production_no_odds_variant() -> VariantConfig:
    """Internal no-odds agreement model aligned to the promoted production contract."""
    production = ALL_VARIANTS["baseline"]()
    variant = VariantConfig(
        name="_no_odds_production",
        description="internal production no-odds agreement model",
        calibration_method=production.calibration_method,
        calibration_cv=production.calibration_cv,
        impute_with_indicators=production.impute_with_indicators,
        xgb_params=production.xgb_params,
        time_decay_half_life=production.time_decay_half_life,
        odds_noise_std=production.odds_noise_std,
        odds_noise_seed=production.odds_noise_seed,
        odds_noise_mode=production.odds_noise_mode,
    )
    if getattr(production, "_native_nan", False):
        variant._native_nan = True  # type: ignore[attr-defined]
    return variant


def _materialize_variant_contract_features(
    features_df: pd.DataFrame,
    variant: VariantConfig,
) -> pd.DataFrame:
    """
    Materialize model-lab features on top of the promoted production contract.

    Variants are evaluated as deltas from the promoted baseline rather than
    silently falling back to the legacy feature foundation.
    """
    production = ALL_VARIANTS["baseline"]()
    return materialize_contract_transforms(
        features_df,
        add_rematch_features=production.add_rematch_features or variant.add_rematch_features,
        add_line_movement=production.add_line_movement or variant.add_line_movement,
    )


def _predict_batch_with_model(
    features_df: pd.DataFrame,
    model_result: dict,
) -> pd.DataFrame:
    """
    Generate predictions using a pre-trained model result dict.

    Like src.model.predict.predict_batch but works with in-memory models
    (no disk I/O) and handles indicator columns from impute_with_indicators.
    """
    model = model_result["model"]
    feature_cols = model_result["feature_cols"]
    col_medians = model_result["col_medians"]
    impute_strategy = model_result.get("impute_strategy", "native_nan")
    n_indicator = model_result.get("n_indicator_cols", 0)
    indicator_indices = list(model_result.get("indicator_indices", []))

    X = _ordered_feature_frame(features_df, feature_cols).to_numpy(copy=True)

    if impute_strategy == "native_nan":
        # Symmetrized A/B inference, matching production (predict.py):
        # average the original orientation with the swapped orientation so
        # lab probabilities share production's orientation invariance.
        from src.model.orientation import (
            has_directional_columns,
            swap_directional_frame,
        )

        prob_a = model.predict_proba(X)[:, 1]
        if has_directional_columns(list(feature_cols)):
            swapped_df = swap_directional_frame(
                _ordered_feature_frame(features_df, feature_cols),
                columns=list(feature_cols),
            )
            X_swapped = _ordered_feature_frame(swapped_df, feature_cols).to_numpy(copy=True)
            prob_swapped_a = model.predict_proba(X_swapped)[:, 1]
            prob_a = (prob_a + (1.0 - prob_swapped_a)) / 2.0
        result = features_df.copy()
        result["prob_a"] = prob_a
        result["prob_b"] = 1.0 - prob_a
        return result

    # Rebuild the exact indicator schema recorded at training time.
    # Recorded indices win; legacy artifacts without indices fall back to
    # the first n columns that currently contain NaN.
    indicator_cols = []
    if n_indicator > 0:
        recorded_indices = [
            idx for idx in indicator_indices[:n_indicator]
            if 0 <= idx < X.shape[1]
        ]

        if recorded_indices:
            for idx in recorded_indices:
                indicator_cols.append(np.isnan(X[:, idx]).astype(float))
        else:
            for i in range(X.shape[1]):
                if len(indicator_cols) >= n_indicator:
                    break
                mask = np.isnan(X[:, i])
                if mask.any():
                    indicator_cols.append(mask.astype(float))

        while len(indicator_cols) < n_indicator:
            indicator_cols.append(np.zeros(X.shape[0]))

    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

    if indicator_cols:
        X = np.column_stack([X] + indicator_cols)

    proba = model.predict_proba(X)

    result = features_df.copy()
    result["prob_a"] = proba[:, 1]
    result["prob_b"] = proba[:, 0]
    return result


def build_variant_features(
    fights_df: pd.DataFrame,
    variant: VariantConfig,
    *,
    save_artifacts: bool = False,
) -> pd.DataFrame:
    """Build a feature frame from raw fights for a specific variant."""
    if variant.name == "betsapi_challenger" and not save_artifacts:
        from src.data.betsapi_mma import augment_features_with_betsapi_mma

        features_df = build_features(fights_df)
        features_df = augment_features_with_betsapi_mma(
            features_df,
            save_artifacts=False,
        )
    elif variant.feature_builder_fn is not None:
        features_df = variant.feature_builder_fn(fights_df)
    else:
        features_df = build_features(fights_df)
    return apply_variant_feature_transforms(features_df, variant)


def resolve_variant_feature_columns(
    features_df: pd.DataFrame,
    variant: VariantConfig,
    *,
    feature_family: str | None = None,
    feature_cols: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve primary and no-odds feature columns for a variant run."""
    if feature_cols is not None:
        missing = [column for column in feature_cols if column not in features_df.columns]
        if missing:
            raise ValueError(
                "explicit named-spec feature contract is missing columns: "
                f"{missing}"
            )
        resolved_feature_cols = list(feature_cols)
    elif feature_family is not None:
        resolved_feature_cols = get_feature_family_columns(features_df, feature_family)
    elif variant.name == "betsapi_challenger":
        resolved_feature_cols = get_betsapi_challenger_feature_columns(features_df)
    else:
        resolved_feature_cols = get_feature_columns(features_df)

    no_odds_cols = get_feature_columns_no_odds(
        features_df,
        base_feature_cols=resolved_feature_cols,
    )
    no_odds_cols = [column for column in no_odds_cols if column in features_df.columns]
    return resolved_feature_cols, no_odds_cols


def _select_fold_feature_columns(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    variant: VariantConfig,
) -> tuple[list[str], list[str]]:
    """Apply fold-specific feature pruning for a variant."""
    fold_feature_cols = list(feature_cols)

    if variant.max_features:
        from xgboost import XGBClassifier

        X_quick = train_df[feature_cols].values.copy()
        y_quick = train_df["target"].values
        for i in range(X_quick.shape[1]):
            mask = np.isnan(X_quick[:, i])
            X_quick[mask, i] = 0.0
        quick_xgb = XGBClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        quick_xgb.fit(X_quick, y_quick)
        importance = dict(zip(feature_cols, quick_xgb.feature_importances_))
        fold_feature_cols = sorted(
            importance,
            key=importance.get,
            reverse=True,
        )[:variant.max_features]

    fold_no_odds_cols = get_feature_columns_no_odds(
        train_df,
        base_feature_cols=fold_feature_cols,
    )
    fold_no_odds_cols = [column for column in fold_no_odds_cols if column in train_df.columns]
    if not fold_no_odds_cols:
        fold_no_odds_cols = get_feature_columns_no_odds(
            train_df,
            base_feature_cols=feature_cols,
        )
        fold_no_odds_cols = [column for column in fold_no_odds_cols if column in train_df.columns]

    return fold_feature_cols, fold_no_odds_cols


def _require_locked_confirmation_fit_binding(
    *,
    lock_payload: dict,
    variant: VariantConfig,
    base_feature_cols: list[str],
    retrain_months: int,
    fold_manifest: list[tuple[int, pd.DataFrame]],
    confirmation_fold_count: int,
) -> None:
    """Bind the imminent confirmation fit to the locked package and inputs.

    A valid-but-orphaned claim must not let a direct library caller fit the
    reserve folds with a substituted spec, variant, or features frame
    (DIR-AUD-P2-015). The fitted configuration must be exactly the locked
    candidate or frozen-control package, and the confirmation-partition
    identity/value hashes of the actual frame must equal the lock's selection
    evidence before any fold is fitted.
    """
    from dataclasses import asdict

    from src.model.training_spec import resolve_named_training_spec
    from src.strategy.model_variants import variant_from_named_training_spec
    from src.strategy.run_evaluation import (
        _canonical_json_sha256,
        _evaluation_fight_identity_sha256,
        _evaluation_sample_sha256,
        _manifest_input_value_sha256,
        _partition_evaluation_folds,
        _post_cutoff_predictions,
    )

    locked_packages = [
        package
        for package in (
            lock_payload.get("candidate_package"),
            (lock_payload.get("frozen_control") or {}).get("package"),
        )
        if isinstance(package, dict)
    ]
    if not locked_packages:
        raise ValueError(
            "confirmation package lock does not declare a locked package"
        )
    matched = False
    for package in locked_packages:
        try:
            locked_spec = resolve_named_training_spec(
                str(package.get("model_spec_name"))
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "confirmation lock package spec cannot be resolved"
            ) from exc
        if _canonical_json_sha256(asdict(locked_spec)) != package.get(
            "model_spec_payload_sha256"
        ):
            raise ValueError(
                "locked package named spec changed after the package lock"
            )
        try:
            locked_retrain_months = int(package.get("retrain_months"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "confirmation lock package retrain_months is invalid"
            ) from exc
        expected_variant = _resolve_variant_against_promoted_baseline(
            variant_from_named_training_spec(
                locked_spec.name,
                variant_name=str(package.get("model_variant")),
            )
        )
        if (
            variant == expected_variant
            and getattr(variant, "_native_nan", False)
            == getattr(expected_variant, "_native_nan", False)
            and base_feature_cols == list(locked_spec.feature_cols)
            and int(retrain_months) == locked_retrain_months
        ):
            matched = True
    if not matched:
        raise ValueError(
            "confirmation fit does not match the locked candidate/control package"
        )

    selection_evidence = lock_payload.get("selection_evidence") or {}
    locked_contract = selection_evidence.get("feature_contract_columns")
    if not isinstance(locked_contract, list) or base_feature_cols != [
        str(column) for column in locked_contract
    ]:
        raise ValueError(
            "confirmation fit feature contract differs from the locked selection evidence"
        )
    _selection_manifest, confirmation_manifest = _partition_evaluation_folds(
        fold_manifest,
        reserved_confirmation_folds=confirmation_fold_count,
    )
    confirmation_index = _post_cutoff_predictions(confirmation_manifest)
    if confirmation_index.empty:
        raise ValueError(
            "confirmation fit produced no reserve-fold manifest rows"
        )
    actual_bindings = {
        "confirmation_fold_ids": [
            fold_id for fold_id, _frame in confirmation_manifest
        ],
        "confirmation_evaluation_n_fights": int(len(confirmation_index)),
        "confirmation_evaluation_sample_sha256": _evaluation_sample_sha256(
            confirmation_index
        ),
        "confirmation_evaluation_fight_identity_sha256": (
            _evaluation_fight_identity_sha256(confirmation_index)
        ),
        "confirmation_evaluation_input_value_sha256": (
            _manifest_input_value_sha256(
                confirmation_index,
                feature_contract_columns=base_feature_cols,
            )
        ),
    }
    mismatched = sorted(
        field
        for field, actual in actual_bindings.items()
        if selection_evidence.get(field) != actual
    )
    if mismatched:
        raise ValueError(
            "confirmation features frame differs from the locked package inputs: "
            + ", ".join(mismatched)
        )


def generate_variant_fold_predictions(
    features_df: pd.DataFrame,
    variant: VariantConfig,
    *,
    retrain_months: int = 6,
    initial_train_years: int = 5,
    bet_start_date: str = TRAIN_CUTOFF_DATE,
    feature_family: str | None = None,
    feature_cols: list[str] | None = None,
    entry_offset_days: float | None = None,
    entry_offset_for_features: bool = False,
    require_entry_odds: bool = False,
    allow_closing_odds: bool = False,
    evaluation_partition: str = "selection",
    confirmation_fold_count: int = DEFAULT_CONFIRMATION_FOLD_COUNT,
    return_fold_manifest: bool = False,
    confirmation_claim_sha256: str | None = None,
    confirmation_claim_path: str | Path | None = None,
    allow_all_folds: bool = False,
    include_mirror_diagnostics: bool = False,
    pre_fit_manifest_callback: (
        Callable[[list[tuple[int, pd.DataFrame]]], None] | None
    ) = None,
) -> (
    list[tuple[int, pd.DataFrame]]
    | tuple[list[tuple[int, pd.DataFrame]], list[tuple[int, pd.DataFrame]]]
):
    """Generate walk-forward predictions for an explicitly selected partition.

    ``selection`` never fits or predicts the final confirmation folds.
    ``confirmation`` fits and predicts only those folds and is reserved for the
    one-shot locked-package path. ``all`` is retained for legacy/read-only
    callers outside the remediation promotion workflow.
    """
    if evaluation_partition not in {"all", "selection", "confirmation"}:
        raise ValueError(
            "evaluation_partition must be 'all', 'selection', or 'confirmation'"
        )
    if evaluation_partition == "all" and not allow_all_folds:
        raise ValueError(
            "all-fold model experiments are disabled; use the selection partition"
        )
    if evaluation_partition == "confirmation" and not (
        isinstance(confirmation_claim_sha256, str)
        and len(confirmation_claim_sha256) == 64
        and all(ch in "0123456789abcdef" for ch in confirmation_claim_sha256.lower())
    ):
        raise ValueError(
            "confirmation prediction requires a valid exclusive-claim SHA-256"
        )
    if evaluation_partition == "confirmation":
        claim_path = Path(confirmation_claim_path or "").resolve()
        if not claim_path.is_file():
            raise ValueError("confirmation prediction claim artifact is missing")
        claim_root = (
            Path(__file__).resolve().parents[2]
            / "evidence"
            / "confirmation_claims"
        ).resolve()
        repo_root = Path(__file__).resolve().parents[2]
        def _resolve_claim_bound_path(raw_path: object) -> Path:
            bound = Path(str(raw_path or ""))
            if not bound.is_absolute():
                bound = repo_root / bound
            return bound.resolve()
        try:
            relative_claim = claim_path.relative_to(claim_root)
        except ValueError as exc:
            raise ValueError("confirmation claim is outside the canonical global store") from exc
        if (
            len(relative_claim.parts) != 2
            or relative_claim.name != "confirmation_evaluation_claim.json"
        ):
            raise ValueError("confirmation claim does not use the canonical global path")
        actual_claim_sha256 = hashlib.sha256(claim_path.read_bytes()).hexdigest()
        if actual_claim_sha256 != confirmation_claim_sha256.lower():
            raise ValueError("confirmation prediction claim SHA-256 does not match")
        try:
            claim_payload = json.loads(claim_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("confirmation prediction claim is malformed") from exc
        if (
            claim_payload.get("schema_version") != 1
            or
            claim_payload.get("protocol")
            != "integrity_v2_bounded_performance_recovery"
            or claim_payload.get("status") != "claimed"
        ):
            raise ValueError("confirmation prediction claim has invalid semantics")
        global_key = claim_payload.get("global_claim_key")
        key_material = claim_payload.get("global_claim_key_material")
        if (
            not isinstance(global_key, str)
            or len(global_key) != 64
            or relative_claim.parts[0] != global_key
            or not isinstance(key_material, dict)
            or hashlib.sha256(
                json.dumps(
                    key_material,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            != global_key
            or _resolve_claim_bound_path(claim_payload.get("claim_path"))
            != claim_path
        ):
            raise ValueError("confirmation global claim key is invalid")
        global_result_path = _resolve_claim_bound_path(
            claim_payload.get("global_result_path")
        )
        if global_result_path.parent != claim_path.parent or global_result_path.name != (
            "confirmation_evaluation_result.json"
        ):
            raise ValueError("confirmation global result path is invalid")
        if global_result_path.exists():
            raise ValueError("confirmation result already exists; sample is consumed")
        lock_path = _resolve_claim_bound_path(claim_payload.get("lock_path"))
        lock_sha256 = str(claim_payload.get("lock_sha256") or "").lower()
        if (
            lock_path.name != "confirmation_package_lock.json"
            or not lock_path.is_file()
            or hashlib.sha256(lock_path.read_bytes()).hexdigest() != lock_sha256
        ):
            raise ValueError("confirmation claim does not bind a valid package lock")
        try:
            lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("confirmation package lock is malformed") from exc
        selection = lock_payload.get("selection_evidence") or {}
        frozen_control = lock_payload.get("frozen_control") or {}
        lock_bindings = {
            "package_id": lock_payload.get("package_id"),
            "source_fingerprint": lock_payload.get("source_fingerprint"),
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
            "confirmation_evaluation_input_value_sha256": selection.get(
                "confirmation_evaluation_input_value_sha256"
            ),
            "confirmation_evaluation_n_fights": selection.get(
                "confirmation_evaluation_n_fights"
            ),
            "confirmation_fold_ids": selection.get("confirmation_fold_ids"),
        }
        expected_key_material = {
            "key_contract": "confirmation_sample_consumption_v1",
            "confirmation_evaluation_fight_identity_sha256": selection.get(
                "confirmation_evaluation_fight_identity_sha256"
            ),
        }
        if (
            lock_payload.get("protocol")
            != "integrity_v2_bounded_performance_recovery"
            or any(
                claim_payload.get(field) != value
                for field, value in lock_bindings.items()
            )
            or key_material != expected_key_material
        ):
            raise ValueError("confirmation claim and package lock bindings differ")
        selection_identities = selection.get("selection_fight_identities")
        if not isinstance(selection_identities, list) or not selection_identities:
            raise ValueError("confirmation lock has no selection-exposure ledger")
        selection_identity_sha256 = hashlib.sha256(
            json.dumps(
                selection_identities,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        selection_key_material = {
            "key_contract": "selection_outcome_exposure_v1",
            "selection_evaluation_fight_identity_sha256": selection_identity_sha256,
        }
        selection_exposure_key = hashlib.sha256(
            json.dumps(
                selection_key_material,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        selection_exposure_path = _resolve_claim_bound_path(
            claim_payload.get("selection_exposure_path")
        )
        expected_selection_exposure_path = (
            claim_root
            / "_selection_exposures"
            / f"{selection_exposure_key}.json"
        ).resolve()
        if (
            selection_identity_sha256
            != selection.get("evaluation_fight_identity_sha256")
            or claim_payload.get("selection_exposure_key")
            != selection_exposure_key
            or selection_exposure_path != expected_selection_exposure_path
            or not selection_exposure_path.is_file()
            or hashlib.sha256(selection_exposure_path.read_bytes()).hexdigest()
            != claim_payload.get("selection_exposure_sha256")
        ):
            raise ValueError("confirmation claim has no valid selection-exposure anchor")
        marker_paths = []
        for identity in claim_payload.get("confirmation_fight_identities") or []:
            identity_sha256 = hashlib.sha256(
                json.dumps(
                    identity,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            marker_path = (
                claim_root / "_fight_reservations" / f"{identity_sha256}.json"
            )
            try:
                marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("confirmation reservation marker is missing") from exc
            if marker_payload != {
                "schema_version": 1,
                "protocol": "integrity_v2_bounded_performance_recovery",
                "global_claim_key": global_key,
                "fight_identity_sha256": identity_sha256,
                "fight_identity": identity,
            }:
                raise ValueError("confirmation reservation marker changed")
            marker_paths.append(marker_path)
        require_remotely_anchored_git_artifacts(
            repo_root,
            [claim_path, lock_path, selection_exposure_path, *marker_paths],
            label="confirmation claim",
        )
    if (
        isinstance(confirmation_fold_count, bool)
        or not isinstance(confirmation_fold_count, int)
        or confirmation_fold_count < 1
    ):
        raise ValueError("confirmation_fold_count must be a positive integer")
    variant = _resolve_variant_against_promoted_baseline(variant)
    features_df = features_df.sort_values("event_date").copy()
    features_df = features_df.dropna(subset=["target"])

    base_feature_cols, _ = resolve_variant_feature_columns(
        features_df,
        variant,
        feature_family=feature_family,
        feature_cols=feature_cols,
    )

    if "a_num_fights" in features_df.columns and "b_num_fights" in features_df.columns:
        features_df = features_df[
            (features_df["a_num_fights"] >= 2) & (features_df["b_num_fights"] >= 2)
        ]
    features_df = _prepare_evaluation_odds(
        features_df,
        use_historical_odds=True,
        entry_offset_days=entry_offset_days,
        entry_offset_for_features=entry_offset_for_features,
        require_entry_odds=require_entry_odds,
        historical_merge_fn=_merge_historical_odds,
    )

    dates = pd.to_datetime(features_df["event_date"])
    fold_windows = _walk_forward_fold_windows(
        dates,
        initial_train_years=initial_train_years,
        retrain_months=retrain_months,
        bet_start_date=bet_start_date,
    )

    if evaluation_partition != "all" and len(fold_windows) <= confirmation_fold_count:
        raise ValueError(
            "walk-forward schedule must contain at least one selection fold plus "
            f"{confirmation_fold_count} confirmation folds"
        )

    confirmation_ids = {
        fold_id for fold_id, *_rest in fold_windows[-confirmation_fold_count:]
    }
    if evaluation_partition == "selection":
        selected_windows = [
            window for window in fold_windows if window[0] not in confirmation_ids
        ]
    elif evaluation_partition == "confirmation":
        selected_windows = [
            window for window in fold_windows if window[0] in confirmation_ids
        ]
    else:
        selected_windows = list(fold_windows)

    # The manifest is deliberately richer than the sample identity.  It is
    # created before any fit/predict call and preserves the exact ordered model
    # inputs plus the row-level honest-odds provenance used to construct them.
    # The orchestrator hashes these values independently for the selection and
    # confirmation partitions, so an unchanged fight index cannot conceal
    # changed feature values or a changed T-1 snapshot.
    manifest_columns = ["event_date", "fighter_a", "fighter_b", "target"]
    manifest_columns.extend(
        column
        for column in ("fighter_a_id", "fighter_b_id")
        if column in features_df.columns
    )
    manifest_columns.extend(base_feature_cols)
    manifest_columns.extend(
        column
        for column in features_df.columns
        if column.startswith(("entry_", "opening_", "model_odds_", "market_"))
        or column
        in {
            "a_implied_prob",
            "b_implied_prob",
            "diff_implied_prob",
        }
    )
    manifest_columns = list(dict.fromkeys(manifest_columns))
    missing_manifest = [
        column for column in manifest_columns if column not in features_df.columns
    ]
    if missing_manifest:
        raise ValueError(
            f"walk-forward manifest is missing columns: {missing_manifest}"
        )
    fold_manifest: list[tuple[int, pd.DataFrame]] = []
    for scheduled_fold, scheduled_train_end, scheduled_test_end, _train_idx, test_idx in fold_windows:
        manifest = features_df.iloc[test_idx][manifest_columns].copy()
        manifest["fold"] = scheduled_fold
        manifest["train_end"] = scheduled_train_end
        manifest["test_end"] = scheduled_test_end
        fold_manifest.append((scheduled_fold, manifest))

    if evaluation_partition == "confirmation":
        _require_locked_confirmation_fit_binding(
            lock_payload=lock_payload,
            variant=variant,
            base_feature_cols=base_feature_cols,
            retrain_months=retrain_months,
            fold_manifest=fold_manifest,
            confirmation_fold_count=confirmation_fold_count,
        )
    if evaluation_partition == "selection":
        # Import lazily to avoid a module cycle: the orchestrator owns the
        # canonical identity/ledger contract, while every model-lab selection
        # caller must cross it before the first fit.
        from src.strategy.run_evaluation import _preflight_selection_fold_manifest

        _preflight_selection_fold_manifest(
            fold_manifest,
            confirmation_fold_count=confirmation_fold_count,
        )
    if pre_fit_manifest_callback is not None:
        pre_fit_manifest_callback(fold_manifest)

    fold_predictions: list[tuple[int, pd.DataFrame]] = []
    for fold_num, train_end, test_end, train_idx, test_idx in selected_windows:
        train_df = features_df.iloc[train_idx]
        test_df = features_df.iloc[test_idx]
        logger.info(
            f"  [{variant.name}] Fold {fold_num}: "
            f"Train {len(train_df)}, Test {len(test_df)} "
            f"({train_end.date()} to {test_end.date()})"
        )

        fold_feature_cols, fold_no_odds_cols = _select_fold_feature_columns(
            train_df,
            base_feature_cols,
            variant,
        )

        model_result = train_variant_model(train_df, fold_feature_cols, variant)
        no_odds_variant = _production_no_odds_variant()
        no_odds_result = train_variant_model(train_df, fold_no_odds_cols, no_odds_variant)

        scoring_df = test_df
        predictions = _predict_batch_with_model(scoring_df, model_result)
        no_odds_preds = _predict_batch_with_model(scoring_df, no_odds_result)
        predictions["no_odds_prob_a"] = no_odds_preds["prob_a"]
        predictions["no_odds_prob_b"] = no_odds_preds["prob_b"]
        if include_mirror_diagnostics:
            from src.model.orientation import swap_directional_frame

            swapped_model_inputs = swap_directional_frame(
                scoring_df,
                columns=fold_feature_cols,
            )
            swapped_no_odds_inputs = swap_directional_frame(
                scoring_df,
                columns=fold_no_odds_cols,
            )
            swapped_predictions = _predict_batch_with_model(
                swapped_model_inputs,
                model_result,
            )
            swapped_no_odds_predictions = _predict_batch_with_model(
                swapped_no_odds_inputs,
                no_odds_result,
            )
            # These are actual second inference calls on A/B-swapped contract
            # inputs.  They are retained in the selection prediction artifact
            # so diagnostics can prove symmetry instead of checking the
            # tautological prob_b = 1 - prob_a construction.
            predictions["mirror_swapped_prob_a"] = swapped_predictions[
                "prob_a"
            ].to_numpy()
            predictions["mirror_swapped_no_odds_prob_a"] = (
                swapped_no_odds_predictions["prob_a"].to_numpy()
            )
        for column in ("a_fair_prob_avg", "b_fair_prob_avg"):
            if column in predictions.columns:
                predictions[column] = np.nan

        try:
            odds_policy_kwargs = {}
            if allow_closing_odds or require_entry_odds:
                odds_policy_kwargs = {
                    "allow_closing_odds": allow_closing_odds,
                    "require_entry_odds": require_entry_odds,
                }
            predictions, _ = _resolve_market_odds(
                predictions,
                "a_fair_prob_avg",
                "b_fair_prob_avg",
                **odds_policy_kwargs,
            )
        except ValueError as exc:
            if evaluation_partition != "all":
                raise ValueError(
                    f"reserved fold {fold_num} cannot be bound to market odds"
                ) from exc
            logger.warning(
                "Fold %d skipped (market odds unavailable for %s to %s): %s",
                fold_num, train_end, test_end, exc,
            )
            continue

        predictions = predictions.sort_values("event_date").reset_index(drop=True)
        predictions["bet_eligible"] = (
            predictions["a_market_prob"].notna()
            & predictions["b_market_prob"].notna()
        )
        if require_entry_odds:
            predictions["bet_eligible"] &= (
                predictions["market_verified_prefight"].fillna(False).astype(bool)
            )
        predictions["fold"] = fold_num
        predictions["train_end"] = pd.Timestamp(train_end)
        predictions["test_end"] = pd.Timestamp(test_end)
        fold_predictions.append((fold_num, predictions))

    if return_fold_manifest:
        return fold_predictions, fold_manifest
    return fold_predictions


def run_variant_walkforward(
    features_df: pd.DataFrame,
    variant: VariantConfig,
    retrain_months: int = 6,
    initial_train_years: int = 5,
    initial_bankroll: float = INITIAL_BANKROLL,
    bet_start_date: str = TRAIN_CUTOFF_DATE,
    entry_offset_days: float | None = None,
    entry_offset_for_features: bool = False,
    require_entry_odds: bool = False,
    allow_closing_odds: bool = False,
    evaluation_partition: str = "selection",
    confirmation_fold_count: int = DEFAULT_CONFIRMATION_FOLD_COUNT,
) -> dict:
    """
    Run a walk-forward backtest for a single variant.

    Models train on expanding windows from the start of the dataset, but
    bets are only placed on fights after bet_start_date (default: 2022-01-01).

    Mirrors src.strategy.backtest.run_walkforward_backtest but uses
    variant-specific training, calibration, features, and strategy logic.
    """
    variant = _resolve_variant_against_promoted_baseline(variant)
    features_df = features_df.sort_values("event_date").copy()
    features_df = features_df.dropna(subset=["target"])

    feature_cols = _variant_feature_columns(features_df, variant)
    no_odds_cols = exclude_market_derived_features(feature_cols)

    # Require minimum fighter experience
    if "a_num_fights" in features_df.columns and "b_num_fights" in features_df.columns:
        features_df = features_df[
            (features_df["a_num_fights"] >= 2) & (features_df["b_num_fights"] >= 2)
        ]
    features_df = _prepare_evaluation_odds(
        features_df,
        use_historical_odds=True,
        entry_offset_days=entry_offset_days,
        entry_offset_for_features=entry_offset_for_features,
        require_entry_odds=require_entry_odds,
        historical_merge_fn=_merge_historical_odds,
    )

    if evaluation_partition != "selection":
        raise ValueError(
            "model-lab experiments are selection-only; confirmation requires "
            "the locked one-shot orchestrator"
        )
    full_dates = pd.to_datetime(features_df["event_date"])
    full_schedule = _walk_forward_fold_windows(
        full_dates,
        initial_train_years=initial_train_years,
        retrain_months=retrain_months,
        bet_start_date=bet_start_date,
    )
    if len(full_schedule) <= confirmation_fold_count:
        raise ValueError(
            "model-lab walk-forward requires at least one selection fold plus "
            f"{confirmation_fold_count} confirmation folds"
        )
    exposure_manifest: list[tuple[int, pd.DataFrame]] = []
    for fold_id, train_end, test_end, _train_idx, test_idx in full_schedule:
        frame = features_df.iloc[test_idx][
            ["event_date", "fighter_a", "fighter_b"]
        ].copy()
        frame["fold"] = fold_id
        frame["train_end"] = train_end
        frame["test_end"] = test_end
        exposure_manifest.append((fold_id, frame))
    from src.strategy.run_evaluation import _preflight_selection_fold_manifest

    _preflight_selection_fold_manifest(
        exposure_manifest,
        confirmation_fold_count=confirmation_fold_count,
    )
    confirmation_start = full_schedule[-confirmation_fold_count][1]
    features_df = features_df[full_dates < confirmation_start].copy()

    dates = pd.to_datetime(features_df["event_date"])
    min_date = dates.min()
    max_date = dates.max()
    train_end = min_date + pd.DateOffset(years=initial_train_years)

    bankroll = BankrollManager(
        initial_bankroll=initial_bankroll,
        kelly_fraction=variant.kelly_fraction,
        max_bet_fraction=variant.max_bet_fraction,
    )

    all_bet_log = []
    bankroll_history = [initial_bankroll]
    fold_stats = []
    fold_num = 0

    # Collect all predictions for calibration metrics
    all_y_true = []
    all_y_prob = []

    blend_weight = variant.blend_weight
    bet_start = pd.Timestamp(bet_start_date)

    while train_end < max_date:
        test_end = train_end + pd.DateOffset(months=retrain_months)
        if test_end > max_date:
            test_end = max_date + pd.Timedelta(days=1)

        # Skip folds entirely before the betting window
        if pd.Timestamp(test_end) <= bet_start:
            train_end = test_end
            continue

        train_mask = dates < train_end
        test_mask = (dates >= train_end) & (dates < test_end)

        train_df = features_df[train_mask]
        test_df = features_df[test_mask]

        if len(train_df) < 100 or len(test_df) < 5:
            train_end = test_end
            continue

        fold_num += 1
        logger.info(
            f"  [{variant.name}] Fold {fold_num}: "
            f"Train {len(train_df)}, Test {len(test_df)} "
            f"({train_end.date()} to {test_end.date()})"
        )

        # --- Feature selection (if max_features set) ---
        fold_feature_cols = feature_cols
        if variant.max_features:
            # Quick importance estimate: train a small XGBoost for feature ranking
            from xgboost import XGBClassifier
            X_quick = train_df[feature_cols].values.copy()
            y_quick = train_df["target"].values
            for i in range(X_quick.shape[1]):
                mask = np.isnan(X_quick[:, i])
                X_quick[mask, i] = 0.0
            quick_xgb = XGBClassifier(
                n_estimators=50, max_depth=3, random_state=42, use_label_encoder=False,
                eval_metric="logloss",
            )
            quick_xgb.fit(X_quick, y_quick)
            importance = dict(zip(feature_cols, quick_xgb.feature_importances_))
            top_features = sorted(importance, key=importance.get, reverse=True)[:variant.max_features]
            fold_feature_cols = top_features

        fold_no_odds_cols = list(no_odds_cols)
        if variant.max_features:
            fold_no_odds_cols = exclude_market_derived_features(fold_feature_cols)

        # --- Train primary model ---
        model_result = train_variant_model(train_df, fold_feature_cols, variant)

        # --- Train no-odds model (always production config for agreement filter) ---
        no_odds_variant = _production_no_odds_variant()
        no_odds_result = train_variant_model(train_df, fold_no_odds_cols, no_odds_variant)

        # --- Generate predictions ---
        scoring_df = test_df
        predictions = _predict_batch_with_model(scoring_df, model_result)
        no_odds_preds = _predict_batch_with_model(scoring_df, no_odds_result)
        predictions["no_odds_prob_a"] = no_odds_preds["prob_a"]
        predictions["no_odds_prob_b"] = no_odds_preds["prob_b"]
        for column in ("a_fair_prob_avg", "b_fair_prob_avg"):
            if column in predictions.columns:
                predictions[column] = np.nan

        # Collect for calibration metrics
        valid_mask = predictions["target"].notna()
        all_y_true.extend(predictions.loc[valid_mask, "target"].values.tolist())
        all_y_prob.extend(predictions.loc[valid_mask, "prob_a"].values.tolist())

        # --- Resolve market odds ---
        try:
            odds_policy_kwargs = {}
            if allow_closing_odds or require_entry_odds:
                odds_policy_kwargs = {
                    "allow_closing_odds": allow_closing_odds,
                    "require_entry_odds": require_entry_odds,
                }
            predictions, odds_source = _resolve_market_odds(
                predictions,
                "a_fair_prob_avg",
                "b_fair_prob_avg",
                **odds_policy_kwargs,
            )
        except ValueError as exc:
            logger.warning(
                "Fold %d skipped (market odds unavailable for %s to %s): %s",
                fold_num, train_end, test_end, exc,
            )
            train_end = test_end
            continue

        predictions = predictions.sort_values("event_date").reset_index(drop=True)

        fold_bets = 0
        fold_wins = 0

        # --- Betting loop ---
        for _, row in predictions.iterrows():
            # Only bet on fights after bet_start_date
            if pd.Timestamp(row.get("event_date")) < bet_start:
                continue

            if bankroll.is_stopped:
                break

            model_a = row["prob_a"]
            model_b = row["prob_b"]
            market_a = row["a_market_prob"]
            market_b = row["b_market_prob"]

            if pd.isna(market_a) or pd.isna(market_b):
                continue

            actual_winner_is_a = row["target"] == 1
            no_odds_a = row.get("no_odds_prob_a")
            no_odds_b = row.get("no_odds_prob_b")

            # Dynamic blend weights
            dyn_weight_a = dynamic_blend_weight(model_a, market_a, no_odds_a, blend_weight)
            dyn_weight_b = dynamic_blend_weight(model_b, market_b, no_odds_b, blend_weight)

            # --- Blend (independent weights for both sides) ---
            raw_blend_a = blend_probability(model_a, market_a, dyn_weight_a)
            raw_blend_b = blend_probability(model_b, market_b, dyn_weight_b)
            total = raw_blend_a + raw_blend_b
            if total > 0:
                blend_a = raw_blend_a / total
                blend_b = raw_blend_b / total
            else:
                blend_a = 0.5
                blend_b = 0.5

            edge_a = blend_a - market_a
            edge_b = blend_b - market_b

            # Line movement data
            line_movement = row.get("line_movement")
            line_is_sharp = row.get("line_is_sharp")
            line_steam_move = row.get("line_steam_move")
            if isinstance(line_movement, float) and np.isnan(line_movement):
                line_movement = None

            # Fighter experience
            a_fights = row.get("a_num_fights")
            b_fights = row.get("b_num_fights")
            if isinstance(a_fights, float) and not np.isnan(a_fights):
                a_fights = int(a_fights)
            elif not isinstance(a_fights, int):
                a_fights = None
            if isinstance(b_fights, float) and not np.isnan(b_fights):
                b_fights = int(b_fights)
            elif not isinstance(b_fights, int):
                b_fights = None

            min_edge = variant.min_edge
            placed_bet = False

            if edge_a >= min_edge and edge_a >= edge_b and _passes_filters(
                blend_a, market_a, edge_a, row.get("fighter_a", "A"), no_odds_a,
                line_movement=line_movement, line_is_sharp=line_is_sharp,
                line_steam_move=line_steam_move, bet_side="a",
                a_num_fights=a_fights, b_num_fights=b_fights,
                edge_scaling_base=min_edge,
            ):
                odds = implied_prob_to_decimal_odds(market_a)
                bet_size = bankroll.kelly_bet_size(blend_a, odds)
                if bet_size > 0:
                    bet_idx = len(bankroll.history)
                    bankroll.place_bet(bet_size, row.get("fighter_a", "A"), odds, blend_a, market_a)
                    bankroll.settle_bet(bet_idx, won=actual_winner_is_a)
                    fold_bets += 1
                    placed_bet = True
                    if actual_winner_is_a:
                        fold_wins += 1

                    clv = np.nan
                    if pd.notna(row.get("closing_prob_a")):
                        clv = calculate_closing_line_value(market_a, row["closing_prob_a"])

                    all_bet_log.append({
                        "event_date": row.get("event_date"),
                        "fighter_a": row.get("fighter_a", ""),
                        "fighter_b": row.get("fighter_b", ""),
                        "bet_on": row.get("fighter_a", "A"),
                        "bet_side": "a",
                        "bet_size": bet_size,
                        "odds": odds,
                        "blended_prob": blend_a,
                        "market_prob": market_a,
                        "edge": edge_a,
                        "won": actual_winner_is_a,
                        "profit": bankroll.history[-1]["profit"],
                        "bankroll_after": bankroll.bankroll,
                        "clv": clv,
                        "fold": fold_num,
                    })

            elif edge_b >= min_edge and _passes_filters(
                blend_b, market_b, edge_b, row.get("fighter_b", "B"), no_odds_b,
                line_movement=line_movement, line_is_sharp=line_is_sharp,
                line_steam_move=line_steam_move, bet_side="b",
                a_num_fights=a_fights, b_num_fights=b_fights,
                edge_scaling_base=min_edge,
            ):
                odds = implied_prob_to_decimal_odds(market_b)
                bet_size = bankroll.kelly_bet_size(blend_b, odds)
                if bet_size > 0:
                    bet_idx = len(bankroll.history)
                    bankroll.place_bet(bet_size, row.get("fighter_b", "B"), odds, blend_b, market_b)
                    bankroll.settle_bet(bet_idx, won=not actual_winner_is_a)
                    fold_bets += 1
                    placed_bet = True
                    if not actual_winner_is_a:
                        fold_wins += 1

                    clv = np.nan
                    if pd.notna(row.get("closing_prob_b")):
                        clv = calculate_closing_line_value(market_b, row["closing_prob_b"])

                    all_bet_log.append({
                        "event_date": row.get("event_date"),
                        "fighter_a": row.get("fighter_a", ""),
                        "fighter_b": row.get("fighter_b", ""),
                        "bet_on": row.get("fighter_b", "B"),
                        "bet_side": "b",
                        "bet_size": bet_size,
                        "odds": odds,
                        "blended_prob": blend_b,
                        "market_prob": market_b,
                        "edge": edge_b,
                        "won": not actual_winner_is_a,
                        "profit": bankroll.history[-1]["profit"],
                        "bankroll_after": bankroll.bankroll,
                        "clv": clv,
                        "fold": fold_num,
                    })

            if placed_bet:
                bankroll_history.append(bankroll.bankroll)

        fold_stats.append({
            "fold": fold_num,
            "train_end": str(train_end.date()),
            "test_size": test_mask.sum(),
            "bets": fold_bets,
            "wins": fold_wins,
            "win_rate": fold_wins / fold_bets if fold_bets > 0 else 0,
            "bankroll": bankroll.bankroll,
        })

        train_end = test_end

    # --- Compute stats ---
    stats = bankroll.get_stats()
    bet_log_df = pd.DataFrame(all_bet_log)

    # CLV stats
    if not bet_log_df.empty and "clv" in bet_log_df.columns:
        valid_clv = bet_log_df["clv"].dropna()
        if len(valid_clv) > 0:
            stats["avg_clv"] = valid_clv.mean()
            stats["median_clv"] = valid_clv.median()
            stats["pct_positive_clv"] = (valid_clv > 0).mean()

    # Calibration metrics
    if all_y_true and all_y_prob:
        from sklearn.metrics import brier_score_loss
        y_true_arr = np.array(all_y_true)
        y_prob_arr = np.array(all_y_prob)
        stats["brier_score"] = brier_score_loss(y_true_arr, y_prob_arr)
        stats["ece"] = compute_ece(y_true_arr, y_prob_arr)

    return {
        "stats": stats,
        "bet_log": bet_log_df,
        "bankroll_history": bankroll_history,
        "fold_stats": pd.DataFrame(fold_stats),
        "variant": variant,
    }


def run_experiment(
    variant_names: list[str],
    features_df: Optional[pd.DataFrame] = None,
    initial_bankroll: float = INITIAL_BANKROLL,
    bet_start_date: str = TRAIN_CUTOFF_DATE,
) -> dict:
    """
    Run A/B experiments for the specified variants.

    Always includes baseline as the first variant for comparison.
    Only bets on fights after bet_start_date (default: TRAIN_CUTOFF_DATE = 2022-01-01).

    Returns dict of {variant_name: backtest_result}.
    """
    # --- Load data ---
    if features_df is None:
        # Prefer promoted V5 fullfit artifact, fall back to root
        features_path = PROCESSED_DATA_DIR / "candidates" / "full_live_contract_v5_fullfit" / "features.csv"
        if not features_path.exists():
            features_path = PROCESSED_DATA_DIR / "features.csv"
        if features_path.exists():
            logger.info(f"Loading features from {features_path}")
            features_df = pd.read_csv(features_path, parse_dates=["event_date"])
        else:
            logger.info("No cached features found. Building from Kaggle dataset...")
            from src.data.kaggle_loader import load_kaggle_dataset
            fights_df = load_kaggle_dataset()
            features_df = build_features(fights_df)

    # Ensure baseline is always first
    if "baseline" not in variant_names:
        variant_names = ["baseline"] + variant_names

    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LAB_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"MODEL LAB — Running {len(variant_names)} variants")
    logger.info(f"Output: {run_dir}")
    logger.info(f"{'='*60}\n")

    for name in variant_names:
        if name not in ALL_VARIANTS:
            logger.warning(f"Unknown variant: {name}. Skipping.")
            continue

        variant = _resolve_variant_against_promoted_baseline(ALL_VARIANTS[name]())
        logger.info(f"\n--- {variant.name}: {variant.description} ---")

        # Build features with variant's custom builder if specified
        variant_features = features_df
        if variant.feature_builder_fn is not None:
            logger.info(f"  Using custom feature builder for {variant.name}")
            from src.data.kaggle_loader import load_kaggle_dataset
            fights_df = load_kaggle_dataset()
            variant_features = variant.feature_builder_fn(fights_df)

        variant_features = _materialize_variant_contract_features(variant_features, variant)

        try:
            result = run_variant_walkforward(
                variant_features,
                variant,
                initial_bankroll=initial_bankroll,
                bet_start_date=bet_start_date,
            )
            results[name] = result

            s = result["stats"]
            logger.info(
                f"  Result: ROI {s.get('roi', 0):+.1%}, "
                f"Win rate {s.get('win_rate', 0):.1%}, "
                f"Bets {s.get('total_bets', 0)}, "
                f"Profit ${s.get('total_profit', 0):+.2f}"
            )
            if "brier_score" in s:
                logger.info(f"  Brier: {s['brier_score']:.4f}, ECE: {s.get('ece', 0):.4f}")
            if "avg_clv" in s:
                logger.info(f"  Avg CLV: {s['avg_clv']:+.2%}")

            # Save bet log
            if not result["bet_log"].empty:
                result["bet_log"].to_csv(run_dir / f"{name}_bet_log.csv", index=False)

        except Exception as e:
            logger.error(f"  FAILED: {e}", exc_info=True)

    # --- Comparison ---
    if len(results) >= 2:
        logger.info(f"\n{'='*60}")
        logger.info("VARIANT COMPARISON")
        logger.info(f"{'='*60}")

        comparison = compare_variants(results)
        logger.info(f"\n{comparison.to_string(index=False)}")

        comparison.to_csv(run_dir / "comparison.csv", index=False)
        plot_comparison(results, save_path=str(run_dir / "comparison.png"))

        logger.info(f"\nResults saved to {run_dir}")
    elif len(results) == 1:
        logger.info("Only one variant ran. Need at least 2 for comparison.")

    return results


def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LAB_DIR / "model_lab.log"),
        ],
    )

    parser = argparse.ArgumentParser(description="Model Lab — A/B test model variants")
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated list of variant names to test",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available variants",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available variants and exit",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=INITIAL_BANKROLL,
        help=f"Starting bankroll (default: ${INITIAL_BANKROLL:.2f})",
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable variants:")
        print(f"{'Name':<25} Description")
        print("-" * 70)
        for name, factory in ALL_VARIANTS.items():
            v = factory()
            print(f"  {name:<23} {v.description}")
        return

    if args.all:
        variant_names = list(ALL_VARIANTS.keys())
    elif args.variants:
        variant_names = [v.strip() for v in args.variants.split(",")]
    else:
        # Default: run bug fixes + top improvements
        variant_names = [
            "baseline",
            "blend_b_fix",
            "temporal_sigmoid_cal",
            "missing_indicators",
            "all_bug_fixes",
            "combined_best",
        ]

    run_experiment(variant_names, initial_bankroll=args.bankroll)


if __name__ == "__main__":
    main()
