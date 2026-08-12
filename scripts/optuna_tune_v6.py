"""
Optuna hyperparameter tuning for the V6 model with an untouched outer holdout.

Selection protocol:
  - Inner tuning / selection window: 2022-01-01 through 2024-12-31
  - Outer holdout / final validation: 2025-01-01 up to the confirmation reserve

All trials are optimized only on the inner window. After tuning, the script
evaluates the baseline, best trial, and top-N completed trials on the untouched
outer holdout. Tuned params are not recommended for promotion unless they also
beat the baseline on the outer holdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR
from src.data.kaggle_loader import load_kaggle_dataset
from src.data.ufc_refresh import build_training_dataset_variants
from src.features.build_features import build_features
from src.model.predict import predict_batch
from src.model.train import train_xgboost
from src.model.training_spec import full_live_contract_v6_spec
from src.strategy.model_lab import (
    DEFAULT_CONFIRMATION_FOLD_COUNT,
    _walk_forward_fold_windows,
    generate_variant_fold_predictions,
)
from src.strategy.model_variants import VariantConfig

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PROCESSED_DIR = Path("data/processed")
STUDY_DB = PROCESSED_DIR / "optuna_v6_study.db"
BEST_JSON = PROCESSED_DIR / "optuna_v6_best.json"
SCORES_CSV = PROCESSED_DIR / "optuna_v6_scores.csv"

STUDY_NAME = "v6_hyperparam_tuning_inner_2022_2024_outer_2025plus"
INNER_SELECTION_START = pd.Timestamp("2022-01-01")
INNER_SELECTION_END = pd.Timestamp("2024-12-31")
OUTER_HOLDOUT_START = pd.Timestamp("2025-01-01")

BASELINE_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "time_decay_half_life": 730,
    "odds_noise_std": 0.04,
    "calibration_method": "isotonic",
    "calibration_cv": "timeseries_5fold",
}


def _load_features() -> pd.DataFrame:
    """Load and build the V6 feature frame (cached after first call)."""
    cache_path = PROCESSED_DIR / "optuna_v6_features_cache.parquet"
    if cache_path.exists():
        logger.info("Loading cached features from %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Building features from scratch (first run)...")
    legacy_df = load_kaggle_dataset(RAW_DATA_DIR / "ufc-master.csv")
    variants = build_training_dataset_variants(legacy_df=legacy_df)
    training_df = variants["pulled_all_plus_legacy_market"]
    features_df = build_features(training_df)
    features_df.to_parquet(cache_path, index=False)
    logger.info("Cached features to %s (%d rows)", cache_path, len(features_df))
    return features_df


def _confirmation_reserve_start(features_df: pd.DataFrame) -> pd.Timestamp:
    """Return the first date of the canonical final-two-fold reserve."""
    normalized = features_df.copy()
    normalized["event_date"] = pd.to_datetime(
        normalized["event_date"], errors="coerce"
    )
    normalized = normalized.dropna(subset=["event_date", "target"])
    if {"a_num_fights", "b_num_fights"}.issubset(normalized.columns):
        normalized = normalized[
            (normalized["a_num_fights"] >= 2)
            & (normalized["b_num_fights"] >= 2)
        ]
    normalized = normalized.sort_values("event_date")
    fold_windows = _walk_forward_fold_windows(
        normalized["event_date"],
        initial_train_years=5,
        retrain_months=6,
        bet_start_date=INNER_SELECTION_START.strftime("%Y-%m-%d"),
    )
    if len(fold_windows) <= DEFAULT_CONFIRMATION_FOLD_COUNT:
        raise ValueError(
            "Optuna evaluation requires at least one selection fold plus "
            f"{DEFAULT_CONFIRMATION_FOLD_COUNT} confirmation folds"
        )
    return fold_windows[-DEFAULT_CONFIRMATION_FOLD_COUNT][1]


def _preflight_scored_selection_rows(frame: pd.DataFrame) -> None:
    from src.strategy.run_evaluation import (
        _canonical_evaluation_fight_identities,
        _record_global_selection_exposure,
    )

    _record_global_selection_exposure(
        _canonical_evaluation_fight_identities(frame)
    )


def split_selection_and_holdout_frames(
    features_df: pd.DataFrame,
    *,
    confirmation_start: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split pre-2025 selection from a holdout ending before confirmation."""
    normalized = features_df.copy()
    normalized["event_date"] = pd.to_datetime(normalized["event_date"], errors="coerce")
    normalized = normalized.sort_values("event_date").reset_index(drop=True)
    normalized = normalized.dropna(subset=["event_date", "target"])
    reserve_start = pd.Timestamp(
        confirmation_start
        if confirmation_start is not None
        else _confirmation_reserve_start(normalized)
    )
    if reserve_start <= OUTER_HOLDOUT_START:
        raise ValueError("Confirmation reserve must start after the outer holdout")

    selection_df = normalized[normalized["event_date"] < OUTER_HOLDOUT_START].copy()
    holdout_df = normalized[
        (normalized["event_date"] >= OUTER_HOLDOUT_START)
        & (normalized["event_date"] < reserve_start)
    ].copy()
    return selection_df, holdout_df


def _build_variant_from_params(
    params: dict,
    *,
    name: str,
    description: str,
) -> VariantConfig:
    """Materialize a VariantConfig from an Optuna parameter set."""
    v6_spec = full_live_contract_v6_spec()
    xgb_params = {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "learning_rate": float(params["learning_rate"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "min_child_weight": int(params["min_child_weight"]),
        "gamma": float(params["gamma"]),
        "reg_alpha": float(params["reg_alpha"]),
        "reg_lambda": float(params["reg_lambda"]),
        "scale_pos_weight": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
        "use_label_encoder": False,
    }
    variant = VariantConfig(
        name=name,
        description=description,
        feature_cols=v6_spec.feature_cols,
        xgb_params=xgb_params,
        calibration_method=str(params["calibration_method"]),
        calibration_cv=str(params["calibration_cv"]),
        time_decay_half_life=int(params["time_decay_half_life"]),
        odds_noise_std=float(params["odds_noise_std"]),
        add_rematch_features=True,
    )
    variant._native_nan = True  # type: ignore[attr-defined]
    return variant


def _build_variant(trial: optuna.Trial) -> VariantConfig:
    """Sample hyperparameters for a single Optuna trial."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 0.95, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9, step=0.05),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5, step=0.05),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0, step=0.1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0, step=0.25),
        "time_decay_half_life": trial.suggest_int("time_decay_half_life", 365, 1460, step=365),
        "odds_noise_std": trial.suggest_float("odds_noise_std", 0.02, 0.08, step=0.01),
        "calibration_method": trial.suggest_categorical("calibration_method", ["isotonic", "sigmoid"]),
        "calibration_cv": trial.suggest_categorical("calibration_cv", ["timeseries_5fold", "temporal_holdout"]),
    }
    return _build_variant_from_params(
        params,
        name=f"optuna_trial_{trial.number}",
        description=f"Optuna trial {trial.number}",
    )


def _flatten_fold_predictions(
    fold_predictions: list[tuple[int, pd.DataFrame]],
) -> pd.DataFrame:
    if not fold_predictions:
        return pd.DataFrame()
    combined = pd.concat([frame for _, frame in fold_predictions], ignore_index=True)
    combined["event_date"] = pd.to_datetime(combined["event_date"], errors="coerce")
    return combined.dropna(subset=["event_date"]).sort_values("event_date").reset_index(drop=True)


def score_prediction_window(
    predictions_df: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
    label: str,
) -> dict:
    """Score a prediction frame on a closed or open-ended time window."""
    if predictions_df.empty:
        return {"window": label, "log_loss": float("inf"), "n_predictions": 0}

    mask = predictions_df["event_date"] >= start
    if end is not None:
        mask &= predictions_df["event_date"] <= end
    window_df = predictions_df.loc[mask].copy()
    if window_df.empty:
        return {"window": label, "log_loss": float("inf"), "n_predictions": 0}

    y_true = window_df["target"].astype(int).to_numpy()
    y_prob = np.clip(window_df["prob_a"].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
    return {
        "window": label,
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "n_predictions": int(len(window_df)),
    }


def objective(trial: optuna.Trial, selection_features_df: pd.DataFrame) -> float:
    """Optuna objective: inner-window walk-forward log loss only."""
    variant = _build_variant(trial)
    try:
        fold_predictions = generate_variant_fold_predictions(
            selection_features_df,
            variant,
            retrain_months=6,
            initial_train_years=5,
            bet_start_date=INNER_SELECTION_START.strftime("%Y-%m-%d"),
            feature_cols=variant.feature_cols,
            evaluation_partition="selection",
            confirmation_fold_count=DEFAULT_CONFIRMATION_FOLD_COUNT,
            allow_all_folds=False,
        )
    except Exception as exc:
        logger.warning("Trial %d failed during training: %s", trial.number, exc)
        return float("inf")

    predictions_df = _flatten_fold_predictions(fold_predictions)
    score = score_prediction_window(
        predictions_df,
        start=INNER_SELECTION_START,
        end=INNER_SELECTION_END,
        label="inner_selection",
    )
    if score["n_predictions"] < 50:
        logger.warning("Trial %d: only %d inner-window predictions", trial.number, score["n_predictions"])
        return float("inf")

    trial.report(float(score["log_loss"]), step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    logger.info(
        "Trial %d: inner log_loss=%.4f, n=%d | depth=%d, lr=%.4f, n_est=%d, colsample=%.2f",
        trial.number,
        score["log_loss"],
        score["n_predictions"],
        trial.params["max_depth"],
        trial.params["learning_rate"],
        trial.params["n_estimators"],
        trial.params["colsample_bytree"],
    )
    return float(score["log_loss"])


def _is_degenerate_calibration_error(exc: Exception) -> bool:
    """Return True when calibration CV fails because a fold collapsed to one class."""
    message = str(exc)
    return (
        "Invalid classes inferred from unique values of `y`" in message
        or "at least 2 classes" in message
    )


def _outer_holdout_fit_attempts(variant: VariantConfig) -> list[dict]:
    """Return outer-holdout fit attempts, preserving requested settings first."""
    requested_method = str(variant.calibration_method or "none")
    requested_cv = str(variant.calibration_cv or "timeseries_5fold")

    attempts = [
        {
            "label": "requested",
            "calibrate": requested_method != "none",
            "calibration_method": requested_method,
            "calibration_cv": requested_cv,
            "fallback_reason": "",
        }
    ]

    if requested_method != "none" and requested_cv == "timeseries_5fold":
        attempts.append(
            {
                "label": "temporal_holdout_fallback",
                "calibrate": True,
                "calibration_method": requested_method,
                "calibration_cv": "temporal_holdout",
                "fallback_reason": (
                    "timeseries_5fold calibration produced a degenerate single-class "
                    "split on the pre-holdout training window"
                ),
            }
        )

    if requested_method != "none":
        attempts.append(
            {
                "label": "uncalibrated_fallback",
                "calibrate": False,
                "calibration_method": "none",
                "calibration_cv": requested_cv,
                "fallback_reason": (
                    "all calibration attempts failed on the pre-holdout training window; "
                    "retrying without calibration"
                ),
            }
        )

    return attempts


def _train_then_predict_outer_holdout(
    full_features_df: pd.DataFrame,
    variant: VariantConfig,
    *,
    confirmation_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Train pre-2025 and score only the pre-confirmation outer holdout."""
    features_df = full_features_df.copy()
    features_df["event_date"] = pd.to_datetime(features_df["event_date"], errors="coerce")
    features_df = features_df.dropna(subset=["event_date", "target"]).sort_values("event_date")
    reserve_start = pd.Timestamp(
        confirmation_start
        if confirmation_start is not None
        else _confirmation_reserve_start(features_df)
    )
    if reserve_start <= OUTER_HOLDOUT_START:
        raise ValueError("Confirmation reserve must start after the outer holdout")

    train_df = features_df[features_df["event_date"] < OUTER_HOLDOUT_START].copy()
    holdout_df = features_df[
        (features_df["event_date"] >= OUTER_HOLDOUT_START)
        & (features_df["event_date"] < reserve_start)
    ].copy()
    if len(train_df) < 100:
        raise ValueError(f"Not enough pre-holdout training rows: {len(train_df)}")
    if holdout_df.empty:
        raise ValueError("Outer holdout is empty")
    _preflight_scored_selection_rows(holdout_df)

    feature_cols = list(variant.feature_cols or [])
    last_error: Exception | None = None

    for attempt in _outer_holdout_fit_attempts(variant):
        try:
            model_result = train_xgboost(
                train_df,
                feature_cols,
                calibrate=bool(attempt["calibrate"]),
                impute_strategy="native_nan" if getattr(variant, "_native_nan", False) else "median",
                xgb_params=variant.xgb_params,
                calibration_method=str(attempt["calibration_method"]),
                calibration_cv=str(attempt["calibration_cv"]),
                odds_noise_std=float(variant.odds_noise_std or BASELINE_PARAMS["odds_noise_std"]),
                time_decay_half_life_days=variant.time_decay_half_life,
            )
        except ValueError as exc:
            if not _is_degenerate_calibration_error(exc):
                raise
            last_error = exc
            logger.warning(
                "Outer holdout fit for %s failed with %s calibration (%s). Retrying.",
                variant.name,
                attempt["calibration_cv"],
                exc,
            )
            continue

        predictions = predict_batch(
            holdout_df,
            model_name=variant.name,
            model_result=model_result,
        )
        predictions["event_date"] = pd.to_datetime(predictions["event_date"], errors="coerce")
        predictions = predictions.dropna(subset=["event_date"]).sort_values("event_date").reset_index(drop=True)
        predictions.attrs["outer_holdout_training"] = {
            "requested_calibration_method": str(variant.calibration_method or "none"),
            "requested_calibration_cv": str(variant.calibration_cv or "timeseries_5fold"),
            "effective_calibration_method": str(attempt["calibration_method"]),
            "effective_calibration_cv": str(attempt["calibration_cv"]),
            "fallback_used": bool(attempt["label"] != "requested"),
            "fallback_label": str(attempt["label"]),
            "fallback_reason": str(attempt["fallback_reason"] or ""),
            "train_rows": int(len(train_df)),
            "holdout_rows": int(len(holdout_df)),
            "train_end_date_exclusive": OUTER_HOLDOUT_START.date().isoformat(),
            "holdout_end_date_exclusive": reserve_start.date().isoformat(),
        }
        return predictions

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Outer holdout fit failed without a captured exception for {variant.name}")


def _evaluate_variant(
    *,
    full_features_df: pd.DataFrame,
    selection_features_df: pd.DataFrame,
    variant: VariantConfig,
    selection_score: float | None = None,
) -> dict:
    confirmation_start = _confirmation_reserve_start(full_features_df)
    inner_predictions = _flatten_fold_predictions(
        generate_variant_fold_predictions(
            selection_features_df,
            variant,
            retrain_months=6,
            initial_train_years=5,
            bet_start_date=INNER_SELECTION_START.strftime("%Y-%m-%d"),
            feature_cols=variant.feature_cols,
            evaluation_partition="selection",
            confirmation_fold_count=DEFAULT_CONFIRMATION_FOLD_COUNT,
            allow_all_folds=False,
        )
    )
    inner_metrics = score_prediction_window(
        inner_predictions,
        start=INNER_SELECTION_START,
        end=INNER_SELECTION_END,
        label="inner_selection",
    )
    if selection_score is not None:
        inner_metrics["log_loss"] = float(selection_score)

    outer_predictions = _train_then_predict_outer_holdout(
        full_features_df,
        variant,
        confirmation_start=confirmation_start,
    )
    outer_metrics = score_prediction_window(
        outer_predictions,
        start=OUTER_HOLDOUT_START,
        end=confirmation_start,
        label="outer_holdout",
    )
    return {
        "variant_name": variant.name,
        "params": {
            "n_estimators": variant.xgb_params["n_estimators"],
            "max_depth": variant.xgb_params["max_depth"],
            "learning_rate": variant.xgb_params["learning_rate"],
            "subsample": variant.xgb_params["subsample"],
            "colsample_bytree": variant.xgb_params["colsample_bytree"],
            "min_child_weight": variant.xgb_params["min_child_weight"],
            "gamma": variant.xgb_params["gamma"],
            "reg_alpha": variant.xgb_params["reg_alpha"],
            "reg_lambda": variant.xgb_params["reg_lambda"],
            "time_decay_half_life": variant.time_decay_half_life,
            "odds_noise_std": variant.odds_noise_std,
            "calibration_method": variant.calibration_method,
            "calibration_cv": variant.calibration_cv,
        },
        "inner_selection": inner_metrics,
        "outer_holdout": outer_metrics,
        "outer_training": dict(outer_predictions.attrs.get("outer_holdout_training", {})),
        "inner_predictions": inner_predictions,
        "outer_predictions": outer_predictions,
    }


def _write_prediction_artifact(path: Path, frame: pd.DataFrame) -> None:
    slim = frame.copy()
    keep_cols = [
        column
        for column in [
            "event_date",
            "event_name",
            "fighter_a",
            "fighter_b",
            "target",
            "prob_a",
            "prob_b",
            "fold",
            "train_end",
            "test_end",
        ]
        if column in slim.columns
    ]
    slim[keep_cols].to_csv(path, index=False)


def _persist_report_artifacts(
    baseline_report: dict,
    best_report: dict,
    top_reports: list[dict],
) -> pd.DataFrame:
    rows = []
    report_sets = [
        ("baseline", baseline_report),
        ("best_trial", best_report),
    ]
    for idx, report in enumerate(top_reports, start=1):
        report_sets.append((f"top_trial_{idx}", report))

    for label, report in report_sets:
        for window_key in ("inner_selection", "outer_holdout"):
            metrics = report[window_key]
            rows.append({
                "label": label,
                "variant_name": report["variant_name"],
                "window": metrics["window"],
                "log_loss": metrics["log_loss"],
                "n_predictions": metrics["n_predictions"],
            })

        _write_prediction_artifact(
            PROCESSED_DIR / f"optuna_v6_{label}_inner_predictions.csv",
            report["inner_predictions"],
        )
        _write_prediction_artifact(
            PROCESSED_DIR / f"optuna_v6_{label}_outer_predictions.csv",
            report["outer_predictions"],
        )

    score_df = pd.DataFrame(rows)
    score_df.to_csv(SCORES_CSV, index=False)
    return score_df


def _completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]


def _assert_study_persisted(study: optuna.Study) -> None:
    if not STUDY_DB.exists() or STUDY_DB.stat().st_size <= 0:
        raise RuntimeError(f"Optuna study DB was not persisted: {STUDY_DB}")
    if not _completed_trials(study):
        raise RuntimeError("Optuna study completed with zero persisted trials")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna V6 hyperparameter tuning with outer holdout")
    parser.add_argument("--n-trials", type=int, default=150, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument(
        "--top-n-outer",
        type=int,
        default=3,
        help="How many top completed trials to score on the outer holdout after tuning",
    )
    args = parser.parse_args()

    logger.info("Loading features...")
    full_features_df = _load_features()
    confirmation_start = _confirmation_reserve_start(full_features_df)
    selection_features_df, holdout_df = split_selection_and_holdout_frames(
        full_features_df,
        confirmation_start=confirmation_start,
    )
    logger.info(
        "Features loaded: total=%d rows | selection=%d rows (< %s) | "
        "outer_holdout=%d rows (%s <= date < %s)",
        len(full_features_df),
        len(selection_features_df),
        OUTER_HOLDOUT_START.date(),
        len(holdout_df),
        OUTER_HOLDOUT_START.date(),
        confirmation_start.date(),
    )

    storage = f"sqlite:///{STUDY_DB}"
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.enqueue_trial(dict(BASELINE_PARAMS))

    start = time.time()
    study.optimize(
        lambda trial: objective(trial, selection_features_df),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )
    elapsed = time.time() - start
    _assert_study_persisted(study)

    completed_trials = sorted(_completed_trials(study), key=lambda trial: float(trial.value))
    best_trial = completed_trials[0]

    baseline_variant = _build_variant_from_params(
        BASELINE_PARAMS,
        name="baseline_v6",
        description="Baseline V6 production parameters",
    )
    best_variant = _build_variant_from_params(
        dict(best_trial.params),
        name=f"best_trial_{best_trial.number}",
        description=f"Best Optuna trial {best_trial.number}",
    )

    baseline_report = _evaluate_variant(
        full_features_df=full_features_df,
        selection_features_df=selection_features_df,
        variant=baseline_variant,
    )
    best_report = _evaluate_variant(
        full_features_df=full_features_df,
        selection_features_df=selection_features_df,
        variant=best_variant,
        selection_score=float(best_trial.value),
    )

    top_reports: list[dict] = []
    for trial in completed_trials[: max(int(args.top_n_outer), 0)]:
        if trial.number == best_trial.number:
            top_reports.append(best_report)
            continue
        variant = _build_variant_from_params(
            dict(trial.params),
            name=f"top_trial_{trial.number}",
            description=f"Top Optuna trial {trial.number}",
        )
        top_reports.append(
            _evaluate_variant(
                full_features_df=full_features_df,
                selection_features_df=selection_features_df,
                variant=variant,
                selection_score=float(trial.value),
            )
        )

    score_df = _persist_report_artifacts(baseline_report, best_report, top_reports)
    baseline_outer = float(baseline_report["outer_holdout"]["log_loss"])
    best_outer = float(best_report["outer_holdout"]["log_loss"])
    tuned_beats_outer = bool(best_outer < baseline_outer)

    report = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "study": {
            "name": STUDY_NAME,
            "db_path": str(STUDY_DB.resolve(strict=False)),
            "db_size_bytes": int(STUDY_DB.stat().st_size),
            "completed_trials": len(completed_trials),
            "elapsed_minutes": round(elapsed / 60.0, 1),
        },
        "selection_window": {
            "start": INNER_SELECTION_START.date().isoformat(),
            "end": INNER_SELECTION_END.date().isoformat(),
        },
        "outer_holdout_window": {
            "start": OUTER_HOLDOUT_START.date().isoformat(),
            "end_exclusive": baseline_report["outer_training"][
                "holdout_end_date_exclusive"
            ],
        },
        "baseline": {
            "selection_score": baseline_report["inner_selection"],
            "final_validation_score": baseline_report["outer_holdout"],
            "outer_training": baseline_report.get("outer_training", {}),
            "params": baseline_report["params"],
        },
        "best_trial": {
            "trial_number": best_trial.number,
            "selection_score": best_report["inner_selection"],
            "final_validation_score": best_report["outer_holdout"],
            "outer_training": best_report.get("outer_training", {}),
            "params": best_report["params"],
            "beats_baseline_on_outer_holdout": tuned_beats_outer,
        },
        "top_outer_reports": [
            {
                "variant_name": report_item["variant_name"],
                "selection_score": report_item["inner_selection"],
                "final_validation_score": report_item["outer_holdout"],
                "outer_training": report_item.get("outer_training", {}),
                "params": report_item["params"],
            }
            for report_item in top_reports
        ],
        "promotion_decision": {
            "promote_tuned_params": tuned_beats_outer,
            "reason": (
                "best trial beat the baseline on the untouched outer holdout"
                if tuned_beats_outer
                else "best trial did not beat the baseline on the untouched outer holdout"
            ),
        },
        "artifacts": {
            "scores_csv": str(SCORES_CSV.resolve(strict=False)),
            "baseline_inner_predictions_csv": str((PROCESSED_DIR / "optuna_v6_baseline_inner_predictions.csv").resolve(strict=False)),
            "baseline_outer_predictions_csv": str((PROCESSED_DIR / "optuna_v6_baseline_outer_predictions.csv").resolve(strict=False)),
            "best_inner_predictions_csv": str((PROCESSED_DIR / "optuna_v6_best_trial_inner_predictions.csv").resolve(strict=False)),
            "best_outer_predictions_csv": str((PROCESSED_DIR / "optuna_v6_best_trial_outer_predictions.csv").resolve(strict=False)),
        },
        "score_table": score_df.to_dict(orient="records"),
    }
    BEST_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("=" * 60)
    logger.info("TUNING COMPLETE")
    logger.info("Study: %s", STUDY_NAME)
    logger.info("Completed trials: %d", len(completed_trials))
    logger.info(
        "Baseline inner=%.4f outer=%.4f | Best trial #%d inner=%.4f outer=%.4f",
        baseline_report["inner_selection"]["log_loss"],
        baseline_report["outer_holdout"]["log_loss"],
        best_trial.number,
        best_report["inner_selection"]["log_loss"],
        best_report["outer_holdout"]["log_loss"],
    )
    logger.info(
        "Promotion decision: %s",
        "PROMOTE" if tuned_beats_outer else "DO NOT PROMOTE",
    )
    logger.info("Saved report to %s", BEST_JSON)
    logger.info("Saved score table to %s", SCORES_CSV)
    logger.info("Study DB at %s", STUDY_DB)


if __name__ == "__main__":
    main()
