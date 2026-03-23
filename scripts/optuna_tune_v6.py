"""
Optuna hyperparameter tuning for V6 model.

Searches over XGBoost hyperparameters + training process settings using
walk-forward cross-validation (same structure as production evaluation).

Usage:
    python scripts/optuna_tune_v6.py [--n-trials 150] [--timeout 3600]

Results are saved to:
    data/processed/optuna_v6_study.db   (Optuna SQLite storage, resumable)
    data/processed/optuna_v6_best.json   (best params as JSON)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR
from src.data.kaggle_loader import load_kaggle_dataset
from src.data.ufc_refresh import build_training_dataset_variants
from src.features.build_features import build_features
from src.model.training_spec import full_live_contract_v6_spec
from src.strategy.model_lab import generate_variant_fold_predictions
from src.strategy.model_variants import VariantConfig

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PROCESSED_DIR = Path("data/processed")
STUDY_DB = PROCESSED_DIR / "optuna_v6_study.db"
BEST_JSON = PROCESSED_DIR / "optuna_v6_best.json"

# Cutoff: only evaluate predictions from 2022+
EVAL_CUTOFF = "2022-01-01"


def _load_features() -> pd.DataFrame:
    """Load and build the v6 feature frame (cached after first call)."""
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


def _build_variant(trial: optuna.Trial) -> VariantConfig:
    """Sample hyperparameters for a single Optuna trial."""
    v6_spec = full_live_contract_v6_spec()

    # XGBoost hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 150, 800, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 7)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 0.95, step=0.05)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.3, 0.9, step=0.05)
    min_child_weight = trial.suggest_int("min_child_weight", 3, 20)
    gamma = trial.suggest_float("gamma", 0.0, 0.5, step=0.05)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 2.0, step=0.1)
    reg_lambda = trial.suggest_float("reg_lambda", 0.5, 5.0, step=0.25)

    xgb_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "gamma": gamma,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "scale_pos_weight": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
        "use_label_encoder": False,
    }

    # Training process params
    time_decay = trial.suggest_int("time_decay_half_life", 365, 1460, step=365)
    odds_noise = trial.suggest_float("odds_noise_std", 0.02, 0.08, step=0.01)

    # Calibration method
    cal_method = trial.suggest_categorical("calibration_method", ["isotonic", "sigmoid"])
    cal_cv = trial.suggest_categorical("calibration_cv", ["timeseries_5fold", "temporal_holdout"])

    variant = VariantConfig(
        name=f"optuna_trial_{trial.number}",
        description=f"Optuna trial {trial.number}",
        feature_cols=v6_spec.feature_cols,
        xgb_params=xgb_params,
        calibration_method=cal_method,
        calibration_cv=cal_cv,
        time_decay_half_life=time_decay,
        odds_noise_std=odds_noise,
        add_rematch_features=True,
    )
    # Mark native NaN mode
    variant._native_nan = True

    return variant


def objective(trial: optuna.Trial, features_df: pd.DataFrame) -> float:
    """Optuna objective: walk-forward log loss on 2022+ predictions."""
    variant = _build_variant(trial)

    try:
        fold_predictions = generate_variant_fold_predictions(
            features_df,
            variant,
            retrain_months=6,
            initial_train_years=5,
            bet_start_date=EVAL_CUTOFF,
            feature_cols=variant.feature_cols,
        )
    except Exception as e:
        logger.warning("Trial %d failed during training: %s", trial.number, e)
        return float("inf")

    if not fold_predictions:
        logger.warning("Trial %d produced no folds", trial.number)
        return float("inf")

    predictions_df = pd.concat(
        [preds for _, preds in fold_predictions], ignore_index=True
    )
    predictions_df["event_date"] = pd.to_datetime(
        predictions_df["event_date"], errors="coerce"
    )
    predictions_df = predictions_df[
        predictions_df["event_date"] >= pd.Timestamp(EVAL_CUTOFF)
    ]

    if len(predictions_df) < 50:
        logger.warning("Trial %d: only %d predictions", trial.number, len(predictions_df))
        return float("inf")

    y_true = predictions_df["target"].astype(int).to_numpy()
    y_prob = predictions_df["prob_a"].astype(float).to_numpy()

    # Clip probabilities to avoid log(0)
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)

    ll = log_loss(y_true, y_prob, labels=[0, 1])

    # Report intermediate value for pruning
    trial.report(ll, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    logger.info(
        "Trial %d: log_loss=%.4f, n=%d | depth=%d, lr=%.4f, n_est=%d, colsample=%.2f",
        trial.number, ll, len(predictions_df),
        trial.params["max_depth"],
        trial.params["learning_rate"],
        trial.params["n_estimators"],
        trial.params["colsample_bytree"],
    )

    return ll


def main():
    parser = argparse.ArgumentParser(description="Optuna V6 hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=150, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    args = parser.parse_args()

    logger.info("Loading features...")
    features_df = _load_features()
    logger.info("Features loaded: %d rows, %d columns", *features_df.shape)

    # Create resumable study with SQLite storage
    storage = f"sqlite:///{STUDY_DB}"
    study = optuna.create_study(
        study_name="v6_hyperparam_tuning",
        storage=storage,
        load_if_exists=True,
        direction="minimize",  # minimize log_loss
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    # Enqueue current production params as first trial (baseline)
    study.enqueue_trial({
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
    })

    start = time.time()
    study.optimize(
        lambda trial: objective(trial, features_df),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )
    elapsed = time.time() - start

    # Report results
    best = study.best_trial
    logger.info("=" * 60)
    logger.info("TUNING COMPLETE")
    logger.info("Trials completed: %d", len(study.trials))
    logger.info("Time elapsed: %.1f minutes", elapsed / 60)
    logger.info("Best log_loss: %.4f (trial %d)", best.value, best.number)
    logger.info("Best params:")
    for k, v in sorted(best.params.items()):
        logger.info("  %s: %s", k, v)

    # Save best params
    result = {
        "best_log_loss": best.value,
        "best_trial": best.number,
        "n_trials": len(study.trials),
        "elapsed_minutes": round(elapsed / 60, 1),
        "best_params": best.params,
        "current_production_params": {
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
        },
    }
    BEST_JSON.write_text(json.dumps(result, indent=2))
    logger.info("Best params saved to %s", BEST_JSON)
    logger.info("Study DB at %s (resumable with --n-trials)", STUDY_DB)


if __name__ == "__main__":
    main()
