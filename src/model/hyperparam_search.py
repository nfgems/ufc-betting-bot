"""
Optuna hyperparameter search for XGBoost — isolated from production.

Uses TimeSeriesSplit cross-validation to find optimal hyperparameters
on 2022+ data. Results are saved to logs/optuna/ and can be loaded
by train_experimental.py.

Usage:
    python -m src.model.hyperparam_search
    python -m src.model.hyperparam_search --trials 200 --metric brier
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from src.config import PROCESSED_DATA_DIR, LOGS_DIR, ODDS_NOISE_STD
from src.features.build_features import get_feature_columns, ODDS_FEATURE_NAMES
from src.model.train import _compute_sample_weights, _add_odds_noise

logger = logging.getLogger(__name__)

OPTUNA_DIR = LOGS_DIR / "optuna"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

# Data window (matches train_experimental.py)
DATA_START_DATE = "2022-01-01"
TRAIN_TEST_SPLIT_DATE = "2024-07-01"


def run_optuna_search(
    features_df: pd.DataFrame,
    n_trials: int = 100,
    n_cv_splits: int = 5,
    metric: str = "brier",
    start_date: str = DATA_START_DATE,
    split_date: str = TRAIN_TEST_SPLIT_DATE,
    min_fights: int = 2,
) -> dict:
    """
    Run Optuna Bayesian hyperparameter search with temporal CV.

    Args:
        features_df: Full feature DataFrame
        n_trials: Number of Optuna trials
        n_cv_splits: Number of TimeSeriesSplit folds
        metric: Optimization metric ("brier" or "logloss")
        start_date: Only use data from this date onward
        split_date: Only use training data before this date for search

    Returns dict with best_params and study summary.
    """
    import optuna
    from optuna.pruners import MedianPruner

    # Prepare data (train portion only — never peek at test)
    df = features_df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df[df["event_date"] >= pd.Timestamp(start_date)]
    df = df[df["event_date"] < pd.Timestamp(split_date)]

    if "a_num_fights" in df.columns and "b_num_fights" in df.columns:
        df = df[(df["a_num_fights"] >= min_fights) & (df["b_num_fights"] >= min_fights)]

    df = df.dropna(subset=["target"])
    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols, how="all")
    df = df.sort_values("event_date").reset_index(drop=True)

    X = df[feature_cols].values.copy()
    y = df["target"].values

    # Impute NaNs with median
    col_medians = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

    # Add odds noise
    X = _add_odds_noise(X, feature_cols, noise_std=ODDS_NOISE_STD)

    # Sample weights
    sample_weights = _compute_sample_weights(df)

    # CV splitter
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)
    cv_splits = list(tscv.split(X))

    logger.info(f"Optuna search: {n_trials} trials, {n_cv_splits}-fold temporal CV")
    logger.info(f"Data: {len(X)} fights, {len(feature_cols)} features")
    logger.info(f"Metric: {metric}")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": 1.0,
            "eval_metric": "logloss",
            "random_state": 42,
            "use_label_encoder": False,
        }

        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            w_tr = sample_weights[train_idx] if sample_weights is not None else None

            xgb = XGBClassifier(**params)
            xgb.fit(X_tr, y_tr, sample_weight=w_tr)

            y_prob = xgb.predict_proba(X_val)[:, 1]

            if metric == "brier":
                score = brier_score_loss(y_val, y_prob)
            else:
                score = log_loss(y_val, y_prob)

            scores.append(score)

            # Pruning: report intermediate results
            trial.report(np.mean(scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    # Run optimization
    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        study_name=f"xgb_tuning_{metric}",
    )

    # Seed with production params as first trial
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
    })

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Results
    best_params = study.best_params
    # Add fixed params back
    best_params["scale_pos_weight"] = 1.0
    best_params["eval_metric"] = "logloss"
    best_params["random_state"] = 42
    best_params["use_label_encoder"] = False

    logger.info(f"\nBest {metric}: {study.best_value:.6f}")
    logger.info(f"Best params: {best_params}")

    # Compare against production baseline
    prod_trial = study.trials[0]  # First trial is production params
    logger.info(f"Production {metric}: {prod_trial.value:.6f}")
    improvement = prod_trial.value - study.best_value
    logger.info(f"Improvement: {improvement:+.6f} ({improvement / prod_trial.value * 100:+.2f}%)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "best_params": best_params,
        "best_score": study.best_value,
        "production_score": prod_trial.value,
        "improvement": improvement,
        "metric": metric,
        "n_trials": n_trials,
        "n_cv_splits": n_cv_splits,
        "data_start": start_date,
        "data_end": split_date,
        "n_fights": len(X),
        "n_features": len(feature_cols),
        "timestamp": timestamp,
    }

    # Save as JSON (human-readable)
    json_path = OPTUNA_DIR / f"best_params_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Saved results to {json_path}")

    # Also save as "latest" for easy loading
    latest_path = OPTUNA_DIR / "best_params_latest.json"
    with open(latest_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Save top 10 trials summary
    trials_summary = []
    for t in sorted(study.trials, key=lambda x: x.value if x.value is not None else float("inf"))[:10]:
        if t.value is not None:
            trials_summary.append({
                "number": t.number,
                "value": t.value,
                "params": t.params,
            })

    summary_path = OPTUNA_DIR / f"top_trials_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(trials_summary, f, indent=2, default=str)

    return result


def load_best_params() -> Optional[dict]:
    """Load the best hyperparameters from the most recent Optuna search."""
    latest_path = OPTUNA_DIR / "best_params_latest.json"
    if not latest_path.exists():
        return None

    with open(latest_path) as f:
        result = json.load(f)

    return result.get("best_params")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(OPTUNA_DIR / "hyperparam_search.log"),
        ],
    )

    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for XGBoost")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--metric", type=str, default="brier", choices=["brier", "logloss"],
                        help="Optimization metric")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of TimeSeriesSplit folds")
    parser.add_argument("--start-date", type=str, default=DATA_START_DATE, help="Data start date")
    parser.add_argument("--split-date", type=str, default=TRAIN_TEST_SPLIT_DATE, help="Train cutoff date")
    args = parser.parse_args()

    # Load features — prefer promoted V5 fullfit artifact, fall back to root
    features_path = PROCESSED_DATA_DIR / "candidates" / "full_live_contract_v5_fullfit" / "features.csv"
    if not features_path.exists():
        features_path = PROCESSED_DATA_DIR / "features.csv"
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        logger.error("Run 'python -m src.bot build' first to generate features.")
        sys.exit(1)
    logger.info(f"Using features from {features_path}")

    features_df = pd.read_csv(features_path, parse_dates=["event_date"])

    result = run_optuna_search(
        features_df,
        n_trials=args.trials,
        n_cv_splits=args.cv_splits,
        metric=args.metric,
        start_date=args.start_date,
        split_date=args.split_date,
    )

    logger.info("\nDone! Use the best params with:")
    logger.info("  python -m src.model.train_experimental --use-optuna")


if __name__ == "__main__":
    main()
