"""
Experimental model training — isolated from production.

Trains XGBoost models using only 2022+ data with improvements:
- Temporal cross-validation for calibration (no leakage)
- Optuna-optimized hyperparameters (optional)
- SHAP-based feature selection (optional)
- Experimental features (optional)

All output goes to models/experimental/ — production models are never touched.

Usage:
    python -m src.model.train_experimental
    python -m src.model.train_experimental --use-optuna --use-shap --top-features 40
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from src.config import (
    PROCESSED_DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    TIME_DECAY_ENABLED,
    TIME_DECAY_HALF_LIFE_DAYS,
    ODDS_NOISE_STD,
)
from src.features.build_features import (
    get_feature_columns,
    get_feature_columns_no_odds,
    ODDS_FEATURE_NAMES,
)
from src.model.train import _compute_sample_weights, _add_odds_noise

logger = logging.getLogger(__name__)

# Isolated output directories
EXPERIMENTAL_MODELS_DIR = MODELS_DIR / "experimental"
EXPERIMENTAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Data window
DATA_START_DATE = "2022-01-01"
TRAIN_TEST_SPLIT_DATE = "2024-07-01"


def prepare_experimental_data(
    features_df: pd.DataFrame,
    start_date: str = DATA_START_DATE,
    split_date: str = TRAIN_TEST_SPLIT_DATE,
    min_fights: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Prepare train/test data using only 2022+ fights.

    Returns (train_df, test_df, feature_columns).
    """
    feature_cols = get_feature_columns(features_df)

    df = features_df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])

    # Filter to 2022+ only
    df = df[df["event_date"] >= pd.Timestamp(start_date)]

    # Minimum experience filter
    if "a_num_fights" in df.columns and "b_num_fights" in df.columns:
        df = df[
            (df["a_num_fights"] >= min_fights) & (df["b_num_fights"] >= min_fights)
        ]

    df = df.dropna(subset=["target"])
    df = df.dropna(subset=feature_cols, how="all")

    cutoff = pd.Timestamp(split_date)
    train = df[df["event_date"] < cutoff].copy()
    test = df[df["event_date"] >= cutoff].copy()

    logger.info(
        f"Experimental data (>= {start_date}): "
        f"Train: {len(train)} fights (before {split_date}), "
        f"Test: {len(test)} fights (after {split_date})"
    )
    logger.info(f"Using {len(feature_cols)} features")

    return train, test, feature_cols


def train_experimental_xgboost(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    xgb_params: Optional[dict] = None,
    calibration_method: str = "isotonic",
    calibration_cv: str = "timeseries_5fold",
    impute_with_indicators: bool = True,
    odds_noise_std: float = ODDS_NOISE_STD,
) -> dict:
    """
    Train an experimental XGBoost model with temporal calibration.

    Key differences from production train_xgboost():
    - Uses TimeSeriesSplit for calibration by default (no temporal leakage)
    - Supports custom hyperparameters (e.g., from Optuna)
    - Always adds missing value indicators
    """
    X_train = train_df[feature_cols].values.copy()
    y_train = train_df["target"].values

    # Imputation with missing indicators
    col_medians = np.nanmedian(X_train, axis=0)
    indicator_cols_data = []
    indicator_names = []
    indicator_indices = []  # Track WHICH feature indices had NaN

    if impute_with_indicators:
        for i in range(X_train.shape[1]):
            mask = np.isnan(X_train[:, i])
            if mask.any():
                indicator_cols_data.append(mask.astype(float))
                indicator_names.append(f"{feature_cols[i]}_missing")
                indicator_indices.append(i)
            X_train[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

        if indicator_cols_data:
            X_train = np.column_stack([X_train] + indicator_cols_data)
            logger.info(f"Added {len(indicator_cols_data)} missing value indicator features")
    else:
        for i in range(X_train.shape[1]):
            mask = np.isnan(X_train[:, i])
            X_train[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

    # Add noise to odds features
    X_train = _add_odds_noise(X_train, feature_cols, noise_std=odds_noise_std)

    # Time-decay sample weights
    sample_weights = _compute_sample_weights(train_df)

    # XGBoost training
    if xgb_params is None:
        xgb_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": 1.0,
            "eval_metric": "logloss",
            "random_state": 42,
        }

    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_train, y_train, sample_weight=sample_weights)

    # Calibration with temporal CV
    model = xgb
    if calibration_method != "none":
        if calibration_cv == "timeseries_5fold":
            model = CalibratedClassifierCV(
                xgb, cv=TimeSeriesSplit(n_splits=5), method=calibration_method
            )
            model.fit(X_train, y_train, sample_weight=sample_weights)
        elif calibration_cv == "random_5fold":
            model = CalibratedClassifierCV(xgb, cv=5, method=calibration_method)
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            logger.warning(f"Unknown calibration_cv: {calibration_cv}, skipping calibration")

    # Feature importance
    importance = dict(zip(feature_cols, xgb.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # Extend medians for indicator columns
    if indicator_names:
        indicator_medians = np.zeros(len(indicator_names))
        col_medians = np.concatenate([col_medians, indicator_medians])

    return {
        "model": model,
        "raw_model": xgb,
        "feature_cols": feature_cols,
        "feature_importance": importance,
        "col_medians": col_medians,
        "impute_strategy": "median" if impute_with_indicators else "native_nan",
        "n_indicator_cols": len(indicator_cols_data),
        "indicator_indices": indicator_indices,  # Which feature indices had NaN
        "xgb_params": xgb_params,
        "calibration_method": calibration_method,
        "calibration_cv": calibration_cv,
    }


def evaluate_experimental(
    test_df: pd.DataFrame,
    model_result: dict,
    label: str = "experimental",
) -> dict:
    """Evaluate a model on test data and return metrics."""
    feature_cols = model_result["feature_cols"]
    col_medians = model_result["col_medians"]
    indicator_indices = model_result.get("indicator_indices", [])
    model = model_result["model"]

    X_test = test_df[feature_cols].values.copy()

    # Capture NaN masks BEFORE imputation (needed for indicator columns)
    nan_masks = {}
    if indicator_indices:
        for i in indicator_indices:
            nan_masks[i] = np.isnan(X_test[:, i])

    # Impute NaNs with training medians
    for i in range(X_test.shape[1]):
        mask = np.isnan(X_test[:, i])
        X_test[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

    # Add indicator columns for the SAME features that had NaN during training
    if indicator_indices:
        indicator_cols_data = [nan_masks[i].astype(float) for i in indicator_indices]
        X_test = np.column_stack([X_test] + indicator_cols_data)

    y_true = test_df["target"].values
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    logloss = log_loss(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    baseline_acc = max(y_true.mean(), 1 - y_true.mean())

    metrics = {
        "label": label,
        "accuracy": accuracy,
        "log_loss": logloss,
        "brier_score": brier,
        "baseline_accuracy": baseline_acc,
        "accuracy_vs_baseline": accuracy - baseline_acc,
        "n_test_fights": len(y_true),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Model: {label}")
    logger.info(f"Test fights: {len(y_true)}")
    logger.info(f"Accuracy: {accuracy:.4f} (baseline: {baseline_acc:.4f}, diff: {accuracy - baseline_acc:+.4f})")
    logger.info(f"Log loss: {logloss:.4f}")
    logger.info(f"Brier score: {brier:.4f}")
    logger.info(f"{'='*50}")

    return metrics


def main():
    """Train and evaluate experimental model vs production baseline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(EXPERIMENTAL_MODELS_DIR / "train_experimental.log"),
        ],
    )

    parser = argparse.ArgumentParser(description="Train experimental XGBoost model")
    parser.add_argument("--use-optuna", action="store_true", help="Use Optuna-optimized hyperparameters")
    parser.add_argument("--use-shap", action="store_true", help="Use SHAP-based feature selection")
    parser.add_argument("--top-features", type=int, default=40, help="Number of top features to keep (with --use-shap)")
    parser.add_argument("--use-experimental-features", action="store_true", help="Add experimental features")
    parser.add_argument("--start-date", type=str, default=DATA_START_DATE, help="Data start date")
    parser.add_argument("--split-date", type=str, default=TRAIN_TEST_SPLIT_DATE, help="Train/test split date")
    args = parser.parse_args()

    # Load features — prefer promoted V5 fullfit artifact, fall back to root
    features_path = PROCESSED_DATA_DIR / "candidates" / "full_live_contract_v5_fullfit" / "features.csv"
    if not features_path.exists():
        features_path = PROCESSED_DATA_DIR / "features.csv"
    if not features_path.exists():
        logger.info("No cached features found. Building from Kaggle dataset...")
        from src.data.kaggle_loader import load_kaggle_dataset
        from src.features.build_features import build_features
        fights_df = load_kaggle_dataset()
        features_df = build_features(fights_df)
    else:
        logger.info(f"Using features from {features_path}")
        features_df = pd.read_csv(features_path, parse_dates=["event_date"])

    # Add experimental features if requested
    if args.use_experimental_features:
        from src.features.experimental_features import (
            add_experimental_features,
            get_experimental_feature_columns,
        )
        features_df = add_experimental_features(features_df)

    # Prepare data
    train_df, test_df, feature_cols = prepare_experimental_data(
        features_df,
        start_date=args.start_date,
        split_date=args.split_date,
    )

    # Override feature columns to include experimental features if present
    if args.use_experimental_features:
        feature_cols = get_experimental_feature_columns(train_df)

    # SHAP feature selection
    if args.use_shap:
        from src.model.feature_selection import select_top_features
        feature_cols = select_top_features(
            train_df, feature_cols, n_keep=args.top_features, method="shap"
        )
        logger.info(f"SHAP selected {len(feature_cols)} features")

    # Get Optuna params if requested
    xgb_params = None
    if args.use_optuna:
        from src.model.hyperparam_search import load_best_params
        xgb_params = load_best_params()
        if xgb_params:
            logger.info(f"Using Optuna params: {xgb_params}")
        else:
            logger.warning("No Optuna results found. Run hyperparam_search.py first. Using defaults.")

    # --- Train production baseline (same code path, but on 2022+ data) ---
    logger.info("\n" + "=" * 60)
    logger.info("Training PRODUCTION BASELINE (on 2022+ data)")
    logger.info("=" * 60)
    baseline_result = train_experimental_xgboost(
        train_df,
        get_feature_columns(train_df),
        xgb_params=None,  # Production defaults
        calibration_method="isotonic",
        calibration_cv="random_5fold",  # Production uses random CV
        impute_with_indicators=False,
    )

    # --- Train experimental model ---
    logger.info("\n" + "=" * 60)
    logger.info("Training EXPERIMENTAL model (on 2022+ data)")
    logger.info("=" * 60)
    experimental_result = train_experimental_xgboost(
        train_df,
        feature_cols,
        xgb_params=xgb_params,
        calibration_method="isotonic",
        calibration_cv="timeseries_5fold",  # Temporal CV
        impute_with_indicators=True,
    )

    # --- Evaluate both ---
    baseline_metrics = evaluate_experimental(test_df, baseline_result, "production_baseline")
    experimental_metrics = evaluate_experimental(test_df, experimental_result, "experimental")

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    for metric in ["accuracy", "log_loss", "brier_score"]:
        base_val = baseline_metrics[metric]
        exp_val = experimental_metrics[metric]
        diff = exp_val - base_val
        better = "lower" if metric in ("log_loss", "brier_score") else "higher"
        is_better = diff < 0 if better == "lower" else diff > 0
        symbol = "+" if is_better else "-"
        logger.info(
            f"  {metric:20s}: baseline={base_val:.4f}  experimental={exp_val:.4f}  "
            f"diff={diff:+.4f} [{symbol}]"
        )

    # Save models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_path = EXPERIMENTAL_MODELS_DIR / f"baseline_{timestamp}.pkl"
    experimental_path = EXPERIMENTAL_MODELS_DIR / f"experimental_{timestamp}.pkl"
    joblib.dump(baseline_result, baseline_path)
    joblib.dump(experimental_result, experimental_path)
    logger.info(f"\nSaved baseline to {baseline_path}")
    logger.info(f"Saved experimental to {experimental_path}")


if __name__ == "__main__":
    main()
