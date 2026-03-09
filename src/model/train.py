"""
Model training — trains XGBoost and Logistic Regression models
for UFC fight prediction with probability calibration.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    TRAIN_CUTOFF_DATE, MODELS_DIR, PROCESSED_DATA_DIR,
    TIME_DECAY_ENABLED, TIME_DECAY_HALF_LIFE_DAYS,
    ODDS_NOISE_STD,
)
from src.features.build_features import get_feature_columns, get_feature_columns_no_odds, ODDS_FEATURE_NAMES

logger = logging.getLogger(__name__)


def prepare_train_test(
    features_df: pd.DataFrame,
    cutoff_date: Optional[str] = None,
    min_fights: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Split features into train/test sets by date.
    Filters out fights where either fighter has fewer than min_fights prior bouts.

    Returns (train_df, test_df, feature_columns).
    """
    cutoff = pd.Timestamp(cutoff_date or TRAIN_CUTOFF_DATE)
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

    train = df[df["event_date"] < cutoff].copy()
    test = df[df["event_date"] >= cutoff].copy()

    logger.info(
        f"Train: {len(train)} fights (before {cutoff.date()}), "
        f"Test: {len(test)} fights (after {cutoff.date()})"
    )
    logger.info(f"Using {len(feature_cols)} features")

    return train, test, feature_cols


def _compute_sample_weights(train_df: pd.DataFrame) -> Optional[np.ndarray]:
    """Compute time-decay sample weights based on fight recency."""
    if not TIME_DECAY_ENABLED:
        return None
    if "event_date" not in train_df.columns:
        return None

    dates = pd.to_datetime(train_df["event_date"])
    max_date = dates.max()
    days_ago = (max_date - dates).dt.days.values.astype(float)

    # Exponential decay: weight = 2^(-days_ago / half_life)
    weights = np.power(2.0, -days_ago / TIME_DECAY_HALF_LIFE_DAYS)

    # Normalize so mean weight = 1.0 (preserves effective sample size interpretation)
    weights = weights / weights.mean()

    logger.info(
        f"Time-decay weights: half-life={TIME_DECAY_HALF_LIFE_DAYS}d, "
        f"min={weights.min():.3f}, max={weights.max():.3f}, "
        f"effective N={weights.sum():.0f}/{len(weights)}"
    )
    return weights


def _add_odds_noise(
    X: np.ndarray,
    feature_cols: list[str],
    noise_std: float = ODDS_NOISE_STD,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Add Gaussian noise to odds-derived features to mitigate closing odds leakage.

    Training data contains closing odds (final lines right before the fight), but
    at prediction time we only have current/opening odds. Closing odds are more
    accurate, so the model overfits to their precision. Adding noise simulates
    the gap between current odds and closing odds.

    Implied probabilities are clipped to [0.02, 0.98] after noise.
    """
    if noise_std <= 0:
        return X

    if rng is None:
        rng = np.random.RandomState(42)

    odds_indices = [i for i, col in enumerate(feature_cols) if col in ODDS_FEATURE_NAMES]

    if not odds_indices:
        return X

    X = X.copy()
    for idx in odds_indices:
        col_name = feature_cols[idx]
        noise = rng.normal(0, noise_std, size=X.shape[0])
        X[:, idx] = X[:, idx] + noise

        # Clip implied probabilities to valid range
        if "implied_prob" in col_name and "diff" not in col_name:
            X[:, idx] = np.clip(X[:, idx], 0.02, 0.98)

    noised_cols = [feature_cols[i] for i in odds_indices]
    logger.info(f"Added odds noise (std={noise_std}) to {len(odds_indices)} features: {noised_cols}")

    return X


def train_xgboost(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    calibrate: bool = True,
) -> dict:
    """
    Train an XGBoost classifier with optional probability calibration.

    Returns dict with:
        - model: trained model (or calibrated wrapper)
        - feature_cols: list of feature columns used
        - feature_importance: dict of feature name -> importance
    """
    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values

    # Fill NaNs with column medians + add missing value indicators
    col_medians = np.nanmedian(X_train, axis=0)
    indicator_cols = []
    indicator_names = []
    for i in range(X_train.shape[1]):
        mask = np.isnan(X_train[:, i])
        if mask.any():
            indicator_cols.append(mask.astype(float))
            indicator_names.append(f"{feature_cols[i]}_missing")
        X_train[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

    # Append missing indicator columns (preserves debutant/limited-record signal)
    if indicator_cols:
        X_train = np.column_stack([X_train] + indicator_cols)
        feature_cols = list(feature_cols) + indicator_names
        logger.info(f"Added {len(indicator_cols)} missing value indicator features")

    # Add noise to odds features to mitigate closing odds leakage
    X_train = _add_odds_noise(X_train, feature_cols)

    # Compute time-decay sample weights
    sample_weights = _compute_sample_weights(train_df)

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1.0,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weights)

    model = xgb
    if calibrate:
        model = CalibratedClassifierCV(xgb, cv=5, method="isotonic")
        model.fit(X_train, y_train, sample_weight=sample_weights)

    # Feature importance from the raw XGBoost model
    # xgb sees all features (base + indicators), so importances align with feature_cols
    importance = dict(zip(feature_cols, xgb.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    logger.info("Top 10 features:")
    for feat, imp in list(importance.items())[:10]:
        logger.info(f"  {feat}: {imp:.4f}")

    # Extend col_medians to cover indicator columns (median = 0 for indicators)
    if indicator_names:
        indicator_medians = np.zeros(len(indicator_names))
        col_medians = np.concatenate([col_medians, indicator_medians])

    return {
        "model": model,
        "raw_model": xgb,
        "feature_cols": feature_cols,
        "feature_importance": importance,
        "col_medians": col_medians,
    }


def train_logistic(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """
    Train a Logistic Regression baseline with StandardScaler.
    LR naturally produces calibrated probabilities.
    """
    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values

    col_medians = np.nanmedian(X_train, axis=0)
    for i in range(X_train.shape[1]):
        mask = np.isnan(X_train[:, i])
        X_train[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

    # Add noise to odds features to mitigate closing odds leakage
    X_train = _add_odds_noise(X_train, feature_cols)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])
    pipeline.fit(X_train, y_train)

    # Feature importance via coefficients
    coefs = pipeline.named_steps["lr"].coef_[0]
    importance = dict(zip(feature_cols, np.abs(coefs)))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    return {
        "model": pipeline,
        "feature_cols": feature_cols,
        "feature_importance": importance,
        "col_medians": col_medians,
    }


def train_all_models(features_df: pd.DataFrame) -> dict:
    """
    Train XGBoost, Logistic Regression, and no-odds baseline models.
    Saves models to disk. Returns dict of model results.
    """
    train_df, test_df, feature_cols = prepare_train_test(features_df)

    # Train models
    logger.info("Training XGBoost...")
    xgb_result = train_xgboost(train_df, feature_cols)

    logger.info("Training Logistic Regression...")
    lr_result = train_logistic(train_df, feature_cols)

    # Train no-odds baseline (fighter stats only — measures independent edge)
    no_odds_cols = get_feature_columns_no_odds(features_df)
    no_odds_cols = [c for c in no_odds_cols if c in train_df.columns]
    logger.info(f"Training XGBoost (no-odds baseline, {len(no_odds_cols)} features)...")
    xgb_no_odds_result = train_xgboost(train_df, no_odds_cols)

    # Save models
    xgb_path = MODELS_DIR / "xgboost_model.pkl"
    lr_path = MODELS_DIR / "logistic_model.pkl"
    no_odds_path = MODELS_DIR / "xgboost_no_odds_model.pkl"
    joblib.dump(xgb_result, xgb_path)
    joblib.dump(lr_result, lr_path)
    joblib.dump(xgb_no_odds_result, no_odds_path)
    logger.info(f"Saved XGBoost to {xgb_path}")
    logger.info(f"Saved Logistic Regression to {lr_path}")
    logger.info(f"Saved XGBoost (no-odds) to {no_odds_path}")

    # Save test set for evaluation
    test_path = PROCESSED_DATA_DIR / "test_set.csv"
    test_df.to_csv(test_path, index=False)

    return {
        "xgboost": xgb_result,
        "logistic": lr_result,
        "xgboost_no_odds": xgb_no_odds_result,
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "no_odds_feature_cols": no_odds_cols,
    }


def load_model(model_name: str = "xgboost") -> dict:
    """Load a saved model from disk."""
    path = MODELS_DIR / f"{model_name}_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)
