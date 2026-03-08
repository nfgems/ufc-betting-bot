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

from src.config import TRAIN_CUTOFF_DATE, MODELS_DIR, PROCESSED_DATA_DIR
from src.features.build_features import get_feature_columns

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

    # Fill NaNs with column medians
    col_medians = np.nanmedian(X_train, axis=0)
    for i in range(X_train.shape[1]):
        mask = np.isnan(X_train[:, i])
        X_train[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

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
    xgb.fit(X_train, y_train)

    model = xgb
    if calibrate:
        model = CalibratedClassifierCV(xgb, cv=5, method="isotonic")
        model.fit(X_train, y_train)

    # Feature importance from the raw XGBoost model
    importance = dict(zip(feature_cols, xgb.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    logger.info("Top 10 features:")
    for feat, imp in list(importance.items())[:10]:
        logger.info(f"  {feat}: {imp:.4f}")

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
    Train both XGBoost and Logistic Regression models.
    Saves models to disk. Returns dict of model results.
    """
    train_df, test_df, feature_cols = prepare_train_test(features_df)

    # Train models
    logger.info("Training XGBoost...")
    xgb_result = train_xgboost(train_df, feature_cols)

    logger.info("Training Logistic Regression...")
    lr_result = train_logistic(train_df, feature_cols)

    # Save models
    xgb_path = MODELS_DIR / "xgboost_model.pkl"
    lr_path = MODELS_DIR / "logistic_model.pkl"
    joblib.dump(xgb_result, xgb_path)
    joblib.dump(lr_result, lr_path)
    logger.info(f"Saved XGBoost to {xgb_path}")
    logger.info(f"Saved Logistic Regression to {lr_path}")

    # Save test set for evaluation
    test_path = PROCESSED_DATA_DIR / "test_set.csv"
    test_df.to_csv(test_path, index=False)

    return {
        "xgboost": xgb_result,
        "logistic": lr_result,
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
    }


def load_model(model_name: str = "xgboost") -> dict:
    """Load a saved model from disk."""
    path = MODELS_DIR / f"{model_name}_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)
