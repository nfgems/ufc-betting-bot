"""
SHAP-based feature selection — isolated from production.

Ranks features by their SHAP importance and selects the top N.
Also supports permutation importance as an alternative method.

Usage:
    python -m src.model.feature_selection
    python -m src.model.feature_selection --method permutation --top 30
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.config import PROCESSED_DATA_DIR, LOGS_DIR, ODDS_NOISE_STD
from src.features.build_features import get_feature_columns
from src.model.train import _compute_sample_weights, _add_odds_noise

logger = logging.getLogger(__name__)

FEATURE_SEL_DIR = LOGS_DIR / "feature_selection"
FEATURE_SEL_DIR.mkdir(parents=True, exist_ok=True)

DATA_START_DATE = "2022-01-01"
TRAIN_TEST_SPLIT_DATE = "2024-07-01"


def shap_feature_ranking(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list[str],
    sample_weights: np.ndarray = None,
    max_samples: int = 2000,
) -> list[tuple[str, float]]:
    """
    Rank features by mean |SHAP value| using TreeExplainer.

    Args:
        X_train: Training feature matrix (already imputed)
        y_train: Training labels
        feature_cols: Feature column names
        sample_weights: Optional sample weights
        max_samples: Max samples for SHAP computation (speed vs accuracy)

    Returns sorted list of (feature_name, mean_abs_shap) pairs (highest first).
    """
    import shap

    # Train a quick XGBoost for SHAP analysis
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weights)

    # Subsample for SHAP if dataset is large
    if len(X_train) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), size=max_samples, replace=False)
        X_shap = X_train[idx]
    else:
        X_shap = X_train

    # Compute SHAP values
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_shap)

    # Mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Only rank the original features (not indicator cols if present)
    n_features = min(len(feature_cols), len(mean_abs_shap))
    ranking = list(zip(feature_cols[:n_features], mean_abs_shap[:n_features]))
    ranking.sort(key=lambda x: x[1], reverse=True)

    return ranking


def permutation_feature_ranking(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: list[str],
    sample_weights: np.ndarray = None,
    n_repeats: int = 10,
) -> list[tuple[str, float]]:
    """
    Rank features by permutation importance on validation data.

    More robust than SHAP for detecting features that help generalization,
    but requires a separate validation set.
    """
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import brier_score_loss

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weights)

    # Use neg_brier_score which is built into sklearn and handles predict_proba
    result = permutation_importance(
        xgb, X_val, y_val,
        scoring="neg_brier_score",
        n_repeats=n_repeats,
        random_state=42,
    )

    n_features = min(len(feature_cols), len(result.importances_mean))
    ranking = list(zip(feature_cols[:n_features], result.importances_mean[:n_features]))
    ranking.sort(key=lambda x: x[1], reverse=True)

    return ranking


def select_top_features(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_keep: int = 40,
    method: str = "shap",
) -> list[str]:
    """
    Train a quick model, rank features, return top N feature names.

    Args:
        train_df: Training DataFrame
        feature_cols: Candidate feature columns
        n_keep: Number of features to keep
        method: "shap" or "permutation"

    Returns list of top N feature column names.
    """
    X = train_df[feature_cols].values.copy()
    y = train_df["target"].values

    # Impute NaNs
    col_medians = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

    # Add odds noise
    X = _add_odds_noise(X, feature_cols, noise_std=ODDS_NOISE_STD)

    sample_weights = _compute_sample_weights(train_df)

    if method == "shap":
        ranking = shap_feature_ranking(X, y, feature_cols, sample_weights)
    elif method == "permutation":
        # Use last 20% as validation for permutation importance
        split_idx = int(len(X) * 0.8)
        ranking = permutation_feature_ranking(
            X[:split_idx], y[:split_idx],
            X[split_idx:], y[split_idx:],
            feature_cols, sample_weights[:split_idx] if sample_weights is not None else None,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    selected = [name for name, _ in ranking[:n_keep]]

    logger.info(f"Feature selection ({method}): kept {len(selected)}/{len(feature_cols)} features")
    logger.info("Top 10 features:")
    for name, score in ranking[:10]:
        logger.info(f"  {name:40s} {score:.6f}")

    if len(ranking) > n_keep:
        logger.info(f"\nDropped {len(ranking) - n_keep} features:")
        for name, score in ranking[n_keep:n_keep + 10]:
            logger.info(f"  {name:40s} {score:.6f}")
        if len(ranking) > n_keep + 10:
            logger.info(f"  ... and {len(ranking) - n_keep - 10} more")

    return selected


def generate_report(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """
    Generate a comprehensive feature selection report.

    Runs both SHAP and permutation methods and saves results.
    """
    X = train_df[feature_cols].values.copy()
    y = train_df["target"].values

    col_medians = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

    X = _add_odds_noise(X, feature_cols, noise_std=ODDS_NOISE_STD)
    sample_weights = _compute_sample_weights(train_df)

    # SHAP ranking
    logger.info("Computing SHAP feature importance...")
    shap_ranking = shap_feature_ranking(X, y, feature_cols, sample_weights)

    # Permutation ranking
    logger.info("Computing permutation feature importance...")
    split_idx = int(len(X) * 0.8)
    perm_ranking = permutation_feature_ranking(
        X[:split_idx], y[:split_idx],
        X[split_idx:], y[split_idx:],
        feature_cols,
        sample_weights[:split_idx] if sample_weights is not None else None,
    )

    # Build report
    shap_dict = {name: score for name, score in shap_ranking}
    perm_dict = {name: score for name, score in perm_ranking}

    report_rows = []
    for i, (name, shap_score) in enumerate(shap_ranking):
        perm_score = perm_dict.get(name, 0)
        perm_rank = next((j + 1 for j, (n, _) in enumerate(perm_ranking) if n == name), len(feature_cols))
        report_rows.append({
            "feature": name,
            "shap_rank": i + 1,
            "shap_score": shap_score,
            "perm_rank": perm_rank,
            "perm_score": perm_score,
            "avg_rank": (i + 1 + perm_rank) / 2,
        })

    report_df = pd.DataFrame(report_rows).sort_values("avg_rank")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = FEATURE_SEL_DIR / f"feature_report_{timestamp}.csv"
    report_df.to_csv(report_path, index=False)
    logger.info(f"Saved feature report to {report_path}")

    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("FEATURE RANKING REPORT (sorted by average rank)")
    logger.info(f"{'='*80}")
    logger.info(f"{'Feature':40s} {'SHAP Rank':>10s} {'Perm Rank':>10s} {'Avg Rank':>10s}")
    logger.info("-" * 80)
    for _, row in report_df.head(50).iterrows():
        logger.info(
            f"{row['feature']:40s} {int(row['shap_rank']):>10d} "
            f"{int(row['perm_rank']):>10d} {row['avg_rank']:>10.1f}"
        )

    return {
        "shap_ranking": shap_ranking,
        "perm_ranking": perm_ranking,
        "report": report_df,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(FEATURE_SEL_DIR / "feature_selection.log"),
        ],
    )

    parser = argparse.ArgumentParser(description="SHAP/permutation feature selection")
    parser.add_argument("--method", type=str, default="both", choices=["shap", "permutation", "both"],
                        help="Feature ranking method")
    parser.add_argument("--top", type=int, default=40, help="Number of top features to select")
    parser.add_argument("--start-date", type=str, default=DATA_START_DATE, help="Data start date")
    parser.add_argument("--split-date", type=str, default=TRAIN_TEST_SPLIT_DATE, help="Train cutoff date")
    args = parser.parse_args()

    # Load features
    features_path = PROCESSED_DATA_DIR / "features.csv"
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        sys.exit(1)

    features_df = pd.read_csv(features_path, parse_dates=["event_date"])

    # Filter to 2022+ training data
    df = features_df[features_df["event_date"] >= pd.Timestamp(args.start_date)]
    df = df[df["event_date"] < pd.Timestamp(args.split_date)]
    if "a_num_fights" in df.columns and "b_num_fights" in df.columns:
        df = df[(df["a_num_fights"] >= 2) & (df["b_num_fights"] >= 2)]
    df = df.dropna(subset=["target"])

    feature_cols = get_feature_columns(df)

    if args.method == "both":
        generate_report(df, feature_cols)
    else:
        selected = select_top_features(df, feature_cols, n_keep=args.top, method=args.method)
        logger.info(f"\nSelected {len(selected)} features:")
        for f in selected:
            logger.info(f"  {f}")


if __name__ == "__main__":
    main()
