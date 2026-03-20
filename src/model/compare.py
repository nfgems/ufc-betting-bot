"""
Baseline comparison script — the main deliverable.

Trains production baseline and experimental challenger on the same 2022+ data,
runs walk-forward evaluation, and produces a side-by-side comparison with
statistical significance tests.

No production files are modified. All output goes to logs/comparison/.

Usage:
    python -m src.model.compare
    python -m src.model.compare --use-optuna --use-shap --top-features 40
    python -m src.model.compare --use-experimental-features
    python -m src.model.compare --use-optuna --use-shap --use-experimental-features
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
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

from src.config import PROCESSED_DATA_DIR, LOGS_DIR
from src.features.build_features import get_feature_columns
from src.model.train_experimental import (
    train_experimental_xgboost,
    evaluate_experimental,
    DATA_START_DATE,
    TRAIN_TEST_SPLIT_DATE,
)
from src.strategy.lab_stats import compute_ece

logger = logging.getLogger(__name__)

COMPARISON_DIR = LOGS_DIR / "comparison"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


def mcnemar_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> dict:
    """
    McNemar's test for paired accuracy comparison.

    Tests whether two models have significantly different error rates.
    Small p-value means the accuracy difference is unlikely due to chance.
    """
    # Build contingency table
    a_correct = (preds_a == y_true)
    b_correct = (preds_b == y_true)

    # b = A right, B wrong | c = A wrong, B right
    b = np.sum(a_correct & ~b_correct)
    c = np.sum(~a_correct & b_correct)

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": int(b), "c": int(c)}

    # McNemar's test with continuity correction
    statistic = (abs(b - c) - 1) ** 2 / (b + c)

    # Chi-squared distribution with 1 degree of freedom
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(statistic, df=1)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "b": int(b),  # baseline correct, challenger wrong
        "c": int(c),  # baseline wrong, challenger correct
    }


def bootstrap_metric_comparison(
    y_true: np.ndarray,
    probs_baseline: np.ndarray,
    probs_challenger: np.ndarray,
    metric_fn,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap confidence intervals for the difference in a metric.

    Returns CI for (challenger - baseline). Negative means challenger is better
    for metrics where lower is better (Brier, log loss).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    baseline_val = metric_fn(y_true, probs_baseline)
    challenger_val = metric_fn(y_true, probs_challenger)
    observed_diff = challenger_val - baseline_val

    diffs = np.zeros(n_bootstrap)
    valid_count = 0
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        # Skip bootstrap samples with only one class (log_loss requires both)
        if len(np.unique(y_true[idx])) < 2:
            continue
        base_boot = metric_fn(y_true[idx], probs_baseline[idx])
        chal_boot = metric_fn(y_true[idx], probs_challenger[idx])
        diffs[valid_count] = chal_boot - base_boot
        valid_count += 1
    diffs = diffs[:valid_count]

    if valid_count == 0:
        return {
            "baseline": float(baseline_val),
            "challenger": float(challenger_val),
            "diff": float(observed_diff),
            "ci_lo": 0.0,
            "ci_hi": 0.0,
            "p_value": 1.0,
        }

    return {
        "baseline": float(baseline_val),
        "challenger": float(challenger_val),
        "diff": float(observed_diff),
        "ci_lo": float(np.percentile(diffs, 2.5)),
        "ci_hi": float(np.percentile(diffs, 97.5)),
        "p_value": float(np.mean(diffs >= 0)),  # P(challenger not better) for lower-is-better
    }


def walk_forward_comparison(
    features_df: pd.DataFrame,
    baseline_feature_cols: list[str],
    challenger_feature_cols: list[str],
    challenger_xgb_params: Optional[dict] = None,
    retrain_months: int = 4,
    start_date: str = DATA_START_DATE,
) -> dict:
    """
    Walk-forward backtest comparing baseline and challenger models.

    Both models are retrained on expanding windows every `retrain_months` months.
    Uses only 2022+ data.
    """
    df = features_df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df[df["event_date"] >= pd.Timestamp(start_date)]

    if "a_num_fights" in df.columns and "b_num_fights" in df.columns:
        df = df[(df["a_num_fights"] >= 2) & (df["b_num_fights"] >= 2)]
    df = df.dropna(subset=["target"])
    df = df.sort_values("event_date").reset_index(drop=True)

    dates = pd.to_datetime(df["event_date"])
    min_date = dates.min()
    max_date = dates.max()

    # Start with at least 1 year of training data
    train_end = min_date + pd.DateOffset(years=1)

    all_y_true = []
    all_baseline_probs = []
    all_challenger_probs = []
    fold_results = []
    fold_num = 0

    while train_end < max_date:
        test_end = train_end + pd.DateOffset(months=retrain_months)
        if test_end > max_date:
            test_end = max_date + pd.Timedelta(days=1)

        train_mask = dates < train_end
        test_mask = (dates >= train_end) & (dates < test_end)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(train_df) < 50 or len(test_df) < 5:
            train_end = test_end
            continue

        fold_num += 1
        logger.info(
            f"Fold {fold_num}: Train {len(train_df)}, Test {len(test_df)} "
            f"({train_end.date()} to {test_end.date()})"
        )

        # Train baseline (production config)
        baseline_result = train_experimental_xgboost(
            train_df,
            baseline_feature_cols,
            xgb_params=None,
            calibration_method="isotonic",
            calibration_cv="random_5fold",
            impute_with_indicators=False,
        )

        # Train challenger (experimental config)
        challenger_result = train_experimental_xgboost(
            train_df,
            challenger_feature_cols,
            xgb_params=challenger_xgb_params,
            calibration_method="isotonic",
            calibration_cv="timeseries_5fold",
            impute_with_indicators=True,
        )

        # Predict on test fold
        y_true = test_df["target"].values

        # Baseline predictions
        base_probs = _predict_with_result(test_df, baseline_result)
        # Challenger predictions
        chal_probs = _predict_with_result(test_df, challenger_result)

        all_y_true.extend(y_true.tolist())
        all_baseline_probs.extend(base_probs.tolist())
        all_challenger_probs.extend(chal_probs.tolist())

        # Per-fold metrics
        base_brier = brier_score_loss(y_true, base_probs)
        chal_brier = brier_score_loss(y_true, chal_probs)
        base_acc = accuracy_score(y_true, (base_probs > 0.5).astype(int))
        chal_acc = accuracy_score(y_true, (chal_probs > 0.5).astype(int))

        fold_results.append({
            "fold": fold_num,
            "train_end": str(train_end.date()),
            "test_size": len(y_true),
            "baseline_brier": base_brier,
            "challenger_brier": chal_brier,
            "baseline_accuracy": base_acc,
            "challenger_accuracy": chal_acc,
            "brier_diff": chal_brier - base_brier,
        })

        logger.info(
            f"  Baseline:   Brier={base_brier:.4f}  Acc={base_acc:.4f}"
        )
        logger.info(
            f"  Challenger: Brier={chal_brier:.4f}  Acc={chal_acc:.4f}"
        )

        train_end = test_end

    return {
        "y_true": np.array(all_y_true),
        "baseline_probs": np.array(all_baseline_probs),
        "challenger_probs": np.array(all_challenger_probs),
        "fold_results": pd.DataFrame(fold_results),
    }


def _predict_with_result(
    test_df: pd.DataFrame,
    model_result: dict,
) -> np.ndarray:
    """Generate probability predictions from a model result dict."""
    feature_cols = model_result["feature_cols"]
    col_medians = model_result["col_medians"]
    indicator_indices = model_result.get("indicator_indices", [])
    model = model_result["model"]

    X = test_df[feature_cols].values.copy()

    # Capture NaN masks BEFORE imputation
    nan_masks = {}
    if indicator_indices:
        for i in indicator_indices:
            nan_masks[i] = np.isnan(X[:, i])

    # Impute NaNs with training medians
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

    # Add indicator columns for the SAME features that had NaN during training
    if indicator_indices:
        indicator_cols_data = [nan_masks[i].astype(float) for i in indicator_indices]
        X = np.column_stack([X] + indicator_cols_data)

    return model.predict_proba(X)[:, 1]


def run_comparison(
    features_df: pd.DataFrame,
    baseline_feature_cols: Optional[list[str]] = None,
    challenger_feature_cols: Optional[list[str]] = None,
    challenger_xgb_params: Optional[dict] = None,
    retrain_months: int = 4,
    n_bootstrap: int = 10000,
) -> dict:
    """
    Full comparison pipeline: walk-forward + metrics + statistical tests.

    Returns a comprehensive results dict.
    """
    if baseline_feature_cols is None:
        baseline_feature_cols = get_feature_columns(features_df)
    if challenger_feature_cols is None:
        challenger_feature_cols = baseline_feature_cols

    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD COMPARISON: Baseline vs Challenger")
    logger.info(f"Baseline features: {len(baseline_feature_cols)}")
    logger.info(f"Challenger features: {len(challenger_feature_cols)}")
    if challenger_xgb_params:
        logger.info(f"Challenger params: {challenger_xgb_params}")
    logger.info("=" * 70)

    # Walk-forward evaluation
    wf = walk_forward_comparison(
        features_df,
        baseline_feature_cols,
        challenger_feature_cols,
        challenger_xgb_params=challenger_xgb_params,
        retrain_months=retrain_months,
    )

    y_true = wf["y_true"]
    base_probs = wf["baseline_probs"]
    chal_probs = wf["challenger_probs"]

    if len(y_true) == 0:
        logger.error("No predictions generated. Check data availability.")
        return {}

    # Aggregate metrics
    base_preds = (base_probs > 0.5).astype(int)
    chal_preds = (chal_probs > 0.5).astype(int)

    metrics = {
        "baseline": {
            "accuracy": float(accuracy_score(y_true, base_preds)),
            "brier_score": float(brier_score_loss(y_true, base_probs)),
            "log_loss": float(log_loss(y_true, base_probs)),
            "ece": float(compute_ece(y_true, base_probs)),
        },
        "challenger": {
            "accuracy": float(accuracy_score(y_true, chal_preds)),
            "brier_score": float(brier_score_loss(y_true, chal_probs)),
            "log_loss": float(log_loss(y_true, chal_probs)),
            "ece": float(compute_ece(y_true, chal_probs)),
        },
    }

    # Statistical tests
    logger.info("\n" + "=" * 70)
    logger.info("STATISTICAL TESTS")
    logger.info("=" * 70)

    # McNemar's test (accuracy)
    mcnemar = mcnemar_test(y_true, base_preds, chal_preds)
    metrics["mcnemar"] = mcnemar
    logger.info(
        f"McNemar's test: statistic={mcnemar['statistic']:.4f}, "
        f"p={mcnemar['p_value']:.4f}"
    )
    logger.info(
        f"  Baseline correct/Challenger wrong: {mcnemar['b']}"
    )
    logger.info(
        f"  Baseline wrong/Challenger correct: {mcnemar['c']}"
    )
    if mcnemar["p_value"] < 0.05:
        better = "Challenger" if mcnemar["c"] > mcnemar["b"] else "Baseline"
        logger.info(f"  => SIGNIFICANT at p<0.05: {better} is better")
    else:
        logger.info("  => NOT significant at p<0.05")

    # Bootstrap Brier score
    brier_boot = bootstrap_metric_comparison(
        y_true, base_probs, chal_probs,
        brier_score_loss, n_bootstrap=n_bootstrap,
    )
    metrics["brier_bootstrap"] = brier_boot
    logger.info(
        f"\nBrier score bootstrap (challenger - baseline):"
    )
    logger.info(
        f"  Diff: {brier_boot['diff']:+.6f}  "
        f"95% CI: [{brier_boot['ci_lo']:+.6f}, {brier_boot['ci_hi']:+.6f}]"
    )
    if brier_boot["ci_hi"] < 0:
        logger.info("  => SIGNIFICANT: Challenger has lower (better) Brier score")
    elif brier_boot["ci_lo"] > 0:
        logger.info("  => SIGNIFICANT: Baseline has lower (better) Brier score")
    else:
        logger.info("  => NOT significant: CI includes zero")

    # Bootstrap log loss
    logloss_boot = bootstrap_metric_comparison(
        y_true, base_probs, chal_probs,
        log_loss, n_bootstrap=n_bootstrap,
    )
    metrics["logloss_bootstrap"] = logloss_boot
    logger.info(
        f"\nLog loss bootstrap (challenger - baseline):"
    )
    logger.info(
        f"  Diff: {logloss_boot['diff']:+.6f}  "
        f"95% CI: [{logloss_boot['ci_lo']:+.6f}, {logloss_boot['ci_hi']:+.6f}]"
    )

    # Summary table
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(
        f"{'Metric':<20s} {'Baseline':>12s} {'Challenger':>12s} {'Diff':>12s} {'Significant?':>14s}"
    )
    logger.info("-" * 70)

    for metric_name in ["accuracy", "brier_score", "log_loss", "ece"]:
        base_val = metrics["baseline"][metric_name]
        chal_val = metrics["challenger"][metric_name]
        diff = chal_val - base_val

        if metric_name == "accuracy":
            sig = "YES*" if mcnemar["p_value"] < 0.05 else "no"
        elif metric_name == "brier_score":
            sig = "YES*" if (brier_boot["ci_hi"] < 0 or brier_boot["ci_lo"] > 0) else "no"
        elif metric_name == "log_loss":
            sig = "YES*" if (logloss_boot["ci_hi"] < 0 or logloss_boot["ci_lo"] > 0) else "no"
        else:
            sig = "-"

        logger.info(
            f"{metric_name:<20s} {base_val:>12.4f} {chal_val:>12.4f} {diff:>+12.4f} {sig:>14s}"
        )

    logger.info(f"\nTotal test fights: {len(y_true)}")
    logger.info(f"Walk-forward folds: {len(wf['fold_results'])}")

    # Per-fold breakdown
    logger.info("\nPer-fold breakdown:")
    fold_df = wf["fold_results"]
    logger.info(fold_df.to_string(index=False))

    # Verdict
    logger.info("\n" + "=" * 70)
    brier_better = metrics["challenger"]["brier_score"] < metrics["baseline"]["brier_score"]
    acc_better = metrics["challenger"]["accuracy"] > metrics["baseline"]["accuracy"]
    brier_sig = brier_boot["ci_hi"] < 0

    if brier_better and brier_sig:
        logger.info("VERDICT: Challenger is SIGNIFICANTLY better (lower Brier score)")
    elif brier_better:
        logger.info("VERDICT: Challenger is better but NOT statistically significant")
    elif not brier_better and brier_boot["ci_lo"] > 0:
        logger.info("VERDICT: Baseline is SIGNIFICANTLY better — keep production model")
    else:
        logger.info("VERDICT: No clear winner — models are statistically equivalent")
    logger.info("=" * 70)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = COMPARISON_DIR / f"comparison_{timestamp}.json"
    # Convert numpy types for JSON serialization
    save_metrics = json.loads(json.dumps(metrics, default=str))
    with open(results_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    logger.info(f"\nSaved results to {results_path}")

    fold_path = COMPARISON_DIR / f"fold_results_{timestamp}.csv"
    fold_df.to_csv(fold_path, index=False)
    logger.info(f"Saved fold results to {fold_path}")

    return metrics


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(COMPARISON_DIR / "compare.log"),
        ],
    )

    parser = argparse.ArgumentParser(description="Compare baseline vs experimental model")
    parser.add_argument("--use-optuna", action="store_true", help="Use Optuna-optimized hyperparameters")
    parser.add_argument("--use-shap", action="store_true", help="Use SHAP-based feature selection")
    parser.add_argument("--top-features", type=int, default=40, help="Number of features to keep (with --use-shap)")
    parser.add_argument("--use-experimental-features", action="store_true", help="Add experimental features")
    parser.add_argument("--retrain-months", type=int, default=4, help="Retrain interval in months")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap iterations")
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

    # Compute baseline features BEFORE adding experimental ones
    # so the baseline stays true to production
    baseline_feature_cols = get_feature_columns(features_df)

    # Add experimental features if requested (only affects challenger)
    if args.use_experimental_features:
        from src.features.experimental_features import (
            add_experimental_features,
            get_experimental_feature_columns,
        )
        features_df = add_experimental_features(features_df)
        challenger_feature_cols = get_experimental_feature_columns(features_df)
    else:
        challenger_feature_cols = get_feature_columns(features_df)

    if args.use_shap:
        from src.model.feature_selection import select_top_features
        # Filter to 2022+ training data for SHAP
        train_df = features_df[
            (features_df["event_date"] >= pd.Timestamp(DATA_START_DATE))
            & (features_df["event_date"] < pd.Timestamp(TRAIN_TEST_SPLIT_DATE))
        ]
        if "a_num_fights" in train_df.columns and "b_num_fights" in train_df.columns:
            train_df = train_df[
                (train_df["a_num_fights"] >= 2) & (train_df["b_num_fights"] >= 2)
            ]
        train_df = train_df.dropna(subset=["target"])
        challenger_feature_cols = select_top_features(
            train_df, challenger_feature_cols, n_keep=args.top_features, method="shap"
        )

    # Get Optuna params if requested
    challenger_xgb_params = None
    if args.use_optuna:
        from src.model.hyperparam_search import load_best_params
        challenger_xgb_params = load_best_params()
        if not challenger_xgb_params:
            logger.warning("No Optuna results found. Run hyperparam_search.py first.")
            logger.warning("Continuing with default hyperparameters...")

    # Run comparison
    run_comparison(
        features_df,
        baseline_feature_cols=baseline_feature_cols,
        challenger_feature_cols=challenger_feature_cols,
        challenger_xgb_params=challenger_xgb_params,
        retrain_months=args.retrain_months,
        n_bootstrap=args.bootstrap,
    )


if __name__ == "__main__":
    main()
