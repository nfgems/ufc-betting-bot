"""
Compare old models (pre-fix) vs new models (with odds noise, time decay).
Evaluates accuracy, log loss, calibration, and backtest P&L.
"""

import sys
import logging
import joblib
import numpy as np
import pandas as pd

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/comparison.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def evaluate_on_test(model_result: dict, test_df: pd.DataFrame, label: str) -> dict:
    """Evaluate a model on a test set and return metrics."""
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

    feature_cols = model_result["feature_cols"]
    col_medians = model_result["col_medians"]
    model = model_result["model"]

    X_test = test_df[feature_cols].values.copy()

    # Fill NaNs with training medians
    for i in range(X_test.shape[1]):
        mask = np.isnan(X_test[:, i])
        X_test[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

    y_true = test_df["target"].values
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    logloss = log_loss(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    baseline_acc = max(y_true.mean(), 1 - y_true.mean())

    # Calibration error (ECE)
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    ece = np.mean(np.abs(prob_true - prob_pred))

    metrics = {
        "label": label,
        "accuracy": accuracy,
        "accuracy_vs_baseline": accuracy - baseline_acc,
        "log_loss": logloss,
        "brier_score": brier,
        "ece": ece,
        "n_test": len(y_true),
    }

    logger.info(f"\n{'='*55}")
    logger.info(f"  {label}")
    logger.info(f"{'='*55}")
    logger.info(f"  Test fights:          {len(y_true)}")
    logger.info(f"  Accuracy:             {accuracy:.4f}  (baseline: {baseline_acc:.4f})")
    logger.info(f"  Accuracy vs baseline: {accuracy - baseline_acc:+.4f}")
    logger.info(f"  Log loss:             {logloss:.4f}")
    logger.info(f"  Brier score:          {brier:.4f}")
    logger.info(f"  ECE (calibration):    {ece:.4f}")
    logger.info(f"{'='*55}")

    return metrics


def run_backtest_with_model(model_result: dict, test_df: pd.DataFrame, label: str) -> dict:
    """Run backtest using a specific model and return stats."""
    from src.strategy.backtest import run_backtest

    result = run_backtest(
        test_df=test_df,
        model_result=model_result,
    )

    stats = result["stats"]
    odds_source = result["odds_source"]

    logger.info(f"\n{'='*55}")
    logger.info(f"  BACKTEST: {label}")
    logger.info(f"{'='*55}")
    logger.info(f"  Total bets:     {stats.get('total_bets', 0)}")
    logger.info(f"  Win rate:       {stats.get('win_rate', 0):.1%}")
    logger.info(f"  Total wagered:  ${stats.get('total_wagered', 0):.2f}")
    logger.info(f"  Total profit:   ${stats.get('total_profit', 0):+.2f}")
    logger.info(f"  ROI:            {stats.get('roi', 0):.1%}")
    logger.info(f"  Bankroll:       ${stats.get('bankroll', 0):.2f}")
    logger.info(f"  Max drawdown:   {stats.get('max_drawdown', 0):.1%}")
    logger.info(f"  Avg edge:       {stats.get('avg_edge', 0):.1%}")
    if stats.get("avg_clv"):
        logger.info(f"  Avg CLV:        {stats.get('avg_clv', 0):.1%}")
    logger.info(f"  Odds source:    {odds_source}")
    logger.info(f"{'='*55}")

    return {
        "label": label,
        **{k: v for k, v in stats.items() if not isinstance(v, (list, dict, pd.DataFrame))},
    }


def main():
    from src.data.kaggle_loader import load_kaggle_dataset, save_processed
    from src.features.build_features import build_features, save_features
    from src.model.train import prepare_train_test, train_all_models

    # ---------------------------------------------------------------
    # 1. Load & build features (same data for fair comparison)
    # ---------------------------------------------------------------
    logger.info("Loading data and building features...")
    fights_df = load_kaggle_dataset()
    save_processed(fights_df)
    features_df = build_features(fights_df)
    save_features(features_df)

    # Prepare train/test split
    train_df, test_df, feature_cols = prepare_train_test(features_df)
    logger.info(f"Train: {len(train_df)} fights, Test: {len(test_df)} fights")

    # ---------------------------------------------------------------
    # 2. Load OLD models
    # ---------------------------------------------------------------
    logger.info("\nLoading old (pre-fix) models...")
    old_xgb = joblib.load("models/old/xgboost_model.pkl")
    old_lr = joblib.load("models/old/logistic_model.pkl")
    old_no_odds = joblib.load("models/old/xgboost_no_odds_model.pkl")

    # ---------------------------------------------------------------
    # 3. Train NEW models (with odds noise + time decay)
    # ---------------------------------------------------------------
    logger.info("\nTraining new models (with odds noise + time decay)...")
    results = train_all_models(features_df)

    new_xgb = results["xgboost"]
    new_lr = results["logistic"]
    new_no_odds = results["xgboost_no_odds"]

    # ---------------------------------------------------------------
    # 4. Evaluate on test set — old vs new
    # ---------------------------------------------------------------
    logger.info("\n" + "#" * 60)
    logger.info("#  MODEL EVALUATION COMPARISON")
    logger.info("#" * 60)

    metrics = []
    metrics.append(evaluate_on_test(old_xgb, test_df, "OLD XGBoost"))
    metrics.append(evaluate_on_test(new_xgb, test_df, "NEW XGBoost (odds noise + time decay)"))
    metrics.append(evaluate_on_test(old_lr, test_df, "OLD Logistic"))
    metrics.append(evaluate_on_test(new_lr, test_df, "NEW Logistic (odds noise + time decay)"))
    metrics.append(evaluate_on_test(old_no_odds, test_df, "OLD No-Odds Baseline"))
    metrics.append(evaluate_on_test(new_no_odds, test_df, "NEW No-Odds Baseline (time decay)"))

    # ---------------------------------------------------------------
    # 5. Backtest — old vs new XGBoost
    # ---------------------------------------------------------------
    logger.info("\n" + "#" * 60)
    logger.info("#  BACKTEST COMPARISON")
    logger.info("#" * 60)

    bt_results = []
    bt_results.append(run_backtest_with_model(old_xgb, test_df, "OLD XGBoost"))
    bt_results.append(run_backtest_with_model(new_xgb, test_df, "NEW XGBoost"))

    # ---------------------------------------------------------------
    # 6. Summary table
    # ---------------------------------------------------------------
    logger.info("\n" + "#" * 60)
    logger.info("#  SUMMARY")
    logger.info("#" * 60)

    metrics_df = pd.DataFrame(metrics)
    cols = ["label", "accuracy", "accuracy_vs_baseline", "log_loss", "brier_score", "ece"]
    logger.info(f"\nEvaluation Metrics:\n{metrics_df[cols].to_string(index=False)}")

    bt_df = pd.DataFrame(bt_results)
    bt_cols = ["label", "total_bets", "win_rate", "total_wagered", "total_profit", "roi", "bankroll", "max_drawdown"]
    available_cols = [c for c in bt_cols if c in bt_df.columns]
    logger.info(f"\nBacktest Results:\n{bt_df[available_cols].to_string(index=False)}")

    # Save comparison
    metrics_df.to_csv("logs/old_vs_new_metrics.csv", index=False)
    bt_df.to_csv("logs/old_vs_new_backtest.csv", index=False)
    logger.info("\nResults saved to logs/old_vs_new_metrics.csv and logs/old_vs_new_backtest.csv")


if __name__ == "__main__":
    main()
