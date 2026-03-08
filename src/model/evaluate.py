"""
Model evaluation — accuracy, log-loss, calibration, and comparison to baselines.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

from src.model.predict import predict_batch
from src.model.train import load_model
from src.config import MODELS_DIR, LOGS_DIR

logger = logging.getLogger(__name__)


def evaluate_model(
    test_df: pd.DataFrame,
    model_name: str = "xgboost",
    model_result: Optional[dict] = None,
    save_plots: bool = True,
) -> dict:
    """
    Comprehensive model evaluation on test set.

    Returns dict with all metrics.
    """
    predictions = predict_batch(test_df, model_name=model_name, model_result=model_result)

    y_true = predictions["target"].values
    y_pred = (predictions["prob_a"] > 0.5).astype(int).values
    y_prob = predictions["prob_a"].values

    # Core metrics
    accuracy = accuracy_score(y_true, y_pred)
    logloss = log_loss(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # Baseline: always pick fighter A (or always pick favorite if we have odds)
    baseline_acc = max(y_true.mean(), 1 - y_true.mean())

    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)

    metrics = {
        "accuracy": accuracy,
        "log_loss": logloss,
        "brier_score": brier,
        "baseline_accuracy": baseline_acc,
        "accuracy_vs_baseline": accuracy - baseline_acc,
        "n_test_fights": len(y_true),
        "classification_report": report,
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Test fights: {len(y_true)}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Baseline (majority class): {baseline_acc:.4f}")
    logger.info(f"Accuracy vs baseline: {accuracy - baseline_acc:+.4f}")
    logger.info(f"Log loss: {logloss:.4f}")
    logger.info(f"Brier score: {brier:.4f}")
    logger.info(f"{'='*50}")

    if save_plots:
        _save_evaluation_plots(y_true, y_prob, y_pred, model_name, predictions)

    return metrics


def _save_evaluation_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    predictions: pd.DataFrame,
) -> None:
    """Generate and save evaluation plots."""
    plots_dir = LOGS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Calibration curve
    ax = axes[0, 0]
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    ax.plot(prob_pred, prob_true, marker="o", label=model_name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Actual win rate")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Probability distribution
    ax = axes[0, 1]
    ax.hist(y_prob[y_true == 1], bins=20, alpha=0.6, label="Actual wins (A)", color="green")
    ax.hist(y_prob[y_true == 0], bins=20, alpha=0.6, label="Actual losses (A)", color="red")
    ax.set_xlabel("Predicted P(Fighter A wins)")
    ax.set_ylabel("Count")
    ax.set_title("Probability Distribution by Outcome")
    ax.legend()

    # 3. Confusion matrix
    ax = axes[1, 0]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["B wins", "A wins"],
                yticklabels=["B wins", "A wins"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    # 4. Accuracy by confidence bucket
    ax = axes[1, 1]
    confidence = np.maximum(y_prob, 1 - y_prob)
    correct = (y_pred == y_true).astype(int)
    conf_df = pd.DataFrame({"confidence": confidence, "correct": correct})
    conf_df["conf_bucket"] = pd.cut(conf_df["confidence"], bins=5)
    bucket_acc = conf_df.groupby("conf_bucket", observed=True)["correct"].agg(["mean", "count"])
    bucket_acc["mean"].plot(kind="bar", ax=ax, color="steelblue")
    ax.set_xlabel("Confidence Bucket")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Confidence Level")
    ax.tick_params(axis="x", rotation=45)
    for i, (_, row) in enumerate(bucket_acc.iterrows()):
        ax.text(i, row["mean"] + 0.01, f'n={int(row["count"])}', ha="center", fontsize=8)

    plt.suptitle(f"Model Evaluation: {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = plots_dir / f"{model_name}_evaluation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved evaluation plots to {path}")


def compare_models(
    test_df: pd.DataFrame,
    model_names: list[str] = None,
) -> pd.DataFrame:
    """
    Compare multiple models side by side on the same test set.

    Returns a DataFrame of model metrics.
    """
    if model_names is None:
        model_names = ["xgboost", "logistic"]

    results = []
    for name in model_names:
        try:
            metrics = evaluate_model(test_df, model_name=name, save_plots=True)
            results.append({"model": name, **{k: v for k, v in metrics.items()
                                               if k != "classification_report"}})
        except Exception as e:
            logger.warning(f"Failed to evaluate {name}: {e}")

    comparison = pd.DataFrame(results)
    if not comparison.empty:
        logger.info(f"\nModel Comparison:\n{comparison.to_string(index=False)}")
        path = LOGS_DIR / "model_comparison.csv"
        comparison.to_csv(path, index=False)

    return comparison


def print_feature_importance(model_name: str = "xgboost", top_n: int = 20) -> None:
    """Print and plot top feature importances."""
    model_result = load_model(model_name)
    importance = model_result.get("feature_importance", {})

    if not importance:
        logger.warning("No feature importance data found")
        return

    top_features = dict(list(importance.items())[:top_n])

    logger.info(f"\nTop {top_n} features ({model_name}):")
    for feat, imp in top_features.items():
        bar = "#" * int(imp * 200)
        logger.info(f"  {feat:40s} {imp:.4f} {bar}")

    # Plot
    plots_dir = LOGS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    features = list(top_features.keys())[::-1]
    values = list(top_features.values())[::-1]
    ax.barh(features, values, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances ({model_name})")
    plt.tight_layout()
    path = plots_dir / f"{model_name}_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved feature importance plot to {path}")
