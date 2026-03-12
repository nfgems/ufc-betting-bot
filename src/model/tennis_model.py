"""Stage 1 tennis model: calibrated surface-aware Elo baseline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, TENNIS_MIN_MATCHES
from src.features.tennis_features import (
    STAGE1_TENNIS_MODEL_COLUMNS,
    filter_minimum_history,
    require_stage1_tennis_feature_columns,
)

logger = logging.getLogger(__name__)

TENNIS_MODELS_DIR = MODELS_DIR / "tennis"
TENNIS_MODELS_DIR.mkdir(parents=True, exist_ok=True)

STAGE1_CALIBRATION_METHOD = "sigmoid"
MIN_ROWS_FOR_STAGE1_CALIBRATION = 200
STAGE1_FEATURE_CONTRACT = "stage1_surface_elo_baseline"


def get_stage1_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Return the strict Stage 1 model column contract."""
    return require_stage1_tennis_feature_columns(features_df)


def _validate_stage1_feature_columns(feature_cols: list[str]) -> None:
    if list(feature_cols) != list(STAGE1_TENNIS_MODEL_COLUMNS):
        raise ValueError(
            "Invalid Stage 1 tennis feature contract. Expected "
            + ", ".join(STAGE1_TENNIS_MODEL_COLUMNS)
            + "; got "
            + ", ".join(feature_cols)
        )


def _validate_stage1_model_result(model_result: dict) -> None:
    feature_contract = model_result.get("feature_contract")
    if feature_contract != STAGE1_FEATURE_CONTRACT:
        raise ValueError(
            "Saved tennis model does not match the Stage 1 feature contract: "
            f"{feature_contract!r}"
        )

    feature_cols = list(model_result.get("feature_cols") or [])
    _validate_stage1_feature_columns(feature_cols)

    col_medians = np.asarray(model_result.get("col_medians", []))
    if len(col_medians) != len(STAGE1_TENNIS_MODEL_COLUMNS):
        raise ValueError(
            "Saved tennis model medians do not match the Stage 1 feature contract."
        )


def _prepare_training_frame(features_df: pd.DataFrame, min_matches: int = TENNIS_MIN_MATCHES) -> pd.DataFrame:
    feature_cols = get_stage1_feature_columns(features_df)
    frame = filter_minimum_history(features_df, min_matches=min_matches)
    frame = frame.dropna(subset=["event_date", "target"]).sort_values("event_date").reset_index(drop=True)
    if not feature_cols:
        raise ValueError("No Stage 1 tennis features available for training")
    frame = frame.dropna(subset=feature_cols, how="all").reset_index(drop=True)
    return frame


def _build_estimator() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def _build_calibration_splitter(y_train: np.ndarray) -> Optional[TimeSeriesSplit]:
    """Return the largest valid temporal splitter for Platt scaling."""
    if len(y_train) < MIN_ROWS_FOR_STAGE1_CALIBRATION:
        return None

    max_splits = min(5, max(2, len(y_train) // 80))
    if len(y_train) <= max_splits:
        return None

    for n_splits in range(max_splits, 1, -1):
        splitter = TimeSeriesSplit(n_splits=n_splits)
        valid = True
        for train_idx, test_idx in splitter.split(np.arange(len(y_train))):
            if len(np.unique(y_train[train_idx])) < 2 or len(np.unique(y_train[test_idx])) < 2:
                valid = False
                break
        if valid:
            return splitter
    return None


def _fit_model(train_df: pd.DataFrame, feature_cols: list[str], calibrate: bool = True) -> dict:
    _validate_stage1_feature_columns(feature_cols)
    X_train = train_df[feature_cols].values.copy()
    y_train = train_df["target"].astype(int).values
    col_medians = np.nanmedian(X_train, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)

    for idx in range(X_train.shape[1]):
        mask = np.isnan(X_train[:, idx])
        if mask.any():
            X_train[mask, idx] = col_medians[idx]

    base_estimator = _build_estimator()
    model = base_estimator

    calibration_splitter = _build_calibration_splitter(y_train) if calibrate else None
    if calibration_splitter is not None:
        model = CalibratedClassifierCV(
            estimator=base_estimator,
            cv=calibration_splitter,
            method=STAGE1_CALIBRATION_METHOD,
        )
    elif calibrate and len(train_df) < MIN_ROWS_FOR_STAGE1_CALIBRATION:
        logger.info(
            "Skipping Stage 1 tennis calibration because training set has %s rows; need at least %s.",
            len(train_df),
            MIN_ROWS_FOR_STAGE1_CALIBRATION,
        )
    elif calibrate and len(train_df) >= MIN_ROWS_FOR_STAGE1_CALIBRATION:
        logger.info(
            "Skipping Stage 1 tennis calibration because temporal folds are too small or single-class."
        )

    model.fit(X_train, y_train)
    return {
        "model": model,
        "feature_cols": feature_cols,
        "col_medians": col_medians,
        "feature_contract": STAGE1_FEATURE_CONTRACT,
        "calibration_method": getattr(model, "method", "none"),
    }


def predict_tennis_batch(features_df: pd.DataFrame, model_result: Optional[dict] = None) -> pd.DataFrame:
    """Predict tennis match win probabilities for a feature matrix."""
    if model_result is None:
        model_result = load_tennis_model()

    _validate_stage1_model_result(model_result)
    feature_cols = list(model_result["feature_cols"])
    missing = [column for column in feature_cols if column not in features_df.columns]
    if missing:
        raise ValueError(
            "Missing required Stage 1 tennis feature columns for prediction: "
            + ", ".join(missing)
        )
    X = features_df[feature_cols].values.copy()
    col_medians = np.asarray(model_result["col_medians"])

    for idx in range(X.shape[1]):
        mask = np.isnan(X[:, idx])
        if mask.any():
            X[mask, idx] = col_medians[idx] if idx < len(col_medians) else 0.0

    proba = model_result["model"].predict_proba(X)
    predictions = features_df.copy()
    predictions["prob_a"] = proba[:, 1]
    predictions["prob_b"] = proba[:, 0]
    predictions["predicted_winner"] = np.where(proba[:, 1] >= 0.5, "a", "b")
    predictions["confidence"] = np.maximum(proba[:, 1], proba[:, 0])
    return predictions


def predict_tennis_match(features: dict[str, object], model_result: Optional[dict] = None) -> dict[str, float | str]:
    """Predict one tennis matchup from a feature dict."""
    frame = pd.DataFrame([features])
    predictions = predict_tennis_batch(frame, model_result=model_result)
    row = predictions.iloc[0]
    return {
        "prob_a": float(row["prob_a"]),
        "prob_b": float(row["prob_b"]),
        "predicted_winner": str(row["predicted_winner"]),
        "confidence": float(row["confidence"]),
    }


def calibration_table(y_true: pd.Series, probabilities: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Summarize calibration by prediction bin."""
    calibration = pd.DataFrame({"target": y_true.astype(int), "probability": probabilities.astype(float)})
    calibration["bin"] = pd.cut(
        calibration["probability"],
        bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True,
    )
    return (
        calibration.groupby("bin", observed=False)
        .agg(
            count=("target", "size"),
            avg_pred=("probability", "mean"),
            actual_win_rate=("target", "mean"),
        )
        .reset_index()
    )


def evaluate_prediction_frame(predictions: pd.DataFrame) -> dict[str, object]:
    """Compute Stage 1 predictive metrics."""
    if predictions.empty:
        raise ValueError("No predictions available for evaluation")

    y_true = predictions["target"].astype(int)
    probabilities = predictions["prob_a"].astype(float)
    return {
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "calibration": calibration_table(y_true, probabilities),
    }


def run_walkforward_evaluation(
    features_df: pd.DataFrame,
    initial_train_years: int = 3,
    retrain_months: int = 6,
    min_matches: int = TENNIS_MIN_MATCHES,
) -> dict[str, object]:
    """Run walk-forward predictive evaluation for the Stage 1 baseline."""
    frame = _prepare_training_frame(features_df, min_matches=min_matches)
    if frame.empty:
        raise ValueError("No tennis rows available after minimum-history filtering")

    start_date = frame["event_date"].min() + pd.DateOffset(years=initial_train_years)
    max_date = frame["event_date"].max()

    fold_predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    fold_index = 0

    while start_date <= max_date:
        end_date = start_date + pd.DateOffset(months=retrain_months)
        train_df = frame[frame["event_date"] < start_date].copy()
        test_df = frame[(frame["event_date"] >= start_date) & (frame["event_date"] < end_date)].copy()

        if train_df.empty or test_df.empty:
            start_date = end_date
            continue

        feature_cols = get_stage1_feature_columns(train_df)
        model_result = _fit_model(train_df, feature_cols=feature_cols, calibrate=True)
        fold_pred = predict_tennis_batch(test_df, model_result=model_result)
        metrics = evaluate_prediction_frame(fold_pred)
        fold_pred["fold"] = fold_index
        fold_predictions.append(fold_pred)
        fold_rows.append(
            {
                "fold": fold_index,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "start_date": pd.Timestamp(start_date),
                "end_date": pd.Timestamp(end_date),
                "log_loss": metrics["log_loss"],
                "brier_score": metrics["brier_score"],
            }
        )

        fold_index += 1
        start_date = end_date

    combined_predictions = pd.concat(fold_predictions, ignore_index=True) if fold_predictions else pd.DataFrame()
    overall_metrics = evaluate_prediction_frame(combined_predictions) if not combined_predictions.empty else {}
    return {
        "folds": pd.DataFrame(fold_rows),
        "predictions": combined_predictions,
        "metrics": overall_metrics,
    }


def save_tennis_model(model_result: dict, model_name: str = "surface_elo") -> Path:
    """Persist a tennis model under models/tennis."""
    output_path = TENNIS_MODELS_DIR / f"{model_name}.pkl"
    persisted = dict(model_result)
    persisted.pop("evaluation_predictions", None)
    joblib.dump(persisted, output_path)
    logger.info("Saved tennis model to %s", output_path)
    return output_path


def load_tennis_model(model_name: str = "surface_elo") -> dict:
    """Load a tennis model from models/tennis."""
    path = TENNIS_MODELS_DIR / f"{model_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Tennis model not found: {path}")
    model_result = joblib.load(path)
    _validate_stage1_model_result(model_result)
    return model_result


def train_tennis_model(
    features_df: pd.DataFrame,
    model_name: str = "surface_elo",
    min_matches: int = TENNIS_MIN_MATCHES,
) -> dict:
    """Evaluate the Stage 1 baseline walk-forward, then fit a final saved model."""
    training_frame = _prepare_training_frame(features_df, min_matches=min_matches)
    evaluation = run_walkforward_evaluation(
        features_df=training_frame,
        initial_train_years=3,
        retrain_months=6,
        min_matches=min_matches,
    )
    feature_cols = get_stage1_feature_columns(training_frame)
    final_model = _fit_model(training_frame, feature_cols=feature_cols, calibrate=True)
    result = {
        **final_model,
        "model_name": model_name,
        "training_rows": len(training_frame),
        "evaluation_folds": evaluation["folds"],
        "evaluation_metrics": evaluation["metrics"],
        "evaluation_predictions": evaluation["predictions"],
    }
    save_tennis_model(result, model_name=model_name)
    return result
