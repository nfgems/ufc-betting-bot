"""Tennis models: Stage 1 Elo baseline and Stage 2 lean hybrid."""

from __future__ import annotations

import json
import logging
import os
import shutil
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

from src.config import (
    DEFAULT_TENNIS_MODEL_NAME,
    MODELS_DIR,
    TENNIS_MIN_MATCHES,
    TENNIS_OOS_START_DATE,
    TENNIS_OOS_TEST_WINDOW_MONTHS,
    TENNIS_TRAINING_START_DATE,
)
from src.features.tennis_features import (
    STAGE1_TENNIS_MODEL_COLUMNS,
    STAGE2_TENNIS_MODEL_COLUMNS,
    filter_minimum_history,
    require_stage1_tennis_feature_columns,
    require_stage2_tennis_feature_columns,
)

logger = logging.getLogger(__name__)

TENNIS_MODELS_DIR = MODELS_DIR / "tennis"
TENNIS_MODELS_DIR.mkdir(parents=True, exist_ok=True)

STAGE1_CALIBRATION_METHOD = "sigmoid"
MIN_ROWS_FOR_STAGE1_CALIBRATION = 200
STAGE1_FEATURE_CONTRACT = "stage1_surface_elo_baseline"
STAGE2_FEATURE_CONTRACT = "stage2_lean_hybrid"
TENNIS_OOS_EVALUATION_CONTRACT = "oos_2022plus_2023start_expanding"
TENNIS_LOCKBOX_EVALUATION_CONTRACT = "lockbox_eval_v1"
TENNIS_OOS_ARTIFACT_PREFIX = "oos_2022plus_2023start"
STAGE2_OOS_ARTIFACT_PREFIX = f"{TENNIS_OOS_ARTIFACT_PREFIX}_lean_hybrid"

TENNIS_FEATURE_CONTRACT_COLUMNS = {
    STAGE1_FEATURE_CONTRACT: list(STAGE1_TENNIS_MODEL_COLUMNS),
    STAGE2_FEATURE_CONTRACT: list(STAGE2_TENNIS_MODEL_COLUMNS),
}
TENNIS_FEATURE_CONTRACT_LABELS = {
    STAGE1_FEATURE_CONTRACT: "Stage 1",
    STAGE2_FEATURE_CONTRACT: "Stage 2",
}
TENNIS_OOS_ARTIFACT_PREFIXES = {
    STAGE1_FEATURE_CONTRACT: TENNIS_OOS_ARTIFACT_PREFIX,
    STAGE2_FEATURE_CONTRACT: STAGE2_OOS_ARTIFACT_PREFIX,
}
TENNIS_MODEL_PORTABILITY_DROP_KEYS = {
    "evaluation_predictions",
    "evaluation_folds",
    "evaluation_metrics",
    "evaluation_diagnostics",
    "evaluation_summary",
}
TENNIS_MODEL_NAME_CONTRACTS = {
    "surface_elo": STAGE1_FEATURE_CONTRACT,
    "stage1_surface_elo": STAGE1_FEATURE_CONTRACT,
    "stage1": STAGE1_FEATURE_CONTRACT,
    "lean_hybrid": STAGE2_FEATURE_CONTRACT,
    "hybrid": STAGE2_FEATURE_CONTRACT,
    "stage2": STAGE2_FEATURE_CONTRACT,
    "stage2_lean_hybrid": STAGE2_FEATURE_CONTRACT,
}
TENNIS_DIAGNOSTIC_BUCKETS = {
    "history_bucket": (
        [-np.inf, 5, 10, 20, np.inf],
        ["3-5", "6-10", "11-20", "21+"],
    ),
    "confidence_bucket": (
        [0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 1.0],
        ["0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70-0.75", "0.75+"],
    ),
    "rank_gap_bucket": (
        [-np.inf, 10, 25, 50, 100, np.inf],
        ["0-10", "11-25", "26-50", "51-100", "100+"],
    ),
}


def _persistent_tennis_models_dir() -> Path | None:
    """Use the hosted persistent data volume when the runtime config exposes it."""
    configured_data_dir = str(os.environ.get("UFC_DATA_DIR", "") or "").strip()
    if not configured_data_dir:
        return None
    return Path(configured_data_dir) / "models" / "tennis"


def _tennis_model_paths(model_name: str) -> list[Path]:
    filename = f"{model_name}.pkl"
    paths = [TENNIS_MODELS_DIR / filename]
    persistent_dir = _persistent_tennis_models_dir()
    if persistent_dir is not None:
        persistent_path = persistent_dir / filename
        if persistent_path not in paths:
            paths.append(persistent_path)
    return paths


def _normalize_model_name(model_name: object) -> str:
    return str(model_name or "").strip().lower()


def infer_tennis_feature_contract(
    model_name: str = DEFAULT_TENNIS_MODEL_NAME,
    *,
    feature_contract: Optional[str] = None,
) -> str:
    """Resolve the saved-model feature contract from explicit input or model name."""
    if feature_contract is not None:
        contract = str(feature_contract).strip()
        if contract not in TENNIS_FEATURE_CONTRACT_COLUMNS:
            raise ValueError(f"Unsupported tennis feature contract: {contract}")
        return contract

    normalized_name = _normalize_model_name(model_name)
    if normalized_name in TENNIS_MODEL_NAME_CONTRACTS:
        return TENNIS_MODEL_NAME_CONTRACTS[normalized_name]
    if "hybrid" in normalized_name:
        return STAGE2_FEATURE_CONTRACT
    return STAGE1_FEATURE_CONTRACT


def _feature_contract_columns(feature_contract: str) -> list[str]:
    try:
        return list(TENNIS_FEATURE_CONTRACT_COLUMNS[feature_contract])
    except KeyError as exc:
        raise ValueError(f"Unsupported tennis feature contract: {feature_contract}") from exc


def _artifact_prefix_for_contract(feature_contract: str) -> str:
    return TENNIS_OOS_ARTIFACT_PREFIXES.get(feature_contract, TENNIS_OOS_ARTIFACT_PREFIX)


def get_stage1_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Return the strict Stage 1 model column contract."""
    return require_stage1_tennis_feature_columns(features_df)


def get_stage2_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Return the strict Stage 2 model column contract."""
    return require_stage2_tennis_feature_columns(features_df)


def _validate_feature_columns(feature_cols: list[str], feature_contract: str) -> None:
    expected = _feature_contract_columns(feature_contract)
    if list(feature_cols) != expected:
        raise ValueError(
            f"Invalid {TENNIS_FEATURE_CONTRACT_LABELS.get(feature_contract, 'tennis')} tennis feature contract. Expected "
            + ", ".join(expected)
            + "; got "
            + ", ".join(feature_cols)
        )


def _validate_model_result(model_result: dict) -> None:
    feature_contract = str(model_result.get("feature_contract") or "").strip()
    if feature_contract not in TENNIS_FEATURE_CONTRACT_COLUMNS:
        raise ValueError(
            "Saved tennis model does not match a supported feature contract: "
            f"{feature_contract!r}"
        )

    feature_cols = list(model_result.get("feature_cols") or [])
    _validate_feature_columns(feature_cols, feature_contract)

    col_medians = np.asarray(model_result.get("col_medians", []))
    if len(col_medians) != len(feature_cols):
        raise ValueError(
            "Saved tennis model medians do not match the saved tennis feature contract."
        )


def _coerce_first_available(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in columns:
        if column in frame.columns:
            candidate = pd.to_numeric(frame[column], errors="coerce")
            values = values.where(values.notna(), candidate)
    return values


def _prepare_stage2_feature_frame(features_df: pd.DataFrame) -> pd.DataFrame:
    prepared = features_df.copy()

    if "log_rank_diff" not in prepared.columns:
        a_rank = _coerce_first_available(prepared, ["a_rank", "player_a_rank"])
        b_rank = _coerce_first_available(prepared, ["b_rank", "player_b_rank"])
        prepared["log_rank_diff"] = np.where(
            a_rank.notna() & b_rank.notna(),
            np.log1p(b_rank) - np.log1p(a_rank),
            np.nan,
        )

    if "log_rank_points_diff" not in prepared.columns:
        a_rank_points = _coerce_first_available(prepared, ["a_rank_points", "player_a_rank_points"])
        b_rank_points = _coerce_first_available(prepared, ["b_rank_points", "player_b_rank_points"])
        prepared["log_rank_points_diff"] = np.where(
            a_rank_points.notna() & b_rank_points.notna(),
            np.log1p(a_rank_points) - np.log1p(b_rank_points),
            np.nan,
        )

    if "diff_age" not in prepared.columns:
        a_age = _coerce_first_available(prepared, ["a_age", "player_a_age"])
        b_age = _coerce_first_available(prepared, ["b_age", "player_b_age"])
        prepared["diff_age"] = a_age - b_age

    return prepared


def prepare_tennis_model_features(
    features_df: pd.DataFrame,
    *,
    feature_contract: str,
) -> pd.DataFrame:
    """Materialize the saved-model feature contract on top of a tennis feature frame."""
    if feature_contract == STAGE2_FEATURE_CONTRACT:
        return _prepare_stage2_feature_frame(features_df)
    return features_df.copy()


def require_tennis_feature_columns(features_df: pd.DataFrame, feature_contract: str) -> list[str]:
    """Return the exact required column contract for a tennis model."""
    prepared = prepare_tennis_model_features(features_df, feature_contract=feature_contract)
    if feature_contract == STAGE1_FEATURE_CONTRACT:
        return require_stage1_tennis_feature_columns(prepared)
    if feature_contract == STAGE2_FEATURE_CONTRACT:
        return require_stage2_tennis_feature_columns(prepared)
    raise ValueError(f"Unsupported tennis feature contract: {feature_contract}")


def filter_tennis_training_window(
    frame: pd.DataFrame,
    start_date: str = TENNIS_TRAINING_START_DATE,
) -> pd.DataFrame:
    """Return only tennis rows inside the 2022+ training universe."""
    if "event_date" not in frame.columns:
        raise ValueError("Tennis training data must include an event_date column")

    filtered = frame.copy()
    filtered["event_date"] = pd.to_datetime(filtered["event_date"], errors="coerce")
    filtered = filtered.dropna(subset=["event_date"]).copy()
    filtered = filtered[filtered["event_date"] >= pd.Timestamp(start_date)].copy()
    return filtered.sort_values("event_date").reset_index(drop=True)


def _prepare_training_frame(
    features_df: pd.DataFrame,
    *,
    min_matches: int = TENNIS_MIN_MATCHES,
    feature_contract: str = STAGE1_FEATURE_CONTRACT,
) -> pd.DataFrame:
    frame = filter_tennis_training_window(features_df)
    frame = filter_minimum_history(frame, min_matches=min_matches)
    frame = frame.dropna(subset=["event_date", "target"]).sort_values("event_date").reset_index(drop=True)
    frame = prepare_tennis_model_features(frame, feature_contract=feature_contract)
    feature_cols = require_tennis_feature_columns(frame, feature_contract)
    if not feature_cols:
        raise ValueError("No tennis features available for training")
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


def _fit_model(
    train_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    feature_contract: str,
    calibrate: bool = True,
) -> dict:
    _validate_feature_columns(feature_cols, feature_contract)
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
            "Skipping tennis calibration because training set has %s rows; need at least %s.",
            len(train_df),
            MIN_ROWS_FOR_STAGE1_CALIBRATION,
        )
    elif calibrate and len(train_df) >= MIN_ROWS_FOR_STAGE1_CALIBRATION:
        logger.info(
            "Skipping tennis calibration because temporal folds are too small or single-class."
        )

    model.fit(X_train, y_train)
    return {
        "model": model,
        "feature_cols": feature_cols,
        "col_medians": col_medians,
        "feature_contract": feature_contract,
        "calibration_method": getattr(model, "method", "none"),
        "oos_artifact_prefix": _artifact_prefix_for_contract(feature_contract),
    }


def predict_tennis_batch(features_df: pd.DataFrame, model_result: Optional[dict] = None) -> pd.DataFrame:
    """Predict tennis match win probabilities for a feature matrix."""
    if model_result is None:
        model_result = load_tennis_model()

    _validate_model_result(model_result)
    feature_contract = str(model_result["feature_contract"])
    prepared = prepare_tennis_model_features(features_df, feature_contract=feature_contract)
    feature_cols = list(model_result["feature_cols"])
    missing = [column for column in feature_cols if column not in prepared.columns]
    if missing:
        raise ValueError(
            "Missing required tennis feature columns for prediction: "
            + ", ".join(missing)
        )
    X = prepared[feature_cols].values.copy()
    col_medians = np.asarray(model_result["col_medians"])

    for idx in range(X.shape[1]):
        mask = np.isnan(X[:, idx])
        if mask.any():
            X[mask, idx] = col_medians[idx] if idx < len(col_medians) else 0.0

    proba = model_result["model"].predict_proba(X)
    predictions = prepared.copy()
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
    grouped = (
        calibration.groupby("bin", observed=False)
        .agg(
            count=("target", "size"),
            avg_pred=("probability", "mean"),
            actual_win_rate=("target", "mean"),
        )
        .reset_index()
    )
    grouped["abs_gap"] = (grouped["avg_pred"] - grouped["actual_win_rate"]).abs()
    return grouped


def _expected_calibration_error(calibration: pd.DataFrame) -> float:
    if calibration.empty:
        return float("nan")
    total_count = calibration["count"].sum()
    if total_count <= 0:
        return float("nan")
    return float((calibration["count"] * calibration["abs_gap"]).sum() / total_count)


def _basic_prediction_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    y_true = predictions["target"].astype(int)
    probabilities = predictions["prob_a"].astype(float).clip(1e-6, 1 - 1e-6)
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "rows": int(len(predictions)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "accuracy": float((predicted == y_true).mean()),
        "avg_pred": float(probabilities.mean()),
        "actual_win_rate": float(y_true.mean()),
    }


def _segment_metrics(predictions: pd.DataFrame, segment_column: str) -> pd.DataFrame:
    if segment_column not in predictions.columns:
        return pd.DataFrame(
            columns=[
                segment_column,
                "rows",
                "log_loss",
                "brier_score",
                "accuracy",
                "avg_pred",
                "actual_win_rate",
            ]
        )

    rows: list[dict[str, object]] = []
    for segment_value, segment_frame in predictions.groupby(segment_column, dropna=False):
        metrics = _basic_prediction_metrics(segment_frame)
        rows.append(
            {
                segment_column: "missing" if pd.isna(segment_value) else str(segment_value),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(segment_column).reset_index(drop=True)


def _add_prediction_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    diagnostics = predictions.copy()

    if {"a_num_matches", "b_num_matches"}.issubset(diagnostics.columns):
        diagnostics["min_player_matches"] = diagnostics[["a_num_matches", "b_num_matches"]].min(axis=1)
        bins, labels = TENNIS_DIAGNOSTIC_BUCKETS["history_bucket"]
        diagnostics["history_bucket"] = pd.cut(
            diagnostics["min_player_matches"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )

    bins, labels = TENNIS_DIAGNOSTIC_BUCKETS["confidence_bucket"]
    diagnostics["confidence_bucket"] = pd.cut(
        diagnostics["confidence"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    rank_gap = pd.Series(np.nan, index=diagnostics.index, dtype=float)
    if "diff_rank" in diagnostics.columns:
        rank_gap = pd.to_numeric(diagnostics["diff_rank"], errors="coerce").abs()
    elif "log_rank_diff" in diagnostics.columns:
        rank_gap = pd.to_numeric(diagnostics["log_rank_diff"], errors="coerce").abs()
    diagnostics["rank_gap_abs"] = rank_gap
    rank_gap_bins, rank_gap_labels = TENNIS_DIAGNOSTIC_BUCKETS["rank_gap_bucket"]
    diagnostics["rank_gap_bucket"] = pd.cut(
        diagnostics["rank_gap_abs"],
        bins=rank_gap_bins,
        labels=rank_gap_labels,
        include_lowest=True,
    )

    return diagnostics


def build_tennis_evaluation_diagnostics(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build subgroup diagnostics for the anchored OOS prediction frame."""
    return {
        "tour_metrics": _segment_metrics(predictions, "tour"),
        "surface_metrics": _segment_metrics(predictions, "surface"),
        "history_bucket_metrics": _segment_metrics(predictions, "history_bucket"),
        "confidence_bucket_metrics": _segment_metrics(predictions, "confidence_bucket"),
        "rank_gap_bucket_metrics": _segment_metrics(predictions, "rank_gap_bucket"),
    }


def evaluate_prediction_frame(predictions: pd.DataFrame) -> dict[str, object]:
    """Compute predictive metrics for a tennis prediction frame."""
    if predictions.empty:
        raise ValueError("No predictions available for evaluation")

    y_true = predictions["target"].astype(int)
    probabilities = predictions["prob_a"].astype(float)
    calibration = calibration_table(y_true, probabilities)
    metrics = _basic_prediction_metrics(predictions)
    metrics["calibration"] = calibration
    metrics["ece_10_bin"] = _expected_calibration_error(calibration)
    return metrics


def _format_date(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _build_expanding_oos_windows(
    frame: pd.DataFrame,
    start_date: str = TENNIS_OOS_START_DATE,
    test_window_months: int = TENNIS_OOS_TEST_WINDOW_MONTHS,
) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]], pd.Timestamp]:
    start = pd.Timestamp(start_date)
    if test_window_months <= 0:
        raise ValueError("Tennis OOS test window must be positive")
    if frame.empty:
        raise ValueError("Cannot build tennis OOS windows from an empty frame")

    latest_event_date = pd.Timestamp(frame["event_date"].max())
    if latest_event_date < start:
        raise ValueError(f"No tennis rows available on or after the OOS boundary {start_date}")

    end = latest_event_date.normalize() + pd.Timedelta(days=1)
    if end <= start:
        raise ValueError("Tennis OOS end date must be after the start date")

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current = start
    while current < end:
        next_boundary = min(current + pd.DateOffset(months=test_window_months), end)
        windows.append((current, next_boundary))
        current = next_boundary
    return windows, end


def summarize_tennis_evaluation(
    evaluation_folds: pd.DataFrame,
    evaluation_predictions: pd.DataFrame,
    evaluation_metrics: dict[str, object],
    evaluation_diagnostics: dict[str, pd.DataFrame],
    training_frame: pd.DataFrame,
    oos_end_date_exclusive: pd.Timestamp,
    *,
    min_matches: int = TENNIS_MIN_MATCHES,
    feature_contract: str = STAGE1_FEATURE_CONTRACT,
    model_name: str = DEFAULT_TENNIS_MODEL_NAME,
) -> dict[str, object]:
    folds = evaluation_folds.copy()
    predictions = evaluation_predictions.copy()
    training_rows = len(training_frame)

    summary = {
        "evaluation_contract": TENNIS_OOS_EVALUATION_CONTRACT,
        "feature_contract": feature_contract,
        "model_name": model_name,
        "training_start_date": TENNIS_TRAINING_START_DATE,
        "oos_start_date": TENNIS_OOS_START_DATE,
        "oos_end_date_exclusive": _format_date(oos_end_date_exclusive),
        "test_window_months": int(TENNIS_OOS_TEST_WINDOW_MONTHS),
        "min_matches": int(min_matches),
        "training_rows_full_eligible": int(training_rows),
        "training_date_min": None,
        "training_date_max": None,
        "oos_prediction_rows": int(len(predictions)),
        "log_loss": None,
        "brier_score": None,
        "accuracy": None,
        "ece_10_bin": None,
        "coverage_matches_eligible_rows": False,
        "folds": [],
    }
    if not training_frame.empty:
        summary["training_date_min"] = _format_date(training_frame["event_date"].min())
        summary["training_date_max"] = _format_date(training_frame["event_date"].max())
        summary["training_rows_by_year"] = {
            str(year): int(count)
            for year, count in training_frame["event_date"].dt.year.value_counts().sort_index().items()
        }
        if "tour" in training_frame.columns:
            summary["training_rows_by_tour"] = {
                str(tour): int(count)
                for tour, count in training_frame["tour"].value_counts().sort_index().items()
            }

    if not folds.empty:
        summary["first_fold_start_date"] = _format_date(folds.iloc[0]["start_date"])
        summary["last_fold_end_date_exclusive"] = _format_date(folds.iloc[-1]["end_date"])
        summary["folds"] = [
            {
                "fold": int(row["fold"]),
                "start_date": _format_date(row["start_date"]),
                "end_date_exclusive": _format_date(row["end_date"]),
                "train_rows": int(row["train_rows"]),
                "test_rows": int(row["test_rows"]),
                "log_loss": None if pd.isna(row["log_loss"]) else float(row["log_loss"]),
                "brier_score": None if pd.isna(row["brier_score"]) else float(row["brier_score"]),
            }
            for _, row in folds.iterrows()
        ]

    if evaluation_metrics:
        for key in ["log_loss", "brier_score", "accuracy", "ece_10_bin"]:
            value = evaluation_metrics.get(key)
            summary[key] = None if value is None or pd.isna(value) else float(value)

    if not predictions.empty:
        summary["prediction_date_min"] = _format_date(predictions["event_date"].min())
        summary["prediction_date_max"] = _format_date(predictions["event_date"].max())
        summary["predictions_by_year"] = {
            str(year): int(count)
            for year, count in predictions["event_date"].dt.year.value_counts().sort_index().items()
        }
        if "tour" in predictions.columns:
            summary["predictions_by_tour"] = {
                str(tour): int(count)
                for tour, count in predictions["tour"].value_counts().sort_index().items()
            }

    for diagnostic_name, diagnostic_frame in evaluation_diagnostics.items():
        if not diagnostic_frame.empty:
            summary[diagnostic_name] = diagnostic_frame.to_dict(orient="records")

    return summary


def summarize_tennis_lockbox_evaluation(
    *,
    predictions: pd.DataFrame,
    metrics: dict[str, object],
    diagnostics: dict[str, pd.DataFrame],
    train_df: pd.DataFrame,
    lockbox_start_date: str,
    feature_contract: str,
    model_name: str,
    min_matches: int,
) -> dict[str, object]:
    """Summarize a single holdout lockbox evaluation."""
    summary = {
        "evaluation_contract": TENNIS_LOCKBOX_EVALUATION_CONTRACT,
        "feature_contract": feature_contract,
        "model_name": model_name,
        "training_start_date": TENNIS_TRAINING_START_DATE,
        "lockbox_start_date": lockbox_start_date,
        "min_matches": int(min_matches),
        "train_rows": int(len(train_df)),
        "lockbox_rows": int(len(predictions)),
        "training_date_min": _format_date(train_df["event_date"].min()) if not train_df.empty else None,
        "training_date_max": _format_date(train_df["event_date"].max()) if not train_df.empty else None,
        "prediction_date_min": _format_date(predictions["event_date"].min()) if not predictions.empty else None,
        "prediction_date_max": _format_date(predictions["event_date"].max()) if not predictions.empty else None,
        "log_loss": None,
        "brier_score": None,
        "accuracy": None,
        "ece_10_bin": None,
    }

    for key in ["log_loss", "brier_score", "accuracy", "ece_10_bin"]:
        value = metrics.get(key)
        summary[key] = None if value is None or pd.isna(value) else float(value)

    if not train_df.empty and "tour" in train_df.columns:
        summary["train_rows_by_tour"] = {
            str(tour): int(count)
            for tour, count in train_df["tour"].value_counts().sort_index().items()
        }
    if not predictions.empty and "tour" in predictions.columns:
        summary["lockbox_rows_by_tour"] = {
            str(tour): int(count)
            for tour, count in predictions["tour"].value_counts().sort_index().items()
        }

    for diagnostic_name, diagnostic_frame in diagnostics.items():
        if not diagnostic_frame.empty:
            summary[diagnostic_name] = diagnostic_frame.to_dict(orient="records")

    return summary


def run_walkforward_evaluation(
    features_df: pd.DataFrame,
    *,
    min_matches: int = TENNIS_MIN_MATCHES,
    feature_contract: str = STAGE1_FEATURE_CONTRACT,
    model_name: str = DEFAULT_TENNIS_MODEL_NAME,
) -> dict[str, object]:
    """Run anchored 6-month walk-forward predictive evaluation on the strict 2022+ tennis universe."""
    frame = _prepare_training_frame(features_df, min_matches=min_matches, feature_contract=feature_contract)
    if frame.empty:
        raise ValueError("No tennis rows available after minimum-history filtering")

    oos_start = pd.Timestamp(TENNIS_OOS_START_DATE)
    if frame[frame["event_date"] < oos_start].empty:
        raise ValueError(
            "No tennis rows available between the strict training boundary "
            f"{TENNIS_TRAINING_START_DATE} and the OOS boundary {TENNIS_OOS_START_DATE}"
        )
    windows, oos_end = _build_expanding_oos_windows(
        frame,
        start_date=TENNIS_OOS_START_DATE,
        test_window_months=TENNIS_OOS_TEST_WINDOW_MONTHS,
    )
    eligible_oos = frame[(frame["event_date"] >= oos_start) & (frame["event_date"] < oos_end)].copy()
    if eligible_oos.empty:
        raise ValueError(
            f"No tennis rows available for anchored OOS evaluation on or after {TENNIS_OOS_START_DATE}"
        )

    fold_predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    feature_cols = require_tennis_feature_columns(frame, feature_contract)

    for fold_index, (start_date, end_date) in enumerate(windows):
        train_df = frame[frame["event_date"] < start_date].copy()
        test_df = frame[(frame["event_date"] >= start_date) & (frame["event_date"] < end_date)].copy()
        fold_row = {
            "fold": fold_index,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "start_date": pd.Timestamp(start_date),
            "end_date": pd.Timestamp(end_date),
            "log_loss": np.nan,
            "brier_score": np.nan,
        }

        if train_df.empty:
            raise ValueError(
                f"Fold {fold_index} starts at {TENNIS_OOS_START_DATE} but has no pre-boundary training rows"
            )
        if test_df.empty:
            fold_rows.append(fold_row)
            continue

        model_result = _fit_model(
            train_df,
            feature_cols=feature_cols,
            feature_contract=feature_contract,
            calibrate=True,
        )
        fold_pred = predict_tennis_batch(test_df, model_result=model_result)
        metrics = evaluate_prediction_frame(fold_pred)
        fold_pred["fold"] = fold_index
        fold_pred["fold_start_date"] = pd.Timestamp(start_date)
        fold_pred["fold_end_date"] = pd.Timestamp(end_date)
        fold_predictions.append(fold_pred)
        fold_row["log_loss"] = metrics["log_loss"]
        fold_row["brier_score"] = metrics["brier_score"]
        fold_rows.append(fold_row)

    combined_predictions = pd.concat(fold_predictions, ignore_index=True) if fold_predictions else pd.DataFrame()
    if not combined_predictions.empty:
        combined_predictions = combined_predictions.sort_values(["event_date", "fold"]).reset_index(drop=True)
        combined_predictions = _add_prediction_diagnostics(combined_predictions)
    overall_metrics = evaluate_prediction_frame(combined_predictions) if not combined_predictions.empty else {}
    diagnostics = build_tennis_evaluation_diagnostics(combined_predictions) if not combined_predictions.empty else {}
    if len(combined_predictions) != len(eligible_oos):
        raise ValueError(
            "Anchored tennis OOS predictions do not match the eligible 2022+ universe after minimum-history filtering"
        )

    summary = summarize_tennis_evaluation(
        evaluation_folds=pd.DataFrame(fold_rows),
        evaluation_predictions=combined_predictions,
        evaluation_metrics=overall_metrics,
        evaluation_diagnostics=diagnostics,
        training_frame=frame,
        oos_end_date_exclusive=oos_end,
        min_matches=min_matches,
        feature_contract=feature_contract,
        model_name=model_name,
    )
    summary["eligible_oos_rows"] = int(len(eligible_oos))
    summary["coverage_matches_eligible_rows"] = int(len(combined_predictions)) == int(len(eligible_oos))
    return {
        "folds": pd.DataFrame(fold_rows),
        "predictions": combined_predictions,
        "metrics": overall_metrics,
        "diagnostics": diagnostics,
        "summary": summary,
        "eligible_oos_rows": int(len(eligible_oos)),
    }


def run_lockbox_evaluation(
    features_df: pd.DataFrame,
    *,
    lockbox_start_date: str,
    min_matches: int = TENNIS_MIN_MATCHES,
    feature_contract: str = STAGE2_FEATURE_CONTRACT,
    model_name: str = DEFAULT_TENNIS_MODEL_NAME,
) -> dict[str, object]:
    """Train strictly before a holdout boundary and evaluate only on the lockbox window."""
    frame = _prepare_training_frame(features_df, min_matches=min_matches, feature_contract=feature_contract)
    if frame.empty:
        raise ValueError("No tennis rows available after minimum-history filtering")

    lockbox_start = pd.Timestamp(lockbox_start_date)
    train_df = frame[frame["event_date"] < lockbox_start].copy()
    lockbox_df = frame[frame["event_date"] >= lockbox_start].copy()
    if train_df.empty:
        raise ValueError(f"No tennis training rows are available before lockbox start {lockbox_start_date}")
    if lockbox_df.empty:
        raise ValueError(f"No tennis holdout rows are available on or after lockbox start {lockbox_start_date}")

    feature_cols = require_tennis_feature_columns(train_df, feature_contract)
    model_result = _fit_model(
        train_df,
        feature_cols=feature_cols,
        feature_contract=feature_contract,
        calibrate=True,
    )
    predictions = predict_tennis_batch(lockbox_df, model_result=model_result)
    predictions = predictions.sort_values("event_date").reset_index(drop=True)
    predictions = _add_prediction_diagnostics(predictions)
    metrics = evaluate_prediction_frame(predictions)
    diagnostics = build_tennis_evaluation_diagnostics(predictions)
    summary = summarize_tennis_lockbox_evaluation(
        predictions=predictions,
        metrics=metrics,
        diagnostics=diagnostics,
        train_df=train_df,
        lockbox_start_date=lockbox_start_date,
        feature_contract=feature_contract,
        model_name=model_name,
        min_matches=min_matches,
    )
    return {
        "train_df": train_df,
        "predictions": predictions,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "summary": summary,
    }


def write_tennis_oos_artifacts(model_result: dict, output_dir: Path | str) -> dict[str, Path]:
    """Write explicit 2022+ OOS evaluation artifacts for a tennis predictive model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    folds = model_result.get("evaluation_folds")
    predictions = model_result.get("evaluation_predictions")
    metrics = model_result.get("evaluation_metrics", {})
    diagnostics = model_result.get("evaluation_diagnostics", {})
    calibration = metrics.get("calibration")
    summary = model_result.get("evaluation_summary")
    artifact_prefix = str(model_result.get("oos_artifact_prefix") or TENNIS_OOS_ARTIFACT_PREFIX)

    if not isinstance(folds, pd.DataFrame) or folds.empty:
        raise ValueError("Missing tennis OOS fold evaluation data")
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise ValueError("Missing tennis OOS prediction evaluation data")
    if not isinstance(calibration, pd.DataFrame) or calibration.empty:
        raise ValueError("Missing tennis OOS calibration evaluation data")
    if not isinstance(summary, dict) or not summary:
        raise ValueError("Missing tennis OOS summary evaluation data")

    artifacts = {
        "folds": output_path / f"{artifact_prefix}_folds.csv",
        "predictions": output_path / f"{artifact_prefix}_predictions.csv",
        "calibration": output_path / f"{artifact_prefix}_calibration.csv",
        "summary": output_path / f"{artifact_prefix}_summary.json",
    }
    folds.to_csv(artifacts["folds"], index=False)
    predictions.to_csv(artifacts["predictions"], index=False)
    calibration.to_csv(artifacts["calibration"], index=False)
    artifacts["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for diagnostic_name, diagnostic_frame in diagnostics.items():
        if isinstance(diagnostic_frame, pd.DataFrame) and not diagnostic_frame.empty:
            artifacts[diagnostic_name] = output_path / f"{artifact_prefix}_{diagnostic_name}.csv"
            diagnostic_frame.to_csv(artifacts[diagnostic_name], index=False)

    return artifacts


def write_tennis_lockbox_artifacts(
    evaluation_result: dict,
    output_dir: Path | str,
    *,
    model_name: str,
    lockbox_start_date: str,
) -> dict[str, Path]:
    """Write holdout lockbox evaluation artifacts for a tennis predictive model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    predictions = evaluation_result.get("predictions")
    metrics = evaluation_result.get("metrics", {})
    diagnostics = evaluation_result.get("diagnostics", {})
    summary = evaluation_result.get("summary")
    calibration = metrics.get("calibration")
    date_token = pd.Timestamp(lockbox_start_date).strftime("%Y%m%d")
    safe_model_name = _normalize_model_name(model_name).replace("-", "_")
    artifact_prefix = f"lockbox_{date_token}_{safe_model_name}"

    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise ValueError("Missing tennis lockbox prediction data")
    if not isinstance(calibration, pd.DataFrame) or calibration.empty:
        raise ValueError("Missing tennis lockbox calibration data")
    if not isinstance(summary, dict) or not summary:
        raise ValueError("Missing tennis lockbox summary data")

    artifacts = {
        "predictions": output_path / f"{artifact_prefix}_predictions.csv",
        "calibration": output_path / f"{artifact_prefix}_calibration.csv",
        "summary": output_path / f"{artifact_prefix}_summary.json",
    }
    predictions.to_csv(artifacts["predictions"], index=False)
    calibration.to_csv(artifacts["calibration"], index=False)
    artifacts["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for diagnostic_name, diagnostic_frame in diagnostics.items():
        if isinstance(diagnostic_frame, pd.DataFrame) and not diagnostic_frame.empty:
            artifacts[diagnostic_name] = output_path / f"{artifact_prefix}_{diagnostic_name}.csv"
            diagnostic_frame.to_csv(artifacts[diagnostic_name], index=False)

    return artifacts


def _portable_tennis_model_payload(model_result: dict) -> dict:
    """Strip evaluation-only payloads so runtime model loading stays lightweight and portable."""
    persisted = dict(model_result)
    for key in TENNIS_MODEL_PORTABILITY_DROP_KEYS:
        persisted.pop(key, None)
    return persisted


def save_tennis_model(model_result: dict, model_name: str = DEFAULT_TENNIS_MODEL_NAME) -> Path:
    """Persist a tennis model under models/tennis."""
    persisted = _portable_tennis_model_payload(model_result)
    output_paths = _tennis_model_paths(model_name)

    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(persisted, output_path)

    primary_output = output_paths[0]
    logger.info("Saved tennis model to %s", primary_output)
    for mirrored_output in output_paths[1:]:
        logger.info("Mirrored tennis model to %s", mirrored_output)
    return primary_output


def load_tennis_model(model_name: str = DEFAULT_TENNIS_MODEL_NAME) -> dict:
    """Load a tennis model from models/tennis."""
    paths = _tennis_model_paths(model_name)
    primary_path = paths[0]
    found_existing_path = False
    load_errors: list[tuple[Path, Exception]] = []
    for path in paths:
        if not path.exists():
            continue
        found_existing_path = True
        try:
            model_result = joblib.load(path)
            _validate_model_result(model_result)
        except Exception as exc:
            load_errors.append((path, exc))
            logger.warning("Failed to load tennis model from %s: %s", path, exc)
            continue
        if path != primary_path and not primary_path.exists():
            try:
                primary_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, primary_path)
                logger.info("Restored tennis model cache from %s to %s", path, primary_path)
            except OSError as exc:
                logger.warning(
                    "Loaded tennis model from fallback %s but could not restore %s: %s",
                    path,
                    primary_path,
                    exc,
                )
        return model_result
    if found_existing_path and load_errors:
        if all(isinstance(exc, ValueError) for _, exc in load_errors):
            raise load_errors[0][1]
        formatted_errors = "; ".join(
            f"{path}: {type(exc).__name__}: {exc}"
            for path, exc in load_errors
        )
        raise RuntimeError(
            f"Failed to load tennis model '{model_name}' from available paths: {formatted_errors}"
        )
    raise FileNotFoundError(f"Tennis model not found: {primary_path}")


def ensure_tennis_model(
    model_name: str = DEFAULT_TENNIS_MODEL_NAME,
    *,
    features_df: Optional[pd.DataFrame] = None,
    min_matches: int = TENNIS_MIN_MATCHES,
) -> dict:
    """Load a saved tennis model or train and persist one from available features."""
    try:
        return load_tennis_model(model_name=model_name)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        if features_df is None:
            raise
        if features_df.empty:
            raise ValueError("Cannot rebuild tennis model from an empty processed feature frame.")
        logger.warning(
            "Tennis model '%s' is unavailable (%s). Rebuilding from processed features.",
            model_name,
            exc,
        )

    logger.warning(
        "Tennis model '%s' rebuild starting from %s processed feature rows.",
        model_name,
        len(features_df),
    )
    return train_tennis_model(
        features_df,
        model_name=model_name,
        min_matches=min_matches,
    )


def train_tennis_model(
    features_df: pd.DataFrame,
    model_name: str = DEFAULT_TENNIS_MODEL_NAME,
    min_matches: int = TENNIS_MIN_MATCHES,
    *,
    feature_contract: Optional[str] = None,
) -> dict:
    """Evaluate the strict 2022+ tennis model, then fit a final saved model on all eligible rows."""
    resolved_contract = infer_tennis_feature_contract(model_name, feature_contract=feature_contract)
    training_frame = _prepare_training_frame(
        features_df,
        min_matches=min_matches,
        feature_contract=resolved_contract,
    )
    evaluation = run_walkforward_evaluation(
        features_df=features_df,
        min_matches=min_matches,
        feature_contract=resolved_contract,
        model_name=model_name,
    )
    feature_cols = require_tennis_feature_columns(training_frame, resolved_contract)
    final_model = _fit_model(
        training_frame,
        feature_cols=feature_cols,
        feature_contract=resolved_contract,
        calibrate=True,
    )
    result = {
        **final_model,
        "model_name": model_name,
        "training_rows": len(training_frame),
        "training_start_date": TENNIS_TRAINING_START_DATE,
        "training_date_min": _format_date(training_frame["event_date"].min()),
        "training_date_max": _format_date(training_frame["event_date"].max()),
        "evaluation_folds": evaluation["folds"],
        "evaluation_metrics": evaluation["metrics"],
        "evaluation_diagnostics": evaluation.get("diagnostics", {}),
        "evaluation_predictions": evaluation["predictions"],
        "evaluation_summary": evaluation["summary"],
        "model_family": "logistic_regression",
        "oos_artifact_prefix": _artifact_prefix_for_contract(resolved_contract),
    }
    save_tennis_model(result, model_name=model_name)
    return result
