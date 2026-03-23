"""
Prediction module — generates win probabilities for upcoming fights.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.model.train import load_model

logger = logging.getLogger(__name__)


def _resolve_impute_strategy(model_result: dict) -> str:
    """Return the artifact's declared imputation mode or fail closed."""
    strategy = model_result.get("impute_strategy")
    if strategy in {"native_nan", "median"}:
        return strategy

    model_obj = model_result.get("model")
    if model_obj is not None:
        model_type = type(model_obj).__name__
        if model_type == "Pipeline":
            return "median"

    training_spec = model_result.get("training_spec")
    if isinstance(training_spec, dict):
        spec_strategy = training_spec.get("impute_strategy")
        if spec_strategy in {"native_nan", "median"}:
            return spec_strategy

    raise ValueError(
        "Model artifact is missing a valid impute_strategy. "
        "Resave or retrain the artifact before using it for prediction."
    )


def _ordered_feature_frame(features_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a frame with the requested columns, filling absent ones with NaN."""
    ordered = pd.DataFrame(index=features_df.index)
    for column in columns:
        if column in features_df.columns:
            ordered[column] = pd.to_numeric(features_df[column], errors="coerce")
        else:
            ordered[column] = np.nan
    return ordered


def _build_batch_matrix(features_df: pd.DataFrame, model_result: dict) -> np.ndarray:
    """Materialize the exact inference matrix for a batch using a saved model dict."""
    feature_cols = model_result["feature_cols"]
    col_medians = model_result["col_medians"]
    impute_strategy = _resolve_impute_strategy(model_result)

    if impute_strategy == "native_nan":
        return _ordered_feature_frame(features_df, feature_cols).to_numpy(copy=True)

    base_cols = [c for c in feature_cols if not c.endswith("_missing")]
    missing_indicator_cols = [c for c in feature_cols if c.endswith("_missing")]
    X_base = _ordered_feature_frame(features_df, base_cols).to_numpy(copy=True)

    indicator_arrays = []
    for i, col in enumerate(base_cols):
        if f"{col}_missing" in missing_indicator_cols:
            indicator_arrays.append(np.isnan(X_base[:, i]).astype(float))

    for i in range(X_base.shape[1]):
        mask = np.isnan(X_base[:, i])
        if mask.any():
            X_base[mask, i] = col_medians[i] if i < len(col_medians) and not np.isnan(col_medians[i]) else 0.0

    if indicator_arrays:
        return np.column_stack([X_base] + indicator_arrays)
    return X_base


def predict_fight(
    features: dict,
    model_name: str = "xgboost",
    model_result: Optional[dict] = None,
) -> dict:
    """
    Predict a single fight.

    Args:
        features: dict of feature_name -> value for this fight
        model_name: which model to use ("xgboost" or "logistic")
        model_result: pre-loaded model dict (if None, loads from disk)

    Returns dict with:
        - prob_a: probability fighter A wins
        - prob_b: probability fighter B wins
        - predicted_winner: "a" or "b"
        - confidence: max(prob_a, prob_b)
    """
    if model_result is None:
        model_result = load_model(model_name)

    model = model_result["model"]
    feature_cols = model_result["feature_cols"]
    col_medians = model_result["col_medians"]
    impute_strategy = _resolve_impute_strategy(model_result)

    if impute_strategy == "native_nan":
        # Native NaN mode: pass NaN directly to XGBoost (no imputation)
        values = [features.get(col, np.nan) for col in feature_cols]
        X = np.array([values])
    else:
        # Legacy median imputation path
        # Separate base features from missing indicators
        base_cols = [c for c in feature_cols if not c.endswith("_missing")]
        missing_indicator_cols = [c for c in feature_cols if c.endswith("_missing")]

        # Build feature vector from base features
        base_values = [features.get(col, np.nan) for col in base_cols]
        X_base = np.array([base_values])

        # Generate missing indicators (1 if NaN, 0 otherwise)
        indicators = [float(np.isnan(X_base[0, i])) for i, col in enumerate(base_cols)
                      if f"{col}_missing" in missing_indicator_cols]

        # Fill NaNs with training medians
        for i in range(X_base.shape[1]):
            if np.isnan(X_base[0, i]):
                X_base[0, i] = col_medians[i] if i < len(col_medians) and not np.isnan(col_medians[i]) else 0.0

        # Combine base features + indicators
        if indicators:
            X = np.column_stack([X_base, np.array([indicators])])
        else:
            X = X_base

    proba = model.predict_proba(X)[0]
    prob_a = proba[1]  # Probability of class 1 (fighter A wins)
    prob_b = proba[0]  # Probability of class 0 (fighter B wins)

    return {
        "prob_a": float(prob_a),
        "prob_b": float(prob_b),
        "predicted_winner": "a" if prob_a > prob_b else "b",
        "confidence": float(max(prob_a, prob_b)),
    }


def predict_batch(
    features_df: pd.DataFrame,
    model_name: str = "xgboost",
    model_result: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Predict multiple fights at once.

    Args:
        features_df: DataFrame with feature columns
        model_name: which model to use
        model_result: pre-loaded model dict

    Returns the input DataFrame with added prediction columns.
    """
    if model_result is None:
        model_result = load_model(model_name)

    model = model_result["model"]
    X = _build_batch_matrix(features_df, model_result)

    proba = model.predict_proba(X)

    result = features_df.copy()
    result["prob_a"] = proba[:, 1]
    result["prob_b"] = proba[:, 0]
    result["predicted_winner"] = np.where(proba[:, 1] > 0.5, "a", "b")
    result["confidence"] = np.maximum(proba[:, 0], proba[:, 1])

    logger.info(
        f"Predicted {len(result)} fights. "
        f"Mean confidence: {result['confidence'].mean():.3f}"
    )

    return result


def predict_upcoming(
    fighter_a_features: dict,
    fighter_b_features: dict,
    model_name: str = "xgboost",
) -> dict:
    """
    Predict an upcoming fight given pre-computed rolling stats for each fighter.

    This is a convenience wrapper that builds the differential features
    and runs prediction.

    Args:
        fighter_a_features: dict of rolling stats for fighter A
        fighter_b_features: dict of rolling stats for fighter B
        model_name: which model to use

    Returns prediction dict with probabilities.
    """
    model_result = load_model(model_name)
    feature_cols = model_result["feature_cols"]

    # Build feature dict by combining both fighters' stats
    features = {}

    # Copy individual stats
    for key, val in fighter_a_features.items():
        features[f"a_{key}"] = val
    for key, val in fighter_b_features.items():
        features[f"b_{key}"] = val

    # Compute differentials
    for key in fighter_a_features:
        a_val = fighter_a_features.get(key)
        b_val = fighter_b_features.get(key)
        if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
            features[f"diff_{key}"] = a_val - b_val

    # Validate feature coverage against model's expected features
    missing = [f for f in feature_cols if f not in features]
    if missing:
        logger.warning(
            "predict_upcoming: %d features missing from input (will be NaN): %s",
            len(missing), missing[:10],
        )

    return predict_fight(features, model_result=model_result)
