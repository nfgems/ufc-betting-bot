"""
Model training — trains XGBoost and Logistic Regression models
for UFC fight prediction with probability calibration.
"""

import logging
import inspect
import subprocess
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    TRAIN_CUTOFF_DATE, MODELS_DIR, PROCESSED_DATA_DIR,
    TIME_DECAY_ENABLED, TIME_DECAY_HALF_LIFE_DAYS,
    ODDS_NOISE_STD,
)
from src.features.build_features import (
    ODDS_FEATURE_NAMES,
    exclude_market_derived_features,
    get_feature_columns,
    get_feature_columns_no_odds,
)

logger = logging.getLogger(__name__)


def _call_train_logistic(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    odds_noise_std: float,
) -> dict:
    """Call train_logistic while remaining compatible with older test doubles."""
    signature = inspect.signature(train_logistic)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs or "odds_noise_std" in signature.parameters:
        return train_logistic(
            train_df,
            feature_cols,
            odds_noise_std=odds_noise_std,
        )
    return train_logistic(train_df, feature_cols)


def _resolve_repo_git_hash() -> str:
    """Return the current repo HEAD hash, or an empty string if unavailable."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def _dead_train_feature_columns(train_df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Return declared feature columns that are entirely missing on trainable rows."""
    dead_columns: list[str] = []
    for column in feature_cols:
        if column not in train_df.columns:
            continue
        if train_df[column].isna().all():
            dead_columns.append(column)
    return dead_columns


def prepare_train_test(
    features_df: pd.DataFrame,
    cutoff_date: Optional[str] = None,
    min_fights: int = 2,
    feature_cols: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Split features into train/test sets by date.
    Filters out fights where either fighter has fewer than min_fights prior bouts.

    If *start_date* is provided, training rows before that date are dropped.
    If *end_date* is provided, training rows on/after that date are dropped.
    The test set is unaffected.

    Returns (train_df, test_df, feature_columns).
    """
    cutoff = pd.Timestamp(cutoff_date or TRAIN_CUTOFF_DATE)
    feature_cols = feature_cols or get_feature_columns(features_df)

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

    # Apply optional lower-bound filter to training data only
    if start_date:
        start = pd.Timestamp(start_date)
        pre_filter_count = len(train)
        train = train[train["event_date"] >= start].copy()
        logger.info(
            f"train_start_date={start.date()}: dropped {pre_filter_count - len(train)} "
            f"pre-{start.date()} training rows"
        )
    if end_date:
        end = pd.Timestamp(end_date)
        pre_filter_count = len(train)
        train = train[train["event_date"] < end].copy()
        logger.info(
            f"train_end_date={end.date()}: dropped {pre_filter_count - len(train)} "
            f"training rows on/after {end.date()}"
        )

    logger.info(
        f"Train: {len(train)} fights (before {cutoff.date()}), "
        f"Test: {len(test)} fights (after {cutoff.date()})"
    )
    logger.info(f"Using {len(feature_cols)} features")

    return train, test, feature_cols


def _compute_sample_weights(
    train_df: pd.DataFrame,
    *,
    half_life_days: float | None = None,
) -> Optional[np.ndarray]:
    """Compute time-decay sample weights based on fight recency."""
    if not TIME_DECAY_ENABLED:
        return None
    if "event_date" not in train_df.columns:
        return None

    effective_half_life = half_life_days if half_life_days is not None else TIME_DECAY_HALF_LIFE_DAYS

    dates = pd.to_datetime(train_df["event_date"])
    max_date = dates.max()
    days_ago = (max_date - dates).dt.days.values.astype(float)

    # Exponential decay: weight = 2^(-days_ago / half_life)
    weights = np.power(2.0, -days_ago / effective_half_life)

    # Normalize so mean weight = 1.0 (preserves effective sample size interpretation)
    weights = weights / weights.mean()

    logger.info(
        f"Time-decay weights: half-life={effective_half_life}d, "
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
        rng = np.random.RandomState()

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
    impute_strategy: str = "native_nan",
    xgb_params: dict | None = None,
    calibration_method: str = "isotonic",
    calibration_cv: str = "timeseries_5fold",
    odds_noise_std: float = ODDS_NOISE_STD,
    time_decay_half_life_days: float | None = None,
) -> dict:
    """
    Train an XGBoost classifier with optional probability calibration.

    Args:
        impute_strategy: "native_nan" preserves NaN for XGBoost's native
            missing-value handling. "median" uses legacy median imputation.
        xgb_params: Override XGBoost hyperparameters (None = production defaults).
        calibration_method: "isotonic", "sigmoid", or "none".
        calibration_cv: "timeseries_5fold" or "random_5fold".

    Returns dict with:
        - model: trained model (or calibrated wrapper)
        - feature_cols: list of feature columns used
        - feature_importance: dict of feature name -> importance
        - col_medians: median array (for LogisticRegression or legacy compat)
        - impute_strategy: which strategy was used
    """
    X_train = train_df[feature_cols].values.copy()
    y_train = train_df["target"].values

    # Compute medians regardless (needed for col_medians metadata)
    col_medians = np.nanmedian(X_train, axis=0)

    if impute_strategy == "median":
        # Legacy path: median imputation + missing indicators
        indicator_cols = []
        indicator_names = []
        for i in range(X_train.shape[1]):
            mask = np.isnan(X_train[:, i])
            if mask.any():
                indicator_cols.append(mask.astype(float))
                indicator_names.append(f"{feature_cols[i]}_missing")
            X_train[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

        if indicator_cols:
            X_train = np.column_stack([X_train] + indicator_cols)
            feature_cols = list(feature_cols) + indicator_names
            logger.info(f"Added {len(indicator_cols)} missing value indicator features")
    else:
        # native_nan: XGBoost handles NaN natively — no imputation
        n_nan = np.isnan(X_train).sum()
        logger.info(f"Native NaN mode: {n_nan} NaN values preserved for XGBoost")

    # Add noise to odds features to mitigate closing odds leakage
    X_train = _add_odds_noise(X_train, feature_cols, noise_std=odds_noise_std)

    # Compute time-decay sample weights
    sample_weights = _compute_sample_weights(
        train_df,
        half_life_days=time_decay_half_life_days,
    )

    # XGBoost hyperparameters
    default_params = {
        "n_estimators": 135,
        "max_depth": 7,
        "learning_rate": 0.0124,
        "subsample": 0.659,
        "colsample_bytree": 0.706,
        "min_child_weight": 6,
        "gamma": 0.444,
        "reg_alpha": 0.00443,
        "reg_lambda": 0.00772,
        "scale_pos_weight": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    params = xgb_params if xgb_params is not None else default_params
    xgb = XGBClassifier(**params)

    model = xgb
    if calibrate and calibration_method != "none":
        from sklearn.base import clone as _sklearn_clone

        if calibration_cv == "timeseries_5fold":
            cv = TimeSeriesSplit(n_splits=5)
        else:
            cv = 5  # random_5fold
        # Pass an *unfitted* clone so CalibratedClassifierCV fits fresh
        # base estimators inside each CV fold — avoids data leakage.
        model = CalibratedClassifierCV(
            _sklearn_clone(xgb), cv=cv, method=calibration_method,
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        # Fit the raw xgb on the full training set for feature importances
        xgb.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        xgb.fit(X_train, y_train, sample_weight=sample_weights)

    # Feature importance from the raw XGBoost model
    importance = dict(zip(feature_cols, xgb.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    logger.info("Top 10 features:")
    for feat, imp in list(importance.items())[:10]:
        logger.info(f"  {feat}: {imp:.4f}")

    # Extend col_medians for indicator columns if median imputation was used
    if impute_strategy == "median":
        n_indicators = len([c for c in feature_cols if c.endswith("_missing")])
        if n_indicators > 0:
            col_medians = np.concatenate([col_medians, np.zeros(n_indicators)])

    return {
        "model": model,
        "raw_model": xgb,
        "feature_cols": feature_cols,
        "feature_importance": importance,
        "col_medians": col_medians,
        "impute_strategy": impute_strategy,
    }


def train_logistic(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    odds_noise_std: float = ODDS_NOISE_STD,
) -> dict:
    """
    Train a Logistic Regression baseline with StandardScaler.
    LR naturally produces calibrated probabilities.
    """
    X_train = train_df[feature_cols].values.copy()
    y_train = train_df["target"].values

    col_medians = np.nanmedian(X_train, axis=0)
    for i in range(X_train.shape[1]):
        mask = np.isnan(X_train[:, i])
        X_train[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

    # Add noise to odds features to mitigate closing odds leakage
    X_train = _add_odds_noise(X_train, feature_cols, noise_std=odds_noise_std)

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


def train_all_models(
    features_df: pd.DataFrame,
    spec: "NamedModelTrainingSpec | None" = None,
    *,
    models_dir: Path | None = None,
    test_set_path: Path | None = None,
) -> dict:
    """
    Train XGBoost, Logistic Regression, and no-odds baseline models.
    Saves models to disk. Returns dict of model results.

    If a NamedModelTrainingSpec is provided, it overrides the default
    feature selection, imputation strategy, and hyperparameters.
    """
    if spec is not None:
        # Validate the spec's feature contract
        violations = spec.validate_feature_contract()
        if violations:
            for v in violations:
                logger.error(v)
            raise ValueError(f"Training spec has {len(violations)} contract violations")

        from src.model.training_spec import (
            compute_feature_family_coverage,
            materialize_and_validate_spec_features,
        )
        from datetime import datetime

        if not spec.trained_at:
            spec.trained_at = datetime.now().isoformat()
        if not spec.git_hash:
            spec.git_hash = _resolve_repo_git_hash()
        features_df = materialize_and_validate_spec_features(features_df, spec)

        # Use the spec's ordered feature list after materializing every requested transform.
        feature_cols = list(spec.feature_cols)

        cutoff = spec.train_cutoff_date
        train_df, test_df, _ = prepare_train_test(
            features_df,
            cutoff_date=cutoff,
            feature_cols=feature_cols,
            start_date=spec.train_start_date or None,
            end_date=getattr(spec, "train_end_date", "") or None,
        )
        dead_train_columns = _dead_train_feature_columns(train_df, feature_cols)
        if dead_train_columns:
            raise ValueError(
                f"Training spec '{spec.name}' has dead train-time contract columns "
                f"for dataset variant '{spec.dataset_variant}': {dead_train_columns}. "
                "Refusing to train until the dataset or spec is fixed."
            )
        required_family_coverage = dict(getattr(spec, "required_feature_family_coverage_pct", {}) or {})
        if required_family_coverage:
            coverage_summary = compute_feature_family_coverage(
                train_df,
                feature_cols=feature_cols,
                family_names=list(required_family_coverage.keys()),
            )
            coverage_failures = []
            for family_name, min_pct in required_family_coverage.items():
                family_summary = coverage_summary.get(family_name)
                if family_summary is None:
                    coverage_failures.append(
                        f"{family_name}: no active feature columns found in the training frame"
                    )
                    continue
                actual_pct = float(family_summary["coverage_pct"])
                if actual_pct + 1e-9 < float(min_pct):
                    coverage_failures.append(
                        f"{family_name}: {actual_pct:.2f}% complete coverage "
                        f"({family_summary['rows_complete']}/{family_summary['rows_total']}) "
                        f"< required {float(min_pct):.2f}%"
                    )
            if coverage_failures:
                joined = "; ".join(coverage_failures)
                raise ValueError(
                    f"Training spec '{spec.name}' failed required external-family coverage gates: {joined}"
                )
        impute_strategy = spec.impute_strategy
        xgb_params = spec.xgb_params
        calibration_method = spec.calibration_method
        calibration_cv = spec.calibration_cv
        time_decay_half_life_days = spec.time_decay_half_life
        odds_noise_std = spec.odds_noise_std
        logger.info(f"Training with spec '{spec.name}': {len(feature_cols)} features, "
                     f"impute={impute_strategy}, cal={calibration_method}")
    else:
        train_df, test_df, feature_cols = prepare_train_test(features_df)
        impute_strategy = "median"  # Legacy default
        xgb_params = None
        calibration_method = "isotonic"
        calibration_cv = "timeseries_5fold"
        time_decay_half_life_days = None
        odds_noise_std = ODDS_NOISE_STD

        # SHAP feature selection — reduce to top 40 features to avoid overfitting
        try:
            from src.model.feature_selection import select_top_features
            feature_cols = select_top_features(
                train_df, feature_cols, n_keep=40, method="shap"
            )
            logger.info(f"SHAP selected {len(feature_cols)} features")
        except Exception as e:
            logger.warning(f"SHAP feature selection failed, using all features: {e}")

    # Train models
    logger.info("Training XGBoost...")
    xgb_result = train_xgboost(
        train_df, feature_cols,
        impute_strategy=impute_strategy,
        xgb_params=xgb_params,
        calibration_method=calibration_method,
        calibration_cv=calibration_cv,
        odds_noise_std=odds_noise_std,
        time_decay_half_life_days=time_decay_half_life_days,
    )

    logger.info("Training Logistic Regression...")
    # LR always uses median imputation (StandardScaler cannot handle NaN)
    lr_result = _call_train_logistic(
        train_df,
        feature_cols,
        odds_noise_std=odds_noise_std,
    )

    # Train no-odds baseline (fighter stats only — measures independent edge)
    no_odds_cols = get_feature_columns_no_odds(features_df)
    no_odds_cols = [c for c in no_odds_cols if c in train_df.columns]
    if spec is not None:
        # Constrain no-odds cols to the promoted contract minus market-derived features.
        no_odds_cols = exclude_market_derived_features(feature_cols)
    logger.info(f"Training XGBoost (no-odds baseline, {len(no_odds_cols)} features)...")
    xgb_no_odds_result = train_xgboost(
        train_df, no_odds_cols,
        impute_strategy=impute_strategy,
        xgb_params=xgb_params,
        calibration_method=calibration_method,
        calibration_cv=calibration_cv,
        odds_noise_std=odds_noise_std,
        time_decay_half_life_days=time_decay_half_life_days,
    )

    # Embed spec in all model dicts for portability (no sidecar dependency)
    if spec is not None:
        from dataclasses import asdict
        spec_dict = asdict(spec)
        xgb_result["training_spec"] = spec_dict
        lr_result["training_spec"] = spec_dict
        xgb_no_odds_result["training_spec"] = spec_dict

    # Save models
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    test_set_path = Path(test_set_path) if test_set_path is not None else PROCESSED_DATA_DIR / "test_set.csv"
    models_dir.mkdir(parents=True, exist_ok=True)
    test_set_path.parent.mkdir(parents=True, exist_ok=True)

    xgb_path = models_dir / "xgboost_model.pkl"
    lr_path = models_dir / "logistic_model.pkl"
    no_odds_path = models_dir / "xgboost_no_odds_model.pkl"
    joblib.dump(xgb_result, xgb_path)
    joblib.dump(lr_result, lr_path)
    joblib.dump(xgb_no_odds_result, no_odds_path)
    logger.info(f"Saved XGBoost to {xgb_path}")
    logger.info(f"Saved Logistic Regression to {lr_path}")
    logger.info(f"Saved XGBoost (no-odds) to {no_odds_path}")

    # Save training spec alongside model if provided
    if spec is not None:
        spec_path = models_dir / f"{spec.name}_spec.json"
        spec.save(spec_path)
    else:
        spec_path = None

    # Save test set for evaluation
    test_df.to_csv(test_set_path, index=False)

    return {
        "xgboost": xgb_result,
        "logistic": lr_result,
        "xgboost_no_odds": xgb_no_odds_result,
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "no_odds_feature_cols": no_odds_cols,
        "spec": spec,
        "models_dir": models_dir,
        "test_set_path": test_set_path,
        "spec_path": spec_path,
    }


def load_model(model_name: str = "xgboost") -> dict:
    """Load a saved model from disk by model alias or explicit artifact path.

    Rejects artifacts that do not embed the exact training feature contract.
    """
    model_ref = Path(model_name)
    if model_ref.suffix == ".pkl" or model_ref.is_absolute() or any(sep in model_name for sep in ("/", "\\")):
        path = model_ref
    else:
        path = MODELS_DIR / f"{model_name}_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    result = joblib.load(path)

    artifact_cols = result.get("feature_cols")
    spec = result.get("training_spec")
    if not isinstance(spec, dict):
        raise ValueError(
            f"Model artifact {path} is missing an embedded training_spec. "
            "Retrain the model with the current contract before using it."
        )
    if not isinstance(artifact_cols, list):
        raise ValueError(f"Model artifact {path} is missing feature_cols.")

    spec_cols = spec.get("feature_cols")
    if not isinstance(spec_cols, list):
        raise ValueError(f"Model artifact {path} has an invalid embedded training_spec feature contract.")
    if artifact_cols != spec_cols:
        raise ValueError(
            f"Model artifact {path} failed feature-contract validation: "
            "feature_cols do not exactly match embedded training_spec.feature_cols."
        )
    return result
