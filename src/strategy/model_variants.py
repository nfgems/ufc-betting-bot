"""
Model variant definitions for the model lab.

Each variant is a VariantConfig that overrides specific aspects of the
production baseline. Variants are tested in isolation to measure causal impact.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from src.config import (
    BLEND_WEIGHT,
    BLEND_WEIGHT_MIN,
    BLEND_WEIGHT_MAX,
    BLEND_CONFIDENCE_THRESHOLD,
    BLEND_AGREEMENT_BOOST,
    KELLY_FRACTION,
    MAX_BET_FRACTION,
    MIN_EDGE_THRESHOLD,
    INITIAL_BANKROLL,
    TIME_DECAY_ENABLED,
    TIME_DECAY_HALF_LIFE_DAYS,
    ODDS_NOISE_STD,
    CONVICTION_MIN_MODEL_PROB,
    ROLLING_WINDOW,
    ELO_K_FACTOR,
    ELO_INITIAL,
)
from src.features.build_features import (
    get_feature_columns,
    get_feature_columns_no_odds,
    ODDS_FEATURE_NAMES,
    STAT_COLUMNS,
)
from src.model.train import _compute_sample_weights, _add_odds_noise

logger = logging.getLogger(__name__)


@dataclass
class VariantConfig:
    """Configuration for a model variant experiment."""

    name: str
    description: str

    # Model training
    train_fn: Optional[Callable] = None  # Custom training function
    calibration_method: str = "isotonic"  # "isotonic", "sigmoid", "none"
    calibration_cv: str = "random_5fold"  # "random_5fold", "temporal_holdout", "timeseries_5fold"

    # Features
    feature_builder_fn: Optional[Callable] = None  # Custom feature builder
    feature_cols: Optional[list[str]] = None  # Exact contract columns when required
    use_ewm: bool = False
    ewm_halflife: int = 3  # Fights
    impute_with_indicators: bool = False  # Add _is_missing binary columns

    # Strategy
    blend_weight: float = BLEND_WEIGHT
    min_edge: float = MIN_EDGE_THRESHOLD
    kelly_fraction: float = KELLY_FRACTION
    max_bet_fraction: float = MAX_BET_FRACTION
    use_independent_blend_b: bool = True  # Use dyn_weight_b for side B (was False pre-fix)

    # XGBoost hyperparameters (None = use production defaults)
    xgb_params: Optional[dict] = None

    # Training overrides
    time_decay_half_life: Optional[int] = None  # Override TIME_DECAY_HALF_LIFE_DAYS
    odds_noise_std: Optional[float] = None  # Override ODDS_NOISE_STD

    # Feature selection
    max_features: Optional[int] = None  # If set, keep only top N by importance

    # Extra features to add
    add_elo_momentum: bool = False
    add_strength_of_schedule: bool = False
    add_rematch_features: bool = False
    add_line_movement: bool = False  # Merge historical line movement features


# ---------------------------------------------------------------------------
# Training helpers used by variants
# ---------------------------------------------------------------------------

def _production_xgb_params() -> dict:
    """Return the production XGBoost hyperparameters."""
    return {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
        "use_label_encoder": False,
    }


def train_variant_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    variant: VariantConfig,
) -> dict:
    """
    Train a model according to the variant configuration.

    Mirrors src.model.train.train_xgboost but with variant-specific overrides
    for calibration, imputation, and hyperparameters.
    """
    X_train = train_df[feature_cols].values.copy()
    y_train = train_df["target"].values.copy()

    # --- Imputation ---
    col_medians = np.nanmedian(X_train, axis=0)
    indicator_cols = []
    indicator_indices: list[int] = []

    # Check if this variant should use native NaN handling
    # (inherits from NamedModelTrainingSpec if available)
    use_native_nan = getattr(variant, '_native_nan', False)

    if use_native_nan:
        # Native NaN: XGBoost handles missing values — no imputation
        n_nan = np.isnan(X_train).sum()
        logger.info(f"Native NaN mode: {n_nan} NaN values preserved for XGBoost")
    elif variant.impute_with_indicators:
        # Add binary indicator columns for missing values
        for i in range(X_train.shape[1]):
            mask = np.isnan(X_train[:, i])
            if mask.any():
                indicator_indices.append(i)
                indicator = np.zeros(X_train.shape[0])
                indicator[mask] = 1.0
                indicator_cols.append(indicator)
                X_train[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0
            else:
                # No NaNs in this column
                pass
        if indicator_cols:
            X_train = np.column_stack([X_train] + indicator_cols)
            logger.info(f"Added {len(indicator_cols)} missing-value indicator columns")
    else:
        for i in range(X_train.shape[1]):
            mask = np.isnan(X_train[:, i])
            X_train[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

    # --- Odds noise ---
    noise_std = variant.odds_noise_std if variant.odds_noise_std is not None else ODDS_NOISE_STD
    X_train = _add_odds_noise(X_train, feature_cols, noise_std=noise_std)

    # --- Sample weights ---
    if variant.time_decay_half_life is not None:
        # Override half-life for this variant
        import src.config as cfg
        original_half_life = cfg.TIME_DECAY_HALF_LIFE_DAYS
        cfg.TIME_DECAY_HALF_LIFE_DAYS = variant.time_decay_half_life
        sample_weights = _compute_sample_weights(train_df)
        cfg.TIME_DECAY_HALF_LIFE_DAYS = original_half_life
    else:
        sample_weights = _compute_sample_weights(train_df)

    # --- XGBoost ---
    params = variant.xgb_params or _production_xgb_params()
    xgb = XGBClassifier(**params)

    # --- Calibration ---
    model = xgb
    if variant.calibration_method != "none":
        from sklearn.base import clone as _sklearn_clone

        method = variant.calibration_method  # "isotonic" or "sigmoid"

        if variant.calibration_cv == "temporal_holdout":
            # Chronological 80/20 split — train on 80%, calibrate on held-out 20%
            split_idx = int(len(train_df) * 0.8)
            X_inner = X_train[:split_idx]
            y_inner = y_train[:split_idx]
            X_cal = X_train[split_idx:]
            y_cal = y_train[split_idx:]
            w_inner = sample_weights[:split_idx] if sample_weights is not None else None

            xgb_inner = XGBClassifier(**params)
            xgb_inner.fit(X_inner, y_inner, sample_weight=w_inner)
            # FrozenEstimator prevents refitting; use a proper train/test split
            # where the calibrator is evaluated on data the base model never saw.
            train_indices = np.array([], dtype=int)
            test_indices = np.arange(len(X_cal))
            model = CalibratedClassifierCV(
                FrozenEstimator(xgb_inner),
                cv=[(train_indices, test_indices)],
                method=method,
            )
            model.fit(X_cal, y_cal)

        elif variant.calibration_cv == "timeseries_5fold":
            # TimeSeriesSplit respects temporal ordering — pass unfitted clone
            model = CalibratedClassifierCV(
                _sklearn_clone(xgb), cv=TimeSeriesSplit(n_splits=5), method=method,
            )
            model.fit(X_train, y_train, sample_weight=sample_weights)

        else:
            # Production default: random 5-fold — pass unfitted clone
            model = CalibratedClassifierCV(_sklearn_clone(xgb), cv=5, method=method)
            model.fit(X_train, y_train, sample_weight=sample_weights)

        # Fit raw xgb on full training data for feature importances
        xgb.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        xgb.fit(X_train, y_train, sample_weight=sample_weights)

    # Feature importance from raw XGBoost (before calibration wrapper)
    importance = dict(zip(feature_cols, xgb.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    return {
        "model": model,
        "raw_model": xgb,
        "feature_cols": feature_cols,
        "feature_importance": importance,
        "col_medians": col_medians,
        "n_indicator_cols": len(indicator_cols),
        "indicator_indices": indicator_indices,
        "impute_strategy": "native_nan" if use_native_nan else "median",
    }


# ---------------------------------------------------------------------------
# EWM rolling stats (variant 7)
# ---------------------------------------------------------------------------

def build_features_ewm(
    fights_df: pd.DataFrame,
    halflife: int = 3,
) -> pd.DataFrame:
    """
    Build features using exponentially weighted rolling stats instead of flat window.

    Reuses the production build_features pipeline but patches _compute_rolling_stats
    to use EWM.
    """
    from src.features.build_features import (
        build_features,
        _compute_per_fight_stats,
        _compute_rolling_stats,
    )
    import src.features.build_features as bf_module

    # Monkey-patch _compute_rolling_stats with EWM version
    original_fn = bf_module._compute_rolling_stats

    def _ewm_rolling_stats(fighter_fights, window=ROLLING_WINDOW):
        fighter_fights = fighter_fights.sort_values("event_date").copy()
        stats_to_roll = STAT_COLUMNS + [f"opp_{s}" for s in STAT_COLUMNS] + ["won"]
        for stat in stats_to_roll:
            if stat in fighter_fights.columns:
                shifted = fighter_fights[stat].shift(1)
                fighter_fights[f"roll_{stat}"] = (
                    shifted.ewm(halflife=halflife, min_periods=1).mean()
                )
        # Win streak (same as production)
        wins = fighter_fights["won"].shift(1).fillna(0)
        streaks = []
        current_streak = 0
        for w in wins:
            if w == 1:
                current_streak += 1
            else:
                current_streak = 0
            streaks.append(current_streak)
        fighter_fights["current_win_streak"] = streaks
        fighter_fights["num_fights"] = range(1, len(fighter_fights) + 1)
        dates = fighter_fights["event_date"]
        fighter_fights["days_since_last_fight"] = dates.diff().dt.days.fillna(365)
        return fighter_fights

    bf_module._compute_rolling_stats = _ewm_rolling_stats
    try:
        features = build_features(fights_df)
    finally:
        bf_module._compute_rolling_stats = original_fn

    return features


# ---------------------------------------------------------------------------
# Weight class mode fix (variant 3 / bug 4)
# ---------------------------------------------------------------------------

def build_features_wc_mode_fix(fights_df: pd.DataFrame) -> pd.DataFrame:
    """Build features with weight class move detection using mode instead of first-seen."""
    from src.features.build_features import build_features
    import src.features.build_features as bf_module

    original_fn = bf_module._detect_weight_class_moves

    def _detect_wc_moves_mode(features: pd.DataFrame) -> None:
        wc_weight = {
            "strawweight": 115, "women's strawweight": 115,
            "flyweight": 125, "women's flyweight": 125,
            "bantamweight": 135, "women's bantamweight": 135,
            "featherweight": 145, "women's featherweight": 145,
            "lightweight": 155, "welterweight": 170,
            "middleweight": 185, "light heavyweight": 205,
            "heavyweight": 265, "catch weight": None,
        }

        def _wc_to_weight(wc_str):
            if pd.isna(wc_str):
                return None
            for k, v in wc_weight.items():
                if k in str(wc_str).lower():
                    return v
            return None

        # Build rolling prior-mode weight class per fighter (per-fight basis)
        fighter_wc_history: dict[str, Counter] = {}
        wc_move_a: dict[int, int] = {}
        wc_move_b: dict[int, int] = {}
        for idx, row in features.sort_values("event_date").iterrows():
            wc_w = _wc_to_weight(row.get("weight_class"))
            if wc_w is None:
                wc_move_a[idx] = 0
                wc_move_b[idx] = 0
                continue

            # Compute flag using prior mode BEFORE updating history
            fa = row.get("fighter_a")
            fb = row.get("fighter_b")
            fa_home = fighter_wc_history[fa].most_common(1)[0][0] if fa and fa in fighter_wc_history else None
            fb_home = fighter_wc_history[fb].most_common(1)[0][0] if fb and fb in fighter_wc_history else None
            wc_move_a[idx] = 1 if (fa_home and wc_w != fa_home) else 0
            wc_move_b[idx] = 1 if (fb_home and wc_w != fb_home) else 0

            # Now update history with this fight
            for col in ["fighter_a", "fighter_b"]:
                fighter = row.get(col)
                if fighter:
                    if fighter not in fighter_wc_history:
                        fighter_wc_history[fighter] = Counter()
                    fighter_wc_history[fighter][wc_w] += 1

        a_moving = [wc_move_a.get(idx, 0) for idx in features.index]
        b_moving = [wc_move_b.get(idx, 0) for idx in features.index]

        features["a_wc_move"] = a_moving
        features["b_wc_move"] = b_moving
        features["diff_wc_move"] = features["a_wc_move"] - features["b_wc_move"]

    bf_module._detect_weight_class_moves = _detect_wc_moves_mode
    try:
        features = build_features(fights_df)
    finally:
        bf_module._detect_weight_class_moves = original_fn

    return features


def build_features_betsapi_challenger(fights_df: pd.DataFrame) -> pd.DataFrame:
    """Build production UFC features, then add saved BetsAPI MMA features."""
    from src.data.betsapi_mma import augment_features_with_betsapi_mma

    features = build_features(fights_df)
    return augment_features_with_betsapi_mma(features, save_artifacts=True)


# ---------------------------------------------------------------------------
# Elo momentum feature (variant 9)
# ---------------------------------------------------------------------------

def add_elo_momentum(features_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add Elo momentum feature: slope of Elo ratings over last N fights.

    A positive slope means the fighter is on an upswing.
    """
    features_df = features_df.copy()
    sorted_df = features_df.sort_values("event_date", kind="mergesort").copy()

    # Build per-fighter Elo history chronologically
    fighter_elo_history: dict[str, list[float]] = {}

    for _, row in sorted_df.iterrows():
        for prefix, col in [("a_", "fighter_a"), ("b_", "fighter_b")]:
            fighter = row.get(col)
            elo = row.get(f"{prefix}elo")
            if not fighter or pd.isna(elo):
                continue
            if fighter not in fighter_elo_history:
                fighter_elo_history[fighter] = []
            fighter_elo_history[fighter].append(elo)

    # Compute slope for each fighter at each fight
    def _elo_slope(history: list[float], idx: int, window: int) -> float:
        """Slope of last `window` Elo values up to index `idx`."""
        start = max(0, idx - window)
        values = history[start:idx]
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope

    # Map back to features
    fighter_elo_idx: dict[str, int] = {}
    a_momentum = []
    b_momentum = []

    for _, row in sorted_df.iterrows():
        for prefix, col, momentum_list in [("a_", "fighter_a", a_momentum),
                                           ("b_", "fighter_b", b_momentum)]:
            fighter = row.get(col)
            if not fighter or fighter not in fighter_elo_history:
                momentum_list.append(0.0)
                continue
            idx = fighter_elo_idx.get(fighter, 0)
            slope = _elo_slope(fighter_elo_history[fighter], idx, window)
            momentum_list.append(slope)
            fighter_elo_idx[fighter] = idx + 1

    sorted_df["a_elo_momentum"] = a_momentum
    sorted_df["b_elo_momentum"] = b_momentum
    sorted_df["diff_elo_momentum"] = sorted_df["a_elo_momentum"] - sorted_df["b_elo_momentum"]

    features_df.loc[sorted_df.index, "a_elo_momentum"] = sorted_df["a_elo_momentum"]
    features_df.loc[sorted_df.index, "b_elo_momentum"] = sorted_df["b_elo_momentum"]
    features_df.loc[sorted_df.index, "diff_elo_momentum"] = sorted_df["diff_elo_momentum"]

    return features_df


# ---------------------------------------------------------------------------
# Variant factory functions
# ---------------------------------------------------------------------------

def baseline() -> VariantConfig:
    """Production baseline — promoted full live contract."""
    baseline_variant = full_live_contract()
    baseline_variant.name = "baseline"
    from src.model.training_spec import full_live_contract_spec

    baseline_variant.description = (
        f"Promoted production contract ({full_live_contract_spec().name})"
    )
    return baseline_variant


def _full_live_contract_feature_cols() -> list[str]:
    from src.model.training_spec import full_live_contract_spec

    return list(full_live_contract_spec().feature_cols)


def blend_b_fix() -> VariantConfig:
    """Bug fix: use independent dynamic blend weight for fighter B."""
    return VariantConfig(
        name="blend_b_fix",
        description="Use dyn_weight_b for fighter B blend instead of 1-blend_a",
        use_independent_blend_b=True,
    )



def wc_mode_fix() -> VariantConfig:
    """Bug fix: use mode instead of first-seen for weight class home detection."""
    return VariantConfig(
        name="wc_mode_fix",
        description="Weight class move detection using Counter.most_common()",
        feature_builder_fn=build_features_wc_mode_fix,
    )


def temporal_sigmoid_calibration() -> VariantConfig:
    """Improvement: temporal holdout + sigmoid calibration."""
    return VariantConfig(
        name="temporal_sigmoid_cal",
        description="Chronological 80/20 holdout with sigmoid calibration (2 params)",
        calibration_method="sigmoid",
        calibration_cv="temporal_holdout",
    )


def timeseries_sigmoid_calibration() -> VariantConfig:
    """Improvement: TimeSeriesSplit + sigmoid calibration."""
    return VariantConfig(
        name="timeseries_sigmoid_cal",
        description="TimeSeriesSplit(n_splits=5) with sigmoid calibration",
        calibration_method="sigmoid",
        calibration_cv="timeseries_5fold",
    )


def missing_indicators() -> VariantConfig:
    """Improvement: add binary indicators for imputed values."""
    return VariantConfig(
        name="missing_indicators",
        description="Add _is_missing binary columns alongside median imputation",
        impute_with_indicators=True,
    )


def ewm_rolling() -> VariantConfig:
    """Improvement: exponentially weighted rolling stats."""
    return VariantConfig(
        name="ewm_rolling",
        description="EWM(halflife=3 fights) instead of flat rolling window of 5",
        use_ewm=True,
        ewm_halflife=3,
        feature_builder_fn=build_features_ewm,
    )


def shap_feature_selection() -> VariantConfig:
    """Improvement: keep only top 40 features by importance."""
    return VariantConfig(
        name="shap_feature_select",
        description="Prune to top 40 features by XGBoost gain importance",
        max_features=40,
    )


def elo_momentum_variant() -> VariantConfig:
    """New feature: Elo momentum (slope over last 5 fights)."""
    return VariantConfig(
        name="elo_momentum",
        description="Add Elo momentum (slope) as new feature",
        add_elo_momentum=True,
    )


def combined_bug_fixes() -> VariantConfig:
    """All bug fixes combined."""
    return VariantConfig(
        name="all_bug_fixes",
        description="blend_b fix + wc mode fix (conviction EV now always on)",
        use_independent_blend_b=True,
        feature_builder_fn=build_features_wc_mode_fix,
    )


# ---------------------------------------------------------------------------
# Strength of Schedule feature
# ---------------------------------------------------------------------------

def add_strength_of_schedule(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add average opponent Elo (Strength of Schedule) as features.
    SoS = rolling average of opponents' Elo at fight time over last 5 fights.
    """
    from src.features.build_features import EloSystem, ELO_K_FACTOR, ELO_INITIAL

    features_df = features_df.sort_values("event_date").copy()

    # Build per-fighter opponent Elo history chronologically
    elo = EloSystem(k=ELO_K_FACTOR, initial=ELO_INITIAL)
    fighter_opp_elos: dict[str, list[float]] = {}

    for _, row in features_df.iterrows():
        fa = row.get("fighter_a", "")
        fb = row.get("fighter_b", "")
        if not fa or not fb:
            continue

        # Record opponent Elo before update
        if fa not in fighter_opp_elos:
            fighter_opp_elos[fa] = []
        if fb not in fighter_opp_elos:
            fighter_opp_elos[fb] = []

        fighter_opp_elos[fa].append(elo.get_rating(fb))
        fighter_opp_elos[fb].append(elo.get_rating(fa))

        # Update Elo
        winner = row.get("winner", None)
        elo.update(fa, fb, winner)

    # Compute rolling SoS (avg opponent Elo of last 5 opponents BEFORE this fight)
    fighter_opp_idx: dict[str, int] = {}
    a_sos = []
    b_sos = []

    for _, row in features_df.iterrows():
        for col, sos_list in [("fighter_a", a_sos), ("fighter_b", b_sos)]:
            fighter = row.get(col, "")
            if fighter and fighter in fighter_opp_elos:
                idx = fighter_opp_idx.get(fighter, 0)
                # Use opponents BEFORE this fight (up to idx, not including idx)
                past_opp_elos = fighter_opp_elos[fighter][:idx]
                if past_opp_elos:
                    recent = past_opp_elos[-5:]  # Last 5
                    sos_list.append(np.mean(recent))
                else:
                    sos_list.append(ELO_INITIAL)
                fighter_opp_idx[fighter] = idx + 1
            else:
                sos_list.append(ELO_INITIAL)

    features_df["a_sos"] = a_sos
    features_df["b_sos"] = b_sos
    features_df["diff_sos"] = features_df["a_sos"] - features_df["b_sos"]

    return features_df


# ---------------------------------------------------------------------------
# Rematch / Head-to-Head feature
# ---------------------------------------------------------------------------

def add_rematch_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rematch detection features:
    - is_rematch: binary flag if fighters have fought before
    - h2h_record_diff: A's wins minus B's wins in prior meetings
    """
    features_df = features_df.sort_values("event_date").copy()

    # Build H2H history chronologically
    h2h: dict[tuple[str, str], list[str]] = {}  # (sorted pair) -> list of winners

    is_rematch = []
    h2h_diff = []

    for _, row in features_df.iterrows():
        fa = row.get("fighter_a", "")
        fb = row.get("fighter_b", "")
        winner = row.get("winner", "")
        pair = tuple(sorted([fa, fb]))

        prior_fights = h2h.get(pair, [])
        if prior_fights:
            is_rematch.append(1)
            a_prior_wins = sum(1 for w in prior_fights if w == fa)
            b_prior_wins = sum(1 for w in prior_fights if w == fb)
            h2h_diff.append(a_prior_wins - b_prior_wins)
        else:
            is_rematch.append(0)
            h2h_diff.append(0)

        # Record this fight's result
        if pair not in h2h:
            h2h[pair] = []
        if winner:
            h2h[pair].append(winner)

    features_df["is_rematch"] = is_rematch
    features_df["h2h_record_diff"] = h2h_diff

    return features_df


def apply_variant_feature_transforms(
    features_df: pd.DataFrame,
    variant: VariantConfig,
) -> pd.DataFrame:
    """Apply post-build feature transforms for a variant."""
    variant_features = features_df.copy()

    if variant.name == "elo_momentum" or variant.add_elo_momentum:
        variant_features = add_elo_momentum(variant_features)
    if variant.add_strength_of_schedule:
        variant_features = add_strength_of_schedule(variant_features)
    if variant.add_rematch_features:
        variant_features = add_rematch_features(variant_features)

    return variant_features


# ---------------------------------------------------------------------------
# Phase 2 variant factories
# ---------------------------------------------------------------------------

def lower_edge_025() -> VariantConfig:
    """Test lower edge threshold at 2.5%."""
    return VariantConfig(
        name="lower_edge_025",
        description="MIN_EDGE_THRESHOLD = 0.025 (vs 0.03 production)",
        min_edge=0.025,
    )


def lower_edge_020() -> VariantConfig:
    """Test lower edge threshold at 2%."""
    return VariantConfig(
        name="lower_edge_020",
        description="MIN_EDGE_THRESHOLD = 0.02 (vs 0.03 production)",
        min_edge=0.02,
    )


def higher_blend_035() -> VariantConfig:
    """Test higher model blend weight at 35%."""
    return VariantConfig(
        name="higher_blend_035",
        description="BLEND_WEIGHT = 0.35 (vs 0.30 production)",
        blend_weight=0.35,
    )


def higher_blend_040() -> VariantConfig:
    """Test higher model blend weight at 40%."""
    return VariantConfig(
        name="higher_blend_040",
        description="BLEND_WEIGHT = 0.40 (vs 0.30 production)",
        blend_weight=0.40,
    )


def shorter_decay() -> VariantConfig:
    """Test 1-year time decay half-life instead of 2 years."""
    return VariantConfig(
        name="shorter_decay",
        description="TIME_DECAY_HALF_LIFE = 365 days (vs 730 production)",
        time_decay_half_life=365,
    )


def higher_odds_noise_06() -> VariantConfig:
    """Test higher odds noise at 6%."""
    return VariantConfig(
        name="higher_odds_noise_06",
        description="ODDS_NOISE_STD = 0.06 (vs 0.04 production)",
        odds_noise_std=0.06,
    )


def higher_odds_noise_08() -> VariantConfig:
    """Test higher odds noise at 8%."""
    return VariantConfig(
        name="higher_odds_noise_08",
        description="ODDS_NOISE_STD = 0.08 (vs 0.04 production)",
        odds_noise_std=0.08,
    )


def strength_of_schedule_variant() -> VariantConfig:
    """New feature: strength of schedule (avg opponent Elo)."""
    return VariantConfig(
        name="strength_of_schedule",
        description="Add rolling avg opponent Elo (SoS) feature",
        add_strength_of_schedule=True,
    )


def rematch_variant() -> VariantConfig:
    """New feature: rematch detection and H2H record."""
    return VariantConfig(
        name="rematch_features",
        description="Add is_rematch flag and H2H record differential",
        add_rematch_features=True,
    )


def rematch_native_nan() -> VariantConfig:
    """Rematch features with native NaN handling (no median imputation)."""
    v = VariantConfig(
        name="rematch_native_nan",
        description="Rematch features + native NaN (XGBoost handles missing values)",
        add_rematch_features=True,
        calibration_method="isotonic",
        calibration_cv="timeseries_5fold",
    )
    v._native_nan = True  # type: ignore[attr-defined]
    return v


def full_live_contract() -> VariantConfig:
    """
    Phase 2: Full live contract — ALL features available at both training and inference.

    Includes: rematch, elo momentum, strength of schedule, line movement.
    Rankings and method odds are already in features.csv.
    Native NaN for XGBoost (no median imputation).
    """
    from src.model.training_spec import full_live_contract_spec

    spec = full_live_contract_spec()
    v = VariantConfig(
        name="full_live_contract",
        description=(
            f"Promoted contract from {spec.name}: rematch + Elo momentum + SoS "
            "with native NaN. Line movement stays outside the training "
            "contract until historical coverage is honest."
        ),
        feature_cols=_full_live_contract_feature_cols(),
        add_elo_momentum=spec.add_elo_momentum,
        add_strength_of_schedule=spec.add_strength_of_schedule,
        add_rematch_features=spec.add_rematch_features,
        add_line_movement=spec.add_line_movement,
        calibration_method="isotonic",
        calibration_cv="timeseries_5fold",
    )
    v._native_nan = True  # type: ignore[attr-defined]
    return v


def xgb_less_reg() -> VariantConfig:
    """XGBoost with less regularization."""
    params = _production_xgb_params()
    params.update({"max_depth": 4, "gamma": 0.0, "min_child_weight": 3})
    return VariantConfig(
        name="xgb_less_reg",
        description="XGB: depth=4, gamma=0, min_child_weight=3",
        xgb_params=params,
    )


def xgb_deeper() -> VariantConfig:
    """XGBoost with deeper trees and slower learning."""
    params = _production_xgb_params()
    params.update({"max_depth": 6, "learning_rate": 0.03, "n_estimators": 500})
    return VariantConfig(
        name="xgb_deeper",
        description="XGB: depth=6, lr=0.03, n_est=500",
        xgb_params=params,
    )


def combined_best() -> VariantConfig:
    """Bug fixes + sigmoid calibration + missing indicators."""
    return VariantConfig(
        name="combined_best",
        description="All bug fixes + temporal sigmoid cal + missing indicators",
        use_independent_blend_b=True,
        calibration_method="sigmoid",
        calibration_cv="temporal_holdout",
        impute_with_indicators=True,
        feature_builder_fn=build_features_wc_mode_fix,
    )


def combined_v2() -> VariantConfig:
    """Stack best Phase 2 improvements: lower edge + higher blend."""
    return VariantConfig(
        name="combined_v2",
        description="min_edge=0.025 + blend_weight=0.35",
        min_edge=0.025,
        blend_weight=0.35,
    )


def combined_v3() -> VariantConfig:
    """Stack top 3: lower edge + higher blend + higher odds noise."""
    return VariantConfig(
        name="combined_v3",
        description="min_edge=0.025 + blend=0.35 + odds_noise=0.06",
        min_edge=0.025,
        blend_weight=0.35,
        odds_noise_std=0.06,
    )


def betsapi_challenger() -> VariantConfig:
    """Challenger variant that augments production features with BetsAPI MMA data."""
    return VariantConfig(
        name="betsapi_challenger",
        description="Production feature set plus BetsAPI MMA market and integrity features",
        feature_builder_fn=build_features_betsapi_challenger,
    )


def github_production_replica() -> VariantConfig:
    """Replica of the live GitHub model for head-to-head evaluation."""
    return VariantConfig(
        name="github_production_replica",
        description=(
            "Replicates nfgems/ufc-betting-bot: train.py XGB params, "
            "isotonic/timeseries_5fold, top-40 features, blend=0.30"
        ),
        xgb_params={
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
            "use_label_encoder": False,
        },
        calibration_method="isotonic",
        calibration_cv="timeseries_5fold",
        max_features=40,
        blend_weight=0.30,
    )


def rematch_vs_github() -> VariantConfig:
    """Rematch features with the GitHub model's architecture and calibration."""
    return VariantConfig(
        name="rematch_vs_github",
        description=(
            "rematch_features + GitHub XGB params + isotonic/timeseries_5fold "
            "+ top-40 features + blend=0.30"
        ),
        xgb_params={
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
            "use_label_encoder": False,
        },
        calibration_method="isotonic",
        calibration_cv="timeseries_5fold",
        max_features=40,
        blend_weight=0.30,
        add_rematch_features=True,
    )


def rematch_sigmoid() -> VariantConfig:
    """Rematch features with temporal sigmoid calibration."""
    return VariantConfig(
        name="rematch_sigmoid",
        description=(
            "Rematch/H2H features + temporal-holdout sigmoid calibration "
            "(year-stable calibration experiment)"
        ),
        add_rematch_features=True,
        calibration_method="sigmoid",
        calibration_cv="temporal_holdout",
    )


# Registry of all available variants
ALL_VARIANTS = {
    # Phase 1 variants
    "baseline": baseline,
    "blend_b_fix": blend_b_fix,
    "wc_mode_fix": wc_mode_fix,
    "temporal_sigmoid_cal": temporal_sigmoid_calibration,
    "timeseries_sigmoid_cal": timeseries_sigmoid_calibration,
    "missing_indicators": missing_indicators,
    "ewm_rolling": ewm_rolling,
    "shap_feature_select": shap_feature_selection,
    "elo_momentum": elo_momentum_variant,
    "all_bug_fixes": combined_bug_fixes,
    "combined_best": combined_best,
    # Phase 2 variants
    "lower_edge_025": lower_edge_025,
    "lower_edge_020": lower_edge_020,
    "higher_blend_035": higher_blend_035,
    "higher_blend_040": higher_blend_040,
    "shorter_decay": shorter_decay,
    "higher_odds_noise_06": higher_odds_noise_06,
    "higher_odds_noise_08": higher_odds_noise_08,
    "strength_of_schedule": strength_of_schedule_variant,
    "rematch_features": rematch_variant,
    "rematch_native_nan": rematch_native_nan,
    "full_live_contract": full_live_contract,
    "xgb_less_reg": xgb_less_reg,
    "xgb_deeper": xgb_deeper,
    "combined_v2": combined_v2,
    "combined_v3": combined_v3,
    "betsapi_challenger": betsapi_challenger,
    "rematch_sigmoid": rematch_sigmoid,
    "github_production_replica": github_production_replica,
    "rematch_vs_github": rematch_vs_github,
}
