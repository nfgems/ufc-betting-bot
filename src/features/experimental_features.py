"""
Experimental features — isolated from production build_features.py.

Adds new derived features to an existing feature DataFrame without
modifying the production feature pipeline.

Usage:
    from src.features.experimental_features import add_experimental_features
    features_df = add_experimental_features(features_df)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_fight_pace(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fight pace features: significant strikes per minute of fight time.

    Higher pace fighters tend to push the action and force finishes.
    """
    features_df = features_df.copy()

    for prefix in ["a_", "b_"]:
        slpm_col = f"{prefix}roll_slpm"
        sapm_col = f"{prefix}roll_sapm"

        if slpm_col in features_df.columns and sapm_col in features_df.columns:
            # Total striking output (landed + absorbed = proxy for fight activity)
            features_df[f"{prefix}fight_pace"] = (
                features_df[slpm_col] + features_df[sapm_col]
            )

    if "a_fight_pace" in features_df.columns and "b_fight_pace" in features_df.columns:
        features_df["diff_fight_pace"] = (
            features_df["a_fight_pace"] - features_df["b_fight_pace"]
        )
        logger.info("Added fight pace features")

    return features_df


def add_cage_time_efficiency(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cage time efficiency: striking output relative to control time.

    Fighters who can land strikes without needing ground control are
    more versatile and harder to game-plan against.
    """
    features_df = features_df.copy()

    for prefix in ["a_", "b_"]:
        sig_str_col = f"{prefix}roll_sig_str_landed"
        ctrl_col = f"{prefix}roll_ctrl_seconds"

        if sig_str_col in features_df.columns and ctrl_col in features_df.columns:
            ctrl = features_df[ctrl_col].clip(lower=1)
            features_df[f"{prefix}ctrl_efficiency"] = (
                features_df[sig_str_col] / ctrl
            )

    if "a_ctrl_efficiency" in features_df.columns and "b_ctrl_efficiency" in features_df.columns:
        features_df["diff_ctrl_efficiency"] = (
            features_df["a_ctrl_efficiency"] - features_df["b_ctrl_efficiency"]
        )
        logger.info("Added cage time efficiency features")

    return features_df


def add_age_nonlinearity(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add age nonlinearity features.

    age_over_35: binary flag — fighters 35+ are past their athletic prime, performance drops.
    age_under_25: binary flag — UFC tends to give young prospects favorable matchups.
    age_squared: captures the U-shaped relationship between age and performance.
    """
    features_df = features_df.copy()

    for prefix in ["a_", "b_"]:
        age_col = f"{prefix}age"
        if age_col in features_df.columns:
            age = features_df[age_col]
            features_df[f"{prefix}age_over_35"] = np.where(age.isna(), np.nan, (age > 35).astype(float))
            features_df[f"{prefix}age_under_25"] = np.where(age.isna(), np.nan, (age < 25).astype(float))
            features_df[f"{prefix}age_squared"] = age ** 2

    for feat in ["age_over_35", "age_under_25", "age_squared"]:
        a_col, b_col = f"a_{feat}", f"b_{feat}"
        if a_col in features_df.columns and b_col in features_df.columns:
            features_df[f"diff_{feat}"] = features_df[a_col] - features_df[b_col]

    logger.info("Added age nonlinearity features (age_over_35, age_under_25, age_squared)")
    return features_df


def add_ko_absorption(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    KO absorption rate: rolling average of knockdowns absorbed per fight.

    Fighters who get dropped frequently are at higher risk of KO loss.
    Uses the opponent KD stat (opp_kd) which is already rolled.
    """
    features_df = features_df.copy()

    for prefix in ["a_", "b_"]:
        opp_kd_col = f"{prefix}roll_opp_kd"
        if opp_kd_col in features_df.columns:
            features_df[f"{prefix}ko_absorption"] = features_df[opp_kd_col]

    if "a_ko_absorption" in features_df.columns and "b_ko_absorption" in features_df.columns:
        features_df["diff_ko_absorption"] = (
            features_df["a_ko_absorption"] - features_df["b_ko_absorption"]
        )
        logger.info("Added KO absorption features")

    return features_df


def add_strikes_avoided(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Strikes avoided percentage: what fraction of opponent strikes miss.

    Derived from rolling str_def (UFCStats defense percentage).
    This is a cleaner signal than raw str_def for head movement / elusiveness.
    """
    features_df = features_df.copy()

    for prefix in ["a_", "b_"]:
        sig_landed = f"{prefix}roll_opp_sig_str_landed"
        sig_attempted = f"{prefix}roll_opp_sig_str_attempted"
        if sig_landed in features_df.columns and sig_attempted in features_df.columns:
            attempted = features_df[sig_attempted].replace(0, np.nan)
            features_df[f"{prefix}strikes_avoided_pct"] = (
                1.0 - features_df[sig_landed] / attempted
            )

    if "a_strikes_avoided_pct" in features_df.columns and "b_strikes_avoided_pct" in features_df.columns:
        features_df["diff_strikes_avoided_pct"] = (
            features_df["a_strikes_avoided_pct"] - features_df["b_strikes_avoided_pct"]
        )
        logger.info("Added strikes avoided features")

    return features_df


def add_pace_mismatch(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pace mismatch: absolute difference in fight pace between fighters.

    Large pace mismatches can indicate style clashes where one fighter
    is comfortable at a rhythm the other is not.
    """
    features_df = features_df.copy()

    if "a_fight_pace" in features_df.columns and "b_fight_pace" in features_df.columns:
        features_df["pace_mismatch"] = (
            features_df["a_fight_pace"] - features_df["b_fight_pace"]
        ).abs()
        logger.info("Added pace mismatch feature")

    return features_df


def add_experimental_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all experimental features to the feature DataFrame.

    This is the main entry point. Does NOT modify build_features.py.
    """
    logger.info(f"Adding experimental features to {len(features_df)} fights...")

    features_df = add_fight_pace(features_df)
    features_df = add_cage_time_efficiency(features_df)
    features_df = add_age_nonlinearity(features_df)
    features_df = add_ko_absorption(features_df)
    features_df = add_strikes_avoided(features_df)
    # opp_strength is now computed in build_features.py as a proper
    # strength-of-schedule (rolling avg of past opponents' win rates)
    features_df = add_pace_mismatch(features_df)

    added = [c for c in features_df.columns if c in EXPERIMENTAL_FEATURE_NAMES]
    logger.info(f"Added {len(added)} experimental feature columns")

    return features_df


# Names of all experimental feature columns.
# get_feature_columns() only picks up diff_ prefixed columns automatically.
# The a_/b_ columns need to be explicitly added when using experimental features.
EXPERIMENTAL_FEATURE_NAMES = [
    "a_fight_pace", "b_fight_pace", "diff_fight_pace",
    "a_ctrl_efficiency", "b_ctrl_efficiency", "diff_ctrl_efficiency",
    # Age nonlinearity
    "a_age_over_35", "b_age_over_35", "diff_age_over_35",
    "a_age_under_25", "b_age_under_25", "diff_age_under_25",
    "a_age_squared", "b_age_squared", "diff_age_squared",
    # KO absorption
    "a_ko_absorption", "b_ko_absorption", "diff_ko_absorption",
    # Strikes avoided
    "a_strikes_avoided_pct", "b_strikes_avoided_pct", "diff_strikes_avoided_pct",
    # opp_strength moved to build_features.py (proper SOS computation)
    # Pace mismatch
    "pace_mismatch",
]


def get_experimental_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """
    Get feature columns including experimental features.

    Wraps get_feature_columns() and adds experimental columns that
    the production function wouldn't pick up (a_/b_ prefixed ones).
    """
    from src.features.build_features import get_feature_columns

    base_cols = get_feature_columns(features_df)

    # Add a_/b_ experimental columns that get_feature_columns() misses
    extra = [c for c in EXPERIMENTAL_FEATURE_NAMES if c in features_df.columns and c not in base_cols]
    if extra:
        logger.info(f"Adding {len(extra)} experimental feature columns: {extra}")

    return base_cols + extra
