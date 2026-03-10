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

from src.features.build_features import EloSystem, ELO_K_FACTOR, ELO_INITIAL

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
                features_df[slpm_col].fillna(0) + features_df[sapm_col].fillna(0)
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
            ctrl = features_df[ctrl_col].clip(lower=1).fillna(60)
            features_df[f"{prefix}ctrl_efficiency"] = (
                features_df[sig_str_col].fillna(0) / ctrl
            )

    if "a_ctrl_efficiency" in features_df.columns and "b_ctrl_efficiency" in features_df.columns:
        features_df["diff_ctrl_efficiency"] = (
            features_df["a_ctrl_efficiency"] - features_df["b_ctrl_efficiency"]
        )
        logger.info("Added cage time efficiency features")

    return features_df


def add_quality_adjusted_stats(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add opponent-quality-adjusted win rate.

    win_pct * (avg_opponent_elo / 1500) gives more credit to fighters
    who beat strong opponents and less to those who pad records.
    """
    features_df = features_df.sort_values("event_date").copy()

    # Build per-fighter opponent Elo history
    elo = EloSystem(k=ELO_K_FACTOR, initial=ELO_INITIAL)
    fighter_opp_elos: dict[str, list[float]] = {}

    for _, row in features_df.iterrows():
        fa = row.get("fighter_a", "")
        fb = row.get("fighter_b", "")
        if not fa or not fb:
            continue

        if fa not in fighter_opp_elos:
            fighter_opp_elos[fa] = []
        if fb not in fighter_opp_elos:
            fighter_opp_elos[fb] = []

        fighter_opp_elos[fa].append(elo.get_rating(fb))
        fighter_opp_elos[fb].append(elo.get_rating(fa))

        winner = row.get("winner", None)
        elo.update(fa, fb, winner)

    # Compute quality-adjusted win rate for each row
    fighter_opp_idx: dict[str, int] = {}
    a_adj_win_pct = []
    b_adj_win_pct = []

    for _, row in features_df.iterrows():
        for col, adj_list in [("fighter_a", a_adj_win_pct), ("fighter_b", b_adj_win_pct)]:
            fighter = row.get(col, "")
            prefix = "a_" if col == "fighter_a" else "b_"
            win_pct = row.get(f"{prefix}win_pct", 0.5)
            if pd.isna(win_pct):
                win_pct = 0.5

            if fighter and fighter in fighter_opp_elos:
                idx = fighter_opp_idx.get(fighter, 0)
                past_opp_elos = fighter_opp_elos[fighter][:idx]
                if past_opp_elos:
                    avg_opp_elo = np.mean(past_opp_elos[-5:])
                    adj_list.append(win_pct * (avg_opp_elo / ELO_INITIAL))
                else:
                    adj_list.append(win_pct)
                fighter_opp_idx[fighter] = idx + 1
            else:
                adj_list.append(win_pct)

    features_df["a_adj_win_pct"] = a_adj_win_pct
    features_df["b_adj_win_pct"] = b_adj_win_pct
    features_df["diff_adj_win_pct"] = features_df["a_adj_win_pct"] - features_df["b_adj_win_pct"]

    logger.info("Added quality-adjusted win rate features")
    return features_df


def add_experimental_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all experimental features to the feature DataFrame.

    This is the main entry point. Does NOT modify build_features.py.
    """
    logger.info(f"Adding experimental features to {len(features_df)} fights...")

    features_df = add_fight_pace(features_df)
    features_df = add_cage_time_efficiency(features_df)
    features_df = add_quality_adjusted_stats(features_df)

    new_features = [
        c for c in features_df.columns
        if c in [
            "a_fight_pace", "b_fight_pace", "diff_fight_pace",
            "a_ctrl_efficiency", "b_ctrl_efficiency", "diff_ctrl_efficiency",
            "a_adj_win_pct", "b_adj_win_pct", "diff_adj_win_pct",
        ]
    ]
    logger.info(f"Added {len(new_features)} experimental feature columns: {new_features}")

    return features_df


# Names of all experimental feature columns.
# get_feature_columns() only picks up diff_ prefixed columns automatically.
# The a_/b_ columns need to be explicitly added when using experimental features.
EXPERIMENTAL_FEATURE_NAMES = [
    "a_fight_pace", "b_fight_pace", "diff_fight_pace",
    "a_ctrl_efficiency", "b_ctrl_efficiency", "diff_ctrl_efficiency",
    "a_adj_win_pct", "b_adj_win_pct", "diff_adj_win_pct",
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
