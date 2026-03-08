"""
Feature engineering for UFC fight prediction.

Computes rolling fighter stats, Elo ratings, and fighter differentials
from historical fight data. All features are computed using only data
available BEFORE each fight (no data leakage).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import ROLLING_WINDOW, ELO_INITIAL, ELO_K_FACTOR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Elo rating system
# ---------------------------------------------------------------------------

class EloSystem:
    """Elo rating tracker for fighters."""

    def __init__(self, k: float = ELO_K_FACTOR, initial: float = ELO_INITIAL):
        self.k = k
        self.initial = initial
        self.ratings: dict[str, float] = {}
        self.fight_counts: dict[str, int] = {}

    def get_rating(self, fighter: str) -> float:
        return self.ratings.get(fighter, self.initial)

    def get_fight_count(self, fighter: str) -> int:
        return self.fight_counts.get(fighter, 0)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(self, fighter_a: str, fighter_b: str, winner: Optional[str]) -> tuple[float, float]:
        """
        Update ratings after a fight. Returns (new_rating_a, new_rating_b).
        winner=None means a draw.
        """
        ra = self.get_rating(fighter_a)
        rb = self.get_rating(fighter_b)

        ea = self.expected_score(ra, rb)
        eb = 1.0 - ea

        if winner == fighter_a:
            sa, sb = 1.0, 0.0
        elif winner == fighter_b:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        # Dynamic K: higher for fighters with fewer fights
        ka = self.k * (1.5 if self.get_fight_count(fighter_a) < 5 else 1.0)
        kb = self.k * (1.5 if self.get_fight_count(fighter_b) < 5 else 1.0)

        self.ratings[fighter_a] = ra + ka * (sa - ea)
        self.ratings[fighter_b] = rb + kb * (sb - eb)
        self.fight_counts[fighter_a] = self.get_fight_count(fighter_a) + 1
        self.fight_counts[fighter_b] = self.get_fight_count(fighter_b) + 1

        return self.ratings[fighter_a], self.ratings[fighter_b]


# ---------------------------------------------------------------------------
# Rolling stats computation
# ---------------------------------------------------------------------------

STAT_COLUMNS = [
    "slpm", "sapm", "str_acc", "str_def",
    "td_avg", "td_acc", "td_def", "sub_avg",
    "sig_str_landed", "sig_str_attempted",
    "td_landed", "td_attempted",
    "kd", "sub_att", "rev", "ctrl_seconds",
]


def _compute_per_fight_stats(fights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract per-fight performance for each fighter from the fights DataFrame.
    Returns a long-format DataFrame with one row per fighter per fight.
    """
    records = []

    for _, row in fights_df.iterrows():
        date = row.get("event_date")
        weight_class = row.get("weight_class", "")
        winner = row.get("winner", "")

        for prefix, fighter_col in [("a_", "fighter_a"), ("b_", "fighter_b")]:
            fighter = row.get(fighter_col)
            if pd.isna(fighter) or not fighter:
                continue

            opp_prefix = "b_" if prefix == "a_" else "a_"
            opp_col = "fighter_b" if prefix == "a_" else "fighter_a"

            record = {
                "fighter": fighter,
                "opponent": row.get(opp_col, ""),
                "event_date": date,
                "weight_class": weight_class,
                "won": 1 if winner == fighter else 0,
            }

            # Gather stats for this fighter in this fight
            for stat in STAT_COLUMNS:
                col = f"{prefix}{stat}"
                opp_col_stat = f"{opp_prefix}{stat}"
                record[stat] = row.get(col, np.nan)
                record[f"opp_{stat}"] = row.get(opp_col_stat, np.nan)

            records.append(record)

    return pd.DataFrame(records)


def _compute_rolling_stats(
    fighter_fights: pd.DataFrame, window: int = ROLLING_WINDOW
) -> pd.DataFrame:
    """
    Compute rolling averages for a single fighter's fight history.
    Uses expanding window for fighters with fewer fights than window size.
    Stats are computed from all fights BEFORE the current one (shift(1)).
    """
    fighter_fights = fighter_fights.sort_values("event_date").copy()

    stats_to_roll = STAT_COLUMNS + [f"opp_{s}" for s in STAT_COLUMNS] + ["won"]

    for stat in stats_to_roll:
        if stat in fighter_fights.columns:
            # shift(1) ensures we only use data from BEFORE this fight
            shifted = fighter_fights[stat].shift(1)
            fighter_fights[f"roll_{stat}"] = (
                shifted.rolling(window=window, min_periods=1).mean()
            )

    # Win streak (consecutive wins leading into this fight)
    wins = fighter_fights["won"].shift(1).fillna(0)
    streaks = []
    current_streak = 0
    for w in wins:
        if w == 1:
            current_streak += 1
        else:
            current_streak = 0
        streaks.append(current_streak)
    fighter_fights["win_streak"] = streaks

    # Fights count (experience going into this fight)
    fighter_fights["num_fights"] = range(len(fighter_fights))

    # Days since last fight
    dates = fighter_fights["event_date"]
    fighter_fights["days_since_last_fight"] = dates.diff().dt.days.fillna(365)

    return fighter_fights


# ---------------------------------------------------------------------------
# Main feature building
# ---------------------------------------------------------------------------

def build_features(fights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from a fights DataFrame.

    For each fight, computes:
    - Rolling averages (last N fights) for both fighters
    - Elo ratings for both fighters
    - Differentials (fighter_a - fighter_b) for all stats
    - Win streak, experience, days since last fight

    Returns a DataFrame ready for model training with no data leakage.
    """
    logger.info(f"Building features for {len(fights_df)} fights")

    # Ensure sorted by date
    fights_df = fights_df.sort_values("event_date").reset_index(drop=True)

    # Step 1: Compute per-fight stats in long format
    per_fight = _compute_per_fight_stats(fights_df)
    logger.info(f"Extracted {len(per_fight)} fighter-fight records")

    # Step 2: Compute rolling stats per fighter
    all_rolling = []
    for fighter, group in per_fight.groupby("fighter"):
        rolled = _compute_rolling_stats(group)
        all_rolling.append(rolled)

    rolling_df = pd.concat(all_rolling, ignore_index=True)

    # Step 3: Compute Elo ratings (process fights chronologically)
    elo = EloSystem()
    elo_a_list = []
    elo_b_list = []

    for _, row in fights_df.iterrows():
        fa = row.get("fighter_a", "")
        fb = row.get("fighter_b", "")
        winner = row.get("winner", None)

        # Record pre-fight Elo ratings
        elo_a_list.append(elo.get_rating(fa))
        elo_b_list.append(elo.get_rating(fb))

        # Update after fight
        if fa and fb:
            elo.update(fa, fb, winner)

    fights_df = fights_df.copy()
    fights_df["a_elo"] = elo_a_list
    fights_df["b_elo"] = elo_b_list

    # Step 4: Merge rolling stats back into fights DataFrame
    # For fighter_a
    a_rolling = rolling_df.rename(
        columns={c: f"a_roll_{c.replace('roll_', '')}" if c.startswith("roll_") else f"a_{c}"
                 for c in rolling_df.columns
                 if c not in ["fighter", "opponent", "event_date", "weight_class", "won"]}
    )
    # Keep key columns for merge
    a_merge_cols = ["fighter", "event_date"] + [
        c for c in a_rolling.columns
        if c.startswith("a_roll_") or c in ["win_streak", "num_fights", "days_since_last_fight"]
    ]
    # Rename non-prefixed columns for fighter A
    rename_map = {}
    for c in ["win_streak", "num_fights", "days_since_last_fight"]:
        if c in a_rolling.columns:
            rename_map[c] = f"a_{c}"
    a_rolling = a_rolling.rename(columns=rename_map)
    a_merge_cols = ["fighter", "event_date"] + [
        c for c in a_rolling.columns if c.startswith("a_")
    ]
    a_rolling_deduped = a_rolling[
        [c for c in a_merge_cols if c in a_rolling.columns]
    ].drop_duplicates(subset=["fighter", "event_date"], keep="last")

    # For fighter_b
    b_rolling = rolling_df.copy()
    rename_b = {}
    for c in rolling_df.columns:
        if c.startswith("roll_"):
            rename_b[c] = f"b_roll_{c.replace('roll_', '')}"
        elif c in ["win_streak", "num_fights", "days_since_last_fight"]:
            rename_b[c] = f"b_{c}"
    b_rolling = b_rolling.rename(columns=rename_b)
    b_merge_cols = ["fighter", "event_date"] + [
        c for c in b_rolling.columns if c.startswith("b_")
    ]
    b_rolling_deduped = b_rolling[
        [c for c in b_merge_cols if c in b_rolling.columns]
    ].drop_duplicates(subset=["fighter", "event_date"], keep="last")

    # Merge fighter A rolling stats
    features = fights_df.merge(
        a_rolling_deduped,
        left_on=["fighter_a", "event_date"],
        right_on=["fighter", "event_date"],
        how="left",
    ).drop(columns=["fighter"], errors="ignore")

    # Merge fighter B rolling stats
    features = features.merge(
        b_rolling_deduped,
        left_on=["fighter_b", "event_date"],
        right_on=["fighter", "event_date"],
        how="left",
    ).drop(columns=["fighter"], errors="ignore")

    # Step 5: Compute differentials
    diff_stats = [
        "roll_slpm", "roll_sapm", "roll_str_acc", "roll_str_def",
        "roll_td_avg", "roll_td_acc", "roll_td_def", "roll_sub_avg",
        "roll_sig_str_landed", "roll_td_landed", "roll_kd",
        "roll_won", "elo", "win_streak", "num_fights", "days_since_last_fight",
    ]

    for stat in diff_stats:
        a_col = f"a_{stat}"
        b_col = f"b_{stat}"
        if a_col in features.columns and b_col in features.columns:
            features[f"diff_{stat}"] = features[a_col] - features[b_col]

    # Physical differentials from original data
    for attr in ["height", "reach", "weight", "age"]:
        a_col = f"a_{attr}"
        b_col = f"b_{attr}"
        if a_col in features.columns and b_col in features.columns:
            features[f"diff_{attr}"] = features[a_col] - features[b_col]

    # Strike differential (computed stat)
    if "a_roll_slpm" in features.columns and "a_roll_sapm" in features.columns:
        features["a_strike_diff"] = features["a_roll_slpm"] - features["a_roll_sapm"]
        features["b_strike_diff"] = features["b_roll_slpm"] - features["b_roll_sapm"]
        features["diff_strike_diff"] = features["a_strike_diff"] - features["b_strike_diff"]

    # Stance encoding
    if "a_stance" in features.columns and "b_stance" in features.columns:
        stance_map = {"Orthodox": 0, "Southpaw": 1, "Switch": 2}
        features["a_stance_enc"] = features["a_stance"].map(stance_map).fillna(-1)
        features["b_stance_enc"] = features["b_stance"].map(stance_map).fillna(-1)
        features["same_stance"] = (features["a_stance_enc"] == features["b_stance_enc"]).astype(int)

    # --- New derived features from Kaggle data ---

    # Finish method rates (KO%, Sub%, Dec% of total wins)
    for prefix in ["a_", "b_"]:
        wins_col = f"{prefix}wins"
        if wins_col in features.columns:
            total_wins = features[wins_col].replace(0, np.nan)
            for method, method_col in [("ko_rate", "wins_ko"), ("sub_rate", "wins_sub"),
                                        ("dec_rate", "wins_dec")]:
                src_col = f"{prefix}{method_col}"
                if src_col in features.columns:
                    features[f"{prefix}{method}"] = features[src_col] / total_wins
                    features[f"{prefix}{method}"] = features[f"{prefix}{method}"].fillna(0)

    # Finish rate differentials
    for method in ["ko_rate", "sub_rate", "dec_rate"]:
        a_col = f"a_{method}"
        b_col = f"b_{method}"
        if a_col in features.columns and b_col in features.columns:
            features[f"diff_{method}"] = features[a_col] - features[b_col]

    # Historical odds → implied probability (market wisdom feature)
    for prefix in ["a_", "b_"]:
        odds_col = f"{prefix}odds"
        if odds_col in features.columns:
            odds = features[odds_col].copy()
            # Convert American odds to implied probability
            pos_mask = odds > 0
            neg_mask = odds < 0
            prob = pd.Series(np.nan, index=features.index)
            prob[pos_mask] = 100 / (odds[pos_mask] + 100)
            prob[neg_mask] = (-odds[neg_mask]) / (-odds[neg_mask] + 100)
            features[f"{prefix}implied_prob"] = prob

    if "a_implied_prob" in features.columns and "b_implied_prob" in features.columns:
        features["diff_implied_prob"] = features["a_implied_prob"] - features["b_implied_prob"]

    # Title bout flag (binary)
    if "title_bout" in features.columns:
        features["is_title_bout"] = features["title_bout"].fillna(0).astype(int)

    # Number of rounds (3 vs 5)
    if "num_rounds" in features.columns:
        features["num_rounds_feat"] = features["num_rounds"].fillna(3).astype(float)

    # Empty arena (COVID indicator)
    if "empty_arena" in features.columns:
        features["is_empty_arena"] = features["empty_arena"].fillna(0).astype(int)

    # Ranking features
    for prefix in ["a_", "b_"]:
        wc_rank_col = f"{prefix}wc_rank"
        if wc_rank_col in features.columns:
            # NaN means unranked — fill with a high number (16)
            features[f"{prefix}wc_rank_feat"] = features[wc_rank_col].fillna(16).astype(float)
        pfp_col = f"{prefix}pfp_rank"
        if pfp_col in features.columns:
            features[f"{prefix}pfp_rank_feat"] = features[pfp_col].fillna(16).astype(float)

    if "a_wc_rank_feat" in features.columns and "b_wc_rank_feat" in features.columns:
        features["diff_wc_rank"] = features["a_wc_rank_feat"] - features["b_wc_rank_feat"]
    if "a_pfp_rank_feat" in features.columns and "b_pfp_rank_feat" in features.columns:
        features["diff_pfp_rank"] = features["a_pfp_rank_feat"] - features["b_pfp_rank_feat"]

    # Lose streak and longest win streak differentials
    for stat in ["lose_streak", "longest_win_streak", "total_rounds", "title_bouts", "draws"]:
        a_col = f"a_{stat}"
        b_col = f"b_{stat}"
        if a_col in features.columns and b_col in features.columns:
            features[f"diff_{stat}"] = features[a_col] - features[b_col]

    # Method-specific odds (KO odds, Sub odds, Dec odds)
    for prefix in ["a_", "b_"]:
        for odds_type in ["ko_odds", "sub_odds", "dec_odds"]:
            col = f"{prefix}{odds_type}"
            if col in features.columns:
                odds = features[col].copy()
                pos_mask = odds > 0
                neg_mask = odds < 0
                prob = pd.Series(np.nan, index=features.index)
                prob[pos_mask] = 100 / (odds[pos_mask] + 100)
                prob[neg_mask] = (-odds[neg_mask]) / (-odds[neg_mask] + 100)
                features[f"{prefix}{odds_type}_prob"] = prob

    # Win record ratio (wins / (wins + losses)) — overall quality
    for prefix in ["a_", "b_"]:
        w_col = f"{prefix}wins"
        l_col = f"{prefix}losses"
        if w_col in features.columns and l_col in features.columns:
            total = features[w_col] + features[l_col]
            features[f"{prefix}win_pct"] = (features[w_col] / total.replace(0, np.nan)).fillna(0.5)

    if "a_win_pct" in features.columns and "b_win_pct" in features.columns:
        features["diff_win_pct"] = features["a_win_pct"] - features["b_win_pct"]

    # Weight class encoding
    if "weight_class" in features.columns:
        wc_order = {
            "Strawweight": 0, "Women's Strawweight": 0,
            "Flyweight": 1, "Women's Flyweight": 1,
            "Bantamweight": 2, "Women's Bantamweight": 2,
            "Featherweight": 3, "Women's Featherweight": 3,
            "Lightweight": 4,
            "Welterweight": 5,
            "Middleweight": 6,
            "Light Heavyweight": 7,
            "Heavyweight": 8,
            "Catch Weight": 5,
        }
        features["weight_class_enc"] = features["weight_class"].map(
            lambda x: next(
                (v for k, v in wc_order.items() if k.lower() in str(x).lower()), 5
            )
        )

    # --- Cage rust indicator ---
    # Fighters returning after long layoffs (>365 days) historically underperform
    for prefix in ["a_", "b_"]:
        dslf_col = f"{prefix}days_since_last_fight"
        if dslf_col in features.columns:
            features[f"{prefix}cage_rust"] = (features[dslf_col] > 365).astype(int)
            # Log-scaled layoff (diminishing impact of very long layoffs)
            features[f"{prefix}layoff_log"] = np.log1p(features[dslf_col].fillna(365))
    if "a_cage_rust" in features.columns and "b_cage_rust" in features.columns:
        features["diff_cage_rust"] = features["a_cage_rust"] - features["b_cage_rust"]
    if "a_layoff_log" in features.columns and "b_layoff_log" in features.columns:
        features["diff_layoff_log"] = features["a_layoff_log"] - features["b_layoff_log"]

    # --- Weight class move detection ---
    # Flag fighters who are fighting outside their usual weight class
    if "weight_class" in features.columns:
        _detect_weight_class_moves(features)

    # --- Style matchup interactions ---
    # Striker vs grappler matchup (KO rate vs TD/sub stats)
    for prefix_atk, prefix_def in [("a_", "b_"), ("b_", "a_")]:
        ko_col = f"{prefix_atk}ko_rate"
        td_def_col = f"{prefix_def}roll_td_def"
        sub_col = f"{prefix_atk}sub_rate"
        str_acc_col = f"{prefix_atk}roll_str_acc"
        td_avg_col = f"{prefix_atk}roll_td_avg"
        str_def_col = f"{prefix_def}roll_str_def"

        # Striker advantage: attacker KO rate * (1 - defender striking defense)
        if ko_col in features.columns and str_def_col in features.columns:
            features[f"{prefix_atk}striker_edge"] = (
                features[ko_col].fillna(0) *
                (1.0 - features[str_def_col].fillna(0.5) / 100.0)
            )

        # Grappler advantage: attacker sub rate * (1 - defender TD defense)
        if sub_col in features.columns and td_def_col in features.columns:
            features[f"{prefix_atk}grappler_edge"] = (
                features[sub_col].fillna(0) *
                (1.0 - features[td_def_col].fillna(0.5) / 100.0)
            )

    # Style matchup differentials
    for feat in ["striker_edge", "grappler_edge"]:
        a_col = f"a_{feat}"
        b_col = f"b_{feat}"
        if a_col in features.columns and b_col in features.columns:
            features[f"diff_{feat}"] = features[a_col] - features[b_col]

    # Step 6: Drop rows with insufficient data (first fights for both fighters)
    features["has_data"] = features.get("a_num_fights", pd.Series(0)) + features.get("b_num_fights", pd.Series(0))

    logger.info(f"Built {len(features)} fight feature rows with {len(features.columns)} columns")
    return features


def _detect_weight_class_moves(features: pd.DataFrame) -> None:
    """
    Detect fighters competing outside their usual weight class.
    Uses a fighter's most common weight class from prior fights as their 'home' class.
    """
    wc_weight = {
        "strawweight": 115, "women's strawweight": 115,
        "flyweight": 125, "women's flyweight": 125,
        "bantamweight": 135, "women's bantamweight": 135,
        "featherweight": 145, "women's featherweight": 145,
        "lightweight": 155,
        "welterweight": 170,
        "middleweight": 185,
        "light heavyweight": 205,
        "heavyweight": 265,
        "catch weight": None,
    }

    def _wc_to_weight(wc_str):
        if pd.isna(wc_str):
            return None
        for k, v in wc_weight.items():
            if k in str(wc_str).lower():
                return v
        return None

    fight_wc_weight = features["weight_class"].apply(_wc_to_weight)

    # Build historical mode weight class per fighter
    fighter_home_wc: dict[str, float] = {}
    for _, row in features.sort_values("event_date").iterrows():
        wc_w = _wc_to_weight(row.get("weight_class"))
        if wc_w is None:
            continue
        for prefix, col in [("a_", "fighter_a"), ("b_", "fighter_b")]:
            fighter = row.get(col)
            if not fighter:
                continue
            if fighter not in fighter_home_wc:
                fighter_home_wc[fighter] = wc_w
            # Track most recent frequent class (simple: keep first seen)

    a_moving = []
    b_moving = []
    for _, row in features.iterrows():
        wc_w = _wc_to_weight(row.get("weight_class"))
        fa_home = fighter_home_wc.get(row.get("fighter_a"))
        fb_home = fighter_home_wc.get(row.get("fighter_b"))
        a_moving.append(1 if (wc_w and fa_home and wc_w != fa_home) else 0)
        b_moving.append(1 if (wc_w and fb_home and wc_w != fb_home) else 0)

    features["a_wc_move"] = a_moving
    features["b_wc_move"] = b_moving
    features["diff_wc_move"] = features["a_wc_move"] - features["b_wc_move"]


def get_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Get the list of columns to use as model features."""
    feature_cols = []

    # Differential features
    feature_cols += [c for c in features_df.columns if c.startswith("diff_")]

    # Individual rolling stats
    for prefix in ["a_", "b_"]:
        feature_cols += [
            c for c in features_df.columns
            if c.startswith(f"{prefix}roll_") or c.startswith(f"{prefix}elo")
            or c in [f"{prefix}win_streak", f"{prefix}num_fights",
                     f"{prefix}days_since_last_fight", f"{prefix}strike_diff"]
        ]

    # Encoded categoricals
    feature_cols += [
        c for c in features_df.columns
        if c in ["a_stance_enc", "b_stance_enc", "same_stance", "weight_class_enc"]
    ]

    # Physical attributes
    for prefix in ["a_", "b_"]:
        feature_cols += [
            c for c in features_df.columns
            if c in [f"{prefix}height", f"{prefix}reach", f"{prefix}weight", f"{prefix}age"]
        ]

    # New: finish method rates
    for prefix in ["a_", "b_"]:
        feature_cols += [
            c for c in features_df.columns
            if c in [f"{prefix}ko_rate", f"{prefix}sub_rate", f"{prefix}dec_rate",
                     f"{prefix}win_pct"]
        ]
    feature_cols += [c for c in features_df.columns
                     if c in ["diff_ko_rate", "diff_sub_rate", "diff_dec_rate", "diff_win_pct"]]

    # New: historical odds implied probability
    feature_cols += [c for c in features_df.columns
                     if c in ["a_implied_prob", "b_implied_prob", "diff_implied_prob"]]

    # New: title bout, num rounds, empty arena
    feature_cols += [c for c in features_df.columns
                     if c in ["is_title_bout", "num_rounds_feat", "is_empty_arena"]]

    # New: rankings
    feature_cols += [c for c in features_df.columns
                     if c in ["a_wc_rank_feat", "b_wc_rank_feat", "diff_wc_rank",
                              "a_pfp_rank_feat", "b_pfp_rank_feat", "diff_pfp_rank"]]

    # New: streaks, rounds, title bouts, draws differentials
    feature_cols += [c for c in features_df.columns
                     if c in ["diff_lose_streak", "diff_longest_win_streak",
                              "diff_total_rounds", "diff_title_bouts", "diff_draws"]]

    # New: per-fighter lose streak, longest win streak, total rounds, title bouts
    for prefix in ["a_", "b_"]:
        feature_cols += [
            c for c in features_df.columns
            if c in [f"{prefix}lose_streak", f"{prefix}longest_win_streak",
                     f"{prefix}total_rounds", f"{prefix}title_bouts", f"{prefix}draws"]
        ]

    # New: method-specific odds probabilities
    for prefix in ["a_", "b_"]:
        feature_cols += [
            c for c in features_df.columns
            if c in [f"{prefix}ko_odds_prob", f"{prefix}sub_odds_prob",
                     f"{prefix}dec_odds_prob"]
        ]

    # Line movement features (from historical backfill)
    feature_cols += [c for c in features_df.columns
                     if c in ["line_movement", "line_abs_movement", "line_is_sharp",
                              "line_steam_move", "line_direction_toward_a",
                              "line_direction_toward_b"]]

    # Cage rust and layoff features
    for prefix in ["a_", "b_"]:
        feature_cols += [c for c in features_df.columns
                         if c in [f"{prefix}cage_rust", f"{prefix}layoff_log"]]
    feature_cols += [c for c in features_df.columns
                     if c in ["diff_cage_rust", "diff_layoff_log"]]

    # Weight class move detection
    feature_cols += [c for c in features_df.columns
                     if c in ["a_wc_move", "b_wc_move", "diff_wc_move"]]

    # Style matchup interactions
    for prefix in ["a_", "b_"]:
        feature_cols += [c for c in features_df.columns
                         if c in [f"{prefix}striker_edge", f"{prefix}grappler_edge"]]
    feature_cols += [c for c in features_df.columns
                     if c in ["diff_striker_edge", "diff_grappler_edge"]]

    # Deduplicate and filter to columns that exist
    feature_cols = list(dict.fromkeys(feature_cols))
    feature_cols = [c for c in feature_cols if c in features_df.columns]

    return feature_cols


# Columns that are odds-derived and should be excluded for the no-odds model
ODDS_FEATURE_NAMES = {
    "a_implied_prob", "b_implied_prob", "diff_implied_prob",
    "a_ko_odds_prob", "a_sub_odds_prob", "a_dec_odds_prob",
    "b_ko_odds_prob", "b_sub_odds_prob", "b_dec_odds_prob",
}


def get_feature_columns_no_odds(features_df: pd.DataFrame) -> list[str]:
    """Get feature columns excluding all odds-derived features.

    This enables training a model that relies purely on fighter stats,
    Elo, physical attributes, etc. — used as a baseline to measure
    whether the model has independent edge beyond market consensus.
    """
    all_cols = get_feature_columns(features_df)
    return [c for c in all_cols if c not in ODDS_FEATURE_NAMES]


def get_fighter_ufc_fight_count(fighter_name: str) -> int:
    """
    Look up how many UFC fights a fighter has from the processed dataset.
    Returns 0 if the fighter is not found (i.e., a UFC debutant).
    """
    features_path = PROCESSED_DATA_DIR / "features.csv"
    if not features_path.exists():
        return 0

    try:
        df = pd.read_csv(features_path, usecols=["fighter_a", "fighter_b", "a_num_fights", "b_num_fights"])
    except (ValueError, KeyError):
        return 0

    name_lower = fighter_name.lower()

    # Check as fighter_a
    mask_a = df["fighter_a"].str.lower() == name_lower
    if mask_a.any():
        return int(df.loc[mask_a, "a_num_fights"].max())

    # Check as fighter_b
    mask_b = df["fighter_b"].str.lower() == name_lower
    if mask_b.any():
        return int(df.loc[mask_b, "b_num_fights"].max())

    return 0


def save_features(features_df: pd.DataFrame, filename: str = "features.csv") -> None:
    """Save feature matrix to processed data directory."""
    path = PROCESSED_DATA_DIR / filename
    features_df.to_csv(path, index=False)
    logger.info(f"Saved features to {path}")
