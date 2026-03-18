"""
Feature engineering for UFC fight prediction.

Computes rolling fighter stats, Elo ratings, and fighter differentials
from historical fight data. All features are computed using only data
available BEFORE each fight (no data leakage).
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import ROLLING_WINDOW, EWM_HALFLIFE, ELO_INITIAL, ELO_K_FACTOR, PROCESSED_DATA_DIR

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
_HISTORY_BACKED_FIGHTER_FIELDS = {
    "wins": "prior_wins",
    "losses": "prior_losses",
    "draws": "prior_draws",
    "lose_streak": "prior_lose_streak",
    "longest_win_streak": "prior_longest_win_streak",
    "total_rounds": "prior_total_rounds",
    "title_bouts": "prior_title_bouts",
    "wins_ko": "prior_wins_ko",
    "wins_sub": "prior_wins_sub",
    "wins_dec": "prior_wins_dec",
}


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
        finish_round = pd.to_numeric(pd.Series([row.get("finish_round")]), errors="coerce").iloc[0]
        title_bout = row.get("title_bout", np.nan)

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
                "result_label": _fight_result_label(winner, fighter, row.get(opp_col, "")),
                "method": row.get("method", ""),
                "finish_round": finish_round,
                "title_bout": title_bout,
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
    fighter_fights: pd.DataFrame, halflife: int = EWM_HALFLIFE
) -> pd.DataFrame:
    """
    Compute exponentially weighted rolling averages for a single fighter's fight history.
    Recent fights are weighted more heavily (halflife=3 fights by default).
    Stats are computed from all fights BEFORE the current one (shift(1)).
    """
    fighter_fights = fighter_fights.sort_values("event_date").copy()

    stats_to_roll = STAT_COLUMNS + [f"opp_{s}" for s in STAT_COLUMNS] + ["won"]

    for stat in stats_to_roll:
        if stat in fighter_fights.columns:
            # shift(1) ensures we only use data from BEFORE this fight
            shifted = fighter_fights[stat].shift(1)
            fighter_fights[f"roll_{stat}"] = (
                shifted.ewm(halflife=halflife, min_periods=1).mean()
            )

    (
        fighter_fights["current_win_streak"],
        fighter_fights["prior_wins"],
        fighter_fights["prior_losses"],
        fighter_fights["prior_draws"],
        fighter_fights["prior_lose_streak"],
        fighter_fights["prior_longest_win_streak"],
        fighter_fights["prior_total_rounds"],
        fighter_fights["prior_title_bouts"],
        fighter_fights["prior_wins_ko"],
        fighter_fights["prior_wins_sub"],
        fighter_fights["prior_wins_dec"],
    ) = _career_history_columns(fighter_fights)

    # Fights count (prior completed UFC fights going into this fight)
    fighter_fights["num_fights"] = range(len(fighter_fights))

    # Days since last fight
    dates = fighter_fights["event_date"]
    fighter_fights["days_since_last_fight"] = dates.diff().dt.days.fillna(365)

    return fighter_fights


def _fight_result_label(winner: object, fighter: object, opponent: object) -> str:
    if winner == fighter:
        return "win"
    if winner == opponent:
        return "loss"
    return "draw"


def _career_history_columns(fighter_fights: pd.DataFrame) -> tuple[list[int], ...]:
    current_win_streaks: list[int] = []
    prior_wins: list[int] = []
    prior_losses: list[int] = []
    prior_draws: list[int] = []
    prior_lose_streaks: list[int] = []
    prior_longest_win_streaks: list[int] = []
    prior_total_rounds: list[float] = []
    prior_title_bouts: list[int] = []
    prior_wins_ko: list[int] = []
    prior_wins_sub: list[int] = []
    prior_wins_dec: list[int] = []

    wins = losses = draws = 0
    win_streak = lose_streak = longest_win_streak = 0
    total_rounds = 0.0
    title_bouts = 0
    wins_ko = wins_sub = wins_dec = 0

    for _, row in fighter_fights.iterrows():
        current_win_streaks.append(win_streak)
        prior_wins.append(wins)
        prior_losses.append(losses)
        prior_draws.append(draws)
        prior_lose_streaks.append(lose_streak)
        prior_longest_win_streaks.append(longest_win_streak)
        prior_total_rounds.append(total_rounds)
        prior_title_bouts.append(title_bouts)
        prior_wins_ko.append(wins_ko)
        prior_wins_sub.append(wins_sub)
        prior_wins_dec.append(wins_dec)

        result_label = row.get("result_label", "draw")
        method_group = _method_group(row.get("method"))

        if result_label == "win":
            wins += 1
            win_streak += 1
            lose_streak = 0
            longest_win_streak = max(longest_win_streak, win_streak)
            if method_group == "ko":
                wins_ko += 1
            elif method_group == "sub":
                wins_sub += 1
            elif method_group == "dec":
                wins_dec += 1
        elif result_label == "loss":
            losses += 1
            lose_streak += 1
            win_streak = 0
        else:
            draws += 1
            win_streak = 0
            lose_streak = 0

        finish_round = pd.to_numeric(pd.Series([row.get("finish_round")]), errors="coerce").iloc[0]
        if not pd.isna(finish_round):
            total_rounds += float(finish_round)

        title_flag = pd.to_numeric(pd.Series([row.get("title_bout")]), errors="coerce").iloc[0]
        if not pd.isna(title_flag) and float(title_flag) != 0.0:
            title_bouts += 1

    return (
        current_win_streaks,
        prior_wins,
        prior_losses,
        prior_draws,
        prior_lose_streaks,
        prior_longest_win_streaks,
        prior_total_rounds,
        prior_title_bouts,
        prior_wins_ko,
        prior_wins_sub,
        prior_wins_dec,
    )


def _method_group(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if "dec" in text or "decision" in text:
        return "dec"
    if "sub" in text:
        return "sub"
    if "ko" in text or "tko" in text:
        return "ko"
    return None


def _coerce_nullable_binary(series: pd.Series) -> pd.Series:
    """Coerce a mixed binary series to float 0/1 while preserving unknowns as NaN."""
    coerced = pd.to_numeric(series, errors="coerce").astype(float)
    unresolved = coerced.isna() & series.notna()
    if unresolved.any():
        normalized = series.astype(str).str.strip().str.lower()
        truthy = {"1", "true", "t", "yes", "y"}
        falsy = {"0", "false", "f", "no", "n"}
        coerced.loc[unresolved & normalized.isin(truthy)] = 1.0
        coerced.loc[unresolved & normalized.isin(falsy)] = 0.0
    return coerced


def materialize_honest_context_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild context/ranking features directly from raw columns without fake defaults.

    Historical unknowns must stay NaN instead of being rewritten as title_bout=0,
    empty_arena=0, or rank=16 unless the raw value is actually present.
    """
    features = features_df.copy()

    if "title_bout" in features.columns:
        features["is_title_bout"] = _coerce_nullable_binary(features["title_bout"])

    if "num_rounds" in features.columns:
        features["num_rounds_feat"] = pd.to_numeric(features["num_rounds"], errors="coerce").astype(float)

    if "empty_arena" in features.columns:
        features["is_empty_arena"] = _coerce_nullable_binary(features["empty_arena"])

    for prefix in ["a_", "b_"]:
        wc_rank_col = f"{prefix}wc_rank"
        if wc_rank_col in features.columns:
            features[f"{prefix}wc_rank_feat"] = pd.to_numeric(
                features[wc_rank_col],
                errors="coerce",
            ).astype(float)

        pfp_rank_col = f"{prefix}pfp_rank"
        if pfp_rank_col in features.columns:
            features[f"{prefix}pfp_rank_feat"] = pd.to_numeric(
                features[pfp_rank_col],
                errors="coerce",
            ).astype(float)

    if "a_wc_rank_feat" in features.columns and "b_wc_rank_feat" in features.columns:
        features["diff_wc_rank"] = features["a_wc_rank_feat"] - features["b_wc_rank_feat"]
    if "a_pfp_rank_feat" in features.columns and "b_pfp_rank_feat" in features.columns:
        features["diff_pfp_rank"] = features["a_pfp_rank_feat"] - features["b_pfp_rank_feat"]

    return features


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
    rolling_helper_cols = {
        "fighter",
        "opponent",
        "event_date",
        "weight_class",
        "won",
        "result_label",
        "method",
        "finish_round",
        "title_bout",
    }

    a_rolling = rolling_df.rename(
        columns={c: f"a_roll_{c.replace('roll_', '')}" if c.startswith("roll_") else f"a_{c}"
                 for c in rolling_df.columns
                 if c not in rolling_helper_cols}
    )
    # Keep key columns for merge
    a_merge_cols = ["fighter", "event_date"] + [
        c for c in a_rolling.columns
        if c.startswith("a_roll_") or c in ["current_win_streak", "num_fights", "days_since_last_fight"]
    ]
    # Rename non-prefixed columns for fighter A
    rename_map = {}
    for c in ["current_win_streak", "num_fights", "days_since_last_fight"]:
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
    rename_b = {
        c: f"b_roll_{c.replace('roll_', '')}" if c.startswith("roll_") else f"b_{c}"
        for c in rolling_df.columns
        if c not in rolling_helper_cols
    }
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

    _fill_history_backed_fighter_fields(features)

    # Step 5: Compute differentials
    diff_stats = [
        "roll_slpm", "roll_sapm", "roll_str_acc", "roll_str_def",
        "roll_td_avg", "roll_td_acc", "roll_td_def", "roll_sub_avg",
        "roll_sig_str_landed", "roll_td_landed", "roll_kd",
        "roll_won", "elo", "current_win_streak", "num_fights", "days_since_last_fight",
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

    features = materialize_honest_context_features(features)

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
                (1.0 - features[str_def_col].fillna(50.0) / 100.0)
            )

        # Grappler advantage: attacker sub rate * (1 - defender TD defense)
        if sub_col in features.columns and td_def_col in features.columns:
            features[f"{prefix_atk}grappler_edge"] = (
                features[sub_col].fillna(0) *
                (1.0 - features[td_def_col].fillna(50.0) / 100.0)
            )

    # Style matchup differentials
    for feat in ["striker_edge", "grappler_edge"]:
        a_col = f"a_{feat}"
        b_col = f"b_{feat}"
        if a_col in features.columns and b_col in features.columns:
            features[f"diff_{feat}"] = features[a_col] - features[b_col]

    # Step 6: Add experimental features (fight pace, cage time efficiency, quality-adjusted stats)
    from src.features.experimental_features import add_experimental_features
    features = add_experimental_features(features)

    # Step 7: Drop rows with insufficient data (first fights for both fighters)
    features["has_data"] = features.get("a_num_fights", pd.Series(0)) + features.get("b_num_fights", pd.Series(0))

    logger.info(f"Built {len(features)} fight feature rows with {len(features.columns)} columns")
    return features


def _fill_history_backed_fighter_fields(features: pd.DataFrame) -> None:
    for prefix in ["a_", "b_"]:
        for final_name, history_name in _HISTORY_BACKED_FIGHTER_FIELDS.items():
            target_col = f"{prefix}{final_name}"
            history_col = f"{prefix}{history_name}"
            if history_col not in features.columns:
                continue
            if target_col in features.columns:
                features[target_col] = features[target_col].where(features[target_col].notna(), features[history_col])
            else:
                features[target_col] = features[history_col]
            features.drop(columns=[history_col], inplace=True)


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

    # Build rolling prior-mode weight class per fighter (per-fight basis)
    fighter_wc_history: dict[str, Counter] = {}
    # Store per-row wc_move flags keyed by DataFrame index
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
            or c in [f"{prefix}current_win_streak", f"{prefix}num_fights",
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

    # Strength of Schedule (added by model lab variants)
    feature_cols += [c for c in features_df.columns
                     if c in ["a_sos", "b_sos", "diff_sos"]]

    # Rematch / Head-to-Head (added by model lab variants)
    feature_cols += [c for c in features_df.columns
                     if c in ["is_rematch", "h2h_record_diff"]]

    # Elo momentum (added by model lab variants)
    feature_cols += [c for c in features_df.columns
                     if c in ["a_elo_momentum", "b_elo_momentum", "diff_elo_momentum"]]

    # Experimental features (fight pace, cage time efficiency, quality-adjusted stats)
    for prefix in ["a_", "b_"]:
        feature_cols += [c for c in features_df.columns
                         if c in [f"{prefix}fight_pace", f"{prefix}ctrl_efficiency",
                                  f"{prefix}adj_win_pct"]]

    # Deduplicate and filter to columns that exist
    feature_cols = list(dict.fromkeys(feature_cols))
    feature_cols = [c for c in feature_cols if c in features_df.columns]

    return feature_cols


# Direct odds columns used for odds noise augmentation.
ODDS_FEATURE_NAMES = {
    "a_implied_prob", "b_implied_prob", "diff_implied_prob",
    "a_ko_odds_prob", "a_sub_odds_prob", "a_dec_odds_prob",
    "b_ko_odds_prob", "b_sub_odds_prob", "b_dec_odds_prob",
}


# All market-derived columns that must be excluded from the no-odds model.
MARKET_DERIVED_FEATURE_NAMES = ODDS_FEATURE_NAMES | {
    "line_movement",
    "line_abs_movement",
    "line_is_sharp",
    "line_steam_move",
    "line_direction_toward_a",
    "line_direction_toward_b",
}


def exclude_market_derived_features(feature_cols: list[str]) -> list[str]:
    """Drop market-derived features from a feature-column list."""
    return [c for c in feature_cols if c not in MARKET_DERIVED_FEATURE_NAMES]


def get_feature_columns_no_odds(features_df: pd.DataFrame) -> list[str]:
    """Get feature columns excluding all market-derived features.

    This enables training a model that relies purely on fighter stats,
    Elo, physical attributes, etc. — used as a baseline to measure
    whether the model has independent edge beyond market consensus.
    """
    all_cols = get_feature_columns(features_df)
    return exclude_market_derived_features(all_cols)


# Historical-only core plus the wider expanded BetsAPI family. The name is kept
# for backward compatibility with existing callers and cache/report labels.
BETSAPI_CHALLENGER_FEATURE_NAMES = [
    "betsapi_hist_opening_a_implied_prob",
    "betsapi_hist_opening_b_implied_prob",
    "betsapi_hist_diff_opening_implied_prob",
    "betsapi_hist_opening_bookmakers",
    "betsapi_hist_opening_lead_hours",
    "betsapi_hist_market_disagreement_a",
    "betsapi_hist_market_disagreement_b",
    "betsapi_hist_market_disagreement_abs",
    "betsapi_hist_m162_2_opening_home_implied_prob",
    "betsapi_hist_m162_2_opening_away_implied_prob",
    "betsapi_hist_m162_2_opening_handicap",
    "betsapi_hist_m162_2_end_home_implied_prob",
    "betsapi_hist_m162_2_end_away_implied_prob",
    "betsapi_hist_m162_2_end_handicap",
    "betsapi_hist_m162_2_prob_move_abs",
    "betsapi_hist_m162_2_handicap_move_abs",
    "betsapi_hist_m162_3_opening_over_implied_prob",
    "betsapi_hist_m162_3_opening_under_implied_prob",
    "betsapi_hist_m162_3_opening_handicap",
    "betsapi_hist_m162_3_end_over_implied_prob",
    "betsapi_hist_m162_3_end_under_implied_prob",
    "betsapi_hist_m162_3_end_handicap",
    "betsapi_hist_m162_3_prob_move_abs",
    "betsapi_hist_m162_3_handicap_move_abs",
    "betsapi_snapshot_count",
    "betsapi_a_implied_prob",
    "betsapi_b_implied_prob",
    "betsapi_diff_implied_prob",
    "betsapi_odds_staleness_hours",
    "betsapi_line_move_a",
    "betsapi_line_move_b",
    "betsapi_line_move_abs",
    "betsapi_market_disagreement_a",
    "betsapi_market_disagreement_b",
    "betsapi_market_disagreement_abs",
    "betsapi_hist_bookmaker_count",
    "betsapi_hist_overround",
    "betsapi_hist_prob_move_open_close_a",
    "betsapi_hist_prob_move_open_close_b",
    "betsapi_hist_moneyline_total_agreement",
    "betsapi_hist_handicap_direction",
    "betsapi_hist_total_direction",
    "betsapi_hist_has_late_update",
    "betsapi_hist_has_moneyline",
    "betsapi_hist_has_totals",
    "betsapi_hist_has_handicap",
    "betsapi_hist_event_coverage_pct",
    "betsapi_hist_market_depth",
]

UFCSTATS_REFRESH_ONLY_FEATURE_NAMES = [
    "a_roll_distance_acc",
    "a_roll_clinch_acc",
    "a_roll_ground_acc",
    "a_roll_ctrl_per_minute",
    "a_roll_distance_pct",
    "a_roll_clinch_pct",
    "a_roll_ground_pct",
    "a_roll_strikes_avoided_pct",
    "a_roll_opp_elo_avg",
    "a_roll_adj_win_rate",
    "a_opp_strength_recent_3",
    "a_roll_finish_rate",
    "a_roll_early_finish_rate",
    "a_roll_decision_rate",
    "b_roll_distance_acc",
    "b_roll_clinch_acc",
    "b_roll_ground_acc",
    "b_roll_ctrl_per_minute",
    "b_roll_distance_pct",
    "b_roll_clinch_pct",
    "b_roll_ground_pct",
    "b_roll_strikes_avoided_pct",
    "b_roll_opp_elo_avg",
    "b_roll_adj_win_rate",
    "b_opp_strength_recent_3",
    "b_roll_finish_rate",
    "b_roll_early_finish_rate",
    "b_roll_decision_rate",
    "pace_mismatch",
]

BETSAPI_HISTORICAL_FEATURE_NAMES = [
    "betsapi_hist_opening_a_implied_prob",
    "betsapi_hist_opening_b_implied_prob",
    "betsapi_hist_diff_opening_implied_prob",
    "betsapi_hist_opening_bookmakers",
    "betsapi_hist_opening_lead_hours",
    "betsapi_hist_market_disagreement_a",
    "betsapi_hist_market_disagreement_b",
    "betsapi_hist_market_disagreement_abs",
    "betsapi_hist_m162_2_opening_home_implied_prob",
    "betsapi_hist_m162_2_opening_away_implied_prob",
    "betsapi_hist_m162_2_opening_handicap",
    "betsapi_hist_m162_2_end_home_implied_prob",
    "betsapi_hist_m162_2_end_away_implied_prob",
    "betsapi_hist_m162_2_end_handicap",
    "betsapi_hist_m162_2_prob_move_abs",
    "betsapi_hist_m162_2_handicap_move_abs",
    "betsapi_hist_m162_3_opening_over_implied_prob",
    "betsapi_hist_m162_3_opening_under_implied_prob",
    "betsapi_hist_m162_3_opening_handicap",
    "betsapi_hist_m162_3_end_over_implied_prob",
    "betsapi_hist_m162_3_end_under_implied_prob",
    "betsapi_hist_m162_3_end_handicap",
    "betsapi_hist_m162_3_prob_move_abs",
    "betsapi_hist_m162_3_handicap_move_abs",
]


def get_betsapi_challenger_feature_columns(
    features_df: pd.DataFrame,
    base_feature_cols: Optional[list[str]] = None,
) -> list[str]:
    """Return the explicit numeric feature set for BetsAPI challenger runs."""
    feature_cols = list(base_feature_cols if base_feature_cols is not None else get_feature_columns(features_df))
    feature_cols += [c for c in BETSAPI_CHALLENGER_FEATURE_NAMES if c in features_df.columns]
    return list(dict.fromkeys(feature_cols))


MARKET_DERIVED_DENYLIST = {
    "a_implied_prob",
    "b_implied_prob",
    "diff_implied_prob",
    "a_ko_odds_prob",
    "a_sub_odds_prob",
    "a_dec_odds_prob",
    "b_ko_odds_prob",
    "b_sub_odds_prob",
    "b_dec_odds_prob",
    "line_movement",
    "line_abs_movement",
    "line_is_sharp",
    "line_steam_move",
    "a_line_movement",
    "b_line_movement",
    "diff_line_movement",
    "a_line_abs_movement",
    "b_line_abs_movement",
    "diff_line_abs_movement",
    "a_line_is_sharp",
    "b_line_is_sharp",
    "a_line_steam_move",
    "b_line_steam_move",
    "line_direction_toward_a",
    "line_direction_toward_b",
}

MARKET_DERIVED_PATTERN_TOKENS = (
    "betsapi_",
    "implied_prob",
    "market_prob",
    "fair_prob",
    "closing_prob",
    "line_move",
    "line_movement",
    "line_abs_movement",
    "line_direction_",
    "line_is_sharp",
    "steam_move",
    "_odds",
    "odds_",
)

ODDS_FEATURE_NAMES = tuple(sorted(MARKET_DERIVED_DENYLIST))
MARKET_DERIVED_FEATURE_NAMES = set(MARKET_DERIVED_DENYLIST)


def is_market_derived_feature(column: str) -> bool:
    """Return True when a feature is derived from betting market inputs."""
    if column in MARKET_DERIVED_DENYLIST:
        return True
    lowered = column.lower()
    return any(token in lowered for token in MARKET_DERIVED_PATTERN_TOKENS)


def get_market_derived_feature_columns(feature_cols: list[str]) -> list[str]:
    """Return the market-derived subset of a feature list."""
    return [column for column in feature_cols if is_market_derived_feature(column)]


def filter_feature_columns_for_no_odds(feature_cols: list[str]) -> list[str]:
    """Remove all market-derived columns from a feature list."""
    return [column for column in feature_cols if not is_market_derived_feature(column)]


def exclude_market_derived_features(feature_cols: list[str]) -> list[str]:
    """Backward-compatible wrapper for no-odds feature filtering."""
    return filter_feature_columns_for_no_odds(feature_cols)


def get_production_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Return the production core feature set only."""
    all_cols = get_feature_columns(features_df)
    return [column for column in all_cols if column not in UFCSTATS_REFRESH_ONLY_FEATURE_NAMES]


def get_ufcstats_refreshed_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Return production core plus refresh-only UFCStats additions."""
    feature_cols = get_production_feature_columns(features_df)
    feature_cols += [
        column
        for column in UFCSTATS_REFRESH_ONLY_FEATURE_NAMES
        if column in features_df.columns
    ]
    return list(dict.fromkeys(feature_cols))


def get_betsapi_historical_feature_columns(
    features_df: pd.DataFrame,
    base_feature_cols: Optional[list[str]] = None,
) -> list[str]:
    """Return production core plus the historical-only BetsAPI subset."""
    feature_cols = list(
        base_feature_cols if base_feature_cols is not None else get_production_feature_columns(features_df)
    )
    feature_cols += [
        column
        for column in BETSAPI_HISTORICAL_FEATURE_NAMES
        if column in features_df.columns
    ]
    return list(dict.fromkeys(feature_cols))


def get_feature_family_columns(
    features_df: pd.DataFrame,
    family: str,
) -> list[str]:
    """Return the explicit feature-column set for a named family."""
    production_cols = get_production_feature_columns(features_df)
    refreshed_cols = get_ufcstats_refreshed_feature_columns(features_df)

    if family == "production":
        return production_cols
    if family == "ufcstats_refreshed":
        return refreshed_cols
    if family == "production_betsapi":
        return get_betsapi_historical_feature_columns(features_df, production_cols)
    if family == "production_betsapi_expanded":
        return get_betsapi_challenger_feature_columns(features_df, production_cols)
    if family == "ufcstats_betsapi_expanded":
        return get_betsapi_challenger_feature_columns(features_df, refreshed_cols)
    if family == "no_odds":
        return get_feature_columns_no_odds(features_df, base_feature_cols=production_cols)
    raise ValueError(f"Unknown feature family: {family!r}")


def get_feature_columns_no_odds(
    features_df: pd.DataFrame,
    base_feature_cols: Optional[list[str]] = None,
) -> list[str]:
    """Get feature columns excluding all odds-derived features."""
    all_cols = list(base_feature_cols if base_feature_cols is not None else get_feature_columns(features_df))
    return filter_feature_columns_for_no_odds(all_cols)


# Polymarket name → UFCStats name (for fighters whose market name differs)
FIGHTER_NAME_ALIASES = {
    "Joseph Pyfer": "Joe Pyfer",
}


def get_fighter_ufc_fight_count(fighter_name: str) -> int:
    """
    Look up how many UFC fights a fighter has from the processed dataset.
    Returns 0 if the fighter is not found (i.e., a UFC debutant).
    """
    fighter_name = FIGHTER_NAME_ALIASES.get(fighter_name, fighter_name)

    features_path = PROCESSED_DATA_DIR / "features.csv"
    if not features_path.exists():
        return 0

    try:
        df = pd.read_csv(features_path, usecols=["fighter_a", "fighter_b", "a_num_fights", "b_num_fights"])
    except (ValueError, KeyError):
        return 0

    name_lower = fighter_name.lower()
    mask_a = df["fighter_a"].str.lower() == name_lower
    mask_b = df["fighter_b"].str.lower() == name_lower
    return int(mask_a.sum() + mask_b.sum())


def save_features(features_df: pd.DataFrame, filename: str | Path = "features.csv") -> None:
    """Save feature matrix to processed data directory."""
    path = Path(filename)
    if not path.is_absolute():
        path = PROCESSED_DATA_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(path, index=False)
    logger.info(f"Saved features to {path}")
