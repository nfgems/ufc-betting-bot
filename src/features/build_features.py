"""
Feature engineering for UFC fight prediction.

Computes rolling fighter stats and fighter differentials
from historical fight data. All features are computed using only data
available BEFORE each fight (no data leakage).
"""

import hashlib
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    ROLLING_WINDOW,
    EWM_HALFLIFE,
    ELO_INITIAL,
    ELO_K_FACTOR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.data.io_utils import write_csv_atomically
from src.data.event_context import coerce_nullable_bool
from src.data.name_utils import (
    KNOWN_AMBIGUOUS_FIGHTER_NAME_KEYS,
    REVIEWED_FIGHTER_IDENTITIES,
    derive_ambiguous_fighter_name_keys,
    fighter_identity_is_ambiguous,
    fighter_identity_key,
    normalize_ufcstats_id,
    same_person_name,
)
from src.data.pre_ufc_scraper import _dedupe_supplement_rows
from src.features.stance_utils import encode_stance

logger = logging.getLogger(__name__)


class UnresolvedTrainingFighterIdentityError(ValueError):
    """A collision-prone training row is missing its stable fighter ID."""


_training_ambiguous_name_keys_cache: tuple[float, frozenset[str]] | None = None


def _training_ambiguous_name_keys() -> frozenset[str]:
    """Derive collision groups from the checked-in fighter inventory."""
    global _training_ambiguous_name_keys_cache
    inventory_path = RAW_DATA_DIR / "ufc_fighters_scraped.csv"
    try:
        mtime = inventory_path.stat().st_mtime
    except OSError:
        return KNOWN_AMBIGUOUS_FIGHTER_NAME_KEYS
    if (
        _training_ambiguous_name_keys_cache is not None
        and _training_ambiguous_name_keys_cache[0] == mtime
    ):
        return _training_ambiguous_name_keys_cache[1]
    try:
        inventory = pd.read_csv(inventory_path, usecols=["name", "fighter_url"])
        derived = derive_ambiguous_fighter_name_keys(
            (
                {"name": row.name, "fighter_id": row.fighter_url}
                for row in inventory.itertuples(index=False)
            )
        )
    except (OSError, ValueError) as exc:
        logger.warning("Could not derive fighter identity collisions from %s: %s", inventory_path, exc)
        derived = frozenset()
    resolved = frozenset(set(KNOWN_AMBIGUOUS_FIGHTER_NAME_KEYS) | set(derived))
    _training_ambiguous_name_keys_cache = (mtime, resolved)
    return resolved


def _attach_fighter_identity_keys(fights_df: pd.DataFrame) -> pd.DataFrame:
    """Attach temporary stateful keys while preserving display names and IDs."""
    fights = fights_df.copy()
    unresolved: list[str] = []
    ambiguous_name_keys = _training_ambiguous_name_keys()
    for side in ("a", "b"):
        name_col = f"fighter_{side}"
        id_col = f"fighter_{side}_id"
        key_col = f"__fighter_{side}_key"
        if id_col not in fights.columns:
            fights[id_col] = None
        fights[id_col] = fights[id_col].map(normalize_ufcstats_id)
        fights[key_col] = fights.apply(
            lambda row: fighter_identity_key(
                row.get(name_col),
                row.get(id_col),
                ambiguous_name_keys=ambiguous_name_keys,
            ),
            axis=1,
        )
        bad = fights[name_col].notna() & fights[key_col].isna()
        if bad.any():
            unresolved.extend(
                f"{row.get('event_date')}:{row.get(name_col)}"
                for _, row in fights.loc[bad].head(5).iterrows()
            )
    if unresolved:
        raise UnresolvedTrainingFighterIdentityError(
            "ambiguous training fighter rows require UFCStats IDs: "
            + ", ".join(unresolved)
        )
    return fights


def _stateful_identity_column(frame: pd.DataFrame) -> str:
    return "fighter_key" if "fighter_key" in frame.columns else "fighter"


# ---------------------------------------------------------------------------
# Elo rating system
# ---------------------------------------------------------------------------


class EloSystem:
    """Backward-compatible Elo rating tracker for UFC helpers/tests."""

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
        """Update ratings after a fight and return the new fighter ratings."""
        rating_a = self.get_rating(fighter_a)
        rating_b = self.get_rating(fighter_b)

        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a

        if winner == fighter_a:
            score_a, score_b = 1.0, 0.0
        elif winner == fighter_b:
            score_a, score_b = 0.0, 1.0
        else:
            score_a, score_b = 0.5, 0.5

        k_a = self.k * (1.5 if self.get_fight_count(fighter_a) < 5 else 1.0)
        k_b = self.k * (1.5 if self.get_fight_count(fighter_b) < 5 else 1.0)

        self.ratings[fighter_a] = rating_a + k_a * (score_a - expected_a)
        self.ratings[fighter_b] = rating_b + k_b * (score_b - expected_b)
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

# Extended stats scraped by ufc_refresh.py — position/target shares + control metrics.
# These get the same rolling-average treatment as STAT_COLUMNS.
EXTENDED_STAT_COLUMNS = [
    "head_str_share", "body_str_share", "leg_str_share",
    "distance_str_share", "clinch_str_share", "ground_str_share",
    "control_share", "control_per_td",
    # Per-fight accuracies computed in _compute_per_fight_stats
    "distance_acc", "clinch_acc", "ground_acc",
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

_PRE_UFC_SUPPLEMENT_CANDIDATES = [
    "pre_ufc_career_supplement_v2.csv",
    "pre_ufc_career_supplement.csv",
]
_PRE_UFC_REQUIRED_COLUMNS = frozenset(
    {
        "event_date",
        "fighter_a",
        "fighter_b",
        "winner",
        "method",
        "organization",
    }
)
_SUPPLEMENT_QUARANTINED_IDENTITIES = frozenset(
    {
        "Kai Kamaka",
        "Kai Kamaka III",
        "Mizuki",
        "Gabriel Santos",
    }
)
_AMATEUR_SUPPLEMENT_FILENAME = "amateur_career_supplement.csv"
_REVIEWED_SUPPLEMENT_HISTORY_FILENAME = "reviewed_fighter_history.csv"
_REVIEWED_SUPPLEMENT_HISTORY_CANONICAL_SHA256 = (
    "4a292605f9d5e28990a8aff9148de42402bb0100c9ec454194807170c5531c9c"
)
_REVIEWED_HISTORY_REQUIRED_COLUMNS = frozenset(
    {
        "history_type",
        "event_date",
        "fighter_a",
        "fighter_b",
        "winner",
        "subject_result",
        "method",
        "source",
        "organization",
        "ufcstats_id",
        "ufcstats_url",
        "subject_dob",
        "source_profile_id",
        "source_profile_url",
        "source_profile_dob",
        "dob_match",
    }
)
_PRE_UFC_SUMMARY_COLUMNS = [
    "pre_ufc_total_fights",
    "pre_ufc_wins",
    "pre_ufc_losses",
    "pre_ufc_win_pct",
    "pre_ufc_ko_rate",
    "pre_ufc_sub_rate",
    "pre_ufc_dec_rate",
    "pre_ufc_org_tier_best",
]
_AMATEUR_SUMMARY_COLUMNS = [
    "amateur_total_fights",
    "amateur_wins",
    "amateur_losses",
    "amateur_win_pct",
    "amateur_ko_rate",
    "amateur_sub_rate",
    "amateur_dec_rate",
]

# Exact organization labels for bouts already represented in the tracked
# UFC/UFCStats history.  Keep this deliberately narrow: similarly named
# regional promotions such as WUFC and UFCF are not the UFC, and substring
# matching would silently discard real pre-UFC observations.
_TRACKED_UFC_ORGANIZATION_LABELS = frozenset(
    {
        "ufc",
        "ultimate fighting championship",
        "dana white's contender series",
        "dana whites contender series",
        "road to ufc",
    }
)


def _resolve_pre_ufc_supplement_path() -> Path:
    """Prefer the richer v2 pre-UFC supplement when it exists."""
    for filename in _PRE_UFC_SUPPLEMENT_CANDIDATES:
        candidate = RAW_DATA_DIR / filename
        if candidate.exists():
            return candidate
    return RAW_DATA_DIR / _PRE_UFC_SUPPLEMENT_CANDIDATES[0]


def _resolve_amateur_supplement_path() -> Path:
    """Return the amateur career supplement path."""
    return RAW_DATA_DIR / _AMATEUR_SUPPLEMENT_FILENAME


def _resolve_reviewed_supplement_history_path() -> Path:
    """Return the small stable-ID-backed supplemental history artifact."""
    return RAW_DATA_DIR / _REVIEWED_SUPPLEMENT_HISTORY_FILENAME


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
            side = "a" if prefix == "a_" else "b"
            opp_side = "b" if side == "a" else "a"

            record = {
                "fighter": fighter,
                "fighter_id": row.get(f"fighter_{side}_id"),
                "fighter_key": row.get(f"__fighter_{side}_key") or fighter_identity_key(
                    fighter, row.get(f"fighter_{side}_id")
                ),
                "opponent": row.get(opp_col, ""),
                "opponent_id": row.get(f"fighter_{opp_side}_id"),
                "opponent_key": row.get(f"__fighter_{opp_side}_key") or fighter_identity_key(
                    row.get(opp_col, ""), row.get(f"fighter_{opp_side}_id")
                ),
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

            # Extended stats: position/target shares (scraped by ufc_refresh)
            for stat in EXTENDED_STAT_COLUMNS:
                if stat in ("distance_acc", "clinch_acc", "ground_acc"):
                    continue  # computed below
                col = f"{prefix}{stat}"
                opp_col_stat = f"{opp_prefix}{stat}"
                record[stat] = row.get(col, np.nan)
                record[f"opp_{stat}"] = row.get(opp_col_stat, np.nan)

            # Derive per-fight position accuracies (landed / attempted)
            for pos in ("distance", "clinch", "ground"):
                landed = row.get(f"{prefix}{pos}_landed", np.nan)
                attempted = row.get(f"{prefix}{pos}_attempted", np.nan)
                if pd.notna(landed) and pd.notna(attempted) and attempted > 0:
                    record[f"{pos}_acc"] = landed / attempted
                else:
                    record[f"{pos}_acc"] = np.nan
                # Opponent side
                opp_landed = row.get(f"{opp_prefix}{pos}_landed", np.nan)
                opp_attempted = row.get(f"{opp_prefix}{pos}_attempted", np.nan)
                if pd.notna(opp_landed) and pd.notna(opp_attempted) and opp_attempted > 0:
                    record[f"opp_{pos}_acc"] = opp_landed / opp_attempted
                else:
                    record[f"opp_{pos}_acc"] = np.nan

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

    all_base = STAT_COLUMNS + EXTENDED_STAT_COLUMNS
    stats_to_roll = all_base + [f"opp_{s}" for s in all_base] + ["won"]

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
    fighter_fights["days_since_last_fight"] = dates.diff().dt.days

    return fighter_fights


def _compute_ufc_history_frame(per_fight_df: pd.DataFrame) -> pd.DataFrame:
    """Compute UFC-only counters/streaks aligned to UFC fight rows."""
    history_cols = [
        "current_win_streak",
        "num_fights",
        "prior_wins",
        "prior_losses",
        "prior_draws",
        "prior_lose_streak",
        "prior_longest_win_streak",
        "prior_total_rounds",
        "prior_title_bouts",
        "prior_wins_ko",
        "prior_wins_sub",
        "prior_wins_dec",
    ]
    identity_col = _stateful_identity_column(per_fight_df)
    if per_fight_df.empty:
        return pd.DataFrame(columns=[identity_col, "event_date", *history_cols])

    history_frames = []
    for _fighter, group in per_fight_df.groupby(identity_col):
        group = group.sort_values("event_date").copy()
        (
            group["current_win_streak"],
            group["prior_wins"],
            group["prior_losses"],
            group["prior_draws"],
            group["prior_lose_streak"],
            group["prior_longest_win_streak"],
            group["prior_total_rounds"],
            group["prior_title_bouts"],
            group["prior_wins_ko"],
            group["prior_wins_sub"],
            group["prior_wins_dec"],
        ) = _career_history_columns(group)
        group["num_fights"] = range(len(group))
        history_frames.append(group[[identity_col, "event_date", *history_cols]])

    return pd.concat(history_frames, ignore_index=True)


def _overlay_ufc_history_backed_fields(
    features: pd.DataFrame,
    ufc_per_fight: pd.DataFrame,
) -> pd.DataFrame:
    """Restore UFC-only counters after pre-UFC rows seed rolling stats."""
    history_frame = _compute_ufc_history_frame(ufc_per_fight)
    if history_frame.empty:
        return features

    override_cols = [
        "current_win_streak",
        "num_fights",
        *_HISTORY_BACKED_FIGHTER_FIELDS.values(),
    ]

    updated = features
    history_identity_col = _stateful_identity_column(history_frame)
    for prefix, side in [("a_", "a"), ("b_", "b")]:
        fighter_col = f"fighter_{side}"
        merge_key = f"__fighter_{side}_key" if history_identity_col == "fighter_key" else fighter_col
        existing_cols = [f"{prefix}{col}" for col in override_cols if f"{prefix}{col}" in updated.columns]
        if existing_cols:
            updated = updated.drop(columns=existing_cols)

        renamed = history_frame.rename(
            columns={
                history_identity_col: merge_key,
                **{col: f"{prefix}{col}" for col in override_cols},
            }
        )
        merge_cols = [merge_key, "event_date", *[f"{prefix}{col}" for col in override_cols]]
        updated = updated.merge(renamed[merge_cols], on=[merge_key, "event_date"], how="left")

    return updated


def _loss_method_history_frame(per_fight_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-fighter loss-method decomposition aligned to UFC fight rows (E13).

    For each fighter-fight row, the PRE-fight state of:
    - prior_losses_ko / prior_losses_sub: career losses by KO/TKO and by
      submission (the contract's loss branch counts undifferentiated losses)
    - prior_recent_ko_loss: 1.0 when either of the last two fights was a
      KO/TKO loss, 0.0 otherwise, NaN before the first fight
    """
    identity_col = _stateful_identity_column(per_fight_df)
    if per_fight_df.empty:
        return pd.DataFrame(
            columns=[
                identity_col, "event_date",
                "prior_losses_ko", "prior_losses_sub", "prior_recent_ko_loss",
            ]
        )

    frames = []
    for fighter, group in per_fight_df.groupby(identity_col):
        group = group.sort_values("event_date")
        losses_ko = losses_sub = 0
        ko_loss_flags: list[bool] = []
        records = []
        for _, row in group.iterrows():
            recent = (
                float(any(ko_loss_flags[-2:])) if ko_loss_flags else np.nan
            )
            records.append({
                identity_col: fighter,
                "event_date": row["event_date"],
                "prior_losses_ko": losses_ko,
                "prior_losses_sub": losses_sub,
                "prior_recent_ko_loss": recent,
            })
            is_ko_loss = False
            if row.get("result_label") == "loss":
                method_group = _method_group(row.get("method"))
                if method_group == "ko":
                    losses_ko += 1
                    is_ko_loss = True
                elif method_group == "sub":
                    losses_sub += 1
            ko_loss_flags.append(is_ko_loss)
        frames.append(pd.DataFrame(records))

    return pd.concat(frames, ignore_index=True)


def _add_loss_method_features(
    features: pd.DataFrame,
    ufc_per_fight: pd.DataFrame,
) -> pd.DataFrame:
    """Merge loss-method decomposition features for both fighters (E13)."""
    history = _loss_method_history_frame(ufc_per_fight)
    if history.empty:
        return features

    history_identity_col = _stateful_identity_column(history)
    for prefix, side in [("a_", "a"), ("b_", "b")]:
        fighter_col = f"fighter_{side}"
        merge_key = f"__fighter_{side}_key" if history_identity_col == "fighter_key" else fighter_col
        renamed = history.rename(columns={
            history_identity_col: merge_key,
            "prior_losses_ko": f"{prefix}losses_ko",
            "prior_losses_sub": f"{prefix}losses_sub",
            "prior_recent_ko_loss": f"{prefix}recent_ko_loss",
        })
        renamed = renamed.drop_duplicates(subset=[merge_key, "event_date"], keep="last")
        features = features.merge(
            renamed,
            on=[merge_key, "event_date"],
            how="left",
        )

    for prefix in ("a_", "b_"):
        losses = pd.to_numeric(features.get(f"{prefix}losses"), errors="coerce")
        denominator = losses.replace(0, np.nan)
        features[f"{prefix}loss_ko_rate"] = (
            pd.to_numeric(features[f"{prefix}losses_ko"], errors="coerce") / denominator
        )
        features[f"{prefix}loss_sub_rate"] = (
            pd.to_numeric(features[f"{prefix}losses_sub"], errors="coerce") / denominator
        )

    for stat in ("loss_ko_rate", "loss_sub_rate", "recent_ko_loss"):
        a_col, b_col = f"a_{stat}", f"b_{stat}"
        if a_col in features.columns and b_col in features.columns:
            features[f"diff_{stat}"] = features[a_col] - features[b_col]

    logger.info("Added loss-method decomposition features (E13)")
    return features


def _compute_strength_of_schedule(
    rolling_df: pd.DataFrame, window: int = 5
) -> pd.DataFrame:
    """
    Compute strength of schedule: recency-weighted average of past opponents'
    rolling win rates at the time of each fight.

    For each fighter-fight row, looks back at the previous *window* opponents
    and averages their roll_won values with linear recency weighting
    (most recent opponent gets the highest weight).

    Adds an ``opp_strength`` column to *rolling_df*.
    """
    if "roll_won" not in rolling_df.columns:
        return rolling_df

    rolling_df = rolling_df.copy()

    # Lookup: (fighter, event_date) → roll_won at that fight
    identity_col = _stateful_identity_column(rolling_df)
    opponent_col = "opponent_key" if identity_col == "fighter_key" else "opponent"
    _won_rows = rolling_df[[identity_col, "event_date", "roll_won"]].dropna(
        subset=["roll_won"]
    )
    _lookup: dict[tuple, float] = {
        (r[identity_col], r["event_date"]): r["roll_won"]
        for _, r in _won_rows.iterrows()
    }

    sos_results = np.full(len(rolling_df), np.nan)

    for _fighter, group in rolling_df.groupby(identity_col):
        group = group.sort_values("event_date")
        past_opps: list[tuple[str, object]] = []

        for df_idx, row in group.iterrows():
            if past_opps:
                recent = past_opps[-window:]
                opp_wrs: list[float] = []
                weights: list[float] = []
                for rank, (opp, opp_date) in enumerate(recent, start=1):
                    wr = _lookup.get((opp, opp_date))
                    if wr is not None and not np.isnan(wr):
                        opp_wrs.append(wr)
                        # Linear recency weight: most recent = highest
                        weights.append(float(rank))
                if opp_wrs:
                    sos_results[df_idx] = float(np.average(opp_wrs, weights=weights))

            opp = row.get(opponent_col)
            if pd.notna(opp) and opp:
                past_opps.append((opp, row["event_date"]))

    rolling_df["opp_strength"] = sos_results
    non_null = np.count_nonzero(~np.isnan(sos_results))
    logger.info(
        f"Computed strength-of-schedule for {non_null}/{len(rolling_df)} "
        f"fighter-fight rows (window={window})"
    )
    return rolling_df


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
    prior_title_bouts: list[float] = []
    prior_wins_ko: list[int] = []
    prior_wins_sub: list[int] = []
    prior_wins_dec: list[int] = []

    wins = losses = draws = 0
    win_streak = lose_streak = longest_win_streak = 0
    total_rounds = 0.0
    title_bouts = 0
    title_bouts_unknown = False
    wins_ko = wins_sub = wins_dec = 0

    for _, row in fighter_fights.iterrows():
        current_win_streaks.append(win_streak)
        prior_wins.append(wins)
        prior_losses.append(losses)
        prior_draws.append(draws)
        prior_lose_streaks.append(lose_streak)
        prior_longest_win_streaks.append(longest_win_streak)
        prior_total_rounds.append(total_rounds)
        prior_title_bouts.append(
            np.nan if title_bouts_unknown else float(title_bouts)
        )
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

        title_flag = coerce_nullable_bool(row.get("title_bout"))
        if title_flag is None:
            title_bouts_unknown = True
        elif title_flag:
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


def _load_supplement_raw(path: Path) -> pd.DataFrame:
    """Load a supplement CSV in its raw fight-pair schema."""
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=["event_date"])
    except Exception as exc:
        logger.warning("Failed to load supplement %s: %s", path, exc)
        return pd.DataFrame()


def _supplement_rows_to_long_format(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw supplement rows to per-fighter long format."""
    if raw.empty:
        return pd.DataFrame()

    records: list[dict] = []
    ambiguous_name_keys = _training_ambiguous_name_keys()
    for _, row in raw.iterrows():
        date = row.get("event_date")
        winner = row.get("winner", "")
        method = row.get("method", "")
        finish_round = pd.to_numeric(
            pd.Series([row.get("finish_round")]), errors="coerce"
        ).iloc[0]
        title_bout = row.get("title_bout", np.nan)
        organization = row.get("organization", "")

        for prefix, fighter_col in [("a_", "fighter_a"), ("b_", "fighter_b")]:
            fighter = row.get(fighter_col)
            if pd.isna(fighter) or not fighter:
                continue

            opp_prefix = "b_" if prefix == "a_" else "a_"
            opp_col = "fighter_b" if prefix == "a_" else "fighter_a"

            record = {
                "fighter": fighter,
                "fighter_key": fighter_identity_key(
                    fighter, ambiguous_name_keys=ambiguous_name_keys
                ),
                "opponent": row.get(opp_col, ""),
                "opponent_key": fighter_identity_key(
                    row.get(opp_col, ""), ambiguous_name_keys=ambiguous_name_keys
                ),
                "event_date": date,
                "weight_class": row.get("weight_class", ""),
                "won": 1 if winner == fighter else 0,
                "result_label": (
                    "win" if winner == fighter
                    else "loss" if winner == row.get(opp_col, "")
                    else "draw"
                ),
                "method": method,
                "finish_round": finish_round,
                "title_bout": title_bout,
                "organization": organization,
            }

            # Per-fight stats are NaN for pre-UFC fights
            for stat in STAT_COLUMNS + EXTENDED_STAT_COLUMNS:
                col = f"{prefix}{stat}"
                opp_stat = f"{opp_prefix}{stat}"
                record[stat] = row.get(col, np.nan)
                record[f"opp_{stat}"] = row.get(opp_stat, np.nan)

            records.append(record)

    return pd.DataFrame(records)


def _drop_quarantined_supplement_identities(
    long_df: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    """Fail closed for known collisions in the legacy name-only supplements."""
    if long_df.empty or "fighter" not in long_df.columns:
        return long_df

    ambiguous_name_keys = _training_ambiguous_name_keys()
    quarantined_mask = long_df["fighter"].isin(_SUPPLEMENT_QUARANTINED_IDENTITIES) | long_df[
        "fighter"
    ].map(
        lambda name: fighter_identity_is_ambiguous(
            name, ambiguous_name_keys=ambiguous_name_keys
        )
    )
    if not quarantined_mask.any():
        return long_df

    quarantined_names = sorted(
        long_df.loc[quarantined_mask, "fighter"].unique()
    )
    logger.info(
        "Excluded %d known-ambiguous %s fighter rows; features remain NaN: %s",
        int(quarantined_mask.sum()),
        label,
        ", ".join(quarantined_names),
    )
    return long_df.loc[~quarantined_mask].copy()


def _reviewed_history_group_is_valid(
    group: pd.DataFrame,
    *,
    history_type: str,
    identity: dict[str, object],
) -> bool:
    expected = identity.get("reviewed_history", {}).get(history_type)
    if expected is None:
        return group.empty
    expected_total, expected_wins, expected_losses = expected
    if len(group) != expected_total:
        return False

    canonical_name = str(identity["canonical_name"])
    dob = str(identity["dob"])
    result = group["subject_result"].astype(str).str.strip().str.casefold()
    dates = pd.to_datetime(group["event_date"], errors="coerce", utc=True)
    expected_source_id = str(identity["sherdog_profile_id"])
    expected_source_url = str(identity["sherdog_url"])
    expected_winner = group["fighter_a"].where(result.eq("win"), group["fighter_b"])
    duplicate_key = pd.DataFrame(
        {
            "event_date": dates,
            "opponent": group["fighter_b"].astype(str),
        }
    ).duplicated()

    return bool(
        result.isin({"win", "loss"}).all()
        and int(result.eq("win").sum()) == expected_wins
        and int(result.eq("loss").sum()) == expected_losses
        and dates.notna().all()
        and not duplicate_key.any()
        and group["fighter_a"].eq(canonical_name).all()
        and group["fighter_b"].notna().all()
        and group["fighter_b"].astype(str).str.strip().ne("").all()
        and group["winner"].astype(str).eq(expected_winner.astype(str)).all()
        and group["ufcstats_url"].eq(identity["ufcstats_url"]).all()
        and group["subject_dob"].astype(str).eq(dob).all()
        and group["source"].astype(str).str.casefold().eq("sherdog").all()
        and group["source_profile_id"].astype(str).eq(expected_source_id).all()
        and group["source_profile_url"].eq(expected_source_url).all()
        and group["source_profile_dob"].astype(str).eq(dob).all()
        and group["dob_match"].astype(str).str.casefold().eq("true").all()
    )


def _reviewed_history_canonical_sha256(path: Path) -> str:
    """Hash exact reviewed evidence while treating LF and CRLF as equivalent."""
    text = path.read_text(encoding="utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_reviewed_supplement_history(history_type: str) -> pd.DataFrame:
    """Load one audited history type, rejecting any incomplete fighter group."""
    path = _resolve_reviewed_supplement_history_path()
    if not path.is_file():
        return pd.DataFrame()
    try:
        digest = _reviewed_history_canonical_sha256(path)
    except OSError as exc:
        logger.warning("Failed to hash reviewed fighter history %s: %s", path, exc)
        return pd.DataFrame()
    if digest != _REVIEWED_SUPPLEMENT_HISTORY_CANONICAL_SHA256:
        logger.warning(
            "Ignoring reviewed fighter history %s because its canonical SHA-256 "
            "does not match the approved 40-row evidence artifact",
            path,
        )
        return pd.DataFrame()
    raw = _load_supplement_raw(path)
    missing = sorted(_REVIEWED_HISTORY_REQUIRED_COLUMNS.difference(raw.columns))
    if missing:
        logger.warning(
            "Ignoring reviewed fighter history %s because columns are missing: %s",
            path,
            ", ".join(missing),
        )
        return pd.DataFrame()

    selected = raw[
        raw["history_type"].astype(str).str.strip().str.casefold()
        == history_type.casefold()
    ].copy()
    unknown_ids = set(selected["ufcstats_id"].dropna().astype(str)).difference(
        REVIEWED_FIGHTER_IDENTITIES
    )
    if unknown_ids:
        logger.warning(
            "Ignoring reviewed %s rows with unknown UFCStats identities: %s",
            history_type,
            ", ".join(sorted(unknown_ids)),
        )
        selected = selected[~selected["ufcstats_id"].astype(str).isin(unknown_ids)]

    records: list[dict[str, object]] = []
    ambiguous_name_keys = _training_ambiguous_name_keys()
    for ufcstats_id, identity in REVIEWED_FIGHTER_IDENTITIES.items():
        group = selected[selected["ufcstats_id"].astype(str) == ufcstats_id].copy()
        expected = identity.get("reviewed_history", {}).get(history_type)
        if expected is None:
            if not group.empty:
                logger.warning(
                    "Ignoring unapproved reviewed %s history for UFCStats %s",
                    history_type,
                    ufcstats_id,
                )
            continue
        if not _reviewed_history_group_is_valid(
            group,
            history_type=history_type,
            identity=identity,
        ):
            logger.warning(
                "Ignoring invalid reviewed %s history for UFCStats %s; "
                "features remain NaN",
                history_type,
                ufcstats_id,
            )
            continue

        canonical_name = str(identity["canonical_name"])
        for _, row in group.iterrows():
            result_label = str(row["subject_result"]).strip().casefold()
            record: dict[str, object] = {
                "fighter": canonical_name,
                "fighter_id": ufcstats_id,
                "fighter_key": fighter_identity_key(canonical_name, ufcstats_id),
                "opponent": row["fighter_b"],
                "opponent_key": fighter_identity_key(
                    row["fighter_b"], ambiguous_name_keys=ambiguous_name_keys
                ),
                "event_date": pd.to_datetime(row["event_date"], errors="coerce"),
                "weight_class": row.get("weight_class", ""),
                "won": 1 if result_label == "win" else 0,
                "result_label": result_label,
                "method": row.get("method", ""),
                "finish_round": pd.to_numeric(
                    pd.Series([row.get("finish_round")]), errors="coerce"
                ).iloc[0],
                "title_bout": row.get("title_bout", np.nan),
                "organization": row.get("organization", ""),
            }
            for stat in STAT_COLUMNS + EXTENDED_STAT_COLUMNS:
                record[stat] = np.nan
                record[f"opp_{stat}"] = np.nan
            records.append(record)

    reviewed = pd.DataFrame(records)
    if not reviewed.empty:
        logger.info(
            "Loaded %d stable-ID-reviewed %s fighter-history rows",
            len(reviewed),
            history_type,
        )
    return reviewed


def _is_tracked_ufc_organization(value: object) -> bool:
    """Return whether a supplement organization is a tracked UFC promotion.

    The check is intentionally exact after whitespace/case normalization. A
    broad ``"ufc" in value`` check would incorrectly reject unrelated regional
    promotions such as WUFC and UFCF.
    """
    if pd.isna(value):
        return False
    label = " ".join(str(value).replace("’", "'").strip().casefold().split())
    return label in _TRACKED_UFC_ORGANIZATION_LABELS


def _load_pre_ufc_supplement(
    path: Path,
    *,
    include_reviewed: bool = False,
) -> pd.DataFrame:
    """Load trustworthy, deduplicated pre-UFC rows in long format.

    Mirrored source rows describe the same physical bout and must be collapsed
    before conversion to two fighter-perspective rows. Clearly UFC-owned bouts
    are excluded because they already exist in the tracked UFC history and are
    not pre-UFC evidence.
    """
    raw = _load_supplement_raw(path)
    long_df = pd.DataFrame()
    if not raw.empty:
        missing_columns = sorted(_PRE_UFC_REQUIRED_COLUMNS.difference(raw.columns))
        if missing_columns:
            logger.warning(
                "Ignoring pre-UFC supplement %s because required columns are missing: %s",
                path,
                ", ".join(missing_columns),
            )
        else:
            deduped = _dedupe_supplement_rows(raw)
            if "organization" in deduped.columns:
                ufc_mask = deduped["organization"].map(_is_tracked_ufc_organization)
                deduped = deduped.loc[~ufc_mask].copy()

            long_df = _supplement_rows_to_long_format(deduped)
            long_df = _drop_quarantined_supplement_identities(
                long_df,
                label="pre-UFC",
            )
            if not long_df.empty:
                # An unparseable date can never be proven to precede a modeled
                # fight. Drop it rather than assigning an artificial value.
                long_df["event_date"] = pd.to_datetime(
                    long_df["event_date"], errors="coerce", utc=True
                ).dt.tz_localize(None)
                long_df = long_df.dropna(subset=["event_date"]).reset_index(drop=True)
    if include_reviewed:
        reviewed = _load_reviewed_supplement_history("professional")
        if not reviewed.empty:
            long_df = pd.concat([long_df, reviewed], ignore_index=True)
    return long_df


def _load_amateur_supplement(
    path: Path,
    *,
    include_reviewed: bool = False,
) -> pd.DataFrame:
    """Load legacy amateur rows safely, then append reviewed subject histories."""
    raw = _load_supplement_raw(path)
    if raw.empty:
        legacy_long = pd.DataFrame()
    else:
        legacy_long = _supplement_rows_to_long_format(
            _dedupe_supplement_rows(raw)
        )
        legacy_long = _drop_quarantined_supplement_identities(
            legacy_long,
            label="amateur",
        )

    if include_reviewed:
        reviewed = _load_reviewed_supplement_history("amateur")
        if not reviewed.empty:
            legacy_long = pd.concat([legacy_long, reviewed], ignore_index=True)
    return legacy_long.reset_index(drop=True)


def _tracked_ufc_debut_dates(tracked_fights: pd.DataFrame) -> pd.Series:
    """Return each exact-spelling fighter's first tracked UFC event date.

    ``tracked_fights`` may be either the raw two-fighter fight schema or the
    internal long schema. Identity matching is exact on purpose: a fuzzy or
    suffix-insensitive match can merge two people and is less safe than leaving
    the corresponding feature unknown.
    """
    if tracked_fights.empty or "event_date" not in tracked_fights.columns:
        return pd.Series(dtype="datetime64[ns]")

    identity_frames: list[pd.DataFrame] = []
    if "fighter" in tracked_fights.columns:
        identity_frames.append(tracked_fights[["fighter", "event_date"]].copy())
    else:
        for fighter_col in ("fighter_a", "fighter_b"):
            if fighter_col in tracked_fights.columns:
                identity_frames.append(
                    tracked_fights[[fighter_col, "event_date"]].rename(
                        columns={fighter_col: "fighter"}
                    )
                )

    if not identity_frames:
        return pd.Series(dtype="datetime64[ns]")

    identities = pd.concat(identity_frames, ignore_index=True)
    identities["event_date"] = pd.to_datetime(
        identities["event_date"], errors="coerce", utc=True
    ).dt.tz_localize(None).dt.normalize()
    identities = identities.dropna(subset=["fighter", "event_date"])
    identities["fighter"] = identities["fighter"].astype(str)
    identities = identities[identities["fighter"].str.strip().ne("")]
    if identities.empty:
        return pd.Series(dtype="datetime64[ns]")
    return identities.groupby("fighter", sort=False)["event_date"].min()


def _filter_pre_ufc_rows_before_tracked_debut(
    supplement_df: pd.DataFrame,
    tracked_fights: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only dated supplement rows strictly before each tracked debut.

    The debut is the earliest modeled UFC row for the exact fighter spelling.
    Therefore ``supplement_date < debut_date`` also guarantees that the row is
    strictly earlier than every historical modeled fight for that fighter. A
    missing/ambiguous identity or date fails closed to no rows, which later
    materializes as NaN rather than a fabricated zero.
    """
    if supplement_df.empty:
        return supplement_df.copy()
    required = {"fighter", "event_date"}
    if not required.issubset(supplement_df.columns):
        return supplement_df.iloc[0:0].copy()

    debut_dates = _tracked_ufc_debut_dates(tracked_fights)
    if debut_dates.empty:
        return supplement_df.iloc[0:0].copy()

    filtered = supplement_df.copy()
    filtered["event_date"] = pd.to_datetime(
        filtered["event_date"], errors="coerce", utc=True
    ).dt.tz_localize(None).dt.normalize()
    filtered["__tracked_ufc_debut"] = filtered["fighter"].map(debut_dates)
    valid = (
        filtered["event_date"].notna()
        & filtered["__tracked_ufc_debut"].notna()
        & (filtered["event_date"] < filtered["__tracked_ufc_debut"])
    )
    return (
        filtered.loc[valid]
        .drop(columns="__tracked_ufc_debut")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Org-tier mapping for pre-UFC promotions
# ---------------------------------------------------------------------------

# Tier 1: Major international promotions — records here carry significant weight
_ORG_TIER_1 = {
    "bellator", "bellator mma", "one championship", "one fc", "pfl",
    "professional fighters league", "strikeforce", "pride", "pride fc",
    "wec", "world extreme cagefighting", "affliction", "dream",
    "invicta", "invicta fc", "rizin", "rizin ff", "ksw",
}

# Tier 2: Established feeder promotions — meaningful but less predictive
_ORG_TIER_2 = {
    "cage warriors", "cage warriors fighting championship", "lfa",
    "legacy fighting alliance", "bamma", "cffc", "combate global",
    "combate americas", "titan fc", "titan fighting championships",
    "brave cf", "brave combat federation", "ares", "m-1 global",
    "road fc", "road fighting championship", "shooto", "pancrase",
    "deep", "jungle fight", "fury fc", "cfa", "efn",
    "resurrection fighting alliance", "rfa", "legacy fc",
    "world series of fighting", "wsof",
}

# Everything else is tier 3 (regional / unknown)


def _encode_org_tier(org_name: str) -> float:
    """Map organization name to tier (1=major, 2=feeder, 3=regional/unknown).

    Returns NaN if org_name is empty or missing — never fabricates.
    """
    if not org_name or (isinstance(org_name, float) and np.isnan(org_name)):
        return np.nan
    name_lower = str(org_name).strip().lower()
    if not name_lower:
        return np.nan
    if name_lower in _ORG_TIER_1:
        return 1.0
    if name_lower in _ORG_TIER_2:
        return 2.0
    return 3.0


def _compute_pre_ufc_summary(supplement_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-fighter pre-UFC career summary features.

    Returns a DataFrame with one row per fighter, columns:
        fighter, pre_ufc_total_fights, pre_ufc_wins, pre_ufc_losses,
        pre_ufc_win_pct, pre_ufc_ko_rate, pre_ufc_sub_rate,
        pre_ufc_dec_rate, pre_ufc_org_tier_best (best org tier fought in).

    All values derived from real data or NaN. No estimates.
    """
    if supplement_df.empty:
        return pd.DataFrame()

    summaries: list[dict] = []
    for fighter, group in supplement_df.groupby("fighter"):
        total = len(group)
        wins = (group["won"] == 1).sum()
        losses = (group["result_label"] == "loss").sum()

        # Method rates — from real method strings, NaN if 0 wins
        if wins > 0:
            methods = group.loc[group["won"] == 1, "method"].str.lower().fillna("")
            ko_wins = methods.str.contains("ko|tko|punch|kick|knee|elbow|slam|stomp", regex=True).sum()
            sub_wins = methods.str.contains("sub|submission|choke|armbar|triangle|guillotine|rear.naked|kimura|americana|heel.hook|ankle.lock|arm.triangle|darce|anaconda|twister|calf.slicer|neck.crank", regex=True).sum()
            dec_wins = methods.str.contains("dec|decision|unanimous|split|majority", regex=True).sum()
            ko_rate = float(ko_wins) / wins
            sub_rate = float(sub_wins) / wins
            dec_rate = float(dec_wins) / wins
        else:
            ko_rate = np.nan
            sub_rate = np.nan
            dec_rate = np.nan

        win_pct = wins / total if total > 0 else np.nan

        # Best org tier — lowest number = highest tier promotion fought in
        if "organization" in group.columns:
            org_tiers = group["organization"].apply(_encode_org_tier).dropna()
            best_tier = org_tiers.min() if not org_tiers.empty else np.nan
        else:
            best_tier = np.nan

        summaries.append({
            "fighter": fighter,
            "pre_ufc_total_fights": total,
            "pre_ufc_wins": wins,
            "pre_ufc_losses": losses,
            "pre_ufc_win_pct": win_pct,
            "pre_ufc_ko_rate": ko_rate,
            "pre_ufc_sub_rate": sub_rate,
            "pre_ufc_dec_rate": dec_rate,
            "pre_ufc_org_tier_best": best_tier,
        })

    return pd.DataFrame(summaries)


def _compute_amateur_summary(supplement_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-fighter amateur career summary features from raw supplement rows."""
    if supplement_df.empty:
        return pd.DataFrame()

    if {"fighter", "won", "result_label"}.issubset(supplement_df.columns):
        long_df = supplement_df.copy()
    else:
        deduped_raw = _dedupe_supplement_rows(supplement_df)
        long_df = _supplement_rows_to_long_format(deduped_raw)
        long_df = _drop_quarantined_supplement_identities(
            long_df,
            label="amateur",
        )
    if long_df.empty:
        return pd.DataFrame()

    summaries: list[dict] = []
    for fighter, group in long_df.groupby("fighter"):
        total = len(group)
        wins = (group["won"] == 1).sum()
        losses = (group["result_label"] == "loss").sum()

        if wins > 0:
            methods = group.loc[group["won"] == 1, "method"].str.lower().fillna("")
            ko_wins = methods.str.contains("ko|tko|punch|kick|knee|elbow|slam|stomp", regex=True).sum()
            sub_wins = methods.str.contains("sub|submission|choke|armbar|triangle|guillotine|rear.naked|kimura|americana|heel.hook|ankle.lock|arm.triangle|darce|anaconda|twister|calf.slicer|neck.crank", regex=True).sum()
            dec_wins = methods.str.contains("dec|decision|unanimous|split|majority", regex=True).sum()
            ko_rate = float(ko_wins) / wins
            sub_rate = float(sub_wins) / wins
            dec_rate = float(dec_wins) / wins
        else:
            ko_rate = np.nan
            sub_rate = np.nan
            dec_rate = np.nan

        summaries.append(
            {
                "fighter": fighter,
                "amateur_total_fights": total,
                "amateur_wins": wins,
                "amateur_losses": losses,
                "amateur_win_pct": wins / total if total > 0 else np.nan,
                "amateur_ko_rate": ko_rate,
                "amateur_sub_rate": sub_rate,
                "amateur_dec_rate": dec_rate,
            }
        )

    return pd.DataFrame(summaries)


def _normalize_binary_probability_pair(
    frame: pd.DataFrame,
    a_col: str,
    b_col: str,
    *,
    diff_col: str | None = None,
) -> None:
    """Normalize two observed implied probabilities to no-vig fair probabilities."""
    if a_col not in frame.columns or b_col not in frame.columns:
        return

    a_prob = pd.to_numeric(frame[a_col], errors="coerce")
    b_prob = pd.to_numeric(frame[b_col], errors="coerce")
    total = a_prob + b_prob
    valid = a_prob.notna() & b_prob.notna() & (a_prob > 0) & (b_prob > 0) & (total > 0)

    normalized_a = pd.Series(np.nan, index=frame.index, dtype=float)
    normalized_b = pd.Series(np.nan, index=frame.index, dtype=float)
    normalized_a.loc[valid] = a_prob.loc[valid] / total.loc[valid]
    normalized_b.loc[valid] = b_prob.loc[valid] / total.loc[valid]
    frame[a_col] = normalized_a
    frame[b_col] = normalized_b

    if diff_col is not None:
        frame[diff_col] = frame[a_col] - frame[b_col]


# ---------------------------------------------------------------------------
# Main feature building
# ---------------------------------------------------------------------------

def build_features(fights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from a fights DataFrame.

    For each fight, computes:
    - Rolling averages (last N fights) for both fighters
    - Differentials (fighter_a - fighter_b) for all stats
    - Win streak, experience, days since last fight

    Returns a DataFrame ready for model training with no data leakage.
    """
    logger.info(f"Building features for {len(fights_df)} fights")

    # Ensure sorted by date
    fights_df = _attach_fighter_identity_keys(
        fights_df.sort_values("event_date").reset_index(drop=True)
    )

    # Step 1: Compute per-fight stats in long format
    ufc_per_fight = _compute_per_fight_stats(fights_df)
    per_fight = ufc_per_fight.copy()
    logger.info(f"Extracted {len(per_fight)} fighter-fight records")

    # Step 1b: Merge pre-UFC career supplement (Sherdog) if available.
    # These rows enrich rolling stats / career counters for debut fighters
    # but are filtered out of the final output (we only return UFC fight rows).
    pre_ufc_path = _resolve_pre_ufc_supplement_path()
    _pre_ufc_marker = "__is_pre_ufc"
    pre_ufc_supplement = pd.DataFrame()
    loaded_supplement = _load_pre_ufc_supplement(
        pre_ufc_path,
        include_reviewed=True,
    )
    pre_ufc_supplement = _filter_pre_ufc_rows_before_tracked_debut(
        loaded_supplement,
        ufc_per_fight,
    )
    if not pre_ufc_supplement.empty:
        supplement = pre_ufc_supplement.copy()
        supplement[_pre_ufc_marker] = True
        per_fight[_pre_ufc_marker] = False
        per_fight = pd.concat([supplement, per_fight], ignore_index=True)
        logger.info(
            "Merged %d point-in-time pre-UFC career rows "
            "(fighters enriched: %d)",
            len(supplement),
            supplement["fighter"].nunique(),
        )

    # Step 2: Compute rolling stats per fighter
    all_rolling = []
    rolling_identity_col = _stateful_identity_column(per_fight)
    for _fighter, group in per_fight.groupby(rolling_identity_col):
        rolled = _compute_rolling_stats(group)
        all_rolling.append(rolled)

    rolling_df = pd.concat(all_rolling, ignore_index=True)

    # Filter out pre-UFC supplement rows — they were only needed to seed
    # rolling stats / career counters for debut fighters.
    if _pre_ufc_marker in rolling_df.columns:
        pre_ufc_count = rolling_df[_pre_ufc_marker].sum()
        rolling_df = rolling_df[~rolling_df[_pre_ufc_marker]].reset_index(drop=True)
        rolling_df = rolling_df.drop(columns=[_pre_ufc_marker], errors="ignore")
        if pre_ufc_count:
            logger.info(f"Filtered out {int(pre_ufc_count)} pre-UFC rows after rolling stats computation")

    # Step 2b: Compute strength of schedule (cross-fighter lookups on roll_won)
    rolling_df = _compute_strength_of_schedule(rolling_df)

    fights_df = fights_df.copy()

    # Step 3: Merge rolling stats back into fights DataFrame
    # For fighter_a
    rolling_helper_cols = {
        "fighter",
        "fighter_id",
        "fighter_key",
        "opponent",
        "opponent_id",
        "opponent_key",
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
    rolling_merge_key = _stateful_identity_column(rolling_df)
    a_merge_cols = [rolling_merge_key, "event_date"] + [
        c for c in a_rolling.columns
        if c.startswith("a_roll_") or c in ["current_win_streak", "num_fights", "days_since_last_fight"]
    ]
    # Rename non-prefixed columns for fighter A
    rename_map = {}
    for c in ["current_win_streak", "num_fights", "days_since_last_fight"]:
        if c in a_rolling.columns:
            rename_map[c] = f"a_{c}"
    a_rolling = a_rolling.rename(columns=rename_map)
    a_merge_cols = [rolling_merge_key, "event_date"] + [
        c for c in a_rolling.columns if c.startswith("a_")
    ]
    a_rolling_deduped = a_rolling[
        [c for c in a_merge_cols if c in a_rolling.columns]
    ].drop_duplicates(subset=[rolling_merge_key, "event_date"], keep="last")

    # For fighter_b
    b_rolling = rolling_df.copy()
    rename_b = {
        c: f"b_roll_{c.replace('roll_', '')}" if c.startswith("roll_") else f"b_{c}"
        for c in rolling_df.columns
        if c not in rolling_helper_cols
    }
    b_rolling = b_rolling.rename(columns=rename_b)
    b_merge_cols = [rolling_merge_key, "event_date"] + [
        c for c in b_rolling.columns if c.startswith("b_")
    ]
    b_rolling_deduped = b_rolling[
        [c for c in b_merge_cols if c in b_rolling.columns]
    ].drop_duplicates(subset=[rolling_merge_key, "event_date"], keep="last")

    # Merge fighter A rolling stats
    features = fights_df.merge(
        a_rolling_deduped,
        left_on=["__fighter_a_key", "event_date"],
        right_on=[rolling_merge_key, "event_date"],
        how="left",
    ).drop(columns=[rolling_merge_key], errors="ignore")

    # Merge fighter B rolling stats
    features = features.merge(
        b_rolling_deduped,
        left_on=["__fighter_b_key", "event_date"],
        right_on=[rolling_merge_key, "event_date"],
        how="left",
    ).drop(columns=[rolling_merge_key], errors="ignore")

    features = _overlay_ufc_history_backed_fields(features, ufc_per_fight)
    _fill_history_backed_fighter_fields(features)
    features = _add_loss_method_features(features, ufc_per_fight)

    # Step 5: Compute differentials
    diff_stats = [
        "roll_slpm", "roll_sapm", "roll_str_acc", "roll_str_def",
        "roll_td_avg", "roll_td_acc", "roll_td_def", "roll_sub_avg",
        "roll_sig_str_landed", "roll_td_landed", "roll_kd",
        "roll_won", "current_win_streak", "num_fights", "days_since_last_fight",
        # Extended rolling stats (position/target shares + accuracies)
        "roll_head_str_share", "roll_body_str_share", "roll_leg_str_share",
        "roll_distance_str_share", "roll_clinch_str_share", "roll_ground_str_share",
        "roll_control_share", "roll_control_per_td",
        "roll_distance_acc", "roll_clinch_acc", "roll_ground_acc",
        # Defensive vulnerability: what opponents are allowed to do (E10)
        "roll_opp_td_landed", "roll_opp_td_attempted",
        "roll_opp_ctrl_seconds", "roll_opp_sub_att",
        # Strength of schedule (computed in Step 2b)
        "opp_strength",
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
        features["a_stance_enc"] = pd.to_numeric(features["a_stance"].map(encode_stance), errors="coerce")
        features["b_stance_enc"] = pd.to_numeric(features["b_stance"].map(encode_stance), errors="coerce")
        features["same_stance"] = np.where(
            features["a_stance_enc"].notna() & features["b_stance_enc"].notna(),
            (features["a_stance_enc"] == features["b_stance_enc"]).astype(float),
            np.nan,
        )

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
            feature_col = f"{prefix}implied_prob"
            if feature_col in features.columns:
                existing = pd.to_numeric(features[feature_col], errors="coerce")
                features[feature_col] = prob.combine_first(existing)
            else:
                features[feature_col] = prob

    _normalize_binary_probability_pair(
        features,
        "a_implied_prob",
        "b_implied_prob",
        diff_col="diff_implied_prob",
    )

    # Opening odds → implied probability (for dual-baseline evaluation)
    for prefix in ["a_", "b_"]:
        opening_col = f"{prefix}opening_odds"
        if opening_col in features.columns:
            odds = features[opening_col].copy()
            pos_mask = odds > 0
            neg_mask = odds < 0
            prob = pd.Series(np.nan, index=features.index)
            prob[pos_mask] = 100 / (odds[pos_mask] + 100)
            prob[neg_mask] = (-odds[neg_mask]) / (-odds[neg_mask] + 100)
            features[f"{prefix}opening_implied_prob"] = prob

    _normalize_binary_probability_pair(
        features,
        "a_opening_implied_prob",
        "b_opening_implied_prob",
        diff_col="diff_opening_implied_prob",
    )

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
                feature_col = f"{prefix}{odds_type}_prob"
                if feature_col in features.columns:
                    existing = pd.to_numeric(features[feature_col], errors="coerce")
                    features[feature_col] = prob.combine_first(existing)
                else:
                    features[feature_col] = prob

    # Win record ratio (wins / (wins + losses)) — overall quality
    for prefix in ["a_", "b_"]:
        w_col = f"{prefix}wins"
        l_col = f"{prefix}losses"
        if w_col in features.columns and l_col in features.columns:
            total = features[w_col] + features[l_col]
            features[f"{prefix}win_pct"] = features[w_col] / total.replace(0, np.nan)

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
                (v for k, v in wc_order.items() if k.lower() in str(x).lower()), float("nan")
            )
        )

    # --- Cage rust indicator ---
    # Fighters returning after long layoffs (>365 days) historically underperform
    for prefix in ["a_", "b_"]:
        dslf_col = f"{prefix}days_since_last_fight"
        if dslf_col in features.columns:
            dslf = features[dslf_col]
            features[f"{prefix}cage_rust"] = np.where(dslf.isna(), np.nan, (dslf > 365).astype(float))
            # Log-scaled layoff (diminishing impact of very long layoffs)
            features[f"{prefix}layoff_log"] = np.log1p(dslf)
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
                features[ko_col] *
                (1.0 - features[str_def_col] / 100.0)
            )

        # Grappler advantage: attacker sub rate * (1 - defender TD defense)
        if sub_col in features.columns and td_def_col in features.columns:
            features[f"{prefix_atk}grappler_edge"] = (
                features[sub_col] *
                (1.0 - features[td_def_col] / 100.0)
            )

    # Style matchup differentials
    for feat in ["striker_edge", "grappler_edge"]:
        a_col = f"a_{feat}"
        b_col = f"b_{feat}"
        if a_col in features.columns and b_col in features.columns:
            features[f"diff_{feat}"] = features[a_col] - features[b_col]

    # Step 6a: Add rematch / H2H features
    features = features.sort_values("event_date")
    h2h: dict[tuple[str, str], list[str]] = {}
    is_rematch_vals = []
    h2h_diff_vals = []
    for _, row in features.iterrows():
        fa = str(row.get("__fighter_a_key", ""))
        fb = str(row.get("__fighter_b_key", ""))
        pair = tuple(sorted([fa, fb]))
        prior = h2h.get(pair, [])
        if prior:
            is_rematch_vals.append(1)
            h2h_diff_vals.append(sum(1 for w in prior if w == fa) - sum(1 for w in prior if w == fb))
        else:
            is_rematch_vals.append(0)
            h2h_diff_vals.append(0)
        if pair not in h2h:
            h2h[pair] = []
        winner = row.get("winner", "")
        if winner == row.get("fighter_a"):
            h2h[pair].append(fa)
        elif winner == row.get("fighter_b"):
            h2h[pair].append(fb)
    features["is_rematch"] = is_rematch_vals
    features["h2h_record_diff"] = h2h_diff_vals

    # Step 6b: Add experimental features (fight pace, cage time efficiency, quality-adjusted stats)
    from src.features.experimental_features import add_experimental_features
    features = add_experimental_features(features)

    # Step 6c: Add point-in-time pre-UFC career summary features. The same
    # debut-bounded rows seeded rolling history above, so the aggregate cannot
    # see a future, same-day, or post-UFC supplement result.
    if not pre_ufc_supplement.empty:
        _pre_ufc_summary = _compute_pre_ufc_summary(pre_ufc_supplement)
        if not _pre_ufc_summary.empty:
            _pre_ufc_cols = [c for c in _pre_ufc_summary.columns if c != "fighter"]
            # Merge for fighter_a
            _a_summary = _pre_ufc_summary.rename(
                columns={c: f"a_{c}" for c in _pre_ufc_cols}
            )
            features = features.merge(
                _a_summary, left_on="fighter_a", right_on="fighter",
                how="left"
            ).drop(columns=["fighter"], errors="ignore")
            # Merge for fighter_b
            _b_summary = _pre_ufc_summary.rename(
                columns={c: f"b_{c}" for c in _pre_ufc_cols}
            )
            features = features.merge(
                _b_summary, left_on="fighter_b", right_on="fighter",
                how="left"
            ).drop(columns=["fighter"], errors="ignore")
            # Differentials
            for col in _pre_ufc_cols:
                a_c = f"a_{col}"
                b_c = f"b_{col}"
                if a_c in features.columns and b_c in features.columns:
                    features[f"diff_{col}"] = features[a_c] - features[b_c]
            logger.info(
                "Added %d point-in-time pre-UFC summary features "
                "(coverage: %.1f%% of fighter_a)",
                len(_pre_ufc_cols),
                features["a_pre_ufc_total_fights"].notna().mean() * 100.0,
            )

    # Feature absence is unknown, never a zero-fight synthetic observation.
    for col in _PRE_UFC_SUMMARY_COLUMNS:
        for prefix in ("a_", "b_", "diff_"):
            target = f"{prefix}{col}"
            if target not in features.columns:
                features[target] = np.nan

    amateur_path = _resolve_amateur_supplement_path()
    _amateur_summary = pd.DataFrame()
    _amateur_long = _load_amateur_supplement(
        amateur_path,
        include_reviewed=True,
    )
    if not _amateur_long.empty:
        _amateur_summary = _compute_amateur_summary(_amateur_long)

    if not _amateur_summary.empty:
        _amateur_cols = [c for c in _amateur_summary.columns if c != "fighter"]
        _a_amateur = _amateur_summary.rename(columns={c: f"a_{c}" for c in _amateur_cols})
        features = features.merge(
            _a_amateur,
            left_on="fighter_a",
            right_on="fighter",
            how="left",
        ).drop(columns=["fighter"], errors="ignore")
        _b_amateur = _amateur_summary.rename(columns={c: f"b_{c}" for c in _amateur_cols})
        features = features.merge(
            _b_amateur,
            left_on="fighter_b",
            right_on="fighter",
            how="left",
        ).drop(columns=["fighter"], errors="ignore")
        for col in _amateur_cols:
            a_c = f"a_{col}"
            b_c = f"b_{col}"
            if a_c in features.columns and b_c in features.columns:
                features[f"diff_{col}"] = features[a_c] - features[b_c]
        logger.info(
            "Added %d amateur summary features (coverage: %.1f%% of fighter_a)",
            len(_amateur_cols),
            features["a_amateur_total_fights"].notna().mean() * 100.0,
        )

    for col in _AMATEUR_SUMMARY_COLUMNS:
        for prefix in ("a_", "b_", "diff_"):
            target = f"{prefix}{col}"
            if target not in features.columns:
                features[target] = np.nan

    # Step 7: Drop rows with insufficient data (first fights for both fighters)
    features["has_data"] = features.get("a_num_fights", pd.Series(0)) + features.get("b_num_fights", pd.Series(0))

    logger.info(f"Built {len(features)} fight feature rows with {len(features.columns)} columns")
    return features.drop(columns=["__fighter_a_key", "__fighter_b_key"], errors="ignore")


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
    wc_move_a: dict[int, float] = {}
    wc_move_b: dict[int, float] = {}
    for idx, row in features.sort_values("event_date").iterrows():
        wc_w = _wc_to_weight(row.get("weight_class"))
        if wc_w is None:
            wc_move_a[idx] = np.nan
            wc_move_b[idx] = np.nan
            continue

        # Compute flag using prior mode BEFORE updating history
        fa = row.get("__fighter_a_key") or row.get("fighter_a")
        fb = row.get("__fighter_b_key") or row.get("fighter_b")
        fa_home = fighter_wc_history[fa].most_common(1)[0][0] if fa and fa in fighter_wc_history else None
        fb_home = fighter_wc_history[fb].most_common(1)[0][0] if fb and fb in fighter_wc_history else None
        wc_move_a[idx] = float(fa_home != wc_w) if fa_home is not None else np.nan
        wc_move_b[idx] = float(fb_home != wc_w) if fb_home is not None else np.nan

        # Now update history with this fight
        for side in ["a", "b"]:
            fighter = row.get(f"__fighter_{side}_key") or row.get(f"fighter_{side}")
            if fighter:
                if fighter not in fighter_wc_history:
                    fighter_wc_history[fighter] = Counter()
                fighter_wc_history[fighter][wc_w] += 1

    a_moving = [wc_move_a.get(idx, np.nan) for idx in features.index]
    b_moving = [wc_move_b.get(idx, np.nan) for idx in features.index]

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
            if c.startswith(f"{prefix}roll_")
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

    # Rematch / Head-to-Head (added by model lab variants)
    feature_cols += [c for c in features_df.columns
                     if c in ["is_rematch", "h2h_record_diff"]]

    # Experimental features (fight pace, cage time efficiency)
    for prefix in ["a_", "b_"]:
        feature_cols += [c for c in features_df.columns
                         if c in [f"{prefix}fight_pace", f"{prefix}ctrl_efficiency"]]

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

# Features that were previously refresh-only but are now promoted to v6.
# Kept as a reference — these names should NOT appear in any exclusion lists.
_PROMOTED_FROM_REFRESH_ONLY = [
    # Now computed via EXTENDED_STAT_COLUMNS rolling pipeline
    "a_roll_distance_acc", "b_roll_distance_acc",
    "a_roll_clinch_acc", "b_roll_clinch_acc",
    "a_roll_ground_acc", "b_roll_ground_acc",
    "a_roll_distance_str_share", "b_roll_distance_str_share",
    "a_roll_clinch_str_share", "b_roll_clinch_str_share",
    "a_roll_ground_str_share", "b_roll_ground_str_share",
    "a_roll_head_str_share", "b_roll_head_str_share",
    "a_roll_body_str_share", "b_roll_body_str_share",
    "a_roll_leg_str_share", "b_roll_leg_str_share",
    "a_roll_control_share", "b_roll_control_share",
    "a_roll_control_per_td", "b_roll_control_per_td",
    # Now computed via experimental_features.py / build_features.py
    "a_strikes_avoided_pct", "b_strikes_avoided_pct",
    "a_opp_strength", "b_opp_strength",  # SOS: rolling avg of past opponents' win rates
    "a_ko_absorption", "b_ko_absorption",
    "pace_mismatch",
]

# Remaining refresh-only features not yet promoted (need more work to derive)
UFCSTATS_REFRESH_ONLY_FEATURE_NAMES = [
    "a_roll_ctrl_per_minute",
    "a_roll_adj_win_rate",
    "a_opp_strength_recent_3",
    "a_roll_finish_rate",
    "a_roll_early_finish_rate",
    "a_roll_decision_rate",
    "b_roll_ctrl_per_minute",
    "b_roll_adj_win_rate",
    "b_opp_strength_recent_3",
    "b_roll_finish_rate",
    "b_roll_early_finish_rate",
    "b_roll_decision_rate",
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


MARKET_DERIVED_DENYLIST = set(ODDS_FEATURE_NAMES) | {
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
    "Rafael Cerquiera": "Rafael Cerqueira",
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

    mask_a = df["fighter_a"].fillna("").map(lambda value: same_person_name(fighter_name, value))
    mask_b = df["fighter_b"].fillna("").map(lambda value: same_person_name(fighter_name, value))
    return int(mask_a.sum() + mask_b.sum())


def save_features(features_df: pd.DataFrame, filename: str | Path = "features.csv") -> None:
    """Save feature matrix to processed data directory."""
    path = Path(filename)
    if not path.is_absolute():
        path = PROCESSED_DATA_DIR / path
    write_csv_atomically(features_df, path, refuse_empty=True)
    logger.info(f"Saved features to {path}")
