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
from src.data.name_utils import (
    KNOWN_AMBIGUOUS_FIGHTER_NAME_KEYS,
    REVIEWED_FIGHTER_IDENTITIES,
    derive_ambiguous_fighter_name_keys,
    fighter_identity_key,
    canonical_fighter_name_key,
    normalize_ufcstats_id,
    reviewed_spelling_binding_name,
    same_person_name,
)
from src.data.pre_ufc_scraper import PROVENANCE_COLUMNS, _bound_history_mask, _dedupe_supplement_rows, _has_observed_win_method
from src.features.stance_utils import encode_stance
from src.features.contract_transforms import (
    _coerce_nullable_binary,
    materialize_honest_context_features,
)


from src.data.selected_source_inputs import active_sources, selected_path, selected_cache

logger = logging.getLogger(__name__)


class UnresolvedTrainingFighterIdentityError(ValueError):
    """A collision-prone training row is missing its stable fighter ID."""


_training_ambiguous_name_keys_cache: tuple[
    Path,
    int,
    frozenset[str],
] | None = None


def _training_ambiguous_name_keys(
    raw_data_dir: Path | None = None,
) -> frozenset[str]:
    """Derive collision groups from the selected fighter inventory."""
    global _training_ambiguous_name_keys_cache
    raw_root = Path(raw_data_dir) if raw_data_dir is not None else RAW_DATA_DIR
    inventory_path = selected_path("profiles", raw_root / "ufc_fighters_scraped.csv")
    try:
        inventory_mtime_ns = inventory_path.stat().st_mtime_ns
    except OSError:
        if active_sources() is not None:
            raise
        return KNOWN_AMBIGUOUS_FIGHTER_NAME_KEYS
    inventory_cache_key = inventory_path.resolve(strict=False)
    if (
        _training_ambiguous_name_keys_cache is not None
        and _training_ambiguous_name_keys_cache[0] == inventory_cache_key
        and _training_ambiguous_name_keys_cache[1] == inventory_mtime_ns
    ):
        return _training_ambiguous_name_keys_cache[2]
    try:
        inventory = pd.read_csv(inventory_path, usecols=["name", "fighter_url"])
        derived = derive_ambiguous_fighter_name_keys(
            (
                {"name": row.name, "fighter_id": row.fighter_url}
                for row in inventory.itertuples(index=False)
            )
        )
    except (OSError, ValueError) as exc:
        if active_sources() is not None:
            raise
        logger.warning(
            "Could not derive fighter identity collisions from %s: %s",
            inventory_path,
            exc,
        )
        derived = frozenset()
    resolved = frozenset(set(KNOWN_AMBIGUOUS_FIGHTER_NAME_KEYS) | set(derived))
    _training_ambiguous_name_keys_cache = (
        inventory_cache_key,
        inventory_mtime_ns,
        resolved,
    )
    return resolved


def _attach_fighter_identity_keys(
    fights_df: pd.DataFrame,
    *,
    raw_data_dir: Path | None = None,
) -> pd.DataFrame:
    """Attach temporary stateful keys while preserving display names and IDs."""
    fights = fights_df.copy()
    unresolved: list[str] = []
    input_collision_rows = []
    for side in ("a", "b"):
        name_col = f"fighter_{side}"
        id_col = f"fighter_{side}_id"
        if name_col not in fights.columns or id_col not in fights.columns:
            continue
        input_collision_rows.extend(
            {"name": name, "fighter_id": fighter_id}
            for name, fighter_id in zip(fights[name_col], fights[id_col])
        )
    training_ambiguous_name_keys = (
        _training_ambiguous_name_keys()
        if raw_data_dir is None
        else _training_ambiguous_name_keys(raw_data_dir)
    )
    ambiguous_name_keys = frozenset(
        set(training_ambiguous_name_keys)
        | set(derive_ambiguous_fighter_name_keys(input_collision_rows))
    )
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
        "Bruno Silva",
        "Jean Silva",
        "Joey Gomez",
        "Kai Kamaka",
        "Kai Kamaka III",
        "Michael Davis",
        "Michael McDonald",
        "Mike Davis",
        "Mizuki",
        "Gabriel Santos",
        "Robert Sanchez",
        "Tony Johnson",
    }
)
_AMATEUR_SUPPLEMENT_QUARANTINED_IDENTITIES = frozenset({"Anthony Johnson"})
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

# The tracked UFCStats history also lists a small set of bouts from other
# promotions (WEC, Strikeforce and PRIDE were Zuffa-owned for part of their
# history, but their bouts are not UFC bouts and the tracked-organisation list
# above does not treat them as such). Those rows must not define a fighter's
# "UFC debut" for the pre-UFC supplement cutoff: Drew Dober's 2008 Strikeforce
# bout otherwise discards his entire 2009-2013 regional record. Exact
# lowercase event-name prefixes, deliberately narrow; the list covers every
# non-UFC event name in the tracked history (79 rows, all dated 2012-08-18 or
# earlier, none inside a modeled window).
_NON_UFC_TRACKED_EVENT_PREFIXES = (
    "strikeforce",
    "wec ",
    "pride",
    "meca ",
    "elitexc",
    "cage rage",
    "dream ",
    "shooto",
    "affliction",
    "rotr ",
    "gladiator fc",
)


def _is_non_ufc_tracked_event(value: object) -> bool:
    """True when a tracked event name belongs to a promotion not treated as UFC-owned."""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    label = " ".join(str(value).strip().casefold().split())
    return bool(label) and label.startswith(_NON_UFC_TRACKED_EVENT_PREFIXES)


def _resolve_pre_ufc_supplement_path(
    raw_data_dir: Path | None = None,
) -> Path:
    """Prefer the richer v2 pre-UFC supplement when it exists."""
    if active_sources() is not None:
        return selected_path("professional_history", "")
    raw_root = Path(raw_data_dir) if raw_data_dir is not None else RAW_DATA_DIR
    for filename in _PRE_UFC_SUPPLEMENT_CANDIDATES:
        candidate = raw_root / filename
        if candidate.exists():
            return candidate
    return raw_root / _PRE_UFC_SUPPLEMENT_CANDIDATES[0]


def _resolve_amateur_supplement_path(
    raw_data_dir: Path | None = None,
) -> Path:
    """Return the amateur career supplement path."""
    raw_root = Path(raw_data_dir) if raw_data_dir is not None else RAW_DATA_DIR
    return selected_path("amateur_history", raw_root / _AMATEUR_SUPPLEMENT_FILENAME)


def _resolve_reviewed_supplement_history_path(
    raw_data_dir: Path | None = None,
) -> Path:
    """Return the small stable-ID-backed supplemental history artifact."""
    raw_root = Path(raw_data_dir) if raw_data_dir is not None else RAW_DATA_DIR
    return selected_path("reviewed_history", raw_root / _REVIEWED_SUPPLEMENT_HISTORY_FILENAME)


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
                "fight_row_id": row.get("__fight_row_id"),
                "fighter": fighter,
                "fighter_id": row.get(f"fighter_{side}_id"),
                "fighter_key": (
                    row[f"__fighter_{side}_key"] if f"__fighter_{side}_key" in row
                    else fighter_identity_key(fighter, row.get(f"fighter_{side}_id"))
                ),
                "opponent": row.get(opp_col, ""),
                "opponent_id": row.get(f"fighter_{opp_side}_id"),
                "opponent_key": (
                    row[f"__fighter_{opp_side}_key"] if f"__fighter_{opp_side}_key" in row
                    else fighter_identity_key(row.get(opp_col, ""), row.get(f"fighter_{opp_side}_id"))
                ),
                "event_date": date,
                "weight_class": weight_class,
                "won": 1 if winner == fighter else 0,
                "result_label": _fight_result_label(winner, fighter, row.get(opp_col, "")),
                "method": row.get("method", ""),
                "finish_round": finish_round,
                "title_bout": title_bout,
                # Carried so the pre-UFC debut cutoff can tell UFC-owned events
                # from the few tracked non-UFC bouts; never a model feature.
                "event_name": row.get("event_name", ""),
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
    fighter_fights: pd.DataFrame, halflife: int = EWM_HALFLIFE, *, _apply_observation_cutoffs: bool = True,
) -> pd.DataFrame:
    """
    Compute exponentially weighted rolling averages for a single fighter's fight history.
    Recent fights are weighted more heavily (halflife=3 fights by default).
    Stats are computed from all fights BEFORE the current one (shift(1)).
    """
    retained = _retained_long_mask(fighter_fights)
    if _apply_observation_cutoffs and retained.any():
        # Recompute only the affected owner's timeline at each modeled cutoff.
        # Original bout dates stay intact; capture dates govern availability.
        indexed = fighter_fights.copy().reset_index(drop=True)
        indexed["__availability_row"] = indexed.index
        captures = pd.to_datetime(indexed.get("source_captured_at_utc"), errors="coerce", utc=True)
        outputs = []
        for index, row in indexed.loc[~retained.to_numpy()].iterrows():
            cutoff = pd.to_datetime(row["event_date"], errors="coerce", utc=True)
            available = ~retained.to_numpy() | (captures.notna() & (captures < cutoff))
            rolled = _compute_rolling_stats(indexed.loc[available], halflife, _apply_observation_cutoffs=False)
            outputs.append(rolled.loc[rolled["__availability_row"] == index])
        if not outputs:
            return _compute_rolling_stats(fighter_fights, halflife, _apply_observation_cutoffs=False)
        return pd.concat(outputs, ignore_index=True).drop(columns="__availability_row")
    sort_columns = ["event_date"]
    if "fight_row_id" in fighter_fights.columns:
        sort_columns.append("fight_row_id")
    fighter_fights = fighter_fights.sort_values(
        sort_columns,
        kind="stable",
    ).copy()

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
    row_id_columns = (
        ["fight_row_id"] if "fight_row_id" in per_fight_df.columns else []
    )
    if per_fight_df.empty:
        return pd.DataFrame(
            columns=[identity_col, *row_id_columns, "event_date", *history_cols]
        )

    history_frames = []
    for _fighter, group in per_fight_df.groupby(identity_col):
        group = group.sort_values(
            ["event_date", *row_id_columns],
            kind="stable",
        ).copy()
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
        history_frames.append(
            group[[identity_col, *row_id_columns, "event_date", *history_cols]]
        )

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
        merge_key = (
            f"__fighter_{side}_key"
            if history_identity_col == "fighter_key"
            else fighter_col
        )
        existing_cols = [f"{prefix}{col}" for col in override_cols if f"{prefix}{col}" in updated.columns]
        if existing_cols:
            updated = updated.drop(columns=existing_cols)

        renamed = history_frame.rename(
            columns={
                history_identity_col: merge_key,
                "fight_row_id": "__fight_row_id",
                **{col: f"{prefix}{col}" for col in override_cols},
            }
        )
        row_merge_cols = (
            ["__fight_row_id"]
            if "__fight_row_id" in renamed.columns
            else []
        )
        merge_cols = [
            merge_key,
            *row_merge_cols,
            "event_date",
            *[f"{prefix}{col}" for col in override_cols],
        ]
        updated = updated.merge(
            renamed[merge_cols],
            on=[merge_key, *row_merge_cols, "event_date"],
            how="left",
        )

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
    row_id_columns = (
        ["fight_row_id"] if "fight_row_id" in per_fight_df.columns else []
    )
    if per_fight_df.empty:
        return pd.DataFrame(
            columns=[
                identity_col, *row_id_columns, "event_date",
                "prior_losses_ko", "prior_losses_sub", "prior_recent_ko_loss",
            ]
        )

    frames = []
    for fighter, group in per_fight_df.groupby(identity_col):
        group = group.sort_values(
            ["event_date", *row_id_columns],
            kind="stable",
        )
        losses_ko = losses_sub = 0
        ko_loss_flags: list[bool] = []
        records = []
        for _, row in group.iterrows():
            recent = (
                float(any(ko_loss_flags[-2:])) if ko_loss_flags else np.nan
            )
            records.append({
                identity_col: fighter,
                **{
                    column: row.get(column)
                    for column in row_id_columns
                },
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
        merge_key = (
            f"__fighter_{side}_key"
            if history_identity_col == "fighter_key"
            else fighter_col
        )
        renamed = history.rename(columns={
            history_identity_col: merge_key,
            "fight_row_id": "__fight_row_id",
            "prior_losses_ko": f"{prefix}losses_ko",
            "prior_losses_sub": f"{prefix}losses_sub",
            "prior_recent_ko_loss": f"{prefix}recent_ko_loss",
        })
        row_merge_cols = (
            ["__fight_row_id"]
            if "__fight_row_id" in renamed.columns
            else []
        )
        renamed = renamed.drop_duplicates(
            subset=[merge_key, *row_merge_cols, "event_date"],
            keep="last",
        )
        features = features.merge(
            renamed,
            on=[merge_key, *row_merge_cols, "event_date"],
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

    identity_col = _stateful_identity_column(rolling_df)
    opponent_col = "opponent_key" if identity_col == "fighter_key" else "opponent"

    row_id_columns = (
        ["fight_row_id"] if "fight_row_id" in rolling_df.columns else []
    )
    lookup_columns = [identity_col, *row_id_columns, "event_date", "roll_won"]
    _won_rows = rolling_df[lookup_columns].dropna(
        subset=["roll_won"]
    )
    _lookup: dict[tuple, float] = {
        (
            r[identity_col],
            *(r[column] for column in row_id_columns),
            r["event_date"],
        ): r["roll_won"]
        for _, r in _won_rows.iterrows()
    }

    sos_results = np.full(len(rolling_df), np.nan)

    for _fighter, group in rolling_df.groupby(identity_col):
        group = group.sort_values(
            ["event_date", *row_id_columns],
            kind="stable",
        )
        past_opps: list[tuple] = []

        for df_idx, row in group.iterrows():
            if past_opps:
                recent = past_opps[-window:]
                opp_wrs: list[float] = []
                weights: list[float] = []
                for rank, opponent_observation in enumerate(recent, start=1):
                    wr = _lookup.get(opponent_observation)
                    if wr is not None and not np.isnan(wr):
                        opp_wrs.append(wr)
                        # Linear recency weight: most recent = highest
                        weights.append(float(rank))
                if opp_wrs:
                    sos_results[df_idx] = float(np.average(opp_wrs, weights=weights))

            opp = row.get(opponent_col)
            if pd.notna(opp) and opp:
                past_opps.append(
                    (
                        opp,
                        *(row.get(column) for column in row_id_columns),
                        row["event_date"],
                    )
                )

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
    prior_title_bouts: list[int] = []
    prior_wins_ko: list[int] = []
    prior_wins_sub: list[int] = []
    prior_wins_dec: list[int] = []

    wins = losses = draws = 0
    win_streak = lose_streak = longest_win_streak = 0
    total_rounds = 0.0
    title_bouts = 0
    wins_ko = wins_sub = wins_dec = 0
    unknown_win_method = unknown_rounds = unknown_titles = False
    retained_flags = _retained_long_mask(fighter_fights).to_numpy()

    for index, (_, row) in enumerate(fighter_fights.iterrows()):
        current_win_streaks.append(win_streak)
        prior_wins.append(wins)
        prior_losses.append(losses)
        prior_draws.append(draws)
        prior_lose_streaks.append(lose_streak)
        prior_longest_win_streaks.append(longest_win_streak)
        prior_total_rounds.append(np.nan if unknown_rounds else total_rounds)
        prior_title_bouts.append(np.nan if unknown_titles else title_bouts)
        prior_wins_ko.append(np.nan if unknown_win_method else wins_ko)
        prior_wins_sub.append(np.nan if unknown_win_method else wins_sub)
        prior_wins_dec.append(np.nan if unknown_win_method else wins_dec)

        result_label = row.get("result_label", "draw")
        method_group = _method_group(row.get("method"))
        if retained_flags[index] and result_label == "win" and not _has_observed_win_method(row.get("method")):
            unknown_win_method = True

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
        if retained_flags[index] and pd.isna(finish_round):
            unknown_rounds = True
        if not pd.isna(finish_round):
            total_rounds += float(finish_round)

        title_flag = pd.to_numeric(pd.Series([row.get("title_bout")]), errors="coerce").iloc[0]
        if retained_flags[index] and pd.isna(title_flag):
            unknown_titles = True
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


def _load_supplement_raw(path: Path) -> pd.DataFrame:
    """Load a supplement CSV in its raw fight-pair schema."""
    if not path.is_file():
        return pd.DataFrame()
    from src.data.provider_consumer_references import verify_provider_consumer_artifact
    references = verify_provider_consumer_artifact(path)
    try:
        result = pd.read_csv(path, parse_dates=["event_date"], dtype={column: object for column in PROVENANCE_COLUMNS})
        result.attrs["provider_references"] = references
        return result
    except Exception as exc:
        if active_sources() is not None:
            raise
        logger.warning("Failed to load supplement %s: %s", path, exc)
        return pd.DataFrame()


def _supplement_rows_to_long_format(
    raw: pd.DataFrame,
    *,
    raw_data_dir: Path | None = None,
    expected_division: str | None = None,
) -> pd.DataFrame:
    """Convert raw supplement rows to per-fighter long format."""
    if raw.empty:
        return pd.DataFrame()

    records: list[dict] = []
    ambiguous_name_keys = _training_ambiguous_name_keys(raw_data_dir)
    for _, row in raw.iterrows():
        retained = bool(_bound_history_mask(pd.DataFrame([row])).iloc[0])
        if retained:
            from src.data.provider_observation_history import ProviderObservationError, verify_retained_history_row

            verify_retained_history_row(row.to_dict(), references=raw.attrs.get("provider_references"))
            if expected_division is not None and row.get("history_division") != expected_division:
                raise ProviderObservationError("retained career row belongs to a different professional/amateur track")
        date = row.get("event_date")
        winner = row.get("winner", "")
        method = row.get("method", "")
        finish_round = pd.to_numeric(
            pd.Series([row.get("finish_round")]), errors="coerce"
        ).iloc[0]
        title_bout = row.get("title_bout", np.nan)
        organization = row.get("organization", "")

        for prefix, fighter_col in [("a_", "fighter_a"), ("b_", "fighter_b")]:
            if retained and prefix != "a_":
                # A response bound to its subject does not independently bind
                # the opponent's stable UFC identity.
                continue
            fighter = row.get(fighter_col)
            if pd.isna(fighter) or not fighter:
                continue

            opp_prefix = "b_" if prefix == "a_" else "a_"
            opp_col = "fighter_b" if prefix == "a_" else "fighter_a"

            # A legacy row binds its fighter only through an explicit id column
            # (absent in the accepted supplement today); an ambiguous spelling
            # without one gets no stateful key and therefore no summary.
            id_col = "fighter_a_id" if prefix == "a_" else "fighter_b_id"
            explicit_id = row.get(id_col) if id_col in raw.columns else None
            # Owner decision D10-OLD-GAPS (review finding IS3-2): the opponent
            # side keys through its own explicit id column as well, so a
            # spelling that binds or collides is never credited to the wrong
            # identity from the opponent's perspective. An ordinary name keeps
            # its name key whatever id it carries.
            opp_id_col = "fighter_b_id" if prefix == "a_" else "fighter_a_id"
            opp_explicit_id = row.get(opp_id_col) if opp_id_col in raw.columns else None
            record = {
                "fighter": fighter,
                "fighter_id": normalize_ufcstats_id(explicit_id),
                "fighter_key": fighter_identity_key(
                    fighter,
                    explicit_id,
                    ambiguous_name_keys=ambiguous_name_keys,
                ),
                "opponent": row.get(opp_col, ""),
                "opponent_key": fighter_identity_key(
                    row.get(opp_col, ""),
                    opp_explicit_id,
                    ambiguous_name_keys=ambiguous_name_keys,
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
            if retained:
                record.update({key: row.get(key) for key in PROVENANCE_COLUMNS})
                record["fighter_id"] = row.get("fighter_a_id")
                record["fighter_key"] = f"ufcstats:{row['fighter_a_id']}"

            # Per-fight stats are NaN for pre-UFC fights
            for stat in STAT_COLUMNS + EXTENDED_STAT_COLUMNS:
                col = f"{prefix}{stat}"
                opp_stat = f"{opp_prefix}{stat}"
                record[stat] = row.get(col, np.nan)
                record[f"opp_{stat}"] = row.get(opp_stat, np.nan)

            records.append(record)

    return pd.DataFrame(records)


def _retained_long_mask(frame: pd.DataFrame) -> pd.Series:
    return _bound_history_mask(frame)


def _scope_retained_history_owners(fights: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Use observed stable IDs to join retained owners into the existing chronology."""
    retained = history.loc[_retained_long_mask(history)]
    if retained.empty:
        return fights
    owners = set(retained["fighter_id"])
    result = fights.copy()
    for side in ("a", "b"):
        identity_column = f"fighter_{side}_id"
        if identity_column not in result:
            continue
        ids = result[identity_column].map(normalize_ufcstats_id)
        key_column = f"__fighter_{side}_key"
        if key_column not in result:
            result[key_column] = [fighter_identity_key(name, identity) for name, identity
                                  in zip(result[f"fighter_{side}"], ids)]
        selected = ids.isin(owners)
        result.loc[selected, key_column] = ids.loc[selected].map(lambda identity: f"ufcstats:{identity}")
    return result


def _eligible_retained_history(frame: pd.DataFrame, *, fighter_id: object, available_before: object) -> pd.DataFrame:
    """Select an already verified subject at its actual prediction cutoff."""
    stable_id = normalize_ufcstats_id(fighter_id)
    if not stable_id or frame.empty or "fighter_id" not in frame.columns:
        return frame.iloc[0:0].copy()
    cutoff = pd.to_datetime(available_before, errors="coerce", utc=True)
    capture = pd.to_datetime(frame.get("source_captured_at_utc"), errors="coerce", utc=True)
    dates = pd.to_datetime(frame["event_date"], errors="coerce", utc=True)
    boundary = pd.to_datetime(frame.get("first_ufc_date"), errors="coerce", utc=True)
    valid = (_retained_long_mask(frame) & frame["fighter_id"].eq(stable_id)
             & capture.notna() & (capture < cutoff) & dates.notna()
             & (dates < cutoff) & (dates < boundary))
    return frame.loc[valid].copy()


def _overlay_retained_career_summaries(features: pd.DataFrame, history: pd.DataFrame, *, amateur: bool = False) -> pd.DataFrame:
    retained = history.loc[_retained_long_mask(history)]
    if retained.empty:
        return features
    result = features.copy()
    columns = _AMATEUR_SUMMARY_COLUMNS if amateur else _PRE_UFC_SUMMARY_COLUMNS
    compute = _compute_amateur_summary if amateur else _compute_pre_ufc_summary
    owners = set(retained["fighter_id"])
    for side in ("a", "b"):
        for index, row in result.iterrows():
            stable_id = normalize_ufcstats_id(row.get(f"fighter_{side}_id"))
            if stable_id not in owners:
                continue
            eligible = _eligible_retained_history(retained, fighter_id=stable_id, available_before=row["event_date"])
            summary = compute(eligible.assign(fighter=stable_id))
            values = summary.iloc[0].to_dict() if len(summary) == 1 else {}
            for column in columns:
                result.loc[index, f"{side}_{column}"] = values.get(column, np.nan)
    for column in columns:
        for side in ("a", "b"):
            if f"{side}_{column}" not in result:
                result[f"{side}_{column}"] = np.nan
        result[f"diff_{column}"] = result[f"a_{column}"] - result[f"b_{column}"]
    return result


def _drop_quarantined_supplement_identities(
    long_df: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    """Fail closed for known collisions in the legacy name-only supplements."""
    if long_df.empty or "fighter" not in long_df.columns:
        return long_df

    quarantined_identities = _SUPPLEMENT_QUARANTINED_IDENTITIES
    if label == "amateur":
        quarantined_identities = (
            quarantined_identities
            | _AMATEUR_SUPPLEMENT_QUARANTINED_IDENTITIES
        )
    quarantined_mask = long_df["fighter"].isin(quarantined_identities)
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


def _drop_legacy_rows_of_reviewed_identities(
    long_df: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    """Drop legacy (name-only) rows whose stateful key is a reviewed identity.

    A reviewed identity's history is the audited stable-ID artifact appended by
    ``_load_reviewed_supplement_history``; every one of its tracked spellings
    keys to ``ufcstats:<id>`` through ``fighter_identity_key``. A legacy
    scraped row under any of those spellings would be counted a second time
    next to the reviewed rows once the summary and the rolling seed key on the
    identity (Mizuki: five legacy "Mizuki Inoue" bouts on top of the eighteen
    reviewed ones, found in review on 2026-09-17). Such rows are excluded
    whether or not the reviewed rows are appended: without them the features
    stay NaN, exactly as the spelling quarantine does for "Mizuki". Rows that
    carry their own stable id (the reviewed rows themselves, retained rows)
    are never touched.
    """
    if long_df.empty or "fighter_key" not in long_df.columns:
        return long_df
    reviewed_keys = {f"ufcstats:{ufcstats_id}" for ufcstats_id in REVIEWED_FIGHTER_IDENTITIES}
    mask = long_df["fighter_key"].isin(reviewed_keys) & ~_retained_long_mask(long_df)
    if "fighter_id" in long_df.columns:
        bound_ids = long_df["fighter_id"].map(normalize_ufcstats_id)
        carries_own_id = bound_ids.notna() & ("ufcstats:" + bound_ids.fillna("").astype(str)).eq(
            long_df["fighter_key"].astype(str)
        )
        mask &= ~carries_own_id
    if not mask.any():
        return long_df
    logger.info(
        "Excluded %d legacy %s rows keyed to a reviewed stable identity (%s); "
        "only the reviewed history is used for it",
        int(mask.sum()),
        label,
        ", ".join(sorted(long_df.loc[mask, "fighter"].astype(str).unique())),
    )
    return long_df.loc[~mask].copy()


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


def _load_reviewed_supplement_history(
    history_type: str,
    *,
    raw_data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load one audited history type, rejecting any incomplete fighter group."""
    path = (
        _resolve_reviewed_supplement_history_path()
        if raw_data_dir is None
        else _resolve_reviewed_supplement_history_path(raw_data_dir)
    )
    if not path.is_file():
        return pd.DataFrame()
    try:
        digest = _reviewed_history_canonical_sha256(path)
    except OSError as exc:
        if active_sources() is not None:
            raise
        logger.warning("Failed to hash reviewed fighter history %s: %s", path, exc)
        return pd.DataFrame()
    if digest != _REVIEWED_SUPPLEMENT_HISTORY_CANONICAL_SHA256:
        if active_sources() is not None:
            raise ValueError('Selected reviewed career history differs from its approved original')
        logger.warning(
            "Ignoring reviewed fighter history %s because its canonical SHA-256 "
            "does not match the approved 40-row evidence artifact",
            path,
        )
        return pd.DataFrame()
    raw = _load_supplement_raw(path)
    missing = sorted(_REVIEWED_HISTORY_REQUIRED_COLUMNS.difference(raw.columns))
    if missing:
        if active_sources() is not None:
            raise ValueError(f'Selected reviewed career history lacks columns: {missing}')
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
    ambiguous_name_keys = _training_ambiguous_name_keys(raw_data_dir)
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
                "fighter_key": fighter_identity_key(
                    canonical_name,
                    ufcstats_id,
                ),
                "opponent": row["fighter_b"],
                "opponent_key": fighter_identity_key(
                    row["fighter_b"],
                    ambiguous_name_keys=ambiguous_name_keys,
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
    raw_data_dir: Path | None = None,
    _new_ambiguous_names: frozenset[str] | None = None,
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
            if active_sources() is not None:
                raise ValueError(f"Selected professional history lacks columns: {missing_columns}")
            logger.warning(
                "Ignoring pre-UFC supplement %s because required columns are missing: %s",
                path,
                ", ".join(missing_columns),
            )
        else:
            # NEW owner evidence must precede any identity-based collapse.
            # The ordinary raw collapse cannot distinguish supplied IDs under
            # ordinary spellings; retain all raw rows for the existing long
            # bout reconciliation in the NEW owner domain instead.
            deduped = _dedupe_supplement_rows(raw) if _new_ambiguous_names is None else raw.copy()
            if "organization" in deduped.columns:
                ufc_mask = deduped["organization"].map(_is_tracked_ufc_organization)
                deduped = deduped.loc[~ufc_mask].copy()

            long_df = _supplement_rows_to_long_format(
                deduped,
                raw_data_dir=raw_data_dir,
                expected_division="pro",
            )
            if _new_ambiguous_names is not None:
                long_df = _new_legacy_history_owners(long_df, _new_ambiguous_names)
            long_df = _drop_quarantined_supplement_identities(
                long_df,
                label="pre-UFC",
            )
            long_df = _drop_legacy_rows_of_reviewed_identities(long_df, label="pre-UFC")
            long_df = _drop_duplicate_supplement_bouts(long_df)
            if not long_df.empty:
                # An unparseable date can never be proven to precede a modeled
                # fight. Drop it rather than assigning an artificial value.
                long_df["event_date"] = pd.to_datetime(
                    long_df["event_date"], errors="coerce", utc=True
                ).dt.tz_localize(None)
                long_df = long_df.dropna(subset=["event_date"]).reset_index(drop=True)
    if include_reviewed:
        reviewed = _load_reviewed_supplement_history(
            "professional",
            raw_data_dir=raw_data_dir,
        )
        if not reviewed.empty:
            long_df = pd.concat([long_df, reviewed], ignore_index=True)
    return long_df


def _load_amateur_supplement(
    path: Path,
    *,
    include_reviewed: bool = False,
    raw_data_dir: Path | None = None,
    _new_ambiguous_names: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Load legacy amateur rows safely, then append reviewed subject histories."""
    raw = _load_supplement_raw(path)
    if raw.empty:
        legacy_long = pd.DataFrame()
    else:
        legacy_long = _supplement_rows_to_long_format(
            _dedupe_supplement_rows(raw) if _new_ambiguous_names is None else raw.copy(),
            raw_data_dir=raw_data_dir,
            expected_division="am",
        )
        if _new_ambiguous_names is not None:
            legacy_long = _new_legacy_history_owners(legacy_long, _new_ambiguous_names)
        legacy_long = _drop_quarantined_supplement_identities(
            legacy_long,
            label="amateur",
        )
        legacy_long = _drop_legacy_rows_of_reviewed_identities(legacy_long, label="amateur")
        # Same per-fighter bout collapse as the professional loader (added
        # 2026-09-17): mirrored label-variant copies of one amateur bout were
        # counted twice (9 bouts / 18 records on the accepted file).
        legacy_long = _drop_duplicate_supplement_bouts(legacy_long)

    if include_reviewed:
        reviewed = _load_reviewed_supplement_history(
            "amateur",
            raw_data_dir=raw_data_dir,
        )
        if not reviewed.empty:
            legacy_long = pd.concat([legacy_long, reviewed], ignore_index=True)
    return legacy_long.reset_index(drop=True)


def _new_legacy_history_owners(long_df, ambiguous_names):
    """Set actual/reviewed owner evidence before the existing bout collapse."""
    if type(ambiguous_names) is not frozenset:
        raise ValueError('Exact internal NEW collision domain required')
    if long_df.empty:
        return long_df
    result = long_df.copy()
    ordinary = ~_retained_long_mask(result)
    result.loc[ordinary, 'fighter_key'] = result.loc[ordinary].apply(
        lambda row: _new_observed_owner_key(row.get('fighter'), row.get('fighter_id'), ambiguous_names),
        axis=1,
    )
    return result


def _drop_duplicate_supplement_bouts(long_df: pd.DataFrame) -> pd.DataFrame:
    """Keep one record per fighter identity, calendar day and opponent.

    Each fighter's page names the opponent in its own way, so one bout can
    reach the long frame twice for the same fighter: through the fighter's own
    row and through the opponent's mirrored row when the promotion label
    differs ("Bellator Fighting Championships" / "Bellator"), when the
    opponent's tokens are ordered or spaced differently (Song Kenan / Kenan
    Song, Won Bin Ki / WonBin Ki), or when the opponent is spelled by a
    reviewed ring name or source spelling (Patricio Pitbull / Patricio Freire,
    Kyung Pyo Kim / KyeungPyo Kim). The fighter side is grouped by the same
    stateful identity the summary and the rolling statistics use
    (``fighter_key`` when present: an accent or case variant of one spelling is
    one identity, a reviewed ring name such as Patricio Pitbull / Patricio
    Freire is not folded, and an ambiguous spelling without an id binding is
    left untouched here because it never reaches the professional summary;
    the amateur summary still keys on the spelling); without a key column the
    exact spelling is used. Two different opponents on one day
    (one-night tournaments) are kept. Stable-ID retained rows are left to
    their own rule.
    """
    if long_df.empty or not {"fighter", "opponent", "event_date"}.issubset(long_df.columns):
        return long_df
    from src.data.pre_ufc_scraper import _supplement_pair_name_keys

    retained = _retained_long_mask(long_df)
    identity_column = _stateful_identity_column(long_df)
    unkeyed = pd.Series(False, index=long_df.index)
    if identity_column == "fighter_key":
        unkeyed = long_df["fighter_key"].isna()
    legacy = long_df.loc[~retained & ~unkeyed].copy()
    if legacy.empty:
        return long_df
    keys = legacy["opponent"].map(_supplement_pair_name_keys)
    legacy["__bout_day"] = pd.to_datetime(legacy["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    legacy["__opponent_order_key"] = [key[0] for key in keys]
    legacy["__opponent_flat_key"] = [key[1] for key in keys]
    legacy["__identity"] = legacy[identity_column].astype(str)
    # Within a duplicate group keep the record whose promotion label the tier
    # tables recognise best (1 = major ... 3 = regional/unknown, unlabelled last);
    # the raw frame arrives sorted longest-label-first, and the long variant
    # ("Bellator Fighting Championships", "Jungle Fight Championship") is the
    # one the tables do not know. Ties keep the incoming order. The frame is
    # restored to its original order afterwards.
    legacy["__org_tier"] = _encode_org_tiers(legacy).fillna(4.0)
    legacy = legacy.sort_values("__org_tier", kind="stable")
    before = len(legacy)
    survivors = legacy.drop_duplicates(subset=["__identity", "__bout_day", "__opponent_order_key"], keep="first")
    survivors = survivors.drop_duplicates(subset=["__identity", "__bout_day", "__opponent_flat_key"], keep="first")
    absorbed = legacy.loc[~legacy.index.isin(survivors.index)]
    survivors = _reconcile_collapsed_duplicates(survivors, absorbed)
    dropped = before - len(survivors)
    if dropped:
        logger.info("Dropped %d duplicate pre-UFC bout records (same fighter, day and opponent)", dropped)
    legacy = survivors.drop(columns=["__bout_day", "__opponent_order_key", "__opponent_flat_key", "__org_tier", "__identity"])
    return pd.concat([legacy, long_df.loc[retained | unkeyed]]).sort_index()


def _observed_method_text(value: object) -> str:
    text = "" if value is None else str(value)
    if text.strip().lower() == "nan":
        return ""
    return " ".join(text.strip().casefold().split())


def _reconcile_collapsed_duplicates(survivors: pd.DataFrame, absorbed: pd.DataFrame) -> pd.DataFrame:
    """Never let a collapsed group hide a disagreement between its copies.

    The two copies of one bout come from two source pages. When they agree,
    the survivor (best promotion tier) stands. When they disagree on the
    method, the survivor keeps the bout but its method becomes unknown (NaN):
    a real observation is preferred, an unresolved conflict is not guessed.
    When they disagree on the result itself (the ``won`` flag or the result
    label: a loss on one page against a draw or no contest on the other), the
    bout is dropped altogether: an outcome the two sources contradict cannot
    be counted as a win or a loss. Owner decision 2026-09-17, Step 0
    sub-decision (d).
    """
    if absorbed.empty or survivors.empty:
        return survivors
    by_order: dict[tuple, object] = {}
    by_flat: dict[tuple, object] = {}
    for index, row in survivors.iterrows():
        by_order.setdefault((row["__identity"], row["__bout_day"], row["__opponent_order_key"]), index)
        by_flat.setdefault((row["__identity"], row["__bout_day"], row["__opponent_flat_key"]), index)
    method_conflicts: set = set()
    result_conflicts: set = set()
    has_method = "method" in survivors.columns
    has_won = "won" in survivors.columns
    has_label = "result_label" in survivors.columns
    for _, row in absorbed.iterrows():
        target = by_order.get((row["__identity"], row["__bout_day"], row["__opponent_order_key"]))
        if target is None:
            target = by_flat.get((row["__identity"], row["__bout_day"], row["__opponent_flat_key"]))
        if target is None:
            continue
        kept = survivors.loc[target]
        if has_won:
            kept_won = pd.to_numeric(pd.Series([kept.get("won")]), errors="coerce").iloc[0]
            other_won = pd.to_numeric(pd.Series([row.get("won")]), errors="coerce").iloc[0]
            if pd.notna(kept_won) and pd.notna(other_won) and float(kept_won) != float(other_won):
                result_conflicts.add(target)
                continue
        if has_label:
            kept_label = _observed_method_text(kept.get("result_label"))
            other_label = _observed_method_text(row.get("result_label"))
            if kept_label and other_label and kept_label != other_label:
                result_conflicts.add(target)
                continue
        if has_method:
            kept_method = _observed_method_text(kept.get("method"))
            other_method = _observed_method_text(row.get("method"))
            if kept_method and other_method and kept_method != other_method:
                method_conflicts.add(target)
    if method_conflicts:
        logger.info(
            "Blanked the method of %d collapsed pre-UFC bouts whose source copies disagree",
            len(method_conflicts),
        )
        survivors = survivors.copy()
        survivors.loc[sorted(method_conflicts, key=str), "method"] = np.nan
    if result_conflicts:
        logger.warning(
            "Dropped %d collapsed pre-UFC bouts whose source copies disagree on the result",
            len(result_conflicts),
        )
        survivors = survivors.loc[~survivors.index.isin(result_conflicts)]
    return survivors


def _tracked_fighter_dates(
    tracked_fights: pd.DataFrame,
    *,
    ambiguous_name_keys: frozenset[str] | set[str] | None = None,
) -> pd.DataFrame:
    """Long (fighter, fighter_key, event_date, event_name) rows from either tracked schema.

    ``event_name`` is blank when the caller's frame does not carry it; such rows
    are treated as UFC-owned, which preserves the historical behaviour for
    callers that only know a debut date. ``fighter_key`` is the same stateful
    identity the rolling statistics use: taken from a ``fighter_key`` /
    ``__fighter_<side>_key`` column when the caller carries one, otherwise
    derived from the spelling and any ``fighter_id`` / ``fighter_<side>_id``
    column (an ambiguous spelling without an id yields no key).
    """
    empty = pd.DataFrame(columns=["fighter", "fighter_key", "event_date", "event_name"])
    if tracked_fights.empty or "event_date" not in tracked_fights.columns:
        return empty

    def derive_keys(names, ids):
        return [
            fighter_identity_key(name, fighter_id, ambiguous_name_keys=ambiguous_name_keys)
            for name, fighter_id in zip(names, ids)
        ]

    has_event_name = "event_name" in tracked_fights.columns
    identity_frames: list[pd.DataFrame] = []
    if "fighter" in tracked_fights.columns:
        frame = tracked_fights[["fighter", "event_date"]].copy()
        frame["event_name"] = tracked_fights["event_name"].values if has_event_name else ""
        if "fighter_key" in tracked_fights.columns:
            frame["fighter_key"] = tracked_fights["fighter_key"].values
        else:
            ids = tracked_fights["fighter_id"].values if "fighter_id" in tracked_fights.columns else [None] * len(frame)
            frame["fighter_key"] = derive_keys(frame["fighter"].values, ids)
        identity_frames.append(frame)
    else:
        for side in ("a", "b"):
            fighter_col = f"fighter_{side}"
            if fighter_col in tracked_fights.columns:
                frame = tracked_fights[[fighter_col, "event_date"]].rename(
                    columns={fighter_col: "fighter"}
                )
                frame["event_name"] = tracked_fights["event_name"].values if has_event_name else ""
                key_col = f"__fighter_{side}_key"
                id_col = f"fighter_{side}_id"
                if key_col in tracked_fights.columns:
                    frame["fighter_key"] = tracked_fights[key_col].values
                else:
                    ids = tracked_fights[id_col].values if id_col in tracked_fights.columns else [None] * len(frame)
                    frame["fighter_key"] = derive_keys(frame["fighter"].values, ids)
                identity_frames.append(frame)

    if not identity_frames:
        return empty

    identities = pd.concat(identity_frames, ignore_index=True)
    identities["event_date"] = pd.to_datetime(
        identities["event_date"], errors="coerce", utc=True
    ).dt.tz_localize(None).dt.normalize()
    identities = identities.dropna(subset=["fighter", "event_date"])
    identities["fighter"] = identities["fighter"].astype(str)
    identities = identities[identities["fighter"].str.strip().ne("")]
    identities["fighter_key"] = identities["fighter_key"].where(identities["fighter_key"].notna(), None)
    return identities.reset_index(drop=True)[["fighter", "fighter_key", "event_date", "event_name"]]


def _tracked_ufc_debut_dates(
    tracked_fights: pd.DataFrame,
    *,
    identity_column: str = "fighter",
) -> pd.Series:
    """Return each fighter's first tracked UFC-owned event date.

    ``tracked_fights`` may be either the raw two-fighter fight schema or the
    internal long schema. The series is indexed by the exact spelling by
    default, or by the stateful identity key when ``identity_column`` is
    ``"fighter_key"`` (rows without a key are ignored). Identity matching is
    exact on purpose: a fuzzy or suffix-insensitive match can merge two people
    and is less safe than leaving the corresponding feature unknown. Tracked
    bouts from promotions not treated as UFC-owned (Strikeforce, WEC, PRIDE,
    ...) do not count as a UFC debut. An identity whose tracked bouts are ALL
    such bouts (a Strikeforce-only opponent, say) keeps its earliest tracked
    bout as the cutoff, exactly as before, so its earlier regional record is
    not discarded. Note: for the 79 tracked non-UFC rows themselves (all
    2012-08-18 or earlier, outside every modeled window) the per-fighter
    pre-UFC aggregate may include bouts dated after that row.
    """
    if identity_column not in {"fighter", "fighter_key"}:
        raise ValueError(f"unsupported identity column {identity_column!r}")
    identities = _tracked_fighter_dates(tracked_fights)
    if identity_column == "fighter_key" and not identities.empty:
        identities = identities.loc[identities["fighter_key"].notna()]
    if identities.empty:
        return pd.Series(dtype="datetime64[ns]")
    any_organisation = identities.groupby(identity_column, sort=False)["event_date"].min()
    ufc_owned = identities[~identities["event_name"].map(_is_non_ufc_tracked_event)]
    if ufc_owned.empty:
        return any_organisation
    debut = ufc_owned.groupby(identity_column, sort=False)["event_date"].min()
    fallback = any_organisation[~any_organisation.index.isin(debut.index)]
    if fallback.empty:
        return debut
    return pd.concat([debut, fallback])


def _filter_pre_ufc_rows_before_tracked_debut(
    supplement_df: pd.DataFrame,
    tracked_fights: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only dated supplement rows strictly before each tracked debut.

    The debut is the earliest modeled UFC-owned row for the fighter's stateful
    identity: the ``fighter_key`` the rolling statistics group by when the
    supplement rows carry one (the loader always attaches it), otherwise the
    exact spelling. Therefore ``supplement_date < debut_date`` also guarantees
    that the row is strictly earlier than every historical modeled UFC fight
    for that fighter. A supplement bout dated the same day as ANY tracked bout
    of the same identity is dropped as well: the tracked table already holds
    that bout (for example a WEC or Strikeforce fight listed by UFCStats), so
    keeping the supplement copy would count it twice. A missing/ambiguous
    identity (an ambiguous spelling with no id binding has no key) or date
    fails closed to no rows, which later materializes as NaN rather than a
    fabricated zero.
    """
    if supplement_df.empty:
        return supplement_df.copy()
    required = {"fighter", "event_date"}
    if not required.issubset(supplement_df.columns):
        return supplement_df.iloc[0:0].copy()

    identity_column = _stateful_identity_column(supplement_df)
    debut_dates = _tracked_ufc_debut_dates(tracked_fights, identity_column=identity_column)
    if debut_dates.empty:
        return supplement_df.iloc[0:0].copy()

    filtered = supplement_df.copy()
    filtered["event_date"] = pd.to_datetime(
        filtered["event_date"], errors="coerce", utc=True
    ).dt.tz_localize(None).dt.normalize()
    if identity_column == "fighter_key":
        identities = filtered["fighter_key"].where(filtered["fighter_key"].notna(), None)
    else:
        identities = filtered["fighter"].astype(str)
    filtered["__tracked_ufc_debut"] = identities.map(debut_dates)
    valid = (
        filtered["event_date"].notna()
        & filtered["__tracked_ufc_debut"].notna()
        & (filtered["event_date"] < filtered["__tracked_ufc_debut"])
    )
    tracked_dates = _tracked_fighter_dates(tracked_fights)
    if not tracked_dates.empty:
        tracked_pairs = set(zip(tracked_dates[identity_column], tracked_dates["event_date"]))
        same_day_as_tracked = pd.Series(
            [
                identity is not None and (identity, event_date) in tracked_pairs
                for identity, event_date in zip(identities, filtered["event_date"])
            ],
            index=filtered.index,
        )
        valid &= ~same_day_as_tracked
    retained = _retained_long_mask(filtered)
    if retained.any():
        boundary = pd.to_datetime(filtered["first_ufc_date"], errors="coerce", utc=True).dt.tz_localize(None)
        # Retained rows carry the independently supplied subject debut; grouping
        # later uses the stable owner, never a spelling match to another person.
        valid = (valid & ~retained) | (retained & filtered["event_date"].notna()
                                     & boundary.notna() & (filtered["event_date"] < boundary))
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
    # Exact label variants observed in the accepted supplement (verified
    # 2026-09-17; owner decision to map them). Matching stays exact on the
    # lower-cased label: no substring rule, per the note above
    # _TRACKED_UFC_ORGANIZATION_LABELS.
    "bellator fighting championships", "one fighting championship",
    "rizin fighting federation", "rizin fighting world grand prix",
    "pride fighting championships", "strikeforce / m-1 global",
    "strikeforce sf", "strikeforce-elitexc", "invicta fighting championships",
    "invicta on axs tv", "affliction entertainment", "ksw epic",
    "pfl super fights",
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
    # Exact label variants observed in the accepted supplement (2026-09-17;
    # the six "brave cf NN / ..." co-promotion labels are the exact strings
    # of the file). "brave fc" is NOT here: in the supplement it is a 2009
    # Brazilian promotion (Curitiba), not Brave Combat Federation.
    "cwfc", "cwfc usa", "m-1 challenge", "m-1 global m-1",
    "wkg & m-1 global", "wkg & m-1 challenge",
    "m-1 global / angel fight promotions",
    "m-1 global / absolute championship berkut", "sambo-70 / m-1 global",
    "cage fury fighting championships", "fury fighting championship",
    "legacy fighting championship", "titan fighting championship",
    "ares fighting championship", "jungle fight championship",
    "pancrase hybrid wrestling", "brave cf 53 / octagon",
    "brave cf 88 / wfc", "wef 91 / brave cf", "brave cf 70 / wfc",
    "wfc 24 / brave cf", "brave cf 74 / hxmma",
    "combate americas: mexico vs. spain", "combate americas camma",
    "combate americas: stockton", "wsof global championship wsofgc",
}

# Labels whose meaning depends on the bout date. "BFC" is Bellator Fighting
# Championships only during Bellator's founding era (first event 2009-04-03;
# renamed Bellator MMA in 2013): before 2009 the label belongs to Brazilian
# events and from 2013 on to other promotions. Owner decision 2026-09-17,
# Step 0 sub-decision (a). Outside the window the label is tier 3 as before.
_DATE_SCOPED_ORG_TIERS: dict[str, tuple[pd.Timestamp, pd.Timestamp, float]] = {
    "bfc": (pd.Timestamp("2009-01-01"), pd.Timestamp("2012-12-31"), 1.0),
}

# Everything else is tier 3 (regional / unknown)


def _naive_bout_day(value: object) -> pd.Timestamp:
    """Coerce a bout date to a naive calendar day (NaT when unknown)."""
    try:
        stamp = pd.to_datetime(value, errors="coerce")
    except (TypeError, ValueError):
        return pd.NaT
    if isinstance(stamp, pd.Timestamp) and stamp.tz is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize() if isinstance(stamp, pd.Timestamp) and pd.notna(stamp) else pd.NaT


def _encode_org_tier(org_name: str, event_date: object = None) -> float:
    """Map organization name to tier (1=major, 2=feeder, 3=regional/unknown).

    Returns NaN if org_name is empty or missing — never fabricates. A label in
    ``_DATE_SCOPED_ORG_TIERS`` takes its scoped tier only when ``event_date``
    falls inside the recorded window; without a usable date it stays tier 3.
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
    scoped = _DATE_SCOPED_ORG_TIERS.get(name_lower)
    if scoped is not None:
        start, end, tier = scoped
        day = _naive_bout_day(event_date)
        if pd.notna(day) and start <= day <= end:
            return tier
    return 3.0


def _encode_org_tiers(frame: pd.DataFrame) -> pd.Series:
    """Per-row org tier for a long frame, using each row's own bout date."""
    if "organization" not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    dates = frame["event_date"] if "event_date" in frame.columns else pd.Series(None, index=frame.index, dtype=object)
    return pd.Series(
        [_encode_org_tier(org, date) for org, date in zip(frame["organization"], dates)],
        index=frame.index,
        dtype=float,
    )


def _compute_pre_ufc_summary(
    supplement_df: pd.DataFrame, *, preserve_unknown_methods: bool = False,
) -> pd.DataFrame:
    """Compute per-fighter pre-UFC career summary features.

    Returns a DataFrame with one row per fighter, columns:
        fighter, pre_ufc_total_fights, pre_ufc_wins, pre_ufc_losses,
        pre_ufc_win_pct, pre_ufc_ko_rate, pre_ufc_sub_rate,
        pre_ufc_dec_rate, pre_ufc_org_tier_best (best org tier fought in).

    All values derived from real data or NaN. No estimates. Rows are grouped
    by the same stateful identity the rolling statistics use (``fighter_key``
    when present, else the exact spelling); the ``fighter`` column of the
    result carries that identity. Rows without a key (an ambiguous spelling
    with no id binding) form no group and therefore no summary.
    """
    if supplement_df.empty:
        return pd.DataFrame()

    identity_column = _stateful_identity_column(supplement_df)
    summaries: list[dict] = []
    for fighter, group in supplement_df.groupby(identity_column):
        total = len(group)
        wins = (group["won"] == 1).sum()
        losses = (group["result_label"] == "loss").sum()

        # Method rates — from real method strings, NaN if 0 wins
        complete_methods = group.loc[group["won"] == 1, "method"].map(_has_observed_win_method).all()
        if wins > 0 and (complete_methods or (
            not preserve_unknown_methods and not _retained_long_mask(group).any()
        )):
            methods = group.loc[group["won"] == 1, "method"].astype(object).map(
                lambda value: "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).lower()
            )
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
        # (date-aware for the date-scoped labels).
        if "organization" in group.columns:
            org_tiers = _encode_org_tiers(group).dropna()
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


def _amateur_summary_labels(names, identity_keys) -> np.ndarray:
    """Return the spelling labels of the amateur career summary (owner decision D10).

    The amateur summary is keyed by spelling, not by the stateful identity.
    The reviewed bound spellings of one UFCStats fighter share one label, his
    UFCStats spelling, when their stateful key is that fighter; under any
    other key (a contradicting UFCStats ID) they get none (no group, no
    match). Every other spelling is its own label, unchanged.
    """
    return np.asarray(
        [reviewed_spelling_binding_name(name, key) for name, key in zip(names, identity_keys)],
        dtype=object,
    )


def _compute_amateur_summary(supplement_df: pd.DataFrame, *, observed_owner_summary: bool = False) -> pd.DataFrame:
    """Compute per-fighter amateur career summary features from raw supplement rows."""
    if supplement_df.empty:
        return pd.DataFrame()

    if {"fighter", "won", "result_label"}.issubset(supplement_df.columns):
        long_df = supplement_df.copy()
    else:
        deduped_raw = _dedupe_supplement_rows(supplement_df)
        long_df = _supplement_rows_to_long_format(deduped_raw, expected_division="am")
        long_df = _drop_quarantined_supplement_identities(
            long_df,
            label="amateur",
        )
    if long_df.empty:
        return pd.DataFrame()

    identity_keys = (
        long_df["fighter_key"]
        if "fighter_key" in long_df.columns
        else long_df["fighter"].map(fighter_identity_key)
    )
    summaries: list[dict] = []
    labels = identity_keys if observed_owner_summary else _amateur_summary_labels(long_df["fighter"], identity_keys)
    for fighter, group in long_df.groupby(labels):
        total = len(group)
        wins = (group["won"] == 1).sum()
        losses = (group["result_label"] == "loss").sum()

        complete_methods = group.loc[group["won"] == 1, "method"].map(_has_observed_win_method).all()
        if wins > 0 and (complete_methods or (
            not observed_owner_summary and not _retained_long_mask(group).any()
        )):
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

class _NewFeatureContext:
    """Actual owner/date metadata for the internal NEW-only calculation."""

    def __init__(self, tracked_dates: pd.DataFrame, ambiguous_names: frozenset[str], supplement: pd.DataFrame):
        self.tracked_dates = tracked_dates
        self.ambiguous_names = ambiguous_names
        self.supplement = supplement


def _new_observed_owner_key(name, identity, ambiguous_names, observed_ids=None):
    """Use actual ID evidence; only reviewed bindings can supply a missing ID."""
    reviewed = fighter_identity_key(name, identity, ambiguous_name_keys=ambiguous_names)
    if reviewed is None:
        return None
    actual = normalize_ufcstats_id(identity)
    if actual is not None:
        return f"ufcstats:{actual}"
    if reviewed.startswith("ufcstats:"):
        return reviewed
    name_key = canonical_fighter_name_key(name)
    candidates = (observed_ids or {}).get(name_key, set())
    if name_key not in ambiguous_names and len(candidates) == 1:
        return "ufcstats:" + next(iter(candidates))
    return None


def build_features(
    fights_df: pd.DataFrame,
    *,
    raw_data_dir: Path | None = None,
    _new_context: _NewFeatureContext | None = None,
) -> pd.DataFrame:
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
    fights_df = fights_df.sort_values("event_date", kind="stable").reset_index(
        drop=True
    )
    fights_df["__fight_row_id"] = range(len(fights_df))
    raw_root = Path(raw_data_dir) if raw_data_dir is not None else None
    if _new_context is None:
        fights_df = _attach_fighter_identity_keys(fights_df, raw_data_dir=raw_root)
    elif type(_new_context) is not _NewFeatureContext or not {
        "__fighter_a_key", "__fighter_b_key"
    } <= set(fights_df.columns):
        raise ValueError("Internal observed NEW feature context required")

    # Step 1: Compute per-fight stats in long format
    ufc_per_fight = _compute_per_fight_stats(fights_df)
    per_fight = ufc_per_fight.copy()
    logger.info(f"Extracted {len(per_fight)} fighter-fight records")

    # Step 1b: Merge pre-UFC career supplement (Sherdog) if available.
    # These rows enrich rolling stats / career counters for debut fighters
    # but are filtered out of the final output (we only return UFC fight rows).
    pre_ufc_path = (
        _resolve_pre_ufc_supplement_path()
        if raw_root is None
        else _resolve_pre_ufc_supplement_path(raw_root)
    )
    _pre_ufc_marker = "__is_pre_ufc"
    pre_ufc_supplement = pd.DataFrame()
    loaded_supplement = (
        _load_pre_ufc_supplement(pre_ufc_path, include_reviewed=True, raw_data_dir=raw_root)
        if _new_context is None else _new_context.supplement.copy()
    )
    if _new_context is not None and not loaded_supplement.empty:
        # These owner keys were established before the genuine loader's
        # existing duplicate/reconciliation policy, never after collapse.
        owners = set(fights_df["__fighter_a_key"].dropna()) | set(fights_df["__fighter_b_key"].dropna())
        loaded_supplement = loaded_supplement.loc[loaded_supplement["fighter_key"].isin(owners)].copy()
    if _retained_long_mask(loaded_supplement).any():
        if _new_context is None:
            fights_df = _scope_retained_history_owners(fights_df, loaded_supplement)
        ufc_per_fight = _compute_per_fight_stats(fights_df)
        per_fight = ufc_per_fight.copy()
    tracked_dates = ufc_per_fight
    if _new_context is not None:
        tracked_dates = _new_context.tracked_dates
    pre_ufc_supplement = _filter_pre_ufc_rows_before_tracked_debut(
        loaded_supplement,
        tracked_dates,
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
        "fight_row_id",
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
        "event_name",
    }
    rolling_helper_cols.update(PROVENANCE_COLUMNS)

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
    if "fight_row_id" in a_rolling.columns:
        rename_map["fight_row_id"] = "__fight_row_id"
    a_rolling = a_rolling.rename(columns=rename_map)
    a_merge_cols = [rolling_merge_key, "__fight_row_id", "event_date"] + [
        c for c in a_rolling.columns if c.startswith("a_")
    ]
    a_rolling_deduped = a_rolling[
        [c for c in a_merge_cols if c in a_rolling.columns]
    ].drop_duplicates(
        subset=[rolling_merge_key, "__fight_row_id", "event_date"],
        keep="last",
    )

    # For fighter_b
    b_rolling = rolling_df.copy()
    rename_b = {
        c: f"b_roll_{c.replace('roll_', '')}" if c.startswith("roll_") else f"b_{c}"
        for c in rolling_df.columns
        if c not in rolling_helper_cols
    }
    rename_b["fight_row_id"] = "__fight_row_id"
    b_rolling = b_rolling.rename(columns=rename_b)
    b_merge_cols = [rolling_merge_key, "__fight_row_id", "event_date"] + [
        c for c in b_rolling.columns if c.startswith("b_")
    ]
    b_rolling_deduped = b_rolling[
        [c for c in b_merge_cols if c in b_rolling.columns]
    ].drop_duplicates(
        subset=[rolling_merge_key, "__fight_row_id", "event_date"],
        keep="last",
    )

    # Merge fighter A rolling stats
    features = fights_df.merge(
        a_rolling_deduped,
        left_on=["__fighter_a_key", "__fight_row_id", "event_date"],
        right_on=[rolling_merge_key, "__fight_row_id", "event_date"],
        how="left",
    ).drop(columns=[rolling_merge_key], errors="ignore")

    # Merge fighter B rolling stats
    features = features.merge(
        b_rolling_deduped,
        left_on=["__fighter_b_key", "__fight_row_id", "event_date"],
        right_on=[rolling_merge_key, "__fight_row_id", "event_date"],
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
                    combined = prob.combine_first(existing)
                    if "method_observation_policy" in features:
                        verified = features["method_observation_policy"].notna()
                        combined.loc[verified] = existing.loc[verified]
                    features[feature_col] = combined
                else:
                    if "method_observation_policy" in features:
                        prob = prob.where(features["method_observation_policy"].isna())
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
        _detect_weight_class_moves(features, preserve_unknown=_new_context is not None)

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
    features = features.sort_values(
        ["event_date", "__fight_row_id"],
        kind="stable",
    )
    h2h: dict[tuple[str, str], list[str]] = {}
    is_rematch_vals = []
    h2h_diff_vals = []
    for _, row in features.iterrows():
        fa = str(row.get("__fighter_a_key", ""))
        fb = str(row.get("__fighter_b_key", ""))
        pair = tuple(sorted([fa, fb]))
        prior = h2h.get(pair, [])
        # A prior draw or no-contest establishes a rematch without a winner.
        if pair in h2h:
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
        _pre_ufc_summary = _compute_pre_ufc_summary(
            pre_ufc_supplement.loc[~_retained_long_mask(pre_ufc_supplement)],
            preserve_unknown_methods=_new_context is not None,
        )
        if not _pre_ufc_summary.empty:
            _pre_ufc_cols = [c for c in _pre_ufc_summary.columns if c != "fighter"]
            # The summary is keyed by the same stateful identity as the rolling
            # statistics (fighter_key: UFCStats id for ambiguous spellings,
            # normalized spelling otherwise), so it is merged on the fight
            # row's identity keys rather than on the display spelling. A
            # spelling shared by two UFCStats ids can no longer attach one
            # person's regional record to the other's fight.
            _summary_identity = "__fighter_a_key" if "__fighter_a_key" in features.columns else "fighter_a"
            # Merge for fighter_a
            _a_summary = _pre_ufc_summary.rename(
                columns={c: f"a_{c}" for c in _pre_ufc_cols}
            )
            features = features.merge(
                _a_summary, left_on=_summary_identity, right_on="fighter",
                how="left"
            ).drop(columns=["fighter"], errors="ignore")
            # Merge for fighter_b
            _summary_identity_b = "__fighter_b_key" if "__fighter_b_key" in features.columns else "fighter_b"
            _b_summary = _pre_ufc_summary.rename(
                columns={c: f"b_{c}" for c in _pre_ufc_cols}
            )
            features = features.merge(
                _b_summary, left_on=_summary_identity_b, right_on="fighter",
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

    features = _overlay_retained_career_summaries(features, pre_ufc_supplement)

    # Feature absence is unknown, never a zero-fight synthetic observation.
    for col in _PRE_UFC_SUMMARY_COLUMNS:
        for prefix in ("a_", "b_", "diff_"):
            target = f"{prefix}{col}"
            if target not in features.columns:
                features[target] = np.nan

    amateur_path = (
        _resolve_amateur_supplement_path()
        if raw_root is None
        else _resolve_amateur_supplement_path(raw_root)
    )
    _amateur_summary = pd.DataFrame()
    _amateur_long = _load_amateur_supplement(
        amateur_path,
        include_reviewed=True,
        raw_data_dir=raw_root,
        _new_ambiguous_names=None if _new_context is None else _new_context.ambiguous_names,
    )
    if _new_context is not None and not _amateur_long.empty:
        legacy = ~_retained_long_mask(_amateur_long)
        owners = set(features["__fighter_a_key"].dropna()) | set(features["__fighter_b_key"].dropna())
        _amateur_long = _amateur_long.loc[~legacy | _amateur_long["fighter_key"].isin(owners)].copy()
    if not _amateur_long.empty:
        _amateur_summary = _compute_amateur_summary(_amateur_long.loc[~_retained_long_mask(_amateur_long)],
                                                   observed_owner_summary=_new_context is not None)

    if not _amateur_summary.empty:
        _amateur_cols = [c for c in _amateur_summary.columns if c != "fighter"]
        # Merged on the summary's spelling labels (owner decision D10): a
        # reviewed bound spelling under his UFCStats spelling, every other
        # name as it is.
        for _side in ("a", "b"):
            _key_column = f"__fighter_{_side}_key"
            _keys = (
                features[_key_column]
                if _key_column in features.columns
                else features[f"fighter_{_side}"].map(fighter_identity_key)
            )
            features[f"__amateur_{_side}_label"] = (
                _amateur_summary_labels(features[f"fighter_{_side}"], _keys)
                if _new_context is None else _keys
            )
        _a_amateur = _amateur_summary.rename(columns={c: f"a_{c}" for c in _amateur_cols})
        features = features.merge(
            _a_amateur,
            left_on="__amateur_a_label",
            right_on="fighter",
            how="left",
        ).drop(columns=["fighter"], errors="ignore")
        _b_amateur = _amateur_summary.rename(columns={c: f"b_{c}" for c in _amateur_cols})
        features = features.merge(
            _b_amateur,
            left_on="__amateur_b_label",
            right_on="fighter",
            how="left",
        ).drop(columns=["fighter", "__amateur_a_label", "__amateur_b_label"], errors="ignore")
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

    features = _overlay_retained_career_summaries(features, _amateur_long, amateur=True)

    for col in _AMATEUR_SUMMARY_COLUMNS:
        for prefix in ("a_", "b_", "diff_"):
            target = f"{prefix}{col}"
            if target not in features.columns:
                features[target] = np.nan

    # Step 7: Drop rows with insufficient data (first fights for both fighters)
    features["has_data"] = features.get("a_num_fights", pd.Series(0)) + features.get("b_num_fights", pd.Series(0))

    logger.info(f"Built {len(features)} fight feature rows with {len(features.columns)} columns")
    return features.drop(
        columns=["__fighter_a_key", "__fighter_b_key", "__fight_row_id"],
        errors="ignore",
    )


def build_new_features(
    accepted_fights: pd.DataFrame,
    new_fights: pd.DataFrame,
    *,
    raw_data_dir: Path | None = None,
    historical_identity_associations: list[dict] | None = None,
) -> pd.DataFrame:
    """Calculate only NEW targets with the genuine time-directed dependencies.

    Primary owners need their prior timelines. Core SOS additionally needs the
    prior timeline of each last-five opponent, through that original matchup.
    No opponent-of-opponent graph or historical feature reproduction is needed.
    """
    TARGET_FIELD = "__incremental_refresh_target"
    if TARGET_FIELD in accepted_fights or TARGET_FIELD in new_fights:
        raise ValueError("Private NEW target field already exists")
    if new_fights.empty:
        return new_fights.copy()
    for frame in (accepted_fights, new_fights):
        if not {"event_date", "fighter_a", "fighter_b"} <= set(frame.columns):
            raise ValueError("Observed historical/NEW participant columns required")
    history, targets = accepted_fights.copy(), new_fights.copy()
    history[TARGET_FIELD] = ""
    targets[TARGET_FIELD] = [str(index) for index in range(len(targets))]
    observed = pd.concat([history, targets], ignore_index=True, sort=False)
    collision_rows = [
        {"name": name, "fighter_id": identity}
        for side in ("a", "b")
        for name, identity in zip(
            observed["fighter_" + side],
            observed.get("fighter_" + side + "_id", pd.Series(None, index=observed.index)),
        )
    ]
    # Accepted draw/NC and older source rows may carry blank IDs. Associate
    # those calculation-only roles with existing observed IDs when the entire
    # accepted/new/selected inventory domain has exactly one owner for the
    # canonical name. Stored historical cells remain untouched. Collisions
    # still require the existing explicit reviewed association.
    identity_evidence = list(collision_rows)
    identity_evidence.extend({"name": item["name"], "fighter_id": item["owner_key"].removeprefix("ufcstats:")}
                             for item in historical_identity_associations or [])
    raw_root = Path(raw_data_dir) if raw_data_dir is not None else RAW_DATA_DIR
    inventory_path = selected_path("profiles", raw_root / "ufc_fighters_scraped.csv")
    if inventory_path.exists():
        inventory = pd.read_csv(inventory_path, usecols=["name", "fighter_url"])
        identity_evidence.extend({"name": row.name, "fighter_id": row.fighter_url}
                                 for row in inventory.itertuples(index=False))
    elif active_sources() is not None:
        raise ValueError("Selected fighter inventory is absent")
    ambiguous = frozenset(set(_training_ambiguous_name_keys(raw_data_dir)) |
                          set(derive_ambiguous_fighter_name_keys(identity_evidence)))
    observed_ids = {}
    for evidence in identity_evidence:
        identity = normalize_ufcstats_id(evidence["fighter_id"])
        if identity is not None:
            observed_ids.setdefault(canonical_fighter_name_key(evidence["name"]), set()).add(identity)
    for side in ("a", "b"):
        observed["__fighter_" + side + "_key"] = [
            _new_observed_owner_key(item["name"], item["fighter_id"], ambiguous, observed_ids)
            for item in collision_rows[(0 if side == "a" else len(observed)):
                                       (len(observed) if side == "a" else 2 * len(observed))]
        ]
    exact_associations = {}
    for association in historical_identity_associations or []:
        index, side = association["observed_row"], association["side"]
        if side not in {"a", "b"} or not isinstance(index, int) or not 0 <= index < len(history):
            raise ValueError("Historical identity association has an invalid role")
        if normalize_ufcstats_id(observed.at[index, "fighter_" + side + "_id"]) is not None:
            raise ValueError("Historical identity association would replace a stored stable ID")
        key = association["owner_key"]
        if (normalize_ufcstats_id(key.removeprefix("ufcstats:")) is None
                or canonical_fighter_name_key(association["name"]) != canonical_fighter_name_key(observed.at[index, "fighter_" + side])):
            raise ValueError("Historical identity association differs from its accepted role")
        current = observed.at[index, "__fighter_" + side + "_key"]
        if current is not None and not pd.isna(current) and current != key:
            raise ValueError("Historical source association contradicts an observed unique owner")
        observed.at[index, "__fighter_" + side + "_key"] = key
        exact_associations[index, side] = association
    dates = pd.to_datetime(observed["event_date"], errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
    target_indices = list(range(len(history), len(observed)))
    if dates.iloc[target_indices].isna().any():
        raise ValueError("NEW feature target chronology is unresolved")
    names_by_key = {}
    timelines = {}
    unresolved = []
    for index, row in observed.iterrows():
        for side in ("a", "b"):
            key = row["__fighter_" + side + "_key"]
            name = canonical_fighter_name_key(row["fighter_" + side])
            if key is None or pd.isna(key):
                unresolved.append((index, name, normalize_ufcstats_id(row.get("fighter_" + side + "_id"))))
            else:
                names_by_key.setdefault(key, set()).add(name)
                timelines.setdefault(key, []).append(index)

    def prior(key, cutoff):
        if key is None or pd.isna(key):
            raise UnresolvedTrainingFighterIdentityError("NEW target owner is unresolved")
        owned = timelines.get(key, [])
        if any(pd.isna(dates.iloc[index]) for index in owned):
            raise ValueError("Required NEW owner chronology is unresolved")
        potential = names_by_key[key]
        for index, name, actual_id in unresolved:
            needed = name in potential or (actual_id is not None and key == f"ufcstats:{actual_id}")
            if needed and (pd.isna(dates.iloc[index]) or (dates.iloc[index], index) < cutoff):
                raise UnresolvedTrainingFighterIdentityError("Required prior owner association is unresolved")
        return sorted((index for index in owned if (dates.iloc[index], index) < cutoff),
                      key=lambda index: (dates.iloc[index], index))

    included = set(target_indices)
    required_owners = set()
    for index in target_indices:
        for side in ("a", "b"):
            key = observed.at[index, "__fighter_" + side + "_key"]
            primary = prior(key, (dates.iloc[index], index))
            required_owners.add(key)
            included.update(primary)
            for matchup in primary[-5:]:
                past = observed.loc[matchup]
                opponent_side = "b" if past["__fighter_a_key"] == key else "a"
                opponent = past["__fighter_" + opponent_side + "_key"]
                if opponent is None or pd.isna(opponent):
                    raise UnresolvedTrainingFighterIdentityError("Required last-five opponent owner is unresolved")
                required_owners.add(opponent)
                included.update(prior(opponent, (dates.iloc[matchup], matchup)))

    metadata_indices = sorted({index for owner in required_owners for index in timelines[owner]})
    metadata_columns = ["event_date", "event_name", "fighter_a", "fighter_b",
                        "fighter_a_id", "fighter_b_id", "__fighter_a_key", "__fighter_b_key"]
    metadata = observed.loc[metadata_indices, [name for name in metadata_columns if name in observed]].copy()
    metadata["event_date"] = dates.loc[metadata_indices]
    context = observed.loc[sorted(included)].copy()
    context["event_date"] = dates.loc[sorted(included)]
    associations = []
    for index, row in context.iterrows():
        for side in ("a", "b"):
            if normalize_ufcstats_id(row.get("fighter_" + side + "_id")) is None:
                key = row["__fighter_" + side + "_key"]
                if key is not None and not pd.isna(key):
                    associations.append(exact_associations.get((int(index), side), {
                        "observed_row": int(index), "side": side,
                        "name": row["fighter_" + side], "owner_key": key,
                        "basis": "existing observed unique ID or explicit reviewed binding"}))
    supplement = _load_pre_ufc_supplement(
        _resolve_pre_ufc_supplement_path(raw_data_dir), include_reviewed=True,
        raw_data_dir=raw_data_dir,
        _new_ambiguous_names=ambiguous,
    )
    calculated = build_features(context, raw_data_dir=raw_data_dir,
                                _new_context=_NewFeatureContext(metadata, ambiguous, supplement))
    if TARGET_FIELD not in calculated:
        raise ValueError("Scientific feature producer lost NEW target identity")
    result = calculated.loc[calculated[TARGET_FIELD].isin(targets[TARGET_FIELD])].copy()
    expected = targets[TARGET_FIELD].tolist()
    if len(result) != len(expected) or result[TARGET_FIELD].duplicated().any() or set(result[TARGET_FIELD]) != set(expected):
        raise ValueError("Scientific NEW target occurrence relationship differs")
    result = result.set_index(TARGET_FIELD).loc[expected].reset_index(drop=True)
    result.attrs["new_feature_context"] = {"historical_role_associations": associations,
                                            "context_rows": len(context), "observed_rows": len(observed)}
    return result


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


def _detect_weight_class_moves(features: pd.DataFrame, *, preserve_unknown: bool = False) -> None:
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
    for idx, row in features.sort_values(
        ["event_date", "__fight_row_id"],
        kind="stable",
    ).iterrows():
        wc_w = _wc_to_weight(row.get("weight_class"))
        if wc_w is None:
            wc_move_a[idx] = np.nan if preserve_unknown else 0
            wc_move_b[idx] = np.nan if preserve_unknown else 0
            continue

        # Compute flag using prior mode BEFORE updating history
        fa = row.get("__fighter_a_key") or row.get("fighter_a")
        fb = row.get("__fighter_b_key") or row.get("fighter_b")
        fa_home = fighter_wc_history[fa].most_common(1)[0][0] if fa and fa in fighter_wc_history else None
        fb_home = fighter_wc_history[fb].most_common(1)[0][0] if fb and fb in fighter_wc_history else None
        wc_move_a[idx] = (int(wc_w != fa_home) if fa_home else np.nan) if preserve_unknown else int(bool(fa_home and wc_w != fa_home))
        wc_move_b[idx] = (int(wc_w != fb_home) if fb_home else np.nan) if preserve_unknown else int(bool(fb_home and wc_w != fb_home))

        # Now update history with this fight
        for side in ["a", "b"]:
            fighter = row.get(f"__fighter_{side}_key") or row.get(
                f"fighter_{side}"
            )
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
