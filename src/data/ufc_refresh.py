"""Helpers for building a refreshed UFC training dataset from UFCStats scrapes."""

from __future__ import annotations

import json

import logging
import re
import time
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import RAW_DATA_DIR, UFCSTATS_BASE_URL
from src.data.event_context import infer_empty_arena
from src.data.fight_results import (
    RESULT_CLASS_NORMAL,
    has_result_policy_marker,
    validate_marked_result,
    winner_for_side,
)
from src.data.io_utils import csv_pair_manifest_path, read_csv_pair_verified
from src.data.kaggle_loader import (
    _parse_height_cm,
    _parse_reach_cm,
    _safe_float,
    load_kaggle_dataset,
)
from src.data.name_utils import (
    KNOWN_AMBIGUOUS_FIGHTER_NAME_KEYS,
    canonicalize_reviewed_fighter_name,
    derive_ambiguous_fighter_name_keys,
    fighter_identity_is_ambiguous,
    fighter_identity_key,
    normalize_ufcstats_id,
    normalize_person_name as _normalize_fighter_name,
)
from src.data.profile_integrity import (
    event_specific_weight_lbs,
    load_profile_decisions,
    source_entity_id,
)
from src.data.reviewed_roster_recovery import (
    permanent_profile_locks,
    reviewed_profile_field_holds,
    reviewed_profile_recovery_observations,
)
from src.data.reviewed_roster_identity import (
    load_reviewed_roster_identity_registry,
    reviewed_profile_field_decisions as reviewed_identity_field_decisions,
    reviewed_profile_observations as reviewed_identity_profile_observations,
)
from src.data.rankings_scraper import _canonical_wc
from src.data.ufcstats_http import DEFAULT_UFCSTATS_HEADERS, request_ufcstats
from src.features.stance_utils import encode_stance

logger = logging.getLogger(__name__)

AUTO_REFRESH_DATASET_PATH = RAW_DATA_DIR / "ufc-auto-refresh.csv"
SCRAPED_FIGHTS_PATH = RAW_DATA_DIR / "ufc_fights_scraped.csv"
SCRAPED_FIGHTERS_PATH = RAW_DATA_DIR / "ufc_fighters_scraped.csv"
PROFILE_SUPPLEMENT_PATH = RAW_DATA_DIR / "ufc_fighters_profile_supplement.csv"
OFFICIAL_ACTIVE_ROSTER_PATH = RAW_DATA_DIR / "ufc_active_roster_official.csv"
PULLED_FIGHT_RESULTS_PATH = RAW_DATA_DIR / "ufc-fight-results.csv"
PULLED_FIGHT_STATS_PATH = RAW_DATA_DIR / "ufc-fight-stats.csv"
EVENT_DATES_CACHE_PATH = RAW_DATA_DIR / "ufc-event-dates.csv"
PIERCE_FIGHT_IDENTITIES_PATH = RAW_DATA_DIR / "github_PierceHampton.csv"
_ROUND_SECONDS = 5 * 60
_BOUT_SPLIT_RE = re.compile(r"\s+vs\.?\s+", flags=re.IGNORECASE)
_LANDED_ATTEMPTED_RE = re.compile(r"(?P<landed>\d+)\s+of\s+(?P<attempted>\d+)", flags=re.IGNORECASE)
_UFCSTATS_HEADERS = dict(DEFAULT_UFCSTATS_HEADERS)
DEFAULT_VARIANT_OVERLAP_START = pd.Timestamp("2022-01-01")
DEFAULT_VARIANT_OVERLAP_END = pd.Timestamp("2024-12-31")
TRAINING_DATASET_VARIANTS = (
    "legacy_only",
    "append_only_2026",
    "pulled_all",
    "pulled_all_plus_legacy_market",
    "hybrid_legacy_first",
    "hybrid_pulled_2022plus",
    "best_of_both_field_level",
    "best_of_both_full_history",
)


def _resolve_raw_artifact_path(
    explicit_path: Path | None,
    *,
    raw_data_dir: Path | None,
    relative_path: str | Path,
    canonical_path: Path,
) -> Path:
    """Resolve one raw artifact without falling back outside an explicit raw root."""
    if explicit_path is not None:
        return Path(explicit_path)
    if raw_data_dir is not None:
        return Path(raw_data_dir) / Path(relative_path)
    return Path(canonical_path)

_ambiguous_name_keys_cache: tuple[str, float, frozenset[str]] | None = None


def _ambiguous_name_keys(
    inventory_path: Path | None = None,
) -> frozenset[str]:
    """Derive collision groups from the current UFCStats fighter inventory."""
    global _ambiguous_name_keys_cache
    inventory_path = (
        Path(inventory_path)
        if inventory_path is not None
        else SCRAPED_FIGHTERS_PATH
    )
    try:
        resolved_path = str(inventory_path.resolve())
        mtime = inventory_path.stat().st_mtime
    except OSError:
        return KNOWN_AMBIGUOUS_FIGHTER_NAME_KEYS
    if (
        _ambiguous_name_keys_cache is not None
        and _ambiguous_name_keys_cache[:2] == (resolved_path, mtime)
    ):
        return _ambiguous_name_keys_cache[2]
    try:
        inventory = pd.read_csv(
            inventory_path,
            usecols=["name", "fighter_url"],
        )
        derived = derive_ambiguous_fighter_name_keys(
            (
                {"name": row.name, "fighter_id": row.fighter_url}
                for row in inventory.itertuples(index=False)
            )
        )
    except (OSError, ValueError) as exc:
        logger.warning(
            "Could not derive fighter identity collisions from %s: %s",
            inventory_path,
            exc,
        )
        derived = frozenset()
    resolved = frozenset(set(KNOWN_AMBIGUOUS_FIGHTER_NAME_KEYS) | set(derived))
    _ambiguous_name_keys_cache = (resolved_path, mtime, resolved)
    return resolved


# Exact offline recovery for collision rows outside the Pierce participant-URL
# artifact. New missing rows use the bounded fight-page recovery hook below;
# they are never guessed from a name-only fighter index.
_FIGHT_PARTICIPANT_ID_OVERRIDES: dict[tuple[str, str], str] = {
    ("836a188a90e479a5", _normalize_fighter_name("Jean Silva")): "52ef95b5860fb28c",
    ("fc4ab243d21b795d", _normalize_fighter_name("Lance Gibson Jr.")): "07bc580bdf3eb9c8",
    ("668a44403a130043", _normalize_fighter_name("Bruno Silva")): "294aa73dbf37d281",
    ("de1a3734be60e6a1", _normalize_fighter_name("Jean Silva")): "52ef95b5860fb28c",
    ("57ff0eb2351979c4", _normalize_fighter_name("Bruno Silva")): "294aa73dbf37d281",
    ("fc9168fd102e2e66", _normalize_fighter_name("Bruno Silva")): "294aa73dbf37d281",
    ("2329fd5f35b6d321", _normalize_fighter_name("Jean Silva")): "9211aae062b799d6",
    ("e0ecae31a86cc648", _normalize_fighter_name("Jean Silva")): "52ef95b5860fb28c",
    ("49dc6ce420444a8c", _normalize_fighter_name("Victor Valenzuela")): "078695e385ec2f57",
    ("fe7c434ced7759e5", _normalize_fighter_name("Mike Davis")): "c8661e204c66f325",
    ("87bda6d520fe9677", _normalize_fighter_name("Mike Davis")): "fb3e61720be4690c",
    ("8993e293a63bb5f3", _normalize_fighter_name("Tommy Petersen")): "7aabd61419f747d1",
    ("689384b01b40ef5c", _normalize_fighter_name("Lance Gibson Jr.")): "07bc580bdf3eb9c8",
    ("2e7f8d0ad385876d", _normalize_fighter_name("Victor Valenzuela")): "078695e385ec2f57",
    ("20d74ed23d3e9b3a", _normalize_fighter_name("Bruno Silva")): "294aa73dbf37d281",
    ("d0b24570ede9ca31", _normalize_fighter_name("Victor Valenzuela")): "de277a4abcfeea46",
    ("3859314324918eeb", _normalize_fighter_name("Mike Davis")): "fb3e61720be4690c",
    ("a815dafc89adf33b", _normalize_fighter_name("Thomas Petersen")): "764d39074a352e33",
    ("c1ed08a1f760f9c7", _normalize_fighter_name("Thomas Petersen")): "764d39074a352e33",
    ("0760bc02280574dd", _normalize_fighter_name("Thomas Petersen")): "764d39074a352e33",
    ("d7070ab1b2144f8d", _normalize_fighter_name("Thomas Petersen")): "764d39074a352e33",
}
_LANDED_ATTEMPTED_PULL_COLUMNS = {
    "SIG.STR.": "sig_str",
    "TOTAL STR.": "total_str",
    "TD": "td",
    "HEAD": "head",
    "BODY": "body",
    "LEG": "leg",
    "DISTANCE": "distance",
    "CLINCH": "clinch",
    "GROUND": "ground",
}
_PULLED_COUNT_STAT_FIELDS = [
    "sig_str_landed",
    "sig_str_attempted",
    "total_str_landed",
    "total_str_attempted",
    "td_landed",
    "td_attempted",
    "kd",
    "sub_att",
    "rev",
    "ctrl_seconds",
    "head_landed",
    "head_attempted",
    "body_landed",
    "body_attempted",
    "leg_landed",
    "leg_attempted",
    "distance_landed",
    "distance_attempted",
    "clinch_landed",
    "clinch_attempted",
    "ground_landed",
    "ground_attempted",
]
_PULLED_RATE_STAT_FIELDS = [
    "total_str_pm",
    "sig_str_share",
    "head_str_share",
    "body_str_share",
    "leg_str_share",
    "distance_str_share",
    "clinch_str_share",
    "ground_str_share",
    "control_share",
    "control_per_td",
    "finish_threat",
]
_PULLED_PREFERRED_FIELDS = {
    "fighter_a_id",
    "fighter_b_id",
    "event_name",
    "location",
    "weight_class",
    "winner",
    "method",
    "title_bout",
    "num_rounds",
    "finish_round",
    "finish_time",
    "total_fight_time_secs",
    "empty_arena",
    *[f"a_{field}" for field in _PULLED_COUNT_STAT_FIELDS + _PULLED_RATE_STAT_FIELDS],
    *[f"b_{field}" for field in _PULLED_COUNT_STAT_FIELDS + _PULLED_RATE_STAT_FIELDS],
}
_LEGACY_STATIC_PROFILE_FIELDS = ("height", "reach")
_LEGACY_PREFERRED_FIELDS = {
    "country",
    "gender",
    "a_height",
    "a_reach",
    "a_weight",
    "a_age",
    "a_stance",
    "a_wins",
    "a_losses",
    "a_draws",
    "a_win_streak",
    "a_lose_streak",
    "a_longest_win_streak",
    "a_total_rounds",
    "a_title_bouts",
    "a_wins_ko",
    "a_wins_sub",
    "a_wins_dec",
    "a_wins_dec_maj",
    "a_wins_dec_split",
    "a_wins_tko_doc",
    "a_odds",
    "a_expected_value",
    "a_wc_rank",
    "a_pfp_rank",
    "a_dec_odds",
    "a_sub_odds",
    "a_ko_odds",
    "b_height",
    "b_reach",
    "b_weight",
    "b_age",
    "b_stance",
    "b_wins",
    "b_losses",
    "b_draws",
    "b_win_streak",
    "b_lose_streak",
    "b_longest_win_streak",
    "b_total_rounds",
    "b_title_bouts",
    "b_wins_ko",
    "b_wins_sub",
    "b_wins_dec",
    "b_wins_dec_maj",
    "b_wins_dec_split",
    "b_wins_tko_doc",
    "b_odds",
    "b_expected_value",
    "b_wc_rank",
    "b_pfp_rank",
    "b_dec_odds",
    "b_sub_odds",
    "b_ko_odds",
    "better_rank",
}
_PULLED_ALL_LEGACY_MARKET_OVERLAY_FIELDS = {
    "a_odds",
    "b_odds",
    "a_wc_rank",
    "b_wc_rank",
    "a_pfp_rank",
    "b_pfp_rank",
    "a_ko_odds",
    "a_sub_odds",
    "a_dec_odds",
    "b_ko_odds",
    "b_sub_odds",
    "b_dec_odds",
}


def _decimal_to_american_odds(value) -> float:
    """Convert decimal odds to American odds, preserving missing/invalid values as NaN."""
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or float(numeric) <= 1.0:
        return np.nan
    numeric = float(numeric)
    if numeric >= 2.0:
        return (numeric - 1.0) * 100.0
    return -100.0 / (numeric - 1.0)


def _load_historical_moneyline_rows(
    historical_odds_dir: Path | None = None,
) -> pd.DataFrame:
    """Load historical odds from an explicit directory or the canonical loader."""
    from src.data.selected_source_inputs import active_sources, selected_path
    if active_sources() is not None:
        historical_odds_dir = selected_path("historical_odds", RAW_DATA_DIR / "historical_odds")
    if historical_odds_dir is None:
        try:
            from src.data.historical_backfill import load_all_historical_odds
        except Exception:
            return pd.DataFrame()
        return load_all_historical_odds()

    history_dir = Path(historical_odds_dir)
    source_paths = [
        (
            history_dir / "historical_odds_pre2022_from_cleaned.csv",
            "legacy_pre2022",
        ),
        (history_dir / "historical_odds_bfo.csv", "bfo"),
        *[
            (path, "bfo")
            for path in sorted(
                history_dir.glob("historical_odds_bfo_recovered_*.csv")
            )
        ],
        (history_dir / "historical_odds.csv", "odds_api"),
    ]
    parts: list[pd.DataFrame] = []
    for path, source_kind in source_paths:
        if not path.is_file():
            continue
        try:
            loaded = pd.read_csv(path)
        except (OSError, ValueError) as exc:
            if active_sources() is not None:
                raise
            logger.warning("Could not load staged historical odds %s: %s", path, exc)
            continue
        loaded["source_kind"] = source_kind
        loaded["source_file"] = path.name
        offsets = pd.to_numeric(
            loaded.get("offset_days", pd.Series(np.nan, index=loaded.index)),
            errors="coerce",
        )
        loaded["observed_at"] = pd.to_datetime(
            loaded.get("query_date", pd.Series(pd.NaT, index=loaded.index)),
            errors="coerce",
            utc=True,
        )
        loaded["hours_to_start"] = offsets * 24.0
        loaded["verified_prefight"] = (
            source_kind == "odds_api"
        ) & offsets.notna() & offsets.gt(0.0)
        parts.append(loaded)
    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True, sort=False)
    canonical_columns = (
        "event_date",
        "fighter_a",
        "fighter_b",
        "query_date",
        "offset_days",
        "a_fair_prob",
        "b_fair_prob",
        "a_decimal_odds",
        "b_decimal_odds",
        "num_bookmakers",
        "source_kind",
        "source_file",
        "observed_at",
        "hours_to_start",
        "verified_prefight",
    )
    for column in canonical_columns:
        if column not in combined.columns:
            combined[column] = np.nan
    for column in (
        "a_decimal_odds",
        "b_decimal_odds",
        "a_fair_prob",
        "b_fair_prob",
    ):
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    a_decimal = combined["a_decimal_odds"]
    b_decimal = combined["b_decimal_odds"]
    both_decimal = a_decimal.notna() & b_decimal.notna()
    invalid_decimal = both_decimal & (
        (a_decimal <= 1.0) | (b_decimal <= 1.0)
    )
    duplicated_heavy_favorite = (
        both_decimal
        & np.isclose(a_decimal, b_decimal, rtol=0.0, atol=1e-12)
        & (a_decimal < 2.0)
        & (((1.0 / a_decimal) + (1.0 / b_decimal)) > 1.20)
    )
    combined = combined.loc[
        ~(invalid_decimal | duplicated_heavy_favorite)
    ].copy()
    if combined.empty:
        return combined
    combined["__key_date"] = pd.to_datetime(
        combined["event_date"], errors="coerce", format="mixed"
    ).dt.strftime("%Y-%m-%d")
    combined["__key_a"] = combined["fighter_a"].apply(_normalize_fighter_name)
    combined["__key_b"] = combined["fighter_b"].apply(_normalize_fighter_name)
    if "offset_days" in combined.columns:
        combined["__key_offset"] = pd.to_numeric(
            combined["offset_days"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        combined["__key_offset"] = 0
    combined = combined.drop_duplicates(
        subset=["__key_date", "__key_a", "__key_b", "__key_offset"],
        keep="last",
    )
    return combined.drop(
        columns=["__key_date", "__key_a", "__key_b", "__key_offset"]
    )


def _historical_moneyline_overlay(
    keyed_frame: pd.DataFrame,
    *,
    historical_odds_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Build a fight-keyed historical moneyline overlay from ALL odds sources.

    Uses ``load_all_historical_odds()`` which consolidates The Odds API,
    pre-2022 legacy, and BestFightOdds scrapes into one deduplicated frame.

    For each fight we keep the closest available pre-fight snapshot
    (lowest ``offset_days``), then orient the decimal odds to the keyed frame's
    fighter order before exposing ``a_odds__historical_overlay`` and
    ``b_odds__historical_overlay``.
    """
    if keyed_frame.empty:
        return pd.DataFrame(columns=["fight_key", "a_odds__historical_overlay", "b_odds__historical_overlay"])

    historical_df = _load_historical_moneyline_rows(historical_odds_dir)

    if historical_df.empty:
        return pd.DataFrame(columns=["fight_key", "a_odds__historical_overlay", "b_odds__historical_overlay"])

    required_cols = {"event_date", "fighter_a", "fighter_b", "a_decimal_odds", "b_decimal_odds"}
    if not required_cols.issubset(historical_df.columns):
        return pd.DataFrame(columns=["fight_key", "a_odds__historical_overlay", "b_odds__historical_overlay"])

    historical = historical_df.copy()
    historical = historical.dropna(subset=["event_date", "fighter_a", "fighter_b"])
    if historical.empty:
        return pd.DataFrame(columns=["fight_key", "a_odds__historical_overlay", "b_odds__historical_overlay"])

    historical["event_date"] = pd.to_datetime(historical["event_date"], errors="coerce", format="mixed")
    historical = historical.dropna(subset=["event_date"])
    if historical.empty:
        return pd.DataFrame(columns=["fight_key", "a_odds__historical_overlay", "b_odds__historical_overlay"])

    historical["fight_key"] = _fight_key_series(historical)
    historical = historical.dropna(subset=["fight_key"])
    if historical.empty:
        return pd.DataFrame(columns=["fight_key", "a_odds__historical_overlay", "b_odds__historical_overlay"])

    sort_cols = [column for column in ["offset_days", "query_date", "num_bookmakers"] if column in historical.columns]
    ascending = []
    for column in sort_cols:
        if column == "num_bookmakers":
            ascending.append(False)
        elif column == "offset_days":
            ascending.append(False)  # descending: prefer opening (largest offset) over closing
        else:
            ascending.append(True)
    if sort_cols:
        historical = historical.sort_values(sort_cols, ascending=ascending)
    historical = historical.drop_duplicates(subset=["fight_key"], keep="first")

    overlay = historical[
        ["fight_key", "fighter_a", "fighter_b", "a_decimal_odds", "b_decimal_odds"]
    ].copy()
    overlay["a_decimal_odds"] = overlay["a_decimal_odds"].apply(_decimal_to_american_odds)
    overlay["b_decimal_odds"] = overlay["b_decimal_odds"].apply(_decimal_to_american_odds)
    overlay = overlay.rename(
        columns={
            "fighter_a": "__hist_fighter_a",
            "fighter_b": "__hist_fighter_b",
            "a_decimal_odds": "a_odds__historical_overlay",
            "b_decimal_odds": "b_odds__historical_overlay",
        }
    )

    pulled_lookup = keyed_frame[["fight_key", "fighter_a", "fighter_b"]].copy()
    overlay = pulled_lookup.merge(overlay, on="fight_key", how="left")
    reverse_mask = (
        overlay["__hist_fighter_a"].notna()
        & (overlay["fighter_a"].apply(_normalize_fighter_name) == overlay["__hist_fighter_b"].apply(_normalize_fighter_name))
        & (overlay["fighter_b"].apply(_normalize_fighter_name) == overlay["__hist_fighter_a"].apply(_normalize_fighter_name))
    )
    if reverse_mask.any():
        left_values = overlay.loc[reverse_mask, "a_odds__historical_overlay"].copy()
        right_values = overlay.loc[reverse_mask, "b_odds__historical_overlay"].copy()
        overlay.loc[reverse_mask, "a_odds__historical_overlay"] = right_values
        overlay.loc[reverse_mask, "b_odds__historical_overlay"] = left_values

    return overlay[["fight_key", "a_odds__historical_overlay", "b_odds__historical_overlay"]]


def _historical_opening_moneyline_overlay(
    keyed_frame: pd.DataFrame,
    *,
    historical_odds_dir: Path | None = None,
) -> pd.DataFrame:
    """Build opening-odds overlay (offset_days >= 5, i.e. earliest available snapshot).

    Used for dual-baseline evaluation: closing odds are the main model feature,
    opening odds provide a realistic benchmark of what was available pre-fight.
    """
    empty_cols = ["fight_key", "a_opening_odds__historical_overlay", "b_opening_odds__historical_overlay"]
    if keyed_frame.empty:
        return pd.DataFrame(columns=empty_cols)

    historical_df = _load_historical_moneyline_rows(historical_odds_dir)
    if historical_df.empty:
        return pd.DataFrame(columns=empty_cols)

    required_cols_set = {"event_date", "fighter_a", "fighter_b", "a_decimal_odds", "b_decimal_odds"}
    if not required_cols_set.issubset(historical_df.columns):
        return pd.DataFrame(columns=empty_cols)

    historical = historical_df.copy()
    historical = historical.dropna(subset=["event_date", "fighter_a", "fighter_b"])
    if historical.empty:
        return pd.DataFrame(columns=empty_cols)

    historical["event_date"] = pd.to_datetime(historical["event_date"], errors="coerce", format="mixed")
    historical = historical.dropna(subset=["event_date"])
    if historical.empty:
        return pd.DataFrame(columns=empty_cols)

    # Filter to only opening snapshots (offset_days >= 5)
    if "offset_days" in historical.columns:
        historical = historical[historical["offset_days"] >= 5].copy()
    if historical.empty:
        return pd.DataFrame(columns=empty_cols)

    historical["fight_key"] = _fight_key_series(historical)
    historical = historical.dropna(subset=["fight_key"])
    if historical.empty:
        return pd.DataFrame(columns=empty_cols)

    # Sort by offset_days DESCENDING (highest = earliest snapshot = opening)
    sort_cols = [c for c in ["offset_days", "query_date", "num_bookmakers"] if c in historical.columns]
    ascending = []
    for c in sort_cols:
        if c == "num_bookmakers":
            ascending.append(False)
        else:
            ascending.append(False)  # Descending for offset_days too — want earliest
    if sort_cols:
        historical = historical.sort_values(sort_cols, ascending=ascending)
    historical = historical.drop_duplicates(subset=["fight_key"], keep="first")

    overlay = historical[
        ["fight_key", "fighter_a", "fighter_b", "a_decimal_odds", "b_decimal_odds"]
    ].copy()
    overlay["a_decimal_odds"] = overlay["a_decimal_odds"].apply(_decimal_to_american_odds)
    overlay["b_decimal_odds"] = overlay["b_decimal_odds"].apply(_decimal_to_american_odds)
    overlay = overlay.rename(
        columns={
            "fighter_a": "__hist_fighter_a",
            "fighter_b": "__hist_fighter_b",
            "a_decimal_odds": "a_opening_odds__historical_overlay",
            "b_decimal_odds": "b_opening_odds__historical_overlay",
        }
    )

    pulled_lookup = keyed_frame[["fight_key", "fighter_a", "fighter_b"]].copy()
    overlay = pulled_lookup.merge(overlay, on="fight_key", how="left")
    reverse_mask = (
        overlay["__hist_fighter_a"].notna()
        & (overlay["fighter_a"].apply(_normalize_fighter_name) == overlay["__hist_fighter_b"].apply(_normalize_fighter_name))
        & (overlay["fighter_b"].apply(_normalize_fighter_name) == overlay["__hist_fighter_a"].apply(_normalize_fighter_name))
    )
    if reverse_mask.any():
        left_values = overlay.loc[reverse_mask, "a_opening_odds__historical_overlay"].copy()
        right_values = overlay.loc[reverse_mask, "b_opening_odds__historical_overlay"].copy()
        overlay.loc[reverse_mask, "a_opening_odds__historical_overlay"] = right_values
        overlay.loc[reverse_mask, "b_opening_odds__historical_overlay"] = left_values

    return overlay[["fight_key", "a_opening_odds__historical_overlay", "b_opening_odds__historical_overlay"]]


def _legacy_historical_rankings_overlay(
    keyed_frame: pd.DataFrame,
    *,
    rankings_history_path: Path | None = None,
) -> pd.DataFrame:
    """Build a fight-keyed rankings overlay from a local historical rankings archive."""
    empty = pd.DataFrame(
        columns=["fight_key", "a_wc_rank__historical_overlay", "b_wc_rank__historical_overlay", "a_pfp_rank__historical_overlay", "b_pfp_rank__historical_overlay"]
    )
    if keyed_frame.empty:
        return empty

    history_path = (
        Path(rankings_history_path)
        if rankings_history_path is not None
        else RAW_DATA_DIR / "kaggle_rankings" / "rankings_history_extended.csv"
    )
    if not history_path.exists():
        return empty

    history = pd.read_csv(history_path)
    required_cols = {"date", "fighter", "weightclass", "rank"}
    if history.empty or not required_cols.issubset(history.columns):
        return empty

    history = history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce", format="mixed").dt.normalize()
    history["fighter_norm"] = history["fighter"].apply(_normalize_name)
    history["weightclass_norm"] = history["weightclass"].astype(str).str.strip()
    history["wc_key"] = history["weightclass_norm"].apply(_canonical_wc)
    history["is_pfp"] = history["weightclass_norm"].str.contains("pound", case=False, na=False)
    history["rank"] = pd.to_numeric(history["rank"], errors="coerce")
    history = history.dropna(subset=["date", "fighter_norm", "rank"])
    if history.empty:
        return empty

    required_frame_cols = {"fight_key", "event_date", "fighter_a", "fighter_b", "weight_class"}
    if not required_frame_cols.issubset(keyed_frame.columns):
        return empty

    base = keyed_frame[list(required_frame_cols)].copy()
    base["event_date"] = pd.to_datetime(base["event_date"], errors="coerce", format="mixed").dt.normalize()
    base = base.dropna(subset=["fight_key", "event_date"])
    if base.empty:
        return empty

    snapshot_dates = pd.DataFrame({"rank_date": sorted(history["date"].dropna().unique())})
    event_dates = base[["event_date"]].drop_duplicates().sort_values("event_date")
    dated = pd.merge_asof(
        event_dates,
        snapshot_dates,
        left_on="event_date",
        right_on="rank_date",
        direction="backward",
    )
    base = base.merge(dated, on="event_date", how="left")
    base["wc_key"] = base["weight_class"].apply(_canonical_wc)

    fighter_rows: list[pd.DataFrame] = []
    for side in ("a", "b"):
        fighter_rows.append(
            pd.DataFrame(
                {
                    "fight_key": base["fight_key"],
                    "rank_date": base["rank_date"],
                    "side": side,
                    "fighter_norm": base[f"fighter_{side}"].apply(_normalize_name),
                    "wc_key": base["wc_key"],
                }
            )
        )
    fighters = pd.concat(fighter_rows, ignore_index=True)

    wc_history = history[~history["is_pfp"]][["date", "fighter_norm", "wc_key", "rank"]].rename(
        columns={"date": "rank_date", "rank": "wc_rank"}
    )
    pfp_history = history[history["is_pfp"]][["date", "fighter_norm", "rank"]].rename(
        columns={"date": "rank_date", "rank": "pfp_rank"}
    )

    fighters = fighters.merge(
        wc_history,
        on=["rank_date", "fighter_norm", "wc_key"],
        how="left",
    )
    fighters = fighters.merge(
        pfp_history,
        on=["rank_date", "fighter_norm"],
        how="left",
    )

    pivot = fighters.pivot_table(
        index="fight_key",
        columns="side",
        values=["wc_rank", "pfp_rank"],
        aggfunc="first",
    )
    if pivot.empty:
        return empty

    pivot.columns = [f"{side}_{field}__historical_overlay" for field, side in pivot.columns]
    return pivot.reset_index()



def _observation_cutoff(row):
    """Use a supplied exact fight time; date-only history is midnight, never EOD."""
    for key in ("commence_time", "fight_start_time", "event_date"):
        value = row.get(key)
        if value is not None and not pd.isna(value) and str(value).strip():
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                return None
            if parsed.tzinfo is None:
                if key != "event_date" or parsed != parsed.normalize():
                    return None
                parsed = parsed.tz_localize("UTC")
            return parsed.tz_convert("UTC").isoformat()
    return None


def _historical_rankings_overlay(keyed_frame, *, rankings_history_path=None):
    from src.data.selected_source_inputs import selected_path
    from src.data.rankings_observation_history import (
        HISTORY_SUBDIR, read_finalized_rankings, select_finalized_rankings,
    )
    path = Path(rankings_history_path) if rankings_history_path is not None else RAW_DATA_DIR / "kaggle_rankings" / "rankings_history_extended.csv"
    root = selected_path("rankings_observations", path.parent / HISTORY_SUBDIR)
    if not root.exists():
        return _legacy_historical_rankings_overlay(keyed_frame, rankings_history_path=path)
    rows = read_finalized_rankings(root)
    result = []
    for row in keyed_frame.to_dict("records"):
        selected = {"fight_key": row["fight_key"], "rankings_observation_policy": "reviewed_identity_receipt_history"}
        for side in ("a", "b"):
            cutoff = _observation_cutoff(row)
            values = select_finalized_rankings(rows,
                fighter_id=normalize_ufcstats_id(row.get(f"fighter_{side}_id")) or "",
                weight_class=row.get("weight_class"), as_of=cutoff) if cutoff is not None else {}
            for key in ("wc_rank", "pfp_rank"):
                selected[f"{side}_{key}__historical_overlay"] = values.get(key + "_feat")
        result.append(selected)
    frame = pd.DataFrame(result)
    if not frame.empty and frame["fight_key"].duplicated().any():
        raise ValueError("conflicting stable identities for one rankings fight key")
    return frame


def _apply_observation_overlay(frame, overlay, *, columns, family):
    """Explicit precedence: verified history (including unknown) owns its rows.

    Legacy source values are retained as metadata; differing observed values are
    recorded as conflicts instead of silently creating a mixed-source feature.
    The metadata does not add a training feature.
    """
    if overlay.empty:
        return frame
    policy = family + "_observation_policy"
    frame = frame.merge(overlay, on="fight_key", how="left", validate="many_to_one")
    strict = frame[policy].notna() if policy in frame else pd.Series(False, index=frame.index)
    legacy_values = [{} for _ in range(len(frame))]
    conflicts = [[] for _ in range(len(frame))]
    for column in columns:
        source = column + "__historical_overlay"
        if source not in frame:
            frame[source] = np.nan
        if column in frame:
            for index, (old, new, owned) in enumerate(zip(frame[column], frame[source], strict)):
                if owned and not pd.isna(old):
                    legacy_values[index][column] = float(old)
                    if not pd.isna(new) and float(old) != float(new):
                        conflicts[index].append(column)
            frame[column] = frame[column].combine_first(frame[source])
            frame.loc[strict, column] = frame.loc[strict, source]
        else:
            frame[column] = frame[source]
        frame = frame.drop(columns=[source])
    if strict.any():
        frame[family + "_legacy_values_json"] = [json.dumps(value, sort_keys=True) for value in legacy_values]
        frame[family + "_legacy_conflicts_json"] = [json.dumps(value) for value in conflicts]
    return frame


def _legacy_historical_method_odds_overlay(
    keyed_frame: pd.DataFrame,
    *,
    method_odds_history_path: Path | None = None,
) -> pd.DataFrame:
    """Build a fight-keyed historical method-odds overlay from the local archive."""
    empty = pd.DataFrame(
        columns=[
            "fight_key",
            "a_ko_odds_prob__historical_overlay",
            "a_sub_odds_prob__historical_overlay",
            "a_dec_odds_prob__historical_overlay",
            "b_ko_odds_prob__historical_overlay",
            "b_sub_odds_prob__historical_overlay",
            "b_dec_odds_prob__historical_overlay",
        ]
    )
    if keyed_frame.empty:
        return empty

    history_path = (
        Path(method_odds_history_path)
        if method_odds_history_path is not None
        else RAW_DATA_DIR / "method_odds" / "historical_method_odds_all.csv"
    )
    if not history_path.exists():
        return empty

    history = pd.read_csv(history_path)
    required_cols = {
        "event_date",
        "fighter_a",
        "fighter_b",
        "a_ko_odds_prob",
        "a_sub_odds_prob",
        "a_dec_odds_prob",
        "b_ko_odds_prob",
        "b_sub_odds_prob",
        "b_dec_odds_prob",
    }
    if history.empty or not required_cols.issubset(history.columns):
        return empty

    history = history.copy()
    history["event_date"] = pd.to_datetime(history["event_date"], errors="coerce", format="mixed")
    history = history.dropna(subset=["event_date", "fighter_a", "fighter_b"])
    if history.empty:
        return empty

    history["fight_key"] = _fight_key_series(history)
    history = history.dropna(subset=["fight_key"]).drop_duplicates(subset=["fight_key"], keep="last")
    if history.empty:
        return empty

    overlay = history[
        [
            "fight_key",
            "fighter_a",
            "fighter_b",
            "a_ko_odds_prob",
            "a_sub_odds_prob",
            "a_dec_odds_prob",
            "b_ko_odds_prob",
            "b_sub_odds_prob",
            "b_dec_odds_prob",
        ]
    ].copy()
    overlay = overlay.rename(
        columns={
            "fighter_a": "__hist_fighter_a",
            "fighter_b": "__hist_fighter_b",
            "a_ko_odds_prob": "a_ko_odds_prob__historical_overlay",
            "a_sub_odds_prob": "a_sub_odds_prob__historical_overlay",
            "a_dec_odds_prob": "a_dec_odds_prob__historical_overlay",
            "b_ko_odds_prob": "b_ko_odds_prob__historical_overlay",
            "b_sub_odds_prob": "b_sub_odds_prob__historical_overlay",
            "b_dec_odds_prob": "b_dec_odds_prob__historical_overlay",
        }
    )

    pulled_lookup = keyed_frame[["fight_key", "fighter_a", "fighter_b"]].copy()
    overlay = pulled_lookup.merge(overlay, on="fight_key", how="left")
    reverse_mask = (
        overlay["__hist_fighter_a"].notna()
        & (overlay["fighter_a"].apply(_normalize_fighter_name) == overlay["__hist_fighter_b"].apply(_normalize_fighter_name))
        & (overlay["fighter_b"].apply(_normalize_fighter_name) == overlay["__hist_fighter_a"].apply(_normalize_fighter_name))
    )
    if reverse_mask.any():
        swappable_pairs = [
            ("a_ko_odds_prob__historical_overlay", "b_ko_odds_prob__historical_overlay"),
            ("a_sub_odds_prob__historical_overlay", "b_sub_odds_prob__historical_overlay"),
            ("a_dec_odds_prob__historical_overlay", "b_dec_odds_prob__historical_overlay"),
        ]
        for left, right in swappable_pairs:
            left_values = overlay.loc[reverse_mask, left].copy()
            right_values = overlay.loc[reverse_mask, right].copy()
            overlay.loc[reverse_mask, left] = right_values
            overlay.loc[reverse_mask, right] = left_values

    return overlay[
        [
            "fight_key",
            "a_ko_odds_prob__historical_overlay",
            "a_sub_odds_prob__historical_overlay",
            "a_dec_odds_prob__historical_overlay",
            "b_ko_odds_prob__historical_overlay",
            "b_sub_odds_prob__historical_overlay",
            "b_dec_odds_prob__historical_overlay",
        ]
    ]


def _historical_method_odds_overlay(
    keyed_frame: pd.DataFrame,
    *,
    method_odds_history_path: Path | None = None,
) -> pd.DataFrame:
    """Overlay separately verified observations without relabelling old CSV history."""
    from src.data.selected_source_inputs import selected_path
    from src.data.method_observation_history import (
        FEATURES, HISTORY_SUBDIR, read_finalized_observations, select_finalized_observation,
    )

    history_path = Path(method_odds_history_path) if method_odds_history_path is not None else RAW_DATA_DIR / "method_odds" / "historical_method_odds_all.csv"
    root = selected_path("method_observations", history_path.parent / HISTORY_SUBDIR)
    if not root.exists():
        return _legacy_historical_method_odds_overlay(keyed_frame, method_odds_history_path=history_path)
    rows = read_finalized_observations(root)
    finalized = []
    for row in keyed_frame.to_dict("records"):
        event_date = pd.to_datetime(row.get("event_date"), errors="coerce")
        values = None
        if not pd.isna(event_date) and _observation_cutoff(row) is not None:
            values = select_finalized_observation(
                rows,
                fighter_a_id=normalize_ufcstats_id(row.get("fighter_a_id")) or "",
                fighter_b_id=normalize_ufcstats_id(row.get("fighter_b_id")) or "",
                event_date=event_date.date().isoformat(),
                as_of=_observation_cutoff(row),
                event_id=row.get("event_id"),
                commence_time=row.get("commence_time"),
            )
        finalized.append({"fight_key": row["fight_key"], "method_observation_policy": "stable_identity_receipt_history",
            **{f"{name}__historical_overlay": values[name] if values is not None else np.nan for name in FEATURES}})
    verified = pd.DataFrame(finalized).drop_duplicates()
    if not verified.empty and verified["fight_key"].duplicated().any():
        raise ValueError("conflicting stable identities for one method-odds fight key")
    return verified


def resolve_auto_refresh_dataset_path(output_path: Path | None = None) -> Path:
    """Return the canonical dataset path used for auto-refresh-backed retraining."""
    return Path(output_path) if output_path is not None else AUTO_REFRESH_DATASET_PATH


def choose_best_training_dataset_path(
    base_dataset_path: Path | None,
    refreshed_dataset_path: Path | None = None,
) -> Path | None:
    """
    Prefer the newest available training dataset for auto-retrain decisions.

    The auto-refresh dataset is a strict superset when present, but a manually
    updated base dataset may be newer, so we choose by mtime.
    """
    candidates: list[Path] = []
    if base_dataset_path is not None:
        candidates.append(Path(base_dataset_path))
    refreshed_path = resolve_auto_refresh_dataset_path(refreshed_dataset_path)
    if refreshed_path.exists():
        candidates.append(refreshed_path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime if path.exists() else float("-inf"))


def merge_training_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Public wrapper used by offline source-comparison experiments."""
    return _merge_training_frames(frames)


def build_refreshed_training_dataset(
    *,
    base_dataset_path: Path | None,
    scraped_fights_path: Path | None = None,
    scraped_fighters_path: Path | None = None,
    profile_integrity_decisions_path: Path | None = None,
    prior_profile_integrity_decisions_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    """
    Merge the existing trainable UFC dataset with freshly scraped UFCStats fights.

    The output is written in the loader's normalized schema so auto-retrain can
    consume it directly without mutating the user's canonical raw CSV.
    """
    output_path = resolve_auto_refresh_dataset_path(output_path)
    scraped_fights_path = Path(scraped_fights_path) if scraped_fights_path is not None else SCRAPED_FIGHTS_PATH
    scraped_fighters_path = (
        Path(scraped_fighters_path) if scraped_fighters_path is not None else SCRAPED_FIGHTERS_PATH
    )

    if not scraped_fights_path.exists():
        return {
            "succeeded": False,
            "dataset_path": output_path,
            "reason": f"scraped fights file not found: {scraped_fights_path}",
            "appended_rows": 0,
            "total_rows": 0,
        }

    scraped_raw = pd.read_csv(scraped_fights_path)
    if scraped_raw.empty:
        return {
            "succeeded": False,
            "dataset_path": output_path,
            "reason": f"scraped fights file is empty: {scraped_fights_path}",
            "appended_rows": 0,
            "total_rows": 0,
        }

    fighter_lookup = _load_scraped_fighter_lookup(
        scraped_fighters_path,
        profile_integrity_decisions_path=profile_integrity_decisions_path,
        prior_profile_integrity_decisions_path=prior_profile_integrity_decisions_path,
    )
    scraped_training_df = _scraped_fights_to_training_rows(
        scraped_raw,
        fighter_lookup=fighter_lookup,
        ambiguous_name_keys=_ambiguous_name_keys(scraped_fighters_path),
    )
    if scraped_training_df.empty:
        return {
            "succeeded": False,
            "dataset_path": output_path,
            "reason": "scraped fights did not produce any valid trainable rows",
            "appended_rows": 0,
            "total_rows": 0,
        }

    existing_frames: list[pd.DataFrame] = []
    if base_dataset_path is not None and Path(base_dataset_path).exists():
        existing_frames.append(load_kaggle_dataset(base_dataset_path))
    if output_path.exists():
        existing_frames.append(load_kaggle_dataset(output_path))

    existing_df = merge_training_frames(existing_frames) if existing_frames else pd.DataFrame()
    existing_keys = set(_fight_key_series(existing_df).dropna()) if not existing_df.empty else set()
    scraped_keys = set(_fight_key_series(scraped_training_df).dropna())

    combined_df = merge_training_frames([existing_df, scraped_training_df])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    newest_event_date = combined_df["event_date"].max() if "event_date" in combined_df.columns else pd.NaT
    appended_rows = len(scraped_keys - existing_keys)

    logger.info(
        "Built refreshed UFC training dataset at %s with %d rows (%d appended from UFCStats scrape)",
        output_path,
        len(combined_df),
        appended_rows,
    )

    return {
        "succeeded": True,
        "dataset_path": output_path,
        "reason": None,
        "appended_rows": appended_rows,
        "total_rows": len(combined_df),
        "newest_event_date": newest_event_date.isoformat() if pd.notna(newest_event_date) else None,
    }


def build_training_rows_from_pulled_data(
    *,
    raw_data_dir: Path | None = None,
    fight_results_path: Path | None = None,
    fight_stats_path: Path | None = None,
    scraped_fighters_path: Path | None = None,
    supplemental_profiles_path: Path | None = None,
    official_active_roster_path: Path | None = None,
    profile_integrity_decisions_path: Path | None = None,
    prior_profile_integrity_decisions_path: Path | None = None,
    event_weight_evidence_lookup: Mapping[
        tuple[str, str], Mapping[str, object]
    ]
    | None = None,
    pierce_fight_identities_path: Path | None = None,
    legacy_df: pd.DataFrame | None = None,
    event_dates_cache_path: Path | None = None,
    event_date_lookup: dict[str, object] | None = None,
    event_metadata_lookup: dict[str, dict[str, object]] | None = None,
    force_refresh_event_dates: bool = False,
) -> pd.DataFrame:
    """
    Convert the repo's pulled UFCStats results/stats CSVs into the normalized training schema.

    This is useful for source experiments where we want to compare the legacy
    `ufc-master.csv` dataset against newer locally pulled UFCStats rows without
    requiring a full network scrape of every fight page.
    """
    raw_data_dir = Path(raw_data_dir) if raw_data_dir is not None else None
    fight_results_path = _resolve_raw_artifact_path(
        fight_results_path,
        raw_data_dir=raw_data_dir,
        relative_path="ufc-fight-results.csv",
        canonical_path=PULLED_FIGHT_RESULTS_PATH,
    )
    fight_stats_path = _resolve_raw_artifact_path(
        fight_stats_path,
        raw_data_dir=raw_data_dir,
        relative_path="ufc-fight-stats.csv",
        canonical_path=PULLED_FIGHT_STATS_PATH,
    )
    scraped_fighters_path = _resolve_raw_artifact_path(
        scraped_fighters_path,
        raw_data_dir=raw_data_dir,
        relative_path="ufc_fighters_scraped.csv",
        canonical_path=SCRAPED_FIGHTERS_PATH,
    )
    supplemental_profiles_path = _resolve_raw_artifact_path(
        supplemental_profiles_path,
        raw_data_dir=raw_data_dir,
        relative_path="ufc_fighters_profile_supplement.csv",
        canonical_path=PROFILE_SUPPLEMENT_PATH,
    )
    official_active_roster_path = _resolve_raw_artifact_path(
        official_active_roster_path,
        raw_data_dir=raw_data_dir,
        relative_path="ufc_active_roster_official.csv",
        canonical_path=OFFICIAL_ACTIVE_ROSTER_PATH,
    )
    pierce_fight_identities_path = _resolve_raw_artifact_path(
        pierce_fight_identities_path,
        raw_data_dir=raw_data_dir,
        relative_path="github_PierceHampton.csv",
        canonical_path=PIERCE_FIGHT_IDENTITIES_PATH,
    )
    event_dates_cache_path = _resolve_raw_artifact_path(
        event_dates_cache_path,
        raw_data_dir=raw_data_dir,
        relative_path="ufc-event-dates.csv",
        canonical_path=EVENT_DATES_CACHE_PATH,
    )

    if not fight_results_path.exists():
        raise FileNotFoundError(f"pulled fight results file not found: {fight_results_path}")
    if not fight_stats_path.exists():
        raise FileNotFoundError(f"pulled fight stats file not found: {fight_stats_path}")

    results_df, stats_df = read_csv_pair_verified(
        fight_results_path,
        fight_stats_path,
        manifest_path=csv_pair_manifest_path(fight_results_path, fight_stats_path),
    )
    if results_df.empty:
        return pd.DataFrame()
    event_url_hints = _build_event_url_hints_from_results(results_df)

    if event_metadata_lookup is None and event_date_lookup is None:
        event_metadata_lookup = load_or_fetch_event_metadata_lookup(
            required_events=results_df.get("EVENT"),
            cache_path=event_dates_cache_path,
            force_refresh=force_refresh_event_dates,
            event_url_hints=event_url_hints,
        )

    if event_date_lookup is None:
        if event_metadata_lookup is None:
            event_metadata_lookup = load_or_fetch_event_metadata_lookup(
                required_events=results_df.get("EVENT"),
                cache_path=event_dates_cache_path,
                force_refresh=force_refresh_event_dates,
                event_url_hints=event_url_hints,
            )
        event_date_lookup = {
            key: metadata.get("event_date")
            for key, metadata in event_metadata_lookup.items()
        }
    elif event_metadata_lookup is None:
        event_metadata_lookup = {
            key: {"event_date": value}
            for key, value in event_date_lookup.items()
        }

    scraped_like_df = _pulled_data_to_scraped_rows(
        results_df,
        stats_df,
        event_date_lookup=event_date_lookup,
        event_metadata_lookup=event_metadata_lookup,
    )
    if scraped_like_df.empty:
        return pd.DataFrame()

    fighter_lookup = _load_scraped_fighter_lookup(
        scraped_fighters_path,
        supplemental_profiles_path=supplemental_profiles_path,
        official_active_roster_path=official_active_roster_path,
        profile_integrity_decisions_path=profile_integrity_decisions_path,
        prior_profile_integrity_decisions_path=prior_profile_integrity_decisions_path,
        legacy_df=legacy_df,
    )
    training_df = _scraped_fights_to_training_rows(
        scraped_like_df,
        fighter_lookup=fighter_lookup,
        ambiguous_name_keys=_ambiguous_name_keys(scraped_fighters_path),
        pierce_fight_identities_path=pierce_fight_identities_path,
        event_weight_evidence_lookup=event_weight_evidence_lookup,
    )
    logger.info(
        "Converted pulled UFCStats data into %d normalized training rows spanning %s to %s",
        len(training_df),
        training_df["event_date"].min().date() if not training_df.empty else None,
        training_df["event_date"].max().date() if not training_df.empty else None,
    )
    return training_df


def build_refreshed_training_dataset_from_pulled_data(
    *,
    base_dataset_path: Path | None,
    raw_data_dir: Path | None = None,
    fight_results_path: Path | None = None,
    fight_stats_path: Path | None = None,
    scraped_fighters_path: Path | None = None,
    supplemental_profiles_path: Path | None = None,
    official_active_roster_path: Path | None = None,
    profile_integrity_decisions_path: Path | None = None,
    prior_profile_integrity_decisions_path: Path | None = None,
    pierce_fight_identities_path: Path | None = None,
    legacy_df: pd.DataFrame | None = None,
    event_dates_cache_path: Path | None = None,
    event_date_lookup: dict[str, object] | None = None,
    event_metadata_lookup: dict[str, dict[str, object]] | None = None,
    output_path: Path | None = None,
    force_refresh_event_dates: bool = False,
) -> dict:
    """Build a merged training dataset using legacy rows plus the local pulled UFCStats rows."""
    output_path = resolve_auto_refresh_dataset_path(output_path)
    if legacy_df is None:
        legacy_df = load_kaggle_dataset(base_dataset_path)
    pulled_training_df = build_training_rows_from_pulled_data(
        raw_data_dir=raw_data_dir,
        fight_results_path=fight_results_path,
        fight_stats_path=fight_stats_path,
        scraped_fighters_path=scraped_fighters_path,
        supplemental_profiles_path=supplemental_profiles_path,
        official_active_roster_path=official_active_roster_path,
        profile_integrity_decisions_path=profile_integrity_decisions_path,
        prior_profile_integrity_decisions_path=prior_profile_integrity_decisions_path,
        pierce_fight_identities_path=pierce_fight_identities_path,
        legacy_df=legacy_df,
        event_dates_cache_path=event_dates_cache_path,
        event_date_lookup=event_date_lookup,
        event_metadata_lookup=event_metadata_lookup,
        force_refresh_event_dates=force_refresh_event_dates,
    )
    if pulled_training_df.empty:
        return {
            "succeeded": False,
            "dataset_path": output_path,
            "reason": "pulled fight results/stats did not produce any valid trainable rows",
            "appended_rows": 0,
            "total_rows": 0,
        }

    existing_frames: list[pd.DataFrame] = []
    if legacy_df is not None and not legacy_df.empty:
        existing_frames.append(legacy_df.copy())
    elif base_dataset_path is not None and Path(base_dataset_path).exists():
        existing_frames.append(load_kaggle_dataset(base_dataset_path))
    if output_path.exists():
        existing_frames.append(load_kaggle_dataset(output_path))

    existing_df = merge_training_frames(existing_frames) if existing_frames else pd.DataFrame()
    existing_keys = set(_fight_key_series(existing_df).dropna()) if not existing_df.empty else set()
    pulled_keys = set(_fight_key_series(pulled_training_df).dropna())
    combined_df = merge_training_frames([existing_df, pulled_training_df])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    newest_event_date = combined_df["event_date"].max() if "event_date" in combined_df.columns else pd.NaT
    appended_rows = len(pulled_keys - existing_keys)
    return {
        "succeeded": True,
        "dataset_path": output_path,
        "reason": None,
        "appended_rows": appended_rows,
        "total_rows": len(combined_df),
        "newest_event_date": newest_event_date.isoformat() if pd.notna(newest_event_date) else None,
    }


def _canonicalize_reviewed_training_identities(frame: pd.DataFrame) -> pd.DataFrame:
    """Join only exact, stable-ID-backed name variants before fight keys form."""
    canonical = frame.copy()
    for column in ("fighter_a", "fighter_b", "winner"):
        if column in canonical.columns:
            canonical[column] = canonical[column].map(
                canonicalize_reviewed_fighter_name
            )
    return canonical


def _attach_collision_ids_from_reference(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    ambiguous_name_keys: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Fill collision IDs from the exact same dated matchup in pulled data."""
    enriched = frame.copy()
    if enriched.empty:
        return enriched
    for side in ("a", "b"):
        id_col = f"fighter_{side}_id"
        if id_col not in enriched.columns:
            enriched[id_col] = None

    reference_keys = _fight_key_series(reference)
    reference_ids: dict[tuple[str, str], set[str]] = {}
    for index, row in reference.iterrows():
        fight_key = reference_keys.get(index)
        if not fight_key:
            continue
        for side in ("a", "b"):
            fighter_id = normalize_ufcstats_id(row.get(f"fighter_{side}_id"))
            name_key = _normalize_fighter_name(row.get(f"fighter_{side}"))
            if fighter_id and name_key:
                reference_ids.setdefault((fight_key, name_key), set()).add(
                    fighter_id
                )

    frame_keys = _fight_key_series(enriched)
    if ambiguous_name_keys is None:
        ambiguous_name_keys = _ambiguous_name_keys()
    for index, row in enriched.iterrows():
        fight_key = frame_keys.get(index)
        if not fight_key:
            continue
        for side in ("a", "b"):
            id_col = f"fighter_{side}_id"
            if normalize_ufcstats_id(row.get(id_col)):
                continue
            fighter_name = row.get(f"fighter_{side}")
            if not fighter_identity_is_ambiguous(
                fighter_name,
                ambiguous_name_keys=ambiguous_name_keys,
            ):
                continue
            candidates = reference_ids.get(
                (fight_key, _normalize_fighter_name(fighter_name)),
                set(),
            )
            if len(candidates) == 1:
                enriched.at[index, id_col] = next(iter(candidates))
    return enriched


def build_training_dataset_variants(
    *,
    legacy_df: pd.DataFrame | None = None,
    pulled_df: pd.DataFrame | None = None,
    raw_data_dir: Path | None = None,
    base_dataset_path: Path | None = None,
    fight_results_path: Path | None = None,
    fight_stats_path: Path | None = None,
    scraped_fighters_path: Path | None = None,
    supplemental_profiles_path: Path | None = None,
    official_active_roster_path: Path | None = None,
    profile_integrity_decisions_path: Path | None = None,
    prior_profile_integrity_decisions_path: Path | None = None,
    pierce_fight_identities_path: Path | None = None,
    event_dates_cache_path: Path | None = None,
    historical_odds_dir: Path | None = None,
    rankings_history_path: Path | None = None,
    method_odds_history_path: Path | None = None,
    event_date_lookup: dict[str, object] | None = None,
    event_metadata_lookup: dict[str, dict[str, object]] | None = None,
    force_refresh_event_dates: bool = False,
    overlap_start: str | pd.Timestamp = DEFAULT_VARIANT_OVERLAP_START,
) -> dict[str, pd.DataFrame]:
    """Build the full UFC source-variant matrix used for offline experiments."""
    raw_data_dir = Path(raw_data_dir) if raw_data_dir is not None else None
    base_dataset_path = _resolve_raw_artifact_path(
        base_dataset_path,
        raw_data_dir=raw_data_dir,
        relative_path="ufc-master.csv",
        canonical_path=RAW_DATA_DIR / "ufc-master.csv",
    )
    fight_results_path = _resolve_raw_artifact_path(
        fight_results_path,
        raw_data_dir=raw_data_dir,
        relative_path="ufc-fight-results.csv",
        canonical_path=PULLED_FIGHT_RESULTS_PATH,
    )
    fight_stats_path = _resolve_raw_artifact_path(
        fight_stats_path,
        raw_data_dir=raw_data_dir,
        relative_path="ufc-fight-stats.csv",
        canonical_path=PULLED_FIGHT_STATS_PATH,
    )
    scraped_fighters_path = _resolve_raw_artifact_path(
        scraped_fighters_path,
        raw_data_dir=raw_data_dir,
        relative_path="ufc_fighters_scraped.csv",
        canonical_path=SCRAPED_FIGHTERS_PATH,
    )
    supplemental_profiles_path = _resolve_raw_artifact_path(
        supplemental_profiles_path,
        raw_data_dir=raw_data_dir,
        relative_path="ufc_fighters_profile_supplement.csv",
        canonical_path=PROFILE_SUPPLEMENT_PATH,
    )
    official_active_roster_path = _resolve_raw_artifact_path(
        official_active_roster_path,
        raw_data_dir=raw_data_dir,
        relative_path="ufc_active_roster_official.csv",
        canonical_path=OFFICIAL_ACTIVE_ROSTER_PATH,
    )
    pierce_fight_identities_path = _resolve_raw_artifact_path(
        pierce_fight_identities_path,
        raw_data_dir=raw_data_dir,
        relative_path="github_PierceHampton.csv",
        canonical_path=PIERCE_FIGHT_IDENTITIES_PATH,
    )
    event_dates_cache_path = _resolve_raw_artifact_path(
        event_dates_cache_path,
        raw_data_dir=raw_data_dir,
        relative_path="ufc-event-dates.csv",
        canonical_path=EVENT_DATES_CACHE_PATH,
    )
    historical_odds_dir = (
        Path(historical_odds_dir)
        if historical_odds_dir is not None
        else (raw_data_dir / "historical_odds" if raw_data_dir is not None else None)
    )
    rankings_history_path = _resolve_raw_artifact_path(
        rankings_history_path,
        raw_data_dir=raw_data_dir,
        relative_path=Path("kaggle_rankings") / "rankings_history_extended.csv",
        canonical_path=RAW_DATA_DIR / "kaggle_rankings" / "rankings_history_extended.csv",
    )
    method_odds_history_path = _resolve_raw_artifact_path(
        method_odds_history_path,
        raw_data_dir=raw_data_dir,
        relative_path=Path("method_odds") / "historical_method_odds_all.csv",
        canonical_path=RAW_DATA_DIR / "method_odds" / "historical_method_odds_all.csv",
    )
    if legacy_df is None:
        legacy_df = load_kaggle_dataset(base_dataset_path)
    if pulled_df is None:
        pulled_df = build_training_rows_from_pulled_data(
            raw_data_dir=raw_data_dir,
            fight_results_path=fight_results_path,
            fight_stats_path=fight_stats_path,
            scraped_fighters_path=scraped_fighters_path,
            supplemental_profiles_path=supplemental_profiles_path,
            official_active_roster_path=official_active_roster_path,
            profile_integrity_decisions_path=profile_integrity_decisions_path,
            prior_profile_integrity_decisions_path=prior_profile_integrity_decisions_path,
            pierce_fight_identities_path=pierce_fight_identities_path,
            legacy_df=legacy_df,
            event_dates_cache_path=event_dates_cache_path,
            event_date_lookup=event_date_lookup,
            event_metadata_lookup=event_metadata_lookup,
            force_refresh_event_dates=force_refresh_event_dates,
        )

    pulled_df = _ensure_unique_fight_rows(
        _canonicalize_reviewed_training_identities(pulled_df)
    )
    legacy_df = _ensure_unique_fight_rows(
        _attach_collision_ids_from_reference(
            _canonicalize_reviewed_training_identities(legacy_df),
            pulled_df,
            ambiguous_name_keys=_ambiguous_name_keys(scraped_fighters_path),
        )
    )
    overlap_start_ts = pd.Timestamp(overlap_start)
    legacy_max_date = pd.to_datetime(legacy_df["event_date"], errors="coerce", format="mixed").max()

    return {
        "legacy_only": legacy_df.copy(),
        "append_only_2026": build_append_only_training_dataset(
            legacy_df,
            pulled_df,
            append_after=legacy_max_date,
        ),
        "pulled_all": pulled_df.copy(),
        "pulled_all_plus_legacy_market": build_pulled_all_plus_legacy_market_training_dataset(
            legacy_df,
            pulled_df,
            historical_odds_dir=historical_odds_dir,
            rankings_history_path=rankings_history_path,
            method_odds_history_path=method_odds_history_path,
        ),
        "hybrid_legacy_first": merge_training_frames([legacy_df, pulled_df]),
        "hybrid_pulled_2022plus": build_hybrid_pulled_2022plus_training_dataset(
            legacy_df,
            pulled_df,
            overlap_start=overlap_start_ts,
        ),
        "best_of_both_field_level": build_best_of_both_field_level_training_dataset(
            legacy_df,
            pulled_df,
            overlap_start=overlap_start_ts,
        ),
        "best_of_both_full_history": build_best_of_both_full_history_training_dataset(
            legacy_df,
            pulled_df,
        ),
    }


def build_append_only_training_dataset(
    legacy_df: pd.DataFrame,
    pulled_df: pd.DataFrame,
    *,
    append_after: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Append only strictly new pulled fights after the legacy cutoff."""
    if append_after is None:
        append_after = pd.to_datetime(legacy_df["event_date"], errors="coerce", format="mixed").max()
    append_after_ts = pd.Timestamp(append_after)
    pulled_new = pulled_df[pd.to_datetime(pulled_df["event_date"], errors="coerce", format="mixed") > append_after_ts]
    return merge_training_frames([legacy_df, pulled_new])


def build_pulled_all_plus_legacy_market_training_dataset(
    legacy_df: pd.DataFrame,
    pulled_df: pd.DataFrame,
    *,
    historical_odds_dir: Path | None = None,
    rankings_history_path: Path | None = None,
    method_odds_history_path: Path | None = None,
) -> pd.DataFrame:
    """
    Keep the pulled_all row universe and overlay only legacy market fields.

    This gives V4 expansion experiments access to moneyline, rankings, and
    method-odds raw columns without switching core fight/result/stat sourcing
    away from the pulled dataset.
    """
    source_row = "__pulled_market_source_row"
    if source_row in pulled_df.columns:
        raise ValueError("Reserved pulled market row identity already exists")
    pulled_rows = pulled_df.copy()
    pulled_rows[source_row] = np.arange(len(pulled_rows))
    pulled_keyed = _keyed_training_frame(pulled_rows)
    if pulled_keyed.empty:
        return pulled_keyed.drop(columns=["fight_key", source_row], errors="ignore")

    # A dated matchup need not identify one bout: Sakuraba and Silveira had
    # two distinct fights at Ultimate Japan on 1997-12-21. Keep both source
    # rows. Overlay helpers return one row per requested lookup, so passing
    # repeated keys and merging those rows back would multiply the history.
    source_order = sorted(pulled_keyed[source_row].tolist())
    repeated_keys = set(pulled_keyed.loc[
        pulled_keyed["fight_key"].duplicated(keep=False), "fight_key"
    ])
    market_lookup = pulled_keyed.drop_duplicates(subset="fight_key", keep="first")

    def require_unambiguous_overlay(overlay, *, family):
        if overlay.empty or not repeated_keys:
            return
        value_columns = [column for column in overlay.columns
                         if column.endswith(("__historical_overlay", "__legacy_overlay"))]
        if not value_columns:
            return
        ambiguous = overlay["fight_key"].isin(repeated_keys) & overlay[value_columns].notna().any(axis=1)
        if ambiguous.any():
            keys = sorted(set(overlay.loc[ambiguous, "fight_key"]))
            raise ValueError(
                f"External {family} observation cannot distinguish repeated dated bouts: {keys}"
            )

    historical_overlay = _historical_moneyline_overlay(
        market_lookup,
        historical_odds_dir=historical_odds_dir,
    )
    require_unambiguous_overlay(historical_overlay, family="moneyline")
    if not historical_overlay.empty:
        pulled_keyed = pulled_keyed.merge(historical_overlay, on="fight_key", how="left", validate="many_to_one")
        for column in ("a_odds", "b_odds"):
            overlay_column = f"{column}__historical_overlay"
            if column in pulled_keyed.columns:
                pulled_keyed[column] = pulled_keyed[column].combine_first(pulled_keyed[overlay_column])
            else:
                pulled_keyed[column] = pulled_keyed[overlay_column]
        pulled_keyed = pulled_keyed.drop(
            columns=["a_odds__historical_overlay", "b_odds__historical_overlay"],
            errors="ignore",
        )

    # Opening odds overlay (for dual-baseline evaluation)
    opening_overlay = _historical_opening_moneyline_overlay(
        market_lookup,
        historical_odds_dir=historical_odds_dir,
    )
    require_unambiguous_overlay(opening_overlay, family="opening moneyline")
    if not opening_overlay.empty:
        pulled_keyed = pulled_keyed.merge(opening_overlay, on="fight_key", how="left", validate="many_to_one")
        # Keep opening odds as separate columns (not merged into a_odds/b_odds)
        # so build_features can produce a_opening_implied_prob / b_opening_implied_prob
        for col in ("a_opening_odds__historical_overlay", "b_opening_odds__historical_overlay"):
            target = col.replace("__historical_overlay", "")
            pulled_keyed[target] = pulled_keyed[col]
        pulled_keyed = pulled_keyed.drop(
            columns=["a_opening_odds__historical_overlay", "b_opening_odds__historical_overlay"],
            errors="ignore",
        )

    rankings_overlay = _historical_rankings_overlay(
        market_lookup,
        rankings_history_path=rankings_history_path,
    )
    require_unambiguous_overlay(rankings_overlay, family="rankings")
    pulled_keyed = _apply_observation_overlay(pulled_keyed, rankings_overlay,
        columns=("a_wc_rank", "b_wc_rank", "a_pfp_rank", "b_pfp_rank"), family="rankings")

    method_odds_overlay = _historical_method_odds_overlay(
        market_lookup,
        method_odds_history_path=method_odds_history_path,
    )
    require_unambiguous_overlay(method_odds_overlay, family="method odds")
    pulled_keyed = _apply_observation_overlay(pulled_keyed, method_odds_overlay,
        columns=tuple(f"{side}_{method}_odds_prob" for side in ("a", "b") for method in ("ko", "sub", "dec")),
        family="method")

    legacy_keyed = _keyed_training_frame(legacy_df)
    overlay_fields = sorted(
        column for column in _PULLED_ALL_LEGACY_MARKET_OVERLAY_FIELDS if column in legacy_keyed.columns
    )
    legacy_overlay = legacy_keyed[
        ["fight_key", "fighter_a", "fighter_b", *overlay_fields]
    ].copy()
    legacy_overlay = legacy_overlay.rename(
        columns={
            "fighter_a": "__legacy_fighter_a",
            "fighter_b": "__legacy_fighter_b",
            **{column: f"{column}__legacy_overlay" for column in overlay_fields},
        }
    )

    require_unambiguous_overlay(legacy_overlay, family="legacy market")
    legacy_overlay = legacy_overlay.dropna(subset=["fight_key"]).drop_duplicates(
        subset="fight_key",
        keep="first",
    )
    merged = pulled_keyed.merge(legacy_overlay, on="fight_key", how="left", validate="many_to_one")
    direct_mask = (
        merged["__legacy_fighter_a"].notna()
        & (
            merged["fighter_a"].apply(_normalize_fighter_name)
            == merged["__legacy_fighter_a"].apply(_normalize_fighter_name)
        )
        & (
            merged["fighter_b"].apply(_normalize_fighter_name)
            == merged["__legacy_fighter_b"].apply(_normalize_fighter_name)
        )
    )
    reverse_mask = (
        merged["__legacy_fighter_a"].notna()
        & (
            merged["fighter_a"].apply(_normalize_fighter_name)
            == merged["__legacy_fighter_b"].apply(_normalize_fighter_name)
        )
        & (
            merged["fighter_b"].apply(_normalize_fighter_name)
            == merged["__legacy_fighter_a"].apply(_normalize_fighter_name)
        )
    )
    invalid_mask = merged["__legacy_fighter_a"].notna() & ~(
        direct_mask | reverse_mask
    )
    for column in overlay_fields:
        overlay_column = f"{column}__legacy_overlay"
        merged.loc[invalid_mask, overlay_column] = np.nan
    for root in (
        "odds",
        "wc_rank",
        "pfp_rank",
        "ko_odds",
        "sub_odds",
        "dec_odds",
    ):
        left = f"a_{root}__legacy_overlay"
        right = f"b_{root}__legacy_overlay"
        if left not in merged.columns or right not in merged.columns:
            continue
        left_values = merged.loc[reverse_mask, left].copy()
        right_values = merged.loc[reverse_mask, right].copy()
        merged.loc[reverse_mask, left] = right_values
        merged.loc[reverse_mask, right] = left_values
    for column in overlay_fields:
        overlay_column = f"{column}__legacy_overlay"
        family = "rankings" if column.endswith(("wc_rank", "pfp_rank")) else (
            "method" if column.endswith(("ko_odds", "sub_odds", "dec_odds", "ko_odds_prob", "sub_odds_prob", "dec_odds_prob")) else None)
        policy = str(family) + "_observation_policy"
        strict = merged[policy].notna() if policy in merged else pd.Series(False, index=merged.index)
        if strict.any():
            value_column = family + "_legacy_values_json"
            conflict_column = family + "_legacy_conflicts_json"
            for index in merged.index[strict]:
                original = merged.at[index, overlay_column]
                if pd.isna(original):
                    continue
                values = json.loads(merged.at[index, value_column])
                values[column + "__legacy_dataset"] = float(original)
                merged.at[index, value_column] = json.dumps(values, sort_keys=True)
                current = merged.at[index, column] if column in merged else np.nan
                if not pd.isna(current) and float(current) != float(original):
                    conflicts = json.loads(merged.at[index, conflict_column])
                    conflicts.append(column + "__legacy_dataset")
                    merged.at[index, conflict_column] = json.dumps(conflicts)
        eligible = merged[overlay_column].where(~strict)
        if column in merged.columns:
            merged[column] = merged[column].combine_first(eligible)
        else:
            merged[column] = eligible

    drop_columns = [
        "__legacy_fighter_a",
        "__legacy_fighter_b",
        *[f"{column}__legacy_overlay" for column in overlay_fields],
    ]
    if sorted(merged[source_row].tolist()) != source_order:
        raise ValueError("Market overlays changed the pulled bout row universe")
    return merged.sort_values(source_row, kind="stable").drop(
        columns=["fight_key", source_row, *drop_columns], errors="ignore"
    ).reset_index(drop=True)


def build_hybrid_pulled_2022plus_training_dataset(
    legacy_df: pd.DataFrame,
    pulled_df: pd.DataFrame,
    *,
    overlap_start: str | pd.Timestamp = DEFAULT_VARIANT_OVERLAP_START,
) -> pd.DataFrame:
    """Keep legacy history before the overlap window, but use pulled rows from 2022 onward."""
    overlap_start_ts = pd.Timestamp(overlap_start)
    legacy_before = legacy_df[pd.to_datetime(legacy_df["event_date"], errors="coerce", format="mixed") < overlap_start_ts]
    pulled_from_cutoff = pulled_df[pd.to_datetime(pulled_df["event_date"], errors="coerce", format="mixed") >= overlap_start_ts]
    return merge_training_frames([legacy_before, pulled_from_cutoff])


def build_best_of_both_field_level_training_dataset(
    legacy_df: pd.DataFrame,
    pulled_df: pd.DataFrame,
    *,
    overlap_start: str | pd.Timestamp = DEFAULT_VARIANT_OVERLAP_START,
) -> pd.DataFrame:
    """
    Merge overlap fights field-by-field.

    Legacy remains the source of record for richer fighter metadata, rankings,
    and odds. Pulled UFCStats rows override fight-level result/stat fields where
    the official source is cleaner or the legacy source is missing data.
    """
    overlap_start_ts = pd.Timestamp(overlap_start)
    legacy_keyed = _keyed_training_frame(legacy_df)
    pulled_keyed = _keyed_training_frame(pulled_df)

    legacy_before = legacy_keyed[legacy_keyed["event_date"] < overlap_start_ts].drop(columns="fight_key", errors="ignore")
    legacy_after = legacy_keyed[legacy_keyed["event_date"] >= overlap_start_ts]
    pulled_after = pulled_keyed[pulled_keyed["event_date"] >= overlap_start_ts]

    legacy_map = {
        row["fight_key"]: row
        for _, row in legacy_after.dropna(subset=["fight_key"]).iterrows()
    }
    pulled_map = {
        row["fight_key"]: row
        for _, row in pulled_after.dropna(subset=["fight_key"]).iterrows()
    }

    merged_rows: list[dict] = []
    for fight_key in sorted(set(legacy_map) | set(pulled_map)):
        legacy_row = legacy_map.get(fight_key)
        pulled_row = pulled_map.get(fight_key)
        if legacy_row is not None and pulled_row is not None:
            merged_rows.append(_merge_best_of_both_overlap_row(legacy_row, pulled_row))
        elif legacy_row is not None:
            merged_rows.append(_row_to_training_dict(legacy_row))
        elif pulled_row is not None:
            merged_rows.append(_row_to_training_dict(pulled_row))

    return merge_training_frames([legacy_before, pd.DataFrame(merged_rows)])


def build_best_of_both_full_history_training_dataset(
    legacy_df: pd.DataFrame,
    pulled_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the field-level best-of-both merge across the full legacy history.

    This preserves the richer legacy metadata/odds fields, but backfills
    official UFCStats fight-result and per-fight stat fields for the actual
    pre-2022 training window instead of only the 2022+ overlap period.
    """
    legacy_min_date = pd.to_datetime(
        legacy_df["event_date"],
        errors="coerce",
        format="mixed",
    ).min()
    if pd.isna(legacy_min_date):
        return merge_training_frames([legacy_df, pulled_df])
    return build_best_of_both_field_level_training_dataset(
        legacy_df,
        pulled_df,
        overlap_start=legacy_min_date,
    )


def audit_legacy_pulled_overlap(
    legacy_df: pd.DataFrame,
    pulled_df: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp = DEFAULT_VARIANT_OVERLAP_START,
    end_date: str | pd.Timestamp = DEFAULT_VARIANT_OVERLAP_END,
) -> dict[str, object]:
    """Audit overlapping 2022-2024 fights to drive field-level merge rules."""
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    legacy_keyed = _keyed_training_frame(legacy_df)
    pulled_keyed = _keyed_training_frame(pulled_df)

    legacy_window = _subset_by_date_range(legacy_keyed, start_ts, end_ts)
    pulled_window = _subset_by_date_range(pulled_keyed, start_ts, end_ts)
    overlap_keys = sorted(set(legacy_window["fight_key"].dropna()) & set(pulled_window["fight_key"].dropna()))

    if not overlap_keys:
        empty = pd.DataFrame()
        return {
            "fight_level": empty,
            "field_summary": empty,
            "merge_rules": {"legacy": sorted(_LEGACY_PREFERRED_FIELDS), "pulled": sorted(_PULLED_PREFERRED_FIELDS)},
            "summary": {
                "start_date": start_ts.date().isoformat(),
                "end_date": end_ts.date().isoformat(),
                "overlap_fights": 0,
            },
        }

    legacy_overlap = legacy_window[legacy_window["fight_key"].isin(overlap_keys)].set_index("fight_key", drop=False)
    pulled_overlap = pulled_window[pulled_window["fight_key"].isin(overlap_keys)].set_index("fight_key", drop=False)

    audited_fields = sorted(
        {
            "winner",
            "method",
            "finish_round",
            "finish_time",
            "event_name",
            "weight_class",
            "title_bout",
            "num_rounds",
            "a_height",
            "a_reach",
            "a_weight",
            "a_age",
            "a_stance",
            "b_height",
            "b_reach",
            "b_weight",
            "b_age",
            "b_stance",
            "a_odds",
            "b_odds",
            "a_expected_value",
            "b_expected_value",
            *[f"a_{field}" for field in _PULLED_COUNT_STAT_FIELDS + _PULLED_RATE_STAT_FIELDS],
            *[f"b_{field}" for field in _PULLED_COUNT_STAT_FIELDS + _PULLED_RATE_STAT_FIELDS],
        }
    )

    fight_rows: list[dict] = []
    field_rows: list[dict] = []

    for field in audited_fields:
        legacy_series = legacy_overlap[field] if field in legacy_overlap.columns else pd.Series(index=legacy_overlap.index, dtype="object")
        pulled_series = pulled_overlap[field] if field in pulled_overlap.columns else pd.Series(index=pulled_overlap.index, dtype="object")
        field_rows.append(_field_overlap_summary(field, legacy_series, pulled_series))

    field_summary_df = pd.DataFrame(field_rows).sort_values(["category", "field"]).reset_index(drop=True)

    for fight_key in overlap_keys:
        legacy_row = legacy_overlap.loc[fight_key]
        pulled_row = pulled_overlap.loc[fight_key]
        fight_rows.append(
            {
                "fight_key": fight_key,
                "event_date": pd.to_datetime(legacy_row.get("event_date"), errors="coerce", format="mixed"),
                "fighter_a": legacy_row.get("fighter_a") or pulled_row.get("fighter_a"),
                "fighter_b": legacy_row.get("fighter_b") or pulled_row.get("fighter_b"),
                "winner_match": _values_match("winner", legacy_row.get("winner"), pulled_row.get("winner")),
                "method_match": _values_match("method", legacy_row.get("method"), pulled_row.get("method")),
                "finish_round_match": _values_match("finish_round", legacy_row.get("finish_round"), pulled_row.get("finish_round")),
                "finish_time_match": _values_match("finish_time", legacy_row.get("finish_time"), pulled_row.get("finish_time")),
                "weight_class_match": _values_match("weight_class", legacy_row.get("weight_class"), pulled_row.get("weight_class")),
                "legacy_missing_a_fight_stats": int(sum(pd.isna(legacy_row.get(f"a_{field}")) for field in _PULLED_COUNT_STAT_FIELDS)),
                "pulled_missing_a_fight_stats": int(sum(pd.isna(pulled_row.get(f"a_{field}")) for field in _PULLED_COUNT_STAT_FIELDS)),
                "legacy_missing_b_fight_stats": int(sum(pd.isna(legacy_row.get(f"b_{field}")) for field in _PULLED_COUNT_STAT_FIELDS)),
                "pulled_missing_b_fight_stats": int(sum(pd.isna(pulled_row.get(f"b_{field}")) for field in _PULLED_COUNT_STAT_FIELDS)),
            }
        )

    fight_level_df = pd.DataFrame(fight_rows).sort_values(["event_date", "fighter_a", "fighter_b"]).reset_index(drop=True)
    merge_rules = {
        "legacy": sorted(_LEGACY_PREFERRED_FIELDS),
        "pulled": sorted(_PULLED_PREFERRED_FIELDS),
    }
    return {
        "fight_level": fight_level_df,
        "field_summary": field_summary_df,
        "merge_rules": merge_rules,
        "summary": {
            "start_date": start_ts.date().isoformat(),
            "end_date": end_ts.date().isoformat(),
            "overlap_fights": int(len(overlap_keys)),
            "legacy_window_fights": int(len(legacy_window)),
            "pulled_window_fights": int(len(pulled_window)),
        },
    }


def _merge_training_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Union multiple training frames while preserving richer existing rows."""
    valid_frames = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame()

    combined = pd.concat(valid_frames, ignore_index=True, sort=False)
    combined["event_date"] = pd.to_datetime(combined.get("event_date"), errors="coerce", format="mixed")
    combined["__fight_key"] = _fight_key_series(combined)
    combined = combined.dropna(subset=["event_date", "fighter_a", "fighter_b"])
    combined = combined.drop_duplicates(subset="__fight_key", keep="first")
    combined = combined.sort_values(["event_date", "fighter_a", "fighter_b"]).drop(columns="__fight_key")
    return combined.reset_index(drop=True)


def _ensure_unique_fight_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    keyed = _keyed_training_frame(df)
    keyed = keyed.drop_duplicates(subset="fight_key", keep="first")
    return keyed.drop(columns="fight_key", errors="ignore").reset_index(drop=True)


def _keyed_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    keyed = df.copy()
    if "event_date" in keyed.columns:
        keyed["event_date"] = pd.to_datetime(keyed["event_date"], errors="coerce", format="mixed")
    keyed["fight_key"] = _fight_key_series(keyed)
    keyed = keyed.dropna(subset=["event_date", "fighter_a", "fighter_b"])
    keyed = keyed.sort_values(["event_date", "fighter_a", "fighter_b"]).reset_index(drop=True)
    return keyed


def _subset_by_date_range(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    mask = (pd.to_datetime(df["event_date"], errors="coerce", format="mixed") >= start_ts) & (
        pd.to_datetime(df["event_date"], errors="coerce", format="mixed") < end_ts
    )
    return df.loc[mask].copy()


def _row_to_training_dict(row: pd.Series) -> dict:
    data = row.to_dict()
    data.pop("fight_key", None)
    return data


def _merge_best_of_both_overlap_row(legacy_row: pd.Series, pulled_row: pd.Series) -> dict:
    merged = _row_to_training_dict(legacy_row)
    reversed_orientation = (
        _normalize_name(legacy_row.get("fighter_a"))
        == _normalize_name(pulled_row.get("fighter_b"))
        and _normalize_name(legacy_row.get("fighter_b"))
        == _normalize_name(pulled_row.get("fighter_a"))
    )
    for field in pulled_row.index:
        if field == "fight_key":
            continue
        if reversed_orientation and field == "fighter_a_id":
            pulled_value = pulled_row.get("fighter_b_id")
        elif reversed_orientation and field == "fighter_b_id":
            pulled_value = pulled_row.get("fighter_a_id")
        else:
            pulled_value = pulled_row.get(field)
        legacy_value = legacy_row.get(field)
        if field in _LEGACY_PREFERRED_FIELDS:
            merged[field] = legacy_value if not pd.isna(legacy_value) else pulled_value
            continue
        if field in _PULLED_PREFERRED_FIELDS:
            merged[field] = pulled_value if not pd.isna(pulled_value) else legacy_value
            continue
        if pd.isna(legacy_value) and not pd.isna(pulled_value):
            merged[field] = pulled_value
        elif field not in merged:
            merged[field] = pulled_value
    return merged


def _field_overlap_summary(field: str, legacy_series: pd.Series, pulled_series: pd.Series) -> dict:
    comparable_mask = legacy_series.notna() & pulled_series.notna()
    comparable = int(comparable_mask.sum())
    equal_count = int(
        sum(
            _values_match(field, legacy_series.loc[idx], pulled_series.loc[idx])
            for idx in legacy_series.index[comparable_mask]
        )
    )
    mismatch_count = comparable - equal_count
    numeric_diff = pd.to_numeric(legacy_series, errors="coerce") - pd.to_numeric(pulled_series, errors="coerce")
    numeric_diff = numeric_diff[comparable_mask & numeric_diff.notna()]
    return {
        "field": field,
        "category": _field_category(field),
        "legacy_non_null": int(legacy_series.notna().sum()),
        "pulled_non_null": int(pulled_series.notna().sum()),
        "legacy_only_non_null": int((legacy_series.notna() & pulled_series.isna()).sum()),
        "pulled_only_non_null": int((legacy_series.isna() & pulled_series.notna()).sum()),
        "comparable_rows": comparable,
        "equal_rows": equal_count,
        "mismatch_rows": mismatch_count,
        "equal_pct_when_both": float(equal_count / comparable) if comparable else np.nan,
        "mean_abs_numeric_diff": float(numeric_diff.abs().mean()) if not numeric_diff.empty else np.nan,
        "preferred_source": _recommend_field_source(field, legacy_series, pulled_series),
    }


def _field_category(field: str) -> str:
    if field in {"winner", "method", "finish_round", "finish_time", "event_name", "weight_class", "title_bout", "num_rounds"}:
        return "fight_result"
    if field.startswith(("a_odds", "b_odds", "a_expected_value", "b_expected_value", "a_dec_odds", "b_dec_odds", "a_sub_odds", "b_sub_odds", "a_ko_odds", "b_ko_odds")):
        return "odds"
    if field.startswith(("a_height", "a_reach", "a_weight", "a_age", "a_stance", "b_height", "b_reach", "b_weight", "b_age", "b_stance")):
        return "fighter_metadata"
    if field.startswith(("a_", "b_")):
        return "fight_stats"
    return "misc"


def _recommend_field_source(field: str, legacy_series: pd.Series, pulled_series: pd.Series) -> str:
    if field in _LEGACY_PREFERRED_FIELDS:
        return "legacy"
    if field in _PULLED_PREFERRED_FIELDS:
        return "pulled"
    legacy_non_null = int(legacy_series.notna().sum())
    pulled_non_null = int(pulled_series.notna().sum())
    if pulled_non_null > legacy_non_null:
        return "pulled"
    if legacy_non_null > pulled_non_null:
        return "legacy"
    return "pulled" if _field_category(field) == "fight_stats" else "legacy"


def _values_match(field: str, legacy_value, pulled_value) -> bool:
    left = _normalize_audit_value(field, legacy_value)
    right = _normalize_audit_value(field, pulled_value)
    if left is None and right is None:
        return True
    return left == right


def _normalize_audit_value(field: str, value):
    if value is None or pd.isna(value):
        return None
    if field == "method":
        return _canonical_method(value)
    if field in {"finish_round", "num_rounds", "title_bout"}:
        numeric = _safe_float(value)
        return int(numeric) if numeric is not None and not np.isnan(numeric) else None
    if field.endswith("_share"):
        numeric = _safe_float(value)
        return round(float(numeric), 6) if numeric is not None and not np.isnan(numeric) else None
    numeric = _safe_float(value)
    if numeric is not None and not np.isnan(numeric):
        return round(float(numeric), 6)
    return _normalize_name(value)


def _canonical_method(value) -> str | None:
    text = re.sub(r"[^a-z0-9]+", " ", _normalize_name(value)).strip()
    if not text:
        return None
    if "decision" in text or text in {"u dec", "s dec", "m dec"}:
        if "split" in text or text == "s dec":
            return "decision_split"
        if "majority" in text or text == "m dec":
            return "decision_majority"
        return "decision_unanimous"
    if text in {"sub", "submission"} or "submission" in text:
        return "submission"
    if "doctor" in text:
        return "tko_doctor"
    if "ko" in text or "tko" in text:
        return "ko_tko"
    if "draw" in text:
        return "draw"
    return text


def _fight_key_series(df: pd.DataFrame) -> pd.Series:
    """Build an order-independent key for a fight row."""
    if df.empty:
        return pd.Series(dtype="object")

    def _key(row: pd.Series) -> str | None:
        event_date = pd.to_datetime(row.get("event_date"), errors="coerce", format="mixed")
        fighter_a = _normalize_name(row.get("fighter_a"))
        fighter_b = _normalize_name(row.get("fighter_b"))
        if pd.isna(event_date) or not fighter_a or not fighter_b:
            return None
        low_name, high_name = sorted([fighter_a, fighter_b])
        return f"{event_date.date().isoformat()}|{low_name}|{high_name}"

    return df.apply(_key, axis=1)


def _normalize_name(value) -> str:
    if value is None or pd.isna(value):
        return ""
    import unicodedata
    text = unicodedata.normalize("NFKD", str(value).strip())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).casefold()


def _fight_details_id(value: object) -> str | None:
    text = "" if value is None or pd.isna(value) else str(value)
    match = re.search(
        r"fight-details/([0-9a-f]{16})",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).casefold() if match else None


def _load_pierce_fight_identity_lookup(
    path: Path | None = None,
) -> dict[str, dict[str, str]]:
    """Return fight-URL participant IDs from the existing Pierce artifact."""
    path = Path(path) if path is not None else PIERCE_FIGHT_IDENTITIES_PATH
    if not Path(path).exists():
        return {}
    usecols = ["fight_url", "f_1_name", "f_1_url", "f_2_name", "f_2_url"]
    try:
        source = pd.read_csv(path, usecols=usecols)
    except (OSError, ValueError) as exc:
        logger.warning(
            "Could not load fight participant identities from %s: %s",
            path,
            exc,
        )
        return {}
    lookup: dict[str, dict[str, str]] = {}
    for row in source.itertuples(index=False):
        fight_id = _fight_details_id(row.fight_url)
        if not fight_id:
            continue
        participants: dict[str, str] = {}
        for name, url in (
            (row.f_1_name, row.f_1_url),
            (row.f_2_name, row.f_2_url),
        ):
            fighter_id = normalize_ufcstats_id(url)
            name_key = _normalize_fighter_name(name)
            if name_key and fighter_id:
                participants[name_key] = fighter_id
        if participants:
            lookup[fight_id] = participants
    return lookup


def _attach_local_fighter_ids(
    scraped_df: pd.DataFrame,
    *,
    pierce_fight_identities_path: Path | None = None,
) -> pd.DataFrame:
    """Attach nullable participant IDs from row URLs and reviewed local data."""
    frame = scraped_df.copy()
    pierce = (
        _load_pierce_fight_identity_lookup()
        if pierce_fight_identities_path is None
        else _load_pierce_fight_identity_lookup(pierce_fight_identities_path)
    )
    for side in ("a", "b"):
        id_col = f"fighter_{side}_id"
        url_candidates = (
            f"fighter_{side}_url",
            f"{side}_fighter_url",
            f"fighter_{side}_ufcstats_url",
        )
        if id_col not in frame.columns:
            frame[id_col] = None
        for index, row in frame.iterrows():
            resolved = normalize_ufcstats_id(row.get(id_col))
            if not resolved:
                for url_col in url_candidates:
                    resolved = normalize_ufcstats_id(row.get(url_col))
                    if resolved:
                        break
            fight_id = _fight_details_id(row.get("fight_url") or row.get("URL"))
            fighter_name = row.get(f"fighter_{side}")
            name_key = _normalize_fighter_name(fighter_name)
            if not resolved and fight_id:
                resolved = pierce.get(fight_id, {}).get(name_key)
            if not resolved and fight_id:
                resolved = _FIGHT_PARTICIPANT_ID_OVERRIDES.get(
                    (fight_id, name_key)
                )
            frame.at[index, id_col] = resolved
    return frame


def _scraped_fights_to_training_rows(
    scraped_df: pd.DataFrame,
    *,
    fighter_lookup: dict[str, dict] | None = None,
    ambiguous_name_keys: frozenset[str] | None = None,
    pierce_fight_identities_path: Path | None = None,
    event_weight_evidence_lookup: Mapping[
        tuple[str, str], Mapping[str, object]
    ]
    | None = None,
) -> pd.DataFrame:
    """Convert raw scrape output into the normalized training schema."""
    fighter_lookup = fighter_lookup or {}
    if pierce_fight_identities_path is None:
        scraped_df = _attach_local_fighter_ids(scraped_df)
    else:
        scraped_df = _attach_local_fighter_ids(
            scraped_df,
            pierce_fight_identities_path=pierce_fight_identities_path,
        )
    rows: list[dict] = []
    if ambiguous_name_keys is None:
        ambiguous_name_keys = _ambiguous_name_keys()

    for _, row in scraped_df.iterrows():
        event_date = pd.to_datetime(row.get("event_date"), errors="coerce", format="mixed")
        fighter_a = row.get("fighter_a")
        fighter_b = row.get("fighter_b")
        if pd.isna(event_date) or pd.isna(fighter_a) or pd.isna(fighter_b):
            continue
        fighter_a_id = normalize_ufcstats_id(row.get("fighter_a_id"))
        fighter_b_id = normalize_ufcstats_id(row.get("fighter_b_id"))
        for name, fighter_id in (
            (fighter_a, fighter_a_id),
            (fighter_b, fighter_b_id),
        ):
            if fighter_identity_key(
                name,
                fighter_id,
                ambiguous_name_keys=ambiguous_name_keys,
            ) is None:
                raise ValueError(
                    "unresolved ambiguous fighter identity for "
                    f"{name} on {event_date.date()}"
                )

        weight_class, title_bout = _parse_weight_class(row.get("weight_class"))
        finish_round = _safe_int(row.get("round"))
        finish_time = row.get("time")
        total_fight_time_secs = _round_time_to_total_seconds(finish_round, finish_time)
        fight_minutes = (total_fight_time_secs / 60.0) if total_fight_time_secs and total_fight_time_secs > 0 else np.nan
        scheduled_rounds = _parse_scheduled_rounds(row.get("time_format"))

        policy_version = row.get("result_policy_version")
        winner = row.get("winner")
        target = (
            1
            if winner == fighter_a
            else (0 if winner == fighter_b else np.nan)
        )
        if has_result_policy_marker(policy_version):
            classification = validate_marked_result(
                row.get("official_outcome"),
                row.get("method"),
                policy_version=policy_version,
                persisted_result_class=row.get("result_class"),
                persisted_binary_target_eligible=row.get(
                    "binary_target_eligible"
                ),
            )
            classified_winner = winner_for_side(
                classification.winner_side,
                fighter_a=fighter_a,
                fighter_b=fighter_b,
            )
            if _clean_text(winner) and winner != classified_winner:
                raise ValueError(
                    "marked result winner contradicts official outcome for "
                    f"{fighter_a} vs. {fighter_b} on {event_date.date()}"
                )
            winner = classified_winner
            target = (
                1 if winner == fighter_a else 0
            ) if classification.result_class == RESULT_CLASS_NORMAL else np.nan

        training_row = {
            "event_date": event_date,
            "event_name": row.get("event_title"),
            "location": row.get("location"),
            "weight_class": weight_class,
            "winner": winner,
            "method": row.get("method"),
            "official_outcome": row.get("official_outcome"),
            "result_policy_version": policy_version,
            "result_class": row.get("result_class"),
            "binary_target_eligible": row.get("binary_target_eligible"),
            "result_quarantine_reason": row.get("result_quarantine_reason"),
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "fighter_a_id": fighter_a_id,
            "fighter_b_id": fighter_b_id,
            "target": target,
            "title_bout": float(title_bout) if title_bout is not None else np.nan,
            "num_rounds": _infer_num_rounds(
                title_bout=title_bout,
                finish_round=finish_round,
                method=row.get("method"),
                scheduled_rounds=scheduled_rounds,
            ),
            "finish_round": finish_round,
            "finish_time": finish_time,
            "total_fight_time_secs": total_fight_time_secs,
            "empty_arena": infer_empty_arena(
                event_title=row.get("event_title"),
                location=row.get("location"),
            ),
        }
        training_row.update(_extract_rate_features(row, fight_minutes=fight_minutes, prefix="a_", opp_prefix="b_"))
        training_row.update(_extract_rate_features(row, fight_minutes=fight_minutes, prefix="b_", opp_prefix="a_"))
        training_row.update(
            _extract_fighter_attributes(
                fighter_a,
                event_date,
                fighter_lookup,
                prefix="a_",
                fighter_id=fighter_a_id,
                opponent_id=fighter_b_id,
                fight_url=row.get("fight_url") or row.get("URL"),
                event_weight_evidence=(
                    event_weight_evidence_lookup.get(
                        (
                            str(row.get("fight_url") or row.get("URL") or "").rstrip("/"),
                            fighter_a_id or "",
                        )
                    )
                    if event_weight_evidence_lookup is not None
                    else None
                ),
            )
        )
        training_row.update(
            _extract_fighter_attributes(
                fighter_b,
                event_date,
                fighter_lookup,
                prefix="b_",
                fighter_id=fighter_b_id,
                opponent_id=fighter_a_id,
                fight_url=row.get("fight_url") or row.get("URL"),
                event_weight_evidence=(
                    event_weight_evidence_lookup.get(
                        (
                            str(row.get("fight_url") or row.get("URL") or "").rstrip("/"),
                            fighter_b_id or "",
                        )
                    )
                    if event_weight_evidence_lookup is not None
                    else None
                ),
            )
        )
        rows.append(training_row)

    return pd.DataFrame(rows)


def _parse_weight_class(value) -> tuple[str | None, bool | None]:
    raw = "" if value is None or pd.isna(value) else str(value).strip()
    if not raw:
        return None, None
    title_bout = "title bout" in raw.casefold()
    cleaned = re.sub(r"title bout", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bbout\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
    return cleaned or raw, title_bout


def _safe_int(value) -> int | None:
    parsed = _safe_float(value)
    if parsed is None or np.isnan(parsed):
        return None
    return int(parsed)


def _round_time_to_total_seconds(round_num: int | None, finish_time) -> float | None:
    if round_num is None:
        return None
    seconds_in_round = _clock_to_seconds(finish_time)
    if seconds_in_round is None:
        return None
    return float(max(round_num - 1, 0) * _ROUND_SECONDS + seconds_in_round)


def _clock_to_seconds(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    match = re.fullmatch(r"(\d+):(\d{2})", text)
    if not match:
        return None
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    return float(minutes * 60 + seconds)


def _parse_scheduled_rounds(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    match = re.search(r"(\d+)\s*rnd", str(value), flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _infer_num_rounds(
    *,
    title_bout: bool | None,
    finish_round: int | None,
    method,
    scheduled_rounds: int | None = None,
) -> float | None:
    if scheduled_rounds is not None and scheduled_rounds > 0:
        return float(scheduled_rounds)
    method_text = "" if method is None or pd.isna(method) else str(method)
    if "decision" in method_text.casefold() and finish_round is not None:
        return float(finish_round)
    if title_bout or (finish_round is not None and finish_round > 3):
        return 5.0
    return np.nan


def _extract_rate_features(
    row: pd.Series,
    *,
    fight_minutes: float,
    prefix: str,
    opp_prefix: str,
) -> dict:
    sig_landed = _safe_float(row.get(f"{prefix}sig_str_landed"))
    sig_attempted = _safe_float(row.get(f"{prefix}sig_str_attempted"))
    td_landed = _safe_float(row.get(f"{prefix}td_landed"))
    td_attempted = _safe_float(row.get(f"{prefix}td_attempted"))
    sub_att = _safe_float(row.get(f"{prefix}sub_att"))
    rev = _safe_float(row.get(f"{prefix}rev"))
    kd = _safe_float(row.get(f"{prefix}kd"))
    ctrl_seconds = _clock_to_seconds(row.get(f"{prefix}ctrl"))
    total_str_landed = _safe_float(row.get(f"{prefix}total_str_landed"))
    total_str_attempted = _safe_float(row.get(f"{prefix}total_str_attempted"))
    head_landed = _safe_float(row.get(f"{prefix}head_landed"))
    head_attempted = _safe_float(row.get(f"{prefix}head_attempted"))
    body_landed = _safe_float(row.get(f"{prefix}body_landed"))
    body_attempted = _safe_float(row.get(f"{prefix}body_attempted"))
    leg_landed = _safe_float(row.get(f"{prefix}leg_landed"))
    leg_attempted = _safe_float(row.get(f"{prefix}leg_attempted"))
    distance_landed = _safe_float(row.get(f"{prefix}distance_landed"))
    distance_attempted = _safe_float(row.get(f"{prefix}distance_attempted"))
    clinch_landed = _safe_float(row.get(f"{prefix}clinch_landed"))
    clinch_attempted = _safe_float(row.get(f"{prefix}clinch_attempted"))
    ground_landed = _safe_float(row.get(f"{prefix}ground_landed"))
    ground_attempted = _safe_float(row.get(f"{prefix}ground_attempted"))

    opp_sig_landed = _safe_float(row.get(f"{opp_prefix}sig_str_landed"))
    opp_sig_attempted = _safe_float(row.get(f"{opp_prefix}sig_str_attempted"))
    opp_td_landed = _safe_float(row.get(f"{opp_prefix}td_landed"))
    opp_td_attempted = _safe_float(row.get(f"{opp_prefix}td_attempted"))
    fight_seconds = fight_minutes * 60.0 if not pd.isna(fight_minutes) else np.nan

    return {
        f"{prefix}slpm": _rate_per_minute(sig_landed, fight_minutes),
        f"{prefix}sapm": _rate_per_minute(opp_sig_landed, fight_minutes),
        f"{prefix}str_acc": _pct(sig_landed, sig_attempted),
        f"{prefix}str_def": _defense_pct(opp_sig_landed, opp_sig_attempted),
        f"{prefix}td_avg": _rate_per_15(td_landed, fight_minutes),
        f"{prefix}td_acc": _pct(td_landed, td_attempted),
        f"{prefix}td_def": _defense_pct(opp_td_landed, opp_td_attempted),
        f"{prefix}sub_avg": _rate_per_15(sub_att, fight_minutes),
        f"{prefix}sig_str_landed": sig_landed,
        f"{prefix}sig_str_attempted": sig_attempted,
        f"{prefix}td_landed": td_landed,
        f"{prefix}td_attempted": td_attempted,
        f"{prefix}kd": kd,
        f"{prefix}sub_att": sub_att,
        f"{prefix}rev": rev,
        f"{prefix}ctrl_seconds": ctrl_seconds,
        f"{prefix}total_str_landed": total_str_landed,
        f"{prefix}total_str_attempted": total_str_attempted,
        f"{prefix}head_landed": head_landed,
        f"{prefix}head_attempted": head_attempted,
        f"{prefix}body_landed": body_landed,
        f"{prefix}body_attempted": body_attempted,
        f"{prefix}leg_landed": leg_landed,
        f"{prefix}leg_attempted": leg_attempted,
        f"{prefix}distance_landed": distance_landed,
        f"{prefix}distance_attempted": distance_attempted,
        f"{prefix}clinch_landed": clinch_landed,
        f"{prefix}clinch_attempted": clinch_attempted,
        f"{prefix}ground_landed": ground_landed,
        f"{prefix}ground_attempted": ground_attempted,
        f"{prefix}total_str_pm": _rate_per_minute(total_str_landed, fight_minutes),
        f"{prefix}sig_str_share": _ratio(sig_landed, total_str_landed),
        f"{prefix}head_str_share": _ratio(head_landed, sig_landed),
        f"{prefix}body_str_share": _ratio(body_landed, sig_landed),
        f"{prefix}leg_str_share": _ratio(leg_landed, sig_landed),
        f"{prefix}distance_str_share": _ratio(distance_landed, sig_landed),
        f"{prefix}clinch_str_share": _ratio(clinch_landed, sig_landed),
        f"{prefix}ground_str_share": _ratio(ground_landed, sig_landed),
        f"{prefix}control_share": _ratio(ctrl_seconds, fight_seconds),
        f"{prefix}control_per_td": _ratio(ctrl_seconds, td_landed),
        f"{prefix}finish_threat": _rate_per_minute(_coalesce_zero(kd) + _coalesce_zero(sub_att), fight_minutes),
    }


def _rate_per_minute(value: float | None, fight_minutes: float) -> float:
    if value is None or pd.isna(value) or pd.isna(fight_minutes) or fight_minutes <= 0:
        return np.nan
    return float(value / fight_minutes)


def _rate_per_15(value: float | None, fight_minutes: float) -> float:
    if value is None or pd.isna(value) or pd.isna(fight_minutes) or fight_minutes <= 0:
        return np.nan
    return float(value / fight_minutes * 15.0)


def _pct(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator * 100.0)


def _ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _defense_pct(lnd: float | None, att: float | None) -> float:
    if lnd is None or att is None or pd.isna(lnd) or pd.isna(att) or att <= 0:
        return np.nan
    return float((1.0 - (lnd / att)) * 100.0)


def _coalesce_zero(value: float | None) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _empty_profile_lookup_row() -> dict[str, object]:
    return {
        "height": np.nan,
        "reach": np.nan,
        "weight": np.nan,
        "stance": None,
        "dob": None,
    }


def _profile_value_missing(value) -> bool:
    if value is None or pd.isna(value):
        return True
    if isinstance(value, str):
        return value.strip() in {"", "-", "--", "N/A", "nan", "NaN"}
    return False


def _valid_profile_stance(value) -> bool:
    if _profile_value_missing(value):
        return False
    return bool(pd.notna(encode_stance(value)))


def _consistent_profile_ufcstats_id(row: pd.Series | dict) -> str | None:
    carriers: dict[str, str | None] = {}
    for column in (
        "fighter_id",
        "fighter_url",
        "ufcstats_url",
        "stable_ufcstats_id",
    ):
        value = row.get(column)
        if _profile_value_missing(value):
            continue
        text = str(value).strip()
        if column in {"fighter_url", "ufcstats_url"}:
            carriers[column] = source_entity_id("ufcstats", text) or None
        else:
            carriers[column] = (
                text.casefold()
                if re.fullmatch(r"[0-9a-f]{16}", text, flags=re.IGNORECASE)
                else None
            )
    if not carriers or any(value is None for value in carriers.values()):
        return None
    identities = {value for value in carriers.values() if value}
    return next(iter(identities)) if len(identities) == 1 else None


def _profile_field_missing(field: str, value) -> bool:
    if field == "stance":
        return not _valid_profile_stance(value)
    return _profile_value_missing(value)


def _coerce_profile_height_cm(value) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not pd.isna(numeric):
        return float(numeric)
    parsed = _parse_height_cm(value)
    return float(parsed) if parsed is not None else np.nan


def _coerce_profile_reach_cm(value) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not pd.isna(numeric):
        return float(numeric)
    parsed = _parse_reach_cm(value)
    return float(parsed) if parsed is not None else np.nan


def _coerce_profile_weight_lbs(value) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not pd.isna(numeric):
        return float(numeric)
    return _parse_weight_lbs(value)


def _build_legacy_static_profile_backfill_lookup(legacy_df: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    """Derive conservative static-profile backfills from legacy fight rows.

    Only fields with exactly one observed value across the full legacy history
    are considered safe to reuse.
    """
    if legacy_df is None or legacy_df.empty:
        return {}

    collected: dict[str, dict[str, set[float]]] = {}
    for _, row in legacy_df.iterrows():
        for side in ("a", "b"):
            fighter_key = _normalize_name(row.get(f"fighter_{side}"))
            if not fighter_key:
                continue
            fighter_fields = collected.setdefault(
                fighter_key,
                {field: set() for field in _LEGACY_STATIC_PROFILE_FIELDS},
            )
            for field in _LEGACY_STATIC_PROFILE_FIELDS:
                numeric = pd.to_numeric(pd.Series([row.get(f"{side}_{field}")]), errors="coerce").iloc[0]
                if pd.isna(numeric):
                    continue
                fighter_fields[field].add(round(float(numeric), 2))

    backfill_lookup: dict[str, dict[str, float]] = {}
    for fighter_key, field_values in collected.items():
        resolved = {
            field: next(iter(values))
            for field, values in field_values.items()
            if len(values) == 1
        }
        if resolved:
            backfill_lookup[fighter_key] = resolved

    return backfill_lookup


def _iter_active_roster_alias_names(row: pd.Series | dict) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()

    def _add_alias(value) -> None:
        if _profile_value_missing(value):
            return
        text = str(value or "").strip()
        key = _normalize_name(text)
        if not key or key in seen:
            return
        seen.add(key)
        aliases.append(text)

    alias_fields = ["ufcstats_name", "official_name"]
    if _official_url_identity_trusted(row):
        alias_fields.extend(["profile_name", "slug_name"])
    for field in alias_fields:
        _add_alias(row.get(field))

    if _official_url_identity_trusted(row):
        for field in ("canonical_athlete_url", "official_athlete_url"):
            url = str(row.get(field) or "").strip()
            slug = urlparse(url).path.strip("/").rsplit("/", 1)[-1] if url else ""
            if slug:
                _add_alias(slug.replace("-", " "))
        for value in str(row.get("alternate_slug_names") or "").split("|"):
            _add_alias(value)
    return aliases


def _official_url_identity_trusted(row: pd.Series | dict) -> bool:
    from src.data.ufc_active_roster import _official_url_identity_positively_trusted

    return _official_url_identity_positively_trusted(row)


def _is_test_or_staging_profile(row: pd.Series | dict) -> bool:
    from urllib.parse import urlparse

    test_keys = {"test", "test fighter", "test test", "testy test", "testing test"}
    names = [row.get("official_name"), row.get("profile_name"), row.get("slug_name")]
    names.extend(str(row.get("alternate_slug_names") or "").split("|"))
    name_keys = {
        _normalize_name(value)
        for value in names
        if str(value or "").strip()
    }
    if name_keys & test_keys:
        return True
    url = str(row.get("official_athlete_url") or "").strip()
    path = urlparse(url).path.strip("/").lower() if url else ""
    slug = path.rsplit("/", 1)[-1] if path else ""
    return slug in {"test", "test-fighter", "test-test", "testy-test", "testing-test"}


def _merge_official_active_roster_profile_backfill(
    lookup: dict[str, dict],
    *,
    official_active_roster_path: Path | None = None,
    ambiguous_name_keys: frozenset[str] | None = None,
) -> tuple[int, dict[str, int]]:
    """Merge observed UFC.com roster profile fields without deriving any synthetic DOB values."""
    official_active_roster_path = (
        Path(official_active_roster_path)
        if official_active_roster_path is not None
        else OFFICIAL_ACTIVE_ROSTER_PATH
    )
    if not official_active_roster_path.exists():
        return 0, 0

    roster_df = pd.read_csv(official_active_roster_path)
    if roster_df.empty:
        logger.warning("Official active-roster artifact at %s is empty", official_active_roster_path)
        return 0, 0

    fighters_augmented = 0
    applied_counts = {
        "height": 0,
        "reach": 0,
        "weight": 0,
        "stance": 0,
        "dob": 0,
    }
    if ambiguous_name_keys is None:
        ambiguous_name_keys = _ambiguous_name_keys()
    for _, row in roster_df.iterrows():
        if not _official_url_identity_trusted(row) or _is_test_or_staging_profile(row):
            continue
        roster_ids = {
            value
            for value in (
                normalize_ufcstats_id(row.get("stable_ufcstats_id")),
                normalize_ufcstats_id(row.get("fighter_id")),
                normalize_ufcstats_id(row.get("ufcstats_url")),
            )
            if value
        }
        if len(roster_ids) != 1:
            # UFC.com names/slugs can nominate a profile, but they cannot bind
            # physical fields to model history without one stable UFCStats ID.
            continue
        roster_id = next(iter(roster_ids))
        target_key = f"ufcstats:{roster_id}"
        if target_key not in lookup:
            # Do not seed a new modeled identity from UFC.com alone.
            continue
        dob = pd.to_datetime(
            row.get("birth_date") if "birth_date" in row else row.get("dob"),
            errors="coerce",
            format="mixed",
        )
        parsed_fields = {
            "height": _coerce_profile_height_cm(row.get("height")),
            "reach": _coerce_profile_reach_cm(row.get("reach")),
            "weight": _coerce_profile_weight_lbs(row.get("weight")),
            "stance": row.get("stance"),
            "dob": dob if pd.notna(dob) else None,
        }
        if all(
            _profile_value_missing(value)
            for field, value in parsed_fields.items()
            if field != "dob"
        ) and parsed_fields["dob"] is None:
            continue

        fighter_changed = False
        profile = lookup[target_key]
        for field, value in parsed_fields.items():
            if not _profile_field_missing(field, profile.get(field)):
                continue
            if field == "dob":
                if value is None:
                    continue
            elif _profile_field_missing(field, value):
                continue
            profile[field] = value
            applied_counts[field] += 1
            fighter_changed = True
        if fighter_changed:
            fighters_augmented += 1

    return fighters_augmented, applied_counts


def _merge_active_roster_alias_profile_values(
    lookup: dict[str, dict],
    *,
    official_active_roster_path: Path | None = None,
    ambiguous_name_keys: frozenset[str] | None = None,
) -> tuple[int, dict[str, int]]:
    """Do not propagate model profile values through UFC.com name aliases.

    The historical implementation merged values across normalized roster names.
    That can cross same-name identities even when a UFCStats ID exists elsewhere.
    Stable-ID backfill is handled by
    ``_merge_official_active_roster_profile_backfill``; aliases are discovery
    metadata only and contribute no model values.
    """
    del lookup, official_active_roster_path, ambiguous_name_keys
    return 0, {field: 0 for field in ("height", "reach", "weight", "stance", "dob")}


def _apply_reviewed_current_profile_overlays(
    lookup: dict[str, dict],
    *,
    supplemental_profiles_path: Path | None,
    official_active_roster_path: Path | None,
) -> int:
    """Apply exact current/future-only pins without backfilling model history."""

    applied = 0
    supplement_path = Path(
        supplemental_profiles_path
        if supplemental_profiles_path is not None
        else PROFILE_SUPPLEMENT_PATH
    )
    from src.data.provider_consumer_references import verify_provider_consumer_artifact
    verify_provider_consumer_artifact(supplement_path)
    supplement_rows = (
        pd.read_csv(supplement_path).to_dict(orient="records")
        if supplement_path.exists()
        else []
    )
    observations = reviewed_profile_recovery_observations(supplement_rows)
    for recovery, _observation in observations:
        profile = lookup.get(f"ufcstats:{recovery['stable_ufcstats_id']}")
        if profile is None:
            continue
        field = str(recovery["field"])
        value = recovery["value"]
        current = profile.get(field)
        if not _profile_field_missing(field, current):
            if str(current).strip().casefold() != str(value).strip().casefold():
                raise ValueError(
                    "Reviewed current profile recovery conflicts with a nonblank "
                    f"source value: {recovery['canonical_name']} {field}"
                )
            continue
        profile[field] = value
        applied += 1

    roster_path = Path(
        official_active_roster_path
        if official_active_roster_path is not None
        else OFFICIAL_ACTIVE_ROSTER_PATH
    )
    official_rows = (
        pd.read_csv(roster_path).to_dict(orient="records")
        if roster_path.exists()
        else []
    )
    identity_registry = load_reviewed_roster_identity_registry()
    bindings_by_entity = {
        record["official_source_entity_id"]: record
        for record in identity_registry["identity_bindings"]
    }

    def _target_key(decision: Mapping[str, object]) -> str:
        binding = bindings_by_entity.get(decision["official_source_entity_id"])
        if binding is None:
            return ""
        stable_id = str(binding["stable_ufcstats_id"])
        return (
            f"ufcstats:{stable_id}"
            if stable_id
            else f"ufc:{binding['official_source_entity_id']}"
        )

    def _normalized_value(decision: Mapping[str, object]):
        field = str(decision["profile_field"])
        value = decision["normalized_value"]
        if field == "dob":
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                raise ValueError(
                    f"Reviewed identity DOB is invalid: {decision['cell_id']}"
                )
            return pd.Timestamp(parsed)
        return value

    def _values_match(field: str, current, expected) -> bool:
        if field in {"height", "reach", "weight"}:
            current_numeric = pd.to_numeric(
                pd.Series([current]), errors="coerce"
            ).iloc[0]
            return bool(
                pd.notna(current_numeric)
                and np.isclose(float(current_numeric), float(expected), atol=1e-6)
            )
        if field == "dob":
            parsed = pd.to_datetime(current, errors="coerce")
            return bool(
                pd.notna(parsed)
                and pd.Timestamp(parsed).normalize()
                == pd.Timestamp(expected).normalize()
            )
        return str(current).strip().casefold() == str(expected).strip().casefold()

    selected_identity_observations = reviewed_identity_profile_observations(
        supplement_rows,
        official_rows,
    )
    for decision, _observation in selected_identity_observations:
        key = _target_key(decision)
        if not key:
            raise ValueError(
                f"Reviewed identity field has no approved binding: {decision['cell_id']}"
            )
        profile = lookup.setdefault(key, _empty_profile_lookup_row())
        field = str(decision["profile_field"])
        value = _normalized_value(decision)
        current = profile.get(field)
        if not _profile_field_missing(field, current):
            if not _values_match(field, current, value):
                raise ValueError(
                    "Reviewed identity profile recovery conflicts with a nonblank "
                    f"source value: {decision['cell_id']}"
                )
            continue
        profile[field] = value
        applied += 1

    # UFCStats-backed selectors are validated against the stable lookup itself;
    # they are never fabricated when that exact source profile is absent.
    for decision in reviewed_identity_field_decisions():
        selector = decision["observation_selector"]
        if (
            decision["decision_status"] != "accepted_current_future"
            or selector["carrier"] != "ufcstats"
        ):
            continue
        key = _target_key(decision)
        profile = lookup.get(key)
        if profile is None:
            continue
        field = str(decision["profile_field"])
        expected = _normalized_value(decision)
        current = profile.get(field)
        if _profile_field_missing(field, current) or not _values_match(
            field, current, expected
        ):
            raise ValueError(
                "Reviewed UFCStats identity observation is missing or conflicting: "
                f"{decision['cell_id']}"
            )
    return applied


def _apply_reviewed_identity_profile_field_holds(
    lookup: dict[str, dict],
) -> int:
    """Blank the eight exact 55-review fields whose values remain unaccepted."""

    registry = load_reviewed_roster_identity_registry()
    bindings_by_entity = {
        record["official_source_entity_id"]: record
        for record in registry["identity_bindings"]
    }
    applied = 0
    for decision in registry["field_decisions"]:
        if decision["decision_status"] not in {
            "conflicting_field_hold",
            "owner_field_hold",
        }:
            continue
        binding = bindings_by_entity.get(decision["official_source_entity_id"])
        if binding is None:
            raise ValueError(
                f"Reviewed field hold has no identity binding: {decision['cell_id']}"
            )
        stable_id = str(binding["stable_ufcstats_id"])
        key = (
            f"ufcstats:{stable_id}"
            if stable_id
            else f"ufc:{binding['official_source_entity_id']}"
        )
        profile = lookup.get(key)
        if profile is None:
            continue
        field = str(decision["profile_field"])
        if not _profile_field_missing(field, profile.get(field)):
            applied += 1
        profile[field] = None if field == "stance" else np.nan
    return applied


def _apply_permanent_profile_locks_to_lookup(lookup: dict[str, dict]) -> int:
    """Apply stable-ID owner locks to every current and historical load."""

    from src.data.selected_source_inputs import active_sources
    selected = active_sources()
    if selected is None:
        locks = permanent_profile_locks()
    else:
        from src.data.reviewed_roster_recovery import load_reviewed_roster_recovery_registry
        locks = load_reviewed_roster_recovery_registry(
            selected.path("roster_recovery_registry")
        )["permanent_profile_locks"]
    applied = 0
    for lock in locks:
        profile = lookup.get(f"ufcstats:{lock['stable_ufcstats_id']}")
        if profile is None:
            continue
        locked_dob = pd.Timestamp(lock["normalized_value"])
        current_dob = pd.to_datetime(profile.get("dob"), errors="coerce")
        if pd.isna(current_dob) or pd.Timestamp(current_dob).normalize() != locked_dob:
            profile["dob"] = locked_dob
            applied += 1
    return applied


def _apply_reviewed_profile_field_holds(
    lookup: dict[str, dict],
) -> int:
    """Blank exact stable-ID fields whose reviewed sources still conflict."""

    applied = 0
    for hold in reviewed_profile_field_holds():
        if (
            hold["scope"] != "current_and_future"
            or hold["historical_backfill_allowed"] is not False
        ):
            raise ValueError("Reviewed profile field hold has an invalid temporal scope")
        field = str(hold["field"])
        if field != "height":
            raise ValueError(f"Unsupported reviewed profile field hold: {field}")
        profile = lookup.get(f"ufcstats:{hold['stable_ufcstats_id']}")
        if profile is None:
            continue
        if not _profile_field_missing(field, profile.get(field)):
            applied += 1
        profile[field] = np.nan
    return applied


def _load_scraped_fighter_lookup(
    scraped_fighters_path: Path,
    *,
    supplemental_profiles_path: Path | None = None,
    official_active_roster_path: Path | None = None,
    profile_integrity_decisions_path: Path | None = None,
    prior_profile_integrity_decisions_path: Path | None = None,
    legacy_df: pd.DataFrame | None = None,
    include_reviewed_current_profile_overlays: bool = False,
) -> dict[str, dict]:
    """Load optional fighter profile data for static attributes like height and reach."""
    if profile_integrity_decisions_path is not None:
        decisions = load_profile_decisions(
            profile_integrity_decisions_path,
            prior_decisions_path=prior_profile_integrity_decisions_path,
        )
        lookup: dict[str, dict] = {}

        # The accepted cache is the grandfathered baseline.  It is joined only
        # by stable UFCStats ID; current listed weight is always removed.  The
        # decision artifact then overrides verified corrections and blanks every
        # audited field that was rejected or left unresolved.  This lets a small
        # delta artifact protect changed cells without silently dropping all
        # unrelated accepted profile fields.
        if Path(scraped_fighters_path).exists():
            baseline = pd.read_csv(scraped_fighters_path)
            for _, row in baseline.iterrows():
                fighter_id = _consistent_profile_ufcstats_id(row)
                if not fighter_id:
                    raise ValueError(
                        "Model-safe profile baseline row must carry exactly one "
                        f"consistent UFCStats identity: {row.get('name')}"
                    )
                key = f"ufcstats:{fighter_id}"
                if key in lookup:
                    raise ValueError(
                        f"Duplicate UFCStats identity in model-safe profile baseline: {fighter_id}"
                    )
                dob = pd.to_datetime(row.get("dob"), errors="coerce", format="mixed")
                lookup[key] = {
                    "height": _coerce_profile_height_cm(row.get("height")),
                    "reach": _coerce_profile_reach_cm(row.get("reach")),
                    "weight": np.nan,
                    "stance": (
                        row.get("stance") if _valid_profile_stance(row.get("stance")) else None
                    ),
                    "dob": dob if pd.notna(dob) else None,
                }

        for _, decision in decisions.iterrows():
            fighter_id = normalize_ufcstats_id(decision.get("stable_ufcstats_id"))
            if not fighter_id:
                raise ValueError("Model-safe profile decision is missing a UFCStats ID")
            profile = lookup.setdefault(
                f"ufcstats:{fighter_id}", _empty_profile_lookup_row()
            )
            field = str(decision.get("field") or "").strip().casefold()
            status = str(decision.get("decision_status") or "").strip().casefold()
            value = decision.get("model_safe_value") if status in {"accepted", "owner_adjudicated"} else None
            if field == "height":
                profile[field] = _coerce_profile_height_cm(value)
            elif field == "reach":
                profile[field] = _coerce_profile_reach_cm(value)
            elif field == "stance":
                profile[field] = value if _valid_profile_stance(value) else None
            elif field == "dob":
                dob = pd.to_datetime(value, errors="coerce", format="mixed")
                profile[field] = dob if pd.notna(dob) else None
            elif field == "weight":
                profile[field] = np.nan
        for profile in lookup.values():
            profile["weight"] = np.nan
        _apply_permanent_profile_locks_to_lookup(lookup)
        if include_reviewed_current_profile_overlays:
            _apply_reviewed_current_profile_overlays(
                lookup,
                supplemental_profiles_path=supplemental_profiles_path,
                official_active_roster_path=official_active_roster_path,
            )
            _apply_reviewed_profile_field_holds(lookup)
            _apply_reviewed_identity_profile_field_holds(lookup)
        return lookup

    lookup: dict[str, dict] = {}
    ambiguous_name_keys = _ambiguous_name_keys(scraped_fighters_path)

    def _merge_profile_row(row: pd.Series | dict) -> None:
        name_key = _normalize_name(row.get("name"))
        if not name_key:
            return
        fighter_id = _consistent_profile_ufcstats_id(row)
        if not fighter_id:
            return
        dob = pd.to_datetime(row.get("dob"), errors="coerce", format="mixed")
        primary_key = f"ufcstats:{fighter_id}"
        if primary_key in lookup:
            raise ValueError(
                "Duplicate UFCStats identity in scraped profile artifact: "
                f"{fighter_id}"
            )
        profile = _empty_profile_lookup_row()
        lookup[primary_key] = profile
        merged_fields = {
            "height": _coerce_profile_height_cm(row.get("height")),
            "reach": _coerce_profile_reach_cm(row.get("reach")),
            "weight": _coerce_profile_weight_lbs(row.get("weight")),
            "stance": row.get("stance"),
            "dob": dob if pd.notna(dob) else None,
        }
        for field, value in merged_fields.items():
            if not _profile_field_missing(field, profile.get(field)):
                continue
            if field == "dob":
                if value is not None:
                    profile[field] = value
                continue
            if _profile_field_missing(field, value):
                continue
            profile[field] = value

    if scraped_fighters_path.exists():
        fighters_df = pd.read_csv(scraped_fighters_path)
        if fighters_df.empty or "name" not in fighters_df.columns:
            logger.warning(
                "Scraped fighter profile artifact at %s is empty or missing the required 'name' column",
                scraped_fighters_path,
            )
        else:
            for _, row in fighters_df.iterrows():
                _merge_profile_row(row)
            logger.info("Loaded %d scraped fighter profiles from %s", len(lookup), scraped_fighters_path)
    else:
        logger.info(
            "Scraped fighter profile artifact not found at %s; pulled training rows will rely on any secondary backfills",
            scraped_fighters_path,
        )

    if supplemental_profiles_path is not None or official_active_roster_path is not None:
        logger.info(
            "Ignoring external profile backfills for model features because no "
            "validated profile-integrity decision artifact was supplied"
        )

    legacy_backfills = _build_legacy_static_profile_backfill_lookup(legacy_df)
    backfilled_fields = {field: 0 for field in _LEGACY_STATIC_PROFILE_FIELDS}
    fighters_augmented = 0
    for fighter_key, backfill_values in legacy_backfills.items():
        profile = lookup.setdefault(fighter_key, _empty_profile_lookup_row())
        fighter_changed = False
        for field, value in backfill_values.items():
            if not _profile_value_missing(profile.get(field)):
                continue
            profile[field] = value
            backfilled_fields[field] += 1
            fighter_changed = True
        if fighter_changed:
            fighters_augmented += 1

    if fighters_augmented:
        logger.info(
            "Applied conservative legacy static-profile backfills to %d fighters (height=%d, reach=%d)",
            fighters_augmented,
            backfilled_fields["height"],
            backfilled_fields["reach"],
        )

    # ------------------------------------------------------------------
    # Fighter name aliases — misspellings in raw UFCStats data that
    # prevent profile joins.  Alias key → canonical key.
    # ------------------------------------------------------------------
    _FIGHTER_NAME_ALIASES: dict[str, str] = {
        _normalize_name("Rafael Cerquiera"): _normalize_name("Rafael Cerqueira"),
    }
    for alias_key, canon_key in _FIGHTER_NAME_ALIASES.items():
        if canon_key in lookup and alias_key not in lookup:
            lookup[alias_key] = lookup[canon_key]

    _apply_permanent_profile_locks_to_lookup(lookup)
    if include_reviewed_current_profile_overlays:
        _apply_reviewed_current_profile_overlays(
            lookup,
            supplemental_profiles_path=supplemental_profiles_path,
            official_active_roster_path=official_active_roster_path,
        )
        _apply_reviewed_profile_field_holds(lookup)
        _apply_reviewed_identity_profile_field_holds(lookup)

    return lookup


def _parse_weight_lbs(value) -> float:
    if value is None or pd.isna(value):
        return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    if not match:
        return np.nan
    return float(match.group(1))


def _extract_fighter_attributes(
    fighter_name,
    event_date: pd.Timestamp,
    fighter_lookup: dict[str, dict],
    *,
    prefix: str,
    fighter_id: object = None,
    opponent_id: object = None,
    fight_url: object = None,
    event_weight_evidence: Mapping[str, object] | None = None,
) -> dict:
    normalized_id = normalize_ufcstats_id(fighter_id)
    profile = (
        fighter_lookup.get(f"ufcstats:{normalized_id}")
        if normalized_id
        else None
    )
    # Once a stable identity is supplied, a miss must stay a miss.  Falling
    # back to a same-name row can attach another fighter's immutable fields.
    if normalized_id and profile is None:
        return {}
    if profile is None and not fighter_identity_is_ambiguous(
        fighter_name,
        ambiguous_name_keys=_ambiguous_name_keys(),
    ):
        profile = fighter_lookup.get(_normalize_name(fighter_name))
    if not profile:
        return {}

    age = np.nan
    dob = profile.get("dob")
    if dob is not None and pd.notna(dob):
        age = float((event_date - dob).days / 365.25)

    return {
        f"{prefix}height": profile.get("height"),
        f"{prefix}reach": profile.get("reach"),
        # Current listed/body weight is not evidence of the fighter's weight at
        # this historical event.  The event's weight class remains available as
        # fight context; absent exact fight-bound weigh-in evidence, use NaN.
        f"{prefix}weight": event_specific_weight_lbs(
            event_weight_evidence,
            fighter_id=fighter_id,
            opponent_id=opponent_id,
            event_date=event_date,
            fight_url=fight_url,
            side=prefix.rstrip("_"),
        ),
        f"{prefix}stance": profile.get("stance"),
        f"{prefix}age": age,
    }


def load_or_fetch_event_date_lookup(
    *,
    required_events=None,
    cache_path: Path | None = None,
    force_refresh: bool = False,
    event_url_hints: dict[str, str] | None = None,
) -> dict[str, pd.Timestamp]:
    """Load cached UFC event dates and backfill missing entries from UFCStats metadata."""
    metadata_lookup = load_or_fetch_event_metadata_lookup(
        required_events=required_events,
        cache_path=cache_path,
        force_refresh=force_refresh,
        event_url_hints=event_url_hints,
    )
    return {
        key: metadata.get("event_date")
        for key, metadata in metadata_lookup.items()
        if pd.notna(metadata.get("event_date"))
    }


def load_or_fetch_event_metadata_lookup(
    *,
    required_events=None,
    cache_path: Path | None = None,
    force_refresh: bool = False,
    event_url_hints: dict[str, str] | None = None,
) -> dict[str, dict[str, object]]:
    """Load cached UFC event metadata and backfill missing entries from UFCStats."""
    cache_path = Path(cache_path) if cache_path is not None else EVENT_DATES_CACHE_PATH
    cached = _load_event_metadata_cache(cache_path)
    required_iterable = [] if required_events is None else list(required_events)
    required_keys = {_normalize_name(value) for value in required_iterable if _normalize_name(value)}
    missing_keys = required_keys - set(cached)

    if force_refresh or missing_keys or not cached:
        fetched = _fetch_ufcstats_event_metadata()
        if fetched:
            cached.update(fetched)
            _save_event_metadata_cache(cache_path, cached)
            missing_keys = required_keys - set(cached)

    if missing_keys and event_url_hints:
        fetched_from_fights = _fetch_missing_event_metadata_from_fight_urls(
            missing_keys,
            event_url_hints,
        )
        if fetched_from_fights:
            cached.update(fetched_from_fights)
            _save_event_metadata_cache(cache_path, cached)
            missing_keys = required_keys - set(cached)

    if missing_keys:
        logger.warning("Missing UFC event metadata mappings for %d events", len(missing_keys))
    return cached


def _load_event_metadata_cache(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    cached_df = pd.read_csv(path)
    if cached_df.empty or "event_name" not in cached_df.columns or "event_date" not in cached_df.columns:
        return {}

    lookup: dict[str, dict[str, object]] = {}
    for _, row in cached_df.iterrows():
        key = _normalize_name(row.get("event_name"))
        event_date = pd.to_datetime(row.get("event_date"), errors="coerce", format="mixed")
        if key and pd.notna(event_date):
            location = row.get("location") if "location" in cached_df.columns else None
            location = _clean_text(location) if location is not None and not pd.isna(location) else None
            lookup[key] = {
                "event_date": event_date,
                "location": location,
            }
    return lookup


def _save_event_metadata_cache(path: Path, lookup: dict[str, dict[str, object]]) -> None:
    rows = [
        {
            "event_name": event_name,
            "event_date": metadata["event_date"].date().isoformat(),
            "location": metadata.get("location"),
        }
        for event_name, metadata in sorted(lookup.items())
        if metadata.get("event_date") is not None and pd.notna(metadata.get("event_date"))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_event_url_hints_from_results(results_df: pd.DataFrame) -> dict[str, str]:
    if results_df.empty or "EVENT" not in results_df.columns or "URL" not in results_df.columns:
        return {}

    lookup: dict[str, str] = {}
    for _, row in results_df.iterrows():
        event_key = _normalize_name(row.get("EVENT"))
        fight_url = _clean_text(row.get("URL"))
        if event_key and fight_url and event_key not in lookup:
            lookup[event_key] = fight_url
    return lookup


def _get_ufcstats_soup(url: str) -> BeautifulSoup:
    resp = request_ufcstats(url, headers=_UFCSTATS_HEADERS, timeout=30)
    time.sleep(1.5)  # Rate limit: match scraper.py REQUEST_DELAY
    return BeautifulSoup(resp.text, "lxml")


def _extract_event_url_from_fight_page(fight_url: str) -> str | None:
    soup = _get_ufcstats_soup(fight_url)
    for link in soup.select("a.b-link[href]"):
        href = _clean_text(link.get("href"))
        if "event-details/" in href:
            return href
    return None


def _fetch_event_metadata_from_event_url(event_url: str) -> dict[str, object] | None:
    soup = _get_ufcstats_soup(event_url)
    title_el = soup.select_one("h2.b-content__title span")
    event_name = _clean_text(title_el.get_text(" ", strip=True)) if title_el is not None else ""
    metadata: dict[str, object] = {}
    for item in soup.select("li.b-list__box-list-item"):
        text = _clean_text(item.get_text(" ", strip=True))
        if not text:
            continue
        label, _, raw_value = text.partition(":")
        value = _clean_text(raw_value)
        if not value:
            continue
        key = label.strip().lower()
        if key == "date":
            metadata["event_date"] = pd.to_datetime(value, errors="coerce", format="mixed")
        elif key == "location":
            metadata["location"] = value

    event_date = metadata.get("event_date")
    if not event_name or event_date is None or pd.isna(event_date):
        return None
    return {
        "event_name": event_name,
        "event_date": event_date,
        "location": metadata.get("location"),
    }


def _fetch_missing_event_metadata_from_fight_urls(
    missing_event_keys: set[str],
    event_url_hints: dict[str, str],
) -> dict[str, dict[str, object]]:
    hint_lookup = {
        _normalize_name(key): _clean_text(value)
        for key, value in event_url_hints.items()
        if _normalize_name(key) and _clean_text(value)
    }
    if not missing_event_keys or not hint_lookup:
        return {}

    fetched: dict[str, dict[str, object]] = {}
    event_url_cache: dict[str, dict[str, object] | None] = {}
    for event_key in sorted(missing_event_keys):
        fight_url = hint_lookup.get(event_key)
        if not fight_url:
            continue
        try:
            event_url = _extract_event_url_from_fight_page(fight_url)
            if not event_url:
                continue
            if event_url not in event_url_cache:
                event_url_cache[event_url] = _fetch_event_metadata_from_event_url(event_url)
            metadata = event_url_cache[event_url]
        except requests.RequestException as exc:
            logger.warning("Failed to backfill UFC event metadata for %s: %s", event_key, exc)
            continue

        if metadata is None:
            continue
        fetched[event_key] = {
            "event_date": metadata["event_date"],
            "location": metadata.get("location"),
        }

    if fetched:
        logger.info(
            "Backfilled %d UFC event metadata rows from fight detail pages",
            len(fetched),
        )
    return fetched


def _fetch_ufcstats_event_metadata() -> dict[str, dict[str, object]]:
    """Fetch event title/date/location metadata from the UFCStats completed-events listing."""
    page = 1
    lookup: dict[str, dict[str, object]] = {}
    while True:
        soup = _get_ufcstats_soup(f"{UFCSTATS_BASE_URL}?page={page}")
        rows = soup.select("tr.b-statistics__table-row")
        parsed_rows = 0
        for row in rows:
            link = row.select_one("a.b-link")
            date_el = row.select_one("span.b-statistics__date")
            if link is None or date_el is None:
                continue
            cells = row.select("td")
            event_name = _clean_text(link.get_text(" ", strip=True))
            event_date_text = _clean_text(date_el.get_text(" ", strip=True))
            location = _clean_text(cells[1].get_text(" ", strip=True)) if len(cells) > 1 else None
            event_date = pd.to_datetime(event_date_text, errors="coerce", format="mixed")
            if not event_name or pd.isna(event_date):
                continue
            lookup[_normalize_name(event_name)] = {
                "event_date": event_date,
                "location": location or None,
            }
            parsed_rows += 1
        if parsed_rows == 0:
            break
        page += 1

    logger.info("Fetched %d UFC event metadata rows from UFCStats", len(lookup))
    return lookup


def _pulled_data_to_scraped_rows(
    results_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    *,
    event_date_lookup: dict[str, object],
    event_metadata_lookup: dict[str, dict[str, object]] | None = None,
) -> pd.DataFrame:
    totals_lookup = _aggregate_pulled_stats(stats_df)
    event_metadata_lookup = event_metadata_lookup or {}
    rows: list[dict] = []
    skipped_no_date = 0
    skipped_bad_bout = 0

    for _, row in results_df.iterrows():
        event_title = _clean_text(row.get("EVENT"))
        fighter_a, fighter_b = _split_bout_fighters(row.get("BOUT"))
        if not fighter_a or not fighter_b:
            skipped_bad_bout += 1
            continue

        event_key = _normalize_name(event_title)
        event_metadata = event_metadata_lookup.get(event_key, {})
        event_date_value = event_date_lookup.get(event_key)
        if event_date_value is None:
            event_date_value = event_metadata.get("event_date")
        event_date = pd.to_datetime(event_date_value, errors="coerce", format="mixed")
        if pd.isna(event_date):
            skipped_no_date += 1
            continue

        fighter_a_stats = totals_lookup.get((event_key, _normalize_name(str(row.get("BOUT"))), _normalize_name(fighter_a)), {})
        fighter_b_stats = totals_lookup.get((event_key, _normalize_name(str(row.get("BOUT"))), _normalize_name(fighter_b)), {})
        policy_version = row.get("RESULT_POLICY_VERSION")
        if has_result_policy_marker(policy_version):
            classification = validate_marked_result(
                row.get("OUTCOME"),
                row.get("METHOD"),
                policy_version=policy_version,
                persisted_result_class=row.get("RESULT_CLASS"),
                persisted_binary_target_eligible=row.get(
                    "BINARY_TARGET_ELIGIBLE"
                ),
            )
            winner = winner_for_side(
                classification.winner_side,
                fighter_a=fighter_a,
                fighter_b=fighter_b,
            )
        else:
            winner = _winner_from_outcome(
                row.get("OUTCOME"),
                fighter_a=fighter_a,
                fighter_b=fighter_b,
            )

        converted = {
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "fighter_a_id": normalize_ufcstats_id(row.get("fighter_a_id")),
            "fighter_b_id": normalize_ufcstats_id(row.get("fighter_b_id")),
            "winner": winner,
            "method": row.get("METHOD"),
            "official_outcome": row.get("OUTCOME"),
            "result_policy_version": policy_version,
            "result_class": row.get("RESULT_CLASS"),
            "binary_target_eligible": row.get("BINARY_TARGET_ELIGIBLE"),
            "result_quarantine_reason": row.get("RESULT_QUARANTINE_REASON"),
            "round": row.get("ROUND"),
            "time": row.get("TIME"),
            "time_format": row.get("TIME FORMAT"),
            "weight_class": row.get("WEIGHTCLASS"),
            "fight_url": row.get("URL"),
            "event_title": event_title,
            "event_date": event_date,
            "location": event_metadata.get("location"),
            **_prefix_stats(fighter_a_stats, prefix="a_"),
            **_prefix_stats(fighter_b_stats, prefix="b_"),
        }
        rows.append(converted)

    logger.info(
        "Converted %d pulled UFC results rows into scraped-like fight rows (%d skipped for missing dates, %d skipped for bout parse failures)",
        len(rows),
        skipped_no_date,
        skipped_bad_bout,
    )
    return pd.DataFrame(rows)


def _aggregate_pulled_stats(stats_df: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, object]]:
    if stats_df.empty:
        return {}

    frame = stats_df.copy()
    frame["__event_key"] = frame["EVENT"].map(_normalize_name)
    frame["__bout_key"] = frame["BOUT"].map(_normalize_name)
    frame["__fighter_key"] = frame["FIGHTER"].map(_normalize_name)

    for source_column, target_prefix in _LANDED_ATTEMPTED_PULL_COLUMNS.items():
        landed_col = f"{target_prefix}_landed"
        attempted_col = f"{target_prefix}_attempted"
        if source_column in frame.columns:
            frame[[landed_col, attempted_col]] = frame[source_column].apply(
                lambda value: pd.Series(_parse_landed_attempted(value))
            )
        else:
            frame[landed_col] = np.nan
            frame[attempted_col] = np.nan

    frame["kd"] = frame.get(
        "KD", pd.Series(np.nan, index=frame.index)
    ).apply(_safe_number_or_nan)
    frame["sub_att"] = frame.get(
        "SUB.ATT", pd.Series(np.nan, index=frame.index)
    ).apply(_safe_number_or_nan)
    frame["rev"] = frame.get(
        "REV.", pd.Series(np.nan, index=frame.index)
    ).apply(_safe_number_or_nan)
    frame["ctrl_seconds"] = frame.get(
        "CTRL", pd.Series(np.nan, index=frame.index)
    ).apply(_safe_clock_seconds)

    aggregate_columns = [
        "kd", "sig_str_landed", "sig_str_attempted", "total_str_landed",
        "total_str_attempted", "td_landed", "td_attempted", "sub_att", "rev",
        "ctrl_seconds", "head_landed", "head_attempted", "body_landed",
        "body_attempted", "leg_landed", "leg_attempted", "distance_landed",
        "distance_attempted", "clinch_landed", "clinch_attempted",
        "ground_landed", "ground_attempted",
    ]
    grouped = (
        frame.groupby(
            ["__event_key", "__bout_key", "__fighter_key"], dropna=False
        )[aggregate_columns]
        .sum(min_count=1)
        .reset_index()
    )
    ctrl_unavailable = (
        frame.groupby(
            ["__event_key", "__bout_key", "__fighter_key"], dropna=False
        )["ctrl_seconds"]
        .apply(lambda values: bool(values.isna().any()))
        .rename("__ctrl_unavailable")
        .reset_index()
    )
    grouped = grouped.merge(
        ctrl_unavailable,
        on=["__event_key", "__bout_key", "__fighter_key"],
        how="left",
        validate="one_to_one",
    )

    lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    for _, row in grouped.iterrows():
        key = (row["__event_key"], row["__bout_key"], row["__fighter_key"])
        lookup[key] = {
            "kd": float(row["kd"]),
            "sig_str_landed": float(row["sig_str_landed"]),
            "sig_str_attempted": float(row["sig_str_attempted"]),
            "total_str_landed": float(row["total_str_landed"]),
            "total_str_attempted": float(row["total_str_attempted"]),
            "td_landed": float(row["td_landed"]),
            "td_attempted": float(row["td_attempted"]),
            "sub_att": float(row["sub_att"]),
            "rev": float(row["rev"]),
            "ctrl": (
                None
                if bool(row["__ctrl_unavailable"])
                else _seconds_to_clock(row["ctrl_seconds"])
            ),
            "head_landed": float(row["head_landed"]),
            "head_attempted": float(row["head_attempted"]),
            "body_landed": float(row["body_landed"]),
            "body_attempted": float(row["body_attempted"]),
            "leg_landed": float(row["leg_landed"]),
            "leg_attempted": float(row["leg_attempted"]),
            "distance_landed": float(row["distance_landed"]),
            "distance_attempted": float(row["distance_attempted"]),
            "clinch_landed": float(row["clinch_landed"]),
            "clinch_attempted": float(row["clinch_attempted"]),
            "ground_landed": float(row["ground_landed"]),
            "ground_attempted": float(row["ground_attempted"]),
        }
    return lookup


def _prefix_stats(stats_row: dict[str, object], *, prefix: str) -> dict[str, object]:
    if not stats_row:
        return {}
    return {
        f"{prefix}kd": stats_row.get("kd"),
        f"{prefix}sig_str_landed": stats_row.get("sig_str_landed"),
        f"{prefix}sig_str_attempted": stats_row.get("sig_str_attempted"),
        f"{prefix}total_str_landed": stats_row.get("total_str_landed"),
        f"{prefix}total_str_attempted": stats_row.get("total_str_attempted"),
        f"{prefix}td_landed": stats_row.get("td_landed"),
        f"{prefix}td_attempted": stats_row.get("td_attempted"),
        f"{prefix}sub_att": stats_row.get("sub_att"),
        f"{prefix}rev": stats_row.get("rev"),
        f"{prefix}ctrl": stats_row.get("ctrl"),
        f"{prefix}head_landed": stats_row.get("head_landed"),
        f"{prefix}head_attempted": stats_row.get("head_attempted"),
        f"{prefix}body_landed": stats_row.get("body_landed"),
        f"{prefix}body_attempted": stats_row.get("body_attempted"),
        f"{prefix}leg_landed": stats_row.get("leg_landed"),
        f"{prefix}leg_attempted": stats_row.get("leg_attempted"),
        f"{prefix}distance_landed": stats_row.get("distance_landed"),
        f"{prefix}distance_attempted": stats_row.get("distance_attempted"),
        f"{prefix}clinch_landed": stats_row.get("clinch_landed"),
        f"{prefix}clinch_attempted": stats_row.get("clinch_attempted"),
        f"{prefix}ground_landed": stats_row.get("ground_landed"),
        f"{prefix}ground_attempted": stats_row.get("ground_attempted"),
    }


def _clean_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def _split_bout_fighters(value) -> tuple[str | None, str | None]:
    text = _clean_text(value)
    if not text:
        return None, None
    parts = _BOUT_SPLIT_RE.split(text, maxsplit=1)
    if len(parts) != 2:
        return None, None
    fighter_a, fighter_b = parts[0].strip(), parts[1].strip()
    if not fighter_a or not fighter_b:
        return None, None
    return fighter_a, fighter_b


def _winner_from_outcome(value, *, fighter_a: str, fighter_b: str) -> str | None:
    outcome = _clean_text(value).upper()
    parts = [part.strip() for part in outcome.split("/") if part.strip()]
    if parts == ["W", "L"]:
        return fighter_a
    if parts == ["L", "W"]:
        return fighter_b
    return None


def _parse_landed_attempted(value) -> tuple[float, float]:
    text = _clean_text(value)
    match = _LANDED_ATTEMPTED_RE.search(text)
    if match:
        return float(match.group("landed")), float(match.group("attempted"))
    numeric = _safe_float(text)
    if numeric is None or np.isnan(numeric):
        return np.nan, np.nan
    return float(numeric), float(numeric)


def _safe_number_or_nan(value) -> float:
    numeric = _safe_float(value)
    if numeric is None or np.isnan(numeric):
        return np.nan
    return float(numeric)


def _safe_clock_seconds(value) -> float:
    seconds = _clock_to_seconds(value)
    return float(seconds) if seconds is not None and not pd.isna(seconds) else np.nan


def _seconds_to_clock(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    total_seconds = max(int(round(float(value))), 0)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"
