"""Helpers for building a refreshed UFC training dataset from UFCStats scrapes."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import RAW_DATA_DIR, UFCSTATS_BASE_URL
from src.data.kaggle_loader import (
    _parse_height_cm,
    _parse_reach_cm,
    _safe_float,
    load_kaggle_dataset,
)

logger = logging.getLogger(__name__)

AUTO_REFRESH_DATASET_PATH = RAW_DATA_DIR / "ufc-auto-refresh.csv"
SCRAPED_FIGHTS_PATH = RAW_DATA_DIR / "ufc_fights_scraped.csv"
SCRAPED_FIGHTERS_PATH = RAW_DATA_DIR / "ufc_fighters_scraped.csv"
PULLED_FIGHT_RESULTS_PATH = RAW_DATA_DIR / "ufc-fight-results.csv"
PULLED_FIGHT_STATS_PATH = RAW_DATA_DIR / "ufc-fight-stats.csv"
EVENT_DATES_CACHE_PATH = RAW_DATA_DIR / "ufc-event-dates.csv"
_ROUND_SECONDS = 5 * 60
_BOUT_SPLIT_RE = re.compile(r"\s+vs\.?\s+", flags=re.IGNORECASE)
_LANDED_ATTEMPTED_RE = re.compile(r"(?P<landed>\d+)\s+of\s+(?P<attempted>\d+)", flags=re.IGNORECASE)
_UFCSTATS_HEADERS = {"User-Agent": "Mozilla/5.0"}
DEFAULT_VARIANT_OVERLAP_START = pd.Timestamp("2022-01-01")
DEFAULT_VARIANT_OVERLAP_END = pd.Timestamp("2024-12-31")
TRAINING_DATASET_VARIANTS = (
    "legacy_only",
    "append_only_2026",
    "pulled_all",
    "hybrid_legacy_first",
    "hybrid_pulled_2022plus",
    "best_of_both_field_level",
    "best_of_both_full_history",
)
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
    "event_name",
    "weight_class",
    "winner",
    "method",
    "title_bout",
    "num_rounds",
    "finish_round",
    "finish_time",
    "total_fight_time_secs",
    *[f"a_{field}" for field in _PULLED_COUNT_STAT_FIELDS + _PULLED_RATE_STAT_FIELDS],
    *[f"b_{field}" for field in _PULLED_COUNT_STAT_FIELDS + _PULLED_RATE_STAT_FIELDS],
}
_LEGACY_PREFERRED_FIELDS = {
    "location",
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
    "empty_arena",
}


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

    fighter_lookup = _load_scraped_fighter_lookup(scraped_fighters_path)
    scraped_training_df = _scraped_fights_to_training_rows(scraped_raw, fighter_lookup=fighter_lookup)
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
    fight_results_path: Path | None = None,
    fight_stats_path: Path | None = None,
    event_dates_cache_path: Path | None = None,
    event_date_lookup: dict[str, object] | None = None,
    force_refresh_event_dates: bool = False,
) -> pd.DataFrame:
    """
    Convert the repo's pulled UFCStats results/stats CSVs into the normalized training schema.

    This is useful for source experiments where we want to compare the legacy
    `ufc-master.csv` dataset against newer locally pulled UFCStats rows without
    requiring a full network scrape of every fight page.
    """
    fight_results_path = Path(fight_results_path) if fight_results_path is not None else PULLED_FIGHT_RESULTS_PATH
    fight_stats_path = Path(fight_stats_path) if fight_stats_path is not None else PULLED_FIGHT_STATS_PATH
    event_dates_cache_path = (
        Path(event_dates_cache_path) if event_dates_cache_path is not None else EVENT_DATES_CACHE_PATH
    )

    if not fight_results_path.exists():
        raise FileNotFoundError(f"pulled fight results file not found: {fight_results_path}")
    if not fight_stats_path.exists():
        raise FileNotFoundError(f"pulled fight stats file not found: {fight_stats_path}")

    results_df = pd.read_csv(fight_results_path)
    stats_df = pd.read_csv(fight_stats_path)
    if results_df.empty:
        return pd.DataFrame()

    if event_date_lookup is None:
        event_date_lookup = load_or_fetch_event_date_lookup(
            required_events=results_df.get("EVENT"),
            cache_path=event_dates_cache_path,
            force_refresh=force_refresh_event_dates,
        )

    scraped_like_df = _pulled_data_to_scraped_rows(
        results_df,
        stats_df,
        event_date_lookup=event_date_lookup,
    )
    if scraped_like_df.empty:
        return pd.DataFrame()

    training_df = _scraped_fights_to_training_rows(scraped_like_df, fighter_lookup={})
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
    fight_results_path: Path | None = None,
    fight_stats_path: Path | None = None,
    event_dates_cache_path: Path | None = None,
    event_date_lookup: dict[str, object] | None = None,
    output_path: Path | None = None,
    force_refresh_event_dates: bool = False,
) -> dict:
    """Build a merged training dataset using legacy rows plus the local pulled UFCStats rows."""
    output_path = resolve_auto_refresh_dataset_path(output_path)
    pulled_training_df = build_training_rows_from_pulled_data(
        fight_results_path=fight_results_path,
        fight_stats_path=fight_stats_path,
        event_dates_cache_path=event_dates_cache_path,
        event_date_lookup=event_date_lookup,
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
    if base_dataset_path is not None and Path(base_dataset_path).exists():
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


def build_training_dataset_variants(
    *,
    legacy_df: pd.DataFrame | None = None,
    pulled_df: pd.DataFrame | None = None,
    base_dataset_path: Path | None = None,
    fight_results_path: Path | None = None,
    fight_stats_path: Path | None = None,
    event_dates_cache_path: Path | None = None,
    event_date_lookup: dict[str, object] | None = None,
    force_refresh_event_dates: bool = False,
    overlap_start: str | pd.Timestamp = DEFAULT_VARIANT_OVERLAP_START,
) -> dict[str, pd.DataFrame]:
    """Build the full UFC source-variant matrix used for offline experiments."""
    if legacy_df is None:
        dataset_path = Path(base_dataset_path) if base_dataset_path is not None else RAW_DATA_DIR / "ufc-master.csv"
        legacy_df = load_kaggle_dataset(dataset_path)
    if pulled_df is None:
        pulled_df = build_training_rows_from_pulled_data(
            fight_results_path=fight_results_path,
            fight_stats_path=fight_stats_path,
            event_dates_cache_path=event_dates_cache_path,
            event_date_lookup=event_date_lookup,
            force_refresh_event_dates=force_refresh_event_dates,
        )

    legacy_df = _ensure_unique_fight_rows(legacy_df)
    pulled_df = _ensure_unique_fight_rows(pulled_df)
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
    for field in pulled_row.index:
        if field == "fight_key":
            continue
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
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _scraped_fights_to_training_rows(
    scraped_df: pd.DataFrame,
    *,
    fighter_lookup: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Convert raw scrape output into the normalized training schema."""
    fighter_lookup = fighter_lookup or {}
    rows: list[dict] = []

    for _, row in scraped_df.iterrows():
        event_date = pd.to_datetime(row.get("event_date"), errors="coerce", format="mixed")
        fighter_a = row.get("fighter_a")
        fighter_b = row.get("fighter_b")
        if pd.isna(event_date) or pd.isna(fighter_a) or pd.isna(fighter_b):
            continue

        weight_class, title_bout = _parse_weight_class(row.get("weight_class"))
        finish_round = _safe_int(row.get("round"))
        finish_time = row.get("time")
        total_fight_time_secs = _round_time_to_total_seconds(finish_round, finish_time)
        fight_minutes = (total_fight_time_secs / 60.0) if total_fight_time_secs and total_fight_time_secs > 0 else np.nan
        scheduled_rounds = _parse_scheduled_rounds(row.get("time_format"))

        training_row = {
            "event_date": event_date,
            "event_name": row.get("event_title"),
            "weight_class": weight_class,
            "winner": row.get("winner"),
            "method": row.get("method"),
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "target": 1 if row.get("winner") == fighter_a else (0 if row.get("winner") == fighter_b else np.nan),
            "title_bout": int(title_bout),
            "num_rounds": _infer_num_rounds(
                title_bout=title_bout,
                finish_round=finish_round,
                method=row.get("method"),
                scheduled_rounds=scheduled_rounds,
            ),
            "finish_round": finish_round,
            "finish_time": finish_time,
            "total_fight_time_secs": total_fight_time_secs,
        }
        training_row.update(_extract_rate_features(row, fight_minutes=fight_minutes, prefix="a_", opp_prefix="b_"))
        training_row.update(_extract_rate_features(row, fight_minutes=fight_minutes, prefix="b_", opp_prefix="a_"))
        training_row.update(_extract_fighter_attributes(fighter_a, event_date, fighter_lookup, prefix="a_"))
        training_row.update(_extract_fighter_attributes(fighter_b, event_date, fighter_lookup, prefix="b_"))
        rows.append(training_row)

    return pd.DataFrame(rows)


def _parse_weight_class(value) -> tuple[str | None, bool]:
    raw = "" if value is None or pd.isna(value) else str(value).strip()
    if not raw:
        return None, False
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
    title_bout: bool,
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


def _load_scraped_fighter_lookup(scraped_fighters_path: Path) -> dict[str, dict]:
    """Load optional fighter profile data for static attributes like height and reach."""
    if not scraped_fighters_path.exists():
        return {}

    fighters_df = pd.read_csv(scraped_fighters_path)
    if fighters_df.empty or "name" not in fighters_df.columns:
        return {}

    lookup: dict[str, dict] = {}
    for _, row in fighters_df.iterrows():
        name_key = _normalize_name(row.get("name"))
        if not name_key:
            continue
        dob = pd.to_datetime(row.get("dob"), errors="coerce", format="mixed")
        lookup[name_key] = {
            "height": _parse_height_cm(row.get("height")),
            "reach": _parse_reach_cm(row.get("reach")),
            "weight": _parse_weight_lbs(row.get("weight")),
            "stance": row.get("stance"),
            "dob": dob if pd.notna(dob) else None,
        }
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
) -> dict:
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
        f"{prefix}weight": profile.get("weight"),
        f"{prefix}stance": profile.get("stance"),
        f"{prefix}age": age,
    }


def load_or_fetch_event_date_lookup(
    *,
    required_events=None,
    cache_path: Path | None = None,
    force_refresh: bool = False,
) -> dict[str, pd.Timestamp]:
    """Load cached UFC event dates and backfill missing entries from the UFCStats event listing."""
    cache_path = Path(cache_path) if cache_path is not None else EVENT_DATES_CACHE_PATH
    cached = _load_event_date_cache(cache_path)
    required_iterable = [] if required_events is None else list(required_events)
    required_keys = {_normalize_name(value) for value in required_iterable if _normalize_name(value)}
    missing_keys = required_keys - set(cached)

    if force_refresh or missing_keys or not cached:
        fetched = _fetch_ufcstats_event_dates()
        if fetched:
            cached.update(fetched)
            _save_event_date_cache(cache_path, cached)
            missing_keys = required_keys - set(cached)

    if missing_keys:
        logger.warning("Missing UFC event-date mappings for %d events", len(missing_keys))
    return cached


def _load_event_date_cache(path: Path) -> dict[str, pd.Timestamp]:
    if not path.exists():
        return {}
    cached_df = pd.read_csv(path)
    if cached_df.empty or "event_name" not in cached_df.columns or "event_date" not in cached_df.columns:
        return {}

    lookup: dict[str, pd.Timestamp] = {}
    for _, row in cached_df.iterrows():
        key = _normalize_name(row.get("event_name"))
        event_date = pd.to_datetime(row.get("event_date"), errors="coerce", format="mixed")
        if key and pd.notna(event_date):
            lookup[key] = event_date
    return lookup


def _save_event_date_cache(path: Path, lookup: dict[str, pd.Timestamp]) -> None:
    rows = [
        {"event_name": event_name, "event_date": event_date.date().isoformat()}
        for event_name, event_date in sorted(lookup.items())
        if pd.notna(event_date)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _fetch_ufcstats_event_dates() -> dict[str, pd.Timestamp]:
    """Fetch event title/date pairs from the UFCStats completed-events listing."""
    page = 1
    lookup: dict[str, pd.Timestamp] = {}
    while True:
        resp = requests.get(f"{UFCSTATS_BASE_URL}?page={page}", headers=_UFCSTATS_HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        rows = soup.select("tr.b-statistics__table-row")
        parsed_rows = 0
        for row in rows:
            link = row.select_one("a.b-link")
            date_el = row.select_one("span.b-statistics__date")
            if link is None or date_el is None:
                continue
            event_name = _clean_text(link.get_text(" ", strip=True))
            event_date_text = _clean_text(date_el.get_text(" ", strip=True))
            event_date = pd.to_datetime(event_date_text, errors="coerce", format="mixed")
            if not event_name or pd.isna(event_date):
                continue
            lookup[_normalize_name(event_name)] = event_date
            parsed_rows += 1
        if parsed_rows == 0:
            break
        page += 1

    logger.info("Fetched %d UFC event dates from UFCStats", len(lookup))
    return lookup


def _pulled_data_to_scraped_rows(
    results_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    *,
    event_date_lookup: dict[str, object],
) -> pd.DataFrame:
    totals_lookup = _aggregate_pulled_stats(stats_df)
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
        event_date = pd.to_datetime(event_date_lookup.get(event_key), errors="coerce", format="mixed")
        if pd.isna(event_date):
            skipped_no_date += 1
            continue

        fighter_a_stats = totals_lookup.get((event_key, _normalize_name(str(row.get("BOUT"))), _normalize_name(fighter_a)), {})
        fighter_b_stats = totals_lookup.get((event_key, _normalize_name(str(row.get("BOUT"))), _normalize_name(fighter_b)), {})
        winner = _winner_from_outcome(row.get("OUTCOME"), fighter_a=fighter_a, fighter_b=fighter_b)

        converted = {
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "winner": winner,
            "method": row.get("METHOD"),
            "round": row.get("ROUND"),
            "time": row.get("TIME"),
            "time_format": row.get("TIME FORMAT"),
            "weight_class": row.get("WEIGHTCLASS"),
            "fight_url": row.get("URL"),
            "event_title": event_title,
            "event_date": event_date,
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
            frame[landed_col] = 0.0
            frame[attempted_col] = 0.0

    frame["kd"] = frame["KD"].apply(_safe_number_or_zero)
    frame["sub_att"] = frame["SUB.ATT"].apply(_safe_number_or_zero)
    frame["rev"] = frame["REV."].apply(_safe_number_or_zero)
    frame["ctrl_seconds"] = frame["CTRL"].apply(_safe_clock_seconds)

    grouped = (
        frame.groupby(["__event_key", "__bout_key", "__fighter_key"], dropna=False)
        .agg(
            kd=("kd", "sum"),
            sig_str_landed=("sig_str_landed", "sum"),
            sig_str_attempted=("sig_str_attempted", "sum"),
            total_str_landed=("total_str_landed", "sum"),
            total_str_attempted=("total_str_attempted", "sum"),
            td_landed=("td_landed", "sum"),
            td_attempted=("td_attempted", "sum"),
            sub_att=("sub_att", "sum"),
            rev=("rev", "sum"),
            ctrl_seconds=("ctrl_seconds", "sum"),
            head_landed=("head_landed", "sum"),
            head_attempted=("head_attempted", "sum"),
            body_landed=("body_landed", "sum"),
            body_attempted=("body_attempted", "sum"),
            leg_landed=("leg_landed", "sum"),
            leg_attempted=("leg_attempted", "sum"),
            distance_landed=("distance_landed", "sum"),
            distance_attempted=("distance_attempted", "sum"),
            clinch_landed=("clinch_landed", "sum"),
            clinch_attempted=("clinch_attempted", "sum"),
            ground_landed=("ground_landed", "sum"),
            ground_attempted=("ground_attempted", "sum"),
        )
        .reset_index()
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
            "ctrl": _seconds_to_clock(row["ctrl_seconds"]),
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
        return 0.0, 0.0
    return float(numeric), float(numeric)


def _safe_number_or_zero(value) -> float:
    numeric = _safe_float(value)
    if numeric is None or np.isnan(numeric):
        return 0.0
    return float(numeric)


def _safe_clock_seconds(value) -> float:
    seconds = _clock_to_seconds(value)
    return float(seconds) if seconds is not None and not pd.isna(seconds) else 0.0


def _seconds_to_clock(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    total_seconds = max(int(round(float(value))), 0)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"
