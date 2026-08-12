"""
Historical odds backfill — fetches opening, midweek, and closing odds
from The Odds API for all fights in the test set (and optionally training set).

This enables:
  1. Realistic backtesting against the odds available at bet placement time
  2. Line movement features for model training
  3. Closing Line Value (CLV) as a validation metric
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.line_movement import (
    LINE_MOVEMENT_FEATURE_COLS,
    analysis_to_line_movement_features,
    compute_line_movement_analysis,
)
from src.data.name_utils import normalize_person_name, same_person_name
from src.data.odds_client import OddsClient
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

BACKFILL_DIR = RAW_DATA_DIR / "historical_odds"
BACKFILL_DIR.mkdir(parents=True, exist_ok=True)

# How many days before the event to snapshot odds
SNAPSHOT_OFFSETS = [7, 3, 1]  # 7 days out (opening), 3 days (midweek), 1 day (closing)

# Rate limiting: The Odds API has per-second limits
REQUEST_DELAY = 1.5  # seconds between requests
BACKFILL_KEY_COLUMNS = ["event_date", "fighter_a", "fighter_b", "offset_days"]
MAX_EQUAL_ODDS_OVERROUND = 1.20

_LINE_HISTORY_CACHE: tuple[tuple, pd.DataFrame] | None = None
_ALL_ODDS_CACHE: tuple[tuple, pd.DataFrame] | None = None
_ROLE_CACHE: dict[tuple, pd.DataFrame] = {}


def _path_set_signature(paths: list[Path]) -> tuple:
    """Cheap in-process cache key that changes when any source file changes."""
    records = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        records.append((path.name, stat.st_size, stat.st_mtime_ns))
    return tuple(records)


def _normalize_backfill_event_date(value) -> str:
    if pd.isna(value):
        return ""
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value).strip()


def _normalize_backfill_name(value) -> str:
    return normalize_person_name(value)


def _backfill_pair_key(fighter_a, fighter_b) -> str:
    return "|".join(sorted([_normalize_backfill_name(fighter_a), _normalize_backfill_name(fighter_b)]))


def _canonical_backfill_key(
    event_date,
    fighter_a,
    fighter_b,
    offset_days,
) -> tuple[str, str, str, int]:
    return (
        _normalize_backfill_event_date(event_date),
        _normalize_backfill_name(fighter_a),
        _normalize_backfill_name(fighter_b),
        int(offset_days),
    )


def _with_canonical_backfill_keys(df: pd.DataFrame) -> pd.DataFrame:
    keyed = df.copy()
    keyed["_key_event_date"] = keyed["event_date"].apply(_normalize_backfill_event_date)
    keyed["_key_fighter_a"] = keyed["fighter_a"].apply(_normalize_backfill_name)
    keyed["_key_fighter_b"] = keyed["fighter_b"].apply(_normalize_backfill_name)
    keyed["_key_offset_days"] = keyed["offset_days"].astype(int)
    return keyed


def _match_fighters(api_event: dict, fighter_a: str, fighter_b: str) -> bool:
    """Check if an API event matches a fight by fighter names."""
    home = api_event.get("home_team", "")
    away = api_event.get("away_team", "")
    return (
        same_person_name(fighter_a, home) and same_person_name(fighter_b, away)
    ) or (
        same_person_name(fighter_a, away) and same_person_name(fighter_b, home)
    )


def _has_implausible_equal_favorite_odds(a_decimal: float, b_decimal: float) -> bool:
    """Detect duplicated heavy-favorite prices without rejecting normal pick'em vig."""
    if a_decimal <= 1.0 or b_decimal <= 1.0:
        return False
    if not np.isclose(a_decimal, b_decimal, rtol=0.0, atol=1e-12):
        return False
    if a_decimal >= 2.0:
        return False
    return (1.0 / a_decimal) + (1.0 / b_decimal) > MAX_EQUAL_ODDS_OVERROUND


def _extract_fight_odds(api_events: list[dict], fighter_a: str, fighter_b: str) -> dict | None:
    """Extract odds for a specific fight from API response."""
    for event in api_events:
        if not _match_fighters(event, fighter_a, fighter_b):
            continue

        home = event.get("home_team", "")
        away = event.get("away_team", "")
        home_is_a = same_person_name(fighter_a, home)

        all_bookmaker_odds = []
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                home_odds = outcomes.get(home)
                away_odds = outcomes.get(away)
                if home_odds and away_odds:
                    a_odds = home_odds if home_is_a else away_odds
                    b_odds = away_odds if home_is_a else home_odds
                    if a_odds <= 1.0 or b_odds <= 1.0:
                        continue
                    if _has_implausible_equal_favorite_odds(a_odds, b_odds):
                        continue
                    a_imp = 1.0 / a_odds
                    b_imp = 1.0 / b_odds
                    total = a_imp + b_imp
                    all_bookmaker_odds.append({
                        "bookmaker": bookmaker.get("title", ""),
                        "a_decimal_odds": a_odds,
                        "b_decimal_odds": b_odds,
                        "a_implied_prob": a_imp,
                        "b_implied_prob": b_imp,
                        "a_fair_prob": a_imp / total,
                        "b_fair_prob": b_imp / total,
                    })

        if not all_bookmaker_odds:
            return None

        # Consensus across bookmakers
        df = pd.DataFrame(all_bookmaker_odds)
        return {
            "a_fair_prob": df["a_fair_prob"].mean(),
            "b_fair_prob": df["b_fair_prob"].mean(),
            "a_decimal_odds": df["a_decimal_odds"].mean(),
            "b_decimal_odds": df["b_decimal_odds"].mean(),
            "num_bookmakers": len(df),
        }

    return None


def backfill_event_date(
    client: OddsClient,
    event_date: str,
    fights: pd.DataFrame,
    offset_days: int,
) -> list[dict]:
    """
    Fetch historical odds for fights on a given event date at a specific offset.

    Args:
        client: OddsClient instance
        event_date: The fight date (e.g., "2023-06-10")
        fights: DataFrame of fights on this date with fighter_a, fighter_b columns
        offset_days: How many days before the event to query (7=opening, 1=closing)

    Returns list of dicts with odds data per fight.
    """
    query_date = pd.Timestamp(event_date) - timedelta(days=offset_days)
    query_iso = query_date.strftime("%Y-%m-%dT12:00:00Z")

    try:
        api_data = client.get_historical_odds(query_iso)
    except Exception as e:
        logger.warning(f"Failed to fetch odds for {event_date} (offset -{offset_days}d): {e}")
        return []

    time.sleep(REQUEST_DELAY)

    results = []
    for _, fight in fights.iterrows():
        fa = fight["fighter_a"]
        fb = fight["fighter_b"]

        odds = _extract_fight_odds(api_data, fa, fb)
        if odds:
            results.append({
                "event_date": event_date,
                "fighter_a": fa,
                "fighter_b": fb,
                "query_date": query_date.strftime("%Y-%m-%d"),
                "offset_days": offset_days,
                **odds,
            })

    return results


def _existing_backfill_keys(df: pd.DataFrame) -> set[tuple[str, str, str, int]]:
    if df.empty:
        return set()

    key_frame = df[BACKFILL_KEY_COLUMNS].copy()
    return {
        _canonical_backfill_key(
            row.event_date,
            row.fighter_a,
            row.fighter_b,
            row.offset_days,
        )
        for row in key_frame.itertuples(index=False)
    }


def run_backfill(
    fights_df: Optional[pd.DataFrame] = None,
    offsets: Optional[list[int]] = None,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Run full historical odds backfill for all fights.

    Args:
        fights_df: DataFrame with event_date, fighter_a, fighter_b.
                   If None, loads from test_set.csv.
        offsets: List of day offsets to query (default: [7, 3, 1])
        resume: If True, skip only fight/offset keys already backfilled.

    Returns DataFrame with all historical odds data.
    """
    if offsets is None:
        offsets = SNAPSHOT_OFFSETS

    if fights_df is None:
        test_path = PROCESSED_DATA_DIR / "test_set.csv"
        if not test_path.exists():
            raise FileNotFoundError("Test set not found. Run 'train' first.")
        fights_df = pd.read_csv(test_path, parse_dates=["event_date"])

    client = OddsClient()
    output_path = BACKFILL_DIR / "historical_odds.csv"

    # Resume support: load existing data
    existing = pd.DataFrame()
    if resume and output_path.exists():
        existing = pd.read_csv(output_path)
        logger.info(f"Resuming backfill: {len(existing)} records already collected")
    existing_keys = _existing_backfill_keys(existing)

    # Group fights by event date
    event_dates = sorted(fights_df["event_date"].unique())
    all_results = []

    total_dates = len(event_dates)
    api_calls = 0

    for i, event_date in enumerate(event_dates):
        event_str = pd.Timestamp(event_date).strftime("%Y-%m-%d")
        event_fights = fights_df[fights_df["event_date"] == event_date]

        for offset in offsets:
            fights_to_fetch = event_fights
            if existing_keys:
                missing_rows = []
                for _, fight in event_fights.iterrows():
                    key = _canonical_backfill_key(
                        event_str,
                        fight["fighter_a"],
                        fight["fighter_b"],
                        offset,
                    )
                    if key not in existing_keys:
                        missing_rows.append(fight.to_dict())
                if not missing_rows:
                    continue
                fights_to_fetch = pd.DataFrame(missing_rows)

            results = backfill_event_date(client, event_str, fights_to_fetch, offset)
            all_results.extend(results)
            for row in results:
                existing_keys.add(
                    _canonical_backfill_key(
                        row["event_date"],
                        row["fighter_a"],
                        row["fighter_b"],
                        row["offset_days"],
                    )
                )
            api_calls += 1

            # Periodic save (every 10 API calls)
            if api_calls % 10 == 0:
                _save_progress(existing, all_results, output_path)
                logger.info(
                    f"Progress: {i+1}/{total_dates} event dates, "
                    f"{api_calls} API calls, "
                    f"{len(all_results)} new records"
                )

    # Final save
    final_df = _save_progress(existing, all_results, output_path)

    # Summary
    if not final_df.empty:
        fights_with_odds = final_df.groupby(["event_date", "fighter_a", "fighter_b"]).size()
        logger.info(f"\nBackfill complete!")
        logger.info(f"  Total records: {len(final_df)}")
        logger.info(f"  Fights with odds: {len(fights_with_odds)}")
        logger.info(f"  API calls made: {api_calls}")

        # Coverage by offset
        for offset in offsets:
            count = len(final_df[final_df["offset_days"] == offset])
            logger.info(f"  Offset -{offset}d: {count} fight records")
    else:
        logger.warning("No historical odds data collected.")

    return final_df


def _save_progress(existing: pd.DataFrame, new_results: list[dict], path: Path) -> pd.DataFrame:
    """Save combined existing + new results to disk."""
    new_df = pd.DataFrame(new_results)
    if existing.empty and new_df.empty:
        return pd.DataFrame()

    parts = [df for df in [existing, new_df] if not df.empty]
    combined = pd.concat(parts, ignore_index=True)
    combined = _with_canonical_backfill_keys(combined)
    combined = combined.drop_duplicates(
        subset=["_key_event_date", "_key_fighter_a", "_key_fighter_b", "_key_offset_days"],
        keep="last",
    )
    combined = combined.drop(
        columns=["_key_event_date", "_key_fighter_a", "_key_fighter_b", "_key_offset_days"],
    )
    combined.to_csv(path, index=False)
    return combined


def load_historical_odds() -> pd.DataFrame:
    """Load the backfilled historical odds data from the primary file only.

    DEPRECATED: Use load_all_historical_odds() instead, which consolidates
    every odds source (primary, pre-2022, BFO scrapes) into one deduplicated
    frame.  This function exists for backward compatibility.
    """
    path = BACKFILL_DIR / "historical_odds.csv"
    if not path.exists():
        logger.warning("No historical odds data found. Run backfill first.")
        return pd.DataFrame()
    return pd.read_csv(path)


# ── Canonical columns every odds source must provide ─────────────────
_CANONICAL_ODDS_COLS = [
    "event_date", "fighter_a", "fighter_b", "query_date", "offset_days",
    "a_fair_prob", "b_fair_prob", "a_decimal_odds", "b_decimal_odds",
    "num_bookmakers", "source_kind", "source_file", "observed_at",
    "commence_time", "hours_to_start", "verified_prefight",
]

VERIFIED_PREFIGHT_SOURCE_KINDS = frozenset({"odds_api", "line_history"})


def _stamp_odds_provenance(
    frame: pd.DataFrame,
    *,
    source_kind: str,
    source_file: str,
) -> pd.DataFrame:
    """Attach row-level source/timing evidence without upgrading legacy data.

    Only the Odds API historical endpoint and persisted live line-history
    snapshots can be treated as verified pre-fight observations. BFO rows and
    the old pre-2022 overlay remain useful as closing/reference prices, but
    their historical ``offset_days`` labels are not trusted as provenance.
    """
    result = frame.copy()
    result["source_kind"] = source_kind
    if source_file or "source_file" not in result.columns:
        result["source_file"] = source_file

    if source_kind == "line_history":
        observed = pd.to_datetime(
            result.get("snapshot_time", pd.Series(pd.NaT, index=result.index)),
            errors="coerce",
            utc=True,
        )
        commence = pd.to_datetime(
            result.get("commence_time", pd.Series(pd.NaT, index=result.index)),
            errors="coerce",
            utc=True,
        )
        hours = (commence - observed).dt.total_seconds() / 3600.0
        result["observed_at"] = observed
        result["commence_time"] = commence
        result["hours_to_start"] = hours
        result["verified_prefight"] = observed.notna() & commence.notna() & (hours > 0.0)
        return result

    observed = pd.to_datetime(
        result.get("query_date", pd.Series(pd.NaT, index=result.index)),
        errors="coerce",
        utc=True,
    )
    result["observed_at"] = observed
    if source_kind == "odds_api":
        offsets = pd.to_numeric(
            result.get("offset_days", pd.Series(np.nan, index=result.index)),
            errors="coerce",
        )
        result["hours_to_start"] = offsets * 24.0
        result["verified_prefight"] = offsets.notna() & (offsets > 0.0)
    else:
        # The legacy/BFO query date and offset fields were reconstructed and
        # do not prove when a price was observable.
        result["hours_to_start"] = 0.0
        result["verified_prefight"] = False
    return result


def load_line_history_odds(line_history_dir: Optional[Path] = None) -> pd.DataFrame:
    """Read persisted ``odds_*.csv`` snapshots into the canonical odds shape.

    This is intentionally a small importer rather than a second collector. It
    trusts the snapshot and commence timestamps written by ``line_tracker``,
    rejects post-start/malformed rows, and averages bookmakers per fight and
    snapshot exactly as the historical loader expects.
    """
    if line_history_dir is None:
        from src.data.line_tracker import LINE_HISTORY_DIR

        line_history_dir = LINE_HISTORY_DIR

    global _LINE_HISTORY_CACHE

    paths = sorted(Path(line_history_dir).glob("odds_*.csv"))
    signature = (str(Path(line_history_dir).resolve(strict=False)), _path_set_signature(paths))
    if _LINE_HISTORY_CACHE is not None and _LINE_HISTORY_CACHE[0] == signature:
        return _LINE_HISTORY_CACHE[1].copy()

    parts: list[pd.DataFrame] = []
    for path in paths:
        try:
            raw = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Failed to load line-history snapshot %s: %s", path.name, exc)
            continue
        required = {"fighter_a", "fighter_b", "commence_time", "snapshot_time"}
        if raw.empty or not required.issubset(raw.columns):
            continue
        raw = raw.copy()
        if "a_decimal_odds" not in raw.columns and "a_odds" in raw.columns:
            raw["a_decimal_odds"] = raw["a_odds"]
        if "b_decimal_odds" not in raw.columns and "b_odds" in raw.columns:
            raw["b_decimal_odds"] = raw["b_odds"]
        for side in ("a", "b"):
            fair_col = f"{side}_fair_prob"
            implied_col = f"{side}_implied_prob"
            decimal_col = f"{side}_decimal_odds"
            if fair_col not in raw.columns:
                if implied_col in raw.columns:
                    raw[fair_col] = pd.to_numeric(raw[implied_col], errors="coerce")
                elif decimal_col in raw.columns:
                    raw[fair_col] = 1.0 / pd.to_numeric(raw[decimal_col], errors="coerce")

        a_prob = pd.to_numeric(raw.get("a_fair_prob"), errors="coerce")
        b_prob = pd.to_numeric(raw.get("b_fair_prob"), errors="coerce")
        total = a_prob + b_prob
        valid_total = total.notna() & (total > 0.0)
        raw.loc[valid_total, "a_fair_prob"] = a_prob[valid_total] / total[valid_total]
        raw.loc[valid_total, "b_fair_prob"] = b_prob[valid_total] / total[valid_total]
        raw["source_file"] = path.name
        parts.append(raw)

    if not parts:
        empty = pd.DataFrame(columns=_CANONICAL_ODDS_COLS)
        empty.attrs["odds_snapshot_key"] = signature
        _LINE_HISTORY_CACHE = (signature, empty)
        return empty.copy()

    combined = pd.concat(parts, ignore_index=True)
    combined = _drop_invalid_moneyline_rows(combined)
    combined = _stamp_odds_provenance(
        combined,
        source_kind="line_history",
        source_file="",
    )
    combined = combined[combined["verified_prefight"]].copy()
    # UFC historical tables use the North-American broadcast/card date. Late
    # Saturday fights often commence after midnight UTC, so a raw UTC date
    # would miss the same fight by one day.
    combined["event_date"] = pd.to_datetime(
        combined["commence_time"], errors="coerce", utc=True
    ).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    combined["query_date"] = pd.to_datetime(
        combined["snapshot_time"], errors="coerce", utc=True
    ).dt.strftime("%Y-%m-%d")
    combined["offset_days"] = combined["hours_to_start"] / 24.0

    group_cols = [
        "event_date", "fighter_a", "fighter_b", "snapshot_time",
        "commence_time", "query_date", "offset_days", "source_kind",
        "source_file", "observed_at", "hours_to_start", "verified_prefight",
    ]
    aggregate = {
        "a_fair_prob": "mean",
        "b_fair_prob": "mean",
        "a_decimal_odds": "mean",
        "b_decimal_odds": "mean",
    }
    result = combined.groupby(group_cols, dropna=False, as_index=False).agg(aggregate)
    result["num_bookmakers"] = combined.groupby(group_cols, dropna=False).size().values
    result.attrs["odds_snapshot_key"] = signature
    _LINE_HISTORY_CACHE = (signature, result)
    return result.copy()


def _drop_invalid_moneyline_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose observed decimal odds cannot describe a binary market."""
    if df.empty:
        return df

    result = df.copy()
    for col in ("a_decimal_odds", "b_decimal_odds", "a_fair_prob", "b_fair_prob"):
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    a_dec = result.get("a_decimal_odds", pd.Series(np.nan, index=result.index))
    b_dec = result.get("b_decimal_odds", pd.Series(np.nan, index=result.index))
    both_decimal = a_dec.notna() & b_dec.notna()
    invalid_decimal = both_decimal & ((a_dec <= 1.0) | (b_dec <= 1.0))
    implied_overround = (1.0 / a_dec) + (1.0 / b_dec)
    duplicated_heavy_favorite = (
        both_decimal
        & np.isclose(a_dec, b_dec, rtol=0.0, atol=1e-12)
        & (a_dec < 2.0)
        & (implied_overround > MAX_EQUAL_ODDS_OVERROUND)
    )
    invalid_mask = invalid_decimal | duplicated_heavy_favorite

    if invalid_decimal.any():
        logger.warning(
            "Dropped %d historical odds rows with invalid decimal odds",
            int(invalid_decimal.sum()),
        )
    if duplicated_heavy_favorite.any():
        logger.info(
            "Dropped %d historical odds rows with duplicated heavy-favorite decimal odds",
            int(duplicated_heavy_favorite.sum()),
        )
    return result.loc[~invalid_mask].copy()


def load_all_historical_odds() -> pd.DataFrame:
    """Load and deduplicate *every* historical odds source into one frame.

    Sources (loaded in priority order, last write wins on duplicates):
      1. historical_odds_pre2022_from_cleaned.csv  (pre-2022 legacy)
      2. historical_odds_bfo.csv                   (BFO scrapes)
      3. historical_odds_bfo_recovered_*.csv        (BFO recovery batches)
      4. historical_odds.csv                        (The Odds API backfill)
      5. line_history/odds_*.csv                    (timestamped live snapshots)

    Rows carry source and observation-time evidence. Exact duplicate
    observations are resolved in the order above; role selection later uses
    only verified Odds API/line-history rows for model inputs and fills.

    Returns a DataFrame with at least the canonical columns:
        event_date, fighter_a, fighter_b, query_date, offset_days,
        a_fair_prob, b_fair_prob, a_decimal_odds, b_decimal_odds,
        num_bookmakers
    """
    global _ALL_ODDS_CACHE

    pre2022_path = BACKFILL_DIR / "historical_odds_pre2022_from_cleaned.csv"
    bfo_path = BACKFILL_DIR / "historical_odds_bfo.csv"
    recovery_paths = sorted(BACKFILL_DIR.glob("historical_odds_bfo_recovered_*.csv"))
    primary_path = BACKFILL_DIR / "historical_odds.csv"
    source_paths = [pre2022_path, bfo_path, *recovery_paths, primary_path]
    # Keep the companion archive scoped to the same raw-data root. Tests and
    # offline rebuilds monkeypatch BACKFILL_DIR to an isolated snapshot; using
    # the module-global line tracker path here would leak unrelated live rows.
    line_history = load_line_history_odds(BACKFILL_DIR.parent / "line_history")
    cache_key = (
        str(BACKFILL_DIR.resolve(strict=False)),
        _path_set_signature(source_paths),
        line_history.attrs.get("odds_snapshot_key"),
    )
    if _ALL_ODDS_CACHE is not None and _ALL_ODDS_CACHE[0] == cache_key:
        return _ALL_ODDS_CACHE[1].copy()

    parts: list[pd.DataFrame] = []

    def append_source(path: Path, source_kind: str) -> None:
        try:
            loaded = pd.read_csv(path)
        except Exception as exc:
            logger.warning("Failed to load %s odds %s: %s", source_kind, path.name, exc)
            return
        parts.append(
            _stamp_odds_provenance(
                loaded,
                source_kind=source_kind,
                source_file=path.name,
            )
        )

    # 1. Pre-2022 legacy
    if pre2022_path.exists():
        append_source(pre2022_path, "legacy_pre2022")

    # 2. BFO scrapes (named exactly historical_odds_bfo.csv)
    if bfo_path.exists():
        append_source(bfo_path, "bfo")

    # 3. BFO recovery batches (any file matching historical_odds_bfo_recovered_*.csv)
    for recovery_path in recovery_paths:
        append_source(recovery_path, "bfo")

    # 4. Primary Odds API backfill (highest priority)
    if primary_path.exists():
        append_source(primary_path, "odds_api")

    # Runtime line history is the most precise source because it records both
    # observation and scheduled start timestamps. Its absence is normal in
    # development/CI and does not make the historical loader unusable.
    if not line_history.empty:
        parts.append(line_history)

    if not parts:
        logger.warning(
            "No historical odds files found in %s. Run backfill first.", BACKFILL_DIR
        )
        empty = pd.DataFrame(columns=_CANONICAL_ODDS_COLS)
        empty.attrs["odds_snapshot_key"] = cache_key
        _ALL_ODDS_CACHE = (cache_key, empty)
        return empty.copy()

    combined = pd.concat(parts, ignore_index=True)

    # Keep only canonical columns + any extras that happen to exist
    for col in _CANONICAL_ODDS_COLS:
        if col not in combined.columns:
            combined[col] = np.nan
    combined = _drop_invalid_moneyline_rows(combined)

    # Deduplicate on normalised keys (last occurrence wins = highest-priority source)
    combined["_key_date"] = combined["event_date"].apply(_normalize_backfill_event_date)
    combined["_key_a"] = combined["fighter_a"].apply(_normalize_backfill_name)
    combined["_key_b"] = combined["fighter_b"].apply(_normalize_backfill_name)
    combined["_key_observed"] = pd.to_datetime(
        combined["observed_at"], errors="coerce", utc=True
    ).astype(str)
    combined = combined.drop_duplicates(
        subset=["_key_date", "_key_a", "_key_b", "_key_observed"], keep="last"
    )
    combined = combined.drop(columns=["_key_date", "_key_a", "_key_b", "_key_observed"])

    n_fights = combined.groupby(
        [combined["event_date"].astype(str).str[:10], "fighter_a", "fighter_b"]
    ).ngroups
    logger.info(
        "Loaded %d historical odds rows (%d unique fights) from %d source files",
        len(combined), n_fights, len(parts),
    )

    combined.attrs["odds_snapshot_key"] = cache_key
    _ALL_ODDS_CACHE = (cache_key, combined)
    return combined.copy()


def build_historical_odds_roles(
    historical_df: pd.DataFrame,
    *,
    entry_offset_days: float | None = 1.0,
) -> pd.DataFrame:
    """Select honest opening/entry/closing rows for each fight.

    Opening and entry are restricted to verified pre-fight observations;
    offset-zero/reference rows can only supply closing/CLV metadata. Entry is
    the closest observation at least ``entry_offset_days`` before start.
    """
    if historical_df is None or historical_df.empty:
        return pd.DataFrame()

    snapshot_key = historical_df.attrs.get("odds_snapshot_key")
    role_cache_key = (
        snapshot_key,
        None if entry_offset_days is None else float(entry_offset_days),
    )
    if snapshot_key is not None and role_cache_key in _ROLE_CACHE:
        return _ROLE_CACHE[role_cache_key].copy()

    working = historical_df.copy()
    if "source_kind" not in working.columns:
        working["source_kind"] = "odds_api"
    if "source_file" not in working.columns:
        working["source_file"] = "unknown"
    offsets = pd.to_numeric(working.get("offset_days"), errors="coerce")
    if "hours_to_start" not in working.columns:
        working["hours_to_start"] = offsets * 24.0
    else:
        working["hours_to_start"] = pd.to_numeric(
            working["hours_to_start"], errors="coerce"
        )
    if "observed_at" not in working.columns:
        working["observed_at"] = pd.to_datetime(
            working.get("query_date"), errors="coerce", utc=True
        )
    working["observed_at"] = pd.to_datetime(
        working["observed_at"], errors="coerce", utc=True
    )
    working["commence_time"] = pd.to_datetime(
        working.get(
            "commence_time",
            pd.Series(pd.NaT, index=working.index),
        ),
        errors="coerce",
        utc=True,
    )
    if "verified_prefight" not in working.columns:
        working["verified_prefight"] = (
            working["source_kind"].isin(VERIFIED_PREFIGHT_SOURCE_KINDS)
            & working["hours_to_start"].notna()
            & (working["hours_to_start"] > 0.0)
        )
    else:
        working["verified_prefight"] = working["verified_prefight"].fillna(False).astype(bool)

    for column in ("a_fair_prob", "b_fair_prob", "a_decimal_odds", "b_decimal_odds"):
        working[column] = pd.to_numeric(working.get(column), errors="coerce")
    working = working[
        working["a_fair_prob"].between(0.0, 1.0, inclusive="both")
        & working["b_fair_prob"].between(0.0, 1.0, inclusive="both")
        & working["a_decimal_odds"].gt(1.0)
        & working["b_decimal_odds"].gt(1.0)
    ].copy()
    if working.empty:
        return pd.DataFrame()

    working["_event_date_key"] = working["event_date"].apply(_normalize_backfill_event_date)
    working["_pair_key"] = working.apply(
        lambda row: _backfill_pair_key(row.get("fighter_a", ""), row.get("fighter_b", "")),
        axis=1,
    )
    role_rows: list[dict] = []
    minimum_entry_hours = (
        None if entry_offset_days is None else float(entry_offset_days) * 24.0
    )

    def select_row(group: pd.DataFrame, role: str) -> pd.Series | None:
        if role == "closing":
            eligible = group[group["hours_to_start"].notna() & (group["hours_to_start"] >= 0.0)]
            ascending = True
        elif role == "opening":
            eligible = group[group["verified_prefight"] & (group["hours_to_start"] > 0.0)]
            ascending = False
        else:
            if minimum_entry_hours is None:
                return None
            eligible = group[
                group["verified_prefight"]
                & (group["hours_to_start"] >= minimum_entry_hours)
            ]
            ascending = True
        if eligible.empty:
            return None
        return eligible.sort_values(
            ["hours_to_start", "observed_at", "source_file"],
            ascending=[ascending, not ascending, True],
            kind="stable",
        ).iloc[0]

    for (event_key, pair_key), raw_group in working.groupby(
        ["_event_date_key", "_pair_key"], dropna=False
    ):
        display = raw_group.iloc[0]
        group = _reorient_backfill_group(
            raw_group,
            str(display["fighter_a"]),
            str(display["fighter_b"]),
        )
        output = {
            "event_date": event_key,
            "fighter_a": str(display["fighter_a"]),
            "fighter_b": str(display["fighter_b"]),
            "_pair_key": pair_key,
        }
        for role in ("opening", "entry", "closing"):
            output.update(
                {
                    f"{role}_prob_a": np.nan,
                    f"{role}_prob_b": np.nan,
                    f"{role}_odds_a": np.nan,
                    f"{role}_odds_b": np.nan,
                    f"{role}_source_kind": "",
                    f"{role}_source_file": "",
                    f"{role}_observed_at": pd.NaT,
                    f"{role}_commence_time": pd.NaT,
                    f"{role}_hours_to_start": np.nan,
                    f"{role}_verified_prefight": False,
                }
            )
        output["entry_offset_actual"] = np.nan
        for role in ("opening", "entry", "closing"):
            selected = select_row(group, role)
            if selected is None:
                continue
            output.update(
                {
                    f"{role}_prob_a": selected["a_fair_prob"],
                    f"{role}_prob_b": selected["b_fair_prob"],
                    f"{role}_odds_a": selected["a_decimal_odds"],
                    f"{role}_odds_b": selected["b_decimal_odds"],
                    f"{role}_source_kind": selected["source_kind"],
                    f"{role}_source_file": selected["source_file"],
                    f"{role}_observed_at": selected["observed_at"],
                    f"{role}_commence_time": selected["commence_time"],
                    f"{role}_hours_to_start": selected["hours_to_start"],
                    f"{role}_verified_prefight": bool(selected["verified_prefight"]),
                }
            )
            if role == "entry":
                output["entry_offset_actual"] = float(selected["hours_to_start"]) / 24.0
        role_rows.append(output)
    result = pd.DataFrame(role_rows)
    if snapshot_key is not None:
        if len(_ROLE_CACHE) >= 4:
            _ROLE_CACHE.clear()
        _ROLE_CACHE[role_cache_key] = result
    return result.copy()


def merge_historical_odds_roles(
    fights: pd.DataFrame,
    historical_df: Optional[pd.DataFrame] = None,
    *,
    entry_offset_days: float | None = 1.0,
) -> pd.DataFrame:
    """Merge selected roles onto fights and orient prices to each fight row."""
    result = fights.copy()
    if historical_df is None:
        historical_df = load_all_historical_odds()
    roles = build_historical_odds_roles(
        historical_df,
        entry_offset_days=entry_offset_days,
    )
    if roles.empty or not {"event_date", "fighter_a", "fighter_b"}.issubset(result.columns):
        return result

    result["_odds_event_date"] = result["event_date"].apply(_normalize_backfill_event_date)
    result["_odds_pair_key"] = result.apply(
        lambda row: _backfill_pair_key(row.get("fighter_a", ""), row.get("fighter_b", "")),
        axis=1,
    )
    roles = roles.rename(
        columns={
            "event_date": "_odds_event_date",
            "fighter_a": "_odds_source_fighter_a",
            "fighter_b": "_odds_source_fighter_b",
            "_pair_key": "_odds_pair_key",
        }
    )
    merged = result.merge(
        roles,
        on=["_odds_event_date", "_odds_pair_key"],
        how="left",
        validate="many_to_one",
    )
    reverse = (
        merged["_odds_source_fighter_a"].notna()
        & (merged["fighter_a"].apply(_normalize_backfill_name)
           == merged["_odds_source_fighter_b"].apply(_normalize_backfill_name))
    )
    for role in ("opening", "entry", "closing"):
        for value in ("prob", "odds"):
            left = f"{role}_{value}_a"
            right = f"{role}_{value}_b"
            if left not in merged.columns or right not in merged.columns:
                continue
            left_values = merged.loc[reverse, left].copy()
            merged.loc[reverse, left] = merged.loc[reverse, right].values
            merged.loc[reverse, right] = left_values.values
    return merged.drop(
        columns=[
            "_odds_event_date", "_odds_pair_key",
            "_odds_source_fighter_a", "_odds_source_fighter_b",
        ],
        errors="ignore",
    )


def get_fight_odds_timeline(
    fighter_a: str,
    fighter_b: str,
    historical_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Get the odds timeline for a specific fight (opening → midweek → closing).

    Returns DataFrame sorted by offset_days descending (opening first).
    """
    if historical_df is None:
        historical_df = load_all_historical_odds()

    if historical_df.empty:
        return pd.DataFrame()

    fa = _normalize_backfill_name(fighter_a)
    fb = _normalize_backfill_name(fighter_b)

    mask = (
        (historical_df["fighter_a"].apply(_normalize_backfill_name) == fa) &
        (historical_df["fighter_b"].apply(_normalize_backfill_name) == fb)
    ) | (
        (historical_df["fighter_a"].apply(_normalize_backfill_name) == fb) &
        (historical_df["fighter_b"].apply(_normalize_backfill_name) == fa)
    )

    timeline = historical_df[mask].sort_values("offset_days", ascending=False)
    return timeline


def compute_line_movement_from_backfill(historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute line movement features from backfilled historical odds.

    For each fight, computes:
      - opening_prob_a/b: probability at earliest snapshot (7 days out)
      - closing_prob_a/b: probability at latest snapshot (1 day out)
      - line_movement: closing - opening probability shift
      - line_abs_movement: absolute shift
      - line_is_sharp: > 5% total movement
      - line_steam_move: consistent direction across all snapshots
    """
    if historical_df.empty:
        return pd.DataFrame()

    working = historical_df.copy()
    if "verified_prefight" in working.columns:
        working = working[working["verified_prefight"].fillna(False).astype(bool)]
    if working.empty:
        return pd.DataFrame()
    working["_event_date_key"] = working["event_date"].apply(_normalize_backfill_event_date)
    working["_pair_key"] = working.apply(
        lambda row: _backfill_pair_key(row.get("fighter_a", ""), row.get("fighter_b", "")),
        axis=1,
    )
    results = []

    for (_event_date_key, _pair_key), group in working.groupby(
        ["_event_date_key", "_pair_key"], dropna=False
    ):
        sort_cols = ["offset_days"] + (["query_date"] if "query_date" in group.columns else [])
        ascending = [False] + ([True] if "query_date" in group.columns else [])
        group = group.sort_values(sort_cols, ascending=ascending)
        display_row = group.iloc[0]
        target_fa = display_row.get("fighter_a", "")
        target_fb = display_row.get("fighter_b", "")
        group = _reorient_backfill_group(group, target_fa, target_fb)

        if len(group) < 2:
            continue
        analysis = compute_line_movement_analysis(group["a_fair_prob"].tolist())
        feature_values = analysis_to_line_movement_features(analysis)

        results.append({
            "event_date": display_row.get("event_date"),
            "fighter_a": target_fa,
            "fighter_b": target_fb,
            "opening_prob_a": analysis["opening_prob_a"],
            "opening_prob_b": analysis["opening_prob_b"],
            "closing_prob_a": analysis["current_prob_a"],
            "closing_prob_b": analysis["current_prob_b"],
            **feature_values,
            "num_snapshots": len(group),
        })

    return pd.DataFrame(results)


def _reorient_backfill_group(group: pd.DataFrame, fighter_a: str, fighter_b: str) -> pd.DataFrame:
    """Orient a mixed historical group to a single requested fighter order."""
    target_a = _normalize_backfill_name(fighter_a)
    target_b = _normalize_backfill_name(fighter_b)
    oriented = group.copy()
    row_a = oriented["fighter_a"].apply(_normalize_backfill_name)
    row_b = oriented["fighter_b"].apply(_normalize_backfill_name)
    reverse_mask = (row_a == target_b) & (row_b == target_a)
    if not reverse_mask.any():
        return oriented

    swappable_pairs = [
        ("fighter_a", "fighter_b"),
        ("a_fair_prob", "b_fair_prob"),
        ("a_decimal_odds", "b_decimal_odds"),
    ]
    for left, right in swappable_pairs:
        if left not in oriented.columns or right not in oriented.columns:
            continue
        left_values = oriented.loc[reverse_mask, left].copy()
        right_values = oriented.loc[reverse_mask, right].copy()
        oriented.loc[reverse_mask, left] = right_values
        oriented.loc[reverse_mask, right] = left_values

    return oriented


def merge_line_movement_features(
    features_df: pd.DataFrame,
    historical_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge line movement features onto a features frame.

    Always returns the six contract columns, filled with NaN where no valid
    historical movement exists.
    """
    result = features_df.copy()
    for col in LINE_MOVEMENT_FEATURE_COLS:
        if col not in result.columns:
            result[col] = np.nan

    required_keys = {"event_date", "fighter_a", "fighter_b"}
    if not required_keys.issubset(result.columns):
        return result

    if historical_df is None:
        historical_df = load_all_historical_odds()
    if historical_df.empty:
        return result

    movement = compute_line_movement_from_backfill(historical_df)
    if movement.empty:
        return result

    left = result.copy()
    right = movement.copy()

    left["_line_event_date"] = pd.to_datetime(left["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    right["_line_event_date"] = pd.to_datetime(right["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    left["_line_fighter_a"] = left["fighter_a"].apply(_normalize_backfill_name)
    left["_line_fighter_b"] = left["fighter_b"].apply(_normalize_backfill_name)
    left["_line_pair_key"] = left.apply(
        lambda row: _backfill_pair_key(row.get("fighter_a", ""), row.get("fighter_b", "")),
        axis=1,
    )
    right["_line_fighter_a"] = right["fighter_a"].apply(_normalize_backfill_name)
    right["_line_fighter_b"] = right["fighter_b"].apply(_normalize_backfill_name)
    right["_line_pair_key"] = right.apply(
        lambda row: _backfill_pair_key(row.get("fighter_a", ""), row.get("fighter_b", "")),
        axis=1,
    )

    merge_keys = ["_line_event_date", "_line_pair_key"]
    right_cols = merge_keys + [
        "_line_fighter_a",
        "_line_fighter_b",
    ] + list(LINE_MOVEMENT_FEATURE_COLS)
    merged = left.merge(
        right[right_cols].drop_duplicates(subset=merge_keys, keep="last"),
        on=merge_keys,
        how="left",
        suffixes=("", "__hist"),
    )

    reverse_mask = (
        merged["_line_fighter_a__hist"].notna()
        & (merged["_line_fighter_a"] == merged["_line_fighter_b__hist"])
        & (merged["_line_fighter_b"] == merged["_line_fighter_a__hist"])
    )
    if reverse_mask.any():
        if "line_movement__hist" in merged.columns:
            merged.loc[reverse_mask, "line_movement__hist"] = -merged.loc[reverse_mask, "line_movement__hist"]
        if {
            "line_direction_toward_a__hist",
            "line_direction_toward_b__hist",
        }.issubset(merged.columns):
            toward_a = merged.loc[reverse_mask, "line_direction_toward_a__hist"].copy()
            toward_b = merged.loc[reverse_mask, "line_direction_toward_b__hist"].copy()
            merged.loc[reverse_mask, "line_direction_toward_a__hist"] = toward_b
            merged.loc[reverse_mask, "line_direction_toward_b__hist"] = toward_a

    for col in LINE_MOVEMENT_FEATURE_COLS:
        hist_col = f"{col}__hist"
        if hist_col in merged.columns:
            merged[col] = merged[hist_col].combine_first(merged[col])

    drop_cols = merge_keys + [
        "_line_fighter_a",
        "_line_fighter_b",
        "_line_fighter_a__hist",
        "_line_fighter_b__hist",
    ] + [
        f"{col}__hist" for col in LINE_MOVEMENT_FEATURE_COLS if f"{col}__hist" in merged.columns
    ]
    merged = merged.drop(columns=drop_cols, errors="ignore")
    return merged
