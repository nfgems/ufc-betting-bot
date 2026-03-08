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

import pandas as pd
import numpy as np

from src.data.odds_client import OddsClient
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

BACKFILL_DIR = RAW_DATA_DIR / "historical_odds"
BACKFILL_DIR.mkdir(parents=True, exist_ok=True)

# How many days before the event to snapshot odds
SNAPSHOT_OFFSETS = [7, 3, 1]  # 7 days out (opening), 3 days (midweek), 1 day (closing)

# Rate limiting: The Odds API has per-second limits
REQUEST_DELAY = 1.5  # seconds between requests


def _match_fighters(api_event: dict, fighter_a: str, fighter_b: str) -> bool:
    """Check if an API event matches a fight by fighter names."""
    home = (api_event.get("home_team") or "").lower()
    away = (api_event.get("away_team") or "").lower()
    fa = fighter_a.lower()
    fb = fighter_b.lower()

    # Try exact match
    if (fa in home and fb in away) or (fb in home and fa in away):
        return True

    # Try last-name match (handles "John Smith" vs "Smith")
    fa_last = fa.split()[-1] if fa else ""
    fb_last = fb.split()[-1] if fb else ""
    if fa_last and fb_last:
        if (fa_last in home and fb_last in away) or (fb_last in home and fa_last in away):
            return True

    return False


def _extract_fight_odds(api_events: list[dict], fighter_a: str, fighter_b: str) -> dict | None:
    """Extract odds for a specific fight from API response."""
    for event in api_events:
        if not _match_fighters(event, fighter_a, fighter_b):
            continue

        home = event.get("home_team", "")
        away = event.get("away_team", "")

        # Determine which API fighter maps to fighter_a
        home_is_a = fighter_a.lower() in home.lower() or fighter_a.split()[-1].lower() in home.lower()

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
        resume: If True, skip event dates already backfilled.

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

    # Group fights by event date
    event_dates = sorted(fights_df["event_date"].unique())
    all_results = []

    total_dates = len(event_dates)
    api_calls = 0

    for i, event_date in enumerate(event_dates):
        event_str = pd.Timestamp(event_date).strftime("%Y-%m-%d")
        event_fights = fights_df[fights_df["event_date"] == event_date]

        for offset in offsets:
            # Skip if already backfilled
            if not existing.empty:
                already_done = existing[
                    (existing["event_date"] == event_str) &
                    (existing["offset_days"] == offset)
                ]
                if len(already_done) > 0:
                    continue

            results = backfill_event_date(client, event_str, event_fights, offset)
            all_results.extend(results)
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
    combined = combined.drop_duplicates(
        subset=["event_date", "fighter_a", "fighter_b", "offset_days"],
        keep="last",
    )
    combined.to_csv(path, index=False)
    return combined


def load_historical_odds() -> pd.DataFrame:
    """Load the backfilled historical odds data."""
    path = BACKFILL_DIR / "historical_odds.csv"
    if not path.exists():
        logger.warning("No historical odds data found. Run backfill first.")
        return pd.DataFrame()
    return pd.read_csv(path)


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
        historical_df = load_historical_odds()

    if historical_df.empty:
        return pd.DataFrame()

    fa = fighter_a.lower()
    fb = fighter_b.lower()

    mask = (
        (historical_df["fighter_a"].str.lower() == fa) &
        (historical_df["fighter_b"].str.lower() == fb)
    ) | (
        (historical_df["fighter_a"].str.lower() == fb) &
        (historical_df["fighter_b"].str.lower() == fa)
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

    results = []

    for (event_date, fa, fb), group in historical_df.groupby(
        ["event_date", "fighter_a", "fighter_b"]
    ):
        group = group.sort_values("offset_days", ascending=False)  # opening first

        if len(group) < 2:
            continue

        opening_a = group.iloc[0]["a_fair_prob"]
        closing_a = group.iloc[-1]["a_fair_prob"]
        opening_b = group.iloc[0]["b_fair_prob"]
        closing_b = group.iloc[-1]["b_fair_prob"]

        movement = closing_a - opening_a
        abs_movement = abs(movement)

        # Sharp move: > 5% total shift
        is_sharp = abs_movement > 0.05

        # Steam move: all shifts in the same direction
        shifts = group["a_fair_prob"].diff().dropna()
        steam_move = False
        if len(shifts) >= 2:
            steam_move = all(s > 0.005 for s in shifts) or all(s < -0.005 for s in shifts)

        if movement > 0.02:
            direction = "toward_a"
        elif movement < -0.02:
            direction = "toward_b"
        else:
            direction = "stable"

        results.append({
            "event_date": event_date,
            "fighter_a": fa,
            "fighter_b": fb,
            "opening_prob_a": opening_a,
            "opening_prob_b": opening_b,
            "closing_prob_a": closing_a,
            "closing_prob_b": closing_b,
            "line_movement": movement,
            "line_abs_movement": abs_movement,
            "line_is_sharp": 1 if is_sharp else 0,
            "line_steam_move": 1 if steam_move else 0,
            "line_direction_toward_a": 1 if direction == "toward_a" else 0,
            "line_direction_toward_b": 1 if direction == "toward_b" else 0,
            "num_snapshots": len(group),
        })

    return pd.DataFrame(results)
