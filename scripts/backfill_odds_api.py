"""
Backfill missing moneyline odds from The Odds API historical endpoint.

Covers fights where the Kaggle odds dataset had no data (Jan-Feb 2025,
Oct 2025-Mar 2026, and any name-mismatched fights).
"""

import sys
import time
from datetime import timedelta
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ODDS_API_KEY, RAW_DATA_DIR

API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "mma_mixed_martial_arts"


def norm(name):
    if pd.isna(name):
        return ""
    return str(name).strip().lower().replace(".", "").replace("'", "").replace("-", " ")


def fuzzy_score(a, b):
    return SequenceMatcher(None, a, b).ratio()


def fetch_historical_odds(date_str):
    """Fetch historical MMA odds for a given ISO date string."""
    resp = requests.get(
        f"{API_BASE}/historical/sports/{SPORT}/odds",
        params={
            "apiKey": ODDS_API_KEY,
            "date": date_str,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
        },
        timeout=30,
    )
    remaining = resp.headers.get("x-requests-remaining", "?")
    if resp.status_code != 200:
        print(f"  API error {resp.status_code}: {resp.text[:200]}")
        return [], remaining
    data = resp.json().get("data", [])
    return data, remaining


def extract_odds_from_api(events):
    """Extract fighter odds from API response events."""
    results = {}
    for event in events:
        home = norm(event.get("home_team", ""))
        away = norm(event.get("away_team", ""))
        if not home or not away:
            continue

        # Get best available odds (first bookmaker with h2h)
        for bm in event.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                odds_map = {}
                for outcome in market.get("outcomes", []):
                    n = norm(outcome.get("name", ""))
                    odds_map[n] = outcome.get("price")
                if odds_map:
                    key = frozenset([home, away])
                    if key not in results:
                        results[key] = odds_map
                    break
            if frozenset([home, away]) in results:
                break

    return results


def match_fighter(target, api_names, threshold=0.70):
    """Fuzzy match a fighter name against API names."""
    best = None
    best_score = 0
    for name in api_names:
        s = fuzzy_score(target, name)
        if s > best_score:
            best_score = s
            best = name
    return best if best_score >= threshold else None


def main():
    master_path = RAW_DATA_DIR / "ufc-master.csv"
    master = pd.read_csv(master_path, low_memory=False)
    master["Date"] = pd.to_datetime(master["Date"], errors="coerce")

    # Find fights still missing odds
    needs = (master["Date"] >= "2025-01-01") & (master["RedOdds"].isna())
    missing_dates = sorted(master.loc[needs, "Date"].dt.date.unique())
    print(f"Fights still missing odds: {needs.sum()}")
    print(f"Unique dates to query: {len(missing_dates)}")

    total_matched = 0
    total_api_calls = 0

    for event_date in missing_dates:
        # Query API with date at noon UTC (day before event to get pre-fight odds)
        query_date = f"{event_date}T12:00:00Z"
        events, remaining = fetch_historical_odds(query_date)
        total_api_calls += 1
        time.sleep(0.5)

        if not events:
            # Try the day of the event
            query_date2 = f"{event_date - timedelta(days=1)}T12:00:00Z"
            events, remaining = fetch_historical_odds(query_date2)
            total_api_calls += 1
            time.sleep(0.5)

        if not events:
            print(f"  {event_date}: no API data (remaining: {remaining})")
            continue

        api_odds = extract_odds_from_api(events)
        all_api_names = set()
        for key in api_odds:
            all_api_names.update(key)

        # Match to master rows for this date
        date_mask = needs & (master["Date"].dt.date == event_date)
        date_matched = 0

        for idx in master.index[date_mask]:
            row = master.loc[idx]
            rn = norm(row["RedFighter"])
            bn = norm(row["BlueFighter"])

            # Try exact match
            key = frozenset([rn, bn])
            if key in api_odds:
                odds = api_odds[key]
                master.loc[idx, "RedOdds"] = odds.get(rn, np.nan)
                master.loc[idx, "BlueOdds"] = odds.get(bn, np.nan)
                date_matched += 1
                continue

            # Try fuzzy match
            rn_match = match_fighter(rn, all_api_names)
            bn_match = match_fighter(bn, all_api_names)
            if rn_match and bn_match:
                key = frozenset([rn_match, bn_match])
                if key in api_odds:
                    odds = api_odds[key]
                    master.loc[idx, "RedOdds"] = odds.get(rn_match, np.nan)
                    master.loc[idx, "BlueOdds"] = odds.get(bn_match, np.nan)
                    date_matched += 1

        total_matched += date_matched
        fights_on_date = date_mask.sum()
        print(f"  {event_date}: {date_matched}/{fights_on_date} matched ({len(events)} API events, remaining: {remaining})")

    # Save
    master.to_csv(master_path, index=False)

    # Final stats
    still_missing = (master["Date"] >= "2025-01-01") & (master["RedOdds"].isna())
    print(f"\nAPI calls used: {total_api_calls}")
    print(f"New matches from API: {total_matched}")
    print(f"Still missing odds: {still_missing.sum()}/{(master['Date'] >= '2025-01-01').sum()}")
    print(f"Saved to {master_path}")


if __name__ == "__main__":
    main()
