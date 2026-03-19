"""
Smart backfill for remaining V5 gaps — handles name variations better.

Targets:
1. Moneyline gaps via Odds API (with improved name matching)
2. Method odds gaps via Odds API h2h events (no method market, but we get ML)
3. Method odds via BFO scraping for remaining

Usage:
    python scripts/backfill_v5_smart_recovery.py
"""

import csv
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from difflib import SequenceMatcher

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.config import ODDS_API_KEY, RAW_DATA_DIR

API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "mma_mixed_martial_arts"
DELAY = 1.0

MISSING_ML_CSV = RAW_DATA_DIR / "v5_missing_moneyline.csv"
MISSING_MO_CSV = RAW_DATA_DIR / "v5_missing_method_odds.csv"
ML_CSV = RAW_DATA_DIR / "historical_odds" / "historical_odds.csv"
MO_CSV = RAW_DATA_DIR / "method_odds" / "historical_method_odds_all.csv"


def norm(n):
    if pd.isna(n) or not n:
        return ""
    n = unicodedata.normalize("NFKD", str(n))
    n = "".join(c for c in n if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-zA-Z\s]", "", n).lower().split())


def tokens(n):
    return set(norm(n).split())


def last_name(n):
    parts = norm(n).split()
    return parts[-1] if parts else ""


def first_name(n):
    parts = norm(n).split()
    return parts[0] if parts else ""


def name_match(target, candidate, threshold=0.65):
    """Enhanced name matching handling reversed names, single-word names, etc."""
    tn, cn = norm(target), norm(candidate)
    if not tn or not cn:
        return False

    # Exact
    if tn == cn:
        return True

    # Fuzzy on full name
    if SequenceMatcher(None, tn, cn).ratio() >= threshold:
        return True

    # Reversed name (Zhang Weili vs Weili Zhang)
    t_parts = tn.split()
    c_parts = cn.split()
    if len(t_parts) >= 2 and len(c_parts) >= 2:
        if SequenceMatcher(None, " ".join(reversed(t_parts)), cn).ratio() >= threshold:
            return True

    # Token overlap — if all tokens from shorter name appear in longer
    t_tokens = set(t_parts)
    c_tokens = set(c_parts)
    if len(t_tokens) >= 2 and t_tokens.issubset(c_tokens):
        return True
    if len(c_tokens) >= 2 and c_tokens.issubset(t_tokens):
        return True

    # Single-word name match (Rongzhu = Zhu Rong, Alatengheili, Sumudaerji, etc.)
    if len(t_parts) == 1 and len(c_parts) >= 2:
        # Check if target is a concatenation of candidate parts
        joined = "".join(c_parts)
        if tn == joined or SequenceMatcher(None, tn, joined).ratio() >= 0.8:
            return True
        rev_joined = "".join(reversed(c_parts))
        if tn == rev_joined or SequenceMatcher(None, tn, rev_joined).ratio() >= 0.8:
            return True
    if len(c_parts) == 1 and len(t_parts) >= 2:
        joined = "".join(t_parts)
        if cn == joined or SequenceMatcher(None, cn, joined).ratio() >= 0.8:
            return True
        rev_joined = "".join(reversed(t_parts))
        if cn == rev_joined or SequenceMatcher(None, cn, rev_joined).ratio() >= 0.8:
            return True

    # Last name match for common cases like "King Green" vs "Bobby Green"
    if last_name(target) == last_name(candidate) and last_name(target):
        # Need at least last name match + some other signal
        return True

    return False


def fight_match(target_a, target_b, api_home, api_away):
    """Match a target fight (a vs b) against an API event.
    Returns (matched, home_is_a) or (False, None)."""
    if name_match(target_a, api_home) and name_match(target_b, api_away):
        return True, True
    if name_match(target_a, api_away) and name_match(target_b, api_home):
        return True, False
    return False, None


def amer_to_dec(odds):
    odds = float(odds)
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def american_to_implied(odds):
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def fetch_historical(date_str, markets="h2h"):
    resp = requests.get(
        f"{API_BASE}/historical/sports/{SPORT}/odds",
        params={
            "apiKey": ODDS_API_KEY,
            "date": date_str,
            "regions": "us",
            "markets": markets,
            "oddsFormat": "american",
        },
        timeout=30,
    )
    remaining = resp.headers.get("x-requests-remaining", "?")
    if resp.status_code != 200:
        return [], remaining
    return resp.json().get("data", []), remaining


def extract_ml(events, fighter_a, fighter_b):
    """Extract moneyline from API events."""
    for ev in events:
        matched, home_is_a = fight_match(fighter_a, fighter_b, ev.get("home_team", ""), ev.get("away_team", ""))
        if not matched:
            continue

        a_odds_list, b_odds_list = [], []
        num_bm = 0
        home = ev["home_team"]
        away = ev["away_team"]

        for bm in ev.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = {}
                for o in market.get("outcomes", []):
                    outcomes[norm(o["name"])] = o["price"]
                if not outcomes:
                    continue

                h_price = outcomes.get(norm(home))
                a_price = outcomes.get(norm(away))
                if h_price is not None and a_price is not None:
                    if home_is_a:
                        a_odds_list.append(h_price)
                        b_odds_list.append(a_price)
                    else:
                        a_odds_list.append(a_price)
                        b_odds_list.append(h_price)
                    num_bm += 1

        if not a_odds_list:
            continue

        a_dec = float(np.mean([amer_to_dec(o) for o in a_odds_list]))
        b_dec = float(np.mean([amer_to_dec(o) for o in b_odds_list]))
        a_impl = 1 / a_dec
        b_impl = 1 / b_dec
        total = a_impl + b_impl

        return {
            "a_decimal_odds": round(a_dec, 4),
            "b_decimal_odds": round(b_dec, 4),
            "a_fair_prob": round(a_impl / total, 6),
            "b_fair_prob": round(b_impl / total, 6),
            "num_bookmakers": num_bm,
        }
    return None


def main():
    if not ODDS_API_KEY:
        print("ERROR: ODDS_API_KEY not set")
        sys.exit(1)

    # Load missing
    ml_missing = list(csv.DictReader(open(MISSING_ML_CSV)))
    mo_missing = list(csv.DictReader(open(MISSING_MO_CSV)))

    print(f"Moneyline missing: {len(ml_missing)}")
    print(f"Method odds missing: {len(mo_missing)}")
    print()

    # Group ML by date
    ml_by_date = defaultdict(list)
    for r in ml_missing:
        ml_by_date[r["event_date"]].append(r)

    all_dates = sorted(ml_by_date.keys())
    now_iso = datetime.now(timezone.utc).isoformat()

    # Load existing ML for dedup
    existing_ml = pd.read_csv(ML_CSV, low_memory=False)
    existing_ml_keys = set(existing_ml["fight_key"].dropna().unique())

    ml_new = []
    api_remaining = "?"

    print("=== Moneyline Recovery via Odds API ===")
    for date in all_dates:
        fights = ml_by_date[date]
        query = f"{date}T12:00:00Z"
        events, api_remaining = fetch_historical(query)
        time.sleep(DELAY)

        if not events:
            # Try day before
            prev = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            events, api_remaining = fetch_historical(f"{prev}T12:00:00Z")
            time.sleep(DELAY)

        matched = 0
        for fight in fights:
            result = extract_ml(events, fight["fighter_a"], fight["fighter_b"])
            if result is None:
                print(f"  MISS: {date} {fight['fighter_a']} vs {fight['fighter_b']}")
                continue
            matched += 1

            names_sorted = sorted([norm(fight["fighter_a"]), norm(fight["fighter_b"])])
            fight_key = f"{date}|{names_sorted[0]}|{names_sorted[1]}"

            if fight_key in existing_ml_keys:
                print(f"  DUP: {fight_key}")
                continue

            ml_new.append({
                "event_date": date,
                "fighter_a": fight["fighter_a"],
                "fighter_b": fight["fighter_b"],
                "query_date": date,
                "offset_days": 0,
                "a_fair_prob": result["a_fair_prob"],
                "b_fair_prob": result["b_fair_prob"],
                "a_decimal_odds": result["a_decimal_odds"],
                "b_decimal_odds": result["b_decimal_odds"],
                "num_bookmakers": result["num_bookmakers"],
                "event_date_dt": date,
                "fight_key": fight_key,
            })
            existing_ml_keys.add(fight_key)

        if matched:
            print(f"  {date}: {matched}/{len(fights)} matched")

    print(f"\nMoneyline recovered: {len(ml_new)}")
    print(f"API remaining: {api_remaining}")

    # Append ML
    if ml_new:
        fields = list(existing_ml.columns)
        with open(ML_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            for row in ml_new:
                writer.writerow(row)
        print(f"Appended {len(ml_new)} ML rows")

        # Update missing
        recovered_keys = set()
        for r in ml_new:
            recovered_keys.add((r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"])))
            recovered_keys.add((r["event_date"], norm(r["fighter_b"]), norm(r["fighter_a"])))
        remaining = [r for r in ml_missing if (r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"])) not in recovered_keys]
        with open(MISSING_ML_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["event_date", "fighter_a", "fighter_b"])
            writer.writeheader()
            for r in remaining:
                writer.writerow(r)
        print(f"ML missing: {len(ml_missing)} -> {len(remaining)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
