"""
Backfill missing method odds AND moneyline from The Odds API historical endpoint.

Priority 1: Method odds 2025-2026 (598 missing) — CRITICAL
Priority 2: Moneyline 2025-2026 (270 missing) — MEDIUM

Usage:
    python scripts/backfill_v5_odds_api_historical.py [--dry-run] [--method-only] [--ml-only] [--max-dates N]
"""

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from src.config import ODDS_API_KEY, RAW_DATA_DIR

API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "mma_mixed_martial_arts"
REQUEST_DELAY = 1.0  # seconds between API calls

# ── Paths ──────────────────────────────────────────────────────────────
MISSING_METHOD_CSV = RAW_DATA_DIR / "v5_missing_method_odds.csv"
MISSING_ML_CSV = RAW_DATA_DIR / "v5_missing_moneyline.csv"
METHOD_ODDS_CSV = RAW_DATA_DIR / "method_odds" / "historical_method_odds_all.csv"
MONEYLINE_CSV = RAW_DATA_DIR / "historical_odds" / "historical_odds.csv"


def norm(name):
    if pd.isna(name):
        return ""
    return str(name).strip().lower().replace(".", "").replace("'", "").replace("-", " ")


def fuzzy_score(a, b):
    return SequenceMatcher(None, a, b).ratio()


def match_name(target, candidates, threshold=0.72):
    best, best_score = None, 0
    for c in candidates:
        s = fuzzy_score(target, c)
        if s > best_score:
            best_score = s
            best = c
    return best if best_score >= threshold else None


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
    """Fetch historical MMA odds for a given ISO date and market."""
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
    used = resp.headers.get("x-requests-used", "?")
    if resp.status_code == 422:
        return [], remaining, used, "market_unsupported"
    if resp.status_code != 200:
        print(f"  API error {resp.status_code}: {resp.text[:200]}")
        return [], remaining, used, f"error_{resp.status_code}"
    data = resp.json().get("data", [])
    return data, remaining, used, "ok"


# ── Method odds extraction ─────────────────────────────────────────────

METHOD_MARKETS = [
    "method_of_victory",
    "fighter_method_of_victory",
    "winning_method",
]

_METHOD_KEYWORDS = {
    "ko": ["ko", "tko", "knockout"],
    "sub": ["sub", "submission"],
    "dec": ["dec", "decision", "points"],
}


def classify_method(text):
    t = text.lower()
    for method, keywords in _METHOD_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return method
    return None


def identify_side(text, name_a, name_b):
    t = norm(text)
    a_n, b_n = norm(name_a), norm(name_b)
    a_match = a_n in t or fuzzy_score(a_n, t) > 0.65
    b_match = b_n in t or fuzzy_score(b_n, t) > 0.65
    # Also try last name
    a_last = a_n.split()[-1] if a_n.split() else ""
    b_last = b_n.split()[-1] if b_n.split() else ""
    if not a_match and a_last and a_last in t:
        a_match = True
    if not b_match and b_last and b_last in t:
        b_match = True
    if a_match and not b_match:
        return "a"
    if b_match and not a_match:
        return "b"
    return None


def extract_method_odds_from_events(events, fighter_a, fighter_b):
    """Try to extract method-of-victory probs for a specific fight from API events."""
    a_n, b_n = norm(fighter_a), norm(fighter_b)

    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        h_n, aw_n = norm(home), norm(away)

        # Match fight
        matched = False
        home_is_a = None
        if (fuzzy_score(a_n, h_n) >= 0.72 and fuzzy_score(b_n, aw_n) >= 0.72):
            matched, home_is_a = True, True
        elif (fuzzy_score(a_n, aw_n) >= 0.72 and fuzzy_score(b_n, h_n) >= 0.72):
            matched, home_is_a = True, False

        if not matched:
            continue

        api_a = home if home_is_a else away
        api_b = away if home_is_a else home

        probs = {"a_ko": [], "a_sub": [], "a_dec": [], "b_ko": [], "b_sub": [], "b_dec": []}

        for bm in event.get("bookmakers", []):
            for market in bm.get("markets", []):
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    method = classify_method(name)
                    if method is None:
                        continue
                    side = identify_side(name, api_a, api_b)
                    if side is None:
                        continue
                    prob = american_to_implied(outcome.get("price"))
                    if np.isnan(prob):
                        continue
                    probs[f"{side}_{method}"].append(prob)

        if not any(probs.values()):
            continue

        result = {}
        for key, vals in probs.items():
            result[f"{key}_odds_prob"] = float(np.mean(vals)) if vals else np.nan
        return result

    return None


# ── Moneyline extraction ───────────────────────────────────────────────

def extract_moneyline_from_events(events, fighter_a, fighter_b):
    """Extract moneyline odds for a specific fight from API events."""
    a_n, b_n = norm(fighter_a), norm(fighter_b)

    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        h_n, aw_n = norm(home), norm(away)

        matched = False
        home_is_a = None
        if (fuzzy_score(a_n, h_n) >= 0.72 and fuzzy_score(b_n, aw_n) >= 0.72):
            matched, home_is_a = True, True
        elif (fuzzy_score(a_n, aw_n) >= 0.72 and fuzzy_score(b_n, h_n) >= 0.72):
            matched, home_is_a = True, False

        if not matched:
            continue

        # Collect h2h odds across bookmakers
        a_odds_list, b_odds_list = [], []
        num_bookmakers = 0
        for bm in event.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = {norm(o["name"]): o["price"] for o in market.get("outcomes", []) if "name" in o and "price" in o}
                if not outcomes:
                    continue

                if home_is_a:
                    a_price = outcomes.get(norm(home))
                    b_price = outcomes.get(norm(away))
                else:
                    a_price = outcomes.get(norm(away))
                    b_price = outcomes.get(norm(home))

                if a_price is not None and b_price is not None:
                    a_odds_list.append(a_price)
                    b_odds_list.append(b_price)
                    num_bookmakers += 1

        if not a_odds_list:
            continue

        # Convert American to decimal
        def amer_to_dec(odds):
            odds = float(odds)
            if odds > 0:
                return 1 + odds / 100
            return 1 + 100 / abs(odds)

        a_dec = float(np.mean([amer_to_dec(o) for o in a_odds_list]))
        b_dec = float(np.mean([amer_to_dec(o) for o in b_odds_list]))

        a_impl = 1 / a_dec
        b_impl = 1 / b_dec
        total = a_impl + b_impl
        a_fair = a_impl / total
        b_fair = b_impl / total

        return {
            "a_decimal_odds": a_dec,
            "b_decimal_odds": b_dec,
            "a_fair_prob": a_fair,
            "b_fair_prob": b_fair,
            "num_bookmakers": num_bookmakers,
        }

    return None


def load_missing(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def dates_from_missing(rows):
    dates = sorted(set(r["event_date"] for r in rows))
    return dates


def fights_on_date(rows, date):
    return [r for r in rows if r["event_date"] == date]


def main():
    parser = argparse.ArgumentParser(description="Backfill V5 odds gaps from Odds API historical endpoint")
    parser.add_argument("--dry-run", action="store_true", help="Probe API without writing")
    parser.add_argument("--method-only", action="store_true", help="Only backfill method odds")
    parser.add_argument("--ml-only", action="store_true", help="Only backfill moneyline")
    parser.add_argument("--max-dates", type=int, default=0, help="Limit number of dates to query (0=all)")
    parser.add_argument("--start-date", type=str, default="", help="Start from this date (YYYY-MM-DD)")
    args = parser.parse_args()

    if not ODDS_API_KEY:
        print("ERROR: ODDS_API_KEY not set. Export it or add to .env")
        sys.exit(1)

    do_method = not args.ml_only
    do_ml = not args.method_only

    # Load missing files
    method_missing = load_missing(MISSING_METHOD_CSV) if do_method else []
    ml_missing = load_missing(MISSING_ML_CSV) if do_ml else []

    # Combine unique dates
    all_dates = sorted(set(
        dates_from_missing(method_missing) + dates_from_missing(ml_missing)
    ))
    if args.start_date:
        all_dates = [d for d in all_dates if d >= args.start_date]
    if args.max_dates > 0:
        all_dates = all_dates[:args.max_dates]

    print(f"Dates to query: {len(all_dates)}")
    print(f"Method odds missing: {len(method_missing)} fights")
    print(f"Moneyline missing: {len(ml_missing)} fights")
    print()

    # ── Probe: check if method_of_victory market works ──
    if do_method and all_dates:
        probe_date = f"{all_dates[-1]}T12:00:00Z"  # most recent date
        print(f"Probing method_of_victory market on {all_dates[-1]}...")
        for mkt in METHOD_MARKETS:
            events, remaining, used, status = fetch_historical(probe_date, markets=mkt)
            print(f"  Market '{mkt}': status={status}, events={len(events)}, remaining={remaining}")
            time.sleep(REQUEST_DELAY)
            if status == "ok" and events:
                # Check if any events have method-of-victory outcomes
                has_method = False
                for ev in events:
                    for bm in ev.get("bookmakers", []):
                        for m in bm.get("markets", []):
                            if m.get("key") == mkt:
                                has_method = True
                                break
                print(f"    -> Has method outcomes: {has_method}")
                if has_method:
                    print(f"    -> Using market: {mkt}")
                    METHOD_MARKET_KEY = mkt
                    break
        else:
            print("\nWARNING: No method-of-victory market found on historical API.")
            print("  Will fall back to BFO scraping for method odds.")
            METHOD_MARKET_KEY = None
            do_method_api = False
        if 'METHOD_MARKET_KEY' in dir() and METHOD_MARKET_KEY:
            do_method_api = True
        print()

    if args.dry_run:
        print("DRY RUN — exiting after probe.")
        return

    # ── Main backfill loop ──────────────────────────────────────────────
    method_new_rows = []
    ml_new_rows = []
    api_calls = 0
    now_iso = datetime.utcnow().isoformat() + "Z"

    for i, date in enumerate(all_dates):
        date_method_fights = fights_on_date(method_missing, date) if do_method else []
        date_ml_fights = fights_on_date(ml_missing, date) if do_ml else []

        if not date_method_fights and not date_ml_fights:
            continue

        print(f"[{i+1}/{len(all_dates)}] {date}: {len(date_method_fights)} method, {len(date_ml_fights)} ML missing")

        # Fetch h2h (moneyline) — also used for method if combined market
        query_date = f"{date}T12:00:00Z"

        # Try moneyline
        ml_events = []
        if date_ml_fights:
            ml_events, remaining, used, status = fetch_historical(query_date, markets="h2h")
            api_calls += 1
            time.sleep(REQUEST_DELAY)

            if not ml_events:
                # Try day before
                prev = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
                ml_events, remaining, used, status = fetch_historical(f"{prev}T12:00:00Z", markets="h2h")
                api_calls += 1
                time.sleep(REQUEST_DELAY)

            ml_matched = 0
            for fight in date_ml_fights:
                result = extract_moneyline_from_events(ml_events, fight["fighter_a"], fight["fighter_b"])
                if result is None:
                    continue
                ml_matched += 1

                # Build fight_key: date|name_a_lower|name_b_lower (alphabetically sorted)
                names_sorted = sorted([norm(fight["fighter_a"]), norm(fight["fighter_b"])])
                fight_key = f"{date}|{names_sorted[0]}|{names_sorted[1]}"

                ml_new_rows.append({
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
            if ml_matched:
                print(f"    ML: {ml_matched}/{len(date_ml_fights)} matched")

        # Try method odds
        method_events = []
        if date_method_fights and do_method and locals().get("do_method_api"):
            method_events, remaining, used, status = fetch_historical(query_date, markets=METHOD_MARKET_KEY)
            api_calls += 1
            time.sleep(REQUEST_DELAY)

            if not method_events:
                prev = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
                method_events, remaining, used, status = fetch_historical(f"{prev}T12:00:00Z", markets=METHOD_MARKET_KEY)
                api_calls += 1
                time.sleep(REQUEST_DELAY)

            method_matched = 0
            for fight in date_method_fights:
                result = extract_method_odds_from_events(method_events, fight["fighter_a"], fight["fighter_b"])
                if result is None:
                    continue
                method_matched += 1
                method_new_rows.append({
                    "event_date": date,
                    "fighter_a": fight["fighter_a"],
                    "fighter_b": fight["fighter_b"],
                    "event_title": "",
                    "source": "odds_api_historical",
                    "source_url": "",
                    "captured_at": now_iso,
                    "a_ko_odds_prob": result.get("a_ko_odds_prob", ""),
                    "a_sub_odds_prob": result.get("a_sub_odds_prob", ""),
                    "a_dec_odds_prob": result.get("a_dec_odds_prob", ""),
                    "b_ko_odds_prob": result.get("b_ko_odds_prob", ""),
                    "b_sub_odds_prob": result.get("b_sub_odds_prob", ""),
                    "b_dec_odds_prob": result.get("b_dec_odds_prob", ""),
                })
            if method_matched:
                print(f"    Method: {method_matched}/{len(date_method_fights)} matched")

        print(f"    API remaining: {remaining}")

        # Safety: stop if running low
        try:
            if int(remaining) < 10:
                print(f"\nWARNING: API quota low ({remaining} remaining). Stopping.")
                break
        except (ValueError, TypeError):
            pass

    # ── Append results ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"API calls used: {api_calls}")
    print(f"Method odds recovered: {len(method_new_rows)}")
    print(f"Moneyline recovered: {len(ml_new_rows)}")

    if ml_new_rows:
        # Deduplicate against existing fight_keys
        existing_ml = pd.read_csv(MONEYLINE_CSV, low_memory=False)
        existing_keys = set(existing_ml["fight_key"].dropna().unique())
        new_ml_deduped = [r for r in ml_new_rows if r["fight_key"] not in existing_keys]
        print(f"Moneyline after dedup: {len(new_ml_deduped)} new rows (skipped {len(ml_new_rows) - len(new_ml_deduped)} dupes)")

        if new_ml_deduped:
            ml_fields = list(existing_ml.columns)
            with open(MONEYLINE_CSV, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ml_fields)
                for row in new_ml_deduped:
                    writer.writerow(row)
            print(f"  -> Appended {len(new_ml_deduped)} rows to {MONEYLINE_CSV}")

    if method_new_rows:
        existing_method = pd.read_csv(METHOD_ODDS_CSV, low_memory=False)
        # Deduplicate on (event_date, fighter_a_norm, fighter_b_norm)
        existing_method_keys = set()
        for _, r in existing_method.iterrows():
            key = (str(r["event_date"]), norm(r["fighter_a"]), norm(r["fighter_b"]))
            existing_method_keys.add(key)
            # Also add reversed
            existing_method_keys.add((str(r["event_date"]), norm(r["fighter_b"]), norm(r["fighter_a"])))

        new_method_deduped = []
        for row in method_new_rows:
            key = (row["event_date"], norm(row["fighter_a"]), norm(row["fighter_b"]))
            if key not in existing_method_keys:
                new_method_deduped.append(row)
                existing_method_keys.add(key)

        print(f"Method odds after dedup: {len(new_method_deduped)} new rows (skipped {len(method_new_rows) - len(new_method_deduped)} dupes)")

        if new_method_deduped:
            method_fields = list(existing_method.columns)
            with open(METHOD_ODDS_CSV, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=method_fields)
                for row in new_method_deduped:
                    writer.writerow(row)
            print(f"  -> Appended {len(new_method_deduped)} rows to {METHOD_ODDS_CSV}")

    # ── Update missing CSVs ─────────────────────────────────────────────
    if ml_new_rows:
        recovered_ml_keys = set()
        for r in ml_new_rows:
            recovered_ml_keys.add((r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"])))
            recovered_ml_keys.add((r["event_date"], norm(r["fighter_b"]), norm(r["fighter_a"])))

        remaining_ml = [r for r in ml_missing if (r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"])) not in recovered_ml_keys]
        with open(MISSING_ML_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["event_date", "fighter_a", "fighter_b"])
            writer.writeheader()
            for r in remaining_ml:
                writer.writerow({"event_date": r["event_date"], "fighter_a": r["fighter_a"], "fighter_b": r["fighter_b"]})
        print(f"  -> Updated {MISSING_ML_CSV}: {len(ml_missing)} → {len(remaining_ml)} remaining")

    if method_new_rows:
        recovered_method_keys = set()
        for r in method_new_rows:
            recovered_method_keys.add((r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"])))
            recovered_method_keys.add((r["event_date"], norm(r["fighter_b"]), norm(r["fighter_a"])))

        remaining_method = [r for r in method_missing if (r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"])) not in recovered_method_keys]
        with open(MISSING_METHOD_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["event_date", "fighter_a", "fighter_b"])
            writer.writeheader()
            for r in remaining_method:
                writer.writerow({"event_date": r["event_date"], "fighter_a": r["fighter_a"], "fighter_b": r["fighter_b"]})
        print(f"  -> Updated {MISSING_METHOD_CSV}: {len(method_missing)} → {len(remaining_method)} remaining")

    print("\nDone.")


if __name__ == "__main__":
    main()
