# -*- coding: utf-8 -*-
"""
Map all post-2014 missing method-odds fights to their BFO event page URLs.

Steps:
1. Compute missing fights: jansen88 fights (2014+) minus method_odds fights
2. Use cached bfo_event_urls.json where available
3. Use hardcoded fallback URLs for known events that BFO search cannot find
4. Search BFO archive for remaining events, validating by date
5. Save mapping CSV: event_date, fighter1, fighter2, event_name, bfo_url
"""
import json
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parent.parent
JANSEN_PATH = REPO_ROOT / "data/raw/jansen88_ufc_data.csv"
METHOD_ODDS_PATH = REPO_ROOT / "data/raw/method_odds/historical_method_odds_all.csv"
BFO_URLS_PATH = REPO_ROOT / "data/raw/historical_odds/bfo_event_urls.json"
OUTPUT_PATH = REPO_ROOT / "data/raw/method_odds/bfo_event_url_map.csv"

BFO_BASE = "https://www.bestfightodds.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}
DELAY = 2  # seconds between requests

# Hardcoded BFO event URLs for events that the search API has trouble finding.
# These were manually verified by searching for UFC Fight Night numbered events.
HARDCODED_URLS = {
    # date -> BFO URL (manually verified)
    "2022-02-05": f"{BFO_BASE}/events/ufc-fight-night-202-2339",
    "2023-01-14": f"{BFO_BASE}/events/ufc-fight-night-218-2666",
    "2023-03-11": f"{BFO_BASE}/events/ufc-fight-night-221-2719",
    "2023-04-22": f"{BFO_BASE}/events/ufc-fight-night-223-2770",
    "2023-05-13": f"{BFO_BASE}/events/ufc-fight-night-smith-vs-walker-2850",
    "2023-06-17": f"{BFO_BASE}/events/ufc-fight-night-227-2849",
    "2023-06-24": f"{BFO_BASE}/events/ufc-fight-night-228-2848",
    "2023-07-15": f"{BFO_BASE}/events/ufc-fight-night-225-2891",
}


def normalize_name(s):
    """Normalize fighter name for matching: strip accents, lowercase, collapse spaces."""
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-zA-Z\s]", "", s)
    return " ".join(s.lower().split())


def parse_bfo_date(date_text):
    """Parse a BFO date string like 'Sep 10th 2023' into YYYY-MM-DD."""
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_text.strip())
    for fmt in ["%b %d %Y", "%B %d %Y"]:
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def build_search_queries(event_name):
    """Build a prioritized list of search queries for an event."""
    queries = []

    # Extract UFC number if present (e.g., "UFC 293", "UFC 175")
    num_match = re.search(r"UFC\s+(\d{2,3})\b", event_name)
    if num_match:
        queries.append(f"UFC {num_match.group(1)}")

    # Extract headliner names after colon
    colon_match = re.search(r":\s*(.+)", event_name)
    if colon_match:
        headliner = colon_match.group(1).strip()
        queries.append(headliner)

    # For "UFC on FOX" / "UFC on ESPN" events
    fox_match = re.match(r"(UFC on (?:FOX|ESPN))\b", event_name)
    if fox_match and colon_match:
        queries.append(f"{fox_match.group(1)} {colon_match.group(1).strip()}")

    # For UFC Fight Night with a number (e.g., "UFC Fight Night 56")
    fn_num_match = re.search(r"UFC Fight Night\s+(\d+)", event_name)
    if fn_num_match:
        queries.append(f"UFC Fight Night {fn_num_match.group(1)}")

    # For TUF finales
    if "Ultimate Fighter" in event_name:
        tuf_match = re.search(r"(Ultimate Fighter[^:]*)", event_name)
        if tuf_match:
            queries.append(tuf_match.group(1).strip())

    # Full event name as last resort
    queries.append(event_name)

    # Deduplicate while preserving order
    seen = set()
    result = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            result.append(q)
    return result


def compute_missing_fights():
    """Find all 2014+ fights in jansen88 that are NOT in method_odds."""
    print("Loading datasets...")
    jansen = pd.read_csv(JANSEN_PATH)
    method = pd.read_csv(METHOD_ODDS_PATH)

    # Filter to 2014+
    jansen_2014 = jansen[jansen["event_date"] >= "2014-01-01"].copy()
    print(f"  jansen88 2014+ fights: {len(jansen_2014)}")
    print(f"  method_odds fights: {len(method)}")

    # Create normalized keys for matching
    jansen_2014["f1n"] = jansen_2014["fighter1"].apply(normalize_name)
    jansen_2014["f2n"] = jansen_2014["fighter2"].apply(normalize_name)
    jansen_2014["key"] = jansen_2014["event_date"] + "|" + jansen_2014["f1n"] + "|" + jansen_2014["f2n"]
    jansen_2014["key_rev"] = jansen_2014["event_date"] + "|" + jansen_2014["f2n"] + "|" + jansen_2014["f1n"]

    method["fan"] = method["fighter_a"].apply(normalize_name)
    method["fbn"] = method["fighter_b"].apply(normalize_name)
    method["key"] = method["event_date"] + "|" + method["fan"] + "|" + method["fbn"]
    m_keys = set(method["key"])

    missing = jansen_2014[
        ~jansen_2014["key"].isin(m_keys) & ~jansen_2014["key_rev"].isin(m_keys)
    ].copy()

    print(f"  Missing fights (2014+): {len(missing)}")
    return missing[["event_date", "fighter1", "fighter2", "event_name"]].reset_index(drop=True)


def load_cached_urls():
    """Load previously scraped BFO event URLs."""
    if BFO_URLS_PATH.exists():
        with open(BFO_URLS_PATH) as f:
            return json.load(f)
    return {}


def search_bfo_event(session, event_name, event_date):
    """Search BFO for an event URL, validating by date.

    Returns the URL if found, None otherwise.
    """
    queries = build_search_queries(event_name)
    target_dt = datetime.strptime(event_date, "%Y-%m-%d")

    for query in queries:
        url = f"{BFO_BASE}/search?query={requests.utils.quote(query)}"
        try:
            resp = session.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"    Search request failed for '{query}': {e}")
            time.sleep(DELAY)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        # Parse the table rows - BFO search results are in a table
        rows = soup.select("table tr")

        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue

            # Find event link
            link = row.find("a", href=re.compile(r"/events/"))
            if not link:
                continue

            href = link.get("href", "")
            link_text = link.get_text(strip=True)

            # Only consider UFC events
            if "ufc" not in href.lower() and "ufc" not in link_text.lower():
                continue

            # Try to find the date in this row
            row_text = row.get_text()
            date_match = re.search(
                r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)\s+\d{4}",
                row_text
            )
            if date_match:
                parsed_date = parse_bfo_date(date_match.group())
                if parsed_date:
                    parsed_dt = datetime.strptime(parsed_date, "%Y-%m-%d")
                    # Allow +/- 2 day tolerance (multi-day events, timezone differences)
                    day_diff = abs((parsed_dt - target_dt).days)
                    if day_diff <= 2:
                        full_url = href if href.startswith("http") else f"{BFO_BASE}{href}"
                        return full_url

        time.sleep(DELAY)

    return None


def main():
    missing = compute_missing_fights()
    cached_urls = load_cached_urls()

    # Group missing fights by event (date + name)
    events = {}
    for _, row in missing.iterrows():
        key = f"{row['event_date']}|{row['event_name']}"
        if key not in events:
            events[key] = {
                "event_date": row["event_date"],
                "event_name": row["event_name"],
                "fights": [],
            }
        events[key]["fights"].append((row["fighter1"], row["fighter2"]))

    print(f"\nUnique events with missing fights: {len(events)}")

    # Phase 1: Match from cached URLs (strict: date must be in the cached key)
    event_urls = {}
    unresolved = []
    for key, info in events.items():
        matched = False
        for cached_key, cached_url in cached_urls.items():
            if info["event_date"] in cached_key:
                event_urls[key] = cached_url
                matched = True
                break
        if not matched:
            unresolved.append(key)

    print(f"Matched from cache: {len(event_urls)}")

    # Phase 2: Match from hardcoded fallback URLs
    still_unresolved = []
    for key in unresolved:
        info = events[key]
        if info["event_date"] in HARDCODED_URLS:
            event_urls[key] = HARDCODED_URLS[info["event_date"]]
            print(f"  Hardcoded: {info['event_date']} - {info['event_name']} -> {HARDCODED_URLS[info['event_date']]}")
        else:
            still_unresolved.append(key)

    print(f"Matched from hardcoded: {len(unresolved) - len(still_unresolved)}")
    print(f"Need to search BFO: {len(still_unresolved)}")

    # Phase 3: Search BFO for remaining unresolved events
    if still_unresolved:
        print("\nSearching BFO for unresolved events...")
        session = requests.Session()

        for i, key in enumerate(still_unresolved):
            info = events[key]
            print(f"  [{i+1}/{len(still_unresolved)}] {info['event_date']} - {info['event_name']}")

            url = search_bfo_event(session, info["event_name"], info["event_date"])

            if url:
                event_urls[key] = url
                print(f"    -> {url}")
            else:
                print(f"    -> NOT FOUND")

    # Build output rows
    rows = []
    found_count = 0
    not_found_count = 0
    not_found_events = []
    for key, info in events.items():
        bfo_url = event_urls.get(key, "")
        if bfo_url:
            found_count += len(info["fights"])
        else:
            not_found_count += len(info["fights"])
            not_found_events.append(
                f"  {info['event_date']} - {info['event_name']} ({len(info['fights'])} fights)"
            )
        for f1, f2 in info["fights"]:
            rows.append({
                "event_date": info["event_date"],
                "fighter1": f1,
                "fighter2": f2,
                "event_name": info["event_name"],
                "bfo_url": bfo_url,
            })

    result = pd.DataFrame(rows)
    result = result.sort_values(["event_date", "event_name"]).reset_index(drop=True)

    # Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(result)} fight mappings to {OUTPUT_PATH}")
    print(f"  Fights with BFO URL: {found_count}")
    print(f"  Fights without BFO URL: {not_found_count}")
    print(f"  Events with URLs: {sum(1 for v in event_urls.values() if v)}/{len(events)}")

    if not_found_events:
        print(f"\nEvents without BFO URL:")
        for e in not_found_events:
            print(e)


if __name__ == "__main__":
    main()
