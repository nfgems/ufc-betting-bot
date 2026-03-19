"""
Backfill missing method-of-victory odds from BestFightOdds.

Strategy:
1. For each event date, pick a fighter from the missing list
2. Search BFO for that fighter -> get their fighter page URL
3. Scrape fighter page to find the event URL matching the target date
4. Navigate to event page, extract all method props for all fights
5. Match to missing fighters

Usage:
    python scripts/backfill_v5_method_odds_bfo.py [--max-events N] [--start-date YYYY-MM-DD]
"""

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
METHOD_ODDS_PATH = REPO_ROOT / "data" / "raw" / "method_odds" / "historical_method_odds_all.csv"
MISSING_PATH = REPO_ROOT / "data" / "raw" / "v5_missing_method_odds.csv"
CHECKPOINT_PATH = REPO_ROOT / "data" / "raw" / "method_odds" / "v5_bfo_checkpoint.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DELAY = 2.5

# Month name -> number mapping for BFO date parsing
MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def norm(n):
    if pd.isna(n):
        return ""
    n = unicodedata.normalize("NFKD", str(n))
    n = "".join(c for c in n if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-zA-Z\s]", "", n).lower().split())


def last_name(n):
    parts = norm(n).split()
    return parts[-1] if parts else ""


def parse_american(t):
    c = re.sub(r"[\u25b2\u25bc\u2191\u2193\s]", "", str(t).strip())
    m = re.match(r"^([+-]?\d+)$", c)
    return float(m.group(1)) if m else None


def american_to_prob(odds):
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def parse_bfo_date(text):
    """Parse BFO date string like 'Mar 1st 2026' to 'YYYY-MM-DD'."""
    m = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d+)\w*\s+(\d{4})", text, re.IGNORECASE)
    if not m:
        return None
    month = MONTH_MAP.get(m.group(1).lower()[:3])
    day = int(m.group(2))
    year = int(m.group(3))
    if month:
        return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def extract_props_playwright(page):
    """Extract method-of-victory props from a BFO event page using Playwright."""
    rows = page.query_selector_all("tr")
    all_props = {}
    for row in rows:
        cells = row.query_selector_all("td, th")
        if not cells:
            continue
        label = cells[0].inner_text().strip()
        label_lower = label.lower()
        method = None
        if (
            re.search(r"wins by (tko/?ko|ko/?tko|knockout)", label_lower)
            and "round" not in label_lower
            and "via" not in label_lower
        ):
            method = "ko"
        elif re.search(r"wins by sub", label_lower) and "round" not in label_lower:
            method = "sub"
        elif re.search(r"wins by (decision|dec)", label_lower) and not any(
            w in label_lower for w in ["unanimous", "split", "majority"]
        ):
            method = "dec"
        if method:
            fighter = re.sub(r"\s*wins\s+by\s+.*$", "", label, flags=re.IGNORECASE).strip()
            odds_vals = [parse_american(c.inner_text().strip()) for c in cells[1:]]
            odds_vals = [v for v in odds_vals if v is not None]
            if odds_vals:
                fn = norm(fighter)
                if fn not in all_props:
                    all_props[fn] = {"raw": fighter}
                all_props[fn][method] = american_to_prob(np.mean(odds_vals))
    return all_props


def fuzzy_match(fighter_name, all_props):
    """Match a fighter name to BFO extracted props using fuzzy matching."""
    fn = norm(fighter_name)
    ln = last_name(fighter_name)
    if fn in all_props:
        return all_props[fn]
    # Exact last name
    for bfo_name, data in all_props.items():
        bfo_last = bfo_name.split()[-1] if bfo_name.split() else ""
        if bfo_last == ln and ln:
            return data
    # Fuzzy
    best_score, best_match = 0, None
    for bfo_name, data in all_props.items():
        bfo_last = bfo_name.split()[-1] if bfo_name.split() else ""
        score = max(fuzz.ratio(fn, bfo_name), fuzz.ratio(ln, bfo_last) * 1.2)
        if score > best_score:
            best_score = score
            best_match = data
    if best_score >= 65:
        return best_match
    # Substring
    for bfo_name, data in all_props.items():
        if ln and ln in bfo_name:
            return data
    return None


def find_fighter_page_url(fighter_name):
    """Search BFO for a fighter and return their fighter page URL."""
    try:
        resp = requests.get(
            "https://www.bestfightodds.com/search",
            params={"query": fighter_name},
            headers=HEADERS,
            timeout=20,
        )
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "lxml")
        fn = norm(fighter_name)
        ln = last_name(fighter_name)

        # Look for fighter page links
        candidates = []
        for link in soup.select("a[href*='/fighters/']"):
            href = link.get("href", "")
            text = link.get_text(strip=True)
            if "/fighters/" in href and text:
                link_norm = norm(text)
                # Score match
                score = fuzz.ratio(fn, link_norm)
                ln_score = fuzz.ratio(ln, link_norm.split()[-1] if link_norm.split() else "")
                best = max(score, ln_score * 1.2)
                if best >= 70:
                    full = f"https://www.bestfightodds.com{href}" if not href.startswith("http") else href
                    candidates.append((best, text, full))

        if candidates:
            candidates.sort(key=lambda x: -x[0])
            return candidates[0][2]
        return None
    except Exception as e:
        print(f"    Fighter search error for '{fighter_name}': {e}")
        return None


def find_event_url_from_fighter_page(fighter_page_url, target_date):
    """
    Scrape a BFO fighter page and find the event URL matching the target date.
    Returns the event URL or None.
    """
    try:
        resp = requests.get(fighter_page_url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "lxml")

        # Parse the table rows which contain event dates and links
        table = soup.select_one("table")
        if not table:
            return None

        for row in table.select("tr"):
            event_link = row.select_one("a[href*='/events/']")
            if not event_link:
                continue
            row_text = row.get_text(" ", strip=True)
            parsed_date = parse_bfo_date(row_text)
            if parsed_date == target_date:
                href = event_link["href"]
                return f"https://www.bestfightodds.com{href}" if not href.startswith("http") else href

        # Try +/- 1 day tolerance (BFO dates can differ by timezone)
        from datetime import timedelta
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        for delta in [1, -1]:
            alt_date = (target_dt + timedelta(days=delta)).strftime("%Y-%m-%d")
            for row in table.select("tr"):
                event_link = row.select_one("a[href*='/events/']")
                if not event_link:
                    continue
                row_text = row.get_text(" ", strip=True)
                parsed_date = parse_bfo_date(row_text)
                if parsed_date == alt_date:
                    href = event_link["href"]
                    return f"https://www.bestfightodds.com{href}" if not href.startswith("http") else href

        return None
    except Exception as e:
        print(f"    Fighter page parse error: {e}")
        return None


def discover_event_url(target_date, fighters):
    """
    Discover the BFO event URL for a given date by checking fighter pages.
    Try up to 3 fighters until we find the event URL.
    """
    tried_fighters = set()
    # Try fighters from multiple matchups
    all_fighters = []
    for f_a, f_b in fighters:
        all_fighters.extend([f_a, f_b])

    for fighter in all_fighters[:6]:
        fn = norm(fighter)
        if fn in tried_fighters:
            continue
        tried_fighters.add(fn)

        # Search for fighter
        fighter_url = find_fighter_page_url(fighter)
        time.sleep(DELAY)
        if not fighter_url:
            continue

        # Get event URL from fighter page
        event_url = find_event_url_from_fighter_page(fighter_url, target_date)
        time.sleep(DELAY)
        if event_url:
            return event_url

    return None


def scrape_event_props(page, url):
    """Navigate to a BFO event page and extract method props."""
    try:
        page.goto(url, timeout=25000)
        page.wait_for_timeout(3000)
        props = extract_props_playwright(page)
        return props
    except Exception as e:
        print(f"    Scrape error: {e}")
        return {}


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"processed_dates": [], "recovered": [], "event_urls": {},
            "stats": {"total_recovered": 0, "total_missed": 0}}


def save_checkpoint(ckpt):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=0, help="Max events to process (0=all)")
    parser.add_argument("--start-date", type=str, default="", help="Start from this date")
    args = parser.parse_args()

    # Load missing fights
    missing = pd.read_csv(MISSING_PATH)
    print(f"Total missing method odds: {len(missing)}")

    # Group by event_date
    event_groups = defaultdict(list)
    for _, r in missing.iterrows():
        event_groups[r["event_date"]].append((r["fighter_a"], r["fighter_b"]))

    dates = sorted(event_groups.keys())
    if args.start_date:
        dates = [d for d in dates if d >= args.start_date]

    # Load checkpoint
    ckpt = load_checkpoint()
    processed = set(ckpt["processed_dates"])
    all_recovered = ckpt["recovered"]
    cached_urls = ckpt.get("event_urls", {})

    dates_to_process = [d for d in dates if d not in processed]
    if args.max_events > 0:
        dates_to_process = dates_to_process[: args.max_events]

    print(f"Dates to process: {len(dates_to_process)} (of {len(dates)} total)")
    print(f"Already processed: {len(processed)}")
    print(f"Already recovered: {len(all_recovered)}")
    print()

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for idx, event_date in enumerate(dates_to_process):
            fighters = event_groups[event_date]
            print(f"\n[{idx+1}/{len(dates_to_process)}] {event_date}: {len(fighters)} fights")

            # Check cached URL first
            event_url = cached_urls.get(event_date)
            if not event_url:
                event_url = discover_event_url(event_date, fighters)
                if event_url:
                    cached_urls[event_date] = event_url
                    ckpt["event_urls"] = cached_urls

            if not event_url:
                print(f"  Could not find BFO event page")
                ckpt["processed_dates"].append(event_date)
                ckpt["stats"]["total_missed"] += len(fighters)
                save_checkpoint(ckpt)
                continue

            print(f"  Event URL: {event_url.split('/')[-1]}")

            # Scrape event page for method props
            props = scrape_event_props(page, event_url)
            time.sleep(DELAY)

            if not props:
                print(f"  No method props on event page")
                ckpt["processed_dates"].append(event_date)
                ckpt["stats"]["total_missed"] += len(fighters)
                save_checkpoint(ckpt)
                continue

            print(f"  Found props for {len(props)} fighters")

            # Match fighters to props
            event_recovered = 0
            event_missed = 0
            now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

            for f_a, f_b in fighters:
                a_data = fuzzy_match(f_a, props)
                b_data = fuzzy_match(f_b, props)

                a_ko = a_data.get("ko") if a_data else None
                a_sub = a_data.get("sub") if a_data else None
                a_dec = a_data.get("dec") if a_data else None
                b_ko = b_data.get("ko") if b_data else None
                b_sub = b_data.get("sub") if b_data else None
                b_dec = b_data.get("dec") if b_data else None

                has_any = any(v is not None for v in [a_ko, a_sub, a_dec, b_ko, b_sub, b_dec])
                if has_any:
                    rec = {
                        "event_date": event_date,
                        "fighter_a": f_a,
                        "fighter_b": f_b,
                        "event_title": "",
                        "source": "bfo_v5_backfill",
                        "source_url": event_url,
                        "captured_at": now_iso,
                        "a_ko_odds_prob": round(a_ko, 6) if a_ko is not None else "",
                        "a_sub_odds_prob": round(a_sub, 6) if a_sub is not None else "",
                        "a_dec_odds_prob": round(a_dec, 6) if a_dec is not None else "",
                        "b_ko_odds_prob": round(b_ko, 6) if b_ko is not None else "",
                        "b_sub_odds_prob": round(b_sub, 6) if b_sub is not None else "",
                        "b_dec_odds_prob": round(b_dec, 6) if b_dec is not None else "",
                    }
                    all_recovered.append(rec)
                    event_recovered += 1
                else:
                    event_missed += 1
                    print(f"    MISS: {f_a} vs {f_b}")

            print(f"  Recovered {event_recovered}/{len(fighters)}")
            ckpt["processed_dates"].append(event_date)
            ckpt["recovered"] = all_recovered
            ckpt["stats"]["total_recovered"] += event_recovered
            ckpt["stats"]["total_missed"] += event_missed
            save_checkpoint(ckpt)

        browser.close()

    # Append all recovered rows to the main method odds CSV
    if all_recovered:
        print(f"\n{'='*60}")
        print(f"Total recovered: {len(all_recovered)}")

        existing = pd.read_csv(METHOD_ODDS_PATH, low_memory=False)
        rec_df = pd.DataFrame(all_recovered)

        # Ensure columns match
        for col in existing.columns:
            if col not in rec_df.columns:
                rec_df[col] = ""
        rec_df = rec_df[existing.columns]

        # Deduplicate
        existing_keys = set()
        for _, r in existing.iterrows():
            key = (str(r["event_date"])[:10], norm(r["fighter_a"]), norm(r["fighter_b"]))
            existing_keys.add(key)
            existing_keys.add((str(r["event_date"])[:10], norm(r["fighter_b"]), norm(r["fighter_a"])))

        new_rows = []
        for _, r in rec_df.iterrows():
            key = (str(r["event_date"])[:10], norm(r["fighter_a"]), norm(r["fighter_b"]))
            if key not in existing_keys:
                new_rows.append(r)
                existing_keys.add(key)

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            new_df.to_csv(METHOD_ODDS_PATH, mode="a", header=False, index=False)
            print(f"Appended {len(new_rows)} new rows to {METHOD_ODDS_PATH}")
        else:
            print("All recovered rows already exist in the file.")

        # Update missing CSV
        recovered_keys = set()
        for r in all_recovered:
            recovered_keys.add((r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"])))
            recovered_keys.add((r["event_date"], norm(r["fighter_b"]), norm(r["fighter_a"])))

        still_missing = []
        for _, r in missing.iterrows():
            key = (r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"]))
            if key not in recovered_keys:
                still_missing.append({"event_date": r["event_date"], "fighter_a": r["fighter_a"], "fighter_b": r["fighter_b"]})

        with open(MISSING_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["event_date", "fighter_a", "fighter_b"])
            writer.writeheader()
            for r in still_missing:
                writer.writerow(r)
        print(f"Updated {MISSING_PATH}: {len(missing)} -> {len(still_missing)} remaining")

    print("\nDone.")


if __name__ == "__main__":
    main()
