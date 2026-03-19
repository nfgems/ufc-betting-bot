# -*- coding: utf-8 -*-
"""
Comprehensive BFO scraper using Playwright for ALL remaining gaps.

Strategy:
1. For each missing fight date, find the correct BFO event URL
2. Load the page with Playwright (renders JS-loaded odds)
3. Parse moneyline + method-of-victory (MOV) prop odds
4. Match to missing fights by last name
5. Save to CSV files

Key improvement: uses fighters from ufc-master.csv (who ARE on BFO)
to search for the correct event, not the missing fighters.
"""
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Force flush on every print
import functools
print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
METHOD_ODDS_PATH = REPO_ROOT / "data/raw/method_odds/historical_method_odds_all.csv"
METHOD_MISSING_PATH = REPO_ROOT / "data/raw/method_odds/post2014_method_missing.csv"
MONEYLINE_PATH = REPO_ROOT / "data/raw/historical_odds/historical_odds_pre2022_from_cleaned.csv"
MONEYLINE_MISSING_PATH = REPO_ROOT / "data/raw/historical_odds/pre2022_odds_still_missing.csv"
BFO_URLS_PATH = REPO_ROOT / "data/raw/historical_odds/bfo_event_urls.json"
CHECKPOINT_PATH = REPO_ROOT / "data/raw/method_odds/bfo_all_gaps_checkpoint.json"


def norm(n):
    if pd.isna(n):
        return ""
    n = unicodedata.normalize("NFKD", str(n))
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = re.sub(r"[^a-zA-Z\s]", "", n)
    return " ".join(n.lower().split())


def last_name(n):
    parts = norm(n).split()
    return parts[-1] if parts else ""


def parse_american(text):
    c = re.sub(r"[\u25b2\u25bc\u2191\u2193\s\u2014]", "", str(text).strip())
    m = re.match(r"^([+-]?\d+)$", c)
    return float(m.group(1)) if m else None


def american_to_prob(odds):
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def american_to_decimal(odds):
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return (odds / 100) + 1
    return (100 / abs(odds)) + 1


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"processed_dates": [], "method_recovered": [], "ml_recovered": [],
            "failed_dates": [], "event_urls": {}}


def save_checkpoint(ckpt):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f, indent=2)


def find_event_url(page, date, master_fighters, known_urls):
    """Find BFO event URL for a date using known fighters from master."""
    # Check known URLs first
    for key, url in known_urls.items():
        if date in key:
            return url

    # Search BFO using main event fighters from the master dataset
    for red, blue in master_fighters[:5]:
        # Try the main event fighters first (last fighters in the list = usually main event)
        query = f"{red} {blue}"
        try:
            page.goto(
                f"https://www.bestfightodds.com/search?query={query}",
                wait_until="domcontentloaded", timeout=15000
            )
            time.sleep(1.5)
            links = page.query_selector_all("a[href*='/events/ufc']")
            for link in links:
                href = link.get_attribute("href") or ""
                text = link.inner_text().strip().lower()
                if "/events/ufc" in href:
                    full = href if href.startswith("http") else f"https://www.bestfightodds.com{href}"
                    return full
        except Exception as e:
            print(f"    Search error for '{query}': {e}")
        time.sleep(1.5)

    return None


def parse_event_odds(page, url):
    """Load BFO event page with Playwright, parse all odds."""
    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_selector("table.odds-table td", timeout=15000)
        time.sleep(2)
    except Exception as e:
        print(f"    Page load/wait error: {e}")
        # Try anyway
        time.sleep(3)

    moneyline = {}  # last_name -> {name, avg_american, decimal, n_books}
    method = {}     # (last_name, ko/sub/dec) -> prob

    rows = page.query_selector_all("table.odds-table tr")

    for row in rows:
        cells = row.query_selector_all("td, th")
        if not cells:
            continue

        try:
            name_text = cells[0].inner_text().strip()
        except Exception:
            continue

        if not name_text or len(name_text) < 2:
            continue

        # Collect odds values
        odds_vals = []
        for cell in cells[1:]:
            try:
                val = parse_american(cell.inner_text().strip())
                if val is not None:
                    odds_vals.append(val)
            except Exception:
                pass

        if not odds_vals:
            continue

        name_lower = name_text.lower()

        # Method of victory prop
        method_type = None
        if "by tko" in name_lower or "by ko" in name_lower or "wins by ko" in name_lower:
            method_type = "ko"
        elif "by sub" in name_lower or "wins by sub" in name_lower:
            method_type = "sub"
        elif "by dec" in name_lower or "wins by decision" in name_lower:
            method_type = "dec"

        if method_type:
            fighter = re.sub(
                r"\s*(wins\s+)?(by\s+)?(ko/?tko|knockout|submission|decision|points|sub).*$",
                "", name_text, flags=re.IGNORECASE
            ).strip()
            fkey = last_name(fighter)
            avg = np.mean(odds_vals)
            prob = american_to_prob(avg)
            if fkey and prob:
                method[(fkey, method_type)] = prob
            continue

        # Skip non-moneyline props
        skip_keywords = [
            "over", "under", "round", "inside", "distance", "not ",
            "handicap", "scorecard", "action", "unanimous", "split",
            "wins in", "majority", "goes the", "fight ends",
            "total ", "points"
        ]
        if any(kw in name_lower for kw in skip_keywords):
            continue

        # Moneyline row
        fkey = last_name(name_text)
        if fkey and len(fkey) > 1:
            avg = np.mean(odds_vals)
            moneyline[fkey] = {
                "name": name_text,
                "avg_american": avg,
                "decimal": american_to_decimal(avg),
                "n_books": len(odds_vals),
            }

    return moneyline, method


def main():
    # Load missing fights
    method_missing = pd.read_csv(METHOD_MISSING_PATH)
    print(f"Method odds missing: {len(method_missing)} fights")

    try:
        ml_missing = pd.read_csv(MONEYLINE_MISSING_PATH)
        ml_missing_set = set()
        for _, r in ml_missing.iterrows():
            ml_missing_set.add((str(r["event_date"]), norm(str(r["fighter_a"])), norm(str(r["fighter_b"]))))
        print(f"Moneyline missing: {len(ml_missing_set)} fights")
    except Exception:
        ml_missing_set = set()

    # Group method missing by date
    date_groups = defaultdict(list)
    for _, row in method_missing.iterrows():
        date_groups[str(row["event_date"])].append((str(row["fighter_a"]), str(row["fighter_b"])))

    # Load master for headliner lookups
    master = pd.read_csv(REPO_ROOT / "data/raw/ufc-master.csv")
    master["Date"] = pd.to_datetime(master["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Load known BFO URLs
    known_urls = {}
    if BFO_URLS_PATH.exists():
        with open(BFO_URLS_PATH) as f:
            known_urls = json.load(f)

    ckpt = load_checkpoint()
    processed = set(ckpt["processed_dates"])
    all_method = ckpt["method_recovered"]
    all_ml = ckpt["ml_recovered"]
    event_url_cache = ckpt.get("event_urls", {})

    dates_to_process = sorted(d for d in date_groups if d not in processed)
    print(f"Dates to process: {len(dates_to_process)}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = ctx.new_page()

        for di, date in enumerate(dates_to_process):
            fighters = date_groups[date]
            print(f"\n[{di+1}/{len(dates_to_process)}] {date}: {len(fighters)} fights")

            # Get fighters from master for this date (to search BFO)
            date_master = master[master["Date"] == date]
            if date_master.empty:
                print(f"  No master data for {date}, using missing fighters for search")
                master_fighters = [(fa, fb) for fa, fb in fighters]
            else:
                # Reverse order so main event is first
                master_fighters = [
                    (str(r["RedFighter"]), str(r["BlueFighter"]))
                    for _, r in date_master.iloc[::-1].iterrows()
                ]

            # Find event URL
            if date in event_url_cache:
                event_url = event_url_cache[date]
                print(f"  Cached URL: {event_url}")
            else:
                event_url = find_event_url(page, date, master_fighters, known_urls)
                if event_url:
                    event_url_cache[date] = event_url
                    print(f"  Found URL: {event_url}")
                else:
                    print(f"  No event URL found")
                    ckpt["processed_dates"].append(date)
                    ckpt["failed_dates"].append(date)
                    ckpt["event_urls"] = event_url_cache
                    save_checkpoint(ckpt)
                    continue

            # Parse odds from the event page
            ml_data, method_data = parse_event_odds(page, event_url)
            print(f"  Parsed: {len(ml_data)} moneyline, {len(method_data)} method props")

            # Verify we have the right event by checking fighter overlap
            needed_lasts = set()
            for fa, fb in fighters:
                needed_lasts.add(last_name(fa))
                needed_lasts.add(last_name(fb))

            found_lasts = set(ml_data.keys())
            overlap = needed_lasts & found_lasts
            if not overlap:
                print(f"  WARNING: No fighter overlap! Wrong event page.")
                # Try to re-search with our missing fighters
                event_url2 = find_event_url(page, date, fighters, {})
                if event_url2 and event_url2 != event_url:
                    print(f"  Retrying with: {event_url2}")
                    ml_data, method_data = parse_event_odds(page, event_url2)
                    event_url_cache[date] = event_url2
                    event_url = event_url2
                    found_lasts = set(ml_data.keys())
                    overlap = needed_lasts & found_lasts
                    print(f"  Retry parsed: {len(ml_data)} moneyline, {len(method_data)} method props, overlap={len(overlap)}")

            # Match to missing fights
            method_count = 0
            ml_count = 0

            for fa, fb in fighters:
                fa_last = last_name(fa)
                fb_last = last_name(fb)

                # Method odds
                a_ko = method_data.get((fa_last, "ko"))
                a_sub = method_data.get((fa_last, "sub"))
                a_dec = method_data.get((fa_last, "dec"))
                b_ko = method_data.get((fb_last, "ko"))
                b_sub = method_data.get((fb_last, "sub"))
                b_dec = method_data.get((fb_last, "dec"))

                found_method = any(v is not None for v in [a_ko, a_sub, a_dec, b_ko, b_sub, b_dec])
                if found_method:
                    all_method.append({
                        "event_date": date,
                        "fighter_a": fa,
                        "fighter_b": fb,
                        "event_title": "",
                        "source": "bfo_playwright",
                        "source_url": event_url,
                        "captured_at": "2026-03-18T00:00:00Z",
                        "a_ko_odds_prob": round(a_ko, 6) if a_ko else "",
                        "a_sub_odds_prob": round(a_sub, 6) if a_sub else "",
                        "a_dec_odds_prob": round(a_dec, 6) if a_dec else "",
                        "b_ko_odds_prob": round(b_ko, 6) if b_ko else "",
                        "b_sub_odds_prob": round(b_sub, 6) if b_sub else "",
                        "b_dec_odds_prob": round(b_dec, 6) if b_dec else "",
                    })
                    method_count += 1
                    print(f"    METHOD: {fa} vs {fb}")

                # Moneyline
                ml_key = (date, norm(fa), norm(fb))
                if ml_key in ml_missing_set:
                    a_ml = ml_data.get(fa_last)
                    b_ml = ml_data.get(fb_last)
                    if a_ml and b_ml and a_ml["decimal"] and b_ml["decimal"]:
                        ai = 1.0 / a_ml["decimal"]
                        bi = 1.0 / b_ml["decimal"]
                        t = ai + bi
                        all_ml.append({
                            "event_date": date,
                            "fighter_a": fa,
                            "fighter_b": fb,
                            "query_date": date,
                            "offset_days": 0,
                            "a_fair_prob": round(ai / t, 6),
                            "b_fair_prob": round(bi / t, 6),
                            "a_decimal_odds": round(a_ml["decimal"], 6),
                            "b_decimal_odds": round(b_ml["decimal"], 6),
                            "num_bookmakers": min(a_ml["n_books"], b_ml["n_books"]),
                        })
                        ml_count += 1
                        print(f"    ML:     {fa} vs {fb}")

                if not found_method and ml_key not in ml_missing_set:
                    print(f"    MISS:   {fa} vs {fb}")

            ckpt["processed_dates"].append(date)
            ckpt["method_recovered"] = all_method
            ckpt["ml_recovered"] = all_ml
            ckpt["event_urls"] = event_url_cache
            save_checkpoint(ckpt)
            print(f"  Total: {method_count} method, {ml_count} ML from {date}")
            time.sleep(3)

        browser.close()

    # Save results
    if all_method:
        existing = pd.read_csv(METHOD_ODDS_PATH)
        combined = pd.concat([existing, pd.DataFrame(all_method)], ignore_index=True)
        combined = combined.drop_duplicates(subset=["event_date", "fighter_a", "fighter_b"], keep="last")
        combined.to_csv(METHOD_ODDS_PATH, index=False)
        print(f"\nMethod odds: +{len(all_method)} new, {len(combined)} total")

    if all_ml:
        existing_ml = pd.read_csv(MONEYLINE_PATH)
        combined_ml = pd.concat([existing_ml, pd.DataFrame(all_ml)], ignore_index=True)
        combined_ml = combined_ml.drop_duplicates(subset=["event_date", "fighter_a", "fighter_b"], keep="last")
        combined_ml.to_csv(MONEYLINE_PATH, index=False)
        print(f"Moneyline: +{len(all_ml)} new, {len(combined_ml)} total")

    failed = ckpt.get("failed_dates", [])
    print(f"\nProcessed: {len(ckpt['processed_dates'])} dates, Failed: {len(failed)}")
    if failed:
        print(f"Failed dates: {failed}")


if __name__ == "__main__":
    main()
