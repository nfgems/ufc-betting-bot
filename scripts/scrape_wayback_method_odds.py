"""
Scrape BFO via Wayback Machine for missing method odds (KO/Sub/Dec probabilities).

BFO event pages have moneyline tables. Method-of-victory (MOV) odds are on separate
prop pages, typically at /events/EVENT-SLUG and there may be a "Prop" or "Method"
tab. We also try BFO directly first.

Groups fights by event date to minimize requests.
"""
import csv
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parent.parent
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
METHOD_ODDS_PATH = REPO_ROOT / "data/raw/method_odds/historical_method_odds_all.csv"
MISSING_PATH = REPO_ROOT / "data/raw/method_odds/post2014_method_missing.csv"
MONEYLINE_PATH = REPO_ROOT / "data/raw/historical_odds/historical_odds_pre2022_from_cleaned.csv"
CHECKPOINT_PATH = REPO_ROOT / "data/raw/method_odds/wayback_scrape_checkpoint.json"
REQUEST_DELAY = 3.0  # seconds between archive.org requests
BFO_DELAY = 2.0  # seconds between BFO requests
MAX_RETRIES = 2


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


def parse_american(t):
    """Parse American odds from text like +150 or -200."""
    c = re.sub(r"[\u25b2\u25bc\u2191\u2193\s]", "", str(t).strip())
    m = re.match(r"^([+-]?\d+)$", c)
    return float(m.group(1)) if m else None


def american_to_prob(odds):
    """Convert American odds to implied probability."""
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def search_bfo_events(query):
    """Search BFO for event URLs matching a query."""
    urls = set()
    try:
        resp = requests.get(
            "https://www.bestfightodds.com/search",
            params={"query": query},
            headers=HEADERS,
            timeout=20,
        )
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "lxml")
            for link in soup.select("a[href*='/events/']"):
                href = link.get("href", "")
                if "/events/" in href and href not in ("/events", "/events/"):
                    full = href if href.startswith("http") else f"https://www.bestfightodds.com{href}"
                    urls.add(full)
    except Exception as e:
        print(f"    BFO search error: {e}")
    return urls


def fetch_wayback(url, retries=MAX_RETRIES):
    """Find and fetch a Wayback Machine snapshot."""
    for attempt in range(retries):
        try:
            cdx = requests.get(
                "https://web.archive.org/cdx/search/cdx",
                params={
                    "url": url,
                    "output": "json",
                    "limit": 5,
                    "fl": "timestamp,statuscode",
                    "filter": "statuscode:200",
                },
                headers=HEADERS,
                timeout=30,
            )
            if cdx.status_code == 429:
                print("    Rate limited by archive.org. Waiting 60s...")
                time.sleep(60)
                continue
            if cdx.status_code != 200:
                return None
            data = cdx.json()
            if len(data) <= 1:
                return None
            ts = data[1][0]
            time.sleep(REQUEST_DELAY)
            wb = requests.get(
                f"https://web.archive.org/web/{ts}/{url}",
                headers=HEADERS,
                timeout=45,
            )
            if wb.status_code == 429:
                print("    Rate limited by archive.org. Waiting 60s...")
                time.sleep(60)
                continue
            return wb.text if wb.status_code == 200 else None
        except Exception as e:
            print(f"    Wayback fetch error: {e}")
            time.sleep(REQUEST_DELAY * 2)
    return None


def fetch_bfo_direct(url):
    """Try fetching BFO page directly (they may still serve historical pages)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None


def parse_bfo_event_page(html):
    """
    Parse ALL odds tables from a BFO event page.
    Returns list of dicts with fighter names and odds (moneyline and/or prop).
    BFO pages may have multiple tables: moneyline, method of victory props, etc.
    """
    soup = BeautifulSoup(html, "lxml")
    results = []

    # Find all odds tables
    tables = soup.select("table.odds-table")

    for table in tables:
        # Check if this table has a header indicating method of victory
        prev = table.find_previous(["h2", "h3", "h4", "div"])
        table_type = "unknown"
        if prev:
            header_text = prev.get_text(strip=True).lower()
            if "method" in header_text or "prop" in header_text or "victory" in header_text:
                table_type = "method"
            elif "fight" in header_text or "moneyline" in header_text or "odds" in header_text:
                table_type = "moneyline"

        rows = table.select("tr")
        i = 0
        while i < len(rows):
            row = rows[i]
            if "pr" in (row.get("class", []) or []):
                i += 1
                continue
            cells = row.select("td, th")
            if not cells or not cells[0].get_text(strip=True):
                i += 1
                continue

            # Check if this looks like a fighter name row
            name_text = cells[0].get_text(strip=True)
            if not name_text or len(name_text) < 2:
                i += 1
                continue

            # Get odds values from this row
            odds_vals = []
            for cell in cells[1:]:
                val = parse_american(cell.get_text(strip=True))
                if val is not None:
                    odds_vals.append(val)

            if odds_vals:
                results.append({
                    "name": name_text,
                    "odds": odds_vals,
                    "avg_odds": np.mean(odds_vals),
                    "table_type": table_type,
                })

            i += 1

    return results


def parse_method_props_from_rows(all_rows):
    """
    Try to identify method-of-victory props from parsed odds rows.
    BFO method props look like:
      "Fighter Name by KO/TKO"    +250
      "Fighter Name by Submission" +450
      "Fighter Name by Decision"   +300

    Returns dict: {(fighter_last_name, method): implied_probability}
    """
    method_odds = {}
    for row in all_rows:
        name = row["name"]
        name_lower = name.lower()

        method = None
        if "ko" in name_lower or "tko" in name_lower or "knockout" in name_lower:
            method = "ko"
        elif "sub" in name_lower or "submission" in name_lower:
            method = "sub"
        elif "dec" in name_lower or "decision" in name_lower or "points" in name_lower:
            method = "dec"

        if method:
            # Extract fighter name (remove method part)
            fighter = re.sub(
                r"\s*(by\s+)?(ko/?tko|knockout|submission|decision|points|sub).*$",
                "", name, flags=re.IGNORECASE
            ).strip()
            fighter_key = last_name(fighter)
            prob = american_to_prob(row["avg_odds"])
            if fighter_key and prob:
                method_odds[(fighter_key, method)] = prob

    return method_odds


def find_event_urls_for_date(fighters_on_date):
    """Search BFO for event URLs using fighter names from a specific date."""
    all_urls = set()

    # Try searching by pairs of fighter last names
    for fa, fb in fighters_on_date[:3]:  # Limit searches
        query = f"{fa} {fb}"
        urls = search_bfo_events(query)
        all_urls.update(urls)
        time.sleep(BFO_DELAY)
        if urls:
            break  # Found the event

    # If no results, try individual fighter names
    if not all_urls:
        for fa, fb in fighters_on_date[:2]:
            for name in [fa, fb]:
                urls = search_bfo_events(name)
                all_urls.update(urls)
                time.sleep(BFO_DELAY)
                if urls:
                    break
            if all_urls:
                break

    return all_urls


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"processed_dates": [], "recovered": [], "failed_dates": []}


def save_checkpoint(ckpt):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f, indent=2)


def main():
    missing = pd.read_csv(MISSING_PATH)
    print(f"Total missing method odds fights: {len(missing)}")

    # Group by date
    date_groups = defaultdict(list)
    for _, row in missing.iterrows():
        date = str(row["event_date"])
        date_groups[date].append((str(row["fighter_a"]), str(row["fighter_b"])))

    print(f"Unique event dates: {len(date_groups)}")

    ckpt = load_checkpoint()
    processed = set(ckpt["processed_dates"])
    all_recovered = ckpt["recovered"]

    dates_to_process = sorted(d for d in date_groups if d not in processed)
    print(f"Dates remaining: {len(dates_to_process)}")

    request_count = 0
    max_requests = 200  # Safety limit

    for date_idx, date in enumerate(dates_to_process):
        if request_count >= max_requests:
            print(f"\n  Reached {max_requests} request limit. Saving checkpoint and stopping.")
            break

        fighters = date_groups[date]
        print(f"\n[{date_idx+1}/{len(dates_to_process)}] {date}: {len(fighters)} fights to find")

        # Step 1: Search BFO for event URLs
        event_urls = find_event_urls_for_date(fighters)
        request_count += min(3, len(fighters))

        if not event_urls:
            print(f"  No BFO event URLs found for {date}")
            ckpt["processed_dates"].append(date)
            ckpt["failed_dates"].append(date)
            save_checkpoint(ckpt)
            continue

        print(f"  Found {len(event_urls)} event URLs")

        # Step 2: Try BFO direct first, then Wayback
        html = None
        used_url = None
        for url in list(event_urls)[:3]:
            # Try direct BFO
            html = fetch_bfo_direct(url)
            request_count += 1
            if html:
                used_url = url
                print(f"  Direct BFO success: {url}")
                break
            time.sleep(BFO_DELAY)

            # Try Wayback
            html = fetch_wayback(url)
            request_count += 2  # CDX + fetch
            if html:
                used_url = url
                print(f"  Wayback success: {url}")
                break
            time.sleep(REQUEST_DELAY)

            if request_count >= max_requests:
                break

        if not html:
            print(f"  Could not fetch any page for {date}")
            ckpt["processed_dates"].append(date)
            ckpt["failed_dates"].append(date)
            save_checkpoint(ckpt)
            continue

        # Step 3: Parse the page
        all_rows = parse_bfo_event_page(html)
        method_odds = parse_method_props_from_rows(all_rows)

        if method_odds:
            print(f"  Found {len(method_odds)} method prop entries")

        # Step 4: Also try the prop sub-page if it exists
        # BFO prop pages are often at the same URL with /prop or similar
        prop_urls_to_try = []
        if used_url:
            base = used_url.rstrip("/")
            prop_urls_to_try.extend([
                base + "/prop",
                base.replace("/events/", "/events/method/"),
            ])

        for prop_url in prop_urls_to_try:
            if request_count >= max_requests:
                break
            prop_html = fetch_bfo_direct(prop_url)
            request_count += 1
            if not prop_html:
                prop_html = fetch_wayback(prop_url)
                request_count += 2
            if prop_html:
                prop_rows = parse_bfo_event_page(prop_html)
                prop_method = parse_method_props_from_rows(prop_rows)
                if prop_method:
                    method_odds.update(prop_method)
                    print(f"  Found {len(prop_method)} additional method props from prop page")
            time.sleep(REQUEST_DELAY)

        # Step 5: Also parse moneyline-style method odds from the main page
        # Some BFO pages have method odds as separate fight rows like:
        #   "Fighter A"  odds
        #   "Fighter B"  odds
        # followed by prop rows for the same fighters

        # Step 6: Match found odds to missing fights
        date_recovered = 0
        for fa, fb in fighters:
            fa_last = last_name(fa)
            fb_last = last_name(fb)

            a_ko = method_odds.get((fa_last, "ko"))
            a_sub = method_odds.get((fa_last, "sub"))
            a_dec = method_odds.get((fa_last, "dec"))
            b_ko = method_odds.get((fb_last, "ko"))
            b_sub = method_odds.get((fb_last, "sub"))
            b_dec = method_odds.get((fb_last, "dec"))

            # Check if we found at least some odds
            found_any = any(v is not None for v in [a_ko, a_sub, a_dec, b_ko, b_sub, b_dec])
            if found_any:
                rec = {
                    "event_date": date,
                    "fighter_a": fa,
                    "fighter_b": fb,
                    "event_title": "",
                    "source": "wayback_bfo",
                    "source_url": used_url or "",
                    "captured_at": "2026-03-18T00:00:00Z",
                    "a_ko_odds_prob": round(a_ko, 6) if a_ko else "",
                    "a_sub_odds_prob": round(a_sub, 6) if a_sub else "",
                    "a_dec_odds_prob": round(a_dec, 6) if a_dec else "",
                    "b_ko_odds_prob": round(b_ko, 6) if b_ko else "",
                    "b_sub_odds_prob": round(b_sub, 6) if b_sub else "",
                    "b_dec_odds_prob": round(b_dec, 6) if b_dec else "",
                }
                all_recovered.append(rec)
                date_recovered += 1
                print(f"    FOUND: {fa} vs {fb}")
            else:
                print(f"    MISS:  {fa} vs {fb}")

        ckpt["processed_dates"].append(date)
        ckpt["recovered"] = all_recovered
        save_checkpoint(ckpt)
        print(f"  Recovered {date_recovered}/{len(fighters)} from {date}")

    # Save all recovered to CSV
    if all_recovered:
        existing = pd.read_csv(METHOD_ODDS_PATH)
        new_df = pd.DataFrame(all_recovered)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["event_date", "fighter_a", "fighter_b"], keep="last"
        )
        combined.to_csv(METHOD_ODDS_PATH, index=False)
        print(f"\nTotal recovered this session: {len(all_recovered)}")
        print(f"Method odds file now has {len(combined)} rows")

    print(f"\nTotal requests made: {request_count}")
    print(f"Dates processed: {len(ckpt['processed_dates'])}")
    print(f"Dates failed: {len(ckpt.get('failed_dates', []))}")


if __name__ == "__main__":
    main()
