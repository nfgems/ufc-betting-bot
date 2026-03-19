"""
Scrape BFO using Playwright (headless browser) for method-of-victory odds.

BFO loads odds via JavaScript, so we need a real browser to render the page.
This script:
1. Groups missing fights by event date
2. Searches BFO for the correct event URL
3. Opens the event page in a headless browser
4. Waits for odds to load
5. Parses moneyline + method-of-victory odds
6. Saves results to the method odds CSV and moneyline CSV
"""
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

# Fix Windows encoding issues with Unicode characters
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
METHOD_ODDS_PATH = REPO_ROOT / "data/raw/method_odds/historical_method_odds_all.csv"
MISSING_PATH = REPO_ROOT / "data/raw/method_odds/post2014_method_missing.csv"
MONEYLINE_PATH = REPO_ROOT / "data/raw/historical_odds/historical_odds_pre2022_from_cleaned.csv"
MONEYLINE_MISSING_PATH = REPO_ROOT / "data/raw/historical_odds/pre2022_odds_still_missing.csv"
CHECKPOINT_PATH = REPO_ROOT / "data/raw/method_odds/bfo_playwright_checkpoint.json"
PAGE_DELAY = 4.0  # seconds between page loads


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
    """Parse American odds from cell text like '+250' or '-150'."""
    text = re.sub(r"[\u25b2\u25bc\u2191\u2193\s]", "", str(text).strip())
    m = re.match(r"^([+-]?\d+)$", text)
    return float(m.group(1)) if m else None


def american_to_prob(odds):
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def american_to_decimal(odds):
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"processed_dates": [], "method_recovered": [], "moneyline_recovered": [], "failed_dates": []}


def save_checkpoint(ckpt):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f, indent=2)


def search_bfo_event_url(page, fighters_on_date):
    """Search BFO for the event URL for fighters on a specific date."""
    for fa, fb in fighters_on_date[:3]:
        query = f"{fa} {fb}"
        try:
            page.goto(f"https://www.bestfightodds.com/search?query={query}", wait_until="domcontentloaded", timeout=15000)
            time.sleep(1)
            links = page.query_selector_all("a[href*='/events/']")
            urls = set()
            for link in links:
                href = link.get_attribute("href")
                if href and "/events/" in href and href not in ("/events", "/events/"):
                    if not href.startswith("http"):
                        href = f"https://www.bestfightodds.com{href}"
                    # Filter to likely UFC events
                    if "ufc" in href.lower():
                        urls.add(href)
            if urls:
                return urls
        except Exception as e:
            print(f"    Search error: {e}")
        time.sleep(1)

    # Broader search
    for fa, fb in fighters_on_date[:2]:
        for name in [fa, fb]:
            try:
                page.goto(f"https://www.bestfightodds.com/search?query={name}", wait_until="domcontentloaded", timeout=15000)
                time.sleep(1)
                links = page.query_selector_all("a[href*='/events/']")
                for link in links:
                    href = link.get_attribute("href")
                    if href and "/events/" in href and "ufc" in href.lower():
                        if not href.startswith("http"):
                            href = f"https://www.bestfightodds.com{href}"
                        return {href}
            except Exception:
                pass
            time.sleep(1)

    return set()


def parse_event_page(page, url, fighters_needed):
    """
    Load BFO event page, wait for odds to render, parse all fights.
    Returns dict of (fighter_a_last, fighter_b_last) -> odds data.
    """
    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
        # Wait for odds cells to appear
        page.wait_for_selector("table.odds-table td", timeout=10000)
        time.sleep(2)  # Extra wait for all cells
    except Exception as e:
        print(f"    Page load error: {e}")
        return {}, {}

    moneyline_data = {}
    method_data = {}

    # Parse all tables
    tables = page.query_selector_all("table.odds-table")

    for table in tables:
        rows = table.query_selector_all("tr")
        i = 0
        while i < len(rows):
            row = rows[i]
            row_class = row.get_attribute("class") or ""

            # Skip separator rows
            if "pointed" in row_class or "pointed" in row_class:
                i += 1
                continue

            cells = row.query_selector_all("td, th")
            if not cells:
                i += 1
                continue

            name_text = cells[0].inner_text().strip()
            if not name_text or len(name_text) < 2:
                i += 1
                continue

            # Collect odds from all cells
            odds_vals = []
            for cell in cells[1:]:
                cell_text = cell.inner_text().strip()
                val = parse_american(cell_text)
                if val is not None:
                    odds_vals.append(val)

            # Determine if this is a moneyline row or a prop row
            name_lower = name_text.lower()

            # Check for method of victory props
            method_match = None
            if "by tko" in name_lower or "by ko" in name_lower:
                method_match = "ko"
            elif "by sub" in name_lower:
                method_match = "sub"
            elif "by dec" in name_lower or "by decision" in name_lower:
                method_match = "dec"

            if method_match and odds_vals:
                # Extract fighter name from prop label
                fighter = re.sub(
                    r"\s*(wins\s+)?(by\s+)?(ko/?tko|knockout|submission|decision|points|sub).*$",
                    "", name_text, flags=re.IGNORECASE
                ).strip()
                fighter_key = last_name(fighter)
                avg_odds = np.mean(odds_vals)
                prob = american_to_prob(avg_odds)
                if fighter_key and prob:
                    method_data[(fighter_key, method_match)] = prob

            elif odds_vals and not any(kw in name_lower for kw in [
                "over", "under", "round", "inside", "distance", "not ",
                "handicap", "scorecard", "action", "unanimous", "split",
                "wins in", "majority"
            ]):
                # This looks like a moneyline row
                fighter_key = last_name(name_text)
                avg_odds = np.mean(odds_vals)
                moneyline_data[fighter_key] = {
                    "name": name_text,
                    "avg_american": avg_odds,
                    "decimal": american_to_decimal(avg_odds),
                    "n_books": len(odds_vals),
                }

            i += 1

    return moneyline_data, method_data


def main():
    missing_method = pd.read_csv(MISSING_PATH)
    print(f"Total missing method odds fights: {len(missing_method)}")

    # Also load moneyline missing
    try:
        missing_ml = pd.read_csv(MONEYLINE_MISSING_PATH)
        ml_missing_set = {
            (str(r["event_date"]), norm(str(r["fighter_a"])), norm(str(r["fighter_b"])))
            for _, r in missing_ml.iterrows()
        }
    except Exception:
        ml_missing_set = set()

    # Group by date
    date_groups = defaultdict(list)
    for _, row in missing_method.iterrows():
        date = str(row["event_date"])
        date_groups[date].append((str(row["fighter_a"]), str(row["fighter_b"])))

    print(f"Unique event dates: {len(date_groups)}")

    ckpt = load_checkpoint()
    processed = set(ckpt["processed_dates"])
    all_method_recovered = ckpt["method_recovered"]
    all_ml_recovered = ckpt["moneyline_recovered"]

    dates_to_process = sorted(d for d in date_groups if d not in processed)
    print(f"Dates remaining: {len(dates_to_process)}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = context.new_page()

        for date_idx, date in enumerate(dates_to_process):
            fighters = date_groups[date]
            print(f"\n[{date_idx+1}/{len(dates_to_process)}] {date}: {len(fighters)} fights")

            # Search for event URL
            event_urls = search_bfo_event_url(page, fighters)
            if not event_urls:
                print(f"  No BFO event URLs found")
                ckpt["processed_dates"].append(date)
                ckpt["failed_dates"].append(date)
                save_checkpoint(ckpt)
                continue

            # Try each URL until we find one with matching fighters
            best_ml = {}
            best_method = {}
            best_url = None

            for url in list(event_urls)[:3]:
                print(f"  Trying: {url}")
                ml_data, method_data = parse_event_page(page, url, fighters)

                # Check if this page has our fighters
                all_fighter_lasts = set()
                for fa, fb in fighters:
                    all_fighter_lasts.add(last_name(fa))
                    all_fighter_lasts.add(last_name(fb))

                ml_keys = set(ml_data.keys())
                overlap = all_fighter_lasts & ml_keys
                if len(overlap) >= 1:
                    best_ml = ml_data
                    best_method = method_data
                    best_url = url
                    print(f"  Matched {len(overlap)} fighters, {len(ml_data)} moneyline, {len(method_data)} method props")
                    break

                time.sleep(PAGE_DELAY)

            if not best_url:
                print(f"  No matching event page found")
                ckpt["processed_dates"].append(date)
                ckpt["failed_dates"].append(date)
                save_checkpoint(ckpt)
                continue

            # Match to missing fights
            date_method_count = 0
            date_ml_count = 0
            for fa, fb in fighters:
                fa_last = last_name(fa)
                fb_last = last_name(fb)

                # Method odds
                a_ko = best_method.get((fa_last, "ko"))
                a_sub = best_method.get((fa_last, "sub"))
                a_dec = best_method.get((fa_last, "dec"))
                b_ko = best_method.get((fb_last, "ko"))
                b_sub = best_method.get((fb_last, "sub"))
                b_dec = best_method.get((fb_last, "dec"))

                found_method = any(v is not None for v in [a_ko, a_sub, a_dec, b_ko, b_sub, b_dec])
                if found_method:
                    rec = {
                        "event_date": date,
                        "fighter_a": fa,
                        "fighter_b": fb,
                        "event_title": "",
                        "source": "bfo_playwright",
                        "source_url": best_url,
                        "captured_at": "2026-03-18T00:00:00Z",
                        "a_ko_odds_prob": round(a_ko, 6) if a_ko else "",
                        "a_sub_odds_prob": round(a_sub, 6) if a_sub else "",
                        "a_dec_odds_prob": round(a_dec, 6) if a_dec else "",
                        "b_ko_odds_prob": round(b_ko, 6) if b_ko else "",
                        "b_sub_odds_prob": round(b_sub, 6) if b_sub else "",
                        "b_dec_odds_prob": round(b_dec, 6) if b_dec else "",
                    }
                    all_method_recovered.append(rec)
                    date_method_count += 1
                    print(f"    METHOD: {fa} vs {fb}")

                # Moneyline (if also missing)
                ml_key = (date, norm(fa), norm(fb))
                if ml_key in ml_missing_set:
                    a_ml = best_ml.get(fa_last)
                    b_ml = best_ml.get(fb_last)
                    if a_ml and b_ml:
                        ad = a_ml["decimal"]
                        bd = b_ml["decimal"]
                        if ad and bd:
                            ai, bi = 1.0 / ad, 1.0 / bd
                            t = ai + bi
                            ml_rec = {
                                "event_date": date,
                                "fighter_a": fa,
                                "fighter_b": fb,
                                "query_date": date,
                                "offset_days": 0,
                                "a_fair_prob": round(ai / t, 6),
                                "b_fair_prob": round(bi / t, 6),
                                "a_decimal_odds": round(ad, 6),
                                "b_decimal_odds": round(bd, 6),
                                "num_bookmakers": min(a_ml["n_books"], b_ml["n_books"]),
                            }
                            all_ml_recovered.append(ml_rec)
                            date_ml_count += 1
                            print(f"    ML:     {fa} vs {fb}")

                if not found_method:
                    print(f"    MISS:   {fa} vs {fb}")

            ckpt["processed_dates"].append(date)
            ckpt["method_recovered"] = all_method_recovered
            ckpt["moneyline_recovered"] = all_ml_recovered
            save_checkpoint(ckpt)
            print(f"  Recovered: {date_method_count} method, {date_ml_count} moneyline")

            time.sleep(PAGE_DELAY)

        browser.close()

    # Save method odds
    if all_method_recovered:
        existing = pd.read_csv(METHOD_ODDS_PATH)
        new_df = pd.DataFrame(all_method_recovered)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["event_date", "fighter_a", "fighter_b"], keep="last"
        )
        combined.to_csv(METHOD_ODDS_PATH, index=False)
        print(f"\nMethod odds: recovered {len(all_method_recovered)}, total {len(combined)}")

    # Save moneyline
    if all_ml_recovered:
        existing_ml = pd.read_csv(MONEYLINE_PATH)
        new_ml = pd.DataFrame(all_ml_recovered)
        combined_ml = pd.concat([existing_ml, new_ml], ignore_index=True)
        combined_ml = combined_ml.drop_duplicates(
            subset=["event_date", "fighter_a", "fighter_b"], keep="last"
        )
        combined_ml.to_csv(MONEYLINE_PATH, index=False)
        print(f"Moneyline: recovered {len(all_ml_recovered)}, total {len(combined_ml)}")

    print(f"\nDone. Processed {len(ckpt['processed_dates'])} dates.")
    print(f"Failed dates: {len(ckpt.get('failed_dates', []))}")


if __name__ == "__main__":
    main()
