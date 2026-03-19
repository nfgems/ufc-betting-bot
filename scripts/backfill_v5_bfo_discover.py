"""
Discover correct BFO event URLs via fighter page navigation, then scrape method props.
Uses Playwright to handle JS-rendered content.
"""

import csv
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8")

from src.config import RAW_DATA_DIR

METHOD_PATH = RAW_DATA_DIR / "method_odds" / "historical_method_odds_all.csv"
MISSING_PATH = RAW_DATA_DIR / "v5_missing_method_odds.csv"

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def norm(n):
    if not n:
        return ""
    n = unicodedata.normalize("NFKD", str(n))
    n = "".join(c for c in n if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-zA-Z\s]", "", n).lower().split())


def parse_american(text):
    cleaned = re.sub(r"[\u25b2\u25bc\u2191\u2193\u2206\s]", "", str(text).strip())
    m = re.match(r"^([+-]?\d+)$", cleaned)
    return float(m.group(1)) if m else None


def american_to_prob(odds):
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def parse_bfo_date(text):
    m = re.search(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d+)\w*\s+(\d{4})",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    month = MONTH_MAP.get(m.group(1).lower()[:3])
    day = int(m.group(2))
    year = int(m.group(3))
    return f"{year:04d}-{month:02d}-{day:02d}" if month else None


def match_fighter(target, prop_names):
    tn = norm(target)
    tl = tn.split()[-1] if tn.split() else ""
    if tn in prop_names:
        return tn
    for pn in prop_names:
        pl = pn.split()[-1] if pn.split() else ""
        if pl == tl and tl and len(tl) > 2:
            return pn
    best_score, best = 0, None
    for pn in prop_names:
        score = SequenceMatcher(None, tn, pn).ratio()
        t_parts = tn.split()
        if len(t_parts) >= 2:
            score = max(
                score,
                SequenceMatcher(None, " ".join(reversed(t_parts)), pn).ratio(),
            )
        if score > best_score:
            best_score = score
            best = pn
    if best_score >= 0.50:
        return best
    for pn in prop_names:
        if tl and len(tl) > 3 and tl in pn:
            return pn
    return None


def extract_props_pw(page):
    """Extract method props using Playwright."""
    rows = page.query_selector_all("tr")
    all_props = {}
    for row in rows:
        cells = row.query_selector_all("td, th")
        if not cells:
            continue
        label = cells[0].inner_text().strip()
        ll = label.lower()
        method = None
        if (
            re.search(r"wins by (tko/?ko|ko/?tko|knockout)", ll)
            and "round" not in ll
            and "via" not in ll
        ):
            method = "ko"
        elif re.search(r"wins by sub", ll) and "round" not in ll:
            method = "sub"
        elif re.search(r"wins by (decision|dec)", ll) and not any(
            w in ll for w in ["unanimous", "split", "majority"]
        ):
            method = "dec"
        if method:
            fighter = re.sub(
                r"\s*wins\s+by\s+.*$", "", label, flags=re.IGNORECASE
            ).strip()
            odds_vals = [parse_american(c.inner_text().strip()) for c in cells[1:]]
            odds_vals = [v for v in odds_vals if v is not None]
            if odds_vals:
                fn = norm(fighter)
                if fn not in all_props:
                    all_props[fn] = {"raw": fighter}
                all_props[fn][method] = american_to_prob(float(np.mean(odds_vals)))
    return all_props


def discover_event_url_pw(page, fighter_name, target_date):
    """Use Playwright to navigate fighter page and find event URL."""
    # Search BFO for fighter
    fn = norm(fighter_name)
    page.goto(
        f"https://www.bestfightodds.com/search?query={fighter_name}",
        timeout=20000,
    )
    page.wait_for_timeout(3000)

    # Find fighter page link
    fighter_url = None
    for link in page.query_selector_all("a[href*='/fighters/']"):
        text = link.inner_text().strip()
        if SequenceMatcher(None, fn, norm(text)).ratio() >= 0.65:
            href = link.get_attribute("href")
            fighter_url = (
                f"https://www.bestfightodds.com{href}"
                if not href.startswith("http")
                else href
            )
            break

    if not fighter_url:
        return None

    # Navigate to fighter page
    page.goto(fighter_url, timeout=20000)
    page.wait_for_timeout(3000)

    # Find event URL matching target date (+/- 2 days)
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    for row in page.query_selector_all("tr"):
        event_link = row.query_selector("a[href*='/events/']")
        if not event_link:
            continue
        row_text = row.inner_text().strip()
        parsed = parse_bfo_date(row_text)
        if parsed:
            parsed_dt = datetime.strptime(parsed, "%Y-%m-%d")
            if abs((parsed_dt - target_dt).days) <= 2:
                href = event_link.get_attribute("href")
                url = (
                    f"https://www.bestfightodds.com{href}"
                    if not href.startswith("http")
                    else href
                )
                # Validate: make sure URL is an actual event, not the homepage
                if "/events/" in url and not url.endswith("bestfightodds.com/"):
                    return url

    return None


def main():
    missing = list(csv.DictReader(open(MISSING_PATH)))
    print(f"Method odds missing: {len(missing)}")

    existing = pd.read_csv(METHOD_PATH, low_memory=False)
    existing_keys = set()
    for _, r in existing.iterrows():
        k = (str(r["event_date"])[:10], norm(r["fighter_a"]), norm(r["fighter_b"]))
        existing_keys.add(k)
        existing_keys.add(
            (str(r["event_date"])[:10], norm(r["fighter_b"]), norm(r["fighter_a"]))
        )

    # Group by date
    targets = defaultdict(list)
    for r in missing:
        d = r["event_date"]
        if d >= "2025-07-01":
            targets[d].append((r["fighter_a"], r["fighter_b"]))

    print(f"Target dates: {sorted(targets.keys())}")

    from playwright.sync_api import sync_playwright

    recovered = []
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    discovered_urls = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for date in sorted(targets.keys()):
            fights = targets[date]
            print(f"\n=== {date}: {len(fights)} fights ===")

            # Try to discover correct event URL via fighter pages
            event_url = None
            tried = set()
            for fa, fb in fights:
                for fighter in [fa, fb]:
                    fn = norm(fighter)
                    if fn in tried:
                        continue
                    tried.add(fn)
                    print(f"  Searching via: {fighter}")
                    url = discover_event_url_pw(page, fighter, date)
                    time.sleep(2)
                    if url:
                        # Validate: navigate and check title
                        page.goto(url, timeout=20000)
                        page.wait_for_timeout(3000)
                        title = page.title()
                        content_len = len(page.content())
                        print(f"    Found: {url}")
                        print(f"    Title: {title}, Size: {content_len}")

                        # Reject if it's the homepage
                        if "Best Fight Odds" == title.strip() or content_len > 2000000:
                            print(f"    REJECTED (homepage redirect)")
                            continue

                        event_url = url
                        break
                if event_url:
                    break

            if not event_url:
                print(f"  No valid event page found")
                continue

            discovered_urls[date] = event_url

            # Extract method props
            if event_url != page.url:
                page.goto(event_url, timeout=20000)
                page.wait_for_timeout(5000)

            props = extract_props_pw(page)
            print(f"  Method props: {len(props)} fighters")

            if not props:
                print(f"  No method props on page")
                continue

            prop_names = list(props.keys())
            print(f"  Props fighters: {sorted(prop_names)[:10]}")

            matched = 0
            for fa, fb in fights:
                a_key = match_fighter(fa, prop_names)
                b_key = match_fighter(fb, prop_names)
                a_data = props.get(a_key) if a_key else None
                b_data = props.get(b_key) if b_key else None

                vals = {}
                for prefix, data in [("a", a_data), ("b", b_data)]:
                    for method in ["ko", "sub", "dec"]:
                        v = data.get(method) if data else None
                        vals[f"{prefix}_{method}_odds_prob"] = (
                            round(v, 6) if v is not None else ""
                        )

                if any(v for v in vals.values() if v != ""):
                    rec = {
                        "event_date": date,
                        "fighter_a": fa,
                        "fighter_b": fb,
                        "event_title": "",
                        "source": "bfo_discover_v5",
                        "source_url": event_url,
                        "captured_at": now_iso,
                    }
                    rec.update(vals)
                    recovered.append(rec)
                    matched += 1
                else:
                    print(f"    MISS: {fa} vs {fb}")

            print(f"  Recovered: {matched}/{len(fights)}")
            time.sleep(2)

        browser.close()

    print(f"\nTotal recovered: {len(recovered)}")
    print(f"Discovered URLs: {discovered_urls}")

    if recovered:
        fields = list(existing.columns)
        new_rows = []
        for r in recovered:
            k = (r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"]))
            if k not in existing_keys:
                new_rows.append(r)
                existing_keys.add(k)

        if new_rows:
            with open(METHOD_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                for row in new_rows:
                    out = {field: row.get(field, "") for field in fields}
                    writer.writerow(out)
            print(f"Appended {len(new_rows)} rows")

            rec_keys = set()
            for r in new_rows:
                rec_keys.add(
                    (r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"]))
                )
                rec_keys.add(
                    (r["event_date"], norm(r["fighter_b"]), norm(r["fighter_a"]))
                )
            still = [
                r
                for r in missing
                if (r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"]))
                not in rec_keys
            ]
            with open(MISSING_PATH, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["event_date", "fighter_a", "fighter_b"]
                )
                writer.writeheader()
                for r in still:
                    writer.writerow(r)
            print(f"Missing: {len(missing)} -> {len(still)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
