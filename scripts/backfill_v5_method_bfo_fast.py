"""
Fast BFO method odds backfill using requests (no Playwright).

Strategy:
1. For each missing fight, find the BFO event page via fighter search
2. Scrape static HTML for method-of-victory props
3. Match fighters and extract implied probabilities

Usage:
    python scripts/backfill_v5_method_bfo_fast.py [--max-events N] [--start-date YYYY-MM-DD]
"""

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parent.parent
METHOD_PATH = REPO / "data" / "raw" / "method_odds" / "historical_method_odds_all.csv"
MISSING_PATH = REPO / "data" / "raw" / "v5_missing_method_odds.csv"
CHECKPOINT_PATH = REPO / "data" / "raw" / "method_odds" / "v5_bfo_fast_checkpoint.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
DELAY = 2.0

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def norm(n):
    if pd.isna(n) or not n:
        return ""
    n = unicodedata.normalize("NFKD", str(n))
    n = "".join(c for c in n if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-zA-Z\s]", "", n).lower().split())


def last_name(n):
    parts = norm(n).split()
    return parts[-1] if parts else ""


def parse_american(text):
    """Parse American odds from text, handling unicode arrows."""
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
    m = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d+)\w*\s+(\d{4})", text, re.IGNORECASE)
    if not m:
        return None
    month = MONTH_MAP.get(m.group(1).lower()[:3])
    day = int(m.group(2))
    year = int(m.group(3))
    if month:
        return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def fuzzy_match_name(target, candidates, threshold=0.60):
    """Match target name to a set of candidate names."""
    tn = norm(target)
    tl = last_name(target)

    # Exact match
    if tn in candidates:
        return tn

    # Last name match
    for cn in candidates:
        cl = cn.split()[-1] if cn.split() else ""
        if cl == tl and tl:
            return cn

    # Fuzzy
    best_score, best = 0, None
    for cn in candidates:
        score = SequenceMatcher(None, tn, cn).ratio()
        # Also try reversed name
        t_parts = tn.split()
        if len(t_parts) >= 2:
            rev_score = SequenceMatcher(None, " ".join(reversed(t_parts)), cn).ratio()
            score = max(score, rev_score)
        # Last name boost
        cl = cn.split()[-1] if cn.split() else ""
        if cl == tl and tl:
            score = max(score, 0.75)
        if score > best_score:
            best_score = score
            best = cn

    if best_score >= threshold:
        return best

    # Substring fallback
    for cn in candidates:
        if tl and tl in cn:
            return cn

    return None


def extract_props_from_soup(soup):
    """Extract method-of-victory props from BFO event page HTML."""
    all_props = {}

    for row in soup.select("tr"):
        cells = row.select("td, th")
        if not cells:
            continue
        label = cells[0].get_text(strip=True)
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
            odds_vals = []
            for cell in cells[1:]:
                val = parse_american(cell.get_text(strip=True))
                if val is not None:
                    odds_vals.append(val)

            if odds_vals:
                fn = norm(fighter)
                if fn not in all_props:
                    all_props[fn] = {"raw": fighter}
                all_props[fn][method] = american_to_prob(float(np.mean(odds_vals)))

    return all_props


def find_fighter_page_url(fighter_name):
    """Search BFO for a fighter page URL."""
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

        candidates = []
        for link in soup.select("a[href*='/fighters/']"):
            href = link.get("href", "")
            text = link.get_text(strip=True)
            if "/fighters/" in href and text:
                link_norm = norm(text)
                score = SequenceMatcher(None, fn, link_norm).ratio()
                # Try reversed
                fn_parts = fn.split()
                if len(fn_parts) >= 2:
                    rev_score = SequenceMatcher(None, " ".join(reversed(fn_parts)), link_norm).ratio()
                    score = max(score, rev_score)
                ln_score = SequenceMatcher(None, last_name(fighter_name), link_norm.split()[-1] if link_norm.split() else "").ratio()
                best = max(score, ln_score * 1.15)
                if best >= 0.65:
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
    """Find event URL from fighter's BFO page matching target date."""
    try:
        resp = requests.get(fighter_page_url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "lxml")

        for row in soup.select("tr"):
            event_link = row.select_one("a[href*='/events/']")
            if not event_link:
                continue
            row_text = row.get_text(" ", strip=True)
            parsed_date = parse_bfo_date(row_text)

            if parsed_date == target_date:
                href = event_link["href"]
                return f"https://www.bestfightodds.com{href}" if not href.startswith("http") else href

        # Try +/- 1 day tolerance
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        for delta in [1, -1]:
            alt_date = (target_dt + timedelta(days=delta)).strftime("%Y-%m-%d")
            for row in soup.select("tr"):
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
    """Find BFO event URL by checking fighter pages."""
    tried = set()
    all_fighters = []
    for fa, fb in fighters:
        all_fighters.extend([fa, fb])

    for fighter in all_fighters[:6]:
        fn = norm(fighter)
        if fn in tried:
            continue
        tried.add(fn)

        fighter_url = find_fighter_page_url(fighter)
        time.sleep(DELAY)
        if not fighter_url:
            continue

        event_url = find_event_url_from_fighter_page(fighter_url, target_date)
        time.sleep(DELAY)
        if event_url:
            return event_url

    return None


def scrape_event_props_static(url):
    """Scrape method props from BFO event page using requests."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return {}
        soup = BeautifulSoup(resp.text, "lxml")
        return extract_props_from_soup(soup)
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
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--start-date", type=str, default="")
    args = parser.parse_args()

    missing = pd.read_csv(MISSING_PATH)
    print(f"Total missing method odds: {len(missing)}")

    event_groups = defaultdict(list)
    for _, r in missing.iterrows():
        event_groups[r["event_date"]].append((r["fighter_a"], r["fighter_b"]))

    dates = sorted(event_groups.keys())
    if args.start_date:
        dates = [d for d in dates if d >= args.start_date]

    # Load checkpoint — merge both old Playwright and new fast checkpoints
    ckpt = load_checkpoint()

    # Also load old Playwright checkpoint for processed dates and cached URLs
    old_ckpt_path = REPO / "data" / "raw" / "method_odds" / "v5_bfo_checkpoint.json"
    if old_ckpt_path.exists():
        with open(old_ckpt_path) as f:
            old_ckpt = json.load(f)
        # Merge cached event URLs
        old_urls = old_ckpt.get("event_urls", {})
        for k, v in old_urls.items():
            if k not in ckpt.get("event_urls", {}):
                ckpt.setdefault("event_urls", {})[k] = v

    processed = set(ckpt["processed_dates"])
    all_recovered = ckpt["recovered"]
    cached_urls = ckpt.get("event_urls", {})

    dates_to_process = [d for d in dates if d not in processed]
    if args.max_events > 0:
        dates_to_process = dates_to_process[:args.max_events]

    print(f"Dates to process: {len(dates_to_process)} (of {len(dates)} total)")
    print(f"Already processed: {len(processed)}")
    print(f"Already recovered: {len(all_recovered)}")
    print()

    for idx, event_date in enumerate(dates_to_process):
        fighters = event_groups[event_date]
        print(f"\n[{idx+1}/{len(dates_to_process)}] {event_date}: {len(fighters)} fights")

        # Check cached URL
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

        print(f"  Event URL: .../{event_url.split('/')[-1]}")

        # Scrape
        props = scrape_event_props_static(event_url)
        time.sleep(DELAY)

        if not props:
            print(f"  No method props on event page")
            ckpt["processed_dates"].append(event_date)
            ckpt["stats"]["total_missed"] += len(fighters)
            save_checkpoint(ckpt)
            continue

        print(f"  Found props for {len(props)} fighters")

        event_recovered = 0
        now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        prop_names = list(props.keys())
        for fa, fb in fighters:
            a_key = fuzzy_match_name(fa, prop_names)
            b_key = fuzzy_match_name(fb, prop_names)

            a_data = props.get(a_key) if a_key else None
            b_data = props.get(b_key) if b_key else None

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
                    "fighter_a": fa,
                    "fighter_b": fb,
                    "event_title": "",
                    "source": "bfo_v5_fast_backfill",
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
                print(f"    MISS: {fa} vs {fb}")

        print(f"  Recovered {event_recovered}/{len(fighters)}")
        ckpt["processed_dates"].append(event_date)
        ckpt["recovered"] = all_recovered
        ckpt["stats"]["total_recovered"] += event_recovered
        save_checkpoint(ckpt)

    # Append all recovered rows
    if all_recovered:
        print(f"\n{'='*60}")
        print(f"Total recovered: {len(all_recovered)}")

        existing = pd.read_csv(METHOD_PATH, low_memory=False)
        existing_keys = set()
        for _, r in existing.iterrows():
            k = (str(r["event_date"])[:10], norm(r["fighter_a"]), norm(r["fighter_b"]))
            existing_keys.add(k)
            existing_keys.add((str(r["event_date"])[:10], norm(r["fighter_b"]), norm(r["fighter_a"])))

        new_rows = []
        for r in all_recovered:
            k = (r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"]))
            if k not in existing_keys:
                new_rows.append(r)
                existing_keys.add(k)

        if new_rows:
            fields = list(existing.columns)
            with open(METHOD_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                for row in new_rows:
                    out = {field: row.get(field, "") for field in fields}
                    writer.writerow(out)
            print(f"Appended {len(new_rows)} new rows to {METHOD_PATH}")
        else:
            print("All recovered rows already exist.")

        # Update missing CSV
        recovered_keys = set()
        for r in all_recovered:
            recovered_keys.add((r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"])))
            recovered_keys.add((r["event_date"], norm(r["fighter_b"]), norm(r["fighter_a"])))

        still_missing = []
        for _, r in missing.iterrows():
            k = (r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"]))
            if k not in recovered_keys:
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
