"""
Try Wayback Machine snapshots of BFO event pages for method odds recovery.
Also try OddsPortal historical and direct BFO event URL construction.
"""

import csv
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.config import RAW_DATA_DIR

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
DELAY = 2.0
METHOD_PATH = RAW_DATA_DIR / "method_odds" / "historical_method_odds_all.csv"
MISSING_PATH = RAW_DATA_DIR / "v5_missing_method_odds.csv"


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


def extract_props(soup):
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
        elif (
            re.search(r"wins by (decision|dec)", label_lower)
            and not any(w in label_lower for w in ["unanimous", "split", "majority"])
        ):
            method = "dec"
        if method:
            fighter = re.sub(r"\s*wins\s+by\s+.*$", "", label, flags=re.IGNORECASE).strip()
            odds_vals = [parse_american(c.get_text(strip=True)) for c in cells[1:]]
            odds_vals = [v for v in odds_vals if v is not None]
            if odds_vals:
                fn = norm(fighter)
                if fn not in all_props:
                    all_props[fn] = {"raw": fighter}
                all_props[fn][method] = american_to_prob(float(np.mean(odds_vals)))
    return all_props


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
            score = max(score, SequenceMatcher(None, " ".join(reversed(t_parts)), pn).ratio())
        if score > best_score:
            best_score = score
            best = pn
    if best_score >= 0.50:
        return best
    for pn in prop_names:
        if tl and len(tl) > 3 and tl in pn:
            return pn
    return None


def try_wayback(bfo_url, date, fa, fb):
    """Try Wayback Machine for a BFO event page."""
    try:
        # Use CDX API for better results
        cdx_url = f"https://web.archive.org/cdx/search/cdx?url={bfo_url}&output=json&limit=5"
        resp = requests.get(cdx_url, headers=HEADERS, timeout=15)
        time.sleep(1)
        if resp.status_code != 200 or not resp.text.strip():
            return None

        rows = resp.json()
        if len(rows) <= 1:  # Header only
            return None

        # Get the best snapshot (closest to event date)
        for row in rows[1:]:
            timestamp = row[1]  # e.g. 20231104120000
            wb_url = f"https://web.archive.org/web/{timestamp}/{bfo_url}"
            resp2 = requests.get(wb_url, headers=HEADERS, timeout=20)
            time.sleep(DELAY)
            if resp2.status_code != 200:
                continue

            soup = BeautifulSoup(resp2.text, "lxml")
            props = extract_props(soup)
            if not props:
                continue

            prop_names = list(props.keys())
            a_key = match_fighter(fa, prop_names)
            b_key = match_fighter(fb, prop_names)
            a_data = props.get(a_key) if a_key else None
            b_data = props.get(b_key) if b_key else None

            vals = {}
            for prefix, data in [("a", a_data), ("b", b_data)]:
                for method in ["ko", "sub", "dec"]:
                    v = data.get(method) if data else None
                    vals[f"{prefix}_{method}_odds_prob"] = round(v, 6) if v is not None else ""

            if any(v for v in vals.values() if v != ""):
                return vals, wb_url
            else:
                print(f"    Wayback had {len(props)} fighters but no match. Props: {list(props.keys())[:8]}")
                return None

    except Exception as e:
        print(f"    Wayback error: {e}")
    return None


def try_oddsportal(date, fa, fb):
    """Try OddsPortal for historical MMA prop odds."""
    try:
        # OddsPortal URL format for UFC events
        year = date[:4]
        search_url = f"https://www.oddsportal.com/mma/usa/ufc/results/#/page/1/"
        resp = requests.get(search_url, headers=HEADERS, timeout=15)
        time.sleep(DELAY)
        # OddsPortal is heavily JS-rendered, unlikely to work with requests
        return None
    except Exception:
        return None


def main():
    missing = list(csv.DictReader(open(MISSING_PATH)))
    print(f"Method odds missing: {len(missing)}")

    existing = pd.read_csv(METHOD_PATH, low_memory=False)
    existing_keys = set()
    for _, r in existing.iterrows():
        k = (str(r["event_date"])[:10], norm(r["fighter_a"]), norm(r["fighter_b"]))
        existing_keys.add(k)
        existing_keys.add((str(r["event_date"])[:10], norm(r["fighter_b"]), norm(r["fighter_a"])))

    # Load cached URLs
    cached_urls = {}
    for ckpt_name in ["v5_bfo_fast_checkpoint.json", "v5_bfo_checkpoint.json"]:
        ckpt_path = RAW_DATA_DIR / "method_odds" / ckpt_name
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                for k, v in json.load(f).get("event_urls", {}).items():
                    if k not in cached_urls:
                        cached_urls[k] = v

    recovered = []
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Focus on 2023-2024 fights that had BFO pages but didn't match
    recoverable = [
        r for r in missing
        if r["event_date"] >= "2019-01-01"
        and r["event_date"] <= "2025-06-30"
        and r["event_date"] in cached_urls
    ]

    print(f"Attempting Wayback for {len(recoverable)} fights with cached BFO URLs")

    for r in recoverable:
        date, fa, fb = r["event_date"], r["fighter_a"], r["fighter_b"]
        k = (date, norm(fa), norm(fb))
        if k in existing_keys:
            continue

        bfo_url = cached_urls[date]
        result = try_wayback(bfo_url, date, fa, fb)

        if result:
            vals, source_url = result
            rec = {
                "event_date": date,
                "fighter_a": fa,
                "fighter_b": fb,
                "event_title": "",
                "source": "wayback_bfo",
                "source_url": source_url,
                "captured_at": now_iso,
            }
            rec.update(vals)
            recovered.append(rec)
            existing_keys.add(k)
            print(f"  RECOVERED: {date} {fa} vs {fb}")
        else:
            print(f"  MISS: {date} {fa} vs {fb}")

    print(f"\nRecovered from Wayback: {len(recovered)}")

    if recovered:
        fields = list(existing.columns)
        with open(METHOD_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            for row in recovered:
                out = {field: row.get(field, "") for field in fields}
                writer.writerow(out)
        print(f"Appended {len(recovered)} rows")

        recovered_keys = set()
        for r in recovered:
            recovered_keys.add((r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"])))
            recovered_keys.add((r["event_date"], norm(r["fighter_b"]), norm(r["fighter_a"])))

        still = [
            r for r in missing
            if (r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"])) not in recovered_keys
        ]
        with open(MISSING_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["event_date", "fighter_a", "fighter_b"])
            writer.writeheader()
            for r in still:
                writer.writerow(r)
        print(f"Missing: {len(missing)} -> {len(still)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
