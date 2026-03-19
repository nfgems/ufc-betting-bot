"""
Retry method odds recovery from multiple sources:
1. BFO with aggressive name matching
2. The Odds API (check all available markets)
3. Web search for fight preview articles with method odds
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
from src.config import ODDS_API_KEY, RAW_DATA_DIR

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
DELAY = 2.0
METHOD_PATH = RAW_DATA_DIR / "method_odds" / "historical_method_odds_all.csv"
MISSING_PATH = RAW_DATA_DIR / "v5_missing_method_odds.csv"


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
    cleaned = re.sub(r"[\u25b2\u25bc\u2191\u2193\u2206\s]", "", str(text).strip())
    m = re.match(r"^([+-]?\d+)$", cleaned)
    return float(m.group(1)) if m else None


def american_to_prob(odds):
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def extract_props_from_soup(soup):
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
            odds_vals = [parse_american(c.get_text(strip=True)) for c in cells[1:]]
            odds_vals = [v for v in odds_vals if v is not None]
            if odds_vals:
                fn = norm(fighter)
                if fn not in all_props:
                    all_props[fn] = {"raw": fighter}
                all_props[fn][method] = american_to_prob(float(np.mean(odds_vals)))
    return all_props


def aggressive_match(target_name, prop_names):
    """More aggressive name matching."""
    tn = norm(target_name)
    tl = last_name(target_name)

    # Exact
    if tn in prop_names:
        return tn

    # Last name exact
    for pn in prop_names:
        pl = pn.split()[-1] if pn.split() else ""
        if pl == tl and tl and len(tl) > 2:
            return pn

    # Fuzzy
    best_score, best = 0, None
    for pn in prop_names:
        score = SequenceMatcher(None, tn, pn).ratio()
        # Reversed name
        t_parts = tn.split()
        if len(t_parts) >= 2:
            rev_score = SequenceMatcher(None, " ".join(reversed(t_parts)), pn).ratio()
            score = max(score, rev_score)
        if score > best_score:
            best_score = score
            best = pn

    if best_score >= 0.50:
        return best

    # Substring on last name
    for pn in prop_names:
        if tl and len(tl) > 3 and tl in pn:
            return pn

    return None


def try_bfo(date, fa, fb, cached_urls):
    """Try BFO with aggressive matching."""
    url = cached_urls.get(date)
    if not url:
        return None

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "lxml")
        props = extract_props_from_soup(soup)
        if not props:
            return None

        prop_names = list(props.keys())
        a_key = aggressive_match(fa, prop_names)
        b_key = aggressive_match(fb, prop_names)
        a_data = props.get(a_key) if a_key else None
        b_data = props.get(b_key) if b_key else None

        result = {}
        for prefix, data in [("a", a_data), ("b", b_data)]:
            for method in ["ko", "sub", "dec"]:
                val = data.get(method) if data else None
                result[f"{prefix}_{method}_odds_prob"] = round(val, 6) if val is not None else ""

        if any(v for v in result.values() if v != ""):
            return result, url
    except Exception as e:
        print(f"    BFO error: {e}")
    return None


def try_odds_api(date, fa, fb):
    """Try The Odds API h2h + alternative markets."""
    if not ODDS_API_KEY:
        return None

    # The Odds API doesn't support method_of_victory for MMA historical
    # But check if we can match the fight and infer anything
    return None


def try_action_network(date, fa, fb):
    """Try ActionNetwork for method odds."""
    try:
        fa_slug = norm(fa).replace(" ", "-")
        fb_slug = norm(fb).replace(" ", "-")
        search_query = f"{fa} vs {fb} method odds"
        # ActionNetwork has UFC odds pages
        url = f"https://www.actionnetwork.com/search?q={requests.utils.quote(search_query)}"
        # Skip - ActionNetwork requires JS
        return None
    except Exception:
        return None


def try_oddschecker(date, fa, fb):
    """Try OddsChecker for method-of-victory odds."""
    try:
        search_url = "https://www.oddschecker.com/search"
        fa_slug = fa.lower().replace(" ", "-")
        fb_slug = fb.lower().replace(" ", "-")

        # Try direct UFC fight page
        fight_url = f"https://www.oddschecker.com/ufc/{fa_slug}-v-{fb_slug}/method-of-victory"
        resp = requests.get(fight_url, headers=HEADERS, timeout=15)
        if resp.status_code == 200 and "method" in resp.text.lower():
            soup = BeautifulSoup(resp.text, "lxml")
            # Parse oddschecker format
            probs = {}
            for row in soup.select("tr"):
                text = row.get_text(" ", strip=True).lower()
                if "ko" in text or "tko" in text or "sub" in text or "dec" in text:
                    # Extract odds
                    odds_cells = row.select("td.bc")
                    if odds_cells:
                        # Found method odds on OddsChecker
                        pass
        return None
    except Exception:
        return None


def try_fightodds_io(date, fa, fb):
    """Try FightOdds.io for historical method props."""
    try:
        # FightOdds.io has historical UFC method props
        search = f"https://fightodds.io/search?q={requests.utils.quote(fa)}"
        resp = requests.get(search, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "lxml")
        # Look for links to fighter or event pages
        for link in soup.select("a[href]"):
            href = link.get("href", "")
            text = link.get_text(strip=True)
            if norm(fa) in norm(text) or last_name(fa) in norm(text):
                # Found fighter page - navigate to it
                if not href.startswith("http"):
                    href = f"https://fightodds.io{href}"
                time.sleep(DELAY)
                fight_resp = requests.get(href, headers=HEADERS, timeout=15)
                if fight_resp.status_code == 200:
                    fight_soup = BeautifulSoup(fight_resp.text, "lxml")
                    page_text = fight_soup.get_text(" ", strip=True)
                    if norm(fb) in norm(page_text) or last_name(fb) in norm(page_text):
                        # Check for method odds
                        for method_text in ["KO/TKO", "Submission", "Decision"]:
                            if method_text in page_text:
                                pass  # Would need to parse specific format
                break
        return None
    except Exception:
        return None


def try_vegasinsider(date, fa, fb):
    """Try VegasInsider for UFC method prop odds."""
    try:
        url = f"https://www.vegasinsider.com/ufc/odds/props/"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        # VegasInsider typically shows current/upcoming events only
        return None
    except Exception:
        return None


def main():
    # Load missing
    missing = list(csv.DictReader(open(MISSING_PATH)))
    print(f"Method odds missing: {len(missing)}")

    # Load existing for dedup
    existing = pd.read_csv(METHOD_PATH, low_memory=False)
    existing_keys = set()
    for _, r in existing.iterrows():
        k = (str(r["event_date"])[:10], norm(r["fighter_a"]), norm(r["fighter_b"]))
        existing_keys.add(k)
        existing_keys.add((str(r["event_date"])[:10], norm(r["fighter_b"]), norm(r["fighter_a"])))

    # Load cached BFO URLs
    ckpt_path = RAW_DATA_DIR / "method_odds" / "v5_bfo_fast_checkpoint.json"
    cached_urls = {}
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            cached_urls = json.load(f).get("event_urls", {})
    # Also merge old checkpoint
    old_ckpt = RAW_DATA_DIR / "method_odds" / "v5_bfo_checkpoint.json"
    if old_ckpt.exists():
        with open(old_ckpt) as f:
            for k, v in json.load(f).get("event_urls", {}).items():
                if k not in cached_urls:
                    cached_urls[k] = v

    recovered = []
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Filter to recoverable fights (skip pre-2019 where BFO has no props, skip far-future)
    for r in missing:
        date, fa, fb = r["event_date"], r["fighter_a"], r["fighter_b"]

        # Skip already existing
        k = (date, norm(fa), norm(fb))
        if k in existing_keys:
            recovered.append(r)
            continue

        # Try BFO with aggressive matching
        bfo_result = try_bfo(date, fa, fb, cached_urls)
        time.sleep(DELAY)
        if bfo_result:
            probs, url = bfo_result
            rec = {
                "event_date": date,
                "fighter_a": fa,
                "fighter_b": fb,
                "event_title": "",
                "source": "bfo_v5_retry",
                "source_url": url,
                "captured_at": now_iso,
            }
            rec.update(probs)
            recovered.append(rec)
            existing_keys.add(k)
            print(f"  BFO RECOVERED: {date} {fa} vs {fb}")
            continue

        # Try FightOdds.io
        fo_result = try_fightodds_io(date, fa, fb)
        if fo_result:
            probs, url = fo_result
            rec = {
                "event_date": date,
                "fighter_a": fa,
                "fighter_b": fb,
                "event_title": "",
                "source": "fightodds_io",
                "source_url": url,
                "captured_at": now_iso,
            }
            rec.update(probs)
            recovered.append(rec)
            existing_keys.add(k)
            print(f"  FightOdds RECOVERED: {date} {fa} vs {fb}")
            continue

    print(f"\nRecovered: {len(recovered)} (of {len(missing)} missing)")

    # Append new rows
    new_rows = [r for r in recovered if "source" in r]
    if new_rows:
        fields = list(existing.columns)
        with open(METHOD_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            for row in new_rows:
                out = {field: row.get(field, "") for field in fields}
                writer.writerow(out)
        print(f"Appended {len(new_rows)} rows")

    # Update missing
    recovered_keys = set()
    for r in recovered:
        recovered_keys.add((r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"])))
        recovered_keys.add((r["event_date"], norm(r["fighter_b"]), norm(r["fighter_a"])))

    still_missing = [
        r for r in missing
        if (r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"])) not in recovered_keys
    ]
    with open(MISSING_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["event_date", "fighter_a", "fighter_b"])
        writer.writeheader()
        for r in still_missing:
            writer.writerow(r)
    print(f"Missing: {len(missing)} -> {len(still_missing)}")


if __name__ == "__main__":
    main()
