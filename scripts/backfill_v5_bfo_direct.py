"""
Search BFO for individual fight pages to recover method odds
that were missed by event-page scraping.
"""

import csv
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
DELAY = 2.5
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


def search_bfo(query):
    """Search BFO and return all fight/event links found."""
    try:
        resp = requests.get(
            "https://www.bestfightodds.com/search",
            params={"query": query},
            headers=HEADERS,
            timeout=20,
        )
        time.sleep(DELAY)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "lxml")
        links = []
        for link in soup.select("a[href]"):
            href = link.get("href", "")
            text = link.get_text(" ", strip=True)
            if "/fights/" in href or "/events/" in href:
                full = (
                    f"https://www.bestfightodds.com{href}"
                    if not href.startswith("http")
                    else href
                )
                links.append((full, text, "fight" if "/fights/" in href else "event"))
        return links
    except Exception as e:
        print(f"    Search error: {e}")
        return []


def extract_props_from_page(url):
    """Extract method-of-victory props from any BFO page."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        time.sleep(DELAY)
        if resp.status_code != 200:
            return {}
        soup = BeautifulSoup(resp.text, "lxml")

        props = {}
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
                fighter = re.sub(
                    r"\s*wins\s+by\s+.*$", "", label, flags=re.IGNORECASE
                ).strip()
                odds_vals = [
                    parse_american(c.get_text(strip=True)) for c in cells[1:]
                ]
                odds_vals = [v for v in odds_vals if v is not None]
                if odds_vals:
                    fn = norm(fighter)
                    if fn not in props:
                        props[fn] = {"raw": fighter}
                    props[fn][method] = american_to_prob(float(np.mean(odds_vals)))
        return props
    except Exception as e:
        print(f"    Page error: {e}")
        return {}


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


def try_recover(date, fa, fb):
    """Try multiple BFO search strategies to find method odds."""
    fa_last = norm(fa).split()[-1] if norm(fa).split() else ""
    fb_last = norm(fb).split()[-1] if norm(fb).split() else ""

    # Strategy 1: Search for fighter A
    links = search_bfo(fa)
    for url, text, ptype in links:
        if ptype == "fight":
            # Check if this fight page mentions both fighters
            props = extract_props_from_page(url)
            if props:
                pnames = list(props.keys())
                a_key = match_fighter(fa, pnames)
                b_key = match_fighter(fb, pnames)
                if a_key or b_key:
                    return props, pnames, url

    # Strategy 2: Search for fighter B
    links = search_bfo(fb)
    for url, text, ptype in links:
        if ptype == "fight":
            props = extract_props_from_page(url)
            if props:
                pnames = list(props.keys())
                a_key = match_fighter(fa, pnames)
                b_key = match_fighter(fb, pnames)
                if a_key or b_key:
                    return props, pnames, url

    # Strategy 3: Search for "fighter_a vs fighter_b"
    links = search_bfo(f"{fa} vs {fb}")
    for url, text, ptype in links:
        props = extract_props_from_page(url)
        if props:
            pnames = list(props.keys())
            a_key = match_fighter(fa, pnames)
            b_key = match_fighter(fb, pnames)
            if a_key or b_key:
                return props, pnames, url

    return None, None, None


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

    # Focus on 2023-2024 fights
    targets = [
        r
        for r in missing
        if "2023" <= r["event_date"][:4] <= "2024"
    ]
    print(f"Targeting {len(targets)} fights from 2023-2024")

    recovered = []
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for r in targets:
        date, fa, fb = r["event_date"], r["fighter_a"], r["fighter_b"]
        k = (date, norm(fa), norm(fb))
        if k in existing_keys:
            continue

        print(f"\n  {date}: {fa} vs {fb}")
        props, pnames, url = try_recover(date, fa, fb)

        if props and pnames:
            a_key = match_fighter(fa, pnames)
            b_key = match_fighter(fb, pnames)
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
                    "source": "bfo_fight_direct",
                    "source_url": url,
                    "captured_at": now_iso,
                }
                rec.update(vals)
                recovered.append(rec)
                existing_keys.add(k)
                print(f"    RECOVERED from {url.split('/')[-1]}")
            else:
                print(f"    Props found but no fighter match. Props: {pnames[:5]}")
        else:
            print(f"    No BFO page with props found")

    print(f"\nRecovered: {len(recovered)}/{len(targets)}")

    if recovered:
        fields = list(existing.columns)
        with open(METHOD_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            for row in recovered:
                out = {field: row.get(field, "") for field in fields}
                writer.writerow(out)
        print(f"Appended {len(recovered)} rows")

        rec_keys = set()
        for r in recovered:
            rec_keys.add((r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"])))
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
