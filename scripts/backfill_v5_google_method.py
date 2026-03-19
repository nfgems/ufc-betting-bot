"""
Google search for method-of-victory odds from various sources.
Targets fights where BFO failed to provide method props.
"""

import csv
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8")

from src.config import RAW_DATA_DIR

METHOD_PATH = RAW_DATA_DIR / "method_odds" / "historical_method_odds_all.csv"
MISSING_PATH = RAW_DATA_DIR / "v5_missing_method_odds.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def norm(n):
    if not n:
        return ""
    n = unicodedata.normalize("NFKD", str(n))
    n = "".join(c for c in n if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-zA-Z\s]", "", n).lower().split())


def american_to_prob(odds):
    """Convert American odds to implied probability."""
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    if odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def decimal_to_prob(odds):
    """Convert decimal odds to implied probability."""
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    if odds <= 0:
        return None
    return 1.0 / odds


def parse_odds_from_text(text):
    """Extract American odds from text like '+250', '-150', etc."""
    matches = re.findall(r"[+-]\d{3,}", text)
    return [float(m) for m in matches]


def google_search(query, num_results=5):
    """Simple Google search via requests."""
    try:
        resp = requests.get(
            "https://www.google.com/search",
            params={"q": query, "num": num_results},
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        results = []
        for div in soup.select("div.g, div[data-sokoban-container]"):
            link = div.select_one("a[href]")
            if link:
                href = link.get("href", "")
                if href.startswith("/url?q="):
                    href = href.split("/url?q=")[1].split("&")[0]
                if href.startswith("http"):
                    snippet = div.get_text(" ", strip=True)[:300]
                    results.append((href, snippet))
        return results[:num_results]
    except Exception as e:
        print(f"    Google error: {e}")
        return []


def extract_method_odds_from_article(url, fa, fb):
    """Fetch an article and try to extract method-of-victory odds."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        text = resp.text
        soup = BeautifulSoup(text, "lxml")
        page_text = soup.get_text(" ", strip=True)

        # Check both fighters are mentioned
        fa_n, fb_n = norm(fa), norm(fb)
        fa_last = fa_n.split()[-1] if fa_n.split() else ""
        fb_last = fb_n.split()[-1] if fb_n.split() else ""

        page_lower = page_text.lower()
        if not (fa_last in page_lower and fb_last in page_lower):
            return None

        # Look for method odds patterns in the text
        # Pattern: "Fighter wins by KO/TKO +250" or "KO/TKO: +250"
        result = {
            "a_ko_odds_prob": "",
            "a_sub_odds_prob": "",
            "a_dec_odds_prob": "",
            "b_ko_odds_prob": "",
            "b_sub_odds_prob": "",
            "b_dec_odds_prob": "",
        }

        # Method keywords
        ko_pattern = r"(?:ko|tko|knockout|ko/tko|tko/ko)"
        sub_pattern = r"(?:sub|submission)"
        dec_pattern = r"(?:dec|decision|points)"

        # Try to find sections about each fighter
        for prefix, fighter, fighter_last in [("a", fa, fa_last), ("b", fb, fb_last)]:
            # Look for patterns like "Fighter wins by KO/TKO: +250"
            for method, method_key, pattern in [
                ("ko", ko_pattern, ko_pattern),
                ("sub", sub_pattern, sub_pattern),
                ("dec", dec_pattern, dec_pattern),
            ]:
                # Pattern: "LastName ... wins by method ... +/-NNN"
                regex = rf"{re.escape(fighter_last)}\s+(?:\w+\s+){{0,8}}(?:wins?\s+by\s+)?{pattern}[^\d+-]*([+-]\d{{3,}})"
                m = re.search(regex, page_lower)
                if m:
                    prob = american_to_prob(m.group(1))
                    if prob:
                        result[f"{prefix}_{method}_odds_prob"] = round(prob, 6)
                        continue

                # Pattern: "method ... LastName ... +/-NNN"
                regex2 = rf"{pattern}[^\d+-]*{re.escape(fighter_last)}[^\d+-]*([+-]\d{{3,}})"
                m2 = re.search(regex2, page_lower)
                if m2:
                    prob = american_to_prob(m2.group(1))
                    if prob:
                        result[f"{prefix}_{method}_odds_prob"] = round(prob, 6)

        has_any = any(v for v in result.values() if v != "")
        return result if has_any else None

    except Exception as e:
        return None


def search_covers_com(fa, fb):
    """Search Covers.com for method odds."""
    try:
        fa_slug = fa.lower().replace(" ", "-")
        fb_slug = fb.lower().replace(" ", "-")
        url = f"https://www.covers.com/sport/ufc/matchup/{fa_slug}-vs-{fb_slug}"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "lxml")
            text = soup.get_text(" ", strip=True)
            if "method" in text.lower() or "ko" in text.lower():
                return extract_method_odds_from_article(url, fa, fb)
    except Exception:
        pass
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

    # Target: all remaining 2023+ fights (skip pre-2019 which truly didn't have props)
    targets = [r for r in missing if r["event_date"] >= "2023-01-01"]
    print(f"Targeting {len(targets)} fights from 2023+")

    recovered = []
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for i, r in enumerate(targets):
        date, fa, fb = r["event_date"], r["fighter_a"], r["fighter_b"]
        k = (date, norm(fa), norm(fb))
        if k in existing_keys:
            continue

        print(f"\n[{i+1}/{len(targets)}] {date}: {fa} vs {fb}")

        # Google search for method odds
        queries = [
            f'"{fa}" "{fb}" method of victory odds',
            f'"{fa}" vs "{fb}" KO TKO submission decision odds',
            f'{fa} {fb} UFC prop odds method',
        ]

        found = False
        for query in queries:
            results = google_search(query)
            time.sleep(2)

            for url, snippet in results:
                # Skip non-relevant domains
                if any(
                    d in url
                    for d in [
                        "youtube.com",
                        "reddit.com",
                        "twitter.com",
                        "facebook.com",
                        "instagram.com",
                    ]
                ):
                    continue

                print(f"  Checking: {url[:80]}")
                odds = extract_method_odds_from_article(url, fa, fb)
                time.sleep(1)

                if odds:
                    rec = {
                        "event_date": date,
                        "fighter_a": fa,
                        "fighter_b": fb,
                        "event_title": "",
                        "source": "web_search",
                        "source_url": url,
                        "captured_at": now_iso,
                    }
                    rec.update(odds)
                    recovered.append(rec)
                    existing_keys.add(k)
                    print(f"  RECOVERED from {url[:60]}")
                    found = True
                    break

            if found:
                break

        if not found:
            print(f"  No method odds found via search")

    print(f"\nTotal recovered: {len(recovered)}")

    if recovered:
        fields = list(existing.columns)
        new_rows = []
        for r in recovered:
            k = (r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"]))
            if k not in existing_keys or k in {
                (r2["event_date"], norm(r2["fighter_a"]), norm(r2["fighter_b"]))
                for r2 in recovered
            }:
                new_rows.append(r)

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
