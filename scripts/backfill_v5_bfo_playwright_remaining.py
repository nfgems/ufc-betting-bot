"""
Use Playwright to scrape BFO event pages that need JS rendering for method props.
Targets the 2025 events that returned no props via static HTML.
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8")

from src.config import RAW_DATA_DIR

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


def extract_props_playwright(page):
    """Extract method props from current page using Playwright."""
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


# Event URLs to try (including alternate pages)
EVENT_URLS = {
    "2025-07-12": [
        "https://www.bestfightodds.com/events/ufc-fight-night-3748",
        "https://www.bestfightodds.com/events/ufc-fight-night-3749",
    ],
    "2025-07-19": [
        "https://www.bestfightodds.com/events/ufc-3706",
        "https://www.bestfightodds.com/events/ufc-3707",
    ],
    "2025-08-22": [
        "https://www.bestfightodds.com/events/ufc-3818",
        "https://www.bestfightodds.com/events/ufc-3819",
    ],
}


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

    # Filter to target dates
    targets = {}
    for r in missing:
        d = r["event_date"]
        if d in EVENT_URLS:
            targets.setdefault(d, []).append((r["fighter_a"], r["fighter_b"]))

    print(f"Target dates: {list(targets.keys())}")
    for d, fights in targets.items():
        print(f"  {d}: {len(fights)} fights")

    from playwright.sync_api import sync_playwright

    recovered = []
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for date, fights in sorted(targets.items()):
            print(f"\n=== {date}: {len(fights)} fights ===")

            best_props = {}
            best_url = ""

            for url in EVENT_URLS[date]:
                print(f"  Trying: {url}")
                try:
                    page.goto(url, timeout=30000)
                    page.wait_for_timeout(5000)  # Wait for JS to render

                    # Check page content length
                    content = page.content()
                    print(f"    Page content: {len(content)} chars")

                    props = extract_props_playwright(page)
                    print(f"    Method props: {len(props)} fighters")

                    if len(props) > len(best_props):
                        best_props = props
                        best_url = url

                    if props:
                        print(f"    Fighters: {sorted(props.keys())[:8]}")
                except Exception as e:
                    print(f"    Error: {e}")

                time.sleep(2)

            if not best_props:
                print(f"  No method props found on any page")
                continue

            print(f"\n  Best page: {best_url} ({len(best_props)} fighters)")
            prop_names = list(best_props.keys())

            matched = 0
            for fa, fb in fights:
                a_key = match_fighter(fa, prop_names)
                b_key = match_fighter(fb, prop_names)
                a_data = best_props.get(a_key) if a_key else None
                b_data = best_props.get(b_key) if b_key else None

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
                        "source": "bfo_playwright_v5",
                        "source_url": best_url,
                        "captured_at": now_iso,
                    }
                    rec.update(vals)
                    recovered.append(rec)
                    matched += 1
                else:
                    print(f"    MISS: {fa} vs {fb}")

            print(f"  Recovered: {matched}/{len(fights)}")

        browser.close()

    print(f"\nTotal recovered: {len(recovered)}")

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
