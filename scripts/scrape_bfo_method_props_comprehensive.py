"""
Comprehensive BFO method prop scraper for all 229 missing fights.

Strategy:
1. Search BFO for each unique event name to find event URLs
2. Load each event page via Playwright (JS-rendered odds)
3. Extract method-of-victory props (KO/TKO, Submission, Decision)
4. Match to missing fighters and save

BFO pages contain rows like:
  "Fighter wins by TKO/KO"      +250  +300  ...
  "Fighter wins by submission"   +450  +500  ...
  "Fighter wins by decision"     -150  -120  ...
"""
import csv
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

REPO_ROOT = Path(__file__).resolve().parent.parent
METHOD_ODDS_PATH = REPO_ROOT / "data/raw/method_odds/historical_method_odds_all.csv"
OUTPUT_PATH = REPO_ROOT / "data/raw/method_odds/bfo_comprehensive_recovery.csv"
CHECKPOINT_PATH = REPO_ROOT / "data/raw/method_odds/bfo_comprehensive_checkpoint.json"
BFO_DELAY = 3.0


def norm(n):
    if pd.isna(n):
        return ""
    n = unicodedata.normalize("NFKD", str(n))
    n = "".join(c for c in n if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-zA-Z\s]", "", n).lower().split())


def last_name(n):
    parts = norm(n).split()
    return parts[-1] if parts else ""


def first_name(n):
    parts = norm(n).split()
    return parts[0] if parts else ""


def parse_american(t):
    c = re.sub(r"[\u25b2\u25bc\u2191\u2193\s]", "", str(t).strip())
    m = re.match(r"^([+-]?\d+)$", c)
    return float(m.group(1)) if m else None


def american_to_prob(odds):
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def compute_missing_fights():
    """Compute the full list of 229 missing fights."""
    kaggle = pd.read_csv(REPO_ROOT / "data/raw/jansen88_ufc_data.csv")
    kaggle["event_date"] = pd.to_datetime(kaggle["event_date"])
    fights = kaggle[kaggle["event_date"] >= "2014-01-01"].copy()

    method = pd.read_csv(METHOD_ODDS_PATH)
    method["event_date"] = pd.to_datetime(method["event_date"], format="mixed")

    covered = set()
    for _, r in method.iterrows():
        d = r["event_date"].strftime("%Y-%m-%d")
        a, b = last_name(r["fighter_a"]), last_name(r["fighter_b"])
        covered.add((d, a, b))
        covered.add((d, b, a))

    missing = []
    for _, r in fights.iterrows():
        d = r["event_date"].strftime("%Y-%m-%d")
        a, b = last_name(r["fighter1"]), last_name(r["fighter2"])
        if (d, a, b) not in covered:
            missing.append({
                "event_date": d,
                "fighter1": r["fighter1"],
                "fighter2": r["fighter2"],
                "event_name": r["event_name"],
            })

    return pd.DataFrame(missing)


def search_bfo_event(page, event_name):
    """Search BFO for an event and return the best matching URL."""
    # Clean event name for search
    search_terms = event_name
    # Remove common prefixes that might not match
    for prefix in ["The Ultimate Fighter:", "UFC on FOX:", "UFC on ESPN:", "UFC Fight Night:"]:
        if search_terms.startswith(prefix):
            search_terms = search_terms.replace(prefix, "").strip()
            break

    # Try with the full event name first
    page.goto(f"https://www.bestfightodds.com/search?query={search_terms[:60]}", timeout=30000)
    page.wait_for_timeout(2000)

    links = page.query_selector_all('a[href*="/events/"]')
    results = []
    for link in links:
        href = link.get_attribute("href") or ""
        text = link.inner_text().strip()
        if "/events/" in href and text:
            full_url = f"https://www.bestfightodds.com{href}" if not href.startswith("http") else href
            results.append((text, full_url))

    return results


def search_bfo_by_fighters(page, fighter1, fighter2):
    """Search BFO by fighter names to find the event."""
    last1, last2 = last_name(fighter1), last_name(fighter2)
    query = f"{last1} {last2}"
    page.goto(f"https://www.bestfightodds.com/search?query={query}", timeout=30000)
    page.wait_for_timeout(2000)

    links = page.query_selector_all('a[href*="/events/"]')
    results = []
    for link in links:
        href = link.get_attribute("href") or ""
        text = link.inner_text().strip()
        if "/events/" in href and text:
            full_url = f"https://www.bestfightodds.com{href}" if not href.startswith("http") else href
            results.append((text, full_url))

    return results


def extract_method_props(page):
    """
    Extract all method-of-victory props from the current BFO event page.

    Returns dict: {normalized_fighter_last_name: {method: probability}}
    """
    rows = page.query_selector_all("tr")
    fighter_methods = defaultdict(dict)

    for row in rows:
        cells = row.query_selector_all("td, th")
        if not cells:
            continue

        label = cells[0].inner_text().strip()
        label_lower = label.lower()

        # Match patterns like "Fighter wins by TKO/KO", "Fighter wins by submission", etc.
        # Only match top-level method (not round-specific or technique-specific)
        method = None
        if re.search(r'wins by (tko/?ko|ko/?tko|knockout)', label_lower) and 'round' not in label_lower and 'via' not in label_lower:
            method = "ko"
        elif re.search(r'wins by sub', label_lower) and 'round' not in label_lower:
            method = "sub"
        elif re.search(r'wins by (decision|dec)', label_lower) and 'unanimous' not in label_lower and 'split' not in label_lower and 'majority' not in label_lower:
            method = "dec"

        if method:
            # Extract fighter name from the label
            fighter_name = re.sub(
                r"\s*wins\s+by\s+.*$", "", label, flags=re.IGNORECASE
            ).strip()

            # Get odds values
            odds_vals = []
            for cell in cells[1:]:
                val = parse_american(cell.inner_text().strip())
                if val is not None:
                    odds_vals.append(val)

            if odds_vals:
                avg_odds = np.mean(odds_vals)
                prob = american_to_prob(avg_odds)
                fighter_key = last_name(fighter_name)
                fighter_first = first_name(fighter_name)
                # Store with both last name and full key for disambiguation
                key = f"{fighter_first}_{fighter_key}" if fighter_first else fighter_key
                fighter_methods[key][method] = prob
                # Also store by last name only for fallback matching
                if fighter_key not in fighter_methods or method not in fighter_methods[fighter_key]:
                    fighter_methods[fighter_key][method] = prob

    return dict(fighter_methods)


def match_fighter(fighter_name, method_data):
    """Try to match a fighter name to method data using various strategies."""
    fn = first_name(fighter_name)
    ln = last_name(fighter_name)

    # Try full key first
    full_key = f"{fn}_{ln}"
    if full_key in method_data:
        return method_data[full_key]

    # Try last name
    if ln in method_data:
        return method_data[ln]

    # Try fuzzy matching on last name
    for key in method_data:
        key_parts = key.split("_")
        key_last = key_parts[-1] if key_parts else key
        if key_last == ln:
            return method_data[key]

    # Try partial matching (e.g., "dos Santos" -> "santos")
    for key in method_data:
        if ln in key or key.endswith(ln):
            return method_data[key]

    return None


def find_best_event_url(search_results, event_name, event_date):
    """Pick the best matching event URL from search results."""
    event_lower = event_name.lower()
    date_year = event_date[:4]

    # Extract event number if present (e.g., "UFC 282" -> "282")
    num_match = re.search(r'ufc\s+(\d+)', event_lower)
    event_num = num_match.group(1) if num_match else None

    scored = []
    for text, url in search_results:
        text_lower = text.lower()
        score = 0

        # Exact event number match is very strong signal
        if event_num:
            url_num = re.search(r'ufc-(\d+)', url.lower())
            if url_num and url_num.group(1) == event_num:
                score += 100
            elif event_num in text_lower:
                score += 50

        # Check for key name matches
        for word in event_lower.split():
            if len(word) > 3 and word in text_lower:
                score += 5

        # Check for year in URL text
        if date_year in text_lower or date_year in url:
            score += 10

        # Penalize wrong promotions
        if 'bellator' in text_lower and 'bellator' not in event_lower:
            score -= 50
        if 'one' in text_lower.split() and 'one' not in event_lower.lower().split():
            score -= 30

        scored.append((score, text, url))

    scored.sort(key=lambda x: -x[0])
    if scored and scored[0][0] > 0:
        return scored[0][2], scored[0][1]
    return None, None


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"processed_events": [], "recovered": [], "failed_events": [], "event_url_cache": {}}


def save_checkpoint(ckpt):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f, indent=2)


def main():
    missing = compute_missing_fights()
    print(f"Total missing fights: {len(missing)}")

    # Group by event
    event_groups = defaultdict(list)
    for _, r in missing.iterrows():
        key = f"{r['event_date']}|{r['event_name']}"
        event_groups[key].append(r)

    print(f"Unique events: {len(event_groups)}")

    ckpt = load_checkpoint()
    processed = set(ckpt["processed_events"])
    all_recovered = ckpt["recovered"]
    event_url_cache = ckpt.get("event_url_cache", {})

    events_to_process = sorted(k for k in event_groups if k not in processed)
    print(f"Events remaining: {len(events_to_process)}")

    if not events_to_process:
        print("All events already processed!")
        return

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        total_recovered = 0
        total_missed = 0

        for event_idx, event_key in enumerate(events_to_process):
            event_date, event_name = event_key.split("|", 1)
            fighters = event_groups[event_key]
            print(f"\n[{event_idx+1}/{len(events_to_process)}] {event_date} | {event_name} | {len(fighters)} fights")

            # Step 1: Find the BFO event URL
            event_url = event_url_cache.get(event_key)

            if not event_url:
                # Search by event name
                results = search_bfo_event(page, event_name)
                time.sleep(BFO_DELAY)

                url, matched_name = find_best_event_url(results, event_name, event_date)

                if not url:
                    # Fallback: search by fighter names
                    for fight in fighters[:2]:
                        results2 = search_bfo_by_fighters(page, fight["fighter1"], fight["fighter2"])
                        time.sleep(BFO_DELAY)
                        results.extend(results2)

                    url, matched_name = find_best_event_url(results, event_name, event_date)

                if url:
                    event_url = url
                    event_url_cache[event_key] = url
                    print(f"  Found: {matched_name} -> {url}")
                else:
                    print(f"  FAILED: No BFO event URL found")
                    ckpt["processed_events"].append(event_key)
                    ckpt["failed_events"].append(event_key)
                    ckpt["event_url_cache"] = event_url_cache
                    save_checkpoint(ckpt)
                    total_missed += len(fighters)
                    continue

            # Step 2: Load the event page and extract method props
            try:
                page.goto(event_url, timeout=30000)
                page.wait_for_timeout(4000)  # Wait for JS to render odds

                method_data = extract_method_props(page)
                print(f"  Extracted method props for {len(method_data)} fighters")

            except Exception as e:
                print(f"  ERROR loading page: {e}")
                ckpt["processed_events"].append(event_key)
                ckpt["failed_events"].append(event_key)
                ckpt["event_url_cache"] = event_url_cache
                save_checkpoint(ckpt)
                total_missed += len(fighters)
                continue

            # Step 3: Match to missing fighters
            event_recovered = 0
            for fight in fighters:
                f1 = fight["fighter1"]
                f2 = fight["fighter2"]

                f1_methods = match_fighter(f1, method_data)
                f2_methods = match_fighter(f2, method_data)

                a_ko = f1_methods.get("ko") if f1_methods else None
                a_sub = f1_methods.get("sub") if f1_methods else None
                a_dec = f1_methods.get("dec") if f1_methods else None
                b_ko = f2_methods.get("ko") if f2_methods else None
                b_sub = f2_methods.get("sub") if f2_methods else None
                b_dec = f2_methods.get("dec") if f2_methods else None

                found_any = any(v is not None for v in [a_ko, a_sub, a_dec, b_ko, b_sub, b_dec])

                if found_any:
                    rec = {
                        "event_date": event_date,
                        "fighter_a": f1,
                        "fighter_b": f2,
                        "event_title": event_name,
                        "source": "bfo_comprehensive",
                        "source_url": event_url,
                        "captured_at": "2026-03-18T00:00:00Z",
                        "a_ko_odds_prob": round(a_ko, 6) if a_ko else "",
                        "a_sub_odds_prob": round(a_sub, 6) if a_sub else "",
                        "a_dec_odds_prob": round(a_dec, 6) if a_dec else "",
                        "b_ko_odds_prob": round(b_ko, 6) if b_ko else "",
                        "b_sub_odds_prob": round(b_sub, 6) if b_sub else "",
                        "b_dec_odds_prob": round(b_dec, 6) if b_dec else "",
                    }
                    all_recovered.append(rec)
                    event_recovered += 1
                    total_recovered += 1
                    methods_found = []
                    if a_ko: methods_found.append(f"a_ko={a_ko:.3f}")
                    if a_sub: methods_found.append(f"a_sub={a_sub:.3f}")
                    if a_dec: methods_found.append(f"a_dec={a_dec:.3f}")
                    if b_ko: methods_found.append(f"b_ko={b_ko:.3f}")
                    if b_sub: methods_found.append(f"b_sub={b_sub:.3f}")
                    if b_dec: methods_found.append(f"b_dec={b_dec:.3f}")
                    print(f"    FOUND: {f1} vs {f2} | {', '.join(methods_found)}")
                else:
                    total_missed += 1
                    # Show what method data we DID find for debugging
                    if method_data:
                        available = list(method_data.keys())[:5]
                        print(f"    MISS:  {f1} vs {f2} | available keys: {available}")
                    else:
                        print(f"    MISS:  {f1} vs {f2} | no method data on page")

            print(f"  Recovered {event_recovered}/{len(fighters)}")

            # Save checkpoint after each event
            ckpt["processed_events"].append(event_key)
            ckpt["recovered"] = all_recovered
            ckpt["event_url_cache"] = event_url_cache
            save_checkpoint(ckpt)

            time.sleep(BFO_DELAY)

        browser.close()

    # Save all recovered to CSV
    if all_recovered:
        rec_df = pd.DataFrame(all_recovered)
        rec_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSaved {len(all_recovered)} recovered records to {OUTPUT_PATH}")

        # Also merge into the main method odds file
        existing = pd.read_csv(METHOD_ODDS_PATH)
        combined = pd.concat([existing, rec_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["event_date", "fighter_a", "fighter_b"], keep="last"
        )
        combined.to_csv(METHOD_ODDS_PATH, index=False)
        print(f"Method odds file now has {len(combined)} rows")

    print(f"\n=== SUMMARY ===")
    print(f"Total recovered: {total_recovered}")
    print(f"Total missed: {total_missed}")
    print(f"Events processed: {len(ckpt['processed_events'])}")
    print(f"Events failed: {len(ckpt.get('failed_events', []))}")


if __name__ == "__main__":
    main()
