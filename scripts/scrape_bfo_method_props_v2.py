"""
BFO method prop scraper v2 — with page validation and smart URL discovery.

Key fixes from v1:
- Validates that the loaded page actually matches the expected event
- Handles BFO redirects to homepage/wrong pages
- Searches by multiple strategies (event name, fighter names, event number)
- Only processes 2017+ events (method props didn't exist before ~2017)
- Also attempts 2014-2016 events but expects most to have no method props
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
# Force line-buffered output
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

REPO_ROOT = Path(__file__).resolve().parent.parent
METHOD_ODDS_PATH = REPO_ROOT / "data/raw/method_odds/historical_method_odds_all.csv"
OUTPUT_PATH = REPO_ROOT / "data/raw/method_odds/bfo_comprehensive_recovery.csv"
CHECKPOINT_PATH = REPO_ROOT / "data/raw/method_odds/bfo_comprehensive_v2_checkpoint.json"
BFO_DELAY = 2.5


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


def validate_page(page, event_name, expected_fighters):
    """Check if the loaded page actually shows the expected event."""
    title = page.title().lower()

    # If title contains the event name or number, it's valid
    event_lower = event_name.lower()

    # Extract UFC number if present
    num_match = re.search(r'ufc\s+(\d+)', event_lower)
    if num_match:
        ufc_num = num_match.group(1)
        if f'ufc {ufc_num}' in title or f'ufc-{ufc_num}' in title:
            return True

    # Check for event name keywords in title
    for word in event_lower.split():
        if len(word) > 4 and word in title:
            return True

    # Check if any expected fighters appear on the page
    body_text = page.inner_text('body').lower()
    for f1, f2 in expected_fighters:
        ln1, ln2 = last_name(f1).lower(), last_name(f2).lower()
        if ln1 in body_text or ln2 in body_text:
            return True

    # Generic homepage or wrong event
    if 'best fight odds' in title and 'for' not in title:
        return False

    return False


def search_bfo_urls(page, queries):
    """Search BFO with multiple queries and collect all event URLs."""
    all_results = []
    seen_urls = set()

    for query in queries:
        try:
            page.goto(f"https://www.bestfightodds.com/search?query={query[:80]}", timeout=20000)
            page.wait_for_timeout(1500)

            links = page.query_selector_all('a[href*="/events/"]')
            for link in links:
                href = link.get_attribute("href") or ""
                text = link.inner_text().strip()
                if "/events/" in href and text and href not in seen_urls:
                    full_url = f"https://www.bestfightodds.com{href}" if not href.startswith("http") else href
                    seen_urls.add(href)
                    all_results.append((text, full_url))

            time.sleep(BFO_DELAY)
        except Exception as e:
            print(f"    Search error for '{query}': {e}")
            time.sleep(BFO_DELAY)

    return all_results


def score_url(text, url, event_name, event_date):
    """Score how well a URL matches our target event."""
    text_lower = text.lower()
    url_lower = url.lower()
    event_lower = event_name.lower()
    date_year = event_date[:4]
    score = 0

    # Extract UFC event number
    num_match = re.search(r'ufc\s+(\d+)', event_lower)
    if num_match:
        event_num = num_match.group(1)
        if re.search(rf'ufc[- ]{event_num}\b', url_lower):
            score += 100
        elif event_num in text_lower:
            score += 50

    # Check "Fight Night" number match
    fn_match = re.search(r'fight night[:\s]+(\w+)', event_lower)
    if fn_match:
        fn_id = fn_match.group(1).lower()
        if fn_id in url_lower:
            score += 30

    # Check for headliner names in URL
    headliner_match = re.search(r':\s*(.+?)(?:\s+\d|$)', event_name)
    if headliner_match:
        headliners = headliner_match.group(1).lower().split(' vs ')
        for h in headliners:
            h = h.strip().split()[-1]  # Last name
            if h in url_lower:
                score += 40

    # Year match
    if date_year in text_lower:
        score += 10

    # Penalize wrong promotions
    if 'bellator' in url_lower and 'bellator' not in event_lower:
        score -= 100
    if 'one-' in url_lower and 'one' not in event_lower.split():
        score -= 50
    if 'pfl' in url_lower and 'pfl' not in event_lower:
        score -= 50

    return score


def find_event_url(page, event_name, event_date, fighters):
    """Find the correct BFO URL for an event using multiple search strategies."""
    # Build search queries
    queries = []

    # Strategy 1: Full event name
    queries.append(event_name)

    # Strategy 2: UFC number
    num_match = re.search(r'UFC\s+(\d+)', event_name)
    if num_match:
        queries.append(f"UFC {num_match.group(1)}")

    # Strategy 3: Headliner names
    headliner_match = re.search(r':\s*(.+?)$', event_name)
    if headliner_match:
        queries.append(headliner_match.group(1).strip())

    # Strategy 4: Fighter names from missing fights
    for f1, f2 in fighters[:2]:
        queries.append(f"{last_name(f1)} {last_name(f2)}")

    # Remove duplicates while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique_queries.append(q)

    results = search_bfo_urls(page, unique_queries[:4])

    if not results:
        return None

    # Score and rank results
    scored = [(score_url(text, url, event_name, event_date), text, url) for text, url in results]
    scored.sort(key=lambda x: -x[0])

    # Return the best match if score is positive
    if scored and scored[0][0] > 0:
        return scored[0][2]

    return None


def extract_method_props(page):
    """Extract method-of-victory props from a BFO event page."""
    rows = page.query_selector_all("tr")
    fighter_methods = defaultdict(lambda: defaultdict(list))

    for row in rows:
        cells = row.query_selector_all("td, th")
        if not cells:
            continue

        label = cells[0].inner_text().strip()
        label_lower = label.lower()

        # Match "Fighter wins by TKO/KO" (exclude round-specific and technique-specific)
        method = None
        if re.search(r'wins by (tko/?ko|ko/?tko|knockout)', label_lower):
            if 'round' not in label_lower and 'via' not in label_lower:
                method = "ko"
        elif re.search(r'wins by sub', label_lower):
            if 'round' not in label_lower:
                method = "sub"
        elif re.search(r'wins by (decision|dec)', label_lower):
            if not any(w in label_lower for w in ['unanimous', 'split', 'majority']):
                method = "dec"

        if method:
            fighter_name = re.sub(r"\s*wins\s+by\s+.*$", "", label, flags=re.IGNORECASE).strip()

            odds_vals = []
            for cell in cells[1:]:
                val = parse_american(cell.inner_text().strip())
                if val is not None:
                    odds_vals.append(val)

            if odds_vals:
                avg_odds = np.mean(odds_vals)
                prob = american_to_prob(avg_odds)
                ln = last_name(fighter_name)
                fn = first_name(fighter_name)
                fighter_methods[(fn, ln)][method] = prob

    # Build lookup: try (first, last) first, fall back to last-name-only
    result = {}
    last_name_map = defaultdict(list)
    for (fn, ln), methods in fighter_methods.items():
        result[f"{fn}_{ln}"] = dict(methods)
        last_name_map[ln].append((fn, dict(methods)))

    # Add last-name-only entries for unique last names
    for ln, entries in last_name_map.items():
        if len(entries) == 1:
            result[ln] = entries[0][1]

    return result, last_name_map


def match_fighter(fighter_name, result_dict, last_name_map):
    """Match a fighter to method data."""
    fn = first_name(fighter_name)
    ln = last_name(fighter_name)

    # Try exact (first_last) key
    key = f"{fn}_{ln}"
    if key in result_dict:
        return result_dict[key]

    # Try last name only (unambiguous)
    if ln in result_dict:
        return result_dict[ln]

    # Try fuzzy: if only one fighter with this last name
    if ln in last_name_map:
        entries = last_name_map[ln]
        if len(entries) == 1:
            return entries[0][1]
        # Multiple fighters with same last name — try first name match
        for entry_fn, methods in entries:
            if entry_fn == fn:
                return methods

    # Try partial last name match (e.g., "dos Santos" -> last part "santos")
    for key in result_dict:
        if key.endswith(f"_{ln}") or key == ln:
            return result_dict[key]

    return None


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"processed_events": [], "recovered": [], "failed_events": [], "no_method_props": []}


def save_checkpoint(ckpt):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f, indent=2)


def main():
    missing = compute_missing_fights()
    print(f"Total missing fights: {len(missing)}")

    event_groups = defaultdict(list)
    for _, r in missing.iterrows():
        key = f"{r['event_date']}|{r['event_name']}"
        event_groups[key].append((r["fighter1"], r["fighter2"]))

    print(f"Unique events: {len(event_groups)}")

    ckpt = load_checkpoint()
    processed = set(ckpt["processed_events"])
    all_recovered = ckpt["recovered"]

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
        total_no_props = 0

        for event_idx, event_key in enumerate(events_to_process):
            event_date, event_name = event_key.split("|", 1)
            fighters = event_groups[event_key]
            print(f"\n[{event_idx+1}/{len(events_to_process)}] {event_date} | {event_name} | {len(fighters)} fights")

            # Step 1: Find the BFO event URL
            event_url = find_event_url(page, event_name, event_date, fighters)

            if not event_url:
                print(f"  NO URL FOUND")
                ckpt["processed_events"].append(event_key)
                ckpt["failed_events"].append(event_key)
                save_checkpoint(ckpt)
                total_missed += len(fighters)
                continue

            print(f"  URL: {event_url.split('/')[-1]}")

            # Step 2: Load the event page
            try:
                page.goto(event_url, timeout=30000)
                page.wait_for_timeout(4000)
            except Exception as e:
                print(f"  LOAD ERROR: {e}")
                ckpt["processed_events"].append(event_key)
                ckpt["failed_events"].append(event_key)
                save_checkpoint(ckpt)
                total_missed += len(fighters)
                continue

            # Step 3: Validate the page
            if not validate_page(page, event_name, fighters):
                page_title = page.title()
                print(f"  WRONG PAGE: '{page_title[:60]}'")

                # Try one more time with a fighter-name search
                for f1, f2 in fighters[:1]:
                    alt_results = search_bfo_urls(page, [f"{f1} {f2}"])
                    if alt_results:
                        alt_scored = [(score_url(t, u, event_name, event_date), t, u) for t, u in alt_results]
                        alt_scored.sort(key=lambda x: -x[0])
                        if alt_scored and alt_scored[0][0] > 0:
                            event_url = alt_scored[0][2]
                            print(f"  RETRY URL: {event_url.split('/')[-1]}")
                            page.goto(event_url, timeout=30000)
                            page.wait_for_timeout(4000)
                            if not validate_page(page, event_name, fighters):
                                print(f"  STILL WRONG PAGE after retry")
                                ckpt["processed_events"].append(event_key)
                                ckpt["failed_events"].append(event_key)
                                save_checkpoint(ckpt)
                                total_missed += len(fighters)
                                continue

            # Step 4: Extract method props
            result_dict, last_name_map = extract_method_props(page)
            num_fighters_with_props = len(result_dict)

            if num_fighters_with_props == 0:
                print(f"  NO METHOD PROPS on page (pre-2017 event?)")
                ckpt["processed_events"].append(event_key)
                ckpt["no_method_props"].append(event_key)
                save_checkpoint(ckpt)
                total_no_props += len(fighters)
                continue

            print(f"  Found props for {num_fighters_with_props} fighters")

            # Step 5: Match fighters
            event_recovered = 0
            for f1, f2 in fighters:
                f1_methods = match_fighter(f1, result_dict, last_name_map)
                f2_methods = match_fighter(f2, result_dict, last_name_map)

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
                        "source": "bfo_comprehensive_v2",
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
                    parts = []
                    if a_ko: parts.append(f"a_ko={a_ko:.3f}")
                    if a_sub: parts.append(f"a_sub={a_sub:.3f}")
                    if a_dec: parts.append(f"a_dec={a_dec:.3f}")
                    if b_ko: parts.append(f"b_ko={b_ko:.3f}")
                    if b_sub: parts.append(f"b_sub={b_sub:.3f}")
                    if b_dec: parts.append(f"b_dec={b_dec:.3f}")
                    print(f"    FOUND: {f1} vs {f2} | {', '.join(parts)}")
                else:
                    total_missed += 1
                    ln1, ln2 = last_name(f1), last_name(f2)
                    available = [k for k in result_dict.keys() if not k.startswith('_')][:8]
                    print(f"    MISS:  {f1} vs {f2} (looking for {ln1}/{ln2}) | page has: {available}")

            print(f"  Recovered {event_recovered}/{len(fighters)}")

            ckpt["processed_events"].append(event_key)
            ckpt["recovered"] = all_recovered
            save_checkpoint(ckpt)
            time.sleep(BFO_DELAY)

        browser.close()

    # Save recovered records
    if all_recovered:
        rec_df = pd.DataFrame(all_recovered)
        rec_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSaved {len(all_recovered)} recovered records to {OUTPUT_PATH}")

        # Merge into main method odds file
        existing = pd.read_csv(METHOD_ODDS_PATH)
        combined = pd.concat([existing, rec_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["event_date", "fighter_a", "fighter_b"], keep="last"
        )
        combined.to_csv(METHOD_ODDS_PATH, index=False)
        print(f"Method odds file now has {len(combined)} rows")

    print(f"\n=== SUMMARY ===")
    print(f"Total recovered: {total_recovered}")
    print(f"Total missed (no match): {total_missed}")
    print(f"Total no props (pre-2017): {total_no_props}")
    print(f"Events processed: {len(ckpt['processed_events'])}")
    print(f"Events failed: {len(ckpt.get('failed_events', []))}")
    print(f"Events no props: {len(ckpt.get('no_method_props', []))}")


if __name__ == "__main__":
    main()
