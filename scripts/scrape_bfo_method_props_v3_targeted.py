"""
Targeted second-pass BFO method prop scraper.

Focuses on 2018+ fights that were missed by v2 due to:
1. Name mismatches (Cyborg vs Justino, Blachowicz vs Brachowicz, etc.)
2. Events classified as "no props" that actually have props
3. Fights on events that were processed but fighters not matched

Uses fuzzy matching and manual name aliases.
"""
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

sys.stdout.reconfigure(encoding='utf-8')
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

REPO_ROOT = Path(__file__).resolve().parent.parent
METHOD_ODDS_PATH = REPO_ROOT / "data/raw/method_odds/historical_method_odds_all.csv"
OUTPUT_PATH = REPO_ROOT / "data/raw/method_odds/bfo_v3_targeted_recovery.csv"
CHECKPOINT_PATH = REPO_ROOT / "data/raw/method_odds/bfo_v3_checkpoint.json"
V2_CHECKPOINT = REPO_ROOT / "data/raw/method_odds/bfo_comprehensive_v2_checkpoint.json"
BFO_DELAY = 2.5

# Known name aliases: our_name -> BFO_name(s)
NAME_ALIASES = {
    "cristiane justino": ["cyborg", "cris cyborg"],
    "valentina shevchenko": ["shevchenko", "valentina"],
    "jan blachowicz": ["brachowicz", "blachowicz"],
    "zhang weili": ["weili", "zhang"],
    "song yadong": ["yadong", "song"],
    "song kenan": ["kenan", "song"],
    "wu yanan": ["wu", "yanan"],
    "kyung ho kang": ["kang"],
    "da woon jung": ["jung"],
    "jeong yeong lee": ["lee"],
    "montserrat conejo ruiz": ["ruiz", "conejo"],
    "mauricio rua": ["shogun", "rua"],
    "dooho choi": ["choi"],
    "tatsuro taira": ["taira"],
    "amanda nunes": ["nunes"],
}


def norm(n):
    if pd.isna(n):
        return ""
    n = unicodedata.normalize("NFKD", str(n))
    n = "".join(c for c in n if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-zA-Z\s]", "", n).lower().split())


def last_name(n):
    parts = norm(n).split()
    return parts[-1] if parts else ""


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
                "year": r["event_date"].year,
            })

    return pd.DataFrame(missing)


def extract_all_method_props(page):
    """Extract ALL method props from page, returning raw fighter->method->prob mapping."""
    rows = page.query_selector_all("tr")
    all_props = {}  # full_label -> {method: prob, raw_name: str}

    for row in rows:
        cells = row.query_selector_all("td, th")
        if not cells:
            continue

        label = cells[0].inner_text().strip()
        label_lower = label.lower()

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
                fn_norm = norm(fighter_name)
                if fn_norm not in all_props:
                    all_props[fn_norm] = {"raw_name": fighter_name}
                all_props[fn_norm][method] = prob

    return all_props


def fuzzy_match_fighter(fighter_name, all_props):
    """Try multiple strategies to match a fighter to BFO method props."""
    fn = norm(fighter_name)
    ln = last_name(fighter_name)

    # Strategy 1: Exact full name match
    if fn in all_props:
        return all_props[fn]

    # Strategy 2: Last name match
    for bfo_name, data in all_props.items():
        if bfo_name.split()[-1] == ln:
            return data

    # Strategy 3: Check aliases
    for alias_key, aliases in NAME_ALIASES.items():
        if fn == alias_key or ln == alias_key.split()[-1]:
            for alias in aliases:
                alias_norm = norm(alias)
                for bfo_name, data in all_props.items():
                    if alias_norm in bfo_name or bfo_name.endswith(alias_norm):
                        return data

    # Strategy 4: Fuzzy match on full name
    best_score = 0
    best_match = None
    for bfo_name, data in all_props.items():
        score = fuzz.ratio(fn, bfo_name)
        # Also try last name vs last name
        bfo_last = bfo_name.split()[-1] if bfo_name.split() else ""
        ln_score = fuzz.ratio(ln, bfo_last)
        max_score = max(score, ln_score * 1.2)  # Boost last-name matches
        if max_score > best_score:
            best_score = max_score
            best_match = data

    if best_score >= 70:
        return best_match

    # Strategy 5: Partial name match
    for bfo_name, data in all_props.items():
        if ln in bfo_name or any(part in bfo_name for part in fn.split() if len(part) > 3):
            return data

    return None


def search_and_load_event(page, event_name, fighters):
    """Search BFO, find and load the correct event page."""
    # Build search queries
    queries = []

    # UFC number
    num_match = re.search(r'UFC\s+(\d+)', event_name)
    if num_match:
        queries.append(f"UFC {num_match.group(1)}")

    # Headliner names
    headliner_match = re.search(r':\s*(.+?)$', event_name)
    if headliner_match:
        headliners = headliner_match.group(1).strip()
        queries.append(headliners)

    # Fighter names
    for f1, f2 in fighters[:2]:
        queries.append(f"{last_name(f1)} {last_name(f2)}")

    # Full event name
    queries.append(event_name[:60])

    for query in queries[:4]:
        try:
            page.goto(f"https://www.bestfightodds.com/search?query={query}", timeout=20000)
            page.wait_for_timeout(1500)

            links = page.query_selector_all('a[href*="/events/"]')
            for link in links[:10]:
                href = link.get_attribute("href") or ""
                text = link.inner_text().strip()
                if "/events/" in href and text:
                    # Score this result
                    if num_match:
                        ufc_num = num_match.group(1)
                        if re.search(rf'ufc-{ufc_num}\b', href.lower()):
                            url = f"https://www.bestfightodds.com{href}" if not href.startswith("http") else href
                            return url, text

                    # Check for headliner names in URL
                    if headliner_match:
                        for name in headliner_match.group(1).lower().split(' vs '):
                            name_last = name.strip().split()[-1]
                            if name_last in href.lower():
                                url = f"https://www.bestfightodds.com{href}" if not href.startswith("http") else href
                                return url, text

            time.sleep(BFO_DELAY)
        except Exception as e:
            print(f"    Search error: {e}")
            time.sleep(BFO_DELAY)

    return None, None


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"processed_events": [], "recovered": []}


def save_checkpoint(ckpt):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f, indent=2)


def main():
    missing = compute_missing_fights()
    # Focus on 2018+ fights only (2014-2017 don't have populated method props on BFO)
    target = missing[missing["year"] >= 2018].copy()
    print(f"Total 2018+ missing fights: {len(target)}")

    # Group by event
    event_groups = defaultdict(list)
    for _, r in target.iterrows():
        key = f"{r['event_date']}|{r['event_name']}"
        event_groups[key].append((r["fighter1"], r["fighter2"]))

    print(f"Unique events: {len(event_groups)}")

    # Load v2 checkpoint for event URL cache
    v2_ckpt = {}
    if V2_CHECKPOINT.exists():
        with open(V2_CHECKPOINT) as f:
            v2_data = json.load(f)
            # v2 doesn't have URL cache in format we need, but we can try

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

        for event_idx, event_key in enumerate(events_to_process):
            event_date, event_name = event_key.split("|", 1)
            fighters = event_groups[event_key]
            print(f"\n[{event_idx+1}/{len(events_to_process)}] {event_date} | {event_name} | {len(fighters)} fights")

            # Find and load event page
            event_url, matched = search_and_load_event(page, event_name, fighters)

            if not event_url:
                print(f"  NO URL FOUND")
                ckpt["processed_events"].append(event_key)
                save_checkpoint(ckpt)
                continue

            print(f"  URL: {event_url.split('/')[-1]} ({matched})")

            try:
                page.goto(event_url, timeout=30000)
                page.wait_for_timeout(5000)  # Extra wait for JS
            except Exception as e:
                print(f"  LOAD ERROR: {e}")
                ckpt["processed_events"].append(event_key)
                save_checkpoint(ckpt)
                continue

            # Validate page
            title = page.title()
            if 'best fight odds' in title.lower() and 'for' not in title.lower():
                # Redirected to homepage
                print(f"  REDIRECTED TO HOMEPAGE")
                ckpt["processed_events"].append(event_key)
                save_checkpoint(ckpt)
                continue

            # Extract ALL method props
            all_props = extract_all_method_props(page)
            if not all_props:
                print(f"  NO METHOD PROPS (empty odds)")
                ckpt["processed_events"].append(event_key)
                save_checkpoint(ckpt)
                continue

            print(f"  Found props for {len(all_props)} fighters: {list(all_props.keys())[:8]}")

            # Match fighters with fuzzy matching
            event_recovered = 0
            for f1, f2 in fighters:
                f1_data = fuzzy_match_fighter(f1, all_props)
                f2_data = fuzzy_match_fighter(f2, all_props)

                a_ko = f1_data.get("ko") if f1_data else None
                a_sub = f1_data.get("sub") if f1_data else None
                a_dec = f1_data.get("dec") if f1_data else None
                b_ko = f2_data.get("ko") if f2_data else None
                b_sub = f2_data.get("sub") if f2_data else None
                b_dec = f2_data.get("dec") if f2_data else None

                found_any = any(v is not None for v in [a_ko, a_sub, a_dec, b_ko, b_sub, b_dec])

                if found_any:
                    rec = {
                        "event_date": event_date,
                        "fighter_a": f1,
                        "fighter_b": f2,
                        "event_title": event_name,
                        "source": "bfo_v3_targeted",
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
                    matched_names = []
                    if f1_data: matched_names.append(f"a={f1_data.get('raw_name','?')}")
                    if f2_data: matched_names.append(f"b={f2_data.get('raw_name','?')}")
                    print(f"    FOUND: {f1} vs {f2} | {', '.join(matched_names)}")
                else:
                    ln1, ln2 = last_name(f1), last_name(f2)
                    avail = [f"{k}({','.join(m for m in ['ko','sub','dec'] if m in v)})" for k, v in list(all_props.items())[:10]]
                    print(f"    MISS:  {f1} vs {f2} (need {ln1}/{ln2}) | page: {avail}")

            print(f"  Recovered {event_recovered}/{len(fighters)}")

            ckpt["processed_events"].append(event_key)
            ckpt["recovered"] = all_recovered
            save_checkpoint(ckpt)
            time.sleep(BFO_DELAY)

        browser.close()

    # Save and merge
    if all_recovered:
        rec_df = pd.DataFrame(all_recovered)
        rec_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSaved {len(all_recovered)} to {OUTPUT_PATH}")

        existing = pd.read_csv(METHOD_ODDS_PATH)
        combined = pd.concat([existing, rec_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["event_date", "fighter_a", "fighter_b"], keep="last")
        combined.to_csv(METHOD_ODDS_PATH, index=False)
        print(f"Method odds file now has {len(combined)} rows")

    print(f"\nTotal recovered this pass: {len(all_recovered)}")


if __name__ == "__main__":
    main()
