"""
Wayback Machine scraper for BFO method props on events where the live page has empty odds.

Targets 2014-2018 events where BFO has method prop labels but no odds values.
The Wayback Machine archived these pages when the odds were still populated.
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
import requests
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz

sys.stdout.reconfigure(encoding='utf-8')
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

REPO_ROOT = Path(__file__).resolve().parent.parent
METHOD_ODDS_PATH = REPO_ROOT / "data/raw/method_odds/historical_method_odds_all.csv"
OUTPUT_PATH = REPO_ROOT / "data/raw/method_odds/wayback_bfo_recovery.csv"
CHECKPOINT_PATH = REPO_ROOT / "data/raw/method_odds/wayback_bfo_checkpoint.json"
V2_CHECKPOINT = REPO_ROOT / "data/raw/method_odds/bfo_comprehensive_v2_checkpoint.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
WB_DELAY = 4.0  # Be respectful to Wayback


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


def get_wayback_snapshot(url, max_snapshots=5):
    """Get the best Wayback Machine snapshot for a URL."""
    try:
        cdx = requests.get(
            "https://web.archive.org/cdx/search/cdx",
            params={
                "url": url,
                "output": "json",
                "limit": max_snapshots,
                "fl": "timestamp,statuscode",
                "filter": "statuscode:200",
            },
            headers=HEADERS,
            timeout=30,
        )
        if cdx.status_code != 200:
            return []
        data = cdx.json()
        if len(data) <= 1:
            return []
        return [row[0] for row in data[1:]]
    except Exception as e:
        print(f"    CDX error: {e}")
        return []


def fetch_wayback_page(url, timestamp):
    """Fetch a specific Wayback Machine snapshot."""
    try:
        wb_url = f"https://web.archive.org/web/{timestamp}id_/{url}"
        resp = requests.get(wb_url, headers=HEADERS, timeout=45)
        if resp.status_code == 200:
            return resp.text
    except Exception as e:
        print(f"    Fetch error: {e}")
    return None


def extract_method_props_from_html(html):
    """Extract method props from a BFO HTML page."""
    soup = BeautifulSoup(html, 'lxml')
    rows = soup.select('tr')
    all_props = {}

    for row in rows:
        cells = row.select('td, th')
        if not cells:
            continue

        label = cells[0].get_text(strip=True)
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
                val = parse_american(cell.get_text(strip=True))
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
    """Fuzzy match a fighter to BFO props."""
    fn = norm(fighter_name)
    ln = last_name(fighter_name)

    # Exact match
    if fn in all_props:
        return all_props[fn]

    # Last name match
    for bfo_name, data in all_props.items():
        if bfo_name.split()[-1] == ln:
            return data

    # Fuzzy match
    best_score = 0
    best_match = None
    for bfo_name, data in all_props.items():
        score = max(fuzz.ratio(fn, bfo_name), fuzz.ratio(ln, bfo_name.split()[-1]) * 1.2)
        if score > best_score:
            best_score = score
            best_match = data

    if best_score >= 70:
        return best_match

    # Partial match
    for bfo_name, data in all_props.items():
        if ln in bfo_name:
            return data

    return None


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
    print(f"Total missing fights: {len(missing)}")

    # Get event URLs from v2 checkpoint
    v2_url_cache = {}
    if V2_CHECKPOINT.exists():
        with open(V2_CHECKPOINT) as f:
            v2_data = json.load(f)
            # The v2 checkpoint doesn't store URLs directly, but we can use the event slug
            # to construct the BFO URL

    # Focus on events where BFO has labels but empty odds (2014-2018)
    target = missing[missing["year"] <= 2018].copy()
    print(f"Target fights (2014-2018): {len(target)}")

    event_groups = defaultdict(list)
    for _, r in target.iterrows():
        key = f"{r['event_date']}|{r['event_name']}"
        event_groups[key].append((r["fighter1"], r["fighter2"]))

    print(f"Unique events: {len(event_groups)}")

    ckpt = load_checkpoint()
    processed = set(ckpt["processed_events"])
    all_recovered = ckpt["recovered"]

    events_to_process = sorted(k for k in event_groups if k not in processed)
    print(f"Events remaining: {len(events_to_process)}")

    for event_idx, event_key in enumerate(events_to_process):
        event_date, event_name = event_key.split("|", 1)
        fighters = event_groups[event_key]
        print(f"\n[{event_idx+1}/{len(events_to_process)}] {event_date} | {event_name} | {len(fighters)} fights")

        # Step 1: Find BFO event URL via search
        search_query = event_name[:60]
        num_match = re.search(r'UFC\s+(\d+)', event_name)
        if num_match:
            search_query = f"UFC {num_match.group(1)}"

        # Use BFO search to find event URL
        try:
            resp = requests.get(
                "https://www.bestfightodds.com/search",
                params={"query": search_query},
                headers=HEADERS,
                timeout=20,
            )
            time.sleep(WB_DELAY)

            event_url = None
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'lxml')
                for link in soup.select('a[href*="/events/"]'):
                    href = link.get("href", "")
                    text = link.get_text(strip=True)
                    if "/events/" in href and text:
                        if num_match:
                            ufc_num = num_match.group(1)
                            if re.search(rf'ufc-{ufc_num}\b', href.lower()):
                                event_url = f"https://www.bestfightodds.com{href}"
                                print(f"  BFO URL: {event_url.split('/')[-1]}")
                                break
                        else:
                            # Match by headliner names
                            headliner_match = re.search(r':\s*(.+?)$', event_name)
                            if headliner_match:
                                for name in headliner_match.group(1).lower().split(' vs '):
                                    if name.strip().split()[-1] in href.lower():
                                        event_url = f"https://www.bestfightodds.com{href}"
                                        print(f"  BFO URL: {event_url.split('/')[-1]}")
                                        break
                            if event_url:
                                break
        except Exception as e:
            print(f"  Search error: {e}")

        if not event_url:
            # Fallback: search by fighter names
            for f1, f2 in fighters[:1]:
                try:
                    resp = requests.get(
                        "https://www.bestfightodds.com/search",
                        params={"query": f"{last_name(f1)} {last_name(f2)}"},
                        headers=HEADERS,
                        timeout=20,
                    )
                    time.sleep(WB_DELAY)
                    if resp.status_code == 200:
                        soup = BeautifulSoup(resp.text, 'lxml')
                        for link in soup.select('a[href*="/events/"]'):
                            href = link.get("href", "")
                            if "/events/" in href:
                                event_url = f"https://www.bestfightodds.com{href}"
                                print(f"  BFO URL (via fighter search): {event_url.split('/')[-1]}")
                                break
                except Exception:
                    pass

        if not event_url:
            print(f"  NO BFO URL found")
            ckpt["processed_events"].append(event_key)
            save_checkpoint(ckpt)
            continue

        # Step 2: Check Wayback for snapshots
        timestamps = get_wayback_snapshot(event_url)
        time.sleep(WB_DELAY)

        if not timestamps:
            print(f"  NO WAYBACK SNAPSHOTS")
            ckpt["processed_events"].append(event_key)
            save_checkpoint(ckpt)
            continue

        print(f"  Found {len(timestamps)} snapshots: {timestamps}")

        # Step 3: Try each snapshot until we find one with method props
        all_props = {}
        used_timestamp = None
        for ts in timestamps:
            html = fetch_wayback_page(event_url, ts)
            time.sleep(WB_DELAY)

            if html:
                props = extract_method_props_from_html(html)
                if len(props) > len(all_props):
                    all_props = props
                    used_timestamp = ts
                    if len(props) >= 4:  # Good enough
                        break

        if not all_props:
            print(f"  SNAPSHOTS HAVE NO METHOD PROPS")
            ckpt["processed_events"].append(event_key)
            save_checkpoint(ckpt)
            continue

        print(f"  Extracted props for {len(all_props)} fighters from snapshot {used_timestamp}")

        # Step 4: Match fighters
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
                    "source": "wayback_bfo_v2",
                    "source_url": f"https://web.archive.org/web/{used_timestamp}/{event_url}",
                    "captured_at": f"{used_timestamp[:4]}-{used_timestamp[4:6]}-{used_timestamp[6:8]}",
                    "a_ko_odds_prob": round(a_ko, 6) if a_ko else "",
                    "a_sub_odds_prob": round(a_sub, 6) if a_sub else "",
                    "a_dec_odds_prob": round(a_dec, 6) if a_dec else "",
                    "b_ko_odds_prob": round(b_ko, 6) if b_ko else "",
                    "b_sub_odds_prob": round(b_sub, 6) if b_sub else "",
                    "b_dec_odds_prob": round(b_dec, 6) if b_dec else "",
                }
                all_recovered.append(rec)
                event_recovered += 1
                print(f"    FOUND: {f1} vs {f2}")
            else:
                ln1, ln2 = last_name(f1), last_name(f2)
                avail = list(all_props.keys())[:8]
                print(f"    MISS:  {f1} vs {f2} (need {ln1}/{ln2}) | page: {avail}")

        print(f"  Recovered {event_recovered}/{len(fighters)}")

        ckpt["processed_events"].append(event_key)
        ckpt["recovered"] = all_recovered
        save_checkpoint(ckpt)

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

    print(f"\nTotal recovered: {len(all_recovered)}")


if __name__ == "__main__":
    main()
