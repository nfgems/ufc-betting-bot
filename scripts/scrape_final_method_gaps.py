"""
Final pass scraper for remaining ~73 missing method odds fights.

Combines all strategies:
1. BFO direct (Playwright) for events that still load correctly
2. Wayback Machine for events that redirect
3. Fuzzy name matching with aliases
4. Tries multiple BFO event URL formats
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
OUTPUT_PATH = REPO_ROOT / "data/raw/method_odds/final_recovery.csv"
CHECKPOINT_PATH = REPO_ROOT / "data/raw/method_odds/final_checkpoint.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DELAY = 3.0


def norm(n):
    if pd.isna(n): return ""
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
    if odds is None or odds == 0: return None
    if odds > 0: return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def compute_missing():
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
            missing.append({"event_date": d, "fighter1": r["fighter1"], "fighter2": r["fighter2"],
                          "event_name": r["event_name"], "year": r["event_date"].year})
    return pd.DataFrame(missing)


def extract_props_from_html(html):
    soup = BeautifulSoup(html, 'lxml')
    rows = soup.select('tr')
    all_props = {}
    for row in rows:
        cells = row.select('td, th')
        if not cells: continue
        label = cells[0].get_text(strip=True)
        label_lower = label.lower()
        method = None
        if re.search(r'wins by (tko/?ko|ko/?tko|knockout)', label_lower) and 'round' not in label_lower and 'via' not in label_lower:
            method = "ko"
        elif re.search(r'wins by sub', label_lower) and 'round' not in label_lower:
            method = "sub"
        elif re.search(r'wins by (decision|dec)', label_lower) and not any(w in label_lower for w in ['unanimous', 'split', 'majority']):
            method = "dec"
        if method:
            fighter = re.sub(r"\s*wins\s+by\s+.*$", "", label, flags=re.IGNORECASE).strip()
            odds_vals = [parse_american(c.get_text(strip=True)) for c in cells[1:]]
            odds_vals = [v for v in odds_vals if v is not None]
            if odds_vals:
                fn = norm(fighter)
                if fn not in all_props:
                    all_props[fn] = {"raw": fighter}
                all_props[fn][method] = american_to_prob(np.mean(odds_vals))
    return all_props


def extract_props_playwright(page):
    rows = page.query_selector_all("tr")
    all_props = {}
    for row in rows:
        cells = row.query_selector_all("td, th")
        if not cells: continue
        label = cells[0].inner_text().strip()
        label_lower = label.lower()
        method = None
        if re.search(r'wins by (tko/?ko|ko/?tko|knockout)', label_lower) and 'round' not in label_lower and 'via' not in label_lower:
            method = "ko"
        elif re.search(r'wins by sub', label_lower) and 'round' not in label_lower:
            method = "sub"
        elif re.search(r'wins by (decision|dec)', label_lower) and not any(w in label_lower for w in ['unanimous', 'split', 'majority']):
            method = "dec"
        if method:
            fighter = re.sub(r"\s*wins\s+by\s+.*$", "", label, flags=re.IGNORECASE).strip()
            odds_vals = [parse_american(c.inner_text().strip()) for c in cells[1:]]
            odds_vals = [v for v in odds_vals if v is not None]
            if odds_vals:
                fn = norm(fighter)
                if fn not in all_props:
                    all_props[fn] = {"raw": fighter}
                all_props[fn][method] = american_to_prob(np.mean(odds_vals))
    return all_props


def fuzzy_match(fighter_name, all_props):
    fn = norm(fighter_name)
    ln = last_name(fighter_name)
    if fn in all_props: return all_props[fn]
    for bfo_name, data in all_props.items():
        bfo_last = bfo_name.split()[-1] if bfo_name.split() else ""
        if bfo_last == ln: return data
    best_score, best_match = 0, None
    for bfo_name, data in all_props.items():
        bfo_last = bfo_name.split()[-1] if bfo_name.split() else ""
        score = max(fuzz.ratio(fn, bfo_name), fuzz.ratio(ln, bfo_last) * 1.2)
        if score > best_score:
            best_score = score
            best_match = data
    if best_score >= 65: return best_match
    for bfo_name, data in all_props.items():
        if ln in bfo_name: return data
    return None


def find_bfo_urls(event_name, fighters):
    """Search BFO for event URLs."""
    urls = []
    queries = []
    num_match = re.search(r'UFC\s+(\d+)', event_name)
    if num_match:
        queries.append(f"UFC {num_match.group(1)}")
    headliner = re.search(r':\s*(.+?)$', event_name)
    if headliner:
        queries.append(headliner.group(1).strip())
    for f1, f2 in fighters[:2]:
        queries.append(f"{last_name(f1)} {last_name(f2)}")

    for query in queries[:3]:
        try:
            resp = requests.get("https://www.bestfightodds.com/search",
                              params={"query": query}, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'lxml')
                for link in soup.select('a[href*="/events/"]'):
                    href = link.get("href", "")
                    text = link.get_text(strip=True)
                    if "/events/" in href and text:
                        full = f"https://www.bestfightodds.com{href}" if not href.startswith("http") else href
                        if num_match and re.search(rf'ufc-{num_match.group(1)}\b', href.lower()):
                            urls.insert(0, (text, full))
                        elif headliner:
                            for name in headliner.group(1).lower().split(' vs '):
                                if name.strip().split()[-1] in href.lower():
                                    urls.append((text, full))
                                    break
                        else:
                            urls.append((text, full))
            time.sleep(DELAY)
        except Exception:
            time.sleep(DELAY)

    return urls


def try_wayback(url):
    """Try to get method props from Wayback Machine."""
    try:
        cdx = requests.get(
            "https://web.archive.org/cdx/search/cdx",
            params={"url": url, "output": "json", "limit": 5,
                    "fl": "timestamp,statuscode", "filter": "statuscode:200"},
            headers=HEADERS, timeout=20,
        )
        if cdx.status_code != 200: return {}
        data = cdx.json()
        if len(data) <= 1: return {}

        best_props = {}
        for row in data[1:]:
            ts = row[0]
            time.sleep(DELAY)
            try:
                wb = requests.get(f"https://web.archive.org/web/{ts}id_/{url}",
                                headers=HEADERS, timeout=45)
                if wb.status_code == 200:
                    props = extract_props_from_html(wb.text)
                    if len(props) > len(best_props):
                        best_props = props
                    if len(props) >= 4:
                        break
            except Exception:
                continue
        return best_props
    except Exception:
        return {}


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"processed": [], "recovered": []}


def save_checkpoint(ckpt):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f, indent=2)


def main():
    missing = compute_missing()
    print(f"Total missing: {len(missing)}")

    event_groups = defaultdict(list)
    for _, r in missing.iterrows():
        key = f"{r['event_date']}|{r['event_name']}"
        event_groups[key].append((r["fighter1"], r["fighter2"]))

    ckpt = load_checkpoint()
    processed = set(ckpt["processed"])
    all_recovered = ckpt["recovered"]

    events_to_process = sorted(k for k in event_groups if k not in processed)
    print(f"Events remaining: {len(events_to_process)}")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for idx, event_key in enumerate(events_to_process):
            event_date, event_name = event_key.split("|", 1)
            fighters = event_groups[event_key]
            print(f"\n[{idx+1}/{len(events_to_process)}] {event_date} | {event_name} | {len(fighters)} fights")

            # Step 1: Find BFO URLs
            bfo_results = find_bfo_urls(event_name, fighters)
            if not bfo_results:
                print(f"  NO BFO URLs")
                ckpt["processed"].append(event_key)
                save_checkpoint(ckpt)
                continue

            all_props = {}

            # Step 2: Try each URL via Playwright first
            for text, url in bfo_results[:3]:
                try:
                    page.goto(url, timeout=20000)
                    page.wait_for_timeout(4000)
                    title = page.title()

                    # Check if page is correct (not redirected)
                    if 'for' in title.lower() or event_name.split(':')[0].lower() in title.lower():
                        props = extract_props_playwright(page)
                        if len(props) > len(all_props):
                            all_props = props
                            if len(props) >= 4:
                                print(f"  BFO direct: {url.split('/')[-1]} -> {len(props)} fighters")
                                break
                except Exception:
                    pass
                time.sleep(DELAY)

            # Step 3: If BFO direct didn't work, try Wayback
            if len(all_props) < 2:
                for text, url in bfo_results[:2]:
                    print(f"  Trying Wayback for {url.split('/')[-1]}...")
                    wb_props = try_wayback(url)
                    if len(wb_props) > len(all_props):
                        all_props = wb_props
                        print(f"  Wayback: {len(wb_props)} fighters")
                    if len(all_props) >= 4:
                        break

            if not all_props:
                print(f"  NO METHOD PROPS found anywhere")
                ckpt["processed"].append(event_key)
                save_checkpoint(ckpt)
                continue

            print(f"  Props for {len(all_props)} fighters: {list(all_props.keys())[:8]}")

            # Step 4: Match fighters
            event_recovered = 0
            for f1, f2 in fighters:
                f1_data = fuzzy_match(f1, all_props)
                f2_data = fuzzy_match(f2, all_props)
                a_ko = f1_data.get("ko") if f1_data else None
                a_sub = f1_data.get("sub") if f1_data else None
                a_dec = f1_data.get("dec") if f1_data else None
                b_ko = f2_data.get("ko") if f2_data else None
                b_sub = f2_data.get("sub") if f2_data else None
                b_dec = f2_data.get("dec") if f2_data else None
                found_any = any(v is not None for v in [a_ko, a_sub, a_dec, b_ko, b_sub, b_dec])
                if found_any:
                    rec = {
                        "event_date": event_date, "fighter_a": f1, "fighter_b": f2,
                        "event_title": event_name, "source": "bfo_final_pass",
                        "source_url": "", "captured_at": "2026-03-18",
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
                    print(f"    MISS:  {f1} vs {f2} ({last_name(f1)}/{last_name(f2)})")

            print(f"  Recovered {event_recovered}/{len(fighters)}")
            ckpt["processed"].append(event_key)
            ckpt["recovered"] = all_recovered
            save_checkpoint(ckpt)

        browser.close()

    if all_recovered:
        rec_df = pd.DataFrame(all_recovered)
        rec_df.to_csv(OUTPUT_PATH, index=False)
        existing = pd.read_csv(METHOD_ODDS_PATH)
        combined = pd.concat([existing, rec_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["event_date", "fighter_a", "fighter_b"], keep="last")
        combined.to_csv(METHOD_ODDS_PATH, index=False)
        print(f"\nMerged {len(all_recovered)} into main file ({len(combined)} total)")

    print(f"\nTotal recovered this pass: {len(all_recovered)}")


if __name__ == "__main__":
    main()
