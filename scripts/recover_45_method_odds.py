"""
Targeted Playwright scraper to recover method odds for the 45 missing fights.
Uses BFO event pages with JS rendering to extract KO/SUB/DEC prop odds.
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

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_RECOVERY = REPO_ROOT / "data/raw/method_odds/historical_method_odds_manual_recovery_20260319.csv"
OUTPUT_UNRESOLVED = REPO_ROOT / "data/raw/method_odds/historical_method_odds_manual_unresolved_20260319.csv"
CHECKPOINT_PATH = REPO_ROOT / "data/raw/method_odds/manual_recovery_checkpoint.json"
HEADERS_ROW = [
    "event_date", "fighter_a", "fighter_b", "event_title",
    "source", "source_url", "captured_at",
    "a_ko_odds_prob", "a_sub_odds_prob", "a_dec_odds_prob",
    "b_ko_odds_prob", "b_sub_odds_prob", "b_dec_odds_prob",
]
DELAY = 3.0

# ── The 45 missing fights with known/guessed BFO event URLs ──────────────
FIGHTS = [
    # 2014-06-28: UFC Fight Night 44
    {"date": "2014-06-28", "fa": "Joe Ellenberger", "fb": "James Moontasri",
     "event": "UFC Fight Night: Swanson vs Stephens",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-44-swanson-vs-stephens-800"]},
    {"date": "2014-06-28", "fa": "Cody Gibson", "fb": "Johnny Bedford",
     "event": "UFC Fight Night: Swanson vs Stephens",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-44-swanson-vs-stephens-800"]},
    {"date": "2014-06-28", "fa": "Marcelo Guimaraes", "fb": "Andy Enz",
     "event": "UFC Fight Night: Swanson vs Stephens",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-44-swanson-vs-stephens-800"]},
    {"date": "2014-06-28", "fa": "Ray Borg", "fb": "Shane Howell",
     "event": "UFC Fight Night: Swanson vs Stephens",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-44-swanson-vs-stephens-800"]},
    {"date": "2014-06-28", "fa": "Aleksei Oleinik", "fb": "Anthony Hamilton",
     "event": "UFC Fight Night: Swanson vs Stephens",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-44-swanson-vs-stephens-800"]},
    # 2014-07-06: TUF 19 Finale
    {"date": "2014-07-06", "fa": "Robert Drysdale", "fb": "Keith Berish",
     "event": "The Ultimate Fighter: Team Edgar vs. Team Penn Finale",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-the-ultimate-fighter-19-finale-832"]},
    # 2014-12-12: TUF 20 Finale
    {"date": "2014-12-12", "fa": "KJ Noons", "fb": "Daron Cruickshank",
     "event": "The Ultimate Fighter: A Champion Will Be Crowned Finale",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-the-ultimate-fighter-20-finale-912"]},
    {"date": "2014-12-12", "fa": "Joanne Wood", "fb": "Seo Hee Ham",
     "event": "The Ultimate Fighter: A Champion Will Be Crowned Finale",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-the-ultimate-fighter-20-finale-912"]},
    # 2015-02-28: UFC 184
    {"date": "2015-02-28", "fa": "Roman Salazar", "fb": "Norifumi Yamamoto",
     "event": "UFC 184: Rousey vs Zingano",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-184-rousey-vs-zingano-896"]},
    # 2015-05-30: UFC FN 67
    {"date": "2015-05-30", "fa": "Rony Jason", "fb": "Damon Jackson",
     "event": "UFC Fight Night: Condit vs Alves",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-67-alves-vs-condit-957"]},
    # 2015-07-18: UFC FN 72 (alias: Joanne Wood = Joanne Calderwood in jansen88)
    {"date": "2015-07-18", "fa": "Joanne Wood", "fb": "Cortney Casey",
     "event": "UFC Fight Night: Bisping vs Leites",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-72-bisping-vs-leites-975"],
     "alias_note": "jansen88 lists 'Joanne Wood' but BFO/actual fighter is 'Joanne Calderwood'. Data exists under Calderwood."},
    # 2015-09-26: UFC FN 75
    {"date": "2015-09-26", "fa": "Mizuto Hirota", "fb": "Teruto Ishihara",
     "event": "UFC Fight Night: Barnett vs Nelson",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-75-nelson-vs-barnett-1003"]},
    # 2016-01-02: UFC 195 (alias: Nina Nunes = Nina Ansaroff)
    {"date": "2016-01-02", "fa": "Justine Kish", "fb": "Nina Nunes",
     "event": "UFC 195: Lawler vs Condit",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-195-lawler-vs-condit-1012"],
     "alias_note": "Nina Nunes listed as Nina Ansaroff on BFO (married name change). Data exists under Ansaroff."},
    # 2016-01-30: UFC on FOX 18
    {"date": "2016-01-30", "fa": "Damon Jackson", "fb": "Levan Makashvili",
     "event": "UFC on FOX: Johnson vs. Bader",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-on-fox-18-johnson-vs-bader-1038"]},
    # 2016-09-17: UFC FN: Poirier vs Johnson
    {"date": "2016-09-17", "fa": "Alejandro Perez", "fb": "Albert Morales",
     "event": "UFC Fight Night: Poirier vs. Johnson",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-94-poirier-vs-johnson-1160",
                   "https://www.bestfightodds.com/events/ufc-fight-night-94-johnson-vs-poirier-1160"]},
    # 2016-11-19: UFC FN 100
    {"date": "2016-11-19", "fa": "Cezar Ferreira", "fb": "Jack Hermansson",
     "event": "UFC Fight Night: Bader vs. Nogueira",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-100-bader-vs-nogueira-1185"]},
    {"date": "2016-11-19", "fa": "Johnny Eduardo", "fb": "Manvel Gamburyan",
     "event": "UFC Fight Night: Bader vs. Nogueira",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-100-bader-vs-nogueira-1185"]},
    {"date": "2016-11-19", "fa": "Francimar Barroso", "fb": "Darren Stewart",
     "event": "UFC Fight Night: Bader vs. Nogueira",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-100-bader-vs-nogueira-1185"]},
    # 2017-06-17: UFC FN 111
    {"date": "2017-06-17", "fa": "Li Jingliang", "fb": "Frank Camacho",
     "event": "UFC Fight Night: Holm vs. Correia",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-111-holm-vs-correia-1277"]},
    # 2017-07-07: TUF 25 Finale
    {"date": "2017-07-07", "fa": "Drakkar Klose", "fb": "Marc Diakiese",
     "event": "The Ultimate Fighter: Redemption Finale",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-the-ultimate-fighter-25-finale-johnson-vs-gaethje-1289"]},
    # 2022-02-05: UFC FN Hermansson vs Strickland
    {"date": "2022-02-05", "fa": "Jailton Almeida", "fb": "Danilo Marques",
     "event": "UFC Fight Night: Hermansson vs. Strickland",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-hermansson-vs-strickland-2285"]},
    # 2022-11-12: UFC 281
    {"date": "2022-11-12", "fa": "Zhang Weili", "fb": "Carla Esparza",
     "event": "UFC 281: Adesanya vs. Pereira",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-281-adesanya-vs-pereira-2407"]},
    {"date": "2022-11-12", "fa": "Dustin Poirier", "fb": "Michael Chandler",
     "event": "UFC 281: Adesanya vs. Pereira",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-281-adesanya-vs-pereira-2407"]},
    # 2022-12-10: UFC 282
    {"date": "2022-12-10", "fa": "Ilia Topuria", "fb": "Bryce Mitchell",
     "event": "UFC 282: Blachowicz vs. Ankalaev",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-282-blachowicz-vs-ankalaev-2436"]},
    # 2023-01-14: UFC FN Strickland vs Imavov
    {"date": "2023-01-14", "fa": "Mateusz Rebecki", "fb": "Nick Fiore",
     "event": "UFC Fight Night: Strickland vs. Imavov",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-strickland-vs-imavov-2457"]},
    # 2023-02-04: UFC FN Lewis vs Spivac
    {"date": "2023-02-04", "fa": "Dooho Choi", "fb": "Kyle Nelson",
     "event": "UFC Fight Night: Lewis vs. Spivac",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-lewis-vs-spivac-2472"]},
    {"date": "2023-02-04", "fa": "JeongYeong Lee", "fb": "Yizha",
     "event": "UFC Fight Night: Lewis vs. Spivac",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-lewis-vs-spivac-2472"]},
    # 2023-03-11: UFC FN Yan vs Dvalishvili
    {"date": "2023-03-11", "fa": "Ariane Lipski", "fb": "JJ Aldrich",
     "event": "UFC Fight Night: Yan vs. Dvalishvili",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-yan-vs-dvalishvili-2490"]},
    # 2023-04-08: UFC 287
    {"date": "2023-04-08", "fa": "Israel Adesanya", "fb": "Alex Pereira",
     "event": "UFC 287: Pereira vs. Adesanya 2",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-287-pereira-vs-adesanya-2-2497"]},
    # 2023-06-17: UFC FN Vettori vs Cannonier
    {"date": "2023-06-17", "fa": "Dan Argueta", "fb": "Ronnie Lawrence",
     "event": "UFC Fight Night: Vettori vs. Cannonier",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-vettori-vs-cannonier-2545"]},
    # 2023-06-24: UFC FN Emmett vs Topuria
    {"date": "2023-06-24", "fa": "Austen Lane", "fb": "Justin Tafa",
     "event": "UFC Fight Night: Emmett vs. Topuria",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-emmett-vs-topuria-2546"]},
    # 2023-07-08: UFC 290 (9 fights)
    {"date": "2023-07-08", "fa": "Alexander Volkanovski", "fb": "Yair Rodriguez",
     "event": "UFC 290: Volkanovski vs. Rodriguez",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-290-volkanovski-vs-rodriguez-2553"]},
    {"date": "2023-07-08", "fa": "Alexandre Pantoja", "fb": "Brandon Moreno",
     "event": "UFC 290: Volkanovski vs. Rodriguez",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-290-volkanovski-vs-rodriguez-2553"]},
    {"date": "2023-07-08", "fa": "Dricus Du Plessis", "fb": "Robert Whittaker",
     "event": "UFC 290: Volkanovski vs. Rodriguez",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-290-volkanovski-vs-rodriguez-2553"]},
    {"date": "2023-07-08", "fa": "Dan Hooker", "fb": "Jalin Turner",
     "event": "UFC 290: Volkanovski vs. Rodriguez",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-290-volkanovski-vs-rodriguez-2553"]},
    {"date": "2023-07-08", "fa": "Bo Nickal", "fb": "Val Woodburn",
     "event": "UFC 290: Volkanovski vs. Rodriguez",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-290-volkanovski-vs-rodriguez-2553"]},
    {"date": "2023-07-08", "fa": "Robbie Lawler", "fb": "Niko Price",
     "event": "UFC 290: Volkanovski vs. Rodriguez",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-290-volkanovski-vs-rodriguez-2553"]},
    {"date": "2023-07-08", "fa": "Tatsuro Taira", "fb": "Edgar Chairez",
     "event": "UFC 290: Volkanovski vs. Rodriguez",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-290-volkanovski-vs-rodriguez-2553"]},
    {"date": "2023-07-08", "fa": "Denise Gomes", "fb": "Yazmin Jauregui",
     "event": "UFC 290: Volkanovski vs. Rodriguez",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-290-volkanovski-vs-rodriguez-2553"]},
    {"date": "2023-07-08", "fa": "Alonzo Menifield", "fb": "Jimmy Crute",
     "event": "UFC 290: Volkanovski vs. Rodriguez",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-290-volkanovski-vs-rodriguez-2553"]},
    # 2023-08-05: UFC FN Sandhagen vs Font
    {"date": "2023-08-05", "fa": "Billy Quarantillo", "fb": "Damon Jackson",
     "event": "UFC Fight Night: Sandhagen vs. Font",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-sandhagen-vs-font-2561"]},
    {"date": "2023-08-05", "fa": "Sean Woodson", "fb": "Dennis Buzukja",
     "event": "UFC Fight Night: Sandhagen vs. Font",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-sandhagen-vs-font-2561"]},
    {"date": "2023-08-05", "fa": "Assu Almabayev", "fb": "Ode Osbourne",
     "event": "UFC Fight Night: Sandhagen vs. Font",
     "bfo_urls": ["https://www.bestfightodds.com/events/ufc-fight-night-sandhagen-vs-font-2561"]},
    # 2023-09-16: Noche UFC
    {"date": "2023-09-16", "fa": "Alexa Grasso", "fb": "Valentina Shevchenko",
     "event": "UFC Fight Night: Grasso vs. Shevchenko 2",
     "bfo_urls": ["https://www.bestfightodds.com/events/noche-ufc-grasso-vs-shevchenko-2-2580",
                   "https://www.bestfightodds.com/events/ufc-fight-night-grasso-vs-shevchenko-2-2580"]},
    {"date": "2023-09-16", "fa": "Edgar Chairez", "fb": "Daniel Lacerda",
     "event": "UFC Fight Night: Grasso vs. Shevchenko 2",
     "bfo_urls": ["https://www.bestfightodds.com/events/noche-ufc-grasso-vs-shevchenko-2-2580",
                   "https://www.bestfightodds.com/events/ufc-fight-night-grasso-vs-shevchenko-2-2580"]},
]

# ── Name aliases for fuzzy matching ──────────────────────────────────────
ALIASES = {
    "joe ellenberger": ["jake ellenberger", "joe ellenberger"],
    "aleksei oleinik": ["alexey oleynik", "aleksei oleinik"],
    "kj noons": ["k.j. noons", "kj noons"],
    "seo hee ham": ["seohee ham", "seo hee ham"],
    "norifumi yamamoto": ["kid yamamoto", "norifumi yamamoto"],
    "rony jason": ["rony jason ferreira", "rony jason"],
    "li jingliang": ["jingliang li", "li jingliang"],
    "jailton almeida": ["jailton almeida junior", "jailton almeida"],
    "zhang weili": ["weili zhang", "zhang weili"],
    "ilia topuria": ["ilia topuria"],
    "dooho choi": ["do ho choi", "doo ho choi", "dooho choi"],
    "jeongyeong lee": ["jeong yeong lee", "jeongyeong lee"],
    "yizha": ["yizha", "yi zha"],
    "ariane lipski": ["ariane lipski"],
    "jj aldrich": ["j.j. aldrich", "jessica-rose clark", "jj aldrich"],
    "alex pereira": ["alex pereira"],
    "dan argueta": ["daniel argueta", "dan argueta"],
    "dricus du plessis": ["dricus du plessis"],
    "bo nickal": ["bo nickal"],
    "val woodburn": ["val woodburn"],
    "denise gomes": ["denise gomes"],
    "yazmin jauregui": ["yazmin jauregui"],
    "assu almabayev": ["assu almabayev"],
    "alexa grasso": ["alexa grasso"],
    "edgar chairez": ["edgar chairez"],
    "daniel lacerda": ["daniel lacerda"],
}


def norm(n):
    if not n:
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


def extract_props_playwright(page):
    """Extract method props from a rendered BFO page via Playwright."""
    rows = page.query_selector_all("tr")
    all_props = {}
    for row in rows:
        cells = row.query_selector_all("td, th")
        if not cells:
            continue
        label = cells[0].inner_text().strip()
        label_lower = label.lower()
        method = None
        if re.search(r"wins by (tko/?ko|ko/?tko|knockout)", label_lower) and "round" not in label_lower and "via" not in label_lower:
            method = "ko"
        elif re.search(r"wins by sub", label_lower) and "round" not in label_lower:
            method = "sub"
        elif re.search(r"wins by (decision|dec)", label_lower) and not any(
            w in label_lower for w in ["unanimous", "split", "majority"]
        ):
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
    """Match fighter name to extracted props, using aliases."""
    fn = norm(fighter_name)
    ln = last_name(fighter_name)

    # Direct match
    if fn in all_props:
        return all_props[fn]

    # Last name match
    for bfo_name, data in all_props.items():
        bfo_last = bfo_name.split()[-1] if bfo_name.split() else ""
        if bfo_last == ln:
            return data

    # Alias match
    aliases = ALIASES.get(fn, [])
    for alias in aliases:
        alias_n = norm(alias)
        alias_ln = last_name(alias)
        if alias_n in all_props:
            return all_props[alias_n]
        for bfo_name, data in all_props.items():
            bfo_last = bfo_name.split()[-1] if bfo_name.split() else ""
            if bfo_last == alias_ln:
                return data

    # Partial match - first name + last name
    for bfo_name, data in all_props.items():
        if ln in bfo_name:
            return data

    return None


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"processed": [], "recovered": [], "unresolved": []}


def save_checkpoint(ckpt):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f, indent=2, ensure_ascii=False)


def main():
    ckpt = load_checkpoint()
    processed_keys = set(ckpt["processed"])
    recovered = ckpt["recovered"]
    unresolved = ckpt["unresolved"]

    # Group fights by BFO URL to avoid refetching
    url_groups = {}
    for fight in FIGHTS:
        if fight.get("alias_note"):
            # Already resolved by alias - record in unresolved with note
            key = f"{fight['date']}|{fight['fa']}|{fight['fb']}"
            if key not in processed_keys:
                unresolved.append({
                    "fight": f"{fight['date']}: {fight['fa']} vs {fight['fb']}",
                    "attempted_sources": "BFO existing data",
                    "urls_checked": "",
                    "why_unresolved": f"ALIAS_RESOLVED: {fight['alias_note']}",
                })
                ckpt["processed"].append(key)
                processed_keys.add(key)
                print(f"ALIAS: {fight['date']}: {fight['fa']} vs {fight['fb']} -> {fight['alias_note']}")
            continue

        primary_url = fight["bfo_urls"][0]
        if primary_url not in url_groups:
            url_groups[primary_url] = {
                "urls": fight["bfo_urls"],
                "fights": [],
            }
        url_groups[primary_url]["fights"].append(fight)

    print(f"Unique BFO event pages to scrape: {len(url_groups)}")
    print(f"Already processed: {len(processed_keys)}")
    print(f"Already recovered: {len(recovered)}")

    from playwright.sync_api import sync_playwright

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for url_idx, (primary_url, group) in enumerate(url_groups.items()):
            fights_in_group = group["fights"]
            all_urls = group["urls"]

            # Skip if all fights in this group already processed
            remaining = [f for f in fights_in_group
                        if f"{f['date']}|{f['fa']}|{f['fb']}" not in processed_keys]
            if not remaining:
                continue

            print(f"\n[{url_idx+1}/{len(url_groups)}] {primary_url.split('/')[-1]}")
            print(f"  {len(remaining)} fights to find")

            all_props = {}

            # Try each URL variant
            for url in all_urls:
                try:
                    page.goto(url, timeout=30000)
                    page.wait_for_timeout(5000)  # Wait for JS to render
                    props = extract_props_playwright(page)
                    if len(props) > len(all_props):
                        all_props = props
                        source_url = url
                    if len(props) >= 4:
                        break
                except Exception as e:
                    print(f"  Error loading {url}: {e}")
                time.sleep(DELAY)

            if all_props:
                print(f"  Found props for {len(all_props)} fighters: {list(all_props.keys())[:10]}")
            else:
                print(f"  NO method props found on page")

            # Try Wayback if no props found
            if not all_props:
                import requests
                for url in all_urls[:1]:
                    print(f"  Trying Wayback for {url.split('/')[-1]}...")
                    try:
                        cdx = requests.get(
                            "https://web.archive.org/cdx/search/cdx",
                            params={"url": url, "output": "json", "limit": 5,
                                    "fl": "timestamp,statuscode", "filter": "statuscode:200"},
                            timeout=20,
                        )
                        if cdx.status_code == 200:
                            data = cdx.json()
                            if len(data) > 1:
                                for row in data[1:]:
                                    ts = row[0]
                                    time.sleep(DELAY)
                                    try:
                                        from bs4 import BeautifulSoup
                                        wb = requests.get(
                                            f"https://web.archive.org/web/{ts}id_/{url}",
                                            timeout=45,
                                        )
                                        if wb.status_code == 200:
                                            soup = BeautifulSoup(wb.text, "lxml")
                                            rows = soup.select("tr")
                                            for tr in rows:
                                                cells = tr.select("td, th")
                                                if not cells:
                                                    continue
                                                lbl = cells[0].get_text(strip=True).lower()
                                                method = None
                                                if re.search(r"wins by (tko/?ko|ko/?tko|knockout)", lbl) and "round" not in lbl:
                                                    method = "ko"
                                                elif re.search(r"wins by sub", lbl) and "round" not in lbl:
                                                    method = "sub"
                                                elif re.search(r"wins by (decision|dec)", lbl) and not any(w in lbl for w in ["unanimous", "split", "majority"]):
                                                    method = "dec"
                                                if method:
                                                    fighter = re.sub(r"\s*wins\s+by\s+.*$", "", cells[0].get_text(strip=True), flags=re.IGNORECASE).strip()
                                                    odds_vals = [parse_american(c.get_text(strip=True)) for c in cells[1:]]
                                                    odds_vals = [v for v in odds_vals if v is not None]
                                                    if odds_vals:
                                                        fn = norm(fighter)
                                                        if fn not in all_props:
                                                            all_props[fn] = {"raw": fighter}
                                                        all_props[fn][method] = american_to_prob(np.mean(odds_vals))
                                            if len(all_props) >= 4:
                                                source_url = f"https://web.archive.org/web/{ts}/{url}"
                                                print(f"  Wayback: found {len(all_props)} fighters")
                                                break
                                    except Exception:
                                        continue
                    except Exception as e:
                        print(f"  Wayback error: {e}")

            # Match each fight
            for fight in remaining:
                key = f"{fight['date']}|{fight['fa']}|{fight['fb']}"
                fa_data = fuzzy_match(fight["fa"], all_props)
                fb_data = fuzzy_match(fight["fb"], all_props)

                a_ko = fa_data.get("ko") if fa_data else None
                a_sub = fa_data.get("sub") if fa_data else None
                a_dec = fa_data.get("dec") if fa_data else None
                b_ko = fb_data.get("ko") if fb_data else None
                b_sub = fb_data.get("sub") if fb_data else None
                b_dec = fb_data.get("dec") if fb_data else None

                found_any = any(v is not None for v in [a_ko, a_sub, a_dec, b_ko, b_sub, b_dec])

                if found_any:
                    rec = {
                        "event_date": fight["date"],
                        "fighter_a": fight["fa"],
                        "fighter_b": fight["fb"],
                        "event_title": fight["event"],
                        "source": "bfo_manual_recovery",
                        "source_url": source_url if all_props else "",
                        "captured_at": now,
                        "a_ko_odds_prob": round(a_ko, 6) if a_ko is not None else "",
                        "a_sub_odds_prob": round(a_sub, 6) if a_sub is not None else "",
                        "a_dec_odds_prob": round(a_dec, 6) if a_dec is not None else "",
                        "b_ko_odds_prob": round(b_ko, 6) if b_ko is not None else "",
                        "b_sub_odds_prob": round(b_sub, 6) if b_sub is not None else "",
                        "b_dec_odds_prob": round(b_dec, 6) if b_dec is not None else "",
                    }
                    recovered.append(rec)
                    print(f"    FOUND: {fight['fa']} vs {fight['fb']} "
                          f"[a_ko={a_ko and round(a_ko,3)}, a_sub={a_sub and round(a_sub,3)}, a_dec={a_dec and round(a_dec,3)}, "
                          f"b_ko={b_ko and round(b_ko,3)}, b_sub={b_sub and round(b_sub,3)}, b_dec={b_dec and round(b_dec,3)}]")
                else:
                    unresolved.append({
                        "fight": f"{fight['date']}: {fight['fa']} vs {fight['fb']}",
                        "attempted_sources": "BFO Playwright, Wayback Machine",
                        "urls_checked": "; ".join(all_urls),
                        "why_unresolved": "No method prop odds found on BFO page (props not posted for this fight)",
                    })
                    print(f"    MISS: {fight['fa']} vs {fight['fb']}")

                ckpt["processed"].append(key)
                processed_keys.add(key)

            ckpt["recovered"] = recovered
            ckpt["unresolved"] = unresolved
            save_checkpoint(ckpt)

        browser.close()

    # Write output files
    if recovered:
        with open(OUTPUT_RECOVERY, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS_ROW)
            writer.writeheader()
            for rec in recovered:
                writer.writerow(rec)
        print(f"\nRecovery file: {OUTPUT_RECOVERY} ({len(recovered)} rows)")

    with open(OUTPUT_UNRESOLVED, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fight", "attempted_sources", "urls_checked", "why_unresolved"])
        writer.writeheader()
        for rec in unresolved:
            writer.writerow(rec)
    print(f"Unresolved file: {OUTPUT_UNRESOLVED} ({len(unresolved)} rows)")

    print(f"\n=== SUMMARY ===")
    print(f"Total recovered: {len(recovered)}")
    print(f"Total unresolved: {len(unresolved)}")
    alias_resolved = sum(1 for u in unresolved if "ALIAS_RESOLVED" in u.get("why_unresolved", ""))
    print(f"  - Alias-resolved (data exists under different name): {alias_resolved}")
    print(f"  - Truly unresolved (no method props on BFO): {len(unresolved) - alias_resolved}")


if __name__ == "__main__":
    main()
