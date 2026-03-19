"""
Wayback-only recovery pass for the remaining missing method odds fights.
BFO live pages redirect historical events to current content, so we must use Wayback Machine.
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
import requests
from bs4 import BeautifulSoup

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
REQ_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DELAY = 4.0

# Events to scrape via Wayback, grouped by BFO URL
EVENTS = [
    # 2014 events that had no method props on earlier pass
    {"bfo_url": "https://www.bestfightodds.com/events/ufc-fight-night-44-swanson-vs-stephens-800",
     "event": "UFC Fight Night: Swanson vs Stephens",
     "fights": [
         ("2014-06-28", "Joe Ellenberger", "James Moontasri"),
         ("2014-06-28", "Cody Gibson", "Johnny Bedford"),
         ("2014-06-28", "Marcelo Guimaraes", "Andy Enz"),
         ("2014-06-28", "Ray Borg", "Shane Howell"),
         ("2014-06-28", "Aleksei Oleinik", "Anthony Hamilton"),
     ]},
    # 2015 events
    {"bfo_url": "https://www.bestfightodds.com/events/ufc-fight-night-67-alves-vs-condit-957",
     "event": "UFC Fight Night: Condit vs Alves",
     "fights": [("2015-05-30", "Rony Jason", "Damon Jackson")]},
    {"bfo_url": "https://www.bestfightodds.com/events/ufc-fight-night-75-nelson-vs-barnett-1003",
     "event": "UFC Fight Night: Barnett vs Nelson",
     "fights": [("2015-09-26", "Mizuto Hirota", "Teruto Ishihara")]},
    # 2016 events
    {"bfo_url": "https://www.bestfightodds.com/events/ufc-fight-night-94-poirier-vs-johnson-1160",
     "event": "UFC Fight Night: Poirier vs. Johnson",
     "fights": [("2016-09-17", "Alejandro Perez", "Albert Morales")]},
    {"bfo_url": "https://www.bestfightodds.com/events/ufc-fight-night-100-bader-vs-nogueira-1185",
     "event": "UFC Fight Night: Bader vs. Nogueira",
     "fights": [
         ("2016-11-19", "Cezar Ferreira", "Jack Hermansson"),
         ("2016-11-19", "Johnny Eduardo", "Manvel Gamburyan"),
         ("2016-11-19", "Francimar Barroso", "Darren Stewart"),
     ]},
    # 2017 events
    {"bfo_url": "https://www.bestfightodds.com/events/ufc-fight-night-111-holm-vs-correia-1277",
     "event": "UFC Fight Night: Holm vs. Correia",
     "fights": [("2017-06-17", "Li Jingliang", "Frank Camacho")]},
    # 2022 events
    {"bfo_url": "https://www.bestfightodds.com/events/ufc-fight-night-hermansson-vs-strickland-2285",
     "event": "UFC Fight Night: Hermansson vs. Strickland",
     "fights": [("2022-02-05", "Jailton Almeida", "Danilo Marques")]},
    {"bfo_url": "https://www.bestfightodds.com/events/ufc-281-adesanya-vs-pereira-2407",
     "event": "UFC 281: Adesanya vs. Pereira",
     "fights": [
         ("2022-11-12", "Zhang Weili", "Carla Esparza"),
         ("2022-11-12", "Dustin Poirier", "Michael Chandler"),
     ]},
    {"bfo_url": "https://www.bestfightodds.com/events/ufc-282-blachowicz-vs-ankalaev-2436",
     "event": "UFC 282: Blachowicz vs. Ankalaev",
     "fights": [("2022-12-10", "Ilia Topuria", "Bryce Mitchell")]},
    # 2023 events
    {"bfo_url": "https://www.bestfightodds.com/events/ufc-fight-night-strickland-vs-imavov-2457",
     "event": "UFC Fight Night: Strickland vs. Imavov",
     "fights": [("2023-01-14", "Mateusz Rebecki", "Nick Fiore")]},
    {"bfo_url": "https://www.bestfightodds.com/events/ufc-fight-night-lewis-vs-spivac-2472",
     "event": "UFC Fight Night: Lewis vs. Spivac",
     "fights": [
         ("2023-02-04", "Dooho Choi", "Kyle Nelson"),
         ("2023-02-04", "JeongYeong Lee", "Yizha"),
     ]},
    {"bfo_url": "https://www.bestfightodds.com/events/ufc-fight-night-yan-vs-dvalishvili-2490",
     "event": "UFC Fight Night: Yan vs. Dvalishvili",
     "fights": [("2023-03-11", "Ariane Lipski", "JJ Aldrich")]},
    {"bfo_url": "https://www.bestfightodds.com/events/ufc-287-pereira-vs-adesanya-2-2497",
     "event": "UFC 287: Pereira vs. Adesanya 2",
     "fights": [("2023-04-08", "Israel Adesanya", "Alex Pereira")]},
    {"bfo_url": "https://www.bestfightodds.com/events/ufc-fight-night-vettori-vs-cannonier-2545",
     "event": "UFC Fight Night: Vettori vs. Cannonier",
     "fights": [("2023-06-17", "Dan Argueta", "Ronnie Lawrence")]},
    {"bfo_url": "https://www.bestfightodds.com/events/ufc-fight-night-emmett-vs-topuria-2546",
     "event": "UFC Fight Night: Emmett vs. Topuria",
     "fights": [("2023-06-24", "Austen Lane", "Justin Tafa")]},
    {"bfo_url": "https://www.bestfightodds.com/events/ufc-290-volkanovski-vs-rodriguez-2553",
     "event": "UFC 290: Volkanovski vs. Rodriguez",
     "fights": [
         ("2023-07-08", "Alexander Volkanovski", "Yair Rodriguez"),
         ("2023-07-08", "Alexandre Pantoja", "Brandon Moreno"),
         ("2023-07-08", "Dricus Du Plessis", "Robert Whittaker"),
         ("2023-07-08", "Dan Hooker", "Jalin Turner"),
         ("2023-07-08", "Bo Nickal", "Val Woodburn"),
         ("2023-07-08", "Robbie Lawler", "Niko Price"),
         ("2023-07-08", "Tatsuro Taira", "Edgar Chairez"),
         ("2023-07-08", "Denise Gomes", "Yazmin Jauregui"),
         ("2023-07-08", "Alonzo Menifield", "Jimmy Crute"),
     ]},
    {"bfo_url": "https://www.bestfightodds.com/events/ufc-fight-night-sandhagen-vs-font-2561",
     "event": "UFC Fight Night: Sandhagen vs. Font",
     "fights": [
         ("2023-08-05", "Billy Quarantillo", "Damon Jackson"),
         ("2023-08-05", "Sean Woodson", "Dennis Buzukja"),
         ("2023-08-05", "Assu Almabayev", "Ode Osbourne"),
     ]},
    {"bfo_url": "https://www.bestfightodds.com/events/noche-ufc-grasso-vs-shevchenko-2-2580",
     "event": "UFC Fight Night: Grasso vs. Shevchenko 2",
     "fights": [
         ("2023-09-16", "Alexa Grasso", "Valentina Shevchenko"),
         ("2023-09-16", "Edgar Chairez", "Daniel Lacerda"),
     ]},
]

ALIASES = {
    "joe ellenberger": ["joe ellenberger"],
    "aleksei oleinik": ["alexey oleynik", "aleksei oleinik"],
    "kj noons": ["k.j. noons", "kj noons"],
    "seo hee ham": ["seohee ham", "seo hee ham"],
    "norifumi yamamoto": ["kid yamamoto", "norifumi yamamoto"],
    "rony jason": ["rony jason ferreira", "rony jason"],
    "li jingliang": ["jingliang li", "li jingliang", "jingliang"],
    "jailton almeida": ["jailton almeida junior", "jailton almeida"],
    "zhang weili": ["weili zhang", "zhang weili", "weili"],
    "dooho choi": ["do ho choi", "doo ho choi", "dooho choi"],
    "jeongyeong lee": ["jeong yeong lee", "jeongyeong lee"],
    "yizha": ["yizha", "yi zha"],
    "jj aldrich": ["j.j. aldrich", "jj aldrich"],
    "alex pereira": ["alex pereira"],
    "dan argueta": ["daniel argueta", "dan argueta"],
    "dricus du plessis": ["dricus du plessis"],
    "assu almabayev": ["assu almabayev"],
    "alexa grasso": ["alexa grasso"],
    "edgar chairez": ["edgar chairez"],
    "daniel lacerda": ["daniel lacerda"],
    "cezar ferreira": ["cezar mutante", "cezar ferreira"],
    "francimar barroso": ["francimar barroso"],
    "johnny eduardo": ["johnny eduardo"],
    "manvel gamburyan": ["manvel gamburyan", "manny gamburyan"],
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


def extract_props_from_html(html):
    soup = BeautifulSoup(html, "lxml")
    rows = soup.select("tr")
    all_props = {}
    for row in rows:
        cells = row.select("td, th")
        if not cells:
            continue
        label = cells[0].get_text(strip=True)
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
            odds_vals = [parse_american(c.get_text(strip=True)) for c in cells[1:]]
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

    if fn in all_props:
        return all_props[fn]

    for bfo_name, data in all_props.items():
        bfo_last = bfo_name.split()[-1] if bfo_name.split() else ""
        if bfo_last == ln and ln:
            return data

    aliases = ALIASES.get(fn, [])
    for alias in aliases:
        alias_n = norm(alias)
        alias_ln = last_name(alias)
        if alias_n in all_props:
            return all_props[alias_n]
        for bfo_name, data in all_props.items():
            bfo_last = bfo_name.split()[-1] if bfo_name.split() else ""
            if bfo_last == alias_ln and alias_ln:
                return data

    for bfo_name, data in all_props.items():
        if ln and ln in bfo_name:
            return data

    return None


def try_wayback(url, max_snapshots=10):
    """Get method props from Wayback Machine snapshots."""
    try:
        cdx = requests.get(
            "https://web.archive.org/cdx/search/cdx",
            params={
                "url": url, "output": "json", "limit": max_snapshots,
                "fl": "timestamp,statuscode", "filter": "statuscode:200",
            },
            headers=REQ_HEADERS, timeout=30,
        )
        if cdx.status_code != 200:
            print(f"    CDX API returned {cdx.status_code}")
            return {}, ""
        data = cdx.json()
        if len(data) <= 1:
            print(f"    No Wayback snapshots found")
            return {}, ""

        print(f"    Found {len(data)-1} snapshots")
        best_props = {}
        best_url = ""

        for row in data[1:]:
            ts = row[0]
            wb_url = f"https://web.archive.org/web/{ts}id_/{url}"
            time.sleep(DELAY)
            try:
                resp = requests.get(wb_url, headers=REQ_HEADERS, timeout=60)
                if resp.status_code == 200:
                    props = extract_props_from_html(resp.text)
                    if len(props) > len(best_props):
                        best_props = props
                        best_url = f"https://web.archive.org/web/{ts}/{url}"
                        print(f"    Snapshot {ts}: {len(props)} fighters")
                    if len(props) >= 8:
                        break
                else:
                    print(f"    Snapshot {ts}: HTTP {resp.status_code}")
            except Exception as e:
                print(f"    Snapshot {ts}: error - {e}")
                continue

        return best_props, best_url
    except Exception as e:
        print(f"    Wayback error: {e}")
        return {}, ""


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

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for evt_idx, evt in enumerate(EVENTS):
        bfo_url = evt["bfo_url"]
        event_name = evt["event"]
        fights = evt["fights"]

        remaining = [(d, fa, fb) for d, fa, fb in fights if f"{d}|{fa}|{fb}" not in processed_keys]
        if not remaining:
            continue

        print(f"\n[{evt_idx+1}/{len(EVENTS)}] {bfo_url.split('/')[-1]}")
        print(f"  {len(remaining)} fights to find")

        all_props, source_url = try_wayback(bfo_url)

        if all_props:
            print(f"  Best snapshot: {len(all_props)} fighters: {list(all_props.keys())[:12]}")
        else:
            print(f"  NO method props found in Wayback")

        for date, fa, fb in remaining:
            key = f"{date}|{fa}|{fb}"
            fa_data = fuzzy_match(fa, all_props) if all_props else None
            fb_data = fuzzy_match(fb, all_props) if all_props else None

            a_ko = fa_data.get("ko") if fa_data else None
            a_sub = fa_data.get("sub") if fa_data else None
            a_dec = fa_data.get("dec") if fa_data else None
            b_ko = fb_data.get("ko") if fb_data else None
            b_sub = fb_data.get("sub") if fb_data else None
            b_dec = fb_data.get("dec") if fb_data else None

            found_any = any(v is not None for v in [a_ko, a_sub, a_dec, b_ko, b_sub, b_dec])

            if found_any:
                rec = {
                    "event_date": date,
                    "fighter_a": fa,
                    "fighter_b": fb,
                    "event_title": event_name,
                    "source": "wayback_bfo_manual_recovery",
                    "source_url": source_url,
                    "captured_at": now,
                    "a_ko_odds_prob": round(a_ko, 6) if a_ko is not None else "",
                    "a_sub_odds_prob": round(a_sub, 6) if a_sub is not None else "",
                    "a_dec_odds_prob": round(a_dec, 6) if a_dec is not None else "",
                    "b_ko_odds_prob": round(b_ko, 6) if b_ko is not None else "",
                    "b_sub_odds_prob": round(b_sub, 6) if b_sub is not None else "",
                    "b_dec_odds_prob": round(b_dec, 6) if b_dec is not None else "",
                }
                recovered.append(rec)
                print(f"    FOUND: {fa} vs {fb} "
                      f"[a_ko={a_ko and round(a_ko,3)}, a_sub={a_sub and round(a_sub,3)}, a_dec={a_dec and round(a_dec,3)}, "
                      f"b_ko={b_ko and round(b_ko,3)}, b_sub={b_sub and round(b_sub,3)}, b_dec={b_dec and round(b_dec,3)}]")
            else:
                urls_checked = bfo_url
                unresolved.append({
                    "fight": f"{date}: {fa} vs {fb}",
                    "attempted_sources": "Wayback Machine (BFO event page)",
                    "urls_checked": urls_checked,
                    "why_unresolved": "No method prop odds found in any Wayback snapshot" if not all_props else f"Props found for {len(all_props)} fighters but no match for {last_name(fa)}/{last_name(fb)}",
                })
                print(f"    MISS: {fa} vs {fb}")

            ckpt["processed"].append(key)
            processed_keys.add(key)

        ckpt["recovered"] = recovered
        ckpt["unresolved"] = unresolved
        save_checkpoint(ckpt)

    # Write final output files
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
    print(f"  - Truly unresolved: {len(unresolved) - alias_resolved}")


if __name__ == "__main__":
    main()
