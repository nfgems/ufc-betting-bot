"""Scrape BFO event pages from Wayback Machine for missing moneyline + method odds."""

from __future__ import annotations

import csv
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
ML_OUTPUT = REPO_ROOT / "data" / "raw" / "historical_odds" / "historical_odds_pre2022_from_cleaned.csv"
ML_MISSING = REPO_ROOT / "data" / "raw" / "historical_odds" / "pre2022_odds_still_missing.csv"
METHOD_OUTPUT = REPO_ROOT / "data" / "raw" / "method_odds" / "historical_method_odds_all.csv"


def norm(name):
    if pd.isna(name):
        return ""
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"[^a-zA-Z\s]", "", name)
    return " ".join(name.lower().split())


def fuzzy_match(n1, n2):
    a, b = norm(n1), norm(n2)
    if not a or not b:
        return False
    if a == b:
        return True
    ta, tb = a.split(), b.split()
    if ta[-1] == tb[-1]:
        return True
    if len(set(ta) & set(tb)) >= 2:
        return True
    return False


def parse_american(text):
    cleaned = re.sub(r"[\u25b2\u25bc\u2191\u2193\s]", "", text.strip())
    match = re.match(r"^([+-]?\d+)$", cleaned)
    return float(match.group(1)) if match else None


def am2dec(odds):
    if odds > 0:
        return (odds / 100.0) + 1.0
    elif odds < 0:
        return (100.0 / abs(odds)) + 1.0
    return np.nan


def am2imp(odds):
    if odds > 0:
        return 100.0 / (odds + 100.0)
    elif odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return np.nan


def parse_bfo_wayback_page(html):
    """Parse a BFO event page and return list of fight dicts with ML + method odds."""
    soup = BeautifulSoup(html, "lxml")
    tables = soup.select("table.odds-table")
    if len(tables) < 2:
        return []

    table = tables[1]
    rows = table.select("tr")
    fights = []
    i = 0

    while i < len(rows):
        row = rows[i]
        if "pr" in (row.get("class", []) or []):
            i += 1
            continue
        cells = row.select("td, th")
        if not cells:
            i += 1
            continue
        first_text = cells[0].get_text(strip=True)
        if not first_text or len(cells) <= 1:
            i += 1
            continue
        if cells[1].get("data-b"):
            i += 1
            continue

        fighter_a = first_text
        a_odds_cells = cells[1:]

        # Find fighter B
        j = i + 1
        fighter_b_row = None
        while j < len(rows):
            nr = rows[j]
            if "pr" in (nr.get("class", []) or []):
                j += 1
                continue
            nc = nr.select("td, th")
            if not nc:
                j += 1
                continue
            nt = nc[0].get_text(strip=True)
            if not nt:
                j += 1
                continue
            if len(nc) > 1 and nc[1].get("data-b"):
                j += 1
                continue
            fighter_b_row = nr
            break

        if not fighter_b_row:
            i += 1
            continue

        b_cells = fighter_b_row.select("td, th")
        fighter_b = b_cells[0].get_text(strip=True)
        b_odds_cells = b_cells[1:]

        a_odds, b_odds = [], []
        for ac, bc in zip(a_odds_cells, b_odds_cells):
            ao = parse_american(ac.get_text(strip=True))
            bo = parse_american(bc.get_text(strip=True))
            if ao is not None and bo is not None:
                a_odds.append(ao)
                b_odds.append(bo)

        fight = {"fighter_a": fighter_a, "fighter_b": fighter_b}
        if a_odds and b_odds:
            fight["a_dec"] = np.mean([am2dec(o) for o in a_odds])
            fight["b_dec"] = np.mean([am2dec(o) for o in b_odds])
            fight["num_books"] = len(a_odds)

        # Collect prop rows for method odds
        prop_start = j + 1
        k = prop_start
        a_ko, b_ko, a_sub, b_sub, a_dec_m, b_dec_m = [], [], [], [], [], []

        while k < len(rows):
            pr = rows[k]
            if "pr" not in (pr.get("class", []) or []):
                break
            pr_text = pr.get_text(" ", strip=True).lower()

            # Identify method type
            is_ko = bool(re.search(r"\b(ko|tko|knockout)\b", pr_text))
            is_sub = bool(re.search(r"\b(sub|submission)\b", pr_text))
            is_dec = bool(re.search(r"\b(dec|decision|points)\b", pr_text))

            if is_ko or is_sub or is_dec:
                # Which fighter?
                a_norm = norm(fighter_a)
                b_norm = norm(fighter_b)
                a_last = a_norm.split()[-1] if a_norm.split() else ""
                b_last = b_norm.split()[-1] if b_norm.split() else ""

                side = None
                if a_last and re.search(rf"\b{re.escape(a_last)}\b", pr_text):
                    side = "a"
                elif b_last and re.search(rf"\b{re.escape(b_last)}\b", pr_text):
                    side = "b"

                if side:
                    pr_cells = pr.select("td, th")
                    odds_vals = []
                    for pc in pr_cells[1:]:
                        o = parse_american(pc.get_text(strip=True))
                        if o is not None:
                            odds_vals.append(am2imp(o))

                    if odds_vals:
                        avg_prob = float(np.mean(odds_vals))
                        if is_ko:
                            (a_ko if side == "a" else b_ko).append(avg_prob)
                        elif is_sub:
                            (a_sub if side == "a" else b_sub).append(avg_prob)
                        elif is_dec:
                            (a_dec_m if side == "a" else b_dec_m).append(avg_prob)
            k += 1

        if a_ko or b_ko or a_sub or b_sub or a_dec_m or b_dec_m:
            fight["a_ko_prob"] = float(np.mean(a_ko)) if a_ko else np.nan
            fight["b_ko_prob"] = float(np.mean(b_ko)) if b_ko else np.nan
            fight["a_sub_prob"] = float(np.mean(a_sub)) if a_sub else np.nan
            fight["b_sub_prob"] = float(np.mean(b_sub)) if b_sub else np.nan
            fight["a_dec_prob"] = float(np.mean(a_dec_m)) if a_dec_m else np.nan
            fight["b_dec_prob"] = float(np.mean(b_dec_m)) if b_dec_m else np.nan

        fights.append(fight)
        i = k if k > j else j + 1

    return fights


def main():
    # Get BFO event URLs from Wayback CDX
    logger.info("Querying Wayback CDX for BFO event URLs...")
    all_bfo = {}

    for from_d, to_d in [("20100101", "20140101"), ("20140101", "20180101"), ("20180101", "20220101")]:
        for attempt in range(3):
            try:
                resp = requests.get(
                    "https://web.archive.org/cdx/search/cdx",
                    params={
                        "url": "www.bestfightodds.com/events/ufc*",
                        "output": "json",
                        "from": from_d,
                        "to": to_d,
                        "fl": "timestamp,original",
                        "filter": "statuscode:200",
                        "collapse": "urlkey",
                        "limit": 1000,
                    },
                    headers=HEADERS,
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for row in data[1:]:
                        ts, url = row[0], row[1]
                        if url not in all_bfo or ts < all_bfo[url]:
                            all_bfo[url] = ts
                    logger.info("  %s-%s: %d URLs", from_d, to_d, len(data) - 1)
                    break
                time.sleep(5)
            except Exception:
                time.sleep(10)

    logger.info("Total BFO event URLs: %d", len(all_bfo))

    # Load missing fights
    missing = pd.read_csv(ML_MISSING)
    logger.info("Missing moneyline fights: %d", len(missing))

    # Load existing method odds to check for gaps
    existing_method = pd.read_csv(METHOD_OUTPUT)
    method_keys = set()
    for _, r in existing_method.iterrows():
        k = (str(r["event_date"])[:10], norm(r["fighter_a"]), norm(r["fighter_b"]))
        method_keys.add(k)
        method_keys.add((str(r["event_date"])[:10], norm(r["fighter_b"]), norm(r["fighter_a"])))

    # Load training fights without method odds
    cleaned = pd.read_csv("data/processed/fights_cleaned.csv")
    cleaned["event_date"] = pd.to_datetime(cleaned["event_date"]).dt.strftime("%Y-%m-%d")
    pre2022 = cleaned[cleaned["event_date"] < "2022-01-01"]

    ml_recovered = []
    method_recovered = []
    events_tried = 0

    # Filter to UFC events only
    ufc_urls = {url: ts for url, ts in all_bfo.items() if "/events/ufc" in url}
    logger.info("UFC event URLs to try: %d", len(ufc_urls))

    for bfo_url, ts in sorted(ufc_urls.items(), key=lambda x: x[0]):
        events_tried += 1
        wayback_url = f"https://web.archive.org/web/{ts}/{bfo_url}"

        try:
            resp = requests.get(wayback_url, headers=HEADERS, timeout=45)
            if resp.status_code != 200:
                continue
        except Exception as e:
            if "Connection" in str(type(e).__name__):
                logger.warning("Rate limited at %d, waiting 30s...", events_tried)
                time.sleep(30)
                try:
                    resp = requests.get(wayback_url, headers=HEADERS, timeout=45)
                    if resp.status_code != 200:
                        continue
                except Exception:
                    continue
            else:
                continue

        fights = parse_bfo_wayback_page(resp.text)
        if not fights:
            time.sleep(1)
            continue

        # Match moneyline against missing
        for _, mf in missing.iterrows():
            fa, fb, date = mf["fighter_a"], mf["fighter_b"], mf["event_date"]
            for pf in fights:
                if "a_dec" not in pf:
                    continue
                matched = reversed_m = False
                if fuzzy_match(fa, pf["fighter_a"]) and fuzzy_match(fb, pf["fighter_b"]):
                    matched = True
                elif fuzzy_match(fa, pf["fighter_b"]) and fuzzy_match(fb, pf["fighter_a"]):
                    matched = True
                    reversed_m = True
                if matched:
                    ad = pf["b_dec"] if reversed_m else pf["a_dec"]
                    bd = pf["a_dec"] if reversed_m else pf["b_dec"]
                    ai, bi = 1.0 / ad, 1.0 / bd
                    t = ai + bi
                    ml_recovered.append({
                        "event_date": date, "fighter_a": fa, "fighter_b": fb,
                        "query_date": date, "offset_days": 0,
                        "a_fair_prob": round(ai / t, 6), "b_fair_prob": round(bi / t, 6),
                        "a_decimal_odds": round(ad, 6), "b_decimal_odds": round(bd, 6),
                        "num_bookmakers": pf["num_books"],
                    })
                    logger.info("  ML: %s %s vs %s (%d books)", date, fa, fb, pf["num_books"])
                    break

        # Match method odds against training fights missing method
        for _, tf in pre2022.iterrows():
            fa, fb, date = tf["fighter_a"], tf["fighter_b"], tf["event_date"]
            k = (date, norm(fa), norm(fb))
            if k in method_keys:
                continue
            for pf in fights:
                if "a_ko_prob" not in pf:
                    continue
                matched = reversed_m = False
                if fuzzy_match(fa, pf["fighter_a"]) and fuzzy_match(fb, pf["fighter_b"]):
                    matched = True
                elif fuzzy_match(fa, pf["fighter_b"]) and fuzzy_match(fb, pf["fighter_a"]):
                    matched = True
                    reversed_m = True
                if matched:
                    if reversed_m:
                        row = {
                            "a_ko_odds_prob": pf.get("b_ko_prob", np.nan),
                            "a_sub_odds_prob": pf.get("b_sub_prob", np.nan),
                            "a_dec_odds_prob": pf.get("b_dec_prob", np.nan),
                            "b_ko_odds_prob": pf.get("a_ko_prob", np.nan),
                            "b_sub_odds_prob": pf.get("a_sub_prob", np.nan),
                            "b_dec_odds_prob": pf.get("a_dec_prob", np.nan),
                        }
                    else:
                        row = {
                            "a_ko_odds_prob": pf.get("a_ko_prob", np.nan),
                            "a_sub_odds_prob": pf.get("a_sub_prob", np.nan),
                            "a_dec_odds_prob": pf.get("a_dec_prob", np.nan),
                            "b_ko_odds_prob": pf.get("b_ko_prob", np.nan),
                            "b_sub_odds_prob": pf.get("b_sub_prob", np.nan),
                            "b_dec_odds_prob": pf.get("b_dec_prob", np.nan),
                        }
                    row.update({
                        "event_date": date, "fighter_a": fa, "fighter_b": fb,
                        "event_title": "", "source": "bestfightodds_wayback",
                        "source_url": wayback_url, "captured_at": ts,
                    })
                    method_recovered.append(row)
                    method_keys.add(k)
                    break

        time.sleep(2)

    logger.info("Events tried: %d", events_tried)
    logger.info("Moneyline recovered: %d", len(ml_recovered))
    logger.info("Method odds recovered: %d", len(method_recovered))

    # Save moneyline
    if ml_recovered:
        existing_ml = pd.read_csv(ML_OUTPUT)
        combined = pd.concat([existing_ml, pd.DataFrame(ml_recovered)], ignore_index=True)
        combined = combined.drop_duplicates(subset=["event_date", "fighter_a", "fighter_b"], keep="last")
        combined.to_csv(ML_OUTPUT, index=False)
        logger.info("ML updated: %d rows", len(combined))

        # Update missing
        rec_keys = {(r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"])) for r in ml_recovered}
        still = [r for _, r in missing.iterrows()
                 if (r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"])) not in rec_keys]
        pd.DataFrame(still).to_csv(ML_MISSING, index=False)
        logger.info("Still missing: %d", len(still))

    # Save method odds
    if method_recovered:
        combined_m = pd.concat([existing_method, pd.DataFrame(method_recovered)], ignore_index=True)
        combined_m["_key"] = (combined_m["event_date"].astype(str).str[:10] + "|" +
                              combined_m["fighter_a"].apply(norm) + "|" +
                              combined_m["fighter_b"].apply(norm))
        combined_m = combined_m.drop_duplicates(subset="_key", keep="last").drop(columns=["_key"])
        combined_m.to_csv(METHOD_OUTPUT, index=False)
        logger.info("Method odds updated: %d rows", len(combined_m))


if __name__ == "__main__":
    main()
