"""Targeted Wayback BFO scraping for the last ~13 missing moneyline fights."""
import re, sys, time, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parent.parent
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
ML_PATH = REPO_ROOT / "data/raw/historical_odds/historical_odds_pre2022_from_cleaned.csv"
MISS_PATH = REPO_ROOT / "data/raw/historical_odds/pre2022_odds_still_missing.csv"


def norm(n):
    if pd.isna(n):
        return ""
    n = unicodedata.normalize("NFKD", str(n))
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = re.sub(r"[^a-zA-Z\s]", "", n)
    return " ".join(n.lower().split())


def parse_am(t):
    c = re.sub(r"[\u25b2\u25bc\u2191\u2193\s]", "", t.strip())
    m = re.match(r"^([+-]?\d+)$", c)
    return float(m.group(1)) if m else None


def a2d(o):
    if o > 0:
        return (o / 100) + 1
    elif o < 0:
        return (100 / abs(o)) + 1
    return 0


def fetch_wayback(bfo_url):
    """Find and fetch a Wayback snapshot of a BFO page."""
    try:
        cdx = requests.get(
            "https://web.archive.org/cdx/search/cdx",
            params={"url": bfo_url, "output": "json", "limit": 5,
                    "fl": "timestamp,statuscode", "filter": "statuscode:200"},
            headers=HEADERS, timeout=30,
        )
        if cdx.status_code != 200:
            return None
        data = cdx.json()
        if len(data) <= 1:
            return None
        ts = data[1][0]
        wb = requests.get(f"https://web.archive.org/web/{ts}/{bfo_url}", headers=HEADERS, timeout=45)
        return wb.text if wb.status_code == 200 else None
    except Exception:
        return None


def parse_fights_from_bfo(html):
    """Parse fighter names + odds from BFO HTML."""
    soup = BeautifulSoup(html, "lxml")
    tables = soup.select("table.odds-table")
    if len(tables) < 2:
        return []
    rows = tables[1].select("tr")
    fights = []
    i = 0
    while i < len(rows):
        row = rows[i]
        if "pr" in (row.get("class", []) or []):
            i += 1
            continue
        cells = row.select("td, th")
        if not cells or not cells[0].get_text(strip=True) or (len(cells) > 1 and cells[1].get("data-b")):
            i += 1
            continue
        fighter_a = cells[0].get_text(strip=True)
        a_cells = cells[1:]
        j = i + 1
        while j < len(rows):
            nr = rows[j]
            if "pr" in (nr.get("class", []) or []):
                j += 1
                continue
            nc = nr.select("td, th")
            if nc and nc[0].get_text(strip=True) and not (len(nc) > 1 and nc[1].get("data-b")):
                break
            j += 1
        if j >= len(rows):
            break
        b_cells = rows[j].select("td, th")
        fighter_b = b_cells[0].get_text(strip=True)
        ao, bo = [], []
        for ac, bc in zip(a_cells, b_cells[1:]):
            a_v, b_v = parse_am(ac.get_text(strip=True)), parse_am(bc.get_text(strip=True))
            if a_v is not None and b_v is not None:
                ao.append(a_v)
                bo.append(b_v)
        if ao and bo:
            fights.append({"fa": fighter_a, "fb": fighter_b, "ao": ao, "bo": bo})
        i = j + 1
        while i < len(rows) and "pr" in (rows[i].get("class", []) or []):
            i += 1
    return fights


def main():
    missing = pd.read_csv(MISS_PATH)
    print(f"Missing: {len(missing)}")
    recovered = []

    for _, mrow in missing.iterrows():
        date, fa, fb = mrow["event_date"], mrow["fighter_a"], mrow["fighter_b"]
        fa_last = norm(fa).split()[-1] if norm(fa).split() else ""
        fb_last = norm(fb).split()[-1] if norm(fb).split() else ""

        # Build BFO search queries
        queries = [f"{fa} {fb}", fa, fb]
        bfo_urls = set()
        for q in queries:
            try:
                resp = requests.get("https://www.bestfightodds.com/search",
                                    params={"query": q}, headers=HEADERS, timeout=20)
                if resp.status_code != 200:
                    continue
                soup = BeautifulSoup(resp.text, "lxml")
                for link in soup.select("a[href*='/events/']"):
                    href = link.get("href", "")
                    if "/events/" in href and href not in ("/events", "/events/"):
                        full = href if href.startswith("http") else f"https://www.bestfightodds.com{href}"
                        bfo_urls.add(full)
                time.sleep(1.5)
            except Exception:
                time.sleep(3)
                continue

        if not bfo_urls:
            print(f"  {date} {fa} vs {fb}: no BFO URLs found")
            continue

        # Try Wayback for each URL
        found = False
        for bfo_url in list(bfo_urls)[:5]:
            html = fetch_wayback(bfo_url)
            if not html:
                time.sleep(2)
                continue
            fights = parse_fights_from_bfo(html)
            for pf in fights:
                pfa = norm(pf["fa"]).split()[-1] if norm(pf["fa"]).split() else ""
                pfb = norm(pf["fb"]).split()[-1] if norm(pf["fb"]).split() else ""
                rev = False
                if fa_last == pfa and fb_last == pfb:
                    pass
                elif fa_last == pfb and fb_last == pfa:
                    rev = True
                else:
                    continue
                a_am = pf["bo"] if rev else pf["ao"]
                b_am = pf["ao"] if rev else pf["bo"]
                ad = np.mean([a2d(o) for o in a_am])
                bd = np.mean([a2d(o) for o in b_am])
                if ad > 0 and bd > 0:
                    ai, bi = 1.0 / ad, 1.0 / bd
                    t = ai + bi
                    recovered.append({
                        "event_date": date, "fighter_a": fa, "fighter_b": fb,
                        "query_date": date, "offset_days": 0,
                        "a_fair_prob": round(ai / t, 6), "b_fair_prob": round(bi / t, 6),
                        "a_decimal_odds": round(ad, 6), "b_decimal_odds": round(bd, 6),
                        "num_bookmakers": len(a_am),
                    })
                    print(f"  FOUND: {date} {fa} vs {fb} ({len(a_am)} books)")
                    found = True
                    break
            if found:
                break
            time.sleep(2)

    print(f"\nRecovered: {len(recovered)}")
    if recovered:
        ml = pd.read_csv(ML_PATH)
        combined = pd.concat([ml, pd.DataFrame(recovered)], ignore_index=True)
        combined = combined.drop_duplicates(subset=["event_date", "fighter_a", "fighter_b"], keep="last")
        combined.to_csv(ML_PATH, index=False)
        print(f"Updated: {len(combined)} rows")

        rec_keys = {(r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"])) for r in recovered}
        still = [r for _, r in missing.iterrows()
                 if (r["event_date"], norm(r["fighter_a"]), norm(r["fighter_b"])) not in rec_keys]
        pd.DataFrame(still).to_csv(MISS_PATH, index=False)
        print(f"Still missing: {len(still)}")


if __name__ == "__main__":
    main()
