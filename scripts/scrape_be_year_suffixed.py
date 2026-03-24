"""Targeted BetExplorer scraper using year-suffixed URL slugs.

Key insight: BetExplorer uses /tournament-name-YYYY/ slugs for historical seasons,
not ?season=YYYY params. This scraper targets all missing team events with correct URLs.
"""

import re
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://www.betexplorer.com/tennis"


def get_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    d = webdriver.Chrome(options=opts)
    d.set_page_load_timeout(20)
    return d


def parse_page(driver, url, tname, tour, etype):
    rows = []
    try:
        driver.get(url)
        time.sleep(3)
        tables = driver.find_elements(By.CSS_SELECTOR, "table")
        if not tables:
            return rows
        trs = tables[0].find_elements(By.TAG_NAME, "tr")
        rnd = ""
        for tr in trs:
            tds = tr.find_elements(By.TAG_NAME, "td")
            ths = tr.find_elements(By.TAG_NAME, "th")
            cells = ths + tds
            if not cells:
                continue
            ft = cells[0].text.strip()
            if cells[0].get_attribute("colspan") and ft:
                rnd = ft
                continue
            if len(cells) < 4:
                continue
            pt = cells[0].text.strip()
            if not pt or len(pt) < 5:
                continue
            m = re.match(r"^(.+?\.)\s*-\s*(.+)$", pt)
            if not m:
                m = re.match(r"^(.+?)\s+-\s+(.+)$", pt)
            if not m:
                continue
            p1, p2 = m.group(1).strip(), m.group(2).strip()
            if len(p1) < 3 or len(p2) < 3:
                continue
            st = cells[1].text.strip()
            sm = re.match(r"(\d+):(\d+)", st)
            if not sm:
                continue
            s1, s2 = int(sm.group(1)), int(sm.group(2))
            if s1 == s2:
                continue
            w = p1 if s1 > s2 else p2
            lo = p2 if s1 > s2 else p1
            oc = [c for c in cells if "odds" in (c.get_attribute("class") or "")]
            op1 = op2 = None
            for i, o in enumerate(oc[:2]):
                v = None
                try:
                    t2 = o.text.strip()
                    if t2 and t2 != "-":
                        v = float(t2)
                except ValueError:
                    pass
                if v is None:
                    try:
                        dd = o.get_attribute("data-odd")
                        if dd:
                            v = float(dd)
                    except (ValueError, TypeError):
                        pass
                if i == 0:
                    op1 = v
                else:
                    op2 = v
            dt = cells[-1].text.strip()
            dm = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", dt)
            ds = f"{dm.group(3)}-{dm.group(2)}-{dm.group(1)}" if dm else ""
            if not ds:
                continue
            iw = (w == p1)
            rows.append({
                "Date": ds, "Winner": w, "Loser": lo, "score": st, "round": rnd,
                "odds_w": op1 if iw else op2, "odds_l": op2 if iw else op1,
                "tournament": tname, "tour": tour, "event_type": etype, "source": "betexplorer",
            })
    except Exception as e:
        pass  # Skip silently
    return rows


def try_urls(driver, urls):
    all_rows = []
    for url, name, tour, etype in urls:
        rows = parse_page(driver, url, name, tour, etype)
        n = sum(1 for r in rows if r["odds_w"] is not None)
        if rows:
            print(f"  {name}: {len(rows)} ({n} with odds)")
        all_rows.extend(rows)
        time.sleep(1)
    return all_rows


def main():
    driver = get_driver()
    all_data = []

    try:
        # === DAVIS CUP ===
        print("=== Davis Cup ===")
        urls = []
        for y in [2022, 2023, 2024, 2025]:
            urls.append((f"{BASE}/atp-singles/davis-cup-world-group-{y}/results/", f"dc-wg-{y}", "atp", "davis_cup"))
            urls.append((f"{BASE}/atp-singles/davis-cup-finals-{y}/results/", f"dc-fin-{y}", "atp", "davis_cup"))
            urls.append((f"{BASE}/atp-singles/davis-cup-qualifiers-{y}/results/", f"dc-qual-{y}", "atp", "davis_cup"))
            urls.append((f"{BASE}/atp-singles/davis-cup-group-i-{y}/results/", f"dc-g1-{y}", "atp", "davis_cup"))
            urls.append((f"{BASE}/atp-singles/davis-cup-world-group-i-{y}/results/", f"dc-wg1-{y}", "atp", "davis_cup"))
            urls.append((f"{BASE}/atp-singles/davis-cup-world-group-ii-{y}/results/", f"dc-wg2-{y}", "atp", "davis_cup"))
        all_data.extend(try_urls(driver, urls))

        # === BJK CUP ===
        print("\n=== BJK Cup ===")
        urls = []
        for y in [2022, 2023, 2024, 2025]:
            urls.append((f"{BASE}/wta-singles/billie-jean-king-cup-world-group-{y}/results/", f"bjk-wg-{y}", "wta", "bjk_cup"))
            urls.append((f"{BASE}/wta-singles/billie-jean-king-cup-finals-{y}/results/", f"bjk-fin-{y}", "wta", "bjk_cup"))
            urls.append((f"{BASE}/wta-singles/billie-jean-king-cup-qualifiers-{y}/results/", f"bjk-qual-{y}", "wta", "bjk_cup"))
            urls.append((f"{BASE}/wta-singles/billie-jean-king-cup-group-i-{y}/results/", f"bjk-g1-{y}", "wta", "bjk_cup"))
            urls.append((f"{BASE}/wta-singles/billie-jean-king-cup-group-ii-{y}/results/", f"bjk-g2-{y}", "wta", "bjk_cup"))
        all_data.extend(try_urls(driver, urls))

        # === UNITED CUP ===
        print("\n=== United Cup ===")
        urls = []
        for y in [2023, 2024, 2025, 2026]:
            urls.append((f"{BASE}/atp-singles/united-cup-{y}/results/", f"uc-atp-{y}", "atp", "united_cup"))
            urls.append((f"{BASE}/wta-singles/united-cup-{y}/results/", f"uc-wta-{y}", "wta", "united_cup"))
        all_data.extend(try_urls(driver, urls))

        # === ATP CUP ===
        print("\n=== ATP Cup ===")
        urls = [(f"{BASE}/atp-singles/atp-cup-{y}/results/", f"atp-cup-{y}", "atp", "atp_cup") for y in [2022]]
        all_data.extend(try_urls(driver, urls))

        # === NEXTGEN ===
        print("\n=== NextGen ===")
        urls = []
        for y in [2022, 2023, 2024]:
            for s in ["next-gen-finals-jeddah", "next-gen-finals-milan", "next-gen-finals", "nextgen-atp-finals"]:
                urls.append((f"{BASE}/atp-singles/{s}-{y}/results/", f"ng-{s}-{y}", "atp", "nextgen"))
        all_data.extend(try_urls(driver, urls))

        # === LAVER CUP ===
        print("\n=== Laver Cup ===")
        urls = [(f"{BASE}/atp-singles/laver-cup-{y}/results/", f"lc-{y}", "atp", "laver_cup") for y in [2022, 2023, 2024]]
        all_data.extend(try_urls(driver, urls))

        # === WTA 125 GAPS ===
        print("\n=== WTA 125 Gaps ===")
        urls = []
        for slug in ["antalya", "antalya-2", "antalya-3", "austin", "canberra",
                      "midland", "mumbai", "manila", "oeiras-2", "oeiras-3",
                      "les-sables-dolonne", "les-sables-d-olonne"]:
            for y in [2022, 2023, 2024, 2025]:
                urls.append((f"{BASE}/challenger-women-singles/{slug}-{y}/results/",
                             f"w125-{slug}-{y}", "wta", "125"))
        all_data.extend(try_urls(driver, urls))

    finally:
        driver.quit()

    if not all_data:
        print("\nNo data!")
        return

    df = pd.DataFrame(all_data)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"] >= "2022-01-01"].drop_duplicates(subset=["Date", "Winner", "Loser"], keep="first")
    both = (df["odds_w"].notna() & df["odds_l"].notna()).sum()
    print(f"\nTOTAL: {len(df)} matches, {both} with both odds")
    for et, g in df.groupby("event_type"):
        nb = (g["odds_w"].notna() & g["odds_l"].notna()).sum()
        print(f"  {et}: {len(g)} ({nb} with odds)")

    # Merge with existing
    out = ROOT / "data" / "raw" / "tennis" / "betexplorer_team_events_odds.csv"
    try:
        ex = pd.read_csv(out)
        ex["Date"] = pd.to_datetime(ex["Date"], errors="coerce")
        combined = pd.concat([ex, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date", "Winner", "Loser"], keep="first")
    except FileNotFoundError:
        combined = df
    combined.to_csv(out, index=False)
    print(f"Saved {len(combined)} total rows to {out}")


if __name__ == "__main__":
    main()
