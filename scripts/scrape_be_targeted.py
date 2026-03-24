"""Targeted BetExplorer scraper — one category at a time, fast Selenium with short timeouts."""

import re
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_TEAM_EVENTS = ROOT / "data" / "raw" / "tennis" / "betexplorer_team_events_odds.csv"
OUTPUT_WTA125 = ROOT / "data" / "raw" / "tennis" / "betexplorer_wta125_odds.csv"


def get_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    d = webdriver.Chrome(options=opts)
    d.set_page_load_timeout(15)
    return d


def _parse_loaded_page(driver):
    """Parse the currently loaded BetExplorer results page."""
    rows = []
    try:
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
            m = re.match(r'^(.+?\.)\s*-\s*(.+)$', pt)
            if not m:
                m = re.match(r'^(.+?)\s+-\s+(.+)$', pt)
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
            })
    except Exception as e:
        print(f"    ERR: {e}")
    return rows


def _normalize_stage_url(url):
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def parse_page(driver, url):
    """Parse a BetExplorer results page plus any stage sub-pages."""
    rows = []
    stage_urls = []
    try:
        driver.get(url)
        time.sleep(2)
        rows.extend(_parse_loaded_page(driver))

        base_no_fragment = url.split("#", 1)[0]
        base_path = urlsplit(base_no_fragment).path
        for link in driver.find_elements(By.CSS_SELECTOR, "a[href*='?stage=']"):
            href = (link.get_attribute("href") or "").strip()
            if not href:
                continue
            normalized = _normalize_stage_url(href)
            parts = urlsplit(normalized)
            if parts.path != base_path or "?stage=" not in normalized:
                continue
            if normalized not in stage_urls:
                stage_urls.append(normalized)

        for stage_url in stage_urls:
            driver.get(stage_url)
            time.sleep(2)
            rows.extend(_parse_loaded_page(driver))
    except Exception as e:
        print(f"    ERR: {e}")
    return rows


BASE = "https://www.betexplorer.com/tennis"


def scrape_category(driver, category, filters=None):
    """Scrape a specific category of URLs. Returns DataFrame."""
    all_rows = []
    filters = [item.lower() for item in (filters or [])]

    if category == "bjk_finals":
        urls = []
        for y in [2022, 2023, 2024, 2025]:
            urls.append((f"{BASE}/wta-singles/billie-jean-king-cup-finals-{y}/results/", f"bjk-finals-{y}", "wta", "bjk_cup"))
        for url, name, tour, etype in urls:
            rows = parse_page(driver, url)
            n = sum(1 for r in rows if r["odds_w"] is not None)
            print(f"  {name}: {len(rows)} ({n} odds)", flush=True)
            for r in rows:
                r.update(tournament=name, tour=tour, event_type=etype, source="betexplorer")
            all_rows.extend(rows)
            time.sleep(1)

    elif category == "bjk_playoffs":
        urls = []
        for y in [2022, 2023, 2024, 2025]:
            urls.append((f"{BASE}/wta-singles/billie-jean-king-cup-qualifiers-{y}/results/", f"bjk-qual-{y}", "wta", "bjk_cup"))
            urls.append((f"{BASE}/wta-singles/billie-jean-king-cup-playoffs-{y}/results/", f"bjk-po-{y}", "wta", "bjk_cup"))
        for url, name, tour, etype in urls:
            rows = parse_page(driver, url)
            n = sum(1 for r in rows if r["odds_w"] is not None)
            print(f"  {name}: {len(rows)} ({n} odds)", flush=True)
            for r in rows:
                r.update(tournament=name, tour=tour, event_type=etype, source="betexplorer")
            all_rows.extend(rows)
            time.sleep(1)

    elif category == "bjk_wg":
        urls = []
        for y in [2022, 2023, 2024, 2025]:
            urls.append((f"{BASE}/wta-singles/billie-jean-king-cup-world-group-{y}/results/", f"bjk-wg-{y}", "wta", "bjk_cup"))
        for url, name, tour, etype in urls:
            rows = parse_page(driver, url)
            n = sum(1 for r in rows if r["odds_w"] is not None)
            print(f"  {name}: {len(rows)} ({n} odds)", flush=True)
            for r in rows:
                r.update(tournament=name, tour=tour, event_type=etype, source="betexplorer")
            all_rows.extend(rows)
            time.sleep(1)

    elif category == "bjk_g1":
        urls = []
        for y in [2022, 2023, 2024, 2025]:
            urls.append((f"{BASE}/wta-singles/billie-jean-king-cup-group-i-{y}/results/", f"bjk-g1-{y}", "wta", "bjk_cup"))
            urls.append((f"{BASE}/wta-singles/billie-jean-king-cup-group-ii-{y}/results/", f"bjk-g2-{y}", "wta", "bjk_cup"))
        for url, name, tour, etype in urls:
            rows = parse_page(driver, url)
            n = sum(1 for r in rows if r["odds_w"] is not None)
            print(f"  {name}: {len(rows)} ({n} odds)", flush=True)
            for r in rows:
                r.update(tournament=name, tour=tour, event_type=etype, source="betexplorer")
            all_rows.extend(rows)
            time.sleep(1)

    elif category == "davis_finals":
        urls = []
        for y in [2022, 2023, 2024, 2025]:
            urls.append((f"{BASE}/atp-singles/davis-cup-finals-{y}/results/", f"dc-finals-{y}", "atp", "davis_cup"))
        for url, name, tour, etype in urls:
            rows = parse_page(driver, url)
            n = sum(1 for r in rows if r["odds_w"] is not None)
            print(f"  {name}: {len(rows)} ({n} odds)", flush=True)
            for r in rows:
                r.update(tournament=name, tour=tour, event_type=etype, source="betexplorer")
            all_rows.extend(rows)
            time.sleep(1)

    elif category == "davis_quals":
        urls = []
        for y in [2022, 2023, 2024, 2025]:
            urls.append((f"{BASE}/atp-singles/davis-cup-qualifiers-{y}/results/", f"dc-qual-{y}", "atp", "davis_cup"))
        for url, name, tour, etype in urls:
            rows = parse_page(driver, url)
            n = sum(1 for r in rows if r["odds_w"] is not None)
            print(f"  {name}: {len(rows)} ({n} odds)", flush=True)
            for r in rows:
                r.update(tournament=name, tour=tour, event_type=etype, source="betexplorer")
            all_rows.extend(rows)
            time.sleep(1)

    elif category == "davis_wg":
        urls = []
        for y in [2022, 2023, 2024, 2025]:
            urls.append((f"{BASE}/atp-singles/davis-cup-world-group-{y}/results/", f"dc-wg-{y}", "atp", "davis_cup"))
            urls.append((f"{BASE}/atp-singles/davis-cup-world-group-i-{y}/results/", f"dc-wg1-{y}", "atp", "davis_cup"))
            urls.append((f"{BASE}/atp-singles/davis-cup-world-group-ii-{y}/results/", f"dc-wg2-{y}", "atp", "davis_cup"))
        for url, name, tour, etype in urls:
            rows = parse_page(driver, url)
            n = sum(1 for r in rows if r["odds_w"] is not None)
            print(f"  {name}: {len(rows)} ({n} odds)", flush=True)
            for r in rows:
                r.update(tournament=name, tour=tour, event_type=etype, source="betexplorer")
            all_rows.extend(rows)
            time.sleep(1)

    elif category == "united_cup":
        urls = []
        for y in [2023, 2024, 2025, 2026]:
            urls.append((f"{BASE}/atp-singles/united-cup-{y}/results/", f"uc-atp-{y}", "atp", "united_cup"))
            urls.append((f"{BASE}/wta-singles/united-cup-{y}/results/", f"uc-wta-{y}", "wta", "united_cup"))
        for url, name, tour, etype in urls:
            rows = parse_page(driver, url)
            n = sum(1 for r in rows if r["odds_w"] is not None)
            print(f"  {name}: {len(rows)} ({n} odds)", flush=True)
            for r in rows:
                r.update(tournament=name, tour=tour, event_type=etype, source="betexplorer")
            all_rows.extend(rows)
            time.sleep(1)

    elif category == "atp_cup":
        urls = [(f"{BASE}/atp-singles/atp-cup-{y}/results/", f"atp-cup-{y}", "atp", "atp_cup") for y in [2022]]
        for url, name, tour, etype in urls:
            rows = parse_page(driver, url)
            n = sum(1 for r in rows if r["odds_w"] is not None)
            print(f"  {name}: {len(rows)} ({n} odds)", flush=True)
            for r in rows:
                r.update(tournament=name, tour=tour, event_type=etype, source="betexplorer")
            all_rows.extend(rows)
            time.sleep(1)

    elif category == "nextgen":
        urls = []
        for y in [2022, 2023, 2024, 2025, 2026]:
            for slug in ["next-gen-finals-jeddah", "next-gen-finals-milan", "next-gen-finals",
                         "nextgen-atp-finals", "nitto-atp-next-gen-finals", "next-gen-atp-finals",
                         "next-gen-finals-jeddah-saudi-arabia", "next-gen-finals-saudi-arabia"]:
                urls.append((f"{BASE}/atp-singles/{slug}-{y}/results/", f"ng-{slug}-{y}", "atp", "nextgen"))
        for url, name, tour, etype in urls:
            rows = parse_page(driver, url)
            n = sum(1 for r in rows if r["odds_w"] is not None)
            if rows:
                print(f"  {name}: {len(rows)} ({n} odds)", flush=True)
            for r in rows:
                r.update(tournament=name, tour=tour, event_type=etype, source="betexplorer")
            all_rows.extend(rows)
            time.sleep(0.5)

    elif category == "laver_cup":
        urls = [(f"{BASE}/atp-singles/laver-cup-{y}/results/", f"lc-{y}", "atp", "laver_cup") for y in [2022, 2023, 2024]]
        for url, name, tour, etype in urls:
            rows = parse_page(driver, url)
            n = sum(1 for r in rows if r["odds_w"] is not None)
            print(f"  {name}: {len(rows)} ({n} odds)", flush=True)
            for r in rows:
                r.update(tournament=name, tour=tour, event_type=etype, source="betexplorer")
            all_rows.extend(rows)
            time.sleep(1)

    elif category == "wta125":
        urls = []
        slugs = [
            "antalya", "antalya-2", "antalya-3",
            "austin", "canberra", "midland", "mumbai", "manila",
            "oeiras", "oeiras-2", "oeiras-3", "oeiras-indoor", "oeiras-indoor-2",
            "les-sables-dolonne", "les-sables-d-olonne",
            "la-bisbal-demporda", "la-bisbal-d-emporda",
            "newport", "san-sebastian",
            "vic", "parma", "birmingham", "ilkley", "valencia", "bastad",
            "contrexeville", "warsaw", "guadalajara", "montreux", "ljubljana",
            "jingshan", "cosenza", "samsun", "suzhou", "mallorca",
            "rio-de-janeiro", "florianopolis", "queretaro", "rovereto",
            "cali", "tucuman", "buenos-aires", "angers", "quito",
            "cancun", "puerto-vallarta", "saint-malo",
        ]
        for slug in slugs:
            if filters and not any(filter_text in slug for filter_text in filters):
                continue
            for y in [2022, 2023, 2024, 2025, 2026]:
                urls.append((f"{BASE}/challenger-women-singles/{slug}-{y}/results/",
                             f"w125-{slug}-{y}", "wta", "125"))
        for url, name, tour, etype in urls:
            rows = parse_page(driver, url)
            n = sum(1 for r in rows if r["odds_w"] is not None)
            if rows:
                print(f"  {name}: {len(rows)} ({n} odds)", flush=True)
            for r in rows:
                r.update(tournament=name, tour=tour, event_type=etype, source="betexplorer")
            all_rows.extend(rows)
            time.sleep(0.5)

    elif category == "miami":
        urls = [
            (f"{BASE}/atp-singles/miami-2026/results/", "miami-atp-2026", "atp", "main_tour"),
            (f"{BASE}/wta-singles/miami-2026/results/", "miami-wta-2026", "wta", "main_tour"),
        ]
        for url, name, tour, etype in urls:
            rows = parse_page(driver, url)
            n = sum(1 for r in rows if r["odds_w"] is not None)
            print(f"  {name}: {len(rows)} ({n} odds)", flush=True)
            for r in rows:
                r.update(tournament=name, tour=tour, event_type=etype, source="betexplorer")
            all_rows.extend(rows)
            time.sleep(1)

    return all_rows


def main():
    known_categories = {
        "bjk_finals", "bjk_playoffs", "bjk_wg", "bjk_g1",
        "davis_finals", "davis_quals", "davis_wg",
        "united_cup", "atp_cup", "nextgen", "laver_cup",
        "wta125", "miami",
    }
    raw_args = sys.argv[1:]
    if raw_args and raw_args[0] in known_categories:
        categories = [raw_args[0]]
        filters = raw_args[1:]
    else:
        categories = raw_args if raw_args else [
        "bjk_finals", "bjk_playoffs", "bjk_wg", "bjk_g1",
        "davis_finals", "davis_quals", "davis_wg",
        "united_cup", "atp_cup", "nextgen", "laver_cup",
        "wta125", "miami",
        ]
        filters = []
    output_path = OUTPUT_WTA125 if categories and set(categories) == {"wta125"} else OUTPUT_TEAM_EVENTS

    driver = get_driver()
    all_data = []

    try:
        for cat in categories:
            print(f"\n=== {cat.upper()} ===", flush=True)
            rows = scrape_category(driver, cat, filters=filters if cat == "wta125" else None)
            all_data.extend(rows)
            print(f"  -> {len(rows)} total for {cat}", flush=True)
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

    if output_path == OUTPUT_WTA125:
        out_df = df[df["event_type"] == "125"].copy()
        out_df = out_df.rename(columns={"odds_w": "B365W", "odds_l": "B365L"})
        out_df = out_df[["Date", "Winner", "Loser", "B365W", "B365L", "tournament", "tour", "source"]]
        try:
            ex = pd.read_csv(output_path)
            ex["Date"] = pd.to_datetime(ex["Date"], errors="coerce")
            combined = pd.concat([ex, out_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["Date", "Winner", "Loser"], keep="last")
        except FileNotFoundError:
            combined = out_df
    else:
        try:
            ex = pd.read_csv(output_path)
            ex["Date"] = pd.to_datetime(ex["Date"], errors="coerce")
            combined = pd.concat([ex, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["Date", "Winner", "Loser"], keep="first")
        except FileNotFoundError:
            combined = df

    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined)} total rows to {output_path}")


if __name__ == "__main__":
    main()
