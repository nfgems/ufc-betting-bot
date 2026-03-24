"""Comprehensive BetExplorer scraper for ALL remaining odds gaps.

Tries year-suffixed slugs, season params, and direct URLs.
Covers: Davis Cup, BJK Cup, United Cup, ATP Cup, Olympics, NextGen, Laver Cup, WTA 125s.
"""

import re
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "raw" / "tennis" / "betexplorer_all_gaps_odds.csv"

BASE = "https://www.betexplorer.com/tennis"


def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(30)
    return driver


def parse_results_page(driver, url, tournament_name, tour, event_type):
    """Parse a BetExplorer results page using Selenium."""
    rows = []
    try:
        driver.get(url)
        time.sleep(3)

        try:
            table = driver.find_element(By.CSS_SELECTOR, "table")
        except Exception:
            return rows

        tr_elements = table.find_elements(By.TAG_NAME, "tr")
        current_round = ""

        for tr in tr_elements:
            tds = tr.find_elements(By.TAG_NAME, "td")
            ths = tr.find_elements(By.TAG_NAME, "th")
            cells = ths + tds
            if not cells:
                continue

            first_text = cells[0].text.strip()
            colspan = cells[0].get_attribute("colspan")
            if colspan and first_text:
                current_round = first_text
                continue

            if len(cells) < 4:
                continue

            players_text = cells[0].text.strip()
            if not players_text or len(players_text) < 5:
                continue

            # Split player names
            m = re.match(r'^(.+?\.)\s*-\s*(.+)$', players_text)
            if not m:
                m = re.match(r'^(.+?)\s+-\s+(.+)$', players_text)
            if not m:
                continue

            player1, player2 = m.group(1).strip(), m.group(2).strip()
            if len(player1) < 3 or len(player2) < 3:
                continue

            # Score
            score_text = cells[1].text.strip() if len(cells) > 1 else ""
            score_match = re.match(r"(\d+):(\d+)", score_text)
            if not score_match:
                continue
            sets1, sets2 = int(score_match.group(1)), int(score_match.group(2))
            if sets1 == sets2:
                continue
            winner = player1 if sets1 > sets2 else player2
            loser = player2 if sets1 > sets2 else player1

            # Odds
            odds_cells = [c for c in cells if "table-main__odds" in (c.get_attribute("class") or "")]
            odds_p1, odds_p2 = None, None

            if len(odds_cells) >= 2:
                for i, oc in enumerate(odds_cells[:2]):
                    val = None
                    try:
                        t = oc.text.strip()
                        if t and t != "-":
                            val = float(t)
                    except ValueError:
                        pass
                    if val is None:
                        try:
                            d = oc.get_attribute("data-odd")
                            if d:
                                val = float(d)
                        except (ValueError, TypeError):
                            pass
                    if i == 0:
                        odds_p1 = val
                    else:
                        odds_p2 = val

            # Date
            date_text = cells[-1].text.strip()
            dm = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", date_text)
            date_str = f"{dm.group(3)}-{dm.group(2)}-{dm.group(1)}" if dm else ""
            if not date_str:
                continue

            is_p1_winner = (winner == player1)
            rows.append({
                "Date": date_str,
                "Winner": winner,
                "Loser": loser,
                "score": score_text,
                "round": current_round,
                "odds_w": odds_p1 if is_p1_winner else odds_p2,
                "odds_l": odds_p2 if is_p1_winner else odds_p1,
                "tournament": tournament_name,
                "tour": tour,
                "event_type": event_type,
                "source": "betexplorer",
            })

    except Exception as e:
        print(f"    Error: {e}")
    return rows


def try_url(driver, url, name, tour, event_type):
    """Try a URL and return rows if it has data."""
    rows = parse_results_page(driver, url, name, tour, event_type)
    n_odds = sum(1 for r in rows if r.get("odds_w") is not None)
    if rows:
        print(f"  {name}: {len(rows)} matches ({n_odds} with odds)")
    return rows


def main():
    driver = get_driver()
    all_data = []

    try:
        # ===== DAVIS CUP =====
        print("=" * 60)
        print("DAVIS CUP SINGLES")
        print("=" * 60)

        # Davis Cup uses year-suffixed slugs AND the base slug with ?season=
        dc_divisions = [
            "davis-cup-world-group",
            "davis-cup-world-group-ii",
            "davis-cup-group-i",
            "davis-cup-group-ii",
            "davis-cup-group-iii",
            "davis-cup-group-iv",
            "davis-cup-group-v",
            "davis-cup-finals",
            "davis-cup-qualifiers",
        ]
        for div in dc_divisions:
            for year in [2022, 2023, 2024, 2025]:
                # Try year-suffixed slug first
                url1 = f"{BASE}/atp-singles/{div}-{year}/results/"
                rows = try_url(driver, url1, f"{div}-{year}", "atp", "davis_cup")
                all_data.extend(rows)
                time.sleep(1)

                if not rows:
                    # Try season param
                    url2 = f"{BASE}/atp-singles/{div}/results/?season={year}"
                    rows = try_url(driver, url2, f"{div}-s{year}", "atp", "davis_cup")
                    all_data.extend(rows)
                    time.sleep(1)

        # ===== BJK CUP =====
        print("\n" + "=" * 60)
        print("BJK CUP SINGLES")
        print("=" * 60)

        bjk_divisions = [
            "billie-jean-king-cup-world-group",
            "billie-jean-king-cup-finals",
            "billie-jean-king-cup-qualifiers",
            "billie-jean-king-cup-group-i",
            "billie-jean-king-cup-group-ii",
            "billie-jean-king-cup-group-iii",
            "billie-jean-king-cup-group-iv",
        ]
        for div in bjk_divisions:
            for year in [2022, 2023, 2024, 2025]:
                url1 = f"{BASE}/wta-singles/{div}-{year}/results/"
                rows = try_url(driver, url1, f"{div}-{year}", "wta", "bjk_cup")
                all_data.extend(rows)
                time.sleep(1)

                if not rows:
                    url2 = f"{BASE}/wta-singles/{div}/results/?season={year}"
                    rows = try_url(driver, url2, f"{div}-s{year}", "wta", "bjk_cup")
                    all_data.extend(rows)
                    time.sleep(1)

        # ===== UNITED CUP =====
        print("\n" + "=" * 60)
        print("UNITED CUP SINGLES")
        print("=" * 60)

        for tour_path, tour in [("atp-singles", "atp"), ("wta-singles", "wta")]:
            for year in [2023, 2024, 2025, 2026]:
                # Try year-suffixed
                url1 = f"{BASE}/{tour_path}/united-cup-{year}/results/"
                rows = try_url(driver, url1, f"united-cup-{tour}-{year}", tour, "united_cup")
                all_data.extend(rows)
                time.sleep(1)

                if not rows:
                    url2 = f"{BASE}/{tour_path}/united-cup/results/?season={year}"
                    rows = try_url(driver, url2, f"united-cup-{tour}-s{year}", tour, "united_cup")
                    all_data.extend(rows)
                    time.sleep(1)

        # ===== ATP CUP =====
        print("\n" + "=" * 60)
        print("ATP CUP SINGLES")
        print("=" * 60)

        for year in [2022]:
            url1 = f"{BASE}/atp-singles/atp-cup-{year}/results/"
            rows = try_url(driver, url1, f"atp-cup-{year}", "atp", "atp_cup")
            all_data.extend(rows)
            time.sleep(1)
            if not rows:
                url2 = f"{BASE}/atp-singles/atp-cup/results/?season={year}"
                rows = try_url(driver, url2, f"atp-cup-s{year}", "atp", "atp_cup")
                all_data.extend(rows)
                time.sleep(1)

        # ===== OLYMPICS 2024 =====
        print("\n" + "=" * 60)
        print("OLYMPICS 2024")
        print("=" * 60)

        for name, path, tour in [
            ("olympics-atp-2024", f"{BASE}/atp-singles/olympic-games/results/?season=2024", "atp"),
            ("olympics-wta-2024", f"{BASE}/wta-singles/olympic-games/results/?season=2024", "wta"),
            ("olympics-atp-2024-v2", f"{BASE}/atp-singles/olympic-games-2024/results/", "atp"),
            ("olympics-wta-2024-v2", f"{BASE}/wta-singles/olympic-games-2024/results/", "wta"),
            ("olympics-atp-paris", f"{BASE}/atp-singles/olympic-games-paris/results/", "atp"),
            ("olympics-wta-paris", f"{BASE}/wta-singles/olympic-games-paris/results/", "wta"),
        ]:
            rows = try_url(driver, path, name, tour, "olympics")
            all_data.extend(rows)
            time.sleep(1)

        # ===== LAVER CUP =====
        print("\n" + "=" * 60)
        print("LAVER CUP")
        print("=" * 60)

        for year in [2022, 2023, 2024]:
            url1 = f"{BASE}/atp-singles/laver-cup-{year}/results/"
            rows = try_url(driver, url1, f"laver-cup-{year}", "atp", "laver_cup")
            all_data.extend(rows)
            time.sleep(1)
            if not rows:
                url2 = f"{BASE}/atp-singles/laver-cup/results/?season={year}"
                rows = try_url(driver, url2, f"laver-cup-s{year}", "atp", "laver_cup")
                all_data.extend(rows)
                time.sleep(1)

        # ===== NEXTGEN FINALS =====
        print("\n" + "=" * 60)
        print("NEXTGEN FINALS")
        print("=" * 60)

        nextgen_slugs = [
            "next-gen-finals-jeddah",
            "next-gen-finals-milan",
            "next-gen-finals",
            "nextgen-finals",
            "nitto-atp-next-gen-finals",
        ]
        for slug in nextgen_slugs:
            for year in [2022, 2023, 2024]:
                url1 = f"{BASE}/atp-singles/{slug}-{year}/results/"
                rows = try_url(driver, url1, f"{slug}-{year}", "atp", "nextgen")
                all_data.extend(rows)
                time.sleep(1)
                if not rows:
                    url2 = f"{BASE}/atp-singles/{slug}/results/?season={year}"
                    rows = try_url(driver, url2, f"{slug}-s{year}", "atp", "nextgen")
                    all_data.extend(rows)
                    time.sleep(1)

        # ===== WTA 125 GAPS (tournaments with 0 matches in previous scrape) =====
        print("\n" + "=" * 60)
        print("WTA 125 GAPS")
        print("=" * 60)

        # These 125s returned 0 matches - try year-suffixed slugs
        wta125_gaps = [
            "antalya", "antalya-2", "antalya-3",
            "austin", "canberra", "midland", "mumbai", "manila",
            "oeiras-2", "oeiras-3",
            "les-sables-dolonne", "les-sables-d-olonne",
            "la-bisbal-demporda", "la-bisbal-d-emporda",
        ]
        for slug in wta125_gaps:
            for year in [2022, 2023, 2024, 2025]:
                url1 = f"{BASE}/challenger-women-singles/{slug}-{year}/results/"
                rows = try_url(driver, url1, f"wta125-{slug}-{year}", "wta", "125")
                all_data.extend(rows)
                time.sleep(0.5)
                if not rows:
                    url2 = f"{BASE}/challenger-women-singles/{slug}/results/?season={year}"
                    rows = try_url(driver, url2, f"wta125-{slug}-s{year}", "wta", "125")
                    all_data.extend(rows)
                    time.sleep(0.5)

    finally:
        driver.quit()

    if not all_data:
        print("\n*** NO DATA ***")
        return

    df = pd.DataFrame(all_data)
    print(f"\n{'='*60}")
    print(f"TOTAL SCRAPED: {len(df)} matches")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"] >= "2022-01-01"].copy()
    df = df.drop_duplicates(subset=["Date", "Winner", "Loser"], keep="first")
    print(f"After filtering + dedup: {len(df)}")

    both_odds = (df["odds_w"].notna() & df["odds_l"].notna()).sum()
    print(f"With both odds: {both_odds}")

    print(f"\nBy event type:")
    for et, g in df.groupby("event_type"):
        n_both = (g["odds_w"].notna() & g["odds_l"].notna()).sum()
        print(f"  {et}: {len(g)} ({n_both} with both odds)")

    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
