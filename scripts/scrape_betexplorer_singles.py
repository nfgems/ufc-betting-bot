"""Scrape BetExplorer.com for individual singles match odds in team events + other gaps.

BetExplorer lists individual match results under /tennis/atp-singles/ and /tennis/wta-singles/,
NOT under teams-men/ or teams-women/ (those show team-level results only).

Targets: Davis Cup, BJK Cup, United Cup, ATP Cup, Olympics, NextGen, Laver Cup, Miami 2026
Output: data/raw/tennis/betexplorer_team_events_odds.csv
"""

import re
import sys
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "raw" / "tennis" / "betexplorer_team_events_odds.csv"


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

        # Find results table
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

            # Round header
            colspan = cells[0].get_attribute("colspan")
            if colspan and first_text:
                current_round = first_text
                continue

            if len(cells) < 4:
                continue

            # Match row
            players_text = cells[0].text.strip()
            if not players_text or len(players_text) < 5:
                continue

            # Split player names: "Surname I. - Surname2 J." or "Surname I.-Surname2 J."
            m = re.match(r'^(.+?\.)\s*-\s*(.+)$', players_text)
            if not m:
                m = re.match(r'^(.+?)\s+-\s+(.+)$', players_text)
            if not m:
                continue

            player1 = m.group(1).strip()
            player2 = m.group(2).strip()

            if not player1 or not player2 or len(player1) < 3 or len(player2) < 3:
                continue

            # Score
            score_text = cells[1].text.strip() if len(cells) > 1 else ""
            score_match = re.match(r"(\d+):(\d+)", score_text)
            if not score_match:
                continue
            sets1, sets2 = int(score_match.group(1)), int(score_match.group(2))
            if sets1 == sets2:
                continue  # Can't determine winner
            winner = player1 if sets1 > sets2 else player2
            loser = player2 if sets1 > sets2 else player1

            # Get odds from rendered cells
            odds_cells = [c for c in cells if "table-main__odds" in (c.get_attribute("class") or "")]
            odds_p1, odds_p2 = None, None

            if len(odds_cells) >= 2:
                try:
                    t1 = odds_cells[0].text.strip()
                    if t1 and t1 != "-":
                        odds_p1 = float(t1)
                except (ValueError, IndexError):
                    pass
                try:
                    t2 = odds_cells[1].text.strip()
                    if t2 and t2 != "-":
                        odds_p2 = float(t2)
                except (ValueError, IndexError):
                    pass

            # Also try data-odd attributes
            if odds_p1 is None and len(odds_cells) >= 2:
                try:
                    d1 = odds_cells[0].get_attribute("data-odd")
                    if d1:
                        odds_p1 = float(d1)
                except (ValueError, TypeError):
                    pass
            if odds_p2 is None and len(odds_cells) >= 2:
                try:
                    d2 = odds_cells[1].get_attribute("data-odd")
                    if d2:
                        odds_p2 = float(d2)
                except (ValueError, TypeError):
                    pass

            # Date (last cell)
            date_text = cells[-1].text.strip()
            date_match = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", date_text)
            date_str = f"{date_match.group(3)}-{date_match.group(2)}-{date_match.group(1)}" if date_match else ""

            if not date_str:
                continue

            is_p1_winner = (winner == player1)
            odds_w = odds_p1 if is_p1_winner else odds_p2
            odds_l = odds_p2 if is_p1_winner else odds_p1

            rows.append({
                "Date": date_str,
                "Winner": winner,
                "Loser": loser,
                "score": score_text,
                "round": current_round,
                "odds_w": odds_w,
                "odds_l": odds_l,
                "tournament": tournament_name,
                "tour": tour,
                "event_type": event_type,
                "source": "betexplorer",
            })

    except Exception as e:
        print(f"    Error: {e}")

    return rows


def scrape_with_seasons(driver, name, base_url, tour, event_type, seasons):
    """Scrape across multiple seasons using ?season=YYYY."""
    all_rows = []
    for season in seasons:
        url = f"{base_url}?season={season}"
        print(f"  {name} {season}...", end=" ", flush=True)
        rows = parse_results_page(driver, url, f"{name}-{season}", tour, event_type)
        n_with = sum(1 for r in rows if r.get("odds_w") is not None)
        all_rows.extend(rows)
        print(f"{len(rows)} matches ({n_with} with odds)")
        time.sleep(1.5)
    return all_rows


def main():
    driver = get_driver()
    all_data = []
    seasons = [2022, 2023, 2024, 2025]

    try:
        # ========== Davis Cup (under atp-singles) ==========
        print("=" * 60)
        print("Davis Cup singles")
        print("=" * 60)

        davis_bases = [
            ("davis-cup-wg", "https://www.betexplorer.com/tennis/atp-singles/davis-cup-world-group/results/"),
            ("davis-cup-wg2", "https://www.betexplorer.com/tennis/atp-singles/davis-cup-world-group-ii/results/"),
            ("davis-cup-g1", "https://www.betexplorer.com/tennis/atp-singles/davis-cup-group-i/results/"),
            ("davis-cup-g2", "https://www.betexplorer.com/tennis/atp-singles/davis-cup-group-ii/results/"),
            ("davis-cup-g3", "https://www.betexplorer.com/tennis/atp-singles/davis-cup-group-iii/results/"),
            ("davis-cup-g4", "https://www.betexplorer.com/tennis/atp-singles/davis-cup-group-iv/results/"),
            ("davis-cup-g5", "https://www.betexplorer.com/tennis/atp-singles/davis-cup-group-v/results/"),
        ]
        for name, base_url in davis_bases:
            rows = scrape_with_seasons(driver, name, base_url, "atp", "davis_cup", seasons)
            all_data.extend(rows)

        # ========== BJK Cup (under wta-singles) ==========
        print("\n" + "=" * 60)
        print("BJK Cup singles")
        print("=" * 60)

        bjk_bases = [
            ("bjk-cup-wg", "https://www.betexplorer.com/tennis/wta-singles/billie-jean-king-cup-world-group/results/"),
            ("bjk-cup-g1", "https://www.betexplorer.com/tennis/wta-singles/billie-jean-king-cup-group-i/results/"),
            ("bjk-cup-g2", "https://www.betexplorer.com/tennis/wta-singles/billie-jean-king-cup-group-ii/results/"),
            ("bjk-cup-g3", "https://www.betexplorer.com/tennis/wta-singles/billie-jean-king-cup-group-iii/results/"),
            ("bjk-cup-g4", "https://www.betexplorer.com/tennis/wta-singles/billie-jean-king-cup-group-iv/results/"),
        ]
        for name, base_url in bjk_bases:
            rows = scrape_with_seasons(driver, name, base_url, "wta", "bjk_cup", seasons)
            all_data.extend(rows)

        # ========== United Cup (under atp-singles AND wta-singles) ==========
        print("\n" + "=" * 60)
        print("United Cup singles")
        print("=" * 60)

        uc_bases = [
            ("united-cup-atp", "https://www.betexplorer.com/tennis/atp-singles/united-cup/results/", "atp"),
            ("united-cup-wta", "https://www.betexplorer.com/tennis/wta-singles/united-cup/results/", "wta"),
        ]
        for name, base_url, tour in uc_bases:
            rows = scrape_with_seasons(driver, name, base_url, tour, "united_cup", [2023, 2024, 2025, 2026])
            all_data.extend(rows)

        # ========== ATP Cup ==========
        print("\n" + "=" * 60)
        print("ATP Cup singles")
        print("=" * 60)

        rows = scrape_with_seasons(driver, "atp-cup",
            "https://www.betexplorer.com/tennis/atp-singles/atp-cup/results/",
            "atp", "atp_cup", [2022])
        all_data.extend(rows)

        # ========== Olympics 2024 (under atp-singles and wta-singles) ==========
        print("\n" + "=" * 60)
        print("Olympics 2024 singles")
        print("=" * 60)

        olympics_bases = [
            ("olympics-atp", "https://www.betexplorer.com/tennis/atp-singles/olympic-games/results/", "atp"),
            ("olympics-wta", "https://www.betexplorer.com/tennis/wta-singles/olympic-games/results/", "wta"),
        ]
        for name, base_url, tour in olympics_bases:
            print(f"  {name}...", end=" ", flush=True)
            rows = parse_results_page(driver, f"{base_url}?season=2024", f"{name}-2024", tour, "olympics")
            n_with = sum(1 for r in rows if r.get("odds_w") is not None)
            all_data.extend(rows)
            print(f"{len(rows)} matches ({n_with} with odds)")
            time.sleep(1.5)

        # ========== Laver Cup ==========
        print("\n" + "=" * 60)
        print("Laver Cup singles")
        print("=" * 60)

        rows = scrape_with_seasons(driver, "laver-cup",
            "https://www.betexplorer.com/tennis/atp-singles/laver-cup/results/",
            "atp", "laver_cup", [2022, 2023, 2024])
        all_data.extend(rows)

        # ========== NextGen Finals ==========
        print("\n" + "=" * 60)
        print("NextGen Finals singles")
        print("=" * 60)

        nextgen_bases = [
            ("nextgen-jeddah", "https://www.betexplorer.com/tennis/atp-singles/next-gen-finals-jeddah/results/"),
            ("nextgen-milan", "https://www.betexplorer.com/tennis/atp-singles/next-gen-finals-milan/results/"),
            ("nextgen", "https://www.betexplorer.com/tennis/atp-singles/next-gen-finals/results/"),
        ]
        for name, base_url in nextgen_bases:
            rows = scrape_with_seasons(driver, name, base_url, "atp", "nextgen", [2022, 2023, 2024])
            all_data.extend(rows)

        # ========== Main tour gaps ==========
        print("\n" + "=" * 60)
        print("Main tour gaps")
        print("=" * 60)

        main_gaps = [
            ("atp-finals", "https://www.betexplorer.com/tennis/atp-singles/london/results/", "atp", [2025]),
            ("atp-finals-alt", "https://www.betexplorer.com/tennis/atp-singles/nitto-atp-finals/results/", "atp", [2025]),
            ("metz", "https://www.betexplorer.com/tennis/atp-singles/metz/results/", "atp", [2025]),
            ("miami-atp", "https://www.betexplorer.com/tennis/atp-singles/miami/results/", "atp", [2026]),
            ("miami-wta", "https://www.betexplorer.com/tennis/wta-singles/miami/results/", "wta", [2026]),
            ("guadalajara", "https://www.betexplorer.com/tennis/wta-singles/guadalajara/results/", "wta", [2022]),
            ("guadalajara-2", "https://www.betexplorer.com/tennis/wta-singles/guadalajara-2/results/", "wta", [2022]),
        ]
        for name, base_url, tour, yrs in main_gaps:
            rows = scrape_with_seasons(driver, name, base_url, tour, "main_tour", yrs)
            all_data.extend(rows)

    finally:
        driver.quit()

    if not all_data:
        print("\n*** NO DATA ***")
        return

    df = pd.DataFrame(all_data)
    print(f"\n{'='*60}")
    print(f"TOTAL SCRAPED: {len(df)} matches")

    # Filter to 2022+
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"] >= "2022-01-01"].copy()
    df = df.drop_duplicates(subset=["Date", "Winner", "Loser"], keep="first")
    print(f"After filtering to 2022+ and dedup: {len(df)}")

    both_odds = (df["odds_w"].notna() & df["odds_l"].notna()).sum()
    print(f"With both odds: {both_odds}")

    print(f"\nBy event type:")
    for et, g in df.groupby("event_type"):
        n_both = (g["odds_w"].notna() & g["odds_l"].notna()).sum()
        print(f"  {et}: {len(g)} matches ({n_both} with both odds)")

    print(f"\nBy year:")
    print(df.groupby(df["Date"].dt.year).size().to_string())

    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
