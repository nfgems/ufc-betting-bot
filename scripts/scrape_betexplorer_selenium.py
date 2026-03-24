"""Scrape BetExplorer.com odds using Selenium for JS-rendered content.

Gets both winner and loser odds from the rendered page.
Targets: WTA 125s, Davis Cup, BJK Cup, United Cup, Olympics, Laver Cup, NextGen.
"""

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

ROOT = Path(__file__).resolve().parent.parent

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(30)
    return driver


def get_tournament_links(base_path):
    """Get tournament links from index page using requests (no JS needed)."""
    url = f"https://www.betexplorer.com{base_path}"
    r = requests.get(url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")

    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(base_path) and href != base_path and href != base_path.rstrip("/"):
            slug = href.rstrip("/").split("/")[-1]
            if slug and slug != "fixtures":
                links.add(href.rstrip("/"))
    return sorted(links)


def parse_results_selenium(driver, url, tournament_name, tour, event_type):
    """Parse a results page using Selenium to get JS-rendered odds."""
    rows = []
    try:
        driver.get(url)
        time.sleep(2)  # Wait for JS

        # Find the results table
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

            # Split player names on the dash between them
            # Handle hyphenated names like "Friedsam A-L." by looking for ". -" or ".-" pattern
            # BetExplorer format: "Player1 I.-Player2 I."
            m = re.match(r'^(.+?\.)\s*-\s*(.+)$', players_text)
            if not m:
                # Try without period requirement
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
            winner = player1 if sets1 > sets2 else player2
            loser = player2 if sets1 > sets2 else player1

            # Get odds from rendered cells
            odds_cells = [c for c in cells if "table-main__odds" in (c.get_attribute("class") or "")]
            odds_p1 = None
            odds_p2 = None

            if len(odds_cells) >= 2:
                try:
                    t1 = odds_cells[0].text.strip()
                    if t1:
                        odds_p1 = float(t1)
                except (ValueError, IndexError):
                    pass
                try:
                    t2 = odds_cells[1].text.strip()
                    if t2:
                        odds_p2 = float(t2)
                except (ValueError, IndexError):
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
        print(f"    Error parsing {url}: {e}")

    return rows


def scrape_with_season(driver, base_url, tournament_name, tour, event_type, years=None):
    """Scrape a tournament results page, optionally with season parameters."""
    all_rows = []

    if years is None:
        years = [None]  # Just scrape the default (current) page

    for year in years:
        if year:
            url = f"{base_url}?season={year}"
        else:
            url = base_url
        rows = parse_results_selenium(driver, url, tournament_name, tour, event_type)
        all_rows.extend(rows)

    return all_rows


def scrape_with_seasons(driver, name, base_url, tour, event_type, seasons):
    """Scrape a tournament across multiple seasons."""
    all_rows = []
    for season in seasons:
        url = f"{base_url}?season={season}"
        print(f"  {name} {season}...", end=" ", flush=True)
        rows = parse_results_selenium(driver, url, f"{name}-{season}", tour, event_type)
        n_with = sum(1 for r in rows if r.get("odds_w") is not None)
        all_rows.extend(rows)
        print(f"{len(rows)} matches ({n_with} with odds)")
        time.sleep(1)
    return all_rows


def main():
    driver = get_driver()
    all_data = []
    seasons = [2022, 2023, 2024, 2025]

    try:
        # ========== Davis Cup ==========
        print("=" * 60)
        print("Scraping Davis Cup (all groups, all seasons)")
        print("=" * 60)

        davis_bases = [
            ("davis-cup-world-group", "https://www.betexplorer.com/tennis/teams-men/davis-cup-world-group/results/"),
            ("davis-cup-group-i", "https://www.betexplorer.com/tennis/teams-men/davis-cup-group-i/results/"),
            ("davis-cup-group-ii", "https://www.betexplorer.com/tennis/teams-men/davis-cup-group-ii/results/"),
            ("davis-cup-group-iii", "https://www.betexplorer.com/tennis/teams-men/davis-cup-group-iii/results/"),
            ("davis-cup-group-iv", "https://www.betexplorer.com/tennis/teams-men/davis-cup-group-iv/results/"),
            ("davis-cup-group-v", "https://www.betexplorer.com/tennis/teams-men/davis-cup-group-v/results/"),
        ]

        for name, base_url in davis_bases:
            rows = scrape_with_seasons(driver, name, base_url, "atp", "davis_cup", seasons)
            all_data.extend(rows)

        # ========== BJK Cup ==========
        print("\n" + "=" * 60)
        print("Scraping BJK Cup (all groups, all seasons)")
        print("=" * 60)

        bjk_bases = [
            ("bjk-cup-world-group", "https://www.betexplorer.com/tennis/teams-women/billie-jean-king-cup-world-group/results/"),
            ("bjk-cup-group-i", "https://www.betexplorer.com/tennis/teams-women/billie-jean-king-cup-group-i/results/"),
            ("bjk-cup-group-ii", "https://www.betexplorer.com/tennis/teams-women/billie-jean-king-cup-group-ii/results/"),
            ("bjk-cup-group-iii", "https://www.betexplorer.com/tennis/teams-women/billie-jean-king-cup-group-iii/results/"),
            ("bjk-cup-group-iv", "https://www.betexplorer.com/tennis/teams-women/billie-jean-king-cup-group-iv/results/"),
        ]

        for name, base_url in bjk_bases:
            rows = scrape_with_seasons(driver, name, base_url, "wta", "bjk_cup", seasons)
            all_data.extend(rows)

        # ========== United Cup ==========
        print("\n" + "=" * 60)
        print("Scraping United Cup (all seasons)")
        print("=" * 60)

        rows = scrape_with_seasons(driver, "united-cup",
            "https://www.betexplorer.com/tennis/teams-mix/united-cup/results/",
            "mixed", "united_cup", [2023, 2024, 2025])
        all_data.extend(rows)

        # ========== ATP Cup ==========
        print("\n" + "=" * 60)
        print("Scraping ATP Cup")
        print("=" * 60)

        rows = scrape_with_seasons(driver, "atp-cup",
            "https://www.betexplorer.com/tennis/teams-men/atp-cup/results/",
            "atp", "atp_cup", [2022])
        all_data.extend(rows)

        # ========== Laver Cup ==========
        print("\n" + "=" * 60)
        print("Scraping Laver Cup")
        print("=" * 60)

        rows = scrape_with_seasons(driver, "laver-cup",
            "https://www.betexplorer.com/tennis/teams-men/laver-cup/results/",
            "atp", "laver_cup", [2022, 2023, 2024])
        all_data.extend(rows)

        # ========== Olympics 2024 ==========
        print("\n" + "=" * 60)
        print("Scraping Olympics 2024")
        print("=" * 60)

        for name, url, tour in [
            ("olympics-atp", "https://www.betexplorer.com/tennis/world/atp-olympic-games/results/", "atp"),
            ("olympics-wta", "https://www.betexplorer.com/tennis/world/wta-olympic-games/results/", "wta"),
        ]:
            print(f"  {name}...", end=" ", flush=True)
            rows = parse_results_selenium(driver, url, name, tour, "olympics")
            n_with = sum(1 for r in rows if r.get("odds_w") is not None)
            all_data.extend(rows)
            print(f"{len(rows)} matches ({n_with} with odds)")
            time.sleep(1)

        # ========== NextGen Finals ==========
        print("\n" + "=" * 60)
        print("Scraping NextGen Finals")
        print("=" * 60)

        nextgen_bases = [
            ("nextgen-jeddah", "https://www.betexplorer.com/tennis/world/atp-next-gen-finals-jeddah/results/"),
            ("nextgen", "https://www.betexplorer.com/tennis/world/atp-next-gen-finals/results/"),
        ]
        for name, base_url in nextgen_bases:
            rows = scrape_with_seasons(driver, name, base_url, "atp", "nextgen", [2022, 2023, 2024])
            all_data.extend(rows)

        # ========== WTA 125s ==========
        print("\n" + "=" * 60)
        print("Scraping WTA 125 events")
        print("=" * 60)

        tourney_paths = get_tournament_links("/tennis/challenger-women-singles/")
        print(f"Found {len(tourney_paths)} WTA 125 tournaments")

        for i, path in enumerate(tourney_paths):
            name = path.rstrip("/").split("/")[-1]
            url = f"https://www.betexplorer.com{path}/results/"
            print(f"  [{i+1}/{len(tourney_paths)}] {name}...", end=" ", flush=True)

            rows = parse_results_selenium(driver, url, name, "wta", "125")
            all_data.extend(rows)
            n_with_odds = sum(1 for r in rows if r["odds_w"] is not None and r["odds_l"] is not None)
            print(f"{len(rows)} matches ({n_with_odds} with odds)")
            time.sleep(0.3)

        # ========== Main tour gaps ==========
        print("\n" + "=" * 60)
        print("Scraping main tour gaps (ATP Finals, Metz, Guadalajara)")
        print("=" * 60)

        main_tour_urls = [
            ("atp-finals-2025", "https://www.betexplorer.com/tennis/italy/atp-nitto-finals/results/?season=2025", "atp"),
            ("atp-finals-2025-alt", "https://www.betexplorer.com/tennis/united-kingdom/atp-tour-finals/results/?season=2025", "atp"),
            ("metz-2025", "https://www.betexplorer.com/tennis/france/atp-metz/results/?season=2025", "atp"),
            ("guadalajara-wta-2022", "https://www.betexplorer.com/tennis/mexico/wta-guadalajara-2/results/", "wta"),
            ("miami-wta-2026", "https://www.betexplorer.com/tennis/usa/wta-miami/results/?season=2026", "wta"),
            ("miami-atp-2026", "https://www.betexplorer.com/tennis/usa/atp-miami/results/?season=2026", "atp"),
        ]

        for name, url, tour in main_tour_urls:
            print(f"  {name}...", end=" ", flush=True)
            rows = parse_results_selenium(driver, url, name, tour, "main_tour")
            n_with = sum(1 for r in rows if r.get("odds_w") is not None)
            all_data.extend(rows)
            print(f"{len(rows)} matches ({n_with} with odds)")
            time.sleep(1)

    finally:
        driver.quit()

    if not all_data:
        print("\nNo data scraped!")
        return

    df = pd.DataFrame(all_data)
    print(f"\n{'='*60}")
    print(f"TOTAL SCRAPED: {len(df)} matches")
    print(f"{'='*60}")

    # Filter to 2022+
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"] >= "2022-01-01"].copy()
    print(f"After filtering to 2022+: {len(df)} matches")

    # Stats
    both_odds = (df["odds_w"].notna() & df["odds_l"].notna()).sum()
    any_odds = (df["odds_w"].notna() | df["odds_l"].notna()).sum()
    print(f"With both odds: {both_odds}")
    print(f"With any odds: {any_odds}")
    print(f"\nBy event type:")
    for et, g in df.groupby("event_type"):
        n_both = (g["odds_w"].notna() & g["odds_l"].notna()).sum()
        print(f"  {et}: {len(g)} matches ({n_both} with both odds)")
    print(f"\nBy year:")
    print(df.groupby(df["Date"].dt.year).size().to_string())

    # Save
    output_path = ROOT / "data" / "raw" / "tennis" / "betexplorer_team_events_odds.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
