"""
Scrape OddsPortal individual match pages for team event odds.

Strategy:
1. Navigate to the tournament results page
2. Extract individual match page URLs from the page
3. Visit each match page to get detailed odds

Output: data/raw/tennis/oddsportal_team_events_odds.csv
"""

import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "raw" / "tennis" / "oddsportal_team_events_odds.csv"


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
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })
    driver.set_page_load_timeout(45)
    return driver


def extract_match_links_from_page(driver, results_url):
    """Extract individual match page URLs from a results page."""
    match_urls = []
    try:
        driver.get(results_url)
        time.sleep(4)  # Wait for JS rendering

        # Get the page source
        html = driver.page_source

        # OddsPortal match URLs pattern: /tennis/world/tournament/player1-player2-matchid/
        # or they may have data attributes
        links = re.findall(r'href="(/tennis/[^"]*?/[A-Za-z].*?-[A-Za-z].*?/[A-Za-z0-9]+/)"', html)
        match_urls.extend(links)

        # Also try finding links via Selenium
        try:
            link_elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='/tennis/']")
            for el in link_elements:
                href = el.get_attribute("href") or ""
                # Match pages typically have a hash-like ID at the end
                if re.search(r'/[A-Za-z0-9]{6,}/$', href) and '-' in href.split('/')[-2]:
                    path = href.replace("https://www.oddsportal.com", "")
                    if path not in match_urls:
                        match_urls.append(path)
        except Exception:
            pass

    except Exception as e:
        print(f"    Error loading results page: {e}")

    return list(set(match_urls))


def scrape_match_detail(driver, match_url):
    """Scrape odds from an individual match detail page."""
    try:
        full_url = f"https://www.oddsportal.com{match_url}" if not match_url.startswith("http") else match_url
        driver.get(full_url)
        time.sleep(3)

        html = driver.page_source
        body_text = driver.find_element(By.TAG_NAME, "body").text

        # Extract player names from page title or header
        title_m = re.search(r'<title>([^<]+)</title>', html)
        if not title_m:
            return None
        title = title_m.group(1)

        # Parse "Player1 I. - Player2 J. | OddsPortal" or "Player1 vs Player2"
        players_m = re.match(
            r'(.+?)\s*(?:vs\.?|[-–])\s*(.+?)(?:\s*\||$)',
            title
        )
        if not players_m:
            return None

        player1 = players_m.group(1).strip()
        player2 = players_m.group(2).strip()

        # Remove "OddsPortal" suffix if present
        player2 = re.sub(r'\s*\|?\s*OddsPortal.*$', '', player2).strip()

        if len(player1) < 3 or len(player2) < 3:
            return None

        # Extract score from body text
        # Look for patterns like "2:1", "3:0" (sets), or "Final result: X:Y"
        score_m = re.search(r'(?:Final\s+result:?\s*)?(\d+)\s*:\s*(\d+)', body_text[:2000])
        if not score_m:
            return None

        s1, s2 = int(score_m.group(1)), int(score_m.group(2))

        # Determine winner
        if s1 > s2:
            winner, loser = player1, player2
        else:
            winner, loser = player2, player1

        # Extract odds - try multiple strategies
        odds1, odds2 = None, None

        # Strategy 1: Look for data-odd attributes
        odds_matches = re.findall(r'data-odd="(\d+\.\d+)"', html)
        if len(odds_matches) >= 2:
            odds1 = float(odds_matches[0])
            odds2 = float(odds_matches[1])

        # Strategy 2: Look for odds in visible elements
        if odds1 is None:
            try:
                odds_els = driver.find_elements(By.CSS_SELECTOR, "[class*='odds-val'], [class*='oddsCell'], [data-odd]")
                odds_vals = []
                for el in odds_els:
                    text = el.text.strip()
                    data = el.get_attribute("data-odd")
                    try:
                        v = float(data or text)
                        if 1.01 < v < 100.0:
                            odds_vals.append(v)
                    except (ValueError, TypeError):
                        pass
                if len(odds_vals) >= 2:
                    odds1 = odds_vals[0]
                    odds2 = odds_vals[1]
            except Exception:
                pass

        # Strategy 3: Parse from body text
        if odds1 is None:
            odds_text_m = re.findall(r'(\d+\.\d{2})', body_text[:5000])
            valid_odds = [float(o) for o in odds_text_m if 1.01 < float(o) < 100.0]
            if len(valid_odds) >= 2:
                odds1 = valid_odds[0]
                odds2 = valid_odds[1]

        # Assign odds to winner/loser
        if odds1 is not None and odds2 is not None:
            if s1 > s2:
                odds_w, odds_l = odds1, odds2
            else:
                odds_w, odds_l = odds2, odds1
        else:
            odds_w, odds_l = None, None

        # Extract date
        date_str = ""
        date_m = re.search(r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{4})', body_text[:3000])
        if date_m:
            months = {"jan": "01", "feb": "02", "mar": "03", "apr": "04",
                      "may": "05", "jun": "06", "jul": "07", "aug": "08",
                      "sep": "09", "oct": "10", "nov": "11", "dec": "12"}
            month = months.get(date_m.group(2)[:3].lower(), "")
            if month:
                date_str = f"{date_m.group(3)}-{month}-{int(date_m.group(1)):02d}"

        if not date_str:
            date_m2 = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', body_text[:3000])
            if date_m2:
                date_str = f"{date_m2.group(3)}-{date_m2.group(2)}-{date_m2.group(1)}"

        return {
            "Date": date_str,
            "Winner": winner,
            "Loser": loser,
            "odds_w": odds_w,
            "odds_l": odds_l,
            "source": "oddsportal",
        }

    except Exception as e:
        return None


def main():
    driver = get_driver()
    all_data = []

    # Define target tournaments with multiple URL patterns to try
    targets = [
        # Davis Cup World Group
        *[
            (f"Davis Cup WG {year}", "atp", "davis_cup",
             [f"https://www.oddsportal.com/tennis/world/atp-davis-cup-world-group-{year}/results/",
              f"https://www.oddsportal.com/tennis/world/atp-davis-cup-{year}/results/"])
            for year in [2022, 2023, 2024]
        ],
        # United Cup
        *[
            (f"United Cup {year}", "mixed", "united_cup",
             [f"https://www.oddsportal.com/tennis/world/atp-united-cup-{year}/results/",
              f"https://www.oddsportal.com/tennis/world/wta-united-cup-{year}/results/"])
            for year in [2023, 2024]
        ],
        # Olympics 2024
        ("Olympics ATP 2024", "atp", "olympics",
         ["https://www.oddsportal.com/tennis/world/atp-olympic-games-2024/results/",
          "https://www.oddsportal.com/tennis/france/atp-olympic-games-2024/results/"]),
        ("Olympics WTA 2024", "wta", "olympics",
         ["https://www.oddsportal.com/tennis/world/wta-olympic-games-2024/results/",
          "https://www.oddsportal.com/tennis/france/wta-olympic-games-2024/results/"]),
        # Guadalajara WTA 1000 Feb 2022
        ("Guadalajara WTA 1000 2022", "wta", "guadalajara",
         ["https://www.oddsportal.com/tennis/mexico/wta-guadalajara-2022/results/",
          "https://www.oddsportal.com/tennis/mexico/wta-guadalajara/results/"]),
        # Metz 2025
        ("Metz 2025", "atp", "main_tour",
         ["https://www.oddsportal.com/tennis/france/atp-metz-2025/results/",
          "https://www.oddsportal.com/tennis/france/atp-moselle-open-2025/results/"]),
        # ATP Finals 2025
        ("ATP Finals 2025", "atp", "main_tour",
         ["https://www.oddsportal.com/tennis/italy/atp-nitto-atp-finals-2025/results/",
          "https://www.oddsportal.com/tennis/italy/atp-tour-finals-2025/results/",
          "https://www.oddsportal.com/tennis/world/atp-nitto-atp-finals-2025/results/"]),
    ]

    try:
        for tournament, tour, event_type, urls in targets:
            print(f"\n{'='*60}")
            print(f"  {tournament}")
            print(f"{'='*60}")

            match_links = []
            for url in urls:
                print(f"  Trying: {url}")
                links = extract_match_links_from_page(driver, url)
                if links:
                    match_links.extend(links)
                    print(f"    Found {len(links)} match links")
                    break
                else:
                    print(f"    No match links found")
                time.sleep(2)

            if not match_links:
                print(f"  No match pages found for {tournament}")
                continue

            # Visit each match page
            match_links = list(set(match_links))[:150]  # Cap at 150
            for i, link in enumerate(match_links):
                result = scrape_match_detail(driver, link)
                if result:
                    result["tournament"] = tournament
                    result["tour"] = tour
                    result["event_type"] = event_type
                    all_data.append(result)

                if (i + 1) % 10 == 0:
                    n_with_odds = sum(1 for r in all_data if r.get("odds_w"))
                    print(f"    Processed {i+1}/{len(match_links)} ({len(all_data)} total, {n_with_odds} with odds)")

                time.sleep(1.5)

    finally:
        driver.quit()

    if not all_data:
        print("\n*** NO DATA SCRAPED ***")
        return

    df = pd.DataFrame(all_data)
    df = df.drop_duplicates(subset=["Winner", "Loser", "Date"], keep="first")

    print(f"\n{'='*60}")
    print(f"TOTAL: {len(df)} matches")
    with_odds = df["odds_w"].notna().sum()
    print(f"With odds: {with_odds}")
    print("\nBy event type:")
    print(df.groupby("event_type").size().to_string())

    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
