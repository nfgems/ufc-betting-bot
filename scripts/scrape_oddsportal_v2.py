"""
Scrape OddsPortal.com individual match pages for team event + missing tournament odds.

Strategy:
1. Navigate to tournament results pages, scroll to load all matches
2. Extract individual match page URLs
3. Visit each match page to parse odds (American format -> decimal)

Output: data/raw/tennis/oddsportal_team_events_odds.csv
"""

import csv
import re
import sys
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
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


def american_to_decimal(american_str):
    """Convert American odds string to decimal odds."""
    try:
        val = int(american_str.replace("+", "").replace(" ", ""))
        if val > 0:
            return round(1 + val / 100, 2)
        elif val < 0:
            return round(1 + 100 / abs(val), 2)
    except (ValueError, ZeroDivisionError):
        pass
    # Try parsing as decimal directly
    try:
        f = float(american_str)
        if 1.0 < f < 100.0:
            return f
    except ValueError:
        pass
    return None


def extract_match_urls(driver, results_url):
    """Extract individual match page URLs from a results page."""
    try:
        driver.get(results_url)
        time.sleep(4)

        # Scroll to load all results
        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1)

        # Extract match URLs - they have the pattern: player-firstname-player2-firstname-HASHID
        links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/tennis/']")
        match_urls = set()
        for link in links:
            href = link.get_attribute("href") or ""
            # Match pages have player names and a hash ID
            if re.search(r'/[a-z]+-[a-z]+-[a-z]+-[a-z]+-[A-Za-z0-9]{6,}/$', href):
                match_urls.add(href)

        return list(match_urls)

    except Exception as e:
        print(f"    Error: {e}")
        return []


def parse_match_page(driver, match_url):
    """Parse a single match detail page for odds."""
    try:
        driver.get(match_url)
        time.sleep(3)

        body = driver.find_element(By.TAG_NAME, "body")
        text = body.text

        # Extract player names from title
        title = driver.title
        players_m = re.match(
            r'(.+?)\s*[-–]\s*(.+?)(?:\s+Odds|\s+\|)',
            title
        )
        if not players_m:
            return None

        player1 = players_m.group(1).strip()
        player2 = players_m.group(2).strip()

        if len(player1) < 3 or len(player2) < 3:
            return None

        # Extract date
        date_str = ""
        # Pattern: "Sunday, 27 Nov 2022, 09:10"
        date_m = re.search(
            r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s*'
            r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{4})',
            text
        )
        if date_m:
            months = {"jan": "01", "feb": "02", "mar": "03", "apr": "04",
                      "may": "05", "jun": "06", "jul": "07", "aug": "08",
                      "sep": "09", "oct": "10", "nov": "11", "dec": "12"}
            month = months.get(date_m.group(2)[:3].lower(), "")
            if month:
                date_str = f"{date_m.group(3)}-{month}-{int(date_m.group(1)):02d}"

        if not date_str:
            return None

        # Extract score
        score_m = re.search(r'Final\s+result\s*(\d+)\s*:\s*(\d+)', text)
        if not score_m:
            score_m = re.search(r'(\d+)\s*:\s*(\d+)\s*\(', text[:1000])
        if not score_m:
            return None

        s1, s2 = int(score_m.group(1)), int(score_m.group(2))

        # Determine winner
        if s1 > s2:
            winner, loser = player1, player2
        elif s2 > s1:
            winner, loser = player2, player1
        else:
            return None  # Can't determine winner

        # Extract odds from visible text
        # Look for patterns like "-250\n+175" or "1.40\n2.75"
        # OddsPortal shows odds from multiple bookmakers
        # Pattern in text: "bet365.us\nCLAIM BONUS\n-250\n+175\n-"
        odds1, odds2 = None, None

        # Try to find odds after bookmaker names
        # Look for consecutive number patterns that look like odds
        lines = text.split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            # Detect bookmaker row: look for odds patterns after bookmaker names
            if re.match(r'^[+-]?\d+$', line) and odds1 is None:
                # This could be an American odds value
                next_val = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if re.match(r'^[+-]?\d+$', next_val):
                    odds1 = american_to_decimal(line)
                    odds2 = american_to_decimal(next_val)
                    break
            # Also check for decimal odds format
            if re.match(r'^\d+\.\d{2}$', line) and odds1 is None:
                next_val = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if re.match(r'^\d+\.\d{2}$', next_val):
                    v1, v2 = float(line), float(next_val)
                    if 1.01 < v1 < 100 and 1.01 < v2 < 100:
                        odds1 = v1
                        odds2 = v2
                        break

        # Also search the full text for "Pinnacle" or average odds
        # Look for the "Average Odds" or specific bookmaker
        avg_m = re.search(r'Average[^0-9]*?(\d+\.\d+)\s+(\d+\.\d+)', text)
        if avg_m and odds1 is None:
            odds1 = float(avg_m.group(1))
            odds2 = float(avg_m.group(2))

        # Assign to winner/loser
        if odds1 is not None and odds2 is not None:
            # odds1 = player1 odds, odds2 = player2 odds
            if s1 > s2:  # player1 won
                odds_w, odds_l = odds1, odds2
            else:
                odds_w, odds_l = odds2, odds1
        else:
            odds_w, odds_l = None, None

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
    # Define ALL tournament URLs to scrape
    targets = []

    # Davis Cup - all groups and years
    for year in [2022, 2023, 2024, 2025]:
        for slug in [
            "atp-davis-cup-world-group",
            "atp-davis-cup-world-group-ii",
            "atp-davis-cup-group-i",
            "atp-davis-cup-group-ii",
            "atp-davis-cup-group-iii",
            "atp-davis-cup-group-iv",
            "atp-davis-cup-group-v",
        ]:
            targets.append((
                f"Davis Cup {slug.replace('atp-davis-cup-', '')} {year}",
                "atp", "davis_cup",
                f"https://www.oddsportal.com/tennis/world/{slug}-{year}/results/"
            ))

    # BJK Cup
    for year in [2022, 2023, 2024]:
        for slug in [
            "wta-billie-jean-king-cup",
            "wta-billie-jean-king-cup-world-group",
            "wta-billie-jean-king-cup-group-i",
            "wta-billie-jean-king-cup-group-ii",
        ]:
            targets.append((
                f"BJK Cup {slug.replace('wta-billie-jean-king-cup', 'wg').replace('-', ' ')} {year}",
                "wta", "bjk_cup",
                f"https://www.oddsportal.com/tennis/world/{slug}-{year}/results/"
            ))

    # United Cup
    for year in [2023, 2024]:
        for prefix in ["atp", "wta"]:
            targets.append((
                f"United Cup {prefix} {year}",
                prefix, "united_cup",
                f"https://www.oddsportal.com/tennis/world/{prefix}-united-cup-{year}/results/"
            ))

    # ATP Cup 2022
    targets.append(("ATP Cup 2022", "atp", "atp_cup",
                     "https://www.oddsportal.com/tennis/world/atp-cup-2022/results/"))

    # Olympics 2024
    targets.append(("Olympics ATP 2024", "atp", "olympics",
                     "https://www.oddsportal.com/tennis/world/atp-olympic-games-2024/results/"))
    targets.append(("Olympics WTA 2024", "wta", "olympics",
                     "https://www.oddsportal.com/tennis/world/wta-olympic-games-2024/results/"))

    # Guadalajara WTA 1000 Feb 2022
    targets.append(("Guadalajara WTA 1000", "wta", "guadalajara",
                     "https://www.oddsportal.com/tennis/mexico/wta-guadalajara-2022/results/"))

    # Metz 2025
    targets.append(("Metz 2025", "atp", "main_tour",
                     "https://www.oddsportal.com/tennis/france/atp-metz-2025/results/"))

    # ATP Finals 2025
    targets.append(("ATP Finals 2025", "atp", "main_tour",
                     "https://www.oddsportal.com/tennis/italy/atp-nitto-atp-finals-2025/results/"))

    # NextGen Finals
    for year in [2022, 2023, 2024]:
        targets.append((
            f"NextGen {year}", "atp", "nextgen",
            f"https://www.oddsportal.com/tennis/world/atp-next-gen-atp-finals-{year}/results/"
        ))

    # Laver Cup
    for year in [2022, 2023, 2024]:
        targets.append((
            f"Laver Cup {year}", "atp", "laver_cup",
            f"https://www.oddsportal.com/tennis/world/laver-cup-{year}/results/"
        ))

    driver = get_driver()
    all_data = []
    total_match_urls = 0

    try:
        for tournament, tour, event_type, results_url in targets:
            print(f"\n  {tournament}...", end=" ", flush=True)

            match_urls = extract_match_urls(driver, results_url)
            if not match_urls:
                print("0 match URLs")
                continue

            total_match_urls += len(match_urls)
            print(f"{len(match_urls)} match URLs", flush=True)

            tournament_matches = 0
            tournament_with_odds = 0

            for match_url in match_urls:
                result = parse_match_page(driver, match_url)
                if result:
                    result["tournament"] = tournament
                    result["tour"] = tour
                    result["event_type"] = event_type
                    all_data.append(result)
                    tournament_matches += 1
                    if result.get("odds_w"):
                        tournament_with_odds += 1

                time.sleep(1)

            print(f"    -> {tournament_matches} parsed ({tournament_with_odds} with odds)")

    finally:
        driver.quit()

    print(f"\n{'='*60}")
    print(f"Total match URLs found: {total_match_urls}")
    print(f"Total parsed: {len(all_data)}")

    if not all_data:
        print("*** NO DATA SCRAPED ***")
        return

    df = pd.DataFrame(all_data)
    df = df.drop_duplicates(subset=["Winner", "Loser", "Date"], keep="first")

    with_odds = df["odds_w"].notna().sum()
    print(f"With odds: {with_odds}")
    print(f"\nBy event type:")
    print(df.groupby("event_type").size().to_string())
    print(f"\nBy tournament:")
    print(df.groupby("tournament").size().sort_values(ascending=False).to_string())

    # Filter to 2022+
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"] >= "2022-01-01"]
    print(f"\nAfter date filter: {len(df)}")

    df.to_csv(OUTPUT, index=False)
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
