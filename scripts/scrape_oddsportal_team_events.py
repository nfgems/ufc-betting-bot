"""
Scrape OddsPortal.com for individual match odds in team events + Olympics + missing tournaments.
Uses Selenium to handle JS-rendered content.

Targets:
- Davis Cup (all groups) individual rubber odds
- BJK Cup individual rubber odds
- United Cup individual match odds
- ATP Cup individual match odds
- Olympics tennis individual match odds
- Guadalajara WTA 1000 2022
- Metz ATP 2025
- ATP Finals 2025
- NextGen Finals

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

YEARS = [2022, 2023, 2024, 2025, 2026]
BODY_STOP_MARKERS = {"MY COUPON", "TOP EVENTS", "ODDSPORTAL"}


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
    # Disable automation detection
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })
    driver.set_page_load_timeout(45)
    return driver


def parse_oddsportal_results(driver, url, tournament, tour, event_type, max_pages=5):
    """Parse match results from an OddsPortal results page."""
    all_rows = []

    for page in range(1, max_pages + 1):
        page_url = url if page == 1 else f"{url}#/page/{page}/"
        try:
            driver.get(page_url)
            time.sleep(3)  # Wait for JS rendering

            # Try to wait for match rows to appear
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "[class*='eventRow'], [class*='match-row'], .table-main tr, a[href*='/tennis/']"))
                )
            except TimeoutException:
                if page == 1:
                    print(f"    No data found at {url}")
                break

            # Get page source and parse
            html = driver.page_source

            # Try multiple parsing strategies for OddsPortal's dynamic content
            rows = _parse_oddsportal_html(html, tournament, tour, event_type)

            if not rows:
                # Try extracting from visible text as fallback
                rows = _parse_from_visible_text(driver, tournament, tour, event_type)

            if not rows and page > 1:
                break

            all_rows.extend(rows)

            if page < max_pages:
                time.sleep(1.5)

        except Exception as e:
            print(f"    Error on page {page}: {e}")
            break

    return all_rows


def _parse_oddsportal_html(html, tournament, tour, event_type):
    """Parse OddsPortal HTML for match data."""
    rows = []

    # Pattern 1: Look for match data in script tags (OddsPortal stores data in JSON)
    json_matches = re.findall(r'"participants":\s*\[([^\]]+)\]', html)

    # Pattern 2: Look for player names in typical OddsPortal format
    # OddsPortal format: "Player1 I. - Player2 J." with odds columns
    match_pattern = re.compile(
        r'(?:class="[^"]*participant[^"]*"[^>]*>|<a[^>]*>)\s*'
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+[A-Z]\.(?:\s*[A-Z]\.)*)\s*'
        r'(?:</a>|</span>|</div>)\s*-\s*'
        r'(?:<[^>]*>)?\s*'
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+[A-Z]\.(?:\s*[A-Z]\.)*)',
        re.IGNORECASE
    )

    # Pattern 3: Look for links to individual match pages
    match_links = re.findall(
        r'href="(/tennis/[^"]*?/[^"]*?-vs-[^"]*?/[^"]*?)"',
        html
    )

    # Pattern 4: New OddsPortal React-based layout - look for data in JSON props
    json_data_pattern = re.findall(r'__NEXT_DATA__.*?({.*?})</script>', html, re.DOTALL)
    if json_data_pattern:
        try:
            data = json.loads(json_data_pattern[0])
            rows.extend(_parse_next_data(data, tournament, tour, event_type))
        except (json.JSONDecodeError, KeyError):
            pass

    return rows


def _parse_next_data(data, tournament, tour, event_type):
    """Parse __NEXT_DATA__ JSON from OddsPortal."""
    rows = []
    try:
        # Navigate the nested structure
        props = data.get("props", {}).get("pageProps", {})
        events = props.get("events", [])
        if not events:
            # Try alternative paths
            for key in props:
                if isinstance(props[key], list):
                    events = props[key]
                    break

        for event in events:
            if not isinstance(event, dict):
                continue

            # Extract participant names
            participants = event.get("participants", [])
            if len(participants) < 2:
                home = event.get("home", "")
                away = event.get("away", "")
            else:
                home = participants[0].get("name", "") if isinstance(participants[0], dict) else str(participants[0])
                away = participants[1].get("name", "") if isinstance(participants[1], dict) else str(participants[1])

            if not home or not away:
                continue

            # Extract date
            timestamp = event.get("dateStart", event.get("timestamp", 0))
            if isinstance(timestamp, (int, float)) and timestamp > 1000000000:
                date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
            else:
                date_str = str(event.get("date", ""))

            # Extract odds
            odds = event.get("odds", {})
            odds_home = odds.get("home", None)
            odds_away = odds.get("away", None)

            # Determine winner from score
            score = event.get("score", "")
            winner, loser = home, away
            odds_w, odds_l = odds_home, odds_away

            score_match = re.match(r"(\d+):(\d+)", str(score))
            if score_match:
                s1, s2 = int(score_match.group(1)), int(score_match.group(2))
                if s2 > s1:
                    winner, loser = away, home
                    odds_w, odds_l = odds_away, odds_home

            rows.append({
                "Date": date_str,
                "Winner": winner,
                "Loser": loser,
                "odds_w": odds_w,
                "odds_l": odds_l,
                "tournament": tournament,
                "tour": tour,
                "event_type": event_type,
                "source": "oddsportal",
            })
    except Exception as e:
        print(f"    Error parsing NEXT_DATA: {e}")

    return rows


def _oddsportal_to_decimal(odd_text):
    """Convert OddsPortal American or decimal odds text to decimal odds."""
    text = str(odd_text).strip().replace("−", "-")
    if not text:
        return None
    if re.fullmatch(r"[+-]\d+", text):
        val = int(text)
        if val > 0:
            return round(1.0 + (val / 100.0), 3)
        return round(1.0 + (100.0 / abs(val)), 3)
    try:
        val = float(text)
        if 1.0 < val < 100.0:
            return val
    except ValueError:
        pass
    return None


def _parse_body_block(block_lines, current_date, tournament, tour, event_type):
    """Parse a single body-text block from OddsPortal's rendered results page."""
    if len(block_lines) < 5:
        return None

    odds1 = _oddsportal_to_decimal(block_lines[-2])
    odds2 = _oddsportal_to_decimal(block_lines[-1])
    if odds1 is None or odds2 is None:
        return None

    core = [line for line in block_lines[:-2] if line not in {"-", "–"}]
    if len(core) < 4:
        return None

    player1 = core[0]
    numeric_positions = [(idx, token) for idx, token in enumerate(core[1:], start=1) if token.isdigit()]
    if len(numeric_positions) < 2:
        return None

    s1_idx, s1_text = numeric_positions[0]
    s2_idx, s2_text = numeric_positions[1]
    player2_candidates = [token for token in core[s2_idx + 1:] if re.search(r"[A-Za-z]", token)]
    if not player2_candidates:
        return None

    player2 = player2_candidates[0]
    s1 = int(s1_text)
    s2 = int(s2_text)
    if s1 == s2:
        return None

    if s1 > s2:
        winner, loser = player1, player2
        odds_w, odds_l = odds1, odds2
    else:
        winner, loser = player2, player1
        odds_w, odds_l = odds2, odds1

    return {
        "Date": current_date,
        "Winner": winner,
        "Loser": loser,
        "odds_w": odds_w,
        "odds_l": odds_l,
        "tournament": tournament,
        "tour": tour,
        "event_type": event_type,
        "source": "oddsportal",
    }


def _parse_from_visible_text(driver, tournament, tour, event_type):
    """Fallback: parse match data from rendered body text."""
    rows = []
    try:
        body = driver.find_element(By.TAG_NAME, "body").text
        lines = [line.strip() for line in body.splitlines() if line.strip()]

        current_date = ""
        i = 0
        while i < len(lines):
            line = lines[i]

            if line in BODY_STOP_MARKERS or line.startswith('Add "'):
                break

            dm = re.match(r"(\d{2})\s+([A-Za-z]{3})\s+(\d{4})(?:\s+-\s+.*)?$", line)
            if dm:
                month_map = {
                    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
                    "may": "05", "jun": "06", "jul": "07", "aug": "08",
                    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
                }
                month = month_map.get(dm.group(2).lower())
                if month:
                    current_date = f"{dm.group(3)}-{month}-{dm.group(1)}"
                i += 1
                continue

            if current_date and re.fullmatch(r"\d{2}:\d{2}", line):
                j = i + 1
                block_lines = []
                while j < len(lines):
                    nxt = lines[j]
                    if nxt in BODY_STOP_MARKERS or nxt.startswith('Add "'):
                        break
                    if re.match(r"(\d{2})\s+([A-Za-z]{3})\s+(\d{4})(?:\s+-\s+.*)?$", nxt):
                        break
                    if re.fullmatch(r"\d{2}:\d{2}", nxt):
                        break
                    block_lines.append(nxt)
                    j += 1

                parsed = _parse_body_block(block_lines, current_date, tournament, tour, event_type)
                if parsed is not None:
                    rows.append(parsed)
                i = j
                continue

            i += 1
    except Exception as e:
        pass

    return rows


def scrape_oddsportal_match_pages(driver, base_url, tournament, tour, event_type):
    """Navigate to individual match pages on OddsPortal to get detailed odds."""
    rows = []
    try:
        driver.get(base_url)
        time.sleep(3)

        # Find all match links
        links = driver.find_elements(By.CSS_SELECTOR, "a[href*='-vs-']")
        match_urls = []
        for link in links:
            href = link.get_attribute("href")
            if href and "/tennis/" in href:
                match_urls.append(href)

        match_urls = list(set(match_urls))
        print(f"    Found {len(match_urls)} match links")

        for i, match_url in enumerate(match_urls[:100]):  # Cap at 100
            try:
                driver.get(match_url)
                time.sleep(2)

                html = driver.page_source

                # Extract player names from title or h1
                title_match = re.search(r'<title>([^<]+)</title>', html)
                if title_match:
                    title = title_match.group(1)
                    players_m = re.match(r'(.+?)\s*(?:vs\.?|[-–])\s*(.+?)(?:\s*\||\s*-\s*OddsPortal)', title)
                    if players_m:
                        player1 = players_m.group(1).strip()
                        player2 = players_m.group(2).strip()
                    else:
                        continue
                else:
                    continue

                # Extract odds
                odds_elements = driver.find_elements(By.CSS_SELECTOR, "[class*='odds'], [data-odd]")
                odds_values = []
                for el in odds_elements:
                    text = el.text.strip()
                    data_odd = el.get_attribute("data-odd")
                    try:
                        val = float(data_odd or text)
                        if 1.0 < val < 100.0:
                            odds_values.append(val)
                    except (ValueError, TypeError):
                        pass

                # Extract score and date from page
                score_m = re.search(r'(\d+)\s*:\s*(\d+)', driver.find_element(By.CSS_SELECTOR, "body").text)
                date_m = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})', driver.find_element(By.CSS_SELECTOR, "body").text)

                if score_m and len(odds_values) >= 2:
                    s1, s2 = int(score_m.group(1)), int(score_m.group(2))
                    if s1 > s2:
                        winner, loser = player1, player2
                        odds_w, odds_l = odds_values[0], odds_values[1]
                    else:
                        winner, loser = player2, player1
                        odds_w, odds_l = odds_values[1], odds_values[0]

                    date_str = ""
                    if date_m:
                        months = {"jan": "01", "feb": "02", "mar": "03", "apr": "04",
                                  "may": "05", "jun": "06", "jul": "07", "aug": "08",
                                  "sep": "09", "oct": "10", "nov": "11", "dec": "12"}
                        month = months.get(date_m.group(2)[:3].lower(), "")
                        if month:
                            date_str = f"{date_m.group(3)}-{month}-{int(date_m.group(1)):02d}"

                    rows.append({
                        "Date": date_str,
                        "Winner": winner,
                        "Loser": loser,
                        "odds_w": odds_w,
                        "odds_l": odds_l,
                        "tournament": tournament,
                        "tour": tour,
                        "event_type": event_type,
                        "source": "oddsportal",
                    })

                if (i + 1) % 10 == 0:
                    print(f"    Processed {i+1}/{len(match_urls)} matches ({len(rows)} with odds)")

                time.sleep(1)

            except Exception as e:
                continue

    except Exception as e:
        print(f"    Error: {e}")

    return rows


def main():
    driver = get_driver()
    all_data = []

    try:
        # ============================================================
        # DAVIS CUP - Individual rubbers
        # ============================================================
        print("=" * 70)
        print("DAVIS CUP - Individual Match Odds")
        print("=" * 70)

        davis_cup_slugs = [
            "atp-davis-cup-world-group",
            "atp-davis-cup-world-group-ii",
            "atp-davis-cup-group-i",
            "atp-davis-cup-group-ii",
            "atp-davis-cup-group-iii",
            "atp-davis-cup-group-iv",
            "atp-davis-cup-group-v",
            "atp-davis-cup",
        ]

        for slug in davis_cup_slugs:
            for year in YEARS:
                url = f"https://www.oddsportal.com/tennis/world/{slug}-{year}/results/"
                print(f"  {slug} {year}...", end=" ", flush=True)
                rows = parse_oddsportal_results(driver, url, f"Davis Cup {slug}", "atp", "davis_cup")
                if not rows:
                    # Try match page approach
                    rows = scrape_oddsportal_match_pages(
                        driver, url, f"Davis Cup {slug}", "atp", "davis_cup"
                    )
                all_data.extend(rows)
                n_with_odds = sum(1 for r in rows if r.get("odds_w"))
                print(f"{len(rows)} matches ({n_with_odds} with odds)")
                time.sleep(2)

        # ============================================================
        # BJK CUP - Individual rubbers
        # ============================================================
        print("\n" + "=" * 70)
        print("BJK CUP - Individual Match Odds")
        print("=" * 70)

        bjk_slugs = [
            "wta-billie-jean-king-cup",
            "wta-billie-jean-king-cup-world-group",
            "wta-billie-jean-king-cup-group-i",
            "wta-billie-jean-king-cup-group-ii",
        ]

        for slug in bjk_slugs:
            for year in YEARS:
                url = f"https://www.oddsportal.com/tennis/world/{slug}-{year}/results/"
                print(f"  {slug} {year}...", end=" ", flush=True)
                rows = parse_oddsportal_results(driver, url, f"BJK Cup {slug}", "wta", "bjk_cup")
                all_data.extend(rows)
                n_with_odds = sum(1 for r in rows if r.get("odds_w"))
                print(f"{len(rows)} matches ({n_with_odds} with odds)")
                time.sleep(2)

        # ============================================================
        # UNITED CUP
        # ============================================================
        print("\n" + "=" * 70)
        print("UNITED CUP")
        print("=" * 70)

        united_urls = [
            ("atp", "https://www.oddsportal.com/tennis/australia/atp-united-cup/results/"),
            ("wta", "https://www.oddsportal.com/tennis/australia/wta-united-cup/results/"),
        ]
        for tour_prefix, url in united_urls:
            print(f"  {tour_prefix} United Cup...", end=" ", flush=True)
            rows = parse_oddsportal_results(driver, url, "United Cup", tour_prefix, "united_cup")
            all_data.extend(rows)
            print(f"{len(rows)} matches")
            time.sleep(2)

        # ============================================================
        # ATP CUP
        # ============================================================
        print("\n" + "=" * 70)
        print("ATP CUP")
        print("=" * 70)

        atp_cup_url = "https://www.oddsportal.com/tennis/australia/atp-atp-cup/results/"
        print("  ATP Cup...", end=" ", flush=True)
        rows = parse_oddsportal_results(driver, atp_cup_url, "ATP Cup", "atp", "atp_cup")
        all_data.extend(rows)
        print(f"{len(rows)} matches")
        time.sleep(2)

        # ============================================================
        # OLYMPICS 2024
        # ============================================================
        print("\n" + "=" * 70)
        print("OLYMPICS 2024")
        print("=" * 70)

        olympics_slugs = [
            ("atp-olympic-games-2024", "atp"),
            ("wta-olympic-games-2024", "wta"),
            ("atp-olympic-games", "atp"),
            ("wta-olympic-games", "wta"),
            ("atp-olympics-2024", "atp"),
            ("wta-olympics-2024", "wta"),
        ]

        for slug, tour in olympics_slugs:
            url = f"https://www.oddsportal.com/tennis/world/{slug}/results/"
            print(f"  {slug}...", end=" ", flush=True)
            rows = parse_oddsportal_results(driver, url, "Olympics 2024", tour, "olympics")
            all_data.extend(rows)
            print(f"{len(rows)} matches")
            time.sleep(2)

        # Also try France path for Olympics (Roland Garros was the venue)
        for slug, tour in [("atp-paris-olympics-2024", "atp"), ("wta-paris-olympics-2024", "wta")]:
            url = f"https://www.oddsportal.com/tennis/france/{slug}/results/"
            print(f"  {slug} (France)...", end=" ", flush=True)
            rows = parse_oddsportal_results(driver, url, "Olympics 2024", tour, "olympics")
            all_data.extend(rows)
            print(f"{len(rows)} matches")
            time.sleep(2)

        # ============================================================
        # GUADALAJARA WTA 1000 2022
        # ============================================================
        print("\n" + "=" * 70)
        print("GUADALAJARA WTA 1000 2022")
        print("=" * 70)

        guad_slugs = [
            ("wta-guadalajara-2022", "wta"),
            ("wta-guadalajara", "wta"),
        ]

        for slug, tour in guad_slugs:
            url = f"https://www.oddsportal.com/tennis/mexico/{slug}/results/"
            print(f"  {slug}...", end=" ", flush=True)
            rows = parse_oddsportal_results(driver, url, "Guadalajara WTA 1000", tour, "guadalajara")
            if not rows:
                rows = scrape_oddsportal_match_pages(
                    driver, url, "Guadalajara WTA 1000", tour, "guadalajara"
                )
            all_data.extend(rows)
            print(f"{len(rows)} matches")
            time.sleep(2)

        # ============================================================
        # METZ ATP 2025
        # ============================================================
        print("\n" + "=" * 70)
        print("METZ ATP 2025 + ATP FINALS 2025")
        print("=" * 70)

        for slug in ["atp-metz-2025", "atp-metz"]:
            url = f"https://www.oddsportal.com/tennis/france/{slug}/results/"
            print(f"  {slug}...", end=" ", flush=True)
            rows = parse_oddsportal_results(driver, url, "Metz", "atp", "main_tour")
            all_data.extend(rows)
            print(f"{len(rows)} matches")
            time.sleep(2)

        for slug in ["atp-nitto-atp-finals-2025", "atp-nitto-finals-2025", "atp-tour-finals-2025"]:
            url = f"https://www.oddsportal.com/tennis/italy/{slug}/results/"
            print(f"  {slug}...", end=" ", flush=True)
            rows = parse_oddsportal_results(driver, url, "ATP Finals", "atp", "main_tour")
            all_data.extend(rows)
            print(f"{len(rows)} matches")
            time.sleep(2)

        # ============================================================
        # LAVER CUP
        # ============================================================
        print("\n" + "=" * 70)
        print("LAVER CUP")
        print("=" * 70)

        for year in YEARS:
            url = f"https://www.oddsportal.com/tennis/world/laver-cup-{year}/results/"
            print(f"  Laver Cup {year}...", end=" ", flush=True)
            rows = parse_oddsportal_results(driver, url, "Laver Cup", "atp", "laver_cup")
            all_data.extend(rows)
            print(f"{len(rows)} matches")
            time.sleep(2)

        # ============================================================
        # NEXTGEN FINALS
        # ============================================================
        print("\n" + "=" * 70)
        print("NEXTGEN FINALS")
        print("=" * 70)

        for year in YEARS:
            for slug in [f"atp-next-gen-finals-{year}", f"atp-next-gen-atp-finals-{year}"]:
                url = f"https://www.oddsportal.com/tennis/world/{slug}/results/"
                print(f"  {slug}...", end=" ", flush=True)
                rows = parse_oddsportal_results(driver, url, "NextGen Finals", "atp", "nextgen")
                all_data.extend(rows)
                print(f"{len(rows)} matches")
                time.sleep(2)

    finally:
        driver.quit()

    # ============================================================
    # RESULTS
    # ============================================================
    if not all_data:
        print("\n*** NO DATA SCRAPED ***")
        print("OddsPortal may be blocking automated access.")
        print("Will try Flashscore as backup...")
        return

    df = pd.DataFrame(all_data)
    print(f"\n{'=' * 70}")
    print(f"TOTAL SCRAPED: {len(df)} matches")

    # Deduplicate
    df = df.drop_duplicates(subset=["Winner", "Loser", "Date", "tournament"], keep="first")
    print(f"After dedup: {len(df)}")

    with_odds = (df["odds_w"].notna() & df["odds_l"].notna()).sum()
    print(f"With odds: {with_odds}")

    print("\nBy event type:")
    print(df.groupby("event_type").size().to_string())

    print("\nBy year:")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        valid = df[df["Date"].notna()]
        if len(valid) > 0:
            print(valid.groupby(valid["Date"].dt.year).size().to_string())

    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
