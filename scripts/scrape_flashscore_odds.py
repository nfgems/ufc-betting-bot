"""
Scrape Flashscore.com for individual match odds in team events + Olympics.
Uses Selenium to handle JS-rendered content.

Flashscore has individual singles results for Davis Cup, BJK Cup, United Cup, ATP Cup.
It loads data from its internal API which we capture via page rendering.

Output: data/raw/tennis/flashscore_odds.csv
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
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "raw" / "tennis" / "flashscore_odds.csv"


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


def dismiss_cookie_banner(driver):
    """Dismiss Flashscore cookie consent banner."""
    try:
        btn = driver.find_element(By.ID, "onetrust-accept-btn-handler")
        btn.click()
        time.sleep(0.5)
    except (NoSuchElementException, Exception):
        pass


def click_show_more(driver, max_clicks=10):
    """Click 'Show more matches' button until all results are loaded."""
    for _ in range(max_clicks):
        try:
            show_more = driver.find_element(By.CSS_SELECTOR, "a.event__more, [class*='showMore'], [class*='event__more']")
            if show_more.is_displayed():
                driver.execute_script("arguments[0].click();", show_more)
                time.sleep(1.5)
            else:
                break
        except (NoSuchElementException, StaleElementReferenceException):
            break


def parse_flashscore_results(driver, url, tournament, tour, event_type):
    """Parse match results from a Flashscore results page."""
    rows = []
    try:
        driver.get(url)
        time.sleep(3)
        dismiss_cookie_banner(driver)
        time.sleep(1)

        # Click show more to load all results
        click_show_more(driver, max_clicks=15)
        time.sleep(1)

        # Get all event rows
        events = driver.find_elements(By.CSS_SELECTOR, "[class*='event__match'], .event__match")
        if not events:
            # Try alternative selectors
            events = driver.find_elements(By.CSS_SELECTOR, "div[id^='g_1_']")

        print(f"    Found {len(events)} event elements")

        current_date = ""
        for event in events:
            try:
                text = event.text.strip()
                if not text:
                    continue

                # Check if this is a date header
                date_match = re.match(r'(\d{2})\.(\d{2})\.(\d{4})', text)
                if date_match:
                    current_date = f"{date_match.group(3)}-{date_match.group(2)}-{date_match.group(1)}"
                    continue

                # Parse match row
                # Flashscore format varies:
                # "HH:MM  Player1 I.  Score  Player2 J.  Odds1  Odds2"
                # or just "Player1 I.  2  1  Player2 J."
                lines = text.split("\n")
                if len(lines) < 2:
                    continue

                # Try to extract player names
                player1 = ""
                player2 = ""
                score1 = ""
                score2 = ""
                odds1 = None
                odds2 = None

                # Strategy: look for participant elements within this event
                try:
                    participants = event.find_elements(By.CSS_SELECTOR,
                        "[class*='participant'], [class*='homeParticipant'], [class*='awayParticipant']")
                    if len(participants) >= 2:
                        player1 = participants[0].text.strip()
                        player2 = participants[1].text.strip()
                except Exception:
                    pass

                if not player1 or not player2:
                    # Fallback: parse from text lines
                    # Typical Flashscore text layout:
                    # Line 0: time or empty
                    # Line 1: player1
                    # Line 2: player2
                    # Lines 3+: scores and odds
                    for line in lines:
                        line = line.strip()
                        # Skip time strings and empty lines
                        if re.match(r'^\d{2}:\d{2}$', line) or not line:
                            continue
                        if re.match(r'^\d+$', line):  # Score number
                            if not score1:
                                score1 = line
                            else:
                                score2 = line
                            continue
                        if re.match(r'^\d+\.\d+$', line):  # Odds
                            if odds1 is None:
                                odds1 = float(line)
                            else:
                                odds2 = float(line)
                            continue
                        # Must be a player name
                        if not player1:
                            player1 = line
                        elif not player2:
                            player2 = line

                if not player1 or not player2:
                    continue

                # Try to get score elements
                try:
                    score_els = event.find_elements(By.CSS_SELECTOR,
                        "[class*='score'], [class*='event__score']")
                    if score_els:
                        score_text = score_els[0].text.strip()
                        sm = re.match(r'(\d+)\s*[-–:]\s*(\d+)', score_text)
                        if sm:
                            score1, score2 = sm.group(1), sm.group(2)
                except Exception:
                    pass

                # Try to get odds elements
                try:
                    odds_els = event.find_elements(By.CSS_SELECTOR,
                        "[class*='odds'], [class*='event__odds']")
                    if len(odds_els) >= 2:
                        try:
                            odds1 = float(odds_els[0].text.strip())
                        except ValueError:
                            pass
                        try:
                            odds2 = float(odds_els[1].text.strip())
                        except ValueError:
                            pass
                except Exception:
                    pass

                # Determine winner
                if score1 and score2:
                    try:
                        s1, s2 = int(score1), int(score2)
                        if s1 > s2:
                            winner, loser = player1, player2
                            odds_w = odds1
                            odds_l = odds2
                        else:
                            winner, loser = player2, player1
                            odds_w = odds2
                            odds_l = odds1
                    except ValueError:
                        winner, loser = player1, player2
                        odds_w, odds_l = odds1, odds2
                else:
                    winner, loser = player1, player2
                    odds_w, odds_l = odds1, odds2

                # Get date from event if available
                try:
                    event_id = event.get_attribute("id") or ""
                    # Flashscore stores date in data attributes sometimes
                    date_el = event.find_element(By.CSS_SELECTOR, "[class*='event__time']")
                    date_text = date_el.text.strip()
                    dm = re.match(r'(\d{2})\.(\d{2})\.(\d{4})', date_text)
                    if dm:
                        current_date = f"{dm.group(3)}-{dm.group(2)}-{dm.group(1)}"
                    else:
                        dm2 = re.match(r'(\d{2})\.(\d{2})\.\s*(\d{2}:\d{2})', date_text)
                        if dm2:
                            # Only month.day, need year from URL
                            year_match = re.search(r'(\d{4})', url)
                            year = year_match.group(1) if year_match else "2024"
                            current_date = f"{year}-{dm2.group(2)}-{dm2.group(1)}"
                except Exception:
                    pass

                if winner and loser and len(winner) > 2 and len(loser) > 2:
                    rows.append({
                        "Date": current_date,
                        "Winner": winner,
                        "Loser": loser,
                        "odds_w": odds_w,
                        "odds_l": odds_l,
                        "tournament": tournament,
                        "tour": tour,
                        "event_type": event_type,
                        "source": "flashscore",
                    })

            except StaleElementReferenceException:
                continue
            except Exception as e:
                continue

    except Exception as e:
        print(f"    Error: {e}")

    return rows


def extract_match_details_from_body_text(body_text):
    lines = [line.strip() for line in body_text.splitlines() if line.strip()]
    if not lines:
        return None

    date_str = ""
    player1 = ""
    player2 = ""
    score1 = None
    score2 = None

    for idx, line in enumerate(lines):
        date_match = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{4})\s+\d{2}:\d{2}", line)
        if not date_match:
            continue

        date_str = f"{date_match.group(3)}-{date_match.group(2)}-{date_match.group(1)}"
        for offset in range(1, 6):
            if idx + offset >= len(lines):
                break
            candidate = lines[idx + offset]
            if re.search(r"[A-Za-z]", candidate) and candidate not in {"ATP", "WTA", "MATCH", "ODDS", "H2H"}:
                player1 = candidate
                break

        for pos in range(idx + 1, min(len(lines), idx + 14)):
            if lines[pos] != "-":
                continue
            if pos - 1 < 0 or pos + 1 >= len(lines):
                continue
            if not lines[pos - 1].isdigit() or not lines[pos + 1].isdigit():
                continue
            score1 = int(lines[pos - 1])
            score2 = int(lines[pos + 1])
            for later in range(pos + 2, min(len(lines), pos + 8)):
                candidate = lines[later]
                if candidate in {"FINISHED", "ATP", "WTA"}:
                    continue
                if re.search(r"[A-Za-z]", candidate) and not re.fullmatch(r"[A-Z]{2,}", candidate):
                    player2 = candidate
                    break
            break

        if date_str:
            break

    if (not player1 or not player2) and lines:
        title_match = re.match(r"(.+?) v (.+?) \((\d{2}/\d{2}/\d{4})\)", lines[0])
        if title_match:
            player1 = player1 or title_match.group(1)
            player2 = player2 or title_match.group(2)
            if not date_str:
                month_day_year = datetime.strptime(title_match.group(3), "%d/%m/%Y")
                date_str = month_day_year.strftime("%Y-%m-%d")

    odds1 = None
    odds2 = None
    for idx, line in enumerate(lines):
        if line != "HOME/AWAY":
            continue
        decimal_values = []
        for candidate in lines[idx + 1:idx + 12]:
            if re.fullmatch(r"\d+\.\d+", candidate):
                decimal_values.append(float(candidate))
            if len(decimal_values) == 2:
                break
        if len(decimal_values) == 2:
            odds1, odds2 = decimal_values
            break

    if not date_str or not player1 or not player2 or odds1 is None or odds2 is None:
        return None

    if score1 is None or score2 is None:
        winner, loser, odds_w, odds_l = player1, player2, odds1, odds2
    elif score1 > score2:
        winner, loser, odds_w, odds_l = player1, player2, odds1, odds2
    else:
        winner, loser, odds_w, odds_l = player2, player1, odds2, odds1

    return {
        "Date": date_str,
        "Winner": winner,
        "Loser": loser,
        "odds_w": odds_w,
        "odds_l": odds_l,
    }


def scrape_flashscore_via_match_links(driver, url, tournament, tour, event_type):
    """Alternative approach: find and visit individual match detail pages."""
    rows = []
    try:
        driver.get(url)
        time.sleep(3)
        dismiss_cookie_banner(driver)
        time.sleep(1)
        click_show_more(driver, max_clicks=15)
        time.sleep(1)

        # Find all match links
        links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/match/']")
        match_urls = list(set(link.get_attribute("href") for link in links if link.get_attribute("href")))
        print(f"    Found {len(match_urls)} match detail links")

        for i, match_url in enumerate(match_urls[:200]):
            try:
                driver.get(match_url)
                time.sleep(1.5)
                body_text = driver.find_element(By.TAG_NAME, "body").text
                parsed = extract_match_details_from_body_text(body_text)
                if parsed is None:
                    continue

                parsed.update({
                    "tournament": tournament,
                    "tour": tour,
                    "event_type": event_type,
                    "source": "flashscore",
                })
                rows.append(parsed)

                if (i + 1) % 20 == 0:
                    print(f"    Processed {i+1}/{len(match_urls)} ({len(rows)} scraped)")

                time.sleep(0.5)

            except Exception:
                continue

    except Exception as e:
        print(f"    Error: {e}")

    return rows


def main():
    driver = get_driver()
    all_data = []

    try:
        filters = [arg.strip().lower() for arg in sys.argv[1:] if arg.strip()]
        year_filters = [int(arg) for arg in filters if arg.isdigit()]
        text_filters = [arg for arg in filters if not arg.isdigit()]
        # Flashscore URL patterns for individual singles results
        targets = [
            # Davis Cup individual singles
            ("davis-cup-world-group", "atp", "davis_cup", "tennis/atp-singles/davis-cup-world-group"),
            ("davis-cup-world-group-ii", "atp", "davis_cup", "tennis/atp-singles/davis-cup-world-group-ii"),

            # BJK Cup individual singles
            ("bjk-cup-world-group", "wta", "bjk_cup", "tennis/wta-singles/billie-jean-king-cup"),
            ("bjk-cup-world-group", "wta", "bjk_cup", "tennis/wta-singles/billie-jean-king-cup-world-group"),

            # United Cup individual matches
            ("united-cup-atp", "atp", "united_cup", "tennis/atp-singles/united-cup"),
            ("united-cup-wta", "wta", "united_cup", "tennis/wta-singles/united-cup"),

            # ATP Cup
            ("atp-cup", "atp", "atp_cup", "tennis/atp-singles/atp-cup"),

            # Olympics
            ("olympics-atp", "atp", "olympics", "tennis/atp-singles/olympic-games"),
            ("olympics-wta", "wta", "olympics", "tennis/wta-singles/olympic-games"),

            # Laver Cup
            ("laver-cup", "atp", "laver_cup", "tennis/atp-singles/laver-cup"),

            # NextGen
            ("nextgen-finals", "atp", "nextgen", "tennis/atp-singles/next-gen-atp-finals"),
        ]
        if text_filters:
            targets = [
                target for target in targets
                if any(filter_text in " ".join(map(str, target)).lower() for filter_text in text_filters)
            ]

        for tournament, tour, event_type, path in targets:
            # Try with year suffixes
            years_to_try = year_filters or ([2022, 2023, 2024, 2025, 2026] if event_type != "olympics" else [2024])

            for year in years_to_try:
                url = f"https://www.flashscore.com/{path}-{year}/results/"
                print(f"\n  {tournament} {year}...", end=" ", flush=True)

                rows = parse_flashscore_results(driver, url, f"{tournament}-{year}", tour, event_type)

                if not rows or not any(
                    row.get("odds_w") is not None and row.get("odds_l") is not None
                    for row in rows
                ):
                    # Try match detail pages approach
                    rows = scrape_flashscore_via_match_links(driver, url, f"{tournament}-{year}", tour, event_type)

                all_data.extend(rows)
                n_with_odds = sum(1 for r in rows if r.get("odds_w"))
                print(f"{len(rows)} matches ({n_with_odds} with odds)")
                time.sleep(2)

            # Also try without year suffix (might show all years)
            url = f"https://www.flashscore.com/{path}/results/"
            print(f"\n  {tournament} (all years)...", end=" ", flush=True)
            rows = parse_flashscore_results(driver, url, tournament, tour, event_type)
            all_data.extend(rows)
            print(f"{len(rows)} matches")
            time.sleep(2)

        # ============================================================
        # GUADALAJARA WTA 1000 2022
        # ============================================================
        if not text_filters or any("guadalajara" in filter_text for filter_text in text_filters):
            print("\n" + "=" * 70)
            print("GUADALAJARA WTA 1000 2022")
            print("=" * 70)

            guad_paths = [
                "tennis/wta-singles/guadalajara-2022",
                "tennis/wta-singles/wta-1000-guadalajara-2022",
                "tennis/wta-singles/guadalajara",
            ]
            for path in guad_paths:
                url = f"https://www.flashscore.com/{path}/results/"
                print(f"  {path}...", end=" ", flush=True)
                rows = parse_flashscore_results(driver, url, "Guadalajara WTA 1000", "wta", "guadalajara")
                all_data.extend(rows)
                print(f"{len(rows)} matches")
                time.sleep(2)

        # ============================================================
        # METZ ATP 2025
        # ============================================================
        if not text_filters or any(filter_text in {"metz", "atp finals", "finals"} for filter_text in text_filters):
            print("\n" + "=" * 70)
            print("METZ ATP 2025 + ATP FINALS 2025")
            print("=" * 70)

            for path in ["tennis/atp-singles/metz-2025", "tennis/atp-singles/metz"]:
                url = f"https://www.flashscore.com/{path}/results/"
                print(f"  {path}...", end=" ", flush=True)
                rows = parse_flashscore_results(driver, url, "Metz", "atp", "main_tour")
                all_data.extend(rows)
                print(f"{len(rows)} matches")
                time.sleep(2)

            for path in ["tennis/atp-singles/nitto-atp-finals-2025", "tennis/atp-singles/atp-finals-2025"]:
                url = f"https://www.flashscore.com/{path}/results/"
                print(f"  {path}...", end=" ", flush=True)
                rows = parse_flashscore_results(driver, url, "ATP Finals", "atp", "main_tour")
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
        return

    df = pd.DataFrame(all_data)
    df = df.drop_duplicates(subset=["Winner", "Loser", "Date", "tournament"], keep="first")

    print(f"\n{'=' * 70}")
    print(f"TOTAL SCRAPED: {len(df)} matches")
    with_odds = (df["odds_w"].notna() & df["odds_l"].notna()).sum()
    print(f"With odds: {with_odds}")
    print("\nBy event type:")
    print(df.groupby("event_type").size().to_string())
    print("\nBy tournament:")
    print(df.groupby("tournament").size().sort_values(ascending=False).head(20).to_string())

    if OUTPUT.exists():
        existing = pd.read_csv(OUTPUT, low_memory=False)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["Winner", "Loser", "Date", "tournament"], keep="last")

    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
