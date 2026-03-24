"""Scrape specific OddsPortal results pages from rendered body text.

This is a targeted helper for pages where the generic scraper is too brittle,
but the rendered text clearly contains historical match rows and odds.
"""

import re
import sys
import time
from pathlib import Path

import pandas as pd
from selenium.webdriver.common.by import By

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scrape_oddsportal_team_events import OUTPUT, get_driver, _oddsportal_to_decimal

DATE_RE = re.compile(r"^(\d{2}) ([A-Za-z]{3}) (\d{4})(?: - .*)?$")
TIME_RE = re.compile(r"^\d{2}:\d{2}$")
MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}
STOP_MARKERS = {"MY COUPON", "TOP EVENTS", "ODDSPORTAL"}


def parse_block(block_lines, current_date, tournament, tour, event_type):
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


def wait_for_results_text(driver, timeout=20, year=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            body = driver.find_element(By.TAG_NAME, "body").text
        except Exception:
            body = ""

        if body:
            has_date = bool(DATE_RE.search(body))
            if has_date and (year is None or f" {year}" in body or f"{year} " in body):
                return body
        time.sleep(1)

    return driver.find_element(By.TAG_NAME, "body").text


def click_year_link(driver, year):
    year_text = str(year)
    links = driver.find_elements(By.LINK_TEXT, year_text)
    if not links:
        return False
    driver.execute_script("arguments[0].click();", links[0])
    return True


def extract_rows_from_body(body, tournament, tour, event_type):
    lines = [line.strip() for line in body.splitlines() if line.strip()]

    # Start scanning at the first real date row on the page.
    start_idx = None
    for i, line in enumerate(lines):
        if DATE_RE.match(line):
            start_idx = i
            break
    if start_idx is None:
        return []

    rows = []
    current_date = ""
    i = start_idx
    while i < len(lines):
        line = lines[i]
        if line in STOP_MARKERS or line.startswith('Add "'):
            break

        dm = DATE_RE.match(line)
        if dm:
            month = MONTH_MAP.get(dm.group(2).lower())
            if month:
                current_date = f"{dm.group(3)}-{month}-{dm.group(1)}"
            i += 1
            continue

        if current_date and TIME_RE.fullmatch(line):
            j = i + 1
            block_lines = []
            while j < len(lines):
                nxt = lines[j]
                if nxt in STOP_MARKERS or nxt.startswith('Add "'):
                    break
                if DATE_RE.match(nxt) or TIME_RE.fullmatch(nxt):
                    break
                block_lines.append(nxt)
                j += 1

            parsed = parse_block(block_lines, current_date, tournament, tour, event_type)
            if parsed is not None:
                rows.append(parsed)
            i = j
            continue

        i += 1

    return rows


def click_page_link(driver, page_text):
    links = [link for link in driver.find_elements(By.LINK_TEXT, str(page_text)) if link.is_displayed()]
    if not links:
        return False
    driver.execute_script("arguments[0].click();", links[0])
    return True


def scrape_page(driver, url, tournament, tour, event_type, year=None, max_pages=5):
    driver.get(url)
    body = wait_for_results_text(driver, timeout=12)

    if year is not None and click_year_link(driver, year):
        body = wait_for_results_text(driver, timeout=12, year=year)

    rows = []
    seen_bodies = set()

    for page_num in range(1, max_pages + 1):
        body = wait_for_results_text(driver, timeout=12, year=year)
        body_key = hash(body)
        if body_key in seen_bodies:
            break
        seen_bodies.add(body_key)
        rows.extend(extract_rows_from_body(body, tournament, tour, event_type))

        next_page_num = page_num + 1
        if next_page_num > max_pages:
            break
        if not click_page_link(driver, next_page_num):
            break
        time.sleep(3)

    return rows


def merge_rows(rows):
    new = pd.DataFrame(rows)
    old = pd.read_csv(OUTPUT) if OUTPUT.exists() else pd.DataFrame()
    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "Winner", "Loser", "event_type"], keep="last")
    combined.to_csv(OUTPUT, index=False)
    return new, combined


def main():
    targets = [
        {
            "url": "https://www.oddsportal.com/tennis/australia/atp-atp-cup/results/",
            "tournament": "Atp Cup",
            "tour": "atp",
            "event_type": "atp_cup",
            "years": [2022],
            "include_current": False,
        },
        {
            "url": "https://www.oddsportal.com/tennis/australia/atp-united-cup/results/",
            "tournament": "United Cup",
            "tour": "atp",
            "event_type": "united_cup",
            "years": [2025, 2024, 2023],
            "include_current": True,
        },
        {
            "url": "https://www.oddsportal.com/tennis/australia/wta-united-cup/results/",
            "tournament": "United Cup",
            "tour": "wta",
            "event_type": "united_cup",
            "years": [2025, 2024, 2023],
            "include_current": True,
        },
        {
            "url": "https://www.oddsportal.com/tennis/world/wta-billie-jean-king-cup-world-group/results/",
            "tournament": "BJK WG singles",
            "tour": "wta",
            "event_type": "bjk_cup",
            "years": [2024, 2023, 2022],
            "include_current": True,
        },
        {
            "url": "https://www.oddsportal.com/tennis/world/wta-billie-jean-king-cup-group-i/results/",
            "tournament": "BJK G1 singles",
            "tour": "wta",
            "event_type": "bjk_cup",
            "years": [2025, 2024, 2023, 2022],
            "include_current": True,
        },
        {
            "url": "https://www.oddsportal.com/tennis/world/wta-billie-jean-king-cup-group-ii/results/",
            "tournament": "BJK G2 singles",
            "tour": "wta",
            "event_type": "bjk_cup",
            "years": [2025, 2024, 2023, 2022],
            "include_current": True,
        },
    ]
    if len(sys.argv) > 1:
        # Allow passing specific URLs for ad-hoc runs.
        targets = [
            {
                "url": url,
                "tournament": "manual",
                "tour": "wta",
                "event_type": "bjk_cup",
                "years": [],
                "include_current": True,
            }
            for url in sys.argv[1:]
        ]

    driver = get_driver()
    all_rows = []
    try:
        for target in targets:
            url = target["url"]
            tournament = target["tournament"]
            tour = target["tour"]
            event_type = target["event_type"]

            if target.get("include_current", True):
                rows = scrape_page(driver, url, tournament, tour, event_type)
                print(f"{url} current -> {len(rows)} rows")
                all_rows.extend(rows)

            for year in target.get("years", []):
                rows = scrape_page(driver, url, tournament, tour, event_type, year=year)
                print(f"{url} {year} -> {len(rows)} rows")
                all_rows.extend(rows)
    finally:
        driver.quit()

    if not all_rows:
        print("No rows scraped.")
        return

    new, combined = merge_rows(all_rows)
    print(f"New rows: {len(new)}")
    print(f"Combined rows: {len(combined)}")
    print(new["event_type"].value_counts().to_dict())
    print(new["Date"].astype(str).str[:4].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
