"""Scrape OddsPortal WTA 125 / Challenger Women pages via rendered body text."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scrape_oddsportal_manual_pages import get_driver, scrape_page

OUTPUT = ROOT / "data" / "raw" / "tennis" / "oddsportal_wta125_odds.csv"

TARGETS = [
    {
        "url": "https://www.oddsportal.com/tennis/australia/challenger-women-singles-canberra/results/",
        "tournament": "Canberra 125",
        "years": [2025, 2024],
    },
    {
        "url": "https://www.oddsportal.com/tennis/usa/challenger-women-singles-midland/results/",
        "tournament": "Midland 125",
        "years": [2024, 2023, 2022],
    },
    {
        "url": "https://www.oddsportal.com/tennis/turkey/challenger-women-singles-antalya/results/",
        "tournament": "Antalya 125 #1",
        "years": [2025, 2024],
    },
    {
        "url": "https://www.oddsportal.com/tennis/turkey/challenger-women-singles-antalya-2/results/",
        "tournament": "Antalya 125 #2",
        "years": [2025, 2024],
    },
    {
        "url": "https://www.oddsportal.com/tennis/turkey/challenger-women-singles-antalya-3/results/",
        "tournament": "Antalya 125 #3",
        "years": [2025, 2024],
    },
    {
        "url": "https://www.oddsportal.com/tennis/portugal/challenger-women-singles-oeiras/results/",
        "tournament": "Oeiras 125 Indoor #1",
        "years": [2025, 2024],
    },
    {
        "url": "https://www.oddsportal.com/tennis/portugal/challenger-women-singles-oeiras-2/results/",
        "tournament": "Oeiras 125 Indoor #2",
        "years": [2026, 2025, 2024],
    },
    {
        "url": "https://www.oddsportal.com/tennis/india/challenger-women-singles-mumbai/results/",
        "tournament": "Mumbai 125",
        "years": [2026, 2025],
    },
    {
        "url": "https://www.oddsportal.com/tennis/usa/challenger-women-singles-austin/results/",
        "tournament": "Austin 125 #1",
        "years": [2026, 2025],
    },
    {
        "url": "https://www.oddsportal.com/tennis/philippines/challenger-women-singles-manila/results/",
        "tournament": "Manila 125",
        "years": [2025],
    },
    {
        "url": "https://www.oddsportal.com/tennis/france/challenger-women-singles-les-sables-d-olonne/results/",
        "tournament": "Les Sables D'Olonne 125",
        "years": [2025],
    },
    {
        "url": "https://www.oddsportal.com/tennis/usa/challenger-women-singles-newport/results/",
        "tournament": "Newport 125",
        "years": [2025],
    },
    {
        "url": "https://www.oddsportal.com/tennis/spain/challenger-women-singles-san-sebastian/results/",
        "tournament": "San Sebastian 125",
        "years": [2025],
    },
    {
        "url": "https://www.oddsportal.com/tennis/united-kingdom/challenger-women-singles-birmingham/results/",
        "tournament": "Birmingham 125",
        "years": [2025],
    },
    {
        "url": "https://www.oddsportal.com/tennis/brazil/challenger-women-singles-rio-de-janeiro/results/",
        "tournament": "Rio De Janeiro 125",
        "years": [2025],
    },
    {
        "url": "https://www.oddsportal.com/tennis/brazil/challenger-women-singles-florianopolis/results/",
        "tournament": "Florianopolis 125",
        "years": [2025],
    },
    {
        "url": "https://www.oddsportal.com/tennis/argentina/challenger-women-singles-tucuman/results/",
        "tournament": "Tucuman 125",
        "years": [2025],
    },
]


def target_matches_filter(target, filters):
    if not filters:
        return True

    haystacks = [
        target["tournament"].lower(),
        target["url"].lower(),
    ]
    return any(any(filter_text in haystack for haystack in haystacks) for filter_text in filters)


def merge_rows(rows):
    new = pd.DataFrame(rows)
    old = pd.read_csv(OUTPUT) if OUTPUT.exists() else pd.DataFrame()
    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "Winner", "Loser"], keep="last")
    combined.to_csv(OUTPUT, index=False)
    return new, combined


def main():
    filters = [arg.strip().lower() for arg in sys.argv[1:] if arg.strip()]
    targets = [target for target in TARGETS if target_matches_filter(target, filters)]
    if filters and not targets:
        print(f"No targets matched filters: {filters}")
        return

    driver = get_driver()
    all_rows = []

    try:
        for target in targets:
            url = target["url"]
            tournament = target["tournament"]
            rows = scrape_page(driver, url, tournament, "wta", "125")
            print(f"{tournament} current -> {len(rows)} rows")
            all_rows.extend(rows)

            for year in target["years"]:
                rows = scrape_page(driver, url, tournament, "wta", "125", year=year)
                print(f"{tournament} {year} -> {len(rows)} rows")
                all_rows.extend(rows)
    finally:
        driver.quit()

    if not all_rows:
        print("No rows scraped.")
        return

    new, combined = merge_rows(all_rows)
    print(f"New rows: {len(new)}")
    print(f"Combined rows: {len(combined)}")
    print(new["Date"].astype(str).str[:4].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
