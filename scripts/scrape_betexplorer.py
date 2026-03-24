"""Scrape BetExplorer.com for tennis odds not covered by tennis-data.co.uk.

Targets: WTA 125s, Davis Cup, BJK Cup, United Cup, Olympics, Laver Cup, NextGen.
Data is extracted from results pages where odds appear in data-odd attributes.
"""

import re
import time
import sys
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Tournament categories to scrape
CATEGORIES = {
    # (base_url, tour, event_type)
    "wta_125": {
        "index_url": "https://www.betexplorer.com/tennis/challenger-women-singles/",
        "tour": "wta",
        "event_type": "125",
    },
    "davis_cup": {
        "direct_urls": [
            "https://www.betexplorer.com/tennis/teams-men/davis-cup-world-group/results/",
            "https://www.betexplorer.com/tennis/teams-men/davis-cup-group-i/results/",
            "https://www.betexplorer.com/tennis/teams-men/davis-cup-group-ii/results/",
            "https://www.betexplorer.com/tennis/teams-men/davis-cup-group-iii/results/",
        ],
        "tour": "atp",
        "event_type": "davis_cup",
    },
    "bjk_cup": {
        "direct_urls": [
            "https://www.betexplorer.com/tennis/teams-women/billie-jean-king-cup-world-group/results/",
            "https://www.betexplorer.com/tennis/teams-women/billie-jean-king-cup-group-i/results/",
            "https://www.betexplorer.com/tennis/teams-women/billie-jean-king-cup-group-ii/results/",
        ],
        "tour": "wta",
        "event_type": "bjk_cup",
    },
    "united_cup": {
        "direct_urls": [
            "https://www.betexplorer.com/tennis/teams-mix/united-cup/results/",
        ],
        "tour": "mixed",
        "event_type": "united_cup",
    },
    "laver_cup": {
        "direct_urls": [
            "https://www.betexplorer.com/tennis/teams-men/laver-cup/results/",
        ],
        "tour": "atp",
        "event_type": "laver_cup",
    },
    "olympics_atp": {
        "direct_urls": [
            "https://www.betexplorer.com/tennis/world/atp-olympic-games/results/",
        ],
        "tour": "atp",
        "event_type": "olympics",
    },
    "olympics_wta": {
        "direct_urls": [
            "https://www.betexplorer.com/tennis/world/wta-olympic-games/results/",
        ],
        "tour": "wta",
        "event_type": "olympics",
    },
    "nextgen": {
        "direct_urls": [
            "https://www.betexplorer.com/tennis/world/atp-next-gen-finals-jeddah/results/",
            "https://www.betexplorer.com/tennis/world/atp-next-gen-finals/results/",
        ],
        "tour": "atp",
        "event_type": "nextgen",
    },
}


def fetch_page(url, retries=3):
    """Fetch a page with retries."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.text
            print(f"  HTTP {r.status_code} for {url}")
        except Exception as e:
            print(f"  Error fetching {url}: {e}")
        if attempt < retries - 1:
            time.sleep(2)
    return None


def get_tournament_links(index_url):
    """Get all tournament result page URLs from an index page."""
    html = fetch_page(index_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    base_path = index_url.replace("https://www.betexplorer.com", "")

    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(base_path) and href != base_path and href.count("/") >= 4:
            links.add(href.rstrip("/"))

    return sorted(links)


def parse_results_page(html, tournament_name, tour, event_type):
    """Parse match results and odds from a BetExplorer results page."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    table = soup.find("table")
    if not table:
        return rows

    current_round = ""
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue

        # Round header row
        first_text = cells[0].get_text(strip=True)
        if cells[0].get("colspan") and first_text:
            current_round = first_text
            continue

        # Match row: players | score | odds1 | odds2 | date
        if len(cells) < 4:
            continue

        players_text = cells[0].get_text(strip=True)
        if "-" not in players_text or len(players_text) < 5:
            continue

        # Parse player names: "Player1 I.-Player2 I."
        # BetExplorer uses dash between players
        parts = players_text.split("-", 1)
        if len(parts) != 2:
            # Try splitting on last dash before a capital letter
            m = re.match(r"^(.+?\s\w\.)\s*-\s*(.+)$", players_text)
            if m:
                parts = [m.group(1), m.group(2)]
            else:
                continue

        player1 = parts[0].strip()
        player2 = parts[1].strip()

        if not player1 or not player2:
            continue

        # Score
        score_text = cells[1].get_text(strip=True) if len(cells) > 1 else ""

        # Determine winner from score
        score_match = re.match(r"(\d+):(\d+)", score_text)
        if score_match:
            sets1, sets2 = int(score_match.group(1)), int(score_match.group(2))
            winner = player1 if sets1 > sets2 else player2
            loser = player2 if sets1 > sets2 else player1
        else:
            # Can't determine winner
            continue

        # Odds - look for data-odd attributes
        odds_cells = [c for c in cells if "table-main__odds" in " ".join(c.get("class", []))]
        odds1 = None
        odds2 = None
        odds1_is_winner = False

        for oc in odds_cells:
            odd_val = oc.get("data-odd")
            is_colored = "colored" in " ".join(oc.get("class", []))

            if odd_val:
                odd_float = float(odd_val)
                if odds1 is None:
                    odds1 = odd_float
                    odds1_is_winner = is_colored
                else:
                    odds2 = odd_float
            elif is_colored and odds1 is None:
                # Winner's odds might not have data-odd in HTML (loaded by JS)
                odds1_is_winner = True

        # Date
        date_text = cells[-1].get_text(strip=True) if cells else ""
        date_match = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", date_text)
        if date_match:
            date_str = f"{date_match.group(3)}-{date_match.group(2)}-{date_match.group(1)}"
        else:
            date_str = ""

        # We have partial odds from HTML (loser's odds in data-odd)
        # The colored cell = winner but its data-odd is often missing (loaded by JS)
        # We can still capture what's available
        row = {
            "Date": date_str,
            "Winner": winner,
            "Loser": loser,
            "player1": player1,
            "player2": player2,
            "score": score_text,
            "round": current_round,
            "tournament": tournament_name,
            "tour": tour,
            "event_type": event_type,
            "source": "betexplorer",
        }

        # Assign odds: we know which cell is which player
        # Cell order: odds for player1, odds for player2
        if len(odds_cells) >= 2:
            odd1_val = odds_cells[0].get("data-odd")
            odd2_val = odds_cells[1].get("data-odd")
            is_p1_winner = (winner == player1)

            if odd1_val:
                row["odds_p1"] = float(odd1_val)
            if odd2_val:
                row["odds_p2"] = float(odd2_val)

            # Map to winner/loser odds
            if is_p1_winner:
                row["odds_w"] = row.get("odds_p1")
                row["odds_l"] = row.get("odds_p2")
            else:
                row["odds_w"] = row.get("odds_p2")
                row["odds_l"] = row.get("odds_p1")

        rows.append(row)

    return rows


def scrape_category(category_name, config):
    """Scrape all tournaments for a category."""
    print(f"\n{'='*60}")
    print(f"Scraping: {category_name}")
    print(f"{'='*60}")

    all_rows = []
    tour = config["tour"]
    event_type = config["event_type"]

    if "index_url" in config:
        # Get tournament links from index
        print(f"Fetching tournament list from {config['index_url']}...")
        tourney_paths = get_tournament_links(config["index_url"])
        print(f"Found {len(tourney_paths)} tournaments")

        for i, path in enumerate(tourney_paths):
            tourney_name = path.rstrip("/").split("/")[-1]
            results_url = f"https://www.betexplorer.com{path}/results/"

            print(f"  [{i+1}/{len(tourney_paths)}] {tourney_name}...", end=" ", flush=True)
            html = fetch_page(results_url)
            if html:
                rows = parse_results_page(html, tourney_name, tour, event_type)
                all_rows.extend(rows)
                print(f"{len(rows)} matches")
            else:
                print("FAILED")

            time.sleep(0.5)  # Polite delay

    elif "direct_urls" in config:
        for url in config["direct_urls"]:
            tourney_name = url.split("/")[-2] if url.endswith("/results/") else url.split("/")[-1]
            print(f"  Fetching {tourney_name}...", end=" ", flush=True)
            html = fetch_page(url)
            if html:
                rows = parse_results_page(html, tourney_name, tour, event_type)
                all_rows.extend(rows)
                print(f"{len(rows)} matches")
            else:
                print("FAILED")
            time.sleep(0.5)

    print(f"  Total for {category_name}: {len(all_rows)} matches")
    return all_rows


def main():
    all_data = []

    for cat_name, config in CATEGORIES.items():
        rows = scrape_category(cat_name, config)
        all_data.extend(rows)

    if not all_data:
        print("\nNo data scraped!")
        return

    df = pd.DataFrame(all_data)
    print(f"\n{'='*60}")
    print(f"TOTAL SCRAPED: {len(df)} matches")
    print(f"{'='*60}")

    # Filter to 2022+ only
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"] >= "2022-01-01"].copy()
    print(f"After filtering to 2022+: {len(df)} matches")

    # Summary
    print(f"\nBy event type:")
    print(df.groupby("event_type").size().to_string())
    print(f"\nWith winner odds: {df['odds_w'].notna().sum()}")
    print(f"With loser odds: {df['odds_l'].notna().sum()}")
    print(f"With both odds: {(df['odds_w'].notna() & df['odds_l'].notna()).sum()}")

    # Save
    output_path = ROOT / "data" / "raw" / "tennis" / "betexplorer_odds.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
