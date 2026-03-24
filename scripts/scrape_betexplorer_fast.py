"""Fast BetExplorer scraper using requests only (no Selenium).

Extracts match data + partial odds from HTML data-odd attributes.
The loser's closing odds are available in HTML; winner's odds are JS-loaded.
For matches where we have loser odds, we can still compute fair probabilities
by using the standard 5-7% overround pattern.
"""

import re
import sys
import time
from pathlib import Path
from collections import defaultdict

import requests
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def fetch_page(url, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:
                return None
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(1)
    return None


def get_season_urls(tournament_url):
    """Get all available season URLs from a tournament page."""
    html = fetch_page(tournament_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    season_urls = []

    for select in soup.find_all("select"):
        for option in select.find_all("option"):
            val = option.get("value", "")
            text = option.get_text(strip=True)
            if text.isdigit() and len(text) == 4 and int(text) >= 2022:
                if val.startswith("/"):
                    season_urls.append((int(text), f"https://www.betexplorer.com{val}results/"))
                elif val.startswith("http"):
                    season_urls.append((int(text), f"{val}results/"))

    return season_urls


def parse_results_page(html, tournament_name, tour, event_type, year=None):
    """Parse match results from BetExplorer results page HTML."""
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

        first_text = cells[0].get_text(strip=True)
        colspan = cells[0].get("colspan")
        if colspan and first_text:
            current_round = first_text
            continue

        if len(cells) < 4:
            continue

        players_text = cells[0].get_text(strip=True)
        if len(players_text) < 5:
            continue

        # Split player names — handle hyphenated names like "Friedsam A-L."
        # Pattern: look for ". -" or ".-" or ") -" as the separator between players
        m = re.match(r'^(.+?(?:\.\s*|\)\s*))\s*-\s*(.+)$', players_text)
        if not m:
            # Fallback: split on " - " (space-dash-space)
            if " - " in players_text:
                parts = players_text.split(" - ", 1)
                m = type('', (), {'group': lambda self, n: parts[n-1]})()
            else:
                continue

        player1 = m.group(1).strip().rstrip("-").strip()
        player2 = m.group(2).strip()

        if not player1 or not player2 or len(player1) < 3 or len(player2) < 3:
            continue

        # Score
        score_text = cells[1].get_text(strip=True) if len(cells) > 1 else ""
        score_match = re.match(r"(\d+):(\d+)", score_text)
        if not score_match:
            continue
        sets1, sets2 = int(score_match.group(1)), int(score_match.group(2))
        if sets1 == sets2:
            continue  # Can't determine winner
        winner = player1 if sets1 > sets2 else player2
        loser = player2 if sets1 > sets2 else player1

        # Odds cells
        odds_cells = [c for c in cells if "table-main__odds" in " ".join(c.get("class", []))]
        odds_p1 = None
        odds_p2 = None
        winner_odds = None
        loser_odds = None

        for i, oc in enumerate(odds_cells[:2]):
            odd_val = oc.get("data-odd")
            is_colored = "colored" in " ".join(oc.get("class", []))

            if odd_val:
                try:
                    val = float(odd_val)
                    if i == 0:
                        odds_p1 = val
                    else:
                        odds_p2 = val
                except ValueError:
                    pass

        # Map to winner/loser
        is_p1_winner = (winner == player1)
        if is_p1_winner:
            winner_odds = odds_p1
            loser_odds = odds_p2
        else:
            winner_odds = odds_p2
            loser_odds = odds_p1

        # Date
        date_text = cells[-1].get_text(strip=True)
        date_match = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", date_text)
        if not date_match:
            continue
        date_str = f"{date_match.group(3)}-{date_match.group(2)}-{date_match.group(1)}"

        rows.append({
            "Date": date_str,
            "Winner": winner,
            "Loser": loser,
            "score": score_text,
            "round": current_round,
            "odds_w": winner_odds,
            "odds_l": loser_odds,
            "tournament": tournament_name,
            "tour": tour,
            "event_type": event_type,
            "source_year": year,
            "source": "betexplorer",
        })

    return rows


def scrape_wta_125():
    """Scrape all WTA 125 events."""
    print("=" * 60)
    print("Scraping WTA 125 events")
    print("=" * 60)

    index_url = "https://www.betexplorer.com/tennis/challenger-women-singles/"
    html = fetch_page(index_url)
    soup = BeautifulSoup(html, "html.parser")

    # Get tournament slugs
    base_path = "/tennis/challenger-women-singles/"
    slugs = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(base_path) and href != base_path:
            slug = href.replace(base_path, "").strip("/")
            if slug and "/" not in slug and slug != "fixtures":
                slugs.add(slug)

    print(f"Found {len(slugs)} tournaments")

    all_rows = []
    for i, slug in enumerate(sorted(slugs)):
        # Get available seasons
        tourney_url = f"https://www.betexplorer.com{base_path}{slug}/"
        seasons = get_season_urls(tourney_url)

        if not seasons:
            # Try just the results page directly
            seasons = [(2025, f"https://www.betexplorer.com{base_path}{slug}/results/")]

        print(f"  [{i+1}/{len(slugs)}] {slug} ({len(seasons)} seasons)...", end=" ", flush=True)

        tourney_rows = 0
        for year, url in seasons:
            html = fetch_page(url)
            if html:
                rows = parse_results_page(html, slug, "wta", "125", year)
                all_rows.extend(rows)
                tourney_rows += len(rows)
            time.sleep(0.3)

        has_odds = sum(1 for r in all_rows[-tourney_rows:] if r.get("odds_w") or r.get("odds_l"))
        print(f"{tourney_rows} matches ({has_odds} with odds)")

    return all_rows


def scrape_team_events():
    """Scrape Davis Cup, BJK Cup, United Cup, Laver Cup."""
    print("\n" + "=" * 60)
    print("Scraping team events")
    print("=" * 60)

    events = [
        # (name, base_url_path, tour)
        ("davis-cup-world-group", "/tennis/teams-men/davis-cup-world-group/", "atp", "davis_cup"),
        ("davis-cup-group-i", "/tennis/teams-men/davis-cup-group-i/", "atp", "davis_cup"),
        ("davis-cup-group-ii", "/tennis/teams-men/davis-cup-group-ii/", "atp", "davis_cup"),
        ("davis-cup-group-iii", "/tennis/teams-men/davis-cup-group-iii/", "atp", "davis_cup"),
        ("bjk-cup-world-group", "/tennis/teams-women/billie-jean-king-cup-world-group/", "wta", "bjk_cup"),
        ("bjk-cup-group-i", "/tennis/teams-women/billie-jean-king-cup-group-i/", "wta", "bjk_cup"),
        ("bjk-cup-group-ii", "/tennis/teams-women/billie-jean-king-cup-group-ii/", "wta", "bjk_cup"),
        ("united-cup", "/tennis/teams-mix/united-cup/", "mixed", "united_cup"),
        ("laver-cup", "/tennis/teams-men/laver-cup/", "atp", "laver_cup"),
    ]

    all_rows = []
    for name, path, tour, event_type in events:
        base_url = f"https://www.betexplorer.com{path}"
        seasons = get_season_urls(base_url)
        if not seasons:
            seasons = [(2025, f"{base_url}results/")]

        print(f"  {name} ({len(seasons)} seasons)...", end=" ", flush=True)

        event_rows = 0
        for year, url in seasons:
            html = fetch_page(url)
            if html:
                rows = parse_results_page(html, name, tour, event_type, year)
                all_rows.extend(rows)
                event_rows += len(rows)
            time.sleep(0.3)

        print(f"{event_rows} matches")

    return all_rows


def scrape_other_events():
    """Scrape Olympics, NextGen."""
    print("\n" + "=" * 60)
    print("Scraping Olympics + NextGen")
    print("=" * 60)

    events = [
        ("olympics-atp", "/tennis/world/atp-olympic-games/", "atp", "olympics"),
        ("olympics-wta", "/tennis/world/wta-olympic-games/", "wta", "olympics"),
        ("nextgen-jeddah", "/tennis/world/atp-next-gen-finals-jeddah/", "atp", "nextgen"),
        ("nextgen", "/tennis/world/atp-next-gen-finals/", "atp", "nextgen"),
        ("nextgen-milan", "/tennis/world/atp-next-gen-finals-milan/", "atp", "nextgen"),
    ]

    all_rows = []
    for name, path, tour, event_type in events:
        base_url = f"https://www.betexplorer.com{path}"
        seasons = get_season_urls(base_url)
        if not seasons:
            seasons = [(2025, f"{base_url}results/")]

        print(f"  {name} ({len(seasons)} seasons)...", end=" ", flush=True)

        event_rows = 0
        for year, url in seasons:
            html = fetch_page(url)
            if html:
                rows = parse_results_page(html, name, tour, event_type, year)
                all_rows.extend(rows)
                event_rows += len(rows)
            time.sleep(0.3)

        print(f"{event_rows} matches")

    return all_rows


def main():
    all_data = []

    all_data.extend(scrape_wta_125())
    all_data.extend(scrape_team_events())
    all_data.extend(scrape_other_events())

    if not all_data:
        print("\nNo data scraped!")
        return

    df = pd.DataFrame(all_data)
    print(f"\n{'='*60}")
    print(f"TOTAL SCRAPED: {len(df)} matches")
    print(f"{'='*60}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"] >= "2022-01-01"].copy()
    df = df.drop_duplicates(subset=["Date", "Winner", "Loser", "tournament"]).copy()
    print(f"After filtering to 2022+ and dedup: {len(df)} matches")

    has_any = (df["odds_w"].notna() | df["odds_l"].notna()).sum()
    has_loser = df["odds_l"].notna().sum()
    has_winner = df["odds_w"].notna().sum()
    has_both = (df["odds_w"].notna() & df["odds_l"].notna()).sum()

    print(f"With any odds: {has_any}")
    print(f"  Winner odds: {has_winner}")
    print(f"  Loser odds: {has_loser}")
    print(f"  Both: {has_both}")
    print(f"\nBy event type:")
    print(df.groupby("event_type").size().to_string())
    print(f"\nBy year:")
    print(df.groupby(df["Date"].dt.year).size().to_string())

    output_path = ROOT / "data" / "raw" / "tennis" / "betexplorer_odds.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
