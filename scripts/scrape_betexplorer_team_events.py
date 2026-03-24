"""
Scrape historical tennis team event odds from BetExplorer.com
Davis Cup, BJK Cup, United Cup (2022-2026)
"""
import requests
from bs4 import BeautifulSoup
import re
import csv
import time
import os
import pickle

BASE_URL = "https://www.betexplorer.com"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

ajax_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}

session = requests.Session()
session.headers.update(headers)

# Define all tournaments
TOURNAMENTS = {
    "Davis Cup": {
        "base_paths": [
            "/tennis/teams-men/davis-cup-world-group/",
            "/tennis/teams-men/davis-cup-world-group-i/",
            "/tennis/teams-men/davis-cup-world-group-ii/",
            "/tennis/teams-men/davis-cup-group-i/",
            "/tennis/teams-men/davis-cup-group-ii/",
            "/tennis/teams-men/davis-cup-group-iii/",
            "/tennis/teams-men/davis-cup-group-iv/",
            "/tennis/teams-men/davis-cup-group-v/",
        ],
        "tour": "Davis Cup",
    },
    "BJK Cup": {
        "base_paths": [
            "/tennis/teams-women/billie-jean-king-cup-world-group/",
            "/tennis/teams-women/billie-jean-king-cup-world-group-ii/",
            "/tennis/teams-women/billie-jean-king-cup-group-i/",
            "/tennis/teams-women/billie-jean-king-cup-group-ii/",
            "/tennis/teams-women/billie-jean-king-cup-group-iii/",
            "/tennis/teams-women/billie-jean-king-cup-group-iv/",
        ],
        "tour": "BJK Cup",
    },
    "United Cup": {
        "base_paths": [
            "/tennis/teams-mix/united-cup/",
        ],
        "tour": "United Cup",
    },
}

YEARS = [2022, 2023, 2024, 2025, 2026]
CURRENT_YEAR = 2026  # Most recent season uses the base path without year suffix


def get_season_url(base_path, year):
    """Construct the season URL. Current year uses base path; older years append year."""
    if year == CURRENT_YEAR:
        return base_path
    return base_path.rstrip("/") + f"-{year}/"


def get_results_page(url):
    """Fetch results page and return BeautifulSoup object."""
    results_url = BASE_URL + url + "results/"
    try:
        r = session.get(results_url, timeout=15)
        if r.status_code == 200 and len(r.text) > 1000:
            return BeautifulSoup(r.text, "html.parser"), results_url
    except Exception as e:
        print(f"  Error fetching {results_url}: {e}")
    return None, results_url


def extract_tournament_name(soup):
    """Extract tournament name from page title."""
    title = soup.find("title")
    if title:
        t = title.get_text(strip=True)
        return t.split(",")[0].strip().replace(" Results", "").replace(" Teams", "")
    return ""


def parse_results(soup, year):
    """Parse match results from a results page."""
    matches = []
    rows = soup.find_all("tr")
    current_round = ""

    for row in rows:
        cells = row.find_all(["td", "th"])
        if len(cells) < 5:
            if len(cells) >= 1:
                text = cells[0].get_text(strip=True)
                if text and "Betting" not in text and "Bookmaker" not in text:
                    current_round = text
            continue

        teams_cell = cells[0]
        score_cell = cells[1]
        odds1_cell = cells[2]
        odds2_cell = cells[3]
        date_cell = cells[4]

        teams = teams_cell.get_text(strip=True)
        score = score_cell.get_text(strip=True)

        if not teams or ":" not in score:
            continue

        # Parse score
        score_parts = score.split(":")
        try:
            s1, s2 = int(score_parts[0]), int(score_parts[1])
        except (ValueError, IndexError):
            continue

        if s1 == s2:
            continue  # skip draws

        winner_side = 1 if s1 > s2 else 2

        # Get match link and ID
        link = teams_cell.find("a", href=True)
        match_id = None
        match_href = None
        if link:
            match_href = link["href"]
            id_match = re.search(r"/([A-Za-z0-9]+)/$", match_href)
            if id_match:
                match_id = id_match.group(1)

        # Parse date
        date_text = date_cell.get_text(strip=True)
        date_re = re.match(r"(\d{2})\.(\d{2})\.", date_text)
        if date_re:
            day, month = date_re.group(1), date_re.group(2)
            date_str = f"{year}-{month}-{day}"
        else:
            date_str = str(year)

        # Get partial odds
        odds1_val = odds1_cell.get("data-odd", "")
        odds2_val = odds2_cell.get("data-odd", "")
        odds1_colored = "colored" in (odds1_cell.get("class") or [])

        matches.append({
            "team_text": teams,
            "score": score,
            "date": date_str,
            "match_id": match_id,
            "match_href": match_href,
            "winner_side": winner_side,
            "odds1_val": odds1_val,
            "odds2_val": odds2_val,
            "odds1_is_winner": odds1_colored,
            "round": current_round,
        })

    return matches


def fetch_match_odds(match_id):
    """Fetch average closing odds for a match via AJAX endpoint."""
    url = f"{BASE_URL}/match-odds-old/{match_id}/1/ha/1/en/"
    try:
        r = session.get(url, headers=ajax_headers, timeout=15)
        if r.status_code != 200:
            return None, None
        data = r.json()
        if "odds" not in data:
            return None, None

        soup = BeautifulSoup(data["odds"], "html.parser")

        # Try 1: Average odds from tfoot
        tfoot = soup.find("tfoot")
        if tfoot:
            all_odds = [c.get("data-odd") for c in tfoot.find_all(attrs={"data-odd": True})]
            if len(all_odds) >= 2 and all_odds[0] and all_odds[1]:
                return all_odds[0], all_odds[1]

        # Try 2: Fall back to individual bookmaker odds rows
        bookie_odds_pairs = []
        all_rows = soup.find_all("tr")
        for row in all_rows:
            # Skip the header row and the average/tfoot row
            cells = row.find_all("td")
            # Find cells with table-main__detail-odds class that have non-empty data-odd
            detail_cells = []
            for td in cells:
                cls = td.get("class") or []
                cls_str = " ".join(cls)
                if "table-main__detail-odds" in cls_str and td.get("data-odd", ""):
                    detail_cells.append(td)
            if len(detail_cells) >= 2:
                o1 = detail_cells[0].get("data-odd", "")
                o2 = detail_cells[1].get("data-odd", "")
                if o1 and o2:
                    try:
                        bookie_odds_pairs.append((float(o1), float(o2)))
                    except ValueError:
                        pass

        if bookie_odds_pairs:
            avg1 = round(sum(p[0] for p in bookie_odds_pairs) / len(bookie_odds_pairs), 2)
            avg2 = round(sum(p[1] for p in bookie_odds_pairs) / len(bookie_odds_pairs), 2)
            return str(avg1), str(avg2)

        return None, None
    except Exception as e:
        return None, None


def split_team_names(team_text):
    """Split 'Team1-Team2' into (team1, team2).
    Handles hyphenated country names like 'South Korea'."""
    # Common multi-word team names
    multi_word = [
        "South Korea", "South Africa", "North Macedonia", "New Zealand",
        "Great Britain", "Czech Republic", "El Salvador", "Hong Kong",
        "Ivory Coast", "Puerto Rico", "Dominican Republic", "Saudi Arabia",
        "Sri Lanka", "Costa Rica", "Burkina Faso", "Chinese Taipei",
        "Bosnia and Herzegovina", "Bosnia Herzegovina", "Trinidad and Tobago",
        "United Arab Emirates",
    ]

    for mw in multi_word:
        # Check if multi-word name is at the start
        if team_text.startswith(mw + "-"):
            team1 = mw
            team2 = team_text[len(mw) + 1:]
            return team1, team2
        # Check if at the end
        if team_text.endswith("-" + mw):
            team2 = mw
            team1 = team_text[:-(len(mw) + 1)]
            return team1, team2

    # Default: split on first hyphen
    parts = team_text.split("-", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return team_text, ""


def main():
    output_dir = r"c:\Users\Evan\betting-bot\ufc-betting-bot\data\raw\tennis"
    output_file = os.path.join(output_dir, "betexplorer_team_events_odds.csv")
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1: Collect all matches
    print("=== Phase 1: Collecting matches from results pages ===")
    all_matches = []

    for tour_key, tour_info in TOURNAMENTS.items():
        for base_path in tour_info["base_paths"]:
            for year in YEARS:
                season_url = get_season_url(base_path, year)
                soup, full_url = get_results_page(season_url)

                if soup is None:
                    continue

                tournament_label = extract_tournament_name(soup)
                if not tournament_label:
                    tournament_label = tour_key

                matches = parse_results(soup, year)

                for m in matches:
                    m["tournament"] = tournament_label
                    m["tour"] = tour_info["tour"]
                    m["year"] = year

                all_matches.extend(matches)
                if matches:
                    print(f"  {season_url}: {len(matches)} matches")
                time.sleep(0.3)

    print(f"\nTotal matches collected: {len(all_matches)}")

    # Phase 2: Fetch full odds for matches that have match IDs
    print("\n=== Phase 2: Fetching odds via AJAX ===")
    matches_with_id = [m for m in all_matches if m["match_id"]]
    print(f"Matches with match_id: {len(matches_with_id)}")

    fetched = 0
    failed = 0
    skipped_no_id = 0

    for i, m in enumerate(all_matches):
        if not m["match_id"]:
            skipped_no_id += 1
            continue

        odds1, odds2 = fetch_match_odds(m["match_id"])
        if odds1 and odds2:
            m["odds_team1"] = odds1
            m["odds_team2"] = odds2
            fetched += 1
        else:
            # Fall back to partial odds from listing page
            m["odds_team1"] = ""
            m["odds_team2"] = ""
            # Use what we have from the listing
            if m["odds1_val"] and m["odds2_val"]:
                m["odds_team1"] = m["odds1_val"] if not m["odds1_is_winner"] else m["odds1_val"]
                m["odds_team2"] = m["odds2_val"]
            failed += 1

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(all_matches)} (fetched={fetched}, failed={failed})")
        time.sleep(0.4)  # Be polite

    print(f"\nOdds fetch complete: fetched={fetched}, failed={failed}, no_id={skipped_no_id}")

    # Phase 3: Build output CSV
    print("\n=== Phase 3: Writing CSV ===")
    rows_written = 0

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Winner", "Loser", "odds_w", "odds_l", "tournament", "tour", "source"])

        for m in all_matches:
            team1, team2 = split_team_names(m["team_text"])
            if not team1 or not team2:
                continue

            odds_t1 = m.get("odds_team1", "")
            odds_t2 = m.get("odds_team2", "")

            if not odds_t1 or not odds_t2:
                continue  # Skip matches without complete odds

            if m["winner_side"] == 1:
                winner, loser = team1, team2
                odds_w, odds_l = odds_t1, odds_t2
            else:
                winner, loser = team2, team1
                odds_w, odds_l = odds_t2, odds_t1

            writer.writerow([
                m["date"],
                winner,
                loser,
                odds_w,
                odds_l,
                m["tournament"],
                m["tour"],
                "betexplorer.com",
            ])
            rows_written += 1

    print(f"\nWrote {rows_written} rows to {output_file}")
    print("Done!")


if __name__ == "__main__":
    main()
