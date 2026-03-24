"""
Scrape historical WTA 125 odds from BetExplorer.com (2022-2026).
Outputs: data/raw/tennis/betexplorer_wta125_odds.csv
"""
import requests
import re
import csv
import time
import sys
from datetime import datetime

BASE = "https://www.betexplorer.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
YEARS = [2022, 2023, 2024, 2025, 2026]
OUTPUT = r"c:\Users\Evan\betting-bot\ufc-betting-bot\data\raw\tennis\betexplorer_wta125_odds.csv"
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def fetch(url, retries=3):
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=20)
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:
                return None
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"  FAIL {url}: {e}", file=sys.stderr)
                return None
        time.sleep(1.5 * (attempt + 1))
    return None


def get_tournament_slugs():
    """Fetch all WTA 125 tournament slugs from index page."""
    text = fetch(f"{BASE}/tennis/challenger-women-singles/")
    if not text:
        return []
    links = re.findall(
        r'href="/tennis/challenger-women-singles/([^/"]+)/"', text
    )
    return sorted(set(links))


def extract_odd(cell_html):
    """Extract odds value from a table cell HTML."""
    # Try data-odd on the td itself
    m = re.search(r'data-odd="([^"]+)"', cell_html)
    return m.group(1) if m else ""


def parse_results_page(html, tournament_name):
    """Parse match rows from a results page."""
    matches = []
    # Find all match rows
    for m in re.finditer(
        r'<tr><td class="h-text-left"><a[^>]*class="in-match"[^>]*>'
        r'(.*?)</a></td>(.*?)</tr>',
        html, re.DOTALL
    ):
        players_html = m.group(1)
        rest_html = m.group(2)

        # Extract player names and winner
        pm = re.match(
            r'<span>((?:<strong>)?.*?(?:</strong>)?)</span>'
            r' - '
            r'<span>((?:<strong>)?.*?(?:</strong>)?)</span>',
            players_html
        )
        if not pm:
            continue

        p1_raw, p2_raw = pm.group(1), pm.group(2)
        p1_is_winner = "<strong>" in p1_raw
        p2_is_winner = "<strong>" in p2_raw

        if not p1_is_winner and not p2_is_winner:
            continue  # skip walkovers / incomplete

        p1 = re.sub(r"</?strong>", "", p1_raw).strip()
        p2 = re.sub(r"</?strong>", "", p2_raw).strip()

        # Extract odds cells (two table-main__odds cells)
        odds_cells = re.findall(
            r'<td class="table-main__odds[^"]*"[^>]*>.*?</td>',
            rest_html, re.DOTALL
        )
        if len(odds_cells) < 2:
            continue

        odd1 = extract_odd(odds_cells[0])
        odd2 = extract_odd(odds_cells[1])

        if not odd1 or not odd2:
            continue

        # Extract date
        dm = re.search(r'(\d{2}\.\d{2}\.\d{4})', rest_html)
        if not dm:
            continue
        date_str = dm.group(1)

        # Convert date format
        try:
            dt = datetime.strptime(date_str, "%d.%m.%Y")
            date_iso = dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

        # Assign winner/loser and their odds
        if p1_is_winner:
            winner, loser = p1, p2
            w_odd, l_odd = odd1, odd2
        else:
            winner, loser = p2, p1
            w_odd, l_odd = odd2, odd1

        matches.append({
            "Date": date_iso,
            "Winner": winner,
            "Loser": loser,
            "B365W": w_odd,
            "B365L": l_odd,
            "tournament": tournament_name,
            "tour": "wta",
            "source": "betexplorer",
        })

    return matches


def main():
    print("Fetching tournament list...")
    slugs = get_tournament_slugs()
    print(f"Found {len(slugs)} tournaments")

    all_matches = []
    failed = []

    for i, slug in enumerate(slugs):
        tournament_name = slug.replace("-", " ").title()
        # Remove trailing year from display name if present (e.g., "Reus 2023")
        # but keep it - it's part of the tournament identity

        for year in YEARS:
            url = f"{BASE}/tennis/challenger-women-singles/{slug}-{year}/results/"
            # For the current/default season, also try without year
            html = fetch(url)

            if html is None or "data-odd" not in html:
                # Try without year suffix (for current season or tournaments named differently)
                if year == 2025:
                    url2 = f"{BASE}/tennis/challenger-women-singles/{slug}/results/"
                    html2 = fetch(url2)
                    if html2 and "data-odd" in html2:
                        html = html2
                        url = url2

            if html and "data-odd" in html:
                matches = parse_results_page(html, tournament_name)
                if matches:
                    # Filter to only include matches from the target year
                    matches = [m for m in matches if m["Date"].startswith(str(year))]
                    all_matches.extend(matches)
                    print(f"  [{i+1}/{len(slugs)}] {slug} {year}: {len(matches)} matches")
                else:
                    pass  # No matches with odds for this year - normal
            else:
                pass  # Tournament didn't exist this year - normal

            # Rate limit
            time.sleep(0.4)

        # Progress
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(slugs)} tournaments, {len(all_matches)} total matches")

    # Write CSV
    print(f"\nWriting {len(all_matches)} matches to {OUTPUT}")
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Date", "Winner", "Loser", "B365W", "B365L",
            "tournament", "tour", "source"
        ])
        writer.writeheader()
        # Sort by date
        all_matches.sort(key=lambda x: x["Date"])
        writer.writerows(all_matches)

    print(f"Done! {len(all_matches)} matches saved.")
    if failed:
        print(f"Failed URLs: {len(failed)}")
        for f in failed[:10]:
            print(f"  {f}")


if __name__ == "__main__":
    main()
