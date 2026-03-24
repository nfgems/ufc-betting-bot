"""Recover tennis odds from search-indexed article pages.

Current implementation targets SportsbookWire pages discovered via DuckDuckGo
lite results. These pages often contain exact player names, tournament, and
moneyline odds for otherwise missing matches.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fix_tennis_odds_matching import (
    names_match_flexible,
    normalize_round_name,
    normalize_text,
    tournament_norm_candidates,
)

DIAGNOSTICS_PATH = ROOT / "data" / "processed" / "tennis" / "unmatched_diagnostics.csv"
OUTPUT_PATH = ROOT / "data" / "raw" / "tennis" / "manual_search_articles_odds.csv"

REQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    )
}
SEARCH_DELAY_SECONDS = 1.0
FETCH_DELAY_SECONDS = 1.0
MAX_DATE_DELTA_DAYS = 14

MONTH_MAP = {
    "jan": "01",
    "january": "01",
    "feb": "02",
    "february": "02",
    "mar": "03",
    "march": "03",
    "apr": "04",
    "april": "04",
    "may": "05",
    "jun": "06",
    "june": "06",
    "jul": "07",
    "july": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}


def american_to_decimal(value: int) -> float:
    if value > 0:
        return round(1.0 + value / 100.0, 6)
    return round(1.0 + 100.0 / abs(value), 6)


def load_rows(filters: list[str]) -> pd.DataFrame:
    df = pd.read_csv(DIAGNOSTICS_PATH, low_memory=False)
    df = df[df["classification"] == "source_absent"].copy()
    if filters:
        lowered = [f.lower() for f in filters]
        mask = df["bucket"].astype(str).str.lower().apply(
            lambda bucket: any(token in bucket for token in lowered)
        )
        df = df[mask].copy()
    df = df.drop_duplicates(subset=["event_date", "winner", "loser_name", "bucket"]).reset_index(drop=True)
    return df


def normalize_article_tournament(name: str) -> str:
    norm = normalize_text(name)
    overrides = {
        "moselle open": "open de moselle",
    }
    return overrides.get(norm, norm)


def search_yahoo(session: requests.Session, query: str) -> list[dict]:
    url = "https://search.yahoo.com/search?p=" + requests.utils.quote(query)
    response = session.get(url, headers=REQ_HEADERS, timeout=45)
    response.raise_for_status()
    time.sleep(SEARCH_DELAY_SECONDS)

    soup = BeautifulSoup(response.text, "lxml")
    results = []
    for block in soup.select("div.algo"):
        link = block.select_one(".compTitle a[href]")
        if not link:
            continue
        raw_url = link.get("href", "")
        yahoo_match = re.search(r"/RU=([^/]+)/RK=", raw_url)
        resolved_url = unquote(yahoo_match.group(1)) if yahoo_match else raw_url
        snippet = block.select_one(".compText p")
        results.append({
            "title": block.select_one("h3").get_text(" ", strip=True) if block.select_one("h3") else "",
            "url": resolved_url,
            "snippet": snippet.get_text(" ", strip=True) if snippet else "",
        })
    return results


def parse_sportsbookwire_page(html: str) -> dict | None:
    text = BeautifulSoup(html, "lxml").get_text("\n", strip=True)

    matchup_match = re.search(r"([A-Za-z .'\-]+) vs\. ([A-Za-z .'\-]+) matchup info", text)
    if not matchup_match:
        return None
    player1 = matchup_match.group(1).strip()
    player2 = matchup_match.group(2).strip()

    odds_pairs = re.findall(r"([A-Za-z .'\-]+)'s odds to win match:\s*([+-]\d+)", text)
    if len(odds_pairs) < 2:
        return None

    odds_map = {}
    for name, odds_text in odds_pairs[:2]:
        odds_map[name.strip()] = int(odds_text)

    tournament_match = re.search(r"Tournament:\s*([A-Za-z0-9 '&().\-]+)", text)
    tournament = tournament_match.group(1).strip() if tournament_match else ""

    article_year_match = re.search(
        r"(?:Updated\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2},\s+(\d{4})",
        text,
        flags=re.IGNORECASE,
    )
    article_year = article_year_match.group(1) if article_year_match else ""

    page_date = ""
    date_match = re.search(
        r"Date:\s*(?:[A-Za-z]+,\s*)?([A-Za-z]+)\s+(\d{1,2})",
        text,
        flags=re.IGNORECASE,
    )
    if article_year and date_match:
        month = MONTH_MAP.get(date_match.group(1).lower().rstrip("."))
        if month:
            page_date = f"{article_year}-{month}-{int(date_match.group(2)):02d}"

    round_match = re.search(r"Round:\s*([A-Za-z ]+)", text)
    page_round = round_match.group(1).strip() if round_match else ""

    return {
        "page_player1": player1,
        "page_player2": player2,
        "page_tournament": tournament,
        "page_date": page_date,
        "page_round": page_round,
        "american_odds": odds_map,
        "text": text,
    }


def row_matches_page(row: pd.Series, page_data: dict) -> tuple[bool, float | None, float | None, str]:
    winner = row["winner"]
    loser = row["loser_name"]

    player1 = page_data["page_player1"]
    player2 = page_data["page_player2"]
    odds_map = page_data["american_odds"]

    def odds_for(name: str) -> float | None:
        for candidate_name, american in odds_map.items():
            if names_match_flexible(name, candidate_name):
                return american_to_decimal(int(american))
        return None

    if names_match_flexible(winner, player1) and names_match_flexible(loser, player2):
        return True, odds_for(winner), odds_for(loser), "sportsbookwire"
    if names_match_flexible(winner, player2) and names_match_flexible(loser, player1):
        return True, odds_for(winner), odds_for(loser), "sportsbookwire"
    return False, None, None, ""


def page_tournament_compatible(row: pd.Series, page_data: dict) -> bool:
    page_tournament = page_data.get("page_tournament", "")
    if not page_tournament:
        return True

    event_date = pd.to_datetime(row.get("event_date"), errors="coerce")
    year = int(event_date.year) if pd.notna(event_date) else None
    tour = row.get("tour", "")
    row_candidates = tournament_norm_candidates(row.get("tourney_name", ""), tour, year)
    page_norm = normalize_article_tournament(page_tournament)
    page_candidates = {page_norm}
    page_candidates |= tournament_norm_candidates(page_tournament, tour, year)

    if not row_candidates or not page_candidates:
        return True

    for left in row_candidates:
        for right in page_candidates:
            if left == right:
                return True
            if len(left) >= 4 and len(right) >= 4 and (left in right or right in left):
                return True

    return False


def page_round_compatible(row: pd.Series, page_data: dict) -> bool:
    row_round = normalize_round_name(row.get("round", ""))
    page_round = normalize_round_name(page_data.get("page_round", ""))
    if not row_round or not page_round:
        return True
    return row_round == page_round


def merge_rows(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    new_df = pd.DataFrame(rows)
    old_df = pd.read_csv(OUTPUT_PATH, low_memory=False) if OUTPUT_PATH.exists() else pd.DataFrame()
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "Winner", "Loser", "source_url"], keep="last")
    combined.to_csv(OUTPUT_PATH, index=False)
    return new_df, combined


def main():
    filters = [arg.strip() for arg in sys.argv[1:] if arg.strip()]
    rows = load_rows(filters)
    if rows.empty:
        print("No unresolved source_absent rows selected.")
        return

    session = requests.Session()
    recovered = []
    seen_urls = set()

    for _, row in rows.iterrows():
        base_queries = [
            f'site:sportsbookwire.usatoday.com "{row["winner"]}" "{row["loser_name"]}" "{row["tourney_name"]}" tennis odds',
            f'site:sportsbookwire.usatoday.com "{row["winner"]}" "{row["loser_name"]}" "{row["bucket"]}" tennis odds',
            f'site:sportsbookwire.usatoday.com "{row["winner"]}" "{row["loser_name"]}" {pd.to_datetime(row["event_date"]).year} tennis odds',
        ]

        matched = False
        for query in base_queries:
            try:
                results = search_yahoo(session, query)
            except Exception:
                continue
            candidates = [
                result for result in results
                if result["url"] and "sportsbookwire.usatoday.com" in result["url"]
            ]
            for candidate in candidates[:3]:
                url = candidate["url"]
                if url in seen_urls:
                    continue

                response = session.get(url, headers=REQ_HEADERS, timeout=45)
                time.sleep(FETCH_DELAY_SECONDS)
                if response.status_code != 200:
                    continue

                page_data = parse_sportsbookwire_page(response.text)
                if not page_data:
                    continue

                page_date = pd.to_datetime(page_data.get("page_date"), errors="coerce")
                row_date = pd.to_datetime(row["event_date"], errors="coerce")
                if pd.isna(page_date) or pd.isna(row_date):
                    continue
                if abs((page_date - row_date).days) > MAX_DATE_DELTA_DAYS:
                    continue
                if not page_tournament_compatible(row, page_data):
                    continue
                if not page_round_compatible(row, page_data):
                    continue

                ok, odds_w, odds_l, source = row_matches_page(row, page_data)
                if not ok or odds_w is None or odds_l is None:
                    continue

                recovered.append({
                    "Date": page_data["page_date"],
                    "Winner": row["winner"],
                    "Loser": row["loser_name"],
                    "odds_w": odds_w,
                    "odds_l": odds_l,
                    "tournament": page_data["page_tournament"] or row["tourney_name"],
                    "tour": row["tour"],
                    "event_type": "manual_search_article",
                    "source": source,
                    "source_url": url,
                    "source_note": candidate["snippet"],
                    "round": normalize_round_name(page_data["page_round"]),
                    "bucket": row["bucket"],
                })
                seen_urls.add(url)
                matched = True
                print(f"Recovered {row['winner']} vs {row['loser_name']} from {url}")
                break

            if matched:
                break

    if not recovered:
        print("No rows recovered.")
        return

    new_df, combined_df = merge_rows(recovered)
    print(f"Recovered rows this run: {len(new_df)}")
    print(f"Combined article rows: {len(combined_df)}")
    print(new_df["bucket"].value_counts().to_string())


if __name__ == "__main__":
    main()
