"""Recover tennis odds from Welcome.Bet using the site's sitemap.

This is a targeted manual-web source for rows still unresolved in
data/processed/tennis/unmatched_diagnostics.csv. It preserves provenance in a
separate raw file and only writes rows when the page names, date, and numeric
odds all validate against an unmatched match row.
"""

from __future__ import annotations

import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fix_tennis_odds_matching import (
    PLAYER_ALTERNATE_SURNAMES,
    names_match_flexible,
    normalize_text,
)

DIAGNOSTICS_PATH = ROOT / "data" / "processed" / "tennis" / "unmatched_diagnostics.csv"
OUTPUT_PATH = ROOT / "data" / "raw" / "tennis" / "welcome_bet_odds.csv"
SITEMAP_CACHE_PATH = ROOT / "data" / "raw" / "tennis" / "welcome_bet_post_urls.txt"

REQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    )
}
SITEMAP_DELAY_SECONDS = 0.35
PAGE_DELAY_SECONDS = 1.0
MAX_DATE_DELTA_DAYS = 2

MONTHS = {
    1: "jan",
    2: "feb",
    3: "mar",
    4: "apr",
    5: "may",
    6: "jun",
    7: "jul",
    8: "aug",
    9: "sep",
    10: "oct",
    11: "nov",
    12: "dec",
}


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def build_name_variants(name: str) -> list[str]:
    norm = normalize_text(name)
    if not norm:
        return []

    tokens = norm.split()
    if not tokens:
        return []

    variants = {norm}
    alt_surnames = PLAYER_ALTERNATE_SURNAMES.get(norm, [])
    if len(tokens) >= 2 and alt_surnames:
        first_part = " ".join(tokens[:-1])
        for alt_surname in alt_surnames:
            alt_surname = normalize_text(alt_surname)
            if alt_surname:
                variants.add(f"{first_part} {alt_surname}".strip())

    return sorted(slugify(variant) for variant in variants if slugify(variant))


def row_player_variant_sets(row: pd.Series) -> tuple[set[str], set[str]]:
    winner_names = {
        row.get("winner"),
        row.get("player_a"),
        row.get("player_b"),
    }
    loser_names = {
        row.get("loser_name"),
        row.get("player_a"),
        row.get("player_b"),
    }

    winner_variants = set()
    loser_variants = set()

    for name in winner_names:
        for variant in build_name_variants(name):
            winner_variants.add(variant)

    for name in loser_names:
        for variant in build_name_variants(name):
            loser_variants.add(variant)

    winner_variants.discard("")
    loser_variants.discard("")
    return winner_variants, loser_variants


def fetch_xml(session: requests.Session, url: str) -> BeautifulSoup:
    response = session.get(url, headers=REQ_HEADERS, timeout=60)
    response.raise_for_status()
    time.sleep(SITEMAP_DELAY_SECONDS)
    return BeautifulSoup(response.text, "xml")


def load_sitemap_urls(session: requests.Session, refresh: bool = False) -> list[str]:
    if SITEMAP_CACHE_PATH.exists() and not refresh:
        urls = [
            line.strip()
            for line in SITEMAP_CACHE_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if urls:
            return urls

    sitemap_index = fetch_xml(session, "https://welcome.bet/sitemap.xml")
    sitemap_urls = [
        loc.get_text(strip=True)
        for loc in sitemap_index.find_all("loc")
        if "post-sitemap" in loc.get_text(strip=True)
    ]
    all_urls = []
    for sitemap_url in sitemap_urls:
        soup = fetch_xml(session, sitemap_url)
        urls = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
        all_urls.extend(urls)

    unique_urls = sorted(set(all_urls))
    SITEMAP_CACHE_PATH.write_text("\n".join(unique_urls) + "\n", encoding="utf-8")
    return unique_urls


def build_date_index(urls: list[str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for url in urls:
        path = urlparse(url).path.strip("/").lower()
        match = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)-(\d{2})-(\d{4})/?$", path)
        if not match:
            continue
        date_slug = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        index[date_slug].append(url)
    return index


def extract_page_match_data(html: str) -> dict | None:
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text("\n", strip=True)

    heading = soup.find("h1")
    heading_text = heading.get_text(" ", strip=True) if heading else ""
    if not heading_text:
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        heading_text = re.sub(r"\s*\|\s*Welcome\.Bet\s*$", "", title)

    title_match = re.search(
        r"(.+?)\s+vs\s+(.+?),\s+([A-Za-z]{3})\s+(\d{2}),\s+(\d{4})",
        heading_text,
        flags=re.IGNORECASE,
    )
    if not title_match:
        return None

    player1 = re.sub(r"\s+predictions?$", "", title_match.group(1).strip(), flags=re.IGNORECASE)
    player2 = re.sub(r"\s+predictions?$", "", title_match.group(2).strip(), flags=re.IGNORECASE)
    month_name = title_match.group(3).lower()
    month_number = {abbr: month for month, abbr in MONTHS.items()}[month_name]
    match_date = f"{title_match.group(5)}-{month_number:02d}-{title_match.group(4)}"

    tournament = ""
    tournament_match = re.search(r"\n([A-Za-z0-9'&().:/#,\- ]+?),\s+on\s+[A-Za-z]+,\s+\d{1,2}:\d{2}\s+[ap]m\s+ET", text)
    if tournament_match:
        tournament = tournament_match.group(1).strip()

    odds_match = re.search(
        r"bookmakers have given odds of\s+(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if odds_match:
        odds1 = float(odds_match.group(1))
        odds2 = float(odds_match.group(2))
    else:
        section_text = text
        if "Betting Odds" in section_text:
            section_text = section_text.split("Betting Odds", 1)[1]
        for marker in ["Latest Results", "Latest Head To Head", "Related Articles", "Leagues"]:
            if marker in section_text:
                section_text = section_text.split(marker, 1)[0]
                break

        decimal_values = re.findall(r"\b\d+(?:\.\d+)\b", section_text)
        if len(decimal_values) < 2:
            return None
        odds1 = float(decimal_values[0])
        odds2 = float(decimal_values[1])

    return {
        "page_player1": player1,
        "page_player2": player2,
        "Date": match_date,
        "page_tournament": tournament,
        "player1_odds": odds1,
        "player2_odds": odds2,
    }


def row_matches_page(row: pd.Series, page_data: dict) -> tuple[bool, float | None, float | None]:
    winner = row.get("winner")
    loser = row.get("loser_name")
    player1 = page_data["page_player1"]
    player2 = page_data["page_player2"]

    if names_match_flexible(winner, player1) and names_match_flexible(loser, player2):
        return True, page_data["player1_odds"], page_data["player2_odds"]
    if names_match_flexible(winner, player2) and names_match_flexible(loser, player1):
        return True, page_data["player2_odds"], page_data["player1_odds"]
    return False, None, None


def find_candidate_urls(row: pd.Series, date_index: dict[str, list[str]]) -> list[str]:
    event_date = pd.to_datetime(row.get("event_date"), errors="coerce")
    if pd.isna(event_date):
        return []

    date_urls = []
    seen_date_urls = set()
    for day_offset in range(-MAX_DATE_DELTA_DAYS, MAX_DATE_DELTA_DAYS + 1):
        candidate_date = event_date + timedelta(days=day_offset)
        date_slug = f"{MONTHS[candidate_date.month]}-{candidate_date.day:02d}-{candidate_date.year}"
        for url in date_index.get(date_slug, []):
            if url in seen_date_urls:
                continue
            date_urls.append(url)
            seen_date_urls.add(url)
    if not date_urls:
        return []

    winner_variants, loser_variants = row_player_variant_sets(row)
    if not winner_variants or not loser_variants:
        return []

    scored = []
    for url in date_urls:
        path = urlparse(url).path.strip("/").lower()
        has_winner = any(variant in path for variant in winner_variants)
        has_loser = any(variant in path for variant in loser_variants)
        if not (has_winner and has_loser):
            continue

        score = 0
        for variant in winner_variants:
            if variant in path:
                score += len(variant)
        for variant in loser_variants:
            if variant in path:
                score += len(variant)
        scored.append((score, url))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [url for _, url in scored[:6]]


def load_rows(filters: list[str]) -> pd.DataFrame:
    diagnostics_df = pd.read_csv(DIAGNOSTICS_PATH, low_memory=False)
    diagnostics_df = diagnostics_df[
        diagnostics_df["classification"].isin(["source_absent", "source_has_no_numeric_odds"])
    ].copy()

    if filters:
        lowered = [value.lower() for value in filters]
        mask = diagnostics_df["bucket"].astype(str).str.lower().apply(
            lambda bucket: any(token in bucket for token in lowered)
        )
        diagnostics_df = diagnostics_df[mask].copy()

    diagnostics_df = diagnostics_df.drop_duplicates(
        subset=["event_date", "winner", "loser_name", "bucket"]
    ).reset_index(drop=True)
    return diagnostics_df


def merge_rows(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    new_df = pd.DataFrame(rows)
    old_df = pd.read_csv(OUTPUT_PATH, low_memory=False) if OUTPUT_PATH.exists() else pd.DataFrame()
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "Winner", "Loser", "source_url"], keep="last")
    combined.to_csv(OUTPUT_PATH, index=False)
    return new_df, combined


def main():
    filters = [arg.strip() for arg in sys.argv[1:] if arg.strip()]
    refresh = False
    if "--refresh-sitemap" in filters:
        filters.remove("--refresh-sitemap")
        refresh = True

    target_rows = load_rows(filters)
    if target_rows.empty:
        print("No unmatched rows selected.")
        return

    session = requests.Session()
    urls = load_sitemap_urls(session, refresh=refresh)
    date_index = build_date_index(urls)
    print(f"Loaded {len(urls)} Welcome.Bet post URLs")
    print(f"Date index buckets: {len(date_index)}")
    print(f"Target unmatched rows: {len(target_rows)}")

    recovered_rows = []
    seen_urls = set()

    for _, row in target_rows.iterrows():
        candidate_urls = find_candidate_urls(row, date_index)
        if not candidate_urls:
            continue

        for url in candidate_urls:
            if url in seen_urls:
                continue

            try:
                response = session.get(url, headers=REQ_HEADERS, timeout=45)
            except requests.RequestException:
                continue
            time.sleep(PAGE_DELAY_SECONDS)
            if response.status_code != 200:
                continue

            page_data = extract_page_match_data(response.text)
            if not page_data:
                continue
            page_date = pd.to_datetime(page_data["Date"], errors="coerce")
            row_date = pd.to_datetime(row["event_date"], errors="coerce")
            if pd.isna(page_date) or pd.isna(row_date):
                continue
            if abs((page_date - row_date).days) > MAX_DATE_DELTA_DAYS:
                continue

            matched, odds_w, odds_l = row_matches_page(row, page_data)
            if not matched or odds_w is None or odds_l is None:
                continue

            recovered_rows.append({
                "Date": page_data["Date"],
                "Winner": row["winner"],
                "Loser": row["loser_name"],
                "odds_w": float(odds_w),
                "odds_l": float(odds_l),
                "tournament": page_data["page_tournament"] or row.get("tourney_name", ""),
                "tour": row.get("tour", ""),
                "event_type": "manual_web",
                "source": "welcome_bet",
                "source_url": url,
                "page_tournament": page_data["page_tournament"],
                "page_player1": page_data["page_player1"],
                "page_player2": page_data["page_player2"],
                "bucket": row.get("bucket", ""),
            })
            merge_rows([recovered_rows[-1]])
            seen_urls.add(url)
            print(
                f"Recovered {row['event_date']} | {row['winner']} vs {row['loser_name']} "
                f"from {url}"
            )
            break

    if not recovered_rows:
        print("No rows recovered.")
        return

    new_df = pd.DataFrame(recovered_rows)
    combined_df = pd.read_csv(OUTPUT_PATH, low_memory=False) if OUTPUT_PATH.exists() else new_df.copy()
    print(f"Recovered rows this run: {len(new_df)}")
    print(f"Combined Welcome.Bet rows: {len(combined_df)}")
    print(new_df["bucket"].value_counts().to_string())


if __name__ == "__main__":
    main()
