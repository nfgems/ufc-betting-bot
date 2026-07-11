"""Recover historical moneyline gaps from BestFightOdds without mutating base CSVs.

This script:
1. Recomputes the current unresolved fight queue from local odds sources.
2. Uses BestFightOdds fighter pages to find the event page for each missing fight.
3. Parses bookmaker moneylines from the event page.
4. Writes recovered rows and unresolved rows to append-only CSV outputs.

It does not overwrite the main historical odds files. That is intentional so
parallel sessions can merge safely later.
"""

from __future__ import annotations

import argparse
import math
import re
import time
import unicodedata
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
HISTORICAL_DIR = RAW_DIR / "historical_odds"
HEADERS = {"User-Agent": "Mozilla/5.0"}
BFO_REQUEST_DELAY_SECONDS = 0.35

HISTORICAL_PATHS = (
    HISTORICAL_DIR / "historical_odds.csv",
    HISTORICAL_DIR / "historical_odds_pre2022_from_cleaned.csv",
)

ALIAS_GROUPS = [
    ["Rong Zhu", "Rongzhu", "Zhu Rong"],
    ["Alatengheili", "Heili Alateng"],
    ["Ariane da Silva", "Ariane Lipski", "Ariane Lipski da Silva"],
    ["Blood Diamond", "Mike Mathetha"],
    ["Ian Machado Garry", "Ian Garry"],
    ["Song Kenan", "Kenan Song"],
    ["CJ Vergara", "C.J. Vergara", "C J Vergara", "Cj Vergara"],
    ["Aoriqileng", "Aori Qileng"],
    ["King Green", "Bobby Green"],
    ["Viacheslav Borshchev", "Slava Claus", "Viacheslav Borschev"],
    ["Terrance McKinney", "Terrence McKinney"],
    ["Brendon Marotte", "Brendan Marotte"],
    ["Zachary Reese", "Zach Reese"],
    ["JJ Aldrich", "J.J. Aldrich"],
    ["Da'Mon Blackshear", "DaMon Blackshear"],
    ["Casey O'Neill", "Casey ONeill"],
    ["Don'Tale Mayes", "Dontale Mayes"],
    ["Veronica Hardy", "Veronica Macedo"],
    ["Ravena Oliveira", "Ravena Oliveira da Silva"],
    ["Montserrat Conejo Ruiz", "Montserrat Ruiz"],
    ["Dooho Choi", "Doo Ho Choi", "Choi Doo-Ho"],
    ["Shara Magomedov", "Sharapudin Magomedov"],
    ["Tecia Pennington", "Tecia Torres"],
    ["Elves Brener", "Elves Brenner"],
    ["Charles Radtke", "Charlie Radtke"],
    ["Daniel Lacerda", "Daniel da Silva", "Daniel da Silva Lacerda"],
    ["Bruno Silva", "Bruno Silva Blindado"],
]


def _build_alias_map() -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for group in ALIAS_GROUPS:
        for name in group:
            aliases.setdefault(name, [])
            for other in group:
                if other != name and other not in aliases[name]:
                    aliases[name].append(other)
    return aliases


ALIASES = _build_alias_map()


def norm(value) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(text.split())


def last(value) -> str:
    parts = norm(value).split()
    return parts[-1] if parts else ""


def parse_am(text) -> float | None:
    match = re.search(r"[+-]\d+", str(text))
    return float(match.group(0)) if match else None


def american_to_decimal(odds: float) -> float:
    if odds > 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / abs(odds))


def alias_names(name: str) -> list[str]:
    values = [name]
    for value in ALIASES.get(name, []):
        if value not in values:
            values.append(value)
    return values


def alias_norms(name: str) -> set[str]:
    return {norm(value) for value in alias_names(name)}


def has_num(value) -> bool:
    try:
        numeric = float(value)
    except Exception:
        return False
    return not math.isnan(numeric)


def date_fragments(event_date: str) -> list[str]:
    event_day = pd.Timestamp(event_date)
    fragments: list[str] = []
    for offset in (0, 1, -1):
        day = event_day + pd.Timedelta(days=offset)
        suffix = "th" if 11 <= day.day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day.day % 10, "th")
        fragments.append(f"{day.strftime('%b')} {day.day}{suffix} {day.year}")
    return fragments


@lru_cache(maxsize=None)
def get_soup(url: str) -> BeautifulSoup:
    time.sleep(BFO_REQUEST_DELAY_SECONDS)
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


@lru_cache(maxsize=None)
def search_fighter(query: str) -> list[tuple[str, str]]:
    time.sleep(BFO_REQUEST_DELAY_SECONDS)
    response = requests.get(
        "https://www.bestfightodds.com/search",
        params={"query": query},
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for link in soup.select('a[href*="/fighters/"]'):
        href = link.get("href", "")
        text = link.get_text(" ", strip=True)
        if not href or not text:
            continue
        url = f"https://www.bestfightodds.com{href}" if href.startswith("/") else href
        if url in seen:
            continue
        seen.add(url)
        results.append((text, url))
    return results


def candidate_fighter_urls(name: str) -> list[tuple[str, str]]:
    targets = alias_norms(name)
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for query in alias_names(name):
        for label, url in search_fighter(query):
            if url in seen:
                continue
            if norm(label) in targets:
                seen.add(url)
                results.append((label, url))
    return results[:8]


def row_odds_cells(row) -> list[float]:
    values: list[float] = []
    for cell in row.select("td.but-sg"):
        parsed = parse_am(cell.get_text(" ", strip=True))
        if parsed is not None:
            values.append(parsed)
    return values


def parse_event_page(event_url: str) -> list[tuple[str, str, list[float], list[float]]]:
    soup = get_soup(event_url)
    tables = soup.select("table.odds-table")
    if len(tables) < 2:
        return []

    name_rows = tables[0].select("tr")
    odds_rows = tables[1].select("tr")
    fights: list[tuple[str, str, list[float], list[float]]] = []
    limit = min(len(name_rows), len(odds_rows))
    i = 1
    while i < limit - 1:
        if "pr" in (name_rows[i].get("class") or []) or "pr" in (odds_rows[i].get("class") or []):
            i += 1
            continue
        if "pr" in (name_rows[i + 1].get("class") or []) or "pr" in (odds_rows[i + 1].get("class") or []):
            i += 1
            continue

        fighter_a = name_rows[i].get_text(" ", strip=True)
        fighter_b = name_rows[i + 1].get_text(" ", strip=True)
        odds_a = row_odds_cells(odds_rows[i])
        odds_b = row_odds_cells(odds_rows[i + 1])
        if fighter_a and fighter_b and odds_a and odds_b:
            fights.append((fighter_a, fighter_b, odds_a, odds_b))
            i += 2
            continue
        i += 1
    return fights


def find_event_via_fighter(fighter_name: str, opponent_name: str, event_date: str):
    opponent_aliases = alias_norms(opponent_name)
    fragments = date_fragments(event_date)

    for fighter_label, fighter_url in candidate_fighter_urls(fighter_name):
        soup = get_soup(fighter_url)
        current_event_url = None
        current_event_text = ""
        for row in soup.select("tr"):
            event_links = row.select('a[href*="/events/"]')
            if event_links:
                current_event_url = f"https://www.bestfightodds.com{event_links[0].get('href')}"
                current_event_text = " | ".join(link.get_text(" ", strip=True) for link in event_links)

            row_text = " | ".join(cell.get_text(" ", strip=True) for cell in row.select("th,td"))
            haystack = norm(row_text)
            if not any(alias in haystack for alias in opponent_aliases):
                continue

            combined_text = f"{current_event_text} | {row_text}"
            if current_event_url and any(fragment in combined_text for fragment in fragments):
                return {
                    "event_url": current_event_url,
                    "fighter_page_url": fighter_url,
                    "fighter_page_label": fighter_label,
                    "matched_row": row_text,
                }
    return None


def find_odds_from_event(event_url: str, fighter_a: str, fighter_b: str):
    fighter_a_aliases = alias_norms(fighter_a)
    fighter_b_aliases = alias_norms(fighter_b)
    fighter_a_last = {last(value) for value in fighter_a_aliases}
    fighter_b_last = {last(value) for value in fighter_b_aliases}

    for name_a, name_b, odds_a, odds_b in parse_event_page(event_url):
        norm_a = norm(name_a)
        norm_b = norm(name_b)
        if (norm_a in fighter_a_aliases and norm_b in fighter_b_aliases) or (
            last(norm_a) in fighter_a_last and last(norm_b) in fighter_b_last
        ):
            return name_a, name_b, odds_a, odds_b
        if (norm_b in fighter_a_aliases and norm_a in fighter_b_aliases) or (
            last(norm_b) in fighter_a_last and last(norm_a) in fighter_b_last
        ):
            return name_b, name_a, odds_b, odds_a
    return None


def load_true_missing_queue(start_date: str = "2014-01-01", end_date: str = "2023-12-31") -> pd.DataFrame:
    known_paths = [
        *HISTORICAL_PATHS,
        HISTORICAL_DIR / "historical_odds_bfo.csv",
        *sorted(HISTORICAL_DIR.glob("historical_odds_bfo_recovered_*.csv")),
    ]
    historical_parts = [
        pd.read_csv(path, parse_dates=["event_date"])[["event_date", "fighter_a", "fighter_b"]]
        for path in known_paths
        if path.exists()
    ]
    historical = (
        pd.concat(historical_parts, ignore_index=True)
        .dropna(subset=["event_date", "fighter_a", "fighter_b"])
    )
    known = {
        (
            pd.Timestamp(row.event_date).strftime("%Y-%m-%d"),
            tuple(sorted([norm(row.fighter_a), norm(row.fighter_b)])),
        )
        for row in historical.itertuples(index=False)
    }

    fights = pd.read_csv(REPO_ROOT / "data" / "processed" / "fights_cleaned.csv", parse_dates=["event_date"])
    fights = fights[(fights["event_date"] >= start_date) & (fights["event_date"] <= end_date)].copy()
    fights["event_date_str"] = fights["event_date"].dt.strftime("%Y-%m-%d")
    fights["pair_key"] = fights.apply(
        lambda row: tuple(sorted([norm(row["fighter_a"]), norm(row["fighter_b"])])),
        axis=1,
    )
    still_missing = fights[
        [(date_str, pair_key) not in known for date_str, pair_key in zip(fights["event_date_str"], fights["pair_key"])]
    ].copy()

    local_rows: list[tuple[str, str, str]] = []

    jansen = pd.read_csv(RAW_DIR / "jansen88_ufc_data.csv")
    jansen.columns = [column.strip() for column in jansen.columns]
    jansen["event_date"] = pd.to_datetime(jansen["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for row in jansen.dropna(subset=["event_date", "fighter1", "fighter2"]).itertuples(index=False):
        if has_num(getattr(row, "favourite_odds")) and has_num(getattr(row, "underdog_odds")):
            local_rows.append((str(row.event_date), norm(row.fighter1), norm(row.fighter2)))

    master = pd.read_csv(RAW_DIR / "ufc-master.csv")
    master["Date"] = pd.to_datetime(master["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for row in master.dropna(subset=["Date", "RedFighter", "BlueFighter"]).itertuples(index=False):
        if has_num(getattr(row, "RedOdds")) and has_num(getattr(row, "BlueOdds")):
            local_rows.append((str(row.Date), norm(row.RedFighter), norm(row.BlueFighter)))

    bfo = pd.read_csv(RAW_DIR / "bfo_iankotliar_odds.csv")
    bfo.columns = [column.strip() for column in bfo.columns]
    bfo["Card_Date"] = pd.to_datetime(bfo["Card_Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for row in bfo.dropna(subset=["Card_Date", "Winner", "Loser"]).itertuples(index=False):
        if has_num(getattr(row, "meanodds_win")) and has_num(getattr(row, "meanodds_lose")):
            local_rows.append((str(row.Card_Date), norm(row.Winner), norm(row.Loser)))

    pierce = pd.read_csv(RAW_DIR / "github_PierceHampton.csv")
    pierce["event_date"] = pd.to_datetime(pierce["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for row in pierce.dropna(subset=["event_date", "f_1_name", "f_2_name"]).itertuples(index=False):
        if has_num(getattr(row, "f_1_odds")) and has_num(getattr(row, "f_2_odds")):
            local_rows.append((str(row.event_date), norm(row.f_1_name), norm(row.f_2_name)))

    by_date: dict[str, list[tuple[str, str]]] = {}
    for event_date, fighter_a, fighter_b in local_rows:
        by_date.setdefault(event_date, []).append((fighter_a, fighter_b))

    queue_rows: list[dict[str, object]] = []
    for row in still_missing.itertuples(index=False):
        fighter_a_aliases = alias_norms(row.fighter_a)
        fighter_b_aliases = alias_norms(row.fighter_b)
        fighter_a_last = {last(value) for value in fighter_a_aliases}
        fighter_b_last = {last(value) for value in fighter_b_aliases}
        matched_local = False
        for local_a, local_b in by_date.get(row.event_date_str, []):
            if (
                (local_a in fighter_a_aliases and local_b in fighter_b_aliases)
                or (local_b in fighter_a_aliases and local_a in fighter_b_aliases)
                or (last(local_a) in fighter_a_last and last(local_b) in fighter_b_last)
                or (last(local_b) in fighter_a_last and last(local_a) in fighter_b_last)
            ):
                matched_local = True
                break
        if not matched_local:
            queue_rows.append(
                {
                    "event_date": row.event_date_str,
                    "event_name": row.event_name,
                    "fighter_a": row.fighter_a,
                    "fighter_b": row.fighter_b,
                }
            )

    return pd.DataFrame(queue_rows).drop_duplicates()


def recover_queue(queue: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    recovered_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []

    for item in queue.itertuples(index=False):
        try:
            match = find_event_via_fighter(item.fighter_a, item.fighter_b, item.event_date)
            if match is None:
                match = find_event_via_fighter(item.fighter_b, item.fighter_a, item.event_date)

            odds = find_odds_from_event(match["event_url"], item.fighter_a, item.fighter_b) if match else None
        except Exception as exc:
            unresolved_rows.append(
                {
                    "event_date": item.event_date,
                    "event_name": item.event_name,
                    "fighter_a": item.fighter_a,
                    "fighter_b": item.fighter_b,
                    "event_url": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        if odds is None:
            unresolved_rows.append(
                {
                    "event_date": item.event_date,
                    "event_name": item.event_name,
                    "fighter_a": item.fighter_a,
                    "fighter_b": item.fighter_b,
                    "event_url": match["event_url"] if match else "",
                    "error": "odds_not_found",
                }
            )
            continue

        _, _, odds_a, odds_b = odds
        decimal_a = float(np.mean([american_to_decimal(value) for value in odds_a]))
        decimal_b = float(np.mean([american_to_decimal(value) for value in odds_b]))
        implied_a = 1.0 / decimal_a
        implied_b = 1.0 / decimal_b
        total = implied_a + implied_b

        recovered_rows.append(
            {
                "event_date": item.event_date,
                "fighter_a": item.fighter_a,
                "fighter_b": item.fighter_b,
                "query_date": item.event_date,
                "offset_days": 0,
                "a_fair_prob": round(implied_a / total, 6),
                "b_fair_prob": round(implied_b / total, 6),
                "a_decimal_odds": round(decimal_a, 6),
                "b_decimal_odds": round(decimal_b, 6),
                "num_bookmakers": min(len(odds_a), len(odds_b)),
                "source_url": match["event_url"],
                "source_fighter_page": match["fighter_page_url"],
                "source_row_text": match["matched_row"],
            }
        )

    recovered = pd.DataFrame(recovered_rows)
    unresolved = pd.DataFrame(unresolved_rows)
    return recovered, unresolved


def default_output_paths() -> tuple[Path, Path]:
    stamp = date.today().isoformat().replace("-", "")
    return (
        HISTORICAL_DIR / f"historical_odds_bfo_recovered_{stamp}.csv",
        HISTORICAL_DIR / f"historical_odds_bfo_unresolved_{stamp}.csv",
    )


def main() -> int:
    default_recovered, default_unresolved = default_output_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered-output", type=Path, default=default_recovered)
    parser.add_argument("--unresolved-output", type=Path, default=default_unresolved)
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--end-date", default="2023-12-31")
    args = parser.parse_args()

    queue = load_true_missing_queue(start_date=args.start_date, end_date=args.end_date)
    recovered, unresolved = recover_queue(queue)

    args.recovered_output.parent.mkdir(parents=True, exist_ok=True)
    recovered.to_csv(args.recovered_output, index=False)
    unresolved.to_csv(args.unresolved_output, index=False)

    print(
        {
            "queue_fights": int(len(queue)),
            "recovered_fights": int(len(recovered)),
            "unresolved_fights": int(len(unresolved)),
            "recovered_output": str(args.recovered_output),
            "unresolved_output": str(args.unresolved_output),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
