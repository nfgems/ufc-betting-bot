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
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
import unicodedata
from datetime import date, datetime, timezone
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

# BFO currently lists prediction-market/exchange prices beside sportsbook
# moneylines.  Those markets can briefly show crossed or very thin prices and
# must not be treated as independent bookmaker quotes.  Only the reviewed
# sportsbook ID/name pairs below may enter a consensus.  An unfamiliar ID, or
# a familiar ID whose label has changed, is deliberately ignored until it has
# been reviewed.
EXCLUDED_MARKET_IDS = {28, 29}  # Polymarket, Kalshi
EXCLUDED_MARKET_NAMES = {"polymarket", "kalshi"}
APPROVED_SPORTSBOOK_MARKETS = {
    20: frozenset({"betway"}),
    21: frozenset({"fanduel"}),
    22: frozenset({"draftkings"}),
    23: frozenset({"betmgm"}),
    24: frozenset({"caesars"}),
    25: frozenset({"betrivers"}),
    26: frozenset({"unibet"}),
}
SPORTSBOOK_DISPLAY_NAMES = {
    20: "BetWay",
    21: "FanDuel",
    22: "DraftKings",
    23: "BetMGM",
    24: "Caesars",
    25: "BetRivers",
    26: "Unibet",
}
MIN_PAIRED_SPORTSBOOKS = 3
MIN_BOOK_OVERROUND = 0.95
MAX_BOOK_OVERROUND = 1.20
MAX_CONSENSUS_PROBABILITY_RANGE = 0.20

HISTORICAL_PATHS = (
    HISTORICAL_DIR / "historical_odds.csv",
    HISTORICAL_DIR / "historical_odds_pre2022_from_cleaned.csv",
)

ALIAS_GROUPS = [
    ["Rong Zhu", "Rongzhu", "Zhu Rong"],
    ["Alatengheili", "Heili Alateng"],
    ["Ariane da Silva", "Ariane Lipski", "Ariane Lipski da Silva"],
    ["Blood Diamond", "Mike Mathetha", "Mike Diamond"],
    ["Kazula Vargas", "Rodrigo Vargas"],
    ["Sumudaerji", "Sumudaerji Sumudaerji"],
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


_PAGE_FETCH_METADATA: dict[str, dict[str, str]] = {}


@lru_cache(maxsize=None)
def get_soup(url: str) -> BeautifulSoup:
    time.sleep(BFO_REQUEST_DELAY_SECONDS)
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    _PAGE_FETCH_METADATA[url] = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "content_sha256": hashlib.sha256(response.content).hexdigest(),
    }
    return BeautifulSoup(response.text, "lxml")


def _walk_json_values(value):
    """Yield every dict in a JSON-LD value, including nested ``@graph`` nodes."""
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json_values(child)


def _sports_event_date_evidence(soup: BeautifulSoup) -> tuple[list[str], int]:
    """Extract SportsEvent dates and count malformed JSON-LD/date evidence."""
    dates: set[str] = set()
    parse_errors = 0
    for script in soup.select('script[type="application/ld+json"]'):
        raw = script.string or script.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            parse_errors += 1
            continue
        for node in _walk_json_values(payload):
            node_types = node.get("@type", [])
            if isinstance(node_types, str):
                node_types = [node_types]
            if "SportsEvent" not in node_types:
                continue
            start_date = node.get("startDate")
            try:
                parsed = pd.Timestamp(start_date)
            except (TypeError, ValueError):
                parse_errors += 1
                continue
            if pd.isna(parsed):
                parse_errors += 1
                continue
            dates.add(parsed.date().isoformat())
    return sorted(dates), parse_errors


def _sports_event_start_dates(soup: BeautifulSoup) -> list[str]:
    dates, _ = _sports_event_date_evidence(soup)
    return dates


def event_page_metadata(event_url: str) -> dict[str, object]:
    """Return date and immutable-enough fetch evidence for one BFO event page."""
    soup = get_soup(event_url)
    fetch_metadata = _PAGE_FETCH_METADATA.get(event_url, {})
    serialized_page = soup.encode("utf-8")
    start_dates, date_parse_errors = _sports_event_date_evidence(soup)
    return {
        "url": event_url,
        "start_dates": start_dates,
        "date_parse_errors": date_parse_errors,
        "fetched_at_utc": fetch_metadata.get(
            "fetched_at_utc",
            datetime.now(timezone.utc).isoformat(),
        ),
        "content_sha256": fetch_metadata.get(
            "content_sha256",
            hashlib.sha256(serialized_page).hexdigest(),
        ),
    }


def _event_date_delta_days(requested_date: str, observed_date: str) -> int | None:
    try:
        requested_timestamp = pd.Timestamp(requested_date)
        observed_timestamp = pd.Timestamp(observed_date)
        if pd.isna(requested_timestamp) or pd.isna(observed_timestamp):
            return None
        requested = requested_timestamp.date()
        observed = observed_timestamp.date()
        return (observed - requested).days
    except (TypeError, ValueError):
        return None


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


def _bfo_market_id(value) -> int | None:
    """Extract BFO's bookmaker ID from a ``data-b`` or ``data-li`` value."""
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else None


def _market_label_keys(cell) -> set[str]:
    """Return compact labels exposed by one BFO market-header cell."""
    values = [cell.get_text(" ", strip=True)]
    values.extend(
        str(node.get(attribute) or "")
        for node in cell.select("img,a")
        for attribute in ("alt", "title")
    )
    return {norm(value).replace(" ", "") for value in values if norm(value)}


def _approved_market_ids(odds_table) -> set[int]:
    """Return only reviewed sportsbook columns with the expected BFO label."""
    header = odds_table.select_one("tr")
    if header is None:
        return set()

    observations: dict[int, list[bool]] = {}
    for cell in header.select("th,td"):
        market_id = _bfo_market_id(cell.get("data-b") or cell.get("data-li"))
        if market_id is None:
            continue
        labels = _market_label_keys(cell)
        expected_labels = APPROVED_SPORTSBOOK_MARKETS.get(market_id, frozenset())
        is_exchange = market_id in EXCLUDED_MARKET_IDS or any(
            exchange_name in label
            for exchange_name in EXCLUDED_MARKET_NAMES
            for label in labels
        )
        observations.setdefault(market_id, []).append(
            bool(expected_labels & labels) and not is_exchange
        )

    return {
        market_id
        for market_id, checks in observations.items()
        if market_id in APPROVED_SPORTSBOOK_MARKETS
        and market_id not in EXCLUDED_MARKET_IDS
        and checks
        and all(checks)
    }


def _bfo_quote_identity(value) -> tuple[int, int] | None:
    """Extract ``(side, matchup_id)`` from a BFO quote's ``data-li`` value."""
    parts = [int(part) for part in re.findall(r"\d+", str(value or ""))]
    if len(parts) < 3:
        return None
    return parts[1], parts[2]


def _odds_row_identity(row, *, approved_market_ids: set[int]) -> tuple[int, int] | None:
    identities: set[tuple[int, int]] = set()
    for cell in row.select("td.but-sg"):
        market_id = _bfo_market_id(cell.get("data-li") or cell.get("data-b"))
        if market_id not in approved_market_ids or parse_am(cell.get_text(" ", strip=True)) is None:
            continue
        identity = _bfo_quote_identity(cell.get("data-li"))
        if identity is None:
            return None
        identities.add(identity)
    return next(iter(identities)) if len(identities) == 1 else None


def _canonical_fighter_href(value) -> str | None:
    match = re.search(r"/fighters/[^/?#]+", str(value or ""), flags=re.IGNORECASE)
    return match.group(0).rstrip("/").casefold() if match else None


def _fighter_identity_from_row(row) -> tuple[str, str] | None:
    """Read one unambiguous fighter name/link, ignoring hidden admin bout IDs."""
    identities: dict[str, dict[str, str]] = {}
    for link in row.select("a[href]"):
        fighter_href = _canonical_fighter_href(link.get("href"))
        if fighter_href is None:
            continue
        name_node = link.select_one(".t-b-fcc") or link
        fighter_name = name_node.get_text(" ", strip=True)
        canonical_name = norm(fighter_name)
        if not canonical_name or not re.search(r"[a-z]", canonical_name):
            return None
        identities.setdefault(fighter_href, {})[canonical_name] = fighter_name

    if len(identities) != 1:
        return None
    fighter_href, names = next(iter(identities.items()))
    if len(names) != 1:
        return None
    return next(iter(names.values())), fighter_href


def _canonical_pair_orientation(
    candidate_a: str,
    candidate_b: str,
    requested_a: str,
    requested_b: str,
) -> int | None:
    """Return 1/direct or -1/reversed for one exact, alias-aware full pair."""
    candidate_a_norm = norm(candidate_a)
    candidate_b_norm = norm(candidate_b)
    requested_a_aliases = alias_norms(requested_a)
    requested_b_aliases = alias_norms(requested_b)
    direct = candidate_a_norm in requested_a_aliases and candidate_b_norm in requested_b_aliases
    reverse = candidate_b_norm in requested_a_aliases and candidate_a_norm in requested_b_aliases
    if direct == reverse:
        return None
    return 1 if direct else -1


def row_odds_cells(row, *, approved_market_ids: set[int]) -> dict[int, float]:
    """Parse one moneyline row keyed by BFO bookmaker ID.

    Keeping the ID is important: it lets us pair the two fighter prices from
    the same sportsbook before removing vig.  Positional lists silently pair
    the wrong books whenever one side has a missing cell.
    """
    values: dict[int, float] = {}
    for cell in row.select("td.but-sg"):
        market_id = _bfo_market_id(cell.get("data-li") or cell.get("data-b"))
        if market_id not in approved_market_ids or market_id in EXCLUDED_MARKET_IDS:
            continue
        parsed = parse_am(cell.get_text(" ", strip=True))
        if parsed is not None:
            values[market_id] = parsed
    return values


def sportsbook_consensus_details(
    odds_a: dict[int, float],
    odds_b: dict[int, float],
) -> tuple[dict[str, float | int] | None, list[dict[str, object]], str]:
    """Build a robust no-vig consensus from paired sportsbook quotes.

    Each book is de-vigged independently, then the median fair probability is
    used.  This avoids the large distortions caused by averaging decimal odds
    first and bounds the influence of any remaining bad sportsbook quote.
    """
    paired: list[tuple[float, float, float]] = []
    quote_records: list[dict[str, object]] = []
    approved_ids = set(APPROVED_SPORTSBOOK_MARKETS) - EXCLUDED_MARKET_IDS
    for market_id in sorted(set(odds_a) & set(odds_b) & approved_ids):
        decimal_a = american_to_decimal(odds_a[market_id])
        decimal_b = american_to_decimal(odds_b[market_id])
        implied_a = 1.0 / decimal_a
        implied_b = 1.0 / decimal_b
        total = implied_a + implied_b
        fair_a = implied_a / total
        accepted = MIN_BOOK_OVERROUND <= total <= MAX_BOOK_OVERROUND
        quote_records.append(
            {
                "market_id": market_id,
                "book_name": SPORTSBOOK_DISPLAY_NAMES[market_id],
                "a_american": float(odds_a[market_id]),
                "b_american": float(odds_b[market_id]),
                "a_decimal": decimal_a,
                "b_decimal": decimal_b,
                "overround": total,
                "a_fair_prob": fair_a,
                "accepted": accepted,
                "rejection_reason": "" if accepted else "book_overround_out_of_range",
            }
        )
        if accepted:
            paired.append((decimal_a, decimal_b, fair_a))

    if len(paired) < MIN_PAIRED_SPORTSBOOKS:
        return None, quote_records, "fewer_than_three_valid_paired_sportsbooks"

    fair_probabilities = np.asarray([value[2] for value in paired], dtype=float)
    if float(np.max(fair_probabilities) - np.min(fair_probabilities)) > MAX_CONSENSUS_PROBABILITY_RANGE:
        return None, quote_records, "sportsbook_consensus_probability_range_exceeded"

    fair_a = float(np.median(fair_probabilities))
    return (
        {
            "a_fair_prob": fair_a,
            "b_fair_prob": 1.0 - fair_a,
            "a_decimal_odds": float(np.median([value[0] for value in paired])),
            "b_decimal_odds": float(np.median([value[1] for value in paired])),
            "num_bookmakers": len(paired),
        },
        quote_records,
        "",
    )


def sportsbook_consensus(
    odds_a: dict[int, float],
    odds_b: dict[int, float],
) -> dict[str, float | int] | None:
    consensus, _, _ = sportsbook_consensus_details(odds_a, odds_b)
    return consensus


def parse_event_page_details(event_url: str) -> list[dict[str, object]]:
    soup = get_soup(event_url)
    tables = soup.select("table.odds-table")
    if len(tables) < 2:
        return []

    name_rows = tables[0].select("tr")
    odds_rows = tables[1].select("tr")
    fights: list[dict[str, object]] = []
    approved_market_ids = _approved_market_ids(tables[1])
    limit = min(len(name_rows), len(odds_rows))
    i = 1
    while i < limit - 1:
        if "pr" in (name_rows[i].get("class") or []) or "pr" in (odds_rows[i].get("class") or []):
            i += 1
            continue
        if "pr" in (name_rows[i + 1].get("class") or []) or "pr" in (odds_rows[i + 1].get("class") or []):
            i += 1
            continue

        identity_a = _fighter_identity_from_row(name_rows[i])
        identity_b = _fighter_identity_from_row(name_rows[i + 1])
        row_identity_a = _odds_row_identity(odds_rows[i], approved_market_ids=approved_market_ids)
        row_identity_b = _odds_row_identity(odds_rows[i + 1], approved_market_ids=approved_market_ids)
        if (
            identity_a is None
            or identity_b is None
            or identity_a[1] == identity_b[1]
            or norm(identity_a[0]) == norm(identity_b[0])
            or row_identity_a is None
            or row_identity_b is None
            or row_identity_a[0] != 1
            or row_identity_b[0] != 2
            or row_identity_a[1] != row_identity_b[1]
        ):
            i += 1
            continue

        fighter_a, fighter_a_href = identity_a
        fighter_b, fighter_b_href = identity_b
        odds_a = row_odds_cells(odds_rows[i], approved_market_ids=approved_market_ids)
        odds_b = row_odds_cells(odds_rows[i + 1], approved_market_ids=approved_market_ids)
        if odds_a and odds_b:
            fights.append(
                {
                    "fighter_a": fighter_a,
                    "fighter_b": fighter_b,
                    "fighter_a_href": fighter_a_href,
                    "fighter_b_href": fighter_b_href,
                    "matchup_id": row_identity_a[1],
                    "odds_a": odds_a,
                    "odds_b": odds_b,
                }
            )
            i += 2
            continue
        i += 1
    return fights


def parse_event_page(event_url: str) -> list[tuple[str, str, dict[int, float], dict[int, float]]]:
    """Return the stable public tuple shape used by the recovery/test callers."""
    return [
        (
            str(fight["fighter_a"]),
            str(fight["fighter_b"]),
            dict(fight["odds_a"]),
            dict(fight["odds_b"]),
        )
        for fight in parse_event_page_details(event_url)
    ]


def find_event_via_fighter(fighter_name: str, opponent_name: str, event_date: str):
    fighter_aliases = alias_norms(fighter_name)
    opponent_aliases = alias_norms(opponent_name)
    fragments = {fragment.casefold() for fragment in date_fragments(event_date)}
    matches: dict[str, dict[str, str]] = {}

    for fighter_label, fighter_url in candidate_fighter_urls(fighter_name):
        if norm(fighter_label) not in fighter_aliases:
            continue
        soup = get_soup(fighter_url)
        table = soup.select_one("table.team-stats-table")
        if table is None:
            continue

        current_event_url = None
        current_event_text = ""
        pending_fighter_row: tuple[str, str] | None = None
        fighter_url_key = _canonical_fighter_href(fighter_url)
        for row in table.select("tr"):
            row_text = " | ".join(cell.get_text(" ", strip=True) for cell in row.select("th,td"))
            event_links = row.select('a[href*="/events/"]')
            if event_links:
                href = str(event_links[0].get("href") or "").strip()
                event_url = href if href.startswith("http") else f"https://www.bestfightodds.com{href}"
                if event_url != current_event_url:
                    current_event_url = event_url
                    current_event_text = row_text
                elif row_text:
                    current_event_text = f"{current_event_text} | {row_text}"

            classes = row.get("class") or []
            if "main-row" in classes:
                identity = _fighter_identity_from_row(row)
                if (
                    current_event_url
                    and identity is not None
                    and identity[1] == fighter_url_key
                    and norm(identity[0]) in fighter_aliases
                ):
                    pending_fighter_row = (current_event_url, f"{current_event_text} | {row_text}")
                else:
                    pending_fighter_row = None
                continue

            if pending_fighter_row is None:
                continue

            opponent_identity = _fighter_identity_from_row(row)
            candidate_event_url, candidate_event_text = pending_fighter_row
            pending_fighter_row = None
            if opponent_identity is None or norm(opponent_identity[0]) not in opponent_aliases:
                continue

            combined_text = f"{candidate_event_text} | {row_text}".casefold()
            if any(fragment in combined_text for fragment in fragments):
                matches[candidate_event_url] = {
                    "event_url": candidate_event_url,
                    "fighter_page_url": fighter_url,
                    "fighter_page_label": fighter_label,
                    "matched_row": row_text,
                }

    return next(iter(matches.values())) if len(matches) == 1 else None


def _safe_merge_oriented_matches(
    matches: list[tuple[str, str, dict[int, float], dict[int, float]]],
    requested_a: str,
    requested_b: str,
) -> tuple[tuple[str, str, dict[int, float], dict[int, float]] | None, str, bool]:
    """Merge alias-split BFO rows only when their paired books cannot collide."""
    if not matches:
        return None, "exact_fighter_pair_not_found", False
    if len(matches) == 1:
        return matches[0], "", False

    label_pairs = [(norm(match[0]), norm(match[1])) for match in matches]
    if len(set(label_pairs)) != len(label_pairs):
        return None, "duplicate_exact_fighter_pair", False

    approved_ids = set(APPROVED_SPORTSBOOK_MARKETS) - EXCLUDED_MARKET_IDS
    used_market_ids: set[int] = set()
    merged_a: dict[int, float] = {}
    merged_b: dict[int, float] = {}
    for _, _, odds_a, odds_b in matches:
        paired_ids = set(odds_a) & set(odds_b) & approved_ids
        if not paired_ids:
            return None, "alias_split_row_has_no_paired_sportsbook", False
        if paired_ids & used_market_ids:
            return None, "alias_split_rows_overlap_sportsbook_ids", False
        used_market_ids.update(paired_ids)
        for market_id in paired_ids:
            merged_a[market_id] = odds_a[market_id]
            merged_b[market_id] = odds_b[market_id]

    return (requested_a, requested_b, merged_a, merged_b), "", True


def _oriented_event_details(
    event_url: str,
    fighter_a: str,
    fighter_b: str,
) -> list[dict[str, object]]:
    oriented: list[dict[str, object]] = []
    approved_ids = set(APPROVED_SPORTSBOOK_MARKETS) - EXCLUDED_MARKET_IDS
    for fight in parse_event_page_details(event_url):
        name_a = str(fight["fighter_a"])
        name_b = str(fight["fighter_b"])
        orientation = _canonical_pair_orientation(name_a, name_b, fighter_a, fighter_b)
        if orientation is None:
            continue
        if orientation == 1:
            oriented_a = name_a
            oriented_b = name_b
            href_a = fight.get("fighter_a_href")
            href_b = fight.get("fighter_b_href")
            odds_a = dict(fight["odds_a"])
            odds_b = dict(fight["odds_b"])
            orientation_label = "direct"
        else:
            oriented_a = name_b
            oriented_b = name_a
            href_a = fight.get("fighter_b_href")
            href_b = fight.get("fighter_a_href")
            odds_a = dict(fight["odds_b"])
            odds_b = dict(fight["odds_a"])
            orientation_label = "reversed"
        oriented.append(
            {
                "fighter_a": oriented_a,
                "fighter_b": oriented_b,
                "fighter_a_href": href_a,
                "fighter_b_href": href_b,
                "matchup_id": fight.get("matchup_id"),
                "orientation": orientation_label,
                "paired_market_ids": sorted(set(odds_a) & set(odds_b) & approved_ids),
            }
        )
    return oriented


def find_odds_from_event(
    event_url: str,
    fighter_a: str,
    fighter_b: str,
    event_date: str | None = None,
    *,
    observation: dict[str, object] | None = None,
):
    if observation is not None or event_date is not None:
        page_metadata = event_page_metadata(event_url)
        start_dates = list(page_metadata["start_dates"])
        start_date = start_dates[0] if len(start_dates) == 1 else None
        page_metadata["start_date"] = start_date
        if observation is not None:
            observation["event_page"] = page_metadata
            observation["requested_fighters"] = {
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
            }
        if event_date is not None:
            if start_date is None or int(page_metadata["date_parse_errors"]) > 0:
                if observation is not None:
                    observation["match_error"] = (
                        "event_page_date_missing_malformed_or_ambiguous"
                    )
                return None
            date_delta = _event_date_delta_days(event_date, start_date)
            page_metadata["requested_event_date"] = event_date
            page_metadata["date_delta_days"] = date_delta
            if date_delta is None:
                if observation is not None:
                    observation["match_error"] = "invalid_requested_or_event_page_date"
                return None
            if abs(date_delta) > 1:
                if observation is not None:
                    observation["match_error"] = "event_page_date_mismatch"
                return None

    matches: list[tuple[str, str, dict[int, float], dict[int, float]]] = []
    for name_a, name_b, odds_a, odds_b in parse_event_page(event_url):
        orientation = _canonical_pair_orientation(name_a, name_b, fighter_a, fighter_b)
        if orientation == 1:
            matches.append((name_a, name_b, odds_a, odds_b))
        elif orientation == -1:
            matches.append((name_b, name_a, odds_b, odds_a))

    result, error, merged = _safe_merge_oriented_matches(matches, fighter_a, fighter_b)
    if observation is not None:
        observation["matched_bfo_rows"] = _oriented_event_details(
            event_url,
            fighter_a,
            fighter_b,
        )
        observation["alias_split_merge_applied"] = merged
        observation["match_error"] = error
    return result


def _git_bytes(*args: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return b""
    return completed.stdout if completed.returncode == 0 else b""


@lru_cache(maxsize=1)
def parser_fingerprint() -> dict[str, object]:
    parser_paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "scripts" / "revalidate_bfo_recovery_file.py",
    ]
    file_hashes: dict[str, str] = {}
    fingerprint_material = bytearray()
    for path in parser_paths:
        if not path.exists():
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        content = path.read_bytes()
        file_hashes[relative] = hashlib.sha256(content).hexdigest()
        fingerprint_material.extend(relative.encode("utf-8"))
        fingerprint_material.extend(b"\0")
        fingerprint_material.extend(content)
        fingerprint_material.extend(b"\0")

    relative_paths = list(file_hashes)
    dirty_diff = _git_bytes("diff", "--binary", "HEAD", "--", *relative_paths)
    dirty_status = _git_bytes("status", "--short", "--", *relative_paths)
    fingerprint_material.extend(dirty_diff)
    fingerprint_material.extend(dirty_status)
    git_head = _git_bytes("rev-parse", "HEAD").decode("ascii", errors="ignore").strip()
    return {
        "git_head": git_head or None,
        "file_sha256": file_hashes,
        "dirty_diff_sha256": hashlib.sha256(fingerprint_material).hexdigest(),
    }


def _json_clean(value):
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _provenance_text(value) -> str:
    cleaned = _json_clean(value)
    return "" if cleaned is None else str(cleaned)


def build_bfo_provenance_record(
    row: dict[str, object],
    *,
    input_batch: str,
    output_batch: str,
    event_url: str,
    observation: dict[str, object],
    paired_quotes: list[dict[str, object]],
    consensus: dict[str, float | int] | None,
    decision: str,
    rejection_reason: str,
) -> dict[str, object]:
    """Create one self-contained, machine-reconcilable BFO audit record."""
    event_date = _provenance_text(row.get("event_date"))
    query_date = _provenance_text(row.get("query_date")) or event_date
    offset_days = row.get("offset_days", 0)
    csv_values = None
    if consensus is not None:
        csv_values = {
            "a_fair_prob": round(float(consensus["a_fair_prob"]), 6),
            "b_fair_prob": round(float(consensus["b_fair_prob"]), 6),
            "a_decimal_odds": round(float(consensus["a_decimal_odds"]), 6),
            "b_decimal_odds": round(float(consensus["b_decimal_odds"]), 6),
            "num_bookmakers": int(consensus["num_bookmakers"]),
        }
    event_page = dict(observation.get("event_page") or {"url": event_url})
    event_page.setdefault("url", event_url)
    event_page.setdefault("start_date", None)
    event_page.setdefault("fetched_at_utc", None)
    event_page.setdefault("content_sha256", None)
    record = {
        "schema_version": 1,
        "input_batch": input_batch,
        "output_batch": output_batch,
        "recovery_key": {
            "event_date": event_date,
            "fighter_a": _provenance_text(row.get("fighter_a")),
            "fighter_b": _provenance_text(row.get("fighter_b")),
            "query_date": query_date,
            "offset_days": offset_days,
        },
        "event_page": event_page,
        "requested_fighters": observation.get(
            "requested_fighters",
            {
                "fighter_a": _provenance_text(row.get("fighter_a")),
                "fighter_b": _provenance_text(row.get("fighter_b")),
            },
        ),
        "matched_bfo_rows": observation.get("matched_bfo_rows", []),
        "alias_split_merge_applied": bool(
            observation.get("alias_split_merge_applied", False)
        ),
        "paired_quotes": paired_quotes,
        "consensus": consensus,
        "csv_values": csv_values,
        "thresholds": {
            "minimum_paired_sportsbooks": MIN_PAIRED_SPORTSBOOKS,
            "minimum_book_overround": MIN_BOOK_OVERROUND,
            "maximum_book_overround": MAX_BOOK_OVERROUND,
            "maximum_consensus_probability_range": MAX_CONSENSUS_PROBABILITY_RANGE,
            "maximum_event_date_delta_days": 1,
        },
        "decision": decision,
        "rejection_reason": rejection_reason,
        "parser": parser_fingerprint(),
    }
    return _json_clean(record)


def write_jsonl_atomically(records: list[dict[str, object]], path: Path) -> None:
    """Atomically write JSONL and verify that every record round-trips exactly."""
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(_json_clean(record), sort_keys=True, allow_nan=False))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        observed = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        expected = [_json_clean(record) for record in records]
        if observed != expected:
            raise RuntimeError(f"failed to verify written JSONL {path}")
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


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
        matched_local = False
        for local_a, local_b in by_date.get(row.event_date_str, []):
            if (
                (local_a in fighter_a_aliases and local_b in fighter_b_aliases)
                or (local_b in fighter_a_aliases and local_a in fighter_b_aliases)
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


def recover_queue(
    queue: pd.DataFrame,
    *,
    provenance_records: list[dict[str, object]] | None = None,
    input_batch: str = "computed_true_missing_queue",
    output_batch: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    recovered_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []

    for item in queue.itertuples(index=False):
        source_row = dict(item._asdict())
        event_date = pd.Timestamp(item.event_date).strftime("%Y-%m-%d")
        source_row["event_date"] = event_date
        source_row.setdefault("query_date", event_date)
        source_row.setdefault("offset_days", 0)
        observation: dict[str, object] = {
            "requested_fighters": {
                "fighter_a": str(item.fighter_a),
                "fighter_b": str(item.fighter_b),
            }
        }
        paired_quotes: list[dict[str, object]] = []
        consensus: dict[str, float | int] | None = None
        match = None
        try:
            match = find_event_via_fighter(item.fighter_a, item.fighter_b, event_date)
            if match is None:
                match = find_event_via_fighter(item.fighter_b, item.fighter_a, event_date)

            odds = (
                find_odds_from_event(
                    match["event_url"],
                    item.fighter_a,
                    item.fighter_b,
                    event_date,
                    observation=observation,
                )
                if match
                else None
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            unresolved_rows.append(
                {
                    "event_date": event_date,
                    "event_name": item.event_name,
                    "fighter_a": item.fighter_a,
                    "fighter_b": item.fighter_b,
                    "event_url": match["event_url"] if match else "",
                    "error": error,
                }
            )
            if provenance_records is not None:
                provenance_records.append(
                    build_bfo_provenance_record(
                        source_row,
                        input_batch=input_batch,
                        output_batch=output_batch,
                        event_url=match["event_url"] if match else "",
                        observation=observation,
                        paired_quotes=paired_quotes,
                        consensus=consensus,
                        decision="rejected",
                        rejection_reason=error,
                    )
                )
            continue

        if odds is None:
            error = str(
                observation.get("match_error")
                or ("event_not_found" if match is None else "odds_not_found")
            )
            unresolved_rows.append(
                {
                    "event_date": event_date,
                    "event_name": item.event_name,
                    "fighter_a": item.fighter_a,
                    "fighter_b": item.fighter_b,
                    "event_url": match["event_url"] if match else "",
                    "error": error,
                }
            )
            if provenance_records is not None:
                provenance_records.append(
                    build_bfo_provenance_record(
                        source_row,
                        input_batch=input_batch,
                        output_batch=output_batch,
                        event_url=match["event_url"] if match else "",
                        observation=observation,
                        paired_quotes=paired_quotes,
                        consensus=consensus,
                        decision="rejected",
                        rejection_reason=error,
                    )
                )
            continue

        _, _, odds_a, odds_b = odds
        consensus, paired_quotes, consensus_error = sportsbook_consensus_details(
            odds_a,
            odds_b,
        )
        if consensus is None:
            unresolved_rows.append(
                {
                    "event_date": event_date,
                    "event_name": item.event_name,
                    "fighter_a": item.fighter_a,
                    "fighter_b": item.fighter_b,
                    "event_url": match["event_url"],
                    "error": consensus_error,
                }
            )
            if provenance_records is not None:
                provenance_records.append(
                    build_bfo_provenance_record(
                        source_row,
                        input_batch=input_batch,
                        output_batch=output_batch,
                        event_url=match["event_url"],
                        observation=observation,
                        paired_quotes=paired_quotes,
                        consensus=consensus,
                        decision="rejected",
                        rejection_reason=consensus_error,
                    )
                )
            continue

        recovered_row = {
            "event_date": event_date,
            "fighter_a": item.fighter_a,
            "fighter_b": item.fighter_b,
            "query_date": event_date,
            "offset_days": 0,
            "a_fair_prob": round(float(consensus["a_fair_prob"]), 6),
            "b_fair_prob": round(float(consensus["b_fair_prob"]), 6),
            "a_decimal_odds": round(float(consensus["a_decimal_odds"]), 6),
            "b_decimal_odds": round(float(consensus["b_decimal_odds"]), 6),
            "num_bookmakers": int(consensus["num_bookmakers"]),
            "source_url": match["event_url"],
            "source_fighter_page": match["fighter_page_url"],
            "source_row_text": match["matched_row"],
        }
        recovered_rows.append(recovered_row)
        if provenance_records is not None:
            provenance_records.append(
                build_bfo_provenance_record(
                    recovered_row,
                    input_batch=input_batch,
                    output_batch=output_batch,
                    event_url=match["event_url"],
                    observation=observation,
                    paired_quotes=paired_quotes,
                    consensus=consensus,
                    decision="accepted",
                    rejection_reason="",
                )
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


def default_provenance_path(recovered_output: Path) -> Path:
    return recovered_output.with_name(f"{recovered_output.stem}.provenance.jsonl")


def main() -> int:
    default_recovered, default_unresolved = default_output_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered-output", type=Path, default=default_recovered)
    parser.add_argument("--unresolved-output", type=Path, default=default_unresolved)
    parser.add_argument(
        "--provenance-output",
        type=Path,
        help=(
            "Companion JSONL audit ledger. Defaults beside recovered-output with "
            "a .provenance.jsonl suffix."
        ),
    )
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--end-date", default="2023-12-31")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help=(
            "Permit overwriting an existing recovered-output file. Off by default: "
            "recovered batches are append-only inputs to load_all_historical_odds, and "
            "a rerun overwriting an earlier batch with a smaller (possibly empty) "
            "result would silently destroy recovered odds."
        ),
    )
    args = parser.parse_args()
    provenance_output = args.provenance_output or default_provenance_path(
        args.recovered_output
    )

    existing_append_only_outputs = [
        path
        for path in (args.recovered_output, provenance_output)
        if path.exists()
    ]
    if existing_append_only_outputs and not args.allow_overwrite:
        parser.error(
            "Refusing to overwrite existing recovered/provenance batch "
            f"{existing_append_only_outputs[0]}; "
            "pick a new output name or pass --allow-overwrite."
        )
    resolved_outputs = {
        path.expanduser().resolve(strict=False)
        for path in (args.recovered_output, args.unresolved_output, provenance_output)
    }
    if len(resolved_outputs) != 3:
        parser.error("recovered, unresolved, and provenance outputs must be distinct")

    queue = load_true_missing_queue(start_date=args.start_date, end_date=args.end_date)
    provenance_records: list[dict[str, object]] = []
    recovered, unresolved = recover_queue(
        queue,
        provenance_records=provenance_records,
        output_batch=str(args.recovered_output),
    )

    args.recovered_output.parent.mkdir(parents=True, exist_ok=True)
    recovered.to_csv(args.recovered_output, index=False)
    unresolved.to_csv(args.unresolved_output, index=False)
    write_jsonl_atomically(provenance_records, provenance_output)

    print(
        {
            "queue_fights": int(len(queue)),
            "recovered_fights": int(len(recovered)),
            "unresolved_fights": int(len(unresolved)),
            "recovered_output": str(args.recovered_output),
            "unresolved_output": str(args.unresolved_output),
            "provenance_output": str(provenance_output),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
