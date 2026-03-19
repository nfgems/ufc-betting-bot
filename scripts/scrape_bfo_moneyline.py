"""Scrape BestFightOdds.com for historical moneyline odds.

Scrapes event pages to extract multi-bookmaker moneyline consensus odds
for UFC fights. Designed for pre-2022 backfill where The Odds API has no
coverage.

Output: data/raw/historical_odds/historical_odds_bfo.csv
Schema matches historical_odds.csv:
  event_date, fighter_a, fighter_b, query_date, offset_days,
  a_fair_prob, b_fair_prob, a_decimal_odds, b_decimal_odds, num_bookmakers
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
REQUEST_DELAY = 2.0
OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "historical_odds" / "historical_odds_bfo.csv"
CHECKPOINT_PATH = REPO_ROOT / "data" / "raw" / "historical_odds" / "bfo_scrape_checkpoint.json"
EVENTS_CACHE_PATH = REPO_ROOT / "data" / "raw" / "historical_odds" / "bfo_event_urls.json"
_PARSED_EVENT_PAGE_CACHE: dict[str, list[dict]] = {}

OUTPUT_COLUMNS = [
    "event_date", "fighter_a", "fighter_b", "query_date", "offset_days",
    "a_fair_prob", "b_fair_prob", "a_decimal_odds", "b_decimal_odds",
    "num_bookmakers", "event_name", "source_url",
]


def _get(url: str, params: dict | None = None, retries: int = 3) -> requests.Response:
    """GET with retries and rate limiting."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                logger.warning("Rate limited, waiting %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp
        except requests.RequestException as exc:
            if attempt < retries - 1:
                logger.warning("Request failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(5 * (attempt + 1))
            else:
                raise
    raise RuntimeError(f"Failed after {retries} retries: {url}")


def _american_to_decimal(odds: float) -> float:
    if odds > 0:
        return (odds / 100.0) + 1.0
    elif odds < 0:
        return (100.0 / abs(odds)) + 1.0
    return np.nan


def _parse_american_odds(text: str) -> float | None:
    """Parse American odds from BFO cell text, stripping arrows etc."""
    cleaned = re.sub(r"[▲▼↑↓\s]", "", text.strip())
    match = re.match(r"^([+-]?\d+)$", cleaned)
    if match:
        return float(match.group(1))
    return None


def _normalize_event_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _exact_ufc_number_score(event_name: str, candidate_text: str) -> int:
    match = re.search(r"\bufc\s+(\d+)\b", str(event_name), flags=re.IGNORECASE)
    if not match:
        return 0
    event_no = match.group(1)
    candidate_norm = _normalize_event_text(candidate_text)
    exact_pattern = rf"\bufc\s+{re.escape(event_no)}\b"
    if re.search(exact_pattern, candidate_norm):
        return 100
    # Explicitly reject substring collisions like UFC 2 -> UFC 200.
    if "ufc" in candidate_norm:
        return -100
    return -100


def search_bfo_events(query: str) -> list[tuple[str, str]]:
    """Search BFO and return list of (url, title) for event pages."""
    try:
        resp = _get("https://www.bestfightodds.com/search", params={"query": query})
    except Exception as exc:
        logger.warning("BFO search failed for '%s': %s", query, exc)
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    results = []
    seen = set()
    for link in soup.select("a[href*='/events/']"):
        href = link.get("href", "")
        if not href or href in ("/events", "/events/"):
            continue
        full_url = href if href.startswith("http") else f"https://www.bestfightodds.com{href}"
        if full_url in seen:
            continue
        seen.add(full_url)
        title = link.get_text(strip=True)
        results.append((full_url, title))
    return results


def parse_bfo_event_page(url: str) -> list[dict]:
    """Parse a BFO event page and extract moneyline odds for all fights.

    Returns list of dicts with keys:
      fighter_a, fighter_b, a_decimal_odds, b_decimal_odds,
      a_fair_prob, b_fair_prob, num_bookmakers
    """
    if url in _PARSED_EVENT_PAGE_CACHE:
        return list(_PARSED_EVENT_PAGE_CACHE[url])

    try:
        resp = _get(url)
    except Exception as exc:
        logger.warning("Failed to fetch BFO event page %s: %s", url, exc)
        _PARSED_EVENT_PAGE_CACHE[url] = []
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    tables = soup.select("table.odds-table")
    if len(tables) < 2:
        logger.warning("No odds table found at %s", url)
        _PARSED_EVENT_PAGE_CACHE[url] = []
        return []

    # Second table is the main odds table
    table = tables[1]
    rows = table.select("tr")

    fights = []
    i = 0
    while i < len(rows):
        row = rows[i]
        row_class = row.get("class", [])

        # Skip header rows, prop rows, and empty rows
        if "pr" in row_class:
            i += 1
            continue

        cells = row.select("td, th")
        if not cells:
            i += 1
            continue

        # Check if this is a fighter name row (first cell has fighter name, no data-b)
        first_cell = cells[0]
        first_text = first_cell.get_text(strip=True)

        # Header row check (has data-b attributes in non-first cells)
        if len(cells) > 1 and cells[1].get("data-b"):
            i += 1
            continue

        # Skip empty/separator rows
        if not first_text or first_text.isspace():
            i += 1
            continue

        # This should be fighter A row - check if next non-prop row is fighter B
        fighter_a_name = first_text
        fighter_a_odds_cells = cells[1:]  # odds cells

        # Find next non-prop, non-empty row for fighter B
        j = i + 1
        fighter_b_row = None
        while j < len(rows):
            next_row = rows[j]
            next_class = next_row.get("class", [])
            if "pr" in next_class:
                j += 1
                continue
            next_cells = next_row.select("td, th")
            if not next_cells:
                j += 1
                continue
            next_text = next_cells[0].get_text(strip=True)
            if not next_text or next_text.isspace():
                j += 1
                continue
            # Check if it has data-b (header row)
            if len(next_cells) > 1 and next_cells[1].get("data-b"):
                j += 1
                continue
            fighter_b_row = next_row
            break

        if fighter_b_row is None:
            i += 1
            continue

        fighter_b_cells = fighter_b_row.select("td, th")
        fighter_b_name = fighter_b_cells[0].get_text(strip=True)
        fighter_b_odds_cells = fighter_b_cells[1:]

        # Parse odds from each bookmaker column
        a_odds_list = []
        b_odds_list = []
        for a_cell, b_cell in zip(fighter_a_odds_cells, fighter_b_odds_cells):
            a_text = a_cell.get_text(strip=True)
            b_text = b_cell.get_text(strip=True)
            a_odds = _parse_american_odds(a_text)
            b_odds = _parse_american_odds(b_text)
            if a_odds is not None and b_odds is not None:
                a_odds_list.append(a_odds)
                b_odds_list.append(b_odds)

        if a_odds_list and b_odds_list:
            # Compute consensus (average) decimal odds
            a_dec = np.mean([_american_to_decimal(o) for o in a_odds_list])
            b_dec = np.mean([_american_to_decimal(o) for o in b_odds_list])

            # Fair probabilities (remove vig)
            a_implied = 1.0 / a_dec
            b_implied = 1.0 / b_dec
            total = a_implied + b_implied
            a_fair = a_implied / total
            b_fair = b_implied / total

            fights.append({
                "fighter_a": fighter_a_name,
                "fighter_b": fighter_b_name,
                "a_decimal_odds": round(a_dec, 6),
                "b_decimal_odds": round(b_dec, 6),
                "a_fair_prob": round(a_fair, 6),
                "b_fair_prob": round(b_fair, 6),
                "num_bookmakers": len(a_odds_list),
            })

        # Skip past fighter B and its props
        i = j + 1
        # Skip trailing props for this fight
        while i < len(rows):
            check_row = rows[i]
            if "pr" in (check_row.get("class", []) or []):
                i += 1
            else:
                break

    _PARSED_EVENT_PAGE_CACHE[url] = fights
    return list(fights)


def _normalize_for_match(name: str) -> str:
    """Normalize a fighter name for fuzzy matching."""
    import unicodedata
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"[^a-zA-Z\s]", "", name)
    return " ".join(name.lower().split())


def _names_match(name1: str, name2: str) -> bool:
    """Check if two fighter names refer to the same person."""
    n1 = _normalize_for_match(name1)
    n2 = _normalize_for_match(name2)
    if n1 == n2:
        return True
    # Check last name match + first initial
    tokens1 = n1.split()
    tokens2 = n2.split()
    if tokens1 and tokens2 and tokens1[-1] == tokens2[-1]:
        return True
    return False


def _count_target_fight_matches(target_fights: list[dict], candidate_fights: list[dict]) -> int:
    matches = 0
    for target in target_fights:
        target_a = str(target.get("fighter_a", ""))
        target_b = str(target.get("fighter_b", ""))
        for candidate in candidate_fights:
            cand_a = str(candidate.get("fighter_a", ""))
            cand_b = str(candidate.get("fighter_b", ""))
            if (
                _names_match(target_a, cand_a) and _names_match(target_b, cand_b)
            ) or (
                _names_match(target_a, cand_b) and _names_match(target_b, cand_a)
            ):
                matches += 1
                break
    return matches


def _load_event_url_cache() -> dict[str, str]:
    if EVENTS_CACHE_PATH.exists():
        try:
            payload = json.loads(EVENTS_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return {str(k): str(v) for k, v in payload.items()}
        except Exception as exc:
            logger.warning("Failed to load BFO event URL cache: %s", exc)
    return {}


def _save_event_url_cache(cache: dict[str, str]) -> None:
    EVENTS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVENTS_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _construct_bfo_search_query(event_name: str) -> str:
    """Make a good BFO search query from a UFC event name."""
    # Strip common prefixes and clean up
    q = event_name
    # Try the event number for UFC numbered events
    m = re.search(r"UFC\s+(\d+)", q)
    if m:
        return f"UFC {m.group(1)}"
    # For fight nights, use a shorter form
    m = re.search(r"UFC Fight Night:?\s*(.+)", q)
    if m:
        headliner = m.group(1).strip()
        # Take first headliner name
        parts = re.split(r"\s+vs\.?\s+", headliner, maxsplit=1)
        if parts:
            return f"UFC {parts[0].strip()}"
    # TUF finales
    if "Ultimate Fighter" in q:
        return q[:60]
    # UFC on FOX/FX/FUEL
    m = re.search(r"(UFC on (?:FOX|FX|FUEL TV)):?\s*(.+)", q)
    if m:
        return f"{m.group(1)} {m.group(2).split('vs')[0].strip()}"
    return q[:60]


def discover_bfo_event_url(
    event_name: str,
    event_date: str,
    fights: list[dict],
) -> str | None:
    """Find the BFO URL for a UFC event by searching and validating.

    Args:
        event_name: UFC event name
        event_date: YYYY-MM-DD date string
        fights: list of dicts with fighter_a, fighter_b keys for validation
    """
    cache_key = f"{event_date}|{event_name}"
    url_cache = _load_event_url_cache()
    cached_url = url_cache.get(cache_key)
    if cached_url:
        return cached_url

    query = _construct_bfo_search_query(event_name)
    results = search_bfo_events(query)

    if not results:
        # Try with full event name
        results = search_bfo_events(event_name[:80])

    if not results:
        # Try with just "UFC NNN" for numbered events
        m = re.search(r"UFC\s+(\d+)", event_name)
        if m:
            results = search_bfo_events(f"UFC {m.group(1)}")

    if not results:
        return None

    # Score each result by exact event identity plus actual target-fight overlap.
    best_url = None
    best_score = -1
    for url, title in results:
        combined_text = f"{title} {url}"
        identity_score = _exact_ufc_number_score(event_name, combined_text)
        if identity_score < 0:
            continue
        parsed_fights = parse_bfo_event_page(url)
        match_count = _count_target_fight_matches(fights, parsed_fights)
        if match_count <= 0:
            continue
        score = identity_score + (match_count * 1000)
        if score > best_score:
            best_score = score
            best_url = url

    if best_url:
        url_cache[cache_key] = best_url
        _save_event_url_cache(url_cache)
    return best_url


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text())
    return {"completed_events": [], "failed_events": []}


def save_checkpoint(checkpoint: dict) -> None:
    CHECKPOINT_PATH.write_text(json.dumps(checkpoint, indent=2))


def save_rows(rows: list[dict], path: Path) -> None:
    """Append rows to CSV, creating with header if needed."""
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-csv", type=Path,
                        default=REPO_ROOT / "data" / "raw" / "historical_odds" / "pre2022_events.csv")
    parser.add_argument("--fights-csv", type=Path,
                        default=REPO_ROOT / "data" / "raw" / "historical_odds" / "pre2022_fights_needed.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of events to scrape")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.no_resume:
        args.resume = False

    # Load events and fights
    events_df = pd.read_csv(args.events_csv)
    fights_df = pd.read_csv(args.fights_csv)

    # Group fights by event date
    fights_by_date = {}
    for _, row in fights_df.iterrows():
        date = row["event_date"]
        if date not in fights_by_date:
            fights_by_date[date] = []
        fights_by_date[date].append({
            "fighter_a": row["fighter_a"],
            "fighter_b": row["fighter_b"],
        })

    # Load checkpoint for resume
    checkpoint = load_checkpoint() if args.resume else {"completed_events": [], "failed_events": []}
    completed = set(checkpoint["completed_events"])

    # Filter events
    events = []
    for _, row in events_df.iterrows():
        event_name = str(row.get("event_name", ""))
        event_date = str(row["event_date"])
        if event_date in completed:
            continue
        if event_name == "nan" or not event_name:
            continue
        events.append({"event_name": event_name, "event_date": event_date})

    if args.limit:
        events = events[:args.limit]

    logger.info("Events to scrape: %d (already completed: %d)", len(events), len(completed))

    total_fights_scraped = 0
    total_fights_matched = 0

    for idx, event in enumerate(events):
        event_name = event["event_name"]
        event_date = event["event_date"]
        target_fights = fights_by_date.get(event_date, [])

        logger.info("[%d/%d] %s (%s) - %d fights",
                    idx + 1, len(events), event_name, event_date, len(target_fights))

        # Find BFO event URL
        bfo_url = discover_bfo_event_url(event_name, event_date, target_fights)
        if not bfo_url:
            logger.warning("  Could not find BFO page for %s", event_name)
            checkpoint["failed_events"].append(event_date)
            save_checkpoint(checkpoint)
            continue

        logger.info("  BFO URL: %s", bfo_url)

        # Parse event page
        bfo_fights = parse_bfo_event_page(bfo_url)
        if not bfo_fights:
            logger.warning("  No fights parsed from %s", bfo_url)
            checkpoint["failed_events"].append(event_date)
            save_checkpoint(checkpoint)
            continue

        total_fights_scraped += len(bfo_fights)
        logger.info("  Parsed %d fights from BFO", len(bfo_fights))

        # Match BFO fights to our target fights
        rows = []
        for target in target_fights:
            target_a = target["fighter_a"]
            target_b = target["fighter_b"]

            # Find matching BFO fight
            matched = None
            reversed_match = False
            for bfo_fight in bfo_fights:
                bfo_a = bfo_fight["fighter_a"]
                bfo_b = bfo_fight["fighter_b"]

                if _names_match(target_a, bfo_a) and _names_match(target_b, bfo_b):
                    matched = bfo_fight
                    break
                elif _names_match(target_a, bfo_b) and _names_match(target_b, bfo_a):
                    matched = bfo_fight
                    reversed_match = True
                    break

            if matched is None:
                continue

            total_fights_matched += 1
            if reversed_match:
                rows.append({
                    "event_date": event_date,
                    "fighter_a": target_a,
                    "fighter_b": target_b,
                    "query_date": event_date,
                    "offset_days": 0,
                    "a_fair_prob": matched["b_fair_prob"],
                    "b_fair_prob": matched["a_fair_prob"],
                    "a_decimal_odds": matched["b_decimal_odds"],
                    "b_decimal_odds": matched["a_decimal_odds"],
                    "num_bookmakers": matched["num_bookmakers"],
                    "event_name": event_name,
                    "source_url": bfo_url,
                })
            else:
                rows.append({
                    "event_date": event_date,
                    "fighter_a": target_a,
                    "fighter_b": target_b,
                    "query_date": event_date,
                    "offset_days": 0,
                    "a_fair_prob": matched["a_fair_prob"],
                    "b_fair_prob": matched["b_fair_prob"],
                    "a_decimal_odds": matched["a_decimal_odds"],
                    "b_decimal_odds": matched["b_decimal_odds"],
                    "num_bookmakers": matched["num_bookmakers"],
                    "event_name": event_name,
                    "source_url": bfo_url,
                })

        if rows:
            save_rows(rows, args.output)
            logger.info("  Matched %d/%d fights, saved", len(rows), len(target_fights))

        checkpoint["completed_events"].append(event_date)
        save_checkpoint(checkpoint)

    logger.info("Done. Scraped %d fights, matched %d", total_fights_scraped, total_fights_matched)
    return 0


if __name__ == "__main__":
    import pandas as pd
    raise SystemExit(main())
