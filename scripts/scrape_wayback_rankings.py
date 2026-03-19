"""Fetch UFC rankings from Wayback Machine snapshots to fill the 2025-05 to 2026-03 gap.

Selects weekly snapshots, fetches the archived UFC.com/rankings page,
parses rankings by division, and appends to rankings_history.csv.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

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
REQUEST_DELAY = 3.0  # be polite to archive.org

RANKINGS_CSV = REPO_ROOT / "data" / "raw" / "kaggle_rankings" / "rankings_history.csv"
RANKINGS_JSON = REPO_ROOT / "data" / "raw" / "rankings" / "rankings_history.json"
OUTPUT_CSV = REPO_ROOT / "data" / "raw" / "kaggle_rankings" / "rankings_history_extended.csv"
OUTPUT_JSON = REPO_ROOT / "data" / "raw" / "rankings" / "rankings_history_extended.json"

# Weight class normalization that PRESERVES gender distinction
_WC_NORMALIZE = {
    "flyweight": "FLYWEIGHT",
    "men's flyweight": "FLYWEIGHT",
    "bantamweight": "BANTAMWEIGHT",
    "men's bantamweight": "BANTAMWEIGHT",
    "featherweight": "FEATHERWEIGHT",
    "men's featherweight": "FEATHERWEIGHT",
    "lightweight": "LIGHTWEIGHT",
    "welterweight": "WELTERWEIGHT",
    "middleweight": "MIDDLEWEIGHT",
    "light heavyweight": "LIGHT HEAVYWEIGHT",
    "heavyweight": "HEAVYWEIGHT",
    "women's strawweight": "WOMEN'S STRAWWEIGHT",
    "strawweight": "WOMEN'S STRAWWEIGHT",  # UFC only has women's strawweight
    "women's flyweight": "WOMEN'S FLYWEIGHT",
    "women's bantamweight": "WOMEN'S BANTAMWEIGHT",
    "women's featherweight": "WOMEN'S FEATHERWEIGHT",
    "pound-for-pound": "POUND-FOR-POUND",
    "men's pound-for-pound": "MEN'S POUND-FOR-POUND",
    "women's pound-for-pound": "WOMEN'S POUND-FOR-POUND",
}

# Known women fighters for disambiguation
_KNOWN_WOMEN = {
    "zhang weili", "valentina shevchenko", "rose namajunas", "joanna jedrzejczyk",
    "amanda nunes", "holly holm", "miesha tate", "ronda rousey", "julianna pena",
    "raquel pennington", "jessica andrade", "carla esparza", "yan xiaonan",
    "tatiana suarez", "virna jandiroba", "marina rodriguez", "manon fiorot",
    "erin blanchfield", "alexa grasso", "maycee barber", "natalia silva",
    "norma dumont", "megan anderson", "germaine de randamie", "cris cyborg",
    "amanda lemos", "mackenzie dern", "tracy cortez", "kayla harrison",
}


def _normalize_wc(raw: str) -> str:
    """Normalize weight class name, preserving gender distinction."""
    lower = raw.lower().strip()
    # Direct lookup
    if lower in _WC_NORMALIZE:
        return _WC_NORMALIZE[lower]
    # Fuzzy match
    for key, val in _WC_NORMALIZE.items():
        if key in lower:
            return val
    return raw.upper()


def _is_womens_division(fighters: list[str]) -> bool:
    """Check if a set of fighters are from a women's division."""
    for name in fighters[:5]:
        if name.lower() in _KNOWN_WOMEN:
            return True
    return False


def get_wayback_timestamps(from_date: str, to_date: str) -> list[str]:
    """Get Wayback Machine timestamps for UFC.com/rankings."""
    cdx_url = "https://web.archive.org/cdx/search/cdx"
    params = {
        "url": "ufc.com/rankings",
        "output": "json",
        "from": from_date.replace("-", ""),
        "to": to_date.replace("-", ""),
        "fl": "timestamp,statuscode,digest",
        "filter": "statuscode:200",
        "collapse": "timestamp:8",  # one per day
    }
    resp = requests.get(cdx_url, params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if len(data) <= 1:
        return []
    return [row[0] for row in data[1:]]


def select_weekly_timestamps(timestamps: list[str]) -> list[str]:
    """Select approximately one timestamp per week."""
    if not timestamps:
        return []
    selected = [timestamps[0]]
    last_date = datetime.strptime(timestamps[0][:8], "%Y%m%d")
    for ts in timestamps[1:]:
        current_date = datetime.strptime(ts[:8], "%Y%m%d")
        if (current_date - last_date).days >= 7:
            selected.append(ts)
            last_date = current_date
    return selected


def fetch_wayback_page(timestamp: str) -> str | None:
    """Fetch a Wayback Machine snapshot."""
    url = f"https://web.archive.org/web/{timestamp}/https://www.ufc.com/rankings"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60)
        if resp.status_code != 200:
            logger.warning("Wayback returned %d for %s", resp.status_code, timestamp)
            return None
        time.sleep(REQUEST_DELAY)
        return resp.text
    except Exception as exc:
        logger.warning("Failed to fetch wayback %s: %s", timestamp, exc)
        return None


def parse_ufc_rankings_html(html: str) -> dict[str, list[dict]] | None:
    """Parse UFC.com rankings HTML (including Wayback versions).

    Returns dict of {WEIGHT_CLASS: [{fighter: NAME, rank: N}, ...]}
    """
    soup = BeautifulSoup(html, "lxml")
    result = {}

    # Try multiple selector patterns (UFC.com has changed its HTML over time)
    divisions = soup.select("div.view-grouping")
    if not divisions:
        divisions = soup.select("div.rankings--athlete--container")
    if not divisions:
        # Try finding any section with ranking-related content
        divisions = soup.select("section")
        divisions = [d for d in divisions if d.select("h4, h3, h2")]

    if not divisions:
        logger.warning("Could not find ranking divisions in HTML")
        return None

    for div in divisions:
        # Find division header
        header = div.select_one("h4, h3, .view-grouping-header, h2")
        if not header:
            continue
        division_raw = header.get_text(strip=True)
        if not division_raw or len(division_raw) > 80:
            continue

        # Skip non-ranking sections
        skip_words = {"news", "article", "video", "event", "fight", "ticket"}
        if any(w in division_raw.lower() for w in skip_words):
            continue

        is_pfp = "pound" in division_raw.lower()

        # Find fighters in this division
        fighters = []
        for selector in [
            "tr td.views-field-title a",
            "a.e-author__name",
            "td.views-field a",
        ]:
            found = div.select(selector)
            if found:
                fighters = [el.get_text(strip=True) for el in found if el.get_text(strip=True)]
                break

        if not fighters:
            # Fallback: look for list items or table rows
            for row in div.select("tr, li"):
                text = row.get_text(strip=True)
                match = re.match(r"(?:\d+\.?\s*)?([\w\s.'-]+)", text)
                if match:
                    name = match.group(1).strip()
                    if len(name) > 2 and name.lower() not in {"champion", "title", "division", "rank", "name"}:
                        fighters.append(name)

        if not fighters:
            continue

        # Normalize weight class
        wc_key = _normalize_wc(division_raw)

        # Gender disambiguation for generic names like "flyweight"
        if wc_key in ("FLYWEIGHT", "BANTAMWEIGHT") and _is_womens_division(fighters):
            if wc_key == "FLYWEIGHT":
                wc_key = "WOMEN'S FLYWEIGHT"
            elif wc_key == "BANTAMWEIGHT":
                wc_key = "WOMEN'S BANTAMWEIGHT"

        # Build fighter list
        fighter_list = []
        for rank, name in enumerate(fighters[:16]):
            fighter_list.append({
                "fighter": name.upper(),
                "rank": 0 if rank == 0 and not is_pfp else rank + (0 if is_pfp else 0),
            })

        # For champion (rank 0) vs ranked (rank 1-15)
        if not is_pfp and fighter_list:
            fighter_list[0]["rank"] = 0  # champion
            for i in range(1, len(fighter_list)):
                fighter_list[i]["rank"] = i

        if is_pfp:
            for i, f in enumerate(fighter_list):
                f["rank"] = i + 1

        result[wc_key] = fighter_list

    return result if result else None


def rankings_to_csv_rows(date: str, rankings: dict[str, list[dict]]) -> list[dict]:
    """Convert parsed rankings to CSV rows matching rankings_history.csv schema."""
    rows = []
    for wc, fighters in rankings.items():
        for f in fighters:
            rows.append({
                "date": date,
                "weightclass": wc,
                "fighter": f["fighter"],
                "rank": f["rank"],
            })
    return rows


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-date", default="2025-05-07",
                        help="Start date for Wayback search (after existing data ends)")
    parser.add_argument("--to-date", default="2026-03-18",
                        help="End date for Wayback search")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max snapshots to fetch")
    args = parser.parse_args()

    # Get available timestamps
    logger.info("Querying Wayback Machine for UFC.com/rankings snapshots...")
    all_timestamps = get_wayback_timestamps(args.from_date, args.to_date)
    logger.info("Found %d daily snapshots", len(all_timestamps))

    # Select weekly
    weekly = select_weekly_timestamps(all_timestamps)
    logger.info("Selected %d weekly snapshots", len(weekly))

    if args.limit:
        weekly = weekly[:args.limit]

    # Load existing CSV — prefer the extended version if it exists
    import pandas as pd
    base_csv = OUTPUT_CSV if OUTPUT_CSV.exists() else RANKINGS_CSV
    existing_df = pd.read_csv(base_csv)
    existing_dates = set(existing_df["date"].unique())
    logger.info("Existing CSV (%s) has %d dates", base_csv.name, len(existing_dates))

    # Load existing JSON — prefer the extended version if it exists
    base_json = OUTPUT_JSON if OUTPUT_JSON.exists() else RANKINGS_JSON
    with open(base_json, "r") as f:
        existing_json = json.load(f)

    new_csv_rows = []
    new_json_entries = {}
    fetched = 0
    failed = 0

    for ts in weekly:
        date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
        if date_str in existing_dates:
            logger.info("Skipping %s (already in CSV)", date_str)
            continue

        logger.info("Fetching %s (timestamp %s)...", date_str, ts)
        html = fetch_wayback_page(ts)
        if html is None:
            failed += 1
            continue

        rankings = parse_ufc_rankings_html(html)
        if rankings is None:
            logger.warning("Failed to parse rankings for %s", date_str)
            failed += 1
            continue

        divisions = len(rankings)
        total_fighters = sum(len(f) for f in rankings.values())
        logger.info("  Parsed %d divisions, %d fighters", divisions, total_fighters)

        # Add to CSV rows
        csv_rows = rankings_to_csv_rows(date_str, rankings)
        new_csv_rows.extend(csv_rows)

        # Add to JSON
        new_json_entries[date_str] = rankings

        fetched += 1

    logger.info("Fetched %d snapshots, %d failed", fetched, failed)

    if new_csv_rows:
        # Extend CSV
        new_df = pd.DataFrame(new_csv_rows)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.sort_values(["date", "weightclass", "rank"]).reset_index(drop=True)
        combined_df.to_csv(OUTPUT_CSV, index=False)
        logger.info("Extended CSV saved to %s (%d rows, %d dates)",
                     OUTPUT_CSV, len(combined_df), combined_df["date"].nunique())

        # Extend JSON
        existing_json.update(new_json_entries)
        with open(OUTPUT_JSON, "w") as f:
            json.dump(existing_json, f, indent=2)
        logger.info("Extended JSON saved to %s (%d dates)",
                     OUTPUT_JSON, len(existing_json))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
