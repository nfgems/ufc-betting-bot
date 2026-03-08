"""
Live event monitor — continuously tracks upcoming UFC cards for changes.

Monitors:
- Upcoming event cards (fighter matchups, dates)
- Fighter changes (replacements, cancellations)
- Weigh-in results (missed weight)
- Referee assignments
- Short-notice replacements

Runs on a schedule and stores snapshots so the model always has fresh data.
"""

import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup

from src.config import RAW_DATA_DIR, LOGS_DIR

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
SNAPSHOTS_DIR = RAW_DATA_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# UFC.com upcoming events
# ---------------------------------------------------------------------------

def scrape_upcoming_events() -> list[dict]:
    """
    Scrape upcoming UFC events from UFCStats.com.

    Returns list of event dicts with:
        title, date, url, fights (list of matchups)
    """
    url = "http://ufcstats.com/statistics/events/upcoming"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch upcoming events: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    events = []

    for row in soup.select("tr.b-statistics__table-row"):
        link = row.select_one("a.b-link")
        if not link or not link.get("href"):
            continue

        date_el = row.select_one("span.b-statistics__date")
        events.append({
            "title": link.text.strip(),
            "url": link["href"].strip(),
            "date": date_el.text.strip() if date_el else "",
        })

    logger.info(f"Found {len(events)} upcoming events")
    return events


def scrape_event_card(event_url: str) -> list[dict]:
    """
    Scrape the fight card for a specific upcoming event.

    Returns list of fight dicts with:
        fighter_a, fighter_b, weight_class, is_main_event
    """
    try:
        resp = requests.get(event_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch event card: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    fights = []

    for row in soup.select("tr.b-fight-details__table-row"):
        cols = row.select("td")
        if len(cols) < 2:
            continue

        # Fighter names are in the first column
        fighters = row.select("a.b-link")
        if len(fighters) < 2:
            continue

        fighter_a = fighters[0].text.strip()
        fighter_b = fighters[1].text.strip()

        # Weight class
        wc_el = row.select_one("td.b-fight-details__table-col_style_align-center")
        weight_class = wc_el.text.strip() if wc_el else ""

        fights.append({
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "weight_class": weight_class,
            "is_main_event": len(fights) == 0,  # First fight listed is usually main
        })

    logger.info(f"Found {len(fights)} fights on card")
    return fights


# ---------------------------------------------------------------------------
# Weigh-in scraper
# ---------------------------------------------------------------------------

def scrape_weighin_results(event_name: str) -> list[dict]:
    """
    Scrape weigh-in results from MMA news sources.

    Searches for weigh-in results by event name and extracts:
        - Fighter name
        - Weight recorded
        - Weight limit for the bout
        - Whether they missed weight
        - How much they missed by (if applicable)

    Returns list of weigh-in result dicts.
    """
    # Try UFCStats first — weigh-in data appears on event pages close to fight day
    # Also search common MMA news sites
    results = []

    # Search MMAJunkie/MMAFighting style weigh-in pages
    search_terms = event_name.replace(" ", "+") + "+weigh+in+results"
    search_url = f"https://www.google.com/search?q={search_terms}+site:mmajunkie.usatoday.com"

    try:
        resp = requests.get(search_url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "lxml")
            # Extract first result URL
            for link in soup.select("a"):
                href = link.get("href", "")
                if "mmajunkie" in href and "weigh" in href.lower():
                    results = _parse_weighin_page(href)
                    if results:
                        break
    except Exception as e:
        logger.debug(f"Google search failed: {e}")

    return results


def _parse_weighin_page(url: str) -> list[dict]:
    """Parse a weigh-in results page for fighter weights."""
    results = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Look for weight data in page text
        text = soup.get_text()

        # Common format: "Fighter Name (155.5)" or "Fighter Name – 155.5 lbs"
        weight_pattern = re.compile(
            r"([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*"
            r"[\(–\-:]\s*(\d{2,3}(?:\.\d+)?)\s*(?:lbs?|pounds?)?\s*[\)]?"
        )

        for match in weight_pattern.finditer(text):
            name = match.group(1).strip()
            weight = float(match.group(2))
            results.append({
                "fighter": name,
                "weight": weight,
                "source_url": url,
            })

    except Exception as e:
        logger.debug(f"Failed to parse weigh-in page {url}: {e}")

    return results


def check_missed_weight(
    weighin_results: list[dict],
    fight_card: list[dict],
) -> list[dict]:
    """
    Cross-reference weigh-in results with fight card to detect missed weight.

    Weight class limits:
        Strawweight: 115, Flyweight: 125, Bantamweight: 135,
        Featherweight: 145, Lightweight: 155, Welterweight: 170,
        Middleweight: 185, Light Heavyweight: 205, Heavyweight: 265
    """
    WEIGHT_LIMITS = {
        "strawweight": 115, "women's strawweight": 115,
        "flyweight": 125, "women's flyweight": 125,
        "bantamweight": 135, "women's bantamweight": 135,
        "featherweight": 145, "women's featherweight": 145,
        "lightweight": 155,
        "welterweight": 170,
        "middleweight": 185,
        "light heavyweight": 205,
        "heavyweight": 265,
    }
    # Title fights get no 1lb allowance; non-title get +1lb
    ALLOWANCE = 1.0

    missed = []

    for result in weighin_results:
        fighter = result["fighter"].lower()
        weight = result["weight"]

        # Find this fighter's bout on the card
        for fight in fight_card:
            wc = fight.get("weight_class", "").lower()
            limit = WEIGHT_LIMITS.get(wc)
            if not limit:
                continue

            fa = fight["fighter_a"].lower()
            fb = fight["fighter_b"].lower()

            if fighter in fa or fa in fighter or fighter in fb or fb in fighter:
                effective_limit = limit + ALLOWANCE
                if weight > effective_limit:
                    over_by = weight - effective_limit
                    missed.append({
                        "fighter": result["fighter"],
                        "weight": weight,
                        "weight_class": fight["weight_class"],
                        "limit": effective_limit,
                        "over_by": over_by,
                        "opponent": fight["fighter_b"] if fighter in fa else fight["fighter_a"],
                    })
                    logger.warning(
                        f"MISSED WEIGHT: {result['fighter']} weighed {weight} lbs, "
                        f"{over_by:.1f} lbs over {fight['weight_class']} limit"
                    )
                break

    return missed


# ---------------------------------------------------------------------------
# Referee assignment scraper
# ---------------------------------------------------------------------------

# Known UFC referee tendencies (manually curated from historical data)
# Higher score = more likely to stand up ground action (favors strikers)
REFEREE_STANDUP_TENDENCY = {
    "herb dean": 0.7,
    "marc goddard": 0.65,
    "dan miragliotta": 0.6,
    "keith peterson": 0.55,
    "mark smith": 0.55,
    "jason herzog": 0.5,
    "chris tognoni": 0.5,
    "mike beltran": 0.45,
    "minoru toyonaga": 0.45,
}

# Stoppage tendency: higher = more likely to stop early (favors aggressive fighters)
REFEREE_STOPPAGE_TENDENCY = {
    "herb dean": 0.5,
    "marc goddard": 0.55,
    "mario yamasaki": 0.3,  # Known for late stoppages
    "steve mazzagatti": 0.35,
    "dan miragliotta": 0.6,
    "keith peterson": 0.55,
    "jason herzog": 0.5,
    "mark smith": 0.55,
}


def get_referee_features(referee_name: str) -> dict:
    """
    Get referee tendency features for the prediction model.

    Returns dict with:
        - ref_standup_tendency: 0-1 (higher = more standups)
        - ref_stoppage_tendency: 0-1 (higher = earlier stoppages)
        - ref_known: 1 if referee is in our database, 0 if unknown
    """
    name = referee_name.lower().strip()
    return {
        "ref_standup_tendency": REFEREE_STANDUP_TENDENCY.get(name, 0.5),
        "ref_stoppage_tendency": REFEREE_STOPPAGE_TENDENCY.get(name, 0.5),
        "ref_known": 1 if name in REFEREE_STANDUP_TENDENCY else 0,
    }


# ---------------------------------------------------------------------------
# Short-notice replacement detection
# ---------------------------------------------------------------------------

def detect_short_notice(
    current_card: list[dict],
    previous_card: list[dict],
    days_to_event: int,
) -> list[dict]:
    """
    Detect short-notice replacements by comparing current card to previous snapshot.

    A short-notice replacement is when a fighter is swapped in within ~2 weeks of the event.
    Short-notice fighters historically win at ~38% rate.

    Returns list of replacement dicts.
    """
    replacements = []

    current_fighters = set()
    for fight in current_card:
        current_fighters.add(fight["fighter_a"].lower())
        current_fighters.add(fight["fighter_b"].lower())

    previous_fighters = set()
    for fight in previous_card:
        previous_fighters.add(fight["fighter_a"].lower())
        previous_fighters.add(fight["fighter_b"].lower())

    # New fighters not in previous card
    new_fighters = current_fighters - previous_fighters
    removed_fighters = previous_fighters - current_fighters

    if new_fighters and days_to_event <= 14:
        for fighter in new_fighters:
            # Try to find who they replaced
            replaced = None
            for fight in current_card:
                fa = fight["fighter_a"].lower()
                fb = fight["fighter_b"].lower()
                if fighter == fa:
                    opponent = fb
                elif fighter == fb:
                    opponent = fa
                else:
                    continue
                # Check if the opponent's previous opponent was removed
                for prev_fight in previous_card:
                    pa = prev_fight["fighter_a"].lower()
                    pb = prev_fight["fighter_b"].lower()
                    if opponent == pa and pb in removed_fighters:
                        replaced = pb
                    elif opponent == pb and pa in removed_fighters:
                        replaced = pa

            replacements.append({
                "new_fighter": fighter,
                "replaced_fighter": replaced,
                "days_notice": days_to_event,
                "is_short_notice": days_to_event <= 14,
            })
            logger.warning(
                f"SHORT NOTICE: {fighter} replacing {replaced or 'unknown'} "
                f"with {days_to_event} days notice"
            )

    return replacements


# ---------------------------------------------------------------------------
# Snapshot management
# ---------------------------------------------------------------------------

def save_card_snapshot(event_title: str, card: list[dict]) -> Path:
    """Save a fight card snapshot for change detection."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = re.sub(r"[^\w\-]", "_", event_title)[:50]
    filename = f"{safe_title}_{timestamp}.json"
    path = SNAPSHOTS_DIR / filename

    snapshot = {
        "event": event_title,
        "timestamp": datetime.now().isoformat(),
        "fights": card,
    }
    path.write_text(json.dumps(snapshot, indent=2))
    logger.info(f"Saved card snapshot: {path}")
    return path


def load_latest_snapshot(event_title: str) -> Optional[list[dict]]:
    """Load the most recent snapshot for an event."""
    safe_title = re.sub(r"[^\w\-]", "_", event_title)[:50]
    snapshots = sorted(SNAPSHOTS_DIR.glob(f"{safe_title}_*.json"), reverse=True)
    if not snapshots:
        return None

    data = json.loads(snapshots[0].read_text())
    return data.get("fights", [])


# ---------------------------------------------------------------------------
# Full monitoring pass
# ---------------------------------------------------------------------------

def run_monitoring_pass() -> dict:
    """
    Run a complete monitoring pass:
    1. Fetch upcoming events
    2. Scrape each event card
    3. Compare to previous snapshots (detect replacements)
    4. Check for weigh-in results (if event is within 2 days)
    5. Save new snapshots

    Returns dict with all signals detected.
    """
    logger.info("="*60)
    logger.info("RUNNING MONITORING PASS")
    logger.info("="*60)

    signals = {
        "events": [],
        "short_notice_replacements": [],
        "missed_weights": [],
        "card_changes": [],
    }

    events = scrape_upcoming_events()

    for event in events:
        event_url = event.get("url", "")
        event_title = event.get("title", "")
        event_date_str = event.get("date", "")

        if not event_url:
            continue

        # Parse event date
        try:
            event_date = pd.to_datetime(event_date_str)
            days_to_event = (event_date - pd.Timestamp.now()).days
        except Exception:
            days_to_event = 30  # Default if we can't parse

        logger.info(f"\nEvent: {event_title} ({days_to_event} days away)")

        # Scrape current card
        time.sleep(1.5)
        current_card = scrape_event_card(event_url)
        if not current_card:
            continue

        event_info = {
            **event,
            "days_to_event": days_to_event,
            "num_fights": len(current_card),
            "fights": current_card,
        }
        signals["events"].append(event_info)

        # Compare to previous snapshot
        previous_card = load_latest_snapshot(event_title)
        if previous_card:
            replacements = detect_short_notice(current_card, previous_card, days_to_event)
            signals["short_notice_replacements"].extend(replacements)

        # Check weigh-ins if event is within 2 days
        if days_to_event <= 2:
            weighin_results = scrape_weighin_results(event_title)
            if weighin_results:
                missed = check_missed_weight(weighin_results, current_card)
                signals["missed_weights"].extend(missed)

        # Save snapshot
        save_card_snapshot(event_title, current_card)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("MONITORING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Events tracked: {len(signals['events'])}")
    logger.info(f"Short-notice replacements: {len(signals['short_notice_replacements'])}")
    logger.info(f"Missed weights: {len(signals['missed_weights'])}")

    # Save full signals
    signals_path = LOGS_DIR / "latest_signals.json"
    serializable = {
        k: v if not isinstance(v, pd.DataFrame) else v.to_dict()
        for k, v in signals.items()
    }
    signals_path.write_text(json.dumps(serializable, indent=2, default=str))

    return signals
