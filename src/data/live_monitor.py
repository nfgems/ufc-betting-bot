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
from urllib.parse import urljoin

import requests
import pandas as pd
from bs4 import BeautifulSoup

from src.config import RAW_DATA_DIR, LOGS_DIR, UFCSTATS_UPCOMING_URL
from src.data.event_context import infer_empty_arena
from src.data.name_utils import normalize_cross_source_name, normalize_person_name

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
UFC_COM_EVENTS_URL = "https://www.ufc.com/events"
UFC_COM_BASE_URL = "https://www.ufc.com"
SNAPSHOTS_DIR = RAW_DATA_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
_UFC_COM_EVENT_CTA_TEXT = {
    "Available",
    "Fight Card",
    "How to Watch",
    "Tickets",
    "View Event Details",
    "View Fight Card",
    "Watch Replay",
}

_UPCOMING_WEIGHT_CLASS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bwomen(?:'s|s)?\s+strawweight\b", re.IGNORECASE), "Women's Strawweight"),
    (re.compile(r"\bwomen(?:'s|s)?\s+flyweight\b", re.IGNORECASE), "Women's Flyweight"),
    (re.compile(r"\bwomen(?:'s|s)?\s+bantamweight\b", re.IGNORECASE), "Women's Bantamweight"),
    (re.compile(r"\bwomen(?:'s|s)?\s+featherweight\b", re.IGNORECASE), "Women's Featherweight"),
    (re.compile(r"\blight\s*heavyweight\b", re.IGNORECASE), "Light Heavyweight"),
    (re.compile(r"\bstrawweight\b", re.IGNORECASE), "Strawweight"),
    (re.compile(r"\bflyweight\b", re.IGNORECASE), "Flyweight"),
    (re.compile(r"\bbantamweight\b", re.IGNORECASE), "Bantamweight"),
    (re.compile(r"\bfeatherweight\b", re.IGNORECASE), "Featherweight"),
    (re.compile(r"\blightweight\b", re.IGNORECASE), "Lightweight"),
    (re.compile(r"\bwelterweight\b", re.IGNORECASE), "Welterweight"),
    (re.compile(r"\bmiddleweight\b", re.IGNORECASE), "Middleweight"),
    (re.compile(r"\bheavyweight\b", re.IGNORECASE), "Heavyweight"),
    (re.compile(r"\bcatch\s*weight\b", re.IGNORECASE), "Catch Weight"),
    (re.compile(r"\bopen\s*weight\b", re.IGNORECASE), "Open Weight"),
]


def _tracked_fight_pair_key(fighter_a: str, fighter_b: str) -> str:
    return "|".join(sorted([normalize_cross_source_name(fighter_a), normalize_cross_source_name(fighter_b)]))


def _attach_event_identity(tracked_fights: list[dict]) -> list[dict]:
    """Enrich tracked fights with Odds API event identity when available."""
    if not tracked_fights:
        return tracked_fights

    try:
        from src.data.odds_client import OddsClient

        client = OddsClient()
        odds = client.get_live_odds()
        odds_df = client.odds_to_dataframe(odds)
    except Exception as exc:
        logger.warning("Could not enrich tracked fights with event identity: %s", exc)
        return tracked_fights

    if odds_df.empty:
        return tracked_fights

    candidates = client.get_consensus_odds(odds_df)
    if candidates.empty:
        return tracked_fights

    candidates = candidates.copy()
    candidates["pair_key"] = candidates.apply(
        lambda row: _tracked_fight_pair_key(row.get("fighter_a", ""), row.get("fighter_b", "")),
        axis=1,
    )
    candidates["commence_time_parsed"] = pd.to_datetime(candidates["commence_time"], errors="coerce", utc=True)
    candidates["commence_day"] = candidates["commence_time_parsed"].dt.tz_localize(None).dt.normalize()

    enriched = []
    for fight in tracked_fights:
        updated = dict(fight)
        if updated.get("event_id") and updated.get("commence_time"):
            enriched.append(updated)
            continue

        pair_key = _tracked_fight_pair_key(updated.get("fighter_a", ""), updated.get("fighter_b", ""))
        matched = candidates[candidates["pair_key"] == pair_key].copy()
        if matched.empty:
            enriched.append(updated)
            continue

        event_date = pd.to_datetime(updated.get("event_date"), errors="coerce")
        if pd.notna(event_date):
            matched["event_gap"] = (matched["commence_day"] - event_date.normalize()).abs()
            matched = matched.sort_values(["event_gap", "commence_time_parsed", "event_id"])
        else:
            matched = matched.sort_values(["commence_time_parsed", "event_id"])

        best = matched.iloc[0]
        updated["event_id"] = updated.get("event_id") or best.get("event_id", "")
        updated["commence_time"] = updated.get("commence_time") or best.get("commence_time", "")
        enriched.append(updated)

    return enriched


# ---------------------------------------------------------------------------
# Upcoming event scraping
# ---------------------------------------------------------------------------

def _looks_like_browser_challenge(html: str) -> bool:
    text = html.lower()
    return (
        "checking your browser" in text
        or "requires javascript" in text
        or "enable javascript" in text
    )


def _full_url(url: str, base_url: str = UFC_COM_BASE_URL) -> str:
    return urljoin(base_url, url)


def _format_ufc_com_event_title(href: str, headline: str) -> str:
    headline = re.sub(r"\s+", " ", headline or "").strip()
    href_lower = (href or "").lower()
    if not headline:
        return ""
    if headline.upper().startswith("UFC "):
        return headline
    if "/ufc-fight-night" in href_lower:
        return f"UFC Fight Night: {headline}"
    match = re.search(r"/ufc-(\d+)(?:\b|$|-)", href_lower)
    if match:
        return f"UFC {match.group(1)}: {headline}"
    return headline


def _event_year_from_ufc_com_href(href: str) -> int | None:
    match = re.search(r"(?:^|-)(20\d{2})(?:$|-)", href or "")
    if not match:
        return None
    return int(match.group(1))


def _parse_ufc_com_event_date(raw_text: str, href: str) -> str:
    text = re.sub(r"\s+", " ", raw_text or "").strip()
    match = re.search(r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+([A-Z][a-z]{2})\s+(\d{1,2})\b", text)
    if not match:
        return text

    month_abbr, day = match.groups()
    year = _event_year_from_ufc_com_href(href)
    if year is None:
        today = datetime.now().date()
        candidate = datetime.strptime(f"{month_abbr} {int(day)} {today.year}", "%b %d %Y").date()
        if (today - candidate).days > 180:
            candidate = candidate.replace(year=today.year + 1)
        year = candidate.year

    parsed = datetime.strptime(f"{month_abbr} {int(day)} {year}", "%b %d %Y")
    return f"{parsed.strftime('%B')} {parsed.day}, {parsed.year}"


def _ufc_com_event_text_parts(article) -> list[str]:
    return [
        re.sub(r"\s+", " ", part).strip()
        for part in article.get_text("|", strip=True).split("|")
        if re.sub(r"\s+", " ", part).strip()
        and re.sub(r"\s+", " ", part).strip() not in _UFC_COM_EVENT_CTA_TEXT
    ]


def _ufc_com_event_date_part_index(text_parts: list[str]) -> int | None:
    for idx, part in enumerate(text_parts):
        if re.search(r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+[A-Z][a-z]{2}\s+\d{1,2}\b", part):
            return idx
    return None


def _extract_ufc_com_event_location_from_article(article) -> str:
    text_parts = _ufc_com_event_text_parts(article)
    date_idx = _ufc_com_event_date_part_index(text_parts)
    if date_idx is None:
        return ""

    if len(text_parts) <= date_idx + 1:
        return ""

    location_parts = []
    for part in text_parts[date_idx + 1:]:
        normalized = part.strip(" ,")
        if not normalized:
            continue
        location_parts.append(normalized)

    return ", ".join(location_parts)


def _scrape_ufc_com_upcoming_events() -> list[dict]:
    try:
        resp = requests.get(UFC_COM_EVENTS_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch UFC.com events: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    events: list[dict] = []
    seen_urls: set[str] = set()

    for article in soup.select("article.c-card-event--result"):
        article_text = article.get_text(" ", strip=True)
        if "Watch Replay" in article_text:
            continue

        event_link = None
        for link in article.find_all("a", href=True):
            href = str(link.get("href", ""))
            if "/event/" in href:
                event_link = href
                break
        if not event_link:
            continue

        full_event_url = _full_url(event_link)
        if full_event_url in seen_urls:
            continue
        seen_urls.add(full_event_url)

        text_parts = _ufc_com_event_text_parts(article)
        date_idx = _ufc_com_event_date_part_index(text_parts)
        if date_idx is not None and date_idx > 0:
            headline = text_parts[date_idx - 1]
        else:
            headline = text_parts[0] if text_parts else ""
        raw_date = text_parts[date_idx] if date_idx is not None else ""
        title = _format_ufc_com_event_title(event_link, headline)
        events.append(
            {
                "title": title,
                "url": full_event_url,
                "date": _parse_ufc_com_event_date(raw_date, event_link),
                "location": _extract_ufc_com_event_location_from_article(article),
                "source": "ufc.com",
            }
        )

    logger.info(f"Found {len(events)} upcoming events via UFC.com")
    return events


def _scrape_ufcstats_upcoming_events() -> list[dict]:
    url = UFCSTATS_UPCOMING_URL
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch UFCStats upcoming events: {e}")
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

    if events:
        logger.info(f"Found {len(events)} upcoming events via UFCStats")
        return events

    if _looks_like_browser_challenge(resp.text):
        logger.warning("UFCStats upcoming events returned a browser-check page")
    else:
        logger.warning("UFCStats upcoming events returned no event rows")
    return []


def scrape_upcoming_events() -> list[dict]:
    """
    Scrape upcoming UFC events from the official UFC schedule.

    UFC.com is the primary upcoming-card source because UFCStats can lag or omit
    scheduled fights. UFCStats remains an HTTP fallback for resilience.
    """
    events = _scrape_ufc_com_upcoming_events()
    if events:
        return events

    logger.warning("UFC.com upcoming events returned no event rows; falling back to UFCStats")
    return _scrape_ufcstats_upcoming_events()


def _extract_ufc_com_event_location(soup: BeautifulSoup) -> str:
    location_el = soup.select_one(".field--name-venue")
    if location_el is None:
        return ""
    parts = [
        re.sub(r"\s+", " ", part).strip(" ,")
        for part in location_el.get_text("|", strip=True).split("|")
        if re.sub(r"\s+", " ", part).strip(" ,")
    ]
    return ", ".join(parts)


def _extract_ufc_com_corner_name(fight, selector: str) -> str:
    name_el = fight.select_one(f".c-listing-fight__names-row {selector}") or fight.select_one(selector)
    if name_el is None:
        return ""
    return re.sub(r"\s+", " ", name_el.get_text(" ", strip=True)).strip()


def _extract_ufc_com_weight_class(fight) -> str:
    class_el = (
        fight.select_one(".c-listing-fight__class--desktop .c-listing-fight__class-text")
        or fight.select_one(".c-listing-fight__class-text")
    )
    if class_el is None:
        return ""
    return re.sub(r"\s+", " ", class_el.get_text(" ", strip=True)).strip()


def _scrape_ufc_com_event_card(event_url: str) -> list[dict]:
    try:
        resp = requests.get(event_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch UFC.com event card: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    event_location = _extract_ufc_com_event_location(soup)
    fights = []

    for fight in soup.select(".c-listing-fight"):
        fighter_a = _extract_ufc_com_corner_name(fight, ".c-listing-fight__corner-name--red")
        fighter_b = _extract_ufc_com_corner_name(fight, ".c-listing-fight__corner-name--blue")
        if not fighter_a or not fighter_b:
            continue

        weight_class = _extract_ufc_com_weight_class(fight)
        row_text = fight.get_text(" ", strip=True).lower()
        is_main_event = len(fights) == 0
        is_title_bout = (
            "title bout" in row_text
            or "championship" in row_text
            or "interim" in row_text
        )
        fights.append(
            {
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "weight_class": weight_class,
                "is_main_event": is_main_event,
                "is_title_bout": is_title_bout,
                "num_rounds": 5 if (is_main_event or is_title_bout) else 3,
                "location": event_location,
            }
        )

    logger.info(f"Found {len(fights)} fights on UFC.com card")
    return fights


def scrape_event_card(event_url: str) -> list[dict]:
    """
    Scrape the fight card for a specific upcoming event.

    Returns list of fight dicts with:
        fighter_a, fighter_b, weight_class, is_main_event, is_title_bout, num_rounds
    """
    if "ufc.com/event/" in (event_url or ""):
        return _scrape_ufc_com_event_card(event_url)

    try:
        resp = requests.get(event_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch event card: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    fights = []
    event_location = _extract_event_location(soup)

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
        weight_class = _extract_upcoming_weight_class(row)

        row_text = row.get_text(" ", strip=True).lower()
        row_html = str(row).lower()
        is_main_event = len(fights) == 0  # First fight listed is usually main
        is_title_bout = bool(
            row.select_one("img[src*='belt']")
            or "title bout" in row_text
            or "title bout" in row_html
            or "championship" in row_text
            or "championship" in row_html
            or "interim" in row_text
            or "interim" in row_html
        )
        num_rounds = 5 if (is_main_event or is_title_bout) else 3

        fights.append({
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "weight_class": weight_class,
            "is_main_event": is_main_event,
            "is_title_bout": is_title_bout,
            "num_rounds": num_rounds,
            "location": event_location,
        })

    logger.info(f"Found {len(fights)} fights on card")
    return fights


def _extract_event_location(soup: BeautifulSoup) -> str:
    """Extract the event location from a UFCStats event page."""
    for item in soup.select("li.b-list__box-list-item"):
        text = re.sub(r"\s+", " ", item.get_text(" ", strip=True)).strip()
        if text.startswith("Location:"):
            return text.replace("Location:", "", 1).strip()
    return ""


def _extract_upcoming_weight_class(row) -> str:
    """
    Extract the UFC weight class from an upcoming event-card row.

    UFCStats upcoming rows may include multiple centered columns, with the first
    one blank. Resolve by scanning cell text for known division labels instead of
    trusting the first aligned cell.
    """
    candidate_texts = [
        re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()
        for cell in row.select("td")
    ]
    candidate_texts.append(re.sub(r"\s+", " ", row.get_text(" ", strip=True)).strip())

    for text in candidate_texts:
        if not text:
            continue
        for pattern, label in _UPCOMING_WEIGHT_CLASS_PATTERNS:
            if pattern.search(text.replace("’", "'")):
                return label

    return ""


def collect_upcoming_fight_contexts() -> list[dict]:
    """Return upcoming fight metadata enriched with Odds API event identity."""
    contexts: list[dict] = []

    for event in scrape_upcoming_events():
        event_url = event.get("url", "")
        if not event_url:
            continue

        card = scrape_event_card(event_url)
        if not card:
            continue

        for fight in card:
            contexts.append(
                {
                    "event_title": event.get("title", ""),
                    "event_date": event.get("date", ""),
                    "location": fight.get("location", ""),
                    "fighter_a": fight.get("fighter_a", ""),
                    "fighter_b": fight.get("fighter_b", ""),
                    "weight_class": fight.get("weight_class", ""),
                    "is_main_event": bool(fight.get("is_main_event", False)),
                    "is_title_bout": bool(fight.get("is_title_bout", False)),
                    "is_empty_arena": infer_empty_arena(
                        event_title=event.get("title", ""),
                        location=fight.get("location", ""),
                    ),
                    "num_rounds": int(
                        fight.get(
                            "num_rounds",
                            5 if (fight.get("is_main_event") or fight.get("is_title_bout")) else 3,
                        )
                    ),
                }
            )

    return _attach_event_identity(contexts)


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
    NON_TITLE_ALLOWANCE = 1.0

    missed = []

    for result in weighin_results:
        fighter = result["fighter"].lower().strip()
        weight = result["weight"]

        # Find this fighter's bout on the card
        for fight in fight_card:
            wc = fight.get("weight_class", "").lower()
            limit = WEIGHT_LIMITS.get(wc)
            if not limit:
                continue

            fa = fight["fighter_a"].lower().strip()
            fb = fight["fighter_b"].lower().strip()

            if fighter == fa or fighter == fb:
                is_title = bool(fight.get("is_title_bout") or fight.get("title_bout"))
                allowance = 0.0 if is_title else NON_TITLE_ALLOWANCE
                effective_limit = limit + allowance
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
    if name and name not in REFEREE_STANDUP_TENDENCY:
        logger.warning("Unknown referee '%s' — using default tendencies", referee_name)
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

def save_card_snapshot(event_title: str, card: list[dict], event_date: str = "") -> Path:
    """Save a fight card snapshot for change detection."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = re.sub(r"[^\w\-]", "_", event_title)[:50]
    filename = f"{safe_title}_{timestamp}.json"
    path = SNAPSHOTS_DIR / filename

    snapshot = {
        "event": event_title,
        "event_date": event_date,
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
        "rankings_snapshot": None,
        "method_odds_snapshot": None,
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
        save_card_snapshot(event_title, current_card, event_date=event_date_str)

    tracked_fights = []
    for event in signals["events"]:
        for fight in event.get("fights", []):
            tracked_fights.append(
                {
                    "event_title": event.get("title", ""),
                    "event_date": event.get("date", ""),
                    "fighter_a": fight.get("fighter_a", ""),
                    "fighter_b": fight.get("fighter_b", ""),
                    "weight_class": fight.get("weight_class", ""),
                    "event_id": fight.get("event_id", "") or event.get("event_id", ""),
                    "commence_time": fight.get("commence_time", "") or event.get("commence_time", ""),
                }
            )
    tracked_fights = _attach_event_identity(tracked_fights)

    try:
        from src.data.rankings_scraper import collect_rankings_snapshot

        rankings_snapshot = collect_rankings_snapshot()
        signals["rankings_snapshot"] = {
            "status": rankings_snapshot.get("status"),
            "source": rankings_snapshot.get("source"),
            "snapshot_time": rankings_snapshot.get("snapshot_time"),
            "snapshot_path": rankings_snapshot.get("snapshot_path"),
        }
    except Exception as e:
        logger.error(f"Rankings snapshot collection failed: {e}")

    try:
        from src.data.method_odds import collect_method_odds_snapshot

        method_snapshot = collect_method_odds_snapshot(tracked_fights=tracked_fights)
        signals["method_odds_snapshot"] = {
            "status": method_snapshot.get("status"),
            "record_count": method_snapshot.get("record_count", 0),
            "snapshot_time": method_snapshot.get("snapshot_time"),
            "snapshot_path": method_snapshot.get("snapshot_path"),
        }
    except Exception as e:
        logger.error(f"Method-odds snapshot collection failed: {e}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("MONITORING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Events tracked: {len(signals['events'])}")
    logger.info(f"Short-notice replacements: {len(signals['short_notice_replacements'])}")
    logger.info(f"Missed weights: {len(signals['missed_weights'])}")
    if signals["rankings_snapshot"] is not None:
        logger.info(
            "Rankings snapshot: %s via %s",
            signals["rankings_snapshot"].get("status"),
            signals["rankings_snapshot"].get("source"),
        )
    if signals["method_odds_snapshot"] is not None:
        logger.info(
            "Method-odds snapshot: %s (%s records)",
            signals["method_odds_snapshot"].get("status"),
            signals["method_odds_snapshot"].get("record_count", 0),
        )

    # Save full signals
    signals_path = LOGS_DIR / "latest_signals.json"
    serializable = {
        k: v if not isinstance(v, pd.DataFrame) else v.to_dict()
        for k, v in signals.items()
    }
    signals_path.write_text(json.dumps(serializable, indent=2, default=str))

    return signals
