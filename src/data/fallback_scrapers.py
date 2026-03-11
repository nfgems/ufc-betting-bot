"""
Fallback fighter data scrapers — Sherdog and Tapology.

Used when UFCStats.com has no data for a fighter (e.g., regional/Contender
Series fighters). Provides partial feature data (physical attributes, record,
fight history with methods) but not per-fight striking/grappling stats.
"""

import logging
import re
import time
from datetime import datetime
from typing import Optional

import numpy as np
import requests
from bs4 import BeautifulSoup

from src.config import SHERDOG_BASE_URL, SHERDOG_SEARCH_URL

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
REQUEST_DELAY = 1.5  # Slightly longer than UFCStats to be polite

# Session caches
_sherdog_url_cache: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_soup(url: str) -> BeautifulSoup:
    """Fetch a URL and return parsed BeautifulSoup."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return BeautifulSoup(resp.text, "lxml")


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _safe_float(value, default=np.nan) -> float:
    if value is None or value == "" or value == "--" or value == "N/A":
        return default
    try:
        return float(str(value).replace("%", "").strip())
    except (ValueError, TypeError):
        return default


def _empty_fight_dict() -> dict:
    """Return a fight dict with all per-fight stats set to NaN."""
    return {
        "kd": np.nan,
        "sig_str_landed": np.nan,
        "sig_str_attempted": np.nan,
        "td_landed": np.nan,
        "td_attempted": np.nan,
        "sub_att": np.nan,
        "rev": np.nan,
        "ctrl_seconds": np.nan,
        "opp_kd": np.nan,
        "opp_sig_str_landed": np.nan,
        "opp_sig_str_attempted": np.nan,
        "opp_td_landed": np.nan,
        "opp_td_attempted": np.nan,
        "opp_sub_att": np.nan,
        "opp_rev": np.nan,
        "opp_ctrl_seconds": np.nan,
        "detail_url": "",
    }


def _empty_profile_stats() -> dict:
    """Return career rate stats as NaN (not available on fallback sources)."""
    return {
        "slpm": np.nan,
        "str_acc": np.nan,
        "sapm": np.nan,
        "str_def": np.nan,
        "td_avg": np.nan,
        "td_acc": np.nan,
        "td_def": np.nan,
        "sub_avg": np.nan,
    }


# ---------------------------------------------------------------------------
# Sherdog scraper
# ---------------------------------------------------------------------------

def search_sherdog(fighter_name: str) -> Optional[str]:
    """
    Search Sherdog for a fighter by name. Returns their full profile URL.

    Uses the fightfinder search endpoint and fuzzy-matches the results.
    """
    if fighter_name in _sherdog_url_cache:
        return _sherdog_url_cache[fighter_name]

    name_lower = fighter_name.lower().strip()
    parts = name_lower.split()
    if not parts:
        return None

    try:
        search_url = f"{SHERDOG_SEARCH_URL}?SearchTxt={requests.utils.quote(fighter_name)}"
        soup = _get_soup(search_url)
    except Exception as e:
        logger.warning(f"Sherdog search failed for '{fighter_name}': {e}")
        return None

    # Results are in table.fightfinder_result
    table = soup.find("table", class_="fightfinder_result")
    if not table:
        return None

    best_url = None
    best_score = 0

    for row in table.find_all("tr"):
        link = row.find("a", href=lambda h: h and "/fighter/" in h)
        if not link:
            continue

        found_name = _clean_text(link.text).lower()
        href = link.get("href", "")

        # Exact match
        if found_name == name_lower:
            full_url = f"{SHERDOG_BASE_URL}{href}" if href.startswith("/") else href
            _sherdog_url_cache[fighter_name] = full_url
            return full_url

        # Reversed name match (Eastern name order)
        found_parts = found_name.split()
        if len(found_parts) >= 2:
            reversed_name = f"{found_parts[-1]} {' '.join(found_parts[:-1])}"
            if reversed_name == name_lower:
                full_url = f"{SHERDOG_BASE_URL}{href}" if href.startswith("/") else href
                _sherdog_url_cache[fighter_name] = full_url
                return full_url

        # Score partial matches
        score = 0
        if len(found_parts) >= 2 and len(parts) >= 2:
            if found_parts[-1] == parts[-1]:
                score += 5
            if found_parts[0] == parts[0]:
                score += 5
            elif found_parts[0] and parts[0] and found_parts[0][0] == parts[0][0]:
                score += 2
        if name_lower in found_name or found_name in name_lower:
            score += 3

        if score > best_score:
            best_score = score
            best_url = f"{SHERDOG_BASE_URL}{href}" if href.startswith("/") else href

    if best_url and best_score >= 5:
        _sherdog_url_cache[fighter_name] = best_url
        return best_url

    return None


def scrape_sherdog_page(fighter_url: str, fighter_name: str) -> tuple[dict, list[dict]]:
    """
    Scrape a Sherdog fighter profile page in a single request.

    Returns (profile_dict, fights_list). Profile matches UFCStats format;
    career rate stats are NaN. Fights are in chronological order (oldest first).
    """
    soup = _get_soup(fighter_url)

    # --- Profile ---

    # Name from h1
    name_el = soup.find("h1")
    name = _clean_text(name_el.text) if name_el else ""

    # Record from winloses divs
    wins, losses, draws = 0, 0, 0
    for div in soup.find_all("div", class_="winloses"):
        spans = div.find_all("span")
        if len(spans) >= 2:
            label = spans[0].text.strip().lower()
            count = _safe_float(spans[1].text.strip(), default=0)
            if label == "wins":
                wins = int(count)
            elif label == "losses":
                losses = int(count)
            elif label == "draws":
                draws = int(count)

    record_str = f"{wins}-{losses}-{draws}"

    # Physical attributes from bio table
    # Structure: <tr><td>LABEL</td><td>VALUE</td></tr>
    height = np.nan
    weight = np.nan
    age = np.nan

    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) != 2:
            continue

        label = tds[0].text.strip().upper()
        value = tds[1].text.strip()

        if label == "HEIGHT":
            # Format: 5'10" / 177.8 cm — parse the imperial part
            match = re.search(r"(\d+)'(\d+)", value)
            if match:
                height = int(match.group(1)) * 12 + int(match.group(2))
        elif label == "WEIGHT":
            # Format: 185 lbs / 83.91 kg — parse the lbs part
            match = re.search(r"(\d+)\s*lbs", value)
            if match:
                weight = float(match.group(1))
        elif label == "AGE":
            # Format: 25 / Dec 30, 2000
            match = re.search(r"(\d+)\s*/", value)
            if match:
                age = float(match.group(1))

    profile = {
        "name": name,
        "fighter_url": fighter_url,
        "record": record_str,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "height": height,
        "reach": np.nan,  # Sherdog does not provide reach
        "weight": weight,
        "stance": "",  # Sherdog does not provide stance
        "age": age,
        **_empty_profile_stats(),
    }

    # --- Fight history ---

    fights = []

    # Fight history is in table.fighter (there may be multiple — pro, amateur)
    # We take the first one (pro fights)
    fight_tables = soup.find_all("table", class_="fighter")
    if not fight_tables:
        return profile, fights

    table = fight_tables[0]
    rows = table.find_all("tr")

    for row in rows[1:]:  # skip header
        tds = row.find_all("td")
        if len(tds) < 6:
            continue

        try:
            # Column 0: Result (win/loss/draw/nc)
            result_text = tds[0].text.strip().lower()
            if result_text == "win":
                won = 1
            elif result_text == "draw":
                won = 0  # Treated as not-a-win for streak purposes
            else:
                won = 0

            # Column 1: Opponent
            opp_link = tds[1].find("a")
            opponent = _clean_text(opp_link.text) if opp_link else _clean_text(tds[1].text)

            # Column 2: Event + date
            # Date is embedded in the event cell text, format: "...Mon / DD / YYYY"
            event_text = tds[2].get_text()
            date_match = re.search(
                r"([A-Z][a-z]{2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})", event_text
            )
            event_date = None
            if date_match:
                try:
                    event_date = datetime.strptime(
                        f"{date_match.group(1)} {date_match.group(2)} {date_match.group(3)}",
                        "%b %d %Y",
                    )
                except ValueError:
                    pass

            # Column 3: Method — in <b> tag, referee is in <span class="sub_line">
            method_cell = tds[3]
            method_b = method_cell.find("b")
            method = _clean_text(method_b.text) if method_b else _clean_text(method_cell.get_text())

            # Column 4: Round
            round_text = tds[4].text.strip()
            round_finished = int(round_text) if round_text.isdigit() else None

            # Detect title bout from event name
            event_link = tds[2].find("a")
            event_name = _clean_text(event_link.text).lower() if event_link else ""
            is_title = "title" in event_name or "championship" in event_name

            fight = {
                "event_date": event_date,
                "opponent": opponent,
                "won": won,
                "method": method,
                "round_finished": round_finished,
                "is_title_bout": is_title,
                **_empty_fight_dict(),
            }
            fights.append(fight)

        except Exception as e:
            logger.debug(f"Sherdog: failed to parse fight row for {fighter_name}: {e}")
            continue

    # Reverse to chronological order (Sherdog shows most recent first)
    fights.reverse()
    return profile, fights


# ---------------------------------------------------------------------------
# Tapology scraper (stub — Cloudflare-protected, needs cloudscraper)
# ---------------------------------------------------------------------------

def search_tapology(fighter_name: str) -> Optional[str]:
    """
    Search Tapology for a fighter by name.

    NOTE: Tapology uses Cloudflare challenge protection. This will fail with
    standard requests. To enable, install `cloudscraper` and replace the
    requests.get call with cloudscraper.create_scraper().get().
    """
    # Tapology is Cloudflare-protected; standard requests always return 403.
    # Stubbed out for future implementation with cloudscraper or similar.
    logger.debug(f"Tapology lookup skipped for '{fighter_name}' (Cloudflare-protected)")
    return None


def scrape_tapology_profile(fighter_url: str) -> dict:
    """Stub — not yet implemented due to Cloudflare protection."""
    return {}


def scrape_tapology_fights(fighter_url: str, fighter_name: str) -> list[dict]:
    """Stub — not yet implemented due to Cloudflare protection."""
    return []


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def fallback_lookup(fighter_name: str) -> Optional[tuple[dict, list[dict]]]:
    """
    Try fallback sources (Sherdog → Tapology) for a fighter's data.

    Returns (profile_dict, fights_list) or None if all sources fail.
    Both dicts match the UFCStats format used by _compute_rolling_for_fighter.
    """
    # Try Sherdog first
    try:
        sherdog_url = search_sherdog(fighter_name)
        if sherdog_url:
            logger.info(f"  Found {fighter_name} on Sherdog: {sherdog_url}")
            profile, fights = scrape_sherdog_page(sherdog_url, fighter_name)
            if profile and profile.get("name"):
                return profile, fights
    except Exception as e:
        logger.warning(f"Sherdog fallback failed for {fighter_name}: {e}")

    # Try Tapology (currently stubbed due to Cloudflare)
    try:
        tapology_url = search_tapology(fighter_name)
        if tapology_url:
            logger.info(f"  Found {fighter_name} on Tapology: {tapology_url}")
            profile = scrape_tapology_profile(tapology_url)
            if profile and profile.get("name"):
                fights = scrape_tapology_fights(tapology_url, fighter_name)
                return profile, fights
    except Exception as e:
        logger.warning(f"Tapology fallback failed for {fighter_name}: {e}")

    return None


def clear_fallback_cache():
    """Clear all fallback scraper caches."""
    _sherdog_url_cache.clear()
