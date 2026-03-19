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

try:
    import cloudscraper
except ImportError:  # pragma: no cover - optional dependency
    cloudscraper = None

from src.config import (
    FIGHTDX_BASE_URL,
    MARTIALBOT_BASE_URL,
    MARTIALBOT_SEARCH_URL,
    SHERDOG_BASE_URL,
    SHERDOG_SEARCH_URL,
    TAPOLOGY_BASE_URL,
    TAPOLOGY_SEARCH_URL,
)
from src.data.name_utils import normalize_person_name

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
TAPOLOGY_REQUEST_DELAY = 2.0
TAPOLOGY_TIMEOUT_SECONDS = 45
TAPOLOGY_MAX_RETRIES = 4
MARTIALBOT_REQUEST_DELAY = 1.5

# Session caches
_sherdog_url_cache: dict[str, str] = {}
_tapology_url_cache: dict[str, str] = {}
_martialbot_url_cache: dict[str, str] = {}
_fightdx_url_cache: dict[str, str] = {}
_tapology_scraper = None
_last_tapology_request_at = 0.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_soup(url: str) -> BeautifulSoup:
    """Fetch a URL and return parsed BeautifulSoup."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return BeautifulSoup(resp.text, "lxml")


def _build_tapology_scraper():
    if cloudscraper is None:
        raise RuntimeError("Tapology scraping requires the optional 'cloudscraper' dependency")
    return cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )


def _get_tapology_soup(url: str, *, params: dict | None = None) -> BeautifulSoup:
    """Fetch a Tapology page with challenge-aware retries."""
    global _tapology_scraper, _last_tapology_request_at

    last_error: Exception | None = None
    for attempt in range(1, TAPOLOGY_MAX_RETRIES + 1):
        if _tapology_scraper is None:
            _tapology_scraper = _build_tapology_scraper()

        sleep_for = TAPOLOGY_REQUEST_DELAY - (time.monotonic() - _last_tapology_request_at)
        if sleep_for > 0:
            time.sleep(sleep_for)

        try:
            resp = _tapology_scraper.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=TAPOLOGY_TIMEOUT_SECONDS,
            )
            _last_tapology_request_at = time.monotonic()
            if resp.status_code == 200 and resp.text:
                return BeautifulSoup(resp.text, "lxml")
            if resp.status_code in {429, 503}:
                backoff = TAPOLOGY_REQUEST_DELAY * attempt
                logger.warning(
                    "Tapology request to %s failed with %s (attempt %d/%d); retrying in %.1fs",
                    url,
                    resp.status_code,
                    attempt,
                    TAPOLOGY_MAX_RETRIES,
                    backoff,
                )
                _tapology_scraper = _build_tapology_scraper()
                time.sleep(backoff)
                continue
            resp.raise_for_status()
        except Exception as exc:  # pragma: no cover - network-only branch
            last_error = exc
            if attempt >= TAPOLOGY_MAX_RETRIES:
                break
            backoff = TAPOLOGY_REQUEST_DELAY * attempt
            logger.warning(
                "Tapology request to %s failed (%s); retrying in %.1fs",
                url,
                exc,
                backoff,
            )
            _tapology_scraper = _build_tapology_scraper() if cloudscraper is not None else None
            time.sleep(backoff)

    raise RuntimeError(f"Tapology request failed for {url}") from last_error


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _slugify_person_name(name: str) -> str:
    return normalize_person_name(name).replace(" ", "-")


def _sleep_after_request(delay_seconds: float) -> None:
    if delay_seconds > 0:
        time.sleep(delay_seconds)


def _safe_float(value, default=np.nan) -> float:
    if value is None or value == "" or value == "--" or value == "N/A":
        return default
    try:
        return float(str(value).replace("%", "").strip())
    except (ValueError, TypeError):
        return default


def _inches_to_cm(value: float) -> float:
    if value is None or np.isnan(value):
        return np.nan
    return round(float(value) * 2.54, 2)


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
    height_raw = ""
    weight_raw = ""
    age_raw = ""
    dob = ""

    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) != 2:
            continue

        label = tds[0].text.strip().upper()
        value = tds[1].text.strip()

        if label == "HEIGHT":
            # Format: 5'10" / 177.8 cm — parse the imperial part
            height_raw = value
            match = re.search(r"(\d+)'(\d+)", value)
            if match:
                height = _inches_to_cm(int(match.group(1)) * 12 + int(match.group(2)))
        elif label == "WEIGHT":
            # Format: 185 lbs / 83.91 kg — parse the lbs part
            weight_raw = value
            match = re.search(r"(\d+)\s*lbs", value)
            if match:
                weight = float(match.group(1))
        elif label == "AGE":
            # Format: 25 / Dec 30, 2000
            age_raw = value
            match = re.search(r"(\d+)\s*/", value)
            if match:
                age = float(match.group(1))
            dob_match = re.search(r"/\s*([A-Z][a-z]{2,8}\s+\d{1,2},\s+\d{4})", value)
            if dob_match:
                dob = dob_match.group(1).strip()

    profile = {
        "name": name,
        "fighter_url": fighter_url,
        "record": record_str,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "height_raw": height_raw,
        "height": height,
        "reach": np.nan,  # Sherdog does not provide reach
        "weight_raw": weight_raw,
        "weight": weight,
        "stance": "",  # Sherdog does not provide stance
        "age_raw": age_raw,
        "age": age,
        "dob": dob,
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
# Tapology scraper
# ---------------------------------------------------------------------------

def _name_score(query_key: str, candidate_key: str) -> int:
    if not query_key or not candidate_key:
        return 0
    if query_key == candidate_key:
        return 100

    query_tokens = query_key.split()
    candidate_tokens = candidate_key.split()
    score = 0
    if query_tokens and candidate_tokens and query_tokens[-1] == candidate_tokens[-1]:
        score += 5
    if query_tokens and candidate_tokens and query_tokens[0] == candidate_tokens[0]:
        score += 5
    score += len(set(query_tokens) & set(candidate_tokens))
    if query_key in candidate_key or candidate_key in query_key:
        score += 3
    return score


def _tapology_stat_card_values(soup: BeautifulSoup, label: str) -> list[str]:
    label_el = soup.find(string=lambda s: isinstance(s, str) and s.strip() == label)
    if not label_el or not getattr(label_el, "parent", None) or not getattr(label_el.parent, "parent", None):
        return []
    card = label_el.parent.parent
    return [_clean_text(text) for text in card.stripped_strings if _clean_text(text)]


def _parse_tapology_title_name(soup: BeautifulSoup) -> str:
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    match = re.match(r'^(.*?)\s*(?:\(".*"\))?\s*\|\s*MMA Fighter Page \|\s*Tapology$', title)
    return match.group(1).strip() if match else ""


def search_tapology(fighter_name: str) -> Optional[str]:
    """Search Tapology for a fighter by name and return their full profile URL."""
    if fighter_name in _tapology_url_cache:
        return _tapology_url_cache[fighter_name]
    if cloudscraper is None:
        logger.warning("Tapology lookup skipped for '%s': cloudscraper is not installed", fighter_name)
        return None

    try:
        soup = _get_tapology_soup(TAPOLOGY_SEARCH_URL, params={"term": fighter_name})
    except Exception as exc:
        logger.warning("Tapology search failed for '%s': %s", fighter_name, exc)
        return None

    query_key = normalize_person_name(fighter_name)
    best_url = None
    best_score = 0
    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        if "/fightcenter/fighters/" not in href:
            continue
        candidate_name = _clean_text(link.get_text(" ", strip=True).replace('"', " "))
        candidate_key = normalize_person_name(candidate_name)
        score = _name_score(query_key, candidate_key)
        if score > best_score:
            best_score = score
            best_url = f"{TAPOLOGY_BASE_URL}{href}" if href.startswith("/") else href

    if best_url and best_score >= 8:
        _tapology_url_cache[fighter_name] = best_url
        return best_url
    return None


def scrape_tapology_profile(fighter_url: str) -> dict:
    """Scrape a Tapology fighter page for static profile attributes."""
    soup = _get_tapology_soup(fighter_url)

    age_card = _tapology_stat_card_values(soup, "Age")
    height_card = _tapology_stat_card_values(soup, "Height")
    reach_card = _tapology_stat_card_values(soup, "Reach")
    weight_card = _tapology_stat_card_values(soup, "Weight")

    dob = age_card[1].strip() if len(age_card) >= 2 and re.match(r"^\d{4}-\d{2}-\d{2}$", age_card[1].strip()) else ""
    height_raw = ""
    if len(height_card) >= 3:
        height_raw = f"{height_card[1]} ({height_card[2]})"
    elif len(height_card) >= 2:
        height_raw = height_card[1]

    reach_raw = ""
    if len(reach_card) >= 2 and reach_card[1] not in {"N/A", "??"}:
        reach_raw = reach_card[1]
        if len(reach_card) >= 3 and reach_card[2] not in {"N/A", "??"}:
            reach_raw = f"{reach_raw} ({reach_card[2]})"

    weight_raw = ""
    if len(weight_card) >= 2 and weight_card[1] not in {"N/A", "??"}:
        weight_raw = f"{weight_card[1]} lbs"

    record = ""
    summary_text = ""
    for div in soup.find_all("div", class_=True):
        text = " ".join(div.stripped_strings)
        if "Name:" in text and "Pro MMA Record:" in text and len(text) < 1500:
            summary_text = text
            record_match = re.search(r"Pro MMA Record:\s*([0-9\-]+)", text)
            if record_match:
                record = record_match.group(1).strip()
            break

    wins = losses = draws = 0
    if record:
        parts = record.split("-")
        if len(parts) >= 3:
            try:
                wins = int(parts[0])
                losses = int(parts[1])
                draws = int(parts[2])
            except ValueError:
                wins = losses = draws = 0

    return {
        "name": _parse_tapology_title_name(soup),
        "fighter_url": fighter_url,
        "record": record,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "height_raw": height_raw,
        "height": np.nan,
        "reach_raw": reach_raw,
        "reach": np.nan,
        "weight_raw": weight_raw,
        "weight": np.nan,
        "stance": "",
        "age_raw": age_card[1].strip() if len(age_card) >= 2 else "",
        "age": np.nan,
        "dob": dob,
        "summary_text": summary_text,
        **_empty_profile_stats(),
    }


def search_martialbot(fighter_name: str) -> Optional[str]:
    """Search MartialBot for a fighter by name and return their full profile URL."""
    if fighter_name in _martialbot_url_cache:
        return _martialbot_url_cache[fighter_name]

    try:
        response = requests.get(
            MARTIALBOT_SEARCH_URL,
            params={"term": fighter_name},
            headers=HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        _sleep_after_request(MARTIALBOT_REQUEST_DELAY)
    except Exception as exc:
        logger.warning("MartialBot search failed for '%s': %s", fighter_name, exc)
        return None

    query_key = normalize_person_name(fighter_name)
    best_url = None
    best_score = 0
    for result in payload.get("results", []):
        candidate_name = _clean_text(
            str(result.get("display_name") or result.get("name") or "")
        )
        candidate_key = normalize_person_name(candidate_name)
        score = _name_score(query_key, candidate_key)
        candidate_id = str(result.get("id") or "").strip()
        if score > best_score and candidate_id:
            best_score = score
            best_url = f"{MARTIALBOT_BASE_URL}/mma/fighters/{candidate_id}"

    if best_url and best_score >= 8:
        _martialbot_url_cache[fighter_name] = best_url
        return best_url
    return None


def _martialbot_profile_details(soup: BeautifulSoup) -> dict[str, str]:
    details: dict[str, str] = {}
    for dt in soup.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        if not dd:
            continue
        key = _clean_text(dt.get_text(" ", strip=True))
        value = _clean_text(dd.get_text(" ", strip=True))
        if key:
            details[key] = value
    return details


def _parse_record_triplet(record: str) -> tuple[int, int, int]:
    match = re.match(r"^\s*(\d+)-(\d+)(?:-(\d+))?", str(record or "").strip())
    if not match:
        return 0, 0, 0
    wins = int(match.group(1))
    losses = int(match.group(2))
    draws = int(match.group(3) or 0)
    return wins, losses, draws


def _parse_fightdx_heading_name(soup: BeautifulSoup) -> str:
    heading = soup.find("h1")
    if heading:
        return _clean_text(heading.get_text(" ", strip=True))
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    if "|" in title:
        return _clean_text(title.split("|", 1)[0])
    return _clean_text(title)


def search_fightdx(fighter_name: str) -> Optional[str]:
    """Resolve a FightDX profile URL from the fighter's normalized slug."""
    if fighter_name in _fightdx_url_cache:
        return _fightdx_url_cache[fighter_name]

    slug = _slugify_person_name(fighter_name)
    if not slug:
        return None

    url = f"{FIGHTDX_BASE_URL}/{slug}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, "lxml")
        candidate_name = _parse_fightdx_heading_name(soup)
        score = _name_score(
            normalize_person_name(fighter_name),
            normalize_person_name(candidate_name),
        )
        if score < 8:
            return None
        _fightdx_url_cache[fighter_name] = url
        _sleep_after_request(REQUEST_DELAY)
        return url
    except Exception as exc:
        logger.warning("FightDX lookup failed for '%s': %s", fighter_name, exc)
        return None


def scrape_fightdx_profile(fighter_url: str) -> dict:
    """Scrape a FightDX fighter page for static profile attributes."""
    soup = _get_soup(fighter_url)
    name = _parse_fightdx_heading_name(soup)

    details: dict[str, str] = {}
    for label in soup.select("span.info-stat-label"):
        value = label.find_next_sibling("span", class_="info-stat-value")
        if not value:
            continue
        label_text = _clean_text("".join(str(node) for node in label.contents))
        value_text = _clean_text("".join(str(node) for node in value.contents))
        if not label_text:
            continue
        details[label_text] = value_text

    weight_raw = details.get("Weight", "")
    dob = details.get("Date of Birth", "")
    record = details.get("Record", "")
    wins, losses, draws = _parse_record_triplet(record)

    return {
        "name": name,
        "fighter_url": fighter_url,
        "record": record,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "height_raw": details.get("Height", ""),
        "height": np.nan,
        "reach_raw": details.get("Reach", ""),
        "reach": np.nan,
        "weight_raw": weight_raw,
        "weight": np.nan,
        "stance": details.get("Style", ""),
        "age_raw": details.get("Age", ""),
        "age": np.nan,
        "dob": "" if dob in {"", "-"} else dob,
        **_empty_profile_stats(),
    }


def scrape_martialbot_profile(fighter_url: str) -> dict:
    """Scrape a MartialBot fighter page for static profile attributes."""
    soup = _get_soup(fighter_url)
    details = _martialbot_profile_details(soup)
    heading = soup.find("h1")
    name = _clean_text(heading.get_text(" ", strip=True)) if heading else ""
    record = details.get("Record", "")
    wins, losses, draws = _parse_record_triplet(record)

    height_raw = details.get("Height", "")
    reach_raw = details.get("Reach", "")

    return {
        "name": name,
        "fighter_url": fighter_url,
        "record": record,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "height_raw": height_raw,
        "height": np.nan,
        "reach_raw": reach_raw,
        "reach": np.nan,
        "weight_raw": "",
        "weight": np.nan,
        "stance": details.get("Stance", ""),
        "age_raw": details.get("Age", ""),
        "age": np.nan,
        "dob": details.get("Born", ""),
        **_empty_profile_stats(),
    }


def scrape_tapology_fights(fighter_url: str, fighter_name: str) -> list[dict]:
    """Tapology fight-history scraping is not yet implemented."""
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

    # Try Tapology for static profile recovery when Sherdog fails or lacks fields
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

    # Try MartialBot for static profile recovery when other sources miss
    try:
        martialbot_url = search_martialbot(fighter_name)
        if martialbot_url:
            logger.info(f"  Found {fighter_name} on MartialBot: {martialbot_url}")
            profile = scrape_martialbot_profile(martialbot_url)
            if profile and profile.get("name"):
                return profile, []
    except Exception as e:
        logger.warning(f"MartialBot fallback failed for {fighter_name}: {e}")

    # Try FightDX for static profile recovery when other sources miss
    try:
        fightdx_url = search_fightdx(fighter_name)
        if fightdx_url:
            logger.info(f"  Found {fighter_name} on FightDX: {fightdx_url}")
            profile = scrape_fightdx_profile(fightdx_url)
            if profile and profile.get("name"):
                return profile, []
    except Exception as e:
        logger.warning(f"FightDX fallback failed for {fighter_name}: {e}")

    return None


def clear_fallback_cache():
    """Clear all fallback scraper caches."""
    _sherdog_url_cache.clear()
    _tapology_url_cache.clear()
    _martialbot_url_cache.clear()
    _fightdx_url_cache.clear()
    global _tapology_scraper, _last_tapology_request_at
    _tapology_scraper = None
    _last_tapology_request_at = 0.0
