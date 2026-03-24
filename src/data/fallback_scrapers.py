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
from difflib import SequenceMatcher
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
from src.data.name_utils import (
    normalize_cross_source_name,
    normalize_person_name,
    same_person_name,
)

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
TAPOLOGY_REQUEST_DELAY = 3.0
TAPOLOGY_TIMEOUT_SECONDS = 45
TAPOLOGY_MAX_RETRIES = 4
MARTIALBOT_REQUEST_DELAY = 1.5
FIGHTDX_SITE_BASE_URL = FIGHTDX_BASE_URL.rsplit("/person", 1)[0]
FIGHTDX_SITEMAP_INDEX_URL = f"{FIGHTDX_SITE_BASE_URL.rstrip('/')}/sitemap.xml"
FIGHTDX_SITEMAP_REQUEST_DELAY = 0.1

# Session caches
_sherdog_url_cache: dict[str, str] = {}
_tapology_url_cache: dict[str, str] = {}
_martialbot_url_cache: dict[str, str] = {}
_fightdx_url_cache: dict[str, str] = {}
_fightdx_person_urls_cache: list[str] | None = None
_tapology_scraper = None
_last_tapology_request_at = 0.0
_tapology_blocked: bool | None = None  # None = not yet tested
_tapology_search_blocked = False

_MANUAL_SEARCH_ALIASES: dict[str, list[str]] = {
    "dmitrii smoliakov": ["Dmitry Smoliakov", "Dmitry Smolyakov"],
    "rafael cerquiera": ["Rafael Cerqueira"],
    "seokhyeon ko": ["Seok Hyeon Ko", "Seok-hyeon Ko"],
    "tsuyoshi kohsaka": ["Tsuyoshi Kosaka"],
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class TapologyRequestError(RuntimeError):
    def __init__(
        self,
        url: str,
        *,
        status_code: int | None = None,
        detail: str = "",
    ) -> None:
        self.url = url
        self.status_code = status_code
        message = f"Tapology request failed for {url}"
        if status_code is not None:
            message += f" (status {status_code})"
        if detail:
            message += f": {detail}"
        super().__init__(message)


def _get_soup(url: str, *, max_retries: int = 2) -> BeautifulSoup:
    """Fetch a URL and return parsed BeautifulSoup with retry on timeout."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return BeautifulSoup(resp.text, "lxml")
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            if attempt < max_retries:
                backoff = REQUEST_DELAY * attempt
                logger.debug("Request to %s timed out (attempt %d/%d); retrying in %.1fs", url, attempt, max_retries, backoff)
                time.sleep(backoff)
    raise last_exc  # type: ignore[misc]


def _build_tapology_scraper():
    if cloudscraper is None:
        raise RuntimeError("Tapology scraping requires the optional 'cloudscraper' dependency")
    import platform as _plat

    os_platform = "linux" if _plat.system().lower() == "linux" else "windows"
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": os_platform, "mobile": False}
    )
    scraper.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            )
            if os_platform == "linux"
            else (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": HEADERS["Accept"],
            "Accept-Language": HEADERS["Accept-Language"],
        }
    )
    return scraper


def _check_tapology_blocked() -> bool:
    """One-time probe: can this runtime reach Tapology at all?"""
    global _tapology_blocked

    if _tapology_blocked is not None:
        return _tapology_blocked

    try:
        scraper = _build_tapology_scraper()
        resp = scraper.get(f"{TAPOLOGY_BASE_URL}/fightcenter", timeout=15)
        _tapology_blocked = resp.status_code == 403
    except Exception:
        _tapology_blocked = True

    if _tapology_blocked:
        logger.info(
            "Tapology is blocked from this environment; all Tapology lookups will be skipped"
        )
    else:
        logger.info("Tapology connectivity check passed")
    return _tapology_blocked


def _get_tapology_soup(
    url: str,
    *,
    params: dict | None = None,
    max_retries: int | None = None,
    retry_statuses: set[int] | None = None,
) -> BeautifulSoup:
    """Fetch a Tapology page with challenge-aware retries."""
    global _tapology_scraper, _last_tapology_request_at

    if _check_tapology_blocked():
        raise TapologyRequestError(
            url,
            status_code=403,
            detail="Tapology blocked from this environment",
        )

    max_attempts = max(1, int(max_retries or TAPOLOGY_MAX_RETRIES))
    retry_statuses = set(retry_statuses or {403, 429, 503})
    last_error: Exception | None = None
    last_status: int | None = None
    for attempt in range(1, max_attempts + 1):
        if _tapology_scraper is None:
            _tapology_scraper = _build_tapology_scraper()

        sleep_for = TAPOLOGY_REQUEST_DELAY - (time.monotonic() - _last_tapology_request_at)
        if sleep_for > 0:
            time.sleep(sleep_for)

        try:
            resp = _tapology_scraper.get(
                url,
                params=params,
                timeout=TAPOLOGY_TIMEOUT_SECONDS,
            )
            _last_tapology_request_at = time.monotonic()
        except Exception as exc:  # pragma: no cover - network-only branch
            last_error = exc
            if attempt >= max_attempts:
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
            continue

        if resp.status_code == 200 and resp.text:
            return BeautifulSoup(resp.text, "lxml")

        last_status = int(resp.status_code)
        if resp.status_code in retry_statuses:
            last_error = TapologyRequestError(url, status_code=resp.status_code)
            if attempt >= max_attempts:
                break
            backoff = TAPOLOGY_REQUEST_DELAY * (2 ** attempt)  # exponential: 6, 12, 24, 48s
            logger.warning(
                "Tapology request to %s returned %s (attempt %d/%d); "
                "rebuilding session and retrying in %.1fs",
                url,
                resp.status_code,
                attempt,
                max_attempts,
                backoff,
            )
            _tapology_scraper = _build_tapology_scraper()
            time.sleep(backoff)
            continue

        try:
            resp.raise_for_status()
        except Exception as exc:
            raise TapologyRequestError(url, status_code=resp.status_code) from exc

        raise TapologyRequestError(url, status_code=resp.status_code, detail="empty response body")

    raise TapologyRequestError(url, status_code=last_status) from last_error


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _titleize_slug(value: str) -> str:
    text = re.sub(r"^\d+-", "", str(value or "").strip().lower())
    text = text.replace("-", " ").strip()
    tokens = text.split()
    if len(tokens) > 2 and tokens[-1].isalpha() and len(tokens[-1]) <= 4:
        tokens = tokens[:-1]
    text = " ".join(tokens)
    text = re.sub(r"\bm 1\b", "m-1", text)
    replacements = {
        "lfa": "LFA",
        "ufc": "UFC",
        "kotc": "KOTC",
        "cffc": "CFFC",
        "cwfc": "CWFC",
        "ec": "EC",
    }
    tokens = []
    for token in text.split():
        tokens.append(replacements.get(token, token.capitalize()))
    return " ".join(tokens)


def _slugify_person_name(name: str) -> str:
    return normalize_person_name(name).replace(" ", "-")


def _split_camel_token(token: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(token or ""))


def _name_query_variants(fighter_name: str) -> list[str]:
    tokens = str(fighter_name or "").strip().split()
    if not tokens:
        return []

    variants = [fighter_name]
    variants.extend(_MANUAL_SEARCH_ALIASES.get(normalize_person_name(fighter_name), []))

    first_token = tokens[0]
    if len(first_token) == 2 and first_token.isalpha():
        dotted_initials = ".".join(first_token.upper()) + "."
        variants.append(f"{dotted_initials} {' '.join(tokens[1:])}".strip())

    spaced_name = " ".join(_split_camel_token(token) for token in tokens).strip()
    if spaced_name and spaced_name != fighter_name:
        variants.append(spaced_name)

    if re.search(r"\bjunior\b", fighter_name, flags=re.IGNORECASE):
        variants.append(re.sub(r"\bjunior\b", "Jr.", fighter_name, flags=re.IGNORECASE))
    if re.search(r"\bjr\.?\b", fighter_name, flags=re.IGNORECASE):
        variants.append(re.sub(r"\bjr\.?\b", "Junior", fighter_name, flags=re.IGNORECASE))

    return list(dict.fromkeys(variant.strip() for variant in variants if str(variant).strip()))


def _strip_name_nicknames(value: str) -> str:
    text = str(value or "")
    text = re.sub(r'"[^"]+"', " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    return _clean_text(text)


def _name_variants(value: str, href: str = "") -> set[str]:
    variants: set[str] = set()
    text = _clean_text(str(value or ""))
    if text:
        variants.add(text)
        stripped = _strip_name_nicknames(text)
        if stripped:
            variants.add(stripped)

    slug = str(href or "").rstrip("/").split("/")[-1]
    slug = re.sub(r"^\d+-", "", slug)
    slug_text = _clean_text(slug.replace("-", " "))
    if slug_text:
        variants.add(slug_text)
        stripped_slug = _strip_name_nicknames(slug_text)
        if stripped_slug:
            variants.add(stripped_slug)

    return {variant for variant in variants if variant}


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


def _parse_height_cm(raw: str) -> float:
    """Parse height from any common format to centimeters.

    Handles: "5'10\"", "5'10\" (177.8 cm)", "5' 10", "178 cm", etc.
    """
    if not raw or raw in ("--", "N/A", "??"):
        return np.nan
    # Try feet'inches first
    match = re.search(r"(\d+)'?\s*(\d+)", raw)
    if match:
        inches = int(match.group(1)) * 12 + int(match.group(2))
        return _inches_to_cm(inches)
    # Try direct cm value
    cm_match = re.search(r"(\d+(?:\.\d+)?)\s*cm", raw)
    if cm_match:
        return round(float(cm_match.group(1)), 2)
    return np.nan


def _parse_reach_cm(raw: str) -> float:
    """Parse reach from any common format to centimeters.

    Handles: "74\"", "74", "74 (in)", "188 cm", "1.85m", etc.
    """
    if not raw or raw in ("--", "N/A", "??"):
        return np.nan
    # Try cm first (longer number likely cm)
    cm_match = re.search(r"(\d+(?:\.\d+)?)\s*cm", raw)
    if cm_match:
        return round(float(cm_match.group(1)), 2)
    # Try meters (e.g. "1.85m")
    m_match = re.search(r"(\d+\.\d+)\s*m\b", raw)
    if m_match:
        return round(float(m_match.group(1)) * 100, 2)
    # Try inches with quote mark (e.g. 75.0")
    inch_match = re.search(r'(\d+(?:\.\d+)?)\s*["\u2033]', raw)
    if inch_match:
        return _inches_to_cm(float(inch_match.group(1)))
    # Fallback: bare number
    match = re.search(r"(\d+)", raw)
    if match:
        val = float(match.group(1))
        # Sanity: reach in inches is typically 60-85; in cm it's 150-220
        if val > 120:
            return round(val, 2)  # Already cm
        return _inches_to_cm(val)
    return np.nan


def _parse_weight_lbs(raw: str) -> float:
    """Parse weight to lbs from any common format.

    Handles: "185 lbs", "170 lbs / 77.11 kg", "185", etc.
    """
    if not raw or raw in ("--", "N/A", "??"):
        return np.nan
    match = re.search(r"(\d+)", raw)
    if match:
        return float(match.group(1))
    return np.nan


def _parse_age_from_raw(raw: str) -> float:
    """Parse a direct age value like '25' or '25 years'."""
    if not raw or raw in ("--", "N/A", "??"):
        return np.nan
    match = re.search(r"(\d+)", raw)
    if match:
        val = float(match.group(1))
        if 15 <= val <= 65:  # Sanity check for fighter age range
            return val
    return np.nan


def _parse_dob_to_age(dob_str: str) -> float:
    """Parse DOB string to age in years.

    Handles: "Sep 22, 1989", "September 22, 1989", "1989-09-22", etc.
    """
    if not dob_str or dob_str in ("--", "N/A", "-", "??"):
        return np.nan
    from datetime import datetime
    now = datetime.now()
    for fmt in ["%b %d, %Y", "%B %d, %Y", "%Y-%m-%d", "%m/%d/%Y"]:
        try:
            dob = datetime.strptime(dob_str.strip(), fmt)
            age = (now - dob).days / 365.25
            return round(age, 1)
        except ValueError:
            continue
    return np.nan


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

    if not normalize_person_name(fighter_name):
        return None

    best_url = None
    best_score = 0

    for query in _name_query_variants(fighter_name):
        try:
            search_url = f"{SHERDOG_SEARCH_URL}?SearchTxt={requests.utils.quote(query)}"
            soup = _get_soup(search_url)
        except Exception as e:
            logger.warning(f"Sherdog search failed for '{query}': {e}")
            continue

        table = soup.find("table", class_="fightfinder_result")
        candidate_links = []
        if table:
            for row in table.find_all("tr"):
                link = row.find("a", href=lambda h: h and "/fighter/" in h)
                if link:
                    candidate_links.append(link)
        else:
            candidate_links = soup.find_all("a", href=lambda h: h and "/fighter/" in h)

        for link in candidate_links:
            found_name = _clean_text(link.text)
            href = link.get("href", "")
            full_url = f"{SHERDOG_BASE_URL}{href}" if href.startswith("/") else href
            score = _best_name_score(fighter_name, found_name, href)
            if score >= 100:
                _sherdog_url_cache[fighter_name] = full_url
                return full_url

            if score > best_score:
                best_score = score
                best_url = full_url

    if best_url and best_score >= 10:
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
            event_text = tds[2].get_text(" ", strip=True)
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
            event_name = _clean_text(event_link.text) if event_link else _clean_text(event_text)
            event_name_lower = event_name.lower()
            is_title = "title" in event_name_lower or "championship" in event_name_lower

            fight = {
                "result": result_text,
                "event_date": event_date,
                "event_name": event_name,
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

    query_compact = query_key.replace(" ", "")
    candidate_compact = candidate_key.replace(" ", "")
    if query_compact and query_compact == candidate_compact:
        return 95

    query_tokens = query_key.split()
    candidate_tokens = candidate_key.split()
    score = 0
    if query_tokens and candidate_tokens and query_tokens[-1] == candidate_tokens[-1]:
        score += 6
    if query_tokens and candidate_tokens and query_tokens[0] == candidate_tokens[0]:
        score += 6
    elif query_tokens and candidate_tokens and query_tokens[0][0] == candidate_tokens[0][0]:
        score += 2

    score += 2 * len(set(query_tokens) & set(candidate_tokens))

    if query_key in candidate_key or candidate_key in query_key:
        score += 8
    if query_compact and candidate_compact and (query_compact in candidate_compact or candidate_compact in query_compact):
        score += 8

    if query_tokens and len(query_tokens) <= len(candidate_tokens):
        for idx in range(len(candidate_tokens) - len(query_tokens) + 1):
            if candidate_tokens[idx:idx + len(query_tokens)] == query_tokens:
                score += 10
                break

    if len(query_tokens) == len(candidate_tokens):
        score += 3 * sum(
            1
            for query_token, candidate_token in zip(query_tokens, candidate_tokens)
            if SequenceMatcher(None, query_token, candidate_token).ratio() >= 0.8
        )

    ratio = SequenceMatcher(None, query_compact, candidate_compact).ratio()
    if ratio >= 0.97:
        score += 20
    elif ratio >= 0.92:
        score += 12
    elif ratio >= 0.85:
        score += 8
    elif ratio >= 0.75:
        score += 4

    return score


def _best_name_score(query: str, candidate_name: str, href: str = "") -> int:
    candidate_variants = _name_variants(candidate_name, href)
    for variant in candidate_variants:
        if same_person_name(query, variant):
            return 100

    query_keys = {
        normalize_person_name(query),
        normalize_cross_source_name(query),
    }
    candidate_keys = {
        key
        for variant in candidate_variants
        for key in (normalize_person_name(variant), normalize_cross_source_name(variant))
        if key
    }

    best_score = 0
    for query_key in query_keys:
        for candidate_key in candidate_keys:
            best_score = max(best_score, _name_score(query_key, candidate_key))
    return best_score



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


def search_tapology_candidates(fighter_name: str, limit: int = 5) -> list[str]:
    global _tapology_search_blocked
    # If Tapology is completely blocked, don't bother searching at all —
    # we'd just find URLs we can't scrape.
    if _check_tapology_blocked():
        return []
    scored_urls: dict[str, int] = {}
    if not _tapology_search_blocked:
        for query in _name_query_variants(fighter_name):
            try:
                soup = _get_tapology_soup(
                    TAPOLOGY_SEARCH_URL,
                    params={"term": query},
                    max_retries=1,
                    retry_statuses={429, 503},
                )
            except TapologyRequestError as exc:
                if exc.status_code == 403:
                    _tapology_search_blocked = True
                    logger.info(
                        "Tapology native search returned 403; disabling native search for this runtime"
                    )
                    break
                logger.warning("Tapology search failed for '%s': %s", query, exc)
                continue
            except Exception as exc:
                logger.warning("Tapology search failed for '%s': %s", query, exc)
                continue

            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                if "/fightcenter/fighters/" not in href:
                    continue
                candidate_name = _clean_text(link.get_text(" ", strip=True).replace('"', " "))
                score = _best_name_score(fighter_name, candidate_name, href)
                if score <= 0:
                    continue
                full_url = f"{TAPOLOGY_BASE_URL}{href}" if href.startswith("/") else href
                previous = scored_urls.get(full_url, 0)
                if score > previous:
                    scored_urls[full_url] = score

    ranked_urls = sorted(scored_urls.items(), key=lambda item: item[1], reverse=True)
    return [url for url, score in ranked_urls if score >= 8][:limit]


def search_tapology(fighter_name: str) -> Optional[str]:
    """Search Tapology for a fighter by name and return their full profile URL."""
    if fighter_name in _tapology_url_cache:
        return _tapology_url_cache[fighter_name]

    candidates = search_tapology_candidates(fighter_name, limit=1)
    if candidates:
        best_url = candidates[0]
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

    age_raw = age_card[1].strip() if len(age_card) >= 2 else ""
    # Compute age: prefer DOB (more precise), fall back to raw age string
    age = _parse_dob_to_age(dob) if dob else _parse_age_from_raw(age_raw)

    return {
        "name": _parse_tapology_title_name(soup),
        "fighter_url": fighter_url,
        "record": record,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "height_raw": height_raw,
        "height": _parse_height_cm(height_raw),
        "reach_raw": reach_raw,
        "reach": _parse_reach_cm(reach_raw),
        "weight_raw": weight_raw,
        "weight": _parse_weight_lbs(weight_raw),
        "stance": "",
        "age_raw": age_raw,
        "age": age,
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


def _load_fightdx_person_urls() -> list[str]:
    global _fightdx_person_urls_cache

    if _fightdx_person_urls_cache is not None:
        return _fightdx_person_urls_cache

    person_sitemap_urls: list[str] = []
    try:
        response = requests.get(FIGHTDX_SITEMAP_INDEX_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "xml")
        person_sitemap_urls = [
            loc.get_text(strip=True)
            for loc in soup.find_all("loc")
            if "_people.xml" in loc.get_text(strip=True)
        ]
    except Exception as exc:
        logger.warning("FightDX sitemap index lookup failed: %s", exc)
        _fightdx_person_urls_cache = []
        return _fightdx_person_urls_cache

    person_urls: list[str] = []
    seen_urls: set[str] = set()
    for sitemap_url in person_sitemap_urls:
        try:
            response = requests.get(sitemap_url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "xml")
            for loc in soup.find_all("loc"):
                candidate_url = loc.get_text(strip=True)
                if "/person/" not in candidate_url or candidate_url in seen_urls:
                    continue
                seen_urls.add(candidate_url)
                person_urls.append(candidate_url)
            _sleep_after_request(FIGHTDX_SITEMAP_REQUEST_DELAY)
        except Exception as exc:
            logger.warning("FightDX sitemap page lookup failed for '%s': %s", sitemap_url, exc)
            continue

    _fightdx_person_urls_cache = person_urls
    return _fightdx_person_urls_cache


def _search_fightdx_sitemap_candidates(fighter_name: str, limit: int = 5) -> list[str]:
    scored_urls: dict[str, int] = {}
    for candidate_url in _load_fightdx_person_urls():
        score = _best_name_score(fighter_name, "", candidate_url)
        if score <= 0:
            continue
        previous = scored_urls.get(candidate_url, 0)
        if score > previous:
            scored_urls[candidate_url] = score

    ranked_urls = sorted(scored_urls.items(), key=lambda item: item[1], reverse=True)
    return [url for url, score in ranked_urls if score >= 8][:limit]


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
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            candidate_name = _parse_fightdx_heading_name(soup)
            score = _name_score(
                normalize_person_name(fighter_name),
                normalize_person_name(candidate_name),
            )
            if score >= 8:
                _fightdx_url_cache[fighter_name] = url
                _sleep_after_request(REQUEST_DELAY)
                return url
    except Exception as exc:
        logger.warning("FightDX lookup failed for '%s': %s", fighter_name, exc)
    for candidate_url in _search_fightdx_sitemap_candidates(fighter_name):
        try:
            response = requests.get(candidate_url, headers=HEADERS, timeout=30)
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, "lxml")
            candidate_name = _parse_fightdx_heading_name(soup)
            verified_score = _name_score(
                normalize_person_name(fighter_name),
                normalize_person_name(candidate_name),
            )
            if verified_score < 8:
                continue
            _fightdx_url_cache[fighter_name] = candidate_url
            _sleep_after_request(REQUEST_DELAY)
            return candidate_url
        except Exception as exc:
            logger.warning("FightDX sitemap lookup failed for '%s': %s", fighter_name, exc)
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

    height_raw = details.get("Height", "")
    reach_raw = details.get("Reach", "")
    weight_raw = details.get("Weight", "")
    age_raw = details.get("Age", "")
    dob = details.get("Date of Birth", "")
    dob = "" if dob in {"", "-"} else dob
    record = details.get("Record", "")
    wins, losses, draws = _parse_record_triplet(record)

    # Compute age: prefer DOB, fall back to raw age string
    age = _parse_dob_to_age(dob) if dob else _parse_age_from_raw(age_raw)

    return {
        "name": name,
        "fighter_url": fighter_url,
        "record": record,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "height_raw": height_raw,
        "height": _parse_height_cm(height_raw),
        "reach_raw": reach_raw,
        "reach": _parse_reach_cm(reach_raw),
        "weight_raw": weight_raw,
        "weight": _parse_weight_lbs(weight_raw),
        "stance": details.get("Style", ""),
        "age_raw": age_raw,
        "age": age,
        "dob": dob,
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
    weight_raw = details.get("Weight", "")
    age_raw = details.get("Age", "")
    dob = details.get("Born", "")

    # Compute age: prefer DOB, fall back to raw age string
    age = _parse_dob_to_age(dob) if dob else _parse_age_from_raw(age_raw)

    return {
        "name": name,
        "fighter_url": fighter_url,
        "record": record,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "height_raw": height_raw,
        "height": _parse_height_cm(height_raw),
        "reach_raw": reach_raw,
        "reach": _parse_reach_cm(reach_raw),
        "weight_raw": weight_raw,
        "weight": _parse_weight_lbs(weight_raw),
        "stance": details.get("Stance", ""),
        "age_raw": age_raw,
        "age": age,
        "dob": dob,
        **_empty_profile_stats(),
    }


def scrape_tapology_fights(fighter_url: str, fighter_name: str) -> list[dict]:
    """Scrape Tapology fight history blocks for a fighter page."""
    soup = _get_tapology_soup(fighter_url)
    fights: list[dict] = []

    for block in soup.select("[data-bout-id]"):
        if block.get("data-sport") != "mma":
            continue
        if block.get("data-division") != "pro":
            continue

        status = str(block.get("data-status") or "").strip().lower()
        if status in {"cancelled", "booking", "scheduled"}:
            continue

        block_texts = [
            _clean_text(text)
            for text in block.stripped_strings
            if _clean_text(text)
        ]

        result_row = block.find("div", class_="result")
        result_children = result_row.find_all("div", recursive=False) if result_row else []
        method_code = ""
        if len(result_children) >= 2:
            method_code = _clean_text(result_children[1].get_text(" ", strip=True)).upper()

        fighter_links = block.find_all(
            "a",
            href=lambda h: h and "/fightcenter/fighters/" in h,
            title=lambda t: t and "Fighter Page" in t,
        )
        if not fighter_links:
            continue
        opponent = _clean_text(fighter_links[0].get_text(" ", strip=True))
        if not opponent:
            continue

        bout_links = block.find_all(
            "a",
            href=lambda h: h and "/fightcenter/bouts/" in h,
            title=lambda t: t and "Bout Page" in t,
        )
        bout_texts = [
            _clean_text(link.get_text(" ", strip=True))
            for link in bout_links
            if _clean_text(link.get_text(" ", strip=True))
        ]
        method_detail = bout_texts[0] if bout_texts else ""
        secondary_detail = bout_texts[1] if len(bout_texts) > 1 else ""

        event_links = block.find_all(
            "a",
            href=lambda h: h and "/fightcenter/events/" in h,
        )
        event_name = ""
        date_text = ""
        for link in event_links:
            text = _clean_text(link.get_text(" ", strip=True))
            if not text:
                continue
            if re.match(r"^\d{4}\s+[A-Z][a-z]{2}\s+\d{1,2}$", text):
                date_text = text
            elif len(text) > len(event_name):
                event_name = text

        if not date_text:
            for link in block.find_all("a", href=True):
                text = _clean_text(link.get_text(" ", strip=True))
                if re.match(r"^\d{4}\s+[A-Z][a-z]{2}\s+\d{1,2}$", text):
                    date_text = text
                    break

        event_date = None
        if date_text:
            try:
                event_date = datetime.strptime(date_text, "%Y %b %d")
            except ValueError:
                event_date = None

        promotion_name = ""
        promo_link = block.find("a", href=lambda h: h and "/fightcenter/promotions/" in h)
        if promo_link:
            slug = str(promo_link.get("href") or "").rstrip("/").split("/")[-1]
            promotion_name = _titleize_slug(slug)
        if not promotion_name:
            for idx, text in enumerate(block_texts[:-1]):
                if text in {"League:", "Promotion:"}:
                    promotion_name = block_texts[idx + 1]
                    break
        if not promotion_name:
            promotion_name = event_name

        title_bout = 1 if block.find(class_="fighterBeltIcon") else 0

        finish_round = np.nan
        round_match = re.search(r"\bR(\d+)\b", method_detail)
        if round_match:
            finish_round = int(round_match.group(1))

        method_label = method_code
        if method_code == "DEC":
            decision_detail = (secondary_detail or method_detail).lower()
            if "split" in decision_detail:
                method_label = "Decision (Split)"
            elif "majority" in decision_detail:
                method_label = "Decision (Majority)"
            else:
                method_label = "Decision (Unanimous)"
        elif method_code == "SUB":
            finish = secondary_detail or method_detail.split("·", 1)[0]
            method_label = f"Submission ({finish})" if finish else "Submission"
        elif method_code in {"TKO", "KO"}:
            finish = secondary_detail or method_detail.split("·", 1)[0]
            method_label = f"{method_code} ({finish})" if finish else method_code
        elif method_code == "DQ":
            method_label = "Disqualification"
        elif method_code == "NC":
            method_label = "No Contest"
        elif method_code == "DRAW":
            method_label = "Draw"
        elif method_detail:
            method_label = method_detail

        fights.append(
            {
                "event_date": event_date,
                "event_name": event_name,
                "organization": promotion_name,
                "opponent": opponent,
                "result": status,
                "won": 1 if status == "win" else 0,
                "method": method_label,
                "round_finished": finish_round,
                "is_title_bout": title_bout,
                **_empty_fight_dict(),
            }
        )

    fights.reverse()
    return fights


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
    global _fightdx_person_urls_cache, _tapology_scraper, _last_tapology_request_at
    global _tapology_blocked, _tapology_search_blocked
    _fightdx_person_urls_cache = None
    _tapology_scraper = None
    _last_tapology_request_at = 0.0
    _tapology_blocked = None
    _tapology_search_blocked = False
