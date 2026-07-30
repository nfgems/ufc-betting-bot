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

import copy
import json
import logging
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
import pandas as pd
from bs4 import BeautifulSoup

from src.config import (
    CARD_SNAPSHOT_MAX_FILES,
    CARD_SNAPSHOT_MAX_PER_EVENT,
    CARD_SNAPSHOT_PAST_EVENT_RETENTION_DAYS,
    CARD_SNAPSHOT_PRUNE_INTERVAL_SECONDS,
    CARD_SNAPSHOT_UNKNOWN_DATE_RETENTION_DAYS,
    LOGS_DIR,
    LIVE_EVENT_CONTEXT_REUSE_TTL_SECONDS,
    RAW_DATA_DIR,
    UFCSTATS_UPCOMING_URL,
)
from src.data.event_context import infer_empty_arena
from src.data.io_utils import write_json_atomically
from src.data.name_utils import normalize_cross_source_name, normalize_person_name

logger = logging.getLogger(__name__)

_METHOD_ODDS_ALERT_INCIDENT_KEY = "method-odds:expected-coverage"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "close",
}
UFC_COM_EVENTS_URL = "https://www.ufc.com/events"
UFC_COM_BASE_URL = "https://www.ufc.com"
UPSTREAM_FETCH_TIMEOUT_SECONDS = (5, 30)
UPSTREAM_FETCH_ATTEMPTS = 3
UPSTREAM_FETCH_RETRY_DELAY_SECONDS = 1.0
# Successful fetches are cached briefly so callers within one betting cycle (the
# upcoming-card scrape and the freshness guard's completed-events scrape both hit
# /events) share one request instead of fetching the same page twice. Failures are
# negative-cached separately so one outage cycle pays the retry ladder once per
# URL instead of once per caller.
UPSTREAM_HTML_CACHE_TTL_SECONDS = 180.0
UPSTREAM_FETCH_FAILURE_TTL_SECONDS = 60.0
_UPSTREAM_HTML_CACHE: dict[str, tuple[float, str]] = {}
_UPSTREAM_FETCH_FAILURE_CACHE: dict[str, float] = {}
_UPSTREAM_FETCH_ALERT_ACTIVE_URLS: set[str] = set()
_UPSTREAM_FETCH_RECOVERY_PROBED_URLS: set[str] = set()
_UPSTREAM_CACHE_LOCK = threading.RLock()
_UPSTREAM_FETCH_LOCK = threading.Lock()
_UFCSTATS_SESSION: requests.Session | None = None
_UFCSTATS_FETCH_LOCK = threading.Lock()
UPCOMING_EVENT_CARD_REQUEST_DELAY_SECONDS = 1.5
_UPCOMING_EVENT_CARDS_LOCK = threading.Lock()
_UPCOMING_EVENT_CARDS_CACHE: tuple[
    float,
    tuple[int, int],
    list[dict],
    bool,
] | None = None
SNAPSHOTS_DIR = RAW_DATA_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
_card_snapshot_state_lock = threading.RLock()
_card_snapshot_prune_lock = threading.Lock()
_last_card_snapshot_prune_monotonic = 0.0
_UFC_COM_EVENT_CTA_TEXT = {
    "Available",
    "Fight Card",
    "How to Watch",
    "Tickets",
    "View Event Details",
    "View Fight Card",
    "Watch Replay",
}


class _EventCardResult(list):
    """List-compatible card result that preserves source-health for empty cards."""

    def __init__(self, rows=(), *, source_healthy: bool):
        super().__init__(rows)
        self.source_healthy = bool(source_healthy)


def _method_odds_snapshot_log_message(snapshot: dict) -> tuple[int, str]:
    status = str(snapshot.get("status", "") or "unknown").lower()
    try:
        record_count = int(snapshot.get("record_count", 0) or 0)
    except (TypeError, ValueError):
        record_count = 0
    if status == "success":
        covered = int(snapshot.get("covered_fight_count", record_count) or 0)
        tracked = int(snapshot.get("tracked_fight_count", covered) or 0)
        coverage = f"; coverage {covered}/{tracked}" if tracked else ""
        return logging.INFO, f"Method-odds snapshot: {status} ({record_count} records){coverage}"

    availability_expected = bool(snapshot.get("availability_expected"))
    expected_fight_count = int(snapshot.get("expected_fight_count", 0) or 0)
    expected_covered_count = int(snapshot.get("expected_covered_fight_count", 0) or 0)
    expected_missing_count = int(snapshot.get("expected_missing_fight_count", 0) or 0)
    try:
        expected_window_hours = float(snapshot.get("expected_window_hours", 48) or 48)
    except (TypeError, ValueError):
        expected_window_hours = 48.0
    log_level = (
        logging.WARNING
        if status == "failed" or availability_expected or expected_missing_count
        else logging.INFO
    )
    state = status if status in {"failed", "partial"} else "unavailable"
    expectation = (
        f"; props expected for {expected_fight_count} fight(s) within "
        f"{expected_window_hours:g}h; covered {expected_covered_count}/{expected_fight_count}"
        if availability_expected
        else ""
    )

    if status == "partial":
        covered = int(snapshot.get("covered_fight_count", record_count) or 0)
        tracked = int(snapshot.get("tracked_fight_count", covered) or 0)
        return (
            log_level,
            f"Method-odds snapshot: partial ({record_count} records; coverage "
            f"{covered}/{tracked}){expectation}; {expected_missing_count} expected fight(s) missing",
        )

    latest_usable = snapshot.get("latest_usable_snapshot")
    if isinstance(latest_usable, dict) and latest_usable:
        fallback_time = latest_usable.get("snapshot_time") or "unknown time"
        try:
            fallback_count = int(latest_usable.get("record_count", 0) or 0)
        except (TypeError, ValueError):
            fallback_count = 0
        freshness = "stale" if latest_usable.get("is_stale") else "fresh"
        return (
            log_level,
            f"Method-odds snapshot: {state} (0 records){expectation}; "
            f"latest matching snapshot is {freshness} from {fallback_time} ({fallback_count} records)",
        )

    return log_level, f"Method-odds snapshot: {state} (0 records){expectation}; no matching fallback snapshot"

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
        or "just a moment" in text
        or "__cf_chl" in text
        or "cf-chl-" in text
    )


def _full_url(url: str, base_url: str = UFC_COM_BASE_URL) -> str:
    return urljoin(base_url, url)


def _cached_upstream_html(url: str) -> Optional[str]:
    with _UPSTREAM_CACHE_LOCK:
        cached = _UPSTREAM_HTML_CACHE.get(url)
        if cached is None:
            return None
        cached_at, cached_html = cached
        if time.monotonic() - cached_at > UPSTREAM_HTML_CACHE_TTL_SECONDS:
            _UPSTREAM_HTML_CACHE.pop(url, None)
            return None
        return cached_html


def _store_upstream_html(url: str, html: str) -> None:
    with _UPSTREAM_CACHE_LOCK:
        now = time.monotonic()
        # Prune on write: event-card URLs for completed cards are never read again,
        # so without this the cache grows for the life of the process.
        for key in [
            key
            for key, (cached_at, _) in list(_UPSTREAM_HTML_CACHE.items())
            if now - cached_at > UPSTREAM_HTML_CACHE_TTL_SECONDS
        ]:
            _UPSTREAM_HTML_CACHE.pop(key, None)
        _UPSTREAM_HTML_CACHE[url] = (now, html)


def _upstream_fetch_recently_failed(url: str, *, label: str) -> bool:
    with _UPSTREAM_CACHE_LOCK:
        failed_at = _UPSTREAM_FETCH_FAILURE_CACHE.get(url)
        if failed_at is None:
            return False
        age_seconds = time.monotonic() - failed_at
        if age_seconds > UPSTREAM_FETCH_FAILURE_TTL_SECONDS:
            _UPSTREAM_FETCH_FAILURE_CACHE.pop(url, None)
            return False
    logger.info(
        "%s fetch skipped: same URL failed %.0fs ago (retrying after %.0fs)",
        label,
        age_seconds,
        UPSTREAM_FETCH_FAILURE_TTL_SECONDS,
    )
    return True


def _record_upstream_fetch_failure(url: str) -> None:
    with _UPSTREAM_CACHE_LOCK:
        now = time.monotonic()
        for key in [
            key
            for key, failed_at in list(_UPSTREAM_FETCH_FAILURE_CACHE.items())
            if now - failed_at > UPSTREAM_FETCH_FAILURE_TTL_SECONDS
        ]:
            _UPSTREAM_FETCH_FAILURE_CACHE.pop(key, None)
        _UPSTREAM_FETCH_FAILURE_CACHE[url] = now


def _upstream_fetch_incident_key(url: str) -> str:
    return f"upstream-fetch:{url}"


def _fetch_upstream_html(
    url: str,
    *,
    label: str,
    timeout=UPSTREAM_FETCH_TIMEOUT_SECONDS,
    attempts: int = UPSTREAM_FETCH_ATTEMPTS,
    required_selector: str | None = None,
    manage_incident: bool = True,
) -> Optional[str]:
    """Fetch a live source with bounded retries and warning-level failures."""
    # Keep cache checks, the network request, and health-state transitions in
    # one critical section. This makes the cache a true single-flight producer
    # and prevents a slower failure from re-opening an incident after a
    # concurrent request already succeeded.
    with _UPSTREAM_FETCH_LOCK:
        cached_html = _cached_upstream_html(url)
        if (
            cached_html is not None
            and required_selector
            and BeautifulSoup(cached_html, "lxml").select_one(required_selector) is None
        ):
            with _UPSTREAM_CACHE_LOCK:
                _UPSTREAM_HTML_CACHE.pop(url, None)
            cached_html = None
        if cached_html is not None:
            return cached_html
        if _upstream_fetch_recently_failed(url, label=label):
            return None

        last_exc: Exception | None = None
        for attempt in range(1, max(1, attempts) + 1):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=timeout)
                resp.raise_for_status()
                response_html = str(resp.text or "")
                if not response_html.strip():
                    raise RuntimeError(f"{label} returned an empty response body")
                if _looks_like_browser_challenge(response_html):
                    raise RuntimeError(
                        f"{label} returned a browser challenge/interstitial"
                    )
                if required_selector and BeautifulSoup(
                    response_html, "lxml"
                ).select_one(required_selector) is None:
                    raise RuntimeError(
                        f"{label} response missing expected markup: {required_selector}"
                    )
                _store_upstream_html(url, response_html)
                # Signal the first real success in every process so a durable alert
                # from before a restart can close. Later healthy successes stay
                # quiet unless this process observed a new failure.
                if manage_incident and (
                    url in _UPSTREAM_FETCH_ALERT_ACTIVE_URLS
                    or url not in _UPSTREAM_FETCH_RECOVERY_PROBED_URLS
                ):
                    logger.info(
                        "%s fetch health probe succeeded",
                        label,
                        extra={
                            "alert_recovered_incident_keys": [
                                _upstream_fetch_incident_key(url)
                            ]
                        },
                    )
                if manage_incident:
                    _UPSTREAM_FETCH_RECOVERY_PROBED_URLS.add(url)
                    _UPSTREAM_FETCH_ALERT_ACTIVE_URLS.discard(url)
                return response_html
            except Exception as exc:
                last_exc = exc
                if attempt < attempts:
                    retry_delay = UPSTREAM_FETCH_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
                    logger.info(
                        "%s fetch failed (attempt %s/%s): %s; retrying in %.1fs",
                        label,
                        attempt,
                        attempts,
                        exc,
                        retry_delay,
                    )
                    time.sleep(retry_delay)

        _record_upstream_fetch_failure(url)
        if manage_incident:
            _UPSTREAM_FETCH_ALERT_ACTIVE_URLS.add(url)
        logger.warning(
            "%s fetch failed after %s attempt(s): %s",
            label,
            max(1, attempts),
            last_exc,
            extra=(
                {"alert_incident_key": _upstream_fetch_incident_key(url)}
                if manage_incident
                else None
            ),
        )
        return None


def _fetch_ufcstats_html(
    url: str,
    *,
    label: str,
    required_selector: str,
) -> Optional[str]:
    """Fetch a UFCStats page through the challenge-solving client.

    UFCStats fronts plain HTTP clients with a JavaScript proof-of-work gate, so the
    generic requests path returns the browser-check page instead of data. The shared
    session persists the gate cookie; the solver re-solves when the gate reappears.
    The lock serializes the betting-loop and background-monitor threads because
    requests.Session cookie handling is not thread-safe, and it deduplicates
    concurrent proof-of-work solves.
    """
    global _UFCSTATS_SESSION

    def _validation_error(html: object) -> str:
        response_html = str(html or "")
        if not response_html.strip():
            return "an empty response body"
        lower_html = response_html[:6000].casefold()
        blocked_markers = (
            "<title>access denied",
            "<title>forbidden",
            "access to this page has been denied",
            "request blocked",
            "you have been blocked",
            "cf-error-details",
        )
        if _looks_like_browser_challenge(response_html) or any(
            marker in lower_html for marker in blocked_markers
        ):
            return "a browser challenge/block page"
        if (
            required_selector
            and BeautifulSoup(response_html, "lxml").select_one(required_selector) is None
        ):
            return f"markup missing expected selector: {required_selector}"
        return ""

    with _UFCSTATS_FETCH_LOCK:
        cached_html = _cached_upstream_html(url)
        cached_validation_error = (
            _validation_error(cached_html) if cached_html is not None else ""
        )
        if cached_validation_error:
            with _UPSTREAM_CACHE_LOCK:
                _UPSTREAM_HTML_CACHE.pop(url, None)
            logger.info(
                "Discarding invalid cached %s HTML: %s",
                label,
                cached_validation_error,
            )
            cached_html = None
        if cached_html is not None:
            return cached_html
        if _upstream_fetch_recently_failed(url, label=label):
            return None

        try:
            from src.data.ufcstats_http import request_ufcstats

            if _UFCSTATS_SESSION is None:
                _UFCSTATS_SESSION = requests.Session()
            response = request_ufcstats(url, session=_UFCSTATS_SESSION)
            response_html = str(response.text or "")
            validation_error = _validation_error(response_html)
            if validation_error:
                raise RuntimeError(f"{label} returned {validation_error}")
            _store_upstream_html(url, response_html)
            return response_html
        except Exception as exc:
            _record_upstream_fetch_failure(url)
            logger.warning("%s fetch failed: %s", label, exc)
            return None


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


def _scrape_ufc_com_events(*, include_completed: bool = False) -> Optional[list[dict]]:
    """Parse the UFC.com schedule. Returns None when the fetch itself failed, so
    callers can distinguish an unreachable source from a page with no event rows."""
    html = _fetch_upstream_html(
        UFC_COM_EVENTS_URL,
        label="UFC.com events",
        required_selector="article.c-card-event--result",
    )
    if html is None:
        return None

    soup = BeautifulSoup(html, "lxml")
    events: list[dict] = []
    seen_urls: set[str] = set()

    for article in soup.select("article.c-card-event--result"):
        article_text = article.get_text(" ", strip=True)
        is_completed = "Watch Replay" in article_text
        if is_completed and not include_completed:
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
        event = {
            "title": title,
            "url": full_event_url,
            "date": _parse_ufc_com_event_date(raw_date, event_link),
            "location": _extract_ufc_com_event_location_from_article(article),
            "source": "ufc.com",
        }
        if include_completed:
            event["status"] = "completed" if is_completed else "upcoming"
        events.append(event)

    scope = "events" if include_completed else "upcoming events"
    logger.info("Found %d %s via UFC.com", len(events), scope)
    return events


def _scrape_ufc_com_upcoming_events() -> Optional[list[dict]]:
    return _scrape_ufc_com_events(include_completed=False)


def scrape_ufc_com_events(*, include_completed: bool = False) -> list[dict]:
    return _scrape_ufc_com_events(include_completed=include_completed) or []


def _scrape_ufcstats_upcoming_events() -> list[dict]:
    url = UFCSTATS_UPCOMING_URL
    html = _fetch_ufcstats_html(
        url,
        label="UFCStats upcoming events",
        required_selector="tr.b-statistics__table-row",
    )
    if html is None:
        return []

    soup = BeautifulSoup(html, "lxml")
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

    if _looks_like_browser_challenge(html):
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

    fallback_events = _scrape_ufcstats_upcoming_events()
    if events is None:
        log_fn = logger.warning if not fallback_events else logger.info
        log_fn(
            "UFC.com upcoming events fetch failed; %s UFCStats fallback",
            "no events found from" if not fallback_events else "using",
        )
    else:
        log_fn = logger.warning if not fallback_events else logger.info
        log_fn(
            "UFC.com upcoming events fetched but parsed zero event rows; %s UFCStats fallback",
            "no events found from" if not fallback_events else "using",
        )
    return fallback_events


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


def _normalized_web_resource_identity(url: object) -> tuple[str, str]:
    parsed = urlparse(str(url or "").strip())
    host = str(parsed.hostname or "").casefold().removeprefix("www.")
    path = re.sub(r"/+$", "", parsed.path or "/") or "/"
    return host, path


def _normalized_event_date(value: object) -> str:
    """Normalize event dates for stable title/date fallback identity."""
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        if isinstance(parsed, pd.Timestamp) and not pd.isna(parsed):
            return parsed.date().isoformat()
    except MemoryError:
        raise
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def event_identity_key(event: dict) -> str:
    """Return a stable event key, preferring canonical event URL over fallbacks."""
    host, path = _normalized_web_resource_identity(event.get("url"))
    if host and path:
        if host.endswith(".ufc.com"):
            host = "ufc.com"
        return f"url:{host}{path.casefold()}"

    event_id = re.sub(r"\s+", " ", str(event.get("event_id") or "")).strip()
    if event_id:
        return f"id:{event_id.casefold()}"

    title = re.sub(
        r"\s+",
        " ",
        str(event.get("title") or event.get("event") or ""),
    ).strip().casefold()
    return f"title-date:{title}|{_normalized_event_date(event.get('date') or event.get('event_date'))}"


def _event_identity_metadata(event: dict) -> dict:
    """Build persisted event identity metadata from a live event or snapshot."""
    title = str(
        event.get("title")
        or event.get("event_title")
        or event.get("event")
        or ""
    ).strip()
    event_date = str(event.get("date") or event.get("event_date") or "").strip()
    event_url = str(event.get("url") or event.get("event_url") or "").strip()
    event_id = str(event.get("event_id") or "").strip()
    identity_source = {
        "title": title,
        "date": event_date,
        "url": event_url,
        "event_id": event_id,
    }
    return {
        "event_key": str(event.get("event_key") or event_identity_key(identity_source)),
        "event_title": title,
        "event_date": event_date,
        "event_url": event_url,
        "event_id": event_id,
    }


def _ufc_com_event_page_is_valid_empty(
    soup: BeautifulSoup,
    event_url: str,
) -> bool:
    """Accept zero bouts only on a positively identified UFC event page."""
    canonical = soup.select_one('link[rel~="canonical"][href]')
    canonical_url = urljoin(event_url, str(canonical.get("href") or "")) if canonical else ""
    canonical_matches = bool(canonical_url) and (
        _normalized_web_resource_identity(canonical_url)
        == _normalized_web_resource_identity(event_url)
    )
    body = soup.body
    body_classes = {
        str(value or "").strip()
        for value in ((body.get("class") or []) if body is not None else [])
    }
    has_event_structure = bool(
        "page-node-type-event" in body_classes
        and soup.select_one(".node--type-event")
        and soup.select_one(".c-hero .c-hero__headline")
    )
    page_text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).casefold()
    has_explicit_empty_marker = bool(
        re.search(
            r"\b(?:(?:fight card|fights?|bouts?) "
            r"(?:is )?(?:coming soon|to be announced)|"
            r"(?:fights?|bouts?) (?:will be )?(?:announced|added) soon|"
            r"no (?:fights?|bouts?) (?:have been )?(?:announced|scheduled))\b",
            page_text,
        )
    )
    return canonical_matches and (has_event_structure or has_explicit_empty_marker)


def _scrape_ufc_com_event_card(event_url: str) -> list[dict]:
    html = _fetch_upstream_html(
        event_url,
        label="UFC.com event card",
        # Full card URLs are ephemeral. A card can leave the schedule before a
        # later successful fetch, so a durable active incident keyed by this URL
        # would be unrecoverable across restarts. These warnings age normally.
        manage_incident=False,
    )
    if html is None:
        return _EventCardResult(source_healthy=False)

    soup = BeautifulSoup(html, "lxml")
    event_location = _extract_ufc_com_event_location(soup)
    fights = []
    fight_blocks = soup.select(".c-listing-fight")

    for fight in fight_blocks:
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

    if not fights and not fight_blocks and _ufc_com_event_page_is_valid_empty(
        soup,
        event_url,
    ):
        logger.info("UFC.com event card is announced with zero bouts: %s", event_url)
        return _EventCardResult(source_healthy=True)

    if not fights:
        with _UPSTREAM_CACHE_LOCK:
            _UPSTREAM_HTML_CACHE.pop(event_url, None)
        _record_upstream_fetch_failure(event_url)
        logger.warning(
            "UFC.com event card fetched but parsed zero complete fighter rows: %s",
            event_url,
        )
        return _EventCardResult(source_healthy=False)
    logger.info(f"Found {len(fights)} fights on UFC.com card")
    return _EventCardResult(fights, source_healthy=True)


def scrape_event_card(event_url: str) -> list[dict]:
    """
    Scrape the fight card for a specific upcoming event.

    Returns list of fight dicts with:
        fighter_a, fighter_b, weight_class, is_main_event, is_title_bout, num_rounds
    """
    if "ufc.com/event/" in (event_url or ""):
        return _scrape_ufc_com_event_card(event_url)

    if "ufcstats.com" in (event_url or "").lower():
        html = _fetch_ufcstats_html(
            event_url,
            label="UFCStats event card",
            required_selector="tr.b-fight-details__table-row",
        )
    else:
        html = _fetch_upstream_html(
            event_url,
            label="UFCStats event card",
            required_selector="tr.b-fight-details__table-row",
        )
    if html is None:
        return []

    soup = BeautifulSoup(html, "lxml")
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


def _event_fight_contexts(event: dict, card: list[dict]) -> list[dict]:
    contexts: list[dict] = []
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
    return contexts


def _recent_completed_contexts_for_expected_fights(
    expected_fights: object,
    existing_contexts: list[dict],
) -> list[dict]:
    expected_rows = []
    if hasattr(expected_fights, "to_dict"):
        try:
            expected_rows = expected_fights.to_dict("records")
        except Exception:
            expected_rows = []
    elif isinstance(expected_fights, (list, tuple)):
        expected_rows = [row for row in expected_fights if isinstance(row, dict)]
    elif isinstance(expected_fights, dict):
        expected_rows = [expected_fights]
    if not expected_rows:
        return []

    existing_pairs = {
        _tracked_fight_pair_key(context.get("fighter_a", ""), context.get("fighter_b", ""))
        for context in existing_contexts
    }
    expected_pairs = {
        _tracked_fight_pair_key(row.get("fighter_a", ""), row.get("fighter_b", ""))
        for row in expected_rows
    }
    missing_pairs = {pair for pair in expected_pairs - existing_pairs if pair and pair != "|"}
    if not missing_pairs:
        return []

    expected_dates: set = set()
    for row in expected_rows:
        parsed = pd.to_datetime(row.get("commence_time"), errors="coerce", utc=True)
        if pd.notna(parsed):
            expected_dates.add(parsed.date())
    if not expected_dates:
        return []

    all_events = _scrape_ufc_com_events(include_completed=True) or []
    recovered: list[dict] = []
    for event in all_events:
        if str(event.get("status", "") or "").lower() != "completed":
            continue
        event_date = pd.to_datetime(event.get("date"), errors="coerce")
        if pd.isna(event_date) or not any(
            abs((expected_date - event_date.date()).days) <= 1 for expected_date in expected_dates
        ):
            continue

        event_url = event.get("url", "")
        if not event_url:
            continue
        card = scrape_event_card(event_url)
        if not card:
            continue
        card_pairs = {
            _tracked_fight_pair_key(fight.get("fighter_a", ""), fight.get("fighter_b", ""))
            for fight in card
        }
        if not card_pairs.intersection(missing_pairs):
            continue
        recovered.extend(_event_fight_contexts(event, card))
        missing_pairs.difference_update(card_pairs)
        if not missing_pairs:
            break
    return recovered


def clear_upcoming_event_cards_cache() -> None:
    """Clear the process-wide event/card snapshot (primarily for tests)."""
    global _UPCOMING_EVENT_CARDS_CACHE
    with _UPCOMING_EVENT_CARDS_LOCK:
        _UPCOMING_EVENT_CARDS_CACHE = None


def collect_upcoming_event_cards(*, force_refresh: bool = False) -> list[dict]:
    """Fetch all upcoming cards once and share them across monitor/betting loops.

    The lock covers the producer call, providing single-flight behavior when the
    two hosted threads overlap. The default reuse window is just below the
    10-minute betting cadence, preserving that loop's card freshness. Failed or
    structurally uncertain scans are shared only for the short failure cooldown;
    a positively validated announced card with zero bouts is a complete scan.
    """
    global _UPCOMING_EVENT_CARDS_CACHE
    producer_identity = (id(scrape_upcoming_events), id(scrape_event_card))
    now = time.monotonic()
    with _UPCOMING_EVENT_CARDS_LOCK:
        if not force_refresh and _UPCOMING_EVENT_CARDS_CACHE is not None:
            cached_at, cached_identity, cached_rows, cached_complete = (
                _UPCOMING_EVENT_CARDS_CACHE
            )
            reuse_ttl = (
                LIVE_EVENT_CONTEXT_REUSE_TTL_SECONDS
                if cached_complete
                else UPSTREAM_FETCH_FAILURE_TTL_SECONDS
            )
            if (
                cached_identity == producer_identity
                and now - cached_at <= reuse_ttl
            ):
                return copy.deepcopy(cached_rows)

        collected: list[dict] = []
        scan_complete = True
        card_request_count = 0
        events = scrape_upcoming_events()
        if not events:
            scan_complete = False
        for event in events:
            event_url = event.get("url", "")
            if not event_url:
                scan_complete = False
                continue
            if card_request_count:
                time.sleep(UPCOMING_EVENT_CARD_REQUEST_DELAY_SECONDS)
            card_request_count += 1
            card = scrape_event_card(event_url)
            if card:
                collected.append({"event": dict(event), "fights": list(card)})
            elif not bool(getattr(card, "source_healthy", False)):
                scan_complete = False

        if collected and not scan_complete:
            logger.info(
                "Upcoming event-card scan was partial (%d/%d cards); sharing it for only %.0fs",
                len(collected),
                len(events),
                UPSTREAM_FETCH_FAILURE_TTL_SECONDS,
            )

        _UPCOMING_EVENT_CARDS_CACHE = (
            time.monotonic(),
            producer_identity,
            copy.deepcopy(collected),
            scan_complete,
        )
        return copy.deepcopy(collected)


def collect_upcoming_fight_contexts(expected_fights: object = None) -> list[dict]:
    """Return live fight metadata enriched with Odds API event identity.

    UFC.com can mark a late US card completed while some bookmaker fights still
    have future UTC commence times. In that boundary window, recover the matching
    completed card so callers retain its official local event date and context.
    """
    contexts: list[dict] = []

    for event_card in collect_upcoming_event_cards():
        contexts.extend(
            _event_fight_contexts(event_card.get("event", {}), event_card.get("fights", []))
        )

    contexts.extend(_recent_completed_contexts_for_expected_fights(expected_fights, contexts))

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

def _card_fighter_key(fight: dict, field: str) -> str:
    return normalize_cross_source_name(str(fight.get(field) or ""))


def _nonnegative_days(value: object) -> int | None:
    """Coerce a persisted countdown without allowing negative event-day text."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return None


def detect_short_notice(
    current_card: list[dict],
    previous_card: list[dict],
    days_to_event: int,
    *,
    emit_warning: bool = True,
    event_identity: dict | None = None,
) -> list[dict]:
    """
    Detect short-notice replacements by comparing current card to previous snapshot.

    A short-notice replacement is when a fighter is swapped in within ~2 weeks of the event.
    Short-notice fighters historically win at ~38% rate.

    ``days_to_event`` is the remaining time when our card source first exposes the
    replacement. It is not necessarily the fighter's actual amount of notice,
    because the source may publish a previously announced change later.

    Returns list of replacement dicts.
    """
    replacements = []
    raw_days_to_event = _nonnegative_days(days_to_event)
    if raw_days_to_event is None:
        return replacements
    if raw_days_to_event > 14:
        return replacements
    detected_days = raw_days_to_event

    current_fighters = {}
    for fight in current_card:
        for field in ("fighter_a", "fighter_b"):
            name = str(fight.get(field) or "").strip()
            key = _card_fighter_key(fight, field)
            if key:
                current_fighters.setdefault(key, name)

    previous_fighters = {}
    for fight in previous_card:
        for field in ("fighter_a", "fighter_b"):
            name = str(fight.get(field) or "").strip()
            key = _card_fighter_key(fight, field)
            if key:
                previous_fighters.setdefault(key, name)

    current_keys = set(current_fighters)
    previous_keys = set(previous_fighters)

    # New fighters not in previous card.
    new_fighters = current_keys - previous_keys
    removed_fighters = previous_keys - current_keys

    for fighter_key in sorted(new_fighters):
        replaced_key = None
        for fight in current_card:
            fa = _card_fighter_key(fight, "fighter_a")
            fb = _card_fighter_key(fight, "fighter_b")
            if fighter_key == fa:
                opponent = fb
            elif fighter_key == fb:
                opponent = fa
            else:
                continue

            # A short-notice replacement requires a one-for-one opponent match:
            # the opponent stayed on the card and their previous opponent left.
            for prev_fight in previous_card:
                pa = _card_fighter_key(prev_fight, "fighter_a")
                pb = _card_fighter_key(prev_fight, "fighter_b")
                if opponent == pa and pb in removed_fighters:
                    replaced_key = pb
                    break
                if opponent == pb and pa in removed_fighters:
                    replaced_key = pa
                    break
            if replaced_key:
                break

        if not replaced_key:
            logger.info(
                "Near-event card change for %s, but no one-for-one replacement was found",
                current_fighters[fighter_key],
            )
            continue

        fighter = current_fighters[fighter_key]
        replaced = previous_fighters[replaced_key]
        replacement = {
            "new_fighter": fighter,
            "replaced_fighter": replaced,
            "days_until_event_at_detection": detected_days,
            "is_short_notice": True,
        }
        if event_identity:
            replacement.update(_event_identity_metadata(event_identity))
        replacements.append(replacement)
        if emit_warning:
            logger.warning(
                f"SHORT NOTICE: {fighter} replacing {replaced}; "
                f"late-detected with {detected_days} days until event"
            )

    return replacements


# ---------------------------------------------------------------------------
# Snapshot management
# ---------------------------------------------------------------------------

def _snapshot_event_target(
    event_title: str,
    *,
    event_date: str = "",
    event_url: str = "",
    event_id: str = "",
    event_key: str = "",
) -> dict:
    return _event_identity_metadata(
        {
            "title": event_title,
            "date": event_date,
            "url": event_url,
            "event_id": event_id,
            "event_key": event_key,
        }
    )


def _snapshot_matches_event(payload: dict, target: dict) -> bool:
    stored = _event_identity_metadata(payload)
    stored_key = stored["event_key"]
    target_key = target["event_key"]
    if stored_key == target_key:
        return True

    stored_has_stable_key = stored_key.startswith(("url:", "id:"))
    target_has_stable_key = target_key.startswith(("url:", "id:"))
    if stored_has_stable_key and target_has_stable_key:
        return False

    stored_title = re.sub(r"\s+", " ", stored["event_title"]).strip().casefold()
    target_title = re.sub(r"\s+", " ", target["event_title"]).strip().casefold()
    if not stored_title or stored_title != target_title:
        return False

    stored_date = _normalized_event_date(stored["event_date"])
    target_date = _normalized_event_date(target["event_date"])
    return not (stored_date and target_date and stored_date != target_date)


def _latest_card_snapshot_payload(
    event_title: str,
    *,
    event_date: str = "",
    event_url: str = "",
    event_id: str = "",
    event_key: str = "",
    snapshot_dir: Path | None = None,
) -> tuple[Path, dict] | None:
    """Return the newest readable snapshot for exactly this event identity."""
    root = SNAPSHOTS_DIR if snapshot_dir is None else Path(snapshot_dir)
    target = _snapshot_event_target(
        event_title,
        event_date=event_date,
        event_url=event_url,
        event_id=event_id,
        event_key=event_key,
    )
    matches = []
    for path in root.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read card snapshot %s: %s", path, exc)
            continue
        if not isinstance(data, dict):
            continue
        if not _snapshot_matches_event(data, target):
            continue
        matches.append((path, data))
    if not matches:
        return None
    return max(matches, key=lambda item: _card_snapshot_sort_key(*item))


def _card_snapshot_history(
    event_title: str,
    *,
    event_date: str = "",
    event_url: str = "",
    event_id: str = "",
    event_key: str = "",
    snapshot_dir: Path | None = None,
) -> list[tuple[Path, dict]]:
    """Return readable snapshots for exactly this event identity, oldest first."""
    root = SNAPSHOTS_DIR if snapshot_dir is None else Path(snapshot_dir)
    target = _snapshot_event_target(
        event_title,
        event_date=event_date,
        event_url=event_url,
        event_id=event_id,
        event_key=event_key,
    )
    snapshots = []
    for path in root.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read card snapshot %s: %s", path, exc)
            continue
        if not isinstance(data, dict):
            continue
        if not _snapshot_matches_event(data, target):
            continue
        snapshots.append((path, data))
    return sorted(snapshots, key=lambda item: _card_snapshot_sort_key(*item))


def _card_snapshot_sort_key(path: Path, payload: dict) -> tuple[float, str]:
    """Order snapshots by capture metadata, with filesystem/path fallbacks."""
    captured = _card_snapshot_epoch(payload.get("timestamp"))
    if captured is None:
        try:
            captured = float(path.stat().st_mtime)
        except OSError:
            captured = 0.0
    return captured, path.name


def _merge_active_short_notice_replacements(
    current_card: list[dict],
    *replacement_groups: object,
    event_identity: dict | None = None,
) -> list[dict]:
    """Merge and retain only replacement fighters still present on the card."""
    current_fighters: dict[str, str] = {}
    for fight in current_card:
        for field in ("fighter_a", "fighter_b"):
            key = _card_fighter_key(fight, field)
            display_name = str(fight.get(field) or "").strip()
            if key:
                current_fighters.setdefault(key, display_name)

    event_metadata = (
        _event_identity_metadata(event_identity)
        if isinstance(event_identity, dict)
        else None
    )
    merged: dict[tuple[str, str], dict] = {}
    for group in replacement_groups:
        if not isinstance(group, list):
            continue
        for raw_replacement in group:
            if not isinstance(raw_replacement, dict):
                continue
            new_fighter = str(raw_replacement.get("new_fighter") or "").strip()
            replaced_fighter = str(
                raw_replacement.get("replaced_fighter") or ""
            ).strip()
            new_key = normalize_cross_source_name(new_fighter)
            replaced_key = normalize_cross_source_name(replaced_fighter)
            if not new_key or not replaced_key or new_key not in current_fighters:
                continue

            replacement = copy.deepcopy(raw_replacement)
            raw_countdown = replacement.get(
                "days_until_event_at_detection",
                replacement.get("days_notice"),
            )
            replacement.pop("days_notice", None)
            if raw_countdown is not None:
                countdown = _nonnegative_days(raw_countdown)
                if countdown is None:
                    continue
                replacement["days_until_event_at_detection"] = countdown

            # Persist normalized keys for comparisons, but refresh display names
            # from the current card so source accent/casing changes do not linger.
            replacement["new_fighter"] = current_fighters[new_key]
            replacement["replaced_fighter"] = current_fighters.get(
                replaced_key,
                replaced_fighter,
            )
            replacement["new_fighter_key"] = new_key
            replacement["replaced_fighter_key"] = replaced_key
            replacement["is_short_notice"] = True
            if event_metadata is not None:
                replacement.update(event_metadata)
            merged[(new_key, replaced_key)] = replacement
    return list(merged.values())


def _persisted_short_notice_state_is_valid(
    value: object,
    *,
    event_key: str = "",
) -> bool:
    if not isinstance(value, list):
        return False
    for replacement in value:
        if not isinstance(replacement, dict):
            return False
        if not normalize_cross_source_name(
            str(replacement.get("new_fighter") or "")
        ):
            return False
        if not normalize_cross_source_name(
            str(replacement.get("replaced_fighter") or "")
        ):
            return False
        persisted_event_key = str(replacement.get("event_key") or "")
        if persisted_event_key and event_key and persisted_event_key != event_key:
            return False
        for countdown_key in ("days_until_event_at_detection", "days_notice"):
            if (
                countdown_key in replacement
                and _nonnegative_days(replacement.get(countdown_key)) is None
            ):
                return False
    return True


def _snapshot_days_to_event(payload: dict) -> int | None:
    """Return the event countdown at snapshot capture time when metadata permits."""
    try:
        event_date = pd.to_datetime(payload.get("event_date"), utc=True, errors="coerce")
        captured_at = pd.to_datetime(payload.get("timestamp"), utc=True, errors="coerce")
        if (
            not isinstance(event_date, pd.Timestamp)
            or not isinstance(captured_at, pd.Timestamp)
            or pd.isna(event_date)
            or pd.isna(captured_at)
        ):
            return None
        return max(0, int((event_date.date() - captured_at.date()).days))
    except MemoryError:
        raise
    except Exception:
        return None


def _reconstruct_short_notice_replacements(
    event_identity: dict,
) -> tuple[list[dict], bool]:
    """
    Rebuild active replacement state from legacy snapshots lacking persisted state.

    This is intentionally silent: a historical detection should restore the
    current signal after an upgrade/restart, not emit a fresh incident warning.
    """
    event_metadata = _event_identity_metadata(event_identity)
    history = _card_snapshot_history(
        event_metadata["event_title"],
        event_date=event_metadata["event_date"],
        event_url=event_metadata["event_url"],
        event_id=event_metadata["event_id"],
        event_key=event_metadata["event_key"],
    )
    if not history:
        return [], False

    first_payload = history[0][1]
    first_card = first_payload.get("fights", [])
    first_card_is_valid = isinstance(first_card, list)
    previous_card = first_card if first_card_is_valid else None
    first_state = first_payload.get("short_notice_replacements")
    first_snapshot_event_key = _event_identity_metadata(first_payload)["event_key"]
    first_state_valid = (
        "short_notice_replacements" in first_payload
        and _persisted_short_notice_state_is_valid(
            first_state,
            event_key=first_snapshot_event_key,
        )
    )
    if "short_notice_replacements" in first_payload and not first_state_valid:
        logger.warning(
            "Quarantining malformed short-notice state in snapshot for %s",
            event_metadata["event_title"],
        )
    active = (
        _merge_active_short_notice_replacements(
            first_card,
            first_state,
            event_identity=event_metadata,
        )
        if first_state_valid and first_card_is_valid
        else copy.deepcopy(first_state)
        if first_state_valid
        else []
    )
    trustworthy = first_state_valid
    replay_gap = not first_state_valid and not first_card_is_valid

    for _, payload in history[1:]:
        current_card = payload.get("fights", [])
        persisted_state = payload.get("short_notice_replacements")
        snapshot_event_key = _event_identity_metadata(payload)["event_key"]
        persisted_state_valid = (
            "short_notice_replacements" in payload
            and _persisted_short_notice_state_is_valid(
                persisted_state,
                event_key=snapshot_event_key,
            )
        )
        if persisted_state_valid:
            # Persisted state is authoritative for snapshots written by the
            # carry-forward implementation.
            active = (
                _merge_active_short_notice_replacements(
                    current_card,
                    persisted_state,
                    event_identity=event_metadata,
                )
                if isinstance(current_card, list)
                else copy.deepcopy(persisted_state)
            )
            trustworthy = True
            replay_gap = False
        else:
            if "short_notice_replacements" in payload:
                logger.warning(
                    "Quarantining malformed short-notice state in snapshot for %s",
                    event_metadata["event_title"],
                )
            if not isinstance(current_card, list):
                trustworthy = False
                replay_gap = True
                previous_card = None
                continue
            active = _merge_active_short_notice_replacements(
                current_card,
                active,
                event_identity=event_metadata,
            )
            days_to_event = _snapshot_days_to_event(payload)
            if days_to_event is not None and previous_card:
                detected = detect_short_notice(
                    current_card,
                    previous_card,
                    days_to_event,
                    emit_warning=False,
                    event_identity=event_metadata,
                )
                active = _merge_active_short_notice_replacements(
                    current_card,
                    active,
                    detected,
                    event_identity=event_metadata,
                )
                if not replay_gap:
                    trustworthy = True
            else:
                trustworthy = False
                replay_gap = True
        previous_card = current_card if isinstance(current_card, list) else None
    return active, trustworthy


def _card_snapshot_epoch(value: object) -> float | None:
    """Parse a persisted date string without letting malformed metadata escape."""
    try:
        if not isinstance(value, str) or not value.strip():
            return None
        parsed = pd.to_datetime(value.strip(), utc=True, errors="coerce")
        if not isinstance(parsed, pd.Timestamp) or pd.isna(parsed):
            return None
        return float(parsed.timestamp())
    except MemoryError:
        raise
    except Exception:
        return None


def _card_snapshot_metadata_supplied(value: object) -> bool:
    """Return whether a persisted metadata field contains a non-empty value."""
    return value is not None and (
        not isinstance(value, str) or bool(value.strip())
    )


def _card_snapshot_retention_entry(
    path: Path,
) -> tuple[str, dict, tuple[str, ...]] | None:
    """Build one retention entry completely before shared scan state changes."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None

    raw_event = payload.get("event")
    if not isinstance(raw_event, str) or not raw_event.strip():
        return None
    event = _event_identity_metadata(payload)["event_key"]

    modified = path.stat().st_mtime
    raw_timestamp = payload.get("timestamp")
    raw_event_date = payload.get("event_date")
    captured = _card_snapshot_epoch(raw_timestamp)
    event_epoch = _card_snapshot_epoch(raw_event_date)

    fallback_fields = []
    if captured is None and _card_snapshot_metadata_supplied(raw_timestamp):
        fallback_fields.append("timestamp")
    if event_epoch is None and _card_snapshot_metadata_supplied(raw_event_date):
        fallback_fields.append("event_date")

    return (
        event,
        {
            "path": path,
            "captured": modified if captured is None else captured,
            "event_epoch": event_epoch,
        },
        tuple(fallback_fields),
    )


def prune_card_snapshots(
    *,
    snapshot_dir: Path | None = None,
    now: float | None = None,
    force: bool = False,
) -> int:
    """Prune card history without deleting the sole current event snapshot."""
    global _last_card_snapshot_prune_monotonic

    root = SNAPSHOTS_DIR if snapshot_dir is None else Path(snapshot_dir)
    if not root.exists():
        return 0

    monotonic = time.monotonic()
    with _card_snapshot_prune_lock:
        if (
            not force
            and _last_card_snapshot_prune_monotonic
            and monotonic - _last_card_snapshot_prune_monotonic
            < CARD_SNAPSHOT_PRUNE_INTERVAL_SECONDS
        ):
            return 0
        _last_card_snapshot_prune_monotonic = monotonic

        current = time.time() if now is None else float(now)
        entries_by_event: dict[str, list[dict]] = {}
        skipped_files = 0
        metadata_fallback_files = 0
        for path in root.glob("*.json"):
            try:
                result = _card_snapshot_retention_entry(path)
            except MemoryError:
                raise
            except Exception as exc:
                skipped_files += 1
                logger.warning(
                    "Skipping card snapshot %s during retention scan (%s): %s",
                    path,
                    type(exc).__name__,
                    exc,
                )
                continue
            if result is None:
                skipped_files += 1
                logger.warning(
                    "Skipping unclassifiable card snapshot %s during retention scan",
                    path,
                )
                continue

            event, entry, fallback_fields = result
            if fallback_fields:
                metadata_fallback_files += 1
                logger.warning(
                    "Card snapshot retention using fallback metadata for %s: %s",
                    path,
                    ", ".join(fallback_fields),
                )

            # Commit only a fully constructed entry so a per-file failure can
            # never leave an empty or partially populated event bucket behind.
            entries_by_event.setdefault(event, []).append(entry)

        remove: set[Path] = set()
        protected: set[Path] = set()
        all_entries: list[dict] = []
        for event_entries in entries_by_event.values():
            event_entries.sort(key=lambda item: item["captured"], reverse=True)
            all_entries.extend(event_entries)
            newest = event_entries[0]
            event_epoch = next(
                (
                    item["event_epoch"]
                    for item in event_entries
                    if item["event_epoch"] is not None
                ),
                None,
            )

            event_expired = bool(
                CARD_SNAPSHOT_PAST_EVENT_RETENTION_DAYS > 0
                and event_epoch is not None
                and current
                > event_epoch + CARD_SNAPSHOT_PAST_EVENT_RETENTION_DAYS * 24 * 60 * 60
            )
            if event_expired:
                remove.update(item["path"] for item in event_entries)
                continue

            if event_epoch is None and CARD_SNAPSHOT_UNKNOWN_DATE_RETENTION_DAYS > 0:
                unknown_cutoff = (
                    current - CARD_SNAPSHOT_UNKNOWN_DATE_RETENTION_DAYS * 24 * 60 * 60
                )
                for item in event_entries:
                    if item["captured"] < unknown_cutoff:
                        remove.add(item["path"])

            # Keep the newest usable snapshot for every unexpired event even
            # when applying per-event and global safety caps.
            if newest["path"] not in remove:
                protected.add(newest["path"])

            survivors = [item for item in event_entries if item["path"] not in remove]
            for item in survivors[CARD_SNAPSHOT_MAX_PER_EVENT:]:
                if item["path"] not in protected:
                    remove.add(item["path"])

        survivors = [item for item in all_entries if item["path"] not in remove]
        excess = max(0, len(survivors) - CARD_SNAPSHOT_MAX_FILES)
        if excess:
            removable = sorted(
                (item for item in survivors if item["path"] not in protected),
                key=lambda item: item["captured"],
            )
            remove.update(item["path"] for item in removable[:excess])

        removed = 0
        for path in remove:
            try:
                path.unlink()
                removed += 1
            except FileNotFoundError:
                continue
            except OSError:
                continue
        if removed:
            logger.info("Pruned %d old card snapshot files from %s", removed, root)
        if skipped_files or metadata_fallback_files:
            logger.warning(
                "Card snapshot retention summary for %s: "
                "removed=%d, skipped=%d, metadata_fallbacks=%d",
                root,
                removed,
                skipped_files,
                metadata_fallback_files,
            )
        return removed


def save_card_snapshot(
    event_title: str,
    card: list[dict],
    event_date: str = "",
    *,
    event_url: str = "",
    event_id: str = "",
    event_key: str = "",
    short_notice_replacements: list[dict] | None = None,
) -> Path:
    """Save changed card/signal state and opportunistically prune old history."""
    event_metadata = _snapshot_event_target(
        event_title,
        event_date=event_date,
        event_url=event_url,
        event_id=event_id,
        event_key=event_key,
    )
    stable_identity_supplied = bool(event_url or event_id or event_key)
    replacement_state_supplied = short_notice_replacements is not None
    replacement_state = _merge_active_short_notice_replacements(
        card,
        short_notice_replacements or [],
        event_identity=event_metadata,
    )
    latest = _latest_card_snapshot_payload(
        event_title,
        event_date=event_date,
        event_url=event_url,
        event_id=event_id,
        event_key=event_metadata["event_key"],
    )
    if latest is not None:
        latest_path, latest_payload = latest
        if (
            str(latest_payload.get("event", "") or "") == event_title
            and str(latest_payload.get("event_date", "") or "")
            == str(event_date or "")
            and (
                not stable_identity_supplied
                or str(latest_payload.get("event_key") or "")
                == event_metadata["event_key"]
            )
            and latest_payload.get("fights", []) == card
            and (
                not replacement_state_supplied
                or "short_notice_replacements" in latest_payload
            )
            and latest_payload.get("short_notice_replacements", [])
            == replacement_state
        ):
            logger.debug("Card snapshot unchanged; reusing %s", latest_path)
            try:
                prune_card_snapshots()
            except MemoryError:
                raise
            except Exception as exc:
                logger.warning("Card snapshot retention skipped: %s", exc)
            return latest_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = re.sub(r"[^\w\-]", "_", event_title)[:50]
    filename = f"{safe_title}_{timestamp}.json"
    path = SNAPSHOTS_DIR / filename
    if path.exists():
        # Two real card changes can be observed within one second (especially
        # in tests or manual refreshes); never overwrite the earlier state.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = SNAPSHOTS_DIR / f"{safe_title}_{timestamp}.json"

    snapshot = {
        "event": event_title,
        "event_date": event_date,
        "event_url": event_url,
        "event_id": event_id,
        "event_key": event_metadata["event_key"],
        "timestamp": datetime.now().isoformat(),
        "fights": card,
        "short_notice_replacements": replacement_state,
    }
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomically(snapshot, path)
    logger.info("Saved card snapshot: %s", path)
    try:
        prune_card_snapshots()
    except MemoryError:
        raise
    except Exception as exc:
        logger.warning("Card snapshot retention skipped: %s", exc)
    return path


def load_latest_snapshot(
    event_title: str,
    *,
    event_date: str = "",
    event_url: str = "",
    event_id: str = "",
    event_key: str = "",
) -> Optional[list[dict]]:
    """Load the most recent readable snapshot for an event."""
    latest = _latest_card_snapshot_payload(
        event_title,
        event_date=event_date,
        event_url=event_url,
        event_id=event_id,
        event_key=event_key,
    )
    if latest is None:
        return None
    return latest[1].get("fights", [])


def _update_short_notice_event_state_unlocked(
    event: dict,
    current_card: list[dict],
    days_to_event: int,
) -> list[dict]:
    """Derive and persist one event's active short-notice state."""
    event_metadata = _event_identity_metadata(event)
    latest_snapshot = _latest_card_snapshot_payload(
        event_metadata["event_title"],
        event_date=event_metadata["event_date"],
        event_url=event_metadata["event_url"],
        event_id=event_metadata["event_id"],
        event_key=event_metadata["event_key"],
    )
    event_replacements: list[dict] = []
    skip_untrustworthy_save = False

    if latest_snapshot is not None:
        _, latest_payload = latest_snapshot
        previous_card = latest_payload.get("fights", [])
        if not isinstance(previous_card, list):
            previous_card = []

        persisted_replacements = latest_payload.get("short_notice_replacements")
        state_is_present = "short_notice_replacements" in latest_payload
        state_is_valid = (
            state_is_present
            and _persisted_short_notice_state_is_valid(
                persisted_replacements,
                event_key=_event_identity_metadata(latest_payload)["event_key"],
            )
        )
        if state_is_valid:
            event_replacements = _merge_active_short_notice_replacements(
                current_card,
                persisted_replacements,
                event_identity=event_metadata,
            )
        else:
            reconstructed, trustworthy = _reconstruct_short_notice_replacements(
                event_metadata
            )
            event_replacements = _merge_active_short_notice_replacements(
                current_card,
                reconstructed,
                event_identity=event_metadata,
            )
            skip_untrustworthy_save = state_is_present and not trustworthy

        if previous_card:
            detected_replacements = detect_short_notice(
                current_card,
                previous_card,
                days_to_event,
                event_identity=event_metadata,
            )
            if detected_replacements:
                skip_untrustworthy_save = False
            event_replacements = _merge_active_short_notice_replacements(
                current_card,
                event_replacements,
                detected_replacements,
                event_identity=event_metadata,
            )

    if not skip_untrustworthy_save:
        save_card_snapshot(
            event_metadata["event_title"],
            current_card,
            event_date=event_metadata["event_date"],
            event_url=event_metadata["event_url"],
            event_id=event_metadata["event_id"],
            event_key=event_metadata["event_key"],
            short_notice_replacements=event_replacements,
        )
    return event_replacements


def _update_short_notice_event_state(
    event: dict,
    current_card: list[dict],
    days_to_event: int,
) -> list[dict]:
    """Serialize snapshot read/detect/write for one in-process monitoring pass."""
    with _card_snapshot_state_lock:
        return _update_short_notice_event_state_unlocked(
            event,
            current_card,
            days_to_event,
        )


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

    event_cards = collect_upcoming_event_cards()

    for event_card in event_cards:
        event = event_card.get("event", {})
        current_card = event_card.get("fights", [])
        event_url = event.get("url", "")
        event_title = event.get("title", "")
        event_date_str = event.get("date", "")

        if not event_url:
            continue

        # Parse event date
        try:
            event_date = pd.to_datetime(event_date_str)
            days_to_event = max(
                0,
                int((event_date.date() - pd.Timestamp.now().date()).days),
            )
        except Exception:
            days_to_event = 30  # Default if we can't parse

        logger.info(f"\nEvent: {event_title} ({days_to_event} days away)")

        if not current_card:
            continue

        event_identity = _event_identity_metadata(event)
        event_info = {
            **event,
            "event_key": event_identity["event_key"],
            "days_to_event": days_to_event,
            "num_fights": len(current_card),
            "fights": current_card,
        }
        signals["events"].append(event_info)

        # Compare to the previous snapshot, then carry confirmed replacements
        # forward while the replacement fighter remains on this event's card.
        event_replacements = _update_short_notice_event_state(
            event,
            current_card,
            days_to_event,
        )
        signals["short_notice_replacements"].extend(event_replacements)

        # Check weigh-ins if event is within 2 days
        if days_to_event <= 2:
            weighin_results = scrape_weighin_results(event_title)
            if weighin_results:
                missed = check_missed_weight(weighin_results, current_card)
                signals["missed_weights"].extend(missed)

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
        method_snapshot_summary = {
            "status": method_snapshot.get("status"),
            "record_count": method_snapshot.get("record_count", 0),
            "snapshot_time": method_snapshot.get("snapshot_time"),
            "snapshot_path": method_snapshot.get("snapshot_path"),
            "availability_expected": bool(method_snapshot.get("availability_expected")),
            "expected_fight_count": method_snapshot.get("expected_fight_count", 0),
            "expected_window_hours": method_snapshot.get("expected_window_hours", 48),
        }
        for key in (
            "coverage_status",
            "tracked_fight_count",
            "covered_fight_count",
            "missing_fight_count",
            "expected_coverage_status",
            "expected_covered_fight_count",
            "expected_missing_fight_count",
            "expected_event_count",
            "expected_events",
            "missing_expected_fights",
        ):
            if key in method_snapshot:
                method_snapshot_summary[key] = method_snapshot.get(key)
        latest_usable = method_snapshot.get("latest_usable_snapshot")
        if isinstance(latest_usable, dict) and latest_usable:
            method_snapshot_summary["latest_usable_snapshot"] = latest_usable
        signals["method_odds_snapshot"] = method_snapshot_summary
    except Exception as e:
        logger.error(
            "Method-odds snapshot collection failed: %s",
            e,
            extra={"alert_incident_key": _METHOD_ODDS_ALERT_INCIDENT_KEY},
        )

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
        log_level, log_message = _method_odds_snapshot_log_message(signals["method_odds_snapshot"])
        if log_level >= logging.WARNING:
            logger.log(
                log_level,
                log_message,
                extra={"alert_incident_key": _METHOD_ODDS_ALERT_INCIDENT_KEY},
            )
        else:
            try:
                tracked_fight_count = int(
                    signals["method_odds_snapshot"].get("tracked_fight_count", 0) or 0
                )
            except (TypeError, ValueError):
                tracked_fight_count = 0
            recovery_context = (
                {
                    "alert_recovered_incident_keys": [
                        _METHOD_ODDS_ALERT_INCIDENT_KEY
                    ]
                }
                if tracked_fight_count > 0
                else {}
            )
            logger.log(
                log_level,
                log_message,
                extra=recovery_context,
            )

    # Save full signals
    signals_path = LOGS_DIR / "latest_signals.json"
    serializable = {
        k: v if not isinstance(v, pd.DataFrame) else v.to_dict()
        for k, v in signals.items()
    }
    signals_path.write_text(json.dumps(serializable, indent=2, default=str))

    return signals
