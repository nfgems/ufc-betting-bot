"""
Snapshot-backed method-of-victory odds helpers.

Collection may use live sources, but inference reads only the latest durable
snapshot so feature generation is reproducible.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from bs4 import BeautifulSoup

from src.config import ODDS_API_KEY, RAW_DATA_DIR
from src.data.name_utils import name_appears_in_text, normalize_person_name, same_person_name

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
REQUEST_DELAY = 1.5
METHOD_ODDS_CACHE_TTL_SECONDS = 300
METHOD_ODDS_SNAPSHOT_MAX_AGE = timedelta(days=2)

METHOD_ODDS_SNAPSHOT_DIR = RAW_DATA_DIR / "method_odds"
METHOD_ODDS_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_SCHEMA_VERSION = 1

_METHOD_RE = {
    "ko": re.compile(r"\b(?:ko|tko|knockout)\b", re.IGNORECASE),
    "sub": re.compile(r"\b(?:sub|submission)\b", re.IGNORECASE),
    "dec": re.compile(r"\b(?:dec|decision|points)\b", re.IGNORECASE),
}
_METHOD_FEATURE_COLUMNS = [
    "a_ko_odds_prob",
    "a_sub_odds_prob",
    "a_dec_odds_prob",
    "b_ko_odds_prob",
    "b_sub_odds_prob",
    "b_dec_odds_prob",
]
_method_odds_cache: dict[tuple[str, str, str, str, str], tuple[float, dict]] = {}


def _now_iso() -> str:
    return datetime.now().isoformat()


def _american_to_implied_prob(odds: float) -> float:
    """
    Convert American odds to implied probability.

    Invalid or zero odds return NaN.
    """
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def _normalize_name(name: str) -> str:
    """Normalize fighter names for matching."""
    return normalize_person_name(name)


def _name_tokens(name: str) -> list[str]:
    return [token for token in _normalize_name(name).split() if token]


def _last_name_token(name: str) -> str:
    tokens = _name_tokens(name)
    return tokens[-1] if tokens else ""


def _names_match(query_name: str, candidate_name: str) -> bool:
    """Exact full-name match first, then controlled token fallback."""
    if same_person_name(query_name, candidate_name):
        return True

    # Do not let a single surname token match a longer fighter name like
    # "Silva" -> "Bruno Silva". The text fallback is only safe for full-name
    # queries that may appear inside a larger market label.
    if len(_name_tokens(query_name)) < 2:
        return False

    return name_appears_in_text(query_name, candidate_name)


def _match_fight(home_name: str, away_name: str, fighter_a: str, fighter_b: str) -> tuple[bool, Optional[bool]]:
    """Return whether an event matches and whether home corresponds to fighter A."""
    a_home = same_person_name(fighter_a, home_name)
    a_away = same_person_name(fighter_a, away_name)
    b_home = same_person_name(fighter_b, home_name)
    b_away = same_person_name(fighter_b, away_name)

    if a_home and b_away and not b_home and not a_away:
        return True, True
    if a_away and b_home and not a_home and not b_away:
        return True, False
    if a_home and b_away:
        return True, True
    if a_away and b_home:
        return True, False
    return False, None


def _identify_fighter_side(text: str, fighter_a: str, fighter_b: str) -> Optional[str]:
    """Identify which fighter a text snippet refers to."""
    a_match = _names_match(fighter_a, text)
    b_match = _names_match(fighter_b, text)
    if a_match and not b_match:
        return "a"
    if b_match and not a_match:
        return "b"

    normalized_text = _normalize_name(text)
    a_last = _last_name_token(fighter_a)
    b_last = _last_name_token(fighter_b)
    if a_last and b_last and a_last != b_last:
        a_last_match = re.search(rf"\b{re.escape(a_last)}\b", normalized_text) is not None
        b_last_match = re.search(rf"\b{re.escape(b_last)}\b", normalized_text) is not None
        if a_last_match and not b_last_match:
            return "a"
        if b_last_match and not a_last_match:
            return "b"
    return None


_BFO_WINS_BY_RE = re.compile(
    r"^(?!not\s)(.+?)\s+(?:wins\s+)?by\s+(tko/?ko|ko/?tko|knockout|submission|decision)\s*$",
    re.IGNORECASE,
)


def _classify_method(text: str, *, strict: bool = False) -> Optional[str]:
    """Classify a prop row's method.

    When *strict* is True (used for BFO event pages), only the exact
    ``X wins by Y`` label format is accepted — negated markets, round-specific
    rows, and subset markets are all rejected.

    When *strict* is False (default, used for Odds API outcomes), a broad
    keyword fallback is used.
    """
    th_text = text.strip()
    bfo_match = _BFO_WINS_BY_RE.match(th_text)
    if bfo_match:
        method_str = bfo_match.group(2).lower()
        if "ko" in method_str or "knockout" in method_str:
            return "ko"
        if "sub" in method_str:
            return "sub"
        if "dec" in method_str:
            return "dec"

    if strict:
        return None

    # Broad fallback for non-BFO sources (odds API records, etc.)
    normalized = _normalize_name(text)
    for method, pattern in _METHOD_RE.items():
        if pattern.search(normalized):
            return method
    return None


def _avg_or_nan(probs: list[float]) -> float:
    valid = [prob for prob in probs if not np.isnan(prob)]
    return float(np.mean(valid)) if valid else np.nan


def _nan_result() -> dict:
    return {column: np.nan for column in _METHOD_FEATURE_COLUMNS}


def _cache_key(
    fighter_a: str,
    fighter_b: str,
    event_id: Optional[str] = None,
    commence_time: Optional[str] = None,
    as_of_date: Optional[str] = None,
) -> tuple[str, str, str, str, str]:
    return (
        _normalize_name(fighter_a),
        _normalize_name(fighter_b),
        str(event_id or ""),
        str(commence_time or ""),
        str(as_of_date or ""),
    )


def _get_cached_method_odds(
    fighter_a: str,
    fighter_b: str,
    *,
    event_id: Optional[str] = None,
    commence_time: Optional[str] = None,
    as_of_date: Optional[str] = None,
) -> Optional[dict]:
    key = _cache_key(
        fighter_a,
        fighter_b,
        event_id=event_id,
        commence_time=commence_time,
        as_of_date=as_of_date,
    )
    cached = _method_odds_cache.get(key)
    if cached is None:
        return None
    cached_at, cached_value = cached
    if time.monotonic() - cached_at > METHOD_ODDS_CACHE_TTL_SECONDS:
        _method_odds_cache.pop(key, None)
        return None
    return dict(cached_value)


def _set_cached_method_odds(
    fighter_a: str,
    fighter_b: str,
    value: dict,
    *,
    event_id: Optional[str] = None,
    commence_time: Optional[str] = None,
    as_of_date: Optional[str] = None,
) -> None:
    _method_odds_cache[
        _cache_key(
            fighter_a,
            fighter_b,
            event_id=event_id,
            commence_time=commence_time,
            as_of_date=as_of_date,
        )
    ] = (
        time.monotonic(),
        dict(value),
    )


def _collect_method_probs() -> dict[str, list[float]]:
    return {
        "a_ko": [],
        "a_sub": [],
        "a_dec": [],
        "b_ko": [],
        "b_sub": [],
        "b_dec": [],
    }


def _finalize_method_probs(prob_lists: dict[str, list[float]]) -> Optional[dict]:
    if not any(prob_lists.values()):
        return None
    return {
        "a_ko_odds_prob": _avg_or_nan(prob_lists["a_ko"]),
        "a_sub_odds_prob": _avg_or_nan(prob_lists["a_sub"]),
        "a_dec_odds_prob": _avg_or_nan(prob_lists["a_dec"]),
        "b_ko_odds_prob": _avg_or_nan(prob_lists["b_ko"]),
        "b_sub_odds_prob": _avg_or_nan(prob_lists["b_sub"]),
        "b_dec_odds_prob": _avg_or_nan(prob_lists["b_dec"]),
    }


def _merge_method_results(existing: Optional[dict], incoming: Optional[dict]) -> Optional[dict]:
    if incoming is None:
        return existing
    if existing is None:
        return dict(incoming)

    merged = {}
    for column in _METHOD_FEATURE_COLUMNS:
        existing_val = existing.get(column, np.nan)
        incoming_val = incoming.get(column, np.nan)
        if np.isnan(existing_val):
            merged[column] = incoming_val
        elif np.isnan(incoming_val):
            merged[column] = existing_val
        else:
            merged[column] = float(np.mean([existing_val, incoming_val]))
    return merged


def _parse_datetime_like(value: object) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _as_of_cutoff(value: object) -> Optional[datetime]:
    cutoff = _parse_datetime_like(value)
    if cutoff is None:
        return None
    text = str(value or "").strip()
    if "T" not in text and " " not in text:
        cutoff = cutoff + timedelta(days=1) - timedelta(microseconds=1)
    return cutoff


def _jsonable_method_result(result: dict) -> dict:
    payload = {}
    for column in _METHOD_FEATURE_COLUMNS:
        value = result.get(column, np.nan)
        payload[column] = None if np.isnan(value) else float(value)
    return payload


def _method_result_from_json(payload: dict) -> dict:
    result = {}
    for column in _METHOD_FEATURE_COLUMNS:
        value = payload.get(column)
        result[column] = np.nan if value is None else float(value)
    return result


def _snapshot_path(snapshot_time: str) -> Path:
    timestamp = datetime.fromisoformat(snapshot_time).strftime("%Y%m%d_%H%M%S")
    return METHOD_ODDS_SNAPSHOT_DIR / f"method_odds_{timestamp}.json"


def _save_snapshot(snapshot: dict) -> Path:
    path = _snapshot_path(snapshot["snapshot_time"])
    path.write_text(json.dumps(snapshot, indent=2))
    logger.info("Saved method-odds snapshot: %s", path)
    return path


def _load_snapshot(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Failed to read method-odds snapshot %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        return None
    data.setdefault("snapshot_path", str(path))
    return data


def _snapshot_is_stale(snapshot: dict, *, max_age: Optional[timedelta] = None) -> bool:
    if max_age is None:
        max_age = METHOD_ODDS_SNAPSHOT_MAX_AGE
    if max_age is None:
        return False
    snapshot_time = _parse_datetime_like(snapshot.get("snapshot_time"))
    if snapshot_time is None:
        return False
    return datetime.now(timezone.utc) - snapshot_time > max_age


def load_latest_method_odds_snapshot(
    *,
    require_records: bool = True,
    allow_stale: bool = False,
    max_age: Optional[timedelta] = None,
    as_of_date: Optional[str] = None,
) -> Optional[dict]:
    cutoff = _as_of_cutoff(as_of_date)
    for path in sorted(METHOD_ODDS_SNAPSHOT_DIR.glob("method_odds_*.json"), reverse=True):
        snapshot = _load_snapshot(path)
        if snapshot is None:
            continue
        if cutoff is not None:
            snapshot_time = _parse_datetime_like(snapshot.get("snapshot_time"))
            if snapshot_time is None or snapshot_time > cutoff:
                continue
        if require_records and not snapshot.get("records"):
            continue
        if cutoff is None and not allow_stale and _snapshot_is_stale(snapshot, max_age=max_age):
            continue
        return snapshot
    return None


def _snapshot_record(
    *,
    fighter_a: str,
    fighter_b: str,
    method_odds: dict,
    source: str,
    captured_at: str,
    event_id: str = "",
    commence_time: str = "",
    event_title: str = "",
) -> dict:
    return {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "fighter_a_norm": _normalize_name(fighter_a),
        "fighter_b_norm": _normalize_name(fighter_b),
        "event_id": str(event_id or ""),
        "commence_time": str(commence_time or ""),
        "event_title": str(event_title or ""),
        "source": source,
        "captured_at": captured_at,
        "method_odds": _jsonable_method_result(method_odds),
    }


def _orient_result_for_query(record: dict, home_is_a: bool) -> dict:
    raw = _method_result_from_json(record.get("method_odds", {}))
    if home_is_a:
        return raw
    return {
        "a_ko_odds_prob": raw["b_ko_odds_prob"],
        "a_sub_odds_prob": raw["b_sub_odds_prob"],
        "a_dec_odds_prob": raw["b_dec_odds_prob"],
        "b_ko_odds_prob": raw["a_ko_odds_prob"],
        "b_sub_odds_prob": raw["a_sub_odds_prob"],
        "b_dec_odds_prob": raw["a_dec_odds_prob"],
    }


def _record_candidates(
    fighter_a: str,
    fighter_b: str,
    *,
    event_id: Optional[str] = None,
    commence_time: Optional[str] = None,
    as_of_date: Optional[str] = None,
) -> list[tuple[dict, bool]]:
    snapshot = load_latest_method_odds_snapshot(
        require_records=True,
        allow_stale=as_of_date is not None,
        as_of_date=as_of_date,
    )
    if snapshot is None:
        stale_snapshot = load_latest_method_odds_snapshot(require_records=True, allow_stale=True)
        if as_of_date is not None:
            logger.warning("No method-odds snapshot is available on or before %s", as_of_date)
        elif stale_snapshot is not None and _snapshot_is_stale(stale_snapshot):
            logger.warning("Newest method-odds snapshot is stale: %s", stale_snapshot.get("snapshot_time"))
        return []

    requested_commence = _parse_datetime_like(commence_time)
    candidates: list[tuple[dict, bool]] = []
    for record in snapshot.get("records", []):
        matched, home_is_a = _match_fight(
            record.get("fighter_a", ""),
            record.get("fighter_b", ""),
            fighter_a,
            fighter_b,
        )
        if not matched or home_is_a is None:
            continue

        record_event_id = str(record.get("event_id", "") or "")
        if event_id and record_event_id and record_event_id != str(event_id):
            continue

        if requested_commence is not None:
            record_commence = _parse_datetime_like(record.get("commence_time"))
            if record_commence is not None and record_commence != requested_commence:
                continue

        candidates.append((record, home_is_a))

    return candidates


def _choose_snapshot_record(
    fighter_a: str,
    fighter_b: str,
    *,
    event_id: Optional[str] = None,
    commence_time: Optional[str] = None,
    as_of_date: Optional[str] = None,
) -> Optional[tuple[dict, bool]]:
    candidates = _record_candidates(
        fighter_a,
        fighter_b,
        event_id=event_id,
        commence_time=commence_time,
        as_of_date=as_of_date,
    )
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    if event_id:
        exact = [candidate for candidate in candidates if candidate[0].get("event_id") == str(event_id)]
        if len(exact) == 1:
            return exact[0]

    unique_keys = {
        (
            candidate[0].get("event_id", ""),
            candidate[0].get("commence_time", ""),
            candidate[0].get("fighter_a_norm", ""),
            candidate[0].get("fighter_b_norm", ""),
        )
        for candidate in candidates
    }
    if len(unique_keys) == 1:
        return candidates[0]

    logger.warning("Ambiguous method-odds snapshot match for %s vs %s", fighter_a, fighter_b)
    return None


def _extract_method_probs_from_event(
    event: dict,
    home_is_a: bool,
    fighter_a: str,
    fighter_b: str,
) -> Optional[dict]:
    """
    Extract method-of-victory implied probabilities from an API event.
    """
    home_name = event.get("home_team", "")
    away_name = event.get("away_team", "")

    a_name = home_name if home_is_a else away_name
    b_name = away_name if home_is_a else home_name
    prob_lists = _collect_method_probs()

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            for outcome in market.get("outcomes", []):
                outcome_name = outcome.get("name", "")
                method = _classify_method(outcome_name)
                if method is None:
                    continue

                fighter_side = _identify_fighter_side(outcome_name, a_name, b_name)
                if fighter_side is None:
                    fighter_side = _identify_fighter_side(outcome_name, fighter_a, fighter_b)
                if fighter_side is None:
                    continue

                prob = _american_to_implied_prob(outcome.get("price"))
                if np.isnan(prob):
                    continue

                prob_lists[f"{fighter_side}_{method}"].append(prob)

    return _finalize_method_probs(prob_lists)


_BFO_TRANSIENT_CODES = {500, 502, 503, 504}
_BFO_MAX_RETRIES = 2
_BFO_RETRY_BACKOFF = 3.0


def _bfo_get(url: str, **kwargs) -> Optional[requests.Response]:
    """GET with retry on transient server errors. Returns None on failure."""
    kwargs.setdefault("headers", HEADERS)
    kwargs.setdefault("timeout", 30)
    last_exc: Optional[Exception] = None
    for attempt in range(_BFO_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, **kwargs)
            if resp.status_code in _BFO_TRANSIENT_CODES and attempt < _BFO_MAX_RETRIES:
                logger.debug("BFO transient %s on %s, retry %d", resp.status_code, url, attempt + 1)
                time.sleep(_BFO_RETRY_BACKOFF * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            if attempt < _BFO_MAX_RETRIES:
                time.sleep(_BFO_RETRY_BACKOFF * (attempt + 1))
                continue
    logger.warning("BFO request failed after retries: %s — %s", url, last_exc)
    return None


def _bfo_find_fighter_url(fighter_name: str) -> Optional[str]:
    """Search BFO for a fighter and return the best matching fighter page URL."""
    resp = _bfo_get("https://www.bestfightodds.com/search", params={"query": fighter_name})
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    for link in soup.select('a[href*="/fighters/"]'):
        link_text = link.get_text(strip=True)
        if _names_match(fighter_name, link_text):
            href = link["href"]
            return href if href.startswith("http") else f"https://www.bestfightodds.com{href}"
    return None


def _bfo_find_event_url_from_fighter_page(
    fighter_page_url: str,
    opponent_name: str,
) -> Optional[str]:
    """Fetch a BFO fighter page and find the event URL for a matchup with the opponent."""
    time.sleep(REQUEST_DELAY)
    resp = _bfo_get(fighter_page_url)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.select_one("table.team-stats-table")
    if table is None:
        return None

    rows = table.select("tr")
    current_event_href: Optional[str] = None
    expect_opponent = False

    for tr in rows:
        classes = tr.get("class") or []

        # Event header rows contain the event link
        event_link = tr.select_one('a[href*="/events/"]')
        if event_link:
            current_event_href = event_link.get("href", "")
            if not current_event_href.startswith("http"):
                current_event_href = f"https://www.bestfightodds.com{current_event_href}"

        # Main-row is the searched fighter; the next row is the opponent
        if "main-row" in classes:
            expect_opponent = True
            continue

        if expect_opponent:
            expect_opponent = False
            opponent_link = tr.select_one('a[href*="/fighters/"]')
            if opponent_link and _names_match(opponent_name, opponent_link.get_text(strip=True)):
                if current_event_href:
                    return current_event_href

    return None


def _search_bfo_candidate_url(
    fighter_a: str,
    fighter_b: str,
    *,
    event_title: Optional[str] = None,
) -> Optional[str]:
    """Find the BFO event page for a matchup by searching for a fighter,
    navigating to their profile, and finding the event with the opponent."""
    # Try both orderings — search for fighter A first, then B as fallback
    for searcher, opponent in [(fighter_a, fighter_b), (fighter_b, fighter_a)]:
        fighter_url = _bfo_find_fighter_url(searcher)
        if not fighter_url:
            logger.debug("BFO: no fighter page found for %s", searcher)
            time.sleep(REQUEST_DELAY)
            continue

        event_url = _bfo_find_event_url_from_fighter_page(fighter_url, opponent)
        if event_url:
            logger.debug("BFO: found event page %s for %s vs %s", event_url, fighter_a, fighter_b)
            return event_url

        logger.debug("BFO: fighter page for %s had no matchup with %s", searcher, opponent)

    return None


def _parse_bfo_method_odds(soup: BeautifulSoup, fighter_a: str, fighter_b: str) -> Optional[dict]:
    """
    Parse method-of-victory odds from a BestFightOdds page.

    If the page does not clearly reference both fighters or the props table
    cannot be parsed confidently, return None.
    """
    page_text = soup.get_text(" ", strip=True)
    if not (_names_match(fighter_a, page_text) and _names_match(fighter_b, page_text)):
        return None

    prob_lists = _collect_method_probs()

    for row in soup.select("tr"):
        # Use the <th> label text for classification (avoids mixing odds
        # into the method/fighter matching). Fall back to the first <td> if
        # no <th> exists (some fixture / older BFO formats use <td> labels).
        th = row.select_one("th")
        if th:
            label_text = th.get_text(" ", strip=True)
        else:
            first_td = row.select_one("td")
            label_text = first_td.get_text(" ", strip=True) if first_td else ""
        row_text = row.get_text(" ", strip=True)
        classify_text = label_text or row_text

        method = _classify_method(classify_text, strict=True)
        if method is None:
            continue

        fighter_side = _identify_fighter_side(classify_text, fighter_a, fighter_b)
        if fighter_side is None:
            continue

        odds_strings = re.findall(r"[+-]\d+", row_text)
        if not odds_strings:
            odds_strings = [
                cell.get_text(strip=True)
                for cell in row.select("td")
                if re.fullmatch(r"[+-]\d+", cell.get_text(strip=True))
            ]

        if not odds_strings:
            continue

        parsed_row = False
        for odds_text in odds_strings:
            prob = _american_to_implied_prob(odds_text)
            if np.isnan(prob):
                continue
            prob_lists[f"{fighter_side}_{method}"].append(prob)
            parsed_row = True

        if not parsed_row:
            return None

    return _finalize_method_probs(prob_lists)


def _scrape_bestfightodds(
    fighter_a: str,
    fighter_b: str,
    *,
    event_title: Optional[str] = None,
) -> Optional[dict]:
    fight_url = _search_bfo_candidate_url(fighter_a, fighter_b, event_title=event_title)
    if not fight_url:
        logger.debug("BFO: could not find a confident fight page for %s vs %s", fighter_a, fighter_b)
        return None

    time.sleep(REQUEST_DELAY)
    resp = _bfo_get(fight_url)
    if resp is None:
        logger.warning("BFO fight page fetch failed for %s vs %s", fighter_a, fighter_b)
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    method_probs = _parse_bfo_method_odds(soup, fighter_a, fighter_b)
    if method_probs is not None:
        logger.info("Got method odds from BestFightOdds for %s vs %s", fighter_a, fighter_b)
    else:
        logger.debug("BFO: event page found but no method props available for %s vs %s", fighter_a, fighter_b)
    return method_probs


def _collect_api_snapshot_records() -> tuple[list[dict], list[dict]]:
    source_runs: list[dict] = []
    if not ODDS_API_KEY:
        source_runs.append(
            {
                "source": "odds_api",
                "status": "failed",
                "captured_at": _now_iso(),
                "error": "missing ODDS_API_KEY",
                "record_count": 0,
            }
        )
        return [], source_runs

    base_url = "https://api.the-odds-api.com/v4"
    sport_key = "mma_mixed_martial_arts"
    method_market_keys = ["method_of_victory", "outrights"]
    merged_records: dict[tuple[str, str, str], dict] = {}

    for market_key in method_market_keys:
        captured_at = _now_iso()
        try:
            # Explicitly bypass any proxy env vars (e.g. SOCKS proxy for
            # Polymarket geoblock) — the Odds API is a public endpoint.
            resp = requests.get(
                f"{base_url}/sports/{sport_key}/odds",
                params={
                    "apiKey": ODDS_API_KEY,
                    "regions": "us",
                    "markets": market_key,
                    "oddsFormat": "american",
                },
                timeout=30,
                proxies={"http": None, "https": None},
            )
            if resp.status_code == 422:
                source_runs.append(
                    {
                        "source": f"odds_api:{market_key}",
                        "status": "failed",
                        "captured_at": captured_at,
                        "error": "market unsupported",
                        "record_count": 0,
                    }
                )
                continue
            resp.raise_for_status()
            events = resp.json()
        except Exception as exc:
            logger.debug("Odds API %s market failed: %s", market_key, exc)
            source_runs.append(
                {
                    "source": f"odds_api:{market_key}",
                    "status": "failed",
                    "captured_at": captured_at,
                    "error": str(exc),
                    "record_count": 0,
                }
            )
            continue

        time.sleep(REQUEST_DELAY)

        record_count = 0
        for event in events:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            matched, home_is_a = _match_fight(home, away, home, away)
            if not matched or home_is_a is None:
                continue
            method_probs = _extract_method_probs_from_event(event, home_is_a, home, away)
            if method_probs is None:
                continue

            key = (
                str(event.get("id", "") or ""),
                _normalize_name(home),
                _normalize_name(away),
            )
            existing = merged_records.get(key)
            merged_method_odds = _merge_method_results(existing.get("method_odds") if existing else None, method_probs)
            merged_records[key] = {
                "event_id": str(event.get("id", "") or ""),
                "commence_time": str(event.get("commence_time", "") or ""),
                "fighter_a": home,
                "fighter_b": away,
                "method_odds": merged_method_odds,
                "source": f"odds_api:{market_key}",
                "captured_at": captured_at,
            }
            record_count += 1

        source_runs.append(
            {
                "source": f"odds_api:{market_key}",
                "status": "success" if record_count else "failed",
                "captured_at": captured_at,
                "record_count": record_count,
                "error": "" if record_count else "no confident method odds parsed",
            }
        )

    records = [
        _snapshot_record(
            fighter_a=record["fighter_a"],
            fighter_b=record["fighter_b"],
            method_odds=record["method_odds"],
            source=record["source"],
            captured_at=record["captured_at"],
            event_id=record["event_id"],
            commence_time=record["commence_time"],
        )
        for record in merged_records.values()
        if record.get("method_odds") is not None
    ]
    return records, source_runs


def _collect_bfo_records_for_missing(tracked_fights: list[dict], existing_records: list[dict]) -> tuple[list[dict], dict]:
    def _identity_keys(payload: dict) -> set[tuple[str, str, str, str]]:
        fighter_a_norm = payload.get("fighter_a_norm") or _normalize_name(payload.get("fighter_a", ""))
        fighter_b_norm = payload.get("fighter_b_norm") or _normalize_name(payload.get("fighter_b", ""))
        keys: set[tuple[str, str, str, str]] = set()
        event_id = str(payload.get("event_id", "") or "")
        commence_time = str(payload.get("commence_time", "") or "")
        if event_id:
            keys.add(("event_id", event_id, fighter_a_norm, fighter_b_norm))
        if commence_time:
            keys.add(("commence_time", commence_time, fighter_a_norm, fighter_b_norm))
        return keys

    existing_keys = {
        identity_key
        for record in existing_records
        for identity_key in _identity_keys(record)
    }

    records: list[dict] = []
    attempted = 0
    for fight in tracked_fights:
        fighter_a = fight.get("fighter_a", "")
        fighter_b = fight.get("fighter_b", "")
        fight_keys = _identity_keys(fight)
        if fight_keys and any(key in existing_keys for key in fight_keys):
            continue

        attempted += 1
        method_probs = _scrape_bestfightodds(
            fighter_a,
            fighter_b,
            event_title=fight.get("event_title") or fight.get("event"),
        )
        if method_probs is None:
            continue

        records.append(
            _snapshot_record(
                fighter_a=fighter_a,
                fighter_b=fighter_b,
                method_odds=method_probs,
                source="bestfightodds",
                captured_at=_now_iso(),
                event_id=str(fight.get("event_id", "") or ""),
                commence_time=str(fight.get("commence_time", "") or ""),
                event_title=str(fight.get("event_title", "") or fight.get("event", "") or ""),
            )
        )

    source_run = {
        "source": "bestfightodds",
        "status": "success" if records else "failed",
        "captured_at": _now_iso(),
        "record_count": len(records),
        "error": "" if records else ("no fights attempted" if attempted == 0 else "no confident pages parsed"),
    }
    return records, source_run


def collect_method_odds_snapshot(*, tracked_fights: Optional[list[dict]] = None) -> dict:
    """
    Collect method odds into a timestamped normalized snapshot.

    Inference should not call this. Use it from scheduled monitor/collector jobs.
    """
    records, source_runs = _collect_api_snapshot_records()

    if tracked_fights:
        bfo_records, bfo_run = _collect_bfo_records_for_missing(tracked_fights, records)
        source_runs.append(bfo_run)
        records.extend(bfo_records)

    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_time": _now_iso(),
        "status": "success" if records else "failed",
        "record_count": len(records),
        "records": records,
        "sources": source_runs,
    }
    path = _save_snapshot(snapshot)
    snapshot["snapshot_path"] = str(path)
    return snapshot


def get_method_odds(
    fighter_a: str,
    fighter_b: str,
    *,
    event_id: Optional[str] = None,
    commence_time: Optional[str] = None,
    as_of_date: Optional[str] = None,
) -> dict:
    """
    Read method-of-victory implied probabilities from the latest snapshot.

    If no unique, confident snapshot match exists, return NaNs.
    """
    cached = _get_cached_method_odds(
        fighter_a,
        fighter_b,
        event_id=event_id,
        commence_time=commence_time,
        as_of_date=as_of_date,
    )
    if cached is not None:
        return cached

    chosen = _choose_snapshot_record(
        fighter_a,
        fighter_b,
        event_id=event_id,
        commence_time=commence_time,
        as_of_date=as_of_date,
    )
    if chosen is None:
        nan_result = _nan_result()
        _set_cached_method_odds(
            fighter_a,
            fighter_b,
            nan_result,
            event_id=event_id,
            commence_time=commence_time,
            as_of_date=as_of_date,
        )
        return nan_result

    record, home_is_a = chosen
    result = _orient_result_for_query(record, home_is_a)
    _set_cached_method_odds(
        fighter_a,
        fighter_b,
        result,
        event_id=event_id,
        commence_time=commence_time,
        as_of_date=as_of_date,
    )
    return result
