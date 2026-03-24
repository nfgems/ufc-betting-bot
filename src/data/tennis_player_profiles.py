"""Official ATP/WTA tennis player profile collection and safe match enrichment."""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

logger = logging.getLogger(__name__)

TENNIS_RAW_DIR = RAW_DATA_DIR / "tennis"
ATP_PROFILE_CACHE_PATH = TENNIS_RAW_DIR / "atp" / "atp_player_profiles_official.csv"
WTA_PROFILE_CACHE_PATH = TENNIS_RAW_DIR / "wta" / "wta_player_profiles_official.csv"
TENNIS_PROFILE_SUPPLEMENT_PATH = TENNIS_RAW_DIR / "player_profiles_verified_supplement.csv"
TENNIS_PROFILE_TARGETS_PATH = PROCESSED_DATA_DIR / "tennis" / "player_profile_targets.csv"
TENNIS_PROFILE_REMAINING_TARGETS_PATH = PROCESSED_DATA_DIR / "tennis" / "player_profile_remaining_targets.csv"
TENNIS_PROFILE_SUMMARY_PATH = PROCESSED_DATA_DIR / "tennis" / "player_profile_enrichment_summary.json"

ATP_PLAYER_PROFILE_API_URL = "https://www.atptour.com/en/-/www/players/hero/{player_id}"
WTA_PLAYER_API_URL = "https://api.wtatennis.com/tennis/players/{player_id}"
WTA_PLAYER_DETAILED_API_URL = "https://api.wtatennis.com/tennis/players/{player_id}/detailed"
WTA_PLAYER_PROFILE_URL = "https://www.wtatennis.com/players/{player_id}/{slug}/"
TENNIS_EXPLORER_SEARCH_URL = "https://www.tennisexplorer.com/res/ajax/search.php"
TENNIS_EXPLORER_PLAYER_URL = "https://www.tennisexplorer.com/player/{slug}/"
TENNIS_EXPLORER_MATCH_DETAIL_MAX_LINKS = 16
TENNIS_EXPLORER_MATCH_DETAIL_MAX_EXACT_PAGES = 6
WIKIPEDIA_PAGE_URL = "https://{language}.wikipedia.org/wiki/{slug}"
WIKIPEDIA_SEARCH_API_URL = "https://{language}.wikipedia.org/w/api.php"
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"

REQUEST_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
PLAYER_PROFILE_REQUEST_PAUSE_SECONDS = 0.25
WIKIPEDIA_LANGUAGES = ("en", "de", "fr", "es", "it", "pl", "ru")
PLAYER_PROFILE_COLUMNS = [
    "tour",
    "player_id",
    "full_name",
    "first_name",
    "last_name",
    "country_code",
    "country_name",
    "birth_date",
    "birth_place",
    "hand_code",
    "hand_description",
    "backhand_code",
    "backhand_description",
    "height_cm",
    "weight_kg",
    "pro_year",
    "coach",
    "active_status",
    "active_status_description",
    "current_singles_rank",
    "current_singles_rank_source",
    "current_doubles_rank",
    "profile_url",
    "source_api_url",
    "source_page_url",
    "source_kind",
    "fetch_status",
    "fetched_at_utc",
    "observed_rows",
    "latest_observed_event_date",
    "error",
    "source_notes",
]
ENRICHABLE_PROFILE_FIELDS = [
    "player_a_age",
    "player_b_age",
    "player_a_hand",
    "player_b_hand",
    "player_a_height_cm",
    "player_b_height_cm",
]
PROFILE_RETRY_DAYS = 30
PROFILE_RETRY_OBSERVED_ROW_DELTA = 3


def _profile_cache_path(tour: str) -> Path:
    tour_norm = str(tour or "").strip().lower()
    if tour_norm == "atp":
        return ATP_PROFILE_CACHE_PATH
    if tour_norm == "wta":
        return WTA_PROFILE_CACHE_PATH
    raise ValueError(f"Unsupported tennis tour for profile cache: {tour!r}")


def _ensure_profile_dirs() -> None:
    ATP_PROFILE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    WTA_PROFILE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TENNIS_PROFILE_SUPPLEMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TENNIS_PROFILE_TARGETS_PATH.parent.mkdir(parents=True, exist_ok=True)


_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

_BROWSER_HEADERS: dict[str, str] = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


def _request_session(session: Optional[requests.Session] = None) -> requests.Session:
    session = session or requests.Session()
    if str(session.headers.get("User-Agent") or "").startswith("python-requests/"):
        session.headers["User-Agent"] = _BROWSER_USER_AGENT
    else:
        session.headers.setdefault("User-Agent", _BROWSER_USER_AGENT)
    for key, value in _BROWSER_HEADERS.items():
        session.headers.setdefault(key, value)
    return session


def _get_with_retries(
    session: requests.Session,
    url: str,
    timeout: int = 60,
    max_attempts: int = 5,
    **kwargs,
) -> requests.Response:
    response: Optional[requests.Response] = None
    for attempt in range(max_attempts):
        response = session.get(url, timeout=timeout, **kwargs)
        if response.status_code not in REQUEST_RETRY_STATUS_CODES:
            return response

        if attempt >= max_attempts - 1:
            return response

        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            wait_seconds = max(float(retry_after), 1.0)
        elif response.status_code == 429:
            wait_seconds = float(10 * (attempt + 1))
        else:
            wait_seconds = float(min(2**attempt, 8))
        logger.warning(
            "Retrying tennis player profile request %s after HTTP %s in %.1f seconds (attempt %s/%s)",
            url,
            response.status_code,
            wait_seconds,
            attempt + 2,
            max_attempts,
        )
        time.sleep(wait_seconds)

    if response is None:
        raise requests.RequestException(f"Failed to fetch {url}")
    return response


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _normalized_name_key(value: object) -> str:
    return _normalize_text(value)


def _initial_surname_key(value: object) -> str:
    tokens = _normalize_text(value).split()
    if len(tokens) < 2 or not tokens[0]:
        return ""
    surname = " ".join(tokens[1:]).strip()
    if not surname:
        return ""
    return f"{tokens[0][0]} {surname}".strip()


def _match_has_initial_name(value: object) -> bool:
    tokens = _normalize_text(value).split()
    return len(tokens) >= 2 and len(tokens[0]) == 1


def _slugify_name(value: object) -> str:
    text = _normalize_text(value).replace(" ", "-")
    return re.sub(r"-{2,}", "-", text).strip("-")


def canonical_player_id(tour: object, player_id: object) -> str:
    if player_id is None or pd.isna(player_id):
        return ""

    text = str(player_id).strip()
    if not text:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]

    if str(tour or "").strip().lower() == "atp":
        return text.upper()
    return text


def _missing_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return series.isna() | series.astype(str).str.strip().eq("")
    return series.isna()


def _find_person_object(payload: object) -> Optional[dict[str, object]]:
    if isinstance(payload, dict):
        if str(payload.get("@type") or "").lower() == "person":
            return payload
        for value in payload.values():
            found = _find_person_object(value)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_person_object(item)
            if found is not None:
                return found
    return None


def _wta_structured_person_and_soup(html: str) -> tuple[dict[str, object], BeautifulSoup]:
    soup = BeautifulSoup(html, "lxml")
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw_text = script.get_text(strip=True)
        if not raw_text:
            continue
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            continue
        person = _find_person_object(payload)
        if person is not None:
            return person, soup
    return {}, soup


def _wta_info_blocks(soup: BeautifulSoup) -> dict[str, str]:
    info: dict[str, str] = {}
    for block in soup.select(".profile-bio__info-block"):
        title_node = block.select_one(".profile-bio__info-title")
        value_nodes = block.select(".profile-bio__info-content")
        title = " ".join(title_node.get_text(" ", strip=True).split()) if title_node else ""
        value = ""
        for node in value_nodes:
            candidate = " ".join(node.get_text(" ", strip=True).split())
            if candidate:
                value = candidate
                break
        if title:
            info[title] = value
    return info


def _wta_summary_coach(soup: BeautifulSoup) -> str:
    for item in soup.select(".profile-bio__summary-item"):
        text = " ".join(item.get_text(" ", strip=True).split())
        if text.lower().startswith("coached by "):
            return text[len("Coached by ") :].strip()
    return ""


def _parse_wta_height_cm(height_text: object) -> Optional[int]:
    text = str(height_text or "").strip()
    if not text:
        return None
    if text in {"-", "N/A"}:
        return None
    match = re.search(r"(\d{3}(?:\.\d+)?)\s*cm\b", text, flags=re.IGNORECASE)
    if match:
        return int(round(float(match.group(1))))
    match = re.search(r"(\d(?:\.\d+)?)\s*m\b", text, flags=re.IGNORECASE)
    if match:
        return int(round(float(match.group(1)) * 100))
    match = re.search(r"(\d{3}(?:\.\d+)?)\b", text)
    if match:
        value = float(match.group(1))
        if value >= 140:
            return int(round(value))
    match = re.search(r"(\d+)\s*'\s*(\d+)\s*(?:\"|in)?", text)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        return int(round((feet * 12 + inches) * 2.54))
    return None


def _hand_code_from_description(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if "left" in text:
        return "L"
    if "right" in text:
        return "R"
    return ""


def _hand_description_from_code(hand_code: object) -> str:
    code = str(hand_code or "").strip().upper()
    if code == "L":
        return "Left-Handed"
    if code == "R":
        return "Right-Handed"
    return ""


def _ensure_profile_columns(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    for column in PLAYER_PROFILE_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    return working[PLAYER_PROFILE_COLUMNS].copy()


def _safe_int(value: object) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _safe_date_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return ""
    return timestamp.date().isoformat()


def _safe_timestamp_utc(value: object) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return pd.NaT
    return timestamp


def _names_match(expected_name: object, actual_name: object) -> bool:
    expected = _normalize_text(expected_name)
    actual = _normalize_text(actual_name)
    return bool(expected) and bool(actual) and expected == actual


def _blank_profile_row(tour: str, player_id: object, *, source_kind: str, fetch_status: str, error: str = "") -> dict[str, object]:
    row = {column: "" for column in PLAYER_PROFILE_COLUMNS}
    row.update(
        {
            "tour": str(tour or "").lower(),
            "player_id": canonical_player_id(tour, player_id),
            "source_kind": source_kind,
            "fetch_status": fetch_status,
            "fetched_at_utc": _now_utc_iso(),
            "error": error,
        }
    )
    return row


def _profile_identity_key(
    tour: object,
    player_id: object,
    player_name: object,
) -> tuple[str, str]:
    tour_key = str(tour or "").strip().lower()
    canonical_id = canonical_player_id(tour_key, player_id)
    if canonical_id:
        return tour_key, f"id:{canonical_id}"
    name_key = _normalized_name_key(player_name)
    return tour_key, f"name:{name_key}" if name_key else ""


def _profile_attempt_key(
    tour: object,
    player_id: object,
    player_name: object,
    source_kind: object,
) -> tuple[str, str, str]:
    tour_key, identity_key = _profile_identity_key(tour, player_id, player_name)
    return tour_key, identity_key, str(source_kind or "").strip().lower()


def _stamp_profile_observation_metadata(
    row: dict[str, object],
    *,
    target: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    stamped = dict(row)
    if target is None:
        return stamped
    stamped["observed_rows"] = _safe_int(target.get("affected_rows"))
    stamped["latest_observed_event_date"] = _safe_date_text(target.get("latest_event_date"))
    return stamped


def _should_retry_cached_profile_attempt(
    existing_row: Optional[dict[str, object]],
    *,
    target: Optional[dict[str, object]] = None,
    now_utc: Optional[pd.Timestamp] = None,
) -> bool:
    if not existing_row:
        return True

    status = str(existing_row.get("fetch_status") or "").strip().lower()
    if status in {"ok", "partial"}:
        return False
    if status == "name_mismatch":
        return False
    if status != "not_found":
        return True

    now_utc = now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC")
    fetched_at = _safe_timestamp_utc(existing_row.get("fetched_at_utc"))
    if pd.isna(fetched_at):
        return True
    if (now_utc - fetched_at) >= pd.Timedelta(days=PROFILE_RETRY_DAYS):
        return True

    if target is None:
        return False
    current_rows = _safe_int(target.get("affected_rows"))
    previous_rows = _safe_int(existing_row.get("observed_rows"))
    if current_rows is None or previous_rows is None:
        return False
    return current_rows >= (previous_rows + PROFILE_RETRY_OBSERVED_ROW_DELTA)


def fetch_atp_player_profile(
    player_id: object,
    session: Optional[requests.Session] = None,
    expected_name: object = None,
) -> dict[str, object]:
    session = _request_session(session)
    canonical_id = canonical_player_id("atp", player_id)
    api_url = ATP_PLAYER_PROFILE_API_URL.format(player_id=canonical_id.lower())
    response = _get_with_retries(session, api_url, timeout=60)
    if response.status_code == 404:
        row = _blank_profile_row(
            "atp",
            canonical_id,
            source_kind="atp_official_hero_json",
            fetch_status="not_found",
            error="Official ATP player profile endpoint returned 404.",
        )
        row["source_api_url"] = api_url
        return row

    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError:
        row = _blank_profile_row(
            "atp",
            canonical_id,
            source_kind="atp_official_hero_json",
            fetch_status="not_found",
            error="Official ATP player profile endpoint returned non-JSON content.",
        )
        row["source_api_url"] = api_url
        return row
    if not isinstance(payload, dict) or not payload:
        row = _blank_profile_row(
            "atp",
            canonical_id,
            source_kind="atp_official_hero_json",
            fetch_status="not_found",
            error="Official ATP player profile endpoint returned an empty payload.",
        )
        row["source_api_url"] = api_url
        return row

    full_name = " ".join(
        part for part in [str(payload.get("FirstName") or "").strip(), str(payload.get("LastName") or "").strip()] if part
    )
    if expected_name and full_name and not (
        _names_match(expected_name, full_name) or _names_match_by_initial_and_surname(expected_name, full_name)
    ):
        row = _blank_profile_row(
            "atp",
            canonical_id,
            source_kind="atp_official_hero_json",
            fetch_status="name_mismatch",
            error=f"Official ATP profile name mismatch: expected {expected_name!r}, got {full_name!r}.",
        )
        row["full_name"] = full_name
        row["source_api_url"] = api_url
        return row

    profile_url = str(payload.get("ScRelativeUrlPlayerProfile") or "").strip()
    if profile_url and not profile_url.startswith("http"):
        profile_url = f"https://www.atptour.com{profile_url}"

    row = _blank_profile_row(
        "atp",
        canonical_id,
        source_kind="atp_official_hero_json",
        fetch_status="ok",
    )
    row.update(
        {
            "full_name": full_name,
            "first_name": str(payload.get("FirstName") or "").strip(),
            "last_name": str(payload.get("LastName") or "").strip(),
            "country_code": str(payload.get("NatlId") or "").strip(),
            "country_name": str(payload.get("Nationality") or "").strip(),
            "birth_date": _safe_date_text(payload.get("BirthDate")),
            "birth_place": str(payload.get("BirthCity") or "").strip(),
            "hand_code": str((payload.get("PlayHand") or {}).get("Id") or "").strip(),
            "hand_description": str((payload.get("PlayHand") or {}).get("Description") or "").strip(),
            "backhand_code": str((payload.get("BackHand") or {}).get("Id") or "").strip(),
            "backhand_description": str((payload.get("BackHand") or {}).get("Description") or "").strip(),
            "height_cm": _safe_int(payload.get("HeightCm")),
            "weight_kg": _safe_int(payload.get("WeightKg")),
            "pro_year": _safe_int(payload.get("ProYear")),
            "coach": str(payload.get("Coach") or "").strip(),
            "active_status": str((payload.get("Active") or {}).get("Id") or "").strip(),
            "active_status_description": str((payload.get("Active") or {}).get("Description") or "").strip(),
            "current_singles_rank": _safe_int(payload.get("SglRank")),
            "current_singles_rank_source": "profile_current",
            "current_doubles_rank": _safe_int(payload.get("DblRank")),
            "profile_url": profile_url,
            "source_api_url": api_url,
            "source_page_url": profile_url,
        }
    )
    return row


def _wta_detailed_bio_payload(player_id: object, session: Optional[requests.Session] = None) -> dict[str, object]:
    session = _request_session(session)
    canonical_id = canonical_player_id("wta", player_id)
    response = _get_with_retries(session, WTA_PLAYER_DETAILED_API_URL.format(player_id=canonical_id), timeout=60)
    if response.status_code == 404:
        return {}
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return {}
    bio = payload.get("bio") or {}
    return bio if isinstance(bio, dict) else {}


def fetch_wta_player_profile(
    player_id: object,
    session: Optional[requests.Session] = None,
    expected_name: object = None,
) -> dict[str, object]:
    session = _request_session(session)
    canonical_id = canonical_player_id("wta", player_id)
    api_url = WTA_PLAYER_API_URL.format(player_id=canonical_id)
    api_response = _get_with_retries(session, api_url, timeout=60)
    if api_response.status_code == 404:
        row = _blank_profile_row(
            "wta",
            canonical_id,
            source_kind="wta_official_api_plus_profile_page",
            fetch_status="not_found",
            error="Official WTA player API returned 404.",
        )
        row["source_api_url"] = api_url
        return row

    api_response.raise_for_status()
    api_payload = api_response.json()
    full_name = str(api_payload.get("fullName") or "").strip()
    if expected_name and full_name and not (
        _names_match(expected_name, full_name) or _names_match_by_initial_and_surname(expected_name, full_name)
    ):
        row = _blank_profile_row(
            "wta",
            canonical_id,
            source_kind="wta_official_api_plus_profile_page",
            fetch_status="name_mismatch",
            error=f"Official WTA profile name mismatch: expected {expected_name!r}, got {full_name!r}.",
        )
        row["full_name"] = full_name
        row["source_api_url"] = api_url
        return row

    slug = _slugify_name(full_name)
    page_url = WTA_PLAYER_PROFILE_URL.format(player_id=canonical_id, slug=slug)
    page_response = _get_with_retries(session, page_url, timeout=60)

    row = _blank_profile_row(
        "wta",
        canonical_id,
        source_kind="wta_official_api_plus_profile_page",
        fetch_status="ok",
    )
    row.update(
        {
            "full_name": full_name,
            "first_name": str(api_payload.get("firstName") or "").strip(),
            "last_name": str(api_payload.get("lastName") or "").strip(),
            "country_code": str(api_payload.get("countryCode") or "").strip(),
            "birth_date": _safe_date_text(api_payload.get("dateOfBirth")),
            "source_api_url": api_url,
            "source_page_url": page_url,
            "profile_url": page_url,
        }
    )

    detailed_bio = _wta_detailed_bio_payload(canonical_id, session=session)
    if detailed_bio:
        detailed_height_cm = _parse_wta_height_cm(detailed_bio.get("height"))
        detailed_hand_description = str(detailed_bio.get("playhand") or "").strip()
        detailed_country_name = str(detailed_bio.get("countryname") or "").strip()
        detailed_birth_place = str(detailed_bio.get("birthcity") or "").strip()
        row.update(
            {
                "country_name": detailed_country_name or row.get("country_name"),
                "birth_date": _safe_date_text(detailed_bio.get("dateofbirth") or row.get("birth_date")),
                "birth_place": detailed_birth_place or row.get("birth_place"),
                "hand_code": _hand_code_from_description(detailed_hand_description) or row.get("hand_code"),
                "hand_description": detailed_hand_description or row.get("hand_description"),
                "height_cm": detailed_height_cm if detailed_height_cm is not None else row.get("height_cm"),
                "current_singles_rank": _safe_int(detailed_bio.get("sglrank")) or row.get("current_singles_rank"),
                "current_doubles_rank": _safe_int(detailed_bio.get("dblrank")) or row.get("current_doubles_rank"),
            }
        )

    if page_response.status_code == 404:
        row["fetch_status"] = "partial"
        row["error"] = "Official WTA player page returned 404; API fields only."
        return row

    page_response.raise_for_status()
    person_payload, soup = _wta_structured_person_and_soup(page_response.text)
    info_blocks = _wta_info_blocks(soup)
    property_values = {
        str(item.get("name") or "").strip(): str(item.get("value") or "").strip()
        for item in list(person_payload.get("additionalProperty") or [])
        if isinstance(item, dict)
    }

    birth_place = str(info_blocks.get("Birthplace") or "").strip()
    if not birth_place:
        birth_place_payload = person_payload.get("birthPlace") or {}
        address_payload = birth_place_payload.get("address") or {}
        birth_locality = str(address_payload.get("addressLocality") or "").strip()
        birth_country = str(address_payload.get("addressCountry") or "").strip()
        birth_place = ", ".join(part for part in [birth_locality, birth_country] if part)

    height_cm = _parse_wta_height_cm(info_blocks.get("Height")) or _parse_wta_height_cm(detailed_bio.get("height"))
    hand_description = str(info_blocks.get("Plays") or property_values.get("Plays") or "").strip()
    if not hand_description:
        hand_description = str(detailed_bio.get("playhand") or "").strip()

    row.update(
        {
            "country_name": str((person_payload.get("nationality") or {}).get("name") or "").strip(),
            "birth_date": _safe_date_text(person_payload.get("birthDate") or row.get("birth_date")),
            "birth_place": birth_place,
            "hand_code": _hand_code_from_description(hand_description),
            "hand_description": hand_description,
            "height_cm": height_cm,
            "coach": _wta_summary_coach(soup),
            "current_singles_rank": _safe_int(property_values.get("WTA Singles Rank")) or row.get("current_singles_rank"),
            "current_singles_rank_source": "profile_current",
            "current_doubles_rank": _safe_int(property_values.get("WTA Doubles Rank")) or row.get("current_doubles_rank"),
            "source_page_url": page_url,
            "profile_url": str(person_payload.get("url") or page_url).strip() or page_url,
        }
    )
    return row


def _load_profile_supplement() -> pd.DataFrame:
    if not TENNIS_PROFILE_SUPPLEMENT_PATH.exists():
        return pd.DataFrame(columns=PLAYER_PROFILE_COLUMNS)
    frame = pd.read_csv(TENNIS_PROFILE_SUPPLEMENT_PATH, low_memory=False)
    frame = _ensure_profile_columns(frame)
    frame["tour"] = frame["tour"].astype(str).str.lower()
    frame["player_id"] = frame.apply(lambda row: canonical_player_id(row.get("tour"), row.get("player_id")), axis=1)
    return frame.reset_index(drop=True)


def _write_profile_supplement(frame: pd.DataFrame) -> Path:
    _ensure_profile_dirs()
    working = _ensure_profile_columns(frame)
    if not working.empty:
        working["tour"] = working["tour"].astype(str).str.lower()
        working["player_id"] = working.apply(lambda row: canonical_player_id(row.get("tour"), row.get("player_id")), axis=1)
        working = working.sort_values(
            ["tour", "player_id", "full_name", "source_kind", "source_page_url", "fetched_at_utc"],
            na_position="last",
        )
        dedupe_cols = ["tour", "player_id", "full_name", "source_kind", "source_page_url", "source_api_url", "source_notes"]
        dedupe_cols = [column for column in dedupe_cols if column in working.columns]
        if dedupe_cols:
            working = working.drop_duplicates(subset=dedupe_cols, keep="last").reset_index(drop=True)
    working.to_csv(TENNIS_PROFILE_SUPPLEMENT_PATH, index=False)
    return TENNIS_PROFILE_SUPPLEMENT_PATH


def _known_profile_birth_date(
    profiles_df: pd.DataFrame,
    *,
    tour: object,
    player_id: object,
    player_name: object,
) -> str:
    profiles = _ensure_profile_columns(profiles_df)
    if profiles.empty:
        return ""
    working = profiles.copy()
    working["tour"] = working["tour"].astype(str).str.lower()
    canonical_tour = str(tour or "").lower()
    canonical_id = canonical_player_id(canonical_tour, player_id)
    if canonical_id:
        id_matches = working[
            (working["tour"] == canonical_tour)
            & (
                working.apply(
                    lambda row: canonical_player_id(row.get("tour"), row.get("player_id")),
                    axis=1,
                )
                == canonical_id
            )
        ]
        value = _unique_non_missing_value(id_matches["birth_date"])
        if value:
            return str(value)

    name_key = _normalized_name_key(player_name)
    if not name_key:
        return ""
    name_matches = working[
        (working["tour"] == canonical_tour)
        & (working["full_name"].map(_normalized_name_key) == name_key)
    ]
    value = _unique_non_missing_value(name_matches["birth_date"])
    return str(value or "")


def _wikipedia_page_slug(player_name: object) -> str:
    return requests.utils.quote(str(player_name or "").strip().replace(" ", "_"))


def _wikipedia_has_exact_article(page_html: object) -> bool:
    text = str(page_html or "")
    if not text:
        return False
    no_article_markers = [
        "Wikipedia does not have an article with this exact name",
        "Diese Seite existiert nicht",
        "Wikipedia hat keinen Artikel mit diesem genauen Namen",
    ]
    return not any(marker in text for marker in no_article_markers)


def _wikipedia_is_tennis_page(page_html: object, soup: BeautifulSoup) -> bool:
    infobox = soup.select_one("table.infobox")
    if infobox is None:
        return False
    info_text = infobox.get_text(" | ", strip=True).lower()
    html_text = str(page_html or "").lower()
    tennis_markers = [
        "tennis",
        "tournament titles",
        "career record",
        "trophies and finals",
        "trophäen",
        "karrierebilanz",
        "tennisspieler",
    ]
    return any(marker in info_text or marker in html_text for marker in tennis_markers)


def _wikipedia_infobox_birth_date(soup: BeautifulSoup) -> str:
    bday = soup.select_one(".bday")
    if bday is not None:
        return _safe_date_text(bday.get_text(strip=True))
    info_text = soup.select_one("table.infobox")
    if info_text is None:
        return ""
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", info_text.get_text(" | ", strip=True))
    if not match:
        return ""
    return _safe_date_text(match.group(1))


def _wikipedia_infobox_height_cm(soup: BeautifulSoup) -> Optional[int]:
    infobox = soup.select_one("table.infobox")
    if infobox is None:
        return None
    text = infobox.get_text(" | ", strip=True)
    match = re.search(r"Height\s*[:|]\s*[^|]*?(\d{3})\s*cm", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"Height\s*[:|]\s*[^|]*?(\d(?:\.\d+)?)\s*m\b", text, flags=re.IGNORECASE)
    if match:
        return int(round(float(match.group(1)) * 100))
    match = re.search(r"Height\s*[:|]\s*[^|]*?(\d+)\s*ft\s*(\d+)\s*in", text, flags=re.IGNORECASE)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        return int(round((feet * 12 + inches) * 2.54))
    return None


def _wikipedia_infobox_hand_text(soup: BeautifulSoup) -> str:
    infobox = soup.select_one("table.infobox")
    if infobox is None:
        return ""
    text = infobox.get_text(" | ", strip=True)
    patterns = [
        r"Plays\s*[:|]\s*([^|]+)",
        r"Spielhand\s*[:|]\s*([^|]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return " ".join(str(match.group(1) or "").split()).strip()
    return ""


def _hand_code_from_wikipedia_text(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if any(token in text for token in ["left", "links"]):
        return "L"
    if any(token in text for token in ["right", "rechts"]):
        return "R"
    return ""


def _wikipedia_wikibase_item_id(page_html: object) -> str:
    match = re.search(r'"wgWikibaseItemId"\s*:\s*"(Q\d+)"', str(page_html or ""))
    return str(match.group(1)) if match else ""


def _wikidata_entity(session: requests.Session, qid: object) -> dict[str, object]:
    qid_text = str(qid or "").strip()
    if not qid_text:
        return {}
    response = _get_with_retries(session, WIKIDATA_ENTITY_URL.format(qid=qid_text), timeout=60)
    if response.status_code == 404:
        return {}
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return {}
    entity = (payload.get("entities") or {}).get(qid_text)
    return entity if isinstance(entity, dict) else {}


def _wikidata_birth_date(entity: dict[str, object]) -> str:
    claims = list((entity.get("claims") or {}).get("P569") or [])
    if not claims:
        return ""
    value = ((claims[0].get("mainsnak") or {}).get("datavalue") or {}).get("value") or {}
    return _safe_date_text(str(value.get("time") or ""))


def _wikidata_height_cm(entity: dict[str, object]) -> Optional[int]:
    heights: list[int] = []
    for claim in list((entity.get("claims") or {}).get("P2048") or []):
        value = ((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value") or {}
        amount = value.get("amount")
        unit = str(value.get("unit") or "")
        if amount in [None, ""] or not unit.endswith("Q174728"):
            continue
        try:
            heights.append(int(round(float(amount))))
        except (TypeError, ValueError):
            continue
    unique_heights = sorted(set(heights))
    if len(unique_heights) == 1:
        return unique_heights[0]
    return None


def _tennisexplorer_candidate_name(display_name: object) -> str:
    text = str(display_name or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+\([A-Z]{2,3}\)\s*$", "", text).strip()
    if "," in text:
        last_name, first_name = [part.strip() for part in text.split(",", 1)]
        return " ".join(part for part in [first_name, last_name] if part).strip()
    return text


def _names_match_by_initial_and_surname(expected_name: object, actual_name: object) -> bool:
    expected_tokens = _normalize_text(expected_name).split()
    actual_tokens = _normalize_text(actual_name).split()
    if len(expected_tokens) < 2 or len(actual_tokens) < 2:
        return False
    expected_initial = expected_tokens[0][:1]
    actual_initial = actual_tokens[0][:1]
    suffix_tokens = {"jr", "sr", "ii", "iii", "iv", "v"}
    expected_surname_tokens = expected_tokens[1:]
    actual_surname_tokens = actual_tokens[1:]
    while expected_surname_tokens and expected_surname_tokens[-1] in suffix_tokens:
        expected_surname_tokens = expected_surname_tokens[:-1]
    while actual_surname_tokens and actual_surname_tokens[-1] in suffix_tokens:
        actual_surname_tokens = actual_surname_tokens[:-1]
    if not expected_initial or not actual_initial or not expected_surname_tokens:
        return False

    def suffix_matches(shorter: list[str], longer: list[str]) -> bool:
        return len(shorter) <= len(longer) and shorter == longer[-len(shorter) :]

    return expected_initial == actual_initial and (
        expected_surname_tokens == actual_surname_tokens
        or suffix_matches(expected_surname_tokens, actual_surname_tokens)
        or suffix_matches(actual_surname_tokens, expected_surname_tokens)
    )


def _tennisexplorer_birth_date(text: object) -> str:
    match = re.search(r"Age:\s*\d+\s*\((\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})\)", str(text or ""))
    if not match:
        return ""
    day = int(match.group(1))
    month = int(match.group(2))
    year = int(match.group(3))
    # Tennis Explorer sometimes falls back to 1 January when only a birth year is known.
    if day == 1 and month == 1:
        return ""
    return f"{year:04d}-{month:02d}-{day:02d}"


def _tennisexplorer_exact_date(text: object) -> str:
    match = re.search(r"(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})", str(text or ""))
    if not match:
        return ""
    day = int(match.group(1))
    month = int(match.group(2))
    year = int(match.group(3))
    try:
        return datetime(year, month, day).date().isoformat()
    except ValueError:
        return ""


def _tennisexplorer_height_cm(text: object) -> Optional[int]:
    match = re.search(r"Height(?:\s*/\s*Weight)?\s*:\s*(\d+)\s*cm", str(text or ""), flags=re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+)\s*cm\b", str(text or ""), flags=re.IGNORECASE)
        if not match:
            return None
    return int(match.group(1))


def _tennisexplorer_player_slug(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"/player/([^/?#]+)/?", text, flags=re.IGNORECASE)
    if match:
        return str(match.group(1) or "").strip().lower()
    return text.strip("/").split("?", 1)[0].split("#", 1)[0].strip().lower()


def _tennisexplorer_match_detail_urls(soup: BeautifulSoup, *, page_url: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for node in soup.select('a[href*="/match-detail/?id="]'):
        href = str(node.get("href") or "").strip()
        if not href:
            continue
        url = urljoin(page_url, href)
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
        if len(urls) >= TENNIS_EXPLORER_MATCH_DETAIL_MAX_LINKS:
            break
    return urls


def _tennisexplorer_match_detail_snapshot(html: str, *, player_slug: str) -> dict[str, object]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.result.gDetail.noMgB")
    if table is None:
        return {}

    header_links = table.select("thead th.plName a")
    if len(header_links) < 2:
        return {}

    player_side = ""
    match_player_name = ""
    for index, link in enumerate(header_links[:2]):
        link_slug = _tennisexplorer_player_slug(link.get("href"))
        if link_slug and link_slug == player_slug:
            player_side = "left" if index == 0 else "right"
            match_player_name = " ".join(link.get_text(" ", strip=True).split())
            break
    if not player_side:
        return {}

    detail_rows: dict[str, dict[str, str]] = {}
    for row in table.select("tbody tr"):
        label_node = row.find("th")
        if label_node is None:
            continue
        label = _normalize_text(label_node.get_text(" ", strip=True))
        if not label:
            continue
        left_node = row.select_one("td.tr")
        right_node = row.select_one("td.tl")
        detail_rows[label] = {
            "left": " ".join(left_node.get_text(" ", strip=True).split()) if left_node else "",
            "right": " ".join(right_node.get_text(" ", strip=True).split()) if right_node else "",
        }

    def side_value(label: str) -> str:
        return str((detail_rows.get(label) or {}).get(player_side) or "").strip()

    height_text = side_value("height")
    if not height_text:
        for label, values in detail_rows.items():
            if label.startswith("height"):
                height_text = str(values.get(player_side) or "").strip()
                if height_text:
                    break
    hand_description = side_value("plays")
    hand_code = _hand_code_from_description(hand_description)
    return {
        "match_player_name": match_player_name,
        "birth_date": _tennisexplorer_exact_date(side_value("birthdate")),
        "hand_code": hand_code,
        "hand_description": _hand_description_from_code(hand_code) or (hand_description.title() if hand_description else ""),
        "height_cm": _tennisexplorer_height_cm(height_text),
    }


def _tennisexplorer_match_detail_fallback(
    session: requests.Session,
    *,
    player_slug: str,
    page_url: str,
    page_soup: BeautifulSoup,
    known_birth_date: str,
) -> dict[str, object]:
    if not player_slug or not known_birth_date:
        return {}

    detail_urls = _tennisexplorer_match_detail_urls(page_soup, page_url=page_url)
    if not detail_urls:
        return {}

    detail_rows: list[dict[str, object]] = []
    detail_urls_checked = 0
    for detail_url in detail_urls:
        response = _get_with_retries(session, detail_url, timeout=60)
        if response.status_code == 404:
            continue
        response.raise_for_status()
        detail_urls_checked += 1

        snapshot = _tennisexplorer_match_detail_snapshot(response.text, player_slug=player_slug)
        if not snapshot:
            continue
        if snapshot.get("birth_date") != known_birth_date:
            continue

        snapshot["detail_url"] = detail_url
        detail_rows.append(snapshot)
        if len(detail_rows) >= TENNIS_EXPLORER_MATCH_DETAIL_MAX_EXACT_PAGES:
            break

    if not detail_rows:
        return {}

    detail_df = pd.DataFrame(detail_rows)
    height_cm = _unique_non_missing_value(detail_df["height_cm"], numeric=True)
    hand_code = _unique_non_missing_value(detail_df["hand_code"], allowed_values={"L", "R"}) or ""
    hand_description = _unique_non_missing_value(detail_df["hand_description"])
    if hand_code and not hand_description:
        hand_description = _hand_description_from_code(hand_code)

    source_page_url = ""
    for detail_row in detail_rows:
        if height_cm is not None and _safe_int(detail_row.get("height_cm")) == height_cm:
            source_page_url = str(detail_row.get("detail_url") or "").strip()
            break
        if hand_code and str(detail_row.get("hand_code") or "").strip() == hand_code:
            source_page_url = str(detail_row.get("detail_url") or "").strip()
            break
    if not source_page_url:
        source_page_url = str(detail_rows[0].get("detail_url") or "").strip()

    notes = [
        f"match_detail_pages_checked={detail_urls_checked}",
        f"exact_birthdate_pages={len(detail_rows)}",
    ]

    non_missing_heights = pd.to_numeric(detail_df["height_cm"], errors="coerce").dropna()
    if not non_missing_heights.empty:
        unique_heights = sorted({int(round(value)) for value in non_missing_heights.tolist()})
        if len(unique_heights) > 1:
            notes.append(
                "conflicting_match_detail_heights=" + ",".join(str(value) for value in unique_heights)
            )

    non_missing_hands = detail_df["hand_code"].fillna("").astype(str).str.strip()
    non_missing_hands = non_missing_hands[non_missing_hands.isin(["L", "R"])]
    unique_hands = sorted(set(non_missing_hands.tolist()))
    if len(unique_hands) > 1:
        notes.append("conflicting_match_detail_hands=" + ",".join(unique_hands))

    return {
        "birth_date": known_birth_date,
        "hand_code": hand_code,
        "hand_description": hand_description or "",
        "height_cm": height_cm,
        "source_page_url": source_page_url,
        "source_notes": "; ".join(notes),
    }


def fetch_tennisexplorer_player_profile(
    tour: object,
    player_id: object,
    player_name: object,
    *,
    known_birth_date: object = None,
    session: Optional[requests.Session] = None,
) -> dict[str, object]:
    session = _request_session(session)
    normalized_expected = _normalize_text(player_name)
    known_birth = _safe_date_text(known_birth_date)
    response = session.get(
        TENNIS_EXPLORER_SEARCH_URL,
        params={"s": str(player_name or "").strip(), "t": "p", "all": 1},
        timeout=60,
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    response.raise_for_status()
    payload = response.json()
    links = list(payload.get("links") or [])

    matched_link: Optional[dict[str, object]] = None
    matched_name = ""
    for link in links:
        candidate_name = _tennisexplorer_candidate_name(link.get("name"))
        if _names_match(player_name, candidate_name) or _names_match_by_initial_and_surname(player_name, candidate_name):
            matched_link = link
            matched_name = candidate_name
            break

    if matched_link is not None:
        candidate_links = [matched_link]
    elif known_birth:
        candidate_links = list(links)
    else:
        candidate_links = []
    for link in candidate_links:
        if not isinstance(link, dict):
            continue

        slug = str(link.get("url") or "").strip()
        if not slug:
            continue

        candidate_name = _tennisexplorer_candidate_name(link.get("name"))
        page_url = TENNIS_EXPLORER_PLAYER_URL.format(slug=slug)
        page_response = _get_with_retries(session, page_url, timeout=60)
        page_response.raise_for_status()
        page_soup = BeautifulSoup(page_response.text, "html.parser")
        page_text = page_soup.get_text("\n", strip=True)

        birth_date = _tennisexplorer_birth_date(page_text)
        if known_birth and birth_date and birth_date != known_birth:
            continue

        hand_description_raw = ""
        hand_match = re.search(r"Plays:\s*(left|right)", page_text, flags=re.IGNORECASE)
        if hand_match:
            hand_description_raw = f"{hand_match.group(1).strip().title()}-Handed"

        row = _blank_profile_row(
            str(tour or "").lower(),
            player_id,
            source_kind="tennisexplorer_profile",
            fetch_status="partial",
        )
        row.update(
            {
                "full_name": candidate_name or str(player_name or "").strip(),
                "birth_date": birth_date,
                "hand_code": _hand_code_from_description(hand_description_raw),
                "hand_description": hand_description_raw,
                "height_cm": _tennisexplorer_height_cm(page_text),
                "profile_url": page_url,
                "source_api_url": TENNIS_EXPLORER_SEARCH_URL,
                "source_page_url": page_url,
                "source_notes": f"query={player_name!r}",
            }
        )
        if row.get("birth_date") or row.get("hand_code") or row.get("height_cm") not in ["", None]:
            row["fetch_status"] = "ok"
        else:
            row["fetch_status"] = "not_found"
            row["error"] = "Tennis Explorer profile did not expose exact birth date, hand, or height."

        safely_name_matched = _names_match(player_name, candidate_name) or _names_match_by_initial_and_surname(
            player_name,
            candidate_name,
        )
        if known_birth and safely_name_matched and (
            _safe_int(row.get("height_cm")) is None or not str(row.get("hand_code") or "").strip()
        ):
            detail_row = _tennisexplorer_match_detail_fallback(
                session,
                player_slug=_tennisexplorer_player_slug(slug),
                page_url=page_url,
                page_soup=page_soup,
                known_birth_date=known_birth,
            )
            if detail_row:
                if not row.get("birth_date") and detail_row.get("birth_date"):
                    row["birth_date"] = detail_row.get("birth_date")
                if _safe_int(row.get("height_cm")) is None and _safe_int(detail_row.get("height_cm")) is not None:
                    row["height_cm"] = _safe_int(detail_row.get("height_cm"))
                if not str(row.get("hand_code") or "").strip() and str(detail_row.get("hand_code") or "").strip():
                    row["hand_code"] = str(detail_row.get("hand_code") or "").strip()
                    row["hand_description"] = str(detail_row.get("hand_description") or "").strip()
                if str(detail_row.get("source_page_url") or "").strip():
                    row["source_page_url"] = str(detail_row.get("source_page_url") or "").strip()
                detail_notes = str(detail_row.get("source_notes") or "").strip()
                if detail_notes:
                    row["source_notes"] = f"{row['source_notes']}; {detail_notes}"
                if row.get("birth_date") or row.get("hand_code") or _safe_int(row.get("height_cm")) is not None:
                    row["fetch_status"] = "ok"
                    row["error"] = ""
        if known_birth and birth_date == known_birth and not _names_match(player_name, candidate_name):
            row["source_notes"] = f"{row['source_notes']}; accepted_via_birth_date_match"
        if normalized_expected and candidate_name and not safely_name_matched:
            row["fetch_status"] = "name_mismatch"
            row["error"] = f"Tennis Explorer profile name mismatch: expected {player_name!r}, got {candidate_name!r}."
        return row

    row = _blank_profile_row(
        str(tour or "").lower(),
        player_id,
        source_kind="tennisexplorer_profile",
        fetch_status="not_found",
        error="Tennis Explorer search did not return a safely matchable profile.",
    )
    row["full_name"] = str(player_name or "").strip()
    row["source_api_url"] = TENNIS_EXPLORER_SEARCH_URL
    row["source_notes"] = f"query={player_name!r}"
    return row


def _wikipedia_search_titles(session: requests.Session, language: str, player_name: object) -> list[str]:
    response = _get_with_retries(
        session,
        WIKIPEDIA_SEARCH_API_URL.format(language=language),
        timeout=60,
        params={
            "action": "query",
            "list": "search",
            "srsearch": str(player_name or "").strip(),
            "format": "json",
            "srlimit": 5,
        },
    )
    if response.status_code == 404:
        return []
    response.raise_for_status()
    payload = response.json()
    search_rows = ((payload or {}).get("query") or {}).get("search") or []
    return [str(item.get("title") or "").strip() for item in search_rows if str(item.get("title") or "").strip()]


def fetch_wikipedia_player_profile(
    tour: object,
    player_id: object,
    player_name: object,
    *,
    known_birth_date: object = None,
    session: Optional[requests.Session] = None,
) -> dict[str, object]:
    session = _request_session(session)
    row = _blank_profile_row(
        str(tour or "").lower(),
        player_id,
        source_kind="wikipedia_page_plus_wikidata",
        fetch_status="not_found",
    )
    row["full_name"] = str(player_name or "").strip()

    known_birth = _safe_date_text(known_birth_date)
    if not known_birth:
        row["error"] = "Known birth date unavailable for safe Wikipedia identity matching."
        return row

    for language in WIKIPEDIA_LANGUAGES:
        candidate_slugs = [_wikipedia_page_slug(player_name).replace("-", "_")]
        for title in _wikipedia_search_titles(session, language, player_name):
            candidate_slug = quote(str(title).replace(" ", "_"))
            if candidate_slug not in candidate_slugs:
                candidate_slugs.append(candidate_slug)

        for slug in candidate_slugs:
            page_url = WIKIPEDIA_PAGE_URL.format(language=language, slug=slug)
            page_response = _get_with_retries(session, page_url, timeout=60)
            if page_response.status_code == 404 or not _wikipedia_has_exact_article(page_response.text):
                continue

            soup = BeautifulSoup(page_response.text, "html.parser")
            if not _wikipedia_is_tennis_page(page_response.text, soup):
                continue

            qid = _wikipedia_wikibase_item_id(page_response.text)
            entity = _wikidata_entity(session, qid) if qid else {}
            page_birth_date = _wikipedia_infobox_birth_date(soup)
            entity_birth_date = _wikidata_birth_date(entity)
            matched_birth_date = _safe_date_text(page_birth_date or entity_birth_date or known_birth)
            if matched_birth_date and matched_birth_date != known_birth:
                continue

            hand_text = _wikipedia_infobox_hand_text(soup)
            hand_code = _hand_code_from_wikipedia_text(hand_text)
            height_cm = _wikipedia_infobox_height_cm(soup)
            if height_cm is None:
                height_cm = _wikidata_height_cm(entity)

            if not hand_code and height_cm is None:
                continue

            row.update(
                {
                    "birth_date": known_birth,
                    "hand_code": hand_code,
                    "hand_description": _hand_description_from_code(hand_code) or hand_text,
                    "height_cm": height_cm,
                    "profile_url": page_response.url,
                    "source_api_url": WIKIDATA_ENTITY_URL.format(qid=qid) if qid else "",
                    "source_page_url": page_response.url,
                    "fetch_status": "ok",
                    "source_notes": f"lang={language}; qid={qid or ''}",
                }
            )
            return row

    row["error"] = "Wikipedia pages did not expose safe hand or height values for this player."
    return row


def _load_profile_cache(tour: str) -> pd.DataFrame:
    path = _profile_cache_path(tour)
    if not path.exists():
        return pd.DataFrame(columns=PLAYER_PROFILE_COLUMNS)
    frame = pd.read_csv(path, low_memory=False)
    frame = _ensure_profile_columns(frame)
    frame["tour"] = frame["tour"].astype(str).str.lower()
    frame["player_id"] = frame["player_id"].map(lambda value: canonical_player_id(tour, value))
    return frame[PLAYER_PROFILE_COLUMNS].copy()


def _write_profile_cache(tour: str, rows_by_id: dict[str, dict[str, object]]) -> Path:
    _ensure_profile_dirs()
    path = _profile_cache_path(tour)
    frame = pd.DataFrame(list(rows_by_id.values()), columns=PLAYER_PROFILE_COLUMNS)
    if not frame.empty:
        frame["tour"] = frame["tour"].astype(str).str.lower()
        frame["player_id"] = frame["player_id"].map(lambda value: canonical_player_id(tour, value))
        frame = frame.sort_values(["tour", "player_id"]).reset_index(drop=True)
    frame.to_csv(path, index=False)
    return path


def load_tennis_player_profiles(
    tours: tuple[str, ...] | list[str] = ("atp", "wta"),
    include_failed: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for tour in tours:
        frame = _load_profile_cache(str(tour).lower())
        if frame.empty:
            continue
        frames.append(frame)
    supplement = _load_profile_supplement()
    if not supplement.empty:
        supplement = supplement[supplement["tour"].isin([str(tour).lower() for tour in tours])].copy()
        if not supplement.empty:
            frames.append(supplement)

    if not frames:
        return pd.DataFrame(columns=PLAYER_PROFILE_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = _ensure_profile_columns(combined)
    combined["tour"] = combined["tour"].astype(str).str.lower()
    combined["player_id"] = combined.apply(lambda row: canonical_player_id(row.get("tour"), row.get("player_id")), axis=1)
    if not include_failed and "fetch_status" in combined.columns:
        combined = combined[combined["fetch_status"].isin(["ok", "partial"])].copy()
    return combined.reset_index(drop=True)


def collect_tennis_player_profile_targets(
    matches_df: pd.DataFrame,
    missing_only: bool = True,
    official_window_only: bool = False,
    official_start_year: int = 2025,
) -> pd.DataFrame:
    empty = pd.DataFrame(
        columns=[
            "tour",
            "player_id",
            "player_name",
            "needs_age",
            "needs_hand",
            "needs_height",
            "affected_rows",
            "latest_event_date",
            "missing_fields",
        ]
    )
    if matches_df.empty:
        return empty

    frames: list[pd.DataFrame] = []
    for prefix in ["player_a", "player_b"]:
        required_columns = ["tour", "event_date", f"{prefix}_id", prefix, f"{prefix}_age", f"{prefix}_hand", f"{prefix}_height_cm"]
        missing_columns = [column for column in required_columns if column not in matches_df.columns]
        if missing_columns:
            raise ValueError(
                "Tennis match frame is missing required columns for player profile targeting: "
                + ", ".join(missing_columns)
            )

        side = matches_df[required_columns].copy()
        side = side.rename(
            columns={
                f"{prefix}_id": "player_id",
                prefix: "player_name",
                f"{prefix}_age": "player_age",
                f"{prefix}_hand": "player_hand",
                f"{prefix}_height_cm": "player_height_cm",
            }
        )
        side["event_date"] = pd.to_datetime(side["event_date"], errors="coerce")
        if official_window_only:
            side = side[side["event_date"].dt.year >= int(official_start_year)].copy()
        side["tour"] = side["tour"].astype(str).str.lower()
        side["player_id"] = side.apply(lambda row: canonical_player_id(row["tour"], row["player_id"]), axis=1)
        side["player_name"] = side["player_name"].fillna("").astype(str).str.strip()
        side["needs_age"] = _missing_mask(side["player_age"])
        side["needs_hand"] = _missing_mask(side["player_hand"])
        side["needs_height"] = _missing_mask(side["player_height_cm"])
        if missing_only:
            side = side[side[["needs_age", "needs_hand", "needs_height"]].any(axis=1)].copy()
        side = side[side["player_id"].astype(str).str.len() > 0].copy()
        side = side[side["player_name"].astype(str).str.len() > 0].copy()
        frames.append(side[["tour", "player_id", "player_name", "event_date", "needs_age", "needs_hand", "needs_height"]])

    if not frames:
        return empty

    combined = pd.concat(frames, ignore_index=True)
    preferred_names = (
        combined.assign(name_len=combined["player_name"].astype(str).str.len())
        .sort_values(["tour", "player_id", "name_len"], ascending=[True, True, False])
        .drop_duplicates(subset=["tour", "player_id"])[["tour", "player_id", "player_name"]]
    )
    aggregated = (
        combined.groupby(["tour", "player_id"], as_index=False)
        .agg(
            needs_age=("needs_age", "max"),
            needs_hand=("needs_hand", "max"),
            needs_height=("needs_height", "max"),
            affected_rows=("player_id", "size"),
            latest_event_date=("event_date", "max"),
        )
    )
    targets = preferred_names.merge(aggregated, on=["tour", "player_id"], how="inner")
    targets["latest_event_date"] = pd.to_datetime(targets["latest_event_date"], errors="coerce").dt.date.astype(str)
    targets.loc[targets["latest_event_date"].eq("NaT"), "latest_event_date"] = ""
    targets["missing_fields"] = targets.apply(
        lambda row: ",".join(
            field
            for field, needed in [
                ("age", bool(row["needs_age"])),
                ("hand", bool(row["needs_hand"])),
                ("height_cm", bool(row["needs_height"])),
            ]
            if needed
        ),
        axis=1,
    )
    return targets.sort_values(["tour", "player_name", "player_id"]).reset_index(drop=True)


def collect_live_tennis_player_profile_seed_targets(
    matchups_df: pd.DataFrame,
    *,
    profiles_df: Optional[pd.DataFrame] = None,
    missing_only: bool = True,
) -> pd.DataFrame:
    empty = pd.DataFrame(
        columns=[
            "tour",
            "player_id",
            "player_name",
            "needs_age",
            "needs_hand",
            "needs_height",
            "affected_rows",
            "latest_event_date",
            "missing_fields",
        ]
    )
    if matchups_df.empty:
        return empty

    required_columns = ["tour", "fighter_a", "fighter_b"]
    missing_columns = [column for column in required_columns if column not in matchups_df.columns]
    if missing_columns:
        raise ValueError(
            "Tennis live matchup frame is missing required columns for player profile seeding: "
            + ", ".join(missing_columns)
        )

    frames: list[pd.DataFrame] = []
    for column in ["fighter_a", "fighter_b"]:
        side = matchups_df[["tour", column]].copy()
        side = side.rename(columns={column: "player_name"})
        if "commence_time" in matchups_df.columns:
            side["event_date"] = pd.to_datetime(matchups_df["commence_time"], errors="coerce")
        else:
            side["event_date"] = pd.NaT
        side["tour"] = side["tour"].astype(str).str.lower()
        side["player_name"] = side["player_name"].fillna("").astype(str).str.strip()
        side = side[side["player_name"].astype(str).str.len() > 0].copy()
        side["player_id"] = ""
        frames.append(side[["tour", "player_id", "player_name", "event_date"]])

    if not frames:
        return empty

    combined = pd.concat(frames, ignore_index=True)
    combined["profile_name_key"] = combined["player_name"].map(_normalized_name_key)
    combined = combined[combined["profile_name_key"].astype(str).str.len() > 0].copy()
    if combined.empty:
        return empty

    preferred_names = (
        combined.assign(name_len=combined["player_name"].astype(str).str.len())
        .sort_values(["tour", "profile_name_key", "name_len"], ascending=[True, True, False])
        .drop_duplicates(subset=["tour", "profile_name_key"])[["tour", "profile_name_key", "player_name"]]
    )
    aggregated = (
        combined.groupby(["tour", "profile_name_key"], as_index=False)
        .agg(
            affected_rows=("player_name", "size"),
            latest_event_date=("event_date", "max"),
        )
    )
    targets = preferred_names.merge(aggregated, on=["tour", "profile_name_key"], how="inner")

    profiles = profiles_df.copy() if profiles_df is not None else load_tennis_player_profiles()
    lookup = _build_profile_name_lookup(profiles, key_builder=_normalized_name_key)
    if lookup.empty:
        targets["needs_age"] = True
        targets["needs_hand"] = True
        targets["needs_height"] = True
    else:
        lookup = lookup.rename(columns={"profile_name_key": "profile_name_key_lookup"})
        targets = targets.merge(
            lookup.rename(
                columns={
                    "profile_name_key_lookup": "profile_name_key",
                    "birth_date": "profile_birth_date",
                    "hand_code": "profile_hand_code",
                    "height_cm": "profile_height_cm",
                }
            )[["tour", "profile_name_key", "profile_birth_date", "profile_hand_code", "profile_height_cm"]],
            on=["tour", "profile_name_key"],
            how="left",
        )
        targets["needs_age"] = _missing_mask(targets["profile_birth_date"])
        targets["needs_hand"] = _missing_mask(targets["profile_hand_code"])
        targets["needs_height"] = _missing_mask(targets["profile_height_cm"])
        targets = targets.drop(columns=["profile_birth_date", "profile_hand_code", "profile_height_cm"])

    if missing_only:
        targets = targets[targets[["needs_age", "needs_hand", "needs_height"]].any(axis=1)].copy()
    if targets.empty:
        return empty

    targets["player_id"] = ""
    targets["latest_event_date"] = pd.to_datetime(targets["latest_event_date"], errors="coerce").dt.date.astype(str)
    targets.loc[targets["latest_event_date"].eq("NaT"), "latest_event_date"] = ""
    targets["missing_fields"] = targets.apply(
        lambda row: ",".join(
            field
            for field, needed in [
                ("age", bool(row["needs_age"])),
                ("hand", bool(row["needs_hand"])),
                ("height_cm", bool(row["needs_height"])),
            ]
            if needed
        ),
        axis=1,
    )
    return targets[
        [
            "tour",
            "player_id",
            "player_name",
            "needs_age",
            "needs_hand",
            "needs_height",
            "affected_rows",
            "latest_event_date",
            "missing_fields",
        ]
    ].sort_values(["tour", "player_name"]).reset_index(drop=True)


def write_tennis_player_profile_targets(
    targets_df: pd.DataFrame,
    path: Optional[Path] = None,
) -> Path:
    _ensure_profile_dirs()
    output_path = path or TENNIS_PROFILE_TARGETS_PATH
    targets_df.to_csv(output_path, index=False)
    return output_path


def remaining_tennis_player_profile_targets(
    targets_df: pd.DataFrame,
    profiles_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if targets_df.empty:
        return targets_df.copy()

    profiles = profiles_df.copy() if profiles_df is not None else load_tennis_player_profiles()
    if profiles.empty:
        return targets_df.copy()

    ok_profiles = profiles.copy()
    ok_profiles["tour"] = ok_profiles["tour"].astype(str).str.lower()
    ok_profiles["player_id"] = ok_profiles.apply(
        lambda row: canonical_player_id(row.get("tour"), row.get("player_id")),
        axis=1,
    )
    ok_profiles = ok_profiles[["tour", "player_id"]].drop_duplicates()

    remaining = targets_df.copy()
    remaining["tour"] = remaining["tour"].astype(str).str.lower()
    remaining["player_id"] = remaining.apply(
        lambda row: canonical_player_id(row.get("tour"), row.get("player_id")),
        axis=1,
    )
    remaining = remaining.merge(ok_profiles.assign(_matched=1), on=["tour", "player_id"], how="left")
    remaining = remaining[remaining["_matched"].isna()].drop(columns=["_matched"])
    return remaining.reset_index(drop=True)


def write_tennis_player_profile_remaining_targets(
    targets_df: pd.DataFrame,
    profiles_df: Optional[pd.DataFrame] = None,
    path: Optional[Path] = None,
    already_filtered: bool = False,
) -> Path:
    _ensure_profile_dirs()
    output_path = path or TENNIS_PROFILE_REMAINING_TARGETS_PATH
    remaining = targets_df.copy() if already_filtered else remaining_tennis_player_profile_targets(targets_df, profiles_df=profiles_df)
    remaining.to_csv(output_path, index=False)
    return output_path


def download_tennis_player_profiles(
    players_df: pd.DataFrame,
    force: bool = False,
    session: Optional[requests.Session] = None,
) -> list[Path]:
    if players_df.empty:
        return []

    _ensure_profile_dirs()
    session = _request_session(session)
    saved_paths: list[Path] = []

    for tour in sorted(players_df["tour"].astype(str).str.lower().unique()):
        cache_df = _load_profile_cache(tour)
        rows_by_id = {
            canonical_player_id(tour, row.get("player_id")): {column: row.get(column, "") for column in PLAYER_PROFILE_COLUMNS}
            for row in cache_df.to_dict("records")
            if canonical_player_id(tour, row.get("player_id"))
        }
        tour_targets = players_df.loc[players_df["tour"].astype(str).str.lower() == tour, ["player_id", "player_name"]].copy()
        tour_targets["player_id"] = tour_targets["player_id"].map(lambda value: canonical_player_id(tour, value))
        tour_targets = tour_targets.drop_duplicates(subset=["player_id"]).reset_index(drop=True)

        fetch_fn = fetch_atp_player_profile if tour == "atp" else fetch_wta_player_profile
        for target in tour_targets.to_dict("records"):
            player_id = str(target.get("player_id") or "").strip()
            player_name = target.get("player_name")
            if not player_id:
                continue

            existing = rows_by_id.get(player_id)
            if not force and existing is not None and not _should_retry_cached_profile_attempt(
                existing,
                target=target,
            ):
                continue

            try:
                row = fetch_fn(player_id, session=session, expected_name=player_name)
            except requests.RequestException as exc:
                logger.warning("Failed to fetch official %s player profile %s: %s", tour.upper(), player_id, exc)
                continue

            row = _stamp_profile_observation_metadata(row, target=target)
            rows_by_id[player_id] = {column: row.get(column, "") for column in PLAYER_PROFILE_COLUMNS}
            path = _write_profile_cache(tour, rows_by_id)
            if path not in saved_paths:
                saved_paths.append(path)
            time.sleep(PLAYER_PROFILE_REQUEST_PAUSE_SECONDS)

        if rows_by_id:
            path = _write_profile_cache(tour, rows_by_id)
            if path not in saved_paths:
                saved_paths.append(path)

    return saved_paths


def download_secondary_tennis_player_profiles(
    players_df: pd.DataFrame,
    force: bool = False,
    session: Optional[requests.Session] = None,
) -> list[Path]:
    if players_df.empty:
        return []

    _ensure_profile_dirs()
    session = _request_session(session)
    supplement = _load_profile_supplement()
    rows = supplement.to_dict("records")
    saved_paths: list[Path] = []

    existing_rows_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        status = str(row.get("fetch_status") or "").strip()
        if status not in {"ok", "partial", "not_found", "name_mismatch"}:
            continue
        full_name = row.get("full_name") or row.get("player_name")
        row_key = _profile_attempt_key(
            row.get("tour"),
            row.get("player_id"),
            full_name,
            row.get("source_kind"),
        )
        if not row_key[1]:
            continue
        existing = existing_rows_by_key.get(row_key)
        if existing is None or _safe_timestamp_utc(row.get("fetched_at_utc")) >= _safe_timestamp_utc(existing.get("fetched_at_utc")):
            existing_rows_by_key[row_key] = row
    profiles_for_secondary = load_tennis_player_profiles(include_failed=True)

    for target in players_df.to_dict("records"):
        tour = str(target.get("tour") or "").lower()
        player_id = canonical_player_id(tour, target.get("player_id"))
        player_name = target.get("player_name")
        row_key = _profile_attempt_key(tour, player_id, player_name, "tennisexplorer_profile")
        existing = existing_rows_by_key.get(row_key)
        if not force and existing is not None and not _should_retry_cached_profile_attempt(existing, target=target):
            continue
        known_birth_date = _known_profile_birth_date(
            profiles_for_secondary,
            tour=tour,
            player_id=player_id,
            player_name=player_name,
        )
        try:
            row = fetch_tennisexplorer_player_profile(
                tour=tour,
                player_id=player_id,
                player_name=player_name,
                known_birth_date=known_birth_date,
                session=session,
            )
        except requests.RequestException as exc:
            logger.warning("Failed to fetch Tennis Explorer profile for %s %s: %s", tour.upper(), player_name, exc)
            continue
        row = _stamp_profile_observation_metadata(row, target=target)
        rows.append({column: row.get(column, "") for column in PLAYER_PROFILE_COLUMNS})
        existing_rows_by_key[row_key] = row
        path = _write_profile_supplement(pd.DataFrame(rows, columns=PLAYER_PROFILE_COLUMNS))
        if path not in saved_paths:
            saved_paths.append(path)
        time.sleep(PLAYER_PROFILE_REQUEST_PAUSE_SECONDS)

    profiles_for_wikipedia = load_tennis_player_profiles(include_failed=True)
    for target in players_df.to_dict("records"):
        tour = str(target.get("tour") or "").lower()
        player_id = canonical_player_id(tour, target.get("player_id"))
        player_name = target.get("player_name")
        row_key = _profile_attempt_key(tour, player_id, player_name, "wikipedia_page_plus_wikidata")
        existing = existing_rows_by_key.get(row_key)
        if not force and existing is not None and not _should_retry_cached_profile_attempt(existing, target=target):
            continue

        known_birth_date = _known_profile_birth_date(
            profiles_for_wikipedia,
            tour=tour,
            player_id=player_id,
            player_name=player_name,
        )
        try:
            row = fetch_wikipedia_player_profile(
                tour=tour,
                player_id=player_id,
                player_name=player_name,
                known_birth_date=known_birth_date,
                session=session,
            )
        except requests.RequestException as exc:
            logger.warning("Failed to fetch Wikipedia profile for %s %s: %s", tour.upper(), player_name, exc)
            continue
        row = _stamp_profile_observation_metadata(row, target=target)
        rows.append({column: row.get(column, "") for column in PLAYER_PROFILE_COLUMNS})
        existing_rows_by_key[row_key] = row
        path = _write_profile_supplement(pd.DataFrame(rows, columns=PLAYER_PROFILE_COLUMNS))
        if path not in saved_paths:
            saved_paths.append(path)
        time.sleep(PLAYER_PROFILE_REQUEST_PAUSE_SECONDS)

    if rows:
        path = _write_profile_supplement(pd.DataFrame(rows, columns=PLAYER_PROFILE_COLUMNS))
        if path not in saved_paths:
            saved_paths.append(path)

    return saved_paths


def _unique_non_missing_value(series: pd.Series, *, numeric: bool = False, allowed_values: Optional[set[str]] = None):
    if numeric:
        values = pd.to_numeric(series, errors="coerce").dropna()
        unique_values = sorted(set(values.round(3).tolist()))
        if len(unique_values) == 1:
            value = unique_values[0]
            return int(value) if float(value).is_integer() else float(value)
        return None

    values = series.fillna("").astype(str).str.strip()
    values = values[values != ""]
    if allowed_values is not None:
        values = values[values.isin(sorted(allowed_values))]
    unique_values = sorted(set(values.tolist()))
    if len(unique_values) == 1:
        return unique_values[0]
    return None


def _profile_source_priority(source_kind: object) -> int:
    kind = str(source_kind or "").strip().lower()
    if kind in {
        "atp_official_hero_json",
        "wta_official_api_plus_profile_page",
        "manual_verified_atp_media_guide",
        "manual_verified_atp_official_alias",
        "manual_verified_wta_official_alias",
        "manual_verified_wta_match_notes_pdf",
        "manual_verified_wta_search_snippet",
        "manual_verified_official_event_entry",
        "manual_verified_official_college_roster",
        "manual_verified_official_college_alias",
    }:
        return 300
    if kind in {
        "manual_verified_espn_player_page",
        "manual_verified_espn_search_snippet",
        "manual_verified_tennisboard_player_page",
        "manual_verified_historical_raw_match_archive",
    }:
        return 250
    if kind in {
        "manual_verified_wikipedia_page",
        "manual_verified_public_consensus",
        "wikipedia_page_plus_wikidata",
    }:
        return 200
    if kind in {
        "tennisexplorer_profile",
        "tennisexplorer_profile_manual_seed",
    }:
        return 100
    return 0


def _preferred_profile_value(
    group: pd.DataFrame,
    column: str,
    *,
    numeric: bool = False,
    allowed_values: Optional[set[str]] = None,
):
    if column not in group.columns:
        return None
    working = group[[column, "source_kind"]].copy()
    working["_priority"] = working["source_kind"].map(_profile_source_priority)
    for priority in sorted(set(working["_priority"].tolist()), reverse=True):
        tier = working[working["_priority"] == priority]
        value = _unique_non_missing_value(tier[column], numeric=numeric, allowed_values=allowed_values)
        if value is not None:
            return value
        if numeric:
            tier_values = pd.to_numeric(tier[column], errors="coerce").dropna()
            if not tier_values.empty:
                return None
            continue
        tier_values = tier[column].fillna("").astype(str).str.strip()
        if allowed_values is not None:
            tier_values = tier_values[tier_values.isin(sorted(allowed_values))]
        else:
            tier_values = tier_values[tier_values != ""]
        if not tier_values.empty:
            return None
    return None


def _build_profile_name_lookup(
    profiles_df: pd.DataFrame,
    *,
    key_builder,
) -> pd.DataFrame:
    if profiles_df.empty:
        return pd.DataFrame(
            columns=[
                "tour",
                "profile_name_key",
                "birth_date",
                "birth_place",
                "country_code",
                "country_name",
                "hand_code",
                "hand_description",
                "backhand_code",
                "backhand_description",
                "height_cm",
                "weight_kg",
            ]
        )

    profiles = _ensure_profile_columns(profiles_df)
    profiles = profiles.copy()
    profiles["tour"] = profiles["tour"].astype(str).str.lower()
    profiles["profile_name_key"] = profiles["full_name"].map(key_builder)
    profiles = profiles[profiles["profile_name_key"].astype(str).str.len() > 0].copy()
    if profiles.empty:
        return pd.DataFrame(columns=["tour", "profile_name_key"])

    rows: list[dict[str, object]] = []
    for (tour, key), group in profiles.groupby(["tour", "profile_name_key"], dropna=False):
        row = {
            "tour": tour,
            "profile_name_key": key,
            "birth_date": _preferred_profile_value(group, "birth_date"),
            "birth_place": _preferred_profile_value(group, "birth_place"),
            "country_code": _preferred_profile_value(group, "country_code"),
            "country_name": _preferred_profile_value(group, "country_name"),
            "hand_code": _preferred_profile_value(group, "hand_code", allowed_values={"L", "R"}),
            "hand_description": _preferred_profile_value(group, "hand_description"),
            "backhand_code": _preferred_profile_value(group, "backhand_code"),
            "backhand_description": _preferred_profile_value(group, "backhand_description"),
            "height_cm": _preferred_profile_value(group, "height_cm", numeric=True),
            "weight_kg": _preferred_profile_value(group, "weight_kg", numeric=True),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _build_profile_id_lookup(profiles_df: pd.DataFrame) -> pd.DataFrame:
    if profiles_df.empty:
        return pd.DataFrame(
            columns=[
                "tour",
                "player_id_key",
                "birth_date",
                "birth_place",
                "country_code",
                "country_name",
                "hand_code",
                "hand_description",
                "backhand_code",
                "backhand_description",
                "height_cm",
                "weight_kg",
            ]
        )

    profiles = _ensure_profile_columns(profiles_df)
    profiles = profiles.copy()
    profiles["tour"] = profiles["tour"].astype(str).str.lower()
    profiles["player_id_key"] = profiles.apply(lambda row: canonical_player_id(row.get("tour"), row.get("player_id")), axis=1)
    profiles = profiles[profiles["player_id_key"].astype(str).str.len() > 0].copy()
    rows: list[dict[str, object]] = []
    for (tour, player_id_key), group in profiles.groupby(["tour", "player_id_key"], dropna=False):
        rows.append(
            {
                "tour": tour,
                "player_id_key": player_id_key,
                "birth_date": _preferred_profile_value(group, "birth_date"),
                "birth_place": _preferred_profile_value(group, "birth_place"),
                "country_code": _preferred_profile_value(group, "country_code"),
                "country_name": _preferred_profile_value(group, "country_name"),
                "hand_code": _preferred_profile_value(group, "hand_code", allowed_values={"L", "R"}),
                "hand_description": _preferred_profile_value(group, "hand_description"),
                "backhand_code": _preferred_profile_value(group, "backhand_code"),
                "backhand_description": _preferred_profile_value(group, "backhand_description"),
                "height_cm": _preferred_profile_value(group, "height_cm", numeric=True),
                "weight_kg": _preferred_profile_value(group, "weight_kg", numeric=True),
            }
        )
    return pd.DataFrame(rows)


def _apply_profile_name_backfill(
    working_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    *,
    key_builder,
    abbreviated_only: bool = False,
) -> pd.DataFrame:
    lookup = _build_profile_name_lookup(profiles_df, key_builder=key_builder)
    if lookup.empty:
        return working_df

    working = working_df.copy()
    for prefix in ["player_a", "player_b"]:
        key_column = f"{prefix}_profile_name_key"
        working[key_column] = working[prefix].map(key_builder)
        renamed = lookup.rename(
            columns={
                "profile_name_key": key_column,
                "birth_date": f"{prefix}_alias_birth_date",
                "birth_place": f"{prefix}_alias_birth_place",
                "country_code": f"{prefix}_alias_country_code",
                "country_name": f"{prefix}_alias_country_name",
                "hand_code": f"{prefix}_alias_hand_code",
                "hand_description": f"{prefix}_alias_hand_description",
                "backhand_code": f"{prefix}_alias_backhand_code",
                "backhand_description": f"{prefix}_alias_backhand_description",
                "height_cm": f"{prefix}_alias_height_cm",
                "weight_kg": f"{prefix}_alias_weight_kg",
            }
        )
        working = working.merge(renamed, on=["tour", key_column], how="left")

        eligible_mask = pd.Series(True, index=working.index)
        if abbreviated_only:
            eligible_mask = working[prefix].map(_match_has_initial_name).fillna(False)

        def fill_alias(destination: str, source_column: str) -> None:
            if source_column not in working.columns:
                return
            if destination not in working.columns:
                working[destination] = pd.NA
            mask = eligible_mask & _missing_mask(working[destination])
            if not mask.any():
                return
            values = working.loc[mask, source_column]
            if pd.api.types.is_numeric_dtype(working[destination]) or pd.api.types.is_numeric_dtype(values):
                working.loc[mask, destination] = pd.to_numeric(values, errors="coerce")
            else:
                working.loc[mask, destination] = values.astype(object)

        fill_alias(f"{prefix}_birth_date", f"{prefix}_alias_birth_date")
        fill_alias(f"{prefix}_birth_place", f"{prefix}_alias_birth_place")
        fill_alias(f"{prefix}_country_code", f"{prefix}_alias_country_code")
        fill_alias(f"{prefix}_country_name", f"{prefix}_alias_country_name")
        fill_alias(f"{prefix}_backhand", f"{prefix}_alias_backhand_description")
        fill_alias(f"{prefix}_weight_kg", f"{prefix}_alias_weight_kg")
        fill_alias(f"{prefix}_hand", f"{prefix}_alias_hand_code")
        fill_alias(f"{prefix}_height_cm", f"{prefix}_alias_height_cm")

        derived_age = _derive_exact_age_from_birth_date(working["event_date"], working[f"{prefix}_birth_date"])
        _fill_missing_column(
            working,
            f"{prefix}_age",
            derived_age.where(eligible_mask, pd.NA),
        )

        alias_columns = [column for column in working.columns if column.startswith(f"{prefix}_alias_")]
        working = working.drop(columns=alias_columns + [key_column])
    return working


def _build_historical_static_backfill_lookup(matches_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for prefix in ["player_a", "player_b"]:
        required_columns = [prefix, f"{prefix}_hand", f"{prefix}_height_cm"]
        missing_columns = [column for column in required_columns if column not in matches_df.columns]
        if missing_columns:
            continue
        side = matches_df[required_columns].copy()
        side.columns = ["player_name", "hand", "height_cm"]
        side["player_name_key"] = side["player_name"].map(_normalized_name_key)
        rows.append(side)
    if not rows:
        return pd.DataFrame(columns=["player_name_key", "hand", "height_cm"])

    combined = pd.concat(rows, ignore_index=True)
    combined = combined[combined["player_name_key"].astype(str).str.len() > 0].copy()
    output_rows: list[dict[str, object]] = []
    for key, group in combined.groupby("player_name_key", dropna=False):
        output_rows.append(
            {
                "player_name_key": key,
                "hand": _unique_non_missing_value(group["hand"], allowed_values={"L", "R"}),
                "height_cm": _unique_non_missing_value(group["height_cm"], numeric=True),
            }
        )
    return pd.DataFrame(output_rows)


def _derive_exact_age_from_birth_date(event_dates: pd.Series, birth_dates: pd.Series) -> pd.Series:
    event_ts = pd.to_datetime(event_dates, errors="coerce")
    birth_ts = pd.to_datetime(birth_dates, errors="coerce")
    age = pd.Series(pd.NA, index=event_ts.index, dtype="Float64")
    valid = event_ts.notna() & birth_ts.notna() & (event_ts >= birth_ts)
    if not valid.any():
        return age

    # This is an exact derivation from official birth dates and the match date,
    # not an estimate from a current profile age snapshot.
    age.loc[valid] = ((event_ts.loc[valid] - birth_ts.loc[valid]).dt.days / 365.2425).round(1)
    return age


def _fill_missing_column(frame: pd.DataFrame, destination: str, source: pd.Series) -> None:
    if destination not in frame.columns:
        if pd.api.types.is_numeric_dtype(source):
            frame[destination] = pd.to_numeric(source, errors="coerce").astype(float)
        else:
            frame[destination] = source
        return
    mask = _missing_mask(frame[destination])
    if not mask.any():
        return

    fill_values = source.loc[mask]
    if pd.api.types.is_numeric_dtype(frame[destination]):
        numeric_values = pd.to_numeric(fill_values, errors="coerce")
        frame.loc[mask, destination] = numeric_values.to_numpy(dtype=float, na_value=float("nan"))
        return
    frame.loc[mask, destination] = fill_values.astype(object)


def enrich_tennis_matches_with_player_profiles(
    matches_df: pd.DataFrame,
    profiles_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if matches_df.empty:
        return matches_df.copy()

    profiles = profiles_df.copy() if profiles_df is not None else load_tennis_player_profiles()
    source_matches = matches_df.copy()
    working = matches_df.copy()
    working["tour"] = working["tour"].astype(str).str.lower()

    profiles = _ensure_profile_columns(profiles)
    profiles = profiles.copy()
    profiles["tour"] = profiles["tour"].astype(str).str.lower()
    join_profiles = _build_profile_id_lookup(profiles)

    for prefix in ["player_a", "player_b"]:
        player_id_key = f"{prefix}_id_key"
        working[player_id_key] = working.apply(lambda row: canonical_player_id(row.get("tour"), row.get(f"{prefix}_id")), axis=1)
        renamed = join_profiles.rename(
            columns={
                "player_id_key": player_id_key,
                "birth_date": f"{prefix}_profile_birth_date",
                "birth_place": f"{prefix}_profile_birth_place",
                "country_code": f"{prefix}_profile_country_code",
                "country_name": f"{prefix}_profile_country_name",
                "hand_code": f"{prefix}_profile_hand_code",
                "hand_description": f"{prefix}_profile_hand_description",
                "backhand_code": f"{prefix}_profile_backhand_code",
                "backhand_description": f"{prefix}_profile_backhand_description",
                "height_cm": f"{prefix}_profile_height_cm",
                "weight_kg": f"{prefix}_profile_weight_kg",
            }
        )
        working = working.merge(renamed, on=["tour", player_id_key], how="left")

        _fill_missing_column(working, f"{prefix}_birth_date", working[f"{prefix}_profile_birth_date"])
        _fill_missing_column(working, f"{prefix}_birth_place", working[f"{prefix}_profile_birth_place"])
        _fill_missing_column(working, f"{prefix}_country_code", working[f"{prefix}_profile_country_code"])
        _fill_missing_column(working, f"{prefix}_country_name", working[f"{prefix}_profile_country_name"])
        _fill_missing_column(working, f"{prefix}_backhand", working[f"{prefix}_profile_backhand_description"])
        _fill_missing_column(working, f"{prefix}_weight_kg", working[f"{prefix}_profile_weight_kg"])
        _fill_missing_column(working, f"{prefix}_hand", working[f"{prefix}_profile_hand_code"])
        _fill_missing_column(working, f"{prefix}_height_cm", working[f"{prefix}_profile_height_cm"])

        derived_age = _derive_exact_age_from_birth_date(working["event_date"], working[f"{prefix}_birth_date"])
        _fill_missing_column(working, f"{prefix}_age", derived_age)

        profile_columns = [column for column in working.columns if column.startswith(f"{prefix}_profile_")]
        working = working.drop(columns=profile_columns + [player_id_key])

    # Official data frequently re-keys the same player across source windows.
    # Use only unique-by-name field values across cached profiles as a conservative alias layer.
    working = _apply_profile_name_backfill(
        working,
        profiles,
        key_builder=_normalized_name_key,
        abbreviated_only=False,
    )
    working = _apply_profile_name_backfill(
        working,
        profiles,
        key_builder=_initial_surname_key,
        abbreviated_only=True,
    )

    # Static fields observed elsewhere in the historical corpus are safe to carry
    # forward only when the exact normalized player name has a single unique value.
    historical_lookup = _build_historical_static_backfill_lookup(source_matches)
    if not historical_lookup.empty:
        for prefix in ["player_a", "player_b"]:
            key_column = f"{prefix}_history_name_key"
            working[key_column] = working[prefix].map(_normalized_name_key)
            renamed = historical_lookup.rename(
                columns={
                    "player_name_key": key_column,
                    "hand": f"{prefix}_history_hand",
                    "height_cm": f"{prefix}_history_height_cm",
                }
            )
            working = working.merge(renamed, on=key_column, how="left")
            _fill_missing_column(working, f"{prefix}_hand", working[f"{prefix}_history_hand"])
            _fill_missing_column(working, f"{prefix}_height_cm", working[f"{prefix}_history_height_cm"])
            working = working.drop(columns=[key_column, f"{prefix}_history_hand", f"{prefix}_history_height_cm"])

    return working


def summarize_tennis_player_profile_enrichment(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    profiles_df: Optional[pd.DataFrame] = None,
) -> dict[str, object]:
    before = before_df.copy()
    after = after_df.copy()
    profiles = profiles_df.copy() if profiles_df is not None else load_tennis_player_profiles(include_failed=True)

    if "event_date" in before.columns:
        before["event_date"] = pd.to_datetime(before["event_date"], errors="coerce")
        before["year"] = before["event_date"].dt.year
    if "event_date" in after.columns:
        after["event_date"] = pd.to_datetime(after["event_date"], errors="coerce")
        after["year"] = after["event_date"].dt.year

    def coverage_payload(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
        payload: dict[str, dict[str, int]] = {}
        for column in ENRICHABLE_PROFILE_FIELDS:
            if column not in frame.columns:
                payload[column] = {"non_null_rows": 0, "total_rows": int(len(frame))}
                continue
            payload[column] = {
                "non_null_rows": int((~_missing_mask(frame[column])).sum()),
                "total_rows": int(len(frame)),
            }
        return payload

    def coverage_by_year(frame: pd.DataFrame) -> dict[str, dict[str, dict[str, int]]]:
        if "year" not in frame.columns:
            return {}
        output: dict[str, dict[str, dict[str, int]]] = {}
        for year in sorted(frame["year"].dropna().astype(int).unique()):
            subset = frame[frame["year"] == year].copy()
            output[str(year)] = coverage_payload(subset)
        return output

    filled_counts: dict[str, int] = {}
    for column in ENRICHABLE_PROFILE_FIELDS:
        before_non_null = int((~_missing_mask(before[column])).sum()) if column in before.columns else 0
        after_non_null = int((~_missing_mask(after[column])).sum()) if column in after.columns else 0
        filled_counts[column] = max(after_non_null - before_non_null, 0)

    profile_status_counts = (
        profiles["fetch_status"].fillna("unknown").astype(str).value_counts().to_dict()
        if not profiles.empty and "fetch_status" in profiles.columns
        else {}
    )
    profile_tour_counts = (
        profiles["tour"].fillna("unknown").astype(str).value_counts().to_dict() if not profiles.empty and "tour" in profiles.columns else {}
    )

    return {
        "summary_generated_at_utc": _now_utc_iso(),
        "profile_rows_loaded": int(len(profiles)),
        "profile_status_counts": {str(key): int(value) for key, value in profile_status_counts.items()},
        "profile_tour_counts": {str(key): int(value) for key, value in profile_tour_counts.items()},
        "rows_before": int(len(before)),
        "rows_after": int(len(after)),
        "filled_counts": {str(key): int(value) for key, value in filled_counts.items()},
        "coverage_before": coverage_payload(before),
        "coverage_after": coverage_payload(after),
        "coverage_after_by_year": coverage_by_year(after),
    }


def write_tennis_player_profile_enrichment_summary(
    summary: dict[str, object],
    path: Optional[Path] = None,
) -> Path:
    _ensure_profile_dirs()
    output_path = path or TENNIS_PROFILE_SUMMARY_PATH
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
