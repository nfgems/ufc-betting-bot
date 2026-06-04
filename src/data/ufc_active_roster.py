"""Official UFC active-roster scraping helpers."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import RAW_DATA_DIR
from src.data.io_utils import write_csv_atomically
from src.data.name_utils import normalize_cross_source_name, same_person_name

logger = logging.getLogger(__name__)

OFFICIAL_ACTIVE_ROSTER_URL = "https://www.ufc.com/athletes/all?filters%5B0%5D=status%3A23"
OFFICIAL_ACTIVE_ROSTER_PATH = RAW_DATA_DIR / "ufc_active_roster_official.csv"
OFFICIAL_ACTIVE_ROSTER_IDENTITY_AUDIT_PATH = RAW_DATA_DIR / "ufc_active_roster_identity_audit.csv"
POWERSLAP_SEARCH_URL = "https://www.powerslap.com/wp-json/wp/v2/striker"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_REQUEST_DELAY_SECONDS = 0.35
_REQUEST_TIMEOUT_SECONDS = 30
_REQUEST_MAX_RETRIES = 3
_REQUEST_RETRY_BACKOFF_SECONDS = 1.0
_POWERSLAP_MATCH_LIMIT = 10
_PROFILE_ACTIVE_STATUS_KEY = "active"
_PROFILE_STATUS_KEYS = frozenset(
    {
        _PROFILE_ACTIVE_STATUS_KEY,
        "not fighting",
        "retired",
        "inactive",
    }
)
_ROSTER_GROWTH_FALLBACK_MIN_CACHED_ROWS = 100
_ROSTER_GROWTH_FALLBACK_RATIO = 1.35
_ROSTER_GROWTH_FALLBACK_ABSOLUTE_ROWS = 250
_powerslap_profile_cache: dict[str, dict[str, str]] = {}
_UNTRUSTED_OFFICIAL_PROFILE_FIELDS = (
    "profile_name",
    "canonical_athlete_url",
    "profile_division",
    "profile_record",
    "profile_status",
    "profile_source_tag",
    "birthplace",
    "age",
    "height",
    "reach",
    "weight",
    "octagon_debut",
    "alternate_slug_names",
)
_UNTRUSTED_SLUG_ALIAS_FIELDS = ("slug_name", "alternate_slug_names")
_TEST_PROFILE_NAME_KEYS = {
    "test",
    "test fighter",
    "test test",
    "testy test",
    "testing test",
}
IDENTITY_AUDIT_COLUMNS = (
    "official_name",
    "official_athlete_url",
    "profile_name",
    "slug_name",
    "identity_status",
    "identity_reason",
    "action",
)
RETAINED_ROSTER_COLUMNS = (
    "active_roster_live_present",
    "active_roster_retained_from_previous",
    "active_roster_retained_at_utc",
    "active_roster_missing_from_live_reason",
)


def _clean_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _status_key(value: object) -> str:
    return normalize_cross_source_name(value)


def _get_soup(url: str, *, session: requests.Session | None = None) -> BeautifulSoup:
    client = session or requests.Session()
    last_exc: Exception | None = None
    for attempt in range(1, _REQUEST_MAX_RETRIES + 1):
        try:
            response = client.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            time.sleep(_REQUEST_DELAY_SECONDS)
            return BeautifulSoup(response.text, "lxml")
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt >= _REQUEST_MAX_RETRIES:
                break
            backoff_seconds = _REQUEST_RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "Official UFC request failed for %s on attempt %d/%d: %s; retrying in %.1fs",
                url,
                attempt,
                _REQUEST_MAX_RETRIES,
                exc,
                backoff_seconds,
            )
            time.sleep(backoff_seconds)
    raise last_exc if last_exc is not None else RuntimeError(f"Failed to fetch {url}")


def _absolute_url(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return urljoin("https://www.ufc.com", text)


def _slug_to_name(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = urlparse(text).path if "://" in text else text
    slug = str(path).rstrip("/").rsplit("/", 1)[-1]
    return _clean_text(slug.replace("-", " "))


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _falsey(value: object) -> bool:
    return str(value or "").strip().lower() in {"0", "false", "no", "off"}


def _official_url_identity_trusted(row: dict[str, object]) -> bool:
    """Return whether slug/profile aliases from UFC.com are safe identity aliases."""
    status = str(row.get("official_url_identity_status") or "").strip().lower()
    explicit = row.get("official_url_identity_valid")
    if status in {"mismatch", "test_profile"}:
        return False
    if explicit is None or (isinstance(explicit, float) and pd.isna(explicit)):
        return True
    return not _falsey(explicit)


def _is_test_or_staging_profile(row: dict[str, object]) -> bool:
    names = [
        row.get("official_name"),
        row.get("profile_name"),
        row.get("slug_name"),
        *str(row.get("alternate_slug_names") or "").split("|"),
    ]
    name_keys = {normalize_cross_source_name(value) for value in names if str(value or "").strip()}
    if name_keys & _TEST_PROFILE_NAME_KEYS:
        return True

    url = str(row.get("official_athlete_url") or "").strip().casefold()
    path = urlparse(url).path.strip("/").casefold()
    slug = path.rsplit("/", 1)[-1] if path else ""
    return slug in {"test", "test-fighter", "test-test", "testy-test", "testing-test"}


def _validate_official_url_identity(row: dict[str, object]) -> dict[str, object]:
    official_name = _clean_text(row.get("official_name"))
    profile_name = _clean_text(row.get("profile_name"))
    slug_name = _clean_text(row.get("slug_name"))
    reasons: list[str] = []

    if _is_test_or_staging_profile(row):
        return {
            "official_url_identity_valid": False,
            "official_url_identity_status": "test_profile",
            "official_url_identity_reason": "test_or_staging_profile",
        }

    if official_name:
        profile_matches = bool(profile_name) and _same_identity_name(official_name, profile_name)
        slug_matches = bool(slug_name) and _same_identity_name(official_name, slug_name)
        if profile_name and not profile_matches:
            reasons.append(f"profile_name_mismatch:{profile_name}")
        if slug_name and not slug_matches:
            if profile_matches:
                return {
                    "official_url_identity_valid": True,
                    "official_url_identity_status": "slug_mismatch_profile_valid",
                    "official_url_identity_reason": f"slug_name_mismatch:{slug_name}",
                }
            reasons.append(f"slug_name_mismatch:{slug_name}")

    if reasons:
        return {
            "official_url_identity_valid": False,
            "official_url_identity_status": "mismatch",
            "official_url_identity_reason": "|".join(reasons),
        }

    return {
        "official_url_identity_valid": True,
        "official_url_identity_status": "valid",
        "official_url_identity_reason": "",
    }


def _identity_audit_row(row: dict[str, object], *, action: str) -> dict[str, object]:
    return {
        "official_name": _clean_text(row.get("official_name")),
        "official_athlete_url": _clean_text(row.get("official_athlete_url")),
        "profile_name": _clean_text(row.get("profile_name")),
        "slug_name": _clean_text(row.get("slug_name")),
        "identity_status": _clean_text(row.get("official_url_identity_status")),
        "identity_reason": _clean_text(row.get("official_url_identity_reason")),
        "action": action,
    }


def _quarantine_untrusted_official_profile_fields(row: dict[str, object]) -> None:
    for field in _UNTRUSTED_OFFICIAL_PROFILE_FIELDS:
        row[field] = ""


def _quarantine_untrusted_slug_alias_fields(row: dict[str, object]) -> None:
    for field in _UNTRUSTED_SLUG_ALIAS_FIELDS:
        row[field] = ""


def _verified_inactive_profile_status(row: dict[str, object]) -> bool:
    if not _official_url_identity_trusted(row):
        return False
    status_key = _status_key(row.get("profile_status"))
    return bool(status_key and status_key != _PROFILE_ACTIVE_STATUS_KEY)


def _live_roster_growth_exceeds_cached_guard(live_df: pd.DataFrame, cached_df: pd.DataFrame) -> bool:
    cached_rows = int(len(cached_df))
    live_rows = int(len(live_df))
    if cached_rows < _ROSTER_GROWTH_FALLBACK_MIN_CACHED_ROWS or live_rows <= cached_rows:
        return False

    allowed_rows = max(
        int(cached_rows * _ROSTER_GROWTH_FALLBACK_RATIO),
        cached_rows + _ROSTER_GROWTH_FALLBACK_ABSOLUTE_ROWS,
    )
    return live_rows > allowed_rows


def _cached_roster_retention_exceeds_live_guard(live_df: pd.DataFrame, cached_df: pd.DataFrame) -> bool:
    live_rows = int(len(live_df))
    cached_rows = int(len(cached_df))
    if live_rows < _ROSTER_GROWTH_FALLBACK_MIN_CACHED_ROWS or cached_rows <= live_rows:
        return False

    allowed_cached_rows = max(
        int(live_rows * _ROSTER_GROWTH_FALLBACK_RATIO),
        live_rows + _ROSTER_GROWTH_FALLBACK_ABSOLUTE_ROWS,
    )
    return cached_rows > allowed_cached_rows


def _same_identity_name(left: object, right: object) -> bool:
    if same_person_name(left, right):
        return True
    left_key = normalize_cross_source_name(left).replace(" ", "")
    right_key = normalize_cross_source_name(right).replace(" ", "")
    return bool(left_key and right_key and left_key == right_key)


def _write_identity_audit(path: Path | None, rows: list[dict[str, object]]) -> None:
    if path is None:
        return
    audit_df = pd.DataFrame(rows, columns=IDENTITY_AUDIT_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(path, index=False)


def _roster_merge_key(row: pd.Series | dict[str, object]) -> str:
    official_url = _normalize_roster_url(row.get("official_athlete_url"))
    if official_url:
        return f"url:{official_url}"
    official_name = normalize_cross_source_name(row.get("official_name"))
    if official_name:
        return f"name:{official_name}"
    return ""


def _normalize_roster_url(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.casefold() in {"nan", "none"}:
        return ""
    return text.rstrip("/").casefold()


def _roster_identity_name_keys(row: pd.Series | dict[str, object]) -> set[str]:
    names: list[object] = [
        row.get("official_name"),
        row.get("ufcstats_name"),
    ]
    if _official_url_identity_trusted(row):
        names.extend(
            [
                row.get("profile_name"),
                row.get("slug_name"),
                *str(row.get("alternate_slug_names") or "").split("|"),
            ]
        )
    keys: set[str] = set()
    for name in names:
        if name is None:
            continue
        try:
            if pd.isna(name):
                continue
        except (TypeError, ValueError):
            pass
        key = normalize_cross_source_name(name)
        if key and key != "nan":
            keys.add(key)
    return keys


def _roster_identity_keys(
    row: pd.Series | dict[str, object],
    *,
    include_url: bool = True,
) -> set[str]:
    keys: set[str] = set()
    if include_url:
        official_url = _normalize_roster_url(row.get("official_athlete_url"))
        if official_url:
            keys.add(f"url:{official_url}")
    ufcstats_url = _normalize_roster_url(row.get("ufcstats_url"))
    if ufcstats_url:
        keys.add(f"ufcstats_url:{ufcstats_url}")
    keys.update(f"name:{name}" for name in _roster_identity_name_keys(row))
    return keys


def _roster_rows_same_identity(
    left: pd.Series | dict[str, object],
    right: pd.Series | dict[str, object],
) -> bool:
    left_url = _normalize_roster_url(left.get("official_athlete_url"))
    right_url = _normalize_roster_url(right.get("official_athlete_url"))
    if left_url and right_url and left_url == right_url:
        return True

    left_ufcstats_url = _normalize_roster_url(left.get("ufcstats_url"))
    right_ufcstats_url = _normalize_roster_url(right.get("ufcstats_url"))
    if left_ufcstats_url and right_ufcstats_url:
        return left_ufcstats_url == right_ufcstats_url

    left_name_keys = _roster_identity_name_keys(left)
    right_name_keys = _roster_identity_name_keys(right)
    if not (left_name_keys & right_name_keys):
        return False

    left_official = _clean_text(left.get("official_name"))
    right_official = _clean_text(right.get("official_name"))
    if left_official and right_official and not _same_identity_name(left_official, right_official):
        return False

    return True


def _apply_canonical_official_athlete_url(row: dict[str, object]) -> None:
    canonical_url = _clean_text(row.get("canonical_athlete_url"))
    if not canonical_url or canonical_url.casefold() in {"nan", "none"}:
        return
    profile_name = _clean_text(row.get("profile_name"))
    official_name = _clean_text(row.get("official_name"))
    if profile_name and official_name and not _same_identity_name(profile_name, official_name):
        return
    row["official_athlete_url"] = canonical_url
    row["slug_name"] = _slug_to_name(canonical_url)


def _restore_cached_official_urls_for_live_aliases(
    live_df: pd.DataFrame,
    cached_df: pd.DataFrame,
) -> pd.DataFrame:
    if live_df.empty or cached_df.empty or "official_athlete_url" not in live_df.columns:
        return live_df

    cached_by_identity: dict[str, list[pd.Series]] = {}
    for _, cached_row in cached_df.iterrows():
        for key in _roster_identity_keys(cached_row, include_url=False):
            cached_by_identity.setdefault(key, []).append(cached_row)

    if not cached_by_identity:
        return live_df

    live = live_df.copy()
    for index, live_row in live.iterrows():
        live_url = _normalize_roster_url(live_row.get("official_athlete_url"))
        if str(live_row.get("official_url_identity_status") or "").strip() != "slug_mismatch_profile_valid":
            continue
        for key in _roster_identity_keys(live_row, include_url=False):
            for cached_row in cached_by_identity.get(key, []):
                cached_url = _clean_text(cached_row.get("official_athlete_url"))
                if not cached_url or _normalize_roster_url(cached_url) == live_url:
                    continue
                if _roster_rows_same_identity(cached_row, live_row):
                    live.at[index, "official_athlete_url"] = cached_url
                    live.at[index, "slug_name"] = _slug_to_name(cached_url)
                    break
            else:
                continue
            break
    return live


def _merge_cached_roster_rows_missing_from_live(
    live_df: pd.DataFrame,
    cached_df: pd.DataFrame,
    *,
    retained_at_utc: str,
    excluded_keys: set[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Retain previously tracked roster rows that disappear from the latest UFC.com scrape."""
    if live_df.empty or cached_df.empty:
        return live_df, []
    live_df = _restore_cached_official_urls_for_live_aliases(live_df, cached_df)
    if len(live_df) >= len(cached_df):
        return live_df, []
    excluded_keys = excluded_keys or set()

    live_rows = [row for _, row in live_df.iterrows()]
    retained_rows: list[pd.Series] = []
    retained_details: list[dict[str, str]] = []
    for _, cached_row in cached_df.iterrows():
        key = _roster_merge_key(cached_row)
        identity_keys = _roster_identity_keys(cached_row)
        if not key or not identity_keys:
            continue
        if any(_roster_rows_same_identity(cached_row, live_row) for live_row in live_rows):
            continue
        if key in excluded_keys:
            continue
        retained_rows.append(cached_row)
        retained_details.append(
            {
                "official_name": _clean_text(cached_row.get("official_name")),
                "official_athlete_url": _clean_text(cached_row.get("official_athlete_url")),
                "ufcstats_url": _clean_text(cached_row.get("ufcstats_url")),
            }
        )

    if not retained_rows:
        return live_df, []

    live = live_df.copy()
    retained = pd.DataFrame(retained_rows).copy()
    for column in RETAINED_ROSTER_COLUMNS:
        if column not in live.columns:
            live[column] = ""
        if column not in retained.columns:
            retained[column] = ""

    live["active_roster_live_present"] = True
    live["active_roster_retained_from_previous"] = False
    live["active_roster_retained_at_utc"] = ""
    live["active_roster_missing_from_live_reason"] = ""

    retained["active_roster_live_present"] = False
    retained["active_roster_retained_from_previous"] = True
    retained["active_roster_retained_at_utc"] = retained_at_utc
    retained["active_roster_missing_from_live_reason"] = "absent_from_latest_ufc_active_roster_sync"
    retained["coverage_eligible"] = False

    columns = list(live.columns)
    for column in retained.columns:
        if column not in columns:
            columns.append(column)
    live = live.reindex(columns=columns)
    retained = retained.reindex(columns=columns)
    merged = pd.concat([live, retained], ignore_index=True, sort=False)
    if "official_name" in merged.columns and "official_athlete_url" in merged.columns:
        merged = merged.sort_values(["official_name", "official_athlete_url"]).reset_index(drop=True)
    return merged, retained_details


def _parse_roster_card(card) -> dict[str, object] | None:
    link = card.find("a", href=lambda href: href and "/athlete/" in href)
    name_el = card.select_one("span.c-listing-athlete__name")
    if link is None or name_el is None:
        return None

    division_el = card.select_one("span.c-listing-athlete__title .field__item")
    record_el = card.select_one("span.c-listing-athlete__record")
    nickname_el = card.select_one("span.c-listing-athlete__nickname")
    athlete_url = _absolute_url(link.get("href"))
    slug_name = _slug_to_name(athlete_url)

    return {
        "official_name": _clean_text(name_el.get_text(" ", strip=True)),
        "official_athlete_url": athlete_url,
        "nickname": _clean_text(nickname_el.get_text(" ", strip=True)) if nickname_el else "",
        "division": _clean_text(division_el.get_text(" ", strip=True)) if division_el else "",
        "record": _clean_text(record_el.get_text(" ", strip=True)) if record_el else "",
        "slug_name": slug_name,
    }


def _parse_alternate_slug_names(soup: BeautifulSoup) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for link in soup.select("link[rel='alternate'][href]"):
        href = str(link.get("href") or "").strip()
        if "/athlete/" not in href:
            continue
        name = _slug_to_name(href)
        key = normalize_cross_source_name(name)
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def _parse_canonical_athlete_url(soup: BeautifulSoup) -> str:
    for link in soup.select("link[rel][href]"):
        rel = link.get("rel") or []
        rel_values = {rel.casefold()} if isinstance(rel, str) else {str(value).casefold() for value in rel}
        href = str(link.get("href") or "").strip()
        if "canonical" in rel_values and "/athlete/" in href:
            return _absolute_url(href)
    return ""


def _filter_alternate_slug_names(names: list[str], *, reference_name: str) -> list[str]:
    reference_tokens = [token for token in normalize_cross_source_name(reference_name).split() if token]
    if not reference_tokens:
        return names

    filtered: list[str] = []
    seen: set[str] = set()
    reference_first = reference_tokens[0]
    reference_last = reference_tokens[-1]
    for name in names:
        tokens = [token for token in normalize_cross_source_name(name).split() if token]
        if not tokens:
            continue
        if tokens[0] != reference_first and tokens[-1] != reference_last:
            continue
        key = normalize_cross_source_name(name)
        if key in seen:
            continue
        seen.add(key)
        filtered.append(name)
    return filtered


def _dedupe_names(values: list[object]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        key = normalize_cross_source_name(text)
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(text)
    return names


def _parse_bio_fields(soup: BeautifulSoup) -> dict[str, str]:
    fields: dict[str, str] = {}
    for field in soup.select("div.c-bio__field"):
        label_el = field.select_one("div.c-bio__label")
        text_el = field.select_one("div.c-bio__text")
        if label_el is None or text_el is None:
            continue
        label = _clean_text(label_el.get_text(" ", strip=True))
        text = _clean_text(text_el.get_text(" ", strip=True))
        if label and text:
            fields[label] = text
    return fields


def _normalize_official_inches_measurement(value: object) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return f"{text} in"
    return text


def _rendered_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return _clean_text(BeautifulSoup(text, "lxml").get_text(" ", strip=True))


def _combat_sport_query_names(row: dict[str, object]) -> list[str]:
    values = [row.get("official_name"), row.get("ufcstats_name")]
    if _official_url_identity_trusted(row):
        values.extend(
            [
                row.get("profile_name"),
                row.get("slug_name"),
                *str(row.get("alternate_slug_names") or "").split("|"),
            ]
        )
    return _dedupe_names(values)


def _search_powerslap_profile(
    fighter_name: str,
    *,
    session: requests.Session | None = None,
) -> dict[str, str]:
    cache_key = normalize_cross_source_name(fighter_name)
    if not cache_key:
        return {}
    if cache_key in _powerslap_profile_cache:
        return _powerslap_profile_cache[cache_key]

    client = session or requests.Session()
    try:
        response = client.get(
            POWERSLAP_SEARCH_URL,
            params={"search": fighter_name, "per_page": _POWERSLAP_MATCH_LIMIT},
            headers=_HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        time.sleep(_REQUEST_DELAY_SECONDS)
        results = response.json()
    except Exception as exc:
        logger.debug("Power Slap profile search failed for %s: %s", fighter_name, exc)
        _powerslap_profile_cache[cache_key] = {}
        return {}

    for item in results:
        candidate_name = _rendered_text((item.get("title") or {}).get("rendered"))
        if not candidate_name or not same_person_name(fighter_name, candidate_name):
            continue
        profile = {
            "profile_name": candidate_name,
            "profile_url": _clean_text(item.get("link")),
        }
        _powerslap_profile_cache[cache_key] = profile
        return profile

    _powerslap_profile_cache[cache_key] = {}
    return {}


def _classify_combat_sport(
    row: dict[str, object],
    *,
    session: requests.Session | None = None,
) -> dict[str, str]:
    if str(row.get("ufcstats_url") or "").strip():
        return {
            "combat_sport": "mma",
            "combat_sport_reason": "ufcstats_profile_match",
            "combat_sport_profile_url": "",
        }

    for query_name in _combat_sport_query_names(row):
        profile = _search_powerslap_profile(query_name, session=session)
        profile_name = _clean_text(profile.get("profile_name"))
        if not profile_name:
            continue
        if any(
            same_person_name(alias, profile_name)
            for alias in _combat_sport_query_names(row)
        ):
            return {
                "combat_sport": "power_slap",
                "combat_sport_reason": "powerslap_profile_match",
                "combat_sport_profile_url": _clean_text(profile.get("profile_url")),
            }

    return {
        "combat_sport": "mma",
        "combat_sport_reason": "",
        "combat_sport_profile_url": "",
    }


def scrape_official_athlete_profile(
    athlete_url: str,
    *,
    session: requests.Session | None = None,
) -> dict[str, object]:
    soup = _get_soup(athlete_url, session=session)
    hero_tags = [_clean_text(tag.get_text(" ", strip=True)) for tag in soup.select("p.hero-profile__tag")]
    hero_tags = [tag for tag in hero_tags if tag]
    hero_name_el = soup.select_one("h1.hero-profile__name")
    division_el = soup.select_one("p.hero-profile__division-title")
    record_el = soup.select_one("p.hero-profile__division-body")
    hero_name = _clean_text(hero_name_el.get_text(" ", strip=True)) if hero_name_el else ""
    division = _clean_text(division_el.get_text(" ", strip=True)) if division_el else ""
    record = _clean_text(record_el.get_text(" ", strip=True)) if record_el else ""
    bio_fields = _parse_bio_fields(soup)

    status = ""
    source_tag = ""
    division_key = normalize_cross_source_name(division)
    for tag in hero_tags:
        lower = tag.casefold()
        tag_key = normalize_cross_source_name(tag)
        if "division" in lower and not division:
            division = tag
            division_key = normalize_cross_source_name(tag)
        elif tag_key in _PROFILE_STATUS_KEYS and not status:
            status = tag
        elif (
            tag_key
            and division_key
            and (tag_key == division_key or tag_key in division_key or division_key in tag_key)
        ):
            continue
        elif not source_tag:
            source_tag = tag

    alternate_slug_names = _filter_alternate_slug_names(
        _parse_alternate_slug_names(soup),
        reference_name=hero_name or _slug_to_name(athlete_url),
    )

    return {
        "profile_name": hero_name,
        "canonical_athlete_url": _parse_canonical_athlete_url(soup),
        "profile_division": division,
        "profile_record": record,
        "profile_status": status,
        "profile_source_tag": source_tag,
        "birthplace": bio_fields.get("Place of Birth", ""),
        "age": bio_fields.get("Age", ""),
        "height": _normalize_official_inches_measurement(bio_fields.get("Height", "")),
        "reach": _normalize_official_inches_measurement(bio_fields.get("Reach", "")),
        "weight": bio_fields.get("Weight", ""),
        "octagon_debut": bio_fields.get("Octagon Debut", ""),
        "alternate_slug_names": "|".join(alternate_slug_names),
    }


def _load_local_ufcstats_candidates() -> dict[str, list[dict[str, str]]]:
    candidates: dict[str, list[dict[str, str]]] = {}

    def _append_candidate(name: object, url: object, source: str) -> None:
        name_text = _clean_text(name)
        url_text = str(url or "").strip()
        key = normalize_cross_source_name(name_text)
        if not key or not url_text:
            return
        candidates.setdefault(key, []).append(
            {"name": name_text, "fighter_url": url_text, "source": source}
        )

    scraped_path = RAW_DATA_DIR / "ufc_fighters_scraped.csv"
    if scraped_path.exists():
        scraped_df = pd.read_csv(scraped_path, usecols=["name", "fighter_url"])
        for _, row in scraped_df.dropna(subset=["fighter_url"]).iterrows():
            _append_candidate(row.get("name"), row.get("fighter_url"), "ufc_fighters_scraped")

    details_path = RAW_DATA_DIR / "ufc-fighter-details.csv"
    if details_path.exists():
        details_df = pd.read_csv(details_path, usecols=["FIRST", "LAST", "URL"])
        name_series = (details_df["FIRST"].fillna("").astype(str) + " " + details_df["LAST"].fillna("").astype(str)).str.strip()
        for index, url in details_df["URL"].items():
            _append_candidate(name_series.iloc[index], url, "ufc_fighter_details")

    return candidates


def _resolve_local_ufcstats_profile(
    row: dict[str, object],
    *,
    candidates: dict[str, list[dict[str, str]]],
) -> dict[str, object]:
    alias_names = [
        row.get("official_name"),
    ]
    if _official_url_identity_trusted(row):
        alias_names.extend(
            [
                row.get("profile_name"),
                row.get("slug_name"),
                *str(row.get("alternate_slug_names") or "").split("|"),
            ]
        )
    match_pool: dict[str, dict[str, str]] = {}
    for alias in alias_names:
        key = normalize_cross_source_name(alias)
        if not key:
            continue
        for candidate in candidates.get(key, []):
            match_pool[candidate["fighter_url"]] = candidate

    if len(match_pool) == 1:
        candidate = next(iter(match_pool.values()))
        return {
            "ufcstats_name": candidate["name"],
            "ufcstats_url": candidate["fighter_url"],
            "ufcstats_resolution": candidate["source"],
        }

    preferred: list[dict[str, str]] = []
    for candidate in match_pool.values():
        if any(same_person_name(alias, candidate["name"]) for alias in alias_names if alias):
            preferred.append(candidate)
    if len(preferred) == 1:
        candidate = preferred[0]
        return {
            "ufcstats_name": candidate["name"],
            "ufcstats_url": candidate["fighter_url"],
            "ufcstats_resolution": candidate["source"],
        }

    return {
        "ufcstats_name": "",
        "ufcstats_url": "",
        "ufcstats_resolution": "",
    }


def _resolve_via_live_search(
    row: dict[str, object],
    *,
    session: requests.Session | None = None,
) -> dict[str, object]:
    from src.data.fighter_lookup import search_fighter_url
    from src.data.scraper import scrape_fighter

    queries = [
        row.get("official_name"),
    ]
    if _official_url_identity_trusted(row):
        queries.extend(
            [
                row.get("profile_name"),
                row.get("slug_name"),
                *str(row.get("alternate_slug_names") or "").split("|"),
            ]
        )
    seen_queries: set[str] = set()
    for query in queries:
        query_text = _clean_text(query)
        key = normalize_cross_source_name(query_text)
        if not key or key in seen_queries:
            continue
        seen_queries.add(key)
        fighter_url = search_fighter_url(query_text)
        if not fighter_url:
            continue
        try:
            profile = scrape_fighter(fighter_url)
        except Exception as exc:
            logger.debug("Failed to scrape UFCStats profile %s for %s: %s", fighter_url, query_text, exc)
            profile = None
        if not profile:
            return {
                "ufcstats_name": "",
                "ufcstats_url": fighter_url,
                "ufcstats_resolution": "live_search_url_only",
            }
        return {
            "ufcstats_name": _clean_text(profile.get("name")),
            "ufcstats_url": fighter_url,
            "ufcstats_resolution": "live_search",
        }
    return {
        "ufcstats_name": "",
        "ufcstats_url": "",
        "ufcstats_resolution": "",
    }


def scrape_official_active_roster(
    *,
    fetch_profile_details: bool = True,
    max_pages: int | None = None,
    resolve_ufcstats: bool = True,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    client = session or requests.Session()
    rows: list[dict[str, object]] = []
    identity_audit_rows: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    page = 0
    candidates = _load_local_ufcstats_candidates() if resolve_ufcstats else {}

    while True:
        if max_pages is not None and page >= max_pages:
            break
        page_url = OFFICIAL_ACTIVE_ROSTER_URL if page == 0 else f"{OFFICIAL_ACTIVE_ROSTER_URL}&page={page}"
        soup = _get_soup(page_url, session=client)
        cards = soup.select("div.c-listing-athlete-flipcard")
        if not cards:
            break

        page_added = 0
        for card in cards:
            parsed = _parse_roster_card(card)
            if not parsed:
                continue
            athlete_url = str(parsed.get("official_athlete_url") or "").strip()
            athlete_url_key = _normalize_roster_url(athlete_url)
            if not athlete_url_key or athlete_url_key in seen_urls:
                continue
            seen_urls.add(athlete_url_key)
            if fetch_profile_details:
                try:
                    parsed.update(scrape_official_athlete_profile(athlete_url, session=client))
                    _apply_canonical_official_athlete_url(parsed)
                except Exception as exc:
                    logger.warning("Failed to scrape official athlete profile %s: %s", athlete_url, exc)
            else:
                parsed.setdefault("profile_name", "")
                parsed.setdefault("profile_division", "")
                parsed.setdefault("profile_record", "")
                parsed.setdefault("profile_status", "")
                parsed.setdefault("profile_source_tag", "")
                parsed.setdefault("birthplace", "")
                parsed.setdefault("age", "")
                parsed.setdefault("height", "")
                parsed.setdefault("reach", "")
                parsed.setdefault("weight", "")
                parsed.setdefault("octagon_debut", "")
                parsed.setdefault("alternate_slug_names", "")
                parsed.setdefault("combat_sport", "")
                parsed.setdefault("combat_sport_reason", "")
                parsed.setdefault("combat_sport_profile_url", "")

            canonical_url_key = _normalize_roster_url(parsed.get("official_athlete_url"))
            if canonical_url_key:
                if canonical_url_key != athlete_url_key and canonical_url_key in seen_urls:
                    continue
                seen_urls.add(canonical_url_key)

            parsed.update(_validate_official_url_identity(parsed))
            if str(parsed.get("official_url_identity_status") or "") == "test_profile":
                identity_audit_rows.append(
                    _identity_audit_row(parsed, action="excluded_test_profile")
                )
                logger.info(
                    "Excluded UFC test/staging athlete row from active roster: %s (%s)",
                    parsed.get("official_name"),
                    parsed.get("official_athlete_url"),
                )
                continue
            if str(parsed.get("official_url_identity_status") or "") == "slug_mismatch_profile_valid":
                identity_audit_rows.append(
                    _identity_audit_row(parsed, action="quarantined_untrusted_slug_alias")
                )
                logger.info(
                    "Suppressed untrusted UFC.com slug alias for %s at %s: %s",
                    parsed.get("official_name"),
                    parsed.get("official_athlete_url"),
                    parsed.get("official_url_identity_reason"),
                )
                _quarantine_untrusted_slug_alias_fields(parsed)
            if not _official_url_identity_trusted(parsed):
                identity_audit_rows.append(
                    _identity_audit_row(parsed, action="quarantined_untrusted_url_identity")
                )
                logger.warning(
                    "Quarantined untrusted UFC.com identity fields for %s at %s: %s",
                    parsed.get("official_name"),
                    parsed.get("official_athlete_url"),
                    parsed.get("official_url_identity_reason"),
                )
                _quarantine_untrusted_official_profile_fields(parsed)
            if fetch_profile_details and _verified_inactive_profile_status(parsed):
                identity_audit_rows.append(
                    _identity_audit_row(parsed, action="excluded_inactive_profile_status")
                )
                logger.info(
                    "Excluded non-active UFC athlete row from active roster: %s (%s, status=%s)",
                    parsed.get("official_name"),
                    parsed.get("official_athlete_url"),
                    parsed.get("profile_status"),
                )
                continue

            if resolve_ufcstats:
                resolution = _resolve_local_ufcstats_profile(parsed, candidates=candidates)
                if not resolution.get("ufcstats_url"):
                    resolution = _resolve_via_live_search(parsed, session=client)
                parsed.update(resolution)
            parsed.update(_classify_combat_sport(parsed, session=client))
            parsed["coverage_eligible"] = str(parsed.get("combat_sport") or "").strip().lower() != "power_slap"
            rows.append(parsed)
            page_added += 1

        logger.info("Scraped official UFC active roster page %d with %d athlete cards", page, page_added)
        page += 1

    if not rows:
        df = pd.DataFrame()
        df.attrs["identity_audit_rows"] = identity_audit_rows
        return df

    df = pd.DataFrame(rows)
    df = df.sort_values(["official_name", "official_athlete_url"]).reset_index(drop=True)
    df.attrs["identity_audit_rows"] = identity_audit_rows
    return df


def sync_official_active_roster(
    *,
    output_path: Path | None = None,
    fetch_profile_details: bool = True,
    max_pages: int | None = None,
    resolve_ufcstats: bool = True,
    allow_cached_fallback: bool = True,
    identity_audit_path: Path | None = OFFICIAL_ACTIVE_ROSTER_IDENTITY_AUDIT_PATH,
) -> pd.DataFrame:
    output_path = Path(output_path) if output_path is not None else OFFICIAL_ACTIVE_ROSTER_PATH
    try:
        df = scrape_official_active_roster(
            fetch_profile_details=fetch_profile_details,
            max_pages=max_pages,
            resolve_ufcstats=resolve_ufcstats,
        )
        identity_audit_rows = list(df.attrs.get("identity_audit_rows") or [])
        _write_identity_audit(identity_audit_path, identity_audit_rows)
        intentionally_excluded_keys = {
            key
            for key in (
                _roster_merge_key(row)
                for row in identity_audit_rows
                if isinstance(row, dict)
                and str(row.get("action") or "").strip() == "excluded_test_profile"
            )
            if key
        }
        retained_missing_live_rows: list[dict[str, str]] = []
        discarded_suspicious_cached_rows = 0
        if output_path.exists():
            try:
                cached_df = pd.read_csv(output_path)
            except Exception as exc:
                logger.warning("Failed to load cached UFC active-roster rows for live sync guard: %s", exc)
                cached_df = pd.DataFrame()

            if _live_roster_growth_exceeds_cached_guard(df, cached_df):
                raise RuntimeError(
                    "Official UFC roster live sync returned "
                    f"{len(df)} rows versus cached {len(cached_df)} rows; "
                    "refusing to overwrite cached roster because this exceeds the growth guard"
                )

            if _cached_roster_retention_exceeds_live_guard(df, cached_df):
                discarded_suspicious_cached_rows = int(len(cached_df) - len(df))
                logger.warning(
                    "Discarding %d stale cached UFC active-roster row(s): cached roster has %d rows "
                    "versus %d live active rows, which exceeds the retention guard",
                    discarded_suspicious_cached_rows,
                    len(cached_df),
                    len(df),
                )
            elif not cached_df.empty:
                try:
                    df, retained_missing_live_rows = _merge_cached_roster_rows_missing_from_live(
                        df,
                        cached_df,
                        retained_at_utc=datetime.now(timezone.utc).isoformat(),
                        excluded_keys=intentionally_excluded_keys,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to preserve cached UFC active-roster rows missing from live sync: %s",
                        exc,
                    )
        write_csv_atomically(df, output_path, refuse_empty=True)
        df.attrs["identity_audit_rows"] = identity_audit_rows
        df.attrs["sync_source"] = "live"
        df.attrs["sync_fallback_used"] = False
        df.attrs["retained_missing_live_rows"] = retained_missing_live_rows
        df.attrs["discarded_suspicious_cached_rows"] = discarded_suspicious_cached_rows
        logger.info("Saved official UFC active roster with %d rows to %s", len(df), output_path)
        if retained_missing_live_rows:
            preview = ", ".join(
                row.get("official_name", "")
                for row in retained_missing_live_rows[:10]
                if row.get("official_name")
            )
            logger.warning(
                "Retained %d previously tracked UFC active-roster row(s) missing from live sync%s",
                len(retained_missing_live_rows),
                f": {preview}" if preview else "",
            )
        return df
    except Exception as exc:
        if not allow_cached_fallback or not output_path.exists():
            raise
        try:
            cached_df = pd.read_csv(output_path)
        except Exception as cached_exc:
            raise RuntimeError(
                f"Official UFC roster sync failed ({exc}) and cached roster fallback at {output_path} "
                f"could not be loaded: {cached_exc}"
            ) from exc
        if cached_df.empty:
            raise RuntimeError(
                f"Official UFC roster sync failed ({exc}) and cached roster fallback at {output_path} is empty"
            ) from exc

        cached_mtime_iso = datetime.fromtimestamp(
            output_path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()
        cached_df.attrs["sync_source"] = "cached"
        cached_df.attrs["sync_fallback_used"] = True
        cached_df.attrs["sync_error"] = str(exc)
        cached_df.attrs["sync_cached_snapshot_mtime_utc"] = cached_mtime_iso
        logger.warning(
            "Official UFC roster sync failed; reusing cached roster snapshot from %s at %s: %s",
            output_path,
            cached_mtime_iso,
            exc,
        )
        return cached_df
