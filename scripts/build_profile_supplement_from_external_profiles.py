"""
Build or extend the local supplemental fighter-profile artifact from public sources.

This focuses on the remaining recoverable V4 profile gaps without fabricating data.
It currently uses:
- ESPN's MMA athlete API for height, reach, weight, stance, and exact DOB when available
- MartialBot for height, reach, stance, and exact DOB when available
- FightDX for height, reach, weight, stance, and exact DOB when available
- Tapology for height, reach, weight, and exact DOB when available
- Sherdog for height, weight, and DOB when available
- Wikipedia infoboxes for reach, stance, and exact DOB when available
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import RAW_DATA_DIR  # noqa: E402
from src.data.fallback_scrapers import (  # noqa: E402
    clear_fallback_cache,
    scrape_espn_profile,
    scrape_sherdog_page,
    scrape_fightdx_profile,
    scrape_martialbot_profile,
    scrape_tapology_profile,
    search_espn,
    search_sherdog,
    search_fightdx,
    search_martialbot,
    search_tapology,
)
from src.data.name_utils import normalize_person_name, same_person_name  # noqa: E402

logger = logging.getLogger(__name__)


DEFAULT_INPUT = RAW_DATA_DIR / "ufc_fighters_scraped.csv"
DEFAULT_OUTPUT = RAW_DATA_DIR / "ufc_fighters_profile_supplement.csv"
TARGET_FIELDS = ("height", "reach", "weight", "stance", "dob")
TARGET_GAP_FIELDS = {"height", "reach", "weight", "stance", "age"}
ALL_SOURCES = ("martialbot", "espn", "wikipedia", "fightdx", "tapology", "sherdog")
SOURCE_RECOVERABLE_FIELDS = {
    "martialbot": ("height", "reach", "stance", "dob"),
    "espn": ("height", "reach", "weight", "stance", "dob"),
    "wikipedia": ("reach", "stance", "dob"),
    "fightdx": ("height", "reach", "weight", "stance", "dob"),
    "tapology": ("height", "reach", "weight", "dob"),
    "sherdog": ("height", "weight", "dob"),
}
SOURCE_LOG_LABELS = {
    "martialbot": "MartialBot",
    "espn": "ESPN",
    "wikipedia": "Wikipedia",
    "fightdx": "FightDX",
    "tapology": "Tapology",
    "sherdog": "Sherdog",
}
_BASE_PROFILE_COLUMNS = ("name", *TARGET_FIELDS)
_SINGLE_NAME_COLUMNS = ("name", "ufcstats_name", "official_name", "profile_name", "slug_name")
_MULTI_NAME_COLUMNS = ("alternate_slug_names", "search_names")
MARTIALBOT_SEARCH_NAME_OVERRIDES = {
    "tsuyoshi kohsaka": "Tsuyoshi Kosaka",
}
TAPOLOGY_SEARCH_NAME_OVERRIDES = {
    "felix lee mitchell": "Felix Mitchell",
}
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}
WIKIPEDIA_429_MAX_ATTEMPTS = 4
WIKIPEDIA_429_BASE_BACKOFF_SECONDS = 10.0
WIKIPEDIA_429_MAX_BACKOFF_SECONDS = 60.0
_WIKIPEDIA_ALLOWED_TITLE_QUALIFIERS = (
    "fighter",
    "mixed martial artist",
    "martial artist",
    "boxer",
    "kickboxer",
    "grappler",
    "wrestler",
)
_TEST_PROFILE_NAME_KEYS = {"test", "test fighter", "test test", "testy test", "testing test"}


def _blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() in {"", "-", "--", "N/A", "nan", "??"}


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _dedupe_names(values: list[object]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        if _blank(value):
            continue
        text = _clean_text(value)
        key = normalize_person_name(text)
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(text)
    return names


def _first_nonblank(*values: object) -> str:
    for value in values:
        if not _blank(value):
            return _clean_text(value)
    return ""


def _row_candidate_names(row: pd.Series | dict[str, object]) -> list[str]:
    names: list[object] = []
    trusted_official_url_identity = _official_url_identity_trusted(row)
    for column in _SINGLE_NAME_COLUMNS:
        if column in {"profile_name", "slug_name"} and not trusted_official_url_identity:
            continue
        names.append(row.get(column))
    for column in _MULTI_NAME_COLUMNS:
        if column == "alternate_slug_names" and not trusted_official_url_identity:
            continue
        names.extend(str(row.get(column) or "").split("|"))
    return _dedupe_names(names)


def _official_url_identity_trusted(row: pd.Series | dict[str, object]) -> bool:
    status = str(row.get("official_url_identity_status") or "").strip().lower()
    if status in {"mismatch", "test_profile"}:
        return False
    explicit = row.get("official_url_identity_valid")
    if _blank(explicit):
        return True
    return str(explicit).strip().lower() not in {"0", "false", "no", "off"}


def _ensure_profile_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in _BASE_PROFILE_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    if "search_names" not in result.columns:
        result["search_names"] = ""
    return result


def _load_scraped_profile_rows(scraped_fighters_path: Path) -> pd.DataFrame:
    if not scraped_fighters_path.exists():
        return pd.DataFrame(columns=[*_BASE_PROFILE_COLUMNS, "search_names"])

    scraped_df = pd.read_csv(scraped_fighters_path)
    if scraped_df.empty or "name" not in scraped_df.columns:
        raise SystemExit(f"Scraped fighter profile artifact is empty or missing 'name': {scraped_fighters_path}")

    scraped_df = _ensure_profile_columns(scraped_df)
    scraped_df["search_names"] = scraped_df["name"].fillna("").astype(str)
    return scraped_df


def _merged_field_value(
    scraped_row: dict[str, object] | None,
    source_row: dict[str, object],
    field: str,
) -> object:
    if scraped_row is not None and not _blank(scraped_row.get(field)):
        return scraped_row.get(field)
    if field in source_row and not _blank(source_row.get(field)):
        return source_row.get(field)
    return ""


def _candidate_source_row_is_coverage_eligible(row: dict[str, object]) -> bool:
    if _is_test_or_staging_profile(row):
        return False
    coverage_eligible = row.get("coverage_eligible")
    if not _blank(coverage_eligible):
        return str(coverage_eligible).strip().lower() not in {"0", "false", "no", "off"}
    return _clean_text(row.get("combat_sport")).casefold() != "power_slap"


def _is_test_or_staging_profile(row: dict[str, object]) -> bool:
    names = [row.get("official_name"), row.get("profile_name"), row.get("slug_name")]
    names.extend(str(row.get("alternate_slug_names") or "").split("|"))
    name_keys = {normalize_person_name(value) for value in names if str(value or "").strip()}
    if name_keys & _TEST_PROFILE_NAME_KEYS:
        return True
    url = str(row.get("official_athlete_url") or "").strip()
    path = urlparse(url).path.strip("/").lower() if url else ""
    slug = path.rsplit("/", 1)[-1] if path else ""
    return slug in {"test", "test-fighter", "test-test", "testy-test", "testing-test"}


def _build_candidate_universe(
    scraped_fighters_path: Path,
    *,
    candidate_source_csv: Path | None = None,
) -> pd.DataFrame:
    scraped_df = _load_scraped_profile_rows(scraped_fighters_path)
    if candidate_source_csv is None:
        return scraped_df

    candidate_source_df = pd.read_csv(candidate_source_csv)
    if candidate_source_df.empty:
        return pd.DataFrame(columns=[*_BASE_PROFILE_COLUMNS, "search_names"])

    scraped_lookup = {
        normalize_person_name(row.get("name")): row
        for row in scraped_df.to_dict(orient="records")
        if not _blank(row.get("name"))
    }

    candidate_lookup: dict[str, dict[str, object]] = {}
    for _, source_row in candidate_source_df.iterrows():
        source_dict = source_row.to_dict()
        if not _candidate_source_row_is_coverage_eligible(source_dict):
            continue
        aliases = _row_candidate_names(source_dict)
        if not aliases:
            continue

        scraped_match = None
        for alias in aliases:
            scraped_match = scraped_lookup.get(normalize_person_name(alias))
            if scraped_match is not None:
                break

        canonical_name = _first_nonblank(
            (scraped_match or {}).get("name"),
            source_dict.get("ufcstats_name"),
            source_dict.get("name"),
            source_dict.get("official_name"),
            source_dict.get("profile_name"),
            aliases[0] if aliases else "",
        )
        canonical_key = normalize_person_name(canonical_name)
        if not canonical_key:
            continue

        merged_row = candidate_lookup.setdefault(
            canonical_key,
            {
                "name": canonical_name,
                "height": "",
                "reach": "",
                "weight": "",
                "stance": "",
                "dob": "",
                "search_names": "",
            },
        )
        if not merged_row["name"]:
            merged_row["name"] = canonical_name

        merged_aliases = _dedupe_names(
            [
                merged_row.get("name"),
                canonical_name,
                *(merged_row.get("search_names", "").split("|")),
                *(scraped_match or {}).get("search_names", "").split("|"),
                *aliases,
            ]
        )
        merged_row["search_names"] = "|".join(merged_aliases)

        for field in TARGET_FIELDS:
            if _blank(merged_row.get(field)):
                merged_row[field] = _merged_field_value(scraped_match, source_dict, field)

    return _ensure_profile_columns(pd.DataFrame(candidate_lookup.values()))


def _filter_gap_candidates(
    candidate_universe: pd.DataFrame,
    *,
    gap_audit_csv: Path | None = None,
) -> pd.DataFrame:
    if candidate_universe.empty:
        return candidate_universe.copy()

    if gap_audit_csv is not None:
        gap_df = pd.read_csv(gap_audit_csv)
        gap_keys = {
            normalize_person_name(row.get("fighter_name"))
            for _, row in gap_df.iterrows()
            if row.get("field") in TARGET_GAP_FIELDS and not _blank(row.get("fighter_name"))
        }

        def _matches_gap(row: pd.Series) -> bool:
            names = [row.get("name"), *str(row.get("search_names") or "").split("|")]
            return any(normalize_person_name(name) in gap_keys for name in names)

        return candidate_universe[candidate_universe.apply(_matches_gap, axis=1)].copy()

    mask = pd.Series(False, index=candidate_universe.index)
    for field in TARGET_FIELDS:
        if field not in candidate_universe.columns:
            continue
        mask = mask | candidate_universe[field].apply(_blank)
    return candidate_universe[mask].copy()


def _load_candidates(
    scraped_fighters_path: Path,
    *,
    gap_audit_csv: Path | None = None,
    candidate_source_csv: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_universe = _build_candidate_universe(
        scraped_fighters_path,
        candidate_source_csv=candidate_source_csv,
    )
    candidates = _filter_gap_candidates(candidate_universe, gap_audit_csv=gap_audit_csv)
    return candidate_universe, candidates


def _load_existing_rows(output_path: Path) -> list[dict[str, object]]:
    if not output_path.exists():
        return []
    existing_df = pd.read_csv(output_path)
    if existing_df.empty:
        return []
    return existing_df.to_dict(orient="records")


def _build_effective_profile_state(
    candidate_universe: pd.DataFrame,
    existing_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    state: dict[str, dict[str, object]] = {}

    def _merge_row(row: dict[str, object] | pd.Series) -> None:
        name_key = normalize_person_name(row.get("name"))
        if not name_key:
            return
        profile = state.setdefault(name_key, {field: "" for field in TARGET_FIELDS})
        for field in TARGET_FIELDS:
            if _blank(profile.get(field)) and not _blank(row.get(field)):
                profile[field] = row.get(field)

    for _, row in candidate_universe.iterrows():
        _merge_row(row)
    for row in existing_rows:
        _merge_row(row)
    return state


def _normalize_existing_source_keys(existing_rows: list[dict[str, object]]) -> set[tuple[str, str]]:
    return {
        (normalize_person_name(row.get("name")), str(row.get("source", "")).strip().lower())
        for row in existing_rows
        if not _blank(row.get("name")) and not _blank(row.get("source"))
    }


def _build_base_row(
    fighter_name: str,
    *,
    source: str,
    source_name: str,
    search_name: str,
    fighter_url: str,
) -> dict[str, object]:
    return {
        "name": fighter_name,
        "source": source,
        "source_name": source_name,
        "search_name": search_name,
        "fighter_url": fighter_url,
        "height": "",
        "reach": "",
        "weight": "",
        "stance": "",
        "dob": "",
    }


def _search_names_for_source(
    row: pd.Series | dict[str, object],
    *,
    override_map: dict[str, str] | None = None,
) -> list[str]:
    names = _row_candidate_names(row)
    if override_map is None:
        return names

    search_names: list[str] = []
    for name in names:
        search_names.append(override_map.get(normalize_person_name(name), name))
    return _dedupe_names(search_names)


def _resolve_profile_url(
    row: pd.Series,
    search_fn,
    *,
    override_map: dict[str, str] | None = None,
) -> tuple[str, str]:
    for search_name in _search_names_for_source(row, override_map=override_map):
        fighter_url = search_fn(search_name)
        if fighter_url:
            return search_name, fighter_url
    return "", ""


def _profile_name_matches_candidate(
    row: pd.Series | dict[str, object],
    profile_name: object,
) -> bool:
    candidate_name = _clean_text(profile_name)
    if not candidate_name:
        return False
    return any(
        same_person_name(candidate_name, known_name)
        for known_name in _row_candidate_names(row)
    )


def _recover_profile_fields(
    supplement: dict[str, object],
    current_profile: dict[str, object],
    source_profile: dict[str, object],
    *,
    field_map: dict[str, str],
) -> bool:
    recovered_any = False
    for target_field, source_field in field_map.items():
        if not _blank(current_profile.get(target_field)):
            continue
        source_value = source_profile.get(source_field)
        if _blank(source_value):
            continue
        supplement[target_field] = source_value
        recovered_any = True
    return recovered_any


def _update_state_from_row(state: dict[str, dict[str, object]], row: dict[str, object]) -> None:
    fighter_key = normalize_person_name(row.get("name"))
    if not fighter_key:
        return
    profile = state.setdefault(fighter_key, {field: "" for field in TARGET_FIELDS})
    for field in TARGET_FIELDS:
        if _blank(profile.get(field)) and not _blank(row.get(field)):
            profile[field] = row.get(field)


def _source_has_recoverable_gap(
    source: str,
    fighter_key: str,
    current_state: dict[str, dict[str, object]],
) -> bool:
    recoverable_fields = SOURCE_RECOVERABLE_FIELDS.get(source, ())
    if not recoverable_fields:
        return False
    current_profile = current_state.setdefault(
        fighter_key,
        {field: "" for field in TARGET_FIELDS},
    )
    return any(_blank(current_profile.get(field)) for field in recoverable_fields)


def _build_source_row(
    source: str,
    session: requests.Session,
    row: pd.Series,
    current_state: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    if source == "martialbot":
        return _build_martialbot_row(row, current_state)
    if source == "espn":
        return _build_espn_row(row, current_state)
    if source == "wikipedia":
        return _build_wikipedia_row(session, row, current_state)
    if source == "fightdx":
        return _build_fightdx_row(row, current_state)
    if source == "tapology":
        return _build_tapology_row(row, current_state)
    if source == "sherdog":
        return _build_sherdog_row(row, current_state)
    return None


def _build_tapology_row(
    scraped_row: pd.Series,
    current_state: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    fighter_name = str(scraped_row.get("name", "") or "").strip()
    fighter_key = normalize_person_name(fighter_name)
    search_name, fighter_url = _resolve_profile_url(
        scraped_row,
        search_tapology,
        override_map=TAPOLOGY_SEARCH_NAME_OVERRIDES,
    )
    if not fighter_url:
        return None

    profile = scrape_tapology_profile(fighter_url)
    if not _profile_name_matches_candidate(scraped_row, profile.get("name")):
        return None
    current_profile = current_state.setdefault(fighter_key, {field: "" for field in TARGET_FIELDS})
    supplement = _build_base_row(
        fighter_name,
        source="tapology",
        source_name=str(profile.get("name", "") or ""),
        search_name=search_name,
        fighter_url=fighter_url,
    )

    recovered_any = _recover_profile_fields(
        supplement,
        current_profile,
        profile,
        field_map={
            "height": "height_raw",
            "reach": "reach_raw",
            "weight": "weight_raw",
            "dob": "dob",
        },
    )
    return supplement if recovered_any else None


def _build_sherdog_row(
    scraped_row: pd.Series,
    current_state: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    fighter_name = str(scraped_row.get("name", "") or "").strip()
    fighter_key = normalize_person_name(fighter_name)
    search_name, fighter_url = _resolve_profile_url(scraped_row, search_sherdog)
    if not fighter_url:
        return None

    profile, _fights = scrape_sherdog_page(fighter_url, search_name)
    if not _profile_name_matches_candidate(scraped_row, profile.get("name")):
        return None
    profile = dict(profile)
    height_value = profile.get("height")
    weight_value = profile.get("weight")
    if pd.isna(height_value) or float(height_value) <= 0:
        profile["height_raw"] = ""
    if pd.isna(weight_value) or float(weight_value) <= 0:
        profile["weight_raw"] = ""
    current_profile = current_state.setdefault(fighter_key, {field: "" for field in TARGET_FIELDS})
    supplement = _build_base_row(
        fighter_name,
        source="sherdog",
        source_name=str(profile.get("name", "") or ""),
        search_name=search_name,
        fighter_url=fighter_url,
    )

    recovered_any = _recover_profile_fields(
        supplement,
        current_profile,
        profile,
        field_map={
            "height": "height_raw",
            "weight": "weight_raw",
            "dob": "dob",
        },
    )
    return supplement if recovered_any else None


def _build_fightdx_row(
    scraped_row: pd.Series,
    current_state: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    fighter_name = str(scraped_row.get("name", "") or "").strip()
    fighter_key = normalize_person_name(fighter_name)
    search_name, fighter_url = _resolve_profile_url(scraped_row, search_fightdx)
    if not fighter_url:
        return None

    profile = scrape_fightdx_profile(fighter_url)
    if not _profile_name_matches_candidate(scraped_row, profile.get("name")):
        return None
    current_profile = current_state.setdefault(fighter_key, {field: "" for field in TARGET_FIELDS})
    supplement = _build_base_row(
        fighter_name,
        source="fightdx",
        source_name=str(profile.get("name", "") or ""),
        search_name=search_name,
        fighter_url=fighter_url,
    )

    recovered_any = _recover_profile_fields(
        supplement,
        current_profile,
        profile,
        field_map={
            "height": "height_raw",
            "reach": "reach_raw",
            "weight": "weight_raw",
            "stance": "stance",
            "dob": "dob",
        },
    )
    return supplement if recovered_any else None


def _build_espn_row(
    scraped_row: pd.Series,
    current_state: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    fighter_name = str(scraped_row.get("name", "") or "").strip()
    fighter_key = normalize_person_name(fighter_name)
    search_name, fighter_url = _resolve_profile_url(scraped_row, search_espn)
    if not fighter_url:
        return None

    profile = scrape_espn_profile(fighter_url)
    if not _profile_name_matches_candidate(scraped_row, profile.get("name")):
        return None
    current_profile = current_state.setdefault(fighter_key, {field: "" for field in TARGET_FIELDS})
    supplement = _build_base_row(
        fighter_name,
        source="espn",
        source_name=str(profile.get("name", "") or ""),
        search_name=search_name,
        fighter_url=fighter_url,
    )

    recovered_any = _recover_profile_fields(
        supplement,
        current_profile,
        profile,
        field_map={
            "height": "height_raw",
            "reach": "reach_raw",
            "weight": "weight_raw",
            "stance": "stance",
            "dob": "dob",
        },
    )
    return supplement if recovered_any else None


def _build_martialbot_row(
    scraped_row: pd.Series,
    current_state: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    fighter_name = str(scraped_row.get("name", "") or "").strip()
    fighter_key = normalize_person_name(fighter_name)
    search_name, fighter_url = _resolve_profile_url(
        scraped_row,
        search_martialbot,
        override_map=MARTIALBOT_SEARCH_NAME_OVERRIDES,
    )
    if not fighter_url:
        return None

    profile = scrape_martialbot_profile(fighter_url)
    if not _profile_name_matches_candidate(scraped_row, profile.get("name")):
        return None
    current_profile = current_state.setdefault(fighter_key, {field: "" for field in TARGET_FIELDS})
    supplement = _build_base_row(
        fighter_name,
        source="martialbot",
        source_name=str(profile.get("name", "") or ""),
        search_name=search_name,
        fighter_url=fighter_url,
    )

    recovered_any = _recover_profile_fields(
        supplement,
        current_profile,
        profile,
        field_map={
            "height": "height_raw",
            "reach": "reach_raw",
            "stance": "stance",
            "dob": "dob",
        },
    )
    return supplement if recovered_any else None


def _wikipedia_429_wait_seconds(response: requests.Response, attempt: int) -> float:
    retry_after = str(response.headers.get("Retry-After", "") or "").strip()
    if retry_after:
        try:
            return min(float(retry_after), WIKIPEDIA_429_MAX_BACKOFF_SECONDS)
        except ValueError:
            pass
    return min(
        WIKIPEDIA_429_BASE_BACKOFF_SECONDS * (2 ** max(0, attempt - 1)),
        WIKIPEDIA_429_MAX_BACKOFF_SECONDS,
    )


def _wiki_api(session: requests.Session, **params) -> dict:
    for attempt in range(1, WIKIPEDIA_429_MAX_ATTEMPTS + 1):
        response = session.get(WIKIPEDIA_API_URL, params=params, headers=WIKIPEDIA_HEADERS, timeout=30)
        if response.status_code == 429 and attempt < WIKIPEDIA_429_MAX_ATTEMPTS:
            wait_seconds = _wikipedia_429_wait_seconds(response, attempt)
            logger.info(
                "Wikipedia API returned 429; retrying in %.1fs (attempt %d/%d)",
                wait_seconds,
                attempt + 1,
                WIKIPEDIA_429_MAX_ATTEMPTS,
            )
            time.sleep(wait_seconds)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("unreachable Wikipedia API retry state")


def _wikipedia_title_matches_candidate(search_name: str, title: str) -> bool:
    raw_title = str(title or "").strip()
    base_title = re.sub(r"\s*\([^)]*\)\s*$", "", raw_title).strip()
    if not same_person_name(search_name, base_title or raw_title):
        return False
    qualifier_match = re.search(r"\(([^)]*)\)", raw_title)
    if not qualifier_match:
        return True
    qualifier = qualifier_match.group(1).strip().casefold()
    return any(token in qualifier for token in _WIKIPEDIA_ALLOWED_TITLE_QUALIFIERS)


def _wikipedia_find_title(session: requests.Session, fighter_name: str) -> str | None:
    search_name = fighter_name

    exact_page = _wiki_api(
        session,
        action="query",
        titles=search_name,
        prop="info",
        format="json",
        formatversion=2,
    )["query"]["pages"][0]
    if not exact_page.get("missing") and _wikipedia_title_matches_candidate(
        fighter_name,
        exact_page.get("title", ""),
    ):
        return exact_page.get("title")

    search_results = _wiki_api(
        session,
        action="query",
        list="search",
        srsearch=search_name,
        format="json",
        formatversion=2,
        srlimit=5,
    ).get("query", {}).get("search", [])

    normalized_search_name = normalize_person_name(search_name)
    for result in search_results:
        title = result.get("title", "")
        if _wikipedia_title_matches_candidate(search_name, title):
            return title
        candidate_key = normalize_person_name(title)
        if normalized_search_name and (
            normalized_search_name in candidate_key or candidate_key in normalized_search_name
        ) and _wikipedia_title_matches_candidate(search_name, title):
            return title
    return None


def _extract_infobox_field(wikitext: str, field: str) -> str:
    if not wikitext:
        return ""

    in_infobox = False
    brace_depth = 0
    for raw_line in wikitext.splitlines():
        line = raw_line.rstrip()
        if not in_infobox and line.startswith("{{Infobox"):
            in_infobox = True
        if not in_infobox:
            continue

        brace_depth += line.count("{{") - line.count("}}")
        if re.match(rf"^\|\s*{re.escape(field)}\s*=", line):
            return line.split("=", 1)[1].strip()
        if brace_depth <= 0:
            break
    return ""


def _parse_wikipedia_template_number(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    fraction_match = re.fullmatch(r"(\d+)\s*\+\s*(\d+)\s*/\s*(\d+)", text)
    if fraction_match:
        whole = float(fraction_match.group(1))
        numerator = float(fraction_match.group(2))
        denominator = float(fraction_match.group(3))
        if denominator:
            return whole + (numerator / denominator)
    try:
        return float(text)
    except ValueError:
        return None


def _parse_wikipedia_reach(raw_value: str) -> str:
    text = str(raw_value or "").strip()
    if _blank(text):
        return ""
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\{\{efn\|.*?\}\}", "", text, flags=re.IGNORECASE)
    text = text.replace("[[", "").replace("]]", "").replace("{{nbsp}}", " ")
    text = text.strip()

    convert_match = re.search(r"\{\{\s*convert\|([^}]*)\}\}", text, flags=re.IGNORECASE)
    if convert_match:
        parts = [part.strip() for part in convert_match.group(1).split("|") if part.strip()]
        if len(parts) >= 2:
            numeric = _parse_wikipedia_template_number(parts[0])
            unit = parts[1].lower()
            if numeric is not None and unit in {"in", "inch", "inches"}:
                return f"{numeric:g} in"
            if numeric is not None and unit == "cm":
                return f"{numeric:g} cm"

    metric_match = re.search(r"(\d+(?:\.\d+)?)\s*cm\b", text, flags=re.IGNORECASE)
    if metric_match:
        return f"{metric_match.group(1)} cm"
    imperial_match = re.search(r"(\d+(?:\.\d+)?)\s*in\b", text, flags=re.IGNORECASE)
    if imperial_match:
        return f"{imperial_match.group(1)} in"
    return ""


def _parse_wikipedia_stance(raw_value: str) -> str:
    text = str(raw_value or "").strip()
    if _blank(text):
        return ""
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\{\{efn\|.*?\}\}", "", text, flags=re.IGNORECASE)
    text = text.replace("[[", "").replace("]]", "").replace("{{nbsp}}", " ")
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    lowered = text.casefold()
    stance_labels = (
        ("Open Stance", "open stance"),
        ("Southpaw", "southpaw"),
        ("Orthodox", "orthodox"),
        ("Switch", "switch"),
        ("Sideways", "sideways"),
    )
    for label, token in stance_labels:
        if token in lowered:
            return label
    return ""


def _parse_wikipedia_birth_date(raw_value: str) -> str:
    text = str(raw_value or "").strip()
    if _blank(text):
        return ""

    template_match = re.search(r"\{\{\s*birth[^|}]*\|(.+?)\}\}", text, flags=re.IGNORECASE)
    if template_match:
        parts = [part.strip() for part in template_match.group(1).split("|") if part.strip()]
        named: dict[str, int] = {}
        numeric_parts: list[int] = []
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip().lower()
                value = value.strip()
                if key in {"year", "month", "day"} and value.isdigit():
                    named[key] = int(value)
                continue
            if part.isdigit():
                numeric_parts.append(int(part))

        year = named.get("year")
        month = named.get("month")
        day = named.get("day")
        if year and month and day:
            return f"{year:04d}-{month:02d}-{day:02d}"
        if len(numeric_parts) >= 3:
            return f"{numeric_parts[0]:04d}-{numeric_parts[1]:02d}-{numeric_parts[2]:02d}"
        return ""

    plain_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if plain_match:
        return plain_match.group(0)
    return ""


def _build_wikipedia_row(
    session: requests.Session,
    scraped_row: pd.Series,
    current_state: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    fighter_name = str(scraped_row.get("name", "") or "").strip()
    fighter_key = normalize_person_name(fighter_name)
    title = None
    search_name = ""
    for search_name_candidate in _search_names_for_source(scraped_row):
        title = _wikipedia_find_title(session, search_name_candidate)
        if title:
            search_name = search_name_candidate
            break
    if not title:
        return None

    page = _wiki_api(
        session,
        action="query",
        titles=title,
        prop="revisions",
        rvprop="content",
        format="json",
        formatversion=2,
    )["query"]["pages"][0]
    wikitext = page.get("revisions", [{}])[0].get("content", "")
    reach = _parse_wikipedia_reach(_extract_infobox_field(wikitext, "reach"))
    stance = _parse_wikipedia_stance(_extract_infobox_field(wikitext, "stance"))
    dob = _parse_wikipedia_birth_date(_extract_infobox_field(wikitext, "birth_date"))

    current_profile = current_state.setdefault(fighter_key, {field: "" for field in TARGET_FIELDS})
    supplement = _build_base_row(
        fighter_name,
        source="wikipedia",
        source_name=title,
        search_name=search_name,
        fighter_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
    )

    recovered_any = _recover_profile_fields(
        supplement,
        current_profile,
        {"reach": reach, "stance": stance, "dob": dob},
        field_map={"reach": "reach", "stance": "stance", "dob": "dob"},
    )
    return supplement if recovered_any else None


def run_profile_supplement_refresh(
    *,
    scraped_fighters_path: Path = DEFAULT_INPUT,
    gap_audit_csv: Path | None = None,
    candidate_source_csv: Path | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    sources: list[str] | tuple[str, ...] | None = None,
    limit: int | None = None,
) -> dict[str, object]:
    """Refresh the supplemental profile artifact and return a structured summary."""
    clear_fallback_cache()
    candidate_universe, candidates = _load_candidates(
        scraped_fighters_path,
        gap_audit_csv=gap_audit_csv,
        candidate_source_csv=candidate_source_csv,
    )
    candidates = candidates.sort_values("name").reset_index(drop=True)
    if limit is not None:
        candidates = candidates.head(limit).copy()

    existing_rows = _load_existing_rows(output_path)
    existing_source_keys = _normalize_existing_source_keys(existing_rows)
    current_state = _build_effective_profile_state(candidate_universe, existing_rows)

    results: list[dict[str, object]] = []
    selected_sources = tuple(dict.fromkeys(sources or ALL_SOURCES))
    source_recoveries = {source: 0 for source in ALL_SOURCES}
    attempted = 0

    with requests.Session() as session:
        for _, row in candidates.iterrows():
            attempted += 1
            fighter_key = normalize_person_name(row.get("name"))
            if not fighter_key:
                continue

            for source in selected_sources:
                if (fighter_key, source) in existing_source_keys:
                    continue
                if not _source_has_recoverable_gap(source, fighter_key, current_state):
                    continue
                try:
                    source_row = _build_source_row(source, session, row, current_state)
                except Exception as exc:
                    logger.warning(
                        "%s profile lookup failed for '%s': %s",
                        SOURCE_LOG_LABELS.get(source, source),
                        row.get("name"),
                        exc,
                    )
                    source_row = None
                if source_row is None:
                    continue
                results.append(source_row)
                existing_source_keys.add((fighter_key, source))
                _update_state_from_row(current_state, source_row)
                source_recoveries[source] += 1

    combined_rows = existing_rows + results
    output_df = pd.DataFrame(combined_rows)
    if not output_df.empty:
        output_df = output_df.sort_values(["name", "source"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    return {
        "candidate_universe_rows": int(len(candidate_universe)),
        "candidate_rows": int(len(candidates)),
        "attempted_rows": attempted,
        "recovered_rows": int(len(results)),
        "selected_sources": list(selected_sources),
        "recovered_by_source": source_recoveries,
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scraped-fighters-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--gap-audit-csv", type=Path, default=None)
    parser.add_argument(
        "--candidate-source-csv",
        type=Path,
        default=None,
        help=(
            "Optional source CSV with fighter names to target. Supports active-roster-style "
            "columns like official_name, ufcstats_name, profile_name, slug_name, "
            "alternate_slug_names, and any already-known profile fields such as weight."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=ALL_SOURCES,
        default=list(ALL_SOURCES),
        help="Subset of profile sources to run. Defaults to all supported sources.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = run_profile_supplement_refresh(
        scraped_fighters_path=args.scraped_fighters_path,
        gap_audit_csv=args.gap_audit_csv,
        candidate_source_csv=args.candidate_source_csv,
        output_path=args.output,
        sources=args.sources,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Candidates: {summary['candidate_rows']}")
    print(f"Attempted: {summary['attempted_rows']}")
    print(f"Recovered: {summary['recovered_rows']}")
    print(f"Recovered by source: {summary['recovered_by_source']}")
    print(f"Output: {summary['output_path']}")


if __name__ == "__main__":
    main()
