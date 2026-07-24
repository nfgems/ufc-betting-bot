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
    TapologyRequestError,
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
    search_tapology_candidates,
)
from src.data.name_utils import normalize_person_name, same_person_name  # noqa: E402
from src.data.ufc_active_roster import (  # noqa: E402
    _apply_profile_status_eligibility,
    _explicit_boolean,
    _row_has_power_slap_evidence,
)
from src.features.stance_utils import encode_stance  # noqa: E402

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
    "abdul azeem badakhshi": "Abdul Azim Badakhshi",
    "felix lee mitchell": "Felix Mitchell",
}
_TAPOLOGY_READER_RUNTIME_BLOCK_STATUSES = {401, 403, 451}
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


def _valid_stance(value: object) -> bool:
    if _blank(value):
        return False
    return bool(pd.notna(encode_stance(value)))


def _valid_dob(value: object) -> bool:
    if _blank(value):
        return False
    return bool(pd.notna(pd.to_datetime(value, errors="coerce", format="mixed")))


def _field_present(field: str, value: object) -> bool:
    if field == "stance":
        return _valid_stance(value)
    if field == "dob":
        return _valid_dob(value)
    return not _blank(value)


def _normalized_field_value(field: str, value: object) -> object:
    if field in TARGET_FIELDS and not _field_present(field, value):
        return ""
    return value


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
    if trusted_official_url_identity:
        for column in ("canonical_athlete_url", "official_athlete_url"):
            url = str(row.get(column) or "").strip()
            slug = urlparse(url).path.strip("/").rsplit("/", 1)[-1] if url else ""
            if slug:
                names.append(slug.replace("-", " "))
    for column in _MULTI_NAME_COLUMNS:
        if column == "alternate_slug_names" and not trusted_official_url_identity:
            continue
        names.extend(str(row.get(column) or "").split("|"))
    return _dedupe_names(names)


def _official_url_identity_trusted(row: pd.Series | dict[str, object]) -> bool:
    status = (
        str(row.get("official_url_identity_status") or "")
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )
    if status in {"mismatch", "test profile"}:
        return False
    explicit = row.get("official_url_identity_valid")
    if _blank(explicit):
        return True
    return _explicit_boolean(explicit) is not False


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
    if scraped_row is not None and _field_present(field, scraped_row.get(field)):
        return scraped_row.get(field)
    if field in source_row and _field_present(field, source_row.get(field)):
        return source_row.get(field)
    return ""


def _candidate_refresh_priority(row: dict[str, object]) -> int:
    """Read only an explicit priority supplied by a current, trusted caller.

    Date-shaped roster columns are intentionally not inferred here: stale
    ``next_event_date`` values previously looked "upcoming" forever. The hosted
    workflow now joins live UFC card identities and writes this explicit field.
    """
    explicit = row.get("profile_refresh_priority")
    if not _blank(explicit):
        try:
            return max(int(float(explicit)), 0)
        except (TypeError, ValueError):
            pass
    return 0


def _candidate_source_row_is_coverage_eligible(row: dict[str, object]) -> bool:
    if _is_test_or_staging_profile(row):
        return False
    roster_shaped = any(
        column in row
        for column in (
            "official_name",
            "official_athlete_url",
            "profile_status",
            "active_roster_status_reason",
        )
    )
    if roster_shaped:
        normalized = _apply_profile_status_eligibility(
            pd.DataFrame([row])
        ).iloc[0]
        return (
            _explicit_boolean(normalized.get("coverage_eligible")) is True
        )
    if _row_has_power_slap_evidence(row):
        return False
    coverage_eligible = row.get("coverage_eligible")
    if not _blank(coverage_eligible):
        return _explicit_boolean(coverage_eligible) is True
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
                "_refresh_priority": 0,
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
        merged_row["_refresh_priority"] = max(
            int(merged_row.get("_refresh_priority") or 0),
            _candidate_refresh_priority(source_dict),
        )

        for field in TARGET_FIELDS:
            if not _field_present(field, merged_row.get(field)):
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
        mask = mask | ~candidate_universe[field].apply(lambda value: _field_present(field, value))
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
            if not _field_present(field, profile.get(field)) and _field_present(field, row.get(field)):
                profile[field] = row.get(field)

    for _, row in candidate_universe.iterrows():
        _merge_row(row)
    for row in existing_rows:
        _merge_row(row)
    return state


def _source_row_key(row: dict[str, object]) -> tuple[str, str]:
    return (normalize_person_name(row.get("name")), str(row.get("source", "")).strip().lower())


def _merge_source_rows(existing_rows: list[dict[str, object]], recovered_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows_by_source: dict[tuple[str, str], dict[str, object]] = {}
    ordered_keys: list[tuple[str, str]] = []

    def _merge(row: dict[str, object]) -> None:
        key = _source_row_key(row)
        if not key[0] or not key[1]:
            key = (f"__row_{len(ordered_keys)}", "")
        clean_row = dict(row)
        for field in TARGET_FIELDS:
            clean_row[field] = _normalized_field_value(field, clean_row.get(field))
        target = rows_by_source.get(key)
        if target is None:
            rows_by_source[key] = clean_row
            ordered_keys.append(key)
            return
        for field, value in clean_row.items():
            if field in TARGET_FIELDS:
                if not _field_present(field, target.get(field)) and _field_present(field, value):
                    target[field] = value
                continue
            if _blank(target.get(field)) and not _blank(value):
                target[field] = value

    for row in existing_rows:
        _merge(row)
    for row in recovered_rows:
        _merge(row)
    return [rows_by_source[key] for key in ordered_keys]


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


def _tapology_profile_candidates(
    row: pd.Series,
    *,
    limit_per_name: int = 5,
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen_urls: set[str] = set()
    discovery_reports: list[dict[str, object]] = []
    for search_name in _search_names_for_source(
        row,
        override_map=TAPOLOGY_SEARCH_NAME_OVERRIDES,
    ):
        discovery: dict[str, object] = {}
        fighter_urls = search_tapology_candidates(
            search_name,
            limit=limit_per_name,
            diagnostics=discovery,
        )
        discovery_reports.append(discovery)
        for fighter_url in fighter_urls:
            if fighter_url in seen_urls:
                continue
            seen_urls.add(fighter_url)
            candidates.append((search_name, fighter_url))
    if not candidates and discovery_reports and not any(
        bool(report.get("healthy")) for report in discovery_reports
    ):
        details = "; ".join(
            str(report.get("attempts") or "no usable discovery channel")
            for report in discovery_reports
        )
        raise TapologyRequestError(
            str(row.get("name") or "Tapology profile discovery"),
            detail=f"Tapology discovery unavailable: {details}",
        )
    return candidates


def _tapology_error_is_runtime_reader_block(exc: TapologyRequestError) -> bool:
    detail = str(getattr(exc, "detail", "") or "").lower()
    return (
        "reader" in detail
        and (
            getattr(exc, "status_code", None) in _TAPOLOGY_READER_RUNTIME_BLOCK_STATUSES
            or detail in {"reader fallback disabled", "reader fallback unavailable"}
            or "reader circuit open" in detail
        )
    )


def _tapology_error_is_source_access_failure(exc: TapologyRequestError) -> bool:
    status_code = getattr(exc, "status_code", None)
    detail = str(getattr(exc, "detail", "") or "").casefold()
    return (
        status_code in {401, 403, 429, 451, 500, 502, 503, 504}
        or _tapology_error_is_runtime_reader_block(exc)
        or "cloudflare" in detail
        or "blocked from this environment" in detail
        or "request failed" in detail
        or "browser fallback failed" in detail
        or (status_code is None and "non-profile tapology content" not in detail)
    )


def _profile_name_matches_candidate(
    row: pd.Series | dict[str, object],
    profile_name: object,
    fighter_url: str = "",
) -> bool:
    candidate_name = _clean_text(profile_name)
    if not candidate_name:
        return False
    return any(
        same_person_name(candidate_name, known_name)
        or _profile_identity_supported(known_name, candidate_name, fighter_url)
        for known_name in _row_candidate_names(row)
    )


def _profile_identity_supported(query_name: object, profile_name: object, fighter_url: str = "") -> bool:
    query_tokens = normalize_person_name(query_name).split()
    if not query_tokens:
        return False

    variants = [_clean_text(profile_name)]
    slug = urlparse(str(fighter_url or "")).path.strip("/").rsplit("/", 1)[-1]
    if slug:
        variants.append(re.sub(r"^\d+-", "", slug).replace("-", " "))

    for variant in variants:
        candidate_tokens = normalize_person_name(variant).split()
        if not candidate_tokens:
            continue
        if query_tokens == candidate_tokens or query_tokens == list(reversed(candidate_tokens)):
            return True
        if len(query_tokens) >= 2 and query_tokens[0] in candidate_tokens and query_tokens[-1] in candidate_tokens:
            return True
    return False


def _recover_profile_fields(
    supplement: dict[str, object],
    current_profile: dict[str, object],
    source_profile: dict[str, object],
    *,
    field_map: dict[str, str],
) -> bool:
    recovered_any = False
    for target_field, source_field in field_map.items():
        if _field_present(target_field, current_profile.get(target_field)):
            continue
        source_value = source_profile.get(source_field)
        if not _field_present(target_field, source_value):
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
        if not _field_present(field, profile.get(field)) and _field_present(field, row.get(field)):
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
    return any(not _field_present(field, current_profile.get(field)) for field in recoverable_fields)


def _candidate_has_recoverable_gap(
    row: pd.Series,
    selected_sources: tuple[str, ...],
    current_state: dict[str, dict[str, object]],
) -> bool:
    fighter_key = normalize_person_name(row.get("name"))
    if not fighter_key:
        return False
    return any(
        _source_has_recoverable_gap(source, fighter_key, current_state)
        for source in selected_sources
    )


def _order_refresh_candidates(candidates: pd.DataFrame, *, offset: int = 0) -> pd.DataFrame:
    """Honor explicit current-card priority, then rotate the stable backlog."""
    if candidates.empty:
        return candidates.copy()
    ordered = candidates.copy()
    if "_refresh_priority" not in ordered.columns:
        ordered["_refresh_priority"] = 0
    ordered["_refresh_priority"] = pd.to_numeric(
        ordered["_refresh_priority"], errors="coerce"
    ).fillna(0)
    ordered = ordered.sort_values(
        ["_refresh_priority", "name"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    priority = ordered[ordered["_refresh_priority"] > 0]
    backlog = ordered[ordered["_refresh_priority"] <= 0].reset_index(drop=True)
    if not backlog.empty:
        normalized_offset = max(int(offset or 0), 0) % len(backlog)
        if normalized_offset:
            backlog = pd.concat(
                [backlog.iloc[normalized_offset:], backlog.iloc[:normalized_offset]],
                ignore_index=True,
            )
    return pd.concat([priority, backlog], ignore_index=True)


def _select_refresh_candidates(
    candidates: pd.DataFrame,
    *,
    limit: int | None,
    offset: int = 0,
    rotation_index: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Reserve space for both candidate classes, then rotate within each class.

    ``rotation_index`` is deliberately converted to a row offset only after the
    priority/backlog allocation is known. This prevents either class from
    starving and keeps each hosted stride equal to the rows actually attempted
    from that class in each run.
    """
    ordered = _order_refresh_candidates(candidates, offset=0)
    if ordered.empty:
        return ordered, {
            "priority_slots": 0,
            "backlog_slots": 0,
            "priority_rotation_stride": 0,
            "backlog_rotation_stride": 0,
            "effective_priority_offset": 0,
            "effective_backlog_offset": 0,
        }

    priority = ordered[ordered["_refresh_priority"] > 0].reset_index(drop=True)
    backlog = ordered[ordered["_refresh_priority"] <= 0].reset_index(drop=True)
    normalized_offset = max(int(offset or 0), 0)
    single_slot_uses_explicit_offset = False

    if limit is None:
        priority_slots = len(priority)
        backlog_slots = len(backlog)
        priority_turn_index = max(int(rotation_index or 0), 0)
        backlog_turn_index = priority_turn_index
    else:
        normalized_limit = max(int(limit), 0)
        normalized_rotation_index = max(int(rotation_index or 0), 0)
        priority_turn_index = normalized_rotation_index
        backlog_turn_index = normalized_rotation_index
        if normalized_limit == 1 and not priority.empty and not backlog.empty:
            # A one-row run cannot contain both classes. Either rotation input
            # acts as the deterministic turn number; this preserves the CLI's
            # explicit-offset path instead of pinning every run to priority.
            turn_index = (
                normalized_rotation_index
                if rotation_index is not None
                else normalized_offset
            )
            single_slot_uses_explicit_offset = rotation_index is None
            if turn_index % 2 == 0:
                priority_slots = 1
                backlog_slots = 0
                priority_turn_index = turn_index // 2
            else:
                priority_slots = 0
                backlog_slots = 1
                backlog_turn_index = turn_index // 2
        elif normalized_limit == 0:
            priority_slots = 0
            backlog_slots = 0
        elif not priority.empty and not backlog.empty:
            # Explicitly prioritized fighters retain every available slot except the one
            # required to ensure the ordinary backlog continues making progress.
            priority_slots = min(len(priority), normalized_limit - 1)
            backlog_slots = min(len(backlog), normalized_limit - priority_slots)
        elif not priority.empty:
            priority_slots = min(len(priority), normalized_limit)
            backlog_slots = 0
        else:
            priority_slots = 0
            backlog_slots = min(len(backlog), normalized_limit)

    effective_priority_offset = 0
    effective_offset = normalized_offset
    if rotation_index is not None:
        effective_priority_offset = priority_turn_index * priority_slots
        effective_offset += backlog_turn_index * backlog_slots
    elif single_slot_uses_explicit_offset:
        effective_priority_offset = priority_turn_index * priority_slots
        effective_offset = backlog_turn_index * backlog_slots
    if not priority.empty and priority_slots:
        effective_priority_offset %= len(priority)
        if effective_priority_offset:
            priority = pd.concat(
                [
                    priority.iloc[effective_priority_offset:],
                    priority.iloc[:effective_priority_offset],
                ],
                ignore_index=True,
            )
    else:
        effective_priority_offset = 0
    if not backlog.empty and backlog_slots:
        effective_offset %= len(backlog)
        if effective_offset:
            backlog = pd.concat(
                [backlog.iloc[effective_offset:], backlog.iloc[:effective_offset]],
                ignore_index=True,
            )
    else:
        effective_offset = 0

    selected = pd.concat(
        [priority.head(priority_slots), backlog.head(backlog_slots)],
        ignore_index=True,
    )
    return selected, {
        "priority_slots": int(priority_slots),
        "backlog_slots": int(backlog_slots),
        "priority_rotation_stride": int(priority_slots),
        "backlog_rotation_stride": int(backlog_slots),
        "effective_priority_offset": int(effective_priority_offset),
        "effective_backlog_offset": int(effective_offset),
    }


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
    tapology_candidates = _tapology_profile_candidates(scraped_row)
    if not tapology_candidates:
        return None

    for search_name, fighter_url in tapology_candidates:
        try:
            profile = scrape_tapology_profile(fighter_url)
        except TapologyRequestError as exc:
            logger.info(
                "Tapology profile candidate rejected for '%s' (%s): %s",
                fighter_name,
                fighter_url,
                exc,
            )
            if _tapology_error_is_source_access_failure(exc):
                raise
            continue
        if not (
            _profile_name_matches_candidate(scraped_row, profile.get("name"), fighter_url)
            or same_person_name(search_name, profile.get("name"))
            or _profile_identity_supported(search_name, profile.get("name"), fighter_url)
        ):
            logger.info(
                "Tapology profile candidate name mismatch for '%s' (%s): %s",
                fighter_name,
                fighter_url,
                profile.get("name"),
            )
            continue
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
        if recovered_any:
            return supplement
    return None


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
    candidate_offset: int = 0,
    candidate_rotation_index: int | None = None,
) -> dict[str, object]:
    """Refresh the supplemental profile artifact and return a structured summary."""
    clear_fallback_cache(preserve_environment_blocks=True)
    candidate_universe, candidates = _load_candidates(
        scraped_fighters_path,
        gap_audit_csv=gap_audit_csv,
        candidate_source_csv=candidate_source_csv,
    )
    existing_rows = _load_existing_rows(output_path)
    current_state = _build_effective_profile_state(candidate_universe, existing_rows)
    selected_sources = tuple(dict.fromkeys(sources or ALL_SOURCES))
    gap_candidate_rows = int(len(candidates))
    candidates = candidates[
        candidates.apply(
            _candidate_has_recoverable_gap,
            axis=1,
            args=(selected_sources, current_state),
        )
    ].copy()
    recoverable_candidate_rows = int(len(candidates))
    candidates, selection = _select_refresh_candidates(
        candidates,
        limit=limit,
        offset=candidate_offset,
        rotation_index=candidate_rotation_index,
    )

    results: list[dict[str, object]] = []
    source_recoveries = {source: 0 for source in ALL_SOURCES}
    source_errors = {source: 0 for source in ALL_SOURCES}
    source_error_details: list[dict[str, str]] = []
    attempted = 0

    with requests.Session() as session:
        total_candidates = int(len(candidates))
        for index, (_, row) in enumerate(candidates.iterrows(), start=1):
            attempted += 1
            fighter_key = normalize_person_name(row.get("name"))
            if not fighter_key:
                continue
            logger.info(
                "Profile supplement refresh candidate %d/%d: %s",
                index,
                total_candidates,
                row.get("name"),
            )

            for source in selected_sources:
                if not _source_has_recoverable_gap(source, fighter_key, current_state):
                    continue
                try:
                    source_row = _build_source_row(source, session, row, current_state)
                except Exception as exc:
                    source_errors[source] += 1
                    if len(source_error_details) < 25:
                        source_error_details.append(
                            {
                                "source": source,
                                "fighter": str(row.get("name") or ""),
                                "error": str(exc),
                            }
                        )
                    log_fn = (
                        logger.info
                        if getattr(exc, "external_source_alert_reported", False)
                        else logger.warning
                    )
                    log_fn(
                        "%s profile lookup failed for '%s': %s",
                        SOURCE_LOG_LABELS.get(source, source),
                        row.get("name"),
                        exc,
                    )
                    source_row = None
                if source_row is None:
                    continue
                results.append(source_row)
                _update_state_from_row(current_state, source_row)
                source_recoveries[source] += 1

    combined_rows = _merge_source_rows(existing_rows, results)
    output_df = pd.DataFrame(combined_rows)
    if not output_df.empty:
        output_df = output_df.sort_values(["name", "source"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    return {
        "candidate_universe_rows": int(len(candidate_universe)),
        "gap_candidate_rows": gap_candidate_rows,
        "recoverable_candidate_rows": recoverable_candidate_rows,
        "candidate_rows": int(len(candidates)),
        "candidate_offset": max(int(candidate_offset or 0), 0),
        "candidate_rotation_index": (
            None
            if candidate_rotation_index is None
            else max(int(candidate_rotation_index), 0)
        ),
        **selection,
        "candidate_names": [str(name) for name in candidates.get("name", [])],
        "attempted_rows": attempted,
        "recovered_rows": int(len(results)),
        "selected_sources": list(selected_sources),
        "recovered_by_source": source_recoveries,
        "source_errors": source_errors,
        "source_error_count": int(sum(source_errors.values())),
        "source_error_details": source_error_details,
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
    parser.add_argument(
        "--candidate-offset",
        type=int,
        default=0,
        help="Rotate the non-priority recoverable backlog before applying --limit.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = run_profile_supplement_refresh(
        scraped_fighters_path=args.scraped_fighters_path,
        gap_audit_csv=args.gap_audit_csv,
        candidate_source_csv=args.candidate_source_csv,
        output_path=args.output,
        sources=args.sources,
        limit=args.limit,
        candidate_offset=args.candidate_offset,
    )
    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Candidates: {summary['candidate_rows']}")
    print(f"Attempted: {summary['attempted_rows']}")
    print(f"Recovered: {summary['recovered_rows']}")
    print(f"Recovered by source: {summary['recovered_by_source']}")
    print(f"Source errors: {summary['source_errors']}")
    print(f"Output: {summary['output_path']}")


if __name__ == "__main__":
    main()
