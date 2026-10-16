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
from contextlib import nullcontext
import json
import logging
import re
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import RAW_DATA_DIR  # noqa: E402
from src.data.fallback_scrapers import (  # noqa: E402
    ESPNSourceUnavailableError,
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
from src.data.io_utils import write_csv_atomically  # noqa: E402
from src.data.name_utils import (  # noqa: E402
    normalize_person_name,
    normalize_ufcstats_id,
    same_person_name,
)
from src.data.profile_integrity import (  # noqa: E402
    _profile_value,
    load_profile_decisions,
    source_entity_id,
    verified_binding_keys,
)
from src.data.ufc_active_roster import (  # noqa: E402
    _apply_profile_status_eligibility,
    _explicit_boolean,
    _official_url_identity_positively_trusted,
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


class ProfileIdentityUnverifiedError(RuntimeError):
    """A discovered source page lacks a positive stable-identity binding."""


class RetainedProfileProcessingError(RuntimeError):
    """Retained input/processing refusal; never a source outage or retry request."""


class ProfileFieldAuthorizationError(ProfileIdentityUnverifiedError):
    """A fetched field is absent from or differs from its reviewed decision."""


def _scalar_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if getattr(missing, "ndim", 0) != 0:
        return False
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _blank(value: object) -> bool:
    if _scalar_missing(value):
        return True
    return str(value).strip() in {"", "-", "--", "N/A", "nan", "??"}


def _clean_text(value: object) -> str:
    if _scalar_missing(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


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
            url = _clean_text(row.get(column))
            slug = urlparse(url).path.strip("/").rsplit("/", 1)[-1] if url else ""
            if slug:
                names.append(slug.replace("-", " "))
    for column in _MULTI_NAME_COLUMNS:
        if column == "alternate_slug_names" and not trusted_official_url_identity:
            continue
        names.extend(_clean_text(row.get(column)).split("|"))
    return _dedupe_names(names)


def _official_url_identity_trusted(row: pd.Series | dict[str, object]) -> bool:
    return _official_url_identity_positively_trusted(row)


def _stable_ufcstats_id_from_row(row: pd.Series | dict[str, object]) -> str:
    carriers: list[tuple[str, str]] = []
    for column in ("stable_ufcstats_id", "fighter_id", "ufcstats_url"):
        raw_value = row.get(column)
        if _blank(raw_value):
            continue
        stable_id = normalize_ufcstats_id(raw_value)
        if not stable_id:
            raise ProfileIdentityUnverifiedError(
                f"Malformed UFCStats identity carrier {column}={_clean_text(raw_value)!r}"
            )
        carriers.append((column, stable_id))

    fighter_url = _clean_text(row.get("fighter_url"))
    if fighter_url:
        stable_id = normalize_ufcstats_id(fighter_url)
        if stable_id:
            carriers.append(("fighter_url", stable_id))
        else:
            parsed = urlparse(fighter_url)
            hostname = (parsed.hostname or "").casefold().rstrip(".")
            if hostname == "ufcstats.com" or hostname.endswith(".ufcstats.com"):
                raise ProfileIdentityUnverifiedError(
                    f"Malformed UFCStats identity carrier fighter_url={fighter_url!r}"
                )

    stable_ids = {stable_id for _column, stable_id in carriers}
    if len(stable_ids) > 1:
        detail = ", ".join(f"{column}={stable_id}" for column, stable_id in carriers)
        raise ProfileIdentityUnverifiedError(
            f"Conflicting UFCStats identity carriers: {detail}"
        )
    return next(iter(stable_ids), "")


def _profile_state_key(row: pd.Series | dict[str, object]) -> str:
    """Keep profile state isolated by stable identity whenever one is known."""

    stable_id = _stable_ufcstats_id_from_row(row)
    if stable_id:
        return f"ufcstats:{stable_id}"
    return normalize_person_name(row.get("name"))


def _normalized_url(value: object) -> str:
    return _clean_text(value).rstrip("/")


def _captured_redirect_chain(value: object) -> tuple[str, ...] | None:
    """Normalize an explicitly captured redirect chain; None means unavailable."""

    if _scalar_missing(value):
        return None
    parsed: object = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            return None
    if not isinstance(parsed, (list, tuple)):
        return None
    urls: list[str] = []
    for item in parsed:
        if not isinstance(item, str) or not item.strip():
            return None
        url = _normalized_url(item)
        if urlparse(url).scheme.casefold() not in {"http", "https"}:
            return None
        urls.append(url)
    return tuple(urls)


def _verified_binding_metadata(
    decisions: pd.DataFrame,
    verified_bindings: set[tuple[str, str, str]],
) -> dict[tuple[str, str, str], dict[str, object]]:
    """Retain the reviewed URL/redirect identity behind each positive binding."""

    metadata: dict[tuple[str, str, str], dict[str, object]] = {}
    for _, row in decisions.iterrows():
        stable_id = normalize_ufcstats_id(row.get("stable_ufcstats_id")) or ""
        for prefix in ("primary", "corroborating"):
            source = _clean_text(row.get(f"{prefix}_source")).casefold().replace(
                ".com", ""
            )
            entity_id = _clean_text(row.get(f"{prefix}_source_entity_id"))
            key = (stable_id, source, entity_id)
            if key not in verified_bindings:
                continue
            candidate = {
                "requested_url": _normalized_url(
                    row.get(f"{prefix}_requested_url")
                ),
                "effective_url": _normalized_url(row.get(f"{prefix}_source_url")),
                "canonical_url": _normalized_url(
                    row.get(f"{prefix}_canonical_url")
                ),
                "redirect_chain": _captured_redirect_chain(
                    row.get(f"{prefix}_redirect_chain")
                ),
            }
            if not any(
                candidate.get(key)
                for key in ("requested_url", "effective_url", "canonical_url")
            ) and candidate.get("redirect_chain") is None:
                # Unit callers may inject a reviewed binding independently of
                # response-route metadata. Real loaded decision artifacts have
                # already passed the complete route validation.
                continue
            previous = metadata.get(key)
            if previous is not None and previous != candidate:
                raise ValueError(
                    "Verified profile decisions contain conflicting response metadata "
                    f"for {source}:{entity_id} bound to UFCStats {stable_id}"
                )
            metadata[key] = candidate
    return metadata


def _reviewed_field_authorizations(
    decisions: pd.DataFrame,
    verified_bindings: set[tuple[str, str, str]],
) -> dict[tuple[str, str, str, str], str]:
    """Pin each importable field to its exact reviewed normalized observation."""

    authorizations: dict[tuple[str, str, str, str], str] = {}
    immutable_fields = {"height", "reach", "stance", "dob"}
    for _, row in decisions.iterrows():
        stable_id = normalize_ufcstats_id(row.get("stable_ufcstats_id")) or ""
        field = _clean_text(row.get("field")).casefold()
        status = _clean_text(row.get("decision_status")).casefold()
        if not (
            (field in immutable_fields and status == "accepted")
            or (field == "weight" and status == "current_only")
        ):
            continue
        for prefix in ("primary", "corroborating"):
            source = _clean_text(row.get(f"{prefix}_source")).casefold().replace(
                ".com", ""
            )
            entity_id = _clean_text(row.get(f"{prefix}_source_entity_id"))
            binding = (stable_id, source, entity_id)
            if binding not in verified_bindings:
                continue
            normalized_value = _clean_text(
                row.get(f"{prefix}_normalized_value")
            )
            token = _profile_value(field, normalized_value)
            if not token:
                continue
            key = (*binding, field)
            previous = authorizations.get(key)
            if previous is not None and previous != token:
                raise ValueError(
                    "Reviewed profile decisions contain conflicting normalized "
                    f"values for {source}:{entity_id} {field} bound to UFCStats {stable_id}"
                )
            authorizations[key] = token
    return authorizations


def _reviewed_binding_name_keys(
    decisions: pd.DataFrame,
    verified_bindings: set[tuple[str, str, str]],
) -> dict[tuple[str, str, str], set[str]]:
    """Return suffix-preserving exact/reviewer-approved names per binding."""

    names_by_binding: dict[tuple[str, str, str], set[str]] = {}
    for _, row in decisions.iterrows():
        stable_id = normalize_ufcstats_id(row.get("stable_ufcstats_id")) or ""
        canonical_name = _clean_text(row.get("canonical_name"))
        for prefix in ("primary", "corroborating"):
            source = _clean_text(row.get(f"{prefix}_source")).casefold().replace(
                ".com", ""
            )
            entity_id = _clean_text(row.get(f"{prefix}_source_entity_id"))
            binding = (stable_id, source, entity_id)
            if binding not in verified_bindings:
                continue
            match_status = _clean_text(
                row.get(f"{prefix}_name_match_status")
            ).casefold()
            if match_status not in {"exact", "approved_alias"}:
                continue
            reviewed_names = [canonical_name]
            reviewed_names.extend(
                [
                    _clean_text(row.get(f"{prefix}_display_name")),
                    _clean_text(row.get(f"{prefix}_api_name")),
                ]
            )
            keys = {
                normalize_person_name(name)
                for name in reviewed_names
                if _clean_text(name)
            }
            if keys:
                names_by_binding.setdefault(binding, set()).update(keys)
    return names_by_binding


def _authorize_recovered_source_fields(
    source_row: dict[str, object],
    *,
    stable_id: str,
    source: str,
    entity_id: str,
    authorizations: dict[tuple[str, str, str, str], str],
) -> None:
    """Reject a response unless every field it would add matches reviewed bytes."""

    for field in TARGET_FIELDS:
        value = source_row.get(field)
        if not _field_present(field, value):
            continue
        observed = _profile_value(field, value)
        expected = authorizations.get((stable_id, source, entity_id, field))
        if not expected:
            raise ProfileFieldAuthorizationError(
                f"{source} {field} for UFCStats {stable_id} lacks a reviewed field authorization"
            )
        if observed != expected:
            raise ProfileFieldAuthorizationError(
                f"{source} {field} for UFCStats {stable_id} changed from its reviewed normalized value"
            )


def _copy_response_identity_metadata(
    target: dict[str, object],
    profile: dict[str, object],
) -> None:
    for key in (
        "requested_url",
        "effective_url",
        "canonical_url",
        "redirect_chain",
        "source_entity_id",
        "api_athlete_id",
        "api_requested_url",
        "api_effective_url",
        "api_redirect_chain",
        "sport",
        "transport_kind",
        "reader_target_url",
        "origin_redirect_chain_captured",
    ):
        if key in profile:
            target[key] = profile.get(key)


def _transport_entity_id(source: str, url: str) -> str:
    """Resolve an entity from a source's real non-page transport route."""

    if source == "espn":
        parsed = urlparse(url)
        if (
            parsed.scheme.casefold() != "https"
            or (parsed.hostname or "").casefold().rstrip(".")
            != "sports.core.api.espn.com"
            or parsed.query
            or parsed.fragment
            or not re.fullmatch(
                r"/v2/sports/mma/athletes/\d+/?",
                parsed.path,
                flags=re.IGNORECASE,
            )
        ):
            return ""
        return source_entity_id(source, url)
    if source == "martialbot":
        parsed = urlparse(url)
        if not parsed.path.endswith(".data"):
            return ""
        public_url = parsed._replace(
            path=parsed.path.removesuffix(".data"),
            query="",
            fragment="",
        ).geturl()
        return source_entity_id(source, public_url)
    if source == "wikipedia":
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").casefold().rstrip(".")
        if (
            parsed.scheme.casefold() != "https"
            or hostname != "en.wikipedia.org"
            or parsed.path != "/w/api.php"
            or parsed.fragment
        ):
            return ""
        titles = parse_qs(parsed.query).get("titles", [])
        if len(titles) != 1 or not _clean_text(titles[0]):
            return ""
        title = unquote(_clean_text(titles[0])).replace(" ", "_")
        return source_entity_id(
            source,
            f"https://en.wikipedia.org/wiki/{title}",
        )
    if source == "tapology":
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").casefold().rstrip(".")
        prefix = "/https://www.tapology.com/"
        if (
            parsed.scheme.casefold() != "https"
            or hostname != "r.jina.ai"
            or not parsed.path.startswith(prefix)
            or parsed.query
            or parsed.fragment
        ):
            return ""
        target_path = parsed.path.removeprefix(prefix)
        if not target_path:
            return ""
        return source_entity_id(
            source,
            f"https://www.tapology.com/{target_path}",
        )
    return source_entity_id(source, url)


def _validated_response_identity(
    *,
    source: str,
    stable_id: str,
    requested_url: object,
    response: dict[str, object],
    verified_bindings: set[tuple[str, str, str]],
    verified_binding_metadata: dict[
        tuple[str, str, str], dict[str, object]
    ] | None = None,
) -> dict[str, object]:
    """Validate the response identity captured by the fetch, never the search URL."""

    source_key = _clean_text(source).casefold().replace(".com", "")
    requested = _normalized_url(requested_url)
    actual_requested = _normalized_url(response.get("requested_url"))
    effective = _normalized_url(response.get("effective_url"))
    canonical = _normalized_url(response.get("canonical_url"))
    redirect_chain = _captured_redirect_chain(response.get("redirect_chain"))
    reported_entity = _clean_text(response.get("source_entity_id"))
    if not all((requested, actual_requested, effective, canonical, reported_entity)):
        raise ProfileIdentityUnverifiedError(
            f"{source_key} response identity metadata is incomplete for UFCStats {stable_id}"
        )
    if redirect_chain is None:
        raise ProfileIdentityUnverifiedError(
            f"{source_key} response redirect chain was not captured for UFCStats {stable_id}"
        )
    candidates = {
        key
        for key in verified_bindings
        if key[0] == stable_id and key[1] == source_key
    }
    if len(candidates) != 1:
        raise ProfileIdentityUnverifiedError(
            f"{source_key} response has no unambiguous verified binding for UFCStats {stable_id}"
        )
    binding = next(iter(candidates))
    expected_entity = binding[2]
    if reported_entity != expected_entity:
        raise ProfileIdentityUnverifiedError(
            f"{source_key} response entity {reported_entity or 'missing'} does not match "
            f"verified entity {expected_entity} for UFCStats {stable_id}"
        )
    discovery_entity = source_entity_id(source_key, requested)
    if discovery_entity != expected_entity:
        raise ProfileIdentityUnverifiedError(
            f"{source_key} discovery URL resolves to entity "
            f"{discovery_entity or 'missing'}, not verified entity {expected_entity}"
        )
    canonical_entity = source_entity_id(source_key, canonical)
    if canonical_entity != expected_entity:
        raise ProfileIdentityUnverifiedError(
            f"{source_key} canonical URL resolves to entity "
            f"{canonical_entity or 'missing'}, not verified entity {expected_entity}"
        )

    tapology_reader_transport = (
        source_key == "tapology"
        and _clean_text(response.get("transport_kind")).casefold()
        == "jina_reader"
    )
    uses_non_page_transport = (
        source_key in {"espn", "martialbot", "wikipedia"}
        or tapology_reader_transport
    )
    if tapology_reader_transport:
        reader_target = _normalized_url(response.get("reader_target_url"))
        if source_entity_id(source_key, reader_target) != expected_entity:
            raise ProfileIdentityUnverifiedError(
                "tapology reader target does not match the verified Tapology entity "
                f"for UFCStats {stable_id}"
            )
        if response.get("origin_redirect_chain_captured") is not False:
            raise ProfileIdentityUnverifiedError(
                "tapology reader response did not distinguish wrapper history from "
                f"origin redirects for UFCStats {stable_id}"
            )
    for label, url in (
        ("requested", actual_requested),
        ("effective", effective),
        *(("redirect", url) for url in redirect_chain),
    ):
        observed_entity = (
            _transport_entity_id(source_key, url)
            if uses_non_page_transport
            else source_entity_id(source_key, url)
        )
        if observed_entity != expected_entity:
            raise ProfileIdentityUnverifiedError(
                f"{source_key} {label} URL resolves to entity "
                f"{observed_entity or 'missing'}, not verified entity {expected_entity}"
            )
    if redirect_chain and redirect_chain[-1] != effective:
        raise ProfileIdentityUnverifiedError(
            f"{source_key} redirect chain does not end at its effective URL"
        )

    if uses_non_page_transport and redirect_chain and not tapology_reader_transport:
        raise ProfileIdentityUnverifiedError(
            f"{source_key} transport redirected without a reviewed transport binding"
        )

    expected = (verified_binding_metadata or {}).get(binding)
    if expected is None:
        if uses_non_page_transport:
            if canonical != requested:
                raise ProfileIdentityUnverifiedError(
                    f"{source_key} response canonical URL differs from its discovery URL "
                    "without reviewed response metadata"
                )
        elif redirect_chain or not (actual_requested == effective == canonical == requested):
            raise ProfileIdentityUnverifiedError(
                f"{source_key} response redirected or changed canonical URL without "
                "reviewed response metadata"
            )
    else:
        if any(expected.get(key) in {None, ""} for key in ("requested_url", "effective_url", "canonical_url")):
            raise ProfileIdentityUnverifiedError(
                f"{source_key} verified binding lacks complete reviewed response metadata"
            )
        if expected.get("redirect_chain") is None:
            raise ProfileIdentityUnverifiedError(
                f"{source_key} verified binding lacks a captured redirect chain"
            )
        if tapology_reader_transport:
            reviewed_urls = (
                expected["requested_url"],
                expected["effective_url"],
                expected["canonical_url"],
                *expected["redirect_chain"],
            )
            if any(
                expected_entity
                not in {
                    source_entity_id(source_key, url),
                    _transport_entity_id(source_key, url),
                }
                for url in reviewed_urls
            ):
                raise ProfileIdentityUnverifiedError(
                    "tapology reviewed response metadata resolves to another entity"
                )
        elif uses_non_page_transport:
            reviewed_urls = (
                expected["requested_url"],
                expected["effective_url"],
                expected["canonical_url"],
                *expected["redirect_chain"],
            )
            if any(
                source_entity_id(source_key, url) != expected_entity
                for url in reviewed_urls
            ):
                raise ProfileIdentityUnverifiedError(
                    f"{source_key} reviewed response metadata resolves to another entity"
                )
            if (
                expected["requested_url"] != requested
                or expected["effective_url"] != canonical
                or expected["canonical_url"] != canonical
                or expected["redirect_chain"]
            ):
                raise ProfileIdentityUnverifiedError(
                    f"{source_key} response route differs from the verified binding"
                )
        else:
            observed = {
                "requested_url": actual_requested,
                "effective_url": effective,
                "canonical_url": canonical,
                "redirect_chain": redirect_chain,
            }
            if observed != expected or actual_requested != requested:
                raise ProfileIdentityUnverifiedError(
                    f"{source_key} response URL or redirect chain differs from the verified binding"
                )

    validated = {
        "requested_url": actual_requested,
        "effective_url": effective,
        "source_url": canonical if uses_non_page_transport else effective,
        "canonical_url": canonical,
        "redirect_chain": json.dumps(list(redirect_chain)),
        "source_entity_id": expected_entity,
    }
    if tapology_reader_transport:
        validated.update(
            {
                "transport_kind": "jina_reader",
                "reader_target_url": reader_target,
                "origin_redirect_chain_captured": False,
            }
        )
    if source_key == "espn":
        api_athlete_id = _first_nonblank(
            response.get("api_athlete_id"),
            response.get("athlete_id"),
            response.get("id"),
        )
        api_requested = _normalized_url(response.get("api_requested_url"))
        api_effective = _normalized_url(response.get("api_effective_url"))
        api_redirect_chain = _captured_redirect_chain(
            response.get("api_redirect_chain")
        )
        sport = _clean_text(response.get("sport")).casefold()
        if not api_athlete_id or api_athlete_id != expected_entity:
            raise ProfileIdentityUnverifiedError(
                f"espn API athlete ID {api_athlete_id or 'missing'} does not match "
                f"verified entity {expected_entity} for UFCStats {stable_id}"
            )
        if not api_requested or not api_effective or api_redirect_chain is None:
            raise ProfileIdentityUnverifiedError(
                f"espn API response URL/redirect metadata is incomplete for UFCStats {stable_id}"
            )
        if sport != "mma":
            raise ProfileIdentityUnverifiedError(
                f"espn API sport {sport or 'missing'} is not MMA for UFCStats {stable_id}"
            )
        api_urls = (api_requested, api_effective, *api_redirect_chain)
        if any(_transport_entity_id("espn", url) != expected_entity for url in api_urls):
            raise ProfileIdentityUnverifiedError(
                f"espn API response URL resolves to a different athlete for UFCStats {stable_id}"
            )
        if api_redirect_chain:
            raise ProfileIdentityUnverifiedError(
                f"espn API response redirected without a reviewed API redirect binding"
            )
        if (
            api_requested != actual_requested
            or api_effective != effective
            or api_redirect_chain != redirect_chain
        ):
            raise ProfileIdentityUnverifiedError(
                f"espn API metadata conflicts with the captured response route for UFCStats {stable_id}"
            )
        validated.update(
            {
                "api_athlete_id": api_athlete_id,
                "api_requested_url": api_requested,
                "api_effective_url": api_effective,
                "api_redirect_chain": "[]",
                "sport": sport,
            }
        )
    return validated


def _ensure_profile_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in _BASE_PROFILE_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    if "search_names" not in result.columns:
        result["search_names"] = ""
    if "stable_ufcstats_id" not in result.columns:
        result["stable_ufcstats_id"] = ""
    for index, row in result.iterrows():
        result.at[index, "stable_ufcstats_id"] = _stable_ufcstats_id_from_row(row)
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
    names.extend(_clean_text(row.get("alternate_slug_names")).split("|"))
    name_keys = {
        normalize_person_name(value) for value in names if _clean_text(value)
    }
    if name_keys & _TEST_PROFILE_NAME_KEYS:
        return True
    url = _clean_text(row.get("official_athlete_url"))
    path = urlparse(url).path.strip("/").lower() if url else ""
    slug = path.rsplit("/", 1)[-1] if path else ""
    return slug in {"test", "test-fighter", "test-test", "testy-test", "testing-test"}


def _build_candidate_universe(
    scraped_fighters_path: Path,
    *,
    candidate_source_csv: Path | None = None,
    accepted_membership: dict[str, object] | None = None,
) -> pd.DataFrame:
    scraped_df = _load_scraped_profile_rows(scraped_fighters_path)
    if candidate_source_csv is None and accepted_membership is None:
        return scraped_df

    if accepted_membership is not None:
        from src.data.provider_membership import load_accepted_rosterwatch_membership
        if candidate_source_csv is not None and Path(candidate_source_csv).resolve() != Path(accepted_membership["path"]).resolve():
            raise ValueError("Candidate CSV differs from accepted membership input")
        candidate_source_df = load_accepted_rosterwatch_membership(
            Path(accepted_membership["path"]), expected_sha256=accepted_membership["sha256"],
            expected_generation=accepted_membership["generation"],
        )
    else:
        candidate_source_df = pd.read_csv(candidate_source_csv)
    if candidate_source_df.empty:
        return pd.DataFrame(columns=[*_BASE_PROFILE_COLUMNS, "search_names"])

    scraped_rows = scraped_df.to_dict(orient="records")
    scraped_by_id = {
        stable_id: row
        for row in scraped_rows
        if (stable_id := _stable_ufcstats_id_from_row(row))
    }
    candidate_lookup: dict[str, dict[str, object]] = {}
    for _, source_row in candidate_source_df.iterrows():
        source_dict = source_row.to_dict()
        eligible = (bool(source_dict["coverage_eligible"]) if accepted_membership is not None
                    else _candidate_source_row_is_coverage_eligible(source_dict))
        if not eligible:
            continue
        aliases = _row_candidate_names(source_dict)
        if not aliases:
            continue

        source_stable_id = _stable_ufcstats_id_from_row(source_dict)
        if not source_stable_id:
            # Name-only roster/candidate rows may nominate future discovery work,
            # but they cannot acquire a UFCStats identity or profile state here.
            continue
        scraped_match = scraped_by_id.get(source_stable_id)

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

        stable_id = source_stable_id
        candidate_key = f"ufcstats:{stable_id}"

        merged_row = candidate_lookup.setdefault(
            candidate_key,
            {
                "name": canonical_name,
                "stable_ufcstats_id": stable_id,
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
            names = [row.get("name"), *_clean_text(row.get("search_names")).split("|")]
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
    accepted_membership: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_universe = _build_candidate_universe(
        scraped_fighters_path,
        candidate_source_csv=candidate_source_csv,
        **({"accepted_membership": accepted_membership} if accepted_membership is not None else {}),
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
        identity_key = _profile_state_key(row)
        if not identity_key:
            return
        profile = state.setdefault(identity_key, {field: "" for field in TARGET_FIELDS})
        for field in TARGET_FIELDS:
            if not _field_present(field, profile.get(field)) and _field_present(field, row.get(field)):
                profile[field] = row.get(field)

    for _, row in candidate_universe.iterrows():
        _merge_row(row)
    for row in existing_rows:
        _merge_row(row)
    return state


def _source_row_key(row: dict[str, object]) -> tuple[str, str]:
    stable_id = _stable_ufcstats_id_from_row(row)
    source = str(row.get("source", "")).strip().lower()
    entity_id = _clean_text(row.get("source_entity_id"))
    identity_key = f"ufcstats:{stable_id}" if stable_id else normalize_person_name(row.get("name"))
    source_key = f"{source}:{entity_id}" if entity_id else source
    observation_id = _clean_text(row.get("source_observation_id"))
    if observation_id:
        # Fields acquired in different responses retain their own original time
        # and evidence. Never fill an older row under its unrelated provenance.
        source_key = f"{source_key}:observation:{observation_id}"
    return (identity_key, source_key)


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
        "discovery_url": fighter_url,
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


def _require_fetched_profile_name_match(
    *,
    source: str,
    row: pd.Series | dict[str, object],
    profile_name: object,
    fighter_url: object,
    reviewed_name_keys: set[str] | None = None,
) -> None:
    """Turn a fetched contradictory/missing display identity into a hard signal."""

    observed = _clean_text(profile_name)
    observed_key = normalize_person_name(observed)
    canonical_name = _clean_text(row.get("name"))
    canonical_key = normalize_person_name(canonical_name)
    suffix_aliases = {
        "jr": "jr",
        "junior": "jr",
        "sr": "sr",
        "senior": "sr",
        "ii": "ii",
        "2nd": "ii",
        "iii": "iii",
        "3rd": "iii",
        "iv": "iv",
        "v": "v",
    }

    def _suffix(name_key: str) -> str:
        tokens = name_key.split()
        return suffix_aliases.get(tokens[-1], "") if tokens else ""

    observed_suffix = _suffix(observed_key)
    canonical_suffix = _suffix(canonical_key)
    suffix_consistent = observed_suffix == canonical_suffix
    if reviewed_name_keys is None:
        allowed_keys = {canonical_key}
        candidate_is_reviewed = True
    else:
        allowed_keys = set(reviewed_name_keys)
        candidate_is_reviewed = canonical_key in allowed_keys
    if not candidate_is_reviewed:
        raise ProfileIdentityUnverifiedError(
            f"{source} candidate identity {canonical_name or 'no name'} is not "
            "authorized by the exact reviewed source binding "
            f"({_clean_text(fighter_url) or 'missing URL'})"
        )
    if (
        observed_key
        and suffix_consistent
        and observed_key in allowed_keys
    ):
        return
    expected = canonical_name or "missing candidate name"
    raise ProfileIdentityUnverifiedError(
        f"{source} fetched profile displayed {observed or 'no name'}, not the "
        f"candidate identity {expected} ({_clean_text(fighter_url) or 'missing URL'})"
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
        if _field_present(target_field, current_profile.get(target_field)):
            continue
        source_value = source_profile.get(source_field)
        if not _field_present(target_field, source_value):
            continue
        supplement[target_field] = source_value
        recovered_any = True
    return recovered_any


def _update_state_from_row(state: dict[str, dict[str, object]], row: dict[str, object]) -> None:
    identity_key = _profile_state_key(row)
    if not identity_key:
        return
    profile = state.setdefault(identity_key, {field: "" for field in TARGET_FIELDS})
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
    identity_key = _profile_state_key(row)
    if not identity_key:
        return False
    return any(
        _source_has_recoverable_gap(source, identity_key, current_state)
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
    *,
    verified_bindings: set[tuple[str, str, str]] | None = None,
    verified_binding_metadata: dict[
        tuple[str, str, str], dict[str, object]
    ] | None = None,
    reviewed_field_authorizations: dict[
        tuple[str, str, str, str], str
    ] | None = None,
    reviewed_binding_names: dict[
        tuple[str, str, str], set[str]
    ] | None = None,
    retained_descriptor: dict[str, object] | None = None,
    retained_mode: bool = False,
    available_before: object = None,
    observation_history_root: Path | None = None,
    accepted_membership: dict[str, object] | None = None,
    provider_transport=None,
    provider_response_format: str = "html",
    provider_capture_log: list | None = None,
) -> dict[str, object] | None:
    stable_id = normalize_ufcstats_id(row.get("stable_ufcstats_id"))
    if not stable_id:
        raise ProfileIdentityUnverifiedError(
            f"{source} discovery for {row.get('name')} has no stable UFCStats ID"
        )
    source_key = _clean_text(source).casefold().replace(".com", "")
    binding_candidates = {
        key
        for key in (verified_bindings or set())
        if key[0] == stable_id and key[1] == source_key
    }
    if len(binding_candidates) != 1:
        # An absent or ambiguous reviewed owner is an expected identity-safe
        # rejection, not a reason to contact a source and potentially report a
        # transport outage for an observation that could never be consumed.
        raise ProfileIdentityUnverifiedError(
            f"{source_key} source has no unambiguous verified binding for UFCStats "
            f"{stable_id}; discovery was not attempted"
        )
    candidate_reviewed_names: set[str] = set()
    for (bound_stable_id, bound_source, _entity_id), names in (
        reviewed_binding_names or {}
    ).items():
        if bound_stable_id == stable_id and bound_source == source_key:
            candidate_reviewed_names.update(names)
    reviewed_name_kwargs = (
        {"reviewed_name_keys": candidate_reviewed_names}
        if candidate_reviewed_names
        else {}
    )

    state_before = {key: dict(value) for key, value in current_state.items()}
    source_row: dict[str, object] | None
    captured_here = False
    if not retained_mode and source_key in {"tapology", "sherdog"}:
        from src.data.provider_observation_history import _time, _provider_number
        from src.data.provider_response_capture import acquire_provider_response, ProviderCaptureRefusal
        if accepted_membership is None or available_before is None or observation_history_root is None:
            raise RetainedProfileProcessingError("Ordinary provider processing requires accepted membership, cutoff, and evidence destination")
        try:
            _time(available_before)
            bound = next(iter(binding_candidates))
            route = _clean_text((verified_binding_metadata or {}).get(bound, {}).get("requested_url")) or bound[2]
            if provider_response_format == "tapology_reader" and route.startswith("https://r.jina.ai/"):
                route = route.removeprefix("https://r.jina.ai/")
            if not _provider_number(source_key, route) or _provider_number(source_key, route) != _provider_number(source_key, bound[2]):
                raise RetainedProfileProcessingError("Reviewed provider binding has an unsupported response route")
            authorized_fields = {key[3] for key in (reviewed_field_authorizations or {}) if key[:3] == bound}
            recoverable_authorized_fields = authorized_fields.intersection(SOURCE_RECOVERABLE_FIELDS[source_key])
            if not recoverable_authorized_fields:
                raise RetainedProfileProcessingError("Ordinary provider processing requires independent recoverable field decisions before acquisition")
            current_profile = current_state.get(_profile_state_key(row), {})
            if all(_field_present(field, current_profile.get(field)) for field in recoverable_authorized_fields):
                # Unfilled fields outside this independent review cannot
                # nominate another source request after authorized work is done.
                return None
            retained_descriptor = acquire_provider_response(
                source_key, route, history_root=observation_history_root,
                transport=provider_transport, response_format=provider_response_format,
                capture_log=provider_capture_log,
            )
        except ProviderCaptureRefusal as exc:
            refusal = RetainedProfileProcessingError(str(exc))
            refusal.captures = tuple(getattr(exc, "captures", ()))
            raise refusal from exc
        captured_here = True
        retained_mode = True
    if retained_mode:
        from src.data.provider_observation_history import load_retained_provider_observation
        if retained_descriptor is None or observation_history_root is None:
            raise RetainedProfileProcessingError("Missing retained response or observation history root")
        try:
            observation = load_retained_provider_observation(
                retained_descriptor, fighter_id=stable_id, fighter_name=_clean_text(row.get("name")),
                available_before=available_before, verified_bindings=verified_bindings or set(),
                verified_binding_metadata=verified_binding_metadata,
                verified_binding_name_keys=reviewed_binding_names,
                history_root=observation_history_root,
                consumer={"kind": "profile", "id": stable_id},
            )
        except ProfileIdentityUnverifiedError:
            raise
        except Exception as exc:
            refusal = RetainedProfileProcessingError(str(exc))
            refusal.captures = (retained_descriptor,) if captured_here else ()
            raise refusal from exc
        profile = observation["profile"]
        source_row = _build_base_row(
            _clean_text(row.get("name")), source=source, source_name=_clean_text(profile.get("name")),
            search_name=_clean_text(row.get("name")),
            fighter_url=(profile["canonical_url"] if profile.get("transport_kind") == "jina_reader" else profile["requested_url"]),
        )
        _copy_response_identity_metadata(source_row, profile)
        source_row.update(observation["row_metadata"])
        field_map = {"height": "height_raw", "reach": "reach_raw", "weight": "weight_raw", "dob": "dob"}
        field_map = {key: value for key, value in field_map.items() if key in SOURCE_RECOVERABLE_FIELDS[source]}
        if not _recover_profile_fields(source_row, current_state.get(_profile_state_key(row), {}), profile, field_map=field_map):
            return None
    elif source == "martialbot":
        source_row = _build_martialbot_row(
            row,
            current_state,
            **reviewed_name_kwargs,
        )
    elif source == "espn":
        source_row = _build_espn_row(
            row,
            current_state,
            verified_bindings=verified_bindings,
            verified_binding_metadata=verified_binding_metadata,
            **reviewed_name_kwargs,
        )
    elif source == "wikipedia":
        source_row = _build_wikipedia_row(
            session,
            row,
            current_state,
            **reviewed_name_kwargs,
        )
    elif source == "fightdx":
        source_row = _build_fightdx_row(
            row,
            current_state,
            **reviewed_name_kwargs,
        )
    elif source == "tapology":
        source_row = _build_tapology_row(
            row,
            current_state,
            **reviewed_name_kwargs,
        )
    elif source == "sherdog":
        source_row = _build_sherdog_row(
            row,
            current_state,
            **reviewed_name_kwargs,
        )
    else:
        source_row = None
    if source_row is None:
        return None

    try:
        response_identity = _validated_response_identity(
            source=source,
            stable_id=stable_id,
            requested_url=source_row.get("discovery_url"),
            response=source_row,
            verified_bindings=verified_bindings or set(),
            verified_binding_metadata=verified_binding_metadata,
        )
        exact_binding = (
            stable_id,
            source_key,
            response_identity["source_entity_id"],
        )
        _require_fetched_profile_name_match(
            source=source,
            row=row,
            profile_name=source_row.get("source_name"),
            fighter_url=response_identity["source_url"],
            reviewed_name_keys=(reviewed_binding_names or {}).get(exact_binding),
        )
        _authorize_recovered_source_fields(
            source_row,
            stable_id=stable_id,
            source=source_key,
            entity_id=response_identity["source_entity_id"],
            authorizations=reviewed_field_authorizations or {},
        )
    except Exception as exc:
        current_state.clear()
        current_state.update(state_before)
        if captured_here:
            exc.captures = (retained_descriptor,)
        raise
    source_row.update(response_identity)
    source_row["fighter_url"] = response_identity["source_url"]
    source_row["stable_ufcstats_id"] = stable_id
    source_row["ufcstats_url"] = f"http://ufcstats.com/fighter-details/{stable_id}"
    source_row["source_identity_status"] = "verified_binding"
    return source_row


def _build_tapology_row(
    scraped_row: pd.Series,
    current_state: dict[str, dict[str, object]],
    *,
    reviewed_name_keys: set[str] | None = None,
) -> dict[str, object] | None:
    fighter_name = _clean_text(scraped_row.get("name"))
    fighter_key = _profile_state_key(scraped_row)
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
        _require_fetched_profile_name_match(
            source="tapology",
            row=scraped_row,
            profile_name=profile.get("name"),
            fighter_url=fighter_url,
            reviewed_name_keys=reviewed_name_keys,
        )
        current_profile = current_state.setdefault(fighter_key, {field: "" for field in TARGET_FIELDS})
        supplement = _build_base_row(
            fighter_name,
            source="tapology",
            source_name=_clean_text(profile.get("name")),
            search_name=search_name,
            fighter_url=fighter_url,
        )
        _copy_response_identity_metadata(supplement, profile)

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
    *,
    reviewed_name_keys: set[str] | None = None,
) -> dict[str, object] | None:
    fighter_name = _clean_text(scraped_row.get("name"))
    fighter_key = _profile_state_key(scraped_row)
    search_name, fighter_url = _resolve_profile_url(scraped_row, search_sherdog)
    if not fighter_url:
        return None

    profile, _fights = scrape_sherdog_page(fighter_url, search_name)
    _require_fetched_profile_name_match(
        source="sherdog",
        row=scraped_row,
        profile_name=profile.get("name"),
        fighter_url=fighter_url,
        reviewed_name_keys=reviewed_name_keys,
    )
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
        source_name=_clean_text(profile.get("name")),
        search_name=search_name,
        fighter_url=fighter_url,
    )
    _copy_response_identity_metadata(supplement, profile)

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
    *,
    reviewed_name_keys: set[str] | None = None,
) -> dict[str, object] | None:
    fighter_name = _clean_text(scraped_row.get("name"))
    fighter_key = _profile_state_key(scraped_row)
    search_name, fighter_url = _resolve_profile_url(scraped_row, search_fightdx)
    if not fighter_url:
        return None

    profile = scrape_fightdx_profile(fighter_url)
    _require_fetched_profile_name_match(
        source="fightdx",
        row=scraped_row,
        profile_name=profile.get("name"),
        fighter_url=fighter_url,
        reviewed_name_keys=reviewed_name_keys,
    )
    current_profile = current_state.setdefault(fighter_key, {field: "" for field in TARGET_FIELDS})
    supplement = _build_base_row(
        fighter_name,
        source="fightdx",
        source_name=_clean_text(profile.get("name")),
        search_name=search_name,
        fighter_url=fighter_url,
    )
    _copy_response_identity_metadata(supplement, profile)

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
    *,
    verified_bindings: set[tuple[str, str, str]] | None = None,
    verified_binding_metadata: dict[
        tuple[str, str, str], dict[str, object]
    ] | None = None,
    reviewed_name_keys: set[str] | None = None,
) -> dict[str, object] | None:
    fighter_name = _clean_text(scraped_row.get("name"))
    fighter_key = _profile_state_key(scraped_row)
    search_name, fighter_url = _resolve_profile_url(scraped_row, search_espn)
    if not fighter_url:
        return None

    profile = scrape_espn_profile(fighter_url)
    _require_fetched_profile_name_match(
        source="espn",
        row=scraped_row,
        profile_name=profile.get("name"),
        fighter_url=fighter_url,
        reviewed_name_keys=reviewed_name_keys,
    )
    stable_id = normalize_ufcstats_id(scraped_row.get("stable_ufcstats_id"))
    if not stable_id:
        raise ProfileIdentityUnverifiedError(
            f"espn discovery for {fighter_name} has no stable UFCStats ID"
        )
    response_identity = _validated_response_identity(
        source="espn",
        stable_id=stable_id,
        requested_url=fighter_url,
        response=profile,
        verified_bindings=verified_bindings or set(),
        verified_binding_metadata=verified_binding_metadata,
    )
    current_profile = current_state.setdefault(fighter_key, {field: "" for field in TARGET_FIELDS})
    supplement = _build_base_row(
        fighter_name,
        source="espn",
        source_name=_clean_text(profile.get("name")),
        search_name=search_name,
        fighter_url=fighter_url,
    )
    _copy_response_identity_metadata(supplement, profile)
    supplement.update(response_identity)
    supplement["fighter_url"] = response_identity["source_url"]

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
    *,
    reviewed_name_keys: set[str] | None = None,
) -> dict[str, object] | None:
    fighter_name = _clean_text(scraped_row.get("name"))
    fighter_key = _profile_state_key(scraped_row)
    search_name, fighter_url = _resolve_profile_url(
        scraped_row,
        search_martialbot,
        override_map=MARTIALBOT_SEARCH_NAME_OVERRIDES,
    )
    if not fighter_url:
        return None

    profile = scrape_martialbot_profile(fighter_url)
    _require_fetched_profile_name_match(
        source="martialbot",
        row=scraped_row,
        profile_name=profile.get("name"),
        fighter_url=fighter_url,
        reviewed_name_keys=reviewed_name_keys,
    )
    current_profile = current_state.setdefault(fighter_key, {field: "" for field in TARGET_FIELDS})
    supplement = _build_base_row(
        fighter_name,
        source="martialbot",
        source_name=_clean_text(profile.get("name")),
        search_name=search_name,
        fighter_url=fighter_url,
    )
    _copy_response_identity_metadata(supplement, profile)

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


def _wiki_api(
    session: requests.Session,
    *,
    include_response_metadata: bool = False,
    **params,
) -> dict | tuple[dict, dict[str, object]]:
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
        payload = response.json()
        if include_response_metadata:
            request = getattr(response, "request", None)
            requested_url = _clean_text(
                getattr(request, "url", "") or WIKIPEDIA_API_URL
            )
            effective_url = _clean_text(
                getattr(response, "url", "") or requested_url
            )
            history = list(getattr(response, "history", None) or [])
            redirect_chain = [
                _clean_text(getattr(item, "url", "")) for item in history
            ]
            if history and (not redirect_chain or redirect_chain[-1] != effective_url):
                redirect_chain.append(effective_url)
            return payload, {
                "requested_url": requested_url,
                "effective_url": effective_url,
                "redirect_chain": redirect_chain,
            }
        return payload
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


def _wikipedia_title_has_allowed_qualifier(title: str) -> bool:
    qualifier_match = re.search(r"\(([^)]*)\)", str(title or "").strip())
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
    text = _clean_text(value)
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
    text = _clean_text(raw_value)
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
    text = _clean_text(raw_value)
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
    text = _clean_text(raw_value)
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
    *,
    reviewed_name_keys: set[str] | None = None,
) -> dict[str, object] | None:
    fighter_name = _clean_text(scraped_row.get("name"))
    fighter_key = _profile_state_key(scraped_row)
    title = None
    search_name = ""
    for search_name_candidate in _search_names_for_source(scraped_row):
        title = _wikipedia_find_title(session, search_name_candidate)
        if title:
            search_name = search_name_candidate
            break
    if not title:
        return None

    payload, api_response = _wiki_api(
        session,
        include_response_metadata=True,
        action="query",
        titles=title,
        prop="revisions",
        rvprop="content",
        format="json",
        formatversion=2,
    )
    page = payload["query"]["pages"][0]
    response_title = _clean_text(page.get("title"))
    if not response_title or not _wikipedia_title_has_allowed_qualifier(response_title):
        raise ProfileIdentityUnverifiedError(
            "wikipedia fetched page displayed "
            f"{response_title or 'no title'}, not candidate {fighter_name}"
        )
    response_name = re.sub(r"\s*\([^)]*\)\s*$", "", response_title).strip()
    _require_fetched_profile_name_match(
        source="wikipedia",
        row=scraped_row,
        profile_name=response_name or response_title,
        fighter_url=f"https://en.wikipedia.org/wiki/{response_title.replace(' ', '_')}",
        reviewed_name_keys=reviewed_name_keys,
    )
    canonical_url = (
        f"https://en.wikipedia.org/wiki/{response_title.replace(' ', '_')}"
        if response_title
        else ""
    )
    wikitext = page.get("revisions", [{}])[0].get("content", "")
    reach = _parse_wikipedia_reach(_extract_infobox_field(wikitext, "reach"))
    stance = _parse_wikipedia_stance(_extract_infobox_field(wikitext, "stance"))
    dob = _parse_wikipedia_birth_date(_extract_infobox_field(wikitext, "birth_date"))

    current_profile = current_state.setdefault(fighter_key, {field: "" for field in TARGET_FIELDS})
    supplement = _build_base_row(
        fighter_name,
        source="wikipedia",
        source_name=response_name,
        search_name=search_name,
        fighter_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
    )
    supplement.update(
        {
            **api_response,
            "canonical_url": canonical_url,
            "source_entity_id": canonical_url.rstrip("/").casefold(),
        }
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
    profile_integrity_decisions_path: Path | None = None,
    prior_profile_integrity_decisions_path: Path | None = None,
    retained_provider_observations: dict[str, dict[str, object]] | None = None,
    accepted_membership: dict[str, object] | None = None,
    available_before: object = None,
    observation_history_root: Path | None = None,
    provider_transport=None,
    provider_response_format: str = "html",
    candidate_priorities: dict[str, int] | None = None,
) -> dict[str, object]:
    """Refresh the supplemental profile artifact and return a structured summary."""
    retained_mode = retained_provider_observations is not None
    if retained_mode and limit is not None and int(limit) <= 0:
        raise RetainedProfileProcessingError("Retained processing limit must be positive")
    if retained_mode and (accepted_membership is None or available_before is None or observation_history_root is None):
        raise RetainedProfileProcessingError("Retained processing requires accepted membership, cutoff, and evidence destination")
    if not retained_mode:
        clear_fallback_cache(preserve_environment_blocks=True)
    candidate_universe, candidates = _load_candidates(
        scraped_fighters_path,
        gap_audit_csv=gap_audit_csv,
        candidate_source_csv=candidate_source_csv,
        **({"accepted_membership": accepted_membership} if accepted_membership is not None else {}),
    )
    existing_rows = _load_existing_rows(output_path)
    current_state = _build_effective_profile_state(candidate_universe, existing_rows)
    selected_sources = tuple(dict.fromkeys(sources or ALL_SOURCES))
    ordinary_provider_mode = not retained_mode and bool(set(selected_sources) & {"tapology", "sherdog"})
    prior_references = None
    if ordinary_provider_mode and output_path.exists():
        from src.data.provider_consumer_references import snapshot_provider_references
        prior_references = snapshot_provider_references(output_path)
    if candidate_priorities:
        if accepted_membership is None:
            raise RetainedProfileProcessingError("Stable candidate priorities require accepted membership")
        # Priority has no authority over membership or the accepted source bytes.
        candidates["profile_refresh_priority"] = candidates["stable_ufcstats_id"].map(candidate_priorities).fillna(0).astype(int).clip(lower=0)
    verified_bindings: set[tuple[str, str, str]] = set()
    verified_response_metadata: dict[
        tuple[str, str, str], dict[str, object]
    ] = {}
    reviewed_field_authorizations: dict[
        tuple[str, str, str, str], str
    ] = {}
    reviewed_binding_names: dict[tuple[str, str, str], set[str]] = {}
    decision_input_pins = []
    if ordinary_provider_mode:
        from src.data.provider_consumer_references import _pin
        decision_input_pins = [_pin(path) for path in (profile_integrity_decisions_path, prior_profile_integrity_decisions_path) if path is not None]
    if (
        prior_profile_integrity_decisions_path is not None
        and profile_integrity_decisions_path is None
    ):
        raise ValueError(
            "A prior profile decision registry requires a current profile decision artifact"
        )
    if profile_integrity_decisions_path is not None:
        profile_decisions = load_profile_decisions(
            profile_integrity_decisions_path,
            prior_decisions_path=prior_profile_integrity_decisions_path,
        )
        verified_bindings = verified_binding_keys(profile_decisions)
        verified_response_metadata = _verified_binding_metadata(
            profile_decisions,
            verified_bindings,
        )
        reviewed_field_authorizations = _reviewed_field_authorizations(
            profile_decisions,
            verified_bindings,
        )
        reviewed_binding_names = _reviewed_binding_name_keys(
            profile_decisions,
            verified_bindings,
        )
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
    unavailable_sources: set[str] = set()
    identity_rejections: list[dict[str, str]] = []
    processing_errors: list[dict[str, str]] = []
    provider_capture_log: list[dict] = []
    attempted = 0

    with (nullcontext(None) if retained_mode else requests.Session()) as session:
        total_candidates = int(len(candidates))
        for index, (_, row) in enumerate(candidates.iterrows(), start=1):
            attempted += 1
            fighter_key = _profile_state_key(row)
            if not fighter_key:
                continue
            logger.info(
                "Profile supplement refresh candidate %d/%d: %s",
                index,
                total_candidates,
                row.get("name"),
            )

            for source in selected_sources:
                if source in unavailable_sources:
                    continue
                if not _source_has_recoverable_gap(source, fighter_key, current_state):
                    continue
                try:
                    source_row = _build_source_row(
                        source,
                        session,
                        row,
                        current_state,
                        verified_bindings=verified_bindings,
                        verified_binding_metadata=verified_response_metadata,
                        reviewed_field_authorizations=reviewed_field_authorizations,
                        reviewed_binding_names=reviewed_binding_names,
                        accepted_membership=accepted_membership,
                        provider_transport=provider_transport,
                        provider_response_format=provider_response_format,
                        provider_capture_log=provider_capture_log,
                        available_before=available_before,
                        observation_history_root=observation_history_root,
                        **({
                            "retained_mode": True,
                            "retained_descriptor": retained_provider_observations.get(_stable_ufcstats_id_from_row(row), {}).get(source),
                        } if retained_mode else {}),
                    )
                except RetainedProfileProcessingError as exc:
                    processing_errors.append({"source": source, "fighter": _clean_text(row.get("name")), "error": str(exc),
                                              "captures": list(getattr(exc, "captures", ()))})
                    source_row = None
                except ProfileIdentityUnverifiedError as exc:
                    identity_rejections.append(
                        {
                            "source": source,
                            "fighter": _clean_text(row.get("name")),
                            "stable_ufcstats_id": _clean_text(
                                row.get("stable_ufcstats_id")
                            ),
                            "reason": str(exc),
                            "captures": list(getattr(exc, "captures", ())),
                        }
                    )
                    logger.info(
                        "%s profile observation rejected for '%s' because its "
                        "identity or reviewed field binding was not verified: %s",
                        SOURCE_LOG_LABELS.get(source, source),
                        row.get("name"),
                        exc,
                    )
                    source_row = None
                except Exception as exc:
                    from src.data.provider_response_capture import ProviderAcquisitionError
                    if retained_mode or (source in {"tapology", "sherdog"} and not isinstance(exc, ProviderAcquisitionError)):
                        processing_errors.append({"source": source, "fighter": _clean_text(row.get("name")), "error": str(exc)})
                        continue
                    espn_source_outage = (
                        source == "espn"
                        and isinstance(exc, ESPNSourceUnavailableError)
                    )
                    if espn_source_outage:
                        unavailable_sources.add(source)
                    source_errors[source] += 1
                    source_error_details.append(
                        {
                            "source": source,
                            "fighter": _clean_text(row.get("name")),
                            "error": str(exc),
                            **({"captures": list(exc.captures)} if hasattr(exc, "captures") else {}),
                        }
                    )
                    if espn_source_outage:
                        logger.info(
                            "ESPN profile source unavailable for '%s'; "
                            "skipping ESPN for the remainder of this refresh: %s",
                            row.get("name"),
                            exc,
                        )
                    else:
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
                if ordinary_provider_mode and source in {"tapology", "sherdog"}:
                    source_row["provider_reference_schema"] = 1
                results.append(source_row)
                _update_state_from_row(current_state, source_row)
                source_recoveries[source] += 1

    source_error_count = int(sum(source_errors.values()))
    candidate_recovered_rows = int(len(results))
    candidate_recovered_by_source = dict(source_recoveries)
    output_existed = output_path.exists()
    output_updated = False
    output_action: str
    if processing_errors:
        output_action = "preserved_due_to_processing_errors" if output_existed else "not_created_due_to_processing_errors"
        published_results = []
        published_recoveries = {source: 0 for source in ALL_SOURCES}
    elif source_error_count:
        # A partially successful discovery run is not a publishable supplement.
        # In particular, do not parse/sort/reserialize a preexisting staged or
        # accepted artifact after a source exception.
        output_action = (
            "preserved_due_to_source_errors"
            if output_existed
            else "not_created_due_to_source_errors"
        )
        published_results: list[dict[str, object]] = []
        published_recoveries = {source: 0 for source in ALL_SOURCES}
    elif identity_rejections:
        # Identity/authorization rejections are expected fail-closed observations,
        # not source outages. They still veto the transaction so a mixed discovery
        # run cannot publish only the rows that happened to pass identity checks.
        output_action = (
            "preserved_due_to_identity_rejections"
            if output_existed
            else "not_created_due_to_identity_rejections"
        )
        published_results = []
        published_recoveries = {source: 0 for source in ALL_SOURCES}
    elif not results:
        output_action = "preserved_no_changes" if output_existed else "not_created_no_changes"
        published_results = []
        published_recoveries = {source: 0 for source in ALL_SOURCES}
    else:
        combined_rows = _merge_source_rows(existing_rows, results)
        output_df = pd.DataFrame(combined_rows)
        if not output_df.empty:
            output_df = output_df.sort_values(["name", "source"]).reset_index(drop=True)
        write_csv_atomically(output_df, output_path, refuse_empty=True)
        if ordinary_provider_mode:
            from src.data.provider_consumer_references import record_provider_consumer_artifact
            try:
                record_provider_consumer_artifact(
                    output_path, evidence_paths=[Path(value) for value in output_df.get("source_evidence_path", []) if not _blank(value)],
                    accepted_membership=accepted_membership,
                    decision_paths=[path for path in (profile_integrity_decisions_path, prior_profile_integrity_decisions_path) if path is not None],
                    expected_decision_pins=decision_input_pins,
                    capture_descriptors=provider_capture_log,
                    parent_references=[prior_references] if prior_references is not None else (),
                    history_root=observation_history_root, kind="profile_csv",
                )
            except (OSError, ValueError, RuntimeError) as exc:
                processing_errors.append({"source": "references", "fighter": "", "error": str(exc)})
        output_updated = True
        output_action = "written_with_incomplete_references" if processing_errors else "updated"
        published_results = [] if processing_errors else results
        published_recoveries = {source: 0 for source in ALL_SOURCES} if processing_errors else source_recoveries

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
        "recovered_rows": int(len(published_results)),
        "candidate_recovered_rows": candidate_recovered_rows,
        "discarded_recovered_rows": candidate_recovered_rows - int(len(published_results)),
        "selected_sources": list(selected_sources),
        "recovered_by_source": published_recoveries,
        "candidate_recovered_by_source": candidate_recovered_by_source,
        "processing_error_count": len(processing_errors),
        "processing_errors": processing_errors,
        "provider_response_captures": provider_capture_log,
        "source_errors": source_errors,
        "source_error_count": source_error_count,
        "source_error_details": source_error_details,
        "identity_rejection_count": len(identity_rejections),
        "identity_rejections": identity_rejections,
        "output_updated": output_updated,
        "output_action": output_action,
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
    parser.add_argument("--profile-integrity-decisions", type=Path, default=None)
    parser.add_argument(
        "--prior-profile-integrity-decisions",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--candidate-offset",
        type=int,
        default=0,
        help="Rotate the non-priority recoverable backlog before applying --limit.",
    )
    parser.add_argument("--retained-provider-observations", type=Path)
    parser.add_argument("--accepted-membership", type=Path)
    parser.add_argument("--available-before")
    parser.add_argument("--observation-history-root", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    retained = (json.loads(args.retained_provider_observations.read_text(encoding="utf-8"))
                if args.retained_provider_observations is not None else None)
    membership = (json.loads(args.accepted_membership.read_text(encoding="utf-8"))
                  if args.accepted_membership is not None else None)

    summary = run_profile_supplement_refresh(
        scraped_fighters_path=args.scraped_fighters_path,
        retained_provider_observations=retained,
        accepted_membership=membership,
        available_before=args.available_before,
        observation_history_root=args.observation_history_root,
        gap_audit_csv=args.gap_audit_csv,
        candidate_source_csv=args.candidate_source_csv,
        output_path=args.output,
        sources=args.sources,
        limit=args.limit,
        candidate_offset=args.candidate_offset,
        profile_integrity_decisions_path=args.profile_integrity_decisions,
        prior_profile_integrity_decisions_path=(
            args.prior_profile_integrity_decisions
        ),
    )
    if args.json:
        print(json.dumps(summary, indent=2))
        if retained is not None and (summary["processing_error_count"] or summary["identity_rejection_count"]):
            raise SystemExit(2)
        return

    print(f"Candidates: {summary['candidate_rows']}")
    print(f"Attempted: {summary['attempted_rows']}")
    print(f"Recovered: {summary['recovered_rows']}")
    print(f"Recovered by source: {summary['recovered_by_source']}")
    print(f"Source errors: {summary['source_errors']}")
    print(f"Output: {summary['output_path']}")
    if retained is not None and (summary["processing_error_count"] or summary["identity_rejection_count"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
