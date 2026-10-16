"""Official UFC active-roster scraping helpers."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from urllib.parse import parse_qsl, urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import RAW_DATA_DIR
from src.data.io_utils import write_csv_atomically
from src.data.name_utils import (
    CURRENT_UFCSTATS_ID_OVERRIDES,
    normalize_cross_source_name,
    normalize_person_name,
    normalize_ufcstats_id,
    reviewed_fighter_identity_id,
    same_person_name,
)
from src.data.profile_integrity import (
    source_entity_id,
    validate_profile_decisions,
    verified_binding_keys,
)
from src.data.reviewed_roster_recovery import (
    apply_reviewed_mma_identity_binding,
    reviewed_mma_identity_binding,
    reviewed_mma_identity_binding_is_complete,
    reviewed_mma_identity_registry_record,
    reviewed_mma_identity_transition_pending_source_canonical,
    reviewed_mma_identity_transition_pending_stable_id,
    reviewed_non_mma_exclusion,
    reviewed_non_mma_identity_match,
    reviewed_official_identity_transition_match,
    reviewed_official_identity_transitions,
    reviewed_unbound_simultaneous_alternate_route,
    reviewed_unbound_simultaneous_route_quarantines,
)
from src.data.reviewed_roster_identity import (
    apply_reviewed_profile_identity_binding,
    reviewed_profile_identity_binding,
    reviewed_profile_identity_binding_record,
    reviewed_ufcstats_identity_rejection,
    reviewed_ufcstats_resolution_blocked,
)
from src.data.ufc_update_schedule import iso_week_identity

logger = logging.getLogger(__name__)

_ROSTER_LIVE_SYNC_INCIDENT_KEY = "ufc-active-roster:live-sync-failed"
ACTIVE_ROSTER_SYNC_GENERATION_COLUMN = "active_roster_sync_generation_id"
_CSV_NUMERIC_TEXT_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
_SEMANTIC_NUMERIC_CSV_COLUMNS = frozenset(
    {
        "age",
        "height",
        "reach",
        "weight",
        "active_roster_consecutive_missing_syncs",
    }
)

OFFICIAL_ACTIVE_ROSTER_URL = "https://www.ufc.com/athletes/all?filters%5B0%5D=status%3A23"
OFFICIAL_ACTIVE_ROSTER_PATH = RAW_DATA_DIR / "ufc_active_roster_official.csv"
OFFICIAL_ACTIVE_ROSTER_IDENTITY_AUDIT_PATH = RAW_DATA_DIR / "ufc_active_roster_identity_audit.csv"
_DEFAULT_IDENTITY_AUDIT_PATH = object()
POWERSLAP_SEARCH_URL = "https://www.powerslap.com/wp-json/wp/v2/striker"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_REQUEST_DELAY_SECONDS = 0.35
_REQUEST_TIMEOUT_SECONDS = 30
_REQUEST_MAX_RETRIES = 3
_REQUEST_RETRY_BACKOFF_SECONDS = 1.0
_REQUEST_RETRY_STATUS_CODES = frozenset({500, 502, 503, 504})
_POWERSLAP_MATCH_LIMIT = 10
_PROFILE_ACTIVE_STATUS_KEY = "active"
_PROFILE_CURRENT_STATUS_KEYS = frozenset({_PROFILE_ACTIVE_STATUS_KEY, "current"})
_PROFILE_STATUS_UNKNOWN = "status_unknown"
_PROFILE_FETCH_OUTCOME_COLUMN = "official_profile_fetch_outcome"
_PROFILE_FETCH_SUCCEEDED = "succeeded"
_PROFILE_FETCH_REQUEST_FAILED = "request_failed"
_PROFILE_FETCH_INVALID_RESPONSE = "invalid_response"
_PROFILE_FETCH_HTTP_ERROR = "http_error"
_PROFILE_FETCH_REQUEST_ERROR = "request_error"
_PROFILE_FETCH_PARSE_FAILED = "parse_failed"
_PROFILE_FETCH_NOT_REQUESTED = "not_requested"
# This destructive authority is deliberately narrower than the reviewed
# route-transition registry. Daria's route correction is approved, but only
# Anderson's separately reviewed inactive lifecycle may remove a source-route
# row, whether both exact routes appear live or a later scan error requires a
# cached fallback.
_INACTIVE_TRANSITION_RECONCILIATION_STABLE_IDS = frozenset(
    {"bbc9ab95fa766493"}
)
# This is separate from inactive-removal authority. It permits only Nakamura's
# exact inherited two-owner shape to reach the later destructive/canonical
# reconciliation; it does not make either route a general alias.
_ACTIVE_DUPLICATE_OWNER_TRANSITION_STABLE_IDS = frozenset(
    {"3618e0632ecdba9c"}
)
# Fresh simultaneous pairs have their own narrower authority. Ravena may
# collapse only when both live endpoints carry complete fresh proof; she never
# receives Nakamura's inherited duplicate-owner removal authority.
_SIMULTANEOUS_ACTIVE_OWNER_TRANSITION_STABLE_IDS = frozenset(
    {"3618e0632ecdba9c", "4948e412fa6ab67d"}
)
_STRICT_FRESH_CACHED_ROUTE_TRANSITION_STABLE_IDS = frozenset(
    {"4948e412fa6ab67d"}
)
_REVIEWED_OFFICIAL_ONLY_NAME_TRANSITIONS = {
    "athlete:matthieu-letho-duclos": {
        "cached_official_name": "Matthieu Letho Duclos",
        "live_official_name": "Matthieu Duclos",
        "official_athlete_url": (
            "https://www.ufc.com/athlete/matthieu-letho-duclos"
        ),
    }
}
_PROFILE_INACTIVE_STATUS_KEYS = frozenset(
    {
        "not fighting",
        "retired",
        "inactive",
        "cut",
        "cut from ufc",
        "released",
        "released from ufc",
    }
)
_PROFILE_STATUS_KEYS = _PROFILE_CURRENT_STATUS_KEYS | _PROFILE_INACTIVE_STATUS_KEYS
_CACHED_STABLE_PROFILE_BIOGRAPHY_FIELDS = (
    "birthplace",
    "height",
    "reach",
    "weight",
    "octagon_debut",
)
ACCEPTED_IMMUTABLE_ROSTER_PROFILE_FIELDS = (
    "octagon_debut",
    "height",
    "reach",
    "dob",
    "stance",
)
_ACCEPTED_IMMUTABLE_PROFILE_FIELDS = ACCEPTED_IMMUTABLE_ROSTER_PROFILE_FIELDS
ACTIVE_ROSTER_CACHED_BIOGRAPHY_PROVENANCE_COLUMNS = (
    "official_profile_cached_biography_fields",
    "official_profile_biography_restoration_reason",
)
_CACHED_PROFILE_BIOGRAPHY_FIELDS_COLUMN = (
    ACTIVE_ROSTER_CACHED_BIOGRAPHY_PROVENANCE_COLUMNS[0]
)
_CACHED_PROFILE_BIOGRAPHY_REASON_COLUMN = (
    ACTIVE_ROSTER_CACHED_BIOGRAPHY_PROVENANCE_COLUMNS[1]
)
_CACHED_PROFILE_BIOGRAPHY_REASON = (
    "cached_stable_biography_after_profile_request_failure"
)
_CACHED_PROFILE_BIOGRAPHY_SUCCESS_CONFLICT_REASON = (
    "cached_accepted_biography_after_successful_profile_conflict"
)
_RETAINED_ROSTER_MAX_CONSECUTIVE_MISSING_SYNCS = 3
_ROSTER_GROWTH_FALLBACK_MIN_CACHED_ROWS = 100
_ROSTER_GROWTH_FALLBACK_RATIO = 1.35
_ROSTER_GROWTH_FALLBACK_ABSOLUTE_ROWS = 250
_ROSTER_SHRINK_INCOMPLETE_MIN_CACHED_ROWS = 100
_ROSTER_SHRINK_INCOMPLETE_RATIO = 0.95
_ROSTER_SHRINK_INCOMPLETE_ABSOLUTE_ROWS = 50
_ROSTER_DESTRUCTIVE_EXCLUSION_RATIO = 0.05
_ROSTER_DESTRUCTIVE_EXCLUSION_ABSOLUTE_ROWS = 50
_PROFILE_STATUS_DRIFT_MIN_COMPARISON_ROWS = 50
_PROFILE_STATUS_DRIFT_MIN_LOST_ROWS = 50
_PROFILE_STATUS_DRIFT_LOST_RATIO = 0.80
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
_OFFICIAL_PROFILE_BINDING_STATUS = "validated_profile_decision_binding"
_OFFICIAL_PROFILE_BINDING_STATUS_COLUMN = "official_profile_identity_binding_status"
_OFFICIAL_PROFILE_BINDING_STABLE_ID_COLUMN = (
    "official_profile_binding_stable_ufcstats_id"
)
_OFFICIAL_PROFILE_BINDING_PREFIX = "official_profile_binding_"
_OFFICIAL_PROFILE_BINDING_SUFFIXES = (
    "source_entity_id",
    "requested_url",
    "source_url",
    "canonical_url",
    "redirect_chain",
    "canonical_name",
    "display_name",
    "identity_status",
    "name_match_status",
    "identity_basis",
    "identity_evidence_url",
    "identity_event_date",
    "identity_opponent_name",
    "identity_opponent_ufcstats_id",
    "identity_organization",
    "identity_fight_url",
    "identity_ufcstats_event_date",
    "identity_ufcstats_opponent_name",
    "identity_opponent_name_match_status",
)
_OFFICIAL_PROFILE_FRESH_TRANSPORT_COLUMNS = (
    "official_profile_fresh_requested_url",
    "official_profile_fresh_effective_url",
    "official_profile_fresh_canonical_url",
    "official_profile_fresh_redirect_chain",
)
OFFICIAL_PROFILE_IDENTITY_PROVENANCE_COLUMNS = (
    _OFFICIAL_PROFILE_BINDING_STATUS_COLUMN,
    _OFFICIAL_PROFILE_BINDING_STABLE_ID_COLUMN,
    *(
        f"{_OFFICIAL_PROFILE_BINDING_PREFIX}{suffix}"
        for suffix in _OFFICIAL_PROFILE_BINDING_SUFFIXES
    ),
    *_OFFICIAL_PROFILE_FRESH_TRANSPORT_COLUMNS,
)
_OFFICIAL_PROFILE_BINDING_PROVENANCE_COLUMNS = (
    _OFFICIAL_PROFILE_BINDING_STATUS_COLUMN,
    _OFFICIAL_PROFILE_BINDING_STABLE_ID_COLUMN,
    *(
        f"{_OFFICIAL_PROFILE_BINDING_PREFIX}{suffix}"
        for suffix in _OFFICIAL_PROFILE_BINDING_SUFFIXES
    ),
)
_UNBOUND_SIMULTANEOUS_ROUTE_OBSERVATIONS_ATTR = (
    "reviewed_unbound_simultaneous_route_observations"
)
_VERIFIED_INACTIVE_PROFILE_OBSERVATIONS_ATTR = (
    "verified_inactive_profile_observations"
)
_RECONCILED_SIMULTANEOUS_ACTIVE_OWNER_TRANSITIONS_ATTR = (
    "reconciled_simultaneous_active_owner_transitions"
)
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
    "ufcstats_url",
    "profile_name",
    "slug_name",
    "alternate_slug_names",
    "identity_status",
    "identity_reason",
    *OFFICIAL_PROFILE_IDENTITY_PROVENANCE_COLUMNS,
    "action",
    ACTIVE_ROSTER_SYNC_GENERATION_COLUMN,
)
RETAINED_ROSTER_COLUMNS = (
    "active_roster_live_present",
    "active_roster_retained_from_previous",
    "active_roster_retained_at_utc",
    "active_roster_first_missing_at_utc",
    "active_roster_consecutive_missing_syncs",
    "active_roster_missing_from_live_reason",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_active_roster_provenance_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Materialize provenance columns at their stable artifact position."""

    normalized = frame.copy()
    for column in ACTIVE_ROSTER_CACHED_BIOGRAPHY_PROVENANCE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""

    columns = [
        column
        for column in normalized.columns
        if column not in ACTIVE_ROSTER_CACHED_BIOGRAPHY_PROVENANCE_COLUMNS
    ]
    if "active_roster_status_reason" in columns:
        position = columns.index("active_roster_status_reason") + 1
    else:
        lifecycle_positions = [
            columns.index(column)
            for column in (*RETAINED_ROSTER_COLUMNS, ACTIVE_ROSTER_SYNC_GENERATION_COLUMN)
            if column in columns
        ]
        position = min(lifecycle_positions, default=len(columns))
    columns[position:position] = list(
        ACTIVE_ROSTER_CACHED_BIOGRAPHY_PROVENANCE_COLUMNS
    )
    return normalized.reindex(columns=columns)


class OfficialUFCProfileResponseUnavailableError(RuntimeError):
    """Raised when a successful HTTP response is not a UFC athlete profile."""


def _clean_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _optional_text(value: object) -> str:
    """Return persisted optional text without leaking pandas NaN sentinels."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "nat"} else text


def _directories_target_same_location(left: Path, right: Path) -> bool:
    try:
        if left.resolve(strict=False) == right.resolve(strict=False):
            return True
    except OSError:
        pass
    try:
        if left.exists() and right.exists() and left.samefile(right):
            return True
    except OSError:
        pass
    return False


def _paths_same_directory_entry(left: Path, right: Path) -> bool:
    """Return whether atomic replacement would update the same directory entry."""
    left = Path(left)
    right = Path(right)
    if (
        os.path.normcase(left.name) == os.path.normcase(right.name)
        and _directories_target_same_location(left.parent, right.parent)
    ):
        return True
    try:
        # Windows may resolve an 8.3 leaf alias to its long canonical name.
        # A file symlink is different: atomic replacement detaches the symlink
        # itself and does not update its target's directory entry.
        if left.is_symlink() or right.is_symlink():
            return False
        resolved_left = left.resolve(strict=False)
        resolved_right = right.resolve(strict=False)
        return (
            os.path.normcase(resolved_left.name)
            == os.path.normcase(resolved_right.name)
            and _directories_target_same_location(
                resolved_left.parent,
                resolved_right.parent,
            )
        )
    except OSError:
        return False


def _paths_target_same_file(left: Path, right: Path) -> bool:
    """Conservatively compare existing targets, including hardlink aliases."""
    left = Path(left)
    right = Path(right)
    if _paths_same_directory_entry(left, right):
        return True
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def _status_key(value: object) -> str:
    return normalize_cross_source_name(value)


def _identity_status_key(value: object) -> str:
    return _status_key(
        _optional_text(value).replace("_", " ").replace("-", " ")
    )


def _get_soup(url: str, *, session: requests.Session | None = None) -> BeautifulSoup:
    client = session or requests.Session()
    last_exc: Exception | None = None
    for attempt in range(1, _REQUEST_MAX_RETRIES + 1):
        try:
            response = client.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            time.sleep(_REQUEST_DELAY_SECONDS)
            soup = BeautifulSoup(response.text, "lxml")
            history = getattr(response, "history", ()) or ()
            history_urls = [
                _optional_text(getattr(hop, "url", "")) for hop in history
            ]
            effective_url = _optional_text(getattr(response, "url", ""))
            redirect_chain = (
                [*history_urls, effective_url]
                if history_urls and effective_url
                else history_urls
            )
            # BeautifulSoup's public tree remains the return value for existing
            # callers.  Transport evidence is attached out of band so the
            # profile parser can retain the route that produced the document.
            soup.__dict__["_ufc_transport_metadata"] = {
                "requested_url": _optional_text(url),
                "effective_url": effective_url,
                "redirect_chain": json.dumps(
                    redirect_chain,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
            return soup
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ) as exc:
            last_exc = exc
            if isinstance(exc, requests.exceptions.HTTPError):
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code not in _REQUEST_RETRY_STATUS_CODES:
                    raise
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
    return _explicit_boolean(value) is True


def _falsey(value: object) -> bool:
    return _explicit_boolean(value) is False


_REVIEWED_UFCSTATS_CARRIER_FIELDS = (
    "stable_ufcstats_id",
    "fighter_id",
    "ufcstats_url",
)


def _strict_reviewed_profile_identity_binding(
    row: Mapping[str, object],
) -> dict[str, object] | None:
    """Require a nonblank reviewed display name in addition to stable carriers."""

    binding = reviewed_profile_identity_binding(row)
    observed_name = normalize_person_name(row.get("official_name"))
    allowed_names = (
        {
            normalize_person_name(name)
            for name in binding["reviewed_official_names"]
        }
        if binding is not None
        else set()
    )
    return binding if observed_name and observed_name in allowed_names else None


def _reviewed_binding_without_ufcstats_carriers(
    row: Mapping[str, object],
) -> dict[str, object] | None:
    """Validate the official entity/name independently of stale ID carriers."""

    carrierless = dict(row)
    for field in _REVIEWED_UFCSTATS_CARRIER_FIELDS:
        carrierless[field] = ""
    return _strict_reviewed_profile_identity_binding(carrierless)


def _clear_reviewed_ufcstats_carriers(
    row: dict[str, object],
    *,
    resolution: str,
) -> None:
    """Detach an unapproved UFCStats identity without excluding the UFC row."""

    for field in _REVIEWED_UFCSTATS_CARRIER_FIELDS:
        if field in row or field == "ufcstats_url":
            row[field] = ""
    row.update(
        {
            "ufcstats_name": "",
            "ufcstats_resolution": resolution,
            "combat_sport": "mma",
            "combat_sport_reason": "official_profile_mma",
            "combat_sport_profile_url": "",
        }
    )


def _apply_reviewed_roster_identity_resolution_policy(
    row: dict[str, object],
    *,
    expected_official_entity: str = "",
) -> tuple[bool, bool]:
    """Apply one approved link or fail closed before name-based discovery.

    Returns ``(binding_accepted, generic_resolution_blocked)``.  Exact UFC
    official entity and reviewed display-name evidence may repair stale ID
    carriers; a changed/unapproved official display name never can.
    """

    current_official_entity = source_entity_id(
        "ufc",
        row.get("official_athlete_url"),
    )
    if (
        expected_official_entity
        and current_official_entity != expected_official_entity
    ):
        _clear_reviewed_ufcstats_carriers(
            row,
            resolution="reviewed_official_identity_route_changed",
        )
        return False, True

    rejection = reviewed_ufcstats_identity_rejection(row)
    binding_record = reviewed_profile_identity_binding_record(row)
    binding = _strict_reviewed_profile_identity_binding(row)

    # A conflicting stable carrier must not prevent an otherwise exact,
    # owner-approved official entity/name binding from replacing discovery.
    if binding is None and binding_record is not None:
        binding = _reviewed_binding_without_ufcstats_carriers(row)
        if binding is not None:
            _clear_reviewed_ufcstats_carriers(
                row,
                resolution="reviewed_roster_identity_rebinding",
            )

    if binding is not None and binding["stable_ufcstats_id"]:
        if not apply_reviewed_profile_identity_binding(row):
            raise ValueError(
                "Reviewed UFC roster identity binding failed after its exact "
                "official entity and name were validated"
            )
        expected_stable_id = str(binding["stable_ufcstats_id"])
        for field in ("stable_ufcstats_id", "fighter_id"):
            if field in row:
                row[field] = expected_stable_id
        row["combat_sport_profile_url"] = ""
        return True, False

    if binding is not None:
        _clear_reviewed_ufcstats_carriers(
            row,
            resolution="reviewed_official_only_identity",
        )
        return True, True

    blocked = bool(
        binding_record is not None
        or rejection is not None
        or reviewed_ufcstats_resolution_blocked(row)
    )
    if blocked:
        _clear_reviewed_ufcstats_carriers(
            row,
            resolution=(
                "reviewed_ufcstats_identity_rejection"
                if rejection is not None
                else "reviewed_ufcstats_resolution_blocked"
            ),
        )
    return False, blocked


def _reviewed_profile_name_authorized(
    row: Mapping[str, object],
    candidate_name: object,
) -> bool:
    """Accept only a contextual alias from the exact reviewed UFC binding."""

    binding = _strict_reviewed_profile_identity_binding(row)
    candidate_key = normalize_person_name(candidate_name)
    if binding is None or not candidate_key:
        return False
    allowed_names = {
        normalize_person_name(name)
        for name in binding["reviewed_official_names"]
    }
    ufcstats_name = normalize_person_name(binding.get("ufcstats_name"))
    if ufcstats_name:
        allowed_names.add(ufcstats_name)
    return candidate_key in allowed_names


def _supported_ufcstats_profile_id(value: object) -> str:
    """Return the stable ID only for the intentionally supported profile URL."""

    text = _optional_text(value)
    parsed = urlparse(text)
    match = re.fullmatch(
        r"/fighter-details/([0-9a-f]{16})/?",
        parsed.path,
        flags=re.IGNORECASE,
    )
    if (
        parsed.scheme.casefold() != "http"
        or (parsed.hostname or "").casefold().rstrip(".") != "ufcstats.com"
        or match is None
        or parsed.query
        or parsed.fragment
    ):
        return ""
    return match.group(1).casefold()


def _trusted_ufc_profile_url(value: object) -> str:
    text = _optional_text(value)
    parsed = urlparse(text)
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    if (
        parsed.scheme.casefold() != "https"
        or hostname not in {"ufc.com", "www.ufc.com"}
        or parsed.query
        or parsed.fragment
    ):
        return ""
    entity_id = source_entity_id("ufc", text)
    return text.rstrip("/") if entity_id else ""


def _binding_column(suffix: str) -> str:
    return f"{_OFFICIAL_PROFILE_BINDING_PREFIX}{suffix}"


def _clear_official_profile_binding(row: dict[str, object]) -> None:
    for column in (
        _OFFICIAL_PROFILE_BINDING_STATUS_COLUMN,
        _OFFICIAL_PROFILE_BINDING_STABLE_ID_COLUMN,
        *(
            _binding_column(suffix)
            for suffix in _OFFICIAL_PROFILE_BINDING_SUFFIXES
        ),
    ):
        row[column] = ""


def _captured_redirect_chain(value: object) -> tuple[str, ...] | None:
    """Return one complete JSON redirect chain or fail closed."""

    text = _optional_text(value)
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None
    redirects: list[str] = []
    for candidate in parsed:
        url = _optional_text(candidate)
        if not url:
            return None
        redirects.append(url)
    return tuple(redirects)


def _fresh_ufc_transport_matches_binding(
    row: pd.Series | dict[str, object],
    binding: dict[str, str],
) -> bool:
    """Require the newly fetched UFC route to equal the reviewed route."""

    fresh_requested = _optional_text(
        row.get("official_profile_fresh_requested_url")
    )
    fresh_effective = _optional_text(
        row.get("official_profile_fresh_effective_url")
    )
    fresh_canonical = _optional_text(
        row.get("official_profile_fresh_canonical_url")
    )
    binding_requested = _optional_text(binding.get("requested_url"))
    binding_effective = _optional_text(binding.get("source_url"))
    binding_canonical = _optional_text(binding.get("canonical_url"))
    route_pairs = (
        (fresh_requested, binding_requested),
        (fresh_effective, binding_effective),
        (fresh_canonical, binding_canonical),
    )
    if any(
        not fresh
        or not reviewed
        or fresh != reviewed
        or not _trusted_ufc_profile_url(fresh)
        or not _trusted_ufc_profile_url(reviewed)
        for fresh, reviewed in route_pairs
    ):
        return False

    entity_id = _optional_text(binding.get("source_entity_id"))
    if not entity_id or any(
        source_entity_id("ufc", value) != entity_id
        for pair in route_pairs
        for value in pair
    ):
        return False

    fresh_redirects = _captured_redirect_chain(
        row.get("official_profile_fresh_redirect_chain")
    )
    reviewed_redirects = _captured_redirect_chain(binding.get("redirect_chain"))
    if fresh_redirects is None or reviewed_redirects is None:
        return False
    if fresh_redirects != reviewed_redirects:
        return False
    if fresh_redirects and fresh_redirects[-1] != fresh_effective:
        return False
    return all(
        _trusted_ufc_profile_url(redirect_url)
        and source_entity_id("ufc", redirect_url) == entity_id
        for redirect_url in fresh_redirects
    )


def _fresh_ufc_names_match_binding(
    row: pd.Series | dict[str, object],
    binding: dict[str, str],
) -> bool:
    """Match fresh page names only to names explicitly carried by the binding."""

    official_name = _clean_text(row.get("official_name"))
    profile_name = _clean_text(row.get("profile_name"))
    canonical_name = _clean_text(binding.get("canonical_name"))
    display_name = _clean_text(binding.get("display_name"))
    status = _identity_status_key(binding.get("name_match_status"))
    if (
        not official_name
        or not profile_name
        or not canonical_name
        or not display_name
        or status not in {"exact", "approved alias"}
    ):
        return False

    allowed = {normalize_person_name(canonical_name)}
    if status == "approved alias":
        allowed.add(normalize_person_name(display_name))
    elif normalize_person_name(display_name) != normalize_person_name(
        canonical_name
    ):
        return False
    return all(
        normalize_person_name(value) in allowed
        for value in (official_name, profile_name)
    )


def _validated_ufc_profile_binding_index(
    decisions: pd.DataFrame | None,
) -> dict[str, list[dict[str, str]]]:
    """Index UFC.com bindings only after the complete registry validates.

    ``validate_profile_decisions`` enforces the source entity, canonical and
    redirect routes, independent UFC fight anchor, and both directions of the
    source-entity/UFCStats one-to-one mapping.  A name search never enters this
    index.
    """

    if decisions is None or decisions.empty:
        return {}
    issues = validate_profile_decisions(decisions)
    if issues:
        codes = sorted({issue.code for issue in issues})
        raise ValueError(
            "UFC profile identity decisions failed closed: " + ",".join(codes)
        )

    verified = verified_binding_keys(decisions)
    by_stable_id: dict[str, dict[tuple[str, ...], dict[str, str]]] = {}
    for _, decision in decisions.iterrows():
        stable_id = normalize_ufcstats_id(decision.get("stable_ufcstats_id")) or ""
        for prefix in ("primary", "corroborating"):
            source = (
                _optional_text(decision.get(f"{prefix}_source"))
                .casefold()
                .replace(".com", "")
            )
            entity_id = _optional_text(
                decision.get(f"{prefix}_source_entity_id")
            )
            if (stable_id, source, entity_id) not in verified or source != "ufc":
                continue
            binding = {
                suffix: _optional_text(
                    decision.get("canonical_name")
                    if suffix == "canonical_name"
                    else decision.get(f"{prefix}_{suffix}")
                )
                for suffix in _OFFICIAL_PROFILE_BINDING_SUFFIXES
            }
            fingerprint = tuple(
                binding[suffix] for suffix in _OFFICIAL_PROFILE_BINDING_SUFFIXES
            )
            by_stable_id.setdefault(stable_id, {})[fingerprint] = binding

    return {
        stable_id: [records[key] for key in sorted(records)]
        for stable_id, records in by_stable_id.items()
    }


def _apply_verified_ufc_profile_binding(
    row: dict[str, object],
    *,
    bindings_by_stable_id: dict[str, list[dict[str, str]]],
) -> None:
    """Attach one exact prevalidated binding to a freshly resolved roster row."""

    _clear_official_profile_binding(row)
    stable_id = _supported_ufcstats_profile_id(row.get("ufcstats_url"))
    current_url = _trusted_ufc_profile_url(row.get("official_athlete_url"))
    current_entity = source_entity_id("ufc", current_url) if current_url else ""
    if not stable_id or not current_entity:
        return

    matches = []
    for binding in bindings_by_stable_id.get(stable_id, []):
        canonical_url = _trusted_ufc_profile_url(binding.get("canonical_url"))
        source_url = _trusted_ufc_profile_url(binding.get("source_url"))
        if (
            binding.get("source_entity_id") == current_entity
            and source_entity_id("ufc", canonical_url) == current_entity
            and source_entity_id("ufc", source_url) == current_entity
            and _normalize_roster_url(canonical_url)
            == _normalize_roster_url(current_url)
            and _fresh_ufc_transport_matches_binding(row, binding)
            and _fresh_ufc_names_match_binding(row, binding)
        ):
            matches.append(binding)
    if len(matches) != 1:
        return

    binding = matches[0]
    row[_OFFICIAL_PROFILE_BINDING_STATUS_COLUMN] = _OFFICIAL_PROFILE_BINDING_STATUS
    row[_OFFICIAL_PROFILE_BINDING_STABLE_ID_COLUMN] = stable_id
    for suffix in _OFFICIAL_PROFILE_BINDING_SUFFIXES:
        row[_binding_column(suffix)] = binding[suffix]


def _official_profile_binding_positively_verified(
    row: pd.Series | dict[str, object],
) -> bool:
    """Recheck the persisted certificate attached by the validated registry."""

    carrier_ids = _present_ufcstats_carrier_ids(row)
    if carrier_ids is None or len(set(carrier_ids)) > 1:
        return False
    if reviewed_mma_identity_binding_is_complete(row):
        return True
    if _strict_reviewed_profile_identity_binding(row) is not None:
        return True
    if (
        _optional_text(row.get(_OFFICIAL_PROFILE_BINDING_STATUS_COLUMN))
        != _OFFICIAL_PROFILE_BINDING_STATUS
    ):
        return False
    stable_id = _supported_ufcstats_profile_id(row.get("ufcstats_url"))
    if (
        not stable_id
        or _optional_text(row.get(_OFFICIAL_PROFILE_BINDING_STABLE_ID_COLUMN))
        != stable_id
    ):
        return False

    entity_id = _optional_text(row.get(_binding_column("source_entity_id")))
    current_url = _trusted_ufc_profile_url(row.get("official_athlete_url"))
    canonical_url = _trusted_ufc_profile_url(
        row.get(_binding_column("canonical_url"))
    )
    source_url = _trusted_ufc_profile_url(row.get(_binding_column("source_url")))
    evidence_url = _trusted_ufc_profile_url(
        row.get(_binding_column("identity_evidence_url"))
    )
    if (
        not entity_id
        or not current_url
        or _normalize_roster_url(current_url) != _normalize_roster_url(canonical_url)
        or any(
            source_entity_id("ufc", value) != entity_id
            for value in (current_url, canonical_url, source_url, evidence_url)
        )
    ):
        return False

    binding = {
        suffix: _optional_text(row.get(_binding_column(suffix)))
        for suffix in _OFFICIAL_PROFILE_BINDING_SUFFIXES
    }
    if not _fresh_ufc_transport_matches_binding(row, binding):
        return False
    if not _fresh_ufc_names_match_binding(row, binding):
        return False

    identity_status = _identity_status_key(
        row.get(_binding_column("identity_status"))
    )
    name_status = _identity_status_key(
        row.get(_binding_column("name_match_status"))
    )
    basis = _identity_status_key(row.get(_binding_column("identity_basis")))
    if (
        identity_status not in {"verified", "reviewed"}
        or name_status not in {"exact", "approved alias"}
        or basis
        not in {
            "accepted reviewed binding",
            "event date opponent",
            "event date opponent organization",
        }
        or not _optional_text(row.get(_binding_column("redirect_chain")))
    ):
        return False

    source_date = pd.to_datetime(
        row.get(_binding_column("identity_event_date")), errors="coerce"
    )
    ufcstats_date = pd.to_datetime(
        row.get(_binding_column("identity_ufcstats_event_date")), errors="coerce"
    )
    opponent_id = normalize_ufcstats_id(
        row.get(_binding_column("identity_opponent_ufcstats_id"))
    )
    source_opponent = _optional_text(
        row.get(_binding_column("identity_opponent_name"))
    )
    ufcstats_opponent = _optional_text(
        row.get(_binding_column("identity_ufcstats_opponent_name"))
    )
    opponent_name_status = _identity_status_key(
        row.get(_binding_column("identity_opponent_name_match_status"))
    )
    fight_url = _optional_text(row.get(_binding_column("identity_fight_url")))
    parsed_fight = urlparse(fight_url)
    if (
        pd.isna(source_date)
        or pd.isna(ufcstats_date)
        or pd.Timestamp(source_date).normalize()
        != pd.Timestamp(ufcstats_date).normalize()
        or not opponent_id
        or opponent_id == stable_id
        or not source_opponent
        or not ufcstats_opponent
        or opponent_name_status not in {"exact", "approved alias"}
        or (
            opponent_name_status == "exact"
            and source_opponent.casefold() != ufcstats_opponent.casefold()
        )
        or _identity_status_key(
            row.get(_binding_column("identity_organization"))
        )
        != "ufc"
        or parsed_fight.scheme.casefold() != "http"
        or (parsed_fight.hostname or "").casefold().rstrip(".") != "ufcstats.com"
        or re.fullmatch(
            r"/fight-details/[0-9a-f]{16}/?",
            parsed_fight.path,
            flags=re.IGNORECASE,
        )
        is None
        or parsed_fight.query
        or parsed_fight.fragment
    ):
        return False
    return True


def _official_url_identity_trusted(row: dict[str, object]) -> bool:
    """Return whether slug/profile aliases from UFC.com are safe identity aliases."""
    return _official_url_identity_positively_trusted(row)


def _official_url_identity_positively_trusted(
    row: pd.Series | dict[str, object],
) -> bool:
    """Require positive evidence before using a UFC.com/UFCStats identity binding."""
    status = _identity_status_key(row.get("official_url_identity_status"))
    explicit = row.get("official_url_identity_valid")
    if status in {"mismatch", "test profile"} or _falsey(explicit):
        return False
    if status not in {"valid", "slug mismatch profile valid"} and not _truthy(
        explicit
    ):
        return False
    return _official_profile_binding_positively_verified(row)


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

    slug_mismatch = False
    if official_name:
        profile_matches = bool(profile_name) and (
            _same_identity_name(official_name, profile_name)
            or _reviewed_profile_name_authorized(row, profile_name)
        )
        slug_matches = bool(slug_name) and (
            _same_identity_name(official_name, slug_name)
            or _reviewed_profile_name_authorized(row, slug_name)
        )
        if profile_name and not profile_matches:
            reasons.append(f"profile_name_mismatch:{profile_name}")
        if slug_name and not slug_matches:
            if profile_matches:
                slug_mismatch = True
            else:
                reasons.append(f"slug_name_mismatch:{slug_name}")
    else:
        reasons.append("official_name_missing")

    if reasons:
        return {
            "official_url_identity_valid": False,
            "official_url_identity_status": "mismatch",
            "official_url_identity_reason": "|".join(reasons),
        }

    if not _official_profile_binding_positively_verified(row):
        return {
            "official_url_identity_valid": False,
            "official_url_identity_status": "unresolved_identity",
            "official_url_identity_reason": (
                "name_slug_match_is_discovery_only_without_validated_"
                "source_entity_and_event_binding"
            ),
        }

    return {
        "official_url_identity_valid": True,
        "official_url_identity_status": (
            "slug_mismatch_profile_valid" if slug_mismatch else "valid"
        ),
        "official_url_identity_reason": (
            f"slug_name_mismatch:{slug_name}" if slug_mismatch else ""
        ),
    }


def _identity_audit_row(row: dict[str, object], *, action: str) -> dict[str, object]:
    audit_row = {
        "official_name": _clean_text(row.get("official_name")),
        "official_athlete_url": _clean_text(row.get("official_athlete_url")),
        "ufcstats_url": _clean_text(row.get("ufcstats_url")),
        "profile_name": _clean_text(row.get("profile_name")),
        "slug_name": _clean_text(row.get("slug_name")),
        "alternate_slug_names": _clean_text(row.get("alternate_slug_names")),
        "identity_status": _clean_text(row.get("official_url_identity_status")),
        "identity_reason": _clean_text(row.get("official_url_identity_reason")),
        "action": action,
    }
    audit_row.update(
        {
            column: _optional_text(row.get(column))
            for column in OFFICIAL_PROFILE_IDENTITY_PROVENANCE_COLUMNS
        }
    )
    return audit_row


def _quarantine_reviewed_mma_identity_conflict(
    row: dict[str, object],
    *,
    reason: str,
) -> None:
    """Remove all usable data from a registered identity that failed binding."""

    row.update(
        {
            _PROFILE_FETCH_OUTCOME_COLUMN: _PROFILE_FETCH_NOT_REQUESTED,
            "official_url_identity_valid": False,
            "official_url_identity_status": "reviewed_mma_identity_conflict",
            "official_url_identity_reason": reason,
            "combat_sport": "mma",
            "combat_sport_reason": "quarantined_reviewed_mma_identity_conflict",
            "combat_sport_profile_url": "",
            "coverage_eligible": False,
            "active_roster_status_reason": "reviewed_mma_identity_conflict",
        }
    )
    for field in (
        "profile_name",
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
        "stable_ufcstats_id",
        "fighter_id",
        "ufcstats_name",
        "ufcstats_url",
        "ufcstats_resolution",
    ):
        row[field] = ""


def _quarantine_untrusted_official_profile_fields(row: dict[str, object]) -> None:
    for field in _UNTRUSTED_OFFICIAL_PROFILE_FIELDS:
        row[field] = ""


def _quarantine_untrusted_slug_alias_fields(row: dict[str, object]) -> None:
    for field in _UNTRUSTED_SLUG_ALIAS_FIELDS:
        row[field] = ""


def _verified_inactive_profile_status(row: pd.Series | dict[str, object]) -> bool:
    if not _official_url_identity_positively_trusted(row):
        return False
    status_key = _status_key(row.get("profile_status"))
    return bool(status_key in _PROFILE_INACTIVE_STATUS_KEYS)


def _inactive_profile_status(row: pd.Series | dict[str, object]) -> bool:
    status_key = _status_key(row.get("profile_status"))
    return bool(status_key in _PROFILE_INACTIVE_STATUS_KEYS)


def _profile_status_is_unknown(row: pd.Series | dict[str, object]) -> bool:
    return _status_key(row.get("profile_status")) not in _PROFILE_STATUS_KEYS


def _profile_status_is_independently_verified_current(
    row: pd.Series | dict[str, object],
) -> bool:
    """Allow an explicit independent verifier to override missing UFC profile status."""
    return _truthy(row.get("active_roster_current_verified"))


def _explicit_boolean(value: object) -> bool | None:
    """Parse a persisted boolean without treating blanks or unknown text as false."""
    if isinstance(value, bool):
        return value
    if isinstance(value, Real):
        numeric = float(value)
        if numeric == 1.0:
            return True
        if numeric == 0.0:
            return False
    text = _optional_text(value).casefold()
    if text in {"true", "yes", "on"}:
        return True
    if text in {"false", "no", "off"}:
        return False
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return None
    if numeric == 1.0:
        return True
    if numeric == 0.0:
        return False
    return None


def _official_url_identity_explicitly_untrusted(
    row: pd.Series | dict[str, object],
) -> bool:
    status = _identity_status_key(row.get("official_url_identity_status"))
    return status in {"mismatch", "test profile"} or _explicit_boolean(
        row.get("official_url_identity_valid")
    ) is False


def _row_has_power_slap_evidence(
    row: pd.Series | dict[str, object],
) -> bool:
    evidence = " ".join(
        _optional_text(row.get(column))
        for column in (
            "combat_sport",
            "combat_sport_reason",
            "combat_sport_profile_url",
        )
    ).casefold()
    compact = "".join(character for character in evidence if character.isalnum())
    return "powerslap" in compact


def _row_is_positively_classified_mma(
    row: pd.Series | dict[str, object],
) -> bool:
    return (
        _status_key(row.get("combat_sport")) == "mma"
        and not _row_has_power_slap_evidence(row)
    )


def _row_is_positively_verified_current_mma(
    row: pd.Series | dict[str, object],
) -> bool:
    return (
        _status_key(row.get("profile_status")) in _PROFILE_CURRENT_STATUS_KEYS
        and _official_url_identity_positively_trusted(row)
        and _row_is_positively_classified_mma(row)
    )


def _apply_profile_status_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """Quarantine rows whose current UFC status was not positively verified.

    UFC.com's athlete listing has returned archive rows despite the active-only
    query parameter, so listing presence alone is not evidence that a fighter is
    current.  Unknown rows remain available for identity/audit purposes but must
    not enter active-roster coverage calculations.
    """
    if df.empty:
        return df

    normalized = df.copy()
    original_index = normalized.index.copy()
    normalized = normalized.reset_index(drop=True)
    if "profile_status" not in normalized.columns:
        normalized["profile_status"] = pd.Series(
            "",
            index=normalized.index,
            dtype="object",
        )
    else:
        normalized["profile_status"] = normalized["profile_status"].astype(
            "object"
        )
    if "coverage_eligible" not in normalized.columns:
        normalized["coverage_eligible"] = pd.Series(
            "",
            index=normalized.index,
            dtype="object",
        )
    else:
        normalized["coverage_eligible"] = normalized[
            "coverage_eligible"
        ].astype("object")
    if "active_roster_status_reason" in normalized.columns:
        normalized["active_roster_status_reason"] = normalized[
            "active_roster_status_reason"
        ].astype("object")

    def set_status_reason(index: object, reason: str) -> None:
        if "active_roster_status_reason" not in normalized.columns:
            normalized["active_roster_status_reason"] = pd.Series(
                "",
                index=normalized.index,
                dtype="object",
            )
        normalized.at[index, "active_roster_status_reason"] = reason

    for index, row in normalized.iterrows():
        if (
            _optional_text(row.get("active_roster_status_reason"))
            == "rosterwatch_authoritative_current"
        ):
            # Roster Watch is authoritative for membership.  A stale UFC
            # profile-status label cannot override that fact; profile fields
            # remain evidence and explicit coverage eligibility is preserved.
            # Positively identified non-MMA evidence still fails closed.
            if _row_has_power_slap_evidence(row):
                normalized.at[index, "coverage_eligible"] = False
                set_status_reason(index, "power_slap_combat_sport")
                continue
            explicit_eligibility = _explicit_boolean(
                row.get("coverage_eligible")
            )
            normalized.at[index, "coverage_eligible"] = bool(
                explicit_eligibility
            )
            continue
        if _inactive_profile_status(row):
            normalized.at[index, "coverage_eligible"] = False
            set_status_reason(
                index,
                "verified_inactive_profile_status"
                if _verified_inactive_profile_status(row)
                else "unverified_inactive_profile_status",
            )
            continue

        if _profile_status_is_unknown(row):
            if _profile_status_is_independently_verified_current(row):
                normalized.at[index, "profile_status"] = "Active"
                set_status_reason(index, "independently_verified_current")
                row = normalized.loc[index]
            else:
                normalized.at[index, "profile_status"] = _PROFILE_STATUS_UNKNOWN
                normalized.at[index, "coverage_eligible"] = False
                set_status_reason(
                    index,
                    "profile_status_missing_or_unrecognized",
                )
                continue

        if _row_has_power_slap_evidence(row):
            normalized.at[index, "coverage_eligible"] = False
            set_status_reason(index, "power_slap_combat_sport")
            continue
        if (
            _optional_text(row.get("active_roster_status_reason"))
            == "cached_verified_current_after_profile_request_failure"
            and _optional_text(row.get(_PROFILE_FETCH_OUTCOME_COLUMN)).casefold()
            in {_PROFILE_FETCH_REQUEST_FAILED, _PROFILE_FETCH_INVALID_RESPONSE}
            and _explicit_boolean(row.get("coverage_eligible")) is True
        ):
            # This retains an already accepted, positively bound cached status
            # after a transport/challenge failure.  It does not authorize any
            # newly observed profile value from the failed response.
            normalized.at[index, "coverage_eligible"] = True
            continue
        if _official_url_identity_explicitly_untrusted(row):
            normalized.at[index, "coverage_eligible"] = False
            set_status_reason(index, "untrusted_official_profile_identity")
            continue

        explicit_eligibility = _explicit_boolean(row.get("coverage_eligible"))
        if explicit_eligibility is not None:
            normalized.at[index, "coverage_eligible"] = explicit_eligibility
            continue

        repaired_eligibility = _row_is_positively_verified_current_mma(row)
        normalized.at[index, "coverage_eligible"] = repaired_eligibility
        if not repaired_eligibility:
            set_status_reason(index, "coverage_eligibility_missing_or_unrecognized")

    normalized.index = original_index
    return normalized


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


def _same_identity_name(left: object, right: object) -> bool:
    """Use suffix-preserving exact names or an explicitly reviewed alias."""

    left_reviewed_id = reviewed_fighter_identity_id(left)
    right_reviewed_id = reviewed_fighter_identity_id(right)
    if left_reviewed_id is not None or right_reviewed_id is not None:
        return bool(
            left_reviewed_id is not None
            and left_reviewed_id == right_reviewed_id
        )
    left_key = normalize_person_name(left)
    right_key = normalize_person_name(right)
    return bool(left_key and right_key and left_key == right_key)


def _write_identity_audit(
    path: Path | None,
    rows: list[dict[str, object]],
    *,
    sync_generation_id: str,
) -> None:
    if path is None:
        return
    audit_df = _identity_audit_frame(
        rows,
        sync_generation_id=sync_generation_id,
    )
    write_csv_atomically(audit_df, path)


def _identity_audit_frame(
    rows: list[dict[str, object]],
    *,
    sync_generation_id: str,
) -> pd.DataFrame:
    stamped_rows = [
        {**row, ACTIVE_ROSTER_SYNC_GENERATION_COLUMN: sync_generation_id}
        for row in rows
    ]
    return pd.DataFrame(stamped_rows, columns=IDENTITY_AUDIT_COLUMNS)


def _semantic_csv_value(
    value: object,
    *,
    normalize_numeric_text: bool = False,
) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return format(numeric, ".15g")
    if normalize_numeric_text:
        text = str(value).strip()
        if _CSV_NUMERIC_TEXT_RE.fullmatch(text):
            numeric = float(text)
            if numeric.is_integer():
                return str(int(numeric))
            return format(numeric, ".15g")
    return str(value)


def _semantic_csv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    semantic = frame.drop(
        columns=[ACTIVE_ROSTER_SYNC_GENERATION_COLUMN],
        errors="ignore",
    ).copy()
    for column_name in semantic.columns:
        column = semantic[column_name]
        semantic[column_name] = column.map(
            lambda value: _semantic_csv_value(
                value,
                normalize_numeric_text=(
                    column_name in _SEMANTIC_NUMERIC_CSV_COLUMNS
                ),
            )
        )
    return semantic.reset_index(drop=True)


def _unbound_canonical_candidate_changes(
    accepted: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    identity_audit_rows: Sequence[Mapping[str, object]] = (),
    reconciled_simultaneous_inactive_transitions: Sequence[
        Mapping[str, object]
    ] = (),
    reconciled_inherited_duplicate_owner_transitions: Sequence[
        Mapping[str, object]
    ] = (),
    verified_inactive_profile_observations: Sequence[
        Mapping[str, object]
    ] = (),
) -> list[dict[str, object]]:
    """Report candidate rows that lack their own binding and alter acceptance.

    Positive identity is deliberately row-scoped.  One certified fighter must
    never authorize a different unresolved fighter in the same generation.
    """

    accepted_rows = [row for _, row in accepted.iterrows()]
    candidate_rows = [row for _, row in candidate.iterrows()]
    destructive_exclusions = [
        row
        for row in identity_audit_rows
        if _optional_text(row.get("action"))
        in {
            "excluded_test_profile",
            "excluded_inactive_profile_status",
            "excluded_non_mma_identity",
        }
    ]
    def exact_authorized_inactive_transition_exclusion(
        exclusion: Mapping[str, object],
        accepted_row: Mapping[str, object],
    ) -> bool:
        if (
            _optional_text(exclusion.get("action"))
            != "excluded_inactive_profile_status"
            or not reviewed_official_identity_transition_match(
                exclusion,
                accepted_row,
            )
        ):
            return False
        if len(reconciled_simultaneous_inactive_transitions) != 1:
            return False
        reconciliation = reconciled_simultaneous_inactive_transitions[0]
        stable_id = _optional_text(reconciliation.get("stable_ufcstats_id"))
        if (
            stable_id not in _INACTIVE_TRANSITION_RECONCILIATION_STABLE_IDS
            or stable_id
            != _supported_ufcstats_profile_id(exclusion.get("ufcstats_url"))
            or _status_key(reconciliation.get("target_profile_status"))
            != "not fighting"
            or _optional_text(reconciliation.get("reason"))
            != "reviewed_transition_target_verified_not_fighting"
        ):
            return False
        stable_observations = [
            observation
            for observation in verified_inactive_profile_observations
            if _supported_ufcstats_profile_id(observation.get("ufcstats_url"))
            == stable_id
        ]
        if len(stable_observations) != 1:
            return False
        observation = stable_observations[0]
        if (
            _optional_text(
                observation.get(_PROFILE_FETCH_OUTCOME_COLUMN)
            ).casefold()
            != _PROFILE_FETCH_SUCCEEDED
            or _status_key(observation.get("profile_status")) != "not fighting"
            or not _verified_inactive_profile_status(observation)
            or not reviewed_official_identity_transition_match(
                observation,
                accepted_row,
            )
        ):
            return False
        return bool(
            _normalize_roster_url(
                reconciliation.get("source_official_athlete_url")
            )
            == _normalize_roster_url(accepted_row.get("official_athlete_url"))
            and normalize_person_name(reconciliation.get("source_official_name"))
            == normalize_person_name(accepted_row.get("official_name"))
            and _normalize_roster_url(
                reconciliation.get("target_official_athlete_url")
            )
            == _normalize_roster_url(exclusion.get("official_athlete_url"))
            == _normalize_roster_url(observation.get("official_athlete_url"))
            and normalize_person_name(reconciliation.get("target_official_name"))
            == normalize_person_name(exclusion.get("official_name"))
            == normalize_person_name(observation.get("official_name"))
        )

    compared_columns = sorted(
        (
            set(accepted.columns)
            | set(candidate.columns)
        )
        - {ACTIVE_ROSTER_SYNC_GENERATION_COLUMN}
    )
    violations: list[dict[str, object]] = []
    for candidate_position, candidate_row in enumerate(candidate_rows):
        matches = [
            (accepted_position, accepted_row)
            for accepted_position, accepted_row in enumerate(accepted_rows)
            if _cached_carry_forward_identity_match(
                candidate_row,
                accepted_row,
            )
        ]
        if len(matches) > 1:
            violations.append(
                {
                    "candidate_position": candidate_position,
                    "official_name": _optional_text(
                        candidate_row.get("official_name")
                    ),
                    "reason": "candidate_matches_multiple_accepted_rows",
                    "accepted_match_count": len(matches),
                    "changed_columns": [],
                }
            )
            continue
        if _official_profile_binding_positively_verified(candidate_row) or (
            len(matches) == 1
            and _strict_fresh_reviewed_live_cached_transition_match(
                candidate_row,
                matches[0][1],
            )
        ):
            # A fully bound new identity is permitted, while a bound update
            # still has to map one-to-one to any accepted identity it touches.
            # Ravena's exact route replacement carries equivalent one-to-one
            # fresh target and persisted source proof when the accepted CSV
            # predates profile-decision provenance columns.
            continue
        if not matches:
            violations.append(
                {
                    "candidate_position": candidate_position,
                    "official_name": _optional_text(
                        candidate_row.get("official_name")
                    ),
                    "reason": "unbound_new_candidate_row",
                    "accepted_match_count": 0,
                    "changed_columns": [],
                }
            )
            continue
        accepted_position, accepted_row = matches[0]
        changed_columns = [
            column
            for column in compared_columns
            if _semantic_csv_value(
                accepted_row.get(column),
                normalize_numeric_text=(
                    column in _SEMANTIC_NUMERIC_CSV_COLUMNS
                ),
            )
            != _semantic_csv_value(
                candidate_row.get(column),
                normalize_numeric_text=(
                    column in _SEMANTIC_NUMERIC_CSV_COLUMNS
                ),
            )
        ]
        if changed_columns:
            violations.append(
                {
                    "candidate_position": candidate_position,
                    "accepted_position": accepted_position,
                    "official_name": _optional_text(
                        candidate_row.get("official_name")
                    ),
                    "reason": "unbound_accepted_row_changed",
                    "accepted_match_count": 1,
                    "changed_columns": changed_columns,
                }
            )
    for accepted_position, accepted_row in enumerate(accepted_rows):
        candidate_matches = [
            candidate_row
            for candidate_row in candidate_rows
            if _cached_carry_forward_identity_match(
                candidate_row,
                accepted_row,
            )
        ]
        if len(candidate_matches) > 1:
            violations.append(
                {
                    "accepted_position": accepted_position,
                    "official_name": _optional_text(
                        accepted_row.get("official_name")
                    ),
                    "reason": "accepted_row_matches_multiple_candidates",
                    "accepted_match_count": 1,
                    "candidate_match_count": len(candidate_matches),
                    "changed_columns": [],
                }
            )
            continue
        if candidate_matches:
            continue
        if _reviewed_duplicate_owner_candidate_removal_authorized(
            accepted_rows,
            candidate_rows,
            accepted_position=accepted_position,
            reconciliations=reconciled_inherited_duplicate_owner_transitions,
        ):
            continue
        exclusion_matches = [
            exclusion
            for exclusion in destructive_exclusions
            if (
                _cached_carry_forward_identity_match(exclusion, accepted_row)
                or (
                    _optional_text(exclusion.get("action"))
                    == "excluded_non_mma_identity"
                    and reviewed_non_mma_identity_match(exclusion, accepted_row)
                )
            )
        ]
        positively_bound_exclusions = [
            exclusion
            for exclusion in exclusion_matches
            if (
                (
                    not reviewed_official_identity_transition_match(
                        exclusion,
                        accepted_row,
                    )
                    and _official_profile_binding_positively_verified(exclusion)
                    and sum(
                        _cached_carry_forward_identity_match(
                            exclusion,
                            other_accepted_row,
                        )
                        for other_accepted_row in accepted_rows
                    )
                    == 1
                )
                or (
                    exact_authorized_inactive_transition_exclusion(
                        exclusion,
                        accepted_row,
                    )
                    and sum(
                        exact_authorized_inactive_transition_exclusion(
                            exclusion,
                            other_accepted_row,
                        )
                        for other_accepted_row in accepted_rows
                    )
                    == 1
                )
                or (
                    _optional_text(exclusion.get("action"))
                    == "excluded_non_mma_identity"
                    and reviewed_non_mma_exclusion(exclusion) is not None
                    and sum(
                        reviewed_non_mma_identity_match(
                            exclusion,
                            other_accepted_row,
                        )
                        for other_accepted_row in accepted_rows
                    )
                    == 1
                )
            )
        ]
        if (
            len(exclusion_matches) == 1
            and len(positively_bound_exclusions) == 1
        ):
            continue
        violations.append(
            {
                "accepted_position": accepted_position,
                "official_name": _optional_text(
                    accepted_row.get("official_name")
                ),
                "reason": (
                    "accepted_row_removed_without_exclusion_evidence"
                    if not exclusion_matches
                    else "accepted_row_removed_without_positive_exclusion_binding"
                ),
                "accepted_match_count": 1,
                "exclusion_match_count": len(exclusion_matches),
                "positive_exclusion_binding_count": len(
                    positively_bound_exclusions
                ),
                "changed_columns": [],
            }
        )
    return violations


def _single_complete_sync_generation(frame: pd.DataFrame) -> str:
    if frame.empty or ACTIVE_ROSTER_SYNC_GENERATION_COLUMN not in frame.columns:
        return ""
    values = frame[ACTIVE_ROSTER_SYNC_GENERATION_COLUMN].map(_optional_text)
    if values.eq("").any():
        return ""
    unique = values.unique().tolist()
    return unique[0] if len(unique) == 1 else ""


def _matching_existing_roster_audit_generation(
    *,
    roster_path: Path,
    audit_path: Path | None,
    candidate_roster: pd.DataFrame,
    candidate_audit: pd.DataFrame,
) -> str:
    """Return the prior generation only when the complete pair is unchanged."""
    if audit_path is None or not roster_path.exists() or not audit_path.exists():
        return ""
    try:
        existing_roster = pd.read_csv(
            roster_path,
            dtype=object,
            keep_default_na=False,
        )
        existing_audit = pd.read_csv(
            audit_path,
            dtype=object,
            keep_default_na=False,
        )
    except Exception:
        return ""

    roster_generation = _single_complete_sync_generation(existing_roster)
    audit_generation = _single_complete_sync_generation(existing_audit)
    if not roster_generation or roster_generation != audit_generation:
        return ""

    existing_roster_columns = [
        column
        for column in existing_roster.columns
        if column != ACTIVE_ROSTER_SYNC_GENERATION_COLUMN
    ]
    candidate_roster_columns = [
        column
        for column in candidate_roster.columns
        if column != ACTIVE_ROSTER_SYNC_GENERATION_COLUMN
    ]
    if existing_roster_columns != candidate_roster_columns:
        return ""
    existing_audit_columns = [
        column
        for column in existing_audit.columns
        if column != ACTIVE_ROSTER_SYNC_GENERATION_COLUMN
    ]
    candidate_audit_columns = [
        column
        for column in candidate_audit.columns
        if column != ACTIVE_ROSTER_SYNC_GENERATION_COLUMN
    ]
    if existing_audit_columns != candidate_audit_columns:
        return ""
    if not _semantic_csv_frame(existing_roster).equals(
        _semantic_csv_frame(candidate_roster)
    ):
        return ""
    if not _semantic_csv_frame(existing_audit).equals(
        _semantic_csv_frame(candidate_audit)
    ):
        return ""
    return roster_generation


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


def _excluded_identity_rows_from_audit_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return structured identities for accepted destructive audit actions.

    Keeping the rows structured is important: flattening a URL and a name into
    one set lets a common name override a conflicting strong identifier.
    """
    excluded: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("action") or "").strip() not in {
            "excluded_test_profile",
            "excluded_inactive_profile_status",
            "excluded_non_mma_identity",
        }:
            continue
        if _roster_identity_keys(row):
            excluded.append(dict(row))
    return excluded


def _roster_rows_same_identity(
    left: pd.Series | dict[str, object],
    right: pd.Series | dict[str, object],
) -> bool:
    left_url = _normalize_roster_url(left.get("official_athlete_url"))
    right_url = _normalize_roster_url(right.get("official_athlete_url"))
    if left_url and right_url and left_url != right_url:
        return False

    left_stable_id = _supported_ufcstats_profile_id(left.get("ufcstats_url"))
    right_stable_id = _supported_ufcstats_profile_id(right.get("ufcstats_url"))
    if left_stable_id and right_stable_id and left_stable_id != right_stable_id:
        return False

    left_official = _clean_text(left.get("official_name"))
    right_official = _clean_text(right.get("official_name"))
    if left_official and right_official and not _same_identity_name(left_official, right_official):
        return False

    if left_stable_id and left_stable_id == right_stable_id:
        return True
    if left_url and left_url == right_url:
        return True

    left_name_keys = _roster_identity_name_keys(left)
    right_name_keys = _roster_identity_name_keys(right)
    return bool(left_name_keys & right_name_keys)


def _live_cached_roster_rows_same_identity(
    live_row: pd.Series | dict[str, object],
    cached_row: pd.Series | dict[str, object],
) -> bool:
    """Allow only a reviewed directional route transition across snapshots."""

    return bool(
        _roster_rows_same_identity(live_row, cached_row)
        or _reviewed_official_only_name_transition_match(live_row, cached_row)
        or _reviewed_live_cached_transition_match(live_row, cached_row)
    )


def _roster_row_matches_excluded_identity(
    row: pd.Series | dict[str, object],
    excluded_row: pd.Series | dict[str, object],
) -> bool:
    """Keep reviewed negative certificates URL-bound and fail closed."""

    if _optional_text(excluded_row.get("action")) == "excluded_non_mma_identity":
        return reviewed_non_mma_identity_match(row, excluded_row)
    return _live_cached_roster_rows_same_identity(excluded_row, row)


def _cached_carry_forward_identity_match(
    live_row: pd.Series | dict[str, object],
    cached_row: pd.Series | dict[str, object],
) -> bool:
    """Require every stable carrier before copying any cached profile value."""

    if _reviewed_official_only_name_transition_match(live_row, cached_row):
        return True
    if _reviewed_live_cached_transition_match(live_row, cached_row):
        return True

    live_stable_id = _supported_ufcstats_profile_id(live_row.get("ufcstats_url"))
    cached_stable_id = _supported_ufcstats_profile_id(
        cached_row.get("ufcstats_url")
    )
    live_url = _trusted_ufc_profile_url(live_row.get("official_athlete_url"))
    cached_url = _trusted_ufc_profile_url(
        cached_row.get("official_athlete_url")
    )
    live_name = _clean_text(live_row.get("official_name"))
    cached_name = _clean_text(cached_row.get("official_name"))
    return bool(
        live_stable_id
        and live_stable_id == cached_stable_id
        and live_url
        and _normalize_roster_url(live_url) == _normalize_roster_url(cached_url)
        and live_name
        and cached_name
        and _same_identity_name(live_name, cached_name)
    )


def _reviewed_official_only_name_transition_match(
    live_row: pd.Series | dict[str, object],
    cached_row: pd.Series | dict[str, object],
) -> bool:
    """Authorize one exact cached-to-live name transition without inventing an ID."""

    live_binding = _strict_reviewed_profile_identity_binding(live_row)
    cached_binding = _strict_reviewed_profile_identity_binding(cached_row)
    if live_binding is None or cached_binding is None:
        return False
    entity = _optional_text(live_binding.get("official_source_entity_id"))
    transition = _REVIEWED_OFFICIAL_ONLY_NAME_TRANSITIONS.get(entity)
    if transition is None:
        return False
    expected_url = _trusted_ufc_profile_url(
        transition["official_athlete_url"]
    )
    live_url = _trusted_ufc_profile_url(live_row.get("official_athlete_url"))
    cached_url = _trusted_ufc_profile_url(cached_row.get("official_athlete_url"))
    return bool(
        entity == cached_binding.get("official_source_entity_id")
        and not _optional_text(live_binding.get("stable_ufcstats_id"))
        and not _optional_text(cached_binding.get("stable_ufcstats_id"))
        and not _optional_text(live_binding.get("ufcstats_url"))
        and not _optional_text(cached_binding.get("ufcstats_url"))
        and _clean_text(live_row.get("official_name"))
        == transition["live_official_name"]
        and _clean_text(cached_row.get("official_name"))
        == transition["cached_official_name"]
        and expected_url
        and live_url
        and cached_url
        and _normalize_roster_url(live_url) == _normalize_roster_url(expected_url)
        and _normalize_roster_url(cached_url)
        == _normalize_roster_url(expected_url)
        and _present_ufcstats_carrier_ids(live_row) == ()
        and _present_ufcstats_carrier_ids(cached_row) == ()
        and _official_url_identity_positively_trusted(live_row)
        and _official_url_identity_positively_trusted(cached_row)
    )


def _legacy_cached_profile_row_is_preservable(
    row: pd.Series | dict[str, object],
) -> bool:
    """Identify an accepted current-MMA row from before binding certificates.

    This is a preservation-only migration rule.  It never makes a legacy row
    a positive source for newly observed values; it only permits an exact,
    uniquely carrier-matched live row to retain the already accepted snapshot.
    """

    predates_binding_certificate = not any(
        _optional_text(row.get(column))
        for column in OFFICIAL_PROFILE_IDENTITY_PROVENANCE_COLUMNS
    )
    return bool(
        predates_binding_certificate
        and _status_key(row.get("profile_status"))
        in _PROFILE_CURRENT_STATUS_KEYS
        and _explicit_boolean(row.get("coverage_eligible")) is True
        and not _official_url_identity_explicitly_untrusted(row)
        and _row_is_positively_classified_mma(row)
    )


def _present_ufcstats_carrier_ids(
    row: pd.Series | dict[str, object],
) -> tuple[str, ...] | None:
    """Return exact present stable carriers, or None for any malformed carrier."""

    carriers: list[str] = []
    for field in _REVIEWED_UFCSTATS_CARRIER_FIELDS:
        raw_value = _optional_text(row.get(field))
        if not raw_value:
            continue
        stable_id = (
            _supported_ufcstats_profile_id(raw_value)
            if field == "ufcstats_url"
            else normalize_ufcstats_id(raw_value) or ""
        )
        if not stable_id:
            return None
        carriers.append(stable_id)
    return tuple(carriers)


def _fresh_official_transport_is_exact_route(
    row: pd.Series | dict[str, object],
    expected_url: str,
) -> bool:
    expected_key = _normalize_roster_url(expected_url)
    if not expected_key:
        return False
    for field in _OFFICIAL_PROFILE_FRESH_TRANSPORT_COLUMNS[:3]:
        route = _trusted_ufc_profile_url(row.get(field))
        if not route or _normalize_roster_url(route) != expected_key:
            return False
    redirect_chain = _captured_redirect_chain(
        row.get("official_profile_fresh_redirect_chain")
    )
    return bool(
        redirect_chain is not None
        and all(
            (trusted := _trusted_ufc_profile_url(route))
            and _normalize_roster_url(trusted) == expected_key
            for route in redirect_chain
        )
    )


def _exact_unbound_live_route_observation(
    row: pd.Series | dict[str, object],
    *,
    expected_name: str,
    expected_url: str,
    expected_stable_id: str | None,
) -> bool:
    """Recognize one exact discovery-only route without authorizing its identity."""

    official_url = _trusted_ufc_profile_url(row.get("official_athlete_url"))
    if (
        normalize_person_name(row.get("official_name"))
        != normalize_person_name(expected_name)
        or not official_url
        or _normalize_roster_url(official_url)
        != _normalize_roster_url(expected_url)
        or _optional_text(row.get(_PROFILE_FETCH_OUTCOME_COLUMN)).casefold()
        != _PROFILE_FETCH_SUCCEEDED
        or _optional_text(row.get("official_url_identity_status"))
        != "unresolved_identity"
        or _explicit_boolean(row.get("official_url_identity_valid")) is not False
        or _optional_text(row.get("official_url_identity_reason"))
        != (
            "name_slug_match_is_discovery_only_without_validated_"
            "source_entity_and_event_binding"
        )
        or _optional_text(row.get("profile_status")) != _PROFILE_STATUS_UNKNOWN
        or _explicit_boolean(row.get("coverage_eligible")) is not False
        or _optional_text(row.get("active_roster_status_reason"))
        != "profile_status_missing_or_unrecognized"
        or _truthy(row.get("active_roster_retained_from_previous"))
        or _official_profile_binding_positively_verified(row)
        or any(
            _optional_text(row.get(column))
            for column in _OFFICIAL_PROFILE_BINDING_PROVENANCE_COLUMNS
        )
        or not _fresh_official_transport_is_exact_route(row, expected_url)
    ):
        return False

    for field in _UNTRUSTED_OFFICIAL_PROFILE_FIELDS:
        if field == "profile_status":
            continue
        if _optional_text(row.get(field)):
            return False

    carrier_ids = _present_ufcstats_carrier_ids(row)
    if carrier_ids is None:
        return False
    if expected_stable_id is None:
        return bool(
            not carrier_ids
            and not _optional_text(row.get("ufcstats_name"))
            and not _optional_text(row.get("ufcstats_resolution"))
            and not _optional_text(row.get("combat_sport"))
            and not _optional_text(row.get("combat_sport_reason"))
            and not _optional_text(row.get("combat_sport_profile_url"))
        )
    return bool(
        carrier_ids
        and set(carrier_ids) == {expected_stable_id}
        and _supported_ufcstats_profile_id(row.get("ufcstats_url"))
        == expected_stable_id
        and normalize_person_name(row.get("ufcstats_name"))
        == normalize_person_name(expected_name)
        and _optional_text(row.get("ufcstats_resolution"))
        == "ufc_fighter_details"
        and _optional_text(row.get("combat_sport")) == "mma"
        and _optional_text(row.get("combat_sport_reason"))
        == "ufcstats_profile_match"
        and not _optional_text(row.get("combat_sport_profile_url"))
    )


def _exact_cached_companion_owner(
    row: pd.Series | dict[str, object],
    *,
    certificate: Mapping[str, object],
) -> bool:
    expected_name = _optional_text(certificate.get("official_name"))
    expected_url = _optional_text(
        certificate.get("companion_official_athlete_url")
    )
    expected_stable_id = _optional_text(
        certificate.get("companion_stable_ufcstats_id")
    )
    official_url = _trusted_ufc_profile_url(row.get("official_athlete_url"))
    canonical_url = _trusted_ufc_profile_url(row.get("canonical_athlete_url"))
    carrier_ids = _present_ufcstats_carrier_ids(row)
    try:
        missing_count = int(
            float(
                _optional_text(
                    row.get("active_roster_consecutive_missing_syncs")
                )
                or "0"
            )
        )
    except ValueError:
        return False
    return bool(
        normalize_person_name(row.get("official_name"))
        == normalize_person_name(expected_name)
        and official_url
        and _normalize_roster_url(official_url)
        == _normalize_roster_url(expected_url)
        and canonical_url
        and _normalize_roster_url(canonical_url)
        == _normalize_roster_url(expected_url)
        and carrier_ids
        and set(carrier_ids) == {expected_stable_id}
        and _supported_ufcstats_profile_id(row.get("ufcstats_url"))
        == expected_stable_id
        and normalize_person_name(row.get("ufcstats_name"))
        == normalize_person_name(expected_name)
        and normalize_person_name(row.get("profile_name"))
        == normalize_person_name(expected_name)
        and _optional_text(row.get("ufcstats_resolution"))
        == "ufc_fighter_details"
        and _optional_text(row.get(_PROFILE_FETCH_OUTCOME_COLUMN)).casefold()
        == _PROFILE_FETCH_SUCCEEDED
        and _explicit_boolean(row.get("official_url_identity_valid")) is True
        and _optional_text(row.get("official_url_identity_status")) == "valid"
        and not _optional_text(row.get("official_url_identity_reason"))
        and _optional_text(row.get("profile_status")) == "Active"
        and _optional_text(row.get("combat_sport")) == "mma"
        and _optional_text(row.get("combat_sport_reason"))
        == "ufcstats_profile_match"
        and not _optional_text(row.get("combat_sport_profile_url"))
        and not _optional_text(row.get("active_roster_status_reason"))
        and _legacy_cached_profile_row_is_preservable(row)
        and _truthy(row.get("active_roster_live_present"))
        and not _truthy(row.get("active_roster_retained_from_previous"))
        and not _optional_text(row.get("active_roster_retained_at_utc"))
        and not _optional_text(row.get("active_roster_first_missing_at_utc"))
        and missing_count == 0
        and not _optional_text(row.get("active_roster_missing_from_live_reason"))
    )


def _quarantine_reviewed_unbound_simultaneous_alternate_routes(
    live_df: pd.DataFrame,
    cached_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Remove only an exact reviewed unbound alternate live observation.

    This is a negative, non-destructive certificate. It never creates an alias,
    copies cached data, binds an identity, or authorizes removal of a cached row.
    Any changed carrier, route, status, binding, or cardinality leaves the live
    frame untouched so the ordinary identity/publication guards still fail closed.
    """

    filtered = live_df.copy()
    filtered.attrs = dict(live_df.attrs)
    audit_rows: list[dict[str, object]] = []
    for certificate in reviewed_unbound_simultaneous_route_quarantines():
        expected_name = _optional_text(certificate.get("official_name"))
        alternate_url = _optional_text(
            certificate.get("alternate_official_athlete_url")
        )
        companion_url = _optional_text(
            certificate.get("companion_official_athlete_url")
        )
        stable_id = _optional_text(
            certificate.get("companion_stable_ufcstats_id")
        )
        if (
            certificate.get("automatic_alias_allowed") is not False
            or certificate.get("automatic_binding_allowed") is not False
            or certificate.get("automatic_redirect_allowed") is not False
            or certificate.get("destructive_exclusion_allowed") is not False
            or not filtered.index.is_unique
            or not cached_df.index.is_unique
        ):
            continue

        expected_observation = {
            "official_name": expected_name,
            "official_athlete_url": alternate_url,
            "profile": dict(certificate.get("alternate_observed_profile") or {}),
        }
        captured_observations = live_df.attrs.get(
            _UNBOUND_SIMULTANEOUS_ROUTE_OBSERVATIONS_ATTR
        )
        if captured_observations != [expected_observation]:
            continue

        def in_certificate_scope(row: pd.Series) -> bool:
            row_name = normalize_person_name(row.get("official_name"))
            row_url = _trusted_ufc_profile_url(row.get("official_athlete_url"))
            row_url_key = _normalize_roster_url(row_url) if row_url else ""
            carrier_ids = _present_ufcstats_carrier_ids(row)
            name_or_route_matches = bool(
                row_name == normalize_person_name(expected_name)
                or row_url_key
                in {
                    _normalize_roster_url(alternate_url),
                    _normalize_roster_url(companion_url),
                }
            )
            return bool(
                name_or_route_matches
                or (
                    carrier_ids is not None
                    and stable_id in set(carrier_ids)
                )
            )

        live_scope = [
            (position, row)
            for position, (_, row) in enumerate(filtered.iterrows())
            if in_certificate_scope(row)
        ]
        cached_scope = [
            row for _, row in cached_df.iterrows() if in_certificate_scope(row)
        ]
        if len(live_scope) != 2 or len(cached_scope) != 1:
            continue
        cached_owner = cached_scope[0]
        if not _exact_cached_companion_owner(
            cached_owner,
            certificate=certificate,
        ):
            continue

        live_companions = [
            (position, row)
            for position, row in live_scope
            if _exact_unbound_live_route_observation(
                row,
                expected_name=expected_name,
                expected_url=companion_url,
                expected_stable_id=stable_id,
            )
        ]
        live_alternates = [
            (position, row)
            for position, row in live_scope
            if _exact_unbound_live_route_observation(
                row,
                expected_name=expected_name,
                expected_url=alternate_url,
                expected_stable_id=None,
            )
        ]
        if len(live_companions) != 1 or len(live_alternates) != 1:
            continue
        companion_position, companion_row = live_companions[0]
        alternate_position, alternate_row = live_alternates[0]
        if companion_position == alternate_position:
            continue

        audit_rows.append(
            _identity_audit_row(
                dict(alternate_row),
                action=(
                    "quarantined_unbound_simultaneous_alternate_official_route"
                ),
            )
        )
        keep_positions = [
            position
            for position in range(len(filtered))
            if position != alternate_position
        ]
        filtered = filtered.iloc[keep_positions].copy()
        filtered.attrs = dict(live_df.attrs)
        # Recheck that the exact companion survived byte/field-equivalent.
        surviving_companion = filtered.iloc[
            companion_position if companion_position < alternate_position else companion_position - 1
        ]
        if not surviving_companion.equals(companion_row):
            return live_df, []
    return filtered, audit_rows


def _cached_live_identity_conflicts(
    live_df: pd.DataFrame,
    cached_df: pd.DataFrame,
) -> list[dict[str, str]]:
    """Find shared strong carriers whose remaining identity evidence conflicts."""

    conflicts: list[dict[str, str]] = []
    for _, live_row in live_df.iterrows():
        live_url = _trusted_ufc_profile_url(
            live_row.get("official_athlete_url")
        )
        live_stable_id = _supported_ufcstats_profile_id(
            live_row.get("ufcstats_url")
        )
        for _, cached_row in cached_df.iterrows():
            cached_url = _trusted_ufc_profile_url(
                cached_row.get("official_athlete_url")
            )
            cached_stable_id = _supported_ufcstats_profile_id(
                cached_row.get("ufcstats_url")
            )
            same_url = bool(
                live_url
                and cached_url
                and _normalize_roster_url(live_url)
                == _normalize_roster_url(cached_url)
            )
            same_stable_id = bool(
                live_stable_id
                and live_stable_id == cached_stable_id
            )
            if not (same_url or same_stable_id):
                continue
            if _reviewed_official_only_name_transition_match(
                live_row,
                cached_row,
            ):
                continue
            if _reviewed_live_cached_transition_match(live_row, cached_row):
                continue
            live_name = _clean_text(live_row.get("official_name"))
            cached_name = _clean_text(cached_row.get("official_name"))
            names_conflict = bool(
                live_name
                and cached_name
                and not _same_identity_name(live_name, cached_name)
            )
            stable_ids_conflict = bool(
                live_stable_id
                and cached_stable_id
                and live_stable_id != cached_stable_id
            )
            urls_conflict = bool(
                live_url
                and cached_url
                and _normalize_roster_url(live_url)
                != _normalize_roster_url(cached_url)
            )
            if not (
                (same_url and (stable_ids_conflict or names_conflict))
                or (same_stable_id and (urls_conflict or names_conflict))
            ):
                continue
            conflicts.append(
                {
                    "live_name": live_name,
                    "cached_name": cached_name,
                    "live_ufcstats_id": live_stable_id,
                    "cached_ufcstats_id": cached_stable_id,
                    "live_official_url": live_url,
                    "cached_official_url": cached_url,
                    "shared_carrier": (
                        "official_athlete_url|stable_ufcstats_id"
                        if same_url and same_stable_id
                        else (
                            "official_athlete_url"
                            if same_url
                            else "stable_ufcstats_id"
                        )
                    ),
                }
            )
    return conflicts


def _restore_cached_current_status_after_profile_request_failures(
    live_df: pd.DataFrame,
    cached_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Restore narrowly approved cached profile data after access was unavailable.

    A successfully fetched UFC profile that simply has no status tag is not
    equivalent to a transport failure or an HTTP-200 challenge/non-profile page
    and must remain quarantined. Likewise, ambiguous cached identities and
    cached rows without positive current-MMA identity evidence are not sources
    for new values. A uniquely carrier-matched accepted row from before binding
    certificates may only preserve its already accepted snapshot, which forces
    direct sync to retain the old bytes. Only immutable/slow-changing biography
    values are eligible; records, age, division, and aliases remain live-only.
    """
    if (
        live_df.empty
        or cached_df.empty
        or _PROFILE_FETCH_OUTCOME_COLUMN not in live_df.columns
    ):
        return live_df, 0

    live = live_df.copy()
    restored_rows = 0
    legacy_preserved_rows = 0
    cached_rows = [row for _, row in cached_df.iterrows()]
    for index, live_row in live.iterrows():
        outcome = _optional_text(live_row.get(_PROFILE_FETCH_OUTCOME_COLUMN)).casefold()
        if outcome not in {
            _PROFILE_FETCH_REQUEST_FAILED,
            _PROFILE_FETCH_INVALID_RESPONSE,
        }:
            continue
        identity_status = _identity_status_key(
            live_row.get("official_url_identity_status")
        )
        if (
            identity_status in {"mismatch", "test profile"}
            or (
                _falsey(live_row.get("official_url_identity_valid"))
                and identity_status != "unresolved identity"
            )
            or _row_has_power_slap_evidence(live_row)
        ):
            continue
        live_status_key = _status_key(live_row.get("profile_status"))
        if (
            not _profile_status_is_unknown(live_row)
            and live_status_key not in _PROFILE_CURRENT_STATUS_KEYS
        ):
            continue

        matches = [
            cached_row
            for cached_row in cached_rows
            if _cached_carry_forward_identity_match(live_row, cached_row)
        ]
        if len(matches) != 1:
            continue

        cached_row = matches[0]
        cached_is_positively_verified = bool(
            _row_is_positively_verified_current_mma(cached_row)
            and _explicit_boolean(cached_row.get("coverage_eligible")) is True
        )
        cached_is_legacy_preservable = (
            _legacy_cached_profile_row_is_preservable(cached_row)
        )
        if not (
            cached_is_positively_verified or cached_is_legacy_preservable
        ):
            continue

        restored_any = False
        if _profile_status_is_unknown(live_row):
            live.at[index, "profile_status"] = _optional_text(
                cached_row.get("profile_status")
            )
            live.at[index, "coverage_eligible"] = True
            if "active_roster_status_reason" not in live.columns:
                live["active_roster_status_reason"] = ""
            live.at[index, "active_roster_status_reason"] = (
                "cached_verified_current_after_profile_request_failure"
            )
            restored_rows += 1
            restored_any = True

        restored_fields: list[str] = []
        for field in _CACHED_STABLE_PROFILE_BIOGRAPHY_FIELDS:
            if _optional_text(live_row.get(field)):
                continue
            cached_value = _optional_text(cached_row.get(field))
            if not cached_value:
                continue
            live.at[index, field] = cached_value
            restored_fields.append(field)
        if restored_fields:
            live.at[index, _CACHED_PROFILE_BIOGRAPHY_FIELDS_COLUMN] = "|".join(
                restored_fields
            )
            live.at[index, _CACHED_PROFILE_BIOGRAPHY_REASON_COLUMN] = (
                _CACHED_PROFILE_BIOGRAPHY_REASON
            )
            restored_any = True
        if cached_is_legacy_preservable and restored_any:
            legacy_preserved_rows += 1

    live.attrs["legacy_accepted_profile_request_failure_preserved_rows"] = (
        legacy_preserved_rows
    )
    return live, restored_rows


def _preserve_accepted_immutable_profile_conflicts(
    live_df: pd.DataFrame,
    cached_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Keep accepted immutable facts when a successful live page disagrees.

    UFC.com can serve different complete athlete-page variants for the same URL.
    An already accepted immutable biography value is therefore retained unless
    the live row represents a unique, trusted identity whose cached value was
    blank. Conflicts remain visible in the returned diagnostics instead of
    silently rotating the roster/audit generation or guessing which source
    variant is authoritative.
    """
    if live_df.empty or cached_df.empty:
        return live_df, []

    live = live_df.copy()
    for column in ACTIVE_ROSTER_CACHED_BIOGRAPHY_PROVENANCE_COLUMNS:
        if column not in live.columns:
            live[column] = ""
    live_rows = [(index, row) for index, row in live.iterrows()]
    cached_rows = [row for _, row in cached_df.iterrows()]
    conflicts: list[dict[str, str]] = []

    def strong_match_basis(
        left: pd.Series | dict[str, object],
        right: pd.Series | dict[str, object],
    ) -> str:
        if not _cached_carry_forward_identity_match(left, right):
            return ""
        return "official_athlete_url|ufcstats_url|official_name"

    for index, live_row in live_rows:
        if (
            _optional_text(live_row.get(_PROFILE_FETCH_OUTCOME_COLUMN)).casefold()
            != _PROFILE_FETCH_SUCCEEDED
            or not _row_is_positively_verified_current_mma(live_row)
            or _explicit_boolean(live_row.get("coverage_eligible")) is not True
        ):
            continue

        matches = [
            (cached_row, match_basis)
            for cached_row in cached_rows
            if (match_basis := strong_match_basis(live_row, cached_row))
        ]
        if len(matches) != 1:
            continue
        cached_row, match_basis = matches[0]
        cached_is_positively_verified = bool(
            _row_is_positively_verified_current_mma(cached_row)
            and _explicit_boolean(cached_row.get("coverage_eligible")) is True
        )
        cached_is_legacy_preservable = (
            _legacy_cached_profile_row_is_preservable(cached_row)
        )
        if not (
            cached_is_positively_verified or cached_is_legacy_preservable
        ):
            continue
        reverse_matches = [
            other_live_row
            for _, other_live_row in live_rows
            if strong_match_basis(other_live_row, cached_row)
        ]
        if len(reverse_matches) != 1:
            continue

        for field in _ACCEPTED_IMMUTABLE_PROFILE_FIELDS:
            accepted_value = _optional_text(cached_row.get(field))
            observed_value = _optional_text(live_row.get(field))
            if not accepted_value or accepted_value == observed_value:
                continue
            live.at[index, field] = accepted_value
            cached_fields = {
                value.strip()
                for value in _optional_text(
                    live.at[index, _CACHED_PROFILE_BIOGRAPHY_FIELDS_COLUMN]
                ).split("|")
                if value.strip()
            }
            cached_fields.add(field)
            live.at[index, _CACHED_PROFILE_BIOGRAPHY_FIELDS_COLUMN] = "|".join(
                candidate
                for candidate in _ACCEPTED_IMMUTABLE_PROFILE_FIELDS
                if candidate in cached_fields
            )
            live.at[index, _CACHED_PROFILE_BIOGRAPHY_REASON_COLUMN] = (
                _CACHED_PROFILE_BIOGRAPHY_SUCCESS_CONFLICT_REASON
            )
            conflict = {
                "official_name": _optional_text(live_row.get("official_name")),
                "official_athlete_url": _optional_text(
                    live_row.get("official_athlete_url")
                ),
                "ufcstats_url": _optional_text(live_row.get("ufcstats_url")),
                "field": field,
                "accepted_value": accepted_value,
                "observed_live_value": observed_value,
                "reason": (
                    "blank_successful_observation"
                    if not observed_value
                    else "conflicting_nonblank_observation"
                ),
                "match_basis": match_basis,
                "fetch_outcome": _PROFILE_FETCH_SUCCEEDED,
                "identity_status": _optional_text(
                    live_row.get("official_url_identity_status")
                ),
                "action": "preserved_cached_accepted_value",
            }
            if cached_is_legacy_preservable:
                conflict["accepted_identity_provenance"] = (
                    "legacy_pre_binding_certificate"
                )
            conflicts.append(conflict)

    return live, conflicts


def _successful_profile_status_loss_counts(
    live_df: pd.DataFrame,
    cached_df: pd.DataFrame,
) -> tuple[int, int]:
    """Count comparable successful profiles and live status losses.

    This is deliberately an aggregate detector, not a per-row restoration path.
    One valid UFC profile with a missing/unrecognized status remains quarantined.
    A broad parser/markup failure, however, must not be allowed to publish an
    almost entirely ineligible roster generation. On a cacheless first sync,
    trusted successful live profiles form the comparison cohort themselves.
    """
    if live_df.empty or _PROFILE_FETCH_OUTCOME_COLUMN not in live_df.columns:
        return 0, 0

    comparable_live_rows = [
        live_row
        for _, live_row in live_df.iterrows()
        if (
            _optional_text(live_row.get(_PROFILE_FETCH_OUTCOME_COLUMN)).casefold()
            == _PROFILE_FETCH_SUCCEEDED
            and _official_url_identity_positively_trusted(live_row)
            and not _row_has_power_slap_evidence(live_row)
        )
    ]
    if cached_df.empty:
        return len(comparable_live_rows), sum(
            1
            for live_row in comparable_live_rows
            if _profile_status_is_unknown(live_row)
        )

    cached_candidates = [
        cached_row
        for _, cached_row in cached_df.iterrows()
        if (
            _row_is_positively_verified_current_mma(cached_row)
            and _explicit_boolean(cached_row.get("coverage_eligible")) is True
        )
    ]
    if not cached_candidates:
        return 0, 0

    matched_pairs: list[tuple[pd.Series, int]] = []
    cached_match_counts: dict[int, int] = {}
    for live_row in comparable_live_rows:
        matches = [
            cached_position
            for cached_position, cached_row in enumerate(cached_candidates)
            if _live_cached_roster_rows_same_identity(live_row, cached_row)
        ]
        if len(matches) != 1:
            continue
        cached_position = matches[0]
        matched_pairs.append((live_row, cached_position))
        cached_match_counts[cached_position] = (
            cached_match_counts.get(cached_position, 0) + 1
        )

    unique_pairs = [
        (live_row, cached_position)
        for live_row, cached_position in matched_pairs
        if cached_match_counts[cached_position] == 1
    ]
    comparison_rows = len(unique_pairs)
    lost_status_rows = sum(
        1 for live_row, _ in unique_pairs if _profile_status_is_unknown(live_row)
    )
    return comparison_rows, lost_status_rows


def _profile_status_loss_exceeds_drift_guard(
    comparison_rows: int,
    lost_status_rows: int,
) -> bool:
    if comparison_rows < _PROFILE_STATUS_DRIFT_MIN_COMPARISON_ROWS:
        return False
    if lost_status_rows < _PROFILE_STATUS_DRIFT_MIN_LOST_ROWS:
        return False
    return (
        lost_status_rows / comparison_rows
        >= _PROFILE_STATUS_DRIFT_LOST_RATIO
    )


def _destructive_cached_exclusions_exceed_guard(
    cached_rows: int,
    excluded_rows: int,
) -> bool:
    """Reject an anomalously large destructive audit cleanup independently."""
    if (
        cached_rows <= 0
        or excluded_rows < _ROSTER_DESTRUCTIVE_EXCLUSION_ABSOLUTE_ROWS
    ):
        return False
    return (
        excluded_rows / cached_rows
        >= _ROSTER_DESTRUCTIVE_EXCLUSION_RATIO
    )


def _apply_canonical_official_athlete_url(row: dict[str, object]) -> None:
    canonical_url = _clean_text(row.get("canonical_athlete_url"))
    if not canonical_url or canonical_url.casefold() in {"nan", "none"}:
        return
    if not _trusted_ufc_profile_url(canonical_url):
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
    excluded_rows: list[dict[str, object]] | None = None,
    sync_complete: bool = False,
    intentionally_removed_rows: list[dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    """Retain previously tracked roster rows that disappear from the latest UFC.com scrape."""
    if live_df.empty or cached_df.empty:
        return live_df, [], []
    live_df = _restore_cached_official_urls_for_live_aliases(live_df, cached_df)
    excluded_rows = excluded_rows or []

    live_rows = [row for _, row in live_df.iterrows()]
    retained_rows: list[pd.Series] = []
    retained_details: list[dict[str, object]] = []
    expired_details: list[dict[str, object]] = []
    for _, cached_row in cached_df.iterrows():
        key = _roster_merge_key(cached_row)
        if not key or not _roster_identity_keys(cached_row):
            continue
        if any(
            _live_cached_roster_rows_same_identity(live_row, cached_row)
            for live_row in live_rows
        ):
            continue
        matched_exclusion = next(
            (
                excluded_row
                for excluded_row in excluded_rows
                if _roster_row_matches_excluded_identity(cached_row, excluded_row)
            ),
            None,
        )
        if matched_exclusion is not None:
            if intentionally_removed_rows is not None:
                intentionally_removed_rows.append(
                    {
                        "official_name": _optional_text(
                            cached_row.get("official_name")
                        ),
                        "official_athlete_url": _optional_text(
                            cached_row.get("official_athlete_url")
                        ),
                        "ufcstats_url": _optional_text(
                            cached_row.get("ufcstats_url")
                        ),
                        "reason": (
                            _optional_text(matched_exclusion.get("action"))
                            or "accepted_identity_audit_exclusion"
                        ),
                    }
                )
            continue
        if _verified_inactive_profile_status(cached_row):
            if intentionally_removed_rows is not None:
                intentionally_removed_rows.append(
                    {
                        "official_name": _optional_text(
                            cached_row.get("official_name")
                        ),
                        "official_athlete_url": _optional_text(
                            cached_row.get("official_athlete_url")
                        ),
                        "ufcstats_url": _optional_text(
                            cached_row.get("ufcstats_url")
                        ),
                        "reason": "cached_verified_inactive_profile_status",
                    }
                )
            continue

        was_retained = _truthy(cached_row.get("active_roster_retained_from_previous"))
        try:
            previous_missing_syncs = int(float(cached_row.get("active_roster_consecutive_missing_syncs") or 0))
        except (TypeError, ValueError):
            previous_missing_syncs = 0
        first_missing_at_utc = (
            _optional_text(cached_row.get("active_roster_first_missing_at_utc"))
            or _optional_text(cached_row.get("active_roster_retained_at_utc"))
            or (retained_at_utc if sync_complete else "")
        )
        retained_counter_at_utc = _optional_text(
            cached_row.get("active_roster_retained_at_utc")
        )
        if sync_complete and (not was_retained or previous_missing_syncs <= 0):
            consecutive_missing_syncs = 1
            retained_counter_at_utc = retained_at_utc
        elif sync_complete:
            prior_counter_at_utc = retained_counter_at_utc or first_missing_at_utc
            try:
                current_observation = datetime.fromisoformat(
                    retained_at_utc.replace("Z", "+00:00")
                )
                prior_observation = datetime.fromisoformat(
                    prior_counter_at_utc.replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise ValueError(
                    "Retained UFC roster lifecycle timestamp is not valid ISO-8601"
                ) from exc
            if current_observation.tzinfo is None or prior_observation.tzinfo is None:
                raise ValueError(
                    "Retained UFC roster lifecycle timestamp must include a timezone"
                )
            current_observation = current_observation.astimezone(timezone.utc)
            prior_observation = prior_observation.astimezone(timezone.utc)
            if prior_observation > current_observation:
                raise ValueError(
                    "Retained UFC roster lifecycle timestamp is later than the current observation"
                )
            if iso_week_identity(current_observation) == iso_week_identity(
                prior_observation
            ):
                consecutive_missing_syncs = previous_missing_syncs
                retained_counter_at_utc = prior_counter_at_utc
            else:
                consecutive_missing_syncs = previous_missing_syncs + 1
                retained_counter_at_utc = retained_at_utc
        else:
            consecutive_missing_syncs = previous_missing_syncs
        detail: dict[str, object] = {
            "official_name": _optional_text(cached_row.get("official_name")),
            "official_athlete_url": _optional_text(cached_row.get("official_athlete_url")),
            "ufcstats_url": _optional_text(cached_row.get("ufcstats_url")),
            "first_missing_at_utc": first_missing_at_utc,
            "consecutive_missing_syncs": consecutive_missing_syncs,
        }
        if (
            sync_complete
            and consecutive_missing_syncs
            >= _RETAINED_ROSTER_MAX_CONSECUTIVE_MISSING_SYNCS
        ):
            expired_details.append(detail)
            continue

        retained_row = cached_row.copy()
        retained_row["active_roster_first_missing_at_utc"] = first_missing_at_utc
        retained_row["active_roster_retained_at_utc"] = retained_counter_at_utc
        retained_row["active_roster_consecutive_missing_syncs"] = consecutive_missing_syncs
        retained_rows.append(retained_row)
        retained_details.append(detail)

    if not retained_rows:
        return live_df, [], expired_details

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
    live["active_roster_first_missing_at_utc"] = ""
    live["active_roster_consecutive_missing_syncs"] = 0
    live["active_roster_missing_from_live_reason"] = ""

    retained["active_roster_live_present"] = False
    retained["active_roster_retained_from_previous"] = True
    retained["active_roster_missing_from_live_reason"] = (
        "absent_from_latest_ufc_active_roster_sync"
        if sync_complete
        else "absent_from_incomplete_ufc_active_roster_sync"
    )
    if sync_complete:
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
    return merged, retained_details, expired_details


def _reconcile_verified_inactive_transitions_on_cached_fallback(
    cached_df: pd.DataFrame,
    observations: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    """Filter only one-to-one, freshly verified inactive transition sources.

    This changes the returned fallback view only.  The cached artifact remains
    byte-for-byte untouched, and any absent or ambiguous transition evidence
    fails closed without removing a row.
    """

    cached_rows = [row for _, row in cached_df.iterrows()]
    candidates: list[tuple[int, int]] = []
    diagnostics: list[dict[str, object]] = []
    for observation_position, observation in enumerate(observations):
        observation_stable_id = _supported_ufcstats_profile_id(
            observation.get("ufcstats_url")
        )
        if (
            observation_stable_id
            not in _INACTIVE_TRANSITION_RECONCILIATION_STABLE_IDS
        ):
            diagnostics.append(
                {
                    "observation_position": observation_position,
                    "official_name": _optional_text(
                        observation.get("official_name")
                    ),
                    "official_athlete_url": _optional_text(
                        observation.get("official_athlete_url")
                    ),
                    "stable_ufcstats_id": observation_stable_id,
                    "reason": "inactive_transition_fallback_not_authorized",
                }
            )
            continue
        if (
            _optional_text(observation.get(_PROFILE_FETCH_OUTCOME_COLUMN)).casefold()
            != _PROFILE_FETCH_SUCCEEDED
            or not _verified_inactive_profile_status(observation)
        ):
            diagnostics.append(
                {
                    "observation_position": observation_position,
                    "reason": "inactive_observation_failed_positive_reverification",
                }
            )
            continue
        matched_positions = [
            cached_position
            for cached_position, cached_row in enumerate(cached_rows)
            if reviewed_official_identity_transition_match(observation, cached_row)
        ]
        if len(matched_positions) != 1:
            diagnostics.append(
                {
                    "observation_position": observation_position,
                    "official_name": _optional_text(observation.get("official_name")),
                    "official_athlete_url": _optional_text(
                        observation.get("official_athlete_url")
                    ),
                    "reason": (
                        "inactive_transition_source_not_found"
                        if not matched_positions
                        else "inactive_transition_source_ambiguous"
                    ),
                    "cached_match_count": len(matched_positions),
                }
            )
            continue
        candidates.append((observation_position, matched_positions[0]))

    cached_match_counts: dict[int, int] = {}
    for _, cached_position in candidates:
        cached_match_counts[cached_position] = (
            cached_match_counts.get(cached_position, 0) + 1
        )

    removal_positions: set[int] = set()
    removed_details: list[dict[str, object]] = []
    for observation_position, cached_position in candidates:
        if cached_match_counts[cached_position] != 1:
            diagnostics.append(
                {
                    "observation_position": observation_position,
                    "reason": "inactive_transition_target_ambiguous",
                    "cached_position": cached_position,
                }
            )
            continue
        cached_row = cached_rows[cached_position]
        removal_positions.add(cached_position)
        removed_details.append(
            {
                "official_name": _optional_text(cached_row.get("official_name")),
                "official_athlete_url": _optional_text(
                    cached_row.get("official_athlete_url")
                ),
                "ufcstats_url": _optional_text(cached_row.get("ufcstats_url")),
                "reason": "excluded_inactive_profile_status",
            }
        )

    if not removal_positions:
        return cached_df, [], diagnostics
    keep_mask = [
        position not in removal_positions for position in range(len(cached_df))
    ]
    filtered = cached_df.iloc[keep_mask].reset_index(drop=True)
    return filtered, removed_details, diagnostics


def _reviewed_transition_for_stable_id(
    stable_id: str,
) -> dict[str, object] | None:
    matches = [
        transition
        for transition in reviewed_official_identity_transitions()
        if _optional_text(transition.get("stable_ufcstats_id")) == stable_id
    ]
    if len(matches) > 1:
        raise ValueError(
            "Reviewed UFC route-transition registry has duplicate stable owners: "
            f"{stable_id}"
        )
    return dict(matches[0]) if matches else None


def _row_is_in_reviewed_transition_scope(
    row: Mapping[str, object] | pd.Series,
    transition: Mapping[str, object],
) -> bool:
    expected_stable_id = _optional_text(transition.get("stable_ufcstats_id"))
    endpoint_entities = {
        _optional_text(transition.get("source_official_source_entity_id")),
        _optional_text(transition.get("target_official_source_entity_id")),
    }
    endpoint_names = {
        normalize_person_name(transition.get("source_official_name")),
        normalize_person_name(transition.get("target_official_name")),
    }
    carrier_ids = _present_ufcstats_carrier_ids(row)
    return bool(
        source_entity_id("ufc", row.get("official_athlete_url"))
        in endpoint_entities
        or normalize_person_name(row.get("official_name")) in endpoint_names
        or (
            carrier_ids is not None
            and expected_stable_id in set(carrier_ids)
        )
    )


def _exact_reviewed_active_transition_endpoint(
    row: Mapping[str, object] | pd.Series,
    *,
    transition: Mapping[str, object],
    endpoint: str,
) -> bool:
    if endpoint not in {"source", "target"}:
        return False
    expected_name = _optional_text(transition.get(f"{endpoint}_official_name"))
    expected_url = _optional_text(
        transition.get(f"{endpoint}_official_athlete_url")
    )
    expected_canonical = _optional_text(
        transition.get(f"{endpoint}_canonical_athlete_url")
    )
    expected_stable_id = _optional_text(transition.get("stable_ufcstats_id"))
    expected_slug_text = _slug_to_name(expected_url)
    expected_slug_name = normalize_person_name(expected_slug_text)
    expected_name_key = normalize_person_name(expected_name)
    expected_identity_status = (
        "valid"
        if expected_slug_name == expected_name_key
        else "slug mismatch profile valid"
    )
    expected_identity_reason = (
        ""
        if expected_identity_status == "valid"
        else f"slug_name_mismatch:{expected_slug_text}"
    )
    if expected_stable_id in _STRICT_FRESH_CACHED_ROUTE_TRANSITION_STABLE_IDS:
        allowed_resolutions = (
            {"reviewed_roster_recovery_binding"}
            if endpoint == "target"
            else {
                "ufc_fighter_details",
                "reviewed_roster_recovery_binding",
            }
        )
        if _optional_text(row.get("ufcstats_resolution")) not in allowed_resolutions:
            return False
    carrier_ids = _present_ufcstats_carrier_ids(row)
    official_url = _trusted_ufc_profile_url(row.get("official_athlete_url"))
    canonical_url = _trusted_ufc_profile_url(row.get("canonical_athlete_url"))
    return bool(
        expected_stable_id
        and normalize_person_name(row.get("official_name"))
        == normalize_person_name(expected_name)
        and normalize_person_name(row.get("profile_name"))
        == normalize_person_name(expected_name)
        and normalize_person_name(row.get("ufcstats_name"))
        == normalize_person_name(transition.get("ufcstats_name"))
        and official_url
        and _normalize_roster_url(official_url)
        == _normalize_roster_url(expected_url)
        and canonical_url
        and _normalize_roster_url(canonical_url)
        == _normalize_roster_url(expected_canonical)
        and (
            (
                normalize_person_name(row.get("slug_name"))
                == expected_slug_name
            )
            if expected_identity_status == "valid"
            else not _optional_text(row.get("slug_name"))
        )
        and _explicit_boolean(row.get("official_url_identity_valid")) is True
        and _identity_status_key(row.get("official_url_identity_status"))
        == expected_identity_status
        and _optional_text(row.get("official_url_identity_reason"))
        == expected_identity_reason
        and carrier_ids
        and set(carrier_ids) == {expected_stable_id}
        and _supported_ufcstats_profile_id(row.get("ufcstats_url"))
        == expected_stable_id
        and _optional_text(row.get(_PROFILE_FETCH_OUTCOME_COLUMN)).casefold()
        == _PROFILE_FETCH_SUCCEEDED
        and reviewed_mma_identity_binding_is_complete(row)
        and _row_is_positively_verified_current_mma(row)
        and _explicit_boolean(row.get("coverage_eligible")) is True
        and _fresh_official_transport_is_exact_route(row, expected_url)
    )


def _exact_reviewed_cached_transition_source_endpoint(
    row: Mapping[str, object] | pd.Series,
    *,
    transition: Mapping[str, object],
) -> bool:
    """Require the exact persisted old-route owner before a fresh transition.

    Accepted roster CSVs predate the fresh-transport columns and may omit the
    redundant ``stable_ufcstats_id``/``fighter_id`` carriers. The persisted
    UFCStats URL, profile fetch, names, route, canonical, status, and MMA
    evidence are nevertheless mandatory here. No value is copied from a name
    or slug match alone.
    """

    expected_name = _optional_text(transition.get("source_official_name"))
    expected_url = _optional_text(
        transition.get("source_official_athlete_url")
    )
    expected_canonical = _optional_text(
        transition.get("source_canonical_athlete_url")
    )
    expected_stable_id = _optional_text(transition.get("stable_ufcstats_id"))
    expected_slug = normalize_person_name(_slug_to_name(expected_url))
    carrier_ids = _present_ufcstats_carrier_ids(row)
    official_url = _trusted_ufc_profile_url(row.get("official_athlete_url"))
    canonical_url = _trusted_ufc_profile_url(row.get("canonical_athlete_url"))
    return bool(
        expected_stable_id
        and normalize_person_name(row.get("official_name"))
        == normalize_person_name(expected_name)
        and normalize_person_name(row.get("profile_name"))
        == normalize_person_name(expected_name)
        and normalize_person_name(row.get("ufcstats_name"))
        == normalize_person_name(transition.get("ufcstats_name"))
        and normalize_person_name(row.get("slug_name")) == expected_slug
        and official_url
        and _normalize_roster_url(official_url)
        == _normalize_roster_url(expected_url)
        and canonical_url
        and _normalize_roster_url(canonical_url)
        == _normalize_roster_url(expected_canonical)
        and carrier_ids
        and set(carrier_ids) == {expected_stable_id}
        and _supported_ufcstats_profile_id(row.get("ufcstats_url"))
        == expected_stable_id
        and _optional_text(row.get(_PROFILE_FETCH_OUTCOME_COLUMN)).casefold()
        == _PROFILE_FETCH_SUCCEEDED
        and _explicit_boolean(row.get("official_url_identity_valid")) is True
        and _identity_status_key(row.get("official_url_identity_status"))
        == "valid"
        and not _optional_text(row.get("official_url_identity_reason"))
        and _status_key(row.get("profile_status"))
        in _PROFILE_CURRENT_STATUS_KEYS
        and _optional_text(row.get("ufcstats_resolution"))
        in {"ufc_fighter_details", "reviewed_roster_recovery_binding"}
        and _status_key(row.get("combat_sport")) == "mma"
        and _optional_text(row.get("combat_sport_reason"))
        == "ufcstats_profile_match"
        and not _optional_text(row.get("combat_sport_profile_url"))
        and not _optional_text(row.get("active_roster_status_reason"))
        and not _row_has_power_slap_evidence(row)
    )


def _reviewed_live_cached_transition_match(
    live_row: Mapping[str, object] | pd.Series,
    cached_row: Mapping[str, object] | pd.Series,
) -> bool:
    """Keep Ravena's carrier exemption fresh-proof-bound and directional."""

    if not reviewed_official_identity_transition_match(live_row, cached_row):
        return False
    carrier_ids = _present_ufcstats_carrier_ids(live_row)
    if not carrier_ids or len(set(carrier_ids)) != 1:
        return False
    stable_id = carrier_ids[0]
    if stable_id not in _STRICT_FRESH_CACHED_ROUTE_TRANSITION_STABLE_IDS:
        return True
    return _strict_fresh_reviewed_live_cached_transition_match(
        live_row,
        cached_row,
        stable_id=stable_id,
    )


def _strict_fresh_reviewed_live_cached_transition_match(
    live_row: Mapping[str, object] | pd.Series,
    cached_row: Mapping[str, object] | pd.Series,
    *,
    stable_id: str = "",
) -> bool:
    """Authorize only Ravena's one-row old-to-live replacement proof."""

    carrier_ids = _present_ufcstats_carrier_ids(live_row)
    if not carrier_ids or len(set(carrier_ids)) != 1:
        return False
    observed_stable_id = carrier_ids[0]
    stable_id = stable_id or observed_stable_id
    if (
        stable_id != observed_stable_id
        or stable_id not in _STRICT_FRESH_CACHED_ROUTE_TRANSITION_STABLE_IDS
        or not reviewed_official_identity_transition_match(
            live_row,
            cached_row,
        )
    ):
        return False
    transition = _reviewed_transition_for_stable_id(stable_id)
    return bool(
        transition is not None
        and _exact_reviewed_active_transition_endpoint(
            live_row,
            transition=transition,
            endpoint="target",
        )
        and _status_key(live_row.get("combat_sport")) == "mma"
        and _optional_text(live_row.get("combat_sport_reason"))
        == "ufcstats_profile_match"
        and not _optional_text(live_row.get("combat_sport_profile_url"))
        and not _optional_text(live_row.get("active_roster_status_reason"))
        and _exact_reviewed_cached_transition_source_endpoint(
            cached_row,
            transition=transition,
        )
    )


def _exact_reviewed_inherited_duplicate_owner(
    row: Mapping[str, object] | pd.Series,
    *,
    transition: Mapping[str, object],
    canonical_endpoint: str,
) -> bool:
    """Recognize one of Nakamura's two already-accepted owner carriers."""

    if canonical_endpoint not in {"source", "target"}:
        return False
    expected_name = _optional_text(transition.get("source_official_name"))
    expected_source_url = _optional_text(
        transition.get("source_official_athlete_url")
    )
    expected_canonical = _optional_text(
        transition.get(f"{canonical_endpoint}_canonical_athlete_url")
    )
    expected_stable_id = _optional_text(transition.get("stable_ufcstats_id"))
    official_url = _trusted_ufc_profile_url(row.get("official_athlete_url"))
    canonical_url = _trusted_ufc_profile_url(row.get("canonical_athlete_url"))
    carrier_ids = _present_ufcstats_carrier_ids(row)
    expected_identity_status = (
        "valid"
        if canonical_endpoint == "source"
        else "slug mismatch profile valid"
    )
    identity_reason = _optional_text(row.get("official_url_identity_reason"))
    expected_reason = (
        ""
        if canonical_endpoint == "source"
        else "slug_name_mismatch:keiichiro nakamura 0"
    )
    try:
        missing_count = int(
            float(
                _optional_text(
                    row.get("active_roster_consecutive_missing_syncs")
                )
                or "0"
            )
        )
    except ValueError:
        return False
    return bool(
        normalize_person_name(row.get("official_name"))
        == normalize_person_name(expected_name)
        and normalize_person_name(row.get("profile_name"))
        == normalize_person_name(expected_name)
        and normalize_person_name(row.get("ufcstats_name"))
        == normalize_person_name(transition.get("ufcstats_name"))
        and normalize_person_name(row.get("slug_name"))
        == normalize_person_name(expected_name)
        and official_url
        and _normalize_roster_url(official_url)
        == _normalize_roster_url(expected_source_url)
        and canonical_url
        and _normalize_roster_url(canonical_url)
        == _normalize_roster_url(expected_canonical)
        and carrier_ids
        and set(carrier_ids) == {expected_stable_id}
        and _supported_ufcstats_profile_id(row.get("ufcstats_url"))
        == expected_stable_id
        and _optional_text(row.get(_PROFILE_FETCH_OUTCOME_COLUMN)).casefold()
        == _PROFILE_FETCH_SUCCEEDED
        and _explicit_boolean(row.get("official_url_identity_valid")) is True
        and _identity_status_key(row.get("official_url_identity_status"))
        == expected_identity_status
        and identity_reason == expected_reason
        and _status_key(row.get("profile_status"))
        in _PROFILE_CURRENT_STATUS_KEYS
        and _optional_text(row.get("ufcstats_resolution"))
        == "ufc_fighter_details"
        and _status_key(row.get("combat_sport")) == "mma"
        and _optional_text(row.get("combat_sport_reason"))
        == "ufcstats_profile_match"
        and not _optional_text(row.get("combat_sport_profile_url"))
        and _explicit_boolean(row.get("coverage_eligible")) is True
        and not _optional_text(row.get("active_roster_status_reason"))
        and _legacy_cached_profile_row_is_preservable(row)
        and _truthy(row.get("active_roster_live_present"))
        and not _truthy(row.get("active_roster_retained_from_previous"))
        and not _optional_text(row.get("active_roster_retained_at_utc"))
        and not _optional_text(row.get("active_roster_first_missing_at_utc"))
        and missing_count == 0
        and not _optional_text(row.get("active_roster_missing_from_live_reason"))
    )


def _reconcile_reviewed_simultaneous_active_owner_transition_rows(
    rows: Sequence[Mapping[str, object]],
    observed_transition_endpoints: Mapping[str, set[str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Collapse only explicitly authorized, freshly proved active route pairs."""

    reconciled_rows = [dict(row) for row in rows]
    reconciliations: list[dict[str, object]] = []
    simultaneous_ids = sorted(
        stable_id
        for stable_id, endpoints in observed_transition_endpoints.items()
        if stable_id in _SIMULTANEOUS_ACTIVE_OWNER_TRANSITION_STABLE_IDS
        and {"source", "target"}.issubset(endpoints)
    )
    for stable_id in simultaneous_ids:
        transition = _reviewed_transition_for_stable_id(stable_id)
        if transition is None:
            raise _simultaneous_reviewed_transition_error(
                stable_id,
                detail="reviewed transition certificate disappeared during scan",
            )
        scope = [
            (position, row)
            for position, row in enumerate(reconciled_rows)
            if _row_is_in_reviewed_transition_scope(row, transition)
        ]
        source_matches = [
            (position, row)
            for position, row in scope
            if _exact_reviewed_active_transition_endpoint(
                row,
                transition=transition,
                endpoint="source",
            )
        ]
        target_matches = [
            (position, row)
            for position, row in scope
            if _exact_reviewed_active_transition_endpoint(
                row,
                transition=transition,
                endpoint="target",
            )
        ]
        if (
            len(scope) != 2
            or len(source_matches) != 1
            or len(target_matches) != 1
            or source_matches[0][0] == target_matches[0][0]
            or not reviewed_official_identity_transition_match(
                target_matches[0][1],
                source_matches[0][1],
            )
        ):
            raise _simultaneous_reviewed_transition_error(
                stable_id,
                detail=(
                    "exact active owner proof failed "
                    f"(scope={len(scope)}, sources={len(source_matches)}, "
                    f"targets={len(target_matches)})"
                ),
            )
        source_position, source_row = source_matches[0]
        _, target_row = target_matches[0]
        reconciled_rows.pop(source_position)
        reconciliations.append(
            {
                "stable_ufcstats_id": stable_id,
                "source_official_name": _optional_text(
                    source_row.get("official_name")
                ),
                "source_official_athlete_url": _optional_text(
                    source_row.get("official_athlete_url")
                ),
                "target_official_name": _optional_text(
                    target_row.get("official_name")
                ),
                "target_official_athlete_url": _optional_text(
                    target_row.get("official_athlete_url")
                ),
                "target_profile_status": _optional_text(
                    target_row.get("profile_status")
                ),
                "reason": "reviewed_active_duplicate_owner_target_preferred",
            }
        )
    return reconciled_rows, reconciliations


def _reconcile_reviewed_inherited_duplicate_owner_cache(
    live_df: pd.DataFrame,
    cached_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Remove only Nakamura's exact two-row inherited cache shape from working state.

    The accepted artifact is never changed here.  Both inherited rows are kept
    in ``accepted_cached_df`` for the later canonical guard, while the working
    cache omits them so neither can overwrite or resurrect the freshly bound
    reviewed target.  Any changed route, owner, carrier, proof, or cardinality
    leaves the cache untouched and lets the ordinary conflict guards reject it.
    """

    if (
        live_df.empty
        or cached_df.empty
        or not live_df.index.is_unique
        or not cached_df.index.is_unique
    ):
        return cached_df, []
    removal_positions: set[int] = set()
    reconciliations: list[dict[str, object]] = []
    live_rows = [row for _, row in live_df.iterrows()]
    cached_rows = [row for _, row in cached_df.iterrows()]
    for stable_id in sorted(_ACTIVE_DUPLICATE_OWNER_TRANSITION_STABLE_IDS):
        transition = _reviewed_transition_for_stable_id(stable_id)
        if transition is None:
            continue
        live_scope = [
            row
            for row in live_rows
            if _row_is_in_reviewed_transition_scope(row, transition)
        ]
        if len(live_scope) != 1 or not _exact_reviewed_active_transition_endpoint(
            live_scope[0],
            transition=transition,
            endpoint="target",
        ):
            continue
        cached_scope = [
            (position, row)
            for position, row in enumerate(cached_rows)
            if _row_is_in_reviewed_transition_scope(row, transition)
        ]
        source_owners = [
            (position, row)
            for position, row in cached_scope
            if _exact_reviewed_inherited_duplicate_owner(
                row,
                transition=transition,
                canonical_endpoint="source",
            )
        ]
        target_canonical_owners = [
            (position, row)
            for position, row in cached_scope
            if _exact_reviewed_inherited_duplicate_owner(
                row,
                transition=transition,
                canonical_endpoint="target",
            )
        ]
        if (
            len(cached_scope) != 2
            or len(source_owners) != 1
            or len(target_canonical_owners) != 1
            or source_owners[0][0] == target_canonical_owners[0][0]
            or not reviewed_official_identity_transition_match(
                live_scope[0],
                source_owners[0][1],
            )
        ):
            continue
        source_position = source_owners[0][0]
        duplicate_position = target_canonical_owners[0][0]
        removal_positions.update({source_position, duplicate_position})
        reconciliations.append(
            {
                "stable_ufcstats_id": stable_id,
                "source_official_name": transition["source_official_name"],
                "source_official_athlete_url": transition[
                    "source_official_athlete_url"
                ],
                "target_official_name": transition["target_official_name"],
                "target_official_athlete_url": transition[
                    "target_official_athlete_url"
                ],
                "accepted_owner_count": 2,
                "accepted_source_position": source_position,
                "removed_duplicate_accepted_position": duplicate_position,
                "reason": "reviewed_inherited_duplicate_owner_collapsed_to_target",
            }
        )
    if not removal_positions:
        return cached_df, []
    keep_mask = [
        position not in removal_positions for position in range(len(cached_df))
    ]
    return cached_df.iloc[keep_mask].reset_index(drop=True), reconciliations


def _reviewed_duplicate_owner_candidate_removal_authorized(
    accepted_rows: Sequence[Mapping[str, object] | pd.Series],
    candidate_rows: Sequence[Mapping[str, object] | pd.Series],
    *,
    accepted_position: int,
    reconciliations: Sequence[Mapping[str, object]],
) -> bool:
    matching_reconciliations = [
        reconciliation
        for reconciliation in reconciliations
        if _optional_text(reconciliation.get("stable_ufcstats_id"))
        in _ACTIVE_DUPLICATE_OWNER_TRANSITION_STABLE_IDS
    ]
    if len(matching_reconciliations) != 1:
        return False
    reconciliation = matching_reconciliations[0]
    stable_id = _optional_text(reconciliation.get("stable_ufcstats_id"))
    transition = _reviewed_transition_for_stable_id(stable_id)
    if transition is None:
        return False
    try:
        source_position = int(reconciliation["accepted_source_position"])
        duplicate_position = int(
            reconciliation["removed_duplicate_accepted_position"]
        )
        accepted_owner_count = int(reconciliation["accepted_owner_count"])
    except (KeyError, TypeError, ValueError):
        return False
    if (
        accepted_position != duplicate_position
        or accepted_owner_count != 2
        or source_position == duplicate_position
        or min(source_position, duplicate_position) < 0
        or max(source_position, duplicate_position) >= len(accepted_rows)
        or _optional_text(reconciliation.get("reason"))
        != "reviewed_inherited_duplicate_owner_collapsed_to_target"
        or _normalize_roster_url(
            reconciliation.get("source_official_athlete_url")
        )
        != _normalize_roster_url(transition["source_official_athlete_url"])
        or _normalize_roster_url(
            reconciliation.get("target_official_athlete_url")
        )
        != _normalize_roster_url(transition["target_official_athlete_url"])
    ):
        return False
    accepted_scope_positions = [
        position
        for position, row in enumerate(accepted_rows)
        if _row_is_in_reviewed_transition_scope(row, transition)
    ]
    candidate_scope = [
        row
        for row in candidate_rows
        if _row_is_in_reviewed_transition_scope(row, transition)
    ]
    if (
        set(accepted_scope_positions) != {source_position, duplicate_position}
        or len(accepted_scope_positions) != 2
        or len(candidate_scope) != 1
    ):
        return False
    source_owner = accepted_rows[source_position]
    duplicate_owner = accepted_rows[duplicate_position]
    target = candidate_scope[0]
    return bool(
        _exact_reviewed_inherited_duplicate_owner(
            source_owner,
            transition=transition,
            canonical_endpoint="source",
        )
        and _exact_reviewed_inherited_duplicate_owner(
            duplicate_owner,
            transition=transition,
            canonical_endpoint="target",
        )
        and _exact_reviewed_active_transition_endpoint(
            target,
            transition=transition,
            endpoint="target",
        )
        and reviewed_official_identity_transition_match(target, source_owner)
        and _cached_carry_forward_identity_match(target, source_owner)
        and not _cached_carry_forward_identity_match(target, duplicate_owner)
    )


def _simultaneous_reviewed_transition_error(
    stable_id: str,
    *,
    detail: str = "",
) -> RuntimeError:
    message = (
        "Official UFC roster exposed both endpoints of a reviewed one-way "
        "identity transition in one scan; refusing to alias or deduplicate "
        f"stable UFCStats owner {stable_id}"
    )
    if detail:
        message = f"{message}; {detail}"
    return RuntimeError(message)


def _reconcile_reviewed_simultaneous_inactive_transition_rows(
    rows: Sequence[Mapping[str, object]],
    observed_transition_endpoints: Mapping[str, set[str]],
    inactive_observations: Sequence[Mapping[str, object]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Collapse only Anderson's exact live source/target pair after inactivity.

    Every other simultaneous reviewed transition remains an immediate error.
    Anderson is reconciled only after one freshly fetched, positively trusted
    target says ``Not Fighting`` and exactly one live source row matches the
    directional certificate.  Missing, active, contradictory, or ambiguous
    evidence fails closed without turning this into an alias/dedup rule.
    """

    reconciled_rows = [dict(row) for row in rows]
    reconciliations: list[dict[str, object]] = []
    transitions_by_stable_id = {
        str(transition["stable_ufcstats_id"]): transition
        for transition in reviewed_official_identity_transitions()
    }
    simultaneous_ids = sorted(
        stable_id
        for stable_id, endpoints in observed_transition_endpoints.items()
        if {"source", "target"}.issubset(endpoints)
    )

    for stable_id in simultaneous_ids:
        if stable_id in _SIMULTANEOUS_ACTIVE_OWNER_TRANSITION_STABLE_IDS:
            continue
        if stable_id not in _INACTIVE_TRANSITION_RECONCILIATION_STABLE_IDS:
            raise _simultaneous_reviewed_transition_error(stable_id)
        transition = transitions_by_stable_id.get(stable_id)
        if transition is None:
            raise _simultaneous_reviewed_transition_error(
                stable_id,
                detail="reviewed transition certificate disappeared during scan",
            )
        source_probe = {
            "official_name": transition["source_official_name"],
            "official_athlete_url": transition["source_official_athlete_url"],
            "stable_ufcstats_id": stable_id,
            "fighter_id": stable_id,
            "ufcstats_url": transition["ufcstats_url"],
        }
        exact_inactive_targets = [
            dict(observation)
            for observation in inactive_observations
            if _optional_text(
                observation.get(_PROFILE_FETCH_OUTCOME_COLUMN)
            ).casefold()
            == _PROFILE_FETCH_SUCCEEDED
            and _status_key(observation.get("profile_status")) == "not fighting"
            and _verified_inactive_profile_status(observation)
            and reviewed_official_identity_transition_match(
                observation,
                source_probe,
            )
        ]
        if len(exact_inactive_targets) != 1:
            raise _simultaneous_reviewed_transition_error(
                stable_id,
                detail=(
                    "expected exactly one freshly verified Not Fighting target, "
                    f"found {len(exact_inactive_targets)}"
                ),
            )
        inactive_target = exact_inactive_targets[0]
        source_positions = [
            position
            for position, row in enumerate(reconciled_rows)
            if reviewed_official_identity_transition_match(inactive_target, row)
        ]
        if len(source_positions) != 1:
            raise _simultaneous_reviewed_transition_error(
                stable_id,
                detail=(
                    "expected exactly one live source row for the verified inactive "
                    f"target, found {len(source_positions)}"
                ),
            )

        source_row = reconciled_rows.pop(source_positions[0])
        reconciliations.append(
            {
                "stable_ufcstats_id": stable_id,
                "source_official_name": _optional_text(
                    source_row.get("official_name")
                ),
                "source_official_athlete_url": _optional_text(
                    source_row.get("official_athlete_url")
                ),
                "target_official_name": _optional_text(
                    inactive_target.get("official_name")
                ),
                "target_official_athlete_url": _optional_text(
                    inactive_target.get("official_athlete_url")
                ),
                "target_profile_status": _optional_text(
                    inactive_target.get("profile_status")
                ),
                "reason": "reviewed_transition_target_verified_not_fighting",
            }
        )

    # The already-recorded, positively bound target observation is the sole
    # destructive exclusion certificate.  A second source-route audit row
    # would make one accepted owner appear to have two removal certificates
    # and correctly trip the canonical publication-safety guard.
    return reconciled_rows, reconciliations


def retained_roster_diagnostics(df: pd.DataFrame) -> list[dict[str, object]]:
    """Reconstruct retained-row diagnostics after CSV attrs have been lost."""
    if df.empty:
        return []
    retained = pd.Series(False, index=df.index, dtype=bool)
    if "active_roster_retained_from_previous" in df.columns:
        retained |= df["active_roster_retained_from_previous"].map(_truthy)
    if "active_roster_live_present" in df.columns:
        retained |= df["active_roster_live_present"].map(_falsey)
    if "active_roster_missing_from_live_reason" in df.columns:
        retained |= df["active_roster_missing_from_live_reason"].map(_optional_text).ne("")

    details: list[dict[str, object]] = []
    for _, row in df.loc[retained].iterrows():
        try:
            consecutive_missing_syncs = int(float(row.get("active_roster_consecutive_missing_syncs") or 0))
        except (TypeError, ValueError):
            consecutive_missing_syncs = 0
        details.append(
            {
                "official_name": _optional_text(row.get("official_name")),
                "official_athlete_url": _optional_text(row.get("official_athlete_url")),
                "ufcstats_url": _optional_text(row.get("ufcstats_url")),
                "first_missing_at_utc": (
                    _optional_text(row.get("active_roster_first_missing_at_utc"))
                    or _optional_text(row.get("active_roster_retained_at_utc"))
                ),
                "consecutive_missing_syncs": consecutive_missing_syncs,
            }
        )
    return details


def _cached_roster_shrink_baseline(
    cached_df: pd.DataFrame,
    *,
    excluded_rows: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Build a durable shrink baseline without ratcheting after a bad scan.

    Rows retained by an incomplete/suspicious scan remain in the baseline. Only
    rows already confirmed missing by a complete scan, plus identities positively
    excluded by the current scan's accepted audit, may lower the comparison.
    """
    if cached_df.empty:
        return cached_df

    confirmed_missing = pd.Series(False, index=cached_df.index, dtype=bool)
    for index, cached_row in cached_df.iterrows():
        try:
            missing_syncs = int(
                float(
                    cached_row.get("active_roster_consecutive_missing_syncs")
                    or 0
                )
            )
        except (TypeError, ValueError):
            missing_syncs = 0
        missing_reason = _optional_text(
            cached_row.get("active_roster_missing_from_live_reason")
        )
        if (
            missing_syncs >= 1
            or missing_reason == "absent_from_latest_ufc_active_roster_sync"
        ):
            confirmed_missing.at[index] = True

    accepted_exclusions = excluded_rows or []
    if accepted_exclusions:
        for index, cached_row in cached_df.loc[~confirmed_missing].iterrows():
            if any(
                _roster_row_matches_excluded_identity(cached_row, excluded_row)
                for excluded_row in accepted_exclusions
            ):
                confirmed_missing.at[index] = True

    return cached_df.loc[~confirmed_missing].copy()


def _cached_roster_identity_overlap(
    live_df: pd.DataFrame,
    cached_baseline_df: pd.DataFrame,
) -> int:
    """Count baseline identities still represented by the current live scan."""
    if live_df.empty or cached_baseline_df.empty:
        return 0
    live_rows = [row for _, row in live_df.iterrows()]
    return sum(
        1
        for _, cached_row in cached_baseline_df.iterrows()
        if any(
            _live_cached_roster_rows_same_identity(live_row, cached_row)
            for live_row in live_rows
        )
    )


def _validate_cached_roster_snapshot(cached_df: pd.DataFrame) -> None:
    """Require enough cached identity data to enforce destructive sync guards."""
    if cached_df.empty:
        raise ValueError("cached UFC active-roster snapshot is empty")
    if "official_name" not in cached_df.columns:
        raise ValueError(
            "cached UFC active-roster snapshot is missing required column: official_name"
        )
    identity_url_columns = [
        column
        for column in ("official_athlete_url", "ufcstats_url")
        if column in cached_df.columns
    ]
    if not identity_url_columns:
        raise ValueError(
            "cached UFC active-roster snapshot has no identity URL column"
        )

    names = cached_df["official_name"].map(_optional_text).ne("")
    urls = pd.Series(False, index=cached_df.index, dtype=bool)
    for column in identity_url_columns:
        urls |= cached_df[column].map(_optional_text).ne("")
    if not bool((names & urls).any()):
        raise ValueError(
            "cached UFC active-roster snapshot has no usable fighter identity rows"
        )


def _is_roster_profile_href(value: object) -> bool:
    """Recognize athlete routes emitted by UFC roster cards."""
    text = str(value or "").strip()
    if not text:
        return False
    parsed = urlparse(text)
    if parsed.scheme.casefold() not in {"", "http", "https"}:
        return False
    host = str(parsed.hostname or "").casefold()
    if host and host != "ufc.com" and not host.endswith(".ufc.com"):
        return False
    path = parsed.path.rstrip("/")
    return "/athlete/" in path or re.fullmatch(r"/node/\d+", path) is not None


def _parse_roster_card(card) -> dict[str, object] | None:
    link = card.find("a", href=_is_roster_profile_href)
    name_el = card.select_one("span.c-listing-athlete__name")
    if link is None or name_el is None:
        return None

    division_el = card.select_one("span.c-listing-athlete__title .field__item")
    record_el = card.select_one("span.c-listing-athlete__record")
    nickname_el = card.select_one("span.c-listing-athlete__nickname")
    athlete_url = _absolute_url(link.get("href"))
    athlete_path = urlparse(athlete_url).path
    slug_name = _slug_to_name(athlete_url) if "/athlete/" in athlete_path else ""

    return {
        "official_name": _clean_text(name_el.get_text(" ", strip=True)),
        "official_athlete_url": athlete_url,
        "nickname": _clean_text(nickname_el.get_text(" ", strip=True)) if nickname_el else "",
        "division": _clean_text(division_el.get_text(" ", strip=True)) if division_el else "",
        "record": _clean_text(record_el.get_text(" ", strip=True)) if record_el else "",
        "slug_name": slug_name,
    }


def _roster_page_has_next(soup: BeautifulSoup) -> bool:
    """Return whether UFC.com's listing explicitly advertises another page."""
    return soup.select_one(
        'a[rel~="next"][href], li.pager__item--next a[href]'
    ) is not None


def _roster_drupal_settings(soup: BeautifulSoup) -> dict[str, object]:
    """Return the page's Drupal settings object when it is valid JSON."""
    node = soup.select_one('script[data-drupal-selector="drupal-settings-json"]')
    if node is None:
        return {}
    payload = node.string or node.get_text("", strip=True)
    if not payload:
        return {}
    try:
        value = json.loads(payload)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _reject_unfiltered_athlete_directory(soup: BeautifulSoup) -> None:
    """Reject UFC's explicit all-athletes/no-filter view before card parsing."""
    settings = _roster_drupal_settings(soup)
    ajax_view_settings = settings.get("views")
    ajax_views = (
        ajax_view_settings.get("ajaxViews")
        if isinstance(ajax_view_settings, Mapping)
        else None
    )
    all_athlete_displays: list[str] = []
    if isinstance(ajax_views, Mapping):
        for view in ajax_views.values():
            if not isinstance(view, Mapping):
                continue
            if _clean_text(view.get("view_name")).casefold() != "all_athletes":
                continue
            all_athlete_displays.append(
                _clean_text(view.get("view_display_id")).casefold()
            )
    form = soup.select_one("form.views-exposed-form")
    form_id = _clean_text(form.get("id") if form is not None else "").casefold()
    if (
        any(display == "page_no_filters" for display in all_athlete_displays)
        or "page-no-filters" in form_id
    ):
        display = ",".join(all_athlete_displays) or "missing"
        raise RuntimeError(
            "active_status_scope_unverified: UFC served an unfiltered athlete "
            f"directory view (display={display}, form_id={form_id or 'missing'})"
        )


def _active_roster_listing_scope(
    soup: BeautifulSoup,
    *,
    require_selected_active_facet: bool,
) -> int | None:
    """Fail closed unless UFC proves this is its filtered active-roster view."""
    settings = _roster_drupal_settings(soup)
    _reject_unfiltered_athlete_directory(soup)
    path_settings = settings.get("path")
    current_query = (
        path_settings.get("currentQuery")
        if isinstance(path_settings, Mapping)
        else None
    )
    filters = (
        current_query.get("filters")
        if isinstance(current_query, Mapping)
        else None
    )
    if not isinstance(filters, list) or "status:23" not in filters:
        raise RuntimeError(
            "active_status_scope_unverified: Drupal request state does not certify "
            "the exact status:23 filter"
        )

    ajax_view_settings = settings.get("views")
    ajax_views = (
        ajax_view_settings.get("ajaxViews")
        if isinstance(ajax_view_settings, Mapping)
        else None
    )
    all_athlete_displays = [
        _clean_text(view.get("view_display_id")).casefold()
        for view in (ajax_views.values() if isinstance(ajax_views, Mapping) else ())
        if isinstance(view, Mapping)
        and _clean_text(view.get("view_name")).casefold() == "all_athletes"
    ]
    if not all_athlete_displays:
        raise RuntimeError(
            "active_status_scope_unverified: UFC did not identify the exact "
            "all-athletes filtered view"
        )

    if not require_selected_active_facet:
        return None

    active_facet = None
    for candidate in soup.select('[data-drupal-facet-item-value="23"]'):
        owner = candidate
        if _clean_text(owner.get("data-drupal-facet-id")).casefold() != "status":
            owner = candidate.find_parent(attrs={"data-drupal-facet-id": "status"})
        if owner is None:
            continue
        classes = {
            _clean_text(value).casefold()
            for value in (candidate.get("class") or [])
        }
        selected = (
            bool(classes & {"is-active", "facet-item--active", "active"})
            or _clean_text(candidate.get("aria-checked")).casefold() == "true"
            or candidate.has_attr("checked")
            or candidate.has_attr("selected")
        )
        label = _clean_text(candidate.get_text(" ", strip=True)).casefold()
        if selected and re.search(r"\bactive\b", label):
            active_facet = candidate
            break
    if active_facet is None:
        raise RuntimeError(
            "active_status_scope_unverified: the exact selected Active/status:23 "
            "facet is absent"
        )

    count_text = _clean_text(active_facet.get("data-drupal-facet-item-count"))
    if not count_text.isdigit() or int(count_text) <= 0:
        raise RuntimeError(
            "active_status_scope_unverified: the selected Active/status:23 facet "
            "does not advertise a positive exact count"
        )
    return int(count_text)


def _active_roster_next_page_url(
    soup: BeautifulSoup,
    *,
    current_url: str,
    expected_page: int,
) -> str | None:
    """Return a same-scope next URL or fail closed on a contradictory pager."""
    node = soup.select_one('a[rel~="next"][href], li.pager__item--next a[href]')
    if node is None:
        return None
    next_url = urljoin(current_url, _clean_text(node.get("href")))
    parsed = urlparse(next_url)
    expected = urlparse(OFFICIAL_ACTIVE_ROSTER_URL)
    if (
        parsed.scheme.casefold() != "https"
        or parsed.hostname is None
        or parsed.hostname.casefold() != expected.hostname.casefold()
        or parsed.path.rstrip("/") != expected.path.rstrip("/")
    ):
        raise RuntimeError(
            "active_status_scope_unverified: next-page URL left the exact UFC "
            "active-roster route"
        )
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    if ("filters[0]", "status:23") not in query_pairs:
        raise RuntimeError(
            "active_status_scope_unverified: next-page URL dropped the exact "
            "status:23 filter"
        )
    page_values = [value for key, value in query_pairs if key == "page"]
    if page_values != [str(expected_page)]:
        raise RuntimeError(
            "active_status_scope_unverified: next-page URL has an unexpected "
            f"page sequence ({page_values!r}, expected {expected_page})"
        )
    return next_url


def _roster_page_is_explicit_empty_terminal(soup: BeautifulSoup) -> bool:
    """Recognize UFC's real exhausted-listing response, not an arbitrary blank page."""
    title = _clean_text(soup.title.get_text(" ", strip=True) if soup.title else "").casefold()
    empty_text = " ".join(
        _clean_text(node.get_text(" ", strip=True)).casefold()
        for node in soup.select(".view-empty")
    )
    return (
        "athletes" in title
        and soup.select_one("form.views-exposed-form") is not None
        and "no result found" in empty_text
    )


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
    transport = soup.__dict__.get("_ufc_transport_metadata", {})
    if not isinstance(transport, dict):
        transport = {}
    hero_tags = [_clean_text(tag.get_text(" ", strip=True)) for tag in soup.select("p.hero-profile__tag")]
    hero_tags = [tag for tag in hero_tags if tag]
    hero_name_el = soup.select_one("h1.hero-profile__name")
    division_el = soup.select_one("p.hero-profile__division-title")
    record_el = soup.select_one("p.hero-profile__division-body")
    hero_name = _clean_text(hero_name_el.get_text(" ", strip=True)) if hero_name_el else ""
    if not hero_name:
        title = _clean_text(
            soup.title.get_text(" ", strip=True) if soup.title else ""
        )
        raise OfficialUFCProfileResponseUnavailableError(
            "UFC athlete response did not contain nonempty hero profile identity "
            f"markup (title={title!r})"
        )
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
    canonical_athlete_url = _parse_canonical_athlete_url(soup)

    return {
        "profile_name": hero_name,
        "canonical_athlete_url": canonical_athlete_url,
        "official_profile_fresh_requested_url": _optional_text(
            transport.get("requested_url")
        ),
        "official_profile_fresh_effective_url": _optional_text(
            transport.get("effective_url")
        ),
        "official_profile_fresh_canonical_url": canonical_athlete_url,
        "official_profile_fresh_redirect_chain": _optional_text(
            transport.get("redirect_chain")
        ),
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


_ROUTE_BOUND_CURRENT_UFCSTATS_IDENTITIES: dict[str, dict[str, str]] = {
    normalize_person_name("Bruno Silva"): {
        "official_athlete_url": "https://www.ufc.com/athlete/bruno-silva",
        "nickname": "Bulldog",
        "division": "Flyweight",
        "profile_division": "Flyweight Division",
        "stable_ufcstats_id": "294aa73dbf37d281",
    },
}


def _current_ufcstats_override_for_roster_row(
    row: Mapping[str, object],
) -> tuple[str, bool]:
    """Return a reviewed current ID and whether the name requires its route.

    The Bruno Silva collision cannot be resolved by display name: UFC.com has
    separate Bulldog and Blindado routes and UFCStats has separate stable
    owners for both.  The exact Bulldog route may select its reviewed ID; any
    absent, substituted, redirected, or cross-owner carrier remains blocked
    from both local and live name discovery.
    """

    if reviewed_unbound_simultaneous_alternate_route(row) is not None:
        return "", True

    name_key = normalize_person_name(row.get("official_name"))
    route_binding = _ROUTE_BOUND_CURRENT_UFCSTATS_IDENTITIES.get(name_key)
    if route_binding is None:
        return CURRENT_UFCSTATS_ID_OVERRIDES.get(name_key, ""), False

    expected_url = route_binding["official_athlete_url"]
    expected_url_key = _normalize_roster_url(expected_url)
    observed_url = _trusted_ufc_profile_url(row.get("official_athlete_url"))
    if (
        not observed_url
        or _normalize_roster_url(observed_url) != expected_url_key
    ):
        return "", True

    for field in (
        "canonical_athlete_url",
        "official_profile_fresh_requested_url",
        "official_profile_fresh_effective_url",
        "official_profile_fresh_canonical_url",
    ):
        raw_route = _optional_text(row.get(field))
        if not raw_route:
            continue
        trusted_route = _trusted_ufc_profile_url(raw_route)
        if (
            not trusted_route
            or _normalize_roster_url(trusted_route) != expected_url_key
        ):
            return "", True

    raw_redirect_chain = _optional_text(
        row.get("official_profile_fresh_redirect_chain")
    )
    if raw_redirect_chain:
        redirect_chain = _captured_redirect_chain(raw_redirect_chain)
        if redirect_chain is None or any(
            not _trusted_ufc_profile_url(route)
            or _normalize_roster_url(route) != expected_url_key
            for route in redirect_chain
        ):
            return "", True

    expected_text = {
        "profile_name": name_key,
        "slug_name": name_key,
        "nickname": normalize_person_name(route_binding["nickname"]),
        "division": normalize_person_name(route_binding["division"]),
        "profile_division": normalize_person_name(
            route_binding["profile_division"]
        ),
    }
    for field, expected_value in expected_text.items():
        raw_value = _optional_text(row.get(field))
        if raw_value and normalize_person_name(raw_value) != expected_value:
            return "", True

    expected_stable_id = route_binding["stable_ufcstats_id"]
    for field in _REVIEWED_UFCSTATS_CARRIER_FIELDS:
        raw_carrier = _optional_text(row.get(field))
        if not raw_carrier:
            continue
        observed_stable_id = (
            _supported_ufcstats_profile_id(raw_carrier)
            if field == "ufcstats_url"
            else normalize_ufcstats_id(raw_carrier) or ""
        )
        if observed_stable_id != expected_stable_id:
            return "", True

    return expected_stable_id, True


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

    override_id, route_bound_override = (
        _current_ufcstats_override_for_roster_row(row)
    )
    if route_bound_override and not override_id:
        return {
            "ufcstats_name": "",
            "ufcstats_url": "",
            "ufcstats_resolution": "",
        }
    if override_id:
        override_matches = [
            candidate
            for candidate in match_pool.values()
            if normalize_ufcstats_id(candidate.get("fighter_url")) == override_id
        ]
        if len(override_matches) == 1:
            candidate = override_matches[0]
            return {
                "ufcstats_name": candidate["name"],
                "ufcstats_url": candidate["fighter_url"],
                "ufcstats_resolution": "reviewed_current_identity",
            }
        return {
            "ufcstats_name": _clean_text(row.get("official_name")),
            "ufcstats_url": f"http://ufcstats.com/fighter-details/{override_id}",
            "ufcstats_resolution": "reviewed_current_identity",
        }

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

    override_id, route_bound_override = (
        _current_ufcstats_override_for_roster_row(row)
    )
    if route_bound_override and not override_id:
        return {
            "ufcstats_name": "",
            "ufcstats_url": "",
            "ufcstats_resolution": "",
        }

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
        fighter_url = search_fighter_url(
            query_text,
            fighter_id=override_id or None,
        )
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
    classify_combat_sport: bool = True,
    profile_identity_decisions: pd.DataFrame | None = None,
    session: requests.Session | None = None,
    _identity_audit_sink: list[dict[str, object]] | None = None,
    _verified_inactive_sink: list[dict[str, object]] | None = None,
    _require_active_listing_scope: bool = False,
) -> pd.DataFrame:
    client = session or requests.Session()
    rows: list[dict[str, object]] = []
    identity_audit_rows = (
        _identity_audit_sink if _identity_audit_sink is not None else []
    )
    verified_inactive_profile_observations = (
        _verified_inactive_sink if _verified_inactive_sink is not None else []
    )
    unbound_simultaneous_route_observations: list[dict[str, object]] = []
    transition_endpoints_by_entity: dict[str, tuple[str, str]] = {}
    for transition in reviewed_official_identity_transitions():
        stable_id = str(transition["stable_ufcstats_id"])
        for endpoint in ("source", "target"):
            transition_endpoints_by_entity[
                str(transition[f"{endpoint}_official_source_entity_id"])
            ] = (stable_id, endpoint)
    observed_transition_endpoints: dict[str, set[str]] = {}
    observed_transition_endpoint_counts: dict[str, dict[str, int]] = {}
    seen_urls: set[str] = set()
    page = 0
    pages_scraped = 0
    selected_card_count = 0
    parsed_card_count = 0
    unique_listing_identity_count = 0
    advertised_active_count: int | None = None
    active_listing_scope_verified = False
    scan_parser_complete = True
    sync_complete = False
    sync_completeness_reason = "not_started"
    candidates = _load_local_ufcstats_candidates() if resolve_ufcstats else {}
    verified_ufc_bindings = _validated_ufc_profile_binding_index(
        profile_identity_decisions
    )

    page_url = OFFICIAL_ACTIVE_ROSTER_URL
    while True:
        if max_pages is not None and page >= max_pages:
            sync_completeness_reason = "max_pages_reached_before_terminal_page"
            break
        soup = _get_soup(page_url, session=client)
        transport_metadata = soup.__dict__.get("_ufc_transport_metadata")
        enforce_active_listing_scope = (
            _require_active_listing_scope
            or isinstance(transport_metadata, Mapping)
        )
        if enforce_active_listing_scope:
            observed_count = _active_roster_listing_scope(
                soup,
                require_selected_active_facet=page == 0,
            )
            if page == 0:
                advertised_active_count = observed_count
                active_listing_scope_verified = True
        else:
            _reject_unfiltered_athlete_directory(soup)
        cards = soup.select("div.c-listing-athlete-flipcard")
        if not cards:
            if page > 0 and _roster_page_is_explicit_empty_terminal(soup):
                sync_complete = scan_parser_complete
                sync_completeness_reason = (
                    "explicit_empty_terminal_page"
                    if scan_parser_complete
                    else "card_parse_coverage_incomplete"
                )
            else:
                sync_completeness_reason = "unexpected_empty_page"
            break
        pages_scraped += 1
        selected_card_count += len(cards)

        page_added = 0
        page_parsed = 0
        page_new_identities = 0
        for card in cards:
            parsed = _parse_roster_card(card)
            if not parsed:
                continue
            page_parsed += 1
            athlete_url = str(parsed.get("official_athlete_url") or "").strip()
            athlete_url_key = _normalize_roster_url(athlete_url)
            route_entity = source_entity_id("ufc", athlete_url)
            transition_endpoint = transition_endpoints_by_entity.get(route_entity)
            if transition_endpoint is not None:
                transition_stable_id, endpoint = transition_endpoint
                if (
                    transition_stable_id
                    in _STRICT_FRESH_CACHED_ROUTE_TRANSITION_STABLE_IDS
                ):
                    endpoint_counts = observed_transition_endpoint_counts.setdefault(
                        transition_stable_id,
                        {},
                    )
                    endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
                    if endpoint_counts[endpoint] != 1:
                        raise _simultaneous_reviewed_transition_error(
                            transition_stable_id,
                            detail=(
                                "duplicate listing cards for exact "
                                f"{endpoint} endpoint"
                            ),
                        )
                endpoints = observed_transition_endpoints.setdefault(
                    transition_stable_id, set()
                )
                if (
                    endpoints
                    and endpoint not in endpoints
                    and transition_stable_id
                    not in (
                        _INACTIVE_TRANSITION_RECONCILIATION_STABLE_IDS
                        | _SIMULTANEOUS_ACTIVE_OWNER_TRANSITION_STABLE_IDS
                    )
                ):
                    raise _simultaneous_reviewed_transition_error(
                        transition_stable_id
                    )
                endpoints.add(endpoint)
            if not athlete_url_key or athlete_url_key in seen_urls:
                continue
            seen_urls.add(athlete_url_key)
            page_new_identities += 1
            unique_listing_identity_count += 1
            non_mma_exclusion = reviewed_non_mma_exclusion(parsed)
            if non_mma_exclusion is not None:
                parsed.update(
                    {
                        _PROFILE_FETCH_OUTCOME_COLUMN: _PROFILE_FETCH_NOT_REQUESTED,
                        "official_url_identity_valid": False,
                        "official_url_identity_status": "reviewed_non_mma_identity",
                        "official_url_identity_reason": non_mma_exclusion["reason"],
                        "combat_sport": "power_slap",
                        "combat_sport_reason": non_mma_exclusion["reason"],
                        "combat_sport_profile_url": "",
                        "coverage_eligible": False,
                    }
                )
                identity_audit_rows.append(
                    _identity_audit_row(parsed, action="excluded_non_mma_identity")
                )
                logger.info(
                    "Excluded reviewed non-MMA UFC identity before profile requests: %s (%s)",
                    parsed.get("official_name"),
                    parsed.get("official_athlete_url"),
                )
                continue
            registered_mma_binding = reviewed_mma_identity_registry_record(parsed)
            initial_reviewed_mma_binding = reviewed_mma_identity_binding(parsed)
            pending_reviewed_transition_id = (
                reviewed_mma_identity_transition_pending_stable_id(parsed)
            )
            pending_reviewed_source_canonical = (
                reviewed_mma_identity_transition_pending_source_canonical(parsed)
            )
            registered_source_transition = bool(
                pending_reviewed_source_canonical
                or (
                    initial_reviewed_mma_binding is not None
                    and any(
                        transition["source_official_source_entity_id"]
                        == initial_reviewed_mma_binding["official_source_entity_id"]
                        for transition in reviewed_official_identity_transitions()
                    )
                )
            )
            if (
                registered_mma_binding is not None
                and initial_reviewed_mma_binding is None
                and not pending_reviewed_transition_id
                and not pending_reviewed_source_canonical
            ):
                _quarantine_reviewed_mma_identity_conflict(
                    parsed,
                    reason=(
                        "registered_official_entity_name_or_stable_id_conflict"
                    ),
                )
                identity_audit_rows.append(
                    _identity_audit_row(
                        parsed,
                        action="quarantined_reviewed_mma_identity_conflict",
                    )
                )
                rows.append(parsed)
                page_added += 1
                logger.warning(
                    "Quarantined changed carriers for reviewed MMA identity: %s (%s)",
                    parsed.get("official_name"),
                    parsed.get("official_athlete_url"),
                )
                continue
            reviewed_identity_entity_before_profile = ""
            if resolve_ufcstats and (
                reviewed_profile_identity_binding_record(parsed) is not None
                or reviewed_ufcstats_resolution_blocked(parsed)
            ):
                reviewed_identity_entity_before_profile = source_entity_id(
                    "ufc",
                    parsed.get("official_athlete_url"),
                )
            if fetch_profile_details:
                try:
                    parsed.update(scrape_official_athlete_profile(athlete_url, session=client))
                    if not registered_source_transition:
                        _apply_canonical_official_athlete_url(parsed)
                    parsed[_PROFILE_FETCH_OUTCOME_COLUMN] = _PROFILE_FETCH_SUCCEEDED
                except Exception as exc:
                    if isinstance(
                        exc,
                        OfficialUFCProfileResponseUnavailableError,
                    ):
                        parsed[_PROFILE_FETCH_OUTCOME_COLUMN] = (
                            _PROFILE_FETCH_INVALID_RESPONSE
                        )
                    elif isinstance(
                        exc,
                        (
                            requests.exceptions.Timeout,
                            requests.exceptions.ConnectionError,
                        ),
                    ):
                        parsed[_PROFILE_FETCH_OUTCOME_COLUMN] = (
                            _PROFILE_FETCH_REQUEST_FAILED
                        )
                    elif isinstance(exc, requests.exceptions.HTTPError):
                        parsed[_PROFILE_FETCH_OUTCOME_COLUMN] = (
                            _PROFILE_FETCH_HTTP_ERROR
                        )
                    elif isinstance(exc, requests.exceptions.RequestException):
                        parsed[_PROFILE_FETCH_OUTCOME_COLUMN] = (
                            _PROFILE_FETCH_REQUEST_ERROR
                        )
                    else:
                        parsed[_PROFILE_FETCH_OUTCOME_COLUMN] = (
                            _PROFILE_FETCH_PARSE_FAILED
                        )
                    logger.warning("Failed to scrape official athlete profile %s: %s", athlete_url, exc)
            else:
                parsed[_PROFILE_FETCH_OUTCOME_COLUMN] = _PROFILE_FETCH_NOT_REQUESTED
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

            unbound_route_certificate = (
                reviewed_unbound_simultaneous_alternate_route(parsed)
            )
            if pending_reviewed_source_canonical and _optional_text(
                parsed.get(_PROFILE_FETCH_OUTCOME_COLUMN)
            ).casefold() != _PROFILE_FETCH_SUCCEEDED:
                _quarantine_reviewed_mma_identity_conflict(
                    parsed,
                    reason="reviewed_source_transition_canonical_not_verified",
                )
                identity_audit_rows.append(
                    _identity_audit_row(
                        parsed,
                        action="quarantined_reviewed_mma_identity_conflict",
                    )
                )
                rows.append(parsed)
                page_added += 1
                continue
            if (
                unbound_route_certificate is not None
                and _optional_text(
                    parsed.get(_PROFILE_FETCH_OUTCOME_COLUMN)
                ).casefold()
                == _PROFILE_FETCH_SUCCEEDED
            ):
                unbound_simultaneous_route_observations.append(
                    {
                        "official_name": _optional_text(
                            parsed.get("official_name")
                        ),
                        "official_athlete_url": _optional_text(
                            parsed.get("official_athlete_url")
                        ),
                        "profile": {
                            field: _optional_text(parsed.get(field))
                            for field in _UNTRUSTED_OFFICIAL_PROFILE_FIELDS
                        },
                    }
                )

            if (
                registered_source_transition
                and _optional_text(
                    parsed.get(_PROFILE_FETCH_OUTCOME_COLUMN)
                ).casefold()
                == _PROFILE_FETCH_SUCCEEDED
                and reviewed_mma_identity_binding(parsed) is None
            ):
                _quarantine_reviewed_mma_identity_conflict(
                    parsed,
                    reason="reviewed_source_transition_conflicted_after_profile",
                )
                identity_audit_rows.append(
                    _identity_audit_row(
                        parsed,
                        action="quarantined_reviewed_mma_identity_conflict",
                    )
                )
                rows.append(parsed)
                page_added += 1
                logger.warning(
                    "Quarantined reviewed source route after contradictory profile "
                    "evidence: %s (%s)",
                    parsed.get("official_name"),
                    parsed.get("official_athlete_url"),
                )
                continue

            canonical_url_key = _normalize_roster_url(parsed.get("official_athlete_url"))
            if canonical_url_key:
                if canonical_url_key != athlete_url_key and canonical_url_key in seen_urls:
                    continue
                seen_urls.add(canonical_url_key)

            # A UFCStats search is discovery only.  It may nominate the stable
            # ID used to look up an independently validated source binding, but
            # the name search itself never authorizes UFC.com biography data.
            reviewed_binding_applied = bool(
                resolve_ufcstats and apply_reviewed_mma_identity_binding(parsed)
            )
            reviewed_profile_binding_applied = False
            reviewed_resolution_blocked = False
            if resolve_ufcstats and not reviewed_binding_applied:
                (
                    reviewed_profile_binding_applied,
                    reviewed_resolution_blocked,
                ) = _apply_reviewed_roster_identity_resolution_policy(
                    parsed,
                    expected_official_entity=(
                        reviewed_identity_entity_before_profile
                    ),
                )
            if (
                resolve_ufcstats
                and not reviewed_binding_applied
                and not reviewed_profile_binding_applied
                and not reviewed_resolution_blocked
                and reviewed_unbound_simultaneous_alternate_route(parsed) is None
            ):
                resolution = _resolve_local_ufcstats_profile(
                    parsed,
                    candidates=candidates,
                )
                if not resolution.get("ufcstats_url"):
                    resolution = _resolve_via_live_search(parsed, session=client)
                parsed.update(resolution)
            if pending_reviewed_transition_id and not reviewed_binding_applied:
                reviewed_binding_applied = bool(
                    resolve_ufcstats
                    and apply_reviewed_mma_identity_binding(parsed)
                )
                if not reviewed_binding_applied:
                    _quarantine_reviewed_mma_identity_conflict(
                        parsed,
                        reason=(
                            "reviewed_transition_stable_id_not_verified_after_"
                            "discovery"
                        ),
                    )
                    identity_audit_rows.append(
                        _identity_audit_row(
                            parsed,
                            action=(
                                "quarantined_reviewed_mma_identity_conflict"
                            ),
                        )
                    )
                    rows.append(parsed)
                    page_added += 1
                    logger.warning(
                        "Quarantined reviewed route transition without its exact "
                        "stable identity: %s (%s)",
                        parsed.get("official_name"),
                        parsed.get("official_athlete_url"),
                    )
                    continue
            _apply_verified_ufc_profile_binding(
                parsed,
                bindings_by_stable_id=verified_ufc_bindings,
            )
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
                verified_inactive_profile_observations.append(dict(parsed))
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

            if (
                classify_combat_sport
                and not reviewed_resolution_blocked
                and reviewed_unbound_simultaneous_alternate_route(parsed) is None
            ):
                parsed.update(_classify_combat_sport(parsed, session=client))
            parsed["coverage_eligible"] = str(parsed.get("combat_sport") or "").strip().lower() != "power_slap"
            if _profile_status_is_unknown(parsed):
                if _profile_status_is_independently_verified_current(parsed):
                    parsed["profile_status"] = "Active"
                    parsed["active_roster_status_reason"] = "independently_verified_current"
                else:
                    parsed["profile_status"] = _PROFILE_STATUS_UNKNOWN
                    parsed["coverage_eligible"] = False
                    parsed["active_roster_status_reason"] = "profile_status_missing_or_unrecognized"
                    identity_audit_rows.append(
                        _identity_audit_row(parsed, action="quarantined_unknown_profile_status")
                    )
                    logger.info(
                        "Quarantined UFC athlete row with unverified current status: %s (%s)",
                        parsed.get("official_name"),
                        parsed.get("official_athlete_url"),
                    )
            else:
                parsed["active_roster_status_reason"] = ""
            rows.append(parsed)
            page_added += 1

        parsed_card_count += page_parsed

        logger.info(
            "Scraped official UFC active roster page %d with %d/%d parseable cards and %d accepted rows",
            page,
            page_parsed,
            len(cards),
            page_added,
        )
        if page_parsed != len(cards):
            scan_parser_complete = False
            sync_completeness_reason = "card_parse_coverage_incomplete"
        if page_new_identities <= 0:
            if enforce_active_listing_scope:
                raise RuntimeError(
                    "active_status_scope_unverified: filtered roster pagination "
                    "made no unique-identity progress"
                )
            if scan_parser_complete:
                sync_completeness_reason = "pagination_made_no_progress"
            break
        if enforce_active_listing_scope:
            if advertised_active_count is None:
                raise RuntimeError(
                    "active_status_scope_unverified: missing advertised active count"
                )
            if unique_listing_identity_count > advertised_active_count:
                raise RuntimeError(
                    "active_status_scope_unverified: filtered roster exceeded its "
                    "advertised exact count "
                    f"({unique_listing_identity_count}>{advertised_active_count})"
                )
            next_page_url = _active_roster_next_page_url(
                soup,
                current_url=page_url,
                expected_page=page + 1,
            )
            if unique_listing_identity_count == advertised_active_count:
                if next_page_url is not None:
                    raise RuntimeError(
                        "active_status_scope_unverified: filtered roster reached its "
                        "advertised exact count while still exposing a next page"
                    )
                sync_complete = scan_parser_complete
                sync_completeness_reason = (
                    "verified_active_scope_exact_count"
                    if scan_parser_complete
                    else "card_parse_coverage_incomplete"
                )
                break
            if next_page_url is None:
                raise RuntimeError(
                    "active_status_scope_unverified: filtered roster ended before "
                    "its advertised exact count "
                    f"({unique_listing_identity_count}<{advertised_active_count})"
                )
            page += 1
            page_url = next_page_url
            continue
        # A missing/changed next-link selector is not positive terminal
        # evidence. Probe the following page; UFC's real terminal response is a
        # normal athletes page with an explicit "No Result Found" view.
        page += 1
        page_url = f"{OFFICIAL_ACTIVE_ROSTER_URL}&page={page}"

    (
        rows,
        simultaneous_active_owner_reconciliations,
    ) = _reconcile_reviewed_simultaneous_active_owner_transition_rows(
        rows,
        observed_transition_endpoints,
    )
    if simultaneous_active_owner_reconciliations:
        logger.info(
            "Reconciled %d exact simultaneous active duplicate-owner route "
            "transition(s) after fresh identity verification",
            len(simultaneous_active_owner_reconciliations),
        )
    (
        rows,
        simultaneous_inactive_reconciliations,
    ) = _reconcile_reviewed_simultaneous_inactive_transition_rows(
        rows,
        observed_transition_endpoints,
        verified_inactive_profile_observations,
    )
    if simultaneous_inactive_reconciliations:
        logger.info(
            "Reconciled %d exact simultaneous Anderson route transition(s) after "
            "fresh Not Fighting verification",
            len(simultaneous_inactive_reconciliations),
        )

    if not rows:
        df = pd.DataFrame()
        df.attrs["identity_audit_rows"] = identity_audit_rows
        df.attrs[_VERIFIED_INACTIVE_PROFILE_OBSERVATIONS_ATTR] = (
            verified_inactive_profile_observations
        )
        df.attrs[_UNBOUND_SIMULTANEOUS_ROUTE_OBSERVATIONS_ATTR] = (
            unbound_simultaneous_route_observations
        )
        df.attrs["sync_complete"] = sync_complete
        df.attrs["sync_completeness_reason"] = sync_completeness_reason
        df.attrs["pages_scraped"] = pages_scraped
        df.attrs["selected_card_count"] = selected_card_count
        df.attrs["parsed_card_count"] = parsed_card_count
        df.attrs["active_listing_scope_verified"] = active_listing_scope_verified
        df.attrs["active_listing_advertised_count"] = advertised_active_count
        df.attrs["unique_listing_identity_count"] = unique_listing_identity_count
        df.attrs["reconciled_simultaneous_inactive_transitions"] = (
            simultaneous_inactive_reconciliations
        )
        df.attrs[_RECONCILED_SIMULTANEOUS_ACTIVE_OWNER_TRANSITIONS_ATTR] = (
            simultaneous_active_owner_reconciliations
        )
        return df

    df = pd.DataFrame(rows)
    df = df.sort_values(["official_name", "official_athlete_url"]).reset_index(drop=True)
    df.attrs["identity_audit_rows"] = identity_audit_rows
    df.attrs[_VERIFIED_INACTIVE_PROFILE_OBSERVATIONS_ATTR] = (
        verified_inactive_profile_observations
    )
    df.attrs[_UNBOUND_SIMULTANEOUS_ROUTE_OBSERVATIONS_ATTR] = (
        unbound_simultaneous_route_observations
    )
    df.attrs["sync_complete"] = sync_complete
    df.attrs["sync_completeness_reason"] = sync_completeness_reason
    df.attrs["pages_scraped"] = pages_scraped
    df.attrs["selected_card_count"] = selected_card_count
    df.attrs["parsed_card_count"] = parsed_card_count
    df.attrs["active_listing_scope_verified"] = active_listing_scope_verified
    df.attrs["active_listing_advertised_count"] = advertised_active_count
    df.attrs["unique_listing_identity_count"] = unique_listing_identity_count
    df.attrs["reconciled_simultaneous_inactive_transitions"] = (
        simultaneous_inactive_reconciliations
    )
    df.attrs[_RECONCILED_SIMULTANEOUS_ACTIVE_OWNER_TRANSITIONS_ATTR] = (
        simultaneous_active_owner_reconciliations
    )
    unknown_status_rows = sum(
        1
        for row in identity_audit_rows
        if str(row.get("action") or "").strip() == "quarantined_unknown_profile_status"
    )
    if unknown_status_rows:
        logger.info(
            "Quarantined %d UFC athlete row(s) with unverified current profile status",
            unknown_status_rows,
        )
    return df


def sync_official_active_roster(
    *,
    output_path: Path | None = None,
    fetch_profile_details: bool = True,
    max_pages: int | None = None,
    resolve_ufcstats: bool = True,
    classify_combat_sport: bool = True,
    profile_identity_decisions: pd.DataFrame | None = None,
    allow_cached_fallback: bool = True,
    identity_audit_path: Path | None | object = _DEFAULT_IDENTITY_AUDIT_PATH,
) -> pd.DataFrame:
    output_path = Path(output_path) if output_path is not None else OFFICIAL_ACTIVE_ROSTER_PATH
    publishes_canonical_roster = _paths_same_directory_entry(
        output_path,
        OFFICIAL_ACTIVE_ROSTER_PATH,
    )
    if identity_audit_path is _DEFAULT_IDENTITY_AUDIT_PATH:
        identity_audit_path = (
            OFFICIAL_ACTIVE_ROSTER_IDENTITY_AUDIT_PATH
            if publishes_canonical_roster
            else None
        )
    if (
        not fetch_profile_details
        and _paths_target_same_file(
            output_path,
            OFFICIAL_ACTIVE_ROSTER_PATH,
        )
    ):
        raise ValueError(
            "Refusing to publish a listing-only UFC roster scan to the canonical "
            "active-roster artifact while profile details are skipped. Fetch profile "
            "details or choose a noncanonical temporary --output path."
        )
    if (
        not classify_combat_sport
        and _paths_target_same_file(
            output_path,
            OFFICIAL_ACTIVE_ROSTER_PATH,
        )
    ):
        raise ValueError(
            "Refusing to publish a UFC roster scan without combat-sport "
            "classification to the canonical active-roster artifact. Enable "
            "classification or choose a noncanonical temporary output path."
        )
    if identity_audit_path is not None:
        identity_audit_path = Path(identity_audit_path)
        if (
            not publishes_canonical_roster
            and _paths_target_same_file(
                identity_audit_path,
                OFFICIAL_ACTIVE_ROSTER_IDENTITY_AUDIT_PATH,
            )
        ):
            raise ValueError(
                "Refusing to publish a noncanonical UFC roster generation to the "
                "canonical identity-audit sidecar. Omit identity_audit_path or "
                "choose a matching noncanonical audit path."
            )
        if _paths_target_same_file(
            identity_audit_path,
            OFFICIAL_ACTIVE_ROSTER_PATH,
        ):
            raise ValueError(
                "Refusing to use the canonical UFC active-roster artifact as the "
                "identity-audit output path."
            )
        if _paths_target_same_file(identity_audit_path, output_path):
            raise ValueError(
                "Refusing to use the same output path for the UFC active roster and "
                "its identity audit."
            )
    accepted_cached_df: pd.DataFrame | None = None
    identity_audit_rows: list[dict[str, object]] = []
    verified_inactive_profile_observations: list[dict[str, object]] = []
    reconciled_simultaneous_inactive_transitions: list[dict[str, object]] = []
    reconciled_simultaneous_active_owner_transitions: list[
        dict[str, object]
    ] = []
    reconciled_inherited_duplicate_owner_transitions: list[
        dict[str, object]
    ] = []
    intentionally_removed_cached_rows: list[dict[str, object]] = []
    restored_profile_request_failure_statuses = 0
    legacy_request_failure_preserved_rows = 0
    preserved_immutable_profile_conflicts: list[dict[str, str]] = []
    quarantined_unbound_alternate_routes: list[dict[str, object]] = []
    try:
        df = scrape_official_active_roster(
            fetch_profile_details=fetch_profile_details,
            max_pages=max_pages,
            resolve_ufcstats=resolve_ufcstats,
            classify_combat_sport=classify_combat_sport,
            profile_identity_decisions=profile_identity_decisions,
            _identity_audit_sink=identity_audit_rows,
            _verified_inactive_sink=verified_inactive_profile_observations,
        )
        returned_identity_audit_rows = list(
            df.attrs.get("identity_audit_rows") or []
        )
        if returned_identity_audit_rows:
            identity_audit_rows = returned_identity_audit_rows
        returned_inactive_observations = list(
            df.attrs.get(_VERIFIED_INACTIVE_PROFILE_OBSERVATIONS_ATTR) or []
        )
        if returned_inactive_observations:
            verified_inactive_profile_observations = returned_inactive_observations
        reconciled_simultaneous_inactive_transitions = list(
            df.attrs.get("reconciled_simultaneous_inactive_transitions") or []
        )
        reconciled_simultaneous_active_owner_transitions = list(
            df.attrs.get(
                _RECONCILED_SIMULTANEOUS_ACTIVE_OWNER_TRANSITIONS_ATTR
            )
            or []
        )
        sync_complete = df.attrs.get("sync_complete") is True
        sync_completeness_reason = str(
            df.attrs.get("sync_completeness_reason") or "missing_explicit_completion_evidence"
        )
        pages_scraped = int(df.attrs.get("pages_scraped") or 0)
        selected_card_count = int(df.attrs.get("selected_card_count") or 0)
        parsed_card_count = int(df.attrs.get("parsed_card_count") or 0)
        intentionally_excluded_rows = _excluded_identity_rows_from_audit_rows(
            identity_audit_rows
        )
        df = _apply_profile_status_eligibility(df)
        if not output_path.exists():
            (
                profile_status_comparison_rows,
                lost_profile_status_rows,
            ) = _successful_profile_status_loss_counts(df, pd.DataFrame())
            if _profile_status_loss_exceeds_drift_guard(
                profile_status_comparison_rows,
                lost_profile_status_rows,
            ):
                raise RuntimeError(
                    "Official UFC roster live sync detected probable profile-status "
                    "markup drift on a cacheless sync: "
                    f"{lost_profile_status_rows} of "
                    f"{profile_status_comparison_rows} trusted profiles lost status "
                    "after successful fetches; refusing to publish a "
                    "mass-quarantined first generation"
                )
        retained_missing_live_rows: list[dict[str, object]] = []
        expired_missing_live_rows: list[dict[str, object]] = []
        intentionally_removed_cached_rows = []
        discarded_suspicious_cached_rows = 0
        if output_path.exists():
            try:
                cached_df = pd.read_csv(output_path)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load existing cached UFC active-roster rows; "
                    "refusing a guardless live overwrite"
                ) from exc
            _validate_cached_roster_snapshot(cached_df)
            accepted_cached_df = cached_df.copy()
            (
                df,
                quarantined_unbound_alternate_routes,
            ) = _quarantine_reviewed_unbound_simultaneous_alternate_routes(
                df,
                cached_df,
            )
            if quarantined_unbound_alternate_routes:
                identity_audit_rows.extend(quarantined_unbound_alternate_routes)
                logger.warning(
                    "Quarantined %d exact unbound simultaneous alternate UFC route "
                    "observation(s) without aliasing or destructive exclusion",
                    len(quarantined_unbound_alternate_routes),
                )
            (
                cached_df,
                reconciled_inherited_duplicate_owner_transitions,
            ) = _reconcile_reviewed_inherited_duplicate_owner_cache(
                df,
                cached_df,
            )
            if reconciled_inherited_duplicate_owner_transitions:
                logger.info(
                    "Prepared %d exact inherited duplicate-owner transition(s) "
                    "for one-row canonical review",
                    len(reconciled_inherited_duplicate_owner_transitions),
                )
            identity_conflicts = _cached_live_identity_conflicts(df, cached_df)
            if identity_conflicts:
                first_conflict = identity_conflicts[0]
                raise RuntimeError(
                    "Official UFC roster live sync found a conflicting cached "
                    "identity carrier; refusing all cached carry-forward and "
                    "preserving the accepted roster: "
                    f"shared={first_conflict['shared_carrier']}, "
                    f"live={first_conflict['live_name']!r}/"
                    f"{first_conflict['live_ufcstats_id'] or 'missing_id'}/"
                    f"{first_conflict['live_official_url'] or 'missing_url'}, "
                    f"cached={first_conflict['cached_name']!r}/"
                    f"{first_conflict['cached_ufcstats_id'] or 'missing_id'}/"
                    f"{first_conflict['cached_official_url'] or 'missing_url'}"
                )
            cached_df = _apply_profile_status_eligibility(cached_df)
            (
                profile_status_comparison_rows,
                lost_profile_status_rows,
            ) = _successful_profile_status_loss_counts(df, cached_df)
            if _profile_status_loss_exceeds_drift_guard(
                profile_status_comparison_rows,
                lost_profile_status_rows,
            ):
                raise RuntimeError(
                    "Official UFC roster live sync detected probable profile-status "
                    "markup drift: "
                    f"{lost_profile_status_rows} of "
                    f"{profile_status_comparison_rows} trusted, cached-current "
                    "profiles lost status after successful fetches; refusing to "
                    "publish a mass-quarantined roster"
                )
            (
                df,
                restored_profile_request_failure_statuses,
            ) = _restore_cached_current_status_after_profile_request_failures(
                df,
                cached_df,
            )
            legacy_request_failure_preserved_rows = int(
                df.attrs.get(
                    "legacy_accepted_profile_request_failure_preserved_rows",
                    0,
                )
                or 0
            )
            df = _apply_profile_status_eligibility(df)
            (
                df,
                preserved_immutable_profile_conflicts,
            ) = _preserve_accepted_immutable_profile_conflicts(df, cached_df)
            if preserved_immutable_profile_conflicts:
                preview = ", ".join(
                    f"{row['official_name']}:{row['field']}"
                    for row in preserved_immutable_profile_conflicts[:10]
                )
                logger.warning(
                    "Preserved %d accepted immutable UFC profile value(s) after "
                    "conflicting successful live observations%s",
                    len(preserved_immutable_profile_conflicts),
                    f": {preview}" if preview else "",
                )

            if _live_roster_growth_exceeds_cached_guard(df, cached_df):
                raise RuntimeError(
                    "Official UFC roster live sync returned "
                    f"{len(df)} rows versus cached {len(cached_df)} rows; "
                    "refusing to overwrite cached roster because this exceeds the growth guard"
                )

            shrink_baseline_df = _cached_roster_shrink_baseline(
                cached_df,
                excluded_rows=intentionally_excluded_rows,
            )
            shrink_baseline_rows = int(len(shrink_baseline_df))
            baseline_identity_overlap = _cached_roster_identity_overlap(
                df,
                shrink_baseline_df,
            )
            if (
                sync_complete
                and shrink_baseline_rows >= _ROSTER_SHRINK_INCOMPLETE_MIN_CACHED_ROWS
                and baseline_identity_overlap
                <= int(shrink_baseline_rows * _ROSTER_SHRINK_INCOMPLETE_RATIO)
                and shrink_baseline_rows - baseline_identity_overlap
                >= _ROSTER_SHRINK_INCOMPLETE_ABSOLUTE_ROWS
            ):
                sync_complete = False
                sync_completeness_reason = (
                    "suspicious_live_shrink:"
                    f"{baseline_identity_overlap}_of_{shrink_baseline_rows}"
                )
                logger.warning(
                    "Treating UFC active-roster scan as incomplete because it matched %d "
                    "of %d cached baseline identities (%d total live rows)",
                    baseline_identity_overlap,
                    shrink_baseline_rows,
                    len(df),
                )

            try:
                (
                    df,
                    retained_missing_live_rows,
                    expired_missing_live_rows,
                ) = _merge_cached_roster_rows_missing_from_live(
                    df,
                    cached_df,
                    retained_at_utc=_utc_now().isoformat(),
                    excluded_rows=intentionally_excluded_rows,
                    sync_complete=sync_complete,
                    intentionally_removed_rows=intentionally_removed_cached_rows,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to preserve cached UFC active-roster rows missing from live sync"
                ) from exc
            if _destructive_cached_exclusions_exceed_guard(
                len(cached_df),
                len(intentionally_removed_cached_rows),
            ):
                raise RuntimeError(
                    "Official UFC roster live sync would destructively remove "
                    f"{len(intentionally_removed_cached_rows)} of "
                    f"{len(cached_df)} cached rows through accepted inactive/test/non-MMA "
                    "identity exclusions; refusing to publish an anomalously "
                    "large destructive cleanup"
                )
        df = _ensure_active_roster_provenance_schema(df)
        legacy_immutable_conflict_count = sum(
            row.get("accepted_identity_provenance")
            == "legacy_pre_binding_certificate"
            for row in preserved_immutable_profile_conflicts
        )
        if (
            legacy_request_failure_preserved_rows
            or legacy_immutable_conflict_count
        ):
            raise RuntimeError(
                "Official UFC roster live sync used an exact stable-ID/URL/name "
                "match only to preserve a pre-certificate accepted snapshot; "
                "refusing candidate publication and preserving every accepted "
                "roster byte "
                f"(transient_rows={legacy_request_failure_preserved_rows}, "
                f"immutable_conflicts={legacy_immutable_conflict_count})"
            )
        unbound_canonical_changes = (
            _unbound_canonical_candidate_changes(
                accepted_cached_df,
                df,
                identity_audit_rows=identity_audit_rows,
                reconciled_simultaneous_inactive_transitions=(
                    reconciled_simultaneous_inactive_transitions
                ),
                reconciled_inherited_duplicate_owner_transitions=(
                    reconciled_inherited_duplicate_owner_transitions
                ),
                verified_inactive_profile_observations=(
                    verified_inactive_profile_observations
                ),
            )
            if publishes_canonical_roster and accepted_cached_df is not None
            else []
        )
        if unbound_canonical_changes:
            first_unbound_change = unbound_canonical_changes[0]
            raise RuntimeError(
                "Canonical UFC roster candidate would publish an accepted-row "
                "change without that fighter's own positively validated profile "
                "identity binding; refusing discovery-only candidate publication "
                "and preserving the accepted roster/audit bytes "
                f"(violations={len(unbound_canonical_changes)}, "
                f"fighter={first_unbound_change['official_name']!r}, "
                f"reason={first_unbound_change['reason']})"
            )
        candidate_audit = _identity_audit_frame(
            identity_audit_rows,
            sync_generation_id="",
        )
        sync_generation_id = _matching_existing_roster_audit_generation(
            roster_path=output_path,
            audit_path=identity_audit_path,
            candidate_roster=df,
            candidate_audit=candidate_audit,
        )
        roster_identity_audit_pair_reused = bool(sync_generation_id)
        audit_write_error = ""
        if roster_identity_audit_pair_reused:
            df[ACTIVE_ROSTER_SYNC_GENERATION_COLUMN] = sync_generation_id
        else:
            sync_generation_id = uuid.uuid4().hex
            df[ACTIVE_ROSTER_SYNC_GENERATION_COLUMN] = sync_generation_id
            write_csv_atomically(df, output_path, refuse_empty=True)
            try:
                # The audit is published only after its roster generation has passed
                # every guard and been atomically accepted. Startup additionally
                # requires this generation ID to match before using audit exclusions.
                _write_identity_audit(
                    identity_audit_path,
                    identity_audit_rows,
                    sync_generation_id=sync_generation_id,
                )
            except Exception as audit_exc:
                audit_write_error = type(audit_exc).__name__
                audit_message = str(audit_exc).strip()
                if audit_message:
                    audit_write_error = f"{audit_write_error}: {audit_message}"
                logger.warning(
                    "Accepted UFC roster generation %s but could not publish its identity audit: %s",
                    sync_generation_id,
                    audit_exc,
                )
        df.attrs["identity_audit_rows"] = identity_audit_rows
        df.attrs["sync_source"] = "live"
        df.attrs["sync_fallback_used"] = False
        df.attrs["sync_complete"] = sync_complete
        df.attrs["sync_completeness_reason"] = sync_completeness_reason
        df.attrs["pages_scraped"] = pages_scraped
        df.attrs["selected_card_count"] = selected_card_count
        df.attrs["parsed_card_count"] = parsed_card_count
        df.attrs["sync_generation_id"] = sync_generation_id
        df.attrs["roster_identity_audit_pair_reused"] = (
            roster_identity_audit_pair_reused
        )
        if audit_write_error:
            df.attrs["identity_audit_write_error"] = audit_write_error
        df.attrs["retained_missing_live_rows"] = retained_missing_live_rows
        df.attrs["expired_missing_live_rows"] = expired_missing_live_rows
        df.attrs["intentionally_removed_cached_rows"] = (
            intentionally_removed_cached_rows
        )
        df.attrs["discarded_suspicious_cached_rows"] = discarded_suspicious_cached_rows
        df.attrs["restored_profile_request_failure_statuses"] = (
            restored_profile_request_failure_statuses
        )
        df.attrs["preserved_immutable_profile_conflicts"] = (
            preserved_immutable_profile_conflicts
        )
        df.attrs["quarantined_unbound_simultaneous_alternate_routes"] = (
            quarantined_unbound_alternate_routes
        )
        df.attrs["reconciled_simultaneous_inactive_transitions"] = (
            reconciled_simultaneous_inactive_transitions
        )
        df.attrs[_RECONCILED_SIMULTANEOUS_ACTIVE_OWNER_TRANSITIONS_ATTR] = (
            reconciled_simultaneous_active_owner_transitions
        )
        df.attrs["reconciled_inherited_duplicate_owner_transitions"] = (
            reconciled_inherited_duplicate_owner_transitions
        )
        logger.info(
            "Saved official UFC active roster with %d rows to %s",
            len(df),
            output_path,
            extra={
                "alert_recovered_incident_keys": [_ROSTER_LIVE_SYNC_INCIDENT_KEY]
            },
        )
        if retained_missing_live_rows:
            preview = ", ".join(
                row.get("official_name", "")
                for row in retained_missing_live_rows[:10]
                if row.get("official_name")
            )
            logger.info(
                "Retained %d previously tracked UFC active-roster row(s) missing from live sync%s",
                len(retained_missing_live_rows),
                f": {preview}" if preview else "",
            )
        if not sync_complete:
            logger.warning(
                "UFC active-roster listing was incomplete (%s; pages=%d); retained-row "
                "missing counters were not advanced",
                sync_completeness_reason,
                pages_scraped,
            )
        if expired_missing_live_rows:
            preview = ", ".join(
                str(row.get("official_name") or "")
                for row in expired_missing_live_rows[:10]
                if row.get("official_name")
            )
            logger.info(
                "Expired %d UFC active-roster row(s) after %d consecutive complete live-sync misses%s",
                len(expired_missing_live_rows),
                _RETAINED_ROSTER_MAX_CONSECUTIVE_MISSING_SYNCS,
                f": {preview}" if preview else "",
            )
        return df
    except Exception as exc:
        if not allow_cached_fallback or not output_path.exists():
            raise
        try:
            cached_df = pd.read_csv(output_path)
            _validate_cached_roster_snapshot(cached_df)
        except Exception as cached_exc:
            raise RuntimeError(
                f"Official UFC roster sync failed ({exc}) and cached roster fallback at {output_path} "
                f"could not be safely loaded: {cached_exc}"
            ) from exc

        cached_df = _apply_profile_status_eligibility(cached_df)
        (
            cached_df,
            fallback_inactive_removals,
            fallback_inactive_diagnostics,
        ) = _reconcile_verified_inactive_transitions_on_cached_fallback(
            cached_df,
            verified_inactive_profile_observations,
        )
        intentionally_removed_cached_rows = list(fallback_inactive_removals)

        cached_mtime_iso = datetime.fromtimestamp(
            output_path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()
        cached_df.attrs["sync_source"] = "cached"
        cached_df.attrs["sync_fallback_used"] = True
        cached_df.attrs["sync_error"] = str(exc)
        cached_df.attrs["sync_cached_snapshot_mtime_utc"] = cached_mtime_iso
        cached_df.attrs["sync_complete"] = False
        cached_df.attrs["sync_completeness_reason"] = "cached_fallback_after_live_sync_error"
        cached_df.attrs["identity_audit_rows"] = identity_audit_rows
        cached_df.attrs["intentionally_removed_cached_rows"] = (
            intentionally_removed_cached_rows
        )
        cached_df.attrs["inactive_transition_fallback_diagnostics"] = (
            fallback_inactive_diagnostics
        )
        cached_df.attrs["retained_missing_live_rows"] = retained_roster_diagnostics(cached_df)
        cached_df.attrs["restored_profile_request_failure_statuses"] = (
            restored_profile_request_failure_statuses
        )
        cached_df.attrs["legacy_accepted_profile_request_failure_preserved_rows"] = (
            legacy_request_failure_preserved_rows
        )
        cached_df.attrs["preserved_immutable_profile_conflicts"] = (
            preserved_immutable_profile_conflicts
        )
        cached_df.attrs["quarantined_unbound_simultaneous_alternate_routes"] = (
            quarantined_unbound_alternate_routes
        )
        cached_df.attrs[_RECONCILED_SIMULTANEOUS_ACTIVE_OWNER_TRANSITIONS_ATTR] = (
            reconciled_simultaneous_active_owner_transitions
        )
        cached_df.attrs["reconciled_inherited_duplicate_owner_transitions"] = (
            reconciled_inherited_duplicate_owner_transitions
        )
        logger.warning(
            "Official UFC roster sync failed; reusing cached roster snapshot from %s at %s: %s",
            output_path,
            cached_mtime_iso,
            exc,
            extra={"alert_incident_key": _ROSTER_LIVE_SYNC_INCIDENT_KEY},
        )
        return cached_df
