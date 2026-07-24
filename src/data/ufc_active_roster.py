"""Official UFC active-roster scraping helpers."""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import RAW_DATA_DIR
from src.data.io_utils import write_csv_atomically
from src.data.name_utils import normalize_cross_source_name, same_person_name

logger = logging.getLogger(__name__)

_ROSTER_LIVE_SYNC_INCIDENT_KEY = "ufc-active-roster:live-sync-failed"
ACTIVE_ROSTER_SYNC_GENERATION_COLUMN = "active_roster_sync_generation_id"

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
_CACHED_PROFILE_BIOGRAPHY_FIELDS_COLUMN = (
    "official_profile_cached_biography_fields"
)
_CACHED_PROFILE_BIOGRAPHY_REASON_COLUMN = (
    "official_profile_biography_restoration_reason"
)
_CACHED_PROFILE_BIOGRAPHY_REASON = (
    "cached_stable_biography_after_profile_request_failure"
)
_RETAINED_ROSTER_MAX_CONSECUTIVE_MISSING_SYNCS = 3
_ROSTER_GROWTH_FALLBACK_MIN_CACHED_ROWS = 100
_ROSTER_GROWTH_FALLBACK_RATIO = 1.35
_ROSTER_GROWTH_FALLBACK_ABSOLUTE_ROWS = 250
_ROSTER_SHRINK_INCOMPLETE_MIN_CACHED_ROWS = 100
_ROSTER_SHRINK_INCOMPLETE_RATIO = 0.95
_ROSTER_SHRINK_INCOMPLETE_ABSOLUTE_ROWS = 50
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
            return BeautifulSoup(response.text, "lxml")
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


def _official_url_identity_trusted(row: dict[str, object]) -> bool:
    """Return whether slug/profile aliases from UFC.com are safe identity aliases."""
    status = _identity_status_key(row.get("official_url_identity_status"))
    explicit = row.get("official_url_identity_valid")
    if status in {"mismatch", "test profile"}:
        return False
    if explicit is None or (isinstance(explicit, float) and pd.isna(explicit)):
        return True
    return not _falsey(explicit)


def _official_url_identity_positively_trusted(
    row: pd.Series | dict[str, object],
) -> bool:
    """Require positive identity evidence before destructively removing a row."""
    status = _identity_status_key(row.get("official_url_identity_status"))
    explicit = row.get("official_url_identity_valid")
    if status in {"mismatch", "test profile"} or _falsey(explicit):
        return False
    return status in {"valid", "slug mismatch profile valid"} or _truthy(explicit)


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
        "ufcstats_url": _clean_text(row.get("ufcstats_url")),
        "profile_name": _clean_text(row.get("profile_name")),
        "slug_name": _clean_text(row.get("slug_name")),
        "alternate_slug_names": _clean_text(row.get("alternate_slug_names")),
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
    if same_person_name(left, right):
        return True
    left_key = normalize_cross_source_name(left).replace(" ", "")
    right_key = normalize_cross_source_name(right).replace(" ", "")
    return bool(left_key and right_key and left_key == right_key)


def _write_identity_audit(
    path: Path | None,
    rows: list[dict[str, object]],
    *,
    sync_generation_id: str,
) -> None:
    if path is None:
        return
    stamped_rows = [
        {**row, ACTIVE_ROSTER_SYNC_GENERATION_COLUMN: sync_generation_id}
        for row in rows
    ]
    audit_df = pd.DataFrame(stamped_rows, columns=IDENTITY_AUDIT_COLUMNS)
    write_csv_atomically(audit_df, path)


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
    if left_url and right_url and left_url == right_url:
        return True

    left_ufcstats_url = _normalize_roster_url(left.get("ufcstats_url"))
    right_ufcstats_url = _normalize_roster_url(right.get("ufcstats_url"))
    if left_ufcstats_url and right_ufcstats_url:
        return left_ufcstats_url == right_ufcstats_url

    if left_url and right_url:
        return False

    left_name_keys = _roster_identity_name_keys(left)
    right_name_keys = _roster_identity_name_keys(right)
    if not (left_name_keys & right_name_keys):
        return False

    left_official = _clean_text(left.get("official_name"))
    right_official = _clean_text(right.get("official_name"))
    if left_official and right_official and not _same_identity_name(left_official, right_official):
        return False

    return True


def _restore_cached_current_status_after_profile_request_failures(
    live_df: pd.DataFrame,
    cached_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Restore narrowly approved cached profile data after access was unavailable.

    A successfully fetched UFC profile that simply has no status tag is not
    equivalent to a transport failure or an HTTP-200 challenge/non-profile page
    and must remain quarantined. Likewise, ambiguous cached identities and
    cached rows without positive current-MMA identity evidence are not safe
    sources. Only immutable/slow-changing measurements from the UFC biography
    schema are eligible; records, age, division, and aliases remain live-only.
    """
    if (
        live_df.empty
        or cached_df.empty
        or _PROFILE_FETCH_OUTCOME_COLUMN not in live_df.columns
    ):
        return live_df, 0

    live = live_df.copy()
    restored_rows = 0
    cached_rows = [row for _, row in cached_df.iterrows()]
    for index, live_row in live.iterrows():
        outcome = _optional_text(live_row.get(_PROFILE_FETCH_OUTCOME_COLUMN)).casefold()
        if outcome not in {
            _PROFILE_FETCH_REQUEST_FAILED,
            _PROFILE_FETCH_INVALID_RESPONSE,
        }:
            continue
        if (
            _official_url_identity_explicitly_untrusted(live_row)
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
            if _roster_rows_same_identity(live_row, cached_row)
        ]
        if len(matches) != 1:
            continue

        cached_row = matches[0]
        if (
            not _row_is_positively_verified_current_mma(cached_row)
            or _explicit_boolean(cached_row.get("coverage_eligible")) is not True
        ):
            continue

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

        restored_fields: list[str] = []
        for field in _CACHED_STABLE_PROFILE_BIOGRAPHY_FIELDS:
            if _optional_text(live_row.get(field)):
                continue
            cached_value = cached_row.get(field)
            if not _optional_text(cached_value):
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

    return live, restored_rows


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
            if _roster_rows_same_identity(live_row, cached_row)
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
    excluded_rows: list[dict[str, object]] | None = None,
    sync_complete: bool = False,
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
        if any(_roster_rows_same_identity(cached_row, live_row) for live_row in live_rows):
            continue
        if any(
            _roster_rows_same_identity(cached_row, excluded_row)
            for excluded_row in excluded_rows
        ):
            continue
        if _verified_inactive_profile_status(cached_row):
            continue

        was_retained = _truthy(cached_row.get("active_roster_retained_from_previous"))
        try:
            previous_missing_syncs = int(float(cached_row.get("active_roster_consecutive_missing_syncs") or 0))
        except (TypeError, ValueError):
            previous_missing_syncs = 0
        if sync_complete:
            consecutive_missing_syncs = previous_missing_syncs + 1 if was_retained else 1
        else:
            consecutive_missing_syncs = previous_missing_syncs
        first_missing_at_utc = (
            _optional_text(cached_row.get("active_roster_first_missing_at_utc"))
            or _optional_text(cached_row.get("active_roster_retained_at_utc"))
            or (retained_at_utc if sync_complete else "")
        )
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
    retained["active_roster_retained_at_utc"] = retained["active_roster_first_missing_at_utc"]
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
                _roster_rows_same_identity(cached_row, excluded_row)
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
            _roster_rows_same_identity(cached_row, live_row)
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


def _roster_page_has_next(soup: BeautifulSoup) -> bool:
    """Return whether UFC.com's listing explicitly advertises another page."""
    return soup.select_one(
        'a[rel~="next"][href], li.pager__item--next a[href]'
    ) is not None


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
    classify_combat_sport: bool = True,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    client = session or requests.Session()
    rows: list[dict[str, object]] = []
    identity_audit_rows: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    page = 0
    pages_scraped = 0
    selected_card_count = 0
    parsed_card_count = 0
    scan_parser_complete = True
    sync_complete = False
    sync_completeness_reason = "not_started"
    candidates = _load_local_ufcstats_candidates() if resolve_ufcstats else {}

    while True:
        if max_pages is not None and page >= max_pages:
            sync_completeness_reason = "max_pages_reached_before_terminal_page"
            break
        page_url = OFFICIAL_ACTIVE_ROSTER_URL if page == 0 else f"{OFFICIAL_ACTIVE_ROSTER_URL}&page={page}"
        soup = _get_soup(page_url, session=client)
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
            if not athlete_url_key or athlete_url_key in seen_urls:
                continue
            seen_urls.add(athlete_url_key)
            page_new_identities += 1
            if fetch_profile_details:
                try:
                    parsed.update(scrape_official_athlete_profile(athlete_url, session=client))
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
            if classify_combat_sport:
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
            if scan_parser_complete:
                sync_completeness_reason = "pagination_made_no_progress"
            break
        # A missing/changed next-link selector is not positive terminal
        # evidence. Probe the following page; UFC's real terminal response is a
        # normal athletes page with an explicit "No Result Found" view.
        page += 1

    if not rows:
        df = pd.DataFrame()
        df.attrs["identity_audit_rows"] = identity_audit_rows
        df.attrs["sync_complete"] = sync_complete
        df.attrs["sync_completeness_reason"] = sync_completeness_reason
        df.attrs["pages_scraped"] = pages_scraped
        df.attrs["selected_card_count"] = selected_card_count
        df.attrs["parsed_card_count"] = parsed_card_count
        return df

    df = pd.DataFrame(rows)
    df = df.sort_values(["official_name", "official_athlete_url"]).reset_index(drop=True)
    df.attrs["identity_audit_rows"] = identity_audit_rows
    df.attrs["sync_complete"] = sync_complete
    df.attrs["sync_completeness_reason"] = sync_completeness_reason
    df.attrs["pages_scraped"] = pages_scraped
    df.attrs["selected_card_count"] = selected_card_count
    df.attrs["parsed_card_count"] = parsed_card_count
    unknown_status_rows = sum(
        1
        for row in identity_audit_rows
        if str(row.get("action") or "").strip() == "quarantined_unknown_profile_status"
    )
    if unknown_status_rows:
        logger.warning(
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
    try:
        df = scrape_official_active_roster(
            fetch_profile_details=fetch_profile_details,
            max_pages=max_pages,
            resolve_ufcstats=resolve_ufcstats,
            classify_combat_sport=classify_combat_sport,
        )
        identity_audit_rows = list(df.attrs.get("identity_audit_rows") or [])
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
        discarded_suspicious_cached_rows = 0
        restored_profile_request_failure_statuses = 0
        if output_path.exists():
            try:
                cached_df = pd.read_csv(output_path)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load existing cached UFC active-roster rows; "
                    "refusing a guardless live overwrite"
                ) from exc
            _validate_cached_roster_snapshot(cached_df)
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
            df = _apply_profile_status_eligibility(df)

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
                    retained_at_utc=datetime.now(timezone.utc).isoformat(),
                    excluded_rows=intentionally_excluded_rows,
                    sync_complete=sync_complete,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to preserve cached UFC active-roster rows missing from live sync"
                ) from exc
        sync_generation_id = uuid.uuid4().hex
        df[ACTIVE_ROSTER_SYNC_GENERATION_COLUMN] = sync_generation_id
        write_csv_atomically(df, output_path, refuse_empty=True)
        audit_write_error = ""
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
            audit_write_error = str(audit_exc)
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
        if audit_write_error:
            df.attrs["identity_audit_write_error"] = audit_write_error
        df.attrs["retained_missing_live_rows"] = retained_missing_live_rows
        df.attrs["expired_missing_live_rows"] = expired_missing_live_rows
        df.attrs["discarded_suspicious_cached_rows"] = discarded_suspicious_cached_rows
        df.attrs["restored_profile_request_failure_statuses"] = (
            restored_profile_request_failure_statuses
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

        cached_mtime_iso = datetime.fromtimestamp(
            output_path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()
        cached_df.attrs["sync_source"] = "cached"
        cached_df.attrs["sync_fallback_used"] = True
        cached_df.attrs["sync_error"] = str(exc)
        cached_df.attrs["sync_cached_snapshot_mtime_utc"] = cached_mtime_iso
        cached_df.attrs["retained_missing_live_rows"] = retained_roster_diagnostics(cached_df)
        logger.warning(
            "Official UFC roster sync failed; reusing cached roster snapshot from %s at %s: %s",
            output_path,
            cached_mtime_iso,
            exc,
            extra={"alert_incident_key": _ROSTER_LIVE_SYNC_INCIDENT_KEY},
        )
        return cached_df
