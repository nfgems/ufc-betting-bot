"""Fail-closed quality checks for staged UFC data generations.

The checks in this module are deliberately side-effect free.  A refresh writes
to an isolated candidate directory, calls these helpers, and only publishes the
candidate when every blocking check passes.  Missing-value handling is shared
here so a numeric zero is never mistaken for an absent source value.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from numbers import Number
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from src.data.ufc_active_roster import (
    ACCEPTED_IMMUTABLE_ROSTER_PROFILE_FIELDS,
    ACTIVE_ROSTER_CACHED_BIOGRAPHY_PROVENANCE_COLUMNS,
    _roster_identity_keys,
    _roster_rows_same_identity,
    _same_identity_name,
)
from src.data.fight_results import (
    RESULT_CLASS_DQ,
    RESULT_CLASS_DRAW,
    RESULT_CLASS_NO_CONTEST,
    RESULT_CLASS_NORMAL,
    RESULT_POLICY_VERSION,
    parse_policy_boolean,
    validate_marked_result,
)
from src.data.name_utils import (
    canonical_fighter_name_key,
    normalize_person_name,
    normalize_ufcstats_id,
)
from src.data.profile_integrity import profile_publication_integrity_report
from src.data.reviewed_roster_recovery import (
    reviewed_official_identity_transition_match,
)


RESULT_POLICY_COLUMNS = (
    "fighter_a_id",
    "fighter_b_id",
    "RESULT_POLICY_VERSION",
    "RESULT_CLASS",
    "BINARY_TARGET_ELIGIBLE",
    "RESULT_QUARANTINE_REASON",
)
PROFILE_PHYSICAL_FIELDS = ("dob", "height", "weight", "reach", "stance")
PROFILE_STRIKING_FIELDS = ("slpm", "sapm", "str_acc", "str_def")
PROFILE_GRAPPLING_FIELDS = ("td_avg", "td_acc", "td_def", "sub_avg")
PROFILE_OBSERVED_FIELDS = (
    *PROFILE_PHYSICAL_FIELDS,
    *PROFILE_STRIKING_FIELDS,
    *PROFILE_GRAPPLING_FIELDS,
)
PROCESSED_RESULT_POLICY_COLUMNS = (
    "official_outcome",
    "result_policy_version",
    "result_class",
    "binary_target_eligible",
    "result_quarantine_reason",
    "fighter_a_id",
    "fighter_b_id",
)
COVERAGE_FLOOR_ENV_TO_METRIC = {
    "UFC_REFRESH_MIN_OVERALL_REACH_PCT": "overall_reach_pct",
    "UFC_REFRESH_MIN_OVERALL_STANCE_PCT": "overall_stance_pct",
    "UFC_REFRESH_MIN_NEW_FIGHTER_REACH_PCT": "new_fighter_reach_pct",
    "UFC_REFRESH_MIN_NEW_FIGHTER_STANCE_PCT": "new_fighter_stance_pct",
    "UFC_REFRESH_MIN_NEW_FIGHTER_FULL_PHYSICAL_PCT": "new_fighter_full_physical_pct",
}

RESULT_BASE_COLUMNS = (
    "EVENT",
    "BOUT",
    "OUTCOME",
    "WEIGHTCLASS",
    "METHOD",
    "ROUND",
    "TIME",
    "TIME FORMAT",
    "REFEREE",
    "DETAILS",
    "URL",
)
STAT_COLUMNS = (
    "EVENT",
    "BOUT",
    "ROUND",
    "FIGHTER",
    "KD",
    "SIG.STR.",
    "SIG.STR. %",
    "TOTAL STR.",
    "TD",
    "TD %",
    "SUB.ATT",
    "REV.",
    "CTRL",
    "HEAD",
    "BODY",
    "LEG",
    "DISTANCE",
    "CLINCH",
    "GROUND",
)
_PAIR_STAT_COLUMNS = (
    "SIG.STR.",
    "TOTAL STR.",
    "TD",
    "HEAD",
    "BODY",
    "LEG",
    "DISTANCE",
    "CLINCH",
    "GROUND",
)
_COUNT_STAT_COLUMNS = ("KD", "SUB.ATT", "REV.")
_PERCENT_STAT_COLUMNS = ("SIG.STR. %", "TD %")
_MISSING_TEXT = frozenset({"", "--", "---", "n/a", "na", "nan", "none", "null"})
_BOUT_SPLIT_RE = re.compile(r"\s+(?:vs\.?|v\.?)\s+", flags=re.IGNORECASE)
_PAIR_RE = re.compile(
    r"^\s*(?P<landed>\d+(?:\.\d+)?)\s*(?:of|/)\s*(?P<attempted>\d+(?:\.\d+)?)\s*$",
    flags=re.IGNORECASE,
)
_CLOCK_RE = re.compile(r"^\s*(?P<minutes>\d+):(?P<seconds>\d{2})\s*$")


def value_is_missing(value: object) -> bool:
    """Return whether a source value is absent while treating zero as observed."""

    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return True
    if not isinstance(value, str):
        return False
    return value.strip().casefold() in _MISSING_TEXT


def _text(value: object) -> str:
    return "" if value_is_missing(value) else " ".join(str(value).strip().split())


def _normalized(value: object) -> str:
    return normalize_person_name(_text(value))


def _fight_url_id(value: object) -> str:
    text = _text(value)
    if not text:
        return ""
    parsed = urlparse(text)
    segment = parsed.path.rstrip("/").split("/")[-1]
    return segment.casefold() if segment else ""


def _bout_participants(value: object) -> tuple[str, str] | None:
    parts = [_normalized(part) for part in _BOUT_SPLIT_RE.split(_text(value))]
    if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
        return None
    return tuple(sorted(parts))


def stable_raw_fight_key(row: Mapping[str, object]) -> str | None:
    """Build a fight identity that is invariant to displayed A/B order."""

    url_id = _fight_url_id(row.get("URL") or row.get("fight_url"))
    if url_id:
        return f"ufcstats:{url_id}"
    event = _normalized(row.get("EVENT") or row.get("event_name"))
    participants = _bout_participants(row.get("BOUT"))
    if not participants:
        names = sorted(
            filter(
                None,
                (
                    _normalized(row.get("fighter_a")),
                    _normalized(row.get("fighter_b")),
                ),
            )
        )
        participants = tuple(names) if len(names) == 2 and names[0] != names[1] else None
    if not event or not participants:
        return None
    return f"event:{event}|fighters:{participants[0]}|{participants[1]}"


def _processed_event_date(row: Mapping[str, object]) -> str | None:
    event_date = pd.to_datetime(row.get("event_date"), errors="coerce")
    if pd.isna(event_date):
        return None
    return event_date.date().isoformat()


def stable_processed_name_date_key(row: Mapping[str, object]) -> str | None:
    """Return the legacy-compatible processed identity using names and date."""

    event_date = _processed_event_date(row)
    if event_date is None:
        return None
    names = sorted(
        filter(
            None,
            (
                _normalized(row.get("fighter_a")),
                _normalized(row.get("fighter_b")),
            ),
        )
    )
    if len(names) != 2 or names[0] == names[1]:
        return None
    return f"date:{event_date}|fighters:name:{names[0]}|name:{names[1]}"


def stable_processed_fight_key(row: Mapping[str, object]) -> str | None:
    """Build an order-independent identity for a normalized processed row."""

    event_date = _processed_event_date(row)
    if event_date is None:
        return None
    fighter_ids = sorted(
        value
        for value in (
            normalize_ufcstats_id(row.get("fighter_a_id")),
            normalize_ufcstats_id(row.get("fighter_b_id")),
        )
        if value
    )
    if len(fighter_ids) == 2 and fighter_ids[0] != fighter_ids[1]:
        participants = (f"id:{fighter_ids[0]}", f"id:{fighter_ids[1]}")
    else:
        return stable_processed_name_date_key(row)
    return f"date:{event_date}|fighters:{participants[0]}|{participants[1]}"


def stable_raw_stat_key(row: Mapping[str, object]) -> str | None:
    fight_row = dict(row)
    fight_row["URL"] = ""
    fight_key = stable_raw_fight_key(fight_row)
    fighter = _normalized(row.get("FIGHTER"))
    round_text = _text(row.get("ROUND")).casefold().replace("round", "").strip()
    try:
        round_number = int(float(round_text))
    except (TypeError, ValueError):
        return None
    if not fight_key or not fighter or round_number < 1:
        return None
    return f"{fight_key}|round:{round_number}|fighter:{fighter}"


def stable_profile_key(row: Mapping[str, object]) -> str | None:
    for column in ("fighter_id", "fighter_url", "ufcstats_url"):
        value = row.get(column)
        fighter_id = normalize_ufcstats_id(value)
        if not fighter_id and column != "fighter_id":
            fighter_id = _fight_url_id(value)
        if fighter_id:
            return f"ufcstats:{fighter_id}"
    name = _normalized(row.get("name") or row.get("official_name") or row.get("ufcstats_name"))
    return f"name:{name}" if name else None


def _records(frame: pd.DataFrame) -> Iterable[dict[str, object]]:
    return frame.to_dict(orient="records")


def duplicate_key_report(
    frame: pd.DataFrame,
    key_builder,
    *,
    label: str,
) -> dict[str, object]:
    keys = [key_builder(row) for row in _records(frame)]
    invalid_rows = sum(key is None for key in keys)
    counts = Counter(key for key in keys if key is not None)
    duplicates = {str(key): count for key, count in counts.items() if count > 1}
    errors: list[str] = []
    if invalid_rows:
        errors.append(f"{label}_invalid_identity:{invalid_rows}")
    if duplicates:
        errors.append(f"{label}_duplicate_identity:{len(duplicates)}")
    return {
        "passed": not errors,
        "errors": errors,
        "invalid_identity_rows": invalid_rows,
        "duplicate_identity_count": len(duplicates),
        "duplicates": dict(list(duplicates.items())[:25]),
    }


def _signature_value(value: object) -> tuple[str, object]:
    if value_is_missing(value):
        return ("missing", "")
    if isinstance(value, bool):
        return ("boolean", value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ("text", _text(value).casefold())
    if math.isfinite(numeric):
        return ("numeric", numeric)
    return ("text", _text(value).casefold())


def _row_signature(
    row: Mapping[str, object],
    columns: Sequence[str],
) -> tuple[tuple[str, tuple[str, object]], ...]:
    return tuple((column, _signature_value(row.get(column))) for column in columns)


def baseline_aware_duplicate_key_report(
    accepted: pd.DataFrame,
    candidate: pd.DataFrame,
    key_builder,
    *,
    label: str,
) -> dict[str, object]:
    """Grandfather unchanged identity anomalies and reject every delta.

    Invalid legacy rows and legacy duplicate groups are compared by their
    values across the accepted schema.  This prevents an apparent "improvement"
    from silently removing or rewriting accepted history while allowing the
    untouched baseline to remain publishable.
    """

    accepted_rows = list(_records(accepted))
    candidate_rows = list(_records(candidate))
    columns = [column for column in accepted.columns if column in candidate.columns]
    accepted_keys = [key_builder(row) for row in accepted_rows]
    candidate_keys = [key_builder(row) for row in candidate_rows]

    accepted_invalid = Counter(
        _row_signature(row, columns)
        for row, key in zip(accepted_rows, accepted_keys)
        if key is None
    )
    candidate_invalid = Counter(
        _row_signature(row, columns)
        for row, key in zip(candidate_rows, candidate_keys)
        if key is None
    )
    new_invalid = candidate_invalid - accepted_invalid
    missing_invalid = accepted_invalid - candidate_invalid

    accepted_groups: dict[str, list[dict[str, object]]] = {}
    candidate_groups: dict[str, list[dict[str, object]]] = {}
    for row, key in zip(accepted_rows, accepted_keys):
        if key is not None:
            accepted_groups.setdefault(str(key), []).append(row)
    for row, key in zip(candidate_rows, candidate_keys):
        if key is not None:
            candidate_groups.setdefault(str(key), []).append(row)

    new_duplicate_units = 0
    changed_or_removed_legacy_duplicates = 0
    grandfathered_duplicate_keys: list[str] = []
    candidate_duplicate_details: dict[str, int] = {}
    all_keys = set(accepted_groups) | set(candidate_groups)
    for key in sorted(all_keys):
        accepted_rows_for_key = accepted_groups.get(key, [])
        candidate_rows_for_key = candidate_groups.get(key, [])
        accepted_units = max(0, len(accepted_rows_for_key) - 1)
        candidate_units = max(0, len(candidate_rows_for_key) - 1)
        if candidate_units:
            candidate_duplicate_details[key] = len(candidate_rows_for_key)
        new_duplicate_units += max(0, candidate_units - accepted_units)
        if not accepted_units:
            continue
        accepted_signatures = Counter(
            _row_signature(row, columns) for row in accepted_rows_for_key
        )
        candidate_signatures = Counter(
            _row_signature(row, columns) for row in candidate_rows_for_key
        )
        missing_signatures = accepted_signatures - candidate_signatures
        if candidate_units < accepted_units or missing_signatures:
            changed_or_removed_legacy_duplicates += max(
                accepted_units - candidate_units,
                sum(missing_signatures.values()),
                1,
            )
        else:
            grandfathered_duplicate_keys.append(key)

    errors: list[str] = []
    if new_invalid:
        errors.append(f"{label}_invalid_identity:{sum(new_invalid.values())}")
    if missing_invalid:
        errors.append(
            f"{label}_historical_invalid_identity_changed_or_removed:"
            f"{sum(missing_invalid.values())}"
        )
    if new_duplicate_units:
        errors.append(f"{label}_duplicate_identity:{new_duplicate_units}")
    if changed_or_removed_legacy_duplicates:
        errors.append(
            f"{label}_historical_duplicate_changed_or_removed:"
            f"{changed_or_removed_legacy_duplicates}"
        )
    grandfathered_invalid = accepted_invalid & candidate_invalid
    notes: list[dict[str, object]] = []
    if grandfathered_invalid:
        notes.append(
            {
                "reason": "grandfathered_legacy_invalid_identities",
                "count": sum(grandfathered_invalid.values()),
            }
        )
    if grandfathered_duplicate_keys:
        notes.append(
            {
                "reason": "grandfathered_legacy_duplicate_identities",
                "count": len(grandfathered_duplicate_keys),
                "identities": grandfathered_duplicate_keys[:25],
            }
        )
    return {
        "passed": not errors,
        "errors": errors,
        "notes": notes,
        "new_invalid_identity_count": sum(new_invalid.values()),
        "grandfathered_invalid_identity_count": sum(grandfathered_invalid.values()),
        "new_duplicate_identity_count": new_duplicate_units,
        "grandfathered_duplicate_identity_count": len(
            grandfathered_duplicate_keys
        ),
        "duplicates": dict(list(candidate_duplicate_details.items())[:25]),
    }


def roster_identity_report(
    frame: pd.DataFrame,
    *,
    label: str = "active_roster",
    accepted: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Reject invalid, duplicate, and one-to-many roster identities.

    The active-roster collector already has conflict-aware identity semantics.
    Reusing them here avoids treating two same-name fighters with conflicting
    strong URLs as duplicates. A separate strong-carrier scan is intentional:
    it catches one UFCStats ID bound to multiple UFC entities and the inverse,
    even though conflict-aware equality correctly says those rows differ.
    """

    rows = list(_records(frame))
    keys_by_row = [_roster_identity_keys(row) for row in rows]
    invalid_rows = [index for index, keys in enumerate(keys_by_row) if not keys]
    rows_by_key: dict[str, list[int]] = {}
    for index, keys in enumerate(keys_by_row):
        for key in keys:
            rows_by_key.setdefault(key, []).append(index)

    def shares_strong_carrier(left_keys: set[str], right_keys: set[str]) -> bool:
        return any(
            key.startswith(("ufcstats_url:", "url:"))
            for key in left_keys & right_keys
        )

    duplicate_pairs: set[tuple[int, int]] = set()
    for positions in rows_by_key.values():
        for left, right in combinations(sorted(set(positions)), 2):
            if _roster_rows_same_identity(
                rows[left], rows[right]
            ) or shares_strong_carrier(keys_by_row[left], keys_by_row[right]):
                duplicate_pairs.add((left, right))

    def strong_carrier_conflict_pairs(
        source_rows: list[dict[str, object]],
        source_keys: list[set[str]],
    ) -> set[tuple[int, int]]:
        """Catch one-to-many strong carriers independently of identity matching."""

        conflicts: set[tuple[int, int]] = set()
        for left, right in combinations(range(len(source_rows)), 2):
            left_keys = source_keys[left]
            right_keys = source_keys[right]
            left_ufcstats = {
                key for key in left_keys if key.startswith("ufcstats_url:")
            }
            right_ufcstats = {
                key for key in right_keys if key.startswith("ufcstats_url:")
            }
            left_official = {
                key for key in left_keys if key.startswith("url:")
            }
            right_official = {
                key for key in right_keys if key.startswith("url:")
            }
            shared_ufcstats = left_ufcstats & right_ufcstats
            shared_official = left_official & right_official
            if not (shared_ufcstats or shared_official):
                continue

            left_name = _text(source_rows[left].get("official_name"))
            right_name = _text(source_rows[right].get("official_name"))
            names_conflict = bool(
                left_name
                and right_name
                and not _same_identity_name(left_name, right_name)
            )
            official_urls_conflict = bool(
                left_official and right_official and left_official != right_official
            )
            ufcstats_ids_conflict = bool(
                left_ufcstats and right_ufcstats and left_ufcstats != right_ufcstats
            )
            if (
                shared_ufcstats and (official_urls_conflict or names_conflict)
            ) or (
                shared_official and (ufcstats_ids_conflict or names_conflict)
            ):
                conflicts.add((left, right))
        return conflicts

    strong_conflict_pairs = strong_carrier_conflict_pairs(rows, keys_by_row)

    def strongest_shared_identity(
        left_keys: set[str],
        right_keys: set[str],
    ) -> str:
        shared = left_keys & right_keys
        for prefix in ("ufcstats_url:", "url:", "name:"):
            matches = sorted(key for key in shared if key.startswith(prefix))
            if matches:
                return matches[0]
        # Conflict-aware roster matching requires a shared identity in normal
        # operation. Keep an explicit fail-closed bucket if that invariant ever
        # changes instead of falling back to mutable row bytes.
        return "unresolved_shared_roster_identity"

    details = [
        {
            "rows": [left, right],
            "shared_identity_keys": sorted(keys_by_row[left] & keys_by_row[right]),
            "comparison_identity": strongest_shared_identity(
                keys_by_row[left], keys_by_row[right]
            ),
        }
        for left, right in sorted(duplicate_pairs)[:25]
    ]
    strong_conflict_details = [
        {
            "rows": [left, right],
            "shared_strong_carrier_keys": sorted(
                key
                for key in keys_by_row[left] & keys_by_row[right]
                if key.startswith(("ufcstats_url:", "url:"))
            ),
            "left_name": _text(rows[left].get("official_name")),
            "right_name": _text(rows[right].get("official_name")),
            "left_official_url": _text(
                rows[left].get("official_athlete_url")
            ),
            "right_official_url": _text(
                rows[right].get("official_athlete_url")
            ),
            "left_ufcstats_url": _text(rows[left].get("ufcstats_url")),
            "right_ufcstats_url": _text(rows[right].get("ufcstats_url")),
        }
        for left, right in sorted(strong_conflict_pairs)[:25]
    ]
    if accepted is not None:
        accepted_rows = list(_records(accepted))
        accepted_keys_by_row = [
            _roster_identity_keys(row) for row in accepted_rows
        ]
        columns = [column for column in accepted.columns if column in frame.columns]
        accepted_invalid = Counter(
            _row_signature(row, columns)
            for row, keys in zip(accepted_rows, accepted_keys_by_row)
            if not keys
        )
        candidate_invalid = Counter(
            _row_signature(row, columns)
            for row, keys in zip(rows, keys_by_row)
            if not keys
        )

        def duplicate_pair_counter(
            source_rows: list[dict[str, object]],
            source_keys: list[set[str]],
        ) -> Counter[str]:
            by_key: dict[str, list[int]] = {}
            for position, keys in enumerate(source_keys):
                for key in keys:
                    by_key.setdefault(key, []).append(position)
            pairs: set[tuple[int, int]] = set()
            for positions in by_key.values():
                for left, right in combinations(sorted(set(positions)), 2):
                    if _roster_rows_same_identity(
                        source_rows[left], source_rows[right]
                    ) or shares_strong_carrier(
                        source_keys[left], source_keys[right]
                    ):
                        pairs.add((left, right))
            return Counter(
                strongest_shared_identity(
                    source_keys[left],
                    source_keys[right],
                )
                for left, right in pairs
            )

        def strong_conflict_counter(
            source_rows: list[dict[str, object]],
            source_keys: list[set[str]],
        ) -> Counter[str]:
            pairs = strong_carrier_conflict_pairs(source_rows, source_keys)

            def row_identity(position: int) -> str:
                strong_keys = sorted(
                    key
                    for key in source_keys[position]
                    if key.startswith(("ufcstats_url:", "url:"))
                )
                name = normalize_person_name(
                    source_rows[position].get("official_name")
                )
                return "|".join([*strong_keys, f"name:{name}"])

            return Counter(
                "<->".join(sorted((row_identity(left), row_identity(right))))
                for left, right in pairs
            )

        accepted_duplicate_pairs = duplicate_pair_counter(
            accepted_rows, accepted_keys_by_row
        )
        candidate_duplicate_pairs = duplicate_pair_counter(rows, keys_by_row)
        accepted_strong_conflicts = strong_conflict_counter(
            accepted_rows,
            accepted_keys_by_row,
        )
        candidate_strong_conflicts = strong_conflict_counter(rows, keys_by_row)
        new_invalid = candidate_invalid - accepted_invalid
        missing_invalid = accepted_invalid - candidate_invalid
        new_duplicate_pairs = candidate_duplicate_pairs - accepted_duplicate_pairs
        missing_duplicate_pairs = accepted_duplicate_pairs - candidate_duplicate_pairs
        grandfathered_invalid = accepted_invalid & candidate_invalid
        grandfathered_duplicates = (
            accepted_duplicate_pairs & candidate_duplicate_pairs
        )
        new_strong_conflicts = (
            candidate_strong_conflicts - accepted_strong_conflicts
        )
        missing_strong_conflicts = (
            accepted_strong_conflicts - candidate_strong_conflicts
        )
        grandfathered_strong_conflicts = (
            accepted_strong_conflicts & candidate_strong_conflicts
        )
        errors: list[str] = []
        if new_invalid:
            errors.append(f"{label}_invalid_identity:{sum(new_invalid.values())}")
        if missing_invalid:
            errors.append(
                f"{label}_historical_invalid_identity_changed_or_removed:"
                f"{sum(missing_invalid.values())}"
            )
        if new_duplicate_pairs:
            errors.append(
                f"{label}_duplicate_identity:{sum(new_duplicate_pairs.values())}"
            )
        if missing_duplicate_pairs:
            errors.append(
                f"{label}_historical_duplicate_changed_or_removed:"
                f"{sum(missing_duplicate_pairs.values())}"
            )
        if new_strong_conflicts:
            errors.append(
                f"{label}_strong_carrier_conflict:"
                f"{sum(new_strong_conflicts.values())}"
            )
        if missing_strong_conflicts:
            errors.append(
                f"{label}_historical_strong_carrier_conflict_changed_or_removed:"
                f"{sum(missing_strong_conflicts.values())}"
            )
        notes: list[dict[str, object]] = []
        if grandfathered_invalid:
            notes.append(
                {
                    "reason": "grandfathered_legacy_invalid_identities",
                    "count": sum(grandfathered_invalid.values()),
                }
            )
        if grandfathered_duplicates:
            notes.append(
                {
                    "reason": "grandfathered_legacy_duplicate_identities",
                    "count": sum(grandfathered_duplicates.values()),
                }
            )
        if grandfathered_strong_conflicts:
            notes.append(
                {
                    "reason": "grandfathered_legacy_strong_carrier_conflicts",
                    "count": sum(grandfathered_strong_conflicts.values()),
                }
            )
        return {
            "passed": not errors,
            "errors": errors,
            "notes": notes,
            "invalid_identity_rows": invalid_rows[:25],
            "invalid_identity_count": len(invalid_rows),
            "duplicate_identity_count": len(duplicate_pairs),
            "new_invalid_identity_count": sum(new_invalid.values()),
            "new_duplicate_identity_count": sum(new_duplicate_pairs.values()),
            "grandfathered_invalid_identity_count": sum(
                grandfathered_invalid.values()
            ),
            "grandfathered_duplicate_identity_count": sum(
                grandfathered_duplicates.values()
            ),
            "duplicates": details,
            "strong_carrier_conflict_count": len(strong_conflict_pairs),
            "new_strong_carrier_conflict_count": sum(
                new_strong_conflicts.values()
            ),
            "grandfathered_strong_carrier_conflict_count": sum(
                grandfathered_strong_conflicts.values()
            ),
            "strong_carrier_conflicts": strong_conflict_details,
        }

    errors: list[str] = []
    if invalid_rows:
        errors.append(f"{label}_invalid_identity:{len(invalid_rows)}")
    if duplicate_pairs:
        errors.append(f"{label}_duplicate_identity:{len(duplicate_pairs)}")
    if strong_conflict_pairs:
        errors.append(
            f"{label}_strong_carrier_conflict:{len(strong_conflict_pairs)}"
        )
    return {
        "passed": not errors,
        "errors": errors,
        "invalid_identity_rows": invalid_rows[:25],
        "invalid_identity_count": len(invalid_rows),
        "duplicate_identity_count": len(duplicate_pairs),
        "duplicates": details,
        "strong_carrier_conflict_count": len(strong_conflict_pairs),
        "strong_carrier_conflicts": strong_conflict_details,
    }


def roster_immutable_field_report(
    accepted: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    label: str = "active_roster",
) -> dict[str, object]:
    """Reject silent rewrites of accepted immutable UFC biography fields."""

    accepted_rows = list(_records(accepted))
    candidate_rows = list(_records(candidate))
    accepted_keys = [_roster_identity_keys(row) for row in accepted_rows]
    candidate_keys = [_roster_identity_keys(row) for row in candidate_rows]

    def strong_keys(keys: set[str]) -> set[str]:
        return {
            key
            for key in keys
            if key.startswith(("ufcstats_url:", "url:"))
        }

    pair_candidates: list[tuple[int, int, set[str]]] = []
    for accepted_position, before_keys in enumerate(accepted_keys):
        for candidate_position, after_keys in enumerate(candidate_keys):
            shared = strong_keys(before_keys) & strong_keys(after_keys)
            if shared:
                pair_candidates.append(
                    (accepted_position, candidate_position, shared)
                )
    accepted_match_counts = Counter(
        accepted_position
        for accepted_position, _, _ in pair_candidates
    )
    candidate_match_counts = Counter(
        candidate_position
        for _, candidate_position, _ in pair_candidates
    )
    unique_pairs = [
        (accepted_position, candidate_position, shared)
        for accepted_position, candidate_position, shared in pair_candidates
        if accepted_match_counts[accepted_position] == 1
        and candidate_match_counts[candidate_position] == 1
    ]
    compared_fields = [
        field
        for field in ACCEPTED_IMMUTABLE_ROSTER_PROFILE_FIELDS
        if field in accepted.columns and field in candidate.columns
    ]
    changes: list[dict[str, object]] = []
    binding_conflicts: list[dict[str, object]] = []
    for accepted_position, candidate_position, shared in unique_pairs:
        before = accepted_rows[accepted_position]
        after = candidate_rows[candidate_position]
        identity = sorted(
            shared,
            key=lambda key: (not key.startswith("ufcstats_url:"), key),
        )[0]
        before_strong = strong_keys(accepted_keys[accepted_position])
        after_strong = strong_keys(candidate_keys[candidate_position])
        before_ids = {
            key for key in before_strong if key.startswith("ufcstats_url:")
        }
        after_ids = {
            key for key in after_strong if key.startswith("ufcstats_url:")
        }
        before_urls = {
            key for key in before_strong if key.startswith("url:")
        }
        after_urls = {
            key for key in after_strong if key.startswith("url:")
        }
        names_conflict = bool(
            _text(before.get("official_name"))
            and _text(after.get("official_name"))
            and not _same_identity_name(
                before.get("official_name"),
                after.get("official_name"),
            )
        )
        carrier_conflict = bool(
            not reviewed_official_identity_transition_match(after, before)
            and (
                (before_ids and after_ids and before_ids != after_ids)
                or (before_urls and after_urls and before_urls != after_urls)
                or names_conflict
            )
        )
        if carrier_conflict:
            binding_conflicts.append(
                {
                    "identity": identity,
                    "accepted_name": _text(before.get("official_name")),
                    "candidate_name": _text(after.get("official_name")),
                    "accepted_strong_keys": sorted(before_strong),
                    "candidate_strong_keys": sorted(after_strong),
                }
            )
        for field in compared_fields:
            accepted_value = before.get(field)
            candidate_value = after.get(field)
            if _values_equal(accepted_value, candidate_value):
                continue
            changes.append(
                {
                    "identity": identity,
                    "official_name": _text(after.get("official_name")),
                    "field": field,
                    "accepted": accepted_value,
                    "candidate": candidate_value,
                }
            )

    # A grandfathered duplicate identity cannot be paired row-by-row: every
    # accepted row matches every candidate row carrying the same UFC/UFCStats
    # identity.  Compare immutable values as a multiset over each ambiguous
    # connected component so row ordering cannot hide a rewrite and so the
    # same group is reported only once when it shares both strong carriers.
    accepted_neighbors: dict[int, set[int]] = {}
    candidate_neighbors: dict[int, set[int]] = {}
    shared_by_pair: dict[tuple[int, int], set[str]] = {}
    for accepted_position, candidate_position, shared in pair_candidates:
        accepted_neighbors.setdefault(accepted_position, set()).add(
            candidate_position
        )
        candidate_neighbors.setdefault(candidate_position, set()).add(
            accepted_position
        )
        shared_by_pair[(accepted_position, candidate_position)] = shared

    def signature_details(
        signatures: Counter,
    ) -> list[dict[str, object]]:
        details: list[dict[str, object]] = []
        for signature, count in sorted(
            signatures.items(), key=lambda item: repr(item[0])
        ):
            details.append(
                {
                    "count": count,
                    "values": {
                        field: {"kind": kind, "value": value}
                        for field, (kind, value) in signature
                    },
                }
            )
        return details

    ambiguous_group_changes: list[dict[str, object]] = []
    visited_accepted: set[int] = set()
    visited_candidate: set[int] = set()
    for start in sorted(accepted_neighbors):
        if start in visited_accepted:
            continue
        component_accepted: set[int] = set()
        component_candidate: set[int] = set()
        pending_accepted = [start]
        pending_candidate: list[int] = []
        while pending_accepted or pending_candidate:
            while pending_accepted:
                accepted_position = pending_accepted.pop()
                if accepted_position in component_accepted:
                    continue
                component_accepted.add(accepted_position)
                pending_candidate.extend(
                    accepted_neighbors.get(accepted_position, ())
                )
            while pending_candidate:
                candidate_position = pending_candidate.pop()
                if candidate_position in component_candidate:
                    continue
                component_candidate.add(candidate_position)
                pending_accepted.extend(
                    candidate_neighbors.get(candidate_position, ())
                )
        visited_accepted.update(component_accepted)
        visited_candidate.update(component_candidate)
        if len(component_accepted) == 1 and len(component_candidate) == 1:
            continue

        accepted_signatures = Counter(
            _row_signature(accepted_rows[position], compared_fields)
            for position in component_accepted
        )
        candidate_signatures = Counter(
            _row_signature(candidate_rows[position], compared_fields)
            for position in component_candidate
        )
        if accepted_signatures == candidate_signatures:
            continue
        shared_carriers = sorted(
            {
                carrier
                for accepted_position in component_accepted
                for candidate_position in component_candidate
                for carrier in shared_by_pair.get(
                    (accepted_position, candidate_position), ()
                )
            }
        )
        ambiguous_group_changes.append(
            {
                "shared_strong_carrier_keys": shared_carriers,
                "accepted_positions": sorted(component_accepted),
                "candidate_positions": sorted(component_candidate),
                "accepted_immutable_multiset": signature_details(
                    accepted_signatures
                ),
                "candidate_immutable_multiset": signature_details(
                    candidate_signatures
                ),
            }
        )

    errors: list[str] = []
    if binding_conflicts:
        errors.append(
            f"{label}_cross_snapshot_identity_conflict:"
            f"{len(binding_conflicts)}"
        )
    if changes:
        errors.append(f"{label}_immutable_fields_changed:{len(changes)}")
    if ambiguous_group_changes:
        errors.append(
            f"{label}_ambiguous_immutable_group_changed:"
            f"{len(ambiguous_group_changes)}"
        )
    return {
        "passed": not errors,
        "errors": errors,
        "compared_fields": compared_fields,
        "changed_field_count": len(changes),
        "changed_fields": changes[:25],
        "cross_snapshot_identity_conflict_count": len(binding_conflicts),
        "cross_snapshot_identity_conflicts": binding_conflicts[:25],
        "ambiguous_immutable_group_change_count": len(
            ambiguous_group_changes
        ),
        "ambiguous_immutable_group_changes": ambiguous_group_changes[:25],
    }


def _dtype_contract(series: pd.Series) -> str:
    observed = series.map(lambda value: not value_is_missing(value))
    if not bool(observed.any()):
        # A CSV column containing only blanks commonly arrives as float64.
        # With no observed values, that inferred dtype is not a real contract.
        return "empty"
    dtype = series.dtype
    if is_bool_dtype(dtype):
        return "boolean"
    if is_numeric_dtype(dtype):
        return "numeric"
    if is_datetime64_any_dtype(dtype):
        return "datetime"
    if is_string_dtype(dtype) or is_object_dtype(dtype):
        observed_values = series[observed].tolist()
        if observed_values and all(isinstance(value, bool) for value in observed_values):
            return "boolean"
        if observed_values and all(
            isinstance(value, Number) and not isinstance(value, bool)
            for value in observed_values
        ):
            # concat can temporarily widen numeric-plus-None columns to object;
            # the same values round-trip through CSV as nullable numeric.
            return "numeric"
        return "text"
    return f"dtype:{dtype}"


def _dtype_contract_compatible(expected: str, actual: str) -> bool:
    if expected == "empty" or actual == "empty":
        return True
    # Integer -> float is the normal pandas widening when a CSV gains a blank.
    return expected == actual


def exact_schema_report(
    frame: pd.DataFrame,
    expected_columns: Sequence[str],
    *,
    label: str,
    reference_frame: pd.DataFrame | None = None,
) -> dict[str, object]:
    actual = [str(column) for column in frame.columns]
    expected = [str(column) for column in expected_columns]
    missing = [column for column in expected if column not in actual]
    unexpected = [column for column in actual if column not in expected]
    order_matches = actual == expected
    errors: list[str] = []
    if missing:
        errors.append(f"{label}_missing_columns:{','.join(missing)}")
    if unexpected:
        errors.append(f"{label}_unexpected_columns:{','.join(unexpected)}")
    if not missing and not unexpected and not order_matches:
        errors.append(f"{label}_column_order_changed")

    expected_dtype_contracts: dict[str, str] = {}
    actual_dtype_contracts = {
        column: _dtype_contract(frame[column]) for column in actual
    }
    dtype_mismatches: list[dict[str, str]] = []
    if reference_frame is not None:
        for column in expected:
            if column not in reference_frame.columns or column not in frame.columns:
                continue
            expected_contract = _dtype_contract(reference_frame[column])
            actual_contract = actual_dtype_contracts[column]
            expected_dtype_contracts[column] = expected_contract
            if not _dtype_contract_compatible(expected_contract, actual_contract):
                dtype_mismatches.append(
                    {
                        "column": column,
                        "expected": expected_contract,
                        "actual": actual_contract,
                    }
                )
    if dtype_mismatches:
        errors.append(
            f"{label}_dtype_mismatch:"
            + ",".join(detail["column"] for detail in dtype_mismatches)
        )
    return {
        "passed": not errors,
        "errors": errors,
        "expected_columns": expected,
        "actual_columns": actual,
        "expected_dtype_contracts": expected_dtype_contracts,
        "actual_dtype_contracts": actual_dtype_contracts,
        "dtype_mismatches": dtype_mismatches,
    }


def _values_equal(left: object, right: object) -> bool:
    if value_is_missing(left) and value_is_missing(right):
        return True
    if value_is_missing(left) != value_is_missing(right):
        return False
    try:
        left_number = float(left)
        right_number = float(right)
    except (TypeError, ValueError):
        return _text(left).casefold() == _text(right).casefold()
    if math.isfinite(left_number) and math.isfinite(right_number):
        return math.isclose(left_number, right_number, rel_tol=1e-10, abs_tol=1e-10)
    return left_number == right_number


def historical_row_change_report(
    accepted: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    key_builder,
    label: str,
    columns: Sequence[str] | None = None,
    allow_blank_to_observed: bool = False,
    identity_alias_builder=None,
) -> dict[str, object]:
    """Reject missing or changed accepted rows and report bounded details."""

    accepted_index = {
        key: row
        for row in _records(accepted)
        if (key := key_builder(row)) is not None
    }
    candidate_index = {
        key: row
        for row in _records(candidate)
        if (key := key_builder(row)) is not None
    }
    matched_rows = {
        key: candidate_index[key]
        for key in set(accepted_index) & set(candidate_index)
    }
    used_candidate_keys = set(accepted_index) & set(candidate_index)
    bridged_identities: list[dict[str, str]] = []
    alias_ambiguities: list[dict[str, object]] = []
    if identity_alias_builder is not None:
        accepted_aliases: dict[str, list[str]] = {}
        candidate_aliases: dict[str, list[str]] = {}
        for key, row in accepted_index.items():
            alias = identity_alias_builder(row)
            if alias:
                accepted_aliases.setdefault(alias, []).append(key)
        for key, row in candidate_index.items():
            alias = identity_alias_builder(row)
            if alias:
                candidate_aliases.setdefault(alias, []).append(key)
        alias_ambiguities = [
            {
                "alias": alias,
                "accepted_count": len(accepted_alias_keys),
                "candidate_count": len(candidate_aliases.get(alias, [])),
            }
            for alias, accepted_alias_keys in sorted(accepted_aliases.items())
            if len(candidate_aliases.get(alias, [])) > len(accepted_alias_keys)
        ]
        for accepted_key in sorted(set(accepted_index) - set(matched_rows)):
            alias = identity_alias_builder(accepted_index[accepted_key])
            # The alias bridge exists only to enrich a legacy name/date key
            # with strong IDs.  Never let it conceal loss or replacement of
            # strong IDs that were already accepted.
            if not alias or accepted_key != alias:
                continue
            accepted_alias_keys = accepted_aliases.get(alias or "", [])
            candidate_alias_keys = candidate_aliases.get(alias or "", [])
            if len(accepted_alias_keys) != 1 or len(candidate_alias_keys) != 1:
                continue
            candidate_key = candidate_alias_keys[0]
            if candidate_key == alias:
                continue
            if candidate_key in used_candidate_keys:
                continue
            matched_rows[accepted_key] = candidate_index[candidate_key]
            used_candidate_keys.add(candidate_key)
            bridged_identities.append(
                {
                    "accepted_identity": accepted_key,
                    "candidate_identity": candidate_key,
                    "alias": str(alias),
                }
            )

    missing_keys = sorted(set(accepted_index) - set(matched_rows))
    compare_columns = list(columns) if columns is not None else [
        column for column in accepted.columns if column in candidate.columns
    ]
    changed_details: list[dict[str, object]] = []
    changed_field_count = 0
    for key in sorted(matched_rows):
        before = accepted_index[key]
        after = matched_rows[key]
        for column in compare_columns:
            old = before.get(column)
            new = after.get(column)
            if allow_blank_to_observed and value_is_missing(old) and not value_is_missing(new):
                continue
            if _values_equal(old, new):
                continue
            changed_field_count += 1
            if len(changed_details) < 25:
                changed_details.append(
                    {"identity": key, "field": column, "accepted": old, "candidate": new}
                )
    errors: list[str] = []
    if missing_keys:
        errors.append(f"{label}_historical_rows_removed:{len(missing_keys)}")
    if changed_field_count:
        errors.append(f"{label}_historical_fields_changed:{changed_field_count}")
    if alias_ambiguities:
        errors.append(f"{label}_historical_alias_ambiguity:{len(alias_ambiguities)}")
    new_keys = sorted(set(candidate_index) - used_candidate_keys)
    return {
        "passed": not errors,
        "errors": errors,
        "accepted_identity_count": len(accepted_index),
        "candidate_identity_count": len(candidate_index),
        "missing_identity_count": len(missing_keys),
        "missing_identities": missing_keys[:25],
        "changed_field_count": changed_field_count,
        "changed_fields": changed_details,
        "bridged_identity_count": len(bridged_identities),
        "bridged_identities": bridged_identities[:25],
        "alias_ambiguity_count": len(alias_ambiguities),
        "alias_ambiguities": alias_ambiguities[:25],
        "new_identity_count": len(new_keys),
        "new_identities": new_keys,
    }


def preserve_accepted_processed_history(
    accepted: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    label: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Restore accepted processed rows while retaining only genuine appends.

    Full-history rebuilds are useful for producing features for a new fight, but
    newly observed profile fields must not rewrite rows that an earlier data
    generation already accepted.  This helper matches every accepted row by the
    same stable identity used by the publication gate, permits the existing
    unique name/date bridge for legacy identity enrichment, and returns the
    candidate unchanged when preservation would be ambiguous or incomplete.
    """

    accepted_rows = list(_records(accepted))
    candidate_rows = list(_records(candidate))
    accepted_keys = [stable_processed_fight_key(row) for row in accepted_rows]
    candidate_keys = [stable_processed_fight_key(row) for row in candidate_rows]
    accepted_counts = Counter(key for key in accepted_keys if key is not None)
    candidate_counts = Counter(key for key in candidate_keys if key is not None)

    def identity_digest(keys: Sequence[str | None]) -> str:
        digest = hashlib.sha256()
        for key in sorted(str(key) for key in keys if key is not None):
            digest.update(key.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()
    accepted_duplicates = {
        str(key): count for key, count in accepted_counts.items() if count > 1
    }
    candidate_duplicates = {
        str(key): count for key, count in candidate_counts.items() if count > 1
    }
    missing_columns = [column for column in accepted.columns if column not in candidate]
    errors: list[str] = []
    accepted_invalid = sum(key is None for key in accepted_keys)
    candidate_invalid = sum(key is None for key in candidate_keys)
    if missing_columns:
        errors.append(f"{label}_accepted_columns_missing:{len(missing_columns)}")
    if accepted_invalid:
        errors.append(f"{label}_accepted_invalid_identity:{accepted_invalid}")
    if candidate_invalid:
        errors.append(f"{label}_candidate_invalid_identity:{candidate_invalid}")
    if accepted_duplicates:
        errors.append(f"{label}_accepted_duplicate_identity:{len(accepted_duplicates)}")
    if candidate_duplicates:
        errors.append(f"{label}_candidate_duplicate_identity:{len(candidate_duplicates)}")

    report: dict[str, object] = {
        "passed": False,
        "errors": errors,
        "accepted_identity_count": len(accepted_counts),
        "candidate_identity_count": len(candidate_counts),
        "accepted_identity_sha256": identity_digest(accepted_keys),
        "candidate_identity_sha256": identity_digest(candidate_keys),
        "accepted_invalid_identity_rows": accepted_invalid,
        "candidate_invalid_identity_rows": candidate_invalid,
        "accepted_duplicate_identities": dict(list(accepted_duplicates.items())[:25]),
        "candidate_duplicate_identities": dict(list(candidate_duplicates.items())[:25]),
        "missing_columns": missing_columns,
        "schema_equal": list(accepted.columns) == list(candidate.columns),
        "matched_identity_count": 0,
        "bridged_identity_count": 0,
        "bridged_identities": [],
        "alias_ambiguity_count": 0,
        "alias_ambiguities": [],
        "missing_identity_count": 0,
        "missing_identities": [],
        "new_identity_count": 0,
        "new_identities": [],
        "restored_row_count": 0,
        "restored_field_count": 0,
    }
    if errors:
        return candidate.copy(), report

    accepted_by_key = {str(key): pos for pos, key in enumerate(accepted_keys)}
    candidate_by_key = {str(key): pos for pos, key in enumerate(candidate_keys)}
    matched_positions: dict[int, int] = {}
    used_candidate_positions: set[int] = set()
    for key, accepted_position in accepted_by_key.items():
        candidate_position = candidate_by_key.get(key)
        if candidate_position is None:
            continue
        matched_positions[accepted_position] = candidate_position
        used_candidate_positions.add(candidate_position)

    accepted_aliases: dict[str, list[int]] = {}
    candidate_aliases: dict[str, list[int]] = {}
    for position, row in enumerate(accepted_rows):
        alias = stable_processed_name_date_key(row)
        if alias:
            accepted_aliases.setdefault(alias, []).append(position)
    for position, row in enumerate(candidate_rows):
        alias = stable_processed_name_date_key(row)
        if alias:
            candidate_aliases.setdefault(alias, []).append(position)

    alias_ambiguities = [
        {
            "alias": alias,
            "accepted_count": len(accepted_alias_positions),
            "candidate_count": len(candidate_aliases.get(alias, [])),
        }
        for alias, accepted_alias_positions in sorted(accepted_aliases.items())
        if len(candidate_aliases.get(alias, [])) > len(accepted_alias_positions)
    ]
    if alias_ambiguities:
        errors.append(f"{label}_historical_alias_ambiguity:{len(alias_ambiguities)}")
        report.update(
            {
                "errors": errors,
                "alias_ambiguity_count": len(alias_ambiguities),
                "alias_ambiguities": alias_ambiguities[:25],
            }
        )
        return candidate.copy(), report

    bridges: list[dict[str, str]] = []
    for accepted_position in range(len(accepted_rows)):
        if accepted_position in matched_positions:
            continue
        alias = stable_processed_name_date_key(accepted_rows[accepted_position])
        accepted_key = str(accepted_keys[accepted_position])
        if not alias or accepted_key != alias:
            continue
        accepted_alias_positions = accepted_aliases.get(alias or "", [])
        candidate_alias_positions = candidate_aliases.get(alias or "", [])
        if len(accepted_alias_positions) != 1 or len(candidate_alias_positions) != 1:
            continue
        candidate_position = candidate_alias_positions[0]
        if candidate_position in used_candidate_positions:
            continue
        if str(candidate_keys[candidate_position]) == alias:
            continue
        matched_positions[accepted_position] = candidate_position
        used_candidate_positions.add(candidate_position)
        bridges.append(
            {
                "accepted_identity": accepted_key,
                "candidate_identity": str(candidate_keys[candidate_position]),
                "alias": str(alias),
            }
        )

    missing_positions = [
        position
        for position in range(len(accepted_rows))
        if position not in matched_positions
    ]
    new_positions = [
        position
        for position in range(len(candidate_rows))
        if position not in used_candidate_positions
    ]
    missing_identities = [str(accepted_keys[position]) for position in missing_positions]
    new_identities = [str(candidate_keys[position]) for position in new_positions]
    if missing_positions:
        errors.append(f"{label}_historical_rows_removed:{len(missing_positions)}")
        report.update(
            {
                "errors": errors,
                "matched_identity_count": len(matched_positions),
                "bridged_identity_count": len(bridges),
                "bridged_identities": bridges[:25],
                "missing_identity_count": len(missing_positions),
                "missing_identities": missing_identities[:25],
                "new_identity_count": len(new_positions),
                "new_identities": new_identities,
            }
        )
        return candidate.copy(), report

    common_columns = [column for column in accepted.columns if column in candidate]
    restored_field_count = 0
    restored_row_positions: set[int] = set()
    ordered_candidate_positions: list[int] = []
    for accepted_position in range(len(accepted_rows)):
        candidate_position = matched_positions[accepted_position]
        ordered_candidate_positions.append(candidate_position)
        before = accepted_rows[accepted_position]
        after = candidate_rows[candidate_position]
        row_changed = False
        for column in common_columns:
            if _values_equal(before.get(column), after.get(column)):
                continue
            restored_field_count += 1
            row_changed = True
        if row_changed:
            restored_row_positions.add(accepted_position)

    # Start from the matched candidate rows so any explicitly evolved columns
    # survive, then restore every accepted column and append only unused rows.
    restored_accepted = candidate.iloc[ordered_candidate_positions].reset_index(drop=True).copy()
    accepted_reset = accepted.reset_index(drop=True)
    for column in common_columns:
        restored_accepted[column] = accepted_reset[column]
    new_rows = candidate.iloc[new_positions].reset_index(drop=True).copy()
    preserved = pd.concat([restored_accepted, new_rows], ignore_index=True, sort=False)
    preserved = preserved.loc[:, list(candidate.columns)]

    report.update(
        {
            "passed": True,
            "errors": [],
            "matched_identity_count": len(matched_positions),
            "bridged_identity_count": len(bridges),
            "bridged_identities": bridges[:25],
            "missing_identity_count": 0,
            "missing_identities": [],
            "new_identity_count": len(new_positions),
            "new_identities": new_identities,
            "restored_row_count": len(restored_row_positions),
            "restored_field_count": restored_field_count,
        }
    )
    return preserved, report


def _event_participant_signature(row: Mapping[str, object]) -> str | None:
    event = _normalized(row.get("EVENT") or row.get("event_name"))
    fighter_ids = sorted(
        value
        for value in (
            normalize_ufcstats_id(row.get("fighter_a_id")),
            normalize_ufcstats_id(row.get("fighter_b_id")),
        )
        if value
    )
    if len(fighter_ids) == 2 and fighter_ids[0] != fighter_ids[1]:
        participants = (f"id:{fighter_ids[0]}", f"id:{fighter_ids[1]}")
    else:
        bout_participants = _bout_participants(row.get("BOUT"))
        if bout_participants is None:
            names = sorted(
                filter(
                    None,
                    (
                        _normalized(row.get("fighter_a")),
                        _normalized(row.get("fighter_b")),
                    ),
                )
            )
            bout_participants = (
                tuple(names)
                if len(names) == 2 and names[0] != names[1]
                else None
            )
        if bout_participants is None:
            return None
        participants = (
            f"name:{bout_participants[0]}",
            f"name:{bout_participants[1]}",
        )
    if not event:
        return None
    return f"event:{event}|fighters:{participants[0]}|{participants[1]}"


def _ordered_bout_participants(value: object) -> tuple[str, str] | None:
    """Return normalized bout names without losing the official A/B order."""
    parts = [_normalized(part) for part in _BOUT_SPLIT_RE.split(_text(value))]
    if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
        return None
    return parts[0], parts[1]


def _normalized_event_date_evidence(
    event_date_lookup: Mapping[str, object] | None,
) -> tuple[dict[str, str], list[dict[str, object]]]:
    normalized: dict[str, str] = {}
    conflicts: list[dict[str, object]] = []
    for event_name, raw_date in (event_date_lookup or {}).items():
        key = _normalized(event_name)
        event_date = _processed_event_date({"event_date": raw_date})
        if not key or event_date is None:
            continue
        previous = normalized.get(key)
        if previous is not None:
            conflicts.append(
                {
                    "event": key,
                    "first_date": previous,
                    "duplicate_date": event_date,
                    "reason": (
                        "conflicting_event_date"
                        if previous != event_date
                        else "duplicate_event_date"
                    ),
                }
            )
            continue
        normalized[key] = event_date
    return normalized, conflicts


def _processed_side_map_for_raw_result(
    raw_row: Mapping[str, object],
    processed_row: Mapping[str, object],
) -> tuple[dict[str, str], list[dict[str, object]]]:
    """Map raw A/B to processed A/B without severing names from stable IDs."""
    raw_names = _ordered_bout_participants(raw_row.get("BOUT"))
    if raw_names is None:
        return {}, [
            {
                "reason": "raw_bout_orientation_invalid",
                "expected": "two ordered raw participants",
                "actual": raw_row.get("BOUT"),
            }
        ]

    side_map: dict[str, str] = {}
    issues: list[dict[str, object]] = []
    for position, raw_side in enumerate(("a", "b")):
        raw_name = raw_names[position]
        raw_name_key = canonical_fighter_name_key(raw_name)
        raw_id = normalize_ufcstats_id(raw_row.get(f"fighter_{raw_side}_id"))
        matches: list[str] = []
        for processed_side in ("a", "b"):
            processed_id = normalize_ufcstats_id(
                processed_row.get(f"fighter_{processed_side}_id")
            )
            processed_name_key = canonical_fighter_name_key(
                _text(processed_row.get(f"fighter_{processed_side}"))
            )
            if raw_id:
                if processed_id == raw_id:
                    matches.append(processed_side)
            elif raw_name_key and processed_name_key == raw_name_key:
                matches.append(processed_side)
        if len(matches) != 1:
            issues.append(
                {
                    "reason": "participant_identity_orientation_unresolved",
                    "expected": {
                        "raw_side": raw_side,
                        "fighter_name": raw_name,
                        "fighter_id": raw_id,
                    },
                    "actual": matches,
                }
            )
            continue

        processed_side = matches[0]
        side_map[raw_side] = processed_side
        processed_name = processed_row.get(f"fighter_{processed_side}")
        processed_name_key = canonical_fighter_name_key(_text(processed_name))
        if not raw_name_key or processed_name_key != raw_name_key:
            issues.append(
                {
                    "reason": "participant_name_id_orientation_mismatch",
                    "expected": {
                        "raw_side": raw_side,
                        "fighter_name": raw_name,
                        "fighter_id": raw_id,
                    },
                    "actual": {
                        "processed_side": processed_side,
                        "fighter_name": processed_name,
                        "fighter_id": normalize_ufcstats_id(
                            processed_row.get(f"fighter_{processed_side}_id")
                        ),
                    },
                }
            )

    if len(side_map) == 2 and len(set(side_map.values())) != 2:
        issues.append(
            {
                "reason": "participant_identity_orientation_ambiguous",
                "expected": {"a": "one side", "b": "the other side"},
                "actual": side_map,
            }
        )
    return side_map, issues


def _raw_processed_result_binding_mismatches(
    raw_row: Mapping[str, object],
    processed_row: Mapping[str, object],
    *,
    event_date_evidence: Mapping[str, str],
    require_event_date_evidence: bool,
) -> list[dict[str, object]]:
    """Validate result semantics after pairing one raw and processed append."""
    identity = stable_raw_fight_key(raw_row) or "unknown"
    mismatches: list[dict[str, object]] = []

    def add(reason: str, expected: object, actual: object) -> None:
        mismatches.append(
            {
                "identity": identity,
                "reason": reason,
                "expected": expected,
                "actual": actual,
            }
        )

    try:
        classification = validate_marked_result(
            raw_row.get("OUTCOME"),
            raw_row.get("METHOD"),
            policy_version=raw_row.get("RESULT_POLICY_VERSION"),
            persisted_result_class=raw_row.get("RESULT_CLASS"),
            persisted_binary_target_eligible=raw_row.get(
                "BINARY_TARGET_ELIGIBLE"
            ),
        )
    except ValueError as exc:
        add("raw_result_policy_invalid", "valid marked raw result", str(exc))
        return mismatches

    processed_policy_version = _text(processed_row.get("result_policy_version"))
    if processed_policy_version != RESULT_POLICY_VERSION:
        add(
            "result_policy_version_mismatch",
            RESULT_POLICY_VERSION,
            processed_policy_version,
        )
    processed_result_class = _text(processed_row.get("result_class")).casefold()
    if processed_result_class != classification.result_class:
        add(
            "result_class_mismatch",
            classification.result_class,
            processed_result_class,
        )
    try:
        processed_eligible = parse_policy_boolean(
            processed_row.get("binary_target_eligible")
        )
    except ValueError as exc:
        add(
            "binary_target_eligibility_invalid",
            classification.binary_target_eligible,
            str(exc),
        )
        processed_eligible = None
    if processed_eligible != classification.binary_target_eligible:
        add(
            "binary_target_eligibility_mismatch",
            classification.binary_target_eligible,
            processed_eligible,
        )

    raw_outcome = _text(raw_row.get("OUTCOME"))
    processed_outcome = _text(processed_row.get("official_outcome"))
    if processed_outcome != raw_outcome:
        add("official_outcome_mismatch", raw_outcome, processed_outcome)
    if not _values_equal(raw_row.get("METHOD"), processed_row.get("method")):
        add("method_mismatch", raw_row.get("METHOD"), processed_row.get("method"))

    raw_names = _ordered_bout_participants(raw_row.get("BOUT"))
    processed_side_map, participant_issues = _processed_side_map_for_raw_result(
        raw_row,
        processed_row,
    )
    for issue in participant_issues:
        add(issue["reason"], issue["expected"], issue["actual"])

    processed_target = processed_row.get("target")
    processed_winner = canonical_fighter_name_key(
        _text(processed_row.get("winner"))
    )
    if classification.result_class in {RESULT_CLASS_NORMAL, RESULT_CLASS_DQ}:
        raw_winner_side = str(classification.winner_side or "")
        winner_side = processed_side_map.get(raw_winner_side)
        if winner_side is None:
            add(
                "official_winner_identity_unresolved",
                classification.winner_side,
                None,
            )
        else:
            raw_winner_position = 0 if raw_winner_side == "a" else 1
            expected_winner = canonical_fighter_name_key(
                raw_names[raw_winner_position] if raw_names is not None else None
            )
            if processed_winner != expected_winner:
                add("official_winner_mismatch", expected_winner, processed_winner)
            if classification.result_class == RESULT_CLASS_NORMAL:
                expected_target = 1.0 if winner_side == "a" else 0.0
                try:
                    actual_target = float(processed_target)
                except (TypeError, ValueError):
                    actual_target = None
                if value_is_missing(processed_target) or actual_target != expected_target:
                    add("binary_target_mismatch", expected_target, processed_target)
            elif not value_is_missing(processed_target):
                add("dq_binary_target_must_be_blank", None, processed_target)
    else:
        if not value_is_missing(processed_row.get("winner")):
            add("nonbinary_winner_must_be_blank", None, processed_row.get("winner"))
        if not value_is_missing(processed_target):
            add("nonbinary_target_must_be_blank", None, processed_target)

    event_key = _normalized(raw_row.get("EVENT"))
    expected_event_date = event_date_evidence.get(event_key)
    if expected_event_date is None and require_event_date_evidence:
        add("event_date_evidence_missing", "one staged event date", None)
    elif expected_event_date is not None:
        actual_event_date = _processed_event_date(processed_row)
        if actual_event_date != expected_event_date:
            add("event_date_mismatch", expected_event_date, actual_event_date)
    return mismatches


def processed_append_result_report(
    *,
    accepted_processed: pd.DataFrame,
    candidate_processed: pd.DataFrame,
    candidate_results: pd.DataFrame,
    new_fight_keys: set[str],
    event_date_lookup: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Require each processed append to equal one newly accepted raw result."""
    identity_columns = [
        column
        for column in (
            "event_date",
            "event_name",
            "fighter_a",
            "fighter_b",
            "fighter_a_id",
            "fighter_b_id",
        )
        if column in accepted_processed.columns or column in candidate_processed.columns
    ]
    accepted_identity = accepted_processed.reindex(columns=identity_columns)
    candidate_identity = candidate_processed.reindex(columns=identity_columns)
    history = historical_row_change_report(
        accepted_identity,
        candidate_identity,
        key_builder=stable_processed_fight_key,
        identity_alias_builder=stable_processed_name_date_key,
        label="processed_append_identity",
        columns=[],
    )
    actual_new_keys = set(history.get("new_identities") or [])
    actual_rows = [
        row
        for row in _records(candidate_processed)
        if stable_processed_fight_key(row) in actual_new_keys
    ]
    expected_rows = [
        row
        for row in _records(candidate_results)
        if stable_raw_fight_key(row) in new_fight_keys
    ]
    expected_signatures = [_event_participant_signature(row) for row in expected_rows]
    actual_signatures = [_event_participant_signature(row) for row in actual_rows]
    invalid_expected = sum(signature is None for signature in expected_signatures)
    invalid_actual = sum(signature is None for signature in actual_signatures)
    expected_counts = Counter(
        signature for signature in expected_signatures if signature is not None
    )
    actual_counts = Counter(
        signature for signature in actual_signatures if signature is not None
    )
    missing = expected_counts - actual_counts
    unexpected = actual_counts - expected_counts
    event_date_evidence, event_date_conflicts = _normalized_event_date_evidence(
        event_date_lookup
    )
    # A new raw result cannot be published without independently staged event
    # metadata.  Keep the argument optional for no-change and legacy callers,
    # but fail closed as soon as there is an append to bind.
    require_event_date_evidence = bool(expected_rows)
    expected_by_signature: dict[str, list[dict[str, object]]] = {}
    actual_by_signature: dict[str, list[dict[str, object]]] = {}
    for row, signature in zip(expected_rows, expected_signatures):
        if signature is not None:
            expected_by_signature.setdefault(signature, []).append(row)
    for row, signature in zip(actual_rows, actual_signatures):
        if signature is not None:
            actual_by_signature.setdefault(signature, []).append(row)
    binding_mismatches: list[dict[str, object]] = []
    for signature in sorted(set(expected_by_signature) & set(actual_by_signature)):
        raw_matches = expected_by_signature[signature]
        processed_matches = actual_by_signature[signature]
        if len(raw_matches) != 1 or len(processed_matches) != 1:
            binding_mismatches.append(
                {
                    "identity": signature,
                    "reason": "ambiguous_raw_processed_binding",
                    "expected": len(raw_matches),
                    "actual": len(processed_matches),
                }
            )
            continue
        binding_mismatches.extend(
            _raw_processed_result_binding_mismatches(
                raw_matches[0],
                processed_matches[0],
                event_date_evidence=event_date_evidence,
                require_event_date_evidence=require_event_date_evidence,
            )
        )
    errors: list[str] = []
    if history.get("passed") is not True:
        errors.append("processed_append_identity_gate_failed")
    if invalid_expected:
        errors.append(f"processed_append_raw_identity_invalid:{invalid_expected}")
    if invalid_actual:
        errors.append(f"processed_append_identity_invalid:{invalid_actual}")
    if missing:
        errors.append(f"processed_append_missing_for_new_results:{sum(missing.values())}")
    if unexpected:
        errors.append(f"processed_append_without_new_result:{sum(unexpected.values())}")
    if event_date_conflicts:
        errors.append(
            f"processed_append_event_date_evidence_conflict:{len(event_date_conflicts)}"
        )
    if binding_mismatches:
        errors.append(
            f"processed_append_raw_binding_mismatch:{len(binding_mismatches)}"
        )
    return {
        "passed": not errors,
        "errors": errors,
        "expected_append_count": len(expected_rows),
        "actual_append_count": len(actual_rows),
        "invalid_expected_identity_count": invalid_expected,
        "invalid_actual_identity_count": invalid_actual,
        "missing_append_count": int(sum(missing.values())),
        "missing_appends": dict(list(missing.items())[:25]),
        "unexpected_append_count": int(sum(unexpected.values())),
        "unexpected_appends": dict(list(unexpected.items())[:25]),
        "event_date_evidence_count": len(event_date_evidence),
        "event_date_evidence_conflicts": event_date_conflicts[:25],
        "raw_binding_mismatch_count": len(binding_mismatches),
        "raw_binding_mismatches": binding_mismatches[:25],
        "identity_history": history,
    }


def _profile_group_state(row: Mapping[str, object], fields: Sequence[str]) -> dict[str, object]:
    present = [field for field in fields if not value_is_missing(row.get(field))]
    missing = [field for field in fields if value_is_missing(row.get(field))]
    return {
        "present": present,
        "missing": missing,
        "complete": not missing,
        "empty": not present,
        "partial": bool(present and missing),
    }


def profile_pull_report(
    accepted: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    fighter_ids_with_completed_fights: set[str] | None = None,
) -> dict[str, object]:
    """Detect profile field loss and suspicious partial statistical bundles."""

    completed = set(fighter_ids_with_completed_fights or set())
    accepted_rows = list(_records(accepted))
    accepted_keys = [stable_profile_key(row) for row in accepted_rows]
    accepted_key_counts = Counter(key for key in accepted_keys if key is not None)
    accepted_index = {
        key: row
        for key, row in zip(accepted_keys, accepted_rows)
        if key is not None
    }
    candidate_index: dict[str, dict[str, object]] = {}
    for row in _records(candidate):
        key = stable_profile_key(row)
        if key is None:
            continue
        candidate_index[key] = row

    completeness_groups: dict[str, dict[str, object]] = {}
    for name, fields in (
        ("physical", PROFILE_PHYSICAL_FIELDS),
        ("striking", PROFILE_STRIKING_FIELDS),
        ("grappling", PROFILE_GRAPPLING_FIELDS),
    ):
        counts = Counter()
        for row in candidate_index.values():
            state = _profile_group_state(row, fields)
            status = "complete" if state["complete"] else "empty" if state["empty"] else "partial"
            counts[status] += 1
        total = len(candidate_index)
        completeness_groups[name] = {
            "total": total,
            "complete": int(counts["complete"]),
            "partial": int(counts["partial"]),
            "empty": int(counts["empty"]),
            "complete_pct": round(100.0 * counts["complete"] / total, 6) if total else 0.0,
            "partial_pct": round(100.0 * counts["partial"] / total, 6) if total else 0.0,
            "empty_pct": round(100.0 * counts["empty"] / total, 6) if total else 0.0,
        }

    losses: list[dict[str, object]] = []
    suspicious: list[dict[str, object]] = []
    source_limited: list[dict[str, object]] = []
    for key, row in candidate_index.items():
        before = accepted_index.get(key)
        if before is not None:
            for field in PROFILE_OBSERVED_FIELDS:
                if not value_is_missing(before.get(field)) and value_is_missing(row.get(field)):
                    losses.append({"identity": key, "field": field, "accepted": before.get(field)})

        changed = before is None or any(
            not _values_equal(before.get(field), row.get(field))
            for field in PROFILE_OBSERVED_FIELDS
        )
        if not changed:
            continue
        striking = _profile_group_state(row, PROFILE_STRIKING_FIELDS)
        grappling = _profile_group_state(row, PROFILE_GRAPPLING_FIELDS)
        partial_groups = [
            name for name, state in (("striking", striking), ("grappling", grappling))
            if state["partial"]
        ]
        if partial_groups:
            suspicious.append(
                {
                    "identity": key,
                    "groups": partial_groups,
                    "striking": striking,
                    "grappling": grappling,
                }
            )
        elif striking["empty"] and grappling["empty"]:
            if key in completed:
                suspicious.append(
                    {
                        "identity": key,
                        "groups": ["striking", "grappling"],
                        "reason": "all_rate_statistics_missing_despite_completed_fight",
                    }
                )
            else:
                source_limited.append(
                    {"identity": key, "reason": "no_published_rate_statistics_for_debutant"}
                )

    errors: list[str] = []
    candidate_key_counts = Counter(
        key
        for row in _records(candidate)
        if (key := stable_profile_key(row)) is not None
    )
    new_duplicate_units = sum(
        max(0, count - max(1, accepted_key_counts.get(key, 0)))
        for key, count in candidate_key_counts.items()
    )
    if new_duplicate_units:
        errors.append(f"profile_duplicate_identity:{new_duplicate_units}")
    missing_profiles = sorted(set(accepted_index) - set(candidate_index))
    if missing_profiles:
        errors.append(f"profile_historical_rows_removed:{len(missing_profiles)}")
    if losses:
        errors.append(f"profile_previously_observed_fields_lost:{len(losses)}")
    if suspicious:
        errors.append(f"profile_suspicious_partial_bundles:{len(suspicious)}")
    return {
        "passed": not errors,
        "errors": errors,
        "accepted_profiles": len(accepted_index),
        "candidate_profiles": len(candidate_index),
        "bundle_completeness": completeness_groups,
        "missing_profiles": missing_profiles[:25],
        "field_losses": losses[:25],
        "suspicious_profiles": suspicious[:25],
        "source_limited_profiles": source_limited[:25],
        "source_limited_count": len(source_limited),
    }


def incoming_result_policy_report(
    accepted: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, object]:
    """Independently validate policy metadata on every newly appended result."""

    accepted_keys = {
        key for row in _records(accepted) if (key := stable_raw_fight_key(row)) is not None
    }
    errors: list[str] = []
    details: list[dict[str, object]] = []
    class_counts: Counter[str] = Counter()
    new_keys: set[str] = set()
    for row in _records(candidate):
        key = stable_raw_fight_key(row)
        if key is None or key in accepted_keys:
            continue
        new_keys.add(key)
        try:
            classification = validate_marked_result(
                row.get("OUTCOME"),
                row.get("METHOD"),
                policy_version=row.get("RESULT_POLICY_VERSION"),
                persisted_result_class=row.get("RESULT_CLASS"),
                persisted_binary_target_eligible=row.get("BINARY_TARGET_ELIGIBLE"),
            )
        except ValueError as exc:
            details.append({"identity": key, "reason": str(exc)})
            continue
        class_counts[classification.result_class] += 1
        if classification.result_class not in {
            RESULT_CLASS_NORMAL,
            RESULT_CLASS_DQ,
            RESULT_CLASS_DRAW,
            RESULT_CLASS_NO_CONTEST,
        }:
            details.append({"identity": key, "reason": "unsupported_result_class"})
        if _text(row.get("RESULT_POLICY_VERSION")) != RESULT_POLICY_VERSION:
            details.append({"identity": key, "reason": "unexpected_result_policy_version"})
    if details:
        errors.append(f"incoming_results_invalid_or_quarantined:{len(details)}")
    return {
        "passed": not errors,
        "errors": errors,
        "new_fight_count": len(new_keys),
        "new_fight_keys": sorted(new_keys),
        "result_class_counts": dict(sorted(class_counts.items())),
        "binary_target_excluded_count": int(
            class_counts[RESULT_CLASS_DQ]
            + class_counts[RESULT_CLASS_DRAW]
            + class_counts[RESULT_CLASS_NO_CONTEST]
        ),
        "details": details[:25],
    }


def _parse_nonnegative(value: object) -> float | None:
    if value_is_missing(value):
        return None
    try:
        parsed = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def fight_statistics_report(
    results: pd.DataFrame,
    statistics: pd.DataFrame,
    *,
    new_fight_keys: set[str] | None = None,
) -> dict[str, object]:
    """Validate complete, nonnegative statistical bundles for new fights."""

    errors: list[str] = []
    details: list[dict[str, object]] = []
    required_result = [column for column in RESULT_BASE_COLUMNS if column not in results.columns]
    required_stats = [column for column in STAT_COLUMNS if column not in statistics.columns]
    if required_result:
        errors.append(f"fight_results_missing_columns:{','.join(required_result)}")
    if required_stats:
        errors.append(f"fight_statistics_missing_columns:{','.join(required_stats)}")
    if errors:
        return {"passed": False, "errors": errors, "details": details}

    selected_keys = set(new_fight_keys or set())
    if not selected_keys:
        return {"passed": True, "errors": [], "details": [], "fights_checked": 0}

    stats_by_fight: dict[str, list[dict[str, object]]] = {}
    for row in _records(statistics):
        key = stable_raw_fight_key(row)
        if key:
            stats_by_fight.setdefault(key, []).append(row)

    fights_checked = 0
    for result in _records(results):
        key = stable_raw_fight_key(result)
        if key not in selected_keys:
            continue
        fights_checked += 1
        reasons: list[str] = []
        participants = _bout_participants(result.get("BOUT"))
        finish_round = _parse_nonnegative(result.get("ROUND"))
        if participants is None:
            reasons.append("invalid_participants")
        if finish_round is None or int(finish_round) != finish_round or not 1 <= finish_round <= 12:
            reasons.append("invalid_finish_round")
            expected_rounds: set[int] = set()
        else:
            expected_rounds = set(range(1, int(finish_round) + 1))
        result_without_url = dict(result)
        result_without_url["URL"] = ""
        rows = stats_by_fight.get(stable_raw_fight_key(result_without_url) or "", [])
        by_round: dict[int, list[dict[str, object]]] = {}
        for stat in rows:
            round_value = _text(stat.get("ROUND")).casefold().replace("round", "").strip()
            try:
                round_number = int(float(round_value))
            except ValueError:
                reasons.append("invalid_stat_round")
                continue
            by_round.setdefault(round_number, []).append(stat)
            for column in _COUNT_STAT_COLUMNS:
                if _parse_nonnegative(stat.get(column)) is None:
                    reasons.append(f"invalid_{column}")
            for column in _PERCENT_STAT_COLUMNS:
                value = stat.get(column)
                if isinstance(value, str) and value.strip() == "---":
                    continue
                parsed = _parse_nonnegative(value)
                if parsed is None or parsed > 100:
                    reasons.append(f"invalid_{column}")
            for column in _PAIR_STAT_COLUMNS:
                match = _PAIR_RE.match(_text(stat.get(column)))
                if not match:
                    reasons.append(f"invalid_{column}")
                    continue
                landed = float(match.group("landed"))
                attempted = float(match.group("attempted"))
                if landed > attempted:
                    reasons.append(f"landed_exceeds_attempted_{column}")
            raw_ctrl = stat.get("CTRL")
            if isinstance(raw_ctrl, str) and raw_ctrl.strip() == "--":
                # UFCStats uses this exact marker when control time is unavailable.
                # Preserve it as missing; it is not an observed zero-duration clock.
                continue
            clock = _CLOCK_RE.match(_text(raw_ctrl))
            if not clock or int(clock.group("seconds")) >= 60:
                reasons.append("invalid_CTRL")
        if set(by_round) != expected_rounds:
            reasons.append("incomplete_round_set")
        for round_number in expected_rounds:
            round_rows = by_round.get(round_number, [])
            round_fighters = {_normalized(row.get("FIGHTER")) for row in round_rows}
            if len(round_rows) != 2 or participants is None or round_fighters != set(participants):
                reasons.append(f"incomplete_round_{round_number}_fighters")
        if reasons:
            details.append({"identity": key, "reasons": list(dict.fromkeys(reasons))})
    missing_results = sorted(selected_keys - {
        key for row in _records(results) if (key := stable_raw_fight_key(row)) is not None
    })
    if missing_results:
        errors.append(f"new_fight_results_missing:{len(missing_results)}")
    if details:
        errors.append(f"new_fight_statistical_bundles_invalid:{len(details)}")
    return {
        "passed": not errors,
        "errors": errors,
        "details": details[:25],
        "fights_checked": fights_checked,
    }


def _processed_value_anomalies(
    frame: pd.DataFrame,
) -> tuple[Counter, list[dict[str, object]]]:
    anomalies: Counter = Counter()
    details: list[dict[str, object]] = []
    required = ("event_date", "fighter_a", "fighter_b", "target")
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        return anomalies, [
            {
                "reason": "missing_columns",
                "columns": missing_columns,
                "rows": 1,
            }
        ]
    dates = pd.to_datetime(frame["event_date"], errors="coerce")
    rows = list(_records(frame))

    def row_identity(row: Mapping[str, object]) -> object:
        return stable_processed_name_date_key(row) or _row_signature(
            row, list(frame.columns)
        )

    for position, row in enumerate(rows):
        identity = row_identity(row)
        if bool(dates.isna().iloc[position]):
            anomalies[(identity, "event_date", "invalid_date")] += 1
        a_name = _normalized(row.get("fighter_a"))
        b_name = _normalized(row.get("fighter_b"))
        if not a_name or not b_name or a_name == b_name:
            anomalies[(identity, "participants", "invalid_participants")] += 1
        target = row.get("target")
        if not value_is_missing(target):
            try:
                parsed_target = float(target)
            except (TypeError, ValueError):
                parsed_target = math.nan
            if parsed_target not in {0.0, 1.0}:
                anomalies[(identity, "target", "invalid_binary_target")] += 1

    numeric_columns = [
        column
        for column in frame.columns
        if column.endswith(("_acc", "_def", "_share", "_prob"))
        or column in {"a_odds", "b_odds", "a_opening_odds", "b_opening_odds"}
    ]
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid_nonblank = frame[column].map(lambda value: not value_is_missing(value)) & values.isna()
        if column.endswith(("_share", "_prob")):
            outside = values.notna() & ((values < 0) | (values > 1))
        elif column.endswith(("_acc", "_def")):
            outside = values.notna() & ((values < 0) | (values > 100))
        else:
            outside = values.notna() & (values <= 1)
        for position, is_invalid in enumerate(invalid_nonblank.to_numpy()):
            if not is_invalid:
                continue
            anomalies[
                (row_identity(rows[position]), column, "non_numeric")
            ] += 1
        for position, is_outside in enumerate(outside.to_numpy()):
            if not is_outside:
                continue
            anomalies[
                (row_identity(rows[position]), column, "out_of_range")
            ] += 1

    grouped = Counter()
    for (_identity, field, reason), count in anomalies.items():
        grouped[(field, reason)] += count
    details.extend(
        {"field": field, "reason": reason, "rows": count}
        for (field, reason), count in sorted(grouped.items())
    )
    return anomalies, details


def processed_value_report(
    frame: pd.DataFrame,
    *,
    accepted: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Check values while grandfathering only unchanged accepted anomalies."""

    required = ("event_date", "fighter_a", "fighter_b", "target")
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        return {
            "passed": False,
            "errors": [f"processed_missing_columns:{','.join(missing_columns)}"],
            "details": [],
            "notes": [],
        }

    candidate_anomalies, candidate_details = _processed_value_anomalies(frame)
    if accepted is None:
        reason_counts = Counter()
        numeric_groups = set()
        for (_identity, field, reason), count in candidate_anomalies.items():
            reason_counts[reason] += count
            if reason in {"non_numeric", "out_of_range"}:
                numeric_groups.add((field, reason))
        errors: list[str] = []
        if reason_counts["invalid_date"]:
            errors.append(f"processed_invalid_dates:{reason_counts['invalid_date']}")
        if reason_counts["invalid_participants"]:
            errors.append(
                "processed_invalid_participants:"
                f"{reason_counts['invalid_participants']}"
            )
        if reason_counts["invalid_binary_target"]:
            errors.append(
                "processed_invalid_binary_targets:"
                f"{reason_counts['invalid_binary_target']}"
            )
        if numeric_groups:
            errors.append(
                f"processed_numeric_range_violations:{len(numeric_groups)}"
            )
        return {
            "passed": not errors,
            "errors": errors,
            "details": candidate_details[:25],
            "notes": [],
        }

    accepted_anomalies, _accepted_details = _processed_value_anomalies(accepted)
    new_anomalies = candidate_anomalies - accepted_anomalies
    missing_anomalies = accepted_anomalies - candidate_anomalies
    grandfathered = accepted_anomalies & candidate_anomalies
    new_reason_counts = Counter()
    new_numeric_groups = set()
    for (_identity, field, reason), count in new_anomalies.items():
        new_reason_counts[reason] += count
        if reason in {"non_numeric", "out_of_range"}:
            new_numeric_groups.add((field, reason))

    errors: list[str] = []
    if new_reason_counts["invalid_date"]:
        errors.append(f"processed_invalid_dates:{new_reason_counts['invalid_date']}")
    if new_reason_counts["invalid_participants"]:
        errors.append(
            "processed_invalid_participants:"
            f"{new_reason_counts['invalid_participants']}"
        )
    if new_reason_counts["invalid_binary_target"]:
        errors.append(
            "processed_invalid_binary_targets:"
            f"{new_reason_counts['invalid_binary_target']}"
        )
    if new_numeric_groups:
        errors.append(
            f"processed_numeric_range_violations:{len(new_numeric_groups)}"
        )
    if missing_anomalies:
        errors.append(
            "processed_historical_value_anomalies_changed_or_removed:"
            f"{sum(missing_anomalies.values())}"
        )

    notes: list[dict[str, object]] = []
    if grandfathered:
        groups = Counter()
        for (_identity, field, reason), count in grandfathered.items():
            groups[(field, reason)] += count
        notes.append(
            {
                "reason": "grandfathered_legacy_value_anomalies",
                "count": sum(grandfathered.values()),
                "groups": [
                    {"field": field, "reason": reason, "rows": count}
                    for (field, reason), count in sorted(groups.items())
                ][:25],
            }
        )
    new_details = []
    new_groups = Counter()
    for (_identity, field, reason), count in new_anomalies.items():
        new_groups[(field, reason)] += count
    new_details.extend(
        {"field": field, "reason": reason, "rows": count}
        for (field, reason), count in sorted(new_groups.items())
    )
    return {
        "passed": not errors,
        "errors": errors,
        "details": new_details[:25],
        "notes": notes,
        "grandfathered_anomaly_count": sum(grandfathered.values()),
        "new_anomaly_count": sum(new_anomalies.values()),
    }


def processed_result_policy_report(frame: pd.DataFrame) -> dict[str, object]:
    """Ensure marked processed rows retain the approved unusual-result semantics."""

    if "result_policy_version" not in frame.columns:
        return {"passed": True, "errors": [], "rows_checked": 0, "details": []}
    details: list[dict[str, object]] = []
    rows_checked = 0
    for index, row in frame.iterrows():
        marker = _text(row.get("result_policy_version"))
        if not marker:
            continue
        rows_checked += 1
        result_class = _text(row.get("result_class")).casefold()
        target_missing = value_is_missing(row.get("target"))
        winner_missing = value_is_missing(row.get("winner"))
        reason = ""
        if marker != RESULT_POLICY_VERSION:
            reason = "unsupported_policy_marker"
        elif result_class == RESULT_CLASS_NORMAL and target_missing:
            reason = "normal_result_missing_binary_target"
        elif result_class == RESULT_CLASS_DQ and (not target_missing or winner_missing):
            reason = "dq_requires_winner_and_blank_binary_target"
        elif result_class in {RESULT_CLASS_DRAW, RESULT_CLASS_NO_CONTEST} and (
            not target_missing or not winner_missing
        ):
            reason = "draw_or_no_contest_requires_blank_winner_and_target"
        elif result_class not in {
            RESULT_CLASS_NORMAL,
            RESULT_CLASS_DQ,
            RESULT_CLASS_DRAW,
            RESULT_CLASS_NO_CONTEST,
        }:
            reason = "unknown_or_contradictory_result_class"
        if reason:
            details.append({"row": int(index), "result_class": result_class, "reason": reason})
    errors = (
        [f"processed_result_policy_violations:{len(details)}"] if details else []
    )
    return {
        "passed": not errors,
        "errors": errors,
        "rows_checked": rows_checked,
        "details": details[:25],
    }


def expected_evolved_schema(
    accepted: pd.DataFrame,
    *,
    appended_columns: Sequence[str],
    require_appended: bool,
) -> list[str]:
    expected = [str(column) for column in accepted.columns]
    if require_appended:
        expected.extend(column for column in appended_columns if column not in expected)
    return expected


def expected_processed_evolved_schema(
    accepted: pd.DataFrame,
    *,
    appended_columns: Sequence[str],
) -> list[str]:
    """Place new result/identity metadata at the processed builder's fixed slots."""
    expected = [str(column) for column in accepted.columns]
    missing = [column for column in appended_columns if column not in expected]
    outcome_columns = [
        column
        for column in (
            "official_outcome",
            "result_policy_version",
            "result_class",
            "binary_target_eligible",
            "result_quarantine_reason",
        )
        if column in missing
    ]
    identity_columns = [
        column for column in ("fighter_a_id", "fighter_b_id") if column in missing
    ]
    for anchor, columns in (("method", outcome_columns), ("fighter_b", identity_columns)):
        if not columns:
            continue
        position = expected.index(anchor) + 1 if anchor in expected else len(expected)
        expected[position:position] = columns
    positioned = set(outcome_columns) | set(identity_columns)
    expected.extend(column for column in missing if column not in positioned)
    return expected


def expected_active_roster_evolved_schema(
    accepted: pd.DataFrame,
) -> list[str]:
    """Place the first stable roster-provenance fields at their fixed slot."""

    expected = [
        str(column)
        for column in accepted.columns
        if column not in ACTIVE_ROSTER_CACHED_BIOGRAPHY_PROVENANCE_COLUMNS
    ]
    position = (
        expected.index("active_roster_status_reason") + 1
        if "active_roster_status_reason" in expected
        else len(expected)
    )
    expected[position:position] = list(
        ACTIVE_ROSTER_CACHED_BIOGRAPHY_PROVENANCE_COLUMNS
    )
    return expected


def publication_quality_report(
    *,
    accepted_results: pd.DataFrame,
    candidate_results: pd.DataFrame,
    accepted_statistics: pd.DataFrame,
    candidate_statistics: pd.DataFrame,
    accepted_roster: pd.DataFrame,
    candidate_roster: pd.DataFrame,
    accepted_profiles: pd.DataFrame,
    candidate_profiles: pd.DataFrame,
    accepted_processed: pd.DataFrame,
    candidate_processed: pd.DataFrame,
    accepted_features: pd.DataFrame,
    candidate_features: pd.DataFrame,
    coverage_snapshot: Mapping[str, object],
    coverage_floors: Mapping[str, float],
    profile_integrity_decisions: pd.DataFrame | None = None,
    fighter_ids_with_completed_fights: set[str] | None = None,
    event_date_lookup: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run the complete side-effect-free gate for one staged generation."""

    result_policy = incoming_result_policy_report(accepted_results, candidate_results)
    new_fight_keys = set(result_policy["new_fight_keys"])
    result_schema = exact_schema_report(
        candidate_results,
        expected_evolved_schema(
            accepted_results,
            appended_columns=RESULT_POLICY_COLUMNS,
            require_appended=bool(new_fight_keys),
        ),
        label="fight_results",
        reference_frame=accepted_results,
    )
    processed_appended_columns = (
        list(PROCESSED_RESULT_POLICY_COLUMNS)
        if new_fight_keys
        else [
            column
            for column in PROCESSED_RESULT_POLICY_COLUMNS
            if column in candidate_processed.columns
        ]
    )
    roster_schema_evolved = any(
        column in candidate_roster.columns
        for column in ACTIVE_ROSTER_CACHED_BIOGRAPHY_PROVENANCE_COLUMNS
    )
    reports: dict[str, Mapping[str, object]] = {
        "fight_results_schema": result_schema,
        "fight_statistics_schema": exact_schema_report(
            candidate_statistics,
            list(accepted_statistics.columns),
            label="fight_statistics",
            reference_frame=accepted_statistics,
        ),
        "active_roster_schema": exact_schema_report(
            candidate_roster,
            (
                expected_active_roster_evolved_schema(accepted_roster)
                if roster_schema_evolved
                else list(accepted_roster.columns)
            ),
            label="active_roster",
            reference_frame=accepted_roster,
        ),
        "fighter_profiles_schema": exact_schema_report(
            candidate_profiles,
            list(accepted_profiles.columns),
            label="fighter_profiles",
            reference_frame=accepted_profiles,
        ),
        "processed_schema": exact_schema_report(
            candidate_processed,
            expected_processed_evolved_schema(
                accepted_processed,
                appended_columns=processed_appended_columns,
            ),
            label="processed_fights",
            reference_frame=accepted_processed,
        ),
        "processed_features_schema": exact_schema_report(
            candidate_features,
            list(accepted_features.columns),
            label="processed_features",
            reference_frame=accepted_features,
        ),
        "active_roster_identities": roster_identity_report(
            candidate_roster,
            accepted=accepted_roster,
        ),
        "active_roster_immutable_fields": roster_immutable_field_report(
            accepted_roster,
            candidate_roster,
        ),
        "fight_result_identities": baseline_aware_duplicate_key_report(
            accepted_results,
            candidate_results,
            stable_raw_fight_key,
            label="fight_results",
        ),
        "fight_stat_identities": baseline_aware_duplicate_key_report(
            accepted_statistics,
            candidate_statistics,
            stable_raw_stat_key,
            label="fight_statistics",
        ),
        "fighter_profile_identities": baseline_aware_duplicate_key_report(
            accepted_profiles,
            candidate_profiles,
            stable_profile_key,
            label="fighter_profiles",
        ),
        "fighter_profile_integrity": profile_publication_integrity_report(
            accepted_profiles,
            candidate_profiles,
            profile_integrity_decisions,
        ),
        "processed_identities": baseline_aware_duplicate_key_report(
            accepted_processed,
            candidate_processed,
            stable_processed_fight_key,
            label="processed_fights",
        ),
        "processed_feature_identities": baseline_aware_duplicate_key_report(
            accepted_features,
            candidate_features,
            stable_processed_fight_key,
            label="processed_features",
        ),
        "historical_fight_results": historical_row_change_report(
            accepted_results,
            candidate_results,
            key_builder=stable_raw_fight_key,
            label="fight_results",
            columns=[column for column in RESULT_BASE_COLUMNS if column in accepted_results.columns],
        ),
        "historical_fight_statistics": historical_row_change_report(
            accepted_statistics,
            candidate_statistics,
            key_builder=stable_raw_stat_key,
            label="fight_statistics",
            columns=[column for column in STAT_COLUMNS if column in accepted_statistics.columns],
        ),
        "historical_processed_fights": historical_row_change_report(
            accepted_processed,
            candidate_processed,
            key_builder=stable_processed_fight_key,
            identity_alias_builder=stable_processed_name_date_key,
            label="processed_fights",
            columns=[
                column
                for column in accepted_processed.columns
                if column in candidate_processed.columns
            ],
        ),
        "historical_processed_features": historical_row_change_report(
            accepted_features,
            candidate_features,
            key_builder=stable_processed_fight_key,
            identity_alias_builder=stable_processed_name_date_key,
            label="processed_features",
            columns=[
                column
                for column in accepted_features.columns
                if column in candidate_features.columns
            ],
        ),
        "processed_appends_match_new_results": processed_append_result_report(
            accepted_processed=accepted_processed,
            candidate_processed=candidate_processed,
            candidate_results=candidate_results,
            new_fight_keys=new_fight_keys,
            event_date_lookup=event_date_lookup,
        ),
        "incoming_result_policy": result_policy,
        "new_fight_statistics": fight_statistics_report(
            candidate_results,
            candidate_statistics,
            new_fight_keys=new_fight_keys,
        ),
        "fighter_profile_bundles": profile_pull_report(
            accepted_profiles,
            candidate_profiles,
            fighter_ids_with_completed_fights=fighter_ids_with_completed_fights,
        ),
        "processed_values": processed_value_report(
            candidate_processed,
            accepted=accepted_processed,
        ),
        "processed_result_policy": processed_result_policy_report(candidate_processed),
        "coverage_floors": evaluate_coverage_floors(
            coverage_snapshot,
            coverage_floors,
        ),
    }
    combined = combine_quality_reports(reports)
    combined["new_fight_count"] = len(new_fight_keys)
    combined["new_fight_keys"] = sorted(new_fight_keys)
    combined["coverage_snapshot"] = dict(coverage_snapshot)
    combined["coverage_floors"] = dict(coverage_floors)
    combined["unusual_result_exclusions"] = result_policy.get(
        "binary_target_excluded_count", 0
    )
    return combined


def evaluate_coverage_floors(
    snapshot: Mapping[str, object],
    floors: Mapping[str, float],
) -> dict[str, object]:
    """Evaluate existing floors; equality passes and source-limited new gaps note."""

    source_limited_keys = {
        "new_fighter_reach_pct": "new_fighter_reach_source_limited_only",
        "new_fighter_stance_pct": "new_fighter_stance_source_limited_only",
        "new_fighter_full_physical_pct": "new_fighter_full_physical_source_limited_only",
    }
    failures: list[dict[str, object]] = []
    notes: list[dict[str, object]] = []
    for metric, floor in floors.items():
        value = snapshot.get(metric)
        alert_eligible = snapshot.get("new_fighter_rows_alert_eligible")
        if (
            metric.startswith("new_fighter_")
            and alert_eligible is not None
            and int(alert_eligible or 0) == 0
        ):
            notes.append(
                {
                    "metric": metric,
                    "value": value,
                    "floor": float(floor),
                    "reason": "no_alert_eligible_new_fighters",
                }
            )
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            failures.append({"metric": metric, "value": value, "floor": float(floor), "reason": "missing"})
            continue
        if not math.isfinite(numeric):
            failures.append({"metric": metric, "value": value, "floor": float(floor), "reason": "missing"})
            continue
        if numeric >= float(floor):
            continue
        source_flag = source_limited_keys.get(metric)
        if source_flag and snapshot.get(source_flag) is True:
            notes.append({"metric": metric, "value": numeric, "floor": float(floor), "reason": "source_limited"})
        else:
            failures.append({"metric": metric, "value": numeric, "floor": float(floor), "reason": "below_floor"})
    return {"passed": not failures, "failures": failures, "notes": notes}


def _pct_metric(metric: object) -> float | None:
    if not isinstance(metric, Mapping):
        return None
    try:
        value = float(metric.get("pct"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def coverage_snapshot_from_audit(
    audit: Mapping[str, object] | None,
    alert_summary: Mapping[str, object] | None,
) -> dict[str, object]:
    """Extract the established dashboard coverage metrics for publication use."""

    audit = dict(audit or {})
    alert_summary = dict(alert_summary or {})
    overall = audit.get("overall_summary") or {}
    source_limited = alert_summary.get("source_limited_missing_fields") or {}
    split_summary = (
        alert_summary.get("split_summary_stable_identity")
        or audit.get("split_summary_stable_identity")
        or alert_summary.get("split_summary_alias_aware")
        or alert_summary.get("split_summary_official_name")
        or audit.get("split_summary_alias_aware")
        or audit.get("split_summary_official_name")
        or {}
    )
    newly_added = split_summary.get("newly_added_active_roster") or {}
    new_context = alert_summary.get("newly_added_active_roster") or {}

    def source_metric(field: str) -> Mapping[str, object]:
        value = source_limited.get(field) or {}
        return value if isinstance(value, Mapping) else {}

    reach_source = source_metric("reach")
    stance_source = source_metric("stance")
    physical_source = source_metric("full_physical_bundle")
    return {
        "active_roster_rows": audit.get("active_roster_rows"),
        "overall_full_physical_pct": _pct_metric(
            overall.get("full_physical_bundle_present")
        ),
        "overall_reach_pct": _pct_metric(overall.get("reach_present")),
        "overall_stance_pct": _pct_metric(overall.get("stance_present")),
        "new_fighter_full_physical_pct": _pct_metric(
            newly_added.get("full_physical_bundle_present")
        ),
        "new_fighter_reach_pct": _pct_metric(newly_added.get("reach_present")),
        "new_fighter_stance_pct": _pct_metric(newly_added.get("stance_present")),
        "new_fighter_rows_total": new_context.get("rows_total"),
        "new_fighter_rows_alert_eligible": new_context.get("rows_alert_eligible"),
        "new_fighter_rows_in_grace": new_context.get("rows_in_grace"),
        "new_fighter_grace_days": alert_summary.get("new_fighter_grace_days"),
        "new_fighter_identity_match_method": alert_summary.get("identity_match_method"),
        "new_fighter_reach_rows_missing": reach_source.get("rows_missing"),
        "new_fighter_reach_rows_source_limited": reach_source.get("rows_source_limited"),
        "new_fighter_reach_source_limited_only": bool(
            reach_source.get("source_limited_only")
        ),
        "new_fighter_stance_rows_missing": stance_source.get("rows_missing"),
        "new_fighter_stance_rows_source_limited": stance_source.get("rows_source_limited"),
        "new_fighter_stance_source_limited_only": bool(
            stance_source.get("source_limited_only")
        ),
        "new_fighter_full_physical_rows_missing": physical_source.get("rows_missing"),
        "new_fighter_full_physical_rows_source_limited": physical_source.get(
            "rows_source_limited"
        ),
        "new_fighter_full_physical_source_limited_only": bool(
            physical_source.get("source_limited_only")
        ),
    }


def configured_coverage_floors(
    environ: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Load the existing configured floors, rejecting malformed percentages."""

    source = os.environ if environ is None else environ
    floors: dict[str, float] = {}
    for env_name, metric in COVERAGE_FLOOR_ENV_TO_METRIC.items():
        raw = str(source.get(env_name, "") or "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be a percentage, got {raw!r}") from exc
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"{env_name} must be between 0 and 100, got {raw!r}")
        floors[metric] = value
    return floors


def read_csv_required(path: Path | str, *, label: str) -> pd.DataFrame:
    artifact = Path(path)
    if not artifact.is_file():
        raise ValueError(f"{label}_missing:{artifact}")
    try:
        frame = pd.read_csv(artifact)
    except Exception as exc:
        raise ValueError(f"{label}_unreadable:{artifact}:{exc}") from exc
    if frame.empty:
        raise ValueError(f"{label}_empty:{artifact}")
    return frame


def combine_quality_reports(reports: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    errors: list[str] = []
    for name, report in reports.items():
        for error in report.get("errors", []) or []:
            errors.append(f"{name}:{error}")
        if report.get("passed") is False and not report.get("errors"):
            errors.append(f"{name}:failed")
    return {"passed": not errors, "errors": errors, "checks": dict(reports)}
