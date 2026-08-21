"""Small, dependency-free helpers for portable historical regressions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from itertools import groupby
import re
import unicodedata


class PortableContractError(ValueError):
    """Raised when a portable regression contract cannot be verified exactly."""


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_UFCSTATS_ID_RE = re.compile(
    r"^(?:"
    r"([0-9a-f]{16})"
    r"|(?:(?:https?://)?(?:www\.)?ufcstats\.com/)?/?fighter-details/"
    r"([0-9a-f]{16})/?(?:[?#][^\s]*)?"
    r")$",
    re.I,
)


def verify_sha256_bindings(
    actual: Mapping[str, str],
    expected: Mapping[str, str],
) -> dict[str, str]:
    """Verify a complete in-memory set of named SHA-256 bindings."""
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise PortableContractError(
            f"binding labels differ: missing={missing}, unexpected={unexpected}"
        )

    verified: dict[str, str] = {}
    for label in sorted(expected):
        expected_digest = expected[label]
        actual_digest = actual[label]
        if not isinstance(expected_digest, str) or not _SHA256_RE.fullmatch(
            expected_digest
        ):
            raise PortableContractError(f"invalid expected SHA-256 for {label}")
        if not isinstance(actual_digest, str) or not _SHA256_RE.fullmatch(
            actual_digest
        ):
            raise PortableContractError(f"invalid actual SHA-256 for {label}")
        if actual_digest != expected_digest:
            raise PortableContractError(
                f"SHA-256 mismatch for {label}: "
                f"expected {expected_digest}, got {actual_digest}"
            )
        verified[label] = actual_digest
    return verified


def _normalized_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _normalized_ufcstats_id(value: object) -> str | None:
    text = str(value or "").strip()
    if text.casefold() in {"", "nan", "none", "null", "n/a"}:
        return None
    match = _UFCSTATS_ID_RE.fullmatch(text)
    return (match.group(1) or match.group(2)).lower() if match else None


def fighter_identity_key(
    name: object,
    ufcstats_id: object = None,
    *,
    ambiguous_names: Iterable[str] = (),
) -> str:
    """Mirror the approved closure policy without importing recovery code."""
    normalized_name = _normalized_name(name)
    if not normalized_name:
        raise PortableContractError("fighter name is blank")
    ambiguous = {_normalized_name(value) for value in ambiguous_names}
    if normalized_name not in ambiguous:
        return f"name:{normalized_name}"

    normalized_id = _normalized_ufcstats_id(ufcstats_id)
    if normalized_id is None:
        raise PortableContractError(
            f"ambiguous fighter {normalized_name!r} requires a UFCStats ID"
        )
    return f"ufcstats:{normalized_id}"


def _event_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        raise PortableContractError("event date is blank")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise PortableContractError(f"invalid event date {text!r}") from exc


def fight_physical_key(row: Mapping[str, object]) -> str:
    """Return the name-based physical-fight key used by the historical audit."""
    pair = sorted((_normalized_name(row.get("fighter_a")), _normalized_name(row.get("fighter_b"))))
    if not all(pair):
        raise PortableContractError("physical fight key requires two fighter names")
    return f"{_event_date(row.get('event_date')).isoformat()}|name:{pair[0]}|name:{pair[1]}"


def strict_date_causal_closure(
    rows: Iterable[Mapping[str, object]],
    added_physical_keys: Iterable[str],
    *,
    ambiguous_names: Iterable[str] = (),
) -> frozenset[str]:
    """Find retained fights reachable through strictly earlier affected history."""
    materialized: list[tuple[date, int, str, tuple[str, str]]] = []
    seen_physical_keys: set[str] = set()
    ambiguous = tuple(ambiguous_names)
    for position, row in enumerate(rows):
        physical_key = fight_physical_key(row)
        if physical_key in seen_physical_keys:
            raise PortableContractError(f"duplicate physical fight key {physical_key}")
        seen_physical_keys.add(physical_key)
        identities = tuple(
            fighter_identity_key(
                row.get(f"fighter_{side}"),
                row.get(f"fighter_{side}_id"),
                ambiguous_names=ambiguous,
            )
            for side in ("a", "b")
        )
        materialized.append((_event_date(row.get("event_date")), position, physical_key, identities))

    seeds = set(added_physical_keys)
    absent_seeds = sorted(seeds - seen_physical_keys)
    if absent_seeds:
        raise PortableContractError(f"added fight keys are absent: {absent_seeds}")

    affected_since: dict[str, date] = {}
    closure: set[str] = set()
    materialized.sort(key=lambda item: (item[0], item[1]))
    for event_day, day_rows in groupby(materialized, key=lambda item: item[0]):
        newly_affected: set[str] = set()
        for _, _, physical_key, identities in day_rows:
            if physical_key in seeds:
                newly_affected.update(identities)
                continue
            if any(
                identity in affected_since and affected_since[identity] < event_day
                for identity in identities
            ):
                closure.add(physical_key)
                newly_affected.update(identities)
        for identity in newly_affected:
            affected_since.setdefault(identity, event_day)
    return frozenset(closure)
