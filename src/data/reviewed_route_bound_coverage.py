"""Exact, audit-only certificates for route-bound UFC.com profile coverage.

These records close a coverage-accounting gap only.  They are not name aliases,
do not populate training data, and cannot authorize profile-field backfill.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

from src.data.name_utils import normalize_ufcstats_id
from src.data.profile_integrity import source_entity_id


REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "ufc_route_bound_profile_coverage_v1.json"
)
_EXPECTED_SCOPE = "active_roster_profile_coverage_only"
_EXPECTED_BINDING_METHOD = (
    "exact_official_route_plus_unique_ufcstats_performance_fingerprint_plus_"
    "official_result_and_exact_fight_anchor"
)
_TOP_LEVEL_KEYS = {
    "schema_version",
    "scope",
    "model_backfill_allowed",
    "reviewed_at_utc",
    "binding_method",
    "performance_fingerprint_population",
    "certificates",
}
_CERTIFICATE_KEYS = {
    "certificate_id",
    "official_name",
    "official_athlete_url",
    "official_source_entity_id",
    "ufcstats_name",
    "stable_ufcstats_id",
    "ufcstats_url",
    "profile_field",
    "normalized_value_inches",
    "performance_fingerprint",
    "fight_anchor",
    "legacy_official_url",
}
_FINGERPRINT_KEYS = {"slpm", "sapm", "td_avg", "sub_avg"}
_FIGHT_ANCHOR_KEYS = {
    "opponent",
    "event_date",
    "fight_id",
    "ufcstats_fight_url",
    "official_result_url",
}


def _clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(value != value):
            return ""
    except (TypeError, ValueError):
        pass
    if not isinstance(value, str) and str(value).strip().casefold() in {
        "nan",
        "<na>",
    }:
        return ""
    return str(value).strip()


def _exact_ufc_athlete_url(value: object) -> str:
    raw = _clean(value).rstrip("/")
    parsed = urlparse(raw)
    if (
        parsed.scheme.casefold() != "https"
        or (parsed.hostname or "").casefold().rstrip(".") != "www.ufc.com"
        or re.fullmatch(r"/athlete/[a-z0-9-]+", parsed.path) is None
        or parsed.query
        or parsed.fragment
    ):
        return ""
    return raw


def _exact_ufcstats_url(value: object, *, kind: str) -> str:
    raw = _clean(value).rstrip("/")
    parsed = urlparse(raw)
    if (
        parsed.scheme.casefold() != "http"
        or (parsed.hostname or "").casefold().rstrip(".") != "ufcstats.com"
        or re.fullmatch(fr"/{kind}/[0-9a-f]{{16}}", parsed.path, re.I) is None
        or parsed.query
        or parsed.fragment
    ):
        return ""
    return raw


def _exact_official_evidence_url(value: object) -> str:
    raw = _clean(value)
    parsed = urlparse(raw)
    if (
        parsed.scheme.casefold() != "https"
        or (parsed.hostname or "").casefold().rstrip(".") != "www.ufc.com"
        or not parsed.path.startswith("/")
        or parsed.fragment
    ):
        return ""
    return raw


def _positive_finite_number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("coverage certificate value must be numeric") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError("coverage certificate value must be finite and positive")
    return number


def _nonnegative_finite_number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("coverage fingerprint value must be numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError("coverage fingerprint value must be finite and nonnegative")
    return number


def _explicit_true(value: object) -> bool:
    if value is True:
        return True
    return _clean(value).casefold() in {"true", "yes", "1", "1.0"}


def _reach_inches(value: object) -> float | None:
    text = _clean(value)
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(?:in|\")", text, re.I)
    if match is None:
        return None
    number = float(match.group(1))
    return number if math.isfinite(number) and number > 0 else None


def _validate_certificate(record: object) -> dict[str, object]:
    if not isinstance(record, dict) or set(record) != _CERTIFICATE_KEYS:
        raise ValueError("coverage certificate has an unexpected schema")
    certificate_id = _clean(record["certificate_id"])
    if re.fullmatch(r"COV-R[0-9]{3}", certificate_id) is None:
        raise ValueError("coverage certificate ID is invalid")

    official_name = _clean(record["official_name"])
    official_url = _exact_ufc_athlete_url(record["official_athlete_url"])
    official_entity = _clean(record["official_source_entity_id"])
    if (
        not official_name
        or not official_url
        or source_entity_id("ufc", official_url) != official_entity
    ):
        raise ValueError(f"{certificate_id}: invalid official-route owner")

    ufcstats_name = _clean(record["ufcstats_name"])
    stable_id = _clean(record["stable_ufcstats_id"]).casefold()
    ufcstats_url = _exact_ufcstats_url(record["ufcstats_url"], kind="fighter-details")
    if (
        not ufcstats_name
        or re.fullmatch(r"[0-9a-f]{16}", stable_id) is None
        or not ufcstats_url
        or normalize_ufcstats_id(ufcstats_url) != stable_id
    ):
        raise ValueError(f"{certificate_id}: invalid stable UFCStats owner")
    if record["profile_field"] != "reach":
        raise ValueError(f"{certificate_id}: only reach coverage is authorized")
    normalized_inches = _positive_finite_number(record["normalized_value_inches"])

    fingerprint = record["performance_fingerprint"]
    if not isinstance(fingerprint, dict) or set(fingerprint) != _FINGERPRINT_KEYS:
        raise ValueError(f"{certificate_id}: invalid performance fingerprint")
    validated_fingerprint = {
        key: _nonnegative_finite_number(fingerprint[key])
        for key in sorted(_FINGERPRINT_KEYS)
    }

    anchor = record["fight_anchor"]
    if not isinstance(anchor, dict) or set(anchor) != _FIGHT_ANCHOR_KEYS:
        raise ValueError(f"{certificate_id}: invalid fight anchor")
    opponent = _clean(anchor["opponent"])
    try:
        date.fromisoformat(_clean(anchor["event_date"]))
    except ValueError as exc:
        raise ValueError(f"{certificate_id}: invalid anchor date") from exc
    fight_id = _clean(anchor["fight_id"]).casefold()
    fight_url = _exact_ufcstats_url(
        anchor["ufcstats_fight_url"],
        kind="fight-details",
    )
    official_result_url = _exact_official_evidence_url(
        anchor["official_result_url"]
    )
    if (
        not opponent
        or re.fullmatch(r"[0-9a-f]{16}", fight_id) is None
        or not fight_url
        or fight_url.rsplit("/", 1)[-1].casefold() != fight_id
        or not official_result_url
    ):
        raise ValueError(f"{certificate_id}: incomplete fight evidence")

    legacy_url = _clean(record["legacy_official_url"])
    if legacy_url and not _exact_official_evidence_url(legacy_url):
        raise ValueError(f"{certificate_id}: invalid legacy official URL")

    return {
        **record,
        "certificate_id": certificate_id,
        "official_name": official_name,
        "official_athlete_url": official_url,
        "official_source_entity_id": official_entity,
        "ufcstats_name": ufcstats_name,
        "stable_ufcstats_id": stable_id,
        "ufcstats_url": ufcstats_url,
        "normalized_value_inches": normalized_inches,
        "performance_fingerprint": validated_fingerprint,
        "fight_anchor": {
            **anchor,
            "opponent": opponent,
            "fight_id": fight_id,
            "ufcstats_fight_url": fight_url,
            "official_result_url": official_result_url,
        },
        "legacy_official_url": legacy_url,
    }


@lru_cache(maxsize=1)
def load_route_bound_coverage_registry() -> dict[str, object]:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != _TOP_LEVEL_KEYS:
        raise ValueError("route-bound coverage registry has an unexpected schema")
    if payload["schema_version"] != 1 or payload["scope"] != _EXPECTED_SCOPE:
        raise ValueError("route-bound coverage registry version or scope is invalid")
    if payload["model_backfill_allowed"] is not False:
        raise ValueError("route-bound coverage certificates cannot authorize backfill")
    if payload["binding_method"] != _EXPECTED_BINDING_METHOD:
        raise ValueError("route-bound coverage binding method is invalid")
    if payload["performance_fingerprint_population"] != 4512:
        raise ValueError("route-bound coverage fingerprint population is invalid")
    reviewed_at = _clean(payload["reviewed_at_utc"])
    try:
        parsed_reviewed_at = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("route-bound coverage reviewed timestamp is invalid") from exc
    if parsed_reviewed_at.tzinfo is None:
        raise ValueError("route-bound coverage reviewed timestamp must be timezone-aware")

    records = payload["certificates"]
    if not isinstance(records, list) or len(records) != 7:
        raise ValueError("route-bound coverage registry must contain exactly seven records")
    validated = [_validate_certificate(record) for record in records]
    for key in (
        "certificate_id",
        "official_name",
        "official_source_entity_id",
        "stable_ufcstats_id",
    ):
        values = [str(record[key]).casefold() for record in validated]
        if len(values) != len(set(values)):
            raise ValueError(f"route-bound coverage registry reuses {key}")
    return {**payload, "certificates": validated}


def route_bound_coverage_certificate(
    row: Mapping[str, object],
    field: str,
) -> dict[str, object] | None:
    """Return the one exact audit certificate matching this current source cell."""

    if field != "reach":
        return None
    official_url = _exact_ufc_athlete_url(row.get("official_athlete_url"))
    canonical_url = _exact_ufc_athlete_url(row.get("canonical_athlete_url"))
    ufcstats_url = _exact_ufcstats_url(
        row.get("ufcstats_url"),
        kind="fighter-details",
    )
    reach_inches = _reach_inches(row.get("reach"))
    if (
        not official_url
        or canonical_url != official_url
        or not ufcstats_url
        or reach_inches is None
        or _clean(row.get("official_profile_fetch_outcome")).casefold()
        != "succeeded"
        or not _explicit_true(row.get("official_url_identity_valid"))
        or _clean(row.get("official_url_identity_status")).casefold() != "valid"
    ):
        return None

    matches = []
    for certificate in load_route_bound_coverage_registry()["certificates"]:
        stable_carriers = []
        invalid_stable_carrier = False
        for carrier_field in (
            "stable_ufcstats_id",
            "fighter_id",
            "ufcstats_url",
        ):
            carrier = _clean(row.get(carrier_field))
            if not carrier:
                continue
            stable_id = normalize_ufcstats_id(carrier)
            if stable_id is None:
                invalid_stable_carrier = True
                break
            stable_carriers.append(stable_id)

        optional_names_match = all(
            not (value := _clean(row.get(field)))
            or value in {
                str(certificate["official_name"]),
                str(certificate["official_name"]).casefold(),
            }
            for field in ("profile_name", "slug_name")
        )
        optional_entities_match = all(
            not (value := _clean(row.get(field)))
            or value == certificate["official_source_entity_id"]
            for field in ("official_source_entity_id",)
        )
        optional_fresh_urls_match = all(
            not (value := _clean(row.get(field)))
            or _exact_ufc_athlete_url(value)
            == certificate["official_athlete_url"]
            for field in (
                "official_profile_fresh_requested_url",
                "official_profile_fresh_effective_url",
                "official_profile_fresh_canonical_url",
            )
        )
        fresh_redirect_chain = _clean(
            row.get("official_profile_fresh_redirect_chain")
        )
        fresh_redirects_match = True
        if fresh_redirect_chain:
            try:
                fresh_redirects_match = json.loads(fresh_redirect_chain) == []
            except (TypeError, ValueError):
                fresh_redirects_match = False
        binding_fields_present = bool(
            _clean(row.get("official_profile_identity_binding_status"))
            or any(
                _clean(value)
                for field, value in row.items()
                if str(field).startswith("official_profile_binding_")
            )
        )
        if (
            not invalid_stable_carrier
            and stable_carriers
            and set(stable_carriers)
            == {certificate["stable_ufcstats_id"]}
            and _clean(row.get("official_name"))
            == certificate["official_name"]
            and official_url == certificate["official_athlete_url"]
            and source_entity_id("ufc", official_url)
            == certificate["official_source_entity_id"]
            and _clean(row.get("ufcstats_name")) == certificate["ufcstats_name"]
            and normalize_ufcstats_id(ufcstats_url)
            == certificate["stable_ufcstats_id"]
            and ufcstats_url == certificate["ufcstats_url"]
            and optional_names_match
            and optional_entities_match
            and optional_fresh_urls_match
            and fresh_redirects_match
            and not binding_fields_present
            and _clean(row.get("combat_sport")).casefold() == "mma"
            and math.isclose(
                reach_inches,
                float(certificate["normalized_value_inches"]),
                abs_tol=1e-9,
            )
        ):
            matches.append(certificate)
    if len(matches) > 1:
        raise ValueError("route-bound coverage cell matched multiple certificates")
    return dict(matches[0]) if matches else None
