"""Narrow, owner-reviewed roster recovery decisions.

The registry is intentionally keyed by stable source entities. Names are only
checked as reviewed display evidence and are never sufficient to select a row.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import unquote, urlparse

from src.data.name_utils import normalize_person_name, normalize_ufcstats_id
from src.data.profile_integrity import source_entity_id


REVIEWED_ROSTER_RECOVERY_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "ufc_roster_recovery_decisions_v1.json"
)
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "canonical_queue_sha256",
        "decision_report_sha256",
        "permanent_profile_locks",
        "profile_field_holds",
        "unbound_simultaneous_route_quarantines",
        "non_mma_identity_exclusions",
        "mma_identity_bindings",
        "official_identity_transitions",
        "profile_recoveries",
    }
)
_PERMANENT_PROFILE_LOCK_KEYS = frozenset(
    {
        "canonical_name",
        "stable_ufcstats_id",
        "ufcstats_url",
        "field",
        "normalized_value",
        "artifact_value",
        "rejected_value",
        "decision_status",
        "automatic_replacement_allowed",
        "evidence_sha256",
    }
)
_PROFILE_FIELD_HOLD_KEYS = frozenset(
    {
        "canonical_name",
        "reviewed_names",
        "stable_ufcstats_id",
        "ufcstats_url",
        "field",
        "conflicting_normalized_values",
        "queue_index",
        "decision_status",
        "automatic_release_allowed",
        "scope",
        "historical_backfill_allowed",
        "reason",
        "evidence_urls",
        "evidence_sha256",
    }
)
_UNBOUND_SIMULTANEOUS_ROUTE_QUARANTINE_KEYS = frozenset(
    {
        "official_name",
        "alternate_official_athlete_url",
        "alternate_official_source_entity_id",
        "alternate_observed_profile",
        "companion_official_athlete_url",
        "companion_official_source_entity_id",
        "companion_stable_ufcstats_id",
        "companion_ufcstats_url",
        "automatic_alias_allowed",
        "automatic_binding_allowed",
        "automatic_redirect_allowed",
        "destructive_exclusion_allowed",
        "reason",
        "evidence_urls",
    }
)
_UNBOUND_ALTERNATE_OBSERVED_PROFILE_KEYS = frozenset(
    {
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
    }
)
_NON_MMA_KEYS = frozenset(
    {
        "official_name",
        "official_athlete_url",
        "official_source_entity_id",
        "reason",
        "queue_indices",
        "evidence_entity_ids",
        "evidence_urls",
    }
)
_MMA_BINDING_KEYS = frozenset(
    {
        "official_name",
        "reviewed_official_names",
        "official_athlete_url",
        "official_source_entity_id",
        "ufcstats_name",
        "stable_ufcstats_id",
        "ufcstats_url",
        "queue_indices",
        "identity_basis",
        "evidence_urls",
    }
)
_OFFICIAL_IDENTITY_TRANSITION_KEYS = frozenset(
    {
        "source_official_name",
        "source_official_athlete_url",
        "source_official_source_entity_id",
        "source_canonical_athlete_url",
        "source_canonical_evidence_required",
        "target_official_name",
        "target_official_athlete_url",
        "target_official_source_entity_id",
        "target_canonical_athlete_url",
        "target_canonical_evidence_required",
        "ufcstats_name",
        "stable_ufcstats_id",
        "ufcstats_url",
        "direction",
        "automatic_reverse_allowed",
        "identity_basis",
        "evidence_urls",
        "fight_anchors",
    }
)
_OFFICIAL_IDENTITY_TRANSITION_FIGHT_ANCHOR_KEYS = frozenset(
    {
        "event_date",
        "opponent_name",
        "opponent_ufcstats_id",
        "fight_url",
    }
)
_PROFILE_RECOVERY_KEYS = frozenset(
    {
        "canonical_name",
        "reviewed_source_names",
        "official_athlete_url",
        "official_source_entity_id",
        "stable_ufcstats_id",
        "ufcstats_url",
        "field",
        "value",
        "source",
        "source_name",
        "source_entity_id",
        "source_url",
        "requested_url",
        "effective_url",
        "canonical_url",
        "redirect_chain",
        "retrieval_route",
        "retrieved_at_utc",
        "observation_id",
        "capture_sha256",
        "queue_index",
        "scope",
        "historical_backfill_allowed",
        "decision_status",
        "identity_basis",
        "evidence_urls",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_VALID_STANCES = frozenset({"Orthodox", "Southpaw", "Switch", "Open Stance"})
_PINNED_OFFICIAL_IDENTITY_TRANSITIONS = {
    "bbc9ab95fa766493": {
        "source_official_name": "Anderson Berinja",
        "source_official_athlete_url": "https://www.ufc.com/athlete/anderson-berinja",
        "source_official_source_entity_id": "athlete:anderson-berinja",
        "source_canonical_athlete_url": "https://www.ufc.com/athlete/anderson-berinja",
        "source_canonical_evidence_required": False,
        "target_official_name": "Anderson dos Santos",
        "target_official_athlete_url": "https://www.ufc.com/athlete/anderson-dos-santos",
        "target_official_source_entity_id": "athlete:anderson-dos-santos",
        "target_canonical_athlete_url": "https://www.ufc.com/athlete/anderson-dos-santos",
        "target_canonical_evidence_required": False,
        "ufcstats_name": "Anderson Dos Santos",
        "ufcstats_url": "http://ufcstats.com/fighter-details/bbc9ab95fa766493",
        "evidence_urls": frozenset(
            {
                "https://www.ufc.com/athlete/anderson-berinja",
                "https://www.ufc.com/athlete/anderson-dos-santos",
                "http://ufcstats.com/fighter-details/bbc9ab95fa766493",
                "http://ufcstats.com/fight-details/9a678f346946b6a4",
                "http://ufcstats.com/fight-details/0c27552ea7bc8c8a",
                "http://ufcstats.com/fight-details/8f1b727757a60c07",
                "http://ufcstats.com/fight-details/c7bc8750e9503805",
                "https://www.ufc.com/news/blaydes-vs-lewis-fight-fight-preview-ufc-vegas-15",
                "https://www.ufc.com/news/fight-by-fight-preview-ufc-265-lewis-vs-gane-card-breakdown",
            }
        ),
        "fight_anchors": frozenset(
            {
                ("2021-08-07", "miles johns", "46e0d677f91bacc0", "http://ufcstats.com/fight-details/9a678f346946b6a4"),
                ("2020-11-28", "martin day", "ea3dcc417c322064", "http://ufcstats.com/fight-details/0c27552ea7bc8c8a"),
                ("2019-06-22", "andre ewell", "5f6f049d54728e88", "http://ufcstats.com/fight-details/8f1b727757a60c07"),
                ("2018-11-17", "nad narimani", "776a58a67665a171", "http://ufcstats.com/fight-details/c7bc8750e9503805"),
            }
        ),
    },
    "07047eb7d17fb0a2": {
        "source_official_name": "Daria Zhelezniakova",
        "source_official_athlete_url": "https://www.ufc.com/athlete/daria-zheleznyakova",
        "source_official_source_entity_id": "athlete:daria-zheleznyakova",
        "source_canonical_athlete_url": "https://www.ufc.com/athlete/dariya-zheleznyakova",
        "source_canonical_evidence_required": True,
        "target_official_name": "Daria Zhelezniakova",
        "target_official_athlete_url": "https://www.ufc.com/athlete/dariya-zheleznyakova",
        "target_official_source_entity_id": "athlete:dariya-zheleznyakova",
        "target_canonical_athlete_url": "https://www.ufc.com/athlete/dariya-zheleznyakova",
        "target_canonical_evidence_required": False,
        "ufcstats_name": "Daria Zhelezniakova",
        "ufcstats_url": "http://ufcstats.com/fighter-details/07047eb7d17fb0a2",
        "evidence_urls": frozenset(
            {
                "https://www.ufc.com/athlete/daria-zheleznyakova",
                "https://www.ufc.com/athlete/dariya-zheleznyakova",
                "http://ufcstats.com/fighter-details/07047eb7d17fb0a2",
                "http://ufcstats.com/fight-details/f1c3abce15ef053e",
                "http://ufcstats.com/fight-details/7ba432e218edc17d",
                "http://ufcstats.com/fight-details/fff3218758c513b8",
            }
        ),
        "fight_anchors": frozenset(
            {
                ("2024-03-23", "montserrat rendon", "60193e707634e560", "http://ufcstats.com/fight-details/f1c3abce15ef053e"),
                ("2024-09-28", "ailin perez", "06e4245d16fc5315", "http://ufcstats.com/fight-details/7ba432e218edc17d"),
                ("2025-06-21", "melissa mullins", "bd532e9a2b93870c", "http://ufcstats.com/fight-details/fff3218758c513b8"),
            }
        ),
    },
    "3618e0632ecdba9c": {
        "source_official_name": "Keiichiro Nakamura",
        "source_official_athlete_url": "https://www.ufc.com/athlete/keiichiro-nakamura",
        "source_official_source_entity_id": "athlete:keiichiro-nakamura",
        "source_canonical_athlete_url": "https://www.ufc.com/athlete/keiichiro-nakamura",
        "source_canonical_evidence_required": True,
        "target_official_name": "Keiichiro Nakamura",
        "target_official_athlete_url": "https://www.ufc.com/athlete/keiichiro-nakamura-0",
        "target_official_source_entity_id": "athlete:keiichiro-nakamura-0",
        "target_canonical_athlete_url": "https://www.ufc.com/athlete/keiichiro-nakamura-0",
        "target_canonical_evidence_required": False,
        "ufcstats_name": "Keiichiro Nakamura",
        "ufcstats_url": "http://ufcstats.com/fighter-details/3618e0632ecdba9c",
        "evidence_urls": frozenset(
            {
                "https://www.ufc.com/athlete/keiichiro-nakamura",
                "https://www.ufc.com/athlete/keiichiro-nakamura-0",
                "http://ufcstats.com/fighter-details/3618e0632ecdba9c",
                "http://ufcstats.com/fight-details/780bae82bdfc7112",
                "http://ufcstats.com/fight-details/28648815e5f46bd9",
            }
        ),
        "fight_anchors": frozenset(
            {
                ("2026-01-31", "sebastian szalay", "f201f21cf2c79eb2", "http://ufcstats.com/fight-details/780bae82bdfc7112"),
                ("2025-08-22", "kaiwen", "0565706b1e7f7e7c", "http://ufcstats.com/fight-details/28648815e5f46bd9"),
            }
        ),
    },
    "4948e412fa6ab67d": {
        "source_official_name": "Ravena Oliveira",
        "source_official_athlete_url": "https://www.ufc.com/athlete/ravena-oliveira",
        "source_official_source_entity_id": "athlete:ravena-oliveira",
        "source_canonical_athlete_url": "https://www.ufc.com/athlete/ravena-oliveira",
        "source_canonical_evidence_required": True,
        "target_official_name": "Ravena Oliveira",
        "target_official_athlete_url": "https://www.ufc.com/athlete/ravena-oliveira-morais",
        "target_official_source_entity_id": "athlete:ravena-oliveira-morais",
        "target_canonical_athlete_url": "https://www.ufc.com/athlete/ravena-oliveira-morais",
        "target_canonical_evidence_required": False,
        "ufcstats_name": "Ravena Oliveira",
        "ufcstats_url": "http://ufcstats.com/fighter-details/4948e412fa6ab67d",
        "evidence_urls": frozenset(
            {
                "https://www.ufc.com/athlete/ravena-oliveira",
                "https://www.ufc.com/athlete/ravena-oliveira-morais",
                "http://ufcstats.com/fighter-details/4948e412fa6ab67d",
                "http://ufcstats.com/fight-details/5b7b5b01f4dcbd9c",
                "http://ufcstats.com/fight-details/8d9458121f0c4b95",
                "http://ufcstats.com/fight-details/216bc9b661485017",
                "http://ufcstats.com/fight-details/d463229133909215",
            }
        ),
        "fight_anchors": frozenset(
            {
                ("2023-10-14", "tainara lisboa", "2c5f5749569eef66", "http://ufcstats.com/fight-details/5b7b5b01f4dcbd9c"),
                ("2025-10-18", "stephanie luciano", "1b35af4d1529adf6", "http://ufcstats.com/fight-details/8d9458121f0c4b95"),
                ("2026-03-21", "shanelle dyer", "cfbb3edab781dfd2", "http://ufcstats.com/fight-details/216bc9b661485017"),
                ("2026-08-08", "juliana miller", "aa0c573da7119292", "http://ufcstats.com/fight-details/d463229133909215"),
            }
        ),
    },
}


def _text(value: object) -> str:
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


def _row_has_power_slap_evidence(row: Mapping[str, object]) -> bool:
    """Mirror the active-roster three-carrier Power Slap hard block."""

    evidence = " ".join(
        _text(row.get(field))
        for field in (
            "combat_sport",
            "combat_sport_reason",
            "combat_sport_profile_url",
        )
    ).casefold()
    compact = "".join(character for character in evidence if character.isalnum())
    return "powerslap" in compact


def _require_exact_keys(record: Mapping[str, object], keys: frozenset[str], label: str) -> None:
    actual = frozenset(record)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ValueError(
            f"{label} has an invalid schema (missing={missing}, extra={extra})"
        )


def _require_nonempty_text(record: Mapping[str, object], fields: Sequence[str], label: str) -> None:
    missing = [field for field in fields if not _text(record.get(field))]
    if missing:
        raise ValueError(f"{label} has blank required fields: {missing}")


def _require_string_list(value: object, label: str, *, minimum: int = 1) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(f"{label} must contain at least {minimum} item(s)")
    items = [_text(item) for item in value]
    if any(not item for item in items) or len(items) != len(set(items)):
        raise ValueError(f"{label} must contain unique nonblank strings")
    return items


def _require_queue_indices(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must contain queue indices")
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 1 or item > 118 for item in value):
        raise ValueError(f"{label} contains an invalid queue index")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicate queue indices")
    return list(value)


def _validated_transition_anchor_set(value: object, label: str) -> frozenset[tuple[str, str, str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must contain fight anchors")
    anchors: list[tuple[str, str, str, str]] = []
    for index, raw_anchor in enumerate(value, start=1):
        anchor_label = f"{label} fight anchor {index}"
        if not isinstance(raw_anchor, dict):
            raise ValueError(f"{anchor_label} must be an object")
        _require_exact_keys(
            raw_anchor,
            _OFFICIAL_IDENTITY_TRANSITION_FIGHT_ANCHOR_KEYS,
            anchor_label,
        )
        _require_nonempty_text(
            raw_anchor,
            ("event_date", "opponent_name", "opponent_ufcstats_id", "fight_url"),
            anchor_label,
        )
        event_date = _text(raw_anchor["event_date"])
        opponent_name = normalize_person_name(raw_anchor["opponent_name"])
        opponent_id = normalize_ufcstats_id(raw_anchor["opponent_ufcstats_id"])
        fight_url = _text(raw_anchor["fight_url"])
        parsed_fight = urlparse(fight_url)
        if (
            re.fullmatch(r"\d{4}-\d{2}-\d{2}", event_date) is None
            or not opponent_name
            or not opponent_id
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
            raise ValueError(f"{anchor_label} is inconsistent")
        anchors.append((event_date, opponent_name, opponent_id, fight_url))
    if len(anchors) != len(set(anchors)):
        raise ValueError(f"{label} contains duplicate fight anchors")
    return frozenset(anchors)


def _wikipedia_entity_id(value: object) -> str:
    parsed = urlparse(_text(value))
    if (
        parsed.scheme.casefold() != "https"
        or (parsed.hostname or "").casefold().rstrip(".") != "en.wikipedia.org"
        or parsed.query
        or parsed.fragment
    ):
        return ""
    match = re.fullmatch(r"/wiki/([^/]+)", parsed.path)
    return f"page-title:{unquote(match.group(1))}" if match else ""


def _validate_registry(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("Reviewed roster recovery registry must be a JSON object")
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "reviewed roster recovery registry")
    if payload.get("schema_version") != 1:
        raise ValueError("Reviewed roster recovery registry schema_version must be 1")
    for field in ("canonical_queue_sha256", "decision_report_sha256"):
        value = _text(payload.get(field)).casefold()
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"Reviewed roster recovery registry has invalid {field}")

    exclusions = payload.get("non_mma_identity_exclusions")
    bindings = payload.get("mma_identity_bindings")
    transitions = payload.get("official_identity_transitions")
    recoveries = payload.get("profile_recoveries")
    locks = payload.get("permanent_profile_locks")
    holds = payload.get("profile_field_holds")
    route_quarantines = payload.get("unbound_simultaneous_route_quarantines")
    if not isinstance(locks, list) or len(locks) != 1:
        raise ValueError("Permanent profile lock registry must contain exactly 1 record")
    if not isinstance(holds, list) or len(holds) != 1:
        raise ValueError("Profile field hold registry must contain exactly 1 record")
    if not isinstance(route_quarantines, list) or len(route_quarantines) != 1:
        raise ValueError(
            "Unbound simultaneous-route quarantine registry must contain exactly 1 record"
        )
    if not isinstance(exclusions, list) or len(exclusions) != 3:
        raise ValueError("Reviewed non-MMA exclusion registry must contain exactly 3 records")
    if not isinstance(bindings, list) or len(bindings) != 13:
        raise ValueError("Reviewed MMA identity registry must contain exactly 13 records")
    expected_transition_count = len(_PINNED_OFFICIAL_IDENTITY_TRANSITIONS)
    if (
        not isinstance(transitions, list)
        or len(transitions) != expected_transition_count
    ):
        raise ValueError(
            "Reviewed official identity transition registry must contain exactly "
            f"{expected_transition_count} records"
        )
    if not isinstance(recoveries, list) or len(recoveries) != 1:
        raise ValueError("Reviewed profile recovery registry must contain exactly 1 record")

    lock = locks[0]
    if not isinstance(lock, dict):
        raise ValueError("Permanent profile lock 1 must be an object")
    _require_exact_keys(lock, _PERMANENT_PROFILE_LOCK_KEYS, "permanent profile lock 1")
    _require_nonempty_text(
        lock,
        (
            "canonical_name",
            "stable_ufcstats_id",
            "ufcstats_url",
            "field",
            "normalized_value",
            "artifact_value",
            "rejected_value",
            "decision_status",
            "evidence_sha256",
        ),
        "permanent profile lock 1",
    )
    if (
        lock["canonical_name"] != "Ethyn Ewing"
        or lock["stable_ufcstats_id"] != "d776b5814f2400b4"
        or lock["field"] != "dob"
        or lock["normalized_value"] != "1998-02-16"
        or lock["artifact_value"] != "Feb 16, 1998"
        or lock["rejected_value"] != "1996-02-19"
        or lock["decision_status"] != "owner_locked"
        or lock["automatic_replacement_allowed"] is not False
        or lock["evidence_sha256"]
        != "e6c47ad40851371bddbc3326737f478111e07d6fd27f56415a61157cb211322d"
        or source_entity_id("ufcstats", lock["ufcstats_url"])
        != lock["stable_ufcstats_id"]
    ):
        raise ValueError("Ethyn Ewing's permanent owner lock is inconsistent")

    hold = holds[0]
    if not isinstance(hold, dict):
        raise ValueError("Profile field hold 1 must be an object")
    _require_exact_keys(hold, _PROFILE_FIELD_HOLD_KEYS, "profile field hold 1")
    _require_nonempty_text(
        hold,
        (
            "canonical_name",
            "stable_ufcstats_id",
            "ufcstats_url",
            "field",
            "decision_status",
            "scope",
            "reason",
            "evidence_sha256",
        ),
        "profile field hold 1",
    )
    reviewed_names = _require_string_list(
        hold["reviewed_names"], "profile field hold 1.reviewed_names"
    )
    conflicting_values = _require_string_list(
        hold["conflicting_normalized_values"],
        "profile field hold 1.conflicting_normalized_values",
        minimum=2,
    )
    _require_string_list(
        hold["evidence_urls"], "profile field hold 1.evidence_urls", minimum=3
    )
    stable_id = normalize_ufcstats_id(hold["stable_ufcstats_id"])
    if (
        hold["canonical_name"] != "Wendri Patilima"
        or normalize_person_name(hold["canonical_name"])
        not in {normalize_person_name(name) for name in reviewed_names}
        or stable_id != "66d42e9503fa108e"
        or source_entity_id("ufcstats", hold["ufcstats_url"]) != stable_id
        or hold["field"] != "height"
        or set(conflicting_values) != {"167.64 cm", "175.26 cm"}
        or hold["queue_index"] != 115
        or hold["decision_status"] != "conflicting_field_hold"
        or hold["automatic_release_allowed"] is not False
        or hold["scope"] != "current_and_future"
        or hold["historical_backfill_allowed"] is not False
        or hold["evidence_sha256"] != payload["decision_report_sha256"]
    ):
        raise ValueError("Wendri Patilima's reviewed height hold is inconsistent")

    route_quarantine = route_quarantines[0]
    label = "unbound simultaneous-route quarantine 1"
    if not isinstance(route_quarantine, dict):
        raise ValueError(f"{label} must be an object")
    _require_exact_keys(
        route_quarantine,
        _UNBOUND_SIMULTANEOUS_ROUTE_QUARANTINE_KEYS,
        label,
    )
    _require_nonempty_text(
        route_quarantine,
        (
            "official_name",
            "alternate_official_athlete_url",
            "alternate_official_source_entity_id",
            "companion_official_athlete_url",
            "companion_official_source_entity_id",
            "companion_stable_ufcstats_id",
            "companion_ufcstats_url",
            "reason",
        ),
        label,
    )
    quarantine_evidence_urls = _require_string_list(
        route_quarantine["evidence_urls"],
        f"{label}.evidence_urls",
        minimum=3,
    )
    alternate_entity = source_entity_id(
        "ufc", route_quarantine["alternate_official_athlete_url"]
    )
    companion_entity = source_entity_id(
        "ufc", route_quarantine["companion_official_athlete_url"]
    )
    companion_stable_id = normalize_ufcstats_id(
        route_quarantine["companion_stable_ufcstats_id"]
    )
    observed_profile = route_quarantine.get("alternate_observed_profile")
    if not isinstance(observed_profile, dict):
        raise ValueError(f"{label}.alternate_observed_profile must be an object")
    _require_exact_keys(
        observed_profile,
        _UNBOUND_ALTERNATE_OBSERVED_PROFILE_KEYS,
        f"{label}.alternate_observed_profile",
    )
    expected_observed_profile = {
        "profile_name": "Casey Kenney",
        "canonical_athlete_url": "https://www.ufc.com/athlete/casey-kenney-0",
        "profile_division": "Bantamweight Division",
        "profile_record": "",
        "profile_status": "Not Fighting",
        "profile_source_tag": "",
        "birthplace": "",
        "age": "",
        "height": "",
        "reach": "",
        "weight": "",
        "octagon_debut": "Aug. 25, 2026",
        "alternate_slug_names": "casey kenney 0",
    }
    expected_quarantine_evidence_urls = {
        "https://www.ufc.com/athlete/casey-kenney",
        "https://www.ufc.com/athlete/casey-kenney-0",
        "http://ufcstats.com/fighter-details/e7bba7ce4d6c7b07",
    }
    if (
        route_quarantine["official_name"] != "Casey Kenney"
        or route_quarantine["alternate_official_athlete_url"]
        != "https://www.ufc.com/athlete/casey-kenney-0"
        or observed_profile != expected_observed_profile
        or alternate_entity
        != route_quarantine["alternate_official_source_entity_id"]
        or route_quarantine["companion_official_athlete_url"]
        != "https://www.ufc.com/athlete/casey-kenney"
        or companion_entity
        != route_quarantine["companion_official_source_entity_id"]
        or not alternate_entity
        or not companion_entity
        or alternate_entity == companion_entity
        or companion_stable_id != "e7bba7ce4d6c7b07"
        or source_entity_id(
            "ufcstats", route_quarantine["companion_ufcstats_url"]
        )
        != companion_stable_id
        or route_quarantine["automatic_alias_allowed"] is not False
        or route_quarantine["automatic_binding_allowed"] is not False
        or route_quarantine["automatic_redirect_allowed"] is not False
        or route_quarantine["destructive_exclusion_allowed"] is not False
        or set(quarantine_evidence_urls) != expected_quarantine_evidence_urls
    ):
        raise ValueError("Casey Kenney's route quarantine is inconsistent")

    seen_official_entities: set[str] = set()
    exclusion_queue_indices: set[int] = set()
    for position, record in enumerate(exclusions, start=1):
        label = f"non-MMA exclusion {position}"
        if not isinstance(record, dict):
            raise ValueError(f"{label} must be an object")
        _require_exact_keys(record, _NON_MMA_KEYS, label)
        _require_nonempty_text(
            record,
            ("official_name", "official_athlete_url", "official_source_entity_id", "reason"),
            label,
        )
        if record["reason"] != "power_slap_non_mma":
            raise ValueError(f"{label} has an unsupported reason")
        entity_id = source_entity_id("ufc", record["official_athlete_url"])
        if entity_id != record["official_source_entity_id"]:
            raise ValueError(f"{label} official UFC entity is inconsistent")
        if entity_id in seen_official_entities:
            raise ValueError(f"Duplicate reviewed official UFC entity: {entity_id}")
        seen_official_entities.add(entity_id)
        queue_indices = _require_queue_indices(record["queue_indices"], f"{label}.queue_indices")
        if exclusion_queue_indices.intersection(queue_indices):
            raise ValueError(f"{label} overlaps another exclusion queue record")
        exclusion_queue_indices.update(queue_indices)
        _require_string_list(record["evidence_entity_ids"], f"{label}.evidence_entity_ids")
        _require_string_list(record["evidence_urls"], f"{label}.evidence_urls")
    if exclusion_queue_indices != {5, 6, 7, 13, 14, 15, 16, 40}:
        raise ValueError("Reviewed non-MMA exclusions must cover the exact 8 rejected cells")

    seen_stable_ids: set[str] = set()
    binding_queue_indices: set[int] = set()
    for position, record in enumerate(bindings, start=1):
        label = f"MMA identity binding {position}"
        if not isinstance(record, dict):
            raise ValueError(f"{label} must be an object")
        _require_exact_keys(record, _MMA_BINDING_KEYS, label)
        _require_nonempty_text(
            record,
            (
                "official_name",
                "official_athlete_url",
                "official_source_entity_id",
                "ufcstats_name",
                "stable_ufcstats_id",
                "ufcstats_url",
                "identity_basis",
            ),
            label,
        )
        names = _require_string_list(
            record["reviewed_official_names"], f"{label}.reviewed_official_names"
        )
        if normalize_person_name(record["official_name"]) not in {
            normalize_person_name(name) for name in names
        }:
            raise ValueError(f"{label} does not authorize its canonical official name")
        entity_id = source_entity_id("ufc", record["official_athlete_url"])
        if entity_id != record["official_source_entity_id"]:
            raise ValueError(f"{label} official UFC entity is inconsistent")
        if entity_id in seen_official_entities:
            raise ValueError(f"Duplicate reviewed official UFC entity: {entity_id}")
        seen_official_entities.add(entity_id)
        stable_id = normalize_ufcstats_id(record["stable_ufcstats_id"])
        if (
            not stable_id
            or stable_id != source_entity_id("ufcstats", record["ufcstats_url"])
        ):
            raise ValueError(f"{label} UFCStats identity is inconsistent")
        if stable_id in seen_stable_ids:
            raise ValueError(f"Duplicate reviewed UFCStats identity: {stable_id}")
        seen_stable_ids.add(stable_id)
        queue_indices = _require_queue_indices(record["queue_indices"], f"{label}.queue_indices")
        if binding_queue_indices.intersection(queue_indices):
            raise ValueError(f"{label} overlaps another binding queue record")
        binding_queue_indices.update(queue_indices)
        _require_string_list(record["evidence_urls"], f"{label}.evidence_urls", minimum=2)
    if binding_queue_indices != {2, 3, 8, 17, 20, 21, 22, 23, 41, 70, 71, 72, 74, 97, 98, 106, 116, 117}:
        raise ValueError("Reviewed MMA bindings must cover the exact 18 recovered cells")

    seen_transition_entities: set[str] = set()
    seen_transition_stable_ids: set[str] = set()
    for index, transition in enumerate(transitions, start=1):
        label = f"official identity transition {index}"
        if not isinstance(transition, dict):
            raise ValueError(f"{label} must be an object")
        _require_exact_keys(transition, _OFFICIAL_IDENTITY_TRANSITION_KEYS, label)
        _require_nonempty_text(
            transition,
            (
                "source_official_name",
                "source_official_athlete_url",
                "source_official_source_entity_id",
                "source_canonical_athlete_url",
                "target_official_name",
                "target_official_athlete_url",
                "target_official_source_entity_id",
                "target_canonical_athlete_url",
                "ufcstats_name",
                "stable_ufcstats_id",
                "ufcstats_url",
                "direction",
                "identity_basis",
            ),
            label,
        )
        if any(
            not isinstance(transition[field], bool)
            for field in (
                "source_canonical_evidence_required",
                "target_canonical_evidence_required",
                "automatic_reverse_allowed",
            )
        ):
            raise ValueError(f"{label} has invalid boolean controls")
        transition_stable_id = normalize_ufcstats_id(
            transition["stable_ufcstats_id"]
        )
        expected = _PINNED_OFFICIAL_IDENTITY_TRANSITIONS.get(
            transition_stable_id or ""
        )
        evidence_urls = _require_string_list(
            transition["evidence_urls"], f"{label}.evidence_urls", minimum=2
        )
        fight_anchors = _validated_transition_anchor_set(
            transition["fight_anchors"], label
        )
        source_entity = source_entity_id(
            "ufc", transition["source_official_athlete_url"]
        )
        target_entity = source_entity_id(
            "ufc", transition["target_official_athlete_url"]
        )
        source_canonical_entity = source_entity_id(
            "ufc", transition["source_canonical_athlete_url"]
        )
        target_canonical_entity = source_entity_id(
            "ufc", transition["target_canonical_athlete_url"]
        )
        source_bindings = [
            record
            for record in bindings
            if record["official_source_entity_id"] == source_entity
        ]
        expected_source_binding_count = (
            1 if transition_stable_id == "bbc9ab95fa766493" else 0
        )
        pinned_fields = (
            "source_official_name",
            "source_official_athlete_url",
            "source_official_source_entity_id",
            "source_canonical_athlete_url",
            "source_canonical_evidence_required",
            "target_official_name",
            "target_official_athlete_url",
            "target_official_source_entity_id",
            "target_canonical_athlete_url",
            "target_canonical_evidence_required",
            "ufcstats_name",
            "ufcstats_url",
        )
        if (
            expected is None
            or any(transition[field] != expected[field] for field in pinned_fields)
            or source_entity != transition["source_official_source_entity_id"]
            or target_entity != transition["target_official_source_entity_id"]
            or not source_entity
            or not target_entity
            or source_entity == target_entity
            or source_canonical_entity
            not in {source_entity, target_entity}
            or target_canonical_entity != target_entity
            or source_entity in seen_transition_entities
            or target_entity in seen_transition_entities
            or target_entity in seen_official_entities
            or transition_stable_id in seen_transition_stable_ids
            or source_entity_id("ufcstats", transition["ufcstats_url"])
            != transition_stable_id
            or transition["direction"] != "source_to_target"
            or transition["automatic_reverse_allowed"] is not False
            or len(source_bindings) != expected_source_binding_count
            or (
                source_bindings
                and (
                    source_bindings[0]["official_name"]
                    != transition["source_official_name"]
                    or source_bindings[0]["stable_ufcstats_id"]
                    != transition_stable_id
                    or source_bindings[0]["ufcstats_url"]
                    != transition["ufcstats_url"]
                    or source_bindings[0]["ufcstats_name"]
                    != transition["ufcstats_name"]
                )
            )
            or set(evidence_urls) != expected["evidence_urls"]
            or fight_anchors != expected["fight_anchors"]
        ):
            raise ValueError(f"{label} reviewed transition is inconsistent")
        seen_transition_entities.update({source_entity, target_entity})
        seen_transition_stable_ids.add(transition_stable_id)
    if seen_transition_stable_ids != set(_PINNED_OFFICIAL_IDENTITY_TRANSITIONS):
        raise ValueError("Reviewed official identity transition set is incomplete")

    recovery = recoveries[0]
    label = "profile recovery 1"
    if not isinstance(recovery, dict):
        raise ValueError(f"{label} must be an object")
    _require_exact_keys(recovery, _PROFILE_RECOVERY_KEYS, label)
    _require_nonempty_text(
        recovery,
        (
            "canonical_name",
            "official_athlete_url",
            "official_source_entity_id",
            "stable_ufcstats_id",
            "ufcstats_url",
            "field",
            "value",
            "source",
            "source_name",
            "source_entity_id",
            "source_url",
            "requested_url",
            "effective_url",
            "canonical_url",
            "retrieval_route",
            "retrieved_at_utc",
            "observation_id",
            "capture_sha256",
            "scope",
            "decision_status",
            "identity_basis",
        ),
        label,
    )
    _require_string_list(recovery["reviewed_source_names"], f"{label}.reviewed_source_names")
    _require_string_list(recovery["evidence_urls"], f"{label}.evidence_urls", minimum=2)
    if recovery["queue_index"] != 2:
        raise ValueError("The reviewed profile recovery must be canonical queue cell 2")
    if recovery["source"] != "wikipedia" or recovery["field"] != "stance":
        raise ValueError("The reviewed profile recovery must be the exact Wikipedia stance pin")
    if recovery["value"] not in _VALID_STANCES:
        raise ValueError("The reviewed profile recovery has an invalid stance")
    if (
        recovery["requested_url"] != recovery["source_url"]
        or recovery["effective_url"] != recovery["source_url"]
        or recovery["canonical_url"] != recovery["source_url"]
        or recovery["redirect_chain"] != []
        or recovery["retrieval_route"] != "direct"
        or recovery["observation_id"] != "B01-O0014"
        or recovery["capture_sha256"]
        != "0c991fd130c77bdf5375274ef79d7b9a41e5125963163f7f893a6e937101dc37"
        or recovery["scope"] != "current_and_future"
        or recovery["historical_backfill_allowed"] is not False
        or recovery["decision_status"]
        != "owner_approved_current_future_exception"
    ):
        raise ValueError("The reviewed profile recovery provenance is inconsistent")
    if source_entity_id("ufc", recovery["official_athlete_url"]) != recovery["official_source_entity_id"]:
        raise ValueError("The reviewed profile recovery official UFC entity is inconsistent")
    stable_id = normalize_ufcstats_id(recovery["stable_ufcstats_id"])
    if not stable_id or stable_id != source_entity_id("ufcstats", recovery["ufcstats_url"]):
        raise ValueError("The reviewed profile recovery UFCStats identity is inconsistent")
    if _wikipedia_entity_id(recovery["source_url"]) != recovery["source_entity_id"]:
        raise ValueError("The reviewed profile recovery Wikipedia entity is inconsistent")
    if normalize_person_name(recovery["source_name"]) not in {
        normalize_person_name(name) for name in recovery["reviewed_source_names"]
    }:
        raise ValueError("The reviewed profile recovery source name is not authorized")
    return payload


@lru_cache(maxsize=None)
def load_reviewed_roster_recovery_registry(
    path: Path = REVIEWED_ROSTER_RECOVERY_REGISTRY_PATH,
) -> dict[str, object]:
    registry_path = Path(path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    return _validate_registry(payload)


def _record_for_official_entity(section: str, row: Mapping[str, object]) -> dict[str, object] | None:
    entity_id = source_entity_id("ufc", row.get("official_athlete_url"))
    if not entity_id:
        return None
    records = load_reviewed_roster_recovery_registry()[section]
    matches = [
        record
        for record in records
        if record["official_source_entity_id"] == entity_id
    ]
    if len(matches) > 1:
        raise ValueError(f"Reviewed roster registry is ambiguous for UFC entity {entity_id}")
    return dict(matches[0]) if matches else None


def reviewed_official_identity_transitions() -> list[dict[str, object]]:
    return [
        dict(record)
        for record in load_reviewed_roster_recovery_registry()[
            "official_identity_transitions"
        ]
    ]


def reviewed_unbound_simultaneous_route_quarantines() -> list[dict[str, object]]:
    """Return exact negative route certificates; these never authorize identity."""

    return [
        dict(record)
        for record in load_reviewed_roster_recovery_registry()[
            "unbound_simultaneous_route_quarantines"
        ]
    ]


def reviewed_unbound_simultaneous_alternate_route(
    row: Mapping[str, object],
) -> dict[str, object] | None:
    """Match only a registered alternate UFC entity, never a display name."""

    entity_id = source_entity_id("ufc", row.get("official_athlete_url"))
    if not entity_id:
        return None
    matches = [
        record
        for record in reviewed_unbound_simultaneous_route_quarantines()
        if record["alternate_official_source_entity_id"] == entity_id
    ]
    if len(matches) > 1:
        raise ValueError(
            "Reviewed unbound alternate-route registry is ambiguous for UFC entity "
            f"{entity_id}"
        )
    return dict(matches[0]) if matches else None


def _transition_for_endpoint_entity(
    row: Mapping[str, object], endpoint: str
) -> dict[str, object] | None:
    entity_id = source_entity_id("ufc", row.get("official_athlete_url"))
    if not entity_id or endpoint not in {"source", "target"}:
        return None
    matches = [
        transition
        for transition in reviewed_official_identity_transitions()
        if transition[f"{endpoint}_official_source_entity_id"] == entity_id
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Reviewed roster transition is ambiguous for {endpoint} UFC entity "
            f"{entity_id}"
        )
    return matches[0] if matches else None


def _row_has_exact_transition_stable_id(
    row: Mapping[str, object], expected_stable_id: str
) -> bool:
    observed: list[str] = []
    for field in ("ufcstats_url", "stable_ufcstats_id", "fighter_id"):
        value = row.get(field)
        text = _text(value)
        if not text or text.casefold() in {"nan", "none", "nat", "<na>"}:
            continue
        stable_id = normalize_ufcstats_id(value)
        if not stable_id:
            return False
        observed.append(stable_id)
    return bool(observed) and set(observed) == {expected_stable_id}


def _row_has_no_contradictory_transition_evidence(
    row: Mapping[str, object],
    transition: Mapping[str, object],
    endpoint: str,
) -> bool:
    """Reject a route certificate when any present fresh evidence disagrees."""

    expected_name = normalize_person_name(
        transition.get(f"{endpoint}_official_name")
    )
    expected_url = _text(
        transition.get(f"{endpoint}_official_athlete_url")
    ).rstrip("/")
    expected_canonical_url = _text(
        transition.get(f"{endpoint}_canonical_athlete_url")
    ).rstrip("/")
    expected_entity = _text(
        transition.get(f"{endpoint}_official_source_entity_id")
    )
    expected_stable_id = _text(transition.get("stable_ufcstats_id"))
    expected_ufcstats_name = normalize_person_name(
        transition.get("ufcstats_name")
    )

    for field in ("profile_name", "official_profile_binding_display_name"):
        value = _text(row.get(field))
        if value and normalize_person_name(value) != expected_name:
            return False

    route_slug = unquote(urlparse(expected_url).path.rstrip("/").rsplit("/", 1)[-1])
    expected_slug_name = normalize_person_name(route_slug.replace("-", " "))
    slug_name = _text(row.get("slug_name"))
    if slug_name and normalize_person_name(slug_name) != expected_slug_name:
        return False

    binding_canonical_name = _text(
        row.get("official_profile_binding_canonical_name")
    )
    if binding_canonical_name and normalize_person_name(
        binding_canonical_name
    ) not in {expected_name, expected_ufcstats_name}:
        return False

    ufcstats_name = _text(row.get("ufcstats_name"))
    if (
        ufcstats_name
        and normalize_person_name(ufcstats_name) != expected_ufcstats_name
    ):
        return False

    for field in (
        "official_profile_binding_requested_url",
        "official_profile_binding_source_url",
        "official_profile_fresh_requested_url",
        "official_profile_fresh_effective_url",
    ):
        value = _text(row.get(field)).rstrip("/")
        if value and value != expected_url:
            return False

    for field in (
        "canonical_athlete_url",
        "official_profile_binding_canonical_url",
        "official_profile_fresh_canonical_url",
    ):
        value = _text(row.get(field)).rstrip("/")
        if value and value != expected_canonical_url:
            return False
    canonical_required = transition.get(
        f"{endpoint}_canonical_evidence_required"
    ) is True
    if (
        canonical_required
        and _text(row.get("canonical_athlete_url")).rstrip("/")
        != expected_canonical_url
    ):
        return False

    binding_entity = _text(row.get("official_profile_binding_source_entity_id"))
    if binding_entity and binding_entity != expected_entity:
        return False
    binding_stable_id = _text(
        row.get("official_profile_binding_stable_ufcstats_id")
    )
    if binding_stable_id:
        normalized_binding_id = normalize_ufcstats_id(binding_stable_id)
        if normalized_binding_id != expected_stable_id:
            return False

    identity_valid = _text(row.get("official_url_identity_valid")).casefold()
    if identity_valid and identity_valid not in {"1", "1.0", "true", "yes", "y"}:
        return False
    identity_status = (
        _text(row.get("official_url_identity_status"))
        .casefold()
        .replace("_", " ")
        .replace("-", " ")
    )
    if identity_status and identity_status not in {
        "valid",
        "slug mismatch profile valid",
    }:
        return False
    fetch_outcome = _text(row.get("official_profile_fetch_outcome")).casefold()
    if fetch_outcome and fetch_outcome != "succeeded":
        return False
    binding_status = _text(
        row.get("official_profile_binding_identity_status")
    ).casefold()
    if binding_status and binding_status not in {"verified", "reviewed"}:
        return False
    persisted_binding_status = _text(
        row.get("official_profile_identity_binding_status")
    )
    persisted_binding_required_fields = (
        "official_profile_binding_stable_ufcstats_id",
        "official_profile_binding_source_entity_id",
        "official_profile_binding_requested_url",
        "official_profile_binding_source_url",
        "official_profile_binding_canonical_url",
        "official_profile_binding_redirect_chain",
        "official_profile_binding_canonical_name",
        "official_profile_binding_display_name",
        "official_profile_binding_identity_status",
        "official_profile_binding_name_match_status",
        "official_profile_binding_identity_basis",
        "official_profile_binding_identity_evidence_url",
        "official_profile_binding_identity_event_date",
        "official_profile_binding_identity_opponent_name",
        "official_profile_binding_identity_opponent_ufcstats_id",
        "official_profile_binding_identity_organization",
        "official_profile_binding_identity_fight_url",
        "official_profile_binding_identity_ufcstats_event_date",
        "official_profile_binding_identity_ufcstats_opponent_name",
        "official_profile_binding_identity_opponent_name_match_status",
    )
    if persisted_binding_status and (
        persisted_binding_status != "validated_profile_decision_binding"
        or any(
            not _text(row.get(field))
            for field in persisted_binding_required_fields
        )
    ):
        return False

    for field in (
        "official_profile_binding_redirect_chain",
        "official_profile_fresh_redirect_chain",
    ):
        redirect_text = _text(row.get(field))
        if not redirect_text:
            continue
        try:
            redirects = json.loads(redirect_text)
        except (TypeError, ValueError):
            return False
        if not isinstance(redirects, list) or any(
            _text(value).rstrip("/") != expected_url for value in redirects
        ):
            return False

    evidence_url = _text(
        row.get("official_profile_binding_identity_evidence_url")
    ).rstrip("/")
    if evidence_url and (
        evidence_url != expected_url
        or source_entity_id("ufc", evidence_url) != expected_entity
    ):
        return False
    organization = _text(
        row.get("official_profile_binding_identity_organization")
    ).casefold()
    if organization and organization != "ufc":
        return False
    name_match_status = (
        _text(row.get("official_profile_binding_name_match_status"))
        .casefold()
        .replace("_", " ")
        .replace("-", " ")
    )
    if name_match_status and name_match_status not in {"exact", "approved alias"}:
        return False
    identity_basis = (
        _text(row.get("official_profile_binding_identity_basis"))
        .casefold()
        .replace("_", " ")
        .replace("-", " ")
    )
    if identity_basis and identity_basis not in {
        "accepted reviewed binding",
        "event date opponent",
        "event date opponent organization",
    }:
        return False

    opponent_name_match_status = (
        _text(
            row.get(
                "official_profile_binding_identity_opponent_name_match_status"
            )
        )
        .casefold()
        .replace("_", " ")
        .replace("-", " ")
    )
    if opponent_name_match_status and opponent_name_match_status != "exact":
        return False

    source_date = _text(row.get("official_profile_binding_identity_event_date"))
    stats_date = _text(
        row.get("official_profile_binding_identity_ufcstats_event_date")
    )
    if (source_date or stats_date) and (
        not source_date
        or not stats_date
        or source_date != stats_date
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", source_date) is None
    ):
        return False
    source_opponent = normalize_person_name(
        _text(row.get("official_profile_binding_identity_opponent_name"))
    )
    stats_opponent = normalize_person_name(
        _text(
            row.get(
                "official_profile_binding_identity_ufcstats_opponent_name"
            )
        )
    )
    if (source_opponent or stats_opponent) and (
        not source_opponent
        or not stats_opponent
        or source_opponent != stats_opponent
    ):
        return False
    opponent_id_value = _text(
        row.get("official_profile_binding_identity_opponent_ufcstats_id")
    )
    opponent_id = ""
    if opponent_id_value:
        opponent_id = normalize_ufcstats_id(opponent_id_value)
        if not opponent_id or opponent_id == expected_stable_id:
            return False
    fight_url = _text(row.get("official_profile_binding_identity_fight_url"))
    if fight_url:
        parsed_fight = urlparse(fight_url)
        if (
            parsed_fight.scheme.casefold() != "http"
            or (parsed_fight.hostname or "").casefold().rstrip(".")
            != "ufcstats.com"
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
    anchor_values = (
        source_date,
        stats_date,
        source_opponent,
        stats_opponent,
        opponent_id,
        fight_url,
    )
    if any(anchor_values):
        if not all(anchor_values):
            return False
        anchor = (source_date, source_opponent, opponent_id, fight_url)
        if anchor not in _validated_transition_anchor_set(
            transition.get("fight_anchors"),
            "reviewed official identity transition",
        ):
            return False
    if identity_basis.startswith("event date opponent") and not all(anchor_values):
        return False
    combat_sport = _text(row.get("combat_sport")).casefold()
    if _row_has_power_slap_evidence(row) or (
        combat_sport and combat_sport != "mma"
    ):
        return False
    return True


def _row_matches_transition_endpoint(
    row: Mapping[str, object],
    transition: Mapping[str, object],
    endpoint: str,
    *,
    require_stable_id: bool,
) -> bool:
    if endpoint not in {"source", "target"}:
        return False
    expected_url = _text(
        transition.get(f"{endpoint}_official_athlete_url")
    ).rstrip("/")
    observed_url = _text(row.get("official_athlete_url")).rstrip("/")
    expected_entity = _text(
        transition.get(f"{endpoint}_official_source_entity_id")
    )
    expected_name = normalize_person_name(
        transition.get(f"{endpoint}_official_name")
    )
    observed_name = normalize_person_name(row.get("official_name"))
    if not (
        expected_url
        and observed_url == expected_url
        and source_entity_id("ufc", observed_url) == expected_entity
        and expected_name
        and observed_name == expected_name
        and _row_has_no_contradictory_transition_evidence(
            row,
            transition,
            endpoint,
        )
    ):
        return False
    expected_stable_id = str(transition["stable_ufcstats_id"])
    carrier_fields_present = any(
        _text(row.get(field)).casefold() not in {"", "nan", "none", "nat", "<na>"}
        for field in ("ufcstats_url", "stable_ufcstats_id", "fighter_id")
    )
    if require_stable_id or carrier_fields_present:
        return _row_has_exact_transition_stable_id(row, expected_stable_id)
    return True


def _transition_endpoint_binding(
    row: Mapping[str, object],
    endpoint: str,
) -> dict[str, object] | None:
    transition = _transition_for_endpoint_entity(row, endpoint)
    if transition is None:
        return None
    source_binding = next(
        (
            record
            for record in load_reviewed_roster_recovery_registry()[
                "mma_identity_bindings"
            ]
            if record["official_source_entity_id"]
            == transition["source_official_source_entity_id"]
        ),
        None,
    )
    if source_binding is not None and any(
        source_binding[field] != transition[transition_field]
        for field, transition_field in (
            ("stable_ufcstats_id", "stable_ufcstats_id"),
            ("ufcstats_url", "ufcstats_url"),
            ("ufcstats_name", "ufcstats_name"),
        )
    ):
        raise ValueError("Reviewed roster transition source binding is inconsistent")
    return {
        "official_name": transition[f"{endpoint}_official_name"],
        "reviewed_official_names": [transition[f"{endpoint}_official_name"]],
        "official_athlete_url": transition[f"{endpoint}_official_athlete_url"],
        "official_source_entity_id": transition[
            f"{endpoint}_official_source_entity_id"
        ],
        "ufcstats_name": transition["ufcstats_name"],
        "stable_ufcstats_id": transition["stable_ufcstats_id"],
        "ufcstats_url": transition["ufcstats_url"],
        "queue_indices": list(source_binding.get("queue_indices", []))
        if source_binding is not None
        else [],
        "identity_basis": transition["identity_basis"],
        "evidence_urls": list(transition["evidence_urls"]),
    }


def reviewed_official_identity_transition_match(
    live_target_row: Mapping[str, object],
    cached_source_row: Mapping[str, object],
) -> bool:
    """Match only the reviewed directional target/live to source/cached route."""

    transition = _transition_for_endpoint_entity(live_target_row, "target")
    return bool(
        transition
        and _row_matches_transition_endpoint(
            live_target_row,
            transition,
            "target",
            require_stable_id=True,
        )
        and _row_matches_transition_endpoint(
            cached_source_row,
            transition,
            "source",
            require_stable_id=True,
        )
    )


def reviewed_non_mma_exclusion(
    row: Mapping[str, object],
) -> dict[str, object] | None:
    """Match only a reviewed stable UFC entity; names never select an exclusion."""

    return _record_for_official_entity("non_mma_identity_exclusions", row)


def reviewed_non_mma_identity_match(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> bool:
    left_record = reviewed_non_mma_exclusion(left)
    right_record = reviewed_non_mma_exclusion(right)
    return bool(
        left_record
        and right_record
        and left_record["official_source_entity_id"]
        == right_record["official_source_entity_id"]
    )


def reviewed_mma_identity_binding(
    row: Mapping[str, object],
) -> dict[str, object] | None:
    """Return the exact URL-bound MMA mapping after reviewed-name validation."""

    target_transition = _transition_for_endpoint_entity(row, "target")
    source_transition = _transition_for_endpoint_entity(row, "source")
    record = reviewed_mma_identity_registry_record(row)
    if record is None:
        return None
    if target_transition is not None and not _row_matches_transition_endpoint(
        row,
        target_transition,
        "target",
        require_stable_id=True,
    ):
        return None
    if source_transition is not None and not _row_matches_transition_endpoint(
        row,
        source_transition,
        "source",
        require_stable_id=False,
    ):
        return None
    official_name_key = normalize_person_name(row.get("official_name"))
    allowed_name_keys = {
        normalize_person_name(name) for name in record["reviewed_official_names"]
    }
    if not official_name_key or official_name_key not in allowed_name_keys:
        return None
    for field in ("ufcstats_url", "stable_ufcstats_id", "fighter_id"):
        value = row.get(field)
        if not _text(value):
            continue
        observed_stable_id = normalize_ufcstats_id(value)
        if observed_stable_id != record["stable_ufcstats_id"]:
            return None
    return record


def reviewed_mma_identity_transition_pending_stable_id(
    row: Mapping[str, object],
) -> bool:
    """Allow only the exact transition target to proceed to ID discovery.

    This is not an identity binding.  It is true only while every UFCStats
    carrier is absent; any present malformed, substituted, or conflicting
    carrier fails closed.  The caller must resolve and then re-run the full
    reviewed binding before using the row.
    """

    transition = _transition_for_endpoint_entity(row, "target")
    if transition is None or any(
        _text(row.get(field))
        for field in ("ufcstats_url", "stable_ufcstats_id", "fighter_id")
    ):
        return False
    return _row_matches_transition_endpoint(
        row,
        transition,
        "target",
        require_stable_id=False,
    )


def reviewed_mma_identity_transition_pending_source_canonical(
    row: Mapping[str, object],
) -> bool:
    """Recognize only the exact source route awaiting its required page canonical.

    This is a fetch permission, not an identity binding.  It cannot carry a
    UFCStats identity or survive a missing, wrong, or unverified canonical page.
    """

    transition = _transition_for_endpoint_entity(row, "source")
    if (
        transition is None
        or transition.get("source_canonical_evidence_required") is not True
        or any(
            _text(row.get(field))
            for field in ("ufcstats_url", "stable_ufcstats_id", "fighter_id")
        )
        or _text(row.get("canonical_athlete_url"))
    ):
        return False
    provisional = dict(row)
    provisional["canonical_athlete_url"] = transition[
        "source_canonical_athlete_url"
    ]
    return _row_matches_transition_endpoint(
        provisional,
        transition,
        "source",
        require_stable_id=False,
    )


def reviewed_mma_identity_registry_record(
    row: Mapping[str, object],
) -> dict[str, object] | None:
    """Return the registered UFC entity before validating mutable row carriers.

    Callers that would otherwise fall back to an unreviewed name or stale
    UFCStats URL use this to distinguish "not registered" from "registered but
    conflicting" and therefore fail closed.
    """

    record = _record_for_official_entity("mma_identity_bindings", row)
    if record is not None:
        return record
    source_record = _transition_endpoint_binding(row, "source")
    return (
        source_record
        if source_record is not None
        else _transition_endpoint_binding(row, "target")
    )


def reviewed_mma_identity_binding_conflicts(
    row: Mapping[str, object],
) -> bool:
    return bool(
        reviewed_mma_identity_registry_record(row) is not None
        and reviewed_mma_identity_binding(row) is None
    )


def apply_reviewed_mma_identity_binding(row: dict[str, object]) -> bool:
    record = reviewed_mma_identity_binding(row)
    if record is None:
        return False
    row.update(
        {
            "ufcstats_name": record["ufcstats_name"],
            "ufcstats_url": record["ufcstats_url"],
            "ufcstats_resolution": "reviewed_roster_recovery_binding",
            "combat_sport": "mma",
            "combat_sport_reason": "ufcstats_profile_match",
        }
    )
    return True


def reviewed_mma_identity_binding_is_complete(row: Mapping[str, object]) -> bool:
    record = reviewed_mma_identity_binding(row)
    return bool(
        record
        and normalize_ufcstats_id(row.get("ufcstats_url"))
        == record["stable_ufcstats_id"]
        and _text(row.get("ufcstats_name")) == record["ufcstats_name"]
    )


def reviewed_profile_recovery_observations(
    rows: Sequence[Mapping[str, object]],
) -> list[tuple[dict[str, object], dict[str, object]]]:
    """Select only supplement observations that exactly match a reviewed pin."""

    selected: list[tuple[dict[str, object], dict[str, object]]] = []
    recoveries = load_reviewed_roster_recovery_registry()["profile_recoveries"]
    for raw_recovery in recoveries:
        recovery = dict(raw_recovery)
        candidates = [
            dict(row)
            for row in rows
            if _text(row.get("source")).casefold() == recovery["source"]
            and _wikipedia_entity_id(row.get("fighter_url"))
            == recovery["source_entity_id"]
        ]
        if len(candidates) > 1:
            raise ValueError(
                "Reviewed profile recovery has duplicate source-entity observations: "
                f"{recovery['source_entity_id']}"
            )
        if not candidates:
            continue
        row = candidates[0]
        allowed_source_names = {
            normalize_person_name(name) for name in recovery["reviewed_source_names"]
        }
        if (
            normalize_person_name(row.get("name"))
            != normalize_person_name(recovery["canonical_name"])
            or normalize_person_name(row.get("source_name"))
            not in allowed_source_names
            or _text(row.get(recovery["field"])).casefold()
            != _text(recovery["value"]).casefold()
        ):
            raise ValueError(
                "Reviewed profile recovery observation conflicts with its exact pin: "
                f"{recovery['canonical_name']} {recovery['field']}"
            )
        selected.append((recovery, row))
    return selected


def permanent_profile_locks() -> list[dict[str, object]]:
    return [
        dict(record)
        for record in load_reviewed_roster_recovery_registry()["permanent_profile_locks"]
    ]


def reviewed_profile_field_holds() -> list[dict[str, object]]:
    """Return exact field-level holds that must blank runtime profile views."""

    return [
        dict(record)
        for record in load_reviewed_roster_recovery_registry()[
            "profile_field_holds"
        ]
    ]


def reviewed_profile_field_hold(
    row: Mapping[str, object],
    *,
    field: str,
) -> dict[str, object] | None:
    """Return an exact stable-ID/name-bound current-and-future field hold.

    A reviewed hold is negative evidence: it may keep a value unavailable, but
    it must never identify a fighter from a name alone. Every present UFCStats
    carrier must therefore be valid and agree, and every present identity name
    must remain inside the reviewed name set.
    """

    field_key = _text(field).casefold()
    carriers: dict[str, str] = {}
    for carrier_field in (
        "fighter_url",
        "ufcstats_url",
        "stable_ufcstats_id",
        "fighter_id",
    ):
        raw_value = row.get(carrier_field)
        text = _text(raw_value)
        if not text or text.casefold() in {"nan", "none", "nat", "<na>"}:
            continue
        normalized = normalize_ufcstats_id(raw_value)
        if normalized is None:
            return None
        carriers[carrier_field] = normalized
    identities = set(carriers.values())
    if len(identities) != 1:
        return None
    stable_id = next(iter(identities))

    matches = [
        hold
        for hold in reviewed_profile_field_holds()
        if _text(hold.get("field")).casefold() == field_key
        and hold.get("stable_ufcstats_id") == stable_id
    ]
    if len(matches) != 1:
        return None
    hold = matches[0]
    allowed_name_keys = {
        normalize_person_name(name) for name in hold["reviewed_names"]
    }
    observed_name_keys = {
        normalized
        for name_field in (
            "official_name",
            "profile_name",
            "ufcstats_name",
            "name",
            "source_name",
        )
        if (normalized := normalize_person_name(row.get(name_field)))
    }
    if not observed_name_keys or not observed_name_keys.issubset(allowed_name_keys):
        return None
    return dict(hold)


def apply_permanent_profile_locks_to_raw_rows(
    rows: Sequence[dict[str, object]],
) -> int:
    """Apply exact stable-ID owner locks before a raw profile artifact is written."""

    changed = 0
    locks_by_id = {
        record["stable_ufcstats_id"]: record for record in permanent_profile_locks()
    }
    for row in rows:
        carriers: dict[str, str] = {}
        for field in ("fighter_url", "ufcstats_url", "stable_ufcstats_id"):
            raw_value = row.get(field)
            text = _text(raw_value)
            if not text or text.casefold() in {"nan", "none", "nat", "<na>"}:
                continue
            normalized = normalize_ufcstats_id(raw_value)
            if normalized is None:
                raise ValueError(
                    "Permanent profile lock received an invalid UFCStats identity "
                    f"carrier: {field}={raw_value!r}"
                )
            carriers[field] = normalized
        identities = set(carriers.values())
        if len(identities) > 1:
            raise ValueError(
                "Permanent profile lock received conflicting UFCStats identity "
                f"carriers: {carriers}"
            )
        stable_id = next(iter(identities)) if identities else None
        lock = locks_by_id.get(stable_id or "")
        if lock is None:
            continue
        field = str(lock["field"])
        locked_value = lock["artifact_value"]
        if _text(row.get(field)) != locked_value:
            row[field] = locked_value
            changed += 1
    return changed
