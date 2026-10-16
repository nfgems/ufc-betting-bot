"""Owner-reviewed stable-identity dispositions for the separate 55-cell audit.

This registry is deliberately separate from the canonical 118-cell recovery
registry.  Exact source entities may nominate current/future profile values;
names are validation evidence only and never select an identity.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlparse

from src.data.name_utils import normalize_person_name, normalize_ufcstats_id
from src.data.profile_integrity import source_entity_id


REVIEWED_ROSTER_IDENTITY_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "ufc_roster_identity_links_v1.json"
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "audit_scope",
        "identity_bindings",
        "source_identity_rejections",
        "field_decisions",
        "identity_link_holds",
    }
)
_BINDING_KEYS = frozenset(
    {
        "official_name",
        "reviewed_official_names",
        "reviewed_nicknames",
        "reviewed_route_aliases",
        "official_athlete_url",
        "official_source_entity_id",
        "ufcstats_name",
        "stable_ufcstats_id",
        "ufcstats_url",
        "mapped_source_entity_ids",
        "decision_cell_ids",
        "identity_basis",
        "evidence_urls",
    }
)
_REJECTION_KEYS = frozenset(
    {
        "official_name",
        "official_athlete_url",
        "official_source_entity_id",
        "source",
        "source_entity_id",
        "source_url",
        "source_identity_cluster",
        "decision_cell_ids",
        "remove_from_profile_supplement",
        "expected_profile_supplement_row",
    }
)
_FIELD_DECISION_KEYS = frozenset(
    {
        "cell_id",
        "official_source_entity_id",
        "audit_field",
        "profile_field",
        "classification",
        "decision_status",
        "normalized_value",
        "normalized_unit",
        "observation_selector",
        "scope",
        "historical_backfill_allowed",
        "automatic_release_allowed",
    }
)
_SELECTOR_KEYS = frozenset(
    {
        "carrier",
        "source",
        "source_entity_id",
        "source_url",
        "source_name",
        "field",
        "raw_value",
    }
)
_HOLD_KEYS = frozenset(
    {
        "official_name",
        "reviewed_official_names",
        "official_athlete_url",
        "official_source_entity_id",
        "candidate_source_entity_ids",
        "decision_cell_ids",
        "decision_status",
        "automatic_release_allowed",
        "requires_non_name_official_anchor",
        "reason",
        "evidence_urls",
    }
)
_EXPECTED_FIELD_COUNTS = {
    "age": 2,
    "height": 26,
    "reach": 18,
    "stance": 4,
    "weight": 5,
}
_EXPECTED_CLASSIFICATION_COUNTS = {
    "verified alias/path miss": 39,
    "conflicting": 9,
    "wrong identity": 5,
    "unresolved": 2,
}
_EXPECTED_STATUS_COUNTS = {
    "accepted_current_future": 41,
    "conflicting_field_hold": 6,
    "owner_field_hold": 1,
    "rejected_wrong_identity": 5,
    "unresolved_identity_hold": 2,
}
_EXPECTED_ACCEPTED_FIELD_COUNTS = {
    "age": 1,
    "height": 18,
    "reach": 16,
    "stance": 4,
    "weight": 2,
}
_EXPECTED_FALSE_CELL_IDS = frozenset(
    {
        "D001",
        "D002",
        "D009",
        "D012",
        "D015",
        "D016",
        "D022",
        "D023",
        "D025",
        "D026",
        "D028",
        "D029",
        "D030",
        "D054",
    }
)
_EXPECTED_NEW_BINDINGS = {
    "athlete:khadzhimurat-bestaev": "e05f4ce199baca91",
    "athlete:michael-page": "a67d071163962af8",
    "athlete:rheza-arianto": "90063913321e7670",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if value != value:  # NaN/NaT without importing a dataframe library.
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "nat", "<na>"} else text


def _require_exact_keys(
    value: Mapping[str, object], keys: frozenset[str], label: str
) -> None:
    actual = frozenset(value)
    if actual != keys:
        raise ValueError(
            f"{label} has invalid keys "
            f"(missing={sorted(keys - actual)}, extra={sorted(actual - keys)})"
        )


def _require_strings(value: object, label: str, *, minimum: int = 1) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(f"{label} must contain at least {minimum} strings")
    items = [_text(item) for item in value]
    if any(not item for item in items) or len(items) != len(set(items)):
        raise ValueError(f"{label} must contain unique nonblank strings")
    return items


def _official_entity(row: Mapping[str, object]) -> str:
    return source_entity_id("ufc", row.get("official_athlete_url"))


def _stable_carriers(row: Mapping[str, object]) -> tuple[set[str], bool]:
    identities: set[str] = set()
    invalid = False
    for field in ("stable_ufcstats_id", "fighter_id", "ufcstats_url"):
        value = row.get(field)
        if not _text(value):
            continue
        stable_id = normalize_ufcstats_id(value)
        if stable_id is None:
            invalid = True
        else:
            identities.add(stable_id)
    return identities, invalid


def _canonical_source_name(value: object) -> str:
    source = _text(value).casefold()
    if source.endswith(".com"):
        source = source[:-4]
    return source if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", source) else ""


def _canonical_source_identity(source: object, entity_id: object) -> str:
    """Return one namespaced, case-insensitive identity key or fail closed."""

    source_key = _canonical_source_name(source)
    value = _text(entity_id).casefold()
    if not source_key or not value:
        return ""
    if source_key == "ufcstats":
        if value.startswith("ufcstats:"):
            value = value.removeprefix("ufcstats:")
        return f"ufcstats:{value}" if re.fullmatch(r"[0-9a-f]{16}", value) else ""
    if source_key == "ufc":
        if value.startswith("ufc:"):
            value = value.removeprefix("ufc:")
        if re.fullmatch(r"athlete:[a-z0-9]+(?:-[a-z0-9]+)*|node:\d+", value):
            return f"ufc:{value}"
        return ""
    prefix = f"{source_key}:"
    if not value.startswith(prefix):
        return ""
    suffix = value[len(prefix):]
    if re.fullmatch(r"[a-z0-9][a-z0-9._:-]*", suffix) is None:
        return ""
    return f"{source_key}:{suffix}"


def _canonical_mapped_source_identity(value: object) -> str:
    text = _text(value)
    source, separator, _ = text.partition(":")
    return _canonical_source_identity(source, text) if separator else ""


def _selector_url_source_identity(source: object, url: object) -> str:
    """Extract a selector's exact source identity from a trusted URL grammar."""

    source_key = _canonical_source_name(source)
    text = _text(url)
    parsed = urlparse(text)
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    netloc = parsed.netloc.casefold().rstrip(".")
    path = parsed.path.rstrip("/")
    if (
        not source_key
        or not text
        or not hostname
        or netloc != hostname
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        return ""

    if source_key in {"espn", "ufc", "ufcstats"}:
        entity = source_entity_id(source_key, text)
        if source_key == "espn" and entity:
            entity = f"espn:{entity}"
        return _canonical_source_identity(source_key, entity)

    if parsed.scheme.casefold() != "https":
        return ""
    if source_key == "tapology":
        if hostname not in {"tapology.com", "www.tapology.com"}:
            return ""
        match = re.fullmatch(
            r"/fightcenter/fighters/(\d+)(?:-[a-z0-9]+(?:-[a-z0-9]+)*)?",
            path,
            flags=re.IGNORECASE,
        )
        return f"tapology:{match.group(1)}" if match else ""
    if source_key == "sherdog":
        if hostname not in {"sherdog.com", "www.sherdog.com"}:
            return ""
        match = re.fullmatch(
            r"/fighter/[a-z0-9-]+-(\d+)",
            path,
            flags=re.IGNORECASE,
        )
        return f"sherdog:{match.group(1)}" if match else ""
    if source_key == "fightdx":
        if hostname not in {"fightdx.com", "www.fightdx.com"}:
            return ""
        match = re.fullmatch(
            r"/person/([a-z0-9]+(?:-[a-z0-9]+)*)",
            path,
            flags=re.IGNORECASE,
        )
        return f"fightdx:{match.group(1).casefold()}" if match else ""
    return ""


def _validate_registry(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("Reviewed roster identity registry must be an object")
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "reviewed roster identity registry")
    if payload.get("schema_version") != 1:
        raise ValueError("Reviewed roster identity registry schema_version must be 1")

    scope = payload.get("audit_scope")
    if not isinstance(scope, dict):
        raise ValueError("Reviewed roster identity audit_scope must be an object")
    for key in (
        "decision_report_sha256",
        "decision_report_csv_sha256",
        "derived_queue_sha256",
    ):
        if _SHA256_RE.fullmatch(_text(scope.get(key)).casefold()) is None:
            raise ValueError(f"Reviewed roster identity audit_scope has invalid {key}")
    if (
        scope.get("decision_report_sha256")
        != "df1b7d02f037c08a018fce49fa5f498e87d1da984d4a7bfc385f9c16eaea271b"
        or scope.get("decision_report_csv_sha256")
        != "2902d9ed0b628c444ad872fd849c9309623bd86eff03fb63f88d679fc1664c21"
        or scope.get("derived_queue_sha256")
        != "b2e0bc9d25a66d868b1696e5934802f0da7104a8666a632f335541ce2e480f76"
        or scope.get("cells") != 55
        or scope.get("fighters") != 31
        or scope.get("field_counts") != _EXPECTED_FIELD_COUNTS
        or scope.get("classification_counts") != _EXPECTED_CLASSIFICATION_COUNTS
        or scope.get("decision_status_counts") != _EXPECTED_STATUS_COUNTS
    ):
        raise ValueError("Reviewed roster identity audit scope is inconsistent")

    bindings = payload.get("identity_bindings")
    rejections = payload.get("source_identity_rejections")
    decisions = payload.get("field_decisions")
    holds = payload.get("identity_link_holds")
    if not isinstance(bindings, list) or len(bindings) != 27:
        raise ValueError("Reviewed identity registry must contain exactly 27 bindings")
    if not isinstance(rejections, list) or len(rejections) != 11:
        raise ValueError("Reviewed identity registry must contain exactly 11 rejections")
    if not isinstance(decisions, list) or len(decisions) != 55:
        raise ValueError("Reviewed identity registry must contain exactly 55 field decisions")
    if not isinstance(holds, list) or len(holds) != 1:
        raise ValueError("Reviewed identity registry must contain exactly one link hold")

    binding_entities: set[str] = set()
    binding_cells: set[str] = set()
    stable_ids: set[str] = set()
    active_source_owners: dict[str, str] = {}
    binding_owned_source_ids: dict[str, set[str]] = {}
    for position, record in enumerate(bindings, start=1):
        label = f"reviewed identity binding {position}"
        if not isinstance(record, dict):
            raise ValueError(f"{label} must be an object")
        _require_exact_keys(record, _BINDING_KEYS, label)
        for field in (
            "official_name",
            "official_athlete_url",
            "official_source_entity_id",
            "identity_basis",
        ):
            if not _text(record.get(field)):
                raise ValueError(f"{label} has blank {field}")
        names = _require_strings(record["reviewed_official_names"], f"{label}.reviewed_official_names")
        _require_strings(record["decision_cell_ids"], f"{label}.decision_cell_ids")
        _require_strings(record["evidence_urls"], f"{label}.evidence_urls", minimum=2)
        if not isinstance(record["reviewed_nicknames"], list) or not isinstance(record["reviewed_route_aliases"], list):
            raise ValueError(f"{label} reviewed alias fields must be lists")
        mapped_source_ids = _require_strings(
            record["mapped_source_entity_ids"],
            f"{label}.mapped_source_entity_ids",
            minimum=0,
        )
        if normalize_person_name(record["official_name"]) not in {
            normalize_person_name(name) for name in names
        }:
            raise ValueError(f"{label} does not authorize its official name")
        entity = source_entity_id("ufc", record["official_athlete_url"])
        if not entity or entity != record["official_source_entity_id"] or entity in binding_entities:
            raise ValueError(f"{label} has an invalid or duplicate official entity")
        binding_entities.add(entity)
        cells = set(record["decision_cell_ids"])
        if binding_cells.intersection(cells):
            raise ValueError(f"{label} overlaps another identity binding")
        binding_cells.update(cells)
        stable_id = normalize_ufcstats_id(record["stable_ufcstats_id"])
        if _text(record["stable_ufcstats_id"]):
            if (
                not stable_id
                or stable_id != source_entity_id("ufcstats", record["ufcstats_url"])
                or not _text(record["ufcstats_name"])
                or stable_id in stable_ids
            ):
                raise ValueError(f"{label} has an invalid UFCStats binding")
            stable_ids.add(stable_id)
        elif record["ufcstats_name"] != "" or record["ufcstats_url"] != "":
            raise ValueError(f"{label} official-only binding has UFCStats carriers")

        owned_source_ids = {_canonical_source_identity("ufc", entity)}
        if stable_id:
            owned_source_ids.add(_canonical_source_identity("ufcstats", stable_id))
        canonical_mapped_ids = [
            _canonical_mapped_source_identity(source_id)
            for source_id in mapped_source_ids
        ]
        if any(not source_id for source_id in canonical_mapped_ids):
            raise ValueError(f"{label} has an invalid mapped source entity ID")
        if len(canonical_mapped_ids) != len(set(canonical_mapped_ids)):
            raise ValueError(
                f"{label}.mapped_source_entity_ids must be unique case-insensitively"
            )
        owned_source_ids.update(canonical_mapped_ids)
        for source_id in sorted(owned_source_ids):
            prior_owner = active_source_owners.get(source_id)
            if prior_owner is not None and prior_owner != entity:
                raise ValueError(
                    f"{label} assigns active source ID {source_id} to multiple fighters"
                )
            active_source_owners[source_id] = entity
        binding_owned_source_ids[entity] = owned_source_ids
    if len(stable_ids) != 21 or len(bindings) - len(stable_ids) != 6:
        raise ValueError("Reviewed identity bindings must be 21 UFCStats + 6 official-only")

    rejected_pairs: set[tuple[str, str]] = set()
    rejected_source_ids: set[str] = set()
    rejection_cells: set[str] = set()
    supplement_removals = 0
    for position, record in enumerate(rejections, start=1):
        label = f"reviewed source rejection {position}"
        if not isinstance(record, dict):
            raise ValueError(f"{label} must be an object")
        _require_exact_keys(record, _REJECTION_KEYS, label)
        for field in (
            "official_name",
            "official_athlete_url",
            "official_source_entity_id",
            "source",
            "source_entity_id",
            "source_url",
            "source_identity_cluster",
        ):
            if not _text(record.get(field)):
                raise ValueError(f"{label} has blank {field}")
        entity = source_entity_id("ufc", record["official_athlete_url"])
        if entity != record["official_source_entity_id"]:
            raise ValueError(f"{label} official entity is inconsistent")
        rejected_source_id = _canonical_source_identity(
            record["source"], record["source_entity_id"]
        )
        if not rejected_source_id:
            raise ValueError(f"{label} has an invalid rejected source entity ID")
        pair = (entity, rejected_source_id)
        if pair in rejected_pairs:
            raise ValueError(f"{label} duplicates a rejected identity pair")
        rejected_pairs.add(pair)
        rejected_source_ids.add(rejected_source_id)
        cells = _require_strings(record["decision_cell_ids"], f"{label}.decision_cell_ids")
        rejection_cells.update(cells)
        remove = record["remove_from_profile_supplement"]
        expected_row = record["expected_profile_supplement_row"]
        if not isinstance(remove, bool):
            raise ValueError(f"{label} removal flag must be boolean")
        if remove:
            supplement_removals += 1
            if not isinstance(expected_row, dict) or expected_row.get("fighter_url") != record["source_url"]:
                raise ValueError(f"{label} has an invalid supplement removal pin")
        elif expected_row is not None:
            raise ValueError(f"{label} non-removal carries a supplement row")
    if supplement_removals != 6:
        raise ValueError("Reviewed source rejections must pin exactly six supplement removals")
    active_rejected_overlap = set(active_source_owners).intersection(rejected_source_ids)
    if active_rejected_overlap:
        raise ValueError(
            "Reviewed source identities cannot be both active and rejected: "
            f"{sorted(active_rejected_overlap)}"
        )

    decision_ids: set[str] = set()
    status_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    accepted_field_counts: Counter[str] = Counter()
    false_ids: set[str] = set()
    decision_entities: set[str] = set()
    for position, record in enumerate(decisions, start=1):
        label = f"reviewed field decision {position}"
        if not isinstance(record, dict):
            raise ValueError(f"{label} must be an object")
        _require_exact_keys(record, _FIELD_DECISION_KEYS, label)
        cell_id = _text(record["cell_id"])
        if not re.fullmatch(r"D\d{3}", cell_id) or cell_id in decision_ids:
            raise ValueError(f"{label} has an invalid or duplicate cell ID")
        decision_ids.add(cell_id)
        decision_entities.add(_text(record["official_source_entity_id"]))
        audit_field = _text(record["audit_field"])
        profile_field = _text(record["profile_field"])
        if audit_field not in _EXPECTED_FIELD_COUNTS or profile_field != ("dob" if audit_field == "age" else audit_field):
            raise ValueError(f"{label} has inconsistent field carriers")
        status = _text(record["decision_status"])
        classification = _text(record["classification"])
        status_counts[status] += 1
        class_counts[classification] += 1
        field_counts[audit_field] += 1
        if (
            record["scope"] != "current_and_future"
            or record["historical_backfill_allowed"] is not False
            or record["automatic_release_allowed"] is not False
        ):
            raise ValueError(f"{label} has invalid temporal scope")
        selector = record["observation_selector"]
        if status == "accepted_current_future":
            accepted_field_counts[audit_field] += 1
            if record["normalized_value"] is None or not isinstance(selector, dict):
                raise ValueError(f"{label} accepted field lacks its exact observation")
            _require_exact_keys(selector, _SELECTOR_KEYS, f"{label}.observation_selector")
            if selector["carrier"] not in {"supplement", "official", "ufcstats"} or selector["field"] != profile_field:
                raise ValueError(f"{label} observation selector is inconsistent")
            for field in ("source", "source_entity_id", "source_url", "source_name", "raw_value"):
                if not _text(selector.get(field)):
                    raise ValueError(f"{label} observation selector has blank {field}")
            carrier = selector["carrier"]
            source_key = _canonical_source_name(selector["source"])
            carrier_source_valid = (
                (carrier == "supplement" and source_key in {"espn", "fightdx", "sherdog", "tapology"})
                or (carrier == "official" and source_key == "ufc")
                or (carrier == "ufcstats" and source_key == "ufcstats")
            )
            owner = _text(record["official_source_entity_id"])
            declared_source_id = _canonical_source_identity(
                source_key, selector["source_entity_id"]
            )
            url_source_id = _selector_url_source_identity(
                source_key, selector["source_url"]
            )
            if (
                not carrier_source_valid
                or owner not in binding_owned_source_ids
                or not declared_source_id
                or declared_source_id != url_source_id
                or declared_source_id not in binding_owned_source_ids[owner]
            ):
                raise ValueError(
                    f"{label} observation selector is not owned by its identity binding"
                )
        else:
            false_ids.add(cell_id)
            if selector is not None or record["normalized_value"] is not None:
                raise ValueError(f"{label} nonaccepted field carries a value")
    if decision_ids != {f"D{number:03d}" for number in range(1, 56)}:
        raise ValueError("Reviewed field decisions must cover exact D001-D055")
    if dict(status_counts) != _EXPECTED_STATUS_COUNTS:
        raise ValueError(f"Reviewed field decision status drift: {dict(status_counts)}")
    if dict(class_counts) != _EXPECTED_CLASSIFICATION_COUNTS:
        raise ValueError(f"Reviewed field classification drift: {dict(class_counts)}")
    if dict(field_counts) != _EXPECTED_FIELD_COUNTS:
        raise ValueError(f"Reviewed field-count drift: {dict(field_counts)}")
    if dict(accepted_field_counts) != _EXPECTED_ACCEPTED_FIELD_COUNTS:
        raise ValueError(f"Reviewed accepted field-count drift: {dict(accepted_field_counts)}")
    if false_ids != _EXPECTED_FALSE_CELL_IDS:
        raise ValueError("Reviewed false-cell set drift")
    if binding_cells != decision_ids - {"D015", "D016", "D022", "D023", "D028", "D029", "D030"}:
        raise ValueError("Reviewed identity binding cell coverage is inconsistent")

    hold = holds[0]
    if not isinstance(hold, dict):
        raise ValueError("Reviewed identity link hold must be an object")
    _require_exact_keys(hold, _HOLD_KEYS, "reviewed identity link hold")
    if (
        hold["official_name"] != "Keifer Roberts"
        or hold["official_source_entity_id"] != "athlete:keifer-roberts"
        or source_entity_id("ufc", hold["official_athlete_url"]) != "athlete:keifer-roberts"
        or hold["candidate_source_entity_ids"] != ["tapology:52400", "sherdog:84920"]
        or hold["decision_cell_ids"] != ["D022", "D023"]
        or hold["decision_status"] != "unresolved_identity_hold"
        or hold["automatic_release_allowed"] is not False
        or hold["requires_non_name_official_anchor"] is not True
    ):
        raise ValueError("Keifer Roberts identity-link hold is inconsistent")
    if not decision_entities.issubset(binding_entities | {record["official_source_entity_id"] for record in rejections} | {"athlete:keifer-roberts"}):
        raise ValueError("Reviewed field decision has no exact official identity disposition")
    return payload


@lru_cache(maxsize=None)
def load_reviewed_roster_identity_registry(
    path: Path = REVIEWED_ROSTER_IDENTITY_REGISTRY_PATH,
) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _validate_registry(payload)


def _binding_for_entity(entity: str) -> dict[str, object] | None:
    matches = [
        record
        for record in load_reviewed_roster_identity_registry()["identity_bindings"]
        if record["official_source_entity_id"] == entity
    ]
    if len(matches) > 1:
        raise ValueError(f"Reviewed roster identity is ambiguous for {entity}")
    return dict(matches[0]) if matches else None


def reviewed_profile_identity_binding_record(
    row: Mapping[str, object],
) -> dict[str, object] | None:
    """Select a binding only by the exact stable UFC official entity."""

    entity = _official_entity(row)
    return _binding_for_entity(entity) if entity else None


def reviewed_profile_identity_binding(
    row: Mapping[str, object],
) -> dict[str, object] | None:
    """Validate reviewed display evidence and every supplied UFCStats carrier."""

    record = reviewed_profile_identity_binding_record(row)
    if record is None:
        return None
    observed_name = normalize_person_name(row.get("official_name"))
    allowed_names = {
        normalize_person_name(name) for name in record["reviewed_official_names"]
    }
    if observed_name and observed_name not in allowed_names:
        return None
    identities, invalid = _stable_carriers(row)
    expected = _text(record["stable_ufcstats_id"])
    if invalid or len(identities) > 1:
        return None
    if identities and identities != ({expected} if expected else set()):
        return None
    return record


def apply_reviewed_profile_identity_binding(row: dict[str, object]) -> bool:
    record = reviewed_profile_identity_binding(row)
    if record is None or not record["stable_ufcstats_id"]:
        return False
    row.update(
        {
            "ufcstats_name": record["ufcstats_name"],
            "ufcstats_url": record["ufcstats_url"],
            "ufcstats_resolution": "reviewed_roster_identity_link",
            "combat_sport": "mma",
            "combat_sport_reason": "ufcstats_profile_match",
        }
    )
    return True


def reviewed_profile_official_lookup_key(row: Mapping[str, object]) -> str:
    """Return the sole approved current-profile lookup key for an official row."""

    record = reviewed_profile_identity_binding(row)
    if record is None:
        return ""
    stable_id = _text(record["stable_ufcstats_id"])
    return (
        f"ufcstats:{stable_id}"
        if stable_id
        else f"ufc:{record['official_source_entity_id']}"
    )


def reviewed_profile_field_decisions(
    row: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    records = load_reviewed_roster_identity_registry()["field_decisions"]
    if row is None:
        return [dict(record) for record in records]
    entity = _official_entity(row)
    if not entity:
        return []
    binding = _binding_for_entity(entity)
    hold = next(
        (
            item
            for item in load_reviewed_roster_identity_registry()["identity_link_holds"]
            if item["official_source_entity_id"] == entity
        ),
        None,
    )
    rejected = [
        item
        for item in load_reviewed_roster_identity_registry()["source_identity_rejections"]
        if item["official_source_entity_id"] == entity
    ]
    allowed_names: set[str] = set()
    if binding:
        allowed_names.update(normalize_person_name(name) for name in binding["reviewed_official_names"])
    if hold:
        allowed_names.update(normalize_person_name(name) for name in hold["reviewed_official_names"])
    allowed_names.update(normalize_person_name(item["official_name"]) for item in rejected)
    observed_name = normalize_person_name(row.get("official_name"))
    if observed_name and observed_name not in allowed_names:
        return []
    return [
        dict(record)
        for record in records
        if record["official_source_entity_id"] == entity
    ]


def _source_url_key(value: object) -> str:
    parsed = urlparse(_text(value))
    return parsed._replace(fragment="").geturl().rstrip("/")


def reviewed_profile_observations(
    supplement_rows: Sequence[Mapping[str, object]],
    official_rows: Sequence[Mapping[str, object]],
) -> list[tuple[dict[str, object], dict[str, object]]]:
    """Select only exact reviewed supplement/UFC observations and raw values."""

    selected: list[tuple[dict[str, object], dict[str, object]]] = []
    for raw_decision in reviewed_profile_field_decisions():
        if raw_decision["decision_status"] != "accepted_current_future":
            continue
        decision = dict(raw_decision)
        selector = dict(decision["observation_selector"])
        carrier = selector["carrier"]
        if carrier == "ufcstats":
            continue
        rows = supplement_rows if carrier == "supplement" else official_rows
        if carrier == "supplement":
            potential = [
                dict(row)
                for row in rows
                if _text(row.get("source")).casefold() == selector["source"]
                and _source_url_key(row.get("fighter_url"))
                == _source_url_key(selector["source_url"])
            ]
            name_field = "source_name"
        elif carrier == "official":
            potential = [
                dict(row)
                for row in rows
                if _source_url_key(row.get("official_athlete_url"))
                == _source_url_key(selector["source_url"])
            ]
            name_field = "official_name"
        else:
            raise ValueError(f"Unsupported reviewed observation carrier: {carrier}")
        if len(potential) > 1:
            raise ValueError(
                "Reviewed profile observation has duplicate exact source entities: "
                f"{selector['source_entity_id']}"
            )
        if not potential:
            continue
        observation = potential[0]
        if (
            _text(observation.get(name_field)) != selector["source_name"]
            or _text(observation.get(selector["field"])) != selector["raw_value"]
        ):
            raise ValueError(
                "Reviewed profile observation conflicts with its exact selector: "
                f"{decision['cell_id']}"
            )
        selected.append((decision, observation))
    return selected


def reviewed_ufcstats_identity_rejection(
    row: Mapping[str, object],
) -> dict[str, object] | None:
    """Return an exact official↔UFCStats negative mapping carried by the row."""

    entity = _official_entity(row)
    if not entity:
        return None
    identities, invalid = _stable_carriers(row)
    if invalid or len(identities) > 1:
        return None
    matches = [
        record
        for record in load_reviewed_roster_identity_registry()["source_identity_rejections"]
        if record["official_source_entity_id"] == entity
        and record["source"] == "ufcstats"
        and record["source_entity_id"].removeprefix("ufcstats:") in identities
    ]
    if len(matches) > 1:
        raise ValueError(f"Multiple reviewed UFCStats rejections match {entity}")
    return dict(matches[0]) if matches else None


def reviewed_ufcstats_resolution_blocked(row: Mapping[str, object]) -> bool:
    """Block name-only UFCStats discovery where the review did not approve it."""

    entity = _official_entity(row)
    if not entity:
        return False
    registry = load_reviewed_roster_identity_registry()
    binding = _binding_for_entity(entity)
    if binding is not None:
        return not bool(binding["stable_ufcstats_id"])
    if any(
        hold["official_source_entity_id"] == entity
        for hold in registry["identity_link_holds"]
    ):
        return True
    return any(
        rejection["official_source_entity_id"] == entity
        for rejection in registry["source_identity_rejections"]
    )


def expected_new_ufcstats_bindings() -> dict[str, str]:
    """Exact 55-layer roster mutations, separate from the canonical 118 layer."""

    return dict(_EXPECTED_NEW_BINDINGS)


__all__ = [
    "REVIEWED_ROSTER_IDENTITY_REGISTRY_PATH",
    "apply_reviewed_profile_identity_binding",
    "expected_new_ufcstats_bindings",
    "load_reviewed_roster_identity_registry",
    "reviewed_profile_field_decisions",
    "reviewed_profile_identity_binding",
    "reviewed_profile_identity_binding_record",
    "reviewed_profile_observations",
    "reviewed_profile_official_lookup_key",
    "reviewed_ufcstats_identity_rejection",
    "reviewed_ufcstats_resolution_blocked",
]
