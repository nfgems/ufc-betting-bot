"""Identity-first safeguards for current and model-safe fighter profiles.

The raw profile cache is a current-source observation.  This module keeps that
separate from the narrower values that may enter model features.  It deliberately
does not perform fuzzy identity resolution: names may nominate a source page,
but only stable source entities bound to a UFCStats ID can authorize data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping
from urllib.parse import urlparse

import pandas as pd

from src.data.name_utils import normalize_ufcstats_id
from src.features.stance_utils import STANCE_ENCODING_MAP


MODEL_SAFE_IMMUTABLE_FIELDS = ("height", "reach", "stance", "dob")
CURRENT_ONLY_WEIGHT_FIELD = "weight"
PROFILE_CHANGE_FIELDS = (*MODEL_SAFE_IMMUTABLE_FIELDS, CURRENT_ONLY_WEIGHT_FIELD)
POSITIVE_IDENTITY_STATUSES = frozenset({"verified", "reviewed"})
POSITIVE_NAME_MATCH_STATUSES = frozenset({"exact", "approved_alias"})
POSITIVE_IDENTITY_BASES = frozenset(
    {
        "accepted_reviewed_binding",
        "event_date_opponent",
        "event_date_opponent_organization",
        "stable_ufcstats_url",
    }
)
PROFILE_DECISION_STATUSES = frozenset(
    {"accepted", "owner_adjudicated", "current_only", "rejected", "unresolved"}
)
MODEL_SAFE_PROFILE_DECISION_STATUSES = frozenset({"accepted", "owner_adjudicated"})
TRUSTED_PROFILE_SOURCE_HOSTS: dict[str, tuple[str, ...]] = {
    "ufcstats": ("ufcstats.com",),
    "ufc": ("ufc.com",),
    "espn": ("espn.com",),
    "fightdx": ("fightdx.com",),
    "martialbot": ("martialbot.com",),
    "sherdog": ("sherdog.com",),
    "wikipedia": ("wikipedia.org",),
    "tapology": ("tapology.com",),
}
TRUSTED_EVENT_WEIGHT_SOURCE_HOSTS: dict[str, tuple[str, ...]] = {
    "ufc_official_weigh_in": ("ufc.com",),
    "espn_event_weigh_in": ("espn.com",),
    "sherdog_event_weigh_in": ("sherdog.com",),
    "tapology_event_weigh_in": ("tapology.com",),
}
EVENT_WEIGHT_IDENTITY_BASES = frozenset(
    {
        "event_fighter_opponent_date_side",
        "event_date_opponent_organization",
        "accepted_reviewed_binding",
    }
)

PROFILE_DECISION_COLUMNS = (
    "stable_ufcstats_id",
    "ufcstats_url",
    "canonical_name",
    "field",
    "raw_current_value",
    "model_safe_value",
    "decision_status",
    "primary_source",
    "primary_source_entity_id",
    "primary_requested_url",
    "primary_source_url",
    "primary_canonical_url",
    "primary_redirect_chain",
    "primary_display_name",
    "primary_api_name",
    "primary_sport",
    "primary_raw_value",
    "primary_normalized_value",
    "primary_identity_status",
    "primary_name_match_status",
    "primary_identity_basis",
    "primary_identity_evidence_url",
    "primary_identity_event_date",
    "primary_identity_opponent_name",
    "primary_identity_opponent_ufcstats_id",
    "primary_identity_organization",
    "primary_identity_fight_url",
    "primary_identity_ufcstats_event_date",
    "primary_identity_ufcstats_opponent_name",
    "primary_identity_opponent_name_match_status",
    "corroborating_source",
    "corroborating_source_entity_id",
    "corroborating_requested_url",
    "corroborating_source_url",
    "corroborating_canonical_url",
    "corroborating_redirect_chain",
    "corroborating_display_name",
    "corroborating_api_name",
    "corroborating_sport",
    "corroborating_raw_value",
    "corroborating_normalized_value",
    "corroborating_identity_status",
    "corroborating_name_match_status",
    "corroborating_identity_basis",
    "corroborating_identity_evidence_url",
    "corroborating_identity_event_date",
    "corroborating_identity_opponent_name",
    "corroborating_identity_opponent_ufcstats_id",
    "corroborating_identity_organization",
    "corroborating_identity_fight_url",
    "corroborating_identity_ufcstats_event_date",
    "corroborating_identity_ufcstats_opponent_name",
    "corroborating_identity_opponent_name_match_status",
    "sources_agree",
    "trusted_conflict",
    "retrieved_at_utc",
    "reason",
)


@dataclass(frozen=True)
class ProfileIntegrityIssue:
    code: str
    stable_ufcstats_id: str = ""
    canonical_name: str = ""
    field: str = ""
    detail: str = ""


def _weight_lbs(value: object) -> float | None:
    text = _text(value).casefold()
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    weight = float(match.group())
    if "kg" in text:
        weight *= 2.20462262185
    return weight


def _event_weight_route_url(source: str, value: object) -> str:
    text_value = _text(value)
    allowed_hosts = TRUSTED_EVENT_WEIGHT_SOURCE_HOSTS.get(source)
    if not allowed_hosts:
        return ""

    def _trusted(candidate: str) -> bool:
        parsed_candidate = urlparse(candidate)
        hostname = (parsed_candidate.hostname or "").casefold().rstrip(".")
        return (
            parsed_candidate.scheme.casefold() == "https"
            and not parsed_candidate.query
            and not parsed_candidate.fragment
            and any(
                hostname == host or hostname.endswith(f".{host}")
                for host in allowed_hosts
            )
            and _event_weight_route_kind(source, parsed_candidate.path) is not None
        )

    if _trusted(text_value):
        return text_value
    parsed = urlparse(text_value)
    if (
        parsed.scheme.casefold() != "https"
        or (parsed.hostname or "").casefold() != "r.jina.ai"
    ):
        return ""
    wrapped = parsed.path.lstrip("/")
    return wrapped if wrapped.startswith("https://") and _trusted(wrapped) else ""


def _event_weight_route_kind(source: str, path_value: object) -> tuple[str, str] | None:
    """Return the reviewed route kind/entity for an event-specific source URL.

    Host trust is insufficient: athlete/fighter pages often contain the same
    numeric IDs or slugs as events.  Only source-specific event or weigh-in
    article shapes are eligible.
    """

    path = _text(path_value).casefold().rstrip("/")
    if source == "ufc_official_weigh_in":
        match = re.fullmatch(r"/event/([a-z0-9]+(?:-[a-z0-9]+)*)", path)
        if match:
            return "event", match.group(1)
        match = re.fullmatch(r"/news/([a-z0-9]+(?:-[a-z0-9]+)*)", path)
        if match and "weigh" in match.group(1) and "ufc" in match.group(1):
            return "article", match.group(1)
        return None
    if source == "espn_event_weigh_in":
        match = re.fullmatch(
            r"/mma/fightcenter/_/id/(\d+)(?:/[a-z0-9-]+)?",
            path,
        )
        if match:
            return "event", match.group(1)
        match = re.fullmatch(
            r"/mma/story/_/id/(\d+)/([a-z0-9-]*weigh[a-z0-9-]*)",
            path,
        )
        if match:
            return "article", match.group(1)
        return None
    if source == "tapology_event_weigh_in":
        match = re.fullmatch(
            r"/fightcenter/events/(\d+)(?:-[a-z0-9-]+)?",
            path,
        )
        return ("event", match.group(1)) if match else None
    if source == "sherdog_event_weigh_in":
        match = re.fullmatch(r"/events/[a-z0-9-]+-(\d+)", path)
        return ("event", match.group(1)) if match else None
    return None


def _event_weight_entity_tokens(source: str, value: object) -> set[str]:
    """Extract exact event/article identifiers from one trusted source route."""

    route_url = _event_weight_route_url(source, value)
    if not route_url:
        return set()
    path = urlparse(route_url).path.rstrip("/").casefold()
    route_kind = _event_weight_route_kind(source, path)
    if route_kind is None:
        return set()
    kind, entity = route_kind
    return {f"{kind}:{entity}", path}


def _event_weight_route_matches_entity(
    source: str,
    value: object,
    entity_id: object,
) -> bool:
    entity = _text(entity_id).casefold().rstrip("/")
    return bool(entity and entity in _event_weight_entity_tokens(source, value))


def _supported_ufcstats_fight_url(value: object) -> str:
    parsed = urlparse(_text(value))
    match = re.fullmatch(
        r"/fight-details/([0-9a-f]{16})/?",
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
    return f"http://ufcstats.com/fight-details/{match.group(1).casefold()}"


def validate_event_weight_evidence(
    evidence: Mapping[str, object],
    *,
    fighter_id: object,
    opponent_id: object,
    event_date: object,
    fight_url: object,
    side: str,
) -> list[str]:
    """Validate an exact fight-bound weigh-in observation.

    A current profile weight is intentionally insufficient.  The observation
    must bind both stable participants, the event date, fight URL, and side.
    """

    errors: list[str] = []
    expected_fighter = normalize_ufcstats_id(fighter_id)
    expected_opponent = normalize_ufcstats_id(opponent_id)
    observed_fighter = normalize_ufcstats_id(evidence.get("stable_ufcstats_id"))
    observed_opponent = normalize_ufcstats_id(evidence.get("opponent_ufcstats_id"))
    if not expected_fighter or observed_fighter != expected_fighter:
        errors.append("fighter_identity_mismatch")
    if not expected_opponent or observed_opponent != expected_opponent:
        errors.append("opponent_identity_mismatch")
    expected_date = pd.to_datetime(event_date, errors="coerce")
    observed_date = pd.to_datetime(evidence.get("event_date"), errors="coerce")
    if (
        pd.isna(expected_date)
        or pd.isna(observed_date)
        or pd.Timestamp(expected_date).normalize() != pd.Timestamp(observed_date).normalize()
    ):
        errors.append("event_date_mismatch")
    expected_url = _supported_ufcstats_fight_url(fight_url)
    observed_url = _supported_ufcstats_fight_url(evidence.get("fight_url"))
    if not expected_url or not observed_url:
        errors.append("fight_url_invalid")
    if not expected_url or observed_url != expected_url:
        errors.append("fight_url_mismatch")
    if _text(evidence.get("side")).casefold() != _text(side).casefold():
        errors.append("fighter_side_mismatch")
    if _text(evidence.get("decision_status")).casefold() != "accepted_event_specific":
        errors.append("event_weight_not_accepted")
    if _text(evidence.get("identity_status")).casefold() not in POSITIVE_IDENTITY_STATUSES:
        errors.append("event_weight_identity_not_verified")
    source = _text(evidence.get("source")).casefold()
    allowed_hosts = TRUSTED_EVENT_WEIGHT_SOURCE_HOSTS.get(source)
    if not allowed_hosts:
        errors.append("event_weight_untrusted_source")
    source_url = _text(evidence.get("source_url"))
    canonical_url = _text(evidence.get("canonical_url"))
    for label, value in (("source", source_url), ("canonical", canonical_url)):
        if not allowed_hosts or not _event_weight_route_url(source, value):
            errors.append(f"event_weight_untrusted_{label}_url")
    entity_id = _text(evidence.get("source_entity_id"))
    if not _event_weight_route_matches_entity(source, canonical_url, entity_id):
        errors.append("event_weight_source_entity_mismatch")
    requested_url = _text(evidence.get("requested_url"))
    if not _event_weight_route_url(source, requested_url):
        errors.append("event_weight_untrusted_requested_url")
    elif not _event_weight_route_matches_entity(source, requested_url, entity_id):
        errors.append("event_weight_requested_entity_mismatch")
    if _event_weight_route_url(source, source_url) and not _event_weight_route_matches_entity(
        source, source_url, entity_id
    ):
        errors.append("event_weight_effective_entity_mismatch")
    redirect_chain = _text(evidence.get("redirect_chain"))
    try:
        import json

        parsed_chain = json.loads(redirect_chain)
    except (TypeError, ValueError):
        parsed_chain = None
    if not isinstance(parsed_chain, list):
        errors.append("event_weight_invalid_redirect_chain")
    elif parsed_chain:
        unwrapped_chain = [
            _event_weight_route_url(source, value) for value in parsed_chain
        ]
        if any(not value for value in unwrapped_chain):
            errors.append("event_weight_untrusted_redirect_url")
        elif any(
            not _event_weight_route_matches_entity(source, value, entity_id)
            for value in unwrapped_chain
        ):
            errors.append("event_weight_redirect_entity_mismatch")
        final_url = (unwrapped_chain[-1] if unwrapped_chain else "").rstrip("/")
        if final_url not in {source_url.rstrip("/"), canonical_url.rstrip("/")}:
            errors.append("event_weight_redirect_entity_mismatch")
    if _text(evidence.get("identity_basis")).casefold() not in EVENT_WEIGHT_IDENTITY_BASES:
        errors.append("event_weight_identity_basis_not_event_bound")
    identity_evidence_url = evidence.get("identity_evidence_url")
    if not _event_weight_route_url(source, identity_evidence_url):
        errors.append("event_weight_untrusted_identity_evidence_url")
    elif not _event_weight_route_matches_entity(
        source, identity_evidence_url, entity_id
    ):
        errors.append("event_weight_identity_evidence_entity_mismatch")
    for field in (
        "source",
        "source_entity_id",
        "requested_url",
        "source_url",
        "canonical_url",
        "redirect_chain",
        "retrieval_route",
        "retrieved_at_utc",
        "identity_basis",
        "identity_evidence_url",
        "reason",
        "raw_value",
        "normalized_weight_lbs",
    ):
        if not _text(evidence.get(field)):
            errors.append(f"event_weight_missing_{field}")
    raw_weight = _weight_lbs(evidence.get("raw_value"))
    normalized_weight = _weight_lbs(evidence.get("normalized_weight_lbs"))
    if (
        raw_weight is None
        or normalized_weight is None
        or not abs(raw_weight - normalized_weight) <= 0.05
    ):
        errors.append("event_weight_raw_normalized_mismatch")
    return errors


def event_specific_weight_lbs(
    evidence: Mapping[str, object] | None,
    *,
    fighter_id: object,
    opponent_id: object,
    event_date: object,
    fight_url: object,
    side: str,
) -> float:
    """Return a verified historical weigh-in or NaN when any binding is absent."""

    if evidence is None or validate_event_weight_evidence(
        evidence,
        fighter_id=fighter_id,
        opponent_id=opponent_id,
        event_date=event_date,
        fight_url=fight_url,
        side=side,
    ):
        return float("nan")
    parsed = _weight_lbs(evidence.get("normalized_weight_lbs"))
    return float(parsed) if parsed is not None else float("nan")


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _boolean(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _text(value).casefold()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def source_entity_id(source: object, url: object) -> str:
    """Return the stable entity identifier carried by a trusted source URL."""

    source_key = _text(source).casefold().replace(".com", "")
    text = _text(url)
    parsed = urlparse(text)
    path = parsed.path.rstrip("/")
    if source_key == "ufcstats":
        match = re.fullmatch(
            r"/fighter-details/([0-9a-f]{16})",
            path,
            flags=re.IGNORECASE,
        )
        if (
            parsed.scheme.casefold() != "http"
            or (parsed.hostname or "").casefold().rstrip(".") != "ufcstats.com"
            or parsed.query
            or parsed.fragment
            or match is None
        ):
            return ""
        return match.group(1).casefold()
    if source_key == "espn":
        hostname = (parsed.hostname or "").casefold().rstrip(".")
        if (
            parsed.scheme.casefold() != "https"
            or not (hostname == "espn.com" or hostname.endswith(".espn.com"))
            or parsed.query
            or parsed.fragment
        ):
            return ""
        web = re.fullmatch(
            r"/mma/fighter(?:/bio)?/_/id/(\d+)(?:/[a-z0-9-]+)?",
            path,
            flags=re.IGNORECASE,
        )
        if web:
            return web.group(1)
        api = re.fullmatch(
            r"/v2/sports/mma/athletes/(\d+)",
            path,
            flags=re.IGNORECASE,
        )
        if api and hostname == "sports.core.api.espn.com":
            return api.group(1)
        common_api = re.fullmatch(
            r"/apis/common/v3/sports/mma/athletes/(\d+)",
            path,
            flags=re.IGNORECASE,
        )
        return (
            common_api.group(1)
            if common_api and hostname == "site.web.api.espn.com"
            else ""
        )
    if source_key == "ufc":
        hostname = (parsed.hostname or "").casefold().rstrip(".")
        if (
            parsed.scheme.casefold() != "https"
            or not (hostname == "ufc.com" or hostname.endswith(".ufc.com"))
            or parsed.query
            or parsed.fragment
        ):
            return ""
        node = re.fullmatch(r"/node/(\d+)", path, flags=re.IGNORECASE)
        if node:
            return f"node:{node.group(1)}"
        athlete = re.fullmatch(
            r"/athlete/([a-z0-9]+(?:-[a-z0-9]+)*)",
            path,
            flags=re.IGNORECASE,
        )
        return f"athlete:{athlete.group(1).casefold()}" if athlete else ""
    return text.rstrip("/").casefold()


def _source_key(value: object) -> str:
    return _text(value).casefold().replace(".com", "")


def _ufcstats_fighter_url_matches(value: object, stable_id: str) -> bool:
    parsed = urlparse(_text(value))
    return (
        parsed.scheme.casefold() == "http"
        and (parsed.hostname or "").casefold() == "ufcstats.com"
        and parsed.path.rstrip("/").casefold()
        == f"/fighter-details/{stable_id}".casefold()
        and not parsed.query
        and not parsed.fragment
    )


def _source_url_is_trusted(source: str, value: object) -> bool:
    hosts = TRUSTED_PROFILE_SOURCE_HOSTS.get(source)
    if not hosts:
        return False
    parsed = urlparse(_text(value))
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    allowed_schemes = {"http"} if source == "ufcstats" else {"https"}
    if parsed.scheme.casefold() not in allowed_schemes or not hostname:
        return False
    if not any(hostname == host or hostname.endswith(f".{host}") for host in hosts):
        return False
    if source in {"ufcstats", "ufc", "espn"}:
        return bool(source_entity_id(source, value))
    return True


def _unwrapped_source_route_url(source: str, value: object) -> str:
    text = _text(value)
    if _source_url_is_trusted(source, text):
        return text
    parsed = urlparse(text)
    if (
        parsed.scheme.casefold() != "https"
        or (parsed.hostname or "").casefold() != "r.jina.ai"
    ):
        return ""
    wrapped = parsed.path.lstrip("/")
    if not wrapped.startswith(("http://", "https://")):
        return ""
    return wrapped if _source_url_is_trusted(source, wrapped) else ""


def _source_name_matches_decision(row: pd.Series, prefix: str, source: str) -> bool:
    canonical_name = _text(row.get("canonical_name"))
    name_status = _text(row.get(f"{prefix}_name_match_status")).casefold()
    if source == "espn":
        observed_names = [
            _text(row.get(f"{prefix}_api_name")),
            _text(row.get(f"{prefix}_display_name")),
        ]
        observed_names = [name for name in observed_names if name]
    else:
        observed_names = [_text(row.get(f"{prefix}_display_name"))]
        observed_names = [name for name in observed_names if name]
    if not canonical_name or not observed_names:
        return False
    if name_status == "exact":
        canonical_key = canonical_name.casefold()
        return all(name.casefold() == canonical_key for name in observed_names)
    # An alias is admissible only after a reviewer explicitly records that
    # disposition.  The independent bout/entity evidence remains mandatory.
    return name_status == "approved_alias"


def _redirect_chain_is_captured_and_consistent(
    row: pd.Series,
    prefix: str,
    source: str,
    entity_id: str,
) -> bool:
    requested_url = _text(row.get(f"{prefix}_requested_url"))
    source_url = _text(row.get(f"{prefix}_source_url"))
    redirect_chain = _text(row.get(f"{prefix}_redirect_chain"))
    if not requested_url or not source_url or not redirect_chain:
        return False
    if urlparse(requested_url).scheme.casefold() not in {"http", "https"}:
        return False
    requested_source_url = _unwrapped_source_route_url(source, requested_url)
    if not requested_source_url:
        return False
    requested_entity = source_entity_id(source, requested_source_url)
    if not requested_entity or requested_entity != entity_id:
        return False
    effective_source_url = _unwrapped_source_route_url(source, source_url)
    if not effective_source_url:
        return False
    try:
        import json

        parsed_chain = json.loads(redirect_chain)
    except (TypeError, ValueError):
        parsed_chain = [redirect_chain]
    if not isinstance(parsed_chain, list):
        return False
    redirect_urls = [_text(value) for value in parsed_chain if _text(value)]
    if not redirect_urls:
        return requested_source_url.rstrip("/") == effective_source_url.rstrip("/")
    unwrapped_redirects = [
        _unwrapped_source_route_url(source, value) for value in redirect_urls
    ]
    if any(not value for value in unwrapped_redirects):
        return False
    redirect_entities = [
        source_entity_id(source, redirect_url)
        for redirect_url in unwrapped_redirects
    ]
    identities_match = bool(
        redirect_entities
        and all(
            redirect_entity and redirect_entity == entity_id
            for redirect_entity in redirect_entities
        )
    )
    return bool(
        identities_match
        and unwrapped_redirects[-1].rstrip("/")
        == effective_source_url.rstrip("/")
    )


def _external_identity_event_anchor_positive(
    row: pd.Series,
    prefix: str,
    stable_id: str,
    source: str,
    entity_id: str,
) -> bool:
    source_event_date = pd.to_datetime(
        row.get(f"{prefix}_identity_event_date"), errors="coerce"
    )
    ufcstats_event_date = pd.to_datetime(
        row.get(f"{prefix}_identity_ufcstats_event_date"), errors="coerce"
    )
    source_opponent_name = _text(row.get(f"{prefix}_identity_opponent_name"))
    ufcstats_opponent_name = _text(
        row.get(f"{prefix}_identity_ufcstats_opponent_name")
    )
    opponent_name_match_status = _text(
        row.get(f"{prefix}_identity_opponent_name_match_status")
    ).casefold()
    opponent_id = normalize_ufcstats_id(
        row.get(f"{prefix}_identity_opponent_ufcstats_id")
    )
    organization = _text(row.get(f"{prefix}_identity_organization")).casefold()
    fight_url = _text(row.get(f"{prefix}_identity_fight_url"))
    parsed_fight = urlparse(fight_url)
    fight_id_match = re.fullmatch(
        r"/fight-details/([0-9a-f]{16})/?",
        parsed_fight.path,
        flags=re.IGNORECASE,
    )
    if (
        pd.isna(source_event_date)
        or pd.isna(ufcstats_event_date)
        or pd.Timestamp(source_event_date).normalize()
        != pd.Timestamp(ufcstats_event_date).normalize()
        or not source_opponent_name
        or not ufcstats_opponent_name
        or opponent_name_match_status not in POSITIVE_NAME_MATCH_STATUSES
        or not opponent_id
        or opponent_id == stable_id
        or organization != "ufc"
        or parsed_fight.scheme.casefold() != "http"
        or (parsed_fight.hostname or "").casefold() != "ufcstats.com"
        or not fight_id_match
        or parsed_fight.query
        or parsed_fight.fragment
    ):
        return False
    if (
        opponent_name_match_status == "exact"
        and source_opponent_name.casefold() != ufcstats_opponent_name.casefold()
    ):
        return False
    evidence_url = _text(row.get(f"{prefix}_identity_evidence_url"))
    if not _source_url_is_trusted(source, evidence_url):
        return False
    evidence_entity = source_entity_id(source, evidence_url)
    return bool(evidence_entity and evidence_entity == entity_id)


def _source_binding_positive(row: pd.Series, prefix: str) -> bool:
    source = _source_key(row.get(f"{prefix}_source"))
    entity_id = _text(row.get(f"{prefix}_source_entity_id"))
    source_url = _text(row.get(f"{prefix}_source_url"))
    canonical_url = _text(row.get(f"{prefix}_canonical_url"))
    identity_status = _text(row.get(f"{prefix}_identity_status")).casefold()
    name_status = _text(row.get(f"{prefix}_name_match_status")).casefold()
    basis = _text(row.get(f"{prefix}_identity_basis")).casefold()
    evidence_url = _text(row.get(f"{prefix}_identity_evidence_url"))
    stable_id = normalize_ufcstats_id(row.get("stable_ufcstats_id"))
    if not source or not entity_id or not source_url or not canonical_url or not stable_id:
        return False
    if not _source_url_is_trusted(source, source_url) or not _source_url_is_trusted(
        source, canonical_url
    ):
        return False
    if not _redirect_chain_is_captured_and_consistent(
        row, prefix, source, entity_id
    ):
        return False
    if identity_status not in POSITIVE_IDENTITY_STATUSES:
        return False
    if name_status not in POSITIVE_NAME_MATCH_STATUSES:
        return False
    if basis not in POSITIVE_IDENTITY_BASES or not evidence_url:
        return False
    if source != "ufcstats" and basis == "stable_ufcstats_url":
        return False
    if source != "ufcstats" and not _external_identity_event_anchor_positive(
        row, prefix, stable_id, source, entity_id
    ):
        return False
    if not _source_name_matches_decision(row, prefix, source):
        return False
    extracted = source_entity_id(source, source_url)
    canonical_entity = source_entity_id(source, canonical_url)
    if not extracted or extracted != entity_id or canonical_entity != entity_id:
        return False
    if source == "ufcstats" and entity_id != stable_id:
        return False
    if source == "espn":
        if not _text(row.get(f"{prefix}_api_name")):
            return False
        if _text(row.get(f"{prefix}_sport")).casefold() != "mma":
            return False
    elif not _text(row.get(f"{prefix}_display_name")):
        return False
    return True


def empty_profile_decisions() -> pd.DataFrame:
    """Return a valid, empty decision artifact for a fail-closed comparison."""

    return pd.DataFrame(columns=PROFILE_DECISION_COLUMNS)


def ensure_profile_decision_columns(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in PROFILE_DECISION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Profile integrity decisions are missing required columns: {missing}")
    # Owner instructions and captured disagreements are part of the decision.
    # Older artifacts have only the required columns; preserve any accompanying
    # audit columns instead of stripping the authority for an owner selection.
    extras = [column for column in frame.columns if column not in PROFILE_DECISION_COLUMNS]
    return frame.loc[:, [*PROFILE_DECISION_COLUMNS, *extras]].copy()


def _owner_adjudication_date(row: pd.Series) -> pd.Timestamp:
    recorded = _text(row.get("owner_adjudicated_at_utc")) or _text(
        row.get("owner_adjudication_date")
    )
    return pd.to_datetime(recorded, errors="coerce", utc=True)


def _owner_selection_valid_value(field: str, value: str) -> bool:
    if field == "dob":
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)) and bool(
            pd.notna(pd.to_datetime(value, errors="coerce"))
        )
    if field == "stance":
        return _profile_value(field, value) in STANCE_ENCODING_MAP
    if field in {"height", "reach"}:
        normalized = _profile_value(field, value)
        if not normalized.endswith(":cm"):
            return False
        number = pd.to_numeric(normalized.removesuffix(":cm"), errors="coerce")
        return bool(pd.notna(number) and 0 < number < float("inf"))
    return False


def validate_profile_decisions(
    frame: pd.DataFrame,
    *,
    prior_decisions: pd.DataFrame | None = None,
) -> list[ProfileIntegrityIssue]:
    """Validate decisions and all one-to-one source identity bindings."""

    try:
        decisions = ensure_profile_decision_columns(frame)
    except ValueError as exc:
        return [ProfileIntegrityIssue(code="missing_schema", detail=str(exc))]

    issues: list[ProfileIntegrityIssue] = []
    seen_decisions: set[tuple[str, str]] = set()
    source_to_ufcstats: dict[tuple[str, str], set[str]] = {}
    ufcstats_to_source: dict[tuple[str, str], set[str]] = {}
    prior_owner_decisions: dict[tuple[str, str], pd.Series] = {}

    if prior_decisions is not None:
        prior_issues = validate_profile_decisions(prior_decisions)
        if prior_issues:
            return [
                ProfileIntegrityIssue(
                    code="invalid_prior_profile_decisions",
                    detail="; ".join(issue.code for issue in prior_issues),
                )
            ]
        prior = ensure_profile_decision_columns(prior_decisions)
        for _, prior_row in prior.iterrows():
            prior_stable_id = (
                normalize_ufcstats_id(prior_row.get("stable_ufcstats_id")) or ""
            )
            if _text(prior_row.get("decision_status")).casefold() == "owner_adjudicated":
                prior_owner_decisions[
                    (prior_stable_id, _text(prior_row.get("field")).casefold())
                ] = prior_row
            for prefix in ("primary", "corroborating"):
                if not _source_binding_positive(prior_row, prefix):
                    continue
                prior_source = _source_key(prior_row.get(f"{prefix}_source"))
                prior_entity = _text(prior_row.get(f"{prefix}_source_entity_id"))
                source_to_ufcstats.setdefault(
                    (prior_source, prior_entity), set()
                ).add(prior_stable_id)
                ufcstats_to_source.setdefault(
                    (prior_stable_id, prior_source), set()
                ).add(prior_entity)

    for _, row in decisions.iterrows():
        stable_id = normalize_ufcstats_id(row.get("stable_ufcstats_id")) or ""
        name = _text(row.get("canonical_name"))
        field = _text(row.get("field")).casefold()
        status = _text(row.get("decision_status")).casefold()
        model_value = _text(row.get("model_safe_value"))
        raw_value = _text(row.get("raw_current_value"))
        issue_args = {
            "stable_ufcstats_id": stable_id,
            "canonical_name": name,
            "field": field,
        }

        if not stable_id:
            issues.append(ProfileIntegrityIssue(code="missing_stable_identity", **issue_args))
        if not _ufcstats_fighter_url_matches(row.get("ufcstats_url"), stable_id):
            issues.append(
                ProfileIntegrityIssue(code="ufcstats_url_identity_mismatch", **issue_args)
            )
        if not name:
            issues.append(ProfileIntegrityIssue(code="missing_canonical_name", **issue_args))
        if not _text(row.get("retrieved_at_utc")):
            issues.append(ProfileIntegrityIssue(code="missing_retrieval_time", **issue_args))
        if not _text(row.get("reason")):
            issues.append(ProfileIntegrityIssue(code="missing_decision_reason", **issue_args))
        if field not in PROFILE_CHANGE_FIELDS:
            issues.append(
                ProfileIntegrityIssue(
                    code="unsupported_profile_field", detail=field, **issue_args
                )
            )
        decision_key = (stable_id, field)
        if decision_key in seen_decisions:
            issues.append(ProfileIntegrityIssue(code="duplicate_field_decision", **issue_args))
        seen_decisions.add(decision_key)
        if status not in PROFILE_DECISION_STATUSES:
            issues.append(
                ProfileIntegrityIssue(
                    code="invalid_decision_status", detail=status, **issue_args
                )
            )

        prior_owner = prior_owner_decisions.get(decision_key)
        if prior_owner is not None and (
            status != "owner_adjudicated"
            or _profile_value(field, model_value)
            != _profile_value(field, prior_owner.get("model_safe_value"))
        ):
            prior_date = _owner_adjudication_date(prior_owner)
            current_date = _owner_adjudication_date(row)
            superseded_date = pd.to_datetime(
                _text(row.get("supersedes_owner_adjudication")), errors="coerce", utc=True
            )
            explicit_supersession = bool(
                status == "owner_adjudicated"
                and pd.notna(current_date)
                and pd.notna(prior_date)
                and current_date > prior_date
                and superseded_date == prior_date
                and _text(row.get("owner_adjudication_text"))
                != _text(prior_owner.get("owner_adjudication_text"))
            )
            if not explicit_supersession:
                issues.append(
                    ProfileIntegrityIssue(
                        code="owner_profile_decision_replacement_requires_explicit_supersession",
                        **issue_args,
                    )
                )

        primary_positive = _source_binding_positive(row, "primary")
        corroborating_positive = _source_binding_positive(row, "corroborating")
        for prefix, positive in (
            ("primary", primary_positive),
            ("corroborating", corroborating_positive),
        ):
            if not positive:
                continue
            source = _source_key(row.get(f"{prefix}_source"))
            entity = _text(row.get(f"{prefix}_source_entity_id"))
            source_to_ufcstats.setdefault((source, entity), set()).add(stable_id)
            ufcstats_to_source.setdefault((stable_id, source), set()).add(entity)

        if status == "accepted":
            if field not in MODEL_SAFE_IMMUTABLE_FIELDS:
                issues.append(
                    ProfileIntegrityIssue(
                        code="fighter_level_weight_cannot_be_model_safe", **issue_args
                    )
                )
            if not raw_value or not model_value:
                issues.append(
                    ProfileIntegrityIssue(code="accepted_value_missing", **issue_args)
                )
            if not primary_positive or not corroborating_positive:
                issues.append(
                    ProfileIntegrityIssue(
                        code="accepted_value_lacks_two_verified_identities", **issue_args
                    )
                )
            primary_source = (
                _source_key(row.get("primary_source"))
            )
            corroborating_source = (
                _source_key(row.get("corroborating_source"))
            )
            if field == "stance" and "tapology" in {
                primary_source,
                corroborating_source,
            }:
                issues.append(
                    ProfileIntegrityIssue(
                        code="tapology_stance_evidence_forbidden", **issue_args
                    )
                )
            if primary_source == corroborating_source:
                issues.append(
                    ProfileIntegrityIssue(
                        code="accepted_value_sources_not_independent", **issue_args
                    )
                )
            if _boolean(row.get("sources_agree")) is not True:
                issues.append(
                    ProfileIntegrityIssue(code="accepted_value_sources_disagree", **issue_args)
                )
            if _boolean(row.get("trusted_conflict")) is not False:
                issues.append(
                    ProfileIntegrityIssue(code="accepted_value_has_trusted_conflict", **issue_args)
                )
            primary_normalized = _profile_value(
                field, row.get("primary_normalized_value")
            )
            corroborating_normalized = _profile_value(
                field, row.get("corroborating_normalized_value")
            )
            model_normalized = _profile_value(field, model_value)
            primary_raw_normalized = _profile_value(field, row.get("primary_raw_value"))
            corroborating_raw_normalized = _profile_value(
                field, row.get("corroborating_raw_value")
            )
            if (
                not primary_normalized
                or not corroborating_normalized
                or primary_raw_normalized != primary_normalized
                or corroborating_raw_normalized != corroborating_normalized
                or primary_normalized != corroborating_normalized
                or primary_normalized != model_normalized
            ):
                issues.append(
                    ProfileIntegrityIssue(
                        code="accepted_value_not_equal_across_verified_sources",
                        **issue_args,
                    )
                )
            if field == "stance" and _profile_value(field, model_value) not in STANCE_ENCODING_MAP:
                issues.append(
                    ProfileIntegrityIssue(
                        code="accepted_stance_not_model_encodable", **issue_args
                    )
                )
        elif status == "owner_adjudicated":
            # An explicit owner choice resolves a recorded disagreement without
            # turning it into source consensus. This route does not relax the
            # ordinary two-source acceptance rule above.
            if (
                pd.isna(_owner_adjudication_date(row))
                or not _text(row.get("owner_adjudication_text"))
                or not _text(row.get("independent_verification_status"))
            ):
                issues.append(
                    ProfileIntegrityIssue(code="owner_selection_missing_recorded_authority", **issue_args)
                )
            if not primary_positive and not corroborating_positive:
                issues.append(
                    ProfileIntegrityIssue(code="owner_selection_lacks_verified_identity", **issue_args)
                )
            if not raw_value or not _owner_selection_valid_value(field, model_value):
                issues.append(
                    ProfileIntegrityIssue(code="owner_selection_invalid_immutable_value", **issue_args)
                )
            observations = {
                _profile_value(field, raw_value),
                *(
                    _profile_value(field, row.get(f"{prefix}_raw_value"))
                    for prefix, positive in (
                        ("primary", primary_positive), ("corroborating", corroborating_positive)
                    )
                    if positive
                ),
            }
            if not model_value or _profile_value(field, model_value) not in observations:
                issues.append(
                    ProfileIntegrityIssue(code="owner_selection_has_no_recorded_observation", **issue_args)
                )
            if _boolean(row.get("sources_agree")) is None or _boolean(row.get("trusted_conflict")) is None:
                issues.append(
                    ProfileIntegrityIssue(code="owner_selection_missing_source_disagreement_state", **issue_args)
                )
        elif model_value:
            issues.append(
                ProfileIntegrityIssue(
                    code="nonaccepted_decision_has_model_safe_value", **issue_args
                )
            )

        if status == "current_only" and not primary_positive:
            issues.append(
                ProfileIntegrityIssue(
                    code="current_only_value_lacks_verified_identity", **issue_args
                )
            )
        if status == "current_only":
            if field != CURRENT_ONLY_WEIGHT_FIELD:
                issues.append(
                    ProfileIntegrityIssue(
                        code="current_only_status_is_weight_only", **issue_args
                    )
                )
            primary_raw_normalized = _profile_value(
                field, row.get("primary_raw_value")
            )
            primary_normalized = _profile_value(
                field, row.get("primary_normalized_value")
            )
            if (
                not raw_value
                or not primary_normalized
                or primary_raw_normalized != primary_normalized
                or _profile_value(field, raw_value) != primary_normalized
            ):
                issues.append(
                    ProfileIntegrityIssue(
                        code="current_only_value_not_equal_to_verified_source",
                        **issue_args,
                    )
                )
            if _boolean(row.get("trusted_conflict")) is not False:
                issues.append(
                    ProfileIntegrityIssue(
                        code="current_only_value_has_trusted_conflict", **issue_args
                    )
                )
            if _text(row.get("corroborating_source")):
                if not corroborating_positive or _boolean(row.get("sources_agree")) is not True:
                    issues.append(
                        ProfileIntegrityIssue(
                            code="current_only_corroborator_disagrees", **issue_args
                        )
                    )
                elif _profile_value(
                    field, row.get("primary_normalized_value")
                ) != _profile_value(field, row.get("corroborating_normalized_value")):
                    issues.append(
                        ProfileIntegrityIssue(
                            code="current_only_corroborator_value_mismatch", **issue_args
                        )
                    )

    for (source, entity), ids in source_to_ufcstats.items():
        if len(ids) > 1:
            issues.append(
                ProfileIntegrityIssue(
                    code="source_entity_maps_to_multiple_ufcstats_ids",
                    detail=f"{source}:{entity} -> {sorted(ids)}",
                )
            )
    for (stable_id, source), entities in ufcstats_to_source.items():
        if len(entities) > 1:
            issues.append(
                ProfileIntegrityIssue(
                    code="ufcstats_id_maps_to_multiple_source_entities",
                    stable_ufcstats_id=stable_id,
                    detail=f"{source} -> {sorted(entities)}",
                )
            )
    return issues


def load_profile_decisions(
    path: Path | str,
    *,
    prior_decisions_path: Path | str | None = None,
) -> pd.DataFrame:
    decision_path = Path(path)
    if not decision_path.exists():
        raise FileNotFoundError(f"Profile integrity decision artifact not found: {decision_path}")
    decisions = pd.read_csv(decision_path, dtype=object, keep_default_na=False)
    prior_decisions = None
    if prior_decisions_path is not None:
        prior_path = Path(prior_decisions_path)
        if not prior_path.exists():
            raise FileNotFoundError(
                f"Prior profile integrity decision artifact not found: {prior_path}"
            )
        prior_decisions = pd.read_csv(
            prior_path, dtype=object, keep_default_na=False
        )
    issues = validate_profile_decisions(decisions, prior_decisions=prior_decisions)
    if issues:
        raise ValueError(
            "Profile integrity decision artifact failed validation: "
            + "; ".join(f"{issue.code}:{issue.detail}" for issue in issues)
        )
    current = ensure_profile_decision_columns(decisions)
    if prior_decisions is None:
        return current

    # Decision artifacts are intentionally allowed to be narrow deltas.  A new
    # row replaces the prior decision for that stable-ID/field key (including a
    # newly rejected or unresolved result); every untouched prior key must carry
    # forward so model-safe consumers do not silently blank or revert an earlier
    # accepted correction.
    prior = ensure_profile_decision_columns(prior_decisions)
    current_keys = {
        (
            normalize_ufcstats_id(row.get("stable_ufcstats_id")) or "",
            _text(row.get("field")).casefold(),
        )
        for _, row in current.iterrows()
    }
    retained_prior = prior.loc[
        [
            (
                normalize_ufcstats_id(row.get("stable_ufcstats_id")) or "",
                _text(row.get("field")).casefold(),
            )
            not in current_keys
            for _, row in prior.iterrows()
        ]
    ]
    cumulative = pd.concat([retained_prior, current], ignore_index=True)
    cumulative_issues = validate_profile_decisions(cumulative)
    if cumulative_issues:
        raise ValueError(
            "Cumulative profile integrity decisions failed validation: "
            + "; ".join(
                f"{issue.code}:{issue.detail}" for issue in cumulative_issues
            )
        )
    return ensure_profile_decision_columns(cumulative)


def model_safe_values_by_ufcstats_id(
    decisions: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    issues = validate_profile_decisions(decisions)
    if issues:
        raise ValueError(
            "Cannot materialize model-safe profile values from invalid decisions: "
            + "; ".join(issue.code for issue in issues)
        )
    safe: dict[str, dict[str, str]] = {}
    for _, row in decisions.iterrows():
        if _text(row.get("decision_status")).casefold() not in MODEL_SAFE_PROFILE_DECISION_STATUSES:
            continue
        stable_id = normalize_ufcstats_id(row.get("stable_ufcstats_id"))
        field = _text(row.get("field")).casefold()
        if not stable_id or field not in MODEL_SAFE_IMMUTABLE_FIELDS:
            continue
        safe.setdefault(stable_id, {})[field] = _text(row.get("model_safe_value"))
    return safe


def _profile_value(field: str, value: object) -> str:
    text = _text(value)
    if not text:
        return ""
    if field == "dob":
        parsed = pd.to_datetime(text, errors="coerce")
        return "" if pd.isna(parsed) else pd.Timestamp(parsed).strftime("%Y-%m-%d")
    if field == "stance":
        return re.sub(r"\s+", " ", text).casefold()
    lowered = text.casefold().replace("′", "'").replace("″", '"')
    numbers = [float(part) for part in re.findall(r"[-+]?\d+(?:\.\d+)?", lowered)]
    if numbers:
        number = numbers[0]
        if field == "height":
            feet_inches = re.search(
                r"(\d+(?:\.\d+)?)\s*(?:ft|feet|foot|')\s*"
                r"(\d+(?:\.\d+)?)?\s*(?:in|inches|inch|\")?",
                lowered,
            )
            if feet_inches:
                feet = float(feet_inches.group(1))
                inches = float(feet_inches.group(2) or 0.0)
                return f"{((feet * 12.0) + inches) * 2.54:.12g}:cm"
        if field in {"height", "reach"}:
            if "cm" in lowered:
                return f"{number:.12g}:cm"
            if any(token in lowered for token in ('"', " in", "inch")):
                return f"{number * 2.54:.12g}:cm"
        if field == "weight":
            if "kg" in lowered:
                return f"{number * 2.20462262185:.12g}:lb"
            if any(token in lowered for token in ("lb", "pound")) or re.fullmatch(
                r"[-+]?\d+(?:\.\d+)?", lowered
            ):
                return f"{number:.12g}:lb"
        suffix = re.sub(r"[-+]?\d+(?:\.\d+)?", "", lowered).strip()
        return f"{number:.12g}:{suffix}"
    return text.casefold()


def assess_profile_candidate_changes(
    accepted_profiles: pd.DataFrame,
    candidate_profiles: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    prior_decisions: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Return a publication gate report without mutating either profile frame."""

    decision_issues = validate_profile_decisions(
        decisions, prior_decisions=prior_decisions
    )
    issues: list[ProfileIntegrityIssue] = list(decision_issues)
    decision_lookup: dict[tuple[str, str], pd.Series] = {}
    if not decision_issues:
        decision_lookup = {
            (
                normalize_ufcstats_id(row.get("stable_ufcstats_id")) or "",
                _text(row.get("field")).casefold(),
            ): row
            for _, row in decisions.iterrows()
        }

    def rows_by_id(frame: pd.DataFrame, label: str) -> dict[str, pd.Series]:
        resolved: dict[str, pd.Series] = {}
        for _, row in frame.iterrows():
            identity_carriers: dict[str, str | None] = {}
            for column in (
                "fighter_id",
                "fighter_url",
                "ufcstats_url",
                "stable_ufcstats_id",
            ):
                value = _text(row.get(column))
                if not value:
                    continue
                if column in {"fighter_url", "ufcstats_url"}:
                    identity_carriers[column] = source_entity_id("ufcstats", value) or None
                else:
                    identity_carriers[column] = (
                        value.casefold()
                        if re.fullmatch(r"[0-9a-f]{16}", value, flags=re.IGNORECASE)
                        else None
                    )
            invalid_carriers = sorted(
                column for column, value in identity_carriers.items() if not value
            )
            if invalid_carriers:
                issues.append(
                    ProfileIntegrityIssue(
                        code=f"{label}_invalid_stable_identity_carrier",
                        canonical_name=_text(row.get("name")),
                        detail=",".join(invalid_carriers),
                    )
                )
            stable_ids = {value for value in identity_carriers.values() if value}
            if len(stable_ids) > 1:
                issues.append(
                    ProfileIntegrityIssue(
                        code=f"{label}_conflicting_stable_identity_carriers",
                        canonical_name=_text(row.get("name")),
                        detail=",".join(sorted(stable_ids)),
                    )
                )
                continue
            stable_id = next(iter(stable_ids), None)
            if not stable_id:
                issues.append(
                    ProfileIntegrityIssue(
                        code=f"{label}_row_missing_stable_identity",
                        canonical_name=_text(row.get("name")),
                    )
                )
                continue
            if stable_id in resolved:
                issues.append(
                    ProfileIntegrityIssue(
                        code=f"{label}_duplicate_stable_identity",
                        stable_ufcstats_id=stable_id,
                    )
                )
                continue
            resolved[stable_id] = row
        return resolved

    accepted = rows_by_id(accepted_profiles, "accepted")
    candidate = rows_by_id(candidate_profiles, "candidate")
    for stable_id in sorted(set(accepted).difference(candidate)):
        accepted_row = accepted[stable_id]
        issues.append(
            ProfileIntegrityIssue(
                code="accepted_profile_identity_missing_from_candidate",
                stable_ufcstats_id=stable_id,
                canonical_name=_text(accepted_row.get("name")),
            )
        )
    changed_cells: list[dict[str, str]] = []
    for stable_id, candidate_row in candidate.items():
        accepted_row = accepted.get(stable_id)
        if accepted_row is not None:
            accepted_name = _text(accepted_row.get("name"))
            candidate_name = _text(candidate_row.get("name"))
            if accepted_name.casefold() != candidate_name.casefold():
                changed_cells.append(
                    {
                        "stable_ufcstats_id": stable_id,
                        "canonical_name": candidate_name,
                        "field": "name",
                        "before": accepted_name,
                        "after": candidate_name,
                    }
                )
                issues.append(
                    ProfileIntegrityIssue(
                        code="profile_candidate_name_mismatch",
                        stable_ufcstats_id=stable_id,
                        canonical_name=candidate_name,
                        field="name",
                        detail=accepted_name,
                    )
                )
        for field in PROFILE_CHANGE_FIELDS:
            before = _profile_value(field, accepted_row.get(field) if accepted_row is not None else "")
            after = _profile_value(field, candidate_row.get(field))
            if before == after:
                continue
            name = _text(candidate_row.get("name"))
            changed_cells.append(
                {
                    "stable_ufcstats_id": stable_id,
                    "canonical_name": name,
                    "field": field,
                    "before": before,
                    "after": after,
                }
            )
            if before and not after:
                issues.append(
                    ProfileIntegrityIssue(
                        code="accepted_nonblank_profile_value_lost",
                        stable_ufcstats_id=stable_id,
                        canonical_name=name,
                        field=field,
                    )
                )
                continue
            decision = decision_lookup.get((stable_id, field))
            if decision is None:
                issues.append(
                    ProfileIntegrityIssue(
                        code="profile_change_missing_decision",
                        stable_ufcstats_id=stable_id,
                        canonical_name=name,
                        field=field,
                    )
                )
                continue
            if _text(decision.get("canonical_name")).casefold() != name.casefold():
                issues.append(
                    ProfileIntegrityIssue(
                        code="profile_candidate_name_mismatch",
                        stable_ufcstats_id=stable_id,
                        canonical_name=name,
                        field=field,
                        detail=_text(decision.get("canonical_name")),
                    )
                )
                continue
            if _profile_value(field, decision.get("raw_current_value")) != after:
                issues.append(
                    ProfileIntegrityIssue(
                        code="profile_decision_value_mismatch",
                        stable_ufcstats_id=stable_id,
                        canonical_name=name,
                        field=field,
                    )
                )
                continue
            status = _text(decision.get("decision_status")).casefold()
            allowed = {"accepted", "current_only"}
            if status not in allowed:
                issues.append(
                    ProfileIntegrityIssue(
                        code="unresolved_profile_change_would_publish",
                        stable_ufcstats_id=stable_id,
                        canonical_name=name,
                        field=field,
                        detail=status,
                    )
                )
                continue
            if status == "accepted" and _profile_value(
                field, decision.get("model_safe_value")
            ) != after:
                issues.append(
                    ProfileIntegrityIssue(
                        code="verified_model_safe_value_mismatch",
                        stable_ufcstats_id=stable_id,
                        canonical_name=name,
                        field=field,
                        detail=(
                            "candidate immutable value does not equal the independently "
                            "verified model-safe value"
                        ),
                    )
                )

    return {
        "passed": not issues,
        "changed_cell_count": len(changed_cells),
        "changed_cells": changed_cells,
        "issue_count": len(issues),
        "issues": [asdict(issue) for issue in issues],
    }


def profile_publication_integrity_report(
    accepted_profiles: pd.DataFrame,
    candidate_profiles: pd.DataFrame,
    decisions: pd.DataFrame | None = None,
    *,
    prior_decisions: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Adapt the identity-first profile gate to the repository quality contract.

    Record and career-rate refreshes are deliberately outside this projection.
    When protected profile observations are byte-semantically unchanged, legacy
    rows need no new decision.  Any protected change, however, enters the strict
    stable-ID gate and missing decisions fail closed.
    """

    def projection(frame: pd.DataFrame) -> list[tuple[str, ...]]:
        rows: list[tuple[str, ...]] = []
        for _, row in frame.iterrows():
            identity = next(
                (
                    value
                    for value in (
                        normalize_ufcstats_id(row.get("fighter_id")),
                        normalize_ufcstats_id(row.get("fighter_url")),
                        normalize_ufcstats_id(row.get("ufcstats_url")),
                        normalize_ufcstats_id(row.get("stable_ufcstats_id")),
                    )
                    if value
                ),
                "",
            )
            rows.append(
                (
                    identity,
                    _text(row.get("name")).casefold(),
                    _text(row.get("fighter_id")),
                    _text(row.get("fighter_url")),
                    _text(row.get("ufcstats_url")),
                    _text(row.get("stable_ufcstats_id")),
                    *(_profile_value(field, row.get(field)) for field in PROFILE_CHANGE_FIELDS),
                )
            )
        return sorted(rows)

    supplied_decisions = decisions if decisions is not None else empty_profile_decisions()
    decision_issues = (
        validate_profile_decisions(
            supplied_decisions, prior_decisions=prior_decisions
        )
        if decisions is not None or prior_decisions is not None
        else []
    )
    if projection(accepted_profiles) == projection(candidate_profiles) and not decision_issues:
        return {
            "passed": True,
            "errors": [],
            "details": [],
            "changed_cell_count": 0,
            "changed_cells": [],
            "issue_count": 0,
            "issues": [],
        }

    if projection(accepted_profiles) == projection(candidate_profiles):
        result = {
            "passed": False,
            "changed_cell_count": 0,
            "changed_cells": [],
            "issue_count": len(decision_issues),
            "issues": [asdict(issue) for issue in decision_issues],
        }
    else:
        result = assess_profile_candidate_changes(
            accepted_profiles,
            candidate_profiles,
            supplied_decisions,
            prior_decisions=prior_decisions,
        )
    codes = sorted({str(issue.get("code") or "profile_integrity_failure") for issue in result["issues"]})
    return {
        **result,
        "errors": codes,
        "details": result["issues"],
    }


def verified_binding_keys(decisions: pd.DataFrame) -> set[tuple[str, str, str]]:
    """Return (UFCStats ID, source, entity ID) bindings that passed validation."""

    if validate_profile_decisions(decisions):
        return set()
    keys: set[tuple[str, str, str]] = set()
    for _, row in decisions.iterrows():
        stable_id = normalize_ufcstats_id(row.get("stable_ufcstats_id")) or ""
        for prefix in ("primary", "corroborating"):
            if not _source_binding_positive(row, prefix):
                continue
            keys.add(
                (
                    stable_id,
                    _source_key(row.get(f"{prefix}_source")),
                    _text(row.get(f"{prefix}_source_entity_id")),
                )
            )
    return keys


def issues_as_dicts(issues: Iterable[ProfileIntegrityIssue]) -> list[dict[str, str]]:
    return [asdict(issue) for issue in issues]


__all__ = [
    "CURRENT_ONLY_WEIGHT_FIELD",
    "MODEL_SAFE_IMMUTABLE_FIELDS",
    "MODEL_SAFE_PROFILE_DECISION_STATUSES",
    "PROFILE_CHANGE_FIELDS",
    "PROFILE_DECISION_COLUMNS",
    "ProfileIntegrityIssue",
    "assess_profile_candidate_changes",
    "ensure_profile_decision_columns",
    "empty_profile_decisions",
    "event_specific_weight_lbs",
    "issues_as_dicts",
    "load_profile_decisions",
    "model_safe_values_by_ufcstats_id",
    "profile_publication_integrity_report",
    "source_entity_id",
    "validate_profile_decisions",
    "validate_event_weight_evidence",
    "verified_binding_keys",
]
