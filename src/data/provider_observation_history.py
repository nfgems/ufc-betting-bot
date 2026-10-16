"""Read and retain explicit provider responses without acquiring source inputs.

The caller supplies reviewed owner bindings independently of the saved response.
Original bytes and envelopes use the existing immutable observation publisher;
references are additive and do not authorize any collector or generation pruning.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from urllib.parse import urljoin, urlparse

from src.data.method_observation_history import (
    ObservationRefusal,
    _attached,
    _attachment,
    _json_bytes,
    _safe_path,
    _write_immutable,
)
from src.data.name_utils import normalize_person_name
from src.data.profile_integrity import _source_url_is_trusted, source_entity_id

PARSER_REVISION = "retained_provider_profile_history_v2"
ELIGIBILITY_POLICY = "provider_original_capture_asof_v1"
HISTORY_SUBDIR = "provider_observations_v1"


class ProviderObservationError(ValueError):
    """A retained input is missing, ambiguous, corrupt, or ineligible."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ProviderObservationError(message)


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _time(value: object) -> datetime:
    _require(isinstance(value, (str, datetime)), "explicit observation cutoff/time required")
    try:
        result = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProviderObservationError("invalid observation cutoff/time") from exc
    _require(result.tzinfo is not None, "observation cutoff/time must have timezone")
    return result.astimezone(timezone.utc)


def _text_time(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _provider_number(source: str, url: str) -> str:
    parsed = urlparse(url)
    if not _source_url_is_trusted(source, url) or parsed.username or parsed.password or parsed.port not in {None, 443} or parsed.query or parsed.fragment:
        return ""
    pattern = r"/fightcenter/fighters/(\d+)(?:-[^/]+)?/?" if source == "tapology" else r"/fighter/[^/]+-(\d+)/?"
    match = re.fullmatch(pattern, parsed.path)
    return match.group(1) if match else ""


def _input_document(descriptor: Mapping | Path | str) -> tuple[dict, str]:
    input_path = ""
    if isinstance(descriptor, (Path, str)):
        path = _safe_path(Path(descriptor))
        input_path = str(path)
        document = json.loads(path.read_bytes())
    else:
        document = dict(descriptor) if isinstance(descriptor, Mapping) else None
    _require(isinstance(document, dict), "missing explicit retained provider input")
    _require(type(document.get("schema")) is int and document["schema"] == 1, "unknown provider observation schema")
    if document.get("kind") == HISTORY_SUBDIR:
        _require(set(document) == {"schema", "kind", "source", "response", "envelope", "fighter_id", "provider_id", "parser_revision", "observation_id"}, "invalid finalized provider evidence")
        _require(document["parser_revision"] == PARSER_REVISION, "unknown retained provider parser")
        if input_path:
            _require(Path(input_path).stem == document["observation_id"], "retained provider filename mismatch")
        return document, input_path
    _require(set(document) == {"schema", "source", "raw_response", "response_envelope"}, "invalid explicit provider input descriptor")
    return {
        "schema": 1,
        "kind": HISTORY_SUBDIR,
        "source": document["source"],
        "response": _attachment(document["raw_response"]),
        "envelope": _attachment(document["response_envelope"]),
    }, input_path


def _validated_envelope(body: bytes, envelope_body: bytes, available_before: object) -> tuple[dict, datetime, datetime]:
    envelope = json.loads(envelope_body)
    _require(isinstance(envelope, dict) and set(envelope) == {"schema", "status_code", "requested_url", "effective_url", "redirect_chain", "received_at", "response_sha256", "response_bytes", "format"}, "incomplete saved response envelope")
    _require(type(envelope["schema"]) is int and envelope["schema"] == 1 and type(envelope["status_code"]) is int and envelope["status_code"] == 200, "response does not prove source success")
    _require(type(envelope["response_bytes"]) is int and envelope["response_bytes"] == len(body) and envelope["response_sha256"] == _digest(body) and body, "response envelope does not bind exact bytes")
    _require(all(isinstance(envelope[key], str) and envelope[key] for key in ("requested_url", "effective_url", "format")), "missing response route or format")
    _require(isinstance(envelope["redirect_chain"], list), "original response redirects are unavailable")
    captured, cutoff = _time(envelope["received_at"]), _time(available_before)
    _require(captured < cutoff, "provider observation was not captured before its availability cutoff")
    return envelope, captured, cutoff


def _parse_response(source: str, body: bytes, envelope: dict, *, division: str, fighter_name: str) -> tuple[dict, list[dict], str]:
    from bs4 import BeautifulSoup
    from src.data import fallback_scrapers as parsers

    requested, effective = envelope["requested_url"], envelope["effective_url"]
    fmt = envelope["format"]
    if fmt == "tapology_reader":
        _require(source == "tapology", "reader input is only supported for Tapology")
        markdown = body.decode("utf-8")
        canonical = parsers._tapology_reader_markdown_source_url(markdown)
        declarations = set(re.findall(r"^\s*URL Source:\s*(\S+)\s*$", markdown, flags=re.IGNORECASE | re.MULTILINE))
        _require(len(declarations) == 1 and bool(_provider_number(source, canonical)), "reader response has no unambiguous supported source identity")
        profile = parsers._parse_tapology_reader_profile(canonical, markdown)
        fights = parsers._parse_tapology_reader_fights(markdown, division=division)
        for fight in fights:
            # Reader title text is selected by proximity, without a retained
            # bout container. It cannot establish this particular bout's title.
            fight["is_title_bout"] = None
        reader_links = [link for line in parsers._tapology_reader_lines(markdown) for link in parsers._tapology_reader_links(line)]
        for fight in fights:
            opponents = {_provider_number(source, urljoin(canonical, href)) for name, href, title in reader_links
                         if "Fighter Page" in title and normalize_person_name(name) == normalize_person_name(fight.get("opponent"))}
            _require(len(opponents) == 1 and "" not in opponents and _provider_number(source, canonical) not in opponents,
                     "Tapology reader bout has ambiguous fighter orientation")
        profile.update(transport_kind="jina_reader", reader_target_url=canonical, origin_redirect_chain_captured=False)
        discovery_url = canonical
    else:
        _require(fmt == "html", "unsupported retained provider format")
        soup = BeautifulSoup(body, "html.parser")
        declared = {
            urljoin(effective, str(item.get("href") or item.get("content") or "").strip()).rstrip("/")
            for item in soup.select('link[rel~="canonical"][href], meta[property="og:url"][content]')
            if str(item.get("href") or item.get("content") or "").strip()
        }
        _require(len(declared) == 1, "response canonical identity is missing or conflicting")
        canonical = next(iter(declared))
        discovery_url = requested
        _require(all(_provider_number(source, url) for url in (canonical, requested, effective, *envelope["redirect_chain"])), "unsupported provider origin or fighter route")
        if source == "sherdog":
            profile = parsers._parse_sherdog_profile(soup, canonical)
            tables = soup.find_all("table", class_="fighter")
            sections = [(table, parsers._nearest_sherdog_section_text(table)) for table in tables]
            if division == "am":
                selected = [table for table, section in sections if "amateur" in section]
            else:
                selected = [table for table, section in sections if "professional" in section and "amateur" not in section]
                if not selected and sections:
                    table, section = sections[0]
                    _require(not any(word in section for word in ("amateur", "exhibition")), "Sherdog professional history has another division")
                    selected = [table]
            _require(len(selected) <= 1, "ambiguous Sherdog history division")
            table = selected[0] if selected else None
            if table is not None:
                for row in table.find_all("tr")[1:]:
                    cells = row.find_all("td")
                    if len(cells) < 6:
                        continue
                    for link in cells[1].find_all("a", href=True):
                        opponent_number = _provider_number(source, urljoin(canonical, link["href"]))
                        _require(not opponent_number or opponent_number != _provider_number(source, canonical),
                                 "Sherdog bout links to its own fighter identity")
            fights = parsers._parse_sherdog_fight_table(table, fighter_name) if table is not None else []
            # The legacy parser guesses title status from an event/promotion
            # name (including "ONE Championship"). That does not identify a
            # title bout; retained career evidence keeps this field unknown.
            for fight in fights:
                fight["is_title_bout"] = None
        else:
            profile = parsers._parse_tapology_profile_soup(canonical, soup)
            for block in soup.select("[data-bout-id]"):
                if block.get("data-sport") != "mma" or str(block.get("data-division") or "").lower() not in ({"pro", "professional"} if division == "pro" else {"am", "amateur"}):
                    continue
                if str(block.get("data-status") or "").lower() in {"cancelled", "booking", "scheduled", "upcoming"}:
                    continue
                links = block.find_all("a", href=lambda href: href and "/fightcenter/fighters/" in href,
                                       title=lambda title: title and "Fighter Page" in title)
                entities = {_provider_number(source, urljoin(canonical, link["href"])) for link in links}
                _require(len(entities) == 1 and "" not in entities and _provider_number(source, canonical) not in entities,
                         "Tapology bout has ambiguous fighter orientation")
            fights = parsers._parse_tapology_fights_soup(canonical, fighter_name, soup, division=division)
            for fight in fights:
                # A belt icon inside the actual bout is positive evidence.
                # The parser's zero merely means the icon was absent; it is not
                # an independently observed assertion of a non-title bout.
                fight["is_title_bout"] = 1 if fight.get("is_title_bout") == 1 else None
    profile.update(requested_url=requested, effective_url=effective, canonical_url=canonical,
                   redirect_chain=envelope["redirect_chain"], source_entity_id=source_entity_id(source, canonical))
    # These sources do not provide an authorized stance. A missing record is
    # likewise not an observed 0-0-0 (including the existing reader parser).
    profile["stance"] = ""
    if source == "tapology" and not str(profile.get("record") or "").strip():
        profile.update(wins=float("nan"), losses=float("nan"), draws=float("nan"))
    owner_name = normalize_person_name(profile.get("name"))
    for fight in fights:
        opponent_name = normalize_person_name(fight.get("opponent"))
        _require(opponent_name and opponent_name != owner_name, "career row has ambiguous fighter orientation")
    return profile, fights, discovery_url


def load_retained_provider_observation(
    descriptor: Mapping | Path | str,
    *,
    fighter_id: str,
    fighter_name: str,
    available_before: object,
    verified_bindings: set[tuple[str, str, str]],
    verified_binding_metadata: dict | None = None,
    verified_binding_name_keys: dict | None = None,
    division: str = "pro",
    history_root: Path | None = None,
    consumer: dict | None = None,
) -> dict:
    """Validate original bytes and return their identity-bound parsed contents.

No descriptor can establish its own trusted owner or authorize profile fields.
The profile importer must still enforce its existing per-field permissions.
Without history_root this is a read-only preview. Attachment callers require a
history root and identify their consumer to retain a durable evidence reference.
"""
    try:
        return _load(descriptor, fighter_id=fighter_id, fighter_name=fighter_name,
                     available_before=available_before, verified_bindings=verified_bindings,
                     verified_binding_metadata=verified_binding_metadata,
                     verified_binding_name_keys=verified_binding_name_keys, division=division,
                     history_root=history_root, consumer=consumer)
    except ProviderObservationError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ObservationRefusal) as exc:
        raise ProviderObservationError(f"retained provider input refused: {exc}") from exc


def _load(descriptor: Mapping | Path | str, *, fighter_id: str, fighter_name: str,
          available_before: object, verified_bindings: set, verified_binding_metadata: dict | None,
          verified_binding_name_keys: dict | None, division: str, history_root: Path | None,
          consumer: dict | None) -> dict:
    from scripts.build_profile_supplement_from_external_profiles import (
        ProfileIdentityUnverifiedError,
        _validated_response_identity,
    )

    _require(isinstance(fighter_id, str) and re.fullmatch(r"[0-9a-f]{16}", fighter_id), "missing valid stable fighter identity")
    _require(isinstance(fighter_name, str) and normalize_person_name(fighter_name), "missing target display identity")
    _require(division in {"pro", "am"}, "unsupported career division")
    if consumer is not None:
        _require(history_root is not None, "a consumer reference requires retained provider history")
        _require(isinstance(consumer, dict) and set(consumer) == {"kind", "id"} and consumer["kind"] in {"profile", "pre_ufc"} and isinstance(consumer["id"], str) and consumer["id"].strip(), "invalid provider observation consumer")
    _require(isinstance(verified_bindings, set) and verified_bindings, "independent verified provider bindings required")
    for binding in verified_bindings:
        _require(isinstance(binding, tuple) and len(binding) == 3 and all(isinstance(v, str) and v for v in binding), "malformed independent binding")
    document, input_path = _input_document(descriptor)
    source = document["source"]
    _require(source in {"sherdog", "tapology"}, "unsupported retained provider source")
    body, envelope_body = _attached(document["response"]), _attached(document["envelope"])
    envelope, captured, cutoff = _validated_envelope(body, envelope_body, available_before)
    profile, fights, requested_url = _parse_response(source, body, envelope, division=division, fighter_name=fighter_name)
    if str(profile.get("dob") or "").strip():
        # Existing convenience parsers calculate age against today's clock.
        # A retained profile must reproduce the age at its original capture.
        import pandas as pd
        dob = pd.to_datetime(profile["dob"], errors="coerce", utc=True)
        profile["age"] = round((captured - dob.to_pydatetime()).days / 365.25, 1) if pd.notna(dob) and dob <= captured else float("nan")
    try:
        identity = _validated_response_identity(source=source, stable_id=fighter_id, requested_url=requested_url,
                                                response=profile, verified_bindings=verified_bindings,
                                                verified_binding_metadata=verified_binding_metadata)
    except ProfileIdentityUnverifiedError as exc:
        raise ProviderObservationError(str(exc)) from exc
    provider_id = identity["source_entity_id"]
    binding = (fighter_id, source, provider_id)
    _require({item[0] for item in verified_bindings if item[1:] == binding[1:]} == {fighter_id}, "provider identity has conflicting stable owners")
    actual_name = normalize_person_name(profile.get("name"))
    approved_names = (verified_binding_name_keys or {}).get(binding, set())
    _require(isinstance(approved_names, (set, frozenset)) and all(isinstance(name, str) and name for name in approved_names), "invalid reviewed binding names")
    _require(actual_name and (actual_name == normalize_person_name(fighter_name) or actual_name in approved_names), "response display identity is blank or unreviewed")
    identity_seed = {"source": source, "fighter_id": fighter_id, "provider_id": provider_id,
                     "response_sha256": _digest(body), "envelope_sha256": _digest(envelope_body),
                     "parser_revision": PARSER_REVISION}
    observation_id = _digest(_json_bytes(identity_seed))
    was_finalized = "observation_id" in document
    if was_finalized:
        _require(document["observation_id"] == observation_id and document["fighter_id"] == fighter_id and document["provider_id"] == provider_id, "finalized evidence owner or identity changed")
    document.update(fighter_id=fighter_id, provider_id=provider_id, parser_revision=PARSER_REVISION, observation_id=observation_id)
    evidence_body = _json_bytes(document)
    evidence_path = input_path if was_finalized else ""
    if history_root is not None:
        root = _safe_path(Path(history_root))
        destination = root / HISTORY_SUBDIR / f"{observation_id}.json"
        _write_immutable(destination, evidence_body)
        evidence_path = str(destination)
    if consumer is not None:
        reference = {"schema": 1, "consumer": consumer, "available_before": _text_time(cutoff),
                     "division": division, "eligibility_policy": ELIGIBILITY_POLICY,
                     "observation_id": observation_id, "evidence_path": evidence_path,
                     "evidence_sha256": _digest(evidence_body)}
        reference_body = _json_bytes(reference)
        _write_immutable(Path(history_root) / "provider_observation_references_v1" / f"{_digest(reference_body)}.json", reference_body)
    row_metadata = {"source_observation_id": observation_id, "source_response_sha256": _digest(body),
                    "source_envelope_sha256": _digest(envelope_body), "source_parser_revision": PARSER_REVISION,
                    "source_eligibility_policy": ELIGIBILITY_POLICY, "source_captured_at_utc": _text_time(captured),
                    "source_evidence_path": evidence_path, "source_evidence_sha256": _digest(evidence_body),
                    "source_provider_id": provider_id, "source_fighter_id": fighter_id, "source_division": division}
    profile.update(identity)
    profile.update(row_metadata)
    return {"profile": profile, "fights": fights, "source": source, "provider_id": provider_id,
            "fighter_id": fighter_id, "fighter_name": fighter_name, "captured_at_utc": _text_time(captured),
            "observation_id": observation_id, "parser_revision": PARSER_REVISION,
            "eligibility_policy": ELIGIBILITY_POLICY, "evidence_path": evidence_path,
            "evidence_sha256": _digest(evidence_body), "row_metadata": row_metadata}


def _attachment_values(row: Mapping) -> dict:
    """Canonicalize the saved CSV schema, including explicit missing values."""
    import pandas as pd
    from src.data.pre_ufc_scraper import OUTPUT_COLUMNS, STAT_COLUMNS

    numeric = {"finish_round", "num_rounds", "title_bout"} | {
        f"{side}_{stat}" for side in ("a", "b") for stat in STAT_COLUMNS
    }
    result = {}
    for field in OUTPUT_COLUMNS:
        if field in {"source_attachment_path", "source_attachment_sha256"}:
            continue
        value = row.get(field)
        missing = value is None or bool(pd.isna(value)) or (isinstance(value, str) and value.strip().casefold() in {"", "nan", "none", "<na>"})
        if missing:
            result[field] = None
        elif field in numeric:
            converted = pd.to_numeric(value, errors="coerce")
            _require(pd.notna(converted), f"invalid numeric career attachment field: {field}")
            result[field] = float(converted)
        elif field == "event_date":
            converted = pd.to_datetime(value, errors="coerce", utc=True)
            _require(pd.notna(converted), "invalid career attachment date")
            result[field] = converted.isoformat()
        else:
            result[field] = str(value)
    return result


def finalize_retained_history_rows(rows: list[dict], *, history_root: Path) -> list[dict]:
    """Retain the exact attachment context after independent source validation.

Each context binds the selected subject, orientation, values, missingness and
the original debut/availability cutoffs to the exact provider evidence. It is
additive; no reference inventory or deletion authority is asserted here.
"""
    result = []
    try:
        for row in rows:
            _verify_history_row(row, check_attachment=False)
            document = {"schema": 1, "kind": "provider_history_attachment_v1",
                        "eligibility_policy": ELIGIBILITY_POLICY, "row": _attachment_values(row)}
            body = _json_bytes(document)
            digest = _digest(body)
            destination = _safe_path(Path(history_root)) / "provider_history_attachments_v1" / f"{digest}.json"
            _write_immutable(destination, body)
            result.append({**row, "source_attachment_path": str(destination), "source_attachment_sha256": digest})
    except ProviderObservationError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ObservationRefusal) as exc:
        raise ProviderObservationError(f"retained history attachment refused: {exc}") from exc
    return result


def verify_retained_history_row(row: Mapping, *, references: dict | None = None) -> None:
    """Check an accepted career carrier against its exact retained evidence.

This verifies continuity of earlier accepted owner evidence, not a new identity
decision. The accepted CSV and its evidence pins are the caller's trusted input;
the function never constructs a new reviewed binding from a provider document.
"""
    try:
        _verify_history_row(row, references=references)
    except ProviderObservationError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ObservationRefusal) as exc:
        raise ProviderObservationError(f"retained history row refused: {exc}") from exc


def _verify_history_row(row: Mapping, *, check_attachment: bool = True, references: dict | None = None) -> None:
    import pandas as pd
    from src.data.pre_ufc_scraper import BASE_COLUMNS, STAT_COLUMNS, _rows_from_fights
    from src.data.provider_consumer_references import read_provider_reference

    if check_attachment:
        attachment_path = row.get("source_attachment_path")
        _require(isinstance(attachment_path, str) and Path(attachment_path).is_absolute(), "career row lacks original attachment context")
        attachment_body = read_provider_reference(attachment_path, references=references)
        _require(row.get("source_attachment_sha256") == _digest(attachment_body) == Path(attachment_path).stem,
                 "career attachment hash mismatch")
        attachment = json.loads(attachment_body)
        _require(attachment == {"schema": 1, "kind": "provider_history_attachment_v1",
                                "eligibility_policy": ELIGIBILITY_POLICY, "row": _attachment_values(row)},
                 "career row differs from original accepted attachment context")
    evidence_path = row.get("source_evidence_path")
    _require(isinstance(evidence_path, str) and Path(evidence_path).is_absolute(), "career row lacks exact retained evidence path")
    evidence_body = read_provider_reference(evidence_path, references=references)
    _require(row.get("source_evidence_sha256") == _digest(evidence_body), "career evidence hash mismatch")
    document, _ = _input_document(json.loads(evidence_body))
    _require(Path(evidence_path).stem == document.get("observation_id"), "retained provider filename mismatch")
    _require("observation_id" in document, "career row requires finalized provider evidence")
    source, fighter_id, provider_id = document["source"], document["fighter_id"], document["provider_id"]
    _require(source in {"sherdog", "tapology"} and re.fullmatch(r"[0-9a-f]{16}", fighter_id), "invalid retained owner")
    body, envelope_body = _attached(document["response"]), _attached(document["envelope"])
    envelope, captured, _ = _validated_envelope(body, envelope_body, row.get("available_before"))
    identity_seed = {"source": source, "fighter_id": fighter_id, "provider_id": provider_id,
                     "response_sha256": _digest(body), "envelope_sha256": _digest(envelope_body),
                     "parser_revision": PARSER_REVISION}
    observation_id = _digest(_json_bytes(identity_seed))
    _require(document["observation_id"] == observation_id, "retained career observation identity mismatch")
    expected = {"source_observation_id": observation_id, "source_response_sha256": _digest(body),
                "source_envelope_sha256": _digest(envelope_body), "source_parser_revision": PARSER_REVISION,
                "source_eligibility_policy": ELIGIBILITY_POLICY, "source_captured_at_utc": _text_time(captured),
                "source_provider_id": provider_id, "source_fighter_id": fighter_id,
                "fighter_a_id": fighter_id, "source": source,
                "observation_id": observation_id, "captured_at_utc": _text_time(captured),
                "provider_id": provider_id, "parser_revision": PARSER_REVISION,
                "eligibility_policy": ELIGIBILITY_POLICY, "evidence_path": evidence_path,
                "evidence_sha256": _digest(evidence_body)}
    for field, value in expected.items():
        _require(row.get(field) == value, f"career retained metadata differs: {field}")
    division = row.get("history_division")
    _require(division in {"pro", "am"} and row.get("source_division") == division, "career division differs from retained reference")
    profile, fights, _ = _parse_response(source, body, envelope, division=division, fighter_name=row.get("fighter_a"))
    _require(profile.get("source_entity_id") == provider_id and normalize_person_name(profile.get("name")), "retained career response identity differs")
    observation = {"captured_at_utc": _text_time(captured)}
    candidates = _rows_from_fights(fights, row.get("fighter_a"), row.get("first_ufc_date"), source,
                                   fighter_id=fighter_id, observation=observation,
                                   available_before=row.get("available_before"), division=division)
    matches = [candidate for candidate in candidates if candidate["history_row_id"] == row.get("history_row_id")]
    _require(matches, "career row was not reproduced by original provider evidence")

    def missing(value):
        return value is None or bool(pd.isna(value)) or (isinstance(value, str) and value.strip().casefold() in {"", "nan", "none", "<na>"})

    def same(field, actual, value):
        if missing(value):
            return missing(actual)
        if missing(actual):
            return False
        if field == "event_date":
            return pd.to_datetime(actual, errors="coerce", utc=True) == pd.to_datetime(value, utc=True)
        if field in {"finish_round", "num_rounds", "title_bout"}:
            return pd.to_numeric(actual, errors="coerce") == value
        return str(actual) == str(value)

    fields = list(BASE_COLUMNS) + [f"{side}_{stat}" for side in ("a", "b") for stat in STAT_COLUMNS] + ["fighter_b_id"]
    _require(any(all(same(field, row.get(field), candidate.get(field)) for field in fields) for candidate in matches),
             "career values or fighter orientation differ from original provider response")
