"""Derive source-labelled ranking observations from one preserved roster page.

No acquisition, claim, history, model or membership writes occur here. The caller
must supply an independently reviewed registry pin. Its embedded membership
ledger must substantiate every requested exact name/date-of-birth/UFСStats link.
Blank, interim-champion and ambiguous division values remain unknown.
"""
from __future__ import annotations

import csv
from datetime import date
import io
import json
import re

from bs4 import BeautifulSoup

from src.data.method_observation_history import (
    _attached, _digest, _json_bytes, _read_pin, _require, _time,
)
from src.data.rankings_scopes import WC_ALIASES
from src.data.rosterwatch_site import parse_rosterwatch_site_roster

SOURCE = "roster.watch"
SOURCE_URL = "https://www.roster.watch/"
PARSER = "rosterwatch_exact_current_rank_cells_v1"
POLICY = "rosterwatch_original_receipt_strict_prefight_v1"
PRECEDENCE = "latest_receipt_rosterwatch_tiebreak_v1"
REGISTRY_KIND = "reviewed_rosterwatch_rankings_identity_v1"


def _date(value):
    _require(isinstance(value, str), "Roster Watch identity date must be explicit")
    try:
        valid = date.fromisoformat(value).isoformat() == value
    except ValueError:
        valid = False
    _require(valid, "Invalid Roster Watch identity date")


def _identity_ledger(attachment):
    """Parse the exact attached independently reviewed membership associations."""
    body = _attached(attachment)
    try:
        reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
        required = {"rw_fighter_raw", "rw_dob_raw", "preferred_carrier_type",
                    "preferred_carrier_value", "carrier_resolution_status",
                    "identity_conflict_flags"}
        _require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames)
                 and required <= set(reader.fieldnames), "Incomplete reviewed membership identity evidence")
        rows = list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("Invalid reviewed membership identity evidence") from exc
    _require(rows and all(None not in row and all(row[key] is not None for key in required)
                          for row in rows), "Malformed reviewed membership identity rows")
    by_key, by_carrier = {}, {}
    for row in rows:
        key = (row["rw_fighter_raw"], row["rw_dob_raw"])
        _require(key not in by_key, "Duplicate reviewed Roster Watch identity key")
        by_key[key] = row
        if row["carrier_resolution_status"] == "resolved" and row["preferred_carrier_type"] == "ufcstats":
            carrier = row["preferred_carrier_value"]
            _require(re.fullmatch(r"[0-9a-f]{16}", carrier) and carrier not in by_carrier,
                     "Duplicate or invalid reviewed Roster Watch carrier")
            by_carrier[carrier] = key
    return by_key


def validate_registry(document):
    """Validate supplied authority; parsing a source page never creates it."""
    _require(isinstance(document, dict) and set(document) == {
        "schema", "kind", "source", "rankings_url", "parser_revision", "availability_policy",
        "source_precedence", "collection_id", "identity_evidence", "fighters"},
        "Incomplete independently reviewed Roster Watch registry")
    _require(type(document["schema"]) is int and document["schema"] == 1
             and document["kind"] == REGISTRY_KIND and document["source"] == SOURCE
             and document["rankings_url"] == SOURCE_URL and document["parser_revision"] == PARSER
             and document["availability_policy"] == POLICY and document["source_precedence"] == PRECEDENCE,
             "Unsupported Roster Watch source/parser/policy authority")
    _require(isinstance(document["collection_id"], str) and bool(document["collection_id"].strip()),
             "Missing independently assigned collection identity")
    ledger = _identity_ledger(document["identity_evidence"])
    _require(isinstance(document["fighters"], list) and document["fighters"], "Missing reviewed fighter bindings")
    ids, keys = set(), set()
    for fighter in document["fighters"]:
        _require(isinstance(fighter, dict) and set(fighter) == {
            "fighter_id", "rosterwatch_name", "date_of_birth", "scopes"}, "Incomplete reviewed fighter binding")
        identity = fighter["fighter_id"]
        name = fighter["rosterwatch_name"]
        _require(isinstance(identity, str) and re.fullmatch(r"[0-9a-f]{16}", identity)
                 and identity not in ids, "Invalid or duplicate stable fighter identity")
        _require(isinstance(name, str) and name.strip() == name and bool(name), "Missing exact Roster Watch name")
        _date(fighter["date_of_birth"])
        key = (name, fighter["date_of_birth"])
        _require(key not in keys, "Duplicate reviewed Roster Watch name/date key")
        associated = ledger.get(key)
        _require(associated is not None and associated["carrier_resolution_status"] == "resolved"
                 and associated["preferred_carrier_type"] == "ufcstats"
                 and associated["preferred_carrier_value"] == identity
                 and associated["identity_conflict_flags"] == "",
                 "Reviewed membership evidence does not establish this exact unconflicted carrier")
        ids.add(identity)
        keys.add(key)
        scopes = fighter["scopes"]
        _require(isinstance(scopes, list) and scopes, "Missing independently reviewed ranking scopes")
        seen = set()
        for scope in scopes:
            _require(isinstance(scope, dict) and set(scope) == {"ranking_type", "division"}, "Invalid ranking scope")
            kind, division = scope["ranking_type"], scope["division"]
            _require(isinstance(kind, str) and isinstance(division, str)
                     and ((kind == "division" and division in set(WC_ALIASES.values()))
                          or (kind == "pfp" and division in {"men", "women"}))
                     and (kind, division) not in seen, "Unknown or duplicate reviewed ranking scope")
            seen.add((kind, division))
    return document


def load_reviewed_registry(pin):
    body = _read_pin(pin)
    document = validate_registry(json.loads(body))
    _require(body == _json_bytes(document), "Roster Watch registry must retain exact canonical bytes")
    return document


def _validate_capture(body, metadata):
    _require(isinstance(body, bytes) and body, "Missing original Roster Watch response bytes")
    _require(isinstance(metadata, dict) and set(metadata) == {
        "url", "requested_at_utc", "retrieved_at_utc", "status_code", "headers", "byte_count",
        "sha256", "user_agent", "redirects_followed", "retries"}, "Incomplete original Roster Watch receipt")
    _require(metadata["url"] == SOURCE_URL and type(metadata["status_code"]) is int
             and metadata["status_code"] == 200
             and type(metadata["byte_count"]) is int and metadata["byte_count"] == len(body)
             and metadata["sha256"] == _digest(body)
             and type(metadata["redirects_followed"]) is int and metadata["redirects_followed"] == 0
             and type(metadata["retries"]) is int and metadata["retries"] == 0,
             "Response source, success, bytes or single-request receipt differs")
    _require(isinstance(metadata["user_agent"], str) and bool(metadata["user_agent"].strip())
             and isinstance(metadata["headers"], dict)
             and all(isinstance(key, str) and isinstance(value, str) for key, value in metadata["headers"].items()),
             "Missing original request or response header metadata")
    headers = {key.lower(): value for key, value in metadata["headers"].items()}
    _require(len(headers) == len(metadata["headers"])
             and headers.get("content-type", "").split(";", 1)[0].strip().lower() == "text/html",
             "Unsupported or ambiguous Roster Watch response headers")
    _require(headers.get("cf-mitigated", "").lower() != "challenge", "Roster Watch challenge is not source data")
    requested, received = _time(metadata["requested_at_utc"]), _time(metadata["retrieved_at_utc"])
    _require(requested <= received, "Roster Watch receipt predates request")
    return received


def _current_cell(row, index, raw, kind):
    cells = row.select("td.td-rank")
    _require(len(cells) == 2, "Current Roster Watch rank cells are missing or ambiguous")
    labels = cells[index].select(".rank-value")
    _require(len(labels) == 1, "Explicit current ranking label missing or duplicated")
    label = labels[0].get_text(strip=True)
    if raw in {"", "NA"}:
        _require(label == "", "Missing rank attribute conflicts with current label")
        return None, "source_value_missing", label
    if raw == "16":
        _require(label == "", "Blank rank marker conflicts with current label")
        return None, "blank_marker_not_proven_unranked", label
    if kind == "division" and raw in {"-1", "0"}:
        _require(label == ("C" if raw == "-1" else "IC"), "Champion attribute conflicts with current label")
        return (0, "explicit_champion", label) if raw == "-1" else (None, "interim_champion_unmapped", label)
    _require(re.fullmatch(r"[1-9]|1[0-5]", raw) is not None and label == raw,
             "Unsupported or conflicting explicit current ranking")
    return int(raw), "explicit_rank", label


def observations_from_capture(body: bytes, metadata: dict, registry: dict):
    """Pure derivation; receipt/registry hashes use exact canonical saved JSON.

    The caller writes the supplied metadata and registry with ``_json_bytes``.
    Capture receipt time is availability; HTTP Last-Modified is not promoted to
    a ranking-effective timestamp. Unknown values deliberately shadow absence.
    """
    registry = validate_registry(registry)
    received = _validate_capture(body, metadata)
    records, parse_report = parse_rosterwatch_site_roster(body)
    _require(parse_report.get("complete") is True, "Incomplete original Roster Watch page")
    soup = BeautifulSoup(body, "lxml")
    elements = soup.select_one("table#roster-tbl").find("tbody").find_all("tr", recursive=False)
    _require(len(elements) == len(records), "Roster Watch rank rows differ from membership rows")
    by_key = {(record["fighter"], record["date_of_birth"]): (record, element)
              for record, element in zip(records, elements)}
    registry_sha = _digest(_json_bytes(registry))
    metadata_sha = _digest(_json_bytes(metadata))
    observations = []
    for fighter in registry["fighters"]:
        found = by_key.get((fighter["rosterwatch_name"], fighter["date_of_birth"]))
        for scope in fighter["scopes"]:
            value, raw, label, reason = None, None, None, "reviewed_fighter_absent"
            if found is not None:
                record, element = found
                kind = scope["ranking_type"]
                raw = record["current_rank" if kind == "division" else "current_p4p"]
                value, reason, label = _current_cell(element, 0 if kind == "division" else 1, raw, kind)
                if kind == "division":
                    divisions = json.loads(element.get("data-weightclass") or "[]")
                    if len(divisions) != 1:
                        value, reason = None, "division_association_ambiguous"
                    elif divisions[0] != scope["division"]:
                        value, reason = None, "division_outside_reviewed_scope"
                elif {"male": "men", "female": "women"}.get(record["gender"]) != scope["division"]:
                    value, reason = None, "pfp_group_outside_reviewed_scope"
            observations.append({
                "fighter_id": fighter["fighter_id"], "athlete_url": None, **scope,
                "value": value, "status": "ranked" if value is not None else "unknown",
                "coverage": "explicit_bound_source_row_only", "observed_at": metadata["retrieved_at_utc"],
                "available_at": received.isoformat(), "source_effective_date": None,
                "source": SOURCE, "parser_revision": PARSER, "availability_policy": POLICY,
                "source_precedence": PRECEDENCE, "registry_sha256": registry_sha,
                "response_sha256": _digest(body), "response_envelope_sha256": metadata_sha,
                "source_url": SOURCE_URL, "source_fighter_name": fighter["rosterwatch_name"],
                "source_fighter_dob": fighter["date_of_birth"], "source_rank_raw": raw,
                "source_rank_label": label, "value_reason": reason,
            })
    return observations
