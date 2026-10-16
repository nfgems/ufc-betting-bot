"""Retain reviewed rankings with independently bound identity and receipt time.

The reviewed registry is supplied with an outer file pin by the caller. A source
name, athlete slug, parsed rank or saved reference cannot approve that registry.
Legacy name/date CSVs remain a separate, unsupported source for this reader.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from src.data.method_observation_history import (
    ObservationRefusal, _attached, _attachment, _digest, _json_bytes, _read_pin,
    _require, _safe_path, _time, _write_immutable,
)

HISTORY_SUBDIR = "retained_rankings_observations_v1"
CAPTURE_SUBDIR = "rankings_response_captures_v1"
PARSER = "ufc_rankings_explicit_athlete_rank_v1"
POLICY = "rankings_original_receipt_strict_prefight_v1"
REGISTRY_KIND = "reviewed_ufc_rankings_identity_v1"
SOURCE_URL = "https://www.ufc.com/rankings"
PRECEDENCE = "official_ufc_only_v1"


def _athlete_url(value):
    _require(isinstance(value, str), "missing athlete-link identity")
    parsed = urlparse(urljoin(SOURCE_URL, value))
    _require(parsed.scheme == "https" and parsed.hostname in {"ufc.com", "www.ufc.com"}
             and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment
             and re.fullmatch(r"/athlete/[a-z0-9]+(?:-[a-z0-9]+)*/?", parsed.path),
             "unsupported official athlete identity")
    return "https://www.ufc.com" + parsed.path.rstrip("/")


def _scope(heading):
    from src.data.rankings_scopes import WC_ALIASES
    value = " ".join(str(heading).lower().replace("’", "'").split())
    for group, prefix in (("men", "men's "), ("women", "women's ")):
        labels = {prefix + form + suffix for form in ("pound-for-pound", "pound for pound") for suffix in ("", " top rank")}
        if value in labels:
            return {"ranking_type": "pfp", "division": group}
    if value in {"pound-for-pound", "pound for pound"}:
        return {"ranking_type": "pfp", "division": "men"}
    if value in WC_ALIASES:
        return {"ranking_type": "division", "division": WC_ALIASES[value]}
    return None


def _scope_key(value):
    return value["ranking_type"], value["division"]


def _observed_rank(text):
    value = " ".join(str(text).lower().split())
    if value in {"c", "champion", "ufc champion"}:
        return 0
    match = re.fullmatch(r"(?:t(?:ie)?\s*[-=]?\s*)?(\d{1,2})(?:\s*(?:\(tie\)|=))?", value)
    if match and 1 <= int(match[1]) <= 15:
        return int(match[1])
    return None


def parse_official_rankings_html(html):
    """Keep explicit ranks, athlete links, unresolved rows and table coverage.

Only a complete observed 1..15 rank set supports absence. Ties are retained
exactly; a skipped rank makes coverage conservatively incomplete. Champion is
observed as zero only when the source labels that athlete as champion.
"""
    soup = BeautifulSoup(html, "lxml")
    sections = soup.select("div.view-grouping") or soup.select("div.rankings--athlete--container")
    # UFC publishes a separately labelled "All Meta Rankings" view alongside
    # its ordinary rankings. Its meta-weight-class-rank is a different measure,
    # not a duplicate ordinary table and not supported by this parser policy.
    # Unknown/duplicate ordinary scopes still refuse below.
    ordinary_keys, meta_keys = set(), set()
    for section in sections:
        heading = section.select_one("h4, h3, .view-grouping-header")
        scope = _scope(heading.get_text(" ", strip=True)) if heading else None
        if scope is not None:
            keys = (meta_keys if section.find_parent(class_="view-display-id-meta_rankings")
                    is not None else ordinary_keys)
            keys.add(_scope_key(scope))
    # Only enable the modern interpretation on a layout the legacy parser
    # necessarily refused (overlapping ordinary/meta scopes). Successful saved
    # v1 receipts keep exactly their original champion/P4P interpretation.
    modern_layout = bool(soup.select_one(".view-display-id-block_1")
                         and ordinary_keys.intersection(meta_keys))
    if modern_layout:
        sections = [section for section in sections
                    if section.find_parent(class_="view-display-id-meta_rankings") is None]
    tables, seen = [], set()
    for section in sections:
        header = section.select_one("h4, h3, .view-grouping-header")
        scope = _scope(header.get_text(" ", strip=True)) if header else None
        if scope is None:
            continue
        key = _scope_key(scope)
        _require(key not in seen, "duplicate official ranking scope")
        seen.add(key)
        rows, seen_links = [], set()
        for item in section.select("tr, li.rankingItemsItem, .rankings--athlete--champion"):
            link = item.select_one("a[href*='/athlete/']")
            # A header-only row carries no athlete observation. Other body rows
            # remain unresolved instead of silently shortening the table.
            if link is None and not item.select("td") and "champion" not in " ".join(item.get("class", [])):
                continue
            rank_node = item.select_one(".views-field-weight-class-rank, .views-field-rank, .rank, [data-rank]")
            rank_text = rank_node.get("data-rank", rank_node.get_text(" ", strip=True)) if rank_node else ""
            if not rank_text:
                champion = item.select_one(".rankings--athlete--champion-label, .champion-label")
                if (modern_layout and champion is None
                        and "rankings--athlete--champion" in item.get("class", [])):
                    champion = item.select_one(".info h6 > .text")
                rank_text = champion.get_text(" ", strip=True) if champion else ""
            rank = _observed_rank(rank_text)
            athlete = None
            if link is not None:
                try:
                    athlete = _athlete_url(link.get("href"))
                except ObservationRefusal:
                    pass
            # A P4P hero repeats the explicit #1 table athlete without a rank
            # label. Ignore only that proved presentational repeat; a different
            # URL, missing #1, or any unsupported explicit label stays unknown.
            if (modern_layout and scope["ranking_type"] == "pfp" and not rank_text and athlete
                    and "rankings--athlete--champion" in item.get("class", [])):
                explicit_first = []
                for table_row in section.select("tr"):
                    table_rank = table_row.select_one(".views-field-weight-class-rank, .views-field-rank, .rank, [data-rank]")
                    table_link = table_row.select_one("a[href*='/athlete/']")
                    if table_rank is not None and table_link is not None:
                        label = table_rank.get("data-rank", table_rank.get_text(" ", strip=True))
                        if _observed_rank(label) == 1:
                            try:
                                explicit_first.append(_athlete_url(table_link.get("href")))
                            except ObservationRefusal:
                                pass
                if explicit_first == [athlete]:
                    continue
            duplicate = athlete in seen_links if athlete else False
            if athlete:
                seen_links.add(athlete)
            rows.append({"athlete_url": athlete,
                         "athlete_name": link.get_text(" ", strip=True) if link else None,
                         "observed_rank": rank, "rank_label": rank_text,
                         "resolved": bool(athlete and rank is not None and not duplicate)})
        # Official champion markup can sit outside the table in its own block.
        # It is never inferred from the first table row or an athlete's name.
        valid = bool(rows) and all(row["resolved"] for row in rows)
        ranks = {row["observed_rank"] for row in rows if row["observed_rank"] is not None}
        champion_complete = scope["ranking_type"] == "pfp" or sum(row["observed_rank"] == 0 for row in rows) == 1
        complete = valid and champion_complete and set(range(1, 16)).issubset(ranks)
        tables.append({**scope, "coverage": "complete_top15" if complete else "incomplete",
                       "rows": rows})
    _require(tables and any(table["rows"] for table in tables), "no official ranking observations parsed")
    effective_values = set()
    for element in soup.select("time[data-ranking-effective], .rankings--date time[datetime], meta[name='rankings-effective-date']"):
        value = element.get("datetime") or element.get("content")
        if value:
            effective_values.add(value.strip())
    _require(len(effective_values) <= 1, "conflicting source-effective dates")
    effective = next(iter(effective_values), None)
    if effective is not None:
        _effective_time(effective)
    return {"source": "ufc.com", "parser_revision": PARSER,
            "source_effective_date": effective, "tables": tables}


def _effective_time(value):
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)):
        return _time(value + "T00:00:00Z")
    return _time(value)


def _validate_registry(document):
    _require(isinstance(document, dict) and set(document) == {
        "schema", "kind", "source", "rankings_url", "parser_revision", "availability_policy",
        "source_precedence", "collection_id", "fighters"}, "incomplete independently reviewed rankings registry")
    _require(type(document["schema"]) is int and document["schema"] == 1
             and document["kind"] == REGISTRY_KIND and document["source"] == "ufc.com"
             and document["rankings_url"] == SOURCE_URL and document["parser_revision"] == PARSER
             and document["availability_policy"] == POLICY and document["source_precedence"] == PRECEDENCE,
             "unsupported rankings identity/parser/availability authority")
    _require(isinstance(document["collection_id"], str) and bool(document["collection_id"].strip()),
             "missing independently assigned rankings collection identity")
    _require(isinstance(document["fighters"], list) and document["fighters"], "missing independent fighter bindings")
    ids, urls = set(), set()
    for fighter in document["fighters"]:
        _require(isinstance(fighter, dict) and set(fighter) == {"fighter_id", "athlete_url", "scopes"},
                 "incomplete reviewed athlete binding")
        identity = fighter["fighter_id"]
        _require(isinstance(identity, str) and re.fullmatch(r"[0-9a-f]{16}", identity)
                 and identity not in ids, "invalid or duplicate stable fighter identity")
        url = _athlete_url(fighter["athlete_url"])
        _require(url == fighter["athlete_url"] and url not in urls, "ambiguous athlete-link binding")
        ids.add(identity)
        urls.add(url)
        scopes = fighter["scopes"]
        _require(isinstance(scopes, list) and scopes, "missing independently reviewed ranking eligibility")
        keys = set()
        for scope in scopes:
            _require(isinstance(scope, dict) and set(scope) == {"ranking_type", "division"}, "invalid ranking scope")
            expected = (_scope(scope["division"]) if scope["ranking_type"] == "division" else
                        {"ranking_type": "pfp", "division": scope["division"]} if scope["ranking_type"] == "pfp" and scope["division"] in {"men", "women"} else None)
            _require(expected == scope and _scope_key(scope) not in keys, "unknown or duplicate reviewed ranking scope")
            keys.add(_scope_key(scope))
    return document


def load_reviewed_registry(pin):
    """Caller supplies the independent exact pin; acquisition cannot mint it."""
    return _validate_registry(json.loads(_read_pin(pin)))


def _validate_response(body, envelope):
    _require(isinstance(envelope, dict) and set(envelope) == {
        "schema", "status_code", "requested_url", "issued_url", "effective_url", "redirect_chain", "received_at",
        "format", "response_bytes", "response_sha256"}, "missing original rankings response context")
    _require(type(envelope["schema"]) is int and envelope["schema"] == 1
             and type(envelope["status_code"]) is int and envelope["status_code"] == 200
             and envelope["format"] == "html" and type(envelope["response_bytes"]) is int and envelope["response_bytes"] == len(body)
             and envelope["response_sha256"] == _digest(body), "unproved exact successful rankings response")
    _require(envelope["requested_url"] == SOURCE_URL and envelope["issued_url"] == SOURCE_URL and envelope["effective_url"] == SOURCE_URL,
             "rankings response changed independently approved page ownership")
    redirects = envelope["redirect_chain"]
    _require(isinstance(redirects, list) and all(url == SOURCE_URL for url in redirects),
             "unapproved rankings redirect")
    return _time(envelope["received_at"])


def _observations(snapshot_body, evidence):
    snapshot = json.loads(snapshot_body)
    _require(isinstance(snapshot, dict) and type(snapshot.get("schema_version")) is int and snapshot.get("schema_version") == 2
             and type(snapshot.get("observation_reference_schema")) is int and snapshot.get("observation_reference_schema") == 1
             and snapshot.get("status") == "success" and snapshot.get("source") == "ufc.com"
             and snapshot.get("acquisition_failed") is False, "unsupported rankings snapshot")
    _require(isinstance(evidence, dict) and set(evidence) == {"registry", "response", "envelope", "attempts"},
             "missing rankings finalization evidence")
    registry_body = _attached(evidence["registry"])
    registry = _validate_registry(json.loads(registry_body))
    _require(snapshot.get("collection_id") == registry["collection_id"], "rankings collection identity differs from reviewed input")
    _require(snapshot.get("reviewed_context") == evidence["registry"]["pin"],
             "original reviewed rankings registry pin differs")
    attempts = evidence["attempts"]
    descriptors = snapshot.get("response_captures")
    _require(isinstance(attempts, list) and isinstance(descriptors, list) and attempts
             and len(attempts) == len(descriptors), "incomplete original rankings response attempts")
    for descriptor, attempt in zip(descriptors, attempts):
        _require(isinstance(attempt, dict) and set(attempt) == {"response", "envelope"}, "malformed retained source attempt")
        attempt_body, attempt_envelope = _attached(attempt["response"]), json.loads(_attached(attempt["envelope"]))
        _require(descriptor == {"schema": 1, "source": "ufc_rankings", "raw_response": attempt["response"]["pin"],
                                "response_envelope": attempt["envelope"]["pin"]}, "original response-attempt descriptor differs")
        _require(attempt_envelope.get("response_bytes") == len(attempt_body)
                 and attempt_envelope.get("response_sha256") == _digest(attempt_body), "corrupt original rankings attempt")
    _require(attempts[-1]["response"] == evidence["response"] and attempts[-1]["envelope"] == evidence["envelope"],
             "successful observation does not bind the final original response")
    body, envelope_body = _attached(evidence["response"]), _attached(evidence["envelope"])
    envelope = json.loads(envelope_body)
    received = _validate_response(body, envelope)
    _require(_time(snapshot["snapshot_time"]) >= received, "rankings snapshot predates original receipt")
    parsed = parse_official_rankings_html(body)
    _require(snapshot.get("rankings_observation") == parsed, "rankings values differ from exact original bytes")
    _require(snapshot.get("captured_at") == envelope["received_at"], "rankings captured time is not original receipt")
    effective = parsed["source_effective_date"]
    available = max(received, _effective_time(effective)) if effective else received
    tables = {_scope_key(table): table for table in parsed["tables"]}
    bindings = {fighter["athlete_url"]: fighter for fighter in registry["fighters"]}
    rows = []
    for fighter in registry["fighters"]:
        for scope in fighter["scopes"]:
            table = tables.get(_scope_key(scope))
            matches = [row for row in table["rows"] if row["athlete_url"] == fighter["athlete_url"]] if table else []
            resolved = len(matches) == 1 and matches[0]["resolved"]
            complete = bool(table and table["coverage"] == "complete_top15" and all(
                row["athlete_url"] in bindings and scope in bindings[row["athlete_url"]]["scopes"] for row in table["rows"]))
            value = matches[0]["observed_rank"] if resolved else 16 if complete and not matches else None
            status = "ranked" if resolved else "proven_unranked" if value == 16 else "unknown"
            rows.append({"fighter_id": fighter["fighter_id"], "athlete_url": fighter["athlete_url"], **scope,
                         "value": value, "status": status, "coverage": "complete_top15" if complete else "incomplete_or_unresolved",
                         "observed_at": envelope["received_at"], "available_at": available.isoformat(),
                         "source_effective_date": effective, "source": "ufc.com", "parser_revision": PARSER,
                         "availability_policy": POLICY, "source_precedence": PRECEDENCE,
                         "registry_sha256": _digest(registry_body), "response_sha256": _digest(body),
                         "response_envelope_sha256": _digest(envelope_body)})
    return rows


def finalize_snapshot(path, *, history_root, reviewed_context, capture):
    path = _safe_path(Path(path))
    body = path.read_bytes()
    _require(isinstance(capture, dict) and set(capture) == {"schema", "source", "raw_response", "response_envelope"}
             and capture["schema"] == 1 and capture["source"] == "ufc_rankings", "missing original UFC capture")
    snapshot = json.loads(body)
    evidence = {"registry": _attachment(reviewed_context), "response": _attachment(capture["raw_response"]),
                "envelope": _attachment(capture["response_envelope"]),
                "attempts": [{"response": _attachment(item["raw_response"]), "envelope": _attachment(item["response_envelope"])}
                             for item in snapshot.get("response_captures", [])]}
    observations = _observations(body, evidence)
    document = {"schema": 1, "kind": HISTORY_SUBDIR, "availability_policy": POLICY,
                "snapshot_sha256": _digest(body), "snapshot_body_base64": base64.b64encode(body).decode("ascii"),
                "evidence": evidence, "observations": observations}
    destination = _safe_path(Path(history_root)) / f"{_digest(body)}.json"
    _write_immutable(destination, _json_bytes(document))
    _require(path.read_bytes() == body, "rankings snapshot rotated during finalization")
    return destination


def validate_finalized_document(document):
    """Validate self-contained bytes after original source locations disappear."""
    from src.data.rosterwatch_rankings_collection import HISTORY_KIND, validate_finalized_document as validate_rosterwatch
    if isinstance(document, dict) and document.get("kind") == HISTORY_KIND:
        return validate_rosterwatch(document)
    _require(isinstance(document, dict) and set(document) == {"schema", "kind", "availability_policy",
             "snapshot_sha256", "snapshot_body_base64", "evidence", "observations"}
             and type(document["schema"]) is int and document["schema"] == 1 and document["kind"] == HISTORY_SUBDIR
             and document["availability_policy"] == POLICY, "unknown retained rankings history")
    try:
        body = base64.b64decode(document["snapshot_body_base64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise ObservationRefusal("invalid saved rankings snapshot") from exc
    _require(document["snapshot_sha256"] == _digest(body), "corrupt retained rankings snapshot")
    rows = _observations(body, document["evidence"])
    _require(rows == document["observations"], "retained rankings differ from exact evidence")
    return rows


def read_finalized_rankings(history_root):
    root = _safe_path(Path(history_root))
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.glob("*.json")):
        document = json.loads(_safe_path(path).read_bytes())
        _require(document.get("snapshot_sha256") == path.stem, "retained rankings filename differs")
        rows.extend(validate_finalized_document(document))
    return rows


def _cutoff(value):
    # A date alone cannot prove a same-day prefight receipt. Midnight is the
    # conservative boundary, in both training reconstruction and live lookup.
    return _effective_time(value)


def select_finalized_rankings(rows, *, fighter_id, weight_class=None, as_of, pfp_group=None):
    from src.data.rankings_scraper import _canonical_wc
    result = {"wc_rank_feat": None, "pfp_rank_feat": None}
    if not isinstance(fighter_id, str) or not re.fullmatch(r"[0-9a-f]{16}", fighter_id):
        return result
    cutoff = _cutoff(as_of)
    eligible = [row for row in rows if row["fighter_id"] == fighter_id and _time(row["available_at"]) < cutoff]
    for feature, kind in (("wc_rank_feat", "division"), ("pfp_rank_feat", "pfp")):
        matches = [row for row in eligible if row["ranking_type"] == kind and (
            row["division"] == _canonical_wc(weight_class) if kind == "division" else
            pfp_group is None or row["division"] == pfp_group)]
        if not matches:
            continue
        _require(len({row["division"] for row in matches}) == 1, "ambiguous independently bound P4P scope")
        latest = max(_time(row["available_at"]) for row in matches)
        selected = [row for row in matches if _time(row["available_at"]) == latest]
        # Keep historical originals. At an identical receipt time, the owner's
        # selected Roster Watch source wins; an explicit unknown stays unknown.
        preferred = [row for row in selected if row["source"] == "roster.watch"]
        if preferred:
            selected = preferred
        _require(len({(row["value"], row["status"], row["source"], row["athlete_url"]) for row in selected}) == 1,
                 "conflicting contemporaneous rankings from the selected source")
        result[feature] = selected[0]["value"]
    return result
