"""Durable append-only UFCStats refresh, with one accepted pair pointer.

Accepted CSV bodies and model binaries are never rewritten. Requests are saved
before parsing, and a saved job is recovered before a new job can be planned.
"""
from __future__ import annotations

import argparse
import base64
import csv
from contextlib import nullcontext, contextmanager
from contextvars import ContextVar
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
import uuid
from functools import wraps
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
FILES = ("fights_cleaned.csv", "features.csv")
INDEX_URL = "http://ufcstats.com/statistics/events/completed?page=all"
TARGET = "__incremental_refresh_target"
_PATHS = ContextVar("incremental_refresh_paths", default=(None, None))


def _mapped(value, encode=False):
    state, baseline = _PATHS.get()
    if isinstance(value, dict):
        return {key: _mapped(item, encode) for key, item in value.items()}
    if isinstance(value, list):
        return [_mapped(item, encode) for item in value]
    if isinstance(value, tuple):
        return [_mapped(item, encode) for item in value]
    if isinstance(value, str):
        for token, root in (("${STATE_DIR}", state), ("${BASELINE_DIR}", baseline)):
            if root is None:
                continue
            root_text = str(root).replace("\\", "/").rstrip("/")
            text = value.replace("\\", "/")
            if encode and (text.lower() == root_text.lower() or text.lower().startswith(root_text.lower() + "/")):
                return token + text[len(root_text):]
            if not encode and (text == token or text.startswith(token + "/")):
                return str(Path(root) / text[len(token):].lstrip("/"))
    return value


def _scope(function):
    @wraps(function)
    def scoped(*args, **kwargs):
        state = kwargs.get("state_dir", args[0] if args else None)
        baseline = kwargs.get("baseline_dir", args[1] if len(args) > 1 else None)
        if baseline is None and kwargs.get("selected_manifest"):
            baseline = Path(kwargs["selected_manifest"]).resolve().parent
        if baseline is not None:
            baseline = Path(baseline).resolve()
            if baseline.name == "processed":
                baseline = baseline.parent
        token = _PATHS.set((Path(state).resolve(), baseline))
        try:
            return function(*args, **kwargs)
        finally:
            _PATHS.reset(token)
    return scoped


@contextmanager
def _state_lock(state_dir):
    path = Path(state_dir) / "refresh.lock"
    _mkdir_durable(path.parent)
    stream = path.open("a+b")
    if path.stat().st_size == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError("Another incremental refresh owns this state directory") from None
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == "nt":
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _exclusive(function):
    @wraps(function)
    def locked(*args, **kwargs):
        with _state_lock(kwargs["state_dir"]):
            return function(*args, **kwargs)
    return locked


class UncertainRequestError(RuntimeError):
    """A prior GET was started, but its response was not durably saved."""


class SourceValidationError(ValueError):
    """A saved response definitively failed the acquisition quality contract."""


def _quality(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ValueError as exc:
            raise SourceValidationError(str(exc)) from exc
    return checked


def _pin(body):
    return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def _fsync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _mkdir_durable(path):
    """Persist every new directory's link before saving an intent below it."""
    path = Path(path)
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            raise FileNotFoundError(f"Directory root is absent: {cursor}")
        cursor = cursor.parent
    if not cursor.is_dir():
        raise NotADirectoryError(str(cursor))
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if not directory.is_dir():
                raise
        _fsync_directory(directory.parent)


def _atomic(path, body):
    path = Path(path)
    _mkdir_durable(path.parent)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_write(path, payload):
    _atomic(path, (json.dumps(_mapped(payload, True), sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def _json_read(path):
    return _mapped(json.loads(Path(path).read_bytes()))


def _instant(value=None):
    value = value or datetime.now(timezone.utc)
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("An aware decision instant is required")
    return value


def _url(url, kind=None):
    from src.data.ufcstats_http import normalize_ufcstats_url
    url = normalize_ufcstats_url(url)
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.netloc != "ufcstats.com" or parsed.fragment:
        raise ValueError("Only canonical UFCStats document URLs are allowed")
    if kind and not re.fullmatch("/" + kind + "/[a-f0-9]{16}", parsed.path):
        raise ValueError(f"Invalid UFCStats {kind} URL")
    return url


class ResponseCache:
    """One immutable response envelope per canonical URL in a job namespace."""

    def __init__(self, cache_dir, fetcher=None, *, retry_uncertain=False, request_delay=1.5):
        self.cache_dir = Path(cache_dir)
        self.fetcher = fetcher
        self.retry_uncertain = retry_uncertain
        self.request_delay = request_delay if fetcher is None else 0
        self._session = None

    def _paths(self, url):
        key = hashlib.sha256(url.encode()).hexdigest()
        return self.cache_dir / (key + ".json"), self.cache_dir / (key + ".intent.json")

    def record(self, url):
        url = _url(url)
        path, _ = self._paths(url)
        record = _json_read(path)
        body = base64.b64decode(record["body_base64"], validate=True)
        if record["url"] != url or record["status_code"] != 200 or record["body"] != _pin(body):
            raise ValueError("Cached response integrity check failed")
        return record

    def get_text(self, url):
        url = _url(url)
        path, intent = self._paths(url)
        if path.exists():
            record = self.record(url)
        else:
            if intent.exists() and not self.retry_uncertain:
                raise UncertainRequestError(f"Unanswered saved request: {url}")
            if intent.exists():
                request = _json_read(intent)
                if request["url"] != url:
                    raise ValueError("Saved request intent differs from its canonical URL")
                _instant(request["started_at"])
                request.setdefault("explicit_retry_started_at", []).append(_instant().isoformat())
            else:
                request = {"url": url, "started_at": _instant().isoformat()}
            _json_write(intent, request)
            if self.fetcher is None:
                import requests
                from src.data.ufcstats_http import request_ufcstats
                self._session = self._session or requests.Session()
                response = request_ufcstats(url, session=self._session)
            else:
                response = self.fetcher(url)
            if isinstance(response, str):
                body, encoding, status = response.encode(), "utf-8", 200
            else:
                status = int(response.status_code)
                body = response.content
                encoding = response.encoding or "utf-8"
            if status != 200:
                raise ValueError(f"UFCStats returned HTTP {status}: {url}")
            record = {"schema": 1, "url": url, "received_at": _instant().isoformat(),
                      "status_code": status, "encoding": encoding, "body": _pin(body),
                      "body_base64": base64.b64encode(body).decode("ascii")}
            _json_write(path, record)
            if self.request_delay:
                time.sleep(self.request_delay)
        return base64.b64decode(record["body_base64"]).decode(record["encoding"], errors="strict")


def _rows(body):
    parsed = list(csv.reader(io.StringIO(body.decode("utf-8-sig"), newline="")))
    if not parsed or len(set(parsed[0])) != len(parsed[0]):
        raise ValueError("CSV requires a unique, nonempty header")
    if any(len(row) != len(parsed[0]) for row in parsed[1:]):
        raise ValueError("CSV row width differs from its header")
    return parsed[0], parsed[1:]


def _identities(frame):
    from src.data.name_utils import normalize_person_name, normalize_ufcstats_id
    result = []
    for row in frame.to_dict("records"):
        date = pd.Timestamp(row["event_date"]).date().isoformat()
        names = tuple(sorted(normalize_person_name(str(row["fighter_" + side])) for side in ("a", "b")))
        ids = tuple(sorted(normalize_ufcstats_id(row.get("fighter_" + side + "_id")) or "" for side in ("a", "b")))
        if not all(names):
            raise ValueError("Unresolved fight participant identity")
        result.append((date, names, ids))
    return result


def _verify_snapshot(snapshot):
    frames, bodies = {}, {}
    for name in FILES:
        path = Path(snapshot["artifacts"][name]["path"])
        body = path.read_bytes()
        expected = snapshot["artifacts"][name]
        if any(expected[key] != _pin(body)[key] for key in ("bytes", "sha256")):
            raise ValueError(f"Accepted artifact changed: {name}")
        header, rows = _rows(body)
        if len(rows) != expected["rows"]:
            raise ValueError("Accepted artifact row count changed")
        bodies[name] = body
        frames[name] = pd.read_csv(io.BytesIO(body), low_memory=False)
    if Counter(_identities(frames[FILES[0]])) != Counter(_identities(frames[FILES[1]])):
        raise ValueError("Accepted fights/features occurrence relationship differs")
    for model in snapshot["models"].values():
        if _pin(Path(model["path"]).read_bytes()) != {key: model[key] for key in ("bytes", "sha256")}:
            raise ValueError("Accepted model changed")
    return bodies, frames


@_scope
def load_current_snapshot(state_dir, baseline_dir=None, selected_manifest=None):
    pointer = Path(state_dir) / "accepted.json"
    if pointer.exists():
        snapshot = _json_read(pointer)
        _verify_snapshot(snapshot)
        return snapshot
    if baseline_dir is None:
        raise ValueError("No accepted pointer exists; an explicit baseline is required")
    root = Path(baseline_dir).resolve()
    processed = root / "processed" if (root / "processed").is_dir() else root
    manifest = processed.parent / "selected-runtime-manifest.json"
    source_manifest = processed.parent / "selected-source-manifest.json"
    if selected_manifest:
        explicit = Path(selected_manifest).resolve()
        explicit_document = _json_read(explicit)
        if "processed" in explicit_document and "models" in explicit_document:
            manifest = explicit
        else:
            source_manifest = explicit
    document = _json_read(manifest) if manifest.exists() else {}
    artifacts = {}
    for name in FILES:
        path = processed / name
        body = path.read_bytes()
        if document:
            expected = document["processed"]["files"][name]
            if any(expected[key] != _pin(body)[key] for key in ("bytes", "sha256")):
                raise ValueError("Initial accepted artifact disagrees with its trusted manifest")
        artifacts[name] = {**_pin(body), "path": str(path), "rows": len(_rows(body)[1])}
    models = {role: {**item, "path": str((manifest.parent / item["path"]).resolve())}
              for role, item in document.get("models", {}).items()}
    snapshot = {"schema": 1, "generation_id": "baseline", "job_id": None,
                "processed_dir": str(processed), "fights_path": str(processed / FILES[0]),
                "features_path": str(processed / FILES[1]), "artifacts": artifacts, "models": models,
                "source_manifest_path": str(source_manifest)}
    _verify_snapshot(snapshot)
    return snapshot


def _clean(element):
    return " ".join(element.get_text(" ", strip=True).split()) if element else ""


@_quality
def _discover_events(cache, cutoff, decision_at):
    soup = BeautifulSoup(cache.get_text(INDEX_URL), "lxml")
    events = []
    parsed_count = 0
    for row in soup.select("tr.b-statistics__table-row"):
        link, date = row.select_one("a.b-link"), row.select_one("span.b-statistics__date")
        if not link or not date:
            continue
        actual_date = pd.to_datetime(_clean(date), errors="coerce")
        if pd.isna(actual_date):
            raise ValueError("Completed-events listing contains an unresolved date")
        parsed_count += 1
        if cutoff < actual_date.date() <= decision_at.astimezone(ZoneInfo("America/New_York")).date():
            cells = row.select("td")
            events.append({"url": _url(link["href"].strip(), "event-details"),
                           "name": _clean(link), "date": actual_date.date().isoformat(),
                           "location": _clean(cells[1]) if len(cells) > 1 else ""})
    if not parsed_count:
        raise ValueError("UFCStats completed-events listing did not expose event markup")
    if len({event["url"] for event in events}) != len(events):
        raise ValueError("Completed-events listing contains duplicate event URLs")
    return sorted(events, key=lambda item: (item["date"], item["url"]))


@_quality
def _event_fights(cache, event):
    soup = BeautifulSoup(cache.get_text(event["url"]), "lxml")
    date = next((_clean(item).split(":", 1)[1].strip() for item in soup.select("li.b-list__box-list-item")
                 if _clean(item).startswith("Date:")), None)
    if not date or pd.Timestamp(date).date().isoformat() != event["date"]:
        raise ValueError("Event page does not confirm its listing date")
    links = [_url(row["data-link"].strip(), "fight-details")
             for row in soup.select("tr.b-fight-details__table-row[data-link]")]
    if not links or len(set(links)) != len(links):
        raise ValueError("Completed event lacks a unique nonempty fight list")
    return links


@_quality
def _parse_fight(html, url, event):
    from src.data.fight_results import RESULT_POLICY_VERSION, classify_official_result, winner_for_side
    from src.data.name_utils import normalize_ufcstats_id
    soup = BeautifulSoup(html, "lxml")
    event_link = soup.select_one("h2.b-content__title a.b-link")
    if not event_link or _url(event_link.get("href", ""), "event-details") != event["url"]:
        raise ValueError("Fight page does not confirm its source event")
    names = soup.select("h3.b-fight-details__person-name a")
    persons = soup.select("div.b-fight-details__person")
    if len(names) != 2 or len(persons) < 2:
        raise ValueError("Fight page does not expose both participants")
    ids = [normalize_ufcstats_id(item.get("href")) for item in names]
    if not all(ids) or ids[0] == ids[1]:
        raise ValueError("Fight participants lack distinct stable UFCStats IDs")
    values = {}
    for item in soup.select("i.b-fight-details__text-item, i.b-fight-details__text-item_first"):
        text = _clean(item)
        if ":" in text:
            key, value = text.split(":", 1)
            values[key.strip().lower()] = value.strip()
    method = values.get("method", "")
    outcome = "/".join(_clean(person.select_one("i.b-fight-details__person-status")).upper() for person in persons[:2])
    classification = classify_official_result(outcome, method)
    if classification.quarantined:
        raise ValueError(f"Unverified fight result: {classification.quarantine_reason}")
    row = {"fighter_a": _clean(names[0]), "fighter_b": _clean(names[1]),
           "fighter_a_id": ids[0], "fighter_b_id": ids[1],
           "winner": winner_for_side(classification.winner_side, fighter_a=_clean(names[0]), fighter_b=_clean(names[1])),
           "official_outcome": outcome, "method": method, "round": values.get("round", ""),
           "time": values.get("time", ""), "time_format": values.get("time format", ""),
           "weight_class": _clean(soup.select_one("i.b-fight-details__fight-title")),
           "fight_url": url, "event_title": event["name"], "event_date": event["date"],
           "location": event["location"], "result_policy_version": RESULT_POLICY_VERSION,
           "result_class": classification.result_class, "binary_target_eligible": classification.binary_target_eligible,
           "result_quarantine_reason": classification.quarantine_reason}
    tables = [table for table in soup.select("table") if "js-fight-table" not in (table.get("class") or [])]
    totals = next((table.select_one("tbody tr") for table in tables
                   if table.select_one("tbody tr") and len(table.select_one("tbody tr").select("td")) >= 10), None)
    if totals is None:
        raise ValueError("Fight page lacks paired totals statistics")
    cells = [cell.select("p") for cell in totals.select("td")]
    if any(len(cell) < 2 for cell in cells[:10]):
        raise ValueError("Fight totals do not contain both participant observations")
    def pair(text):
        match = re.fullmatch(r"(\d+)\s+of\s+(\d+)", text)
        if not match or int(match[1]) > int(match[2]):
            raise ValueError("Fight count statistic is not an observed landed/attempted pair")
        return match[1], match[2]
    for side, index in (("a", 0), ("b", 1)):
        for column, field in ((1, "kd"), (7, "sub_att"), (8, "rev"), (9, "ctrl")):
            row[f"{side}_{field}"] = _clean(cells[column][index])
        for column, field in ((2, "sig_str"), (4, "total_str"), (5, "td")):
            landed, attempted = pair(_clean(cells[column][index]))
            row[f"{side}_{field}_landed"], row[f"{side}_{field}_attempted"] = landed, attempted
        for column, field in ((3, "sig_str_pct"), (6, "td_pct")):
            row[f"{side}_{field}"] = _clean(cells[column][index]).replace("%", "")
    breakdown = next((table.select_one("tbody tr") for table in tables
                      if table.select_one("tbody tr") and len(table.select_one("tbody tr").select("td")) == 9), None)
    if breakdown is None:
        raise ValueError("Fight page lacks paired significant-strike breakdown")
    cells = [cell.select("p") for cell in breakdown.select("td")]
    for column, field in ((3, "head"), (4, "body"), (5, "leg"), (6, "distance"), (7, "clinch"), (8, "ground")):
        if len(cells[column]) < 2:
            raise ValueError("Significant-strike breakdown lacks a participant")
        for side, index in (("a", 0), ("b", 1)):
            landed, attempted = pair(_clean(cells[column][index]))
            row[f"{side}_{field}_landed"], row[f"{side}_{field}_attempted"] = landed, attempted
    if not method or not re.fullmatch(r"[1-5]", row["round"]) or not re.fullmatch(r"[0-5]:[0-5]\d", row["time"]):
        raise ValueError("Fight completion chronology is unresolved")
    return row


@_quality
def _profile(html, url):
    soup = BeautifulSoup(html, "lxml")
    name = _clean(soup.select_one("h2.b-content__title span"))
    if not name:
        raise ValueError("Fighter profile lacks its participant name")
    row = {"name": name, "fighter_url": url}
    for item in soup.select("li.b-list__box-list-item"):
        text = _clean(item)
        if ":" in text:
            key, value = text.split(":", 1)
            if key.lower() in {"height", "reach", "weight", "stance", "dob"}:
                row[key.lower()] = value.strip()
    return row


def _historical_identity_associations(accepted_fights, cache):
    """Authenticate blank inherited roles from exact saved profile/fight pairs.

    A same-name fighter cannot inherit another person's timeline. The profile
    supplies the historical date/event, and the independently saved full fight
    supplies the exact participants, stable IDs, result and method.
    """
    from src.data.name_utils import canonical_fighter_name_key, normalize_ufcstats_id, normalize_person_name
    by_identity = {}
    for index, row in accepted_fights.iterrows():
        if any(normalize_ufcstats_id(row.get("fighter_" + side + "_id")) is None for side in ("a", "b")):
            key = (pd.Timestamp(row["event_date"]).date().isoformat(),
                   tuple(sorted(canonical_fighter_name_key(row["fighter_" + side]) for side in ("a", "b"))))
            by_identity.setdefault(key, []).append((index, row))
    result = {}
    for saved in sorted(cache.cache_dir.glob("*.json")):
        if saved.name.endswith(".intent.json"):
            continue
        envelope = _json_read(saved)
        url = envelope["url"]
        if "/fighter-details/" not in url:
            continue
        profile_record = cache.record(url)
        soup = BeautifulSoup(cache.get_text(url), "lxml")
        profile_name = _clean(soup.select_one("h2.b-content__title span"))
        owner = normalize_ufcstats_id(url)
        for bout in soup.select("tr.b-fight-details__table-row[data-link]"):
            names = bout.select("a[href*='/fighter-details/']")
            event_link = bout.select_one("a[href*='/event-details/']")
            if len(names) != 2 or event_link is None:
                continue
            identities = [normalize_ufcstats_id(item.get("href")) for item in names]
            if not all(identities) or identities[0] == identities[1] or owner not in identities:
                continue
            owner_side = identities.index(owner)
            if canonical_fighter_name_key(profile_name) != canonical_fighter_name_key(_clean(names[owner_side])):
                raise ValueError("Saved profile history contradicts its profile owner")
            date_cells = event_link.find_parent("td").select("p")
            if len(date_cells) != 2:
                continue
            date = pd.to_datetime(_clean(date_cells[1]).replace(".", ""), errors="coerce")
            if pd.isna(date):
                continue
            key = (date.date().isoformat(), tuple(sorted(canonical_fighter_name_key(_clean(item)) for item in names)))
            if key not in by_identity:
                continue
            fight_url = _url(bout["data-link"], "fight-details")
            response_path, _ = cache._paths(fight_url)
            if not response_path.exists():
                continue
            fight_record = cache.record(fight_url)
            event = {"url": _url(event_link["href"], "event-details"), "date": key[0],
                     "name": _clean(event_link), "location": ""}
            actual = _parse_fight(cache.get_text(fight_url), fight_url, event)
            source_names = {canonical_fighter_name_key(actual["fighter_" + side]): actual["fighter_" + side + "_id"]
                            for side in ("a", "b")}
            if set(source_names) != set(key[1]) or set(source_names.values()) != set(identities):
                raise ValueError("Saved profile and full historical fight identities disagree")
            for index, inherited in by_identity[key]:
                if (normalize_person_name(str(inherited.get("winner", ""))) != normalize_person_name(str(actual["winner"]))
                        or normalize_person_name(str(inherited.get("method", ""))) != normalize_person_name(actual["method"])):
                    raise ValueError("Saved historical fight evidence contradicts the accepted result/method")
                for side in ("a", "b"):
                    identity = source_names[canonical_fighter_name_key(inherited["fighter_" + side])]
                    stored = normalize_ufcstats_id(inherited.get("fighter_" + side + "_id"))
                    if stored is not None:
                        if stored != identity:
                            raise ValueError("Saved historical source contradicts an accepted stable ID")
                        continue
                    association = {"observed_row": int(index), "side": side, "name": inherited["fighter_" + side],
                                   "owner_key": "ufcstats:" + identity,
                                   "basis": "exact saved historical profile and full fight",
                                   "event_date": key[0], "event_url": event["url"], "fight_url": fight_url,
                                   "cache_dir": str(cache.cache_dir.resolve()),
                                   "profile_url": url, "profile_response": profile_record["body"],
                                   "fight_response": fight_record["body"]}
                    prior = result.get((int(index), side))
                    if prior is not None and prior["owner_key"] != association["owner_key"]:
                        raise ValueError("Historical role has contradictory saved source associations")
                    result[(int(index), side)] = association
    return list(result.values())


def build_new_pair(accepted_fights, raw_rows, profiles_path, source_manifest=None, *, retained_associations=None):
    """Return only new scientific rows; accepted observed fights are read-only context."""
    from src.data.ufc_refresh import _load_scraped_fighter_lookup, _scraped_fights_to_training_rows
    from src.features.build_features import build_new_features
    sources = None
    if source_manifest and Path(source_manifest).exists():
        from src.data.selected_source_inputs import SelectedSourceInputs
        sources = SelectedSourceInputs.from_manifest(source_manifest, processed_data_dir=profiles_path.parent)
        item = {**_pin(profiles_path.read_bytes()), "path": str(profiles_path.resolve())}
        sources = replace(sources, files={**sources.files, "profiles": item})
    with sources.activate() if sources else nullcontext():
        lookup_args = {} if sources is None else {
            "profile_integrity_decisions_path": sources.path("profile_decisions"),
            "prior_profile_integrity_decisions_path": sources.path("prior_profile_decisions"),
        }
        lookup = _load_scraped_fighter_lookup(profiles_path, **lookup_args)
        new = _scraped_fights_to_training_rows(pd.DataFrame(raw_rows), fighter_lookup=lookup)
        if len(new) != len(raw_rows):
            raise ValueError("Every acquired fight must produce exactly one normalized row")
        if TARGET in accepted_fights or TARGET in new:
            raise ValueError("Reserved incremental row marker is already present")
        # Original cells are never used as regenerated output or re-admitted.
        # Existing observed timelines supply the rolling/SOS/career dependencies.
        cache_dir = profiles_path.parent.parent.parent / "cache" / profiles_path.parent.name
        associations = {}
        for retained in retained_associations or []:
            old_cache = ResponseCache(retained["cache_dir"])
            if (old_cache.record(retained["profile_url"])["body"] != retained["profile_response"]
                    or old_cache.record(retained["fight_url"])["body"] != retained["fight_response"]):
                raise ValueError("Retained historical association source changed")
        cache_dirs = {Path(item["cache_dir"]) for item in retained_associations or []}
        if cache_dir.exists():
            cache_dirs.add(cache_dir)
        for directory in sorted(cache_dirs):
            for association in _historical_identity_associations(accepted_fights, ResponseCache(directory)):
                key = association["observed_row"], association["side"]
                if key in associations and associations[key]["owner_key"] != association["owner_key"]:
                    raise ValueError("Original and current historical association evidence conflicts")
                associations.setdefault(key, association)
        output = build_new_features(accepted_fights, new, historical_identity_associations=list(associations.values()))
        output.attrs["authenticated_historical_associations"] = list(associations.values())
        if len(output) != len(new):
            raise ValueError("Scientific new-feature occurrence relationship differs")
        return new, output


def _append(original, delta):
    header, old_rows = _rows(original)
    frame = delta.reindex(columns=header)
    suffix = frame.to_csv(index=False, header=False, lineterminator="\n").encode()
    result = original + (b"\n" if suffix and not original.endswith((b"\n", b"\r")) else b"") + suffix
    result_header, result_rows = _rows(result)
    if result_header != header or result_rows[:len(old_rows)] != old_rows or len(result_rows) != len(old_rows) + len(delta):
        raise ValueError("Accepted CSV cells/order/occurrences are not preserved")
    return result


def _job_path(state_dir, job_id):
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", job_id) or job_id in {".", ".."}:
        raise ValueError("A safe stable refresh job ID is required")
    return Path(state_dir) / "runs" / job_id / "journal.json"


@_scope
def inspect_active_refresh(state_dir):
    active = Path(state_dir) / "active.json"
    if not active.exists():
        return None
    job_id = _json_read(active)["job_id"]
    path = _job_path(state_dir, job_id)
    journal = _json_read(path)
    if journal["phase"] in {"COMPLETED", "FAILED"}:
        return None
    cache = ResponseCache(journal["cache_dir"])
    uncertain = any(not intent.with_name(intent.name.replace(".intent.json", ".json")).exists()
                    for intent in cache.cache_dir.glob("*.intent.json"))
    return {"job_id": job_id, "phase": journal["phase"], "resume_allowed": not uncertain,
            "status": "held" if uncertain else "pending"}


def _commit(state_dir, path, journal, fault):
    candidate = journal["candidate"]
    _verify_snapshot(candidate)
    pointer = Path(state_dir) / "accepted.json"
    already_published = False
    if pointer.exists():
        current = _json_read(pointer)
        if current != candidate and current != journal["baseline"]:
            raise ValueError("Accepted successor advanced outside this original job")
        already_published = current == candidate
    if not already_published:
        _json_write(pointer, candidate)
    fault("after_pointer")
    completed_at = _instant().isoformat()
    journal["phase"] = "COMPLETED"
    journal["result"] = {**candidate, "status": "completed", "new_rows": journal["new_rows"],
                         "total_rows": candidate["artifacts"][FILES[0]]["rows"],
                         "completed_at": completed_at, "finished_at": completed_at,
                         "publication_at": datetime.fromtimestamp(pointer.stat().st_mtime, timezone.utc).isoformat(),
                         "accepted_pointer": str(pointer.resolve())}
    _json_write(path, journal)
    return journal["result"]


@_scope
@_exclusive
def run_incremental_refresh(*, baseline_dir, state_dir, selected_manifest=None, job_id=None,
                            decision_at=None, fetcher=None, retry_uncertain=False, pair_builder=None, fault=None):
    state_dir = Path(state_dir).resolve()
    fault = fault or (lambda phase: None)
    active = inspect_active_refresh(state_dir)
    if active and job_id and active["job_id"] != job_id:
        raise ValueError("Recover the original active job before planning a fresh job")
    job_id = active["job_id"] if active else (job_id or "refresh-" + _instant(decision_at).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
    path = _job_path(state_dir, job_id)
    if path.exists():
        journal = _json_read(path)
        if journal["phase"] == "FAILED":
            return journal["result"]
        if journal["phase"] == "COMPLETED":
            _verify_snapshot(journal["candidate"])
            return journal["result"]
        if journal["phase"] == "STAGED":
            return _commit(state_dir, path, journal, fault)
    else:
        baseline = load_current_snapshot(state_dir, baseline_dir, selected_manifest)
        journal = {"schema": 1, "job_id": job_id, "phase": "ACQUIRING", "baseline": baseline,
                   "decision_at": _instant(decision_at).isoformat(),
                   "cache_dir": str(state_dir / "cache" / job_id), "events": None}
        _json_write(path, journal)
        _json_write(state_dir / "active.json", {"job_id": job_id})
    baseline = journal["baseline"]
    original, frames = _verify_snapshot(baseline)
    cache = ResponseCache(journal["cache_dir"], fetcher, retry_uncertain=retry_uncertain)
    try:
        if journal["events"] is None:
            cutoff = pd.to_datetime(frames[FILES[0]]["event_date"], format="mixed").max().date()
            journal["events"] = _discover_events(cache, cutoff, _instant(journal["decision_at"]))
            _json_write(path, journal)
        raw_rows = []
        for event in journal["events"]:
            for url in _event_fights(cache, event):
                parsed_path = path.parent / "parsed" / (hashlib.sha256(url.encode()).hexdigest() + ".json")
                if parsed_path.exists():
                    parsed = _json_read(parsed_path)
                    if parsed["response"] != cache.record(url)["body"] or parsed["event"] != event:
                        raise ValueError("Parsed result/stat pair disagrees with its original response")
                    row = parsed["row"]
                    if _parse_fight(cache.get_text(url), url, event) != row:
                        raise ValueError("Saved parsed result/stat pair differs from its raw response")
                else:
                    row = _parse_fight(cache.get_text(url), url, event)
                    _json_write(parsed_path, {"event": event, "response": cache.record(url)["body"], "row": row})
                raw_rows.append(row)
                fault("after_parsed_fight")
        new_identities = _identities(pd.DataFrame(raw_rows)) if raw_rows else []
        inherited = _identities(frames[FILES[0]])
        # Stable IDs and the historical name/date domain both forbid additions
        # from becoming another occurrence of a saved historical fight.
        old_names = {(date, names) for date, names, ids in inherited}
        old_ids = {(date, ids) for date, names, ids in inherited if all(ids)}
        new_names = [(date, names) for date, names, ids in new_identities]
        new_ids = [(date, ids) for date, names, ids in new_identities]
        if (len(set(new_names)) != len(new_names) or len(set(new_ids)) != len(new_ids)
                or any((date, names) in old_names or (date, ids) in old_ids for date, names, ids in new_identities)):
            raise SourceValidationError("New fight identities must be unique and disjoint from accepted history")
        profiles = []
        for row in raw_rows:
            for side in ("a", "b"):
                url = _url("http://ufcstats.com/fighter-details/" + row["fighter_" + side + "_id"], "fighter-details")
                if all(item["fighter_url"] != url for item in profiles):
                    observed_profile = _profile(cache.get_text(url), url)
                    from src.data.name_utils import same_person_name
                    if not same_person_name(observed_profile["name"], row["fighter_" + side]):
                        raise SourceValidationError("Fresh profile name differs from its fight participant")
                    profiles.append(observed_profile)
        source_manifest = Path(baseline["source_manifest_path"])
        if source_manifest.exists():
            from src.data.selected_source_inputs import SelectedSourceInputs
            selected = SelectedSourceInputs.from_manifest(source_manifest, processed_data_dir=baseline["processed_dir"])
            old_profiles = pd.read_csv(selected.path("profiles"), low_memory=False)
            from src.data.reviewed_roster_recovery import apply_permanent_profile_locks_to_raw_rows
            with selected.activate():
                apply_permanent_profile_locks_to_raw_rows(profiles)
            if profiles:
                from src.data.name_utils import normalize_ufcstats_id
                identities = {normalize_ufcstats_id(item["fighter_url"]) for item in profiles}
                old_profiles = old_profiles.loc[~old_profiles["fighter_url"].map(normalize_ufcstats_id).isin(identities)]
            profiles_frame = pd.concat([old_profiles, pd.DataFrame(profiles)], ignore_index=True, sort=False)
        else:
            profiles_frame = pd.DataFrame(profiles)
        profiles_path = path.parent / "profiles.csv"
        _atomic(profiles_path, profiles_frame.to_csv(index=False).encode())
        journal["phase"] = "BUILDING"
        _json_write(path, journal)
        if raw_rows:
            if pair_builder is None:
                new_fights, new_features = build_new_pair(frames[FILES[0]], raw_rows, profiles_path, source_manifest,
                    retained_associations=baseline.get("historical_identity_associations", []))
            else:
                new_fights, new_features = pair_builder(frames[FILES[0]], raw_rows, profiles_path, source_manifest)
            if _identities(new_fights) != new_identities or _identities(new_features) != new_identities:
                raise ValueError("Scientific output differs from the acquired new fight occurrence order")
        else:
            new_fights = pd.DataFrame(columns=frames[FILES[0]].columns)
            new_features = pd.DataFrame(columns=frames[FILES[1]].columns)
        generation = state_dir / "generations" / job_id
        processed = generation / "processed"
        artifacts, preservation = {}, {}
        for name, delta in zip(FILES, (new_fights, new_features)):
            body = _append(original[name], delta)
            target = processed / name
            _atomic(target, body)
            artifacts[name] = {**_pin(body), "path": str(target), "rows": len(frames[name]) + len(delta)}
            preservation[name] = {"prefix": _pin(original[name]), "prefix_rows": len(frames[name]),
                                  "prefix_bytes_equal": body.startswith(original[name]), "added_rows": len(delta),
                                  "historical_cells_order_occurrences_preserved": True}
            fault("after_" + name)
        report_path = generation / "refresh-report.json"
        report = {"schema": 1, "job_id": job_id, "decision_at": journal["decision_at"],
                  "new_rows": len(new_fights), "preservation": preservation,
                  "inherited_rows": len(inherited), "inherited_distinct_identities": len(set(inherited)),
                  "new_identities": new_identities, "models": baseline["models"],
                  "source_manifest_path": str(source_manifest), "cache_dir": journal["cache_dir"],
                  "scientific_context": new_features.attrs.get("new_feature_context", {}),
                  "profiles": {**_pin(profiles_path.read_bytes()), "path": str(profiles_path)},
                  "source_manifest": ({**_pin(source_manifest.read_bytes()), "path": str(source_manifest)}
                                      if source_manifest.exists() else None),
                  "new_stats": raw_rows, "historical_artifacts_rewritten": False, "model_training_performed": False}
        _json_write(report_path, report)
        candidate = {"schema": 1, "generation_id": job_id, "job_id": job_id,
                     "generation_dir": str(generation), "processed_dir": str(processed),
                     "fights_path": str(processed / FILES[0]), "features_path": str(processed / FILES[1]),
                     "artifacts": artifacts, "models": baseline["models"],
                     "source_manifest_path": str(source_manifest), "cache_dir": journal["cache_dir"],
                     "profiles_path": str(profiles_path), "report_path": str(report_path)}
        candidate["historical_identity_associations"] = new_features.attrs.get(
            "authenticated_historical_associations", baseline.get("historical_identity_associations", []))
        _verify_snapshot(candidate)
        _verify_snapshot(baseline)
        journal.update(phase="STAGED", candidate=candidate, new_rows=len(new_fights))
        _json_write(path, journal)
        fault("after_staged")
        return _commit(state_dir, path, journal, fault)
    except UncertainRequestError as exc:
        return {"status": "held", "job_id": job_id, "phase": journal["phase"],
                "resume_allowed": False, "reason": str(exc)}
    except SourceValidationError as exc:
        # Only a saved, definitive quality refusal can complete a failed attempt.
        # Any unanswered GET, model drift, pointer drift or publication boundary
        # remains held; it cannot authorize a conditional retry slot.
        _verify_snapshot(baseline)
        pointer = state_dir / "accepted.json"
        if pointer.exists() and _json_read(pointer) != baseline:
            raise ValueError("Failure evidence cannot establish unchanged acceptance") from exc
        if inspect_active_refresh(state_dir)["resume_allowed"] is False:
            return {"status": "held", "job_id": job_id, "phase": journal["phase"],
                    "resume_allowed": False, "reason": str(exc)}
        records = []
        for cached in sorted(cache.cache_dir.glob("*.json")):
            if cached.name.endswith(".intent.json"):
                continue
            record = _json_read(cached)
            cache.record(record["url"])
            records.append({"url": record["url"], "received_at": record["received_at"], "body": record["body"]})
        result = {"status": "failed", "failure_complete": True, "job_id": job_id,
                  "reason": str(exc), "decision_at": journal["decision_at"],
                  "finished_at": _instant().isoformat(), "accepted_pointer_unchanged": True,
                  "publication_performed": False, "responses": records}
        journal.update(phase="FAILED", result=result)
        _json_write(path, journal)
        return result


@_scope
def recover_incremental_refresh(*, state_dir, baseline_dir=None, selected_manifest=None,
                                fetcher=None, decision_at=None, retry_uncertain=False, pair_builder=None, fault=None):
    active = inspect_active_refresh(state_dir)
    if active is None:
        return None
    if not active["resume_allowed"] and not retry_uncertain:
        return active
    journal = _json_read(_job_path(state_dir, active["job_id"]))
    return run_incremental_refresh(baseline_dir=baseline_dir or journal["baseline"]["processed_dir"],
                                   state_dir=state_dir, selected_manifest=selected_manifest,
                                   job_id=active["job_id"], fetcher=fetcher, retry_uncertain=retry_uncertain,
                                   pair_builder=pair_builder, fault=fault)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--selected-manifest")
    parser.add_argument("--job-id")
    parser.add_argument("--decision-at")
    parser.add_argument("--retry-uncertain", action="store_true", help="Explicitly retry an unanswered read-only GET")
    args = parser.parse_args()
    result = run_incremental_refresh(**vars(args))
    logger.info("Incremental refresh %s: %s new rows", result["status"], result.get("new_rows"))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
