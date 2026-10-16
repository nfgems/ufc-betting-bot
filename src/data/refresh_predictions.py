"""Fresh, saved-input predictions with the selected models kept unchanged.

Collection is bounded to announced UFCStats cards and their exact fighter IDs.
Responses are retained before parsing.  A completed job can be reused after a
restart; neither model fitting nor betting is part of this module.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)
UPCOMING_URL = "http://ufcstats.com/statistics/events/upcoming?page=all"
MODEL_CONTRACT = "full_live_contract_v6_durability_2022"
SCHEMA = 1


class RefreshPredictionError(ValueError):
    """Inputs cannot safely produce a complete saved prediction report."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pin(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": _sha256(path)}


def _portable_pin(pin: dict, state: Path) -> dict:
    return {**pin, "path": os.path.relpath(pin["path"], state).replace("\\", "/")}


def _binding(evidence: dict) -> dict:
    """Paths may relocate; generation and original bytes must remain exact."""
    return {"accepted_generation": evidence["accepted_generation"],
            "release_id": evidence["release_id"],
            "model_pins": {role: {key: item[key] for key in ("bytes", "sha256")}
                           for role, item in evidence["model_pins"].items()},
            "data_pins": {name: {key: item[key] for key in ("bytes", "sha256")}
                          for name, item in evidence["data_pins"].items()},
            "selected_runtime_manifest": {key: evidence["selected_runtime_manifest"][key]
                                          for key in ("bytes", "sha256")}}


def _verify_file(path: Path, pin: dict) -> None:
    if (not path.is_file() or path.stat().st_size != pin.get("bytes")
            or _sha256(path) != pin.get("sha256")):
        raise RefreshPredictionError(f"Prediction input pin mismatch: {path.name}")


def _json_value(value):
    """Use JSON null for genuine missing values; never invent replacements."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _make_directory(path: Path) -> None:
    """Persist new directory links before saving an original intent/input."""
    missing = []
    current = Path(path)
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        if os.name != "nt":
            descriptor = os.open(directory.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def _save_json(path: Path, document: dict) -> None:
    from src.data.io_utils import write_bytes_atomically
    _make_directory(path.parent)
    body = (json.dumps(_json_value(document), indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")
    write_bytes_atomically(body, path)


def _decision_time(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    instant = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00"))
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise RefreshPredictionError("Prediction decision instant must be timezone aware")
    return instant.astimezone(timezone.utc)


def _date(value: str) -> date:
    text = str(value or "").strip()
    for pattern in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            pass
    raise RefreshPredictionError(f"Invalid official event date: {text}")


def _detail_url(value: str, role: str) -> str:
    parsed = urlsplit(str(value or "").strip())
    if (parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"ufcstats.com", "www.ufcstats.com"}
            or parsed.username or parsed.password or parsed.port is not None
            or parsed.query or parsed.fragment
            or not re.fullmatch(r"/" + role + r"-details/[0-9a-f]{16}", parsed.path)):
        raise RefreshPredictionError("Untrusted or malformed UFCStats detail URL")
    return "http://ufcstats.com" + parsed.path


def parse_upcoming_events(html: str) -> list[dict]:
    """Parse exact event routes and official dates from the UFCStats index."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    events = []
    seen = set()
    for row in soup.select("tr.b-statistics__table-row"):
        link = row.select_one("a.b-link[href*='event-details/']")
        if link is None:
            continue
        date_node = row.select_one("span.b-statistics__date")
        if date_node is None:
            raise RefreshPredictionError("Upcoming event has no official date")
        url = _detail_url(link["href"], "event")
        if url in seen:
            raise RefreshPredictionError("Duplicate event route in upcoming index")
        title = link.get_text(" ", strip=True)
        if not title:
            raise RefreshPredictionError("Upcoming event has no title")
        seen.add(url)
        events.append({"title": title, "url": url,
                       "event_date": _date(date_node.get_text(strip=True)).isoformat()})
    if not events:
        raise RefreshPredictionError("UFCStats upcoming index has no validated events")
    return sorted(events, key=lambda row: (row["event_date"], row["url"]))


def parse_event_card(html: str, event: dict) -> list[dict]:
    """Use the running release's exact card parsing and event-context rules."""
    from bs4 import BeautifulSoup
    from src.data.event_context import infer_empty_arena
    from src.data.live_monitor import _extract_event_location, _extract_upcoming_weight_class
    soup = BeautifulSoup(html, "lxml")
    title_node = soup.select_one("h2.b-content__title .b-content__title-highlight")
    if title_node is None or title_node.get_text(" ", strip=True) != event["title"]:
        raise RefreshPredictionError("Card body title differs from its original upcoming event")
    observed_dates = []
    for node in soup.select("li.b-list__box-list-item"):
        label = node.select_one("i.b-list__box-item-title")
        if label is not None and label.get_text(" ", strip=True) == "Date:":
            observed_dates.append(node.get_text(" ", strip=True).split(":", 1)[1].strip())
    if len(observed_dates) != 1 or _date(observed_dates[0]).isoformat() != event["event_date"]:
        raise RefreshPredictionError("Card body date differs from its original upcoming event")
    location = _extract_event_location(soup)
    fights = []
    seen = set()
    completed_rows = 0
    card_row_count = 0
    for row in soup.select("tr.b-fight-details__table-row"):
        if not row.select("td"):
            continue
        main_event = card_row_count == 0
        card_row_count += 1
        status = row.select("td")[0].get_text(" ", strip=True).casefold()
        if status in {"win", "loss", "w", "l", "w l", "l w", "draw", "d", "d d",
                      "draw draw", "nc", "nc nc", "no contest", "no contest no contest"}:
            completed_rows += 1
            continue
        links = row.select("a.b-link[href*='fighter-details/']")
        if len(links) != 2:
            raise RefreshPredictionError("Upcoming card contains an incomplete fighter row")
        urls = [_detail_url(link["href"], "fighter") for link in links]
        ids = [url.rsplit("/", 1)[1] for url in urls]
        if ids[0] == ids[1] or tuple(sorted(ids)) in seen:
            raise RefreshPredictionError("Upcoming card contains a duplicate/conflicting fighter pair")
        seen.add(tuple(sorted(ids)))
        text = row.get_text(" ", strip=True).casefold()
        title = bool(row.select_one("img[src*='belt']")
                     or any(marker in text for marker in ("title bout", "championship", "interim")))
        weight_class = _extract_upcoming_weight_class(row)
        if not weight_class:
            raise RefreshPredictionError("Upcoming card has no observed weight class")
        fight = {"event_title": event["title"], "event_date": event["event_date"],
                 "event_url": event["url"], "event_id": event["url"].rsplit("/", 1)[1],
                 "weight_class": weight_class, "is_title_bout": title,
                 "is_main_event": main_event, "num_rounds": 5 if title or main_event else 3,
                 "location": location,
                 "is_empty_arena": infer_empty_arena(event["title"], location)}
        for side, link, identity, url in zip(("a", "b"), links, ids, urls):
            name = link.get_text(" ", strip=True)
            if not name:
                raise RefreshPredictionError("Upcoming card has a blank fighter name")
            fight[f"fighter_{side}"] = name
            fight[f"fighter_{side}_id"] = identity
            fight[f"fighter_{side}_url"] = url
        fights.append(fight)
    if not fights and not completed_rows:
        raise RefreshPredictionError("Upcoming event card has no validated bouts")
    return fights


def parse_fighter_profile(html: str, url: str) -> dict:
    """Retain source strings using the existing UFCStats profile CSV schema."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    name_node = soup.select_one("h2.b-content__title span")
    if name_node is None or not name_node.get_text(strip=True):
        raise RefreshPredictionError("UFCStats profile has no fighter identity")
    record_node = soup.select_one("span.b-content__title-record")
    result = {"name": name_node.get_text(" ", strip=True),
              "fighter_url": _detail_url(url, "fighter"),
              "record": record_node.get_text(" ", strip=True).replace("Record:", "").strip()
              if record_node else ""}
    fields = {"Height": "height", "Weight": "weight", "Reach": "reach", "STANCE": "stance",
              "DOB": "dob", "SLpM": "slpm", "Str. Acc.": "str_acc", "SApM": "sapm",
              "Str. Def": "str_def", "TD Avg.": "td_avg", "TD Acc.": "td_acc",
              "TD Def.": "td_def", "Sub. Avg.": "sub_avg"}
    for node in soup.select("li.b-list__box-list-item"):
        text = re.sub(r"\s+", " ", node.get_text(" ", strip=True))
        if ":" not in text:
            continue
        label, value = text.split(":", 1)
        if label.strip() in fields:
            result[fields[label.strip()]] = value.strip().replace("%", "")
    # Unpublished values are represented as missing strings, not default stats.
    for field in fields.values():
        result.setdefault(field, "")
    return result


def _load_snapshot(accepted_pointer, selected_manifest: Path) -> dict:
    if isinstance(accepted_pointer, dict):
        result = dict(accepted_pointer)
        if result.get("processed_dir") and result.get("artifacts"):
            # This is the caller's original completion, not a request to adopt
            # a later unrelated pointer at the same mutable path.
            return result
        accepted_pointer = result["accepted_pointer"]
    pointer = Path(accepted_pointer).resolve()
    if pointer.name != "accepted.json":
        raise RefreshPredictionError("An explicit accepted.json pointer is required")
    from src.data.incremental_refresh import load_current_snapshot
    return load_current_snapshot(state_dir=pointer.parent,
                                 baseline_dir=selected_manifest.parent,
                                 selected_manifest=selected_manifest)


def _load_models(manifest_path: Path, manifest: dict) -> tuple[dict, dict]:
    import joblib
    root = manifest_path.parent.resolve()
    pins = {}
    loaded = {}
    if set(manifest.get("models", {})) != {"primary", "no_odds", "logistic"}:
        raise RefreshPredictionError("Selected runtime must pin all three model roles")
    for role, record in manifest["models"].items():
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root):
            raise RefreshPredictionError("Selected model escapes release root")
        _verify_file(path, record)
        pins[role] = _pin(path)
        if role == "logistic":
            continue
        # Direct loading prevents legacy load_model's optional artifact repair.
        artifact = joblib.load(path)
        spec = artifact.get("training_spec", {})
        expected_name = MODEL_CONTRACT + ("_no_odds" if role == "no_odds" else "")
        expected_count = 202 if role == "no_odds" else 211
        columns = artifact.get("feature_cols")
        if (not isinstance(columns, list) or len(columns) != expected_count
                or len(set(columns)) != len(columns) or columns != spec.get("feature_cols")
                or spec.get("name") != expected_name
                or artifact.get("impute_strategy", spec.get("impute_strategy")) != "native_nan"
                or artifact.get("model") is None):
            raise RefreshPredictionError("Selected model does not carry the accepted native-NaN contract")
        loaded[role] = artifact
    return loaded, pins


def _merged_profiles(sources, observations: list[dict], path: Path) -> dict:
    import pandas as pd
    from src.data.name_utils import normalize_ufcstats_id, normalize_person_name
    from src.data.io_utils import write_bytes_atomically
    frame = pd.read_csv(sources.path("profiles"), dtype=str, keep_default_na=False)
    ids = frame["fighter_url"].map(normalize_ufcstats_id)
    if ids.isna().any() or ids.duplicated().any():
        raise RefreshPredictionError("Accepted profile inventory has invalid/duplicate stable IDs")
    records = frame.to_dict("records")
    positions = {identity: index for index, identity in enumerate(ids)}
    for observation in observations:
        identity = normalize_ufcstats_id(observation["fighter_url"])
        if identity in positions:
            prior = records[positions[identity]]
            # Current source naming cannot silently replace a permanent owner.
            if normalize_person_name(prior["name"]) != normalize_person_name(observation["name"]):
                observation = {**observation, "name": prior["name"]}
            records[positions[identity]] = {**prior, **observation}
        else:
            positions[identity] = len(records)
            records.append(observation)
    out = io.StringIO(newline="")
    pd.DataFrame(records, columns=frame.columns).to_csv(out, index=False, lineterminator="\n")
    _make_directory(path.parent)
    write_bytes_atomically(out.getvalue().encode("utf-8"), path)
    return _pin(path)


def _observed_aliases(sources, cards: list[dict], observations: dict, cache, path: Path) -> dict:
    """Bind new source spellings only to the ID on its retained exact route."""
    from src.data.name_utils import same_person_name
    aliases = json.loads(sources.path("identity_aliases").read_bytes())
    known = {(item["fighter_id"], item["name"]) for item in aliases}
    for fight in cards:
        card_evidence = cache.record(fight["event_url"])
        for side in ("a", "b"):
            identity = fight[f"fighter_{side}_id"]
            profile = observations[fight[f"fighter_{side}_url"]]
            if not same_person_name(fight[f"fighter_{side}"], profile["name"]):
                raise RefreshPredictionError("Fresh profile body name differs from its card participant")
            for name in (fight[f"fighter_{side}"], profile["name"]):
                if (identity, name) not in known:
                    known.add((identity, name))
                    aliases.append({"fighter_id": identity, "name": name,
                                    "evidence": {"event_url": fight["event_url"],
                                                 "event_body_sha256": card_evidence["body"]["sha256"],
                                                 "profile_url": profile["fighter_url"],
                                                 "profile_body_sha256": cache.record(profile["fighter_url"])["body"]["sha256"]}})
    from src.data.io_utils import write_bytes_atomically
    _make_directory(path.parent)
    write_bytes_atomically((json.dumps(aliases, sort_keys=True, indent=2) + "\n").encode(), path)
    return _pin(path)


def _response_evidence(cache, url: str, state: Path) -> tuple[dict, dict]:
    record = cache.record(url)
    path = Path(cache.cache_dir) / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    pin = _pin(path)
    return {"url": url, "received_at": record["received_at"],
            "status_code": record["status_code"], "body": record["body"],
            "envelope": _portable_pin(pin, state)}, pin


def _document_digest(document: dict) -> str:
    body = json.dumps(_json_value(document), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _empty_observation_inventory(roots: dict, state: Path) -> dict:
    """This route admits no current market, method, ranking, or line files."""
    inventory = {}
    for name, path in roots.items():
        if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
            raise RefreshPredictionError("Live observation roots must remain empty for this original refresh")
        inventory[name] = {"path": path.relative_to(state).as_posix(), "entries": []}
    return inventory


def _completed_bout(record: dict, *, fight: dict, calculation: dict,
                    calculation_digest: str, feature_cols: list[str]) -> dict:
    """Admit one original atomic result without recalculating its features."""
    expected_card = _json_value(fight)
    if (not isinstance(record, dict) or record.get("schema") != SCHEMA or record.get("status") != "completed"
            or record.get("calculation_sha256") != calculation_digest
            or record.get("binding") != calculation["binding"]
            or record.get("decision_at") != calculation["decision_at"]
            or record.get("card") != expected_card):
        raise RefreshPredictionError("Saved bout differs from its original calculation binding")
    row = record.get("row")
    if not isinstance(row, dict) or record.get("row_sha256") != _document_digest(row):
        raise RefreshPredictionError("Saved bout result digest differs from its original row")
    if any(row.get(key) != value for key, value in expected_card.items()):
        raise RefreshPredictionError("Saved bout identity/context differs from its original card")
    features = row.get("model_features")
    if (not isinstance(features, dict) or set(features) != set(feature_cols)
            or any(value is not None and (not isinstance(value, (int, float))
                                         or not math.isfinite(value)) for value in features.values())
            or row.get("missing_feature_count") != sum(value is None for value in features.values())
            or row.get("market_inputs_available") is not False):
        raise RefreshPredictionError("Saved bout does not retain the original native-NaN feature contract")
    for prefix in ("", "no_odds_"):
        a, b = row.get(prefix + "prob_a"), row.get(prefix + "prob_b")
        if (not all(type(value) in (int, float) and math.isfinite(value)
                    and 0 <= value <= 1 for value in (a, b))
                or not math.isclose(a + b, 1.0, abs_tol=1e-12)):
            raise RefreshPredictionError("Saved bout has invalid original win probabilities")
    winner = "a" if row["prob_a"] > row["prob_b"] else "b"
    if (row.get("predicted_winner") != winner
            or row.get("predicted_winner_name") != fight["fighter_" + winner]
            or row.get("confidence") != max(row["prob_a"], row["prob_b"])):
        raise RefreshPredictionError("Saved bout outcome differs from its original probabilities")
    return row


def run_refresh_predictions(*, accepted_pointer, selected_manifest, state_dir,
                            decision_at=None, force_refresh: bool = False,
                            response_cache=None, max_events: int = 1,
                            horizon_days: int = 45) -> dict:
    """Save probabilities from current accepted history and pinned v9 models.

    Use a distinct ``state_dir`` for each original scheduled job.  On recovery
    the same directory and decision instant reuse both raw responses and the
    complete report.  No trading configuration or model file is written.
    """
    from src.data.selected_source_inputs import SelectedSourceInputs
    from src.data.fighter_lookup import build_fight_features
    from src.model.training_spec import NamedModelTrainingSpec
    from src.model.predict import predict_fight
    state = Path(state_dir).resolve()
    _make_directory(state)
    observation_roots = {name: state / "observations" / name for name in
                         ("method_observations", "rankings_observations", "line_history")}
    observation_inventory = _empty_observation_inventory(observation_roots, state)
    output = state / "predictions.json"
    instant = _decision_time(decision_at)
    manifest_path = Path(selected_manifest).resolve()
    snapshot = _load_snapshot(accepted_pointer, manifest_path)
    from src.data.incremental_refresh import _verify_snapshot
    _bodies, frames = _verify_snapshot(snapshot)
    processed = Path(snapshot["processed_dir"]).resolve()
    generation = str(snapshot.get("generation_id") or snapshot.get("job_id") or processed)
    manifest = json.loads(manifest_path.read_bytes())
    models, model_pins = _load_models(manifest_path, manifest)
    if set(snapshot.get("models", {})) != set(model_pins) or any(
            any(snapshot["models"][role].get(key) != pin[key] for key in ("bytes", "sha256"))
            for role, pin in model_pins.items()):
        raise RefreshPredictionError("Accepted data pointer changed the original model association")
    data_pins = {}
    for name in ("fights_cleaned.csv", "features.csv"):
        record = snapshot["artifacts"][name]
        path = processed / name
        _verify_file(path, record)
        data_pins[name] = _pin(path)
    evidence = {"accepted_generation": generation, "processed_dir": str(processed),
                "model_pins": model_pins, "data_pins": data_pins,
                "selected_runtime_manifest": _pin(manifest_path),
                "release_id": manifest["release_id"]}
    selection = (manifest_path.parent / manifest["selection"]["path"]).resolve()
    if not selection.is_relative_to(manifest_path.parent):
        raise RefreshPredictionError("Selected source manifest escapes release root")
    _verify_file(selection, manifest["selection"])
    sources = SelectedSourceInputs.from_manifest(selection, processed_data_dir=processed)
    evidence["original_selection"] = _pin(selection)
    if output.is_file() and not force_refresh:
        saved = json.loads(output.read_bytes())
        if (saved.get("schema") == SCHEMA and saved.get("status") == "success"
                and saved.get("binding") == _binding(evidence)):
            for item in saved.get("saved_input_pins", []):
                _verify_file((state / item["path"]).resolve(), item)
            _empty_observation_inventory(observation_roots, state)
            # Rehydrate location metadata without regenerating observations or
            # probabilities; the original decision time and binding persist.
            document = {**saved, **evidence}
            if document != saved:
                _save_json(output, document)
            return {**document, "predictions_path": str(output)}
    intent_path = state / "prediction-intent.json"
    if intent_path.exists():
        intent = json.loads(intent_path.read_bytes())
        if (force_refresh or intent.get("binding") != _binding(evidence)
                or intent.get("max_events") != max_events or intent.get("horizon_days") != horizon_days):
            raise RefreshPredictionError("Original prediction job cannot be replaced; use a new job namespace")
        instant = _decision_time(intent["decision_at"])
    else:
        _save_json(intent_path, {"schema": SCHEMA, "binding": _binding(evidence),
                                "decision_at": instant.isoformat(), "max_events": max_events,
                                "horizon_days": horizon_days})
    if response_cache is None:
        from src.data.incremental_refresh import ResponseCache
        original_intent = json.loads(intent_path.read_bytes())
        cache_path = (state / original_intent["cache_directory"]).resolve() if original_intent.get("cache_directory") else Path(
            snapshot.get("cache_dir") or state / "cache").resolve()
        response_cache = ResponseCache(cache_path)
    original_intent = json.loads(intent_path.read_bytes())
    cache_relative = os.path.relpath(Path(response_cache.cache_dir).resolve(), state).replace("\\", "/")
    if original_intent.get("cache_directory") not in (None, cache_relative):
        raise RefreshPredictionError("Original prediction response cache cannot be replaced")
    if not original_intent.get("cache_directory"):
        _save_json(intent_path, {**original_intent, "cache_directory": cache_relative})
    if type(max_events) is not int or not 1 <= max_events <= 12:
        raise RefreshPredictionError("Upcoming card limit must be between one and twelve")
    if type(horizon_days) is not int or not 1 <= horizon_days <= 90:
        raise RefreshPredictionError("Prediction horizon must be between one and ninety days")
    events = parse_upcoming_events(response_cache.get_text(UPCOMING_URL))
    # UFC's official local event date is separate from its UTC broadcast time.
    from zoneinfo import ZoneInfo
    today = instant.astimezone(ZoneInfo("America/New_York")).date()
    import pandas as pd
    dates = pd.to_datetime(frames["fights_cleaned.csv"]["event_date"], errors="coerce")
    if dates.isna().any() or dates.empty:
        raise RefreshPredictionError("Accepted completed history has an unresolved date")
    completed_through = dates.max().date()
    eligible = [row for row in events
                if today <= _date(row["event_date"]) <= today + timedelta(days=horizon_days)
                and _date(row["event_date"]) > completed_through]
    selected_events = eligible[:max_events]
    cards = []
    observations = {}
    source_urls = [UPCOMING_URL]
    for event in selected_events:
        fights = parse_event_card(response_cache.get_text(event["url"]), event)
        source_urls.append(event["url"])
        cards.extend(fights)
        for fight in fights:
            for side in ("a", "b"):
                url = fight[f"fighter_{side}_url"]
                if url not in observations:
                    observations[url] = parse_fighter_profile(response_cache.get_text(url), url)
                    source_urls.append(url)
    profile_path = state / "inputs" / "current_profiles.csv"
    profile_pin = _merged_profiles(sources, list(observations.values()), profile_path)
    alias_pin = _observed_aliases(sources, cards, observations, response_cache,
                                  state / "inputs" / "observed_identity_aliases.json")
    # Retained professional/amateur history and owner decisions remain exact.
    # Current odds are deliberately absent unless an observed live input exists.
    sources = replace(sources, files={**sources.files, "profiles": profile_pin,
                                     "identity_aliases": alias_pin},
                      live_observation_paths=observation_roots)
    spec = NamedModelTrainingSpec.from_json(json.dumps(models["primary"]["training_spec"]))
    if spec.feature_cols != models["primary"]["feature_cols"]:
        raise RefreshPredictionError("Inference spec decoding changed the accepted feature order")
    responses, response_pins = [], []
    for url in source_urls:
        record, pin = _response_evidence(response_cache, url, state)
        responses.append(record)
        response_pins.append(pin)
    _empty_observation_inventory(observation_roots, state)
    calculation = _json_value({"schema": SCHEMA, "binding": _binding(evidence),
                   "decision_at": instant.isoformat(), "cards": cards,
                   "live_observation_inventory": observation_inventory,
                   "original_selection": {key: evidence["original_selection"][key]
                                          for key in ("bytes", "sha256")},
                   "selected_input_pins": [_portable_pin(pin, state) for pin in
                                           (profile_pin, alias_pin, *response_pins)]})
    calculation_path = state / "calculation-intent.json"
    if calculation_path.exists():
        if json.loads(calculation_path.read_bytes()) != calculation:
            raise RefreshPredictionError("Original bout calculation inputs cannot be replaced")
    else:
        _save_json(calculation_path, calculation)
    calculation_digest = _document_digest(calculation)
    rows, bout_pins = [], []
    # One selected context verifies every original input before processing the
    # complete card; readers share its cache and retain the original history.
    with sources.activate():
        for index, fight in enumerate(cards):
            _empty_observation_inventory(observation_roots, state)
            # Exact retained card routes support the submitted source names.
            for side in ("a", "b"):
                sources.validate_fighter(fight[f"fighter_{side}"], fight[f"fighter_{side}_id"])
            bout_path = state / "bouts" / (
                f"{index:03d}-{fight['event_id']}-{fight['fighter_a_id']}-{fight['fighter_b_id']}.json")
            if bout_path.exists():
                row = _completed_bout(json.loads(bout_path.read_bytes()), fight=fight,
                    calculation=calculation, calculation_digest=calculation_digest,
                    feature_cols=spec.feature_cols)
                _empty_observation_inventory(observation_roots, state)
                rows.append(row)
                bout_pins.append(_pin(bout_path))
                continue
            features, provenance = build_fight_features(
                fighter_a=fight["fighter_a"], fighter_b=fight["fighter_b"],
                fighter_a_id=fight["fighter_a_id"], fighter_b_id=fight["fighter_b_id"],
                event_date=fight["event_date"], event_id=fight["event_id"],
                weight_class=fight["weight_class"], is_title_bout=fight["is_title_bout"],
                num_rounds=fight["num_rounds"], is_empty_arena=fight["is_empty_arena"],
                training_spec=spec, include_provenance=True, processed_data_dir=processed,
                profile_integrity_decisions_path=sources.path("profile_decisions"),
                prior_profile_integrity_decisions_path=sources.path("prior_profile_decisions"),
            )
            _empty_observation_inventory(observation_roots, state)
            prediction = predict_fight(features, model_result=models["primary"])
            independent = predict_fight(features, model_result=models["no_odds"])
            for result in (prediction, independent):
                a, b = result["prob_a"], result["prob_b"]
                if not (math.isfinite(a) and math.isfinite(b) and 0 <= a <= 1
                        and 0 <= b <= 1 and math.isclose(a + b, 1.0, abs_tol=1e-12)):
                    raise RefreshPredictionError("Model returned invalid win probabilities")
            row = _json_value({**fight, **prediction, "no_odds_prob_a": independent["prob_a"],
                         "no_odds_prob_b": independent["prob_b"],
                         "predicted_winner_name": fight["fighter_" + prediction["predicted_winner"]],
                         "missing_feature_count": sum(_json_value(value) is None for value in features.values()),
                         "market_inputs_available": False, "model_features": _json_value(features),
                         "feature_provenance": _json_value(provenance)})
            record = {"schema": SCHEMA, "status": "completed", "card": _json_value(fight),
                      "calculation_sha256": calculation_digest, "binding": calculation["binding"],
                      "decision_at": calculation["decision_at"], "row": row,
                      "row_sha256": _document_digest(row)}
            _completed_bout(record, fight=fight, calculation=calculation,
                            calculation_digest=calculation_digest, feature_cols=spec.feature_cols)
            _empty_observation_inventory(observation_roots, state)
            _save_json(bout_path, record)
            _empty_observation_inventory(observation_roots, state)
            rows.append(row)
            bout_pins.append(_pin(bout_path))
    sources.verify()
    # Recheck pins before replacing a previously complete report.
    for role, pin in model_pins.items():
        _verify_file(Path(pin["path"]), pin)
    for name, pin in data_pins.items():
        _verify_file(processed / name, pin)
    _empty_observation_inventory(observation_roots, state)
    document = {"schema": SCHEMA, "status": "success", **evidence,
                "binding": _binding(evidence),
                "decision_at": instant.isoformat(), "model_contract": MODEL_CONTRACT,
                "model_training_performed": False, "betting_performed": False,
                "prediction_count": len(rows), "predictions": rows,
                "live_observation_inventory": observation_inventory,
                "coverage": {"eligible_events": len(eligible), "selected_events": len(selected_events),
                             "accepted_completed_through": completed_through.isoformat(),
                             "horizon_days": horizon_days, "max_events": max_events,
                             "bounded": len(eligible) > len(selected_events)},
                "saved_input_pins": [_portable_pin(pin, state) for pin in
                                     (profile_pin, alias_pin, *response_pins,
                                      _pin(calculation_path), *bout_pins)],
                "completed_bouts": [_portable_pin(pin, state) for pin in bout_pins],
                "source_responses": responses}
    _empty_observation_inventory(observation_roots, state)
    _save_json(output, document)
    logger.info("Saved %d predictions using unchanged %s models, accepted generation %s",
                len(rows), MODEL_CONTRACT, generation)
    return {**_json_value(document), "predictions_path": str(output)}
