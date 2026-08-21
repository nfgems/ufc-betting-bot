"""Portable, durable state for the Sunday current-card collector.

This module records source evidence, normalized upcoming targets, card churn,
and the stable-ID queue consumed by the pre-UFC career scraper.  It has no
training, model, method-odds, post-event, or historical-recovery dependency.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd

from src.config import RAW_DATA_DIR
from src.data.io_utils import write_csv_atomically, write_json_atomically
from src.data.live_monitor import event_identity_key
from src.data.name_utils import normalize_cross_source_name, normalize_ufcstats_id

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
FORWARD_COLLECTION_SCHEMA_VERSION = SCHEMA_VERSION
DEFAULT_FORWARD_COLLECTION_ROOT = RAW_DATA_DIR / "practical_forward_collection_v1"

TARGET_COLUMNS = (
    "fight_key",
    "pair_key",
    "event_key",
    "event_id",
    "event_title",
    "event_url",
    "event_date",
    "scheduled_start_utc",
    "fighter_a",
    "fighter_a_id",
    "fighter_a_athlete_url",
    "fighter_b",
    "fighter_b_id",
    "fighter_b_athlete_url",
    "weight_class",
    "identity_status",
    "status",
    "observed_at_utc",
)

CANDIDATE_COLUMNS = (
    "fighter_id",
    "fighter_name",
    "first_ufc_date",
    "aliases",
    "reason",
    "event_keys_json",
    "first_seen_at_utc",
    "last_seen_at_utc",
    "input_fingerprint",
)


def _present(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "nat", "<na>"} else text


def stable_json_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def append_jsonl_unique(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    id_field: str,
) -> int:
    """Append content facts once, using the supplied stable identifier."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if target.exists():
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    existing.add(_present(json.loads(line).get(id_field)))

    appended = 0
    with target.open("a", encoding="utf-8", newline="") as handle:
        for source in rows:
            row = dict(source)
            row_id = _present(row.get(id_field))
            if not row_id or row_id in existing:
                continue
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            existing.add(row_id)
            appended += 1
    return appended


def _utc(value: object = None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        parsed = value
    else:
        text = _present(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def utc_text(value: object = None) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _date_text(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _scheduled_text(value: object) -> str:
    if not _present(value):
        return ""
    try:
        return utc_text(value)
    except (TypeError, ValueError):
        return ""


def _event_metadata(event: Mapping[str, object]) -> dict[str, str]:
    title = _present(event.get("title") or event.get("event_title") or event.get("event"))
    date = _date_text(event.get("date") or event.get("event_date"))
    url = _present(event.get("url") or event.get("event_url"))
    event_id = _present(event.get("event_id"))
    key = _present(event.get("event_key")) or event_identity_key(
        {"title": title, "date": date, "url": url, "event_id": event_id}
    )
    return {
        "event_key": key,
        "event_id": event_id,
        "event_title": title,
        "event_url": url,
        "event_date": date,
        "scheduled_start_utc": _scheduled_text(event.get("commence_time")),
    }


def _resolve_fighter_id(
    name: str,
    supplied: object,
    athlete_url: str,
    resolver: Callable[[str, str], object] | None,
) -> str:
    fighter_id = normalize_ufcstats_id(supplied) or ""
    if not fighter_id and resolver is not None:
        try:
            fighter_id = normalize_ufcstats_id(resolver(name, athlete_url)) or ""
        except Exception as exc:  # resolution failure remains visible downstream
            logger.warning("Stable UFCStats ID resolution failed for %s: %s", name, exc)
    return fighter_id


def build_upcoming_targets(
    event_cards: Sequence[Mapping[str, object]],
    *,
    observed_at: object = None,
    fighter_id_resolver: Callable[[str, str], object] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Normalize upcoming fights while refusing to guess stable identity."""
    observed_at_utc = utc_text(observed_at)
    targets: list[dict[str, str]] = []
    issues: list[dict[str, str]] = []
    seen: set[str] = set()

    for card in event_cards:
        if not isinstance(card, Mapping):
            continue
        nested_event = card.get("event")
        event = nested_event if isinstance(nested_event, Mapping) else card
        meta = _event_metadata(event)
        if not meta["event_key"] or (
            meta["event_key"].startswith("title-date:")
            and (not meta["event_title"] or not meta["event_date"])
        ):
            issues.append({"code": "event_identity_unresolved", "event_key": ""})
            continue

        fights = card.get("fights", [])
        if not isinstance(fights, Sequence) or isinstance(fights, (str, bytes)):
            fights = []
        for fight in fights:
            if not isinstance(fight, Mapping):
                continue
            fighter_a = _present(fight.get("fighter_a"))
            fighter_b = _present(fight.get("fighter_b"))
            if not fighter_a or not fighter_b:
                issues.append(
                    {
                        "code": "missing_fighter_name",
                        "event_key": meta["event_key"],
                        "fighter_a": fighter_a,
                        "fighter_b": fighter_b,
                    }
                )
                continue

            url_a = _present(fight.get("fighter_a_athlete_url"))
            url_b = _present(fight.get("fighter_b_athlete_url"))
            id_a = _resolve_fighter_id(
                fighter_a, fight.get("fighter_a_id"), url_a, fighter_id_resolver
            )
            id_b = _resolve_fighter_id(
                fighter_b, fight.get("fighter_b_id"), url_b, fighter_id_resolver
            )
            same_id_collision = bool(id_a and id_b and id_a == id_b)
            if same_id_collision:
                issues.append(
                    {
                        "code": "same_stable_id_opponents",
                        "event_key": meta["event_key"],
                        "fighter_a": fighter_a,
                        "fighter_b": fighter_b,
                    }
                )
                # A duplicated opponent identity is source ambiguity, not a
                # usable binding for either side of this fight.
                id_a = ""
                id_b = ""
            resolved = bool(id_a and id_b and id_a != id_b)
            if resolved:
                ordered = sorted((id_a, id_b))
                pair_key = "ufcstats:" + "|".join(ordered)
                fight_key = f"{meta['event_key']}|fighters:{ordered[0]}:{ordered[1]}"
            else:
                pair_key = "unresolved:" + "|".join(
                    sorted(
                        (
                            normalize_cross_source_name(fighter_a),
                            normalize_cross_source_name(fighter_b),
                        )
                    )
                )
                fight_key = stable_json_hash(
                    {"schema_version": SCHEMA_VERSION, "event_key": meta["event_key"], "pair_key": pair_key}
                )
                issues.append(
                    {
                        "code": "fighter_identity_unresolved",
                        "event_key": meta["event_key"],
                        "fighter_a": fighter_a,
                        "fighter_b": fighter_b,
                    }
                )

            if fight_key in seen:
                issues.append(
                    {
                        "code": "duplicate_upcoming_fight",
                        "event_key": meta["event_key"],
                        "fighter_a": fighter_a,
                        "fighter_b": fighter_b,
                    }
                )
                continue
            seen.add(fight_key)
            targets.append(
                {
                    "fight_key": fight_key,
                    "pair_key": pair_key,
                    **meta,
                    "scheduled_start_utc": _scheduled_text(
                        fight.get("commence_time") or meta["scheduled_start_utc"]
                    ),
                    "fighter_a": fighter_a,
                    "fighter_a_id": id_a,
                    "fighter_a_athlete_url": url_a,
                    "fighter_b": fighter_b,
                    "fighter_b_id": id_b,
                    "fighter_b_athlete_url": url_b,
                    "weight_class": _present(fight.get("weight_class")),
                    "identity_status": "resolved" if resolved else "unresolved",
                    "status": "active",
                    "observed_at_utc": observed_at_utc,
                }
            )

    targets.sort(key=lambda row: (row["event_date"], row["event_key"], row["fight_key"]))
    return targets, issues


def enrich_event_cards_with_schedule(
    event_cards: Sequence[Mapping[str, object]],
    *,
    context_enricher: Callable[[list[dict]], list[dict]],
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Attach optional market start times without changing card identity."""
    cards = copy.deepcopy(list(event_cards))
    contexts: list[dict[str, object]] = []
    destinations: list[dict[str, object]] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        event_value = card.get("event")
        event = event_value if isinstance(event_value, Mapping) else card
        fights = card.get("fights")
        if not isinstance(fights, list):
            continue
        for fight in fights:
            if not isinstance(fight, dict):
                continue
            contexts.append(
                {
                    "event_title": event.get("title") or event.get("event_title") or "",
                    "event_date": event.get("date") or event.get("event_date") or "",
                    "fighter_a": fight.get("fighter_a") or "",
                    "fighter_b": fight.get("fighter_b") or "",
                    "fighter_a_id": fight.get("fighter_a_id"),
                    "fighter_b_id": fight.get("fighter_b_id"),
                    "commence_time": fight.get("commence_time") or event.get("commence_time") or "",
                }
            )
            destinations.append(fight)

    if not contexts:
        return cards, []
    try:
        enriched = context_enricher(contexts)
    except Exception as exc:
        logger.warning("Upcoming schedule enrichment failed: %s", exc)
        return cards, [{"code": "scheduled_start_enrichment_failed"}]
    if len(enriched) != len(destinations):
        return cards, [{"code": "scheduled_start_enrichment_shape_mismatch"}]

    issues: list[dict[str, str]] = []
    for fight, context in zip(destinations, enriched):
        scheduled = _scheduled_text(context.get("commence_time"))
        if scheduled:
            fight["commence_time"] = scheduled
        else:
            issues.append(
                {
                    "code": "scheduled_start_unavailable",
                    "fighter_a": _present(fight.get("fighter_a")),
                    "fighter_b": _present(fight.get("fighter_b")),
                }
            )
    return cards, issues


def _change(
    kind: str,
    before: Mapping[str, object] | None,
    after: Mapping[str, object] | None,
    observed_at_utc: str,
) -> dict[str, object]:
    old = dict(before or {})
    new = dict(after or {})
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "change_type": kind,
        "observed_at_utc": observed_at_utc,
        "fight_key_before": _present(old.get("fight_key")),
        "fight_key_after": _present(new.get("fight_key")),
        "pair_key": _present(new.get("pair_key") or old.get("pair_key")),
        "event_key_before": _present(old.get("event_key")),
        "event_key_after": _present(new.get("event_key")),
        "event_date_before": _present(old.get("event_date")),
        "event_date_after": _present(new.get("event_date")),
        "scheduled_start_utc_before": _present(old.get("scheduled_start_utc")),
        "scheduled_start_utc_after": _present(new.get("scheduled_start_utc")),
    }
    for side in ("a", "b"):
        payload[f"fighter_{side}_before"] = _present(old.get(f"fighter_{side}"))
        payload[f"fighter_{side}_id_before"] = _present(old.get(f"fighter_{side}_id"))
        payload[f"fighter_{side}_after"] = _present(new.get(f"fighter_{side}"))
        payload[f"fighter_{side}_id_after"] = _present(new.get(f"fighter_{side}_id"))
    payload["change_id"] = stable_json_hash(
        {key: value for key, value in payload.items() if key != "observed_at_utc"}
    )
    return payload


def diff_upcoming_targets(
    previous_targets: Sequence[Mapping[str, object]],
    current_targets: Sequence[Mapping[str, object]],
    *,
    scan_complete: bool,
    observed_at: object = None,
) -> tuple[list[dict[str, object]], list[str]]:
    """Describe card churn; incomplete scans can never create removals."""
    stamp = utc_text(observed_at)
    old_by_key = {_present(row.get("fight_key")): row for row in previous_targets}
    new_by_key = {_present(row.get("fight_key")): row for row in current_targets}
    old_by_pair: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    new_by_pair: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in previous_targets:
        old_by_pair[_present(row.get("pair_key"))].append(row)
    for row in current_targets:
        new_by_pair[_present(row.get("pair_key"))].append(row)

    changes: list[dict[str, object]] = []
    matched_old: set[str] = set()
    matched_new: set[str] = set()
    for key in sorted(set(old_by_key).intersection(new_by_key)):
        old, new = old_by_key[key], new_by_key[key]
        matched_old.add(key)
        matched_new.add(key)
        if (
            _present(old.get("fighter_a_id")) == _present(new.get("fighter_b_id"))
            and _present(old.get("fighter_b_id")) == _present(new.get("fighter_a_id"))
            and _present(old.get("fighter_a_id"))
        ):
            changes.append(_change("a_b_reversal", old, new, stamp))
        if (
            _present(old.get("event_date")) != _present(new.get("event_date"))
            or _present(old.get("scheduled_start_utc"))
            != _present(new.get("scheduled_start_utc"))
        ):
            changes.append(_change("postponed_or_rescheduled", old, new, stamp))

    for pair_key in sorted(set(old_by_pair).intersection(new_by_pair)):
        old_rows = [row for row in old_by_pair[pair_key] if _present(row.get("fight_key")) not in matched_old]
        new_rows = [row for row in new_by_pair[pair_key] if _present(row.get("fight_key")) not in matched_new]
        if len(old_rows) == len(new_rows) == 1:
            old, new = old_rows[0], new_rows[0]
            matched_old.add(_present(old.get("fight_key")))
            matched_new.add(_present(new.get("fight_key")))
            changes.append(_change("postponed_or_rescheduled", old, new, stamp))

    removed = [row for key, row in old_by_key.items() if key not in matched_old and key not in new_by_key]
    added = [row for key, row in new_by_key.items() if key not in matched_new and key not in old_by_key]
    used_removed: set[str] = set()
    used_added: set[str] = set()
    for old in removed:
        old_ids = {_present(old.get("fighter_a_id")), _present(old.get("fighter_b_id"))} - {""}
        candidates = []
        for new in added:
            if _present(new.get("fight_key")) in used_added:
                continue
            new_ids = {_present(new.get("fighter_a_id")), _present(new.get("fighter_b_id"))} - {""}
            if old.get("event_key") == new.get("event_key") and len(old_ids & new_ids) == 1:
                candidates.append(new)
        if len(candidates) == 1:
            new = candidates[0]
            used_removed.add(_present(old.get("fight_key")))
            used_added.add(_present(new.get("fight_key")))
            changes.append(_change("late_replacement", old, new, stamp))

    for row in added:
        if _present(row.get("fight_key")) not in used_added:
            changes.append(_change("fight_added", None, row, stamp))
    remaining = [row for row in removed if _present(row.get("fight_key")) not in used_removed]
    warnings: list[str] = []
    if scan_complete:
        changes.extend(_change("fight_cancelled_or_removed", row, None, stamp) for row in remaining)
    elif remaining:
        warnings.append("incomplete_scan_removals_suppressed")

    unique = {_present(row["change_id"]): row for row in changes}
    return [unique[key] for key in sorted(unique)], warnings


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return []


def _create_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError(f"immutable_forward_evidence_collision:{path}")


def _persist_source_responses(
    root: Path,
    responses: Sequence[Mapping[str, object]],
    *,
    fallback_time: str,
) -> list[dict[str, object]]:
    facts: list[dict[str, object]] = []
    for response in responses:
        url = _present(response.get("url") or response.get("request_url"))
        body = response.get("body", response.get("html"))
        raw = body if isinstance(body, bytes) else str(body or "").encode("utf-8")
        if not url or not raw:
            continue
        digest = hashlib.sha256(raw).hexdigest()
        raw_path = root / "raw" / "upcoming" / f"{digest}.html"
        _create_immutable(raw_path, raw)
        identity = {
            "schema_version": SCHEMA_VERSION,
            "request_url": url,
            "retrieval_time_utc": utc_text(response.get("retrieval_time_utc") or fallback_time),
            "raw_sha256": digest,
        }
        facts.append(
            {
                **identity,
                "response_id": stable_json_hash(identity),
                "raw_path": raw_path.as_posix(),
                "byte_count": len(raw),
            }
        )
    append_jsonl_unique(root / "upcoming-response-ledger.jsonl", facts, id_field="response_id")
    return facts


def persist_upcoming_discovery(
    *,
    event_cards: Sequence[Mapping[str, object]],
    scan_complete: bool,
    storage_root: Path = DEFAULT_FORWARD_COLLECTION_ROOT,
    observed_at: object = None,
    fighter_id_resolver: Callable[[str, str], object] | None = None,
    source_responses: Sequence[Mapping[str, object]] = (),
    require_raw_evidence: bool = False,
) -> dict[str, object]:
    """Record an attempt and advance the complete baseline only when healthy."""
    root = Path(storage_root)
    stamp = utc_text(observed_at)
    targets, issues = build_upcoming_targets(
        event_cards,
        observed_at=stamp,
        fighter_id_resolver=fighter_id_resolver,
    )
    raw_facts = _persist_source_responses(root, source_responses, fallback_time=stamp)
    publication_complete = bool(
        scan_complete
        and targets
        and not issues
        and (raw_facts or not require_raw_evidence)
    )
    complete_path = root / "latest-complete-upcoming-targets.csv"
    observed_path = root / "latest-observed-upcoming-targets.csv"
    previous = _read_csv_rows(complete_path)
    changes, diff_warnings = diff_upcoming_targets(
        previous,
        targets,
        scan_complete=publication_complete,
        observed_at=stamp,
    )
    effective_by_key = {_present(row.get("fight_key")): dict(row) for row in previous}
    effective_by_key.update({_present(row.get("fight_key")): dict(row) for row in targets})
    effective = (
        list(targets)
        if publication_complete
        else [effective_by_key[key] for key in sorted(effective_by_key)]
    )

    snapshot_payload = {
        "schema_version": SCHEMA_VERSION,
        "observed_at_utc": stamp,
        "scan_complete": bool(scan_complete),
        "publication_complete": publication_complete,
        "event_cards": list(event_cards),
        "targets": targets,
        "issues": issues,
        "raw_source_responses": [
            {key: fact[key] for key in ("response_id", "request_url", "retrieval_time_utc", "raw_sha256", "raw_path")}
            for fact in raw_facts
        ],
    }
    snapshot_bytes = (
        json.dumps(snapshot_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    snapshot_digest = hashlib.sha256(snapshot_bytes).hexdigest()
    snapshot_path = root / "upcoming_snapshots" / f"{_utc(stamp):%Y%m%dT%H%M%SZ}_{snapshot_digest[:16]}.json"
    _create_immutable(
        snapshot_path,
        snapshot_bytes,
    )

    write_csv_atomically(pd.DataFrame(targets, columns=TARGET_COLUMNS), observed_path)
    if publication_complete:
        write_csv_atomically(pd.DataFrame(targets, columns=TARGET_COLUMNS), complete_path)
    observations: list[dict[str, object]] = []
    for target in targets:
        row: dict[str, object] = dict(target)
        row["target_observation_id"] = stable_json_hash(
            {"schema_version": SCHEMA_VERSION, "observed_at_utc": stamp, "target": target}
        )
        observations.append(row)
    append_jsonl_unique(root / "upcoming-target-ledger.jsonl", observations, id_field="target_observation_id")
    append_jsonl_unique(root / "card-change-ledger.jsonl", changes, id_field="change_id")

    warnings = list(diff_warnings)
    if not scan_complete:
        warnings.append("upcoming_scan_incomplete")
    if not targets:
        warnings.append("upcoming_scan_zero_fights")
    warnings.extend(sorted({_present(issue.get("code")) or "upcoming_normalization_issue" for issue in issues}))
    if require_raw_evidence and not raw_facts:
        warnings.append("upcoming_raw_evidence_missing")
    warnings = list(dict.fromkeys(warnings))
    run_id = stable_json_hash(
        {"schema_version": SCHEMA_VERSION, "observed_at_utc": stamp, "snapshot_sha256": snapshot_digest}
    )
    receipt = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "observed_at_utc": stamp,
        "status": "healthy" if not warnings else "incomplete",
        "scan_complete": bool(scan_complete),
        "publication_complete": publication_complete,
        "event_count": len(event_cards),
        "target_count": len(targets),
        "effective_target_count": len(effective),
        "resolved_target_count": sum(row["identity_status"] == "resolved" for row in targets),
        "issue_count": len(issues),
        "change_count": len(changes),
        "raw_source_response_count": len(raw_facts),
        "warning_codes": warnings,
        "snapshot_path": snapshot_path.as_posix(),
        "snapshot_sha256": snapshot_digest,
    }
    append_jsonl_unique(root / "discovery-attempt-ledger.jsonl", [receipt], id_field="run_id")
    write_json_atomically(receipt, root / "latest-discovery-receipt.json")
    return {**receipt, "targets": targets, "effective_targets": effective, "issues": issues, "changes": changes}


def load_completed_debut_boundaries(path: Path) -> dict[str, str]:
    target = Path(path)
    if not target.exists():
        return {}
    header = set(pd.read_csv(target, nrows=0).columns)
    if "event_date" not in header:
        return {}
    columns = [column for column in ("event_date", "fighter_a_id", "fighter_b_id") if column in header]
    if len(columns) == 1:
        return {}
    boundaries: dict[str, str] = {}
    for row in pd.read_csv(target, usecols=columns, dtype=object).to_dict("records"):
        date = _date_text(row.get("event_date"))
        for side in ("a", "b"):
            fighter_id = normalize_ufcstats_id(row.get(f"fighter_{side}_id")) or ""
            if fighter_id and date and (fighter_id not in boundaries or date < boundaries[fighter_id]):
                boundaries[fighter_id] = date
    return boundaries


def _roster_facts(path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    target = Path(path)
    if not target.exists():
        return {}, {}
    roster = pd.read_csv(target, dtype=object)
    boundaries: dict[str, str] = {}
    ids_by_name: dict[str, set[str]] = defaultdict(set)
    for row in roster.to_dict("records"):
        fighter_id = normalize_ufcstats_id(row.get("ufcstats_url")) or ""
        if not fighter_id:
            continue
        debut = _date_text(row.get("octagon_debut"))
        if debut and (fighter_id not in boundaries or debut < boundaries[fighter_id]):
            boundaries[fighter_id] = debut
        for field in ("official_name", "ufcstats_name", "profile_name", "slug_name"):
            name = _present(row.get(field))
            key = normalize_cross_source_name(name) if name else ""
            if key:
                ids_by_name[key].add(fighter_id)
    return boundaries, dict(ids_by_name)


def load_roster_debut_boundaries(path: Path) -> dict[str, str]:
    return _roster_facts(Path(path))[0]


def load_known_fighter_ids_by_name(
    completed_fights_path: Path,
    roster_path: Path,
) -> dict[str, set[str]]:
    """Collect existing name/ID bindings for conservative collision checks."""
    _, ids_by_name = _roster_facts(Path(roster_path))
    completed = Path(completed_fights_path)
    if not completed.exists():
        return ids_by_name
    header = set(pd.read_csv(completed, nrows=0).columns)
    columns = [
        column
        for column in (
            "fighter_a",
            "fighter_a_id",
            "fighter_b",
            "fighter_b_id",
        )
        if column in header
    ]
    if not columns:
        return ids_by_name
    for row in pd.read_csv(completed, usecols=columns, dtype=object).to_dict("records"):
        for side in ("a", "b"):
            name = _present(row.get(f"fighter_{side}"))
            fighter_id = normalize_ufcstats_id(row.get(f"fighter_{side}_id")) or ""
            if name and fighter_id:
                ids_by_name.setdefault(normalize_cross_source_name(name), set()).add(
                    fighter_id
                )
    return ids_by_name


def _decoded_list(value: object) -> list[str]:
    text = _present(value)
    if not text:
        return []
    try:
        decoded = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        decoded = None
    if isinstance(decoded, list):
        return [_present(item) for item in decoded if _present(item)]
    return [item.strip() for item in text.split("|") if item.strip()]


def build_pre_ufc_candidate_rows(
    targets: Sequence[Mapping[str, object]],
    *,
    completed_boundaries: Mapping[str, str] | None = None,
    roster_boundaries: Mapping[str, str] | None = None,
    changes: Sequence[Mapping[str, object]] = (),
    previous_candidates: Sequence[Mapping[str, object]] = (),
    known_ids_by_name: Mapping[str, set[str]] | None = None,
    observed_at: object = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build a stable-ID queue with an explicit strict pre-UFC boundary."""
    stamp = utc_text(observed_at)
    previous_by_id = {
        fighter_id: dict(row)
        for row in previous_candidates
        if (fighter_id := normalize_ufcstats_id(row.get("fighter_id")))
    }
    identities: dict[str, dict[str, object]] = {}
    ids_by_name: dict[str, set[str]] = defaultdict(set)
    for name, values in (known_ids_by_name or {}).items():
        ids_by_name[name].update(
            fighter_id for value in values if (fighter_id := normalize_ufcstats_id(value))
        )
    for fighter_id, prior in previous_by_id.items():
        names = set(_decoded_list(prior.get("aliases")))
        if _present(prior.get("fighter_name")):
            names.add(_present(prior.get("fighter_name")))
        identities[fighter_id] = {
            "names": names,
            "dates": {_date_text(prior.get("first_ufc_date"))} - {""},
            "events": set(_decoded_list(prior.get("event_keys_json"))),
            "reasons": set(_decoded_list(prior.get("reason"))),
            "current": False,
        }
        for name in names:
            ids_by_name[normalize_cross_source_name(name)].add(fighter_id)

    for target in targets:
        event_date = _date_text(target.get("event_date"))
        for side in ("a", "b"):
            fighter_id = normalize_ufcstats_id(target.get(f"fighter_{side}_id")) or ""
            name = _present(target.get(f"fighter_{side}"))
            if not fighter_id or not name:
                continue
            ids_by_name[normalize_cross_source_name(name)].add(fighter_id)
            identity = identities.setdefault(
                fighter_id,
                {"names": set(), "dates": set(), "events": set(), "reasons": set(), "current": False},
            )
            identity["names"].add(name)
            identity["current"] = True
            if event_date:
                identity["dates"].add(event_date)
            if _present(target.get("event_key")):
                identity["events"].add(_present(target.get("event_key")))

    reasons_by_id: dict[str, set[str]] = defaultdict(set)
    for change in changes:
        reason = _present(change.get("change_type")) or "upcoming_card"
        for side in ("a", "b"):
            fighter_id = normalize_ufcstats_id(change.get(f"fighter_{side}_id_after")) or ""
            if fighter_id:
                reasons_by_id[fighter_id].add(reason)

    ambiguous_ids = {
        fighter_id
        for name, fighter_ids in ids_by_name.items()
        if name and len(fighter_ids) > 1
        for fighter_id in fighter_ids
    }
    rows: list[dict[str, str]] = []
    issues: list[dict[str, str]] = []
    for fighter_id, identity in sorted(identities.items()):
        names = sorted(identity["names"])
        if not names:
            issues.append({"code": "missing_candidate_name", "fighter_id": fighter_id, "fighter_name": ""})
            continue
        if fighter_id in ambiguous_ids:
            issues.append(
                {"code": "ambiguous_same_name_stable_ids", "fighter_id": fighter_id, "fighter_name": names[0]}
            )
            if fighter_id in previous_by_id:
                rows.append({column: _present(previous_by_id[fighter_id].get(column)) for column in CANDIDATE_COLUMNS})
            continue
        dates = {
            _date_text((completed_boundaries or {}).get(fighter_id)),
            _date_text((roster_boundaries or {}).get(fighter_id)),
            *identity["dates"],
        } - {""}
        if not dates:
            issues.append(
                {"code": "missing_first_ufc_boundary", "fighter_id": fighter_id, "fighter_name": names[0]}
            )
            continue
        prior = previous_by_id.get(fighter_id, {})
        reasons = set(identity["reasons"]) | reasons_by_id[fighter_id]
        if not reasons:
            reasons.add("upcoming_card")
        boundary = min(dates)
        rows.append(
            {
                "fighter_id": fighter_id,
                "fighter_name": names[0],
                "first_ufc_date": boundary,
                "aliases": json.dumps(names, separators=(",", ":")),
                "reason": "|".join(sorted(reasons)),
                "event_keys_json": json.dumps(sorted(identity["events"]), separators=(",", ":")),
                "first_seen_at_utc": _present(prior.get("first_seen_at_utc")) or stamp,
                "last_seen_at_utc": stamp if identity["current"] else (_present(prior.get("last_seen_at_utc")) or stamp),
                "input_fingerprint": stable_json_hash({"fighter_id": fighter_id, "first_ufc_date": boundary}),
            }
        )
    return rows, issues


def publish_pre_ufc_candidate_queue(
    targets: Sequence[Mapping[str, object]],
    *,
    changes: Sequence[Mapping[str, object]] = (),
    completed_fights_path: Path,
    roster_path: Path,
    storage_root: Path = DEFAULT_FORWARD_COLLECTION_ROOT,
    observed_at: object = None,
) -> dict[str, object]:
    root = Path(storage_root)
    queue_path = root / "pre-ufc-candidate-queue.csv"
    roster_boundaries = load_roster_debut_boundaries(Path(roster_path))
    ids_by_name = load_known_fighter_ids_by_name(
        Path(completed_fights_path), Path(roster_path)
    )
    rows, issues = build_pre_ufc_candidate_rows(
        targets,
        completed_boundaries=load_completed_debut_boundaries(Path(completed_fights_path)),
        roster_boundaries=roster_boundaries,
        changes=changes,
        previous_candidates=_read_csv_rows(queue_path),
        known_ids_by_name=ids_by_name,
        observed_at=observed_at,
    )
    write_csv_atomically(pd.DataFrame(rows, columns=CANDIDATE_COLUMNS), queue_path)
    warnings = sorted({_present(issue.get("code")) for issue in issues})
    if not rows:
        warnings.append("pre_ufc_candidate_queue_empty")
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_text(observed_at),
        "status": "incomplete" if warnings else "complete",
        "candidate_count": len(rows),
        "issue_count": len(issues),
        "warning_codes": list(dict.fromkeys(warnings)),
        "queue_path": queue_path.as_posix(),
        "issues": issues,
    }
    write_json_atomically(report, root / "pre-ufc-candidate-report.json")
    return {**report, "rows": rows}


def collection_health(reports: Iterable[Mapping[str, object]]) -> dict[str, object]:
    warnings: list[str] = []
    failed: list[str] = []
    incomplete: list[str] = []
    for report in reports:
        stage = _present(report.get("stage") or report.get("task")) or "unknown"
        status = _present(report.get("status") or report.get("health_state")) or "healthy"
        warnings.extend(_present(value) for value in (report.get("warning_codes") or []) if _present(value))
        if status in {"failed", "error"}:
            failed.append(stage)
        elif status not in {"healthy", "success", "complete", "skipped"}:
            incomplete.append(stage)
    return {
        "health_state": "failed" if failed else "incomplete" if incomplete or warnings else "healthy",
        "failed_stages": failed,
        "incomplete_stages": incomplete,
        "warning_codes": list(dict.fromkeys(warnings)),
    }
