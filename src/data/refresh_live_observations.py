"""Durable current official-card observations for the orders-disabled consumer."""
from __future__ import annotations

import csv
from datetime import date, datetime, timezone
import json
from pathlib import Path

from src.data.incremental_refresh import ResponseCache, _discover_events
from src.data.name_utils import normalize_person_name
from src.data.refresh_predictions import UPCOMING_URL, parse_upcoming_events, parse_event_card


def _identity(row):
    return (row["event_date"], tuple(sorted(
        (str(row[f"fighter_{side}_id"]), normalize_person_name(row[f"fighter_{side}"]))
        for side in ("a", "b"))))


def observe_publication(consumer, observation_dir, *, now=None, fetcher=None):
    """Reuse saved responses within a five-minute observation, never producer state.

    Every actual GET retains the existing request intent/response accounting.
    Unknown requests remain unknown. A later interval is a distinct observation.
    """
    from src.bot import _runtime_bundle_freshness_assessment
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Official observation requires an aware instant")
    namespace = Path(observation_dir) / str(int(now.timestamp()) // 300)
    cache = ResponseCache(namespace, fetcher=fetcher)
    events = parse_upcoming_events(cache.get_text(UPCOMING_URL))
    future = [row for row in events if date.fromisoformat(row["event_date"]) >= now.date()]
    if not future:
        raise ValueError("Current official index has no upcoming card")
    event = future[0]
    card = parse_event_card(cache.get_text(event["url"]), event)
    saved = {_identity(row): row for row in consumer.rows}
    current = {_identity(row): row for row in card}
    changed = set(saved) != set(current)
    for key in saved.keys() & current.keys():
        for field in ("weight_class", "num_rounds", "is_title_bout", "is_empty_arena"):
            if saved[key].get(field) != current[key].get(field):
                changed = True
    with Path(consumer.snapshot["fights_path"]).open(encoding="utf-8-sig", newline="") as stream:
        cutoff = max(date.fromisoformat(row["event_date"][:10]) for row in csv.DictReader(stream))
    new_completed = _discover_events(cache, cutoff, now)
    # Include only observed completed dates. The final CSV cutoff supplies the
    # processed age, never the earlier baseline manifest's cutoff.
    reference = date.fromisoformat(event["event_date"])
    primary = consumer.models["primary"]
    built_at = primary.get("built_at") or primary.get("trained_at") or primary.get("training_date")
    if not built_at:
        staging = Path(consumer.service.baseline) / "staging_manifest.json"
        if staging.exists():
            built_at = json.loads(staging.read_bytes()).get("built_at")
    summary = {"built_at": built_at, "processed_snapshot_max_event_date": cutoff.isoformat()}
    strict, advisory = _runtime_bundle_freshness_assessment(
        summary, reference_date=reference,
        completed_event_dates={date.fromisoformat(row["date"]) for row in new_completed})
    reasons = (["current_official_card_changed"] if changed else [])
    if strict:
        reasons.append("final_publication_freshness_blocked")
    records = [cache.record(url) for url in (UPCOMING_URL, event["url"])]
    return {"official_card_observed": True, "official_card_matches_publication": not changed,
            "event_date": event["event_date"], "event_id": event["url"].rsplit("/", 1)[1],
            "official_bout_count": len(card), "processed_snapshot_max_event_date": cutoff.isoformat(),
            "completed_events_missing": new_completed, "strict_messages": strict,
            "advisory_messages": advisory, "model_built_at": str(built_at) if built_at else None,
            "observation_namespace": namespace.name,
            "observed_at_utc": max(row["received_at"] for row in records),
            "trade_block_reasons": reasons}
