"""Ordinary collector connections to independently pinned offline-reviewed inputs.

Provisioning a future collection decision remains separate. A missing or rotated
pin refuses before transport; callers cannot manufacture identity from discovery.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import re

from src.data.method_observation_history import ObservationRefusal
from src.data.observation_consumer_references import _pin


CONTEXT_PREFIXES = {"method": "METHOD_ODDS", "rankings": "RANKINGS"}


class ObservationContextError(ObservationRefusal):
    """An explicit context prerequisite failed before source acquisition."""

    def __init__(self, source, reason_code, message):
        super().__init__(message)
        self.source = source
        self.reason_code = reason_code


def validate_observation_context(source, path, expected_sha256):
    """Validate a caller-supplied independent pin; never mint trust from a file."""
    if source not in CONTEXT_PREFIXES:
        raise ObservationContextError(source, "unsupported_source", "unsupported observation source")
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ObservationContextError(source, "not_configured", "independently pinned observation context is not configured")
    try:
        pin = _pin(Path(path))
    except (OSError, ValueError, TypeError) as exc:
        raise ObservationContextError(source, "context_unreadable", "independently pinned observation context cannot be read safely") from exc
    if pin["sha256"] != expected_sha256:
        raise ObservationContextError(source, "pin_mismatch", "independent observation context differs from configured pin")
    try:
        if source == "method":
            from src.data.method_observation_history import load_reviewed_bindings
            load_reviewed_bindings(pin)
        else:
            _rankings_provider(pin).load_reviewed_registry(pin)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ObservationContextError(source, "context_invalid", "independently pinned observation context failed source-specific validation") from exc
    return pin


def _rankings_provider(context):
    """Dispatch only from the exact independently pinned source registry."""
    from src.data.method_observation_history import _read_pin
    from src.data import rankings_observation_history, rosterwatch_rankings
    kind = json.loads(_read_pin(context)).get("kind")
    if kind == rosterwatch_rankings.REGISTRY_KIND:
        return rosterwatch_rankings
    if kind == rankings_observation_history.REGISTRY_KIND:
        return rankings_observation_history
    raise ObservationContextError("rankings", "unsupported_source", "unknown reviewed rankings provider")


def configured_observation_context(source):
    from src import config

    if source not in CONTEXT_PREFIXES:
        raise ObservationContextError(source, "unsupported_source", "unsupported observation source")
    prefix = CONTEXT_PREFIXES[source]
    return validate_observation_context(source,
        getattr(config, prefix + "_REVIEWED_BINDINGS_PATH"),
        getattr(config, prefix + "_REVIEWED_BINDINGS_SHA256"))


def observation_context_preflight():
    """Read-only readiness report, with no source, claim, or snapshot writes."""
    results = {}
    for source, prefix in CONTEXT_PREFIXES.items():
        try:
            pin = configured_observation_context(source)
            results[source] = {"status": "ready", "context": pin}
        except ObservationContextError as exc:
            results[source] = {"status": "blocked", "reason_code": exc.reason_code,
                "error": str(exc), "required_settings": [prefix + "_REVIEWED_BINDINGS_PATH", prefix + "_REVIEWED_BINDINGS_SHA256"]}
    return {"ready": all(value["status"] == "ready" for value in results.values()), "sources": results}


def observation_collection_preflight(*, inventory_path, inventory_sha256,
                                     observed_at, not_after, now=None):
    """Check exact saved inventory, timing and unused claims without acquisition.

    The caller must independently review the inventory's provenance and freshness
    bound. This diagnostic cannot grant source authority or lock a future claim.
    Snapshot roots match the ordinary collectors, including raw/rankings.
    """
    from src import config
    from src.data.method_observation_history import _read_pin, _safe_path, _time
    from src.data.rankings_observation_history import CAPTURE_SUBDIR

    report = observation_context_preflight()
    report.update(ready=False, source_authority_created=False,
                  collection_claimed=False, inventory_errors=[])
    current = now if now is not None else datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("Preflight current time must have a timezone")
    errors = report["inventory_errors"]
    fights = None
    try:
        if not isinstance(inventory_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", inventory_sha256) is None:
            raise ValueError("independent inventory SHA256 is required")
        pin = _pin(Path(inventory_path))
        if pin["sha256"] != inventory_sha256:
            raise ValueError("tracked inventory differs from independently supplied pin")
        fights = json.loads(_read_pin(pin))
        if not isinstance(fights, list) or not fights or not all(isinstance(row, dict) for row in fights):
            raise ValueError("tracked inventory must be a nonempty list of fight objects")
        if not _time(observed_at) <= current < _time(not_after):
            raise ValueError("tracked inventory observation is future-dated or its reviewed window expired")
        report["inventory"] = {**pin, "observed_at": observed_at, "not_after": not_after,
                               "fight_count": len(fights)}
        for index, fight in enumerate(fights):
            missing = [key for key in ("event_id", "fighter_a", "fighter_b", "fighter_a_id", "fighter_b_id", "commence_time")
                       if not isinstance(fight.get(key), str) or not fight[key].strip()]
            if missing:
                errors.append({"fight_index": index, "reason": "missing_tracked_identity", "fields": missing})
            elif any(re.fullmatch(r"[0-9a-f]{16}", fight[key]) is None for key in ("fighter_a_id", "fighter_b_id")) or fight["fighter_a_id"] == fight["fighter_b_id"]:
                errors.append({"fight_index": index, "reason": "invalid_tracked_stable_identity"})
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append({"reason": "invalid_inventory", "error": str(exc)})
        fights = None

    for source, row in report["sources"].items():
        if row["status"] != "ready":
            continue
        try:
            context = json.loads(_read_pin(row["context"]))
            rosterwatch = source == "rankings" and context.get("source") == "roster.watch"
            if rosterwatch:
                from src.data.rosterwatch_rankings_collection import CAPTURE_SUBDIR as rosterwatch_subdir
            subdir = ("method_response_captures_v1" if source == "method" else
                      rosterwatch_subdir if rosterwatch else CAPTURE_SUBDIR)
            snapshot_dir = Path(config.RAW_DATA_DIR) / ("method_odds" if source == "method" else "rankings")
            claim = _safe_path(snapshot_dir / subdir / "collections" / hashlib.sha256(context["collection_id"].encode()).hexdigest())
            row["collection_claim_path"] = str(claim)
            if claim.exists():
                if rosterwatch:
                    from src.data.rosterwatch_rankings_collection import inspect_replay
                    from src.data.rankings_observation_history import HISTORY_SUBDIR
                    from src.data.rankings_scraper import RANKINGS_SNAPSHOT_MAX_AGE
                    from src.data.selected_source_inputs import selected_path
                    retained = selected_path("rankings_observations", Path(config.RAW_DATA_DIR) / "kaggle_rankings" / HISTORY_SUBDIR)
                    replay = inspect_replay(row["context"], snapshot_root=snapshot_dir, finalized_root=retained)
                    captured = _time(replay["captured_at"])
                    if captured > current:
                        raise ValueError("saved Roster Watch rankings receipt is newer than preflight time")
                    if current - captured > RANKINGS_SNAPSHOT_MAX_AGE:
                        raise ValueError("saved Roster Watch rankings exceed current-use age limit")
                    row.update(collection_id_unused_at_inspection=False, replay=replay)
                    continue
                raise ValueError("collection identifier already has a claim; a fresh arm requires a new reviewed identifier")
            existing_parent = None
            for parent in claim.parents:
                if parent.exists():
                    if not parent.is_dir():
                        raise ValueError("collection claim parent is not a directory")
                    if not os.access(parent, os.X_OK):
                        raise ValueError("collection claim parent cannot be traversed by the preflight user")
                    if existing_parent is None:
                        existing_parent = parent
            if existing_parent is None or not os.access(existing_parent, os.W_OK | os.X_OK):
                raise ValueError("collection claim parent cannot be written by the preflight user")
            row["collection_id_unused_at_inspection"] = True
            row["claim_parent_access_for_preflight_user"] = True
            if source == "method":
                bindings = context["bindings"]
                fields = ("event_id", "fighter_a", "fighter_b", "fighter_a_id", "fighter_b_id", "commence_time")
                wanted = [tuple(binding[field] for field in fields) for binding in bindings]
                actual = [tuple(fight.get(field) for field in fields) for fight in fights] if fights is not None else []
                if len(actual) != len(wanted) or any(actual.count(key) != 1 for key in wanted):
                    raise ValueError("reviewed/tracked method fight inventory differs")
                if any(not current < _time(binding["cutoff_at"]) < _time(binding["commence_time"]) for binding in bindings):
                    raise ValueError("method collection cutoff already reached")
                row["exact_inventory_and_prefight_cutoffs_valid"] = True
        except (OSError, ValueError, TypeError, KeyError) as exc:
            row.update(status="blocked", reason_code="collection_admission_refused", error=str(exc))
    report["ready"] = not errors and all(row["status"] == "ready" for row in report["sources"].values())
    report["checked_at"] = current.isoformat()
    report["qualification"] = "Point-in-time technical preflight for the current process user only; run as the application user. Independent source/inventory review and immediate runtime recheck are still required. No claim is reserved."
    return report


def observation_stage_error(source, error):
    """Preserve refusals separately from unavailable/empty successful results."""
    return {"stage": source + "_snapshot", "status": "blocked" if isinstance(error, ObservationRefusal) else "failed",
        "reason_code": getattr(error, "reason_code", "observation_refused" if isinstance(error, ObservationRefusal) else "collection_failed"),
        "error_type": type(error).__name__, "error": str(error)}


def observation_snapshot_error(source, snapshot):
    """A normal return with failed acquisition is not a clean empty observation."""
    status = snapshot.get("status")
    source_failed = snapshot.get("acquisition_failed") is True or any(
        item.get("status") == "failed" or bool(item.get("failures"))
        for item in snapshot.get("sources", []) if isinstance(item, dict))
    # Retained strict checkpoints can contain the earlier ambiguous empty status.
    # Preserve their bytes while refusing to interpret them as market absence.
    if (source in {"method", "method_odds"} and snapshot.get("observation_reference_schema") == 1
            and not snapshot.get("records") and status in {"blocked", "unavailable"} and not source_failed):
        return {"stage": source + "_snapshot", "status": "blocked",
            "reason_code": "no_usable_method_values_in_inspected_response",
            "error": "No usable method values were established; market availability remains unknown",
            "snapshot_status": status}
    if status in {"failed", "processing_refused", "blocked", "partial"} or source_failed:
        return {"stage": source + "_snapshot", "status": status if status in {"partial", "blocked"} else "failed",
            "reason_code": "source_acquisition_failed" if source_failed else "incomplete_collection",
            "error": "Observation collector returned an incomplete result", "snapshot_status": status}
    return None


def collect_reviewed_method_snapshot(*, tracked_fights=None):
    from src.data.method_odds import collect_method_odds_snapshot

    return collect_method_odds_snapshot(tracked_fights=tracked_fights, strict=True,
        reviewed_context=configured_observation_context("method"))


def collect_reviewed_rankings_snapshot(*, rosterwatch_page=None):
    context = configured_observation_context("rankings")
    provider = _rankings_provider(context)
    if provider.REGISTRY_KIND == "reviewed_rosterwatch_rankings_identity_v1":
        from src import config
        from src.data.rosterwatch_rankings_collection import collect_rankings_snapshot
        from src.data.rankings_observation_history import HISTORY_SUBDIR
        from src.data.selected_source_inputs import selected_path
        return collect_rankings_snapshot(reviewed_context=context,
            snapshot_root=Path(config.RAW_DATA_DIR) / "rankings",
            finalized_root=selected_path("rankings_observations", Path(config.RAW_DATA_DIR) / "kaggle_rankings" / HISTORY_SUBDIR),
            captured_page=rosterwatch_page)
    if rosterwatch_page is not None:
        raise ObservationContextError("rankings", "wrong_source", "Roster Watch response cannot enter an official UFC context")
    from src.data.rankings_scraper import collect_rankings_snapshot

    return collect_rankings_snapshot(strict=True,
        reviewed_context=context)
