"""Run the recurring UFC active-roster refresh pipeline in one command.

This is the schedulable path for:
- syncing the authoritative Roster Watch active roster
- refreshing missing UFCStats-backed profile/fight rows for active fighters
- rebuilding processed UFC artifacts
- re-auditing active-roster profile completeness
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from collections.abc import Mapping, Collection
from contextlib import contextmanager
from functools import wraps
from uuid import uuid4
from collections import deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_active_roster_profile_completeness import run_audit
from scripts.backfill_active_roster_ufcstats import (
    FIGHTERS_PATH as BACKFILL_FIGHTERS_PATH,
    PROFILE_SCRAPE_FAILURES_PATH as BACKFILL_PROFILE_FAILURES_PATH,
    RESULTS_PATH as BACKFILL_RESULTS_PATH,
    STATS_PATH as BACKFILL_STATS_PATH,
    run_backfill,
)
from scripts.build_profile_supplement_from_external_profiles import (
    ALL_SOURCES as PROFILE_SUPPLEMENT_ALL_SOURCES,
    TARGET_FIELDS as PROFILE_SUPPLEMENT_TARGET_FIELDS,
    _merge_source_rows as _merge_profile_supplement_source_rows,
    _source_row_key as _profile_supplement_source_row_key,
    run_profile_supplement_refresh,
)
from scripts.rebuild_ufc_processed_artifacts import run_rebuild
from src.model.production_bundle import (
    DEFAULT_MANIFEST_PATH,
    PRODUCTION_BUNDLE_ENV,
    _CURRENT_JEAN_SILVA_UFCSTATS_ID,
    _OLD_JEAN_SILVA_UFCSTATS_ID,
    _REQUIRED_COLLISION_IDENTITY_SENTINELS,
    _processed_identity_integrity_guard,
    get_processed_snapshot_fingerprints,
    is_hosted_runtime,
    reconcile_production_bundle_manifest,
)
from src.config import DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, PROJECT_ROOT, ROSTERWATCH_IDENTITY_REVIEW_PATH
from src.data.publication_identity import (
    APPROVED_V7_BASELINE_LINEAGE,
    _file_content_identity, _accepted_data_generation_identity, _accepted_raw_generation_identity, _validated_calendar_claim,
)
from src.data.io_utils import (
    csv_pair_manifest_path,
    verify_csv_pair_manifest,
    write_csv_atomically,
)
from src.data.refresh_publication import (
    PublicationPointer,
    RawArtifact,
    ResultsStatsPair,
    publication_journal_path,
    publish_raw_artifacts_transaction,
    recover_pending_publication,
    stage_accepted_raw_artifacts,
)
from src.data.ufc_data_quality import (
    configured_coverage_floors,
    coverage_snapshot_from_audit,
    preserve_accepted_processed_history,
    publication_quality_report,
    read_csv_required,
)
from src.data.name_utils import (
    CURRENT_UFCSTATS_ID_OVERRIDES,
    normalize_person_name,
    normalize_ufcstats_id,
)
from src.data.reviewed_roster_recovery import (
    reviewed_mma_identity_binding,
    reviewed_mma_identity_registry_record,
    reviewed_mma_identity_transition_pending_stable_id,
    reviewed_official_identity_transition_match,
    reviewed_profile_field_hold,
)
from src.data.reviewed_roster_identity import reviewed_profile_field_decisions
from src.data.profile_integrity import source_entity_id
from src.data.ufc_active_roster import (
    ACTIVE_ROSTER_SYNC_GENERATION_COLUMN,
    IDENTITY_AUDIT_COLUMNS,
    OFFICIAL_PROFILE_IDENTITY_PROVENANCE_COLUMNS,
    OFFICIAL_ACTIVE_ROSTER_PATH,
    OFFICIAL_ACTIVE_ROSTER_IDENTITY_AUDIT_PATH,
    _explicit_boolean,
    _identity_audit_frame,
    _paths_target_same_file,
    _roster_row_matches_excluded_identity,
    _roster_identity_keys,
    _roster_rows_same_identity,
    _semantic_csv_frame,
    _single_complete_sync_generation,
    retained_roster_diagnostics,
    sync_official_active_roster,
)
from src.data.rosterwatch_active_roster import (
    ROSTERWATCH_DELTA_LOG_FILENAME,
    ROSTERWATCH_LEGACY_LEDGER_FILENAME,
    ROSTERWATCH_OWNER_REGISTRY_FILENAME,
    ROSTERWATCH_RECURRING_LEDGER_FILENAME,
    ROSTERWATCH_SOURCE_LEDGER_FILENAME,
    ROSTERWATCH_STATE_FILENAME,
    ROSTERWATCH_LEARNED_ALIASES_FILENAME,
    ROSTERWATCH_IDENTITY_CAPTURE_FILENAME,
    load_rosterwatch_identity_provider,
    sync_rosterwatch_active_roster,
)
from src.data.rosterwatch_identity import emit_identity_review_alerts
from src.features.stance_utils import encode_stance

logger = logging.getLogger(__name__)

DEFAULT_DATASET_VARIANT = "pulled_all_plus_legacy_market"
DEFAULT_REBUILD_OUTPUT_SUBDIRS = ["."]
REFRESH_GENERATIONS_SUBDIR = "ufc_refresh_generations"
# Keep disposable raw generation paths short enough for long pair-manifest
# filenames under Windows test/runtime roots.
RAW_REFRESH_GENERATIONS_SUBDIR = "urf"
RETAIN_INACTIVE_REFRESH_GENERATIONS = 2
RETAIN_INACTIVE_RAW_REFRESH_GENERATIONS = 2
# The owner-approved weekly line starts from v7, even before the v7 model is
# promoted. A calendar claim may not touch sources unless the active accepted
# data manifest carries this exact, reviewed lineage binding.

DEFAULT_AUDIT_JSON = DATA_DIR / "tmp" / "active_roster_profile_completeness_scheduled_latest.json"
DEFAULT_AUDIT_CSV = DATA_DIR / "tmp" / "active_roster_profile_completeness_scheduled_latest.csv"
DEFAULT_UNRESOLVED_PROFILE_JSON = DATA_DIR / "tmp" / "active_roster_profile_unresolved_scheduled_latest.json"
DEFAULT_UNRESOLVED_PROFILE_CSV = DATA_DIR / "tmp" / "active_roster_profile_unresolved_scheduled_latest.csv"
PROFILE_SUPPLEMENT_PATH = RAW_DATA_DIR / "ufc_fighters_profile_supplement.csv"
PRE_UFC_REMAINING_NAME = "ufc_active_roster_pre_ufc_remaining.csv"
PRE_UFC_CHECKPOINT_NAME = "pre_ufc_career_checkpoint_v2.json"
DEFAULT_PROFILE_SUPPLEMENT_REFRESH_SOURCES = ("espn", "fightdx", "martialbot", "tapology", "sherdog", "wikipedia")
DEFAULT_HOSTED_PROFILE_SUPPLEMENT_REFRESH_SOURCES = tuple(
    source for source in DEFAULT_PROFILE_SUPPLEMENT_REFRESH_SOURCES if source != "tapology"
)
NEW_FIGHTER_ALERT_GRACE_DAYS_ENV = "UFC_REFRESH_NEW_FIGHTER_ALERT_GRACE_DAYS"
DEFAULT_NEW_FIGHTER_ALERT_GRACE_DAYS = 7
CACHED_ROSTER_FIRST_ENV = "UFC_REFRESH_USE_CACHED_ROSTER_FIRST"
CACHED_ROSTER_MAX_AGE_HOURS_ENV = "UFC_REFRESH_CACHED_ROSTER_MAX_AGE_HOURS"
DEFAULT_CACHED_ROSTER_MAX_AGE_HOURS = 72.0
ROSTERWATCH_SOURCE_PATH_ENV = "UFC_ROSTERWATCH_SOURCE_PATH"
ROSTERWATCH_SOURCE_URL_ENV = "UFC_ROSTERWATCH_SOURCE_URL"
ROSTERWATCH_SOURCE_CUTOFF_ENV = "UFC_ROSTERWATCH_SOURCE_CUTOFF"
ROSTERWATCH_INITIAL_MIGRATION_ENV = "UFC_ROSTERWATCH_ALLOW_INITIAL_MIGRATION"
ROSTERWATCH_STATE_SUBDIR = "rosterwatch"
ROSTERWATCH_SNAPSHOTS_SUBDIR = "snapshots"
ROSTERWATCH_EVENT_SAFETY_NET_HORIZON_DAYS = 42
ROSTERWATCH_EVENT_SAFETY_NET_MAX_EVENTS = 12
PROFILE_REPORT_FIELDS = ("age", "weight", "height", "reach", "stance")
PROFILE_REPORT_COLUMNS = (
    "official_name",
    "field",
    "why_missing_code",
    "why_missing",
    "octagon_debut",
    "official_athlete_url",
    "ufcstats_name",
    "ufcstats_url",
    "stable_ufcstats_id",
    "profile_source_alias",
    "official_field_value",
    "scraped_profile_match",
    "scraped_profile_name",
    "scraped_field_value",
    "supplement_rows",
    "supplement_sources",
    "supplement_field_values",
    "source_availability",
    "model_feature_fields",
    "repair_action",
)
_SOURCE_LIMITED_ALERT_REASON_CODES = frozenset(
    {
        "official_age_blank_and_no_dob_source",
        "no_ufcstats_url_and_supplement_blank",
        "no_ufcstats_url_and_no_supplement_rows",
        "ufcstats_blank_and_supplement_blank",
        "ufcstats_blank_and_no_supplement_rows",
        "reviewed_field_hold",
    }
)

# Image-bundled raw data path (used to detect stale volume copies)
_IMAGE_RAW_DIR = Path("/app/data/raw")
_ACTIVE_ROSTER_GUARD_ARTIFACT = "ufc_active_roster_official"
_ROSTER_GUARD_IDENTITY_COLUMNS = (
    "official_name",
    "official_athlete_url",
    "ufcstats_name",
    "ufcstats_url",
    "stable_ufcstats_id",
    "fighter_id",
    "profile_name",
    "slug_name",
    "alternate_slug_names",
    "canonical_athlete_url",
    "official_profile_fetch_outcome",
    "official_url_identity_status",
    "official_url_identity_valid",
    "combat_sport",
    *OFFICIAL_PROFILE_IDENTITY_PROVENANCE_COLUMNS,
)


def _profile_report_identity_key(row: dict[str, object] | pd.Series) -> str:
    """Build an owner-safe source key; normalized names never bind reports."""

    official_url = _string_value(row.get("official_athlete_url"))
    official_entity = source_entity_id("ufc", official_url)
    if official_url and not official_entity:
        return ""
    supplied_ids: list[str] = []
    for field in ("stable_ufcstats_id", "fighter_id", "ufcstats_url"):
        raw = row.get(field)
        text = _string_value(raw)
        if not text:
            continue
        stable_id = normalize_ufcstats_id(raw)
        if not stable_id:
            return ""
        supplied_ids.append(stable_id)
    stable_ids = set(supplied_ids)
    if len(stable_ids) > 1:
        return ""
    stable_id = next(iter(stable_ids), "")
    if official_entity and stable_id:
        return f"ufc:{official_entity}|ufcstats:{stable_id}"
    if official_entity:
        return f"ufc:{official_entity}"
    return f"ufcstats:{stable_id}" if stable_id else ""

_READ_ONLY_REBUILD_INPUTS = (
    ("ufc-master.csv", True),
    ("github_PierceHampton.csv", False),
    ("historical_odds", False),
    (str(Path("kaggle_rankings") / "rankings_history_extended.csv"), False),
    (str(Path("method_odds") / "historical_method_odds_all.csv"), False),
    (str(Path("method_odds") / "finalized_observations_v1"), False),
    (str(Path("kaggle_rankings") / "retained_rankings_observations_v1"), False),
    ("pre_ufc_career_supplement_v2.csv", False),
    ("pre_ufc_career_supplement.csv", False),
    ("amateur_career_supplement.csv", False),
    ("reviewed_fighter_history.csv", False),
)


def _raw_publication_artifacts(*roots: Path) -> tuple[RawArtifact, ...]:
    """Resolve mutable raw artifacts and immutable Roster Watch snapshots."""
    results_relative = BACKFILL_RESULTS_PATH.name
    stats_relative = BACKFILL_STATS_PATH.name
    pair_manifest_relative = csv_pair_manifest_path(
        Path(results_relative),
        Path(stats_relative),
    ).name
    artifacts = [
        RawArtifact(OFFICIAL_ACTIVE_ROSTER_PATH.name),
        RawArtifact(OFFICIAL_ACTIVE_ROSTER_IDENTITY_AUDIT_PATH.name, required=False),
        RawArtifact(BACKFILL_FIGHTERS_PATH.name),
        RawArtifact(BACKFILL_PROFILE_FAILURES_PATH.name, required=False),
        RawArtifact(results_relative),
        RawArtifact(stats_relative),
        # Existing deployments predate the pair marker; staging can synthesize it
        # and publication creates it only as part of an accepted transaction.
        RawArtifact(pair_manifest_relative, required=False),
        RawArtifact(PROFILE_SUPPLEMENT_PATH.name, required=False),
        RawArtifact(PRE_UFC_REMAINING_NAME),
        RawArtifact(PRE_UFC_CHECKPOINT_NAME),
        RawArtifact("ufc-event-dates.csv", required=False),
    ]
    rosterwatch_root = Path(ROSTERWATCH_STATE_SUBDIR)
    for filename in (
        ROSTERWATCH_STATE_FILENAME,
        ROSTERWATCH_SOURCE_LEDGER_FILENAME,
        ROSTERWATCH_LEGACY_LEDGER_FILENAME,
        ROSTERWATCH_OWNER_REGISTRY_FILENAME,
        ROSTERWATCH_RECURRING_LEDGER_FILENAME,
        ROSTERWATCH_DELTA_LOG_FILENAME,
        Path(ROSTERWATCH_IDENTITY_REVIEW_PATH).name,
    ):
        artifacts.append(
            RawArtifact((rosterwatch_root / filename).as_posix(), required=False)
        )

    snapshot_relatives: set[str] = set()
    for filename in (ROSTERWATCH_LEARNED_ALIASES_FILENAME, ROSTERWATCH_IDENTITY_CAPTURE_FILENAME):
        if any((Path(root) / rosterwatch_root / filename).exists() for root in roots):
            artifacts.append(RawArtifact((rosterwatch_root / filename).as_posix(), required=False))
    for root in roots:
        snapshot_root = (
            Path(root)
            / ROSTERWATCH_STATE_SUBDIR
            / ROSTERWATCH_SNAPSHOTS_SUBDIR
        )
        if not snapshot_root.exists():
            continue
        if not snapshot_root.is_dir() or snapshot_root.is_symlink():
            raise RuntimeError(
                f"Roster Watch snapshot root is not a safe directory: {snapshot_root}"
            )
        for snapshot in sorted(snapshot_root.glob("*.csv")):
            if not snapshot.is_file() or snapshot.is_symlink():
                raise RuntimeError(
                    f"Roster Watch snapshot is not a safe regular file: {snapshot}"
                )
            snapshot_relatives.add(snapshot.relative_to(Path(root)).as_posix())
        # Raw roster.watch website captures (page + retrieval metadata) are
        # immutable evidence for every derived snapshot and publish with it.
        site_root = snapshot_root / "site"
        if site_root.exists():
            if not site_root.is_dir() or site_root.is_symlink():
                raise RuntimeError(
                    f"Roster Watch site capture root is not a safe directory: {site_root}"
                )
            for pattern in ("*.html", "*.meta.json"):
                for capture in sorted(site_root.glob(pattern)):
                    if not capture.is_file() or capture.is_symlink():
                        raise RuntimeError(
                            f"Roster Watch site capture is not a safe regular file: {capture}"
                        )
                    snapshot_relatives.add(capture.relative_to(Path(root)).as_posix())
    artifacts.extend(
        RawArtifact(relative)
        for relative in sorted(snapshot_relatives)
    )
    from src.data.provider_consumer_references import provider_reference_artifacts
    from src.data.observation_consumer_references import observation_reference_artifacts
    declared = {artifact.relative_path for artifact in artifacts}
    for root in roots:
        for relative in sorted(set(provider_reference_artifacts(root)) | set(observation_reference_artifacts(root))):
            if relative not in declared:
                artifacts.append(RawArtifact(relative, required=False))
                declared.add(relative)
    return tuple(artifacts)


def _raw_results_stats_pair() -> ResultsStatsPair:
    results_relative = BACKFILL_RESULTS_PATH.name
    stats_relative = BACKFILL_STATS_PATH.name
    return ResultsStatsPair(
        results_path=results_relative,
        stats_path=stats_relative,
        manifest_path=csv_pair_manifest_path(
            Path(results_relative),
            Path(stats_relative),
        ).name,
    )


def _file_row_count(path: Path) -> int | None:
    """Return the CSV row count (excluding header) or None if the file doesn't exist."""
    if not path.exists():
        return None
    try:
        return int(pd.read_csv(path, usecols=[0]).shape[0])
    except Exception:
        return None


def _file_snapshot(path: Path) -> dict[str, object]:
    """Return diagnostic metadata about a file."""
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "exists": True,
            "size_bytes": int(stat.st_size),
            "row_count": _file_row_count(path) if path.suffix.lower() == ".csv" else None,
            "mtime_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        }
    except OSError:
        return {"path": str(path), "exists": True}


def _event_date_evidence_lookup(path: Path) -> dict[str, str]:
    """Read valid, non-conflicting event dates from the staged raw cache."""
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    required = {"event_name", "event_date"}
    if not required.issubset(frame.columns):
        raise ValueError(
            "event-date evidence is missing columns: "
            + ",".join(sorted(required - set(frame.columns)))
        )

    lookup: dict[str, str] = {}
    for row in frame.to_dict(orient="records"):
        event_key = normalize_person_name(row.get("event_name"))
        parsed = pd.to_datetime(row.get("event_date"), errors="coerce", format="mixed")
        if not event_key or pd.isna(parsed):
            continue
        event_date = parsed.date().isoformat()
        previous = lookup.get(event_key)
        if previous is not None:
            duplicate_kind = "conflicts" if previous != event_date else "is duplicated"
            raise ValueError(
                f"event-date evidence {duplicate_kind} for "
                f"{event_key}: {previous} and {event_date}"
            )
        lookup[event_key] = event_date
    return lookup


def _refresh_input_artifact_inventory(raw_root: Path) -> dict[str, object]:
    """Fingerprint the accepted raw inputs that can affect a weekly rebuild."""
    artifacts: dict[str, dict[str, object]] = {}

    def add_file(path: Path, relative_path: Path) -> None:
        key = relative_path.as_posix()
        identity = _file_content_identity(path)
        existing = artifacts.get(key)
        if existing is not None and existing != identity:
            raise RuntimeError(f"refresh input was discovered twice with different bytes: {key}")
        artifacts[key] = identity

    pair_manifest_relative = Path(_raw_results_stats_pair().manifest_path).as_posix()
    for artifact in _raw_publication_artifacts(raw_root):
        relative = Path(artifact.relative_path)
        # This marker is regenerated inside the atomic transaction after the
        # pointer is prepared. The two files it binds are fingerprinted here.
        if relative.as_posix() == pair_manifest_relative:
            continue
        path = raw_root / relative
        if not path.exists():
            if artifact.required:
                raise RuntimeError(f"required accepted refresh input is missing: {relative}")
            continue
        if not path.is_file():
            raise RuntimeError(f"accepted refresh input is not a file: {relative}")
        add_file(path, relative)

    for relative_text, required in _READ_ONLY_REBUILD_INPUTS:
        relative = Path(relative_text)
        path = raw_root / relative
        if not path.exists():
            if required:
                raise RuntimeError(f"required accepted rebuild input is missing: {relative}")
            continue
        if path.is_file():
            add_file(path, relative)
            continue
        if not path.is_dir():
            raise RuntimeError(f"accepted rebuild input has an unsupported type: {relative}")
        for child in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
            if child.is_file():
                add_file(child, child.relative_to(raw_root))

    return {
        "schema_version": "ufc_refresh_input_inventory_v1",
        "artifacts": artifacts,
    }


def _accepted_update_report(
    *,
    audit_alert_summary: dict[str, object],
    backfill_summary: dict[str, object],
    coverage_snapshot: dict[str, object],
    coverage_floors: dict[str, float],
) -> dict[str, object]:
    """Select the compact existing update metrics needed for candidate review."""
    new_fighters = audit_alert_summary.get("newly_added_active_roster")
    if not isinstance(new_fighters, dict):
        new_fighters = {}
    source_limited = audit_alert_summary.get("source_limited_missing_fields")
    if not isinstance(source_limited, dict):
        source_limited = {}

    def _count(mapping: dict[str, object], key: str) -> int:
        try:
            return max(int(mapping.get(key) or 0), 0)
        except (TypeError, ValueError):
            return 0

    result_classes = backfill_summary.get("result_class_counts")
    exclusions = backfill_summary.get("result_exclusion_counts")
    if not isinstance(result_classes, dict):
        result_classes = {}
    if not isinstance(exclusions, dict):
        exclusions = {}
    return {
        "schema_version": 1,
        "new_fighters": {
            "total": _count(new_fighters, "rows_total"),
            "in_grace": _count(new_fighters, "rows_in_grace"),
            "alert_eligible": _count(new_fighters, "rows_alert_eligible"),
            "source_limited_missing_fields": source_limited,
        },
        "completed_fights": {
            "fight_count": _count(backfill_summary, "new_result_rows"),
            "result_rows": _count(backfill_summary, "new_result_rows"),
            "stat_rows": _count(backfill_summary, "new_stat_rows"),
        },
        "unusual_results": {
            "result_class_counts": result_classes,
            "exclusion_counts": exclusions,
            "quarantined_count": len(backfill_summary.get("quarantined_fight_urls") or []),
        },
        "coverage": {
            "snapshot": coverage_snapshot,
            "floors": coverage_floors,
        },
    }


def _stage_read_only_rebuild_inputs(
    *,
    accepted_root: Path,
    candidate_root: Path,
) -> dict[str, object]:
    """Freeze every non-mutable rebuild input inside the candidate raw tree."""
    accepted_root = accepted_root.resolve(strict=True)
    candidate_root = candidate_root.resolve(strict=True)
    artifacts: list[dict[str, object]] = []

    def _copy_one(source: Path, target: Path, relative: Path) -> None:
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"unsafe_read_only_rebuild_input:{source}")
        before = _file_content_identity(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise RuntimeError(f"unsafe_staged_rebuild_input:{target}")
        else:
            shutil.copyfile(source, target)
        candidate = _file_content_identity(target)
        after = _file_content_identity(source)
        if before != after or candidate != before:
            raise RuntimeError(f"read_only_rebuild_input_changed_during_stage:{relative}")
        artifacts.append(
            {
                "relative_path": relative.as_posix(),
                "sha256": str(before["sha256"]),
                "bytes": int(before["bytes"]),
            }
        )

    for relative_text, required in _READ_ONLY_REBUILD_INPUTS:
        relative = Path(relative_text)
        source = (accepted_root / relative).resolve(strict=False)
        target = (candidate_root / relative).resolve(strict=False)
        if accepted_root not in source.parents or candidate_root not in target.parents:
            raise RuntimeError(f"read_only_rebuild_input_path_escape:{relative_text}")
        if not source.exists():
            if required:
                raise FileNotFoundError(f"required_read_only_rebuild_input_missing:{source}")
            continue
        if source.is_symlink():
            raise RuntimeError(f"unsafe_read_only_rebuild_input:{source}")
        if source.is_file():
            _copy_one(source, target, relative)
            continue
        if not source.is_dir():
            raise RuntimeError(f"unsafe_read_only_rebuild_input:{source}")
        before_members = sorted(
            path.relative_to(source)
            for path in source.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        unsafe_members = [path for path in source.rglob("*") if path.is_symlink()]
        if unsafe_members:
            raise RuntimeError(f"symlink_in_read_only_rebuild_input:{unsafe_members[0]}")
        for member in before_members:
            _copy_one(source / member, target / member, relative / member)
        after_members = sorted(
            path.relative_to(source)
            for path in source.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        if before_members != after_members:
            raise RuntimeError(f"read_only_rebuild_input_membership_changed:{relative}")
    from src.data.provider_consumer_references import provider_reference_artifacts
    from src.data.observation_consumer_references import observation_reference_artifacts
    observation_paths = set(observation_reference_artifacts(accepted_root))
    for relative_text in sorted(set(provider_reference_artifacts(accepted_root)) | observation_paths):
        relative = Path(relative_text)
        # Mutable candidate inputs may have changed after the original stage;
        # their own sidecars must validate below. Read-only inputs and immutable
        # manifest objects retain the exact accepted bytes.
        readonly_names = {name for name, _required in _READ_ONLY_REBUILD_INPUTS}
        if ((candidate_root / relative).exists() and relative_text not in readonly_names
                and "provider_consumer_manifests_v1" not in relative.parts
                and "observation_consumer_manifests_v1" not in relative.parts
                and relative_text not in observation_paths):
            continue
        _copy_one(accepted_root / relative, candidate_root / relative, relative)
    provider_reference_artifacts(candidate_root)
    observation_reference_artifacts(candidate_root)
    return {
        "accepted_root": str(accepted_root),
        "candidate_root": str(candidate_root),
        "artifacts": artifacts,
    }


def _truthy_env(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default) or "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_float_env(name: str, default: float) -> float:
    raw = str(os.getenv(name, str(default)) or "").strip()
    try:
        return float(raw)
    except Exception:
        return default


def _cached_roster_max_age_hours() -> float:
    return max(_safe_float_env(CACHED_ROSTER_MAX_AGE_HOURS_ENV, DEFAULT_CACHED_ROSTER_MAX_AGE_HOURS), 0.0)


def _load_fresh_cached_roster_for_hosted_refresh(path: Path) -> pd.DataFrame | None:
    """Use the mounted roster snapshot first so result backfill is not blocked by UFC.com."""
    if not is_hosted_runtime() or not _truthy_env(CACHED_ROSTER_FIRST_ENV, "1"):
        return None
    if not path.exists():
        return None

    try:
        stat = path.stat()
        max_age_hours = _cached_roster_max_age_hours()
        age_hours = (datetime.now(timezone.utc).timestamp() - stat.st_mtime) / 3600.0
        if max_age_hours > 0 and age_hours > max_age_hours:
            logger.info(
                "Cached UFC roster is %.1fh old, beyond %.1fh max; doing live roster sync before backfill",
                age_hours,
                max_age_hours,
            )
            return None
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to load cached UFC roster for hosted refresh fast path: %s", exc)
        return None

    if df.empty:
        return None

    df.attrs["sync_source"] = "cached_runtime_preexisting"
    df.attrs["cached_snapshot_mtime_utc"] = datetime.fromtimestamp(
        stat.st_mtime,
        tz=timezone.utc,
    ).isoformat()
    df.attrs["cached_snapshot_age_hours"] = round(age_hours, 3)
    df.attrs["retained_missing_live_rows"] = retained_roster_diagnostics(df)
    return df


def _log_resolved_data_paths() -> dict[str, str]:
    """Log and return all resolved data paths used by the refresh pipeline."""
    paths = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "DATA_DIR": str(DATA_DIR),
        "RAW_DATA_DIR": str(RAW_DATA_DIR),
        "PROCESSED_DATA_DIR": str(PROCESSED_DATA_DIR),
        "OFFICIAL_ACTIVE_ROSTER_PATH": str(OFFICIAL_ACTIVE_ROSTER_PATH),
        "BACKFILL_FIGHTERS_PATH": str(BACKFILL_FIGHTERS_PATH),
        "UFC_DATA_DIR_env": os.environ.get("UFC_DATA_DIR", "(unset)"),
        "RAILWAY_VOLUME_MOUNT_PATH_env": os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "(unset)"),
        "is_hosted_runtime": str(is_hosted_runtime()),
    }
    logger.info(
        "UFC refresh resolved data paths: %s",
        json.dumps(paths, indent=2),
    )
    return paths


def _row_drop_guard_files(
    *,
    raw_data_dir: Path | None = None,
    processed_data_dir: Path | None = None,
) -> dict[str, Path]:
    raw_root = Path(raw_data_dir) if raw_data_dir is not None else RAW_DATA_DIR
    processed_root = (
        Path(processed_data_dir)
        if processed_data_dir is not None
        else PROCESSED_DATA_DIR
    )
    return {
        "ufc_fighters_scraped": raw_root / BACKFILL_FIGHTERS_PATH.name,
        "ufc_active_roster_official": raw_root / OFFICIAL_ACTIVE_ROSTER_PATH.name,
        "ufc_fight_results": raw_root / BACKFILL_RESULTS_PATH.name,
        "ufc_fight_stats": raw_root / BACKFILL_STATS_PATH.name,
        "processed_fights_cleaned": processed_root / "fights_cleaned.csv",
        "processed_features": processed_root / "features.csv",
        "processed_test_set": processed_root / "test_set.csv",
    }


def _roster_guard_identity_snapshot(path: Path) -> list[dict[str, str]] | None:
    """Read the minimum persisted identity fields needed for exact drop checks."""
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None

    rows: list[dict[str, str]] = []
    for _, row in frame.iterrows():
        identity: dict[str, str] = {}
        for column in _ROSTER_GUARD_IDENTITY_COLUMNS:
            value = row.get(column)
            try:
                if pd.isna(value):
                    value = ""
            except (TypeError, ValueError):
                pass
            identity[column] = (
                "" if value is None else str(value).strip()
            )
        if _roster_identity_keys(identity):
            rows.append(identity)
    return rows


def _row_guard_snapshot(
    *,
    raw_data_dir: Path | None = None,
    processed_data_dir: Path | None = None,
) -> dict[str, dict[str, object]]:
    snapshots: dict[str, dict[str, object]] = {}
    for name, path in _row_drop_guard_files(
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
    ).items():
        snapshot = _file_snapshot(path)
        if name == _ACTIVE_ROSTER_GUARD_ARTIFACT:
            identity_rows = _roster_guard_identity_snapshot(path)
            if identity_rows is not None:
                snapshot["identity_rows"] = identity_rows
        snapshots[name] = snapshot
    return snapshots


def _new_refresh_generation_dir(*, now: datetime | None = None) -> Path:
    """Return a unique processed directory that no active manifest addresses."""
    observed_at = now or datetime.now(timezone.utc)
    generation_id = (
        f"refresh-{observed_at.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:12]}"
    )
    return _refresh_generations_root() / generation_id


def _new_raw_refresh_generation_dir(*, now: datetime | None = None) -> Path:
    """Return an isolated sibling tree for candidate raw-data writes."""
    observed_at = now or datetime.now(timezone.utc)
    generation_id = f"r-{observed_at.strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:6]}"
    return _raw_refresh_generations_root() / generation_id


def _refresh_generations_root() -> Path:
    """Keep candidate processed generations outside the accepted processed tree."""
    return PROCESSED_DATA_DIR.parent / "tmp" / REFRESH_GENERATIONS_SUBDIR


def _raw_refresh_generations_root() -> Path:
    """Keep candidate raw generations outside the accepted raw tree."""
    return RAW_DATA_DIR.parent / "tmp" / RAW_REFRESH_GENERATIONS_SUBDIR


def _active_refresh_generation_dir(generations_root: Path) -> Path | None:
    manifest = _active_production_manifest_path()
    if not manifest.is_file():
        return None
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("active production manifest is not a JSON object")
    raw = str(payload.get("processed_dir") or "").strip()
    if not raw:
        raise ValueError("active production manifest has no processed_dir")
    candidate = Path(raw).resolve(strict=False)
    root = generations_root.resolve(strict=False)
    if candidate.parent != root or not candidate.name.startswith("refresh-"):
        return None
    return candidate


def _prune_refresh_generations(
    *,
    retain_inactive: int = RETAIN_INACTIVE_REFRESH_GENERATIONS,
    protected: Collection[Path] = (),
    fence_reason: str | None = None,
) -> dict[str, object]:
    """Bound inactive refresh storage while never deleting the addressed generation.

    ``protected`` lists generation directories that a recorded refresh cycle still references;
    they are retained regardless of age and reported under ``protected``.
    A ``fence_reason`` refuses cleanup before reference/manifest inspection when
    the complete cross-consumer reference inventory is not available.
    """
    root = _refresh_generations_root()
    summary: dict[str, object] = {
        "root": str(root),
        "active_generation": "",
        "retained_inactive": [],
        "protected": [],
        "removed": [],
        "failures": [],
    }
    if fence_reason is not None:
        summary["fenced"] = str(fence_reason)
        return summary
    protected_resolved = {Path(path).resolve(strict=False) for path in protected}
    if not root.exists():
        return summary
    if root.is_symlink() or not root.is_dir():
        summary["failures"] = [f"unsafe_generation_root:{root}"]
        return summary

    resolved_root = root.resolve(strict=True)
    manifest = _active_production_manifest_path()
    manifest_before = manifest.read_bytes() if manifest.is_file() else None
    try:
        active = _active_refresh_generation_dir(resolved_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        summary["failures"] = [f"active_manifest_unresolved:{exc}"]
        return summary
    summary["active_generation"] = str(active) if active is not None else ""
    candidates: list[Path] = []
    for child in root.iterdir():
        if child.is_symlink() or not child.is_dir() or not child.name.startswith("refresh-"):
            continue
        resolved = child.resolve(strict=True)
        if resolved.parent != resolved_root:
            summary["failures"].append(f"unsafe_generation_path:{resolved}")
            continue
        if active is not None and resolved == active:
            continue
        if resolved in protected_resolved:
            summary["protected"].append(str(resolved))
            continue
        candidates.append(resolved)

    keep_count = max(int(retain_inactive), 0)
    candidates.sort(key=lambda path: path.name, reverse=True)
    retained = candidates[:keep_count]
    summary["retained_inactive"] = [str(path) for path in retained]
    for path in candidates[keep_count:]:
        manifest_now = manifest.read_bytes() if manifest.is_file() else None
        if manifest_now != manifest_before:
            summary["failures"].append("active_manifest_changed_during_cleanup")
            break
        try:
            shutil.rmtree(path)
            summary["removed"].append(str(path))
        except OSError as exc:
            summary["failures"].append(f"{path}:{exc}")
    return summary


def _prune_raw_refresh_generations(
    current_generation: Path,
    *,
    retain_inactive: int = RETAIN_INACTIVE_RAW_REFRESH_GENERATIONS,
    protected: Collection[Path] = (),
    fence_reason: str | None = None,
) -> dict[str, object]:
    """Bound raw candidate storage while preserving the current exact child.

    ``protected`` lists raw generation directories that a recorded refresh cycle still references
    (retained acquisition evidence); they are never removed and are reported under ``protected``.
    ``fence_reason`` preserves every generation while any reference source is unresolved.
    """
    root = _raw_refresh_generations_root()
    summary: dict[str, object] = {
        "root": str(root),
        "current_generation": str(current_generation),
        "retained_inactive": [],
        "protected": [],
        "removed": [],
        "failures": [],
    }
    if fence_reason is not None:
        summary["fenced"] = str(fence_reason)
        return summary
    protected_resolved = {Path(path).resolve(strict=False) for path in protected}
    if not root.exists():
        summary["failures"] = [f"raw_generation_root_missing:{root}"]
        return summary
    if root.is_symlink() or not root.is_dir():
        summary["failures"] = [f"unsafe_raw_generation_root:{root}"]
        return summary
    resolved_root = root.resolve(strict=True)
    current = Path(current_generation)
    if current.is_symlink() or not current.is_dir():
        summary["failures"] = [f"unsafe_current_raw_generation:{current}"]
        return summary
    resolved_current = current.resolve(strict=True)
    if (
        resolved_current.parent != resolved_root
        or not resolved_current.name.startswith("r-")
    ):
        summary["failures"] = [
            f"current_raw_generation_outside_declared_root:{resolved_current}"
        ]
        return summary

    candidates: list[Path] = []
    for child in root.iterdir():
        if child == current:
            continue
        if child.is_symlink():
            summary["failures"].append(f"unsafe_raw_generation_symlink:{child}")
            continue
        if not child.is_dir() or not child.name.startswith("r-"):
            continue
        resolved = child.resolve(strict=True)
        if resolved.parent != resolved_root:
            summary["failures"].append(f"unsafe_raw_generation_path:{resolved}")
            continue
        if resolved in protected_resolved:
            summary["protected"].append(str(resolved))
            continue
        candidates.append(resolved)

    candidates.sort(key=lambda path: path.name, reverse=True)
    keep_count = max(int(retain_inactive), 0)
    retained = candidates[:keep_count]
    summary["retained_inactive"] = [str(path) for path in retained]
    for path in candidates[keep_count:]:
        if path.parent != resolved_root or path == resolved_current:
            summary["failures"].append(f"refused_raw_generation_removal:{path}")
            continue
        try:
            shutil.rmtree(path)
            summary["removed"].append(str(path))
        except OSError as exc:
            summary["failures"].append(f"{path}:{exc}")
    return summary


@contextmanager
def _refresh_writer_lock():
    """Serialize refresh generations and manifest publication across processes."""
    root = _refresh_generations_root()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".scheduled-refresh.lock"
    handle = lock_path.open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("scheduled_ufc_refresh_already_running") from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("scheduled_ufc_refresh_already_running") from exc
        acquired = True
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _serialized_refresh(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _refresh_writer_lock():
            return function(*args, **kwargs)

    return wrapped


def _staged_rebuild_output_subdirs(
    staged_processed_dir: Path,
    requested_output_subdirs: list[str],
) -> list[str]:
    """Keep all rebuild outputs inside the isolated processed candidate."""
    outputs = ["."]
    canonical_root = PROCESSED_DATA_DIR.resolve(strict=False)
    active_processed_dir = _active_production_processed_dir()
    staged_root = staged_processed_dir.resolve(strict=False)
    seen = {os.path.normcase(str(staged_root))}
    for value in requested_output_subdirs:
        text = str(value or "").strip()
        if text in {"", ".", "base"}:
            continue
        candidate = Path(text)
        legacy_resolved = (
            candidate.resolve(strict=False)
            if candidate.is_absolute()
            else (PROCESSED_DATA_DIR / candidate).resolve(strict=False)
        )
        if legacy_resolved == canonical_root:
            continue
        if active_processed_dir is not None and legacy_resolved == active_processed_dir:
            raise RuntimeError(
                "refusing to rebuild directly into the active production "
                f"processed_dir: {legacy_resolved}"
            )
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(
                "scheduled refresh output_subdirs must be relative paths inside "
                "the isolated candidate generation"
            )
        resolved = (staged_root / candidate).resolve(strict=False)
        if staged_root not in resolved.parents:
            raise ValueError("scheduled refresh output_subdir escapes candidate generation")
        identity = os.path.normcase(str(resolved))
        if identity in seen:
            continue
        seen.add(identity)
        outputs.append(text)
    return outputs


def _active_production_manifest_path() -> Path:
    configured = str(os.getenv(PRODUCTION_BUNDLE_ENV, "") or "").strip()
    candidate = Path(configured) if configured else DEFAULT_MANIFEST_PATH
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def recover_scheduled_refresh_publication() -> dict[str, object]:
    """Recover any killed raw-plus-processed publication before serving data."""
    active_manifest = _active_production_manifest_path()
    return recover_pending_publication(publication_journal_path(active_manifest))


def _publication_recovery_for_refresh(
    *, publication_capable: bool
) -> dict[str, object]:
    """Recover only for a publication-capable run; otherwise inspect, never mutate."""
    active_manifest = _active_production_manifest_path()
    journal_path = publication_journal_path(active_manifest)
    if publication_capable:
        return recover_scheduled_refresh_publication()
    if journal_path.exists():
        raise RuntimeError(
            "pending publication journal exists during a publication-disabled "
            f"refresh; refusing read-only candidate execution: {journal_path}"
        )
    return {
        "action": "skipped",
        "reason": "publication_disabled",
        "journal_path": str(journal_path),
    }


def _active_approved_v7_baseline_lineage(
    *,
    required: bool,
) -> dict[str, object] | None:
    """Validate continuity from a separately reviewed v7 baseline manifest."""
    manifest_path = _active_production_manifest_path()
    if not manifest_path.is_file():
        if required:
            raise RuntimeError("approved v7 accepted-data manifest is missing")
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"active production manifest is unreadable: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("active production manifest is not a JSON object")

    publication = manifest.get("accepted_data_publication")
    lineage = publication.get("baseline_lineage") if isinstance(publication, dict) else None
    if lineage is None:
        if required:
            raise RuntimeError(
                "approved v7 accepted-data lineage is missing; refusing the weekly refresh"
            )
        return None
    if lineage != APPROVED_V7_BASELINE_LINEAGE:
        raise RuntimeError(
            "accepted data lacks the exact owner-approved v7 baseline binding"
        )

    manifest_processed = str(manifest.get("processed_dir") or "").strip()
    publication_processed = str(publication.get("processed_dir") or "").strip()
    if not manifest_processed or not publication_processed:
        raise RuntimeError("approved v7 accepted-data processed path is missing")
    manifest_dir = Path(manifest_processed)
    publication_dir = Path(publication_processed)
    if not manifest_dir.is_absolute():
        manifest_dir = PROJECT_ROOT / manifest_dir
    if not publication_dir.is_absolute():
        publication_dir = PROJECT_ROOT / publication_dir
    manifest_dir = manifest_dir.resolve(strict=False)
    publication_dir = publication_dir.resolve(strict=False)
    if manifest_dir != publication_dir:
        raise RuntimeError("approved v7 accepted-data processed paths disagree")

    actual_fingerprints = get_processed_snapshot_fingerprints(manifest_dir)
    for field, actual in actual_fingerprints.items():
        if publication.get(field) != actual or manifest.get(field) != actual:
            raise RuntimeError(
                f"approved v7 accepted-data artifact does not match its manifest: {field}"
            )
    actual_generation = _accepted_data_generation_identity(manifest_dir)[
        "accepted_generation_id"
    ]
    if publication.get("accepted_generation_id") != actual_generation:
        raise RuntimeError("approved v7 accepted-data generation identity is invalid")
    actual_refresh_inputs = _refresh_input_artifact_inventory(RAW_DATA_DIR)
    if publication.get("refresh_input_inventory") != actual_refresh_inputs:
        raise RuntimeError(
            "approved v7 accepted raw roster/data inputs do not match their manifest"
        )
    recorded_raw_generation_id = str(
        publication.get("accepted_raw_generation_id") or ""
    ).strip()
    expected_raw_generation_id = _accepted_raw_generation_identity(
        actual_refresh_inputs
    )["accepted_raw_generation_id"]
    if (
        recorded_raw_generation_id
        and recorded_raw_generation_id != expected_raw_generation_id
    ):
        raise RuntimeError("approved v7 accepted raw generation identity is invalid")
    if (
        int(publication.get("schema_version") or 1) >= 2
        and not recorded_raw_generation_id
    ):
        raise RuntimeError("approved v7 accepted raw generation identity is missing")
    return dict(lineage)


def _active_production_processed_dir() -> Path | None:
    """Resolve the manifest-addressed processed directory before any rebuild."""
    manifest = _active_production_manifest_path()
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"active production manifest is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("active production manifest is not a JSON object")
    raw = str(payload.get("processed_dir") or "").strip()
    if not raw:
        raise RuntimeError("active production manifest has no processed_dir")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve(strict=False)


def _prepare_validated_processed_generation(
    processed_dir: Path,
    *,
    update_report: dict[str, object],
    accepted_at_utc: str,
    calendar_claim: dict[str, object] | None = None,
    baseline_lineage: dict[str, object] | None = None,
    refresh_input_inventory: dict[str, object],
) -> tuple[dict[str, object], PublicationPointer]:
    """Prevalidate a processed pointer without mutating the accepted manifest."""
    active_manifest = _active_production_manifest_path()
    active_manifest.parent.mkdir(parents=True, exist_ok=True)
    active_before = active_manifest.read_bytes() if active_manifest.exists() else None
    with TemporaryDirectory(
        prefix="ufc_refresh_manifest_prevalidation_",
        dir=active_manifest.parent,
    ) as temporary:
        candidate_manifest = Path(temporary) / "production-bundle.json"
        if active_before is not None:
            shutil.copyfile(active_manifest, candidate_manifest)
        summary = reconcile_production_bundle_manifest(
            target_manifest_path=candidate_manifest,
            processed_dir=processed_dir,
        )
        candidate_bytes = candidate_manifest.read_bytes()
        candidate_payload = json.loads(candidate_bytes.decode("utf-8"))
        if not isinstance(candidate_payload, dict):
            raise RuntimeError("Prevalidated production manifest is not a JSON object")
        active_now = active_manifest.read_bytes() if active_manifest.exists() else None
        if active_now != active_before:
            raise RuntimeError("Active production manifest changed during refresh prevalidation")
        processed_identity = get_processed_snapshot_fingerprints(processed_dir)
        accepted_generation = _accepted_data_generation_identity(processed_dir)
        accepted_generation_id = str(accepted_generation["accepted_generation_id"])
        raw_generation = _accepted_raw_generation_identity(refresh_input_inventory)
        for field, value in processed_identity.items():
            candidate_payload[field] = value
        candidate_payload["update_report"] = update_report
        accepted_publication = {
            "schema_version": 2,
            "accepted_generation_id": accepted_generation_id,
            **raw_generation,
            "accepted_at_utc": accepted_at_utc,
            "processed_dir": str(processed_dir.resolve(strict=False)),
            "update_report": update_report,
            "refresh_input_inventory": refresh_input_inventory,
            **processed_identity,
        }
        if calendar_claim is not None:
            accepted_publication["calendar_claim"] = dict(calendar_claim)
        if baseline_lineage is not None:
            accepted_publication["baseline_lineage"] = dict(baseline_lineage)
        candidate_payload["accepted_data_publication"] = accepted_publication
        candidate_bytes = (json.dumps(candidate_payload, indent=2) + "\n").encode("utf-8")
        summary = {
            **summary,
            **processed_identity,
            "accepted_generation_id": accepted_generation_id,
            **raw_generation,
        }
    return (
        {**summary, "manifest_path": str(active_manifest)},
        PublicationPointer(
            target_path=active_manifest,
            content=candidate_bytes,
            journal_path=publication_journal_path(active_manifest),
        ),
    )


def _snapshot_with_staged_processed_files(
    snapshot: dict[str, dict[str, object]],
    *,
    processed_dir: Path,
) -> dict[str, dict[str, object]]:
    staged = dict(snapshot)
    staged["processed_fights_cleaned"] = _file_snapshot(processed_dir / "fights_cleaned.csv")
    staged["processed_features"] = _file_snapshot(processed_dir / "features.csv")
    return staged


def _copy_file_bytes_atomically(source: Path, target: Path) -> None:
    """Replace a staged file with an exact accepted byte copy."""
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _preserve_staged_processed_history(
    *,
    accepted_dir: Path,
    staged_dir: Path,
) -> dict[str, object]:
    """Make the rebuilt processed generation append-only across both CSVs."""
    from src.data.io_utils import write_csv_atomically
    from src.data.provider_consumer_references import snapshot_provider_references, record_provider_consumer_artifact
    from src.data.observation_consumer_references import snapshot_observation_references, record_observation_consumer_artifact

    planned: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]] = {}
    provider_parents = {}
    observation_parents = {}
    artifact_reports: dict[str, dict[str, object]] = {}
    accepted_file_identities: dict[str, dict[str, object]] = {}
    staged_before_identities: dict[str, dict[str, object]] = {}
    try:
        for filename in ("fights_cleaned.csv", "features.csv"):
            accepted_path = accepted_dir / filename
            candidate_path = staged_dir / filename
            provider_parents[filename] = [ref for ref in (
                snapshot_provider_references(accepted_path), snapshot_provider_references(candidate_path)
            ) if ref is not None]
            observation_parents[filename] = [ref for ref in (
                snapshot_observation_references(accepted_path), snapshot_observation_references(candidate_path)
            ) if ref is not None]
            accepted_file_identities[filename] = _file_content_identity(accepted_path)
            staged_before_identities[filename] = _file_content_identity(candidate_path)
            accepted = read_csv_required(accepted_path, label=f"accepted_{filename}")
            candidate = read_csv_required(candidate_path, label=f"candidate_{filename}")
            preserved, report = preserve_accepted_processed_history(
                accepted,
                candidate,
                label=filename.removesuffix(".csv"),
            )
            artifact_reports[filename] = report
            planned[filename] = (accepted, candidate, preserved, report)
    except Exception as exc:
        return {
            "passed": False,
            "errors": [
                "processed_history_preservation_execution_failed:"
                f"{type(exc).__name__}:{exc}"
            ],
            "new_identity_count": 0,
            "byte_copied_artifacts": [],
            "artifacts": artifact_reports,
        }

    errors = [
        error
        for report in artifact_reports.values()
        for error in report.get("errors", [])
    ]
    fight_new_identities = set(
        artifact_reports["fights_cleaned.csv"].get("new_identities") or []
    )
    feature_new_identities = set(
        artifact_reports["features.csv"].get("new_identities") or []
    )
    if (
        artifact_reports["fights_cleaned.csv"].get("accepted_identity_count")
        != artifact_reports["features.csv"].get("accepted_identity_count")
        or artifact_reports["fights_cleaned.csv"].get("accepted_identity_sha256")
        != artifact_reports["features.csv"].get("accepted_identity_sha256")
    ):
        errors.append("processed_artifact_accepted_identity_mismatch")
    if fight_new_identities != feature_new_identities:
        errors.append("processed_artifact_new_identity_mismatch")
    if errors:
        return {
            "passed": False,
            "errors": errors,
            "new_identity_count": len(fight_new_identities),
            "byte_copied_artifacts": [],
            "artifacts": artifact_reports,
        }

    byte_copied: list[str] = []
    try:
        for filename, (accepted, candidate, preserved, report) in planned.items():
            accepted_path = accepted_dir / filename
            staged_path = staged_dir / filename
            if (
                int(report.get("new_identity_count") or 0) == 0
                and list(accepted.columns) == list(candidate.columns)
                and len(accepted) == len(candidate)
            ):
                _copy_file_bytes_atomically(accepted_path, staged_path)
                byte_copied.append(filename)
            else:
                write_csv_atomically(preserved, staged_path, refuse_empty=True)
            if provider_parents[filename]:
                record_provider_consumer_artifact(staged_path, parent_references=provider_parents[filename],
                    history_root=staged_dir, kind="preserved_rebuilt_artifact")
            if observation_parents[filename]:
                record_observation_consumer_artifact(staged_path, parent_references=observation_parents[filename],
                    kind="preserved_rebuilt_artifact")
    except Exception as exc:
        return {
            "passed": False,
            "errors": [
                "processed_history_preservation_write_failed:"
                f"{type(exc).__name__}:{exc}"
            ],
            "new_identity_count": len(fight_new_identities),
            "byte_copied_artifacts": byte_copied,
            "artifacts": artifact_reports,
        }

    accepted_after_identities = {
        filename: _file_content_identity(accepted_dir / filename)
        for filename in planned
    }
    staged_after_identities = {
        filename: _file_content_identity(staged_dir / filename)
        for filename in planned
    }
    if accepted_after_identities != accepted_file_identities:
        return {
            "passed": False,
            "errors": ["accepted_processed_changed_during_preservation"],
            "new_identity_count": len(fight_new_identities),
            "byte_copied_artifacts": byte_copied,
            "artifacts": artifact_reports,
        }
    for filename, report in artifact_reports.items():
        report["accepted_file"] = accepted_file_identities[filename]
        report["staged_before"] = staged_before_identities[filename]
        report["staged_after"] = staged_after_identities[filename]
        report["staged_matches_accepted_bytes"] = (
            staged_after_identities[filename] == accepted_file_identities[filename]
        )
    post_write_errors = [
        f"{filename}:no_change_bytes_differ_from_accepted"
        for filename, report in artifact_reports.items()
        if int(report.get("new_identity_count") or 0) == 0
        and staged_after_identities[filename] != accepted_file_identities[filename]
    ]
    if post_write_errors:
        return {
            "passed": False,
            "errors": post_write_errors,
            "new_identity_count": len(fight_new_identities),
            "new_identities": sorted(fight_new_identities),
            "byte_copied_artifacts": byte_copied,
            "artifacts": artifact_reports,
        }

    return {
        "passed": True,
        "errors": [],
        "new_identity_count": len(fight_new_identities),
        "new_identities": sorted(fight_new_identities),
        "restored_row_count": sum(
            int(report.get("restored_row_count") or 0)
            for report in artifact_reports.values()
        ),
        "restored_field_count": sum(
            int(report.get("restored_field_count") or 0)
            for report in artifact_reports.values()
        ),
        "byte_copied_artifacts": byte_copied,
        "artifacts": artifact_reports,
    }


def _preserve_rebuild_processed_history(
    *,
    accepted_dir: Path,
    staged_root: Path,
    rebuild_summary: dict[str, object],
) -> dict[str, object]:
    """Reconcile every directory materialized by one full-history rebuild."""
    raw_outputs = rebuild_summary.get("outputs")
    outputs = raw_outputs if isinstance(raw_outputs, list) else []
    if not outputs:
        return {
            "passed": False,
            "errors": ["processed_history_preservation_missing_rebuild_outputs"],
            "output_reports": {},
        }

    root = staged_root.resolve(strict=False)
    target_dirs: list[Path] = []
    seen: set[str] = set()
    for output in outputs:
        output = output if isinstance(output, dict) else {}
        raw_output_dir = str(output.get("output_dir") or "").strip()
        target = Path(raw_output_dir) if raw_output_dir else root
        if not target.is_absolute():
            target = root / target
        target = target.resolve(strict=False)
        if target != root and root not in target.parents:
            return {
                "passed": False,
                "errors": [f"processed_history_output_path_escape:{target}"],
                "output_reports": {},
            }
        identity = os.path.normcase(str(target))
        if identity in seen:
            continue
        seen.add(identity)
        target_dirs.append(target)
    if os.path.normcase(str(root)) not in seen:
        return {
            "passed": False,
            "errors": ["processed_history_preservation_missing_base_output"],
            "output_reports": {},
        }

    output_reports: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    expected_new_identities: set[str] | None = None
    for target in target_dirs:
        report = _preserve_staged_processed_history(
            accepted_dir=accepted_dir,
            staged_dir=target,
        )
        relative = "." if target == root else target.relative_to(root).as_posix()
        output_reports[relative] = report
        if report.get("passed") is not True:
            errors.extend(f"{relative}:{error}" for error in report.get("errors", []))
            continue
        new_identities = set(report.get("new_identities") or [])
        if expected_new_identities is None:
            expected_new_identities = new_identities
        elif new_identities != expected_new_identities:
            errors.append(f"{relative}:processed_output_new_identity_mismatch")

    return {
        "passed": not errors,
        "errors": errors,
        "output_count": len(target_dirs),
        "new_identity_count": len(expected_new_identities or set()),
        "new_identities": sorted(expected_new_identities or set()),
        "output_reports": output_reports,
    }


def _append_outcome_reason(reasons: list[str], reason: str) -> None:
    reason = str(reason or "").strip()
    if reason and reason not in reasons:
        reasons.append(reason)


def _refresh_outcome_reasons(
    *,
    limit_fighters: int | None,
    roster_summary: dict[str, object],
    backfill_summary: dict[str, object],
    profile_supplement_summary: dict[str, object] | None,
    rebuild_summary: dict[str, object] | None,
    staged_processed_dir: Path | None,
    audit_summary: dict[str, object] | None,
    row_drop_guard: dict[str, object],
    generation_cleanup: dict[str, object],
    identity_integrity_guard: dict[str, object] | None = None,
    skip_rebuild: bool,
    skip_audit: bool,
    upcoming_summary: dict[str, object] | None = None,
) -> list[str]:
    """Return bounded, operator-visible reasons an attempt cannot publish."""
    reasons: list[str] = []
    if limit_fighters is not None:
        _append_outcome_reason(reasons, "partial_refresh_limit")

    roster_source = str(roster_summary.get("source") or "").strip().casefold()
    membership_source = str(
        roster_summary.get("membership_source") or ""
    ).strip().casefold()
    if int(roster_summary.get("rows") or 0) <= 0:
        _append_outcome_reason(reasons, "roster_empty")
    if roster_source != "rosterwatch":
        _append_outcome_reason(reasons, f"roster_source_{roster_source or 'unknown'}")
    if membership_source != "rosterwatch":
        _append_outcome_reason(
            reasons,
            f"roster_membership_source_{membership_source or 'unknown'}",
        )
    if roster_summary.get("used_cached_fallback") is True:
        _append_outcome_reason(reasons, "rosterwatch_cached_fallback")
    if roster_summary.get("acquisition_complete") is not True:
        _append_outcome_reason(reasons, "rosterwatch_acquisition_incomplete")
    if roster_summary.get("validation_complete") is not True:
        _append_outcome_reason(reasons, "rosterwatch_validation_incomplete")
    if roster_summary.get("reconciliation_complete") is not True:
        _append_outcome_reason(reasons, "rosterwatch_reconciliation_incomplete")
    if roster_summary.get("sync_complete") is not True:
        _append_outcome_reason(reasons, "roster_scan_incomplete")
    if str(roster_summary.get("sync_error") or "").strip():
        _append_outcome_reason(reasons, "roster_sync_error")
    if str(roster_summary.get("identity_audit_write_error") or "").strip():
        _append_outcome_reason(reasons, "roster_identity_audit_write_error")

    upcoming = upcoming_summary or {}
    if upcoming_summary is not None and upcoming.get("scan_complete") is not True:
        _append_outcome_reason(reasons, "upcoming_card_scan_incomplete")
    if str(upcoming.get("source_error") or "").strip():
        _append_outcome_reason(reasons, "upcoming_card_source_failure")

    if int(backfill_summary.get("scraped_profile_scrape_failures") or 0):
        _append_outcome_reason(reasons, "ufcstats_profile_scrape_failures")
    if backfill_summary.get("scraped_profile_rate_bundle_blocked") is True:
        _append_outcome_reason(reasons, "ufcstats_suspicious_profile_rate_bundles")
    if int(backfill_summary.get("fighter_fight_list_scrape_failures") or 0):
        _append_outcome_reason(reasons, "ufcstats_fight_list_failures")
    status_counts = backfill_summary.get("fight_detail_status_counts") or {}
    status_counts = status_counts if isinstance(status_counts, dict) else {}
    failed_urls = backfill_summary.get("failed_fight_urls") or []
    partial_urls = backfill_summary.get("partial_fight_urls") or []
    source_deferred_urls = backfill_summary.get("source_deferred_fight_urls") or []
    failed = max(
        len(failed_urls) if isinstance(failed_urls, list) else 0,
        int(status_counts.get("failed") or 0),
    )
    partial = max(
        len(partial_urls) if isinstance(partial_urls, list) else 0,
        int(status_counts.get("partial") or 0),
    )
    source_deferred = max(
        len(source_deferred_urls) if isinstance(source_deferred_urls, list) else 0,
        int(status_counts.get("source_deferred") or 0),
    )
    complete = (
        int(status_counts.get("complete") or 0)
        if "complete" in status_counts
        else int(backfill_summary.get("new_result_rows") or 0)
    )
    if failed:
        _append_outcome_reason(reasons, "ufcstats_fight_detail_failures")
    if partial:
        _append_outcome_reason(reasons, "ufcstats_partial_fight_observations")
    due = int(backfill_summary.get("missing_fight_urls_found") or 0)
    if due > complete + source_deferred + partial + failed:
        _append_outcome_reason(reasons, "ufcstats_fight_attempts_incomplete")
    if due > 0 and complete <= 0 and source_deferred < due:
        _append_outcome_reason(reasons, "ufcstats_zero_completed_fights")
    if (
        int(backfill_summary.get("new_result_rows") or 0) > 0
        and int(backfill_summary.get("new_stat_rows") or 0) <= 0
    ):
        _append_outcome_reason(reasons, "ufcstats_results_without_stats")

    supplement = profile_supplement_summary or {}
    if str(supplement.get("action") or "").strip().casefold() == "error":
        _append_outcome_reason(reasons, "profile_supplement_error")
    source_errors = supplement.get("source_errors") or {}
    if int(supplement.get("source_error_count") or 0) or any(
        int(count or 0) > 0 for count in source_errors.values()
    ):
        _append_outcome_reason(reasons, "profile_supplement_source_errors")

    if skip_rebuild:
        _append_outcome_reason(reasons, "rebuild_skipped")
    elif rebuild_summary is None:
        _append_outcome_reason(reasons, "rebuild_missing")
    preservation = (
        rebuild_summary.get("historical_processed_preservation")
        if isinstance(rebuild_summary, dict)
        else None
    )
    if isinstance(preservation, dict) and preservation.get("passed") is not True:
        _append_outcome_reason(reasons, "historical_processed_preservation_failed")
    if staged_processed_dir is not None:
        for filename, reason in (
            ("fights_cleaned.csv", "staged_fights_missing_or_empty"),
            ("features.csv", "staged_features_missing_or_empty"),
        ):
            snapshot = _file_snapshot(staged_processed_dir / filename)
            if snapshot.get("exists") is not True or int(snapshot.get("row_count") or 0) <= 0:
                _append_outcome_reason(reasons, reason)
    if skip_audit:
        _append_outcome_reason(reasons, "audit_skipped")
    elif audit_summary is None:
        _append_outcome_reason(reasons, "audit_missing")
    if row_drop_guard.get("violations"):
        _append_outcome_reason(reasons, "row_drop_guard_violations")
    if generation_cleanup.get("failures"):
        _append_outcome_reason(reasons, "refresh_generation_cleanup_failures")
    if (identity_integrity_guard or {}).get("violations"):
        _append_outcome_reason(
            reasons,
            "processed_identity_integrity_guard_violations",
        )
    return reasons


def _identity_row_index(
    rows: list[dict[str, object]],
) -> dict[str, list[int]]:
    by_key: dict[str, list[int]] = {}
    for position, row in enumerate(rows):
        for key in _roster_identity_keys(row):
            by_key.setdefault(key, []).append(position)
    return by_key


def _identity_row_sort_key(
    row: dict[str, object],
) -> tuple[str, ...]:
    return tuple(sorted(_roster_identity_keys(row)))


def _identity_strong_key_count(row: dict[str, object]) -> int:
    return sum(
        key.startswith(("url:", "ufcstats_url:"))
        for key in _roster_identity_keys(row)
    )


def _identity_match_strength(
    source_row: dict[str, object],
    target_row: dict[str, object],
) -> int:
    common_keys = (
        _roster_identity_keys(source_row)
        & _roster_identity_keys(target_row)
    )
    if any(key.startswith("url:") for key in common_keys):
        return 3
    if any(key.startswith("ufcstats_url:") for key in common_keys):
        return 2
    return 1 if common_keys else 0


def _maximum_identity_row_matching(
    source_rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
    *,
    reviewed_transition_direction: str = "",
) -> dict[int, int]:
    """Return a deterministic maximum one-to-one identity matching.

    Each new source is matched only through an augmenting path, so after every
    iteration the matching is maximum for the processed prefix. Strong-ID rows
    are processed first; a later weak row can move one only when the strong row
    remains matched elsewhere, preventing an equal-cardinality name-only match
    from displacing the sole URL-constrained match.
    """
    if reviewed_transition_direction not in {
        "",
        "source_to_target",
        "target_to_source",
    }:
        raise ValueError("Unsupported reviewed identity transition direction")

    def rows_match(
        source_row: dict[str, object], target_row: dict[str, object]
    ) -> bool:
        if _roster_rows_same_identity(source_row, target_row):
            return True
        if reviewed_transition_direction == "source_to_target":
            return reviewed_official_identity_transition_match(
                target_row, source_row
            )
        if reviewed_transition_direction == "target_to_source":
            return reviewed_official_identity_transition_match(
                source_row, target_row
            )
        return False

    target_index = _identity_row_index(target_rows)
    target_keys = [
        _identity_row_sort_key(row)
        for row in target_rows
    ]
    adjacency: dict[int, list[int]] = {}
    for source_position, source_row in enumerate(source_rows):
        candidate_positions: set[int] = set()
        for key in _roster_identity_keys(source_row):
            candidate_positions.update(target_index.get(key, ()))
        adjacency[source_position] = sorted(
            (
                target_position
                for target_position in candidate_positions
                if rows_match(source_row, target_rows[target_position])
            ),
            key=lambda target_position: (
                -_identity_match_strength(
                    source_row,
                    target_rows[target_position],
                ),
                target_keys[target_position],
                target_position,
            ),
        )

    source_order = sorted(
        range(len(source_rows)),
        key=lambda source_position: (
            -_identity_strong_key_count(source_rows[source_position]),
            len(adjacency[source_position]),
            _identity_row_sort_key(source_rows[source_position]),
            source_position,
        ),
    )
    target_owner: dict[int, int] = {}
    source_target: dict[int, int] = {}
    for starting_source in source_order:
        queued_sources = deque([starting_source])
        visited_sources = {starting_source}
        target_predecessor: dict[int, int] = {}
        free_target: int | None = None

        while queued_sources and free_target is None:
            source_position = queued_sources.popleft()
            for target_position in adjacency[source_position]:
                if target_position in target_predecessor:
                    continue
                target_predecessor[target_position] = source_position
                owner = target_owner.get(target_position)
                if owner is None:
                    free_target = target_position
                    break
                if owner not in visited_sources:
                    visited_sources.add(owner)
                    queued_sources.append(owner)

        if free_target is None:
            continue

        target_position = free_target
        while True:
            source_position = target_predecessor[target_position]
            previous_target = source_target.get(source_position)
            source_target[source_position] = target_position
            target_owner[target_position] = source_position
            if previous_target is None:
                break
            target_position = previous_target

    return source_target


def _unmatched_identity_rows(
    source_rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return source identities without a one-to-one match in the target."""
    matching = _maximum_identity_row_matching(
        source_rows,
        target_rows,
        reviewed_transition_direction="source_to_target",
    )
    return [
        row
        for position, row in enumerate(source_rows)
        if position not in matching
    ]


def _partition_explained_identity_rows(
    missing_rows: list[dict[str, object]],
    explanations: dict[str, list[dict[str, object]]],
) -> tuple[dict[str, int], list[dict[str, object]]]:
    """Consume exact missing identities using one-to-one lifecycle explanations."""
    explanation_rows: list[dict[str, object]] = []
    explanation_row_reasons: list[str] = []
    for reason in sorted(explanations):
        for row in explanations[reason]:
            if not isinstance(row, dict):
                continue
            explanation_rows.append(row)
            explanation_row_reasons.append(str(reason))

    matching = _maximum_identity_row_matching(
        explanation_rows,
        missing_rows,
        reviewed_transition_direction="target_to_source",
    )
    reason_counts: dict[str, int] = {}
    for source_position in sorted(matching):
        reason = explanation_row_reasons[source_position]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    explained_missing_positions = set(matching.values())
    return (
        reason_counts,
        [
            missing_rows[position]
            for position in range(len(missing_rows))
            if position not in explained_missing_positions
        ],
    )


def _identity_row_preview(
    rows: list[dict[str, object]],
    *,
    limit: int = 20,
) -> list[dict[str, str]]:
    return [
        {
            "official_name": str(row.get("official_name") or "").strip(),
            "official_athlete_url": str(
                row.get("official_athlete_url") or ""
            ).strip(),
            "ufcstats_url": str(row.get("ufcstats_url") or "").strip(),
        }
        for row in rows[:limit]
    ]


def _build_row_drop_guard(
    pre_state: dict[str, dict[str, object]],
    post_state: dict[str, dict[str, object]],
    *,
    explained_drop_reasons: dict[str, dict[str, int]] | None = None,
    explained_identity_reasons: (
        dict[str, dict[str, list[dict[str, object]]]] | None
    ) = None,
) -> dict[str, object]:
    violations: list[dict[str, object]] = []
    explained_drops: list[dict[str, object]] = []
    explained_drop_reasons = explained_drop_reasons or {}
    explained_identity_reasons = explained_identity_reasons or {}
    for name, pre in pre_state.items():
        post = post_state.get(name) or {}
        pre_rows = pre.get("row_count")
        post_rows = post.get("row_count")
        if pre_rows is None or post_rows is None:
            continue
        try:
            pre_count = int(pre_rows)
            post_count = int(post_rows)
        except Exception:
            continue
        rows_lost = max(int(pre_count - post_count), 0)
        pre_identity_rows = pre.get("identity_rows")
        post_identity_rows = post.get("identity_rows")
        if (
            name in explained_identity_reasons
            and isinstance(pre_identity_rows, list)
            and isinstance(post_identity_rows, list)
        ):
            missing_identity_rows = _unmatched_identity_rows(
                pre_identity_rows,
                post_identity_rows,
            )
            reason_counts, unexpected_identity_rows = (
                _partition_explained_identity_rows(
                    missing_identity_rows,
                    explained_identity_reasons.get(name) or {},
                )
            )
            explained_identity_rows = (
                len(missing_identity_rows)
                - len(unexpected_identity_rows)
            )
            unexplained_net_rows = max(
                rows_lost - explained_identity_rows,
                0,
            )
            unexpected_rows_lost = max(
                len(unexpected_identity_rows),
                unexplained_net_rows,
            )
            if not missing_identity_rows and not rows_lost:
                continue

            detail: dict[str, object] = {
                "artifact": name,
                "path": (
                    post.get("path")
                    or pre.get("path")
                    or str(_row_drop_guard_files().get(name, ""))
                ),
                "pre_rows": pre_count,
                "post_rows": post_count,
                "rows_lost": rows_lost,
                "identity_rows_lost": len(missing_identity_rows),
            }
            if explained_identity_rows:
                if rows_lost:
                    detail["explained_rows_lost"] = min(
                        rows_lost,
                        explained_identity_rows,
                    )
                detail["explained_identity_rows_lost"] = (
                    explained_identity_rows
                )
                detail["explanations"] = reason_counts
            if unexpected_rows_lost:
                detail["unexpected_rows_lost"] = unexpected_rows_lost
                detail["unexpected_identity_rows_lost"] = len(
                    unexpected_identity_rows
                )
                if unexpected_identity_rows:
                    detail["unexpected_identities"] = (
                        _identity_row_preview(unexpected_identity_rows)
                    )
                violations.append(detail)
            else:
                explained_drops.append(detail)
            continue

        if post_count < pre_count:

            reasons: dict[str, int] = {}
            for reason, raw_count in (
                explained_drop_reasons.get(name) or {}
            ).items():
                try:
                    count = max(int(raw_count), 0)
                except (TypeError, ValueError):
                    continue
                if count:
                    reasons[str(reason)] = count
            explained_capacity = sum(reasons.values())
            explained_rows_lost = min(rows_lost, explained_capacity)
            unexpected_rows_lost = rows_lost - explained_rows_lost
            detail: dict[str, object] = {
                "artifact": name,
                "path": post.get("path") or pre.get("path") or str(_row_drop_guard_files().get(name, "")),
                "pre_rows": pre_count,
                "post_rows": post_count,
                "rows_lost": rows_lost,
            }
            if explained_rows_lost:
                detail["explained_rows_lost"] = explained_rows_lost
                detail["explanations"] = reasons
            if unexpected_rows_lost:
                if explained_rows_lost:
                    detail["unexpected_rows_lost"] = unexpected_rows_lost
                violations.append(detail)
            else:
                explained_drops.append(detail)

    return {
        "ok": not violations,
        "checked_artifacts": len(pre_state),
        "violations": violations,
        "explained_drops": explained_drops,
    }


def _seed_stale_scraped_fighters(
    *, runtime_path: Path | None = None
) -> dict[str, object]:
    """If running on hosted and the image has a richer scraped-fighters file, merge it in.

    The entrypoint's copy_tree_missing only seeds files that don't exist on the volume,
    so subsequent deploys with locally-enriched data never update the volume copy.
    This function detects that scenario and merges the image data.
    """
    image_path = _IMAGE_RAW_DIR / "ufc_fighters_scraped.csv"
    runtime_path = Path(runtime_path) if runtime_path is not None else BACKFILL_FIGHTERS_PATH

    if not image_path.exists() or image_path == runtime_path or image_path.resolve() == runtime_path.resolve():
        return {"action": "skip", "reason": "no image/runtime divergence"}

    if not runtime_path.exists():
        return {"action": "skip", "reason": "runtime file missing (entrypoint should have seeded it)"}

    try:
        image_df = pd.read_csv(image_path)
        runtime_df = pd.read_csv(runtime_path)
    except Exception as exc:
        logger.warning("Failed to compare image vs runtime scraped fighters: %s", exc)
        return {"action": "error", "reason": str(exc)}

    image_rows = len(image_df)
    runtime_rows = len(runtime_df)

    # Count encoder-valid stance values as the key enrichment signal.
    def _stance_count(df: pd.DataFrame) -> int:
        if "stance" not in df.columns:
            return 0
        return int(df["stance"].apply(_valid_stance).sum())

    image_stance = _stance_count(image_df)
    runtime_stance = _stance_count(runtime_df)

    if image_stance <= runtime_stance and image_rows <= runtime_rows:
        logger.info(
            "Scraped fighters volume copy is current: runtime=%d rows/%d stance, image=%d rows/%d stance",
            runtime_rows, runtime_stance, image_rows, image_stance,
        )
        return {
            "action": "skip",
            "reason": "runtime copy is at least as rich as image",
            "runtime_rows": runtime_rows,
            "runtime_stance": runtime_stance,
            "image_rows": image_rows,
            "image_stance": image_stance,
        }

    # Merge: take runtime as base, fill in any new rows or enriched fields from image
    from src.data.io_utils import write_csv_atomically

    runtime_urls = set(
        runtime_df.get("fighter_url", pd.Series(dtype="object"))
        .fillna("").astype(str).str.strip()
    ) - {""}
    image_new = image_df[
        ~image_df.get("fighter_url", pd.Series(dtype="object"))
        .fillna("").astype(str).str.strip()
        .isin(runtime_urls)
    ]
    merged = pd.concat([runtime_df, image_new], ignore_index=True, sort=False)
    merged = merged.drop_duplicates(subset=["fighter_url"], keep="first")

    # For existing rows, fill blank stance/reach/height from image where available
    image_by_url = {}
    for _, row in image_df.iterrows():
        url = str(row.get("fighter_url") or "").strip()
        if url:
            image_by_url[url] = row

    updated_fields = 0
    for idx, row in merged.iterrows():
        url = str(row.get("fighter_url") or "").strip()
        image_row = image_by_url.get(url)
        if image_row is None:
            continue
        for field in ("stance", "reach", "height", "weight", "dob"):
            current = row.get(field)
            if _field_has_report_value(field, current):
                continue
            image_val = image_row.get(field)
            if _field_has_report_value(field, image_val):
                merged.at[idx, field] = image_val
                updated_fields += 1

    write_csv_atomically(merged, runtime_path, refuse_empty=True)
    merged_stance = _stance_count(merged)

    logger.info(
        "Seeded stale scraped fighters from image: runtime %d→%d rows, stance %d→%d, updated %d fields, added %d new rows",
        runtime_rows, len(merged), runtime_stance, merged_stance, updated_fields, len(image_new),
    )
    return {
        "action": "merged",
        "runtime_rows_before": runtime_rows,
        "runtime_rows_after": len(merged),
        "runtime_stance_before": runtime_stance,
        "runtime_stance_after": merged_stance,
        "new_rows_from_image": len(image_new),
        "updated_fields": updated_fields,
    }


def _profile_supplement_field_count(df: pd.DataFrame) -> int:
    total = 0
    for field in PROFILE_SUPPLEMENT_TARGET_FIELDS:
        if field not in df.columns:
            continue
        total += int(df[field].apply(lambda value: _field_has_report_value(field, value)).sum())
    return total


def _seed_stale_profile_supplement(
    *, runtime_path: Path | None = None
) -> dict[str, object]:
    """Merge image-bundled profile supplement rows into the hosted runtime volume.

    GitHub-hosted refresh jobs can update the repository copy, but Railway keeps
    raw data on a persistent volume. This carries richer deployed supplement rows
    into that volume without overwriting runtime-only rows.
    """
    image_path = _IMAGE_RAW_DIR / "ufc_fighters_profile_supplement.csv"
    runtime_path = Path(runtime_path) if runtime_path is not None else PROFILE_SUPPLEMENT_PATH

    if not image_path.exists() or image_path == runtime_path or image_path.resolve() == runtime_path.resolve():
        return {"action": "skip", "reason": "no image/runtime divergence"}

    try:
        image_df = pd.read_csv(image_path)
        runtime_df = pd.read_csv(runtime_path) if runtime_path.exists() else pd.DataFrame()
    except Exception as exc:
        logger.warning("Failed to compare image vs runtime profile supplement: %s", exc)
        return {"action": "error", "reason": str(exc)}

    image_rows = image_df.to_dict(orient="records") if not image_df.empty else []
    runtime_rows = runtime_df.to_dict(orient="records") if not runtime_df.empty else []
    runtime_keys = {
        key for row in runtime_rows if (key := _profile_supplement_source_row_key(row))[0] and key[1]
    }
    runtime_by_key = {
        key: row for row in runtime_rows if (key := _profile_supplement_source_row_key(row))[0] and key[1]
    }

    combined_rows = _merge_profile_supplement_source_rows(runtime_rows, image_rows)
    merged_df = pd.DataFrame(combined_rows)
    if not merged_df.empty and {"name", "source"}.issubset(merged_df.columns):
        merged_df = merged_df.sort_values(["name", "source"]).reset_index(drop=True)
    combined_by_key = {
        key: row
        for row in combined_rows
        if (key := _profile_supplement_source_row_key(row))[0] and key[1]
    }

    image_new_keys = {
        key
        for row in image_rows
        if (key := _profile_supplement_source_row_key(row))[0] and key[1] and key not in runtime_keys
    }
    updated_fields = 0
    for row in image_rows:
        key = _profile_supplement_source_row_key(row)
        runtime_row = runtime_by_key.get(key)
        if runtime_row is None:
            continue
        for field in PROFILE_SUPPLEMENT_TARGET_FIELDS:
            if _field_has_report_value(field, runtime_row.get(field)):
                continue
            if _field_has_report_value(field, row.get(field)):
                updated_fields += 1

    sanitized_fields = 0
    for key, runtime_row in runtime_by_key.items():
        combined_row = combined_by_key.get(key)
        if combined_row is None:
            continue
        for field in PROFILE_SUPPLEMENT_TARGET_FIELDS:
            runtime_value = runtime_row.get(field)
            combined_value = combined_row.get(field)
            if (
                not _blank(runtime_value)
                and not _field_has_report_value(field, runtime_value)
                and _blank(combined_value)
            ):
                sanitized_fields += 1

    if not image_new_keys and updated_fields == 0 and sanitized_fields == 0:
        logger.info(
            "Profile supplement volume copy is current: runtime=%d rows/%d fields, image=%d rows/%d fields",
            len(runtime_df),
            _profile_supplement_field_count(runtime_df),
            len(image_df),
            _profile_supplement_field_count(image_df),
        )
        return {
            "action": "skip",
            "reason": "runtime copy already contains image supplement rows",
            "runtime_rows": int(len(runtime_df)),
            "runtime_field_count": _profile_supplement_field_count(runtime_df),
            "image_rows": int(len(image_df)),
            "image_field_count": _profile_supplement_field_count(image_df),
        }

    from src.data.io_utils import write_csv_atomically

    write_csv_atomically(merged_df, runtime_path, refuse_empty=True)
    logger.info(
        (
            "Seeded stale profile supplement from image: runtime %d->%d rows, "
            "updated %d fields, added %d new rows, sanitized %d invalid fields"
        ),
        len(runtime_df),
        len(merged_df),
        updated_fields,
        len(image_new_keys),
        sanitized_fields,
    )
    return {
        "action": "merged",
        "runtime_rows_before": int(len(runtime_df)),
        "runtime_rows_after": int(len(merged_df)),
        "runtime_field_count_before": _profile_supplement_field_count(runtime_df),
        "runtime_field_count_after": _profile_supplement_field_count(merged_df),
        "image_rows": int(len(image_df)),
        "image_field_count": _profile_supplement_field_count(image_df),
        "new_rows_from_image": int(len(image_new_keys)),
        "updated_fields": int(updated_fields),
        "sanitized_fields": int(sanitized_fields),
    }


_DESTRUCTIVE_ROSTER_AUDIT_ACTIONS = frozenset(
    {
        "excluded_test_profile",
        "excluded_inactive_profile_status",
        "excluded_non_mma_identity",
    }
)

# The reviewed Anderson route transition is unusual: the accepted audit contains
# one older, target-route inactive certificate from before the stable carrier was
# discovered.  A later partial live crawl can produce the same target certificate
# with the reviewed UFCStats identity attached.  Only this exact identity may
# replace that older blank-carrier certificate during cached-fallback
# materialization.  Keeping the allowlist local and identity-bound avoids turning
# audit merging into a general name/slug deduplication mechanism.
_REVIEWED_INACTIVE_AUDIT_SUPERSESSION_STABLE_IDS = frozenset(
    {"bbc9ab95fa766493"}
)


def _identity_audit_records_without_generation(
    frame: pd.DataFrame,
) -> list[dict[str, object]]:
    columns = [
        column
        for column in IDENTITY_AUDIT_COLUMNS
        if column != ACTIVE_ROSTER_SYNC_GENERATION_COLUMN
    ]
    return [
        {column: row.get(column, "") for column in columns}
        for row in frame.to_dict(orient="records")
    ]


def _staged_identity_audit_rows(
    *,
    staged_roster_path: Path,
    staged_audit_path: Path,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Load only a generation-consistent staged accepted roster/audit pair."""

    try:
        staged_roster = pd.read_csv(
            staged_roster_path,
            dtype=object,
            keep_default_na=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not read the staged accepted roster before fallback reconciliation"
        ) from exc
    roster_generation = _single_complete_sync_generation(staged_roster)
    if not staged_audit_path.exists():
        return staged_roster, []
    try:
        staged_audit = pd.read_csv(
            staged_audit_path,
            dtype=object,
            keep_default_na=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not read the staged accepted roster identity audit before "
            "fallback reconciliation"
        ) from exc
    missing_columns = sorted(set(IDENTITY_AUDIT_COLUMNS) - set(staged_audit.columns))
    if missing_columns:
        raise RuntimeError(
            "Staged accepted roster identity audit is missing required columns: "
            + ", ".join(missing_columns)
        )
    if not staged_audit.empty:
        audit_generation = _single_complete_sync_generation(staged_audit)
        if not roster_generation or audit_generation != roster_generation:
            raise RuntimeError(
                "Staged accepted roster and identity audit do not have one matching "
                "complete generation"
            )
    return staged_roster, _identity_audit_records_without_generation(staged_audit)


def _audit_row_matches_any_roster_row(
    audit_row: dict[str, object],
    roster_df: pd.DataFrame,
) -> bool:
    return bool(
        _string_value(audit_row.get("action"))
        in _DESTRUCTIVE_ROSTER_AUDIT_ACTIONS
        and any(
            _roster_row_matches_excluded_identity(row, audit_row)
            for _, row in roster_df.iterrows()
        )
    )


def _normalized_audit_carrier_ids(row: dict[str, object]) -> set[str]:
    return {
        stable_id
        for field in ("ufcstats_url", "stable_ufcstats_id", "fighter_id")
        if (stable_id := normalize_ufcstats_id(row.get(field)))
    }


def _reviewed_inactive_audit_endpoint_overlaps(
    prior_row: dict[str, object],
    selected_row: dict[str, object],
    *,
    stable_id: str,
) -> bool:
    """Detect any strong overlap with one selected reviewed target certificate."""

    prior_url = _string_value(prior_row.get("official_athlete_url")).rstrip("/").casefold()
    selected_url = (
        _string_value(selected_row.get("official_athlete_url"))
        .rstrip("/")
        .casefold()
    )
    prior_name = normalize_person_name(prior_row.get("official_name"))
    selected_name = normalize_person_name(selected_row.get("official_name"))
    return bool(
        (prior_url and prior_url == selected_url)
        or (prior_name and prior_name == selected_name)
        or stable_id in _normalized_audit_carrier_ids(prior_row)
    )


def _prior_audit_positions_superseded_by_reviewed_inactive_targets(
    prior_audit_rows: list[dict[str, object]],
    selected_partial_rows: list[dict[str, object]],
) -> set[int]:
    """Select only the one reviewed blank-carrier certificate being upgraded."""

    superseded_positions: set[int] = set()
    for selected_row in selected_partial_rows:
        selected_carrier_ids = _normalized_audit_carrier_ids(selected_row)
        if len(selected_carrier_ids) != 1:
            raise RuntimeError(
                "Reconciled cached fallback selected inactive audit row does not "
                "have exactly one stable identity carrier"
            )
        stable_id = next(iter(selected_carrier_ids))
        if stable_id not in _REVIEWED_INACTIVE_AUDIT_SUPERSESSION_STABLE_IDS:
            continue

        overlapping_positions = [
            position
            for position, prior_row in enumerate(prior_audit_rows)
            if _string_value(prior_row.get("action"))
            in _DESTRUCTIVE_ROSTER_AUDIT_ACTIONS
            and _reviewed_inactive_audit_endpoint_overlaps(
                prior_row,
                selected_row,
                stable_id=stable_id,
            )
        ]
        if not overlapping_positions:
            continue
        if len(overlapping_positions) != 1:
            raise RuntimeError(
                "Reconciled cached fallback found multiple prior destructive "
                "audit rows for one reviewed inactive transition target"
            )

        position = overlapping_positions[0]
        prior_row = prior_audit_rows[position]
        prior_url = _string_value(prior_row.get("official_athlete_url")).rstrip("/")
        selected_url = _string_value(selected_row.get("official_athlete_url")).rstrip("/")
        prior_name = normalize_person_name(prior_row.get("official_name"))
        selected_name = normalize_person_name(selected_row.get("official_name"))
        if not (
            _string_value(prior_row.get("action"))
            == "excluded_inactive_profile_status"
            and prior_url == selected_url
            and prior_name
            and prior_name == selected_name
            and reviewed_mma_identity_transition_pending_stable_id(prior_row)
        ):
            raise RuntimeError(
                "Reconciled cached fallback prior destructive audit row conflicts "
                "with the reviewed inactive transition target"
            )
        if position in superseded_positions:
            raise RuntimeError(
                "Reconciled cached fallback attempted to supersede one prior "
                "destructive audit row more than once"
            )
        superseded_positions.add(position)
    return superseded_positions


def _materialize_reconciled_cached_roster_candidate(
    roster_df: pd.DataFrame,
    *,
    fallback_removals: list[dict[str, object]],
    candidate_roster_path: Path,
    candidate_audit_path: Path,
    accepted_roster_path: Path,
    accepted_audit_path: Path,
) -> pd.DataFrame:
    """Write a filtered fallback pair without re-authorizing partial exclusions."""

    for candidate_path, accepted_path, label in (
        (candidate_roster_path, accepted_roster_path, "roster"),
        (candidate_audit_path, accepted_audit_path, "identity audit"),
    ):
        if _paths_target_same_file(candidate_path, accepted_path):
            raise RuntimeError(
                "Refusing to materialize a reconciled cached fallback over the "
                f"accepted UFC {label}"
            )
    if _paths_target_same_file(candidate_roster_path, candidate_audit_path):
        raise RuntimeError(
            "Refusing to materialize the reconciled roster and identity audit to "
            "the same candidate path"
        )

    staged_roster, prior_audit_rows = _staged_identity_audit_rows(
        staged_roster_path=candidate_roster_path,
        staged_audit_path=candidate_audit_path,
    )
    expected_removal_positions: set[int] = set()
    for removal in fallback_removals:
        matches = [
            position
            for position, (_, row) in enumerate(staged_roster.iterrows())
            if _roster_rows_same_identity(row, removal)
        ]
        if len(matches) != 1 or matches[0] in expected_removal_positions:
            raise RuntimeError(
                "Reconciled cached fallback removal does not identify exactly one "
                "row in the staged accepted roster"
            )
        expected_removal_positions.add(matches[0])
    expected_candidate = staged_roster.iloc[
        [
            position
            for position in range(len(staged_roster))
            if position not in expected_removal_positions
        ]
    ].reset_index(drop=True)
    if not _semantic_csv_frame(expected_candidate).equals(
        _semantic_csv_frame(roster_df)
    ):
        raise RuntimeError(
            "Reconciled cached fallback is not exactly the staged accepted roster "
            "minus its reviewed inactive removal rows"
        )

    partial_audit_rows = [
        dict(row)
        for row in (roster_df.attrs.get("identity_audit_rows") or [])
        if isinstance(row, dict)
    ]
    selected_partial_rows: list[dict[str, object]] = []
    used_partial_positions: set[int] = set()
    for removal in fallback_removals:
        matches = [
            position
            for position, row in enumerate(partial_audit_rows)
            if _string_value(row.get("action"))
            == "excluded_inactive_profile_status"
            and reviewed_official_identity_transition_match(row, removal)
        ]
        if len(matches) != 1 or matches[0] in used_partial_positions:
            raise RuntimeError(
                "Reconciled cached fallback removal does not have exactly one "
                "matching partial-live inactive identity audit row"
            )
        used_partial_positions.add(matches[0])
        selected_partial_rows.append(partial_audit_rows[matches[0]])

    superseded_prior_positions = (
        _prior_audit_positions_superseded_by_reviewed_inactive_targets(
            prior_audit_rows,
            selected_partial_rows,
        )
    )
    combined_audit_rows: list[dict[str, object]] = []
    seen_audit_rows: set[tuple[str, ...]] = set()
    semantic_columns = [
        column
        for column in IDENTITY_AUDIT_COLUMNS
        if column != ACTIVE_ROSTER_SYNC_GENERATION_COLUMN
    ]
    retained_prior_audit_rows = [
        row
        for position, row in enumerate(prior_audit_rows)
        if position not in superseded_prior_positions
    ]
    for raw_row in [*retained_prior_audit_rows, *selected_partial_rows]:
        row = {column: raw_row.get(column, "") for column in semantic_columns}
        identity = tuple(_string_value(row.get(column)) for column in semantic_columns)
        if identity in seen_audit_rows:
            continue
        seen_audit_rows.add(identity)
        combined_audit_rows.append(row)
    for selected_row in selected_partial_rows:
        selected_carrier_ids = _normalized_audit_carrier_ids(selected_row)
        if not (
            selected_carrier_ids
            & _REVIEWED_INACTIVE_AUDIT_SUPERSESSION_STABLE_IDS
        ):
            continue
        matching_final_rows = [
            row
            for row in combined_audit_rows
            if _string_value(row.get("action"))
            in _DESTRUCTIVE_ROSTER_AUDIT_ACTIONS
            and _reviewed_inactive_audit_endpoint_overlaps(
                row,
                selected_row,
                stable_id=next(iter(selected_carrier_ids)),
            )
        ]
        if (
            len(matching_final_rows) != 1
            or _string_value(matching_final_rows[0].get("action"))
            != "excluded_inactive_profile_status"
            or _normalized_audit_carrier_ids(matching_final_rows[0])
            != selected_carrier_ids
        ):
            raise RuntimeError(
                "Reconciled candidate identity audit does not contain exactly one "
                "full reviewed inactive transition certificate"
            )
    if any(
        _audit_row_matches_any_roster_row(row, roster_df)
        for row in selected_partial_rows
    ):
        raise RuntimeError(
            "Reconciled candidate identity audit would destructively exclude an "
            "identity still present in the filtered fallback roster"
        )

    reconciled_generation_id = uuid4().hex
    reconciled_attrs = dict(roster_df.attrs)
    candidate_roster = roster_df.copy()
    candidate_roster.attrs.update(reconciled_attrs)
    candidate_roster[ACTIVE_ROSTER_SYNC_GENERATION_COLUMN] = (
        reconciled_generation_id
    )
    candidate_audit = _identity_audit_frame(
        combined_audit_rows,
        sync_generation_id=reconciled_generation_id,
    )
    write_csv_atomically(candidate_roster, candidate_roster_path, refuse_empty=True)
    try:
        write_csv_atomically(
            candidate_audit,
            candidate_audit_path,
            refuse_empty=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Reconciled candidate roster was isolated but its matching identity "
            "audit could not be written; refusing to continue toward publication"
        ) from exc
    try:
        written_roster = pd.read_csv(
            candidate_roster_path,
            dtype=object,
            keep_default_na=False,
        )
        written_audit = pd.read_csv(
            candidate_audit_path,
            dtype=object,
            keep_default_na=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Reconciled candidate roster/identity-audit pair could not be read back; "
            "refusing to continue toward publication"
        ) from exc
    if (
        _single_complete_sync_generation(written_roster)
        != reconciled_generation_id
        or _single_complete_sync_generation(written_audit)
        != reconciled_generation_id
    ):
        raise RuntimeError(
            "Reconciled candidate roster and identity audit did not retain one "
            "matching complete generation; refusing to continue toward publication"
        )
    candidate_roster.attrs["sync_generation_id"] = reconciled_generation_id
    candidate_roster.attrs["reconciled_candidate_fallback_materialized"] = True
    candidate_roster.attrs["reconciled_candidate_identity_audit_rows"] = len(
        combined_audit_rows
    )
    candidate_roster.attrs["reconciled_candidate_partial_audit_rows_selected"] = len(
        selected_partial_rows
    )
    return candidate_roster


def _roster_summary(df: pd.DataFrame, *, output_path: Path) -> dict[str, object]:
    attrs = dict(getattr(df, "attrs", {}) or {})
    profile_status = (
        df.get("profile_status", pd.Series(index=df.index, dtype="object"))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )
    status_unknown = profile_status.isin({"", "status_unknown"})
    summary = {
        "rows": int(len(df)),
        "resolved_ufcstats_urls": int(df.get("ufcstats_url", pd.Series(dtype="object")).fillna("").astype(str).str.strip().ne("").sum()),
        "resolved_ufcstats_names": int(df.get("ufcstats_name", pd.Series(dtype="object")).fillna("").astype(str).str.strip().ne("").sum()),
        "with_profile_details": int((~status_unknown).sum()),
        "output_path": str(output_path),
    }
    summary["status_unknown_rows"] = int(status_unknown.sum())
    sync_source = str(attrs.get("sync_source") or "live").strip() or "live"
    summary["source"] = sync_source
    summary["used_cached_fallback"] = bool(attrs.get("sync_fallback_used"))
    summary["sync_complete"] = attrs.get("sync_complete") is True
    summary["production_ready"] = attrs.get("production_ready") is True
    summary["identity_complete"] = attrs.get("identity_complete") is True
    summary["identity_review_path"] = str(attrs.get("identity_review_path") or "")
    summary["identity_review_sha256"] = str(attrs.get("identity_review_sha256") or "")
    review = attrs.get("identity_review")
    summary["identity_counts"] = dict(review.get("counts") or {}) if isinstance(review, Mapping) else {}
    summary["identity_recheck"] = dict(attrs.get("identity_recheck") or {})
    summary["sync_completeness_reason"] = str(
        attrs.get("sync_completeness_reason")
        or "missing_explicit_completion_evidence"
    )
    summary["membership_source"] = str(attrs.get("membership_source") or "").strip()
    summary["acquisition_complete"] = attrs.get("acquisition_complete") is True
    summary["validation_complete"] = attrs.get("validation_complete") is True
    summary["reconciliation_complete"] = attrs.get("reconciliation_complete") is True
    summary["unresolved_identity_carriers"] = int(
        attrs.get("unresolved_identity_carriers") or 0
    )
    summary["source_cutoff"] = str(attrs.get("source_cutoff") or "").strip()
    summary["request_count"] = int(attrs.get("request_count") or 0)
    summary["snapshot_path"] = str(attrs.get("snapshot_path") or "").strip()
    summary["snapshot_sha256"] = str(attrs.get("snapshot_sha256") or "").strip()
    summary["snapshot_bytes"] = int(attrs.get("snapshot_bytes") or 0)
    summary["rosterwatch_state_path"] = str(
        attrs.get("rosterwatch_state_path") or ""
    ).strip()
    delta = attrs.get("delta")
    summary["delta"] = dict(delta) if isinstance(delta, Mapping) else {}
    summary["pages_scraped"] = int(attrs.get("pages_scraped") or 0)
    summary["selected_card_count"] = int(attrs.get("selected_card_count") or 0)
    summary["parsed_card_count"] = int(attrs.get("parsed_card_count") or 0)
    sync_error = str(attrs.get("sync_error") or "").strip()
    if sync_error:
        summary["sync_error"] = sync_error
    identity_audit_write_error = str(
        attrs.get("identity_audit_write_error") or ""
    ).strip()
    if identity_audit_write_error:
        summary["identity_audit_write_error"] = identity_audit_write_error
    cached_snapshot_mtime = str(attrs.get("sync_cached_snapshot_mtime_utc") or "").strip()
    if not cached_snapshot_mtime:
        cached_snapshot_mtime = str(attrs.get("cached_snapshot_mtime_utc") or "").strip()
    if cached_snapshot_mtime:
        summary["cached_snapshot_mtime_utc"] = cached_snapshot_mtime
    cached_snapshot_age_hours = attrs.get("cached_snapshot_age_hours")
    if cached_snapshot_age_hours is not None:
        try:
            summary["cached_snapshot_age_hours"] = float(cached_snapshot_age_hours)
        except Exception:
            pass
    retained_missing_live_rows = attrs.get("retained_missing_live_rows")
    if not isinstance(retained_missing_live_rows, list):
        retained_missing_live_rows = retained_roster_diagnostics(df)
    if isinstance(retained_missing_live_rows, list):
        summary["retained_missing_live_rows"] = int(len(retained_missing_live_rows))
        summary["retained_missing_live_fighters"] = [
            row
            for row in retained_missing_live_rows
            if isinstance(row, dict)
        ][:50]
    expired_missing_live_rows = attrs.get("expired_missing_live_rows")
    if isinstance(expired_missing_live_rows, list):
        summary["expired_missing_live_rows"] = int(len(expired_missing_live_rows))
        summary["expired_missing_live_fighters"] = [
            row for row in expired_missing_live_rows if isinstance(row, dict)
        ][:50]
    intentionally_removed_cached_rows = attrs.get(
        "intentionally_removed_cached_rows"
    )
    if isinstance(intentionally_removed_cached_rows, list):
        summary["intentionally_removed_cached_rows"] = int(
            len(intentionally_removed_cached_rows)
        )
        summary["intentionally_removed_cached_fighters"] = [
            row
            for row in intentionally_removed_cached_rows
            if isinstance(row, dict)
        ][:50]
    fallback_diagnostics = attrs.get("inactive_transition_fallback_diagnostics")
    if isinstance(fallback_diagnostics, list):
        summary["inactive_transition_fallback_diagnostics"] = len(
            fallback_diagnostics
        )
        summary["inactive_transition_fallback_diagnostic_details"] = [
            row for row in fallback_diagnostics if isinstance(row, dict)
        ][:50]
    for reconciliation_attr in (
        "reconciled_simultaneous_inactive_transitions",
        "reconciled_simultaneous_active_owner_transitions",
        "reconciled_inherited_duplicate_owner_transitions",
    ):
        reconciliation_rows = attrs.get(reconciliation_attr)
        if not isinstance(reconciliation_rows, list):
            continue
        reconciliation_details = [
            row for row in reconciliation_rows if isinstance(row, dict)
        ][:50]
        summary[reconciliation_attr] = len(reconciliation_rows)
        summary[f"{reconciliation_attr}_details"] = reconciliation_details
        summary[f"{reconciliation_attr}_details_truncated"] = max(
            0,
            len(reconciliation_rows) - len(reconciliation_details),
        )
    summary["reconciled_candidate_fallback_materialized"] = bool(
        attrs.get("reconciled_candidate_fallback_materialized")
    )
    if summary["reconciled_candidate_fallback_materialized"]:
        summary["reconciled_candidate_identity_audit_rows"] = int(
            attrs.get("reconciled_candidate_identity_audit_rows") or 0
        )
        summary["reconciled_candidate_partial_audit_rows_selected"] = int(
            attrs.get("reconciled_candidate_partial_audit_rows_selected") or 0
        )
    preserved_immutable_profile_conflicts = attrs.get(
        "preserved_immutable_profile_conflicts"
    )
    if isinstance(preserved_immutable_profile_conflicts, list):
        summary["preserved_immutable_profile_conflicts"] = int(
            len(preserved_immutable_profile_conflicts)
        )
        summary["preserved_immutable_profile_conflict_details"] = [
            row
            for row in preserved_immutable_profile_conflicts
            if isinstance(row, dict)
        ][:50]
    quarantined_unbound_alternate_routes = attrs.get(
        "quarantined_unbound_simultaneous_alternate_routes"
    )
    if isinstance(quarantined_unbound_alternate_routes, list):
        summary["quarantined_unbound_simultaneous_alternate_routes"] = int(
            len(quarantined_unbound_alternate_routes)
        )
        summary["quarantined_unbound_simultaneous_alternate_route_details"] = [
            row
            for row in quarantined_unbound_alternate_routes
            if isinstance(row, dict)
        ][:50]
    try:
        discarded_suspicious_cached_rows = int(attrs.get("discarded_suspicious_cached_rows") or 0)
    except Exception:
        discarded_suspicious_cached_rows = 0
    if discarded_suspicious_cached_rows:
        summary["discarded_suspicious_cached_rows"] = discarded_suspicious_cached_rows
    identity_audit_rows = attrs.get("identity_audit_rows")
    if isinstance(identity_audit_rows, list):
        summary["identity_audit_rows"] = int(len(identity_audit_rows))
        action_counts: dict[str, int] = {}
        for row in identity_audit_rows:
            if not isinstance(row, dict):
                continue
            action = str(row.get("action") or "unknown").strip() or "unknown"
            action_counts[action] = action_counts.get(action, 0) + 1
        summary["identity_audit_action_counts"] = action_counts
    return summary


def _preserved_official_field_exclusions(
    roster_df: pd.DataFrame,
) -> set[tuple[str, str, str]]:
    """Keep cached successful-conflict values out of current-source coverage."""

    exclusions: set[tuple[str, str, str]] = set()
    details = roster_df.attrs.get("preserved_immutable_profile_conflicts")
    if not isinstance(details, list):
        return exclusions
    for detail in details:
        if not isinstance(detail, dict):
            continue
        field = _string_value(detail.get("field"))
        official_entity = source_entity_id(
            "ufc",
            detail.get("official_athlete_url"),
        )
        stable_id = normalize_ufcstats_id(detail.get("ufcstats_url")) or ""
        if field in {"height", "reach"} and official_entity:
            exclusions.add((official_entity, stable_id, field))
    return exclusions


def _write_json(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _summarize_audit_frame(frame: pd.DataFrame) -> dict[str, object]:
    total = int(len(frame))
    if total == 0:
        return {"rows": 0}

    summary: dict[str, object] = {"rows": total}
    for field in (
        "age_present",
        "weight_present",
        "division_present",
        "height_present",
        "reach_present",
        "stance_present",
        "full_physical_bundle_present",
    ):
        if field not in frame.columns:
            continue
        count = int(frame[field].sum())
        summary[field] = {
            "count": count,
            "pct": round((count / total) * 100.0, 2),
        }
    return summary


def _blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() in {"", "-", "--", "nan", "NaN", "N/A"}


def _valid_stance(value: object) -> bool:
    if _blank(value):
        return False
    return bool(pd.notna(encode_stance(value)))


def _valid_dob(value: object) -> bool:
    if _blank(value):
        return False
    return bool(pd.notna(pd.to_datetime(value, errors="coerce", format="mixed")))


def _field_has_report_value(field: str, value: object) -> bool:
    if field == "stance":
        return _valid_stance(value)
    if field == "dob":
        return _valid_dob(value)
    return not _blank(value)


def _string_value(value: object) -> str:
    return "" if _blank(value) else str(value).strip()


def _canonical_upcoming_response_metadata(
    value: object,
) -> tuple[dict[str, object] | None, str]:
    """Return JSON-canonical response metadata and its deterministic digest."""
    if value is None:
        return None, ""
    if not isinstance(value, Mapping):
        raise ValueError("not_an_object")
    if not value:
        return None, ""
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("not_json_safe") from exc
    if not isinstance(normalized, dict):
        raise ValueError("not_an_object")
    return normalized, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _summarize_upcoming_source_responses(
    value: object,
    structural_issues: list[str],
) -> list[dict[str, object]]:
    """Retain deterministic response evidence without retaining response bodies."""
    if not isinstance(value, list):
        structural_issues.append("upcoming_source_responses_not_a_list")
        return []

    evidence: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    for response in value:
        if not isinstance(response, Mapping):
            structural_issues.append("upcoming_source_response_not_an_object")
            continue

        # The live snapshot schema carries the originally requested resource in
        # `url`. Do not accept an alternate key that could relabel evidence.
        url = _string_value(response.get("url"))
        retrieval_time = _string_value(response.get("retrieval_time_utc"))
        body = response.get("body")
        if not url:
            structural_issues.append("upcoming_source_response_url_missing")
        if not retrieval_time:
            structural_issues.append(
                "upcoming_source_response_retrieval_time_missing"
            )
        if isinstance(body, str):
            raw_body = body.encode("utf-8")
        elif isinstance(body, bytes):
            raw_body = body
        else:
            structural_issues.append("upcoming_source_response_body_invalid")
            continue
        if not raw_body:
            structural_issues.append("upcoming_source_response_body_empty")
        if not url:
            continue
        if url in seen_urls:
            structural_issues.append("upcoming_source_response_url_duplicate")
        seen_urls.add(url)

        item: dict[str, object] = {
            "url": url,
            "retrieval_time_utc": retrieval_time,
            "body_sha256": hashlib.sha256(raw_body).hexdigest(),
            "byte_count": len(raw_body),
        }
        try:
            metadata, metadata_hash = _canonical_upcoming_response_metadata(
                response.get("response_metadata")
            )
        except ValueError:
            structural_issues.append(
                "upcoming_source_response_metadata_invalid"
            )
        else:
            if metadata is not None:
                item["response_metadata"] = metadata
                item["response_metadata_sha256"] = metadata_hash
        evidence.append(item)
    return evidence


def _normalized_athlete_resource(value: object) -> str:
    text = _string_value(value)
    if not text:
        return ""
    parsed = urlparse(text)
    hostname = str(parsed.hostname or "").casefold().removeprefix("www.")
    path = str(parsed.path or "").rstrip("/").casefold()
    return f"{hostname}{path}" if hostname and path else ""


def _build_upcoming_backfill_target(
    official_roster: pd.DataFrame,
    snapshot: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Add resolved announced participants only to the transient backfill target."""
    official = official_roster.copy()
    official.attrs = dict(getattr(official_roster, "attrs", {}) or {})
    official_ids: set[str] = set()
    ids_by_athlete: dict[str, set[str]] = {}
    for row in official.to_dict(orient="records"):
        recovery_binding = reviewed_mma_identity_binding(row)
        if (
            recovery_binding is None
            and reviewed_mma_identity_registry_record(row) is not None
        ):
            continue
        reviewed_id = (
            recovery_binding["stable_ufcstats_id"]
            if recovery_binding is not None
            else CURRENT_UFCSTATS_ID_OVERRIDES.get(
                normalize_person_name(row.get("official_name"))
            )
        )
        fighter_id = reviewed_id or normalize_ufcstats_id(row.get("ufcstats_url"))
        if fighter_id:
            official_ids.add(fighter_id)
        for field in (
            "official_athlete_url",
            "canonical_athlete_url",
            "athlete_url",
        ):
            athlete_key = _normalized_athlete_resource(row.get(field))
            if athlete_key and fighter_id:
                ids_by_athlete.setdefault(athlete_key, set()).add(fighter_id)

    event_cards = snapshot.get("event_cards") or []
    structural_issues: list[str] = []
    event_count = 0
    fight_count = 0
    participants_seen = 0
    resolved_count = 0
    already_in_roster = 0
    duplicate_count = 0
    source_limited: list[dict[str, str]] = []
    synthetic_rows: list[dict[str, object]] = []
    seen_participants: set[str] = set()
    event_evidence: list[dict[str, object]] = []
    if not isinstance(event_cards, list):
        structural_issues.append("event_cards_not_a_list")
        event_cards = []

    for event_card in event_cards:
        if not isinstance(event_card, dict):
            structural_issues.append("event_card_not_an_object")
            continue
        event_count += 1
        fights = event_card.get("fights")
        if not isinstance(fights, list):
            structural_issues.append("event_fights_not_a_list")
            continue
        event = event_card.get("event")
        event = event if isinstance(event, Mapping) else {}
        event_url = _string_value(event.get("url"))
        event_marker = _string_value(event.get("card_announcement_status"))
        valid_fights = [fight for fight in fights if isinstance(fight, dict)]
        event_participant_count = sum(
            1
            for fight in valid_fights
            for side in ("a", "b")
            if _string_value(fight.get(f"fighter_{side}"))
        )
        current_event_evidence: dict[str, object] = {
            "event_url": event_url,
            "event_title": _string_value(event.get("title")),
            "event_date": _string_value(event.get("date")),
            "event_source": _string_value(event.get("source")),
            "card_announcement_status": event_marker,
            "fight_count": len(valid_fights),
            "participant_count": event_participant_count,
        }
        empty_card_proof = _string_value(event_card.get("empty_card_proof"))
        if empty_card_proof:
            current_event_evidence["empty_card_proof"] = empty_card_proof
        event_evidence.append(current_event_evidence)
        if not fights:
            # The live collector omits positively identified announced cards
            # with zero bouts. An explicit empty card object is therefore stale
            # or synthetic evidence and cannot complete a refresh scan.
            structural_issues.append("upcoming_event_card_empty")
            continue
        for fight in fights:
            if not isinstance(fight, dict):
                structural_issues.append("fight_not_an_object")
                continue
            fight_count += 1
            fight_participant_keys: list[str] = []
            for side in ("a", "b"):
                name = _string_value(fight.get(f"fighter_{side}"))
                athlete_url = _string_value(
                    fight.get(f"fighter_{side}_athlete_url")
                )
                raw_id = _string_value(fight.get(f"fighter_{side}_id"))
                if not name:
                    structural_issues.append("upcoming_fighter_name_missing")
                    continue
                participants_seen += 1
                name_key = normalize_person_name(name)
                athlete_key = _normalized_athlete_resource(athlete_url)
                supplied_id = normalize_ufcstats_id(raw_id) or ""
                if raw_id and not supplied_id:
                    structural_issues.append("upcoming_fighter_id_invalid")
                known_ids = set(ids_by_athlete.get(athlete_key, set()))
                if supplied_id and known_ids and supplied_id not in known_ids:
                    structural_issues.append("upcoming_identity_conflict")
                    resolved_id = ""
                    source_reason = "identity_conflict"
                elif supplied_id:
                    resolved_id = supplied_id
                    source_reason = ""
                elif len(known_ids) == 1:
                    resolved_id = next(iter(known_ids))
                    source_reason = ""
                elif len(known_ids) > 1:
                    structural_issues.append("upcoming_identity_ambiguous")
                    resolved_id = ""
                    source_reason = "identity_ambiguous"
                else:
                    resolved_id = ""
                    source_reason = "ufcstats_identity_or_profile_not_published"

                participant_key = (
                    f"ufcstats:{resolved_id}"
                    if resolved_id
                    else f"athlete:{athlete_key}"
                    if athlete_key
                    else f"name:{name_key}"
                )
                fight_participant_keys.append(participant_key)
                if participant_key in seen_participants:
                    duplicate_count += 1
                    continue
                seen_participants.add(participant_key)
                if not resolved_id:
                    source_limited.append(
                        {
                            "fighter": name,
                            "athlete_url": athlete_url,
                            "reason": source_reason,
                        }
                    )
                    continue

                resolved_count += 1
                if resolved_id in official_ids:
                    already_in_roster += 1
                    continue
                official_ids.add(resolved_id)
                synthetic_rows.append(
                    {
                        "official_name": name,
                        "ufcstats_name": name,
                        "ufcstats_url": (
                            f"http://ufcstats.com/fighter-details/{resolved_id}"
                        ),
                        "official_athlete_url": athlete_url,
                        "profile_status": "Active",
                        "coverage_eligible": True,
                        "active_roster_status_reason": (
                            "upcoming_card_backfill_target"
                        ),
                    }
                )
            if (
                len(fight_participant_keys) == 2
                and fight_participant_keys[0] == fight_participant_keys[1]
            ):
                structural_issues.append("upcoming_fight_duplicate_identity")

    expanded = official
    if synthetic_rows:
        expanded = pd.concat(
            [official, pd.DataFrame(synthetic_rows)],
            ignore_index=True,
            sort=False,
        )
        expanded.attrs = dict(getattr(official_roster, "attrs", {}) or {})

    source_evidence = _summarize_upcoming_source_responses(
        snapshot.get("source_responses"),
        structural_issues,
    )
    structural_issues = list(dict.fromkeys(structural_issues))
    scan_complete = snapshot.get("scan_complete") is True and not structural_issues
    summary: dict[str, object] = {
        "status": "complete" if scan_complete else "incomplete",
        "scan_complete": scan_complete,
        "collected_at_utc": _string_value(snapshot.get("collected_at_utc")),
        "events_seen": event_count,
        "fights_seen": fight_count,
        "participants_seen": participants_seen,
        "unique_participants": len(seen_participants),
        "duplicate_participants_deduplicated": duplicate_count,
        "participants_resolved": resolved_count,
        "participants_already_in_official_roster": already_in_roster,
        "participants_added_to_backfill": len(synthetic_rows),
        "participants_source_limited": len(source_limited),
        "source_limited_participants": source_limited[:50],
        "source_limited_participants_truncated": max(0, len(source_limited) - 50),
        "official_roster_rows": int(len(official_roster)),
        "backfill_target_rows": int(len(expanded)),
        "structural_issues": structural_issues[:50],
        "structural_issues_truncated": max(0, len(structural_issues) - 50),
        "source_response_count": len(source_evidence),
        "source_evidence": source_evidence,
        "event_evidence_count": len(event_evidence),
        "event_evidence": event_evidence,
        # Deprecated compatibility fields. The former one-event empty-shell
        # exception is retired; every event now relies on the ordinary complete
        # card scan and these remain empty by construction.
        "reviewed_unannounced_empty_event_count": 0,
        "reviewed_unannounced_empty_event_evidence": [],
    }
    return expanded, summary


def _collect_upcoming_backfill_target(
    official_roster: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    source_cutoff = str(
        getattr(official_roster, "attrs", {}).get("source_cutoff") or ""
    ).strip()
    try:
        cutoff_date = date.fromisoformat(source_cutoff)
    except ValueError as exc:
        return official_roster.copy(), {
            "status": "failed",
            "scan_complete": False,
            "source_error": f"RosterWatchCutoffError: {exc}",
            "events_seen": 0,
            "fights_seen": 0,
            "participants_seen": 0,
            "participants_added_to_backfill": 0,
            "participants_source_limited": 0,
            "official_roster_rows": int(len(official_roster)),
            "backfill_target_rows": int(len(official_roster)),
        }
    horizon = cutoff_date + timedelta(
        days=ROSTERWATCH_EVENT_SAFETY_NET_HORIZON_DAYS
    )
    try:
        from src.data.live_monitor import collect_upcoming_event_cards_snapshot

        snapshot = collect_upcoming_event_cards_snapshot(
            force_refresh=True,
            not_before=source_cutoff,
            not_after=horizon.isoformat(),
            max_events=ROSTERWATCH_EVENT_SAFETY_NET_MAX_EVENTS,
        )
        if not isinstance(snapshot, dict):
            raise TypeError("upcoming snapshot provider did not return an object")
    except Exception as exc:
        return official_roster.copy(), {
            "status": "failed",
            "scan_complete": False,
            "source_error": f"{type(exc).__name__}: {exc}",
            "events_seen": 0,
            "fights_seen": 0,
            "participants_seen": 0,
            "participants_added_to_backfill": 0,
            "participants_source_limited": 0,
            "official_roster_rows": int(len(official_roster)),
            "backfill_target_rows": int(len(official_roster)),
        }
    expanded, summary = _build_upcoming_backfill_target(official_roster, snapshot)
    summary["source_cutoff"] = source_cutoff
    summary["horizon_end"] = horizon.isoformat()
    summary["horizon_days"] = ROSTERWATCH_EVENT_SAFETY_NET_HORIZON_DAYS
    summary["event_request_ceiling"] = ROSTERWATCH_EVENT_SAFETY_NET_MAX_EVENTS
    return expanded, summary


def _official_url_identity_trusted(row: dict[str, object]) -> bool:
    from src.data.ufc_active_roster import _official_url_identity_positively_trusted

    return _official_url_identity_positively_trusted(row)


def _new_fighter_split_column(frame: pd.DataFrame) -> str:
    if "split_stable_identity" in frame.columns:
        return "split_stable_identity"
    if "split_alias_aware" in frame.columns:
        return "split_alias_aware"
    return "split_official_name"


def _report_alias_keys(row: dict[str, object]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()

    def _add_key(value: object) -> None:
        key = normalize_person_name(_string_value(value))
        if not key or key in seen:
            return
        seen.add(key)
        keys.append(key)

    alias_fields = ["official_name", "ufcstats_name"]
    if _official_url_identity_trusted(row):
        alias_fields.extend(["profile_name", "slug_name"])
    for field in alias_fields:
        _add_key(row.get(field))
    if _official_url_identity_trusted(row):
        for field in ("canonical_athlete_url", "official_athlete_url"):
            url = _string_value(row.get(field))
            slug = urlparse(url).path.strip("/").rsplit("/", 1)[-1] if url else ""
            if slug:
                _add_key(slug.replace("-", " "))
        for value in str(row.get("alternate_slug_names") or "").split("|"):
            _add_key(value)
    return keys


def _one_consistent_stable_id(
    row: dict[str, object],
    *,
    carrier_fields: tuple[str, ...],
) -> str:
    observed: list[str] = []
    for field in carrier_fields:
        value = _string_value(row.get(field))
        if not value:
            continue
        stable_id = normalize_ufcstats_id(value)
        if stable_id is None:
            return ""
        observed.append(stable_id)
    return next(iter(set(observed))) if len(set(observed)) == 1 else ""


def _load_scraped_profile_indexes(
    path: Path,
) -> dict[str, list[dict[str, object]]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    rows = frame.to_dict(orient="records")
    by_stable_id: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        stable_id = _one_consistent_stable_id(
            row,
            carrier_fields=(
                "stable_ufcstats_id",
                "fighter_id",
                "ufcstats_url",
                "fighter_url",
            ),
        )
        if not stable_id:
            continue
        by_stable_id.setdefault(stable_id, []).append(row)
    return by_stable_id


def _load_supplement_stable_index(
    path: Path,
) -> dict[str, list[dict[str, object]]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    rows = frame.to_dict(orient="records")
    by_stable_id: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        stable_id = _one_consistent_stable_id(
            row,
            carrier_fields=("stable_ufcstats_id", "fighter_id", "ufcstats_url"),
        )
        if not stable_id:
            continue
        by_stable_id.setdefault(stable_id, []).append(row)
    return by_stable_id


def _match_scraped_profile(
    active_row: dict[str, object],
    *,
    scraped_by_stable_id: dict[str, list[dict[str, object]]],
) -> tuple[str, dict[str, object] | None]:
    stable_id = _one_consistent_stable_id(
        active_row,
        carrier_fields=("stable_ufcstats_id", "fighter_id", "ufcstats_url"),
    )
    if not stable_id:
        return "", None
    rows = scraped_by_stable_id.get(stable_id) or []
    if len(rows) == 1:
        return "stable_ufcstats_id", rows[0]
    return "", None


def _matched_supplement_rows(
    active_row: dict[str, object],
    *,
    supplement_by_stable_id: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    stable_id = _one_consistent_stable_id(
        active_row,
        carrier_fields=("stable_ufcstats_id", "fighter_id", "ufcstats_url"),
    )
    if not stable_id:
        return []
    return list(supplement_by_stable_id.get(stable_id) or [])


def _field_support_column(field: str) -> str:
    return "dob" if field == "age" else field


def _official_report_value(active_row: dict[str, object], field: str) -> object:
    if field == "stance":
        return ""
    return active_row.get(field)


def _first_nonblank_string(value: object) -> str:
    return "" if _blank(value) else str(value).strip()


def _unique_nonblank_values(rows: list[dict[str, object]], field: str) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not _field_has_report_value(field, row.get(field)):
            continue
        text = str(row.get(field)).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        values.append(text)
    return " | ".join(values)


def _missing_field_reason(
    *,
    field: str,
    active_row: dict[str, object],
    scraped_profile: dict[str, object] | None,
    supplement_rows: list[dict[str, object]],
) -> tuple[str, str]:
    support_column = _field_support_column(field)
    recovery_hold = reviewed_profile_field_hold(active_row, field=field)
    if recovery_hold is not None:
        stable_id = str(recovery_hold.get("stable_ufcstats_id") or "").strip()
        status = str(recovery_hold.get("decision_status") or "").strip()
        return (
            "reviewed_field_hold",
            f"Reviewed recovery hold ufcstats:{stable_id}:{field} keeps this "
            f"{field} unavailable with status {status}; source observations "
            "must not bypass that hold.",
        )
    reviewed_holds = [
        decision
        for decision in reviewed_profile_field_decisions(active_row)
        if str(decision.get("audit_field") or "").strip() == field
        and str(decision.get("decision_status") or "").strip()
        != "accepted_current_future"
    ]
    if reviewed_holds:
        held = sorted(reviewed_holds, key=lambda item: str(item.get("cell_id") or ""))[0]
        cell_id = str(held.get("cell_id") or "").strip()
        status = str(held.get("decision_status") or "").strip()
        return (
            "reviewed_field_hold",
            f"Reviewed decision {cell_id} keeps this {field} unavailable "
            f"with status {status}; source observations must not bypass that hold.",
        )
    official_value = _official_report_value(active_row, field)
    official_has_value = _field_has_report_value(field, official_value)
    scraped_has_value = scraped_profile is not None and _field_has_report_value(
        field,
        scraped_profile.get(support_column),
    )
    supplement_has_rows = bool(supplement_rows)
    supplement_has_value = any(
        _field_has_report_value(field, row.get(support_column))
        for row in supplement_rows
    )
    has_ufcstats_url = bool(_string_value(active_row.get("ufcstats_url")))

    if field == "age":
        if official_has_value:
            return (
                "official_age_present_but_audit_missed",
                "Official UFC roster age is already present, so the audit missed an available value.",
            )
        if scraped_has_value or supplement_has_value:
            return (
                "official_age_blank_but_dob_available_elsewhere",
                (
                    "Official UFC roster age is blank. A date of birth exists in UFCStats or supplement data, "
                    "but the current audit only counts official-roster age."
                ),
            )
        return (
            "official_age_blank_and_no_dob_source",
            "Official UFC roster age is blank and no date of birth was recovered from UFCStats or supplement sources.",
        )

    if official_has_value:
        return (
            "official_value_present_but_audit_lookup_missed",
            f"An official UFC roster {field} value exists, so the lookup missed an available value.",
        )
    if scraped_has_value or supplement_has_value:
        return (
            "source_value_present_but_audit_lookup_missed",
            f"A {field} value exists in UFCStats or supplement data, so the lookup missed an available value.",
        )
    if not has_ufcstats_url:
        if supplement_has_rows:
            detail = (
                "No UFCStats profile is resolved. Supplement rows exist, but they are blank for this field."
            )
            if field == "stance":
                detail = (
                    "No UFCStats profile is resolved. Supplement rows exist, but they are blank for stance. "
                    "Current automated sources are still source-limited for stance on these new fighters."
                )
            return ("no_ufcstats_url_and_supplement_blank", detail)
        detail = "No UFCStats profile is resolved and no supplement row was recovered for this field."
        if field == "stance":
            detail = (
                "No UFCStats profile is resolved and no supplement row was recovered for stance. "
                "Current automated sources are still source-limited for stance on these new fighters."
            )
        return ("no_ufcstats_url_and_no_supplement_rows", detail)
    if scraped_profile is None:
        if supplement_has_rows:
            return (
                "ufcstats_url_present_but_scraped_profile_missing_and_supplement_blank",
                "A UFCStats URL is resolved, but no scraped UFCStats profile row was available after refresh, and supplement rows are blank for this field.",
            )
        return (
            "ufcstats_url_present_but_scraped_profile_missing",
            "A UFCStats URL is resolved, but no scraped UFCStats profile row was available after refresh.",
        )
    if supplement_has_rows:
        detail = "UFCStats is blank for this field, and supplement rows are also blank."
        if field == "stance":
            detail = (
                "UFCStats is blank for stance, and supplement rows are also blank. "
                "Current automated sources are still source-limited for stance on these new fighters."
            )
        return ("ufcstats_blank_and_supplement_blank", detail)
    detail = "UFCStats is blank for this field, and no supplement row was recovered."
    if field == "stance":
        detail = (
            "UFCStats is blank for stance, and no supplement row was recovered. "
            "Current automated sources are still source-limited for stance on these new fighters."
        )
    return ("ufcstats_blank_and_no_supplement_rows", detail)


_MODEL_FEATURE_FIELDS_BY_PROFILE_FIELD = {
    "age": "a_age | b_age | diff_age",
    "height": "a_height | b_height | diff_height",
    "reach": "a_reach | b_reach | diff_reach",
    "weight": "a_weight | b_weight | diff_weight",
    "stance": "a_stance_enc | b_stance_enc | same_stance",
}


def _source_availability_for_reason(reason_code: str) -> str:
    if reason_code == "reviewed_field_hold":
        return "reviewed_hold"
    if reason_code in {
        "official_age_present_but_audit_missed",
        "official_value_present_but_audit_lookup_missed",
        "source_value_present_but_audit_lookup_missed",
        "official_age_blank_but_dob_available_elsewhere",
    }:
        return "available_but_pipeline_missed"
    if reason_code.startswith("no_ufcstats_url"):
        return "no_ufcstats_profile_resolved"
    if reason_code.startswith("ufcstats_url_present_but_scraped_profile_missing"):
        return "ufcstats_profile_scrape_missing"
    if reason_code == "official_age_blank_and_no_dob_source":
        return "source_limited_blank"
    if reason_code.startswith("ufcstats_blank"):
        return "source_limited_blank"
    return "unknown"


def _repair_action_for_reason(reason_code: str) -> str:
    if reason_code == "reviewed_field_hold":
        return "retain reviewed hold; separate owner review required"
    if reason_code in {
        "official_age_present_but_audit_missed",
        "official_value_present_but_audit_lookup_missed",
        "source_value_present_but_audit_lookup_missed",
        "official_age_blank_but_dob_available_elsewhere",
    }:
        return "repair audit/profile merge lookup"
    if reason_code.startswith("no_ufcstats_url"):
        return "resolve UFCStats URL or recover trusted external profile fields"
    if reason_code.startswith("ufcstats_url_present_but_scraped_profile_missing"):
        return "retry UFCStats profile scrape"
    if reason_code.startswith("ufcstats_blank"):
        return "recover trusted external profile field if available"
    return "review manually"


def _build_profile_repair_queue(report_df: pd.DataFrame) -> list[dict[str, object]]:
    if report_df.empty:
        return []

    keyed = report_df.copy()
    keyed["_profile_identity_key"] = keyed.apply(
        _profile_report_identity_key,
        axis=1,
    )
    keyed = keyed[keyed["_profile_identity_key"].ne("")].copy()
    queue: list[dict[str, object]] = []
    for identity_key, group in keyed.groupby("_profile_identity_key", sort=True):
        official_name = _string_value(group.iloc[0].get("official_name"))
        missing_fields = sorted(str(field) for field in group["field"].dropna().astype(str).unique())
        source_availability = sorted(
            str(value) for value in group["source_availability"].dropna().astype(str).unique()
        )
        reason_codes = sorted(str(value) for value in group["why_missing_code"].dropna().astype(str).unique())
        model_features = sorted(
            {
                token.strip()
                for value in group["model_feature_fields"].dropna().astype(str)
                for token in value.split("|")
                if token.strip()
            }
        )
        queue.append(
            {
                "official_name": official_name,
                "official_athlete_url": _string_value(
                    group.iloc[0].get("official_athlete_url")
                ),
                "stable_ufcstats_id": _one_consistent_stable_id(
                    group.iloc[0].to_dict(),
                    carrier_fields=(
                        "stable_ufcstats_id",
                        "fighter_id",
                        "ufcstats_url",
                    ),
                ),
                "ufcstats_url": _string_value(
                    group.iloc[0].get("ufcstats_url")
                ),
                "profile_identity_key": identity_key,
                "missing_fields": missing_fields,
                "missing_field_count": int(len(group)),
                "source_availability": source_availability,
                "reason_codes": reason_codes,
                "model_feature_fields": model_features,
                "source_limited_only": all(
                    code in _SOURCE_LIMITED_ALERT_REASON_CODES for code in reason_codes
                ),
                "repair_actions": sorted(
                    str(value) for value in group["repair_action"].dropna().astype(str).unique()
                ),
            }
        )
    queue.sort(key=lambda row: (-int(row["missing_field_count"]), str(row["official_name"])))
    return queue


def _profile_report_fighter_count(report_df: pd.DataFrame) -> int:
    if report_df.empty:
        return 0
    keys = report_df.apply(_profile_report_identity_key, axis=1)
    return int(keys[keys.ne("")].nunique())


def _build_unresolved_profile_report(
    *,
    active_roster_path: Path,
    scraped_fighters_path: Path,
    audit_df: pd.DataFrame,
    supplement_path: Path = PROFILE_SUPPLEMENT_PATH,
) -> tuple[dict[str, object], pd.DataFrame]:
    report_df = pd.DataFrame(columns=PROFILE_REPORT_COLUMNS)
    if (
        audit_df.empty
        or "official_name" not in audit_df.columns
        or "split_official_name" not in audit_df.columns
    ):
        return {"rows": 0, "fighters": 0, "fields": {}, "reasons": {}, "reasons_by_field": {}}, report_df

    if not active_roster_path.exists():
        return {"rows": 0, "fighters": 0, "fields": {}, "reasons": {}, "reasons_by_field": {}}, report_df

    active_df = pd.read_csv(active_roster_path)
    if active_df.empty or "official_name" not in active_df.columns:
        return {"rows": 0, "fighters": 0, "fields": {}, "reasons": {}, "reasons_by_field": {}}, report_df

    active_df["_profile_identity_key"] = active_df.apply(
        _profile_report_identity_key,
        axis=1,
    )
    active_identity_counts = active_df["_profile_identity_key"].value_counts()
    active_rows = (
        active_df[
            active_df["_profile_identity_key"].ne("")
            & active_df["_profile_identity_key"].map(active_identity_counts).eq(1)
        ]
        .set_index("_profile_identity_key")
        .to_dict(orient="index")
    )
    scraped_by_stable_id = _load_scraped_profile_indexes(scraped_fighters_path)
    supplement_by_stable_id = _load_supplement_stable_index(supplement_path)

    report_rows: list[dict[str, object]] = []
    split_column = _new_fighter_split_column(audit_df)
    tracked = audit_df[audit_df[split_column].fillna("").eq("newly_added_active_roster")].copy()
    if "coverage_eligible" in tracked.columns:
        tracked = tracked[tracked["coverage_eligible"].fillna(True)].copy()
    for _, audit_row in tracked.iterrows():
        official_name = str(audit_row.get("official_name") or "").strip()
        if not official_name:
            continue
        active_row = active_rows.get(_profile_report_identity_key(audit_row))
        if active_row is None:
            continue
        scraped_match, scraped_profile = _match_scraped_profile(
            active_row,
            scraped_by_stable_id=scraped_by_stable_id,
        )
        supplement_rows = _matched_supplement_rows(
            active_row,
            supplement_by_stable_id=supplement_by_stable_id,
        )
        supplement_sources = sorted(
            {
                _string_value(row.get("source"))
                for row in supplement_rows
                if _string_value(row.get("source"))
            }
        )

        for field in PROFILE_REPORT_FIELDS:
            audit_column = f"{field}_present"
            if audit_column not in audit_row or bool(audit_row.get(audit_column)):
                continue
            reason_code, reason = _missing_field_reason(
                field=field,
                active_row=active_row,
                scraped_profile=scraped_profile,
                supplement_rows=supplement_rows,
            )
            source_availability = _source_availability_for_reason(reason_code)
            report_rows.append(
                {
                    "official_name": official_name,
                    "field": field,
                    "why_missing_code": reason_code,
                    "why_missing": reason,
                    "octagon_debut": _string_value(active_row.get("octagon_debut")),
                    "official_athlete_url": _string_value(active_row.get("official_athlete_url")),
                    "ufcstats_name": _string_value(active_row.get("ufcstats_name")),
                    "ufcstats_url": _string_value(active_row.get("ufcstats_url")),
                    "stable_ufcstats_id": _string_value(
                        active_row.get("stable_ufcstats_id")
                    ),
                    "profile_source_alias": _string_value(audit_row.get("profile_source_alias")),
                    "official_field_value": _string_value(_official_report_value(active_row, field)),
                    "scraped_profile_match": scraped_match,
                    "scraped_profile_name": _string_value((scraped_profile or {}).get("name")),
                    "scraped_field_value": _string_value((scraped_profile or {}).get(_field_support_column(field))),
                    "supplement_rows": int(len(supplement_rows)),
                    "supplement_sources": " | ".join(supplement_sources),
                    "supplement_field_values": _unique_nonblank_values(supplement_rows, _field_support_column(field)),
                    "source_availability": source_availability,
                    "model_feature_fields": _MODEL_FEATURE_FIELDS_BY_PROFILE_FIELD.get(field, ""),
                    "repair_action": _repair_action_for_reason(reason_code),
                }
            )

    if report_rows:
        report_df = pd.DataFrame(report_rows, columns=PROFILE_REPORT_COLUMNS)
        report_df = report_df.sort_values(["official_name", "field"]).reset_index(drop=True)

    summary = {
        "rows": int(len(report_df)),
        "fighters": _profile_report_fighter_count(report_df),
        "fields": report_df["field"].value_counts().sort_index().to_dict() if not report_df.empty else {},
        "reasons": report_df["why_missing_code"].value_counts().sort_index().to_dict() if not report_df.empty else {},
        "reasons_by_field": (
            {
                field: group["why_missing_code"].value_counts().sort_index().to_dict()
                for field, group in report_df.groupby("field", sort=True)
            }
            if not report_df.empty
            else {}
        ),
        "source_availability": (
            report_df["source_availability"].value_counts().sort_index().to_dict()
            if not report_df.empty
            else {}
        ),
        "repair_queue": _build_profile_repair_queue(report_df),
    }
    return summary, report_df


def _profile_supplement_refresh_enabled() -> bool:
    raw = str(os.getenv("UFC_REFRESH_PROFILE_SUPPLEMENT_ENABLED", "1") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _profile_supplement_refresh_limit() -> int | None:
    raw = str(os.getenv("UFC_REFRESH_PROFILE_SUPPLEMENT_LIMIT", "") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value > 0 else None


def _profile_supplement_refresh_rotation_index(
    now_utc: datetime | None = None,
) -> int:
    """Return a stable cadence bucket that advances each hosted refresh cycle.

    Railway's in-process scheduler has no run-number environment variable.
    Bucketing UTC time by its configured 1-24 hour cadence gives every scheduled
    run a deterministic index while surviving process restarts.
    """
    raw_interval = str(os.getenv("UFC_REFRESH_INTERVAL_HOURS", "24") or "").strip()
    try:
        interval_hours = float(raw_interval)
    except (TypeError, ValueError):
        interval_hours = 24.0
    if not interval_hours > 0:
        interval_hours = 24.0
    interval_hours = min(max(interval_hours, 1.0), 24.0)

    current = now_utc or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)
    return max(int(current.timestamp() // (interval_hours * 60 * 60)), 0)


def _profile_supplement_refresh_sources() -> list[str]:
    raw = str(os.getenv("UFC_REFRESH_PROFILE_SUPPLEMENT_SOURCES", "") or "").strip()
    if not raw:
        if is_hosted_runtime():
            return list(DEFAULT_HOSTED_PROFILE_SUPPLEMENT_REFRESH_SOURCES)
        return list(DEFAULT_PROFILE_SUPPLEMENT_REFRESH_SOURCES)
    selected: list[str] = []
    for token in raw.split(","):
        source = token.strip().lower()
        if source in PROFILE_SUPPLEMENT_ALL_SOURCES and source not in selected:
            selected.append(source)
    return selected or list(DEFAULT_PROFILE_SUPPLEMENT_REFRESH_SOURCES)


def _build_profile_gap_candidate_frame(
    *,
    active_roster_path: Path,
    audit_df: pd.DataFrame,
) -> pd.DataFrame:
    if audit_df.empty or "official_name" not in audit_df.columns:
        return pd.DataFrame()

    split_column = _new_fighter_split_column(audit_df)
    target_rows = audit_df.loc[
            audit_df[split_column].fillna("").eq("newly_added_active_roster")
            & (audit_df.get("coverage_eligible", pd.Series(True, index=audit_df.index)).fillna(True))
            & (~audit_df["full_physical_bundle_present"].fillna(False)),
        ].copy()
    target_identity_keys = {
        key
        for key in target_rows.apply(_profile_report_identity_key, axis=1)
        if key
    }
    if not target_identity_keys:
        return pd.DataFrame()

    active_df = pd.read_csv(active_roster_path)
    if active_df.empty or "official_name" not in active_df.columns:
        return pd.DataFrame()
    active_df["_profile_identity_key"] = active_df.apply(
        _profile_report_identity_key,
        axis=1,
    )
    identity_counts = active_df["_profile_identity_key"].value_counts()
    candidate_df = (
        active_df[
            active_df["_profile_identity_key"].isin(target_identity_keys)
            & active_df["_profile_identity_key"].map(identity_counts).eq(1)
        ]
        .copy()
        .drop(columns=["_profile_identity_key"])
        .sort_values("official_name")
        .reset_index(drop=True)
    )
    return candidate_df


def _new_fighter_alert_grace_days() -> int:
    raw = str(os.getenv(NEW_FIGHTER_ALERT_GRACE_DAYS_ENV, DEFAULT_NEW_FIGHTER_ALERT_GRACE_DAYS) or "").strip()
    try:
        value = int(raw)
    except Exception:
        return DEFAULT_NEW_FIGHTER_ALERT_GRACE_DAYS
    return max(value, 0)


def _build_profile_audit_alert_summary(
    *,
    active_roster_path: Path,
    audit_df: pd.DataFrame,
    as_of_utc: datetime | None = None,
    new_fighter_grace_days: int | None = None,
    unresolved_df: pd.DataFrame | None = None,
) -> dict[str, object]:
    grace_days = (
        _new_fighter_alert_grace_days()
        if new_fighter_grace_days is None
        else max(int(new_fighter_grace_days), 0)
    )
    as_of = as_of_utc or datetime.now(timezone.utc)
    split_column = _new_fighter_split_column(audit_df)
    identity_match_method = {
        "split_stable_identity": "stable_ufcstats_identity",
        "split_alias_aware": "alias_aware",
        "split_official_name": "official_name",
    }[split_column]
    base_summary = {
        "as_of_date_utc": as_of.date().isoformat(),
        "new_fighter_grace_days": grace_days,
        "identity_match_method": identity_match_method,
        "newly_added_active_roster": {
            "rows_total": 0,
            "rows_alert_eligible": 0,
            "rows_in_grace": 0,
        },
        "source_limited_missing_fields": _build_new_fighter_source_limited_alert_summary(
            alert_eligible=pd.DataFrame(),
            unresolved_df=unresolved_df,
        ),
        "split_summary_official_name": {
            "newly_added_active_roster": {"rows": 0},
        },
        "split_summary_alias_aware": {
            "newly_added_active_roster": {"rows": 0},
        },
        "split_summary_stable_identity": {
            "newly_added_active_roster": {"rows": 0},
        },
    }
    if audit_df.empty or "official_name" not in audit_df.columns or split_column not in audit_df.columns:
        return base_summary

    if active_roster_path.exists():
        active_df = pd.read_csv(active_roster_path).copy()
    else:
        active_df = pd.DataFrame(columns=["official_name", "octagon_debut"])

    active_df["_profile_identity_key"] = active_df.apply(
        _profile_report_identity_key,
        axis=1,
    )
    if "octagon_debut" in active_df.columns:
        active_df["_octagon_debut_dt"] = pd.to_datetime(active_df["octagon_debut"], errors="coerce", format="mixed")
    else:
        active_df["_octagon_debut_dt"] = pd.NaT

    active_dates = active_df[["_profile_identity_key", "_octagon_debut_dt"]]
    if not active_dates.empty:
        identity_counts = active_dates["_profile_identity_key"].value_counts()
        active_dates = (
            active_dates[
                active_dates["_profile_identity_key"].ne("")
                & active_dates["_profile_identity_key"].map(identity_counts).eq(1)
            ]
        )

    merged = audit_df.copy()
    merged["_profile_identity_key"] = merged.apply(
        _profile_report_identity_key,
        axis=1,
    )
    merged = merged.merge(active_dates, on="_profile_identity_key", how="left")

    eligible_mask = merged.get("coverage_eligible", pd.Series(True, index=merged.index)).fillna(True)
    new_fighters = merged[
        merged[split_column].fillna("").eq("newly_added_active_roster")
        & eligible_mask
    ].copy()
    if new_fighters.empty:
        return base_summary

    cutoff = pd.Timestamp(as_of.date()) - pd.Timedelta(days=grace_days)
    alert_eligible = new_fighters[
        new_fighters["_octagon_debut_dt"].isna()
        | (new_fighters["_octagon_debut_dt"] <= cutoff)
    ].copy()
    source_limited_missing_fields = _build_new_fighter_source_limited_alert_summary(
        alert_eligible=alert_eligible,
        unresolved_df=unresolved_df,
    )

    split_summaries: dict[str, dict[str, dict[str, object]]] = {}
    new_fighter_counts_by_method: dict[str, dict[str, int]] = {}
    for candidate_split_column, method_name in (
        ("split_official_name", "official_name"),
        ("split_alias_aware", "alias_aware"),
        ("split_stable_identity", "stable_ufcstats_identity"),
    ):
        if candidate_split_column not in merged.columns:
            continue
        method_new = merged[
            merged[candidate_split_column].fillna("").eq("newly_added_active_roster")
            & eligible_mask
        ].copy()
        method_alert_eligible = method_new[
            method_new["_octagon_debut_dt"].isna()
            | (method_new["_octagon_debut_dt"] <= cutoff)
        ].copy()
        split_summaries[candidate_split_column] = {
            "newly_added_active_roster": _summarize_audit_frame(method_alert_eligible),
        }
        new_fighter_counts_by_method[method_name] = {
            "rows_total": int(len(method_new)),
            "rows_alert_eligible": int(len(method_alert_eligible)),
            "rows_in_grace": int(len(method_new) - len(method_alert_eligible)),
        }

    return {
        "as_of_date_utc": as_of.date().isoformat(),
        "new_fighter_grace_days": grace_days,
        "identity_match_method": identity_match_method,
        "new_fighter_counts_by_method": new_fighter_counts_by_method,
        "newly_added_active_roster": {
            "rows_total": int(len(new_fighters)),
            "rows_alert_eligible": int(len(alert_eligible)),
            "rows_in_grace": int(len(new_fighters) - len(alert_eligible)),
        },
        "source_limited_missing_fields": source_limited_missing_fields,
        "split_summary_official_name": {
            "newly_added_active_roster": (
                split_summaries.get("split_official_name", {})
                .get("newly_added_active_roster", {"rows": 0})
            ),
        },
        "split_summary_alias_aware": {
            "newly_added_active_roster": (
                split_summaries.get("split_alias_aware", {})
                .get("newly_added_active_roster", _summarize_audit_frame(alert_eligible))
            ),
        },
        "split_summary_stable_identity": {
            "newly_added_active_roster": (
                split_summaries.get("split_stable_identity", {})
                .get("newly_added_active_roster", _summarize_audit_frame(alert_eligible))
            ),
        },
    }


def _build_new_fighter_source_limited_alert_summary(
    *,
    alert_eligible: pd.DataFrame,
    unresolved_df: pd.DataFrame | None,
) -> dict[str, dict[str, object]]:
    field_specs = {
        "reach": {
            "audit_column": "reach_present",
            "required_fields": ("reach",),
        },
        "stance": {
            "audit_column": "stance_present",
            "required_fields": ("stance",),
        },
        "full_physical_bundle": {
            "audit_column": "full_physical_bundle_present",
            "required_fields": ("age", "weight", "height", "reach", "stance"),
        },
    }
    base_summary = {
        field: {
            "rows_missing": 0,
            "rows_source_limited": 0,
            "source_limited_only": False,
        }
        for field in field_specs
    }
    if (
        alert_eligible.empty
        or unresolved_df is None
        or unresolved_df.empty
        or "official_name" not in alert_eligible.columns
        or "official_name" not in unresolved_df.columns
        or "field" not in unresolved_df.columns
        or "why_missing_code" not in unresolved_df.columns
    ):
        return base_summary

    unresolved_reason_codes: dict[tuple[str, str], set[str]] = {}
    eligible_identity_keys = {
        key
        for key in alert_eligible.apply(_profile_report_identity_key, axis=1)
        if key
    }
    if not eligible_identity_keys:
        return base_summary

    for row in unresolved_df.to_dict(orient="records"):
        key = _profile_report_identity_key(row)
        field = _string_value(row.get("field"))
        reason_code = _string_value(row.get("why_missing_code"))
        if (
            not key
            or key not in eligible_identity_keys
            or not field
            or not reason_code
        ):
            continue
        unresolved_reason_codes.setdefault((key, field), set()).add(reason_code)

    if not unresolved_reason_codes:
        return base_summary

    def _row_missing_fields(row: pd.Series, summary_field: str, required_fields: tuple[str, ...]) -> list[str]:
        if summary_field != "full_physical_bundle":
            return list(required_fields)
        missing_fields: list[str] = []
        for field in required_fields:
            audit_column = f"{field}_present"
            present = row.get(audit_column)
            if pd.isna(present) or not bool(present):
                missing_fields.append(field)
        return missing_fields

    for summary_field, spec in field_specs.items():
        missing_rows = alert_eligible[~alert_eligible[spec["audit_column"]].fillna(False)].copy()
        if missing_rows.empty:
            continue
        rows_missing = int(len(missing_rows))
        rows_source_limited = 0
        for _, row in missing_rows.iterrows():
            key = _profile_report_identity_key(row)
            missing_fields = _row_missing_fields(row, summary_field, spec["required_fields"])
            if not key or not missing_fields:
                continue
            if all(
                (reason_codes := unresolved_reason_codes.get((key, field)))
                and reason_codes.issubset(_SOURCE_LIMITED_ALERT_REASON_CODES)
                for field in missing_fields
            ):
                rows_source_limited += 1

        base_summary[summary_field] = {
            "rows_missing": rows_missing,
            "rows_source_limited": int(rows_source_limited),
            "source_limited_only": bool(rows_missing > 0 and rows_source_limited == rows_missing),
        }

    return base_summary


def _maybe_refresh_profile_supplement(
    *,
    active_roster_path: Path,
    scraped_fighters_path: Path,
    audit_df: pd.DataFrame,
    partial_refresh: bool,
    output_path: Path | None = None,
) -> dict[str, object] | None:
    output_path = Path(output_path) if output_path is not None else PROFILE_SUPPLEMENT_PATH
    if partial_refresh:
        return {"action": "skip", "reason": "partial refresh"}
    if not _profile_supplement_refresh_enabled():
        return {"action": "skip", "reason": "disabled by UFC_REFRESH_PROFILE_SUPPLEMENT_ENABLED"}

    candidate_df = _build_profile_gap_candidate_frame(
        active_roster_path=active_roster_path,
        audit_df=audit_df,
    )
    if candidate_df.empty:
        return {"action": "skip", "reason": "no new active-roster profile gaps"}

    limit = _profile_supplement_refresh_limit()
    sources = _profile_supplement_refresh_sources()
    rotation_index = (
        _profile_supplement_refresh_rotation_index()
        if is_hosted_runtime()
        else None
    )
    with TemporaryDirectory(prefix="ufc_profile_gap_candidates_") as tmp_dir:
        candidate_source_path = Path(tmp_dir) / "active_roster_profile_gap_candidates.csv"
        candidate_df.to_csv(candidate_source_path, index=False)
        summary = run_profile_supplement_refresh(
            scraped_fighters_path=scraped_fighters_path,
            candidate_source_csv=candidate_source_path,
            output_path=output_path,
            sources=sources,
            limit=limit,
            candidate_rotation_index=rotation_index,
        )
    return {
        "action": "completed",
        "target_rows": int(len(candidate_df)),
        "limit": limit,
        "selected_sources": list(sources),
        **summary,
    }


@_serialized_refresh
def run_scheduled_refresh(
    *,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
    output_subdirs: list[str] | None = None,
    limit_fighters: int | None = None,
    audit_json_path: Path | None = DEFAULT_AUDIT_JSON,
    audit_csv_path: Path | None = DEFAULT_AUDIT_CSV,
    unresolved_json_path: Path | None = DEFAULT_UNRESOLVED_PROFILE_JSON,
    unresolved_csv_path: Path | None = DEFAULT_UNRESOLVED_PROFILE_CSV,
    skip_rebuild: bool = False,
    skip_audit: bool = False,
    update_production_manifest: bool | None = None,
    refresh_existing_profiles: bool = False,
    calendar_claim: dict[str, object] | None = None,
) -> dict[str, object]:
    calendar_claim = _validated_calendar_claim(calendar_claim)
    if update_production_manifest is None:
        should_update_production_manifest = is_hosted_runtime() or bool(
            str(os.getenv(PRODUCTION_BUNDLE_ENV, "") or "").strip()
        )
    else:
        should_update_production_manifest = bool(update_production_manifest)
    publication_capable_refresh = bool(
        should_update_production_manifest
        and not skip_rebuild
        and not skip_audit
        and limit_fighters is None
    )
    publication_recovery = _publication_recovery_for_refresh(
        publication_capable=publication_capable_refresh
    )
    baseline_lineage = _active_approved_v7_baseline_lineage(
        required=publication_capable_refresh,
    )
    publication_manifest_path = _active_production_manifest_path()
    publication_manifest_bytes_before: bytes | None = None
    publication_manifest_payload_before: dict[str, object] | None = None
    if publication_capable_refresh:
        publication_manifest_bytes_before = publication_manifest_path.read_bytes()
        parsed_manifest = json.loads(
            publication_manifest_bytes_before.decode("utf-8")
        )
        if not isinstance(parsed_manifest, dict):
            raise RuntimeError("active production manifest is not a JSON object")
        publication_manifest_payload_before = parsed_manifest
    # --- diagnostic: log all resolved paths and pre-refresh file state ---
    resolved_paths = _log_resolved_data_paths()
    accepted_raw_dir = RAW_DATA_DIR
    accepted_processed_dir = (
        _active_production_processed_dir()
        or PROCESSED_DATA_DIR
    )
    if (
        publication_manifest_bytes_before is not None
        and publication_manifest_path.read_bytes()
        != publication_manifest_bytes_before
    ):
        raise RuntimeError("Active production manifest changed before refresh staging")
    candidate_raw_dir = _new_raw_refresh_generation_dir()
    raw_publication_artifacts = _raw_publication_artifacts(accepted_raw_dir)
    raw_results_stats_pair = _raw_results_stats_pair()
    accepted_pair_manifest_path = (
        accepted_raw_dir / raw_results_stats_pair.manifest_path
    )
    accepted_pair_manifest_bytes_before: bytes | None = None
    if accepted_pair_manifest_path.is_file():
        accepted_pair_manifest_bytes_before = (
            accepted_pair_manifest_path.read_bytes()
        )
        if verify_csv_pair_manifest(
            accepted_raw_dir / raw_results_stats_pair.results_path,
            accepted_raw_dir / raw_results_stats_pair.stats_path,
            manifest_path=accepted_pair_manifest_path,
        ) is None:
            raise RuntimeError("accepted results/statistics pair marker is missing")
        if (
            accepted_pair_manifest_path.read_bytes()
            != accepted_pair_manifest_bytes_before
        ):
            raise RuntimeError(
                "Accepted results/statistics pair marker changed before refresh staging"
            )
    elif accepted_pair_manifest_path.exists():
        raise RuntimeError(
            "accepted results/statistics pair marker is not a file"
        )
    raw_stage_receipt = stage_accepted_raw_artifacts(
        accepted_root=accepted_raw_dir,
        candidate_root=candidate_raw_dir,
        artifacts=raw_publication_artifacts,
        pair=raw_results_stats_pair,
    )
    read_only_raw_stage_receipt = _stage_read_only_rebuild_inputs(
        accepted_root=accepted_raw_dir,
        candidate_root=candidate_raw_dir,
    )
    raw_generation_cleanup = _prune_raw_refresh_generations(
        candidate_raw_dir,
        fence_reason="complete_training_monthly_recovery_reference_inventory_not_adopted",
    )
    accepted_roster_path = accepted_raw_dir / OFFICIAL_ACTIVE_ROSTER_PATH.name
    candidate_roster_path = candidate_raw_dir / OFFICIAL_ACTIVE_ROSTER_PATH.name
    accepted_roster_audit_path = (
        accepted_raw_dir / OFFICIAL_ACTIVE_ROSTER_IDENTITY_AUDIT_PATH.name
    )
    candidate_roster_audit_path = (
        candidate_raw_dir / OFFICIAL_ACTIVE_ROSTER_IDENTITY_AUDIT_PATH.name
    )
    accepted_scraped_fighters_path = accepted_raw_dir / BACKFILL_FIGHTERS_PATH.name
    scraped_fighters_path = candidate_raw_dir / BACKFILL_FIGHTERS_PATH.name
    candidate_profile_failures_path = (
        candidate_raw_dir / BACKFILL_PROFILE_FAILURES_PATH.name
    )
    accepted_results_path = accepted_raw_dir / BACKFILL_RESULTS_PATH.name
    candidate_results_path = candidate_raw_dir / BACKFILL_RESULTS_PATH.name
    accepted_stats_path = accepted_raw_dir / BACKFILL_STATS_PATH.name
    candidate_stats_path = candidate_raw_dir / BACKFILL_STATS_PATH.name
    accepted_profile_supplement_path = accepted_raw_dir / PROFILE_SUPPLEMENT_PATH.name
    candidate_profile_supplement_path = candidate_raw_dir / PROFILE_SUPPLEMENT_PATH.name
    candidate_event_dates_path = candidate_raw_dir / "ufc-event-dates.csv"
    processed_fights_path = accepted_processed_dir / "fights_cleaned.csv"
    resolved_paths["CANDIDATE_RAW_DATA_DIR"] = str(candidate_raw_dir)
    resolved_paths["ACCEPTED_PROCESSED_DATA_DIR"] = str(accepted_processed_dir)

    pre_refresh_state = {
        "active_roster": _file_snapshot(accepted_roster_path),
        "scraped_fighters": _file_snapshot(accepted_scraped_fighters_path),
        "processed_fights": _file_snapshot(processed_fights_path),
    }
    pre_refresh_row_guard_state = _row_guard_snapshot(
        raw_data_dir=accepted_raw_dir,
        processed_data_dir=accepted_processed_dir,
    )
    logger.info(
        "UFC refresh pre-state: roster=%s scraped_fighters=%s processed_fights=%s",
        pre_refresh_state["active_roster"],
        pre_refresh_state["scraped_fighters"],
        pre_refresh_state["processed_fights"],
    )

    generation_cleanup = _prune_refresh_generations(
        fence_reason="complete_training_monthly_recovery_reference_inventory_not_adopted",
    )
    requested_output_subdirs = [
        str(value) for value in (output_subdirs or list(DEFAULT_REBUILD_OUTPUT_SUBDIRS))
    ]
    staged_processed_dir = None if skip_rebuild else _new_refresh_generation_dir()
    rebuild_output_subdirs = (
        []
        if staged_processed_dir is None
        else _staged_rebuild_output_subdirs(
            staged_processed_dir,
            requested_output_subdirs,
        )
    )

    # --- seed stale scraped fighters from image if hosted ---
    seed_summary: dict[str, object] | None = None
    profile_supplement_seed_summary: dict[str, object] | None = None
    if is_hosted_runtime():
        seed_summary = _seed_stale_scraped_fighters(
            runtime_path=scraped_fighters_path,
        )
        profile_supplement_seed_summary = _seed_stale_profile_supplement(
            runtime_path=candidate_profile_supplement_path,
        )

    identity_review_path = candidate_raw_dir / ROSTERWATCH_STATE_SUBDIR / Path(ROSTERWATCH_IDENTITY_REVIEW_PATH).name
    previous_identity_review = None
    if identity_review_path.is_file():
        try:
            previous_identity_review = json.loads(identity_review_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Previous candidate identity review could not be read: %s", identity_review_path)
    roster_df = sync_rosterwatch_active_roster(
        output_path=candidate_roster_path,
        state_root=candidate_raw_dir / ROSTERWATCH_STATE_SUBDIR,
        identity_review_path=identity_review_path,
        identity_provider=load_rosterwatch_identity_provider(candidate_raw_dir / ROSTERWATCH_STATE_SUBDIR),
        source_path=(
            str(os.getenv(ROSTERWATCH_SOURCE_PATH_ENV, "") or "").strip() or None
        ),
        source_url=(
            str(os.getenv(ROSTERWATCH_SOURCE_URL_ENV, "") or "").strip() or None
        ),
        source_cutoff=(
            str(os.getenv(ROSTERWATCH_SOURCE_CUTOFF_ENV, "") or "").strip()
            or None
        ),
        allow_initial_migration=_truthy_env(
            ROSTERWATCH_INITIAL_MIGRATION_ENV,
            "0",
        ),
    )
    # Include the snapshot just acquired by this attempt as well as every
    # previously accepted immutable Roster Watch snapshot in any publication.
    raw_publication_artifacts = _raw_publication_artifacts(
        accepted_raw_dir,
        candidate_raw_dir,
    )
    roster_summary = _roster_summary(roster_df, output_path=candidate_roster_path)
    rosterwatch_membership_ready = (
        roster_summary.get("source") == "rosterwatch"
        and roster_summary.get("membership_source") == "rosterwatch"
        and roster_summary.get("used_cached_fallback") is not True
        and roster_summary.get("acquisition_complete") is True
        and roster_summary.get("validation_complete") is True
        and roster_summary.get("reconciliation_complete") is True
        and roster_summary.get("sync_complete") is True
        and not str(roster_summary.get("sync_error") or "").strip()
    )
    if not rosterwatch_membership_ready:
        raise RuntimeError(
            "Roster Watch authoritative membership sync failed closed before "
            "the UFC event safety net or any enrichment: "
            + json.dumps(roster_summary, ensure_ascii=False, sort_keys=True)
        )
    identity_review = roster_df.attrs.get("identity_review")
    if isinstance(identity_review, dict):
        emit_identity_review_alerts(identity_review, previous_identity_review)
    preserved_official_field_exclusions = _preserved_official_field_exclusions(
        roster_df
    )
    backfill_roster_df, upcoming_summary = _collect_upcoming_backfill_target(
        roster_df
    )

    backfill_kwargs: dict[str, object] = {
        "refresh_roster": False,
        "limit_fighters": limit_fighters,
        "roster_df": backfill_roster_df,
        "fighters_path": scraped_fighters_path,
        "profile_failures_path": candidate_profile_failures_path,
        "results_path": candidate_results_path,
        "stats_path": candidate_stats_path,
        "preserve_existing_profile_change_fields": not refresh_existing_profiles,
    }
    if refresh_existing_profiles:
        backfill_kwargs["refresh_existing_profiles"] = True
    backfill_summary = run_backfill(**backfill_kwargs)

    rebuild_summary: dict[str, object] | None = None
    processed_history_preservation: dict[str, object] | None = None
    processed_history_preservation_attempts: list[dict[str, object]] = []
    if staged_processed_dir is not None:
        rebuild_summary = run_rebuild(
            dataset_variant=dataset_variant,
            output_subdirs=rebuild_output_subdirs,
            update_production_manifest=False,
            raw_data_dir=candidate_raw_dir,
            processed_data_dir=staged_processed_dir,
            base_dataset_path=candidate_raw_dir / "ufc-master.csv",
            fight_results_path=candidate_results_path,
            fight_stats_path=candidate_stats_path,
            scraped_fighters_path=scraped_fighters_path,
            supplemental_profiles_path=candidate_profile_supplement_path,
            official_active_roster_path=candidate_roster_path,
            pierce_fight_identities_path=(
                candidate_raw_dir / "github_PierceHampton.csv"
            ),
            event_dates_cache_path=candidate_event_dates_path,
            historical_odds_dir=candidate_raw_dir / "historical_odds",
            rankings_history_path=(
                candidate_raw_dir
                / "kaggle_rankings"
                / "rankings_history_extended.csv"
            ),
            method_odds_history_path=(
                candidate_raw_dir
                / "method_odds"
                / "historical_method_odds_all.csv"
            ),
            isolated_staging=True,
        )
        rebuild_summary.pop("production_bundle", None)
        processed_history_preservation = _preserve_rebuild_processed_history(
            accepted_dir=accepted_processed_dir,
            staged_root=staged_processed_dir,
            rebuild_summary=rebuild_summary,
        )
        processed_history_preservation_attempts.append(
            processed_history_preservation
        )
        rebuild_summary["historical_processed_preservation"] = (
            processed_history_preservation
        )
        rebuild_summary["historical_processed_preservation_attempts"] = list(
            processed_history_preservation_attempts
        )

    profile_supplement_summary: dict[str, object] | None = None
    if not skip_audit:
        _pre_audit_summary, pre_audit_df = run_audit(
            active_roster_path=candidate_roster_path,
            processed_fights_path=(
                staged_processed_dir / "fights_cleaned.csv"
                if staged_processed_dir is not None
                else processed_fights_path
            ),
            scraped_fighters_path=scraped_fighters_path,
            supplemental_profiles_path=candidate_profile_supplement_path,
            excluded_route_bound_official_fields=(
                preserved_official_field_exclusions
            ),
        )
        try:
            profile_supplement_summary = _maybe_refresh_profile_supplement(
                active_roster_path=candidate_roster_path,
                scraped_fighters_path=scraped_fighters_path,
                audit_df=pre_audit_df,
                partial_refresh=bool(limit_fighters is not None),
                output_path=candidate_profile_supplement_path,
            )
        except Exception as exc:
            logger.warning("Profile supplement refresh failed: %s", exc, exc_info=True)
            profile_supplement_summary = {"action": "error", "reason": str(exc)}
        if profile_supplement_summary:
            logger.info("Targeted profile supplement refresh result: %s", profile_supplement_summary)
            if int(profile_supplement_summary.get("recovered_rows") or 0) > 0 and not skip_rebuild:
                logger.info(
                    "Profile supplement recovered %s row(s); rebuilding processed artifacts again "
                    "so recovered profile fields are reflected in features",
                    profile_supplement_summary.get("recovered_rows"),
                )
                rebuild_summary = run_rebuild(
                    dataset_variant=dataset_variant,
                    output_subdirs=rebuild_output_subdirs,
                    update_production_manifest=False,
                    raw_data_dir=candidate_raw_dir,
                    processed_data_dir=staged_processed_dir,
                    base_dataset_path=candidate_raw_dir / "ufc-master.csv",
                    fight_results_path=candidate_results_path,
                    fight_stats_path=candidate_stats_path,
                    scraped_fighters_path=scraped_fighters_path,
                    supplemental_profiles_path=candidate_profile_supplement_path,
                    official_active_roster_path=candidate_roster_path,
                    pierce_fight_identities_path=(
                        candidate_raw_dir / "github_PierceHampton.csv"
                    ),
                    event_dates_cache_path=candidate_event_dates_path,
                    historical_odds_dir=candidate_raw_dir / "historical_odds",
                    rankings_history_path=(
                        candidate_raw_dir
                        / "kaggle_rankings"
                        / "rankings_history_extended.csv"
                    ),
                    method_odds_history_path=(
                        candidate_raw_dir
                        / "method_odds"
                        / "historical_method_odds_all.csv"
                    ),
                    isolated_staging=True,
                )
                rebuild_summary.pop("production_bundle", None)
                processed_history_preservation = (
                    _preserve_rebuild_processed_history(
                        accepted_dir=accepted_processed_dir,
                        staged_root=staged_processed_dir,
                        rebuild_summary=rebuild_summary,
                    )
                )
                processed_history_preservation_attempts.append(
                    processed_history_preservation
                )
                rebuild_summary["historical_processed_preservation"] = (
                    processed_history_preservation
                )
                rebuild_summary["historical_processed_preservation_attempts"] = list(
                    processed_history_preservation_attempts
                )

    audit_summary: dict[str, object] | None = None
    audit_alert_summary: dict[str, object] | None = None
    unresolved_summary: dict[str, object] | None = None
    unresolved_df = pd.DataFrame(columns=PROFILE_REPORT_COLUMNS)
    if not skip_audit:
        # Verify the audit reads the SAME files the backfill just wrote to
        if str(scraped_fighters_path.resolve()) != str(
            (candidate_raw_dir / BACKFILL_FIGHTERS_PATH.name).resolve()
        ):
            logger.error(
                "PATH MISMATCH: audit scraped_fighters_path=%s != backfill FIGHTERS_PATH=%s",
                scraped_fighters_path, BACKFILL_FIGHTERS_PATH,
            )

        audit_summary, audit_df = run_audit(
            active_roster_path=candidate_roster_path,
            processed_fights_path=(
                staged_processed_dir / "fights_cleaned.csv"
                if staged_processed_dir is not None
                else processed_fights_path
            ),
            scraped_fighters_path=scraped_fighters_path,
            supplemental_profiles_path=candidate_profile_supplement_path,
            excluded_route_bound_official_fields=(
                preserved_official_field_exclusions
            ),
        )
        if audit_json_path is not None:
            audit_json_path.parent.mkdir(parents=True, exist_ok=True)
            audit_json_path.write_text(json.dumps(audit_summary, indent=2) + "\n", encoding="utf-8")
        if audit_csv_path is not None:
            audit_csv_path.parent.mkdir(parents=True, exist_ok=True)
            audit_df.sort_values(["split_alias_aware", "official_name"]).to_csv(audit_csv_path, index=False)
        unresolved_summary, unresolved_df = _build_unresolved_profile_report(
            active_roster_path=candidate_roster_path,
            scraped_fighters_path=scraped_fighters_path,
            audit_df=audit_df,
            supplement_path=candidate_profile_supplement_path,
        )
        audit_alert_summary = _build_profile_audit_alert_summary(
            active_roster_path=candidate_roster_path,
            audit_df=audit_df,
            unresolved_df=unresolved_df,
        )
        if unresolved_json_path is not None or unresolved_csv_path is not None:
            if unresolved_json_path is not None:
                _write_json(unresolved_json_path, unresolved_summary)
            if unresolved_csv_path is not None:
                unresolved_csv_path.parent.mkdir(parents=True, exist_ok=True)
                unresolved_df.to_csv(unresolved_csv_path, index=False)
            logger.info("Post-refresh unresolved profile report: %s", unresolved_summary)

    # --- diagnostic: post-refresh file state ---
    post_refresh_state = {
        "active_roster": _file_snapshot(candidate_roster_path),
        "scraped_fighters": _file_snapshot(scraped_fighters_path),
        "processed_fights": _file_snapshot(
            staged_processed_dir / "fights_cleaned.csv"
            if staged_processed_dir is not None
            else processed_fights_path
        ),
    }
    post_refresh_row_guard_state = _row_guard_snapshot(
        raw_data_dir=candidate_raw_dir,
        processed_data_dir=(
            staged_processed_dir
            if staged_processed_dir is not None
            else accepted_processed_dir
        ),
    )
    if staged_processed_dir is not None:
        post_refresh_row_guard_state = _snapshot_with_staged_processed_files(
            post_refresh_row_guard_state,
            processed_dir=staged_processed_dir,
        )
    roster_attrs = dict(getattr(roster_df, "attrs", {}) or {})
    roster_identity_explanations = {
        reason: rows
        for reason in (
            "expired_missing_live_rows",
            "intentionally_removed_cached_rows",
        )
        if isinstance((rows := roster_attrs.get(reason)), list)
    }
    row_drop_guard = _build_row_drop_guard(
        pre_refresh_row_guard_state,
        post_refresh_row_guard_state,
        explained_identity_reasons={
            _ACTIVE_ROSTER_GUARD_ARTIFACT: roster_identity_explanations,
        },
    )
    if row_drop_guard["violations"]:
        logger.warning("UFC refresh row-drop guard violations: %s", row_drop_guard["violations"])
    if row_drop_guard["explained_drops"]:
        logger.info(
            "UFC refresh row-count drops explained by artifact lifecycle: %s",
            row_drop_guard["explained_drops"],
        )
    logger.info(
        "UFC refresh post-state: roster=%s scraped_fighters=%s processed_fights=%s",
        post_refresh_state["active_roster"],
        post_refresh_state["scraped_fighters"],
        post_refresh_state["processed_fights"],
    )

    identity_integrity_guard = (
        _processed_identity_integrity_guard(staged_processed_dir)
        if staged_processed_dir is not None
        else None
    )
    if (identity_integrity_guard or {}).get("violations"):
        logger.warning(
            "UFC refresh processed-identity guard violations: %s",
            identity_integrity_guard["violations"],
        )

    coverage_snapshot = coverage_snapshot_from_audit(
        audit_summary,
        audit_alert_summary,
    )
    coverage_floors: dict[str, float] = {}
    publication_quality: dict[str, object]
    try:
        coverage_floors = configured_coverage_floors()
        event_date_evidence = _event_date_evidence_lookup(
            candidate_event_dates_path
        )
        completed_profile_ids = {
            f"ufcstats:{fighter_id}"
            for fighter_id in (
                backfill_summary.get(
                    "fighter_ids_with_completed_ufcstats_bouts"
                )
                or []
            )
            if str(fighter_id or "").strip()
        }
        publication_quality = publication_quality_report(
            accepted_results=read_csv_required(
                accepted_results_path,
                label="accepted_fight_results",
            ),
            candidate_results=read_csv_required(
                candidate_results_path,
                label="candidate_fight_results",
            ),
            accepted_statistics=read_csv_required(
                accepted_stats_path,
                label="accepted_fight_statistics",
            ),
            candidate_statistics=read_csv_required(
                candidate_stats_path,
                label="candidate_fight_statistics",
            ),
            accepted_roster=read_csv_required(
                accepted_roster_path,
                label="accepted_active_roster",
            ),
            candidate_roster=read_csv_required(
                candidate_roster_path,
                label="candidate_active_roster",
            ),
            accepted_profiles=read_csv_required(
                accepted_scraped_fighters_path,
                label="accepted_fighter_profiles",
            ),
            candidate_profiles=read_csv_required(
                scraped_fighters_path,
                label="candidate_fighter_profiles",
            ),
            accepted_processed=read_csv_required(
                accepted_processed_dir / "fights_cleaned.csv",
                label="accepted_processed_fights",
            ),
            candidate_processed=read_csv_required(
                staged_processed_dir / "fights_cleaned.csv"
                if staged_processed_dir is not None
                else Path("__missing_staged_processed_fights__.csv"),
                label="candidate_processed_fights",
            ),
            accepted_features=read_csv_required(
                accepted_processed_dir / "features.csv",
                label="accepted_processed_features",
            ),
            candidate_features=read_csv_required(
                staged_processed_dir / "features.csv"
                if staged_processed_dir is not None
                else Path("__missing_staged_processed_features__.csv"),
                label="candidate_processed_features",
            ),
            coverage_snapshot=coverage_snapshot,
            coverage_floors=coverage_floors,
            fighter_ids_with_completed_fights=completed_profile_ids,
            event_date_lookup=event_date_evidence,
        )
    except Exception as exc:
        publication_quality = {
            "passed": False,
            "errors": [f"quality_gate_execution_failed:{type(exc).__name__}:{exc}"],
            "checks": {},
            "coverage_snapshot": coverage_snapshot,
            "coverage_floors": coverage_floors,
        }

    outcome_reasons = _refresh_outcome_reasons(
        limit_fighters=limit_fighters,
        roster_summary=roster_summary,
        backfill_summary=backfill_summary,
        profile_supplement_summary=profile_supplement_summary,
        rebuild_summary=rebuild_summary,
        staged_processed_dir=staged_processed_dir,
        audit_summary=audit_summary,
        row_drop_guard=row_drop_guard,
        generation_cleanup=generation_cleanup,
        identity_integrity_guard=identity_integrity_guard,
        skip_rebuild=skip_rebuild,
        skip_audit=skip_audit,
        upcoming_summary=upcoming_summary,
    )
    if backfill_summary.get("result_quarantine_blocked") is True:
        _append_outcome_reason(outcome_reasons, "incoming_result_quarantined")
    if publication_quality.get("passed") is not True:
        _append_outcome_reason(outcome_reasons, "publication_quality_gate_failed")
    if raw_generation_cleanup.get("failures"):
        _append_outcome_reason(outcome_reasons, "raw_refresh_generation_cleanup_failures")
    continuity_green = not outcome_reasons
    publication_ready = bool(continuity_green and staged_processed_dir is not None)
    published = False
    publication_action = "not_published"
    publication_noop = False
    published_processed_dir: Path | None = None
    production_bundle_summary: dict[str, object] | None = None
    data_publication_receipt: dict[str, object] | None = None
    if publication_ready and should_update_production_manifest:
        if baseline_lineage is None:
            raise RuntimeError(
                "refusing to publish without the approved v7 accepted-data lineage"
            )
        accepted_at_utc = datetime.now(timezone.utc).isoformat()
        update_report = _accepted_update_report(
            audit_alert_summary=audit_alert_summary,
            backfill_summary=backfill_summary,
            coverage_snapshot=coverage_snapshot,
            coverage_floors=coverage_floors,
        )
        refresh_input_inventory = _refresh_input_artifact_inventory(candidate_raw_dir)
        if (
            publication_manifest_bytes_before is None
            or publication_manifest_payload_before is None
        ):
            raise RuntimeError(
                "accepted production manifest snapshot is unavailable at publication"
            )
        if (
            publication_manifest_path.read_bytes()
            != publication_manifest_bytes_before
        ):
            raise RuntimeError("Active production manifest changed during refresh")
        boundary_lineage = _active_approved_v7_baseline_lineage(required=True)
        if boundary_lineage != baseline_lineage:
            raise RuntimeError("Approved v7 baseline lineage changed during refresh")

        pair_marker_unchanged = False
        if accepted_pair_manifest_bytes_before is None:
            if accepted_pair_manifest_path.exists():
                raise RuntimeError(
                    "Accepted results/statistics pair marker changed during refresh"
                )
        else:
            try:
                accepted_pair_manifest_bytes_now = (
                    accepted_pair_manifest_path.read_bytes()
                )
            except OSError as exc:
                raise RuntimeError(
                    "Accepted results/statistics pair marker changed during refresh"
                ) from exc
            if (
                accepted_pair_manifest_bytes_now
                != accepted_pair_manifest_bytes_before
            ):
                raise RuntimeError(
                    "Accepted results/statistics pair marker changed during refresh"
                )
            accepted_pair_payload = verify_csv_pair_manifest(
                accepted_raw_dir / raw_results_stats_pair.results_path,
                accepted_raw_dir / raw_results_stats_pair.stats_path,
                manifest_path=accepted_pair_manifest_path,
            )
            candidate_pair_payload = verify_csv_pair_manifest(
                candidate_raw_dir / raw_results_stats_pair.results_path,
                candidate_raw_dir / raw_results_stats_pair.stats_path,
                manifest_path=(
                    candidate_raw_dir / raw_results_stats_pair.manifest_path
                ),
            )
            pair_marker_unchanged = bool(
                accepted_pair_payload is not None
                and candidate_pair_payload is not None
            )

        accepted_publication = publication_manifest_payload_before.get(
            "accepted_data_publication"
        )
        if not isinstance(accepted_publication, dict):
            raise RuntimeError("accepted data publication metadata is missing")

        accepted_generation = _accepted_data_generation_identity(staged_processed_dir)
        accepted_raw_generation = _accepted_raw_generation_identity(
            refresh_input_inventory
        )
        processed_identity = get_processed_snapshot_fingerprints(
            staged_processed_dir
        )
        recorded_raw_generation_id = str(
            accepted_publication.get("accepted_raw_generation_id") or ""
        ).strip()
        raw_generation_matches = bool(
            recorded_raw_generation_id
            == accepted_raw_generation["accepted_raw_generation_id"]
            or (
                int(accepted_publication.get("schema_version") or 1) == 1
                and not recorded_raw_generation_id
            )
        )
        candidate_matches_accepted = bool(
            calendar_claim is None
            and pair_marker_unchanged
            and raw_generation_matches
            and accepted_publication.get("refresh_input_inventory")
            == refresh_input_inventory
            and accepted_publication.get("accepted_generation_id")
            == accepted_generation["accepted_generation_id"]
            and all(
                accepted_publication.get(field) == value
                for field, value in processed_identity.items()
            )
        )
        if candidate_matches_accepted:
            accepted_at_utc = str(
                accepted_publication.get("accepted_at_utc") or ""
            ).strip()
            if not accepted_at_utc:
                raise RuntimeError(
                    "accepted data publication timestamp is missing"
                )
            publication_action = "unchanged"
            publication_noop = True
            published_processed_dir = accepted_processed_dir
            data_publication_receipt = {
                **accepted_generation,
                **accepted_raw_generation,
                "accepted_at_utc": accepted_at_utc,
                "observed_unchanged_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "processed_dir": str(accepted_processed_dir),
                "update_report": accepted_publication.get("update_report"),
                "attempt_update_report": update_report,
                "raw": {
                    "accepted_root": str(accepted_raw_dir),
                    "candidate_root": str(candidate_raw_dir),
                    "status": "unchanged",
                    "reason": "candidate_refresh_inputs_match_accepted_inventory",
                    "pointer": None,
                    "artifacts": [],
                    "pair_manifest": {
                        "path": str(accepted_pair_manifest_path),
                        **_file_content_identity(
                            accepted_pair_manifest_path
                        ),
                    },
                },
                "refresh_input_inventory": refresh_input_inventory,
                "baseline_lineage": dict(baseline_lineage),
                "publication_action": publication_action,
            }
            published = True
            logger.info(
                "Candidate raw and processed artifacts match the accepted "
                "generation; leaving the production manifest unchanged"
            )
        else:
            production_bundle_summary, processed_pointer = (
                _prepare_validated_processed_generation(
                    staged_processed_dir,
                    update_report=update_report,
                    accepted_at_utc=accepted_at_utc,
                    calendar_claim=calendar_claim,
                    baseline_lineage=baseline_lineage,
                    refresh_input_inventory=refresh_input_inventory,
                )
            )
            with publish_raw_artifacts_transaction(
                accepted_root=accepted_raw_dir,
                candidate_root=candidate_raw_dir,
                artifacts=raw_publication_artifacts,
                pair=raw_results_stats_pair,
                pointer=processed_pointer,
            ) as raw_receipt:
                pass
            if production_bundle_summary is not None:
                prepared_generation_id = str(
                    production_bundle_summary.get("accepted_generation_id") or ""
                ).strip()
                if prepared_generation_id != accepted_generation["accepted_generation_id"]:
                    raise RuntimeError(
                        "Accepted processed generation changed after manifest prevalidation"
                    )
            publication_action = "committed"
            published_processed_dir = staged_processed_dir
            data_publication_receipt = {
                **accepted_generation,
                **accepted_raw_generation,
                "accepted_at_utc": accepted_at_utc,
                "processed_dir": str(staged_processed_dir),
                "update_report": update_report,
                "raw": raw_receipt.as_dict(),
                "refresh_input_inventory": refresh_input_inventory,
                "publication_action": publication_action,
            }
            if calendar_claim is not None:
                data_publication_receipt["calendar_claim"] = dict(calendar_claim)
            if baseline_lineage is not None:
                data_publication_receipt["baseline_lineage"] = dict(baseline_lineage)
            published = True
            if rebuild_summary is not None:
                rebuild_summary["production_bundle"] = production_bundle_summary
    if not continuity_green:
        logger.warning(
            "UFC refresh continuity gate remained non-green; staged generation was not published: %s",
            outcome_reasons,
        )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "outcome_status": "green" if continuity_green else "incomplete",
        "continuity_green": continuity_green,
        "outcome_reasons": outcome_reasons,
        "publication_requested": should_update_production_manifest,
        "publication_ready": publication_ready,
        "published": published,
        "publication_action": publication_action,
        "publication_noop": publication_noop,
        "requested_output_subdirs": list(requested_output_subdirs),
        "materialized_output_subdirs": list(rebuild_output_subdirs),
        "staged_processed_dir": str(staged_processed_dir) if staged_processed_dir else "",
        "staged_raw_dir": str(candidate_raw_dir),
        "active_processed_dir": (
            str(published_processed_dir) if published_processed_dir else ""
        ),
        "limit_fighters": int(limit_fighters) if limit_fighters is not None else None,
        "partial_refresh": bool(limit_fighters is not None),
        "resolved_paths": resolved_paths,
        "pre_refresh_file_state": pre_refresh_state,
        "post_refresh_file_state": post_refresh_state,
        "row_drop_guard": row_drop_guard,
        "generation_cleanup": generation_cleanup,
        "raw_generation_cleanup": raw_generation_cleanup,
        "raw_stage_receipt": raw_stage_receipt.as_dict(),
        "read_only_raw_stage_receipt": read_only_raw_stage_receipt,
        "publication_quality": publication_quality,
        "coverage_snapshot": coverage_snapshot,
        "coverage_floors": coverage_floors,
        "data_publication_receipt": data_publication_receipt,
        "publication_recovery": publication_recovery,
        "processed_identity_integrity_guard": identity_integrity_guard,
        "processed_history_preservation": processed_history_preservation,
        "processed_history_preservation_attempts": (
            processed_history_preservation_attempts
        ),
        "seed_stale_scraped_fighters": seed_summary,
        "seed_stale_profile_supplement": profile_supplement_seed_summary,
        "roster_sync": roster_summary,
        "upcoming_card_snapshot": upcoming_summary,
        "ufcstats_backfill": backfill_summary,
        "profile_supplement_refresh": profile_supplement_summary,
        "rebuild": rebuild_summary,
        "production_bundle": production_bundle_summary,
        "profile_audit": audit_summary,
        "profile_audit_alert_summary": audit_alert_summary,
        "profile_unresolved_report": unresolved_summary,
        "profile_audit_json_path": str(audit_json_path) if audit_json_path is not None else "",
        "profile_audit_csv_path": str(audit_csv_path) if audit_csv_path is not None else "",
        "profile_unresolved_json_path": str(unresolved_json_path) if unresolved_json_path is not None else "",
        "profile_unresolved_csv_path": str(unresolved_csv_path) if unresolved_csv_path is not None else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-variant",
        default=DEFAULT_DATASET_VARIANT,
        help="Which pulled UFC training-row universe to rebuild.",
    )
    parser.add_argument(
        "--output-subdir",
        action="append",
        default=None,
        help=(
            "Additional isolated processed output subdir(s) to materialize; "
            "the publication candidate is always built in an unaddressed generation."
        ),
    )
    parser.add_argument("--limit-fighters", type=int, default=None)
    parser.add_argument(
        "--refresh-existing-profiles",
        action="store_true",
        help="Refresh changed nonblank UFCStats facts for already-complete profiles.",
    )
    parser.add_argument("--skip-rebuild", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--audit-json-path", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--audit-csv-path", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--unresolved-json-path", type=Path, default=DEFAULT_UNRESOLVED_PROFILE_JSON)
    parser.add_argument("--unresolved-csv-path", type=Path, default=DEFAULT_UNRESOLVED_PROFILE_CSV)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_scheduled_refresh(
        dataset_variant=args.dataset_variant,
        output_subdirs=args.output_subdir,
        limit_fighters=args.limit_fighters,
        audit_json_path=args.audit_json_path,
        audit_csv_path=args.audit_csv_path,
        unresolved_json_path=args.unresolved_json_path,
        unresolved_csv_path=args.unresolved_csv_path,
        skip_rebuild=args.skip_rebuild,
        skip_audit=args.skip_audit,
        refresh_existing_profiles=args.refresh_existing_profiles,
    )
    _write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("continuity_green") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
