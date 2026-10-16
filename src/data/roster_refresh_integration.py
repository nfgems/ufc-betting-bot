"""Offline bridge: protected refresh cycle -> full processing -> quality -> publication -> completion.

This module connects two completed bodies of work without adding a source request:

* the protected acquisition workflow (``roster_hosted_*``: store, controller, stages, recurrence),
  which owns request admission, exact response retention, accounting and restart recovery; and
* the existing full pipeline (saved-evidence candidate builder, raw staging, isolated rebuild,
  publication quality gate and the journaled raw/pointer publication transaction).

The bridge never talks to a source.  Processing runs under a process-level network guard, reads
only bytes the protected store already retained, and writes one exclusive, atomically linked and
fsynced receipt per stage so a restart resumes from durable evidence instead of repeating
acquisition or publication.  One cycle runs under an exclusive lock.

Outcomes are distinct and durable (``completion.json``)::

    uncertain            store carries an uncertain request     -> nothing is processed or completed
    source_failure       reconciled complete membership failure -> accepted state untouched
    processing_failure   processing raised a deterministic error-> retained inputs kept, no reacquisition
    quality_rejected     quality gate did not pass              -> no publication
    publication_rejected referenced identity evidence missing   -> rejected before the transaction
    published            changed candidate committed            -> new accepted generation
    unchanged            candidate equals accepted              -> accepted bytes, generation and
                                                                   original publication time preserved

The data-only publication pointer carries the accepted manifest's model section forward unchanged.
Model artifacts are never loaded here; semantic model acceptance belongs to the release stage.
Processing is only valid within 24 hours of the retained membership request (the candidate
builder's ceiling); a later restart records a processing failure rather than reacquiring.

``network_forbidden`` patches process-global transports; the bridge is meant to run in the
dedicated restricted worker process, not inside a server thread pool.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import socket
import uuid

import pandas as pd

from src.data.refresh_publication import (
    PublicationPointer,
    RawArtifact,
    publication_journal_path,
    publish_raw_artifacts_transaction,
    recover_pending_publication,
    stage_accepted_raw_artifacts,
)
from src.data.roster_hosted_store import file_pin, verify_pin
from src.data.rosterwatch_identity_evidence import STORE_DIRECTORY as IDENTITY_EVIDENCE_DIRECTORY
from src.data.ufc_data_quality import (
    configured_coverage_floors,
    coverage_snapshot_from_audit,
    publication_quality_report,
    read_csv_required,
)
from src.model.production_bundle import (
    _processed_identity_integrity_guard,
    get_processed_snapshot_fingerprints,
)

logger = logging.getLogger(__name__)

SCHEMA = 1
RETAINED_INPUTS_KIND = "roster_refresh_retained_inputs_v1"
PROCESSING_KIND = "roster_refresh_processing_v1"
QUALITY_KIND = "roster_refresh_quality_decision_v1"
PUBLICATION_KIND = "roster_refresh_publication_v1"
UNCHANGED_KIND = "roster_refresh_unchanged_success_v1"
COMPLETION_KIND = "roster_refresh_cycle_completion_v1"
FAILURE_KIND = "roster_refresh_stage_failure_v1"
RECOVERY_KIND = "roster_refresh_publication_recovery_v1"
BASELINE_KIND = "roster_refresh_accepted_baseline_v1"
INVENTORY_SCHEMA = "ufc_refresh_input_inventory_v1"
PUBLISHED_BY = "roster_refresh_integration"

OUTCOME_UNCERTAIN = "uncertain"
OUTCOME_SOURCE_FAILURE = "source_failure"
OUTCOME_PROCESSING_FAILURE = "processing_failure"
OUTCOME_QUALITY_REJECTED = "quality_rejected"
OUTCOME_PUBLICATION_REJECTED = "publication_rejected"
OUTCOME_PUBLISHED = "published"
OUTCOME_UNCHANGED = "unchanged"
OUTCOMES = frozenset({
    OUTCOME_UNCERTAIN, OUTCOME_SOURCE_FAILURE, OUTCOME_PROCESSING_FAILURE, OUTCOME_QUALITY_REJECTED,
    OUTCOME_PUBLICATION_REJECTED, OUTCOME_PUBLISHED, OUTCOME_UNCHANGED,
})
SUCCESS_OUTCOMES = frozenset({OUTCOME_PUBLISHED, OUTCOME_UNCHANGED})

RETAINED_INPUTS_FILE = "retained-inputs-manifest.json"
PROCESSING_FILE = "processing-receipt.json"
QUALITY_FILE = "quality-decision.json"
PUBLICATION_FILE = "publication-receipt.json"
UNCHANGED_FILE = "unchanged-success.json"
COMPLETION_FILE = "completion.json"
BASELINE_FILE = "accepted-baseline.json"
LOCK_FILE = "lock"
ATTEMPTS_SUBDIR = "processing-attempts"
ROSTER_FILE = "ufc_active_roster_official.csv"
ROSTERWATCH_SUBDIR = "rosterwatch"
IDENTITY_CAPTURES_FILE = "rosterwatch_identity_captures.json"
MAX_CANDIDATE_WINDOW_SECONDS = 86400
PUBLISHER_SCRATCH_SUFFIXES = (".publish.tmp", ".rollback.bak", ".stage.tmp", ".pointer.tmp", ".recover.tmp")

# Manifest keys the data-only pointer is allowed to change; everything else is carried forward.
DATA_ONLY_POINTER_KEYS = frozenset({
    "processed_dir", "processed_fights_sha256", "processed_features_sha256",
    "processed_fights_bytes", "processed_features_bytes", "update_report", "accepted_data_publication",
})


def _review_filename() -> str:
    from src.config import ROSTERWATCH_IDENTITY_REVIEW_PATH

    return Path(ROSTERWATCH_IDENTITY_REVIEW_PATH).name


# Per-observation Roster Watch evidence that a valid *unchanged* membership legitimately rewrites.
# These files are retained with the cycle (candidate raw generation) and are excluded from the
# unchanged comparison; every operational artifact (roster CSV, registry/ledger decision columns,
# learned aliases, results, statistics, profiles, pre-UFC artifacts, processed generation, pair
# marker, identity-evidence closure) must still be identical.
UNCHANGED_COMPARISON_EXCLUDED_PREFIXES = (f"{ROSTERWATCH_SUBDIR}/snapshots/",)


def unchanged_comparison_excluded_files() -> frozenset[str]:
    return frozenset({
        f"{ROSTERWATCH_SUBDIR}/rosterwatch_state.json",
        f"{ROSTERWATCH_SUBDIR}/{_review_filename()}",
        f"{ROSTERWATCH_SUBDIR}/rosterwatch_delta_log.jsonl",
        f"{ROSTERWATCH_SUBDIR}/rosterwatch_recurring_reconciliation_ledger.csv",
        # Captured identity pages are acquisition evidence retained with the cycle; any identity
        # decision they produce lands in the owner registry, which IS compared.
        f"{ROSTERWATCH_SUBDIR}/{IDENTITY_CAPTURES_FILE}",
    })


# Bookkeeping columns an identical membership legitimately rewrites (measured on the ordinary sync:
# an identical later-date membership, with or without pending owners, changes exactly these and
# nothing else).  Decision columns (owner id, carriers, status, decision/binding JSON, pending_since,
# conflict flags, current flags) are still compared cell by cell.
UNCHANGED_COMPARISON_OBSERVATION_COLUMNS = {
    f"{ROSTERWATCH_SUBDIR}/rosterwatch_owner_registry.csv": frozenset({
        "last_seen_cutoff", "last_identity_check", "identity_source_checks_json",
        "identity_candidate_pages_json", "identity_check_reasons_json"}),
    f"{ROSTERWATCH_SUBDIR}/rosterwatch_membership_ledger.csv": frozenset({
        "ever_seen", "was_current_previous_snapshot", "legacy_owner_key",
        "legacy_primary_row_ordinal", "all_legacy_row_ordinals", "match_method"}),
}


class IntegrationError(RuntimeError):
    """The bridge refused to continue; nothing was acquired or published."""


class UncertainAcquisition(IntegrationError):
    """The store carries an uncertain request; no processing may start."""


class ProcessingFailure(IntegrationError):
    """Processing failed after successful acquisition; retained inputs are preserved."""


class QualityRejected(IntegrationError):
    """The quality gate rejected the candidate; nothing was published."""


class IdentityEvidenceClosureError(IntegrationError):
    """Referenced identity evidence is missing or corrupt; publication is refused."""


class CycleBusy(IntegrationError):
    """Another runner holds this cycle's lock."""


class NetworkForbidden(AssertionError):
    """Processing attempted a network operation."""


# --------------------------------------------------------------------------------------
# small durable-file helpers
# --------------------------------------------------------------------------------------

def _require(condition: object, message: str) -> None:
    if not condition:
        raise IntegrationError(message)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new(path: Path, content: bytes) -> dict[str, object]:
    """Atomic exclusive create: complete read-only bytes appear under ``path`` or nothing does.

    The bytes are written and fsynced under a temporary name, made read-only, then renamed to
    the final name (which fails if the name already exists) so a crash can never leave a partial
    or writable receipt under the durable name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o400)
        from src.data.roster_refresh_files import publish_new

        publish_new(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)
    return file_pin(path)


def _save(path: Path, value: object) -> dict[str, object]:
    return _write_new(path, _json_bytes(value))


def _load(pin: Mapping[str, object]) -> object:
    return json.loads(Path(str(verify_pin(pin)["path"])).read_bytes())


def _load_path(path: Path) -> object:
    return json.loads(path.read_bytes())


def _time(value: object) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, "timestamps require a timezone")
    return parsed.astimezone(timezone.utc)


def _identity(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": _sha(content)}


def _tree_files(root: Path) -> dict[str, dict[str, object]]:
    """Relative posix path -> identity for every regular file below ``root``."""
    result: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise IntegrationError(f"symbolic link inside accepted or candidate tree: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = _identity(path)
    return result


def _scheduled():
    """Import the selected scheduled-refresh helpers lazily (heavy module)."""
    import scripts.run_scheduled_ufc_refresh as scheduled

    return scheduled


def _rebuild():
    from scripts.rebuild_ufc_processed_artifacts import run_rebuild

    return run_rebuild


@contextmanager
def network_forbidden():
    """Fail closed on any socket/HTTP attempt for the duration of processing (process-global)."""
    import http.client
    import urllib.request

    import requests

    def forbidden(*_args, **_kwargs):
        raise NetworkForbidden("processing must consume retained inputs only; network is forbidden")

    patched = [
        (socket, "create_connection", socket.create_connection),
        (socket.socket, "connect", socket.socket.connect),
        (socket.socket, "connect_ex", socket.socket.connect_ex),
        (http.client.HTTPConnection, "connect", http.client.HTTPConnection.connect),
        (requests.sessions.Session, "request", requests.sessions.Session.request),
        (urllib.request, "urlopen", urllib.request.urlopen),
    ]
    for owner, name, _original in patched:
        setattr(owner, name, forbidden)
    try:
        yield
    finally:
        for owner, name, original in patched:
            setattr(owner, name, original)


@contextmanager
def cycle_lock(cycle_root: Path):
    """Exclusive per-cycle lock; a second runner is refused instead of racing the receipts."""
    cycle_root.mkdir(parents=True, exist_ok=True)
    handle = (cycle_root / LOCK_FILE).open("a+b")
    try:
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:  # pragma: no cover - Windows fallback
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise CycleBusy(f"cycle is already running: {cycle_root}") from exc
        yield
    finally:
        handle.close()


# --------------------------------------------------------------------------------------
# identity evidence closure and artifact lists
# --------------------------------------------------------------------------------------

def identity_evidence_closure(state_root: Path | str) -> dict[str, object]:
    """Enumerate and verify every identity-evidence object the Roster Watch state references.

    Returns the complete verified closure (referenced manifests and chunks plus the whole store,
    because the store verifies itself on open) relative to ``state_root``.  Missing or corrupt
    referenced objects, unparseable reference sources and a failed store verification all raise
    :class:`IdentityEvidenceClosureError`; callers must refuse publication then.
    """
    from src.data.rosterwatch_identity_evidence import REFERENCE_PREFIX, IdentityEvidenceStore

    state_root = Path(state_root)
    store_root = state_root / IDENTITY_EVIDENCE_DIRECTORY
    references: set[str] = set()
    marker = b"roster-identity-evidence:"
    try:
        def scan_csv(name: str) -> None:
            path = state_root / name
            if not path.is_file() or marker not in path.read_bytes():
                return
            import csv
            import io

            previous = csv.field_size_limit()
            try:
                csv.field_size_limit(134217728)
                for row in csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"), newline="")):
                    for value in row.values():
                        if isinstance(value, str) and value.startswith("roster-identity-evidence:"):
                            references.add(value)
            finally:
                csv.field_size_limit(previous)

        scan_csv("rosterwatch_owner_registry.csv")
        scan_csv("rosterwatch_membership_ledger.csv")
        aliases = state_root / "rosterwatch_learned_aliases.json"
        if aliases.is_file():
            payload = _load_path(aliases)
            for alias in (payload.get("aliases") if isinstance(payload, dict) else None) or []:
                decision = alias.get("decision") if isinstance(alias, dict) else None
                reference = decision.get("identity_evidence_reference") if isinstance(decision, dict) else None
                if isinstance(reference, str) and reference.startswith("roster-identity-evidence:"):
                    references.add(reference)
        originals = store_root / "legacy-originals.json"
        if originals.is_file():
            payload = _load_path(originals)
            files = (payload.get("files") if isinstance(payload, dict) else None) or {}
            for entry in files.values():
                if isinstance(entry, dict) and entry.get("present") is True and isinstance(entry.get("reference"), str):
                    references.add(str(entry["reference"]))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise IdentityEvidenceClosureError(f"identity evidence reference source is unreadable: {exc}") from exc

    if not references and not store_root.exists():
        return {"state_root": str(state_root), "store_present": False, "references": [],
                "artifacts": [], "objects": {}, "closure_sha256": _sha(b"")}
    if references and not store_root.is_dir():
        raise IdentityEvidenceClosureError("referenced identity evidence store is absent")
    try:
        store = IdentityEvidenceStore(store_root)
    except Exception as exc:  # the store verifies itself on open and fails closed
        raise IdentityEvidenceClosureError(f"identity evidence store failed verification: {exc}") from exc
    objects: dict[str, dict[str, object]] = {}
    for reference in sorted(references):
        if not reference.startswith(REFERENCE_PREFIX):
            raise IdentityEvidenceClosureError(f"unsupported identity evidence reference: {reference}")
        digest = reference[len(REFERENCE_PREFIX):]
        manifest_path = store_root / "objects" / f"{digest}.json"
        if not manifest_path.is_file():
            raise IdentityEvidenceClosureError(f"referenced identity evidence manifest is missing: {reference}")
        manifest_bytes = manifest_path.read_bytes()
        if _sha(manifest_bytes) != digest:
            raise IdentityEvidenceClosureError(f"referenced identity evidence manifest is corrupt: {reference}")
        try:
            manifest = json.loads(manifest_bytes)
        except ValueError as exc:
            raise IdentityEvidenceClosureError(f"referenced identity evidence manifest is unreadable: {reference}") from exc
        objects[f"{IDENTITY_EVIDENCE_DIRECTORY}/objects/{digest}.json"] = {"bytes": len(manifest_bytes), "sha256": digest}
        for chunk in (manifest.get("chunks") if isinstance(manifest, dict) else None) or []:
            chunk_digest = str(chunk.get("sha256") or "") if isinstance(chunk, dict) else ""
            chunk_path = store_root / "objects" / f"{chunk_digest}.bin"
            if not chunk_digest or not chunk_path.is_file():
                raise IdentityEvidenceClosureError(f"referenced identity evidence chunk is missing: {chunk_digest or '?'}")
            chunk_bytes = chunk_path.read_bytes()
            if _sha(chunk_bytes) != chunk_digest or len(chunk_bytes) != int(chunk.get("bytes") or -1):
                raise IdentityEvidenceClosureError(f"referenced identity evidence chunk is corrupt: {chunk_digest}")
            objects[f"{IDENTITY_EVIDENCE_DIRECTORY}/objects/{chunk_digest}.bin"] = {"bytes": len(chunk_bytes), "sha256": chunk_digest}
        try:
            store.get_bytes(reference)
        except Exception as exc:
            raise IdentityEvidenceClosureError(f"referenced identity evidence does not resolve: {exc}") from exc
    for path in sorted(store_root.rglob("*")):
        if path.is_file():
            objects.setdefault(path.relative_to(state_root).as_posix(), _identity(path))
    canonical = json.dumps(objects, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"state_root": str(state_root), "store_present": True, "references": sorted(references),
            "artifacts": sorted(objects), "objects": objects, "closure_sha256": _sha(canonical)}


def raw_publication_artifacts_with_evidence(*roots: Path) -> tuple[RawArtifact, ...]:
    """The selected raw publication list plus the complete identity-evidence store of each root."""
    scheduled = _scheduled()
    artifacts = list(scheduled._raw_publication_artifacts(*roots))
    declared = {os.path.normcase(artifact.relative_path) for artifact in artifacts}
    for root in roots:
        closure = identity_evidence_closure(Path(root) / ROSTERWATCH_SUBDIR)
        for relative in closure["artifacts"]:
            relative_path = f"{ROSTERWATCH_SUBDIR}/{relative}"
            if os.path.normcase(relative_path) not in declared:
                artifacts.append(RawArtifact(relative_path, required=False))
                declared.add(os.path.normcase(relative_path))
    return tuple(artifacts)


def refresh_input_inventory(raw_root: Path) -> dict[str, object]:
    """The selected refresh-input inventory, unchanged.

    The accepted manifest's ``refresh_input_inventory`` is recomputed by the existing lineage
    validator with this exact function, so the bridge records the identity-evidence closure as a
    separate field (``identity_evidence_closure_sha256``) instead of extending the inventory.
    """
    inventory = _scheduled()._refresh_input_artifact_inventory(Path(raw_root))
    _require(inventory.get("schema_version") == INVENTORY_SCHEMA, "unexpected refresh input inventory schema")
    return inventory


def sweep_publisher_scratch(state_root: Path) -> list[str]:
    """Remove publisher scratch files left inside the accepted evidence store by a hard kill.

    The publisher stages every artifact as a same-directory temporary file; the evidence store
    refuses to open while such a file sits in its root or in ``objects/``.  Only files carrying the
    publisher's scratch suffixes and a non-store name are removed, and only after any journal is gone.
    """
    store_root = Path(state_root) / IDENTITY_EVIDENCE_DIRECTORY
    objects = store_root / "objects"
    removed: list[str] = []
    if not store_root.is_dir():
        return removed
    expected_root = {"format.json", "objects", "legacy-originals.json"}
    for directory, keep in ((store_root, lambda name: name in expected_root),
                            (objects, lambda name: name.partition(".")[1] and name.partition(".")[2] in {"bin", "json"}
                                                   and len(name.partition(".")[0]) == 64)):
        if not directory.is_dir():
            continue
        swept = False
        for path in sorted(directory.iterdir()):
            if keep(path.name):
                continue
            if path.is_file() and any(path.name.endswith(marker) for marker in PUBLISHER_SCRATCH_SUFFIXES):
                path.unlink()
                removed.append(path.relative_to(store_root).as_posix())
                swept = True
        if swept:
            _fsync_directory(directory)
    return removed


# --------------------------------------------------------------------------------------
# stage 1: retained inputs
# --------------------------------------------------------------------------------------

def cycle_identity(binding_pin: Mapping[str, object]) -> str:
    binding = _load(binding_pin)
    root = _load(binding["root_plan"])
    return f"{root['slot_id']}-{str(binding['root_plan']['sha256'])[:12]}"


def collect_retained_inputs(store, binding_pin: Mapping[str, object], cycle_root: Path, *, now: datetime) -> dict[str, object]:
    """Bind one completed protected cycle's exact retained receipts into an immutable manifest.

    No source request, charge, stage or store mutation occurs.  The cycle outcome is recomputed by
    the recurrence adapter (``validated_cycle_status``), which validates the chain at its original
    time and therefore survives the original root lifetime.
    """
    from src.data.roster_hosted_needs import BASE_ROLES, MEMBERSHIP_URL
    from src.data.roster_hosted_recurrence import validated_cycle_status

    now = _time(now)
    cycle_root = Path(cycle_root)
    target = cycle_root / RETAINED_INPUTS_FILE
    if target.is_file():
        return {"manifest": file_pin(target), "existing": True, **_load_path(target)}
    state = store.snapshot()
    if state["uncertain_requests"]:
        raise UncertainAcquisition(
            "uncertain requests are charged and unresolved; no processing, replay or completion: "
            + ", ".join(sorted(state["uncertain_requests"])))
    binding = _load(binding_pin)
    _require(binding.get("mode") == "disabled_controller_v1" and binding.get("store_root") == store.root.as_posix(),
             "Original controller binding required")
    root_plan = _load(binding["root_plan"])
    outcome = validated_cycle_status(store, binding_pin, now=now)
    _require(store.snapshot() == state, "Accounting changed while binding retained inputs")
    pins = dict(outcome["input_pins"] if outcome["status"] == "succeeded" else outcome["prior_inputs"])
    _require(set(pins) == set(BASE_ROLES), "Complete base input roles required")
    for role in BASE_ROLES:
        verify_pin(pins[role])
    receipts = []
    for request_id, charge in sorted(state["requests"].items()):
        if not str(charge.get("plan_id", "")).startswith(root_plan["plan_id"]):
            continue
        result = state["results"].get(request_id)
        _require(result is not None, f"charged request without retained result: {request_id}")
        row = {"request_id": request_id, "plan_id": charge["plan_id"], "source": charge["source"],
               "method": charge["method"], "url": charge["url"], "charged_at_utc": charge["at_utc"],
               "outcome": result["outcome"], "result_at_utc": result["at_utc"], "body": verify_pin(result["blob"])}
        for key in ("envelope", "raw_response"):
            if key in result:
                row[key] = verify_pin(result[key])
        if "envelope" in result:
            metadata = _load(result["envelope"]).get("metadata", {})
            row["response_status"] = metadata.get("response", {}).get("status")
            row["request_started_at"] = metadata.get("request", {}).get("started_at")
            row["response_completed_at"] = metadata.get("response", {}).get("completed_at")
            row["response_complete"] = metadata.get("response", {}).get("complete")
        receipts.append(row)
    membership_input = _load(pins["membership"])
    membership = {"descriptor": pins["membership"], "body": verify_pin(membership_input["body"]),
                  "envelope": verify_pin(membership_input["envelope"])}
    envelope = _load(membership_input["envelope"])
    membership.update({
        "plan_id": envelope.get("plan_id"),
        "request_id": envelope.get("request", {}).get("request_id"),
        "url": envelope.get("request", {}).get("url"),
        "request_started_at": envelope.get("request", {}).get("started_at"),
        "response_completed_at": envelope.get("response", {}).get("completed_at"),
        "response_status": envelope.get("response", {}).get("status"),
        "body_sha256": envelope.get("response", {}).get("body_sha256"),
    })
    _require(membership["url"] == MEMBERSHIP_URL, "membership descriptor must bind the canonical Roster Watch GET")
    _require(membership["body_sha256"] == membership["body"]["sha256"], "membership envelope/body hash mismatch")
    owners = _load(pins["owners"])
    manifest = {
        "schema": SCHEMA, "kind": RETAINED_INPUTS_KIND, "cycle_id": cycle_identity(binding_pin),
        "collected_at_utc": now.isoformat(), "binding": binding_pin, "root_plan": binding["root_plan"],
        "root_review": binding["root_review"], "slot_id": root_plan["slot_id"], "plan_id": root_plan["plan_id"],
        "issued_at": root_plan["issued_at"], "expires_at": root_plan["expires_at"],
        "store_root": store.root.as_posix(), "store_head": state["head"],
        "accepted_pointer": binding["accepted_pointer"], "expected_release_sha256": binding["expected_release_sha256"],
        "acquisition_status": outcome["status"], "input_pins": pins, "membership": membership,
        "owners": {"pin": pins["owners"], "seed_root": owners["seed_root"], "seed_inventory": owners["seed_inventory"]},
        "catalog": pins["catalog"], "release": pins["release"], "request_receipts": receipts,
        "pending_identity_rows": outcome.get("pending_identity_rows", []),
        "source_requests": 0, "publication_authorized": False,
    }
    if outcome["status"] == "failed":
        manifest["failure"] = file_pin(Path(str(binding["output_root"])) / "reconciled-membership-failure.json")
        manifest["failure_reason"] = outcome.get("reason")
        manifest["retry_authorized"] = False
    cycle_root.mkdir(parents=True, exist_ok=True)
    pin = _save(target, manifest)
    return {"manifest": pin, "existing": False, **manifest}


# --------------------------------------------------------------------------------------
# stage 2: processing (retained inputs only)
# --------------------------------------------------------------------------------------

def _seed_matches_accepted_raw(seed_root: Path, accepted_raw_root: Path) -> dict[str, object]:
    """The protected accepted seed and the pipeline's accepted raw roster tree must be one baseline.

    The identity capture catalog travels as the separate ``catalog`` input pin and is not part of
    the seed comparison.
    """
    seed_files = _tree_files(seed_root)
    accepted_subset: dict[str, dict[str, object]] = {}
    roster = accepted_raw_root / ROSTER_FILE
    _require(roster.is_file(), "accepted raw roster is missing")
    accepted_subset[ROSTER_FILE] = _identity(roster)
    state_root = accepted_raw_root / ROSTERWATCH_SUBDIR
    if state_root.is_dir():
        for relative, identity in _tree_files(state_root).items():
            accepted_subset[f"{ROSTERWATCH_SUBDIR}/{relative}"] = identity
    captures = f"{ROSTERWATCH_SUBDIR}/{IDENTITY_CAPTURES_FILE}"
    seed_files.pop(captures, None)
    accepted_subset.pop(captures, None)
    differences = sorted(relative for relative in set(seed_files) | set(accepted_subset)
                         if seed_files.get(relative) != accepted_subset.get(relative))
    _require(not differences, "protected accepted seed differs from the accepted raw roster tree: "
             + ", ".join(differences[:5]))
    return {"seed_root": str(seed_root), "files": len(seed_files)}


def _require_event_date_cache_complete(candidate_raw: Path) -> dict[str, object]:
    """Every candidate result event must already be in the retained event-date cache.

    ``build_training_rows_from_pulled_data`` would otherwise fetch missing event metadata from
    UFCStats; a missing input is reported explicitly here instead of becoming a hidden request.
    """
    from src.data.ufc_refresh import _load_event_metadata_cache, _normalize_name

    scheduled = _scheduled()
    results_path = candidate_raw / scheduled.BACKFILL_RESULTS_PATH.name
    cache_path = candidate_raw / "ufc-event-dates.csv"
    results = pd.read_csv(results_path) if results_path.is_file() else pd.DataFrame()
    required = {_normalize_name(value) for value in results.get("EVENT", pd.Series(dtype=object)) if _normalize_name(value)}
    cached = set(_load_event_metadata_cache(cache_path)) if cache_path.is_file() else set()
    missing = sorted(required - cached)
    _require(not missing, "retained event-date cache does not cover candidate events (no fetch fallback): "
             + ", ".join(missing[:5]))
    return {"cache": str(cache_path), "required_events": len(required), "cached_events": len(cached)}


def _copy_tree_over(source: Path, target: Path) -> list[str]:
    written: list[str] = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise IntegrationError(f"symbolic link inside candidate tree: {path}")
        if path.is_file():
            relative = path.relative_to(source)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            shutil.copyfile(path, temporary)
            os.replace(temporary, destination)
            written.append(relative.as_posix())
    return written


def _new_attempt_root(cycle_root: Path) -> tuple[Path, int]:
    attempts = cycle_root / ATTEMPTS_SUBDIR
    attempts.mkdir(parents=True, exist_ok=True)
    ordinal = 1 + sum(1 for child in attempts.iterdir() if child.is_dir() and child.name.startswith("attempt-"))
    attempt = attempts / f"attempt-{ordinal:03d}-{uuid.uuid4().hex[:8]}"
    attempt.mkdir()
    return attempt, ordinal


def prune_retained_generations(integration_root: Path, *, current_raw: Path | None = None) -> dict[str, object]:
    """Run the selected generation pruners with every cycle-referenced generation protected."""
    scheduled = _scheduled()
    protected = protected_generation_dirs(integration_root)
    summary: dict[str, object] = {"protected_count": len(protected)}
    fence = "Complete accepted/training/monthly/recovery reference inventory is not yet adopted"
    summary["processed"] = scheduled._prune_refresh_generations(protected=protected, fence_reason=fence)
    if current_raw is not None:
        summary["raw"] = scheduled._prune_raw_refresh_generations(current_raw, protected=protected, fence_reason=fence)
    return summary


def process_retained_inputs(
    manifest_pin: Mapping[str, object],
    *,
    accepted_raw_root: Path,
    accepted_processed_root: Path,
    cycle_root: Path,
    clock: Callable[[], datetime],
    integration_root: Path | None = None,
    dataset_variant: str | None = None,
    identity_check_cap: int = 40,
) -> dict[str, object]:
    """Run the existing full processing on retained inputs with the network guard installed.

    Every invocation without a processing receipt is a fresh attempt in its own directory, so an
    earlier crashed attempt is retained as evidence and never turns a later restart into a
    fabricated processing failure.
    """
    from src.data import roster_hosted_candidate as candidate

    cycle_root = Path(cycle_root)
    target = cycle_root / PROCESSING_FILE
    if target.is_file():
        return {"receipt": file_pin(target), "existing": True, **_load_path(target)}
    manifest = _load(manifest_pin)
    _require(manifest.get("kind") == RETAINED_INPUTS_KIND, "retained inputs manifest required")
    if manifest["acquisition_status"] != "succeeded":
        raise IntegrationError("processing is only defined for a successful acquisition")
    scheduled = _scheduled()
    run_rebuild = _rebuild()
    accepted_raw_root = Path(accepted_raw_root).resolve(strict=True)
    accepted_processed_root = Path(accepted_processed_root).resolve(strict=True)
    attempt_root, ordinal = _new_attempt_root(cycle_root)
    started = _time(clock())
    dataset_variant = dataset_variant or scheduled.DEFAULT_DATASET_VARIANT
    receipt: dict[str, object] = {
        "schema": SCHEMA, "kind": PROCESSING_KIND, "cycle_id": manifest["cycle_id"], "manifest": manifest_pin,
        "attempt": ordinal, "attempt_root": str(attempt_root),
        "started_at_utc": started.isoformat(), "source_requests": 0, "network_guard": "installed",
        "dataset_variant": dataset_variant, "accepted_raw_root": str(accepted_raw_root),
        "accepted_processed_root": str(accepted_processed_root),
        "candidate_evidence_mode": "offline_regression",
        "candidate_evidence_mode_note": ("plan/request identity and original capture timing are bound by the "
                                         "retained-inputs manifest; the builder's current-timing check is not repeated"),
    }
    try:
        with network_forbidden():
            seed_root = Path(str(manifest["owners"]["seed_root"]))
            receipt["baseline"] = _seed_matches_accepted_raw(seed_root, accepted_raw_root)
            storage = "shared_v1" if (seed_root / ROSTERWATCH_SUBDIR / IDENTITY_EVIDENCE_DIRECTORY).exists() else "legacy_inline"
            requested = _time(manifest["membership"]["request_started_at"])
            spec = {
                "schema_version": 1, "publication_requested": False, "network_requests": 0,
                "evidence_mode": "offline_regression", "identity_evidence_storage": storage,
                "seed_root": str(seed_root), "seed_inventory": manifest["owners"]["seed_inventory"],
                "output_root": str(attempt_root / "roster-candidate"),
                "membership_body": manifest["membership"]["body"], "membership_envelope": manifest["membership"]["envelope"],
                "identity_catalog": manifest["catalog"], "max_membership_age_seconds": MAX_CANDIDATE_WINDOW_SECONDS,
                "max_identity_age_seconds": 604800, "identity_check_cap": int(identity_check_cap),
                "run_started_at": requested.isoformat(),
                "expires_at": (requested + timedelta(seconds=MAX_CANDIDATE_WINDOW_SECONDS)).isoformat(),
                "acquisition_plan_id": manifest["membership"]["plan_id"],
                "bridge_binding": {"cycle_id": manifest["cycle_id"], "request_id": manifest["membership"]["request_id"]},
            }
            spec_pin = _save(attempt_root / "roster-candidate-spec.json", spec)
            report = candidate.prepare_candidate(spec_pin["path"], spec_pin["sha256"], clock=clock)
            _require(report["status"] == "prepared_candidate_only" and report["publication_performed"] is False
                     and report["network_requests"] == 0, "ordinary candidate preparation did not complete")
            prepared = Path(str(spec["output_root"])) / "candidate"
            receipt["roster"] = {
                "terminal": file_pin(Path(str(spec["output_root"])) / "candidate-terminal.json"),
                "spec": spec_pin, "current_members": report["current_members"], "total_owners": report["total_owners"],
                "pending_members": report["pending_members"], "identity_evidence_storage": report["identity_evidence_storage"],
                "membership_observed_at": report["membership_observed_at"],
            }
            if integration_root is not None:
                receipt["generation_pruning_before_staging"] = prune_retained_generations(Path(integration_root))
            candidate_raw = scheduled._new_raw_refresh_generation_dir(now=_time(clock()))
            candidate_raw.mkdir(parents=True)
            artifacts = raw_publication_artifacts_with_evidence(accepted_raw_root)
            pair = scheduled._raw_results_stats_pair()
            stage_receipt = stage_accepted_raw_artifacts(
                accepted_root=accepted_raw_root, candidate_root=candidate_raw, artifacts=artifacts, pair=pair)
            read_only = scheduled._stage_read_only_rebuild_inputs(accepted_root=accepted_raw_root, candidate_root=candidate_raw)
            overlay = _copy_tree_over(prepared, candidate_raw)
            receipt["staging"] = {"candidate_raw_root": str(candidate_raw), "raw_stage": stage_receipt.as_dict(),
                                  "read_only_inputs": read_only["artifacts"], "roster_overlay": overlay}
            if integration_root is not None:
                receipt["generation_pruning_after_staging"] = prune_retained_generations(
                    Path(integration_root), current_raw=candidate_raw)
            receipt["event_date_cache"] = _require_event_date_cache_complete(candidate_raw)
            staged = scheduled._new_refresh_generation_dir(now=_time(clock()))
            rebuild_summary = run_rebuild(
                dataset_variant=dataset_variant, output_subdirs=["."], update_production_manifest=False,
                raw_data_dir=candidate_raw, processed_data_dir=staged,
                base_dataset_path=candidate_raw / "ufc-master.csv",
                fight_results_path=candidate_raw / scheduled.BACKFILL_RESULTS_PATH.name,
                fight_stats_path=candidate_raw / scheduled.BACKFILL_STATS_PATH.name,
                scraped_fighters_path=candidate_raw / scheduled.BACKFILL_FIGHTERS_PATH.name,
                supplemental_profiles_path=candidate_raw / scheduled.PROFILE_SUPPLEMENT_PATH.name,
                official_active_roster_path=candidate_raw / ROSTER_FILE,
                pierce_fight_identities_path=candidate_raw / "github_PierceHampton.csv",
                event_dates_cache_path=candidate_raw / "ufc-event-dates.csv",
                historical_odds_dir=candidate_raw / "historical_odds",
                rankings_history_path=candidate_raw / "kaggle_rankings" / "rankings_history_extended.csv",
                method_odds_history_path=candidate_raw / "method_odds" / "historical_method_odds_all.csv",
                isolated_staging=True)
            rebuild_summary.pop("production_bundle", None)
            preservation = scheduled._preserve_rebuild_processed_history(
                accepted_dir=accepted_processed_root, staged_root=staged, rebuild_summary=rebuild_summary)
            from src.data.provider_consumer_references import provider_reference_inventory
            from src.data.observation_consumer_references import observation_reference_inventory
            receipt.update({
                "staged_processed_root": str(staged), "rebuild": rebuild_summary,
                "processed_history_preservation": preservation,
                "candidate_inventory": refresh_input_inventory(candidate_raw),
                "staged_processed_identity": get_processed_snapshot_fingerprints(staged),
                "provider_reference_inventory": provider_reference_inventory(staged),
                "observation_reference_inventory": observation_reference_inventory(staged),
                "identity_evidence_closure": identity_evidence_closure(candidate_raw / ROSTERWATCH_SUBDIR),
                "completed_at_utc": _time(clock()).isoformat(),
            })
    except NetworkForbidden as exc:
        _save(attempt_root / "processing-failed.json",
              {"schema": SCHEMA, "kind": FAILURE_KIND, "stage": "processing", "attempt": ordinal,
               "error": f"NetworkForbidden: {exc}", "retained_inputs_preserved": True, "reacquisition_authorized": False,
               "partial": {k: v for k, v in receipt.items() if k in ("roster", "staging")}})
        raise ProcessingFailure(f"processing attempted a network request: {exc}") from exc
    except Exception as exc:
        _save(attempt_root / "processing-failed.json",
              {"schema": SCHEMA, "kind": FAILURE_KIND, "stage": "processing", "attempt": ordinal,
               "error": f"{type(exc).__name__}: {exc}", "retained_inputs_preserved": True,
               "reacquisition_authorized": False, "partial": {k: v for k, v in receipt.items() if k in ("roster", "staging")}})
        raise ProcessingFailure(f"processing failed after successful acquisition: {type(exc).__name__}: {exc}") from exc
    pin = _save(target, receipt)
    return {"receipt": pin, "existing": False, **receipt}


# --------------------------------------------------------------------------------------
# stage 3: quality decision
# --------------------------------------------------------------------------------------

def _accepted_relative_identity_guard(accepted_processed: Path, staged: Path) -> dict[str, object]:
    """Require exactly the stable collision identities the accepted generation already carries.

    The production guard's default sentinel set describes the production dataset.  A candidate must
    never lose a sentinel that the accepted generation has; sentinels absent from the accepted
    generation (isolated fixtures, partial datasets) are reported, not fabricated as requirements.
    """
    from src.model.production_bundle import _REQUIRED_COLLISION_IDENTITY_SENTINELS

    accepted_guard = _processed_identity_integrity_guard(accepted_processed)
    missing_marker = "required_collision_identity_missing:"
    missing_in_accepted = {violation.split(missing_marker, 1)[1] for violation in accepted_guard.get("violations") or []
                           if missing_marker in violation}
    required = frozenset(sentinel for sentinel in _REQUIRED_COLLISION_IDENTITY_SENTINELS
                         if f"{sentinel[0]}:{sentinel[1]}" not in missing_in_accepted)
    guard = _processed_identity_integrity_guard(staged, required_collision_identities=required)
    guard["required_collision_identities"] = sorted(f"{name}:{identity}" for name, identity in required)
    guard["sentinels_absent_from_accepted_generation"] = sorted(missing_in_accepted)
    guard["accepted_generation_violations"] = list(accepted_guard.get("violations") or [])
    return guard


def decide_quality(
    processing_pin: Mapping[str, object],
    *,
    accepted_raw_root: Path,
    accepted_processed_root: Path,
    cycle_root: Path,
    clock: Callable[[], datetime],
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run the existing side-effect-free publication gate on accepted vs candidate artifacts."""
    cycle_root = Path(cycle_root)
    target = cycle_root / QUALITY_FILE
    if target.is_file():
        return {"decision": file_pin(target), "existing": True, **_load_path(target)}
    processing = _load(processing_pin)
    _require(processing.get("kind") == PROCESSING_KIND, "processing receipt required")
    scheduled = _scheduled()
    accepted_raw = Path(accepted_raw_root)
    accepted_processed = Path(accepted_processed_root)
    candidate_raw = Path(str(processing["staging"]["candidate_raw_root"]))
    staged = Path(str(processing["staged_processed_root"]))
    decision: dict[str, object] = {"schema": SCHEMA, "kind": QUALITY_KIND, "cycle_id": processing["cycle_id"],
                                   "processing": processing_pin, "decided_at_utc": _time(clock()).isoformat(),
                                   "audit_coverage": "not_completed"}
    report: dict[str, object]
    try:
        floors = configured_coverage_floors(environ)
        from src.data.roster_refresh_audit import measure

        audit_pin, snapshot, audit_alerts = measure(processing_pin, cycle_root=cycle_root, clock=clock)
        decision.update(audit_coverage="measured", coverage_audit=audit_pin,
                        coverage_snapshot=snapshot, coverage_floors=floors,
                        audit_alert_summary=audit_alerts)
        report = publication_quality_report(
            accepted_results=read_csv_required(accepted_raw / scheduled.BACKFILL_RESULTS_PATH.name, label="accepted_fight_results"),
            candidate_results=read_csv_required(candidate_raw / scheduled.BACKFILL_RESULTS_PATH.name, label="candidate_fight_results"),
            accepted_statistics=read_csv_required(accepted_raw / scheduled.BACKFILL_STATS_PATH.name, label="accepted_fight_statistics"),
            candidate_statistics=read_csv_required(candidate_raw / scheduled.BACKFILL_STATS_PATH.name, label="candidate_fight_statistics"),
            accepted_roster=read_csv_required(accepted_raw / ROSTER_FILE, label="accepted_active_roster"),
            candidate_roster=read_csv_required(candidate_raw / ROSTER_FILE, label="candidate_active_roster"),
            accepted_profiles=read_csv_required(accepted_raw / scheduled.BACKFILL_FIGHTERS_PATH.name, label="accepted_fighter_profiles"),
            candidate_profiles=read_csv_required(candidate_raw / scheduled.BACKFILL_FIGHTERS_PATH.name, label="candidate_fighter_profiles"),
            accepted_processed=read_csv_required(accepted_processed / "fights_cleaned.csv", label="accepted_processed_fights"),
            candidate_processed=read_csv_required(staged / "fights_cleaned.csv", label="candidate_processed_fights"),
            accepted_features=read_csv_required(accepted_processed / "features.csv", label="accepted_processed_features"),
            candidate_features=read_csv_required(staged / "features.csv", label="candidate_processed_features"),
            coverage_snapshot=snapshot, coverage_floors=floors,
            fighter_ids_with_completed_fights=set(),
            event_date_lookup=scheduled._event_date_evidence_lookup(candidate_raw / "ufc-event-dates.csv"),
        )
        guard = _accepted_relative_identity_guard(accepted_processed, staged)
        decision["execution_failed"] = False
    except Exception as exc:
        report = {"passed": False, "errors": [f"quality_gate_execution_failed:{type(exc).__name__}:{exc}"], "checks": {}}
        guard = {"violations": ["quality_gate_execution_failed"]}
        decision["execution_failed"] = True
    report_pin = _save(cycle_root / f"quality-report-{uuid.uuid4().hex}.json", report)
    passed = bool(report.get("passed") is True and not guard.get("violations"))
    decision.update({
        "passed": passed, "errors": list(report.get("errors") or []) + [f"processed_identity_guard:{v}" for v in guard.get("violations") or []],
        "checks": {name: bool(isinstance(check, dict) and check.get("passed") is True)
                   for name, check in (report.get("checks") or {}).items()},
        "processed_identity_guard": guard, "report": report_pin,
    })
    pin = _save(target, decision)
    return {"decision": pin, "existing": False, **decision}


# --------------------------------------------------------------------------------------
# stage 4: publication or unchanged success
# --------------------------------------------------------------------------------------

def _accepted_publication(payload: Mapping[str, object]) -> dict[str, object]:
    publication = payload.get("accepted_data_publication")
    _require(isinstance(publication, dict), "accepted data publication metadata is missing")
    _require(str(publication.get("accepted_at_utc") or "").strip(), "accepted data publication timestamp is missing")
    return dict(publication)


def _operational_inventory(inventory: Mapping[str, object]) -> dict[str, object]:
    """Drop per-observation Roster Watch evidence from an inventory before an unchanged comparison."""
    excluded = unchanged_comparison_excluded_files()
    artifacts = inventory.get("artifacts") or {}
    return {relative: identity for relative, identity in artifacts.items()
            if relative not in excluded and relative not in UNCHANGED_COMPARISON_OBSERVATION_COLUMNS
            and not any(relative.startswith(prefix) for prefix in UNCHANGED_COMPARISON_EXCLUDED_PREFIXES)}


def _semantic_csv_differences(accepted_root: Path, candidate_root: Path, accepted_inventory: Mapping[str, object]) -> dict[str, list[str]]:
    """Compare the registry/ledger ignoring only their declared observation columns.

    The accepted files on disk must also match the accepted manifest's recorded inventory, so the
    semantic comparison never runs against drifted accepted bytes.
    """
    import csv
    import io

    differences: dict[str, list[str]] = {}
    recorded = accepted_inventory.get("artifacts") or {}
    for relative, ignored in UNCHANGED_COMPARISON_OBSERVATION_COLUMNS.items():
        accepted_path = Path(accepted_root) / relative
        candidate_path = Path(candidate_root) / relative
        if not accepted_path.is_file() and not candidate_path.is_file():
            continue
        if accepted_path.is_file() != candidate_path.is_file():
            differences[relative] = ["presence"]
            continue
        recorded_identity = recorded.get(relative)
        if recorded_identity is not None and recorded_identity.get("sha256") != _identity(accepted_path)["sha256"]:
            differences[relative] = ["accepted_state_drift"]
            continue
        previous = csv.field_size_limit()
        try:
            csv.field_size_limit(134217728)
            frames = [list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"), newline="")))
                      for path in (accepted_path, candidate_path)]
        finally:
            csv.field_size_limit(previous)
        accepted_rows, candidate_rows = frames
        accepted_columns = list(accepted_rows[0].keys()) if accepted_rows else []
        candidate_columns = list(candidate_rows[0].keys()) if candidate_rows else []
        reasons: list[str] = []
        if accepted_columns != candidate_columns:
            reasons.append("columns")
        if len(accepted_rows) != len(candidate_rows):
            reasons.append("row_count")
        if not reasons:
            changed = sorted({column for left, right in zip(accepted_rows, candidate_rows) for column in accepted_columns
                              if column not in ignored and left.get(column, "") != right.get(column, "")})
            reasons.extend(f"column:{column}" for column in changed)
        if reasons:
            differences[relative] = reasons
    return differences


def candidate_matches_accepted(
    *,
    accepted_publication: Mapping[str, object],
    accepted_raw_root: Path,
    candidate_raw_root: Path,
    candidate_inventory: Mapping[str, object],
    staged_processed_root: Path,
    candidate_closure: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The selected unchanged rule without the ``calendar_claim is None`` condition.

    Accepted-generation identity, raw-generation identity, the results/statistics pair marker, the
    operational raw inventory and the identity-evidence closure must all be identical.  New
    per-observation Roster Watch evidence (snapshots, state pointer, review, delta log, recurring
    ledger, capture catalog, bookkeeping columns) is retained with the cycle and is not a data
    change; every other artifact must match byte for byte.
    """
    from src.data.io_utils import verify_csv_pair_manifest

    scheduled = _scheduled()
    pair = scheduled._raw_results_stats_pair()
    accepted_marker = Path(accepted_raw_root) / pair.manifest_path
    candidate_marker = Path(candidate_raw_root) / pair.manifest_path
    pair_unchanged = False
    if accepted_marker.is_file() and candidate_marker.is_file():
        accepted_pair = verify_csv_pair_manifest(
            Path(accepted_raw_root) / pair.results_path, Path(accepted_raw_root) / pair.stats_path, manifest_path=accepted_marker)
        candidate_pair = verify_csv_pair_manifest(
            Path(candidate_raw_root) / pair.results_path, Path(candidate_raw_root) / pair.stats_path, manifest_path=candidate_marker)
        pair_unchanged = bool(accepted_pair is not None and candidate_pair is not None
                              and [r["sha256"] for r in accepted_pair["artifacts"]] == [r["sha256"] for r in candidate_pair["artifacts"]])
    accepted_generation = scheduled._accepted_data_generation_identity(Path(staged_processed_root))
    processed_identity = get_processed_snapshot_fingerprints(Path(staged_processed_root))
    accepted_inventory = accepted_publication.get("refresh_input_inventory") or {}
    operational_accepted = _operational_inventory(accepted_inventory)
    operational_candidate = _operational_inventory(candidate_inventory)
    differences = sorted(relative for relative in set(operational_accepted) | set(operational_candidate)
                         if operational_accepted.get(relative) != operational_candidate.get(relative))
    semantic = _semantic_csv_differences(Path(accepted_raw_root), Path(candidate_raw_root), accepted_inventory)
    differences.extend(f"{relative}:{'+'.join(reasons)}" for relative, reasons in sorted(semantic.items()))
    accepted_artifacts = accepted_inventory.get("artifacts") or {}
    candidate_artifacts = candidate_inventory.get("artifacts") or {}
    observation_only = sorted(relative for relative in set(accepted_artifacts) | set(candidate_artifacts)
                              if accepted_artifacts.get(relative) != candidate_artifacts.get(relative)
                              and relative not in differences)
    checks = {
        "pair_marker_unchanged": pair_unchanged,
        "accepted_generation_id_matches": accepted_publication.get("accepted_generation_id") == accepted_generation["accepted_generation_id"],
        "processed_identity_matches": all(accepted_publication.get(field) == value for field, value in processed_identity.items()),
        "operational_inventory_matches": not differences,
    }
    if candidate_closure is not None:
        accepted_closure = identity_evidence_closure(Path(accepted_raw_root) / ROSTERWATCH_SUBDIR)
        recorded = accepted_publication.get("identity_evidence_closure_sha256")
        checks["identity_evidence_closure_matches"] = bool(
            accepted_closure["closure_sha256"] == candidate_closure.get("closure_sha256")
            and (recorded in (None, "") or recorded == accepted_closure["closure_sha256"]))
    return {"unchanged": all(checks.values()), "checks": checks, "operational_differences": differences,
            "observation_only_differences": observation_only,
            "observation_columns_ignored": {k: sorted(v) for k, v in UNCHANGED_COMPARISON_OBSERVATION_COLUMNS.items()},
            "candidate_generation_id": accepted_generation["accepted_generation_id"]}


def data_only_pointer(
    accepted_manifest_path: Path,
    *,
    staged_processed_root: Path,
    accepted_generation_id: str,
    accepted_raw_generation_id: str,
    accepted_at_utc: str,
    update_report: Mapping[str, object],
    refresh_input_inventory: Mapping[str, object],
    calendar_claim: Mapping[str, object] | None,
    baseline_lineage: Mapping[str, object] | None,
    processed_identity: Mapping[str, object],
    cycle_id: str,
    identity_evidence_closure_sha256: str | None = None,
    identity_evidence_artifact_count: int = 0,
) -> tuple[PublicationPointer, dict[str, object]]:
    """Build the pointer bytes for a data-only update; the model section is carried forward as-is.

    ``reconcile_production_bundle_manifest`` loads model artifacts, which this bridge never does.
    The release stage owns semantic model acceptance; here every non-data key stays byte-identical.
    """
    accepted_manifest_path = Path(accepted_manifest_path)
    before_bytes = accepted_manifest_path.read_bytes()
    before = json.loads(before_bytes.decode("utf-8"))
    _require(isinstance(before, dict), "accepted production manifest is not a JSON object")
    payload = dict(before)
    payload["processed_dir"] = str(Path(staged_processed_root).resolve(strict=False))
    payload.update(processed_identity)
    payload["update_report"] = dict(update_report)
    publication = {
        "schema_version": 2, "accepted_generation_id": accepted_generation_id,
        "accepted_raw_generation_id": accepted_raw_generation_id, "accepted_at_utc": accepted_at_utc,
        "processed_dir": payload["processed_dir"], "update_report": dict(update_report),
        "refresh_input_inventory": dict(refresh_input_inventory), **processed_identity,
        "published_by": PUBLISHED_BY, "bridge_cycle_id": cycle_id, "model_section_carried_forward": True,
        "identity_evidence_closure_sha256": identity_evidence_closure_sha256,
        "identity_evidence_artifact_count": int(identity_evidence_artifact_count),
    }
    if calendar_claim is not None:
        publication["calendar_claim"] = dict(calendar_claim)
    if baseline_lineage is not None:
        publication["baseline_lineage"] = dict(baseline_lineage)
    payload["accepted_data_publication"] = publication
    carried = {key: value for key, value in before.items() if key not in DATA_ONLY_POINTER_KEYS}
    _require(all(payload.get(key) == value for key, value in carried.items()), "model section drifted in data-only pointer")
    content = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    pointer = PublicationPointer(target_path=accepted_manifest_path, content=content,
                                 journal_path=publication_journal_path(accepted_manifest_path))
    summary = {"manifest_before_sha256": _sha(before_bytes), "pointer_sha256": _sha(content),
               "carried_forward_keys": sorted(carried), "carried_forward_sha256": _sha(_json_bytes(carried))}
    return pointer, summary


def _publication_already_committed(accepted_manifest_path: Path, *, cycle_id: str, staged: Path) -> dict[str, object] | None:
    """Detect that the accepted pointer already addresses this cycle's staged generation."""
    payload = json.loads(Path(accepted_manifest_path).read_bytes().decode("utf-8"))
    publication = payload.get("accepted_data_publication") if isinstance(payload, dict) else None
    if not isinstance(publication, dict):
        return None
    if (publication.get("published_by") == PUBLISHED_BY and publication.get("bridge_cycle_id") == cycle_id
            and publication.get("processed_dir") == str(Path(staged).resolve(strict=False))):
        return publication
    return None


def _bind_accepted_roots(accepted_raw_root: Path, accepted_processed_root: Path) -> None:
    """The bridge must publish into the exact trees the selected helpers address."""
    scheduled = _scheduled()
    _require(Path(accepted_raw_root).resolve() == Path(scheduled.RAW_DATA_DIR).resolve(),
             "accepted raw root must be the configured RAW_DATA_DIR")
    active = scheduled._active_production_processed_dir() or Path(scheduled.PROCESSED_DATA_DIR)
    _require(Path(accepted_processed_root).resolve() == Path(active).resolve(),
             "accepted processed root must be the manifest-addressed active processed directory")


def publish_or_record_unchanged(quality_pin, **kwargs):
    """Publish one verified stage and retain every ordinary refusal at this API boundary."""
    try:
        return _publish_or_record_unchanged(quality_pin, **kwargs)
    except Exception as exc:
        _record_publication_refusal(kwargs["cycle_root"], exc, clock=kwargs["clock"],
                                    stage="publication", quality_pin=quality_pin)
        raise


def _publish_or_record_unchanged(
    quality_pin: Mapping[str, object],
    *,
    accepted_raw_root: Path,
    accepted_processed_root: Path,
    accepted_manifest_path: Path,
    cycle_root: Path,
    clock: Callable[[], datetime],
    calendar_claim: Mapping[str, object] | None = None,
    update_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Publish a changed candidate through the journaled transaction, or record an unchanged success."""
    cycle_root = Path(cycle_root)
    publication_target = cycle_root / PUBLICATION_FILE
    unchanged_target = cycle_root / UNCHANGED_FILE
    if publication_target.is_file():
        return {"receipt": file_pin(publication_target), "existing": True, "outcome": OUTCOME_PUBLISHED, **_load_path(publication_target)}
    if unchanged_target.is_file():
        return {"receipt": file_pin(unchanged_target), "existing": True, "outcome": OUTCOME_UNCHANGED, **_load_path(unchanged_target)}
    decision = _load(quality_pin)
    _require(decision.get("kind") == QUALITY_KIND, "quality decision required")
    if decision.get("passed") is not True:
        raise QualityRejected("quality gate did not pass: " + "; ".join(str(e) for e in decision.get("errors") or [])[:2000])
    processing = _load(decision["processing"])
    from src.data.roster_refresh_audit import verify as verify_coverage_audit

    verify_coverage_audit(decision["coverage_audit"], processing_pin=decision["processing"], decision=decision)
    _require(get_processed_snapshot_fingerprints(Path(processing["staged_processed_root"]))
             == processing["staged_processed_identity"], "processed generation changed since processing")
    from src.data.provider_consumer_references import provider_reference_inventory
    from src.data.observation_consumer_references import observation_reference_inventory
    _require(observation_reference_inventory(Path(processing["staged_processed_root"]))
             == processing.get("observation_reference_inventory", {}), "processed observation references changed since processing")
    _require(provider_reference_inventory(Path(processing["staged_processed_root"]))
             == processing.get("provider_reference_inventory", {}), "processed provider references changed since processing")
    scheduled = _scheduled()
    accepted_manifest_path = Path(accepted_manifest_path)
    _require(accepted_manifest_path.resolve() == scheduled._active_production_manifest_path().resolve(),
             "accepted manifest path must be the configured active production manifest")
    _bind_accepted_roots(accepted_raw_root, accepted_processed_root)
    calendar_claim = scheduled._validated_calendar_claim(dict(calendar_claim) if calendar_claim else None)
    accepted_raw = Path(accepted_raw_root)
    candidate_raw = Path(str(processing["staging"]["candidate_raw_root"]))
    staged = Path(str(processing["staged_processed_root"]))
    cycle_id = str(processing["cycle_id"])
    pending = publication_journal_path(accepted_manifest_path)
    _require(not pending.exists(), f"pending publication journal must be recovered first: {pending}")
    committed = _publication_already_committed(accepted_manifest_path, cycle_id=cycle_id, staged=staged)
    if committed is not None:
        return _record_committed_publication(cycle_root, publication=committed, processing_pin=decision["processing"],
                                             quality_pin=quality_pin, accepted_manifest_path=accepted_manifest_path,
                                             clock=clock, claim_context=calendar_claim,
                                             recovery={"action": "accepted_pointer_already_addresses_cycle"})
    baseline_lineage = scheduled._active_approved_v7_baseline_lineage(required=True)
    manifest_before = accepted_manifest_path.read_bytes()
    accepted_publication = _accepted_publication(json.loads(manifest_before.decode("utf-8")))
    try:
        # Referenced identity evidence on both sides must resolve completely before any work.
        closure = identity_evidence_closure(candidate_raw / ROSTERWATCH_SUBDIR)
        artifacts = raw_publication_artifacts_with_evidence(accepted_raw, candidate_raw)
    except IdentityEvidenceClosureError as exc:
        _save(cycle_root / f"publication-rejected-{uuid.uuid4().hex}.json",
              {"schema": SCHEMA, "kind": FAILURE_KIND, "stage": "publication", "error": str(exc), "accepted_state_untouched": True})
        raise
    candidate_inventory = refresh_input_inventory(candidate_raw)
    _require(candidate_inventory == processing["candidate_inventory"], "candidate raw generation changed since processing")
    comparison = candidate_matches_accepted(
        accepted_publication=accepted_publication, accepted_raw_root=accepted_raw, candidate_raw_root=candidate_raw,
        candidate_inventory=candidate_inventory, staged_processed_root=staged, candidate_closure=closure)
    pair = scheduled._raw_results_stats_pair()
    now = _time(clock())
    if comparison["unchanged"]:
        accepted_marker = accepted_raw / pair.manifest_path
        receipt = {
            "schema": SCHEMA, "kind": UNCHANGED_KIND, "cycle_id": cycle_id, "outcome": OUTCOME_UNCHANGED,
            "completed_at_utc": now.isoformat(), "calendar_claim": calendar_claim, "quality": quality_pin,
            "processing": decision["processing"],
            "accepted_generation_id": accepted_publication["accepted_generation_id"],
            "accepted_raw_generation_id": accepted_publication.get("accepted_raw_generation_id"),
            "accepted_at_utc": accepted_publication["accepted_at_utc"],
            "original_publication_preserved": True, "publication_performed": False,
            "accepted_manifest": {"path": str(accepted_manifest_path), **_identity(accepted_manifest_path)},
            "accepted_pair_marker": ({"path": str(accepted_marker), **_identity(accepted_marker)} if accepted_marker.is_file() else None),
            "comparison": comparison, "candidate_inventory": candidate_inventory,
            "retained_capture_evidence_root": str(candidate_raw), "retained_staged_processed_root": str(staged),
            "retained_observation_only_files": comparison["observation_only_differences"],
            "identity_evidence_closure_sha256": closure["closure_sha256"],
        }
        _require(accepted_manifest_path.read_bytes() == manifest_before, "accepted manifest changed during unchanged check")
        pin = _save(unchanged_target, receipt)
        return {"receipt": pin, "existing": False, **receipt}
    update = dict(update_report) if update_report is not None else scheduled._accepted_update_report(
        audit_alert_summary=decision["audit_alert_summary"], backfill_summary={},
        coverage_snapshot=decision["coverage_snapshot"], coverage_floors=decision["coverage_floors"])
    raw_generation = scheduled._accepted_raw_generation_identity(candidate_inventory)
    processed_identity = get_processed_snapshot_fingerprints(staged)
    pointer, pointer_summary = data_only_pointer(
        accepted_manifest_path, staged_processed_root=staged,
        accepted_generation_id=str(comparison["candidate_generation_id"]),
        accepted_raw_generation_id=raw_generation["accepted_raw_generation_id"], accepted_at_utc=now.isoformat(),
        update_report=update, refresh_input_inventory=candidate_inventory, calendar_claim=calendar_claim,
        baseline_lineage=baseline_lineage, processed_identity=processed_identity, cycle_id=cycle_id,
        identity_evidence_closure_sha256=closure["closure_sha256"] if closure.get("store_present") else None,
        identity_evidence_artifact_count=len(closure.get("artifacts") or []))
    _require(accepted_manifest_path.read_bytes() == manifest_before, "accepted manifest changed before publication")
    try:
        with publish_raw_artifacts_transaction(
                accepted_root=accepted_raw, candidate_root=candidate_raw, artifacts=artifacts, pair=pair, pointer=pointer) as transaction:
            pass
    except BaseException as exc:
        if isinstance(exc, Exception):
            _save(cycle_root / f"publication-failed-{uuid.uuid4().hex}.json",
                  {"schema": SCHEMA, "kind": FAILURE_KIND, "stage": "publication", "error": f"{type(exc).__name__}: {exc}",
                   "republish_from_retained_candidate_allowed": True, "reacquisition_authorized": False})
        raise
    _require(transaction.status == "committed", f"publication transaction did not commit: {transaction.status}")
    _require(_sha(accepted_manifest_path.read_bytes()) == pointer_summary["pointer_sha256"], "published pointer bytes differ")
    receipt = {
        "schema": SCHEMA, "kind": PUBLICATION_KIND, "cycle_id": cycle_id, "outcome": OUTCOME_PUBLISHED,
        "completed_at_utc": _time(clock()).isoformat(), "calendar_claim": calendar_claim, "quality": quality_pin,
        "processing": decision["processing"], "accepted_generation_id": comparison["candidate_generation_id"],
        "accepted_raw_generation_id": raw_generation["accepted_raw_generation_id"], "accepted_at_utc": now.isoformat(),
        "previous_accepted_generation_id": accepted_publication["accepted_generation_id"],
        "previous_accepted_at_utc": accepted_publication["accepted_at_utc"],
        "transaction": transaction.as_dict(), "pointer": pointer_summary,
        "accepted_manifest": {"path": str(accepted_manifest_path), **_identity(accepted_manifest_path)},
        "comparison": comparison, "identity_evidence_closure_sha256": closure["closure_sha256"],
        "published_artifact_count": len(artifacts), "publication_performed": True, "republished": False,
    }
    pin = _save(publication_target, receipt)
    return {"receipt": pin, "existing": False, **receipt}


def _record_committed_publication(
    cycle_root: Path,
    *,
    publication: Mapping[str, object],
    processing_pin: Mapping[str, object],
    quality_pin: Mapping[str, object],
    accepted_manifest_path: Path,
    clock: Callable[[], datetime],
    claim_context: Mapping[str, object] | None,
    recovery: Mapping[str, object],
) -> dict[str, object]:
    """The accepted pointer already carries this cycle's candidate bytes; record it, never republish."""
    processing = _load(processing_pin)
    staged = Path(str(processing["staged_processed_root"]))
    from src.data.provider_consumer_references import provider_reference_inventory
    from src.data.observation_consumer_references import observation_reference_inventory
    _require(observation_reference_inventory(staged) == processing.get("observation_reference_inventory", {}),
             "processed observation references changed during recovery")
    _require(provider_reference_inventory(staged) == processing.get("provider_reference_inventory", {}),
             "recovered provider reference closure differs from processed original")
    expected = _scheduled()._accepted_data_generation_identity(staged)["accepted_generation_id"]
    _require(publication.get("accepted_generation_id") == expected
             and publication.get("processed_dir") == str(staged.resolve(strict=False)),
             "committed pointer does not address this cycle's staged generation")
    receipt = {
        "schema": SCHEMA, "kind": PUBLICATION_KIND, "cycle_id": processing["cycle_id"], "outcome": OUTCOME_PUBLISHED,
        "completed_at_utc": _time(clock()).isoformat(), "calendar_claim": claim_context, "quality": quality_pin,
        "processing": processing_pin, "accepted_generation_id": publication["accepted_generation_id"],
        "accepted_raw_generation_id": publication.get("accepted_raw_generation_id"),
        "accepted_at_utc": publication["accepted_at_utc"], "recovered_from": dict(recovery),
        "accepted_manifest": {"path": str(accepted_manifest_path), **_identity(Path(accepted_manifest_path))},
        "identity_evidence_closure_sha256": publication.get("identity_evidence_closure_sha256"),
        "publication_performed": True, "republished": False,
    }
    pin = _save(Path(cycle_root) / PUBLICATION_FILE, receipt)
    return {"receipt": pin, "existing": False, **receipt}


# --------------------------------------------------------------------------------------
# stage 5: accepted-baseline advancement and durable completion (+ calendar)
# --------------------------------------------------------------------------------------

def advance_accepted_baseline(
    cycle_root: Path,
    *,
    integration_root: Path,
    store,
    publication_pin: Mapping[str, object],
    manifest_pin: Mapping[str, object],
    accepted_raw_root: Path,
) -> dict[str, object]:
    """Resume the exact baseline intent, preserving partial bytes after interruption."""
    from src.data.roster_refresh_baseline import advance

    return advance(cycle_root, integration_root=integration_root, store=store,
                   publication_pin=publication_pin, manifest_pin=manifest_pin,
                   accepted_raw_root=accepted_raw_root)


def calendar_claim_context(claim: Mapping[str, object]) -> dict[str, object]:
    """The small durable weekly-claim identity the publisher embeds (schema 1)."""
    return {"schema_version": 1, "claim_id": str(claim["claim_id"]), "iso_week": str(claim["iso_week"]),
            "slot": str(claim["slot"]), "claimed_at_utc": str(claim.get("claimed_at_utc") or claim.get("now_utc")),
            "due_at_utc": str(claim["due_at_utc"])}


def complete_cycle(
    cycle_root: Path,
    *,
    outcome: str,
    clock: Callable[[], datetime],
    stage_pins: Mapping[str, object],
    schedule=None,
    claim_id: str | None = None,
    monthly_input_request: Mapping[str, object] | None = None,
    monthly_input_request_provider=None,
    error: str | None = None,
) -> dict[str, object]:
    """Write the single durable completion record; complete the calendar claim exactly once.

    A published cycle completes the claim with the new generation and its publication time.  An
    unchanged cycle completes the claim with the retained generation and this cycle's completion
    time: the accepted data and its original publication timestamp stay untouched, and the
    qualifying cycle time is what monthly preparation may use.  A calendar refusal is recorded in
    the completion instead of preventing it.
    """
    from src.data.ufc_update_schedule import UFCUpdateScheduleError

    _require(outcome in OUTCOMES, f"unknown outcome: {outcome}")
    cycle_root = Path(cycle_root)
    target = cycle_root / COMPLETION_FILE
    if target.is_file():
        existing = _load_path(target)
        _require(existing.get("outcome") == outcome, f"completion outcome conflict: {existing.get('outcome')} vs {outcome}")
        _require(existing.get("claim_id") == claim_id, "completion claim conflict")
        _require(existing.get("stages") == dict(stage_pins), "completion stages conflict")
        pin = file_pin(target)
        calendar = _reconcile_cycle_calendar(cycle_root, existing, pin, schedule=schedule,
                                             monthly_input_request=monthly_input_request,
                                             monthly_input_request_provider=monthly_input_request_provider, now=_time(clock()))
        return {"completion": pin, "existing": True, **existing, **calendar}
    now = _time(clock())
    record: dict[str, object] = {
        "schema": SCHEMA, "kind": COMPLETION_KIND, "cycle_id": cycle_root.name, "outcome": outcome,
        "completed_at_utc": now.isoformat(), "stages": dict(stage_pins), "claim_id": claim_id,
        "source_requests": 0, "reacquisition_authorized": False, "republication_authorized": False,
    }
    if error:
        record["error"] = error
    receipt_pin = stage_pins.get("publication") or stage_pins.get("unchanged")
    if receipt_pin:
        receipt = _load(receipt_pin)
        record["accepted_generation_id"] = receipt.get("accepted_generation_id")
        record["accepted_at_utc"] = receipt.get("accepted_at_utc")
        record["publication_performed"] = receipt.get("publication_performed")
        record["qualifying_cycle_time_utc"] = receipt.get("accepted_at_utc") if outcome == OUTCOME_PUBLISHED else now.isoformat()
    # The data completion must survive a hard cut before any calendar mutation.
    pin = _save(target, record)
    calendar = _reconcile_cycle_calendar(cycle_root, record, pin, schedule=schedule,
                                         monthly_input_request=monthly_input_request, now=now,
                                         monthly_input_request_provider=monthly_input_request_provider,
                                         newly_completed=True)
    return {"completion": pin, "existing": False, **record, **calendar}


def _reconcile_cycle_calendar(cycle_root, record, completion_pin, *, schedule,
                              monthly_input_request=None, monthly_input_request_provider=None,
                              now, newly_completed=False):
    """Reconcile from the preserved original completion, with no pipeline re-entry."""
    from src.data.ufc_update_schedule import UFCUpdateScheduleError

    if schedule is None or not record.get("claim_id"):
        return {}
    claim_id = record["claim_id"]
    expected_context = None
    provisioning_error = None
    if monthly_input_request_provider is not None and monthly_input_request is None and record["outcome"] in SUCCESS_OUTCOMES:
        # Ordinary completion is durable before lookup; recovery still needs
        # independent authority if its original calendar write was interrupted.
        data_recorded = False
        try:
            if newly_completed:
                schedule.complete_data_update(claim_id, succeeded=True,
                    accepted_update_id=str(record["accepted_generation_id"]),
                    accepted_at=_time(record["qualifying_cycle_time_utc"]), now=now,
                    completion_evidence_pin=completion_pin)
                data_recorded = True
            state = schedule.load_state()
            week, slot, _entry, attempt = schedule._locate_claim(state["weekly"], claim_id, context="weekly")
            context = schedule._monthly_context(accepted_id=record["accepted_generation_id"],
                accepted_at=_time(record["qualifying_cycle_time_utc"]), week=week, slot=slot, claim_id=claim_id, state=state)
            monthly_input_request = monthly_input_request_provider.request_for_claim(claim_id,
                context=context, completion_pin=completion_pin,
                policy_binding=(attempt.get("monthly_policy") or {}).get("binding"))
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            provisioning_error = f"{type(exc).__name__}: {exc}"
            if data_recorded:
                schedule.record_monthly_preparation_refusal(claim_id, provisioning_error, now=now)
    if not newly_completed:
        state = schedule.load_state()
        originals = [attempt.get("completion_evidence_pin")
                     for entry in state["weekly"].values()
                     for attempt in entry.get("slots", {}).values()
                     if attempt.get("claim_id") == claim_id]
        if completion_pin not in originals:
            if monthly_input_request is None:
                return {"schedule_completion": {"calendar_reconciliation_required": True,
                        "error": provisioning_error or "original independently bound completion evidence required",
                        "claim_id": claim_id}}
            from src.model.monthly_input import inspect_monthly_completion
            from src.model import prepared_input
            context = inspect_monthly_completion(monthly_input_request)
            accepted = prepared_input._json(prepared_input._capture(monthly_input_request["acceptance"]))
            _require(accepted["completion"] == completion_pin, "original completion pin conflict")
            _require(context["claim_id"] == claim_id
                     and context["accepted_update_id"] == record.get("accepted_generation_id")
                     and context["accepted_at_utc"] == record.get("qualifying_cycle_time_utc"),
                     "original completion context conflict")
            expected_context = context
    try:
        if record["outcome"] in SUCCESS_OUTCOMES:
            result = schedule.complete_data_update(
                claim_id, succeeded=True, accepted_update_id=str(record["accepted_generation_id"]),
                accepted_at=_time(record["qualifying_cycle_time_utc"]), now=now,
                monthly_input_request=monthly_input_request, completion_evidence_pin=completion_pin,
                expected_monthly_context=expected_context)
        else:
            result = schedule.complete_data_update(
                claim_id, succeeded=False, error=f"{record['outcome']}:{record.get('error', '')}"[:512], now=now,
                completion_evidence_pin=completion_pin)
    except UFCUpdateScheduleError as exc:
        result = {"error": f"{type(exc).__name__}: {exc}", "claim_id": claim_id,
                  "calendar_reconciliation_required": True}
    if provisioning_error is not None:
        result["monthly_provisioning_error"] = provisioning_error
    # Completion bytes are immutable; calendar attempts have their own history.
    history = Path(cycle_root) / "calendar-completions"
    comparable = {"completion": completion_pin, "result": result}
    for prior in sorted(history.glob("*.json")) if history.is_dir() else []:
        saved = _load_path(prior)
        if saved.get("result") == result and saved.get("completion") == completion_pin:
            return {"schedule_completion": result, "schedule_completion_record": file_pin(prior)}
        old = saved.get("result", {})
        monthly = result.get("monthly_registration") or {}
        if (result.get("changed") is False and not monthly.get("changed")
            and old.get("succeeded") == result.get("succeeded")
            and old.get("iso_week") == result.get("iso_week")
            and old.get("slot") == result.get("slot") and saved.get("completion") == completion_pin
            and (old.get("monthly_registration") or {}).get("registered") == monthly.get("registered")):
            return {"schedule_completion": result, "schedule_completion_record": file_pin(prior)}
    record_pin = _save(history / f"attempt-{uuid.uuid4().hex}.json", comparable)
    return {"schedule_completion": result, "schedule_completion_record": record_pin}


# --------------------------------------------------------------------------------------
# orchestration with restart recovery
# --------------------------------------------------------------------------------------

def _pin_if(path: Path) -> dict[str, object] | None:
    return file_pin(path) if path.is_file() else None


def reconcile_publication(
    cycle_root: Path,
    *,
    accepted_manifest_path: Path,
    accepted_raw_root: Path,
    clock: Callable[[], datetime],
) -> dict[str, object]:
    """Recover an interrupted publication from the pointer's journal, without republishing.

    ``roll_forward`` means every artifact and the pointer already carry the committed bytes;
    ``roll_back`` restores the exact accepted bytes.  Once no journal remains, publisher scratch
    files a hard kill left inside the accepted evidence store are removed so the store opens again.
    """
    cycle_root = Path(cycle_root)
    journal = publication_journal_path(Path(accepted_manifest_path))
    result: dict[str, object] = {"action": "none", "journal": str(journal)}
    if journal.exists():
        recovery = recover_pending_publication(journal)
        result = {"action": recovery.get("action"), "journal": str(journal), **recovery}
    if not journal.exists():
        swept = sweep_publisher_scratch(Path(accepted_raw_root) / ROSTERWATCH_SUBDIR)
        if swept:
            result["evidence_store_scratch_removed"] = swept
    if result.get("action") != "none" or result.get("evidence_store_scratch_removed"):
        result["record"] = _save(cycle_root / f"publication-recovery-{uuid.uuid4().hex}.json",
                                 {"schema": SCHEMA, "kind": RECOVERY_KIND, "recovered_at_utc": _time(clock()).isoformat(), **result})
    return result


def _record_publication_refusal(cycle_root, error, *, clock, stage, quality_pin=None):
    """Retain refusals without inventing rollback, completion or fresh acquisition authority."""
    return _save(Path(cycle_root) / f"publication-refusal-{uuid.uuid4().hex}.json", {
        "schema": SCHEMA, "kind": FAILURE_KIND, "cycle_id": Path(cycle_root).name,
        "stage": stage, "recorded_at_utc": _time(clock()).isoformat(),
        "error_type": type(error).__name__, "error": str(error), "quality": quality_pin,
        "accepted_state_requires_reconciliation": True, "cycle_completed": False,
        "retry_from_retained_inputs_only": True, "reacquisition_authorized": False})


def run_cycle(
    store,
    binding_pin: Mapping[str, object],
    *,
    integration_root: Path,
    accepted_raw_root: Path,
    accepted_manifest_path: Path,
    accepted_processed_root: Path | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    schedule=None,
    claim: Mapping[str, object] | None = None,
    monthly_input_request: Mapping[str, object] | None = None,
    monthly_input_request_provider=None,
    environ: Mapping[str, str] | None = None,
    dataset_variant: str | None = None,
) -> dict[str, object]:
    """Drive one protected cycle through the full pipeline to durable completion; safe to re-run.

    ``accepted_processed_root`` defaults to the manifest-addressed active processed directory,
    resolved only after any pending publication journal has been recovered.
    """
    integration_root = Path(integration_root)
    if schedule is not None and monthly_input_request is None and monthly_input_request_provider is None:
        from src.model.monthly_provisioning import configure_monthly_calendar_from_env
        monthly_input_request_provider = configure_monthly_calendar_from_env(environ)["provider"]
    cycle_id = cycle_identity(binding_pin)
    cycle_root = integration_root / "cycles" / cycle_id
    completion = cycle_root / COMPLETION_FILE
    with cycle_lock(cycle_root):
        return _run_cycle_locked(store, binding_pin, cycle_id=cycle_id, cycle_root=cycle_root,
                                 integration_root=integration_root, accepted_raw_root=Path(accepted_raw_root),
                                 accepted_processed_root=Path(accepted_processed_root) if accepted_processed_root else None,
                                 accepted_manifest_path=Path(accepted_manifest_path), clock=clock, schedule=schedule,
                                 claim=claim, monthly_input_request=monthly_input_request,
                                 monthly_input_request_provider=monthly_input_request_provider,
                                 environ=environ, dataset_variant=dataset_variant)


def _run_cycle_locked(store, binding_pin, *, cycle_id, cycle_root, integration_root, accepted_raw_root,
                      accepted_processed_root, accepted_manifest_path, clock, schedule, claim, monthly_input_request,
                      monthly_input_request_provider, environ, dataset_variant):
    completion = cycle_root / COMPLETION_FILE
    if completion.is_file():
        saved = _load_path(completion)
        _require(saved.get("claim_id") == (str(claim["claim_id"]) if claim else None), "completion claim conflict")
        pin = file_pin(completion)
        calendar = _reconcile_cycle_calendar(cycle_root, saved, pin, schedule=schedule,
                                             monthly_input_request=monthly_input_request,
                                             monthly_input_request_provider=monthly_input_request_provider, now=_time(clock()))
        return {"cycle_id": cycle_id, "cycle_root": str(cycle_root), "restarted": True,
                "completion": pin, **saved, **calendar}
    claim_id = str(claim["claim_id"]) if claim else None
    if schedule is not None and claim_id:
        schedule.protect_data_completion(claim_id, now=_time(clock()))
    claim_context = calendar_claim_context(claim) if claim else None
    stage_pins: dict[str, object] = {}
    # 0. an interrupted publication is reconciled before anything reads accepted state
    try:
        recovery = reconcile_publication(cycle_root, accepted_manifest_path=accepted_manifest_path,
                                         accepted_raw_root=accepted_raw_root, clock=clock)
    except Exception as exc:
        _record_publication_refusal(cycle_root, exc, clock=clock, stage="publication_recovery")
        raise
    if "record" in recovery:
        stage_pins["publication_recovery"] = recovery["record"]
    if accepted_processed_root is None:
        scheduled = _scheduled()
        accepted_processed_root = Path(scheduled._active_production_processed_dir() or scheduled.PROCESSED_DATA_DIR)

    def finish(outcome: str, *, error: str | None = None) -> dict[str, object]:
        if outcome == OUTCOME_PUBLISHED and "retained_inputs" in stage_pins:
            baseline = advance_accepted_baseline(cycle_root, integration_root=integration_root, store=store,
                                                 publication_pin=stage_pins["publication"],
                                                 manifest_pin=stage_pins["retained_inputs"], accepted_raw_root=accepted_raw_root)
            stage_pins["accepted_baseline"] = baseline["baseline"]
        done = complete_cycle(cycle_root, outcome=outcome, clock=clock, stage_pins=stage_pins,
                              schedule=schedule, claim_id=claim_id, monthly_input_request=monthly_input_request,
                              monthly_input_request_provider=monthly_input_request_provider, error=error)
        return {"cycle_id": cycle_id, "cycle_root": str(cycle_root), "restarted": False, **done}

    for name, key in ((RETAINED_INPUTS_FILE, "retained_inputs"), (PROCESSING_FILE, "processing"), (QUALITY_FILE, "quality")):
        pin = _pin_if(cycle_root / name)
        if pin:
            stage_pins[key] = pin
    existing_publication = _pin_if(cycle_root / PUBLICATION_FILE)
    existing_unchanged = _pin_if(cycle_root / UNCHANGED_FILE)
    if existing_publication:
        stage_pins["publication"] = existing_publication
        return finish(OUTCOME_PUBLISHED)
    if existing_unchanged:
        stage_pins["unchanged"] = existing_unchanged
        return finish(OUTCOME_UNCHANGED)

    # 1. retained inputs (uncertain acquisition never completes)
    inputs = collect_retained_inputs(store, binding_pin, cycle_root, now=clock())
    stage_pins["retained_inputs"] = inputs["manifest"]
    if inputs["acquisition_status"] == "failed":
        stage_pins["failure"] = inputs["failure"]
        return finish(OUTCOME_SOURCE_FAILURE, error=str(inputs.get("failure_reason") or "membership_failure"))

    # 2. processing (a fresh attempt whenever no receipt exists)
    try:
        processing = process_retained_inputs(
            inputs["manifest"], accepted_raw_root=accepted_raw_root, accepted_processed_root=accepted_processed_root,
            cycle_root=cycle_root, clock=clock, integration_root=integration_root, dataset_variant=dataset_variant)
    except ProcessingFailure as exc:
        return finish(OUTCOME_PROCESSING_FAILURE, error=str(exc))
    stage_pins["processing"] = processing["receipt"]

    # 3. quality
    quality = decide_quality(processing["receipt"], accepted_raw_root=accepted_raw_root,
                             accepted_processed_root=accepted_processed_root, cycle_root=cycle_root, clock=clock, environ=environ)
    stage_pins["quality"] = quality["decision"]
    if quality["passed"] is not True:
        return finish(OUTCOME_QUALITY_REJECTED, error="; ".join(str(e) for e in quality["errors"])[:512])

    # 4. publication (never republish committed work)
    try:
        published = publish_or_record_unchanged(
            quality["decision"], accepted_raw_root=accepted_raw_root, accepted_processed_root=accepted_processed_root,
            accepted_manifest_path=accepted_manifest_path, cycle_root=cycle_root, clock=clock, calendar_claim=claim_context)
    except Exception as exc:
        if isinstance(exc, IdentityEvidenceClosureError):
            return finish(OUTCOME_PUBLICATION_REJECTED, error=str(exc))
        raise
    stage_pins["publication" if published["outcome"] == OUTCOME_PUBLISHED else "unchanged"] = published["receipt"]
    return finish(published["outcome"])


# --------------------------------------------------------------------------------------
# retention protection
# --------------------------------------------------------------------------------------

def referenced_retention(integration_root: Path) -> dict[str, object]:
    """Everything the recorded cycles still reference; pruners must keep all of it."""
    integration_root = Path(integration_root)
    blob_digests: set[str] = set()
    candidate_raw_roots: set[str] = set()
    staged_processed_roots: set[str] = set()
    evidence_roots: set[str] = set()
    cycles: list[str] = []
    cycles_dir = integration_root / "cycles"
    for cycle_root in sorted(cycles_dir.glob("*")) if cycles_dir.is_dir() else []:
        manifest = cycle_root / RETAINED_INPUTS_FILE
        if not manifest.is_file():
            continue
        cycles.append(cycle_root.name)
        payload = _load_path(manifest)
        for receipt in payload.get("request_receipts") or []:
            for key in ("body", "envelope", "raw_response"):
                pin = receipt.get(key)
                if isinstance(pin, dict) and pin.get("sha256"):
                    blob_digests.add(str(pin["sha256"]))
        membership = payload.get("membership") or {}
        for key in ("body", "envelope"):
            pin = membership.get(key)
            if isinstance(pin, dict) and pin.get("sha256"):
                blob_digests.add(str(pin["sha256"]))

        def note(receipt: Mapping[str, object]) -> None:
            candidate_raw = str((receipt.get("staging") or {}).get("candidate_raw_root") or "")
            if candidate_raw:
                candidate_raw_roots.add(candidate_raw)
                evidence_roots.add(str(Path(candidate_raw) / ROSTERWATCH_SUBDIR / IDENTITY_EVIDENCE_DIRECTORY))
            staged = str(receipt.get("staged_processed_root") or "")
            if staged:
                staged_processed_roots.add(staged)

        processing = cycle_root / PROCESSING_FILE
        if processing.is_file():
            note(_load_path(processing))
        for failed in (cycle_root / ATTEMPTS_SUBDIR).glob("attempt-*/processing-failed.json"):
            note((_load_path(failed).get("partial") or {}))
        for name in (PUBLICATION_FILE, UNCHANGED_FILE):
            path = cycle_root / name
            if path.is_file():
                receipt = _load_path(path)
                if receipt.get("retained_capture_evidence_root"):
                    candidate_raw_roots.add(str(receipt["retained_capture_evidence_root"]))
                if receipt.get("retained_staged_processed_root"):
                    staged_processed_roots.add(str(receipt["retained_staged_processed_root"]))
    return {"cycles": cycles, "store_blob_digests": sorted(blob_digests),
            "candidate_raw_roots": sorted(candidate_raw_roots), "staged_processed_roots": sorted(staged_processed_roots),
            "identity_evidence_roots": sorted(evidence_roots)}


def protected_generation_dirs(integration_root: Path) -> set[Path]:
    """Generation directories that pruning must never remove while a cycle references them."""
    references = referenced_retention(integration_root)
    return {Path(p).resolve(strict=False) for p in references["candidate_raw_roots"] + references["staged_processed_roots"]}


def held_store_digests(integration_root: Path) -> set[str]:
    """Response digests to pass to ``HostedControlStore.retention_eligibility(held_digests=...)``."""
    return set(referenced_retention(integration_root)["store_blob_digests"])
