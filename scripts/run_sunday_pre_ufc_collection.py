"""Run the narrow Sunday current-card and pre-UFC continuity contract.

The command discovers the current card, persists raw evidence and lifecycle
state, publishes a stable-ID candidate queue, and invokes the bounded pre-UFC
collector.  It never trains, promotes, deploys, or runs method/post-event work.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import socket
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence
from urllib.parse import urlparse

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.io_utils import write_json_atomically
from src.data.name_utils import normalize_ufcstats_id
from src.data.practical_forward_collection import (
    DEFAULT_FORWARD_COLLECTION_ROOT,
    append_jsonl_unique,
    collection_health,
    enrich_event_cards_with_schedule,
    persist_upcoming_discovery,
    publish_pre_ufc_candidate_queue,
    stable_json_hash,
    utc_text,
)

logger = logging.getLogger(__name__)

CANONICAL_FIGHTS_PATH = PROCESSED_DATA_DIR / "fights_cleaned.csv"
OFFICIAL_ROSTER_PATH = RAW_DATA_DIR / "ufc_active_roster_official.csv"
DEFAULT_SUMMARY_PATH = DATA_DIR / "tmp" / "sunday_pre_ufc_collection_latest.json"
LOCK_STALE_AFTER = timedelta(hours=12)


def _utc(value: object = None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _pid_is_running(value: object) -> bool:
    try:
        pid = int(value)
        if pid <= 0:
            return False
        os.kill(pid, 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


@contextmanager
def _single_writer(storage_root: Path):
    """Serialize all mutable forward state and recover only abandoned locks."""
    root = Path(storage_root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".coordinator.lock"
    token = secrets.token_hex(16)
    hostname = socket.gethostname()
    descriptor: int | None = None

    for attempt in range(2):
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as exc:
            stale = False
            try:
                payload = json.loads(lock_path.read_text(encoding="utf-8"))
                started = _utc(payload.get("started_at_utc"))
                same_host_live = (
                    str(payload.get("hostname") or "") == hostname
                    and _pid_is_running(payload.get("pid"))
                )
                stale = _utc() - started > LOCK_STALE_AFTER and not same_host_live
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                try:
                    modified = datetime.fromtimestamp(lock_path.stat().st_mtime, tz=timezone.utc)
                    stale = _utc() - modified > LOCK_STALE_AFTER
                except OSError:
                    stale = False
            if attempt == 0 and stale:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            raise RuntimeError("sunday_pre_ufc_collection_already_running") from exc

    if descriptor is None:
        raise RuntimeError("sunday_pre_ufc_collection_lock_unavailable")
    lock_payload = {
        "token": token,
        "pid": os.getpid(),
        "hostname": hostname,
        "started_at_utc": utc_text(),
    }
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(lock_payload, sort_keys=True))
        yield
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
        if str(current.get("token") or "") == token:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _normalized_resource(value: object) -> str:
    parsed = urlparse(str(value or "").strip())
    hostname = str(parsed.hostname or "").casefold().removeprefix("www.")
    return f"{hostname}{parsed.path.rstrip('/').casefold()}"


def build_stable_id_resolver(
    roster_path: Path = OFFICIAL_ROSTER_PATH,
) -> Callable[[str, str], object]:
    """Resolve only reviewed roster athlete-URL bindings, without name guessing."""
    by_athlete_url: dict[str, str] = {}
    if Path(roster_path).exists():
        try:
            roster = pd.read_csv(roster_path, dtype=object)
            for row in roster.to_dict("records"):
                athlete_key = _normalized_resource(
                    row.get("official_athlete_url") or row.get("athlete_url")
                )
                fighter_id = normalize_ufcstats_id(row.get("ufcstats_url")) or ""
                if athlete_key and fighter_id:
                    by_athlete_url[athlete_key] = fighter_id
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            logger.warning("Could not load roster identity bindings: %s", exc)

    def resolve(_name: str, athlete_url: str) -> object:
        return by_athlete_url.get(_normalized_resource(athlete_url))

    return resolve


def _default_discovery_provider(*, force_refresh: bool) -> dict[str, object]:
    from src.data.live_monitor import collect_upcoming_event_cards_snapshot

    return collect_upcoming_event_cards_snapshot(force_refresh=force_refresh)


def _default_context_enricher(rows: list[dict]) -> list[dict]:
    from src.data.live_monitor import enrich_upcoming_fight_contexts

    return enrich_upcoming_fight_contexts(rows)


def _default_pre_ufc_runner(queue_path: Path, summary_path: Path) -> dict[str, object]:
    from scripts.build_pre_ufc_career_supplement import main as pre_ufc_main

    exit_code = pre_ufc_main(
        [
            "--resume",
            "--retry-zero-rows",
            "--candidate-file",
            str(queue_path),
            "--summary-json",
            str(summary_path),
        ]
    )
    payload: dict[str, object] = {}
    if summary_path.exists():
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload = loaded
    payload["exit_code"] = int(exit_code)
    status = str(payload.get("status") or "failed")
    retryable = int(exit_code) == 2 and status == "incomplete"
    if exit_code and not retryable and status not in {
        "failed",
        "invalid_candidates",
        "integrity_guard_failed",
    }:
        payload["status"] = "failed"
        status = "failed"
    if status in {"complete", "incomplete"} and int(exit_code) in {0, 2}:
        from src.data.fighter_lookup import clear_cache

        clear_cache(preserve_environment_blocks=True)
    return payload


def _stage(
    name: str,
    payload: Mapping[str, object],
    *,
    status: str | None = None,
    warnings: Sequence[str] = (),
) -> dict[str, object]:
    report = dict(payload)
    report["stage"] = name
    report["status"] = str(status or report.get("status") or "healthy")
    report["warning_codes"] = list(
        dict.fromkeys(
            [
                *(str(value) for value in report.get("warning_codes", []) or []),
                *(str(value) for value in warnings),
            ]
        )
    )
    return report


def _failed_stage(name: str, exc: Exception) -> dict[str, object]:
    return _stage(
        name,
        {"error_type": type(exc).__name__, "error": str(exc)},
        status="failed",
        warnings=[f"{name}_failed"],
    )


def _record_daily_receipt(
    root: Path,
    report: Mapping[str, object],
    *,
    observed_at_utc: str,
) -> None:
    receipt = {
        "stage": "daily_collection",
        "scheduled_slot": _utc(observed_at_utc).date().isoformat(),
        "observed_at_utc": observed_at_utc,
        "status": str(report.get("status") or "unknown"),
        "warning_codes": list(report.get("warning_codes") or []),
    }
    receipt["receipt_id"] = stable_json_hash(receipt)
    append_jsonl_unique(root / "stage-attempt-ledger.jsonl", [receipt], id_field="receipt_id")


def _run_unlocked(
    *,
    mode: str,
    execution_contract: str,
    storage_root: Path,
    now: object = None,
    completed_fights_path: Path = CANONICAL_FIGHTS_PATH,
    roster_path: Path = OFFICIAL_ROSTER_PATH,
    discovery_provider: Callable[..., Mapping[str, object]] | None = None,
    context_enricher: Callable[[list[dict]], list[dict]] | None = None,
    fighter_id_resolver: Callable[[str, str], object] | None = None,
    pre_ufc_runner: Callable[[Path, Path], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    if mode != "daily":
        raise ValueError(f"unsupported_sunday_collection_mode:{mode}")
    if execution_contract != "sunday_pre_ufc":
        raise ValueError(f"unsupported_sunday_execution_contract:{execution_contract}")

    root = Path(storage_root)
    started = utc_text(now)
    stages: list[dict[str, object]] = []
    discovery: dict[str, object] | None = None

    try:
        provider = discovery_provider or _default_discovery_provider
        snapshot = dict(provider(force_refresh=True))
        cards, schedule_issues = enrich_event_cards_with_schedule(
            list(snapshot.get("event_cards") or []),
            context_enricher=context_enricher or _default_context_enricher,
        )
        discovery = persist_upcoming_discovery(
            event_cards=cards,
            scan_complete=bool(snapshot.get("scan_complete")),
            storage_root=root,
            observed_at=snapshot.get("collected_at_utc") or started,
            fighter_id_resolver=fighter_id_resolver or build_stable_id_resolver(roster_path),
            source_responses=list(snapshot.get("source_responses") or []),
            require_raw_evidence=True,
        )
        schedule_warnings = sorted(
            {str(issue.get("code") or "scheduled_start_issue") for issue in schedule_issues}
        )
        stages.append(
            _stage(
                "discovery",
                discovery,
                status="complete"
                if discovery.get("status") == "healthy" and not schedule_warnings
                else "incomplete",
                warnings=schedule_warnings,
            )
        )
    except Exception as exc:
        stages.append(_failed_stage("discovery", exc))

    candidates: dict[str, object] | None = None
    if discovery is not None:
        try:
            candidates = publish_pre_ufc_candidate_queue(
                list(discovery.get("targets") or []),
                changes=list(discovery.get("changes") or []),
                completed_fights_path=Path(completed_fights_path),
                roster_path=Path(roster_path),
                storage_root=root,
                observed_at=started,
            )
            stages.append(
                _stage(
                    "pre_ufc_candidates",
                    candidates,
                    status="incomplete" if candidates.get("warning_codes") else "complete",
                )
            )
        except Exception as exc:
            stages.append(_failed_stage("pre_ufc_candidates", exc))

    if candidates is not None:
        try:
            summary_path = root / "latest-pre-ufc-run-summary.json"
            payload = dict(
                (pre_ufc_runner or _default_pre_ufc_runner)(
                    Path(str(candidates["queue_path"])), summary_path
                )
            )
            status = str(payload.get("status") or "failed")
            exit_code = int(payload.get("exit_code", 0) or 0)
            retryable = exit_code == 2 and status == "incomplete"
            failed = (
                (exit_code != 0 and not retryable)
                or status in {"failed", "invalid_candidates", "integrity_guard_failed"}
            )
            stages.append(
                _stage(
                    "pre_ufc",
                    payload,
                    status="failed" if failed else "complete" if status == "complete" else "incomplete",
                    warnings=[] if status == "complete" and not failed else ["pre_ufc_failed" if failed else "pre_ufc_incomplete"],
                )
            )
        except Exception as exc:
            stages.append(_failed_stage("pre_ufc", exc))

    health = collection_health(stages)
    daily = _stage(
        "daily_collection",
        {},
        status="complete" if health["health_state"] == "healthy" else health["health_state"],
        warnings=health["warning_codes"],
    )
    _record_daily_receipt(root, daily, observed_at_utc=started)
    run = {
        "schema_version": 1,
        "run_id": stable_json_hash(
            {
                "mode": mode,
                "execution_contract": execution_contract,
                "started_at_utc": started,
                "stage_statuses": [[row.get("stage"), row.get("status")] for row in stages],
            }
        ),
        "mode": mode,
        "execution_contract": execution_contract,
        "generated_at_utc": started,
        "status": health["health_state"],
        **health,
        "stages": stages,
        "execution_contract_selected": True,
        "model_training_or_promotion_performed": False,
        "method_or_post_event_collection_performed": False,
    }
    write_json_atomically(run, root / "latest-run-summary.json")
    append_jsonl_unique(root / "run-ledger.jsonl", [run], id_field="run_id")
    return run


def run_collection_cycle(
    *,
    mode: str,
    execution_contract: str,
    storage_root: Path = DEFAULT_FORWARD_COLLECTION_ROOT,
    **kwargs,
) -> dict[str, object]:
    with _single_writer(Path(storage_root)):
        return _run_unlocked(
            mode=mode,
            execution_contract=execution_contract,
            storage_root=Path(storage_root),
            **kwargs,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("daily",), required=True)
    parser.add_argument(
        "--execution-contract",
        choices=("sunday_pre_ufc",),
        required=True,
    )
    parser.add_argument("--now", help="UTC ISO override for deterministic local tests")
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_FORWARD_COLLECTION_ROOT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_collection_cycle(
            mode=args.mode,
            execution_contract=args.execution_contract,
            now=args.now,
            storage_root=args.storage_root,
        )
    except Exception as exc:
        try:
            generated_at = utc_text(args.now)
        except (TypeError, ValueError):
            generated_at = utc_text()
        summary = {
            "schema_version": 1,
            "mode": args.mode,
            "execution_contract": args.execution_contract,
            "generated_at_utc": generated_at,
            "status": "failed",
            "health_state": "failed",
            "failed_stages": ["coordinator"],
            "incomplete_stages": [],
            "warning_codes": ["coordinator_failed"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "execution_contract_selected": True,
            "model_training_or_promotion_performed": False,
            "method_or_post_event_collection_performed": False,
        }
    write_json_atomically(summary, args.summary_json)
    print(json.dumps(summary, indent=2))
    if summary["status"] == "failed":
        return 1
    if summary["status"] != "healthy":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
