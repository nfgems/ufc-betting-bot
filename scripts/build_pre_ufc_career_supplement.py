"""
Build supplemental pro and amateur pre-UFC fight history from Sherdog + Tapology.

This script:
1. Loads UFC fight history to identify fighters and their first UFC appearance
2. Scrapes each fighter's full pre-UFC pro history (Sherdog AND Tapology, takes best)
3. Scrapes each fighter's amateur MMA history into a separate supplement
4. Filters both tracks to pre-UFC dates only (before their first UFC appearance)
5. Preserves the event/promotion name as an `organization` field
6. Saves separate CSVs for later feature-pipeline integration

Per-fight stats (sig str, TD, control, etc.) remain NaN because regional sources
do not publish UFCStats-style box scores.

Usage:
    python scripts/build_pre_ufc_career_supplement.py [--max-fighters N] [--resume]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR  # noqa: E402
from src.data.name_utils import (  # noqa: E402
    normalize_cross_source_name,
    normalize_person_name,
    normalize_ufcstats_id,
)
from src.data.io_utils import write_csv_atomically, write_json_atomically  # noqa: E402
from src.data.pre_ufc_scraper import (  # noqa: E402
    OUTPUT_COLUMNS,
    _dedupe_supplement_rows,
    scrape_fighter_amateur_fights,
    scrape_fighter_pre_ufc_fights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
OUTPUT_PATH = RAW_DATA_DIR / "pre_ufc_career_supplement_v2.csv"
CHECKPOINT_PATH = RAW_DATA_DIR / "pre_ufc_career_checkpoint_v2.json"
AMATEUR_OUTPUT_PATH = RAW_DATA_DIR / "amateur_career_supplement.csv"
AMATEUR_CHECKPOINT_PATH = RAW_DATA_DIR / "amateur_career_checkpoint.json"
OFFICIAL_ROSTER_PATH = RAW_DATA_DIR / "ufc_active_roster_official.csv"
DEFAULT_FIGHTS_PATH = PROCESSED_DATA_DIR / "fights_cleaned.csv"
MAX_RETRY_ATTEMPTS = 3
_STABLE_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")
_RETRYABLE_STATUSES = frozenset({"zero_rows", "transient_failure"})
_EXHAUSTED_STATUSES = frozenset({"exhausted_zero_rows", "exhausted_transient_failure"})


def _find_best_features_csv() -> Path:
    """Return the canonical processed chronology, never a model-candidate snapshot."""
    return DEFAULT_FIGHTS_PATH


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _normalize_stable_id(value: object) -> str | None:
    candidate = _clean_text(value).casefold()
    return candidate if _STABLE_ID_PATTERN.fullmatch(candidate) else None


def _stable_id_from_url(value: object) -> str | None:
    return normalize_ufcstats_id(value)


def _normalize_boundary(value: object) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def _parse_aliases(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                raw_values = decoded
            else:
                raw_values = re.split(r"[|;]", text)
        else:
            raw_values = re.split(r"[|;]", text)
    aliases: list[str] = []
    for item in raw_values:
        alias = _clean_text(item)
        if alias and alias not in aliases:
            aliases.append(alias)
    return aliases


def _candidate_fingerprint(candidate: dict[str, Any]) -> str:
    """Fingerprint only the stable identity and strict debut boundary.

    Display-name/alias changes may improve source discovery, but they must not
    reset the bounded retry budget for an otherwise unchanged identity input.
    """
    payload = {
        "fighter_id": _normalize_stable_id(candidate.get("fighter_id")) or "",
        "first_ufc_date": _normalize_boundary(candidate.get("first_ufc_date")) or "",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _candidate_key(candidate: dict[str, Any]) -> str:
    fighter_id = _normalize_stable_id(candidate.get("fighter_id"))
    if fighter_id:
        return f"ufcstats:{fighter_id}"
    return f"name:{normalize_person_name(candidate.get('fighter_name'))}"


def _finalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    aliases = _parse_aliases(candidate.get("aliases"))
    fighter_name = _clean_text(candidate.get("fighter_name"))
    if fighter_name and fighter_name not in aliases:
        aliases.insert(0, fighter_name)
    finalized = {
        **candidate,
        "fighter_id": _normalize_stable_id(candidate.get("fighter_id")),
        "fighter_name": fighter_name,
        "first_ufc_date": _normalize_boundary(candidate.get("first_ufc_date")),
        "aliases": aliases,
    }
    finalized["input_fingerprint"] = _candidate_fingerprint(finalized)
    return finalized


def identify_ufc_fighters(
    fights_path: Path,
    max_ufc_fights: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Find identity-scoped fighters and their first tracked UFC appearance."""
    header = pd.read_csv(fights_path, nrows=0).columns
    columns = ["fighter_a", "fighter_b", "event_date"]
    for column in ("fighter_a_id", "fighter_b_id"):
        if column in header:
            columns.append(column)
    df = pd.read_csv(fights_path, usecols=columns)
    logger.info("Loaded %d fights from %s", len(df), fights_path)

    appearances: list[dict[str, Any]] = []
    ids_by_name: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        for side in ("a", "b"):
            display_name = _clean_text(row.get(f"fighter_{side}"))
            name_key = normalize_person_name(display_name)
            if not display_name or not name_key:
                continue
            fighter_id = _normalize_stable_id(row.get(f"fighter_{side}_id"))
            if fighter_id:
                ids_by_name.setdefault(name_key, set()).add(fighter_id)
            appearances.append(
                {
                    "fighter_name": display_name,
                    "name_key": name_key,
                    "fighter_id": fighter_id,
                    "event_date": _normalize_boundary(row.get("event_date")),
                }
            )

    grouped: dict[str, dict[str, Any]] = {}
    for appearance in appearances:
        fighter_id = appearance["fighter_id"]
        if fighter_id is None:
            possible_ids = ids_by_name.get(appearance["name_key"], set())
            if len(possible_ids) == 1:
                fighter_id = next(iter(possible_ids))
            elif len(possible_ids) > 1:
                continue
        identity_key = f"ufcstats:{fighter_id}" if fighter_id else f"name:{appearance['name_key']}"
        candidate = grouped.setdefault(
            identity_key,
            {
                "fighter_id": fighter_id,
                "fighter_name": appearance["fighter_name"],
                "aliases": set(),
                "ufc_fights": 0,
                "first_ufc_date": None,
                "source_input_fingerprint": "canonical_processed_fights",
            },
        )
        candidate["aliases"].add(appearance["fighter_name"])
        candidate["ufc_fights"] += 1
        event_date = appearance["event_date"]
        if event_date and (
            candidate["first_ufc_date"] is None or event_date < candidate["first_ufc_date"]
        ):
            candidate["first_ufc_date"] = event_date

    candidates: dict[str, dict[str, Any]] = {}
    for key, candidate in grouped.items():
        if max_ufc_fights is not None and candidate["ufc_fights"] > max_ufc_fights:
            continue
        candidate["aliases"] = sorted(candidate["aliases"])
        finalized = _finalize_candidate(candidate)
        if finalized["first_ufc_date"] is None:
            logger.warning("Skipping %s: no parseable first-UFC boundary", finalized["fighter_name"])
            continue
        candidates[key] = finalized

    logger.info(
        "Found %d identity-scoped fighters%s",
        len(candidates),
        f" with <= {max_ufc_fights} UFC fights" if max_ufc_fights is not None else "",
    )
    return dict(sorted(candidates.items()))



# scrape_fighter_pre_ufc_fights / scrape_fighter_amateur_fights are imported from src.data.pre_ufc_scraper


def _dedupe_rows(rows: list[dict], *, dedupe_mirrors: bool = False) -> pd.DataFrame:
    """Deduplicate fight rows, preferring entries that preserve organization metadata."""
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(rows)
    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    if dedupe_mirrors:
        df = _dedupe_supplement_rows(df)

    df["_fighter_a_key"] = df["fighter_a"].map(normalize_person_name)
    df["_fighter_b_key"] = df["fighter_b"].map(normalize_person_name)
    df["_org_present"] = df["organization"].fillna("").astype(str).str.len()
    df = df.sort_values(["_org_present"], ascending=False)
    df = df.drop_duplicates(subset=["event_date", "_fighter_a_key", "_fighter_b_key"], keep="first")
    df = df.drop(columns=["_fighter_a_key", "_fighter_b_key", "_org_present"], errors="ignore")
    return df.reindex(columns=OUTPUT_COLUMNS)


def _save_rows(
    rows: list[dict],
    *,
    output_path: Path | None = None,
    dedupe_mirrors: bool = False,
) -> None:
    resolved_output_path = output_path or OUTPUT_PATH
    output_df = _dedupe_rows(rows, dedupe_mirrors=dedupe_mirrors)
    write_csv_atomically(output_df, resolved_output_path)


def _load_existing_rows(
    *,
    output_path: Path | None = None,
    replace_fighters: set[str] | None = None,
) -> list[dict]:
    """Load current supplement rows, optionally excluding specific fighters."""
    resolved_output_path = output_path or OUTPUT_PATH
    if not resolved_output_path.exists():
        return []

    existing_df = pd.read_csv(resolved_output_path)
    if replace_fighters:
        existing_df = existing_df[~existing_df["fighter_a"].isin(replace_fighters)]
    return existing_df.to_dict("records")


class CandidateFileError(ValueError):
    """The explicit candidate queue cannot be resolved without guessing."""


def _candidate_records_from_json(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        raise CandidateFileError("candidate JSON must contain an object or list")
    if isinstance(payload.get("candidates"), list):
        return [dict(row) for row in payload["candidates"] if isinstance(row, dict)]
    if {"fighter_id", "fighter_name", "first_ufc_date"}.issubset(payload):
        return [dict(payload)]

    records: list[dict[str, Any]] = []
    for fighter_id, value in payload.items():
        if not isinstance(value, dict):
            raise CandidateFileError("candidate JSON mapping values must be objects")
        record = dict(value)
        record.setdefault("fighter_id", fighter_id)
        records.append(record)
    return records


def load_candidate_file(path: Path) -> dict[str, dict[str, Any]]:
    """Load and validate an explicit stable-ID candidate queue from CSV or JSON."""
    if not path.is_file():
        raise CandidateFileError(f"candidate file does not exist: {path}")
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        records = pd.read_csv(path, dtype=object).to_dict("records")
    elif suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CandidateFileError(f"candidate JSON is invalid: {exc}") from exc
        records = _candidate_records_from_json(payload)
    else:
        raise CandidateFileError("candidate file must use .csv or .json")

    candidates: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    ids_by_name: dict[str, set[str]] = {}
    for index, record in enumerate(records, start=1):
        fighter_id = _normalize_stable_id(record.get("fighter_id"))
        fighter_name = _clean_text(record.get("fighter_name"))
        first_ufc_date = _normalize_boundary(record.get("first_ufc_date"))
        if fighter_id is None:
            errors.append(f"row {index}: fighter_id must be exactly 16 hexadecimal characters")
            continue
        if not fighter_name:
            errors.append(f"row {index}: fighter_name is required")
            continue
        if first_ufc_date is None:
            errors.append(f"row {index}: first_ufc_date must be parseable")
            continue

        aliases = _parse_aliases(
            record.get("aliases")
            if _clean_text(record.get("aliases"))
            else record.get("aliases_json")
        )
        ufc_fights_value = pd.to_numeric(record.get("ufc_fights"), errors="coerce")
        candidate = _finalize_candidate(
            {
                "fighter_id": fighter_id,
                "fighter_name": fighter_name,
                "first_ufc_date": first_ufc_date,
                "aliases": aliases,
                "ufc_fights": int(ufc_fights_value) if pd.notna(ufc_fights_value) else 0,
                "source_input_fingerprint": _clean_text(record.get("input_fingerprint")),
            }
        )
        key = _candidate_key(candidate)
        previous = candidates.get(key)
        if previous is not None and previous["input_fingerprint"] != candidate["input_fingerprint"]:
            errors.append(f"row {index}: conflicting entries for stable identity {fighter_id}")
            continue
        candidates[key] = candidate
        for name in candidate["aliases"]:
            name_key = normalize_cross_source_name(name)
            if name_key:
                ids_by_name.setdefault(name_key, set()).add(fighter_id)

    ambiguous = {
        name: sorted(fighter_ids)
        for name, fighter_ids in ids_by_name.items()
        if len(fighter_ids) > 1
    }
    if ambiguous:
        for name, fighter_ids in sorted(ambiguous.items()):
            errors.append(
                f"ambiguous normalized name '{name}' maps to multiple fighter IDs: "
                + ", ".join(fighter_ids)
            )
    if errors:
        raise CandidateFileError("; ".join(errors))
    if not candidates:
        raise CandidateFileError("candidate file contains no valid candidates")
    return dict(sorted(candidates.items()))


def load_checkpoint(path: Path | None = None) -> dict:
    """Load scraping checkpoint (fighters already processed)."""
    checkpoint_path = path or CHECKPOINT_PATH
    if checkpoint_path.exists():
        payload = json.loads(checkpoint_path.read_text())
        if not isinstance(payload, dict):
            payload = {}
    else:
        payload = {}
    if not isinstance(payload.get("processed"), dict):
        payload["processed"] = {}
    if not isinstance(payload.get("failed"), list):
        payload["failed"] = []
    if not isinstance(payload.get("identity_state"), dict):
        payload["identity_state"] = {}
    payload.setdefault("schema_version", 2)
    return payload


def load_official_roster_lookup() -> dict[str, dict[str, object]]:
    """Load explicit-fighter fallback info from the official roster artifact."""
    if not OFFICIAL_ROSTER_PATH.exists():
        return {}

    roster_df = pd.read_csv(OFFICIAL_ROSTER_PATH)
    if roster_df.empty:
        return {}

    lookup: dict[str, dict[str, object]] = {}
    for _, row in roster_df.iterrows():
        first_ufc_date = _normalize_boundary(row.get("octagon_debut"))
        fighter_id = _stable_id_from_url(row.get("ufcstats_url"))
        primary_name = _clean_text(row.get("ufcstats_name")) or _clean_text(
            row.get("official_name")
        )
        info = _finalize_candidate({
            "fighter_id": fighter_id,
            "fighter_name": primary_name,
            "ufc_fights": 0,
            "first_ufc_date": first_ufc_date,
            "aliases": [row.get("official_name"), row.get("ufcstats_name")],
            "source_input_fingerprint": "official_active_roster",
        })
        for field in ("official_name", "ufcstats_name"):
            name = _clean_text(row.get(field))
            key = normalize_cross_source_name(name)
            if key and key not in lookup:
                lookup[key] = info
    return lookup


def save_checkpoint(checkpoint: dict, path: Path | None = None) -> None:
    """Save scraping checkpoint."""
    checkpoint_path = path or CHECKPOINT_PATH
    checkpoint["schema_version"] = 2
    write_json_atomically(checkpoint, checkpoint_path)


def _legacy_processed_count(checkpoint: dict, candidate: dict[str, Any]) -> int | None:
    processed = checkpoint.get("processed", {})
    for name in candidate.get("aliases", []):
        if name not in processed:
            continue
        try:
            return int(processed[name])
        except (TypeError, ValueError):
            return None
    return None


def _prepare_identity_state(
    checkpoint: dict,
    key: str,
    candidate: dict[str, Any],
    *,
    allow_legacy_migration: bool,
) -> tuple[dict[str, Any], bool]:
    """Return current state and whether its input fingerprint forced a reset."""
    states = checkpoint.setdefault("identity_state", {})
    fingerprint = candidate["input_fingerprint"]
    state = states.get(key)
    new_identity = not isinstance(state, dict)
    fingerprint_changed = (
        isinstance(state, dict) and state.get("input_fingerprint") != fingerprint
    )
    if new_identity or fingerprint_changed:
        state = {
            "fighter_id": candidate.get("fighter_id"),
            "fighter_name": candidate["fighter_name"],
            "first_ufc_date": candidate["first_ufc_date"],
            "input_fingerprint": fingerprint,
            "attempts": 0,
            "row_count": None,
            "status": "pending",
            "last_error": "",
            "updated_at": None,
        }
        if allow_legacy_migration and new_identity:
            legacy_count = _legacy_processed_count(checkpoint, candidate)
            if legacy_count is not None and legacy_count > 0:
                state.update(
                    {
                        "row_count": legacy_count,
                        "status": "success",
                        "updated_at": "legacy_checkpoint_migration",
                    }
                )
            elif legacy_count == 0:
                state.update(
                    {
                        "row_count": 0,
                        "status": "zero_rows",
                        "updated_at": "legacy_checkpoint_migration",
                    }
                )
        states[key] = state
    else:
        state.update(
            {
                "fighter_id": candidate.get("fighter_id"),
                "fighter_name": candidate["fighter_name"],
                "first_ufc_date": candidate["first_ufc_date"],
            }
        )
    return state, fingerprint_changed


def _state_should_scrape(state: dict[str, Any]) -> bool:
    status = str(state.get("status") or "pending")
    attempts = int(state.get("attempts") or 0)
    if status == "success" or status in _EXHAUSTED_STATUSES:
        return False
    if status in _RETRYABLE_STATUSES:
        return attempts < MAX_RETRY_ATTEMPTS
    return True


def _record_identity_attempt(
    checkpoint: dict,
    key: str,
    candidate: dict[str, Any],
    *,
    row_count: int | None,
    error: Exception | str | None = None,
) -> dict[str, Any]:
    state, _ = _prepare_identity_state(
        checkpoint,
        key,
        candidate,
        allow_legacy_migration=False,
    )
    attempts = int(state.get("attempts") or 0) + 1
    if error is not None:
        status = "exhausted_transient_failure" if attempts >= MAX_RETRY_ATTEMPTS else "transient_failure"
        normalized_row_count = None
        last_error = str(error)
    elif int(row_count or 0) == 0:
        status = "exhausted_zero_rows" if attempts >= MAX_RETRY_ATTEMPTS else "zero_rows"
        normalized_row_count = 0
        last_error = ""
    else:
        status = "success"
        normalized_row_count = int(row_count)
        last_error = ""
    state.update(
        {
            "fighter_id": candidate.get("fighter_id"),
            "fighter_name": candidate["fighter_name"],
            "first_ufc_date": candidate["first_ufc_date"],
            "input_fingerprint": candidate["input_fingerprint"],
            "attempts": attempts,
            "row_count": normalized_row_count,
            "status": status,
            "last_error": last_error,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    checkpoint.setdefault("identity_state", {})[key] = state
    if error is None:
        checkpoint.setdefault("processed", {})[candidate["fighter_name"]] = int(row_count or 0)
    if error is not None:
        if candidate["fighter_name"] not in checkpoint.setdefault("failed", []):
            checkpoint["failed"].append(candidate["fighter_name"])
    else:
        checkpoint["failed"] = [
            name for name in checkpoint.setdefault("failed", [])
            if name != candidate["fighter_name"]
        ]
    return state


def _incomplete_identity_states(checkpoint: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, state in checkpoint.get("identity_state", {}).items():
        if not isinstance(state, dict) or state.get("status") == "success":
            continue
        rows.append({"identity_key": key, **state})
    return rows


def _finalize_output(
    rows: list[dict],
    *,
    output_path: Path,
    label: str,
    dedupe_mirrors: bool = False,
    allow_fighter_row_reduction: set[str] | None = None,
) -> dict[str, object]:
    """Persist one supplement output with guardrails against accidental data loss."""
    existing_row_count = 0
    existing_fighter_counts: dict[str, int] = {}
    existing_df = pd.DataFrame()
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        existing_row_count = len(existing_df)
        if "fighter_a" in existing_df.columns:
            existing_fighter_counts = existing_df.groupby("fighter_a").size().to_dict()

    saved_row_count = 0
    guard_preserved: list[str] = []
    allowed_reductions: list[str] = []
    new_fighters_added: list[str] = []
    allowed_reduction_names = set(allow_fighter_row_reduction or set())
    allowed_reduction_count = 0

    if rows:
        df = _dedupe_rows(rows, dedupe_mirrors=dedupe_mirrors)

        if existing_fighter_counts and "fighter_a" in df.columns:
            new_fighter_counts = df.groupby("fighter_a").size().to_dict()
            for fighter, old_count in existing_fighter_counts.items():
                new_count = new_fighter_counts.get(fighter, 0)
                if fighter in allowed_reduction_names:
                    if new_count < old_count:
                        allowed_reduction_count += old_count - new_count
                        allowed_reductions.append(
                            f"{fighter} ({old_count} rows replaced by {new_count})"
                        )
                    continue
                if new_count < old_count:
                    old_fighter_rows = existing_df[existing_df["fighter_a"] == fighter]
                    df = df[df["fighter_a"] != fighter]
                    df = pd.concat([df, old_fighter_rows], ignore_index=True)
                    guard_preserved.append(f"{fighter} ({old_count} rows kept, new had {new_count})")

            if guard_preserved:
                df = _dedupe_rows(df.to_dict("records"), dedupe_mirrors=dedupe_mirrors)

        protected_existing_count = max(0, existing_row_count - allowed_reduction_count)
        if protected_existing_count > 0 and len(df) < protected_existing_count * 0.9:
            logger.error(
                "INTEGRITY GUARD (%s): Total row count would drop from %d to %d (>10%% loss). "
                "Aborting save to protect existing data.",
                label,
                existing_row_count,
                len(df),
            )
            raise RuntimeError(f"{label} supplement integrity guard triggered")

        write_csv_atomically(df, output_path)
        saved_row_count = len(df)
        logger.info("Saved %d %s rows to %s", len(df), label, output_path)

        if existing_fighter_counts:
            current_fighters = set(df["fighter_a"].dropna().unique())
            old_fighters = set(existing_fighter_counts.keys())
            new_fighters_added = sorted(current_fighters - old_fighters)
    elif existing_row_count > 0:
        logger.info("No new %s rows scraped; existing data preserved unchanged", label)
    else:
        logger.info("No %s fights found", label)

    return {
        "existing_row_count": existing_row_count,
        "saved_row_count": saved_row_count,
        "guard_preserved": guard_preserved,
        "allowed_reductions": allowed_reductions,
        "new_fighters_added": new_fighters_added,
    }


def _ambiguous_candidate_identity_keys(
    candidates: dict[str, dict[str, Any]],
) -> tuple[set[str], list[str]]:
    identities_by_name: dict[str, set[str]] = {}
    candidate_keys_by_name: dict[str, set[str]] = {}
    for key, candidate in candidates.items():
        stable_id = _normalize_stable_id(candidate.get("fighter_id"))
        if not stable_id:
            continue
        for name in candidate.get("aliases", []):
            normalized = normalize_cross_source_name(name)
            if not normalized:
                continue
            identities_by_name.setdefault(normalized, set()).add(stable_id)
            candidate_keys_by_name.setdefault(normalized, set()).add(key)
    ambiguous_names = sorted(
        name for name, fighter_ids in identities_by_name.items() if len(fighter_ids) > 1
    )
    rejected_keys = {
        key for name in ambiguous_names for key in candidate_keys_by_name.get(name, set())
    }
    return rejected_keys, ambiguous_names


def _write_summary_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    write_json_atomically(payload, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build pro and amateur pre-UFC career supplements from Sherdog and Tapology"
    )
    parser.add_argument("--max-fighters", type=int, default=None,
                        help="Limit number of fighters to scrape (for testing)")
    parser.add_argument("--max-ufc-fights", type=int, default=None,
                        help="Optional cap: scrape fighters with <= N UFC fights")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just identify candidates, don't scrape")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--retry-zero-rows", action="store_true",
                        help="Re-scrape fighters previously checkpointed with 0 rows")
    parser.add_argument("--fighters", nargs="+", default=None,
                        help="Explicit fighter names to scrape (comma-separated or repeated)")
    parser.add_argument("--candidate-file", type=Path, default=None,
                        help="Explicit stable-ID candidate queue (.csv or .json)")
    parser.add_argument("--summary-json", type=Path, default=None,
                        help="Write a machine-readable orchestration summary")
    args = parser.parse_args(argv)

    if args.candidate_file is not None and args.fighters:
        parser.error("--candidate-file and --fighters cannot be combined")

    fights_path = _find_best_features_csv()
    candidate_source = str(fights_path)
    try:
        if args.candidate_file is not None:
            candidates = load_candidate_file(args.candidate_file)
            candidate_source = str(args.candidate_file)
        else:
            logger.info("Using canonical training data from: %s", fights_path)
            candidates = identify_ufc_fighters(
                fights_path,
                max_ufc_fights=args.max_ufc_fights,
            )
    except (CandidateFileError, OSError, ValueError) as exc:
        logger.error("Candidate queue rejected: %s", exc)
        _write_summary_json(
            args.summary_json,
            {
                "status": "invalid_candidates",
                "candidate_source": candidate_source,
                "error": str(exc),
            },
        )
        return 2

    rejected_keys, ambiguous_names = _ambiguous_candidate_identity_keys(candidates)
    if ambiguous_names:
        logger.error(
            "Rejecting %d candidates because normalized names map to multiple stable IDs: %s",
            len(rejected_keys),
            ", ".join(ambiguous_names),
        )
        candidates = {
            key: info for key, info in candidates.items() if key not in rejected_keys
        }

    if args.dry_run:
        print(f"\nWould scrape {len(candidates)} fighters. Top 20:")
        for _key, info in sorted(candidates.items())[:20]:
            print(
                f"  {info['fighter_name']} ({info.get('fighter_id') or 'name-only'}): "
                f"{info['ufc_fights']} UFC fights, first: {info['first_ufc_date']}"
            )
        _write_summary_json(
            args.summary_json,
            {
                "status": "dry_run",
                "candidate_source": candidate_source,
                "candidate_count": len(candidates),
                "ambiguous_names_rejected": ambiguous_names,
            },
        )
        return 0

    def _new_checkpoint() -> dict[str, Any]:
        return {"schema_version": 2, "processed": {}, "failed": [], "identity_state": {}}

    checkpoint = load_checkpoint(CHECKPOINT_PATH) if args.resume else _new_checkpoint()
    amateur_checkpoint = (
        load_checkpoint(AMATEUR_CHECKPOINT_PATH)
        if args.resume
        else _new_checkpoint()
    )

    explicit_fighters: list[str] = []
    if args.fighters:
        for raw_value in args.fighters:
            explicit_fighters.extend(
                value.strip() for value in str(raw_value).split(",") if value.strip()
            )

    explicit_queue = args.candidate_file is not None
    if explicit_fighters:
        roster_lookup = load_official_roster_lookup()
        resolved: dict[str, dict[str, Any]] = {}
        aliases_to_keys: dict[str, list[str]] = {}
        for key, candidate in candidates.items():
            for alias in candidate.get("aliases", []):
                aliases_to_keys.setdefault(normalize_cross_source_name(alias), []).append(key)
        for name in dict.fromkeys(explicit_fighters):
            matches = aliases_to_keys.get(normalize_cross_source_name(name), [])
            if len(matches) == 1:
                key = matches[0]
                resolved[key] = candidates[key]
                continue
            if len(matches) > 1:
                logger.error("Explicit fighter '%s' is ambiguous across stable identities", name)
                continue
            roster_info = roster_lookup.get(normalize_cross_source_name(name))
            if (
                roster_info is not None
                and roster_info.get("fighter_id")
                and roster_info.get("first_ufc_date")
            ):
                resolved[_candidate_key(roster_info)] = roster_info
            else:
                logger.error(
                    "Explicit fighter '%s' has no unambiguous stable ID and debut boundary",
                    name,
                )
        candidates = resolved
        explicit_queue = True

    scrape_decisions: dict[str, tuple[bool, bool]] = {}
    to_scrape: dict[str, dict[str, Any]] = {}
    reset_count = 0
    for key, candidate in candidates.items():
        allow_legacy_migration = bool(args.resume and not explicit_queue)
        pro_state, pro_reset = _prepare_identity_state(
            checkpoint,
            key,
            candidate,
            allow_legacy_migration=allow_legacy_migration,
        )
        amateur_state, amateur_reset = _prepare_identity_state(
            amateur_checkpoint,
            key,
            candidate,
            allow_legacy_migration=allow_legacy_migration,
        )
        reset_count += int(pro_reset) + int(amateur_reset)
        pro_should_scrape = _state_should_scrape(pro_state)
        amateur_should_scrape = _state_should_scrape(amateur_state)
        if args.retry_zero_rows:
            pro_should_scrape = pro_should_scrape or (
                str(pro_state.get("status")) == "zero_rows"
                and int(pro_state.get("attempts") or 0) < MAX_RETRY_ATTEMPTS
            )
            amateur_should_scrape = amateur_should_scrape or (
                str(amateur_state.get("status")) == "zero_rows"
                and int(amateur_state.get("attempts") or 0) < MAX_RETRY_ATTEMPTS
            )
        if pro_should_scrape or amateur_should_scrape:
            to_scrape[key] = candidate
            scrape_decisions[key] = (pro_should_scrape, amateur_should_scrape)

    if args.resume:
        logger.info(
            "Resuming: pro=%d legacy processed, amateur=%d legacy processed, %d queued",
            len(checkpoint["processed"]),
            len(amateur_checkpoint["processed"]),
            len(to_scrape),
        )
    if explicit_fighters:
        logger.info(
            "Explicit fighter override: %d requested, %d resolved via candidate/official roster set",
            len(explicit_fighters),
            len(candidates),
        )
    if args.retry_zero_rows:
        logger.info("Retrying %d bounded zero-row candidates", len(to_scrape))

    if args.max_fighters:
        to_scrape = dict(list(to_scrape.items())[:args.max_fighters])
        scrape_decisions = {key: scrape_decisions[key] for key in to_scrape}

    logger.info("Scraping %d fighters for pro and amateur history...", len(to_scrape))

    pro_rows: list[dict] = []
    amateur_rows: list[dict] = []
    if OUTPUT_PATH.exists():
        pro_rows = _load_existing_rows(output_path=OUTPUT_PATH)
        logger.info("Loaded %d existing pre-UFC fight rows", len(pro_rows))
    if AMATEUR_OUTPUT_PATH.exists():
        amateur_rows = _load_existing_rows(output_path=AMATEUR_OUTPUT_PATH)
        logger.info("Loaded %d existing amateur fight rows", len(amateur_rows))

    scraped_count = 0
    found_count = 0
    amateur_scraped_count = 0
    amateur_found_count = 0

    pro_zero_count = 0
    amateur_zero_count = 0
    pro_transient_count = 0
    amateur_transient_count = 0
    pro_successful_replacements: set[str] = set()
    amateur_successful_replacements: set[str] = set()

    # A checkpoint is durable only after its corresponding output has been
    # persisted. Snapshots let a later integrity failure roll back only the
    # current, uncommitted batch.
    committed_checkpoint = copy.deepcopy(checkpoint)
    committed_amateur_checkpoint = copy.deepcopy(amateur_checkpoint)

    def _commit_current_outputs() -> tuple[dict[str, object] | None, dict[str, object] | None, str | None]:
        nonlocal checkpoint
        nonlocal amateur_checkpoint
        nonlocal committed_checkpoint
        nonlocal committed_amateur_checkpoint

        try:
            pro_result = _finalize_output(
                pro_rows,
                output_path=OUTPUT_PATH,
                label="pre-UFC",
                allow_fighter_row_reduction=pro_successful_replacements,
            )
        except RuntimeError:
            checkpoint = copy.deepcopy(committed_checkpoint)
            amateur_checkpoint = copy.deepcopy(committed_amateur_checkpoint)
            save_checkpoint(checkpoint, CHECKPOINT_PATH)
            save_checkpoint(amateur_checkpoint, AMATEUR_CHECKPOINT_PATH)
            return None, None, "pre-UFC"
        save_checkpoint(checkpoint, CHECKPOINT_PATH)
        committed_checkpoint = copy.deepcopy(checkpoint)

        try:
            amateur_result = _finalize_output(
                amateur_rows,
                output_path=AMATEUR_OUTPUT_PATH,
                label="amateur",
                dedupe_mirrors=True,
                allow_fighter_row_reduction=amateur_successful_replacements,
            )
        except RuntimeError:
            amateur_checkpoint = copy.deepcopy(committed_amateur_checkpoint)
            save_checkpoint(amateur_checkpoint, AMATEUR_CHECKPOINT_PATH)
            return pro_result, None, "amateur"
        save_checkpoint(amateur_checkpoint, AMATEUR_CHECKPOINT_PATH)
        committed_amateur_checkpoint = copy.deepcopy(amateur_checkpoint)
        return pro_result, amateur_result, None

    def _return_integrity_guard_failure(failed_track: str) -> int:
        _write_summary_json(
            args.summary_json,
            {
                "status": "integrity_guard_failed",
                "candidate_source": candidate_source,
                "failed_track": failed_track,
                "checkpoint_success_rolled_back": True,
            },
        )
        return 1

    for i, (identity_key, info) in enumerate(to_scrape.items()):
        name = info["fighter_name"]
        if (i + 1) % 50 == 0:
            logger.info("Progress: %d/%d fighters processed", i + 1, len(to_scrape))
            _pro_progress, _amateur_progress, failed_track = _commit_current_outputs()
            if failed_track is not None:
                return _return_integrity_guard_failure(failed_track)

        pro_should_scrape, amateur_should_scrape = scrape_decisions[identity_key]

        if pro_should_scrape:
            try:
                rows = scrape_fighter_pre_ufc_fights(name, info["first_ufc_date"])
                scraped_count += 1
                if rows:
                    pro_rows = [
                        row for row in pro_rows if row.get("fighter_a") != name
                    ]
                    pro_rows.extend(rows)
                    pro_successful_replacements.add(name)
                    found_count += 1
                    logger.info("  %s: %d pre-UFC fights found", name, len(rows))
                else:
                    pro_zero_count += 1
                    logger.warning("  %s: zero pre-UFC rows; bounded retry remains missing", name)
                _record_identity_attempt(
                    checkpoint,
                    identity_key,
                    info,
                    row_count=len(rows),
                )
            except Exception as e:
                pro_transient_count += 1
                logger.warning("  %s: pre-UFC transient scrape error: %s", name, e)
                _record_identity_attempt(
                    checkpoint,
                    identity_key,
                    info,
                    row_count=None,
                    error=e,
                )

        if amateur_should_scrape:
            try:
                rows = scrape_fighter_amateur_fights(name, info["first_ufc_date"])
                amateur_scraped_count += 1
                if rows:
                    amateur_rows = [
                        row for row in amateur_rows if row.get("fighter_a") != name
                    ]
                    amateur_rows.extend(rows)
                    amateur_successful_replacements.add(name)
                    amateur_found_count += 1
                    logger.info("  %s: %d amateur fights found", name, len(rows))
                else:
                    amateur_zero_count += 1
                    logger.warning("  %s: zero amateur rows; bounded retry remains missing", name)
                _record_identity_attempt(
                    amateur_checkpoint,
                    identity_key,
                    info,
                    row_count=len(rows),
                )
            except Exception as e:
                amateur_transient_count += 1
                logger.warning("  %s: amateur transient scrape error: %s", name, e)
                _record_identity_attempt(
                    amateur_checkpoint,
                    identity_key,
                    info,
                    row_count=None,
                    error=e,
                )

        # _get_soup() already enforces per-request pacing, so no extra
        # per-fighter delay is needed here.
        time.sleep(0)

    pro_summary, amateur_summary, failed_track = _commit_current_outputs()
    if failed_track is not None:
        return _return_integrity_guard_failure(failed_track)
    assert pro_summary is not None
    assert amateur_summary is not None

    logger.info("\n" + "=" * 60)
    logger.info("SCRAPE SUMMARY")
    logger.info("=" * 60)
    logger.info("  Pro fighters scraped this run:      %d", scraped_count)
    logger.info("  Pro fighters with rows found:       %d", found_count)
    logger.info(
        "  Pro rows before/after/net:          %d / %d / %+d",
        pro_summary["existing_row_count"],
        pro_summary["saved_row_count"],
        int(pro_summary["saved_row_count"]) - int(pro_summary["existing_row_count"]),
    )
    logger.info("  Amateur fighters scraped this run:  %d", amateur_scraped_count)
    logger.info("  Amateur fighters with rows found:   %d", amateur_found_count)
    logger.info(
        "  Amateur rows before/after/net:      %d / %d / %+d",
        amateur_summary["existing_row_count"],
        amateur_summary["saved_row_count"],
        int(amateur_summary["saved_row_count"]) - int(amateur_summary["existing_row_count"]),
    )
    if pro_summary["new_fighters_added"]:
        logger.info("  New pro fighters added (%d):", len(pro_summary["new_fighters_added"]))
        for name in pro_summary["new_fighters_added"][:20]:
            logger.info(f"    + {name}")
        if len(pro_summary["new_fighters_added"]) > 20:
            logger.info("    ... and %d more", len(pro_summary["new_fighters_added"]) - 20)
    if amateur_summary["new_fighters_added"]:
        logger.info("  New amateur fighters added (%d):", len(amateur_summary["new_fighters_added"]))
        for name in amateur_summary["new_fighters_added"][:20]:
            logger.info(f"    + {name}")
        if len(amateur_summary["new_fighters_added"]) > 20:
            logger.info("    ... and %d more", len(amateur_summary["new_fighters_added"]) - 20)
    if pro_summary["guard_preserved"]:
        logger.info("  Pro guard preserved existing data (%d):", len(pro_summary["guard_preserved"]))
        for msg in pro_summary["guard_preserved"]:
            logger.info(f"    ! {msg}")
    if amateur_summary["guard_preserved"]:
        logger.info(
            "  Amateur guard preserved existing data (%d):",
            len(amateur_summary["guard_preserved"]),
        )
        for msg in amateur_summary["guard_preserved"]:
            logger.info(f"    ! {msg}")
    logger.info("  Pro failed:                        %d", len(checkpoint["failed"]))
    if checkpoint["failed"]:
        for name in checkpoint["failed"][-10:]:
            logger.info(f"    x {name}")
    logger.info("  Amateur failed:                    %d", len(amateur_checkpoint["failed"]))
    if amateur_checkpoint["failed"]:
        for name in amateur_checkpoint["failed"][-10:]:
            logger.info(f"    x {name}")
    current_keys = set(candidates)
    pro_incomplete = [
        row for row in _incomplete_identity_states(checkpoint)
        if row["identity_key"] in current_keys
    ]
    amateur_incomplete = [
        row for row in _incomplete_identity_states(amateur_checkpoint)
        if row["identity_key"] in current_keys
    ]
    incomplete_count = len(pro_incomplete) + len(amateur_incomplete)
    warning_codes: list[str] = []
    if incomplete_count:
        warning_codes.append("pre_ufc_identity_retry_incomplete")
        logger.warning(
            "Pre-UFC collection remains incomplete for %d identity/track entries; missing values stay missing",
            incomplete_count,
        )
    if ambiguous_names:
        warning_codes.append("pre_ufc_ambiguous_identity_rejected")
    if not candidates:
        warning_codes.append("pre_ufc_candidate_queue_empty")
    status = "incomplete" if warning_codes else "complete"
    logger.info("=" * 60)

    _write_summary_json(
        args.summary_json,
        {
            "status": status,
            "warning_codes": warning_codes,
            "candidate_source": candidate_source,
            "candidate_count": len(candidates),
            "queued_count": len(to_scrape),
            "ambiguous_names_rejected": ambiguous_names,
            "checkpoint_fingerprint_resets": reset_count,
            "max_retry_attempts": MAX_RETRY_ATTEMPTS,
            "pro": {
                "attempted": scraped_count,
                "with_rows": found_count,
                "zero_rows": pro_zero_count,
                "transient_failures": pro_transient_count,
                "incomplete": pro_incomplete,
                **pro_summary,
            },
            "amateur": {
                "attempted": amateur_scraped_count,
                "with_rows": amateur_found_count,
                "zero_rows": amateur_zero_count,
                "transient_failures": amateur_transient_count,
                "incomplete": amateur_incomplete,
                **amateur_summary,
            },
        },
    )
    return 2 if status == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
