"""Assemble and validate one immutable local production-candidate bundle.

This script is intentionally unable to publish into the canonical ``models`` or
``data/processed`` directories.  It copies a fully trained candidate into a new
repository-local staging root, records all identities needed for review and
rollback, and invokes the strict staged production-bundle validator before it
reports success.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.io_utils import copy_file_atomically, write_json_atomically
from src.model.production_bundle import (
    ProductionBundleError,
    _expected_no_odds_spec_payload,
    load_production_bundle,
    validate_production_bundle,
)
from src.model.training_spec import resolve_named_training_spec

import scripts.build_model_input_inventory as model_input_inventory


MODEL_FILENAMES = {
    "primary": "xgboost_model.pkl",
    "no_odds": "xgboost_no_odds_model.pkl",
    "logistic": "logistic_model.pkl",
}
PROCESSED_FILENAMES = (
    "fights_cleaned.csv",
    "features.csv",
    "test_set.csv",
    "test_set.csv.metadata.json",
)
FIT_ONLY_SPEC_FIELDS = frozenset({"git_hash", "trained_at"})
FULLFIT_ALLOWED_DIFFERENCES = frozenset(
    {"name", "description", "train_cutoff_date"}
)
MAX_EVIDENCE_FILE_BYTES = 10 * 1024 * 1024
MAX_EVIDENCE_TOTAL_BYTES = 50 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
RUNTIME_HASH_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
APPROVED_EVALUATION_SPEC = "full_live_contract_v6_durability"
APPROVED_FULLFIT_SPEC = (
    "full_live_contract_v6_durability_corrected_20260805_fullfit"
)
APPROVED_EVALUATION_PAYLOAD_SHA256 = (
    "68f2fd6d851224ab395fe469b17a9974d87b8b48d812e2108636d6b889352f45"
)
APPROVED_FIGHTS_SHA256 = (
    "f863f99406a78afe7c8869650176f42eef94c626f1684ed6e89e22abbbcc9fea"
)
APPROVED_FEATURES_SHA256 = (
    "7949168f55996d9510023e928b319beffecb57eaf90beb94ef9983235b18872b"
)
APPROVED_TRAIN_FIGHTS_SHA256 = (
    "77a9071d8991a2458644fcf8a3b41b681d9da5f266b47deff4dd8614ef8e6f75"
)
APPROVED_TRAIN_FEATURES_SHA256 = (
    "5c6b4cb328e4e7f66e13305a381614ca4023d545df560bb4fc1d7a84e6423183"
)
SEMANTIC_EQUIVALENCE_ATOL = 1e-12
ASSEMBLY_INVENTORY_ALLOWED_CHANGED_PATHS = frozenset(
    {
        "scripts/build_staged_production_bundle.py",
        "scripts/parity_replay.py",
    }
)
APPROVED_BFO_LEDGER_NAME = (
    "bfo_revalidation_20260805_head_source.provenance.jsonl"
)
APPROVED_BFO_LEDGER_SHA256 = (
    "2bc0e3797b427607498e1714f6e890a13c90c24063c4606f0d631deebcc4a217"
)
APPROVED_BFO_CSVS = {
    "historical_odds_bfo_recovered_20260319.csv": (
        45,
        "a28ca0bc38e41a64a85b3d4f2f0cd06566e2ea2b2972acd9f44f5c67655389ed",
    ),
    "historical_odds_bfo_recovered_20260529_fullfit_gap.csv": (
        90,
        "07affdf9caafe32d8ae38a3c0cedfffaf6090eb58501b07841066d8ba6e1a9a3",
    ),
    "historical_odds_bfo_recovered_20260711_guard_gap.csv": (
        53,
        "8498ccd2a21f0e3d1a16c4c457680e956d23ff9333fd987ca83de10ed9f7c83e",
    ),
    "historical_odds_bfo_recovered_auto_20260722_run29887204421_1.csv": (
        22,
        "334d380ba180c50277a3a745045a480c3752a279e7971c65b98c2d342b5fb80b",
    ),
    "historical_odds_bfo_recovered_auto_20260728_run30341844205_1.csv": (
        11,
        "3823b6415d3df72c122387287a2947237c7a09f0a283ff32ff5495c3c7834ed2",
    ),
    "historical_odds_bfo_recovered_auto_20260804_run30891790168_1.csv": (
        13,
        "17a135cff444358296974ca74581be04263503e35bb0998ef3453b70e4856b58",
    ),
}
SAFE_EVIDENCE_SUFFIXES = frozenset({".csv", ".json", ".md", ".txt"})
SENSITIVE_EVIDENCE_TOKENS = (
    ".env",
    "credential",
    "secret",
    "private_key",
    "private-key",
    "account_ledger",
    "account-ledger",
    "bet_ledger",
    "bet-ledger",
)


class StagingBundleError(RuntimeError):
    """Raised when a candidate cannot be staged without weakening a gate."""


@dataclass(frozen=True)
class BundleInputs:
    staging_root: Path
    candidate_models_dir: Path
    candidate_processed_dir: Path
    evaluation_spec_name: str
    input_inventory_path: Path
    assembly_inventory_path: Path
    bfo_provenance_path: Path
    selection_evidence_paths: tuple[Path, ...]
    previous_manifest_path: Path
    previous_readyz_path: Path
    previous_deployed_git_sha: str
    previous_runtime_lookup_hashes: dict[str, str]
    expected_fights_sha256: str
    expected_features_sha256: str
    training_argv: tuple[str, ...]
    bundle_id: str | None = None
    inference_sample_rows: int = 32


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _file_identity(path: Path) -> dict[str, object]:
    return {"sha256": _sha256_file(path), "bytes": int(path.stat().st_size)}


def _aggregate_identities(rows: Iterable[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: str(value["path"])):
        digest.update(str(row["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).lower().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _repo_path(
    value: Path,
    *,
    repo_root: Path,
    label: str,
    must_exist: bool = True,
) -> Path:
    raw = value if value.is_absolute() else repo_root / value
    try:
        resolved = raw.resolve(strict=must_exist)
    except OSError as exc:
        raise StagingBundleError(f"{label} cannot be resolved: {raw}: {exc}") from exc
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise StagingBundleError(f"{label} must stay inside the repository: {raw}") from exc
    return resolved


def _existing_file(value: Path, *, repo_root: Path, label: str) -> Path:
    path = _repo_path(value, repo_root=repo_root, label=label)
    if not path.is_file():
        raise StagingBundleError(f"{label} is not a file: {path}")
    return path


def _strict_candidate_dir(
    value: Path,
    *,
    repo_root: Path,
    allowed_root: Path,
    label: str,
) -> Path:
    path = _repo_path(value, repo_root=repo_root, label=label)
    if not path.is_dir():
        raise StagingBundleError(f"{label} is not a directory: {path}")
    try:
        relative = path.relative_to(allowed_root.resolve(strict=False))
    except ValueError as exc:
        raise StagingBundleError(
            f"{label} must be below {allowed_root.resolve(strict=False)}: {path}"
        ) from exc
    if not relative.parts:
        raise StagingBundleError(f"{label} must name an isolated candidate run: {path}")
    return path


def _exact_child_file(directory: Path, filename: str, *, label: str) -> Path:
    path = directory / filename
    if not path.is_file():
        raise StagingBundleError(f"Missing exact {label} path: {path}")
    if path.resolve(strict=True).parent != directory.resolve(strict=True):
        raise StagingBundleError(f"{label} must be a real file directly under {directory}")
    return path.resolve(strict=True)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StagingBundleError(f"{label} is not valid UTF-8 JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StagingBundleError(f"{label} must contain a JSON object: {path}")
    return payload


def _validate_inventory_payload(
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    payload = _load_json_object(path, label=label)
    if payload.get("schema_version") != 1:
        raise StagingBundleError(f"{label} must use schema_version 1")
    if not str(payload.get("run_id") or "").strip() or not str(
        payload.get("generated_at_utc") or ""
    ).strip():
        raise StagingBundleError(f"{label} is missing run/timestamp identity")
    rows = payload.get("files")
    if not isinstance(rows, list) or not rows or not all(
        isinstance(row, dict) for row in rows
    ):
        raise StagingBundleError(f"{label} files must be a nonempty object list")
    paths = [str(row.get("path") or "") for row in rows]
    if any(not path for path in paths) or len(paths) != len(set(paths)):
        raise StagingBundleError(f"{label} paths must be nonempty and unique")
    for row in rows:
        expected_category = (
            "raw_input" if str(row["path"]).startswith("data/raw/") else "source"
        )
        if (
            row.get("category") != expected_category
            or not isinstance(row.get("bytes"), int)
            or int(row["bytes"]) < 0
            or not SHA256_RE.fullmatch(str(row.get("sha256") or ""))
        ):
            raise StagingBundleError(
                f"{label} row identity is invalid for {row['path']}"
            )
    source_count = sum(row.get("category") == "source" for row in rows)
    raw_count = sum(row.get("category") == "raw_input" for row in rows)
    if (
        payload.get("file_count") != len(rows)
        or payload.get("source_file_count") != source_count
        or payload.get("raw_input_file_count") != raw_count
        or str(payload.get("inventory_sha256") or "").lower()
        != _aggregate_identities(rows)
    ):
        raise StagingBundleError(f"{label} counts or aggregate are invalid")
    status_lines = payload.get("git_status")
    if not isinstance(status_lines, list) or not all(
        isinstance(line, str) for line in status_lines
    ):
        raise StagingBundleError(f"{label} git_status must be a string list")
    status_bytes = ("\n".join(status_lines) + ("\n" if status_lines else "")).encode()
    if _sha256_bytes(status_bytes) != payload.get("git_status_sha256"):
        raise StagingBundleError(f"{label} git_status hash is internally invalid")
    if not GIT_SHA_RE.fullmatch(str(payload.get("git_head") or "")) or not SHA256_RE.fullmatch(
        str(payload.get("git_diff_sha256") or "")
    ):
        raise StagingBundleError(f"{label} git identity is invalid")
    return payload


def _raw_inventory(payload: dict[str, Any]) -> dict[str, object]:
    raw_rows = [
        row
        for row in payload.get("files", [])
        if isinstance(row, dict) and row.get("category") == "raw_input"
    ]
    return {
        "file_count": len(raw_rows),
        "inventory_sha256": _aggregate_identities(raw_rows),
        "files": raw_rows,
    }


def _validate_input_inventories(
    pretraining_path: Path,
    assembly_path: Path,
    *,
    repo_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, object],
    dict[str, object],
]:
    pretraining = _validate_inventory_payload(
        pretraining_path,
        label="Pretraining model input inventory",
    )
    assembly = _validate_inventory_payload(
        assembly_path,
        label="Assembly model input inventory",
    )

    module_root = model_input_inventory.REPO_ROOT.resolve(strict=True)
    if module_root != repo_root:
        raise StagingBundleError(
            "Model inventory helper root does not match the staging repository"
        )
    current = model_input_inventory.build_inventory(
        run_id=str(assembly.get("run_id") or "staging-verification")
    )
    current_comparison = model_input_inventory.compare_inventories(assembly, current)
    if not current_comparison.get("ok"):
        raise StagingBundleError(
            "Assembly model input inventory scoped files changed after it was captured: "
            f"added={current_comparison.get('added')}, "
            f"removed={current_comparison.get('removed')}, "
            f"changed={current_comparison.get('changed')}"
        )
    if str(assembly.get("inventory_sha256") or "").lower() != str(
        current.get("inventory_sha256") or ""
    ).lower():
        raise StagingBundleError("Assembly model input inventory aggregate hash is invalid")
    for field in ("git_head", "git_diff_sha256"):
        if str(assembly.get(field) or "").lower() != str(current.get(field) or "").lower():
            raise StagingBundleError(
                f"Assembly model input inventory {field} no longer matches the repository"
            )

    pretraining_to_assembly = model_input_inventory.compare_inventories(
        pretraining, assembly
    )
    added = list(pretraining_to_assembly.get("added") or [])
    removed = list(pretraining_to_assembly.get("removed") or [])
    changed = list(pretraining_to_assembly.get("changed") or [])
    if (
        added
        or removed
        or set(changed) != ASSEMBLY_INVENTORY_ALLOWED_CHANGED_PATHS
        or not pretraining_to_assembly.get("git_head_matches")
    ):
        raise StagingBundleError(
            "Pretraining-to-assembly scoped diff must contain only the allowlisted "
            "posttraining validation/packaging changes: "
            f"added={added}, removed={removed}, changed={changed}, "
            f"git_head_matches={pretraining_to_assembly.get('git_head_matches')}"
        )
    pretraining_raw = _raw_inventory(pretraining)
    assembly_raw = _raw_inventory(assembly)
    if pretraining_raw != assembly_raw:
        raise StagingBundleError(
            "Raw inputs must be byte-identical between pretraining and assembly inventories"
        )
    inventory_delta = {
        "allowlisted_changed_paths": sorted(
            ASSEMBLY_INVENTORY_ALLOWED_CHANGED_PATHS
        ),
        "added": added,
        "removed": removed,
        "changed": changed,
        "git_head_matches": True,
        "git_diff_hash_matches": bool(
            pretraining_to_assembly.get("git_diff_matches")
        ),
        "all_raw_inputs_identical": True,
        "only_allowlisted_assembly_change": True,
        "pretraining_inventory_sha256": pretraining["inventory_sha256"],
        "assembly_inventory_sha256": assembly["inventory_sha256"],
    }
    return pretraining, assembly, pretraining_raw, inventory_delta


def _validate_bfo_ledger(
    path: Path,
    *,
    repo_root: Path,
    inventory_payload: dict[str, Any],
) -> dict[str, object]:
    from scripts.recover_bfo_moneyline_gaps import SPORTSBOOK_DISPLAY_NAMES

    raw_root = (repo_root / "data" / "raw").resolve(strict=True)
    try:
        relative_raw = path.relative_to(raw_root)
    except ValueError as exc:
        raise StagingBundleError(
            f"BFO provenance ledger must be under data/raw: {path}"
        ) from exc
    if path.name != APPROVED_BFO_LEDGER_NAME or _sha256_file(path) != APPROVED_BFO_LEDGER_SHA256:
        raise StagingBundleError("BFO provenance ledger is not the approved corrected ledger")

    repo_relative = path.relative_to(repo_root).as_posix()
    inventory_rows = {
        str(row.get("path")): row
        for row in inventory_payload.get("files", [])
        if isinstance(row, dict)
    }
    inventory_row = inventory_rows.get(repo_relative)
    if inventory_row is None or inventory_row.get("category") != "raw_input":
        raise StagingBundleError(
            "BFO provenance ledger is missing from the raw input inventory"
        )

    ledger_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StagingBundleError(
                    f"BFO provenance ledger line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise StagingBundleError(
                    f"BFO provenance ledger line {line_number} is not an object"
                )
            ledger_rows.append(row)
    if len(ledger_rows) != 244:
        raise StagingBundleError("BFO provenance ledger must contain exactly 244 records")

    odds_root = path.parent
    actual_csv_names = {item.name for item in odds_root.glob("historical_odds_bfo_recovered_*.csv")}
    if actual_csv_names != set(APPROVED_BFO_CSVS):
        raise StagingBundleError("BFO corrected CSV set is not exactly the approved six files")
    csv_rows: dict[str, dict[tuple[object, ...], dict[str, str]]] = {}
    csv_identities: list[dict[str, object]] = []
    key_fields = ("event_date", "fighter_a", "fighter_b", "query_date", "offset_days")
    for filename, (expected_rows, expected_sha) in APPROVED_BFO_CSVS.items():
        csv_path = odds_root / filename
        if _sha256_file(csv_path) != expected_sha:
            raise StagingBundleError(f"Corrected BFO CSV hash is not approved: {filename}")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_rows:
            raise StagingBundleError(f"Corrected BFO CSV row count is invalid: {filename}")
        indexed: dict[tuple[object, ...], dict[str, str]] = {}
        for row in rows:
            try:
                key = tuple(row[field] for field in key_fields[:-1]) + (
                    int(row["offset_days"]),
                )
            except (KeyError, ValueError) as exc:
                raise StagingBundleError(f"Corrected BFO CSV schema is invalid: {filename}") from exc
            if key in indexed:
                raise StagingBundleError(f"Corrected BFO CSV has a duplicate recovery key: {key}")
            indexed[key] = row
        csv_rows[filename] = indexed
        csv_identities.append(
            {
                "path": csv_path.relative_to(repo_root).as_posix(),
                "rows": len(rows),
                **_file_identity(csv_path),
            }
        )

    required_top = {
        "schema_version",
        "decision",
        "recovery_key",
        "requested_fighters",
        "input_batch",
        "parser",
        "thresholds",
        "event_page",
        "matched_bfo_rows",
        "paired_quotes",
        "consensus",
        "csv_values",
        "rejection_reason",
    }
    accepted_keys: dict[str, set[tuple[object, ...]]] = {
        filename: set() for filename in APPROVED_BFO_CSVS
    }
    decisions = {"accepted": 0, "rejected": 0}
    for index, row in enumerate(ledger_rows, start=1):
        if not required_top.issubset(row) or row.get("schema_version") != 1:
            raise StagingBundleError(f"BFO ledger record {index} has an invalid schema")
        decision = row.get("decision")
        if decision not in decisions:
            raise StagingBundleError(f"BFO ledger record {index} has an invalid decision")
        decisions[decision] += 1
        parser = row.get("parser")
        parser_files = parser.get("file_sha256") if isinstance(parser, dict) else None
        if (
            not isinstance(parser, dict)
            or not GIT_SHA_RE.fullmatch(str(parser.get("git_head") or ""))
            or not SHA256_RE.fullmatch(str(parser.get("dirty_diff_sha256") or ""))
            or not isinstance(parser_files, dict)
            or set(parser_files)
            != {
                "scripts/recover_bfo_moneyline_gaps.py",
                "scripts/revalidate_bfo_recovery_file.py",
            }
            or not all(SHA256_RE.fullmatch(str(value or "")) for value in parser_files.values())
        ):
            raise StagingBundleError(f"BFO ledger record {index} lacks parser identity")
        if any(
            inventory_rows.get(parser_path, {}).get("sha256") != parser_sha
            for parser_path, parser_sha in parser_files.items()
        ):
            raise StagingBundleError(
                f"BFO ledger record {index} parser hashes do not match inventoried source"
            )
        recovery = row.get("recovery_key")
        requested = row.get("requested_fighters")
        if not isinstance(recovery, dict) or not isinstance(requested, dict):
            raise StagingBundleError(f"BFO ledger record {index} lacks fight identity")
        try:
            key = tuple(recovery[field] for field in key_fields[:-1]) + (
                int(recovery["offset_days"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StagingBundleError(f"BFO ledger record {index} has a bad recovery key") from exc
        if (
            requested.get("fighter_a") != recovery.get("fighter_a")
            or requested.get("fighter_b") != recovery.get("fighter_b")
        ):
            raise StagingBundleError(f"BFO ledger record {index} fighter identity disagrees")
        batch_marker = ":data/raw/historical_odds/"
        input_batch = str(row.get("input_batch") or "")
        if batch_marker not in input_batch:
            raise StagingBundleError(f"BFO ledger record {index} has a bad input batch")
        filename = input_batch.split(batch_marker, 1)[1]
        if filename not in APPROVED_BFO_CSVS:
            raise StagingBundleError(f"BFO ledger record {index} names an unapproved input")

        thresholds = row.get("thresholds")
        if not isinstance(thresholds, dict) or thresholds.get("minimum_paired_sportsbooks") != 3:
            raise StagingBundleError(f"BFO ledger record {index} has invalid thresholds")
        if decision == "rejected":
            if not str(row.get("rejection_reason") or "").strip():
                raise StagingBundleError(f"Rejected BFO ledger record {index} lacks a reason")
            continue

        event_page = row.get("event_page")
        consensus = row.get("consensus")
        csv_values = row.get("csv_values")
        quotes = row.get("paired_quotes")
        if (
            not isinstance(event_page, dict)
            or not str(event_page.get("url") or "").startswith(("http://", "https://"))
            or not SHA256_RE.fullmatch(str(event_page.get("content_sha256") or ""))
            or not str(event_page.get("fetched_at_utc") or "").strip()
            or not isinstance(consensus, dict)
            or not isinstance(csv_values, dict)
            or not isinstance(quotes, list)
            or not row.get("matched_bfo_rows")
            or row.get("rejection_reason") not in ("", None)
        ):
            raise StagingBundleError(f"Accepted BFO ledger record {index} lacks provenance")
        try:
            datetime.fromisoformat(str(event_page["fetched_at_utc"]).replace("Z", "+00:00"))
            a_prob = float(consensus["a_fair_prob"])
            b_prob = float(consensus["b_fair_prob"])
            books = int(consensus["num_bookmakers"])
            accepted_quotes = [quote for quote in quotes if quote.get("accepted") is True]
            market_ids = [int(quote["market_id"]) for quote in accepted_quotes]
        except (KeyError, TypeError, ValueError) as exc:
            raise StagingBundleError(f"Accepted BFO ledger record {index} is malformed") from exc
        if (
            not math.isfinite(a_prob)
            or not math.isfinite(b_prob)
            or not 0.0 < a_prob < 1.0
            or not 0.0 < b_prob < 1.0
            or abs(a_prob + b_prob - 1.0) > 1e-9
            or books < 3
            or len(accepted_quotes) != books
            or len(market_ids) != len(set(market_ids))
            or any(market_id not in SPORTSBOOK_DISPLAY_NAMES for market_id in market_ids)
            or any(
                quote.get("book_name") != SPORTSBOOK_DISPLAY_NAMES[int(quote["market_id"])]
                for quote in accepted_quotes
            )
        ):
            raise StagingBundleError(f"Accepted BFO ledger record {index} fails quote semantics")
        for quote in accepted_quotes:
            try:
                overround = float(quote["overround"])
                quote_prob = float(quote["a_fair_prob"])
            except (KeyError, TypeError, ValueError) as exc:
                raise StagingBundleError(f"Accepted BFO quote in record {index} is malformed") from exc
            if (
                not str(quote.get("book_name") or "").strip()
                or not math.isfinite(overround)
                or not float(thresholds["minimum_book_overround"])
                <= overround
                <= float(thresholds["maximum_book_overround"])
                or not 0.0 < quote_prob < 1.0
            ):
                raise StagingBundleError(f"Accepted BFO quote in record {index} is invalid")
        csv_row = csv_rows[filename].get(key)
        if csv_row is None or key in accepted_keys[filename]:
            raise StagingBundleError(f"BFO ledger record {index} does not reconcile uniquely")
        for field in (
            "a_fair_prob",
            "b_fair_prob",
            "a_decimal_odds",
            "b_decimal_odds",
            "num_bookmakers",
        ):
            if float(csv_row[field]) != float(csv_values[field]):
                raise StagingBundleError(f"BFO ledger record {index} CSV values disagree")
        if csv_row.get("source_url") != event_page.get("url"):
            raise StagingBundleError(f"BFO ledger record {index} source URL disagrees")
        accepted_keys[filename].add(key)

    if decisions != {"accepted": 234, "rejected": 10}:
        raise StagingBundleError(f"BFO ledger decisions are incomplete: {decisions}")
    for filename, indexed in csv_rows.items():
        if accepted_keys[filename] != set(indexed):
            raise StagingBundleError(f"BFO ledger does not exactly reconcile {filename}")
    identity = _file_identity(path)
    return {
        "source_path": repo_relative,
        "raw_relative_path": relative_raw.as_posix(),
        "line_count": len(ledger_rows),
        "accepted_records": decisions["accepted"],
        "rejected_records": decisions["rejected"],
        "corrected_csv_files": csv_identities,
        "corrected_csv_aggregate_sha256": _aggregate_identities(csv_identities),
        **identity,
    }


def _registered_payload(spec_name: str, *, label: str) -> dict[str, Any]:
    try:
        payload = asdict(resolve_named_training_spec(spec_name))
    except (TypeError, ValueError) as exc:
        raise StagingBundleError(f"{label} is not a registered training spec: {exc}") from exc
    return payload


def _without_fit_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(payload)
    for field in FIT_ONLY_SPEC_FIELDS:
        result.pop(field, None)
    return result


def _effective_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a deferred odds seed to its documented effective XGB seed."""
    result = _without_fit_metadata(payload)
    if result.get("odds_noise_seed") is None:
        xgb_params = result.get("xgb_params")
        if not isinstance(xgb_params, dict) or xgb_params.get("random_state") is None:
            raise StagingBundleError(
                "A deferred odds_noise_seed requires an explicit XGBoost random_state"
            )
        result["odds_noise_seed"] = int(xgb_params["random_state"])
    return result


def _load_model_artifact(path: Path, *, label: str) -> dict[str, Any]:
    try:
        import joblib

        result = joblib.load(path)
    except Exception as exc:
        raise StagingBundleError(f"Unable to load {label} artifact {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise StagingBundleError(f"{label} artifact is not a model-result mapping: {path}")
    spec = result.get("training_spec")
    feature_cols = result.get("feature_cols")
    if not isinstance(spec, dict) or not isinstance(feature_cols, list):
        raise StagingBundleError(f"{label} artifact has no complete embedded contract")
    if feature_cols != spec.get("feature_cols"):
        raise StagingBundleError(
            f"{label} feature_cols do not exactly match its embedded training spec"
        )
    return result


def _validate_contracts(
    *,
    model_paths: dict[str, Path],
    sidecar_path: Path,
    evaluation_spec_name: str,
    expected_git_head: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    results = {
        label: _load_model_artifact(path, label=label)
        for label, path in model_paths.items()
    }
    embedded = {
        label: deepcopy(result["training_spec"]) for label, result in results.items()
    }
    for label, spec in embedded.items():
        git_hash = str(spec.get("git_hash") or "").strip().lower()
        trained_at = str(spec.get("trained_at") or "").strip()
        if git_hash != expected_git_head.lower():
            raise StagingBundleError(
                f"{label} embedded git_hash does not match the inventoried training source HEAD"
            )
        try:
            datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise StagingBundleError(
                f"{label} embedded trained_at is missing or invalid"
            ) from exc
    primary = embedded["primary"]
    fullfit_name = str(primary.get("name") or "").strip()
    if not fullfit_name:
        raise StagingBundleError("Primary embedded spec has no name")
    registered_fullfit = _registered_payload(fullfit_name, label="Full-fit spec")
    registered_evaluation = _registered_payload(
        evaluation_spec_name, label="Evaluation spec"
    )
    if (
        evaluation_spec_name != APPROVED_EVALUATION_SPEC
        or fullfit_name != APPROVED_FULLFIT_SPEC
    ):
        raise StagingBundleError(
            "Only the approved corrected durability evaluation/full-fit pair may be staged"
        )
    if _canonical_json_sha256(registered_evaluation) != APPROVED_EVALUATION_PAYLOAD_SHA256:
        raise StagingBundleError(
            "Registered evaluation payload no longer matches the corrected comparison contract"
        )
    if _without_fit_metadata(primary) != _without_fit_metadata(registered_fullfit):
        raise StagingBundleError(
            "Primary embedded spec differs from the registered full-fit spec beyond "
            "git_hash/trained_at"
        )

    saved = _load_json_object(sidecar_path, label="Saved full-fit spec sidecar")
    if saved != primary:
        raise StagingBundleError(
            "Saved full-fit spec sidecar does not exactly match the primary embedded spec"
        )
    if embedded["logistic"] != primary:
        raise StagingBundleError(
            "Logistic embedded spec does not exactly match the primary embedded spec"
        )
    expected_no_odds = _expected_no_odds_spec_payload(primary)
    if embedded["no_odds"] != expected_no_odds:
        raise StagingBundleError(
            "No-odds embedded spec is not the exact derived name/description/features variant"
        )

    if registered_fullfit.get("odds_noise_seed") != 42 or (
        registered_fullfit.get("xgb_params") or {}
    ).get("random_state") != 42:
        raise StagingBundleError(
            "The selected full-fit contract must explicitly pin both model and odds-noise seeds to 42"
        )
    eval_compare = _effective_contract(registered_evaluation)
    fullfit_compare = _effective_contract(registered_fullfit)
    differences = {
        key
        for key in set(eval_compare) | set(fullfit_compare)
        if eval_compare.get(key) != fullfit_compare.get(key)
    }
    if differences != FULLFIT_ALLOWED_DIFFERENCES:
        raise StagingBundleError(
            "Evaluation/full-fit registered specs must differ exactly in name, "
            f"description, and cutoff; observed={sorted(differences)}"
        )
    return registered_evaluation, registered_fullfit, results


def _csv_identity(path: Path) -> dict[str, object]:
    import pandas as pd

    try:
        columns = list(pd.read_csv(path, nrows=0).columns)
        if "event_date" not in columns:
            raise StagingBundleError(f"CSV is missing event_date: {path}")
        dates = pd.read_csv(path, usecols=["event_date"])["event_date"]
    except StagingBundleError:
        raise
    except Exception as exc:
        raise StagingBundleError(f"Unable to inspect CSV {path}: {exc}") from exc
    parsed = pd.to_datetime(dates, errors="coerce")
    if len(dates) and parsed.isna().any():
        raise StagingBundleError(f"CSV contains unparseable event_date values: {path}")
    max_date = None if parsed.empty else parsed.max().date().isoformat()
    return {
        **_file_identity(path),
        "rows": int(len(dates)),
        "columns": len(columns),
        "column_names_sha256": _canonical_json_sha256(columns),
        "max_event_date": max_date,
    }


def _validate_test_set_metadata(
    *,
    test_set_path: Path,
    metadata_path: Path,
    primary_spec: dict[str, Any],
    expected_test_frame,
) -> dict[str, object]:
    import pandas as pd

    metadata = _load_json_object(metadata_path, label="Test-set metadata")
    test_identity = _file_identity(test_set_path)
    if str(metadata.get("test_set_sha256") or "").lower() != test_identity["sha256"]:
        raise StagingBundleError("Test-set metadata hash does not match test_set.csv")
    if metadata.get("training_spec") != primary_spec:
        raise StagingBundleError(
            "Test-set metadata training_spec does not exactly match the primary artifact"
        )
    feature_cols = primary_spec.get("feature_cols")
    expected_feature_hash = _canonical_json_sha256(feature_cols)
    if metadata.get("spec_name") != primary_spec.get("name"):
        raise StagingBundleError("Test-set metadata spec_name is incorrect")
    if metadata.get("feature_count") != len(feature_cols):
        raise StagingBundleError("Test-set metadata feature_count is incorrect")
    if str(metadata.get("feature_hash") or "").lower() != expected_feature_hash:
        raise StagingBundleError("Test-set metadata feature_hash is incorrect")
    try:
        frame = pd.read_csv(test_set_path)
    except Exception as exc:
        raise StagingBundleError(f"Unable to read test_set.csv: {exc}") from exc
    if metadata.get("row_count") != len(frame):
        raise StagingBundleError("Test-set metadata row_count is incorrect")
    expected_csv = expected_test_frame.to_csv(index=False).encode("utf-8")
    if test_set_path.read_bytes() != expected_csv:
        raise StagingBundleError(
            "test_set.csv does not exactly match the split reconstructed from features.csv"
        )
    return {
        **test_identity,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "metadata_sha256": _sha256_file(metadata_path),
        "metadata_bytes": int(metadata_path.stat().st_size),
    }


def _finite_inference(
    *,
    features_path: Path,
    primary_spec: dict[str, Any],
    model_results: dict[str, dict[str, Any]],
    sample_rows: int,
) -> tuple[dict[str, object], object, object]:
    import numpy as np
    import pandas as pd

    from src.model.predict import predict_batch
    from src.model.train import _mirror_augment_training_rows, prepare_train_test
    from src.model.training_spec import materialize_and_validate_spec_features

    if sample_rows < 1:
        raise StagingBundleError("inference_sample_rows must be at least one")
    try:
        features = pd.read_csv(features_path, low_memory=False)
    except Exception as exc:
        raise StagingBundleError(f"Unable to read training features: {exc}") from exc
    declared = list(primary_spec.get("feature_cols") or [])
    missing = [column for column in declared if column not in features.columns]
    if missing:
        raise StagingBundleError(
            f"Training features are missing declared contract columns: {missing}"
        )
    try:
        registered_spec = resolve_named_training_spec(str(primary_spec.get("name") or ""))
        materialized = materialize_and_validate_spec_features(features, registered_spec)
        training, test, _ = prepare_train_test(
            materialized,
            cutoff_date=str(primary_spec.get("train_cutoff_date") or ""),
            feature_cols=declared,
            start_date=str(primary_spec.get("train_start_date") or "") or None,
            end_date=str(primary_spec.get("train_end_date") or "") or None,
        )
    except Exception as exc:
        raise StagingBundleError(f"Unable to reconstruct the training sample: {exc}") from exc
    if training.empty:
        raise StagingBundleError("No eligible training rows are available for inference")
    row_reconciliation: dict[str, object] = {}
    for label, result in model_results.items():
        observed = result.get("observed_training_rows")
        effective = result.get("effective_training_rows")
        _, mirrored = _mirror_augment_training_rows(training, result["feature_cols"])
        expected_effective = len(training) * (2 if mirrored else 1)
        if observed != len(training) or effective != expected_effective:
            raise StagingBundleError(
                f"{label} artifact training-row metadata does not match reconstructed data: "
                f"observed={observed}/{len(training)}, effective={effective}/{expected_effective}"
            )
        row_reconciliation[label] = {
            "observed_training_rows": int(observed),
            "effective_training_rows": int(effective),
            "mirror_augmentation": mirrored,
        }
    count = min(int(sample_rows), len(training))
    positions = np.linspace(0, len(training) - 1, num=count, dtype=int)
    sample = training.iloc[positions].copy()
    summary: dict[str, object] = {
        "sample_rows": count,
        "eligible_training_rows": int(len(training)),
        "reconstructed_test_rows": int(len(test)),
        "artifact_row_reconciliation": row_reconciliation,
        "sample_event_date_min": sample["event_date"].min().date().isoformat(),
        "sample_event_date_max": sample["event_date"].max().date().isoformat(),
    }
    for label, result in model_results.items():
        try:
            predicted = predict_batch(sample, model_result=result)
            probabilities = predicted["prob_a"].to_numpy(dtype=float)
        except Exception as exc:
            raise StagingBundleError(f"Finite inference failed for {label}: {exc}") from exc
        if (
            len(probabilities) != count
            or not np.isfinite(probabilities).all()
            or (probabilities < 0.0).any()
            or (probabilities > 1.0).any()
        ):
            raise StagingBundleError(
                f"{label} did not emit finite probabilities in [0, 1]"
            )
        summary[label] = {
            "finite_probability_count": int(len(probabilities)),
            "probability_min": float(probabilities.min()),
            "probability_max": float(probabilities.max()),
        }
    return summary, training, test


def _files_byte_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(1024 * 1024)
            right_chunk = right_handle.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _read_semantic_csv(path: Path, *, label: str):
    import pandas as pd

    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise StagingBundleError(f"Unable to read {label} CSV: {path}: {exc}") from exc


def _semantic_frame_equivalence(
    audit_frame: object,
    trainer_frame: object,
    *,
    label: str,
    atol: float = SEMANTIC_EQUIVALENCE_ATOL,
) -> dict[str, object]:
    import numpy as np
    import pandas as pd

    if atol < 0:
        raise StagingBundleError("Semantic-equivalence atol must be nonnegative")
    if audit_frame.shape != trainer_frame.shape:
        raise StagingBundleError(
            f"{label} semantic shape mismatch: "
            f"audit={audit_frame.shape}, trainer={trainer_frame.shape}"
        )
    audit_columns = list(audit_frame.columns)
    trainer_columns = list(trainer_frame.columns)
    if audit_columns != trainer_columns:
        raise StagingBundleError(f"{label} semantic column order mismatch")
    audit_dtypes = [str(dtype) for dtype in audit_frame.dtypes]
    trainer_dtypes = [str(dtype) for dtype in trainer_frame.dtypes]
    if audit_dtypes != trainer_dtypes:
        raise StagingBundleError(f"{label} semantic dtype mismatch")

    required_exact = ["event_date", "fighter_a", "fighter_b", "winner", "target"]
    missing_required = [column for column in required_exact if column not in audit_columns]
    if missing_required:
        raise StagingBundleError(
            f"{label} semantic identity is missing key/target columns: {missing_required}"
        )
    audit_missing = audit_frame.isna().to_numpy(dtype=bool)
    trainer_missing = trainer_frame.isna().to_numpy(dtype=bool)
    if not np.array_equal(audit_missing, trainer_missing):
        raise StagingBundleError(f"{label} semantic NaN-mask mismatch")

    for column in required_exact:
        if not audit_frame[column].equals(trainer_frame[column]):
            kind = "key" if column in {"event_date", "fighter_a", "fighter_b"} else "target"
            raise StagingBundleError(
                f"{label} semantic {kind} mismatch in column {column}"
            )

    numeric_columns: list[str] = []
    exact_nonfloat_columns: list[str] = []
    max_abs_delta = 0.0
    max_abs_delta_column: str | None = None
    numerically_changed_cells = 0
    for column in audit_columns:
        if column in required_exact:
            continue
        if pd.api.types.is_float_dtype(audit_frame[column].dtype):
            numeric_columns.append(column)
            audit_values = audit_frame[column].to_numpy(dtype=float, copy=False)
            trainer_values = trainer_frame[column].to_numpy(dtype=float, copy=False)
            if not np.array_equal(np.isposinf(audit_values), np.isposinf(trainer_values)) or not np.array_equal(
                np.isneginf(audit_values), np.isneginf(trainer_values)
            ):
                raise StagingBundleError(
                    f"{label} semantic infinity-mask mismatch in column {column}"
                )
            finite = np.isfinite(audit_values) & np.isfinite(trainer_values)
            deltas = np.abs(audit_values[finite] - trainer_values[finite])
            column_max = float(deltas.max()) if deltas.size else 0.0
            if column_max > atol:
                raise StagingBundleError(
                    f"{label} semantic numeric drift exceeds atol={atol:g} in "
                    f"column {column}: max_abs_delta={column_max:.17g}"
                )
            numerically_changed_cells += int(np.count_nonzero(deltas))
            if column_max > max_abs_delta:
                max_abs_delta = column_max
                max_abs_delta_column = column
        else:
            exact_nonfloat_columns.append(column)
            if not audit_frame[column].equals(trainer_frame[column]):
                value_kind = (
                    "integer/bool"
                    if pd.api.types.is_numeric_dtype(audit_frame[column].dtype)
                    else "nonnumeric"
                )
                raise StagingBundleError(
                    f"{label} semantic {value_kind} mismatch in column {column}"
                )

    packed_missing = np.packbits(audit_missing.reshape(-1)).tobytes()
    column_order_sha = _canonical_json_sha256(audit_columns)
    dtype_sequence_sha = _canonical_json_sha256(audit_dtypes)
    key_rows = audit_frame[["event_date", "fighter_a", "fighter_b"]]
    target_rows = audit_frame[["winner", "target"]]
    return {
        "equivalent": True,
        "rows": int(len(audit_frame)),
        "columns": int(len(audit_columns)),
        "column_order_exact": True,
        "column_order_sha256": column_order_sha,
        "dtypes_exact": True,
        "dtype_sequence_sha256": dtype_sequence_sha,
        "key_columns": ["event_date", "fighter_a", "fighter_b"],
        "key_values_and_order_exact": True,
        "key_rows_sha256": _sha256_bytes(
            key_rows.to_csv(index=False).encode("utf-8")
        ),
        "target_columns": ["winner", "target"],
        "target_values_and_order_exact": True,
        "target_rows_sha256": _sha256_bytes(
            target_rows.to_csv(index=False).encode("utf-8")
        ),
        "nan_masks_exact": True,
        "nan_mask_sha256": _sha256_bytes(packed_missing),
        "integer_bool_and_nonnumeric_values_exact": True,
        "exact_nonfloat_column_count": len(exact_nonfloat_columns),
        "numeric_column_count": len(numeric_columns),
        "numeric_rtol": 0.0,
        "numeric_atol": float(atol),
        "numeric_max_abs_delta": max_abs_delta,
        "numeric_max_abs_delta_column": max_abs_delta_column,
        "numerically_changed_cells": numerically_changed_cells,
    }


def _semantic_csv_equivalence(
    audit_path: Path,
    trainer_path: Path,
    *,
    label: str,
    atol: float = SEMANTIC_EQUIVALENCE_ATOL,
) -> dict[str, object]:
    audit_frame = _read_semantic_csv(audit_path, label=f"audit {label}")
    trainer_frame = _read_semantic_csv(trainer_path, label=f"trainer {label}")
    report = _semantic_frame_equivalence(
        audit_frame,
        trainer_frame,
        label=label,
        atol=atol,
    )
    return {
        **report,
        "audit_sha256": _sha256_file(audit_path),
        "trainer_sha256": _sha256_file(trainer_path),
        "byte_equal": _files_byte_equal(audit_path, trainer_path),
    }


def _eligible_split_equivalence(
    audit_features: object,
    trainer_features: object,
    *,
    primary_spec: dict[str, Any],
) -> dict[str, object]:
    from src.model.train import prepare_train_test
    from src.model.training_spec import materialize_and_validate_spec_features

    try:
        registered_spec = resolve_named_training_spec(str(primary_spec.get("name") or ""))
        declared = list(primary_spec.get("feature_cols") or [])

        def split(frame):
            materialized = materialize_and_validate_spec_features(frame, registered_spec)
            training, test, _ = prepare_train_test(
                materialized,
                cutoff_date=str(primary_spec.get("train_cutoff_date") or ""),
                feature_cols=declared,
                start_date=str(primary_spec.get("train_start_date") or "") or None,
                end_date=str(primary_spec.get("train_end_date") or "") or None,
            )
            return training, test

        audit_training, audit_test = split(audit_features)
        trainer_training, trainer_test = split(trainer_features)
    except Exception as exc:
        raise StagingBundleError(
            f"Unable to reconstruct audit/trainer eligibility: {exc}"
        ) from exc

    identity_columns = ["event_date", "fighter_a", "fighter_b", "winner", "target"]
    for split_label, audit_split, trainer_split in (
        ("training", audit_training, trainer_training),
        ("test", audit_test, trainer_test),
    ):
        audit_indices = [int(index) for index in audit_split.index]
        trainer_indices = [int(index) for index in trainer_split.index]
        if audit_indices != trainer_indices or not audit_split[
            identity_columns
        ].reset_index(drop=True).equals(
            trainer_split[identity_columns].reset_index(drop=True)
        ):
            raise StagingBundleError(
                f"Audit/trainer {split_label} eligibility or row identity mismatch"
            )
    training_indices = [int(index) for index in audit_training.index]
    test_indices = [int(index) for index in audit_test.index]
    return {
        "equivalent": True,
        "training_rows": len(training_indices),
        "test_rows": len(test_indices),
        "training_eligible_indices_sha256": _canonical_json_sha256(training_indices),
        "test_eligible_indices_sha256": _canonical_json_sha256(test_indices),
        "identity_columns": identity_columns,
        "identity_values_and_order_exact": True,
    }


def _probability_sha256(values: object) -> str:
    import numpy as np

    canonical = np.asarray(values, dtype="<f8")
    return _sha256_bytes(np.ascontiguousarray(canonical).tobytes())


def _prediction_invariance(
    audit_features: object,
    trainer_features: object,
    *,
    model_results: dict[str, dict[str, Any]],
) -> dict[str, object]:
    import numpy as np

    from src.model.predict import _predict_prob_a_symmetrized

    if len(audit_features) != len(trainer_features):
        raise StagingBundleError("Prediction-invariance frames have different row counts")
    report: dict[str, object] = {
        "equivalent": True,
        "rows": int(len(audit_features)),
        "xgboost_policy": "bit_identical",
        "logistic_policy": {"rtol": 0.0, "atol": SEMANTIC_EQUIVALENCE_ATOL},
    }
    for label in ("primary", "no_odds", "logistic"):
        result = model_results.get(label)
        if not isinstance(result, dict):
            raise StagingBundleError(f"Prediction-invariance model is missing: {label}")
        try:
            audit_probabilities = np.asarray(
                _predict_prob_a_symmetrized(audit_features, result), dtype=float
            )
            trainer_probabilities = np.asarray(
                _predict_prob_a_symmetrized(trainer_features, result), dtype=float
            )
        except Exception as exc:
            raise StagingBundleError(
                f"Audit/trainer prediction invariance failed for {label}: {exc}"
            ) from exc
        if (
            audit_probabilities.shape != trainer_probabilities.shape
            or audit_probabilities.shape != (len(audit_features),)
            or not np.isfinite(audit_probabilities).all()
            or not np.isfinite(trainer_probabilities).all()
            or (audit_probabilities < 0.0).any()
            or (audit_probabilities > 1.0).any()
            or (trainer_probabilities < 0.0).any()
            or (trainer_probabilities > 1.0).any()
        ):
            raise StagingBundleError(
                f"Audit/trainer prediction invariance produced invalid probabilities for {label}"
            )
        deltas = np.abs(audit_probabilities - trainer_probabilities)
        max_abs_delta = float(deltas.max()) if deltas.size else 0.0
        bit_identical = bool(np.array_equal(audit_probabilities, trainer_probabilities))
        if label in {"primary", "no_odds"}:
            if not bit_identical:
                raise StagingBundleError(
                    f"{label} audit/trainer predictions are not bit-identical; "
                    f"max_abs_delta={max_abs_delta:.17g}"
                )
        elif not np.allclose(
            audit_probabilities,
            trainer_probabilities,
            rtol=0.0,
            atol=SEMANTIC_EQUIVALENCE_ATOL,
        ):
            raise StagingBundleError(
                "logistic audit/trainer prediction drift exceeds "
                f"atol={SEMANTIC_EQUIVALENCE_ATOL:g}; "
                f"max_abs_delta={max_abs_delta:.17g}"
            )
        report[label] = {
            "probability_count": int(len(audit_probabilities)),
            "audit_probability_sha256": _probability_sha256(audit_probabilities),
            "trainer_probability_sha256": _probability_sha256(trainer_probabilities),
            "bit_identical": bit_identical,
            "max_abs_delta": max_abs_delta,
            "required_atol": 0.0
            if label in {"primary", "no_odds"}
            else SEMANTIC_EQUIVALENCE_ATOL,
        }
    return report


def _replay_trainer_preprocessing(
    *,
    audit_fights_path: Path,
    trainer_fights_path: Path,
    trainer_features_path: Path,
    fullfit_spec_name: str,
) -> dict[str, object]:
    from src.bot import _load_training_dataframe
    from src.data.kaggle_loader import save_processed
    from src.features.build_features import build_features, save_features

    spec = resolve_named_training_spec(fullfit_spec_name)
    with tempfile.TemporaryDirectory(prefix="ufc-preprocessing-replay-") as temporary:
        replay_root = Path(temporary)
        replay_fights = replay_root / "fights_cleaned.csv"
        replay_features = replay_root / "features.csv"
        try:
            fights_frame = _load_training_dataframe(
                data_path=audit_fights_path,
                spec=spec,
            )
            save_processed(fights_frame, filename=replay_fights)
            features_frame = build_features(fights_frame)
            save_features(features_frame, filename=replay_features)
        except Exception as exc:
            raise StagingBundleError(
                f"Unable to replay the trainer preprocessing path: {exc}"
            ) from exc
        fights_match = _files_byte_equal(replay_fights, trainer_fights_path)
        features_match = _files_byte_equal(replay_features, trainer_features_path)
        replay_fights_identity = _file_identity(replay_fights)
        replay_features_identity = _file_identity(replay_features)
        if not fights_match or not features_match:
            raise StagingBundleError(
                "Trainer preprocessing replay does not byte-match the completed train "
                "outputs: "
                f"fights={fights_match} ({replay_fights_identity['sha256']}), "
                f"features={features_match} ({replay_features_identity['sha256']})"
            )

    audit_fights_sha = _sha256_file(audit_fights_path)
    trainer_fights_sha = _sha256_file(trainer_fights_path)
    audit_source_equals_trainer_output = audit_fights_sha == trainer_fights_sha
    if audit_source_equals_trainer_output:
        raise StagingBundleError(
            "Approved audit source must remain distinct from the completed trainer output"
        )
    return {
        "preprocessing_replay_byte_match": True,
        "audit_source_equals_trainer_output": False,
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "replay_path": [
            "src.bot._load_training_dataframe",
            "src.data.kaggle_loader.save_processed",
            "src.features.build_features.build_features",
            "src.features.build_features.save_features",
        ],
        "fights": {
            "audit_source_sha256": audit_fights_sha,
            "trainer_output_sha256": trainer_fights_sha,
            "replay_output_sha256": replay_fights_identity["sha256"],
            "replay_output_bytes": replay_fights_identity["bytes"],
            "byte_match": True,
        },
        "features": {
            "trainer_output_sha256": _sha256_file(trainer_features_path),
            "replay_output_sha256": replay_features_identity["sha256"],
            "replay_output_bytes": replay_features_identity["bytes"],
            "byte_match": True,
        },
        "explanation": (
            "The audit CSV is the direct --data input. The trainer reads, parses, "
            "sorts, and reserializes it before feature construction, so benign "
            "floating-point CSV text round-trips change bytes. Replaying that exact "
            "load/build/save path reproduces the completed train files byte-for-byte."
        ),
    }


def _audit_trainer_relationship(
    *,
    audit_fights_path: Path,
    audit_features_path: Path,
    trainer_fights_path: Path,
    trainer_features_path: Path,
    primary_spec: dict[str, Any],
    model_results: dict[str, dict[str, Any]],
) -> dict[str, object]:
    fights_report = _semantic_csv_equivalence(
        audit_fights_path,
        trainer_fights_path,
        label="fights",
    )
    features_report = _semantic_csv_equivalence(
        audit_features_path,
        trainer_features_path,
        label="features",
    )
    if fights_report["byte_equal"] or features_report["byte_equal"]:
        raise StagingBundleError(
            "Approved audit and trainer snapshots must retain both distinct exact identities"
        )
    audit_features = _read_semantic_csv(
        audit_features_path, label="audit features eligibility"
    )
    trainer_features = _read_semantic_csv(
        trainer_features_path, label="trainer features eligibility"
    )
    eligibility = _eligible_split_equivalence(
        audit_features,
        trainer_features,
        primary_spec=primary_spec,
    )
    predictions = _prediction_invariance(
        audit_features,
        trainer_features,
        model_results=model_results,
    )
    return {
        "equivalent": True,
        "policy": {
            "numeric_rtol": 0.0,
            "numeric_atol": SEMANTIC_EQUIVALENCE_ATOL,
            "exact": [
                "shape",
                "column_order",
                "dtypes",
                "key_values_and_order",
                "target_values_and_order",
                "eligibility",
                "nan_masks",
                "nonnumeric_values",
            ],
        },
        "fights": fights_report,
        "features": features_report,
        "eligibility": eligibility,
        "prediction_invariance": predictions,
    }


def _argv_option(argv: Sequence[str], option: str) -> str:
    matches: list[str] = []
    for index, token in enumerate(argv):
        if token == option:
            if index + 1 >= len(argv):
                raise StagingBundleError(f"Training argv option {option} has no value")
            matches.append(argv[index + 1])
        elif token.startswith(f"{option}="):
            matches.append(token.split("=", 1)[1])
    if len(matches) != 1 or not str(matches[0]).strip():
        raise StagingBundleError(
            f"Training argv must contain exactly one {option} value"
        )
    return matches[0]


def _validate_training_argv(
    argv: Sequence[str],
    *,
    repo_root: Path,
    fullfit_spec_name: str,
    candidate_models_dir: Path,
    candidate_processed_dir: Path,
    expected_fights_sha256: str,
    expected_features_sha256: str,
) -> dict[str, object]:
    if not argv or any(not isinstance(token, str) or not token for token in argv):
        raise StagingBundleError("Exact training argv must be a nonempty string array")
    executable_token = argv[0]
    executable_path = Path(executable_token)
    if not executable_path.is_absolute():
        direct = (repo_root / executable_path).resolve(strict=False)
        executable_path = direct if direct.exists() else Path(
            shutil.which(executable_token) or direct
        )
    try:
        same_python = os.path.samefile(executable_path, Path(sys.executable))
    except OSError:
        same_python = False
    if not same_python:
        raise StagingBundleError(
            "Training argv Python executable must match the interpreter building the bundle: "
            f"argv={executable_path}, builder={sys.executable}"
        )
    if len(argv) != 10 or list(argv[1:4]) != ["-m", "src.bot", "train"] or list(
        argv[4::2]
    ) != ["--data", "--spec", "--output-subdir"]:
        raise StagingBundleError(
            "Training argv must exactly be: PYTHON -m src.bot train --data PATH "
            "--spec NAME --output-subdir PATH"
        )
    if _argv_option(argv, "--spec") != fullfit_spec_name:
        raise StagingBundleError("Training argv --spec does not match the primary artifact")

    output_subdir_raw = _argv_option(argv, "--output-subdir")
    output_subdir = Path(output_subdir_raw)
    if output_subdir.is_absolute() or ".." in output_subdir.parts:
        raise StagingBundleError("Training --output-subdir must be safe and relative")
    expected_models = (repo_root / "models" / output_subdir).resolve(strict=False)
    expected_processed = (
        repo_root / "data" / "processed" / output_subdir
    ).resolve(strict=False)
    if expected_models != candidate_models_dir or expected_processed != candidate_processed_dir:
        raise StagingBundleError(
            "Candidate directories do not exactly match training --output-subdir"
        )

    data_raw = Path(_argv_option(argv, "--data"))
    data_path = _existing_file(data_raw, repo_root=repo_root, label="Training --data input")
    if data_path.name != "fights_cleaned.csv" or data_path.parent == candidate_processed_dir:
        raise StagingBundleError(
            "Training --data must be the independent audit rebuild's fights_cleaned.csv"
        )
    candidates_root = (repo_root / "data" / "processed" / "candidates").resolve(
        strict=True
    )
    try:
        audit_relative = data_path.parent.relative_to(candidates_root)
    except ValueError as exc:
        raise StagingBundleError(
            "Training --data audit directory must be strictly below data/processed/candidates"
        ) from exc
    if not audit_relative.parts:
        raise StagingBundleError("Training audit rebuild must use a distinct run directory")
    audit_features = _exact_child_file(
        data_path.parent,
        "features.csv",
        label="independent audit features",
    )
    candidate_fights = candidate_processed_dir / "fights_cleaned.csv"
    candidate_features = candidate_processed_dir / "features.csv"
    audit_fights_sha = _sha256_file(data_path)
    audit_features_sha = _sha256_file(audit_features)
    candidate_fights_sha = _sha256_file(candidate_fights)
    candidate_features_sha = _sha256_file(candidate_features)
    if not SHA256_RE.fullmatch(expected_fights_sha256) or not SHA256_RE.fullmatch(
        expected_features_sha256
    ):
        raise StagingBundleError("Expected corrected snapshot hashes must be 64 hex characters")
    if (
        expected_fights_sha256.lower() != APPROVED_FIGHTS_SHA256
        or expected_features_sha256.lower() != APPROVED_FEATURES_SHA256
    ):
        raise StagingBundleError(
            "Caller snapshot hashes do not match the allowlisted corrected comparison snapshot"
        )
    if audit_fights_sha != expected_fights_sha256.lower():
        raise StagingBundleError(
            "Independent audit fights do not match the approved controlling snapshot hash"
        )
    if audit_features_sha != expected_features_sha256.lower():
        raise StagingBundleError(
            "Independent audit features do not match the approved controlling snapshot hash"
        )
    if candidate_fights_sha != APPROVED_TRAIN_FIGHTS_SHA256:
        raise StagingBundleError(
            "Completed trainer fights do not match the approved train-output hash"
        )
    if candidate_features_sha != APPROVED_TRAIN_FEATURES_SHA256:
        raise StagingBundleError(
            "Completed trainer features do not match the approved train-output hash"
        )
    if audit_fights_sha == candidate_fights_sha or audit_features_sha == candidate_features_sha:
        raise StagingBundleError(
            "Approved audit and completed trainer outputs must retain distinct exact identities"
        )
    return {
        "argv": list(argv),
        "argv_sha256": _canonical_json_sha256(list(argv)),
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "data_argument": str(data_raw),
        "data_input_path": data_path.relative_to(repo_root).as_posix(),
        "data_input_sha256": _sha256_file(data_path),
        "data_input_bytes": int(data_path.stat().st_size),
        "output_subdir": output_subdir.as_posix(),
        "independent_audit_snapshot": {
            "processed_dir": data_path.parent.relative_to(repo_root).as_posix(),
            "fights": {
                "path": data_path.relative_to(repo_root).as_posix(),
                "staged_path": "provenance/independent_audit_snapshot/fights_cleaned.csv",
                "sha256": audit_fights_sha,
                "bytes": int(data_path.stat().st_size),
            },
            "features": {
                "path": audit_features.relative_to(repo_root).as_posix(),
                "staged_path": "provenance/independent_audit_snapshot/features.csv",
                "sha256": audit_features_sha,
                "bytes": int(audit_features.stat().st_size),
            },
            "audit_source_equals_trainer_output": False,
            "controlling_corrected_snapshot": {
                "fights_sha256": expected_fights_sha256.lower(),
                "features_sha256": expected_features_sha256.lower(),
                "append_only_delta_used": False,
            },
            "completed_trainer_snapshot": {
                "fights_sha256": candidate_fights_sha,
                "features_sha256": candidate_features_sha,
                "fights_bytes": int(candidate_fights.stat().st_size),
                "features_bytes": int(candidate_features.stat().st_size),
            },
        },
    }


def _package_versions() -> dict[str, object]:
    distributions = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
        "xgboost": "xgboost",
        "joblib": "joblib",
    }
    versions: dict[str, str] = {}
    for label, distribution in distributions.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise StagingBundleError(
                f"Required training package is not installed: {distribution}"
            ) from exc
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve(strict=True)),
            "full_version": sys.version,
        },
        "packages": versions,
    }


def _selection_evidence(
    paths: Sequence[Path], *, repo_root: Path
) -> tuple[list[dict[str, object]], int]:
    if not paths:
        raise StagingBundleError("At least one selection evidence file is required")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    total = 0
    for raw_path in paths:
        path = _existing_file(raw_path, repo_root=repo_root, label="Selection evidence")
        relative = path.relative_to(repo_root).as_posix()
        lowered = relative.lower()
        if (
            not (relative.startswith("logs/") or relative.startswith("docs/"))
            or path.suffix.lower() not in SAFE_EVIDENCE_SUFFIXES
            or any(token in lowered for token in SENSITIVE_EVIDENCE_TOKENS)
        ):
            raise StagingBundleError(
                f"Selection evidence path is not in the approved safe scope: {relative}"
            )
        if relative in seen:
            raise StagingBundleError(f"Duplicate selection evidence path: {relative}")
        seen.add(relative)
        size = int(path.stat().st_size)
        if size > MAX_EVIDENCE_FILE_BYTES:
            raise StagingBundleError(
                f"Selection evidence exceeds {MAX_EVIDENCE_FILE_BYTES} bytes: {relative}"
            )
        total += size
        rows.append({"path": relative, "bytes": size, "sha256": _sha256_file(path)})
    if total > MAX_EVIDENCE_TOTAL_BYTES:
        raise StagingBundleError(
            f"Selection evidence exceeds {MAX_EVIDENCE_TOTAL_BYTES} total bytes"
        )
    return rows, total


def _validate_previous_rollback(
    *,
    manifest_path: Path,
    readyz_path: Path,
    repo_root: Path,
    deployed_git_sha: str,
    runtime_lookup_hashes: dict[str, str],
) -> dict[str, object]:
    if not GIT_SHA_RE.fullmatch(deployed_git_sha):
        raise StagingBundleError("Previous deployed Git SHA must be exactly 40 hex characters")
    required_runtime_hashes = {
        "processed_fights_sha256",
        "processed_features_sha256",
    }
    if set(runtime_lookup_hashes) != required_runtime_hashes:
        raise StagingBundleError(
            "Previous runtime lookup hashes must contain exactly processed fights/features"
        )
    normalized_runtime: dict[str, str] = {}
    for key, value in runtime_lookup_hashes.items():
        if not RUNTIME_HASH_KEY_RE.fullmatch(key) or not SHA256_RE.fullmatch(value):
            raise StagingBundleError(
                f"Invalid previous runtime lookup hash {key}={value!r}"
            )
        normalized_runtime[key] = value.lower()

    expected_manifest_path = (repo_root / "models/current_production_model.json").resolve(
        strict=True
    )
    if manifest_path != expected_manifest_path:
        raise StagingBundleError(
            "Previous source manifest must be models/current_production_model.json"
        )
    payload = _load_json_object(manifest_path, label="Previous source manifest")
    required_manifest_fields = (
        "bundle_id",
        "model_spec_name",
        "snapshot_max_event_date",
        "built_at",
        "git_sha",
        "processed_fights_sha256",
        "processed_features_sha256",
        "processed_fights_bytes",
        "processed_features_bytes",
        "model_sha256",
        "no_odds_model_sha256",
    )
    missing_fields = [field for field in required_manifest_fields if payload.get(field) in (None, "")]
    if missing_fields:
        raise StagingBundleError(
            f"Previous source manifest is missing rollback fields: {missing_fields}"
        )
    canonical_models = (repo_root / "models").resolve(strict=True)
    local_models: dict[str, dict[str, object]] = {}
    manifest_hash_fields = {
        "primary": "model_sha256",
        "no_odds": "no_odds_model_sha256",
        "logistic": "logistic_model_sha256",
    }
    model_results: dict[str, dict[str, Any]] = {}
    for label, filename in MODEL_FILENAMES.items():
        path = _exact_child_file(
            canonical_models, filename, label=f"previous local {label} model"
        )
        model_results[label] = _load_model_artifact(path, label=f"previous {label}")
        identity = _file_identity(path)
        pinned = str(payload.get(manifest_hash_fields[label]) or "").strip().lower()
        if pinned and pinned != identity["sha256"]:
            raise StagingBundleError(
                f"Previous source manifest {manifest_hash_fields[label]} does not match "
                f"the local rollback artifact"
            )
        local_models[label] = {
            "path": path.relative_to(repo_root).as_posix(),
            **identity,
        }
    primary_spec = deepcopy(model_results["primary"]["training_spec"])
    if model_results["logistic"]["training_spec"] != primary_spec:
        raise StagingBundleError("Previous logistic contract does not match previous primary")
    if model_results["no_odds"]["training_spec"] != _expected_no_odds_spec_payload(
        primary_spec
    ):
        raise StagingBundleError("Previous no-odds contract is not the exact derived variant")
    if primary_spec.get("name") != payload["model_spec_name"]:
        raise StagingBundleError("Previous source manifest spec does not match local models")
    previous_sidecar = _exact_child_file(
        canonical_models,
        f"{payload['model_spec_name']}_spec.json",
        label="previous saved spec sidecar",
    )
    previous_saved_spec = _load_json_object(previous_sidecar, label="Previous saved spec")
    if previous_saved_spec != primary_spec:
        raise StagingBundleError("Previous saved spec does not exactly match local models")

    local_processed: dict[str, object] = {"available": False, "mutable_lookup": True}
    processed_dir = repo_root / "data/processed"
    local_fights = processed_dir / "fights_cleaned.csv"
    local_features = processed_dir / "features.csv"
    if local_fights.is_file() and local_features.is_file():
        fights_identity = _file_identity(local_fights)
        features_identity = _file_identity(local_features)
        local_processed = {
            "available": True,
            "mutable_lookup": True,
            "fights": {
                "path": local_fights.relative_to(repo_root).as_posix(),
                **fights_identity,
                "matches_source_manifest": fights_identity["sha256"]
                == str(payload["processed_fights_sha256"]).lower(),
            },
            "features": {
                "path": local_features.relative_to(repo_root).as_posix(),
                **features_identity,
                "matches_source_manifest": features_identity["sha256"]
                == str(payload["processed_features_sha256"]).lower(),
            },
        }

    readyz = _load_json_object(readyz_path, label="Previous /readyz evidence")
    ready_bundle = readyz.get("production_bundle")
    if readyz.get("ready") is not True or not isinstance(ready_bundle, dict):
        raise StagingBundleError("Previous /readyz evidence is not a ready bundle response")
    ready_deployed = str(
        ready_bundle.get("deployed_git_sha") or ready_bundle.get("git_sha") or ""
    ).lower()
    expected_ready_fields = {
        "bundle_id": str(payload["bundle_id"]),
        "model_spec_name": str(payload["model_spec_name"]),
        **normalized_runtime,
    }
    if ready_deployed != deployed_git_sha.lower():
        raise StagingBundleError("Previous /readyz deployed SHA does not match caller identity")
    for field, expected in expected_ready_fields.items():
        if str(ready_bundle.get(field) or "").lower() != expected.lower():
            raise StagingBundleError(
                f"Previous /readyz production_bundle.{field} does not match rollback identity"
            )
    artifact_training_sha = str(primary_spec.get("git_hash") or "").lower()
    return {
        "source_manifest": {
            "path": manifest_path.relative_to(repo_root).as_posix(),
            **_file_identity(manifest_path),
            "payload_sha256": _canonical_json_sha256(payload),
            "payload": payload,
        },
        "saved_training_spec": {
            "path": previous_sidecar.relative_to(repo_root).as_posix(),
            **_file_identity(previous_sidecar),
            "payload": previous_saved_spec,
        },
        "local_model_artifacts": local_models,
        "source_manifest_immutable_training_snapshot": {
            "snapshot_max_event_date": payload["snapshot_max_event_date"],
            "processed_fights_sha256": payload["processed_fights_sha256"],
            "processed_features_sha256": payload["processed_features_sha256"],
            "processed_fights_bytes": payload["processed_fights_bytes"],
            "processed_features_bytes": payload["processed_features_bytes"],
        },
        "local_processed_lookup_observation": local_processed,
        "identity_labels": {
            "source_manifest_git_sha": str(payload["git_sha"]),
            "artifact_training_git_sha": artifact_training_sha,
            "deployed_git_sha": deployed_git_sha.lower(),
            "source_manifest_git_sha_is_stale": str(payload["git_sha"]).lower()
            != artifact_training_sha,
        },
        "deployed_git_sha": deployed_git_sha.lower(),
        "runtime_lookup_hashes": dict(sorted(normalized_runtime.items())),
        "readyz_evidence": {
            "source_path": readyz_path.relative_to(repo_root).as_posix(),
            "staged_path": "rollback/previous_readyz.json",
            **_file_identity(readyz_path),
            "payload": readyz,
            "attests_model_hashes": False,
            "model_hash_attestation_limitation": (
                "The legacy live /readyz response does not expose model hashes; local "
                "rollback artifacts are pinned separately."
            ),
        },
    }


def _copy_with_identity(source: Path, destination: Path, expected_sha256: str) -> None:
    copy_file_atomically(source, destination)
    if _sha256_file(destination) != expected_sha256:
        raise StagingBundleError(
            f"Copied artifact hash mismatch: source={source}, destination={destination}"
        )


def _remove_owned_temp(path: Path, *, expected_parent: Path) -> None:
    if not path.exists():
        return
    resolved_parent = path.resolve(strict=False).parent
    if resolved_parent != expected_parent.resolve(strict=True) or not path.name.startswith(
        ".bundle-build-"
    ):
        raise StagingBundleError(f"Refusing to clean unexpected temporary path: {path}")
    shutil.rmtree(path)


def _rich_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise StagingBundleError(f"Staging manifest section {key!r} must be an object")
    return value


def _staged_file(staging_root: Path, relative: object, *, label: str) -> Path:
    raw = Path(str(relative or ""))
    if raw.is_absolute() or not raw.parts or ".." in raw.parts:
        raise StagingBundleError(f"{label} has an unsafe staged path: {relative!r}")
    path = (staging_root / raw).resolve(strict=True)
    try:
        path.relative_to(staging_root)
    except ValueError as exc:
        raise StagingBundleError(f"{label} escapes the staging root") from exc
    if not path.is_file():
        raise StagingBundleError(f"{label} is not a staged file: {path}")
    return path


def _verify_rich_file(path: Path, record: dict[str, Any], *, label: str) -> None:
    if (
        str(record.get("sha256") or "").lower() != _sha256_file(path)
        or record.get("bytes") != int(path.stat().st_size)
    ):
        raise StagingBundleError(f"{label} rich identity does not match its staged file")


def validate_rich_staged_manifest(
    manifest_path: Path,
    *,
    expected_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Read back and verify every schema-v1 rich staging identity."""
    path = manifest_path.resolve(strict=True)
    staging_root = path.parent
    payload = _load_json_object(path, label="Staging manifest")
    if expected_payload is not None and payload != expected_payload:
        raise StagingBundleError("Read-back staging manifest differs from assembled payload")
    if payload.get("staging_schema_version") != 1 or payload.get("manifest_version") != 3:
        raise StagingBundleError("Unsupported or missing rich staging manifest schema")

    expected_core_paths = {
        "model_path": staging_root / "models/xgboost_model.pkl",
        "no_odds_model_path": staging_root / "models/xgboost_no_odds_model.pkl",
        "logistic_model_path": staging_root / "models/logistic_model.pkl",
        "processed_dir": staging_root / "processed",
    }
    for field, expected in expected_core_paths.items():
        if Path(str(payload.get(field) or "")).resolve(strict=True) != expected.resolve(
            strict=True
        ):
            raise StagingBundleError(f"Rich manifest core path {field} is not exact")

    artifacts = _rich_object(payload, "model_artifacts")
    if set(artifacts) != set(MODEL_FILENAMES):
        raise StagingBundleError("Rich manifest must describe exactly all three models")
    core_model_hashes = {
        "primary": "model_sha256",
        "no_odds": "no_odds_model_sha256",
        "logistic": "logistic_model_sha256",
    }
    embedded_specs: dict[str, dict[str, Any]] = {}
    staged_model_results: dict[str, dict[str, Any]] = {}
    used_staged_paths: set[str] = set()
    for label, filename in MODEL_FILENAMES.items():
        record = artifacts[label]
        if not isinstance(record, dict) or record.get("staged_path") != f"models/{filename}":
            raise StagingBundleError(f"Rich {label} model staged path is not exact")
        model_path = _staged_file(staging_root, record["staged_path"], label=label)
        _verify_rich_file(model_path, record, label=label)
        if record["sha256"] != payload.get(core_model_hashes[label]):
            raise StagingBundleError(f"Rich {label} hash disagrees with the core manifest")
        result = _load_model_artifact(model_path, label=f"staged {label}")
        staged_model_results[label] = result
        spec = deepcopy(result["training_spec"])
        embedded_specs[label] = spec
        if (
            record.get("embedded_training_spec") != spec
            or record.get("embedded_training_spec_sha256")
            != _canonical_json_sha256(spec)
            or record.get("feature_count") != len(result["feature_cols"])
        ):
            raise StagingBundleError(f"Rich {label} embedded contract identity is invalid")
        used_staged_paths.add(str(record["staged_path"]))
    if embedded_specs["logistic"] != embedded_specs["primary"] or embedded_specs[
        "no_odds"
    ] != _expected_no_odds_spec_payload(embedded_specs["primary"]):
        raise StagingBundleError("Rich model embedded contracts do not reconcile")
    if payload.get("model_spec_name") != embedded_specs["primary"].get("name") or payload.get(
        "no_odds_model_spec_name"
    ) != embedded_specs["no_odds"].get("name"):
        raise StagingBundleError("Rich model spec names disagree with embedded contracts")

    saved_spec = _rich_object(payload, "saved_fullfit_spec")
    spec_path = _staged_file(staging_root, saved_spec.get("staged_path"), label="saved spec")
    _verify_rich_file(spec_path, saved_spec, label="saved spec")
    if (
        _load_json_object(spec_path, label="Staged saved spec") != embedded_specs["primary"]
        or saved_spec.get("payload") != embedded_specs["primary"]
    ):
        raise StagingBundleError("Rich saved full-fit spec does not reconcile")
    used_staged_paths.add(str(saved_spec["staged_path"]))

    registered = _rich_object(payload, "registered_training_specs")
    for label in ("selected_evaluation", "selected_fullfit"):
        item = registered.get(label)
        if not isinstance(item, dict) or item.get("sha256") != _canonical_json_sha256(
            item.get("payload")
        ):
            raise StagingBundleError(f"Rich registered {label} hash is invalid")
    if (
        registered["selected_evaluation"]["sha256"]
        != APPROVED_EVALUATION_PAYLOAD_SHA256
        or _without_fit_metadata(registered["selected_fullfit"]["payload"])
        != _without_fit_metadata(embedded_specs["primary"])
    ):
        raise StagingBundleError("Rich registered full-fit contract is invalid")
    if registered.get("allowed_differences") != sorted(FULLFIT_ALLOWED_DIFFERENCES):
        raise StagingBundleError("Rich registered spec difference policy is invalid")

    snapshot = _rich_object(payload, "immutable_training_snapshot")
    if snapshot.get("immutable") is not True or snapshot.get(
        "snapshot_max_event_date"
    ) != payload.get("snapshot_max_event_date"):
        raise StagingBundleError("Rich immutable snapshot identity is invalid")
    for label, filename, core_hash, core_bytes in (
        ("fights", "fights_cleaned.csv", "processed_fights_sha256", "processed_fights_bytes"),
        ("features", "features.csv", "processed_features_sha256", "processed_features_bytes"),
    ):
        record = snapshot.get(label)
        if not isinstance(record, dict) or record.get("staged_path") != f"processed/{filename}":
            raise StagingBundleError(f"Rich training {label} path is invalid")
        artifact_path = _staged_file(staging_root, record["staged_path"], label=label)
        actual = _csv_identity(artifact_path)
        if any(record.get(key) != value for key, value in actual.items()):
            raise StagingBundleError(f"Rich training {label} CSV identity is invalid")
        if record["sha256"] != payload.get(core_hash) or record["bytes"] != payload.get(
            core_bytes
        ):
            raise StagingBundleError(f"Rich training {label} disagrees with core fields")
        used_staged_paths.add(str(record["staged_path"]))
    if snapshot["fights"]["max_event_date"] != snapshot["features"]["max_event_date"]:
        raise StagingBundleError("Rich training snapshot dates disagree")
    cutoff = snapshot.get("cutoff_safety")
    if not isinstance(cutoff, dict) or cutoff.get("effective_buffer_days", -1) < cutoff.get(
        "required_minimum_buffer_days", 60
    ):
        raise StagingBundleError("Rich cutoff safety identity is invalid")

    test_record = snapshot.get("test_set")
    if not isinstance(test_record, dict):
        raise StagingBundleError("Rich test-set identity is missing")
    test_path = _staged_file(staging_root, test_record.get("staged_path"), label="test set")
    metadata_path = _staged_file(
        staging_root, test_record.get("metadata_staged_path"), label="test metadata"
    )
    if (
        _sha256_file(test_path) != test_record.get("sha256")
        or int(test_path.stat().st_size) != test_record.get("bytes")
        or _sha256_file(metadata_path) != test_record.get("metadata_sha256")
        or int(metadata_path.stat().st_size) != test_record.get("metadata_bytes")
    ):
        raise StagingBundleError("Rich staged test-set identity is invalid")
    used_staged_paths.update(
        {str(test_record["staged_path"]), str(test_record["metadata_staged_path"])}
    )

    source = _rich_object(payload, "source_identity")
    inventory_records = {
        "pretraining": (
            source.get("pretraining_inventory_artifact"),
            source.get("complete_pretraining_inventory"),
            "provenance/pretraining_model_input_inventory.json",
        ),
        "assembly": (
            source.get("assembly_inventory_artifact"),
            source.get("complete_assembly_inventory"),
            "provenance/assembly_model_input_inventory.json",
        ),
    }
    staged_inventories: dict[str, dict[str, Any]] = {}
    for inventory_label, (record, embedded, expected_staged_path) in inventory_records.items():
        if not isinstance(record, dict) or record.get("staged_path") != expected_staged_path:
            raise StagingBundleError(
                f"Rich {inventory_label} source inventory artifact is missing or misplaced"
            )
        inventory_path = _staged_file(
            staging_root,
            record.get("staged_path"),
            label=f"{inventory_label} source inventory",
        )
        _verify_rich_file(
            inventory_path,
            record,
            label=f"{inventory_label} source inventory",
        )
        inventory_payload = _validate_inventory_payload(
            inventory_path,
            label=f"Staged {inventory_label} source inventory",
        )
        if inventory_payload != embedded:
            raise StagingBundleError(
                f"Rich embedded and copied {inventory_label} inventories disagree"
            )
        staged_inventories[inventory_label] = inventory_payload
        used_staged_paths.add(str(record["staged_path"]))

    inventory_comparison = model_input_inventory.compare_inventories(
        staged_inventories["pretraining"],
        staged_inventories["assembly"],
    )
    observed_delta = {
        "allowlisted_changed_paths": sorted(
            ASSEMBLY_INVENTORY_ALLOWED_CHANGED_PATHS
        ),
        "added": list(inventory_comparison.get("added") or []),
        "removed": list(inventory_comparison.get("removed") or []),
        "changed": list(inventory_comparison.get("changed") or []),
        "git_head_matches": bool(inventory_comparison.get("git_head_matches")),
        "git_diff_hash_matches": bool(inventory_comparison.get("git_diff_matches")),
        "all_raw_inputs_identical": _raw_inventory(staged_inventories["pretraining"])
        == _raw_inventory(staged_inventories["assembly"]),
        "only_allowlisted_assembly_change": True,
        "pretraining_inventory_sha256": staged_inventories["pretraining"][
            "inventory_sha256"
        ],
        "assembly_inventory_sha256": staged_inventories["assembly"][
            "inventory_sha256"
        ],
    }
    if (
        set(observed_delta["changed"]) != ASSEMBLY_INVENTORY_ALLOWED_CHANGED_PATHS
        or observed_delta["added"]
        or observed_delta["removed"]
        or observed_delta["git_head_matches"] is not True
        or observed_delta["all_raw_inputs_identical"] is not True
        or observed_delta != source.get("pretraining_to_assembly_delta")
    ):
        raise StagingBundleError("Rich dual source inventories do not reconcile")

    inventory_rows = staged_inventories["pretraining"]["files"]

    raw = _rich_object(payload, "raw_input_provenance")
    expected_raw_rows = [
        row for row in inventory_rows if row.get("category") == "raw_input"
    ]
    raw_inventory = raw.get("complete_raw_inventory")
    if not isinstance(raw_inventory, dict) or raw_inventory.get("files") != expected_raw_rows or raw_inventory.get(
        "inventory_sha256"
    ) != _aggregate_identities(expected_raw_rows):
        raise StagingBundleError("Rich raw-input inventory does not reconcile")
    ledger = raw.get("bfo_ledger")
    if not isinstance(ledger, dict):
        raise StagingBundleError("Rich BFO provenance ledger identity is missing")
    ledger_path = _staged_file(staging_root, ledger.get("staged_path"), label="BFO ledger")
    _verify_rich_file(ledger_path, ledger, label="BFO ledger")
    parsed_lines = sum(
        1
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and isinstance(json.loads(line), dict)
    )
    if parsed_lines != ledger.get("line_count"):
        raise StagingBundleError("Rich BFO provenance row count is invalid")
    corrected_csvs = ledger.get("corrected_csv_files")
    raw_by_path = {row.get("path"): row for row in expected_raw_rows}
    if (
        ledger.get("sha256") != APPROVED_BFO_LEDGER_SHA256
        or ledger.get("accepted_records") != 234
        or ledger.get("rejected_records") != 10
        or not isinstance(corrected_csvs, list)
        or {Path(str(row.get("path"))).name for row in corrected_csvs}
        != set(APPROVED_BFO_CSVS)
        or any(
            raw_by_path.get(row.get("path"), {}).get("sha256") != row.get("sha256")
            or APPROVED_BFO_CSVS[Path(str(row.get("path"))).name][1]
            != row.get("sha256")
            for row in corrected_csvs
        )
        or ledger.get("corrected_csv_aggregate_sha256")
        != _aggregate_identities(corrected_csvs)
    ):
        raise StagingBundleError("Rich corrected BFO CSV identities do not reconcile")
    used_staged_paths.add(str(ledger["staged_path"]))

    evidence = _rich_object(payload, "selection_evidence")
    evidence_files = evidence.get("files")
    if not isinstance(evidence_files, list) or not evidence_files:
        raise StagingBundleError("Rich selection evidence is missing")
    evidence_identities = []
    for record in evidence_files:
        if not isinstance(record, dict):
            raise StagingBundleError("Rich selection evidence record is invalid")
        evidence_path = _staged_file(
            staging_root, record.get("staged_path"), label="selection evidence"
        )
        _verify_rich_file(evidence_path, record, label="selection evidence")
        evidence_identities.append(
            {"path": record.get("path"), "bytes": record.get("bytes"), "sha256": record.get("sha256")}
        )
        if str(record["staged_path"]) in used_staged_paths:
            raise StagingBundleError("Rich manifest duplicates a staged artifact path")
        used_staged_paths.add(str(record["staged_path"]))
    if (
        evidence.get("file_count") != len(evidence_files)
        or evidence.get("total_bytes") != sum(item["bytes"] for item in evidence_identities)
        or evidence.get("aggregate_sha256") != _aggregate_identities(evidence_identities)
    ):
        raise StagingBundleError("Rich selection evidence aggregate is invalid")

    invocation = _rich_object(payload, "training_invocation")
    if invocation.get("argv_sha256") != _canonical_json_sha256(invocation.get("argv")):
        raise StagingBundleError("Rich training argv identity is invalid")
    audit = invocation.get("independent_audit_snapshot")
    if not isinstance(audit, dict) or audit.get("audit_source_equals_trainer_output") is not False:
        raise StagingBundleError("Rich independent audit identity is invalid")
    controlling = audit.get("controlling_corrected_snapshot")
    if controlling != {
        "fights_sha256": APPROVED_FIGHTS_SHA256,
        "features_sha256": APPROVED_FEATURES_SHA256,
        "append_only_delta_used": False,
    }:
        raise StagingBundleError("Rich controlling corrected snapshot identity is invalid")
    audit_paths: dict[str, Path] = {}
    for audit_label, filename, approved_hash in (
        ("fights", "fights_cleaned.csv", APPROVED_FIGHTS_SHA256),
        ("features", "features.csv", APPROVED_FEATURES_SHA256),
    ):
        record = audit.get(audit_label)
        expected_staged_path = f"provenance/independent_audit_snapshot/{filename}"
        if (
            not isinstance(record, dict)
            or record.get("staged_path") != expected_staged_path
            or record.get("sha256") != approved_hash
        ):
            raise StagingBundleError(
                f"Rich independent audit {audit_label} identity is invalid"
            )
        audit_path = _staged_file(
            staging_root,
            record.get("staged_path"),
            label=f"independent audit {audit_label}",
        )
        _verify_rich_file(
            audit_path,
            record,
            label=f"independent audit {audit_label}",
        )
        audit_paths[audit_label] = audit_path
        used_staged_paths.add(str(record["staged_path"]))

    completed = audit.get("completed_trainer_snapshot")
    expected_completed = {
        "fights_sha256": APPROVED_TRAIN_FIGHTS_SHA256,
        "features_sha256": APPROVED_TRAIN_FEATURES_SHA256,
        "fights_bytes": snapshot["fights"]["bytes"],
        "features_bytes": snapshot["features"]["bytes"],
    }
    if (
        completed != expected_completed
        or payload.get("processed_fights_sha256") != APPROVED_TRAIN_FIGHTS_SHA256
        or payload.get("processed_features_sha256") != APPROVED_TRAIN_FEATURES_SHA256
    ):
        raise StagingBundleError("Rich completed trainer snapshot identity is invalid")

    replay = audit.get("preprocessing_replay")
    if (
        not isinstance(replay, dict)
        or replay.get("preprocessing_replay_byte_match") is not True
        or replay.get("audit_source_equals_trainer_output") is not False
        or replay.get("fights", {}).get("audit_source_sha256")
        != APPROVED_FIGHTS_SHA256
        or replay.get("fights", {}).get("trainer_output_sha256")
        != APPROVED_TRAIN_FIGHTS_SHA256
        or replay.get("fights", {}).get("replay_output_sha256")
        != APPROVED_TRAIN_FIGHTS_SHA256
        or replay.get("fights", {}).get("replay_output_bytes")
        != snapshot["fights"]["bytes"]
        or replay.get("features", {}).get("trainer_output_sha256")
        != APPROVED_TRAIN_FEATURES_SHA256
        or replay.get("features", {}).get("replay_output_sha256")
        != APPROVED_TRAIN_FEATURES_SHA256
        or replay.get("features", {}).get("replay_output_bytes")
        != snapshot["features"]["bytes"]
        or replay.get("fights", {}).get("byte_match") is not True
        or replay.get("features", {}).get("byte_match") is not True
    ):
        raise StagingBundleError("Rich preprocessing replay receipt is invalid")

    observed_relationship = _audit_trainer_relationship(
        audit_fights_path=audit_paths["fights"],
        audit_features_path=audit_paths["features"],
        trainer_fights_path=staging_root / "processed/fights_cleaned.csv",
        trainer_features_path=staging_root / "processed/features.csv",
        primary_spec=embedded_specs["primary"],
        model_results=staged_model_results,
    )
    if observed_relationship != audit.get("semantic_equivalence"):
        raise StagingBundleError(
            "Rich audit/trainer semantic or prediction diagnostics do not reconcile"
        )

    environment = _rich_object(payload, "assembly_validation_environment")
    if environment.get("provenance_level") != "deterministic_preprocessing_replay_same_interpreter" or set(
        (environment.get("packages") or {}).keys()
    ) != {"numpy", "pandas", "scikit-learn", "xgboost", "joblib"}:
        raise StagingBundleError("Rich assembly validation environment is incomplete")
    inference = _rich_object(payload, "finite_inference")
    sample_rows = inference.get("sample_rows")
    if not isinstance(sample_rows, int) or sample_rows < 1:
        raise StagingBundleError("Rich finite-inference sample identity is invalid")
    for label in MODEL_FILENAMES:
        result = inference.get(label)
        if not isinstance(result, dict) or result.get("finite_probability_count") != sample_rows:
            raise StagingBundleError(f"Rich finite inference is incomplete for {label}")

    rollback = _rich_object(payload, "previous_rollback_identity")
    source_manifest = rollback.get("source_manifest")
    readyz = rollback.get("readyz_evidence")
    if not isinstance(source_manifest, dict) or source_manifest.get(
        "payload_sha256"
    ) != _canonical_json_sha256(source_manifest.get("payload")) or not isinstance(readyz, dict):
        raise StagingBundleError("Rich previous rollback identity is invalid")
    old_manifest_payload = source_manifest["payload"]
    expected_old_snapshot = {
        "snapshot_max_event_date": old_manifest_payload.get("snapshot_max_event_date"),
        "processed_fights_sha256": old_manifest_payload.get("processed_fights_sha256"),
        "processed_features_sha256": old_manifest_payload.get("processed_features_sha256"),
        "processed_fights_bytes": old_manifest_payload.get("processed_fights_bytes"),
        "processed_features_bytes": old_manifest_payload.get("processed_features_bytes"),
    }
    local_lookup = rollback.get("local_processed_lookup_observation")
    if (
        rollback.get("source_manifest_immutable_training_snapshot")
        != expected_old_snapshot
        or not isinstance(local_lookup, dict)
        or local_lookup.get("mutable_lookup") is not True
        or "local_immutable_processed_snapshot" in rollback
    ):
        raise StagingBundleError(
            "Rich rollback identity mixes immutable training and mutable lookup data"
        )
    readyz_path = _staged_file(
        staging_root, readyz.get("staged_path"), label="previous readyz evidence"
    )
    _verify_rich_file(readyz_path, readyz, label="previous readyz evidence")
    if _load_json_object(readyz_path, label="Staged readyz evidence") != readyz.get("payload"):
        raise StagingBundleError("Rich copied /readyz evidence payload is invalid")
    used_staged_paths.add(str(readyz["staged_path"]))
    runtime_hashes = rollback.get("runtime_lookup_hashes")
    ready_bundle = readyz["payload"].get("production_bundle")
    if not isinstance(runtime_hashes, dict) or not isinstance(ready_bundle, dict) or any(
        ready_bundle.get(key) != value for key, value in runtime_hashes.items()
    ):
        raise StagingBundleError("Rich mutable runtime lookup hashes do not reconcile")

    return {
        "manifest_path": str(path),
        "manifest_sha256": _sha256_file(path),
        "verified_staged_file_count": len(used_staged_paths),
        "pretraining_source_inventory_sha256": staged_inventories["pretraining"][
            "inventory_sha256"
        ],
        "assembly_source_inventory_sha256": staged_inventories["assembly"][
            "inventory_sha256"
        ],
        "selection_evidence_sha256": evidence["aggregate_sha256"],
        "rollback_readyz_sha256": readyz["sha256"],
    }


def assemble_staged_bundle(
    inputs: BundleInputs,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Build one new staging root and return its strict validation summary."""
    root = repo_root.resolve(strict=True)
    staging_root = _repo_path(
        inputs.staging_root,
        repo_root=root,
        label="Staging root",
        must_exist=False,
    )
    if staging_root == root or staging_root.exists():
        raise StagingBundleError(
            f"Staging root must be a new, non-repository directory: {staging_root}"
        )
    allowed_staging_roots = [
        (root / ".codex_stage").resolve(strict=False),
        (root / "logs").resolve(strict=False),
    ]
    if not any(
        staging_root != namespace and namespace in staging_root.parents
        for namespace in allowed_staging_roots
    ):
        raise StagingBundleError(
            "Staging root must be strictly below .codex_stage/ or logs/"
        )
    if not staging_root.parent.is_dir():
        raise StagingBundleError(
            f"Staging root parent must already exist: {staging_root.parent}"
        )

    candidate_models = _strict_candidate_dir(
        inputs.candidate_models_dir,
        repo_root=root,
        allowed_root=root / "models" / "candidates",
        label="Candidate models directory",
    )
    candidate_processed = _strict_candidate_dir(
        inputs.candidate_processed_dir,
        repo_root=root,
        allowed_root=root / "data" / "processed" / "candidates",
        label="Candidate processed directory",
    )
    if candidate_models == (root / "models").resolve(strict=False):
        raise StagingBundleError("Canonical models directory cannot be staged from")
    if candidate_processed == (root / "data" / "processed").resolve(strict=False):
        raise StagingBundleError("Canonical processed directory cannot be staged from")
    for source_dir in (candidate_models, candidate_processed):
        if staging_root == source_dir or source_dir in staging_root.parents:
            raise StagingBundleError("Staging root must be isolated from candidate sources")

    model_paths = {
        label: _exact_child_file(candidate_models, filename, label=f"candidate {label}")
        for label, filename in MODEL_FILENAMES.items()
    }
    processed_paths = {
        filename: _exact_child_file(
            candidate_processed, filename, label=f"candidate {filename}"
        )
        for filename in PROCESSED_FILENAMES
    }

    primary_result = _load_model_artifact(model_paths["primary"], label="primary")
    fullfit_name = str(primary_result["training_spec"].get("name") or "").strip()
    sidecar_path = _exact_child_file(
        candidate_models,
        f"{fullfit_name}_spec.json",
        label="candidate full-fit spec sidecar",
    )
    pretraining_inventory_path = _existing_file(
        inputs.input_inventory_path,
        repo_root=root,
        label="Pretraining model input inventory",
    )
    assembly_inventory_path = _existing_file(
        inputs.assembly_inventory_path,
        repo_root=root,
        label="Assembly model input inventory",
    )
    (
        pretraining_inventory,
        assembly_inventory,
        raw_inventory,
        inventory_delta,
    ) = _validate_input_inventories(
        pretraining_inventory_path,
        assembly_inventory_path,
        repo_root=root,
    )
    registered_eval, registered_fullfit, model_results = _validate_contracts(
        model_paths=model_paths,
        sidecar_path=sidecar_path,
        evaluation_spec_name=inputs.evaluation_spec_name,
        expected_git_head=str(pretraining_inventory["git_head"]),
    )
    bfo_path = _existing_file(
        inputs.bfo_provenance_path,
        repo_root=root,
        label="BFO provenance ledger",
    )
    bfo_identity = _validate_bfo_ledger(
        bfo_path, repo_root=root, inventory_payload=pretraining_inventory
    )
    evidence_rows, evidence_total = _selection_evidence(
        inputs.selection_evidence_paths, repo_root=root
    )
    previous_manifest = _existing_file(
        inputs.previous_manifest_path,
        repo_root=root,
        label="Previous source manifest",
    )
    previous_readyz = _existing_file(
        inputs.previous_readyz_path,
        repo_root=root,
        label="Previous /readyz evidence",
    )
    rollback = _validate_previous_rollback(
        manifest_path=previous_manifest,
        readyz_path=previous_readyz,
        repo_root=root,
        deployed_git_sha=inputs.previous_deployed_git_sha,
        runtime_lookup_hashes=inputs.previous_runtime_lookup_hashes,
    )
    invocation = _validate_training_argv(
        inputs.training_argv,
        repo_root=root,
        fullfit_spec_name=fullfit_name,
        candidate_models_dir=candidate_models,
        candidate_processed_dir=candidate_processed,
        expected_fights_sha256=inputs.expected_fights_sha256,
        expected_features_sha256=inputs.expected_features_sha256,
    )
    audit_snapshot = invocation["independent_audit_snapshot"]
    audit_fights_path = root / str(audit_snapshot["fights"]["path"])
    audit_features_path = root / str(audit_snapshot["features"]["path"])
    relationship = _audit_trainer_relationship(
        audit_fights_path=audit_fights_path,
        audit_features_path=audit_features_path,
        trainer_fights_path=processed_paths["fights_cleaned.csv"],
        trainer_features_path=processed_paths["features.csv"],
        primary_spec=model_results["primary"]["training_spec"],
        model_results=model_results,
    )
    replay = _replay_trainer_preprocessing(
        audit_fights_path=audit_fights_path,
        trainer_fights_path=processed_paths["fights_cleaned.csv"],
        trainer_features_path=processed_paths["features.csv"],
        fullfit_spec_name=fullfit_name,
    )
    audit_snapshot["semantic_equivalence"] = relationship
    audit_snapshot["preprocessing_replay"] = replay

    fights_identity = _csv_identity(processed_paths["fights_cleaned.csv"])
    features_identity = _csv_identity(processed_paths["features.csv"])
    if not fights_identity["max_event_date"] or not features_identity["max_event_date"]:
        raise StagingBundleError("Training fights/features snapshots must not be empty")
    if fights_identity["max_event_date"] != features_identity["max_event_date"]:
        raise StagingBundleError(
            "Training fights/features snapshots must have the same maximum event date"
        )
    snapshot_max_date = str(fights_identity["max_event_date"])
    try:
        cutoff_date = datetime.fromisoformat(
            str(registered_fullfit["train_cutoff_date"])
        ).date()
        snapshot_date = datetime.fromisoformat(snapshot_max_date).date()
    except (KeyError, ValueError) as exc:
        raise StagingBundleError("Training cutoff/snapshot dates must be valid ISO dates") from exc
    current_date = datetime.now(timezone.utc).date()
    snapshot_buffer_days = (cutoff_date - snapshot_date).days
    current_buffer_days = (cutoff_date - current_date).days
    required_buffer_days = min(snapshot_buffer_days, current_buffer_days)
    if required_buffer_days < 60:
        raise StagingBundleError(
            "Full-fit training cutoff has less than the required 60-day safety buffer"
        )
    inference_summary, reconstructed_training, reconstructed_test = _finite_inference(
        features_path=processed_paths["features.csv"],
        primary_spec=model_results["primary"]["training_spec"],
        model_results=model_results,
        sample_rows=inputs.inference_sample_rows,
    )
    test_identity = _validate_test_set_metadata(
        test_set_path=processed_paths["test_set.csv"],
        metadata_path=processed_paths["test_set.csv.metadata.json"],
        primary_spec=model_results["primary"]["training_spec"],
        expected_test_frame=reconstructed_test,
    )
    eligible_training_rows = len(reconstructed_training)

    model_identities = {
        label: {
            "source_path": path.relative_to(root).as_posix(),
            "staged_path": f"models/{MODEL_FILENAMES[label]}",
            **_file_identity(path),
            "embedded_training_spec": deepcopy(
                model_results[label]["training_spec"]
            ),
            "embedded_training_spec_sha256": _canonical_json_sha256(
                model_results[label]["training_spec"]
            ),
            "feature_count": len(model_results[label]["feature_cols"]),
        }
        for label, path in model_paths.items()
    }
    built_at = datetime.now(timezone.utc).isoformat()
    bundle_id = inputs.bundle_id or f"ufc-production-{snapshot_max_date.replace('-', '')}-{fullfit_name}"
    if not bundle_id.strip() or any(character.isspace() for character in bundle_id):
        raise StagingBundleError("bundle_id must be nonempty and contain no whitespace")

    final_models = staging_root / "models"
    final_processed = staging_root / "processed"
    manifest_path = staging_root / "staging_manifest.json"
    manifest: dict[str, object] = {
        "manifest_version": 3,
        "staging_schema_version": 1,
        "bundle_id": bundle_id,
        "model_spec_name": fullfit_name,
        "no_odds_model_spec_name": f"{fullfit_name}_no_odds",
        "model_path": str(final_models / MODEL_FILENAMES["primary"]),
        "no_odds_model_path": str(final_models / MODEL_FILENAMES["no_odds"]),
        "logistic_model_path": str(final_models / MODEL_FILENAMES["logistic"]),
        "processed_dir": str(final_processed),
        "snapshot_max_event_date": snapshot_max_date,
        "built_at": built_at,
        "manifest_updated_at": built_at,
        "git_sha": str(pretraining_inventory["git_head"]),
        "training_source_git_sha": str(pretraining_inventory["git_head"]),
        "model_sha256": model_identities["primary"]["sha256"],
        "no_odds_model_sha256": model_identities["no_odds"]["sha256"],
        "logistic_model_sha256": model_identities["logistic"]["sha256"],
        "processed_fights_sha256": fights_identity["sha256"],
        "processed_features_sha256": features_identity["sha256"],
        "processed_fights_bytes": fights_identity["bytes"],
        "processed_features_bytes": features_identity["bytes"],
        "source_identity": {
            "base_git_sha": pretraining_inventory["git_head"],
            "tracked_diff_sha256": pretraining_inventory["git_diff_sha256"],
            "pre_training_dirty_status": {
                "git_dirty": pretraining_inventory.get("git_dirty"),
                "git_status_sha256": pretraining_inventory.get("git_status_sha256"),
                "git_status": pretraining_inventory.get("git_status"),
            },
            "assembly_dirty_status": {
                "git_dirty": assembly_inventory.get("git_dirty"),
                "git_status_sha256": assembly_inventory.get("git_status_sha256"),
                "git_status": assembly_inventory.get("git_status"),
            },
            "pretraining_inventory_artifact": {
                "role": "frozen_model_and_raw_input_provenance",
                "source_path": pretraining_inventory_path.relative_to(root).as_posix(),
                "staged_path": "provenance/pretraining_model_input_inventory.json",
                **_file_identity(pretraining_inventory_path),
            },
            "assembly_inventory_artifact": {
                "role": "current_assembly_source_and_raw_input_identity",
                "source_path": assembly_inventory_path.relative_to(root).as_posix(),
                "staged_path": "provenance/assembly_model_input_inventory.json",
                **_file_identity(assembly_inventory_path),
            },
            "complete_pretraining_inventory": pretraining_inventory,
            "complete_assembly_inventory": assembly_inventory,
            "pretraining_to_assembly_delta": inventory_delta,
        },
        "registered_training_specs": {
            "selected_evaluation": {
                "payload": registered_eval,
                "sha256": _canonical_json_sha256(registered_eval),
            },
            "selected_fullfit": {
                "payload": registered_fullfit,
                "sha256": _canonical_json_sha256(registered_fullfit),
            },
            "allowed_differences": sorted(FULLFIT_ALLOWED_DIFFERENCES),
        },
        "model_artifacts": model_identities,
        "saved_fullfit_spec": {
            "source_path": sidecar_path.relative_to(root).as_posix(),
            "staged_path": f"models/{sidecar_path.name}",
            **_file_identity(sidecar_path),
            "payload": model_results["primary"]["training_spec"],
        },
        "immutable_training_snapshot": {
            "immutable": True,
            "snapshot_max_event_date": snapshot_max_date,
            "eligible_training_rows": eligible_training_rows,
            "cutoff_safety": {
                "exclusive_train_cutoff_date": cutoff_date.isoformat(),
                "validation_current_utc_date": current_date.isoformat(),
                "snapshot_buffer_days": snapshot_buffer_days,
                "current_date_buffer_days": current_buffer_days,
                "required_minimum_buffer_days": 60,
                "effective_buffer_days": required_buffer_days,
            },
            "fights": {
                "source_path": processed_paths["fights_cleaned.csv"].relative_to(root).as_posix(),
                "staged_path": "processed/fights_cleaned.csv",
                **fights_identity,
            },
            "features": {
                "source_path": processed_paths["features.csv"].relative_to(root).as_posix(),
                "staged_path": "processed/features.csv",
                **features_identity,
            },
            "test_set": {
                "source_path": processed_paths["test_set.csv"].relative_to(root).as_posix(),
                "staged_path": "processed/test_set.csv",
                "metadata_staged_path": "processed/test_set.csv.metadata.json",
                **test_identity,
            },
        },
        "raw_input_provenance": {
            "complete_raw_inventory": raw_inventory,
            "bfo_ledger": {
                **bfo_identity,
                "staged_path": f"provenance/{bfo_path.relative_to(root).as_posix()}",
            },
        },
        "selection_evidence": {
            "aggregate_sha256": _aggregate_identities(evidence_rows),
            "file_count": len(evidence_rows),
            "total_bytes": evidence_total,
            "files": [
                {
                    **row,
                    "source_path": row["path"],
                    "staged_path": f"evidence/{row['path']}",
                }
                for row in evidence_rows
            ],
        },
        "training_invocation": invocation,
        "assembly_validation_environment": {
            **_package_versions(),
            "provenance_level": "deterministic_preprocessing_replay_same_interpreter",
            "statement": (
                "The assembler replayed the trainer's exact load/build/save preprocessing "
                "path with this interpreter and required byte-identical train outputs."
            ),
        },
        "finite_inference": inference_summary,
        "previous_rollback_identity": rollback,
    }

    temp_root = Path(
        tempfile.mkdtemp(prefix=".bundle-build-", dir=staging_root.parent)
    ).resolve(strict=True)
    try:
        temp_models = temp_root / "models"
        temp_processed = temp_root / "processed"
        for label, source in model_paths.items():
            identity = model_identities[label]
            destination = temp_models / MODEL_FILENAMES[label]
            _copy_with_identity(source, destination, str(identity["sha256"]))
        _copy_with_identity(
            sidecar_path,
            temp_models / sidecar_path.name,
            _sha256_file(sidecar_path),
        )
        for filename, source in processed_paths.items():
            _copy_with_identity(
                source, temp_processed / filename, _sha256_file(source)
            )
        _copy_with_identity(
            pretraining_inventory_path,
            temp_root / "provenance" / "pretraining_model_input_inventory.json",
            _sha256_file(pretraining_inventory_path),
        )
        _copy_with_identity(
            assembly_inventory_path,
            temp_root / "provenance" / "assembly_model_input_inventory.json",
            _sha256_file(assembly_inventory_path),
        )
        _copy_with_identity(
            audit_fights_path,
            temp_root / "provenance" / "independent_audit_snapshot" / "fights_cleaned.csv",
            _sha256_file(audit_fights_path),
        )
        _copy_with_identity(
            audit_features_path,
            temp_root / "provenance" / "independent_audit_snapshot" / "features.csv",
            _sha256_file(audit_features_path),
        )
        _copy_with_identity(
            bfo_path,
            temp_root / "provenance" / bfo_path.relative_to(root),
            _sha256_file(bfo_path),
        )
        for row in evidence_rows:
            source = root / str(row["path"])
            _copy_with_identity(
                source,
                temp_root / "evidence" / str(row["path"]),
                str(row["sha256"]),
            )
        _copy_with_identity(
            previous_readyz,
            temp_root / "rollback" / "previous_readyz.json",
            _sha256_file(previous_readyz),
        )
        write_json_atomically(manifest, temp_root / "staging_manifest.json")

        if staging_root.exists():
            raise StagingBundleError(
                f"Staging root appeared during assembly; refusing overwrite: {staging_root}"
            )
        os.rename(temp_root, staging_root)
    except Exception:
        _remove_owned_temp(temp_root, expected_parent=staging_root.parent)
        raise

    try:
        rich_validation = validate_rich_staged_manifest(
            manifest_path, expected_payload=manifest
        )
        bundle = load_production_bundle(manifest_path)
        validation = validate_production_bundle(
            bundle,
            expected_models_dir=final_models,
            expected_processed_dir=final_processed,
        )
    except (ProductionBundleError, Exception) as exc:
        # The final directory is wholly owned by this failed invocation and did
        # not exist before it.  Removing it preserves the fail-closed promise.
        try:
            shutil.rmtree(staging_root)
        except OSError as cleanup_exc:
            raise StagingBundleError(
                f"Staged validation failed ({exc}) and cleanup failed ({cleanup_exc})"
            ) from exc
        raise StagingBundleError(f"Strict staged validation failed: {exc}") from exc

    return {
        "staging_root": str(staging_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "bundle_id": bundle_id,
        "model_spec_name": fullfit_name,
        "snapshot_max_event_date": snapshot_max_date,
        "validation": validation,
        "rich_validation": rich_validation,
        "finite_inference": inference_summary,
    }


def _parse_runtime_hash(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("runtime lookup hashes must use NAME=SHA256")
    key, digest = value.split("=", 1)
    if not RUNTIME_HASH_KEY_RE.fullmatch(key) or not SHA256_RE.fullmatch(digest):
        raise argparse.ArgumentTypeError(f"invalid runtime lookup hash: {value!r}")
    return key, digest.lower()


def _parse_training_argv(value: str) -> tuple[str, ...]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"training argv must be a JSON string array: {exc}"
        ) from exc
    if not isinstance(payload, list) or not payload or not all(
        isinstance(token, str) and token for token in payload
    ):
        raise argparse.ArgumentTypeError("training argv must be a nonempty JSON string array")
    return tuple(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--candidate-models-dir", type=Path, required=True)
    parser.add_argument("--candidate-processed-dir", type=Path, required=True)
    parser.add_argument("--evaluation-spec", required=True)
    parser.add_argument(
        "--input-inventory",
        type=Path,
        required=True,
        help="Frozen pretraining model/source/raw-input inventory.",
    )
    parser.add_argument("--assembly-inventory", type=Path, required=True)
    parser.add_argument("--bfo-provenance", type=Path, required=True)
    parser.add_argument(
        "--selection-evidence", type=Path, action="append", required=True
    )
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--previous-readyz", type=Path, required=True)
    parser.add_argument("--previous-deployed-git-sha", required=True)
    parser.add_argument("--expected-fights-sha256", required=True)
    parser.add_argument("--expected-features-sha256", required=True)
    parser.add_argument(
        "--previous-runtime-lookup-hash",
        action="append",
        type=_parse_runtime_hash,
        required=True,
        metavar="NAME=SHA256",
    )
    parser.add_argument(
        "--training-argv-json",
        type=_parse_training_argv,
        required=True,
        help="Exact training process argv as a JSON string array, including Python.",
    )
    parser.add_argument("--bundle-id")
    parser.add_argument("--inference-sample-rows", type=int, default=32)
    args = parser.parse_args(argv)

    runtime_hashes: dict[str, str] = {}
    for key, digest in args.previous_runtime_lookup_hash:
        if key in runtime_hashes:
            parser.error(f"duplicate previous runtime lookup hash key: {key}")
        runtime_hashes[key] = digest

    inputs = BundleInputs(
        staging_root=args.staging_root,
        candidate_models_dir=args.candidate_models_dir,
        candidate_processed_dir=args.candidate_processed_dir,
        evaluation_spec_name=args.evaluation_spec,
        input_inventory_path=args.input_inventory,
        assembly_inventory_path=args.assembly_inventory,
        bfo_provenance_path=args.bfo_provenance,
        selection_evidence_paths=tuple(args.selection_evidence),
        previous_manifest_path=args.previous_manifest,
        previous_readyz_path=args.previous_readyz,
        previous_deployed_git_sha=args.previous_deployed_git_sha,
        previous_runtime_lookup_hashes=runtime_hashes,
        expected_fights_sha256=args.expected_fights_sha256,
        expected_features_sha256=args.expected_features_sha256,
        training_argv=args.training_argv_json,
        bundle_id=args.bundle_id,
        inference_sample_rows=args.inference_sample_rows,
    )
    try:
        result = assemble_staged_bundle(inputs)
    except StagingBundleError as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
