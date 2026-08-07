"""Validate and restore scheduled BFO carry-forward artifacts.

Each promoted bundle seals the complete list of accepted scheduled BFO batches.
The next clean checkout downloads that small package, verifies it against the
active bundle's readiness identity, and restores the original raw CSV/ledger
pairs before rebuilding processed data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from src.data.io_utils import copy_file_atomically, write_json_atomically


SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CSV_NAME_RE = re.compile(r"^historical_odds_bfo_recovered_[A-Za-z0-9._-]+\.csv$")
BUNDLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
RAW_PREFIX = PurePosixPath("data/raw/historical_odds")


class BfoLineageError(RuntimeError):
    """Raised when scheduled BFO lineage cannot be trusted or restored."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def manifest_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(manifest_bytes(payload)).hexdigest()


def write_manifest(payload: dict[str, Any], path: Path) -> Path:
    validate_manifest_payload(payload)
    target = path.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(manifest_bytes(payload))
    return target


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BfoLineageError(f"duplicate JSON key in BFO lineage: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BfoLineageError(f"cannot read BFO lineage manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BfoLineageError("BFO lineage manifest must be a JSON object")
    return payload


def _require_exact_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise BfoLineageError(f"{label} has an invalid schema")
    return value


def _require_nonnegative_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BfoLineageError(f"{label} must be an integer >= {minimum}")
    return value


def _safe_raw_path(value: object, *, suffix: str, label: str) -> PurePosixPath:
    text = str(value or "")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.parent != RAW_PREFIX:
        raise BfoLineageError(f"{label} must stay directly under {RAW_PREFIX}")
    if not path.name.endswith(suffix):
        raise BfoLineageError(f"{label} has the wrong suffix")
    return path


def _safe_artifact_path(value: object, *, expected_name: str, label: str) -> PurePosixPath:
    path = PurePosixPath(str(value or ""))
    if path.is_absolute() or ".." in path.parts or path != PurePosixPath("batches") / expected_name:
        raise BfoLineageError(f"{label} is not the exact package path")
    return path


def validate_manifest_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = _require_exact_keys(
        payload,
        {
            "schema_version",
            "parent_bundle_id",
            "parent_source_manifest_sha256",
            "previous_lineage_manifest_sha256",
            "batches",
        },
        "BFO lineage manifest",
    )
    if manifest["schema_version"] != SCHEMA_VERSION or not isinstance(manifest["batches"], list):
        raise BfoLineageError("BFO lineage manifest version or batches are invalid")
    if not BUNDLE_ID_RE.fullmatch(str(manifest["parent_bundle_id"] or "")):
        raise BfoLineageError("BFO lineage parent bundle id is invalid")
    if not SHA256_RE.fullmatch(str(manifest["parent_source_manifest_sha256"] or "")):
        raise BfoLineageError("BFO lineage parent source manifest is invalid")
    previous_sha = manifest["previous_lineage_manifest_sha256"]
    if previous_sha is not None and not SHA256_RE.fullmatch(str(previous_sha or "")):
        raise BfoLineageError("BFO lineage previous manifest identity is invalid")

    seen_raw: set[str] = set()
    seen_artifact: set[str] = set()
    batches: list[dict[str, Any]] = []
    for index, item in enumerate(manifest["batches"]):
        batch = _require_exact_keys(
            item,
            {"accepted_records", "rejected_records", "csv", "provenance"},
            f"BFO lineage batch {index}",
        )
        accepted = _require_nonnegative_int(
            batch["accepted_records"], f"BFO lineage batch {index} accepted_records", minimum=1
        )
        rejected = _require_nonnegative_int(
            batch["rejected_records"], f"BFO lineage batch {index} rejected_records"
        )
        csv_record = _require_exact_keys(
            batch["csv"], {"raw_path", "artifact_path", "sha256", "bytes", "rows"},
            f"BFO lineage batch {index} csv",
        )
        provenance = _require_exact_keys(
            batch["provenance"],
            {"raw_path", "artifact_path", "sha256", "bytes", "line_count"},
            f"BFO lineage batch {index} provenance",
        )
        csv_raw = _safe_raw_path(csv_record["raw_path"], suffix=".csv", label="BFO lineage CSV")
        if not CSV_NAME_RE.fullmatch(csv_raw.name):
            raise BfoLineageError("BFO lineage CSV filename is invalid")
        ledger_name = f"{csv_raw.name[:-4]}.provenance.jsonl"
        ledger_raw = _safe_raw_path(
            provenance["raw_path"], suffix=".provenance.jsonl", label="BFO lineage ledger"
        )
        if ledger_raw.name != ledger_name:
            raise BfoLineageError("BFO lineage CSV and ledger are not companions")
        csv_artifact = _safe_artifact_path(
            csv_record["artifact_path"], expected_name=csv_raw.name, label="BFO lineage CSV artifact"
        )
        ledger_artifact = _safe_artifact_path(
            provenance["artifact_path"], expected_name=ledger_name, label="BFO lineage ledger artifact"
        )
        for record, count_key, expected_count, record_label in (
            (csv_record, "rows", accepted, "CSV"),
            (provenance, "line_count", accepted + rejected, "ledger"),
        ):
            if not SHA256_RE.fullmatch(str(record["sha256"] or "")):
                raise BfoLineageError(f"BFO lineage {record_label} SHA-256 is invalid")
            _require_nonnegative_int(record["bytes"], f"BFO lineage {record_label} bytes", minimum=1)
            if _require_nonnegative_int(record[count_key], f"BFO lineage {record_label} {count_key}") != expected_count:
                raise BfoLineageError(f"BFO lineage {record_label} count does not reconcile")
        for path in (csv_raw.as_posix(), ledger_raw.as_posix()):
            if path in seen_raw:
                raise BfoLineageError(f"duplicate BFO lineage raw path: {path}")
            seen_raw.add(path)
        for path in (csv_artifact.as_posix(), ledger_artifact.as_posix()):
            if path in seen_artifact:
                raise BfoLineageError(f"duplicate BFO lineage artifact path: {path}")
            seen_artifact.add(path)
        batches.append(batch)
    return batches


def validate_package(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    path = manifest_path.resolve(strict=True)
    if expected_manifest_sha256 is not None:
        expected = str(expected_manifest_sha256).lower()
        if not SHA256_RE.fullmatch(expected) or sha256_file(path) != expected:
            raise BfoLineageError("BFO lineage manifest does not match active production")
    payload = _load_json_object(path)
    batches = validate_manifest_payload(payload)
    package_root = path.parent
    for index, batch in enumerate(batches):
        for label, record in (("CSV", batch["csv"]), ("ledger", batch["provenance"])):
            artifact = (package_root / str(record["artifact_path"])).resolve(strict=True)
            try:
                artifact.relative_to(package_root)
            except ValueError as exc:
                raise BfoLineageError(f"BFO lineage {label} escapes its package") from exc
            if (
                not artifact.is_file()
                or artifact.stat().st_size != record["bytes"]
                or sha256_file(artifact) != record["sha256"]
            ):
                raise BfoLineageError(f"BFO lineage batch {index} {label} identity is invalid")
        csv_path = package_root / str(batch["csv"]["artifact_path"])
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            if sum(1 for _row in csv.DictReader(handle)) != batch["csv"]["rows"]:
                raise BfoLineageError(f"BFO lineage batch {index} CSV row count is invalid")
        ledger_path = package_root / str(batch["provenance"]["artifact_path"])
        lines = [line for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != batch["provenance"]["line_count"]:
            raise BfoLineageError(f"BFO lineage batch {index} ledger line count is invalid")
        for line in lines:
            try:
                if not isinstance(json.loads(line), dict):
                    raise ValueError("not an object")
            except (json.JSONDecodeError, ValueError) as exc:
                raise BfoLineageError(f"BFO lineage batch {index} ledger is invalid") from exc
    return payload


def restore_package(
    manifest_path: Path,
    *,
    repo_root: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    payload = validate_package(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    root = repo_root.resolve(strict=True)
    package_root = manifest_path.resolve(strict=True).parent
    restored: list[str] = []
    for batch in payload["batches"]:
        for record in (batch["csv"], batch["provenance"]):
            source = package_root / str(record["artifact_path"])
            destination = (root / str(record["raw_path"])).resolve(strict=False)
            try:
                destination.relative_to(root / RAW_PREFIX.as_posix())
            except ValueError as exc:
                raise BfoLineageError("BFO lineage restore target escapes raw odds") from exc
            if destination.exists():
                if not destination.is_file() or sha256_file(destination) != record["sha256"]:
                    raise BfoLineageError(f"refusing to overwrite different BFO lineage file: {destination}")
            else:
                copy_file_atomically(source, destination)
            restored.append(str(record["raw_path"]))
    return {"manifest": payload, "restored_paths": restored}


def _load_readyz(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path.resolve(strict=True))
    bundle = payload.get("production_bundle")
    if payload.get("ready") is not True or not isinstance(bundle, dict):
        raise BfoLineageError("parent readiness evidence is not ready production")
    return bundle


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--parent-readyz", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    report: dict[str, Any] = {"schema_version": 1, "status": "error"}
    try:
        bundle = _load_readyz(args.parent_readyz)
        policy = _load_json_object(args.policy.resolve(strict=True))
        root_source = str(policy.get("root_release", {}).get("source_manifest_sha256") or "")
        parent_source = str(bundle.get("source_manifest_sha256") or "")
        expected_lineage = str(bundle.get("scheduled_bfo_lineage_manifest_sha256") or "")
        if not SHA256_RE.fullmatch(parent_source):
            raise BfoLineageError("parent source manifest identity is missing")
        report["parent_source_manifest_sha256"] = parent_source
        if expected_lineage:
            if args.artifact_root is None:
                raise BfoLineageError("active production requires a BFO lineage artifact")
            manifest_path = args.artifact_root.resolve(strict=True) / MANIFEST_NAME
            restored = restore_package(
                manifest_path,
                repo_root=args.repo_root,
                expected_manifest_sha256=expected_lineage,
            )
            report.update(
                {
                    "status": "restored",
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": expected_lineage,
                    "batch_count": len(restored["manifest"]["batches"]),
                    "restored_paths": restored["restored_paths"],
                }
            )
        else:
            if parent_source != root_source:
                raise BfoLineageError(
                    "non-root active production is missing its BFO lineage identity"
                )
            if args.artifact_root is not None:
                raise BfoLineageError("root baseline must not supply an unbound BFO artifact")
            report.update({"status": "root_baseline", "batch_count": 0, "restored_paths": []})
        write_json_atomically(report, args.report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (BfoLineageError, OSError, ValueError) as exc:
        report["error"] = str(exc)
        write_json_atomically(report, args.report)
        print(f"BFO LINEAGE RESTORE FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
