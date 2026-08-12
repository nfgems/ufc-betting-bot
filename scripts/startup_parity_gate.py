"""Run and cache manifest-bound live/train parity before hosted readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.io_utils import write_json_atomically
from src.model.production_bundle import load_production_bundle, validate_production_bundle
from src.model.training_spec import resolve_named_training_spec


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_receipt(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _require_same_release(source: dict, runtime: dict) -> None:
    fields = (
        "bundle_id",
        "rich_release_id",
        "source_manifest_sha256",
        "installed_manifest_sha256",
        "model_spec_name",
        "model_sha256",
        "no_odds_model_sha256",
        "logistic_model_sha256",
        "immutable_training_fights_sha256",
        "immutable_training_features_sha256",
        "scheduled_refit_policy",
        "confirmed_strategy",
    )
    mismatches = [
        field for field in fields if source.get(field) != runtime.get(field)
    ]
    if mismatches:
        raise RuntimeError(
            "Runtime manifest does not resolve to the immutable source release: "
            + ", ".join(mismatches)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--deployed-sha", default=None)
    args = parser.parse_args()

    deployed_sha = str(
        args.deployed_sha
        or os.getenv("RAILWAY_GIT_COMMIT_SHA")
        or os.getenv("GIT_SHA")
        or ""
    ).strip()
    if not deployed_sha:
        parser.error("deployed git SHA is required for a reusable parity receipt")

    source_bundle = load_production_bundle(args.source_manifest)
    runtime_bundle = load_production_bundle(args.runtime_manifest)
    source_summary = validate_production_bundle(source_bundle)
    runtime_summary = validate_production_bundle(runtime_bundle)
    _require_same_release(source_summary, runtime_summary)

    spec = resolve_named_training_spec(source_bundle.model_spec_name)
    scheduled_policy = source_summary.get("scheduled_refit_policy")
    if (
        not isinstance(scheduled_policy, dict)
        or scheduled_policy.get("policy_schema_version") != 2
    ):
        raise RuntimeError("Startup parity requires a final-v2 scheduled refit policy")
    confirmed_strategy = source_summary.get("confirmed_strategy")
    if not isinstance(confirmed_strategy, dict):
        raise RuntimeError("Startup parity requires a validated confirmed strategy")
    replay_args = {
        "start": args.start,
        "limit": args.limit,
        "seed": args.seed,
        "tol": args.tol,
        "exact": {"mode": "exact", "established_only": False},
        "prefight": {"mode": "prefight", "established_only": True},
    }
    binding = {
        "deployed_git_sha": deployed_sha,
        "bundle_id": source_bundle.bundle_id,
        "model_spec_name": source_bundle.model_spec_name,
        "registered_spec_sha256": _canonical_sha256(asdict(spec)),
        "confirmed_strategy_sha256": _canonical_sha256(confirmed_strategy),
        "source_manifest_sha256": _file_sha256(source_bundle.manifest_path),
        "runtime_manifest_sha256": _file_sha256(runtime_bundle.manifest_path),
        "installed_manifest_sha256": source_summary.get("installed_manifest_sha256"),
        "immutable_fights_sha256": (
            source_summary.get("immutable_training_fights_sha256")
            or source_summary.get("processed_fights_sha256")
        ),
        "immutable_features_sha256": (
            source_summary.get("immutable_training_features_sha256")
            or source_summary.get("processed_features_sha256")
        ),
        "replay_args": replay_args,
    }
    receipt_key = _canonical_sha256(binding)
    existing = _load_receipt(args.receipt)
    if (
        existing is not None
        and existing.get("passed") is True
        and existing.get("receipt_key") == receipt_key
        and existing.get("binding") == binding
    ):
        print(json.dumps({"status": "cached", "receipt_key": receipt_key}))
        return 0

    report_dir = args.receipt.parent / "parity_receipts" / receipt_key
    report_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict] = {}
    replay_script = Path(__file__).with_name("parity_replay.py")
    for mode, established_only in (("exact", False), ("prefight", True)):
        report_path = report_dir / f"{mode}.json"
        command = [
            sys.executable,
            str(replay_script),
            "--manifest",
            str(source_bundle.manifest_path),
            "--mode",
            mode,
            "--start",
            args.start,
            "--limit",
            str(args.limit),
            "--seed",
            str(args.seed),
            "--tol",
            str(args.tol),
            "--out",
            str(report_path),
        ]
        if established_only:
            command.append("--established-only")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode
        report = _load_receipt(report_path)
        if report is None or report.get("passed") is not True:
            raise RuntimeError(f"Parity replay did not emit a passing report: {report_path}")
        reports[mode] = {
            "path": str(report_path),
            "sha256": _file_sha256(report_path),
            "n_fights": report.get("n_fights"),
            "requested_feature_count": report.get("requested_feature_count"),
            "forbidden_mismatch_count": report.get("forbidden_mismatch_count"),
        }

    receipt = {
        "schema_version": 1,
        "receipt_key": receipt_key,
        "passed": True,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "binding": binding,
        "reports": reports,
    }
    write_json_atomically(receipt, args.receipt)
    print(json.dumps({"status": "passed", "receipt_key": receipt_key}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
