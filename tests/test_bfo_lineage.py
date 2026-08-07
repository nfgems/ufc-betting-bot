import json
from pathlib import Path

import pytest

from scripts import bfo_lineage


def _package(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "artifact"
    batches = root / "batches"
    batches.mkdir(parents=True)
    csv_name = "historical_odds_bfo_recovered_auto_test.csv"
    ledger_name = "historical_odds_bfo_recovered_auto_test.provenance.jsonl"
    csv_path = batches / csv_name
    ledger_path = batches / ledger_name
    csv_path.write_text("event_date,fighter_a\n2026-08-01,A\n", encoding="utf-8")
    ledger_path.write_text(json.dumps({"decision": "accepted"}) + "\n", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "parent_bundle_id": "parent-bundle",
        "parent_source_manifest_sha256": "a" * 64,
        "previous_lineage_manifest_sha256": None,
        "batches": [
            {
                "accepted_records": 1,
                "rejected_records": 0,
                "csv": {
                    "raw_path": f"data/raw/historical_odds/{csv_name}",
                    "artifact_path": f"batches/{csv_name}",
                    "sha256": bfo_lineage.sha256_file(csv_path),
                    "bytes": csv_path.stat().st_size,
                    "rows": 1,
                },
                "provenance": {
                    "raw_path": f"data/raw/historical_odds/{ledger_name}",
                    "artifact_path": f"batches/{ledger_name}",
                    "sha256": bfo_lineage.sha256_file(ledger_path),
                    "bytes": ledger_path.stat().st_size,
                    "line_count": 1,
                },
            }
        ],
    }
    manifest = bfo_lineage.write_manifest(payload, root / "manifest.json")
    return manifest, payload


def test_validate_and_restore_exact_original_batch_files(tmp_path: Path) -> None:
    manifest, payload = _package(tmp_path)
    expected_sha = bfo_lineage.sha256_file(manifest)
    repo = tmp_path / "repo"
    repo.mkdir()

    result = bfo_lineage.restore_package(
        manifest,
        repo_root=repo,
        expected_manifest_sha256=expected_sha,
    )

    assert result["manifest"] == payload
    for record in (payload["batches"][0]["csv"], payload["batches"][0]["provenance"]):
        restored = repo / record["raw_path"]
        assert bfo_lineage.sha256_file(restored) == record["sha256"]


def test_package_tamper_fails_closed(tmp_path: Path) -> None:
    manifest, payload = _package(tmp_path)
    (manifest.parent / payload["batches"][0]["csv"]["artifact_path"]).write_text(
        "changed\n",
        encoding="utf-8",
    )

    with pytest.raises(bfo_lineage.BfoLineageError, match="identity is invalid"):
        bfo_lineage.validate_package(
            manifest,
            expected_manifest_sha256=bfo_lineage.sha256_file(manifest),
        )


def test_manifest_rejects_unsafe_or_duplicate_batch_paths(tmp_path: Path) -> None:
    _manifest, payload = _package(tmp_path)
    unsafe = json.loads(json.dumps(payload))
    unsafe["batches"][0]["csv"]["raw_path"] = "../escaped.csv"
    with pytest.raises(bfo_lineage.BfoLineageError, match="directly under"):
        bfo_lineage.validate_manifest_payload(unsafe)

    duplicate = json.loads(json.dumps(payload))
    duplicate["batches"].append(json.loads(json.dumps(duplicate["batches"][0])))
    with pytest.raises(bfo_lineage.BfoLineageError, match="duplicate"):
        bfo_lineage.validate_manifest_payload(duplicate)


def test_restore_cli_allows_only_exact_root_without_artifact(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    root_sha = "a" * 64
    policy = tmp_path / "policy.json"
    readyz = tmp_path / "readyz.json"
    report = tmp_path / "report.json"
    policy.write_text(
        json.dumps({"root_release": {"source_manifest_sha256": root_sha}}),
        encoding="utf-8",
    )
    readyz.write_text(
        json.dumps(
            {
                "ready": True,
                "production_bundle": {"source_manifest_sha256": root_sha},
            }
        ),
        encoding="utf-8",
    )

    assert bfo_lineage.main(
        [
            "--parent-readyz",
            str(readyz),
            "--policy",
            str(policy),
            "--repo-root",
            str(repo),
            "--report",
            str(report),
        ]
    ) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "root_baseline"

    readyz.write_text(
        json.dumps(
            {
                "ready": True,
                "production_bundle": {"source_manifest_sha256": "b" * 64},
            }
        ),
        encoding="utf-8",
    )
    assert bfo_lineage.main(
        [
            "--parent-readyz",
            str(readyz),
            "--policy",
            str(policy),
            "--repo-root",
            str(repo),
            "--report",
            str(report),
        ]
    ) == 1
    assert "missing its BFO lineage identity" in json.loads(
        report.read_text(encoding="utf-8")
    )["error"]
