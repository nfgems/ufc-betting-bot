from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from scripts import check_production_refit_contract as gate
from scripts import bfo_lineage
from src.model.training_spec import resolve_named_training_spec


POLICY_PATH = Path("config/scheduled_refit_policy_v1.json")
NOW = datetime(2026, 8, 6, tzinfo=timezone.utc)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_features(path: Path, *, event_date: str = "2026-08-01") -> None:
    spec = resolve_named_training_spec(
        "full_live_contract_v6_durability_corrected_20260805_fullfit"
    )
    row = {column: 0.0 for column in spec.feature_cols}
    row["event_date"] = event_date
    pd.DataFrame([row]).to_csv(path, index=False)


def _root_readyz(policy: dict, workflow_sha: str) -> dict:
    root = policy["root_release"]
    return {
        "ready": True,
        "production_bundle": {
            "bundle_id": root["bundle_id"],
            "rich_release_id": root["release_id"],
            "source_manifest_sha256": root["source_manifest_sha256"],
            "installed_manifest_sha256": root["installed_manifest_sha256"],
            "model_spec_name": policy["contract"]["fullfit_spec_name"],
            "deployed_git_sha": workflow_sha,
            "training_source_git_sha": root["training_source_git_sha"],
            "model_sha256": root["model_sha256"],
            "no_odds_model_sha256": root["no_odds_model_sha256"],
            "logistic_model_sha256": root["logistic_model_sha256"],
            "immutable_training_fights_sha256": root["processed_fights_sha256"],
            "immutable_training_features_sha256": root["processed_features_sha256"],
            "immutable_training_snapshot_max_event_date": "2026-08-01",
            "independent_audit_fights_canonical_sha256": "c" * 64,
            "independent_audit_features_canonical_sha256": "d" * 64,
            # Mutable runtime lookup data is deliberately different. It is not ancestry.
            "processed_fights_sha256": "a" * 64,
            "processed_features_sha256": "b" * 64,
        },
    }


def _file_record(root: Path, relative: str, content: bytes) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "staged_path": relative,
        "sha256": gate.file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _candidate_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    policy = gate.load_policy(POLICY_PATH)
    workflow_sha = gate.checked_out_git_sha()
    readyz_path = tmp_path / "parent_readyz.json"
    _write_json(readyz_path, _root_readyz(policy, workflow_sha))

    root = tmp_path / "candidate"
    root.mkdir()
    evaluation = resolve_named_training_spec(policy["contract"]["evaluation_spec_name"])
    fullfit = resolve_named_training_spec(policy["contract"]["fullfit_spec_name"])
    fullfit_payload = asdict(fullfit)
    sidecar = root / f"models/{fullfit.name}_spec.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps(fullfit_payload, indent=2), encoding="utf-8")
    saved = {
        "staged_path": f"models/{fullfit.name}_spec.json",
        "sha256": gate.file_sha256(sidecar),
        "bytes": sidecar.stat().st_size,
        "payload": fullfit_payload,
    }

    artifacts = {
        "primary": _file_record(root, "models/xgboost_model.pkl", b"primary"),
        "no_odds": _file_record(root, "models/xgboost_no_odds_model.pkl", b"no-odds"),
        "logistic": _file_record(root, "models/logistic_model.pkl", b"logistic"),
    }
    fights = _file_record(root, "processed/fights_cleaned.csv", b"event_date\n2026-08-01\n")
    feature_path = root / "processed/features.csv"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    _write_features(feature_path)
    features = {
        "staged_path": "processed/features.csv",
        "sha256": gate.file_sha256(feature_path),
        "bytes": feature_path.stat().st_size,
    }

    parent = _root_readyz(policy, workflow_sha)["production_bundle"]
    lineage_payload = {
        "schema_version": 1,
        "parent_bundle_id": parent["bundle_id"],
        "parent_source_manifest_sha256": parent["source_manifest_sha256"],
        "previous_lineage_manifest_sha256": None,
        "batches": [],
    }
    lineage_path = bfo_lineage.write_manifest(
        lineage_payload,
        root / "provenance/bfo_lineage/manifest.json",
    )
    lineage_record = {
        "manifest_staged_path": "provenance/bfo_lineage/manifest.json",
        "manifest_sha256": gate.file_sha256(lineage_path),
        "manifest_bytes": lineage_path.stat().st_size,
        "batch_count": 0,
        "batches": [],
    }
    manifest = {
        "manifest_version": 3,
        "staging_schema_version": 1,
        "bundle_id": "candidate-test",
        "model_spec_name": fullfit.name,
        "no_odds_model_spec_name": f"{fullfit.name}_no_odds",
        "scheduled_refit_policy": {
            "policy_id": policy["policy_id"],
            "sha256": gate.file_sha256(POLICY_PATH),
            "root_bundle_id": policy["root_release"]["bundle_id"],
            "parent_bundle_id": parent["bundle_id"],
            "parent_model_spec_name": parent["model_spec_name"],
            "parent_model_sha256": parent["model_sha256"],
            "parent_no_odds_model_sha256": parent["no_odds_model_sha256"],
            "parent_logistic_model_sha256": parent["logistic_model_sha256"],
            "parent_processed_fights_sha256": parent[
                "immutable_training_fights_sha256"
            ],
            "parent_processed_features_sha256": parent[
                "immutable_training_features_sha256"
            ],
        },
        "registered_training_specs": {
            "selected_evaluation": {
                "payload": asdict(evaluation),
                "sha256": gate.canonical_json_sha256(asdict(evaluation)),
            },
            "selected_fullfit": {
                "payload": fullfit_payload,
                "sha256": gate.canonical_json_sha256(fullfit_payload),
            },
            "allowed_differences": policy["contract"]["allowed_fullfit_differences"],
        },
        "saved_fullfit_spec": saved,
        "model_artifacts": artifacts,
        "model_sha256": artifacts["primary"]["sha256"],
        "no_odds_model_sha256": artifacts["no_odds"]["sha256"],
        "logistic_model_sha256": artifacts["logistic"]["sha256"],
        "immutable_training_snapshot": {
            "immutable": True,
            "fights": fights,
            "features": features,
        },
        "processed_fights_sha256": fights["sha256"],
        "processed_features_sha256": features["sha256"],
        "snapshot_max_event_date": "2026-08-01",
        "raw_input_provenance": {
            "bfo_ledger": {
                "sha256": "c" * 64,
                "corrected_csv_files": [{"sha256": "d" * 64, "rows": 1}],
            },
            "scheduled_bfo_lineage": lineage_record,
        },
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    return POLICY_PATH, manifest_path, readyz_path


def test_shipped_policy_is_strict_and_matches_registry() -> None:
    policy = gate.load_policy(POLICY_PATH)
    errors, evaluation, fullfit = gate.validate_policy_registry(policy)

    assert errors == []
    assert evaluation is not None and fullfit is not None
    assert len(fullfit.feature_cols) == 211
    assert policy["contract"]["minimum_cutoff_buffer_days"] == 1
    assert policy["baseline"]["comparison_role"] == (
        "root_release_reference_thresholds"
    )
    assert policy["baseline"]["evidence_sha256"] == (
        "8d12ae8c9b1b63e9585ea82468be810a911dfbd6c73d2e429179cab3679fb0ba"
    )
    assert policy["baseline"]["evidence_protocol_sha256"] == (
        policy["baseline"]["scheduled_protocol_sha256"]
    )
    assert policy["baseline"]["scheduled_protocol_sha256"] == (
        gate.scheduled_protocol_sha256(policy)
    )
    assert policy["root_release"]["promotion_git_sha"] == (
        "978588e193e1e5d88fa920c1fff5e08e0b63fa84"
    )
    assert gate.validate_workflow_git_lineage(
        policy, workflow_git_sha=gate.checked_out_git_sha()
    ) == []


def test_feature_gate_does_not_hard_fail_on_wall_clock_advance_warning(
    tmp_path: Path,
) -> None:
    policy = gate.load_policy(POLICY_PATH)
    fullfit = resolve_named_training_spec(policy["contract"]["fullfit_spec_name"])
    features = tmp_path / "features.csv"
    _write_features(features, event_date="2026-12-01")

    info, errors = gate.inspect_features(
        features,
        policy=policy,
        fullfit_spec=fullfit,
        now=datetime(2026, 12, 1, tzinfo=timezone.utc),
    )

    assert errors == []
    assert info["cutoff_buffer_days"] == 31


def test_feature_gate_rejects_snapshot_at_exclusive_cutoff(tmp_path: Path) -> None:
    policy = gate.load_policy(POLICY_PATH)
    fullfit = resolve_named_training_spec(policy["contract"]["fullfit_spec_name"])
    features = tmp_path / "features.csv"
    _write_features(features, event_date="2027-01-01")

    _, errors = gate.inspect_features(
        features,
        policy=policy,
        fullfit_spec=fullfit,
        now=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )

    assert any("does not include snapshot max" in error for error in errors)


@pytest.mark.parametrize(
    ("event_date", "now", "expected_error"),
    [
        ("2026-08-07", NOW, "in the future"),
        (
            "2026-08-01",
            datetime(2026, 9, 10, tzinfo=timezone.utc),
            "days old",
        ),
    ],
)
def test_feature_gate_retains_future_and_freshness_checks(
    tmp_path: Path,
    event_date: str,
    now: datetime,
    expected_error: str,
) -> None:
    policy = gate.load_policy(POLICY_PATH)
    fullfit = resolve_named_training_spec(policy["contract"]["fullfit_spec_name"])
    features = tmp_path / "features.csv"
    _write_features(features, event_date=event_date)

    _, errors = gate.inspect_features(
        features,
        policy=policy,
        fullfit_spec=fullfit,
        now=now,
    )

    assert any(expected_error in error for error in errors)


@pytest.mark.parametrize("mutation", ["duplicate", "unknown"])
def test_policy_rejects_duplicate_and_unknown_fields(tmp_path: Path, mutation: str) -> None:
    text = POLICY_PATH.read_text(encoding="utf-8")
    if mutation == "duplicate":
        text = text.replace('"schema_version": 1,', '"schema_version": 1,\n  "schema_version": 1,', 1)
    else:
        text = text.replace('"schema_version": 1,', '"schema_version": 1,\n  "surprise": true,', 1)
    path = tmp_path / "policy.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(gate.ContractInputError):
        gate.load_policy(path)


def test_preflight_accepts_immutable_root_lineage_and_ignores_runtime_lookup_drift(
    tmp_path: Path,
) -> None:
    policy = gate.load_policy(POLICY_PATH)
    features = tmp_path / "features.csv"
    readyz = tmp_path / "readyz.json"
    _write_features(features)
    _write_json(readyz, _root_readyz(policy, gate.checked_out_git_sha()))

    result = gate.evaluate_preflight(
        policy_path=POLICY_PATH,
        features_path=features,
        readyz_path=readyz,
        now=NOW,
    )

    assert result["errors"] == []
    assert result["features"]["snapshot_max_event_date"] == "2026-08-01"
    assert result["active_independent_audit"] == {
        "fights_canonical_sha256": "c" * 64,
        "features_canonical_sha256": "d" * 64,
    }


def test_preflight_rejects_wrong_deployed_head_and_mutable_hash_substitution(
    tmp_path: Path,
) -> None:
    policy = gate.load_policy(POLICY_PATH)
    features = tmp_path / "features.csv"
    readyz = tmp_path / "readyz.json"
    _write_features(features)
    payload = _root_readyz(policy, "e" * 40)
    payload["production_bundle"].pop("immutable_training_features_sha256")
    payload["production_bundle"]["processed_features_sha256"] = policy["root_release"][
        "processed_features_sha256"
    ]
    _write_json(readyz, payload)

    result = gate.evaluate_preflight(
        policy_path=POLICY_PATH,
        features_path=features,
        readyz_path=readyz,
        now=NOW,
    )

    assert any("deployed_git_sha" in error for error in result["errors"])
    assert any("immutable_training_features_sha256" in error for error in result["errors"])


def test_candidate_accepts_exact_policy_bound_artifacts(tmp_path: Path) -> None:
    policy_path, manifest_path, readyz_path = _candidate_fixture(tmp_path)

    result = gate.evaluate_candidate(
        policy_path=policy_path,
        manifest_path=manifest_path,
        parent_readyz_path=readyz_path,
        now=NOW,
    )

    assert result["errors"] == []
    assert result["candidate_manifest"]["bundle_id"] == "candidate-test"


def test_candidate_rejects_parent_runtime_hash_as_immutable_binding(tmp_path: Path) -> None:
    policy_path, manifest_path, readyz_path = _candidate_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scheduled_refit_policy"]["parent_processed_features_sha256"] = "b" * 64
    _write_json(manifest_path, manifest)

    result = gate.evaluate_candidate(
        policy_path=policy_path,
        manifest_path=manifest_path,
        parent_readyz_path=readyz_path,
        now=NOW,
    )

    assert any("parent_processed_features_sha256" in error for error in result["errors"])


def test_candidate_rejects_snapshot_older_than_immediate_parent(tmp_path: Path) -> None:
    policy_path, manifest_path, readyz_path = _candidate_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["snapshot_max_event_date"] = "2026-07-31"
    _write_json(manifest_path, manifest)

    result = gate.evaluate_candidate(
        policy_path=policy_path,
        manifest_path=manifest_path,
        parent_readyz_path=readyz_path,
        now=NOW,
    )

    assert any("predates the immediate parent" in error for error in result["errors"])


def test_candidate_rejects_bfo_lineage_not_chained_to_parent(tmp_path: Path) -> None:
    policy_path, manifest_path, readyz_path = _candidate_fixture(tmp_path)
    candidate_root = manifest_path.parent
    lineage_path = candidate_root / "provenance/bfo_lineage/manifest.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["parent_bundle_id"] = "wrong-parent"
    bfo_lineage.write_manifest(lineage, lineage_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["raw_input_provenance"]["scheduled_bfo_lineage"]
    record["manifest_sha256"] = gate.file_sha256(lineage_path)
    record["manifest_bytes"] = lineage_path.stat().st_size
    _write_json(manifest_path, manifest)

    result = gate.evaluate_candidate(
        policy_path=policy_path,
        manifest_path=manifest_path,
        parent_readyz_path=readyz_path,
        now=NOW,
    )

    assert any("does not chain to its parent" in error for error in result["errors"])


def test_cli_error_always_writes_report(tmp_path: Path) -> None:
    bad_policy = tmp_path / "bad.json"
    bad_policy.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    report = tmp_path / "report.json"

    exit_code = gate.main(
        [
            "preflight",
            "--policy",
            str(bad_policy),
            "--features",
            str(tmp_path / "missing.csv"),
            "--readyz-json",
            str(tmp_path / "missing.json"),
            "--report",
            str(report),
        ]
    )

    assert exit_code == 2
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "ERROR"
