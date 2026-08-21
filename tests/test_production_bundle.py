import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

import scripts.bootstrap_runtime_production_bundle as bootstrap_runtime_bundle
from src.data import fighter_lookup
from src.model import production_bundle


def _write_model(path: Path, *, spec_name: str) -> None:
    payload = {
        "feature_cols": ["demo_feature"],
        "training_spec": {
            "name": spec_name,
            "feature_cols": ["demo_feature"],
        },
        "model": "stub-model",
    }
    joblib.dump(payload, path)


def _configure_bundle_paths(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    for env_name in ("RAILWAY_GIT_COMMIT_SHA", "GITHUB_SHA", "GIT_SHA"):
        monkeypatch.delenv(env_name, raising=False)
    models_dir = tmp_path / "models"
    processed_dir = tmp_path / "data" / "processed"
    models_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    monkeypatch.setattr(production_bundle, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(production_bundle, "DEFAULT_MANIFEST_PATH", models_dir / "current_production_model.json")
    monkeypatch.setattr(production_bundle, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(production_bundle, "MODELS_DIR", models_dir)
    monkeypatch.setattr(production_bundle, "PROCESSED_DATA_DIR", processed_dir)
    production_bundle._cached_snapshot_max_event_date.cache_clear()
    return models_dir, processed_dir


def _pinned_manifest_payload(
    *,
    bundle_id: str,
    models_dir: Path,
    processed_dir: Path,
    built_at: str,
    include_model_hashes: bool = True,
) -> dict:
    payload = {
        "bundle_id": bundle_id,
        "model_spec_name": "prod_spec_new",
        "no_odds_model_spec_name": "prod_spec_new_no_odds",
        "model_path": str(models_dir / "xgboost_model.pkl"),
        "no_odds_model_path": str(models_dir / "xgboost_no_odds_model.pkl"),
        "processed_dir": str(processed_dir),
        "snapshot_max_event_date": production_bundle.get_processed_snapshot_max_event_date(
            processed_dir
        ),
        "built_at": built_at,
        "git_sha": "source-sha",
        **production_bundle.get_processed_snapshot_fingerprints(processed_dir),
    }
    if include_model_hashes:
        payload.update(
            production_bundle.get_model_artifact_fingerprints(
                models_dir / "xgboost_model.pkl",
                models_dir / "xgboost_no_odds_model.pkl",
            )
        )
    return payload


def _write_rich_models(models_dir: Path, *, generation: str) -> dict[str, dict]:
    primary_spec = {
        "name": "prod_spec_rich",
        "description": "Rich production contract",
        "feature_cols": ["demo_feature"],
        "git_hash": "training-sha",
        "trained_at": "2026-08-06T00:00:00Z",
    }
    no_odds_spec = production_bundle._expected_no_odds_spec_payload(primary_spec)
    specs = {
        "primary": primary_spec,
        "no_odds": no_odds_spec,
        "logistic": primary_spec,
    }
    filenames = {
        "primary": "xgboost_model.pkl",
        "no_odds": "xgboost_no_odds_model.pkl",
        "logistic": "logistic_model.pkl",
    }
    for label, filename in filenames.items():
        joblib.dump(
            {
                "feature_cols": list(specs[label]["feature_cols"]),
                "training_spec": specs[label],
                "model": f"stub-{label}-{generation}",
            },
            models_dir / filename,
        )
    return specs


def _scheduled_refit_policy_identity() -> dict:
    return {
        "policy_id": "scheduled-v1",
        "sha256": "1" * 64,
        "root_bundle_id": "root-bundle",
        "parent_bundle_id": "parent-bundle",
        "parent_model_spec_name": "approved-fullfit",
        "parent_model_sha256": "2" * 64,
        "parent_no_odds_model_sha256": "3" * 64,
        "parent_logistic_model_sha256": "4" * 64,
        "parent_processed_fights_sha256": "5" * 64,
        "parent_processed_features_sha256": "6" * 64,
    }


def _write_scheduled_bfo_lineage(release_root: Path) -> dict:
    lineage_root = release_root / "provenance" / "bfo_lineage"
    batches_dir = lineage_root / "batches"
    batches_dir.mkdir(parents=True)
    csv_name = "historical_odds_bfo_recovered_20260806.csv"
    ledger_name = "historical_odds_bfo_recovered_20260806.provenance.jsonl"
    csv_path = batches_dir / csv_name
    ledger_path = batches_dir / ledger_name
    csv_path.write_bytes(b"fighter_name,odds\nExample Fighter,-110\n")
    ledger_path.write_bytes(b'{"accepted": true}\n')
    batch = {
        "accepted_records": 1,
        "rejected_records": 0,
        "csv": {
            "raw_path": f"data/raw/historical_odds/{csv_name}",
            "artifact_path": f"batches/{csv_name}",
            "sha256": production_bundle._file_sha256(csv_path),
            "bytes": csv_path.stat().st_size,
            "rows": 1,
        },
        "provenance": {
            "raw_path": f"data/raw/historical_odds/{ledger_name}",
            "artifact_path": f"batches/{ledger_name}",
            "sha256": production_bundle._file_sha256(ledger_path),
            "bytes": ledger_path.stat().st_size,
            "line_count": 1,
        },
    }
    lineage_payload = {
        "schema_version": 1,
        "parent_bundle_id": "parent-bundle",
        "parent_source_manifest_sha256": "a" * 64,
        "previous_lineage_manifest_sha256": None,
        "batches": [batch],
    }
    lineage_path = lineage_root / "manifest.json"
    lineage_path.write_bytes(
        (
            json.dumps(
                lineage_payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )
    return {
        "manifest_staged_path": "provenance/bfo_lineage/manifest.json",
        "manifest_sha256": production_bundle._file_sha256(lineage_path),
        "manifest_bytes": lineage_path.stat().st_size,
        "batch_count": 1,
        "batches": [batch],
    }


def _write_rich_manifest(
    release_root: Path,
    *,
    models_dir: Path,
    processed_dir: Path,
    generation: str,
    specs: dict[str, dict],
    scheduled_bfo_lineage: bool = False,
) -> Path:
    release_models = release_root / "models"
    release_models.mkdir(parents=True)
    spec_path = release_models / "prod_spec_rich_spec.json"
    spec_path.write_text(json.dumps(specs["primary"]), encoding="utf-8")

    model_paths = {
        "primary": models_dir / "xgboost_model.pkl",
        "no_odds": models_dir / "xgboost_no_odds_model.pkl",
        "logistic": models_dir / "logistic_model.pkl",
    }
    model_hash_fields = {
        "primary": "model_sha256",
        "no_odds": "no_odds_model_sha256",
        "logistic": "logistic_model_sha256",
    }
    fingerprints = production_bundle.get_model_artifact_fingerprints(
        model_paths["primary"],
        model_paths["no_odds"],
        model_paths["logistic"],
    )
    artifacts = {}
    for label, path in model_paths.items():
        artifacts[label] = {
            "staged_path": f"models/{path.name}",
            "sha256": fingerprints[model_hash_fields[label]],
            "bytes": path.stat().st_size,
            "embedded_training_spec": specs[label],
            "embedded_training_spec_sha256": production_bundle._canonical_json_sha256(
                specs[label]
            ),
        }

    processed = production_bundle.get_processed_snapshot_fingerprints(processed_dir)
    selected_fullfit = dict(specs["primary"])
    selected_fullfit["git_hash"] = ""
    selected_fullfit["trained_at"] = ""
    selected_evaluation = {
        **selected_fullfit,
        "name": "evaluation_spec_rich",
    }
    audit_dir = release_root / "provenance" / "independent_audit_snapshot"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_fights = audit_dir / "fights_cleaned.csv"
    audit_features = audit_dir / "features.csv"
    audit_fights.write_bytes((processed_dir / "fights_cleaned.csv").read_bytes())
    audit_features.write_bytes((processed_dir / "features.csv").read_bytes())
    raw_input_provenance = {"generation": generation}
    if scheduled_bfo_lineage:
        raw_input_provenance["scheduled_bfo_lineage"] = (
            _write_scheduled_bfo_lineage(release_root)
        )
    payload = {
        "manifest_version": 3,
        "staging_schema_version": 1,
        "bundle_id": f"rich-bundle-{generation}",
        "model_spec_name": "prod_spec_rich",
        "no_odds_model_spec_name": "prod_spec_rich_no_odds",
        "model_path": str(model_paths["primary"]),
        "no_odds_model_path": str(model_paths["no_odds"]),
        "logistic_model_path": str(model_paths["logistic"]),
        "processed_dir": str(processed_dir),
        "snapshot_max_event_date": production_bundle.get_processed_snapshot_max_event_date(
            processed_dir
        ),
        "built_at": "2026-08-06T00:00:00Z",
        "manifest_updated_at": "2026-08-06T00:00:00Z",
        "git_sha": "training-sha",
        "training_source_git_sha": "training-sha",
        **fingerprints,
        **processed,
        "source_identity": {
            "generation": generation,
            "base_git_sha": "training-sha",
        },
        "registered_training_specs": {
            "selected_evaluation": {
                "payload": selected_evaluation,
                "sha256": production_bundle._canonical_json_sha256(
                    selected_evaluation
                ),
            },
            "selected_fullfit": {
                "payload": selected_fullfit,
                "sha256": production_bundle._canonical_json_sha256(
                    selected_fullfit
                ),
            },
        },
        "model_artifacts": artifacts,
        "saved_fullfit_spec": {
            "staged_path": "models/prod_spec_rich_spec.json",
            "sha256": production_bundle._file_sha256(spec_path),
            "bytes": spec_path.stat().st_size,
            "payload": specs["primary"],
        },
        "immutable_training_snapshot": {
            "immutable": True,
            "snapshot_max_event_date": production_bundle.get_processed_snapshot_max_event_date(
                processed_dir
            ),
            "fights": {
                "sha256": processed["processed_fights_sha256"],
                "bytes": processed["processed_fights_bytes"],
            },
            "features": {
                "sha256": processed["processed_features_sha256"],
                "bytes": processed["processed_features_bytes"],
            },
        },
        "raw_input_provenance": raw_input_provenance,
        "selection_evidence": {"generation": generation},
        "training_invocation": {
            "generation": generation,
            "independent_audit_snapshot": {
                "fights": {
                    "staged_path": (
                        "provenance/independent_audit_snapshot/fights_cleaned.csv"
                    ),
                    "sha256": production_bundle._file_sha256(audit_fights),
                },
                "features": {
                    "staged_path": (
                        "provenance/independent_audit_snapshot/features.csv"
                    ),
                    "sha256": production_bundle._file_sha256(audit_features),
                },
            },
        },
        "assembly_validation_environment": {"generation": generation},
        "finite_inference": {"generation": generation},
        "previous_rollback_identity": {"generation": generation},
    }
    if scheduled_bfo_lineage:
        payload["scheduled_refit_policy"] = _scheduled_refit_policy_identity()
    manifest_path = release_root / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def _write_rich_validation_manifest(
    tmp_path: Path,
    monkeypatch,
    *,
    scheduled_bfo_lineage: bool,
) -> Path:
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    pd.DataFrame({"event_date": ["2026-08-01"]}).to_csv(
        processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"]}).to_csv(
        processed_dir / "features.csv", index=False
    )
    specs = _write_rich_models(models_dir, generation="lineage")
    return _write_rich_manifest(
        tmp_path / "rich-release",
        models_dir=models_dir,
        processed_dir=processed_dir,
        generation="lineage",
        specs=specs,
        scheduled_bfo_lineage=scheduled_bfo_lineage,
    )


def test_git_provenance_separates_ci_source_from_railway_deployment(monkeypatch):
    for name in ("RAILWAY_GIT_COMMIT_SHA", "GITHUB_SHA", "GIT_SHA"):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("GITHUB_SHA", "github-source-sha")
    assert production_bundle._determine_training_source_git_sha({"git_sha": "stale"}) == "github-source-sha"
    assert production_bundle._determine_deployed_git_sha({"deployed_git_sha": "old-deploy"}) is None
    assert production_bundle._determine_git_sha({"git_sha": "stale"}) == "github-source-sha"

    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "railway-deploy-sha")
    assert production_bundle._determine_training_source_git_sha({"git_sha": "stale"}) == "github-source-sha"
    assert production_bundle._determine_deployed_git_sha({}) == "railway-deploy-sha"
    assert production_bundle._determine_git_sha({"git_sha": "stale"}) == "railway-deploy-sha"


def test_load_legacy_bundle_uses_git_sha_as_training_source_compatibility_alias(tmp_path, monkeypatch):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    manifest_path = models_dir / "current_production_model.json"
    manifest_path.write_text(
        json.dumps(
            {
                "bundle_id": "legacy-bundle",
                "model_spec_name": "prod_spec",
                "model_path": str(models_dir / "xgboost_model.pkl"),
                "no_odds_model_path": str(models_dir / "xgboost_no_odds_model.pkl"),
                "processed_dir": str(processed_dir),
                "snapshot_max_event_date": "2026-03-21",
                "built_at": "2026-03-23T00:00:00Z",
                "git_sha": "legacy-ambiguous-sha",
            }
        ),
        encoding="utf-8",
    )

    bundle = production_bundle.load_production_bundle(manifest_path)

    assert bundle.git_sha == "legacy-ambiguous-sha"
    assert bundle.training_source_git_sha == "legacy-ambiguous-sha"
    assert bundle.deployed_git_sha is None


def test_validate_production_bundle_reports_active_runtime_summary(tmp_path, monkeypatch):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)

    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_no_odds")
    _write_model(models_dir / "logistic_model.pkl", spec_name="prod_spec")

    pd.DataFrame({"event_date": ["2026-03-20", "2026-03-21"]}).to_csv(
        processed_dir / "fights_cleaned.csv",
        index=False,
    )
    pd.DataFrame({"event_date": ["2026-03-19", "2026-03-21"]}).to_csv(
        processed_dir / "features.csv",
        index=False,
    )

    manifest_path = tmp_path / "models" / "bundle.json"
    manifest_path.write_text(
        """
{
  "bundle_id": "bundle-1",
  "model_spec_name": "prod_spec",
  "no_odds_model_spec_name": "prod_spec_no_odds",
  "model_path": "${MODELS_DIR}/xgboost_model.pkl",
  "no_odds_model_path": "${MODELS_DIR}/xgboost_no_odds_model.pkl",
  "logistic_model_path": "${MODELS_DIR}/logistic_model.pkl",
  "processed_dir": "${PROCESSED_DATA_DIR}",
  "snapshot_max_event_date": "2026-03-21",
  "built_at": "2026-03-23T00:00:00Z",
  "git_sha": "abc123"
}
""".strip(),
        encoding="utf-8",
    )

    bundle = production_bundle.load_production_bundle(manifest_path)
    summary = production_bundle.validate_production_bundle(bundle)

    assert summary["bundle_id"] == "bundle-1"
    assert summary["model_path"] == str(models_dir / "xgboost_model.pkl")
    assert summary["processed_dir"] == str(processed_dir)
    assert summary["processed_snapshot_max_event_date"] == "2026-03-21"
    assert summary["embedded_model_spec_name"] == "prod_spec"
    assert summary["model_sha256"] == production_bundle._file_sha256(
        models_dir / "xgboost_model.pkl"
    )
    assert summary["no_odds_model_sha256"] == production_bundle._file_sha256(
        models_dir / "xgboost_no_odds_model.pkl"
    )
    assert summary["training_source_git_sha"] == "abc123"
    assert summary["deployed_git_sha"] is None
    assert bundle.logistic_model_sha256 is None
    assert summary["logistic_model_sha256"] == production_bundle._file_sha256(
        models_dir / "logistic_model.pkl"
    )
    assert summary["model_hashes"] == {
        "primary_sha256": summary["model_sha256"],
        "no_odds_sha256": summary["no_odds_model_sha256"],
        "logistic_sha256": summary["logistic_model_sha256"],
    }
    assert summary["rich_manifest_validated"] is False


def test_validate_production_bundle_rejects_stale_processed_snapshot(tmp_path, monkeypatch):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)

    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_no_odds")

    pd.DataFrame({"event_date": ["2026-03-20"]}).to_csv(processed_dir / "fights_cleaned.csv", index=False)
    pd.DataFrame({"event_date": ["2026-03-20"]}).to_csv(processed_dir / "features.csv", index=False)

    manifest_path = tmp_path / "models" / "bundle.json"
    manifest_path.write_text(
        """
{
  "bundle_id": "bundle-1",
  "model_spec_name": "prod_spec",
  "no_odds_model_spec_name": "prod_spec_no_odds",
  "model_path": "${MODELS_DIR}/xgboost_model.pkl",
  "no_odds_model_path": "${MODELS_DIR}/xgboost_no_odds_model.pkl",
  "processed_dir": "${PROCESSED_DATA_DIR}",
  "snapshot_max_event_date": "2026-03-21",
  "built_at": "2026-03-23T00:00:00Z",
  "git_sha": "abc123"
}
""".strip(),
        encoding="utf-8",
    )

    bundle = production_bundle.load_production_bundle(manifest_path)

    with pytest.raises(production_bundle.ProductionBundleError, match="older than expected"):
        production_bundle.validate_production_bundle(bundle)


def test_hosted_processed_dir_resolution_bypasses_candidate_inference(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    candidate_dir = processed_dir / "candidates" / "candidate_spec"
    candidate_dir.mkdir(parents=True)
    processed_dir.mkdir(exist_ok=True)
    (processed_dir / "features.csv").write_text("event_date\n2026-03-21\n", encoding="utf-8")
    (candidate_dir / "features.csv").write_text("event_date\n2026-03-21\n", encoding="utf-8")

    monkeypatch.setattr(fighter_lookup, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(fighter_lookup, "_hosted_processed_data_dir", lambda: processed_dir)

    class _Spec:
        name = "candidate_spec"

    resolved = fighter_lookup._resolve_processed_data_dir(training_spec=_Spec())

    assert resolved == processed_dir


def test_reconcile_production_bundle_manifest_writes_exact_snapshot_fingerprints(tmp_path, monkeypatch):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)

    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_no_odds")
    _write_model(models_dir / "logistic_model.pkl", spec_name="prod_spec")
    pd.DataFrame({"event_date": ["2026-03-20", "2026-03-21"]}).to_csv(processed_dir / "fights_cleaned.csv", index=False)
    pd.DataFrame({"event_date": ["2026-03-20", "2026-03-21"]}).to_csv(processed_dir / "features.csv", index=False)

    source_manifest = tmp_path / "models" / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "bundle_id": "bundle-source",
                "model_spec_name": "prod_spec",
                "no_odds_model_spec_name": "prod_spec_no_odds",
                "built_at": "2026-03-23T00:00:00Z",
                "git_sha": "abc123",
                "selection_basis": "source manifest",
            }
        ),
        encoding="utf-8",
    )
    runtime_manifest = tmp_path / "runtime" / "manifest.json"

    summary = production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=runtime_manifest,
        source_manifest_path=source_manifest,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        logistic_model_path=models_dir / "logistic_model.pkl",
        processed_dir=processed_dir,
    )

    payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    assert payload["bundle_id"] == "ufc-production-20260321-prod_spec"
    assert payload["processed_dir"] == str(processed_dir.resolve(strict=False))
    assert payload["processed_fights_sha256"]
    assert payload["processed_features_sha256"]
    assert payload["training_source_git_sha"] == "abc123"
    assert payload["git_sha"] == "abc123"
    assert "deployed_git_sha" not in payload
    assert payload["logistic_model_sha256"] == production_bundle._file_sha256(
        models_dir / "logistic_model.pkl"
    )
    assert payload["manifest_updated_at"]
    assert payload["built_at"] == "2026-03-23T00:00:00Z"
    assert payload["selection_basis"] == "source manifest"
    assert summary["processed_fights_sha256"] == payload["processed_fights_sha256"]
    assert summary["processed_features_sha256"] == payload["processed_features_sha256"]
    assert summary["training_source_git_sha"] == "abc123"
    assert summary["deployed_git_sha"] is None
    assert summary["logistic_model_sha256"] == payload["logistic_model_sha256"]


def test_reconcile_runtime_preserves_training_source_and_stamps_deployed_sha(tmp_path, monkeypatch):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_no_odds")
    pd.DataFrame({"event_date": ["2026-03-21"]}).to_csv(
        processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-03-21"]}).to_csv(
        processed_dir / "features.csv", index=False
    )
    source_manifest = models_dir / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "bundle_id": "source-bundle",
                "model_spec_name": "prod_spec",
                "no_odds_model_spec_name": "prod_spec_no_odds",
                "model_path": str(models_dir / "xgboost_model.pkl"),
                "no_odds_model_path": str(models_dir / "xgboost_no_odds_model.pkl"),
                "processed_dir": str(processed_dir),
                "snapshot_max_event_date": "2026-03-21",
                "built_at": "2026-03-23T00:00:00Z",
                "git_sha": "training-source-sha",
                "training_source_git_sha": "training-source-sha",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "deployed-railway-sha")
    runtime_manifest = tmp_path / "runtime" / "manifest.json"

    summary = production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=runtime_manifest,
        source_manifest_path=source_manifest,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        processed_dir=processed_dir,
    )

    payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    assert payload["training_source_git_sha"] == "training-source-sha"
    assert payload["deployed_git_sha"] == "deployed-railway-sha"
    assert payload["git_sha"] == "deployed-railway-sha"
    assert summary["training_source_git_sha"] == "training-source-sha"
    assert summary["deployed_git_sha"] == "deployed-railway-sha"


def test_same_spec_rich_refit_replaces_all_rich_identity_sections(tmp_path, monkeypatch):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    pd.DataFrame({"event_date": ["2026-08-01"]}).to_csv(
        processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"]}).to_csv(
        processed_dir / "features.csv", index=False
    )
    runtime_manifest = tmp_path / "runtime" / "manifest.json"

    old_specs = _write_rich_models(models_dir, generation="old")
    old_source = _write_rich_manifest(
        tmp_path / "release-old",
        models_dir=models_dir,
        processed_dir=processed_dir,
        generation="old",
        specs=old_specs,
    )
    production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=runtime_manifest,
        source_manifest_path=old_source,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        logistic_model_path=models_dir / "logistic_model.pkl",
        processed_dir=processed_dir,
    )
    old_payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    old_primary_hash = old_payload["model_sha256"]

    new_specs = _write_rich_models(models_dir, generation="new")
    new_source = _write_rich_manifest(
        tmp_path / "release-new",
        models_dir=models_dir,
        processed_dir=processed_dir,
        generation="new",
        specs=new_specs,
    )
    summary = production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=runtime_manifest,
        source_manifest_path=new_source,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        logistic_model_path=models_dir / "logistic_model.pkl",
        processed_dir=processed_dir,
    )

    payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    assert payload["model_spec_name"] == "prod_spec_rich"
    assert payload["model_sha256"] != old_primary_hash
    for section in production_bundle._RICH_MANIFEST_SECTIONS:
        if section in {
            "registered_training_specs",
            "model_artifacts",
            "saved_fullfit_spec",
            "immutable_training_snapshot",
        }:
            continue
        assert payload[section]["generation"] == "new"
    assert payload["rich_release_root"] == str((tmp_path / "release-new").resolve())
    assert summary["rich_manifest_validated"] is True
    assert summary["rich_release_root"] == payload["rich_release_root"]
    assert summary["model_hashes"] == {
        "primary_sha256": payload["model_sha256"],
        "no_odds_sha256": payload["no_odds_model_sha256"],
        "logistic_sha256": payload["logistic_model_sha256"],
    }
    assert summary["scheduled_bfo_lineage_manifest_sha256"] is None
    assert summary["scheduled_bfo_lineage_batch_count"] == 0


def test_runtime_refresh_repairs_rich_training_source_from_model_identity(
    tmp_path, monkeypatch
):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    pd.DataFrame({"event_date": ["2026-08-01"], "value": [1]}).to_csv(
        processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"], "value": [1]}).to_csv(
        processed_dir / "features.csv", index=False
    )
    specs = _write_rich_models(models_dir, generation="active")
    rich_source = _write_rich_manifest(
        tmp_path / "release-active",
        models_dir=models_dir,
        processed_dir=processed_dir,
        generation="active",
        specs=specs,
    )
    runtime_manifest = tmp_path / "runtime" / "manifest.json"
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "deployed-sha")
    initial = production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=runtime_manifest,
        source_manifest_path=rich_source,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        logistic_model_path=models_dir / "logistic_model.pkl",
        processed_dir=processed_dir,
    )

    contaminated = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    contaminated["training_source_git_sha"] = "legacy-sha"
    contaminated["git_sha"] = "legacy-sha"
    runtime_manifest.write_text(json.dumps(contaminated), encoding="utf-8")
    production_bundle.DEFAULT_MANIFEST_PATH.write_text(
        json.dumps(
            {
                "bundle_id": "legacy-image-bundle",
                "model_spec_name": "prod_spec_rich",
                "no_odds_model_spec_name": "prod_spec_rich_no_odds",
                "model_path": str(models_dir / "xgboost_model.pkl"),
                "no_odds_model_path": str(models_dir / "xgboost_no_odds_model.pkl"),
                "processed_dir": str(processed_dir),
                "snapshot_max_event_date": "2026-08-01",
                "built_at": "2026-03-23T00:00:00Z",
                "git_sha": "legacy-sha",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"event_date": ["2026-08-01"], "value": [2]}).to_csv(
        processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"], "value": [2]}).to_csv(
        processed_dir / "features.csv", index=False
    )

    refreshed = production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=runtime_manifest,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        logistic_model_path=models_dir / "logistic_model.pkl",
        processed_dir=processed_dir,
    )

    payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    assert payload["training_source_git_sha"] == "training-sha"
    assert payload["deployed_git_sha"] == "deployed-sha"
    assert payload["git_sha"] == "deployed-sha"
    assert payload["processed_fights_sha256"] != initial["processed_fights_sha256"]
    assert payload["processed_features_sha256"] != initial["processed_features_sha256"]
    assert payload["immutable_training_snapshot"] == contaminated[
        "immutable_training_snapshot"
    ]
    assert refreshed["training_source_git_sha"] == "training-sha"
    assert refreshed["model_hashes"] == initial["model_hashes"]


def test_reconcile_rejects_stale_rich_identity_without_authoritative_rich_source(
    tmp_path, monkeypatch
):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    pd.DataFrame({"event_date": ["2026-08-01"]}).to_csv(
        processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"]}).to_csv(
        processed_dir / "features.csv", index=False
    )
    specs = _write_rich_models(models_dir, generation="old")
    source = _write_rich_manifest(
        tmp_path / "release-old",
        models_dir=models_dir,
        processed_dir=processed_dir,
        generation="old",
        specs=specs,
    )
    runtime_manifest = tmp_path / "runtime" / "manifest.json"
    production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=runtime_manifest,
        source_manifest_path=source,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        logistic_model_path=models_dir / "logistic_model.pkl",
        processed_dir=processed_dir,
    )

    _write_rich_models(models_dir, generation="unattested-refit")
    with pytest.raises(
        production_bundle.ProductionBundleError,
        match="rich identity does not match the active primary model",
    ):
        production_bundle.reconcile_production_bundle_manifest(
            target_manifest_path=runtime_manifest,
            model_path=models_dir / "xgboost_model.pkl",
            no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
            logistic_model_path=models_dir / "logistic_model.pkl",
            processed_dir=processed_dir,
        )


def test_reconcile_rejects_partial_rich_source_manifest(tmp_path, monkeypatch):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    _write_rich_models(models_dir, generation="new")
    pd.DataFrame({"event_date": ["2026-08-01"]}).to_csv(
        processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"]}).to_csv(
        processed_dir / "features.csv", index=False
    )
    source = tmp_path / "partial-rich.json"
    source.write_text(
        json.dumps(
            {
                "manifest_version": 3,
                "staging_schema_version": 1,
                "bundle_id": "partial",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        production_bundle.ProductionBundleError,
        match="missing required object sections",
    ):
        production_bundle.reconcile_production_bundle_manifest(
            target_manifest_path=tmp_path / "runtime" / "manifest.json",
            source_manifest_path=source,
            model_path=models_dir / "xgboost_model.pkl",
            no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
            logistic_model_path=models_dir / "logistic_model.pkl",
            processed_dir=processed_dir,
        )


def test_reconcile_refuses_to_mutate_immutable_rich_release_manifest(
    tmp_path, monkeypatch
):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    pd.DataFrame({"event_date": ["2026-08-01"]}).to_csv(
        processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"]}).to_csv(
        processed_dir / "features.csv", index=False
    )
    specs = _write_rich_models(models_dir, generation="immutable")
    manifest = _write_rich_manifest(
        tmp_path / "immutable-release",
        models_dir=models_dir,
        processed_dir=processed_dir,
        generation="immutable",
        specs=specs,
    )
    before = manifest.read_bytes()

    with pytest.raises(
        production_bundle.ProductionBundleError,
        match="release manifests are immutable",
    ):
        production_bundle.reconcile_production_bundle_manifest(
            target_manifest_path=manifest,
            source_manifest_path=manifest,
            model_path=models_dir / "xgboost_model.pkl",
            no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
            logistic_model_path=models_dir / "logistic_model.pkl",
            processed_dir=processed_dir,
        )

    assert manifest.read_bytes() == before


def test_reconcile_production_bundle_manifest_drops_stale_identity_metadata(tmp_path, monkeypatch):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)

    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec_new")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_new_no_odds")
    pd.DataFrame({"event_date": ["2026-06-06"]}).to_csv(processed_dir / "fights_cleaned.csv", index=False)
    pd.DataFrame({"event_date": ["2026-06-06"]}).to_csv(processed_dir / "features.csv", index=False)

    stale_identity_metadata = {
        "as_of_date": "2026-05-29",
        "promoted_from": {
            "candidate_label": "old_candidate",
            "spec_path": "models/old_candidate/old_spec.json",
        },
        "promoted_alias_targets": {
            "primary_model": "models/xgboost_model.pkl",
            "spec_path": "models/old_spec.json",
        },
        "selection_basis": "old promotion evidence",
        "rollback_backup_dir": "models/backups/old_pre_promotion",
        "prior_promotion": {
            "bundle_id": "older-bundle",
        },
    }
    stale_manifest_payload = {
        "bundle_id": "bundle-old",
        "model_spec_name": "prod_spec_old",
        "no_odds_model_spec_name": "prod_spec_old_no_odds",
        "model_path": str(models_dir / "xgboost_model.pkl"),
        "no_odds_model_path": str(models_dir / "xgboost_no_odds_model.pkl"),
        "processed_dir": str(processed_dir),
        "snapshot_max_event_date": "2026-05-29",
        "built_at": "2026-05-30T00:00:00Z",
        "git_sha": "oldsha",
        "line": "ufc_live_production",
        "status": "promoted",
        "live_model_alias": "xgboost",
        "live_no_odds_alias": "xgboost_no_odds",
        **stale_identity_metadata,
    }

    runtime_manifest = tmp_path / "runtime" / "manifest.json"
    runtime_manifest.parent.mkdir(parents=True, exist_ok=True)
    runtime_manifest.write_text(json.dumps(stale_manifest_payload), encoding="utf-8")

    source_manifest = tmp_path / "models" / "source.json"
    source_manifest.write_text(json.dumps(stale_manifest_payload), encoding="utf-8")

    production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=runtime_manifest,
        source_manifest_path=source_manifest,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        processed_dir=processed_dir,
    )

    payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    assert payload["bundle_id"] == "ufc-production-20260606-prod_spec_new"
    assert payload["line"] == "ufc_live_production"
    assert payload["status"] == "promoted"
    assert payload["live_model_alias"] == "xgboost"
    assert payload["live_no_odds_alias"] == "xgboost_no_odds"
    for stale_key in stale_identity_metadata:
        assert stale_key not in payload


@pytest.mark.parametrize(
    ("source_built_at", "expected_built_at"),
    [
        ("2026-07-11T06:00:00Z", "2026-07-11T06:00:00Z"),
        ("2026-05-01T00:00:00Z", "2026-06-11T06:05:30Z"),
    ],
)
def test_reconcile_preserved_identity_adopts_only_newer_source_built_at(
    tmp_path, monkeypatch, source_built_at, expected_built_at
):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)

    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_no_odds")
    pd.DataFrame({"event_date": ["2026-06-27"]}).to_csv(processed_dir / "fights_cleaned.csv", index=False)
    pd.DataFrame({"event_date": ["2026-06-27"]}).to_csv(processed_dir / "features.csv", index=False)

    runtime_payload = {
        "bundle_id": "ufc-production-20260627-prod_spec",
        "model_spec_name": "prod_spec",
        "no_odds_model_spec_name": "prod_spec_no_odds",
        "model_path": str(models_dir / "xgboost_model.pkl"),
        "no_odds_model_path": str(models_dir / "xgboost_no_odds_model.pkl"),
        "processed_dir": str(processed_dir),
        "snapshot_max_event_date": "2026-06-27",
        "built_at": "2026-06-11T06:05:30Z",
        "git_sha": "oldsha",
    }
    runtime_manifest = tmp_path / "runtime" / "manifest.json"
    runtime_manifest.parent.mkdir(parents=True, exist_ok=True)
    runtime_manifest.write_text(json.dumps(runtime_payload), encoding="utf-8")

    source_manifest = tmp_path / "models" / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                **runtime_payload,
                "built_at": source_built_at,
                "git_sha": "newsha",
            }
        ),
        encoding="utf-8",
    )

    production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=runtime_manifest,
        source_manifest_path=source_manifest,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        processed_dir=processed_dir,
    )

    payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    assert payload["bundle_id"] == "ufc-production-20260627-prod_spec"
    assert payload["built_at"] == expected_built_at


def test_reconcile_model_hash_change_adopts_source_built_at_even_if_older(tmp_path, monkeypatch):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)

    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_no_odds")
    pd.DataFrame({"event_date": ["2026-06-27"]}).to_csv(processed_dir / "fights_cleaned.csv", index=False)
    pd.DataFrame({"event_date": ["2026-06-27"]}).to_csv(processed_dir / "features.csv", index=False)

    shared_payload = {
        "bundle_id": "ufc-production-20260627-prod_spec",
        "model_spec_name": "prod_spec",
        "no_odds_model_spec_name": "prod_spec_no_odds",
        "model_path": str(models_dir / "xgboost_model.pkl"),
        "no_odds_model_path": str(models_dir / "xgboost_no_odds_model.pkl"),
        "processed_dir": str(processed_dir),
        "snapshot_max_event_date": "2026-06-27",
        "git_sha": "sha",
    }
    runtime_manifest = tmp_path / "runtime" / "manifest.json"
    runtime_manifest.parent.mkdir(parents=True, exist_ok=True)
    runtime_manifest.write_text(
        json.dumps(
            {
                **shared_payload,
                "built_at": "2026-07-11T15:49:51Z",
                "model_sha256": "0" * 64,
                "no_odds_model_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    source_manifest = tmp_path / "models" / "source.json"
    source_manifest.write_text(
        json.dumps({**shared_payload, "built_at": "2026-06-11T06:05:30Z"}),
        encoding="utf-8",
    )

    production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=runtime_manifest,
        source_manifest_path=source_manifest,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        processed_dir=processed_dir,
    )

    payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    assert payload["built_at"] == "2026-06-11T06:05:30Z"
    assert payload["model_sha256"] == production_bundle._file_sha256(models_dir / "xgboost_model.pkl")
    assert payload["no_odds_model_sha256"] == production_bundle._file_sha256(
        models_dir / "xgboost_no_odds_model.pkl"
    )


def test_validate_production_bundle_rejects_model_hash_drift(tmp_path, monkeypatch):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)

    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_no_odds")
    pd.DataFrame({"event_date": ["2026-06-27"]}).to_csv(processed_dir / "fights_cleaned.csv", index=False)
    pd.DataFrame({"event_date": ["2026-06-27"]}).to_csv(processed_dir / "features.csv", index=False)

    manifest_path = tmp_path / "runtime" / "manifest.json"
    production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=manifest_path,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        processed_dir=processed_dir,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["model_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    bundle = production_bundle.load_production_bundle(manifest_path)
    with pytest.raises(production_bundle.ProductionBundleError, match="model hash mismatch"):
        production_bundle.validate_production_bundle(bundle)


def test_validate_production_bundle_rejects_logistic_model_hash_drift(tmp_path, monkeypatch):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)

    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_no_odds")
    logistic_path = models_dir / "logistic_model.pkl"
    _write_model(logistic_path, spec_name="prod_spec")
    pd.DataFrame({"event_date": ["2026-06-27"]}).to_csv(processed_dir / "fights_cleaned.csv", index=False)
    pd.DataFrame({"event_date": ["2026-06-27"]}).to_csv(processed_dir / "features.csv", index=False)

    manifest_path = tmp_path / "runtime" / "manifest.json"
    production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=manifest_path,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        logistic_model_path=logistic_path,
        processed_dir=processed_dir,
    )

    _write_model(logistic_path, spec_name="changed_logistic_spec")

    with pytest.raises(production_bundle.ProductionBundleError, match="logistic model hash mismatch"):
        production_bundle.validate_production_bundle(
            production_bundle.load_production_bundle(manifest_path)
        )


def test_reconcile_logistic_hash_change_does_not_preserve_bundle_identity(tmp_path, monkeypatch):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)

    primary_path = models_dir / "xgboost_model.pkl"
    no_odds_path = models_dir / "xgboost_no_odds_model.pkl"
    logistic_path = models_dir / "logistic_model.pkl"
    _write_model(primary_path, spec_name="prod_spec")
    _write_model(no_odds_path, spec_name="prod_spec_no_odds")
    _write_model(logistic_path, spec_name="prod_spec")
    pd.DataFrame({"event_date": ["2026-06-27"]}).to_csv(processed_dir / "fights_cleaned.csv", index=False)
    pd.DataFrame({"event_date": ["2026-06-27"]}).to_csv(processed_dir / "features.csv", index=False)

    shared_payload = {
        "model_spec_name": "prod_spec",
        "no_odds_model_spec_name": "prod_spec_no_odds",
        "model_path": str(primary_path),
        "no_odds_model_path": str(no_odds_path),
        "logistic_model_path": str(logistic_path),
        "processed_dir": str(processed_dir),
        "snapshot_max_event_date": "2026-06-27",
        "git_sha": "sha",
    }
    runtime_manifest = tmp_path / "runtime" / "manifest.json"
    runtime_manifest.parent.mkdir(parents=True, exist_ok=True)
    runtime_manifest.write_text(
        json.dumps(
            {
                **shared_payload,
                "bundle_id": "custom-old-bundle",
                "built_at": "2026-07-11T15:49:51Z",
                "model_sha256": production_bundle._file_sha256(primary_path),
                "no_odds_model_sha256": production_bundle._file_sha256(no_odds_path),
                "logistic_model_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    source_manifest = models_dir / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                **shared_payload,
                "bundle_id": "source-bundle",
                "built_at": "2026-06-11T06:05:30Z",
            }
        ),
        encoding="utf-8",
    )

    production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=runtime_manifest,
        source_manifest_path=source_manifest,
        model_path=primary_path,
        no_odds_model_path=no_odds_path,
        logistic_model_path=logistic_path,
        processed_dir=processed_dir,
    )

    payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    assert payload["bundle_id"] == "ufc-production-20260627-prod_spec"
    assert payload["built_at"] == "2026-06-11T06:05:30Z"
    assert payload["logistic_model_sha256"] == production_bundle._file_sha256(logistic_path)


def test_validate_production_bundle_rejects_processed_hash_drift(tmp_path, monkeypatch):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)

    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_no_odds")
    pd.DataFrame({"event_date": ["2026-03-21"]}).to_csv(processed_dir / "fights_cleaned.csv", index=False)
    pd.DataFrame({"event_date": ["2026-03-21"]}).to_csv(processed_dir / "features.csv", index=False)

    manifest_path = tmp_path / "runtime" / "manifest.json"
    production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=manifest_path,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        processed_dir=processed_dir,
    )

    pd.DataFrame({"event_date": ["2026-03-21"], "changed": [1]}).to_csv(processed_dir / "features.csv", index=False)

    with pytest.raises(production_bundle.ProductionBundleError, match="processed features snapshot hash mismatch"):
        production_bundle.validate_production_bundle(
            production_bundle.load_production_bundle(manifest_path)
        )


def test_runtime_manifest_needs_source_bootstrap_is_false_for_valid_runtime_snapshot_even_when_source_bundle_changes(tmp_path, monkeypatch):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)

    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec_new")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_new_no_odds")
    pd.DataFrame({"event_date": ["2026-03-21", "2026-03-22"]}).to_csv(processed_dir / "fights_cleaned.csv", index=False)
    pd.DataFrame({"event_date": ["2026-03-21", "2026-03-22"]}).to_csv(processed_dir / "features.csv", index=False)

    source_manifest = tmp_path / "source.json"
    target_manifest = tmp_path / "target.json"
    source_manifest.write_text(
        json.dumps(
            {
                "bundle_id": "bundle-new",
                "model_spec_name": "prod_spec_new",
                "model_path": "models/xgboost_model.pkl",
                "no_odds_model_path": "models/xgboost_no_odds_model.pkl",
                "processed_dir": "data/processed",
                "snapshot_max_event_date": "2026-03-21",
                "built_at": "2026-03-23T00:00:00Z",
                "git_sha": "abc123",
            }
        ),
        encoding="utf-8",
    )
    production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=target_manifest,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        processed_dir=processed_dir,
    )

    assert production_bundle.runtime_manifest_needs_source_bootstrap(
        target_manifest_path=target_manifest,
        source_manifest_path=source_manifest,
    ) is False


def test_runtime_manifest_needs_source_bootstrap_is_false_for_matching_valid_bundle(tmp_path, monkeypatch):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)

    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_no_odds")
    pd.DataFrame({"event_date": ["2026-03-21"]}).to_csv(processed_dir / "fights_cleaned.csv", index=False)
    pd.DataFrame({"event_date": ["2026-03-21"]}).to_csv(processed_dir / "features.csv", index=False)

    source_manifest = tmp_path / "source.json"
    runtime_manifest = tmp_path / "runtime" / "manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "bundle_id": "bundle-same",
                "model_spec_name": "prod_spec",
                "no_odds_model_spec_name": "prod_spec_no_odds",
                "model_path": str(models_dir / "xgboost_model.pkl"),
                "no_odds_model_path": str(models_dir / "xgboost_no_odds_model.pkl"),
                "processed_dir": str(processed_dir),
                "snapshot_max_event_date": "2026-03-21",
                "built_at": "2026-03-23T00:00:00Z",
                "git_sha": "abc123",
            }
        ),
        encoding="utf-8",
    )
    production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=runtime_manifest,
        source_manifest_path=source_manifest,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        processed_dir=processed_dir,
    )

    assert production_bundle.runtime_manifest_needs_source_bootstrap(
        target_manifest_path=runtime_manifest,
        source_manifest_path=source_manifest,
    ) is False


def test_reconcile_production_bundle_manifest_regenerates_bundle_id_when_model_identity_changes(tmp_path, monkeypatch):
    models_dir, processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)

    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec_new")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_new_no_odds")
    pd.DataFrame({"event_date": ["2026-03-22"]}).to_csv(processed_dir / "fights_cleaned.csv", index=False)
    pd.DataFrame({"event_date": ["2026-03-22"]}).to_csv(processed_dir / "features.csv", index=False)

    runtime_manifest = tmp_path / "runtime" / "manifest.json"
    runtime_manifest.parent.mkdir(parents=True, exist_ok=True)
    runtime_manifest.write_text(
        json.dumps(
            {
                "bundle_id": "bundle-old",
                "model_spec_name": "prod_spec_old",
                "no_odds_model_spec_name": "prod_spec_old_no_odds",
                "model_path": str(models_dir / "xgboost_model.pkl"),
                "no_odds_model_path": str(models_dir / "xgboost_no_odds_model.pkl"),
                "processed_dir": str(processed_dir),
                "snapshot_max_event_date": "2026-03-21",
                "built_at": "2026-03-20T00:00:00Z",
                "git_sha": "oldsha",
            }
        ),
        encoding="utf-8",
    )

    source_manifest = tmp_path / "models" / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "bundle_id": "bundle-source",
                "model_spec_name": "prod_spec_new",
                "no_odds_model_spec_name": "prod_spec_new_no_odds",
                "model_path": str(models_dir / "xgboost_model.pkl"),
                "no_odds_model_path": str(models_dir / "xgboost_no_odds_model.pkl"),
                "processed_dir": str(processed_dir),
                "snapshot_max_event_date": "2026-03-21",
                "built_at": "2026-03-23T00:00:00Z",
                "git_sha": "newsha",
                "selection_basis": "source metadata",
            }
        ),
        encoding="utf-8",
    )

    summary = production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=runtime_manifest,
        source_manifest_path=source_manifest,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        processed_dir=processed_dir,
    )

    payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    assert payload["bundle_id"] == "ufc-production-20260322-prod_spec_new"
    assert payload["built_at"] == "2026-03-23T00:00:00Z"
    assert payload["selection_basis"] == "source metadata"
    assert summary["bundle_id"] == payload["bundle_id"]


def test_bootstrap_runtime_production_bundle_preserves_valid_runtime_snapshot_on_redeploy(tmp_path, monkeypatch):
    models_dir, _processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    source_processed_dir = tmp_path / "image" / "processed"
    target_processed_dir = tmp_path / "runtime" / "processed"
    source_processed_dir.mkdir(parents=True)
    target_processed_dir.mkdir(parents=True)

    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec_new")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_new_no_odds")

    pd.DataFrame({"event_date": ["2026-03-20"]}).to_csv(source_processed_dir / "fights_cleaned.csv", index=False)
    pd.DataFrame({"event_date": ["2026-03-20"]}).to_csv(source_processed_dir / "features.csv", index=False)
    pd.DataFrame({"event_date": ["2026-03-22"]}).to_csv(target_processed_dir / "fights_cleaned.csv", index=False)
    pd.DataFrame({"event_date": ["2026-03-22"], "runtime_only": [1]}).to_csv(
        target_processed_dir / "features.csv",
        index=False,
    )

    source_manifest = tmp_path / "models" / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "bundle_id": "bundle-source",
                "model_spec_name": "prod_spec_new",
                "no_odds_model_spec_name": "prod_spec_new_no_odds",
                "model_path": str(models_dir / "xgboost_model.pkl"),
                "no_odds_model_path": str(models_dir / "xgboost_no_odds_model.pkl"),
                "processed_dir": str(source_processed_dir),
                "snapshot_max_event_date": "2026-03-20",
                "built_at": "2026-03-23T00:00:00Z",
                "git_sha": "newsha",
            }
        ),
        encoding="utf-8",
    )

    runtime_manifest = tmp_path / "runtime" / "manifest.json"
    production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=runtime_manifest,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        processed_dir=target_processed_dir,
    )

    summary = bootstrap_runtime_bundle.bootstrap_runtime_production_bundle(
        target_manifest=runtime_manifest,
        source_manifest=source_manifest,
        source_processed_dir=source_processed_dir,
        target_processed_dir=target_processed_dir,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
    )

    preserved_features = pd.read_csv(target_processed_dir / "features.csv")
    assert summary["bootstrap_action"] == "reused_existing_runtime_bundle"
    assert summary["processed_snapshot_max_event_date"] == "2026-03-22"
    assert summary["model_spec_name"] == "prod_spec_new"
    assert "runtime_only" in preserved_features.columns


def test_bootstrap_preserves_manifest_addressed_refresh_generation_on_restart(
    tmp_path,
    monkeypatch,
):
    models_dir, _processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    source_processed_dir = tmp_path / "image" / "processed"
    target_processed_dir = tmp_path / "runtime" / "processed"
    refresh_generation = target_processed_dir / "ufc_refresh_generations" / "refresh-1"
    source_processed_dir.mkdir(parents=True)
    target_processed_dir.mkdir(parents=True)
    refresh_generation.mkdir(parents=True)

    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec_new")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_new_no_odds")

    pd.DataFrame({"event_date": ["2026-03-20"], "source_only": [1]}).to_csv(
        source_processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-03-20"], "source_only": [1]}).to_csv(
        source_processed_dir / "features.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-03-22"], "canonical_only": [1]}).to_csv(
        target_processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-03-22"], "canonical_only": [1]}).to_csv(
        target_processed_dir / "features.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-03-24"], "generation_only": [1]}).to_csv(
        refresh_generation / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-03-24"], "generation_only": [1]}).to_csv(
        refresh_generation / "features.csv", index=False
    )

    source_manifest = tmp_path / "models" / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "bundle_id": "bundle-source",
                "model_spec_name": "prod_spec_new",
                "no_odds_model_spec_name": "prod_spec_new_no_odds",
                "model_path": str(models_dir / "xgboost_model.pkl"),
                "no_odds_model_path": str(models_dir / "xgboost_no_odds_model.pkl"),
                "processed_dir": str(source_processed_dir),
                "snapshot_max_event_date": "2026-03-20",
                "built_at": "2026-03-23T00:00:00Z",
                "git_sha": "newsha",
            }
        ),
        encoding="utf-8",
    )
    runtime_manifest = tmp_path / "runtime" / "manifest.json"
    production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=runtime_manifest,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        processed_dir=refresh_generation,
    )

    summary = bootstrap_runtime_bundle.bootstrap_runtime_production_bundle(
        target_manifest=runtime_manifest,
        source_manifest=source_manifest,
        source_processed_dir=source_processed_dir,
        target_processed_dir=target_processed_dir,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
    )

    payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    assert summary["bootstrap_action"] == "preserved_manifest_addressed_runtime_bundle"
    assert summary["processed_snapshot_max_event_date"] == "2026-03-24"
    assert payload["processed_dir"] == str(refresh_generation.resolve())
    assert "generation_only" in pd.read_csv(refresh_generation / "features.csv").columns
    assert "canonical_only" in pd.read_csv(target_processed_dir / "features.csv").columns


def test_explicit_rich_release_activation_preserves_refreshed_generation_on_restart(
    tmp_path,
    monkeypatch,
):
    models_dir, _processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    release_root = tmp_path / "rich-release"
    source_processed_dir = release_root / "processed"
    target_processed_dir = tmp_path / "runtime" / "processed"
    refresh_generation = target_processed_dir / "ufc_refresh_generations" / "refresh-1"
    source_processed_dir.mkdir(parents=True)
    refresh_generation.mkdir(parents=True)

    pd.DataFrame({"event_date": ["2026-08-01"], "release_only": [1]}).to_csv(
        source_processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"], "release_only": [1]}).to_csv(
        source_processed_dir / "features.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-08"], "refresh_only": [1]}).to_csv(
        refresh_generation / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-08"], "refresh_only": [1]}).to_csv(
        refresh_generation / "features.csv", index=False
    )

    specs = _write_rich_models(models_dir, generation="restart")
    source_manifest = _write_rich_manifest(
        release_root,
        models_dir=models_dir,
        processed_dir=source_processed_dir,
        generation="restart",
        specs=specs,
    )
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_payload["rich_release_root"] = str(release_root.resolve())
    source_manifest.write_text(json.dumps(source_payload), encoding="utf-8")

    runtime_manifest = tmp_path / "runtime" / "manifest.json"
    production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=runtime_manifest,
        source_manifest_path=source_manifest,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        logistic_model_path=models_dir / "logistic_model.pkl",
        processed_dir=source_processed_dir,
    )
    production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=runtime_manifest,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        logistic_model_path=models_dir / "logistic_model.pkl",
        processed_dir=refresh_generation,
    )
    refreshed_payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    assert refreshed_payload["bundle_id"] != source_payload["bundle_id"]
    assert refreshed_payload["rich_release_root"] == source_payload["rich_release_root"]

    summary = bootstrap_runtime_bundle.bootstrap_runtime_production_bundle(
        target_manifest=runtime_manifest,
        source_manifest=source_manifest,
        source_processed_dir=source_processed_dir,
        target_processed_dir=target_processed_dir,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        logistic_model_path=models_dir / "logistic_model.pkl",
        activate_source_generation=True,
    )

    runtime_payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    assert summary["bootstrap_action"] == "preserved_manifest_addressed_runtime_bundle"
    assert summary["processed_snapshot_max_event_date"] == "2026-08-08"
    assert runtime_payload["processed_dir"] == str(refresh_generation.resolve())
    assert "refresh_only" in pd.read_csv(refresh_generation / "features.csv").columns
    assert not target_processed_dir.joinpath("features.csv").exists()


def test_bootstrap_runtime_production_bundle_promotes_newer_source_snapshot(tmp_path, monkeypatch):
    models_dir, _processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    source_processed_dir = tmp_path / "image" / "processed"
    target_processed_dir = tmp_path / "runtime" / "processed"
    source_processed_dir.mkdir(parents=True)
    target_processed_dir.mkdir(parents=True)

    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec_new")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_new_no_odds")

    pd.DataFrame({"event_date": ["2026-05-29"], "source_only": [1]}).to_csv(
        source_processed_dir / "fights_cleaned.csv",
        index=False,
    )
    pd.DataFrame({"event_date": ["2026-05-29"], "source_only": [1]}).to_csv(
        source_processed_dir / "features.csv",
        index=False,
    )
    pd.DataFrame({"event_date": ["2026-05-16"], "runtime_only": [1]}).to_csv(
        target_processed_dir / "fights_cleaned.csv",
        index=False,
    )
    pd.DataFrame({"event_date": ["2026-05-16"], "runtime_only": [1]}).to_csv(
        target_processed_dir / "features.csv",
        index=False,
    )

    source_manifest = tmp_path / "models" / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "bundle_id": "ufc-production-20260529-prod_spec_new",
                "model_spec_name": "prod_spec_new",
                "no_odds_model_spec_name": "prod_spec_new_no_odds",
                "model_path": str(models_dir / "xgboost_model.pkl"),
                "no_odds_model_path": str(models_dir / "xgboost_no_odds_model.pkl"),
                "processed_dir": str(source_processed_dir),
                "snapshot_max_event_date": "2026-05-29",
                "built_at": "2026-05-30T00:32:22Z",
                "git_sha": "newsha",
                **production_bundle.get_processed_snapshot_fingerprints(source_processed_dir),
                **production_bundle.get_model_artifact_fingerprints(
                    models_dir / "xgboost_model.pkl",
                    models_dir / "xgboost_no_odds_model.pkl",
                ),
            }
        ),
        encoding="utf-8",
    )

    runtime_manifest = tmp_path / "runtime" / "manifest.json"
    production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=runtime_manifest,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        processed_dir=target_processed_dir,
    )

    summary = bootstrap_runtime_bundle.bootstrap_runtime_production_bundle(
        target_manifest=runtime_manifest,
        source_manifest=source_manifest,
        source_processed_dir=source_processed_dir,
        target_processed_dir=target_processed_dir,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
    )

    promoted_features = pd.read_csv(target_processed_dir / "features.csv")
    assert summary["bootstrap_action"] == "promoted_source_bundle"
    assert summary["processed_snapshot_max_event_date"] == "2026-05-29"
    assert summary["built_at"] == "2026-05-30T00:32:22Z"
    assert "source_only" in promoted_features.columns
    assert "runtime_only" not in promoted_features.columns


def test_bootstrap_propagates_complete_rich_release_identity(tmp_path, monkeypatch):
    models_dir, _processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    release_root = tmp_path / "rich-release"
    source_processed_dir = release_root / "processed"
    target_processed_dir = tmp_path / "runtime" / "processed"
    source_processed_dir.mkdir(parents=True)
    pd.DataFrame({"event_date": ["2026-08-01"]}).to_csv(
        source_processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"]}).to_csv(
        source_processed_dir / "features.csv", index=False
    )
    specs = _write_rich_models(models_dir, generation="bootstrap")
    source_manifest = _write_rich_manifest(
        release_root,
        models_dir=models_dir,
        processed_dir=source_processed_dir,
        generation="bootstrap",
        specs=specs,
        scheduled_bfo_lineage=True,
    )
    source_manifest_copy = release_root / "provenance" / "source_staging_manifest.json"
    source_manifest_copy.parent.mkdir(exist_ok=True)
    source_manifest_copy.write_bytes(source_manifest.read_bytes())
    target_manifest = tmp_path / "runtime" / "manifest.json"

    summary = bootstrap_runtime_bundle.bootstrap_runtime_production_bundle(
        target_manifest=target_manifest,
        source_manifest=source_manifest,
        source_processed_dir=source_processed_dir,
        target_processed_dir=target_processed_dir,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        logistic_model_path=models_dir / "logistic_model.pkl",
    )

    payload = json.loads(target_manifest.read_text(encoding="utf-8"))
    assert summary["bootstrap_action"] == "promoted_source_bundle"
    assert summary["rich_manifest_validated"] is True
    assert payload["manifest_version"] == 3
    assert payload["source_identity"]["generation"] == "bootstrap"
    assert payload["selection_evidence"]["generation"] == "bootstrap"
    assert payload["rich_release_root"] == str(release_root.resolve())
    assert summary["model_hashes"]["primary_sha256"] == payload["model_sha256"]
    assert summary["rich_release_id"] == release_root.name
    assert summary["installed_manifest_path"] == str(source_manifest)
    assert summary["installed_manifest_sha256"] == production_bundle._file_sha256(
        source_manifest
    )
    assert summary["source_manifest_sha256"] == production_bundle._file_sha256(
        source_manifest_copy
    )
    assert (
        summary["immutable_training_fights_sha256"]
        == payload["processed_fights_sha256"]
    )
    assert (
        summary["immutable_training_features_sha256"]
        == payload["processed_features_sha256"]
    )
    assert summary["immutable_training_snapshot_max_event_date"] == "2026-08-01"
    assert summary["independent_audit_fights_canonical_sha256"] == (
        production_bundle._canonical_text_file_sha256(
            release_root / "provenance/independent_audit_snapshot/fights_cleaned.csv"
        )
    )
    assert summary["independent_audit_features_canonical_sha256"] == (
        production_bundle._canonical_text_file_sha256(
            release_root / "provenance/independent_audit_snapshot/features.csv"
        )
    )
    lineage_record = payload["raw_input_provenance"]["scheduled_bfo_lineage"]
    assert (
        summary["scheduled_bfo_lineage_manifest_sha256"]
        == lineage_record["manifest_sha256"]
    )
    assert summary["scheduled_bfo_lineage_batch_count"] == 1


def test_scheduled_rich_bundle_requires_bfo_lineage_package(tmp_path, monkeypatch):
    manifest_path = _write_rich_validation_manifest(
        tmp_path,
        monkeypatch,
        scheduled_bfo_lineage=False,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["scheduled_refit_policy"] = _scheduled_refit_policy_identity()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        production_bundle.ProductionBundleError,
        match="missing its BFO lineage package",
    ):
        production_bundle.validate_production_bundle(
            production_bundle.load_production_bundle(manifest_path)
        )


def test_scheduled_rich_bundle_rejects_tampered_bfo_lineage_file(
    tmp_path,
    monkeypatch,
):
    manifest_path = _write_rich_validation_manifest(
        tmp_path,
        monkeypatch,
        scheduled_bfo_lineage=True,
    )
    lineage_csv = next(
        (manifest_path.parent / "provenance" / "bfo_lineage" / "batches").glob(
            "*.csv"
        )
    )
    lineage_csv.write_bytes(lineage_csv.read_bytes() + b"Tampered Fighter,+100\n")

    with pytest.raises(
        production_bundle.ProductionBundleError,
        match="BFO lineage batch 0 CSV identity is invalid",
    ):
        production_bundle.validate_production_bundle(
            production_bundle.load_production_bundle(manifest_path)
        )


def test_scheduled_refit_policy_identity_is_strict_and_copy_safe():
    policy = _scheduled_refit_policy_identity()

    validated = production_bundle._validated_scheduled_refit_policy(
        {"scheduled_refit_policy": policy}
    )
    assert validated == policy
    assert validated is not policy

    malformed = dict(policy)
    malformed["unknown"] = "7" * 64
    with pytest.raises(
        production_bundle.ProductionBundleError,
        match="missing or unknown",
    ):
        production_bundle._validated_scheduled_refit_policy(
            {"scheduled_refit_policy": malformed}
        )


def test_bootstrap_runtime_production_bundle_adopts_runtime_snapshot_when_manifest_is_stale(tmp_path, monkeypatch):
    models_dir, _processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    source_processed_dir = tmp_path / "image" / "processed"
    target_processed_dir = tmp_path / "runtime" / "processed"
    source_processed_dir.mkdir(parents=True)
    target_processed_dir.mkdir(parents=True)

    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec_new")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_new_no_odds")

    pd.DataFrame({"event_date": ["2026-03-20"]}).to_csv(source_processed_dir / "fights_cleaned.csv", index=False)
    pd.DataFrame({"event_date": ["2026-03-20"]}).to_csv(source_processed_dir / "features.csv", index=False)
    pd.DataFrame({"event_date": ["2026-03-21"]}).to_csv(target_processed_dir / "fights_cleaned.csv", index=False)
    pd.DataFrame({"event_date": ["2026-03-21"]}).to_csv(target_processed_dir / "features.csv", index=False)

    source_manifest = tmp_path / "models" / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "bundle_id": "bundle-source",
                "model_spec_name": "prod_spec_new",
                "no_odds_model_spec_name": "prod_spec_new_no_odds",
                "model_path": str(models_dir / "xgboost_model.pkl"),
                "no_odds_model_path": str(models_dir / "xgboost_no_odds_model.pkl"),
                "processed_dir": str(source_processed_dir),
                "snapshot_max_event_date": "2026-03-20",
                "built_at": "2026-03-23T00:00:00Z",
                "git_sha": "newsha",
            }
        ),
        encoding="utf-8",
    )

    runtime_manifest = tmp_path / "runtime" / "manifest.json"
    production_bundle.reconcile_production_bundle_manifest(
        target_manifest_path=runtime_manifest,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        processed_dir=target_processed_dir,
    )

    pd.DataFrame({"event_date": ["2026-03-22"]}).to_csv(target_processed_dir / "fights_cleaned.csv", index=False)
    pd.DataFrame({"event_date": ["2026-03-22"], "runtime_only": [1]}).to_csv(
        target_processed_dir / "features.csv",
        index=False,
    )

    summary = bootstrap_runtime_bundle.bootstrap_runtime_production_bundle(
        target_manifest=runtime_manifest,
        source_manifest=source_manifest,
        source_processed_dir=source_processed_dir,
        target_processed_dir=target_processed_dir,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
    )

    preserved_features = pd.read_csv(target_processed_dir / "features.csv")
    assert summary["bootstrap_action"] == "adopted_existing_runtime_snapshot"
    assert summary["processed_snapshot_max_event_date"] == "2026-03-22"
    assert "runtime_only" in preserved_features.columns


def test_bootstrap_runtime_production_bundle_preserves_newer_runtime_snapshot_when_manifest_is_missing(tmp_path, monkeypatch):
    models_dir, _processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    source_processed_dir = tmp_path / "image" / "processed"
    target_processed_dir = tmp_path / "runtime" / "processed"
    source_processed_dir.mkdir(parents=True)
    target_processed_dir.mkdir(parents=True)

    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec_new")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_new_no_odds")

    pd.DataFrame({"event_date": ["2026-03-20"]}).to_csv(source_processed_dir / "fights_cleaned.csv", index=False)
    pd.DataFrame({"event_date": ["2026-03-20"]}).to_csv(source_processed_dir / "features.csv", index=False)
    pd.DataFrame({"event_date": ["2026-03-23"]}).to_csv(target_processed_dir / "fights_cleaned.csv", index=False)
    pd.DataFrame({"event_date": ["2026-03-23"], "runtime_only": [1]}).to_csv(
        target_processed_dir / "features.csv",
        index=False,
    )

    source_manifest = tmp_path / "models" / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "bundle_id": "bundle-source",
                "model_spec_name": "prod_spec_new",
                "no_odds_model_spec_name": "prod_spec_new_no_odds",
                "model_path": str(models_dir / "xgboost_model.pkl"),
                "no_odds_model_path": str(models_dir / "xgboost_no_odds_model.pkl"),
                "processed_dir": str(source_processed_dir),
                "snapshot_max_event_date": "2026-03-20",
                "built_at": "2026-03-23T00:00:00Z",
                "git_sha": "newsha",
            }
        ),
        encoding="utf-8",
    )

    runtime_manifest = tmp_path / "runtime" / "manifest.json"
    summary = bootstrap_runtime_bundle.bootstrap_runtime_production_bundle(
        target_manifest=runtime_manifest,
        source_manifest=source_manifest,
        source_processed_dir=source_processed_dir,
        target_processed_dir=target_processed_dir,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
    )

    preserved_features = pd.read_csv(target_processed_dir / "features.csv")
    assert summary["bootstrap_action"] == "adopted_existing_runtime_snapshot"
    assert summary["processed_snapshot_max_event_date"] == "2026-03-23"
    assert "runtime_only" in preserved_features.columns


def test_bootstrap_equal_date_promotes_newer_approved_source_generation(tmp_path, monkeypatch):
    models_dir, _processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    source_processed_dir = tmp_path / "image" / "processed"
    target_processed_dir = tmp_path / "runtime" / "processed"
    source_processed_dir.mkdir(parents=True)
    target_processed_dir.mkdir(parents=True)
    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec_new")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_new_no_odds")
    _write_model(models_dir / "logistic_model.pkl", spec_name="prod_spec_new")

    pd.DataFrame({"event_date": ["2026-08-01"], "source_only": [1]}).to_csv(
        source_processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"], "source_only": [1]}).to_csv(
        source_processed_dir / "features.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"], "runtime_only": [1]}).to_csv(
        target_processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"], "runtime_only": [1]}).to_csv(
        target_processed_dir / "features.csv", index=False
    )

    source_manifest = tmp_path / "models" / "source.json"
    runtime_manifest = tmp_path / "runtime" / "manifest.json"
    runtime_manifest.parent.mkdir(parents=True, exist_ok=True)
    source_payload = _pinned_manifest_payload(
        bundle_id="new-image-generation",
        models_dir=models_dir,
        processed_dir=source_processed_dir,
        built_at="2026-08-05T12:00:00Z",
    )
    source_payload.update(
        production_bundle.get_model_artifact_fingerprints(
            models_dir / "xgboost_model.pkl",
            models_dir / "xgboost_no_odds_model.pkl",
            models_dir / "logistic_model.pkl",
        )
    )
    source_manifest.write_text(
        json.dumps(source_payload),
        encoding="utf-8",
    )
    runtime_manifest.write_text(
        json.dumps(
            _pinned_manifest_payload(
                bundle_id="old-runtime-generation",
                models_dir=models_dir,
                processed_dir=target_processed_dir,
                built_at="2026-08-04T12:00:00Z",
            )
        ),
        encoding="utf-8",
    )

    summary = bootstrap_runtime_bundle.bootstrap_runtime_production_bundle(
        target_manifest=runtime_manifest,
        source_manifest=source_manifest,
        source_processed_dir=source_processed_dir,
        target_processed_dir=target_processed_dir,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        logistic_model_path=models_dir / "logistic_model.pkl",
    )

    assert summary["bootstrap_action"] == "promoted_source_bundle"
    assert "source_only" in pd.read_csv(target_processed_dir / "features.csv").columns


def test_bootstrap_equal_date_preserves_same_generation_runtime_enrichment(tmp_path, monkeypatch):
    models_dir, _processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    source_processed_dir = tmp_path / "image" / "processed"
    target_processed_dir = tmp_path / "runtime" / "processed"
    source_processed_dir.mkdir(parents=True)
    target_processed_dir.mkdir(parents=True)
    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec_new")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_new_no_odds")

    pd.DataFrame({"event_date": ["2026-08-01"], "source_only": [1]}).to_csv(
        source_processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"], "source_only": [1]}).to_csv(
        source_processed_dir / "features.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"], "runtime_only": [1]}).to_csv(
        target_processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"], "runtime_only": [1]}).to_csv(
        target_processed_dir / "features.csv", index=False
    )

    source_manifest = tmp_path / "models" / "source.json"
    runtime_manifest = tmp_path / "runtime" / "manifest.json"
    runtime_manifest.parent.mkdir(parents=True, exist_ok=True)
    source_manifest.write_text(
        json.dumps(
            _pinned_manifest_payload(
                bundle_id="same-generation",
                models_dir=models_dir,
                processed_dir=source_processed_dir,
                built_at="2026-08-04T12:00:00Z",
            )
        ),
        encoding="utf-8",
    )
    runtime_manifest.write_text(
        json.dumps(
            _pinned_manifest_payload(
                bundle_id="same-generation",
                models_dir=models_dir,
                processed_dir=target_processed_dir,
                built_at="2026-08-05T12:00:00Z",
            )
        ),
        encoding="utf-8",
    )

    summary = bootstrap_runtime_bundle.bootstrap_runtime_production_bundle(
        target_manifest=runtime_manifest,
        source_manifest=source_manifest,
        source_processed_dir=source_processed_dir,
        target_processed_dir=target_processed_dir,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
    )

    assert summary["bootstrap_action"] == "reused_existing_runtime_bundle"
    assert "runtime_only" in pd.read_csv(target_processed_dir / "features.csv").columns


def test_bootstrap_equal_date_conflict_fails_closed_without_generation_identity(tmp_path, monkeypatch):
    models_dir, _processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    source_processed_dir = tmp_path / "image" / "processed"
    target_processed_dir = tmp_path / "runtime" / "processed"
    source_processed_dir.mkdir(parents=True)
    target_processed_dir.mkdir(parents=True)
    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec_new")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_new_no_odds")
    pd.DataFrame({"event_date": ["2026-08-01"], "source_only": [1]}).to_csv(
        source_processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"], "source_only": [1]}).to_csv(
        source_processed_dir / "features.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"], "runtime_only": [1]}).to_csv(
        target_processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"], "runtime_only": [1]}).to_csv(
        target_processed_dir / "features.csv", index=False
    )
    source_manifest = tmp_path / "models" / "source.json"
    source_manifest.write_text(
        json.dumps(
            _pinned_manifest_payload(
                bundle_id="approved-source",
                models_dir=models_dir,
                processed_dir=source_processed_dir,
                built_at="2026-08-05T12:00:00Z",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(production_bundle.ProductionBundleError, match="Runtime manifest is missing built_at"):
        bootstrap_runtime_bundle.bootstrap_runtime_production_bundle(
            target_manifest=tmp_path / "runtime" / "missing-manifest.json",
            source_manifest=source_manifest,
            source_processed_dir=source_processed_dir,
            target_processed_dir=target_processed_dir,
            model_path=models_dir / "xgboost_model.pkl",
            no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        )


def test_bootstrap_rejects_source_manifest_processed_hash_mismatch(tmp_path, monkeypatch):
    models_dir, _processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    source_processed_dir = tmp_path / "image" / "processed"
    target_processed_dir = tmp_path / "runtime" / "processed"
    source_processed_dir.mkdir(parents=True)
    target_processed_dir.mkdir(parents=True)
    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec_new")
    _write_model(models_dir / "xgboost_no_odds_model.pkl", spec_name="prod_spec_new_no_odds")
    pd.DataFrame({"event_date": ["2026-08-02"]}).to_csv(
        source_processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-02"]}).to_csv(
        source_processed_dir / "features.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"]}).to_csv(
        target_processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"]}).to_csv(
        target_processed_dir / "features.csv", index=False
    )
    source_payload = _pinned_manifest_payload(
        bundle_id="corrupt-source",
        models_dir=models_dir,
        processed_dir=source_processed_dir,
        built_at="2026-08-05T12:00:00Z",
    )
    source_payload["processed_features_sha256"] = "0" * 64
    source_manifest = tmp_path / "models" / "source.json"
    source_manifest.write_text(json.dumps(source_payload), encoding="utf-8")

    with pytest.raises(production_bundle.ProductionBundleError, match="processed_features_sha256"):
        bootstrap_runtime_bundle.bootstrap_runtime_production_bundle(
            target_manifest=tmp_path / "runtime" / "manifest.json",
            source_manifest=source_manifest,
            source_processed_dir=source_processed_dir,
            target_processed_dir=target_processed_dir,
            model_path=models_dir / "xgboost_model.pkl",
            no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        )


def test_explicit_release_activation_restores_older_complete_legacy_generation(
    tmp_path, monkeypatch
):
    models_dir, _processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    source_processed_dir = tmp_path / "release" / "processed"
    target_processed_dir = tmp_path / "runtime" / "processed"
    source_processed_dir.mkdir(parents=True)
    target_processed_dir.mkdir(parents=True)

    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec_new")
    _write_model(
        models_dir / "xgboost_no_odds_model.pkl",
        spec_name="prod_spec_new_no_odds",
    )
    _write_model(models_dir / "logistic_model.pkl", spec_name="prod_spec_new")

    pd.DataFrame({"event_date": ["2026-08-01"], "rollback_only": [1]}).to_csv(
        source_processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"], "rollback_only": [1]}).to_csv(
        source_processed_dir / "features.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-08"], "candidate_only": [1]}).to_csv(
        target_processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-08"], "candidate_only": [1]}).to_csv(
        target_processed_dir / "features.csv", index=False
    )

    source_payload = {
        **_pinned_manifest_payload(
            bundle_id="legacy-rollback-release",
            models_dir=models_dir,
            processed_dir=source_processed_dir,
            built_at="2026-08-04T12:00:00Z",
        ),
        **production_bundle.get_model_artifact_fingerprints(
            models_dir / "xgboost_model.pkl",
            models_dir / "xgboost_no_odds_model.pkl",
            models_dir / "logistic_model.pkl",
        ),
        "logistic_model_path": str(models_dir / "logistic_model.pkl"),
    }
    source_manifest = tmp_path / "release" / "manifest.json"
    source_manifest.write_text(json.dumps(source_payload), encoding="utf-8")

    # This deliberately resembles stale rich state from the generation being
    # rolled back. It is incomplete and must not be merged into the explicitly
    # selected, fully hash-pinned predecessor.
    target_manifest = tmp_path / "runtime" / "manifest.json"
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    target_manifest.write_text(
        json.dumps(
            {
                "manifest_version": 3,
                "staging_schema_version": 1,
                "bundle_id": "candidate-generation",
                "model_spec_name": "prod_spec_new",
                "model_path": str(models_dir / "xgboost_model.pkl"),
                "no_odds_model_path": str(models_dir / "xgboost_no_odds_model.pkl"),
                "processed_dir": str(target_processed_dir),
                "snapshot_max_event_date": "2026-08-08",
                "built_at": "2026-08-06T12:00:00Z",
                "git_sha": "candidate-sha",
                "model_sha256": "a" * 64,
                "no_odds_model_sha256": "b" * 64,
                "processed_fights_sha256": "c" * 64,
                "processed_features_sha256": "d" * 64,
                "source_identity": {},
            }
        ),
        encoding="utf-8",
    )

    summary = bootstrap_runtime_bundle.bootstrap_runtime_production_bundle(
        target_manifest=target_manifest,
        source_manifest=source_manifest,
        source_processed_dir=source_processed_dir,
        target_processed_dir=target_processed_dir,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        logistic_model_path=models_dir / "logistic_model.pkl",
        activate_source_generation=True,
    )

    runtime_payload = json.loads(target_manifest.read_text(encoding="utf-8"))
    restored_features = pd.read_csv(target_processed_dir / "features.csv")
    assert summary["bootstrap_action"] == "promoted_source_bundle"
    assert summary["bundle_id"] == "legacy-rollback-release"
    assert runtime_payload["manifest_version"] == 1
    assert "staging_schema_version" not in runtime_payload
    assert runtime_payload["logistic_model_sha256"] == source_payload[
        "logistic_model_sha256"
    ]
    assert "rollback_only" in restored_features.columns
    assert "candidate_only" not in restored_features.columns


def test_explicit_release_activation_preserves_newer_lookup_for_same_generation(
    tmp_path, monkeypatch
):
    models_dir, _processed_dir = _configure_bundle_paths(monkeypatch, tmp_path)
    source_processed_dir = tmp_path / "release" / "processed"
    target_processed_dir = tmp_path / "runtime" / "processed"
    source_processed_dir.mkdir(parents=True)
    target_processed_dir.mkdir(parents=True)

    _write_model(models_dir / "xgboost_model.pkl", spec_name="prod_spec_new")
    _write_model(
        models_dir / "xgboost_no_odds_model.pkl",
        spec_name="prod_spec_new_no_odds",
    )
    pd.DataFrame({"event_date": ["2026-08-01"], "release_only": [1]}).to_csv(
        source_processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-01"], "release_only": [1]}).to_csv(
        source_processed_dir / "features.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-08"], "runtime_only": [1]}).to_csv(
        target_processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame({"event_date": ["2026-08-08"], "runtime_only": [1]}).to_csv(
        target_processed_dir / "features.csv", index=False
    )

    source_payload = _pinned_manifest_payload(
        bundle_id="same-release",
        models_dir=models_dir,
        processed_dir=source_processed_dir,
        built_at="2026-08-04T12:00:00Z",
    )
    source_manifest = tmp_path / "release" / "manifest.json"
    source_manifest.write_text(json.dumps(source_payload), encoding="utf-8")
    target_payload = _pinned_manifest_payload(
        bundle_id="same-release",
        models_dir=models_dir,
        processed_dir=target_processed_dir,
        built_at="2026-08-05T12:00:00Z",
    )
    target_manifest = tmp_path / "runtime" / "manifest.json"
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    target_manifest.write_text(json.dumps(target_payload), encoding="utf-8")

    summary = bootstrap_runtime_bundle.bootstrap_runtime_production_bundle(
        target_manifest=target_manifest,
        source_manifest=source_manifest,
        source_processed_dir=source_processed_dir,
        target_processed_dir=target_processed_dir,
        model_path=models_dir / "xgboost_model.pkl",
        no_odds_model_path=models_dir / "xgboost_no_odds_model.pkl",
        activate_source_generation=True,
    )

    preserved = pd.read_csv(target_processed_dir / "features.csv")
    assert summary["bootstrap_action"] == "reused_existing_runtime_bundle"
    assert "runtime_only" in preserved.columns
    assert "release_only" not in preserved.columns
