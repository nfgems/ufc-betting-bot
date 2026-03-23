import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

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
    assert payload["bundle_id"] == "bundle-source"
    assert payload["processed_dir"] == str(processed_dir.resolve(strict=False))
    assert payload["processed_fights_sha256"]
    assert payload["processed_features_sha256"]
    assert payload["manifest_updated_at"]
    assert summary["processed_fights_sha256"] == payload["processed_fights_sha256"]
    assert summary["processed_features_sha256"] == payload["processed_features_sha256"]


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


def test_runtime_manifest_needs_source_bootstrap_when_bundle_identity_changes(tmp_path, monkeypatch):
    _configure_bundle_paths(monkeypatch, tmp_path)

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
    target_manifest.write_text(
        json.dumps(
            {
                "bundle_id": "bundle-old",
                "model_spec_name": "prod_spec_old",
                "model_path": "models/xgboost_model.pkl",
                "no_odds_model_path": "models/xgboost_no_odds_model.pkl",
                "processed_dir": "data/processed",
                "snapshot_max_event_date": "2026-03-20",
                "built_at": "2026-03-22T00:00:00Z",
                "git_sha": "oldsha",
            }
        ),
        encoding="utf-8",
    )

    assert production_bundle.runtime_manifest_needs_source_bootstrap(
        target_manifest_path=target_manifest,
        source_manifest_path=source_manifest,
    ) is True


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
