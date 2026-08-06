import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.model import production_bundle
from src.model.training_spec import NamedModelTrainingSpec


def _write_model(path: Path, spec_payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "feature_cols": list(spec_payload["feature_cols"]),
            "training_spec": deepcopy(spec_payload),
            "model": "stub-model",
        },
        path,
    )


def _build_staged_bundle(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path, dict]:
    canonical_models_dir = tmp_path / "canonical" / "models"
    canonical_processed_dir = tmp_path / "canonical" / "processed"
    candidate_models_dir = tmp_path / "models" / "candidates" / "candidate_run"
    candidate_processed_dir = tmp_path / "data" / "processed" / "candidates" / "candidate_run"
    candidate_models_dir.mkdir(parents=True)
    candidate_processed_dir.mkdir(parents=True)

    monkeypatch.setattr(production_bundle, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(production_bundle, "MODELS_DIR", canonical_models_dir)
    monkeypatch.setattr(production_bundle, "PROCESSED_DATA_DIR", canonical_processed_dir)
    production_bundle._cached_snapshot_max_event_date.cache_clear()

    registered_spec = NamedModelTrainingSpec(
        name="strict_staged_spec",
        description="Strict staged candidate",
        feature_cols=["demo_feature", "a_implied_prob", "b_implied_prob"],
        calibration_cv="temporal_holdout",
        odds_noise_seed=42,
    )
    fitted_spec = replace(
        registered_spec,
        git_hash="source-sha",
        trained_at="2026-08-05T12:00:00+00:00",
    )
    primary_spec = asdict(fitted_spec)
    no_odds_spec = production_bundle._expected_no_odds_spec_payload(primary_spec)

    primary_path = candidate_models_dir / "xgboost_model.pkl"
    no_odds_path = candidate_models_dir / "xgboost_no_odds_model.pkl"
    logistic_path = candidate_models_dir / "logistic_model.pkl"
    _write_model(primary_path, primary_spec)
    _write_model(no_odds_path, no_odds_spec)
    _write_model(logistic_path, primary_spec)
    (candidate_models_dir / "strict_staged_spec_spec.json").write_text(
        json.dumps(primary_spec, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame({"event_date": ["2026-08-01"]}).to_csv(
        candidate_processed_dir / "fights_cleaned.csv", index=False
    )
    pd.DataFrame(
        {
            "event_date": ["2026-08-01"],
            "demo_feature": [1.0],
            "a_implied_prob": [0.6],
            "b_implied_prob": [0.4],
        }
    ).to_csv(candidate_processed_dir / "features.csv", index=False)

    payload = {
        "manifest_version": 2,
        "bundle_id": "strict-staged-bundle",
        "model_spec_name": "strict_staged_spec",
        "no_odds_model_spec_name": "strict_staged_spec_no_odds",
        "model_path": str(primary_path),
        "no_odds_model_path": str(no_odds_path),
        "logistic_model_path": str(logistic_path),
        "processed_dir": str(candidate_processed_dir),
        "snapshot_max_event_date": "2026-08-01",
        "built_at": "2026-08-05T12:00:00+00:00",
        "git_sha": "source-sha",
        **production_bundle.get_model_artifact_fingerprints(
            primary_path, no_odds_path, logistic_path
        ),
        **production_bundle.get_processed_snapshot_fingerprints(
            candidate_processed_dir
        ),
    }
    manifest_path = candidate_models_dir / "staging_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    import src.model.training_spec as training_spec_module

    monkeypatch.setattr(
        training_spec_module,
        "resolve_named_training_spec",
        lambda name: registered_spec
        if name == registered_spec.name
        else (_ for _ in ()).throw(ValueError(f"Unknown spec {name}")),
    )
    return manifest_path, candidate_models_dir, candidate_processed_dir, payload


def _validate_staged(
    manifest_path: Path,
    candidate_models_dir: Path,
    candidate_processed_dir: Path,
) -> dict:
    return production_bundle.validate_production_bundle(
        production_bundle.load_production_bundle(manifest_path),
        expected_models_dir=candidate_models_dir,
        expected_processed_dir=candidate_processed_dir,
    )


def test_strict_staged_bundle_validates_candidate_without_weakening_canonical_paths(
    tmp_path, monkeypatch
):
    manifest_path, candidate_models_dir, candidate_processed_dir, _ = (
        _build_staged_bundle(tmp_path, monkeypatch)
    )

    summary = _validate_staged(
        manifest_path, candidate_models_dir, candidate_processed_dir
    )

    assert summary["validation_scope"] == "staged"
    assert summary["expected_models_dir"] == str(candidate_models_dir.resolve())
    assert summary["expected_processed_dir"] == str(candidate_processed_dir.resolve())
    assert summary["registered_training_spec_name"] == "strict_staged_spec"
    assert summary["embedded_training_spec_sha256"] == summary[
        "embedded_logistic_training_spec_sha256"
    ]
    assert summary["embedded_no_odds_training_spec_sha256"] != summary[
        "embedded_training_spec_sha256"
    ]

    with pytest.raises(
        production_bundle.ProductionBundleError,
        match="primary alias path mismatch",
    ):
        production_bundle.validate_production_bundle(
            production_bundle.load_production_bundle(manifest_path)
        )


def test_staged_validation_requires_both_exact_candidate_directories(
    tmp_path, monkeypatch
):
    manifest_path, candidate_models_dir, candidate_processed_dir, _ = (
        _build_staged_bundle(tmp_path, monkeypatch)
    )

    with pytest.raises(
        production_bundle.ProductionBundleError,
        match="requires both expected_models_dir and expected_processed_dir",
    ):
        production_bundle.validate_production_bundle(
            production_bundle.load_production_bundle(manifest_path),
            expected_models_dir=candidate_models_dir,
        )

    with pytest.raises(
        production_bundle.ProductionBundleError,
        match="models directory must be isolated from MODELS_DIR",
    ):
        production_bundle.validate_production_bundle(
            production_bundle.load_production_bundle(manifest_path),
            expected_models_dir=production_bundle.MODELS_DIR,
            expected_processed_dir=candidate_processed_dir,
        )

    with pytest.raises(
        production_bundle.ProductionBundleError,
        match="primary alias path mismatch",
    ):
        production_bundle.validate_production_bundle(
            production_bundle.load_production_bundle(manifest_path),
            expected_models_dir=candidate_models_dir.parent / "other_run",
            expected_processed_dir=candidate_processed_dir,
        )


def test_staged_validation_rejects_logistic_full_spec_drift(tmp_path, monkeypatch):
    manifest_path, candidate_models_dir, candidate_processed_dir, payload = (
        _build_staged_bundle(tmp_path, monkeypatch)
    )
    logistic_path = candidate_models_dir / "logistic_model.pkl"
    logistic_result = joblib.load(logistic_path)
    logistic_result["training_spec"]["odds_noise_seed"] = 99
    joblib.dump(logistic_result, logistic_path)
    payload["logistic_model_sha256"] = production_bundle._file_sha256(logistic_path)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(
        production_bundle.ProductionBundleError,
        match="logistic training_spec does not exactly match",
    ):
        _validate_staged(manifest_path, candidate_models_dir, candidate_processed_dir)


def test_staged_validation_rejects_no_odds_drift_beyond_documented_variant(
    tmp_path, monkeypatch
):
    manifest_path, candidate_models_dir, candidate_processed_dir, payload = (
        _build_staged_bundle(tmp_path, monkeypatch)
    )
    no_odds_path = candidate_models_dir / "xgboost_no_odds_model.pkl"
    no_odds_result = joblib.load(no_odds_path)
    no_odds_result["training_spec"]["time_decay_half_life"] = 999
    joblib.dump(no_odds_result, no_odds_path)
    payload["no_odds_model_sha256"] = production_bundle._file_sha256(no_odds_path)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(
        production_bundle.ProductionBundleError,
        match="beyond its documented no-odds variant fields",
    ):
        _validate_staged(manifest_path, candidate_models_dir, candidate_processed_dir)


def test_staged_validation_requires_complete_pinned_fingerprints(tmp_path, monkeypatch):
    manifest_path, candidate_models_dir, candidate_processed_dir, payload = (
        _build_staged_bundle(tmp_path, monkeypatch)
    )
    payload.pop("logistic_model_sha256")
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(
        production_bundle.ProductionBundleError,
        match="missing required fingerprints.*logistic_model_sha256",
    ):
        _validate_staged(manifest_path, candidate_models_dir, candidate_processed_dir)


def test_staged_validation_rejects_saved_spec_drift(tmp_path, monkeypatch):
    manifest_path, candidate_models_dir, candidate_processed_dir, _ = (
        _build_staged_bundle(tmp_path, monkeypatch)
    )
    spec_path = candidate_models_dir / "strict_staged_spec_spec.json"
    saved_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    saved_spec["description"] = "drifted sidecar"
    spec_path.write_text(json.dumps(saved_spec, indent=2), encoding="utf-8")

    with pytest.raises(
        production_bundle.ProductionBundleError,
        match="saved training spec does not exactly match",
    ):
        _validate_staged(manifest_path, candidate_models_dir, candidate_processed_dir)


def test_staged_validation_rejects_incomplete_primary_spec_payload(tmp_path, monkeypatch):
    manifest_path, candidate_models_dir, candidate_processed_dir, payload = (
        _build_staged_bundle(tmp_path, monkeypatch)
    )
    primary_path = candidate_models_dir / "xgboost_model.pkl"
    logistic_path = candidate_models_dir / "logistic_model.pkl"
    no_odds_path = candidate_models_dir / "xgboost_no_odds_model.pkl"
    primary_result = joblib.load(primary_path)
    primary_result["training_spec"].pop("trained_at")
    joblib.dump(primary_result, primary_path)
    logistic_result = joblib.load(logistic_path)
    logistic_result["training_spec"].pop("trained_at")
    joblib.dump(logistic_result, logistic_path)
    no_odds_result = joblib.load(no_odds_path)
    no_odds_result["training_spec"].pop("trained_at")
    joblib.dump(no_odds_result, no_odds_path)
    sidecar_path = candidate_models_dir / "strict_staged_spec_spec.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar.pop("trained_at")
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    payload.update(
        production_bundle.get_model_artifact_fingerprints(
            primary_path, no_odds_path, logistic_path
        )
    )
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(
        production_bundle.ProductionBundleError,
        match="primary training_spec is incomplete",
    ):
        _validate_staged(manifest_path, candidate_models_dir, candidate_processed_dir)
