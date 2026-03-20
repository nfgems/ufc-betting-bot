import json
from pathlib import Path

import joblib

from scripts.repair_no_odds_training_spec import _default_paths_from_manifest, _repair_artifact


def test_default_paths_from_manifest_prefers_active_no_odds_artifacts(tmp_path):
    manifest_path = tmp_path / "current_production_model.json"
    manifest_path.write_text(
        json.dumps(
            {
                "promoted_alias_targets": {"no_odds_model": "models/xgboost_no_odds_model.pkl"},
                "promoted_from": {
                    "no_odds_model": "models/candidates/full_live_contract_v5_fullfit/xgboost_no_odds_model.pkl"
                },
            }
        ),
        encoding="utf-8",
    )

    paths = _default_paths_from_manifest(manifest_path)

    assert paths == [
        Path("models/xgboost_no_odds_model.pkl"),
        Path("models/candidates/full_live_contract_v5_fullfit/xgboost_no_odds_model.pkl"),
    ]


def test_repair_artifact_updates_legacy_no_odds_training_spec(tmp_path):
    artifact_path = tmp_path / "xgboost_no_odds_model.pkl"
    payload = {
        "feature_cols": ["core_feature"],
        "training_spec": {
            "name": "legacy_promoted_spec",
            "description": "Legacy promoted spec",
            "feature_cols": ["core_feature", "a_implied_prob", "b_implied_prob", "diff_implied_prob"],
        },
        "model": "stub-model",
    }
    joblib.dump(payload, artifact_path)

    changed = _repair_artifact(artifact_path)
    repaired = joblib.load(artifact_path)

    assert changed is True
    assert repaired["training_spec"]["feature_cols"] == ["core_feature"]
    assert repaired["training_spec"]["name"] == "legacy_promoted_spec_no_odds"
    assert repaired["training_spec"]["description"] == "Legacy promoted spec (no-odds variant)"


def test_repair_artifact_is_idempotent_for_up_to_date_no_odds_metadata(tmp_path):
    artifact_path = tmp_path / "xgboost_no_odds_model.pkl"
    payload = {
        "feature_cols": ["core_feature"],
        "training_spec": {
            "name": "legacy_promoted_spec_no_odds",
            "description": "Legacy promoted spec (no-odds variant)",
            "feature_cols": ["core_feature"],
        },
        "model": "stub-model",
    }
    joblib.dump(payload, artifact_path)

    changed = _repair_artifact(artifact_path)
    repaired = joblib.load(artifact_path)

    assert changed is False
    assert repaired["training_spec"]["feature_cols"] == ["core_feature"]
