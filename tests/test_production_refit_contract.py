from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from scripts import check_production_refit_contract as gate
from scripts import bfo_lineage
from src.model.training_spec import resolve_named_training_spec


POLICY_PATH = Path("config/scheduled_refit_policy_v1.json")
POLICY_V2_PATH = Path("config/scheduled_refit_policy_v2.json")
NOW = datetime(2026, 8, 6, tzinfo=timezone.utc)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_features(
    path: Path,
    *,
    event_date: str = "2026-08-01",
    spec_name: str = "full_live_contract_v6_durability_corrected_20260805_fullfit",
    policy_v2: bool = False,
) -> None:
    spec = resolve_named_training_spec(spec_name)
    row = {column: 0.0 for column in spec.feature_cols}
    row["event_date"] = event_date
    if policy_v2:
        row.update(
            {
                "a_implied_prob": 0.55,
                "b_implied_prob": 0.45,
                "diff_implied_prob": 0.10,
                "model_odds_source_kind": "odds_api",
                "model_odds_source_file": "fixture.csv",
                "model_odds_observed_at": "2026-07-31T00:00:00Z",
                "model_odds_commence_time": "2026-08-01T00:00:00Z",
                "model_odds_hours_to_start": 24.0,
                "model_odds_verified_prefight": True,
            }
        )
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


def _candidate_fixture(
    tmp_path: Path,
    *,
    policy_path: Path = POLICY_V2_PATH,
    final_policy: bool = True,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> tuple[Path, Path, Path]:
    confirmation_binding = None
    if policy_path == POLICY_V2_PATH and final_policy:
        policy, confirmation_binding, _claim_path = _confirmed_child_fixture(
            tmp_path
        )
        policy_path = tmp_path / "config/scheduled_refit_policy_v2.json"
        if monkeypatch is not None:
            monkeypatch.setattr(
                gate,
                "validate_performance_confirmation",
                lambda *_args, **_kwargs: ([], confirmation_binding),
            )
    else:
        policy = gate.load_policy(policy_path)
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
    _write_features(
        feature_path,
        spec_name=fullfit.name,
        policy_v2=policy["schema_version"] == 2,
    )
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
            "policy_schema_version": policy["schema_version"],
            "policy_id": policy["policy_id"],
            "sha256": gate.file_sha256(policy_path),
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
    if policy["schema_version"] == 2:
        policy_confirmation = policy.get("performance_confirmation", {})
        binding_hashes = {
            "receipt": "1" * 64,
            "confirmation": policy_confirmation.get(
                "confirmation_result_sha256", "2" * 64
            ),
            "strategy": policy_confirmation.get(
                "strategy_config_sha256", "3" * 64
            ),
            "confirmation_input": "4" * 64,
            "feature_contract": "5" * 64,
            "confirmation_features": "6" * 64,
            "source_features": "7" * 64,
            "odds_inventory": "8" * 64,
            "source_fingerprint": "9" * 64,
            "source_inventory": "a" * 64,
            "source_inventory_artifact": "b" * 64,
            "environment_artifact": "c" * 64,
            "environment_payload": "d" * 64,
            "evaluation_protocol": "e" * 64,
            "performance_evidence": "f" * 64,
        }
        manifest["training_input_evidence"] = {
            "schema_version": 1,
            "artifact_receipt_sha256": "0" * 64,
            "artifact_roles": ["logistic", "no_odds", "primary"],
            "policy_id": policy["policy_id"],
            "policy_sha256": gate.file_sha256(policy_path),
            "fullfit_spec_name": fullfit.name,
            "entry_offset_days": float(policy["evaluation"]["entry_offset_days"]),
            "prepared_features_sha256": features["sha256"],
            "prepared_features_bytes": features["bytes"],
            "source_fights_path": str(
                (root / "processed/fights_cleaned.csv").resolve()
            ),
            "source_fights_sha256": fights["sha256"],
            "source_fights_bytes": fights["bytes"],
            "final_track_c_pass_receipt_sha256": binding_hashes["receipt"],
            "performance_confirmation_result_sha256": binding_hashes[
                "confirmation"
            ],
            "confirmed_strategy_config_sha256": binding_hashes["strategy"],
            "confirmation_evaluation_input_value_sha256": binding_hashes[
                "confirmation_input"
            ],
            "feature_contract_count": policy["contract"]["feature_count"],
            "feature_contract_sha256": binding_hashes["feature_contract"],
            "confirmation_features_value_sha256": binding_hashes[
                "confirmation_features"
            ],
            "confirmation_source_features_sha256": binding_hashes[
                "source_features"
            ],
            "confirmation_odds_source_inventory_sha256": binding_hashes[
                "odds_inventory"
            ],
            "confirmation_source_fingerprint": binding_hashes[
                "source_fingerprint"
            ],
            "confirmation_source_inventory_sha256": binding_hashes[
                "source_inventory"
            ],
            "confirmation_source_inventory_artifact_sha256": binding_hashes[
                "source_inventory_artifact"
            ],
            "confirmation_environment_artifact_sha256": binding_hashes[
                "environment_artifact"
            ],
            "confirmation_environment_payload_sha256": binding_hashes[
                "environment_payload"
            ],
            "confirmation_evaluation_protocol_sha256": binding_hashes[
                "evaluation_protocol"
            ],
            "row_count": 1,
            "rows_with_verified_t_minus_entry": 1,
            "rows_missing_t_minus_entry": 0,
            "prepared_odds_columns": [
                "a_implied_prob",
                "b_implied_prob",
                "diff_implied_prob",
            ],
            "provenance_columns": [
                "model_odds_source_kind",
                "model_odds_source_file",
                "model_odds_observed_at",
                "model_odds_commence_time",
                "model_odds_hours_to_start",
                "model_odds_verified_prefight",
            ],
        }
        manifest["performance_confirmation"] = {
            "schema_version": 1,
            "policy_sha256": gate.file_sha256(policy_path),
            "confirmation_result_sha256": binding_hashes["confirmation"],
            "final_track_c_pass_receipt_sha256": binding_hashes["receipt"],
            "evaluation_spec_name": evaluation.name,
            "evaluation_spec_payload_sha256": policy["contract"][
                "evaluation_spec_payload_sha256"
            ],
            "fullfit_spec_name": fullfit.name,
            "fullfit_spec_payload_sha256": policy["contract"][
                "fullfit_spec_payload_sha256"
            ],
            "strategy_config_sha256": binding_hashes["strategy"],
            "dataset_fights_sha256": fights["sha256"],
            "features_sha256": features["sha256"],
            "features_value_sha256": binding_hashes["confirmation_features"],
            "feature_contract_sha256": binding_hashes["feature_contract"],
            "confirmation_evaluation_input_value_sha256": binding_hashes[
                "confirmation_input"
            ],
            "evaluation_protocol_sha256": binding_hashes["evaluation_protocol"],
            "odds_source_inventory_sha256": binding_hashes["odds_inventory"],
            "source_fingerprint": binding_hashes["source_fingerprint"],
            "source_inventory_sha256": binding_hashes["source_inventory"],
            "source_inventory_artifact_sha256": binding_hashes[
                "source_inventory_artifact"
            ],
            "environment_artifact_sha256": binding_hashes[
                "environment_artifact"
            ],
            "environment_payload_sha256": binding_hashes["environment_payload"],
            "performance_evidence_aggregate_sha256": binding_hashes[
                "performance_evidence"
            ],
        }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    return policy_path, manifest_path, readyz_path


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


def test_method_contract_allows_registered_fit_descendant_with_exact_base_semantics():
    base = resolve_named_training_spec("full_live_contract_v6_integrity_205")
    descendant = replace(
        base,
        name="unit_confirmed_descendant",
        description="confirmed fit descendant",
        calibration_method="sigmoid",
        xgb_params={**base.xgb_params, "max_depth": 7},
    )

    assert gate._validate_method_contract_descendant(
        descendant,
        base,
        label="evaluation",
        expected_model_seed=42,
        expected_odds_noise_seed=42,
    ) == []

    changed_contract = replace(
        descendant,
        feature_cols=[*descendant.feature_cols[:-1]],
    )
    errors = gate._validate_method_contract_descendant(
        changed_contract,
        base,
        label="evaluation",
        expected_model_seed=42,
        expected_odds_noise_seed=42,
    )
    assert any("feature columns" in error for error in errors)


def _confirmed_child_fixture(tmp_path: Path, parent_mutator=None):
    parent = gate.load_policy(POLICY_V2_PATH)
    parent_evidence = tmp_path / parent["baseline"]["evidence_path"]
    parent_evidence.parent.mkdir(parents=True, exist_ok=True)
    parent_evidence.write_bytes(
        (Path.cwd() / parent["baseline"]["evidence_path"]).read_bytes()
    )
    if parent_mutator is not None:
        parent_mutator(parent, parent_evidence)
    parent_path = tmp_path / "evidence/policies/pre_winner_policy.json"
    _write_json(parent_path, parent)
    parent_sha = gate.file_sha256(parent_path)

    claim_key = "confirmation-identity-test"
    claim_rel = (
        f"evidence/confirmation_claims/{claim_key}/"
        "confirmation_evaluation_claim.json"
    )
    result_rel = (
        f"evidence/confirmation_claims/{claim_key}/"
        "confirmation_evaluation_result.json"
    )
    claim_path = tmp_path / claim_rel
    _write_json(claim_path, {"schema_version": 1, "global_claim_key": claim_key})
    claim_sha = gate.file_sha256(claim_path)

    strategy = {
        "name": "confirmed_production",
        "s_min_edge": parent["evaluation"]["min_edge"],
        "s_blend_weight": parent["evaluation"]["blend_weight"],
        "s_model_agreement_min_edge": parent["evaluation"][
            "model_agreement_min_edge"
        ],
        "c_min_model_prob": parent["evaluation"]["min_model_probability"],
        "c_min_no_odds_prob": 0.50,
        "c_share": 0.50,
        "c_bet_fraction": 0.05,
        "c_max_bet_fraction": 0.08,
        "c_min_edge": 0.0,
        "c_max_decimal_odds": parent["evaluation"]["max_decimal_odds"],
        "m_enabled": False,
        "m_min_model_prob": 0.65,
        "m_min_no_odds_prob": 0.50,
        "m_bet_fraction": 0.03,
        "m_share": 0.30,
    }
    result_path = tmp_path / result_rel
    result = {
        "schema_version": 1,
        "global_claim_key": claim_key,
        "claim_path": claim_rel,
        "claim_sha256": claim_sha,
        "policy_sha256": parent_sha,
    }
    _write_json(result_path, result)

    child = json.loads(json.dumps(parent))
    baseline_evidence_rel = "evidence/track_c/final_child/track_c_summary.csv"
    baseline_evidence = tmp_path / baseline_evidence_rel
    baseline_evidence.parent.mkdir(parents=True, exist_ok=True)
    baseline_evidence.write_text("final-policy track-c evidence\n", encoding="utf-8")
    child["baseline"].update(
        {
            "comparison_role": "confirmed_winner_track_c_baseline",
            "evidence_path": baseline_evidence_rel,
            "evidence_sha256": gate.file_sha256(baseline_evidence),
            "strategy_total_bets": max(
                child["baseline"]["strategy_total_bets"],
                child["health_limits"]["minimum_strategy_bets"],
            ),
            "strategy_roi": 0.10,
            "strategy_total_profit": 25.0,
            "strategy_avg_clv": 0.01,
            "strategy_max_drawdown_pct": 0.20,
        }
    )
    binding = {
        "package_id": "confirmed-package",
        "package_contract_sha256": "a" * 64,
        "model_spec_name": child["contract"]["evaluation_spec_name"],
        "model_spec_payload_sha256": child["contract"][
            "evaluation_spec_payload_sha256"
        ],
        "fullfit_model_spec_name": child["contract"]["fullfit_spec_name"],
        "fullfit_model_spec_payload_sha256": child["contract"][
            "fullfit_spec_payload_sha256"
        ],
        "strategy_config": strategy,
        "strategy_config_sha256": gate.canonical_json_sha256(strategy),
        "runtime_invariants": {"near_miss_enabled": False},
        "policy_sha256": parent_sha,
    }
    child["performance_confirmation"] = {
        "schema_version": 1,
        "state": "final_track_c_bound",
        "parent_policy_path": parent_path.relative_to(tmp_path).as_posix(),
        "parent_policy_sha256": parent_sha,
        "confirmation_result_path": result_rel,
        "confirmation_result_sha256": gate.file_sha256(result_path),
        "global_claim_key": claim_key,
        "global_claim_path": claim_rel,
        "global_claim_sha256": claim_sha,
        "package_id": binding["package_id"],
        "candidate_package_contract_sha256": binding["package_contract_sha256"],
        "evaluation_spec_name": binding["model_spec_name"],
        "evaluation_spec_payload_sha256": binding["model_spec_payload_sha256"],
        "fullfit_spec_name": binding["fullfit_model_spec_name"],
        "fullfit_spec_payload_sha256": binding[
            "fullfit_model_spec_payload_sha256"
        ],
        "strategy_config": strategy,
        "strategy_config_sha256": binding["strategy_config_sha256"],
    }
    child_path = tmp_path / "config/scheduled_refit_policy_v2.json"
    _write_json(child_path, child)
    loaded_child = gate.load_policy(child_path)
    return loaded_child, binding, claim_path


def test_confirmed_child_requires_exact_durable_parent_pass_and_strategy(tmp_path: Path):
    child, binding, _claim_path = _confirmed_child_fixture(tmp_path)

    errors, validated = gate.validate_performance_confirmation(
        child,
        repo_root=tmp_path,
        confirmation_validator=lambda _path: binding,
    )

    assert errors == []
    assert validated == binding


def test_confirmed_child_keeps_parent_s_floors_when_c_has_distinct_null_cap(
    tmp_path: Path,
):
    child, binding, _claim_path = _confirmed_child_fixture(tmp_path)
    parent = gate.load_policy(
        tmp_path / child["performance_confirmation"]["parent_policy_path"]
    )
    strategy = dict(child["performance_confirmation"]["strategy_config"])
    strategy["c_min_model_prob"] = 0.65
    strategy["c_max_decimal_odds"] = None
    strategy_sha = gate.canonical_json_sha256(strategy)
    child["performance_confirmation"]["strategy_config"] = strategy
    child["performance_confirmation"]["strategy_config_sha256"] = strategy_sha
    binding = {
        **binding,
        "strategy_config": strategy,
        "strategy_config_sha256": strategy_sha,
    }

    errors, validated = gate.validate_performance_confirmation(
        child,
        repo_root=tmp_path,
        confirmation_validator=lambda _path: binding,
    )

    assert errors == []
    assert validated == binding
    assert child["evaluation"]["min_model_probability"] == parent["evaluation"][
        "min_model_probability"
    ]
    assert child["evaluation"]["max_decimal_odds"] == parent["evaluation"][
        "max_decimal_odds"
    ]


def test_confirmed_child_rejects_stale_parent_baseline_protocol(tmp_path: Path):
    """DIR-AUD-P2-024: a parent baseline CSV embedding a stale protocol blocks
    child validation even when the parent policy binds its exact file bytes."""

    def _stale_protocol(parent: dict, evidence_path: Path) -> None:
        embedded = str(parent["baseline"]["evidence_protocol_sha256"]).encode(
            "ascii"
        )
        original = evidence_path.read_bytes()
        stale = original.replace(embedded, b"e" * 64)
        assert stale != original
        evidence_path.write_bytes(stale)
        parent["baseline"]["evidence_sha256"] = gate.file_sha256(evidence_path)

    child, binding, _claim_path = _confirmed_child_fixture(
        tmp_path, parent_mutator=_stale_protocol
    )

    errors, _validated = gate.validate_performance_confirmation(
        child,
        repo_root=tmp_path,
        confirmation_validator=lambda _path: binding,
    )

    assert any(
        "parent policy baseline evidence is stale" in error
        and "protocol_sha256" in error
        for error in errors
    )


def test_confirmed_child_rejects_parent_baseline_evidence_byte_drift(tmp_path: Path):
    """DIR-AUD-P2-024: parent baseline CSV bytes drifting after preservation
    block child validation."""
    child, binding, _claim_path = _confirmed_child_fixture(tmp_path)
    parent = gate.load_policy(
        tmp_path / child["performance_confirmation"]["parent_policy_path"]
    )
    evidence_path = tmp_path / parent["baseline"]["evidence_path"]
    evidence_path.write_bytes(evidence_path.read_bytes() + b"drift\n")

    errors, _validated = gate.validate_performance_confirmation(
        child,
        repo_root=tmp_path,
        confirmation_validator=lambda _path: binding,
    )

    assert any(
        "parent policy baseline evidence is stale" in error
        and "SHA-256 does not match" in error
        for error in errors
    )


def test_confirmed_child_fails_closed_when_global_claim_bytes_change(tmp_path: Path):
    child, binding, claim_path = _confirmed_child_fixture(tmp_path)
    claim_path.write_text("tampered\n", encoding="utf-8")

    errors, validated = gate.validate_performance_confirmation(
        child,
        repo_root=tmp_path,
        confirmation_validator=lambda _path: binding,
    )

    assert validated is None
    assert any("global confirmation claim hash changed" in error for error in errors)


def test_bootstrap_winner_policy_can_run_track_c_but_keeps_parent_baseline(
    tmp_path: Path,
):
    child, binding, _claim_path = _confirmed_child_fixture(tmp_path)
    parent_path = tmp_path / child["performance_confirmation"]["parent_policy_path"]
    parent = gate.load_policy(parent_path)
    parent_evidence = tmp_path / parent["baseline"]["evidence_path"]
    parent_evidence.parent.mkdir(parents=True, exist_ok=True)
    parent_evidence.write_bytes(
        (Path.cwd() / parent["baseline"]["evidence_path"]).read_bytes()
    )
    child["performance_confirmation"]["state"] = "bootstrap_pending_track_c"
    child["baseline"] = json.loads(json.dumps(parent["baseline"]))

    errors, validated = gate.validate_performance_confirmation(
        child,
        repo_root=tmp_path,
        confirmation_validator=lambda _path: binding,
    )

    assert errors == []
    assert validated == binding

    child["baseline"]["strategy_roi"] = 0.01
    errors, _validated = gate.validate_performance_confirmation(
        child,
        repo_root=tmp_path,
        confirmation_validator=lambda _path: binding,
    )
    assert any("exact frozen parent baseline" in error for error in errors)


def test_confirmed_child_rejects_parent_snapshot_outside_evidence(tmp_path: Path):
    child, binding, _claim_path = _confirmed_child_fixture(tmp_path)
    child["performance_confirmation"]["parent_policy_path"] = (
        "config/scheduled_refit_policy_v2.json"
    )

    errors, validated = gate.validate_performance_confirmation(
        child,
        repo_root=tmp_path,
        confirmation_validator=lambda _path: binding,
    )

    assert validated is None
    assert any("must be preserved below evidence" in error for error in errors)


def _final_track_c_receipt_fixture(tmp_path: Path, monkeypatch) -> dict:
    child, binding, _claim_path = _confirmed_child_fixture(tmp_path)
    policy_path = tmp_path / "config/scheduled_refit_policy_v2.json"
    parent_path = tmp_path / child["performance_confirmation"]["parent_policy_path"]
    result_path = tmp_path / child["performance_confirmation"][
        "confirmation_result_path"
    ]

    decision_relative = child["contract"]["method_contract_decision_path"]
    decision_path = tmp_path / decision_relative
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_bytes((Path.cwd() / decision_relative).read_bytes())
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    terminal_relative = decision["terminal_report_path"]
    terminal_path = tmp_path / terminal_relative
    terminal_path.parent.mkdir(parents=True, exist_ok=True)
    terminal_path.write_bytes((Path.cwd() / terminal_relative).read_bytes())

    input_dir = tmp_path / "evidence/final_track_c_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    fights_path = input_dir / "fights_cleaned.csv"
    features_path = input_dir / "features.csv"
    fights_path.write_text(
        "event_date,fighter_a,fighter_b,winner,target\n"
        "2026-08-01,A,B,A,1\n",
        encoding="utf-8",
    )
    features_path.write_text(
        "event_date,fighter_a,fighter_b,target,feature\n"
        "2026-08-01,A,B,1,0.5\n",
        encoding="utf-8",
    )
    odds_path = input_dir / "odds_source_inventory.json"
    source_inventory_path = input_dir / "source_inventory.json"
    environment_path = input_dir / "environment.json"
    _write_json(odds_path, {"entries": []})
    _write_json(source_inventory_path, {"files": []})
    _write_json(environment_path, {"python": "test"})

    features_sha = gate.file_sha256(features_path)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_evidence_path = tmp_path / parent["baseline"]["evidence_path"]
    parent_evidence_path.write_bytes(
        parent_evidence_path.read_bytes().replace(
            str(parent["baseline"]["features_sha256"]).encode("ascii"),
            features_sha.encode("ascii"),
        )
    )
    parent["baseline"]["features_sha256"] = features_sha
    parent["baseline"]["evidence_sha256"] = gate.file_sha256(parent_evidence_path)
    _write_json(parent_path, parent)
    parent_sha = gate.file_sha256(parent_path)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["policy_sha256"] = parent_sha
    _write_json(result_path, result)
    result_sha = gate.file_sha256(result_path)

    child["baseline"]["features_sha256"] = features_sha
    child["performance_confirmation"]["parent_policy_sha256"] = parent_sha
    child["performance_confirmation"]["confirmation_result_sha256"] = result_sha
    _write_json(policy_path, child)
    policy_sha = gate.file_sha256(policy_path)

    evaluation_spec = resolve_named_training_spec(
        child["contract"]["evaluation_spec_name"]
    )
    fullfit_spec = resolve_named_training_spec(child["contract"]["fullfit_spec_name"])
    hashes = {
        "features_value_sha256": "1" * 64,
        "feature_contract_sha256": "2" * 64,
        "confirmation_evaluation_input_value_sha256": "3" * 64,
        "source_fingerprint": "4" * 64,
        "source_inventory_sha256": "5" * 64,
        "environment_payload_sha256": "6" * 64,
        "evaluation_protocol_sha256": "7" * 64,
    }
    binding = {
        **binding,
        **hashes,
        "policy_sha256": parent_sha,
        "dataset_fights_path": fights_path.relative_to(tmp_path).as_posix(),
        "dataset_fights_sha256": gate.file_sha256(fights_path),
        "source_dataset_fights_path": fights_path.relative_to(tmp_path).as_posix(),
        "source_dataset_fights_sha256": gate.file_sha256(fights_path),
        "features_artifact_path": features_path.relative_to(tmp_path).as_posix(),
        "features_artifact_sha256": features_sha,
        "features_value_sha256": hashes["features_value_sha256"],
        "source_features_path": features_path.relative_to(tmp_path).as_posix(),
        "source_features_sha256": features_sha,
        "feature_contract_count": len(evaluation_spec.feature_cols),
        "feature_contract_sha256": hashes["feature_contract_sha256"],
        "confirmation_evaluation_input_value_sha256": hashes[
            "confirmation_evaluation_input_value_sha256"
        ],
        "odds_source_inventory_path": odds_path.relative_to(tmp_path).as_posix(),
        "odds_source_inventory_sha256": gate.file_sha256(odds_path),
        "source_inventory_path": source_inventory_path.relative_to(tmp_path).as_posix(),
        "source_inventory_sha256": hashes["source_inventory_sha256"],
        "source_inventory_artifact_sha256": gate.file_sha256(source_inventory_path),
        "environment_path": environment_path.relative_to(tmp_path).as_posix(),
        "environment_artifact_sha256": gate.file_sha256(environment_path),
        "environment_payload_sha256": hashes["environment_payload_sha256"],
        "evaluation_protocol_sha256": hashes["evaluation_protocol_sha256"],
    }

    artifact_dir = tmp_path / "evidence/final_track_c"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "track_c_summary.csv"
    summary = {
        "strategy_config_sha256": binding["strategy_config_sha256"],
        "policy_sha256": policy_sha,
        "spec": evaluation_spec.name,
        "spec_payload_sha256": child["contract"]["evaluation_spec_payload_sha256"],
        "features_sha256": features_sha,
        "protocol_sha256": child["baseline"]["scheduled_protocol_sha256"],
        "evaluation_sample_sha256": child["baseline"]["evaluation_sample_sha256"],
        **{
            field: float(child["baseline"][field])
            for field in (
                "model_brier_score",
                "model_ece",
                "strategy_total_bets",
                "strategy_roi",
                "strategy_total_profit",
                "strategy_avg_clv",
                "strategy_max_drawdown_pct",
            )
        },
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    artifact_paths = {
        "summary": summary_path,
        "evaluation_index": artifact_dir
        / f"{evaluation_spec.name}_evaluation_index.csv",
        "data_quality": artifact_dir / f"{evaluation_spec.name}_data_quality.csv",
        "predictive": artifact_dir / f"{evaluation_spec.name}_predictive.json",
        "production_bets": artifact_dir
        / f"{evaluation_spec.name}_production_gated_bets.csv",
    }
    pd.DataFrame([{"row": 1}]).to_csv(artifact_paths["evaluation_index"], index=False)
    pd.DataFrame([{"eligible": True}]).to_csv(
        artifact_paths["data_quality"], index=False
    )
    _write_json(artifact_paths["predictive"], {"brier": 0.2})
    pd.DataFrame([{"profit": 1.0}]).to_csv(
        artifact_paths["production_bets"], index=False
    )

    health_metrics = {"fixture": 1.0}
    derived_limits = {"fixture": 2.0}
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "healthy": True,
        "final_policy_path": policy_path.relative_to(tmp_path).as_posix(),
        "final_policy_sha256": policy_sha,
        "confirmation_result_path": result_path.relative_to(tmp_path).as_posix(),
        "confirmation_result_sha256": result_sha,
        "model_spec_name": evaluation_spec.name,
        "model_spec_payload_sha256": child["contract"][
            "evaluation_spec_payload_sha256"
        ],
        "strategy_config_sha256": binding["strategy_config_sha256"],
        "dataset_fights_path": binding["dataset_fights_path"],
        "dataset_fights_sha256": binding["dataset_fights_sha256"],
        "source_dataset_fights_path": binding["source_dataset_fights_path"],
        "source_dataset_fights_sha256": binding["source_dataset_fights_sha256"],
        "features_path": binding["features_artifact_path"],
        "features_sha256": features_sha,
        "features_value_sha256": binding["features_value_sha256"],
        "source_features_path": binding["source_features_path"],
        "source_features_sha256": binding["source_features_sha256"],
        "feature_contract_count": binding["feature_contract_count"],
        "feature_contract_sha256": binding["feature_contract_sha256"],
        "confirmation_evaluation_input_value_sha256": binding[
            "confirmation_evaluation_input_value_sha256"
        ],
        "odds_source_inventory_path": binding["odds_source_inventory_path"],
        "odds_source_inventory_sha256": binding["odds_source_inventory_sha256"],
        "source_fingerprint": binding["source_fingerprint"],
        "source_inventory_path": binding["source_inventory_path"],
        "source_inventory_sha256": binding["source_inventory_sha256"],
        "source_inventory_artifact_sha256": binding[
            "source_inventory_artifact_sha256"
        ],
        "environment_path": binding["environment_path"],
        "environment_artifact_sha256": binding["environment_artifact_sha256"],
        "environment_payload_sha256": binding["environment_payload_sha256"],
        "protocol_sha256": child["baseline"]["scheduled_protocol_sha256"],
        "evaluation_protocol_sha256": binding["evaluation_protocol_sha256"],
        "evaluation_sample_sha256": child["baseline"]["evaluation_sample_sha256"],
        "artifacts": {
            name: {
                "path": path.relative_to(tmp_path).as_posix(),
                "sha256": gate.file_sha256(path),
            }
            for name, path in artifact_paths.items()
        },
        "health_metrics": health_metrics,
        "derived_limits": derived_limits,
    }
    receipt_path = tmp_path / "evidence/final_track_c_pass_receipt.json"
    _write_json(receipt_path, receipt)
    monkeypatch.setattr(
        "src.strategy.run_evaluation.validate_confirmation_result_for_promotion",
        lambda _path: binding,
    )
    return {
        "receipt_path": receipt_path,
        "receipt": receipt,
        "policy_path": policy_path,
        "fullfit_spec": fullfit_spec,
        "binding": binding,
        "artifact_paths": artifact_paths,
        "quality_validator": lambda **_kwargs: {
            "errors": [],
            "metrics": health_metrics,
            "derived_limits": derived_limits,
        },
        "confirmation_validator": lambda _path: binding,
    }


def test_final_track_c_pass_receipt_real_validator_accepts_exact_binding(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _final_track_c_receipt_fixture(tmp_path, monkeypatch)

    validated = gate.validate_final_track_c_pass_receipt(
        fixture["receipt_path"],
        expected_policy_path=fixture["policy_path"],
        expected_fullfit_spec=fixture["fullfit_spec"],
        repo_root=tmp_path,
        quality_validator=fixture["quality_validator"],
        confirmation_validator=fixture["confirmation_validator"],
    )

    assert validated["receipt_path"] == fixture["receipt_path"]
    assert validated["strategy_config_sha256"] == fixture["binding"][
        "strategy_config_sha256"
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("status", "explicit healthy PASS"),
        ("healthy", "explicit healthy PASS"),
        ("policy_path", "binds a different policy path"),
        ("policy_hash", "policy hash is stale"),
        ("artifact", "artifact changed"),
        ("confirmation", "confirmation result hash changed|confirmation result is stale"),
        ("outside_evidence", "durable below evidence"),
        ("wrong_fullfit", "full-fit spec differs"),
    ],
)
def test_final_track_c_pass_receipt_real_validator_rejects_tampering(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
    message: str,
):
    fixture = _final_track_c_receipt_fixture(tmp_path, monkeypatch)
    receipt_path = fixture["receipt_path"]
    expected_fullfit = fixture["fullfit_spec"]
    if mutation in {"status", "healthy", "policy_path", "policy_hash"}:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if mutation == "status":
            payload["status"] = "FAIL"
        elif mutation == "healthy":
            payload["healthy"] = False
        elif mutation == "policy_path":
            payload["final_policy_path"] = (
                "evidence/policies/pre_winner_policy.json"
            )
        else:
            payload["final_policy_sha256"] = "f" * 64
        _write_json(receipt_path, payload)
    elif mutation == "artifact":
        fixture["artifact_paths"]["production_bets"].write_text(
            "tampered\n", encoding="utf-8"
        )
    elif mutation == "confirmation":
        confirmation_path = tmp_path / fixture["receipt"]["confirmation_result_path"]
        confirmation_path.write_text("tampered\n", encoding="utf-8")
    elif mutation == "outside_evidence":
        receipt_path = tmp_path / "receipt.json"
        receipt_path.write_bytes(fixture["receipt_path"].read_bytes())
    else:
        expected_fullfit = resolve_named_training_spec(
            fixture["receipt"]["model_spec_name"]
        )

    with pytest.raises(gate.ContractInputError, match=message):
        gate.validate_final_track_c_pass_receipt(
            receipt_path,
            expected_policy_path=fixture["policy_path"],
            expected_fullfit_spec=expected_fullfit,
            repo_root=tmp_path,
            quality_validator=fixture["quality_validator"],
            confirmation_validator=fixture["confirmation_validator"],
        )


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
    policy = gate.load_policy(POLICY_V2_PATH)
    features = tmp_path / "features.csv"
    readyz = tmp_path / "readyz.json"
    _write_features(
        features,
        spec_name=policy["contract"]["fullfit_spec_name"],
        policy_v2=True,
    )
    _write_json(readyz, _root_readyz(policy, gate.checked_out_git_sha()))

    result = gate.evaluate_preflight(
        policy_path=POLICY_V2_PATH,
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


def test_v2_exact_root_can_bootstrap_from_legacy_spec_but_keeps_hash_binding() -> None:
    policy = gate.load_policy(POLICY_V2_PATH)
    workflow_sha = "e" * 40
    readyz = _root_readyz(policy, workflow_sha)
    readyz["production_bundle"]["model_spec_name"] = (
        "full_live_contract_v6_durability_corrected_20260805_fullfit"
    )

    errors, _ = gate.validate_active_lineage(
        readyz,
        policy=policy,
        policy_sha256=gate.file_sha256(POLICY_V2_PATH),
        workflow_git_sha=workflow_sha,
    )

    assert errors == []

    readyz["production_bundle"]["model_sha256"] = "0" * 64
    errors, _ = gate.validate_active_lineage(
        readyz,
        policy=policy,
        policy_sha256=gate.file_sha256(POLICY_V2_PATH),
        workflow_git_sha=workflow_sha,
    )

    assert any("active root release model_sha256" in error for error in errors)
    assert not any("active production spec" in error for error in errors)


def test_v2_descendant_must_use_new_policy_fullfit_spec() -> None:
    policy = gate.load_policy(POLICY_V2_PATH)
    policy_sha256 = gate.file_sha256(POLICY_V2_PATH)
    workflow_sha = "e" * 40
    readyz = _root_readyz(policy, workflow_sha)
    bundle = readyz["production_bundle"]
    bundle["bundle_id"] = "scheduled-refit-descendant"
    bundle["model_spec_name"] = policy["contract"]["fullfit_spec_name"]
    bundle["scheduled_refit_policy"] = {
        "policy_id": policy["policy_id"],
        "sha256": policy_sha256,
        "root_bundle_id": policy["root_release"]["bundle_id"],
    }

    errors, _ = gate.validate_active_lineage(
        readyz,
        policy=policy,
        policy_sha256=policy_sha256,
        workflow_git_sha=workflow_sha,
    )
    assert errors == []

    bundle["model_spec_name"] = (
        "full_live_contract_v6_durability_corrected_20260805_fullfit"
    )
    errors, _ = gate.validate_active_lineage(
        readyz,
        policy=policy,
        policy_sha256=policy_sha256,
        workflow_git_sha=workflow_sha,
    )

    assert any("active production spec" in error for error in errors)


def test_preflight_rejects_wrong_deployed_head_and_mutable_hash_substitution(
    tmp_path: Path,
) -> None:
    policy = gate.load_policy(POLICY_V2_PATH)
    features = tmp_path / "features.csv"
    readyz = tmp_path / "readyz.json"
    _write_features(
        features,
        spec_name=policy["contract"]["fullfit_spec_name"],
        policy_v2=True,
    )
    payload = _root_readyz(policy, "e" * 40)
    payload["production_bundle"].pop("immutable_training_features_sha256")
    payload["production_bundle"]["processed_features_sha256"] = policy["root_release"][
        "processed_features_sha256"
    ]
    _write_json(readyz, payload)

    result = gate.evaluate_preflight(
        policy_path=POLICY_V2_PATH,
        features_path=features,
        readyz_path=readyz,
        now=NOW,
    )

    assert any("deployed_git_sha" in error for error in result["errors"])
    assert any("immutable_training_features_sha256" in error for error in result["errors"])


def test_scheduled_refit_gates_reject_v1_policy(tmp_path: Path) -> None:
    policy = gate.load_policy(POLICY_PATH)
    features = tmp_path / "features.csv"
    readyz = tmp_path / "readyz.json"
    _write_features(features)
    _write_json(readyz, _root_readyz(policy, gate.checked_out_git_sha()))

    preflight = gate.evaluate_preflight(
        policy_path=POLICY_PATH,
        features_path=features,
        readyz_path=readyz,
        now=NOW,
    )
    policy_path, manifest_path, candidate_readyz = _candidate_fixture(
        tmp_path / "v1-candidate",
        policy_path=POLICY_PATH,
    )
    candidate = gate.evaluate_candidate(
        policy_path=policy_path,
        manifest_path=manifest_path,
        parent_readyz_path=candidate_readyz,
        now=NOW,
    )

    expected = "scheduled refit gate requires policy.schema_version=2"
    assert expected in preflight["errors"]
    assert expected in candidate["errors"]


def test_candidate_accepts_exact_policy_bound_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path, manifest_path, readyz_path = _candidate_fixture(
        tmp_path, monkeypatch=monkeypatch
    )

    result = gate.evaluate_candidate(
        policy_path=policy_path,
        manifest_path=manifest_path,
        parent_readyz_path=readyz_path,
        now=NOW,
    )

    assert result["errors"] == []
    assert result["candidate_manifest"]["bundle_id"] == "candidate-test"


def test_candidate_rejects_intermediate_v2_policy(tmp_path: Path) -> None:
    policy_path, manifest_path, readyz_path = _candidate_fixture(
        tmp_path,
        final_policy=False,
    )

    result = gate.evaluate_candidate(
        policy_path=policy_path,
        manifest_path=manifest_path,
        parent_readyz_path=readyz_path,
        now=NOW,
    )

    assert (
        "candidate full fit requires "
        "policy.performance_confirmation.state='final_track_c_bound'"
        in result["errors"]
    )


def test_v2_candidate_requires_policy_bound_training_input_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path, manifest_path, readyz_path = _candidate_fixture(
        tmp_path, monkeypatch=monkeypatch
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("training_input_evidence")
    _write_json(manifest_path, manifest)

    result = gate.evaluate_candidate(
        policy_path=policy_path,
        manifest_path=manifest_path,
        parent_readyz_path=readyz_path,
        now=NOW,
    )

    assert "candidate has no policy-bound training_input_evidence object" in result[
        "errors"
    ]


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("artifact_roles", ["primary"], "mismatch: artifact_roles"),
        ("prepared_features_sha256", "0" * 64, "mismatch: prepared_features_sha256"),
        ("source_fights_sha256", "0" * 64, "source_fights_sha256 is stale"),
        (
            "final_track_c_pass_receipt_sha256",
            "0" * 64,
            "performance_confirmation mismatch: final_track_c_pass_receipt_sha256",
        ),
        (
            "confirmed_strategy_config_sha256",
            "0" * 64,
            "performance_confirmation mismatch: strategy_config_sha256",
        ),
        ("feature_contract_count", 204, "mismatch: feature_contract_count"),
        (
            "confirmation_source_features_sha256",
            "not-a-hash",
            "confirmation_source_features_sha256 is not SHA-256",
        ),
    ],
)
def test_checker_rejects_expanded_training_input_evidence_drift(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    policy_path, manifest_path, _readyz_path = _candidate_fixture(tmp_path)
    policy = gate.load_policy(policy_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence = manifest["training_input_evidence"]
    evidence[field] = replacement

    errors = gate.validate_training_input_evidence(
        evidence,
        features_path=manifest_path.parent / "processed/features.csv",
        fights_path=manifest_path.parent / "processed/fights_cleaned.csv",
        policy=policy,
        policy_sha256=gate.file_sha256(policy_path),
        performance_confirmation=manifest["performance_confirmation"],
    )

    assert any(message in error for error in errors)


@pytest.mark.parametrize(
    "field",
    ["confirmation_result_sha256", "strategy_config_sha256"],
)
def test_checker_rejects_manifest_performance_drift_from_final_policy(
    tmp_path: Path,
    field: str,
) -> None:
    policy_path, manifest_path, _readyz_path = _candidate_fixture(tmp_path)
    policy = gate.load_policy(policy_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["performance_confirmation"][field] = "0" * 64

    errors = gate.validate_training_input_evidence(
        manifest["training_input_evidence"],
        features_path=manifest_path.parent / "processed/features.csv",
        fights_path=manifest_path.parent / "processed/fights_cleaned.csv",
        policy=policy,
        policy_sha256=gate.file_sha256(policy_path),
        performance_confirmation=manifest["performance_confirmation"],
    )

    assert any(
        f"performance_confirmation/final policy mismatch: {field}" in error
        for error in errors
    )


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
