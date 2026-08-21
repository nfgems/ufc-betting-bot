import json
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import scripts.build_model_input_inventory as inventory_module
from scripts import bfo_lineage
from scripts import build_staged_production_bundle as builder
from src.model.production_bundle import _expected_no_odds_spec_payload
from src.model.train import prepare_train_test
from src.model.training_spec import (
    materialize_and_validate_spec_features,
    resolve_named_training_spec,
)


EVALUATION_SPEC = "full_live_contract_v6_durability"
FULLFIT_SPEC = "full_live_contract_v6_durability_corrected_20260805_fullfit"


class _ConstantModel:
    def predict_proba(self, values):
        count = len(values)
        return np.column_stack([np.full(count, 0.4), np.full(count, 0.6)])


class _LinearModel:
    def predict_proba(self, values):
        probability = np.asarray(values[:, 0], dtype=float)
        return np.column_stack([1.0 - probability, probability])


def _semantic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_date": ["2026-01-01", "2026-01-08"],
            "fighter_a": ["Alpha", "Charlie"],
            "fighter_b": ["Bravo", "Delta"],
            "winner": ["Alpha", "Delta"],
            "target": [1, 0],
            "plain_metric": [0.25, np.nan],
            "exact_count": [3, 4],
            "note": ["one", "two"],
        }
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return builder._sha256_file(path)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _model_result(spec: dict) -> dict:
    return {
        "model": _ConstantModel(),
        "feature_cols": list(spec["feature_cols"]),
        "training_spec": deepcopy(spec),
        "impute_strategy": "native_nan",
        "col_medians": np.zeros(len(spec["feature_cols"])),
        "observed_training_rows": 2,
        "effective_training_rows": 4,
        "mirror_augmentation": True,
    }


@pytest.fixture
def staged_inputs(tmp_path_factory, monkeypatch):
    # Keep the nested atomic-copy destination below the practical Windows path
    # limit while still using pytest's repository-local configured basetemp.
    repo = tmp_path_factory.mktemp("bundle") / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "scripts").mkdir()
    source_odds = Path(__file__).resolve().parents[1] / "data/raw/historical_odds"
    odds_dir = repo / "data/raw/historical_odds"
    odds_dir.mkdir(parents=True)
    for filename in (
        builder.APPROVED_BFO_LEDGER_NAME,
        *builder.APPROVED_BFO_CSVS,
    ):
        shutil.copy2(source_odds / filename, odds_dir / filename)
    # Git may materialize this JSONL with CRLF in a Windows linked worktree;
    # production evidence is byte-pinned to its original LF representation.
    ledger_copy = odds_dir / builder.APPROVED_BFO_LEDGER_NAME
    ledger_copy.write_bytes(ledger_copy.read_bytes().replace(b"\r\n", b"\n"))
    ledger = odds_dir / builder.APPROVED_BFO_LEDGER_NAME
    (repo / "src/dummy.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "scripts/dummy.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "scripts/build_staged_production_bundle.py").write_text(
        "ASSEMBLER_VERSION = 1\n", encoding="utf-8"
    )
    (repo / "scripts/parity_replay.py").write_text(
        "PARITY_VERSION = 1\n", encoding="utf-8"
    )
    source_scripts = Path(__file__).resolve().parents[1] / "scripts"
    for filename in (
        "recover_bfo_moneyline_gaps.py",
        "revalidate_bfo_recovery_file.py",
    ):
        shutil.copy2(source_scripts / filename, repo / "scripts" / filename)
    (repo / ".gitignore").write_text("logs/\nstaging/\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "tests@example.test")
    _git(repo, "config", "user.name", "Tests")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    (repo / "src/dummy.py").write_text("VALUE = 2\n", encoding="utf-8")
    head = _git(repo, "rev-parse", "HEAD")

    canonical_models = repo / "models"
    canonical_models.mkdir()
    previous_spec = asdict(
        resolve_named_training_spec("full_live_contract_v6_durability_fullfit")
    )
    previous_spec["git_hash"] = "d" * 40
    previous_spec["trained_at"] = "2026-08-04T10:32:01"
    previous_specs = {
        "primary": previous_spec,
        "no_odds": _expected_no_odds_spec_payload(previous_spec),
        "logistic": previous_spec,
    }
    previous_hashes = {}
    for label, filename in builder.MODEL_FILENAMES.items():
        path = canonical_models / filename
        joblib.dump(_model_result(previous_specs[label]), path)
        previous_hashes[label] = _sha256(path)
    _write_json(
        canonical_models / "full_live_contract_v6_durability_fullfit_spec.json",
        previous_spec,
    )
    canonical_processed = repo / "data/processed"
    canonical_processed.mkdir(parents=True)
    previous_fights = canonical_processed / "fights_cleaned.csv"
    previous_features = canonical_processed / "features.csv"
    pd.DataFrame({"event_date": ["2026-08-01"]}).to_csv(previous_fights, index=False)
    pd.DataFrame({"event_date": ["2026-08-01"], "value": [1]}).to_csv(
        previous_features, index=False
    )
    previous_manifest = canonical_models / "current_production_model.json"
    _write_json(
        previous_manifest,
        {
            "bundle_id": "previous",
            "model_spec_name": "full_live_contract_v6_durability_fullfit",
            "snapshot_max_event_date": "2026-08-01",
            "built_at": "2026-08-04T10:32:13+00:00",
            "git_sha": "e" * 40,
            "model_sha256": previous_hashes["primary"],
            "no_odds_model_sha256": previous_hashes["no_odds"],
            "processed_fights_sha256": _sha256(previous_fights),
            "processed_features_sha256": _sha256(previous_features),
            "processed_fights_bytes": previous_fights.stat().st_size,
            "processed_features_bytes": previous_features.stat().st_size,
        },
    )
    readyz_path = repo / "logs/previous_readyz.json"
    _write_json(
        readyz_path,
        {
            "ready": True,
            "production_bundle": {
                "bundle_id": "previous",
                "model_spec_name": "full_live_contract_v6_durability_fullfit",
                "deployed_git_sha": "a" * 40,
                "processed_features_sha256": "b" * 64,
                "processed_fights_sha256": "c" * 64,
                "processed_features_bytes": 999,
                "processed_fights_bytes": 888,
            },
        },
    )

    monkeypatch.setattr(inventory_module, "REPO_ROOT", repo)
    pretraining_inventory = inventory_module.build_inventory(run_id="fixture")
    inventory_path = repo / "logs/pretraining_inventory.json"
    _write_json(inventory_path, pretraining_inventory)

    # These are generated after the inventory. Their untracked status changes,
    # while the complete source/raw scope and tracked diff remain unchanged.
    models_dir = repo / "models/candidates/run_train"
    processed_dir = repo / "data/processed/candidates/run_train"
    audit_dir = repo / "data/processed/candidates/run_audit"
    models_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)

    primary_spec = asdict(resolve_named_training_spec(FULLFIT_SPEC))
    primary_spec["git_hash"] = head
    primary_spec["trained_at"] = "2026-08-05T12:00:00+00:00"
    no_odds_spec = _expected_no_odds_spec_payload(primary_spec)
    joblib.dump(_model_result(primary_spec), models_dir / "xgboost_model.pkl")
    joblib.dump(_model_result(no_odds_spec), models_dir / "xgboost_no_odds_model.pkl")
    joblib.dump(_model_result(primary_spec), models_dir / "logistic_model.pkl")
    _write_json(models_dir / f"{FULLFIT_SPEC}_spec.json", primary_spec)

    feature_cols = list(primary_spec["feature_cols"])
    feature_rows = []
    for index, date in enumerate(("2026-07-25", "2026-08-01")):
        row = {column: 0.1 + index * 0.01 for column in feature_cols}
        row.update(
            {
                "event_date": date,
                "target": index % 2,
                "a_num_fights": 3,
                "b_num_fights": 3,
            }
        )
        feature_rows.append(row)
    features = pd.DataFrame(feature_rows)
    features["fighter_a"] = ["A", "C"]
    features["fighter_b"] = ["B", "D"]
    features["winner"] = ["A", "D"]
    audit_features = features.copy(deep=True)
    fights = pd.DataFrame(
        {
            "event_date": ["2026-07-25", "2026-08-01"],
            "fighter_a": ["A", "C"],
            "fighter_b": ["B", "D"],
            "winner": ["A", "D"],
            "target": [1, 0],
            "pace_metric": [0.1234567890123, 0.2345678901234],
        }
    )
    fights.to_csv(processed_dir / "fights_cleaned.csv", index=False)
    features.to_csv(processed_dir / "features.csv", index=False)
    audit_fights = fights.copy()
    audit_fights.loc[0, "pace_metric"] += 5e-13
    audit_features.loc[0, feature_cols[0]] += 5e-13
    audit_fights.to_csv(audit_dir / "fights_cleaned.csv", index=False)
    audit_features.to_csv(audit_dir / "features.csv", index=False)
    materialized = materialize_and_validate_spec_features(
        features, resolve_named_training_spec(FULLFIT_SPEC)
    )
    _, test_set, _ = prepare_train_test(
        materialized,
        cutoff_date=primary_spec["train_cutoff_date"],
        feature_cols=feature_cols,
        start_date=primary_spec["train_start_date"],
    )
    test_path = processed_dir / "test_set.csv"
    test_set.to_csv(test_path, index=False)
    _write_json(
        processed_dir / "test_set.csv.metadata.json",
        {
            "spec_name": FULLFIT_SPEC,
            "feature_count": len(feature_cols),
            "feature_hash": builder._canonical_json_sha256(feature_cols),
            "row_count": len(test_set),
            "training_spec": primary_spec,
            "test_set_sha256": _sha256(test_path),
        },
    )
    evidence = repo / "logs/selection.json"
    _write_json(evidence, {"selected": EVALUATION_SPEC})
    (repo / ".codex_stage").mkdir()
    audit_fights_sha = _sha256(audit_dir / "fights_cleaned.csv")
    audit_features_sha = _sha256(audit_dir / "features.csv")
    train_fights_sha = _sha256(processed_dir / "fights_cleaned.csv")
    train_features_sha = _sha256(processed_dir / "features.csv")
    assert audit_fights_sha != train_fights_sha
    assert audit_features_sha != train_features_sha
    monkeypatch.setattr(builder, "APPROVED_FIGHTS_SHA256", audit_fights_sha)
    monkeypatch.setattr(builder, "APPROVED_FEATURES_SHA256", audit_features_sha)
    monkeypatch.setattr(builder, "APPROVED_TRAIN_FIGHTS_SHA256", train_fights_sha)
    monkeypatch.setattr(builder, "APPROVED_TRAIN_FEATURES_SHA256", train_features_sha)

    def replay_stub(**kwargs):
        trainer_fights = kwargs["trainer_fights_path"]
        trainer_features = kwargs["trainer_features_path"]
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
                "audit_source_sha256": _sha256(kwargs["audit_fights_path"]),
                "trainer_output_sha256": _sha256(trainer_fights),
                "replay_output_sha256": _sha256(trainer_fights),
                "replay_output_bytes": trainer_fights.stat().st_size,
                "byte_match": True,
            },
            "features": {
                "trainer_output_sha256": _sha256(trainer_features),
                "replay_output_sha256": _sha256(trainer_features),
                "replay_output_bytes": trainer_features.stat().st_size,
                "byte_match": True,
            },
            "explanation": "fixture exact preprocessing replay",
        }

    monkeypatch.setattr(builder, "_replay_trainer_preprocessing", replay_stub)

    # Exactly the two review/packaging scripts are allowed to change after the
    # model-provenance inventory was frozen.
    (repo / "scripts/build_staged_production_bundle.py").write_text(
        "ASSEMBLER_VERSION = 2\n", encoding="utf-8"
    )
    (repo / "scripts/parity_replay.py").write_text(
        "PARITY_VERSION = 2\n", encoding="utf-8"
    )
    assembly_inventory = inventory_module.build_inventory(run_id="fixture-assembly")
    assembly_inventory_path = repo / "logs/assembly_inventory.json"
    _write_json(assembly_inventory_path, assembly_inventory)

    inputs = builder.BundleInputs(
        staging_root=Path(".codex_stage/new_bundle"),
        candidate_models_dir=Path("models/candidates/run_train"),
        candidate_processed_dir=Path("data/processed/candidates/run_train"),
        evaluation_spec_name=EVALUATION_SPEC,
        input_inventory_path=Path("logs/pretraining_inventory.json"),
        assembly_inventory_path=Path("logs/assembly_inventory.json"),
        bfo_provenance_path=ledger.relative_to(repo),
        selection_evidence_paths=(evidence.relative_to(repo),),
        previous_manifest_path=previous_manifest.relative_to(repo),
        previous_readyz_path=readyz_path.relative_to(repo),
        previous_deployed_git_sha="a" * 40,
        previous_runtime_lookup_hashes={
            "processed_features_sha256": "b" * 64,
            "processed_fights_sha256": "c" * 64,
        },
        expected_fights_sha256=audit_fights_sha,
        expected_features_sha256=audit_features_sha,
        training_argv=(
            sys.executable,
            "-m",
            "src.bot",
            "train",
            "--data",
            "data/processed/candidates/run_audit/fights_cleaned.csv",
            "--spec",
            FULLFIT_SPEC,
            "--output-subdir",
            "candidates/run_train",
        ),
        inference_sample_rows=2,
    )
    return repo, inputs, pretraining_inventory


def test_assembler_builds_one_strict_bundle_after_candidate_outputs_change_status(
    staged_inputs,
):
    repo, inputs, pretraining_inventory = staged_inputs
    before_manifest = (repo / "models/current_production_model.json").read_bytes()
    before_models = {
        filename: (repo / "models" / filename).read_bytes()
        for filename in builder.MODEL_FILENAMES.values()
    }

    result = builder.assemble_staged_bundle(inputs, repo_root=repo)

    stage = Path(result["staging_root"])
    manifest = json.loads((stage / "staging_manifest.json").read_text(encoding="utf-8"))
    assert result["validation"]["validation_scope"] == "staged"
    assert manifest["source_identity"]["base_git_sha"] == pretraining_inventory["git_head"]
    assert (
        manifest["source_identity"]["pre_training_dirty_status"]["git_status_sha256"]
        != manifest["source_identity"]["assembly_dirty_status"]["git_status_sha256"]
    )
    assert manifest["registered_training_specs"]["selected_evaluation"]["payload"][
        "odds_noise_seed"
    ] is None
    assert manifest["registered_training_specs"]["selected_fullfit"]["payload"][
        "odds_noise_seed"
    ] == 42
    audit = manifest["training_invocation"]["independent_audit_snapshot"]
    assert audit["audit_source_equals_trainer_output"] is False
    assert audit["preprocessing_replay"]["preprocessing_replay_byte_match"] is True
    assert audit["semantic_equivalence"]["equivalent"] is True
    assert audit["semantic_equivalence"]["features"]["numeric_max_abs_delta"] <= 1e-12
    assert manifest["source_identity"]["pretraining_to_assembly_delta"]["changed"] == [
        "scripts/build_staged_production_bundle.py",
        "scripts/parity_replay.py",
    ]
    assert (
        stage / "provenance/independent_audit_snapshot/fights_cleaned.csv"
    ).is_file()
    assert (stage / "provenance/independent_audit_snapshot/features.csv").is_file()
    assert manifest["processed_fights_sha256"] == builder.APPROVED_TRAIN_FIGHTS_SHA256
    assert manifest["processed_features_sha256"] == builder.APPROVED_TRAIN_FEATURES_SHA256
    assert set(manifest["model_artifacts"]) == {"primary", "no_odds", "logistic"}
    assert all(
        manifest["finite_inference"][label]["finite_probability_count"] == 2
        for label in ("primary", "no_odds", "logistic")
    )
    assert (repo / "models/current_production_model.json").read_bytes() == before_manifest
    assert {
        filename: (repo / "models" / filename).read_bytes()
        for filename in builder.MODEL_FILENAMES.values()
    } == before_models


def _scheduled_inputs(staged_inputs):
    repo, inputs, pretraining_inventory = staged_inputs
    previous = json.loads(
        (repo / "models/current_production_model.json").read_text(encoding="utf-8")
    )
    policy = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "config/scheduled_refit_policy_v1.json"
        ).read_text(encoding="utf-8")
    )
    previous_logistic_sha = _sha256(repo / "models/logistic_model.pkl")
    policy["root_release"] = {
        "bundle_id": previous["bundle_id"],
        "release_id": "r-fixture-parent",
        "source_manifest_sha256": "1" * 64,
        "installed_manifest_sha256": "2" * 64,
        "promotion_git_sha": pretraining_inventory["git_head"],
        "training_source_git_sha": "d" * 40,
        "model_sha256": previous["model_sha256"],
        "no_odds_model_sha256": previous["no_odds_model_sha256"],
        "logistic_model_sha256": previous_logistic_sha,
        "processed_fights_sha256": previous["processed_fights_sha256"],
        "processed_features_sha256": previous["processed_features_sha256"],
    }
    policy["baseline"]["evidence_root_source_manifest_sha256"] = "1" * 64
    policy_path = repo / "config/scheduled_refit_policy_v1.json"
    policy_path.parent.mkdir()
    _write_json(policy_path, policy)

    release_root = (repo / "logs/runtime-store/releases/r-fixture-parent").resolve()
    readyz = {
        "ready": True,
        "production_bundle": {
            "bundle_id": previous["bundle_id"],
            "model_spec_name": previous["model_spec_name"],
            "rich_release_id": "r-fixture-parent",
            "rich_release_root": str(release_root),
            "installed_manifest_path": str(release_root / "manifest.json"),
            "installed_manifest_sha256": "2" * 64,
            "source_manifest_sha256": "1" * 64,
            "model_sha256": previous["model_sha256"],
            "no_odds_model_sha256": previous["no_odds_model_sha256"],
            "logistic_model_sha256": previous_logistic_sha,
            "immutable_training_fights_sha256": previous[
                "processed_fights_sha256"
            ],
            "immutable_training_features_sha256": previous[
                "processed_features_sha256"
            ],
            "immutable_training_snapshot_max_event_date": previous[
                "snapshot_max_event_date"
            ],
            "processed_fights_sha256": "c" * 64,
            "processed_features_sha256": "b" * 64,
            "processed_fights_bytes": 888,
            "processed_features_bytes": 999,
            "deployed_git_sha": pretraining_inventory["git_head"],
            "training_source_git_sha": "d" * 40,
        },
    }
    _write_json(repo / "logs/previous_readyz.json", readyz)
    return repo, builder.BundleInputs(
        **{
            **inputs.__dict__,
            "previous_manifest_path": None,
            "previous_deployed_git_sha": None,
            "previous_runtime_lookup_hashes": {},
            "scheduled_refit_policy_path": policy_path.relative_to(repo),
        }
    )


def test_scheduled_assembler_binds_readyz_parent_without_local_predecessor_bytes(
    staged_inputs,
):
    repo, inputs = _scheduled_inputs(staged_inputs)
    # The app can deploy later workflow code while retaining the immutable root
    # release. The separate preflight gate checks ancestry from this historical
    # promotion SHA; the builder must bind readyz to the current checkout.
    policy_path = repo / "config/scheduled_refit_policy_v1.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["root_release"]["promotion_git_sha"] = "f" * 40
    _write_json(policy_path, policy)

    result = builder.assemble_staged_bundle(inputs, repo_root=repo)

    manifest = json.loads(
        Path(result["manifest_path"]).read_text(encoding="utf-8")
    )
    binding = manifest["scheduled_refit_policy"]
    parent = json.loads(
        (repo / "logs/previous_readyz.json").read_text(encoding="utf-8")
    )["production_bundle"]
    assert set(binding) == builder.SCHEDULED_REFIT_MANIFEST_KEYS
    assert binding["parent_model_sha256"] == parent["model_sha256"]
    assert (
        binding["parent_processed_fights_sha256"]
        == parent["immutable_training_fights_sha256"]
    )
    rollback = manifest["previous_rollback_identity"]
    assert rollback["readyz_evidence"]["sole_parent_identity_source"] is True
    assert rollback["source_manifest"]["bytes_local"] is False
    assert "local_model_artifacts" not in rollback
    assert not (Path(result["staging_root"]) / "rollback/previous_installed_manifest.json").exists()
    lineage = manifest["raw_input_provenance"]["scheduled_bfo_lineage"]
    assert lineage["batch_count"] == 0
    assert (
        Path(result["staging_root"]) / "provenance/bfo_lineage/manifest.json"
    ).is_file()


def test_scheduled_assembler_carries_forward_attested_bfo_batch(staged_inputs):
    repo, inputs = _scheduled_inputs(staged_inputs)
    raw_root = repo / "data/raw/historical_odds"
    csv_name = "historical_odds_bfo_recovered_auto_prior.csv"
    ledger_name = "historical_odds_bfo_recovered_auto_prior.provenance.jsonl"
    csv_path = raw_root / csv_name
    ledger_path = raw_root / ledger_name
    csv_path.write_text(
        "event_date,fighter_a,fighter_b\n2026-08-02,A,B\n",
        encoding="utf-8",
    )
    ledger_path.write_text(
        json.dumps({"decision": "accepted"}) + "\n",
        encoding="utf-8",
    )
    artifact_root = repo / ".codex_stage/parent_bfo_lineage"
    artifact_batches = artifact_root / "batches"
    artifact_batches.mkdir(parents=True)
    shutil.copy2(csv_path, artifact_batches / csv_name)
    shutil.copy2(ledger_path, artifact_batches / ledger_name)
    lineage_payload = {
        "schema_version": 1,
        "parent_bundle_id": "older-parent",
        "parent_source_manifest_sha256": "0" * 64,
        "previous_lineage_manifest_sha256": None,
        "batches": [
            {
                "accepted_records": 1,
                "rejected_records": 0,
                "csv": {
                    "raw_path": f"data/raw/historical_odds/{csv_name}",
                    "artifact_path": f"batches/{csv_name}",
                    "sha256": _sha256(csv_path),
                    "bytes": csv_path.stat().st_size,
                    "rows": 1,
                },
                "provenance": {
                    "raw_path": f"data/raw/historical_odds/{ledger_name}",
                    "artifact_path": f"batches/{ledger_name}",
                    "sha256": _sha256(ledger_path),
                    "bytes": ledger_path.stat().st_size,
                    "line_count": 1,
                },
            }
        ],
    }
    lineage_manifest = bfo_lineage.write_manifest(
        lineage_payload,
        artifact_root / "manifest.json",
    )
    readyz_path = repo / "logs/previous_readyz.json"
    readyz = json.loads(readyz_path.read_text(encoding="utf-8"))
    readyz["production_bundle"][
        "scheduled_bfo_lineage_manifest_sha256"
    ] = _sha256(lineage_manifest)
    _write_json(readyz_path, readyz)

    pretraining = inventory_module.build_inventory(run_id="lineage-pretraining")
    _write_json(repo / "logs/pretraining_inventory.json", pretraining)
    assembly = inventory_module.build_inventory(run_id="lineage-assembly")
    _write_json(repo / "logs/assembly_inventory.json", assembly)
    inputs = builder.BundleInputs(
        **{
            **inputs.__dict__,
            "previous_bfo_lineage_manifest_path": lineage_manifest.relative_to(repo),
        }
    )

    result = builder.assemble_staged_bundle(inputs, repo_root=repo)

    stage = Path(result["staging_root"])
    manifest = json.loads((stage / "staging_manifest.json").read_text(encoding="utf-8"))
    lineage = manifest["raw_input_provenance"]["scheduled_bfo_lineage"]
    assert lineage["batch_count"] == 1
    assert lineage["batches"] == lineage_payload["batches"]
    assert _sha256(stage / f"provenance/bfo_lineage/batches/{csv_name}") == _sha256(
        csv_path
    )
    assert _sha256(
        stage / f"provenance/bfo_lineage/batches/{ledger_name}"
    ) == _sha256(ledger_path)


def test_scheduled_bfo_gate_accepts_a_zero_row_append_only_batch(tmp_path: Path):
    repo = tmp_path / "repo"
    odds = repo / "data/raw/historical_odds"
    odds.mkdir(parents=True)
    output = odds / "historical_odds_bfo_recovered_empty.csv"
    output.write_text(
        "event_date,fighter_a,fighter_b,query_date,offset_days\n",
        encoding="utf-8",
    )
    ledger = odds / "historical_odds_bfo_recovered_empty.provenance.jsonl"
    ledger.write_text("", encoding="utf-8")
    inventory = {
        "files": [
            {
                "path": path.relative_to(repo).as_posix(),
                "category": "raw_input",
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (output, ledger)
        ]
    }

    result = builder._validate_scheduled_bfo_ledger(
        ledger,
        repo_root=repo,
        inventory_payload=inventory,
    )

    assert result["accepted_records"] == 0
    assert result["rejected_records"] == 0
    assert result["corrected_csv_files"][0]["rows"] == 0
    assert result["provenance_mode"] == "scheduled_recovery_batch"
    assert result["corrected_csv_files"][0]["staged_path"] == (
        "provenance/data/raw/historical_odds/"
        "historical_odds_bfo_recovered_empty.csv"
    )


def test_scheduled_assembler_does_not_carry_forward_zero_row_bfo_batch(
    staged_inputs,
):
    repo, inputs = _scheduled_inputs(staged_inputs)
    odds = repo / "data/raw/historical_odds"
    output = odds / "historical_odds_bfo_recovered_empty.csv"
    output.write_text(
        "event_date,fighter_a,fighter_b,query_date,offset_days\n",
        encoding="utf-8",
    )
    ledger = odds / "historical_odds_bfo_recovered_empty.provenance.jsonl"
    ledger.write_text("", encoding="utf-8")
    _write_json(
        repo / "logs/pretraining_inventory.json",
        inventory_module.build_inventory(run_id="empty-bfo-pretraining"),
    )
    _write_json(
        repo / "logs/assembly_inventory.json",
        inventory_module.build_inventory(run_id="empty-bfo-assembly"),
    )
    inputs = builder.BundleInputs(
        **{
            **inputs.__dict__,
            "bfo_provenance_path": ledger.relative_to(repo),
        }
    )

    result = builder.assemble_staged_bundle(inputs, repo_root=repo)

    stage = Path(result["staging_root"])
    manifest = json.loads((stage / "staging_manifest.json").read_text(encoding="utf-8"))
    assert manifest["raw_input_provenance"]["scheduled_bfo_lineage"]["batch_count"] == 0
    assert (
        stage
        / "provenance/data/raw/historical_odds/historical_odds_bfo_recovered_empty.csv"
    ).is_file()


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_bfo_gate_accepts_lf_and_crlf_checkout_bytes(staged_inputs, newline: bytes):
    repo, _, _ = staged_inputs
    odds_dir = repo / "data/raw/historical_odds"
    for filename in (
        builder.APPROVED_BFO_LEDGER_NAME,
        *builder.APPROVED_BFO_CSVS,
    ):
        path = odds_dir / filename
        canonical_bytes = path.read_bytes().replace(b"\r\n", b"\n").replace(
            b"\r", b"\n"
        )
        path.write_bytes(canonical_bytes.replace(b"\n", newline))

    current_inventory = inventory_module.build_inventory(
        run_id=f"bfo-newline-{len(newline)}"
    )
    result = builder._validate_bfo_ledger(
        odds_dir / builder.APPROVED_BFO_LEDGER_NAME,
        repo_root=repo,
        inventory_payload=current_inventory,
    )

    assert result["line_count"] == 244
    assert len(result["corrected_csv_files"]) == 6


def test_bfo_canonical_text_digest_accepts_lf_and_crlf_but_rejects_content_change(
    tmp_path: Path,
):
    lf = tmp_path / "lf.csv"
    crlf = tmp_path / "crlf.csv"
    changed = tmp_path / "changed.csv"
    lf.write_bytes(b"a,b\n1,2\n")
    crlf.write_bytes(b"a,b\r\n1,2\r\n")
    changed.write_bytes(b"a,b\n1,3\n")

    assert builder._canonical_text_sha256(lf) == builder._canonical_text_sha256(crlf)
    assert builder._canonical_text_sha256(lf) != builder._canonical_text_sha256(changed)


def test_assembler_accepts_no_posttraining_source_changes(staged_inputs):
    repo, inputs, _ = staged_inputs
    assembly_inventory = repo / inputs.assembly_inventory_path
    pretraining_inventory = repo / inputs.input_inventory_path
    pretraining_inventory.write_bytes(assembly_inventory.read_bytes())

    result = builder.assemble_staged_bundle(inputs, repo_root=repo)

    stage = Path(result["staging_root"])
    manifest = json.loads((stage / "staging_manifest.json").read_text(encoding="utf-8"))
    delta = manifest["source_identity"]["pretraining_to_assembly_delta"]
    assert delta["changed"] == []
    assert delta["only_allowlisted_assembly_change"] is True


@pytest.mark.parametrize(
    "digest",
    (
        builder.APPROVED_EVALUATION_PAYLOAD_SHA256,
        builder.APPROVED_FIGHTS_SHA256,
        builder.APPROVED_FEATURES_SHA256,
        builder.APPROVED_TRAIN_FIGHTS_SHA256,
        builder.APPROVED_TRAIN_FEATURES_SHA256,
    ),
)
def test_approved_snapshot_hashes_are_sha256(digest):
    assert builder.SHA256_RE.fullmatch(digest)


@pytest.mark.parametrize("mode", ["existing", "outside", "canonical_namespace"])
def test_assembler_refuses_existing_or_out_of_repo_staging_roots(
    staged_inputs, tmp_path: Path, mode: str
):
    repo, inputs, _ = staged_inputs
    if mode == "existing":
        target = repo / ".codex_stage/already_here"
        target.mkdir()
    elif mode == "canonical_namespace":
        target = repo / "models/new_stage"
    else:
        target = tmp_path / "outside_bundle"
    # BundleInputs is frozen; reconstruct only the field under test.
    changed = builder.BundleInputs(
        **{**inputs.__dict__, "staging_root": target}
    )
    with pytest.raises(builder.StagingBundleError, match="Staging root"):
        builder.assemble_staged_bundle(changed, repo_root=repo)


def test_assembler_rejects_audit_and_training_rebuild_mismatch(staged_inputs):
    repo, inputs, _ = staged_inputs
    audit_features = repo / "data/processed/candidates/run_audit/features.csv"
    audit_features.write_text(
        audit_features.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(builder.StagingBundleError, match="approved controlling snapshot hash"):
        builder.assemble_staged_bundle(inputs, repo_root=repo)
    assert not (repo / ".codex_stage/new_bundle").exists()


def test_assembler_rejects_internally_matching_but_unapproved_snapshot(staged_inputs):
    repo, inputs, _ = staged_inputs
    for relative in (
        "data/processed/candidates/run_audit/features.csv",
        "data/processed/candidates/run_train/features.csv",
    ):
        path = repo / relative
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(builder.StagingBundleError, match="approved controlling snapshot hash"):
        builder.assemble_staged_bundle(inputs, repo_root=repo)


def test_assembler_rejects_scoped_source_drift_after_inventory(staged_inputs):
    repo, inputs, _ = staged_inputs
    (repo / "src/dummy.py").write_text("VALUE = 3\n", encoding="utf-8")

    with pytest.raises(builder.StagingBundleError, match="scoped files changed"):
        builder.assemble_staged_bundle(inputs, repo_root=repo)
    assert not (repo / ".codex_stage/new_bundle").exists()


def test_assembler_rejects_non_allowlisted_posttraining_inventory_delta(staged_inputs):
    repo, inputs, _ = staged_inputs
    (repo / "scripts/dummy.py").write_text("VALUE = 7\n", encoding="utf-8")
    widened = inventory_module.build_inventory(run_id="fixture-widened-assembly")
    _write_json(repo / "logs/assembly_inventory.json", widened)

    with pytest.raises(
        builder.StagingBundleError,
        match="only the allowlisted posttraining validation/packaging changes",
    ):
        builder.assemble_staged_bundle(inputs, repo_root=repo)
    assert not (repo / ".codex_stage/new_bundle").exists()


def test_assembler_rejects_old_model_fit_sha(staged_inputs):
    repo, inputs, _ = staged_inputs
    path = repo / "models/candidates/run_train/xgboost_model.pkl"
    result = joblib.load(path)
    result["training_spec"]["git_hash"] = "0" * 40
    joblib.dump(result, path)

    with pytest.raises(builder.StagingBundleError, match="embedded git_hash"):
        builder.assemble_staged_bundle(inputs, repo_root=repo)


def test_assembler_rejects_model_training_row_drift(staged_inputs):
    repo, inputs, _ = staged_inputs
    path = repo / "models/candidates/run_train/logistic_model.pkl"
    result = joblib.load(path)
    result["observed_training_rows"] = 999
    joblib.dump(result, path)

    with pytest.raises(builder.StagingBundleError, match="training-row metadata"):
        builder.assemble_staged_bundle(inputs, repo_root=repo)


def test_assembler_rejects_self_hashed_but_wrong_test_split(staged_inputs):
    repo, inputs, _ = staged_inputs
    processed = repo / "data/processed/candidates/run_train"
    wrong_test = pd.read_csv(processed / "features.csv").iloc[:1]
    test_path = processed / "test_set.csv"
    wrong_test.to_csv(test_path, index=False)
    metadata_path = processed / "test_set.csv.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["row_count"] = 1
    metadata["test_set_sha256"] = _sha256(test_path)
    _write_json(metadata_path, metadata)

    with pytest.raises(builder.StagingBundleError, match="split reconstructed"):
        builder.assemble_staged_bundle(inputs, repo_root=repo)


def test_rich_validator_rejects_post_build_evidence_tampering(staged_inputs):
    repo, inputs, _ = staged_inputs
    result = builder.assemble_staged_bundle(inputs, repo_root=repo)
    stage = Path(result["staging_root"])
    staged_evidence = stage / "evidence/logs/selection.json"
    staged_evidence.write_text('{"tampered": true}\n', encoding="utf-8")

    with pytest.raises(builder.StagingBundleError, match="rich identity"):
        builder.validate_rich_staged_manifest(stage / "staging_manifest.json")


def test_rich_validator_rejects_post_build_audit_tampering(staged_inputs):
    repo, inputs, _ = staged_inputs
    result = builder.assemble_staged_bundle(inputs, repo_root=repo)
    stage = Path(result["staging_root"])
    staged_audit = (
        stage / "provenance/independent_audit_snapshot/fights_cleaned.csv"
    )
    staged_audit.write_text(
        staged_audit.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(builder.StagingBundleError, match="rich identity"):
        builder.validate_rich_staged_manifest(stage / "staging_manifest.json")


def test_rich_validator_recomputes_cutoff_from_newest_snapshot(staged_inputs):
    repo, inputs, _ = staged_inputs
    result = builder.assemble_staged_bundle(inputs, repo_root=repo)
    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["immutable_training_snapshot"]["cutoff_safety"][
        "effective_buffer_days"
    ] += 1
    _write_json(manifest_path, manifest)

    with pytest.raises(builder.StagingBundleError, match="cutoff safety identity"):
        builder.validate_rich_staged_manifest(manifest_path)


def test_semantic_gate_accepts_benign_nonzero_csv_roundtrip_drift(tmp_path: Path):
    audit = _semantic_frame()
    trainer = audit.copy()
    trainer.loc[0, "plain_metric"] += 5e-13
    audit_path = tmp_path / "audit.csv"
    trainer_path = tmp_path / "trainer.csv"
    audit.to_csv(audit_path, index=False)
    trainer.to_csv(trainer_path, index=False)

    report = builder._semantic_csv_equivalence(
        audit_path,
        trainer_path,
        label="roundtrip regression",
    )

    assert report["equivalent"] is True
    assert report["byte_equal"] is False
    assert 0.0 < report["numeric_max_abs_delta"] <= 1e-12
    assert report["nan_masks_exact"] is True
    assert report["key_values_and_order_exact"] is True
    assert report["target_values_and_order_exact"] is True


def test_semantic_gate_rejects_numeric_drift_above_tolerance():
    audit = _semantic_frame()
    trainer = audit.copy()
    trainer.loc[0, "plain_metric"] += 2e-12

    with pytest.raises(builder.StagingBundleError, match="numeric drift exceeds"):
        builder._semantic_frame_equivalence(audit, trainer, label="features")


def test_semantic_gate_rejects_nonnumeric_drift():
    audit = _semantic_frame()
    trainer = audit.copy()
    trainer.loc[0, "note"] = "changed"

    with pytest.raises(builder.StagingBundleError, match="nonnumeric mismatch"):
        builder._semantic_frame_equivalence(audit, trainer, label="features")


def test_semantic_gate_rejects_nan_mask_drift():
    audit = _semantic_frame()
    trainer = audit.copy()
    trainer.loc[1, "plain_metric"] = 0.5

    with pytest.raises(builder.StagingBundleError, match="NaN-mask mismatch"):
        builder._semantic_frame_equivalence(audit, trainer, label="features")


def test_semantic_gate_rejects_key_drift():
    audit = _semantic_frame()
    trainer = audit.copy()
    trainer.loc[0, "fighter_a"] = "Not Alpha"

    with pytest.raises(builder.StagingBundleError, match="key mismatch"):
        builder._semantic_frame_equivalence(audit, trainer, label="features")


def test_prediction_gate_rejects_drift_even_when_semantics_are_within_tolerance():
    audit = _semantic_frame().iloc[:1].copy()
    trainer = audit.copy()
    trainer.loc[0, "plain_metric"] += 5e-13
    semantic = builder._semantic_frame_equivalence(
        audit,
        trainer,
        label="features",
    )
    assert semantic["equivalent"] is True
    sensitive_result = {
        "model": _LinearModel(),
        "feature_cols": ["plain_metric"],
        "impute_strategy": "native_nan",
        "col_medians": np.array([0.0]),
    }

    with pytest.raises(builder.StagingBundleError, match="not bit-identical"):
        builder._prediction_invariance(
            audit,
            trainer,
            model_results={"primary": sensitive_result},
        )


def test_assembler_rejects_sensitive_selection_evidence(staged_inputs):
    repo, inputs, _ = staged_inputs
    sensitive = repo / "logs/account_ledger.csv"
    sensitive.write_text("secret\n", encoding="utf-8")
    changed = builder.BundleInputs(
        **{**inputs.__dict__, "selection_evidence_paths": (sensitive.relative_to(repo),)}
    )

    with pytest.raises(builder.StagingBundleError, match="approved safe scope"):
        builder.assemble_staged_bundle(changed, repo_root=repo)


def test_assembler_does_not_reject_valid_snapshot_due_to_wall_clock(staged_inputs, monkeypatch):
    repo, inputs, _ = staged_inputs

    class _LateDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 12, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(builder, "datetime", _LateDatetime)
    result = builder.assemble_staged_bundle(inputs, repo_root=repo)

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    cutoff = manifest["immutable_training_snapshot"]["cutoff_safety"]
    assert cutoff["current_date_buffer_days"] < cutoff["required_minimum_buffer_days"]
    assert cutoff["effective_buffer_days"] == cutoff["snapshot_buffer_days"]


@pytest.mark.parametrize("mutation", ["short", "bad_decision"])
def test_bfo_gate_rejects_incomplete_or_malformed_ledger(
    staged_inputs, monkeypatch, mutation: str
):
    repo, _, _ = staged_inputs
    ledger = repo / "data/raw/historical_odds" / builder.APPROVED_BFO_LEDGER_NAME
    if mutation == "short":
        ledger.write_text('{"schema_version": 1}\n', encoding="utf-8")
        expected = "exactly 244 records"
    else:
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        rows[0]["decision"] = "maybe"
        ledger.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        expected = "invalid decision"
    monkeypatch.setattr(
        builder,
        "APPROVED_BFO_LEDGER_SHA256",
        builder._canonical_text_sha256(ledger),
    )
    current_inventory = inventory_module.build_inventory(run_id="bfo-negative")

    with pytest.raises(builder.StagingBundleError, match=expected):
        builder._validate_bfo_ledger(
            ledger,
            repo_root=repo,
            inventory_payload=current_inventory,
        )


def test_bfo_gate_rejects_corrected_csv_mismatch(staged_inputs):
    repo, _, _ = staged_inputs
    odds_dir = repo / "data/raw/historical_odds"
    csv_path = odds_dir / next(iter(builder.APPROVED_BFO_CSVS))
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    current_inventory = inventory_module.build_inventory(run_id="bfo-csv-negative")

    with pytest.raises(builder.StagingBundleError, match="CSV hash is not approved"):
        builder._validate_bfo_ledger(
            odds_dir / builder.APPROVED_BFO_LEDGER_NAME,
            repo_root=repo,
            inventory_payload=current_inventory,
        )
