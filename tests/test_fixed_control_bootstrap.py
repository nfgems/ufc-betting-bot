from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from src.model.training_spec import resolve_named_training_spec
from src.strategy import control_arm
from src.strategy import run_evaluation as run_eval


def _fake_runtime_code() -> dict:
    environment = {"schema_version": 1, "runtime": "test"}
    environment_sha = run_eval._canonical_json_sha256(environment)
    inventory = {
        "schema_version": 2,
        "source_contract": {"python_glob": "src/**/*.py", "required_scripts": []},
        "sources": [],
        "environment_payload_sha256": environment_sha,
    }
    fingerprint = run_eval._canonical_json_sha256(inventory)
    return {
        "git_sha": "deadbeef",
        "git_dirty": False,
        "source_fingerprint": fingerprint,
        "source_inventory_sha256": fingerprint,
        "source_inventory": inventory,
        "environment": environment,
        "environment_payload_sha256": environment_sha,
    }


def _bootstrap_args(spec_hash: str, *, stage: str = "1-3") -> SimpleNamespace:
    return SimpleNamespace(
        variants="baseline",
        datasets="pulled_all_plus_legacy_market",
        families="production",
        stage=stage,
        max_workers=1,
        max_finalists=1,
        retrain_months=6,
        bootstrap=run_eval.DEFAULT_BOOTSTRAP,
        no_narrow=True,
        execution_mode="realistic",
        entry_offset_days=1.0,
        entry_offset_for_features=True,
        require_entry_odds=True,
        allow_closing_odds=False,
        calibration_method=None,
        allow_resume_mismatch=False,
        freeze_id="integrity_v2_test",
        fixed_control_spec_name="full_live_contract_v6_integrity_205",
        fixed_control_spec_payload_sha256=spec_hash,
        expected_evaluation_sample_sha256="a" * 64,
        expected_evaluation_fights=1761,
        expected_evaluation_folds=10,
        resume=None,
    )


def _fixed_control_stage1_result(named_spec, spec_hash: str) -> dict:
    return {
        "status": "success",
        "model_variant": "baseline",
        "dataset_variant": "pulled_all_plus_legacy_market",
        "feature_family": "production",
        "calibration_method": named_spec.calibration_method,
        "evaluation_partition": "selection",
        "reserved_confirmation_folds": 2,
        "selection_fold_ids": list(range(1, 9)),
        "confirmation_fold_ids": [9, 10],
        "n_predictions": 1400,
        "n_folds": 8,
        "evaluation_sample_sha256": "b" * 64,
        "full_evaluation_sample_sha256": "a" * 64,
        "full_evaluation_n_fights": 1761,
        "full_evaluation_n_folds": 10,
        "confirmation_evaluation_sample_sha256": "c" * 64,
        "confirmation_evaluation_n_fights": 361,
        "confirmation_evaluation_n_folds": 2,
        "model_spec_name": named_spec.name,
        "model_spec_payload_sha256": spec_hash,
        "feature_contract_columns": list(named_spec.feature_cols),
        "feature_contract_count": len(named_spec.feature_cols),
        "feature_contract_sha256": run_eval._canonical_json_sha256(
            list(named_spec.feature_cols)
        ),
        "prediction_rows_sha256": "1" * 64,
        "prediction_values_sha256": "2" * 64,
        "input_provenance": {},
        "input_provenance_payload_sha256": "d" * 64,
        "odds_source_inventory": {},
        "metrics": {
            "brier": 0.2,
            "prediction_rows_sha256": "3" * 64,
            "prediction_values_sha256": "4" * 64,
        },
        "sliced_metrics": {"fresh_window": {"brier": 0.2}},
    }


def test_fixed_control_cli_binds_one_cell_spec_and_honest_odds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named_spec = resolve_named_training_spec("full_live_contract_v6_integrity_205")
    spec_hash = run_eval._canonical_json_sha256(asdict(named_spec))
    args = _bootstrap_args(spec_hash)
    parser = argparse.ArgumentParser()
    original_baseline = run_eval.ALL_VARIANTS["baseline"]
    observed: dict[str, object] = {}

    monkeypatch.setattr(run_eval, "_resolve_run_dir", lambda **_kwargs: tmp_path)
    monkeypatch.setattr(run_eval, "_configure_logging", lambda _path: None)
    monkeypatch.setattr(
        run_eval,
        "_collect_runtime_code_metadata",
        _fake_runtime_code,
    )
    monkeypatch.setattr(run_eval, "_feature_cache_format", lambda: "pickle")

    def fake_stage1(**kwargs):
        variant = run_eval.ALL_VARIANTS["baseline"]()
        observed["stage1_kwargs"] = kwargs
        observed["variant"] = variant
        return [_fixed_control_stage1_result(named_spec, spec_hash)]

    monkeypatch.setattr(run_eval, "_run_stage1", fake_stage1)
    monkeypatch.setattr(
        run_eval,
        "_run_fixed_control_stage3_candidate",
        lambda **kwargs: observed.setdefault("stage3_kwargs", kwargs),
    )

    run_eval._run_fixed_control_bootstrap_cli(args, parser)

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["fixed_control_bootstrap"] is True
    assert metadata["selected_model_spec_name"] == named_spec.name
    assert metadata["selected_model_spec_payload_sha256"] == spec_hash
    assert metadata["datasets"] == ["pulled_all_plus_legacy_market"]
    assert metadata["entry_offset_days"] == 1.0
    assert metadata["entry_offset_for_features"] is True
    assert metadata["require_entry_odds"] is True
    assert metadata["allow_closing_odds"] is False
    assert metadata["execution_mode"] == "realistic"
    assert metadata["reserved_confirmation_folds"] == 2
    variant = observed["variant"]
    assert variant.feature_cols == named_spec.feature_cols
    assert variant.odds_noise_seed == 42
    assert variant.odds_noise_mode == "antithetic"
    stage3_payload = observed["stage3_kwargs"]["payload"]
    assert stage3_payload["candidate_id"] == (
        "baseline_pulled_all_plus_legacy_market_production"
    )
    assert stage3_payload["model_spec_payload_sha256"] == spec_hash
    assert stage3_payload["evaluation_sample_sha256"] == "b" * 64
    assert stage3_payload["full_evaluation_sample_sha256"] == "a" * 64
    assert stage3_payload["prediction_rows_sha256"] == "1" * 64
    assert stage3_payload["prediction_values_sha256"] == "2" * 64
    assert run_eval.ALL_VARIANTS["baseline"] is original_baseline


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("prediction_rows_sha256", None),
        ("prediction_rows_sha256", "not-a-sha256"),
        ("prediction_values_sha256", None),
        ("prediction_values_sha256", "not-a-sha256"),
    ],
)
def test_fixed_control_cli_rejects_missing_or_malformed_stage1_prediction_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad_value: str | None,
) -> None:
    named_spec = resolve_named_training_spec("full_live_contract_v6_integrity_205")
    spec_hash = run_eval._canonical_json_sha256(asdict(named_spec))
    stage1_result = _fixed_control_stage1_result(named_spec, spec_hash)
    if bad_value is None:
        stage1_result.pop(field)
    else:
        stage1_result[field] = bad_value

    monkeypatch.setattr(run_eval, "_resolve_run_dir", lambda **_kwargs: tmp_path)
    monkeypatch.setattr(run_eval, "_configure_logging", lambda _path: None)
    monkeypatch.setattr(run_eval, "_collect_runtime_code_metadata", _fake_runtime_code)
    monkeypatch.setattr(run_eval, "_feature_cache_format", lambda: "pickle")
    monkeypatch.setattr(run_eval, "_run_stage1", lambda **_kwargs: [stage1_result])
    monkeypatch.setattr(
        run_eval,
        "_run_fixed_control_stage3_candidate",
        lambda **_kwargs: pytest.fail("Stage 3 must not run"),
    )

    with pytest.raises(ValueError, match=rf"top-level {field}"):
        run_eval._run_fixed_control_bootstrap_cli(
            _bootstrap_args(spec_hash), argparse.ArgumentParser()
        )

    assert not (tmp_path / "stage3_sweeps").exists()


def test_fixed_control_cli_refuses_stage4(tmp_path: Path) -> None:
    named_spec = resolve_named_training_spec("full_live_contract_v6_integrity_205")
    spec_hash = run_eval._canonical_json_sha256(asdict(named_spec))
    args = _bootstrap_args(spec_hash, stage="1-4")

    with pytest.raises(SystemExit):
        run_eval._run_fixed_control_bootstrap_cli(args, argparse.ArgumentParser())
    assert not any(tmp_path.iterdir())


def _prediction_folds() -> list[tuple[int, pd.DataFrame]]:
    named_spec = resolve_named_training_spec("full_live_contract_v6_integrity_205")
    folds: list[tuple[int, pd.DataFrame]] = []
    for fold, event_date in enumerate(
        ("2025-01-01", "2025-07-01", "2026-01-01"),
        start=1,
    ):
        frame = pd.DataFrame(
            {
                "event_date": pd.to_datetime([event_date]),
                "fighter_a": [f"A{fold}"],
                "fighter_b": [f"B{fold}"],
                "target": [fold % 2],
                "fold": [fold],
                "train_end": pd.to_datetime([event_date]),
                "test_end": pd.to_datetime([event_date]) + pd.Timedelta(days=30),
            }
        )
        for column in named_spec.feature_cols:
            if column not in frame.columns:
                frame[column] = 0.0
        # Exercise the exact publication seam: CSV must preserve genuine
        # empty strings separately from missing values, format every implicated
        # provenance timestamp like the input-value hash, and retain round-trip
        # floats. Historical opening/entry timestamps are UTC-aware while the
        # model-odds copies are deliberately naive, matching the failed run.
        for prefix in ("opening", "entry", "model_odds"):
            missing_source = pd.NA if prefix != "model_odds" else ""
            frame[f"{prefix}_source_kind"] = [
                "" if fold == 1 else "historical" if fold == 2 else missing_source
            ]
            frame[f"{prefix}_source_file"] = [
                "" if fold == 1 else "odds.csv" if fold == 2 else missing_source
            ]
            observed_at = (
                pd.NaT
                if fold == 1
                else pd.Timestamp(f"{event_date} 12:34:56.123456")
            )
            commence_time = (
                pd.NaT
                if fold == 1
                else pd.Timestamp(f"{event_date} 23:45:01.654321")
            )
            frame[f"{prefix}_observed_at"] = pd.to_datetime(
                [observed_at],
                utc=(prefix != "model_odds"),
            )
            frame[f"{prefix}_commence_time"] = pd.to_datetime(
                [commence_time],
                utc=(prefix != "model_odds"),
            )
            if fold == 2:
                verified_prefight = True
            elif fold == 3 and prefix != "model_odds":
                verified_prefight = pd.NA
            else:
                verified_prefight = False
            frame[f"{prefix}_verified_prefight"] = [verified_prefight]
        frame["entry_hours_to_start"] = [
            float("nan") if fold == 1 else 24.123456789012345
        ]
        frame["prob_a"] = 0.6
        frame["prob_b"] = 0.4
        frame["no_odds_prob_a"] = 0.55
        frame["no_odds_prob_b"] = 0.45
        folds.append((fold, frame))
    return folds


@pytest.mark.filterwarnings(
    "ignore:DataFrame is highly fragmented.*:pandas.errors.PerformanceWarning"
)
def test_stage1_cell_publishes_exact_evaluation_protocol_for_freeze_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real Stage-1 producer must publish the protocol the freezer reads."""

    from src.strategy import model_lab

    named_spec = resolve_named_training_spec("full_live_contract_v6_integrity_205")
    spec_hash = run_eval._canonical_json_sha256(asdict(named_spec))
    fold_manifest = _prediction_folds()
    selection_folds = fold_manifest[:1]
    features = pd.concat(
        [frame for _fold_id, frame in fold_manifest],
        ignore_index=True,
    )
    features_path = tmp_path / "features.pkl"
    run_eval._save_cached_frame(features, features_path, "pickle")
    fights_path = tmp_path / "fights.csv"
    fights_path.write_text(
        "event_date,fighter_a,fighter_b\n2025-01-01,A1,B1\n",
        encoding="utf-8",
    )
    source_inventory_path = tmp_path / run_eval.SOURCE_INVENTORY_FILENAME
    run_eval._write_json(source_inventory_path, {"schema_version": 2})
    environment_path = tmp_path / run_eval.ENVIRONMENT_INVENTORY_FILENAME
    environment = {"schema_version": 1, "runtime": "protocol-test"}
    run_eval._write_json(environment_path, environment)
    evaluation_protocol = {
        "protocol_probe": "stage1-to-freeze-cli",
        "reserved_confirmation_folds": 2,
    }
    policy_provenance = {
        "policy_path": str(tmp_path / "policy.json"),
        "policy_sha256": "1" * 64,
        "scheduled_protocol_sha256": "2" * 64,
        "corrected_fights_path": str(fights_path.resolve()),
        "corrected_fights_sha256": run_eval._file_sha256(fights_path),
        "corrected_features_path": str(features_path.resolve()),
        "corrected_features_sha256": run_eval._file_sha256(features_path),
        "source_dataset_fights_path": str(fights_path.resolve()),
        "source_dataset_fights_sha256": run_eval._file_sha256(fights_path),
        "source_features_path": str(features_path.resolve()),
        "source_features_sha256": run_eval._file_sha256(features_path),
    }
    odds_inventory = {
        "schema_version": 1,
        "evaluation_protocol": evaluation_protocol,
        "entries": [],
    }

    monkeypatch.setattr(
        model_lab,
        "_materialize_variant_contract_features",
        lambda frame, _variant: frame,
    )
    monkeypatch.setattr(
        run_eval,
        "resolve_variant_feature_columns",
        lambda *_args, **_kwargs: (list(named_spec.feature_cols), []),
    )
    monkeypatch.setattr(
        run_eval,
        "generate_variant_fold_predictions",
        lambda *_args, **_kwargs: (selection_folds, fold_manifest),
    )
    monkeypatch.setattr(
        run_eval,
        "_compute_model_metrics",
        lambda *_args, **_kwargs: {
            "accuracy": 0.5,
            "brier": 0.25,
            "log_loss": 0.7,
            "ece": 0.1,
        },
    )
    monkeypatch.setattr(
        run_eval,
        "_compute_sliced_metrics",
        lambda *_args, **_kwargs: {},
    )

    result = run_eval._evaluate_cell_worker(
        {
            "cell": asdict(
                run_eval.CellSpec(
                    model_variant="baseline",
                    dataset_variant="pulled_all_plus_legacy_market",
                    feature_family="production",
                    calibration_method=named_spec.calibration_method,
                    retrain_months=6,
                    bootstrap=run_eval.DEFAULT_BOOTSTRAP,
                )
            ),
            "features_path": str(features_path),
            "dataset_fights_path": str(fights_path),
            "reserved_confirmation_folds": 2,
            "model_spec_name": named_spec.name,
            "model_spec_payload_sha256": spec_hash,
            "evaluation_protocol": evaluation_protocol,
            "policy_provenance": policy_provenance,
            "odds_source_inventory": odds_inventory,
            "source_fingerprint": "3" * 64,
            "source_inventory_sha256": "3" * 64,
            "source_inventory_path": str(source_inventory_path),
            "source_inventory_artifact_sha256": run_eval._file_sha256(
                source_inventory_path
            ),
            "environment_path": str(environment_path),
            "environment_artifact_sha256": run_eval._file_sha256(environment_path),
            "environment_payload_sha256": run_eval._canonical_json_sha256(
                environment
            ),
            "entry_offset_days": 1.0,
            "entry_offset_for_features": True,
            "require_entry_odds": True,
            "allow_closing_odds": False,
        }
    )
    cell_path = tmp_path / "stage1_cell.json"
    run_eval._write_json(cell_path, result)
    published = json.loads(cell_path.read_text(encoding="utf-8"))

    assert published["evaluation_protocol"] == evaluation_protocol
    assert (
        published["evaluation_protocol"]
        == published["input_provenance"]["evaluation_protocol"]
    )
    assert published["evaluation_protocol_sha256"] == run_eval._canonical_json_sha256(
        evaluation_protocol
    )


def _attach_fixed_test_provenance(
    tmp_path: Path,
    payload: dict,
    manifest: list[tuple[int, pd.DataFrame]],
    selection_folds: list[tuple[int, pd.DataFrame]],
) -> dict:
    named_spec = resolve_named_training_spec("full_live_contract_v6_integrity_205")
    spec_hash = run_eval._canonical_json_sha256(asdict(named_spec))
    evidence = run_eval._fold_partition_evidence(
        manifest,
        selection_folds,
        reserved_confirmation_folds=2,
        feature_contract_columns=list(named_spec.feature_cols),
    )
    contract = run_eval._ordered_feature_contract_evidence(
        list(named_spec.feature_cols)
    )
    evidence.update(contract)
    source_inventory_path = tmp_path / run_eval.SOURCE_INVENTORY_FILENAME
    source_inventory_path.write_text("{}", encoding="utf-8")
    environment_path = tmp_path / run_eval.ENVIRONMENT_INVENTORY_FILENAME
    environment = {"schema_version": 1, "runtime": "test"}
    run_eval._write_json(environment_path, environment)
    odds_inventory = {"schema_version": 1, "evaluation_protocol": {}, "entries": []}
    input_provenance = {
        **evidence,
        **contract,
        "evaluation_protocol": {},
        "input_provenance_payload_sha256": "7" * 64,
        "dataset_fights_sha256": "8" * 64,
        "dataset_fights_path": str(tmp_path / "fights.csv"),
        "source_dataset_fights_path": str(tmp_path / "fights.csv"),
        "source_dataset_fights_sha256": "8" * 64,
        "features_artifact_sha256": "9" * 64,
        "features_artifact_path": str(tmp_path / "features.csv"),
        "features_value_sha256": "a" * 64,
        "source_features_path": str(tmp_path / "features.csv"),
        "source_features_sha256": "9" * 64,
        "policy_sha256": "b" * 64,
        "scheduled_protocol_sha256": "c" * 64,
        "evaluation_protocol_sha256": "d" * 64,
        "source_inventory_sha256": "e" * 64,
        "source_inventory_path": str(source_inventory_path),
        "source_inventory_artifact_sha256": run_eval._file_sha256(source_inventory_path),
        "environment_path": str(environment_path),
        "environment_artifact_sha256": run_eval._file_sha256(environment_path),
        "environment_payload_sha256": run_eval._canonical_json_sha256(environment),
        "model_spec_name": named_spec.name,
        "model_spec_payload_sha256": spec_hash,
    }
    input_provenance["odds_source_inventory_sha256"] = run_eval._json_artifact_sha256(
        odds_inventory
    )
    payload.update(evidence)
    payload.update(contract)
    selection_predictions = run_eval._post_cutoff_predictions(selection_folds)
    payload.update(
        {
            "calibration_method": named_spec.calibration_method,
            "model_spec_name": named_spec.name,
            "model_spec_payload_sha256": spec_hash,
            "prediction_rows_sha256": run_eval._prediction_rows_sha256(
                selection_predictions
            ),
            "prediction_values_sha256": run_eval._prediction_values_sha256(
                selection_predictions
            ),
            "input_provenance": input_provenance,
            "input_provenance_payload_sha256": input_provenance[
                "input_provenance_payload_sha256"
            ],
            "odds_source_inventory": odds_inventory,
            "odds_source_inventory_sha256": input_provenance[
                "odds_source_inventory_sha256"
            ],
        }
    )
    return payload


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("prediction_rows_sha256", None),
        ("prediction_rows_sha256", "not-a-sha256"),
        ("prediction_values_sha256", None),
        ("prediction_values_sha256", "not-a-sha256"),
    ],
)
def test_fixed_control_stage3_rejects_missing_or_malformed_prediction_hash_before_io(
    tmp_path: Path,
    field: str,
    bad_value: str | None,
) -> None:
    payload = {
        "candidate_id": "baseline_pulled_all_plus_legacy_market_production",
        "prediction_rows_sha256": "1" * 64,
        "prediction_values_sha256": "2" * 64,
    }
    if bad_value is None:
        payload.pop(field)
    else:
        payload[field] = bad_value

    with pytest.raises(ValueError, match=rf"top-level {field}"):
        run_eval._run_fixed_control_stage3_candidate(
            run_dir=tmp_path,
            payload=payload,
            metadata={},
        )

    assert not (tmp_path / "stage3_sweeps").exists()


def test_fixed_control_stage3_writes_only_production_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _prediction_folds()
    selection_folds = manifest[:1]
    full_predictions = pd.concat([frame for _fold, frame in manifest], ignore_index=True)
    sample_hash = run_eval._evaluation_sample_sha256(full_predictions)
    named_spec = resolve_named_training_spec("full_live_contract_v6_integrity_205")
    spec_hash = run_eval._canonical_json_sha256(asdict(named_spec))
    payload = {
        "candidate_id": "baseline_pulled_all_plus_legacy_market_production",
        "model_variant": "baseline",
        "dataset_variant": "pulled_all_plus_legacy_market",
        "feature_family": "production",
        "calibration_method": named_spec.calibration_method,
        "retrain_months": 6,
    }
    _attach_fixed_test_provenance(tmp_path, payload, manifest, selection_folds)
    metadata = {
        "source_fingerprint": "source",
        "execution_mode": "realistic",
        "entry_offset_days": 1.0,
        "entry_offset_for_features": True,
        "require_entry_odds": True,
        "allow_closing_odds": False,
        "selected_model_spec_name": "full_live_contract_v6_integrity_205",
        "selected_model_spec_payload_sha256": spec_hash,
        "expected_evaluation_sample_sha256": sample_hash,
        "expected_evaluation_n_fights": 3,
        "expected_evaluation_n_folds": 3,
        "reserved_confirmation_folds": 2,
        "bootstrap": 5000,
    }
    monkeypatch.setattr(
        run_eval,
        "_generate_walk_forward_predictions",
        lambda **_kwargs: (selection_folds, manifest),
    )
    monkeypatch.setattr(
        run_eval,
        "_validate_input_provenance_files",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        run_eval,
        "_evaluate_config",
        lambda *_args, **_kwargs: {
            "roi": 0.1,
            "total_profit": 5.0,
            "total_bets": 1,
            "max_drawdown": 0.01,
            "avg_clv": 0.02,
            "bet_log": pd.DataFrame(
                [{"event_date": "2025-01-01", "profit": 5.0}]
            ),
            "bankroll_history": pd.DataFrame([{"combined": 505.0}]),
        },
    )
    monkeypatch.setattr(
        run_eval,
        "_summary_row_from_result",
        lambda config, _result: {
            "config": config.name,
            "is_baseline": True,
            "total_bets": 1,
        },
    )

    result = run_eval._run_fixed_control_stage3_candidate(
        run_dir=tmp_path,
        payload=payload,
        metadata=metadata,
    )

    candidate_dir = (
        tmp_path
        / "stage3_sweeps"
        / "baseline_pulled_all_plus_legacy_market_production"
    )
    assert result["promotion_eligible"] is True
    assert (candidate_dir / "production_result.json").exists()
    assert (candidate_dir / "production_bet_log.csv").exists()
    assert (candidate_dir / "production_bankroll_history.csv").exists()
    assert not list(candidate_dir.glob("broad_*"))
    assert not list(candidate_dir.glob("narrow_*"))
    assert not list(candidate_dir.glob("best_*"))
    receipt = json.loads(
        (tmp_path / "fixed_control_bootstrap_receipt.json").read_text()
    )
    assert receipt["research_grids_evaluated"] is False
    assert receipt["stage2_bypassed"] is True
    assert receipt["full_evaluation_sample_sha256"] == sample_hash
    assert receipt["evaluation_partition"] == "selection"
    assert receipt["confirmation_fold_ids"] == [2, 3]
    assert receipt["model_spec_payload_sha256"] == spec_hash
    for field in run_eval._FIXED_CONTROL_PREDICTION_HASH_FIELDS:
        assert result["fixed_control_evidence"][field] == payload[field]
        assert receipt[field] == payload[field]


@pytest.mark.parametrize(
    "field",
    ["prediction_rows_sha256", "prediction_values_sha256"],
)
def test_fixed_control_stage3_rejects_valid_but_wrong_prediction_hash_before_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    manifest = _prediction_folds()
    selection_folds = manifest[:1]
    full_predictions = pd.concat([frame for _fold, frame in manifest], ignore_index=True)
    sample_hash = run_eval._evaluation_sample_sha256(full_predictions)
    named_spec = resolve_named_training_spec("full_live_contract_v6_integrity_205")
    spec_hash = run_eval._canonical_json_sha256(asdict(named_spec))
    payload = {
        "candidate_id": "baseline_pulled_all_plus_legacy_market_production",
        "model_variant": "baseline",
        "dataset_variant": "pulled_all_plus_legacy_market",
        "feature_family": "production",
        "calibration_method": named_spec.calibration_method,
        "retrain_months": 6,
    }
    _attach_fixed_test_provenance(tmp_path, payload, manifest, selection_folds)
    payload[field] = "f" * 64
    metadata = {
        "source_fingerprint": "source",
        "execution_mode": "realistic",
        "entry_offset_days": 1.0,
        "entry_offset_for_features": True,
        "require_entry_odds": True,
        "allow_closing_odds": False,
        "selected_model_spec_name": named_spec.name,
        "selected_model_spec_payload_sha256": spec_hash,
        "expected_evaluation_sample_sha256": sample_hash,
        "expected_evaluation_n_fights": 3,
        "expected_evaluation_n_folds": 3,
        "reserved_confirmation_folds": 2,
        "bootstrap": 5000,
    }
    generation_calls = []

    def generate(**kwargs):
        generation_calls.append(kwargs)
        return selection_folds, manifest

    monkeypatch.setattr(run_eval, "_generate_walk_forward_predictions", generate)
    monkeypatch.setattr(
        run_eval,
        "_validate_input_provenance_files",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        run_eval,
        "_evaluate_config",
        lambda *_args, **_kwargs: pytest.fail("strategy must not run"),
    )

    with pytest.raises(ValueError, match=rf"{field} differs from Stage 1"):
        run_eval._run_fixed_control_stage3_candidate(
            run_dir=tmp_path,
            payload=payload,
            metadata=metadata,
        )

    assert len(generation_calls) == 1
    candidate_dir = (
        tmp_path
        / "stage3_sweeps"
        / "baseline_pulled_all_plus_legacy_market_production"
    )
    assert not (candidate_dir / "production_result.json").exists()
    assert not (tmp_path / "fixed_control_evaluation_index.csv").exists()
    assert not (tmp_path / "fixed_control_bootstrap_receipt.json").exists()


def test_fixed_control_stage3_fails_before_strategy_on_sample_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _prediction_folds()
    selection_folds = manifest[:1]
    named_spec = resolve_named_training_spec("full_live_contract_v6_integrity_205")
    spec_hash = run_eval._canonical_json_sha256(asdict(named_spec))
    payload = {
        "candidate_id": "baseline_pulled_all_plus_legacy_market_production",
        "model_variant": "baseline",
        "dataset_variant": "pulled_all_plus_legacy_market",
        "feature_family": "production",
        "calibration_method": named_spec.calibration_method,
        "retrain_months": 6,
    }
    metadata = {
        "source_fingerprint": "source",
        "execution_mode": "realistic",
        "entry_offset_days": 1.0,
        "entry_offset_for_features": True,
        "require_entry_odds": True,
        "allow_closing_odds": False,
        "selected_model_spec_name": "full_live_contract_v6_integrity_205",
        "selected_model_spec_payload_sha256": spec_hash,
        "expected_evaluation_sample_sha256": "c" * 64,
        "expected_evaluation_n_fights": 3,
        "expected_evaluation_n_folds": 3,
        "reserved_confirmation_folds": 2,
        "bootstrap": 5000,
    }
    _attach_fixed_test_provenance(tmp_path, payload, manifest, selection_folds)
    monkeypatch.setattr(
        run_eval,
        "_generate_walk_forward_predictions",
        lambda **_kwargs: (selection_folds, manifest),
    )
    monkeypatch.setattr(
        run_eval,
        "_validate_input_provenance_files",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        run_eval,
        "_evaluate_config",
        lambda *_args, **_kwargs: pytest.fail("strategy must not run"),
    )

    with pytest.raises(ValueError, match="evaluation index mismatch"):
        run_eval._run_fixed_control_stage3_candidate(
            run_dir=tmp_path,
            payload=payload,
            metadata=metadata,
        )


def test_control_freezer_preserves_and_cross_checks_fixed_control_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    cells = run_dir / "cells"
    candidate_id = "baseline_pulled_all_plus_legacy_market_production"
    sweep = run_dir / "stage3_sweeps" / candidate_id
    cells.mkdir(parents=True)
    sweep.mkdir(parents=True)
    sample_hash = "a" * 64
    spec_hash = "b" * 64
    cell_payload = {
        "metrics": {
            "accuracy": 0.6,
            "brier": 0.2,
            "log_loss": 0.6,
            "ece": 0.04,
        },
        "sliced_metrics": {
            "fresh_window": {"brier": 0.2},
            "by_year": {"2025": {"brier": 0.2, "ece": 0.04}},
        },
        "n_predictions": 1761,
        "n_folds": 10,
        "model_variant": "baseline",
        "feature_family": "production",
        "dataset_variant": "pulled_all_plus_legacy_market",
        "calibration_method": "sigmoid",
        "evaluation_sample_sha256": sample_hash,
        "model_spec_name": "full_live_contract_v6_integrity_205",
        "model_spec_payload_sha256": spec_hash,
    }
    (cells / "baseline__pulled_all_plus_legacy_market__production__sigmoid_metrics.json").write_text(
        json.dumps(cell_payload),
        encoding="utf-8",
    )
    fixed_evidence = {
        "model_spec_name": cell_payload["model_spec_name"],
        "model_spec_payload_sha256": spec_hash,
        "evaluation_sample_sha256": sample_hash,
        "evaluation_n_fights": 1761,
        "evaluation_n_folds": 10,
    }
    (sweep / "production_result.json").write_text(
        json.dumps(
            {
                "promotion_eligible": True,
                "roi": 0.01,
                "total_profit": 1.0,
                "total_bets": 1,
                "max_drawdown": 0.01,
                "fixed_control_evidence": fixed_evidence,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"event_date": "2025-01-01", "profit": 1.0}]).to_csv(
        sweep / "production_bet_log.csv",
        index=False,
    )
    pd.DataFrame([{"combined": 501.0}]).to_csv(
        sweep / "production_bankroll_history.csv",
        index=False,
    )
    frozen = tmp_path / "frozen"
    monkeypatch.setattr(control_arm, "FROZEN_DIR", frozen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "control_arm",
            "--run-dir",
            str(run_dir),
            "--candidate-id",
            candidate_id,
            "--freeze-id",
            "fixed_control_test",
        ],
    )

    control_arm.main()

    metrics = json.loads(
        (
            frozen
            / "control_arm_fixed_control_test"
            / "control_metrics.json"
        ).read_text()
    )
    assert metrics["model_spec_name"] == cell_payload["model_spec_name"]
    assert metrics["model_spec_payload_sha256"] == spec_hash
    assert metrics["evaluation_sample_sha256"] == sample_hash


def _write_integrity_v2_protocol_cli_fixture(
    tmp_path: Path,
    *,
    cell_protocol_case: str,
) -> tuple[Path, str, dict]:
    run_dir = tmp_path / "protocol_cli_run"
    cells_dir = run_dir / "cells"
    candidate_id = "baseline_pulled_all_plus_legacy_market_production"
    sweep_dir = run_dir / "stage3_sweeps" / candidate_id
    cells_dir.mkdir(parents=True)
    sweep_dir.mkdir(parents=True)
    fixed_protocol = {
        "protocol_probe": "exact-fixed-control-protocol",
        "reserved_confirmation_folds": 2,
    }
    compared_fields = tuple(
        dict.fromkeys(
            (
                *control_arm._INTEGRITY_V2_BINDINGS,
                *control_arm._INTEGRITY_V2_PARTITION_BINDING_FIELDS,
                "source_fingerprint",
                *(
                    field
                    for field in control_arm._INTEGRITY_V2_DYNAMIC_PROVENANCE_FIELDS
                    if field
                    not in control_arm._INTEGRITY_V2_STAGE1_MATERIALIZED_PATH_FIELDS
                ),
            )
        )
    )
    shared_evidence = {
        field: f"bound-value-for-{field}" for field in compared_fields
    }
    shared_evidence["evaluation_protocol_sha256"] = (
        run_eval._canonical_json_sha256(fixed_protocol)
    )
    cell_payload = {
        "metrics": {
            "accuracy": 0.6,
            "brier": 0.2,
            "log_loss": 0.6,
            "ece": 0.04,
        },
        "sliced_metrics": {
            "fresh_window": {"brier": 0.2},
            "by_year": {"2025": {"brier": 0.2, "ece": 0.04}},
        },
        "model_variant": "baseline",
        "feature_family": "production",
        "dataset_variant": "pulled_all_plus_legacy_market",
        "calibration_method": "sigmoid",
        "retrain_months": 6,
        **shared_evidence,
    }
    if cell_protocol_case == "matching":
        cell_payload["evaluation_protocol"] = dict(fixed_protocol)
    elif cell_protocol_case == "drifted":
        cell_payload["evaluation_protocol"] = {
            **fixed_protocol,
            "protocol_probe": "drifted-cell-protocol",
        }
    elif cell_protocol_case != "missing":
        raise AssertionError(f"unknown protocol case: {cell_protocol_case}")
    cell_path = (
        cells_dir
        / "baseline__pulled_all_plus_legacy_market__production__sigmoid_metrics.json"
    )
    cell_path.write_text(json.dumps(cell_payload), encoding="utf-8")

    fixed_evidence = {
        **shared_evidence,
        "evaluation_protocol": fixed_protocol,
    }
    (sweep_dir / "production_result.json").write_text(
        json.dumps(
            {
                "promotion_eligible": True,
                "fixed_control_evidence": fixed_evidence,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"event_date": "2025-01-01", "profit": 1.0}]).to_csv(
        sweep_dir / "production_bet_log.csv",
        index=False,
    )
    pd.DataFrame([{"combined": 501.0}]).to_csv(
        sweep_dir / "production_bankroll_history.csv",
        index=False,
    )
    return run_dir, candidate_id, fixed_protocol


@pytest.mark.parametrize(
    ("cell_protocol_case", "should_reach_freeze"),
    [
        ("matching", True),
        ("missing", False),
        ("drifted", False),
    ],
)
def test_integrity_v2_freeze_cli_requires_exact_stage1_evaluation_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cell_protocol_case: str,
    should_reach_freeze: bool,
) -> None:
    run_dir, candidate_id, fixed_protocol = _write_integrity_v2_protocol_cli_fixture(
        tmp_path,
        cell_protocol_case=cell_protocol_case,
    )
    freeze_calls: list[dict] = []

    def fake_freeze_control_arm(**kwargs):
        freeze_calls.append(kwargs)
        return tmp_path / "mocked-freeze-boundary"

    monkeypatch.setattr(control_arm, "freeze_control_arm", fake_freeze_control_arm)
    monkeypatch.setattr(
        control_arm,
        "validate_frozen_control_arm",
        lambda _freeze_id: {"valid": True, "errors": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "control_arm",
            "--run-dir",
            str(run_dir),
            "--candidate-id",
            candidate_id,
            "--freeze-id",
            "integrity_v2_protocol_cli_test",
        ],
    )

    if should_reach_freeze:
        control_arm.main()
        assert len(freeze_calls) == 1
        assert freeze_calls[0]["source_metrics"]["evaluation_protocol"] == (
            fixed_protocol
        )
    else:
        with pytest.raises(SystemExit) as exc_info:
            control_arm.main()
        assert exc_info.value.code == 1
        assert freeze_calls == []
