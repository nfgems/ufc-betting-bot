"""Tests for the control arm module."""

import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

import src.strategy.control_arm as control_arm
from src.model.training_spec import resolve_named_training_spec
from src.strategy.control_arm import (
    freeze_control_arm,
    load_frozen_control_metrics,
    load_frozen_control_trading_artifacts,
    load_frozen_sweep_summary,
    validate_frozen_control_arm,
    validate_frozen_control_arm_for_promotion_gate,
    validate_frozen_control_arm_for_selection_gate,
)

@pytest.fixture
def mock_frozen_root(tmp_path, monkeypatch):
    """Redirect FROZEN_DIR to a temporary directory."""
    monkeypatch.setattr(control_arm, "FROZEN_DIR", tmp_path)
    return tmp_path

def test_freeze_creates_directory_and_files(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.65,
        "brier": 0.21,
        "log_loss": 0.58,
        "ece": 0.04,
        "extra": "ignore me"
    }
    freeze_id = "test_freeze_2026"
    
    # freeze_control_arm returns Path
    result_path = freeze_control_arm(source_metrics, freeze_id)
    
    freeze_dir = mock_frozen_root / "control_arm_test_freeze_2026"
    assert freeze_dir.exists()
    assert result_path == freeze_dir
    assert (freeze_dir / "control_metrics.json").exists()
    assert (freeze_dir / "MANIFEST.md").exists()
    assert (freeze_dir / "checksums.json").exists()

def test_freeze_requires_metrics_keys(mock_frozen_root):
    # Missing 'ece'
    source_metrics = {
        "accuracy": 0.65,
        "brier": 0.21,
        "log_loss": 0.58
    }
    with pytest.raises(ValueError, match="Source metrics missing required keys"):
        freeze_control_arm(source_metrics, "fail_freeze")


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("metric", ["accuracy", "brier", "log_loss", "ece"])
def test_freeze_rejects_missing_or_nonfinite_core_metrics(
    mock_frozen_root,
    metric,
    value,
):
    source_metrics = {
        "accuracy": 0.65,
        "brier": 0.21,
        "log_loss": 0.58,
        "ece": 0.04,
    }
    source_metrics[metric] = value

    with pytest.raises(ValueError, match="finite numbers"):
        freeze_control_arm(source_metrics, f"bad_{metric}")

def test_validate_passes_for_valid_freeze(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.65,
        "brier": 0.21,
        "log_loss": 0.58,
        "ece": 0.04
    }
    freeze_id = "valid_freeze"
    freeze_control_arm(source_metrics, freeze_id)
    
    validation = validate_frozen_control_arm(freeze_id)
    assert validation["valid"] is True

def test_validate_detects_missing_files(mock_frozen_root):
    freeze_id = "missing_files"
    freeze_dir = mock_frozen_root / f"control_arm_{freeze_id}"
    freeze_dir.mkdir(parents=True)
    # Missing checksums.json and control_metrics.json
    
    validation = validate_frozen_control_arm(freeze_id)
    assert validation["valid"] is False
    assert any("Missing required file" in err for err in validation["errors"])

def test_validate_detects_checksum_mismatch(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.65,
        "brier": 0.21,
        "log_loss": 0.58,
        "ece": 0.04
    }
    freeze_id = "tampered_freeze"
    freeze_control_arm(source_metrics, freeze_id)
    
    # Tamper with the metrics file
    metrics_path = mock_frozen_root / f"control_arm_{freeze_id}" / "control_metrics.json"
    content = json.loads(metrics_path.read_text())
    content["accuracy"] = 0.99 # Change value
    metrics_path.write_text(json.dumps(content))
    
    validation = validate_frozen_control_arm(freeze_id)
    assert validation["valid"] is False
    assert any("Checksum mismatch" in err for err in validation["errors"])

def test_load_metrics_returns_correct_values(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.72,
        "brier": 0.18,
        "log_loss": 0.52,
        "ece": 0.02,
        "fresh_data_brier": 0.19,
        "sliced_metrics": {"fresh_window": {"brier": 0.19, "n_samples": 42}},
    }
    freeze_id = "load_test"
    freeze_control_arm(source_metrics, freeze_id)
    
    loaded = load_frozen_control_metrics(freeze_id)
    assert loaded["accuracy"] == 0.72
    assert loaded["ece"] == 0.02
    assert loaded["fresh_data_brier"] == 0.19
    assert loaded["sliced_metrics"]["fresh_window"]["n_samples"] == 42

def test_load_sweep_summary_returns_correct_values(mock_frozen_root):
    source_metrics = {"accuracy": 0.6, "brier": 0.2, "log_loss": 0.6, "ece": 0.05}
    sweep_summary = {"roi": 0.05, "total_bets": 100}
    freeze_id = "sweep_test"
    freeze_control_arm(source_metrics, freeze_id, sweep_summary=sweep_summary)
    
    loaded = load_frozen_sweep_summary(freeze_id)
    assert loaded == sweep_summary

def test_load_sweep_summary_returns_none_if_missing(mock_frozen_root):
    source_metrics = {"accuracy": 0.6, "brier": 0.2, "log_loss": 0.6, "ece": 0.05}
    freeze_id = "no_sweep_test"
    freeze_control_arm(source_metrics, freeze_id) # No sweep summary
    
    loaded = load_frozen_sweep_summary(freeze_id)
    assert loaded is None


def test_freeze_writes_standard_trading_artifacts(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.72,
        "brier": 0.18,
        "log_loss": 0.52,
        "ece": 0.02,
        "fresh_data_brier": 0.19,
        "year_by_year": [{"year": 2025, "brier": 0.18}],
    }
    freeze_control_arm(
        source_metrics,
        "trading_artifacts",
        sweep_summary={"roi": 0.05, "total_bets": 1},
        bet_log=pd.DataFrame([{"event_date": "2025-01-01", "profit": 5.0}]),
        bankroll_history=[500.0, 505.0],
    )

    artifacts = load_frozen_control_trading_artifacts("trading_artifacts")
    assert len(artifacts["bet_log"]) == 1
    assert artifacts["bankroll_history"]["combined"].tolist() == [500.0, 505.0]


def test_freeze_sanitizes_non_finite_sweep_values(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.72,
        "brier": 0.18,
        "log_loss": 0.52,
        "ece": 0.02,
    }
    freeze_control_arm(
        source_metrics,
        "sanitized_sweep",
        sweep_summary={"roi": 0.05, "c_max_decimal_odds": float("inf")},
    )

    sweep_path = mock_frozen_root / "control_arm_sanitized_sweep" / "control_sweep_summary.json"
    raw_text = sweep_path.read_text(encoding="utf-8")
    payload = json.loads(raw_text)

    assert "Infinity" not in raw_text
    assert payload["c_max_decimal_odds"] is None


def test_selection_gate_readiness_requires_fresh_window_brier(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.72,
        "brier": 0.18,
        "log_loss": 0.52,
        "ece": 0.02,
    }
    freeze_control_arm(source_metrics, "selection_incomplete")

    readiness = validate_frozen_control_arm_for_selection_gate("selection_incomplete")
    assert readiness["ready"] is False
    assert any("fresh-window Brier" in error for error in readiness["errors"])


def test_selection_fresh_window_is_partition_relative_and_persists_fail_closed(
    mock_frozen_root,
):
    from src.strategy import run_evaluation

    def manifest_frame(fold, dates, targets):
        return pd.DataFrame(
            {
                "event_date": pd.to_datetime(dates),
                "fighter_a": [f"A{fold}-{index}" for index in range(len(dates))],
                "fighter_b": [f"B{fold}-{index}" for index in range(len(dates))],
                "target": targets,
                "fold": fold,
                "train_end": pd.Timestamp("2024-12-31"),
                "test_end": pd.to_datetime(dates) + pd.Timedelta(days=7),
            }
        )

    fold_manifest = [
        (1, manifest_frame(1, ["2025-02-01", "2025-09-06"], [0, 1])),
        (2, manifest_frame(2, ["2026-02-01"], [0])),
        (3, manifest_frame(3, ["2026-08-01"], [1])),
    ]
    confirmation_snapshots = [frame.copy(deep=True) for _, frame in fold_manifest[1:]]
    scored_selection = fold_manifest[0][1].copy(deep=True)
    scored_selection["prob_a"] = [0.3, 0.8]
    selection_folds = [(1, scored_selection)]

    evidence = run_evaluation._fold_partition_evidence(
        fold_manifest,
        selection_folds,
        reserved_confirmation_folds=2,
    )
    selection_predictions = run_evaluation._post_cutoff_predictions(selection_folds)
    sliced = run_evaluation._compute_sliced_metrics(
        selection_predictions,
        selection_predictions,
        fresh_window_cutoff=pd.Timestamp(evidence["fresh_window_cutoff"]),
    )
    fresh = sliced["fresh_window"]

    assert evidence["evaluation_end_date"] == "2026-08-01"
    assert evidence["fresh_window_cutoff"] == "2025-03-06"
    assert evidence["selection_fold_ids"] == [1]
    assert evidence["confirmation_fold_ids"] == [2, 3]
    assert fresh["n_samples"] == 1
    assert fresh["brier"] == pytest.approx(0.04)
    for (_, confirmation), snapshot in zip(
        fold_manifest[1:], confirmation_snapshots
    ):
        assert "prob_a" not in confirmation
        pd.testing.assert_frame_equal(confirmation, snapshot)

    source_metrics = {
        "accuracy": 0.72,
        "brier": 0.18,
        "log_loss": 0.52,
        "ece": 0.02,
        "fresh_data_brier": fresh["brier"],
        "sliced_metrics": sliced,
    }
    freeze_control_arm(source_metrics, "selection_relative_fresh")
    persisted = json.loads(
        (
            mock_frozen_root
            / "control_arm_selection_relative_fresh"
            / "control_metrics.json"
        ).read_text(encoding="utf-8")
    )
    assert math.isfinite(persisted["fresh_data_brier"])
    assert persisted["fresh_data_brier"] == pytest.approx(0.04)
    ready = validate_frozen_control_arm_for_selection_gate(
        "selection_relative_fresh"
    )
    assert ready["valid"] is ready["ready"] is ready["passes"] is True

    missing_metrics = dict(source_metrics)
    missing_metrics["fresh_data_brier"] = None
    missing_metrics["sliced_metrics"] = {"fresh_window": {"n_samples": 0}}
    freeze_control_arm(missing_metrics, "selection_relative_missing")
    missing = validate_frozen_control_arm_for_selection_gate(
        "selection_relative_missing"
    )
    assert missing["valid"] is True
    assert missing["ready"] is missing["passes"] is False
    assert any("fresh-window Brier" in error for error in missing["errors"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_selection_gate_readiness_rejects_nonfinite_fresh_brier(
    mock_frozen_root,
    value,
):
    source_metrics = {
        "accuracy": 0.72,
        "brier": 0.18,
        "log_loss": 0.52,
        "ece": 0.02,
        "fresh_data_brier": value,
    }
    freeze_control_arm(source_metrics, "selection_nonfinite")

    readiness = validate_frozen_control_arm_for_selection_gate(
        "selection_nonfinite"
    )
    assert readiness["ready"] is False
    assert any("fresh-window Brier" in error for error in readiness["errors"])


def test_promotion_gate_readiness_requires_richer_artifacts(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.72,
        "brier": 0.18,
        "log_loss": 0.52,
        "ece": 0.02,
        "fresh_data_brier": 0.19,
    }
    freeze_control_arm(source_metrics, "promotion_incomplete")

    readiness = validate_frozen_control_arm_for_promotion_gate("promotion_incomplete")
    assert readiness["ready"] is False
    assert any("year-by-year metrics" in error for error in readiness["errors"])
    assert any("control_bet_log.csv" in error for error in readiness["errors"])


def test_promotion_gate_readiness_requires_trading_keys_in_sweep_summary(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.72,
        "brier": 0.18,
        "log_loss": 0.52,
        "ece": 0.02,
        "fresh_data_brier": 0.19,
        "year_by_year": [{"year": 2025, "brier": 0.18, "ece": 0.02}],
    }
    freeze_control_arm(
        source_metrics,
        "promotion_bad_sweep",
        sweep_summary={"total_bets": 1},
        bet_log=[{"event_date": "2025-01-01", "profit": 5.0}],
        bankroll_history=[500.0, 505.0],
    )

    readiness = validate_frozen_control_arm_for_promotion_gate("promotion_bad_sweep")
    assert readiness["ready"] is False
    assert any("usable trading baseline" in error for error in readiness["errors"])


def test_promotion_gate_readiness_passes_with_rich_freeze(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.72,
        "brier": 0.18,
        "log_loss": 0.52,
        "ece": 0.02,
        "fresh_data_brier": 0.19,
        "year_by_year": [{"year": 2025, "brier": 0.18, "ece": 0.02}],
    }
    freeze_control_arm(
        source_metrics,
        "promotion_ready",
        sweep_summary={
            "roi": 0.05,
            "total_profit": 5.0,
            "max_drawdown": 0.10,
            "avg_clv": 0.01,
            "total_bets": 1,
        },
        bet_log=[{"event_date": "2025-01-01", "profit": 5.0}],
        bankroll_history=[500.0, 505.0],
    )

    readiness = validate_frozen_control_arm_for_promotion_gate("promotion_ready")
    assert readiness["ready"] is True


@pytest.mark.parametrize(
    "field",
    ["roi", "total_profit", "max_drawdown", "avg_clv"],
)
@pytest.mark.parametrize(
    "value",
    [None, float("nan"), float("inf"), float("-inf")],
)
def test_promotion_readiness_rejects_missing_or_nonfinite_trading_metrics(
    mock_frozen_root,
    field,
    value,
):
    source_metrics = {
        "accuracy": 0.72,
        "brier": 0.18,
        "log_loss": 0.52,
        "ece": 0.02,
        "fresh_data_brier": 0.19,
        "year_by_year": [{"year": 2025, "brier": 0.18, "ece": 0.02}],
    }
    sweep = {
        "roi": 0.05,
        "total_profit": 5.0,
        "max_drawdown": 0.10,
        "avg_clv": 0.01,
        "total_bets": 20,
    }
    sweep[field] = value
    freeze_control_arm(
        source_metrics,
        f"promotion_nonfinite_{field}",
        sweep_summary=sweep,
        bet_log=[{"event_date": "2025-01-01", "profit": 5.0}],
        bankroll_history=[500.0, 505.0],
    )

    readiness = validate_frozen_control_arm_for_promotion_gate(
        f"promotion_nonfinite_{field}"
    )

    assert readiness["ready"] is False
    assert any("trading baseline" in error or "trading metrics" in error for error in readiness["errors"])


def _integrity_v2_fixture(tmp_path, monkeypatch):
    from src.data import historical_backfill
    from src.strategy import run_evaluation

    source_dir = (tmp_path / "integrity_sources").resolve()
    source_dir.mkdir()
    odds_dir = (source_dir / "odds").resolve()
    odds_dir.mkdir()
    monkeypatch.setattr(control_arm, "INTEGRITY_V2_APPROVED_PROCESSED_ROOT", source_dir)
    monkeypatch.setattr(control_arm, "INTEGRITY_V2_APPROVED_ODDS_ROOT", source_dir)
    monkeypatch.setattr(historical_backfill, "BACKFILL_DIR", odds_dir)

    named_spec = resolve_named_training_spec(control_arm.INTEGRITY_V2_SPEC_NAME)
    feature_contract_columns = list(named_spec.feature_cols)
    frame = pd.DataFrame(
        {
            "event_date": pd.to_datetime(
                ["2025-01-01", "2025-02-01", "2025-03-01"]
            ),
            "fighter_a": ["A", "C", "E"],
            "fighter_b": ["B", "D", "F"],
            "target": [1, 0, 1],
            "fold": [1, 2, 3],
            "train_end": pd.to_datetime(
                ["2024-12-31", "2025-01-31", "2025-02-28"]
            ),
            "test_end": pd.to_datetime(
                ["2025-01-31", "2025-02-28", "2025-03-31"]
            ),
        }
    )
    feature_values = {
        column: [offset, offset + 1, offset + 2]
        for offset, column in enumerate(feature_contract_columns)
        if column not in frame
    }
    frame = pd.concat([frame, pd.DataFrame(feature_values)], axis=1)
    fold_manifest = [
        (fold_id, frame[frame["fold"] == fold_id].copy())
        for fold_id in (1, 2, 3)
    ]
    partition_evidence = run_evaluation._fold_partition_evidence(
        fold_manifest,
        fold_manifest[:1],
        reserved_confirmation_folds=2,
        feature_contract_columns=feature_contract_columns,
    )
    index = frame.copy()
    index["evaluation_partition"] = ["selection", "confirmation", "confirmation"]
    index_path = source_dir / control_arm.INTEGRITY_V2_EVALUATION_INDEX_FILENAME
    index.to_csv(index_path, index=False, lineterminator="\n")
    index_sha = control_arm._sha256(index_path)

    dataset_fights_path = source_dir / "fights_cleaned.csv"
    frame[["event_date", "fighter_a", "fighter_b", "target"]].to_csv(
        dataset_fights_path,
        index=False,
        lineterminator="\n",
    )
    features_artifact_path = source_dir / "features.csv"
    frame.to_csv(features_artifact_path, index=False, lineterminator="\n")
    features_frame = pd.read_csv(features_artifact_path)
    monkeypatch.setattr(
        control_arm,
        "INTEGRITY_V2_CORRECTED_FIGHTS_PATH",
        dataset_fights_path,
    )
    monkeypatch.setattr(
        control_arm,
        "INTEGRITY_V2_CORRECTED_FEATURES_PATH",
        features_artifact_path,
    )

    odds_source_path = odds_dir / "historical_odds.csv"
    odds_source_path.write_text(
        "event_date,home_odds,away_odds\n",
        encoding="utf-8",
    )
    policy_evaluation = control_arm._INTEGRITY_V2_POLICY["evaluation"]
    evaluation_protocol = run_evaluation._evaluation_protocol_payload(
        bet_start_date=policy_evaluation["bet_start_date"],
        execution_mode=policy_evaluation["execution_mode"],
        entry_offset_days=policy_evaluation["entry_offset_days"],
        entry_offset_for_features=policy_evaluation["entry_offset_for_features"],
        require_entry_odds=policy_evaluation["require_entry_odds"],
        allow_closing_odds=False,
        reserved_confirmation_folds=2,
        bootstrap=5000,
        retrain_months=policy_evaluation["retrain_months"],
        policy_evaluation=policy_evaluation,
    )
    odds_inventory = {
        "schema_version": 1,
        "evaluation_protocol": evaluation_protocol,
        "entries": [
            {
                "source_file": odds_source_path.name,
                "resolved_path": str(odds_source_path),
                "sha256": control_arm._sha256(odds_source_path),
            }
        ],
    }
    odds_inventory_path = source_dir / control_arm.INTEGRITY_V2_ODDS_SOURCE_INVENTORY_FILENAME
    run_evaluation._write_json(odds_inventory_path, odds_inventory)

    runtime_code = run_evaluation._collect_runtime_code_metadata()
    source_inventory = runtime_code["source_inventory"]
    source_fingerprint = runtime_code["source_fingerprint"]
    source_inventory_path = source_dir / control_arm.INTEGRITY_V2_SOURCE_INVENTORY_FILENAME
    run_evaluation._write_json(source_inventory_path, source_inventory)
    environment = runtime_code["environment"]
    environment_path = source_dir / control_arm.INTEGRITY_V2_ENVIRONMENT_FILENAME
    run_evaluation._write_json(environment_path, environment)

    policy_path = control_arm.INTEGRITY_V2_POLICY_PATH.resolve()
    static_bindings = dict(control_arm._INTEGRITY_V2_BINDINGS)
    static_bindings.update(
        {
            "source_dataset_fights_sha256": control_arm._sha256(dataset_fights_path),
            "source_features_sha256": control_arm._sha256(features_artifact_path),
            "full_evaluation_sample_sha256": partition_evidence[
                "full_evaluation_sample_sha256"
            ],
            "full_evaluation_n_fights": 3,
            "full_evaluation_n_folds": 3,
        }
    )
    monkeypatch.setattr(control_arm, "_INTEGRITY_V2_BINDINGS", static_bindings)
    monkeypatch.setattr(
        control_arm,
        "_load_policy_bound_integrity_v2_contract",
        lambda require_final_validation=True: (
            control_arm._INTEGRITY_V2_POLICY,
            named_spec,
            evaluation_protocol,
            static_bindings,
        ),
    )

    policy_provenance = {
        "model_spec_name": named_spec.name,
        "model_spec_payload_sha256": static_bindings[
            "model_spec_payload_sha256"
        ],
        "policy_path": str(policy_path),
        "policy_sha256": static_bindings["policy_sha256"],
        "scheduled_protocol_sha256": static_bindings[
            "scheduled_protocol_sha256"
        ],
        "source_dataset_fights_path": str(dataset_fights_path),
        "source_dataset_fights_sha256": control_arm._sha256(dataset_fights_path),
        "source_features_path": str(features_artifact_path),
        "source_features_sha256": control_arm._sha256(features_artifact_path),
        "corrected_fights_path": str(dataset_fights_path),
        "corrected_fights_sha256": control_arm._sha256(dataset_fights_path),
        "corrected_features_path": str(features_artifact_path),
        "corrected_features_sha256": control_arm._sha256(features_artifact_path),
    }
    input_provenance = run_evaluation._build_input_provenance_payload(
        dataset_fights_path=dataset_fights_path,
        features_artifact_path=features_artifact_path,
        features_frame=features_frame,
        feature_contract_columns=feature_contract_columns,
        partition_evidence=partition_evidence,
        policy_provenance=policy_provenance,
        evaluation_protocol=evaluation_protocol,
        odds_source_inventory=odds_inventory,
        source_fingerprint=source_fingerprint,
        source_inventory_sha256=source_fingerprint,
        source_inventory_path=source_inventory_path,
        source_inventory_artifact_sha256=control_arm._sha256(source_inventory_path),
        environment_path=environment_path,
        environment_artifact_sha256=control_arm._sha256(environment_path),
        environment_payload_sha256=runtime_code["environment_payload_sha256"],
        model_spec_name=named_spec.name,
        model_spec_payload_sha256=static_bindings["model_spec_payload_sha256"],
    )
    input_provenance.update(policy_provenance)
    input_provenance.update(partition_evidence)
    input_provenance["input_provenance_payload_sha256"] = (
        control_arm._canonical_json_sha256(
            {
                key: value
                for key, value in input_provenance.items()
                if key != "input_provenance_payload_sha256"
            }
        )
    )
    input_provenance_path = source_dir / control_arm.INTEGRITY_V2_INPUT_PROVENANCE_FILENAME
    run_evaluation._write_json(input_provenance_path, input_provenance)

    all_bindings = {**static_bindings, **partition_evidence}
    common_evidence = {
        **all_bindings,
        "prediction_rows_sha256": "8" * 64,
        "prediction_values_sha256": "9" * 64,
        "role": "fixed_control_construction",
        "stage2_bypassed": True,
        "research_grids_evaluated": False,
        "source_fingerprint": source_fingerprint,
        "evaluation_protocol": evaluation_protocol,
        "input_provenance_path": str(input_provenance_path),
        "input_provenance_sha256": control_arm._sha256(input_provenance_path),
        "input_provenance_payload_sha256": input_provenance[
            "input_provenance_payload_sha256"
        ],
        "odds_source_inventory_path": str(odds_inventory_path),
        "odds_source_inventory_sha256": control_arm._sha256(odds_inventory_path),
        "source_inventory_path": str(source_inventory_path),
        "source_inventory_artifact_sha256": control_arm._sha256(
            source_inventory_path
        ),
        "source_inventory_sha256": source_fingerprint,
        "environment_path": str(environment_path),
        "environment_artifact_sha256": control_arm._sha256(environment_path),
        "environment_payload_sha256": runtime_code[
            "environment_payload_sha256"
        ],
        "source_dataset_fights_path": str(dataset_fights_path),
        "source_dataset_fights_sha256": control_arm._sha256(dataset_fights_path),
        "source_features_path": str(features_artifact_path),
        "source_features_sha256": control_arm._sha256(features_artifact_path),
        "dataset_fights_path": str(dataset_fights_path),
        "dataset_fights_sha256": control_arm._sha256(dataset_fights_path),
        "features_artifact_path": str(features_artifact_path),
        "features_artifact_sha256": control_arm._sha256(features_artifact_path),
        "features_value_sha256": input_provenance["features_value_sha256"],
        "feature_contract_columns": feature_contract_columns,
        "evaluation_index_path": str(index_path),
        "evaluation_index_sha256": index_sha,
    }
    sweep_summary = {
        "promotion_eligible": True,
        "roi": 0.01,
        "total_profit": 2.0,
        "total_bets": 20,
        "max_drawdown": 0.10,
        "avg_clv": 0.01,
        "fixed_control_evidence": common_evidence,
    }
    production_sources = {
        control_arm.INTEGRITY_V2_PRODUCTION_RESULT_FILENAME: sweep_summary,
        control_arm.INTEGRITY_V2_PRODUCTION_ROW_FILENAME: {
            "config": "baseline_production_controls",
            "fixed_control_evidence": common_evidence,
        },
    }
    for artifact_name, payload in production_sources.items():
        (source_dir / artifact_name).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    dates = pd.date_range("2025-01-04", periods=20, freq="7D")
    pd.DataFrame(
        {"event_date": dates, "profit": [0.1] * 20, "bet_size": [10.0] * 20}
    ).to_csv(
        source_dir / control_arm.INTEGRITY_V2_PRODUCTION_BET_LOG_FILENAME,
        index=False,
    )
    pd.DataFrame({"combined": [500.0, 502.0]}).to_csv(
        source_dir / control_arm.INTEGRITY_V2_PRODUCTION_BANKROLL_FILENAME,
        index=False,
    )
    receipt_artifact_names = control_arm.INTEGRITY_V2_RECEIPT_ARTIFACT_FILENAMES
    receipt = {
        "schema_version": 1,
        **common_evidence,
        "candidate_id": "baseline_pulled_all_plus_legacy_market_production",
        "dataset_variant": "pulled_all_plus_legacy_market",
        "feature_family": "production",
        "calibration_method": "sigmoid",
        "retrain_months": 6,
        "execution_mode": "realistic",
        "entry_offset_days": 1.0,
        "entry_offset_for_features": True,
        "require_entry_odds": True,
        "artifacts": {
            artifact_name: {
                "path": str(source_dir / artifact_name),
                "sha256": control_arm._sha256(source_dir / artifact_name),
            }
            for artifact_name in receipt_artifact_names
        },
    }
    receipt_path = source_dir / control_arm.INTEGRITY_V2_RECEIPT_FILENAME
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    source_metrics = {
        "accuracy": 0.62,
        "brier": 0.22,
        "log_loss": 0.64,
        "ece": 0.06,
        "fresh_data_brier": 0.22,
        "year_by_year": {"2025": {"brier": 0.22, "ece": 0.06}},
        "model_variant": "baseline",
        "dataset_variant": "pulled_all_plus_legacy_market",
        "feature_family": "production",
        "calibration_method": "sigmoid",
        "retrain_months": 6,
        **all_bindings,
        "prediction_rows_sha256": common_evidence["prediction_rows_sha256"],
        "prediction_values_sha256": common_evidence["prediction_values_sha256"],
        **{
            field: common_evidence[field]
            for field in control_arm._INTEGRITY_V2_DYNAMIC_PROVENANCE_FIELDS
        },
        "source_fingerprint": common_evidence["source_fingerprint"],
        "evaluation_protocol": common_evidence["evaluation_protocol"],
        "evaluation_index_path": str(index_path),
        "evaluation_index_sha256": index_sha,
    }
    artifacts = {
        control_arm.INTEGRITY_V2_RECEIPT_FILENAME: receipt_path,
        **{
            artifact_name: source_dir / artifact_name
            for artifact_name in receipt_artifact_names
        },
    }
    return source_metrics, sweep_summary, artifacts


def _rebind_integrity_provenance_after_odds_edit(metrics, sweep, artifacts):
    """Refresh exact receipt hashes while retaining a deliberately bad inventory."""
    from src.strategy import run_evaluation

    odds_path = artifacts[control_arm.INTEGRITY_V2_ODDS_SOURCE_INVENTORY_FILENAME]
    odds_payload = json.loads(odds_path.read_text(encoding="utf-8"))
    odds_sha = control_arm._sha256(odds_path)
    provenance_path = artifacts[control_arm.INTEGRITY_V2_INPUT_PROVENANCE_FILENAME]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["odds_source_inventory_sha256"] = odds_sha
    provenance["odds_source_inventory_payload_sha256"] = (
        control_arm._canonical_json_sha256(odds_payload)
    )
    provenance["input_provenance_payload_sha256"] = (
        control_arm._canonical_json_sha256(
            {
                key: value
                for key, value in provenance.items()
                if key != "input_provenance_payload_sha256"
            }
        )
    )
    run_evaluation._write_json(provenance_path, provenance)
    provenance_sha = control_arm._sha256(provenance_path)

    evidence_payloads = (metrics, sweep["fixed_control_evidence"])
    for payload in evidence_payloads:
        payload["odds_source_inventory_sha256"] = odds_sha
        payload["input_provenance_sha256"] = provenance_sha
        payload["input_provenance_payload_sha256"] = provenance[
            "input_provenance_payload_sha256"
        ]

    result_path = artifacts[control_arm.INTEGRITY_V2_PRODUCTION_RESULT_FILENAME]
    run_evaluation._write_json(result_path, sweep)
    row_path = artifacts[control_arm.INTEGRITY_V2_PRODUCTION_ROW_FILENAME]
    row = json.loads(row_path.read_text(encoding="utf-8"))
    row["fixed_control_evidence"] = sweep["fixed_control_evidence"]
    run_evaluation._write_json(row_path, row)

    receipt_path = artifacts[control_arm.INTEGRITY_V2_RECEIPT_FILENAME]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(sweep["fixed_control_evidence"])
    for artifact_name in (
        control_arm.INTEGRITY_V2_ODDS_SOURCE_INVENTORY_FILENAME,
        control_arm.INTEGRITY_V2_INPUT_PROVENANCE_FILENAME,
        control_arm.INTEGRITY_V2_PRODUCTION_RESULT_FILENAME,
        control_arm.INTEGRITY_V2_PRODUCTION_ROW_FILENAME,
    ):
        receipt["artifacts"][artifact_name]["sha256"] = control_arm._sha256(
            artifacts[artifact_name]
        )
    run_evaluation._write_json(receipt_path, receipt)


def test_integrity_v2_freeze_requires_receipt_and_evaluation_index(
    mock_frozen_root,
):
    metrics = {"accuracy": 0.6, "brier": 0.2, "log_loss": 0.6, "ece": 0.05}

    with pytest.raises(ValueError, match="fixed_control_bootstrap_receipt"):
        freeze_control_arm(metrics, "integrity_v2_missing_evidence")


def test_integrity_v2_freeze_preserves_exact_partition_evidence_and_manifest(
    mock_frozen_root,
    tmp_path,
    monkeypatch,
):
    metrics, sweep, artifacts = _integrity_v2_fixture(tmp_path, monkeypatch)
    bet_log = pd.read_csv(
        artifacts[control_arm.INTEGRITY_V2_PRODUCTION_BET_LOG_FILENAME]
    )
    bankroll_history = pd.read_csv(
        artifacts[control_arm.INTEGRITY_V2_PRODUCTION_BANKROLL_FILENAME]
    )

    freeze_dir = freeze_control_arm(
        metrics,
        "integrity_v2_unit",
        sweep_summary=sweep,
        artifact_sources=artifacts,
        bet_log=bet_log,
        bankroll_history=bankroll_history,
    )

    assert validate_frozen_control_arm("integrity_v2_unit")["valid"] is True
    assert validate_frozen_control_arm_for_selection_gate(
        "integrity_v2_unit"
    )["ready"] is True
    assert validate_frozen_control_arm_for_promotion_gate(
        "integrity_v2_unit"
    )["ready"] is True
    assert (freeze_dir / control_arm.INTEGRITY_V2_RECEIPT_FILENAME).is_file()
    assert (
        freeze_dir / control_arm.INTEGRITY_V2_EVALUATION_INDEX_FILENAME
    ).is_file()
    for artifact_name in control_arm.INTEGRITY_V2_RECEIPT_ARTIFACT_FILENAMES:
        assert (freeze_dir / artifact_name).is_file()
    assert (freeze_dir / "control_sweep_summary.json").read_bytes() == (
        freeze_dir / control_arm.INTEGRITY_V2_PRODUCTION_RESULT_FILENAME
    ).read_bytes()
    assert (freeze_dir / "control_bet_log.csv").read_bytes() == (
        freeze_dir / control_arm.INTEGRITY_V2_PRODUCTION_BET_LOG_FILENAME
    ).read_bytes()
    assert (freeze_dir / "control_bankroll_history.csv").read_bytes() == (
        freeze_dir / control_arm.INTEGRITY_V2_PRODUCTION_BANKROLL_FILENAME
    ).read_bytes()
    manifest = (freeze_dir / "MANIFEST.md").read_text(encoding="utf-8")
    assert "spec=full_live_contract_v6_integrity_205" in manifest
    assert "calibration=sigmoid" in manifest
    assert "retrain_months=6" in manifest
    assert "isotonic" not in manifest
    assert "4-month" not in manifest


def test_upstream_fixed_control_writer_receipt_passes_strict_freezer(
    mock_frozen_root,
    tmp_path,
    monkeypatch,
):
    """Exercise the real writer schema through freeze and readiness validation."""

    from src.strategy import run_evaluation

    isolated_repo = (tmp_path / "repo").resolve()
    (isolated_repo / "src").mkdir(parents=True)
    (isolated_repo / "src" / "fixture_source.py").write_text(
        "# isolated integrity-v2 regression source\n",
        encoding="utf-8",
    )
    for relative_path in run_evaluation.EVALUATION_SCRIPT_SOURCE_FILES:
        script_path = isolated_repo / relative_path
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(
            "# isolated integrity-v2 regression script\n",
            encoding="utf-8",
        )
    (isolated_repo / "requirements.txt").write_text(
        "isolated-integrity-v2-regression\n",
        encoding="utf-8",
    )
    isolated_policy_path = isolated_repo / "config" / "scheduled_refit_policy_v2.json"
    isolated_policy_path.parent.mkdir(parents=True)
    isolated_policy_path.write_bytes(control_arm.INTEGRITY_V2_POLICY_PATH.read_bytes())
    monkeypatch.setattr(run_evaluation, "_repo_root", lambda: isolated_repo)
    monkeypatch.setattr(control_arm, "REPO_ROOT", isolated_repo)
    monkeypatch.setattr(
        control_arm,
        "INTEGRITY_V2_POLICY_PATH",
        isolated_policy_path,
    )

    metrics, _hand_built_sweep, fixture_artifacts = _integrity_v2_fixture(
        tmp_path,
        monkeypatch,
    )
    source_dir = fixture_artifacts[control_arm.INTEGRITY_V2_RECEIPT_FILENAME].parent
    named_spec = resolve_named_training_spec(control_arm.INTEGRITY_V2_SPEC_NAME)
    input_provenance = json.loads(
        fixture_artifacts[
            control_arm.INTEGRITY_V2_INPUT_PROVENANCE_FILENAME
        ].read_text(encoding="utf-8")
    )
    odds_inventory = json.loads(
        fixture_artifacts[
            control_arm.INTEGRITY_V2_ODDS_SOURCE_INVENTORY_FILENAME
        ].read_text(encoding="utf-8")
    )
    features = pd.read_csv(input_provenance["features_artifact_path"])
    for column in ("event_date", "train_end", "test_end"):
        features[column] = pd.to_datetime(features[column])
    feature_contract_columns = list(named_spec.feature_cols)
    precision_feature = feature_contract_columns[0]
    nullable_feature = feature_contract_columns[1]
    precision_values = {
        1: 0.12345678901234566,
        2: 1.0000000000000002,
        3: 123456789.12345679,
    }
    manifest: list[tuple[int, pd.DataFrame]] = []
    for fold_id in sorted(features["fold"].unique()):
        fold = features[features["fold"] == fold_id].copy()
        fold_id = int(fold_id)
        fold[precision_feature] = precision_values[fold_id]
        fold[nullable_feature] = pd.array(
            [pd.NA if fold_id == 2 else precision_values[fold_id]],
            dtype="Float64",
        )
        fold["entry_writer_precision_probe"] = pd.array(
            [pd.NA if fold_id == 3 else precision_values[fold_id]],
            dtype="Float64",
        )
        fold["prob_a"] = 0.6
        fold["prob_b"] = 0.4
        fold["no_odds_prob_a"] = 0.55
        fold["no_odds_prob_b"] = 0.45
        manifest.append((fold_id, fold))
    pre_cutoff = manifest[0][1].copy()
    pre_cutoff["event_date"] = pd.Timestamp("2021-12-18")
    pre_cutoff["fighter_a"] = "Pre-cutoff A"
    pre_cutoff["fighter_b"] = "Pre-cutoff B"
    pre_cutoff["train_end"] = pd.Timestamp("2021-11-30")
    pre_cutoff["test_end"] = pd.Timestamp("2021-12-31")
    manifest[0] = (
        manifest[0][0],
        pd.concat([pre_cutoff, manifest[0][1]], ignore_index=True),
    )
    selection_folds = manifest[:1]
    partition = run_evaluation._fold_partition_evidence(
        manifest,
        selection_folds,
        reserved_confirmation_folds=2,
        feature_contract_columns=feature_contract_columns,
    )
    input_provenance.update(partition)
    input_provenance["input_provenance_payload_sha256"] = (
        control_arm._canonical_json_sha256(
            {
                key: value
                for key, value in input_provenance.items()
                if key != "input_provenance_payload_sha256"
            }
        )
    )
    run_evaluation._write_json(
        fixture_artifacts[control_arm.INTEGRITY_V2_INPUT_PROVENANCE_FILENAME],
        input_provenance,
    )
    selection_predictions = run_evaluation._post_cutoff_predictions(selection_folds)
    payload = {
        "candidate_id": "baseline_pulled_all_plus_legacy_market_production",
        "model_variant": "baseline",
        "dataset_variant": "pulled_all_plus_legacy_market",
        "feature_family": "production",
        "calibration_method": named_spec.calibration_method,
        "retrain_months": 6,
        "model_spec_name": named_spec.name,
        "model_spec_payload_sha256": control_arm._INTEGRITY_V2_BINDINGS[
            "model_spec_payload_sha256"
        ],
        "input_provenance": input_provenance,
        "input_provenance_payload_sha256": input_provenance[
            "input_provenance_payload_sha256"
        ],
        "odds_source_inventory": odds_inventory,
        "odds_source_inventory_sha256": input_provenance[
            "odds_source_inventory_sha256"
        ],
        **partition,
        "prediction_rows_sha256": run_evaluation._prediction_rows_sha256(
            selection_predictions
        ),
        "prediction_values_sha256": run_evaluation._prediction_values_sha256(
            selection_predictions
        ),
    }
    metadata = {
        "source_fingerprint": input_provenance["source_fingerprint"],
        "execution_mode": "realistic",
        "entry_offset_days": 1.0,
        "entry_offset_for_features": True,
        "require_entry_odds": True,
        "allow_closing_odds": False,
        "selected_model_spec_name": named_spec.name,
        "selected_model_spec_payload_sha256": payload[
            "model_spec_payload_sha256"
        ],
        "expected_evaluation_sample_sha256": partition[
            "full_evaluation_sample_sha256"
        ],
        "expected_evaluation_n_fights": partition["full_evaluation_n_fights"],
        "expected_evaluation_n_folds": partition["full_evaluation_n_folds"],
        "reserved_confirmation_folds": 2,
        "bootstrap": 5000,
    }
    monkeypatch.setattr(
        run_evaluation,
        "_validate_input_provenance_files",
        lambda *_args, **_kwargs: features,
    )
    monkeypatch.setattr(
        run_evaluation,
        "_generate_walk_forward_predictions",
        lambda **_kwargs: (selection_folds, manifest),
    )
    dates = pd.date_range("2025-01-04", periods=20, freq="7D")

    def production_shaped_result(_folds, config, **_kwargs):
        trader_stats = {
            "total_bets": 10,
            "wins": 6,
            "win_rate": 0.6,
            "total_wagered": 100.0,
            "roi": 0.01,
            "avg_edge": 0.02,
            "avg_bet_size": 10.0,
            "avg_clv": 0.01,
        }
        return {
            "config": config,
            "model_variant": "baseline",
            "mode": "duo_sweep",
            "trader_s": {
                "name": "Single Trader",
                "final_pnl": 1.0,
                "stats": trader_stats,
            },
            "trader_c": {
                "name": "Conviction Trader",
                "final_pnl": 1.0,
                "stats": dict(trader_stats),
            },
            "trader_m": None,
            "combined": {
                "initial_bankroll": 500.0,
                "final_bankroll": 502.0,
                "total_profit": 2.0,
                "total_wagered": 200.0,
                "total_bets": 20,
                "wins": 12,
                "win_rate": 0.6,
                "roi": 0.01,
                "bankroll_growth": 1.004,
                "max_drawdown": 0.10,
                "max_drawdown_pct": 0.10,
                "max_drawdown_duration": 1,
                "avg_clv": 0.01,
            },
            "bet_log": pd.DataFrame(
                {"event_date": dates, "profit": [0.1] * 20, "bet_size": [10.0] * 20}
            ),
            "bankroll_history": pd.DataFrame({"combined": [500.0, 502.0]}),
        }

    monkeypatch.setattr(
        run_evaluation,
        "_evaluate_config",
        production_shaped_result,
    )

    sweep = run_evaluation._run_fixed_control_stage3_candidate(
        run_dir=source_dir,
        payload=payload,
        metadata=metadata,
    )
    receipt_path = source_dir / control_arm.INTEGRITY_V2_RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    artifacts = {
        control_arm.INTEGRITY_V2_RECEIPT_FILENAME: receipt_path,
        **{
            name: Path(binding["path"])
            for name, binding in receipt["artifacts"].items()
        },
    }
    production_result_text = artifacts[
        control_arm.INTEGRITY_V2_PRODUCTION_RESULT_FILENAME
    ].read_text(encoding="utf-8")
    production_row_text = artifacts[
        control_arm.INTEGRITY_V2_PRODUCTION_ROW_FILENAME
    ].read_text(encoding="utf-8")
    assert "Infinity" not in production_result_text
    assert "Infinity" not in production_row_text
    sweep = json.loads(production_result_text)
    production_row = json.loads(production_row_text)
    assert set(sweep) == {
        "combined",
        "config",
        "fixed_control_evidence",
        "mode",
        "model_variant",
        "promotion_eligible",
        "trader_c",
        "trader_m",
        "trader_s",
    }
    assert "evaluation_sample_sha256" not in sweep
    assert sweep["fixed_control_evidence"]["evaluation_sample_sha256"] == partition[
        "evaluation_sample_sha256"
    ]
    assert sweep["config"]["c_max_decimal_odds"] is None
    assert production_row["c_max_decimal_odds"] is None
    assert (
        run_evaluation._normalize_json(
            pd.Series([float("inf")], dtype="float32").iloc[0]
        )
        is None
    )
    evidence = sweep["fixed_control_evidence"]
    index_path = artifacts[control_arm.INTEGRITY_V2_EVALUATION_INDEX_FILENAME]
    published_index = pd.read_csv(index_path, float_precision="round_trip")
    assert len(published_index) == 3
    assert pd.to_datetime(
        published_index["event_date"], utc=True
    ).min() >= pd.Timestamp(run_evaluation.TRAIN_CUTOFF_DATE, tz="UTC")
    selection_index = published_index[
        published_index["evaluation_partition"] == "selection"
    ]
    confirmation_index = published_index[
        published_index["evaluation_partition"] == "confirmation"
    ]
    assert len(selection_index) == evidence["n_predictions"] == 1
    assert len(confirmation_index) == evidence["confirmation_evaluation_n_fights"] == 2
    assert len(published_index) == evidence["full_evaluation_n_fights"]
    for date_column in ("event_date", "train_end", "test_end"):
        published_index[date_column] = pd.to_datetime(
            published_index[date_column],
            errors="raise",
            utc=True,
        )
    selection_index = published_index[
        published_index["evaluation_partition"] == "selection"
    ]
    confirmation_index = published_index[
        published_index["evaluation_partition"] == "confirmation"
    ]
    assert run_evaluation._manifest_input_value_sha256(
        selection_index,
        feature_contract_columns=feature_contract_columns,
    ) == evidence["evaluation_input_value_sha256"]
    assert run_evaluation._manifest_input_value_sha256(
        published_index,
        feature_contract_columns=feature_contract_columns,
    ) == evidence["full_evaluation_input_value_sha256"]
    assert run_evaluation._manifest_input_value_sha256(
        confirmation_index,
        feature_contract_columns=feature_contract_columns,
    ) == evidence["confirmation_evaluation_input_value_sha256"]
    assert control_arm._validate_integrity_v2_index(
        index_path,
        expected_bindings=evidence,
    ) == []
    metrics.update(evidence)
    metrics["model_variant"] = "baseline"
    bet_log = pd.read_csv(
        artifacts[control_arm.INTEGRITY_V2_PRODUCTION_BET_LOG_FILENAME]
    )
    bankroll = pd.read_csv(
        artifacts[control_arm.INTEGRITY_V2_PRODUCTION_BANKROLL_FILENAME]
    )
    cells_dir = source_dir / "cells"
    cells_dir.mkdir()
    run_evaluation._write_json(
        cells_dir
        / "baseline__pulled_all_plus_legacy_market__production__sigmoid_metrics.json",
        {
            **metrics,
            "metrics": {
                "accuracy": metrics["accuracy"],
                "brier": metrics["brier"],
                "log_loss": metrics["log_loss"],
                "ece": metrics["ece"],
            },
            "sliced_metrics": {
                "fresh_window": {"brier": metrics["fresh_data_brier"]},
                "by_year": metrics["year_by_year"],
            },
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "control_arm.py",
            "--run-dir",
            str(source_dir),
            "--candidate-id",
            "baseline_pulled_all_plus_legacy_market_production",
            "--freeze-id",
            "integrity_v2_real_writer",
        ],
    )
    control_arm.main()

    freeze_dir = mock_frozen_root / "control_arm_integrity_v2_real_writer"
    assert {path.name for path in freeze_dir.iterdir()} == {
        *control_arm.INTEGRITY_V2_FROZEN_ARTIFACT_FILENAMES,
        "checksums.json",
        "MANIFEST.md",
    }
    assert all(path.is_file() for path in freeze_dir.iterdir())
    assert (freeze_dir / "control_sweep_summary.json").read_bytes() == artifacts[
        control_arm.INTEGRITY_V2_PRODUCTION_RESULT_FILENAME
    ].read_bytes()

    validation = validate_frozen_control_arm("integrity_v2_real_writer")
    assert validation["valid"] is validation["passes"] is True
    assert validation["errors"] == []
    selection = validate_frozen_control_arm_for_selection_gate(
        "integrity_v2_real_writer"
    )
    assert selection["valid"] is selection["ready"] is selection["passes"] is True
    assert selection["errors"] == []
    promotion = validate_frozen_control_arm_for_promotion_gate(
        "integrity_v2_real_writer"
    )
    assert promotion["valid"] is promotion["ready"] is promotion["passes"] is True
    assert promotion["errors"] == []

    mismatched_sweep = {
        **sweep,
        "evaluation_sample_sha256": evidence["evaluation_sample_sha256"],
    }
    with pytest.raises(
        ValueError,
        match="production_result.json does not match the supplied sweep summary",
    ):
        freeze_control_arm(
            metrics,
            "integrity_v2_real_writer_mismatch",
            sweep_summary=mismatched_sweep,
            artifact_sources=artifacts,
            bet_log=bet_log,
            bankroll_history=bankroll,
        )
    assert not (
        mock_frozen_root / "control_arm_integrity_v2_real_writer_mismatch"
    ).exists()


@pytest.mark.parametrize("source_kind", ["dataset", "features", "odds"])
def test_integrity_v2_readiness_rehashes_external_sources_after_freeze(
    mock_frozen_root,
    tmp_path,
    monkeypatch,
    source_kind,
):
    metrics, sweep, artifacts = _integrity_v2_fixture(tmp_path, monkeypatch)
    freeze_control_arm(
        metrics,
        "integrity_v2_external_rehash",
        sweep_summary=sweep,
        artifact_sources=artifacts,
    )
    provenance = json.loads(
        artifacts[control_arm.INTEGRITY_V2_INPUT_PROVENANCE_FILENAME].read_text(
            encoding="utf-8"
        )
    )
    if source_kind == "dataset":
        source_path = Path(provenance["dataset_fights_path"])
    elif source_kind == "features":
        source_path = Path(provenance["features_artifact_path"])
    else:
        odds_inventory = json.loads(
            artifacts[
                control_arm.INTEGRITY_V2_ODDS_SOURCE_INVENTORY_FILENAME
            ].read_text(encoding="utf-8")
        )
        source_path = Path(odds_inventory["entries"][0]["resolved_path"])
    source_path.write_text(
        source_path.read_text(encoding="utf-8") + "tampered\n",
        encoding="utf-8",
    )

    validation = validate_frozen_control_arm("integrity_v2_external_rehash")

    assert validation["valid"] is False
    assert any(
        "source hash mismatch" in error
        or "source mismatch" in error
        or "value hash mismatch" in error
        for error in validation["errors"]
    )


@pytest.mark.parametrize("mutation", ["extra_artifact", "omitted_checksum"])
def test_integrity_v2_readiness_enforces_exact_artifact_checksum_allowlist(
    mock_frozen_root,
    tmp_path,
    monkeypatch,
    mutation,
):
    metrics, sweep, artifacts = _integrity_v2_fixture(tmp_path, monkeypatch)
    freeze_dir = freeze_control_arm(
        metrics,
        f"integrity_v2_allowlist_{mutation}",
        sweep_summary=sweep,
        artifact_sources=artifacts,
    )
    if mutation == "extra_artifact":
        (freeze_dir / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    else:
        checksums_path = freeze_dir / "checksums.json"
        checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
        checksums.pop("control_metrics.json")
        checksums_path.write_text(json.dumps(checksums), encoding="utf-8")

    validation = validate_frozen_control_arm(
        f"integrity_v2_allowlist_{mutation}"
    )

    assert validation["valid"] is False
    assert any("allowlist mismatch" in error for error in validation["errors"])


def test_integrity_v2_freeze_enforces_exact_receipt_artifact_allowlist(
    mock_frozen_root,
    tmp_path,
    monkeypatch,
):
    metrics, sweep, artifacts = _integrity_v2_fixture(tmp_path, monkeypatch)
    receipt_path = artifacts[control_arm.INTEGRITY_V2_RECEIPT_FILENAME]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"]["unlisted_evidence.json"] = {
        "path": str(tmp_path / "unlisted_evidence.json"),
        "sha256": "0" * 64,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="receipt artifact allowlist mismatch"):
        freeze_control_arm(
            metrics,
            "integrity_v2_receipt_allowlist",
            sweep_summary=sweep,
            artifact_sources=artifacts,
        )


def test_integrity_v2_readiness_rejects_incomplete_current_source_inventory(
    mock_frozen_root,
    tmp_path,
    monkeypatch,
):
    metrics, sweep, artifacts = _integrity_v2_fixture(tmp_path, monkeypatch)
    freeze_control_arm(
        metrics,
        "integrity_v2_source_inventory",
        sweep_summary=sweep,
        artifact_sources=artifacts,
    )
    fake_repo = (tmp_path / "changed_repo").resolve()
    (fake_repo / "src").mkdir(parents=True)
    (fake_repo / "src" / "only.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(control_arm, "REPO_ROOT", fake_repo)

    validation = validate_frozen_control_arm("integrity_v2_source_inventory")

    assert validation["valid"] is False
    assert any("source inventory differs" in error for error in validation["errors"])


def test_integrity_v2_freeze_rejects_policy_that_is_not_finally_validated(
    mock_frozen_root,
    tmp_path,
    monkeypatch,
):
    metrics, sweep, artifacts = _integrity_v2_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        control_arm,
        "_load_policy_bound_integrity_v2_contract",
        lambda require_final_validation=True: (_ for _ in ()).throw(
            RuntimeError("Integrity-v2 final policy is not promotion-ready")
        ),
    )

    with pytest.raises(ValueError, match="final policy is not promotion-ready"):
        freeze_control_arm(
            metrics,
            "integrity_v2_invalid_policy",
            sweep_summary=sweep,
            artifact_sources=artifacts,
        )


def _stale_baseline_evidence_root(tmp_path):
    """Copy the real policy baseline CSV with a perturbed embedded protocol.

    Returns the parsed real policy payload, a fixture repository root holding
    the perturbed CSV at the policy-declared relative path, and the perturbed
    file's SHA-256.
    """
    policy_payload = json.loads(
        control_arm.INTEGRITY_V2_POLICY_PATH.read_text(encoding="utf-8")
    )
    baseline = policy_payload["baseline"]
    evidence_bytes = (
        control_arm.REPO_ROOT / str(baseline["evidence_path"])
    ).read_bytes()
    current_protocol = str(baseline["evidence_protocol_sha256"]).encode("ascii")
    stale_bytes = evidence_bytes.replace(current_protocol, b"e" * 64)
    assert stale_bytes != evidence_bytes
    fixture_root = tmp_path / "stale_evidence_root"
    stale_evidence_path = fixture_root / str(baseline["evidence_path"])
    stale_evidence_path.parent.mkdir(parents=True)
    stale_evidence_path.write_bytes(stale_bytes)
    return policy_payload, fixture_root, hashlib.sha256(stale_bytes).hexdigest()


def _minimal_integrity_v2_freeze_artifacts(tmp_path):
    receipt_path = tmp_path / control_arm.INTEGRITY_V2_RECEIPT_FILENAME
    receipt_path.write_text("{}", encoding="utf-8")
    index_path = tmp_path / control_arm.INTEGRITY_V2_EVALUATION_INDEX_FILENAME
    index_path.write_text("event_date\n", encoding="utf-8")
    return {
        control_arm.INTEGRITY_V2_RECEIPT_FILENAME: receipt_path,
        control_arm.INTEGRITY_V2_EVALUATION_INDEX_FILENAME: index_path,
    }


def test_integrity_v2_freeze_rejects_stale_baseline_evidence_protocol(
    mock_frozen_root,
    tmp_path,
    monkeypatch,
):
    """DIR-AUD-P2-017: a baseline CSV embedding a stale protocol blocks freeze.

    The policy binds the perturbed CSV's exact file SHA, so only the embedded
    ``protocol_sha256`` differs — the intermediate-parent scenario. The freeze
    must raise through the real, unstubbed finality gate before writing.
    """
    from scripts import check_scheduled_refit_quality as quality_gate

    policy_payload, fixture_root, stale_sha = _stale_baseline_evidence_root(
        tmp_path
    )
    policy_payload["baseline"]["evidence_sha256"] = stale_sha
    stale_policy_path = tmp_path / "scheduled_refit_policy_v2_stale.json"
    stale_policy_path.write_text(
        json.dumps(policy_payload, indent=2), encoding="utf-8"
    )
    monkeypatch.setattr(control_arm, "INTEGRITY_V2_POLICY_PATH", stale_policy_path)
    monkeypatch.setattr(quality_gate, "REPO_ROOT", fixture_root)

    metrics = {"accuracy": 0.6, "brier": 0.2, "log_loss": 0.6, "ece": 0.05}
    with pytest.raises(
        ValueError,
        match="baseline evidence is stale.*protocol_sha256",
    ):
        freeze_control_arm(
            metrics,
            "integrity_v2_stale_protocol",
            artifact_sources=_minimal_integrity_v2_freeze_artifacts(tmp_path),
        )
    assert not control_arm._control_arm_dir("integrity_v2_stale_protocol").exists()


def test_integrity_v2_freeze_rejects_baseline_evidence_sha_drift(
    mock_frozen_root,
    tmp_path,
    monkeypatch,
):
    """DIR-AUD-P2-017: baseline CSV bytes drifting from the policy block freeze."""
    from scripts import check_scheduled_refit_quality as quality_gate

    _policy_payload, fixture_root, _stale_sha = _stale_baseline_evidence_root(
        tmp_path
    )
    monkeypatch.setattr(quality_gate, "REPO_ROOT", fixture_root)

    metrics = {"accuracy": 0.6, "brier": 0.2, "log_loss": 0.6, "ece": 0.05}
    with pytest.raises(
        ValueError,
        match="baseline evidence is stale.*SHA-256 does not match",
    ):
        freeze_control_arm(
            metrics,
            "integrity_v2_stale_sha",
            artifact_sources=_minimal_integrity_v2_freeze_artifacts(tmp_path),
        )
    assert not control_arm._control_arm_dir("integrity_v2_stale_sha").exists()


def test_integrity_v2_freeze_succeeds_with_current_baseline_via_real_finality_gate(
    mock_frozen_root,
    tmp_path,
    monkeypatch,
):
    """DIR-AUD-P2-017 positive twin: the byte-current baseline freezes.

    The real finality gate (policy load, registry validation, protocol
    recompute, baseline-evidence currency) runs unstubbed against the real
    repository policy and its current baseline CSV inside the real freeze
    boundary; only the downstream evidence bindings come from the synthetic
    fixture, which the sibling freeze tests exercise independently.
    """
    real_loader = control_arm._load_policy_bound_integrity_v2_contract
    metrics, sweep, artifacts = _integrity_v2_fixture(tmp_path, monkeypatch)
    fixture_contract = control_arm._load_policy_bound_integrity_v2_contract()

    finality_calls = []

    def _real_gate_then_fixture_contract(require_final_validation=True):
        real_loader(require_final_validation=require_final_validation)
        finality_calls.append(require_final_validation)
        return fixture_contract

    monkeypatch.setattr(
        control_arm,
        "_load_policy_bound_integrity_v2_contract",
        _real_gate_then_fixture_contract,
    )

    freeze_dir = freeze_control_arm(
        metrics,
        "integrity_v2_current_baseline",
        sweep_summary=sweep,
        artifact_sources=artifacts,
    )
    assert freeze_dir.is_dir()
    assert finality_calls == [True]
    assert (
        validate_frozen_control_arm("integrity_v2_current_baseline")["valid"]
        is True
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "duplicates"),
        ("unlisted", "unlisted inputs"),
        ("noncanonical", "not canonical"),
    ],
)
def test_integrity_v2_freeze_rejects_bad_odds_inventory_membership(
    mock_frozen_root,
    tmp_path,
    monkeypatch,
    mutation,
    message,
):
    from src.strategy import run_evaluation

    metrics, sweep, artifacts = _integrity_v2_fixture(tmp_path, monkeypatch)
    odds_path = artifacts[control_arm.INTEGRITY_V2_ODDS_SOURCE_INVENTORY_FILENAME]
    odds_payload = json.loads(odds_path.read_text(encoding="utf-8"))
    if mutation == "duplicate":
        odds_payload["entries"].append(dict(odds_payload["entries"][0]))
    elif mutation == "unlisted":
        unlisted_path = odds_path.parent / "unlisted.csv"
        unlisted_path.write_text("event_date,odds\n", encoding="utf-8")
        odds_payload["entries"].append(
            {
                "source_file": unlisted_path.name,
                "resolved_path": str(unlisted_path),
                "sha256": control_arm._sha256(unlisted_path),
            }
        )
    else:
        source_path = Path(odds_payload["entries"][0]["resolved_path"])
        odds_payload["entries"][0]["resolved_path"] = str(
            source_path.parent / ".." / source_path.parent.name / source_path.name
        )
    run_evaluation._write_json(odds_path, odds_payload)
    _rebind_integrity_provenance_after_odds_edit(metrics, sweep, artifacts)

    with pytest.raises(ValueError, match=message):
        freeze_control_arm(
            metrics,
            f"integrity_v2_bad_odds_{mutation}",
            sweep_summary=sweep,
            artifact_sources=artifacts,
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("model_spec_payload_sha256", "0" * 64),
        ("feature_contract_count", 258),
        ("feature_contract_sha256", "0" * 64),
    ],
)
def test_integrity_v2_freeze_rejects_spec_or_feature_contract_drift(
    mock_frozen_root,
    tmp_path,
    monkeypatch,
    field,
    bad_value,
):
    metrics, sweep, artifacts = _integrity_v2_fixture(tmp_path, monkeypatch)
    metrics[field] = bad_value

    with pytest.raises(ValueError, match=field):
        freeze_control_arm(
            metrics,
            "integrity_v2_drift",
            sweep_summary=sweep,
            artifact_sources=artifacts,
        )
    assert not (mock_frozen_root / "control_arm_integrity_v2_drift").exists()


@pytest.mark.parametrize("field", ["source_fingerprint", "evaluation_protocol"])
def test_integrity_v2_freeze_rejects_run_provenance_drift(
    mock_frozen_root,
    tmp_path,
    monkeypatch,
    field,
):
    metrics, sweep, artifacts = _integrity_v2_fixture(tmp_path, monkeypatch)
    if field == "source_fingerprint":
        metrics[field] = "e" * 64
    else:
        metrics[field] = {
            **metrics[field],
            "allow_closing_odds": True,
        }

    with pytest.raises(ValueError, match=field):
        freeze_control_arm(
            metrics,
            f"integrity_v2_{field}_drift",
            sweep_summary=sweep,
            artifact_sources=artifacts,
        )
    assert not (
        mock_frozen_root / f"control_arm_integrity_v2_{field}_drift"
    ).exists()


def test_integrity_v2_freeze_rejects_receipt_bound_artifact_tampering(
    mock_frozen_root,
    tmp_path,
    monkeypatch,
):
    metrics, sweep, artifacts = _integrity_v2_fixture(tmp_path, monkeypatch)
    tampered_path = artifacts[
        control_arm.INTEGRITY_V2_PRODUCTION_BET_LOG_FILENAME
    ]
    tampered_path.write_text(
        tampered_path.read_text(encoding="utf-8") + "2026-01-01,999.0,10.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact hash mismatch.*production_bet_log"):
        freeze_control_arm(
            metrics,
            "integrity_v2_tampered",
            sweep_summary=sweep,
            artifact_sources=artifacts,
        )
    assert not (mock_frozen_root / "control_arm_integrity_v2_tampered").exists()


@pytest.mark.parametrize("source_kind", ["dataset", "odds"])
def test_integrity_v2_freeze_rejects_provenance_source_tampering(
    mock_frozen_root,
    tmp_path,
    monkeypatch,
    source_kind,
):
    metrics, sweep, artifacts = _integrity_v2_fixture(tmp_path, monkeypatch)
    input_provenance = json.loads(
        artifacts[control_arm.INTEGRITY_V2_INPUT_PROVENANCE_FILENAME].read_text(
            encoding="utf-8"
        )
    )
    if source_kind == "dataset":
        source_path = Path(input_provenance["dataset_fights_path"])
        expected_message = "input provenance source hash mismatch"
    else:
        odds_inventory = json.loads(
            artifacts[
                control_arm.INTEGRITY_V2_ODDS_SOURCE_INVENTORY_FILENAME
            ].read_text(encoding="utf-8")
        )
        source_path = Path(odds_inventory["entries"][0]["resolved_path"])
        expected_message = "odds source inventory entry 0 source mismatch"
    source_path.write_text(
        source_path.read_text(encoding="utf-8") + "tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=expected_message):
        freeze_control_arm(
            metrics,
            f"integrity_v2_{source_kind}_source_tampered",
            sweep_summary=sweep,
            artifact_sources=artifacts,
        )
    assert not (
        mock_frozen_root
        / f"control_arm_integrity_v2_{source_kind}_source_tampered"
    ).exists()


def test_integrity_v2_freeze_rejects_stale_sweep_paired_with_valid_receipt(
    mock_frozen_root,
    tmp_path,
    monkeypatch,
):
    metrics, sweep, artifacts = _integrity_v2_fixture(tmp_path, monkeypatch)
    stale_sweep = {**sweep, "roi": 0.99}

    with pytest.raises(ValueError, match="does not match the supplied sweep summary"):
        freeze_control_arm(
            metrics,
            "integrity_v2_stale_sweep",
            sweep_summary=stale_sweep,
            artifact_sources=artifacts,
        )
    assert not (mock_frozen_root / "control_arm_integrity_v2_stale_sweep").exists()

def test_load_metrics_raises_on_missing(mock_frozen_root):
    with pytest.raises(FileNotFoundError):
        load_frozen_control_metrics("non_existent_id")

def test_existing_freeze_20260313_validates():
    """Verify that the actual data/frozen/control_arm_20260313 validates."""
    freeze_dir = control_arm.FROZEN_DIR / "control_arm_20260313"
    if not freeze_dir.exists():
        pytest.skip(f"Generated control-arm freeze not present: {freeze_dir}")
    validation = validate_frozen_control_arm("20260313")
    if validation["valid"]:
        assert True
    else:
        pytest.fail(f"Existing freeze 20260313 failed validation: {validation['errors']}")
