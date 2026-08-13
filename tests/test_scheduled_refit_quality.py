from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from scripts import check_production_refit_contract as contract_gate
from scripts import check_scheduled_refit_quality as quality
from src.model.training_spec import resolve_named_training_spec


POLICY_PATH = Path("config/scheduled_refit_policy_v1.json")
NOW = datetime(2026, 8, 6, tzinfo=timezone.utc)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_test_policy(path: Path) -> dict:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy["baseline"].update(
        {
            "evaluation_start_date": "2026-07-23",
            "evaluation_end_date": "2026-08-01",
            "evaluation_n_fights": 10,
            "evaluation_n_folds": 2,
            "model_brier_score": 0.2,
            "model_ece": 0.05,
            "strategy_total_bets": 3,
            "strategy_roi": 0.1,
            "strategy_total_profit": 3.0,
            "strategy_avg_clv": 0.0,
            "strategy_max_drawdown_pct": 0.1,
        }
    )
    policy["health_limits"].update(
        {
            "minimum_evaluation_fights": 10,
            "minimum_evaluation_folds": 2,
            "minimum_strategy_bets": 3,
            "minimum_strategy_roi": 0.0,
            "minimum_strategy_total_profit": 0.0,
        }
    )
    _write_json(path, policy)
    return contract_gate.load_policy(path)


def _write_features(path: Path) -> None:
    spec = resolve_named_training_spec(
        "full_live_contract_v6_durability_corrected_20260805_fullfit"
    )
    row = {column: 0.0 for column in spec.feature_cols}
    row["event_date"] = "2026-08-01"
    pd.DataFrame([row]).to_csv(path, index=False)


def _evaluation_index() -> pd.DataFrame:
    dates = pd.date_range("2026-07-23", "2026-08-01", freq="D")
    return pd.DataFrame(
        {
            "event_date": dates.strftime("%Y-%m-%d"),
            "fighter_a": [f"A{index}" for index in range(10)],
            "fighter_b": [f"B{index}" for index in range(10)],
            "target": [index % 2 for index in range(10)],
            "fold": [1] * 5 + [2] * 5,
            "train_end": ["2026-07-22"] * 5 + ["2026-07-27"] * 5,
            "test_end": ["2026-07-27"] * 5 + ["2026-08-02"] * 5,
        },
        columns=quality._EVALUATION_INDEX_COLUMNS,
    )


def _bets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": ["production_gated"] * 3,
            "event_date": ["2026-07-23", "2026-07-24", "2026-07-25"],
            "fighter_a": ["A0", "A1", "A2"],
            "fighter_b": ["B0", "B1", "B2"],
            "bet_size": [10.0, 10.0, 10.0],
            "profit": [1.0, 1.0, 1.0],
            "clv": [0.0, 0.0, 0.0],
            "edge": [0.03, 0.03, 0.03],
            "won": [True, True, True],
            "execution_mode": ["realistic"] * 3,
            "bankroll_after": [501.0, 502.0, 503.0],
        }
    )


def _quality_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    policy_path = tmp_path / "policy.json"
    policy = _write_test_policy(policy_path)
    features_path = tmp_path / "features.csv"
    _write_features(features_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    spec_name = policy["contract"]["evaluation_spec_name"]

    evaluation_index = _evaluation_index()
    evaluation_index.to_csv(
        artifacts / f"{spec_name}_evaluation_index.csv",
        index=False,
        lineterminator="\n",
    )
    bets = _bets()
    bets.to_csv(
        artifacts / f"{spec_name}_production_gated_bets.csv",
        index=False,
        lineterminator="\n",
    )
    summary = {
        "spec": spec_name,
        "spec_payload_sha256": policy["contract"]["evaluation_spec_payload_sha256"],
        "policy_sha256": contract_gate.file_sha256(policy_path),
        "features_sha256": contract_gate.file_sha256(features_path),
        "protocol_sha256": contract_gate.scheduled_protocol_sha256(policy),
        "evaluation_sample_sha256": quality.evaluation_index_sha256(evaluation_index),
        "model_seed": 42,
        "odds_noise_seed": 42,
        "execution_mode": "realistic",
        "entry_offset_days": 1.0,
        "entry_offset_for_features": True,
        "evaluation_start_date": "2026-07-23",
        "evaluation_end_date": "2026-08-01",
        "evaluation_n_fights": 10,
        "evaluation_n_folds": 2,
        "model_brier_score": 0.2,
        "model_ece": 0.05,
        "model_n_fights": 10,
        "strategy_execution_mode": "realistic",
        "strategy_total_bets": 3,
        "strategy_win_rate": 1.0,
        "strategy_total_wagered": 30.0,
        "strategy_roi": 0.1,
        "strategy_total_profit": 3.0,
        "strategy_avg_clv": 0.0,
        "strategy_max_drawdown_pct": 0.0,
    }
    pd.DataFrame([summary]).to_csv(
        artifacts / "track_c_summary.csv", index=False, lineterminator="\n"
    )
    return policy_path, features_path, artifacts, summary


def test_quality_gate_accepts_one_bound_arm_and_reconciled_bets(tmp_path: Path) -> None:
    policy, features, artifacts, _summary = _quality_fixture(tmp_path)

    result = quality.evaluate_quality(
        policy_path=policy,
        features_path=features,
        artifacts_dir=artifacts,
        now=NOW,
    )

    assert result["errors"] == []
    assert result["metrics"]["evaluation_n_fights"] == 10
    assert result["metrics"]["strategy_total_profit"] == 3.0
    assert result["baseline_reference"]["comparison_role"] == (
        "root_release_reference_thresholds"
    )
    assert result["baseline_reference"]["shared_settings_verified"] is True
    assert result["baseline_reference"]["evidence_protocol_sha256"] == (
        result["baseline_reference"]["scheduled_protocol_sha256"]
    )


def test_quality_gate_reports_metric_regression_as_unhealthy(tmp_path: Path) -> None:
    policy, features, artifacts, summary = _quality_fixture(tmp_path)
    summary["model_brier_score"] = 0.5
    pd.DataFrame([summary]).to_csv(artifacts / "track_c_summary.csv", index=False)

    result = quality.evaluate_quality(
        policy_path=policy,
        features_path=features,
        artifacts_dir=artifacts,
        now=NOW,
    )

    assert any("Brier" in error for error in result["errors"])


def test_quality_gate_rejects_multiple_arms(tmp_path: Path) -> None:
    policy, features, artifacts, summary = _quality_fixture(tmp_path)
    pd.DataFrame([summary, summary]).to_csv(artifacts / "track_c_summary.csv", index=False)

    with pytest.raises(quality.QualityInputError, match="exactly one arm"):
        quality.evaluate_quality(
            policy_path=policy,
            features_path=features,
            artifacts_dir=artifacts,
            now=NOW,
        )


def test_quality_gate_rejects_tampered_fight_sample(tmp_path: Path) -> None:
    policy, features, artifacts, _summary = _quality_fixture(tmp_path)
    index_path = artifacts / "full_live_contract_v6_durability_evaluation_index.csv"
    frame = pd.read_csv(index_path)
    frame.loc[0, "fighter_a"] = "Tampered"
    frame.to_csv(index_path, index=False)

    with pytest.raises(quality.QualityInputError, match="SHA-256 mismatch"):
        quality.evaluate_quality(
            policy_path=policy,
            features_path=features,
            artifacts_dir=artifacts,
            now=NOW,
        )


def test_quality_gate_rejects_summary_bet_log_mismatch(tmp_path: Path) -> None:
    policy, features, artifacts, summary = _quality_fixture(tmp_path)
    summary["strategy_total_profit"] = 999.0
    pd.DataFrame([summary]).to_csv(artifacts / "track_c_summary.csv", index=False)

    with pytest.raises(quality.QualityInputError, match="strategy_total_profit mismatch"):
        quality.evaluate_quality(
            policy_path=policy,
            features_path=features,
            artifacts_dir=artifacts,
            now=NOW,
        )


def test_quality_cli_exit_codes_and_reports(tmp_path: Path) -> None:
    policy, features, artifacts, summary = _quality_fixture(tmp_path)
    report = tmp_path / "unhealthy.json"
    summary["model_brier_score"] = 0.5
    pd.DataFrame([summary]).to_csv(artifacts / "track_c_summary.csv", index=False)

    assert quality.main(
        [
            "--policy",
            str(policy),
            "--features",
            str(features),
            "--artifacts-dir",
            str(artifacts),
            "--report",
            str(report),
        ]
    ) == 1
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "UNHEALTHY"

    error_report = tmp_path / "error.json"
    assert quality.main(
        [
            "--policy",
            str(policy),
            "--features",
            str(tmp_path / "missing.csv"),
            "--artifacts-dir",
            str(artifacts),
            "--report",
            str(error_report),
        ]
    ) == 2
    assert json.loads(error_report.read_text(encoding="utf-8"))["status"] == "ERROR"
