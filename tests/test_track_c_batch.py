from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import check_production_refit_contract as contract_gate
from scripts import check_scheduled_refit_quality as quality
from scripts import track_c_batch as track
from src.config import TRAIN_CUTOFF_DATE
from src.model.training_spec import NamedModelTrainingSpec, resolve_named_training_spec
from src.strategy import duo_trader_sweep as challenger_sweep
from src.strategy import value as value_strategy
from src.strategy.runtime_strategy import canonical_json_sha256


POLICY_PATH = Path("config/scheduled_refit_policy_v1.json")


def _features_for_spec(spec_name: str) -> pd.DataFrame:
    spec = resolve_named_training_spec(spec_name)
    rows = []
    for date in ("2026-07-25", "2026-08-01"):
        row = {column: 0.0 for column in spec.feature_cols}
        row["event_date"] = date
        rows.append(row)
    return pd.DataFrame(rows)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_date": ["2026-07-25", "2026-08-01"],
            "fighter_a": ["A", "C"],
            "fighter_b": ["B", "D"],
            "target": [1, 0],
            "fold": [1, 1],
            "train_end": ["2026-07-24", "2026-07-24"],
            "test_end": ["2026-08-02", "2026-08-02"],
        }
    )


def _fake_result() -> dict:
    bet_log = pd.DataFrame(
        {
            "strategy": ["production_gated"],
            "event_date": ["2026-07-25"],
            "fighter_a": ["A"],
            "fighter_b": ["B"],
            "bet_size": [10.0],
            "profit": [1.0],
            "clv": [0.0],
            "edge": [0.03],
            "won": [True],
            "execution_mode": ["realistic"],
            "bankroll_after": [501.0],
        }
    )
    return {
        "predictive_metrics": {
            "evaluation_window": {
                "start_date": "2026-07-25",
                "end_date": "2026-08-01",
                "n_fights": 2,
                "n_folds": 1,
            },
            "models": {
                "xgboost": {"brier_score": 0.2, "ece": 0.04, "n_fights": 2}
            },
            "blend": {"overall": {}},
        },
        "summary": pd.DataFrame(
            [
                {
                    "strategy": "production_gated",
                    "total_bets": 1,
                    "win_rate": 1.0,
                    "total_wagered": 10.0,
                    "roi": 0.1,
                    "total_profit": 1.0,
                    "avg_clv": 0.0,
                    "max_drawdown_pct": 0.0,
                    "execution_mode": "realistic",
                }
            ]
        ),
        "predictions": _predictions(),
        "strategy_results": {"production_gated": {"bet_log": bet_log}},
    }


def _locked_strategy() -> dict:
    return {
        "name": "confirmed_winner",
        "s_min_edge": 0.03,
        "s_blend_weight": 0.35,
        "s_model_agreement_min_edge": 0.02,
        "c_min_model_prob": 0.70,
        "c_min_no_odds_prob": 0.55,
        "c_share": 0.40,
        "c_bet_fraction": 0.04,
        "c_max_bet_fraction": 0.07,
        "c_min_edge": 0.01,
        "c_max_decimal_odds": 2.5,
        "m_enabled": False,
        "m_min_model_prob": 0.65,
        "m_min_no_odds_prob": 0.50,
        "m_bet_fraction": 0.03,
        "m_share": 0.30,
    }


def test_confirmed_track_c_builds_exact_full_sweep_config_and_hash():
    strategy = _locked_strategy()
    policy = {
        "performance_confirmation": {
            "strategy_config": strategy,
            "strategy_config_sha256": canonical_json_sha256(strategy),
        }
    }

    config, digest = track._confirmed_sweep_config(policy)

    assert digest == canonical_json_sha256(strategy)
    for field, value in strategy.items():
        assert getattr(config, field) == value


def test_confirmed_track_c_written_log_reconciles_event_end_drawdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = contract_gate.load_policy(POLICY_PATH)
    policy["evaluation"]["execution_mode"] = "legacy"
    strategy = _locked_strategy()
    strategy.update(
        {
            "s_min_edge": 0.30,
            "s_blend_weight": 1.0,
            "s_model_agreement_min_edge": 0.0,
            "c_min_model_prob": 0.80,
            "c_min_no_odds_prob": 0.80,
            "c_min_edge": 0.10,
            "c_share": 0.50,
            "c_bet_fraction": 0.10,
            "c_max_bet_fraction": 0.20,
        }
    )
    policy["performance_confirmation"] = {
        "strategy_config": strategy,
        "strategy_config_sha256": canonical_json_sha256(strategy),
    }
    spec_name = policy["contract"]["evaluation_spec_name"]
    registered_spec = resolve_named_training_spec(spec_name)
    fullfit_spec = resolve_named_training_spec(
        policy["contract"]["fullfit_spec_name"]
    )
    features = _features_for_spec(spec_name)
    features_path = tmp_path / "features.csv"
    features.to_csv(features_path, index=False, lineterminator="\n")
    out = tmp_path / "out"
    predictions = pd.DataFrame(
        [
            {
                "event_date": "2025-01-01",
                "fighter_a": "C Winner",
                "fighter_b": "C Opponent",
                "target": 1,
                "prob_a": 0.80,
                "prob_b": 0.20,
                "a_market_prob": 0.60,
                "b_market_prob": 0.40,
                "no_odds_prob_a": 0.80,
                "no_odds_prob_b": 0.20,
                "closing_prob_a": 0.60,
                "closing_prob_b": 0.40,
                "a_num_fights": 6,
                "b_num_fights": 6,
            },
            {
                "event_date": "2025-01-01",
                "fighter_a": "S Loser",
                "fighter_b": "S Winner",
                "target": 0,
                "prob_a": 0.80,
                "prob_b": 0.20,
                "a_market_prob": 0.40,
                "b_market_prob": 0.60,
                "no_odds_prob_a": 0.80,
                "no_odds_prob_b": 0.20,
                "closing_prob_a": 0.40,
                "closing_prob_b": 0.60,
                "a_num_fights": 6,
                "b_num_fights": 6,
            },
            {
                "event_date": "2025-01-02",
                "fighter_a": "S Winner",
                "fighter_b": "S Opponent",
                "target": 1,
                "prob_a": 0.80,
                "prob_b": 0.20,
                "a_market_prob": 0.40,
                "b_market_prob": 0.60,
                "no_odds_prob_a": 0.80,
                "no_odds_prob_b": 0.20,
                "closing_prob_a": 0.40,
                "closing_prob_b": 0.60,
                "a_num_fights": 6,
                "b_num_fights": 6,
            },
        ]
    )
    predictions["fold"] = 1
    predictions["train_end"] = "2024-12-31"
    predictions["test_end"] = "2025-01-03"
    result = _fake_result()
    result["predictions"] = predictions
    result["predictive_metrics"]["evaluation_window"] = {
        "start_date": "2025-01-01",
        "end_date": "2025-01-02",
        "n_fights": 3,
        "n_folds": 1,
    }

    monkeypatch.setattr(contract_gate, "load_policy", lambda _path: policy)
    monkeypatch.setattr(
        contract_gate,
        "validate_policy_registry",
        lambda _policy: ([], registered_spec, fullfit_spec),
    )
    monkeypatch.setattr(track, "_load_or_rebuild_features", lambda _path: features.copy())
    monkeypatch.setattr(
        track,
        "run_walkforward_strategy_comparison",
        lambda _frame, **_kwargs: result,
    )
    monkeypatch.setattr(value_strategy, "_passes_filters", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        value_strategy,
        "compute_independent_blend_probs",
        lambda model_a, _market_a, _no_odds_a, model_b, _market_b, _no_odds_b, *_args, **_kwargs: (
            model_a,
            model_b,
        ),
    )
    monkeypatch.setattr(
        challenger_sweep.BankrollManager,
        "kelly_bet_size",
        lambda self, _model_prob, _decimal_odds: round(self.total_equity * 0.10, 2),
    )

    assert track.main(
        [
            "--scheduled-policy",
            str(POLICY_PATH),
            "--features-cache",
            str(features_path),
            "--out",
            str(out),
        ]
    ) == 0

    summary_row = pd.read_csv(out / "track_c_summary.csv").iloc[0]
    bet_path = out / f"{spec_name}_production_gated_bets.csv"
    bets = pd.read_csv(bet_path)
    evaluation_index = pd.read_csv(out / f"{spec_name}_evaluation_index.csv")
    assert bets.loc[0, "event_date"] == bets.loc[1, "event_date"]
    per_bet_drawdown = quality.compute_max_drawdown(
        [500.0, *bets["bankroll_after"].tolist()]
    )["max_drawdown_pct"]
    assert per_bet_drawdown > summary_row["strategy_max_drawdown_pct"]

    _reconciled_bets, reconciled = quality._read_and_reconcile_bets(
        bet_path,
        row=summary_row,
        evaluation_index=evaluation_index,
        policy=policy,
    )
    assert reconciled["strategy_max_drawdown_pct"] == pytest.approx(
        summary_row["strategy_max_drawdown_pct"]
    )

    legacy_policy = dict(policy)
    legacy_policy.pop("performance_confirmation")
    legacy_row = summary_row.copy()
    legacy_row["strategy_max_drawdown_pct"] = per_bet_drawdown
    _legacy_bets, legacy_reconciled = quality._read_and_reconcile_bets(
        bet_path,
        row=legacy_row,
        evaluation_index=evaluation_index,
        policy=legacy_policy,
    )
    assert legacy_reconciled["strategy_max_drawdown_pct"] == pytest.approx(
        per_bet_drawdown
    )

    forged = summary_row.copy()
    forged["strategy_max_drawdown_pct"] = per_bet_drawdown
    with pytest.raises(quality.QualityInputError, match="strategy_max_drawdown_pct mismatch"):
        quality._read_and_reconcile_bets(
            bet_path,
            row=forged,
            evaluation_index=evaluation_index,
            policy=policy,
        )


def test_scheduled_mode_runs_exactly_policy_arm_and_persists_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = contract_gate.load_policy(POLICY_PATH)
    spec_name = policy["contract"]["evaluation_spec_name"]
    features = _features_for_spec(spec_name)
    features_path = tmp_path / "features.csv"
    features.to_csv(features_path, index=False, lineterminator="\n")
    out = tmp_path / "out"
    captured: list[dict] = []

    monkeypatch.setattr(track, "_load_or_rebuild_features", lambda _path: features.copy())

    def fake_run(frame: pd.DataFrame, **kwargs: object) -> dict:
        captured.append({"frame": frame, **kwargs})
        return _fake_result()

    monkeypatch.setattr(track, "run_walkforward_strategy_comparison", fake_run)

    assert track.main(
        [
            "--scheduled-policy",
            str(POLICY_PATH),
            "--features-cache",
            str(features_path),
            "--out",
            str(out),
        ]
    ) == 0

    assert len(captured) == 1
    run = captured[0]
    assert run["spec"].name == spec_name
    assert run["retrain_months"] == 6
    assert run["initial_train_years"] == 5
    assert run["min_train_test_fights"] == 2
    assert run["bet_start_date"] == "2022-01-01"
    assert run["execution_mode"] == "realistic"
    assert run["entry_offset_days"] == 1.0
    assert run["entry_offset_for_features"] is True
    assert len(run["strategies"]) == 1
    assert run["strategies"][0].name == "production_gated"
    assert run["execution_config"].mode == "realistic"

    summary = pd.read_csv(out / "track_c_summary.csv")
    assert len(summary) == 1
    assert summary.loc[0, "spec"] == spec_name
    assert summary.loc[0, "policy_sha256"] == contract_gate.file_sha256(POLICY_PATH)
    assert summary.loc[0, "protocol_sha256"] == contract_gate.scheduled_protocol_sha256(
        policy
    )
    assert summary.loc[0, "features_sha256"] == contract_gate.file_sha256(features_path)
    assert (out / f"{spec_name}_evaluation_index.csv").is_file()
    assert (out / f"{spec_name}_production_gated_bets.csv").is_file()


def test_scheduled_mode_rejects_every_protocol_override(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        track.main(
            [
                "--scheduled-policy",
                str(POLICY_PATH),
                "--specs",
                "full_live_contract_v6_tuned",
                "--out",
                str(tmp_path / "out"),
            ]
        )

    assert exc_info.value.code == 2


def test_scheduled_mode_refuses_implicit_feature_rebuild(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="refusing implicit rebuild"):
        track.main(
            [
                "--scheduled-policy",
                str(POLICY_PATH),
                "--features-cache",
                str(tmp_path / "missing.csv"),
                "--out",
                str(tmp_path / "out"),
            ]
        )


def test_scheduled_mode_fails_on_missing_contract_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features_path = tmp_path / "features.csv"
    pd.DataFrame([{"event_date": "2026-08-01"}]).to_csv(features_path, index=False)
    monkeypatch.setattr(
        track,
        "_load_or_rebuild_features",
        lambda _path: pd.DataFrame([{"event_date": "2026-08-01"}]),
    )

    with pytest.raises(ValueError, match="contract columns missing"):
        track.main(
            [
                "--scheduled-policy",
                str(POLICY_PATH),
                "--features-cache",
                str(features_path),
                "--out",
                str(tmp_path / "out"),
            ]
        )


def test_legacy_cli_remains_supported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    features = pd.DataFrame(
        [{"event_date": "2026-08-01", "feature": 1.0}]
    )
    features_path = tmp_path / "features.csv"
    features.to_csv(features_path, index=False)
    out = tmp_path / "out"
    captured: list[dict] = []
    spec = NamedModelTrainingSpec(name="legacy-test", feature_cols=["feature"])
    monkeypatch.setattr(track, "resolve_named_training_spec", lambda _name: spec)
    monkeypatch.setattr(track, "_load_or_rebuild_features", lambda _path: features.copy())

    def fake_run(frame: pd.DataFrame, **kwargs: object) -> dict:
        captured.append({"frame": frame, **kwargs})
        return _fake_result()

    monkeypatch.setattr(track, "run_walkforward_strategy_comparison", fake_run)

    assert track.main(
        [
            "--specs",
            "legacy-test",
            "--features-cache",
            str(features_path),
            "--out",
            str(out),
        ]
    ) == 0

    assert len(captured) == 1
    assert captured[0]["retrain_months"] == 6
    assert captured[0]["bet_start_date"] == TRAIN_CUTOFF_DATE
    assert captured[0]["execution_mode"] == "legacy"
    assert captured[0]["entry_offset_days"] is None
    assert captured[0]["entry_offset_for_features"] is False
    summary = pd.read_csv(out / "track_c_summary.csv")
    assert summary.loc[0, "policy_sha256"] != summary.loc[0, "policy_sha256"]  # NaN


def test_evaluation_sample_hash_is_order_invariant() -> None:
    frame = _predictions()

    assert track._evaluation_sample_sha256(frame) == track._evaluation_sample_sha256(
        frame.iloc[::-1].reset_index(drop=True)
    )


def test_evaluation_sample_rejects_duplicate_fights() -> None:
    frame = pd.concat([_predictions(), _predictions().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate fight identities"):
        track._evaluation_sample_sha256(frame)
