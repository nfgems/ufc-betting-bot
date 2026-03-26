import json
from pathlib import Path

import pandas as pd
import pytest

import src.bot as bot_module
from src.model import train as train_module


def _write_static_test_set(tmp_path: Path) -> Path:
    test_set_path = tmp_path / "test_set.csv"
    pd.DataFrame(
        [
            {
                "event_date": "2024-01-01",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "target": 1,
                "a_market_prob": 0.5,
                "b_market_prob": 0.5,
            }
        ]
    ).to_csv(test_set_path, index=False)
    return test_set_path


def _model_result() -> dict:
    return {
        "artifact_path": str(Path("models") / "xgboost_model.pkl"),
        "training_spec": {
            "name": "full_live_contract_v6_tuned",
            "feature_cols": ["diff_stat"],
            "dataset_variant": "pulled_all_plus_legacy_market",
            "train_start_date": "2014-01-01",
            "train_end_date": "",
            "train_cutoff_date": "2022-01-01",
        },
    }


def _write_metadata(test_set_path: Path, *, cutoff_date: str) -> None:
    metadata = {
        "spec_name": "full_live_contract_v6_tuned",
        "feature_count": 1,
        "feature_hash": train_module._feature_contract_hash(["diff_stat"]),
        "dataset_variant": "pulled_all_plus_legacy_market",
        "train_start_date": "2014-01-01",
        "train_end_date": "",
        "train_cutoff_date": cutoff_date,
        "generated_at": "2026-03-26T00:00:00",
        "row_count": 1,
        "test_set_sha256": "abc123",
    }
    train_module.test_set_metadata_path(test_set_path).write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def test_cmd_backtest_static_rejects_mismatched_metadata(monkeypatch, tmp_path):
    test_set_path = _write_static_test_set(tmp_path)
    _write_metadata(test_set_path, cutoff_date="2027-01-01")

    monkeypatch.setattr("src.model.train.load_model", lambda _model_ref: _model_result())
    monkeypatch.setattr(bot_module, "_resolve_no_odds_model_arg", lambda _model_ref: None)
    monkeypatch.setattr(
        "src.strategy.backtest.run_backtest",
        lambda *args, **kwargs: pytest.fail("run_backtest should not execute on mismatch"),
    )

    args = type(
        "Args",
        (),
        {
            "static": True,
            "model": "xgboost",
            "model_path": None,
            "test_set_path": str(test_set_path),
            "allow_mismatch": False,
            "bankroll": 100.0,
            "min_edge": 0.01,
            "kelly": 0.25,
            "execution_mode": "legacy",
            "retrain_months": 6,
            "initial_years": 5,
        },
    )()

    with pytest.raises(ValueError, match="train_cutoff_date"):
        bot_module.cmd_backtest(args)


def test_cmd_backtest_static_allows_mismatch_override(monkeypatch, tmp_path):
    test_set_path = _write_static_test_set(tmp_path)
    _write_metadata(test_set_path, cutoff_date="2027-01-01")
    calls = {}

    monkeypatch.setattr("src.model.train.load_model", lambda _model_ref: _model_result())
    monkeypatch.setattr(bot_module, "_resolve_no_odds_model_arg", lambda _model_ref: None)
    monkeypatch.setattr(
        "src.strategy.backtest.plot_backtest",
        lambda _result: None,
    )

    def fake_run_backtest(test_df, **kwargs):
        calls["rows"] = len(test_df)
        calls["kwargs"] = kwargs
        return {
            "stats": {"execution_mode": kwargs.get("execution_mode", "legacy")},
            "bet_log": pd.DataFrame(),
            "bankroll_history": [100.0],
        }

    monkeypatch.setattr("src.strategy.backtest.run_backtest", fake_run_backtest)

    args = type(
        "Args",
        (),
        {
            "static": True,
            "model": "xgboost",
            "model_path": None,
            "test_set_path": str(test_set_path),
            "allow_mismatch": True,
            "bankroll": 100.0,
            "min_edge": 0.01,
            "kelly": 0.25,
            "execution_mode": "legacy",
            "retrain_months": 6,
            "initial_years": 5,
        },
    )()

    bot_module.cmd_backtest(args)

    assert calls["rows"] == 1
    assert calls["kwargs"]["execution_mode"] == "legacy"
