import numpy as np
import pandas as pd
import pytest

import src.strategy.duo_trader_sweep as challenger_sweep


def test_betsapi_challenger_sweep_includes_production_baseline_row(monkeypatch):
    predictions = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2024-01-01"),
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "target": 1,
                "prob_a": 0.76,
                "prob_b": 0.24,
                "a_market_prob": 0.60,
                "b_market_prob": 0.40,
                "no_odds_prob_a": 0.71,
                "no_odds_prob_b": 0.29,
                "a_num_fights": 6,
                "b_num_fights": 6,
                "closing_prob_a": 0.64,
                "closing_prob_b": 0.36,
            }
        ]
    )

    monkeypatch.setattr(
        challenger_sweep,
        "_generate_walk_forward_predictions",
        lambda **kwargs: [(1, predictions.copy())],
    )
    monkeypatch.setattr(
        challenger_sweep,
        "_build_sweep_configs",
        lambda variant_name="betsapi_challenger": [
            challenger_sweep._production_sweep_config(
                variant_name=variant_name,
                name="challenger_production_controls",
                is_baseline=False,
            )
        ],
    )

    summary_rows, results = challenger_sweep.run_betsapi_challenger_sweep_comparison(
        initial_bankroll=100.0,
        bet_start_date="2022-01-01",
    )

    assert summary_rows[0]["config"] == "production_baseline"
    assert summary_rows[0]["is_baseline"] is True
    assert summary_rows[0]["model_variant"] == "baseline"
    assert any(row["model_variant"] == "betsapi_challenger" for row in summary_rows)
    assert "avg_clv" in summary_rows[0]
    assert "production_baseline" in results


def test_build_sweep_configs_honors_custom_grid():
    configs = challenger_sweep.build_sweep_configs(
        variant_name="betsapi_challenger",
        s_edges=[0.02],
        s_blends=[0.30],
        s_agreements=[0.01],
        c_model_probs=[0.70],
        c_no_odds_probs=[0.55],
        c_shares=[0.50, 1.00],
        c_min_edges=[0.00],
        c_max_decimal_odds_values=[2.0],
        include_production_controls=True,
    )

    assert len(configs) == 3
    assert configs[0].name == "challenger_production_controls"
    assert configs[1].c_share == 0.50
    assert configs[2].c_share == 1.00


def test_custom_challenger_sweep_summary_includes_param_columns(monkeypatch):
    predictions = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2024-01-01"),
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "target": 1,
                "prob_a": 0.76,
                "prob_b": 0.24,
                "a_market_prob": 0.60,
                "b_market_prob": 0.40,
                "no_odds_prob_a": 0.71,
                "no_odds_prob_b": 0.29,
                "a_num_fights": 6,
                "b_num_fights": 6,
                "closing_prob_a": 0.64,
                "closing_prob_b": 0.36,
            }
        ]
    )

    monkeypatch.setattr(
        challenger_sweep,
        "_generate_walk_forward_predictions",
        lambda **kwargs: [(1, predictions.copy())],
    )

    configs = challenger_sweep.build_sweep_configs(
        variant_name="betsapi_challenger",
        s_edges=[0.02],
        s_blends=[0.30],
        s_agreements=[0.01],
        c_model_probs=[0.70],
        c_no_odds_probs=[0.55],
        c_shares=[0.50],
        c_min_edges=[0.00],
        c_max_decimal_odds_values=[2.0],
        include_production_controls=False,
    )

    summary_rows, _ = challenger_sweep.run_betsapi_challenger_sweep_comparison_with_configs(
        configs,
        initial_bankroll=100.0,
        bet_start_date="2022-01-01",
        include_production_baseline=True,
    )

    baseline_row = summary_rows[0]
    challenger_row = summary_rows[1]
    assert baseline_row["config"] == "production_baseline"
    assert challenger_row["s_min_edge"] == 0.02
    assert challenger_row["s_blend_weight"] == 0.30
    assert challenger_row["c_share"] == 0.50
    assert "s_roi" in challenger_row
    assert "c_roi" in challenger_row


def test_evaluate_config_bankroll_history_is_bet_granular(monkeypatch):
    predictions = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2025-01-01"),
                "fighter_a": "Skip A",
                "fighter_b": "Skip B",
                "target": 1,
                "prob_a": 0.80,
                "prob_b": 0.20,
                "a_market_prob": np.nan,
                "b_market_prob": 0.60,
                "no_odds_prob_a": 0.75,
                "no_odds_prob_b": 0.25,
                "a_num_fights": 6,
                "b_num_fights": 6,
            },
            {
                "event_date": pd.Timestamp("2025-01-02"),
                "fighter_a": "Winner A",
                "fighter_b": "Winner B",
                "target": 1,
                "prob_a": 0.80,
                "prob_b": 0.20,
                "a_market_prob": 0.40,
                "b_market_prob": 0.60,
                "no_odds_prob_a": 0.75,
                "no_odds_prob_b": 0.25,
                "a_num_fights": 6,
                "b_num_fights": 6,
            },
            {
                "event_date": pd.Timestamp("2025-01-03"),
                "fighter_a": "Pass A",
                "fighter_b": "Pass B",
                "target": 1,
                "prob_a": 0.50,
                "prob_b": 0.50,
                "a_market_prob": 0.50,
                "b_market_prob": 0.50,
                "no_odds_prob_a": 0.50,
                "no_odds_prob_b": 0.50,
                "a_num_fights": 6,
                "b_num_fights": 6,
            },
            {
                "event_date": pd.Timestamp("2025-01-04"),
                "fighter_a": "Loser A1",
                "fighter_b": "Loser B1",
                "target": 0,
                "prob_a": 0.70,
                "prob_b": 0.30,
                "a_market_prob": 0.50,
                "b_market_prob": 0.50,
                "no_odds_prob_a": 0.70,
                "no_odds_prob_b": 0.30,
                "a_num_fights": 6,
                "b_num_fights": 6,
            },
            {
                "event_date": pd.Timestamp("2025-01-05"),
                "fighter_a": "Loser A2",
                "fighter_b": "Loser B2",
                "target": 0,
                "prob_a": 0.70,
                "prob_b": 0.30,
                "a_market_prob": 0.50,
                "b_market_prob": 0.50,
                "no_odds_prob_a": 0.70,
                "no_odds_prob_b": 0.30,
                "a_num_fights": 6,
                "b_num_fights": 6,
            },
        ]
    )
    config = challenger_sweep.SweepConfig(
        name="audit_cfg",
        variant_name="baseline",
        s_min_edge=0.05,
        s_blend_weight=1.0,
        s_model_agreement_min_edge=0.0,
        c_min_model_prob=0.99,
        c_min_no_odds_prob=0.99,
        c_min_edge=0.99,
    )

    monkeypatch.setattr(challenger_sweep, "_passes_filters", lambda *args, **kwargs: True)
    monkeypatch.setattr(challenger_sweep, "dynamic_blend_weight", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(
        challenger_sweep.BankrollManager,
        "kelly_bet_size",
        lambda self, model_prob, decimal_odds: 10.0,
    )

    result = challenger_sweep._evaluate_config(
        [(1, predictions)],
        config,
        initial_bankroll=100.0,
        bet_start_date="2022-01-01",
    )
    bankroll_history = result["bankroll_history"]["combined"].tolist()
    summary_row = challenger_sweep._summary_row_from_result(config, result)
    bet_log = result["bet_log"]

    assert len(bankroll_history) == len(result["bet_log"]) + 1
    assert bankroll_history == pytest.approx([100.0, 115.0, 105.0, 95.0])
    assert result["combined"]["max_drawdown"] == pytest.approx((115.0 - 95.0) / 115.0)
    assert result["combined"]["max_drawdown_duration"] == 2
    assert summary_row["max_drawdown_duration"] == 2
    assert summary_row["s_edge_profit_gap"] == pytest.approx(
        max(0.0, summary_row["s_roi"] - summary_row["s_avg_edge"])
    )
    assert summary_row["s_edge_profit_flag"] is (
        summary_row["s_edge_profit_gap"] > 0.05
    )
    assert set(["edge_basis", "blend_edge", "raw_edge", "market_prob", "blended_prob"]).issubset(
        bet_log.columns
    )
    assert bet_log["edge_basis"].tolist() == ["blend", "blend", "blend"]
    assert bet_log["blend_edge"].notna().all()
    assert bet_log["raw_edge"].notna().all()


def test_sort_summary_rows_prefers_non_inflated_s_edge_configs():
    rows = [
        {
            "config": "inflated",
            "roi": 0.20,
            "avg_clv": 0.04,
            "max_dd": 0.15,
            "total_bets": 200,
            "s_edge_profit_gap": 0.08,
            "s_edge_profit_flag": True,
        },
        {
            "config": "credible",
            "roi": 0.16,
            "avg_clv": 0.03,
            "max_dd": 0.18,
            "total_bets": 180,
            "s_edge_profit_gap": 0.03,
            "s_edge_profit_flag": False,
        },
    ]

    challenger_sweep._sort_summary_rows(rows)

    assert rows[0]["config"] == "credible"
