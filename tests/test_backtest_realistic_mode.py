import pandas as pd

import src.strategy.backtest as backtest


def test_realistic_mode_delays_same_event_settlement_and_locks_cash(monkeypatch):
    monkeypatch.setattr(backtest, "_passes_filters", lambda *args, **kwargs: True)

    predictions = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2024-01-01"),
                "event_name": "UFC Test Card",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "target": 1,
                "a_market_prob": 0.50,
                "b_market_prob": 0.50,
                "a_num_fights": 5,
                "b_num_fights": 5,
                "odds_source": "test",
                "xgboost_no_odds_prob_a": 0.75,
                "xgboost_no_odds_prob_b": 0.25,
            },
            {
                "event_date": pd.Timestamp("2024-01-01"),
                "event_name": "UFC Test Card",
                "fighter_a": "Gamma",
                "fighter_b": "Delta",
                "target": 1,
                "a_market_prob": 0.50,
                "b_market_prob": 0.50,
                "a_num_fights": 5,
                "b_num_fights": 5,
                "odds_source": "test",
                "xgboost_no_odds_prob_a": 0.75,
                "xgboost_no_odds_prob_b": 0.25,
            },
        ]
    )

    legacy = backtest._simulate_backtest_predictions(
        predictions,
        backtest.NO_ODDS_SIGNAL_ONLY_STRATEGY,
        initial_bankroll=100.0,
        min_edge=0.01,
        kelly_fraction=1.0,
        max_bet_fraction=1.0,
        execution_mode="legacy",
    )
    realistic = backtest._simulate_backtest_predictions(
        predictions,
        backtest.NO_ODDS_SIGNAL_ONLY_STRATEGY,
        initial_bankroll=100.0,
        min_edge=0.01,
        kelly_fraction=1.0,
        max_bet_fraction=1.0,
        execution_mode="realistic",
        execution_config=backtest.BacktestExecutionConfig(
            mode="realistic",
            assumed_half_spread=0.0,
            synthetic_liquidity_floor=1000.0,
            synthetic_liquidity_peak=1000.0,
        ),
    )

    legacy_sizes = legacy["bet_log"]["bet_size"].tolist()
    realistic_sizes = realistic["bet_log"]["bet_size"].tolist()

    assert legacy_sizes == [50.0, 75.0]
    assert realistic_sizes == [50.0, 50.0]
    assert realistic["bet_log"]["available_cash_before_bet"].tolist() == [100.0, 50.0]
    assert realistic["stats"]["execution_mode"] == "realistic"


def test_realistic_mode_tracks_liquidity_skips_partial_fills_and_slippage(monkeypatch):
    monkeypatch.setattr(backtest, "_passes_filters", lambda *args, **kwargs: True)

    predictions = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2024-01-01"),
                "event_name": "Liquidity Card",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "target": 1,
                "a_market_prob": 0.50,
                "b_market_prob": 0.50,
                "a_num_fights": 5,
                "b_num_fights": 5,
                "assumed_book_liquidity": 80.0,
                "odds_source": "test",
                "xgboost_no_odds_prob_a": 0.75,
                "xgboost_no_odds_prob_b": 0.25,
            },
            {
                "event_date": pd.Timestamp("2024-01-01"),
                "event_name": "Liquidity Card",
                "fighter_a": "Gamma",
                "fighter_b": "Delta",
                "target": 1,
                "a_market_prob": 0.50,
                "b_market_prob": 0.50,
                "a_num_fights": 5,
                "b_num_fights": 5,
                "assumed_book_liquidity": 40.0,
                "odds_source": "test",
                "xgboost_no_odds_prob_a": 0.75,
                "xgboost_no_odds_prob_b": 0.25,
            },
        ]
    )

    result = backtest._simulate_backtest_predictions(
        predictions,
        backtest.NO_ODDS_SIGNAL_ONLY_STRATEGY,
        initial_bankroll=100.0,
        min_edge=0.01,
        kelly_fraction=1.0,
        max_bet_fraction=1.0,
        execution_mode="realistic",
        execution_config=backtest.BacktestExecutionConfig(
            mode="realistic",
            min_book_liquidity=50.0,
            max_slippage=0.05,
            max_bet_vs_book_ratio=0.25,
            assumed_half_spread=0.0,
            synthetic_liquidity_floor=80.0,
            synthetic_liquidity_peak=80.0,
            synthetic_price_step=0.02,
            synthetic_depth_notional_shares=(0.10, 0.15, 0.75),
        ),
    )

    assert result["stats"]["total_bets"] == 1
    assert result["stats"]["partial_fills"] == 1
    assert result["stats"]["skipped_for_liquidity_count"] == 1
    assert result["stats"]["requested_stake_total"] == 100.0
    assert result["stats"]["filled_stake_total"] == 20.0
    assert result["stats"]["fill_rate"] == 0.2
    assert result["bet_log"].loc[0, "requested_stake"] == 50.0
    assert result["bet_log"].loc[0, "bet_size"] == 20.0
    assert bool(result["bet_log"].loc[0, "partial_fill"]) is True
    assert result["bet_log"].loc[0, "fill_price"] > result["bet_log"].loc[0, "quoted_price"]
    assert result["bet_log"].loc[0, "slippage"] > 0
