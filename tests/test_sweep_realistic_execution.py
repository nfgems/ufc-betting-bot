"""E3: realistic execution semantics in the duo-trader sweep."""

import numpy as np
import pandas as pd

from src.strategy.duo_trader_sweep import SweepConfig, _evaluate_config


def _two_fight_event_frame() -> pd.DataFrame:
    """Two fights on the same card, both clear S-trader bets that win."""
    base = {
        "no_odds_prob_a": 0.78,
        "no_odds_prob_b": 0.22,
        "a_num_fights": 6,
        "b_num_fights": 6,
        "closing_prob_a": 0.70,
        "closing_prob_b": 0.30,
        "target": 1,
    }
    return pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2024-01-01"),
                "fighter_a": "Alpha One",
                "fighter_b": "Beta One",
                "prob_a": 0.80,
                "prob_b": 0.20,
                "a_market_prob": 0.60,
                "b_market_prob": 0.40,
                **base,
            },
            {
                "event_date": pd.Timestamp("2024-01-01"),
                "fighter_a": "Alpha Two",
                "fighter_b": "Beta Two",
                "prob_a": 0.80,
                "prob_b": 0.20,
                "a_market_prob": 0.60,
                "b_market_prob": 0.40,
                **base,
            },
        ]
    )


def _config() -> SweepConfig:
    return SweepConfig(
        name="test",
        s_min_edge=0.02,
        s_blend_weight=0.30,
        c_share=0.5,
    )


def test_realistic_mode_defers_settlement_within_event():
    """Fight-1 winnings must not compound into fight-2 stakes on the same card."""
    frame = _two_fight_event_frame()
    legacy = _evaluate_config(
        [(1, frame.copy())], _config(), initial_bankroll=1000.0, bet_start_date="2022-01-01",
    )
    realistic = _evaluate_config(
        [(1, frame.copy())], _config(), initial_bankroll=1000.0, bet_start_date="2022-01-01",
        execution_mode="realistic",
    )

    legacy_log = legacy["bet_log"]
    realistic_log = realistic["bet_log"]
    assert len(legacy_log) == 2
    assert len(realistic_log) == 2

    # Both execution modes defer same-card settlement, so fight 1 winnings
    # cannot increase fight 2's stake.
    assert legacy_log.iloc[1]["bet_size"] <= legacy_log.iloc[0]["bet_size"]

    # Realistic: both fights are sized from the same event-start bankroll
    # (fight 2 actually sees slightly LESS cash because fight 1's stake is
    # still locked), so the second stake cannot exceed the first.
    assert realistic_log.iloc[1]["bet_size"] <= realistic_log.iloc[0]["bet_size"]


def test_realistic_mode_prices_fills_with_spread():
    frame = _two_fight_event_frame()
    realistic = _evaluate_config(
        [(1, frame.copy())], _config(), initial_bankroll=1000.0, bet_start_date="2022-01-01",
        execution_mode="realistic",
    )
    log = realistic["bet_log"]
    # Fill prices must include at least the assumed half-spread over mid.
    assert (log["fill_price"] > log["market_prob"] + 1e-9).all()
    # Settlement is deferred but completed: profits are recorded.
    assert log["profit"].notna().all()


def test_legacy_mode_unchanged_by_execution_plumbing():
    frame = _two_fight_event_frame()
    default_result = _evaluate_config(
        [(1, frame.copy())], _config(), initial_bankroll=1000.0, bet_start_date="2022-01-01",
    )
    explicit_legacy = _evaluate_config(
        [(1, frame.copy())], _config(), initial_bankroll=1000.0, bet_start_date="2022-01-01",
        execution_mode="legacy",
    )
    assert default_result["combined"]["final_bankroll"] == explicit_legacy["combined"]["final_bankroll"]
    assert default_result["combined"]["total_bets"] == explicit_legacy["combined"]["total_bets"]
    # Legacy fills at the exact no-vig mid.
    assert (default_result["bet_log"]["fill_price"] == default_result["bet_log"]["market_prob"]).all()


def test_realistic_mode_roi_not_better_than_legacy():
    """Spread + locked capital can only hurt a winning sequence's ROI."""
    frame = _two_fight_event_frame()
    legacy = _evaluate_config(
        [(1, frame.copy())], _config(), initial_bankroll=1000.0, bet_start_date="2022-01-01",
    )
    realistic = _evaluate_config(
        [(1, frame.copy())], _config(), initial_bankroll=1000.0, bet_start_date="2022-01-01",
        execution_mode="realistic",
    )
    assert realistic["combined"]["roi"] <= legacy["combined"]["roi"] + 1e-9
