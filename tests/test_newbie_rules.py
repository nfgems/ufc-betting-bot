from dataclasses import replace

import pandas as pd
import pytest

import src.strategy.backtest as backtest
from src.strategy.value import (
    LEGACY_BASELINE_NEWBIE_RULE,
    THRESHOLD_DROP_NEWBIE_RULE,
    TIERED_ORG_AWARE_NEWBIE_RULE,
    find_value_bets,
    newbie_penalty,
)


def test_threshold_drop_tags_two_fight_bets_as_newbie():
    adjustment = newbie_penalty(
        2,
        6,
        newbie_rule=THRESHOLD_DROP_NEWBIE_RULE,
    )

    assert adjustment.allowed is True
    assert adjustment.is_newbie_bet is True
    assert adjustment.extra_edge_required == pytest.approx(0.0)
    assert adjustment.size_multiplier == pytest.approx(1.0)


def test_tiered_major_background_adds_extra_edge_and_half_size():
    adjustment = newbie_penalty(
        1,
        6,
        org_tier_a=1.0,
        newbie_rule=TIERED_ORG_AWARE_NEWBIE_RULE,
    )

    assert adjustment.allowed is True
    assert adjustment.is_newbie_bet is True
    assert adjustment.extra_edge_required == pytest.approx(0.02)
    assert adjustment.size_multiplier == pytest.approx(0.50)


def test_tiered_regional_debut_is_skipped():
    adjustment = newbie_penalty(
        0,
        6,
        org_tier_a=3.0,
        newbie_rule=TIERED_ORG_AWARE_NEWBIE_RULE,
    )

    assert adjustment.allowed is False


def test_find_value_bets_tags_threshold_drop_newbies():
    predictions = pd.DataFrame(
        [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "prob_a": 0.70,
                "prob_b": 0.30,
                "a_market_prob": 0.60,
                "b_market_prob": 0.40,
                "no_odds_prob_a": 0.70,
                "no_odds_prob_b": 0.30,
                "a_num_fights": 2,
                "b_num_fights": 7,
            }
        ]
    )

    result = find_value_bets(
        predictions,
        min_edge=0.01,
        newbie_rule=THRESHOLD_DROP_NEWBIE_RULE,
    )

    assert len(result) == 1
    assert bool(result.iloc[0]["is_newbie_bet"]) is True
    assert result.iloc[0]["size_multiplier"] == pytest.approx(1.0)


def test_backtest_threshold_drop_places_two_fight_bet_and_tags_it():
    predictions = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2024-01-01"),
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "target": 1,
                "a_market_prob": 0.50,
                "b_market_prob": 0.50,
                "a_num_fights": 2,
                "b_num_fights": 6,
                "odds_source": "test",
                "xgboost_prob_a": 0.60,
                "xgboost_prob_b": 0.40,
            }
        ]
    )

    legacy_baseline_strategy = replace(
        backtest.ODDS_AWARE_SIGNAL_ONLY_STRATEGY,
        name="legacy_baseline_min_3",
        newbie_rule=LEGACY_BASELINE_NEWBIE_RULE,
    )
    baseline = backtest._simulate_backtest_predictions(
        predictions,
        legacy_baseline_strategy,
        initial_bankroll=100.0,
        min_edge=0.01,
    )
    assert baseline["bet_log"].empty

    threshold_strategy = replace(
        backtest.ODDS_AWARE_SIGNAL_ONLY_STRATEGY,
        name="threshold_min_2",
        newbie_rule=THRESHOLD_DROP_NEWBIE_RULE,
    )
    threshold = backtest._simulate_backtest_predictions(
        predictions,
        threshold_strategy,
        initial_bankroll=100.0,
        min_edge=0.01,
    )

    assert len(threshold["bet_log"]) == 1
    assert bool(threshold["bet_log"].loc[0, "is_newbie_bet"]) is True
    assert threshold["bet_log"].loc[0, "size_multiplier"] == pytest.approx(1.0)


def test_backtest_tiered_major_background_halves_bet_size():
    predictions = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2024-01-01"),
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "target": 1,
                "a_market_prob": 0.50,
                "b_market_prob": 0.50,
                "a_num_fights": 1,
                "b_num_fights": 6,
                "a_pre_ufc_org_tier_best": 1.0,
                "odds_source": "test",
                "xgboost_prob_a": 0.55,
                "xgboost_prob_b": 0.45,
            }
        ]
    )

    tiered_strategy = replace(
        backtest.ODDS_AWARE_SIGNAL_ONLY_STRATEGY,
        name="tiered_org_aware",
        newbie_rule=TIERED_ORG_AWARE_NEWBIE_RULE,
    )
    result = backtest._simulate_backtest_predictions(
        predictions,
        tiered_strategy,
        initial_bankroll=100.0,
        min_edge=0.01,
    )

    assert len(result["bet_log"]) == 1
    assert result["bet_log"].loc[0, "newbie_extra_edge_required"] == pytest.approx(0.02)
    assert result["bet_log"].loc[0, "size_multiplier"] == pytest.approx(0.50)
    assert result["bet_log"].loc[0, "bet_size"] == pytest.approx(1.25)
