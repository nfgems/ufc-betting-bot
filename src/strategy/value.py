"""
Value detection — identifies bets where model probability exceeds market probability.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    MIN_EDGE_THRESHOLD,
    MIN_MODEL_PROB,
    MAX_DECIMAL_ODDS,
    EDGE_SCALING_BASE,
    EDGE_SCALING_RATE,
)

logger = logging.getLogger(__name__)


def decimal_odds_to_implied_prob(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if odds <= 0:
        return 0.0
    return 1.0 / odds


def implied_prob_to_decimal_odds(prob: float) -> float:
    """Convert probability to decimal odds."""
    if prob <= 0:
        return float("inf")
    return 1.0 / prob


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """
    Remove bookmaker vig to get fair probabilities.
    The sum of implied probabilities typically exceeds 1.0 (the overround).
    """
    total = prob_a + prob_b
    if total == 0:
        return 0.5, 0.5
    return prob_a / total, prob_b / total


def scaled_min_edge(decimal_odds: float) -> float:
    """
    Calculate the minimum edge required based on odds magnitude.

    At even money (2.0), requires the base edge (4%).
    For each 1.0 increase in odds, requires an additional 2% edge.
    E.g., at 4.0 odds: 4% + 2% * (4.0 - 2.0) = 8% min edge.
    """
    if decimal_odds <= 2.0:
        return EDGE_SCALING_BASE
    return EDGE_SCALING_BASE + EDGE_SCALING_RATE * (decimal_odds - 2.0)


def _passes_underdog_filters(
    model_prob: float,
    market_prob: float,
    edge: float,
    fighter_name: str,
) -> bool:
    """Check if a potential bet passes underdog safeguards."""
    decimal_odds = implied_prob_to_decimal_odds(market_prob)

    # Filter 1: Minimum model probability
    if model_prob < MIN_MODEL_PROB:
        logger.debug(
            f"Skipping {fighter_name}: model prob {model_prob:.1%} below "
            f"minimum {MIN_MODEL_PROB:.1%}"
        )
        return False

    # Filter 2: Maximum odds cap
    if decimal_odds > MAX_DECIMAL_ODDS:
        logger.debug(
            f"Skipping {fighter_name}: odds {decimal_odds:.2f} exceed "
            f"maximum {MAX_DECIMAL_ODDS:.1f}"
        )
        return False

    # Filter 3: Scaled edge threshold (higher edge required at longer odds)
    required_edge = scaled_min_edge(decimal_odds)
    if edge < required_edge:
        logger.debug(
            f"Skipping {fighter_name}: edge {edge:.1%} below scaled "
            f"threshold {required_edge:.1%} at odds {decimal_odds:.2f}"
        )
        return False

    return True


def find_value_bets(
    predictions: pd.DataFrame,
    min_edge: float = MIN_EDGE_THRESHOLD,
) -> pd.DataFrame:
    """
    Identify value bets where model edge exceeds threshold.

    Applies underdog safeguards:
        - Minimum model probability (default 15%)
        - Maximum decimal odds cap (default 5.0)
        - Scaled edge threshold (higher edge required at longer odds)

    Expects predictions DataFrame with columns:
        - fighter_a, fighter_b
        - prob_a, prob_b (model probabilities)
        - a_market_prob, b_market_prob (market-implied fair probabilities)

    Returns DataFrame of value bets with edge calculations.
    """
    bets = []
    skipped = 0

    for _, row in predictions.iterrows():
        model_a = row.get("prob_a", 0.5)
        model_b = row.get("prob_b", 0.5)
        market_a = row.get("a_market_prob") or row.get("a_fair_prob_avg", 0.5)
        market_b = row.get("b_market_prob") or row.get("b_fair_prob_avg", 0.5)

        # Edge on fighter A
        edge_a = model_a - market_a
        # Edge on fighter B
        edge_b = model_b - market_b

        # Pick the side with the larger edge (if any exceeds threshold)
        if edge_a >= min_edge and edge_a >= edge_b:
            fighter_name = row.get("fighter_a", "A")
            if not _passes_underdog_filters(model_a, market_a, edge_a, fighter_name):
                skipped += 1
                continue
            bets.append({
                "fighter_a": row.get("fighter_a", ""),
                "fighter_b": row.get("fighter_b", ""),
                "bet_on": fighter_name,
                "bet_side": "a",
                "model_prob": model_a,
                "market_prob": market_a,
                "edge": edge_a,
                "decimal_odds": implied_prob_to_decimal_odds(market_a),
                "event_date": row.get("event_date"),
                "weight_class": row.get("weight_class", ""),
                "confidence": row.get("confidence", max(model_a, model_b)),
            })
        elif edge_b >= min_edge:
            fighter_name = row.get("fighter_b", "B")
            if not _passes_underdog_filters(model_b, market_b, edge_b, fighter_name):
                skipped += 1
                continue
            bets.append({
                "fighter_a": row.get("fighter_a", ""),
                "fighter_b": row.get("fighter_b", ""),
                "bet_on": fighter_name,
                "bet_side": "b",
                "model_prob": model_b,
                "market_prob": market_b,
                "edge": edge_b,
                "decimal_odds": implied_prob_to_decimal_odds(market_b),
                "event_date": row.get("event_date"),
                "weight_class": row.get("weight_class", ""),
                "confidence": row.get("confidence", max(model_a, model_b)),
            })

    result = pd.DataFrame(bets)
    if not result.empty:
        result = result.sort_values("edge", ascending=False).reset_index(drop=True)
        logger.info(
            f"Found {len(result)} value bets (min edge: {min_edge:.1%}). "
            f"Avg edge: {result['edge'].mean():.1%}"
        )
    else:
        logger.info(f"No value bets found with edge >= {min_edge:.1%}")

    if skipped:
        logger.info(f"Filtered out {skipped} bets by underdog safeguards")

    return result


def calculate_expected_value(
    model_prob: float,
    decimal_odds: float,
    stake: float = 1.0,
) -> float:
    """
    Calculate expected value of a bet.

    EV = (prob_win * profit) - (prob_loss * stake)
    """
    profit = stake * (decimal_odds - 1)
    ev = (model_prob * profit) - ((1 - model_prob) * stake)
    return ev


def calculate_closing_line_value(
    bet_prob: float,
    closing_prob: float,
) -> float:
    """
    Calculate Closing Line Value (CLV).

    CLV > 0 means you got a better price than the closing line.
    Consistently positive CLV is the strongest indicator of genuine edge.
    """
    if closing_prob <= 0:
        return 0.0
    return (1.0 / bet_prob) / (1.0 / closing_prob) - 1.0
