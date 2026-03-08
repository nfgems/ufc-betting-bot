"""
Value detection — identifies bets where model probability exceeds market probability.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import MIN_EDGE_THRESHOLD

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


def find_value_bets(
    predictions: pd.DataFrame,
    min_edge: float = MIN_EDGE_THRESHOLD,
) -> pd.DataFrame:
    """
    Identify value bets where model edge exceeds threshold.

    Expects predictions DataFrame with columns:
        - fighter_a, fighter_b
        - prob_a, prob_b (model probabilities)
        - a_market_prob, b_market_prob (market-implied fair probabilities)

    Returns DataFrame of value bets with edge calculations.
    """
    bets = []

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
            bets.append({
                "fighter_a": row.get("fighter_a", ""),
                "fighter_b": row.get("fighter_b", ""),
                "bet_on": row.get("fighter_a", ""),
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
            bets.append({
                "fighter_a": row.get("fighter_a", ""),
                "fighter_b": row.get("fighter_b", ""),
                "bet_on": row.get("fighter_b", ""),
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
