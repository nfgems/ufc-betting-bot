"""
Value detection — identifies bets where blended model-market probability
exceeds the market line, with confirmation from both models.

Strategy: Anchor on market odds (they're well-calibrated) and only adjust
where the model has high conviction AND the no-odds model independently agrees.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    MIN_EDGE_THRESHOLD,
    MIN_MODEL_PROB,
    MIN_FIGHTER_FIGHTS,
    MAX_DECIMAL_ODDS,
    EDGE_SCALING_BASE,
    EDGE_SCALING_RATE,
    BLEND_WEIGHT,
    BLEND_WEIGHT_MIN,
    BLEND_WEIGHT_MAX,
    BLEND_CONFIDENCE_THRESHOLD,
    BLEND_AGREEMENT_BOOST,
    REQUIRE_MODEL_AGREEMENT,
    MODEL_AGREEMENT_MIN_EDGE,
    LINE_MOVEMENT_FILTER,
    LINE_AGAINST_EXTRA_EDGE,
    LINE_SHARP_BLOCK,
    CONVICTION_MIN_MODEL_PROB,
    CONVICTION_MIN_NO_ODDS_PROB,
    CONVICTION_BET_FRACTION,
    CONVICTION_CONFIDENCE_BONUS,
    CONVICTION_MAX_BET_FRACTION,
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


def dynamic_blend_weight(
    model_prob: float,
    market_prob: float,
    no_odds_prob: Optional[float] = None,
    base_weight: float = BLEND_WEIGHT,
) -> float:
    """
    Compute a dynamic blend weight based on model confidence and agreement.

    - High model confidence (far from 50%) increases the weight toward BLEND_WEIGHT_MAX
    - Strong no-odds model agreement adds a bonus
    - Low confidence reduces the weight toward BLEND_WEIGHT_MIN
    """
    confidence = abs(model_prob - 0.5) * 2.0  # 0.0 at 50%, 1.0 at 0% or 100%

    # Scale weight linearly based on confidence
    if confidence > (BLEND_CONFIDENCE_THRESHOLD - 0.5) * 2.0:
        # Above threshold: scale from base toward max
        t = min(1.0, (confidence - (BLEND_CONFIDENCE_THRESHOLD - 0.5) * 2.0) /
                (1.0 - (BLEND_CONFIDENCE_THRESHOLD - 0.5) * 2.0))
        weight = base_weight + t * (BLEND_WEIGHT_MAX - base_weight)
    else:
        # Below threshold: scale from min toward base
        t = confidence / max(0.01, (BLEND_CONFIDENCE_THRESHOLD - 0.5) * 2.0)
        weight = BLEND_WEIGHT_MIN + t * (base_weight - BLEND_WEIGHT_MIN)

    # Boost if no-odds model strongly agrees (>5% edge same direction)
    if no_odds_prob is not None:
        model_direction = model_prob - market_prob
        no_odds_direction = no_odds_prob - market_prob
        if (model_direction > 0 and no_odds_direction > 0.05) or \
           (model_direction < 0 and no_odds_direction < -0.05):
            weight = min(BLEND_WEIGHT_MAX, weight + BLEND_AGREEMENT_BOOST)

    return np.clip(weight, BLEND_WEIGHT_MIN, BLEND_WEIGHT_MAX)


def blend_probability(model_prob: float, market_prob: float, weight: float = BLEND_WEIGHT) -> float:
    """
    Blend model probability with market probability.

    The market is well-calibrated, so we anchor on it and only adjust
    where the model disagrees. Default: 30% model, 70% market.
    """
    return weight * model_prob + (1.0 - weight) * market_prob


def scaled_min_edge(decimal_odds: float) -> float:
    """
    Calculate the minimum edge required based on odds magnitude.

    At even money (2.0), requires the base edge (3%).
    For each 1.0 increase in odds, requires an additional 2% edge.
    """
    if decimal_odds <= 2.0:
        return EDGE_SCALING_BASE
    return EDGE_SCALING_BASE + EDGE_SCALING_RATE * (decimal_odds - 2.0)


def _passes_filters(
    blended_prob: float,
    market_prob: float,
    edge: float,
    fighter_name: str,
    no_odds_prob: Optional[float] = None,
    line_movement: Optional[float] = None,
    line_is_sharp: Optional[int] = None,
    line_steam_move: Optional[int] = None,
    bet_side: Optional[str] = None,
    a_num_fights: Optional[int] = None,
    b_num_fights: Optional[int] = None,
) -> bool:
    """Check if a potential bet passes all filters."""
    decimal_odds = implied_prob_to_decimal_odds(market_prob)

    # Filter 0: Fighter experience — skip if either fighter has too few UFC fights
    if a_num_fights is not None and a_num_fights < MIN_FIGHTER_FIGHTS:
        logger.debug(
            f"Skipping {fighter_name}: fighter A has only {a_num_fights} UFC fights "
            f"(minimum: {MIN_FIGHTER_FIGHTS})"
        )
        return False
    if b_num_fights is not None and b_num_fights < MIN_FIGHTER_FIGHTS:
        logger.debug(
            f"Skipping {fighter_name}: fighter B has only {b_num_fights} UFC fights "
            f"(minimum: {MIN_FIGHTER_FIGHTS})"
        )
        return False

    # Filter 1: Minimum blended probability
    if blended_prob < MIN_MODEL_PROB:
        logger.debug(
            f"Skipping {fighter_name}: blended prob {blended_prob:.1%} below "
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

    # Filter 3: Scaled edge threshold
    required_edge = scaled_min_edge(decimal_odds)
    if edge < required_edge:
        logger.debug(
            f"Skipping {fighter_name}: edge {edge:.1%} below scaled "
            f"threshold {required_edge:.1%} at odds {decimal_odds:.2f}"
        )
        return False

    # Filter 4: Model agreement — no-odds model must independently agree
    if REQUIRE_MODEL_AGREEMENT and no_odds_prob is not None:
        no_odds_edge = no_odds_prob - market_prob
        if no_odds_edge < MODEL_AGREEMENT_MIN_EDGE:
            logger.debug(
                f"Skipping {fighter_name}: no-odds model disagrees "
                f"(no-odds edge {no_odds_edge:.1%} < {MODEL_AGREEMENT_MIN_EDGE:.1%})"
            )
            return False

    # Filter 5: Line movement filter — sharp money disagrees
    if LINE_MOVEMENT_FILTER and line_movement is not None and bet_side is not None:
        # line_movement > 0 means market moved toward fighter A
        # If we're betting A and line moved toward B (negative), sharp money disagrees
        line_against = (bet_side == "a" and line_movement < -0.02) or \
                       (bet_side == "b" and line_movement > 0.02)

        if line_against:
            # Require extra edge when line moves against us
            extra_required = required_edge + LINE_AGAINST_EXTRA_EDGE
            if edge < extra_required:
                logger.debug(
                    f"Skipping {fighter_name}: line moved against bet "
                    f"(movement={line_movement:+.1%}), edge {edge:.1%} < "
                    f"required {extra_required:.1%}"
                )
                return False

        # Block if sharp/steam move is clearly against us
        if LINE_SHARP_BLOCK and line_against:
            is_sharp = line_is_sharp == 1 if line_is_sharp is not None else False
            is_steam = line_steam_move == 1 if line_steam_move is not None else False
            if is_sharp or is_steam:
                logger.debug(
                    f"Skipping {fighter_name}: sharp/steam move against bet "
                    f"(sharp={is_sharp}, steam={is_steam})"
                )
                return False

    return True


# Keep old name as alias for backtest.py compatibility
_passes_underdog_filters = lambda model_prob, market_prob, edge, name: _passes_filters(
    model_prob, market_prob, edge, name
)


def find_value_bets(
    predictions: pd.DataFrame,
    min_edge: float = MIN_EDGE_THRESHOLD,
    blend_weight: float = BLEND_WEIGHT,
) -> pd.DataFrame:
    """
    Identify value bets using blended model-market probabilities.

    Strategy:
      1. Blend model prob with market prob (default: 30% model, 70% market)
      2. Compare blended prob to market prob to find edge
      3. Require no-odds model agreement for confirmation
      4. Apply underdog safeguards (min prob, max odds, scaled edge)

    Expects predictions DataFrame with columns:
        - fighter_a, fighter_b
        - prob_a, prob_b (model probabilities)
        - a_market_prob, b_market_prob (market-implied fair probabilities)
        - no_odds_prob_a, no_odds_prob_b (optional: no-odds model probs)

    Returns DataFrame of value bets with edge calculations.
    """
    bets = []
    skipped = 0

    for _, row in predictions.iterrows():
        model_a = row.get("prob_a", 0.5)
        model_b = row.get("prob_b", 0.5)
        market_a = row.get("a_market_prob") or row.get("a_fair_prob_avg", 0.5)
        market_b = row.get("b_market_prob") or row.get("b_fair_prob_avg", 0.5)
        no_odds_a = row.get("no_odds_prob_a")
        no_odds_b = row.get("no_odds_prob_b")

        # Line movement metadata (for sharp money filtering)
        line_movement = row.get("line_movement")
        line_is_sharp = row.get("line_is_sharp")
        line_steam_move = row.get("line_steam_move")
        if isinstance(line_movement, float) and np.isnan(line_movement):
            line_movement = None

        # Fighter experience
        a_fights = row.get("a_num_fights")
        b_fights = row.get("b_num_fights")
        if isinstance(a_fights, float) and not np.isnan(a_fights):
            a_fights = int(a_fights)
        elif not isinstance(a_fights, int):
            a_fights = None
        if isinstance(b_fights, float) and not np.isnan(b_fights):
            b_fights = int(b_fights)
        elif not isinstance(b_fights, int):
            b_fights = None

        # Blend model with market
        blend_a = blend_probability(model_a, market_a, blend_weight)
        blend_b = 1.0 - blend_a

        # Edge = blended - market
        edge_a = blend_a - market_a
        edge_b = blend_b - market_b

        # Pick the side with the larger edge (if any exceeds threshold)
        if edge_a >= min_edge and edge_a >= edge_b:
            fighter_name = row.get("fighter_a", "A")
            if not _passes_filters(
                blend_a, market_a, edge_a, fighter_name, no_odds_a,
                line_movement=line_movement, line_is_sharp=line_is_sharp,
                line_steam_move=line_steam_move, bet_side="a",
                a_num_fights=a_fights, b_num_fights=b_fights,
            ):
                skipped += 1
                continue
            bet = {
                "fighter_a": row.get("fighter_a", ""),
                "fighter_b": row.get("fighter_b", ""),
                "bet_on": fighter_name,
                "bet_side": "a",
                "model_prob": model_a,
                "blended_prob": blend_a,
                "market_prob": market_a,
                "edge": edge_a,
                "decimal_odds": implied_prob_to_decimal_odds(market_a),
                "event_date": row.get("event_date"),
                "weight_class": row.get("weight_class", ""),
                "confidence": max(model_a, model_b),
            }
            # Pass through Polymarket fields if present
            for col in ("token_id_yes", "token_id_no", "market_id",
                        "tick_size", "neg_risk", "volume"):
                if row.get(col) is not None:
                    bet[col] = row[col]
            bets.append(bet)
        elif edge_b >= min_edge:
            fighter_name = row.get("fighter_b", "B")
            if not _passes_filters(
                blend_b, market_b, edge_b, fighter_name, no_odds_b,
                line_movement=line_movement, line_is_sharp=line_is_sharp,
                line_steam_move=line_steam_move, bet_side="b",
                a_num_fights=a_fights, b_num_fights=b_fights,
            ):
                skipped += 1
                continue
            bet = {
                "fighter_a": row.get("fighter_a", ""),
                "fighter_b": row.get("fighter_b", ""),
                "bet_on": fighter_name,
                "bet_side": "b",
                "model_prob": model_b,
                "blended_prob": blend_b,
                "market_prob": market_b,
                "edge": edge_b,
                "decimal_odds": implied_prob_to_decimal_odds(market_b),
                "event_date": row.get("event_date"),
                "weight_class": row.get("weight_class", ""),
                "confidence": max(model_a, model_b),
            }
            for col in ("token_id_yes", "token_id_no", "market_id",
                        "tick_size", "neg_risk", "volume"):
                if row.get(col) is not None:
                    bet[col] = row[col]
            bets.append(bet)

    result = pd.DataFrame(bets)
    if not result.empty:
        result = result.sort_values("edge", ascending=False).reset_index(drop=True)
        logger.info(
            f"Found {len(result)} value bets (min edge: {min_edge:.1%}, "
            f"blend: {blend_weight:.0%} model). "
            f"Avg edge: {result['edge'].mean():.1%}"
        )
    else:
        logger.info(f"No value bets found with edge >= {min_edge:.1%}")

    if skipped:
        logger.info(f"Filtered out {skipped} bets by safeguards")

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


def conviction_bet_size(
    model_prob: float,
    bankroll: float,
) -> float:
    """
    Calculate bet size for a conviction bet.

    Uses a flat percentage of bankroll with a small bonus for extra confidence.
    Since conviction bets target short-odds favorites, sizing is conservative
    to keep risk per bet reasonable despite lower payouts.
    """
    base = CONVICTION_BET_FRACTION * bankroll

    # Bonus: +1% bankroll for every 5% model prob above the 75% threshold
    excess_confidence = max(0.0, model_prob - CONVICTION_MIN_MODEL_PROB)
    bonus_steps = excess_confidence / 0.05
    bonus = bonus_steps * CONVICTION_CONFIDENCE_BONUS * bankroll

    bet = base + bonus
    cap = CONVICTION_MAX_BET_FRACTION * bankroll
    bet = min(bet, cap)

    if bet < 1.0:
        return 0.0
    return round(bet, 2)


def find_conviction_bets(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify conviction bets — fighters that both models agree will win.

    Unlike value bets, conviction bets ignore the market entirely.
    They require dual model agreement:
      1. XGBoost model prob >= 75%
      2. No-odds model prob >= 60%

    The idea: when both models independently agree a fighter wins with high
    confidence, the win rate is very high. Even at short odds the cumulative
    profit from a high strike rate is positive. Market odds are only used
    for execution pricing, not for bet selection.
    """
    bets = []
    skipped = 0

    for _, row in predictions.iterrows():
        model_a = row.get("prob_a", 0.5)
        model_b = row.get("prob_b", 0.5)
        market_a = row.get("a_market_prob") or row.get("a_fair_prob_avg", 0.5)
        market_b = row.get("b_market_prob") or row.get("b_fair_prob_avg", 0.5)
        no_odds_a = row.get("no_odds_prob_a")
        no_odds_b = row.get("no_odds_prob_b")

        # Fighter experience — stricter bar for conviction bets
        a_fights = row.get("a_num_fights")
        b_fights = row.get("b_num_fights")
        if isinstance(a_fights, float) and not np.isnan(a_fights):
            a_fights = int(a_fights)
        elif not isinstance(a_fights, int):
            a_fights = None
        if isinstance(b_fights, float) and not np.isnan(b_fights):
            b_fights = int(b_fights)
        elif not isinstance(b_fights, int):
            b_fights = None

        # Check both sides for conviction
        for side, model_p, market_p, no_odds_p, fighter_name, opp_name, own_fights, opp_fights in [
            ("a", model_a, market_a, no_odds_a,
             row.get("fighter_a", "A"), row.get("fighter_b", "B"), a_fights, b_fights),
            ("b", model_b, market_b, no_odds_b,
             row.get("fighter_b", "B"), row.get("fighter_a", "A"), b_fights, a_fights),
        ]:
            # Gate 1: Model conviction
            if model_p < CONVICTION_MIN_MODEL_PROB:
                continue

            # Gate 2: No-odds model must independently agree
            if no_odds_p is None or no_odds_p < CONVICTION_MIN_NO_ODDS_PROB:
                no_odds_str = f"{no_odds_p:.1%}" if no_odds_p is not None else "N/A"
                logger.debug(
                    f"Conviction skip {fighter_name}: no-odds prob {no_odds_str} "
                    f"< {CONVICTION_MIN_NO_ODDS_PROB:.0%}"
                )
                skipped += 1
                continue

            # Gate 3: Fighter experience — both fighters need minimum UFC fights
            if own_fights is not None and own_fights < MIN_FIGHTER_FIGHTS:
                skipped += 1
                continue
            if opp_fights is not None and opp_fights < MIN_FIGHTER_FIGHTS:
                skipped += 1
                continue

            # All gates passed — this is a conviction bet
            decimal_odds = implied_prob_to_decimal_odds(market_p)
            bet = {
                "fighter_a": row.get("fighter_a", ""),
                "fighter_b": row.get("fighter_b", ""),
                "bet_on": fighter_name,
                "bet_side": side,
                "model_prob": model_p,
                "blended_prob": model_p,  # No blending — pure model conviction
                "market_prob": market_p,
                "no_odds_prob": no_odds_p,
                "edge": model_p - market_p,  # Informational only
                "decimal_odds": decimal_odds,
                "event_date": row.get("event_date"),
                "weight_class": row.get("weight_class", ""),
                "confidence": model_p,
                "conviction_score": (model_p + no_odds_p) / 2.0,
            }
            # Pass through Polymarket fields
            for col in ("token_id_yes", "token_id_no", "market_id",
                        "tick_size", "neg_risk", "volume"):
                if row.get(col) is not None:
                    bet[col] = row[col]
            bets.append(bet)

    result = pd.DataFrame(bets)
    if not result.empty:
        result = result.sort_values("conviction_score", ascending=False).reset_index(drop=True)
        logger.info(
            f"Found {len(result)} conviction bets. "
            f"Avg conviction score: {result['conviction_score'].mean():.1%}, "
            f"Avg model prob: {result['model_prob'].mean():.1%}"
        )
    else:
        logger.info("No conviction bets found (triple agreement not met for any fight)")

    if skipped:
        logger.info(f"Filtered out {skipped} potential conviction bets")

    return result


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
